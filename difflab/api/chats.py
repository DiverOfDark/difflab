"""Chats as data, for the process that serves the page. A reply is queued as a run; the worker writes it."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from difflab import config, prompts, store
from difflab.api import harness, runs
from difflab.store import chats
from difflab.work import queue

TITLE_WORDS = 8


def listing() -> list[dict[str, Any]]:
    with store.connect() as connection:
        return [_summary(one) for one in chats.chats(connection)]


def start() -> dict[str, Any]:
    with store.connect() as connection:
        made = chats.add_chat(connection)
    return one(made.id) or {}


def one(chat_id: int) -> dict[str, Any] | None:
    with store.connect() as connection:
        found = chats.chat(connection, chat_id)
        if found is None:
            return None
        said = chats.messages(connection, chat_id)
        # A run that ended without settling its answer (the process died, or it was cancelled before it began)
        # never will: the answer is closed here rather than left waiting.
        for message in said:
            if message.role == chats.TOOL:
                continue  # `harness.advance` answers a call once its run has settled
            if message.state in chats.PENDING and (message.run_id is None or runs.gone(message.run_id)):
                chats.write(connection, message.id, state=chats.FAILED, error="the run ended without an answer")
        said = chats.messages(connection, chat_id)
    return {
        **_summary(found),
        "system": found.system,
        "summary": found.summary,
        "summary_through": found.summary_through,
        "compact_next": found.compact_next,
        "context": {
            "used": next((m.tokens for m in reversed(said) if m.tokens and m.id > found.summary_through), 0),
            "total": config.VLM_CONTEXT_TOKENS,
            "compacts_at": config.CHAT_COMPACT_TOKENS,
        },
        "default_system": prompts.CHAT_SYSTEM_PROMPT,
        "messages": [message.data() for message in said],
    }


def update(chat_id: int, fields: dict[str, Any]) -> dict[str, Any]:
    with store.connect() as connection:
        chats.update_chat(
            connection, chat_id, **{name: str(fields[name]) for name in ("title", "system") if name in fields}
        )
    return _found(chat_id)


def delete(chat_id: int) -> None:
    stop(chat_id)
    with store.connect() as connection:
        chats.delete_chat(connection, chat_id)


def say(chat_id: int, text: str, media: list[str]) -> dict[str, Any]:
    """The person's message, and a reply to it queued."""
    text = text.strip()
    if not text and not media:
        raise ValueError("say something, or attach something")
    files = [_under_output(name) for name in media]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"no such file: {', '.join(missing)}")
    with store.connect() as connection, store.transaction(connection):
        harness.decline(connection, chat_id)
        found = _open(connection, chat_id)
        chats.add_message(connection, chat_id, chats.USER, text, [str(path) for path in files])
        if not found.title:
            title = " ".join(text.split()[:TITLE_WORDS]) or files[0].name
            chats.update_chat(connection, chat_id, title=title)
    return _answer(chat_id)


def again(chat_id: int) -> dict[str, Any]:
    """The last answer, written afresh."""
    with store.connect() as connection, store.transaction(connection):
        _open(connection, chat_id)
        said = chats.messages(connection, chat_id)
        if not said or said[-1].role != chats.ASSISTANT:
            raise ValueError("there is no answer to write again")
        chats.delete_after(connection, chat_id, said[-1].id)
    return _answer(chat_id)


def rewind(chat_id: int, message_id: int) -> dict[str, Any]:
    """The conversation as it was before this message, so it can be said differently."""
    with store.connect() as connection, store.transaction(connection):
        _open(connection, chat_id)
        chats.delete_after(connection, chat_id, message_id)
    return _found(chat_id)


def stop(chat_id: int) -> None:
    """A run a tool call only follows goes on; one it queued is cancelled."""
    with store.connect() as connection:
        pending = [one for one in chats.messages(connection, chat_id) if one.state in (*chats.PENDING, chats.ASKING)]
    for message in pending:
        follows = message.role == chats.TOOL and message.calls and message.calls[0].get("follows")
        if message.run_id and not follows:
            runs.control(message.run_id, {"cmd": "cancel"})
        if message.state in (chats.QUEUED, chats.ASKING) or message.role == chats.TOOL:
            with store.connect() as connection:
                chats.write(connection, message.id, state=chats.STOPPED)


def compact(chat_id: int) -> dict[str, Any]:
    """The next reply summarises the chat first, keeping its newest messages as said."""
    with store.connect() as connection:
        if chats.chat(connection, chat_id) is None:
            raise LookupError("no such chat")
        chats.update_chat(connection, chat_id, compact_next=1)
    return _found(chat_id)


def allow(chat_id: int, message_id: int, allowed: bool) -> dict[str, Any]:
    harness.allow(chat_id, message_id, allowed)
    return _found(chat_id)


def _answer(chat_id: int) -> dict[str, Any]:
    with store.connect() as connection, store.transaction(connection):
        found = _open(connection, chat_id)
        answer = chats.add_message(connection, chat_id, chats.ASSISTANT, state=chats.QUEUED)
        chats.write(connection, answer.id, run_id=queue.enqueue_chat(connection, answer.id, found.title))
    return _found(chat_id)


def _open(connection: sqlite3.Connection, chat_id: int) -> chats.Chat:
    found = chats.chat(connection, chat_id)
    if found is None:
        raise LookupError("no such chat")
    if any(one.state in chats.PENDING for one in chats.messages(connection, chat_id)):
        raise ValueError("a reply is still being written: stop it first")
    return found


def _found(chat_id: int) -> dict[str, Any]:
    found = one(chat_id)
    if found is None:
        raise LookupError("no such chat")
    return found


def _summary(chat: chats.Chat) -> dict[str, Any]:
    return {
        "id": chat.id,
        "title": chat.title,
        "created": chat.created_at,
        "updated": chat.updated_at,
    }


def _under_output(name: str) -> Path:
    given = Path(name)
    return given if given.is_absolute() else config.OUTPUT_DIR / given
