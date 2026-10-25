"""The VLM behind OpenAI's chat completions, for other tools. A request is a chat of its own that lives as
long as the request; its reply waits its turn on the card like the page's."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from difflab import store
from difflab.api import assets, runs
from difflab.api import chats as chatting
from difflab.store import chats
from difflab.work import queue

MODEL = "difflab"
POLL = 0.25  # seconds; the worker writes the reply so far every `tools.chat.WRITE_EVERY`
EFFORTS = {"low", "medium", "high"}


class RequestError(ValueError):
    """A request this endpoint cannot answer as asked."""


def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "difflab"}]}


def ask(body: dict[str, Any]) -> tuple[int, int]:
    """The request as a chat with its reply queued: `(chat, reply)`."""
    if body.get("tools"):
        raise RequestError("tool calling is not supported")
    if int(body.get("n") or 1) != 1:
        raise RequestError("n must be 1")
    system: list[str] = []
    said: list[tuple[str, str, list[str]]] = []
    for message in body.get("messages") or []:
        role = message.get("role")
        text, media = _content(message.get("content"))
        if role in ("system", "developer"):
            system.append(text)
        elif role in (chats.USER, chats.ASSISTANT):
            said.append((role, text, media))
        else:
            raise RequestError(f"messages of role {role!r} are not supported")
    if not said or said[-1][0] != chats.USER:
        raise RequestError("the last message must be the user's")
    with store.connect() as connection, store.transaction(connection):
        chat = chats.add_chat(connection, title=f"api · {said[-1][1][:60]}", system="\n\n".join(system))
        for role, text, media in said:
            chats.add_message(connection, chat.id, role, text, media)
        answer = chats.add_message(connection, chat.id, chats.ASSISTANT, state=chats.QUEUED)
        run_id = queue.enqueue_chat(connection, answer.id, chat.title, _overrides(body), _shaped(body))
        chats.write(connection, answer.id, run_id=run_id)
    return chat.id, answer.id


def _content(content: Any) -> tuple[str, list[str]]:
    if content is None or isinstance(content, str):
        return content or "", []
    words: list[str] = []
    media: list[str] = []
    for part in content:
        kind = part.get("type")
        if kind == "text":
            words.append(str(part.get("text", "")))
        elif kind == "image_url":
            url = part["image_url"]["url"] if isinstance(part["image_url"], dict) else part["image_url"]
            media.append(_stored(str(url)))
        else:
            raise RequestError(f"content of type {kind!r} is not supported")
    return "\n".join(words), media


def _stored(url: str) -> str:
    if not url.startswith("data:"):
        raise RequestError("images must be sent inline, as data: URLs")
    head, _, data = url.partition(",")
    suffix = mimetypes.guess_extension(head.removeprefix("data:").split(";")[0]) or ".png"
    return str(assets.store(base64.b64decode(data), f"api{suffix}")["path"])


def _overrides(body: dict[str, Any]) -> dict[str, Any]:
    chosen: dict[str, Any] = {}
    for theirs, ours in (("temperature", "temperature"), ("top_p", "top_p"), ("seed", "seed")):
        if body.get(theirs) is not None:
            chosen[ours] = body[theirs]
    cap = body.get("max_completion_tokens") or body.get("max_tokens")
    if cap:
        chosen["max_new_tokens"] = int(cap)
    return chosen


def _shaped(body: dict[str, Any]) -> dict[str, Any]:
    shaped: dict[str, Any] = {}
    wanted = body.get("response_format") or {}
    if wanted.get("type") == "json_object":
        shaped["as_json"] = True
    elif wanted.get("type") == "json_schema":
        shaped["as_json"], shaped["schema"] = True, wanted["json_schema"]["schema"]
    effort = body.get("reasoning_effort")
    if effort is not None:
        if effort in ("none", "minimal"):
            shaped["effort"] = None
        elif effort in EFFORTS:
            shaped["effort"] = effort
        else:
            raise RequestError(f"reasoning_effort {effort!r}: none, minimal, low, medium or high")
    return shaped


async def follow(
    chat_id: int, answer_id: int, gone: Callable[[], Awaitable[bool]]
) -> AsyncIterator[tuple[str, str, chats.Message]]:
    """`(thought, words)` as they grow, the reply's row beside them, until it settles. A caller that
    leaves stops the reply; the chat goes either way."""
    thought = words = ""
    settled = False
    try:
        while True:
            found = await asyncio.to_thread(_read, answer_id)
            if found is None:
                raise RuntimeError("the reply was deleted")
            more_thought = found.thinking[len(thought) :] if found.thinking.startswith(thought) else ""
            more_words = found.text[len(words) :] if found.text.startswith(words) else ""
            thought, words = thought + more_thought, words + more_words
            settled = found.state not in chats.PENDING
            if more_thought or more_words or settled:
                yield more_thought, more_words, found
            if settled:
                return
            if await gone():
                return
            await asyncio.sleep(POLL)
    finally:
        _close(chat_id, settled)  # not awaited: a client that left cancels this task, and an await here never returns


def _read(answer_id: int) -> chats.Message | None:
    with store.connect() as connection:
        found = chats.message(connection, answer_id)
    if found is not None and found.state in chats.PENDING and (found.run_id is None or runs.gone(found.run_id)):
        with store.connect() as connection:
            chats.write(connection, answer_id, state=chats.FAILED, error="the run ended without an answer")
            found = chats.message(connection, answer_id)
    return found


def _close(chat_id: int, settled: bool) -> None:
    if not settled:
        chatting.stop(chat_id)
    with store.connect() as connection:
        chats.delete_chat(connection, chat_id)


def failure(found: chats.Message) -> str | None:
    if found.state == chats.FAILED:
        return found.error or "the reply failed"
    return None


def chunk(made: str, delta: dict[str, Any], finish: str | None = None) -> str:
    data = {
        "id": made,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def completion(made: str, thought: str, words: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": words}
    if thought:
        message["reasoning_content"] = thought
    return {
        "id": made,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }
