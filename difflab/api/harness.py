"""A chat as an agent. An answer's tool calls are the MCP server's tools and the web's, run in-process; a call
that queues a run is answered once the run settles, and the next answer is queued when every call of a turn
has been. A call that spends a long render or deletes something waits for the person to allow it."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from difflab import config, prompts, store
from difflab.api import assets, director, mcp, runs, web
from difflab.models import exl3vl
from difflab.store import chats
from difflab.work import queue

ROUNDS = 8  # answers with tool calls after one message from the person, before one has to do without
MADE_MOST = 6  # of a run's files, the newest handed back as its result
FOLLOWED = ("make_picture", "make_clip", "upscale")  # answered with what their run makes
ASKED = frozenset(
    {name for name, route in director.BY_NAME.items() if route.costly or route.method == "DELETE"} | {"cancel_run"}
)
# "The person did not allow it." read as a hiccup: the model called the same reroll again (2026-10-25).
LOOK = {
    "name": "look",
    "description": "A picture or clip the server holds, shown to you again by its path: one named in the chat's "
    "summary, or any a tool returned.",
    "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
}
DECLINED = "The person declined this call. Do not make it again: ask them what they want instead."


def tools() -> list[dict[str, Any]]:
    """In a fixed order, so the disk cache keeps matching the system prompt they are part of. Without `upload`, which
    the model cannot feed, and `confirm`, which the person gives instead."""
    listed = []
    for tool in [*mcp.TOOLS, *director.TOOLS, *web.TOOLS, LOOK]:
        if tool["name"] == "upload":
            continue
        schema = tool["inputSchema"]
        if "confirm" in schema.get("properties", {}):
            schema = {**schema, "properties": {k: v for k, v in schema["properties"].items() if k != "confirm"}}
        listed.append({"name": tool["name"], "description": tool["description"], "parameters": schema})
    return listed


def prepare(message_id: int, spent: bool) -> tuple[list[dict[str, Any]], str]:
    """The tools a reply may call and what the system prompt is told about them; none once `spent`."""
    with store.connect() as connection:
        answer = chats.message(connection, message_id)
    if answer is None:
        raise LookupError(f"no chat message {message_id}: the chat was deleted")
    if spent:
        return [], prompts.CHAT_TOOLS_SPENT
    return tools(), prompts.CHAT_TOOLS_PROMPT


def after_reply(message_id: int, offered: list[dict[str, Any]]) -> None:
    """The answer's tool calls, each a tool message, run; the next answer queued when they have all settled."""
    if not offered:
        return
    with store.connect() as connection:
        answer = chats.message(connection, message_id)
    if answer is None:
        return
    words, made = exl3vl.tool_calls(answer.text, offered)
    if not made:
        return
    calls = [{"id": f"call_{uuid.uuid4().hex[:12]}", **call} for call in made]
    with store.connect() as connection, store.transaction(connection):
        chats.write(connection, message_id, text=words, calls=calls)
        asked = [
            chats.add_message(connection, answer.chat_id, chats.TOOL, state=chats.WRITING, calls=[call])
            for call in calls
        ]
    for message in asked:
        _run(message)
    _next(answer.chat_id)


def advance() -> None:
    """Every call waiting on a run, answered once it has settled; the worker asks between jobs."""
    with store.connect() as connection:
        waiting = chats.following(connection)
    moved = set()
    for message in waiting:
        call = message.calls[0]
        assert message.run_id is not None
        if runs.gone(message.run_id) or time.time() >= call.get("until", float("inf")):
            _answer_run(message)
            moved.add(message.chat_id)
    for chat_id in moved:
        _next(chat_id)


def allow(chat_id: int, message_id: int, allowed: bool) -> None:
    """The person's answer to a call that waits for it."""
    with store.connect() as connection, store.transaction(connection):
        message = chats.message(connection, message_id)
        if message is None or message.chat_id != chat_id or message.state != chats.ASKING:
            raise ValueError("that call is not waiting to be allowed")
        if allowed:
            chats.write(connection, message_id, state=chats.WRITING, text="")
        else:
            chats.write(connection, message_id, state=chats.DONE, text=DECLINED)
    if allowed:
        _run(message, allowed=True)
    _next(chat_id)


def decline(connection: sqlite3.Connection, chat_id: int) -> None:
    """The calls still waiting to be allowed, when the person says something else instead."""
    for message in chats.messages(connection, chat_id):
        if message.role == chats.TOOL and message.state == chats.ASKING:
            chats.write(connection, message.id, state=chats.DONE, text=DECLINED)


# --- one call --------------------------------------------------------------------------------------
def _run(message: chats.Message, allowed: bool = False) -> None:
    call = message.calls[0]
    name, given = call["name"], dict(call["arguments"])
    try:
        if name in ASKED and not allowed:
            _write(message.id, state=chats.ASKING, text=_cost(name, given))
            return
        if name in director.BY_NAME and director.BY_NAME[name].costly:
            given["confirm"] = True
        if name == "run":
            wait = float(given.pop("wait_seconds", 0) or 0)
            _follow(message, str(given["run_id"]), wait)
            return
        if name == "look":
            text, media = _look(str(given["path"]))
        elif name in web.NAMES:
            text, media = web.call(name, given)
        else:
            text, media = _read(asyncio.run(mcp._call(name, given, "", _staying)))
        if name in FOLLOWED:
            _follow(message, str(json.loads(text)["run_id"]), float("inf"), said=text)
            return
    except (LookupError, ValueError, TypeError) as problem:
        _write(message.id, state=chats.FAILED, error=f"{type(problem).__name__}: {problem}")
        return
    _write(message.id, state=chats.DONE, text=text, media=media)


def _follow(message: chats.Message, run_id: str, wait: float, said: str = "") -> None:
    """The call waits for the run; `run` is one the chat only follows, so stopping the chat does not cancel it."""
    if runs.one(run_id) is None:
        raise LookupError(f"no run {run_id!r}")
    call = {**message.calls[0], "until": time.time() + wait if wait != float("inf") else None}
    if message.calls[0]["name"] == "run":
        call["follows"] = True
    call = {k: v for k, v in call.items() if v is not None}
    _write(message.id, state=chats.WRITING, run_id=run_id, calls=[call], text=said)
    if wait <= 0:
        with store.connect() as connection:
            found = chats.message(connection, message.id)
        assert found is not None
        _answer_run(found)


def _answer_run(message: chats.Message) -> None:
    assert message.run_id is not None
    found = runs.one(message.run_id)
    if found is None:
        _write(message.id, state=chats.FAILED, error=f"run {message.run_id} is gone")
        return
    summary = json.loads(mcp._described(found, "")[0]["text"])
    summary.pop("made", None)
    media = [a["path"] for a in found["artifacts"] if a["kind"] in ("image", "video")][-MADE_MOST:]
    text = json.dumps(summary | {"files": media}, ensure_ascii=False, indent=1)
    _write(message.id, state=chats.DONE, text=text, media=media)


def _look(name: str) -> tuple[str, list[str]]:
    path = Path(name) if Path(name).is_absolute() else config.OUTPUT_DIR / name
    roots = (config.OUTPUT_DIR.resolve(), (config.PROJECT_DIR / "sources").resolve())
    if not path.is_file() or not any(path.resolve().is_relative_to(root) for root in roots):
        raise LookupError(f"no picture or clip at {name}")
    if path.suffix.lower() not in assets.KINDS:
        raise ValueError(f"{name} is not a picture or a clip")
    return name, [name]


def _cost(name: str, given: dict[str, Any]) -> str:
    route = director.BY_NAME.get(name)
    if route is not None and route.costly:
        return route.costly(given).replace(" Call again with confirm: true to queue it.", "")
    shown = ", ".join(f"{k} {v}" for k, v in given.items())
    return f"{name.replace('_', ' ')}: {shown}."


def _read(content: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """An MCP tool's answer as text, its pictures stored as files the chat shows."""
    words, media = [], []
    for part in content:
        if part["type"] == "text":
            words.append(part["text"])
        elif part["type"] == "image":
            suffix = mimetypes.guess_extension(part.get("mimeType") or "image/png") or ".png"
            media.append(assets.store(base64.b64decode(part["data"]), f"tool{suffix}")["path"])
    return "\n".join(words), media


async def _staying() -> bool:
    return False


def _write(message_id: int, **fields: Any) -> None:
    with store.connect() as connection:
        chats.write(connection, message_id, **fields)


# --- the next answer -----------------------------------------------------------------------------
def _next(chat_id: int) -> None:
    """Queued once every call of the newest answer has settled, unless the person spoke or stopped since."""
    with store.connect() as connection, store.transaction(connection):
        said = chats.messages(connection, chat_id)
        answers = [i for i, one in enumerate(said) if one.role == chats.ASSISTANT]
        if not answers or not said[answers[-1]].calls:
            return
        after = said[answers[-1] + 1 :]
        if not after or any(one.role != chats.TOOL for one in after):
            return
        if any(one.state in (*chats.PENDING, chats.ASKING, chats.STOPPED) for one in after):
            return
        spoke = max((i for i, one in enumerate(said) if one.role == chats.USER), default=-1)
        rounds = sum(1 for i in answers if i > spoke and said[i].calls)
        chat = chats.chat(connection, chat_id)
        assert chat is not None
        answer = chats.add_message(connection, chat_id, chats.ASSISTANT, state=chats.QUEUED)
        asked = {"spent": True} if rounds >= ROUNDS else {}
        chats.write(connection, answer.id, run_id=queue.enqueue_chat(connection, answer.id, chat.title, asked=asked))
