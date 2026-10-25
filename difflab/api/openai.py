"""The VLM behind OpenAI's chat completions, for other tools. A request is a chat of its own that lives as
long as the request; its reply waits its turn on the card like the page's."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from difflab import store
from difflab.api import assets, runs
from difflab.api import chats as chatting
from difflab.store import chats
from difflab.work import queue

MODEL = "difflab"
POLL = 0.25  # seconds; the worker writes the reply so far every `tools.chat.WRITE_EVERY`
KEEPALIVE = 15.0  # seconds of a stream with nothing to say, behind a render, before it says so
# The template takes xhigh, medium and low; OpenAI's names map onto them.
EFFORTS = {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "none": None, "minimal": None}

OPEN_CALL = "<tool_call>"
CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
TOO_LONG = re.compile(r"the prompt is ([\d,]+) tokens and the VLM holds ([\d,]+)")


class RequestError(ValueError):
    """A request this endpoint cannot answer as asked."""


@dataclass(frozen=True)
class Asked:
    chat_id: int
    answer_id: int
    tools: list[dict[str, Any]]
    usage: bool  # `stream_options.include_usage`


def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "difflab"}]}


def ask(body: dict[str, Any]) -> Asked:
    """The request as a chat with its reply queued, the conversation handed to the job whole."""
    if int(body.get("n") or 1) != 1:
        raise RequestError("n must be 1")
    tools = [tool["function"] for tool in body.get("tools") or [] if tool.get("type") == "function"]
    if body.get("tool_choice") == "none":
        tools = []
    system: list[str] = []
    conversation: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        role = message.get("role")
        text, media = _content(message.get("content"))
        if role in ("system", "developer"):
            system.append(text)
        elif role == chats.USER:
            conversation.append({"role": role, "text": text, "media": media})
        elif role == chats.ASSISTANT:
            thought = message.get("reasoning_content") or message.get("reasoning") or ""
            calls = [_called(call["function"]) for call in message.get("tool_calls") or []]
            conversation.append({"role": role, "text": text, "thought": thought, "calls": calls})
        elif role == "tool":
            conversation.append({"role": role, "text": text})
        else:
            raise RequestError(f"messages of role {role!r} are not supported")
    if not conversation or conversation[-1]["role"] == chats.ASSISTANT:
        raise RequestError("the last message must be the user's or a tool's")
    asked = {"conversation": conversation, "tools": tools, **_shaped(body)}
    title = next((one["text"] for one in reversed(conversation) if one["role"] == chats.USER), "")
    with store.connect() as connection, store.transaction(connection):
        chat = chats.add_chat(connection, title=f"api · {title[:60]}", system="\n\n".join(system))
        answer = chats.add_message(connection, chat.id, chats.ASSISTANT, state=chats.QUEUED)
        run_id = queue.enqueue_chat(connection, answer.id, chat.title, _overrides(body), asked)
        chats.write(connection, answer.id, run_id=run_id)
    return Asked(chat.id, answer.id, tools, bool((body.get("stream_options") or {}).get("include_usage")))


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


def _called(function: dict[str, Any]) -> dict[str, Any]:
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            arguments = {"arguments": arguments}
    return {"name": function["name"], "arguments": arguments}


def _stored(url: str) -> str:
    if not url.startswith("data:"):
        raise RequestError("images must be sent inline, as data: URLs")
    head, _, data = url.partition(",")
    suffix = mimetypes.guess_extension(head.removeprefix("data:").split(";")[0]) or ".png"
    return str(assets.store(base64.b64decode(data), f"api{suffix}")["path"])


def _overrides(body: dict[str, Any]) -> dict[str, Any]:
    chosen = {name: body[name] for name in ("temperature", "top_p", "seed") if body.get(name) is not None}
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
        if effort not in EFFORTS:
            raise RequestError(f"reasoning_effort {effort!r}: one of {', '.join(EFFORTS)}")
        shaped["effort"] = EFFORTS[effort]
    return shaped


def calls(text: str, tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """The answer's words before its first tool call, and its calls as OpenAI writes them."""
    found = list(CALL.finditer(text))
    if not found:
        return text, []
    kinds = {tool["name"]: (tool.get("parameters") or {}).get("properties") or {} for tool in tools}
    made = []
    for match in found:
        name = match.group(1).strip()
        arguments = {
            key.strip(): _typed(value, kinds.get(name, {}).get(key.strip(), {}))
            for key, value in PARAMETER.findall(match.group(2))
        }
        made.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        )
    return text[: found[0].start()].rstrip(), made


def _typed(value: str, schema: dict[str, Any]) -> Any:
    """A string parameter as written; any other as the JSON it should be, or as written when it is not."""
    if schema.get("type", "string") == "string":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _shown(text: str, settled: bool) -> str:
    """What of the answer so far can go out: nothing from a tool call on, nor what may be the start of one."""
    if OPEN_CALL in text:
        return text[: text.index(OPEN_CALL)].rstrip()
    if settled:
        return text
    for cut in range(min(len(OPEN_CALL) - 1, len(text)), 0, -1):
        if OPEN_CALL.startswith(text[-cut:]):
            return text[:-cut]
    return text


@dataclass(frozen=True)
class Settled:
    message: chats.Message
    usage: dict[str, int]


async def _follow(asked: Asked, gone: Callable[[], Awaitable[bool]]) -> AsyncIterator[tuple[str, str, Settled | None]]:
    """`(thought, words)` as they grow, the settled reply last; `("", "", None)` when there has been
    nothing to say for `KEEPALIVE`. A caller that leaves stops the reply; the chat goes either way."""
    thought = words = ""
    settled: Settled | None = None
    quiet = time.monotonic()
    try:
        while True:
            found = await asyncio.to_thread(_read, asked.answer_id)
            if isinstance(found, Settled):
                settled, message = found, found.message
            else:
                message = found
            more_thought = message.thinking[len(thought) :] if message.thinking.startswith(thought) else ""
            shown = _shown(message.text, settled is not None)
            if settled is not None and message.state == chats.DONE:
                shown = calls(message.text, asked.tools)[0]
            more_words = shown[len(words) :] if shown.startswith(words) else ""
            thought, words = thought + more_thought, words + more_words
            if more_thought or more_words or settled:
                quiet = time.monotonic()
                yield more_thought, more_words, settled
            elif time.monotonic() - quiet >= KEEPALIVE:
                quiet = time.monotonic()
                yield "", "", None
            if settled or await gone():
                return
            await asyncio.sleep(POLL)
    finally:
        # Not awaited: a client that left cancels this task, and an await here never returns.
        _close(asked.chat_id, settled is not None)


def _read(answer_id: int) -> chats.Message | Settled:
    """Settled once its job is too, so the usage the job wrote is there to read."""
    with store.connect() as connection:
        found = chats.message(connection, answer_id)
        if found is None:
            raise RuntimeError("the reply was deleted")
        job = next(iter(store.jobs_of(connection, found.run_id)), None) if found.run_id else None
    if found.state in chats.PENDING:
        if found.run_id is None or runs.gone(found.run_id):
            with store.connect() as connection:
                chats.write(connection, answer_id, state=chats.FAILED, error="the run ended without an answer")
                found = chats.message(connection, answer_id)
            assert found is not None
            return Settled(found, {})
        return found
    if job is not None and not job.settled and not runs.gone(found.run_id or ""):
        return found
    return Settled(found, dict((job.outputs if job else {}).get("usage") or {}))


def _close(chat_id: int, settled: bool) -> None:
    if not settled:
        chatting.stop(chat_id)
    with store.connect() as connection:
        chats.delete_chat(connection, chat_id)


def failure(message: chats.Message) -> tuple[int, dict[str, Any]] | None:
    """The status and OpenAI's error body for a reply that did not finish; overflow in the words agents
    look for before they compact."""
    if message.state == chats.DONE:
        return None
    if message.state == chats.STOPPED:
        return 499, {"message": "the reply was stopped", "type": "server_error"}
    if long := TOO_LONG.search(message.error):
        given, held = (int(n.replace(",", "")) for n in long.groups())
        said = (
            f"This model's maximum context length is {held} tokens. However, your messages resulted in {given} tokens."
        )
        return 400, {"message": said, "type": "invalid_request_error", "code": "context_length_exceeded"}
    return 500, {"message": message.error or "the reply failed", "type": "server_error"}


def _usage(counted: dict[str, int]) -> dict[str, int]:
    prompt, answer = int(counted.get("prompt_tokens", 0)), int(counted.get("completion_tokens", 0))
    return {"prompt_tokens": prompt, "completion_tokens": answer, "total_tokens": prompt + answer}


def _chunk(made: str, delta: dict[str, Any] | None, finish: str | None = None, **more: Any) -> str:
    data = {
        "id": made,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}],
        **more,
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream(asked: Asked, gone: Callable[[], Awaitable[bool]]) -> AsyncIterator[str]:
    made = f"chatcmpl-{uuid.uuid4().hex}"
    yield _chunk(made, {"role": "assistant", "content": ""})
    async for thought, words, settled in _follow(asked, gone):
        if not (thought or words or settled):
            yield ": waiting for the card\n\n"  # an SSE comment, which clients skip
        if thought:
            yield _chunk(made, {"reasoning_content": thought})
        if words:
            yield _chunk(made, {"content": words})
        if settled is None:
            continue
        failed = failure(settled.message)
        if failed:
            yield f"data: {json.dumps({'error': failed[1]}, ensure_ascii=False)}\n\n"
            break
        _, made_calls = calls(settled.message.text, asked.tools)
        if made_calls:
            listed = [{"index": i, **call} for i, call in enumerate(made_calls)]
            yield _chunk(made, {"tool_calls": listed})
        yield _chunk(made, {}, "tool_calls" if made_calls else "stop")
        if asked.usage:
            yield _chunk(made, None, usage=_usage(settled.usage))
    yield "data: [DONE]\n\n"


async def complete(asked: Asked, gone: Callable[[], Awaitable[bool]]) -> tuple[int, dict[str, Any]]:
    settled: Settled | None = None
    async for _, _, last in _follow(asked, gone):
        settled = last
    if settled is None:
        return 499, {"error": {"message": "the client went away", "type": "server_error"}}
    failed = failure(settled.message)
    if failed:
        return failed[0], {"error": failed[1]}
    words, made_calls = calls(settled.message.text, asked.tools)
    message: dict[str, Any] = {"role": "assistant", "content": words or (None if made_calls else "")}
    if settled.message.thinking:
        message["reasoning_content"] = settled.message.thinking
    if made_calls:
        message["tool_calls"] = made_calls
    return 200, {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if made_calls else "stop"}],
        "usage": _usage(settled.usage),
    }
