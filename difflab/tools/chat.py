"""The next reply in a chat with the VLM, written into its message row as it arrives."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import av
from PIL import Image

from difflab import config, prompts, store
from difflab.core import context, ui
from difflab.core.imaging import load_image
from difflab.core.settings import ChatSettings
from difflab.core.types import Frame
from difflab.models import exl3vl
from difflab.models.exl3vl import Turn
from difflab.store import chats
from difflab.tools.ask_vlm import _vlm
from difflab.tools.direct import shrink

VIDEO = {".mp4", ".mov", ".mkv", ".webm"}
WRITE_EVERY = 0.25  # seconds between writes of the reply so far; the page polls at about this rate
PICTURED_KEPT = 4  # messages whose pictures stay shown once a chat outgrows the context
SUMMARY_TOKENS = 1500  # the summary a compaction writes is asked for in 400 words


def reply(
    message_id: int,
    settings: ChatSettings,
    as_json: bool = False,
    schema: dict[str, Any] | None = None,
    effort: str | None = config.VLM_REASONING_EFFORT,
    conversation: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    briefing: str = "",
) -> dict[str, Any]:
    """`conversation` stands in for the chat's own messages: `{"role", "text", "media", "thought", "calls"}`
    each, as the OpenAI endpoint hands them over with the tools they may call. `briefing` follows the
    system prompt."""
    with store.connect() as connection:
        answer = chats.message(connection, message_id)
        if answer is None:
            raise LookupError(f"no chat message {message_id}: the chat was deleted")
        chat = chats.chat(connection, answer.chat_id)
        said = [one for one in chats.messages(connection, answer.chat_id) if one.id < message_id]
        chats.write(connection, message_id, state=chats.WRITING, text="", thinking="", error="")
    assert chat is not None
    system = "\n\n".join(part for part in (chat.system or prompts.CHAT_SYSTEM_PROMPT, briefing) if part)
    usage: dict[str, int] = {}
    last = 0.0

    def heard(thought: str, words: str) -> None:
        nonlocal last
        if time.monotonic() - last >= WRITE_EVERY:
            last = time.monotonic()
            with store.connect() as connection:
                chats.write(connection, message_id, thinking=thought, text=exl3vl.before_calls(words, False))

    def asked(turns: list[Turn], summarising: bool = False) -> str:
        return _vlm(settings, chatting=True).chat(
            turns,
            kept=conversation is None and not summarising,
            system=system,
            title="summary" if summarising else "reply",
            echo=False,
            numbered=True,
            heard=None if summarising else heard,
            as_json=as_json and not summarising,
            schema=None if summarising else schema,
            # The template puts the effort in the system prompt's first line: summarised unthinking, an 18k-token
            # chat matched only the 6.7k opening a call without thinking had cached (2026-10-26).
            effort=effort,
            tools=tools or (),
            usage={} if summarising else usage,
            **settings.generation() | ({"max_new_tokens": SUMMARY_TOKENS} if summarising else {}),
        )

    try:
        if conversation is not None:
            text = asked(
                [
                    Turn(one["role"], [*_shown(one.get("media", [])), one["text"]], one.get("thought", ""),
                         one.get("calls", []))
                    for one in conversation
                ]
            )  # fmt: skip
        else:
            if _due(chat, said):
                with store.connect() as connection:
                    chats.write(connection, message_id, thinking="Summarising the chat so far, to make room.")
                chat = _compact(chat, said, lambda turns: asked(turns, summarising=True))
            text = _fitting([one for one in said if one.id > chat.summary_through], asked, _opening(chat))
    except BaseException as problem:
        state = chats.STOPPED if isinstance(problem, context.Cancelled) else chats.FAILED
        with store.connect() as connection:
            error = f"{type(problem).__name__}: {problem}" if state == chats.FAILED else ""
            chats.write(connection, message_id, state=state, error=error)
        raise
    with store.connect() as connection:
        chats.write(connection, message_id, text=text, state=chats.DONE, tokens=usage.get("prompt_tokens", 0))
        chats.touch(connection, answer.chat_id)
    return {"text": text, "usage": usage}


def _due(chat: chats.Chat, said: list[chats.Message]) -> bool:
    """Asked for, or the last answer was written from a prompt past `config.CHAT_COMPACT_TOKENS`."""
    last = next((one.tokens for one in reversed(said) if one.role == chats.ASSISTANT and one.tokens), 0)
    return chat.compact_next or last >= config.CHAT_COMPACT_TOKENS


def _compact(chat: chats.Chat, said: list[chats.Message], summarise: Callable[[list[Turn]], str]) -> chats.Chat:
    """The chat summarised up to its newest `config.CHAT_KEPT_TURNS` messages from the person, which stay as said."""
    spoke = [one.id for one in said if one.role == chats.USER]
    kept = config.CHAT_KEPT_TURNS if len(spoke) > config.CHAT_KEPT_TURNS else 1
    through = spoke[-kept] - 1 if spoke else 0
    if through > chat.summary_through:
        shown = [one for one in said if one.id > chat.summary_through]
        summary = exl3vl.before_calls(
            _fitting(shown, summarise, _opening(chat), [Turn(chats.USER, [prompts.CHAT_COMPACT_PROMPT])]), True
        )
        changes: dict[str, str | int] = {"summary": summary.strip(), "summary_through": through, "compact_next": 0}
    else:
        changes = {"compact_next": 0}
    with store.connect() as connection:
        chats.update_chat(connection, chat.id, **changes)
        found = chats.chat(connection, chat.id)
    assert found is not None
    return found


def _opening(chat: chats.Chat) -> list[Turn]:
    return [Turn(chats.USER, [f"{prompts.CHAT_SUMMARY_LEAD}\n\n{chat.summary}"])] if chat.summary else []


def _fitting(
    said: list[chats.Message], ask: Callable[[list[Turn]], str], before: list[Turn], after: list[Turn] | None = None
) -> str:
    """Over the context, the pictures of all but the newest messages go before anything said does."""
    pictured = [one.id for one in said if one.media]
    cuts = [0] + [pictured[-kept] for kept in (PICTURED_KEPT, 1) if kept < len(pictured)]
    for at, cut in enumerate(cuts):
        try:
            return ask([*before, *_turns(said, cut), *(after or [])])
        except ValueError as problem:
            if not str(problem).startswith("the prompt is") or at == len(cuts) - 1:
                raise
            ui.note("the chat is over the VLM's context: the pictures of older messages are named only")
    raise AssertionError("unreachable")


def _turns(said: list[chats.Message], pictured_from: int) -> list[Turn]:
    """The chat's own messages as the VLM reads them; pictures of a message before `pictured_from` are named only."""
    turns = []
    for one in said:
        media = _shown(one.media, labelled=True) if one.id >= pictured_from else _named(one.media)
        if one.role == chats.USER:
            turns.append(Turn(chats.USER, [*media, one.text]))
        elif one.role == chats.ASSISTANT and one.state == chats.DONE and (one.text or one.calls):
            calls = [{"name": call["name"], "arguments": call["arguments"]} for call in one.calls]
            turns.append(Turn(chats.ASSISTANT, [one.text], one.thinking, calls))
        elif one.role == chats.TOOL and one.state not in (*chats.PENDING, chats.ASKING):
            turns.append(Turn(chats.TOOL, [*media, one.text or f"failed: {one.error}"]))
    return turns


def _named(media: list[str]) -> list[str | Image.Image]:
    return [f"({_where(Path(name))}, shown earlier)" for name in media]


def _where(path: Path) -> str:
    return str(path.relative_to(config.OUTPUT_DIR)) if path.is_relative_to(config.OUTPUT_DIR) else str(path)


def _shown(media: list[str], labelled: bool = False) -> list[str | Image.Image]:
    """`labelled` puts each file's path before it, as the tools name it."""
    parts: list[str | Image.Image] = []
    for name in media:
        path = Path(name) if Path(name).is_absolute() else config.OUTPUT_DIR / name
        label = f"{_where(path)}:" if labelled else ""
        if path.suffix.lower() in VIDEO:
            frames, every = _frames(path)
            parts += [f"{label} video {path.name}, a frame every {every:.1f} s:".strip(), *frames]
        elif path.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            parts.append(f"{label} (attached sound {path.name}: you cannot hear it)".strip())
        else:
            parts += [label] if label else []
            parts.append(shrink(Frame.from_pil(load_image(path)), config.CHAT_PHOTO_PIXELS))
    return parts


def _frames(path: Path) -> tuple[list[Image.Image], float]:
    """Spread over the whole video, decoding past the frames between rather than holding them."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        seconds = float(stream.duration * stream.time_base) if stream.duration and stream.time_base else 0.0
        if not seconds and container.duration:
            seconds = container.duration / av.time_base
        every = max(config.CHAT_VIDEO_EVERY, seconds / config.CHAT_VIDEO_FRAMES)
        frames: list[Image.Image] = []
        due = 0.0
        for frame in container.decode(video=0):
            if frame.time is None or frame.time + 1e-3 < due:
                continue
            frames.append(shrink(Frame.from_pil(frame.to_image().convert("RGB")), config.CHAT_FRAME_PIXELS))
            due = frame.time + every
            if len(frames) >= config.CHAT_VIDEO_FRAMES:
                break
    if not frames:
        raise ValueError(f"{path.name} has no video frames")
    return frames, every
