"""The next reply in a chat with the VLM, written into its message row as it arrives."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import av
from PIL import Image

from difflab import config, prompts, store
from difflab.core import context
from difflab.core.imaging import load_image
from difflab.core.settings import ChatSettings
from difflab.core.types import Frame
from difflab.models.exl3vl import Turn
from difflab.store import chats
from difflab.tools.ask_vlm import _vlm
from difflab.tools.direct import shrink

VIDEO = {".mp4", ".mov", ".mkv", ".webm"}
WRITE_EVERY = 0.25  # seconds between writes of the reply so far; the page polls at about this rate


def reply(
    message_id: int,
    settings: ChatSettings,
    as_json: bool = False,
    schema: dict[str, Any] | None = None,
    effort: str | None = config.VLM_REASONING_EFFORT,
    conversation: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """`conversation` stands in for the chat's own messages: `{"role", "text", "media", "thought", "calls"}`
    each, as the OpenAI endpoint hands them over with the tools they may call."""
    with store.connect() as connection:
        answer = chats.message(connection, message_id)
        if answer is None:
            raise LookupError(f"no chat message {message_id}: the chat was deleted")
        chat = chats.chat(connection, answer.chat_id)
        said = [one for one in chats.messages(connection, answer.chat_id) if one.id < message_id]
        chats.write(connection, message_id, state=chats.WRITING, text="", thinking="", error="")
    assert chat is not None
    if conversation is None:
        turns = [
            Turn(one.role, [*_shown(one.media), one.text], one.thinking)
            for one in said
            if one.role == chats.USER or (one.state == chats.DONE and one.text)
        ]
    else:
        turns = [
            Turn(
                one["role"], [*_shown(one.get("media", [])), one["text"]], one.get("thought", ""), one.get("calls", [])
            )
            for one in conversation
        ]
    usage: dict[str, int] = {}
    last = 0.0

    def heard(thought: str, words: str) -> None:
        nonlocal last
        if time.monotonic() - last >= WRITE_EVERY:
            last = time.monotonic()
            with store.connect() as connection:
                chats.write(connection, message_id, thinking=thought, text=words)

    try:
        text = _vlm(settings).chat(
            turns,
            system=chat.system or prompts.CHAT_SYSTEM_PROMPT,
            title="reply",
            echo=False,
            numbered=True,
            heard=heard,
            as_json=as_json,
            schema=schema,
            effort=effort,
            tools=tools or (),
            usage=usage,
            **settings.generation(),
        )
    except BaseException as problem:
        state = chats.STOPPED if isinstance(problem, context.Cancelled) else chats.FAILED
        with store.connect() as connection:
            error = f"{type(problem).__name__}: {problem}" if state == chats.FAILED else ""
            chats.write(connection, message_id, state=state, error=error)
        raise
    with store.connect() as connection:
        chats.write(connection, message_id, text=text, state=chats.DONE)
        chats.touch(connection, answer.chat_id)
    return {"text": text, "usage": usage}


def _shown(media: list[str]) -> list[str | Image.Image]:
    parts: list[str | Image.Image] = []
    for name in media:
        path = Path(name)
        if path.suffix.lower() in VIDEO:
            frames, every = _frames(path)
            parts += [f"Video {path.name}, a frame every {every:.1f} s:", *frames]
        elif path.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            parts.append(f"(attached sound {path.name}: you cannot hear it)")
        else:
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
