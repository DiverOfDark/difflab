"""What crosses a seam: the frame the next chunk opens on, and the motion running into it.

Three things cross, and they fix different failures (`core.settings.Continuity` has their prices): the
handoff frame is the sharpest of the chunk's last few, not its final one, which is often mid-blur and
would become the next chunk's whole starting trajectory; the tail is the frames running into it, since
a single frame carries pose but no velocity and without it the subject visibly restarts at every join;
the identity reference, fixed in every chunk, is the only one that fights drift, and it is the shot's
business, not this module's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from difflab.core import media
from difflab.core.cache import cache
from difflab.core.keys import Key, key
from difflab.core.types import Clip, Frame


@dataclass(frozen=True)
class Handoff:
    frame: Frame  # what the next chunk opens on
    tail: Clip | None  # the motion running into it; nothing when no tail is carried
    cut: int  # the index of `frame` in its chunk: the join cuts there


def handoff(clip: Clip, window: int, tail_frames: int, anchor: Frame | None = None, strength: float = 0.0) -> Handoff:
    """The sharpest of the last `window` frames and the `tail_frames` before it, both pulled toward
    `anchor`'s colour by `strength` when one is given (`Continuity.color_anchor`)."""
    cut = media.sharpest_index(clip, window)
    frame = clip.frame(cut)
    tail = clip.tail(tail_frames, before=cut) if tail_frames else None
    if anchor is not None and strength:
        gain = media.color_gain(frame, anchor, strength)
        frame = media.recolored(frame, gain)
        tail = media.recolored(tail, gain) if tail is not None else None
    return Handoff(frame, tail, cut)


def handoff_key(chunk: Key, window: int, tail_frames: int, anchor: Frame | None, strength: float) -> Key:
    """Keyed by the chunk it comes from and what was taken from it. The anchor counts only when it is
    applied, so turning the colour pull off does not move every handoff."""
    pulled = anchor.pixels if anchor is not None and strength else None
    return key("handoff", 1, chunk=str(chunk), window=window, tail_frames=tail_frames, anchor=pulled, strength=strength)


def files(at: Key) -> tuple[Path, Path, Path]:
    """Where a handoff lives: the frame as a PNG, which reloads bit for bit; the tail as tensors; the cut."""
    return cache.path(at, ".png"), cache.path(at, ".tail.pt"), cache.path(at, ".json")


def present(at: Key) -> bool:
    frame, tail, note = files(at)
    if not frame.is_file() or not note.is_file():
        return False
    return tail.is_file() or not json.loads(note.read_text())["tail"]


def save(made: Handoff, at: Key) -> Path:
    frame, tail, note = files(at)
    made.frame.pil().save(frame)
    if made.tail is not None:
        torch.save(
            {
                "frames": made.tail.frames,
                "audio": made.tail.audio,
                "fps": made.tail.fps,
                "rate": made.tail.sample_rate,
            },
            tail,
        )
    note.write_text(json.dumps({"cut": made.cut, "tail": made.tail is not None}))
    return frame


def load_tail(path: Path) -> Clip:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return Clip(state["frames"], state["audio"], fps=state["fps"], sample_rate=state["rate"])


def cut_of(at: Key) -> int:
    return int(json.loads(files(at)[2].read_text())["cut"])
