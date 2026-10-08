"""One video from a chapter's chunks, joined at their handoff frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from difflab.core import media
from difflab.core.keys import Key, key


def assemble(parts: Sequence[tuple[Path, Path]], joins: Sequence[media.Join], destination: Path) -> tuple[int, float]:
    """`parts` are `(chunk.mp4, chunk.audio.pt)` pairs in order. Returns `(frames, seconds)` of what was written."""
    return media.join_files(parts, destination, joins)


def joins_of(inputs: Mapping[str, Any]) -> list[media.Join]:
    """A job's joins. One queued before scenes were told apart blended every join as a chained one does."""
    cuts = [int(cut) for cut in inputs["cuts"]]
    if "fades" not in inputs:
        crossfade = float(inputs.get("crossfade") or 0.0)
        return [media.Join(cut, crossfade, True) for cut in cuts]
    return [
        media.Join(cut, float(fade), bool(chained))
        for cut, fade, chained in zip(cuts, inputs["fades"], inputs["chained"], strict=True)
    ]


def assemble_key(chunks: Sequence[str], joins: Sequence[media.Join]) -> Key:
    """The chunks by their keys, never their paths: the same chunks joined the same way are one video."""
    # Version 2 keys each join's own fade and whether it is chained. The finish's key holds this one, so
    # every chapter finished before it is joined and retimed again on its next render; the enlarged shots
    # are keyed apart and are not redone.
    return key("assemble", 2, chunks=list(chunks), joins=[list(join) for join in joins])
