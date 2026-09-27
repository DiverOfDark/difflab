"""One video from a chapter's chunks, joined at their handoff frames."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from difflab.core import media
from difflab.core.keys import Key, key


def assemble(
    parts: Sequence[tuple[Path, Path]], cuts: Sequence[int | None], crossfade: float, destination: Path
) -> tuple[int, float]:
    """`parts` are `(chunk.mp4, chunk.audio.pt)` pairs in order; `cuts[k]` is where chunk k+1 opened on
    chunk k. Returns `(frames, seconds)` of what was written."""
    return media.join_files(parts, destination, crossfade, cuts)


def assemble_key(chunks: Sequence[str], cuts: Sequence[int | None], crossfade: float) -> Key:
    """The chunks by their keys, never their paths: the same chunks joined the same way are one video."""
    return key("assemble", 1, chunks=list(chunks), cuts=list(cuts), crossfade=crossfade)
