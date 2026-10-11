"""Content-addressed results, so a crash costs one chunk instead of the whole video.

Every step a tool cares about is written under its key: the tool's name and version plus its
inputs. Re-running with the same inputs finds the file and skips the work; change a seed or a prompt
and the key moves, so there is no stale-result problem to reason about.

Only step boundaries are cached, never model weights - those are the registry's business.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch

from difflab import config
from difflab.core import ui
from difflab.core.keys import Key
from difflab.core.types import Frame

T = TypeVar("T")
DEFAULT_DIR = config.OUTPUT_DIR / "blobs"


@dataclass
class Cache:
    directory: Path = DEFAULT_DIR
    enabled: bool = True

    def path(self, key: Key, suffix: str) -> Path:
        folder = self.directory / key.tool
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{key.token}{suffix}"

    def get_or_make(
        self,
        key: Key,
        suffix: str,
        produce: Callable[[], T],
        save: Callable[[T, Path], None],
        load: Callable[[Path], T],
    ) -> T:
        if not self.enabled:
            return produce()
        target = self.path(key, suffix)
        if target.exists():
            ui.note(f"cached {key}")
            os.utime(target)  # work/cleanup.py ages a blob from its last use
            return load(target)
        value = produce()
        save(value, target)
        return value

    def text(self, key: Key, produce: Callable[[], str]) -> str:
        return self.get_or_make(key, ".txt", produce, _write_text, lambda path: path.read_text(encoding="utf-8"))

    def frame(self, key: Key, produce: Callable[[], Frame]) -> Frame:
        """8-bit in and 8-bit out: `Frame.pil` and `Frame.from_pil` round-trip a frame that came from a
        model's own 8-bit output exactly, so a cached picture keys the same as the one it replaced."""
        return self.get_or_make(
            key, ".png", produce, lambda value, path: value.pil().save(path), lambda path: Frame.load_png(path)
        )

    def tensors(self, key: Key, produce: Callable[[], dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return self.get_or_make(
            key,
            ".pt",
            produce,
            lambda value, path: torch.save(value, path),
            lambda path: torch.load(path, map_location="cpu", weights_only=True),
        )


def _write_text(value: str, path: Path) -> None:
    path.write_text(value, encoding="utf-8")


cache = Cache()
