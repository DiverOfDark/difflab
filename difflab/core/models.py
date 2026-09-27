"""One place that decides which heavy model is in memory.

What makes eviction worth doing carefully is that these models do not fit side by side in host RAM,
never mind VRAM: the text encoder maps 25 GB and the DiT 18 GB against 58 GB of system memory. So
dropping one also drops its checkpoint's page cache, which safetensors would otherwise leave behind
(it maps rather than copies).

`budget` is how many heavy models may be resident at once. One is the honest answer on this machine;
the small VAEs are cheap enough that a recipe may raise it.

Light models stay loaded but not on the card. The video VAE alone was 4.85 GB in fp16, and left on
the card through sampling it cost exactly the headroom a 0.8 MP chunk needed - the render ran out of
memory in attention. Moving it across takes a second or two; reloading it would take longer, and keeping
it where it was does not fit.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from difflab.core import ui


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[], Any]
    checkpoints: tuple[Path, ...] = ()  # dropped from the page cache on eviction (a folder: its .safetensors)
    heavy: bool = True  # counts against the residency budget
    unload: Callable[[Any], None] | None = None  # for models that hold their own resources

    def key(self) -> str:
        return self.name


@dataclass
class Registry:
    budget: int = 1
    _resident: dict[str, Any] = field(default_factory=dict)
    _specs: dict[str, ModelSpec] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)  # least recently used first
    _parked: dict[str, torch.device] = field(default_factory=dict)  # light models in host RAM, and their home

    def acquire(self, spec: ModelSpec) -> Any:
        key = spec.key()
        if spec.heavy:
            self._park_light()
        if key in self._resident:
            self._touch(key)
            if key in self._parked:
                self._resident[key].to(self._parked.pop(key))
            elif spec.heavy:
                with ui.step(f"Loading {spec.name}") as detail:
                    detail("skipped, resident")
            return self._resident[key]
        if spec.heavy:
            self._make_room_for(key)
        model = spec.build()
        self._resident[key] = model
        self._specs[key] = spec
        self._touch(key)
        return model

    def keep_only(self, *keep: str) -> None:
        """Names, so a caller need not hold the specs."""
        for key in [k for k in self._resident if k not in keep]:
            self.evict(key)

    def evict(self, key: str) -> None:
        if key not in self._resident:
            return
        spec = self._specs.pop(key)
        model = self._resident.pop(key)
        self._order.remove(key)
        self._parked.pop(key, None)
        if spec.unload is not None:
            spec.unload(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        _drop_page_cache(spec.checkpoints)

    def evict_all(self) -> None:
        self.keep_only()

    def resident(self) -> list[str]:
        return list(self._order)

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _park_light(self) -> None:
        moved = False
        for key, model in self._resident.items():
            if self._specs[key].heavy or key in self._parked or not isinstance(model, torch.nn.Module):
                continue
            device = next((p.device for p in model.parameters()), torch.device("cpu"))
            if device.type == "cpu":
                continue
            model.to("cpu")
            self._parked[key] = device
            moved = True
        if moved:
            gc.collect()
            torch.cuda.empty_cache()

    def _make_room_for(self, incoming: str) -> None:
        heavy = [k for k in self._order if self._specs[k].heavy]
        while len(heavy) >= max(1, self.budget):
            victim = heavy.pop(0)
            ui.note(f"evicting {victim} to make room for {incoming}")
            self.evict(victim)


def _drop_page_cache(paths: Iterable[Path]) -> None:
    files = [file for given in paths for file in (sorted(given.glob("*.safetensors")) if given.is_dir() else [given])]
    for path in files:
        try:
            handle = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(handle)


# A recipe may build its own, but sharing one is the point: it is what lets a block reuse the model
# the block before it loaded.
registry = Registry()
