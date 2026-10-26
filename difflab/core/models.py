"""One place that decides which heavy model is in memory.

What makes eviction worth doing carefully is that these models do not fit side by side in host RAM,
never mind VRAM: the text encoder maps 25 GB and the DiT 18 GB against 58 GB of system memory. So
dropping one also drops its checkpoint's page cache, which safetensors would otherwise leave behind
(it maps rather than copies). A heavy model evicted while host RAM is free is parked there instead, and
`hostcache` frees it the moment the room is wanted.

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

from difflab.core import hostcache, ui
from difflab.models import offload


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[], Any]
    checkpoints: tuple[Path, ...] = ()  # dropped from the page cache on eviction (a folder: its .safetensors)
    heavy: bool = True  # counts against the residency budget
    unload: Callable[[Any], None] | None = None  # for models that hold their own resources
    # Kept in host RAM on eviction while there is room; not for native code holding pointers into the card.
    parks: bool = True

    def key(self) -> str:
        return self.name


@dataclass
class Registry:
    budget: int = 1
    _resident: dict[str, Any] = field(default_factory=dict)
    _specs: dict[str, ModelSpec] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)  # least recently used first
    _parked: dict[str, torch.device] = field(default_factory=dict)  # light models in host RAM, and their home
    _cold: dict[str, tuple[ModelSpec, Any, offload.Parked]] = field(default_factory=dict)  # heavy, evicted to RAM
    # What each build grew the process by, learned on its first load: file sizes overstate it, since what goes
    # to the card passes through the page cache, which the kernel reclaims by itself.
    _footprint: dict[str, int] = field(default_factory=dict)

    def acquire(self, spec: ModelSpec) -> Any:
        key = spec.key()
        if spec.heavy:
            self._park_light()
        if key in self._resident:
            self._touch(key)
            if key in self._parked:
                device = self._parked.pop(key)
                self._roomy(lambda: self._resident[key].to(device), spec)
            elif spec.heavy:
                with ui.step(f"Loading {spec.name}") as detail:
                    detail("skipped, resident")
            return self._resident[key]
        if spec.heavy:
            self._make_room_for(spec)
        model = self._warm(spec)
        if model is None:
            hostcache.room(self._footprint.get(key, 0))
            before = hostcache.rss()
            model = self._roomy(spec.build, spec)
            self._footprint[key] = max(0, hostcache.rss() - before)
        self._resident[key] = model
        self._specs[key] = spec
        self._touch(key)
        return model

    def roomy(self, spec: ModelSpec, work: Callable[[], Any]) -> Any:
        """`work` with `spec`'s model, given the heavy models' room if it runs out: a judge's VLM (17.4 GB)
        left resident beside the video VAE (4.85 GB) fits, and then encoding a reference clip did not."""
        return self._roomy(work, spec)

    def evict(self, key: str, incoming: int = 0, keep: bool = True) -> None:
        """`incoming`: host bytes the model taking its place will want, so it is not parked into its room."""
        if key not in self._resident:
            return
        spec = self._specs.pop(key)
        model = self._resident.pop(key)
        self._order.remove(key)
        self._parked.pop(key, None)
        if not (keep and spec.heavy and spec.parks and self._park(spec, model, incoming)):
            _discard(spec, model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        _drop_page_cache(spec.checkpoints)

    def release(self) -> list[str]:
        released = list(self._order)
        for key in released:
            self.evict(key, keep=False)
        return released + hostcache.release_all("idle")

    def _park(self, spec: ModelSpec, model: Any, incoming: int) -> bool:
        modules = offload.modules_of(model)
        if not modules or not hostcache.calm() or not hostcache.room(offload.on_card(modules) + incoming):
            return False
        key = spec.key()
        try:
            parked = offload.park(modules)
        except RuntimeError as problem:  # out of host memory part way: what moved goes with the model
            ui.note(f"could not keep {spec.name} in RAM ({problem})")
            return False
        self._cold[key] = (spec, model, parked)
        hostcache.hold(key, parked.size, lambda: self._forget(key))
        ui.note(f"keeping {spec.name} in RAM, {parked.size / 1024**3:.1f} GB")
        return True

    def _warm(self, spec: ModelSpec) -> Any:
        key = spec.key()
        if not hostcache.take(key):
            return None
        _, model, parked = self._cold.pop(key)
        with ui.step(f"Loading {spec.name}") as detail:
            try:
                self._roomy(lambda: offload.unpark(parked), spec)
            except torch.OutOfMemoryError:
                _discard(spec, model)
                raise
            except RuntimeError as problem:
                _discard(spec, model)
                detail(f"rebuilt, RAM copy failed ({problem})")
                return None
            detail(f"from RAM, {parked.size / 1024**3:.1f} GB")
        return model

    def _forget(self, key: str) -> None:
        spec, model, _ = self._cold.pop(key)
        _discard(spec, model)
        del model
        gc.collect()

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

    def _roomy(self, make: Callable[[], Any], spec: ModelSpec) -> Any:
        """A light model that runs out of memory beside a heavy one gets the heavy one's room: the VLM,
        left resident by a plan, filled the card the video VAE then needed at the start of a render."""
        try:
            return make()
        except torch.OutOfMemoryError:
            if offload.release():
                ui.note(f"releasing resident blocks to make room for {spec.name}")
                try:
                    return make()
                except torch.OutOfMemoryError:
                    pass
            heavy = [k for k in self._order if self._specs[k].heavy]
            if spec.heavy or not heavy:
                raise
            for victim in heavy:
                ui.note(f"evicting {victim} to make room for {spec.name}")
                self.evict(victim)
            return make()

    def _make_room_for(self, incoming: ModelSpec) -> None:
        heavy = [k for k in self._order if self._specs[k].heavy]
        while len(heavy) >= max(1, self.budget):
            victim = heavy.pop(0)
            ui.note(f"evicting {victim} to make room for {incoming.name}")
            self.evict(victim, self._footprint.get(incoming.key(), 0))


def _discard(spec: ModelSpec, model: Any) -> None:
    if spec.unload is not None:
        spec.unload(model)


def _files(paths: Iterable[Path]) -> list[Path]:
    return [file for given in paths for file in (sorted(given.glob("*.safetensors")) if given.is_dir() else [given])]


def _drop_page_cache(paths: Iterable[Path]) -> None:
    for path in _files(paths):
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
