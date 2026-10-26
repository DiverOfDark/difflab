"""The VLM as a tool: one question about a picture, one answer. The prompt writers build on it."""

from __future__ import annotations

from difflab.core import models
from difflab.core.hub import Checkpoint
from difflab.core.settings import VlmSettings
from difflab.core.types import Frame
from difflab.models.exl3vl import Exl3VL


def spec(checkpoint: str) -> models.ModelSpec:
    pinned = Checkpoint.parse(checkpoint)
    return models.ModelSpec(
        name=f"vlm:{pinned.repo}",
        build=lambda: Exl3VL(pinned),
        # ~17 GB on the card, so it is heavy: evicted when anything else heavy is acquired. Not parked:
        # ExLlamaV3's kernels hold pointers into its tensors. Nor its files kept in tmpfs: loaded from
        # there it took 6.0 s against 5.3-11.1 s off the NVMe, its own setup being most of a load, and
        # writing the 15.3 GB set the machine swapping (2026-10-26).
        checkpoints=(pinned.path,),  # a folder: every .safetensors in it
        unload=lambda vlm: vlm.unload(),
        parks=False,
    )


def evict() -> None:
    """Drop whichever VLM is resident: 17 GB the next model would rather have."""
    for name in models.registry.resident():
        if name.startswith("vlm:"):
            models.registry.evict(name)


def _vlm(settings: VlmSettings) -> Exl3VL:
    model: Exl3VL = models.registry.acquire(spec(settings.checkpoint))
    return model


def ask(frame: Frame, question: str, title: str = "Qwen3.8", settings: VlmSettings | None = None) -> str:
    """Anything you want to know about the picture. The answer streams into a panel."""
    settings = settings or VlmSettings()
    return _vlm(settings).ask(frame.pil(), question, title=title, **settings.generation())
