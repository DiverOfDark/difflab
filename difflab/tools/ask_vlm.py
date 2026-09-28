"""The VLM as a tool: one question about a picture, one answer. The prompt writers build on it."""

from __future__ import annotations

import json

from difflab.core import models
from difflab.core.hub import Checkpoint
from difflab.core.settings import VlmSettings
from difflab.core.types import Frame
from difflab.models.exl3vl import Exl3VL
from difflab.models.qwenvl import QwenVL

Vlm = QwenVL | Exl3VL


def build(checkpoint: Checkpoint) -> Vlm:
    """ExLlamaV3 for an EXL3 checkpoint, transformers for anything else."""
    model_config = json.loads((checkpoint.fetch() / "config.json").read_text(encoding="utf-8"))
    if (model_config.get("quantization_config") or {}).get("quant_method") == "exl3":
        return Exl3VL(checkpoint)
    return QwenVL(checkpoint)


def spec(checkpoint: str) -> models.ModelSpec:
    pinned = Checkpoint.parse(checkpoint)
    return models.ModelSpec(
        name=f"qwenvl:{pinned.repo}",
        build=lambda: build(pinned),
        # Dropped from the page cache on eviction, since H3 maps 25 + 18 GB of a 58 GB machine right
        # after. ~17 GB on the card, so it is heavy: evicted, not parked in host RAM, when anything
        # else heavy is acquired.
        checkpoints=(pinned.path,),  # a folder: every .safetensors in it
        unload=lambda vlm: vlm.unload(),
    )


def evict() -> None:
    """Drop whichever VLM is resident: 17 GB the next model would rather have."""
    for name in models.registry.resident():
        if name.startswith("qwenvl:"):
            models.registry.evict(name)


def _vlm(settings: VlmSettings) -> Vlm:
    model: Vlm = models.registry.acquire(spec(settings.checkpoint))
    return model


def ask(frame: Frame, question: str, title: str = "Qwen3.8", settings: VlmSettings | None = None) -> str:
    """Anything you want to know about the picture. The answer streams into a panel."""
    settings = settings or VlmSettings()
    return _vlm(settings).ask(frame.pil(), question, title=title, **settings.generation())
