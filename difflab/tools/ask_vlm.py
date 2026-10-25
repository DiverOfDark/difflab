"""The VLM as a tool: one question about a picture, one answer. The prompt writers build on it."""

from __future__ import annotations

from difflab import config
from difflab.core import models
from difflab.core.hub import Checkpoint
from difflab.core.settings import VlmSettings
from difflab.core.types import Frame
from difflab.models.exl3vl import Exl3VL


def spec(checkpoint: str, chatting: bool = False) -> models.ModelSpec:
    """`chatting` is the chat's load, on its own longer, quantized cache."""
    pinned = Checkpoint.parse(checkpoint)
    return models.ModelSpec(
        name=f"vlm:{pinned.repo}" + (":chat" if chatting else ""),
        build=lambda: Exl3VL(pinned, config.CHAT_CONTEXT_TOKENS, config.CHAT_KV_BITS) if chatting else Exl3VL(pinned),
        # Dropped from the page cache on eviction, since H3 maps 25 + 18 GB of a 58 GB machine right
        # after. ~17 GB on the card, so it is heavy: evicted, not parked in host RAM, when anything
        # else heavy is acquired.
        checkpoints=(pinned.path,),  # a folder: every .safetensors in it
        unload=lambda vlm: vlm.unload(),
    )


def evict() -> None:
    """Drop whichever VLM is resident: 17 GB the next model would rather have."""
    for name in models.registry.resident():
        if name.startswith("vlm:"):
            models.registry.evict(name)


def _vlm(settings: VlmSettings, chatting: bool = False) -> Exl3VL:
    model: Exl3VL = models.registry.acquire(spec(settings.checkpoint, chatting))
    return model


def ask(frame: Frame, question: str, title: str = "Qwen3.8", settings: VlmSettings | None = None) -> str:
    """Anything you want to know about the picture. The answer streams into a panel."""
    settings = settings or VlmSettings()
    return _vlm(settings).ask(frame.pil(), question, title=title, **settings.generation())
