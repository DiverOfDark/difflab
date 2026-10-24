"""LTX-2.5's per-call knobs. Beside the model rather than in `core`, which imports no model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from difflab.core.settings import Settings, lora_stack, setting

# Every file below matches Lightricks/LTX-2.5@49b093a by sha256, with no licence click to fetch it.
PACK = "deAPI-ai/ltx2-5-22b-dist-int8@a4d65f15437f75c2ccbba4ff42932d0386fcdfe2"


@dataclass(frozen=True)
class LtxSettings(Settings):
    """LTX-2.5 distilled: eight steps at CFG 1."""

    checkpoint: str = setting(
        f"{PACK}:diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "model",
        "the int8 or w4a8 ConvRot DiT: repo@commit:path or civitai:…",
        role="ltx-model",
    )
    # Applied only when a clip has references: the slot embeddings it carries are what places them.
    loras: str = setting(
        "LiconStudio/LTX-2.5-Multiple-Subject-Reference@aa19b48f845995525d8ace60cdc89b842e951ce6"
        ":LTX-2.5-Licon-MSR-V2.safetensors*1",
        "LoRAs",
        "stacked on the model in order, as reference*strength; the first with slot embeddings places the references",
        role="ltx-loras",
        multiple=True,
    )
    # Lightricks' own Gemma 4 fine-tune with igorls' heretic abliteration transplanted onto it (see that repo).
    text_encoder: str = setting(
        "diverofdark/LTX-2.5-Gemma4-12B-heretic-int8-convrot@de8674a79dfae82e13aee624352a78d83501435a"
        ":gemma4-12b-ltx-2.5-heretic-int8-convrot.safetensors",
        "text encoder",
        "the int8 Gemma 4 with its projections: repo@commit:path",
        advanced=True,
        role="ltx-text-encoder",
    )
    video_vae: str = setting(
        f"{PACK}:vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "video VAE",
        "the conv VAE: repo@commit:path",
        advanced=True,
        role="ltx-video-vae",
    )
    audio_vae: str = setting(
        f"{PACK}:vae/ltx-2.5-audio-vae-bf16.safetensors",
        "audio VAE",
        "the audio VAE and its vocoder: repo@commit:path",
        advanced=True,
        role="ltx-audio-vae",
    )
    duration: float = setting(5.0, "duration", "seconds per clip", unit="s", low=0.4, high=20)
    megapixels: float = setting(0.9, "canvas", "1280x704 is 0.86", unit="MP", low=0.1, high=2.5)
    seed: int = setting(42, "seed", "the noise every step starts from", advanced=True)
    reference_strength: float = setting(
        1.0, "reference strength", "how clean the references stay through sampling", low=0, high=1, advanced=True
    )
    previews: bool = setting(True, "previews", "each step's guess at the clip, on the page", keyed=False)
    preview_scale: float = setting(
        0.5,
        "preview size",
        "of the sampled size; half is a quarter of the work",
        low=0.1,
        high=1,
        keyed=False,
        advanced=True,
    )

    def stack(self) -> list[tuple[str, float]]:
        return lora_stack(self.loras)

    def key_part(self) -> dict[str, Any]:
        part = super().key_part()
        part.pop("loras")
        if stack := self.stack():
            part["loras"] = stack
        return part
