"""LTX-2.5's per-call knobs. Beside the model rather than in `core`, which imports no model."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from difflab.core import hub
from difflab.core.settings import Settings, lora_stack, setting
from difflab.models.ltx25 import config

# Every file below matches Lightricks/LTX-2.5@49b093a by sha256, with no licence click to fetch it.
PACK = "deAPI-ai/ltx2-5-22b-dist-int8@a4d65f15437f75c2ccbba4ff42932d0386fcdfe2"
# Fields a 2.5 model never reads: left at their defaults they stay out of the key, so a 2.5 clip keeps its key.
_V23_ONLY = ("text_encoder_23", "loras_23", "msr_23", "open_on_photo_23", "sigmas")


@dataclass(frozen=True)
class LtxSettings(Settings):
    """LTX-2.5 or 2.3, as the model file says, at CFG 1."""

    checkpoint: str = setting(
        f"{PACK}:diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "model",
        "an LTX-2.5 or 2.3 DiT in int8, 4-bit or fp8: repo@commit:path or civitai:…",
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
    # The abliterated Gemma 3 10Eros ships with, as 10Eros's own workflows load it.
    text_encoder_23: str = setting(
        "TenStrip/LTX2.3-10Eros@84a05a13610d78dbe4340d1be23fd8185e10f697"
        ":text_encoders/gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors",
        "2.3 text encoder",
        "Gemma 3 12B for an LTX-2.3 model: repo@commit:path",
        advanced=True,
        role="ltx23-text-encoder",
    )
    loras_23: str = setting(
        "",
        "2.3 LoRAs",
        "stacked on an LTX-2.3 model for every clip, as reference*strength: a DMD LoRA goes here",
        role="ltx23-loras",
        multiple=True,
    )
    # Applied with references only: its pseudo-video is what places them.
    msr_23: str = setting(
        "LiconStudio/LTX-2.3-Multiple-Subject-Reference@593b0b7d2b912e8ecdb2825a34732cee36e720ba"
        ":LTX-2.3-Licon-MSR-V2.safetensors",
        "2.3 MSR",
        "the LoRA that places up to four references on an LTX-2.3 model: repo@commit:path",
        advanced=True,
        role="ltx23-msr",
    )
    # Licon trained the pseudo-video against no opening frame: one pinned at frame 0 beside it cut away after
    # the first frame (2026-10-24, a DMD-distilled 2.3, one subject).
    open_on_photo_23: bool = setting(
        False,
        "2.3 MSR opens on the photo",
        "with references on 2.3, the first photo is the opening frame as well as the scene; off, it is the scene alone",
        advanced=True,
    )
    sigmas: str = setting(
        "",
        "schedule",
        "the noise levels sampled through, from 1 down to 0; empty is the distilled eight steps",
        advanced=True,
    )
    video_vae: str = setting(
        f"{PACK}:vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "video VAE",
        "the conv VAE, when the model file carries none of its own: repo@commit:path",
        advanced=True,
        role="ltx-video-vae",
    )
    audio_vae: str = setting(
        f"{PACK}:vae/ltx-2.5-audio-vae-bf16.safetensors",
        "audio VAE",
        "the audio VAE and its vocoder, when the model file carries none: repo@commit:path",
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

    def stack(self, references: bool) -> list[tuple[str, float]]:
        """What sampling stacks on the DiT: 2.5's only for references, whose slot embeddings place them; 2.3's
        always, and its MSR for references."""
        if self.version() == config.V23:
            return lora_stack(self.loras_23) + ([(self.msr_23, 1.0)] if references and self.msr_23 else [])
        return lora_stack(self.loras) if references else []

    def version(self) -> str:
        """From the file's header: its own `model_version`, else the keyframe marker only 2.5 has."""
        found = hub.header(hub.parse_file(self.checkpoint))
        if found is None:
            return config.V25
        stated = found.metadata.get("model_version", "")
        if stated.startswith(config.V23):
            return config.V23
        if stated.startswith(config.V25) or "model.diffusion_model.keyframes_abs_pos_embedding" in found.tensors:
            return config.V25
        if stated:
            raise ValueError(f"{self.checkpoint}: LTX-{stated}, and this pipeline runs 2.5 and 2.3")
        return config.V23

    def bundled(self, prefix: str) -> bool:
        """Whether the model file carries its own component under `prefix`, as a ComfyUI checkpoint does."""
        found = hub.header(hub.parse_file(self.checkpoint))
        return found is not None and any(name.startswith(prefix) for name in found.tensors)

    def video_vae_source(self) -> tuple[str, str]:
        """The file the video VAE is read from, and the prefix its tensors carry there."""
        return (self.checkpoint, "vae.") if self.bundled("vae.") else (self.video_vae, "")

    def audio_vae_source(self) -> str:
        return self.checkpoint if self.bundled("audio_vae.") and self.bundled("vocoder.") else self.audio_vae

    def schedule(self) -> tuple[float, ...]:
        if not self.sigmas.strip():
            return config.SIGMAS
        chosen = tuple(float(part) for part in self.sigmas.replace(",", " ").split())
        if len(chosen) < 2 or chosen[-1] != 0 or any(b >= a for a, b in itertools.pairwise(chosen)):
            raise ValueError(f"schedule {self.sigmas!r}: falling noise levels ending at 0")
        return chosen

    def key_part(self) -> dict[str, Any]:
        part = super().key_part()
        part.pop("loras")
        part.pop("loras_23")
        if stack := lora_stack(self.loras):
            part["loras"] = stack
        if stack := lora_stack(self.loras_23):
            part["loras_23"] = stack
        defaults = LtxSettings()
        for name in _V23_ONLY:
            if name in part and getattr(self, name) == getattr(defaults, name):
                part.pop(name)
        return part
