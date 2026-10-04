"""MiniMax H3's per-call knobs. Beside the model rather than in `core`, which imports no model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from difflab.core.settings import Settings, VlmSettings, setting

# How a ref2va reference is sized, as `pipeline.reference_size` reads it.
ReferenceSize = Literal["match", "max"]
Sampler = Literal["er_sde", "euler"]
Schedule = Literal["beta57", "shifted"]

# ComfyUI's repackaging of MiniMax-H3: the pruned int8 DiTs, their turbo LoRAs and the two VAEs.
COMFY = "Comfy-Org/MiniMax-H3@7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1"


# A community merge of grafts over H3, both tasks in one file, with a turbo delta fused in: pair it with
# `lora=""`, never with the turbo LoRA as well.
HYBRID = (
    "Comfy-Org/MiniMax-H3@7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1:diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
)


@dataclass(frozen=True)
class VideoSettings(Settings):
    """MiniMax H3. The dev defaults are 2 s at 0.4 MP (~25 s of sampling); a full render is 15 s at
    0.8 MP (~15 min)."""

    # A production's chunk is ref2va (references, and an fl2va first frame beside them), so it renders on
    # Comfy's pruned ref2va with larryvrh's turbo LoRA: the `ref2va_4step` preset. `HYBRID` was the
    # default until 2026-10-04 and serves both tasks, but its TURBO file has a turbo delta baked in, so the
    # turbo LoRA below was applied twice; production 9 grew blotches in flat areas within every chunk.
    checkpoint: str = setting(
        f"{COMFY}:diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "model",
        "the int8 DiT: repo@commit:path",
        role="video-model",
    )
    lora: str = setting(
        "larryvrh/MiniMax-H3-Turbo-Lora@43a74557ac3f6539db8e0f2a959d03feb7a81480:minimax_h3_turbo_v4_step600_ema.safetensors",
        "turbo LoRA",
        "repo@commit:path",
        role="video-lora",
    )
    loras: str = setting(
        "",
        "LoRAs",
        "stacked on the video model with the turbo LoRA",
        role="video-loras",
        multiple=True,
    )
    # Qwen3-VL-32B truncated to 50 layers, int8 ConvRot (26 GB).
    text_encoder: str = setting(
        "Comfy-Org/MiniMax-H3@7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1"
        ":text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "text encoder",
        "the int8 Qwen3-VL: repo@commit:path",
        advanced=True,
        role="video-text-encoder",
    )
    # int8 ConvRot decoder linears (see vae.py): the decode is compute-bound, and int8 runs at 4x fp16 here.
    video_vae: str = setting(
        f"{COMFY}:vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "video VAE",
        "repo@commit:path",
        advanced=True,
        role="video-vae",
    )
    audio_vae: str = setting(
        f"{COMFY}:vae/minimax_h3_audio_vae_fp32.safetensors",
        "audio VAE",
        "repo@commit:path",
        advanced=True,
        role="audio-vae",
    )
    duration: float = setting(2.0, "duration", "seconds per clip, or per chunk of a take", unit="s", low=0.2)
    # ImageScaleToTotalPixels(megapixels, nearest-exact, resolution_steps=32)
    megapixels: float = setting(
        0.4, "canvas", "0.4 is the fast one, 0.98 the benchmarked one", unit="MP", low=0.1, high=2.5
    )
    seed: int = setting(336100010134930, "seed", "a take hangs off this; chunk k uses seed + k", advanced=True)
    steps: int = setting(
        5, "steps", "4-6 with er_sde and beta57; the LoRA was trained for 4", low=1, high=50, advanced=True
    )
    reference_size: ReferenceSize = setting(
        "match",
        "reference size",
        "max holds identity harder and is markedly slower",
        choices=("match", "max"),
        advanced=True,
    )
    sampler: Sampler = setting(
        "er_sde",
        "sampler",
        "er_sde adds fresh noise each step; euler is the reference's",
        choices=("er_sde", "euler"),
        advanced=True,
    )
    schedule: Schedule = setting(
        "beta57",
        "schedule",
        "beta57 spends more steps at both ends; shifted is the reference's linear grid",
        choices=("beta57", "shifted"),
        advanced=True,
    )
    shift_video: float = setting(12.0, "video shift", "flow shift of the video schedule", low=0, advanced=True)
    shift_audio: float = setting(3.0, "audio shift", "flow shift of the audio schedule", low=0, advanced=True)
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
    # taeh3, ComfyUI's 9 MB approximation of H3's VAE (Kijai's 2D build).
    preview_decoder: str = setting(
        "t8star/Taeh3-Comfy@d2d91ae1882a05d98645c912d60441a282cfdb48:models/vae_approx/taeh3_2d_kijai.safetensors",
        "preview decoder",
        "repo@commit:path",
        keyed=False,
        advanced=True,
        role="preview-decoder",
    )

    def stack(self) -> list[tuple[str, float]]:
        """`loras` as (reference, strength), without the ones at strength 0."""
        out = []
        for line in self.loras.splitlines():
            reference, _, strength = line.strip().partition("*")
            if reference and (value := float(strength or 1.0)) != 0.0:
                out.append((reference.strip(), value))
        return out

    def key_part(self) -> dict[str, Any]:
        # Absent while empty, so the clips rendered before `loras` existed keep their keys.
        part = super().key_part()
        part.pop("loras")
        if stack := self.stack():
            part["loras"] = stack
        return part

    @classmethod
    def full_length(cls, **changes: Any) -> VideoSettings:
        """15 s at 0.8 MP in 4 steps: the geometry the reference workflow rendered at."""
        return cls(duration=15.0, megapixels=0.8, steps=4).with_(**changes)


@dataclass(frozen=True)
class VideoPromptSettings(VlmSettings):
    """The VLM writing an H3 caption, as the video graph's AILab_QwenVL_Advanced asked it."""

    max_new_tokens: int = setting(4096, "max new tokens", "a cap, not a target", low=1, advanced=True)
    temperature: float = setting(0.6, "temperature", "lower repeats, higher invents", low=0, high=2, advanced=True)
    top_p: float = setting(
        0.9, "top p", "the graph's; Qwen's own is 0.8 with temperature 0.7", low=0, high=1, advanced=True
    )
    # The node leaves top_k unset, so Qwen's generation_config default of 20 applies.
    top_k: int = setting(20, "top k", "rarely worth moving", low=0, advanced=True)
    seed: int = setting(-1, "seed", "-1 draws a new one each call; a fixed one repeats the words", advanced=True)


@dataclass(frozen=True)
class EnlargeLatentsSettings(Settings):
    """The latent-space upscaler: enlarges before the decode, so H3's own VAE renders the extra detail."""

    # Its weights, and the config.json beside them that holds the normalisation it was trained under.
    checkpoint: str = setting(
        "LBH-123-AI/Minimax_h3_latent_Upscaler@3f941d5d182014dd5c0a5e16330420ee2d4aa0c6"
        ":minimax_h3_latent_upscaler_3d_conv_v1/config.json"
        ",minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
        "model",
        "repo@commit:config.json,weights.safetensors",
        advanced=True,
        role="latent-upscaler",
    )
