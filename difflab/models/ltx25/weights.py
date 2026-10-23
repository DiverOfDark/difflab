"""Lightricks' split Comfy pack into diffusers' and transformers' classes."""

# The renames are diffusers' own (`scripts/convert_ltx2_to_diffusers.py` at 8b3c707, the 2.3 tables 2.5 reuses).
# The int8 ConvRot layers keep their biases, which H3's loader refuses.

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, Gemma4UnifiedForConditionalGeneration

from difflab.models import lora
from difflab.models.convrot import ConvRotLinear, read_quant_meta
from difflab.models.ltx25 import config

DIT_PREFIX = "model.diffusion_model."
DIT_RENAMES = (
    ("patchify_proj", "proj_in"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("audio_prompt_adaln_single", "audio_prompt_adaln"),
    ("prompt_adaln_single", "prompt_adaln"),
)
CONNECTOR_RENAMES = (
    ("video_embeddings_connector", "video_connector"),
    ("audio_embeddings_connector", "audio_connector"),
    ("transformer_1d_blocks", "transformer_blocks"),
    ("text_embedding_projection.audio_aggregate_embed", "audio_text_proj_in"),
    ("text_embedding_projection.video_aggregate_embed", "video_text_proj_in"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
)
VIDEO_VAE_RENAMES = (
    ("down_blocks.0", "down_blocks.0"),
    ("down_blocks.1", "down_blocks.0.downsamplers.0"),
    ("down_blocks.2", "down_blocks.1"),
    ("down_blocks.3", "down_blocks.1.downsamplers.0"),
    ("down_blocks.4", "down_blocks.2"),
    ("down_blocks.5", "down_blocks.2.downsamplers.0"),
    ("down_blocks.6", "down_blocks.3"),
    ("down_blocks.7", "down_blocks.3.downsamplers.0"),
    ("down_blocks.8", "mid_block"),
    ("up_blocks.0", "mid_block"),
    ("up_blocks.1", "up_blocks.0.upsamplers.0"),
    ("up_blocks.2", "up_blocks.0"),
    ("up_blocks.3", "up_blocks.1.upsamplers.0"),
    ("up_blocks.4", "up_blocks.1"),
    ("up_blocks.5", "up_blocks.2.upsamplers.0"),
    ("up_blocks.6", "up_blocks.2"),
    ("last_time_embedder", "time_embedder"),
    ("last_scale_shift_table", "scale_shift_table"),
    ("res_blocks", "resnets"),
    ("per_channel_statistics.mean-of-means", "latents_mean"),
    ("per_channel_statistics.std-of-means", "latents_std"),
    ("up_blocks.7", "up_blocks.3.upsamplers.0"),
    ("up_blocks.8", "up_blocks.3"),
)
AUDIO_VAE_RENAMES = (
    ("per_channel_statistics.mean-of-means", "latents_mean"),
    ("per_channel_statistics.std-of-means", "latents_std"),
)
VOCODER_RENAMES = (
    ("resblocks", "resnets"),
    ("conv_pre", "conv_in"),
    ("conv_post", "conv_out"),
    ("act_post", "act_out"),
    ("downsample.lowpass", "downsample"),
)
SLOT_PREFIX = "diffusion_model.reference_slot_embedding."


@dataclass
class Split:
    dense: dict[str, torch.Tensor]
    quant: dict[str, tuple[torch.Tensor, torch.Tensor]]  # layer -> (int8 weight, fp32 per-row scale)


@dataclass
class SlotEmbedding:
    """MSR's learned identity for reference slot `n`: a Fourier-feature MLP into the 128 latent channels."""

    frequencies: torch.Tensor
    w0: torch.Tensor
    b0: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor

    def __call__(self, slot: int) -> torch.Tensor:
        scaled = torch.tensor([slot / 16.0])
        phases = scaled * self.frequencies.float()
        features = torch.cat((scaled, torch.sin(phases), torch.cos(phases)))
        hidden = nn.functional.silu(nn.functional.linear(features, self.w0.float(), self.b0.float()))
        return nn.functional.linear(hidden, self.w2.float(), self.b2.float())


def layout(component: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((config.LAYOUT.fetch() / component / "config.json").read_text())
    return loaded


def read(path: str | Path, keep: Callable[[str], bool] = lambda _: True) -> Split:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = [k for k in list(f.keys()) if keep(k)]  # safe_open is not a mapping
        layers = {k.removesuffix(".comfy_quant") for k in keys if k.endswith(".comfy_quant")}
        for layer in layers:
            if not read_quant_meta(f.get_tensor(f"{layer}.comfy_quant")).get("convrot"):
                raise ValueError(f"{Path(path).name}: {layer} is int8 without ConvRot, which no kernel here runs")
        quant = {layer: (f.get_tensor(f"{layer}.weight"), f.get_tensor(f"{layer}.weight_scale")) for layer in layers}
        dense = {
            k: f.get_tensor(k)
            for k in keys
            if not k.endswith((".comfy_quant", ".weight_scale")) and k.removesuffix(".weight") not in layers
        }
    return Split(dense, quant)


def _renamed(key: str, renames: Sequence[tuple[str, str]]) -> str:
    for old, new in renames:
        key = key.replace(old, new)
    return key


def place(model: nn.Module, split: Split, rename: Callable[[str], str]) -> None:
    """int8 layers swapped in with their biases, every dense tensor assigned; anything left over raises."""
    dense = {rename(k): v for k, v in split.dense.items()}
    for layer, (weight, scale) in split.quant.items():
        path = rename(layer)
        parent_path, _, name = path.rpartition(".")
        setattr(model.get_submodule(parent_path), name, ConvRotLinear(weight, scale, dense.pop(f"{path}.bias", None)))
    missing, unexpected = model.load_state_dict(dense, strict=False, assign=True)
    # The DiT ships its AdaLN tables in fp32, and diffusers' blocks mix them into bf16 activations unconverted.
    for parameter in model.parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    still_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if missing or unexpected or still_meta:
        raise RuntimeError(
            f"{type(model).__name__}: missing={sorted(missing)[:6]} unexpected={sorted(unexpected)[:6]} "
            f"meta={still_meta[:6]}"
        )


def _dit_name(key: str) -> str:
    key = _renamed(key.removeprefix(DIT_PREFIX), DIT_RENAMES)
    if key.startswith("adaln_single."):
        return "time_embed." + key.removeprefix("adaln_single.")
    if key.startswith("audio_adaln_single."):
        return "audio_time_embed." + key.removeprefix("audio_adaln_single.")
    return key


def load_dit(path: Path, loras: Sequence[tuple[Path, float]]) -> LTX2VideoTransformer3DModel:
    """On the CPU; the caller decides what streams."""
    with init_empty_weights():
        model = LTX2VideoTransformer3DModel.from_config(layout("transformer"))
    place(model, read(path, lambda k: k.startswith(DIT_PREFIX) and "_embeddings_connector." not in k), _dit_name)
    for lora_path, strength in loras:
        adapter = lora.read(lora_path, lambda k: k.removeprefix("diffusion_model."))
        if not adapter.pairs:
            raise ValueError(f"{lora_path.name} holds no LoRA pairs")
        for module, (a, b) in adapter.pairs.items():
            scale = adapter.scales[module] * strength
            lora.attach(model.get_submodule(module), a, b if scale == 1.0 else (b.float() * scale).to(b.dtype))
    return model.requires_grad_(False).eval()


def read_slots(path: Path) -> SlotEmbedding | None:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if f"{SLOT_PREFIX}frequencies" not in list(f.keys()):
            return None
        names = ("frequencies", "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias")
        return SlotEmbedding(*(f.get_tensor(SLOT_PREFIX + name) for name in names))


def load_connectors(dit: Path, text_encoder: Path) -> LTX2TextConnectors:
    with init_empty_weights():
        model = LTX2TextConnectors.from_config(layout("connectors"))
    halves = (
        read(dit, lambda k: "_embeddings_connector." in k),
        read(text_encoder, lambda k: k.startswith("text_embedding_projection.")),
    )
    split = Split(
        {k: v for h in halves for k, v in h.dense.items()}, {k: v for h in halves for k, v in h.quant.items()}
    )
    place(model, split, lambda k: _renamed(k.removeprefix(DIT_PREFIX), CONNECTOR_RENAMES))
    return model.requires_grad_(False).eval()


def _gemma_name(key: str) -> str:
    if key.startswith("model."):
        return "model.language_model." + key.removeprefix("model.")
    if key.startswith("vision_model."):
        return "model.embed_vision." + key.removeprefix("vision_model.")
    if key.startswith("multi_modal_projector."):
        return "model.embed_vision.multimodal_embedder." + key.removeprefix("multi_modal_projector.")
    if key.startswith("audio_projector."):
        return "model.embed_audio." + key.removeprefix("audio_projector.")
    return key


def load_text_encoder(path: Path) -> Gemma4UnifiedForConditionalGeneration:
    """Gemma 4 without its vocabulary head: only the hidden states of every layer are read."""
    gemma_config = AutoConfig.from_pretrained(config.LAYOUT.fetch() / "text_encoder")
    with init_empty_weights():
        model = Gemma4UnifiedForConditionalGeneration._from_config(gemma_config, attn_implementation="sdpa")
    model.lm_head = nn.Identity()
    keep = lambda k: not k.startswith(("text_embedding_projection.", "hf_asset", "tokenizer_json"))  # noqa: E731
    place(model, read(path, keep), _gemma_name)
    return model.requires_grad_(False).eval()


def load_video_vae(path: Path) -> AutoencoderKLLTX2Video:
    with init_empty_weights():
        model = AutoencoderKLLTX2Video.from_config(layout("vae"))
    split = read(path, lambda k: "per_channel_statistics.channel" not in k and "mean-of-stds" not in k)
    place(model, split, lambda k: _renamed(k, VIDEO_VAE_RENAMES))
    return model.requires_grad_(False).eval()


def load_audio_vae(path: Path) -> tuple[AutoencoderKLLTX2Audio, LTX2VocoderWithBWE]:
    with init_empty_weights():
        vae = AutoencoderKLLTX2Audio.from_config(layout("audio_vae"))
        vocoder = LTX2VocoderWithBWE.from_config(layout("vocoder"))
    place(
        vae,
        read(path, lambda k: k.startswith("audio_vae.")),
        lambda k: _renamed(k.removeprefix("audio_vae."), AUDIO_VAE_RENAMES),
    )

    def vocoder_name(key: str) -> str:
        key = _renamed(key.removeprefix("vocoder."), VOCODER_RENAMES)
        return key.replace(".ups.", ".upsamplers.") if key.endswith((".weight", ".bias")) else key

    place(vocoder, read(path, lambda k: k.startswith("vocoder.")), vocoder_name)
    return vae.requires_grad_(False).eval(), vocoder.requires_grad_(False).eval()
