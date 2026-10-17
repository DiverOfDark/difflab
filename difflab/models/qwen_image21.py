"""Qwen-Image 2.1 on ComfyUI's int8 ConvRot repacks: diffusers' pipeline, our loaders, LoRAs in activation space.

The transformer is 7.3 GB of int8 and stays on the card. The Qwen3-VL text encoder runs once per edit,
so its 36 layers stream from RAM like H3's. Processor, scheduler config and VAE come from the official
repo, which is diffusers-shaped; its bf16 transformer and text encoder (14 + 17.5 GB) are never fetched.

Comfy fuses each block's SwiGLU input as `img_mlp.gate_up` = [gate; up], measured against the official
bf16 halves (cosine 0.99997 each way). Its rows carry their own scales, so splitting it into diffusers'
`gate_layer` and `proj` is exact.
"""

import gc
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

from difflab import config, prompts
from difflab.core import ui
from difflab.core.hub import Checkpoint, ModelFile
from difflab.models import lora, offload
from difflab.models.convrot import dequantize, read_quant_meta
from difflab.models.minimax_h3.weights import QuantSpec, swap_quantized_linears
from difflab.models.processors import ImageOnlyQwen3VLProcessor

OFFICIAL = Checkpoint(
    "Qwen/Qwen-Image-2.1",
    "d26bb61231c349cf6b7896fa83353113880e1ba3",
    ("processor/*", "scheduler/*", "vae/*", "transformer/config.json", "text_encoder/config.json"),
)
# Qwen-Image-2.1-Turbo's model_index.json `sample_sigmas`, on a scheduler with no shift and no terminal stretch.
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
TURBO_SCHEDULER = {"use_dynamic_shifting": False, "shift_terminal": None}

_GATE_UP = ".img_mlp.gate_up"
_TEXT_RENAMES = (
    ("model.embed_tokens", "model.language_model.embed_tokens"),
    ("model.layers.", "model.language_model.layers."),
    ("model.norm.", "model.language_model.norm."),
)
_LORA_PREFIXES = ("diffusion_model.", "transformer.", "base_model.model.", "lora_unet_")


def _read(path: str) -> tuple[dict[str, torch.Tensor], dict[str, QuantSpec]]:
    """Dense tensors and ConvRot layers of a Comfy file."""
    dense: dict[str, torch.Tensor] = {}
    quant: dict[str, QuantSpec] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())  # safe_open is not a mapping
        layers = sorted(k.removesuffix(".comfy_quant") for k in keys if k.endswith(".comfy_quant"))
        for layer in layers:
            meta = read_quant_meta(f.get_tensor(f"{layer}.comfy_quant"))
            quant[layer] = QuantSpec(f.get_tensor(f"{layer}.weight"), f.get_tensor(f"{layer}.weight_scale"), meta)
        owned = {f"{layer}.{leaf}" for layer in layers for leaf in ("comfy_quant", "weight", "weight_scale")}
        for key in keys:
            # The Turbo repack marks blocks 1-31 `{"attention": "comfy_kitchen_int8"}`: a hint to ComfyUI to
            # quantize their attention, not a weight. Attention here stays bf16 SDPA.
            if key not in owned and not key.endswith(".comfy_attention.config"):
                dense[key] = f.get_tensor(key)
    return dense, quant


def load_transformer(path: str, loras: Sequence[tuple[str, float]]) -> QwenImage21Transformer2DModel:
    with ui.step("Loading Qwen-Image 2.1 transformer (int8)") as detail:
        dense, quant = _read(path)
        for layer in [name for name in quant if name.endswith(_GATE_UP)]:
            spec = quant.pop(layer)
            block = layer.removesuffix(_GATE_UP)
            halves = zip(spec.weight.chunk(2), spec.scale.chunk(2), strict=True)
            for name, (weight, scale) in zip(("gate_layer", "proj"), halves, strict=True):
                quant[f"{block}.img_mlp.{name}"] = QuantSpec(weight.contiguous(), scale.contiguous(), spec.meta)
        with init_empty_weights():
            model = QwenImage21Transformer2DModel.from_config(
                QwenImage21Transformer2DModel.load_config(OFFICIAL.path / "transformer")
            )
        swap_quantized_linears(model, quant)
        missing, unexpected = model.load_state_dict(dense, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"transformer: missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]}")
        sites = [_apply_lora(model, file, strength) for file, strength in loras]
        detail(" · ".join([f"{len(quant)} int8 layers", *sites]))
    frozen: QwenImage21Transformer2DModel = model.requires_grad_(False).eval()
    return frozen


def _apply_lora(model: nn.Module, file: str, strength: float) -> str:
    """Diffusers', PEFT's, Comfy's (fused `gate_up`) and kohya's (underscored) module names."""
    underscored: dict[str, str] = {name.replace(".", "_"): name for name, _ in model.named_modules() if name}

    def rename(key: str) -> str:
        for prefix in _LORA_PREFIXES:
            key = key.removeprefix(prefix)
        return underscored.get(key, key)

    adapter = lora.read(file, rename)
    name = file.rsplit("/", 1)[-1]
    if not adapter.pairs:
        raise ValueError(
            f"{name} holds no LoRA pairs: a whole model (a fine-tune is ~7 GB as int8 ConvRot, 14 GB as fp16) "
            "goes in the Qwen-Image 2.1 model field, not among the LoRAs"
        )
    applied = 0
    for module, (a, b) in adapter.pairs.items():
        scale = adapter.scales[module] * strength
        if scale != 1.0:
            b = (b.float() * scale).to(b.dtype)
        if module.endswith(_GATE_UP):
            block = module.removesuffix(_GATE_UP)
            for name, half in zip(("gate_layer", "proj"), b.chunk(2), strict=True):
                lora.attach(model.get_submodule(f"{block}.img_mlp.{name}"), a, half.contiguous())
            applied += 1
            continue
        try:
            owner = model.get_submodule(module)
        except AttributeError:
            continue
        lora.attach(owner, a, b)
        applied += 1
    if not applied:
        raise ValueError(f"{name} matches no layer of the Qwen-Image 2.1 transformer: a LoRA for another model?")
    skipped = len(adapter.pairs) - applied
    return f"{name} {applied}" + (f" ({skipped} unmatched)" if skipped else "")


def load_text_encoder(path: str) -> Qwen3VLForConditionalGeneration:
    with ui.step("Loading Qwen3-VL 8B text encoder (int8)") as detail:
        dense, quant = _read(path)
        quant.pop("lm_head")  # the pipeline reads hidden states only
        dense = {_rename_text(key): tensor for key, tensor in dense.items()}
        embed = quant.pop("model.embed_tokens")
        dense["model.language_model.embed_tokens.weight"] = dequantize(embed.weight, embed.scale)
        quant = {_rename_text(layer): spec for layer, spec in quant.items()}
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration._from_config(
                Qwen3VLConfig.from_pretrained(OFFICIAL.path / "text_encoder"), attn_implementation="sdpa"
            )
        model.lm_head = nn.Identity()
        swap_quantized_linears(model, quant)
        missing, unexpected = model.load_state_dict(dense, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"text encoder: missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]}")
        detail(f"{len(quant)} int8 layers")
    frozen: Qwen3VLForConditionalGeneration = model.requires_grad_(False).eval()
    return frozen


def load_vae(folder: Path, path: str) -> AutoencoderKLQwenImage21:
    """Qwen's own VAE config, with the weights from `path`."""
    vae = AutoencoderKLQwenImage21.from_config(AutoencoderKLQwenImage21.load_config(folder))
    found = load_file(path)
    if set(found) != set(vae.state_dict()):
        raise ValueError(
            f"{Path(path).name} is not a Qwen-Image 2.1 VAE in diffusers' layout ({len(found)} tensors, "
            f"{len(set(found) - set(vae.state_dict()))} of them unknown): ComfyUI's repack is laid out otherwise"
        )
    vae.load_state_dict(found)
    return cast(AutoencoderKLQwenImage21, vae.to(torch.bfloat16).eval())


def load_processor() -> ImageOnlyQwen3VLProcessor:
    """The official processor's settings on the PIL image processor: its own is the torchvision one."""
    folder = OFFICIAL.path / "processor"
    image = cast(dict[str, Any], Qwen2VLImageProcessorPil.get_image_processor_dict(folder)[0])
    keep = ("size", "patch_size", "temporal_patch_size", "merge_size", "image_mean", "image_std", "resample")
    return ImageOnlyQwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorPil(**{k: image[k] for k in keep}),
        tokenizer=Qwen2Tokenizer.from_pretrained(folder, local_files_only=True),
        video_processor=None,
        chat_template=(folder / "chat_template.jinja").read_text(),
    )


def _rename_text(key: str) -> str:
    for source, target in _TEXT_RENAMES:
        if key.startswith(source):
            return target + key[len(source) :]
    return key


def _decode(pipe: QwenImage21Pipeline, latents: torch.Tensor, height: int, width: int) -> Image.Image:
    """The pipeline's own decode, at the size it floored to."""
    align = pipe.vae_scale_factor * 2
    latents = pipe._unpack_latents(latents, height // align * align, width // align * align, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    shape = (1, pipe.vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(shape).to(latents)
    std = torch.tensor(pipe.vae.config.latents_std).view(shape).to(latents)
    with torch.no_grad():
        pixels = pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
    decoded: Image.Image = pipe.image_processor.postprocess(pixels, output_type="pil")[0]
    return decoded


class Editor21:
    """Qwen-Image 2.1, one transformer file and one LoRA stack."""

    def __init__(
        self,
        weights: ModelFile,
        text_encoder: ModelFile,
        vae: ModelFile,
        loras: Sequence[tuple[ModelFile, float]] = (),
        device: str = config.DEVICE,
    ) -> None:
        self.weights = weights
        self.text_encoder = text_encoder
        self.vae = vae
        self.loras = loras
        self.device = device
        self._pipe: QwenImage21Pipeline | None = None

    @property
    def pipe(self) -> QwenImage21Pipeline:
        if self._pipe is None:
            self._pipe = self._build()
        return self._pipe

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        steps: int,
        cfg: float,
        seed: int,
        negative_prompt: str = prompts.NEGATIVE_PROMPT,
        donors: Sequence[Image.Image] = (),
        turbo: bool = True,
    ) -> Image.Image:
        """`image` is already at the target size, and every input is encoded at that area."""
        return self._draw([image, *donors], prompt, image.size, steps, cfg, seed, negative_prompt, turbo)

    def create(
        self,
        prompt: str,
        size: tuple[int, int],
        steps: int,
        cfg: float,
        seed: int,
        negative_prompt: str = prompts.NEGATIVE_PROMPT,
        turbo: bool = True,
        references: Sequence[Image.Image] = (),
    ) -> Image.Image:
        """A picture on a fresh canvas, `size` on the /32 grid, with any `references` as its condition images."""
        return self._draw(list(references) or None, prompt, size, steps, cfg, seed, negative_prompt, turbo)

    def _draw(
        self,
        images: list[Image.Image] | None,
        prompt: str,
        size: tuple[int, int],
        steps: int,
        cfg: float,
        seed: int,
        negative_prompt: str,
        turbo: bool,
    ) -> Image.Image:
        pipe = self.pipe
        width, height = size
        base = cast(dict[str, object], FlowMatchEulerDiscreteScheduler.load_config(OFFICIAL.path / "scheduler"))
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config({**base, **(TURBO_SCHEDULER if turbo else {})})
        sigmas = TURBO_SIGMAS if turbo else None
        steps = len(TURBO_SIGMAS) if turbo else steps
        with ui.EditProgress(pipe.transformer, steps) as progress:
            result = pipe(
                image=images,
                prompt=prompt,
                negative_prompt=negative_prompt if cfg > 1 else None,
                true_cfg_scale=cfg,
                num_inference_steps=steps,
                sigmas=sigmas,
                height=height,
                width=width,
                # The condition images are resized to this area; the target's keeps them 1:1 with the output.
                output_resolution=round(math.sqrt(width * height)),
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=progress.on_step_end,
                # Decoded below instead: inside the call the step-0 prefix KV cache is still held, and a 2 MP
                # decode then asked for 2.13 GB more with 21.6 GB in use. Tiling fits but seams the sky every
                # 192 px, its tile stride (2026-10-10).
                output_type="latent",
            )
            latents = result.images
            del result
            torch.cuda.empty_cache()
            return _decode(pipe, latents, height, width)

    def unload(self) -> None:
        self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def _build(self) -> QwenImage21Pipeline:
        official = OFFICIAL.fetch()
        weights, text_file = self.weights.fetch(), self.text_encoder.fetch()
        loras = [(str(source.fetch()), strength) for source, strength in self.loras]
        device = torch.device(self.device)

        transformer = load_transformer(str(weights), loras).to(device)
        text_encoder = load_text_encoder(str(text_file))
        inner = cast(nn.Module, text_encoder.model)
        language_model = cast(nn.Module, inner.language_model)
        offload.stream_blocks(cast(nn.ModuleList, language_model.layers), device)
        for name, child in language_model.named_children():
            if name != "layers":
                child.to(device)
        cast(nn.Module, inner.visual).to(device)
        vae = load_vae(official / "vae", str(self.vae.fetch())).to(device)

        pipe = QwenImage21Pipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(official / "scheduler"),
            vae=vae,
            text_encoder=text_encoder,
            processor=load_processor(),
            transformer=transformer,
        )
        pipe.set_progress_bar_config(disable=True)  # difflab.ui draws the progress
        return pipe
