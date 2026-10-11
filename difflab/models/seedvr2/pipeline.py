"""Loading SeedVR2 and restoring a run of frames with it, window by window."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.nn import functional

from difflab import config
from difflab.core import ui
from difflab.models.convrot import ConvRotLinear
from difflab.models.seedvr2 import color
from difflab.models.seedvr2.dit import NaDiT
from difflab.models.seedvr2.vae import VideoVAE

DTYPE = torch.bfloat16
# One sampling step from pure noise at the schedule's T, so the trailing timesteps and their shift are moot.
TIMESTEP = 1000.0
ALIGN = 16  # the VAE's 8x times the DiT's 2x patch


@dataclass
class Restorer:
    """Kept off the card between its phases: at 1080p the VAE alone peaks at 21.5 GiB of the 24."""

    dit: NaDiT
    text: torch.Tensor  # the checkpoint's positive conditioning; at cfg 1 the negative is never read

    def to(self, device: torch.device | str) -> Restorer:
        self.dit.to(device)
        self.text = self.text.to(device)
        return self


def load_dit(path: Path, device: str = "cpu") -> Restorer:
    with torch.device("meta"):
        model = NaDiT()
    dense: dict[str, torch.Tensor] = {}
    with ui.step("Loading SeedVR2") as detail, safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        quantized = {k.removesuffix(".comfy_quant") for k in keys if k.endswith(".comfy_quant")}
        for layer in sorted(quantized):
            parent, _, name = layer.rpartition(".")
            bias = f.get_tensor(f"{layer}.bias").to(device, DTYPE) if f"{layer}.bias" in keys else None
            linear = ConvRotLinear(
                f.get_tensor(f"{layer}.weight").to(device), f.get_tensor(f"{layer}.weight_scale").to(device), bias
            )
            setattr(model.get_submodule(parent), name, linear)
        owned = {
            f"{layer}.{suffix}" for layer in quantized for suffix in ("weight", "weight_scale", "comfy_quant", "bias")
        }
        for key in keys:
            # rotary_embedding_torch's stored frequencies are recomputed in fp32 rather than read in fp16.
            if key in owned or key.endswith("rope.rope.freqs") or key.endswith("_conditioning"):
                continue
            dense[key] = f.get_tensor(key).to(device, DTYPE)
        text = f.get_tensor("positive_conditioning").to(device, DTYPE)
        missing, unexpected = model.load_state_dict(dense, strict=False, assign=True)
        if missing or unexpected:
            raise KeyError(f"SeedVR2 checkpoint does not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
        detail(f"7B · {len(quantized)} int8 layers")
    return Restorer(model.eval().requires_grad_(False), text)


def load_vae(path: Path, device: str = config.DEVICE) -> VideoVAE:
    with ui.step("Loading SeedVR2 VAE") as detail:
        vae = VideoVAE()
        vae.load_state_dict(load_file(str(path)))
        detail("8x spatial, 4x temporal, 16 channels")
    return vae.to(device, torch.float16).channels_last().eval().requires_grad_(False)


def _padded(frames: torch.Tensor) -> torch.Tensor:
    """Up to 4n+1 frames with the clip's own end run backwards, as the reference pads them."""
    n = len(frames)
    count = 0 if n % 4 == 1 else ((n - 1) // 4 + 1) * 4 + 1 - n
    if not count:
        return frames
    if count >= n:
        return torch.cat([frames, frames[-1:].expand(count, *frames.shape[1:])])
    return torch.cat([frames, frames[-count - 1 : -1].flip(0)])


def _prepared(frames: torch.Tensor, device: str) -> torch.Tensor:
    """`[n, H, W, 3]` in [0, 1] -> `[1, 3, 4n+1, H', W']` in [-1, 1], H' and W' up to the next 16."""
    _, height, width, _ = frames.shape
    x = _padded(frames).to(device, DTYPE).permute(3, 0, 1, 2)[None] * 2 - 1
    return functional.pad(x, (0, (-width) % ALIGN, 0, (-height) % ALIGN))


def starts(frames: int, window: int, overlap: int) -> list[int]:
    out, step = [0], max(1, window - overlap)
    while out[-1] + window < frames:
        out.append(out[-1] + step)
    return out


@torch.inference_mode()
def restore(
    frames: torch.Tensor,
    restorer: Restorer,
    vae: VideoVAE,
    window: int,
    overlap: int,
    seed: int,
    on_step: Callable[[], None] | None = None,
    device: str = config.DEVICE,
) -> Iterator[torch.Tensor]:
    """`frames [T, H, W, 3]` in [0, 1] restored at the same size, `window` frames at a time.

    Every window is encoded, then the DiT goes onto the card for all of them, then every window is
    decoded: the reference's phases, which keep the 8 GB DiT and the VAE's 15 GB peak apart. Neighbours
    share `overlap` frames, cross-faded on a Hann ramp across their middle third as the reference blends
    its batches. `on_step` is called three times a window.
    """
    _, height, width, _ = frames.shape
    beginnings = starts(len(frames), window, overlap)
    latents = []
    for start in beginnings:
        latents.append(vae.encode(_prepared(frames[start : start + window], device))[0].permute(1, 2, 3, 0))
        if on_step is not None:
            on_step()
    torch.cuda.empty_cache()
    restorer.to(device)
    cleaned = []
    try:
        for latent in latents:
            generator = torch.Generator(device).manual_seed(seed)
            noise = torch.randn(latent.shape, generator=generator, device=device, dtype=DTYPE)
            mask = torch.ones_like(latent[..., :1])
            velocity = restorer.dit(torch.cat([noise, latent, mask], dim=-1), restorer.text, TIMESTEP)
            cleaned.append((noise - velocity).permute(3, 0, 1, 2)[None])
            if on_step is not None:
                on_step()
    finally:
        restorer.to("cpu")
        torch.cuda.empty_cache()
    held: torch.Tensor | None = None
    for index, (start, clean) in enumerate(zip(beginnings, cleaned, strict=True)):
        chunk = frames[start : start + window]
        n = len(chunk)
        restored = vae.decode(clean)[0][:, :n, :height, :width].transpose(0, 1).float()
        source = chunk.to(device).permute(0, 3, 1, 2).float() * 2 - 1
        out = ((color.correct(restored, source) + 1) / 2).permute(0, 2, 3, 1).cpu()
        if held is not None:
            out = torch.cat([_blend(held, out[: len(held)]), out[len(held) :]])
        held = None
        if index < len(beginnings) - 1 and overlap:
            held, out = out[-overlap:], out[:-overlap]
        if on_step is not None:
            on_step()
        yield out


def _blend(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    count = len(previous)
    t = torch.linspace(0.0, 1.0, count)
    weight = 0.5 + 0.5 * torch.cos(torch.pi * ((t - 1 / 3) * 3).clamp(0, 1)) if count >= 3 else 1 - t
    weight = weight.view(-1, 1, 1, 1)
    return previous * weight + current * (1 - weight)
