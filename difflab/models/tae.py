"""A tiny decoder for MiniMax H3 latents, for previews: 9 MB against the real VAE's 2.6 GB.

`taeh3` ships with ComfyUI (`models/vae_approx/taeh3.safetensors`) and is what its sampler previews use.
A TAESD-style decoder - plain 2D convolutions and four 2x upsamples - with no temporal layers, so it
gives one image per latent frame, a quarter of the frames the VAE would, which plays as 6 fps at the
clip's real duration.

It takes latents in the space the DiT samples in (the VAE's normalized latents), so a sampler's denoised
estimate goes straight in. The weights are TAESD's layout exactly, and load strictly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn


class _Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class _Block(nn.Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(_conv(n_in, n_out), nn.ReLU(), _conv(n_out, n_out), nn.ReLU(), _conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fuse(self.conv(x) + self.skip(x))
        return out


def _conv(n_in: int, n_out: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, bias=bias)


def _decoder(latent_channels: int = 24) -> nn.Sequential:
    def up() -> nn.Upsample:
        return nn.Upsample(scale_factor=2)

    return nn.Sequential(
        _Clamp(), _conv(latent_channels, 96), nn.ReLU(),
        _Block(96, 96), _Block(96, 96), _Block(96, 96), up(), _conv(96, 96, bias=False),
        _Block(96, 96), _Block(96, 96), _Block(96, 96), up(), _conv(96, 96, bias=False),
        _Block(96, 64), _Block(64, 64), _Block(64, 64), up(), _conv(64, 64, bias=False),
        _Block(64, 64), _Block(64, 64), up(), _conv(64, 64, bias=False),
        _Block(64, 64), _conv(64, 3),
    )  # fmt: skip


def load(path: str | Path, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float16) -> nn.Sequential:
    decoder = _decoder()
    decoder.load_state_dict(load_file(str(path)), strict=True)
    return decoder.to(device, dtype).eval().requires_grad_(False)


@torch.inference_mode()
def decode(decoder: nn.Sequential, latents: torch.Tensor, scale: float = 1.0, batch: int = 8) -> torch.Tensor:
    """`[1, 24, T, h, w]` latents -> `[T, H, W, 3]` uint8 frames on the CPU, one per latent frame.

    `scale` shrinks the latent first: at 0.5 the preview is a quarter of the work, which is plenty to
    judge a take by.
    """
    parameter = next(decoder.parameters())
    frames = latents[0].permute(1, 0, 2, 3)  # [T, 24, h, w]
    out = []
    for start in range(0, len(frames), batch):
        piece = frames[start : start + batch].to(parameter.device, parameter.dtype)
        if scale != 1.0:
            piece = nn.functional.interpolate(piece, scale_factor=scale, mode="bilinear", align_corners=False)
        pixels = decoder(piece).clamp_(0, 1).mul_(255).round_().to(torch.uint8)
        out.append(pixels.permute(0, 2, 3, 1).cpu())
    return torch.cat(out)
