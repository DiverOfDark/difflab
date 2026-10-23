"""taeltx2_3_wide, madebyollin's tiny decoder for LTX-2.3/2.5 latents (MIT, github.com/madebyollin/taehv)."""

# The plain taeltx2_3 decodes 2.3/2.5 blurry (taehv README, issue #20); this 58 MB wide one is the author's fix.
# Unlike taeh3 it has temporal layers and draws all eight frames of every latent frame, so a preview plays at the
# clip's own 24 fps. Decoder half only; the weights are taehv.py's layout and load strictly.

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import torch
from torch import nn

from difflab.core.hub import UrlFile

WEIGHTS = UrlFile(
    "https://raw.githubusercontent.com/madebyollin/taehv/32ac0146b11007cda5a57b60a3b35653361fb8a4/taeltx2_3_wide.pth",
    "007788e6b9cb7f77e8589ae30ba7456b119d38b0d017e1d349c1c1d11e3d6339",
)
PATCH = 4
TRIM = 7  # the first latent frame is one pixel frame, but the decoder grows every one into eight


class _Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class _MemBlock(nn.Module):
    """taehv's WideMemBlock: each frame sees the one before it."""

    def __init__(self, n: int) -> None:
        super().__init__()
        groups = max(1, n // 64)
        self.conv = nn.Sequential(
            nn.Conv2d(n * 2, n, 1), nn.ReLU(), _conv(n, n, groups=groups), nn.ReLU(),
            nn.Conv2d(n, n, 1), nn.ReLU(), _conv(n, n, groups=groups),
        )  # fmt: skip
        self.skip = nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
        """`before` is the frame ahead of `x[0]`: zeros at the clip's start, the last chunk's last after."""
        past = torch.cat([before, x[:-1]])
        out: torch.Tensor = self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))
        return out


class _Grow(nn.Module):
    """One frame into two."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n, n * 2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames, channels, height, width = x.shape
        grown: torch.Tensor = self.conv(x).reshape(frames * 2, channels, height, width)
        return grown


def _conv(n_in: int, n_out: int, bias: bool = True, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, bias=bias, groups=groups)


def _decoder() -> nn.Sequential:
    widths = (1024, 512, 256, 64)
    layers: list[nn.Module] = [_Clamp(), _conv(128, widths[0]), nn.ReLU()]
    for n, following in pairwise(widths):
        layers += [_MemBlock(n), _MemBlock(n), _MemBlock(n), nn.Upsample(scale_factor=2), _Grow(n)]
        layers += [_conv(n, following, bias=False)]
    layers += [nn.ReLU(), _conv(widths[-1], 3 * PATCH**2)]
    return nn.Sequential(*layers)


def load(path: str | Path, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float16) -> nn.Sequential:
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    decoder = _decoder()
    decoder.load_state_dict({k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")})
    return decoder.to(device, dtype).eval().requires_grad_(False)


@torch.inference_mode()
def decode(decoder: nn.Sequential, latents: torch.Tensor, scale: float = 1.0, chunk: int = 2) -> torch.Tensor:
    """Normalized `[1, 128, T, h, w]` latents -> `[8T-7, H, W, 3]` uint8 frames on the CPU, `scale` of full size."""
    # Decoded at the latent's own size and shrunk after: halving a 32x-compressed latent first smeared the
    # picture past reading (2026-10-24). A chunk at a time, each block carrying its last frame on, because at
    # 544x960 a whole 121-frame clip would hold 8 GB in the last layer alone.
    parameter = next(decoder.parameters())
    frames = latents[0].permute(1, 0, 2, 3)
    carried: dict[int, torch.Tensor] = {}
    out = []
    for start in range(0, len(frames), chunk):
        x = frames[start : start + chunk].to(parameter.device, parameter.dtype)
        for index, layer in enumerate(decoder):
            if isinstance(layer, _MemBlock):
                before = carried.get(index, torch.zeros_like(x[:1]))
                carried[index] = x[-1:]
                x = layer(x, before)
            else:
                x = layer(x)
        pixels = nn.functional.pixel_shuffle(x, PATCH).clamp_(0, 1)
        if scale != 1.0:
            pixels = nn.functional.interpolate(pixels, scale_factor=scale, mode="area")
        out.append(pixels.mul_(255).round_().to(torch.uint8).permute(0, 2, 3, 1).cpu())
    return torch.cat(out)[TRIM:]
