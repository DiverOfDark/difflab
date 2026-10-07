"""SeedVR2's "lab" colour correction: the restoration's detail over the input's colour.

A wavelet split keeps the restoration's high frequencies on the input's low ones, then the a* and b*
histograms are matched to the input's outright and L* a fifth of the way. The restorer drifts colour on
its own, and without this a chapter's 4K finish would not grade like its native join.
"""

from __future__ import annotations

import torch
from torch.nn import functional

LUMINANCE_KEPT = 0.8
_RGB_TO_XYZ = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
_XYZ_TO_RGB = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
]
_WHITE = (0.95047, 1.0, 1.08883)
_EPSILON = 6.0 / 29.0
_KAPPA = (29.0 / 3.0) ** 3


def _blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    radius = min(radius, max(1, min(image.shape[-2:]) // 8))
    kernel = torch.tensor([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    kernel = kernel.to(image)[None, None].repeat(image.shape[1], 1, 1, 1)
    padded = functional.pad(image, (radius, radius, radius, radius), mode="replicate")
    return functional.conv2d(padded, kernel, groups=image.shape[1], dilation=radius)


def _split(image: torch.Tensor, levels: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    high = torch.zeros_like(image)
    low = image
    for level in range(levels):
        blurred = _blur(low, 2**level)
        high += low - blurred
        low = blurred
    return high, low


def _to_lab(rgb: torch.Tensor) -> torch.Tensor:
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = torch.einsum("bchw,dc->bdhw", linear, torch.tensor(_RGB_TO_XYZ, device=rgb.device))
    xyz = xyz / torch.tensor(_WHITE, device=rgb.device).view(1, 3, 1, 1)
    f = torch.where(xyz > _EPSILON**3, xyz.clamp_min(0) ** (1 / 3), (xyz * _KAPPA + 16) / 116)
    return torch.stack((116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])), dim=1)


def _from_lab(lab: torch.Tensor) -> torch.Tensor:
    fy = (lab[:, 0] + 16) / 116
    f = torch.stack((lab[:, 1] / 500 + fy, fy, fy - lab[:, 2] / 200), dim=1)
    xyz = torch.where(f > _EPSILON, f**3, (f * 116 - 16) / _KAPPA)
    xyz = xyz * torch.tensor(_WHITE, device=lab.device).view(1, 3, 1, 1)
    linear = torch.einsum("bchw,dc->bdhw", xyz, torch.tensor(_XYZ_TO_RGB, device=lab.device))
    rgb = torch.where(linear > 0.0031308, linear.clamp_min(0) ** (1 / 2.4) * 1.055 - 0.055, linear * 12.92)
    return rgb.clamp(0, 1)


def _match(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """`source`'s values replaced, rank for rank, by `reference`'s: equal sizes, so no resampling."""
    order = source.flatten().argsort()
    out = torch.empty_like(source.flatten())
    out[order] = reference.flatten().sort().values
    return out.view_as(source)


def correct(restored: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """`restored` and `source` `[T, 3, H, W]` in [-1, 1], float32; returns `restored` recoloured."""
    high, _ = _split(restored)
    _, low = _split(source)
    based = (high + low).clamp(-1, 1)
    content, style = _to_lab((based + 1) / 2), _to_lab(((source + 1) / 2).clamp(0, 1))
    lightness = content[:, 0] * LUMINANCE_KEPT + _match(content[:, 0], style[:, 0]) * (1 - LUMINANCE_KEPT)
    lab = torch.stack((lightness, _match(content[:, 1], style[:, 1]), _match(content[:, 2], style[:, 2])), dim=1)
    return _from_lab(lab) * 2 - 1
