"""RealESRGAN's RRDBNet, for upscaling a finished clip.

Sampling cost is quadratic in the packed sequence and so roughly quadratic in pixel count, while this
network is linear in it - so generating small and enlarging afterwards buys resolution, and only
sometimes time. Measured on H3's own output, reconstructing a native 0.79 MP render from a half-size
copy:

    upscaler            PSNR   SSIM   sharpness   shimmer   s/frame
    bicubic            32.95  0.957      0.126     0.727      -
    RealESRGAN x2      31.04  0.946      0.509     1.014    0.137
    4xLSDIR (x4,down)  35.62  0.977      0.560     0.988    0.633
    4xNMKDSuperscale   32.95  0.962      1.113     1.705    0.633

`sharpness` is Laplacian variance against the ground truth's, `shimmer` the frame-to-frame difference
against its own. 4xLSDIR wins outright: downscaling a 4x result supersamples, adding detail and averaging
per-frame noise away at once. Shimmer is a property of the training rather than the factor - 4xLSDIR at
x4 is steadier than RealESRGAN at x2, while NMKD shimmers 1.7x and its sharpness of 1.113 is detail that
was never there. PSNR and SSIM are the wrong columns: bicubic wins them on an eighth of the detail,
because blur is the safe bet for a squared-error metric.

The ceiling matters more than the ranking: nothing here passes ~0.56 of a native render's detail, because
a pixel upscaler can only sharpen what the VAE decoder already drew. Upscaling the *latent* instead has
the decoder render the extra detail, which is why that is the more promising direction.

The architecture is the published one and the checkpoint is ComfyUI's `RealESRGAN_x2.pth`. The x2 variant
is really the x4 network with a `pixel_unshuffle` in front - two spatial factors traded for four
channels, which is why `conv_first` takes 12 inputs rather than 3 - so the declared scale is inferred
from that first layer rather than assumed.

Every frame is enlarged alone, so this cannot invent temporal coherence, only avoid breaking it. A
diffusion video restorer (SeedVR2 is on disk, Apache-2.0, 7B) works across frames and would be better and
far slower.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from difflab.core import ui


class ResidualDenseBlock(nn.Module):
    def __init__(self, features: int = 64, growth: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + 4 * growth, features, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5: torch.Tensor = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, features: int = 64, growth: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(features, growth)
        self.rdb2 = ResidualDenseBlock(features, growth)
        self.rdb3 = ResidualDenseBlock(features, growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RealESRGAN's generator. `scale` is 2 or 4."""

    def __init__(self, blocks: int = 23, features: int = 64, growth: int = 32, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(3 * (4 if scale == 2 else 1), features, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(features, growth) for _ in range(blocks)])
        self.conv_body = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_last = nn.Conv2d(features, 3, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The network always upsamples 4x; scale 2 spends two of those factors on the unshuffle.
        feat = nn.functional.pixel_unshuffle(x, 2) if self.scale == 2 else x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.act(self.conv_up1(nn.functional.interpolate(feat, scale_factor=2.0, mode="nearest")))
        feat = self.act(self.conv_up2(nn.functional.interpolate(feat, scale_factor=2.0, mode="nearest")))
        out: torch.Tensor = self.conv_last(self.act(self.conv_hr(feat)))
        return out


def _from_old_layout(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a pre-RealESRGAN checkpoint (`model.1.sub.N.RDB1.conv1.0`) to this module's names.

    Most community upscalers - 4xLSDIR, the NMKD sets - still ship in the original ESRGAN layout, where
    the whole network is one `nn.Sequential` and the trunk convolution is the sub-block after the last
    RRDB. The weights are identical; only the paths differ.
    """
    subs = [int(key.split(".")[3]) for key in state if key.startswith("model.1.sub.") and "RDB" in key]
    blocks = 1 + max(subs)
    renamed: dict[str, torch.Tensor] = {}
    fixed = {
        "model.0": "conv_first",
        f"model.1.sub.{blocks}": "conv_body",
        "model.3": "conv_up1",
        "model.6": "conv_up2",
        "model.8": "conv_hr",
        "model.10": "conv_last",
    }
    for key, tensor in state.items():
        stem, _, leaf = key.rpartition(".")  # leaf is "weight" or "bias"
        if stem in fixed:
            renamed[f"{fixed[stem]}.{leaf}"] = tensor
            continue
        # model.1.sub.<block>.RDB<n>.conv<n>.0.<leaf>
        parts = key.split(".")
        if len(parts) == 8 and parts[2] == "sub" and parts[4].startswith("RDB"):
            block, rdb, conv = parts[3], parts[4].lower(), parts[5]
            renamed[f"body.{block}.{rdb}.{conv}.{leaf}"] = tensor
            continue
        raise KeyError(f"unrecognised upscaler tensor: {key}")
    return renamed


def load(path: str | Path, device: str = "cuda") -> RRDBNet:
    """Read an ESRGAN-family checkpoint, inferring its shape rather than trusting a config."""
    with ui.step("Loading upscaler") as detail:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
        state = raw.get("params_ema", raw.get("params", raw))
        if any(key.startswith("model.") for key in state):
            state = _from_old_layout(state)
        blocks = 1 + max(int(key.split(".")[1]) for key in state if key.startswith("body."))
        features, first_in = state["conv_first.weight"].shape[:2]
        growth = int(state["body.0.rdb1.conv1.weight"].shape[0])
        scale = 2 if first_in == 12 else 4
        model = RRDBNet(blocks, int(features), growth, scale)
        model.load_state_dict(state)  # strict: a silently skipped layer would just soften the output
        detail(f"x{scale} · {blocks} RRDB · {features}f/{growth}g · {_size(model):.0f} MB")
    return model.eval().requires_grad_(False).to(device)


def _size(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2


def upscale_image(model: RRDBNet, image: torch.Tensor, tile: int = 1024, overlap: int = 64) -> torch.Tensor:
    """`[1, 3, H, W]` in [0, 1] -> `[1, 3, H*scale, W*scale]`, tiled only if it has to be.

    Twenty-three RRDB blocks of 3x3 convolutions see about 350 pixels in every direction, so no
    affordable margin makes a tile boundary exact and cutting on one leaves a visible line (measured:
    1e-2 against the whole frame at 32 px). Tiles are cross-faded across their overlap, which spreads
    that error rather than drawing it, and `tile` defaults high enough that a frame at the sizes this
    project generates goes through in one piece.
    """
    _, _, height, width = image.shape
    if tile <= 0 or (height <= tile and width <= tile):
        whole: torch.Tensor = model(image)
        return whole.clamp(0, 1)

    scale = model.scale
    total = torch.zeros(1, 3, height * scale, width * scale, dtype=torch.float32, device=image.device)
    weight = torch.zeros(1, 1, height * scale, width * scale, dtype=torch.float32, device=image.device)
    step = max(1, tile - overlap)
    for top in range(0, height, step):
        for left in range(0, width, step):
            bottom, right = min(top + tile, height), min(left + tile, width)
            patch = model(image[:, :, top:bottom, left:right]).float()
            ramp = _tile_weight(patch.shape[-2], patch.shape[-1], overlap * scale, patch.device)
            total[:, :, top * scale : bottom * scale, left * scale : right * scale] += patch * ramp
            weight[:, :, top * scale : bottom * scale, left * scale : right * scale] += ramp
            if right >= width:
                break
        if bottom >= height:
            break
    return (total / weight.clamp(min=1e-6)).clamp(0, 1)


def _tile_weight(height: int, width: int, margin: int, device: torch.device) -> torch.Tensor:
    """A tile's contribution: flat in the middle, ramping to zero over `margin` at every edge."""

    def axis(length: int) -> torch.Tensor:
        ramp = torch.ones(length, dtype=torch.float32, device=device)
        edge = min(margin, length // 2)
        if edge > 0:
            fade = torch.linspace(1.0 / (edge + 1), 1.0, edge, device=device)
            ramp[:edge] = fade
            ramp[-edge:] = fade.flip(0)
        return ramp

    return axis(height).view(1, 1, -1, 1) * axis(width).view(1, 1, 1, -1)


def target_size(width: int, height: int, megapixels: float, align: int = 2) -> tuple[int, int]:
    """The size `megapixels` asks for at this aspect ratio, rounded to `align`."""
    factor = math.sqrt(megapixels * 1024 * 1024 / (width * height))
    return (
        max(align, round(width * factor / align) * align),
        max(align, round(height * factor / align) * align),
    )
