"""SeedVR2's causal 3D VAE: 8x spatial, 4x temporal, 16 latent channels.

Causal along time: every conv sees only frames before it, and a clip is encoded four frames (decoded one
latent frame) at a time with each conv carrying its last padded frames across the seam (`memory`).

Every conv is comfy-kitchen's CUTLASS fp16-accumulate implicit GEMM over an NDHWC buffer that the fused
GroupNorm+SiLU+pad kernel writes once. At 1920x1088 and 33 frames on the 3090 (2026-10-11), against the
bf16 cuDNN version with 1024 px tiles: encode 18.1 -> 8.3 s, decode 45.2 -> 27.2 s, peak 21.5 GiB; the
round trip scores 33.0 dB against 33.3, and the decode is 46.8 dB from the old one. The whole frame at
once runs out of memory in the last upsample's shuffle, hence two tiles.
"""

from __future__ import annotations

import math
from enum import Enum

import comfy_kitchen
import torch
from torch import nn
from torch.nn import functional

LATENT_CHANNELS = 16
SCALING = 0.9152
CHANNELS = (128, 256, 512, 512)
LAYERS = 2
GROUPS = 32
FRAMES_PER_SLICE = 4  # the reference config's `slicing.split_size`
SPATIAL = 8
TILE_AREA, OVERLAP = 1088 * 1024, 128  # pixels
VIEW_LIMIT = 2**30  # elements: comfy-kitchen's kernels address one view with 32-bit byte offsets
CL = torch.channels_last_3d


class Memory(Enum):
    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2


def _bordered(body: torch.Tensor, x: torch.Tensor, border: tuple[int, int, int, int]) -> None:
    left, right, top, bottom = border
    h, w = x.shape[-2:]
    if top:
        body[..., :top, :].zero_()
    if bottom:
        body[..., top + h :, :].zero_()
    if left:
        body[..., top : top + h, :left].zero_()
    if right:
        body[..., top : top + h, left + w :].zero_()
    body[..., top : top + h, left : left + w].copy_(x)


class CausalConv3d(nn.Conv3d):
    """Pads time only on the past side, with copies of the first frame or with the previous slice's tail."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        pad = self.padding
        assert isinstance(pad, tuple)
        self.temporal_padding, self.spatial = int(pad[0]), (int(pad[1]), int(pad[2]))
        self.memory: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        memory: Memory = Memory.DISABLED,
        norm: nn.GroupNorm | None = None,
        residual: torch.Tensor | None = None,
        border: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        """`norm` is applied with a SiLU on the way into the buffer; `border` is (left, right, top, bottom)."""
        if self.kernel_size == (1, 1, 1):
            return self._conv(x.contiguous(memory_format=CL), residual)
        if memory != Memory.ACTIVE:
            self.memory = None
        ph, pw = self.spatial
        left, right, top, bottom = border or (pw, pw, ph, ph)
        b, c, t, h, w = x.shape
        halo = self.memory.size(2) if self.memory is not None else 2 * self.temporal_padding
        buf = torch.empty(
            (b, c, halo + t, h + top + bottom, w + left + right), dtype=x.dtype, device=x.device, memory_format=CL
        )
        body = buf[:, :, halo:]
        if norm is not None:
            comfy_kitchen.group_norm_silu_pad3d(
                x, norm.weight, norm.bias, norm.num_groups, norm.eps, (left, right, top, bottom, 0), True,
                zero_pad=True, out=body,
            )  # fmt: skip
        else:
            _bordered(body, x, (left, right, top, bottom))
        if self.memory is not None:
            buf[:, :, :halo].copy_(self.memory)
        elif halo:
            buf[:, :, :halo].copy_(body[:, :, :1].expand(-1, -1, halo, -1, -1))
        cache = self.kernel_size[0] - self.stride[0]
        if memory != Memory.DISABLED and cache:
            self.memory = buf[:, :, -cache:].clone(memory_format=CL)
        return self._conv(buf, residual)

    def _conv(self, x: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        """Over windows of output frames, each input and output view under `VIEW_LIMIT`."""
        kt, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        b, c, d, hp, wp = x.shape
        out = torch.empty(
            (b, self.out_channels, (d - kt) // sd + 1, (hp - kh) // sh + 1, (wp - kw) // sw + 1),
            dtype=x.dtype, device=x.device, memory_format=CL,
        )  # fmt: skip
        if residual is not None:
            residual = residual.contiguous(memory_format=CL)
        frames = out.size(2)
        step = max(1, min((VIEW_LIMIT // (c * hp * wp) - kt) // sd + 1, VIEW_LIMIT // out[0, :, 0].numel(), frames))
        for z0 in range(0, frames, step):
            z1 = min(frames, z0 + step)
            comfy_kitchen.fp16_conv3d(
                x[:, :, z0 * sd : (z1 - 1) * sd + kt], self.weight, self.bias,
                residual=None if residual is None else residual[:, :, z0:z1], stride=self.stride, out=out[:, :, z0:z1],
            )  # fmt: skip
        return out


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, in_channels, eps=1e-6)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(GROUPS, out_channels, eps=1e-6)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = CausalConv3d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor, memory: Memory) -> torch.Tensor:
        h: torch.Tensor = self.conv1(x, memory, self.norm1)
        residual = x if self.conv_shortcut is None else self.conv_shortcut(x, memory)
        out: torch.Tensor = self.conv2(h, memory, self.norm2, residual)
        return out


class Downsample(nn.Module):
    def __init__(self, channels: int, temporal: bool) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            channels,
            channels,
            kernel_size=(3 if temporal else 1, 3, 3),
            stride=(2 if temporal else 1, 2, 2),
            padding=(1 if temporal else 0, 0, 0),
        )

    def forward(self, x: torch.Tensor, memory: Memory) -> torch.Tensor:
        out: torch.Tensor = self.conv(x, memory, border=(0, 1, 0, 1))
        return out


class Upsample(nn.Module):
    """MAGViT-v2's: a 1x1x1 conv to every sub-pixel (and sub-frame), then a shuffle."""

    def __init__(self, channels: int, temporal: bool) -> None:
        super().__init__()
        self.temporal = temporal
        ratio = 4 * (2 if temporal else 1)
        self.upscale_conv = CausalConv3d(channels, channels * ratio, kernel_size=1)
        self.conv = CausalConv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, memory: Memory) -> torch.Tensor:
        z = 2 if self.temporal else 1
        b, _, f, h, w = x.shape
        x = self.upscale_conv(x, memory).view(b, 2, 2, z, -1, f, h, w)
        x = x.permute(0, 4, 5, 3, 6, 1, 7, 2).reshape(b, -1, f * z, h * 2, w * 2)
        if self.temporal and memory != Memory.ACTIVE:
            x = torch.cat([x[:, :, :1], x[:, :, 2:]], dim=2)
        out: torch.Tensor = self.conv(x, memory)
        return out


class Attention(nn.Module):
    """diffusers' single-head spatial attention of a VAE mid-block, frame by frame."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(GROUPS, channels, eps=1e-6)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = x.shape
        frames = x.transpose(1, 2).flatten(0, 1)
        seq = self.group_norm(frames).flatten(2).transpose(1, 2)
        q, k, v = (proj(seq)[:, None] for proj in (self.to_q, self.to_k, self.to_v))
        attended = functional.scaled_dot_product_attention(q, k, v)[:, 0]
        out: torch.Tensor = self.to_out[0](attended).transpose(1, 2).reshape(b * f, c, h, w) + frames
        return out.view(b, f, c, h, w).transpose(1, 2)


class MidBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock(channels, channels), ResnetBlock(channels, channels)])
        self.attentions = nn.ModuleList([Attention(channels)])

    def forward(self, x: torch.Tensor, memory: Memory) -> torch.Tensor:
        x = self.resnets[0](x, memory)
        return self.resnets[1](self.attentions[0](x), memory)  # type: ignore[no-any-return]


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool, temporal: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock(in_channels if i == 0 else out_channels, out_channels) for i in range(LAYERS)]
        )
        self.downsamplers = nn.ModuleList([Downsample(out_channels, temporal)]) if downsample else None

    def forward(self, x: torch.Tensor, memory: Memory) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, memory)
        return self.downsamplers[0](x, memory) if self.downsamplers is not None else x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool, temporal: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock(in_channels if i == 0 else out_channels, out_channels) for i in range(LAYERS + 1)]
        )
        self.upsamplers = nn.ModuleList([Upsample(out_channels, temporal)]) if upsample else None

    def forward(self, x: torch.Tensor, memory: Memory) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, memory)
        return self.upsamplers[0](x, memory) if self.upsamplers is not None else x


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = CausalConv3d(3, CHANNELS[0], 3, padding=1)
        last = len(CHANNELS) - 1
        self.down_blocks = nn.ModuleList(
            [
                # The last two downsampling blocks also halve time, twice: 4x.
                DownBlock(CHANNELS[max(i - 1, 0)], CHANNELS[i], i < last, i >= last - 2)
                for i in range(len(CHANNELS))
            ]
        )
        self.mid_block = MidBlock(CHANNELS[-1])
        self.conv_norm_out = nn.GroupNorm(GROUPS, CHANNELS[-1], eps=1e-6)
        self.conv_out = CausalConv3d(CHANNELS[-1], 2 * LATENT_CHANNELS, 3, padding=1)

    def forward(self, x: torch.Tensor, memory: Memory) -> torch.Tensor:
        x = self.conv_in(x, memory)
        for block in self.down_blocks:
            x = block(x, memory)
        x = self.mid_block(x, memory)
        out: torch.Tensor = self.conv_out(x, memory, self.conv_norm_out)
        return out


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        reversed_channels = list(reversed(CHANNELS))
        self.conv_in = CausalConv3d(LATENT_CHANNELS, CHANNELS[-1], 3, padding=1)
        self.mid_block = MidBlock(CHANNELS[-1])
        last = len(CHANNELS) - 1
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(reversed_channels[max(i - 1, 0)], reversed_channels[i], i < last, i < 2)
                for i in range(len(CHANNELS))
            ]
        )
        self.conv_norm_out = nn.GroupNorm(GROUPS, CHANNELS[0], eps=1e-6)
        self.conv_out = CausalConv3d(CHANNELS[0], 3, 3, padding=1)

    def forward(self, x: torch.Tensor, memory: Memory) -> torch.Tensor:
        x = self.mid_block(self.conv_in(x, memory), memory)
        for block in self.up_blocks:
            x = block(x, memory)
        out: torch.Tensor = self.conv_out(x, memory, self.conv_norm_out)
        return out


def _spans(length: int, rows: int, overlap: int) -> list[tuple[int, int]]:
    """`rows` equal spans covering `length`, neighbours sharing at least `overlap`."""
    if rows == 1:
        return [(0, length)]
    size = math.ceil((length + (rows - 1) * overlap) / rows)
    return [(start, start + size) for start in (round(i * (length - size) / (rows - 1)) for i in range(rows))]


def _tiles(height: int, width: int, area: int, overlap: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The grid of tiles under `area` that covers the least area in total."""
    best: tuple[int, list[tuple[int, int]], list[tuple[int, int]]] | None = None
    for rows in range(1, max(1, height // overlap) + 1):
        for cols in range(1, max(1, width // overlap) + 1):
            ys, xs = _spans(height, rows, overlap), _spans(width, cols, overlap)
            size = (ys[0][1] - ys[0][0]) * (xs[0][1] - xs[0][0])
            if size <= area and (best is None or rows * cols * size < best[0]):
                best = rows * cols * size, ys, xs
    assert best is not None
    return best[1], best[2]


class VideoVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def channels_last(self) -> VideoVAE:
        """Filters stored NDHWC, as the kitchen conv reads them."""
        for module in self.modules():
            if isinstance(module, CausalConv3d):
                module.weight.data = module.weight.data.contiguous(memory_format=CL)
        return self

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """`[1, 3, 4n+1, H, W]` in [-1, 1] -> scaled latents `[1, 16, n+1, H/8, W/8]`, the posterior's mean."""
        return (self._tiled(self.encoder, x.half(), FRAMES_PER_SLICE, False) * SCALING).to(x.dtype)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._tiled(self.decoder, (z / SCALING).half(), max(1, FRAMES_PER_SLICE // 4), True)

    def _tiled(self, net: nn.Module, x: torch.Tensor, frames: int, up: bool) -> torch.Tensor:
        """Spatial tiles under `TILE_AREA` sharing `OVERLAP`, cross-faded on a raised cosine and cut on the
        latent grid."""
        into, out_of = (1, SPATIAL) if up else (SPATIAL, 1)
        height, width = math.ceil(x.shape[-2] / into), math.ceil(x.shape[-1] / into)
        overlap = OVERLAP // SPATIAL
        ys, xs = _tiles(height, width, TILE_AREA // SPATIAL**2, overlap)
        if len(ys) == len(xs) == 1:
            return self._sliced(net, x, frames)[:, : LATENT_CHANNELS if not up else None]
        fade = overlap * out_of
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, 1, fade, device=x.device) * torch.pi)
        result: torch.Tensor | None = None
        count = torch.zeros(height * out_of, width * out_of, device=x.device)
        for top, bottom in ys:
            for left, right in xs:
                piece = x[..., top * into : bottom * into, left * into : right * into]
                out = self._sliced(net, piece, frames)[:, : LATENT_CHANNELS if not up else None]
                out = out[..., : (bottom - top) * out_of, : (right - left) * out_of].float()
                weight_h = torch.ones(out.shape[-2], device=x.device)
                weight_w = torch.ones(out.shape[-1], device=x.device)
                if top:
                    weight_h[:fade] = ramp
                if bottom < height:
                    weight_h[-fade:] = 1 - ramp
                if left:
                    weight_w[:fade] = ramp
                if right < width:
                    weight_w[-fade:] = 1 - ramp
                weight = weight_h[:, None] * weight_w[None]
                if result is None:
                    result = torch.zeros(*out.shape[:3], height * out_of, width * out_of, device=x.device)
                rows, cols = (
                    slice(top * out_of, top * out_of + out.shape[-2]),
                    slice(left * out_of, left * out_of + out.shape[-1]),
                )
                result[..., rows, cols] += out * weight
                count[rows, cols] += weight
        assert result is not None
        return (result / count.clamp_min(1e-6)).to(x.dtype)

    def _sliced(self, net: nn.Module, x: torch.Tensor, size: int) -> torch.Tensor:
        if x.shape[2] - 1 <= size:
            out: torch.Tensor = net(x, Memory.DISABLED)
            return out
        rest = torch.split(x[:, :, 1:], size, dim=2)
        parts = [net(torch.cat([x[:, :, :1], rest[0]], dim=2), Memory.INITIALIZING)]
        parts += [net(piece, Memory.ACTIVE) for piece in rest[1:]]
        for module in self.modules():
            if isinstance(module, CausalConv3d):
                module.memory = None
        return torch.cat(parts, dim=2)
