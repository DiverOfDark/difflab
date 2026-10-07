"""SeedVR2's causal 3D VAE: 8x spatial, 4x temporal, 16 latent channels.

Causal along time: every conv sees only frames before it, and a clip is encoded four frames (decoded one
latent frame) at a time with each conv carrying its last frames across the seam (`memory`). A conv or
norm over more than `MEMORY_LIMIT` GiB splits itself along height, then width, carrying the overlap
its kernel needs - what lets 1080p go through on a 24 GB card.
"""

from __future__ import annotations

import math
from enum import Enum

import torch
from torch import nn
from torch.nn import functional

LATENT_CHANNELS = 16
SCALING = 0.9152
CHANNELS = (128, 256, 512, 512)
LAYERS = 2
GROUPS = 32
FRAMES_PER_SLICE = 4  # the reference config's `slicing.split_size`
MEMORY_LIMIT = 0.5  # GiB, the reference config's `memory_limit` for convs and norms alike
SPATIAL = 8
TILE, OVERLAP = 1024, 128  # pixels


class Memory(Enum):
    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2


def _cache_size(conv: nn.Conv3d, length: int, pad: int, dim: int) -> int:
    kernel = conv.dilation[dim] * (conv.kernel_size[dim] - 1) + 1
    stride = conv.stride[dim]
    out = (length + pad - kernel) // stride + 1
    remain = length + pad - ((out - 1) * stride + kernel)
    return kernel - stride + remain


class CausalConv3d(nn.Conv3d):
    """Pads time only on the past side, with copies of the first frame or with the previous slice's tail."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        pad = self.padding
        assert isinstance(pad, tuple)
        self.temporal_padding, self.spatial = int(pad[0]), (int(pad[1]), int(pad[2]))
        self.memory: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, memory: Memory = Memory.DISABLED) -> torch.Tensor:
        if memory != Memory.ACTIVE:
            self.memory = None
        cache_size = self.kernel_size[0] - self.stride[0]
        if self.memory is not None:
            cache: torch.Tensor | None = self.memory.to(x)
        elif self.temporal_padding:
            cache = x[:, :, :1].repeat(1, 1, self.temporal_padding * 2, 1, 1)
        else:
            cache = None
        if memory != Memory.DISABLED and cache_size:
            if cache_size > x.size(2) and cache is not None:
                x, cache = torch.cat([cache, x], dim=2), None
            if cache_size <= x.size(2):
                self.memory = x[:, :, -cache_size:].detach().contiguous()
        padding = (self.spatial[1], self.spatial[1], self.spatial[0], self.spatial[0], 0, 0)
        return self._limited(x, 3, padding, cache)

    def _limited(
        self, x: torch.Tensor, split_dim: int, padding: tuple[int, ...], prev_cache: torch.Tensor | None
    ) -> torch.Tensor:
        shape = torch.tensor(x.size())
        if prev_cache is not None:
            shape[split_dim - 1] += prev_cache.size(split_dim - 1)
        shape[-3:] += torch.tensor(padding).view(3, 2).sum(-1).flip(0)
        size = float(shape.prod()) * x.element_size() / 1024**3
        if size < MEMORY_LIMIT or split_dim == x.ndim:
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            return functional.conv3d(
                functional.pad(x, padding), self.weight, self.bias, self.stride, 0, self.dilation, self.groups
            )

        splits = math.ceil(size / MEMORY_LIMIT)
        per = x.size(split_dim) // splits
        sizes = [per] * (splits - 1) + [x.size(split_dim) - per * (splits - 1)]
        parts = list(torch.split(x, sizes, dim=split_dim))
        caches = list(torch.split(prev_cache, sizes, dim=split_dim)) if prev_cache is not None else None
        cache: torch.Tensor | None = None
        for index in range(len(parts)):
            if caches is not None:
                parts[index] = torch.cat([caches[index], parts[index]], dim=split_dim - 1)
            left = (parts[index].ndim - split_dim - 1) * 2
            pad = list(padding)
            edge = self.spatial[split_dim - 3]
            pad[left] = edge if index == 0 else 0
            pad[left + 1] = edge if index == len(parts) - 1 else 0
            held = cache.size(split_dim) if cache is not None else 0
            following = _cache_size(self, parts[index].size(split_dim) + held, pad[left] + pad[left + 1], split_dim - 2)
            next_cache = (
                parts[index].transpose(0, split_dim)[-following:].transpose(0, split_dim) if following else None
            )
            parts[index] = self._limited(parts[index], split_dim + 1, tuple(pad), cache)
            cache = next_cache
        return torch.cat(parts, split_dim)


def _norm(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    """Frame by frame, and in channel chunks when big: chunks of whole groups are the same arithmetic."""
    t = x.size(2)
    flat = x.transpose(1, 2).flatten(0, 1)
    if flat.numel() * flat.element_size() / 1024**3 > MEMORY_LIMIT:
        chunks = min(4 if flat.element_size() == 2 else 2, norm.num_groups)
        parts = [
            functional.group_norm(part, norm.num_groups // chunks, w, b, norm.eps)
            for part, w, b in zip(
                flat.chunk(chunks, dim=1), norm.weight.chunk(chunks), norm.bias.chunk(chunks), strict=True
            )
        ]
        flat = torch.cat(parts, dim=1)
    else:
        flat = norm(flat)
    return flat.view(-1, t, *flat.shape[1:]).transpose(1, 2)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, in_channels, eps=1e-6)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(GROUPS, out_channels, eps=1e-6)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = CausalConv3d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor, memory: Memory) -> torch.Tensor:
        h: torch.Tensor = self.conv1(functional.silu(_norm(self.norm1, x), inplace=True), memory)
        h = self.conv2(functional.silu(_norm(self.norm2, h), inplace=True), memory)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x, memory)
        return x + h


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
        out: torch.Tensor = self.conv(functional.pad(x, (0, 1, 0, 1)), memory)
        return out


class Upsample(nn.Module):
    """MAGViT-v2's: a 1x1x1 conv to every sub-pixel (and sub-frame), then a shuffle."""

    def __init__(self, channels: int, temporal: bool) -> None:
        super().__init__()
        self.temporal = temporal
        ratio = 4 * (2 if temporal else 1)
        self.upscale_conv = nn.Conv3d(channels, channels * ratio, kernel_size=1)
        self.conv = CausalConv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, memory: Memory) -> torch.Tensor:
        z = 2 if self.temporal else 1
        b, _, f, h, w = x.shape
        x = self.upscale_conv(x).view(b, 2, 2, z, -1, f, h, w)
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
        out: torch.Tensor = self.conv_out(functional.silu(_norm(self.conv_norm_out, x), inplace=True), memory)
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
        out: torch.Tensor = self.conv_out(functional.silu(_norm(self.conv_norm_out, x), inplace=True), memory)
        return out


class VideoVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """`[1, 3, 4n+1, H, W]` in [-1, 1] -> scaled latents `[1, 16, n+1, H/8, W/8]`, the posterior's mean."""
        return self._tiled(self.encoder, x, FRAMES_PER_SLICE, False) * SCALING

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._tiled(self.decoder, z / SCALING, max(1, FRAMES_PER_SLICE // 4), True)

    def _tiled(self, net: nn.Module, x: torch.Tensor, frames: int, up: bool) -> torch.Tensor:
        """Spatial tiles of `TILE` pixels sharing `OVERLAP`, cross-faded on a raised cosine and cut on the
        latent grid: at 1088x1920 the whole frame peaks past 22 GB, the per-conv temporal caches
        alone holding about 8 GB of it."""
        into, out_of = (1, SPATIAL) if up else (SPATIAL, 1)
        height, width = math.ceil(x.shape[-2] / into), math.ceil(x.shape[-1] / into)
        tile, overlap = TILE // SPATIAL, OVERLAP // SPATIAL
        if height <= tile and width <= tile:
            return self._sliced(net, x, frames)[:, : LATENT_CHANNELS if not up else None]
        stride = tile - overlap
        fade = overlap * out_of
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, 1, fade, device=x.device) * torch.pi)
        result: torch.Tensor | None = None
        count = torch.zeros(height * out_of, width * out_of, device=x.device)
        for top in range(0, height, stride):
            bottom = min(top + tile, height)
            for left in range(0, width, stride):
                right = min(left + tile, width)
                if (top and bottom - top <= overlap) or (left and right - left <= overlap):
                    continue
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
