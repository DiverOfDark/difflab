"""A latent-space upscaler for MiniMax H3 (`LatentResizer3D`, 345 M parameters).

A pixel upscaler works on whatever the VAE decoder already drew, and the best of them recovers about
0.56 of a native render's high-frequency detail (see `esrgan.py`) - a ceiling, not a tuning problem.
Enlarging before the decode has H3's own 36-block ViT decoder render the extra pixels instead, and this
model is temporally aware: depthwise convolutions over time sit between the spatial blocks.

The architecture is read off the released `config.json` and verified against the tensor shapes:

    conv_in -> 18 blocks -> trilinear interpolate -> 18 blocks -> norm_out -> SiLU -> conv_out

Both block stacks are the same layout, 12 `ResBlockEmb3D` interleaved with 6 `TemporalConv` at indices
1, 4, 7, 10, 13 and 16. Every residual block is conditioned on the requested scale through a FiLM pair,
so one checkpoint covers the whole 1.0-4.0 range - which is what lets 0.4 MP go to 2 MP in one step
instead of x2-then-resample.

`T` is preserved; only `H` and `W` grow. The reference workflow does not decode straight after
upscaling: it splits the sigma schedule, denoising partway at the small size, upscaling, then finishing
the remaining sigmas at the target size. Skipping that refinement is cheaper, and whether the result
holds up without it has not been measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from difflab.core import ui

LATENT_CHANNELS = 24
SCALE_RANGE = (1.0, 4.0)


class ResBlockEmb3D(nn.Module):
    """A 3D residual block whose output normalisation is modulated by the scale embedding."""

    def __init__(self, channels: int, embed_dim: int, groups: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * channels))
        self.out_norm = nn.GroupNorm(groups, channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Conv3d(channels, channels, 3, padding=1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.in_layers(x)
        scale, shift = self.emb_layers(emb)[..., None, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        out: torch.Tensor = self.out_layers(h)
        return x + out


class TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel: int = 5, groups: int = 32) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.dwconv = nn.Conv3d(channels, channels, (kernel, 1, 1), padding=(kernel // 2, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # The released config describes this block as norm -> dwconv -> pwconv and omits the
        # activation. Leaving it out loads perfectly, runs, and returns a latent that decodes to a
        # checkerboard - the kind of error only a numerical diff against the reference finds.
        residual: torch.Tensor = self.pwconv(self.dwconv(nn.functional.silu(self.norm(x))))
        return x + residual


class LatentResizer3D(nn.Module):
    """Spatial upscaling of an H3 video latent, conditioned on the factor asked for."""

    def __init__(
        self,
        channels: int = 512,
        embed_dim: int = 64,
        res_blocks: int = 12,
        temporal_blocks: int = 6,
        temporal_kernel: int = 5,
        groups: int = 32,
        period: int = 3,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv3d(LATENT_CHANNELS, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        args = (channels, embed_dim, res_blocks, temporal_blocks, temporal_kernel, groups, period)
        self.in_blocks = self._stack(*args)
        self.out_blocks = self._stack(*args)
        self.norm_out = nn.GroupNorm(groups, channels)
        self.conv_out = nn.Conv3d(channels, LATENT_CHANNELS, 3, padding=1)

    @staticmethod
    def _stack(
        channels: int, embed_dim: int, res_blocks: int, temporal_blocks: int, kernel: int, groups: int, period: int
    ) -> nn.ModuleList:
        """The released layout: a temporal block wherever the index is 1 modulo `period`, i.e.
        R, T, R, R, T, R, R, T, ... Counting residual blocks instead puts every temporal block one
        position late, which loads cleanly into nothing."""
        blocks: list[nn.Module] = []
        temporal_used = 0
        for index in range(res_blocks + temporal_blocks):
            if index % period == 1 and temporal_used < temporal_blocks:
                blocks.append(TemporalConv(channels, kernel, groups))
                temporal_used += 1
            else:
                blocks.append(ResBlockEmb3D(channels, embed_dim, groups))
        return nn.ModuleList(blocks)

    def forward(self, latent: torch.Tensor, height: int, width: int, scale: float) -> torch.Tensor:
        """`[1, 24, T, h, w]` -> `[1, 24, T, height, width]`. `T` is untouched."""
        # The embedding is fed `scale - 1`, so an identity resize sits at zero.
        emb = self.embed(torch.full((latent.shape[0], 1), scale - 1.0, dtype=latent.dtype, device=latent.device))
        h = self.conv_in(latent)
        for block in self.in_blocks:
            h = block(h, emb)
        h = nn.functional.interpolate(h, size=(h.shape[2], height, width), mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            h = block(h, emb)
        out: torch.Tensor = self.conv_out(nn.functional.silu(self.norm_out(h)))
        return out


def load(path: str | Path, device: str = "cuda", dtype: torch.dtype = torch.float16) -> LatentResizer3D:
    """Read the released checkpoint, inferring its shape from the tensors rather than a config file."""
    path = Path(path)
    with ui.step("Loading latent upscaler") as detail:
        state = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu")
        channels = int(state["conv_in.weight"].shape[0])
        embed_dim = int(state["embed.0.weight"].shape[0])
        res_blocks = sum(1 for key in state if key.endswith(".in_layers.2.weight") and key.startswith("in_blocks."))
        temporal = sum(1 for key in state if key.endswith(".dwconv.weight") and key.startswith("in_blocks."))
        kernel = int(state["in_blocks.1.dwconv.weight"].shape[2]) if temporal else 5
        # Where the temporal blocks actually sit, rather than where a count would put them.
        positions = sorted(int(k.split(".")[1]) for k in state if k.endswith(".dwconv.weight") and "in_blocks." in k)
        period = (positions[1] - positions[0]) if len(positions) > 1 else res_blocks + 1
        model = LatentResizer3D(channels, embed_dim, res_blocks, temporal, kernel, period=period)
        model.load_state_dict(state)
        parameters = sum(p.numel() for p in model.parameters())
        detail(f"{parameters / 1e6:.0f} M · {channels}ch · {res_blocks} res + {temporal} temporal")
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def read_normalisation(config: str | Path) -> tuple[list[float], list[float]] | None:
    """The per-channel statistics the model was trained against, if its config is beside it."""
    path = Path(config)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    stats = data.get("latent_normalization", {})
    if not stats.get("enabled"):
        return None
    return list(stats["mean"]), list(stats["std"])
