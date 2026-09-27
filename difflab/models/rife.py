"""RIFE 4.26's IFNet, for putting frames between the frames H3 drew.

H3 renders 24 fps and nothing changes that: the frame rate is baked into the VAE's temporal packing
(`n % 17 == 5`) and into the model's sense of how fast things move. Motion at 48 or 60 is therefore
added afterwards, by a flow network that finds where a pixel went between two frames rather than
blending two pictures and calling the smear a frame.

The architecture is the published one and the checkpoint is ComfyUI's conversion of
`RIFEv4.26_0921`, loaded directly. Two things about this version that the internet still gets wrong:

* **There are five blocks, not four.** 4.25 added `block4`, so `scale_list` has five entries. The
  docstring default in the upstream file (`[8, 4, 2, 1]`) was never updated and would index past the
  end; the inference script always passes the five.
* **`ensemble` and `fastmode` do not exist here.** The refine U-net was dropped after 4.3 and the
  swap-and-average ensemble trick after 4.17. Anything that offers them is describing an older RIFE.

`t` is not a recursion depth: it arrives as a full-resolution channel filled with one number, so any
timestep in [0, 1] is one forward pass. That is what makes 24 -> 60 (t cycling 0, 0.4, 0.8, 0.2, 0.6)
cost the same per output frame as 24 -> 48, rather than needing a power-of-two ladder.

Half precision is fine for the network and wrong for the warp: `grid_sample` on fp16 flow quantises
the sampling grid coarsely enough to show as banding in slow pans, so `warp` upcasts and comes back.

What this cannot do is a cut: RIFE morphs across one, so a retimed edit needs the cut detected and the
frame duplicated instead. A take is one continuous shot by construction, which is why `tools.retime`
does not carry that machinery.

The code is MIT (hzwer's Practical-RIFE, and HolyWu's vs-rife port that this follows for the
precomputed warp grid), and so are the released weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

# The flow pyramid's coarsest level runs at 1/16, and each block halves twice before its trunk, so
# both sides must divide by 64. H3's canvases are multiples of 32 and usually of 64 already.
MODULO = 64
SCALES = (16.0, 8.0, 4.0, 2.0, 1.0)


def conv(in_planes: int, out_planes: int, kernel: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel, stride, padding, bias=True),
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    """The per-image encoder. Its output is reused for every frame made between the same pair, which is
    where the time goes at 60 fps: two encodes serve four interpolations."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        out: torch.Tensor = self.cnn3(x)
        return out


class ResConv(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.relu(self.conv(x) * self.beta + x)
        return out


class IFBlock(nn.Module):
    """One level of the flow pyramid: a residual to the flow, a blend mask, and features onward."""

    def __init__(self, in_planes: int, c: int = 64) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*(ResConv(c) for _ in range(8)))
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(
        self, x: torch.Tensor, flow: torch.Tensor | None, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = nn.functional.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            # The flow is in pixels, so shrinking the picture has to shrink the vectors with it.
            flow = nn.functional.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        tmp = self.lastconv(self.convblock(self.conv0(x)))
        tmp = nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


@dataclass(frozen=True)
class Grid:
    """`grid_sample` wants normalised coordinates, so a flow in pixels is divided by half the width and
    height before it joins the identity grid. Both tensors depend only on the padded size, and a clip is
    thousands of warps at one size."""

    flow_div: torch.Tensor
    base: torch.Tensor
    padding: tuple[int, int, int, int]

    @property
    def padded(self) -> bool:
        return any(self.padding)


def make_grid(width: int, height: int, device: str | torch.device) -> Grid:
    """The grid for a picture of this size, padded up to what the pyramid needs."""
    padded_w = -(-width // MODULO) * MODULO
    padded_h = -(-height // MODULO) * MODULO
    flow_div = torch.tensor([(padded_w - 1.0) / 2.0, (padded_h - 1.0) / 2.0], dtype=torch.float, device=device)
    horizontal = torch.linspace(-1.0, 1.0, padded_w, dtype=torch.float, device=device)
    horizontal = horizontal.view(1, 1, 1, padded_w).expand(-1, -1, padded_h, -1)
    vertical = torch.linspace(-1.0, 1.0, padded_h, dtype=torch.float, device=device)
    vertical = vertical.view(1, 1, padded_h, 1).expand(-1, -1, -1, padded_w)
    base = torch.cat([horizontal, vertical], 1)
    return Grid(flow_div, base, (0, padded_w - width, 0, padded_h - height))


def warp(image: torch.Tensor, flow: torch.Tensor, grid: Grid) -> torch.Tensor:
    """`image` sampled where `flow` points, in fp32 whatever the network is running in."""
    dtype = image.dtype
    flow = flow.to(torch.float)
    normalised = torch.cat([flow[:, 0:1] / grid.flow_div[0], flow[:, 1:2] / grid.flow_div[1]], 1)
    sampled: torch.Tensor = nn.functional.grid_sample(
        input=image.to(torch.float),
        grid=(grid.base + normalised).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.to(dtype)


class IFNet(nn.Module):
    """Five coarse-to-fine passes at one flow field, then one blend of the two warped frames."""

    def __init__(self) -> None:
        super().__init__()
        # A ModuleList, not block0..block4 as attributes: that is how the checkpoint names them, so
        # it loads strictly and a missing tensor is an error rather than a silently untrained layer.
        self.blocks = nn.ModuleList(
            [
                IFBlock(7 + 8, c=192),
                IFBlock(8 + 4 + 8 + 8, c=128),
                IFBlock(8 + 4 + 8 + 8, c=96),
                IFBlock(8 + 4 + 8 + 8, c=64),
                IFBlock(8 + 4 + 8 + 8, c=32),
            ]
        )
        self.encode = Head()

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: torch.Tensor,
        f0: torch.Tensor,
        f1: torch.Tensor,
        grid: Grid,
    ) -> torch.Tensor:
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        flow: torch.Tensor | None = None
        mask: torch.Tensor | None = None
        feat: torch.Tensor | None = None
        warped0, warped1 = img0, img1
        for block, scale in zip(self.blocks, SCALES, strict=True):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale)
            else:
                assert mask is not None and feat is not None  # set together with flow, on the first pass
                warped_f0 = warp(f0, flow[:, :2], grid)
                warped_f1 = warp(f1, flow[:, 2:4], grid)
                stacked = torch.cat((warped0, warped1, warped_f0, warped_f1, timestep, mask, feat), 1)
                residual, mask, feat = block(stacked, flow, scale)
                flow = flow + residual
            warped0 = warp(img0, flow[:, :2], grid)
            warped1 = warp(img1, flow[:, 2:4], grid)
        assert mask is not None
        blend = torch.sigmoid(mask)
        return warped0 * blend + warped1 * (1 - blend)


def load(path: str | Path, device: str | torch.device, dtype: torch.dtype = torch.float16) -> IFNet:
    model = IFNet()
    model.load_state_dict(load_file(str(path)))
    return model.eval().to(device=device, dtype=dtype).requires_grad_(False)
