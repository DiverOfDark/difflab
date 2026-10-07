"""SeedVR2 7B's NaDiT: 36 MM-DiT blocks over a video in 3D windows, the text joined to every window.

One sample at a time, which lets the per-sample bookkeeping of the original (varlen sequences,
repeat-interleaved modulation) collapse to broadcasting. Windows of equal size go through attention as
one batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from diffusers.models.embeddings import get_timestep_embedding
from torch import nn
from torch.nn import functional

DIM = 3072
HEADS = 24
HEAD_DIM = 128
LAYERS = 36
TXT_DIM = 5120
IN_CHANNELS = 33  # the noise, the encoded input and a mask of ones
OUT_CHANNELS = 16
EPS = 1e-5
WINDOW = (4, 3, 3)
# The window count the model was trained with, at 720p; another size keeps the window's extent instead.
REFERENCE_TOKENS = 45 * 80
ROPE_FREQS = 10  # rotary_embedding_torch's "pixel" frequencies for a 21-wide axis: linspace(1, 128, 10) * pi


class Pair(nn.Module):
    """The video and text copies of one layer, as the checkpoint names them."""

    def __init__(self, vid: nn.Module, txt: nn.Module) -> None:
        super().__init__()
        self.vid, self.txt = vid, txt


class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)
        if self.weight is not None:
            out = out * self.weight.float()
        return out.to(x.dtype)


class Modulation(nn.Module):
    """AdaSingle: the timestep's shift, scale and gate plus the layer's own learned ones."""

    def __init__(self) -> None:
        super().__init__()
        for layer in ("attn", "mlp"):
            for part in ("shift", "scale", "gate"):
                self.register_parameter(f"{layer}_{part}", nn.Parameter(torch.zeros(DIM)))

    def parts(self, emb: torch.Tensor, layer: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = ("attn", "mlp").index(layer)
        shift, scale, gate = emb.view(DIM, 2, 3)[:, index].unbind(-1)
        return (
            shift + getattr(self, f"{layer}_shift"),
            scale + getattr(self, f"{layer}_scale"),
            gate + getattr(self, f"{layer}_gate"),
        )


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(DIM, DIM * 4)
        self.proj_out = nn.Linear(DIM * 4, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.proj_out(functional.gelu(self.proj_in(x), approximate="tanh"))
        return out


@dataclass
class Layout:
    """One windowing of the token grid: the window-ordered permutation and the windows by size."""

    order: torch.Tensor  # raster index of each token, window by window
    inverse: torch.Tensor
    freqs: torch.Tensor  # [tokens, 60] rope angles, in window order
    groups: list[torch.Tensor]  # [windows, size] positions in window order, one tensor per size
    windows: int


def _windows(size: tuple[int, int, int], shifted: bool) -> list[tuple[slice, slice, slice]]:
    t, h, w = size
    scale = math.sqrt(REFERENCE_TOKENS / (h * w))
    wh, ww = math.ceil(round(h * scale) / WINDOW[1]), math.ceil(round(w * scale) / WINDOW[2])
    wt = math.ceil(min(t, 30) / WINDOW[0])
    if not shifted:
        nt, nh, nw = math.ceil(t / wt), math.ceil(h / wh), math.ceil(w / ww)
        return [
            (
                slice(it * wt, min((it + 1) * wt, t)),
                slice(ih * wh, min((ih + 1) * wh, h)),
                slice(iw * ww, min((iw + 1) * ww, w)),
            )
            for iw in range(nw)
            if min((iw + 1) * ww, w) > iw * ww
            for ih in range(nh)
            if min((ih + 1) * wh, h) > ih * wh
            for it in range(nt)
            if min((it + 1) * wt, t) > it * wt
        ]
    st, sh, sw = (0.5 if wt < t else 0), (0.5 if wh < h else 0), (0.5 if ww < w else 0)
    nt = math.ceil((t - st) / wt) + 1 if st > 0 else 1
    nh = math.ceil((h - sh) / wh) + 1 if sh > 0 else 1
    nw = math.ceil((w - sw) / ww) + 1 if sw > 0 else 1

    def span(i: int, shift: float, width: int, total: int) -> slice:
        return slice(max(int((i - shift) * width), 0), min(int((i - shift + 1) * width), total))

    return [
        (span(it, st, wt, t), span(ih, sh, wh, h), span(iw, sw, ww, w))
        for iw in range(nw)
        if span(iw, sw, ww, w).stop > span(iw, sw, ww, w).start
        for ih in range(nh)
        if span(ih, sh, wh, h).stop > span(ih, sh, wh, h).start
        for it in range(nt)
        if span(it, st, wt, t).stop > span(it, st, wt, t).start
    ]


def _axial_freqs(f: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    base = torch.linspace(1.0, 128.0, ROPE_FREQS, device=device) * math.pi
    axes = []
    for index, n in enumerate((f, h, w)):
        angles = (torch.linspace(-1.0, 1.0, n, device=device)[:, None] * base).repeat_interleave(2, dim=-1)
        shape = [1, 1, 1, angles.shape[-1]]
        shape[index] = n
        axes.append(angles.view(shape).expand(f, h, w, -1))
    return torch.cat(axes, dim=-1).reshape(-1, 6 * ROPE_FREQS)


def layout(size: tuple[int, int, int], shifted: bool, device: torch.device) -> Layout:
    raster = torch.arange(size[0] * size[1] * size[2], device=device).view(size)
    pieces = [raster[window] for window in _windows(size, shifted)]
    order = torch.cat([piece.flatten() for piece in pieces])
    freqs = torch.cat([_axial_freqs(piece.shape[0], piece.shape[1], piece.shape[2], device) for piece in pieces])
    lengths = [piece.numel() for piece in pieces]
    starts = [0, *torch.tensor(lengths).cumsum(0).tolist()[:-1]]
    by_size: dict[int, list[torch.Tensor]] = {}
    for start, length in zip(starts, lengths, strict=True):
        by_size.setdefault(length, []).append(torch.arange(start, start + length, device=device))
    groups = [torch.stack(spans) for spans in by_size.values()]
    return Layout(order, torch.argsort(order), freqs, groups, len(pieces))


def _rotate(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    width = freqs.shape[-1]
    head, tail = x[..., :width].float(), x[..., width:]
    pairs = head.view(*head.shape[:-1], -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    cos, sin = freqs.cos()[:, None], freqs.sin()[:, None]
    return torch.cat(((head * cos + rotated * sin).to(x.dtype), tail), dim=-1)


def _joined(windows: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """`[n, size, heads, dim]` windows each followed by the whole text, heads first for SDPA."""
    return torch.cat([windows, text.expand(len(windows), -1, -1, -1)], dim=1).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_qkv = Pair(nn.Linear(DIM, DIM * 3, bias=False), nn.Linear(DIM, DIM * 3, bias=False))
        self.proj_out = Pair(nn.Linear(DIM, DIM), nn.Linear(DIM, DIM))
        self.norm_q = Pair(RMSNorm(HEAD_DIM, True), RMSNorm(HEAD_DIM, True))
        self.norm_k = Pair(RMSNorm(HEAD_DIM, True), RMSNorm(HEAD_DIM, True))

    def forward(self, vid: torch.Tensor, txt: torch.Tensor, lay: Layout) -> tuple[torch.Tensor, torch.Tensor]:
        vq, vk, vv = self.proj_qkv.vid(vid)[lay.order].unflatten(-1, (3, HEADS, HEAD_DIM)).unbind(1)
        tq, tk, tv = self.proj_qkv.txt(txt).unflatten(-1, (3, HEADS, HEAD_DIM)).unbind(1)
        vq, tq = self.norm_q.vid(vq), self.norm_q.txt(tq)
        vk, tk = self.norm_k.vid(vk), self.norm_k.txt(tk)
        vq, vk = _rotate(vq, lay.freqs), _rotate(vk, lay.freqs)

        vid_out = torch.empty_like(vq)
        txt_sum = torch.zeros(tq.shape, dtype=torch.float32, device=tq.device)
        for group in lay.groups:
            size = group.shape[1]
            q, k, v = (_joined(a[group], b) for a, b in ((vq, tq), (vk, tk), (vv, tv)))
            out = functional.scaled_dot_product_attention(q, k, v).transpose(1, 2)
            vid_out[group] = out[:, :size]
            txt_sum += out[:, size:].float().sum(0)
        vid_flat = vid_out.flatten(1)[lay.inverse]
        txt_flat = (txt_sum / lay.windows).to(txt.dtype).flatten(1)
        return self.proj_out.vid(vid_flat), self.proj_out.txt(txt_flat)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(DIM, False)
        self.mlp_norm = RMSNorm(DIM, False)
        self.attn = Attention()
        self.mlp = Pair(MLP(), MLP())
        self.ada = Pair(Modulation(), Modulation())

    def forward(
        self, vid: torch.Tensor, txt: torch.Tensor, emb: torch.Tensor, lay: Layout
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vid_mod, txt_mod = self.ada.vid, self.ada.txt
        assert isinstance(vid_mod, Modulation) and isinstance(txt_mod, Modulation)
        (vs, vc, vg), (ts, tc, tg) = vid_mod.parts(emb, "attn"), txt_mod.parts(emb, "attn")
        a_vid, a_txt = self.attn(self.attn_norm(vid) * vc + vs, self.attn_norm(txt) * tc + ts, lay)
        vid, txt = vid + a_vid * vg, txt + a_txt * tg
        (vs, vc, vg), (ts, tc, tg) = vid_mod.parts(emb, "mlp"), txt_mod.parts(emb, "mlp")
        vid = vid + self.mlp.vid(self.mlp_norm(vid) * vc + vs) * vg
        txt = txt + self.mlp.txt(self.mlp_norm(txt) * tc + ts) * tg
        return vid, txt


class TimeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(256, DIM)
        self.proj_hid = nn.Linear(DIM, DIM)
        self.proj_out = nn.Linear(DIM, DIM * 6)

    def forward(self, timestep: float, dtype: torch.dtype) -> torch.Tensor:
        device = self.proj_in.weight.device
        emb = get_timestep_embedding(
            torch.tensor([timestep], device=device), 256, flip_sin_to_cos=False, downscale_freq_shift=0
        )
        out: torch.Tensor = self.proj_out(functional.silu(self.proj_hid(functional.silu(self.proj_in(emb.to(dtype))))))
        return out[0]


class PatchIn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(IN_CHANNELS * 4, DIM)


class PatchOut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(DIM, OUT_CHANNELS * 4)


class NaDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vid_in = PatchIn()
        self.txt_in = nn.Linear(TXT_DIM, DIM)
        self.emb_in = TimeEmbedding()
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.vid_out = PatchOut()

    def forward(self, vid: torch.Tensor, txt: torch.Tensor, timestep: float) -> torch.Tensor:
        """`vid [T, H, W, 33]` latents -> `[T, H, W, 16]`, the velocity; `txt [L, 5120]`."""
        t, h, w, c = vid.shape
        tokens = vid.view(t, h // 2, 2, w // 2, 2, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, 4 * c)
        x = self.vid_in.proj(tokens)
        text = self.txt_in(txt)
        emb = self.emb_in(timestep, x.dtype)
        size = (t, h // 2, w // 2)
        layouts = [layout(size, shifted, x.device) for shifted in (False, True)]
        for index, block in enumerate(self.blocks):
            x, text = block(x, text, emb, layouts[index % 2])
        out: torch.Tensor = self.vid_out.proj(x).view(t, h // 2, w // 2, 2, 2, OUT_CHANNELS)
        return out.permute(0, 1, 3, 2, 4, 5).reshape(t, h, w, OUT_CHANNELS)
