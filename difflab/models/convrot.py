"""ComfyUI "ConvRot" int8 weights: per-row scales plus a 256-wide Hadamard rotation.

Stored per layer: `weight` int8 [N, K], `weight_scale` fp32 [N, 1], and a `comfy_quant` JSON blob
`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`.

The weight was rotated by H before quantizing, so `W = (q * scale).view(N, K/256, 256) @ H`. H is the
4-fold Kronecker power of a sign matrix scaled by 1/16 — symmetric and its own inverse. Verified
against ComfyUI's own dequantizer: max difference 2.4e-07.

Inference is one fused kernel, `comfy_kitchen.int8_linear` - the Apache-2.0 kernel pack ComfyUI itself
calls, so the numerics match the reference exactly. It is 3.55x faster than bf16 on this GPU and the
weights stay 19.5 GB instead of 37.5 GB. The same three steps by hand in PyTorch cost eight passes over
the activation and measured 1.8-3.5x slower (59-105 TOPS against 178-210).
"""

import functools
import json

import comfy_kitchen
import torch
from torch import nn

GROUP = 256
_H4 = [[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]


@functools.cache
def hadamard(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The 256x256 ConvRot basis: kron^4(H4) / 16."""
    h = torch.tensor(_H4, dtype=torch.float64)
    for _ in range(3):
        h = torch.kron(h, torch.tensor(_H4, dtype=torch.float64))
    return (h / 16.0).to(device=device, dtype=dtype)


def is_quantized(state_dict: dict[str, torch.Tensor], prefix: str) -> bool:
    return f"{prefix}.comfy_quant" in state_dict


def read_quant_meta(blob: torch.Tensor) -> dict[str, object]:
    meta: dict[str, object] = json.loads(bytes(blob.tolist()).decode())
    return meta


def rotate(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Apply H to the last dimension in 256-wide groups."""
    *lead, k = x.shape
    return (x.reshape(-1, k // GROUP, GROUP).to(h.dtype) @ h).reshape(*lead, k)


def dequantize(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Undo the quantization and the rotation: a dense [N, K] weight."""
    h = hadamard(weight.device)
    return rotate(weight.to(torch.float32) * scale.to(torch.float32), h).to(dtype)


class ConvRotLinear(nn.Module):
    """int8 W8A8 linear with optional LoRA, matching ComfyUI's quantized path.

    The LoRA is applied in activation space rather than merged: merging into int8 rounds most of the
    delta away.
    """

    weight: torch.Tensor  # int8 [N, K]
    scale: torch.Tensor  # fp32 [N, 1]
    bias: torch.Tensor | None
    lora_a: torch.Tensor | None
    lora_b: torch.Tensor | None

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)  # int8 [N, K]
        self.register_buffer("scale", scale.reshape(-1, 1).float(), persistent=False)  # [N, 1]
        self.register_buffer("bias", bias, persistent=False)
        # Buffers rather than plain attributes, so a LoRA travels with its block when it is streamed.
        self.register_buffer("lora_a", None, persistent=False)
        self.register_buffer("lora_b", None, persistent=False)
        self.lora_scale = 1.0

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    def set_lora(self, a: torch.Tensor, b: torch.Tensor, scale: float = 1.0) -> None:
        self.lora_a, self.lora_b, self.lora_scale = a, b, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = comfy_kitchen.int8_linear(
            x, self.weight, self.scale, self.bias, x.dtype, convrot=True, convrot_groupsize=GROUP
        )
        lora_a, lora_b = self.lora_a, self.lora_b
        if lora_a is not None and lora_b is not None:
            out += ((x @ lora_a.t()) @ lora_b.t()) * self.lora_scale
        return out
