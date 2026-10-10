"""LoRA files, in every naming they ship in, and how a pair lands on a layer."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


@dataclass
class LoraWeights:
    """`W_eff = W + B @ A`, keyed by module path as the model's `rename` leaves it."""

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]
    scales: dict[str, float]


_DOWN = (".lora_A.weight", ".lora_A.default.weight", ".lora_down.weight", ".lora_down")
_UP = (".lora_B.weight", ".lora_B.default.weight", ".lora_up.weight", ".lora_up")


def read(path: str | Path, rename: Callable[[str], str]) -> LoraWeights:
    """Comfy's, kohya's, PEFT's `.default` and diffusers' key suffixes; `rename` maps the module path."""
    down: dict[str, torch.Tensor] = {}
    up: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in list(f.keys()):  # safe_open is not a mapping
            if suffix := next((s for s in _DOWN if key.endswith(s)), None):
                down[rename(key.removesuffix(suffix))] = f.get_tensor(key)
            elif suffix := next((s for s in _UP if key.endswith(s)), None):
                up[rename(key.removesuffix(suffix))] = f.get_tensor(key)
            elif key.endswith(".alpha"):
                alphas[rename(key.removesuffix(".alpha"))] = float(f.get_tensor(key).item())
    pairs = {module: (down[module], up[module]) for module in sorted(down.keys() & up.keys())}
    # Per module: alpha / rank, or 1.0 when the file has no alphas (alpha == rank).
    scales = {module: alphas.get(module, pair[0].shape[0]) / pair[0].shape[0] for module, pair in pairs.items()}
    return LoraWeights(pairs=pairs, scales=scales)


def attach(module: nn.Module, a: torch.Tensor, b: torch.Tensor) -> None:
    """int8 layers keep the LoRA in activation space; bf16 layers merge it.

    Merging into int8 would round most of the delta away, which is why ComfyUI bypasses there — but in a
    bf16 layer the delta sits far above the ULP and merging is exact enough.
    """
    adder = getattr(module, "add_lora", None)
    if adder is not None:
        adder(a, b)
        return
    if isinstance(module, nn.Linear):
        delta = b.to(torch.float32) @ a.to(torch.float32)
        module.weight.data = (module.weight.data.float() + delta.to(module.weight.device)).to(module.weight.dtype)
        return
    raise TypeError(f"{type(module).__name__} cannot take a LoRA")
