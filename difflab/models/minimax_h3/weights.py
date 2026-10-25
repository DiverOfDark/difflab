"""Load ComfyUI's MiniMax H3 checkpoints into diffusers' model classes.

diffusers ships no converter for these files, so the mapping lives here. Three kinds of tensors come
out of a checkpoint:

* **dense** — renamed and assigned straight onto the model (bf16/fp16/fp32);
* **quantized** — int8 ConvRot triples (`weight`, `weight_scale`, `comfy_quant`) that replace an
  `nn.Linear` with a `ConvRotLinear` (see `models/convrot.py`);
* **curve** — the pruned checkpoint's AdaLN tensors, whose input is an 8-dim time curve rather than
  diffusers' 2688-dim timestep embedding (see `transformer.py`).

Naming differences handled here: `blocks.` → `transformer_blocks.`, `token_refiner.blocks.` →
`token_refiner.refiner_blocks.`, the fused `attn.qkv_proj` → `to_q`/`to_k`/`to_v`, `attn.out_proj` →
`attn.to_out.0`, `q_norm`/`k_norm` → `norm_q`/`norm_k`, `mlp.fc1` → `ff.net.0.proj` **with its two
halves swapped** (the reference computes `silu(gate) * value` from `[gate; value]`, diffusers'
`SwiGLU` computes `value * silu(gate)` from `[value; gate]`), and `mlp.fc2` → `ff.net.2`.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from difflab.models.convrot import ConvRotLinear, read_quant_meta

# Modules outside the block stacks.
STANDALONE_RENAMES = {
    "video_patch_proj": "proj_in",
    "audio_patch_proj": "audio_proj_in",
    "condition_proj": "context_embedder",
    "final_layer.video_out": "proj_out",
    "final_layer.audio_out": "audio_proj_out",
    "final_layer.norm": "norm_out.norm",
    "token_refiner.final_norm": "token_refiner.final_norm",
}
# Modules inside a block, once the block prefix has been rewritten.
BLOCK_RENAMES = {
    "attn.out_proj": "attn.to_out.0",
    "attn.q_norm": "attn.norm_q",
    "attn.k_norm": "attn.norm_k",
    "mlp.fc2": "ff.net.2",
    "norm1": "norm1",
    "norm2": "norm2",
}
# Fused layers that need reshaping rather than a plain rename.
FUSED_QKV = "attn.qkv_proj"
FUSED_FF = "mlp.fc1"
# AdaLN projections: present in the checkpoint with an 8-dim input (curve mode).
CURVE_MODULES = {
    "adaln_proj.linear": "adaln_proj.linear",
    "final_layer.adaln_proj.linear": "norm_out.linear",
}


@dataclass
class QuantSpec:
    """An int8 ConvRot layer destined for one `nn.Linear` slot."""

    weight: torch.Tensor  # int8 [N, K]
    scale: torch.Tensor  # fp32 [N, 1]
    meta: dict[str, object]
    bias: torch.Tensor | None = None


@dataclass
class H3Checkpoint:
    dense: dict[str, torch.Tensor] = field(default_factory=dict)
    quant: dict[str, QuantSpec] = field(default_factory=dict)
    curve: dict[str, torch.Tensor] = field(default_factory=dict)
    adaln_t_table: torch.Tensor | None = None
    rope_inv_freq: torch.Tensor | None = None
    # The pruning basis: `curve = (temb_2688 - adaln_mean) @ adaln_basis.T`. Some repacks ship it, the
    # Comfy-Org ones do not; nothing here needs it, since the curve is read from `adaln_t_table`.
    adaln_basis: torch.Tensor | None = None
    adaln_mean: torch.Tensor | None = None

    @property
    def num_blocks(self) -> int:
        blocks = {
            k.split(".")[1] for k in (self.dense | self.quant | self.curve) if k.startswith("transformer_blocks.")
        }
        return len(blocks)


def split_qkv(
    weight: torch.Tensor, scale: torch.Tensor | None = None
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Fused `[q; k; v]` rows -> three tensors. Row splits are safe: the scale is per row and the
    ConvRot rotation runs along the input dimension."""
    chunks = weight.chunk(3, dim=0)
    scales = scale.chunk(3, dim=0) if scale is not None else (None, None, None)
    return [(w.contiguous(), s.contiguous() if s is not None else None) for w, s in zip(chunks, scales, strict=True)]


def swap_swiglu_halves(
    weight: torch.Tensor, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`[gate; value]` -> `[value; gate]`, matching diffusers' SwiGLU row order."""
    gate, value = weight.chunk(2, dim=0)
    out = torch.cat([value, gate], dim=0).contiguous()
    if scale is None:
        return out, None
    gate_s, value_s = scale.chunk(2, dim=0)
    return out, torch.cat([value_s, gate_s], dim=0).contiguous()


def _target_prefix(key: str) -> tuple[str, str] | None:
    """A checkpoint key -> (diffusers module path, leaf)."""
    for source, target in STANDALONE_RENAMES.items():
        if key.startswith(f"{source}."):
            return target, key[len(source) + 1 :]
    for source, target in (
        ("blocks.", "transformer_blocks."),
        ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    ):
        if key.startswith(source):
            rest = key[len(source) :]
            index, _, tail = rest.partition(".")
            if not index.isdigit():
                continue
            for module_source, module_target in BLOCK_RENAMES.items():
                if tail.startswith(f"{module_source}."):
                    return f"{target}{index}.{module_target}", tail[len(module_source) + 1 :]
    return None


def load_checkpoint(path: str | Path) -> H3Checkpoint:
    result = H3Checkpoint()
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())  # safe_open is not a mapping: keys() is its only listing
        key_set = set(keys)
        # Two ways a repack records its int8 layers: a per-layer `comfy_quant` blob (Comfy-Org's own
        # files) or one `_quantization_metadata` JSON in the file header (Star Ultimate Converter's).
        header_meta = _header_quant_meta(f.metadata())
        quant_layers = {k.removesuffix(".comfy_quant") for k in keys if k.endswith(".comfy_quant")}
        quant_layers |= {layer for layer in header_meta if f"{layer}.weight_scale" in key_set}

        for layer in sorted(quant_layers):
            meta = (
                read_quant_meta(f.get_tensor(f"{layer}.comfy_quant"))
                if f"{layer}.comfy_quant" in key_set
                else header_meta[layer]
            )
            spec = QuantSpec(
                weight=f.get_tensor(f"{layer}.weight"),
                scale=f.get_tensor(f"{layer}.weight_scale"),
                meta=meta,
            )
            _place_layer(result, layer, spec.weight, spec.scale, spec.meta)

        quant_suffixes = {"weight", "weight_scale", "comfy_quant"}
        for key in keys:
            owner = next((layer for layer in quant_layers if key.startswith(f"{layer}.")), None)
            if owner is not None:
                suffix = key.removeprefix(f"{owner}.")
                # Some checkpoints quantize condition_proj and both final_layer outputs with their biases.
                # A fused layer's bias would need splitting with it; refuse that rather than drop it.
                target = _target_prefix(f"{owner}.weight")
                if suffix == "bias" and target is not None and target[0] in result.quant:
                    result.quant[target[0]].bias = f.get_tensor(key)
                elif suffix not in quant_suffixes:
                    raise KeyError(f"unexpected tensor under quantized layer {owner}: {key}")
                continue
            tensor = f.get_tensor(key)
            if key == "adaln_t_table":
                result.adaln_t_table = tensor
            elif key == "adaln_basis":
                result.adaln_basis = tensor
            elif key == "adaln_mean":
                result.adaln_mean = tensor
            elif key == "rope.inv_freq":
                result.rope_inv_freq = tensor
            else:
                _place_dense(result, key, tensor)
    return result


def _header_quant_meta(metadata: dict[str, str] | None) -> dict[str, dict[str, object]]:
    """Per-layer quant metadata carried in the safetensors header instead of `comfy_quant` tensors."""
    blob = (metadata or {}).get("_quantization_metadata")
    if blob is None:
        return {}
    layers = json.loads(blob).get("layers", {})
    return {layer: dict(meta) for layer, meta in layers.items()}


def _place_layer(
    result: H3Checkpoint, layer: str, weight: torch.Tensor, scale: torch.Tensor, meta: dict[str, object]
) -> None:
    if layer.endswith(FUSED_QKV):
        base = _require_prefix(layer.removesuffix(f".{FUSED_QKV}") + ".attn")
        for name, (w, s) in zip(("to_q", "to_k", "to_v"), split_qkv(weight, scale), strict=True):
            assert s is not None
            result.quant[f"{base}.{name}"] = QuantSpec(w, s, meta)
    elif layer.endswith(FUSED_FF):
        base = _require_prefix(layer.removesuffix(f".{FUSED_FF}"))
        w, s = swap_swiglu_halves(weight, scale)
        assert s is not None
        result.quant[f"{base}.ff.net.0.proj"] = QuantSpec(w, s, meta)
    else:
        target = _target_prefix(f"{layer}.weight")
        if target is None:
            raise KeyError(f"unmapped quantized layer: {layer}")
        result.quant[target[0]] = QuantSpec(weight, scale, meta)


def _place_dense(result: H3Checkpoint, key: str, tensor: torch.Tensor) -> None:
    # Longest source first: `final_layer.adaln_proj.linear` must win over the per-block `adaln_proj.linear`.
    for source in sorted(CURVE_MODULES, key=len, reverse=True):
        if key.startswith(f"{source}.") or f".{source}." in key:
            block = key.split(f"{source}.")[0]
            prefix = f"transformer_blocks.{block.split('.')[1]}." if block.startswith("blocks.") else ""
            result.curve[f"{prefix}{CURVE_MODULES[source]}.{key.rsplit('.', 1)[1]}"] = tensor
            return
    if key.endswith(f"{FUSED_QKV}.weight"):
        base = _require_prefix(key.removesuffix(f".{FUSED_QKV}.weight") + ".attn")
        for name, (w, _) in zip(("to_q", "to_k", "to_v"), split_qkv(tensor), strict=True):
            result.dense[f"{base}.{name}.weight"] = w
        return
    if key.endswith(f"{FUSED_FF}.weight"):
        base = _require_prefix(key.removesuffix(f".{FUSED_FF}.weight"))
        w, _ = swap_swiglu_halves(tensor)
        result.dense[f"{base}.ff.net.0.proj.weight"] = w
        return
    mapped = _target_prefix(key)
    if mapped is None:
        raise KeyError(f"unmapped tensor: {key}")
    module, leaf = mapped
    result.dense[f"{module}.{leaf}"] = tensor


def _require_prefix(key: str) -> str:
    """The diffusers module path for a block-level module path like `blocks.3.attn`."""
    mapped = _target_prefix(f"{key}.weight")
    if mapped is not None:
        return mapped[0].rsplit(".", 1)[0] if mapped[1] == "weight" else mapped[0]
    if key.startswith("blocks."):
        return f"transformer_blocks.{key.split('.', 2)[1]}" + (f".{key.split('.', 2)[2]}" if key.count(".") > 1 else "")
    if key.startswith("token_refiner.blocks."):
        parts = key.split(".")
        return f"token_refiner.refiner_blocks.{parts[2]}" + (f".{'.'.join(parts[3:])}" if len(parts) > 3 else "")
    raise KeyError(f"unmapped module: {key}")


def swap_quantized_linears(model: nn.Module, specs: dict[str, QuantSpec]) -> int:
    for path, spec in specs.items():
        parent_path, _, name = path.rpartition(".")
        parent = model.get_submodule(parent_path)
        existing = getattr(parent, name)
        bias = spec.bias if spec.bias is not None else getattr(existing, "bias", None)
        if bias is not None and bias.is_meta:
            bias = None
        setattr(parent, name, ConvRotLinear(spec.weight, spec.scale, bias))
    return len(specs)
