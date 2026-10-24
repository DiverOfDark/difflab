"""Whether a model field's loader can take a file, judged from its header against the field's own default."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from difflab import tools
from difflab.core import hub

OK, WARN, BAD, UNKNOWN = "ok", "warn", "bad", "unknown"


@dataclass(frozen=True)
class Fit:
    level: str
    summary: str
    notes: list[str] = field(default_factory=list)
    like: str = ""  # the field this file seems to belong to instead


# What each loader drops or lets be absent. Every other tensor is placed strictly: any other difference in
# names from the field's default refuses the file.
_OPTIONAL: dict[str, tuple[str, ...]] = {
    "video-model": (
        "adaln_proj.linear",
        "norm_out.linear",
        "time_embedder.",
        "time_proj.",
        "adaln_basis",
        "adaln_mean",
    ),
    "ltx-text-encoder": ("hf_asset", "tokenizer_json"),
    "ltx-video-vae": ("per_channel_statistics.channel", "mean-of-stds"),
    "editor": ("__index_timestep_zero__", "logit_scale", "lm_head.weight"),
    "editor-21": ("comfy_attention.config",),
    "restorer": ("rope.rope.freqs", "negative_conditioning"),
    "video-vae": ("decoder.mask_token",),
}
_QUANT_PARTS = (".comfy_quant", ".weight_scale", ".weight_s_rel", ".weight_s_channel", ".weight_codebook")
_W4 = "asym_w4a8_int8"
_LORA_BASE = {"ltx-loras": "ltx-model", "video-loras": "video-model", "editor-21-loras": "editor-21"}
# LTX's loader looks every target up and raises on one it lacks; H3's and Qwen's skip it.
_LORA_STRICT = {"ltx-loras"}
_LORA_PARTS = (
    ".lora_A.weight",
    ".lora_A.default.weight",
    ".lora_down.weight",
    ".lora_down",
    ".lora_B.weight",
    ".lora_B.default.weight",
    ".lora_up.weight",
    ".lora_up",
    ".alpha",
)
_LORA_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "base_model.model.", "lora_unet_")
_FP8 = ("F8_E4M3", "F8_E5M2")


def fit(role: str, value: str) -> Fit:
    value = value.strip()
    if not value:
        return Fit(OK, "")
    try:
        if role == "vlm":
            return _vlm(value)
        file = hub.parse_file(value)
        if (verdict := _by_name(role, file.filename)) is not None:
            return verdict
        found = hub.header(file)
        if found is None:
            return Fit(UNKNOWN, "not checked: only a safetensors file can be read before it is downloaded")
        if role in _LORA_BASE:
            return _lora(role, found)
        if _is_lora(found):
            return Fit(BAD, "a LoRA, not a whole model: it goes in this model's LoRA list")
        reference = hub.header(hub.parse_file(default(role)))
        assert reference is not None
        return _against(role, found, reference)
    except (ValueError, KeyError, OSError) as problem:
        return Fit(UNKNOWN, f"not checked: {problem}")


def default(role: str) -> str:
    for tool in tools.TOOLS:
        for f in dataclasses.fields(tool.settings) if tool.settings is not None else ():
            if f.metadata.get("role") == role:
                return str(f.default).strip().split("\n")[0].split("*")[0]
    raise KeyError(f"no model field has the role {role!r}")


def _by_name(role: str, filename: str) -> Fit | None:
    if filename.endswith(".gguf"):
        return Fit(BAD, "GGUF: nothing here loads it; look for a safetensors or int8 ConvRot release")
    if role == "pixel-upscaler":
        return None if filename.endswith(".safetensors") else Fit(UNKNOWN, "not checked before download")
    if not filename.endswith(".safetensors"):
        return Fit(BAD, "this field loads safetensors only")
    return None


def _quant(header: hub.Header) -> dict[str, dict[str, Any]]:
    if header.quant:
        return header.quant
    # Older H3 packs list their int8 layers in the header metadata instead of a blob per layer.
    listed = json.loads(header.metadata.get("_quantization_metadata", "{}")).get("layers", {})
    return {layer: meta if isinstance(meta, dict) else {} for layer, meta in listed.items()}


def _runs(role: str, meta: dict[str, Any]) -> bool:
    if role == "video-vae":
        return meta == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    grouped = meta.get("convrot_groupsize", 256) == 256
    if meta.get("format") == _W4:
        return role.startswith("ltx-") and grouped
    if not meta and role == "video-model":
        return True  # `_quantization_metadata` names the layer without a format; the loader assumes ConvRot
    return (bool(meta.get("convrot")) and grouped) or role == "video-text-encoder"


def _shapes(header: hub.Header, quant: dict[str, dict[str, Any]]) -> dict[str, tuple[int, ...]]:
    """Each tensor's shape as the model holds it: a 4-bit layer packs two codes a byte."""
    out = {}
    for name, (_, shape) in header.tensors.items():
        if name.endswith(_QUANT_PARTS):
            continue
        layer = name.removesuffix(".weight")
        if quant.get(layer, {}).get("format") == _W4 and name.endswith(".weight"):
            shape = (*shape[:-1], shape[-1] * 2)
        out[name] = shape
    return out


def _against(role: str, found: hub.Header, reference: hub.Header) -> Fit:
    optional = _OPTIONAL.get(role, ())
    kept: Callable[[str], bool] = lambda name: not any(part in name for part in optional)  # noqa: E731
    quant, reference_quant = _quant(found), _quant(reference)
    shapes = {k: v for k, v in _shapes(found, quant).items() if kept(k)}
    wanted = {k: v for k, v in _shapes(reference, reference_quant).items() if kept(k)}
    missing, extra = sorted(wanted.keys() - shapes.keys()), sorted(shapes.keys() - wanted.keys())
    resized = sorted(k for k in wanted.keys() & shapes.keys() if wanted[k] != shapes[k])
    notes = _version(found, reference)
    if missing or extra or resized:
        problems = [
            *([f"lacks {len(missing)} tensors it needs: {_some(missing)}"] if missing else []),
            *([f"has {len(extra)} it does not know: {_some(extra)}"] if extra else []),
            *([f"{len(resized)} differ in shape: {_resized(resized, wanted, shapes)}"] if resized else []),
        ]
        guess = _looks_like(role, found)
        summary = "made for another field" if guess else "another architecture: its tensors don't match"
        if not missing and not extra and len(resized) <= 2:
            summary = "same layout, different size"
        mine = found.metadata.get("model_version")
        if mine and mine != reference.metadata.get("model_version", mine):
            summary = f"version {mine}: this field runs {reference.metadata['model_version']}"
        return Fit(BAD, summary, problems + notes, like=guess or "")
    refused = sorted(layer for layer, meta in quant.items() if not _runs(role, meta))
    if refused:
        formats = sorted({json.dumps(quant[layer], sort_keys=True) for layer in refused})
        return Fit(BAD, f"{len(refused)} layers are quantized in a format no kernel here runs", formats[:3] + notes)
    dense = {
        dtype
        for name, (dtype, _) in found.tensors.items()
        if not name.endswith(_QUANT_PARTS) and name.removesuffix(".weight") not in quant
    }
    allowed = {dtype for name, (dtype, _) in reference.tensors.items()}
    if fp8 := sorted(dense & set(_FP8) - allowed):
        return Fit(BAD, f"raw {'/'.join(fp8)} weights, which this field has no kernel for", notes)
    unpacked = len(reference_quant.keys() - quant.keys())
    if unpacked:
        notes.append(f"{unpacked} layers the default keeps in int8 are full precision here: more memory, slower")
        return Fit(WARN, f"loads, but {unpacked} layers are unquantized", notes)
    return Fit(OK, _describe(quant, dense), notes)


def _describe(quant: dict[str, dict[str, Any]], dense: set[str]) -> str:
    w4 = sum(meta.get("format") == _W4 for meta in quant.values())
    parts = [f"int8 ConvRot ({len(quant) - w4} layers)"] if len(quant) > w4 else []
    parts += [f"4-bit ConvRot ({w4})"] if w4 else []
    parts += [] if quant else ["/".join(sorted(dense & {"BF16", "F16", "F32", *_FP8}))]
    return "fits · " + " + ".join(parts)


def _version(found: hub.Header, reference: hub.Header) -> list[str]:
    mine, theirs = found.metadata.get("model_version"), reference.metadata.get("model_version")
    return [f"model_version {mine}; this field runs {theirs}"] if mine and theirs and mine != theirs else []


def _some(names: list[str]) -> str:
    return ", ".join(names[:3]) + (", …" if len(names) > 3 else "")


def _resized(names: list[str], wanted: dict[str, tuple[int, ...]], shapes: dict[str, tuple[int, ...]]) -> str:
    return ", ".join(f"{n} {list(shapes[n])} not {list(wanted[n])}" for n in names[:2]) + (
        ", …" if len(names) > 2 else ""
    )


def _looks_like(role: str, found: hub.Header) -> str | None:
    """The field whose default shares nearly every tensor name, among those already read once."""
    names = {n for n in found.tensors if not n.endswith(_QUANT_PARTS)}
    best, score = None, 0.0
    for other in _file_roles():
        if other == role:
            continue
        reference = hub.cached_header(hub.parse_file(default(other)))
        if reference is None:
            continue
        theirs = {n for n in reference.tensors if not n.endswith(_QUANT_PARTS)}
        overlap = len(names & theirs) / max(len(names | theirs), 1)
        if overlap > max(score, 0.9):
            best, score = other, overlap
    return best


def _file_roles() -> list[str]:
    roles = {
        str(f.metadata["role"])
        for tool in tools.TOOLS
        for f in (dataclasses.fields(tool.settings) if tool.settings is not None else ())
        if f.metadata.get("role") and not f.metadata.get("multiple")
    }
    return sorted(roles - {"vlm"})


def _is_lora(header: hub.Header) -> bool:
    return any(name.endswith(_LORA_PARTS[:4]) for name in header.tensors)


def _underscored(name: str) -> str:
    for prefix in _LORA_PREFIXES:
        name = name.removeprefix(prefix)
    return name.replace(".", "_")


def _lora(role: str, found: hub.Header) -> Fit:
    targets = {name.removesuffix(s) for name in found.tensors for s in _LORA_PARTS if name.endswith(s)}
    if not targets:
        return Fit(BAD, "holds no LoRA pairs: a whole model goes in the model field, not here")
    base = hub.header(hub.parse_file(default(_LORA_BASE[role])))
    assert base is not None
    modules = _base_modules(role, base)
    matched = {t for t in targets if _lora_key(role, t) in modules}
    notes = [f"made for {who}"] if (who := _trained_on(found)) else []
    if role == "ltx-loras":
        slots = any(name.startswith("diffusion_model.reference_slot_embedding.") for name in found.tensors)
        notes.append("carries MSR slot embeddings: it places references" if slots else "no MSR slot embeddings")
        targets -= {t for t in targets if "reference_slot_embedding" in t}
    if not matched:
        return Fit(BAD, "matches no layer of this model: a LoRA for another one?", [_some(sorted(targets)), *notes])
    if len(matched) < len(targets):
        unmatched = sorted(targets - matched)
        if len(matched) * 2 < len(targets):
            return Fit(BAD, f"only {len(matched)} of {len(targets)} layers are in this model: a LoRA for another one?",
                       [_some(unmatched), *notes])  # fmt: skip
        level = BAD if role in _LORA_STRICT else WARN
        verdict = "the loader refuses" if level == BAD else "those are skipped"
        return Fit(level, f"{len(unmatched)} of {len(targets)} layers aren't in this model: {verdict}",
                   [_some(unmatched), *notes])  # fmt: skip
    return Fit(OK, f"fits · {len(matched)} layers", notes)


def _base_modules(role: str, base: hub.Header) -> set[str]:
    names = {n.rsplit(".", 1)[0] for n in base.tensors if not n.endswith(_QUANT_PARTS)}
    if role == "ltx-loras":
        from difflab.models.ltx25 import weights  # not at the top: it imports diffusers

        names = {weights._dit_name(n + ".x").removesuffix(".x") for n in names if n.startswith(weights.DIT_PREFIX)}
    if role == "video-loras":
        from difflab.models.minimax_h3 import transformer  # not at the top: it imports diffusers

        paths = set()
        for n in names:
            if (target := transformer._lora_target(n)) is not None:
                path, kind = target
                paths |= {path, *(f"{path}.{q}" for q in ("to_q", "to_k", "to_v"))} if kind == "qkv" else {path}
        return paths
    if role == "editor-21-loras":
        halves = ("gate_layer", "proj")
        names |= {n.replace(".gate_up", f".{h}") for n in names if n.endswith(".gate_up") for h in halves}
    return {_underscored(n) for n in names}


def _lora_key(role: str, module: str) -> str | None:
    """What the role's loader looks a LoRA module up as."""
    if role == "video-loras":
        from difflab.models.minimax_h3 import transformer  # not at the top: it imports diffusers

        target = transformer._lora_target(transformer._module_name(module))
        return target[0] if target is not None else None
    return _underscored(module)


def _trained_on(header: hub.Header) -> str | None:
    for key in ("ss_base_model_version", "base_model", "modelspec.architecture"):
        if value := header.metadata.get(key):
            return value
    return None


def _vlm(value: str) -> Fit:
    pinned = hub.Checkpoint.parse(value)
    config = hub.config_json(pinned)
    reference = hub.config_json(hub.Checkpoint.parse(default("vlm")))
    method = (config.get("quantization_config") or {}).get("quant_method")
    if method != "exl3":
        return Fit(BAD, f"{method or 'unquantized'}, not EXL3: the VLM runs on ExLlamaV3; look for an -exl3 repo")
    if "vision_config" not in config:
        return Fit(BAD, "no vision tower: the VLM reads every image it is shown")
    mine, theirs = config.get("architectures") or [], reference.get("architectures") or []
    if mine != theirs:
        return Fit(WARN, f"{', '.join(mine)}, not {', '.join(theirs)}: loads only if ExLlamaV3 knows it",
                   ["the director also drafts with an MTP head, which not every model has"])  # fmt: skip
    return Fit(OK, f"fits · EXL3 {config.get('quantization_config', {}).get('bits', '')} bpw".replace("  ", " "))
