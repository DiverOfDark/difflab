"""The 20 B packed video+audio DiT: diffusers' model, the pruned checkpoint's curve AdaLN, turbo LoRA.

The released checkpoints are *pruned*: the 2688-dim timestep embedding and every AdaLN projection that
consumed it are gone, replaced by an 8-dim time **curve** (`adaln_t_table [1025, 8]`, linearly
interpolated at `t * 1024`) feeding `[96768, 8]` projections, with **no SiLU**. That removes ~13 B
parameters and is exact at inference, because `t` is deterministic. In diffusers' model the timestep
path is `temb = time_embedder(time_proj(timestep))`, so curve mode is a module swap: `time_proj`
becomes identity, `time_embedder` becomes the table lookup, and both AdaLN modules take 8 dims.

The turbo LoRAs were trained against the *unpruned* 2688-dim AdaLN input, which these checkpoints no
longer have, so that part cannot be merged. ComfyUI re-injects it from a bundled
`silu_t_emb_grid [1025, 2688]`: the delta is a pure function of `t`, so it is evaluated once per
timestep and added to the projection's output. Backbone LoRAs are applied in activation space by
`ConvRotLinear` (merging into int8 would round most of the delta away).
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import MiniMaxH3Transformer3DModel
from safetensors import safe_open
from torch import nn

from difflab import config as app_config
from difflab.core import ui
from difflab.core.hub import HubFile
from difflab.models import offload
from difflab.models.convrot import ConvRotLinear
from difflab.models.minimax_h3 import config, weights
from difflab.models.minimax_h3.attention import use_int8_attention

MODALITY_NUM = 3  # video 0, text 1, audio 2


def _bias(linear: nn.Linear) -> torch.Tensor | None:
    return None if linear.bias is None else linear.bias.float()


@dataclass
class SiluGrid:
    """Carries `silu(time_embedder(t))` from the timestep module to the AdaLN LoRAs.

    Deliberately **not** an `nn.Module`: holding the timestep module directly would register it as a
    child of all 50 blocks, and the block streamer would then restore its table to the CPU after every
    block's forward.
    """

    value: torch.Tensor | None = None


class CurveTimeEmbedding(nn.Module):
    """`t` in [0, 1] -> the checkpoint's 8-dim time curve, by linear interpolation of `adaln_t_table`.

    The 2688-dim `silu(time_embedder(t))` grid a turbo LoRA needs is stashed rather than returned:
    both are pure functions of `t`, and diffusers' forward threads only one tensor through.
    """

    table: torch.Tensor
    silu_grid: torch.Tensor | None

    def __init__(self, table: torch.Tensor, silu_grid: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("table", table.float(), persistent=False)
        self.register_buffer("silu_grid", None if silu_grid is None else silu_grid.float(), persistent=False)
        self.grid = SiluGrid()

    def _lerp(self, table: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        rows = table.shape[0]
        position = timestep.to(table.device, torch.float32).clamp(0.0, 1.0) * (rows - 1)
        low = position.floor().long().clamp(0, rows - 1)
        high = (low + 1).clamp(0, rows - 1)
        frac = (position - low.float()).unsqueeze(-1)
        return table[low] * (1.0 - frac) + table[high] * frac

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if self.silu_grid is not None:
            self.grid.value = self._lerp(self.silu_grid, timestep)
        return self._lerp(self.table, timestep)


class CurveAdaLayerNormModulation(nn.Module):
    """Block-level AdaLN projection on the 8-dim curve: same as diffusers' but without the SiLU."""

    lora_a: torch.Tensor | None
    lora_b: torch.Tensor | None

    def __init__(self, hidden_size: int, linear: nn.Linear, grid: SiluGrid) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = linear
        self.grid = grid
        # Buffers, so the LoRA travels with the block when it is streamed onto the GPU.
        self.register_buffer("lora_a", None, persistent=False)
        self.register_buffer("lora_b", None, persistent=False)
        self.lora_scale = 1.0

    def set_lora(self, a: torch.Tensor, b: torch.Tensor, scale: float = 1.0) -> None:
        """LoRA in the unpruned 2688-dim space; applied via the silu grid, never merged."""
        self.lora_a, self.lora_b, self.lora_scale = a, b, scale

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # The projection is fp16 in the checkpoint while the trunk is bf16, and mixing the two promotes
        # activations to fp32 (which then reach the int8 layers). ComfyUI runs curve AdaLN in fp32, so
        # compute there and hand the trunk bf16 — the same dtype the unpruned AdaLN would have produced.
        out = nn.functional.linear(temb.float(), self.linear.weight.float(), _bias(self.linear))
        if self.lora_a is not None and self.lora_b is not None:
            grid = self.grid.value
            if grid is None:
                raise RuntimeError("the AdaLN LoRA needs the silu time grid; load it with the checkpoint")
            delta = (grid.float() @ self.lora_a.float().t()) @ self.lora_b.float().t()
            out = out + delta * self.lora_scale
        modulation = out.to(torch.bfloat16)
        return tuple(modulation.view(-1, 6 * self.hidden_size).chunk(6, dim=-1))


class CurveAdaLayerNormOut(nn.Module):
    """Final norm, shift/scale per row, addressed by timestep index — 8-dim curve input."""

    lora_a: torch.Tensor | None
    lora_b: torch.Tensor | None

    def __init__(self, norm: nn.Module, linear: nn.Linear, grid: SiluGrid) -> None:
        super().__init__()
        self.norm = norm
        self.linear = linear
        self.grid = grid
        self.register_buffer("lora_a", None, persistent=False)
        self.register_buffer("lora_b", None, persistent=False)
        self.lora_scale = 1.0

    def set_lora(self, a: torch.Tensor, b: torch.Tensor, scale: float = 1.0) -> None:
        self.lora_a, self.lora_b, self.lora_scale = a, b, scale

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor) -> torch.Tensor:
        out = nn.functional.linear(temb.float(), self.linear.weight.float(), _bias(self.linear))
        if self.lora_a is not None and self.lora_b is not None:
            grid = self.grid.value
            if grid is not None:
                delta = (grid.float() @ self.lora_a.float().t()) @ self.lora_b.float().t()
                out = out + delta * self.lora_scale
        shift, scale = out.chunk(2, dim=-1)
        normed: torch.Tensor = self.norm(hidden_states)
        rows = timestep_indices.to(hidden_states.device)
        modulated: torch.Tensor = normed * (1 + scale[rows].unsqueeze(0).to(normed.dtype))
        return modulated + shift[rows].unsqueeze(0).to(normed.dtype)


@dataclass
class LoraWeights:
    """`W_eff = W + B @ A`, keyed by the checkpoint's module path (e.g. `blocks.3.attn.qkv_proj`)."""

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]
    scales: dict[str, float]

    @property
    def has_adaln(self) -> bool:
        return any("adaln" in module for module in self.pairs)


def load_lora(path: str | Path) -> LoraWeights:
    """Read either turbo format: bare `lora_A/lora_B` (no alpha) or `diffusion_model.`-prefixed kohya."""
    down: dict[str, torch.Tensor] = {}
    up: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in list(f.keys()):  # safe_open is not a mapping
            name = key.removeprefix("diffusion_model.")
            if name.endswith((".lora_A.weight", ".lora_down.weight")):
                down[name.removesuffix(".lora_A.weight").removesuffix(".lora_down.weight")] = f.get_tensor(key)
            elif name.endswith((".lora_B.weight", ".lora_up.weight")):
                up[name.removesuffix(".lora_B.weight").removesuffix(".lora_up.weight")] = f.get_tensor(key)
            elif name.endswith(".alpha"):
                alphas[name.removesuffix(".alpha")] = float(f.get_tensor(key).item())
    pairs = {module: (down[module], up[module]) for module in sorted(down.keys() & up.keys())}
    # Per module: alpha / rank, or 1.0 when the file has no alphas (alpha == rank, as in the v4 LoRA).
    scales = {module: alphas.get(module, pair[0].shape[0]) / pair[0].shape[0] for module, pair in pairs.items()}
    return LoraWeights(pairs=pairs, scales=scales)


def apply_lora(model: MiniMaxH3Transformer3DModel, lora: LoraWeights) -> int:
    """The base conversion split the fused `qkv_proj` into `to_q/to_k/to_v` and swapped the two halves
    of `mlp.fc1`; the LoRA's `B` has to follow both, while `A` is shared."""
    applied = 0
    for module, (a, b) in lora.pairs.items():
        target = _lora_target(module)
        if target is None:
            continue
        path, kind = target
        scale = lora.scales[module]
        try:
            owner = model.get_submodule(path)
        except AttributeError:
            continue
        if kind == "qkv":
            for name, part in zip(("to_q", "to_k", "to_v"), b.chunk(3, dim=0), strict=True):
                _set_lora(model.get_submodule(f"{path}.{name}"), a, part.contiguous(), scale)
                applied += 1
            continue
        if kind == "ff":
            gate, value = b.chunk(2, dim=0)
            b = torch.cat([value, gate], dim=0).contiguous()
        _set_lora(owner, a, b, scale)
        applied += 1
    return applied


def _lora_target(module: str) -> tuple[str, str] | None:
    """Checkpoint module path -> (diffusers module path, kind)."""
    if module.startswith("blocks."):
        index, _, rest = module.removeprefix("blocks.").partition(".")
        prefix = f"transformer_blocks.{index}"
    elif module.startswith("token_refiner.blocks."):
        index, _, rest = module.removeprefix("token_refiner.blocks.").partition(".")
        prefix = f"token_refiner.refiner_blocks.{index}"
    elif module == "final_layer.adaln_proj.linear":
        return "norm_out", "adaln"
    else:
        return None
    mapping = {
        "attn.qkv_proj": (f"{prefix}.attn", "qkv"),
        "attn.out_proj": (f"{prefix}.attn.to_out.0", "linear"),
        "mlp.fc1": (f"{prefix}.ff.net.0.proj", "ff"),
        "mlp.fc2": (f"{prefix}.ff.net.2", "linear"),
        "adaln_proj.linear": (f"{prefix}.adaln_proj", "adaln"),
    }
    return mapping.get(rest)


def _set_lora(module: nn.Module, a: torch.Tensor, b: torch.Tensor, scale: float) -> None:
    """int8 layers keep the LoRA in activation space; bf16 layers merge it.

    Merging into int8 would round most of the delta away, which is why ComfyUI bypasses there — but
    the token refiner is plain bf16, where the delta sits far above the ULP and merging is exact enough.
    """
    setter = getattr(module, "set_lora", None)
    if setter is not None:
        setter(a, b, scale)
        return
    if isinstance(module, nn.Linear):
        delta = (b.to(torch.float32) @ a.to(torch.float32)) * scale
        module.weight.data = (module.weight.data.float() + delta).to(module.weight.dtype)
        return
    raise TypeError(f"{type(module).__name__} cannot take a LoRA")


def load(
    weights_file: HubFile,
    lora_source: HubFile,
    device: str | torch.device = app_config.DEVICE,
    chunk_tokens: int = config.MATMUL_CHUNK_TOKENS,
) -> MiniMaxH3Transformer3DModel:
    """Build the DiT: int8 weights, curve-mode AdaLN, turbo LoRA, blocks streaming from CPU."""
    dit, lora_file, grid_file = weights_file.fetch(), lora_source.fetch(), config.SILU_TEMB_GRID.fetch()
    with ui.step(f"Loading DiT ({dit.stem})") as detail:
        checkpoint = weights.load_checkpoint(dit)
        with init_empty_weights():
            model = MiniMaxH3Transformer3DModel()  # defaults already match this checkpoint's shapes
        weights.swap_quantized_linears(model, checkpoint.quant)

        silu_grid = _read_silu_grid(grid_file)
        if checkpoint.adaln_t_table is None:
            raise RuntimeError(f"{dit.name} has no adaln_t_table: not a pruned curve-mode checkpoint")
        time_embedding = CurveTimeEmbedding(checkpoint.adaln_t_table, silu_grid)
        model.time_proj = nn.Identity()
        model.time_embedder = time_embedding
        _install_curve_adaln(model, checkpoint.curve, time_embedding.grid)

        missing, unexpected = model.load_state_dict(checkpoint.dense, strict=False, assign=True)
        # The curve modules were installed with their own weights, so their keys are absent by design.
        replaced = ("adaln_proj.linear", "norm_out.linear", "time_embedder.", "time_proj.")
        leftover = [key for key in missing if not any(token in key for token in replaced)]
        still_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        if unexpected or leftover or still_meta:
            raise RuntimeError(
                f"DiT: missing={sorted(leftover)[:6]} unexpected={sorted(unexpected)[:6]} meta={still_meta[:6]}"
            )
        if checkpoint.rope_inv_freq is not None:
            model.rope.inv_freq = checkpoint.rope_inv_freq.to(model.rope.inv_freq.dtype)

        lora = load_lora(lora_file)
        if lora.has_adaln and silu_grid is None:
            raise FileNotFoundError(
                f"{lora_file.name} carries AdaLN pairs in the unpruned 2688-dim space, which this pruned checkpoint "
                f"does not have. They are injected from {grid_file}, which is missing."
            )
        applied = apply_lora(model, lora)
        _chunk_feed_forwards(model, chunk_tokens)
        attentions = use_int8_attention(model)
        detail(f"{_int8_size(model):.1f} GB int8 · {applied} LoRA sites · {attentions} int8 attn")

    model.requires_grad_(False).eval()
    offload.stream_blocks(model.transformer_blocks, device)
    for name, child in model.named_children():
        if name != "transformer_blocks":
            child.to(device)
    return model


class ChunkedFeedForward(nn.Module):
    """Run a block's feed-forward a slice of tokens at a time.

    The SwiGLU intermediate is 28672 wide, so a 15 s clip at 0.8 MP (88k packed rows) materialises
    4.7 GB for it and another 2.4 GB for the gate/value product - per block, on a 24 GB card.
    Slicing the token axis bounds that transient; the result is identical because a feed-forward
    treats every row independently.
    """

    def __init__(self, inner: nn.Module, chunk_tokens: int) -> None:
        super().__init__()
        self.inner = inner
        self.chunk_tokens = chunk_tokens

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        *lead, dim = hidden_states.shape
        flat = hidden_states.reshape(-1, dim)
        if flat.shape[0] <= self.chunk_tokens:
            whole: torch.Tensor = self.inner(hidden_states)
            return whole
        out = torch.empty_like(flat)
        for start in range(0, flat.shape[0], self.chunk_tokens):
            out[start : start + self.chunk_tokens] = self.inner(flat[start : start + self.chunk_tokens])
        return out.reshape(*lead, dim)


def _chunk_feed_forwards(model: MiniMaxH3Transformer3DModel, chunk_tokens: int) -> None:
    """Wrap every feed-forward. Runs last, so the quantised-linear and LoRA paths still resolve."""
    for block in (*model.transformer_blocks, *model.token_refiner.refiner_blocks):
        block.ff = ChunkedFeedForward(block.ff, chunk_tokens)


def _install_curve_adaln(model: MiniMaxH3Transformer3DModel, curve: dict[str, torch.Tensor], grid: SiluGrid) -> None:
    for index, block in enumerate(model.transformer_blocks):
        prefix = f"transformer_blocks.{index}.adaln_proj.linear"
        block.adaln_proj = CurveAdaLayerNormModulation(model.config.hidden_size, _linear_from(curve, prefix), grid)
    model.norm_out = CurveAdaLayerNormOut(model.norm_out.norm, _linear_from(curve, "norm_out.linear"), grid)


def _linear_from(curve: dict[str, torch.Tensor], prefix: str) -> nn.Linear:
    weight, bias = curve[f"{prefix}.weight"], curve[f"{prefix}.bias"]
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=True, device="meta")
    linear.weight = nn.Parameter(weight, requires_grad=False)
    linear.bias = nn.Parameter(bias, requires_grad=False)
    return linear


def _read_silu_grid(path: str | Path) -> torch.Tensor | None:
    if not Path(path).exists():
        return None
    with safe_open(str(path), framework="pt", device="cpu") as f:
        grid: torch.Tensor = f.get_tensor("silu_t_emb_grid")
    return grid


def _int8_size(model: nn.Module) -> float:
    total = sum(m.weight.numel() for m in model.modules() if isinstance(m, ConvRotLinear))
    return total / 1024**3
