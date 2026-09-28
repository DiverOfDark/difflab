"""The H3 text encoder: Qwen3-VL-32B truncated to its first 50 layers.

This is not a chat model. ComfyUI feeds it raw segments with no template, no BOS/EOS and no final
norm, and takes the **raw layer-50 hidden state** as the conditioning (`[1, L, 5120]`). The checkpoint
has neither `model.norm` nor `lm_head`, so both are replaced with `Identity`.

Key mapping (checkpoint -> transformers 5): `model.layers.N.*` -> `model.language_model.layers.N.*`,
`model.embed_tokens.*` -> `model.language_model.embed_tokens.*`, `visual.*` -> `model.visual.*`. The
vision tower is bf16 and maps 1:1, including its fused `attn.qkv` (transformers reshapes it as
`(seq, 3, heads, -1)`, so the rows are `[q_all; k_all; v_all]` and must not be split).

All 350 attention/MLP projections are int8 ConvRot and become `ConvRotLinear` (24.5 GB stays 24.5 GB;
dequantizing to bf16 would need ~50 GB). `model.embed_tokens` is the one exception: it is int8 with a
single scalar scale and no rotation, so it is dequantized to bf16 once.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn
from transformers import PreTrainedTokenizerBase, Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from difflab import config as app_config
from difflab.core import ui
from difflab.models import offload
from difflab.models.convrot import ConvRotLinear, read_quant_meta
from difflab.models.minimax_h3 import config
from difflab.models.minimax_h3.weights import QuantSpec, swap_quantized_linears

# Qwen3-VL-32B, cut to the 50 layers the H3 checkpoint ships.
TEXT_CONFIG = {
    "hidden_size": 5120,
    "num_hidden_layers": 50,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 25600,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5000000.0,
    "rope_scaling": {"rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True},
}
VISION_CONFIG = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "out_hidden_size": 5120,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "num_position_embeddings": 2304,
    "deepstack_visual_indexes": [8, 16, 24],
    "in_channels": 3,
}
# The conditioning is this layer's output, counting the embedding as layer 0.
CONDITION_LAYER = 50

VISION_START, VISION_END, PAD_TOKEN = 151652, 151653, 151643
# Qwen3-VL normalizes images to [-1, 1] (mean/std 0.5), unlike Qwen2.5-VL's CLIP statistics.
IMAGE_MEAN = IMAGE_STD = (0.5, 0.5, 0.5)
PATCH_SIZE, MERGE_SIZE, TEMPORAL_PATCH = 16, 2, 2
MIN_PIXELS, MAX_PIXELS = 3136, 12845056

_RENAMES = (
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.layers.", "model.language_model.layers."),
    ("visual.", "model.visual."),
)


def rename_key(key: str) -> str:
    """Rename a checkpoint key, whether it is a full tensor path or a bare module path."""
    for source, target in _RENAMES:
        stem = source.rstrip(".")
        if key == stem or key.startswith(f"{stem}."):
            return target.rstrip(".") + key[len(stem) :]
    return key


def load(path: str | Path) -> nn.Module:
    """Build the truncated Qwen3-VL text encoder with int8 weights in place."""
    with ui.step("Loading text encoder (Qwen3-VL 32B, int8)") as detail:
        dense: dict[str, torch.Tensor] = {}
        quant: dict[str, QuantSpec] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())  # safe_open is not a mapping
            quantized = {k.removesuffix(".comfy_quant") for k in keys if k.endswith(".comfy_quant")}
            for layer in sorted(quantized):
                meta = read_quant_meta(f.get_tensor(f"{layer}.comfy_quant"))
                weight, scale = f.get_tensor(f"{layer}.weight"), f.get_tensor(f"{layer}.weight_scale")
                target = rename_key(layer)
                if meta.get("convrot"):
                    quant[target] = QuantSpec(weight, scale, meta)
                else:  # embed_tokens: scalar scale, no rotation
                    dense[f"{target}.weight"] = (weight.float() * scale.float()).to(torch.bfloat16)
            for key in keys:
                if any(key.startswith(f"{layer}.") for layer in quantized):
                    continue
                dense[rename_key(key)] = f.get_tensor(key)

        cfg = Qwen3VLConfig(text_config=TEXT_CONFIG, vision_config=VISION_CONFIG)
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration._from_config(cfg, attn_implementation="sdpa")
        # The checkpoint stops at layer 50: no final norm, and the vocab projection is never used.
        model.model.language_model.norm = nn.Identity()
        model.lm_head = nn.Identity()

        swap_quantized_linears(model, quant)
        missing, unexpected = model.load_state_dict(dense, strict=False, assign=True)
        if unexpected or missing:
            raise RuntimeError(f"text encoder: missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]}")
        detail(f"{_sizes(model)} · {len(quant)} int8 layers")
    frozen: nn.Module = model.requires_grad_(False).eval()
    return frozen


def _sizes(model: nn.Module) -> str:
    by_dtype: dict[torch.dtype, int] = {}
    for module in model.modules():
        if isinstance(module, ConvRotLinear):
            by_dtype[torch.int8] = by_dtype.get(torch.int8, 0) + module.weight.numel()
    for param in model.parameters():
        by_dtype[param.dtype] = by_dtype.get(param.dtype, 0) + param.numel() * param.element_size()
    names = {torch.int8: "int8", torch.bfloat16: "bf16", torch.float32: "fp32"}
    parts = [
        f"{size / 1024**3:.2f} GB {names.get(dtype, str(dtype))}"
        for dtype, size in sorted(by_dtype.items(), key=lambda item: -item[1])
        if size >= 0.005 * 1024**3
    ]
    return " + ".join(parts)


@dataclass
class Conditioning:
    """What the DiT consumes: layer-50 hidden states plus one modality tag per row."""

    embeds: torch.Tensor  # [1, L, 5120], fp32, unnormalized
    tags: torch.Tensor  # [L], text 1 / vision 0


def preprocess_image(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`[T, H, W, 3]` in [0, 1] -> (flattened patches `[grid_h*grid_w, 1536]`, `grid_thw [1, 3]`).

    Mirrors ComfyUI's `process_qwen2vl_images`, down to the bilinear resize without `align_corners`.

    `T` is 1 for a still, which is repeated to fill the patch, or 2 for a video block, whose two frames
    fill it between them — that pair is the only motion Qwen ever sees of a reference video, and it is
    why the grid stays `t=1` either way.
    """
    if image.shape[0] not in (1, TEMPORAL_PATCH):
        raise ValueError(f"a vision block is 1 frame or {TEMPORAL_PATCH}, got {image.shape[0]}")
    height, width = image.shape[1], image.shape[2]
    pixels = image.permute(0, 3, 1, 2).float()
    factor = PATCH_SIZE * MERGE_SIZE
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    resized = nn.functional.interpolate(pixels, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGE_MEAN, device=pixels.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=pixels.device).view(1, 3, 1, 1)
    normalized = (resized - mean) / std

    grid_h, grid_w = h_bar // PATCH_SIZE, w_bar // PATCH_SIZE
    stacked = normalized.repeat(TEMPORAL_PATCH, 1, 1, 1) if normalized.shape[0] == 1 else normalized
    patches = stacked.reshape(
        1, TEMPORAL_PATCH, 3, grid_h // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE, grid_w // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_h * grid_w, 3 * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE)
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long, device=pixels.device)
    return flat, grid_thw


@dataclass(frozen=True)
class PresentedReference:
    """A reference as Qwen is shown it, which is not how the DiT is shown it.

    An image becomes one `<Picture i>` vision block. A video becomes a `<Video k>` label followed by
    one block per *pair* of frames, each announced with the timestamp it sits at — Qwen sees the clip
    at 2 fps, while the DiT gets every latent frame the VAE produced.
    """

    kind: str  # "image" | "video" | "audio"
    frames: torch.Tensor | None  # [1, H, W, 3] for an image; [T, H, W, 3] sampled at 2 fps for a video
    timestamps: tuple[float, ...] = ()
    has_audio: bool = False  # a video whose soundtrack is also a reference gets an <Audio j> label

    @classmethod
    def image(cls, frames: torch.Tensor) -> "PresentedReference":
        return cls("image", frames)

    @classmethod
    def video(
        cls, frames: torch.Tensor, timestamps: Sequence[float] | None = None, has_audio: bool = False
    ) -> "PresentedReference":
        stamps = tuple(timestamps) if timestamps is not None else tuple(i / 2.0 for i in range(frames.shape[0]))
        return cls("video", frames, stamps, has_audio)

    @classmethod
    def audio(cls) -> "PresentedReference":
        """A standalone reference soundtrack. Qwen never hears it — it only learns that it is there."""
        return cls("audio", None)


@dataclass(frozen=True)
class VisionSpan:
    """One vision block inside the presentation: where its embeddings sit, and their patch grid."""

    start: int
    size: int
    grid: torch.Tensor  # [1, 3] (t, h, w) in patches


def modality_tags(seq_len: int, spans: Sequence[VisionSpan]) -> torch.Tensor:
    """Text rows tag 1, every vision block tag 0 — including its flanking vision start/end tokens."""
    tags = torch.ones(seq_len, dtype=torch.long)
    for span in spans:
        if span.size:
            tags[max(0, span.start - 1) : span.start + span.size + 1] = 0
    return tags


def mrope_position_ids(seq_len: int, spans: Sequence[VisionSpan], device: torch.device) -> torch.Tensor:
    """ComfyUI's 3-axis MRoPE ids `[3, 1, L]`: text runs sequentially, each vision span gets grid positions.

    Every span consumes `grid.max() // 2` positions on the shared clock rather than its row count, so
    each one leaves behind a gap; `offset` carries that difference forward, which is what lets several
    references share one presentation without their coordinates colliding.
    """
    spans = [span for span in spans if span.size]
    position_ids = torch.zeros((3, seq_len), device=device)
    if not spans:
        position_ids[:] = torch.arange(0, seq_len, device=device)
        return position_ids.unsqueeze(1)

    offset = 0
    for index, span in enumerate(sorted(spans, key=lambda item: item.start)):
        start, size, grid = span.start, span.size, span.grid
        end = start + size
        if index == 0:
            position_ids[:, :start] = torch.arange(0, start, device=device)
        span_length = int(grid.max()) // 2
        position_ids[:, end:] = torch.arange(
            span_length + start + offset, span_length + start + (seq_len - end) + offset, device=device
        )
        position_ids[0, start:end] = start + offset
        rows = int(grid[0][1]) // 2
        position_ids[1, start:end] = (
            torch.arange(start + offset, start + rows + offset, device=device)
            .unsqueeze(1)
            .repeat(1, math.ceil(size / rows))
            .flatten(0)[:size]
        )
        cols = int(grid[0][2]) // 2
        position_ids[2, start:end] = (
            torch.arange(start + offset, start + cols + offset, device=device)
            .unsqueeze(0)
            .repeat(math.ceil(size / cols), 1)
            .flatten(0)[:size]
        )
        offset += span_length - size
    return position_ids.unsqueeze(1)


def encode(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    image: torch.Tensor | None,
    device: str | torch.device = app_config.DEVICE,
    references: Sequence[PresentedReference] = (),
    last_image: torch.Tensor | None = None,
) -> Conditioning:
    """Build the H3 presentation and run it.

        fl2va:  `"<Picture 1>: " <vision> <prompt>`, and `"<Picture 2>: " <vision>` before the prompt for a last frame
        ref2va: per reference, in order, then the prompt:
                  image -> `"<Picture i>: " <vision>`
                  video -> `"<Video k>: "` then `"<T.T seconds>" <vision of 2 frames>` per pair

    No chat template and no special tokens — each text segment is tokenized on its own, exactly as
    ComfyUI does, and the vision embeddings are spliced between the vision start/end tokens. The
    ordinals are 1-based and counted per kind, with `image` taking `<Picture 1>` and `last_image` the
    next, so a prompt can address the references after them as `<Picture 3>`, `<Video 1>` and so on.
    """
    device = torch.device(device)
    inner = cast(nn.Module, model.model)
    language_model = cast(nn.Module, inner.language_model)
    embed_tokens = cast(nn.Module, language_model.embed_tokens)
    visual = cast(nn.Module, inner.visual)

    def text_embeds(text: str) -> torch.Tensor:
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        embedded: torch.Tensor = embed_tokens(ids)
        return embedded

    def token_embed(token_id: int) -> torch.Tensor:
        embedded: torch.Tensor = embed_tokens(torch.tensor([[token_id]], device=device))
        return embedded

    pieces: list[torch.Tensor] = []
    spans: list[VisionSpan] = []
    deepstack: list[torch.Tensor] | None = None

    def add_vision(block: torch.Tensor) -> None:
        nonlocal deepstack
        flat, grid = preprocess_image(block)
        vision = visual(flat.to(device, torch.float32), grid_thw=grid.to(device))
        # `last_hidden_state` is pre-merger (1152-wide); the merged 5120-wide tokens are the pooler output.
        vision_embeds = vision.pooler_output
        features = list(vision.deepstack_features or [])
        # One deepstack stream for the whole presentation: each block's features are appended in
        # sequence order, matching the order their rows appear in `visual_mask`.
        deepstack = (
            features
            if deepstack is None
            else [torch.cat([carried, incoming], dim=0) for carried, incoming in zip(deepstack, features, strict=True)]
        )
        pieces.append(token_embed(VISION_START))
        start = sum(piece.shape[1] for piece in pieces)
        pieces.append(vision_embeds.unsqueeze(0).to(pieces[0].dtype))
        pieces.append(token_embed(VISION_END))
        spans.append(VisionSpan(start=start, size=vision_embeds.shape[0], grid=grid))

    with torch.inference_mode():
        items = [PresentedReference.image(frame) for frame in (image, last_image) if frame is not None]
        items.extend(references)
        ordinals = {"image": 0, "video": 0, "audio": 0}
        for item in items:
            ordinals[item.kind] += 1
            if item.kind == "audio":
                pieces.append(text_embeds(f"<Audio {ordinals['audio']}>: "))
                continue
            assert item.frames is not None  # only an "audio" item has no frames, handled above
            if item.kind == "image":
                pieces.append(text_embeds(f"<Picture {ordinals['image']}>: "))
                add_vision(item.frames)
                continue
            # An odd frame count repeats its last frame, so every block is a full temporal patch.
            if item.has_audio:
                # A soundtrack is announced before the video it belongs to, with its own ordinal.
                ordinals["audio"] += 1
                pieces.append(text_embeds(f"<Audio {ordinals['audio']}>: "))
            frames, stamps = item.frames, list(item.timestamps)
            if frames.shape[0] % TEMPORAL_PATCH:
                frames = torch.cat([frames, frames[-1:]], dim=0)
                stamps = [*stamps, stamps[-1]]
            pieces.append(text_embeds(f"<Video {ordinals['video']}>: "))
            for index in range(0, frames.shape[0], TEMPORAL_PATCH):
                pieces.append(text_embeds(f"<{(stamps[index] + stamps[index + 1]) / 2.0:.1f} seconds>"))
                add_vision(frames[index : index + TEMPORAL_PATCH])
        pieces.append(text_embeds(prompt) if prompt else token_embed(PAD_TOKEN))

        inputs_embeds = torch.cat(pieces, dim=1)
        seq_len = inputs_embeds.shape[1]
        visual_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
        for span in spans:
            visual_mask[0, span.start : span.start + span.size] = True
        out = language_model(
            inputs_embeds=inputs_embeds,
            position_ids=mrope_position_ids(seq_len, spans, device),
            visual_pos_masks=visual_mask if spans else None,
            deepstack_visual_embeds=deepstack if spans else None,
            use_cache=False,
        )
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return Conditioning(embeds=hidden.float(), tags=modality_tags(seq_len, spans))


# H3's tokenizer_config appends these after Qwen's </think> (151668); ComfyUI registers them the same way.
# Unregistered, "<d>" is several plain-text tokens and H3 never sees a dialogue span.
MINIMAX_TOKENS = (
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
)
FIRST_MINIMAX_TOKEN = 151669


def uses_minimax_tokens(prompt: str) -> bool:
    """Whether `prompt` tokenizes differently since the MiniMax tokens were registered."""
    return any(token in prompt for token in MINIMAX_TOKENS)


def load_tokenizer(path: str | Path | None = None) -> PreTrainedTokenizerBase:
    """MiniMax's tokenizer, which already holds H3's tokens; registering them again is a no-op kept as a check."""
    path = path or config.TEXT_TOKENIZER.fetch() / "tokenizer"
    tokenizer: PreTrainedTokenizerBase = Qwen2Tokenizer.from_pretrained(str(path), local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(MINIMAX_TOKENS)})
    ids = tokenizer.convert_tokens_to_ids(list(MINIMAX_TOKENS))
    if ids != list(range(FIRST_MINIMAX_TOKEN, FIRST_MINIMAX_TOKEN + len(MINIMAX_TOKENS))):
        raise RuntimeError(f"the MiniMax tokens landed on {ids}; H3 was trained on 151669-151675")
    return tokenizer


def to_device(model: nn.Module, device: str | torch.device = app_config.DEVICE) -> nn.Module:
    """Resident on the GPU: the vision tower and the embedding table. Streaming: the 50 LM layers.

    The layer stack is 22.7 GB of int8 weights, so it cannot live on a 24 GB card beside its own
    activations; each layer is copied in for its forward and released afterwards (see `offload`).
    """
    target = torch.device(device)
    inner = cast(nn.Module, model.model)
    language_model = cast(nn.Module, inner.language_model)
    offload.stream_blocks(cast(nn.ModuleList, language_model.layers), target)
    cast(nn.Module, inner.visual).to(target)
    cast(nn.Module, language_model.embed_tokens).to(target)
    if hasattr(language_model, "rotary_emb"):
        cast(nn.Module, language_model.rotary_emb).to(target)
    return model
