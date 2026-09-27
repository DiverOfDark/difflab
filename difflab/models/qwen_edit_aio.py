"""Direct loader for the single-file "AIO" checkpoint (ComfyUI CheckpointLoaderSimple).

The transformer, text encoder and VAE are read straight from the .safetensors file. Nothing is converted
on disk and a full bf16 copy is never materialised - it would be twice the size of a model that already
has to be streamed through a 24 GB card. fp8 weights stay fp8 in memory and diffusers' layerwise-casting
hooks upcast each Linear/Conv/Embedding to bf16 only while it runs, which is what ComfyUI does on GPUs
without fp8 matmul; the modules kept out of the hooks (norms, in/out projections) are a small fraction
and are upcast once at load.

`build_*(state_dict)` and `load_*(path)` are split so the assembly can be checked against the file
header, on meta tensors, without reading any weights.
"""

from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from diffusers.hooks import apply_layerwise_casting
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
from safetensors import safe_open
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

from difflab import config
from difflab.core import ui
from difflab.models.processors import ImageOnlyQwen2VLProcessor

STORAGE_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.bfloat16

TRANSFORMER_PREFIX = "model.diffusion_model."
TEXT_ENCODER_PREFIX = "text_encoders.qwen25_7b.transformer."
VAE_PREFIX = "vae."

# Configs of Qwen/Qwen-Image-Edit-2511, the base of the AIO fine-tune. zero_cond_t is the 2511
# feature the AIO marks with its `__index_timestep_zero__` tensor.
TRANSFORMER_CONFIG = {
    "patch_size": 2,
    "in_channels": 64,
    "out_channels": 16,
    "num_layers": 60,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 3584,
    "guidance_embeds": False,
    "axes_dims_rope": [16, 56, 56],
    "zero_cond_t": True,
}
TEXT_ENCODER_CONFIG = {
    "text_config": {
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
        "max_position_embeddings": 128000,
        "max_window_layers": 28,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "rope_scaling": {"type": "default", "rope_type": "default", "mrope_section": [16, 24, 24]},
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "use_sliding_window": False,
        "sliding_window": None,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    },
    "vision_config": {
        "depth": 32,
        "hidden_size": 1280,
        "intermediate_size": 3420,
        "num_heads": 16,
        "out_hidden_size": 3584,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "window_size": 112,
        "fullatt_block_indexes": [7, 15, 23, 31],
        "tokens_per_second": 2,
        "hidden_act": "silu",
        "in_channels": 3,
    },
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "tie_word_embeddings": False,
}
SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
IMAGE_PROCESSOR_CONFIG = {
    "min_pixels": 3136,
    "max_pixels": 12845056,
    "patch_size": 14,
    "merge_size": 2,
    "temporal_patch_size": 2,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
}

# ComfyUI stores the text encoder in the legacy HF layout; transformers 5 nests it one level deeper.
_TEXT_ENCODER_RENAMES = (
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.layers.", "model.language_model.layers."),
    ("model.norm.", "model.language_model.norm."),
    ("visual.", "model.visual."),
)


def read_prefix(path: str | Path, prefix: str, drop: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    """Read every tensor under `prefix` (prefix stripped), in its on-disk dtype."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        # safe_open is not a mapping: keys() is its only way to list tensors.
        keys = [k for k in f.keys() if k.startswith(prefix) and k[len(prefix) :] not in drop]  # noqa: SIM118
        if not keys:
            raise KeyError(f"{path} has no tensors under '{prefix}'")
        return {k[len(prefix) :]: f.get_tensor(k) for k in keys}


def build_transformer(state_dict: dict[str, torch.Tensor]) -> QwenImageTransformer2DModel:
    with init_empty_weights():
        model = QwenImageTransformer2DModel.from_config(TRANSFORMER_CONFIG)
    _assign(model, state_dict)
    _setup_fp8(model)
    return _freeze(model)


def build_text_encoder(state_dict: dict[str, torch.Tensor]) -> Qwen2_5_VLForConditionalGeneration:
    state_dict = {_rename_text_encoder_key(k): v for k, v in state_dict.items()}
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration._from_config(
            Qwen2_5_VLConfig(**TEXT_ENCODER_CONFIG), attn_implementation="sdpa"
        )
    _assign(model, state_dict, allow_missing=("lm_head.weight",))
    # The AIO has no lm_head, and the pipeline only reads hidden states: drop the vocab projection.
    model.lm_head = torch.nn.Identity()
    _setup_fp8(model)
    return _freeze(model)


def build_vae(state_dict: dict[str, torch.Tensor]) -> AutoencoderKLQwenImage:
    with init_empty_weights():
        model = AutoencoderKLQwenImage()  # defaults == Qwen-Image VAE config
    _assign(model, convert_wan_vae_to_diffusers(state_dict))
    return _freeze(model)


def load_transformer(path: str | Path) -> QwenImageTransformer2DModel:
    with ui.step("Loading transformer") as detail:
        model = build_transformer(read_prefix(path, TRANSFORMER_PREFIX, drop=("__index_timestep_zero__",)))
        detail(_sizes(model))
    return model


def load_text_encoder(path: str | Path) -> Qwen2_5_VLForConditionalGeneration:
    with ui.step("Loading text encoder") as detail:
        model = build_text_encoder(read_prefix(path, TEXT_ENCODER_PREFIX, drop=("logit_scale",)))
        detail(_sizes(model))
    return model


def load_vae(path: str | Path) -> AutoencoderKLQwenImage:
    with ui.step("Loading VAE") as detail:
        model = build_vae(read_prefix(path, VAE_PREFIX))
        detail(_sizes(model))
    return model


def load_processor() -> tuple[Qwen2Tokenizer, ImageOnlyQwen2VLProcessor]:
    tokenizer = Qwen2Tokenizer.from_pretrained(config.TOKENIZER.fetch() / "tokenizer", local_files_only=True)
    processor = ImageOnlyQwen2VLProcessor(
        image_processor=Qwen2VLImageProcessorPil(**IMAGE_PROCESSOR_CONFIG),
        tokenizer=tokenizer,
        video_processor=None,
    )
    return tokenizer, processor


def load_scheduler() -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)


def _rename_text_encoder_key(key: str) -> str:
    for old, new in _TEXT_ENCODER_RENAMES:
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def _assign(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], allow_missing: tuple[str, ...] = ()) -> None:
    """Put checkpoint tensors into a meta-initialized model as-is (no copy, no dtype change)."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    missing = set(missing) - set(allow_missing)
    if missing or unexpected:
        raise RuntimeError(
            f"{type(model).__name__}: checkpoint mismatch; missing={sorted(missing)[:10]} unexpected={unexpected[:10]}"
        )


def _freeze[M: torch.nn.Module](model: M) -> M:
    """Inference only. Without this, autograd would keep every per-layer bf16 upcast alive."""
    return model.requires_grad_(False).eval()


def _setup_fp8(model: torch.nn.Module) -> None:
    """fp8 Linear/Conv/Embedding weights get per-layer bf16 upcast hooks; every other fp8 tensor becomes bf16 now."""
    # "auto" skips norms, patch/position embeddings and in/out projections: they compute in bf16.
    apply_layerwise_casting(
        model, storage_dtype=STORAGE_DTYPE, compute_dtype=COMPUTE_DTYPE, skip_modules_pattern="auto"
    )
    for module in model.modules():
        registry = getattr(module, "_diffusers_hook", None)
        if registry is not None and registry.get_hook("layerwise_casting") is not None:
            continue
        for param in module.parameters(recurse=False):
            if param.dtype == STORAGE_DTYPE:
                param.data = param.data.to(COMPUTE_DTYPE)


_DTYPE_NAMES = {torch.float8_e4m3fn: "fp8", torch.bfloat16: "bf16", torch.float32: "fp32"}


def _sizes(model: torch.nn.Module) -> str:
    """Weight memory per dtype, largest first, e.g. '19.01 GB fp8 + 0.04 GB bf16'."""
    by_dtype: dict[torch.dtype, int] = {}
    for p in model.parameters():
        by_dtype[p.dtype] = by_dtype.get(p.dtype, 0) + p.numel() * p.element_size()
    parts = [
        f"{size / 1024**3:.2f} GB {_DTYPE_NAMES.get(dtype, str(dtype))}"
        for dtype, size in sorted(by_dtype.items(), key=lambda item: -item[1])
        if size >= 0.005 * 1024**3
    ]
    return " + ".join(parts)
