"""The two MiniMax H3 VAEs, loaded from ComfyUI checkpoints into diffusers' classes.

Video VAE: causal 3D CNN encoder + 36-block ViT decoder, 16x spatial / 4x temporal. Key differences
from diffusers' naming: `decoder.x_embedder` -> `decoder.proj_in`, the fused `attn.to_qkv` splits into
`to_q`/`to_k`/`to_v` **per head** (see `split_interleaved_qkv`; splitting by row thirds instead looks
plausible and quietly destroys the output), `attn.to_out` -> `attn.to_out.0`, `ff.w1` ->
`ff.net.0.proj` **with its halves swapped** (ComfyUI computes `silu(gate) * value` from
`[gate; value]`, diffusers' SwiGLU wants `[value; gate]`), `ff.w2` -> `ff.net.2`, and
`encoder.down.N.block.M` -> `encoder.down_blocks.N.resnets.M`
(`nin_shortcut` -> `conv_shortcut`, `downsample.conv` -> `downsamplers.0.conv`). The checkpoint's
`decoder.mask_token` is all zeros, which is exactly what diffusers builds, so it is dropped.

The checkpoint is Comfy-Org's int8 one: the decoder's 144 block linears (`to_qkv`, `to_out`, `w1`, `w2`
in each of 36 blocks) are ConvRot int8 with per-row scales, the same format as the DiT, and run as W8A8
through the same fused kernel (`convrot.ConvRotLinear`). Those linears are nearly all of the decode's
work - it is compute-bound at ~76% of this card's fp16 peak - and the 3090 does int8 at four times its
fp16 rate. Everything else is stored at full precision (fp32; the encoder convs fp16), and rounded to
fp16 it is exactly the fp16 checkpoint, so it is cast back: the encoder and the decoder compute in fp16
as before. Two things do differ from the fp16 file. The quantized weights (~0.8% relative error per
matrix), and the latent mean/std, read here at their fp32 originals - which are ComfyUI's own
constants; the fp16 file had them rounded by up to 4e-4. Decoded frames: 58.9 dB PSNR against the fp16
VAE, 0.13/255 mean difference, at 1.7x the speed.

Audio VAE: DAC encoder + BigVGAN decoder, 40 latents/s -> 32 kHz stereo. diffusers keeps the
weight-norm parametrization (`weight_g`/`weight_v`) while the checkpoint stores fused weights, so the
pair is reconstructed on load (`g = ||v||` over every dimension but the first). It stays fp32: the
Snake activations degrade audibly in bf16.

Both checkpoints carry per-channel `latents_mean` / `latents_std` instead of a scalar scaling factor.
"""

import json
import re
import types
from collections.abc import Callable
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from safetensors import safe_open

from difflab.core import ui
from difflab.models.convrot import GROUP, ConvRotLinear, read_quant_meta
from difflab.models.minimax_h3 import config
from difflab.models.minimax_h3.weights import QuantSpec, swap_quantized_linears

DROPPED_VIDEO_KEYS = frozenset({"latents_mean", "latents_std", "decoder.mask_token"})
DROPPED_AUDIO_KEYS = frozenset({"latents_mean", "latents_std"})

# The video VAE encodes and decodes in ImageNet-normalized pixel space, not [0, 1] or [-1, 1].
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_model_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """`[B, 3, H, W]` in [0, 1] -> the ImageNet-normalized space the VAE expects."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=pixels.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=pixels.device).view(1, 3, 1, 1)
    return (pixels.float() - mean) / std


def from_model_pixels(sample: torch.Tensor) -> torch.Tensor:
    """Decoder output `[1, 3, T, H, W]` (ImageNet space) -> frames `[T, H, W, 3]` in [0, 1].

    Converted sixteen frames at a time into one preallocated buffer. Widening the whole output to
    float32 at once, where it sits, doubles it - 9 GB of VRAM at 2 MP, asked for while the decoder
    still holds its own output, which is where a 2 MP decode ran out. And `(x.float() * std + mean)
    .clamp()` is four operators and four full copies, so doing it all at once on the CPU instead costs
    several times 9 GB of host memory for an instant. In slices, it costs the size of the result.
    """
    _, _, count, height, width = sample.shape
    out = torch.empty(count, height, width, 3, dtype=torch.float32)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=sample.device).view(1, 1, 1, 3)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=sample.device).view(1, 1, 1, 3)
    for start in range(0, count, 16):
        piece = sample[0, :, start : start + 16].permute(1, 2, 3, 0).float()
        # Out of place: from an fp32 decoder `.float()` is the inference tensor itself, not a copy.
        out[start : start + 16] = (piece * std + mean).clamp_(0, 1).cpu()
    return out


def _read(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        return {key: f.get_tensor(key) for key in f.keys()}, metadata  # noqa: SIM118


def _latent_stats(tensors: dict[str, torch.Tensor]) -> tuple[list[float], list[float]]:
    return tensors["latents_mean"].float().tolist(), tensors["latents_std"].float().tolist()


VIDEO_DECODER_HEADS = 32  # diffusers' decoder_num_attention_heads; head_dim 64


def split_interleaved_qkv(tensor: torch.Tensor, num_heads: int) -> list[torch.Tensor]:
    """Split a fused QKV projection whose rows are grouped **per head**.

    ComfyUI's decoder does `qkv.view(batch, seq, -1, 3 * head_dim).chunk(3, dim=-1)`, i.e. the rows run
    `[q_0 k_0 v_0 | q_1 k_1 v_1 | ...]` per head rather than `[q_all; k_all; v_all]`. Splitting by row
    thirds instead silently scrambles attention (it still concatenates back to the same matrix, so a
    concat check cannot catch it).
    """
    per_head = tensor.shape[0] // num_heads  # 3 * head_dim
    head_dim = per_head // 3
    grouped = tensor.reshape(num_heads, per_head, *tensor.shape[1:])
    return [
        grouped[:, i * head_dim : (i + 1) * head_dim].reshape(num_heads * head_dim, *tensor.shape[1:]).contiguous()
        for i in range(3)
    ]


def convert_video_vae(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """ComfyUI video-VAE keys -> diffusers keys.

    An int8 layer's `weight_scale` holds one scale per output row, so it goes through exactly what its
    weight does: split per head for `to_qkv`, halves swapped for `w1`.
    """
    out: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if key in DROPPED_VIDEO_KEYS:
            continue
        name = key.replace("decoder.x_embedder.", "decoder.proj_in.")
        name = re.sub(r"encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", name)
        name = re.sub(r"encoder\.down\.(\d+)\.downsample\.conv\.", r"encoder.down_blocks.\1.downsamplers.0.conv.", name)
        name = name.replace("nin_shortcut", "conv_shortcut")

        if ".attn.to_qkv." in name:
            base, _, leaf = name.partition(".attn.to_qkv.")
            for proj, part in zip(
                ("to_q", "to_k", "to_v"), split_interleaved_qkv(tensor, VIDEO_DECODER_HEADS), strict=True
            ):
                out[f"{base}.attn.{proj}.{leaf}"] = part
            continue
        if ".ff.w1." in name:
            gate, value = tensor.chunk(2, dim=0)
            out[name.replace(".ff.w1.", ".ff.net.0.proj.")] = torch.cat([value, gate], dim=0).contiguous()
            continue
        name = re.sub(r"\.attn\.to_out\.(weight|weight_scale|bias)$", r".attn.to_out.0.\1", name)
        name = name.replace(".ff.w2.", ".ff.net.2.")
        out[name] = tensor
    return out


def convert_audio_vae(tensors: dict[str, torch.Tensor], expected: set[str]) -> dict[str, torch.Tensor]:
    """ComfyUI audio-VAE keys -> diffusers keys, re-creating the weight-norm pairs."""
    out: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if key in DROPPED_AUDIO_KEYS:
            continue
        if f"{key}_g" in expected and f"{key}_v" in expected:
            dims = tuple(range(1, tensor.ndim))
            out[f"{key}_g"] = tensor.norm(dim=dims, keepdim=True)
            out[f"{key}_v"] = tensor
        else:
            out[key] = tensor
    return out


def load_video_vae(path: str | Path) -> AutoencoderKLMiniMaxH3:
    with ui.step("Loading video VAE") as detail:
        tensors, metadata = _read(path)
        mean, std = _latent_stats(tensors)
        raw = json.loads(metadata.get("minimax_h3_video_vae", "{}"))
        quantized = _quantized_layers(tensors)
        converted = convert_video_vae({key: _compute_dtype(key, tensor) for key, tensor in tensors.items()})
        specs = {
            key.removesuffix(".weight_scale"): QuantSpec(
                converted.pop(key.removesuffix("_scale")), converted.pop(key), quantized
            )
            for key in [key for key in converted if key.endswith(".weight_scale")]
        }
        with init_empty_weights():
            vae = AutoencoderKLMiniMaxH3(
                latents_mean=tuple(mean),
                latents_std=tuple(std),
                clip_length=int(raw.get("vae_clip_length", 17)),
                token_drop=int(raw.get("vae_token_drop", 3)),
            )
        missing, unexpected = vae.load_state_dict(converted, strict=False, assign=True)
        # The int8 layers' weights arrive with `specs`; their biases are loaded above, and the swap keeps them.
        _check([key for key in missing if key.removesuffix(".weight") not in specs], unexpected, "video VAE")
        swap_quantized_linears(vae, specs)
        vae.tile_batch = config.VAE_TILE_BATCH
        vae._decode_clip = types.MethodType(_decode_clip_batched, vae)
        _norms_in_float32(vae)
        int8 = sum(module.weight.numel() for module in vae.modules() if isinstance(module, ConvRotLinear))
        detail(f"{_size(vae) + int8 / 1024**3:.2f} GB · {len(specs)} int8 layers")
    return vae.eval()


def _quantized_layers(tensors: dict[str, torch.Tensor]) -> dict[str, object]:
    """Drop the `comfy_quant` markers, checking every int8 layer is the ConvRot format `ConvRotLinear` runs.

    Returns that format's description; an fp16 checkpoint has no markers and gives an empty one.
    """
    metas = {key: read_quant_meta(tensors.pop(key)) for key in [k for k in tensors if k.endswith(".comfy_quant")]}
    expected = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}
    for key, meta in metas.items():
        if meta != expected:
            raise RuntimeError(f"video VAE: {key.removesuffix('.comfy_quant')} is {meta}, not {expected}")
    return dict(expected) if metas else {}


def _compute_dtype(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """fp16 for every floating tensor but the int8 layers' scales, which the kernel wants in float32."""
    if not tensor.is_floating_point() or key.endswith(".weight_scale"):
        return tensor
    return tensor.to(torch.float16)


def _norms_in_float32(vae: AutoencoderKLMiniMaxH3) -> None:
    """Keep the RMSNorm weights in float32, as the reference recipe has them.

    The decoder normalizes in float32 on purpose - `norm1(hidden_states.float())` - but its weights are
    loaded as fp16, so every norm met a float32 input with an fp16 weight. PyTorch computes that correctly on
    its generic path, warns that it "cannot dispatch to fused implementation", and is slower for it.
    The norm weights are a few kilobytes; in float32 the fused kernel takes them.
    """
    for module in vae.modules():
        if isinstance(module, torch.nn.RMSNorm) and module.weight is not None:
            module.weight.data = module.weight.data.float()


def _decode_clip_batched(vae: AutoencoderKLMiniMaxH3, z: torch.Tensor) -> torch.Tensor:
    """`_decode_clip`, but decoding the spatial tiles as one batch instead of one at a time.

    diffusers walks the 256x256 tiles in a Python double loop, so a 1120x736 frame costs 24 decoder
    calls and a 15 s clip 504 of them - 18144 transformer-block invocations on tiny inputs, which is
    launch-bound rather than compute-bound. Every tile is exactly `tile_size`, so they stack straight
    onto the batch axis; the decoder treats batch entries independently, so the result is the same
    tiles, blended by the same `_stitch_tiles`.
    """
    if not vae.use_tiling:
        whole: torch.Tensor = vae.decoder(vae.post_quant_conv(z))
        return whole

    ratio = vae.spatial_compression_ratio
    y_indices, y_lengths, y_overlaps = vae._split_tiles(
        z.shape[-2] * ratio, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
    )
    x_indices, x_lengths, x_overlaps = vae._split_tiles(
        z.shape[-1] * ratio, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
    )
    crops = [
        z[..., y // ratio : (y + y_len) // ratio, x // ratio : (x + x_len) // ratio]
        for y, y_len in zip(y_indices, y_lengths, strict=True)
        for x, x_len in zip(x_indices, x_lengths, strict=True)
    ]
    decoded = _decode_tiles(vae, crops)
    columns = len(x_indices)
    rows = [decoded[index * columns : (index + 1) * columns] for index in range(len(y_indices))]
    stitched: torch.Tensor = vae._stitch_tiles(rows, y_overlaps, x_overlaps)
    return stitched


def _decode_tiles(vae: AutoencoderKLMiniMaxH3, crops: list[torch.Tensor]) -> list[torch.Tensor]:
    while True:
        try:
            decoded: list[torch.Tensor] = []
            for start in range(0, len(crops), vae.tile_batch):
                group = torch.cat(crops[start : start + vae.tile_batch], dim=0)
                out = vae.decoder(vae.post_quant_conv(group))
                decoded.extend(out[index : index + 1] for index in range(out.shape[0]))
            return decoded
        except torch.OutOfMemoryError:
            if vae.tile_batch <= 1:
                raise
            vae.tile_batch = max(1, vae.tile_batch // 2)
            torch.cuda.empty_cache()
            ui.note(f"VAE decode out of memory: retrying {vae.tile_batch} tiles at a time")


def load_audio_vae(path: str | Path) -> AutoencoderKLMiniMaxH3Audio:
    with ui.step("Loading audio VAE") as detail:
        tensors, _ = _read(path)
        mean, std = _latent_stats(tensors)
        with init_empty_weights():
            vae = AutoencoderKLMiniMaxH3Audio(
                sampling_rate=config.AUDIO_SAMPLE_RATE, latents_mean=mean, latents_std=std
            )
        expected = set(vae.state_dict())
        missing, unexpected = vae.load_state_dict(convert_audio_vae(tensors, expected), strict=False, assign=True)
        _check(missing, unexpected, "audio VAE")
        detail(f"{_size(vae):.2f} GB fp32 · {config.AUDIO_SAMPLE_RATE // 1000} kHz stereo")
    return vae.eval()


def denormalize(latents: torch.Tensor, mean: list[float], std: list[float], channel_dim: int = 1) -> torch.Tensor:
    """`latent * std + mean`, per channel (these checkpoints have no scalar scaling factor)."""
    shape = [1] * latents.ndim
    shape[channel_dim] = len(mean)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=latents.device).view(shape)
    std_t = torch.tensor(std, dtype=torch.float32, device=latents.device).view(shape)
    return latents.float() * std_t + mean_t


def normalize(latents: torch.Tensor, mean: list[float], std: list[float], channel_dim: int = 1) -> torch.Tensor:
    """`(latent - mean) / std`, the inverse of `denormalize`."""
    shape = [1] * latents.ndim
    shape[channel_dim] = len(mean)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=latents.device).view(shape)
    std_t = torch.tensor(std, dtype=torch.float32, device=latents.device).view(shape)
    return (latents.float() - mean_t) / std_t


KEYFRAME_ENCODE_SEED = 42


def encode_frame(
    vae: AutoencoderKLMiniMaxH3, pixels: torch.Tensor, encode_seed: int = KEYFRAME_ENCODE_SEED
) -> torch.Tensor:
    """Encode one frame `[1, 3, H, W]` in [0, 1] into a normalized latent `[1, 24, 1, H/16, W/16]`.

    The released model's recipe, and all of it matters for reproducing its conditioning: the posterior
    is *sampled* under a generator seeded independently of the request, and the sample is rounded to
    fp16 (~11 bits per latent) before normalization.
    """
    video = to_model_pixels(pixels).unsqueeze(2).to(device=vae.device, dtype=weight_dtype(vae))  # [1, 3, 1, H, W]
    with torch.inference_mode():
        posterior = vae.encode(video).latent_dist
    sampled: torch.Tensor = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
    latent = sampled.to(torch.float16).float().cpu()
    return normalize(latent, vae.config.latents_mean, vae.config.latents_std)


def encode_clip(
    vae: AutoencoderKLMiniMaxH3, pixels: torch.Tensor, encode_seed: int = KEYFRAME_ENCODE_SEED
) -> torch.Tensor:
    """Encode a whole clip `[T, 3, H, W]` in [0, 1] into a normalized latent `[1, 24, t, H/16, W/16]`.

    `encode_frame`'s recipe over a temporal stack rather than one frame: the encoder is causal and
    chunked, so it wants the frames together — a reference video encoded frame by frame would lose the
    motion the reference exists to carry.
    """
    video = to_model_pixels(pixels).permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
    video = video.to(device=vae.device, dtype=weight_dtype(vae))
    with torch.inference_mode():
        posterior = vae.encode(video).latent_dist
    sampled: torch.Tensor = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
    latent = sampled.to(torch.float16).float().cpu()
    return normalize(latent, vae.config.latents_mean, vae.config.latents_std)


def decode_clip_count(vae: AutoencoderKLMiniMaxH3, latent_frames: int) -> int:
    """How many temporal clips `decode` splits `latent_frames` into - diffusers' `_decode` arithmetic."""
    token_drop = vae.config.token_drop
    tokens = latent_frames + token_drop
    padded = tokens + (-tokens) % vae.tokens_chunk_size
    return int(padded // vae.tokens_chunk_size - int(token_drop > 0))


def decode_video(
    vae: AutoencoderKLMiniMaxH3, latents: torch.Tensor, on_clip: Callable[[], None] | None = None
) -> torch.Tensor:
    """Normalized `[1, 24, T, h, w]` -> pixels `[T, H, W, 3]` in [0, 1].

    `on_clip` is called as each temporal clip finishes, `decode_clip_count` of them in all: at 2 MP
    one takes about seven seconds, and a 12 s chunk (17 clips) about two minutes.
    """
    device = next(vae.parameters()).device
    z = denormalize(latents, vae.config.latents_mean, vae.config.latents_std).to(device, weight_dtype(vae))
    decode_clip = vae._decode_clip
    if on_clip is not None:

        def reporting(clip: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = decode_clip(clip)
            on_clip()
            return out

        vae._decode_clip = reporting
    try:
        with torch.inference_mode():
            sample: torch.Tensor = vae.decode(z).sample  # [1, 3, T, H, W] in ImageNet space
    finally:
        vae._decode_clip = decode_clip
    return from_model_pixels(sample)


def encode_audio(vae: AutoencoderKLMiniMaxH3Audio, waveform: torch.Tensor) -> torch.Tensor:
    """Waveform `[2, samples]` at 32 kHz -> normalized latents `[1, 32, 2, T]`, channel-major.

    The inverse of `decode_audio`: the two stereo channels ride in the batch, and the posterior's
    **mean** is taken rather than a sample — the reference pipeline never evaluates the log-variance
    head for a reference soundtrack, so there is no generator to seed here.
    """
    mono = waveform.to(device=vae.device, dtype=torch.float32).unsqueeze(1)  # [2, 1, samples]
    with torch.inference_mode():
        posterior = vae.encode(mono).latent_dist
    latent = posterior.mode().float().cpu()  # [2, 32, T]
    stereo = latent.permute(1, 0, 2).unsqueeze(0)  # [1, 32, 2, T]
    return normalize(stereo, vae.config.latents_mean, vae.config.latents_std)


def decode_audio(vae: AutoencoderKLMiniMaxH3Audio, latents: torch.Tensor) -> torch.Tensor:
    """Normalized `[1, 32, 2, T]` -> waveform `[2, samples]` in [-1, 1]; stereo rides in the batch."""
    z = denormalize(latents, vae.config.latents_mean, vae.config.latents_std)
    stereo = z[0].permute(1, 0, 2).contiguous().to(device=vae.device, dtype=torch.float32)  # [2, 32, T]
    with torch.inference_mode():
        waveform: torch.Tensor = vae.decode(stereo).sample  # [2, 1, samples]
    return waveform[:, 0].clamp(-1, 1).float().cpu()


def weight_dtype(model: torch.nn.Module) -> torch.dtype:
    """The checkpoint's own dtype. `model.dtype` reports fp32 here because some buffers are fp32."""
    return next(model.parameters()).dtype


def _check(missing: list[str], unexpected: list[str], what: str) -> None:
    if missing or unexpected:
        raise RuntimeError(f"{what}: checkpoint mismatch; missing={sorted(missing)[:8]} unexpected={unexpected[:8]}")


def _size(model: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
