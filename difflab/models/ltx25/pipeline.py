"""LTX-2.5 and 2.3 sampling: an opening frame, MSR references and a prompt into video and audio latents."""

# Step for step diffusers' `LTX2ConditionPipeline`. References are ComfyUI's `LTXVAddGuide` as the MSR guide node
# calls it: appended clean tokens with their own RoPE coordinates, cropped after sampling.

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.utils import apply_image_conditioning_crf
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from torch import nn
from transformers import PreTrainedTokenizerBase

from difflab.core.types import Frame
from difflab.models.ltx25 import config
from difflab.models.ltx25.weights import SlotEmbedding


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    frame_count: int

    @property
    def latent_frames(self) -> int:
        return (self.frame_count - 1) // config.FRAME_CYCLE + 1

    @property
    def latent_height(self) -> int:
        return self.height // config.SPATIAL_COMPRESSION

    @property
    def latent_width(self) -> int:
        return self.width // config.SPATIAL_COMPRESSION

    @property
    def tokens_per_frame(self) -> int:
        return self.latent_height * self.latent_width

    @property
    def audio_latents(self) -> int:
        return round(self.frame_count / config.FPS * config.AUDIO_LATENTS_PER_SECOND)


@dataclass(frozen=True)
class Conditioning:
    video: torch.Tensor  # [1, L, 4096]
    audio: torch.Tensor  # [1, L, 2048]
    mask: torch.Tensor  # [1, L]


def plan(width: int, height: int, duration: float, megapixels: float) -> Geometry:
    scale = math.sqrt(megapixels * 1024 * 1024 / (width * height))
    align = config.SPATIAL_COMPRESSION
    return Geometry(
        max(align, round(width * scale / align) * align),
        max(align, round(height * scale / align) * align),
        frame_count(duration),
    )


def frame_count(duration: float) -> int:
    """The shortest 8k+1 frames that last `duration`."""
    frames = max(config.FRAME_CYCLE + 1, round(duration * config.FPS))
    return (frames - 1 + config.FRAME_CYCLE - 1) // config.FRAME_CYCLE * config.FRAME_CYCLE + 1


def encode_prompt(
    gemma: nn.Module, connectors: LTX2TextConnectors, tokenizer: PreTrainedTokenizerBase, prompt: str
) -> Conditioning:
    device = next(connectors.parameters()).device
    tokenizer.padding_side = "left"
    tokens = tokenizer(
        [prompt.strip()],
        padding="max_length",
        max_length=config.PROMPT_TOKENS,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    mask = tokens.attention_mask.to(device)
    hidden = gemma(input_ids=tokens.input_ids.to(device), attention_mask=mask, output_hidden_states=True).hidden_states
    stacked = torch.stack(hidden, dim=-1).flatten(2, 3).to(torch.bfloat16)
    video, audio, connector_mask = connectors(stacked, mask, padding_side="left")
    return Conditioning(video, audio, connector_mask)


def _normalized(vae: AutoencoderKLLTX2Video, latents: torch.Tensor) -> torch.Tensor:
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, torch.float32)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, torch.float32)
    normalized: torch.Tensor = (latents.float() - mean) * vae.config.scaling_factor / std
    return normalized


def _encode(vae: AutoencoderKLLTX2Video, pixels: torch.Tensor) -> torch.Tensor:
    """`[1, 3, F, H, W]` in [-1, 1] -> normalized `[1, 128, f, h, w]`."""
    device = next(vae.parameters()).device
    latents = vae.encode(pixels.to(device, vae.dtype)).latent_dist.mode()
    return _normalized(vae, latents)


def encode_opening(vae: AutoencoderKLLTX2Video, frame: Frame, geometry: Geometry, crf: int) -> torch.Tensor:
    """Re-compressed at the training CRF at its own size, then covered onto the canvas, in diffusers' order."""
    image = (frame.pixels.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    image = apply_image_conditioning_crf(image, crf)
    pixels = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1)[None]
    source_height, source_width = pixels.shape[-2:]
    scale = max(geometry.height / source_height, geometry.width / source_width)
    size = (math.ceil(source_height * scale), math.ceil(source_width * scale))
    pixels = nn.functional.interpolate(pixels, size=size, mode="bilinear", align_corners=False)
    top, left = (size[0] - geometry.height) // 2, (size[1] - geometry.width) // 2
    pixels = pixels[..., top : top + geometry.height, left : left + geometry.width]
    return _encode(vae, (pixels / 127.5 - 1.0)[:, :, None])


def _msr_canvas(frame: Frame, width: int, height: int) -> Frame:
    """The MSR guide's fit: the canvas's aspect family and no smaller is cropped, anything else shown whole on white."""

    def family(w: int, h: int) -> str:
        return "landscape" if w / h >= 1.25 else "portrait" if w / h <= 0.8 else "square"

    if family(frame.width, frame.height) == family(width, height) and not (
        frame.width <= width and frame.height <= height
    ):
        return frame.crop_to_aspect(width / height).resized(width, height)
    scale = min(width / frame.width, height / frame.height)
    fitted = frame.resized(
        max(1, min(width, round(frame.width * scale))), max(1, min(height, round(frame.height * scale)))
    )
    canvas = torch.ones(height, width, 3)
    top, left = (height - fitted.height) // 2, (width - fitted.width) // 2
    canvas[top : top + fitted.height, left : left + fitted.width] = fitted.pixels
    return Frame(canvas)


def encode_reference(
    vae: AutoencoderKLLTX2Video, frame: Frame, geometry: Geometry, slots: SlotEmbedding, slot: int
) -> torch.Tensor:
    """Slot `slot` (from 1): the still held for 33 frames, encoded, its slot's identity added to every token."""
    fitted = _msr_canvas(frame, geometry.width, geometry.height)
    still = (fitted.pixels.permute(2, 0, 1) * 2 - 1)[None, :, None]
    latents = _encode(vae, still.expand(-1, -1, config.MSR_FRAMES, -1, -1))
    return latents + slots(slot).view(1, -1, 1, 1, 1).to(latents)


def _pack(latents: torch.Tensor) -> torch.Tensor:
    """`[1, C, F, H, W]` -> `[1, F*H*W, C]`."""
    return latents.flatten(2).transpose(1, 2)


def sample(
    dit: LTX2VideoTransformer3DModel,
    conditioning: Conditioning,
    geometry: Geometry,
    seed: int,
    sigmas: Sequence[float],
    opening: torch.Tensor | None = None,
    references: Sequence[torch.Tensor] = (),
    reference_strength: float = 1.0,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    landing: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Video latents `[1, 128, F, h, w]` and packed audio latents `[1, L, 128]`, both normalized. `on_step` is
    handed each step's estimate of the finished clip, in the same shapes."""
    device = conditioning.video.device
    generator = torch.Generator(device).manual_seed(seed)
    frames, height, width = geometry.latent_frames, geometry.latent_height, geometry.latent_width
    tokens = frames * geometry.tokens_per_frame
    video = torch.randn((1, tokens, config.LATENT_CHANNELS), generator=generator, device=device)
    audio = torch.randn((1, geometry.audio_latents, config.LATENT_CHANNELS), generator=generator, device=device)
    clean = torch.zeros_like(video)
    strength = torch.zeros((1, tokens, 1), device=device)
    keyframes = torch.zeros((1, tokens, 1), device=device)
    keyframes[:, : geometry.tokens_per_frame] = 1.0  # the causal first frame is a single pixel frame
    coords = [dit.rope.prepare_video_coords(1, frames, height, width, device, fps=config.FPS)]
    if opening is not None:
        packed = _pack(opening.to(device))
        video[:, : packed.shape[1]] = packed
        clean[:, : packed.shape[1]] = packed
        strength[:, : packed.shape[1]] = 1.0
    for index, reference in enumerate(references):
        packed = _pack(reference.to(device))
        video = torch.cat([video, packed], dim=1)
        clean = torch.cat([clean, packed], dim=1)
        strength = torch.cat([strength, strength.new_full((1, packed.shape[1], 1), reference_strength)], dim=1)
        keyframes = torch.cat([keyframes, keyframes.new_zeros((1, packed.shape[1], 1))], dim=1)
        placed = dit.rope.prepare_video_coords(1, reference.shape[2], height, width, device, fps=config.FPS)
        placed[:, 0] += -(len(references) - index) / config.FPS
        coords.append(placed)
    if landing is not None:
        # A keyframe the clip ends on, as diffusers appends one: one pixel frame wide, at the last frame, clean.
        packed = _pack(landing.to(device))
        video = torch.cat([video, packed], dim=1)
        clean = torch.cat([clean, packed], dim=1)
        strength = torch.cat([strength, strength.new_ones((1, packed.shape[1], 1))], dim=1)
        keyframes = torch.cat([keyframes, keyframes.new_zeros((1, packed.shape[1], 1))], dim=1)
        placed = dit.rope.prepare_video_coords(1, 1, height, width, device, fps=config.FPS)
        placed[:, 0] += (geometry.frame_count - 1) / config.FPS
        coords.append(placed)
    video_coords = torch.cat(coords, dim=2)
    audio_coords = dit.audio_rope.prepare_audio_coords(1, geometry.audio_latents, device)

    for step, (sigma, following) in enumerate(itertools.pairwise(sigmas)):
        timestep = torch.full((1,), sigma * 1000.0, device=device)
        with torch.no_grad():
            video_velocity, audio_velocity = dit(
                hidden_states=video.to(torch.bfloat16),
                audio_hidden_states=audio.to(torch.bfloat16),
                encoder_hidden_states=conditioning.video,
                audio_encoder_hidden_states=conditioning.audio,
                timestep=timestep[:, None] * (1 - strength.squeeze(-1)),
                audio_timestep=timestep,
                sigma=timestep,
                audio_sigma=timestep,
                encoder_attention_mask=conditioning.mask,
                audio_encoder_attention_mask=conditioning.mask,
                fps=config.FPS,
                video_coords=video_coords,
                audio_coords=audio_coords,
                use_cross_timestep=True,
                video_keyframes_mask=keyframes,
                return_dict=False,
            )
        denoised = video - video_velocity.float() * sigma
        denoised = denoised * (1 - strength) + clean * strength
        video = denoised + following * (video - denoised) / sigma
        heard = audio - audio_velocity.float() * sigma
        audio = heard + following * (audio - heard) / sigma
        if on_step is not None:
            on_step(step, _unpack(denoised, geometry), heard)
    return _unpack(video, geometry), audio


def _unpack(packed: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    """The clip's own tokens, without the references appended after them, as `[1, 128, F, h, w]`."""
    frames, height, width = geometry.latent_frames, geometry.latent_height, geometry.latent_width
    base = packed[:, : frames * height * width].reshape(1, frames, height, width, config.LATENT_CHANNELS)
    return base.permute(0, 4, 1, 2, 3).contiguous()


def decode_video(vae: AutoencoderKLLTX2Video, latents: torch.Tensor) -> torch.Tensor:
    """Normalized latents -> `[T, H, W, 3]` float32 in [0, 1] on the CPU."""
    device = next(vae.parameters()).device
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device, torch.float32)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(device, torch.float32)
    raw = latents.to(device).float() * std / vae.config.scaling_factor + mean
    with torch.no_grad():
        pixels = vae.decode(raw.to(vae.dtype), return_dict=False)[0]
    frames: torch.Tensor = ((pixels[0].permute(1, 2, 3, 0).float() + 1) / 2).clamp(0, 1).cpu()
    return frames


def decode_audio(vae: AutoencoderKLLTX2Audio, vocoder: LTX2VocoderWithBWE, packed: torch.Tensor) -> torch.Tensor:
    """Normalized packed latents -> `[2, samples]` at 48 kHz on the CPU."""
    device = next(vae.parameters()).device
    raw = packed.to(device).float() * vae.latents_std.to(device).float() + vae.latents_mean.to(device).float()
    mel_bins = vae.config.mel_bins // 4
    latents = raw.unflatten(2, (-1, mel_bins)).transpose(1, 2)
    with torch.no_grad():
        mel = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
        waveform = vocoder(mel.to(next(vocoder.parameters()).dtype))
    stereo: torch.Tensor = waveform[0].float().clamp(-1, 1).cpu()
    return stereo
