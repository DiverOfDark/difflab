"""Enlarging in latent space: the VAE draws the extra pixels, on a tensor 256x smaller than the frames.

Sampling cost grows with the square of the packed sequence, so pixels are the expensive thing to
generate and the cheap thing to add afterwards: 0.4 MP samples about 2.5x faster than 0.8 MP, and
this pass costs a fraction of the difference while landing well above either.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch

from difflab import config
from difflab.core import media, models, ui, video_io
from difflab.core.hub import Checkpoint
from difflab.models import latent_upscaler
from difflab.models.minimax_h3 import config as h3_config
from difflab.models.minimax_h3 import pipeline, vae
from difflab.models.minimax_h3.settings import EnlargeLatentsSettings
from difflab.tools import render_chunk


def _files(settings: EnlargeLatentsSettings) -> tuple[Checkpoint, str, str]:
    """The snapshot, its weights and the config.json beside them."""
    pinned = Checkpoint.parse(settings.checkpoint)
    weights = next((f for f in pinned.files if f.endswith(".safetensors")), None)
    if weights is None:
        raise ValueError(f"{settings.checkpoint}: name the .safetensors file after the colon")
    return pinned, weights, str(Path(weights).parent / "config.json")


def spec(settings: EnlargeLatentsSettings) -> models.ModelSpec:
    pinned, weights, _ = _files(settings)
    return models.ModelSpec(
        name=f"h3_latent_upscaler:{weights}",
        build=lambda: latent_upscaler.load(pinned.fetch() / weights, config.DEVICE),
        checkpoints=(pinned.path / weights,),
        # 345 M parameters, ~0.7 GB in fp16, but heavy: acquiring it evicts the DiT the shots left on the
        # card, which the 4K decode after it needs.
        heavy=True,
    )


def upscale_latents(
    latents: render_chunk.Latents,
    megapixels: float | None = None,
    scale: float | None = None,
    settings: EnlargeLatentsSettings | None = None,
) -> render_chunk.Latents:
    """Enlarge a video latent before it is decoded, so the VAE renders the extra detail itself.

    Give either `megapixels` (the pixel size wanted) or `scale` (a linear factor, 1.0-4.0). The time
    axis is untouched; the audio latent goes through unchanged, since it has no spatial extent.

    The model was trained against the VAE's own per-channel statistics, which are the same ones this
    pipeline normalises with to four decimal places - so the latent is converted back to raw space and
    into the model's, rather than assumed to be interchangeable.
    """
    settings = settings or EnlargeLatentsSettings()
    model: latent_upscaler.LatentResizer3D = models.registry.acquire(spec(settings))
    geometry = latents.geometry
    if megapixels:
        width, height = pipeline.canvas_size(geometry.width, geometry.height, megapixels)
    elif scale:
        align = config.CANVAS_ALIGN
        width = max(align, round(geometry.width * scale / align) * align)
        height = max(align, round(geometry.height * scale / align) * align)
    else:
        raise ValueError("give either megapixels or scale")

    target = pipeline.Geometry(width, height, geometry.frame_count)
    factor = math.sqrt((width * height) / (geometry.width * geometry.height))
    low, high = latent_upscaler.SCALE_RANGE
    if not low <= factor <= high:
        raise ValueError(f"a {factor:.2f}x resize is outside the model's trained range {low}-{high}")

    pinned, _, normalisation = _files(settings)
    stats = latent_upscaler.read_normalisation(pinned.fetch() / normalisation)
    with ui.step(f"Upscaling latents to {width}×{height}") as detail:
        # H3's VAE is unusual among ComfyUI's: its `encode` already returns `(x - mean) / std`. The
        # upscaler was trained behind a node that normalises again with the same constants, so its
        # input space is doubly normalised, and a singly normalised latent loads, runs and decodes to
        # a checkerboard - a distribution the model never saw.
        prepared = vae.normalize(latents.video, *stats) if stats else latents.video
        with torch.inference_mode():
            enlarged = (
                model(
                    prepared.to(config.DEVICE, next(model.parameters()).dtype),
                    height=target.latent_height,
                    width=target.latent_width,
                    scale=factor,
                )
                .float()
                .cpu()
            )
        video = vae.denormalize(enlarged, *stats) if stats else enlarged
        detail(f"{tuple(latents.video.shape[-2:])} → {tuple(video.shape[-2:])} latent · {factor:.2f}x")

    return render_chunk.Latents(video, latents.audio, target, latents.settings, latents.prompt)


def target_size(width: int, height: int, megapixels: float) -> tuple[int, int]:
    """Within a tenth of 16:9 (H3's canvases are 864x480 and the like), the 16:9 frame of `megapixels`,
    3840x2160 by default, filled and cropped; otherwise `megapixels` at the canvas's own aspect. Even
    on both sides, as yuv420p wants."""
    wide = math.sqrt(megapixels * 1e6 * 16 / 9)
    standard = round(wide / 2) * 2, round(wide * 9 / 16 / 2) * 2
    aspect = max(width, height) / min(width, height)
    if abs(aspect / (16 / 9) - 1) < 0.1:
        return standard if width >= height else standard[::-1]
    scale = math.sqrt(megapixels * 1e6 / (width * height))
    return round(width * scale / 2) * 2, round(height * scale / 2) * 2


def finish_chunk(
    latents: render_chunk.Latents,
    destination: Path,
    holds: Sequence[media.Hold] = (),
    settings: EnlargeLatentsSettings | None = None,
) -> tuple[int, int, int]:
    """A sampled chunk enlarged to `settings.megapixels` and written to `destination` without ever being
    whole in memory, `holds` applied frame by frame. Returns `(width, height, frames)`.

    The latent goes as far as the upscaler's 4x allows; a 0.4 MP canvas needs 4.5x to reach 4K, and the
    rest is a bicubic pass over frames the VAE already drew at 6.4 MP.
    """
    settings = settings or EnlargeLatentsSettings()
    geometry = latents.geometry
    width, height = target_size(geometry.width, geometry.height, settings.megapixels)
    # A scale, not megapixels: those round to the 32 grid and can land just past the trained 4x.
    wanted = max(width / geometry.width, height / geometry.height)
    enlarged = upscale_latents(latents, scale=min(wanted, latent_upscaler.SCALE_RANGE[1]), settings=settings)
    model = models.registry.acquire(render_chunk.video_vae_spec(latents.settings))
    clips = vae.decode_clip_count(model, int(enlarged.video.shape[2]))
    writer = video_io.VideoWriter(destination, width, height, h3_config.FPS, crf=16)
    count = 0
    with ui.progress(f"Decoding video {width}×{height}", clips, unit="clip") as advance:
        for frames in vae.stream_video(model, enlarged.video, on_clip=advance):
            if holds:
                frames = media.held(frames, holds[count : count + len(frames)])
            if frames.shape[1:3] != (height, width):
                frames = _filled(frames, width, height)
            writer.write(frames)
            count += len(frames)
    writer.close(None)
    return width, height, count


def _filled(frames: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Scaled to cover `width x height`, the overhang cropped evenly from both sides."""
    scale = max(width / frames.shape[2], height / frames.shape[1])
    size = max(height, round(frames.shape[1] * scale)), max(width, round(frames.shape[2] * scale))
    on_card = frames.to(config.DEVICE).permute(0, 3, 1, 2)
    bigger = torch.nn.functional.interpolate(on_card, size=size, mode="bicubic", align_corners=False)
    top, left = (size[0] - height) // 2, (size[1] - width) // 2
    return bigger[..., top : top + height, left : left + width].clamp_(0, 1).permute(0, 2, 3, 1).cpu()
