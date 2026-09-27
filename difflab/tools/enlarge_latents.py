"""Enlarging in latent space: the VAE draws the extra pixels, on a tensor 256x smaller than the frames.

Sampling cost grows with the square of the packed sequence, so pixels are the expensive thing to
generate and the cheap thing to add afterwards: 0.4 MP samples about 2.5x faster than 0.8 MP, and
this pass costs a fraction of the difference while landing well above either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from difflab import config
from difflab.core import models, ui
from difflab.core.hub import Checkpoint
from difflab.core.types import Clip
from difflab.models import latent_upscaler
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
        heavy=False,  # 345 M parameters, ~0.7 GB in fp16
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


@dataclass(frozen=True)
class Upscaled:
    """A finish that enlarges the latents before decoding them. `render_chunk.Finish`, with a size."""

    megapixels: float
    settings: EnlargeLatentsSettings = field(default_factory=EnlargeLatentsSettings)

    def __call__(self, latents: render_chunk.Latents) -> Clip:
        return render_chunk.decode(upscale_latents(latents, megapixels=self.megapixels, settings=self.settings))

    def size(self, width: int, height: int) -> tuple[int, int]:
        return pipeline.canvas_size(width, height, self.megapixels)


def upscaled(megapixels: float, settings: EnlargeLatentsSettings | None = None) -> Upscaled:
    """`finish=upscaled(2.0)`: sample small, enlarge the latent, decode at 2 MP."""
    return Upscaled(megapixels, settings or EnlargeLatentsSettings())
