"""Enlarging finished frames with an ESRGAN: it can only sharpen what the decoder already drew, and
it is the only route left once a clip has been written to disk.
"""

from __future__ import annotations

import torch

from difflab import config
from difflab.core import models, ui
from difflab.core.hub import HubFile
from difflab.core.settings import EnlargePixelsSettings
from difflab.core.types import Clip
from difflab.models import esrgan


def spec(settings: EnlargePixelsSettings) -> models.ModelSpec:
    file = HubFile.parse(settings.checkpoint)
    return models.ModelSpec(
        name=f"esrgan:{file.filename}",
        build=lambda: esrgan.load(file.fetch(), config.DEVICE),
        checkpoints=(file.path,),
        heavy=False,  # 64 MB: it can sit beside anything
    )


def upscale(
    clip: Clip, megapixels: float | None = None, tile: int = 1024, settings: EnlargePixelsSettings | None = None
) -> Clip:
    """Every frame of `clip`, upscaled. The soundtrack passes through untouched.

    The network's own factor rarely lands on `megapixels` exactly, so the result is resampled the last
    short step with Lanczos-like antialiasing; `None` keeps the native factor.

    Frames are done one at a time rather than batched: the dense blocks hold five concatenated
    activations at full resolution, so a batch buys throughput at a steep memory price, and this pass
    is not where the time goes.
    """
    model: esrgan.RRDBNet = models.registry.acquire(spec(settings or EnlargePixelsSettings()))
    source_h, source_w = int(clip.frames.shape[1]), int(clip.frames.shape[2])
    width, height = (
        esrgan.target_size(source_w, source_h, megapixels)
        if megapixels
        else (source_w * model.scale, source_h * model.scale)
    )
    out = torch.empty(len(clip), height, width, 3, dtype=torch.float32)

    with ui.progress(f"Upscaling to {width}×{height}", len(clip)) as advance:
        for index in range(len(clip)):
            pixels = clip.frames[index].permute(2, 0, 1)[None].to(config.DEVICE)
            with torch.inference_mode():
                large = esrgan.upscale_image(model, pixels, tile=tile)
                large = _resample(large, source_w, source_h, megapixels, model.scale)
            out[index] = large[0].permute(1, 2, 0).float().cpu()
            advance()

    return Clip(
        frames=out,
        audio=clip.audio,
        fps=clip.fps,
        sample_rate=clip.sample_rate,
        prompt=clip.prompt,
    )


def _resample(large: torch.Tensor, source_w: int, source_h: int, megapixels: float | None, scale: int) -> torch.Tensor:
    if not megapixels:
        return large
    width, height = esrgan.target_size(source_w, source_h, megapixels)
    if (large.shape[-1], large.shape[-2]) == (width, height):
        return large
    resampled: torch.Tensor = torch.nn.functional.interpolate(
        large, size=(height, width), mode="bicubic", align_corners=False, antialias=True
    )
    return resampled.clamp(0, 1)
