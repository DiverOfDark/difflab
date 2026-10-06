"""Enlarging finished frames with an ESRGAN: it can only sharpen what the decoder already drew, and
it is the only route left once a clip has been written to disk.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import torch

from difflab import config
from difflab.core import media, models, ui, video_io
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


def finish_size(width: int, height: int, megapixels: float) -> tuple[int, int]:
    """Within a tenth of 16:9 (H3's canvases are 1056x608 and the like), the 16:9 frame of `megapixels`,
    filled and cropped; otherwise `megapixels` at the canvas's own aspect. Even, as yuv420p wants."""
    wide = math.sqrt(megapixels * 1e6 * 16 / 9)
    standard = round(wide / 2) * 2, round(wide * 9 / 16 / 2) * 2
    if abs(max(width, height) / min(width, height) / (16 / 9) - 1) < 0.1:
        return standard if width >= height else standard[::-1]
    scale = math.sqrt(megapixels * 1e6 / (width * height))
    return round(width * scale / 2) * 2, round(height * scale / 2) * 2


def finish(
    frames: Iterable[torch.Tensor],
    count: int,
    destination: Path,
    size: tuple[int, int],
    fps: int,
    settings: EnlargePixelsSettings | None = None,
) -> int:
    """Each `[H, W, 3]` frame enlarged, filled to `size` and written to `destination` as it comes: a 10 s
    chunk whole at 4K is 24 GB of float32. Returns the frames written.

    The ESRGAN rather than H3's latent upscaler: on production 13 (2026-10-06) the latent route, at 2x
    as at 3.6x, redrew the face - another eye shape, an older man, colour fringes on the hair - where
    this kept the render's own and only sharpened it, at the same 2.6 s a 4K frame.
    """
    model: esrgan.RRDBNet = models.registry.acquire(spec(settings or EnlargePixelsSettings()))
    width, height = size
    writer = video_io.VideoWriter(destination, width, height, fps, crf=media.JOIN_CRF)
    written = 0
    with ui.progress(f"Upscaling to {width}×{height}", count, unit="frame") as advance:
        for frame in frames:
            with torch.inference_mode():
                large = esrgan.upscale_image(model, frame.permute(2, 0, 1)[None].to(config.DEVICE))
                writer.write(_filled(large, width, height)[0].permute(1, 2, 0).cpu()[None])
            written += 1
            advance()
    writer.close(None)
    return written


def _filled(large: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Scaled to cover `width x height`, the overhang cropped evenly from both sides."""
    scale = max(width / large.shape[-1], height / large.shape[-2])
    size = max(height, round(large.shape[-2] * scale)), max(width, round(large.shape[-1] * scale))
    fitted = torch.nn.functional.interpolate(large, size=size, mode="bicubic", align_corners=False, antialias=True)
    top, left = (size[0] - height) // 2, (size[1] - width) // 2
    return fitted[..., top : top + height, left : left + width].clamp(0, 1)
