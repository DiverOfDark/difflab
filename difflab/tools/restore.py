"""The finish's enlarging: SeedVR2 7B to half the frame, then a 2x ESRGAN the rest of the way.

Measured on production 13 (2026-10-06), against the ESRGAN alone from the native 1056x608: SeedVR2 kept
the render's face and drew clean edges where 4xLSDIR left ladders and ringing, and its 7B looked better
than its 3B at the same cost. At 1080p on a 3090 the 7B's DiT costs 0.17 s a frame and its VAE 1.8 s,
encode and decode together - the VAE is the bill, and it is arithmetic (each 3D conv already runs at
62 TFLOPS), not overhead.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from difflab import config
from difflab.core import media, models, ui, video_io
from difflab.core.hub import HubFile
from difflab.core.settings import RestoreSettings
from difflab.models import esrgan
from difflab.models.seedvr2 import pipeline


def restorer_spec(checkpoint: str) -> models.ModelSpec:
    file = HubFile.parse(checkpoint)
    return models.ModelSpec(
        name=f"seedvr2:{file.filename}",
        build=lambda: pipeline.load_dit(file.fetch()),
        checkpoints=(file.path,),
    )


def vae_spec(checkpoint: str) -> models.ModelSpec:
    file = HubFile.parse(checkpoint)
    return models.ModelSpec(
        name=f"seedvr2_vae:{file.filename}",
        build=lambda: pipeline.load_vae(file.fetch()),
        checkpoints=(file.path,),
        heavy=False,
    )


def upscaler_spec(checkpoint: str) -> models.ModelSpec:
    """Its own fp16 channels-last copy: 0.63 s a 1080p frame against fp32's 1.49, drifting 0.005 at most."""
    file = HubFile.parse(checkpoint)
    return models.ModelSpec(
        name=f"esrgan-fp16:{file.filename}",
        build=lambda: esrgan.load(file.fetch(), config.DEVICE).half().to(memory_format=torch.channels_last),
        checkpoints=(file.path,),
        heavy=False,
    )


def finish_size(width: int, height: int, megapixels: float) -> tuple[int, int]:
    """Within a tenth of 16:9 (H3's canvases are 1056x608 and the like), the 16:9 frame of `megapixels`,
    filled and cropped; otherwise `megapixels` at the canvas's own aspect. Even on both sides of its
    half, so the restorer's size and the 2x on top of it are both whole."""
    wide = math.sqrt(megapixels * 1e6 * 16 / 9)
    standard = round(wide / 4) * 4, round(wide * 9 / 16 / 4) * 4
    if abs(max(width, height) / min(width, height) / (16 / 9) - 1) < 0.1:
        return standard if width >= height else standard[::-1]
    scale = math.sqrt(megapixels * 1e6 / (width * height))
    return round(width * scale / 4) * 4, round(height * scale / 4) * 4


def _filled(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """`[N, 3, H, W]` scaled to cover `width x height`, the overhang cropped evenly from both sides."""
    scale = max(width / images.shape[-1], height / images.shape[-2])
    size = max(height, round(images.shape[-2] * scale)), max(width, round(images.shape[-1] * scale))
    if size != tuple(images.shape[-2:]):
        images = torch.nn.functional.interpolate(images, size=size, mode="bicubic", align_corners=False, antialias=True)
    top, left = (size[0] - height) // 2, (size[1] - width) // 2
    return images[..., top : top + height, left : left + width].clamp(0, 1)


def finish(
    frames: torch.Tensor, destination: Path, size: tuple[int, int], fps: int, settings: RestoreSettings | None = None
) -> int:
    """`frames [T, H, W, 3]` in [0, 1], restored and enlarged to `size` and written to `destination` as they
    come. Returns the frames written."""
    settings = settings or RestoreSettings()
    width, height = size
    half = (width // 2, height // 2)
    source = torch.cat(
        [
            _filled(piece.to(config.DEVICE).permute(0, 3, 1, 2), *half).permute(0, 2, 3, 1).cpu()
            for piece in torch.split(frames, 16)
        ]
    )
    restorer = models.registry.acquire(restorer_spec(settings.checkpoint))
    vae = models.registry.acquire(vae_spec(settings.vae))
    upscaler: esrgan.RRDBNet = models.registry.acquire(upscaler_spec(settings.upscaler))
    windows = len(pipeline.starts(len(source), settings.frames, settings.overlap))
    writer = video_io.VideoWriter(destination, width, height, fps, crf=media.JOIN_CRF)
    written = 0
    with ui.progress(f"Restoring to {half[0]}×{half[1]}, then {width}×{height}", 3 * windows, unit="step") as advance:
        for restored in pipeline.restore(
            source, restorer, vae, settings.frames, settings.overlap, settings.seed, on_step=advance
        ):
            for frame in restored:
                with torch.inference_mode():
                    picture = frame.permute(2, 0, 1)[None].to(config.DEVICE, torch.float16)
                    large = esrgan.upscale_image(
                        upscaler, picture.contiguous(memory_format=torch.channels_last), tile=0
                    )
                    writer.write(_filled(large.float(), width, height)[0].permute(1, 2, 0).cpu()[None])
                written += 1
    writer.close(None)
    return written
