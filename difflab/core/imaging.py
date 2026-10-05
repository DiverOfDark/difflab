import math
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image, ImageOps

from difflab import config
from difflab.core import ui


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def load_video_frames(
    path: str | Path, limit: int | None = None, start: float = 0.0, size: tuple[int, int] | None = None
) -> tuple[list[Image.Image], float]:
    """A span is seeked to rather than decoded past: a minute of 1080p is 35 GB of float pixels, so the
    chunk of a long video being edited has to be read on its own. The seek lands on the keyframe at or
    before `start`, so frames before it are dropped to keep the first frame the right one."""
    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else 24.0
        if start > 0:
            container.seek(int(start / (stream.time_base or Fraction(1, 1000))), stream=stream, backward=True)
        for frame in container.decode(video=0):
            if start > 0 and frame.time is not None and frame.time < start - 1e-3:
                continue
            picture = frame.to_image().convert("RGB")
            # Resized during decode: a reference clip is encoded at the canvas's size anyway, so
            # carrying 12 s of 1080p (about 7 GB of float pixels) through memory buys nothing.
            frames.append(resize_to(picture, size) if size else picture)
            if limit is not None and len(frames) >= limit:
                break
    if not frames:
        raise ValueError(f"{path} has no video frames at {start:g}s")
    return frames, rate


def video_length(path: str | Path) -> tuple[float, float]:
    """`(seconds, fps)` from the container, without decoding a frame of it."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else 24.0
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base), rate
        if container.duration is not None:
            return container.duration / 1_000_000, rate
        raise ValueError(f"{path} does not say how long it is")


def megapixel_size(width: int, height: int, megapixels: float) -> tuple[int, int]:
    """`width` x `height`'s aspect scaled to `megapixels` millions of pixels."""
    scale = math.sqrt(megapixels * 1_000_000 / (width * height))
    return round(width * scale), round(height * scale)


def resize_to(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.Resampling.LANCZOS)


def save_result(
    img: Image.Image,
    name: str,
    seed: int | None = None,
    out_dir: str | Path = config.OUT_DIR,
) -> tuple[Path, Path]:
    now = datetime.now()
    folder = Path(out_dir) / now.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d-%H%M%S")
    stem = f"{stamp}_{name}_{seed}" if seed is not None else f"{stamp}_{name}"
    png, webp = folder / f"{stem}.png", folder / f"{stem}.webp"
    with ui.step("Saving PNG + WebP") as detail:
        img.save(png, optimize=True)
        img.save(webp, lossless=True, quality=100)
        detail(f"{img.width}×{img.height}")
    return png, webp
