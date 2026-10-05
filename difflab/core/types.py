"""What the blocks hand each other.

Three models meet here and each wants pixels in its own shape: Qwen-VL and Qwen-Image-Edit speak
PIL, MiniMax H3 wants `[T, H, W, 3]` float tensors, and the VAEs want `[B, 3, H, W]`.

A `Clip` owns its frames, and what is taken out of one - `frame`, `segment`, `tail` - is copied
rather than viewed. A view of a single frame keeps its whole clip's storage alive, and a chained long
video keeps one frame and a short tail from every chunk: at 2 MP, views would pin 9 GB per chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class Frame:
    """One still image as `[H, W, 3]` float32 in [0, 1]."""

    pixels: torch.Tensor
    origin: str = ""  # where it came from, for the run panel

    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[-1] != 3:
            raise ValueError(f"a frame is [H, W, 3], got {tuple(self.pixels.shape)}")

    @classmethod
    def from_pil(cls, image: Image.Image, origin: str = "") -> Frame:
        pixels = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).float().div(255.0)
        return cls(pixels, origin)

    @classmethod
    def load_png(cls, path: str | Path) -> Frame:
        """`core.media.frame_from_image` is the one workflows call."""
        return cls.from_pil(Image.open(path), Path(path).name)

    def pil(self) -> Image.Image:
        return Image.fromarray((self.pixels.clamp(0, 1) * 255).round().to(torch.uint8).numpy())

    def batched(self) -> torch.Tensor:
        """`[1, H, W, 3]`, the shape the H3 pipeline plans geometry from."""
        return self.pixels[None]

    def crop_to_aspect(self, ratio: float) -> Frame:
        """Centre-crop to `ratio` (width / height): the clip's shape follows its opening frame, so this
        is how a 3:4 photograph becomes the 9:16 take it opens.

        Cropping rather than padding keeps every pixel real: a letterboxed frame would teach the model
        that the bars are part of the scene.
        """
        height, width = self.height, self.width
        if abs(width / height - ratio) < 1e-6:
            return self
        if width / height > ratio:
            new_width = round(height * ratio)
            left = (width - new_width) // 2
            pixels = self.pixels[:, left : left + new_width]
        else:
            new_height = round(width / ratio)
            top = (height - new_height) // 2
            pixels = self.pixels[top : top + new_height]
        return Frame(pixels.contiguous(), f"{self.origin} ({ratio:.2f} crop)" if self.origin else "")

    def crop(self, box: tuple[int, int, int, int]) -> Frame:
        """The region `(left, top, right, bottom)` in pixels, as PIL's `crop` takes it."""
        left, top, right, bottom = box
        pixels = self.pixels[max(0, top) : min(self.height, bottom), max(0, left) : min(self.width, right)]
        return Frame(pixels.contiguous(), f"{self.origin} (cropped)" if self.origin else "")

    def resized(self, width: int, height: int) -> Frame:
        image = self.pil().resize((width, height), Image.Resampling.LANCZOS)
        return Frame.from_pil(image, f"{self.origin} ({width}x{height})" if self.origin else "")

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    def __repr__(self) -> str:
        return f"Frame({self.width}x{self.height}{', ' + self.origin if self.origin else ''})"


@dataclass(frozen=True)
class Clip:
    """Decoded video: `[T, H, W, 3]` float32 in [0, 1], with the audio H3 generated alongside it."""

    frames: torch.Tensor
    audio: torch.Tensor | None = None  # [2, samples] in [-1, 1]
    fps: int = 24
    sample_rate: int = 32000
    prompt: str = ""

    def __post_init__(self) -> None:
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError(f"a clip is [T, H, W, 3], got {tuple(self.frames.shape)}")

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps

    def frame(self, index: int) -> Frame:
        # A copy, not a view: at 2 MP a view is 9 GB of clip held for the sake of 25 MB of frame.
        return Frame(self.frames[index].clone(), f"frame {index % len(self.frames)}")

    def last_frame(self) -> Frame:
        """The handoff to the next chunk of a long video."""
        return self.frame(len(self.frames) - 1)

    def segment(self, start: int, stop: int) -> Clip:
        """Frames `[start, stop)`, with the audio cut by the same span, so a segment describes one moment
        rather than two — which is what the packed layout assumes of a reference block."""
        start = max(0, start)
        stop = min(len(self), stop)
        audio = None
        if self.audio is not None:
            per_frame = self.sample_rate / self.fps
            audio = self.audio[:, round(start * per_frame) : round(stop * per_frame)]
        return Clip(
            frames=self.frames[start:stop].clone(),  # a copy, for the same reason as `frame`
            audio=audio.clone() if audio is not None else None,
            fps=self.fps,
            sample_rate=self.sample_rate,
            prompt=self.prompt,
        )

    def tail(self, frames: int, before: int | None = None) -> Clip:
        """The motion a chained chunk inherits: the `frames` frames leading *up to* `before` (the end of
        the clip by default), not including it, so the two do not describe the same instant twice."""
        end = len(self) if before is None else before
        return self.segment(end - frames, end)

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __repr__(self) -> str:
        audio = f", {self.audio.shape[1] / self.sample_rate:.2f}s audio" if self.audio is not None else ""
        return f"Clip({len(self)} frames, {self.duration:.2f}s @ {self.fps}fps{audio})"
