"""Mux decoded frames and audio into an MP4 (ComfyUI's CreateVideo + SaveVideo).

H.264 video at 24 fps in yuv420p with BT.709 tags, AAC audio at the VAE's 32 kHz, stereo. The two
streams do not divide evenly — 362 frames at 24 fps is 15.083 s while 603 audio latents at 40 Hz is
15.075 s — so the waveform is trimmed to `ceil(sample_rate / fps * frames)` samples, as ComfyUI does.
"""

from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

from difflab.core import ui
from difflab.core.audio import SAMPLE_RATE, limited


def output_path(out_dir: str | Path, name: str, seed: int | None = None) -> Path:
    """`<out>/<YYYY-MM-DD>/<YYYY-MM-DD-HHMMSS>_<name>[_<seed>].mp4`; a clip that was loaded, joined or
    upscaled has no seed of its own, and borrowing one makes the filename lie."""
    now = datetime.now()
    folder = Path(out_dir) / now.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d-%H%M%S")
    return folder / (f"{stamp}_{name}_{seed}.mp4" if seed is not None else f"{stamp}_{name}.mp4")


def save_video(
    frames: torch.Tensor,
    waveform: torch.Tensor | None,
    path: Path,
    fps: int,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """`frames [T, H, W, 3]` in [0, 1] and `waveform [2, samples]` in [-1, 1] -> an MP4 with both streams.

    `waveform=None` writes video only: H3 always makes a soundtrack, but a clip read back from disk,
    upscaled or cut by hand need not have one.
    """
    with ui.step("Muxing MP4") as detail:
        height, width = int(frames.shape[1]), int(frames.shape[2])
        samples = int(np.ceil(sample_rate / fps * len(frames)))
        audio = limited(waveform[:, :samples].contiguous().float().numpy()) if waveform is not None else None

        with av.open(str(path), "w", format="mp4") as container:
            video_stream = container.add_stream("h264", rate=Fraction(round(fps * 1000), 1000))
            video_stream.width, video_stream.height = width, height
            video_stream.pix_fmt = "yuv420p"
            audio_stream = container.add_stream("aac", rate=sample_rate, layout="stereo") if audio is not None else None

            for frame in frames:
                pixels = (frame.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                picture = av.VideoFrame.from_ndarray(pixels, format="rgb24").reformat(format="yuv420p")
                container.mux(video_stream.encode(picture))
            container.mux(video_stream.encode(None))

            if audio is not None and audio_stream is not None:
                audio_frame = av.AudioFrame.from_ndarray(audio, format="fltp", layout="stereo")
                audio_frame.sample_rate = sample_rate
                audio_frame.pts = 0
                container.mux(audio_stream.encode(audio_frame))
                container.mux(audio_stream.encode(None))

        detail(
            f"{width}x{height} · {len(frames)} frames · {len(frames) / fps:.2f}s"
            + ("" if audio is not None else " · silent")
        )
    return path


class VideoWriter:
    """An MP4 encoded as it is written, so memory is bounded by one chunk however many follow it.

    `save_video`'s one tensor of finished frames is 9 GB per 15 s at 2 MP: holding four chunks and
    joining them asked for about 74 GB on a 58 GB machine, and the OOM killer took the session with it.
    The soundtrack is small - a minute of 32 kHz stereo is 15 MB - so it is collected whole and written
    when the file is closed.
    """

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: int,
        sample_rate: int = SAMPLE_RATE,
        crf: int | None = None,
    ) -> None:
        """`crf` sets x264's quality (lower is better; its default is 23). The index goes at the front of
        the file, so a browser can start playing before it has all of it."""
        self.path, self.fps, self.sample_rate = path, fps, sample_rate
        self.width, self.height = width, height
        self.frames = 0
        self._container = av.open(str(path), "w", format="mp4", options={"movflags": "faststart"})
        options = {"crf": str(crf)} if crf is not None else {}
        self._video = self._container.add_stream("h264", rate=Fraction(round(fps * 1000), 1000), options=options)
        self._video.width, self._video.height = width, height
        self._video.pix_fmt = "yuv420p"
        self._audio = self._container.add_stream("aac", rate=sample_rate, layout="stereo")

    def write(self, frames: torch.Tensor, batch: int = 16) -> None:
        """`frames [N, H, W, 3]`, float in [0, 1] or uint8."""
        if frames.shape[1:3] != (self.height, self.width):
            raise ValueError(f"frames are {tuple(frames.shape[1:3])}, the file is {(self.height, self.width)}")
        for start in range(0, len(frames), batch):
            piece = frames[start : start + batch]
            pixels = (
                piece if piece.dtype == torch.uint8 else (piece.clamp(0, 1) * 255).round().to(torch.uint8)
            ).numpy()
            for picture in pixels:
                frame = av.VideoFrame.from_ndarray(picture, format="rgb24").reformat(format="yuv420p")
                self._container.mux(self._video.encode(frame))
        self.frames += len(frames)

    def close(self, waveform: torch.Tensor | None) -> Path:
        self._container.mux(self._video.encode(None))
        if waveform is not None:
            samples = int(np.ceil(self.sample_rate / self.fps * self.frames))
            audio = limited(waveform[:, :samples].contiguous().float().numpy())
            audio_frame = av.AudioFrame.from_ndarray(audio, format="fltp", layout="stereo")
            audio_frame.sample_rate = self.sample_rate
            audio_frame.pts = 0
            self._container.mux(self._audio.encode(audio_frame))
            self._container.mux(self._audio.encode(None))
        self._container.close()
        return self.path


def probe_duration(path: str | Path, fallback_fps: int = 24) -> float:
    """Seconds of video in a file, from its frame count and rate (the graph's dangling measuring chain)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames and stream.average_rate:
            return float(stream.frames / stream.average_rate)
        frames = sum(1 for _ in container.decode(video=0))
        return float(frames / (stream.average_rate or fallback_fps))
