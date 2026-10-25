"""Reading a soundtrack off a file, at the rate the audio VAE wants.

The VAE is fixed at 32 kHz stereo, so that is what comes back regardless of what the file holds.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

SAMPLE_RATE = 32000
CHANNELS = 2
# The VAE decodes past full scale: one chapter (2026-10-03) peaked at +0.68 dBFS on two samples, which
# AAC keeps and playback clips. Below KNEE nothing changes; above it the peak bends under CEILING (-1 dBFS).
KNEE, CEILING = 0.7, 0.89


def limited(audio: np.ndarray) -> np.ndarray:
    loud = np.abs(audio) > KNEE
    if not loud.any():
        return audio
    room = CEILING - KNEE
    bent = np.sign(audio) * (KNEE + room * np.tanh((np.abs(audio) - KNEE) / room))
    return np.where(loud, bent, audio).astype(audio.dtype)


def load_audio(path: str | Path, rate: int = SAMPLE_RATE) -> torch.Tensor | None:
    """The file's soundtrack as `[2, samples]` float in [-1, 1], or `None` if it has none."""
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray())
    if not chunks:
        return None
    waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).float()
    if waveform.shape[0] == 1:  # a mono file resampled to stereo can still come back single-planar
        waveform = waveform.repeat(CHANNELS, 1)
    return waveform[:CHANNELS].clamp(-1.0, 1.0)


def resampled(waveform: torch.Tensor, rate: int, to: int = SAMPLE_RATE) -> torch.Tensor:
    """`[2, samples]` at `rate` -> the same sound at `to`, through the resampler `load_audio` reads files with."""
    if rate == to:
        return waveform
    frame = av.AudioFrame.from_ndarray(waveform.float().contiguous().numpy(), format="fltp", layout="stereo")
    frame.sample_rate = rate
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=to)
    chunks = [out.to_ndarray() for out in (*resampler.resample(frame), *resampler.resample(None))]
    return torch.from_numpy(np.concatenate(chunks, axis=1)).float()


def trim_to_frames(waveform: torch.Tensor, frames: int, fps: int, rate: int = SAMPLE_RATE) -> torch.Tensor:
    """A reference clip's frame count is trimmed to the grid the VAE encodes in, so its soundtrack has
    to follow, or the two would describe different spans of the same moment."""
    samples = int(Fraction(frames * rate, fps))
    if waveform.shape[1] >= samples:
        return waveform[:, :samples]
    padding = torch.zeros(waveform.shape[0], samples - waveform.shape[1], dtype=waveform.dtype)
    return torch.cat([waveform, padding], dim=1)
