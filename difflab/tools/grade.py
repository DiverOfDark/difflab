"""Grading block: a finished take pulled back to the colour it opened in.

A chain inherits its own colour. Chunk k+1 opens on chunk k's pixels, so whatever the model does to
the palette is carried forward rather than corrected, and what it does is regress toward its own
prior. Measured over a six-chunk take that opens on coals, green rose 38% against red and blue more
than tripled - at every join and at none of them backwards.

`Continuity.color_anchor` fixes the *cause*, by correcting the frame each chunk opens on. It is
the better fix and it is expensive: the opening frame's pixels are part of `render_chunk.chunk_key`, so
turning it on re-renders every chunk after the first. This is the other half - a LUT over the
finished film, at the price of one pass over an mp4.

Two decisions make it a grade rather than a filter. The reference is the take's own opening, not a
white point and not a preset, so the beginning of a graded take is untouched by construction and only
what left it is pulled back. And the correction follows the channel means smoothed over about a
second rather than each frame: drift is slow, so a cloud crossing the sun or a lamp switched on moves
faster than the window and survives, where correcting frame by frame would flatten it and add its own
flicker.

Gain and no offset, and `strength` as an exponent, as in `io.color_gain`.
"""

from __future__ import annotations

from pathlib import Path

import av
import torch

from difflab.core import media as io
from difflab.core import ui, video_io
from difflab.core.audio import load_audio

# A take that fades to black has channel means near zero, and a ratio against them would blow the
# picture out. Wide enough not to bind on real drift: the six-chunk take above needed blue at x0.47,
# so a floor of 0.5 clipped the one channel that had moved most.
GAIN_RANGE = (0.25, 4.0)


def grade(
    source: Path,
    path: Path | None = None,
    strength: float = 0.5,
    reference_seconds: float = 1.0,
    smooth_seconds: float = 1.0,
) -> Path:
    """`source` with its colour drift taken out, written beside it. Returns the new file.

    Read twice - once to measure what the colour does over time, once to correct it - because a gain
    for frame 0 depends on what frame 900 did, and holding a take's frames in memory to save the
    second decode is what the streaming joiner exists to avoid.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength={strength:g}: 0 changes nothing, 1 matches the opening outright")
    path = path or source.with_name(f"{source.stem}_graded{source.suffix}")

    with av.open(str(source)) as probe:
        stream = probe.streams.video[0]
        fps = round(float(stream.average_rate or 24))  # 24 only if the container says nothing
        declared = int(stream.frames or 0)  # 0 for a container that does not say; then nothing to count

    # A whole decode pass before a single frame is written, which on an hour of video is minutes of a
    # page saying "Measuring" and nothing else. A container that will not give a frame count leaves it
    # zero, and `ui.progress` is skipped rather than given a total it would have to invent.
    with ui.progress(f"Measuring {source.name}", declared, unit="frame") as advance:
        rows = []
        for frame in io.read_frames(source):
            rows.append(frame.float().mean(dim=(0, 1)))
            advance()
        means = torch.stack(rows) if rows else torch.zeros(0, 3)
    if not len(means):
        raise ValueError(f"no frames in {source}")

    gains = _gains(means, fps, strength, reference_seconds, smooth_seconds)
    drift = means[-max(1, round(fps)) :].mean(0) / means[: max(1, round(fps))].mean(0).clamp_min(1e-5)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer: video_io.VideoWriter | None = None
    with ui.progress(f"Grading to {path.name}", len(means), unit="frame") as advance:
        for index, frame in enumerate(io.read_frames(source)):
            if writer is None:
                writer = video_io.VideoWriter(path, int(frame.shape[1]), int(frame.shape[0]), fps)
            corrected = (frame.float() * gains[min(index, len(gains) - 1)]).clamp(0, 255).to(torch.uint8)
            writer.write(corrected[None])
            advance()
    assert writer is not None  # the frame count above is not zero
    writer.close(load_audio(source))

    with ui.step("Graded") as detail:
        ends = gains[-1].tolist()
        detail(
            f"drift was R×{drift[0]:.2f} G×{drift[1]:.2f} B×{drift[2]:.2f} · "
            f"last gain R×{ends[0]:.2f} G×{ends[1]:.2f} B×{ends[2]:.2f}"
        )
    ui.line(ui.link(path))
    return path


def _gains(
    means: torch.Tensor, fps: int, strength: float, reference_seconds: float, smooth_seconds: float
) -> torch.Tensor:
    """One per-channel gain per frame. The reference is the opening's own average, so the gain there
    is 1 and the take begins as it always did."""
    reference = means[: max(1, round(reference_seconds * fps))].mean(dim=0).clamp_min(1e-5)
    smoothed = _smooth(means, max(1, round(smooth_seconds * fps)))
    gains = torch.pow(reference / smoothed.clamp_min(1e-5), strength)
    return gains.clamp(*GAIN_RANGE)


def _smooth(means: torch.Tensor, window: int) -> torch.Tensor:
    """A centred moving average over `window` frames. Replicate padding, not zeros: a zero-padded
    average would drag the first and last seconds toward black and invent a correction exactly where
    the take is least in need of one."""
    if window <= 1:
        return means
    pad = window // 2
    padded = torch.nn.functional.pad(means.t()[None], (pad, pad), mode="replicate")
    kernel = torch.ones(3, 1, window, dtype=means.dtype) / window
    averaged = torch.nn.functional.conv1d(padded, kernel, groups=3)[0].t()
    return averaged[: len(means)]
