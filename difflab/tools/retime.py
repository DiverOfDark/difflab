"""Retiming block: the same seconds, more frames. Sound and duration untouched.

H3 draws 24 fps because its VAE packs time that way, so 48 or 60 is a finishing step rather than a
setting: RIFE renders the picture at any moment between two frames, which adds no moment of its own.
The soundtrack passes through as-is, because the clip is exactly as long as it was.

Why this is cheap where sampling is not: one output frame is one forward pass of a 22 MB network at
the clip's own size, and the per-frame encoder's output is shared by every frame made between the
same pair - at 60 fps that is two encodes serving four interpolations. Compare a chunk at 0.98 MP,
where one sampling step is minutes.

Two things it will not do. It cannot go *down*: dropping frames is not interpolation. And it morphs
across a hard cut instead of holding the frame, so this belongs on a take - one continuous shot - and
on a joined film only if the joins are crossfades rather than cuts.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import torch

from difflab import config
from difflab.core import media as io
from difflab.core import models, ui, video_io
from difflab.core.audio import load_audio
from difflab.core.hub import HubFile
from difflab.core.settings import RetimeSettings
from difflab.core.types import Clip
from difflab.models import rife


def spec(settings: RetimeSettings) -> models.ModelSpec:
    file = HubFile.parse(settings.checkpoint)
    return models.ModelSpec(
        name=f"rife:{file.filename}",
        build=lambda: rife.load(file.fetch(), config.DEVICE),
        checkpoints=(file.path,),
        heavy=False,  # 22 MB: it can sit beside anything
    )


def retime(clip: Clip, fps: int = 60, settings: RetimeSettings | None = None) -> Clip:
    """`clip` at `fps`, with frames invented between the ones it has.

    At 24 -> 60 every fifth output frame is an original and the four between it and the next are made,
    with the timestep cycling 0, 0.4, 0.8, 0.2, 0.6. A frame that lands exactly on a source frame is
    that frame, copied - not a forward pass that would reconstruct it approximately.
    """
    if fps == clip.fps:
        return clip
    if fps < clip.fps:
        raise ValueError(f"retiming only adds frames: {clip.fps} fps cannot become {fps}")

    model: rife.IFNet = models.registry.acquire(spec(settings or RetimeSettings()))
    dtype = next(model.parameters()).dtype
    ratio = Fraction(fps, clip.fps)
    source_count = len(clip)
    count = source_count * ratio.numerator // ratio.denominator
    height, width = int(clip.frames.shape[1]), int(clip.frames.shape[2])
    grid = rife.make_grid(width, height, config.DEVICE)

    out = torch.empty(count, height, width, 3, dtype=torch.float32)
    # The encoder's output for the pair currently being worked between: two entries, which is all a
    # forward pass through the clip ever needs.
    cached: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def pair(index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Frame `index` padded and on the card, with its encoding - made once per source frame."""
        if index not in cached:
            image = clip.frames[index].permute(2, 0, 1)[None].to(config.DEVICE, dtype)
            if grid.padded:
                image = torch.nn.functional.pad(image, grid.padding)
            cached[index] = (image, model.encode(image))
        return cached[index]

    with torch.inference_mode(), ui.progress(f"Retiming to {fps} fps", count, unit="frame") as advance:
        for index in range(count):
            source = index * ratio.denominator // ratio.numerator
            t = (index * ratio.denominator % ratio.numerator) / ratio.numerator
            if t == 0:
                out[index] = clip.frames[source]
                advance()
                continue
            image0, f0 = pair(source)
            image1, f1 = pair(min(source + 1, source_count - 1))
            timestep = torch.full_like(image0[:, :1], t)
            made = model(image0, image1, timestep, f0, f1, grid)
            out[index] = made[0, :, :height, :width].permute(1, 2, 0).float().clamp(0, 1).cpu()
            # The earlier of the two is finished with as soon as the source index moves on.
            for stale in [key for key in cached if key < source]:
                del cached[stale]
            advance()

    return Clip(
        frames=out,
        audio=clip.audio,
        fps=fps,
        sample_rate=clip.sample_rate,
        prompt=clip.prompt,
    )


def retime_file(source: Path, path: Path, fps: int = 60, settings: RetimeSettings | None = None) -> tuple[int, float]:
    """A finished video retimed into `path`, a frame at a time. Returns `(frames, seconds)`.

    `retime` above wants the whole clip as float pixels, which is fine for a chunk and fatal for a
    take: a minute at 2 MP is tens of gigabytes of them, the same reason chunks are joined from files
    rather than concatenated in memory. This holds only the pair being worked between, so the cost is
    two frames on the card whatever the video's length.

    The soundtrack is read whole - a minute of 32 kHz stereo is 15 MB - and written back unchanged.
    """
    with av.open(str(source)) as probe:
        stream = probe.streams.video[0]
        source_fps = round(float(stream.average_rate or 24))  # 24 only if the container says nothing
        expected = stream.frames or round(float(probe.duration or 0) / 1_000_000 * source_fps)
    if fps == source_fps:
        raise ValueError(f"{source.name} is already {fps} fps")
    if fps < source_fps:
        raise ValueError(f"retiming only adds frames: {source_fps} fps cannot become {fps}")

    model: rife.IFNet = models.registry.acquire(spec(settings or RetimeSettings()))
    dtype = next(model.parameters()).dtype
    ratio = Fraction(fps, source_fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid: rife.Grid | None = None
    writer: video_io.VideoWriter | None = None

    def prepare(picture: torch.Tensor) -> torch.Tensor:
        """A decoded frame on the card, padded to what the flow pyramid needs."""
        assert grid is not None  # built from the first frame, before anything is prepared
        image = picture.permute(2, 0, 1)[None].to(config.DEVICE, dtype).div(255)
        return torch.nn.functional.pad(image, grid.padding) if grid.padded else image

    total = expected * ratio.numerator // ratio.denominator
    with torch.inference_mode(), ui.progress(f"Retiming to {fps} fps", total, unit="frame") as advance:
        frames = io.read_frames(source)
        first = next(frames, None)
        if first is None:
            raise ValueError(f"no frames in {source}")
        height, width = int(first.shape[0]), int(first.shape[1])
        grid = rife.make_grid(width, height, config.DEVICE)
        writer = video_io.VideoWriter(path, width, height, fps)

        # `before` is source frame `index` and `after` the one following it. Every source frame gets
        # its own turn, the last included - it is its own successor there, so the output frames after
        # it are copies rather than a shorter video.
        index = 0
        before = first
        following = next(frames, None)
        ended = following is None
        after = before if following is None else following
        f_before = model.encode(prepare(before))
        f_after = f_before if following is None else model.encode(prepare(after))
        while True:
            start = -(-index * ratio.numerator // ratio.denominator)
            end = -(-(index + 1) * ratio.numerator // ratio.denominator)
            if ended:
                # The source's true length is only known here, and it decides the output's: floored,
                # like `retime`'s, so a retimed video is never longer than the one it was made from.
                end = min(end, (index + 1) * ratio.numerator // ratio.denominator)
            for out in range(start, end):
                t = (out * ratio.denominator % ratio.numerator) / ratio.numerator
                if t == 0:
                    writer.write(before[None])  # an output frame that lands on a source frame is it
                else:
                    image0, image1 = prepare(before), prepare(after)
                    made = model(image0, image1, torch.full_like(image0[:, :1], t), f_before, f_after, grid)
                    writer.write(made[0, :, :height, :width].permute(1, 2, 0).float().clamp(0, 1).cpu()[None])
                advance()
            if ended:
                break
            index, before, f_before = index + 1, after, f_after
            following = next(frames, None)
            if following is None:
                ended, after, f_after = True, before, f_before
            else:
                after, f_after = following, model.encode(prepare(following))

    written = writer.frames
    writer.close(load_audio(source))
    with ui.step(f"Retimed to {fps} fps") as detail:
        detail(f"{expected} → {written} frames · {written / fps:.2f}s · {ui.link(path)}")
    return written, written / fps
