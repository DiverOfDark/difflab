"""Loading and saving, plus the joins a long video needs."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import overload

import av
import torch
from PIL import Image

from difflab import config
from difflab.core import events, protocol, record, ui, video_io
from difflab.core.audio import SAMPLE_RATE, load_audio
from difflab.core.imaging import load_image, load_video_frame, load_video_frames, save_result
from difflab.core.types import Clip, Frame


def frame_from_image(path: str | Path) -> Frame:
    return Frame.from_pil(load_image(path), Path(path).name)


def frame_from_video(path: str | Path, index: int = 0) -> Frame:
    return Frame.from_pil(load_video_frame(path, index), f"{Path(path).name} (frame {index})")


def clip_from_video(
    path: str | Path,
    limit: int | None = None,
    with_audio: bool = True,
    start: float = 0.0,
    size: tuple[int, int] | None = None,
) -> Clip:
    """Load a video as a `Clip`, for use as a ref2va reference.

    The soundtrack is resampled to the audio VAE's 32 kHz stereo while it decodes. `start` reads a
    span rather than the head of the file, which is how a long video being edited is handed over one
    chunk at a time; a span's soundtrack is not cut to match it, so a span with audio is refused
    rather than silently describing the wrong seconds.

    `size` decodes straight to that size. A reference clip is encoded at the canvas anyway, and 12 s
    of 1080p is about 7 GB of float pixels to carry there.
    """
    if start > 0 and with_audio:
        raise ValueError("a span's soundtrack would not line up with its frames: pass with_audio=False")
    images, rate = load_video_frames(path, limit, start, size)
    frames = torch.stack([Frame.from_pil(image).pixels for image in images])
    waveform = load_audio(path) if with_audio else None
    return Clip(frames=frames, audio=waveform, fps=round(rate), prompt=f"{Path(path).name}")


def sharpest_index(clip: Clip, window: int = 6) -> int:
    """Pick the frame to hand to the next chunk: the sharpest of the last `window`.

    The final frame of a chunk is often its worst — mid-gesture, mid-blink, motion-blurred — and
    whatever it shows becomes the next chunk's opening and then that chunk's whole trajectory. Ranking
    the tail by Laplacian variance costs nothing and takes a systematic quality sink out of the chain.
    """
    window = max(1, min(window, len(clip)))
    candidates = clip.frames[-window:].mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)  # [w, 1, H, W]
    laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=candidates.dtype).view(
        1, 1, 3, 3
    )
    edges = torch.nn.functional.conv2d(candidates, laplacian)
    sharpness = edges.flatten(1).var(dim=1)
    return len(clip) - window + int(sharpness.argmax())


def color_gain(frame: Frame, toward: Frame, strength: float) -> torch.Tensor:
    """The per-channel gain that pulls `frame`'s channel means `strength` of the way to `toward`'s.

    A chain carries its own colour forward: every chunk opens on the one before, so a cast that enters
    once is never corrected, only inherited. Measured over the five-chunk take of production 9, the red-minus-blue
    spread of the handoff frames ran 15.5, 18.0, 20.7, 20.9, 23.9 - it rose at every single seam and
    fell at none, which no scene does by itself.

    Gain rather than gain and offset: a multiplier moves the cast and leaves contrast where the take put
    it, and black stays black. `strength` is the exponent, so 0 is no correction, 1 matches the means
    outright, and a half pulls halfway - worth preferring, because the light in a take is allowed to
    change and a full match argues with it every time it does.
    """
    source = frame.pixels.mean(dim=(0, 1)).clamp_min(1e-5)
    target = toward.pixels.mean(dim=(0, 1)).clamp_min(1e-5)
    return torch.pow(target / source, strength)


@overload
def recolored(item: Frame, gain: torch.Tensor) -> Frame: ...


@overload
def recolored(item: Clip, gain: torch.Tensor) -> Clip: ...


def recolored(item: Clip | Frame, gain: torch.Tensor) -> Clip | Frame:
    """`item` with `color_gain` applied. A handoff frame and its tail must take the same gain, or the
    next chunk opens on one colour and is handed the moment before it in another."""
    if isinstance(item, Frame):
        return Frame((item.pixels * gain).clamp(0, 1), item.origin)
    return replace(item, frames=(item.frames * gain).clamp(0, 1))


def save(item: Clip | Frame, name: str = "clip", seed: int | None = None, announce: bool = True) -> Path:
    """Write a clip or a still where its kind belongs, say where it went, and return the path.

    A clip is written at *its own* fps and sample rate rather than H3's, so a 30 fps clip read back
    from disk and upscaled is not quietly re-timed to 24. A still is a PNG plus the lossless WebP copy
    beside it.

    `seed` only names the file. Leave it out when nothing was sampled - a borrowed seed is a filename
    that lies about where its contents came from. `announce=False` for a caller that shows the path in
    a panel of its own.
    """
    if isinstance(item, Frame):
        path, _ = save_result(item.pil(), name, seed)
    else:
        path = video_io.save_video(
            item.frames,
            item.audio,
            video_io.output_path(config.VIDEO_DIR, name, seed),
            fps=item.fps,
            sample_rate=item.sample_rate,
        )
    _announce(path, "image" if isinstance(item, Frame) else "video", name, seed)
    if announce:
        ui.line(ui.link(path))
    return path


def _announce(path: Path, media: str, name: str, seed: int | None) -> None:
    """The event that makes anything `save` writes appear on the page - the edited photo, a contact
    sheet, a clip that no take joined - none of which belong to a chunk the page already draws."""
    if not path.resolve().is_relative_to(config.OUTPUT_DIR.resolve()):
        return  # written somewhere the server does not serve from; a path there would 404
    events.emit(
        protocol.Artifact(
            media=media,
            name=name,
            seed=seed,
            path=path.resolve().relative_to(config.OUTPUT_DIR.resolve()).as_posix(),
        )
    )
    # `media` is already `store.IMAGE` / `store.VIDEO`. The row carries the step, which is how the next
    # job in a project's list finds this file and how a re-run knows not to make it again.
    record.artifact_made(path, media, seed)


def save_frame(frame: Frame, path: Path) -> Path:
    """One frame, written exactly where it is asked for: `save` names what a workflow produced, this
    is for what it keeps beside its own state, like the frames a source video is cut at."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.pil().save(path)
    return path


def sheet(columns: Sequence[Sequence[Frame]], height: int = 420, gap: int = 8) -> Frame:
    """One column per subject, one row per version: the raw crop on top, what was made beneath.

    For a plate this IS the acceptance check rather than a step towards one, which is why every column
    starts with the original the plate must be judged against.
    """
    tiles = [[frame.pil() for frame in column] for column in columns]
    scaled = [
        [image.resize((max(1, round(image.width * height / image.height)), height)) for image in column]
        for column in tiles
    ]
    rows = max((len(column) for column in scaled), default=0)
    if not rows:
        raise ValueError("a contact sheet needs at least one picture")
    widths = [max(image.width for image in column) for column in scaled]
    canvas = Image.new(
        "RGB",
        (sum(widths) + gap * (len(widths) - 1), rows * height + gap * (rows - 1)),
        (24, 24, 24),
    )
    left = 0
    for column, width in zip(scaled, widths, strict=True):
        for row, image in enumerate(column):
            canvas.paste(image, (left + (width - image.width) // 2, row * (height + gap)))
        left += width + gap
    return Frame.from_pil(canvas, "contact sheet")


def concat(clips: list[Clip], crossfade: float = 0.0, cuts: Sequence[int | None] | None = None) -> Clip:
    """Join chunks end to end into one long clip.

    `cuts` says where each chunk hands over: when chunk k+1 opened on chunk k's frame `cuts[k]`, chunk k
    is kept up to that frame and only the frames the two share in time - chunk k's from the cut on and
    chunk k+1's first ones - are blended, over at most `crossfade` seconds. Without a cut the chunks are
    butted end to end and `crossfade` blends their boundary frames. The audio follows the frames.
    """
    if not clips:
        raise ValueError("nothing to concatenate")
    if len({(c.frames.shape[1], c.frames.shape[2]) for c in clips}) > 1:
        raise ValueError("clips must share a canvas size")

    fps, rate = clips[0].fps, clips[0].sample_rate
    overlap = round(crossfade * fps)
    handovers = list(cuts) if cuts is not None else [None] * len(clips)
    if len(handovers) not in (len(clips) - 1, len(clips)):
        raise ValueError(f"{len(handovers)} cuts for {len(clips)} clips")
    clips = [
        replace(clip, audio=_fit(clip.audio, len(clip), fps, rate)) if clip.audio is not None else clip
        for clip in clips
    ]
    frames = clips[0].frames
    audio = clips[0].audio
    for previous, clip, cut in zip(clips[:-1], clips[1:], handovers[: len(clips) - 1], strict=True):
        blend = overlap
        if cut is not None:  # drop what the next clip redraws, keeping only the frames to blend
            blend = min(overlap, len(previous) - cut)
            drop = len(previous) - cut - blend
            frames = frames[: len(frames) - drop]
            if audio is not None:
                audio = audio[:, : audio.shape[1] - _samples(drop, fps, rate)]
        frames, audio = _join(frames, audio, clip, blend, fps, rate)
    with ui.step("Joining clips") as detail:
        detail(f"{len(clips)} chunks · {len(frames)} frames · {len(frames) / fps:.2f}s")
    return Clip(frames=frames, audio=audio, fps=fps, sample_rate=rate, prompt=clips[0].prompt)


def join_files(
    parts: Sequence[tuple[Path, Path]],
    path: Path,
    crossfade: float = 0.0,
    cuts: Sequence[int | None] | None = None,
    fps: int | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[int, float]:
    """Join chunk files - `(video.mp4, audio.pt)` pairs - into one video, the way `concat` joins clips.
    Returns `(frames, seconds)`.

    Frames stream through one at a time, holding back only the few that wait for a blend, so a minute
    at 2 MP costs a few frames of memory rather than tens of gigabytes.

    `fps` is read off the first part when it is not given, because the chunks on disk are the authority
    on their own rate - whichever model drew them - and a joiner that took it from a config would
    silently re-time a take rendered at anything else.
    """
    if not parts:
        raise ValueError("nothing to join")
    fps = _rate_of(parts[0][0]) if fps is None else fps
    overlap = round(crossfade * fps)
    handovers: list[int | None] = list(cuts) if cuts is not None else [None] * len(parts)
    # Joining reads and rewrites every frame of every chunk, which for an hour-long take is minutes
    # at the very end of a render - the point at which a page that stops moving looks like a death.
    with ui.progress("Joining chunks", len(parts), unit="chunk") as joined:
        frames, seconds = _join_parts(path, parts, handovers, fps, overlap, sample_rate, joined)
    return frames, seconds


def _join_parts(
    path: Path,
    parts: Sequence[tuple[Path, Path]],
    handovers: list[int | None],
    fps: int,
    overlap: int,
    sample_rate: int,
    joined: Callable[[], None],
) -> tuple[int, float]:
    writer: video_io.VideoWriter | None = None
    held: list[torch.Tensor] = []  # the previous part's frames that wait to be blended with this one's head
    audio: torch.Tensor | None = None
    for number, ((video, soundtrack), cut) in enumerate(zip(parts, handovers, strict=True)):
        cut = cut if number + 1 < len(parts) else None  # the last part plays out
        if cut is not None and cut < len(held):
            raise ValueError(f"part {number + 1} hands over at frame {cut}, inside its own blend")
        incoming = len(held)
        queue: deque[torch.Tensor] = deque()
        kept: list[torch.Tensor] = []
        count = 0
        for index, picture in enumerate(read_frames(video)):
            if writer is None:
                writer = video_io.VideoWriter(path, picture.shape[1], picture.shape[0], fps, sample_rate)
            frame = picture.float().div_(255)
            count = index + 1
            if index < len(held):
                ramp = (index + 1) / (len(held) + 1)  # _ramp(len(held))[index]
                writer.write((held[index] * (1 - ramp) + frame * ramp)[None])
            elif cut is None:
                queue.append(frame)
                if len(queue) > overlap:
                    writer.write(queue.popleft()[None])
            elif index < cut:
                writer.write(frame[None])
            elif index < cut + overlap:
                kept.append(frame)  # shared with the next part: blended with its first frames
            # frames past that are redrawn by the next part, which opened on frame `cut`
        held = kept if cut is not None else list(queue)
        # Anchored to the part's own frames, so a track a few samples short cannot slide the next one early.
        waveform = _fit(torch.load(soundtrack, map_location="cpu", weights_only=True), count, fps, sample_rate)
        if cut is not None:
            waveform = waveform[:, : _samples(cut + len(held), fps, sample_rate)]
        audio = waveform if audio is None else _join_audio(audio, waveform, incoming, fps, sample_rate)
        joined()
    if writer is None:
        raise ValueError("nothing to join")
    for frame in held:
        writer.write(frame[None])
    frames = writer.frames
    writer.close(audio)
    return frames, frames / fps


def _samples(frames: int, fps: int, rate: int) -> int:
    """Audio samples that play under `frames` video frames."""
    return round(frames / fps * rate)


def _fit(audio: torch.Tensor, frames: int, fps: int, rate: int) -> torch.Tensor:
    """`audio` trimmed or padded with silence to exactly the length of `frames` video frames."""
    samples = _samples(frames, fps, rate)
    if audio.shape[1] >= samples:
        return audio[:, :samples]
    return torch.nn.functional.pad(audio, (0, samples - audio.shape[1]))


def _ramp(count: int, dtype: torch.dtype) -> torch.Tensor:
    """Blend weights of the incoming side over `count` shared steps: strictly between 0 and 1, so a blend
    of one or two frames still mixes."""
    return torch.arange(1, count + 1, dtype=dtype) / (count + 1)


def _rate_of(video: Path) -> int:
    """A video file's own frame rate, for a joiner that must not invent one."""
    with av.open(str(video)) as container:
        return round(float(container.streams.video[0].average_rate or 24))


def read_frames(path: Path) -> Iterator[torch.Tensor]:
    """A video's frames as `[H, W, 3]` uint8, one at a time."""
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield torch.from_numpy(frame.to_ndarray(format="rgb24"))


def _join_audio(audio: torch.Tensor, incoming: torch.Tensor | None, overlap: int, fps: int, rate: int) -> torch.Tensor:
    """The soundtrack half of `_join`: ramp the boundary over the same span as the frames."""
    if incoming is None:
        return audio
    samples = min(_samples(overlap, fps, rate), audio.shape[1], incoming.shape[1])
    if samples <= 0:
        return torch.cat([audio, incoming], dim=1)
    ramp = _ramp(samples, audio.dtype).view(1, -1)
    mixed = audio[:, -samples:] * (1 - ramp) + incoming[:, :samples] * ramp
    return torch.cat([audio[:, :-samples], mixed, incoming[:, samples:]], dim=1)


def _join(
    frames: torch.Tensor,
    audio: torch.Tensor | None,
    clip: Clip,
    overlap: int,
    fps: int,
    rate: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    overlap = min(overlap, len(frames), len(clip))
    if overlap <= 0:
        return torch.cat([frames, clip.frames]), _cat_audio(audio, clip.audio)

    ramp = _ramp(overlap, frames.dtype).view(-1, 1, 1, 1)
    blended = frames[-overlap:] * (1 - ramp) + clip.frames[:overlap] * ramp
    joined = torch.cat([frames[:-overlap], blended, clip.frames[overlap:]])

    if audio is None or clip.audio is None:
        return joined, _cat_audio(audio, clip.audio)
    samples = min(round(overlap / fps * rate), audio.shape[1], clip.audio.shape[1])
    if samples <= 0:
        return joined, _cat_audio(audio, clip.audio)
    ramp_a = _ramp(samples, audio.dtype).view(1, -1)
    mixed = audio[:, -samples:] * (1 - ramp_a) + clip.audio[:, :samples] * ramp_a
    return joined, torch.cat([audio[:, :-samples], mixed, clip.audio[:, samples:]], dim=1)


def _cat_audio(left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    return torch.cat([left, right], dim=1)
