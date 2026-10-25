"""Loading and saving, plus the joins a long video needs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple, overload

import av
import torch
from PIL import Image

from difflab import config
from difflab.core import events, protocol, record, ui, video_io
from difflab.core.audio import SAMPLE_RATE, load_audio
from difflab.core.imaging import load_image, load_video_frames, save_result
from difflab.core.types import Clip, Frame

# x264's default 23 on the join and on RIFE's retime, after chunks written at 16, halved a 4K finish's
# fine detail (laplacian variance 37 -> 21 2026-10-06): every re-encode is a generation.
JOIN_CRF = 16


def frame_from_image(path: str | Path) -> Frame:
    return Frame.from_pil(load_image(path), Path(path).name)


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
    once is never corrected, only inherited. Measured over a five-chunk take, the red-minus-blue
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


# Measured on a five-chunk take (2026-10-04): within each chunk a* rose by
# 2-3 and the spread of a*, b* and L by about x1.15, and each chunk opened on the one before's colour,
# so a* ran 2.4 (the photo) to 11.0 and chroma spread x2.3 by 38 s. A change of place is far larger than
# one chunk's drift (a blue sky moves b* by tens), so a correction capped at drift's size leaves it alone.
HOLD_SHIFT = 4.0  # most a*/b* mean moved per chunk, in Lab units
HOLD_RATIO = (0.8, 1.25)  # most a spread is scaled per chunk


class Hold(NamedTuple):
    mean: torch.Tensor
    ratio: torch.Tensor
    shift: torch.Tensor


_XYZ = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def _lab(rgb: torch.Tensor) -> torch.Tensor:
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _XYZ.T / _WHITE
    f = torch.where(xyz > 216 / 24389, xyz.clamp_min(0) ** (1 / 3), (24389 / 27 * xyz + 16) / 116)
    return torch.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])), -1)


def _rgb(lab: torch.Tensor) -> torch.Tensor:
    fy = (lab[..., 0] + 16) / 116
    f = torch.stack((fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200), -1)
    xyz = torch.where(f**3 > 216 / 24389, f**3, (116 * f - 16) / (24389 / 27)) * _WHITE
    linear = (xyz @ torch.linalg.inv(_XYZ).T).clamp(0, 1)
    return torch.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def _lab_stats(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lab = _lab(pixels[::4, ::4].float()).reshape(-1, 3)
    return lab.mean(0), lab.std(0).clamp_min(1e-3)


def hold_colour(clip: Clip, toward: Frame, strength: float) -> Clip:
    """`clip` with the colour cast and the spread it gained since `toward`, its opening, taken out.

    Brightness is left to the take, since light may change; the cast and the chroma and contrast spread
    are what a chunk inflates on its own. Measured per frame, smoothed over a second so a passing change
    survives, and capped at one chunk's drift (HOLD_SHIFT, HOLD_RATIO) so a real change of place does."""
    if not strength:
        return clip
    return replace(clip, frames=held(clip.frames, colour_hold(clip.frames, clip.fps, toward, strength)))


def colour_hold(frames: torch.Tensor, fps: int, toward: Frame, strength: float) -> list[Hold]:
    """Per frame, what `hold_colour` does to it: measured on one decode, applicable to another of the same
    latents at any size."""
    target_mean, target_spread = _lab_stats(toward.pixels)
    stats = [_lab_stats(frame) for frame in frames]
    window = max(1, fps)
    means = _smooth(torch.stack([m for m, _ in stats]), window)
    spreads = _smooth(torch.stack([s for _, s in stats]), window)
    out = []
    for mean, spread in zip(means, spreads, strict=True):
        shift = ((target_mean - mean) * strength).clamp(-HOLD_SHIFT, HOLD_SHIFT)
        shift[0] = 0.0
        out.append(Hold(mean, ((target_spread / spread) ** strength).clamp(*HOLD_RATIO), shift))
    return out


def held(frames: torch.Tensor, holds: Sequence[Hold]) -> torch.Tensor:
    out = frames.clone()
    for index, (mean, ratio, shift) in enumerate(holds):
        lab = _lab(out[index].float())
        out[index] = _rgb((lab - mean) * ratio + mean + shift).to(out.dtype)
    return out


def _smooth(rows: torch.Tensor, window: int) -> torch.Tensor:
    """A centred moving average with the ends repeated, so the first and last second are not dragged."""
    pad = window // 2
    padded = torch.nn.functional.pad(rows.t()[None], (pad, pad), mode="replicate")
    kernel = torch.ones(rows.shape[1], 1, window, dtype=rows.dtype) / window
    return torch.nn.functional.conv1d(padded, kernel, groups=rows.shape[1])[0].t()[: len(rows)]


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
    from disk and upscaled is not quietly re-timed to 24.

    `seed` only names the file. Leave it out when nothing was sampled - a borrowed seed is a filename
    that lies about where its contents came from. `announce=False` for a caller that shows the path in
    a panel of its own.
    """
    if isinstance(item, Frame):
        path = save_result(item.pil(), name, seed)
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


def sheet(columns: Sequence[Sequence[Frame]], height: int = 420, gap: int = 8) -> Frame:
    """One column per subject, one row per version: the raw crop on top, what was made beneath."""
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


class Join(NamedTuple):
    """How one part hands over to the next: `cut` is the frame the next part opened on, if `chained`. A
    chained join blends `fade` seconds of the frames both parts share, past the cut; any other join
    fades over the `fade` seconds before it, so 0 is a hard cut."""

    cut: int
    fade: float
    chained: bool


def join_files(
    parts: Sequence[tuple[Path, Path]],
    path: Path,
    joins: Sequence[Join],
    fps: int | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[int, float]:
    """Join chunk files - `(video.mp4, audio.pt)` pairs - into one video, `joins[k]` between part k and
    k+1. Returns `(frames, seconds)`.

    Frames stream through one at a time, holding back only the few that wait for a blend, so a minute
    at 2 MP costs a few frames of memory rather than tens of gigabytes.

    `fps` is read off the first part when it is not given, because the chunks on disk are the authority
    on their own rate - whichever model drew them - and a joiner that took it from a config would
    silently re-time a take rendered at anything else.
    """
    if not parts:
        raise ValueError("nothing to join")
    fps = rate_of(parts[0][0]) if fps is None else fps
    spans = layout([frames_of(video) for video, _ in parts], joins, fps)
    # Joining reads and rewrites every frame of every chunk, which for an hour-long take is minutes
    # at the very end of a render - the point at which a page that stops moving looks like a death.
    with ui.progress("Joining chunks", len(parts), unit="chunk") as joined:
        frames, seconds = _join_parts(path, parts, spans, fps, sample_rate, joined)
    return frames, seconds


class Span(NamedTuple):
    """What a part gives the joined video: its frames from `head` (those before were blended into the part
    before's `kept`) to `end`, then `kept` frames from `end` blended with the next part's head."""

    head: int
    end: int
    kept: int


def layout(counts: Sequence[int], joins: Sequence[Join], fps: int) -> list[Span]:
    spans: list[Span] = []
    head = 0
    for number, count in enumerate(counts):
        if number + 1 == len(counts):
            spans.append(Span(head, count, 0))
            break
        cut, fade, chained = joins[number]
        if cut < head:
            raise ValueError(f"part {number + 1} hands over at frame {cut}, inside its own blend")
        if chained:
            kept = min(round(fade * fps), count - cut)
            spans.append(Span(head, cut, kept))
        else:
            kept = min(round(fade * fps), cut - head)
            spans.append(Span(head, cut - kept, kept))
        head = kept
    return spans


def starts(counts: Sequence[int], joins: Sequence[Join], fps: int) -> list[float]:
    """Where each part's first frame plays in the joined video, in seconds."""
    out, at = [], 0
    for span in layout(counts, joins, fps):
        out.append(at / fps)
        at += span.end
    return out


def frames_of(video: Path) -> int:
    """A video's frame count from its container, or by decoding it when the container does not say."""
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


def _join_parts(
    path: Path,
    parts: Sequence[tuple[Path, Path]],
    spans: Sequence[Span],
    fps: int,
    sample_rate: int,
    joined: Callable[[], None],
) -> tuple[int, float]:
    writer: video_io.VideoWriter | None = None
    held: list[torch.Tensor] = []  # the previous part's frames that wait to be blended with this one's head
    audio: torch.Tensor | None = None
    for (video, soundtrack), span in zip(parts, spans, strict=True):
        kept: list[torch.Tensor] = []
        count = 0
        for index, picture in enumerate(read_frames(video)):
            if writer is None:
                writer = video_io.VideoWriter(path, picture.shape[1], picture.shape[0], fps, sample_rate, crf=JOIN_CRF)
            frame = picture.float().div_(255)
            count = index + 1
            if index < span.head:
                ramp = (index + 1) / (span.head + 1)  # _ramp(span.head)[index]
                writer.write((held[index] * (1 - ramp) + frame * ramp)[None])
            elif index < span.end:
                writer.write(frame[None])
            elif index < span.end + span.kept:
                kept.append(frame)
            # frames past that are redrawn by the next part, which opened on its cut
        # Anchored to the part's own frames, so a track a few samples short cannot slide the next one early.
        waveform = _fit(torch.load(soundtrack, map_location="cpu", weights_only=True), count, fps, sample_rate)
        waveform = waveform[:, : _samples(span.end + span.kept, fps, sample_rate)]
        audio = waveform if audio is None else _join_audio(audio, waveform, span.head, fps, sample_rate)
        held = kept
        joined()
    if writer is None:
        raise ValueError("nothing to join")
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


def rate_of(video: Path) -> int:
    """A video file's own frame rate, for a joiner that must not invent one."""
    with av.open(str(video)) as container:
        return round(float(container.streams.video[0].average_rate or 24))


def read_frames(path: Path) -> Iterator[torch.Tensor]:
    """A video's frames as `[H, W, 3]` uint8, one at a time."""
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield torch.from_numpy(frame.to_ndarray(format="rgb24"))


def _join_audio(audio: torch.Tensor, incoming: torch.Tensor | None, overlap: int, fps: int, rate: int) -> torch.Tensor:
    """The soundtrack half of `_join_parts`: ramp the boundary over the same span as the frames."""
    if incoming is None:
        return audio
    samples = min(_samples(overlap, fps, rate), audio.shape[1], incoming.shape[1])
    if samples <= 0:
        return torch.cat([audio, incoming], dim=1)
    ramp = _ramp(samples, audio.dtype).view(1, -1)
    mixed = audio[:, -samples:] * (1 - ramp) + incoming[:, :samples] * ramp
    return torch.cat([audio[:, :-samples], mixed, incoming[:, samples:]], dim=1)
