"""A joined chapter levelled to a loudness, with its dialogue as subtitles; the chapters as one film.

Through the ffmpeg program rather than PyAV: the ebur128 measurement, the concat demuxer's stream copy,
chapter metadata and a mov_text track are each one command there.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import av

from difflab.core import provenance

# dBFS, the limiter's ceiling. It holds sample peaks; a levelled chapter still measured -0.1 dBTP.
LIMIT = -1.5
AUDIO_BITRATE = "192k"
SILENT = -70.0  # LUFS, what ebur128 reports for a soundtrack of silence


class Cue(NamedTuple):
    start: float
    end: float
    text: str


class Chapter(NamedTuple):
    video: Path
    title: str
    cues: Sequence[Cue]


def _ffmpeg(*arguments: str) -> str:
    program = shutil.which("ffmpeg")
    if program is None:
        raise RuntimeError("assembling needs the ffmpeg program on PATH")
    done = subprocess.run(
        [program, "-hide_banner", "-nostdin", "-nostats", "-y", *arguments], capture_output=True, text=True, check=False
    )
    if done.returncode:
        raise RuntimeError(f"ffmpeg failed: {done.stderr.strip()[-1500:]}")
    return done.stderr


def loudness(video: Path) -> tuple[float, float]:
    """`(integrated LUFS, true peak dBTP)` of the first soundtrack."""
    said = _ffmpeg("-i", str(video), "-map", "0:a:0", "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-")
    summary = said[said.rindex("Integrated loudness:") :]
    level = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary)
    if level is None or peak is None:
        raise RuntimeError(f"ebur128 printed no summary for {video.name}")
    return float(level.group(1)), float(peak.group(1))


def master(video: Path, destination: Path, target: float, cues: Sequence[Cue], track: bool) -> float | None:
    """`video` at `target` LUFS into `destination`, the picture copied, `cues` muxed as a mov_text track when
    `track`. Returns the integrated loudness written, None for silence."""
    with av.open(str(video)) as container:
        rate = container.streams.audio[0].sample_rate if container.streams.audio else None
    with tempfile.TemporaryDirectory() as scratch:
        sound = Path(scratch) / "levelled.wav"
        level = _level(video, sound, target, rate) if rate is not None else None
        subtitles = Path(scratch) / "dialogue.srt"
        subtitles.write_text(srt(cues))
        muxed = track and bool(cues)
        heard = ["-i", str(sound)] if rate is not None else []
        _ffmpeg(
            "-i", str(video), *heard, *(["-i", str(subtitles)] if muxed else []),
            "-map", "0:v:0", *(["-map", "1:a:0", "-c:a", "aac", "-b:a", AUDIO_BITRATE] if heard else []),
            *(["-map", f"{1 + bool(heard)}:0", "-c:s", "mov_text", "-metadata:s:s:0", "handler_name=Dialogue"]
              if muxed else []),
            "-c:v", "copy", "-movflags", "+faststart", str(destination),
        )  # fmt: skip
    return level


LEVEL_PASSES = 6
LEVEL_CLOSE = 0.1  # LU


def _level(video: Path, sound: Path, target: float, rate: int) -> float | None:
    """The soundtrack at `target` LUFS into `sound`, a limiter holding its peaks. Returns the level reached,
    None for silence, which is left as it is."""
    # A gain from a measured pass, not loudnorm: on H3's 32 kHz soundtracks loudnorm's own first pass read
    # 0.9 LU hot (-11.4 against ebur128's -12.3 2026-10-09), and its dynamic mode, which it
    # falls into whenever a chapter's range is over its target, compresses the fights. The limiter takes
    # loudness with the peaks (+3.5 dB of gain gave +1.4 LU), so the gain is raised by what
    # is still missing and applied again.
    level, peak = loudness(video)
    if level <= SILENT:
        _ffmpeg("-i", str(video), "-map", "0:a:0", "-c:a", "pcm_s16le", str(sound))
        return None
    gain, reached = target - level, level
    tried: list[tuple[float, float]] = []
    for _ in range(LEVEL_PASSES):
        chain = f"volume={gain:.2f}dB"
        if peak + gain > LIMIT:
            chain += f",alimiter=limit={10 ** (LIMIT / 20):.4f}:level=false"
        _ffmpeg("-i", str(video), "-map", "0:a:0", "-af", chain, "-ar", str(rate), "-c:a", "pcm_f32le", str(sound))
        reached, _ = loudness(sound)
        if abs(target - reached) < LEVEL_CLOSE:
            break
        tried.append((gain, reached))
        # The secant through the last two passes: under the limiter a decibel of gain buys about half a LU.
        (g0, r0), (g1, r1) = tried[-2:] if len(tried) > 1 else ((gain - 1.0, reached - 1.0), (gain, reached))
        gain += (target - reached) * (g1 - g0) / (r1 - r0) if r1 != r0 else target - reached
    return reached


def film(chapters: Sequence[Chapter], destination: Path, title: str, markers: bool, track: bool) -> list[Cue]:
    """The chapters end to end with their streams copied, a chapter marker at each when `markers`.
    Returns every cue at its time in the film."""
    rates = {chapter.video.name: _frame_rate(chapter.video) for chapter in chapters}
    if len(set(rates.values())) > 1:
        said = ", ".join(f"{name} {rate:g} fps" for name, rate in rates.items())
        raise ValueError(f"the chapters' streams are copied, so they share one frame rate: {said}")
    cues: list[Cue] = []
    meta = [";FFMETADATA1", f"title={_escape(title)}", *(f"{k}={_escape(v)}" for k, v in provenance.MP4_TAGS.items())]
    at = 0.0
    for chapter in chapters:
        length = seconds_of(chapter.video)
        cues += [Cue(at + c.start, at + c.end, c.text) for c in chapter.cues]
        if markers:
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(at * 1000)}",
                     f"END={round((at + length) * 1000)}", f"title={_escape(chapter.title)}"]  # fmt: skip
        at += length
    with tempfile.TemporaryDirectory() as scratch:
        listed, metadata, subtitles = (Path(scratch) / name for name in ("parts.txt", "meta.txt", "dialogue.srt"))
        listed.write_text("".join(f"file '{_quoted(chapter.video)}'\n" for chapter in chapters))
        metadata.write_text("\n".join(meta) + "\n")
        subtitles.write_text(srt(cues))
        muxed = track and bool(cues)
        _ffmpeg(
            "-f", "concat", "-safe", "0", "-i", str(listed), "-i", str(metadata),
            *(["-i", str(subtitles)] if muxed else []),
            "-map", "0:v:0", "-map", "0:a:0?", *(["-map", "2:0"] if muxed else []),
            "-map_metadata", "1", "-map_chapters", "1" if markers else "-1",
            "-c:v", "copy", "-c:a", "copy",
            *(["-c:s", "mov_text", "-metadata:s:s:0", "handler_name=Dialogue"] if muxed else []),
            "-movflags", "+faststart", str(destination),
        )  # fmt: skip
    return cues


def _frame_rate(video: Path) -> float:
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError(f"{video.name} has no video stream")
        rate = container.streams.video[0].average_rate
        return round(float(rate), 3) if rate else 0.0


def seconds_of(video: Path) -> float:
    with av.open(str(video)) as container:
        if container.duration is None:
            raise ValueError(f"{video} does not say how long it is")
        return float(container.duration) / 1_000_000


def srt(cues: Sequence[Cue]) -> str:
    return "".join(f"{n}\n{_stamp(c.start, ',')} --> {_stamp(c.end, ',')}\n{c.text}\n\n" for n, c in enumerate(cues, 1))


def vtt(cues: Sequence[Cue]) -> str:
    return "WEBVTT\n\n" + "".join(f"{_stamp(c.start, '.')} --> {_stamp(c.end, '.')}\n{c.text}\n\n" for c in cues)


def _stamp(seconds: float, mark: str) -> str:
    ms = round(seconds * 1000)
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}{mark}{ms % 1000:03d}"


def _escape(text: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", text)


def _quoted(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")
