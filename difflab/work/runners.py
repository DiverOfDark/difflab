"""What each tool does when a job of it runs: inputs by name in, outputs by name out.

An input is a value, a path, or another job's output; the worker resolves the last before calling.
Paths in outputs are relative to `output/`, the same way artifacts are recorded.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from difflab import config, store
from difflab.core import events, media, protocol, record, ui, video_io
from difflab.core.cache import cache
from difflab.core.hub import HubFile
from difflab.core.imaging import video_length
from difflab.core.keys import key
from difflab.core.settings import (
    Continuity,
    EditSettings,
    EnlargePixelsSettings,
    JudgeSettings,
    RetimeSettings,
    VlmSettings,
)
from difflab.core.types import Clip, Frame
from difflab.models.minimax_h3 import pipeline
from difflab.models.minimax_h3.settings import VideoPromptSettings, VideoSettings
from difflab.screenplay import compiler, drawn
from difflab.tools import (
    assemble,
    compose_keyframe,
    draw_reference,
    edit_image,
    enlarge_pixels,
    grade,
    handoff,
    judge,
    previews,
    render_chunk,
    retime,
    write_prompt,
)
from difflab.work import Skipped
from difflab.work.plan_runners import PLAN_RUNNERS

Runner = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else config.OUTPUT_DIR / path


def _frame(value: Any) -> Frame:
    return media.frame_from_image(_path(value))


def run_write_instruction(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    text = write_prompt.write_edit_instruction(_frame(inputs["photo"]), str(inputs["idea"]), VlmSettings(**settings))
    return {"text": text}


def run_write_video_prompt(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    text = write_prompt.write_video_prompt(
        _frame(inputs["photo"]), str(inputs["idea"]), VideoPromptSettings(**settings)
    )
    return {"text": text}


def run_edit_image(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    chosen = EditSettings(**settings)
    made = edit_image.edit(_frame(inputs["photo"]), str(inputs["instruction"]), chosen)
    png = media.save(made, HubFile.parse(chosen.checkpoint).path.stem, chosen.seed, announce=False)
    return {"image": store.relative(png)}


def run_render_clip(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    chosen = VideoSettings(**settings)
    references = [_frame(p) for p in inputs.get("references", [])]
    clip = render_chunk.render(_frame(inputs["photo"]), str(inputs["prompt"]), chosen, references=references)
    path = media.save(clip, HubFile.parse(chosen.checkpoint).path.stem, chosen.seed, announce=False)
    return {"video": store.relative(path)}


def run_enlarge_pixels(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    clip = media.clip_from_video(_path(inputs["video"]))
    bigger = enlarge_pixels.upscale(clip, float(inputs["megapixels"]), settings=EnlargePixelsSettings(**settings))
    return {"video": store.relative(media.save(bigger, "upscaled", announce=False))}


def run_grade(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    path = grade.grade(_path(inputs["video"]), strength=float(inputs["strength"]))
    return {"video": store.relative(path)}


def run_retime(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    source = _path(inputs["video"])
    fps = int(inputs["fps"])
    destination = source.with_name(f"{source.stem}_{fps}fps{source.suffix}")
    retime.retime_file(source, destination, fps, RetimeSettings(**settings))
    return {"video": store.relative(destination)}


def run_render_shot(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """One shot of a chain: its chunk, then what the next shot opens on. Both are keyed by everything
    they were made of, so a shot already on disk is skipped and a re-rolled one moves its own key and
    the key of every shot chained after it."""
    video = VideoSettings(**settings["video"]).with_(seed=int(inputs["seed"]), duration=float(inputs["seconds"]))
    continuity = Continuity(**settings["continuity"])
    prompt = str(inputs["prompt"])
    opening = _frame(inputs["opening"])
    closing = _frame(inputs["closing"]) if inputs.get("closing") else None
    tail = handoff.load_tail(_path(inputs["tail"])) if inputs.get("tail") else None
    strength = continuity.color_anchor
    anchor = _frame(inputs["anchor"]) if inputs.get("anchor") and strength else None
    references = [_frame(p) for p in inputs.get("references", [])]
    position = int(inputs.get("position", 0))
    clips, scale = _excerpt(inputs.get("source"), opening, video)

    chunk = render_chunk.chunk_key(opening, prompt, video, references, tail, clips, scale, closing)
    # A shot that ends on the next one's keyframe is cut on its very last frame, the one pinned to it.
    window = 1 if closing is not None else continuity.handoff_window
    at = handoff.handoff_key(chunk, window, continuity.tail_frames, anchor, strength)
    latents_file, mp4, soundtrack = cache.path(chunk, ".pt"), cache.path(chunk, ".mp4"), cache.path(chunk, ".audio.pt")
    frame_file, tail_file, _ = handoff.files(at)
    outputs: dict[str, Any] = {
        "key": str(chunk),
        "latents": store.relative(latents_file),
        "chunk": store.relative(mp4),
        "audio": store.relative(soundtrack),
        "frame": store.relative(frame_file),
        "tail": store.relative(tail_file) if continuity.tail_frames else None,
    }
    events.emit(
        protocol.ItemInfo(
            index=position,
            version=1,
            prompt=prompt,
            seed=video.seed,
            opening=_thumbnail(opening, f"thumbs/shot{position + 1}_opening.jpg"),
            references=[_thumbnail(ref, f"thumbs/shot{position + 1}_ref{n}.jpg") for n, ref in enumerate(references)],
        )
    )
    shown = protocol.ItemVideo(index=position, version=1, video=f"/output/{outputs['chunk']}", seed=video.seed)
    if latents_file.is_file() and mp4.is_file() and soundtrack.is_file() and handoff.present(at):
        ui.note(f"shot {position + 1}: already made as {chunk}")
        events.emit(shown)
        return Skipped({**outputs, "cut": handoff.cut_of(at)})

    started = time.perf_counter()
    events.emit(protocol.ItemBegin(index=position))
    previews.target(position, f"shot{position + 1}")
    try:
        latents = render_chunk.cached_latents(
            opening,
            prompt,
            video,
            references,
            tail,
            clips=clips,
            clip_scale=scale,
            canvas_from=opening,
            closing=closing,
        )
    finally:
        previews.target(None, "clip")
    clip = render_chunk.decode(latents)
    with ui.step("Handoff") as detail:
        made = handoff.handoff(clip, window, continuity.tail_frames, anchor, strength)
        handoff.save(made, at)
        carried = f" · {len(made.tail)} frames of tail" if made.tail is not None else ""
        detail(f"frame {made.cut + 1} of {len(clip)}{carried}")
    if not (mp4.is_file() and soundtrack.is_file()):
        with ui.step("Writing chunk") as detail:
            _write_chunk(clip, mp4, soundtrack)
            detail(f"{len(clip)} frames · {clip.duration:.2f}s")
    record.artifact_made(mp4, store.VIDEO, video.seed, key=str(chunk))
    record.artifact_made(frame_file, store.IMAGE, key=str(at))
    events.emit(shown)
    events.emit(protocol.ItemEnd(index=position, seconds=round(time.perf_counter() - started, 1), activity="done"))
    return {**outputs, "cut": made.cut}


def _excerpt(source: Any, opening: Frame, video: VideoSettings) -> tuple[list[Clip], float]:
    """The source's stretch a shot re-creates, read at exactly the frames and size the encoder keeps:
    `pipeline.fitted_clip` is its rule, so nothing is read to be shrunk again."""
    if not isinstance(source, dict) or not source.get("video"):
        return [], 1.0
    path = _path(source["video"])
    start, seconds, scale = float(source.get("start", 0.0)), float(source["seconds"]), float(source.get("scale", 1.0))
    _, fps = video_length(path)
    canvas = render_chunk.plan(opening, video)
    wanted = max(1, round(min(seconds, video.duration) * fps))
    count, width, height = pipeline.fitted_clip(wanted, canvas, scale)
    excerpt = media.clip_from_video(path, limit=count, with_audio=False, start=start, size=(width, height))
    return [excerpt], scale


def run_assemble(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    chunks = [_path(p) for p in inputs["chunks"]]
    audios = [_path(p) for p in inputs["audios"]]
    cuts = [int(cut) for cut in inputs["cuts"]]
    crossfade = float(inputs.get("crossfade") or 0.0)
    at = assemble.assemble_key([str(k) for k in inputs["keys"]], cuts, crossfade)
    with store.connect() as connection:
        existing = store.artifact_by_key(connection, str(at))
    if existing is not None:
        ui.note(f"already joined as {at}")
        events.emit(protocol.Final(video=existing.path))
        return Skipped({"video": existing.path})
    destination = video_io.output_path(config.VIDEO_DIR, str(inputs.get("name") or "chapter"))
    frames, seconds = assemble.assemble(list(zip(chunks, audios, strict=True)), cuts, crossfade, destination)
    record.artifact_made(destination, store.VIDEO, key=str(at))
    events.emit(protocol.Final(video=store.relative(destination), frames=frames, seconds=seconds))
    return {"video": store.relative(destination), "frames": frames, "seconds": seconds}


def run_compose_keyframe(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production_id = int(inputs["production"])
    with store.connect() as connection:
        cards = {m.slug: compiler.card_of(m) for m in store.members_of(connection, production_id)}
    chosen = EditSettings(**settings).with_(seed=int(inputs["seed"]))
    size = (int(inputs["size"][0]), int(inputs["size"][1])) if inputs.get("size") else None
    keep = str(inputs["keep"]) if inputs.get("keep") else None
    base = _frame(inputs["base"])
    if inputs.get("box") and size is not None:
        base = compose_keyframe.around(base, [float(v) for v in inputs["box"]], size[0] / size[1])
    arrangement = str(inputs["arrangement"])
    made = key(
        "compose_keyframe", 1, base=base.pixels, arrangement=arrangement,
        cards={slug: dataclasses.asdict(card) for slug, card in cards.items()}, settings=chosen,
        **({"size": size, "keep": keep} if size or keep else {}),
    )  # fmt: skip
    target = cache.path(made, ".png")
    if target.is_file():
        ui.note(f"keyframe already composed as {made}")
        return Skipped({"frame": store.relative(target), "key": str(made)})
    frame = compose_keyframe.compose(base, arrangement, cards, chosen, keep=keep, size=size)
    frame.pil().save(target)
    record.artifact_made(target, store.IMAGE, chosen.seed, key=str(made))
    return {"frame": store.relative(target), "key": str(made)}


def run_draw_reference(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    with store.connect() as connection:
        member = store.member(connection, int(inputs["member"]))
    if member is None:
        raise LookupError(f"member {inputs['member']} was deleted before it was drawn")
    chosen = EditSettings(**settings).with_(seed=int(inputs["seed"]))
    size = [int(v) for v in inputs["size"]]
    made = drawn.wanted(member, chosen, size)
    target = cache.path(made, ".png")
    if target.is_file():
        ui.note(f"{member.slug} already drawn as {made}")
        return Skipped({"frame": store.relative(target), "key": str(made)})
    frame = draw_reference.draw(compiler.card_of(member), chosen, (size[0], size[1]))
    frame.pil().save(target)
    record.artifact_made(target, store.IMAGE, chosen.seed, key=str(made))
    return {"frame": store.relative(target), "key": str(made)}


def run_judge(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    chosen = JudgeSettings(**settings)
    production_id, kind = int(inputs["production"]), str(inputs["kind"])
    cast = [str(m) for m in inputs.get("cast", [])]
    with store.connect() as connection:
        cards = [compiler.card_of(m) for m in store.members_of(connection, production_id) if m.slug in cast]
    if kind == "clip":
        frames = judge.sampled(media.clip_from_video(_path(inputs["path"]), with_audio=False))
    else:
        frames = [_frame(inputs["path"])]
    verdict = judge.judge(
        frames, kind, str(inputs.get("purpose", "")), str(inputs.get("arrangement", "")), cards, chosen
    )
    ui.panel(
        f"judge · {kind}",
        {
            "score": f"{verdict.score:.2f} against a gate of {chosen.gate:.2f}",
            **{member: f"{likeness:.2f}" for member, likeness in verdict.cast.items()},
            "issues": "; ".join(verdict.issues) or "-",
            "notes": verdict.notes or "-",
        },
        color="green" if verdict.score >= chosen.gate else "yellow",
    )
    return {**verdict.data(), "gate": chosen.gate, "retries": chosen.retries, "passed": verdict.score >= chosen.gate}


def _write_chunk(clip: Clip, video: Path, soundtrack: Path) -> None:
    """Near-lossless, since the chapter is joined from these; the audio beside it, since the mp4's is not."""
    height, width = int(clip.frames.shape[1]), int(clip.frames.shape[2])
    writer = video_io.VideoWriter(video, width, height, clip.fps, clip.sample_rate, crf=16)
    writer.write(clip.frames)
    writer.close(clip.audio)
    samples = int(np.ceil(clip.sample_rate / clip.fps * len(clip)))
    audio = clip.audio[:, :samples].clone() if clip.audio is not None else torch.zeros(2, samples)
    torch.save(audio, soundtrack)


def _thumbnail(frame: Frame, name: str) -> str:
    run = events.active()
    if run is None:
        return ""
    image = frame.pil()
    image.thumbnail((360, 360))
    path = run.path(name)
    image.save(path, quality=85)
    return run.rel(path)


RUNNERS: dict[str, Runner] = {
    "write_instruction": run_write_instruction,
    "write_video_prompt": run_write_video_prompt,
    "edit_image": run_edit_image,
    "render_clip": run_render_clip,
    "enlarge_pixels": run_enlarge_pixels,
    "grade": run_grade,
    "retime": run_retime,
    "render_shot": run_render_shot,
    "assemble": run_assemble,
    "compose_keyframe": run_compose_keyframe,
    "draw_reference": run_draw_reference,
    "judge": run_judge,
    **PLAN_RUNNERS,
}
