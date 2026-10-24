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
from PIL import Image

from difflab import config, store
from difflab.core import audio as audio_
from difflab.core import context, events, media, models, protocol, record, ui, video_io
from difflab.core.cache import cache
from difflab.core.hub import HubFile
from difflab.core.imaging import megapixel_size, video_length
from difflab.core.keys import Key, key
from difflab.core.settings import (
    AssembleSettings,
    ChatSettings,
    Continuity,
    EditSettings,
    EnlargePixelsSettings,
    JudgeSettings,
    RestoreSettings,
    RetimeSettings,
    VlmSettings,
)
from difflab.core.types import Clip, Frame
from difflab.models.ltx25 import config as ltx_config
from difflab.models.ltx25 import pipeline as ltx_pipeline
from difflab.models.ltx25.settings import LtxSettings
from difflab.models.minimax_h3 import config as h3_config
from difflab.models.minimax_h3 import pipeline, vae
from difflab.models.minimax_h3 import prompts as video_prompts
from difflab.models.minimax_h3.settings import VideoPromptSettings, VideoSettings
from difflab.screenplay import compiler, drawn, made, words
from difflab.screenplay import production as production_module
from difflab.screenplay.looks import Form
from difflab.tools import (
    assemble,
    chat,
    compose_keyframe,
    cut_crops,
    direct,
    draw_reference,
    edit_image,
    enlarge_pixels,
    grade,
    handoff,
    judge,
    master,
    previews,
    render_chunk,
    render_ltx,
    restore,
    retime,
    write_prompt,
)
from difflab.tools import cards as cast_cards
from difflab.work import Skipped
from difflab.work.plan_runners import PLAN_RUNNERS

Runner = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else config.OUTPUT_DIR / path


def _frame(value: Any) -> Frame:
    return media.frame_from_image(_path(value))


def _ratio(aspect: Any) -> float:
    w, h = (int(part) for part in str(aspect).split(":"))
    return w / h


def _opening(inputs: dict[str, Any]) -> Frame:
    photo = _frame(inputs["photo"])
    return photo.crop_to_aspect(_ratio(inputs["aspect"])) if inputs.get("aspect") else photo


def _references(inputs: dict[str, Any]) -> list[Frame]:
    return [_frame(p) for p in inputs.get("references", [])]


def run_write_instruction(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    text = write_prompt.write_edit_instruction(
        _opening(inputs),
        str(inputs["idea"]),
        VlmSettings(**settings),
        str(inputs.get("engine", "qwen-image-edit")),
        _references(inputs),
    )
    return {"text": text}


def run_write_image_prompt(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    text, aspect = write_prompt.write_image_prompt(str(inputs["idea"]), VlmSettings(**settings), _references(inputs))
    return {"text": text, "aspect": aspect}


def run_chat(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    return chat.reply(int(inputs["message"]), ChatSettings(**settings))


def run_create_image(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    chosen = EditSettings(**settings).with_(engine="qwen-image-2.1")
    made = edit_image.create(str(inputs["prompt"]), str(inputs["aspect"]), chosen, _references(inputs))
    png = media.save(made, chosen.model_stem, chosen.seed, announce=False)
    return {"image": store.relative(png)}


def run_write_video_prompt(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    if inputs.get("photo") is None:
        frames = _references(inputs)
        return {
            "text": write_prompt.write_reference_caption(frames, str(inputs["idea"]), VideoPromptSettings(**settings))
        }
    system = video_prompts.VLM_SYSTEM_PROMPT_EDIT if inputs.get("still") else video_prompts.VLM_SYSTEM_PROMPT
    text = write_prompt.write_video_prompt(
        _opening(inputs), str(inputs["idea"]), VideoPromptSettings(**settings), system
    )
    return {"text": text}


def run_edit_image(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    chosen = EditSettings(**settings)
    made = edit_image.edit(_opening(inputs), str(inputs["instruction"]), chosen, donors=_references(inputs))
    png = media.save(made, chosen.model_stem, chosen.seed, announce=False)
    return {"image": store.relative(png)}


def run_render_clip(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    chosen = VideoSettings(**settings)
    references = _references(inputs)
    opening = _opening(inputs) if inputs.get("photo") is not None else None
    clip = render_chunk.render(opening, str(inputs["prompt"]), chosen, references=references)
    stem = HubFile.parse(chosen.checkpoint).path.stem
    path = media.save(clip, stem, chosen.seed, announce=False)
    if not inputs.get("still"):
        return {"video": store.relative(path)}
    png = media.save(clip.last_frame(), stem, chosen.seed, announce=False)
    return {"image": store.relative(png), "video": store.relative(path)}


def run_write_ltx_prompt(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    text = write_prompt.write_ltx_prompt(
        _opening(inputs), str(inputs["idea"]), _references(inputs), VideoPromptSettings(**settings)
    )
    return {"text": text}


def run_render_ltx(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    chosen = LtxSettings(**settings)
    clip = render_ltx.render(_opening(inputs), str(inputs["prompt"]), chosen, references=_references(inputs))
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
    if settings.get("engine") == ltx_config.ENGINE:
        return _render_shot_ltx(inputs, settings)
    continuity, _ = Continuity.resolve(settings["continuity"])
    carry = _carry(inputs.get("carry"), continuity.tail_frames, float(inputs.get("carry_late") or 0.0))
    # The carried steps are redrawn under the shot before's last frames and cut away at the join, so the
    # shot runs that much longer to keep its own seconds.
    seconds = float(inputs["seconds"]) + (continuity.tail_frames / h3_config.FPS if carry else 0.0)
    video = VideoSettings(**settings["video"]).with_(seed=int(inputs["seed"]), duration=seconds)
    carried_on = continuity.tail_frames if inputs.get("carried_on") else 0
    prompt = str(inputs["prompt"])
    # A shot opened from its references alone has no first frame: the canvas comes from the production's,
    # and there is no picture to hold its colour to.
    opening = _frame(inputs["opening"]) if inputs.get("opening") else None
    shape = opening if opening is not None else _canvas(inputs.get("canvas"))
    closing = _frame(inputs["closing"]) if inputs.get("closing") else None
    tail = handoff.load_tail(_path(inputs["tail"])) if inputs.get("tail") else None
    strength = continuity.color_anchor
    anchor = _frame(inputs["anchor"]) if inputs.get("anchor") and strength else None
    references = [_frame(p) for p in inputs.get("references", [])]
    position = int(inputs.get("position", 0))
    clips, scale = _excerpt(inputs.get("source"), shape, video)

    chunk = render_chunk.chunk_key(opening, prompt, video, references, tail, clips, scale, closing, shape, carry)
    # A shot that ends on the next one's keyframe is cut on its very last frame, the one pinned to it.
    window = 1 if closing is not None else continuity.handoff_window
    hold = continuity.color_hold if opening is not None else 0.0
    at = handoff.handoff_key(chunk, window, continuity.tail_frames, anchor, strength, hold, carried_on)
    finished = key("held", 1, chunk=str(chunk), hold=hold) if hold else chunk
    latents_file, mp4, soundtrack = (
        cache.path(chunk, ".pt"),
        cache.path(finished, ".mp4"),
        cache.path(chunk, ".audio.pt"),
    )
    frame_file, tail_file, _ = handoff.files(at)
    outputs: dict[str, Any] = {
        "key": str(chunk),
        "prompt": prompt,
        "latents": store.relative(latents_file),
        "chunk": store.relative(mp4),
        "audio": store.relative(soundtrack),
        "frame": store.relative(frame_file),
        "tail": store.relative(tail_file) if continuity.tail_frames and not carried_on else None,
        "audio_late": carry.audio_late if carry is not None else 0.0,
    }
    events.emit(
        protocol.ItemInfo(
            index=position,
            version=1,
            prompt=prompt,
            seed=video.seed,
            opening=_thumbnail(opening, f"thumbs/shot{position + 1}_opening.jpg") if opening is not None else "",
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
            canvas_from=shape,
            closing=closing,
            carry=carry,
        )
    finally:
        previews.target(None, "clip")
    clip = render_chunk.decode(latents)
    if hold and opening is not None:
        with ui.step("Holding colour"):
            clip = media.hold_colour(clip, opening, hold)
    with ui.step("Handoff") as detail:
        made = handoff.handoff(clip, window, continuity.tail_frames, anchor, strength, carried_on)
        handoff.save(made, at)
        carried = f" · {len(made.tail)} frames of tail" if made.tail is not None else ""
        detail(f"frame {made.cut + 1} of {len(clip)}{carried}")
    if not (mp4.is_file() and soundtrack.is_file()):
        with ui.step("Writing chunk") as detail:
            _write_chunk(clip, mp4, soundtrack, carry.audio_late / h3_config.AUDIO_LATENT_FPS if carry else 0.0)
            detail(f"{len(clip)} frames · {clip.duration:.2f}s")
    record.artifact_made(mp4, store.VIDEO, video.seed, key=str(chunk))
    record.artifact_made(frame_file, store.IMAGE, key=str(at))
    events.emit(shown)
    events.emit(protocol.ItemEnd(index=position, seconds=round(time.perf_counter() - started, 1), activity="done"))
    return {**outputs, "cut": made.cut}


def _render_shot_ltx(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """`run_render_shot` on LTX-2.5: no tail or latents cross the join, the shot before's frame alone opens it."""
    continuity, _ = Continuity.resolve(settings["continuity"])
    chosen = LtxSettings(**settings["ltx"]).with_(seed=int(inputs["seed"]), duration=float(inputs["seconds"]))
    prompt = str(inputs["prompt"])
    opening = _frame(inputs["opening"]) if inputs.get("opening") else None
    closing = _frame(inputs["closing"]) if inputs.get("closing") else None
    references = [_frame(p) for p in inputs.get("references", [])]
    size = (int(inputs["canvas"][0]), int(inputs["canvas"][1])) if inputs.get("canvas") else None
    if size is None and opening is not None:
        size = (opening.width, opening.height)
    strength = continuity.color_anchor
    anchor = _frame(inputs["anchor"]) if inputs.get("anchor") and strength else None
    position = int(inputs.get("position", 0))
    chunk = key(
        "render_ltx_shot", 1, prompt=prompt, settings=chosen, size=list(size) if size else None,
        opening=opening.pixels if opening is not None else None,
        closing=closing.pixels if closing is not None else None,
        references=[frame.pixels for frame in references],
    )  # fmt: skip
    window = 1 if closing is not None else continuity.handoff_window
    hold = continuity.color_hold if opening is not None else 0.0
    at = handoff.handoff_key(chunk, window, 0, anchor, strength, hold, 0)
    finished = key("held", 1, chunk=str(chunk), hold=hold) if hold else chunk
    latents_file, mp4, soundtrack = (
        cache.path(chunk, ".pt"),
        cache.path(finished, ".mp4"),
        cache.path(chunk, ".audio.pt"),
    )
    frame_file, _, _ = handoff.files(at)
    outputs: dict[str, Any] = {
        "key": str(chunk),
        "prompt": prompt,
        "latents": store.relative(latents_file),
        "chunk": store.relative(mp4),
        "audio": store.relative(soundtrack),
        "frame": store.relative(frame_file),
        "tail": None,
        "audio_late": 0.0,
    }
    events.emit(
        protocol.ItemInfo(
            index=position,
            version=1,
            prompt=prompt,
            seed=chosen.seed,
            opening=_thumbnail(opening, f"thumbs/shot{position + 1}_opening.jpg") if opening is not None else "",
            references=[_thumbnail(ref, f"thumbs/shot{position + 1}_ref{n}.jpg") for n, ref in enumerate(references)],
        )
    )
    shown = protocol.ItemVideo(index=position, version=1, video=f"/output/{outputs['chunk']}", seed=chosen.seed)
    if latents_file.is_file() and mp4.is_file() and soundtrack.is_file() and handoff.present(at):
        ui.note(f"shot {position + 1}: already made as {chunk}")
        events.emit(shown)
        return Skipped({**outputs, "cut": handoff.cut_of(at)})

    started = time.perf_counter()
    events.emit(protocol.ItemBegin(index=position))
    previews.target(position, f"shot{position + 1}")
    try:
        if latents_file.is_file():
            sampled = torch.load(latents_file, map_location="cpu", weights_only=True)
            video_latents, audio_latents = sampled["video"], sampled["audio"]
        else:
            video_latents, audio_latents = render_ltx.latents(opening, prompt, chosen, references, closing, size)
            torch.save({"video": video_latents.cpu(), "audio": audio_latents.cpu()}, latents_file)
    finally:
        previews.target(None, "clip")
    clip = render_ltx.decode(video_latents, audio_latents, chosen, prompt)
    # The rest of a production joins, masters and finishes H3's 32 kHz.
    assert clip.audio is not None
    clip = dataclasses.replace(
        clip, audio=audio_.resampled(clip.audio, clip.sample_rate), sample_rate=audio_.SAMPLE_RATE
    )
    if hold and opening is not None:
        with ui.step("Holding colour"):
            clip = media.hold_colour(clip, opening, hold)
    with ui.step("Handoff") as detail:
        made = handoff.handoff(clip, window, 0, anchor, strength, 0)
        handoff.save(made, at)
        detail(f"frame {made.cut + 1} of {len(clip)}")
    if not (mp4.is_file() and soundtrack.is_file()):
        with ui.step("Writing chunk") as detail:
            _write_chunk(clip, mp4, soundtrack)
            detail(f"{len(clip)} frames · {clip.duration:.2f}s")
    record.artifact_made(mp4, store.VIDEO, chosen.seed, key=str(chunk))
    record.artifact_made(frame_file, store.IMAGE, key=str(at))
    events.emit(shown)
    events.emit(protocol.ItemEnd(index=position, seconds=round(time.perf_counter() - started, 1), activity="done"))
    return {**outputs, "cut": made.cut}


def _carry(latents: Any, frames: int, late: float) -> pipeline.Carry | None:
    """The shot before's last `frames` and second of sound, from its cached latents; `late` is how far
    behind them its soundtrack runs."""
    if not latents or not frames:
        return None
    sampled = torch.load(_path(latents), map_location="cpu", weights_only=True)
    return pipeline.carry_of(sampled["video"], sampled["audio"], frames, h3_config.AUDIO_LATENT_FPS, late)


def _canvas(size: Any) -> Frame:
    """A blank of the production's canvas: only ever measured, for the shape of a shot with no first frame."""
    if not size:
        raise ValueError("a shot opened from its references alone needs the production's canvas: plan it first")
    return Frame.from_pil(Image.new("RGB", (int(size[0]), int(size[1]))), origin="canvas")


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
    joins = assemble.joins_of(inputs)
    at = assemble.assemble_key([str(k) for k in inputs["keys"]], joins)
    with store.connect() as connection:
        existing = store.artifact_by_key(connection, str(at))
    if existing is not None:
        ui.note(f"already joined as {at}")
        events.emit(protocol.Final(video=existing.path))
        return Skipped({"video": existing.path})
    destination = video_io.output_path(config.VIDEO_DIR, str(inputs.get("name") or "chapter"))
    frames, seconds = assemble.assemble(list(zip(chunks, audios, strict=True)), joins, destination)
    record.artifact_made(destination, store.VIDEO, key=str(at))
    events.emit(protocol.Final(video=store.relative(destination), frames=frames, seconds=seconds))
    return {"video": store.relative(destination), "frames": frames, "seconds": seconds}


def run_enlarge_shot(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """A rendered shot again at the finish's size: its own latents decoded, held to the colour its chunk
    was held to, so the 4K chapter joins and grades exactly as the native one did, then restored."""
    video = VideoSettings(**settings["video"])
    continuity, _ = Continuity.resolve(settings["continuity"])
    restoring, _ = RestoreSettings.resolve(settings["enlarge"])
    hold = continuity.color_hold if inputs.get("opening") else 0.0
    megapixels = float(inputs["megapixels"])
    at = key("enlarged", 3, chunk=str(inputs["key"]), settings=restoring, megapixels=megapixels, hold=hold)
    mp4 = cache.path(at, ".mp4")
    outputs = {"chunk": store.relative(mp4), "key": str(at)}
    if mp4.is_file():
        ui.note(f"already enlarged as {at}")
        return Skipped(outputs)
    sampled = torch.load(_path(inputs["latents"]), map_location="cpu", weights_only=True)["video"]
    with ui.step("Decoding the shot") as detail:
        if settings.get("engine") == ltx_config.ENGINE:
            ltx_vae = models.registry.acquire(render_ltx.video_vae_spec(LtxSettings(**settings["ltx"])))
            frames = ltx_pipeline.decode_video(ltx_vae, sampled)
        else:
            frames = vae.decode_video(models.registry.acquire(render_chunk.video_vae_spec(video)), sampled)
        detail(f"{len(frames)} frames · {frames.shape[2]}×{frames.shape[1]}")
    if hold:
        frames = media.held(frames, media.colour_hold(frames, h3_config.FPS, _frame(inputs["opening"]), hold))
    size = restore.finish_size(int(frames.shape[2]), int(frames.shape[1]), megapixels)
    partial = mp4.with_name(f"{mp4.stem}.partial.mp4")
    written = restore.finish(frames, partial, size, h3_config.FPS, restoring)
    partial.replace(mp4)
    record.artifact_made(mp4, store.VIDEO, key=str(at))
    return {**outputs, "size": list(size), "frames": written}


def run_finish(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """The enlarged chunks joined at the native join's cuts and crossfade, then retimed by RIFE."""
    fps = int(inputs["fps"])
    retiming, _ = RetimeSettings.resolve(settings.get("retime"))
    joins = assemble.joins_of(inputs)
    joined_key = assemble.assemble_key([str(k) for k in inputs["keys"]], joins)
    at = key("finish", 1, joined=str(joined_key), fps=fps, settings=retiming)
    with store.connect() as connection:
        existing = store.artifact_by_key(connection, str(at))
    if existing is not None:
        ui.note(f"already finished as {at}")
        events.emit(protocol.Final(video=existing.path))
        return Skipped({"video": existing.path})
    joined = cache.path(joined_key, ".mp4")
    parts = list(zip([_path(p) for p in inputs["chunks"]], [_path(p) for p in inputs["audios"]], strict=True))
    assemble.assemble(parts, joins, joined)
    destination = video_io.output_path(config.VIDEO_DIR, f"{inputs.get('name') or 'chapter'}_{fps}fps")
    frames, seconds = retime.retime_file(joined, destination, fps, retiming)
    # About a gigabyte a minute at 4K, and nothing reads it once the retimed file exists.
    joined.unlink(missing_ok=True)
    record.artifact_made(destination, store.VIDEO, key=str(at))
    events.emit(protocol.Final(video=store.relative(destination), frames=frames, seconds=seconds))
    return {"video": store.relative(destination), "frames": frames, "seconds": seconds}


def run_master(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """A joined chapter levelled and subtitled; its cues are the dialogue at where each shot plays in it."""
    chosen, _ = AssembleSettings.resolve(settings)
    chunks = [_path(p) for p in inputs["chunks"]]
    joins = assemble.joins_of(inputs)
    starts = media.starts([media.frames_of(chunk) for chunk in chunks], joins, media.rate_of(chunks[0]))
    cues = [
        master.Cue(round(at + float(start), 3), round(at + float(end), 3), str(text))
        for at, lines in zip(starts, inputs["lines"], strict=True)
        for start, end, text in lines
    ]
    subtitled = chosen.subtitles != "off"
    made = key("master", 1, video=str(inputs["video"]), lufs=chosen.lufs, subtitles=chosen.subtitles, cues=cues)
    reused = _reused("master", made)
    if reused is not None:
        return reused
    destination = video_io.output_path(config.VIDEO_DIR, str(inputs.get("name") or "chapter"))
    with ui.step("Levelling the chapter") as detail:
        lufs = master.master(
            _path(inputs["video"]), destination, chosen.lufs, cues, chosen.subtitles == "separate track"
        )
        detail(f"{lufs:.1f} LUFS" if lufs is not None else "silent")
    srt, vtt = _sidecars(destination, cues) if subtitled else (None, None)
    seconds = master.seconds_of(destination)
    record.artifact_made(destination, store.VIDEO, key=str(made))
    events.emit(protocol.Final(video=store.relative(destination), seconds=seconds))
    return {
        "video": store.relative(destination), "srt": srt, "vtt": vtt, "seconds": seconds, "lufs": lufs,
        "cues": [list(cue) for cue in cues], "finished": bool(inputs.get("finished")),
    }  # fmt: skip


def run_film(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """The mastered chapters end to end, their streams copied: nothing is re-encoded but the subtitles."""
    chosen, _ = AssembleSettings.resolve(settings)
    chapters = [
        master.Chapter(_path(video), str(title), [master.Cue(*cue) for cue in cues])
        for video, title, cues in zip(inputs["videos"], inputs["titles"], inputs["cues"], strict=True)
    ]
    made = key(
        "film", 1, chapters=[[str(c.video), c.title, c.cues] for c in chapters], markers=chosen.markers,
        subtitles=chosen.subtitles, title=str(inputs["title"]),
    )  # fmt: skip
    reused = _reused("film", made)
    if reused is not None:
        return reused
    destination = video_io.output_path(config.VIDEO_DIR, f"{inputs.get('name') or 'production'}_film")
    with ui.step("Joining the chapters") as detail:
        cues = master.film(
            chapters, destination, str(inputs["title"]), chosen.markers, chosen.subtitles == "separate track"
        )
        detail(f"{len(chapters)} chapters · {len(cues)} lines")
    srt, vtt = _sidecars(destination, cues) if chosen.subtitles != "off" else (None, None)
    seconds = master.seconds_of(destination)
    record.artifact_made(destination, store.VIDEO, key=str(made))
    events.emit(protocol.Final(video=store.relative(destination), seconds=seconds))
    return {
        "video": store.relative(destination), "srt": srt, "vtt": vtt, "seconds": seconds,
        "chapters": len(chapters), "lines": len(cues),
    }  # fmt: skip


def _reused(tool: str, made: Key) -> dict[str, Any] | None:
    """An earlier job's outputs under the same key, while its video is still on disk."""
    ctx = context.current()
    with store.connect() as connection:
        if ctx.job_id is not None:
            store.set_job_key(connection, ctx.job_id, str(made))
        previous = store.last_done_job(connection, tool, str(made))
    if previous is None or previous.id == ctx.job_id or not _path(previous.outputs["video"]).is_file():
        return None
    ui.note(f"{tool}: already made as {made}")
    events.emit(protocol.Final(video=str(previous.outputs["video"])))
    return Skipped(previous.outputs)


def _sidecars(video: Path, cues: list[master.Cue]) -> tuple[str, str]:
    """The subtitles beside the video as SRT, and as WebVTT, which is what a page's <track> reads."""
    srt, vtt = video.with_suffix(".srt"), video.with_suffix(".vtt")
    srt.write_text(master.srt(cues))
    vtt.write_text(master.vtt(cues))
    return store.relative(srt), store.relative(vtt)


def run_compose_keyframe(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production_id = int(inputs["production"])
    with store.connect() as connection:
        cards = {m.slug: compiler.card_of(m) for m in store.members_of(connection, production_id)}
    cards |= {slug: compiler.formed(cards[slug], form) for slug, form in _forms(inputs).items() if slug in cards}
    chosen = EditSettings(**settings).with_(seed=int(inputs["seed"]))
    size = (int(inputs["size"][0]), int(inputs["size"][1])) if inputs.get("size") else None
    keep = str(inputs["keep"]) if inputs.get("keep") else None
    base = _frame(inputs["base"])
    if inputs.get("box") and size is not None:
        base = compose_keyframe.around(base, [float(v) for v in inputs["box"]], size[0] / size[1])
    arrangement = str(inputs["arrangement"])
    pictures = {str(slug): _frame(path) for slug, path in (inputs.get("pictures") or {}).items()}
    note = str(inputs.get("note") or "")
    looks = {str(m): str(v) for m, v in (inputs.get("looks") or {}).items() if v}
    style = _frame(inputs["style"]) if inputs.get("style") else None
    look = str(inputs.get("look") or "")
    anchor = _frame(inputs["anchor"]) if inputs.get("anchor") else None
    strength = float(inputs.get("anchor_strength") or 0.0) if anchor is not None else 0.0
    made = key(
        "compose_keyframe", 2, base=base.pixels, arrangement=arrangement,
        cards={slug: dataclasses.asdict(card) for slug, card in cards.items()}, settings=chosen,
        **({"size": size, "keep": keep} if size or keep else {}),
        **({"pictures": {slug: frame.pixels for slug, frame in pictures.items()}} if pictures else {}),
        **({"note": note} if note else {}),
        **({"looks": looks} if looks else {}),
        **({"style": style.pixels} if style is not None else {}), **({"look": look} if look else {}),
        **({"anchor": anchor.pixels, "strength": strength} if anchor is not None and strength else {}),
    )  # fmt: skip
    target = cache.path(made, ".png")
    if target.is_file():
        ui.note(f"keyframe already composed as {made}")
        return Skipped({"frame": store.relative(target), "key": str(made)})
    frame = compose_keyframe.compose(
        base, arrangement, cards, chosen, keep=keep, size=size, pictures=pictures, note=note, looks=looks,
        style=style, look=look,
    )  # fmt: skip
    if anchor is not None and strength:
        frame = media.recolored(frame, media.color_gain(frame, anchor, strength))
    frame.pil().save(target)
    record.artifact_made(target, store.IMAGE, chosen.seed, key=str(made))
    return {"frame": store.relative(target), "key": str(made)}


def run_draw_look(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """A look: its source picture changed by the amendment alone, a person's base face handed over beside it."""
    chosen = EditSettings(**settings).with_(seed=int(inputs["seed"]))
    source = _frame(inputs["source"])
    face = _frame(inputs["face"]) if inputs.get("face") else None
    size = (int(inputs["size"][0]), int(inputs["size"][1]))
    kind, amend = str(inputs["kind"]), str(inputs["amend"])
    if inputs.get("form"):
        said = words.COMPOSE_FORM.format(was=inputs["was"], form=inputs["form"], look=amend.rstrip("."))
    else:
        said = words.COMPOSE_LOOK.format(
            noun=inputs["noun"], kept=words.COMPOSE_LOOK_KEEP[kind], amendment=amend.rstrip(".")
        )
    if str(inputs.get("allowed") or "").strip():
        said = f"{said} {words.COMPOSE_LOOK_ALLOWED.format(allowed=str(inputs['allowed']).strip().rstrip('.'))}"
    if face is not None:
        said = f"{said} {words.COMPOSE_LOOK_FACE}"
    if inputs.get("fixes"):
        said = f"{said} {words.COMPOSE_FIXES.format(fixes='; '.join(str(f) for f in inputs['fixes']))}"
    garment = _frame(inputs["garment"]) if inputs.get("garment") else None
    if garment is not None:
        picture = 3 if face is not None else 2
        said = f"{said} {words.COMPOSE_LOOK_GARMENT.format(name=inputs['garment_name'], picture=picture)}"
    # By every picture's pixels: a source drawn again moves every look drawn from it.
    made = key(
        "draw_look", 1, source=source.pixels, amend=amend, said=said, settings=chosen, size=list(size),
        face=face.pixels if face is not None else None,
        **({"garment": garment.pixels} if garment is not None else {}),
    )  # fmt: skip
    target = cache.path(made, ".png")
    outputs = {"frame": store.relative(target), "key": str(made), "size": list(size)}
    if target.is_file():
        ui.note(f"look already drawn as {made}")
        return Skipped(outputs)
    width, height = megapixel_size(*size, chosen.megapixels)
    work = (max(16, width // 16 * 16), max(16, height // 16 * 16))  # the editor floors its output to /16
    donors = [p for p in (face, garment) if p is not None]
    edited = edit_image.edit(source, said, chosen, size=work, donors=donors)
    edited.resized(*size).pil().save(target)
    record.artifact_made(target, store.IMAGE, chosen.seed, key=str(made))
    return outputs


def run_plate_ref(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """A member's reference photo with everything behind it replaced by a flat grey backdrop."""
    chosen = EditSettings(**settings).with_(seed=int(inputs["seed"]))
    picture = _frame(inputs["picture"])
    said = words.COMPOSE_PLATE.format(noun=inputs["noun"], traits=inputs["traits"])
    made = key("plate_ref", 1, picture=picture.pixels, said=said, settings=chosen)
    target = cache.path(made, ".png")
    outputs = {"frame": store.relative(target), "key": str(made)}
    if target.is_file():
        ui.note(f"already plated as {made}")
        return Skipped(outputs)
    edit_image.plate(picture, said, chosen).pil().save(target)
    record.artifact_made(target, store.IMAGE, chosen.seed, key=str(made))
    return outputs


def run_extend_opening(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    if not inputs.get("extend"):
        return Skipped({"opening": str(inputs["opening"])})
    chosen = EditSettings(**settings)
    padded = _frame(inputs["opening"])
    sides = "left and right" if float(inputs["cut"][0]) < 0 else "top and bottom"
    said = words.COMPOSE_EXTEND_OPENING.format(sides=sides)
    made = key("extend_opening", 1, padded=padded.pixels, said=said, settings=chosen)
    target = cache.path(made, ".png")
    outputs = {"opening": store.relative(target)}
    if target.is_file():
        ui.note(f"already extended as {made}")
        return Skipped(outputs)
    edited = edit_image.edit(padded, said, chosen, size=edit_image.vae_reference_size(padded))
    edited.resized(padded.width, padded.height).pil().save(target)
    record.artifact_made(target, store.IMAGE, chosen.seed, key=str(made))
    return outputs


def run_extend_photo(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """A quick job's photo at `aspect`, drawn out as a production's opening is; close to it, cropped instead."""
    chosen = EditSettings(**settings)
    photo = _frame(inputs["photo"])
    ratio = _ratio(inputs["aspect"])
    if not cut_crops.extends(photo, ratio):
        made = key("extend_photo", 1, photo=photo.pixels, ratio=ratio)
        target = cache.path(made, ".png")
        if not target.is_file():
            photo.crop_to_aspect(ratio).pil().save(target)
        return {"image": store.relative(target)}
    box = cut_crops.around(photo, ratio)
    padded = cut_crops.padded(photo, round(box.right - box.left), round(box.bottom - box.top))
    said = words.COMPOSE_EXTEND_OPENING.format(sides="left and right" if box.left < 0 else "top and bottom")
    made = key("extend_photo", 1, padded=padded.pixels, said=said, settings=chosen)
    target = cache.path(made, ".png")
    outputs = {"image": store.relative(target)}
    if target.is_file():
        ui.note(f"already extended as {made}")
        return Skipped(outputs)
    edited = edit_image.edit(padded, said, chosen, size=edit_image.vae_reference_size(padded))
    edited.resized(padded.width, padded.height).pil().save(target)
    record.artifact_made(target, store.IMAGE, chosen.seed, key=str(made))
    return outputs


def run_draw_reference(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    with store.connect() as connection:
        member = store.member(connection, int(inputs["member"]))
    if member is None:
        raise LookupError(f"member {inputs['member']} was deleted before it was drawn")
    member = _place_from_story(member, forced=bool(inputs.get("describe")))
    chosen = EditSettings(**settings).with_(seed=int(inputs["seed"]))
    size = [int(v) for v in inputs["size"]]
    # Keyed without the judge's fixes, so a reference redrawn from them stays current: the picture under
    # this seed is the fixed one, and asking for the seed again finds it.
    made = drawn.wanted(member, chosen, size)
    target = cache.path(made, ".png")
    if target.is_file():
        ui.note(f"{member.slug} already drawn as {made}")
        return Skipped({"frame": store.relative(target), "key": str(made)})
    fixes = [str(fix) for fix in inputs.get("fixes") or []]
    frame = draw_reference.draw(compiler.card_of(member), chosen, (size[0], size[1]), fixes)
    frame.pil().save(target)
    record.artifact_made(target, store.IMAGE, chosen.seed, key=str(made))
    return {"frame": store.relative(target), "key": str(made)}


def _place_from_story(member: store.Member, forced: bool = False) -> store.Member:
    """A place no photo of its own shows, its look written again from the story before it is drawn.

    The look was tidied from a photo the place turned out not to be in: production 13's courtyard
    (2026-10-04) was left one trait, "mountain sect courtyard", and the editor drew a Tibetan
    monastery where the story asked for dark wood eaves, red banners and cherry blossom. A place
    edited by hand keeps its words unless `forced`, which also writes them for a place a photo shows."""
    with store.connect() as connection:
        production = store.production(connection, member.production_id)
        own = [r for r in store.refs_of(connection, member.id) if r.artifact and not r.generated]
        members = store.members_of(connection, member.production_id)
    if member.kind != "environment" or production is None or not production.story.strip():
        return member
    if not forced and (member.pinned or own):
        return member

    def asked(m: store.Member) -> direct.Member:
        return direct.Member(m.slug, m.kind, m.noun, m.possessive, m.find)

    card = cast_cards.invent(production.story, asked(member), {m.slug: asked(m) for m in members})
    identity = {
        **member.identity,
        "traits": [dataclasses.asdict(trait) for trait in card.traits],
        "lights": list(card.lights),
    }
    if identity == member.identity:
        return member
    with store.connect() as connection, store.transaction(connection):
        store.set_identity(connection, member.id, identity)
        production_module.recompile_casting(connection, member.id)
        written = store.member(connection, member.id)
    assert written is not None
    return written


def _forms(inputs: dict[str, Any]) -> dict[str, Form]:
    """What a job's members have turned into, by slug."""
    given = inputs.get("forms") or {}
    return {str(slug): Form(str(f["noun"]), str(f["kind"]), str(f["look"])) for slug, f in given.items()}


def run_judge(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    chosen = JudgeSettings(**settings)
    production_id, kind = int(inputs["production"]), str(inputs["kind"])
    cast = [str(m) for m in inputs.get("cast", [])]
    # A shot's place is in its arrangement, not rated as a member: asked for its likeness the judge left it
    # out of its answer twice running and the verdict failed (production 14, 2026-10-04). A drawn place's own
    # picture is the place.
    framed = kind in ("clip", "keyframe")
    with store.connect() as connection:
        rows = [
            m for m in store.members_of(connection, production_id)
            if m.slug in cast and not (framed and m.kind == "environment")
        ]  # fmt: skip
    forms = _forms(inputs)
    cards = [compiler.formed(compiler.card_of(m), forms.get(m.slug)) for m in rows]
    allowed = {m.slug: m.allowed for m in rows if m.allowed.strip() and m.slug not in forms}
    appearance = {m.slug: m.appearance for m in rows if m.appearance.strip() and m.slug not in forms}
    turned = {str(slug): str(form["was"]) for slug, form in (inputs.get("forms") or {}).items() if form.get("was")}
    if kind == "clip":
        frames = judge.sampled(media.clip_from_video(_path(inputs["path"]), with_audio=False))
    else:
        frames = [_frame(inputs["path"])]
    pictures = {str(slug): _frame(path) for slug, path in (inputs.get("pictures") or {}).items()}
    looks = {str(m): str(v) for m, v in (inputs.get("looks") or {}).items() if v}
    becomes = {str(m): str(v) for m, v in (inputs.get("becomes") or {}).items() if v}
    try:
        verdict = judge.judge(
            frames, kind, str(inputs.get("purpose", "")), str(inputs.get("arrangement", "")), cards, chosen, pictures,
            looks, becomes, allowed, appearance, turned,
        )  # fmt: skip
    except RuntimeError as problem:  # an unusable verdict is a thing to look at, never a reason to end a render
        ui.note(f"judge · {kind}: no usable verdict ({problem})")
        return {"score": None, "issues": ["the judge gave no usable answer"], "gate": chosen.gate, "unjudged": True}
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
    passed = verdict.score >= chosen.gate
    out = {**verdict.data(), "gate": chosen.gate, "retries": chosen.retries, "passed": passed}
    prompt = str(inputs.get("prompt") or "")
    # The rewrite edits an H3 caption's sections; an LTX prompt is retaken on a new seed as it stands.
    if kind != "clip" or passed or not prompt or "subject_definitions:" not in prompt:
        return out
    with store.connect() as connection:
        if not made.retrying(connection, production_id):
            return out
    # Here and not in its own job: the VLM is loaded for the verdict, and the retake waits on both.
    try:
        revised = judge.revise(frames, prompt, verdict)
    except RuntimeError as problem:
        ui.note(f"judge · clip: the retake keeps its prompt ({problem})")
        return out
    return {**out, "revised": revised} if revised != prompt else out


def _write_chunk(clip: Clip, video: Path, soundtrack: Path, late: float = 0.0) -> None:
    """Near-lossless, since the chapter is joined from these; the audio beside it, since the mp4's is not.
    `late` seconds come off the soundtrack's start, so the carried sound meets the shot before's at the
    join; the chunk's own sound then leads its picture by as much, at most 8 ms."""
    height, width = int(clip.frames.shape[1]), int(clip.frames.shape[2])
    writer = video_io.VideoWriter(video, width, height, clip.fps, clip.sample_rate, crf=16)
    writer.write(clip.frames)
    writer.close(clip.audio)
    samples = int(np.ceil(clip.sample_rate / clip.fps * len(clip)))
    audio = clip.audio if clip.audio is not None else torch.zeros(2, samples)
    shift = round(late * clip.sample_rate)
    audio = audio[:, shift:] if shift >= 0 else torch.nn.functional.pad(audio, (-shift, 0))
    audio = torch.nn.functional.pad(audio[:, :samples], (0, max(0, samples - audio.shape[1])))
    torch.save(audio.clone(), soundtrack)


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
    "write_image_prompt": run_write_image_prompt,
    "create_image": run_create_image,
    "chat": run_chat,
    "edit_image": run_edit_image,
    "render_clip": run_render_clip,
    "write_ltx_prompt": run_write_ltx_prompt,
    "render_ltx": run_render_ltx,
    "enlarge_pixels": run_enlarge_pixels,
    "grade": run_grade,
    "retime": run_retime,
    "render_shot": run_render_shot,
    "assemble": run_assemble,
    "enlarge_shot": run_enlarge_shot,
    "finish": run_finish,
    "master": run_master,
    "film": run_film,
    "compose_keyframe": run_compose_keyframe,
    "draw_reference": run_draw_reference,
    "draw_look": run_draw_look,
    "plate_ref": run_plate_ref,
    "extend_opening": run_extend_opening,
    "extend_photo": run_extend_photo,
    "judge": run_judge,
    **PLAN_RUNNERS,
}
