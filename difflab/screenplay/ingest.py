"""How a production begins: a story and its references, or a video to redo.

A story is an idea (one chapter of one scene) or a text, split on its headings, or every 600 words where
it has none, into chapters the director plans one at a time; the production's target length is spread
over them by their words. A reference is a photograph, a video, whose sharp frames become photographs
and which is kept as a motion reference, or a style picture, which is a member of kind `style`. A video
to redo is cut by the render length into shots that each open on the source's own frame at that moment
and are handed that stretch of it to re-create.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from difflab import store
from difflab.core import media
from difflab.core.cache import cache
from difflab.core.imaging import load_image, load_video_frames, video_length
from difflab.core.keys import key
from difflab.core.settings import Continuity, IngestSettings
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import captions, production, settings
from difflab.screenplay import config as knobs
from difflab.tools.survey import KINDS_OF

WORDS_PER_CHAPTER = 600  # a text with no headings: about one chapter of five 12 s shots' worth of story
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", re.M)
REFERENCE_KINDS = (*KINDS_OF, "style", "video")
VIDEOS = {".mp4", ".mov", ".mkv", ".webm"}
FRAMES_AT_MOST = int(next(f for f in dataclasses.fields(IngestSettings) if f.name == "frames").metadata["high"])
SHARPNESS_WIDTH = 512  # px: frames are measured at one width, so a 4K clip and a 720p one share a threshold


@dataclass(frozen=True)
class Chapter:
    title: str
    story: str
    headed: bool  # False: a stretch of WORDS_PER_CHAPTER words of a text with no heading there


def chapters_of(text: str, level: int = 1) -> list[Chapter]:
    """A chapter per heading of `level` or above; text with no such heading over it, every 600 words.
    A text with no heading that fits one chapter keeps its lines: a refined story numbers its beats."""
    found = [m for m in HEADING.finditer(text) if len(m.group(1)) <= level]
    if not HEADING.search(text) and 0 < len(text.split()) <= WORDS_PER_CHAPTER:
        return [Chapter("", production.lines(text), True)]
    parts: list[tuple[str, str]] = [("", text[: found[0].start()] if found else text)]
    for index, match in enumerate(found):
        end = found[index + 1].start() if index + 1 < len(found) else len(text)
        parts.append((match.group(2).strip(), text[match.end() : end]))
    out: list[Chapter] = []
    for title, body in parts:
        story = " ".join(HEADING.sub(lambda m: m.group(2) + ".", body).split())
        if title and story:
            out.append(Chapter(title, story, True))
        elif story:
            words = story.split()
            count = max(1, math.ceil(len(words) / WORDS_PER_CHAPTER))
            each = math.ceil(len(words) / count)
            for n in range(count):
                if words[n * each : (n + 1) * each]:
                    out.append(Chapter(f"chapter {len(out) + 1}", " ".join(words[n * each : (n + 1) * each]), False))
    return out


# --- references -------------------------------------------------------------------------------------
def sharpness(image: Image.Image) -> float:
    """Variance of the 4-neighbour Laplacian of the luma (0-255) at SHARPNESS_WIDTH px wide."""
    grey = image.convert("L")
    grey = grey.resize((SHARPNESS_WIDTH, max(1, round(grey.height * SHARPNESS_WIDTH / grey.width))))
    pixels = np.asarray(grey, dtype=np.float32)
    edges = pixels[:-2, 1:-1] + pixels[2:, 1:-1] + pixels[1:-1, :-2] + pixels[1:-1, 2:] - 4 * pixels[1:-1, 1:-1]
    return round(float(edges.var()), 1)


def frames_of(video: str | Path, count: int, least: float) -> dict[str, Any]:
    """`count` frames spread evenly over the video, each with its sharpness, kept when at least `least`;
    the sharpest is kept whatever it measures, so every video gives a picture."""
    path = store.under_output(video)
    if not path.is_file() or path.suffix.lower() not in VIDEOS:
        raise ValueError(f"no such video: {video}")
    count = min(max(count, 1), FRAMES_AT_MOST)
    seconds, _ = video_length(path)
    stat = path.stat()
    out: list[dict[str, Any]] = []
    for index in range(count):
        at = round(seconds * (index + 0.5) / count, 3)
        made = key("reference_frame", 1, video=str(path), size=stat.st_size, mtime=stat.st_mtime, at=at)
        target = cache.path(made, ".png")
        if not target.is_file():
            # Through a temporary name: the page asks again when a knob moves, and a reader must never
            # open a half-written frame.
            partial = target.with_suffix(f".{os.getpid()}.{threading.get_ident()}.png")
            load_video_frames(path, limit=1, start=at)[0][0].save(partial)
            partial.replace(target)
        measured = sharpness(load_image(target))
        out.append({"path": store.relative(target), "at": at, "sharpness": measured, "kept": measured >= least})
    sharpest = max(out, key=lambda f: float(f["sharpness"]))
    sharpest["kept"] = True
    return {"seconds": round(seconds, 2), "frames": out}


def entity_id(name: str) -> str:
    """A person's name for an entity as a member id: `Tomas Veld` is `tomas_veld`."""
    said = re.sub(r"[^a-z]+", "_", name.lower()).strip("_")
    return said[: knobs.DIRECTOR_ID_LENGTH].rstrip("_")


@dataclass(frozen=True)
class Reference:
    path: str
    kind: str
    entity: str
    guess: str


def references_of(given: list[dict[str, Any]]) -> list[Reference]:
    out: dict[str, Reference] = {}
    for row in given:
        path = store.relative(store.under_output(str(row.get("path", ""))))
        kind = str(row.get("kind") or "")
        if kind not in REFERENCE_KINDS:
            raise ValueError(f"{path}: a reference is one of {', '.join(REFERENCE_KINDS)}, not {kind!r}")
        if not store.under_output(path).is_file():
            raise ValueError(f"no such file: {path}")
        if (kind == "video") != (Path(path).suffix.lower() in VIDEOS):
            raise ValueError(f"{path}: a video is a motion reference, and a motion reference is a video")
        entity = entity_id(str(row.get("entity") or ""))
        if str(row.get("entity") or "").strip() and not entity:
            raise ValueError(f"{path}: an entity's name needs letters")
        # The same file twice is one reference: an upload's name begins with its content hash.
        out.setdefault(path, Reference(path, kind, entity, " ".join(str(row.get("guess") or "").split())))
    return list(out.values())


def guess(paths: list[str], asked_by: str = "person") -> str:
    """What each upload shows, as a run of its own: the references table's guesses before any story is read."""
    found = [store.relative(store.under_output(p)) for p in paths]
    if not found or not all(store.under_output(p).is_file() for p in found):
        raise ValueError("pictures under output/ to describe")
    with store.connect() as connection, store.transaction(connection):
        run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        store.add_run(connection, run_id, title=f"describe {len(found)} upload(s)", asked_by=asked_by, scope="guess")
        for path in found:
            store.add_job(
                connection, run_id, "guess_reference", {"path": path}, label=f"describe {Path(path).name}",
                settings=settings.for_job("guess_reference"),
            )  # fmt: skip
    return run_id


def slug_for(title: str) -> str:
    """A free slug from the title: `The Lighthouse at Vell` is `the-lighthouse-at-vell`, then `-2`."""
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48] or "production"
    with store.connect() as connection:
        taken = {p.slug for p in store.productions(connection)}
    candidate, n = base, 1
    while candidate in taken:
        n += 1
        candidate = f"{base}-{n}"
    return candidate


# --- a production from a story ----------------------------------------------------------------------
def from_story(
    slug: str,
    story: str,
    references: list[Reference],
    target_seconds: float,
    shortest: float,
    title: str = "",
    aspect: float | None = None,
    video: VideoSettings | None = None,
    continuity: Continuity | None = None,
    ingesting: IngestSettings | None = None,
) -> store.Production:
    """The story split by `ingesting.heading_level`. The first location photograph, or the first
    photograph, is where chapter 1 opens."""
    chosen = ingesting or IngestSettings()
    parts = chapters_of(story, chosen.heading_level)
    if not parts:
        raise ValueError("a story, in your own words or as a file: what happens")
    if target_seconds < knobs.DIRECTOR_CLIP_MIN:
        raise ValueError(f"at least {knobs.DIRECTOR_CLIP_MIN:g} seconds: one shot")
    longest = (video or VideoSettings()).duration
    if not knobs.DIRECTOR_CLIP_MIN <= shortest <= longest:
        raise ValueError(f"shots between {knobs.DIRECTOR_CLIP_MIN:g} s and the longest, {longest:g} s")
    rows = []
    for reference in references:
        frames = frames_of(reference.path, chosen.frames, chosen.min_sharpness) if reference.kind == "video" else None
        rows.append({**reference.__dict__, **({"frames": frames["frames"]} if frames else {})})
    photos, labels = _photos(references, rows)
    if not photos:
        raise ValueError("at least one photograph, or a video with a sharp frame: chapter 1 opens on one")
    made = production.create(slug, title, video, continuity)
    if ingesting is not None:
        overrides = {k: v for k, v in ingesting.data().items() if v != IngestSettings().data()[k]}
        with store.connect() as connection, store.transaction(connection):
            store.put_overrides(connection, settings.PRODUCTION, str(made.id), "ingest", overrides)
    with store.connect() as connection, store.transaction(connection):
        first = store.chapters_of(connection, made.id)[0]
        for position, part in enumerate(parts):
            if position == 0:
                connection.execute(
                    "UPDATE chapters SET title = ?, story = ? WHERE id = ?", (part.title, part.story, first.id)
                )
            else:
                store.add_chapter(connection, made.id, part.title, part.story)
        _styles(connection, made.id, [r for r in references if r.kind == "style"])
    manifest = {
        "photos": photos, "labels": labels, "references": rows, "target_seconds": target_seconds,
        "shortest": shortest, "aspect": aspect,
    }  # fmt: skip
    return _begun(made.id, "text", production.lines(story), manifest)


def _photos(references: list[Reference], rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, str]]]:
    """The pictures the survey sees, the first location first, and what the person said each one shows."""
    photos: list[str] = []
    labels: dict[str, dict[str, str]] = {}
    for reference, row in zip(references, rows, strict=True):
        if reference.kind == "style":
            continue
        kept = [f["path"] for f in row.get("frames", []) if f["kept"]] if reference.kind == "video" else [row["path"]]
        for path in kept:
            photos.append(path)
            labels[path] = {"kind": "" if reference.kind == "video" else reference.kind, "entity": reference.entity}
    places = [p for p in photos if labels[p]["kind"] == "location"]
    if places:
        photos.remove(places[0])
        photos.insert(0, places[0])
    return photos, labels


def _styles(connection: Any, production_id: int, styles: list[Reference]) -> None:
    for reference in styles:
        slug = reference.entity or "style"
        member = store.put_member(connection, production_id, slug, "style", slug.replace("_", " "), "its", "")
        if reference.guess and not member.appearance:
            store.update_member(connection, member.id, appearance=reference.guess)
        production.add_style_picture(connection, member, reference.path)


# --- a production that redoes a video ---------------------------------------------------------------
def from_video(
    slug: str,
    source: str | Path,
    instruction: str,
    photos: list[str | Path] | None = None,
    title: str = "",
    clip_scale: float = 1.0,
    video: VideoSettings | None = None,
    continuity: Continuity | None = None,
) -> store.Production:
    """The video cut into shots of the render length - a remainder under half a shot is left alone -
    each opening on the source's frame at that moment, handed that stretch of it as <Video 1>, and
    captioned with `instruction` as the one difference. The photographs ride along as references."""
    path = store.under_output(source)
    if not path.is_file():
        raise ValueError(f"no such video: {source}")
    if not instruction.strip():
        raise ValueError("what should be different about it, in one line")
    if not 0.1 <= clip_scale <= 1.0:
        raise ValueError(f"clip_scale={clip_scale:g}: the source is shown between 0.1 and 1.0 of the canvas")
    kept = [store.relative(store.under_output(p)) for p in photos or []]
    made = production.create(slug, title, video, continuity)
    chosen, _ = settings.effective("render_chunk", made.id)
    carried, _ = settings.effective("handoff", made.id)
    assert isinstance(chosen, VideoSettings) and isinstance(carried, Continuity)
    seconds, fps = video_length(path)
    whole = round(seconds, 3) / chosen.duration
    count = max(1, int(whole) if whole - int(whole) < 0.5 else math.ceil(whole))
    relative = store.relative(path)
    for index in range(count):
        start = index * chosen.duration
        end = min(start + chosen.duration, seconds)
        anchor = _anchor(path, start)
        prompt = captions.caption_for(instruction, len(kept) + 1, keyframe=True)
        prompt = captions.cite_source(prompt, start, end, tail=index > 0 and carried.tail_frames > 0)
        shot = production.add_shot(made.id, prompt, chosen.duration, keyframe=anchor, action=instruction.strip())
        with store.connect() as connection, store.transaction(connection):
            store.update_shot(
                connection,
                shot.id,
                plan={"source": {"video": relative, "start": start, "seconds": end - start, "scale": clip_scale}},
            )
            store.set_prompt(connection, shot.id, prompt, written_by="ingest")
            connection.execute("UPDATE prompts SET pictures = ? WHERE shot_id = ?", (_dump(kept), shot.id))
    return _begun(
        made.id,
        "video",
        instruction.strip(),
        {"photos": kept, "video": relative, "seconds": seconds, "fps": fps, "clip_scale": clip_scale, "shots": count},
    )


def _anchor(video: Path, start: float) -> str:
    """The source's frame at `start`, as a picture under output/: what the shot opens on."""
    stat = video.stat()
    made = key("anchor", 1, video=str(video), size=stat.st_size, mtime=stat.st_mtime, start=round(start, 3))
    target = cache.path(made, ".png")
    if not target.is_file():
        clip = media.clip_from_video(video, limit=1, with_audio=False, start=start)
        clip.frame(0).pil().save(target)
    return store.relative(target)


def _begun(production_id: int, kind: str, story: str, manifest: dict[str, Any]) -> store.Production:
    with store.connect() as connection, store.transaction(connection):
        store.update_production(connection, production_id, story_kind=kind, story=story, manifest=manifest)
        updated = store.production(connection, production_id)
    assert updated is not None
    return updated


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
