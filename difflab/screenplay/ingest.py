"""How a production begins: an idea and photographs, a text, or a video to redo.

An idea makes one chapter of one scene. A text is split on its headings, or every 600 words, into
chapters the director plans one at a time. A video is cut by the render length into shots that each
open on the source's own frame at that moment and are handed that stretch of it to re-create.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from difflab import config, store
from difflab.core import media
from difflab.core.cache import cache
from difflab.core.imaging import video_length
from difflab.core.keys import key
from difflab.core.settings import Continuity
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import captions, production, settings
from difflab.screenplay import config as knobs

WORDS_PER_CHAPTER = 600  # a text with no headings: about one chapter of five 12 s shots' worth of story
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.M)


def from_idea(
    slug: str,
    idea: str,
    photos: list[str | Path],
    take_seconds: float,
    title: str = "",
    aspect: float | None = None,
    video: VideoSettings | None = None,
    continuity: Continuity | None = None,
) -> store.Production:
    """`photos[0]` is where the chapter opens; the rest show the cast. The director fills `take_seconds`
    with as many shots as the idea needs, none longer than the production's render length."""
    if not idea.strip():
        raise ValueError("an idea, in your own words: what happens")
    kept = _photos(photos)
    _long_enough(take_seconds)
    made = production.create(slug, title, video, continuity)
    return _begun(
        made.id, "idea", production.lines(idea), {"photos": kept, "take_seconds": take_seconds, "aspect": aspect}
    )


def _long_enough(take_seconds: float) -> None:
    if take_seconds < knobs.DIRECTOR_CLIP_MIN:
        raise ValueError(f"at least {knobs.DIRECTOR_CLIP_MIN:g} seconds: one shot")


def from_text(
    slug: str,
    text: str,
    photos: list[str | Path],
    take_seconds: float,
    title: str = "",
    aspect: float | None = None,
    video: VideoSettings | None = None,
    continuity: Continuity | None = None,
) -> store.Production:
    """One chapter per heading, or per 600 words without any; each is planned from its own text into
    `take_seconds` of shots. The photographs are the whole production's."""
    parts = chapters_of(text)
    if not parts:
        raise ValueError("a text to make chapters of")
    kept = _photos(photos)
    _long_enough(take_seconds)
    made = production.create(slug, title, video, continuity)
    with store.connect() as connection, store.transaction(connection):
        first = store.chapters_of(connection, made.id)[0]
        for position, (heading, story) in enumerate(parts):
            if position == 0:
                connection.execute("UPDATE chapters SET title = ?, story = ? WHERE id = ?", (heading, story, first.id))
            else:
                store.add_chapter(connection, made.id, heading, story)
    return _begun(made.id, "text", text.strip(), {"photos": kept, "take_seconds": take_seconds, "aspect": aspect})


def chapters_of(text: str) -> list[tuple[str, str]]:
    """`(heading, story)` pairs: the text under each heading, or `WORDS_PER_CHAPTER`-word stretches."""
    found = list(HEADING.finditer(text))
    if found:
        out = []
        for index, match in enumerate(found):
            end = found[index + 1].start() if index + 1 < len(found) else len(text)
            story = " ".join(text[match.end() : end].split())
            if story:
                out.append((match.group(1).strip(), story))
        return out
    words = text.split()
    count = max(1, math.ceil(len(words) / WORDS_PER_CHAPTER))
    each = math.ceil(len(words) / count) if words else 0
    return [
        (f"chapter {n + 1}", " ".join(words[n * each : (n + 1) * each]))
        for n in range(count)
        if words[n * each : (n + 1) * each]
    ]


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
    path = _under_output(source)
    if not path.is_file():
        raise ValueError(f"no such video: {source}")
    if not instruction.strip():
        raise ValueError("what should be different about it, in one line")
    if not 0.1 <= clip_scale <= 1.0:
        raise ValueError(f"clip_scale={clip_scale:g}: the source is shown between 0.1 and 1.0 of the canvas")
    kept = _photos(photos) if photos else []
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


def _photos(photos: list[str | Path]) -> list[str]:
    if not photos:
        raise ValueError("at least one photograph: the first is where the chapter opens")
    kept = []
    for photo in photos:
        path = _under_output(photo)
        if not path.is_file():
            raise ValueError(f"no such photo: {photo}")
        kept.append(store.relative(path))
    return kept


def _begun(production_id: int, kind: str, story: str, manifest: dict[str, Any]) -> store.Production:
    with store.connect() as connection, store.transaction(connection):
        store.update_production(connection, production_id, story_kind=kind, story=story, manifest=manifest)
        updated = store.production(connection, production_id)
    assert updated is not None
    return updated


def _under_output(path: str | Path) -> Path:
    given = Path(path)
    return given if given.is_absolute() else config.OUTPUT_DIR / given


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
