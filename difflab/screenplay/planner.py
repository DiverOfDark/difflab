"""A run from a chapter: one `render_shot` job per shot, each opening on the handoff of the one
before or on its own keyframe - which the shot before then ends on - and one `assemble` at the end.

Every shot gets a job whether or not its clip is made: a chained shot's key is not known until the
shot before it has rendered, so it is the worker that finds the artifact already there and skips.
Rendering a chapter twice therefore samples nothing, and a re-rolled shot re-renders itself and the
shots chained after it and nothing else.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from difflab import config, store
from difflab.core.settings import Continuity, JudgeSettings
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import compiler, settings, words
from difflab.screenplay import config as knobs
from difflab.screenplay import production as productions

_LAST_FRAME = words.SENTENCES["last_frame"].split(":")[0]


def check(
    connection: Any, shots: list[store.Shot], production: store.Production | None = None, keyframes_only: bool = False
) -> list[str]:
    """What stops a chapter rendering, said before a run exists. Making only its keyframes reads no prompt."""
    problems = []
    if not shots:
        problems.append("the chapter has no shots")
    _, lands = compiler.joins([store.keyframe_of(connection, shot.id) for shot in shots], shots, True)
    for shot, landing in [] if keyframes_only else zip(shots, lands, strict=True):
        text = store.prompt_of(connection, shot.id).text
        if text.strip() and landing != (_LAST_FRAME in text):
            problems.append(
                f"shot {shot.position + 1} {'ends' if landing else 'no longer ends'} on shot {shot.position + 2}'s "
                f"keyframe, and its prompt, written by hand, {'does not say' if landing else 'still says'} so: "
                f"{'add' if landing else 'drop'} “{_LAST_FRAME}: …”, or turn the landing "
                f"{'off' if landing else 'on'}"
            )
    for shot in shots:
        keyframe = store.keyframe_of(connection, shot.id)
        prompt = store.prompt_of(connection, shot.id)
        name = f"shot {shot.position + 1}"
        if keyframes_only:
            pass
        elif not prompt.text.strip():
            problems.append(f"{name} has no prompt")
        elif prompt.state == store.STALE and not prompt.pinned:
            problems.append(f"{name}'s prompt was compiled before its story, cast or plan changed: re-plan the chapter")
        if shot.seconds <= 0:
            problems.append(f"{name} lasts {shot.seconds:g}s")
        if keyframe.source == store.PROVIDED:
            if not keyframe.present:
                problems.append(f"{name}'s keyframe is missing from disk: {keyframe.provided}")
        elif keyframe.source == store.COMPOSED and production is not None:
            if base(connection, production, *_opening(connection, production, shots, shot.position)) is None:
                problems.append(
                    f"{name}'s keyframe is composed, and there is no picture to compose it over: no reference of "
                    "the place, and none of who its first frame names"
                )
        elif shot.position == 0:
            problems.append(f"{name} opens the chapter, so it needs a keyframe of its own")
    return problems


def render_chapter(chapter_id: int, asked_by: str = "person", keyframes_only: bool = False) -> str:
    """The chapter's clips, then the join, as rows the worker takes in order. Returns the run id.
    `keyframes_only` stops after the composed keyframes and their verdicts: minutes of clips are not
    spent on a first frame that is still being decided."""
    with store.connect() as connection, store.transaction(connection):
        return _queue_chapter(connection, chapter_id, asked_by, keyframes_only)


def render_production(
    production_id: int, asked_by: str = "person", chapter_position: int | None = None, keyframes_only: bool = False
) -> list[str]:
    """One chapter, or every chapter as a run each in chapter order; one refused chapter queues none."""
    with store.connect() as connection, store.transaction(connection):
        chapters = _chosen(connection, production_id, chapter_position)
        runs = []
        for chapter in chapters:
            try:
                runs.append(_queue_chapter(connection, chapter.id, asked_by, keyframes_only))
            except ValueError as problem:
                raise ValueError(
                    f"chapter {chapter.position + 1}: {problem}" if len(chapters) > 1 else str(problem)
                ) from None
        return runs


def _chosen(connection: Any, production_id: int, position: int | None) -> list[store.Chapter]:
    chapters = store.chapters_of(connection, production_id)
    if position is None:
        return chapters
    if not 0 <= position < len(chapters):
        raise LookupError(f"production {production_id} has {len(chapters)} chapter(s), not {position + 1}")
    return [chapters[position]]


def _queue_chapter(connection: Any, chapter_id: int, asked_by: str, keyframes_only: bool) -> str:
    chapter = store.chapter(connection, chapter_id)
    if chapter is None:
        raise LookupError(f"no chapter {chapter_id}")
    production = store.production(connection, chapter.production_id)
    assert production is not None
    shots = store.shots_of(connection, chapter_id)
    # A prompt compiled before its shot's join changed - a landing added, one turned off - says the
    # wrong pictures; the compiler's own are brought up to date here, a hand-written one is `check`'s.
    _, lands = compiler.joins([store.keyframe_of(connection, shot.id) for shot in shots], shots, True)
    for shot, landing in zip(shots, lands, strict=True):
        prompt = store.prompt_of(connection, shot.id)
        if shot.plan.get("beats") and not prompt.pinned and landing != (_LAST_FRAME in prompt.text):
            productions.recompile(connection, production, chapter, shot.id)
    shots = store.shots_of(connection, chapter_id)
    problems = check(connection, shots, production, keyframes_only)
    if problems:
        raise ValueError(problems[0])
    chosen, _ = settings.effective("render_chunk", production.id)
    video = chosen.data()
    continuity_chosen, _ = settings.effective("handoff", production.id)
    assert isinstance(continuity_chosen, Continuity)
    continuity = continuity_chosen
    continuity_data = continuity.data()
    carries = continuity.tail_frames > 0

    run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    chapters = store.chapters_of(connection, production.id)
    title = production.title if len(chapters) == 1 else f"{production.title} · chapter {chapter.position + 1}"
    spec = {
        "production": production.id, "chapter": chapter.id, "items": _items(shots),
        **({"keyframes_only": True} if keyframes_only else {}),
    }  # fmt: skip
    store.add_run(
        connection,
        run_id,
        title=title,
        spec=spec,
        asked_by=asked_by,
        scope=f"chapter:{chapter.id}",
        production_id=production.id,
    )

    first = store.keyframe_of(connection, shots[0].id)
    anchor = first.provided if continuity.color_anchor else None
    verdicts, _ = settings.effective("judge", production.id)
    assert isinstance(verdicts, JudgeSettings)
    editor, _ = settings.effective("edit_image", production.id)
    keyframes = [store.keyframe_of(connection, shot.id) for shot in shots]
    tails, lands = compiler.joins(keyframes, shots, carries)
    # Every composed keyframe before any clip: the shot before one ends on it, and the editor is
    # loaded once rather than between clips.
    openings: list[Any] = []
    framed: list[tuple[dict[str, Any], str, int]] = []
    for shot, keyframe in zip(shots, keyframes, strict=True):
        cast = [str(m) for m in shot.plan.get("cast", [])]
        judged = bool(cast or shot.action)
        arrangement, where = _opening(connection, production, shots, shot.position)
        if keyframe.source == store.COMPOSED:
            seed = keyframe.seed if keyframe.seed is not None else int(video["seed"]) + 1000 + shot.position
            composed = store.add_job(
                connection, run_id, "compose_keyframe",
                {
                    "production": production.id,
                    "shot": shot.id,
                    "arrangement": arrangement,
                    **(under := base(connection, production, arrangement, where) or {}),
                    "pictures": pictures(connection, production, arrangement, under.get("keep")),
                    "note": keyframe.note,
                    "looks": _looks(connection, production, shot)[0],
                    "size": _canvas(production),
                    "seed": seed,
                },
                label=f"shot {shot.position + 1}: compose the keyframe", settings=editor.data(),
                for_kind="keyframe", for_id=keyframe.id,
            )  # fmt: skip
            store.set_made(connection, "keyframe", keyframe.id, store.QUEUED, current_job=composed.id, seed=seed)
            if judged:
                framed.append((
                    {"production": production.id, "kind": "keyframe", "path": _output(composed.id, "frame"),
                     "purpose": f"the shot's first frame, as the arrangement says, before its action starts "
                                f"({shot.action})",
                     "arrangement": arrangement, "cast": cast,
                     "pictures": pictures(connection, production, arrangement),
                     "looks": _looks(connection, production, shot)[0]},
                    f"shot {shot.position + 1}: judge the keyframe", keyframe.id,
                ))  # fmt: skip
            openings.append(_output(composed.id, "frame"))
            if shot.position == 0:
                anchor = openings[0]  # the chapter's colour is the composed frame's; unused when the anchor is off
        elif keyframe.source == store.PROVIDED:
            openings.append(keyframe.provided)
        else:
            openings.append(None)

    # The editor composes every keyframe, then the VLM judges them all: one swap, not one per shot.
    for inputs, label, keyframe_id in framed:
        store.add_job(
            connection,
            run_id,
            "judge",
            inputs,
            label=label,
            settings=verdicts.data(),
            for_kind="keyframe",
            for_id=keyframe_id,
        )
    if keyframes_only:
        return run_id
    previous: int | None = None
    clip_jobs: list[int] = []
    opened: list[Any] = []
    for shot, keyframe in zip(shots, keyframes, strict=True):
        prompt = store.prompt_of(connection, shot.id)
        clip = store.clip_of(connection, shot.id)
        cast = [str(m) for m in shot.plan.get("cast", [])]
        judged = bool(cast or shot.action)
        opening = openings[shot.position] if openings[shot.position] is not None else _output(previous, "frame")
        tail = _output(previous, "tail") if tails[shot.position] else None
        inputs = {
            "opening": opening,
            "closing": openings[shot.position + 1] if lands[shot.position] else None,
            "prompt": prompt.text,
            "tail": tail,
            "anchor": anchor,
            "seed": clip.seed,
            "seconds": shot.seconds,
            "position": shot.position,
            "count": len(shots),
            "references": list(prompt.pictures),
            "source": shot.plan.get("source"),
        }
        job = store.add_job(
            connection,
            run_id,
            "render_shot",
            inputs,
            label=f"shot {shot.position + 1}: {shot.action or prompt.text[:48]}",
            settings={"video": video, "continuity": continuity_data},
            for_kind="clip",
            for_id=clip.id,
        )
        store.set_made(connection, "clip", clip.id, store.QUEUED, current_job=job.id)
        if keyframe.source == store.CHAINED and previous is not None:
            store.set_made(connection, "keyframe", keyframe.id, store.QUEUED, current_job=previous)
        # Right after its clip, so a failing shot pauses the chain before the shots built on it are
        # rendered (`made._judged`). Measured on production 9: the VLM loads in 11 s and judges in ~10 s,
        # and H3 reloads in ~9 s, beside 633 s for the clip.
        if judged:
            store.add_job(
                connection, run_id, "judge",
                {"production": production.id, "kind": "clip", "path": _output(job.id, "chunk"),
                 "purpose": shot.action, "arrangement": str(shot.plan.get("closing", "")), "cast": cast,
                 "pictures": pictures(connection, production, " ".join(f"{{{m}}}" for m in cast)),
                 "looks": (changed := _looks(connection, production, shot))[0], "becomes": changed[1]},
                label=f"shot {shot.position + 1}: judge the clip", settings=verdicts.data(),
                for_kind="clip", for_id=clip.id,
            )  # fmt: skip
        previous = job.id
        clip_jobs.append(job.id)
        opened.append(opening)
    store.add_job(
        connection,
        run_id,
        "assemble",
        {
            "chunks": [_output(job_id, "chunk") for job_id in clip_jobs],
            "audios": [_output(job_id, "audio") for job_id in clip_jobs],
            "cuts": [_output(job_id, "cut") for job_id in clip_jobs],
            "keys": [_output(job_id, "key") for job_id in clip_jobs],
            "crossfade": continuity.crossfade,
            "name": production.slug,
        },
        label="join the chapter",
    )
    # After the native join, so a watchable chapter exists an hour before the 4K one does.
    enlarging, _ = settings.effective("restore", production.id)
    retiming, _ = settings.effective("retime", production.id)
    enlarged = [
        store.add_job(
            connection,
            run_id,
            "enlarge_shot",
            {
                "latents": _output(job_id, "latents"),
                "key": _output(job_id, "key"),
                "opening": opening,
                "megapixels": config.FINISH_MEGAPIXELS,
            },
            label=f"shot {position + 1}: enlarge",
            settings={"video": video, "continuity": continuity_data, "enlarge": enlarging.data()},
        ).id
        for position, (job_id, opening) in enumerate(zip(clip_jobs, opened, strict=True))
    ]
    store.add_job(
        connection,
        run_id,
        "finish",
        {
            "chunks": [_output(job_id, "chunk") for job_id in enlarged],
            "audios": [_output(job_id, "audio") for job_id in clip_jobs],
            "cuts": [_output(job_id, "cut") for job_id in clip_jobs],
            "keys": [_output(job_id, "key") for job_id in enlarged],
            "crossfade": continuity.crossfade,
            "name": production.slug,
            "fps": config.FINISH_FPS,
        },
        label=f"the chapter enlarged and at {config.FINISH_FPS} fps",
        settings={"retime": retiming.data()},
    )
    return run_id


def base(
    connection: Any, production: store.Production, arrangement: str, where: str | None = None
) -> dict[str, Any] | None:
    """What a composed keyframe is made over. The place's reference - cut from a photo, or drawn from
    its words - or the chapter's opening frame, kept as a room the arrangement's members are put into;
    with neither, the photo of the first person or animal the arrangement names, kept as that member
    and everything around it replaced. `box` is where it is in that photo. `where` is the place the
    shot happens in: a place no photo shows is never drawn over the opening room."""
    members = store.members_of(connection, production.id)
    slugs = {m.id: m.slug for m in members}
    places = {m.id for m in members if m.kind == "environment" and where in (None, m.slug)}
    unseen = any(not m.identity.get("photographed", True) for m in members if m.id in places)
    references = [r for r in store.refs_of_production(connection, production.id) if r.member_id in slugs]
    for reference in references:
        if reference.role == "environment" and reference.member_id in places and reference.artifact:
            return {"base": reference.artifact, "box": None, "keep": None}
    for reference in references:
        if reference.generated and reference.member_id in places and reference.artifact:
            return {"base": reference.artifact, "box": None, "keep": None}
    chapters = store.chapters_of(connection, production.id)
    shots = store.shots_of(connection, chapters[0].id) if chapters else []
    first = store.keyframe_of(connection, shots[0].id) if shots else None
    if not unseen and first is not None and first.source == store.PROVIDED and first.provided:
        return {"base": first.provided, "box": None, "keep": None}
    photos = [str(p) for p in production.manifest.get("photos", [])]
    living = {m.slug for m in members if m.kind in ("person", "animal")}
    for slug in dict.fromkeys(re.findall(r"\{([a-z_]+)\}", arrangement)):
        own = [r for r in references if slugs[r.member_id] == slug and slug in living and r.photo]
        # The first photo before the others, since the survey found it the one the chapter opens on; a
        # whole body before a face; the larger crop of the rest.
        own.sort(key=lambda r: (r.photo != (photos[0] if photos else ""), r.role not in ("view", "body"), -r.quality))
        if own:
            return {"base": own[0].photo, "box": own[0].box, "keep": slug}
    return None


# Picture 1 is the place: three more is what the editor still holds each one to.
PICTURES = 3


def pictures(
    connection: Any, production: store.Production, arrangement: str, keep: str | None = None
) -> dict[str, str]:
    """Each person and animal the arrangement names, by id, as the picture the editor and the judge are
    handed of it: a whole body cut from a photo before a face, a given or drawn one when that is all."""
    members = {m.slug: m for m in store.members_of(connection, production.id)}
    out: dict[str, str] = {}
    for slug in dict.fromkeys(re.findall(r"\{([a-z_]+)\}", arrangement)):
        member = members.get(slug)
        if member is None or member.kind not in ("person", "animal") or slug == keep or len(out) == PICTURES:
            continue
        own = [r for r in store.refs_of(connection, member.id) if r.artifact]
        own.sort(key=lambda r: (r.role not in ("body", "view"), r.generated, r.role == "face", -r.quality))
        if own:
            out[slug] = str(own[0].artifact)
    return out


def _looks(connection: Any, production: store.Production, shot: store.Shot) -> tuple[dict[str, str], dict[str, str]]:
    """What earlier shots changed about each member of this one for good, and what this one changes."""
    chapter = store.chapter(connection, shot.chapter_id)
    assert chapter is not None
    take = compiler.take_of(connection, production, chapter)
    chunk = take.chunks[shot.position]
    carried = {m: look for m in chunk.cast if (look := compiler.look(take, shot.position, m))}
    return carried, {m: v for m, v in chunk.looks.items() if v}


def _opening(
    connection: Any, production: store.Production, shots: list[store.Shot], position: int
) -> tuple[str, str | None]:
    """What a shot opens on, in words, as its prompt says it, and the place it happens in."""
    chapter = store.chapter(connection, shots[position].chapter_id)
    assert chapter is not None
    take = compiler.take_of(connection, production, chapter)
    chunk = take.chunks[position]
    where = next((m for m in chunk.cast if take.cards[m].kind == "environment"), None)
    return compiler.opening(take, position), where


def _canvas(production: store.Production) -> list[int] | None:
    canvas = production.manifest.get("canvas") or {}
    return [int(canvas["width"]), int(canvas["height"])] if canvas.get("width") and canvas.get("height") else None


def _output(job_id: int | None, name: str) -> dict[str, Any]:
    assert job_id is not None
    return {"from_job": job_id, "output": name}


def _items(shots: list[store.Shot]) -> list[str]:
    items, at = [], 0.0
    for shot in shots:
        items.append(f"shot {shot.position + 1}/{len(shots)} · {at:g}–{at + shot.seconds:g}s")
        at += shot.seconds
    return items


def plan_production(
    production_id: int, asked_by: str = "person", chapter_position: int | None = None, fresh: bool = False
) -> list[str]:
    """A plan run per chapter, one or every one in order; `fresh` asks the shots again, not the cast."""
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        if production is None:
            raise LookupError(f"no production {production_id}")
        chapters = store.chapters_of(connection, production_id)
        photos = [str(p) for p in production.manifest.get("photos", [])]
        video, _ = settings.effective("render_chunk", production_id)
        assert isinstance(video, VideoSettings)
        total = productions.take_seconds(production.manifest, video.duration)
        problems = []
        if not production.story.strip():
            problems.append("a plan needs a story: the idea the production is made from")
        if not photos:
            problems.append("a plan needs photographs: the first is where the chapter opens")
        if total < knobs.DIRECTOR_CLIP_MIN:
            problems.append(f"a plan needs a length of at least {knobs.DIRECTOR_CLIP_MIN:g} seconds to fill")
        if problems:
            raise ValueError(problems[0])
        return [
            _plan_chapter(connection, production, chapter, len(chapters), asked_by, video, total, fresh)
            for chapter in _chosen(connection, production_id, chapter_position)
        ]


def _plan_chapter(
    connection: Any,
    production: store.Production,
    chapter: store.Chapter,
    count: int,
    asked_by: str,
    video: VideoSettings,
    total: float,
    fresh: bool,
) -> str:
    production_id = production.id
    photos = [str(p) for p in production.manifest.get("photos", [])]
    aspect = production.manifest.get("aspect")
    continuity, _ = settings.effective("handoff", production_id)
    assert isinstance(continuity, Continuity)
    idea = chapter.story or production.story
    common = {"production": production_id}
    run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    spec = {"production": production_id, "chapter": chapter.id, "scope": "plan", "fresh": fresh}
    taken = {"take": time.time_ns() // 1_000_000 % 1_000_000_000} if fresh else {}
    title = f"{production.title} · plan" if count == 1 else f"{production.title} · plan chapter {chapter.position + 1}"
    store.add_run(
        connection, run_id, title=title, spec=spec, asked_by=asked_by, scope=f"plan:{chapter.id}",
        production_id=production_id,
    )  # fmt: skip

    def add(tool: str, inputs: dict[str, Any], label: str, chosen: dict[str, Any] | None = None) -> int:
        if chosen is None:
            chosen = settings.for_job(tool, production_id)
        return store.add_job(connection, run_id, tool, {**common, **inputs}, label=label, settings=chosen or {}).id

    surveyed = add("survey", {"idea": idea, "photos": photos}, "survey the photographs")
    cast = {"members": _output(surveyed, "members"), "shows": _output(surveyed, "shows")}
    grounds = [
        add("ground", {"photo": photo, "index": i, **cast}, f"find the cast in photo {i + 1}")
        for i, photo in enumerate(photos)
    ]
    found = [_output(job, "found") for job in grounds]
    cut = add(
        "cut_crops", {"photos": photos, "found": found, "aspect": aspect, **cast}, "cut the references",
        {"video": video.data()},
    )  # fmt: skip
    carded = [
        add("cards", {"photo": photo, "index": i, **cast}, f"read the cast in photo {i + 1}")
        for i, photo in enumerate(photos)
    ]
    described = add(
        "describe_opening",
        {
            "opening": _output(cut, "opening"),
            "found": found[0],
            "cut": _output(cut, "cut"),
            "card": _output(carded[0], "card"),
            **cast,
        },
        "describe the opening frame",
    )
    tidied = add(
        "tidy",
        {"idea": idea, "cards": [_output(job, "card") for job in carded], "found": found, **cast},
        "tidy the cards",
    )
    arranged = add(
        "arrange",
        {
            "seen": _output(described, "seen"),
            "in_opening": _output(described, "in_opening"),
            "cards": _output(tidied, "cards"),
            **cast,
        },
        "write the opening sentence",
    )
    add(
        "plan_shots",
        {
            "chapter": chapter.id,
            "idea": idea,
            "take_seconds": total,
            "cards": _output(tidied, "cards"),
            "arrangement": _output(arranged, "arrangement"),
            "seconds": _output(cut, "seconds"),
            "opening": _output(cut, "opening"),
            "tail": continuity.tail_frames > 0,
            "budget": knobs.DIRECTOR_REFERENCE_ROWS,
            **taken,
            **cast,
        },
        f"plan {total:g} seconds of shots",
        {"video": video.data()},
    )
    return run_id
