"""A run from a chapter: one `render_shot` job per shot, each opening on the handoff of the one
before, and one `assemble` at the end.

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

from difflab import store
from difflab.core.settings import Continuity, JudgeSettings
from difflab.screenplay import config as knobs
from difflab.screenplay import settings


def check(connection: Any, shots: list[store.Shot], production: store.Production | None = None) -> list[str]:
    """What stops a chapter rendering, said before a run exists."""
    problems = []
    if not shots:
        problems.append("the chapter has no shots")
    for shot in shots:
        keyframe = store.keyframe_of(connection, shot.id)
        prompt = store.prompt_of(connection, shot.id)
        name = f"shot {shot.position + 1}"
        if not prompt.text.strip():
            problems.append(f"{name} has no prompt")
        elif prompt.state == store.STALE and not prompt.pinned:
            problems.append(f"{name}'s prompt was compiled before its story, cast or plan changed: re-plan the chapter")
        if shot.seconds <= 0:
            problems.append(f"{name} lasts {shot.seconds:g}s")
        if keyframe.source == store.PROVIDED:
            if not keyframe.present:
                problems.append(f"{name}'s keyframe is missing from disk: {keyframe.provided}")
        elif keyframe.source == store.COMPOSED and production is not None:
            if base(connection, production, _arrangement(production, shots, shot.position)) is None:
                problems.append(
                    f"{name}'s keyframe is composed, and there is no picture to compose it over: no reference of "
                    "the place, and none of who its first frame names"
                )
        elif shot.position == 0:
            problems.append(f"{name} opens the chapter, so it needs a keyframe of its own")
    return problems


def render_chapter(chapter_id: int, asked_by: str = "person") -> str:
    """The chapter's clips, then the join, as rows the worker takes in order. Returns the run id."""
    with store.connect() as connection, store.transaction(connection):
        chapter = store.chapter(connection, chapter_id)
        if chapter is None:
            raise LookupError(f"no chapter {chapter_id}")
        production = store.production(connection, chapter.production_id)
        assert production is not None
        shots = store.shots_of(connection, chapter_id)
        problems = check(connection, shots, production)
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
        spec = {"production": production.id, "chapter": chapter.id, "items": _items(shots)}
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
        previous: int | None = None
        clip_jobs: list[int] = []
        for shot in shots:
            keyframe = store.keyframe_of(connection, shot.id)
            prompt = store.prompt_of(connection, shot.id)
            clip = store.clip_of(connection, shot.id)
            cast = [str(m) for m in shot.plan.get("cast", [])]
            judged = verdicts.gate > 0 and bool(cast or shot.action)
            arrangement = _arrangement(production, shots, shot.position)
            opening: Any
            if keyframe.source == store.COMPOSED:
                seed = keyframe.seed if keyframe.seed is not None else int(video["seed"]) + 1000 + shot.position
                composed = store.add_job(
                    connection, run_id, "compose_keyframe",
                    {
                        "production": production.id,
                        "shot": shot.id,
                        "arrangement": arrangement,
                        **(base(connection, production, arrangement) or {}),
                        "size": _canvas(production),
                        "seed": seed,
                    },
                    label=f"shot {shot.position + 1}: compose the keyframe", settings=editor.data(),
                    for_kind="keyframe", for_id=keyframe.id,
                )  # fmt: skip
                store.set_made(connection, "keyframe", keyframe.id, store.QUEUED, current_job=composed.id, seed=seed)
                if judged:
                    store.add_job(
                        connection, run_id, "judge",
                        {"production": production.id, "kind": "keyframe", "path": _output(composed.id, "frame"),
                         "purpose": shot.action, "arrangement": arrangement, "cast": cast},
                        label=f"shot {shot.position + 1}: judge the keyframe", settings=verdicts.data(),
                        for_kind="keyframe", for_id=keyframe.id,
                    )  # fmt: skip
                opening = _output(composed.id, "frame")
                if shot.position == 0:
                    anchor = opening  # the chapter's colour is the composed frame's; unused when the anchor is off
            elif keyframe.source == store.PROVIDED:
                opening = keyframe.provided
            else:
                opening = _output(previous, "frame")
            tail = _output(previous, "tail") if shot.chained and previous is not None and carries else None
            inputs = {
                "opening": opening,
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
            if judged:
                store.add_job(
                    connection, run_id, "judge",
                    {"production": production.id, "kind": "clip", "path": _output(job.id, "chunk"),
                     "purpose": shot.action, "arrangement": str(shot.plan.get("closing", "")), "cast": cast},
                    label=f"shot {shot.position + 1}: judge the clip", settings=verdicts.data(),
                    for_kind="clip", for_id=clip.id,
                )  # fmt: skip
            previous = job.id
            clip_jobs.append(job.id)
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
    return run_id


def render_production(production_id: int, asked_by: str = "person", chapter_position: int = 0) -> str:
    with store.connect() as connection:
        chapters = store.chapters_of(connection, production_id)
    if chapter_position >= len(chapters):
        raise LookupError(f"production {production_id} has {len(chapters)} chapter(s), not {chapter_position + 1}")
    return render_chapter(chapters[chapter_position].id, asked_by)


def base(connection: Any, production: store.Production, arrangement: str) -> dict[str, Any] | None:
    """What a composed keyframe is made over. The place's reference - cut from a photo, or drawn from
    its words - or the chapter's opening frame, kept as a room the arrangement's members are put into;
    with neither, the photo of the first person or animal the arrangement names, kept as that member
    and everything around it replaced. `box` is where it is in that photo."""
    members = store.members_of(connection, production.id)
    slugs = {m.id: m.slug for m in members}
    places = {m.id for m in members if m.kind == "environment"}
    references = [r for r in store.refs_of_production(connection, production.id) if r.member_id in slugs]
    for reference in references:
        if reference.role == "environment" and reference.artifact:
            return {"base": reference.artifact, "box": None, "keep": None}
    for reference in references:
        if reference.generated and reference.member_id in places and reference.artifact:
            return {"base": reference.artifact, "box": None, "keep": None}
    chapters = store.chapters_of(connection, production.id)
    shots = store.shots_of(connection, chapters[0].id) if chapters else []
    first = store.keyframe_of(connection, shots[0].id) if shots else None
    if first is not None and first.source == store.PROVIDED and first.provided:
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


def _arrangement(production: store.Production, shots: list[store.Shot], position: int) -> str:
    """What a shot opens on, in words: the production's opening, or the shot before's closing."""
    if position == 0:
        return str(production.manifest.get("opening_arrangement", ""))
    return str(shots[position - 1].plan.get("closing", ""))


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
    production_id: int, asked_by: str = "person", chapter_position: int = 0, fresh: bool = False
) -> str:
    """A plan run: the director's stages as jobs in dependency order, writing the cast, the shots of
    one chapter and their prompts. Every stage keys itself, so a second plan run with nothing changed
    skips every job; `fresh` asks the shots again, leaving the cast as it was surveyed."""
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        if production is None:
            raise LookupError(f"no production {production_id}")
        chapters = store.chapters_of(connection, production_id)
        photos = [str(p) for p in production.manifest.get("photos", [])]
        count = int(production.manifest.get("shots", 0) or 0)
        aspect = production.manifest.get("aspect")
        problems = []
        if not production.story.strip():
            problems.append("a plan needs a story: the idea the production is made from")
        if not photos:
            problems.append("a plan needs photographs: the first is where the chapter opens")
        if count < 1:
            problems.append("a plan needs how many shots to write")
        if chapter_position >= len(chapters):
            problems.append(f"the production has {len(chapters)} chapter(s), not {chapter_position + 1}")
        if problems:
            raise ValueError(problems[0])
        video, _ = settings.effective("render_chunk", production_id)
        continuity, _ = settings.effective("handoff", production_id)
        assert isinstance(continuity, Continuity)
        chapter = chapters[chapter_position]
        idea = chapter.story or production.story
        common = {"production": production_id}
        run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        spec = {"production": production_id, "chapter": chapter.id, "scope": "plan", "fresh": fresh}
        taken = {"take": time.time_ns() // 1_000_000 % 1_000_000_000} if fresh else {}
        title = (
            f"{production.title} · plan"
            if len(chapters) == 1
            else f"{production.title} · plan chapter {chapter.position + 1}"
        )
        store.add_run(
            connection, run_id, title=title, spec=spec, asked_by=asked_by, scope=f"plan:{chapter.id}",
            production_id=production_id,
        )  # fmt: skip

        def add(tool: str, inputs: dict[str, Any], label: str, chosen: dict[str, Any] | None = None) -> int:
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
        planned = add(
            "plan_shots",
            {
                "chapter": chapter.id,
                "idea": idea,
                "count": count,
                "cards": _output(tidied, "cards"),
                "arrangement": _output(arranged, "arrangement"),
                "seconds": _output(cut, "seconds"),
                "opening": _output(cut, "opening"),
                "tail": continuity.tail_frames > 0,
                "budget": knobs.DIRECTOR_REFERENCE_ROWS,
                **taken,
                **cast,
            },
            f"plan {count} shots",
            {"video": video.data()},
        )
        written = [
            add(
                "write_shot",
                {"chapter": chapter.id, "position": i, "planned": _output(planned, "chunks"), **taken},
                f"write shot {i + 1}",
            )
            for i in range(count)
        ]
        for i in range(count):
            add(
                "compile_shot",
                {"chapter": chapter.id, "position": i, "written": [_output(job, "summary") for job in written]},
                f"compile shot {i + 1}",
            )
    return run_id
