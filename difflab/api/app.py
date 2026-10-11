"""The HTTP interface: runs, uploads, tools, and a run as it happens.

One process serves this and runs the worker beside it (`difflab.api.serve`). A request never renders:
it writes rows the worker picks up, or reads rows the worker wrote. Simplicity over isolation: the
process holds the weights, so a crash in a job restarts the whole thing.

There is no authentication. That is a deliberate choice for now and a bad one to publish: see
`serve.py`, which says so out loud on every start that is not bound to localhost.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from difflab import config
from difflab.api import assets, runs
from difflab.core import hub, profiling
from difflab.core.settings import Continuity, IngestSettings
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import (
    assembling,
    drawn,
    ideas,
    ingest,
    ingesting,
    looks,
    planner,
    production,
    readers,
    rendering,
    settings,
)
from difflab.work import queue
from difflab.work.quick import Job

STATIC = Path(__file__).parent / "static"  # the built Vue app, when there is one

app = FastAPI(title="difflab", docs_url=None, redoc_url=None)


@app.middleware("http")
async def _never_stale(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Browsers keep index.html across a rebuild otherwise, and load a bundle that no longer exists.
    The bundles themselves carry a content hash in their name and may be kept."""
    response = await call_next(request)
    if not request.url.path.startswith("/assets/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


# --- a job, checked ----------------------------------------------------------------------------
@app.post("/api/jobs/check")
async def check_job(request: Request) -> dict[str, Any]:
    """What is wrong with this job, without rendering it. The form asks on every edit."""
    job = _job_from(await request.json())
    return {"problems": job.check(), "title": job.title, "job": job.data()}


# --- productions ---------------------------------------------------------------------------------
@app.get("/api/productions")
def list_productions() -> list[dict[str, Any]]:
    return production.listing()


@app.get("/api/productions/{production_id}")
def read_production(production_id: int) -> dict[str, Any]:
    return _production(production_id)


@app.get("/api/productions/{production_id}/timeline")
def read_timeline(production_id: int, chapter: int | None = None) -> dict[str, Any]:
    """Chapter `chapter` (from 0) or every one: each shot light, and each member's looks as runs, marks, gaps."""
    found = production.timeline(production_id, chapter)
    if found is None:
        raise HTTPException(404, "no such production")
    return found


@app.patch("/api/productions/{production_id}")
async def patch_production(production_id: int, request: Request) -> dict[str, Any]:
    """`title`, `story`, `opening` (the first frame in words). A changed story or opening says how
    much went stale under `stale`."""
    body = await request.json()
    return _staling(production_id, lambda: production.update(production_id, **dict(body)))


@app.get("/api/productions/{production_id}/ideas")
def list_ideas(production_id: int) -> list[dict[str, Any]]:
    try:
        return ideas.listing(production_id)
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem


@app.post("/api/productions/{production_id}/ideas")
async def improve_idea(production_id: int, request: Request) -> dict[str, Any]:
    """A round as a run: `from_round` with `feedback` revises that round; without, a fresh rewrite of `idea`."""
    body = await request.json()
    revises = None if body.get("from_round") in (None, "") else int(body["from_round"])
    try:
        run = ideas.improve(production_id, str(body.get("feedback") or ""), revises, body.get("idea") or None)
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    except (sqlite3.Error, OSError) as problem:
        raise HTTPException(500, f"the run could not be queued ({problem})") from problem
    return {"run": run}


@app.post("/api/productions/{production_id}/ideas/{round_}/adopt")
def adopt_idea(production_id: int, round_: int) -> dict[str, Any]:
    """The round's story becomes the production's; `stale` says how much a re-plan now has to redo."""
    return _staling(production_id, lambda: ideas.adopt(production_id, round_))


@app.post("/api/productions/{production_id}/members")
async def add_member(production_id: int, request: Request) -> dict[str, Any]:
    """A member decided in words: `slug`, `kind`, `noun`, `possessive`, `find`, `traits`, `lights`."""
    body = dict(await request.json())
    try:
        production.add_member(production_id, str(body.pop("slug", "")).strip(), **body)
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except (TypeError, ValueError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return _production(production_id)


@app.patch("/api/members/{member_id}")
async def patch_member(member_id: int, request: Request) -> dict[str, Any]:
    """A member by hand, pinned; `{"pinned": false}` alone gives it back to the director."""
    body = await request.json()
    return _staling(_member_owner(member_id), lambda: production.update_member(member_id, **dict(body)))


@app.delete("/api/members/{member_id}")
def delete_member(member_id: int) -> dict[str, Any]:
    return _staling(_member_owner(member_id), lambda: production.delete_member(member_id))


@app.delete("/api/refs/{ref_id}")
def detach_ref(ref_id: int) -> dict[str, Any]:
    """A reference that does not show its member, let go of for good."""
    owner = production.production_of_ref(ref_id)
    if owner is None:
        raise HTTPException(404, "no such reference")
    return _staling(owner, lambda: production.detach_ref(ref_id))


@app.post("/api/productions/{production_id}/references")
def draw_references(production_id: int) -> dict[str, Any]:
    """Every member with no usable picture, drawn by the editor from its words, as a run."""
    return {"run": _draw(production_id)}


@app.post("/api/members/{member_id}/reference")
async def draw_reference(member_id: int, request: Request) -> dict[str, Any]:
    """One member's picture drawn from its words; drawn again, a new seed unless `seed` is given.
    `describe` has a place's words written from the story first."""
    body = await request.json()
    owner = production.production_of_member(member_id)
    if owner is None:
        raise HTTPException(404, "no such member")
    seed = None if body.get("seed") in (None, "") else int(body["seed"])
    return {"run": _draw(owner, member_id, seed, bool(body.get("describe")))}


def _draw(production_id: int, member_id: int | None = None, seed: int | None = None, describe: bool = False) -> str:
    try:
        return drawn.draw(production_id, member_id, seed, describe=describe)
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    except (sqlite3.Error, OSError) as problem:
        raise HTTPException(500, f"the run could not be queued ({problem})") from problem


@app.post("/api/members/{member_id}/variants")
def draw_variants(member_id: int) -> dict[str, Any]:
    """Three drawings of the member on new seeds, to pick one of."""
    owner = production.production_of_member(member_id)
    if owner is None:
        raise HTTPException(404, "no such member")
    try:
        return {"run": drawn.variants(owner, member_id)}
    except (LookupError, ValueError) as problem:
        raise HTTPException(400, str(problem)) from problem


@app.post("/api/members/{member_id}/variants/pick")
async def pick_variant(member_id: int, request: Request) -> dict[str, Any]:
    body = await request.json()
    return _staling(_member_owner(member_id), lambda: production.pick_drawn(member_id, int(body["seed"])))


@app.post("/api/shots/{shot_id}/keyframe")
async def redo_keyframe(shot_id: int, request: Request) -> dict[str, Any]:
    """The shot's keyframe composed again with `note` added to what the editor is told, and only the
    chapter's keyframes made, not its clips."""
    body = await request.json()
    try:
        chapter = production.redo_keyframe(shot_id, str(body.get("note", "")))
        return {"run": planner.render_chapter(chapter, keyframes_only=True)}
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem


@app.post("/api/productions/{production_id}/keyframes")
def render_keyframes(production_id: int, chapter: int | None = None) -> dict[str, Any]:
    """The composed keyframes and their verdicts, without clips: of chapter `chapter` (from 0), or of every one."""
    return _runs(lambda: planner.render_production(production_id, chapter_position=chapter, keyframes_only=True))


@app.post("/api/members/{member_id}/given")
async def give_reference(member_id: int, request: Request) -> dict[str, Any]:
    """`path`, an uploaded picture of the member, held as its `role` picture; says what went stale."""
    body = await request.json()
    return _staling(
        _member_owner(member_id), lambda: production.give_ref(member_id, str(body["path"]), str(body["role"]))
    )


@app.post("/api/members/{member_id}/wardrobe")
async def dress(member_id: int, request: Request) -> dict[str, Any]:
    """`path`, an uploaded picture of a garment the member can change into, called `name`; no `path` takes it out."""
    body = await request.json()
    path = str(body["path"]) if body.get("path") else None
    return _staling(_member_owner(member_id), lambda: production.dress(member_id, str(body.get("name", "")), path))


@app.post("/api/conflicts/{conflict_id}/resolve")
async def resolve_conflict(conflict_id: int, request: Request) -> dict[str, Any]:
    """`resolution`: which wording is kept, in words; empty opens it again."""
    body = await request.json()
    try:
        owner = production.resolve_conflict(conflict_id, str(body.get("resolution", "")))
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    return _production(owner)


@app.post("/api/conflicts/{conflict_id}/keep")
async def keep_conflict(conflict_id: int, request: Request) -> dict[str, Any]:
    """`keep`: `reference` rewrites the story's sentences to the photograph, `novel` takes the story's trait."""
    body = await request.json()
    owner = production.production_of_conflict(conflict_id)
    if owner is None:
        raise HTTPException(404, "no such conflict")
    return _staling(owner, lambda: production.keep(conflict_id, str(body.get("keep", ""))))


@app.post("/api/productions/{production_id}/unmatched")
async def attach_unmatched(production_id: int, request: Request) -> dict[str, Any]:
    """`photo`, one the survey placed with no member, held as member `to`'s picture - its `garment` of that
    name when one is given - or `to` "ignore"."""
    body = await request.json()
    photo, to, garment = str(body.get("photo", "")), str(body.get("to", "")), str(body.get("garment") or "")
    return _staling(production_id, lambda: production.attach(production_id, photo, to, garment))


@app.post("/api/productions/{production_id}/photos")
async def add_photos(production_id: int, request: Request) -> dict[str, Any]:
    """`paths`, pictures given after ingest: they wait in the bible to be attached, and the next plan reads them."""
    body = await request.json()
    try:
        production.add_photos(production_id, [str(p) for p in body.get("paths") or []])
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    return _production(production_id)


@app.post("/api/productions/{production_id}/bible/approve")
def approve_bible(production_id: int) -> dict[str, Any]:
    """Every member's picture that is made, approved by a person; `left` says which could not be."""
    try:
        _, left = production.approve_bible(production_id)
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    return {**_production(production_id), "left": left}


@app.post("/api/members/{member_id}/describe")
def describe_member(member_id: int) -> dict[str, Any]:
    """The member read again from its photographs as a run, and given back to the director."""
    try:
        return {"run": production.describe_again(member_id)}
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem


@app.delete("/api/productions/{production_id}")
def delete_production(production_id: int) -> dict[str, bool]:
    production.delete(production_id)
    return {"ok": True}


@app.patch("/api/shots/{shot_id}")
async def patch_shot(shot_id: int, request: Request) -> dict[str, Any]:
    body = await request.json()
    return _staling(_shot_owner(shot_id), lambda: production.update_shot(shot_id, **dict(body)))


@app.delete("/api/shots/{shot_id}")
def delete_shot(shot_id: int) -> dict[str, Any]:
    owner = production.production_of_shot(shot_id)
    if owner is None:
        raise HTTPException(404, "no such shot")
    production.remove_shot(shot_id)
    return _production(owner)


@app.post("/api/shots/{shot_id}/reroll")
async def reroll_shot(shot_id: int, request: Request) -> dict[str, Any]:
    """One more attempt at the shot's clip, and the chapter queued again: the shots chained after it
    follow by key, the ones before are skipped."""
    body = await request.json()
    owner = production.production_of_shot(shot_id)
    if owner is None:
        raise HTTPException(404, "no such shot")
    seed, chapter = production.reroll(shot_id, None if body.get("seed") in (None, "") else int(body["seed"]))
    # A run the judge paused for this shot is let go first.
    return {"seed": seed, "run": _runs(lambda: [planner.render_chapter(chapter)])["run"]}


@app.post("/api/shots/{shot_id}/rewrite")
def rewrite_shot(shot_id: int) -> dict[str, Any]:
    """The shot's prose and prompt written again as a run; a prompt written by hand is let go."""
    return _runs(lambda: [production.rewrite(shot_id)])


@app.post("/api/shots/{shot_id}/approve")
def approve_shot(shot_id: int) -> dict[str, Any]:
    """The shot's prompt, keyframe and looks approved, each that is made; `left` says what was not."""
    try:
        owner, left = production.approve_shot(shot_id)
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    return {**_production(owner), "left": left}


# --- castings: who is in a shot, and how they look there --------------------------------------------
@app.post("/api/shots/{shot_id}/castings")
async def add_casting(shot_id: int, request: Request) -> dict[str, Any]:
    """`member` (its slug) into the shot, as it stands unless `parent`, `amend` or `label` say otherwise."""
    body = dict(await request.json())
    return _staling(_shot_owner(shot_id), lambda: production.add_casting(shot_id, str(body.pop("member", "")), **body))


@app.patch("/api/castings/{casting_id}")
async def patch_casting(casting_id: int, request: Request) -> dict[str, Any]:
    """`parent` (null its last look, "base", or an earlier look's casting id), `amend`, `label`, `pictured`."""
    body = dict(await request.json())
    return _staling(_casting_owner(casting_id), lambda: production.edit_casting(casting_id, **body))


@app.delete("/api/castings/{casting_id}")
def delete_casting(casting_id: int) -> dict[str, Any]:
    return _staling(_casting_owner(casting_id), lambda: production.remove_casting(casting_id))


@app.post("/api/castings/{casting_id}/upload")
async def upload_look(casting_id: int, request: Request) -> dict[str, Any]:
    """`path`, an uploaded picture, as the look's own instead of a drawn one, approved."""
    body = await request.json()
    return _staling(_casting_owner(casting_id), lambda: production.upload_look(casting_id, str(body.get("path", ""))))


@app.post("/api/castings/{casting_id}/draw")
def draw_look(casting_id: int) -> dict[str, Any]:
    """The look drawn by the editor from its source's picture and judged, as a run; drawn again, a new seed."""
    owner = _casting_owner(casting_id)
    return _runs(lambda: [looks.draw(owner, [casting_id])])


@app.post("/api/shots/{shot_id}/looks")
def draw_shot_looks(shot_id: int) -> dict[str, Any]:
    """Every look of the shot with no picture yet, or a failed or stale one, drawn as one run."""
    owner = production.production_of_shot(shot_id)
    if owner is None:
        raise HTTPException(404, "no such shot")
    return _runs(lambda: [looks.draw(owner, production.undrawn_looks(shot_id))])


@app.post("/api/productions/{production_id}/looks")
def draw_production_looks(production_id: int, chapter: int | None = None) -> dict[str, Any]:
    """Every look of chapter `chapter` (from 0), or of every one, with no picture yet, drawn as one run."""
    return _runs(lambda: [looks.draw(production_id, production.undrawn_looks_of(production_id, chapter))])


def _staling(owner: int, change: Callable[[], Any]) -> dict[str, Any]:
    """The production after `change`, with `stale`: what the change counted, and the shots it left stale."""
    before = production.stale_shots(owner)
    try:
        moved = change()
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except (TypeError, ValueError, KeyError, OSError) as problem:
        raise HTTPException(400, str(problem)) from problem
    except sqlite3.Error as problem:
        raise HTTPException(500, f"the change could not be written ({problem})") from problem
    return {**_production(owner), "stale": production.affected(owner, before, moved if isinstance(moved, dict) else {})}


def _member_owner(member_id: int) -> int:
    owner = production.production_of_member(member_id)
    if owner is None:
        raise HTTPException(404, "no such member")
    return owner


def _shot_owner(shot_id: int) -> int:
    owner = production.production_of_shot(shot_id)
    if owner is None:
        raise HTTPException(404, "no such shot")
    return owner


def _casting_owner(casting_id: int) -> int:
    owner = production.production_of_casting(casting_id)
    if owner is None:
        raise HTTPException(404, "no such casting")
    return owner


@app.post("/api/made/{kind}/{made_id}/approve")
def approve(kind: str, made_id: int) -> dict[str, Any]:
    """A person's approval: the judge no longer overrules it, and a run paused on the clip goes on."""
    try:
        owner = production.approve(kind, made_id)
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    return _production(owner)


@app.post("/api/ingest/read")
async def read_story(file: UploadFile) -> dict[str, Any]:
    """A story file (epub, pdf, txt, md, docx) as markdown text, for the page to split and send back."""
    data = await file.read()
    try:
        text = await asyncio.to_thread(readers.read, data, file.filename or "story.txt")
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    return {"name": file.filename, "text": text, "words": len(text.split()), "bytes": len(data)}


@app.post("/api/ingest/frames")
async def reference_frames(request: Request) -> dict[str, Any]:
    """`path`, an uploaded video: `frames` frames spread over it, each measured and kept at `min_sharpness`."""
    body = await request.json()
    try:
        return await asyncio.to_thread(
            ingest.frames_of, str(body.get("path", "")), int(body.get("frames") or 12),
            float(body.get("min_sharpness") or 0),
        )  # fmt: skip
    except (TypeError, ValueError, OSError) as problem:
        raise HTTPException(400, str(problem)) from problem


@app.post("/api/ingest/guess")
async def guess_references(request: Request) -> dict[str, Any]:
    """`paths`, uploaded pictures, each described by the VLM as a run: its kind and what it shows."""
    body = await request.json()
    try:
        return {"run": ingest.guess([str(p) for p in body.get("paths") or []])}
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem


@app.post("/api/ingest")
async def ingest_story(request: Request) -> dict[str, Any]:
    """A production from a story and its references: `story`, `references` ({path,
    kind, entity, guess}), `title`, `target_seconds` for the whole film, `shortest` shot, `aspect`, `fit`, and
    partial `video`, `continuity` and `ingest` settings over the code defaults. The bible is built at once;
    `plan` plans every chapter too, stopping before render."""
    body = await request.json()
    try:
        video = VideoSettings(**body["video"]) if body.get("video") else None
        continuity = Continuity(**body["continuity"]) if body.get("continuity") else None
        chosen = IngestSettings(**body["ingest"]) if body.get("ingest") else None
        title = " ".join(str(body.get("title", "")).split())
        if not title:
            raise ValueError("a title, in words")
        made = ingest.from_story(
            str(body.get("slug") or "").strip() or ingest.slug_for(title),
            str(body.get("story", "")),
            ingest.references_of(list(body.get("references") or [])),
            float(body.get("target_seconds") or 0),
            float(body.get("shortest") or 0),
            title=title,
            aspect=None if body.get("aspect") in (None, "") else float(body["aspect"]),
            fit=str(body.get("fit") or "crop"),
            video=video,
            continuity=continuity,
            ingesting=chosen,
        )
    except (TypeError, ValueError, OSError) as problem:
        raise HTTPException(400, str(problem)) from problem
    planned = _runs(lambda: planner.plan_production(made.id, shots=bool(body.get("plan"))))
    return {**_production(made.id), "runs": planned["runs"]}


@app.get("/api/productions/{production_id}/ingesting")
def read_ingesting(production_id: int) -> dict[str, Any]:
    """The newest plan as the ingesting page shows it: four steps, what was found, a log, elapsed and ETA."""
    found = ingesting.status(production_id)
    if found is None:
        raise HTTPException(404, "no such production")
    return found


@app.post("/api/productions/{production_id}/ingesting")
async def steer_ingesting(production_id: int, request: Request) -> dict[str, int]:
    """`{"skip": step}` hands on what that step has not done yet, in every run of the plan; `{"cancel": true}`
    cancels the plan."""
    body = await request.json()
    try:
        if body.get("cancel"):
            return {"cancelled": ingesting.cancel(production_id)}
        return {"skipped": ingesting.skip(production_id, str(body.get("skip", "")))}
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem


@app.get("/api/timings")
def read_timings() -> dict[str, Any]:
    """Per tool, the median seconds a job takes on this machine: what a fresh ingest estimates with."""
    return production.timings_now()


@app.post("/api/ingest/video")
async def ingest_video(request: Request) -> dict[str, Any]:
    """A production that redoes a video: cut by the render length into shots that each open on the
    source's frame and re-create that stretch of it with `instruction` as the one difference."""
    body = await request.json()
    try:
        video = VideoSettings(**body["video"]) if body.get("video") else None
        continuity = Continuity(**body["continuity"]) if body.get("continuity") else None
        title = " ".join(str(body.get("title", "")).split())
        if not title:
            raise ValueError("a title, in words")
        made = ingest.from_video(
            str(body.get("slug") or "").strip() or ingest.slug_for(title),
            str(body.get("source", "")),
            str(body.get("instruction", "")),
            [str(p) for p in body.get("photos") or []],
            title=title,
            clip_scale=float(body.get("clip_scale") or 1.0),
            video=video,
            continuity=continuity,
        )
    except (TypeError, ValueError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return _production(made.id)


@app.post("/api/productions/{production_id}/plan")
def plan_production(
    production_id: int, chapter: int | None = None, fresh: bool = False, shots: bool = True
) -> dict[str, Any]:
    """Chapter `chapter` (from 0) or every one, a plan run each; `fresh` asks for new shots though nothing changed.
    Without `shots`, one run that builds the bible again from the story and the photographs and plans no chapter."""
    return _runs(lambda: planner.plan_production(production_id, chapter_position=chapter, fresh=fresh, shots=shots))


@app.post("/api/productions/{production_id}/render")
def render_production(production_id: int, chapter: int | None = None) -> dict[str, Any]:
    """Chapter `chapter` (from 0), or every one, a run each, queued in chapter order."""
    return _runs(lambda: planner.render_production(production_id, chapter_position=chapter))


@app.post("/api/productions/{production_id}/assemble")
def assemble_production(production_id: int, chapter: int | None = None) -> dict[str, Any]:
    """Chapter `chapter` (from 0), or every one and the film, joined, levelled and subtitled from the clips it
    has; nothing renders."""
    return _runs(lambda: assembling.queue(production_id, chapter))


@app.post("/api/productions/{production_id}/finish")
def finish_production(production_id: int, chapter: int | None = None) -> dict[str, Any]:
    """Chapter `chapter` (from 0), or every one and the film, enlarged to 4K and retimed from the clips it has."""
    return _runs(lambda: assembling.finish(production_id, chapter))


@app.get("/api/productions/{production_id}/assembled")
def read_assembled(production_id: int) -> dict[str, Any]:
    """The Assemble page: each chapter's joins, dialogue and newest masters, and the film."""
    found = assembling.status(production_id)
    if found is None:
        raise HTTPException(404, "no such production")
    return found


@app.get("/api/productions/{production_id}/queue")
def read_queue(production_id: int, chapter: int | None = None) -> dict[str, Any]:
    """The Render page: every shot's clip as the queue holds it, NOW, ETA and the rejection rate."""
    found = rendering.status(production_id, chapter)
    if found is None:
        raise HTTPException(404, "no such production")
    return found


@app.post("/api/productions/{production_id}/queue")
async def steer_queue(production_id: int, request: Request) -> dict[str, Any]:
    """`{"pause": bool}` pauses every live render run of the production after its clip in flight, or lets them
    go; `{"auto_retry": bool}` is the judge's `clip_retry` for the production, read at each verdict."""
    body = await request.json()
    out: dict[str, Any] = {}
    if "pause" in body:
        out["told"] = rendering.hold(production_id, bool(body["pause"]))
    if "auto_retry" in body:
        out["auto_retry"] = rendering.auto_retry(production_id, bool(body["auto_retry"]))
    return out


@app.get("/api/productions/{production_id}/preflight")
def preflight(production_id: int) -> dict[str, Any]:
    """Per chapter position, what would stop it rendering; nothing is queued."""
    try:
        return {"problems": planner.preflight(production_id)}
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem


def _runs(queue_them: Callable[[], list[str]]) -> dict[str, Any]:
    """`run` is the first one queued, the one the worker reaches first."""
    try:
        runs = queue_them()
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    except (sqlite3.Error, OSError) as problem:
        raise HTTPException(500, f"the run could not be queued ({problem})") from problem
    return {"run": runs[0] if runs else None, "runs": runs}


def _production(production_id: int) -> dict[str, Any]:
    found = production.describe(production_id)
    if found is None:
        raise HTTPException(404, "no such production")
    return found


# --- tools -------------------------------------------------------------------------------------
@app.get("/api/tools")
def list_tools(production: int | None = None) -> list[dict[str, Any]]:
    """Every tool with its tasks and its settings schema - what a form draws and what a key includes -
    and, per tool, what this machine and the production override and what that comes to."""
    return settings.describe(production)


@app.put("/api/settings/{tool}")
async def put_settings(tool: str, request: Request) -> dict[str, Any]:
    """The whole override set for one tool in one scope: `{"scope_kind": "machine" | "production",
    "scope_id": "" | "<production id>", "overrides": {...}}`. Empty overrides clear the scope."""
    body = await request.json()
    try:
        return settings.put(
            str(body.get("scope_kind", "machine")),
            str(body.get("scope_id", "")),
            tool,
            dict(body.get("overrides") or {}),
        )
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem


@app.get("/api/models")
def list_models() -> dict[str, list[str]]:
    """Per model role, what was chosen before, newest first."""
    return settings.models()


@app.post("/api/models/resolve")
async def resolve_model(request: Request) -> dict[str, Any]:
    """`{"text": ..., "role": ...}`: a pasted Hugging Face or Civitai page, pinned as a model field of `role`
    stores it - `value` when it means one file, `choices` to pick from when the page holds several."""
    body = await request.json()
    file = settings.takes_file(str(body.get("role", "")))
    try:
        found = await asyncio.to_thread(hub.resolve, str(body.get("text", "")), file)
    except (ValueError, KeyError, OSError) as problem:
        raise HTTPException(400, str(problem)) from problem
    if len(found) == 1:
        return {"value": found[0].value}
    return {"choices": [dataclasses.asdict(choice) for choice in found]}


@app.delete("/api/models")
async def forget_model(request: Request) -> dict[str, list[str]]:
    """`{"role": ..., "value": ...}`: no longer offered. A setting still naming it keeps it."""
    body = await request.json()
    settings.forget(str(body.get("role", "")), str(body.get("value", "")))
    return settings.models()


@app.get("/api/accounts")
def list_accounts() -> dict[str, dict[str, str]]:
    return settings.accounts()


@app.put("/api/accounts/{service}")
async def save_account(service: str, request: Request) -> dict[str, dict[str, str]]:
    """`{"token": ...}`: checked with the service, then used for every download; "" forgets it."""
    body = await request.json()
    try:
        await asyncio.to_thread(settings.save_token, service, str(body.get("token", "")))
    except (ValueError, OSError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return settings.accounts()


@app.post("/api/settings/impact")
async def settings_impact(request: Request) -> dict[str, Any]:
    """What saving these overrides would move, before they are saved."""
    body = await request.json()
    try:
        return settings.impact(
            str(body.get("scope_kind", "")), str(body.get("scope_id", "")), str(body.get("tool", "")),
            dict(body.get("overrides") or {}),
        )  # fmt: skip
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem


# --- uploads -----------------------------------------------------------------------------------
@app.get("/api/uploads")
def list_uploads() -> list[dict[str, Any]]:
    return assets.listing()


@app.post("/api/uploads")
async def upload(files: list[UploadFile]) -> list[dict[str, Any]]:
    stored = []
    for item in files:
        try:
            stored.append(assets.store(await item.read(), item.filename or "photo.jpg"))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    return stored


# Served from /uploads/, not /assets/: the built page puts its own bundles under /assets/, and a
# route that shadows those makes the whole app a blank screen.
@app.get("/uploads/{name}")
def asset(name: str) -> FileResponse:
    path = assets.resolve(name)
    if path is None:
        raise HTTPException(404, "no such file")
    return FileResponse(path)


@app.get("/api/thumb")
def thumb(path: str) -> FileResponse:
    """A small copy of a picture a job names, wherever the pod keeps it."""
    small = assets.preview_of(Path(path))
    if small is None:
        raise HTTPException(404, "no picture there")
    return FileResponse(small)


# --- runs --------------------------------------------------------------------------------------
@app.post("/api/runs")
async def start_run(request: Request) -> dict[str, Any]:
    """Enqueue a job: its tool jobs become rows, and the worker takes them in order. Nothing is
    refused for being busy - there is one card and a queue in front of it."""
    job = _job_from(await request.json())
    problems = job.check()
    if problems:
        raise HTTPException(400, problems[0])
    try:
        run_id = queue.enqueue(job)
    except (sqlite3.Error, OSError) as problem:
        raise HTTPException(500, f"the run could not be queued ({problem})") from problem
    return {"run": run_id, "title": job.title}


@app.post("/api/runs/batch")
async def start_runs(request: Request) -> list[dict[str, Any]]:
    """A list of jobs, each its own run in the order given; one refused queues none of them."""
    jobs = [_job_from(data) for data in await request.json()]
    for index, job in enumerate(jobs):
        problems = job.check()
        if problems:
            raise HTTPException(400, f"{job.title} (#{index + 1}): {problems[0]}")
    try:
        return [{"run": queue.enqueue(job), "title": job.title} for job in jobs]
    except (sqlite3.Error, OSError) as problem:
        raise HTTPException(500, f"the runs could not be queued ({problem})") from problem


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return runs.runs()


@app.get("/api/gpu")
def read_gpu() -> dict[str, Any]:
    """The header's card: what it is doing (idle, planning, drawing, judging, rendering, assembling) and its VRAM."""
    return runs.gpu()


@app.get("/api/runs/{run_id}")
def read_run(run_id: str) -> dict[str, Any]:
    """One run in full: what it was asked for, its jobs with their tasks, and what it made."""
    found = runs.one(Path(run_id).name)
    if found is None:
        raise HTTPException(404, "no such run")
    return found


@app.post("/api/runs/{run_id}/commands")
async def command(run_id: str, request: Request) -> dict[str, bool]:
    problem = runs.control(Path(run_id).name, await request.json())
    if problem:
        raise HTTPException(409 if "not running" in problem or "no job" in problem else 400, problem)
    return {"ok": True}


@app.post("/api/restart")
def restart() -> dict[str, str]:
    """The server restarts on its code as it is now, once no job is running: a job in flight finishes first."""
    from difflab.work import worker

    worker.RESTART.set()
    return {"restart": "when the queue is idle"}


@app.get("/api/profile")
def read_profile() -> dict[str, Any]:
    return profiling.state()


@app.post("/api/profile")
async def arm_profile(request: Request) -> dict[str, Any]:
    """torch.profiler over the next `steps` sampler steps; `stack` adds Python stacks at several times the cost."""
    asked = await request.json()
    steps = asked.get("steps", 2)
    if not isinstance(steps, int) or steps < 1:
        raise HTTPException(400, "steps is a whole number of at least 1")
    return profiling.arm(steps, bool(asked.get("stack", False)))


@app.delete("/api/profile")
def disarm_profile() -> dict[str, Any]:
    return profiling.disarm()


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str) -> StreamingResponse:
    """The run's events as server-sent events: enough to draw it as it stands, then each new one.

    The catch-up first means a page can join a take halfway through, or replay a finished one, with no
    separate route - the fold that draws the page is the same either way.
    """
    name = Path(run_id).name
    if not runs.exists(name):
        raise HTTPException(404, "no such run")
    return StreamingResponse(
        _tail(name),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _tail(run_id: str) -> Any:
    cursor, quiet_since = 0, asyncio.get_event_loop().time()
    batch = runs.catch_up(run_id)
    while True:
        finished = False
        for event in batch:
            cursor = event.id
            yield b"data: " + event.line().encode("utf-8") + b"\n\n"
            finished = finished or event.kind == "run.end"
        now = asyncio.get_event_loop().time()
        if batch:
            quiet_since = now
        elif now - quiet_since > 10:
            yield b": ping\n\n"  # keeps proxies and idle browsers from closing us
            quiet_since = now
        if finished or (not batch and runs.gone(run_id)):
            yield b"event: done\ndata: {}\n\n"
            return
        await asyncio.sleep(0.25)
        batch = runs.since(run_id, cursor)


# --- the files a run made ------------------------------------------------------------------------
@app.get("/runs/{run_id}/{path:path}")
def run_file(run_id: str, path: str) -> FileResponse:
    return _under(config.RUNS_DIR / Path(run_id).name, path)


SERVED_OUTPUT = ("blobs", "images", "videos")  # every path a job delivers, or an event names, is under one


@app.get("/output/{path:path}")
def output_file(path: str) -> FileResponse:
    # Not the rest of output/: it holds the database, with the saved Hugging Face and Civitai tokens,
    # and in the container the weights.
    folder, _, rest = path.partition("/")
    if folder not in SERVED_OUTPUT:
        raise HTTPException(404, "no such file")
    return _under(config.OUTPUT_DIR / folder, rest)


def _under(root: Path, relative: str) -> FileResponse:
    """A file inside `root`, and nothing outside it. FileResponse answers Range, so video seeks."""
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(target)


def _job_from(data: Any) -> Job:
    try:
        return Job.of(data)
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error


# --- the page ------------------------------------------------------------------------------------
if STATIC.is_dir():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="app")
else:

    @app.get("/")
    def unbuilt() -> dict[str, str]:
        return {"difflab": "the page is not built yet - run `npm --prefix frontend run build`"}
