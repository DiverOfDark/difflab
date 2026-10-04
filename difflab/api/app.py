"""The HTTP interface: runs, uploads, tools, and a run as it happens.

One process serves this and runs the worker beside it (`difflab.api.serve`). A request never renders:
it writes rows the worker picks up, or reads rows the worker wrote. Simplicity over isolation: the
process holds the weights, so a crash in a job restarts the whole thing.

There is no authentication. That is a deliberate choice for now and a bad one to publish: see
`serve.py`, which says so out loud on every start that is not bound to localhost.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from difflab import config
from difflab.api import assets, runs
from difflab.core import hub
from difflab.core.settings import Continuity
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import drawn, ideas, ingest, planner, production, settings
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


@app.post("/api/productions")
async def create_production(request: Request) -> dict[str, Any]:
    """A production with one chapter. `video` and `continuity` are partial settings over the code
    defaults; what is not given is not stored either."""
    body = await request.json()
    try:
        video = VideoSettings(**body["video"]) if body.get("video") else None
        continuity = Continuity(**body["continuity"]) if body.get("continuity") else None
        made = production.create(
            str(body.get("slug", "")).strip(), str(body.get("title", "")).strip(), video, continuity
        )
    except (TypeError, ValueError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return _production(made.id)


@app.get("/api/productions/{production_id}")
def read_production(production_id: int) -> dict[str, Any]:
    return _production(production_id)


@app.patch("/api/productions/{production_id}")
async def patch_production(production_id: int, request: Request) -> dict[str, Any]:
    """`title`, `story`, `opening` (the first frame in words). A changed story or opening says how
    much went stale under `stale`."""
    body = await request.json()
    try:
        moved = production.update(production_id, **dict(body))
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except (TypeError, ValueError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return {**_production(production_id), "stale": moved}


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
    try:
        moved = ideas.adopt(production_id, round_)
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    return {**_production(production_id), "stale": moved}


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
    owner = production.production_of_member(member_id)
    if owner is None:
        raise HTTPException(404, "no such member")
    try:
        moved = production.update_member(member_id, **dict(body))
    except (TypeError, ValueError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return {**_production(owner), "stale": moved}


@app.delete("/api/members/{member_id}")
def delete_member(member_id: int) -> dict[str, Any]:
    owner = production.production_of_member(member_id)
    if owner is None:
        raise HTTPException(404, "no such member")
    try:
        moved = production.delete_member(member_id)
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    return {**_production(owner), "stale": moved}


@app.delete("/api/refs/{ref_id}")
def detach_ref(ref_id: int) -> dict[str, Any]:
    """A reference that does not show its member, let go of for good."""
    owner = production.production_of_ref(ref_id)
    if owner is None:
        raise HTTPException(404, "no such reference")
    moved = production.detach_ref(ref_id)
    return {**_production(owner), "stale": moved}


@app.post("/api/productions/{production_id}/references")
def draw_references(production_id: int) -> dict[str, Any]:
    """Every member with no usable picture, drawn by the editor from its words, as a run."""
    return {"run": _draw(production_id)}


@app.post("/api/members/{member_id}/reference")
async def draw_reference(member_id: int, request: Request) -> dict[str, Any]:
    """One member's picture drawn from its words; drawn again, a new seed unless `seed` is given."""
    body = await request.json()
    owner = production.production_of_member(member_id)
    if owner is None:
        raise HTTPException(404, "no such member")
    seed = None if body.get("seed") in (None, "") else int(body["seed"])
    return {"run": _draw(owner, member_id, seed)}


def _draw(production_id: int, member_id: int | None = None, seed: int | None = None) -> str:
    try:
        return drawn.draw(production_id, member_id, seed)
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    except (sqlite3.Error, OSError) as problem:
        raise HTTPException(500, f"the run could not be queued ({problem})") from problem


@app.post("/api/members/{member_id}/given")
async def give_reference(member_id: int, request: Request) -> dict[str, Any]:
    """`path`, an uploaded picture of the member, held as its `role` picture; says what went stale."""
    body = await request.json()
    try:
        moved = production.give_ref(member_id, str(body["path"]), str(body["role"]))
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except (KeyError, ValueError, OSError) as problem:
        raise HTTPException(400, str(problem)) from problem
    owner = production.production_of_member(member_id)
    assert owner is not None
    return {**_production(owner), "stale": moved}


@app.post("/api/conflicts/{conflict_id}/resolve")
async def resolve_conflict(conflict_id: int, request: Request) -> dict[str, Any]:
    """`resolution`: which wording is kept, in words; empty opens it again."""
    body = await request.json()
    try:
        owner = production.resolve_conflict(conflict_id, str(body.get("resolution", "")))
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    return _production(owner)


@app.delete("/api/productions/{production_id}")
def delete_production(production_id: int) -> dict[str, bool]:
    production.delete(production_id)
    return {"ok": True}


@app.post("/api/productions/{production_id}/shots")
async def add_shot(production_id: int, request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        production.add_shot(
            production_id,
            prompt=str(body.get("prompt", "")),
            seconds=float(body.get("seconds") or 0),
            keyframe=body.get("keyframe") or None,
            seed=None if body.get("seed") in (None, "") else int(body["seed"]),
            action=str(body.get("action", "")),
            chained=bool(body.get("chained", True)),
        )
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except (TypeError, ValueError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return _production(production_id)


@app.patch("/api/shots/{shot_id}")
async def patch_shot(shot_id: int, request: Request) -> dict[str, Any]:
    body = await request.json()
    owner = production.production_of_shot(shot_id)
    if owner is None:
        raise HTTPException(404, "no such shot")
    try:
        production.update_shot(shot_id, **dict(body))
    except (TypeError, ValueError, LookupError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return _production(owner)


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
    seed = production.reroll(shot_id, None if body.get("seed") in (None, "") else int(body["seed"]))
    return {"seed": seed, "run": _render(owner)}


@app.post("/api/ingest/idea")
async def ingest_idea(request: Request) -> dict[str, Any]:
    """A production from an idea and photographs: one chapter of `take_seconds`, the first photo where
    it opens. `video` and `continuity` are partial settings over the code defaults."""
    body = await request.json()
    try:
        video = VideoSettings(**body["video"]) if body.get("video") else None
        continuity = Continuity(**body["continuity"]) if body.get("continuity") else None
        made = ingest.from_idea(
            str(body.get("slug", "")).strip(),
            str(body.get("idea", "")),
            [str(p) for p in body.get("photos") or []],
            float(body.get("take_seconds") or 0),
            title=str(body.get("title", "")).strip(),
            aspect=None if body.get("aspect") in (None, "") else float(body["aspect"]),
            video=video,
            continuity=continuity,
        )
    except (TypeError, ValueError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return _production(made.id)


@app.post("/api/ingest/text")
async def ingest_text(request: Request) -> dict[str, Any]:
    """A production from a text and photographs: one chapter per heading, or per 600 words."""
    body = await request.json()
    try:
        video = VideoSettings(**body["video"]) if body.get("video") else None
        continuity = Continuity(**body["continuity"]) if body.get("continuity") else None
        made = ingest.from_text(
            str(body.get("slug", "")).strip(),
            str(body.get("text", "")),
            [str(p) for p in body.get("photos") or []],
            float(body.get("take_seconds") or 0),
            title=str(body.get("title", "")).strip(),
            aspect=None if body.get("aspect") in (None, "") else float(body["aspect"]),
            video=video,
            continuity=continuity,
        )
    except (TypeError, ValueError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return _production(made.id)


@app.post("/api/ingest/video")
async def ingest_video(request: Request) -> dict[str, Any]:
    """A production that redoes a video: cut by the render length into shots that each open on the
    source's frame and re-create that stretch of it with `instruction` as the one difference."""
    body = await request.json()
    try:
        video = VideoSettings(**body["video"]) if body.get("video") else None
        continuity = Continuity(**body["continuity"]) if body.get("continuity") else None
        made = ingest.from_video(
            str(body.get("slug", "")).strip(),
            str(body.get("source", "")),
            str(body.get("instruction", "")),
            [str(p) for p in body.get("photos") or []],
            title=str(body.get("title", "")).strip(),
            clip_scale=float(body.get("clip_scale") or 1.0),
            video=video,
            continuity=continuity,
        )
    except (TypeError, ValueError) as problem:
        raise HTTPException(400, str(problem)) from problem
    return _production(made.id)


@app.post("/api/productions/{production_id}/plan")
def plan_production(production_id: int, chapter: int = 0, fresh: bool = False) -> dict[str, Any]:
    """The director over one chapter's story and the production's photographs: its cast, shots and
    prompts, as a run. `fresh` asks for new shots even when nothing they are planned from changed."""
    try:
        return {"run": planner.plan_production(production_id, chapter_position=chapter, fresh=fresh)}
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    except (sqlite3.Error, OSError) as problem:
        raise HTTPException(500, f"the run could not be queued ({problem})") from problem


@app.post("/api/productions/{production_id}/render")
def render_production(production_id: int, chapter: int = 0) -> dict[str, Any]:
    return {"run": _render(production_id, chapter)}


def _render(production_id: int, chapter: int = 0) -> str:
    try:
        return planner.render_production(production_id, chapter_position=chapter)
    except LookupError as problem:
        raise HTTPException(404, str(problem)) from problem
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    except (sqlite3.Error, OSError) as problem:
        raise HTTPException(500, f"the run could not be queued ({problem})") from problem


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


@app.post("/api/models")
async def remember_model(request: Request) -> dict[str, list[str]]:
    """`{"role": ..., "value": "org/repo@<commit>[:files]"}`: offered for that role from now on."""
    body = await request.json()
    try:
        settings.remember(str(body.get("role", "")), str(body.get("value", "")).strip())
    except ValueError as problem:
        raise HTTPException(400, str(problem)) from problem
    return settings.models()


@app.post("/api/models/resolve")
async def resolve_model(request: Request) -> dict[str, str]:
    """`{"text": ...}`: a pasted Hugging Face or Civitai page, pinned as a model field stores it."""
    body = await request.json()
    try:
        return {"value": await asyncio.to_thread(hub.resolve, str(body.get("text", "")))}
    except (ValueError, KeyError, OSError) as problem:
        raise HTTPException(400, str(problem)) from problem


@app.delete("/api/models")
async def forget_model(request: Request) -> dict[str, list[str]]:
    """`{"role": ..., "value": ...}`: no longer offered. A setting still naming it keeps it."""
    body = await request.json()
    settings.forget(str(body.get("role", "")), str(body.get("value", "")))
    return settings.models()


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


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return runs.runs()


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


@app.get("/output/{path:path}")
def output_file(path: str) -> FileResponse:
    return _under(config.OUTPUT_DIR, path)


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
