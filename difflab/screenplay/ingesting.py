"""The ingesting page: a production's plan runs folded into the four steps it shows, what the chapters
found, and a log; and the page's Skip and Cancel."""

from __future__ import annotations

import time
from typing import Any

from difflab import store
from difflab.core import protocol
from difflab.core.settings import DirectorSettings, IngestSettings
from difflab.screenplay import ingest, planner, settings, timings
from difflab.work.plan_runners import SKIPS

ORDER = ("split", "describe", "read", "reconcile")
NAMES = {"split": "Split", "describe": "Describe references", "read": "Read chapters", "reconcile": "Reconcile"}
TAGS = {"describe": "DESCRIBE", "read": "READ", "reconcile": "RECONCILE"}
BATCH = 10.0  # s: the plan runs of one "plan every chapter" are queued in one transaction, well inside this
KIND = {"person": "CHARACTER", "animal": "CHARACTER", "environment": "LOCATION", "object": "PROP", "style": "STYLE"}


def _batch(connection: Any, production_id: int) -> list[store.Run]:
    """The newest plan of the production: its runs, in the order the worker takes them."""
    plans = [r for r in store.runs(connection) if r.production_id == production_id and r.scope.startswith("plan:")]
    if not plans:
        return []
    newest = max(r.created_at for r in plans)
    return sorted((r for r in plans if r.created_at >= newest - BATCH), key=lambda r: r.id)


def _step(tool: str) -> str:
    return next((step for step, tools in planner.STEPS.items() if tool in tools), "plan")


def status(production_id: int) -> dict[str, Any] | None:
    with store.connect() as connection:
        production = store.production(connection, production_id)
        if production is None:
            return None
        runs = _batch(connection, production_id)
        jobs = {run.id: store.jobs_of(connection, run.id) for run in runs}
        members = store.members_of(connection, production_id, style=True)
        chapters = store.chapters_of(connection, production_id)
        events = [
            event
            for run in runs
            for event in store.events_after(connection, run.id, 0)
            if event.kind in ("job.end", "found", "stats")
        ]
        conflicts = {m.slug: store.conflicts_of(connection, m.id) for m in members}
        measured = timings.measured(connection)
    first = jobs[runs[0].id] if runs else []
    # The steps live in the first run; a later chapter's run reuses them by key, and only its shots are its own.
    steps = {step: [j for j in first if j.tool in tools] for step, tools in planner.STEPS.items()}
    every = [j for listed in jobs.values() for j in listed]
    live = [r for r in runs if r.state in (store.QUEUED, store.RUNNING, store.PAUSED)]
    running = next((j for j in every if j.state == store.RUNNING), None)
    reading = {s.get("chapter_position", 0): j for j in steps["read"] if (s := j.inputs) is not None}
    by_job = {j.id: j for j in every}
    names = planner._named(production.manifest)

    def state(listed: list[store.Job]) -> str:
        if any(j.state == store.RUNNING for j in listed):
            return "running"
        if any(j.state in (store.FAILED, store.CANCELLED) for j in listed):
            return "failed"
        if listed and all(j.settled for j in listed):
            return "done"
        return "pending" if not any(j.settled for j in listed) else "waiting"

    def bar(listed: list[store.Job]) -> float:
        return round(sum(j.settled for j in listed) / len(listed), 3) if listed else 1.0

    read = [j for j in steps["read"] if j.settled]
    last = read[-1].outputs if read else {}
    shows = dict(zip(production.manifest.get("photos", []), last.get("shows", []), strict=False))
    merged = {production.manifest.get("photos", [])[int(m["photo"])]: m["entity"] for m in last.get("merged", [])}
    guessed = {str(j.outputs.get("path")): j for j in steps["describe"] if j.tool == "guess_reference"}

    def described(row: dict[str, Any]) -> tuple[str, str]:
        path = str(row["path"])
        if row.get("kind") == "style":
            return names(path), "style"
        if row.get("kind") == "video":
            frames = row.get("frames", [])
            return names(path), f"{sum(f['kept'] for f in frames)} / {len(frames)} frames sharp"
        if path in merged:
            return names(path), f"merged → {merged[path]}"
        if shows.get(path):
            return names(path), ", ".join(shows[path])
        guess = guessed.get(path)
        return names(path), (str(guess.outputs.get("about") or "…") if guess and guess.settled else "…")

    split = _split(production, chapters)
    rows = {
        "split": split,
        "describe": [described(r) for r in production.manifest.get("references", [])],
        "read": [
            (
                f"ch {c.position + 1:02d}{(' ' + c.title) if c.title else ''}",
                (
                    "skipped"
                    if reading[c.position].state == store.SKIPPED and not reading[c.position].key
                    else ", ".join(str(m["id"]) for m in reading[c.position].outputs.get("new", [])) or "no one new"
                )
                if c.position in reading and reading[c.position].settled
                else "…",
            )
            for c in chapters
        ],
        "reconcile": [
            *[(f"{slug} · {c.novel_says[:40]}", f"{len(c.lines)} lines") for slug, cs in conflicts.items() for c in cs],
            ("voices", f"{sum(1 for m in members if m.voice and m.kind in ('person', 'animal'))} / "
                       f"{sum(1 for m in members if m.kind in ('person', 'animal'))}"),
        ],
    }  # fmt: skip
    listed_steps = []
    for step in ORDER:
        own = steps.get(step, [])
        skippable = any(j.tool in SKIPS and j.state == store.QUEUED for j in own)
        listed_steps.append(
            {
                "id": step, "name": NAMES[step], "state": "done" if step == "split" else state(own),
                "bar": 1.0 if step == "split" else bar(own), "rows": [{"k": k, "v": v} for k, v in rows[step]],
                "jobs": len(own), "skippable": skippable,
            }
        )  # fmt: skip
    planned = [j for j in every if _step(j.tool) == "plan"]
    started = min((r.started_at or r.created_at for r in runs), default=time.time())
    ended = max((r.ended_at or 0 for r in runs), default=0) if not live else 0
    eta = sum(
        float(measured.get(j.tool, {}).get("seconds") or 0)
        - (time.time() - (j.started_at or time.time()) if j.state == store.RUNNING else 0)
        for j in every
        if not j.settled and not (j.tool in SKIPS and _skipping(runs, j))
    )
    vram = next((e.value().get("vram_used") for e in reversed(events) if e.kind == "stats"), None)
    four = [j for step in ("describe", "read", "reconcile") for j in steps[step]]
    return {
        "runs": [{"id": r.id, "state": r.state, "title": r.title} for r in runs],
        "live": [r.id for r in live],
        "steps": listed_steps,
        "current": _step(running.tool)
        if running
        else next((s["id"] for s in listed_steps if s["state"] != "done"), "plan"),
        "progress": bar(four),
        "bible_done": bool(four) and all(j.settled for j in four),
        "plan": {"jobs": len(planned), "settled": sum(j.settled for j in planned)},
        "elapsed": round((ended or time.time()) - started, 1) if runs else 0,
        "eta": round(max(0.0, eta), 1),
        "model": _model(production_id),
        "vram": vram,
        "found": _found(events, members, production.manifest),
        "log": _log(events, by_job, names),
    }


def _skipping(runs: list[store.Run], job: store.Job) -> bool:
    run = next((r for r in runs if r.id == job.run_id), None)
    return job.tool in ((run.data() if run is not None else None) or {}).get("skip", [])


def _split(production: store.Production, chapters: list[store.Chapter]) -> list[tuple[str, str]]:
    words = sum(len(c.story.split()) for c in chapters)
    out = [(f"{len(chapters)} chapter{'s' if len(chapters) != 1 else ''}", f"{words:,} words")]
    if production.story_kind == "text":
        chosen, _ = settings.effective("ingest", production.id)
        assert isinstance(chosen, IngestSettings)
        parts = ingest.chapters_of(production.story, chosen.heading_level)
        out += [
            (f"ch {n:02d} · no heading", f"split at {ingest.WORDS_PER_CHAPTER} w")
            for n, part in enumerate(parts, 1)
            if not part.headed
        ]
        out.insert(
            1,
            ("on", f"h{chosen.heading_level} headings" if any(p.title and p.headed for p in parts) else "no headings"),
        )
    return out


def _model(production_id: int) -> str:
    chosen, _ = settings.effective("survey", production_id)
    assert isinstance(chosen, DirectorSettings)
    return chosen.checkpoint.split("@", 1)[0].rsplit("/", 1)[-1]


def _found(events: list[store.Event], members: list[store.Member], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Each member as a chapter's survey announced it, with what the bible says of it by now."""
    rows = {m.slug: m for m in members}
    videos = {str(r.get("entity")) for r in manifest.get("references", []) if r.get("kind") == "video"}
    out: dict[str, dict[str, Any]] = {}
    for m in members:
        if m.kind == "style":
            out[m.slug] = {"slug": m.slug, "kind": "STYLE", "desc": m.appearance, "src": "PROVIDED", "chapter": -1}
    for event in events:
        if event.kind != "found":
            continue
        data = event.value()
        for m in data.get("members", []):
            slug = str(m.get("id"))
            if slug in out:
                continue
            photos = int(m.get("photos") or 0)
            member = rows.get(slug)
            src = (f"PROVIDED · {'video' if slug in videos else photos}") if photos else "TO GENERATE"
            out[slug] = {
                "slug": slug, "kind": KIND.get(str(m.get("kind")), str(m.get("kind")).upper()),
                "desc": (member.appearance if member is not None and member.appearance else str(m.get("find") or "")),
                "noun": str(m.get("noun") or ""), "src": src, "chapter": int(data.get("chapter") or 0),
            }  # fmt: skip
    return list(out.values())


def _log(events: list[store.Event], jobs: dict[int, store.Job], names: Any) -> list[dict[str, Any]]:
    out = []
    for event in events:
        data = event.value()
        if event.kind == "found":
            said = ", ".join(str(m.get("id")) for m in data.get("members", []))
            out.append(
                {"t": event.t, "tag": "READ", "msg": f"ch {int(data.get('chapter') or 0) + 1:02d} → {said}", "dur": ""}
            )
            continue
        job = jobs.get(event.job_id or -1)
        if event.kind != "job.end" or job is None:
            continue
        status = str(data.get("status") or "")
        message = job.label + (f" → {said}" if (said := _said(job, names)) else "")
        if status == store.SKIPPED and not job.key:
            message = f"{job.label} · skipped"
        elif status not in (store.DONE, store.SKIPPED):
            message = f"{job.label} · {status} {str(data.get('error') or '')[:120]}".rstrip()
        step = _step(job.tool)
        out.append(
            {
                "t": event.t,
                "tag": TAGS.get(step, "PLAN"),
                "msg": message,
                "dur": f"{float(data.get('seconds') or 0):.0f}s",
            }
        )
    return out[-60:]


def _said(job: store.Job, names: Any) -> str:
    """What a finished job came to, in a few words."""
    got = job.outputs
    if job.tool == "guess_reference":
        return f"{names(str(got.get('path', '')))}: {got.get('about', '')}" if got.get("about") else ""
    if job.tool == "same_person":
        ruled = got.get("ruled") or {}
        return (
            ""
            if ruled.get("same") is None
            else ("same" if ruled["same"] else "not the same") + f" · {ruled.get('why', '')}"
        )
    if job.tool == "survey":
        return f"{len(got.get('members', []))} members"
    if job.tool == "cut_crops":
        return f"{len(got.get('crops', []))} references"
    if job.tool == "plan_shots":
        return f"{len(got.get('chunks', []))} shots"
    if job.tool == "tidy":
        return f"{len(got.get('cards', {}))} cards"
    return ""


def skip(production_id: int, step: str) -> int:
    """The step's remaining jobs that can be handed on, skipped in every run of the plan; how many."""
    if step not in planner.STEPS:
        raise ValueError(f"a step is one of {', '.join(planner.STEPS)}")
    tools = [t for t in planner.STEPS[step] if t in SKIPS]
    if not tools:
        raise ValueError(f"{NAMES[step]} cannot be skipped: every later stage reads all of it")
    with store.connect() as connection, store.transaction(connection):
        count = 0
        for run in _batch(connection, production_id):
            if run.state not in (store.QUEUED, store.RUNNING, store.PAUSED):
                continue
            store.skip_tools(connection, run.id, tools)
            count += sum(1 for j in store.jobs_of(connection, run.id) if j.tool in tools and j.state == store.QUEUED)
    return count


def cancel(production_id: int) -> int:
    """Every live run of the plan cancelled; how many."""
    with store.connect() as connection, store.transaction(connection):
        live = [r for r in _batch(connection, production_id) if r.state in (store.QUEUED, store.RUNNING, store.PAUSED)]
        for run in live:
            store.add_command(connection, "run", run.id, protocol.Cancel().data())
    return len(live)
