"""Runs as data, for the process that serves the page. Standard library and `difflab.store`, no model.

Liveness is derived here and nowhere else. A run row says `running` until something writes an end,
and the renders that need reporting most write none - a worker killed by the OOM reaper records
nothing - so a row whose newest event has gone quiet reads `stopped` however the row itself reads.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from difflab import store
from difflab.core import protocol

STALE_AFTER = 15.0  # seconds without an event (the worker's heartbeat is 2 s) before a run counts as gone
# What a page may ask of the run itself, and what it asks of the job in flight.
ON_RUN = (protocol.Pause, protocol.Resume, protocol.Cancel)


def runs() -> list[dict[str, Any]]:
    """Every run, newest first, with the state the page should show."""
    with store.connect() as connection:
        listed = store.runs(connection)
        ids = [record.id for record in listed]
        heard = store.newest_events(connection, ids)
        titled = _named(store.named_events(connection, ids))
        made = {record.id: store.jobs_of(connection, record.id) for record in listed}
    return [
        _summary(record, heard.get(record.id), titled.get(record.id, ("", 0)), made[record.id]) for record in listed
    ]


def one(run_id: str) -> dict[str, Any] | None:
    with store.connect() as connection:
        record = store.run(connection, run_id)
        if record is None:
            return None
        last = store.newest_event(connection, run_id)
        titled = _named(store.named_events(connection, [run_id]))
        jobs = store.jobs_of(connection, run_id)
        tasks = {job.id: store.tasks_of(connection, job.id) for job in jobs}
        artifacts = store.artifacts(connection, run_id)
    return {
        **_summary(record, last, titled.get(run_id, ("", 0)), jobs),
        "spec": record.data(),
        "jobs": [_job(job, tasks[job.id]) for job in jobs],
        "artifacts": [{"path": a.path, "kind": a.kind, "seed": a.seed, "job": a.job_id} for a in artifacts],
    }


def gone(run_id: str) -> bool:
    """True once a run has stopped saying anything for good - it ended, or it died without saying so."""
    summary = one(run_id)
    return summary is None or summary["status"] not in ("live", "queued")


def control(run_id: str, command: dict[str, Any]) -> str:
    """Leave a command for the worker. Returns a problem, or "" when it was written.

    Pause, resume and cancel are the run's; the rest are for the job in flight, which reads them where
    it can act - a chunk boundary or the next sampling step - so nothing here happens at once.
    """
    try:
        parsed = protocol.command(command)
    except ValueError as problem:
        return str(problem)
    summary = one(run_id)
    if summary is None:
        return "no such run"
    # A paused run is waiting, not over: a judge pauses one on a failing clip, and resume is how it goes on.
    if summary["status"] not in ("live", "queued", "paused"):
        return "that run is not running any more"
    if summary["status"] == "paused" and not isinstance(parsed, ON_RUN):
        return "no job of that run is in flight to take it"
    if summary["production"] is not None and not isinstance(parsed, ON_RUN):
        return "a production's shots are re-rolled and edited on the production, not on its run"
    try:
        with store.connect() as connection:
            if isinstance(parsed, ON_RUN):
                store.add_command(connection, "run", run_id, parsed.data())
            else:
                running = next((job for job in summary["jobs"] if job["state"] == store.RUNNING), None)
                if running is None:
                    return "no job of that run is in flight to take it"
                store.add_command(connection, "job", str(running["id"]), parsed.data())
    except sqlite3.Error as failure:
        return f"the command could not be written ({failure})"
    return ""


# --- the event stream ----------------------------------------------------------------------------
def exists(run_id: str) -> bool:
    with store.connect() as connection:
        return store.run(connection, run_id) is not None


def catch_up(run_id: str) -> list[store.Event]:
    with store.connect() as connection:
        return store.catch_up(connection, run_id)


def since(run_id: str, cursor: int) -> list[store.Event]:
    with store.connect() as connection:
        return store.events_after(connection, run_id, cursor)


# --- what a listing needs of a run ----------------------------------------------------------------
def _summary(record: store.Run, last: float | None, titled: tuple[str, int], jobs: list[store.Job]) -> dict[str, Any]:
    title, items = titled
    return {
        "id": record.id,
        "title": title or record.title or record.id,
        "status": _status(record, last),
        "state": record.state,
        "items": items,
        "error": record.error,
        "asked_by": record.asked_by,
        "production": record.production_id,
        "scope": record.scope,
        "created": record.created_at,
        "started": record.started_at,
        "ended": record.ended_at,
        "last": last,
        "jobs_done": sum(1 for job in jobs if job.state in (store.DONE, store.SKIPPED)),
        "jobs_total": len(jobs),
    }


def _status(record: store.Run, last: float | None) -> str:
    """The row says what was recorded; only the newest event says whether it is still happening."""
    if record.settled:
        return record.state
    if record.state in (store.QUEUED, store.PAUSED):
        return record.state
    heard = last if last is not None else record.started_at
    if heard is None:
        return "queued"
    return "live" if time.time() - heard < STALE_AFTER else "stopped"


def _job(job: store.Job, tasks: list[store.Task]) -> dict[str, Any]:
    return {
        "id": job.id,
        "position": job.position,
        "tool": job.tool,
        "label": job.label,
        "state": job.state,
        "inputs": job.inputs,
        "outputs": job.outputs,
        "settings": job.settings,
        "started": job.started_at,
        "ended": job.ended_at,
        "error": job.error,
        "tasks": [
            {
                "name": t.name,
                "state": t.state,
                "done": t.done,
                "total": t.total,
                "unit": t.unit,
                "detail": t.detail,
                "seconds": None if t.started_at is None or t.ended_at is None else round(t.ended_at - t.started_at, 1),
            }
            for t in tasks
        ],
    }


def _named(events: list[store.Event]) -> dict[str, tuple[str, int]]:
    """What a run is called and how many chunks it planned: a `plan` always wins and is also the count,
    a `panel` speaks only for a run that has not planned yet."""
    found: dict[str, tuple[str, int]] = {}
    for event in events:
        data = event.value()
        title = data.get("title")
        if not isinstance(title, str):
            continue
        was, items = found.get(event.run_id, ("", 0))
        if event.kind == "plan":
            found[event.run_id] = (title, len(data.get("items") or []))
        elif not was:
            found[event.run_id] = (title, items)
    return found
