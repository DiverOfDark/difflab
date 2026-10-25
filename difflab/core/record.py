"""Every row a render writes, and the only place a database error is allowed to be nothing.

A render must never be *failed* by its record: a take is two hours of one card and what is written
about it is not the work, so a `sqlite3.Error` here goes to stderr and no further. What is
deliberately not swallowed is `migrate`'s refusal to open a newer schema, which fires on the first call
the process makes.

A connection per call rather than one held open: the busiest caller is `event`, throttled by `core/ui`
to about 1.4 a second, against the ~200 us it costs to open one. Measured: 2,521 events over half an
hour, so a ~620 us WAL write is 0.09% of a render's wall time.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from difflab import store
from difflab.core import context, protocol, provenance


def run_started(run_id: str, at: float, title: str = "", pid: int | None = None) -> None:
    """The row for a run, whether the queue made it or a playground call is opening it now."""
    ctx = context.current()
    ctx.run_id, ctx.job_id, ctx.task_id = run_id, None, None

    def start(c: sqlite3.Connection) -> None:
        if store.run(c, run_id) is None:
            store.add_run(c, run_id, title=title, asked_by="playground", at=at)
        store.start_run(c, run_id, pid, at)

    _write("the run", start)


def run_ended(ok: bool, error: str = "") -> None:
    run_id = context.current().run_id
    if run_id is not None:
        _write("the end of the run", lambda c: store.end_run(c, run_id, ok, error))


def artifact_made(path: Path, kind: str, seed: int | None = None, key: str | None = None) -> None:
    """A file a job delivered, attributed to the job that was running when it appeared."""
    if kind == store.IMAGE:
        provenance.mark_png(path)
    try:
        where = store.relative(path)
    except ValueError:
        return  # written outside output/, which the server does not serve and this does not record
    ctx = context.current()
    run_id, job_id = ctx.run_id, ctx.job_id
    _write(
        "an artifact", lambda c: store.add_artifact(c, where, kind, run_id=run_id, job_id=job_id, key=key, seed=seed)
    )


def event(kind: str, t: float, data: dict[str, Any]) -> None:
    """Every `core.events.Run.emit`. A process that never opened a run records nothing rather than failing."""
    ctx = context.current()
    run_id, job_id, task_id = ctx.run_id, ctx.job_id, ctx.task_id
    if run_id is not None:
        _write("an event", lambda c: store.add_event(c, run_id, t, kind, data, job_id=job_id, task_id=task_id))


def stage(event: protocol.Event) -> None:
    """A job's phases as task rows, read off its stage events: the only thing with a bar."""
    ctx = context.current()
    job_id = ctx.job_id
    if job_id is None:
        return
    if isinstance(event, protocol.StageStart):
        started = _write("the task", lambda c: store.start_task(c, job_id, event.label, event.total, event.unit))
        ctx.task_id = None if started is None else started.id
    elif isinstance(event, protocol.StageProgress):
        done, total = event.done, event.total

        def advance(c: sqlite3.Connection) -> None:
            task = store.running_task(c, job_id, event.label)
            if task is not None:
                store.set_task_progress(c, task.id, done, total)

        _write("the task's progress", advance)
    elif isinstance(event, protocol.StageEnd):

        def end(c: sqlite3.Connection) -> int | None:
            task = store.running_task(c, job_id, event.label)
            if task is None:
                return None
            store.end_task(c, task.id, event.ok, event.detail or "")
            return task.id

        ended = _write("the end of the task", end)
        if ended is not None and ended == ctx.task_id:
            ctx.task_id = None


# `OSError` as well as `sqlite3.Error`: a DB_PATH whose parent is not a directory fails in `connect`'s
# own mkdir, before sqlite3 is reached, and a render must not die of where its database is not.
def _write[T](what: str, action: Callable[[sqlite3.Connection], T]) -> T | None:
    try:
        with store.connect() as connection:
            return action(connection)
    except (sqlite3.Error, OSError) as problem:
        print(f"difflab: {what} was not recorded ({problem})", file=sys.stderr, flush=True)
        return None


def _read[T](what: str, action: Callable[[sqlite3.Connection], T], missing: T) -> T:
    try:
        with store.connect() as connection:
            return action(connection)
    except (sqlite3.Error, OSError) as problem:
        print(f"difflab: {what} could not be read ({problem})", file=sys.stderr, flush=True)
        return missing
