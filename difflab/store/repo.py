"""Every query, over a connection somebody else opened.

Nothing here opens a database: a function that did could not be put in a transaction with the one after
it. Everything returns a value rather than a cursor, which is only valid until the next statement on the
same connection.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from difflab import config
from difflab.store.db import gauge_of
from difflab.store.models import (
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    SKIPPED,
    Artifact,
    Command,
    Event,
    Job,
    Run,
    Task,
)

TABLES = ("runs", "jobs", "tasks", "artifacts", "events", "commands")


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


# --- runs -------------------------------------------------------------------------------------
def runs(connection: sqlite3.Connection, limit: int | None = None) -> list[Run]:
    """Newest first by the id itself: it is a timestamp."""
    sql = "SELECT * FROM runs ORDER BY id DESC"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [Run.of(row) for row in connection.execute(sql).fetchall()]


def run(connection: sqlite3.Connection, run_id: str) -> Run | None:
    row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return None if row is None else Run.of(row)


def add_run(
    connection: sqlite3.Connection,
    run_id: str,
    title: str = "",
    spec: dict[str, Any] | None = None,
    asked_by: str = "person",
    scope: str = "",
    production_id: int | None = None,
    state: str = QUEUED,
    at: float | None = None,
) -> Run:
    moment = _now(at)
    connection.execute(
        "INSERT INTO runs (id, production_id, asked_by, scope, title, spec, state, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, production_id, asked_by, scope, title, None if spec is None else _dump(spec), state, moment),
    )
    made = run(connection, run_id)
    assert made is not None
    return made


def start_run(connection: sqlite3.Connection, run_id: str, pid: int | None, at: float | None = None) -> None:
    connection.execute(
        "UPDATE runs SET state = ?, started_at = coalesce(started_at, ?), pid = ? WHERE id = ?",
        (RUNNING, _now(at), pid, run_id),
    )


def set_run_state(connection: sqlite3.Connection, run_id: str, state: str) -> None:
    connection.execute("UPDATE runs SET state = ? WHERE id = ?", (state, run_id))


def end_run(connection: sqlite3.Connection, run_id: str, ok: bool, error: str = "", at: float | None = None) -> None:
    state = DONE if ok else (CANCELLED if error.startswith("cancelled") else FAILED)
    connection.execute(
        "UPDATE runs SET state = ?, error = ?, ended_at = ? WHERE id = ?", (state, error, _now(at), run_id)
    )


def skip_tools(connection: sqlite3.Connection, run_id: str, tools: list[str]) -> None:
    """The run's queued jobs of these tools are handed on as skipped when the worker reaches them."""
    found = run(connection, run_id)
    spec = (found.data() if found is not None else None) or {}
    skipped = list(dict.fromkeys([*spec.get("skip", []), *tools]))
    connection.execute("UPDATE runs SET spec = ? WHERE id = ?", (_dump({**spec, "skip": skipped}), run_id))


def next_run_id(connection: sqlite3.Connection, stamp: str) -> str:
    """`stamp` with a suffix if a run of that second already exists."""
    # Zero-padded: the worker takes runs in id order, and every chapter of a production is queued in one second.
    candidate, suffix = stamp, 1
    while run(connection, candidate) is not None:
        suffix += 1
        candidate = f"{stamp}-{suffix:03d}"
    return candidate


# --- jobs -------------------------------------------------------------------------------------
def add_job(
    connection: sqlite3.Connection,
    run_id: str,
    tool: str,
    inputs: dict[str, Any],
    label: str = "",
    settings: dict[str, Any] | None = None,
    for_kind: str | None = None,
    for_id: int | None = None,
    key: str | None = None,
) -> Job:
    row = connection.execute("SELECT coalesce(max(position), -1) + 1 FROM jobs WHERE run_id = ?", (run_id,)).fetchone()
    cursor = connection.execute(
        "INSERT INTO jobs (run_id, position, tool, label, for_kind, for_id, inputs, settings, key, state) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, int(row[0]), tool, label, for_kind, for_id, _dump(inputs), _dump(settings or {}), key, QUEUED),
    )
    made = job(connection, int(cursor.lastrowid or 0))
    assert made is not None
    return made


def job(connection: sqlite3.Connection, job_id: int) -> Job | None:
    row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return None if row is None else Job.of(row)


def jobs_of(connection: sqlite3.Connection, run_id: str) -> list[Job]:
    rows = connection.execute("SELECT * FROM jobs WHERE run_id = ? ORDER BY position", (run_id,)).fetchall()
    return [Job.of(row) for row in rows]


def last_done_job(connection: sqlite3.Connection, tool: str, key: str) -> Job | None:
    """The newest finished job of this tool under this key: what a job with the same inputs may reuse."""
    row = connection.execute(
        "SELECT * FROM jobs WHERE tool = ? AND key = ? AND state IN (?, ?) ORDER BY id DESC LIMIT 1",
        (tool, key, DONE, SKIPPED),
    ).fetchone()
    return None if row is None else Job.of(row)


def set_job_key(connection: sqlite3.Connection, job_id: int, key: str) -> None:
    connection.execute("UPDATE jobs SET key = ? WHERE id = ?", (key, job_id))


def queued_jobs(connection: sqlite3.Connection) -> list[Job]:
    """Every job still to do, oldest run first, in the order its run listed it. The worker picks the
    first whose dependencies are settled and whose run is not paused."""
    rows = connection.execute(
        "SELECT jobs.* FROM jobs JOIN runs ON jobs.run_id = runs.id "
        "WHERE jobs.state IN (?, ?) AND runs.state IN (?, ?) ORDER BY runs.id, jobs.position",
        (QUEUED, "blocked", QUEUED, RUNNING),
    ).fetchall()
    return [Job.of(row) for row in rows]


def start_job(connection: sqlite3.Connection, job_id: int, at: float | None = None) -> None:
    connection.execute("UPDATE jobs SET state = ?, started_at = ? WHERE id = ?", (RUNNING, _now(at), job_id))


def end_job(
    connection: sqlite3.Connection,
    job_id: int,
    state: str,
    outputs: dict[str, Any] | None = None,
    error: str = "",
    settings: dict[str, Any] | None = None,
    key: str | None = None,
    at: float | None = None,
) -> None:
    connection.execute(
        "UPDATE jobs SET state = ?, outputs = coalesce(?, outputs), error = ?, settings = coalesce(?, settings), "
        "key = coalesce(?, key), ended_at = ? WHERE id = ?",
        (state, None if outputs is None else _dump(outputs), error, None if settings is None else _dump(settings),
         key, _now(at), job_id),
    )  # fmt: skip


def set_job_state(connection: sqlite3.Connection, job_id: int, state: str) -> None:
    connection.execute("UPDATE jobs SET state = ? WHERE id = ?", (state, job_id))


# --- tasks ------------------------------------------------------------------------------------
def tasks_of(connection: sqlite3.Connection, job_id: int) -> list[Task]:
    rows = connection.execute("SELECT * FROM tasks WHERE job_id = ? ORDER BY position", (job_id,)).fetchall()
    return [Task.of(row) for row in rows]


def start_task(
    connection: sqlite3.Connection,
    job_id: int,
    name: str,
    total: int | None = None,
    unit: str | None = None,
    at: float | None = None,
) -> Task:
    row = connection.execute("SELECT coalesce(max(position), -1) + 1 FROM tasks WHERE job_id = ?", (job_id,)).fetchone()
    cursor = connection.execute(
        "INSERT INTO tasks (job_id, position, name, state, done, total, unit, started_at) "
        "VALUES (?, ?, ?, ?, 0, ?, ?, ?)",
        (job_id, int(row[0]), name, RUNNING, total, unit, _now(at)),
    )
    made = task(connection, int(cursor.lastrowid or 0))
    assert made is not None
    return made


def task(connection: sqlite3.Connection, task_id: int) -> Task | None:
    row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return None if row is None else Task.of(row)


def running_task(connection: sqlite3.Connection, job_id: int, name: str) -> Task | None:
    """The newest task of that name still running: stages nest (a load inside sampling), so the one
    an end belongs to is found by name rather than assumed to be the last one started."""
    row = connection.execute(
        "SELECT * FROM tasks WHERE job_id = ? AND name = ? AND state = ? ORDER BY position DESC LIMIT 1",
        (job_id, name, RUNNING),
    ).fetchone()
    return None if row is None else Task.of(row)


def set_task_progress(connection: sqlite3.Connection, task_id: int, done: int, total: int | None) -> None:
    connection.execute("UPDATE tasks SET done = ?, total = coalesce(?, total) WHERE id = ?", (done, total, task_id))


def end_task(connection: sqlite3.Connection, task_id: int, ok: bool, detail: str = "", at: float | None = None) -> None:
    connection.execute(
        "UPDATE tasks SET state = ?, detail = ?, ended_at = ? WHERE id = ?",
        (DONE if ok else FAILED, detail, _now(at), task_id),
    )


# --- artifacts --------------------------------------------------------------------------------
def artifacts(connection: sqlite3.Connection, run_id: str | None = None) -> list[Artifact]:
    if run_id is None:
        rows = connection.execute("SELECT * FROM artifacts ORDER BY created_at DESC, id DESC").fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at DESC, id DESC", (run_id,)
        ).fetchall()
    return [Artifact.of(row) for row in rows]


def artifacts_of_job(connection: sqlite3.Connection, job_id: int) -> list[Artifact]:
    rows = connection.execute("SELECT * FROM artifacts WHERE job_id = ? ORDER BY id", (job_id,)).fetchall()
    return [Artifact.of(row) for row in rows]


def artifact_by_key(connection: sqlite3.Connection, key: str) -> Artifact | None:
    """The newest artifact made under `key` whose file is still there."""
    rows = connection.execute("SELECT * FROM artifacts WHERE key = ? ORDER BY id DESC", (key,)).fetchall()
    return next((made for made in (Artifact.of(row) for row in rows) if made.exists), None)


def artifact_at(connection: sqlite3.Connection, path: str) -> Artifact | None:
    row = connection.execute("SELECT * FROM artifacts WHERE path = ?", (path,)).fetchone()
    return None if row is None else Artifact.of(row)


def add_artifact(
    connection: sqlite3.Connection,
    path: str,
    kind: str,
    run_id: str | None = None,
    job_id: int | None = None,
    key: str | None = None,
    seed: int | None = None,
    at: float | None = None,
) -> Artifact:
    """`path` is relative to `config.OUTPUT_DIR` - see `relative`. One already here is left as it is."""
    connection.execute(
        "INSERT INTO artifacts (run_id, job_id, path, kind, key, seed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(path) DO NOTHING",
        (run_id, job_id, path, kind, key, seed, _now(at)),
    )
    recorded = artifact_at(connection, path)
    assert recorded is not None
    return recorded


def relative(path: Path | str) -> str:
    """Refuses anything outside `config.OUTPUT_DIR` rather than falling back to an absolute path, so
    `output/` stays something that can be moved, copied or mounted elsewhere and still read."""
    resolved, root = Path(path).resolve(), config.OUTPUT_DIR.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{resolved} is not under {root}, and artifacts are recorded relative to it")
    return resolved.relative_to(root).as_posix()


# --- events -----------------------------------------------------------------------------------
def add_event(
    connection: sqlite3.Connection,
    run_id: str,
    t: float,
    kind: str,
    data: dict[str, Any],
    job_id: int | None = None,
    task_id: int | None = None,
) -> None:
    connection.execute(
        "INSERT INTO events (run_id, job_id, task_id, t, kind, gauge, data) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, job_id, task_id, t, kind, gauge_of(kind, data), _dump(data)),
    )


def events_after(connection: sqlite3.Connection, run_id: str, cursor: int) -> list[Event]:
    rows = connection.execute(
        "SELECT * FROM events WHERE run_id = ? AND id > ? ORDER BY id", (run_id, cursor)
    ).fetchall()
    return [Event.of(row) for row in rows]


def catch_up(connection: sqlite3.Connection, run_id: str) -> list[Event]:
    """Every history row plus the last row per gauge, in id order: what a page joining late folds."""
    rows = connection.execute(
        "SELECT * FROM events WHERE run_id = ? AND (gauge IS NULL OR id IN ("
        "  SELECT max(id) FROM events WHERE run_id = ? AND gauge IS NOT NULL GROUP BY gauge)) ORDER BY id",
        (run_id, run_id),
    ).fetchall()
    return [Event.of(row) for row in rows]


def newest_event(connection: sqlite3.Connection, run_id: str) -> float | None:
    row = connection.execute("SELECT max(t) FROM events WHERE run_id = ?", (run_id,)).fetchone()
    return None if row is None or row[0] is None else float(row[0])


def newest_events(connection: sqlite3.Connection, run_ids: list[str]) -> dict[str, float]:
    if not run_ids:
        return {}
    marks = ",".join("?" * len(run_ids))
    rows = connection.execute(
        f"SELECT run_id, max(t) FROM events WHERE run_id IN ({marks}) GROUP BY run_id", run_ids
    ).fetchall()
    return {str(row[0]): float(row[1]) for row in rows}


def named_events(connection: sqlite3.Connection, run_ids: list[str]) -> list[Event]:
    """The `plan` and `panel` events, which are the two kinds that carry a title."""
    if not run_ids:
        return []
    marks = ",".join("?" * len(run_ids))
    rows = connection.execute(
        f"SELECT * FROM events WHERE run_id IN ({marks}) AND kind IN ('plan', 'panel') ORDER BY id", run_ids
    ).fetchall()
    return [Event.of(row) for row in rows]


# --- commands ---------------------------------------------------------------------------------
def add_command(
    connection: sqlite3.Connection, target_kind: str, target_id: str, command: dict[str, Any], at: float | None = None
) -> None:
    connection.execute(
        "INSERT INTO commands (target_kind, target_id, t, data) VALUES (?, ?, ?, ?)",
        (target_kind, target_id, _now(at), _dump({k: v for k, v in command.items() if k != "t"})),
    )


def take_commands(connection: sqlite3.Connection, target_kind: str, target_id: str) -> list[Command]:
    """Handed over exactly once: read and marked in one statement's worth of rows."""
    rows = connection.execute(
        "SELECT * FROM commands WHERE target_kind = ? AND target_id = ? AND taken = 0 ORDER BY id",
        (target_kind, target_id),
    ).fetchall()
    taken = [Command.of(row) for row in rows]
    if taken:
        marks = ",".join("?" * len(taken))
        connection.execute(f"UPDATE commands SET taken = 1 WHERE id IN ({marks})", [c.id for c in taken])
    return taken


# --- the whole shape --------------------------------------------------------------------------
def counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]) for table in TABLES}


def _now(at: float | None) -> float:
    return time.time() if at is None else at
