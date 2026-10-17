"""What the worker deletes between jobs: idle productions, blobs nobody has used lately, quick runs past their age,
old backups.

A blob a production's made thing points at goes like any other; the thing drops back to `missing`
with its seed kept, so rendering it again lands where it was. What a person gave is never a blob
this deletes: nothing could make it again.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import time
from collections.abc import Sized
from dataclasses import dataclass, field
from pathlib import Path

from difflab import config, store
from difflab.core import events
from difflab.store.screenplay import MADE_TABLES

BLOBS = config.OUTPUT_DIR / "blobs"
_BLOB_PATH = re.compile(r"blobs/[\w./-]+")
_TOKEN = re.compile(r"^[0-9a-f]{20}")
# Columns holding what a person gave, or what a run not yet settled will read.
_GIVEN = (
    "SELECT provided FROM keyframes",
    "SELECT photo FROM refs",
    "SELECT provided FROM refs",
    "SELECT manifest FROM productions",
    "SELECT reference_audio FROM members",
    "SELECT j.inputs || j.outputs FROM jobs j JOIN runs r ON r.id = j.run_id WHERE r.state NOT IN (?, ?, ?, ?)",
)


@dataclass
class Swept:
    productions: list[str] = field(default_factory=list)
    files: int = 0
    bytes: int = 0
    runs: int = 0
    backups: int = 0
    reset: dict[str, int] = field(default_factory=dict)

    def line(self) -> str:
        reset = ", ".join(f"{n} {kind}s" for kind, n in self.reset.items() if n) or "nothing"
        gone = ", ".join(self.productions) or "none"
        return (
            f"cleanup: productions {gone}; {self.bytes / 1024**3:.1f} G in {self.files} blobs, {self.runs} runs, "
            f"{self.backups} backups; back to missing: {reset}"
        )


def sweep(dry: bool = False) -> Swept:
    """Only from the worker's thread with nothing claimed: a job started mid-sweep could read a blob as it goes."""
    swept = Swept()
    with store.connect() as connection:
        _productions(connection, swept, dry)
        kept = _given(connection)
        gone = _blobs(kept, swept, dry)
        _amend(connection, gone, swept, dry)
        _runs(connection, swept, dry)
    _backups(swept, dry)
    return swept


def _productions(connection: sqlite3.Connection, swept: Swept, dry: bool) -> None:
    """Idle since its last edit or run, whichever is later. Its runs go later with the orphans, its blobs
    with their age; what it gave is no longer kept by it."""
    rows = connection.execute(
        "SELECT p.id, p.slug, p.keep, max(p.updated_at, coalesce(max(coalesce(r.ended_at, r.created_at)), 0)) AS at, "
        f"count(CASE WHEN r.state NOT IN ({_marks(store.SETTLED)}) THEN 1 END) AS live "
        "FROM productions p LEFT JOIN runs r ON r.production_id = p.id GROUP BY p.id ORDER BY at DESC",
        store.SETTLED,
    ).fetchall()
    cutoff = time.time() - config.PRODUCTIONS_IDLE_DAYS * 86400
    old = [row for row in rows[config.PRODUCTIONS_KEEP :] if not row["keep"] and not row["live"] and row["at"] < cutoff]
    swept.productions = [row["slug"] for row in old]
    if old and not dry:
        with store.transaction(connection):
            connection.execute(f"DELETE FROM productions WHERE id IN ({_marks(old)})", [row["id"] for row in old])


def _given(connection: sqlite3.Connection) -> set[str]:
    kept: set[str] = set()
    for query in _GIVEN:
        rows = connection.execute(query, store.SETTLED if "?" in query else ())
        for (value,) in rows:
            kept.update(_BLOB_PATH.findall(value or ""))
    return kept


def _blobs(kept: set[str], swept: Swept, dry: bool) -> set[str]:
    """A token's files go together: a cache entry is its `.png` and `.tail.pt` both, or neither."""
    cutoff = time.time() - config.BLOB_KEEP_DAYS * 86400
    groups: dict[tuple[Path, str], list[Path]] = {}
    for folder in BLOBS.iterdir() if BLOBS.exists() else ():
        if not folder.is_dir() or folder.name in config.BLOB_KEEP_TOOLS:
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                token = _TOKEN.match(path.name)
                groups.setdefault((path.parent, token.group() if token else path.name), []).append(path)
    gone: set[str] = set()
    for files in groups.values():
        names = [path.relative_to(config.OUTPUT_DIR).as_posix() for path in files]
        stats = [path.stat() for path in files]
        if any(name in kept for name in names) or any(stat.st_mtime > cutoff for stat in stats):
            continue
        swept.files += len(files)
        swept.bytes += sum(stat.st_size for stat in stats)
        gone.update(names)
        if not dry:
            for path in files:
                path.unlink(missing_ok=True)
    return gone


def _amend(connection: sqlite3.Connection, gone: set[str], swept: Swept, dry: bool) -> None:
    """A chained keyframe is the clip before it's handoff frame, so that clip goes back too: rendering it
    again is what chains the frame forward."""
    with store.transaction(connection):
        for kind, table in MADE_TABLES.items():
            rows = connection.execute(f"SELECT id, artifact FROM {table} WHERE artifact LIKE 'blobs/%'").fetchall()
            lost = [row["id"] for row in rows if row["artifact"] in gone]
            if kind == "keyframe" and lost:
                lost_clips = connection.execute(
                    "SELECT c.id FROM keyframes k JOIN shots s ON s.id = k.shot_id "
                    "JOIN shots p ON p.chapter_id = s.chapter_id AND p.position = s.position - 1 "
                    f"JOIN clips c ON c.shot_id = p.id WHERE k.source = ? AND k.id IN ({_marks(lost)})",
                    (store.CHAINED, *lost),
                ).fetchall()
                _reset(connection, "clips", [row["id"] for row in lost_clips], dry)
                swept.reset["clip"] = swept.reset.get("clip", 0) + len(lost_clips)
            _reset(connection, table, lost, dry)
            swept.reset[kind] = swept.reset.get(kind, 0) + len(lost)
        # A prompt hands its pictures to the render by path; written again, it names them where they are now.
        stale = [
            row["id"]
            for row in connection.execute("SELECT id, pictures FROM prompts WHERE pictures LIKE '%blobs/%'")
            if gone.intersection(_BLOB_PATH.findall(row["pictures"]))
        ]
        if stale and not dry:
            connection.execute(
                f"UPDATE prompts SET state = ? WHERE id IN ({_marks(stale)}) AND state NOT IN (?, ?)",
                (store.STALE, *stale, store.QUEUED, store.RUNNING),
            )
        swept.reset["stale prompt"] = len(stale)


def _reset(connection: sqlite3.Connection, table: str, ids: list[int], dry: bool) -> None:
    if ids and not dry:
        connection.execute(
            f"UPDATE {table} SET state = ?, artifact = NULL WHERE id IN ({_marks(ids)})", (store.MISSING, *ids)
        )


def _runs(connection: sqlite3.Connection, swept: Swept, dry: bool) -> None:
    """Quick runs, and runs of a production since deleted. A production's own runs hold its takes'
    history and the timings its ETA is measured from. The artifact rows outlive them (`ON DELETE SET
    NULL`), so the gallery keeps every image and video."""
    cutoff = time.time() - config.RUNS_KEEP_DAYS * 86400
    old = [
        row["id"]
        for row in connection.execute(
            f"SELECT id FROM runs WHERE state IN ({_marks(store.SETTLED)}) AND created_at < ? "
            "AND (production_id IS NULL OR production_id NOT IN (SELECT id FROM productions))",
            (*store.SETTLED, cutoff),
        )
    ]
    swept.runs = len(old)
    if dry or not old:
        return
    with store.transaction(connection):
        connection.execute(f"DELETE FROM runs WHERE id IN ({_marks(old)})", old)
    for run_id in old:
        shutil.rmtree(config.RUNS_DIR / run_id, ignore_errors=True)
    events.prune(config.RUNS_DIR, config.RUNS_KEEP)


def _backups(swept: Swept, dry: bool) -> None:
    for found in (
        (config.OUTPUT_DIR / "backups").glob("*.sqlite"),
        config.OUTPUT_DIR.glob(f"{store.DB_PATH.name}.*.bak"),
    ):
        old = sorted(found, key=lambda path: path.stat().st_mtime)[: -config.BACKUPS_KEEP]
        swept.backups += len(old)
        if not dry:
            for path in old:
                path.unlink(missing_ok=True)


def _marks(values: Sized) -> str:
    return ", ".join("?" * len(values))


_MADE_PATH = re.compile(r"(?:blobs|images|videos)/[\w./-]+\.\w+")
# Everything else that names a file under output/ and would break without it.
_USED = (
    *_GIVEN[:-1],
    "SELECT pictures FROM prompts",
    *(f"SELECT artifact FROM {table}" for table in MADE_TABLES.values()),
    "SELECT j.inputs || j.outputs FROM jobs j WHERE j.run_id != ?",
    "SELECT path FROM artifacts WHERE run_id IS NULL OR run_id != ?",
)


@dataclass
class Deleted:
    files: int = 0
    bytes: int = 0
    kept: int = 0  # made by the run, but another run or a production still names them


def delete_run(run_id: str) -> Deleted:
    """A settled run and everything it left: its rows, its folder under runs/, and the files it made that nothing
    else names. What it was given stays, and so does the cache an identical job would read again."""
    deleted = Deleted()
    with store.connect() as connection:
        made = {artifact.path for artifact in store.artifacts(connection, run_id)}
        for row in connection.execute("SELECT outputs FROM jobs WHERE run_id = ?", (run_id,)):
            made.update(_MADE_PATH.findall(row["outputs"] or ""))
        used: set[str] = set()
        for query in _USED:
            for (value,) in connection.execute(query, (run_id,) if "?" in query else ()):
                used.update(_MADE_PATH.findall(value or ""))
        doomed = sorted(made - used)
        deleted.kept = len(made) - len(doomed)
        with store.transaction(connection):
            connection.execute("DELETE FROM commands WHERE target_id = ?", (run_id,))
            connection.execute(f"DELETE FROM artifacts WHERE path IN ({_marks(doomed)})", doomed)
            connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    for name in doomed:
        path = store.under_output(name)
        if path.is_file():
            deleted.files += 1
            deleted.bytes += path.stat().st_size
            path.unlink()
    shutil.rmtree(config.RUNS_DIR / run_id, ignore_errors=True)
    return deleted
