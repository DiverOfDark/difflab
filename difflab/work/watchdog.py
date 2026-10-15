"""Deletes the RunPod pod this server runs on once nobody needs it. It runs on the pod, not in the
desktop app, because a closed laptop must not leave a GPU billing."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from difflab import config, store

TICK = 30.0
# Container disk: it outlives `/api/restart`'s exec but not the pod.
STARTED = Path("/tmp/difflab-pod-started")
DELETE_URL = "https://rest.runpod.io/v1/pods/{}"

_heard = time.time()


def beat() -> dict[str, Any]:
    global _heard
    _heard = time.time()
    return state()


def state() -> dict[str, Any]:
    return {
        "pod": config.POD_ID,
        "usd": round(_spent(), 2),
        "spend_limit_usd": config.POD_SPEND_LIMIT_USD,
        "idle_minutes": config.POD_IDLE_MINUTES,
        "exit_when_done": config.POD_EXIT_WHEN_DONE,
    }


class Watchdog(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="difflab-watchdog", daemon=True)
        self.busy_since: float | None = None
        self.busy_at = time.time()
        self.worked = False
        self.ending = ""

    def run(self) -> None:
        _started()
        while True:
            time.sleep(TICK)
            try:
                self.ending = self.ending or self._verdict()
            except sqlite3.Error as problem:
                print(f"difflab: the watchdog could not read the queue ({problem})", flush=True)
                continue
            if self.ending:
                _end(self.ending)

    def _verdict(self) -> str:
        now = time.time()
        with store.connect() as connection:
            active = _active_runs(connection)
            progress = _progress(connection, active)
        if config.POD_USD_PER_HOUR and _spent() >= config.POD_SPEND_LIMIT_USD:
            return f"spend limit ${config.POD_SPEND_LIMIT_USD:.2f} reached"
        if active:
            self.busy_since = self.busy_since or now
            self.busy_at = now
            self.worked = True
            if now - max(progress, self.busy_since) > config.POD_STUCK_MINUTES * 60:
                return f"no progress for {config.POD_STUCK_MINUTES:g} min"
            return ""
        self.busy_since = None
        if config.POD_EXIT_WHEN_DONE and self.worked:
            return "queue done"
        if now - max(_heard, self.busy_at) > config.POD_IDLE_MINUTES * 60:
            return f"idle for {config.POD_IDLE_MINUTES:g} min with the app closed"
        return ""


def _active_runs(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT DISTINCT jobs.run_id FROM jobs JOIN runs ON jobs.run_id = runs.id "
        "WHERE jobs.state IN (?, ?, ?) AND runs.state IN (?, ?)",
        (store.QUEUED, store.BLOCKED, store.RUNNING, store.QUEUED, store.RUNNING),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _progress(connection: sqlite3.Connection, run_ids: list[str]) -> float:
    """The newest event other than the worker's 2 s stats, which it writes whether the job moves or not."""
    newest = 0.0
    for run_id in run_ids:
        row = connection.execute("SELECT max(t) FROM events WHERE run_id = ? AND kind != 'stats'", (run_id,)).fetchone()
        newest = max(newest, float(row[0] or 0.0))
    return newest


def _started() -> float:
    if not STARTED.exists():
        STARTED.write_text(str(time.time()))
    return float(STARTED.read_text())


def _spent() -> float:
    return (time.time() - _started()) / 3600 * config.POD_USD_PER_HOUR


def _end(reason: str) -> None:
    usd = round(_spent(), 2)
    print(f"difflab: deleting pod {config.POD_ID}: {reason}, ${usd:.2f}", flush=True)
    config.POD_ENDED.write_text(
        json.dumps({"pod": config.POD_ID, "reason": reason, "usd": usd, "ended_at": time.time()})
    )
    request = urllib.request.Request(
        DELETE_URL.format(config.POD_ID),
        method="DELETE",
        headers={"Authorization": f"Bearer {config.POD_API_KEY}"},
    )
    try:
        urllib.request.urlopen(request, timeout=30).close()
    except (urllib.error.URLError, OSError) as problem:
        print(f"difflab: the pod could not delete itself ({problem}); trying again", flush=True)
