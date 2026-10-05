"""A run as data: structured events, for anything that watches a render other than the terminal.

Every `ui` call also emits an event here, recorded through `core.record` into `difflab.store`. Those
rows are the whole interface: a watcher needs no hook into the render beyond opening the same
database, so it can live in this process or another one. Anything too big for a row - preview clips,
finished chunks, thumbnails - is still written into the run directory and named by its path relative
to it (`path`, `rel`), which is what `GET /runs/<id>/<path>` serves. Control flows back the same way,
as `commands` rows the worker takes.

What moving the index out of the directory costs, and it is a real cost: `tail -f` and `cat` on a
dead render are gone. `sqlite3 output/factory.db "select * from events where run_id = '<id>'"` is
what replaces them.

A run is one process: it begins with the first event, and ends when the process does.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import shutil
import signal
import sqlite3
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from difflab import config, store
from difflab.core import context, protocol, record

STATS_EVERY = 2.0  # seconds between resource samples; also the liveness heartbeat the dashboard watches


# Progress and failures go to stdout as well as the store, whether or not a page is watching: a
# render driven from a terminal or a container log otherwise shows nothing between the dashboard
# address and the end. Its own handler rather than the root's, so the silencing in `difflab/__init__`
# and any library's basicConfig leave it alone.
class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return f"{record.levelname + ':':<9}{super().format(record)}"


log = logging.getLogger("difflab")
if not log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    # uvicorn's default `levelprefix` shape, so the dashboard's request lines and these read as one log.
    _handler.setFormatter(_Formatter("%(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _log(kind: str, data: dict[str, Any]) -> None:
    if kind == "stage.progress":
        total = data.get("total")
        count = f"{data.get('done')}/{total}" if total else str(data.get("done"))
        log.info("%s: %s %s (%ss)", data.get("label"), count, data.get("unit", "step"), data.get("elapsed"))
    elif kind == "error":
        log.error("%s", data.get("text", "").rstrip())
    elif kind.endswith(".end") and not data.get("ok", True):
        what = data.get("label") or (f"job {data.get('index')}/{data.get('total')}" if kind == "job.end" else kind)
        log.error("%s failed: %s", what, data.get("error") or data.get("detail") or "")


class Run:
    """The sink a context records into: a directory for what is too big for a row, and the rows.

    `run_id` names an existing row when the queue made the run; a playground call passes none and
    gets a fresh, unique id, and the row is opened here because this is the only line every render
    passes through.
    """

    def __init__(self, root: Path, run_id: str | None = None, title: str | None = None) -> None:
        self.started = time.time()
        if run_id is None:
            stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            run_id, suffix = stamp, 1
            while (root / run_id).exists():
                suffix += 1
                run_id = f"{stamp}-{suffix}"
        self.id = run_id
        self.dir = root / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.title = title or Path(sys.argv[0]).name
        self.ended = False
        self._closed = False
        record.run_started(self.id, at=self.started, title=self.title, pid=os.getpid())

    def emit(self, event: protocol.Event) -> None:
        if self._closed:
            return
        data = event.data()
        _log(event.KIND, data)
        record.stage(event)
        record.event(event.KIND, round(time.time(), 3), data)

    def path(self, name: str) -> Path:
        target = self.dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.dir.resolve()).as_posix()

    def end(self, ok: bool, error: str = "") -> None:
        if self.ended:
            return
        self.ended = True
        self.emit(protocol.RunEnd(ok=ok, error=error, seconds=round(time.time() - self.started, 1)))
        record.run_ended(ok, error)
        # Nothing after `run.end`: a watcher stops there, and an event behind it would only ever be
        # seen by a page catching up - which would fold it into a run it has been told is over.
        self._closed = True


def current() -> Run:
    """The context's run, started on first use (along with the dashboard, if it is enabled)."""
    ctx = context.current()
    with ctx.lock:
        if ctx.run is None:
            _prune(config.RUNS_DIR, config.RUNS_KEEP)
            ctx.run = run = Run(config.RUNS_DIR)
            run.emit(protocol.RunStart(title=run.title, pid=os.getpid()))
            _install_exit_hooks(run)
            threading.Thread(target=_sample_resources, args=(run,), daemon=True, name="run-stats").start()
            if config.DASHBOARD:
                from difflab.api.serve import ensure_running, url

                ensure_running()
                # The one thing a render writes to the terminal: the address that draws the rest.
                print(f"DiffLab · watching at {url()}/runs/{run.id}", flush=True)
        return ctx.run


def disable() -> None:
    """Record nothing from this process - for the standalone dashboard, which watches runs rather than being one."""
    context.current().quiet = True


@contextmanager
def quiet() -> Iterator[None]:
    """Record nothing while this is open - scoped, unlike `disable`.

    For work that loads no model and produces no video: a preflight, a parity check. A run opened for
    it would leave a row and a directory behind, and prune an older one to make room.
    """
    with context.bind(context.Context(quiet=True)):
        yield


def active() -> Run | None:
    return None if context.current().quiet else current()


def emit(event: protocol.Event) -> None:
    run = active()
    if run is not None:
        run.emit(event)


def _install_exit_hooks(run: Run) -> None:
    def stopped(number: int, _frame: object) -> None:
        """SIGTERM's default handler does not run `atexit`, so without this the log simply ends and
        anything watching has to infer from 15 s of silence what happened. `web.runs.kill` sends
        exactly this signal, and so does a container being shut down."""
        run.end(ok=False, error=f"stopped ({signal.Signals(number).name})")
        raise SystemExit(128 + number)

    for number in (signal.SIGTERM, signal.SIGHUP):
        with contextlib.suppress(ValueError, OSError):  # not the main thread, or no such signal here
            signal.signal(number, stopped)

    previous = sys.excepthook

    def excepthook(kind: type[BaseException], value: BaseException, tb: TracebackType | None) -> None:
        text = "".join(traceback.format_exception(kind, value, tb))
        run.emit(protocol.Error(text=text[-6000:]))
        run.end(ok=False, error=f"{kind.__name__}: {value}")
        previous(kind, value, tb)

    sys.excepthook = excepthook
    atexit.register(lambda: run.end(ok=True))


def _sample_resources(run: Run) -> None:
    import torch

    while not run.ended:
        data: dict[str, float] = {}
        if torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
            free, total = torch.cuda.mem_get_info()
            data |= {"vram_used": (total - free) / 1024**3, "vram_total": total / 1024**3}
        memory = meminfo()
        if memory:
            data |= {"ram_used": memory[0], "ram_total": memory[1]}
        data["rss"] = rss()
        run.emit(protocol.Stats(**{k: round(v, 2) for k, v in data.items()}))
        time.sleep(STATS_EVERY)


def meminfo() -> tuple[float, float] | None:
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            fields[name] = int(value.split()[0]) / 1024**2  # kB -> GB
        return fields["MemTotal"] - fields["MemAvailable"], fields["MemTotal"]
    except (OSError, KeyError, ValueError):
        return None


def rss() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return 0.0


def _prune(root: Path, keep: int) -> None:
    """Runs hold previews and per-chunk videos; the final video lives elsewhere.

    A run still being written is skipped, however old it is: the worker's thumbnails and previews
    are under it.
    """
    if not root.exists():
        return
    runs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    for stale in runs[: max(0, len(runs) - keep + 1)]:
        if _writing(stale.name):
            print(f"difflab: keeping run {stale.name}, which is still being written", file=sys.stderr, flush=True)
            continue
        shutil.rmtree(stale, ignore_errors=True)


def _writing(run_id: str) -> bool:
    """Whether a run still has the process that opened it.

    The pid, not the heartbeat `web/runs.py` derives liveness from: that rule answers whether a page
    should call a run live, and this asks whether anything still has the directory open, which the OS
    answers. A row that never recorded an end and whose process is gone - what the OOM reaper leaves -
    is prunable, which is what keeps this from being a way to fill a disk.
    """
    try:
        with store.connect() as connection:
            row = store.run(connection, run_id)
    except (sqlite3.Error, OSError) as problem:
        print(f"difflab: {run_id} was kept, not having been looked up ({problem})", file=sys.stderr, flush=True)
        return True
    return row is not None and not row.settled and row.pid is not None and _alive(row.pid)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # someone else's process, which is still a process
    return True
