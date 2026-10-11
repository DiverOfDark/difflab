"""The thread that takes jobs off the store and runs them on the card, one at a time.

One worker per card, started beside the server. It picks the oldest queued job whose inputs are
settled and whose run is not paused, binds a context for it, runs the tool, records the outputs, and
ends the run when its last job settles. A failed job cancels the rest of its run. Pause, resume and
cancel arrive as commands on the run; re-roll, edit, stop and finish as commands on the running job,
which the chained renderer reads for itself.

Simplicity over isolation: the server imports torch and holds the weights, so a crash in a job takes
the process down and a restart takes the twenty seconds torch does.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
import traceback
from typing import Any

from difflab import config, store
from difflab.core import context, events, protocol, ui
from difflab.core.settings import DirectorSettings, JudgeSettings
from difflab.screenplay import made
from difflab.tools import TOOLS, direct
from difflab.work import Skipped, cleanup, plan_runners, runners

# The tools that ask through the director, which takes its model from the job rather than a parameter.
_DIRECTED = {tool.name for tool in TOOLS if tool.settings in (DirectorSettings, JudgeSettings)}

POLL = 0.5  # seconds between looks at an empty queue
# Set by `POST /api/restart`: the process replaces itself with a fresh one between jobs, so code edited
# since it started takes effect without killing a render in flight.
RESTART = threading.Event()
STATS_EVERY = 2.0  # the liveness heartbeat the page watches


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="difflab-worker", daemon=True)
        self.stopping = threading.Event()
        self.ctx: context.Context | None = None  # the job in flight, for the sampler and the commands
        self.swept = 0.0

    def run(self) -> None:
        _settle_orphans()
        threading.Thread(target=self._watch, name="difflab-worker-watch", daemon=True).start()
        while not self.stopping.is_set():
            if not self.step():
                if RESTART.is_set():
                    print("difflab: restarting, the queue is idle", flush=True)
                    # `orig_argv`, not `argv`: under `-m` argv[0] is the module's file, which run as a
                    # script cannot import its own package.
                    os.execv(sys.executable, sys.orig_argv)
                if time.time() - self.swept > config.CLEANUP_EVERY_HOURS * 3600:
                    self.swept = time.time()
                    try:
                        print(cleanup.sweep().line(), flush=True)
                    except (OSError, sqlite3.Error) as problem:
                        print(f"difflab: cleanup stopped ({problem})", flush=True)
                time.sleep(POLL)

    def step(self) -> bool:
        """One job, or nothing. Returns whether there was one."""
        try:
            with store.connect() as connection:
                job = _claim(connection)
        except sqlite3.Error as problem:
            print(f"difflab: the worker could not read the queue ({problem})", flush=True)
            return False
        if job is None:
            return False
        self._execute(job)
        return True

    # --- one job ---------------------------------------------------------------------------------
    def _execute(self, job: store.Job) -> None:
        ctx = context.Context()
        ctx.run_id = job.run_id
        with context.bind(ctx), store.connect() as connection:
            run = store.run(connection, job.run_id)
            listed = store.jobs_of(connection, job.run_id)
            first = store.newest_event(connection, job.run_id) is None
        title = run.title if run is not None else job.run_id
        sink: events.Run = events.Run(config.RUNS_DIR, run_id=job.run_id, title=title)
        ctx.run = sink
        with context.bind(ctx):
            self.ctx = ctx
            if first:
                sink.emit(protocol.RunStart(title=title, pid=os.getpid()))
                spec = run.data() if run is not None else None
                if spec and isinstance(spec.get("items"), list):
                    sink.emit(protocol.Plan(title=title, items=[str(item) for item in spec["items"]]))
            ctx.job_id = job.id
            sink.emit(
                protocol.JobBegin(
                    index=job.position,
                    total=len(listed),
                    position=job.position,
                    mode=job.tool,
                    label=job.label,
                    job=job.id,
                )
            )
            started = time.perf_counter()
            state, error = store.DONE, ""
            outputs: dict[str, Any] = {}
            try:
                inputs = _resolve(job.inputs)
                handed = _skipped(run, job.tool, inputs)
                if handed is not None:
                    outputs = Skipped(handed)
                else:
                    with direct.using(job.settings.get("checkpoint") if job.tool in _DIRECTED else None):
                        outputs = runners.RUNNERS[job.tool](inputs, job.settings)
                if isinstance(outputs, Skipped):
                    state = store.SKIPPED
            except context.Cancelled:
                state, error = store.CANCELLED, "cancelled"
            except Exception as problem:
                state, error = store.FAILED, f"{type(problem).__name__}: {problem}"
                sink.emit(protocol.Error(text="".join(traceback.format_exc())[-6000:]))
            # A run's cancel taken while a job that never looks at it was in flight (composing a keyframe) ended
            # with that job, and production 13's next seven clips rendered on (2026-10-05): the rest goes now.
            cancelled = ctx.cancel.is_set() and state in (store.DONE, store.SKIPPED)
            ctx.task_id = None
            with store.connect() as connection:
                store.end_job(connection, job.id, state, outputs=outputs, error=error)
                made.settle(connection, job, outputs, state)
                if state not in (store.DONE, store.SKIPPED):
                    _cancel_rest(connection, job.run_id, f"{error} in job {job.position + 1}")
                elif cancelled:
                    _cancel_rest(connection, job.run_id, "cancelled")
                remaining = [one for one in store.jobs_of(connection, job.run_id) if not one.settled]
                failed = [one for one in store.jobs_of(connection, job.run_id) if one.state == store.FAILED]
            delivered = next((str(v) for v in outputs.values() if isinstance(v, str)), "")
            sink.emit(
                protocol.JobEnd(
                    index=job.position,
                    total=len(listed),
                    ok=state in (store.DONE, store.SKIPPED),
                    status=state,
                    error=error,
                    made=delivered,
                    seconds=round(time.perf_counter() - started, 1),
                )
            )
            ctx.job_id = None
            for note in made.pending_notes():
                ui.note(note)
            # Once the run has ended, one attempt for everything its verdicts asked: queued after each failing
            # verdict, production 14's eleven clips (2026-10-04) queued seven whole-chapter runs, each with only
            # the seeds re-rolled before it.
            for chapter_id, keyframes_only in made.pending_rerenders() if not remaining else []:
                from difflab.screenplay import planner

                try:
                    again = planner.render_chapter(chapter_id, asked_by="judge", keyframes_only=keyframes_only)
                    ui.note(f"the judge asked for another attempt: run {again} queued")
                except (LookupError, ValueError, sqlite3.Error) as problem:
                    ui.note(f"the judge asked for another attempt, but none could be queued ({problem})")
            if not remaining:
                if cancelled:
                    sink.end(ok=False, error="cancelled")
                else:
                    sink.end(ok=not failed and state != store.CANCELLED, error=error if state != store.DONE else "")
            self.ctx = None

    # --- beside the job: the heartbeat, and the commands that arrive while it runs ------------------
    def _watch(self) -> None:
        while not self.stopping.is_set():
            ctx = self.ctx
            if ctx is not None and ctx.run is not None and not ctx.run.ended:
                with context.bind(ctx):
                    ctx.run.emit(protocol.Stats(**_resources()))
                    for command in _run_commands(ctx.run_id):
                        if isinstance(command, protocol.Cancel):
                            ctx.cancel.set()
                        elif isinstance(command, protocol.Pause):
                            _set_run_state(ctx.run_id, store.PAUSED)
                        elif isinstance(command, protocol.Resume):
                            _set_run_state(ctx.run_id, store.RUNNING)
            time.sleep(STATS_EVERY)


def _claim(connection: sqlite3.Connection) -> store.Job | None:
    """The next job to run, marked running, or `None`. Commands on runs that are waiting are applied
    here, since nothing else looks at a run that has no job in flight."""
    with store.transaction(connection):
        for run in store.runs(connection):
            if run.state in (store.QUEUED, store.RUNNING, store.PAUSED):
                _apply_waiting(connection, run)
        settled: dict[int, str] = {}
        for candidate in store.queued_jobs(connection):
            owner = store.run(connection, candidate.run_id)
            if owner is None or owner.state == store.PAUSED:
                continue
            for dep in candidate.depends_on():
                if dep not in settled:
                    found = store.job(connection, dep)
                    settled[dep] = found.state if found else store.FAILED
            states = [settled[dep] for dep in candidate.depends_on()]
            if any(state in (store.FAILED, store.CANCELLED) for state in states):
                store.end_job(connection, candidate.id, store.CANCELLED, error="an input job did not finish")
                continue
            if all(state in (store.DONE, store.SKIPPED) for state in states):
                store.start_job(connection, candidate.id)
                return store.job(connection, candidate.id)
    return None


def _apply_waiting(connection: sqlite3.Connection, run: store.Run) -> None:
    for command in store.take_commands(connection, "run", run.id):
        try:
            parsed = protocol.command(command.value())
        except ValueError:
            continue
        if isinstance(parsed, protocol.Pause):
            store.set_run_state(connection, run.id, store.PAUSED)
        elif isinstance(parsed, protocol.Resume):
            store.set_run_state(connection, run.id, store.RUNNING if run.started_at else store.QUEUED)
        elif isinstance(parsed, protocol.Cancel):
            _cancel_rest(connection, run.id, "cancelled")
            if not any(one.state == store.RUNNING for one in store.jobs_of(connection, run.id)):
                store.end_run(connection, run.id, ok=False, error="cancelled")


def _settle_orphans() -> None:
    """One worker per process, so a run still `running` at start was left by a process that died - the
    OOM reaper, a kill - and nothing will ever end it. Its job in flight is failed, the rest cancelled;
    queued runs are untouched and taken in turn."""
    why = "the worker died before this ended"
    try:
        with store.connect() as connection, store.transaction(connection):
            for run in store.runs(connection):
                if run.state != store.RUNNING:
                    continue
                for job in store.jobs_of(connection, run.id):
                    if job.state == store.RUNNING:
                        store.end_job(connection, job.id, store.FAILED, error=why)
                _cancel_rest(connection, run.id, why)
                store.end_run(connection, run.id, ok=False, error=why)
                print(f"difflab: run {run.id} was left running by a dead process and is now failed", flush=True)
    except sqlite3.Error as problem:
        print(f"difflab: the orphaned runs could not be settled ({problem})", flush=True)


def _cancel_rest(connection: sqlite3.Connection, run_id: str, why: str) -> None:
    for one in store.jobs_of(connection, run_id):
        if one.state in (store.QUEUED, store.BLOCKED):
            store.end_job(connection, one.id, store.CANCELLED, error=why)


def _skipped(run: store.Run | None, tool: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
    """The outputs a job a person skipped hands on, or None to run it: a step skipped on the ingesting page."""
    spec = (run.data() if run is not None else None) or {}
    skip = plan_runners.SKIPS.get(tool)
    return skip(inputs) if skip is not None and tool in spec.get("skip", []) else None


def _resolve(inputs: dict[str, Any]) -> dict[str, Any]:
    """Another job's output, by name, in place of the reference to it - inside lists too."""
    return {name: _value(name, value) for name, value in inputs.items()}


def _value(name: str, value: Any) -> Any:
    if isinstance(value, dict) and "from_job" in value:
        with store.connect() as connection:
            source = store.job(connection, int(value["from_job"]))
        if source is None or value["output"] not in source.outputs:
            raise ValueError(f"{name}: job {value['from_job']} made no {value['output']!r}")
        return source.outputs[value["output"]]
    if isinstance(value, list):
        return [_value(name, item) for item in value]
    return value


def _run_commands(run_id: str | None) -> list[protocol.Command]:
    if run_id is None:
        return []
    out = []
    try:
        with store.connect() as connection, store.transaction(connection):
            for command in store.take_commands(connection, "run", run_id):
                try:
                    out.append(protocol.command(command.value()))
                except ValueError:
                    continue
    except sqlite3.Error:
        return []
    return out


def _set_run_state(run_id: str | None, state: str) -> None:
    if run_id is None:
        return
    with store.connect() as connection:
        store.set_run_state(connection, run_id, state)


def _resources() -> dict[str, float]:
    import torch

    data: dict[str, float] = {"rss": events.rss()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        free, total = torch.cuda.mem_get_info()
        data |= {"vram_used": (total - free) / 1024**3, "vram_total": total / 1024**3}
    memory = events.meminfo()
    if memory:
        data |= {"ram_used": memory[0], "ram_total": memory[1]}
    return {k: round(v, 2) for k, v in data.items()}
