"""What a run is doing, as events. The page is where you watch it.

Every call here is a `core.events` emission, and the browser (`difflab/api`, `frontend/`) draws them.

**When nothing is recording, these print instead.** `events.quiet()` is how a tool that is not a
render - `take.preflight` - says it wants no run directory and no dashboard, and a report with no
reader is useless. If there is a page, the page has it; if there is not, the terminal does.

Markup in a caller's string (`[dim]`, `[bold green]`) is stripped rather than rendered - and because
`[Shot 1]` in a prompt is not markup, only known style names are taken as tags.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

from difflab.core import context, events, protocol

_LABEL_WIDTH = 34
_DETAIL_WIDTH = 30

# Style names this repository writes in `[...]`. Anything else in brackets is text - a prompt says
# "[Shot 1]" and a caption names "[Video 1]", and neither is a tag.
_STYLES = frozenset(
    (
        "bold",
        "dim",
        "italic",
        "underline",
        "strike",
        "red",
        "green",
        "cyan",
        "yellow",
        "magenta",
        "blue",
        "white",
        "black",
        "link",
    )
)


def _plain(markup: str) -> str:
    def strip(match: re.Match[str]) -> str:
        body = match.group(1)
        if body == "/":
            return ""
        words = body.removeprefix("/").split("=", 1)[0].split()
        return "" if words and all(word in _STYLES for word in words) else match.group(0)

    return re.sub(r"\[([^\[\]]*)\]", strip, markup)


def _watched() -> bool:
    return events.active() is not None


def _say(text: str) -> None:
    """Print, but only when nothing is recording - otherwise the page has it and stdout is quiet."""
    if not _watched():
        print(text, flush=True)


class _Throttle:
    """At most one event every `every` seconds - and always the last one."""

    def __init__(self, every: float = 0.5) -> None:
        self.every, self.last = every, 0.0

    def __call__(self, final: bool = False) -> bool:
        now = time.perf_counter()
        if final or now - self.last >= self.every:
            self.last = now
            return True
        return False


def line(text: str) -> None:
    events.emit(protocol.Log(text=_plain(text)))
    _say(f"     {_plain(text)}")


def note(message: str) -> None:
    """For something the run worked around rather than failed on."""
    text = _plain(message)
    events.emit(protocol.Log(text=text))
    _say(f"  !  {text}")


def panel(title: str, rows: dict[str, str], color: str = "cyan") -> None:
    clean = {key: _plain(value) for key, value in rows.items()}
    events.emit(protocol.Panel(title=_plain(title), rows=clean, color=color))
    if not _watched():
        print(f"\n── {_plain(title)}", flush=True)
        for key, value in clean.items():
            print(f"   {key:<22} {value}", flush=True)


def done(label: str, detail: str, seconds: float) -> None:
    """The `stage.end` anything watching closes a stage on: `step` and `loading` call this rather than
    emitting a second one of their own."""
    events.emit(protocol.StageEnd(label=_plain(label), ok=True, detail=_plain(detail), seconds=round(seconds, 1)))
    _say(f"  ✓  {_plain(label):<{_LABEL_WIDTH}} {_plain(detail):<{_DETAIL_WIDTH}} {seconds:6.1f}s")


@contextmanager
def step(label: str) -> Iterator[Callable[[str], None]]:
    """The yielded function sets the detail the finished line carries."""
    detail = ""

    def set_detail(text: str) -> None:
        nonlocal detail
        detail = text

    t0 = time.perf_counter()
    events.emit(protocol.StageStart(label=label))
    try:
        yield set_detail
    except BaseException:
        events.emit(protocol.StageEnd(label=label, ok=False, seconds=round(time.perf_counter() - t0, 1)))
        _say(f"  ✗  {label}")
        raise
    done(label, detail, time.perf_counter() - t0)


def duration(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds // 60)}m {seconds % 60:02.0f}s"


def link(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path)


class Tally:
    """Where a job of many model calls stands: which of its stages, and which call of that stage.

    A stage's call count is known when it starts (one per photo, one per member...), so the line is
    exact without predicting the whole job up front. Repairs re-ask the same call and do not count.
    """

    def __init__(self, name: str, stages: tuple[str, ...]) -> None:
        self.name, self.stages = name, stages
        self.index, self.calls, self.done = -1, 0, 0
        self._t0 = time.perf_counter()

    def stage(self, name: str, calls: int) -> None:
        """A name this tally does not have is ignored, not raised: a block marks its own phases
        (`animate.sample` says "sampling") and runs under whichever tally is open, which during a take
        is the director's and has entirely different stages. A progress counter must never be the thing
        that ends a render."""
        if name not in self.stages:
            return
        self.index, self.calls, self.done = self.stages.index(name), calls, 0
        self.publish()

    def publish(self) -> None:
        """The director is a quarter of an hour of model calls with no chunks to show yet, so without
        this a page watching it has nothing to draw but a spinner."""
        events.emit(
            protocol.Tally(
                name=self.name,
                stages=list(self.stages),
                index=self.index,
                calls=self.calls,
                done=self.done,
                seconds=round(time.perf_counter() - self._t0, 1),
            )
        )

    def position(self) -> str:
        """ "4/8 opening 2/5": the stage, and the call under way (or the last one, once all are done).
        A stage with no calls of its own says just "4/8 sampling"."""
        if self.index < 0:
            return ""
        where = f"{self.index + 1}/{len(self.stages)} {self.stages[self.index]}"
        if self.calls <= 0:
            return where
        return f"{where} {min(self.done + 1, self.calls)}/{self.calls}"


@contextmanager
def tally(name: str, stages: tuple[str, ...]) -> Iterator[Tally]:
    ctx = context.current()
    outer, opened = ctx.tally, Tally(name, stages)
    ctx.tally = opened
    try:
        yield opened
    finally:
        events.emit(protocol.TallyEnd(name=name, seconds=round(time.perf_counter() - opened._t0, 1)))
        ctx.tally = outer


def stage(name: str, calls: int = 0) -> None:
    """A no-op outside a tally. `calls=0` for a phase that is one thing rather than a set of model
    calls - loading, encoding, sampling: it moves the phase counter with no sub-count to show."""
    tally = context.current().tally
    if tally is not None:
        tally.stage(name, calls)


def called() -> None:
    tally = context.current().tally
    if tally is not None:
        tally.done += 1
        tally.publish()


def _where() -> str:
    tally = context.current().tally
    return f"[{tally.position()}] " if tally is not None and tally.index >= 0 else ""


def answered(title: str, detail: str, seconds: float) -> None:
    done(f"{_where()}{title}", detail, seconds)


@contextmanager
def live_answer(title: str, echo: bool = True) -> Iterator[Callable[[str], None]]:
    """Text streaming out of a model. The yielded function replaces what has arrived so far.

    A throttled word count goes out while it arrives: a 2048-token director call is a minute in which
    a page with nothing in between looks hung. The count rather than the text, because
    `stage.progress` is folded into one line of status and half a paragraph arriving there four times
    a second is not something anyone reads; the text itself rides on `stage.end`, where it is kept.

    `echo` chooses what the finished line says - the answer itself, or how many words it was.
    """
    text = ""
    t0 = time.perf_counter()
    throttle = _Throttle()
    events.emit(protocol.StageStart(label=title, unit="word"))

    def update(latest: str) -> None:
        nonlocal text
        text = latest
        if throttle():
            events.emit(
                protocol.StageProgress(
                    label=title,
                    done=len(latest.split()),
                    unit="word",  # no total: the page says "126 words" and crawls rather than filling
                    elapsed=round(time.perf_counter() - t0, 1),
                )
            )

    try:
        yield update
    except BaseException:
        events.emit(
            protocol.StageEnd(label=title, ok=False, detail=text[-4000:], seconds=round(time.perf_counter() - t0, 1))
        )
        raise
    seconds = time.perf_counter() - t0
    events.emit(protocol.StageEnd(label=title, ok=True, detail=text[-4000:], seconds=round(seconds, 1)))
    _say(f"  ✓  {_where()}{title:<{_LABEL_WIDTH}} {text if echo else f'{len(text.split())} words'}")


@contextmanager
def progress(label: str, total: int, unit: str = "step") -> Iterator[Callable[[], None]]:
    """A counted stage of `total` `unit`s. Yields `advance`, to be called once per unit.

    `total` may be 0, for work whose length is genuinely unknown until it is done - a video whose
    container does not declare its frame count. Such a stage counts as successful if it did anything
    at all: comparing against a total of zero would mark every one of them failed.

    A count may also run PAST its total, which is not an error either: `through` counts a model's
    blocks, and a pipeline that guides runs them twice - once for the prompt and once for the negative
    - so 48 layers report 96 passes. Measured on the first LTX render that used it: the page read
    `81/48` and the stage finished with a cross. Past the total it is reported as unknown, so the page
    goes back to drawing a rising count.
    """
    started, count, throttle = time.perf_counter(), 0, _Throttle()
    events.emit(protocol.StageStart(label=label, total=total, unit=unit))

    def advance() -> None:
        nonlocal count
        count += 1
        if context.current().cancel.is_set():
            raise context.Cancelled
        if throttle(final=count == total):
            events.emit(
                protocol.StageProgress(
                    label=label,
                    done=count,
                    total=total if count <= total else 0,
                    unit=unit,
                    elapsed=round(time.perf_counter() - started, 1),
                )
            )

    try:
        yield advance
    finally:
        seconds = time.perf_counter() - started
        events.emit(
            protocol.StageEnd(label=label, ok=count >= total if total else count > 0, seconds=round(seconds, 1))
        )
        _say(f"  ✓  {label:<{_LABEL_WIDTH}} {f'{count}/{total} {unit}':<{_DETAIL_WIDTH}} {seconds:6.1f}s")


@contextmanager
def through(label: str, blocks: Any, unit: str = "layer") -> Iterator[None]:
    """Count a model's own blocks as they run, for a call that is otherwise one opaque forward.

    Encoding a prompt and decoding a clip are single calls from here - no loop to wrap and no callback
    to pass - but inside each is a list of blocks that runs in order, and a forward hook on every one
    of them is a counter for free.

    `blocks` is anything iterable of modules. They may run more than once, which is what a diffusion
    decoder does, and then the count runs past the total (see `progress`).
    """
    try:
        listed = list(blocks)
    except TypeError:
        listed = []
    if not listed:
        with progress(label, 0, unit=unit) as advance:
            yield
            advance()
        return

    handles = []
    with progress(label, len(listed), unit=unit) as advance:
        try:
            for block in listed:
                handles.append(block.register_forward_hook(lambda *_: advance()))
            yield
        finally:
            for handle in handles:
                handle.remove()


@contextmanager
def transfer(label: str, total: int, measure: Callable[[], int]) -> Iterator[None]:
    """A download that reports nothing itself: `measure()` - the bytes written so far - is read twice
    a second while the block runs, so something watching can draw a bar from it."""
    started, stop = time.perf_counter(), threading.Event()
    events.emit(protocol.StageStart(label=label, total=total, unit="byte"))

    def poll() -> None:
        while not stop.wait(0.5):
            events.emit(
                protocol.StageProgress(
                    label=label,
                    done=min(measure(), total),
                    total=total,
                    elapsed=round(time.perf_counter() - started, 1),
                )
            )

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    ok = False
    try:
        yield
        ok = True
    finally:
        stop.set()
        watcher.join()
        seconds = time.perf_counter() - started
        events.emit(protocol.StageEnd(label=label, ok=ok, seconds=round(seconds, 1)))
        _say(f"  ✓  {label:<{_LABEL_WIDTH}} {f'{total / 1e6:.0f} MB':<{_DETAIL_WIDTH}} {seconds:6.1f}s")


@contextmanager
def loading(label: str, unit: str = "tensor") -> Iterator[Callable[[str], None]]:
    """transformers walks its tensors through `logging.tqdm`; a hook turns any sized loop it starts in
    here into a counted stage (quantizing a 27B from bf16 is minutes of it), and leaves every other one
    to the silent default. The yielded function sets the finished line's detail."""
    from transformers.utils import logging as hf_logging  # heavy, and only a loader needs it

    detail = ""

    def set_detail(text: str) -> None:
        nonlocal detail
        detail = text

    def hook(factory: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        items = args[0] if args else kwargs.get("iterable")
        if items is None or not hasattr(items, "__len__"):
            return factory(*args, **kwargs)

        def walk() -> Iterator[Any]:
            with progress(str(kwargs.get("desc") or label), total=len(items), unit=unit) as advance:
                for item in items:
                    yield item
                    advance()

        return walk()

    t0 = time.perf_counter()
    events.emit(protocol.StageStart(label=label))
    previous = hf_logging.set_tqdm_hook(hook)
    try:
        yield set_detail
    finally:
        hf_logging.set_tqdm_hook(previous)
    done(label, detail, time.perf_counter() - t0)


@contextmanager
def sampling(label: str, steps: int, blocks: int) -> Iterator[Callable[[], None]]:
    """Denoising, counted once per transformer block rather than once per step: a full-size H3 step
    takes about two minutes, so a four-step run reported per step looks hung for most of it. Yields
    `tick`, to be called after each block."""
    total = steps * blocks
    started, count, throttle = time.perf_counter(), 0, _Throttle()
    events.emit(protocol.StageStart(label=label, total=total, unit="block", steps=steps, blocks=blocks))

    def tick() -> None:
        nonlocal count
        count += 1
        if context.current().cancel.is_set():
            raise context.Cancelled
        overview = context.current().overview
        if overview is not None:
            overview.sampling(count / total, time.perf_counter() - started)
        if throttle(final=count == total):
            events.emit(
                protocol.StageProgress(
                    label=label, done=count, total=total, elapsed=round(time.perf_counter() - started, 1)
                )
            )

    try:
        yield tick
    finally:
        seconds = time.perf_counter() - started
        events.emit(protocol.StageEnd(label=label, ok=count == total, seconds=round(seconds, 1)))
        _say(f"  ✓  {label:<{_LABEL_WIDTH}} {f'{count // blocks}/{steps} steps':<{_DETAIL_WIDTH}} {seconds:6.1f}s")
        overview = context.current().overview
        if overview is not None:
            overview.sampled(seconds)


class Overview:
    """Every item a long run has planned, and where the whole of it stands.

    The ETA is built from what has been measured, split into sampling and everything else, because a
    chunk found in the cache skips sampling and would otherwise make the rest look cheap. Before any
    chunk has finished it is a lower bound - the sampling projected from the running item, nothing yet
    known about the decode - and it says so.
    """

    def __init__(self, title: str, items: list[str]) -> None:
        self.title, self.items = title, items
        self.state = ["pending"] * len(items)
        self.seconds = [0.0] * len(items)
        self.activity = [""] * len(items)
        self.current = -1
        self.started = time.perf_counter()
        self._item_started = 0.0
        self._sample_fraction = 0.0
        self._sample_elapsed = 0.0
        self._sampled_seconds = 0.0  # this item's finished sampling, 0 while it has not sampled
        self._sampling_times: list[float] = []
        self._rest_times: list[float] = []
        self._throttle = _Throttle(1.0)
        events.emit(protocol.Plan(title=title, items=items))

    # --- reported by the run -----------------------------------------------------------------------
    def begin(self, index: int) -> None:
        self.current, self.state[index] = index, "running"
        self.activity[index] = "preparing"
        self._item_started = time.perf_counter()
        self._sample_fraction = self._sample_elapsed = self._sampled_seconds = 0.0
        events.emit(protocol.ItemBegin(index=index))
        _say(f"\n── {self.title}: {self.items[index]} · {self.summary()}")
        self._publish(force=True)

    def end(self) -> None:
        index = self.current
        total = time.perf_counter() - self._item_started
        self.state[index], self.seconds[index] = "done", total
        if self._sampled_seconds:
            self._sampling_times.append(self._sampled_seconds)
            self.activity[index] = f"sampled {duration(self._sampled_seconds)}"
        else:
            self.activity[index] = "from cache"
        self._rest_times.append(total - self._sampled_seconds)
        self.current = -1
        events.emit(protocol.ItemEnd(index=index, seconds=round(total, 1), activity=self.activity[index]))
        _say(f"── {self.title}: {self.items[index]} done · {self.summary()}")
        self._publish(force=True)

    def reset(self, start: int) -> None:
        """A re-roll rewinds the chain to `start`."""
        if self.current >= start:
            self.current = -1
        for index in range(start, len(self.items)):
            self.state[index], self.seconds[index], self.activity[index] = "pending", 0.0, ""
        events.emit(protocol.ItemsReset(start=start))
        self._publish(force=True)

    def _publish(self, force: bool = False) -> None:
        if not self._throttle(final=force):
            return
        estimate = self.remaining()
        fraction = self.fraction()
        events.emit(
            protocol.PlanUpdate(
                fraction=round(fraction, 4),
                eta=round(estimate[0]) if estimate else None,
                eta_sure=estimate[1] if estimate else False,
                elapsed=round(time.perf_counter() - self.started),
                states=self.state,
                activity=self.activity,
            )
        )
        hook = context.current().reporting
        if hook is not None:
            hook(fraction)

    # --- reported by the stages --------------------------------------------------------------------
    def sampling(self, fraction: float, elapsed: float) -> None:
        self._sample_fraction, self._sample_elapsed = fraction, elapsed
        if self.current >= 0:
            self.activity[self.current] = f"sampling {fraction:.0%}"
        self._publish()

    def sampled(self, seconds: float) -> None:
        self._sampled_seconds, self._sample_fraction = seconds, 1.0
        if self.current >= 0:
            self.activity[self.current] = "decoding"
        self._publish(force=True)

    # --- the estimate ------------------------------------------------------------------------------
    def remaining(self) -> tuple[float, bool] | None:
        """Seconds left, and whether that is exact-ish (True) or only a lower bound (False)."""
        sampling = sum(self._sampling_times) / len(self._sampling_times) if self._sampling_times else None
        rest = sum(self._rest_times) / len(self._rest_times) if self._rest_times else None
        if sampling is None and 0 < self._sample_fraction < 1:
            sampling = self._sample_elapsed / self._sample_fraction  # projected from the running item
        if sampling is None:
            return None
        pending = self.state.count("pending")
        left = pending * (sampling + (rest or 0.0))
        if self.current >= 0:
            elapsed = time.perf_counter() - self._item_started
            if self._sample_fraction < 1:  # still sampling (or not started): the rest follows
                if self._sample_fraction > 0:
                    left += self._sample_elapsed * (1 - self._sample_fraction) / self._sample_fraction
                else:
                    left += sampling
                left += rest or 0.0
            else:  # sampled or skipped: only the rest is left, less what has already gone into it
                left += max(0.0, (rest or 0.0) - (elapsed - self._sampled_seconds))
        return left, rest is not None

    def fraction(self) -> float:
        estimate = self.remaining()
        if estimate is None:
            return self.state.count("done") / len(self.items)
        spent = time.perf_counter() - self.started
        return spent / (spent + estimate[0]) if spent + estimate[0] > 0 else 1.0

    def summary(self) -> str:
        elapsed = duration(time.perf_counter() - self.started)
        estimate = self.remaining()
        if estimate is None:
            eta = "eta once sampling starts"
        else:
            seconds, sure = estimate
            eta = f"eta {duration(seconds)}" if sure else f"eta ≥ {duration(seconds)} (decode not measured yet)"
        return f"{self.fraction():.0%} · {elapsed} elapsed · {eta}"


@contextmanager
def overview(title: str, items: list[str]) -> Iterator[Overview]:
    """Pinned for the duration of the block, so the stages inside it report into it."""
    ctx = context.current()
    view = Overview(title, items)
    previous, ctx.overview = ctx.overview, view
    try:
        yield view
    finally:
        ctx.overview = previous


@contextmanager
def reporting(hook: Callable[[float], None]) -> Iterator[None]:
    """Forward the running plan's fraction to something outside the render - the worker turns it into
    `job.progress`, so a project can say how far through a job it is without knowing what a chunk is.

    A mode that plans nothing opens no `Overview` and so reports nothing, which is the honest answer:
    an edit is one pipeline call and a director stage is ~800 s of VLM, and neither has a total for a
    fraction to be a share of.
    """
    ctx = context.current()
    previous, ctx.reporting = ctx.reporting, hook
    try:
        yield
    finally:
        ctx.reporting = previous


class EditProgress:
    """One image-edit pipeline call, as three stages: encoding, denoising, decoding.

    Encoding ends when the transformer is first called (a forward pre-hook); each denoising step is
    reported by the pipeline's `callback_on_step_end`.
    """

    def __init__(self, transformer: Any, steps: int) -> None:
        self._transformer = transformer
        self._steps = steps
        self._hook: Any = None
        self._phase: str | None = None
        self._t0 = 0.0
        self._denoising = False
        self._denoised = 0
        self._denoise_started = 0.0
        self._throttle = _Throttle()

    def __enter__(self) -> EditProgress:
        self._start_phase("Encoding prompt + reference image")
        self._hook = self._transformer.register_forward_pre_hook(self._on_first_transformer_call)
        return self

    def on_step_end(self, pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Pipeline `callback_on_step_end`: one denoising step done."""
        self._denoised = step + 1
        if self._throttle(final=self._denoised == self._steps):
            events.emit(
                protocol.StageProgress(
                    label="Denoising",
                    done=self._denoised,
                    total=self._steps,
                    elapsed=round(time.perf_counter() - self._denoise_started, 1),
                )
            )
        if self._denoised == self._steps:
            self._finish_denoising()
            self._start_phase("Decoding")
        return callback_kwargs

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._hook is not None:
            self._hook.remove()
        self._finish_denoising()  # only does something if denoising was interrupted
        self._end_phase(success=exc_type is None)

    def _on_first_transformer_call(self, module: Any, args: Any) -> None:
        self._hook.remove()
        self._hook = None
        self._end_phase(success=True)
        self._denoising = True
        self._denoised = 0
        self._denoise_started = time.perf_counter()
        events.emit(protocol.StageStart(label="Denoising", total=self._steps, unit="step"))

    def _finish_denoising(self) -> None:
        if not self._denoising:
            return
        self._denoising = False
        seconds = time.perf_counter() - self._denoise_started
        events.emit(protocol.StageEnd(label="Denoising", ok=self._denoised == self._steps, seconds=round(seconds, 1)))
        detail = f"{self._denoised}/{self._steps} steps"
        _say(f"  ✓  {'Denoising':<{_LABEL_WIDTH}} {detail:<{_DETAIL_WIDTH}} {seconds:6.1f}s")

    def _start_phase(self, label: str) -> None:
        self._phase, self._t0 = label, time.perf_counter()
        events.emit(protocol.StageStart(label=label))

    def _end_phase(self, success: bool) -> None:
        if self._phase is None:
            return
        if success:
            done(self._phase, "", time.perf_counter() - self._t0)
        else:
            events.emit(
                protocol.StageEnd(label=self._phase, ok=False, seconds=round(time.perf_counter() - self._t0, 1))
            )
        self._phase = None
