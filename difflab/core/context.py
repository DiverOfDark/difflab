"""What a running job carries with it, in one object instead of a global per module.

Bound with a contextvar, so a thread that runs a job sees its own and a thread that runs none sees
the process default. Everything here used to be a module-level variable read from wherever it was
needed: the run recording the events, the tally and overview the page draws, the preview target,
the store rows, the director's settled answers.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from difflab.core.events import Run
    from difflab.core.ui import Overview, Tally


class Cancelled(Exception):  # noqa: N818 - control flow, not an error
    """Abandon the job in flight: a page pressed cancel, and the tools check between steps."""


@dataclass
class Context:
    # The run recording this context's events. `None` until something emits, and never while `quiet`.
    run: Run | None = None
    quiet: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    tally: Tally | None = None
    overview: Overview | None = None
    reporting: Callable[[float], None] | None = None
    # Which clip the next previews belong to. A long video sets it per chunk; a single clip leaves it.
    preview_item: int | None = None
    preview_name: str = "clip"
    previews_broken: bool = False
    # The store rows this run writes into: `record` fills them, `core.media` reads them.
    run_id: str | None = None
    job_id: int | None = None
    task_id: int | None = None
    # Director answers kept with the gate still complaining, for the panel at the end of a direction.
    settled: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


_DEFAULT = Context()
_current: contextvars.ContextVar[Context] = contextvars.ContextVar("difflab.context")


def current() -> Context:
    """The bound context, or the process default: a thread nothing bound one to - the resource
    sampler, a preview writer - reports into the same run as the thread that started it."""
    try:
        return _current.get()
    except LookupError:
        return _DEFAULT


@contextmanager
def bind(context: Context) -> Iterator[Context]:
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)
