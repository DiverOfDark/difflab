"""Every event a run records and every command a page sends, as types.

One class per kind. The producer constructs the class, so a misspelt field fails where it is
written; the store reads `gauge` from the class to collapse a catch-up; `typescript()` writes the
same shapes for the page, so the fold there and the emitters here cannot drift apart.

Gauges: a gauge row carries the current value of something - a bar, a plan, a tally, a preview - so
a page joining late needs only the latest one, and the store's catch-up replays every history row
plus the last row per gauge. Measured over ten real run logs: 13,359 events, of which the gauge
kinds were 95.1% of rows and 82.6% of bytes, and the collapse folded to a byte-identical page state
(13,359 -> 658 events). The invariant a new kind has to satisfy: two events sharing a gauge key must
write the same pieces of state, because the later one will be the only one replayed.

    python -m difflab.core.protocol    # rewrites frontend/src/events.ts
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class Event:
    KIND: ClassVar[str] = ""

    def gauge(self) -> str | None:
        return None

    def data(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --- the process ------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunStart(Event):
    KIND = "run.start"
    title: str
    pid: int


@dataclass(frozen=True)
class RunEnd(Event):
    KIND = "run.end"
    ok: bool
    error: str
    seconds: float


@dataclass(frozen=True)
class Error(Event):
    KIND = "error"
    text: str


@dataclass(frozen=True)
class Stats(Event):
    KIND = "stats"
    rss: float
    vram_used: float | None = None
    vram_total: float | None = None
    ram_used: float | None = None
    ram_total: float | None = None

    def gauge(self) -> str:
        # A sample taken before CUDA was initialised carries RAM alone and must not supersede one that
        # carried VRAM: the page guards the two totals separately.
        present = [name for name in ("vram_total", "ram_total") if getattr(self, name)]
        return "stats:" + "+".join(present)


# --- words on the page ------------------------------------------------------------------------
@dataclass(frozen=True)
class Log(Event):
    KIND = "log"
    text: str


@dataclass(frozen=True)
class Panel(Event):
    KIND = "panel"
    title: str
    rows: dict[str, str]
    color: str


# --- one phase of one job ---------------------------------------------------------------------
@dataclass(frozen=True)
class StageStart(Event):
    KIND = "stage.start"
    label: str
    total: int | None = None
    unit: str | None = None
    steps: int | None = None
    blocks: int | None = None


@dataclass(frozen=True)
class StageProgress(Event):
    KIND = "stage.progress"
    label: str
    done: int
    elapsed: float
    total: int | None = None
    unit: str | None = None

    def gauge(self) -> str:
        return self.KIND


@dataclass(frozen=True)
class StageEnd(Event):
    KIND = "stage.end"
    label: str
    ok: bool
    seconds: float
    detail: str | None = None
    numbers: dict[str, float] | None = None


@dataclass(frozen=True)
class Tally(Event):
    KIND = "tally"
    name: str
    stages: list[str]
    index: int
    calls: int
    done: int
    seconds: float

    def gauge(self) -> str:
        return self.KIND


@dataclass(frozen=True)
class TallyEnd(Event):
    KIND = "tally.end"
    name: str
    seconds: float


# --- a plan of items, and where it stands ------------------------------------------------------
@dataclass(frozen=True)
class Plan(Event):
    KIND = "plan"
    title: str
    items: list[str]


@dataclass(frozen=True)
class ItemBegin(Event):
    KIND = "item.begin"
    index: int


@dataclass(frozen=True)
class ItemEnd(Event):
    KIND = "item.end"
    index: int
    seconds: float
    activity: str


@dataclass(frozen=True)
class ItemInfo(Event):
    KIND = "item.info"
    index: int
    version: int
    prompt: str
    seed: int
    opening: str
    references: list[str]


@dataclass(frozen=True)
class ItemVideo(Event):
    KIND = "item.video"
    index: int
    version: int
    video: str
    seed: int


@dataclass(frozen=True)
class Preview(Event):
    KIND = "preview"
    item: int | None
    name: str
    step: int
    steps: int
    video: str
    poster: str

    def gauge(self) -> str:
        return f"preview:{self.item}"


@dataclass(frozen=True)
class Picture(Event):
    """A picture a job made on its way to its result: a chained edit's every step and amendment."""

    KIND = "picture"
    label: str
    image: str  # relative to the run directory, so it goes when the run is pruned


# --- what was made ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Artifact(Event):
    KIND = "artifact"
    media: str
    name: str
    seed: int | None
    path: str


@dataclass(frozen=True)
class Final(Event):
    KIND = "final"
    video: str
    frames: int | None = None
    seconds: float | None = None


# --- one job of a run's list ------------------------------------------------------------------
@dataclass(frozen=True)
class JobBegin(Event):
    KIND = "job.begin"
    index: int
    total: int
    position: int
    mode: str
    label: str
    job: int | None


@dataclass(frozen=True)
class JobEnd(Event):
    KIND = "job.end"
    index: int
    total: int
    ok: bool
    status: str
    error: str
    made: str
    seconds: float


# --- what a plan found ------------------------------------------------------------------------
@dataclass(frozen=True)
class Found(Event):
    """The members a chapter's survey found that the chapters before it had not."""

    KIND = "found"
    chapter: int  # its position, from 0
    members: list[dict[str, str]]  # slug, kind, noun, find, photos (how many show it)


EVENTS: tuple[type[Event], ...] = tuple(
    cls for cls in vars(sys.modules[__name__]).values() if isinstance(cls, type) and issubclass(cls, Event) and cls.KIND
)
KINDS: dict[str, type[Event]] = {cls.KIND: cls for cls in EVENTS}


def event(kind: str, data: dict[str, Any]) -> Event:
    """A recorded row back as its class. Unknown kinds and fields raise: the store only ever holds
    what an emitter wrote, so a mismatch is a version skew worth seeing."""
    return KINDS[kind](**data)


def gauge(kind: str, data: dict[str, Any]) -> str | None:
    cls = KINDS.get(kind)
    if cls is None:
        return None
    try:
        return cls(**data).gauge()
    except TypeError:
        return None


# --- commands ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Command:
    CMD: ClassVar[str] = ""

    def data(self) -> dict[str, Any]:
        return {"cmd": self.CMD, **dataclasses.asdict(self)}


@dataclass(frozen=True)
class Cancel(Command):
    """Abandon the chunk in flight now."""

    CMD = "cancel"


@dataclass(frozen=True)
class Pause(Command):
    """Let the job in flight finish, then start nothing more of this run until resumed."""

    CMD = "pause"


@dataclass(frozen=True)
class Resume(Command):
    CMD = "resume"


COMMANDS: dict[str, type[Command]] = {cls.CMD: cls for cls in (Cancel, Pause, Resume)}


def command(data: Any) -> Command:
    """A command from the wire, or a ValueError that says what is wrong with it."""
    if not isinstance(data, dict) or data.get("cmd") not in COMMANDS:
        raise ValueError(f"cmd must be one of {sorted(COMMANDS)}")
    cls = COMMANDS[data["cmd"]]
    fields = {f.name: f for f in dataclasses.fields(cls)}
    given = {k: v for k, v in data.items() if k not in ("cmd", "t")}
    unknown = sorted(set(given) - set(fields))
    if unknown:
        raise ValueError(f"{cls.CMD} takes {sorted(fields) or 'nothing'}, not {unknown}")
    for name, f in fields.items():
        if name not in given and f.default is dataclasses.MISSING:
            raise ValueError(f"{cls.CMD} needs {name}")
    if "item" in given and not isinstance(given["item"], int):
        raise ValueError("item (a chunk index) must be an integer")
    if "seed" in given and given["seed"] is not None and not isinstance(given["seed"], int):
        raise ValueError("seed must be an integer")
    if "prompt" in given and not (isinstance(given["prompt"], str) and given["prompt"].strip()):
        raise ValueError("prompt must be a non-empty string")
    return cls(**given)


# --- the same shapes for the page ---------------------------------------------------------------
def typescript() -> str:
    def ts(hint: Any) -> str:
        origin = get_origin(hint)
        if origin is list:
            return f"{ts(get_args(hint)[0])}[]"
        if origin is dict:
            return f"Record<{ts(get_args(hint)[0])}, {ts(get_args(hint)[1])}>"
        if origin is not None and type(None) in get_args(hint):
            inner = [a for a in get_args(hint) if a is not type(None)]
            return f"{ts(inner[0])} | null"
        return {int: "number", float: "number", bool: "boolean", str: "string"}.get(hint, "unknown")

    lines = ["// Written by `python -m difflab.core.protocol`. Do not edit: change the classes and rerun.", ""]
    names = []
    for cls in EVENTS:
        name = cls.__name__ + "Event"
        names.append(name)
        lines.append(f"export interface {name} {{")
        lines.append(f"  kind: '{cls.KIND}'")
        lines.append("  t: number")
        hints = get_type_hints(cls)
        for f in dataclasses.fields(cls):
            optional = "?" if f.default is not dataclasses.MISSING else ""
            lines.append(f"  {f.name}{optional}: {ts(hints[f.name])}")
        lines.append("}")
        lines.append("")
    lines.append("export type Event =")
    lines += [f"  | {name}" for name in names]
    lines.append("")
    lines.append("export const KINDS = [" + ", ".join(f"'{cls.KIND}'" for cls in EVENTS) + "] as const")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "frontend" / "src" / "events.ts"
    target.write_text(typescript())
    print(f"wrote {target}")
