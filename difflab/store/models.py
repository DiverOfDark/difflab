"""A row, once it is out of the database: frozen, because it is an answer about a moment.

`Run.state` holds what was recorded and never what is true now. A worker killed by the OOM reaper
writes no end, so its row says `running` for ever; liveness is the reader's call against the log's
own last line, and that rule lives in `web/runs.py`.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from difflab import config

QUEUED = "queued"
BLOCKED = "blocked"
RUNNING = "running"
PAUSED = "paused"
DONE = "done"
SKIPPED = "skipped"
FAILED = "failed"
CANCELLED = "cancelled"
RUN_STATES = (QUEUED, RUNNING, PAUSED, DONE, FAILED, CANCELLED)
JOB_STATES = (QUEUED, BLOCKED, RUNNING, DONE, SKIPPED, FAILED, CANCELLED)
SETTLED = (DONE, SKIPPED, FAILED, CANCELLED)

VIDEO = "video"
IMAGE = "image"
TEXT = "text"
KINDS = (VIDEO, IMAGE, TEXT)

# A made thing's lifecycle: `missing` until a run picks it up, `made` once a job's
# output is its current attempt, `stale` once an input moved its wanted key.
MISSING = "missing"
MADE = "made"
NEEDS_REVIEW = "needs_review"
APPROVED = "approved"
STALE = "stale"
PINNED = "pinned"
MADE_STATES = (MISSING, QUEUED, RUNNING, MADE, FAILED, NEEDS_REVIEW, APPROVED, STALE, PINNED)
MADE_KINDS = ("keyframe", "prompt", "clip", "reference", "casting")
# Who gave a made thing's last verdict (`reviewed_by`). A person's approval of an artifact outlives a re-render
# that lands the same key, and the judge does not overrule it.
JUDGE = "judge"
PERSON = "person"

PROVIDED = "provided"  # a keyframe someone gave: a file under output/
CHAINED = "chained"  # a keyframe that is the previous shot's handoff frame
COMPOSED = "composed"  # a keyframe the editor makes from the plan's words over a reference of the room
GENERATED = "generated"  # a reference's role when the editor drew it from its member's words
REFS = "refs"  # a keyframe that is no picture: the clip opens wherever its references take it
BASE = "base"  # a casting's parent when it starts from the bible's look, whatever came since


@dataclass(frozen=True)
class Run:
    id: str
    production_id: int | None
    asked_by: str
    scope: str
    title: str
    spec: str | None
    state: str
    created_at: float
    started_at: float | None
    ended_at: float | None
    error: str
    pid: int | None

    @property
    def settled(self) -> bool:
        return self.state in (DONE, FAILED, CANCELLED)

    @property
    def directory(self) -> Path:
        return config.RUNS_DIR / self.id

    def data(self) -> dict[str, Any] | None:
        if self.spec is None:
            return None
        parsed: Any = json.loads(self.spec)
        return dict(parsed)

    @classmethod
    def of(cls, row: sqlite3.Row) -> Run:
        return cls(
            id=_text(row["id"]),
            production_id=_whole(row["production_id"]),
            asked_by=_text(row["asked_by"]),
            scope=_text(row["scope"]),
            title=_text(row["title"]),
            spec=None if row["spec"] is None else _text(row["spec"]),
            state=_text(row["state"]),
            created_at=_number(row["created_at"]),
            started_at=_maybe_number(row["started_at"]),
            ended_at=_maybe_number(row["ended_at"]),
            error=_text(row["error"]),
            pid=_whole(row["pid"]),
        )


@dataclass(frozen=True)
class Job:
    """One tool applied once. `inputs` name values or another job's output; `outputs` name what it
    made; `settings` are the resolved values it ran with."""

    id: int
    run_id: str
    position: int
    tool: str
    label: str
    for_kind: str | None
    for_id: int | None
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    key: str | None
    settings: dict[str, Any]
    state: str
    started_at: float | None
    ended_at: float | None
    error: str

    @property
    def settled(self) -> bool:
        return self.state in SETTLED

    def depends_on(self) -> set[int]:
        """The jobs whose outputs this one's inputs name."""
        found: set[int] = set()
        values = list(self.inputs.values())
        while values:
            value = values.pop()
            if isinstance(value, dict) and "from_job" in value:
                found.add(int(value["from_job"]))
            elif isinstance(value, list):
                values += value
        return found

    @classmethod
    def of(cls, row: sqlite3.Row) -> Job:
        return cls(
            id=int(row["id"]),
            run_id=_text(row["run_id"]),
            position=int(row["position"]),
            tool=_text(row["tool"]),
            label=_text(row["label"]),
            for_kind=None if row["for_kind"] is None else _text(row["for_kind"]),
            for_id=_whole(row["for_id"]),
            inputs=_json(row["inputs"]),
            outputs=_json(row["outputs"]),
            key=None if row["key"] is None else _text(row["key"]),
            settings=_json(row["settings"]),
            state=_text(row["state"]),
            started_at=_maybe_number(row["started_at"]),
            ended_at=_maybe_number(row["ended_at"]),
            error=_text(row["error"]),
        )


@dataclass(frozen=True)
class Task:
    id: int
    job_id: int
    position: int
    name: str
    state: str
    done: int | None
    total: int | None
    unit: str | None
    detail: str
    started_at: float | None
    ended_at: float | None

    @classmethod
    def of(cls, row: sqlite3.Row) -> Task:
        return cls(
            id=int(row["id"]),
            job_id=int(row["job_id"]),
            position=int(row["position"]),
            name=_text(row["name"]),
            state=_text(row["state"]),
            done=_whole(row["done"]),
            total=_whole(row["total"]),
            unit=None if row["unit"] is None else _text(row["unit"]),
            detail=_text(row["detail"]),
            started_at=_maybe_number(row["started_at"]),
            ended_at=_maybe_number(row["ended_at"]),
        )


@dataclass(frozen=True)
class Artifact:
    """`path` is relative to `config.OUTPUT_DIR` - see `repo.relative`. `present` is the row's word;
    the file is the truth, and `exists` asks it."""

    id: int
    run_id: str | None
    job_id: int | None
    path: str
    kind: str
    key: str | None
    seed: int | None
    created_at: float
    present: bool = True

    @property
    def exists(self) -> bool:
        return self.file.is_file()

    @property
    def file(self) -> Path:
        return config.OUTPUT_DIR / self.path

    @classmethod
    def of(cls, row: sqlite3.Row) -> Artifact:
        return cls(
            id=int(row["id"]),
            run_id=None if row["run_id"] is None else _text(row["run_id"]),
            job_id=_whole(row["job_id"]),
            path=_text(row["path"]),
            kind=_text(row["kind"]),
            key=None if row["key"] is None else _text(row["key"]),
            seed=_whole(row["seed"]),
            created_at=_number(row["created_at"]),
            present=bool(row["present"]) if "present" in row.keys() else True,  # noqa: SIM118 - a Row is not a dict
        )


@dataclass(frozen=True, kw_only=True)
class Production:
    """`manifest` holds the settings the production renders at: `video` and `continuity`, as data."""

    id: int
    slug: str
    title: str
    story_kind: str
    story: str
    manifest: dict[str, Any]
    created_at: float
    updated_at: float

    @classmethod
    def of(cls, row: sqlite3.Row) -> Production:
        return cls(
            id=int(row["id"]),
            slug=_text(row["slug"]),
            title=_text(row["title"]),
            story_kind=_text(row["story_kind"]),
            story=_text(row["story"]),
            manifest=_json(row["manifest"]),
            created_at=_number(row["created_at"]),
            updated_at=_number(row["updated_at"]),
        )


@dataclass(frozen=True, kw_only=True)
class Chapter:
    id: int
    production_id: int
    position: int
    title: str
    status: str
    story: str  # its own text to plan from; empty means the production's

    @classmethod
    def of(cls, row: sqlite3.Row) -> Chapter:
        return cls(
            id=int(row["id"]),
            production_id=int(row["production_id"]),
            position=int(row["position"]),
            title=_text(row["title"]),
            status=_text(row["status"]),
            story=_text(row["story"]) if "story" in row.keys() else "",  # noqa: SIM118 - a Row is not a dict
        )


@dataclass(frozen=True, kw_only=True)
class Shot:
    id: int
    chapter_id: int
    position: int
    scene: str
    action: str
    camera: str
    dialogue: str
    seconds: float
    chained: bool
    created_at: float
    updated_at: float
    plan: dict[str, Any]  # the director's chunk, with the writer's prose once written
    pinned: bool = False

    @classmethod
    def of(cls, row: sqlite3.Row) -> Shot:
        return cls(
            id=int(row["id"]),
            chapter_id=int(row["chapter_id"]),
            position=int(row["position"]),
            scene=_text(row["scene"]),
            action=_text(row["action"]),
            camera=_text(row["camera"]),
            dialogue=_text(row["dialogue"]),
            seconds=_number(row["seconds"]),
            chained=bool(row["chained"]),
            created_at=_number(row["created_at"]),
            updated_at=_number(row["updated_at"]),
            plan=_json(row["plan"]) if "plan" in row.keys() else {},  # noqa: SIM118 - a Row is not a dict
            pinned=bool(row["pinned"]),
        )


@dataclass(frozen=True, kw_only=True)
class Made:
    """What every made thing carries: the lifecycle above. `artifact` is the current
    attempt's file, relative to `config.OUTPUT_DIR`."""

    id: int
    state: str
    wanted_key: str | None
    artifact: str | None
    current_job: int | None
    review: str
    reviewed_by: str
    reviewed_at: float | None
    pinned: bool

    @property
    def present(self) -> bool:
        return self.artifact is not None and (config.OUTPUT_DIR / self.artifact).is_file()

    @property
    def file(self) -> Path | None:
        return None if self.artifact is None else config.OUTPUT_DIR / self.artifact


def _made(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "state": _text(row["state"]),
        "wanted_key": None if row["wanted_key"] is None else _text(row["wanted_key"]),
        "artifact": None if row["artifact"] is None else _text(row["artifact"]),
        "current_job": _whole(row["current_job"]),
        "review": _text(row["review"]),
        "reviewed_by": _text(row["reviewed_by"]),
        "reviewed_at": _maybe_number(row["reviewed_at"]),
        "pinned": bool(row["pinned"]),
    }


@dataclass(frozen=True, kw_only=True)
class Keyframe(Made):
    shot_id: int
    source: str
    note: str
    seed: int | None
    provided: str | None
    landed: bool  # the shot before ends on it; meaningless for a chained one

    @classmethod
    def of(cls, row: sqlite3.Row) -> Keyframe:
        return cls(
            **_made(row),
            shot_id=int(row["shot_id"]),
            source=_text(row["source"]),
            note=_text(row["note"]),
            seed=_whole(row["seed"]),
            provided=None if row["provided"] is None else _text(row["provided"]),
            landed=bool(row["landed"]),
        )


@dataclass(frozen=True, kw_only=True)
class Prompt(Made):
    shot_id: int
    text: str
    written_by: str
    pictures: list[str]  # the references the clip is handed, in presentation order, relative to output/
    checks: list[dict[str, Any]]  # the gate's findings on the compiled caption
    written_at: float | None = None
    written_for: str = ""  # "looks" when the looks of its shot moved under it, "" otherwise

    @classmethod
    def of(cls, row: sqlite3.Row) -> Prompt:
        keys = row.keys()
        return cls(
            **_made(row),
            shot_id=int(row["shot_id"]),
            text=_text(row["text"]),
            written_by=_text(row["written_by"]),
            pictures=[str(p) for p in _list(row["pictures"])] if "pictures" in keys else [],
            checks=[dict(c) for c in _list(row["checks"])] if "checks" in keys else [],
            written_at=_maybe_number(row["written_at"]) if "written_at" in keys else None,
            written_for=_text(row["written_for"]) if "written_for" in keys else "",
        )


@dataclass(frozen=True, kw_only=True)
class Clip(Made):
    shot_id: int
    seed: int
    handoff: int | None

    @classmethod
    def of(cls, row: sqlite3.Row) -> Clip:
        return cls(
            **_made(row), shot_id=int(row["shot_id"]), seed=int(row["seed"] or 0), handoff=_whole(row["handoff"])
        )


@dataclass(frozen=True, kw_only=True)
class Member:
    """`identity` is the merged card: `traits` (part, phrase, short, name), `lights`, `photographed`.
    `appearance` is a person's own words for it, '' while its traits say it; `plate` hands its photographs
    over on a flat backdrop."""

    id: int
    production_id: int
    slug: str
    kind: str
    noun: str
    possessive: str
    find: str
    identity: dict[str, Any]
    position: int
    pinned: bool = False
    aliases: str = ""
    appearance: str = ""
    allowed: str = ""
    voice: str = ""
    delivery: str = ""
    reference_audio: str = ""
    plate: bool = False
    wardrobe: list[dict[str, str]] = field(default_factory=list)

    def picture(self, reference: Reference) -> str | None:
        """The file a reference is handed over as: its plate while the member plates and one is drawn."""
        return reference.plate if self.plate and reference.plate else reference.artifact

    @classmethod
    def of(cls, row: sqlite3.Row) -> Member:
        return cls(
            id=int(row["id"]),
            production_id=int(row["production_id"]),
            slug=_text(row["slug"]),
            kind=_text(row["kind"]),
            noun=_text(row["noun"]),
            possessive=_text(row["possessive"]),
            find=_text(row["find"]),
            identity=_json(row["identity"]),
            position=int(row["position"] or 0),
            pinned=bool(row["pinned"]),
            aliases=_text(row["aliases"]),
            appearance=_text(row["appearance"]),
            allowed=_text(row["allowed"]),
            voice=_text(row["voice"]),
            delivery=_text(row["delivery"]),
            reference_audio=_text(row["reference_audio"]),
            plate=bool(row["plate"]),
            wardrobe=[{"name": str(g["name"]), "path": str(g["path"])} for g in _list(row["wardrobe"])],
        )


@dataclass(frozen=True, kw_only=True)
class Reference(Made):
    """A picture of a member: cut from `photo` at `box` (pixels) and resized to `size`, or `provided`.
    `shows` names every member in it, its own first; `rows` is what holding it costs a chunk."""

    member_id: int
    role: str
    photo: str
    box: list[int]
    size: list[int]
    rows: int
    shows: list[str]
    quality: float
    provided: str | None
    seed: int | None = None
    plate: str | None = None  # the picture on a flat backdrop, handed over in its place while its member plates

    @property
    def generated(self) -> bool:
        return self.role == GENERATED

    @classmethod
    def of(cls, row: sqlite3.Row) -> Reference:
        return cls(
            **_made(row),
            member_id=int(row["member_id"]),
            role=_text(row["role"]),
            photo=_text(row["photo"]),
            box=[int(v) for v in _list(row["box"])],
            size=[int(v) for v in _list(row["size"])],
            rows=int(row["rows"] or 0),
            shows=[str(v) for v in _list(row["shows"])],
            quality=_number(row["quality"]),
            provided=None if row["provided"] is None else _text(row["provided"]),
            seed=_whole(row["seed"]),
            plate=None if row["plate"] is None else _text(row["plate"]),
        )


@dataclass(frozen=True, kw_only=True)
class Casting(Made):
    """A member in a shot; with `amend`, a look of its own whose picture the made columns are."""

    shot_id: int
    member_id: int
    position: int
    parent: str | None
    amend: str
    label: str
    origin: str
    size: list[int]
    seed: int | None

    @property
    def amended(self) -> bool:
        return bool(self.amend.strip())

    @classmethod
    def of(cls, row: sqlite3.Row) -> Casting:
        return cls(
            **_made(row),
            shot_id=int(row["shot_id"]),
            member_id=int(row["member_id"]),
            position=int(row["position"]),
            parent=None if row["parent"] is None else _text(row["parent"]),
            amend=_text(row["amend"]),
            label=_text(row["label"]),
            origin=_text(row["origin"]),
            size=[int(v) for v in _list(row["size"])],
            seed=_whole(row["seed"]),
        )


@dataclass(frozen=True, kw_only=True)
class Conflict:
    """Two photographs disagreeing (`reference_shows` empty), or the story against a photograph: `lines`
    are [sentence, rewritten] pairs that keep the photograph, `novel_trait` the trait that keeps the story."""

    id: int
    member_id: int
    novel_says: str
    resolution: str
    reference_shows: str = ""
    lines: list[list[str]] = field(default_factory=list)
    novel_trait: dict[str, Any] = field(default_factory=dict)

    @property
    def textual(self) -> bool:
        return bool(self.reference_shows)

    @classmethod
    def of(cls, row: sqlite3.Row) -> Conflict:
        return cls(
            id=int(row["id"]),
            member_id=int(row["member_id"]),
            novel_says=_text(row["novel_says"]),
            resolution=_text(row["resolution"]),
            reference_shows=_text(row["reference_shows"]),
            lines=[[str(v) for v in pair] for pair in _list(row["lines"])],
            novel_trait=_json(row["novel_trait"]),
        )


@dataclass(frozen=True)
class Idea:
    """One round of the director rewriting a production's idea into a story."""

    id: int
    production_id: int
    round: int
    source: str
    feedback: str
    revises: int | None
    story: str
    changes: list[str]
    questions: list[str]
    run_id: str
    job_id: int | None
    adopted_at: float | None
    created_at: float

    @classmethod
    def of(cls, row: sqlite3.Row) -> Idea:
        return cls(
            id=int(row["id"]),
            production_id=int(row["production_id"]),
            round=int(row["round"]),
            source=_text(row["source"]),
            feedback=_text(row["feedback"]),
            revises=_whole(row["revises"]),
            story=_text(row["story"]),
            changes=[str(c) for c in _list(row["changes"])],
            questions=[str(q) for q in _list(row["questions"])],
            run_id=_text(row["run_id"]),
            job_id=_whole(row["job_id"]),
            adopted_at=_maybe_number(row["adopted_at"]),
            created_at=_number(row["created_at"]),
        )


@dataclass(frozen=True, kw_only=True)
class Setting:
    """Overrides of one tool's code defaults, for this machine (`scope_kind = "machine"`, empty
    `scope_id`) or one production (`scope_id` its id as text)."""

    id: int
    scope_kind: str
    scope_id: str
    tool: str
    overrides: dict[str, Any]
    version: int
    updated_at: float

    @classmethod
    def of(cls, row: sqlite3.Row) -> Setting:
        return cls(
            id=int(row["id"]),
            scope_kind=_text(row["scope_kind"]),
            scope_id=_text(row["scope_id"]),
            tool=_text(row["tool"]),
            overrides=_json(row["overrides"]),
            version=int(row["version"] or 1),
            updated_at=_number(row["updated_at"]),
        )


@dataclass(frozen=True)
class Event:
    """`data` is the event minus the columns beside it; `gauge` is the key it supersedes under, and
    NULL means history."""

    id: int
    run_id: str
    job_id: int | None
    task_id: int | None
    t: float
    kind: str
    gauge: str | None
    data: str

    def value(self) -> dict[str, Any]:
        parsed: Any = json.loads(self.data)
        return dict(parsed)

    def line(self) -> str:
        """The object a page folds, with `t`, `kind` and the job back in it."""
        return json.dumps({"t": self.t, "kind": self.kind, "job": self.job_id, **self.value()}, ensure_ascii=False)

    @classmethod
    def of(cls, row: sqlite3.Row) -> Event:
        return cls(
            id=int(row["id"]),
            run_id=_text(row["run_id"]),
            job_id=_whole(row["job_id"]),
            task_id=_whole(row["task_id"]),
            t=_number(row["t"]),
            kind=_text(row["kind"]),
            gauge=None if row["gauge"] is None else _text(row["gauge"]),
            data=_text(row["data"]),
        )


@dataclass(frozen=True)
class Command:
    """Something a page asked a run or a job to do. `taken` is set by the worker that picked it up."""

    id: int
    target_kind: str
    target_id: str
    t: float
    data: str
    taken: bool

    def value(self) -> dict[str, Any]:
        parsed: Any = json.loads(self.data)
        return {**dict(parsed), "t": self.t}

    @classmethod
    def of(cls, row: sqlite3.Row) -> Command:
        return cls(
            id=int(row["id"]),
            target_kind=_text(row["target_kind"]),
            target_id=_text(row["target_id"]),
            t=_number(row["t"]),
            data=_text(row["data"]),
            taken=bool(row["taken"]),
        )


# SQLite stores whatever it was given; converting on the way out is what makes the types true.
def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _number(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _maybe_number(value: Any) -> float | None:
    return None if value is None else float(value)


def _whole(value: Any) -> int | None:
    return None if value is None else int(value)


def _json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    parsed: Any = json.loads(str(value))
    return dict(parsed)


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    parsed: Any = json.loads(str(value))
    return list(parsed)
