"""One SQLite file, `output/factory.db`. `sqlite3` plus `difflab.config` and `core.protocol`, and nothing
else from this project, so the server can open it without a model package."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from difflab import config
from difflab.core import protocol

DB_PATH = config.OUTPUT_DIR / "factory.db"

# Three of the four are per-connection settings rather than file ones.
PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    # Off by default, per connection, and without it every REFERENCES below is a comment.
    "PRAGMA foreign_keys = ON",
    "PRAGMA busy_timeout = 5000",  # the worker and the server both write
)

# Version N of the schema is the first N of these. Append, never edit: an entry that has shipped has
# already run against a database somewhere, and editing it changes only the ones made after.
MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        # `id` is the run directory's name, "2026-09-24-230820": the directory holds what is too big
        # for a row (previews, thumbnails, chunks), and the event stream URL and the page hold the
        # same string. `spec` is what was asked for, frozen at enqueue; `production_id` is null for
        # a quick job.
        """
        CREATE TABLE runs (
            id            TEXT    PRIMARY KEY,
            production_id INTEGER,
            asked_by      TEXT    NOT NULL DEFAULT 'person',
            scope         TEXT    NOT NULL DEFAULT '',
            title         TEXT    NOT NULL DEFAULT '',
            spec          TEXT,
            state         TEXT    NOT NULL,
            created_at    REAL    NOT NULL,
            started_at    REAL,
            ended_at      REAL,
            error         TEXT    NOT NULL DEFAULT '',
            pid           INTEGER,
            dir_present   INTEGER NOT NULL DEFAULT 1
        )
        """,
        # One tool applied once. `inputs` and `outputs` are JSON by name; an input may name another
        # job's output (`{"from_job": id, "output": name}`), which is the dependency the worker orders
        # by. `settings` is the resolved values the job ran with, recorded so provenance never points
        # at a profile that has since changed. `for_kind`/`for_id` name the made thing this fills,
        # once there are made things.
        """
        CREATE TABLE jobs (
            id         INTEGER PRIMARY KEY,
            run_id     TEXT    NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            position   INTEGER NOT NULL,
            tool       TEXT    NOT NULL,
            label      TEXT    NOT NULL DEFAULT '',
            for_kind   TEXT,
            for_id     INTEGER,
            inputs     TEXT    NOT NULL DEFAULT '{}',
            outputs    TEXT    NOT NULL DEFAULT '{}',
            key        TEXT,
            settings   TEXT    NOT NULL DEFAULT '{}',
            state      TEXT    NOT NULL,
            started_at REAL,
            ended_at   REAL,
            error      TEXT    NOT NULL DEFAULT '',
            progress   REAL
        )
        """,
        "CREATE INDEX jobs_by_run ON jobs(run_id, position)",
        "CREATE INDEX jobs_by_state ON jobs(state)",
        # A phase of one job, written from the stage events as they arrive: the only thing with a bar.
        """
        CREATE TABLE tasks (
            id         INTEGER PRIMARY KEY,
            job_id     INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            position   INTEGER NOT NULL,
            name       TEXT    NOT NULL,
            state      TEXT    NOT NULL,
            done       INTEGER,
            total      INTEGER,
            unit       TEXT,
            detail     TEXT    NOT NULL DEFAULT '',
            started_at REAL,
            ended_at   REAL
        )
        """,
        "CREATE INDEX tasks_by_job ON tasks(job_id, position)",
        # A file a job delivered. `path` is relative to `config.OUTPUT_DIR`; `key` is set once every
        # artifact is made under one (step 4), and until then names the job's cache entry when it has one.
        """
        CREATE TABLE artifacts (
            id         INTEGER PRIMARY KEY,
            run_id     TEXT    REFERENCES runs(id) ON DELETE SET NULL,
            job_id     INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
            path       TEXT    NOT NULL UNIQUE,
            kind       TEXT    NOT NULL DEFAULT '',
            key        TEXT,
            seed       INTEGER,
            created_at REAL    NOT NULL
        )
        """,
        "CREATE INDEX artifacts_by_run ON artifacts(run_id)",
        "CREATE INDEX artifacts_by_job ON artifacts(job_id)",
        # `id` is the cursor a watching page resumes from, so nothing may ever delete a row: SQLite
        # without AUTOINCREMENT reuses the highest rowid, and a reused id is an event a reconnecting
        # page would skip. CASCADE because an event is part of a run rather than a record of something
        # on disk.
        """
        CREATE TABLE events (
            id      INTEGER PRIMARY KEY,
            run_id  TEXT    NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            job_id  INTEGER,
            task_id INTEGER,
            t       REAL    NOT NULL,
            kind    TEXT    NOT NULL,
            gauge   TEXT,
            data    TEXT    NOT NULL
        )
        """,
        "CREATE INDEX events_run ON events(run_id, id)",
        # `target_kind` is `run` or `job`. `taken` rather than a delete: what the person pressed
        # during a render is part of what happened to it.
        """
        CREATE TABLE commands (
            id          INTEGER PRIMARY KEY,
            target_kind TEXT    NOT NULL,
            target_id   TEXT    NOT NULL,
            t           REAL    NOT NULL,
            data        TEXT    NOT NULL,
            taken       INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX commands_by_target ON commands(target_kind, target_id, taken)",
    ),
    # The screenplay, as far as a chain needs it: a production owns ordered chapters of ordered shots,
    # and a shot owns one keyframe, one prompt and one clip. Those three are made things and share the
    # lifecycle columns (`models.Made`): `artifact` is the current attempt's file, `wanted_key` what
    # its inputs say it should be made of, `current_job` the attempt that made or is making it.
    # `manifest` is JSON, the video and continuity settings the production renders at.
    (
        """
        CREATE TABLE productions (
            id         INTEGER PRIMARY KEY,
            slug       TEXT    NOT NULL UNIQUE,
            title      TEXT    NOT NULL DEFAULT '',
            story_kind TEXT    NOT NULL DEFAULT '',
            story      TEXT    NOT NULL DEFAULT '',
            manifest   TEXT    NOT NULL DEFAULT '{}',
            created_at REAL    NOT NULL,
            updated_at REAL    NOT NULL
        )
        """,
        """
        CREATE TABLE chapters (
            id            INTEGER PRIMARY KEY,
            production_id INTEGER NOT NULL REFERENCES productions(id) ON DELETE CASCADE,
            position      INTEGER NOT NULL,
            title         TEXT    NOT NULL DEFAULT '',
            status        TEXT    NOT NULL DEFAULT ''
        )
        """,
        "CREATE INDEX chapters_by_production ON chapters(production_id, position)",
        """
        CREATE TABLE shots (
            id         INTEGER PRIMARY KEY,
            chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
            position   INTEGER NOT NULL,
            scene      TEXT    NOT NULL DEFAULT '',
            action     TEXT    NOT NULL DEFAULT '',
            camera     TEXT    NOT NULL DEFAULT '',
            dialogue   TEXT    NOT NULL DEFAULT '',
            seconds    REAL    NOT NULL,
            chained    INTEGER NOT NULL DEFAULT 1,
            created_at REAL    NOT NULL,
            updated_at REAL    NOT NULL
        )
        """,
        "CREATE INDEX shots_by_chapter ON shots(chapter_id, position)",
        # `source` is provided (a picture under output/, in `provided`) or chained (the previous
        # shot's handoff frame, which the previous clip's job makes).
        """
        CREATE TABLE keyframes (
            id       INTEGER PRIMARY KEY,
            shot_id  INTEGER NOT NULL UNIQUE REFERENCES shots(id) ON DELETE CASCADE,
            source   TEXT    NOT NULL DEFAULT 'chained',
            note     TEXT    NOT NULL DEFAULT '',
            seed     INTEGER,
            provided TEXT,
            state       TEXT    NOT NULL DEFAULT 'missing',
            wanted_key  TEXT,
            artifact    TEXT,
            current_job INTEGER,
            review      TEXT    NOT NULL DEFAULT '',
            reviewed_by TEXT    NOT NULL DEFAULT '',
            reviewed_at REAL,
            pinned      INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE prompts (
            id         INTEGER PRIMARY KEY,
            shot_id    INTEGER NOT NULL UNIQUE REFERENCES shots(id) ON DELETE CASCADE,
            text       TEXT    NOT NULL DEFAULT '',
            written_by TEXT    NOT NULL DEFAULT 'hand',
            state       TEXT    NOT NULL DEFAULT 'missing',
            wanted_key  TEXT,
            artifact    TEXT,
            current_job INTEGER,
            review      TEXT    NOT NULL DEFAULT '',
            reviewed_by TEXT    NOT NULL DEFAULT '',
            reviewed_at REAL,
            pinned      INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE clips (
            id      INTEGER PRIMARY KEY,
            shot_id INTEGER NOT NULL UNIQUE REFERENCES shots(id) ON DELETE CASCADE,
            seed    INTEGER NOT NULL DEFAULT 0,
            handoff INTEGER,
            state       TEXT    NOT NULL DEFAULT 'missing',
            wanted_key  TEXT,
            artifact    TEXT,
            current_job INTEGER,
            review      TEXT    NOT NULL DEFAULT '',
            reviewed_by TEXT    NOT NULL DEFAULT '',
            reviewed_at REAL,
            pinned      INTEGER NOT NULL DEFAULT 0
        )
        """,
        "ALTER TABLE artifacts ADD COLUMN present INTEGER NOT NULL DEFAULT 1",
        "CREATE INDEX artifacts_by_key ON artifacts(key)",
    ),
    # The rest of the screenplay. A member is a fact about one production's world: `identity` is its
    # merged card (traits, lights, photographed), `voice` is for the audio to come. A reference is a
    # picture of a member - cut from a photo by the boxes the survey found, or given - and a made
    # thing; `refs` because REFERENCES is SQL. A shot's `plan` is the director's chunk: purpose,
    # cast, beats, camera, mouths, closing, dialogue, and the writer's prose. A prompt's `pictures`
    # are the references its clip is handed, in presentation order, so the caption's labels and the
    # clip job agree by construction. Settings are overrides of a tool's code defaults, for this
    # machine (`scope_kind = 'machine'`, empty scope) or one production.
    (
        """
        CREATE TABLE members (
            id            INTEGER PRIMARY KEY,
            production_id INTEGER NOT NULL REFERENCES productions(id) ON DELETE CASCADE,
            slug          TEXT    NOT NULL,
            kind          TEXT    NOT NULL DEFAULT '',
            noun          TEXT    NOT NULL DEFAULT '',
            possessive    TEXT    NOT NULL DEFAULT '',
            find          TEXT    NOT NULL DEFAULT '',
            identity      TEXT    NOT NULL DEFAULT '{}',
            voice         TEXT    NOT NULL DEFAULT '{}',
            position      INTEGER NOT NULL DEFAULT 0,
            UNIQUE (production_id, slug)
        )
        """,
        """
        CREATE TABLE refs (
            id        INTEGER PRIMARY KEY,
            member_id INTEGER NOT NULL REFERENCES members(id) ON DELETE CASCADE,
            role      TEXT    NOT NULL DEFAULT '',
            photo     TEXT    NOT NULL DEFAULT '',
            box       TEXT    NOT NULL DEFAULT '[]',
            size      TEXT    NOT NULL DEFAULT '[]',
            rows      INTEGER NOT NULL DEFAULT 0,
            shows     TEXT    NOT NULL DEFAULT '[]',
            quality   REAL    NOT NULL DEFAULT 0,
            provided  TEXT,
            state       TEXT    NOT NULL DEFAULT 'missing',
            wanted_key  TEXT,
            artifact    TEXT,
            current_job INTEGER,
            review      TEXT    NOT NULL DEFAULT '',
            reviewed_by TEXT    NOT NULL DEFAULT '',
            reviewed_at REAL,
            pinned      INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX refs_by_member ON refs(member_id)",
        """
        CREATE TABLE plates (
            id          INTEGER PRIMARY KEY,
            ref_id      INTEGER NOT NULL REFERENCES refs(id) ON DELETE CASCADE,
            instruction TEXT    NOT NULL DEFAULT '',
            region      TEXT    NOT NULL DEFAULT '[]',
            seed        INTEGER,
            state       TEXT    NOT NULL DEFAULT 'missing',
            wanted_key  TEXT,
            artifact    TEXT,
            current_job INTEGER,
            review      TEXT    NOT NULL DEFAULT '',
            reviewed_by TEXT    NOT NULL DEFAULT '',
            reviewed_at REAL,
            pinned      INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE conflicts (
            id              INTEGER PRIMARY KEY,
            member_id       INTEGER NOT NULL REFERENCES members(id) ON DELETE CASCADE,
            novel_says      TEXT    NOT NULL DEFAULT '',
            reference_shows TEXT    NOT NULL DEFAULT '',
            resolution      TEXT    NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE TABLE audio (
            id       INTEGER PRIMARY KEY,
            shot_id  INTEGER NOT NULL UNIQUE REFERENCES shots(id) ON DELETE CASCADE,
            lines    TEXT    NOT NULL DEFAULT '[]',
            voice    TEXT    NOT NULL DEFAULT '',
            duration REAL,
            state       TEXT    NOT NULL DEFAULT 'missing',
            wanted_key  TEXT,
            artifact    TEXT,
            current_job INTEGER,
            review      TEXT    NOT NULL DEFAULT '',
            reviewed_by TEXT    NOT NULL DEFAULT '',
            reviewed_at REAL,
            pinned      INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE settings (
            id         INTEGER PRIMARY KEY,
            scope_kind TEXT    NOT NULL,
            scope_id   TEXT    NOT NULL DEFAULT '',
            tool       TEXT    NOT NULL,
            overrides  TEXT    NOT NULL DEFAULT '{}',
            version    INTEGER NOT NULL DEFAULT 1,
            updated_at REAL    NOT NULL,
            UNIQUE (scope_kind, scope_id, tool)
        )
        """,
        "ALTER TABLE shots ADD COLUMN plan TEXT NOT NULL DEFAULT '{}'",
        "ALTER TABLE prompts ADD COLUMN pictures TEXT NOT NULL DEFAULT '[]'",
        "ALTER TABLE prompts ADD COLUMN checks TEXT NOT NULL DEFAULT '[]'",
    ),
    # A chapter planned from its own text: a story split on its headings.
    ("ALTER TABLE chapters ADD COLUMN story TEXT NOT NULL DEFAULT ''",),
    # A member or a shot edited by hand is pinned, as a prompt written by hand is: a re-plan's survey
    # and tidy leave a pinned member as it is, and its plan_shots leaves a pinned shot's plan.
    (
        "ALTER TABLE members ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE shots ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
    ),
    # A reference the editor drew from its member's words (role 'generated') keeps the seed it was drawn
    # with, so a member edited in words is drawn again the same way and only a regenerate moves it.
    ("ALTER TABLE refs ADD COLUMN seed INTEGER",),
    # Every round of the director rewriting a production's idea, kept so any version can be adopted
    # later. `story` stays empty until the round's job writes it; `revises` is the round it revised.
    (
        """
        CREATE TABLE ideas (
            id            INTEGER PRIMARY KEY,
            production_id INTEGER NOT NULL REFERENCES productions(id) ON DELETE CASCADE,
            round         INTEGER NOT NULL,
            source        TEXT    NOT NULL DEFAULT '',
            feedback      TEXT    NOT NULL DEFAULT '',
            revises       INTEGER,
            story         TEXT    NOT NULL DEFAULT '',
            changes       TEXT    NOT NULL DEFAULT '[]',
            questions     TEXT    NOT NULL DEFAULT '[]',
            run_id        TEXT,
            job_id        INTEGER,
            adopted_at    REAL,
            created_at    REAL    NOT NULL,
            UNIQUE (production_id, round)
        )
        """,
    ),
    # Models chosen for a role on the settings page, offered again until forgotten. A role's code
    # default is never a row: it is always offered.
    (
        """
        CREATE TABLE model_choices (
            role    TEXT NOT NULL,
            value   TEXT NOT NULL,
            used_at REAL NOT NULL,
            PRIMARY KEY (role, value)
        )
        """,
    ),
    # A composed or provided keyframe the shot before it ends on, as its last frame, so the join does
    # not jump. Off, that shot ends where the model takes it and this one opens fresh, without its tail.
    ("ALTER TABLE keyframes ADD COLUMN landed INTEGER NOT NULL DEFAULT 1",),
    # Access tokens set on the settings page, which win over the environment. `account` is who the
    # service said the token belongs to when it was saved; the page never gets the value back.
    (
        """
        CREATE TABLE tokens (
            service  TEXT PRIMARY KEY,
            value    TEXT NOT NULL,
            account  TEXT NOT NULL DEFAULT '',
            saved_at REAL NOT NULL
        )
        """,
    ),
    # Never written or never read since they shipped.
    (
        "DROP TABLE plates",
        "DROP TABLE audio",
        "ALTER TABLE members DROP COLUMN voice",
        "ALTER TABLE conflicts DROP COLUMN reference_shows",
        "ALTER TABLE runs DROP COLUMN dir_present",
    ),
    ("ALTER TABLE jobs DROP COLUMN progress",),
    # Who is in each shot, one row a member, and how they look there. `parent` is the look it starts from:
    # NULL the one it last had, 'base' the bible's, or a casting id. `amend` set makes the casting a look
    # of its own, drawn by the editor from its parent's picture; the made columns are that drawing, `size`
    # its pixels and `seed` the editor's. Backfilled from each plan's `cast` and `looks`, which then goes.
    # `prompts.written_at` and `written_for` say when and why a prompt was last written.
    (
        """
        CREATE TABLE castings (
            id          INTEGER PRIMARY KEY,
            shot_id     INTEGER NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
            member_id   INTEGER NOT NULL REFERENCES members(id) ON DELETE CASCADE,
            position    INTEGER NOT NULL,
            parent      TEXT,
            amend       TEXT    NOT NULL DEFAULT '',
            label       TEXT    NOT NULL DEFAULT '',
            origin      TEXT    NOT NULL DEFAULT '',
            size        TEXT    NOT NULL DEFAULT '[]',
            seed        INTEGER,
            state       TEXT    NOT NULL DEFAULT 'missing',
            wanted_key  TEXT,
            artifact    TEXT,
            current_job INTEGER,
            review      TEXT    NOT NULL DEFAULT '',
            reviewed_by TEXT    NOT NULL DEFAULT '',
            reviewed_at REAL,
            pinned      INTEGER NOT NULL DEFAULT 0,
            UNIQUE (shot_id, member_id)
        )
        """,
        "CREATE INDEX castings_by_shot ON castings(shot_id, position)",
        """
        INSERT INTO castings (shot_id, member_id, position, amend, origin)
        SELECT shots.id, members.id, listed.key,
               coalesce(json_extract(shots.plan, '$.looks."' || listed.value || '"'), ''),
               CASE WHEN coalesce(json_extract(shots.plan, '$.looks."' || listed.value || '"'), '') != ''
                    THEN 'from text · ch ' || (chapters.position + 1) ELSE '' END
        FROM shots
        JOIN chapters ON chapters.id = shots.chapter_id
        JOIN json_each(shots.plan, '$.cast') AS listed
        JOIN members ON members.production_id = chapters.production_id AND members.slug = listed.value
        """,
        "UPDATE shots SET plan = json_remove(plan, '$.looks') WHERE json_type(plan, '$.looks') IS NOT NULL",
        "ALTER TABLE prompts ADD COLUMN written_at REAL",
        "ALTER TABLE prompts ADD COLUMN written_for TEXT NOT NULL DEFAULT ''",
    ),
    # The bible's own words per member: what the story calls it, its appearance as a person wrote it ('' is
    # its traits), what may change between shots, its voice, and whether its photographs are handed over
    # on a flat backdrop (`refs.plate`, the plated picture). A conflict with `reference_shows` set is the
    # story against a photograph: `lines` the story's sentences it rewrites, `novel_trait` the card it
    # takes to keep the story. Each member's voice is the first one a dialogue line gave it.
    (
        "ALTER TABLE members ADD COLUMN aliases TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE members ADD COLUMN appearance TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE members ADD COLUMN allowed TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE members ADD COLUMN voice TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE members ADD COLUMN delivery TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE members ADD COLUMN reference_audio TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE members ADD COLUMN plate INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE refs ADD COLUMN plate TEXT",
        "ALTER TABLE conflicts ADD COLUMN reference_shows TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE conflicts ADD COLUMN lines TEXT NOT NULL DEFAULT '[]'",
        "ALTER TABLE conflicts ADD COLUMN novel_trait TEXT NOT NULL DEFAULT '{}'",
        """
        UPDATE members SET voice = coalesce((
            SELECT json_extract(said.value, '$.voice')
            FROM shots
            JOIN chapters ON chapters.id = shots.chapter_id
            JOIN json_each(shots.plan, '$.dialogue') AS said
            WHERE chapters.production_id = members.production_id
              AND json_extract(said.value, '$.who') = members.slug
              AND coalesce(json_extract(said.value, '$.voice'), '') != ''
            ORDER BY chapters.position, shots.position LIMIT 1
        ), '')
        """,
    ),
    # Garments a person or animal can change into, each {name, path}: handed to the look the plan names
    # after one, never as a picture of the member itself.
    ("ALTER TABLE members ADD COLUMN wardrobe TEXT NOT NULL DEFAULT '[]'",),
)

SCHEMA_VERSION = len(MIGRATIONS)


def gauge_of(kind: str, data: Mapping[str, Any]) -> str | None:
    """Which rows supersede one another: `core.protocol` says, per event class."""
    return protocol.gauge(kind, dict(data))


class _Held(threading.local):
    def __init__(self) -> None:
        self.by_path: dict[Path, sqlite3.Connection] = {}


# One connection per thread for the thread's life. A connection per call deadlocked the process inside
# SQLite 3.51.0's unix VFS: a dozen threads opening and closing the same file, every close trying a WAL
# checkpoint, met on `unixLock` and `findReusableFd` and none came back.
_held = _Held()


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    target = Path(path) if path is not None else DB_PATH
    connection = _held.by_path.get(target)
    if connection is None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Autocommit. sqlite3's legacy implicit transactions begin somewhere between the statements you
        # wrote, which is a bad way to reason about a file two threads share; here anything that must be
        # atomic says so by opening `transaction`.
        connection = sqlite3.connect(target, isolation_level=None)
        connection.row_factory = sqlite3.Row
        for pragma in PRAGMAS:
            connection.execute(pragma)
        migrate(connection)
        _held.by_path[target] = connection
    yield connection


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """IMMEDIATE rather than DEFERRED: under WAL a read-then-write lock upgrade that loses the race
    returns SQLITE_BUSY at once, and `busy_timeout` cannot retry it."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def migrate(connection: sqlite3.Connection) -> int:
    """Returns the version it was at, not the one it is now."""
    was = version(connection)
    if was > SCHEMA_VERSION:
        raise RuntimeError(
            f"{DB_PATH.name} is at schema version {was} and this difflab only knows {SCHEMA_VERSION}: "
            "it was written by a newer one, and opening it here would ignore columns it is relying on"
        )
    if was == SCHEMA_VERSION:
        return was
    with transaction(connection):
        for step in MIGRATIONS[was:]:
            for statement in step:
                connection.execute(statement)
        # PRAGMA takes no bound parameter. This is `len(MIGRATIONS)`, never anything a caller reaches.
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return was


def version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])
