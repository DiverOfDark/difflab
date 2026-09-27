"""The screenplay's rows: productions, chapters, shots, and the made things a shot owns."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from difflab.store.models import (
    CHAINED,
    COMPOSED,
    GENERATED,
    MISSING,
    PINNED,
    PROVIDED,
    Audio,
    Chapter,
    Clip,
    Conflict,
    Idea,
    Keyframe,
    Member,
    Plate,
    Production,
    Prompt,
    Reference,
    Setting,
    Shot,
)

MADE_TABLES = {
    "keyframe": "keyframes",
    "prompt": "prompts",
    "clip": "clips",
    "audio": "audio",
    "reference": "refs",
    "plate": "plates",
}


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


# --- productions ------------------------------------------------------------------------------
def productions(connection: sqlite3.Connection) -> list[Production]:
    rows = connection.execute("SELECT * FROM productions ORDER BY updated_at DESC, id DESC").fetchall()
    return [Production.of(row) for row in rows]


def production(connection: sqlite3.Connection, production_id: int) -> Production | None:
    row = connection.execute("SELECT * FROM productions WHERE id = ?", (production_id,)).fetchone()
    return None if row is None else Production.of(row)


def production_by_slug(connection: sqlite3.Connection, slug: str) -> Production | None:
    row = connection.execute("SELECT * FROM productions WHERE slug = ?", (slug,)).fetchone()
    return None if row is None else Production.of(row)


def add_production(
    connection: sqlite3.Connection,
    slug: str,
    title: str = "",
    manifest: dict[str, Any] | None = None,
    story_kind: str = "",
    story: str = "",
) -> Production:
    """A production with its first chapter: there is always one to put a shot in."""
    now = time.time()
    cursor = connection.execute(
        "INSERT INTO productions (slug, title, story_kind, story, manifest, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (slug, title or slug, story_kind, story, _dump(manifest or {}), now, now),
    )
    made = production(connection, int(cursor.lastrowid or 0))
    assert made is not None
    add_chapter(connection, made.id)
    return made


def update_production(connection: sqlite3.Connection, production_id: int, **fields: Any) -> Production | None:
    allowed = {"slug", "title", "story_kind", "story", "manifest"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"a production has no {', '.join(sorted(unknown))}")
    values = {name: (_dump(value) if name == "manifest" else value) for name, value in fields.items()}
    values["updated_at"] = time.time()
    assignments = ", ".join(f"{name} = ?" for name in values)
    connection.execute(f"UPDATE productions SET {assignments} WHERE id = ?", (*values.values(), production_id))
    return production(connection, production_id)


def touch_production(connection: sqlite3.Connection, production_id: int) -> None:
    connection.execute("UPDATE productions SET updated_at = ? WHERE id = ?", (time.time(), production_id))


def delete_production(connection: sqlite3.Connection, production_id: int) -> None:
    connection.execute("DELETE FROM productions WHERE id = ?", (production_id,))


# --- chapters ---------------------------------------------------------------------------------
def chapters_of(connection: sqlite3.Connection, production_id: int) -> list[Chapter]:
    rows = connection.execute(
        "SELECT * FROM chapters WHERE production_id = ? ORDER BY position", (production_id,)
    ).fetchall()
    return [Chapter.of(row) for row in rows]


def chapter(connection: sqlite3.Connection, chapter_id: int) -> Chapter | None:
    row = connection.execute("SELECT * FROM chapters WHERE id = ?", (chapter_id,)).fetchone()
    return None if row is None else Chapter.of(row)


def add_chapter(connection: sqlite3.Connection, production_id: int, title: str = "", story: str = "") -> Chapter:
    row = connection.execute(
        "SELECT coalesce(max(position), -1) + 1 FROM chapters WHERE production_id = ?", (production_id,)
    ).fetchone()
    cursor = connection.execute(
        "INSERT INTO chapters (production_id, position, title, story) VALUES (?, ?, ?, ?)",
        (production_id, int(row[0]), title, story),
    )
    made = chapter(connection, int(cursor.lastrowid or 0))
    assert made is not None
    return made


# --- shots ------------------------------------------------------------------------------------
def shots_of(connection: sqlite3.Connection, chapter_id: int) -> list[Shot]:
    rows = connection.execute("SELECT * FROM shots WHERE chapter_id = ? ORDER BY position", (chapter_id,)).fetchall()
    return [Shot.of(row) for row in rows]


def shot(connection: sqlite3.Connection, shot_id: int) -> Shot | None:
    row = connection.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
    return None if row is None else Shot.of(row)


def add_shot(
    connection: sqlite3.Connection,
    chapter_id: int,
    seconds: float,
    action: str = "",
    scene: str = "",
    camera: str = "",
    dialogue: str = "",
    chained: bool = True,
) -> Shot:
    """A shot at the end of its chapter, with its three made things missing."""
    now = time.time()
    row = connection.execute(
        "SELECT coalesce(max(position), -1) + 1 FROM shots WHERE chapter_id = ?", (chapter_id,)
    ).fetchone()
    cursor = connection.execute(
        "INSERT INTO shots (chapter_id, position, scene, action, camera, dialogue, seconds, chained, created_at, "
        "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (chapter_id, int(row[0]), scene, action, camera, dialogue, seconds, int(chained), now, now),
    )
    shot_id = int(cursor.lastrowid or 0)
    connection.execute("INSERT INTO keyframes (shot_id, source, state) VALUES (?, ?, ?)", (shot_id, CHAINED, MISSING))
    connection.execute("INSERT INTO prompts (shot_id, state) VALUES (?, ?)", (shot_id, MISSING))
    connection.execute("INSERT INTO clips (shot_id, state) VALUES (?, ?)", (shot_id, MISSING))
    connection.execute("INSERT INTO audio (shot_id, state) VALUES (?, ?)", (shot_id, MISSING))
    made = shot(connection, shot_id)
    assert made is not None
    return made


def update_shot(connection: sqlite3.Connection, shot_id: int, **fields: Any) -> Shot | None:
    allowed = {"scene", "action", "camera", "dialogue", "seconds", "chained", "plan", "pinned"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"a shot has no {', '.join(sorted(unknown))}")
    values = {
        name: (int(value) if name in ("chained", "pinned") else _dump(value) if name == "plan" else value)
        for name, value in fields.items()
    }
    values["updated_at"] = time.time()
    assignments = ", ".join(f"{name} = ?" for name in values)
    connection.execute(f"UPDATE shots SET {assignments} WHERE id = ?", (*values.values(), shot_id))
    return shot(connection, shot_id)


def delete_shot(connection: sqlite3.Connection, shot_id: int) -> None:
    """The row and its made things; the shots after it close the gap."""
    gone = shot(connection, shot_id)
    if gone is None:
        return
    connection.execute("DELETE FROM shots WHERE id = ?", (shot_id,))
    connection.execute(
        "UPDATE shots SET position = position - 1 WHERE chapter_id = ? AND position > ?",
        (gone.chapter_id, gone.position),
    )


# --- the made things of a shot ----------------------------------------------------------------
def keyframe_of(connection: sqlite3.Connection, shot_id: int) -> Keyframe:
    row = connection.execute("SELECT * FROM keyframes WHERE shot_id = ?", (shot_id,)).fetchone()
    if row is None:
        raise LookupError(f"shot {shot_id} has no keyframe row")
    return Keyframe.of(row)


def prompt_of(connection: sqlite3.Connection, shot_id: int) -> Prompt:
    row = connection.execute("SELECT * FROM prompts WHERE shot_id = ?", (shot_id,)).fetchone()
    if row is None:
        raise LookupError(f"shot {shot_id} has no prompt row")
    return Prompt.of(row)


def clip_of(connection: sqlite3.Connection, shot_id: int) -> Clip:
    row = connection.execute("SELECT * FROM clips WHERE shot_id = ?", (shot_id,)).fetchone()
    if row is None:
        raise LookupError(f"shot {shot_id} has no clip row")
    return Clip.of(row)


def provide_keyframe(connection: sqlite3.Connection, shot_id: int, path: str, note: str = "") -> Keyframe:
    """A picture someone gave, relative to output/: it is the keyframe, pinned, with nothing to make."""
    connection.execute(
        "UPDATE keyframes SET source = ?, provided = ?, artifact = ?, note = ?, state = ?, pinned = 1, "
        "wanted_key = NULL, current_job = NULL WHERE shot_id = ?",
        (PROVIDED, path, path, note, PINNED, shot_id),
    )
    return keyframe_of(connection, shot_id)


def chain_keyframe(connection: sqlite3.Connection, shot_id: int) -> Keyframe:
    """Back to the previous shot's handoff frame, which the previous clip's job makes."""
    connection.execute(
        "UPDATE keyframes SET source = ?, provided = NULL, artifact = NULL, note = '', state = ?, pinned = 0, "
        "wanted_key = NULL, current_job = NULL WHERE shot_id = ?",
        (CHAINED, MISSING, shot_id),
    )
    return keyframe_of(connection, shot_id)


def compose_keyframe(connection: sqlite3.Connection, shot_id: int, seed: int | None = None) -> Keyframe:
    """The editor will make it from the plan's words over a reference of the room."""
    connection.execute(
        "UPDATE keyframes SET source = ?, provided = NULL, artifact = NULL, note = '', state = ?, pinned = 0, "
        "seed = coalesce(?, seed), wanted_key = NULL, current_job = NULL WHERE shot_id = ?",
        (COMPOSED, MISSING, seed, shot_id),
    )
    return keyframe_of(connection, shot_id)


def set_prompt(connection: sqlite3.Connection, shot_id: int, text: str, written_by: str = "hand") -> Prompt:
    """Text written by hand is pinned: no tool rewrites it."""
    pinned = written_by == "hand"
    connection.execute(
        "UPDATE prompts SET text = ?, written_by = ?, state = ?, pinned = ? WHERE shot_id = ?",
        (text, written_by, PINNED if pinned else MISSING, int(pinned), shot_id),
    )
    return prompt_of(connection, shot_id)


def set_clip_seed(connection: sqlite3.Connection, shot_id: int, seed: int) -> Clip:
    connection.execute("UPDATE clips SET seed = ? WHERE shot_id = ?", (seed, shot_id))
    return clip_of(connection, shot_id)


def set_made(
    connection: sqlite3.Connection,
    kind: str,
    made_id: int,
    state: str,
    wanted_key: str | None = None,
    artifact: str | None = None,
    current_job: int | None = None,
    review: str | None = None,
    reviewed_by: str | None = None,
    **own: Any,
) -> None:
    """One made thing's lifecycle columns, and any of its own (`handoff` on a clip). A column left
    `None` here keeps its value: `state` is the only thing every call means to move."""
    table = MADE_TABLES[kind]
    values: dict[str, Any] = {"state": state}
    if wanted_key is not None:
        values["wanted_key"] = wanted_key
    if artifact is not None:
        values["artifact"] = artifact
    if current_job is not None:
        values["current_job"] = current_job
    if review is not None:
        values["review"], values["reviewed_by"], values["reviewed_at"] = review, reviewed_by or "", time.time()
    values.update(own)
    assignments = ", ".join(f"{name} = ?" for name in values)
    connection.execute(f"UPDATE {table} SET {assignments} WHERE id = ?", (*values.values(), made_id))


def made_by_shot(connection: sqlite3.Connection, kind: str, shot_id: int) -> Keyframe | Prompt | Clip:
    if kind == "keyframe":
        return keyframe_of(connection, shot_id)
    if kind == "prompt":
        return prompt_of(connection, shot_id)
    if kind == "clip":
        return clip_of(connection, shot_id)
    raise ValueError(f"{kind!r} is not a made thing (one of {', '.join(MADE_TABLES)})")


def made_by_id(
    connection: sqlite3.Connection, kind: str, made_id: int
) -> Keyframe | Prompt | Clip | Audio | Reference | Plate | None:
    table = MADE_TABLES[kind]
    row = connection.execute(f"SELECT * FROM {table} WHERE id = ?", (made_id,)).fetchone()
    if row is None:
        return None
    readers: dict[str, Any] = {
        "keyframe": Keyframe.of,
        "prompt": Prompt.of,
        "clip": Clip.of,
        "audio": Audio.of,
        "reference": Reference.of,
        "plate": Plate.of,
    }
    made: Keyframe | Prompt | Clip | Audio | Reference | Plate = readers[kind](row)
    return made


def set_compiled(
    connection: sqlite3.Connection, shot_id: int, text: str, pictures: list[str], checks: list[dict[str, Any]], key: str
) -> Prompt:
    """A caption the compiler wrote from the plan, with the pictures its clip is handed. Not pinned:
    a plan change recompiles it, and a hand edit (`set_prompt`) pins it."""
    connection.execute(
        "UPDATE prompts SET text = ?, written_by = 'compiler', pictures = ?, checks = ?, wanted_key = ?, "
        "state = 'made', pinned = 0 WHERE shot_id = ?",
        (text, _dump(pictures), _dump(checks), key, shot_id),
    )
    return prompt_of(connection, shot_id)


def audio_of(connection: sqlite3.Connection, shot_id: int) -> Audio:
    row = connection.execute("SELECT * FROM audio WHERE shot_id = ?", (shot_id,)).fetchone()
    if row is None:
        raise LookupError(f"shot {shot_id} has no audio row")
    return Audio.of(row)


# --- the cast ---------------------------------------------------------------------------------
def members_of(connection: sqlite3.Connection, production_id: int) -> list[Member]:
    rows = connection.execute(
        "SELECT * FROM members WHERE production_id = ? ORDER BY position, id", (production_id,)
    ).fetchall()
    return [Member.of(row) for row in rows]


def member(connection: sqlite3.Connection, member_id: int) -> Member | None:
    row = connection.execute("SELECT * FROM members WHERE id = ?", (member_id,)).fetchone()
    return None if row is None else Member.of(row)


def member_by_slug(connection: sqlite3.Connection, production_id: int, slug: str) -> Member | None:
    row = connection.execute(
        "SELECT * FROM members WHERE production_id = ? AND slug = ?", (production_id, slug)
    ).fetchone()
    return None if row is None else Member.of(row)


def put_member(
    connection: sqlite3.Connection,
    production_id: int,
    slug: str,
    kind: str,
    noun: str,
    possessive: str,
    find: str,
    position: int | None = None,
) -> Member:
    """The member by slug, made or updated in place: its references and identity survive a re-survey
    that still names it."""
    found = member_by_slug(connection, production_id, slug)
    if found is None:
        row = connection.execute(
            "SELECT coalesce(max(position), -1) + 1 FROM members WHERE production_id = ?", (production_id,)
        ).fetchone()
        connection.execute(
            "INSERT INTO members (production_id, slug, kind, noun, possessive, find, position) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (production_id, slug, kind, noun, possessive, find, int(row[0]) if position is None else position),
        )
    else:
        connection.execute(
            "UPDATE members SET kind = ?, noun = ?, possessive = ?, find = ?, position = coalesce(?, position) "
            "WHERE id = ?",
            (kind, noun, possessive, find, position, found.id),
        )
    made = member_by_slug(connection, production_id, slug)
    assert made is not None
    return made


def set_identity(connection: sqlite3.Connection, member_id: int, identity: dict[str, Any]) -> None:
    connection.execute("UPDATE members SET identity = ? WHERE id = ?", (_dump(identity), member_id))


def update_member(connection: sqlite3.Connection, member_id: int, **fields: Any) -> Member | None:
    allowed = {"kind", "noun", "possessive", "find", "identity", "pinned"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"a member has no {', '.join(sorted(unknown))}")
    values = {
        name: (_dump(value) if name == "identity" else int(value) if name == "pinned" else value)
        for name, value in fields.items()
    }
    assignments = ", ".join(f"{name} = ?" for name in values)
    connection.execute(f"UPDATE members SET {assignments} WHERE id = ?", (*values.values(), member_id))
    return member(connection, member_id)


def delete_member(connection: sqlite3.Connection, member_id: int) -> None:
    connection.execute("DELETE FROM members WHERE id = ?", (member_id,))


def set_cast_order(connection: sqlite3.Connection, production_id: int, slugs: list[str]) -> None:
    for position, slug in enumerate(slugs):
        connection.execute(
            "UPDATE members SET position = ? WHERE production_id = ? AND slug = ?", (position, production_id, slug)
        )


def drop_members_except(connection: sqlite3.Connection, production_id: int, slugs: list[str]) -> None:
    """Members a re-survey no longer names, with their references."""
    kept = ",".join("?" * len(slugs)) or "''"
    connection.execute(f"DELETE FROM members WHERE production_id = ? AND slug NOT IN ({kept})", (production_id, *slugs))


def refs_of(connection: sqlite3.Connection, member_id: int) -> list[Reference]:
    rows = connection.execute("SELECT * FROM refs WHERE member_id = ? ORDER BY id", (member_id,)).fetchall()
    return [Reference.of(row) for row in rows]


def refs_of_production(connection: sqlite3.Connection, production_id: int) -> list[Reference]:
    rows = connection.execute(
        "SELECT refs.* FROM refs JOIN members ON refs.member_id = members.id "
        "WHERE members.production_id = ? ORDER BY refs.id",
        (production_id,),
    ).fetchall()
    return [Reference.of(row) for row in rows]


def ref(connection: sqlite3.Connection, ref_id: int) -> Reference | None:
    row = connection.execute("SELECT * FROM refs WHERE id = ?", (ref_id,)).fetchone()
    return None if row is None else Reference.of(row)


def delete_ref(connection: sqlite3.Connection, ref_id: int) -> None:
    connection.execute("DELETE FROM refs WHERE id = ?", (ref_id,))


def clear_refs(connection: sqlite3.Connection, production_id: int) -> None:
    """Every cut reference of the production; the given and the drawn ones stay."""
    connection.execute(
        "DELETE FROM refs WHERE provided IS NULL AND role != ? "
        "AND member_id IN (SELECT id FROM members WHERE production_id = ?)",
        (GENERATED, production_id),
    )


def generated_ref(connection: sqlite3.Connection, member_id: int) -> Reference | None:
    row = connection.execute(
        "SELECT * FROM refs WHERE member_id = ? AND role = ? ORDER BY id DESC LIMIT 1", (member_id, GENERATED)
    ).fetchone()
    return None if row is None else Reference.of(row)


def put_generated_ref(connection: sqlite3.Connection, member_id: int, seed: int, size: list[int]) -> Reference:
    """The member's one drawn reference, made or reused: the picture it has stays until the next is drawn."""
    found = generated_ref(connection, member_id)
    if found is None:
        connection.execute(
            "INSERT INTO refs (member_id, role, size, rows, shows, seed, state) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (member_id, GENERATED, _dump(size), (size[0] // 32) * (size[1] // 32), "[]", seed, MISSING),
        )
    else:
        connection.execute(
            "UPDATE refs SET size = ?, rows = ?, seed = ? WHERE id = ?",
            (_dump(size), (size[0] // 32) * (size[1] // 32), seed, found.id),
        )
    made = generated_ref(connection, member_id)
    assert made is not None
    return made


def add_ref(
    connection: sqlite3.Connection,
    member_id: int,
    role: str,
    photo: str,
    box: list[int],
    size: list[int],
    rows: int,
    shows: list[str],
    quality: float,
    artifact: str | None = None,
    key: str | None = None,
    provided: str | None = None,
) -> Reference:
    cursor = connection.execute(
        "INSERT INTO refs (member_id, role, photo, box, size, rows, shows, quality, provided, artifact, wanted_key, "
        "state, pinned) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            member_id, role, photo, _dump(box), _dump(size), rows, _dump(shows), quality, provided, artifact, key,
            PINNED if provided else ("made" if artifact else MISSING), int(bool(provided)),
        ),
    )  # fmt: skip
    row = connection.execute("SELECT * FROM refs WHERE id = ?", (int(cursor.lastrowid or 0),)).fetchone()
    return Reference.of(row)


def conflicts_of(connection: sqlite3.Connection, member_id: int) -> list[Conflict]:
    rows = connection.execute("SELECT * FROM conflicts WHERE member_id = ? ORDER BY id", (member_id,)).fetchall()
    return [Conflict.of(row) for row in rows]


def conflict(connection: sqlite3.Connection, conflict_id: int) -> Conflict | None:
    row = connection.execute("SELECT * FROM conflicts WHERE id = ?", (conflict_id,)).fetchone()
    return None if row is None else Conflict.of(row)


def set_conflicts(connection: sqlite3.Connection, member_id: int, notes: list[str]) -> None:
    """What the tidy stage could not reconcile between photographs, one row each, replacing the last. A
    note said again word for word keeps the resolution it was given."""
    resolved = {c.novel_says: c.resolution for c in conflicts_of(connection, member_id) if c.resolution}
    connection.execute("DELETE FROM conflicts WHERE member_id = ?", (member_id,))
    for note in notes:
        connection.execute(
            "INSERT INTO conflicts (member_id, novel_says, resolution) VALUES (?, ?, ?)",
            (member_id, note, resolved.get(note, "")),
        )


def resolve_conflict(connection: sqlite3.Connection, conflict_id: int, resolution: str) -> None:
    connection.execute("UPDATE conflicts SET resolution = ? WHERE id = ?", (resolution, conflict_id))


# --- the idea, rewritten round by round ---------------------------------------------------------
def ideas_of(connection: sqlite3.Connection, production_id: int) -> list[Idea]:
    rows = connection.execute("SELECT * FROM ideas WHERE production_id = ? ORDER BY round", (production_id,)).fetchall()
    return [Idea.of(row) for row in rows]


def idea(connection: sqlite3.Connection, production_id: int, round_: int) -> Idea | None:
    row = connection.execute(
        "SELECT * FROM ideas WHERE production_id = ? AND round = ?", (production_id, round_)
    ).fetchone()
    return None if row is None else Idea.of(row)


def add_idea(
    connection: sqlite3.Connection, production_id: int, source: str, feedback: str, revises: int | None, run_id: str
) -> Idea:
    """The next round, its story still to be written."""
    last = connection.execute("SELECT max(round) FROM ideas WHERE production_id = ?", (production_id,)).fetchone()[0]
    round_ = int(last or 0) + 1
    connection.execute(
        "INSERT INTO ideas (production_id, round, source, feedback, revises, run_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (production_id, round_, source, feedback, revises, run_id, time.time()),
    )
    made = idea(connection, production_id, round_)
    assert made is not None
    return made


def set_idea_job(connection: sqlite3.Connection, idea_id: int, job_id: int) -> None:
    connection.execute("UPDATE ideas SET job_id = ? WHERE id = ?", (job_id, idea_id))


def write_idea(
    connection: sqlite3.Connection,
    production_id: int,
    round_: int,
    story: str,
    changes: list[str],
    questions: list[str],
) -> None:
    connection.execute(
        "UPDATE ideas SET story = ?, changes = ?, questions = ? WHERE production_id = ? AND round = ?",
        (story, _dump(changes), _dump(questions), production_id, round_),
    )


def adopt_idea(connection: sqlite3.Connection, production_id: int, round_: int) -> None:
    """The one round the production's story was taken from."""
    connection.execute(
        "UPDATE ideas SET adopted_at = CASE WHEN round = ? THEN ? END WHERE production_id = ?",
        (round_, time.time(), production_id),
    )


# --- settings -----------------------------------------------------------------------------------
def model_choices(connection: sqlite3.Connection) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in connection.execute("SELECT role, value FROM model_choices ORDER BY used_at DESC"):
        out.setdefault(row["role"], []).append(row["value"])
    return out


def remember_model(connection: sqlite3.Connection, role: str, value: str) -> None:
    connection.execute(
        "INSERT INTO model_choices (role, value, used_at) VALUES (?, ?, ?) "
        "ON CONFLICT(role, value) DO UPDATE SET used_at = excluded.used_at",
        (role, value, time.time()),
    )


def forget_model(connection: sqlite3.Connection, role: str, value: str) -> None:
    connection.execute("DELETE FROM model_choices WHERE role = ? AND value = ?", (role, value))


def overrides(connection: sqlite3.Connection, scope_kind: str, scope_id: str, tool: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM settings WHERE scope_kind = ? AND scope_id = ? AND tool = ?", (scope_kind, scope_id, tool)
    ).fetchone()
    return {} if row is None else Setting.of(row).overrides


def settings_in(connection: sqlite3.Connection, scope_kind: str, scope_id: str) -> list[Setting]:
    rows = connection.execute(
        "SELECT * FROM settings WHERE scope_kind = ? AND scope_id = ? ORDER BY tool", (scope_kind, scope_id)
    ).fetchall()
    return [Setting.of(row) for row in rows]


def put_overrides(
    connection: sqlite3.Connection, scope_kind: str, scope_id: str, tool: str, values: dict[str, Any]
) -> Setting:
    """The whole override set for one tool in one scope; empty deletes the row."""
    if not values:
        connection.execute(
            "DELETE FROM settings WHERE scope_kind = ? AND scope_id = ? AND tool = ?", (scope_kind, scope_id, tool)
        )
        return Setting(
            id=0, scope_kind=scope_kind, scope_id=scope_id, tool=tool, overrides={}, version=0, updated_at=0.0
        )
    connection.execute(
        "INSERT INTO settings (scope_kind, scope_id, tool, overrides, version, updated_at) VALUES (?, ?, ?, ?, 1, ?) "
        "ON CONFLICT(scope_kind, scope_id, tool) DO UPDATE SET overrides = excluded.overrides, "
        "version = settings.version + 1, updated_at = excluded.updated_at",
        (scope_kind, scope_id, tool, _dump(values), time.time()),
    )
    row = connection.execute(
        "SELECT * FROM settings WHERE scope_kind = ? AND scope_id = ? AND tool = ?", (scope_kind, scope_id, tool)
    ).fetchone()
    return Setting.of(row)
