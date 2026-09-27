"""What a tool runs with: its code defaults, then this machine's overrides, then the production's.

A job records the values it resolved to (`jobs.settings`), so provenance never points at a profile that
has since changed; and only `keyed` fields move a made thing's key, which is what `impact` counts.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from difflab import store, tools
from difflab.core.settings import Settings

MACHINE = "machine"
PRODUCTION = "production"
SCOPES = (MACHINE, PRODUCTION)


def settings_class(tool: str) -> type[Settings]:
    for known in tools.TOOLS:
        if known.name == tool:
            if known.settings is None:
                raise ValueError(f"{tool} has no settings")
            return known.settings
    raise ValueError(f"no tool called {tool!r}")


def effective(tool: str, production_id: int | None = None) -> tuple[Settings, list[str]]:
    """The tool's settings as a job of it would run, and what was dropped from a stale profile."""
    cls = settings_class(tool)
    with store.connect() as connection:
        machine = store.overrides(connection, MACHINE, "", tool)
        own = store.overrides(connection, PRODUCTION, str(production_id), tool) if production_id is not None else {}
    return cls.resolve(machine, own)


def describe(production_id: int | None = None) -> list[dict[str, Any]]:
    """Every tool with its schema and, per scope, what is overridden and what that comes to."""
    out = []
    with store.connect() as connection:
        for tool in tools.TOOLS:
            entry = tool.describe()
            if tool.settings is not None:
                machine = store.overrides(connection, MACHINE, "", tool.name)
                own = (
                    store.overrides(connection, PRODUCTION, str(production_id), tool.name)
                    if production_id is not None
                    else {}
                )
                resolved, notes = tool.settings.resolve(machine, own)
                entry |= {"machine": machine, "production": own, "effective": resolved.data(), "notes": notes}
            out.append(entry)
    return out


def put(scope_kind: str, scope_id: str, tool: str, values: dict[str, Any]) -> dict[str, Any]:
    """The whole override set for one tool in one scope. A value the class refuses is dropped and
    said, not saved."""
    if scope_kind not in SCOPES:
        raise ValueError(f"scope is one of {', '.join(SCOPES)}")
    cls = settings_class(tool)
    resolved, notes = cls.resolve(values)
    defaults = cls()
    kept = {
        name: getattr(resolved, name)
        for name in values
        if name in {f.name for f in dataclasses.fields(cls)}
        and not any(f".{name}:" in note or f"called {name!r}" in note for note in notes)
        and getattr(resolved, name) != getattr(defaults, name)
    }
    with store.connect() as connection, store.transaction(connection):
        store.put_overrides(connection, scope_kind, scope_id, tool, kept)
    production_id = int(scope_id) if scope_kind == PRODUCTION and scope_id else None
    now, _ = effective(tool, production_id)
    return {"overrides": kept, "effective": now.data(), "notes": notes}


def impact(scope_kind: str, scope_id: str, tool: str, values: dict[str, Any]) -> dict[str, Any]:
    """What saving these overrides would move: the keyed fields that change, and how many made things
    in the scope would go stale for it - clips for the chain's tools, prompts for the writer's,
    keyframes for the editor's."""
    if scope_kind not in SCOPES:
        raise ValueError(f"scope is one of {', '.join(SCOPES)}")
    cls = settings_class(tool)
    production_id = int(scope_id) if scope_kind == PRODUCTION and scope_id else None
    before, _ = effective(tool, production_id)
    with store.connect() as connection:
        machine = store.overrides(connection, MACHINE, "", tool)
        own = store.overrides(connection, PRODUCTION, str(production_id), tool) if production_id is not None else {}
    layers = (values, own) if scope_kind == MACHINE else (machine, values)
    after, notes = cls.resolve(*layers)
    keyed = {f.name for f in dataclasses.fields(cls) if f.metadata.get("keyed", True)}
    changed = sorted(name for name in keyed if getattr(before, name) != getattr(after, name))
    stale: dict[str, int] = {}
    if changed:
        with store.connect() as connection:
            shots = _shots_in(connection, production_id)
            if tool in ("render_chunk", "handoff", "assemble"):
                stale["clips"] = sum(1 for shot in shots if store.clip_of(connection, shot.id).artifact)
            elif tool in ("ask_vlm", "write_instruction", "write_video_prompt"):
                stale["prompts"] = sum(
                    1
                    for shot in shots
                    if (prompt := store.prompt_of(connection, shot.id)).artifact and not prompt.pinned
                )
            elif tool == "edit_image":
                stale["keyframes"] = sum(
                    1 for shot in shots if (kf := store.keyframe_of(connection, shot.id)).artifact and not kf.pinned
                )
    return {"changed": changed, "stale": stale, "notes": notes}


def _shots_in(connection: Any, production_id: int | None) -> list[store.Shot]:
    productions = (
        ([p] if production_id is not None and (p := store.production(connection, production_id)) else [])
        if production_id is not None
        else store.productions(connection)
    )
    return [
        shot
        for production in productions
        for chapter in store.chapters_of(connection, production.id)
        for shot in store.shots_of(connection, chapter.id)
    ]
