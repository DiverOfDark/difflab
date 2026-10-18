"""What a tool runs with: its code defaults, then this machine's overrides, then the production's.

A job records the values it resolved to (`jobs.settings`), so provenance never points at a profile that
has since changed; and only `keyed` fields move a made thing's key, which is what `impact` counts.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from difflab import store, tools
from difflab.core import hub
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


def for_job(tool: str, production_id: int | None = None) -> dict[str, Any]:
    """What a job of the tool records and runs with; nothing for a tool with no settings."""
    if next(t for t in tools.TOOLS if t.name == tool).settings is None:
        return {}
    return effective(tool, production_id)[0].data()


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
    below = effective(tool)[0] if scope_kind == PRODUCTION else cls()
    kept = {
        name: getattr(resolved, name)
        for name in values
        if name in {f.name for f in dataclasses.fields(cls)}
        and not any(f".{name}:" in note or f"called {name!r}" in note for note in notes)
        and getattr(resolved, name) != getattr(below, name)
    }
    fields = {f.name: f for f in dataclasses.fields(cls)}
    chosen = [
        (fields[name].metadata["role"], model)
        for name, value in kept.items()
        if fields[name].metadata.get("role") and value
        for model in (_models_in(str(value)) if fields[name].metadata.get("multiple") else [str(value)])
    ]
    for _, model in chosen:
        hub.check(model)
    with store.connect() as connection, store.transaction(connection):
        store.put_overrides(connection, scope_kind, scope_id, tool, kept)
        for role, model in chosen:
            store.remember_model(connection, role, model)
    production_id = int(scope_id) if scope_kind == PRODUCTION and scope_id else None
    now, _ = effective(tool, production_id)
    return {"overrides": kept, "effective": now.data(), "notes": notes}


def _models_in(value: str) -> list[str]:
    """The references of a `multiple` field; a strength that is not a number refuses the lot."""
    out = []
    for line in filter(str.strip, value.splitlines()):
        reference, _, strength = line.partition("*")
        float(strength or 1.0)
        out.append(reference.strip())
    return out


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


def takes_file(role: str) -> bool:
    """Whether a model field of `role` names one file, as its code default does, rather than a whole repo."""
    for tool in tools.TOOLS:
        for field in dataclasses.fields(tool.settings) if tool.settings is not None else ():
            if field.metadata.get("role") == role:
                first = str(field.default).strip().split("\n")[0].split("*")[0]
                if not first:
                    return True
                try:
                    hub.parse_file(first)
                except ValueError:
                    return False
                return True
    return True


def models() -> dict[str, list[str]]:
    """Per role, the models chosen before, newest first; the code default is not among them."""
    with store.connect() as connection:
        return store.model_choices(connection)


def forget(role: str, value: str) -> None:
    with store.connect() as connection, store.transaction(connection):
        store.forget_model(connection, role, value)


def accounts() -> dict[str, dict[str, str]]:
    """Per service, where its token comes from and whose it is; never the token."""
    with store.connect() as connection:
        saved = store.tokens(connection)
    out = {}
    for service, variable in hub.TOKEN_ENV.items():
        if service in saved:
            out[service] = {"source": "settings", "account": saved[service][1], "tail": saved[service][0][-4:]}
        elif os.environ.get(variable):
            out[service] = {"source": f"${variable}", "account": "", "tail": os.environ[variable][-4:]}
        else:
            out[service] = {"source": "", "account": "", "tail": ""}
    return out


def save_token(service: str, value: str) -> None:
    """Checked with the service before it is kept; an empty value forgets the saved one."""
    if service not in hub.TOKEN_ENV:
        raise ValueError(f"no service called {service!r}")
    value = value.strip()
    account = hub.account(service, value) if value else ""
    with store.connect() as connection, store.transaction(connection):
        if value:
            store.put_token(connection, service, value, account)
        else:
            store.drop_token(connection, service)
