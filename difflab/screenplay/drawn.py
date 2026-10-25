"""References the editor draws for members with no picture of their own: which are missing, what each
is keyed by, and the run that draws them."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from typing import Any

from difflab import config as app_config
from difflab import store
from difflab.core.keys import Key, key
from difflab.core.settings import EditSettings, JudgeSettings
from difflab.screenplay import config as knobs
from difflab.screenplay import settings, words

SCOPE = "references:{production}"
ASPECT = {"person": 2 / 3, "animal": 1.0, "object": 1.0}  # a whole standing figure; a bird or a dish
UNPLANNED_ASPECT = 16 / 9  # a place drawn before any plan has fixed the canvas, and no aspect was given
# The editor's own area rather than its 2 MP default. Measured 2026-09-27: 2 MP grew a pebbled
# texture over asphalt and fried skin, and sampled in 52 s where 1 MP took 35 s; a reference is at most
# the canvas's 0.64 MP anyway.
MEGAPIXELS = 1.0
ALIGN = app_config.CANVAS_ALIGN


def size_for(kind: str, manifest: dict[str, Any]) -> list[int]:
    """A place at the canvas's own size; anyone else on the 32 grid near a view crop's area. Both stay
    within the canvas's area, so H3's reference sizing passes them untouched."""
    canvas = manifest.get("canvas") or {}
    if kind == "environment" and canvas.get("width") and canvas.get("height"):
        return [int(canvas["width"]), int(canvas["height"])]
    area = int(canvas["width"]) * int(canvas["height"]) if canvas.get("width") else knobs.DIRECTOR_ENVIRONMENT_PIXELS
    aspect = float(manifest.get("aspect") or UNPLANNED_ASPECT) if kind == "environment" else ASPECT[kind]
    pixels = min(area, knobs.DIRECTOR_ENVIRONMENT_PIXELS if kind == "environment" else knobs.DIRECTOR_VIEW_PIXELS)
    width = max(ALIGN, round(math.sqrt(pixels * aspect) / ALIGN) * ALIGN)
    height = max(ALIGN, round(math.sqrt(pixels / aspect) / ALIGN) * ALIGN)
    while width * height > area:
        width, height = (width - ALIGN, height) if width >= height else (width, height - ALIGN)
    return [width, height]


def wanted(member: store.Member, editor: EditSettings, size: list[int]) -> Key:
    """The words it is drawn from, the editor's settings with its seed, and its size: an edit to the
    member moves it, and so does a regenerate."""
    look = {k: member.identity.get(k, []) for k in ("traits", "lights")}
    if member.appearance.strip():  # only when set, so a member drawn from its traits keeps its key
        look["appearance"] = " ".join(member.appearance.split())
    return key(
        "draw_reference", 1, kind=member.kind, noun=member.noun, look=look, settings=editor, size=list(size),
        words=[words.COMPOSE_REFERENCE_INSTRUCTION[member.kind], words.COMPOSE_REFERENCE_LOOK],
    )  # fmt: skip


def current(reference: store.Reference, member: store.Member, manifest: dict[str, Any], editor: EditSettings) -> bool:
    """Whether a drawn reference still shows the member as its words and the editor's settings say."""
    seed = reference.seed if reference.seed is not None else editor.seed
    return reference.wanted_key == str(wanted(member, editor.with_(seed=seed), size_for(member.kind, manifest)))


def state(reference: store.Reference, member: store.Member, manifest: dict[str, Any], editor: EditSettings) -> str:
    """Its lifecycle state, stale once its wanted key has moved."""
    settled = (store.MADE, store.APPROVED, store.NEEDS_REVIEW)
    if reference.state in settled and not current(reference, member, manifest, editor):
        return store.STALE
    return reference.state


def _detached(manifest: dict[str, Any], slug: str) -> bool:
    return [slug, "", store.GENERATED] in [list(e) for e in manifest.get("detached", [])]


def missing(connection: Any, production: store.Production) -> list[store.Member]:
    """Members with no picture cut from a photograph or given, whose drawn one is absent, stale or
    failed. A drawn one detached by hand is not drawn again unless asked for by name."""
    editor = editor_for(production.id)
    out = []
    for member in store.members_of(connection, production.id):
        references = store.refs_of(connection, member.id)
        if any(r.artifact and not r.generated for r in references):
            continue
        drawn = next((r for r in references if r.generated), None)
        if drawn is None:
            if not _detached(production.manifest, member.slug):
                out.append(member)
        elif not drawn.artifact or state(drawn, member, production.manifest, editor) in (store.STALE, store.FAILED):
            out.append(member)
    return out


def live_run(connection: Any, production_id: int) -> store.Run | None:
    scope = SCOPE.format(production=production_id)
    return next(
        (
            run
            for run in store.runs(connection)
            if run.scope == scope and run.state in (store.QUEUED, store.RUNNING, store.PAUSED)
        ),
        None,
    )


def draw(
    production_id: int,
    member_id: int | None = None,
    seed: int | None = None,
    asked_by: str = "person",
    describe: bool = False,
) -> str:
    """A run drawing every missing reference, or one member's. One member's drawn again without a
    `seed` takes a new one; a first drawing takes the editor's. `describe` has a place's look written
    from the story first, over its own words. Returns the run id."""
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        if production is None:
            raise LookupError(f"no production {production_id}")
        running = live_run(connection, production_id)
        if running is not None:
            raise ValueError(f"references are being drawn already, in run {running.id}")
        if member_id is None:
            chosen = missing(connection, production)
            if not chosen:
                raise ValueError("every member has a reference: nothing is missing")
        else:
            member = store.member(connection, member_id)
            if member is None or member.production_id != production_id:
                raise LookupError(f"no member {member_id} in production {production_id}")
            chosen = [member]
        editor = editor_for(production_id)
        verdicts, _ = settings.effective("judge", production_id)
        assert isinstance(verdicts, JudgeSettings)
        manifest = production.manifest
        run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        store.add_run(
            connection, run_id, title=f"{production.title} · references", asked_by=asked_by,
            scope=SCOPE.format(production=production_id), production_id=production_id,
            spec={"production": production_id, "items": [m.slug for m in chosen]},
        )  # fmt: skip
        for member in chosen:
            existing = store.generated_ref(connection, member.id)
            again = member_id is not None and (
                (existing is not None and existing.artifact is not None) or _detached(manifest, member.slug)
            )
            if seed is not None:
                chosen_seed = seed
            elif again:
                chosen_seed = random.randrange(2**32)
            else:
                chosen_seed = existing.seed if existing is not None and existing.seed is not None else editor.seed
            size = size_for(member.kind, manifest)
            reference = store.put_generated_ref(connection, member.id, chosen_seed, size)
            job = store.add_job(
                connection, run_id, "draw_reference",
                {"production": production_id, "member": member.id, "seed": chosen_seed, "size": size,
                 **({"describe": True} if describe else {})},
                label=f"draw {member.slug}", settings=editor.data(), for_kind="reference", for_id=reference.id,
            )  # fmt: skip
            store.set_made(connection, "reference", reference.id, store.QUEUED, current_job=job.id)
            store.add_job(
                connection, run_id, "judge",
                {"production": production_id, "kind": "reference picture",
                 "path": {"from_job": job.id, "output": "frame"},
                 "purpose": f"a reference picture of the {member.noun} alone, whole and sharp, matching every "
                            "listed trait",
                 "arrangement": f"{{{member.slug}}} alone on a plain backdrop" if member.kind != "environment"
                 else f"{{{member.slug}}}, empty of people and animals",
                 "cast": [member.slug]},
                label=f"judge {member.slug}", settings=verdicts.data(), for_kind="reference", for_id=reference.id,
            )  # fmt: skip
        if member_id is not None and _detached(manifest, chosen[0].slug):
            detached = [
                list(e) for e in manifest.get("detached", []) if list(e) != [chosen[0].slug, "", store.GENERATED]
            ]
            store.update_production(connection, production_id, manifest={**manifest, "detached": detached})
    return run_id


def editor_for(production_id: int) -> EditSettings:
    """The editor's settings for this production, at the size a reference is drawn at."""
    editor, _ = settings.effective("edit_image", production_id)
    assert isinstance(editor, EditSettings)
    return editor.with_(megapixels=MEGAPIXELS)


VARIANTS = 3


def variants(production_id: int, member_id: int, count: int = VARIANTS) -> str:
    """`count` drawings of one member on new seeds, none of them its reference until one is picked."""
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        member = store.member(connection, member_id)
        if production is None or member is None or member.production_id != production_id:
            raise LookupError(f"no member {member_id} in production {production_id}")
        if (running := live_run(connection, production_id)) is not None:
            raise ValueError(f"references are being drawn already, in run {running.id}")
        editor = editor_for(production_id)
        run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        store.add_run(
            connection, run_id, title=f"{production.title} · {member.slug} · {count} drawings", asked_by="person",
            scope=SCOPE.format(production=production_id), production_id=production_id,
            spec={"production": production_id, "items": [f"{member.slug} {n + 1}" for n in range(count)]},
        )  # fmt: skip
        size = size_for(member.kind, production.manifest)
        for n in range(count):
            store.add_job(
                connection, run_id, "draw_reference",
                {"production": production_id, "member": member.id, "seed": random.randrange(2**32), "size": size},
                label=f"draw {member.slug} · {n + 1}", settings=editor.data(), for_kind=VARIANT, for_id=member.id,
            )  # fmt: skip
    return run_id


VARIANT = "variant"  # a drawing that waits to be picked: `made.settle` leaves the member's reference alone


def drawn_variants(connection: Any, member_id: int) -> list[dict[str, Any]]:
    """The member's drawings from its newest variants run, as {seed, frame}."""
    rows = connection.execute(
        "SELECT run_id FROM jobs WHERE tool = 'draw_reference' AND for_kind = ? AND for_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (VARIANT, member_id),
    ).fetchone()
    if rows is None:
        return []
    jobs = connection.execute(
        "SELECT inputs, outputs, state FROM jobs WHERE run_id = ? AND for_kind = ? AND for_id = ? ORDER BY position",
        (rows["run_id"], VARIANT, member_id),
    ).fetchall()
    out = []
    for job in jobs:
        made = json.loads(job["outputs"] or "{}")
        out.append({"seed": int(json.loads(job["inputs"])["seed"]), "frame": made.get("frame"), "state": job["state"]})
    return out


def pick(member_id: int, seed: int) -> None:
    """One of the member's drawings becomes its reference, as if it had been drawn on that seed."""
    with store.connect() as connection, store.transaction(connection):
        member = store.member(connection, member_id)
        production = None if member is None else store.production(connection, member.production_id)
        if member is None or production is None:
            raise LookupError(f"no member {member_id}")
        chosen = next((v for v in drawn_variants(connection, member_id) if v["seed"] == seed and v["frame"]), None)
        if chosen is None:
            raise ValueError(f"{member.slug} has no finished drawing on seed {seed}")
        editor = editor_for(production.id).with_(seed=seed)
        size = size_for(member.kind, production.manifest)
        reference = store.put_generated_ref(connection, member.id, seed, size)
        store.set_made(
            connection, "reference", reference.id, store.MADE, artifact=chosen["frame"],
            wanted_key=str(wanted(member, editor, size)),
        )  # fmt: skip
