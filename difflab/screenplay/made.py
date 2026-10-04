"""A job's end, written back onto the made thing it was for."""

from __future__ import annotations

import json
import random
import sqlite3
from typing import Any

from difflab import store
from difflab.screenplay import production, words


def settle(connection: sqlite3.Connection, job: store.Job, outputs: dict[str, Any], state: str) -> None:
    """The clip a `render_shot` job was for, and the next shot's keyframe when that shot chains: the
    job's handoff frame is that keyframe."""
    if job.for_id is None:
        return
    if job.tool == "judge":
        _judged(connection, job, outputs, state)
        return
    if job.tool == "draw_reference" and job.for_kind == "reference":
        _drawn(connection, job, outputs, state)
        return
    if job.tool == "compose_keyframe" and job.for_kind == "keyframe":
        if state in (store.DONE, store.SKIPPED):
            store.set_made(
                connection,
                "keyframe",
                job.for_id,
                store.MADE,
                wanted_key=outputs.get("key"),
                artifact=outputs.get("frame"),
                current_job=job.id,
            )
        elif state == store.FAILED:
            store.set_made(connection, "keyframe", job.for_id, store.FAILED, current_job=job.id)
        return
    if job.for_kind != "clip":
        return
    clip = store.made_by_id(connection, "clip", job.for_id)
    if not isinstance(clip, store.Clip):
        return
    if state in (store.DONE, store.SKIPPED):
        store.set_made(
            connection,
            "clip",
            clip.id,
            store.MADE,
            wanted_key=outputs.get("key"),
            artifact=outputs.get("chunk"),
            current_job=job.id,
            handoff=outputs.get("cut"),
        )
        _chain_forward(connection, clip.shot_id, outputs.get("frame"), job.id)
    elif state == store.FAILED:
        store.set_made(connection, "clip", clip.id, store.FAILED, current_job=job.id)
    else:
        store.set_made(connection, "clip", clip.id, store.STALE if clip.artifact else store.MISSING)


def _drawn(connection: sqlite3.Connection, job: store.Job, outputs: dict[str, Any], state: str) -> None:
    """A member's drawn reference, and every prompt that casts it compiled again to hand it."""
    made_id = int(job.for_id or 0)
    reference = store.made_by_id(connection, "reference", made_id)
    if not isinstance(reference, store.Reference):
        return
    if state in (store.DONE, store.SKIPPED):
        store.set_made(
            connection,
            "reference",
            made_id,
            store.MADE,
            wanted_key=outputs.get("key"),
            artifact=outputs.get("frame"),
            current_job=job.id,
        )
        production.recompile_casting(connection, reference.member_id)
    elif state == store.FAILED:
        store.set_made(connection, "reference", made_id, store.FAILED, current_job=job.id)
    else:
        store.set_made(connection, "reference", made_id, store.STALE if reference.artifact else store.MISSING)


def _chain_forward(connection: sqlite3.Connection, shot_id: int, frame: str | None, job_id: int) -> None:
    shot = store.shot(connection, shot_id)
    if shot is None or frame is None:
        return
    following = [one for one in store.shots_of(connection, shot.chapter_id) if one.position == shot.position + 1]
    if not following:
        return
    keyframe = store.keyframe_of(connection, following[0].id)
    if keyframe.source == store.CHAINED:
        store.set_made(connection, "keyframe", keyframe.id, store.MADE, artifact=frame, current_job=job_id)


def _judged(connection: sqlite3.Connection, job: store.Job, outputs: dict[str, Any], state: str) -> None:
    """A verdict on a keyframe, a clip or a drawn reference. A keyframe below the gate is composed again
    on a new seed, told the judge's fixes, until the retries are spent; anything else below it, and a
    keyframe out of retries, goes to review."""
    if state != store.DONE or job.for_kind not in ("keyframe", "clip", "reference"):
        return
    kind, made_id = job.for_kind, int(job.for_id or 0)
    thing = store.made_by_id(connection, kind, made_id)
    if thing is None:
        return
    review = json.dumps({k: outputs.get(k) for k in ("score", "cast", "issues", "notes", "gate")})
    if outputs.get("passed"):
        store.set_made(connection, kind, made_id, store.APPROVED, review=review, reviewed_by="judge")
        return
    # No retries for a reference, whose new seed replaces the picture every prompt casting it hands, nor for a
    # clip: its retry is a blind re-seed that re-renders it and every shot chained after it, minutes each.
    if kind in ("reference", "clip"):
        store.set_made(connection, kind, made_id, store.NEEDS_REVIEW, review=review, reviewed_by="judge")
        return
    attempts = int(
        connection.execute(
            "SELECT count(*) FROM jobs WHERE tool = 'judge' AND for_kind = ? AND for_id = ? AND state = ?",
            (kind, made_id, store.DONE),
        ).fetchone()[0]
    )
    if attempts > int(outputs.get("retries", 0)):
        store.set_made(connection, kind, made_id, store.NEEDS_REVIEW, review=review, reviewed_by="judge")
        return
    store.set_made(connection, kind, made_id, store.STALE, review=review, reviewed_by="judge")
    seed = random.randrange(2**32)
    if kind == "clip" and isinstance(thing, store.Clip):
        store.set_clip_seed(connection, thing.shot_id, seed)
        shot = store.shot(connection, thing.shot_id)
    elif isinstance(thing, store.Keyframe):
        store.set_made(connection, "keyframe", made_id, store.STALE, seed=seed)
        fixes = [str(fix) for fix in outputs.get("fixes") or []]
        restyle = [str(member) for member in outputs.get("restyle") or []]
        if restyle:
            fixes.append(
                words.COMPOSE_CLOTHES_NOTE.format(who=" and ".join(f"the {m.replace('_', ' ')}" for m in restyle))
            )
        if fixes:  # the next composition is told what to fix; a note of the user's own stays in front
            own = (thing.note or "").split(JUDGED, 1)[0].strip()
            store.set_keyframe_note(connection, made_id, f"{own} {JUDGED} {'; '.join(fixes)}".strip())
        shot = store.shot(connection, thing.shot_id)
    else:
        return
    if shot is not None:
        run = store.run(connection, job.run_id)
        _again.append((shot.chapter_id, bool(run is not None and (run.data() or {}).get("keyframes_only"))))


JUDGED = "Fix:"  # where a keyframe's note turns from the user's words to the judge's

# Chapters a verdict asked to render again, taken by the worker once the job's rows are written: a run
# queued from inside the settle transaction would open a second connection mid-transaction.
_again: list[tuple[int, bool]] = []


def pending_rerenders() -> list[tuple[int, bool]]:
    """(chapter, keyframes only), as the run that asked was."""
    taken, _again[:] = list(dict.fromkeys(_again)), []
    return taken
