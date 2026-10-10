"""A job's end, written back onto the made thing it was for."""

from __future__ import annotations

import json
import random
import sqlite3
from typing import Any

from difflab import store
from difflab.core.settings import JudgeSettings
from difflab.screenplay import config as knobs
from difflab.screenplay import production, settings, words


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
    if job.tool == "draw_look" and job.for_kind == "casting":
        _look_drawn(connection, job, outputs, state)
        return
    if job.tool == "plate_ref" and job.for_kind == "plate":
        _plated(connection, job, outputs, state)
        return
    if job.tool == "compose_keyframe" and job.for_kind == "keyframe":
        if state in (store.DONE, store.SKIPPED):
            keyframe = store.made_by_id(connection, "keyframe", job.for_id)
            store.set_made(
                connection,
                "keyframe",
                job.for_id,
                _kept(keyframe, outputs, state),
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
            _kept(clip, outputs, state),
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


def _kept(thing: store.Made | None, outputs: dict[str, Any], state: str) -> str:
    """Approved still when a person approved this very key and the job only found it again."""
    held = thing is not None and thing.reviewed_by == store.PERSON and thing.wanted_key == outputs.get("key")
    return store.APPROVED if state == store.SKIPPED and held else store.MADE


def _plated(connection: sqlite3.Connection, job: store.Job, outputs: dict[str, Any], state: str) -> None:
    """A reference's plate, and every prompt casting its member compiled again to hand it."""
    reference = store.ref(connection, int(job.for_id or 0))
    if reference is None or state not in (store.DONE, store.SKIPPED) or not outputs.get("frame"):
        return
    store.set_plate(connection, reference.id, str(outputs["frame"]))
    production.recompile_casting(connection, reference.member_id)


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


def _look_drawn(connection: sqlite3.Connection, job: store.Job, outputs: dict[str, Any], state: str) -> None:
    """A look's picture. A new one makes every look drawn from it stale: their pixels came from the old."""
    made_id = int(job.for_id or 0)
    casting = store.made_by_id(connection, "casting", made_id)
    if not isinstance(casting, store.Casting):
        return
    if state in (store.DONE, store.SKIPPED):
        store.set_made(
            connection, "casting", made_id, _kept(casting, outputs, state), wanted_key=outputs.get("key"),
            artifact=outputs.get("frame"), current_job=job.id, size=json.dumps(outputs.get("size") or []),
        )  # fmt: skip
        if casting.artifact != outputs.get("frame"):
            production.restale_looks(connection, made_id)
        production.recompile_look(connection, made_id)
    elif state == store.FAILED:
        store.set_made(connection, "casting", made_id, store.FAILED, current_job=job.id)
    else:
        store.set_made(connection, "casting", made_id, store.STALE if casting.artifact else store.MISSING)


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
    """A verdict on a keyframe, a clip, a look or a drawn reference. Below the gate a keyframe is composed
    again on a new seed, told the judge's fixes, and a clip rendered again on one, until their attempts are
    spent; anything else below it, and whatever is out of attempts, goes to review."""
    if state != store.DONE or job.for_kind not in ("keyframe", "clip", "reference", "casting"):
        return
    kind, made_id = job.for_kind, int(job.for_id or 0)
    thing = store.made_by_id(connection, kind, made_id)
    if thing is None or (thing.state == store.APPROVED and thing.reviewed_by == store.PERSON):
        return
    verdict = {k: outputs.get(k) for k in ("score", "cast", "issues", "notes", "gate")}
    if isinstance(thing, store.Clip) and not outputs.get("unjudged"):
        _clip_judged(connection, job, thing, outputs, verdict)
        return
    review = json.dumps(verdict)
    if outputs.get("unjudged"):
        store.set_made(connection, kind, made_id, store.NEEDS_REVIEW, review=review, reviewed_by=store.JUDGE)
        return
    if outputs.get("passed"):
        store.set_made(connection, kind, made_id, store.APPROVED, review=review, reviewed_by=store.JUDGE)
        if kind == "casting":
            production.recompile_look(connection, made_id)
        return
    if kind in ("casting", "reference"):
        _picture_judged(connection, job, thing, outputs, verdict)
        return
    attempts = int(
        connection.execute(
            "SELECT count(*) FROM jobs WHERE tool = 'judge' AND for_kind = ? AND for_id = ? AND state = ?",
            (kind, made_id, store.DONE),
        ).fetchone()[0]
    )
    if not isinstance(thing, store.Keyframe) or attempts > int(outputs.get("retries", 0)):
        store.set_made(connection, kind, made_id, store.NEEDS_REVIEW, review=review, reviewed_by=store.JUDGE)
        return
    seed = random.randrange(2**32)
    store.set_made(connection, "keyframe", made_id, store.STALE, review=review, reviewed_by=store.JUDGE, seed=seed)
    fixes = [str(fix) for fix in outputs.get("fixes") or []]
    restyle = [str(member) for member in outputs.get("restyle") or []]
    if restyle:
        fixes.append(words.COMPOSE_CLOTHES_NOTE.format(who=" and ".join(f"the {m.replace('_', ' ')}" for m in restyle)))
    if fixes:  # the next composition is told what to fix; a note of the user's own stays in front
        own = (thing.note or "").split(JUDGED, 1)[0].strip()
        store.set_keyframe_note(connection, made_id, f"{own} {JUDGED} {'; '.join(fixes)}".strip())
    shot = store.shot(connection, thing.shot_id)
    if shot is not None:
        run = store.run(connection, job.run_id)
        _again.append((shot.chapter_id, bool(run is not None and (run.data() or {}).get("keyframes_only"))))


def _picture_judged(
    connection: sqlite3.Connection, job: store.Job, thing: store.Made, outputs: dict[str, Any], verdict: dict[str, Any]
) -> None:
    """A look or a drawn reference below the gate, drawn again on a new seed told the judge's fixes until its
    `retries` are spent; then the best take is put back and goes to review. Every take stays in `review.takes`."""
    kind, made_id = str(job.for_kind), thing.id
    path = job.inputs.get("path")
    draw = store.job(connection, int(path["from_job"])) if isinstance(path, dict) else None
    if draw is None:
        store.set_made(
            connection, kind, made_id, store.NEEDS_REVIEW, review=json.dumps(verdict), reviewed_by=store.JUDGE
        )
        _picture_moved(connection, kind, made_id, thing, False)
        return
    take = {**verdict, "seed": int(draw.inputs.get("seed") or 0), "outputs": draw.outputs}
    takes = [*(production.takes_of(thing.review) if draw.inputs.get("retake") else []), take]
    name = job.label.removeprefix("judge ")
    said = f"{float(outputs.get('score') or 0):.2f} below the gate"
    if len(takes) <= int(outputs.get("retries", 0)):
        seed = random.randrange(2**32)
        fixes = [str(f) for f in outputs.get("fixes") or []] or [str(i) for i in verdict["issues"] or []]
        store.requeue_job(connection, draw.id, {**draw.inputs, "seed": seed, "fixes": fixes, "retake": True})
        store.requeue_job(connection, job.id)
        _seeded(connection, kind, made_id, seed)
        store.set_made(
            connection, kind, made_id, store.QUEUED, current_job=draw.id,
            review=json.dumps({**verdict, "takes": takes}), reviewed_by=store.JUDGE,
        )  # fmt: skip
        _notes.append(f"{name}: {said}; drawn again on seed {seed}, told: {'; '.join(fixes) or 'nothing'}")
        return
    best = max(takes, key=lambda t: float(t.get("score") or 0))
    kept = best["outputs"]
    if best is not take:
        store.set_job_outputs(connection, draw.id, kept)
        _seeded(connection, kind, made_id, int(best["seed"]))
    review = {k: v for k, v in best.items() if k not in ("seed", "outputs")} | {"takes": takes}
    store.set_made(
        connection, kind, made_id, store.NEEDS_REVIEW, wanted_key=kept.get("key"), artifact=kept.get("frame"),
        review=json.dumps(review), reviewed_by=store.JUDGE,
    )  # fmt: skip
    _picture_moved(connection, kind, made_id, thing, best is not take)
    if len(takes) > 1:
        score = float(best.get("score") or 0)
        _notes.append(f"{name}: kept take {takes.index(best) + 1} of {len(takes)} ({score:.2f}) for review")


def _seeded(connection: sqlite3.Connection, kind: str, made_id: int, seed: int) -> None:
    if kind == "casting":
        store.update_casting(connection, made_id, seed=seed)
    else:
        store.set_ref_seed(connection, made_id, seed)


def _picture_moved(connection: sqlite3.Connection, kind: str, made_id: int, thing: store.Made, swapped: bool) -> None:
    """`swapped`: an earlier take put back, so every look drawn from the later one is stale."""
    if kind == "casting":
        if swapped:
            production.restale_looks(connection, made_id)
        production.recompile_look(connection, made_id)
    elif isinstance(thing, store.Reference):
        production.recompile_casting(connection, thing.member_id)


def _clip_judged(
    connection: sqlite3.Connection, job: store.Job, clip: store.Clip, outputs: dict[str, Any], verdict: dict[str, Any]
) -> None:
    """Below the gate, with auto-retry on, the clip's render job and this verdict run again in place on a new
    seed, so the shots after it read the new take; out of takes, the best is put back and goes to review and
    the run goes on. Every take seen stays in `review.takes`, each with its render's outputs."""
    path = job.inputs.get("path")
    render = store.job(connection, int(path["from_job"])) if isinstance(path, dict) else None
    if render is None:
        store.set_made(
            connection, "clip", clip.id, store.NEEDS_REVIEW, review=json.dumps(verdict), reviewed_by=store.JUDGE
        )
        return
    current = render.outputs.get("key")
    take = {**verdict, "seed": int(render.inputs.get("seed") or clip.seed), "outputs": render.outputs}
    earlier = production.takes_of(clip.review)
    # A new render starts its takes afresh; a retake, or the kept take found again by a later run, goes on
    # counting, so a chapter rendered again does not spend three more renders on a clip already in review.
    if not render.inputs.get("retake") and current not in {t["outputs"].get("key") for t in earlier}:
        earlier = []
    keys = [t["outputs"].get("key") for t in earlier]
    if current in keys:  # judged anew in its place
        takes = [take if k == current else t for t, k in zip(earlier, keys, strict=True)]
    else:
        takes = [*earlier, take]
    name = job.label.split(":")[0]
    if outputs.get("passed"):
        store.set_made(
            connection, "clip", clip.id, store.APPROVED, review=json.dumps({**verdict, "takes": takes}),
            reviewed_by=store.JUDGE,
        )  # fmt: skip
        return
    production_id = int(job.inputs.get("production") or 0)
    retrying = _retrying(connection, production_id)
    issues = "; ".join(str(i) for i in verdict["issues"] or []) or "no issues named"
    said = f"{float(outputs.get('score') or 0):.2f} below the gate ({issues})"
    if retrying and len(takes) < knobs.CLIP_TAKES:
        seed = random.randrange(2**32)
        store.set_clip_seed(connection, clip.shot_id, seed)
        store.requeue_job(connection, render.id, {**render.inputs, "seed": seed, "retake": True})
        store.requeue_job(connection, job.id)
        store.set_made(
            connection, "clip", clip.id, store.QUEUED, current_job=render.id,
            review=json.dumps({**verdict, "takes": takes}), reviewed_by=store.JUDGE,
        )  # fmt: skip
        _notes.append(f"{name}: {said}; take {len(takes) + 1} of {knobs.CLIP_TAKES} on seed {seed}")
        return
    best = max(takes, key=lambda t: float(t.get("score") or 0))
    kept = best["outputs"]
    if best is not take:
        store.set_job_outputs(connection, render.id, kept)
        store.set_clip_seed(connection, clip.shot_id, int(best["seed"]))
        _chain_forward(connection, clip.shot_id, kept.get("frame"), render.id)
    review = {k: v for k, v in best.items() if k not in ("seed", "outputs")} | {"takes": takes}
    store.set_made(
        connection, "clip", clip.id, store.NEEDS_REVIEW, wanted_key=kept.get("key"), artifact=kept.get("chunk"),
        review=json.dumps(review), reviewed_by=store.JUDGE, handoff=kept.get("cut"),
    )  # fmt: skip
    if retrying:
        _notes.append(
            f"{name}: {said}; kept take {takes.index(best) + 1} of {len(takes)} ({float(best.get('score') or 0):.2f}) "
            "for review, and the shots after it are built on it"
        )
    elif any(one.tool == "render_shot" and not one.settled for one in store.jobs_of(connection, job.run_id)):
        store.set_run_state(connection, job.run_id, store.PAUSED)  # the shots still to come open on this one
        _notes.append(
            f"{name}: {said}; paused so the shots after it are not built on it. Re-roll the shot, or resume the "
            "run to keep it."
        )


def _retrying(connection: sqlite3.Connection, production_id: int) -> bool:
    """The judge's `clip_retry` as it stands now, not as the run was queued: the Render page flips it mid-run."""
    machine = store.overrides(connection, settings.MACHINE, "", "judge")
    own = store.overrides(connection, settings.PRODUCTION, str(production_id), "judge")
    chosen, _ = JudgeSettings.resolve(machine, own)
    assert isinstance(chosen, JudgeSettings)
    return chosen.clip_retry


JUDGED = "Fix:"  # where a keyframe's note turns from the user's words to the judge's

# Chapters a verdict asked to render again, taken by the worker once the job's rows are written: a run
# queued from inside the settle transaction would open a second connection mid-transaction.
_again: list[tuple[int, bool]] = []

# What the verdicts did to a run - a retake queued, a take kept, a run paused - for the worker to say in it.
_notes: list[str] = []


def pending_notes() -> list[str]:
    taken, _notes[:] = list(_notes), []
    return taken


def pending_rerenders() -> list[tuple[int, bool]]:
    """(chapter, keyframes only), as the run that asked was."""
    taken, _again[:] = list(dict.fromkeys(_again)), []
    return taken
