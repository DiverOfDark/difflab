"""The Render page: every shot's clip as the render queue holds it, and the header over them. VRAM is the shell's
`GET /api/gpu`, not this."""

from __future__ import annotations

import json
import time
from typing import Any

from difflab import store
from difflab.core.settings import JudgeSettings
from difflab.screenplay import config as knobs
from difflab.screenplay import looks, settings, timings
from difflab.screenplay import production as production_

LIVE = (store.QUEUED, store.RUNNING, store.PAUSED)


def status(production_id: int, chapter_position: int | None = None) -> dict[str, Any] | None:
    judge, _ = settings.effective("judge", production_id)
    assert isinstance(judge, JudgeSettings)
    with store.connect() as connection:
        production = store.production(connection, production_id)
        if production is None:
            return None
        resolved = looks.resolve(connection, production)
        keyframes = store.made_of_production(connection, "keyframe", production_id)
        prompts = store.made_of_production(connection, "prompt", production_id)
        clips = store.made_of_production(connection, "clip", production_id)
        chapters = [
            (chapter, store.shots_of(connection, chapter.id))
            for chapter in store.chapters_of(connection, production_id)
            if chapter_position is None or chapter.position == chapter_position
        ]
        runs = sorted(
            (r for r in store.runs(connection) if r.production_id == production_id and r.scope.startswith("chapter:")),
            key=lambda r: r.id,
        )
        live = [r for r in runs if r.state in LIVE]
        jobs = {j.id: j for r in live for j in store.jobs_of(connection, r.id)}
        running = next((j for j in jobs.values() if j.state == store.RUNNING), None)
        tasks = store.tasks_of(connection, running.id) if running is not None else []
        measured = timings.measured(connection)
    paused = {r.id for r in live if r.state == store.PAUSED}
    doing = _doing(running, tasks)
    rows: list[dict[str, Any]] = []
    for chapter, shots in chapters:
        for shot in shots:
            keyframe, prompt, clip = keyframes[shot.id], prompts[shot.id], clips[shot.id]
            assert isinstance(keyframe, store.Keyframe) and isinstance(prompt, store.Prompt)
            assert isinstance(clip, store.Clip)
            state, text = _state(shot, keyframe, clip, resolved, jobs, paused, running, doing)
            rows.append(
                {
                    "id": shot.id,
                    "label": f"{chapter.position + 1:02d}.{shot.position + 1:03d}",
                    "chapter": chapter.position,
                    "action": shot.action,
                    "seconds": shot.seconds,
                    "opening": production_.opening_of(keyframe),
                    "preset": _preset(keyframe, bool(prompt.pictures)),
                    "state": state,
                    "text": text,
                    "clip": {
                        "id": clip.id,
                        "state": clip.state,
                        "seed": clip.seed,
                        "video": clip.artifact if clip.present else None,
                    },
                    "still": keyframe.artifact if keyframe.present else None,
                    "review": _review(clip) if clip.state == store.NEEDS_REVIEW else None,
                }
            )
    judged, rejected, retaken = _rejections(clips[r["id"]] for r in rows)
    now = next((r for r in rows if r["id"] == _shot_of(running, clips)), None)
    return {
        "rows": rows,
        "total": len(rows),
        "done": sum(r["state"] == "done" for r in rows),
        "flagged": sum(r["state"] == "review" for r in rows),
        "now": {"label": now["label"], "doing": doing or "rendering"} if now is not None else None,
        "eta": _eta(rows, running, measured),
        "judged": judged,
        "rejected": rejected,
        "retaken": retaken,
        "status": "paused" if paused else "running" if running else "queued" if live else "idle",
        "runs": [{"id": r.id, "state": r.state, "title": r.title} for r in live],
        "auto_retry": judge.clip_retry,
        "takes": knobs.CLIP_TAKES,
        "gate": judge.gate,
    }


def hold(production_id: int, pause: bool) -> int:
    """Every live chapter run of the production paused after its job in flight, or let go; how many were told."""
    with store.connect() as connection, store.transaction(connection):
        told = 0
        for run in store.runs(connection):
            if run.production_id != production_id or not run.scope.startswith("chapter:"):
                continue
            if (pause and run.state in (store.QUEUED, store.RUNNING)) or (not pause and run.state == store.PAUSED):
                store.add_command(connection, "run", run.id, {"cmd": "pause" if pause else "resume"})
                told += 1
        return told


def auto_retry(production_id: int, on: bool) -> bool:
    with store.connect() as connection:
        own = store.overrides(connection, settings.PRODUCTION, str(production_id), "judge")
    settings.put(settings.PRODUCTION, str(production_id), "judge", {**own, "clip_retry": on})
    return on


def _preset(keyframe: store.Keyframe, references: bool) -> str:
    """H3's mode as `render_chunk.render` names it: a first frame, and a last if any, is fl2va; references ref2va."""
    if keyframe.source == store.REFS:
        return "ref2va"
    return "fl2va+ref" if references else "fl2va"


def _doing(running: store.Job | None, tasks: list[store.Task]) -> str:
    """What the job in flight is at: "step 3/5" while H3 samples, the stage's own name otherwise."""
    if running is None:
        return ""
    if running.tool == "judge":
        return "judging"
    task = next((t for t in reversed(tasks) if t.state == store.RUNNING), None)
    if task is None:
        return ""
    steps = int((running.settings.get("video") or {}).get("steps") or 0)
    if task.name == "Sampling" and steps and task.total:
        return f"step {min(steps, int(task.done or 0) // (task.total // steps) + 1)}/{steps}"
    return task.name.lower()


def _shot_of(running: store.Job | None, clips: dict[int, Any]) -> int | None:
    if running is None or running.for_kind != "clip":
        return None
    return next((shot for shot, clip in clips.items() if clip.id == running.for_id), None)


def _state(
    shot: store.Shot,
    keyframe: store.Keyframe,
    clip: store.Clip,
    resolved: looks.Resolved,
    jobs: dict[int, store.Job],
    paused: set[str],
    running: store.Job | None,
    doing: str,
) -> tuple[str, str]:
    """(state, what the row says): done, review, running, paused, queued, blocked, failed or stale."""
    if running is not None and running.for_kind == "clip" and running.for_id == clip.id:
        return "running", doing.replace("step", "sampling") if doing.startswith("step") else doing or "rendering"
    if clip.state in (store.QUEUED, store.RUNNING):
        job = jobs.get(clip.current_job or 0)
        return ("paused", "paused") if job is not None and job.run_id in paused else ("queued", "queued")
    if clip.state in (store.MADE, store.APPROVED, store.PINNED):
        return "done", "done"
    if clip.state == store.NEEDS_REVIEW:
        return "review", "review"
    if resolved.waiting(shot.id):
        return "blocked", "blocked · look"
    if (
        (keyframe.source == store.COMPOSED and keyframe.state in (store.NEEDS_REVIEW, store.FAILED))
        or (keyframe.source == store.PROVIDED and not keyframe.present)
        or (keyframe.source == store.CHAINED and shot.position == 0)
    ):
        return "blocked", "blocked · keyframe"
    if clip.state == store.FAILED:
        return "failed", "failed"
    return "stale", "stale" if clip.state == store.STALE else "to render"


def _review(clip: store.Clip) -> dict[str, Any]:
    try:
        verdict = json.loads(clip.review or "{}")
    except ValueError:
        verdict = {}
    issues = [str(i) for i in verdict.get("issues") or []]
    takes = production_.takes_of(clip.review)
    return {
        "why": "; ".join(issues) or str(verdict.get("notes") or "the judge gave no reason"),
        "score": verdict.get("score"),
        "takes": len(takes) or 1,
        "seeds": [t.get("seed") for t in takes],
    }


def _rejections(clips: Any) -> tuple[int, int, int]:
    """(takes judged, of them below the gate, takes rendered again), over the clips' verdicts as they stand."""
    counts = [production_.judged(clip.review) for clip in clips]
    return sum(c[0] for c in counts), sum(c[1] for c in counts), sum(c[2] for c in counts)


def _eta(rows: list[dict[str, Any]], running: store.Job | None, measured: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Seconds until the queued clips are rendered and judged, or, with nothing queued, what the rest would take."""
    render, judge = measured["render_shot"], float(measured["judge"]["seconds"])
    queued = [r for r in rows if r["state"] in ("running", "queued", "paused")]
    left = queued or [r for r in rows if r["state"] not in ("done", "review")]

    def each(seconds: float) -> float:
        per = render.get("per_second")
        return (float(per) * seconds if per else float(render["seconds"])) + judge

    total = sum(each(float(r["seconds"])) for r in left)
    if queued and running is not None and running.started_at and running.tool == "render_shot":
        total -= min(time.time() - running.started_at, each(float(running.inputs.get("seconds") or 0)) - judge)
    return {"seconds": round(max(0.0, total)), "clips": len(left), "queued": bool(queued)}
