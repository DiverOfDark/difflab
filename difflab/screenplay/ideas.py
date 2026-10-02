"""A production's idea rewritten by the director round by round, each round a run of one job."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from difflab import store
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import production, settings

SCOPE = "idea:{production}"


def improve(
    production_id: int, feedback: str = "", revises: int | None = None, idea: str | None = None,
    asked_by: str = "person",
) -> str:  # fmt: skip
    """A run revising round `revises` by `feedback`, or rewriting `idea` (else the rounds', else the story) afresh."""
    feedback = " ".join(feedback.split())
    with store.connect() as connection, store.transaction(connection):
        made = store.production(connection, production_id)
        if made is None:
            raise LookupError(f"no production {production_id}")
        if made.story_kind != "idea":
            raise ValueError("only a production begun from an idea has one to improve")
        running = live_run(connection, production_id)
        if running is not None:
            raise ValueError(f"the director is already rewriting this idea, in run {running.id}")
        rounds = store.ideas_of(connection, production_id)
        previous = None
        if revises is not None:
            previous = next((r for r in rounds if r.round == revises), None)
            if previous is None or not previous.story:
                raise LookupError(f"round {revises} has no story to revise")
            if not feedback:
                raise ValueError("say what to change: the feedback is what the revision is made from")
        source = (idea or "").strip() or (previous.source if previous else rounds[-1].source if rounds else made.story)
        if not source.strip():
            raise ValueError("an idea to improve, in your own words")
        photos = [str(p) for p in made.manifest.get("photos", [])]
        video, _ = settings.effective("render_chunk", production_id)
        assert isinstance(video, VideoSettings)
        seconds = float(made.manifest.get("seconds") or video.duration)
        total = production.take_seconds(made.manifest, seconds)
        if total <= 0:
            raise ValueError("the production does not say how long the film is")
        run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        drafted = store.add_idea(
            connection, production_id, source, feedback if previous else "", revises if previous else None, run_id
        )
        store.add_run(
            connection, run_id, title=f"{made.title} · idea round {drafted.round}", asked_by=asked_by,
            scope=SCOPE.format(production=production_id), production_id=production_id,
            spec={"production": production_id, "round": drafted.round, "items": [f"round {drafted.round}"]},
        )  # fmt: skip
        job = store.add_job(
            connection, run_id, "refine_idea",
            {"production": production_id, "round": drafted.round, "idea": source, "photos": photos,
             "take_seconds": round(total, 1), "seconds": round(seconds, 2),
             "previous": previous.story if previous else "", "feedback": feedback},
            label=f"revise round {revises}" if previous else f"rewrite the idea, round {drafted.round}",
            settings=settings.for_job("refine_idea", production_id),
        )  # fmt: skip
        store.set_idea_job(connection, drafted.id, job.id)
    return run_id


def live_run(connection: Any, production_id: int) -> store.Run | None:
    scope = SCOPE.format(production=production_id)
    return next(
        (
            r
            for r in store.runs(connection)
            if r.scope == scope and r.state in (store.QUEUED, store.RUNNING, store.PAUSED)
        ),
        None,
    )


def rounds(connection: Any, production_id: int) -> list[dict[str, Any]]:
    """Every round, with its job's state and how long it took."""
    out = []
    for one in store.ideas_of(connection, production_id):
        job = store.job(connection, one.job_id) if one.job_id is not None else None
        took = job.ended_at - job.started_at if job and job.started_at and job.ended_at else None
        out.append(
            {
                "round": one.round,
                "source": one.source,
                "feedback": one.feedback,
                "revises": one.revises,
                "story": one.story,
                "changes": one.changes,
                "questions": one.questions,
                "run": one.run_id,
                "state": job.state if job else store.FAILED,
                "error": job.error if job else "",
                "seconds": None if took is None else round(took, 1),
                "adopted": one.adopted_at is not None,
                "created": one.created_at,
            }
        )
    return out


def listing(production_id: int) -> list[dict[str, Any]]:
    with store.connect() as connection:
        if store.production(connection, production_id) is None:
            raise LookupError(f"no production {production_id}")
        return rounds(connection, production_id)


def adopt(production_id: int, round_: int) -> dict[str, int]:
    """The round's story as the production's, which makes every prompt stale until a re-plan."""
    with store.connect() as connection:
        chosen = store.idea(connection, production_id, round_)
    if chosen is None:
        raise LookupError(f"production {production_id} has no round {round_}")
    if not chosen.story:
        raise ValueError(f"round {round_} has no story yet")
    moved = production.update(production_id, story=chosen.story)
    with store.connect() as connection, store.transaction(connection):
        store.adopt_idea(connection, production_id, round_)
    return moved
