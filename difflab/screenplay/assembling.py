"""Chapters levelled and subtitled, and the film, from clips already rendered: no job here samples."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from difflab import config, store
from difflab.core.settings import AssembleSettings, Continuity
from difflab.screenplay import config as knobs
from difflab.screenplay import settings

CUE_HOLD = 0.6  # s a line stays up past its spoken length, so the eye can finish it
CUE_SHORTEST = 1.2  # s, the least a line stays up


def _output(job_id: int, name: str) -> dict[str, Any]:
    return {"from_job": job_id, "output": name}


def chosen(production_id: int) -> AssembleSettings:
    found, _ = settings.effective("assemble", production_id)
    assert isinstance(found, AssembleSettings)
    return found


def joins(connection: Any, production: store.Production, shots: list[store.Shot]) -> dict[str, list[Any]]:
    """Per join, the fade and whether the next shot opened on this one's handoff frame. A chained join keeps
    its seam blend; any other crossfades only where the scene changes, and is a cut inside one."""
    continuity, _ = settings.effective("handoff", production.id)
    assert isinstance(continuity, Continuity)
    assembling = chosen(production.id)
    sources = [store.keyframe_of(connection, shot.id).source for shot in shots]
    chained = [source == store.CHAINED for source in sources[1:]] + [False]
    fades = [
        continuity.crossfade if linked else assembling.crossfade if _boundary(shot, after) else 0.0
        for shot, after, linked in zip(shots, [*shots[1:], None], chained, strict=True)
    ]
    return {"fades": fades, "chained": chained}


def _boundary(shot: store.Shot, after: store.Shot | None) -> bool:
    """A shot with no scene named is taken to go on in the one before."""
    return after is not None and bool(shot.scene and after.scene) and shot.scene != after.scene


def lines(shot: store.Shot | None) -> list[list[Any]]:
    """The shot's dialogue as `[start, end, text]` within it, each up about as long as it takes to say."""
    if shot is None:
        return []
    out = []
    for said in shot.plan.get("dialogue", []):
        text, start = str(said.get("line", "")).strip(), float(said.get("start") or 0.0)
        if not text or start >= shot.seconds:
            continue
        spoken = len(text.split()) / knobs.DIRECTOR_WORDS_PER_SECOND + CUE_HOLD
        out.append([start, round(min(start + max(spoken, CUE_SHORTEST), shot.seconds), 3), text])
    return out


def marker(chapter: store.Chapter) -> str:
    return f"Chapter {chapter.position + 1}" + (f" · {chapter.title}" if chapter.title else "")


def _name(production: store.Production, chapter: store.Chapter, chapters: int) -> str:
    return production.slug if chapters == 1 else f"{production.slug}_ch{chapter.position + 1:02d}"


def add_master(
    connection: Any,
    run_id: str,
    production: store.Production,
    chapter: store.Chapter,
    source: dict[str, Any],
    parts: dict[str, Any],
    said: list[list[list[Any]]],
    finished: bool,
) -> int:
    """A master job over `source`, the video a join or finish job makes from `parts`, its own inputs."""
    chapters = len(store.chapters_of(connection, production.id))
    fps = parts.get("fps")
    return store.add_job(
        connection, run_id, "master",
        {
            "video": source, "chunks": parts["chunks"], "cuts": parts["cuts"],
            **({"fades": parts["fades"], "chained": parts["chained"]} if "fades" in parts
               else {"crossfade": parts.get("crossfade", 0.0)}),
            "lines": said, "chapter": chapter.id, "finished": finished,
            "name": _name(production, chapter, chapters) + (f"_{fps}fps" if finished and fps else ""),
        },
        label=f"{marker(chapter).lower()}: level{' the finish' if finished else ''} and subtitle",
        settings=chosen(production.id).data(),
    ).id  # fmt: skip


def add_film(connection: Any, production: store.Production, masters: list[tuple[store.Chapter, int]]) -> str:
    """A run of one film job over each chapter's master, waiting for any still to come."""
    run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    store.add_run(
        connection, run_id, title=f"{production.title} · the film", spec={"production": production.id},
        scope=f"film:{production.id}", production_id=production.id,
    )  # fmt: skip
    store.add_job(
        connection, run_id, "film",
        {
            "videos": [_output(job, "video") for _, job in masters],
            "cues": [_output(job, "cues") for _, job in masters],
            "titles": [marker(chapter) for chapter, _ in masters],
            "title": production.title, "name": production.slug,
        },
        label=f"the film: {len(masters)} chapter{'s' if len(masters) != 1 else ''}",
        settings=chosen(production.id).data(),
    )  # fmt: skip
    return run_id


def film_after(
    connection: Any, production: store.Production, finished: dict[int, int], native: dict[int, int]
) -> str | None:
    """The film run over each chapter's master: the one just queued (chapter id: job id) or the newest made;
    None while a chapter has none. Finished chapters only once every chapter is finished, since the film's
    streams are copied and a 24 fps chapter cannot follow a 60 fps one."""
    chapters = store.chapters_of(connection, production.id)
    done = _masters(connection, production.id)
    for queued, kind in ((finished, True), (native, False)):
        picked = [(c, queued.get(c.id) or done.get((c.id, kind))) for c in chapters]
        if all(job for _, job in picked):
            return add_film(connection, production, [(c, int(job or 0)) for c, job in picked])
    return None


def _masters(connection: Any, production_id: int) -> dict[tuple[int, bool], int]:
    """The newest settled master job per (chapter id, finished) whose video is still there."""
    out: dict[tuple[int, bool], int] = {}
    for job in _jobs(connection, production_id, "master"):
        chapter = job.inputs.get("chapter")
        if chapter is not None and _present(job.outputs.get("video")):
            out[(int(chapter), bool(job.outputs.get("finished")))] = job.id
    return out


def _jobs(connection: Any, production_id: int, tool: str, scope: str | None = None) -> list[store.Job]:
    """Settled jobs of `tool` in the production's runs, oldest first."""
    rows = connection.execute(
        "SELECT jobs.* FROM jobs JOIN runs ON runs.id = jobs.run_id WHERE runs.production_id = ? AND jobs.tool = ? "
        "AND jobs.state IN (?, ?) AND (? IS NULL OR runs.scope = ?) ORDER BY jobs.id",
        (production_id, tool, store.DONE, store.SKIPPED, scope, scope),
    ).fetchall()
    return [store.Job.of(row) for row in rows]


def _present(path: Any) -> bool:
    return isinstance(path, str) and (config.OUTPUT_DIR / path).is_file()


def _finish_of(connection: Any, chapter: store.Chapter) -> store.Job | None:
    found = [
        job
        for job in _jobs(connection, chapter.production_id, "finish", f"chapter:{chapter.id}")
        if _present(job.outputs.get("video"))
    ]
    return found[-1] if found else None


def _shot_of_render(connection: Any, reference: Any) -> store.Shot | None:
    """The shot a render job's output belongs to, while it still exists."""
    if not isinstance(reference, dict) or "from_job" not in reference:
        return None
    row = connection.execute(
        "SELECT clips.shot_id FROM jobs JOIN clips ON clips.id = jobs.for_id "
        "WHERE jobs.id = ? AND jobs.for_kind = 'clip'",
        (int(reference["from_job"]),),
    ).fetchone()
    return store.shot(connection, int(row[0])) if row else None


def _rendered(connection: Any, shot: store.Shot) -> store.Job:
    clip = store.clip_of(connection, shot.id)
    job = store.job(connection, clip.current_job) if clip.current_job is not None else None
    if (
        job is None
        or job.tool != "render_shot"
        or job.state not in (store.DONE, store.SKIPPED)
        or not _present(job.outputs.get("chunk"))
    ):
        raise ValueError(f"shot {shot.position + 1} has no rendered clip: render it first")
    return job


def queue(production_id: int, chapter_position: int | None = None) -> list[str]:
    """Join, level and subtitle one chapter or every one from the clips it has, and the film when every
    chapter is asked for. A chapter finished before is levelled as it is, not finished again."""
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        if production is None:
            raise LookupError(f"no production {production_id}")
        chapters = store.chapters_of(connection, production_id)
        if chapter_position is not None and not 0 <= chapter_position < len(chapters):
            raise LookupError(f"production {production_id} has {len(chapters)} chapter(s), not {chapter_position + 1}")
        wanted = chapters if chapter_position is None else [chapters[chapter_position]]
        rendered = {}
        for chapter in wanted:
            shots = store.shots_of(connection, chapter.id)
            if not shots:
                raise ValueError(f"{marker(chapter).lower()} has no shots")
            try:
                rendered[chapter.id] = (shots, [_rendered(connection, shot) for shot in shots])
            except ValueError as problem:
                raise ValueError(
                    f"{marker(chapter).lower()}: {problem}" if len(chapters) > 1 else str(problem)
                ) from None
        run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        what = "the film" if chapter_position is None else marker(wanted[0]).lower()
        store.add_run(
            connection, run_id, title=f"{production.title} · assemble {what}", spec={"production": production.id},
            scope=f"assemble:{production.id}", production_id=production.id,
        )  # fmt: skip
        finished: dict[int, int] = {}
        native: dict[int, int] = {}
        for chapter in wanted:
            shots, jobs = rendered[chapter.id]
            parts = {
                **{many: [_output(job.id, many[:-1]) for job in jobs] for many in ("chunks", "audios", "cuts", "keys")},
                **joins(connection, production, shots),
            }
            joined = store.add_job(
                connection, run_id, "assemble", {**parts, "name": _name(production, chapter, len(chapters))},
                label=f"{marker(chapter).lower()}: join", settings=chosen(production.id).data(),
            ).id  # fmt: skip
            native[chapter.id] = add_master(
                connection, run_id, production, chapter, _output(joined, "video"), parts,
                [lines(shot) for shot in shots], finished=False,
            )  # fmt: skip
            finish = _finish_of(connection, chapter)
            if finish is not None:
                finished[chapter.id] = add_master(
                    connection, run_id, production, chapter, _output(finish.id, "video"), finish.inputs,
                    [lines(_shot_of_render(connection, ref)) for ref in finish.inputs.get("audios", [])],
                    finished=True,
                )  # fmt: skip
        runs = [run_id]
        if chapter_position is None:
            film = film_after(connection, production, finished, native)
            if film is not None:
                runs.append(film)
        return runs


def status(production_id: int) -> dict[str, Any] | None:
    """The Assemble page: per chapter its joins, dialogue and newest masters, the film, and what is under way."""
    with store.connect() as connection:
        production = store.production(connection, production_id)
        if production is None:
            return None
        masters = {job.id: job for job in _jobs(connection, production_id, "master")}
        newest = _masters(connection, production_id)
        chapters = []
        for chapter in store.chapters_of(connection, production_id):
            shots = store.shots_of(connection, chapter.id)
            current = {store.clip_of(connection, shot.id).current_job for shot in shots}
            joined = joins(connection, production, shots)
            made = {}
            for finished in (False, True):
                job = masters.get(newest.get((chapter.id, finished), 0))
                if job is not None:
                    used = {int(ref["from_job"]) for ref in job.inputs["cuts"] if isinstance(ref, dict)}
                    made["finished" if finished else "native"] = {**_made(job), "current": used == current}
            chapters.append(
                {
                    "id": chapter.id, "position": chapter.position, "marker": marker(chapter),
                    "shots": len(shots), "boundaries": sum(f > 0 and not c for f, c in
                                                            zip(joined["fades"], joined["chained"], strict=True)),
                    "chained": sum(joined["chained"]), "lines": sum(len(lines(shot)) for shot in shots), **made,
                }
            )  # fmt: skip
        films = _jobs(connection, production_id, "film")
        film = next((job for job in reversed(films) if _present(job.outputs.get("video"))), None)
        busy = connection.execute(
            "SELECT count(*) FROM jobs JOIN runs ON runs.id = jobs.run_id WHERE runs.production_id = ? "
            "AND jobs.tool IN ('master', 'film') AND jobs.state IN (?, ?)",
            (production_id, store.QUEUED, store.RUNNING),
        ).fetchone()[0]
        return {
            "chapters": chapters,
            "film": None if film is None else _made(film),
            "busy": int(busy),
        }


def _made(job: store.Job) -> dict[str, Any]:
    cues = job.outputs.get("cues")
    return {
        **{name: value for name, value in job.outputs.items() if name != "cues"},
        **({"lines": len(cues)} if isinstance(cues, list) else {}),
        "job": job.id, "run": job.run_id, "hash": (job.key or "").rsplit("/", 1)[-1][:7],
    }  # fmt: skip
