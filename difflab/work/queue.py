"""A run from a job: the tool jobs it takes, in dependency order, as rows the worker will pick up.

No model is imported here, so the server can enqueue without torch.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from difflab import store
from difflab.screenplay import settings
from difflab.work.quick import Job


@dataclass(frozen=True)
class JobSpec:
    tool: str
    inputs: dict[str, Any]
    label: str = ""
    settings: dict[str, Any] = field(default_factory=dict)


def output_of(position: int, name: str) -> dict[str, Any]:
    """An input that names an earlier job's output, by its position in the same run."""
    return {"from_position": position, "output": name}


def plan(job: Job) -> list[JobSpec]:
    """Which tool jobs a quick job is."""
    if job.mode == "picture":
        return _picture(job)
    jobs: list[JobSpec] = []
    photo, cut = _fitted(job, str(job.photos[0]) if job.photos else None, jobs)
    if job.mode == "video":
        if job.idea:
            jobs.append(
                JobSpec(
                    "write_video_prompt",
                    {"photo": photo, "idea": job.idea, **cut},
                    "write the prompt",
                    job.video_vlm.data(),
                )
            )
            prompt: Any = output_of(len(jobs) - 1, "text")
        else:
            prompt = job.prompt
        references = [str(p) for p in job.photos[1:]]
        jobs.append(
            JobSpec(
                "render_clip",
                {"photo": photo, "prompt": prompt, "references": references, **cut},
                "render the clip",
                job.video.data(),
            )
        )
    elif job.mode == "upscale":
        jobs.append(
            JobSpec(
                "enlarge_pixels",
                {"video": str(job.clip), "megapixels": job.finish_megapixels},
                "enlarge",
                settings.effective("enlarge_pixels")[0].data(),
            )
        )
    else:
        raise ValueError(f"mode {job.mode!r} is not one a run is planned from")
    # The finishes, each on the video the job before it made.
    if job.grade_strength is not None:
        jobs.append(
            JobSpec("grade", {"video": output_of(len(jobs) - 1, "video"), "strength": job.grade_strength}, "grade")
        )
    if job.retime_fps is not None:
        jobs.append(
            JobSpec(
                "retime",
                {"video": output_of(len(jobs) - 1, "video"), "fps": job.retime_fps},
                f"retime to {job.retime_fps} fps",
                settings.effective("retime")[0].data(),
            )
        )
    return jobs


def _fitted(job: Job, photo: Any, jobs: list[JobSpec]) -> tuple[Any, dict[str, Any]]:
    """The photo brought to the job's aspect: drawn out by a job of its own, or a cut its readers make."""
    if not job.aspect:
        # Only when given, so a job without one keeps the inputs every earlier run recorded.
        return photo, {}
    if job.fit == "crop":
        return photo, {"aspect": job.aspect}
    jobs.append(JobSpec("extend_photo", {"photo": photo, "aspect": job.aspect}, "draw the photo out", job.edit.data()))
    return output_of(len(jobs) - 1, "image"), {}


def _picture(job: Job) -> list[JobSpec]:
    """A start photo is edited; without one, the words and any references are drawn on a fresh canvas."""
    jobs: list[JobSpec] = []
    photo = str(job.photos[0]) if job.start else None
    references = [str(p) for p in job.photos[1 if job.start else 0 :]]
    # Only when there are any: a job's inputs are its row, and a run without references reads as it did.
    given: dict[str, Any] = {"references": references} if references else {}
    if photo is None:
        if job.idea:
            jobs.append(
                JobSpec("write_image_prompt", {"idea": job.idea, **given}, "write the description", job.vlm.data())
            )
            prompt_: Any = output_of(0, "text")
            aspect: Any = job.aspect or output_of(0, "aspect")
        else:
            prompt_, aspect = job.prompt, job.aspect or "1:1"
        jobs.append(
            JobSpec("create_image", {"prompt": prompt_, "aspect": aspect, **given}, "draw the picture", job.edit.data())
        )
        return jobs
    # ref2va has no opening to bring to an aspect: its canvas is the first reference's.
    cut: dict[str, Any] = {}
    if job.edit.engine == "minimax-h3" and job.edit.opening_h3 == "free":
        photo, references = None, [str(p) for p in job.photos]
    else:
        photo, cut = _fitted(job, photo, jobs)
    if job.edit.engine == "minimax-h3":
        if job.idea:
            jobs.append(
                JobSpec(
                    "write_video_prompt",
                    {
                        "photo": photo,
                        "idea": job.idea,
                        "still": True,
                        **cut,
                        **({} if photo else {"references": references}),
                    },
                    "write the prompt",
                    job.vlm.data(),
                )
            )
            prompt_ = output_of(len(jobs) - 1, "text")
        else:
            prompt_ = job.prompt
        jobs.append(
            JobSpec(
                "render_clip",
                {"photo": photo, "prompt": prompt_, "references": references, "still": True, **cut},
                "render the edit",
                job.video.with_(duration=job.edit.seconds_h3, megapixels=job.edit.megapixels_h3).data(),
            )
        )
        return jobs
    if job.idea:
        jobs.append(
            JobSpec(
                "write_instruction",
                {"photo": photo, "idea": job.idea, "engine": job.edit.engine, **given, **cut},
                "write the instruction",
                job.vlm.data(),
            )
        )
        instruction: Any = output_of(len(jobs) - 1, "text")
    else:
        instruction = job.prompt
    jobs.append(
        JobSpec(
            "edit_image",
            {"photo": photo, "instruction": instruction, **given, **cut},
            "edit the photo",
            job.edit.data(),
        )
    )
    return jobs


def enqueue(job: Job, asked_by: str = "person") -> str:
    """The run's rows, in one transaction. Returns the run's id, which is also its directory's name."""
    problems = job.check()
    if problems:
        raise ValueError(problems[0])
    with store.connect() as connection, store.transaction(connection):
        run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        store.add_run(connection, run_id, title=job.title, spec=job.data(), asked_by=asked_by)
        ids: list[int] = []
        for spec in plan(job):
            inputs = {name: _bound(value, ids) for name, value in spec.inputs.items()}
            made = store.add_job(connection, run_id, spec.tool, inputs, label=spec.label, settings=spec.settings)
            ids.append(made.id)
    return run_id


def _bound(value: Any, ids: list[int]) -> Any:
    """Positions become ids once the rows exist: `{"from_position": n}` -> `{"from_job": id}`."""
    if isinstance(value, dict) and "from_position" in value:
        return {"from_job": ids[int(value["from_position"])], "output": value["output"]}
    return value


def enqueue_chat(connection: sqlite3.Connection, message_id: int, title: str) -> str:
    """One run for one reply, so it waits its turn on the card like everything else. Inside the caller's
    transaction: a reply seen without its run is one whose run is gone."""
    run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    store.add_run(connection, run_id, title=f"chat · {title}", scope="chat")
    store.add_job(connection, run_id, "chat", {"message": message_id}, label="reply", settings=settings.for_job("chat"))
    return run_id
