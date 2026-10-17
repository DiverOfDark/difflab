"""A production made from comic pages: one shot per panel, in reading order, each opening on its panel
redrawn as an anime frame and landing on the next panel where the reader says the action carries on.

A plan run reads every page, names the cast once over all of them, then lays the panels out as shots;
that job queues the rest of the run, since only then is the number of panels known: each panel cleaned
of its lettering, the bible read from the cleaned panels, one colour sheet per character, and the chapter's
shots filled by the director.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from difflab import store
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import config as knobs
from difflab.screenplay import settings
from difflab.tools import read_page
from difflab.tools.plan_shots import on_grid

SHEET = store.SHEET
KIND = "manga"


def is_manga(production: store.Production | None) -> bool:
    return production is not None and production.story_kind == KIND


def shortest(longest: float) -> float:
    return on_grid(knobs.MANGA_SHOT_MIN, longest)


def seconds_for(script: read_page.Script, longest: float) -> tuple[float, bool]:
    """A panel's length from its lettering, and whether that lettering fits the longest shot at all."""
    said = sum(len(b.text.split()) for b in script.balloons if b.kind in read_page.SPOKEN)
    read = sum(len(b.text.split()) for b in script.balloons if b.kind not in read_page.SPOKEN)
    lettered = said / knobs.DIRECTOR_WORDS_PER_SECOND + read / knobs.MANGA_READ_WORDS_PER_SECOND
    wanted = lettered + knobs.DIRECTOR_DIALOGUE_TAIL if lettered else 0.0
    floor = shortest(longest)
    if wanted <= floor:
        return floor, True
    seconds = on_grid(wanted, longest)
    while seconds < wanted and seconds < longest:
        seconds = on_grid(seconds + 1 / 24 * 17, longest)
    return seconds, wanted <= longest + 1e-6


def story_of(scripts: list[read_page.Script]) -> str:
    """The panels as text: what the chapter view shows, and what the survey reads."""
    lines = []
    for number, script in enumerate(scripts, 1):
        said = " ".join(
            f'{b.speaker or "narration"} ({b.kind}): "{b.text}"' for b in script.balloons
        )  # fmt: skip
        lines.append(f"{number}. {script.action}" + (f" {said}" if said else ""))
    return "\n".join(lines)


def chapter_pages(manifest: dict[str, Any]) -> list[list[int]]:
    pages = list(manifest.get("pages", []))
    given = [list(map(int, c)) for c in manifest.get("chapter_pages", [])]
    return given or [list(range(len(pages)))]


def plan_production(
    connection: Any, production: store.Production, chapter_position: int | None, asked_by: str, fresh: bool,
    shots: bool,
) -> list[str]:  # fmt: skip
    """One plan run reads every page and lays out every chapter; with `shots`, it fills the chosen chapter, or
    each, the first run's rows reused by key in the runs after it."""
    chapters = store.chapters_of(connection, production.id)
    if not production.manifest.get("pages"):
        raise ValueError("a manga production needs its pages")
    chosen = chapters if chapter_position is None else [chapters[chapter_position]]
    if not shots:
        chosen = chapters[:1]
    return [_queue(connection, production, chapter, asked_by, fresh, shots) for chapter in chosen]


def _queue(
    connection: Any, production: store.Production, chapter: store.Chapter, asked_by: str, fresh: bool, shots: bool
) -> str:
    manifest = production.manifest
    pages = [str(p) for p in manifest["pages"]]
    reading = str(manifest.get("reading") or "rtl")
    video, _ = settings.effective("render_chunk", production.id)
    assert isinstance(video, VideoSettings)
    run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    spec = {"production": production.id, "chapter": chapter.id, "scope": "plan", "fresh": fresh}
    title = f"{production.title} · {'plan' if shots else 'bible'}"
    count = len(store.chapters_of(connection, production.id))
    if shots and count > 1:
        title += f" chapter {chapter.position + 1}"
    store.add_run(
        connection, run_id, title=title, spec=spec, asked_by=asked_by, scope=f"plan:{chapter.id}",
        production_id=production.id,
    )  # fmt: skip

    def add(tool: str, inputs: dict[str, Any], label: str) -> int:
        chosen = settings.for_job(tool, production.id)
        return store.add_job(
            connection, run_id, tool, {"production": production.id, **inputs}, label=label, settings=chosen
        ).id

    for reference in manifest.get("references", []):
        add("guess_reference", {"path": str(reference["path"])}, "describe the style picture")
    reads: list[int] = []
    for group in chapter_pages(manifest):
        for at, index in enumerate(group):
            last = at == len(group) - 1
            reads.append(
                add(
                    "read_page",
                    {
                        "page": pages[index], "index": index, "reading": reading,
                        "next": None if last else pages[group[at + 1]],
                        "names": {"from_job": reads[-1], "output": "names"} if reads else [],
                    },
                    f"read page {index + 1}",
                )
            )  # fmt: skip
    found = [{"from_job": job, "output": "panels"} for job in reads]
    cast = add("cast_panels", {"pages": found}, "name the cast")
    add(
        "lay_panels",
        {
            "pages": found, "cast": {"from_job": cast, "output": "characters"}, "chapter": chapter.id,
            "shots": shots, **({"take": int(datetime.now().timestamp()) % 1_000_000_000} if fresh else {}),
        },
        "lay the panels out as shots",
    )  # fmt: skip
    return run_id
