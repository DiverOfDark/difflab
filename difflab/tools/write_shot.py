"""One call per shot: prose only, around the sentences the compiler already owns. Then the gate
over the whole chapter."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import replace
from typing import Any

from difflab.core import ui
from difflab.models.minimax_h3 import caption
from difflab.screenplay import checks as lint
from difflab.screenplay import compiler as compose
from difflab.screenplay import config as h3
from difflab.screenplay import words
from difflab.tools import direct


def _with_prose(take: compose.Take, index: int, data: Any) -> compose.Take:
    chunk = take.chunks[index]
    texts = iter(str(text) for text in data["beats"])
    beats = tuple(
        beat if beat.phase == "settle" else dataclasses.replace(beat, prose=next(texts)) for beat in chunk.beats
    )
    filled = dataclasses.replace(chunk, beats=beats, summary=str(data["summary"]), sounds=str(data["sounds"]))
    return dataclasses.replace(take, chunks=(*take.chunks[:index], filled, *take.chunks[index + 1 :]))


def _directive(take: compose.Take, index: int) -> str:
    return "[video continuation" if take.tail and index else "[keyframe completion + reference generation]"


def chunk_problems(take: compose.Take, index: int) -> list[caption.Problem]:
    compiled = compose.compile_chunk(take, index)
    labels = compiled.presentation.labels()
    found = caption.ide_rules(compiled.prompt) + lint.chunk_rules(compiled.prompt, labels, _directive(take, index))
    # A take is up to forty chunks and the gate reports them in one list: a problem that does not say
    # where it is sends its reader through every chunk in the folder.
    return [replace(problem, message=f"chunk {index + 1}: {problem.message}") for problem in found]


def write(take: compose.Take) -> compose.Take:
    """Every chunk, in order."""
    for index in range(len(take.chunks)):
        take = write_one(take, index)
    return take


def write_one(take: compose.Take, index: int) -> compose.Take:
    """One call for chunk `index`: prose only, around the sentences the compiler already owns."""
    chunk = take.chunks[index]
    cast = "\n".join(
        f"{c.id} | {c.noun} | {c.full()}" for c in (take.cards[m] for m in take.members if m in chunk.cast)
    )
    written = [beat for beat in chunk.beats if beat.phase != "settle"]
    beats = "\n".join(
        words.WRITE_BEAT.format(
            number=i + 1,
            phase=b.phase,
            length=b.end - b.start,
            who=", ".join(f"{{{m}}}" for m in b.who),
            action=b.action,
        )
        + "".join(f" (dialogue: {{{line.who}}} {{say}})" for line in chunk.dialogue if b.start <= line.start < b.end)
        for i, b in enumerate(written)
    )
    mouths = (
        words.SENTENCES["mouths"].format(actions=compose.join(chunk.mouths))
        if chunk.mouths
        else words.SENTENCES["mouths_none"]
    )
    question = words.WRITE_USER.format(
        cast=cast, opening=compose.opening(take, index), closing=chunk.closing,
        camera=words.CAMERA_MOVES[chunk.camera_move].format(toward="{" + chunk.camera_toward + "}"),
        mouths=mouths, purpose=chunk.purpose, count=len(written), beats=beats,
    )  # fmt: skip

    def gate(data: Any, index: int = index, before: compose.Take = take) -> list[caption.Problem]:
        planned = before.chunks[index]
        to_write = [beat for beat in planned.beats if beat.phase != "settle"]
        if len(data["beats"]) != len(to_write):
            return [direct.problem(f'"beats" has exactly {len(to_write)} strings, one per listed beat')]
        where = f"write · chunk {index + 1}"
        living = {m for m in planned.cast if direct.member_card(before.cards, m, where).kind in compose.LIVING}
        found = [
            problem
            for beat, text in zip(to_write, data["beats"], strict=True)
            for problem in lint.writer_rules(str(text), set(planned.cast), set(beat.who) & living)
        ]
        found += lint.writer_rules(str(data["summary"]), set(planned.cast)) + lint.writer_rules(
            str(data["sounds"]), set(planned.cast)
        )
        if not set(re.findall(r"\{([a-z_]+)\}", str(data["summary"]))) & living:
            found.append(direct.problem('the "summary" names the members who act as {id}, in braces'))
        prose = [str(text) for text in [*data["beats"], data["summary"]]]
        found += lint.vocal_rules(prose, str(data["sounds"]), planned.mouths)
        return found or chunk_problems(_with_prose(before, index, data), index)

    data = direct.ask_json(
        words.WRITE_SYSTEM, question, [], gate, f"write · chunk {index + 1}",
        h3.DIRECTOR_WRITE_TOKENS, h3.DIRECTOR_WRITE_TEMPERATURE,
    )  # fmt: skip
    return _with_prose(take, index, data)


# --- compile, check, save ------------------------------------------------------------------------------
def check_take(take: compose.Take) -> tuple[list[compose.Compiled], list[caption.Problem]]:
    """Compile every chunk and run the whole gate once more, across chunks too.

    What it finds is shown and the take goes on: every rule here is one a stage already asked its
    model to meet with repairs, so an answer still breaking it was kept on purpose, and refusing now
    would throw away the whole write stage over what that stage settled.
    """
    compiled = [compose.compile_chunk(take, index) for index in range(len(take.chunks))]
    problems = lint.cross_chunk_rules(take, [c.prompt for c in compiled]) + lint.plan_rules(take)
    for index in range(len(take.chunks)):
        problems += chunk_problems(take, index)
    if problems:
        rows = {f"{p.severity} · {p.code} {i}": p.message for i, p in enumerate(problems)}
        ui.panel("director: the gate", rows, color="red" if caption.errors(problems) else "yellow")
    return compiled, problems
