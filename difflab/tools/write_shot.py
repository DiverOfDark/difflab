"""One call per shot: prose only, around the sentences the compiler already owns. Then the gate
over the whole chapter."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import replace
from typing import Any

from difflab.models.exl3vl import ANY_JSON
from difflab.models.minimax_h3 import caption
from difflab.screenplay import checks as lint
from difflab.screenplay import compiler as compose
from difflab.screenplay import config as h3
from difflab.screenplay import words
from difflab.tools import direct, ruling

PROSE = {"type": "string", "maxLength": h3.DIRECTOR_WRITE_CHARS}


def shape(count: int) -> dict[str, Any]:
    """The writer's answer with exactly `count` beats: a beat cut off at the cap once closed the array after it."""
    beats = {"type": "array", "items": PROSE, "minItems": count, "maxItems": count}
    return {
        **ANY_JSON,
        "type": "object",
        "properties": {"beats": beats, "summary": PROSE, "sounds": PROSE},
        "required": ["beats", "summary", "sounds"],
        "additionalProperties": False,
    }


def trimmed(text: str) -> str:
    """`text` cut to the whole sentences that fit DIRECTOR_WRITE_WORDS; the first is kept whatever its length."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    kept = sentences[:1]
    for sentence in sentences[1:]:
        if len(" ".join([*kept, sentence]).split()) > h3.DIRECTOR_WRITE_WORDS or sentence[-1] not in ".!?":
            break
        kept.append(sentence)
    return " ".join(kept)


def _with_prose(take: compose.Take, index: int, data: Any) -> compose.Take:
    chunk = take.chunks[index]
    texts = iter(str(text) for text in data["beats"])
    wanted = compose.written(chunk)
    beats = tuple(dataclasses.replace(beat, prose=next(texts)) if beat in wanted else beat for beat in chunk.beats)
    filled = dataclasses.replace(chunk, beats=beats, summary=str(data["summary"]), sounds=str(data["sounds"]))
    return dataclasses.replace(take, chunks=(*take.chunks[:index], filled, *take.chunks[index + 1 :]))


def _directive(take: compose.Take, index: int) -> str:
    if compose.carried(take, index):
        return "[video continuation"
    return caption.directive(False, not compose.unframed(take, index))


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
        f"{c.id} | {c.noun} | {c.full()}"
        + (f", and now {now.before}" if (now := chunk.states.get(c.id)) and now.before else "")
        for c in (take.cards[m] for m in take.members if m in chunk.cast)
    )
    written = compose.written(chunk)
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
        to_write = compose.written(planned)
        if len(data["beats"]) != len(to_write):
            return [direct.problem(f'"beats" has exactly {len(to_write)} strings, one per listed beat')]
        data["beats"] = [trimmed(str(text)) for text in data["beats"]]  # `ask_json` returns this same dict
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
        # A {say} with no line behind it reached the compiler as a KeyError on the speaker, which the
        # repair read as "does not fit the shape" and could not fix: a giggle written as {woman} {say}.
        for number, (beat, text) in enumerate(zip(to_write, data["beats"], strict=True), 1):
            lines = sum(beat.start <= line.start < beat.end for line in planned.dialogue)
            if str(text).count("{say}") != lines:
                found.append(direct.problem(f"beat {number} has {lines} line(s) of dialogue, so write {{say}} "
                                            f"{lines} time(s) there; a laugh, giggle or sigh is written as the "
                                            "motion and sound it is, with no {say}"))  # fmt: skip
        if not set(re.findall(r"\{([a-z_]+)\}", str(data["summary"]))) & living:
            found.append(direct.problem('the "summary" names the members who act as {id}, in braces'))
        prose = [str(text) for text in [*data["beats"], data["summary"]]]
        found += ruling.voices(prose, str(data["sounds"]), planned.mouths)
        return found or chunk_problems(_with_prose(before, index, data), index)

    data = direct.ask_json(
        words.WRITE_SYSTEM, question, [], gate, f"write · chunk {index + 1}",
        h3.DIRECTOR_WRITE_TOKENS, h3.DIRECTOR_WRITE_TEMPERATURE, schema=shape(len(written)),
    )  # fmt: skip
    return _with_prose(take, index, data)
