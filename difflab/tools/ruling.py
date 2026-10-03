"""The VLM as script supervisor: the gate's questions of meaning, which word lists answered before."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from difflab.core import record
from difflab.core.cache import cache
from difflab.core.keys import key
from difflab.models.exl3vl import ANY_JSON
from difflab.models.minimax_h3.caption import Problem
from difflab.screenplay import checks as lint
from difflab.screenplay import compiler as compose
from difflab.screenplay import config as h3
from difflab.screenplay import words
from difflab.tools import direct

QUOTE = {"type": "string", "maxLength": 200}
FLAG = {"type": "boolean"}


def _object(**properties: Any) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _array(item: dict[str, Any], count: int | None = None) -> dict[str, Any]:
    return {"type": "array", "items": item, **({"minItems": count, "maxItems": count} if count is not None else {})}


def _number(most: int) -> dict[str, Any]:
    return {"type": "integer", "minimum": 0, "maximum": max(most, 0)}


def ask(title: str, system: str, question: str, schema: dict[str, Any]) -> Any:
    """Greedy, and kept by its question: the gate re-checks every answer it keeps, on every run."""
    shape = {**ANY_JSON, **schema}
    kept = cache.path(
        key("ruling", 1, model=direct.MODEL.get(), system=system, question=question, schema=shape), ".json"
    )
    if cache.enabled and kept.exists():
        return json.loads(kept.read_text(encoding="utf-8"))
    raw = direct.ask(system, question, [], f"ruling · {title}", h3.RULING_TOKENS, schema=shape)
    try:
        data = direct.parse_json(raw)
    except ValueError as error:
        raise direct.DirectorError(f"ruling · {title}: {error}") from None
    if cache.enabled:
        kept.write_text(json.dumps(data), encoding="utf-8")
    return data


def _decide(check: str, word_lists: Callable[[], list[Problem]], ruled: list[Problem], question: str) -> list[Problem]:
    if h3.RULINGS == "on":
        return ruled
    listed = word_lists()
    if [p.message for p in listed] != [p.message for p in ruled]:
        record.event("ruling", round(time.time(), 3), {
            "check": check, "word_lists": [p.message for p in listed], "ruling": [p.message for p in ruled],
            "question": question,
        })  # fmt: skip
    return listed


def _plain(text: str) -> str:
    return " ".join(text.lower().split())


def _fail(code: str, message: str) -> Problem:
    return Problem("error", code, message, True)


def story(idea: str, events: list[tuple[str, int]], take: compose.Take) -> list[Problem]:
    """Every event of the IDEA is in the story, shown by a moving beat of its clip with the IDEA's own verb."""
    out, kept = lint.story_bounds(idea, events, take)
    clips = "\n".join(
        f"Clip {index + 1}\n" + "\n".join(f"  {n}. [{b.phase}] {b.action}" for n, b in enumerate(chunk.beats, 1))
        for index, chunk in enumerate(take.chunks)
    )
    listed = "\n".join(f'{n}. "{event}" - clip {clip}' for n, (event, clip) in enumerate(kept, 1))
    question = words.RULING_STORY_USER.format(idea=lint.actions_of(idea).strip(), clips=clips, events=listed)
    most = max((len(chunk.beats) for chunk in take.chunks), default=0)
    schema = _object(events=_array(_object(beat=_number(most), faithful=FLAG), len(kept)), missing=_array(QUOTE))
    data = ask("story", words.RULING_STORY_SYSTEM, question, schema)
    for (event, clip), verdict in zip(kept, data["events"], strict=True):
        beats, number = take.chunks[clip - 1].beats, int(verdict["beat"])
        if not 1 <= number <= len(beats):
            out.append(_fail("story", f"story: '{event}' is in clip {clip}, but no beat's action there shows it: say "
                             "it in the action of the beat that shows it, with the IDEA's own verb"))  # fmt: skip
        elif beats[number - 1].phase in lint.QUIET:
            out.append(_fail("story", f"story: '{event}' happens in an airlock or settle beat of clip {clip}, which "
                             "hold still: move it to a build, primary, handoff or reaction beat"))  # fmt: skip
        elif not verdict["faithful"]:
            out.append(_fail("story", f"story: '{event}' is beat {number} of clip {clip}, but its action does "
                             "something else: say it with the IDEA's own verb and count"))  # fmt: skip
    out += [
        _fail("story", f"story: the IDEA's '{quote}' is in no event; add it with the clip that shows it")
        for quote in map(str, data["missing"])
        if quote.strip() and _plain(quote) in _plain(idea)
    ]
    return _decide("story", lambda: lint.story_rules(idea, events, take), out, question)


def mouths(take: compose.Take) -> list[Problem]:
    """Each clip's "mouths" against its beats, both ways. The settle is written from the closing, so it is left out."""
    written = [compose.written(chunk) for chunk in take.chunks]

    def who(beat: compose.Beat) -> str:
        return ", ".join(f"{{{m}}} ({take.cards[m].possessive})" if m in take.cards else f"{{{m}}}" for m in beat.who)

    clips = "\n".join(
        f"Clip {index + 1}\n"
        + "\n".join(f"  {n}. [{b.phase}] who: {who(b) or '-'} - {b.action}" for n, b in enumerate(beats, 1))
        for index, beats in enumerate(written)
    )
    entries = [(index, str(entry)) for index, chunk in enumerate(take.chunks) for entry in chunk.mouths]
    listed = "\n".join(f'{n}. clip {index + 1}: "{entry}"' for n, (index, entry) in enumerate(entries, 1)) or "(none)"
    question = words.RULING_MOUTHS_USER.format(clips=clips, mouths=listed)
    most = max((len(beats) for beats in written), default=0)
    unlisted = _object(clip=_number(len(take.chunks)), beat=_number(most), what=QUOTE)
    schema = _object(mouths=_array(_object(smile=FLAG, beat=_number(most)), len(entries)), unlisted=_array(unlisted))
    data = ask("mouths", words.RULING_MOUTHS_SYSTEM, question, schema)
    out = []
    for (index, entry), verdict in zip(entries, data["mouths"], strict=True):
        if verdict["smile"]:
            out.append(_fail("mouths", f"clip {index + 1}: '{entry}' - a smile keeps the lips closed; leave it out "
                             "of mouths"))  # fmt: skip
        elif not 1 <= int(verdict["beat"]) <= len(written[index]):
            out.append(_fail("mouths", f"clip {index + 1}: mouths has '{entry}', but no beat's action has it done: "
                             "add it to the beat where it happens, or leave it out of mouths"))  # fmt: skip
    for found in data["unlisted"]:
        clip, what = int(found["clip"]), str(found["what"])
        if 1 <= clip <= len(take.chunks):
            out.append(_fail("mouths", f"clip {clip}: beat {found['beat']}'s action has {what}, which opens a mouth: "
                             f"list it in mouths ('{what}')"))  # fmt: skip
    return _decide("mouths", lambda: lint.mouth_rules(take), out, question)


def voices(prose: list[str], sounds: str, allowed: tuple[str, ...]) -> list[Problem]:
    """The writer's mouths against the clip's mouth rule, both ways; `prose` is its beats, then its summary."""
    listed = "\n".join(f'{n}. "{entry}"' for n, entry in enumerate(allowed, 1)) or "(none: every mouth stays closed)"
    question = words.RULING_VOICES_USER.format(
        allowed=listed, prose="\n".join(f"- {text}" for text in prose), sounds=sounds.strip() or "-"
    )
    schema = _object(allowed=_array(_object(shown=FLAG, audible=FLAG, heard=FLAG), len(allowed)), extra=_array(QUOTE))
    data = ask("voices", words.RULING_VOICES_SYSTEM, question, schema)
    out = []
    for entry, verdict in zip(allowed, data["allowed"], strict=True):
        if not verdict["shown"]:
            out.append(_fail("mouths", f"the mouth rule allows '{entry}': show it in the beat where it happens"))
        if verdict["audible"] and not verdict["heard"]:
            out.append(_fail("mouths", f"the mouth rule allows '{entry}': put its sound in \"sounds\""))
    written = _plain(" ".join([*prose, sounds]))
    out += [
        _fail("mouths", f"'{quote}' is a mouth movement the mouth rule does not allow: leave it out")
        for quote in map(str, data["extra"])
        if quote.strip() and _plain(quote) in written
    ]
    return _decide("voices", lambda: lint.vocal_rules(prose, sounds, allowed), out, question)
