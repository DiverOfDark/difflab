"""The VLM as script supervisor: what the gate asks about the meaning of the plan's and the writer's words."""

from __future__ import annotations

import json
from typing import Any

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


def _plain(text: str) -> str:
    return " ".join(text.lower().split())


def _fail(code: str, message: str) -> Problem:
    return Problem("error", code, message, True)


def events(idea: str, opening: str) -> list[str]:
    """The IDEA's events in its own words, from the IDEA and the opening alone: every repair of a plan is held to
    the same list."""
    actions = lint.actions_of(idea).strip()
    data = ask(
        "events",
        words.RULING_EVENTS_SYSTEM,
        words.RULING_EVENTS_USER.format(idea=actions, opening=opening.strip() or "-"),
        _object(events=_array(QUOTE)),
    )
    return list(dict.fromkeys(q for q in map(str, data["events"]) if q.strip()))


def story(idea: str, listed: list[tuple[str, int]], take: compose.Take) -> list[Problem]:
    """Every event of the IDEA shown by a moving beat with the IDEA's own verb, one question per event."""
    out, kept = lint.story_bounds(idea, listed, take)
    clips = "\n".join(
        f"Clip {index + 1}\n" + "\n".join(f"  {n}. [{b.phase}] {b.action}" for n, b in enumerate(chunk.beats, 1))
        for index, chunk in enumerate(take.chunks)
    )
    story = "\n".join(f'- "{event}" - clip {clip}' for event, clip in kept) or "(empty)"
    most = max((len(chunk.beats) for chunk in take.chunks), default=0)
    schema = _object(clip=_number(len(take.chunks)), beat=_number(most), faithful=FLAG)
    for wish in events(idea, take.opening):
        question = words.RULING_STORY_USER.format(clips=clips, story=story, event=f'"{wish}"')
        found = ask("story", words.RULING_STORY_SYSTEM, question, schema)
        clip, number = int(found["clip"]), int(found["beat"])
        beats = take.chunks[clip - 1].beats if 1 <= clip <= len(take.chunks) else ()
        if not 1 <= number <= len(beats):
            out.append(_fail("story", f"story: no beat's action shows the IDEA's '{wish}': show it in the action of "
                             "the beat where it happens, with the IDEA's own verb, and list it in story with that "
                             "clip"))  # fmt: skip
        elif (phase := beats[number - 1].phase) in lint.QUIET:
            out.append(_fail("story", f"story: the IDEA's '{wish}' happens in the {phase} beat of clip {clip}, which "
                             "holds still: move it to a build, primary, handoff or reaction beat"))  # fmt: skip
        elif not found["faithful"]:
            out.append(_fail("story", f"story: beat {number} of clip {clip} shows the IDEA's '{wish}' with another "
                             "action or count: say it with the IDEA's own verb and count"))  # fmt: skip
    return out


def mouths(take: compose.Take) -> list[Problem]:
    """Each clip's "mouths" against its beats, both ways, one question per clip. The settle is written from the
    closing, so it is left out."""
    out = []
    for index, chunk in enumerate(take.chunks):
        beats, entries = compose.written(chunk), [str(entry) for entry in chunk.mouths]

        def who(beat: compose.Beat) -> str:
            return ", ".join(
                f"{{{m}}} ({take.cards[m].possessive})" if m in take.cards else f"{{{m}}}" for m in beat.who
            )

        question = words.RULING_MOUTHS_USER.format(
            beats="\n".join(f"{n}. [{b.phase}] who: {who(b) or '-'} - {b.action}" for n, b in enumerate(beats, 1)),
            mouths="\n".join(f'{n}. "{entry}"' for n, entry in enumerate(entries, 1)) or "(none)",
        )
        opening = _object(beat=_number(len(beats)), quote=QUOTE, what=QUOTE, entries=_array(_number(len(entries))))
        schema = _object(openings=_array(opening), mouths=_array(_object(opens=FLAG), len(entries)))
        data = ask("mouths", words.RULING_MOUTHS_SYSTEM, question, schema)
        clip, done, unlisted = f"clip {index + 1}", set(), dict[str, str]()
        for found in data["openings"]:
            number, what = int(found["beat"]), str(found["what"])
            named = {int(n) for n in found["entries"]} & set(range(1, len(entries) + 1))
            if not 1 <= number <= len(beats) or _plain(str(found["quote"])) not in _plain(beats[number - 1].action):
                continue
            if named:
                done |= named
            else:
                unlisted.setdefault(_plain(what), f"{clip}: beat {number}'s action has {what}, which opens a mouth: "
                                    f"list it in mouths ('{what}')")  # fmt: skip
        for n, (said, verdict) in enumerate(zip(entries, data["mouths"], strict=True), 1):
            if not verdict["opens"]:
                out.append(_fail("mouths", f"{clip}: '{said}' opens no mouth (a smile keeps the lips closed): leave "
                                 "it out of mouths"))  # fmt: skip
            elif n not in done:
                out.append(_fail("mouths", f"{clip}: mouths has '{said}', but no beat's action has it done: add it "
                                 "to the beat where it happens, or leave it out of mouths"))  # fmt: skip
        out += [_fail("mouths", message) for message in unlisted.values()]
    return out


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
    return out


def traits(member: direct.Member, members: dict[str, direct.Member], listed: Any) -> list[Problem]:
    """A card's traits that are not its own lasting look: another member's part, a thing held, a passing state."""
    others = [m for m in members.values() if m.id != member.id]
    phrases = [str(trait.get("phrase", "")) for trait in listed if isinstance(trait, dict)]
    if not phrases:
        return []
    question = words.RULING_TRAITS_USER.format(
        member=f"{member.id}, the {member.noun} ({member.kind})",
        others="\n".join(f"- {m.id}: the {m.noun} ({m.kind})" for m in others) or "(none)",
        traits="\n".join(f'{n}. "{phrase}"' for n, phrase in enumerate(phrases, 1)),
    )
    verdict = _object(
        verdict={"type": "string", "enum": ["own", "other", "held", "moment"]},
        of={"type": "string", "enum": ["", *(m.id for m in others)]},
    )
    data = ask("traits", words.RULING_TRAITS_SYSTEM, question, _object(traits=_array(verdict, len(phrases))))
    out = []
    for phrase, found in zip(phrases, data["traits"], strict=True):
        if found["verdict"] == "other" and found["of"]:
            out.append(_fail("answer", f"{member.id}: '{phrase}' is part of {found['of']}, which has its own card: "
                             "leave it out"))  # fmt: skip
        elif found["verdict"] == "held":
            out.append(_fail("answer", f"{member.id}: '{phrase}' is held, not worn: leave it out"))
        elif found["verdict"] == "moment":
            out.append(_fail("answer", f"{member.id}: '{phrase}' is a state of the photo's moment, which the shot "
                             "changes: record the look, not the pose of the moment"))  # fmt: skip
    return out


def leaving(take: compose.Take) -> list[Problem]:
    """A camera that travels with a member, over a closing that has it going away from the camera."""
    out = []
    for index, chunk in enumerate(take.chunks):
        if chunk.camera_move not in ("follow", "crane_up") or chunk.camera_toward not in take.cards:
            continue
        card = take.cards[chunk.camera_toward]
        question = words.RULING_LEAVING_USER.format(
            member=f"{{{card.id}}}, the {card.noun}", move=chunk.camera_move, closing=chunk.closing
        )
        if ask("leaving", words.RULING_LEAVING_SYSTEM, question, _object(away=FLAG))["away"]:
            out.append(_fail("camera", f"clip {index + 1}: its closing has {{{card.id}}} going away from the camera, "
                             f"which \"{chunk.camera_move}\" keeps close: use \"stay\" (rule 16)"))  # fmt: skip
    return out
