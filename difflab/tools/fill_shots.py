"""The director filling shots it is given: a comic's panels, one clip each, their length, cast and words fixed.

One call per chapter, like `plan_shots`, so the model sees every join. The balloons are never retyped: the
model times each by its number and the program puts the comic's words in, so every line is said word for
word, once, in its own shot."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from difflab.core import ui
from difflab.models.minimax_h3 import caption
from difflab.screenplay import checks as lint
from difflab.screenplay import compiler as compose
from difflab.screenplay import config as h3
from difflab.screenplay import words
from difflab.tools import direct, read_page, ruling

PHASES = ("carry", "build", "primary", "reaction", "settle")
BEATS = (1, 4)


@dataclass(frozen=True)
class Given:
    """One panel as a clip: what the director may not change."""

    seconds: float
    cast: tuple[str, ...]  # the people and animals drawn in it, as member ids
    action: str
    balloons: tuple[read_page.Balloon, ...]  # every balloon, speakers as member ids
    lands: bool  # the clip ends on the next panel
    landed: bool  # the clip before ends on this one

    @property
    def spoken(self) -> tuple[read_page.Balloon, ...]:
        return tuple(b for b in self.balloons if b.kind in read_page.SPOKEN)


def _said(balloon: read_page.Balloon) -> float:
    return len(balloon.text.split()) / h3.DIRECTOR_WORDS_PER_SECOND


def _cast(given: list[Given], index: int) -> tuple[str, ...]:
    """Who is in a clip: its panel's people, and the next panel's when it lands there, since they come in."""
    panel = given[index]
    after = given[index + 1].cast if panel.lands and index + 1 < len(given) else ()
    return tuple(dict.fromkeys([*panel.cast, *after]))


def take_from(data: Any, base: compose.Take, given: list[Given]) -> compose.Take:
    chunks = []
    voices: dict[str, str] = {}
    for index, (clip, panel) in enumerate(zip(data["clips"], given, strict=True)):
        place = str(clip["place"])
        beats = tuple(
            compose.Beat(
                float(b["start"]), float(b["end"]), str(b["phase"]), tuple(str(w) for w in b["who"]), str(b["action"])
            )
            for b in clip["beats"]
        )
        if beats and abs(beats[-1].end - panel.seconds) <= 0.3:
            beats = (*beats[:-1], dataclasses.replace(beats[-1], end=round(panel.seconds, 3)))
        lines = []
        for entry in clip.get("dialogue", []):
            balloon = panel.spoken[int(entry["balloon"]) - 1]
            lines.append(compose.Line(balloon.speaker, float(entry["start"]), "English", balloon.text))
            if str(entry.get("voice") or "").strip():
                voices.setdefault(balloon.speaker, " ".join(str(entry["voice"]).split()))
        cut = "" if index == 0 or panel.landed else "new"
        chunks.append(
            compose.Chunk(
                purpose=str(clip["purpose"]),
                cast=tuple(dict.fromkeys([*_cast(given, index), place])),
                beats=beats,
                camera_move=str(clip["camera"]["move"]),
                camera_toward=str(clip["camera"]["toward"]),
                mouths=tuple(str(m) for m in clip.get("mouths", [])),
                closing=str(clip["closing"]).strip(),
                irreversible_change=str(clip.get("irreversible_change") or ""),
                dialogue=tuple(lines),
                seconds=panel.seconds,
                cut=cut,
                opening=str(clip.get("opening") or "").strip() if cut else "",
                keyframe=panel.landed,
                ambience=str(clip.get("ambience") or "").strip(),
                scene=" ".join(str(clip.get("scene") or "").split()),
            )
        )
    first = str(data["clips"][0].get("opening") or "").strip() if data["clips"] else ""
    return dataclasses.replace(
        base, opening=first, ambience=str(data["ambience"]).strip(), chunks=tuple(chunks), voices=voices
    )


def fill(base: compose.Take, given: list[Given], aliases: dict[str, str]) -> compose.Take:
    """`base.cards` holds every member; `aliases` are the reader's names for each, so the panels' words find it."""
    living = [m for m in base.members if base.cards[m].kind in compose.LIVING]
    places = [m for m in base.members if base.cards[m].kind == "environment"]
    cast = "\n".join(
        f"{c.id} | {c.kind} | {c.noun} | {c.full()}" + (f" | also called {aliases[c.id]}" if aliases.get(c.id) else "")
        for c in (base.cards[m] for m in [*living, *places])
    )
    clips = []
    for number, panel in enumerate(given, 1):
        if panel.lands:
            joins = "ENDS ON THE NEXT PANEL (lands)"
        else:
            joins = "the next clip cuts" if number < len(given) else "the last clip"
        spoken = "; ".join(f'{n}. {{{b.speaker}}} ({b.kind}): "{b.text}"' for n, b in enumerate(panel.spoken, 1))
        clips.append(words.MANGA_FILL_CLIP.format(
            number=number, seconds=round(panel.seconds, 2), joins=joins, action=panel.action,
            cast=", ".join(f"{{{m}}}" for m in panel.cast) or "nobody", balloons=spoken or "none",
        ))  # fmt: skip
    moves = "\n".join(f"- {name}: {s.format(toward='the member')}" for name, s in words.CAMERA_MOVES.items())
    system = words.MANGA_FILL_SYSTEM.format(
        beat_min=h3.DIRECTOR_BEAT_MIN, words_per_second=h3.DIRECTOR_WORDS_PER_SECOND,
        dialogue_tail=h3.DIRECTOR_DIALOGUE_TAIL, closing_words=h3.DIRECTOR_CLOSING_WORDS,
    )  # fmt: skip
    question = words.MANGA_FILL_USER.format(
        cast=cast, places=", ".join(places) or "-", clips="\n".join(clips), moves=moves
    )

    def gate(data: Any) -> list[caption.Problem]:
        if len(data["clips"]) != len(given):
            return [direct.problem(f'"clips" has exactly {len(given)} entries, one per clip given, in order')]
        out = []
        for number, (clip, panel) in enumerate(zip(data["clips"], given, strict=True), 1):
            out += _clip_problems(number, clip, panel, given, places)
        if out:
            return out
        take = take_from(data, base, given)
        return ruling.mouths(take) + ruling.leaving(take)

    data = direct.ask_json(system, question, [], gate, "fill", h3.DIRECTOR_TOKENS["plan"] + 4096,
                           repairs=h3.DIRECTOR_PLAN_REPAIRS)  # fmt: skip
    take = take_from(data, base, given)
    for index, chunk in enumerate(take.chunks):
        ui.panel(f"fill · clip {index + 1}", {
            "length": f"{chunk.seconds:.2f}s", "purpose": chunk.purpose,
            **{f"{b.start:g}–{b.end:g}s {b.phase}": b.action for b in chunk.beats},
            "camera": f"{chunk.camera_move} → {chunk.camera_toward}", "closing": chunk.closing,
        })  # fmt: skip
    return take


def _clip_problems(
    number: int, clip: Any, panel: Given, given: list[Given], places: list[str]
) -> list[caption.Problem]:
    out = []
    name = f"clip {number}"
    seconds = panel.seconds
    place = str(clip["place"])
    if place not in places:
        out.append(direct.problem(f'{name}: "place" is one of {", ".join(places)}'))
    allowed = {*_cast(given, number - 1), place}
    beats = list(clip["beats"])
    if not BEATS[0] <= len(beats) <= BEATS[1]:
        out.append(direct.problem(f'{name}: "beats" has {BEATS[0]} to {BEATS[1]} beats'))
    at = 0.0
    for n, beat in enumerate(beats, 1):
        start, end = float(beat["start"]), float(beat["end"])
        if abs(start - at) > 0.05:
            out.append(direct.problem(f"{name}: beat {n} starts at {start:g}; it starts where the one before ends, "
                                      f"{at:g}"))  # fmt: skip
        if end <= start:
            out.append(direct.problem(f"{name}: beat {n} ends before it starts"))
        if str(beat["phase"]) not in PHASES:
            out.append(direct.problem(f'{name}: beat {n}\'s "phase" is one of {", ".join(PHASES)}'))
        stray = sorted({str(w) for w in beat["who"]} - allowed)
        if stray:
            out.append(direct.problem(f'{name}: beat {n} moves {", ".join(stray)}, who is not in the clip: its '
                                      f'"who" is from {", ".join(sorted(allowed))}'))  # fmt: skip
        at = end
    if beats and abs(at - seconds) > 0.3:
        out.append(direct.problem(f"{name}: the beats end at {at:g}; the clip lasts {seconds:g} seconds, so the "
                                  f"last beat ends at {seconds:g}"))  # fmt: skip
    if str(clip["camera"]["move"]) not in words.CAMERA_MOVES:
        out.append(direct.problem(f'{name}: the camera "move" is one of {", ".join(words.CAMERA_MOVES)}'))
    if str(clip["camera"]["toward"]) not in allowed:
        out.append(direct.problem(f'{name}: the camera moves "toward" one of {", ".join(sorted(allowed))}'))
    opens = number == 1 or not panel.landed
    if opens and not str(clip.get("scene") or "").strip():
        out.append(direct.problem(f'{name}: "scene" is empty, and the clip opens a scene: give its heading'))
    if opens and not str(clip.get("opening") or "").strip():
        out.append(direct.problem(f'{name}: "opening" is empty: describe the clip\'s first frame, the panel'))
    out += _dialogue(name, clip, panel)
    said = " ".join([str(clip.get("opening") or ""), str(clip["closing"]), *(str(b["action"]) for b in beats)])
    named = sorted(set(compose.PLACEHOLDER.findall(said)) - allowed)
    if named:
        out.append(direct.problem(f"{name} names {', '.join(f'{{{m}}}' for m in named)}, who is not in it: name "
                                  f"only {', '.join(f'{{{m}}}' for m in sorted(allowed))}"))  # fmt: skip
    closing = str(clip["closing"])
    if not closing.strip():
        out.append(direct.problem(f'{name}: "closing" describes its last frame'))
    elif panel.lands and number < len(given):
        after = given[number]
        missing = [m for m in after.cast if f"{{{m}}}" not in closing]
        if missing:
            out.append(direct.problem(f"{name} ends on the next panel, so its closing names everyone in it: add "
                                      + ", ".join(f"{{{m}}}" for m in missing)))  # fmt: skip
        elif after.action and not set(lint.tokens(after.action)) & set(lint.tokens(closing)) - _COMMON:
            out.append(direct.problem(f"{name} ends on the next panel, and its closing says nothing of what that "
                                      f'panel shows: "{after.action}"'))  # fmt: skip
    return out


_COMMON = {"the", "a", "an", "of", "and", "in", "on", "to", "at", "is", "his", "her", "its", "their", "with"}


def _dialogue(name: str, clip: Any, panel: Given) -> list[caption.Problem]:
    out = []
    entries = list(clip.get("dialogue") or [])
    numbers = [int(e["balloon"]) for e in entries]
    expected = list(range(1, len(panel.spoken) + 1))
    if numbers != expected:
        said = ", ".join(map(str, expected)) or "none"
        out.append(direct.problem(f'{name}: "dialogue" has one entry per spoken balloon, in order: balloons {said}'))
        return out
    ends = 0.0
    for entry, balloon in zip(entries, panel.spoken, strict=True):
        start = float(entry["start"])
        if start < ends - 0.05:
            out.append(direct.problem(f"{name}: balloon {entry['balloon']} starts at {start:g}, before the line "
                                      f"before it ends at {ends:.2f}"))  # fmt: skip
        ends = max(start, ends) + _said(balloon)
    latest = panel.seconds - h3.DIRECTOR_DIALOGUE_TAIL
    if entries and ends > latest + 0.05 and sum(_said(b) for b in panel.spoken) <= latest:
        out.append(
            direct.problem(
                f"{name}: its last line ends at {ends:.2f}; lines end by {latest:.2f}, "
                f"{h3.DIRECTOR_DIALOGUE_TAIL:g} seconds before the clip does: start them earlier"
            )
        )
    return out
