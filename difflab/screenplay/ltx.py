"""A chapter's plan in, LTX-2.5 prompts out: the plan `compiler` makes H3 captions from, said the LTX way."""

# An "Image N:" line per MSR reference, as LiconStudio's examples write them, then one paragraph after
# Lightricks' own captions: the first frame, each beat in order, the camera, the lines, the ending, the sound.

from __future__ import annotations

import dataclasses
import re

from difflab.models.ltx25 import config as ltx
from difflab.screenplay import compiler, words
from difflab.screenplay.compiler import Card, Compiled, Presentation, Take

# Who is handed a slot first when there are more members than MSR has slots: who acts, what they hold, the place.
SLOT_ORDER = {"person": 0, "animal": 1, "object": 2, "environment": 3}
# MSR reads each Image as a subject of its own, so a member gets one picture: a look's own, then the whole of
# it, as LiconStudio's reference sheets show their people, before a face.
ROLE_ORDER = {"body": 1, "view": 1, "object": 1, "environment": 1, "face": 2}


def pictures(take: Take, index: int) -> tuple[str, ...]:
    """One of the crops `compiler.references` would hold per member, at most five; on 2.3 at most four, and no
    place, since the opening frame is its scene."""
    held = compiler.references(take, index)
    cast = take.chunks[index].cast
    if take.ltx == ltx.V23:
        cast = tuple(m for m in cast if take.cards[m].kind != "environment")
    order = sorted(cast, key=lambda m: (SLOT_ORDER[take.cards[m].kind], take.members.index(m)))
    chosen: list[str] = []
    for member in order:
        own = [key for key in held if take.refs[key].member == member]
        own.sort(key=lambda key: (not take.refs[key].look, ROLE_ORDER[take.refs[key].role], -take.refs[key].quality))
        if own:
            chosen.append(own[0])
    return tuple(chosen[: ltx.MSR23_MAX_SUBJECTS if take.ltx == ltx.V23 else ltx.MSR_MAX_REFERENCES])


def _plain(text: str, bible: dict[str, Card]) -> str:
    """A look's words name the member they were drawn from as {id}: said here as the thing it was."""
    return compiler.PLACEHOLDER.sub(lambda m: f"the {bible[m.group(1)].noun}" if m.group(1) in bible else "", text)


def _described(card: Card, bible: dict[str, Card]) -> str:
    traits = _plain(", ".join(trait.phrase for trait in card.ordered()), bible)
    head = f"Scene, {card.noun}" if card.kind == "environment" else card.noun[:1].upper() + card.noun[1:]
    lights = f", lit by {compiler.join(list(card.lights))}" if card.kind == "environment" and card.lights else ""
    return f"{head}, {traits}{lights}, photorealistic natural texture."


def compile_chunk(take: Take, index: int) -> Compiled:
    bible = take.cards
    take = dataclasses.replace(take, cards=compiler.cards_at(take, index))
    chunk = take.chunks[index]
    keys = pictures(take, index)
    word = "Figure" if take.ltx == ltx.V23 else "Image"
    image = {take.refs[key].member: f"{word} {n + 1}" for n, key in reversed(list(enumerate(keys)))}
    lines = [f"{word} {n + 1}: {_described(take.cards[take.refs[key].member], bible)}" for n, key in enumerate(keys)]
    seen: set[str] = set()

    def named(text: str, introduce: bool = True) -> str:
        def label(match: re.Match[str]) -> str:
            member = match.group(1)
            card = take.cards.get(member)
            if card is None:
                return match.group(0)
            said = image.get(member) or f"the {card.noun}"
            if not introduce or member in seen or card.kind == "environment":
                return said
            seen.add(member)
            short = _plain(card.short(), bible)
            if member in image:
                return f"{said}, the {card.noun} with {short},"
            return f"the {card.noun} with {short}"

        return compiler.PLACEHOLDER.sub(label, text)

    pending = list(chunk.dialogue)

    def spoken(text: str) -> str:
        while "{say}" in text and pending:
            line = pending.pop(0)
            head, rest = text.split("{say}", 1)
            text = f'{head}says in {compiler.voice(take, line.who)}, "{line.line}"{rest}'
        return text.replace("{say}", "")

    def sentence(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip().rstrip(".").strip()
        text = re.sub(r",\s*([,.;:])", r"\1", text)
        return f"{text[:1].upper()}{text[1:]}." if text else ""

    # The arrangement is a list of what is where, not a sentence: production 1's begins "{alaska} a small dark
    # shape high in the sky".
    body = [sentence(f"At first: {named(compiler.opening(take, index))}")]
    if take.look:
        body.append(sentence(words.STYLE_LOOK.format(look=take.look.rstrip("."))))
    camera = words.CAMERA_MOVES[chunk.camera_move].format(toward=named(f"{{{chunk.camera_toward}}}", False))
    camera_said = False
    for beat in compiler.written(chunk):
        body.append(sentence(spoken(named(beat.prose.strip() or beat.action.strip()))))
        if not camera_said and beat.phase in ("primary", "handoff"):
            body.append(camera)
            camera_said = True
    if not camera_said:
        body.append(camera)
    for line in pending:
        who = image.get(line.who) or f"the {take.cards[line.who].noun}"
        body.append(sentence(f'{who} says in {compiler.voice(take, line.who)}, "{line.line}"'))
    ending = "The shot ends exactly on this frame" if compiler.lands(take, index) else "The shot ends as"
    body.append(sentence(f"{ending}: {named(chunk.closing, False)}"))
    body.append(sentence(f"Heard throughout: {compiler.ambience(take, index)}"))
    body.append(sentence(named(chunk.sounds, False)))
    paragraph = " ".join(part for part in body if part)
    prompt = "\n".join(lines) + ("\n\n" if lines else "") + paragraph
    return Compiled(prompt, Presentation(keys, False), keys, {})
