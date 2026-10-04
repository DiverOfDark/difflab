"""A shot's first frame composed with the editor: the room from a reference, and the cast placed by the
plan's words, each person and animal handed over as its own picture where it has one. A member with no
picture is carried by its words. A place with no photograph is drawn around a photo of a member
instead, which the editor keeps."""

from __future__ import annotations

import re
from collections.abc import Sequence

from difflab.core.settings import EditSettings
from difflab.core.types import Frame
from difflab.screenplay import compiler, words
from difflab.tools import edit_image

PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def instruction(
    arrangement: str,
    cards: dict[str, compiler.Card],
    keep: str | None = None,
    pictured: Sequence[str] = (),
    note: str = "",
) -> str:
    """The plan's opening sentence with each {id} said as the member it names, in the editor's words: a
    member in `pictured` by its picture (Picture 2 onward, in that order), the others by their traits."""
    said: set[str] = set()

    def member(match: re.Match[str]) -> str:
        card = cards.get(match.group(1))
        if card is None:
            return match.group(1).replace("_", " ")
        if card.id in said:
            return f"the {card.noun}"
        said.add(card.id)
        if card.id in pictured:
            return f"the {card.noun} from Picture {list(pictured).index(card.id) + 2}"
        return f"the {card.noun} with {card.short()}"

    scene = PLACEHOLDER.sub(member, arrangement).strip().rstrip(".")
    if keep is not None and keep in cards:
        out = words.COMPOSE_KEEP_INSTRUCTION.format(noun=cards[keep].noun, scene=scene)
    else:
        out = words.COMPOSE_INSTRUCTION.format(scene=scene)
    parts = [out, *([words.COMPOSE_LIKENESS] if pictured else []), words.COMPOSE_ALONE]
    told = " ".join(note.replace("Fix:", " ").split()).rstrip(".")  # `made.JUDGED` marks the judge's half
    if told:
        parts.append(words.COMPOSE_NOTE.format(note=told))
    return " ".join(parts)


def around(frame: Frame, box: list[float], ratio: float) -> Frame:
    """The largest cut of `ratio` (width / height) from `frame`, centred on `box` as far as its edges allow."""
    width, height = frame.width, frame.height
    wide, high = (width, round(width / ratio)) if width / height < ratio else (round(height * ratio), height)
    wide, high = min(wide, width), min(high, height)
    left = round(min(max((box[0] + box[2]) / 2 - wide / 2, 0), width - wide))
    top = round(min(max((box[1] + box[3]) / 2 - high / 2, 0), height - high))
    return frame.crop((left, top, left + wide, top + high))


def compose(
    base: Frame,
    arrangement: str,
    cards: dict[str, compiler.Card],
    settings: EditSettings,
    keep: str | None = None,
    size: tuple[int, int] | None = None,
    pictures: dict[str, Frame] | None = None,
    note: str = "",
) -> Frame:
    """`pictures` is each member handed over as its own picture, by id, in the order they are named."""
    given = pictures or {}
    said = instruction(arrangement, cards, keep, list(given), note)
    return edit_image.edit(base, said, settings, size=size, donors=list(given.values()))
