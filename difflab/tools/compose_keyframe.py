"""A shot's first frame composed with the editor: the room from a reference, the cast placed by the
plan's words. A member with no photograph has no picture to take a face from; the words carry it. A
place with no photograph is drawn around a photo of a member instead, which the editor keeps."""

from __future__ import annotations

import re

from difflab.core.settings import EditSettings
from difflab.core.types import Frame
from difflab.screenplay import compiler, words
from difflab.tools import edit_image

PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def instruction(arrangement: str, cards: dict[str, compiler.Card], keep: str | None = None) -> str:
    """The plan's opening sentence with each {id} said as the member it names, in the editor's words."""
    said: set[str] = set()

    def member(match: re.Match[str]) -> str:
        card = cards.get(match.group(1))
        if card is None:
            return match.group(1).replace("_", " ")
        if card.id in said:
            return f"the {card.noun}"
        said.add(card.id)
        return f"the {card.noun} with {card.short()}"

    scene = PLACEHOLDER.sub(member, arrangement).strip().rstrip(".")
    if keep is not None and keep in cards:
        return words.COMPOSE_KEEP_INSTRUCTION.format(noun=cards[keep].noun, scene=scene)
    return words.COMPOSE_INSTRUCTION.format(scene=scene)


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
) -> Frame:
    return edit_image.edit(base, instruction(arrangement, cards, keep), settings, size=size)
