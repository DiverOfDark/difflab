"""What one uploaded picture shows, before any story is read: the row's guess and the kind it starts as."""

from __future__ import annotations

from typing import Any

from difflab.core.types import Frame
from difflab.models.minimax_h3 import caption
from difflab.screenplay import config as h3
from difflab.screenplay import words
from difflab.tools import direct

KINDS = ("character", "location", "prop", "style")
_TEXT = {"type": "string", "maxLength": 160}
SCHEMA = {
    "type": "object",
    "properties": {"kind": {"enum": list(KINDS)}, "about": _TEXT, "look": _TEXT},
    "required": ["kind", "about", "look"],
    "additionalProperties": False,
}


def guess(frame: Frame) -> dict[str, str]:
    def gate(data: Any) -> list[caption.Problem]:
        return [] if str(data.get("kind")) in KINDS else [direct.problem(f'"kind" is one of {", ".join(KINDS)}')]

    image = direct.shrink(frame, h3.DIRECTOR_PHOTO_PIXELS)
    data = direct.ask_json(
        words.GUESS_SYSTEM, words.GUESS_USER, [image], gate, "guess", h3.RULING_TOKENS, schema=SCHEMA
    )
    return {name: " ".join(str(data.get(name) or "").split()) for name in ("kind", "about", "look")}
