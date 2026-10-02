"""A rough idea rewritten into a story the planner renders well, or a rewrite revised by feedback."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from difflab.core import ui
from difflab.core.types import Frame
from difflab.models.minimax_h3 import caption
from difflab.screenplay import config as h3
from difflab.screenplay import words
from difflab.tools import direct

NUMBERED = re.compile(r"^\s*(?:beat\s*)?\d+\s*[.):-]\s*", re.I)
MARKDOWN = re.compile(r"\*\*|__|^#+\s|^\s*[-*]\s", re.M)


@dataclass(frozen=True)
class Draft:
    place: str
    beats: tuple[str, ...]
    sound: str
    changes: tuple[str, ...]
    questions: tuple[str, ...]

    def story(self) -> str:
        numbered = "\n".join(f"{i}. {beat}" for i, beat in enumerate(self.beats, 1))
        return f"{self.place}\n\n{numbered}\n\nSound: {self.sound}"


def _draft(data: Any) -> Draft:
    return Draft(
        place=" ".join(str(data["place"]).split()),
        beats=tuple(NUMBERED.sub("", " ".join(str(b).split())) for b in data["beats"]),
        sound=re.sub(r"^\s*sound\s*:\s*", "", " ".join(str(data["sound"]).split()), flags=re.I),
        changes=tuple(" ".join(str(c).split()) for c in data.get("changes", []) if str(c).strip()),
        questions=tuple(" ".join(str(q).split()) for q in data.get("questions", []) if str(q).strip()),
    )


def refine(
    idea: str, frames: list[Frame], total: float, longest: float, previous: str = "", feedback: str = ""
) -> Draft:
    """The first rewrite of `idea`, or `previous` revised by `feedback`. `direct.TAKE` is the round.

    `total` is the most the film may last; a beat is one shot of DIRECTOR_CLIP_MIN to `longest` seconds."""
    beat_words = h3.IDEA_BEAT_WORDS[0]
    pictures = f"{len(frames)} photograph{'s' if len(frames) != 1 else ''}"
    shortest = h3.DIRECTOR_CLIP_MIN
    fewest, fullest = 1, max(1, math.floor(total / shortest))
    shape = {
        "total": round(total), "shortest": shortest, "longest": round(longest, 1), "least": fewest, "most": fullest,
        "beat_words": beat_words, "pictures": pictures,
    }  # fmt: skip
    if previous:
        question = words.IDEA_REVISE_USER.format(
            idea=idea.strip(), previous=previous.strip(), feedback=feedback.strip() or "Another pass.", **shape
        )
    else:
        question = words.IDEA_USER.format(idea=idea.strip(), **shape)

    def gate(data: Any) -> list[caption.Problem]:
        draft = _draft(data)
        out = []
        if not fewest <= len(draft.beats) <= fullest:
            out.append(direct.problem(f'"beats" has {len(draft.beats)} entries; {fewest} to {fullest} fit the LENGTH, '
                                      "one per main event"))  # fmt: skip
        for number, beat in enumerate(draft.beats, 1):
            length = len(beat.split())
            if not length:
                out.append(direct.problem(f"beat {number} is empty"))
            elif length > h3.IDEA_BEAT_WORDS[1]:
                out.append(direct.problem(f"beat {number} runs {length} words: at most {beat_words}"))
        least, most = h3.IDEA_PLACE_WORDS
        place = len(draft.place.split())
        if not least <= place <= most:
            out.append(direct.problem(f'"place" runs {place} words: the opening paragraph is 35 to 80'))
        if not draft.sound:
            out.append(direct.problem('"sound" is the sound line: the place, then each beat\'s key moment'))
        if not draft.changes:
            out.append(direct.problem('"changes" lists what you changed from the IDEA, and why'))
        if any(MARKDOWN.search(text) for text in (draft.place, *draft.beats, draft.sound)):
            out.append(direct.problem("plain prose: no markdown, no bold, no bullets, no headings"))
        return out

    images = [direct.shrink(frame, h3.DIRECTOR_SURVEY_PIXELS) for frame in frames]
    data = direct.ask_json(words.IDEA_SYSTEM, question, images, gate, "idea", h3.IDEA_TOKENS, h3.IDEA_TEMPERATURE)
    draft = _draft(data)
    ui.panel("idea · the story", {"place": draft.place, **{f"beat {i}": b for i, b in enumerate(draft.beats, 1)}})
    return draft
