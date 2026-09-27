"""The VLM as a continuity supervisor: does a keyframe or a clip show the shot it was made for, with
the cast looking like their identities say?"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from difflab.core import media
from difflab.core.settings import JudgeSettings
from difflab.core.types import Clip, Frame
from difflab.models.minimax_h3 import caption
from difflab.screenplay import compiler, words
from difflab.screenplay import config as h3
from difflab.tools import direct

SAMPLED_FRAMES = 4  # of a clip, evenly spaced, as one sheet: enough to see a gesture happen, cheap to look at


@dataclass(frozen=True)
class Verdict:
    score: float  # 0..1: how well it shows the shot and its cast
    cast: dict[str, float]  # each member's likeness, 0..1, by slug; 0 when absent
    issues: list[str]
    notes: str

    def data(self) -> dict[str, Any]:
        return {"score": self.score, "cast": self.cast, "issues": self.issues, "notes": self.notes}


def judge(
    frames: list[Frame], kind: str, purpose: str, arrangement: str, cast: list[compiler.Card], settings: JudgeSettings
) -> Verdict:
    """`kind` is keyframe or clip; `arrangement` is what should be seen, members as {id}."""
    expected = {card.id for card in cast}

    def parse(data: Any) -> Verdict:
        likeness = {str(m["id"]): float(m["likeness"]) for m in data.get("cast", [])}
        return Verdict(
            float(data["score"]),
            {member: likeness.get(member, 0.0) for member in expected},
            [str(issue) for issue in data.get("issues", [])],
            str(data.get("notes", "")),
        )

    def gate(data: Any) -> list[caption.Problem]:
        verdict = parse(data)
        out = [] if 0.0 <= verdict.score <= 1.0 else [direct.problem('"score" is a number from 0 to 1')]
        out += [
            direct.problem(f'{m}: "likeness" is a number from 0 to 1')
            for m, v in verdict.cast.items()
            if not 0 <= v <= 1
        ]
        missing = expected - {str(m.get("id")) for m in data.get("cast", [])}
        out += [direct.problem(f'"cast" has an entry for {m}') for m in sorted(missing)]
        return out

    question = words.JUDGE_USER.format(
        kind=kind,
        purpose=purpose.strip() or "-",
        arrangement=arrangement.strip() or "-",
        cast="\n".join(f"- {c.id}: the {c.noun}, {c.full()}" for c in cast) or "- nobody in particular",
        count=len(frames),
    )
    images = [direct.shrink(frame, h3.DIRECTOR_PHOTO_PIXELS) for frame in frames]
    data = direct.ask_json(words.JUDGE_SYSTEM, question, images, gate, f"judge · {kind}", settings.max_new_tokens)
    return parse(data)


def sampled(clip: Clip, count: int = SAMPLED_FRAMES) -> list[Frame]:
    """`count` frames spread over the clip, first and last included."""
    if len(clip) <= count:
        return [clip.frame(i) for i in range(len(clip))]
    step = (len(clip) - 1) / (count - 1)
    return [clip.frame(round(i * step)) for i in range(count)]


def sheet(frames: list[Frame]) -> Frame:
    return media.sheet([[frame] for frame in frames], height=540)
