"""The VLM as a continuity supervisor: does a keyframe or a clip show the shot it was made for, with
the cast looking like their identities say?"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    fixes: list[str] = field(default_factory=list)  # what a redo is told: never looks, which come from pictures
    restyle: list[str] = field(default_factory=list)  # members whose clothes differ from their photo

    def data(self) -> dict[str, Any]:
        return {
            "score": self.score, "cast": self.cast, "issues": self.issues, "notes": self.notes,
            "fixes": self.fixes, "restyle": self.restyle,
        }  # fmt: skip


# Judged alone against their own words, and redrawn from the verdict's fixes (`made._picture_judged`).
PICTURES = ("look picture", "reference picture")


def judge(
    frames: list[Frame],
    kind: str,
    purpose: str,
    arrangement: str,
    cast: list[compiler.Card],
    settings: JudgeSettings,
    pictures: dict[str, Frame] | None = None,
    looks: dict[str, str] | None = None,
    becomes: dict[str, str] | None = None,
    allowed: dict[str, str] | None = None,
    appearance: dict[str, str] | None = None,
) -> Verdict:
    """`kind` is keyframe or clip; `arrangement` is what should be seen, members as {id}; `pictures` are
    members' own reference photos by id, shown after the frames so clothes are judged against them;
    `looks` is what earlier shots changed about a member for good, and `becomes` what this one changes.
    `allowed` is what the bible lets differ from the photo, `appearance` a member's look as a person wrote it."""
    given = {slug: frame for slug, frame in (pictures or {}).items() if slug in {card.id for card in cast}}
    numbered = {slug: len(frames) + n for n, slug in enumerate(given, 1)}
    expected = {card.id for card in cast}

    def parse(data: Any) -> Verdict:
        likeness = {str(m["id"]): float(m["likeness"]) for m in data.get("cast", [])}
        strangers = [f"not in the shot: {s}" for s in map(str, data.get("strangers", [])) if s.strip()]
        restyle = [str(m["id"]) for m in data.get("cast", []) if m.get("outfit") is False and str(m["id"]) in given]
        changed = [f"{member}: clothes differ from their photo" for member in restyle]
        score = float(data["score"])
        # Held in code as well as asked: the model rounds its own rule. Not on a clip, whose retry is a blind
        # re-seed of minutes; a keyframe's retry is told what to fix (`made._judged`).
        if (strangers or changed) and kind != "clip":
            score = min(score, 0.4)
        return Verdict(
            score,
            {member: likeness.get(member, 0.0) for member in expected},
            [str(issue) for issue in data.get("issues", [])] + strangers + changed,
            str(data.get("notes", "")),
            [str(fix) for fix in data.get("fixes", [])] + [f"remove {s}" for s in data.get("strangers", []) if s],
            restyle,
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
        cast="\n".join(
            f"- {c.id}: the {c.noun}, {(appearance or {}).get(c.id) or c.full()}"
            + (f"; changed since its photo: {looks[c.id]}" if looks and looks.get(c.id) else "")
            + (f"; may differ from its photo in: {allowed[c.id]}" if allowed and allowed.get(c.id) else "")
            + (f"; during this shot it changes to: {becomes[c.id]}" if becomes and becomes.get(c.id) else "")
            + (f" (reference photo: Picture {numbered[c.id]})" if c.id in numbered else "")
            for c in cast
        )
        or "- nobody in particular",
        count=len(frames),
        fixes=words.JUDGE_FIXES_PICTURE if kind in PICTURES else words.JUDGE_FIXES_SCENE,
        references=f", then {len(given)} reference photo(s)" if given else "",
    )
    images = [direct.shrink(frame, h3.DIRECTOR_PHOTO_PIXELS) for frame in [*frames, *given.values()]]
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
