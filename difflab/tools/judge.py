"""The VLM as a continuity supervisor: does a keyframe or a clip show the shot it was made for, with
the cast looking like their identities say?"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from difflab.core.settings import JudgeSettings
from difflab.core.types import Clip, Frame
from difflab.models.minimax_h3 import caption
from difflab.screenplay import checks, compiler, words
from difflab.screenplay import config as h3
from difflab.tools import direct

# A glitch shorter than this can fall between two frames. At the largest canvas yet (1280x736, 920
# vision tokens a frame) a 12.25 s clip is 26 frames, ~24k tokens: it needs the VLM's 64k context.
SAMPLED_EVERY = 0.5


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
    turned: dict[str, str] | None = None,
) -> Verdict:
    """`kind` is keyframe or clip; `arrangement` is what should be seen, members as {id}; `pictures` are
    members' own reference photos by id, shown before the frames so clothes are judged against them;
    `looks` is what earlier shots changed about a member for good, and `becomes` what this one changes.
    `allowed` is what the bible lets differ from the photo, `appearance` a member's look as a person wrote it.
    `turned` is what a member that has turned into something else was in its reference photo."""
    given = {slug: frame for slug, frame in (pictures or {}).items() if slug in {card.id for card in cast}}
    numbered = {slug: n for n, slug in enumerate(given, 1)}
    expected = {card.id for card in cast}

    def parse(data: Any) -> Verdict:
        likeness = {str(m["id"]): float(m["likeness"]) for m in data.get("cast", [])}
        strangers = [f"not in the shot: {s}" for s in map(str, data.get("strangers", [])) if s.strip()]
        restyle = [
            str(m["id"])
            for m in data.get("cast", [])
            if m.get("outfit") is False and str(m["id"]) in given and str(m["id"]) not in (turned or {})
        ]
        changed = [f"{member}: clothes differ from their photo" for member in restyle]
        score = float(data["score"])
        # Held in code as well as asked: the model rounds its own rule. Not on a clip, whose every retake
        # costs minutes of render; a keyframe's retry costs seconds (`made._judged`).
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
            + (words.JUDGE_FORM.format(was=turned[c.id]) if turned and c.id in turned and c.id in numbered else "")
            + (f" (reference photo: Picture {numbered[c.id]})" if c.id in numbered else "")
            for c in cast
        )
        or "- nobody in particular",
        count=len(frames),
        fixes=words.JUDGE_FIXES_PICTURE if kind in PICTURES else words.JUDGE_FIXES_SCENE,
        references=f"First {len(given)} reference photo(s), then the shot. " if given else "",
    )
    # The photos lead so that every judging of the same cast opens the same way, which the VLM keeps on disk.
    images = [direct.shrink(frame, h3.DIRECTOR_PHOTO_PIXELS) for frame in [*given.values(), *frames]]
    data = direct.ask_json(
        words.JUDGE_SYSTEM, question, images, gate, f"judge · {kind}", settings.max_new_tokens, steady=len(given)
    )
    return parse(data)


def revise(frames: list[Frame], prompt: str, verdict: Verdict) -> str:
    """The clip's prompt with its summary and description rewritten to the verdict, for its retake."""
    summary = caption.body(prompt, "summary")
    definitions = caption.body(prompt, "subject_definitions")
    presented = list(dict.fromkeys(re.findall(r"<(?:Picture|Video|Audio)\s+\d+>", definitions)))
    directive = found.group(0) if (found := re.match(r"\[[^\]]+\]", summary)) else ""

    def written(data: Any) -> str:
        return _replaced(
            _replaced(prompt, "summary", str(data["summary"])), "detailed_description", str(data["description"])
        )

    def broken(text: str) -> dict[str, str]:
        problems = caption.ide_rules(text) + checks.chunk_rules(text, presented, directive)
        # Keyed without digits, so a description already too long is not a new fault for being longer still.
        return {re.sub(r"\d+", "#", p.message): p.message for p in caption.errors(problems)}

    already, labelled = broken(prompt), set(caption.LABEL.findall(prompt))

    def gate(data: Any) -> list[caption.Problem]:
        text = written(data)
        out = [direct.problem(message) for bare, message in sorted(broken(text).items()) if bare not in already]
        out += [direct.problem(f"{label} is not in the prompt; keep its labels") for label in
                sorted(set(caption.LABEL.findall(text)) - labelled)]  # fmt: skip
        return out

    question = words.CLIP_REVISE_USER.format(
        count=len(frames),
        issues="\n".join(f"- {issue}" for issue in verdict.issues) or "- none named",
        fixes="\n".join(f"- {fix}" for fix in verdict.fixes) or "- none named",
        summary=summary,
        description=caption.body(prompt, "detailed_description"),
    )
    images = [direct.shrink(frame, h3.DIRECTOR_PHOTO_PIXELS) for frame in frames]
    data = direct.ask_json(
        words.CLIP_REVISE_SYSTEM, question, images, gate, "judge · revise the clip", h3.DIRECTOR_WRITE_TOKENS,
        settle=False,
    )  # fmt: skip
    return written(data)


def _replaced(text: str, name: str, new: str) -> str:
    for section, _, content, end in caption.sections(text):
        if section == name:
            return text[:content] + new.strip() + "\n\n" + text[end:]
    raise ValueError(f"the caption has no {name}: section")


def sampled(clip: Clip, every: float = SAMPLED_EVERY) -> list[Frame]:
    """A frame every `every` seconds from the first, and the last."""
    step = max(1, round(every * clip.fps))
    return [clip.frame(i) for i in dict.fromkeys([*range(0, len(clip), step), len(clip) - 1])]
