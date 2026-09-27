"""The whole chapter in one call, so the model sees every join while it decides where beats end."""

from __future__ import annotations

import dataclasses
from typing import Any

from difflab.core import ui
from difflab.models.minimax_h3 import caption
from difflab.screenplay import checks as lint
from difflab.screenplay import compiler as compose
from difflab.screenplay import config as h3
from difflab.screenplay import words
from difflab.tools import direct


def _take_from_plan(data: Any, base: compose.Take) -> compose.Take:
    """The plan's JSON as a take. The environment is in every chunk, whatever the plan says."""
    environment = next(m for m in base.members if base.cards[m].kind == "environment")
    chunks = []
    for clip in data["clips"]:
        beats = tuple(
            compose.Beat(
                float(b["start"]), float(b["end"]), str(b["phase"]), tuple(str(w) for w in b["who"]), str(b["action"])
            )
            for b in clip["beats"]
        )
        lines = tuple(
            compose.Line(str(d["who"]), float(d["start"]), str(d["language"]), str(d["line"]), str(d["voice"]))
            for d in clip.get("dialogue", [])
        )
        chunks.append(
            compose.Chunk(
                purpose=str(clip["purpose"]),
                cast=tuple(dict.fromkeys([*(str(m) for m in clip["cast"]), environment])),
                beats=beats,
                camera_move=str(clip["camera"]["move"]),
                camera_toward=str(clip["camera"]["toward"]),
                mouths=tuple(str(m) for m in clip.get("mouths", [])),
                closing=str(clip["closing"]).strip(),
                faces_at_close=tuple(str(m) for m in clip["faces_at_close"]),
                irreversible_change=str(clip["irreversible_change"]),
                dialogue=lines,
            )
        )
    return dataclasses.replace(base, ambience=str(data["ambience"]).strip(), chunks=tuple(chunks))


def plan(base: compose.Take, idea: str, count: int) -> compose.Take:
    """The whole take in one call, so the model sees every join while it decides where beats end.

    Members are then numbered by when they first appear, so a <Subject N> keeps its number from the
    chunk it enters to the end, and the sentences that name it stay the same words.
    """
    cast = "\n".join(f"{c.id} | {c.kind} | {c.noun} | {c.full()}" for c in (base.cards[m] for m in base.members))
    moves = "\n".join(
        f"- {name}: {sentence.format(toward='the member')}" for name, sentence in words.CAMERA_MOVES.items()
    )
    window = h3.DIRECTOR_MAIN_BEAT_WINDOW
    early, late = (fraction * base.seconds for fraction in window)
    question = words.PLAN_USER.format(
        idea=idea.strip(), cast=cast, opening=base.opening, count=count, seconds=base.seconds,
        total=count * base.seconds, early=early, late=late, moves=moves, beat_min=h3.DIRECTOR_BEAT_MIN,
    )  # fmt: skip
    system = words.PLAN_SYSTEM.format(
        airlock=h3.DIRECTOR_AIRLOCK[0], airlock_min=h3.DIRECTOR_AIRLOCK[1], settle=h3.DIRECTOR_SETTLE[0],
        settle_min=h3.DIRECTOR_SETTLE[1], early=window[0] * 100, late=window[1] * 100,
        handoff_min=h3.DIRECTOR_HANDOFF_MIN, max_living=h3.DIRECTOR_MAX_LIVING,
        dialogue_tail=h3.DIRECTOR_DIALOGUE_TAIL, words_per_second=h3.DIRECTOR_WORDS_PER_SECOND,
        closing_words=h3.DIRECTOR_CLOSING_WORDS,
    )  # fmt: skip

    def story(data: Any) -> tuple[tuple[str, int], ...]:
        return tuple((str(e["event"]), int(e["clip"])) for e in data.get("story", []))

    def gate(data: Any) -> list[caption.Problem]:
        if len(data["clips"]) != count:
            return [direct.problem(f'"clips" has exactly {count} entries')]
        take = _take_from_plan(data, base)
        out = lint.plan_rules(take)
        if not story(data):
            out.append(direct.problem('"story" lists every action of the IDEA with the clip that shows it'))
        return out + lint.story_rules(idea, list(story(data)), take)

    data = direct.ask_json(
        system, question, [], gate, "plan", h3.DIRECTOR_TOKENS["plan"], repairs=h3.DIRECTOR_PLAN_REPAIRS
    )
    take = dataclasses.replace(_take_from_plan(data, base), story=story(data))
    ui.panel("plan · the idea, clip by clip", {event: f"clip {clip}" for event, clip in take.story})
    first = {m: next((i for i, c in enumerate(take.chunks) if m in c.cast), len(take.chunks)) for m in take.members}
    order = tuple(sorted(take.members, key=lambda m: (first[m], take.members.index(m))))
    for index, chunk in enumerate(take.chunks):
        ui.panel(
            f"plan · chunk {index + 1}",
            {
                "purpose": chunk.purpose,
                **{f"{b.start:g}–{b.end:g}s {b.phase}": b.action for b in chunk.beats},
                "camera": f"{chunk.camera_move} → {chunk.camera_toward}",
                "closing": chunk.closing,
            },
        )
    return dataclasses.replace(take, members=order)
