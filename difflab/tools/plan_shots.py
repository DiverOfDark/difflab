"""The whole chapter in one call, so the model sees every join while it decides where beats end."""

from __future__ import annotations

import dataclasses
from typing import Any

from difflab.core import ui
from difflab.models.minimax_h3 import caption
from difflab.models.minimax_h3 import config as model
from difflab.screenplay import checks as lint
from difflab.screenplay import compiler as compose
from difflab.screenplay import config as h3
from difflab.screenplay import words
from difflab.tools import direct, ruling


def on_grid(seconds: float, longest: float) -> float:
    """The nearest length H3 renders exactly (17n+5 frames), at most `longest`, itself on the grid."""
    frames = max(model.FRAME_PHASE, round((round(seconds * model.FPS) - model.FRAME_PHASE) / model.FRAME_CYCLE)
                 * model.FRAME_CYCLE + model.FRAME_PHASE)  # fmt: skip
    return min(frames / model.FPS, longest)


def _take_from_plan(data: Any, base: compose.Take) -> compose.Take:
    """The plan's JSON as a take. A clip that names no place happens in the first one.

    Each clip's length goes onto the frame grid here and its last beat takes up the difference (under
    0.36 s), so the model never does frame arithmetic."""
    places = [m for m in base.members if base.cards[m].kind == "environment"]
    chunks = []
    for clip in data["clips"]:
        asked = float(clip["seconds"])
        seconds = on_grid(asked, base.seconds)
        beats = tuple(
            compose.Beat(
                float(b["start"]), float(b["end"]), str(b["phase"]), tuple(str(w) for w in b["who"]), str(b["action"])
            )
            for b in clip["beats"]
        )
        if beats and abs(beats[-1].end - asked) <= 0.3:
            beats = (*beats[:-1], dataclasses.replace(beats[-1], end=round(seconds, 3)))
        named = [str(m) for m in clip["cast"]]
        lines = tuple(
            compose.Line(str(d["who"]), float(d["start"]), str(d["language"]), str(d["line"]), str(d["voice"]))
            for d in clip.get("dialogue", [])
        )
        chunks.append(
            compose.Chunk(
                purpose=str(clip["purpose"]),
                cast=tuple(dict.fromkeys([*named, *([] if set(named) & set(places) else places[:1])])),
                beats=beats,
                camera_move=str(clip["camera"]["move"]),
                camera_toward=str(clip["camera"]["toward"]),
                mouths=tuple(str(m) for m in clip.get("mouths", [])),
                closing=str(clip["closing"]).strip(),
                faces_at_close=tuple(str(m) for m in clip["faces_at_close"]),
                irreversible_change=str(clip["irreversible_change"]),
                dialogue=lines,
                seconds=seconds,
                cut=str(clip.get("cut") or "").replace("none", ""),
                opening=str(clip.get("opening") or "").strip(),
                looks={str(m): str(v).strip() for m, v in (clip.get("looks") or {}).items()},
                ambience=str(clip.get("ambience") or "").strip(),
            )
        )
    return dataclasses.replace(base, ambience=str(data["ambience"]).strip(), chunks=tuple(chunks))


def plan(base: compose.Take, idea: str, total: float) -> compose.Take:
    """The whole take in one call, so the model sees every join while it decides where beats end.

    `total` is the most the take may last; the model chooses how many clips and how long each, between
    DIRECTOR_CLIP_MIN and `base.seconds`.

    Members are then numbered by when they first appear, so a <Subject N> keeps its number from the
    chunk it enters to the end, and the sentences that name it stay the same words.
    """
    cast = "\n".join(f"{c.id} | {c.kind} | {c.noun} | {c.full()}" for c in (base.cards[m] for m in base.members))
    moves = "\n".join(
        f"- {name}: {sentence.format(toward='the member')}" for name, sentence in words.CAMERA_MOVES.items()
    )
    window = h3.DIRECTOR_MAIN_BEAT_WINDOW
    shortest, longest = h3.DIRECTOR_CLIP_MIN, round(base.seconds, 1)
    question = words.PLAN_USER.format(
        idea=idea.strip(), cast=cast, opening=base.opening, total=round(total), shortest=shortest, longest=longest,
        early=window[0] * 100, late=window[1] * 100, moves=moves, beat_min=h3.DIRECTOR_BEAT_MIN,
    )  # fmt: skip
    system = words.PLAN_SYSTEM.format(
        longest=longest, look_words=h3.DIRECTOR_LOOK_WORDS,
        airlock=h3.DIRECTOR_AIRLOCK[0], airlock_min=h3.DIRECTOR_AIRLOCK[1], settle=h3.DIRECTOR_SETTLE[0],
        settle_min=h3.DIRECTOR_SETTLE[1], quiet_max=h3.DIRECTOR_QUIET_MAX, early=window[0] * 100, late=window[1] * 100,
        handoff_min=h3.DIRECTOR_HANDOFF_MIN, max_living=h3.DIRECTOR_MAX_LIVING,
        dialogue_tail=h3.DIRECTOR_DIALOGUE_TAIL, words_per_second=h3.DIRECTOR_WORDS_PER_SECOND,
        closing_words=h3.DIRECTOR_CLOSING_WORDS,
    )  # fmt: skip

    def story(data: Any) -> tuple[tuple[str, int], ...]:
        return tuple((str(e["event"]), int(e["clip"])) for e in data.get("story", []))

    def gate(data: Any) -> list[caption.Problem]:
        if not data["clips"]:
            return [direct.problem('"clips" has one entry per main event of the IDEA')]
        out = []
        for number, clip in enumerate(data["clips"], 1):
            if not shortest - 0.05 <= float(clip["seconds"]) <= longest + 0.05:
                out.append(direct.problem(f'clip {number}: "seconds" is {clip["seconds"]}; make it {shortest:g} to '
                                          f"{longest:g}, or merge it into a neighbour or split it"))  # fmt: skip
        planned = sum(float(clip["seconds"]) for clip in data["clips"])
        # The count as well as the rule: told only the sum, production 13's 9-clip plan (2026-10-05) kept every
        # move at 7.3 s through three repairs and came back 68-73 s for a 60 s take, though 6 s clips fit.
        if planned > (1 + h3.DIRECTOR_TAKE_SLACK) * total:
            count = len(data["clips"])
            out.append(direct.problem(f'the clips\' "seconds" add up to {planned:g}; the take lasts at most {total:g}, '
                                      f"so cut {planned - total:g} seconds: {count} clips average at most "
                                      f"{total / count:.1f} s, so shorten each clip's main beat and its settle "
                                      f"toward the least they may be, or merge two"))  # fmt: skip
        # Never returned alone: an answer stopped here counted one problem and was kept over answers that
        # broke fewer rules, unchecked - a 72 s plan whose closings named members missing from the next cast.
        take = _take_from_plan(data, base)
        out += lint.plan_rules(take) + ruling.mouths(take) + ruling.leaving(take)
        if not story(data):
            out.append(direct.problem('"story" lists every action of the IDEA with the clip that shows it'))
        return out + ruling.story(idea, list(story(data)), take)

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
                "length": f"{chunk.seconds:.2f}s",
                "purpose": chunk.purpose,
                **{f"{b.start:g}–{b.end:g}s {b.phase}": b.action for b in chunk.beats},
                "camera": f"{chunk.camera_move} → {chunk.camera_toward}",
                "closing": chunk.closing,
            },
        )
    return dataclasses.replace(take, members=order)
