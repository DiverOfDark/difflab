"""The whole chapter in one call, so the model sees every join while it decides where beats end."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from PIL import Image

from difflab import store
from difflab.core import ui
from difflab.models.minimax_h3 import caption
from difflab.models.minimax_h3 import config as model
from difflab.screenplay import checks as lint
from difflab.screenplay import compiler as compose
from difflab.screenplay import config as h3
from difflab.screenplay import words
from difflab.tools import direct, ruling


@dataclass(frozen=True)
class Version:
    """One look a member has had before the take: `name` "base" is its card's."""

    name: str
    words: str
    picture: Image.Image | None


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
    chunks: list[compose.Chunk] = []
    voices: dict[str, str] = {}
    for clip in data["clips"]:
        cut = str(clip.get("cut") or "").replace("none", "")
        # A clip that goes on, or cuts back to the opening, stays in the scene it returns to.
        scene = " ".join(str(clip.get("scene") or "").split())
        if not scene and chunks:
            scene = chunks[0].scene if cut == "back" else chunks[-1].scene
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
            compose.Line(str(d["who"]), float(d["start"]), str(d["language"]), str(d["line"]))
            for d in clip.get("dialogue", [])
        )
        for d in clip.get("dialogue", []):
            if str(d.get("voice") or "").strip():
                voices.setdefault(str(d["who"]), " ".join(str(d["voice"]).split()))
        changes = {str(m): _look(v) for m, v in (clip.get("looks") or {}).items()}
        made = {m: change for m, change in changes.items() if change[0]}
        chunks.append(
            compose.Chunk(
                purpose=str(clip["purpose"]),
                cast=tuple(dict.fromkeys([*named, *([] if set(named) & set(places) else places[:1])])),
                beats=beats,
                camera_move=str(clip["camera"]["move"]),
                camera_toward=str(clip["camera"]["toward"]),
                mouths=tuple(str(m) for m in clip.get("mouths", [])),
                closing=str(clip["closing"]).strip(),
                irreversible_change=str(clip["irreversible_change"]),
                dialogue=lines,
                seconds=seconds,
                cut=cut,
                opening=str(clip.get("opening") or "").strip(),
                keyframe=bool(chunks) and not cut and clip.get("keyframe") is True,
                looks={m: look for m, (look, _, _) in made.items()},
                look_names={m: name for m, (_, name, _) in made.items()},
                look_from={m: start for m, (_, _, start) in changes.items() if start},
                ambience=str(clip.get("ambience") or "").strip(),
                scene=scene,
            )
        )
    return dataclasses.replace(base, ambience=str(data["ambience"]).strip(), chunks=tuple(chunks), voices=voices)


def _look(given: Any) -> tuple[str, str, str]:
    """A version as (its look, its name, the version it starts from); a bare string is a look left unnamed."""
    if not isinstance(given, dict):
        return " ".join(str(given).split()), "", ""

    def said(field: str) -> str:
        return " ".join(str(given.get(field) or "").split())

    if said("use"):
        return "", "", said("use")
    return said("look"), said("name"), said("from")


def _versions(take: compose.Take, known: dict[str, list[Version]]) -> list[caption.Problem]:
    """Every version a clip starts from is one its member has by then, and every new one has a name of its own."""
    have = {m: {store.BASE, *(v.name for v in known.get(m, []))} for m in take.members}
    out = []
    for number, chunk in enumerate(take.chunks, 1):
        for member, start in chunk.look_from.items():
            names = have.setdefault(member, {store.BASE})
            if member not in chunk.cast:
                out.append(direct.problem(f"clip {number}: 'looks' changes {{{member}}}, who is not in its cast"))
            elif start not in names:
                listed = ", ".join(f'"{n}"' for n in sorted(names))
                out.append(direct.problem(f'clip {number}: {{{member}}} has no version "{start}" by then: name one '
                                          f"of {listed}, or make a new one (rule 15)"))  # fmt: skip
        for member, name in chunk.look_names.items():
            names = have.setdefault(member, {store.BASE})
            if name in names:
                out.append(direct.problem(f'clip {number}: {{{member}}} has a version "{name}" already: give '
                                          f'{{"use": "{name}"}} to show it again, or name the new one '
                                          "otherwise"))  # fmt: skip
            names.add(name)
    return out


def _state(
    known: dict[str, list[Version]], now: dict[str, str], opening: Image.Image | None
) -> tuple[str, list[Image.Image]]:
    """The VERSIONS block and its pictures, Picture 1 the opening photograph."""
    images = [] if opening is None else [opening]
    lines = []
    for member, versions in known.items():
        said = []
        for version in versions:
            if version.picture is not None:
                images.append(version.picture)
            shown = f" (Picture {len(images)})" if version.picture is not None else " (no picture yet)"
            mark = ", now" if now.get(member, store.BASE) == version.name else ""
            said.append(f'"{version.name}"' + (f": {version.words}" if version.words else "") + shown + mark)
        lines.append(f"- {member}: " + "; ".join(said))
    return "\n".join(lines), images


def plan(
    base: compose.Take,
    idea: str,
    total: float,
    aliases: dict[str, str] | None = None,
    shortest: float = h3.DIRECTOR_CLIP_MIN,
    wardrobe: dict[str, list[str]] | None = None,
    versions: dict[str, list[Version]] | None = None,
    now: dict[str, str] | None = None,
    opening: Image.Image | None = None,
) -> compose.Take:
    """The whole take in one call, so the model sees every join while it decides where beats end.

    `total` is the most the take may last; the model chooses how many clips and how long each, between
    `shortest` (never under DIRECTOR_CLIP_MIN) and `base.seconds`.

    Members are then numbered by when they first appear, so a <Subject N> keeps its number from the
    chunk it enters to the end, and the sentences that name it stay the same words. `aliases` are what the
    IDEA calls a member, so its mentions are found under any of its names. `versions` are each member's
    looks before the take, base first, and `now` names the one it enters with.
    """
    aliases = {m: a for m, a in (aliases or {}).items() if a.strip()}
    cast = "\n".join(
        f"{c.id} | {c.kind} | {c.noun} | {c.full()}" + (f" | {aliases.get(c.id) or '-'}" if aliases else "")
        for c in (base.cards[m] for m in base.members)
    )
    moves = "\n".join(
        f"- {name}: {sentence.format(toward='the member')}" for name, sentence in words.CAMERA_MOVES.items()
    )
    window = h3.DIRECTOR_MAIN_BEAT_WINDOW
    shortest, longest = max(h3.DIRECTOR_CLIP_MIN, min(shortest, base.seconds)), round(base.seconds, 1)
    question = words.PLAN_USER.format(
        idea=idea.strip(), cast=cast, aliased=" | also called in the IDEA" if aliases else "", opening=base.opening,
        total=round(total), shortest=shortest, longest=longest,
        early=window[0] * 100, late=window[1] * 100, moves=moves, beat_min=h3.DIRECTOR_BEAT_MIN,
    )  # fmt: skip
    worn = "\n".join(f"- {m}: {', '.join(names)}" for m, names in (wardrobe or {}).items() if names)
    if worn:
        question += "\n\n" + words.PLAN_WARDROBE.format(wardrobe=worn)
    known = versions or {}
    listed, images = _state(known, now or {}, opening)
    if listed:
        question += "\n\n" + words.PLAN_VERSIONS.format(versions=listed)
    system = words.PLAN_SYSTEM.format(
        longest=longest, look_words=h3.DIRECTOR_LOOK_WORDS, name_words=h3.DIRECTOR_LOOK_NAME_WORDS,
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
            if (number == 1 or clip.get("cut") == "new") and not str(clip.get("scene") or "").strip():
                opens = "opens the take" if number == 1 else 'is a "new" cut'
                out.append(direct.problem(f'clip {number}: "scene" is empty, and the clip {opens}: give its heading'))
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
                                      f"{total / count:.1f} s, so shorten each clip's main beat and its closing beat "
                                      f"toward the least they may be, or merge two"))  # fmt: skip
        # Never returned alone: an answer stopped here counted one problem and was kept over answers that
        # broke fewer rules, unchecked - a 72 s plan whose closings named members missing from the next cast.
        take = _take_from_plan(data, base)
        out += lint.plan_rules(take) + _versions(take, known) + ruling.mouths(take) + ruling.leaving(take)
        out += ruling.changed(take)
        if not story(data):
            out.append(direct.problem('"story" lists every action of the IDEA with the clip that shows it'))
        return out + ruling.story(idea, list(story(data)), take)

    data = direct.ask_json(
        system, question, images, gate, "plan", h3.DIRECTOR_TOKENS["plan"], repairs=h3.DIRECTOR_PLAN_REPAIRS
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
