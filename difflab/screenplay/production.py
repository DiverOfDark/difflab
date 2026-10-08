"""A production and its shots, as the page and the playground make them. Every call is one transaction."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from difflab import config, store
from difflab.core import media
from difflab.core.keys import key
from difflab.core.settings import Continuity, EditSettings, Settings
from difflab.models.minimax_h3 import pipeline
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import compiler, crops, drawn, looks, settings, timings, words
from difflab.screenplay import config as knobs

_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]*")
MEMBER_ID = re.compile(rf"[a-z][a-z_]{{0,{knobs.DIRECTOR_ID_LENGTH - 1}}}")  # as `tools.direct.ID` asks the survey
MEMBER_KINDS = ("person", "animal", "object", "environment")
OPENING_NOTE = "the opening frame"  # what the planner writes on the keyframe it cuts from the first photo
CUT_NOTE = "the plan's cut"  # on a keyframe the planner composes for a "new" cut
ASPECTS = (0.25, 4.0)  # width over height a canvas may be given


def take_seconds(manifest: dict[str, Any], longest: float) -> float:
    """How long the director makes a chapter: as given, or a production from before it chose the count
    as its shot count times the render length."""
    if given := float(manifest.get("take_seconds") or 0):
        return given
    return int(manifest.get("shots", 0) or 0) * longest


def create(
    slug: str,
    title: str = "",
    video: VideoSettings | None = None,
    continuity: Continuity | None = None,
) -> store.Production:
    """A production with one chapter, rendering at these settings - the code defaults when none."""
    if not _SLUG.fullmatch(slug):
        raise ValueError(f"slug {slug!r}: lower-case letters, digits, _ and -, and it names files")
    with store.connect() as connection, store.transaction(connection):
        if store.production_by_slug(connection, slug) is not None:
            raise ValueError(f"a production called {slug!r} exists")
        made = store.add_production(connection, slug, title or slug.replace("_", " ").replace("-", " "))
        for tool, given in (("render_chunk", video), ("handoff", continuity)):
            if given is not None:
                store.put_overrides(connection, settings.PRODUCTION, str(made.id), tool, _overrides(given))
        return made


def add_shot(
    production_id: int,
    prompt: str,
    seconds: float,
    keyframe: str | Path | None = None,
    seed: int | None = None,
    action: str = "",
    chained: bool = True,
    chapter: int = 0,
) -> store.Shot:
    """A shot at the end of a chapter, its prompt pinned. `keyframe` is a picture under output/; without
    one the shot opens on the previous shot's handoff frame. The seed is the production's plus the
    shot's position unless given: one seed across shots makes them rhyme rather than continue."""
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        if production is None:
            raise LookupError(f"no production {production_id}")
        chapters = store.chapters_of(connection, production_id)
        if chapter >= len(chapters):
            raise LookupError(f"production {production.slug} has {len(chapters)} chapter(s), not {chapter + 1}")
        shot = store.add_shot(connection, chapters[chapter].id, seconds, action=action, chained=chained)
        store.set_prompt(connection, shot.id, prompt.strip())
        if keyframe is not None:
            store.provide_keyframe(connection, shot.id, store.relative(_under_output(keyframe)))
        video, _ = settings.effective("render_chunk", production_id)
        assert isinstance(video, VideoSettings)
        store.set_clip_seed(connection, shot.id, seed if seed is not None else video.seed + shot.position)
        store.touch_production(connection, production_id)
        return shot


def update_shot(shot_id: int, **changes: Any) -> store.Shot:
    """`prompt`, `keyframe` (a path, `composed`, `opening` for the planner's cut of the first photo - on a
    later shot, a cut back to it that opens on the opening arrangement - or
    "" to chain again), `opening` (`keyframe`, `chain` or `refs`: a keyframe kept as it is when it has one
    and composed when not, the shot before's last frame, or no first frame at all), `landed` (whether the
    shot before ends on that keyframe), `seed`, the plan's `closing`, `cast`, `camera_move`,
    `camera_toward`, `dialogue` and `sound`, `seconds` (held to `SHOT_SECONDS`), and the shot's own
    columns. A plan edit pins the shot and recompiles its prompt at once, and a closing the next shot's
    too, since it opens on it; a keyframe change recompiles the shot and the one before, since whether that
    one lands on it and whether a tail crosses are both in their words."""
    with store.connect() as connection, store.transaction(connection):
        shot = store.shot(connection, shot_id)
        if shot is None:
            raise LookupError(f"no shot {shot_id}")
        chapter = store.chapter(connection, shot.chapter_id)
        assert chapter is not None
        joined = False
        if "prompt" in changes:
            store.set_prompt(connection, shot_id, str(changes.pop("prompt")).strip())
        if "keyframe" in changes:
            given = changes.pop("keyframe")
            if given == store.COMPOSED:
                store.compose_keyframe(connection, shot_id)
            elif given == "opening":
                opening = _opening_frame(connection, chapter.production_id)
                if opening is None:
                    raise ValueError("a shot opens on the first photo once a plan has cut it")
                store.provide_keyframe(connection, shot_id, opening, note=OPENING_NOTE)
            elif given:
                store.provide_keyframe(connection, shot_id, store.relative(_under_output(str(given))))
            else:
                store.chain_keyframe(connection, shot_id)
            joined = True
        if "opening" in changes:
            _open(connection, shot, str(changes.pop("opening")))
            joined = True
        if "seconds" in changes:
            low, high = knobs.SHOT_SECONDS
            seconds = min(max(float(changes.pop("seconds")), low), high)
            if seconds != shot.seconds:
                store.update_shot(connection, shot_id, seconds=seconds)
                _stale(connection, [shot], prompts=False)
                joined = True
        if "landed" in changes:
            store.land_keyframe(connection, shot_id, bool(changes.pop("landed")))
            joined = True
        if "seed" in changes:
            store.set_clip_seed(connection, shot_id, int(changes.pop("seed")))
        planned = {name: changes.pop(name) for name in PLAN_FIELDS if name in changes}
        if planned:
            _edit_plan(connection, chapter, shot, **planned)
        if changes:
            store.update_shot(connection, shot_id, **changes)
        if joined:
            production = store.production(connection, chapter.production_id)
            assert production is not None
            shots = store.shots_of(connection, chapter.id)
            for one in shots[max(0, shot.position - 1) : shot.position + 1]:
                if one.plan.get("beats"):
                    recompile(connection, production, chapter, one.id)
        store.touch_production(connection, chapter.production_id)
        updated = store.shot(connection, shot_id)
        assert updated is not None
        return updated


PLAN_FIELDS = ("actions", "closing", "cast", "camera_move", "camera_toward", "dialogue", "sound")
OPENINGS = ("keyframe", "chain", "refs")


def _open(connection: Any, shot: store.Shot, opening: str) -> None:
    keyframe = store.keyframe_of(connection, shot.id)
    if opening == "refs":
        store.refs_keyframe(connection, shot.id)
    elif opening == "chain":
        if shot.position == 0:
            raise ValueError("the chapter's first shot has no shot before it to open on")
        store.chain_keyframe(connection, shot.id)
    elif opening == "keyframe":
        if keyframe.source not in (store.COMPOSED, store.PROVIDED):
            store.compose_keyframe(connection, shot.id)
    else:
        raise ValueError(f"a shot opens on one of {', '.join(OPENINGS)}")


def opening_of(keyframe: store.Keyframe) -> str:
    return {store.CHAINED: "chain", store.REFS: "refs"}.get(keyframe.source, "keyframe")


def _edit_plan(connection: Any, chapter: store.Chapter, shot: store.Shot, **planned: Any) -> None:
    production = store.production(connection, chapter.production_id)
    assert production is not None
    members = {m.slug: m for m in store.members_of(connection, production.id)}
    plan = dict(shot.plan)
    if not plan.get("beats"):
        raise ValueError("this shot has no plan to edit: plan the chapter first")
    shots = store.shots_of(connection, chapter.id)
    following = shots[shot.position + 1] if shot.position + 1 < len(shots) else None
    if "actions" in planned:
        # One per beat, null keeping it. An edited beat drops the writer's prose for it, and the prompt
        # says the action as planned until a re-plan writes it again.
        given = list(planned["actions"] or [])
        if len(given) != len(plan["beats"]):
            raise ValueError(f"one action per beat: this shot has {len(plan['beats'])}, null to keep one")
        beats = []
        for beat, action in zip(plan["beats"], given, strict=True):
            action = " ".join(str(action).split()) if action is not None else ""
            unknown = sorted(set(compiler.PLACEHOLDER.findall(action)) - set(plan.get("cast", [])))
            if unknown:
                raise ValueError(f"{', '.join(unknown)} is not in this shot's cast: cast it first")
            beats.append({**beat, "action": action, "prose": ""} if action and action != beat["action"] else beat)
        plan["beats"] = beats
    if "closing" in planned:
        closing = " ".join(str(planned["closing"]).split())
        closed = list(dict.fromkeys(compiler.PLACEHOLDER.findall(closing)))
        if not closed:
            raise ValueError("the closing names who is on screen as {id}, e.g. {cat}")
        unknown = sorted(set(closed) - set(members))
        if unknown:
            raise ValueError(f"no member called {', '.join(unknown)}")
        after = set(following.plan.get("cast", [])) if following is not None else set(closed)
        missing = sorted(set(closed) - after)
        if missing:
            raise ValueError(
                f"the next shot opens on this closing, and {', '.join(missing)} is not in its cast: put it there first"
            )
        plan["closing"] = closing
        plan["faces_at_close"] = [m for m in closed if members[m].kind in compiler.LIVING]
        plan["cast"] = list(dict.fromkeys([*plan.get("cast", []), *closed]))
    if "cast" in planned:
        cast = [str(m) for m in planned["cast"]]
        unknown = [m for m in cast if m not in members]
        if unknown:
            raise ValueError(f"no member called {', '.join(unknown)}")
        environments = [m for m in members.values() if m.kind == "environment"]
        cast = list(dict.fromkeys([*cast, *(e.slug for e in environments)]))
        opening = compiler.opening(compiler.take_of(connection, production, chapter), shot.position)
        named = set(compiler.PLACEHOLDER.findall(_said(plan) + " " + opening))
        dropped = sorted(named - set(cast))
        if dropped:
            raise ValueError(
                f"{', '.join(dropped)} is named in this shot's beats, its closing or its opening: it stays in the cast "
                "until a re-plan writes it out"
            )
        plan["cast"] = cast
    if "camera_move" in planned:
        move = str(planned["camera_move"])
        if move not in words.CAMERA_MOVES:
            raise ValueError(f"the camera move is one of {', '.join(words.CAMERA_MOVES)}")
        plan["camera_move"] = move
    if "camera_toward" in planned:
        plan["camera_toward"] = str(planned["camera_toward"])
    if plan.get("camera_toward") not in plan.get("cast", []):
        raise ValueError("the camera moves toward a member in the shot's cast")
    if "dialogue" in planned:
        beats = [b for b in plan["beats"] if b.get("phase") != "settle"]
        main = next((b for b in beats if b.get("phase") in ("primary", "handoff")), beats[0] if beats else {})
        lines = []
        for line in planned["dialogue"] or []:
            who, said = str(line.get("who", "")), " ".join(str(line.get("line", "")).split())
            if not said:
                continue
            if who not in plan.get("cast", []) or members[who].kind not in compiler.LIVING:
                raise ValueError(f"{who or 'a line'}: only a person or animal in the shot's cast speaks")
            lines.append(
                {
                    "who": who,
                    "start": float(line.get("start") or main.get("start", 0.0)),
                    "language": str(line.get("language") or "English"),
                    "line": said,
                    "voice": str(line.get("voice") or f"a natural {members[who].noun}'s voice"),
                }
            )
        plan["dialogue"] = lines
        spoken = [f"{{{who}}}'s spoken line" for who in dict.fromkeys(line["who"] for line in lines)]
        kept = [m for m in plan.get("mouths", []) if not str(m).endswith("'s spoken line")]
        plan["mouths"] = [*kept, *spoken]
    if "sound" in planned:  # the shot's own sounds, after the place's
        plan["sounds"] = " ".join(str(planned["sound"] or "").split())
    move, toward = plan.get("camera_move", ""), plan.get("camera_toward", "")
    dialogue = " / ".join(f"{line['who']}: {line['line']}" for line in plan.get("dialogue", []))
    _sync_castings(connection, shot, [str(m) for m in plan.get("cast", [])])
    store.update_shot(connection, shot.id, plan=plan, camera=f"{move} → {toward}", dialogue=dialogue, pinned=True)
    recompile(connection, production, chapter, shot.id)
    if "closing" in planned and following is not None:
        recompile(connection, production, chapter, following.id)  # it opens on this closing
    if "closing" in planned or "cast" in planned:  # the castings may have moved what later shots carry
        for later in shots[shot.position + 1 :]:
            if later.plan.get("beats"):
                recompile(connection, production, chapter, later.id)


def _sync_castings(connection: Any, shot: store.Shot, cast: list[str]) -> None:
    """The shot's castings to its plan's cast: a member added comes as it stands, one dropped goes."""
    chapter = store.chapter(connection, shot.chapter_id)
    assert chapter is not None
    members = {m.slug: m.id for m in store.members_of(connection, chapter.production_id)}
    held = {c.member_id: c for c in store.castings_of(connection, shot.id)}
    for slug in cast:
        if members[slug] not in held:
            store.add_casting(connection, shot.id, members[slug])
    for member_id, casting in held.items():
        if member_id not in {members[slug] for slug in cast}:
            store.delete_casting(connection, casting.id)


def _said(plan: dict[str, Any]) -> str:
    beats = plan.get("beats", [])
    parts = [str(b.get(k, "")) for b in beats for k in ("action", "prose")]
    parts += ["{" + str(who) + "}" for b in beats for who in b.get("who", [])]
    return " ".join([*parts, *(str(plan.get(k, "")) for k in ("closing", "summary", "sounds"))])


def recompile(
    connection: Any, production: store.Production, chapter: store.Chapter, shot_id: int, written_for: str = ""
) -> None:
    """The prompt compiled again from the shot's plan as it now stands - code, no model - as
    `compile_shot` would; a plan it cannot compile leaves the prompt stale for a re-plan to write.
    `written_for` says why, when a page should know: "looks" when its looks moved under it."""
    shots = store.shots_of(connection, chapter.id)
    position = next(one.position for one in shots if one.id == shot_id)
    prompt = store.prompt_of(connection, shot_id)
    if prompt.pinned:
        _stale(connection, [shots[position]], prompts=False)
        return
    try:
        take = compiler.take_of(connection, production, chapter)
        compiled = compiler.compile_chunk(take, position)
    except (KeyError, StopIteration, ValueError):
        _stale(connection, [shots[position]])
        return
    pictures = compiler.pictures_of(connection, production)
    handed = [pictures[name] for name in compiled.references if name in pictures]
    made = key(
        "compile_shot", 1, production=production.id, shot=shot_id, prompt=compiled.prompt, pictures=compiled.references
    )
    if compiled.prompt != prompt.text or handed != prompt.pictures:
        store.set_compiled(connection, shot_id, compiled.prompt, handed, prompt.checks, str(made), written_for)
        _stale(connection, [shots[position]], prompts=prompt.state == store.STALE)


def pick_drawn(member_id: int, seed: int) -> dict[str, int]:
    """One of a member's variant drawings made its reference; the shots casting it go stale."""
    drawn.pick(member_id, seed)
    with store.connect() as connection, store.transaction(connection):
        member = store.member(connection, member_id)
        assert member is not None
        recompile_casting(connection, member_id)
        return _stale(connection, _shots_of(connection, member.production_id, member.slug))


def redo_keyframe(shot_id: int, note: str = "") -> int:
    """The shot's first frame composed again on a new seed, told `note` as well. Returns its chapter."""
    with store.connect() as connection, store.transaction(connection):
        shot = store.shot(connection, shot_id)
        if shot is None:
            raise LookupError(f"no shot {shot_id}")
        store.compose_keyframe(connection, shot_id, seed=random.randrange(2**32), note=" ".join(note.split()))
        _stale(connection, [shot])
        return shot.chapter_id


def recompile_casting(connection: Any, member_id: int) -> None:
    """Every prompt of a shot that casts the member, compiled again: its pictures changed."""
    member = store.member(connection, member_id)
    production = None if member is None else store.production(connection, member.production_id)
    if member is None or production is None:
        return
    for chapter in store.chapters_of(connection, production.id):
        for shot in store.shots_of(connection, chapter.id):
            if member.slug in shot.plan.get("cast", []):
                recompile(connection, production, chapter, shot.id)


def _stale(connection: Any, shots: Iterable[store.Shot], prompts: bool = True) -> dict[str, int]:
    """What these shots have made, marked stale: their prompts (unless written by hand), their composed
    keyframes and their clips. How many of each moved."""
    moved = {"prompts": 0, "keyframes": 0, "clips": 0}
    for shot in shots:
        prompt = store.prompt_of(connection, shot.id)
        if prompts and prompt.text and not prompt.pinned and prompt.state != store.STALE:
            store.set_made(connection, "prompt", prompt.id, store.STALE)
            moved["prompts"] += 1
        keyframe = store.keyframe_of(connection, shot.id)
        if keyframe.source == store.COMPOSED and keyframe.artifact and keyframe.state != store.STALE:
            store.set_made(connection, "keyframe", keyframe.id, store.STALE)
            moved["keyframes"] += 1
        clip = store.clip_of(connection, shot.id)
        if clip.artifact and clip.state != store.STALE:
            store.set_made(connection, "clip", clip.id, store.STALE)
            moved["clips"] += 1
    return moved


def _shots_of(connection: Any, production_id: int, casting: str | None = None) -> list[store.Shot]:
    return [
        shot
        for chapter in store.chapters_of(connection, production_id)
        for shot in store.shots_of(connection, chapter.id)
        if casting is None or casting in shot.plan.get("cast", [])
    ]


def _opening_frame(connection: Any, production_id: int) -> str | None:
    """The opening frame the last plan run cut from the first photo."""
    row = connection.execute(
        "SELECT outputs FROM jobs WHERE tool = 'cut_crops' AND state IN (?, ?) "
        "AND json_extract(inputs, '$.production') = ? ORDER BY id DESC LIMIT 1",
        (store.DONE, store.SKIPPED, production_id),
    ).fetchone()
    return None if row is None else json.loads(row[0]).get("opening")


def update(production_id: int, **changes: Any) -> dict[str, int]:
    """`title`; `story`, the idea or text it is planned from; `opening`, the first frame in words with
    each member as {id} ("" gives it back to what the planner reads off the first photo); `aspect`, the
    canvas's width over height (None: the first photo's). A changed story, opening or aspect makes every
    prompt, composed keyframe and clip stale until a re-plan."""
    unknown = set(changes) - {"title", "story", "opening", "aspect"}
    if unknown:
        raise ValueError(f"a production has no {', '.join(sorted(unknown))} to edit")
    moved = {"prompts": 0, "keyframes": 0, "clips": 0}
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        if production is None:
            raise LookupError(f"no production {production_id}")
        if "title" in changes:
            title = " ".join(str(changes["title"]).split())
            if not title:
                raise ValueError("a title, in words")
            store.update_production(connection, production_id, title=title)
        replan = False
        if "story" in changes:
            story = str(changes["story"])
            if production.story_kind == "video":
                raise ValueError("a video production's story is its captions' one change: begin it again to change it")
            if production.story_kind == "text":
                _resplit(connection, production_id, story)
                story = story.strip()
            else:
                story = lines(story)
            if not story:
                raise ValueError("a story, in your own words: what happens")
            if story != production.story:
                store.update_production(connection, production_id, story=story)
                replan = True
        if "opening" in changes:
            words_ = " ".join(str(changes["opening"]).split())
            slugs = {m.slug for m in store.members_of(connection, production_id)}
            strangers = sorted(set(compiler.PLACEHOLDER.findall(words_)) - slugs)
            if strangers:
                raise ValueError(f"no member called {', '.join(strangers)}: name members as {{id}}")
            if words_ and not compiler.PLACEHOLDER.search(words_):
                raise ValueError("name who is in the first frame as {id}, e.g. {cat}")
            manifest = {k: v for k, v in production.manifest.items() if k != "opening_words"}
            if words_:
                manifest["opening_words"] = words_
            if manifest != production.manifest:
                store.update_production(connection, production_id, manifest=manifest)
                replan = True
        redraw = 0
        if "aspect" in changes:
            aspect = None if changes["aspect"] in (None, "") else float(changes["aspect"])
            if aspect is not None and not ASPECTS[0] <= aspect <= ASPECTS[1]:
                raise ValueError(f"an aspect between {ASPECTS[0]:g} and {ASPECTS[1]:g}, width over height")
            now = store.production(connection, production_id)
            assert now is not None
            if aspect != now.manifest.get("aspect"):
                # The canvas goes with it: the next plan cuts the new one, and until then a place is drawn
                # at the new aspect rather than at a canvas no render will use.
                manifest = {k: v for k, v in now.manifest.items() if k != "canvas"}
                store.update_production(connection, production_id, manifest={**manifest, "aspect": aspect})
                redraw = sum(
                    1
                    for m in store.members_of(connection, production_id)
                    if m.kind == "environment"
                    for r in store.refs_of(connection, m.id)
                    if r.generated and r.artifact
                )
                replan = True
        if replan:
            moved = _stale(connection, _shots_of(connection, production_id))
        moved["references"] = redraw
    return moved


def lines(text: str) -> str:
    """Spaces tidied within each line, and one blank line at most between paragraphs."""
    kept = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for i, line in enumerate(kept) if line or (i and kept[i - 1])).strip()


def _resplit(connection: Any, production_id: int, text: str) -> None:
    from difflab.screenplay.ingest import chapters_of  # ingest begins productions through this module

    parts = chapters_of(text)
    chapters = store.chapters_of(connection, production_id)
    if len(parts) < len(chapters):
        raise ValueError(f"the text makes {len(parts)} chapter(s) and the production has {len(chapters)}")
    for position, (heading, story) in enumerate(parts):
        if position < len(chapters):
            connection.execute(
                "UPDATE chapters SET title = ?, story = ? WHERE id = ?", (heading, story, chapters[position].id)
            )
        else:
            store.add_chapter(connection, production_id, heading, story)


# --- the cast, by hand -----------------------------------------------------------------------------
def update_member(member_id: int, **changes: Any) -> dict[str, int]:
    """`noun`, `kind`, `possessive`, `find`, `traits` ({part, phrase, short, name} each), `lights`, and
    `photographed`: false says it is decided in words, and lets go of every picture of it. The member is
    pinned - a re-plan leaves it as written - and every shot that casts it goes stale."""
    unknown = set(changes) - {"noun", "kind", "possessive", "find", "traits", "lights", "photographed", "pinned"}
    if unknown:
        raise ValueError(f"a member has no {', '.join(sorted(unknown))} to edit")
    with store.connect() as connection, store.transaction(connection):
        member = store.member(connection, member_id)
        if member is None:
            raise LookupError(f"no member {member_id}")
        if changes.get("pinned") is False and len(changes) == 1:
            store.update_member(connection, member_id, pinned=False)
            return {"prompts": 0, "keyframes": 0, "clips": 0}
        others = [m for m in store.members_of(connection, member.production_id) if m.id != member_id]
        kind = str(changes.get("kind", member.kind))
        identity = dict(member.identity)
        if "traits" in changes:
            identity["traits"] = _traits(changes["traits"])
        if "lights" in changes:
            identity["lights"] = [" ".join(str(x).split()) for x in changes["lights"] or [] if str(x).strip()]
        if "photographed" in changes:
            identity["photographed"] = bool(changes["photographed"])
        fields = {
            "kind": kind,
            "noun": " ".join(str(changes.get("noun", member.noun)).split()),
            "possessive": str(changes.get("possessive", member.possessive)),
            "find": " ".join(str(changes.get("find", member.find)).split()),
        }
        _check_member(member.slug, fields, identity, others)
        if not identity.get("photographed", True):
            for reference in store.refs_of(connection, member_id):
                if not reference.generated:
                    store.delete_ref(connection, reference.id)
        store.update_member(connection, member_id, **fields, identity=identity, pinned=True)
        moved = _stale(connection, _shots_of(connection, member.production_id, member.slug))
        store.touch_production(connection, member.production_id)
    return moved


def add_member(production_id: int, slug: str, **given: Any) -> store.Member:
    """A member decided in words, pinned: a re-plan casts it where the story needs it."""
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        if production is None:
            raise LookupError(f"no production {production_id}")
        if not MEMBER_ID.fullmatch(slug):
            raise ValueError(f"{slug!r}: an id is one lower-case word, or words joined by _")
        others = store.members_of(connection, production_id)
        if any(m.slug == slug for m in others):
            raise ValueError(f"{slug} is already a member")
        fields = {
            "kind": str(given.get("kind", "")),
            "noun": " ".join(str(given.get("noun") or slug.replace("_", " ")).split()),
            "possessive": str(given.get("possessive") or ("its" if given.get("kind") != "person" else "his")),
            "find": " ".join(str(given.get("find", "")).split()),
        }
        identity = {
            "traits": _traits(given.get("traits") or []),
            "lights": [" ".join(str(x).split()) for x in given.get("lights") or [] if str(x).strip()],
            "photographed": False,
        }
        _check_member(slug, fields, identity, others)
        made = store.put_member(
            connection, production_id, slug, fields["kind"], fields["noun"], fields["possessive"], fields["find"]
        )
        store.update_member(connection, made.id, identity=identity, pinned=True)
        dropped = [str(m) for m in production.manifest.get("dropped", []) if m != slug]
        store.update_production(connection, production_id, manifest={**production.manifest, "dropped": dropped})
        found = store.member(connection, made.id)
        assert found is not None
        return found


def delete_member(member_id: int) -> dict[str, int]:
    """The member and its pictures, and remembered as dropped so a re-plan's survey does not bring it back."""
    with store.connect() as connection, store.transaction(connection):
        member = store.member(connection, member_id)
        if member is None:
            raise LookupError(f"no member {member_id}")
        if member.kind == "environment":
            raise ValueError("every chapter happens somewhere: edit the place instead of deleting it")
        casting = _shots_of(connection, member.production_id, member.slug)
        pinned = [shot.position + 1 for shot in casting if shot.pinned]
        if pinned:
            raise ValueError(f"shot(s) {', '.join(map(str, pinned))} were edited by hand and cast {member.slug}")
        production = store.production(connection, member.production_id)
        assert production is not None
        dropped = sorted({*map(str, production.manifest.get("dropped", [])), member.slug})
        store.update_production(connection, production.id, manifest={**production.manifest, "dropped": dropped})
        moved = _stale(connection, casting)
        store.delete_member(connection, member_id)
    return moved


def gaps(production: store.Production, member: store.Member, references: list[store.Reference]) -> list[dict[str, str]]:
    """The last cut's gaps for this member that no given or drawn picture has filled since."""
    held = {compiler.DRAWN_ROLE[member.kind] if r.generated else r.role for r in references if r.artifact}
    return [
        {"role": str(gap["role"]), "why": str(gap["why"])}
        for gap in production.manifest.get("gaps", [])
        if gap["member"] == member.slug and gap["role"] not in held
    ]


def give_ref(member_id: int, path: str, role: str) -> dict[str, int]:
    """A picture of the member from outside the photographs, fitted the way a crop is so H3 takes it as it is."""
    with store.connect() as connection, store.transaction(connection):
        member = store.member(connection, member_id)
        production = None if member is None else store.production(connection, member.production_id)
        if member is None or production is None:
            raise LookupError(f"no member {member_id}")
        if role not in crops.WANTS[member.kind]:
            raise ValueError(f"a {member.kind} takes a {' or '.join(crops.WANTS[member.kind])} picture, not {role}")
        given, kept, box, size = _fitted(production, path, role, f"{member.slug}_{role}")
        store.add_ref(
            connection, member.id, role, given, list(box.pixels()), list(size), (size[0] // 32) * (size[1] // 32),
            [member.slug], box.area, artifact=kept, provided=given,
        )  # fmt: skip
        return _stale(connection, _shots_of(connection, production.id, member.slug))


def _fitted(production: store.Production, path: str, role: str, name: str) -> tuple[str, str, crops.Box, list[int]]:
    """A picture from outside, fitted the way a crop is so H3 takes it as it is: (given, kept, box, size)."""
    canvas = production.manifest.get("canvas")
    if not canvas:
        raise ValueError("plan the chapter first: its canvas decides the picture's size")
    source = Path(path) if Path(path).is_absolute() else config.OUTPUT_DIR / path
    frame = media.frame_from_image(source)
    pixels = knobs.DIRECTOR_ENVIRONMENT_PIXELS if role == "environment" else knobs.DIRECTOR_VIEW_PIXELS
    whole = crops.Box(0, 0, frame.width, frame.height)
    box, size = crops.fitted(whole, pixels, (frame.width, frame.height), canvas["width"] * canvas["height"])
    geometry = pipeline.Geometry(int(canvas["width"]), int(canvas["height"]), int(canvas["frame_count"]))
    if pipeline.reference_size(*size, geometry, "match") != size:
        raise ValueError(f"a {size[0]}×{size[1]} picture would be resampled by H3's reference sizing")
    folder = config.OUTPUT_DIR / "blobs" / "given"
    folder.mkdir(parents=True, exist_ok=True)
    kept = folder / f"{source.stem}_{name}.png"
    frame.crop(box.pixels()).resized(*size).pil().save(kept)
    return store.relative(source), store.relative(kept), box, list(size)


def detach_ref(ref_id: int) -> dict[str, int]:
    """A picture that does not show its member, let go of: remembered by photo and role, so the next
    cut does not attach it again."""
    with store.connect() as connection, store.transaction(connection):
        reference = store.ref(connection, ref_id)
        if reference is None:
            raise LookupError(f"no reference {ref_id}")
        member = store.member(connection, reference.member_id)
        production = None if member is None else store.production(connection, member.production_id)
        assert member is not None and production is not None
        entry = [member.slug, reference.photo, reference.role]
        detached = [list(e) for e in production.manifest.get("detached", [])]
        if entry not in detached:
            detached.append(entry)
        store.update_production(connection, production.id, manifest={**production.manifest, "detached": detached})
        store.delete_ref(connection, ref_id)
        moved = _stale(connection, _shots_of(connection, production.id, member.slug))
        if reference.generated:
            recompile_casting(connection, member.id)
        return moved


def resolve_conflict(conflict_id: int, resolution: str) -> int:
    """Which wording wins, recorded on the conflict; "" opens it again. Returns the production."""
    with store.connect() as connection, store.transaction(connection):
        found = store.conflict(connection, conflict_id)
        member = None if found is None else store.member(connection, found.member_id)
        if found is None or member is None:
            raise LookupError(f"no conflict {conflict_id}")
        store.resolve_conflict(connection, conflict_id, " ".join(resolution.split()))
        return member.production_id


def production_of_member(member_id: int) -> int | None:
    with store.connect() as connection:
        member = store.member(connection, member_id)
        return None if member is None else member.production_id


def production_of_ref(ref_id: int) -> int | None:
    with store.connect() as connection:
        reference = store.ref(connection, ref_id)
        member = None if reference is None else store.member(connection, reference.member_id)
        return None if member is None else member.production_id


def _traits(given: Any) -> list[dict[str, str]]:
    out = []
    for trait in given or []:
        phrase = " ".join(str(trait.get("phrase", "")).split())
        if not phrase:
            continue
        short = " ".join(str(trait.get("short") or phrase).split())
        name = " ".join(str(trait.get("name") or short.split()[-1]).split())
        out.append({"part": str(trait.get("part", "")), "phrase": phrase, "short": short, "name": name})
    return out


def _check_member(slug: str, fields: dict[str, str], identity: dict[str, Any], others: list[store.Member]) -> None:
    """What the survey's and the tidy stage's gates ask of a member the director made."""
    kind = fields["kind"]
    if kind not in MEMBER_KINDS:
        raise ValueError(f"{slug}: the kind is one of {', '.join(MEMBER_KINDS)}")
    if fields["possessive"] not in ("his", "her", "their", "its"):
        raise ValueError(f"{slug}: the possessive is his, her, their or its")
    if not fields["noun"]:
        raise ValueError(f"{slug}: a noun, what it is in a word or two")
    if kind != "environment" and not any(m.kind == "environment" for m in others):
        raise ValueError(f"{slug} is the chapter's only place: it stays an environment")
    traits = identity.get("traits", [])
    if not traits:
        raise ValueError(f"{slug}: at least one trait that must not drift")
    for trait in traits:
        if trait["part"] not in knobs.DIRECTOR_TRAIT_ORDER:
            raise ValueError(f"{slug}: a trait's part is one of {', '.join(knobs.DIRECTOR_TRAIT_ORDER)}")
    if kind == "environment" and not identity.get("lights"):
        raise ValueError(f"{slug}: a place says what lights it, at least one light")


def remove_shot(shot_id: int) -> None:
    with store.connect() as connection, store.transaction(connection):
        store.delete_shot(connection, shot_id)


def reroll(shot_id: int, seed: int | None = None) -> tuple[int, int]:
    """One more attempt at a shot's clip: a new seed, and the clip goes stale. The shots chained after
    it go stale by key when the next run reaches them. Returns the seed and the shot's chapter."""
    chosen = seed if seed is not None else random.randrange(2**32)
    with store.connect() as connection, store.transaction(connection):
        shot = store.shot(connection, shot_id)
        if shot is None:
            raise LookupError(f"no shot {shot_id}")
        _reroll(connection, shot, chosen)
    return chosen, shot.chapter_id


def _reroll(connection: Any, shot: store.Shot, seed: int, by: str | None = None) -> None:
    clip = store.set_clip_seed(connection, shot.id, seed)
    state = store.STALE if clip.artifact else store.MISSING
    store.set_made(connection, "clip", clip.id, state, review=clip.review if by else None, reviewed_by=by)
    # A run the judge paused on this chapter would build its later shots on the clip being replaced.
    _tell_paused(connection, shot.chapter_id, "cancel")


def _tell_paused(connection: Any, chapter_id: int, cmd: str) -> None:
    for run in store.runs(connection):
        if run.scope == f"chapter:{chapter_id}" and run.state == store.PAUSED:
            store.add_command(connection, "run", run.id, {"cmd": cmd})


# --- castings: who is in a shot, and how they look there --------------------------------------------
def add_casting(shot_id: int, member: str, **changes: Any) -> int:
    """A member into the shot, as it stands unless `parent`, `amend` or `label` say otherwise."""
    with store.connect() as connection, store.transaction(connection):
        _, production = _shot_and_production(connection, shot_id)
        found = store.member_by_slug(connection, production.id, member)
        if found is None:
            raise ValueError(f"no member called {member}")
        held = {c.member_id: c for c in store.castings_of(connection, shot_id)}
        if found.id in held:
            raise ValueError(f"{member} is in this shot already")
        kinds = {m.id: m.kind for m in store.members_of(connection, production.id)}
        if found.kind == "environment" and any(kinds[m] == "environment" for m in held):
            raise ValueError("a shot happens in one place: change the place's look instead")
        casting = store.add_casting(connection, shot_id, found.id)
        _recast(connection, production, casting, changes)
        return production.id


def edit_casting(casting_id: int, **changes: Any) -> int:
    """`parent` (None its last look, "base", or an earlier look's id), `amend`, `label`; restales its shots."""
    unknown = set(changes) - {"parent", "amend", "label"}
    if unknown:
        raise ValueError(f"a casting has no {', '.join(sorted(unknown))} to edit")
    with store.connect() as connection, store.transaction(connection):
        casting = store.casting(connection, casting_id)
        if casting is None:
            raise LookupError(f"no casting {casting_id}")
        _, production = _shot_and_production(connection, casting.shot_id)
        _recast(connection, production, casting, changes)
        return production.id


def remove_casting(casting_id: int) -> int:
    """The member out of the shot, unless its words name it."""
    with store.connect() as connection, store.transaction(connection):
        casting = store.casting(connection, casting_id)
        if casting is None:
            raise LookupError(f"no casting {casting_id}")
        shot, production = _shot_and_production(connection, casting.shot_id)
        member = store.member(connection, casting.member_id)
        assert member is not None
        if member.kind == "environment":
            raise ValueError("every shot happens somewhere: change the place's look instead")
        chapter = store.chapter(connection, shot.chapter_id)
        assert chapter is not None
        said = _said(shot.plan)
        if shot.plan.get("beats"):
            said += " " + compiler.opening(compiler.take_of(connection, production, chapter), shot.position)
        if member.slug in compiler.PLACEHOLDER.findall(said):
            raise ValueError(f"{member.slug} is named in this shot's beats, its closing or its opening: it stays")
        _recast(connection, production, casting, None)
        return production.id


def upload_look(casting_id: int, path: str) -> int:
    """A given picture as the look's own instead of a drawn one, approved by whoever gave it."""
    with store.connect() as connection, store.transaction(connection):
        casting = store.casting(connection, casting_id)
        if casting is None:
            raise LookupError(f"no casting {casting_id}")
        if not casting.amended:
            raise ValueError("a member as it stands shows its last look: give it an amendment first")
        _, production = _shot_and_production(connection, casting.shot_id)
        member = store.member(connection, casting.member_id)
        assert member is not None
        role = compiler.DRAWN_ROLE[member.kind]
        _, kept, _, size = _fitted(production, path, role, f"{member.slug}_look{casting.id}")
        store.set_made(
            connection, "casting", casting.id, store.APPROVED, artifact=kept, review="", reviewed_by=store.PERSON,
            size=json.dumps(size), pinned=1, wanted_key=None,
        )  # fmt: skip
        restale_looks(connection, casting.id)
        recompile_look(connection, casting.id)
        return production.id


def _shot_and_production(connection: Any, shot_id: int) -> tuple[store.Shot, store.Production]:
    shot = store.shot(connection, shot_id)
    chapter = None if shot is None else store.chapter(connection, shot.chapter_id)
    production = None if chapter is None else store.production(connection, chapter.production_id)
    if shot is None or production is None:
        raise LookupError(f"no shot {shot_id}")
    return shot, production


def _recast(
    connection: Any, production: store.Production, casting: store.Casting, changes: dict[str, Any] | None
) -> None:
    """One casting changed, or removed (`changes` None); every shot it showed in or shows in goes stale."""
    member = store.member(connection, casting.member_id)
    shot = store.shot(connection, casting.shot_id)
    assert member is not None and shot is not None
    before = looks.resolve(connection, production)
    touched = {shot.id, *before.placed[shot.id][member.slug].carried_into}
    if changes is None:
        restale_looks(connection, casting.id)
        store.delete_casting(connection, casting.id)
        cast = [m for m in shot.plan.get("cast", []) if m != member.slug]
    else:
        fields: dict[str, Any] = {}
        if "parent" in changes:
            fields["parent"] = _parent(before, shot, member.slug, changes["parent"])
        if "amend" in changes:
            fields["amend"] = " ".join(str(changes["amend"] or "").split())
        if "label" in changes:
            fields["label"] = " ".join(str(changes["label"] or "").split())
        if "amend" in fields and fields["amend"] and casting.amend != fields["amend"] and not casting.origin:
            fields["origin"] = "by hand"
        store.update_casting(connection, casting.id, **fields)
        redrawn = any(name in fields and fields[name] != getattr(casting, name) for name in ("amend", "parent"))
        if redrawn:
            store.set_made(connection, "casting", casting.id, store.STALE if casting.artifact else store.MISSING)
            restale_looks(connection, casting.id)
        cast = list(dict.fromkeys([*shot.plan.get("cast", []), member.slug]))
    if shot.plan:
        store.update_shot(connection, shot.id, plan={**shot.plan, "cast": cast}, pinned=True)
    else:
        store.touch_shot(connection, shot.id)
    after = looks.resolve(connection, production)
    if changes is not None:
        touched |= set(after.placed[shot.id][member.slug].carried_into)
    shots = [one for one in _shots_of(connection, production.id) if one.id in touched]
    _stale(connection, shots, prompts=False)
    for one in shots:
        chapter = store.chapter(connection, one.chapter_id)
        if chapter is not None and one.plan.get("beats"):
            recompile(connection, production, chapter, one.id, written_for="looks")


def _parent(resolved: looks.Resolved, shot: store.Shot, member: str, given: Any) -> str | None:
    if given in (None, ""):
        return None
    if str(given) == store.BASE:
        return store.BASE
    look = resolved.looks.get(int(given)) if str(given).isdigit() else None
    if look is None or look.member != member or look.casting is None:
        raise ValueError(f"{given} is not a look of {member}")
    if resolved.order.index(look.casting.shot_id) >= resolved.order.index(shot.id):
        raise ValueError(f"“{look.label}” comes after this shot: a look starts from one before it")
    return str(look.casting.id)


def restale_looks(connection: Any, casting_id: int) -> None:
    """Every look drawn, at any remove, from this one's picture."""
    casting = store.casting(connection, casting_id)
    if casting is None:
        return
    _, production = _shot_and_production(connection, casting.shot_id)
    resolved = looks.resolve(connection, production)
    for look in resolved.looks.values():
        ancestor, descends = look.source, False
        while ancestor is not None and not descends:
            descends = ancestor == casting_id
            ancestor = resolved.looks[ancestor].source if ancestor in resolved.looks else None
        if descends and look.casting is not None and look.casting.artifact and look.casting.state != store.STALE:
            store.set_made(connection, "casting", look.casting.id, store.STALE)


def recompile_look(connection: Any, casting_id: int) -> None:
    """Every prompt of a shot that shows this look, compiled again: its picture or its state moved."""
    casting = store.casting(connection, casting_id)
    if casting is None:
        return
    _, production = _shot_and_production(connection, casting.shot_id)
    resolved = looks.resolve(connection, production)
    for chapter in store.chapters_of(connection, production.id):
        for shot in store.shots_of(connection, chapter.id):
            placed = resolved.placed.get(shot.id, {}).values()
            if shot.plan.get("beats") and any(casting_id in (p.source, p.look) for p in placed):
                recompile(connection, production, chapter, shot.id, written_for="looks")


def rewrite(shot_id: int, asked_by: str = "person") -> str:
    """The shot's prose and prompt written again as a run, a prompt written by hand let go."""
    with store.connect() as connection, store.transaction(connection):
        shot, production = _shot_and_production(connection, shot_id)
        if not shot.plan.get("beats"):
            raise ValueError("plan the chapter first: a rewrite writes the prose of a planned shot")
        store.release_prompt(connection, shot_id)
        run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        store.add_run(
            connection, run_id, title=f"{production.title} · rewrite shot {shot.position + 1}", asked_by=asked_by,
            scope=f"plan:{shot.chapter_id}", production_id=production.id,
            spec={"production": production.id, "chapter": shot.chapter_id, "items": [f"shot {shot.position + 1}"]},
        )  # fmt: skip
        common = {"production": production.id, "chapter": shot.chapter_id, "position": shot.position}
        written = store.add_job(
            connection, run_id, "write_shot", {**common, "take": time.time_ns() // 1_000_000 % 1_000_000_000},
            label=f"write shot {shot.position + 1}", settings=settings.for_job("write_shot", production.id),
        )  # fmt: skip
        store.add_job(
            connection, run_id, "compile_shot", {**common, "written": [{"from_job": written.id, "output": "summary"}]},
            label=f"compile shot {shot.position + 1}", settings=settings.for_job("compile_shot", production.id),
        )  # fmt: skip
        return run_id


# --- a person's verdict ------------------------------------------------------------------------
def review(kind: str, made_id: int, approved: bool) -> int:
    """Approved as it is, or rejected so the next run makes it anew. Returns the production."""
    if kind not in _REJECT:
        raise LookupError(f"{kind!r} is not something a person approves (one of {', '.join(_REJECT)})")
    with store.connect() as connection, store.transaction(connection):
        thing = store.made_by_id(connection, kind, made_id)
        if thing is None:
            raise LookupError(f"no {kind} {made_id}")
        owner = _OWNER[kind](connection, thing)
        if not approved:
            _REJECT[kind](connection, thing)
        else:
            _approve(connection, kind, thing)
        if isinstance(thing, store.Casting):
            recompile_look(connection, thing.id)
        return owner


def _approve(connection: Any, kind: str, thing: store.Made) -> None:
    made = thing.artifact or (isinstance(thing, store.Prompt) and thing.text.strip())
    if not made or thing.state not in (store.MADE, store.NEEDS_REVIEW, store.APPROVED, store.PINNED):
        raise ValueError(f"the {kind} is {thing.state.replace('_', ' ')}: approve it once it is made")
    store.set_made(connection, kind, thing.id, store.APPROVED, review=thing.review, reviewed_by=store.PERSON)
    if isinstance(thing, store.Clip):
        shot = store.shot(connection, thing.shot_id)
        assert shot is not None
        _tell_paused(connection, shot.chapter_id, "resume")


def approve_shot(shot_id: int) -> tuple[int, list[str]]:
    """The shot's prompt, keyframe and looks approved, each that is made; the production and what was not."""
    with store.connect() as connection, store.transaction(connection):
        shot = store.shot(connection, shot_id)
        if shot is None:
            raise LookupError(f"no shot {shot_id}")
        owner = _shot_owner(connection, store.prompt_of(connection, shot_id))
        things: list[tuple[str, store.Made]] = [
            ("prompt", store.prompt_of(connection, shot_id)),
            ("keyframe", store.keyframe_of(connection, shot_id)),
        ]
        things += [("casting", c) for c in store.castings_of(connection, shot_id) if c.amended]
        left = []
        for kind, thing in things:
            if isinstance(thing, store.Keyframe) and thing.source in (store.CHAINED, store.REFS):
                continue
            if thing.state == store.APPROVED and thing.reviewed_by == store.PERSON:
                continue
            try:
                _approve(connection, kind, thing)
            except ValueError as problem:
                left.append(str(problem))
                continue
            if isinstance(thing, store.Casting):
                recompile_look(connection, thing.id)
        return owner, left


def _shot_owner(connection: Any, thing: store.Made) -> int:
    assert isinstance(thing, store.Keyframe | store.Prompt | store.Clip)
    shot = store.shot(connection, thing.shot_id)
    chapter = None if shot is None else store.chapter(connection, shot.chapter_id)
    assert chapter is not None
    return chapter.production_id


def _casting_owner(connection: Any, thing: store.Made) -> int:
    assert isinstance(thing, store.Casting)
    shot = store.shot(connection, thing.shot_id)
    chapter = None if shot is None else store.chapter(connection, shot.chapter_id)
    assert chapter is not None
    return chapter.production_id


def _reject_casting(connection: Any, thing: store.Made) -> None:
    assert isinstance(thing, store.Casting)
    if not thing.amended:
        raise ValueError("a member as it stands has no picture of its own to reject")
    store.set_made(connection, "casting", thing.id, store.STALE, review=thing.review, reviewed_by=store.PERSON)


def _reference_owner(connection: Any, thing: store.Made) -> int:
    assert isinstance(thing, store.Reference)
    member = store.member(connection, thing.member_id)
    assert member is not None
    return member.production_id


def _reject_keyframe(connection: Any, thing: store.Made) -> None:
    assert isinstance(thing, store.Keyframe)
    if thing.source != store.COMPOSED:
        raise ValueError(
            "only a composed keyframe is made again: give another picture, or compose this one"
            if thing.source == store.PROVIDED
            else "a chained keyframe is the last frame of the shot before: reject that clip"
        )
    store.compose_keyframe(connection, thing.shot_id, seed=random.randrange(2**32), note=thing.note)
    store.set_made(connection, "keyframe", thing.id, store.MISSING, review=thing.review, reviewed_by=store.PERSON)
    shot = store.shot(connection, thing.shot_id)
    assert shot is not None
    _stale(connection, [shot], prompts=False)


def _reject_prompt(connection: Any, thing: store.Made) -> None:
    store.set_made(connection, "prompt", thing.id, store.STALE, review=thing.review, reviewed_by=store.PERSON)


def _reject_clip(connection: Any, thing: store.Made) -> None:
    assert isinstance(thing, store.Clip)
    shot = store.shot(connection, thing.shot_id)
    assert shot is not None
    _reroll(connection, shot, random.randrange(2**32), by=store.PERSON)


def _reject_reference(connection: Any, thing: store.Made) -> None:
    assert isinstance(thing, store.Reference)
    if not thing.generated:
        raise ValueError("a picture cut from a photograph is detached, not drawn again")
    store.set_made(
        connection, "reference", thing.id, store.STALE, review=thing.review, reviewed_by=store.PERSON,
        seed=random.randrange(2**32),
    )  # fmt: skip


# Per kind a person can judge: the production it belongs to, and what rejecting it does.
_OWNER = {
    "keyframe": _shot_owner,
    "prompt": _shot_owner,
    "clip": _shot_owner,
    "reference": _reference_owner,
    "casting": _casting_owner,
}
_REJECT = {
    "keyframe": _reject_keyframe,
    "prompt": _reject_prompt,
    "clip": _reject_clip,
    "reference": _reject_reference,
    "casting": _reject_casting,
}


def production_of_casting(casting_id: int) -> int | None:
    with store.connect() as connection:
        casting = store.casting(connection, casting_id)
    return None if casting is None else production_of_shot(casting.shot_id)


def undrawn_looks(shot_id: int) -> list[int]:
    """The shot's own looks with no picture to show: never drawn, failed, or stale."""
    with store.connect() as connection:
        return _undrawn(connection, shot_id)


def undrawn_looks_of(production_id: int, chapter_position: int | None = None) -> list[int]:
    """Every such look of the production, or of the chapter at `chapter_position`."""
    with store.connect() as connection:
        chapters = [c for c in store.chapters_of(connection, production_id) if chapter_position in (None, c.position)]
        return [
            c
            for chapter in chapters
            for s in store.shots_of(connection, chapter.id)
            for c in _undrawn(connection, s.id)
        ]


def _undrawn(connection: Any, shot_id: int) -> list[int]:
    wanted = (store.MISSING, store.FAILED, store.STALE)
    return [c.id for c in store.castings_of(connection, shot_id) if c.amended and c.state in wanted]


def production_of_shot(shot_id: int) -> int | None:
    with store.connect() as connection:
        shot = store.shot(connection, shot_id)
        chapter = None if shot is None else store.chapter(connection, shot.chapter_id)
        return None if chapter is None else chapter.production_id


def delete(production_id: int) -> None:
    with store.connect() as connection, store.transaction(connection):
        store.delete_production(connection, production_id)


def listing() -> list[dict[str, Any]]:
    with store.connect() as connection:
        found = store.productions(connection)
        out = []
        for production in found:
            chapters = store.chapters_of(connection, production.id)
            shots = sum(len(store.shots_of(connection, chapter.id)) for chapter in chapters)
            out.append({**_production(production), "chapters": len(chapters), "shots": shots})
        return out


def describe(production_id: int) -> dict[str, Any] | None:
    """The production whole: chapters, their shots, and each shot's made things as they stand."""
    from difflab.screenplay.ideas import rounds  # ideas adopts a story through this module

    with store.connect() as connection:
        production = store.production(connection, production_id)
        if production is None:
            return None
        resolved = looks.resolve(connection, production)
        cast = {m.id: m for m in store.members_of(connection, production.id)}
        bases = {m.slug: looks.base_picture(connection, m) for m in cast.values()}
        chapters = []
        for chapter in store.chapters_of(connection, production.id):
            shots = [
                _shot(connection, chapter, shot, resolved, cast, bases)
                for shot in store.shots_of(connection, chapter.id)
            ]
            chapters.append({"id": chapter.id, "position": chapter.position, "title": chapter.title, "shots": shots})
        runs = [
            {"id": run.id, "title": run.title, "state": run.state, "scope": run.scope, "created": run.created_at}
            for run in store.runs(connection)
            if run.production_id == production.id
        ]
        editor = drawn.editor_for(production.id)
        members = [_member(connection, production, member, editor, resolved, bases) for member in cast.values()]
        return {
            **_production(production),
            "members": members,
            "missing_references": [m.slug for m in drawn.missing(connection, production)],
            "ideas": rounds(connection, production.id),
            "chapters": chapters,
            "runs": runs,
            "camera_moves": list(words.CAMERA_MOVES),
            "trait_parts": list(knobs.DIRECTOR_TRAIT_ORDER),
            "hashes": _hashes(connection, production, chapters, members),
            "timings": timings.measured(connection),
        }


def timeline(production_id: int, chapter_position: int | None = None) -> dict[str, Any] | None:
    """Every shot of one chapter or all, light, and each member's looks across them as runs, marks and gaps."""
    with store.connect() as connection:
        production = store.production(connection, production_id)
        if production is None:
            return None
        resolved = looks.resolve(connection, production)
        members = store.members_of(connection, production.id)
        made = {
            kind: store.made_of_production(connection, kind, production.id) for kind in ("keyframe", "prompt", "clip")
        }
        shots: list[dict[str, Any]] = []
        for chapter in store.chapters_of(connection, production.id):
            if chapter_position is not None and chapter.position != chapter_position:
                continue
            for shot in store.shots_of(connection, chapter.id):
                keyframe, prompt, clip = (made[kind][shot.id] for kind in ("keyframe", "prompt", "clip"))
                assert isinstance(keyframe, store.Keyframe)
                shots.append(
                    {
                        "id": shot.id,
                        "chapter": chapter.position,
                        "position": shot.position,
                        "scene": shot.scene,
                        "seconds": shot.seconds,
                        "opening": opening_of(keyframe),
                        "pipe": [_dot(keyframe), _dot(prompt), _dot(clip)],
                        "waiting": len(resolved.waiting(shot.id)),
                        "clip": clip.artifact if clip.present else None,
                        "still": keyframe.artifact if keyframe.present else None,
                        "action": shot.action[:TIMELINE_ACTION],
                    }
                )
        ids = [s["id"] for s in shots]
        tracks = [track for member in members if (track := _track(resolved, member, ids))]
        return {"shots": shots, "tracks": tracks}


TIMELINE_ACTION = 160  # characters of a shot's action a timeline block's tooltip carries


def _dot(made: store.Made) -> str:
    """done | review | stale | pending: a pipe dot under a shot block."""
    if made.state in (store.MADE, store.APPROVED, store.PINNED):
        return "done"
    if made.state in (store.NEEDS_REVIEW, store.FAILED):
        return "review"
    return "stale" if made.state == store.STALE else "pending"


def _track(resolved: looks.Resolved, member: store.Member, ids: list[int]) -> dict[str, Any] | None:
    """A member's runs by index into `ids`, split where its look changes, reverts or goes off screen."""
    placed = [resolved.placed[shot_id].get(member.slug) for shot_id in ids]
    at = [i for i, one in enumerate(placed) if one is not None]
    if not at:
        return None
    runs: list[dict[str, Any]] = []
    marks: list[dict[str, Any]] = []
    for i in range(at[0], at[-1] + 1):
        one = placed[i]
        if one is None:
            if runs and runs[-1]["gap"]:
                runs[-1]["b"] = i
            else:
                runs.append({"a": i, "b": i, "gap": True})
            continue
        look = resolved.look(member.slug, one.look)
        if one.casting.amended:
            marks.append(
                {"at": i, "look": look.id, "label": look.label, "image": _image(look), "words": one.casting.amend}
            )
        last = runs[-1] if runs else None
        if last is None or last["gap"] or last["look"] != look.id or one.casting.amended or one.revert:
            runs.append(
                {"a": i, "b": i, "gap": False, "look": look.id, "label": look.label, "image": _image(look),
                 "revert": one.revert, "mark": one.casting.amended}
            )  # fmt: skip
        else:
            last["b"] = i
    return {"member": member.slug, "kind": member.kind, "runs": runs, "marks": marks}


def _hashes(
    connection: Any, production: store.Production, chapters: list[dict[str, Any]], members: list[dict[str, Any]]
) -> dict[str, str]:
    """A short digest per stage of what it holds: it moves when the stage does."""
    # An upload's name begins with its content hash, so the photos' paths stand for their pixels.
    shots = [shot for chapter in chapters for shot in chapter["shots"]]
    stories = [chapter.story for chapter in store.chapters_of(connection, production.id)]
    newest = {
        (scope, tool): video
        for scope, tool, video in connection.execute(
            "SELECT runs.scope, jobs.tool, json_extract(jobs.outputs, '$.video') FROM jobs "
            "JOIN runs ON runs.id = jobs.run_id WHERE runs.production_id = ? AND runs.scope LIKE 'chapter:%' "
            "AND jobs.tool IN ('assemble', 'finish') AND jobs.state IN (?, ?) ORDER BY jobs.id",
            (production.id, store.DONE, store.SKIPPED),
        )
    }
    return {
        "ingest": _digest(production.story_kind, production.story, stories, production.manifest.get("photos", [])),
        "bible": _digest(
            [[{k: m[k] for k in _BIBLE}, [r["key"] or r["artifact"] for r in m["references"]]] for m in members]
        ),
        "shots": _digest(
            [
                [s[k] for k in _SHOTS]
                + [s["prompt"]["text"], s["keyframe"]["key"]]
                + [[c["member"], c["parent"], c["amend"], c["key"]] for c in s["castings"]]
                for s in shots
            ]
        ),
        "render": _digest([s["clip"]["key"] for s in shots]),
        "assemble": _digest(sorted(newest.items())),
    }


_BIBLE = ("slug", "kind", "noun", "possessive", "find", "identity")
_SHOTS = ("scene", "seconds", "plan")


def _digest(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:7]


def _production(production: store.Production) -> dict[str, Any]:
    return {
        "id": production.id,
        "slug": production.slug,
        "title": production.title,
        "story_kind": production.story_kind,
        "story": production.story,
        "manifest": production.manifest,
        "created": production.created_at,
        "updated": production.updated_at,
    }


def _overrides(given: Settings) -> dict[str, Any]:
    """Only what differs from the code defaults: a saved profile should say what was chosen."""
    defaults = type(given)().data()
    return {name: value for name, value in given.data().items() if value != defaults[name]}


def _member(
    connection: Any,
    production: store.Production,
    member: store.Member,
    editor: EditSettings,
    resolved: looks.Resolved,
    bases: dict[str, store.Reference | None],
) -> dict[str, Any]:
    def reference(ref: store.Reference) -> dict[str, Any]:
        made = _made(ref)
        if ref.generated:
            made |= {"state": drawn.state(ref, member, production.manifest, editor), "seed": ref.seed}
        return {**made, "role": ref.role, "photo": ref.photo, "box": ref.box, "size": ref.size, "shows": ref.shows}

    return {
        "id": member.id,
        "slug": member.slug,
        "kind": member.kind,
        "noun": member.noun,
        "possessive": member.possessive,
        "find": member.find,
        "identity": member.identity,
        "position": member.position,
        "pinned": member.pinned,
        "references": [reference(ref) for ref in store.refs_of(connection, member.id)],
        "variants": drawn.drawn_variants(connection, member.id),
        "gaps": gaps(production, member, store.refs_of(connection, member.id)),
        "conflicts": [
            {"id": conflict.id, "says": conflict.novel_says, "resolution": conflict.resolution}
            for conflict in store.conflicts_of(connection, member.id)
        ],
        "looks": [
            {
                **_look(resolved, look, bases),
                "amend": look.casting.amend,
                "origin": look.casting.origin,
                "source": _look(resolved, resolved.look(member.slug, look.source), bases),
                "carried_into": list(resolved.placed[look.casting.shot_id][member.slug].carried_into),
                "made": _made(look.casting),
            }
            for look in resolved.looks.values()
            if look.member == member.slug and look.casting is not None
        ],
        "off_screen": [list(gap) for gap in resolved.gaps.get(member.slug, ())],
    }


def _look(resolved: looks.Resolved, look: looks.Look, bases: dict[str, store.Reference | None]) -> dict[str, Any]:
    """A look as a page names it: `id` None is the bible's, whose picture is the member's base one."""
    base = bases.get(look.member)
    return {
        "id": look.id,
        "member": look.member,
        "label": look.label,
        "words": look.text,
        "shot": None if look.casting is None else look.casting.shot_id,
        "image": _image(look),
        "artifact": look.casting.artifact if look.casting is not None else base.artifact if base else None,
    }


def _image(look: looks.Look) -> str:
    """done | flag | running | pending | failed, as the design's state chips say them."""
    if look.casting is None or look.approved:
        return "done"
    return {
        store.NEEDS_REVIEW: "flag", store.QUEUED: "running", store.RUNNING: "running", store.FAILED: "failed",
    }.get(look.casting.state, "pending")  # fmt: skip


def _castings(
    connection: Any, shot: store.Shot, resolved: looks.Resolved, cast: dict[int, store.Member],
    bases: dict[str, store.Reference | None],
) -> list[dict[str, Any]]:  # fmt: skip
    out = []
    for casting in store.castings_of(connection, shot.id):
        slug = cast[casting.member_id].slug
        placed = resolved.placed[shot.id][slug]
        out.append(
            {
                **_made(casting),
                "member": slug,
                "member_id": casting.member_id,
                "kind": cast[casting.member_id].kind,
                "position": casting.position,
                "parent": casting.parent,
                "amend": casting.amend,
                "label": casting.label,
                "origin": casting.origin,
                "size": casting.size,
                "amended": casting.amended,
                "source": _look(resolved, resolved.look(slug, placed.source), bases),
                "look": _look(resolved, resolved.look(slug, placed.look), bases),
                "latest": _look(resolved, resolved.look(slug, placed.latest), bases),
                "is_revert": placed.revert,
                "back": placed.back,
                "carried_into": list(placed.carried_into),
            }
        )
    return out


def _marker(prompt: store.Prompt, shot: store.Shot) -> dict[str, Any] | None:
    """Kept by hand, kept by hand while the shot moved, recompiled for its looks, or compiled N ago."""
    at = prompt.written_at
    if prompt.pinned:
        # One PATCH can write the prompt and the shot's own columns a moment apart.
        changed = at is not None and shot.updated_at > at + 1.0
        label = "hand-edited · shot changed since" if changed else "hand-edited · pinned"
        return {"kind": "changed" if changed else "pinned", "label": label, "at": at}
    if at is None or not prompt.text:
        return None
    if prompt.written_for == "looks":
        return {"kind": "rewritten", "label": "rewritten from changed states", "at": at}
    return {"kind": "written", "label": f"written {_ago(time.time() - at)} ago", "at": at}


def _ago(seconds: float) -> str:
    for size, unit in ((86400, "d"), (3600, "h"), (60, "min")):
        if seconds >= size:
            return f"{int(seconds // size)} {unit}"
    return f"{int(seconds)} s"


def _shot(
    connection: Any,
    chapter: store.Chapter,
    shot: store.Shot,
    resolved: looks.Resolved,
    cast: dict[int, store.Member],
    bases: dict[str, store.Reference | None],
) -> dict[str, Any]:
    keyframe = store.keyframe_of(connection, shot.id)
    prompt = store.prompt_of(connection, shot.id)
    clip = store.clip_of(connection, shot.id)
    return {
        "id": shot.id,
        "chapter": chapter.position,
        "position": shot.position,
        "action": shot.action,
        "scene": shot.scene,
        "camera": shot.camera,
        "dialogue": shot.dialogue,
        "seconds": shot.seconds,
        "chained": shot.chained,
        "pinned": shot.pinned,
        "opening": opening_of(keyframe),
        "sound": str(shot.plan.get("sounds", "")),
        "castings": _castings(connection, shot, resolved, cast, bases),
        "waiting": [look.id for look in resolved.waiting(shot.id)],
        "keyframe": {
            **_made(keyframe),
            "source": keyframe.source,
            "provided": keyframe.provided,
            "note": keyframe.note,
            "landed": keyframe.landed,
            "made_from": _made_from(connection, keyframe),
        },
        "prompt": {
            **_made(prompt),
            "text": prompt.text,
            "written_by": prompt.written_by,
            "pictures": prompt.pictures,
            "checks": prompt.checks,
            "marker": _marker(prompt, shot),
        },
        "plan": shot.plan,
        "clip": {**_made(clip), "seed": clip.seed, "handoff": clip.handoff},
    }


def _made_from(connection: Any, keyframe: store.Keyframe) -> dict[str, Any] | None:
    """What a composed keyframe was made over and handed: the place's picture and the cast's, by id."""
    job = store.job(connection, keyframe.current_job) if keyframe.current_job else None
    if job is None or job.tool != "compose_keyframe":
        return None
    return {
        "base": job.inputs.get("base"),
        "pictures": job.inputs.get("pictures") or {},
        "note": job.inputs.get("note"),
    }


def _made(made: store.Made) -> dict[str, Any]:
    return {
        "id": made.id,
        "state": made.state,
        "artifact": made.artifact,
        "present": made.present,
        "key": made.wanted_key,
        "job": made.current_job,
        "pinned": made.pinned,
        "review": made.review,
        "reviewed_by": made.reviewed_by,
        "reviewed_at": made.reviewed_at,
    }


def _under_output(path: str | Path) -> Path:
    given = Path(path)
    return given if given.is_absolute() else config.OUTPUT_DIR / given
