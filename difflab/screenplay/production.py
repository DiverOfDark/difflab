"""A production and its shots, as the page and the playground make them. Every call is one transaction."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from difflab import config, store
from difflab.core import media
from difflab.core.keys import key
from difflab.core.settings import Continuity, EditSettings, Settings
from difflab.models.minimax_h3 import pipeline
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import compiler, crops, drawn, settings, words
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
    "" to chain again), `landed` (whether the shot before ends on that keyframe), `seed`, the plan's
    `closing`, `cast`, `camera_move`, `camera_toward` and `dialogue`, and the shot's own columns. A plan
    edit pins the shot and recompiles its prompt at once, and a closing the next shot's too, since it
    opens on it; a keyframe change recompiles the shot and the one before, since whether that one lands
    on it and whether a tail crosses are both in their words."""
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


PLAN_FIELDS = ("actions", "closing", "cast", "camera_move", "camera_toward", "dialogue")


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
    move, toward = plan.get("camera_move", ""), plan.get("camera_toward", "")
    dialogue = " / ".join(f"{line['who']}: {line['line']}" for line in plan.get("dialogue", []))
    store.update_shot(connection, shot.id, plan=plan, camera=f"{move} → {toward}", dialogue=dialogue, pinned=True)
    recompile(connection, production, chapter, shot.id)
    if "closing" in planned and following is not None:
        recompile(connection, production, chapter, following.id)  # it opens on this closing


def _said(plan: dict[str, Any]) -> str:
    beats = plan.get("beats", [])
    parts = [str(b.get(k, "")) for b in beats for k in ("action", "prose")]
    parts += ["{" + str(who) + "}" for b in beats for who in b.get("who", [])]
    return " ".join([*parts, *(str(plan.get(k, "")) for k in ("closing", "summary", "sounds"))])


def recompile(connection: Any, production: store.Production, chapter: store.Chapter, shot_id: int) -> None:
    """The prompt compiled again from the shot's plan as it now stands - code, no model - as
    `compile_shot` would; a plan it cannot compile leaves the prompt stale for a re-plan to write."""
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
        store.set_compiled(connection, shot_id, compiled.prompt, handed, prompt.checks, str(made))
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
        kept = folder / f"{source.stem}_{member.slug}_{role}.png"
        frame.crop(box.pixels()).resized(*size).pil().save(kept)
        given = store.relative(source)
        store.add_ref(
            connection, member.id, role, given, list(box.pixels()), list(size), (size[0] // 32) * (size[1] // 32),
            [member.slug], box.area, artifact=store.relative(kept), provided=given,
        )  # fmt: skip
        return _stale(connection, _shots_of(connection, production.id, member.slug))


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


def reroll(shot_id: int, seed: int | None = None) -> int:
    """One more attempt at a shot's clip: a new seed, and the clip goes stale. The shots chained after
    it go stale by key when the next run reaches them. Returns the seed."""
    chosen = seed if seed is not None else random.randrange(2**32)
    with store.connect() as connection, store.transaction(connection):
        clip = store.set_clip_seed(connection, shot_id, chosen)
        store.set_made(connection, "clip", clip.id, store.STALE if clip.artifact else store.MISSING)
    return chosen


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
        chapters = []
        for chapter in store.chapters_of(connection, production.id):
            shots = [_shot(connection, shot) for shot in store.shots_of(connection, chapter.id)]
            chapters.append({"id": chapter.id, "position": chapter.position, "title": chapter.title, "shots": shots})
        runs = [
            {"id": run.id, "title": run.title, "state": run.state, "scope": run.scope, "created": run.created_at}
            for run in store.runs(connection)
            if run.production_id == production.id
        ]
        editor = drawn.editor_for(production.id)
        members = [
            _member(connection, production, member, editor) for member in store.members_of(connection, production.id)
        ]
        return {
            **_production(production),
            "members": members,
            "missing_references": [m.slug for m in drawn.missing(connection, production)],
            "ideas": rounds(connection, production.id),
            "chapters": chapters,
            "runs": runs,
            "camera_moves": list(words.CAMERA_MOVES),
            "trait_parts": list(knobs.DIRECTOR_TRAIT_ORDER),
        }


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
    connection: Any, production: store.Production, member: store.Member, editor: EditSettings
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
    }


def _shot(connection: Any, shot: store.Shot) -> dict[str, Any]:
    keyframe = store.keyframe_of(connection, shot.id)
    prompt = store.prompt_of(connection, shot.id)
    clip = store.clip_of(connection, shot.id)
    return {
        "id": shot.id,
        "position": shot.position,
        "action": shot.action,
        "scene": shot.scene,
        "camera": shot.camera,
        "dialogue": shot.dialogue,
        "seconds": shot.seconds,
        "chained": shot.chained,
        "pinned": shot.pinned,
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
    }


def _under_output(path: str | Path) -> Path:
    given = Path(path)
    return given if given.is_absolute() else config.OUTPUT_DIR / given
