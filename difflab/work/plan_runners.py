"""What each stage of a plan run does: the director's tools over the store's rows.

A stage job keys itself by everything it reads - the production, its inputs, the model, every
prompt's words and every knob - and a finished job under the same key is reused whole, rows included:
a plan run asked again with nothing changed writes nothing and asks the VLM nothing. The production
is part of every key on purpose: two productions planned from the same photos are two sets of rows.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from difflab import config, store
from difflab.core import context, media, protocol, ui
from difflab.core.cache import cache
from difflab.core.keys import Key, key
from difflab.core.types import Frame
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import checks, compiler, crops, words
from difflab.screenplay import config as knobs
from difflab.screenplay import production as production_module
from difflab.screenplay import settings as chosen
from difflab.screenplay.production import CUT_NOTE, OPENING_NOTE
from difflab.tools import (
    cards,
    cut_crops,
    direct,
    ground,
    guess,
    opening,
    plan_shots,
    refine_idea,
    ruling,
    survey,
    write_shot,
)  # fmt: skip
from difflab.work import Skipped


def _path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else config.OUTPUT_DIR / path


def _frame(value: Any) -> Frame:
    return media.frame_from_image(_path(value))


def _fingerprint() -> dict[str, Any]:
    """The model, every prompt's words and every knob: a change to any of them is a different plan."""
    # The judge's, the composer's and the idea editor's words are not the planner's: a change to them
    # re-plans nothing.
    others = ("JUDGE_", "COMPOSE_", "IDEA_")
    prompts = {name: value for name, value in vars(words).items() if name.isupper() and not name.startswith(others)}
    settings = {name: value for name, value in vars(knobs).items() if name.startswith("DIRECTOR_")}
    return {"model": direct.MODEL.get(), "prompts": prompts, "knobs": settings}


def _taken(inputs: dict[str, Any]) -> dict[str, int]:
    return {"take": int(inputs["take"])} if inputs.get("take") else {}


@contextlib.contextmanager
def _take(inputs: dict[str, Any]) -> Iterator[None]:
    token = direct.TAKE.set(int(inputs.get("take") or 0))
    try:
        yield
    finally:
        direct.TAKE.reset(token)


def _start(tool: str, made: Key) -> dict[str, Any] | None:
    """Record the running job's key, and hand back an earlier job's outputs under it if there is one."""
    ctx = context.current()
    with store.connect() as connection:
        if ctx.job_id is not None:
            store.set_job_key(connection, ctx.job_id, str(made))
        previous = store.last_done_job(connection, tool, str(made))
    if previous is None or previous.id == ctx.job_id:
        return None
    ui.note(f"{tool}: already done as {made}")
    return Skipped(previous.outputs)


def _members(listed: list[dict[str, Any]]) -> dict[str, direct.Member]:
    return {str(m["id"]): direct.Member(**{k: str(v) for k, v in m.items()}) for m in listed}


def _found(listed: list[dict[str, Any]]) -> list[crops.Found]:
    return [
        crops.Found(
            str(f["label"]),
            crops.Box(*(float(v) for v in f["box"])),
            crops.Box(*(float(v) for v in f["head"])) if f.get("head") else None,
            str(f.get("part", "")),
        )
        for f in listed
    ]


def _found_data(listed: list[crops.Found]) -> list[dict[str, Any]]:
    return [
        {
            "label": f.label,
            "box": list(f.box.pixels()),
            "head": list(f.head.pixels()) if f.head else None,
            "part": f.part,
        }
        for f in listed
    ]


def _shows(listed: list[list[str]]) -> list[tuple[str, ...]]:
    return [tuple(str(m) for m in shown) for shown in listed]


def _card(slug: str, member: direct.Member, identity: dict[str, Any]) -> compiler.Card:
    return compiler.Card(
        slug,
        member.kind,
        member.noun,
        member.possessive,
        tuple(
            compiler.Trait(str(t["part"]), str(t["phrase"]), str(t["short"]), str(t["name"]))
            for t in identity["traits"]
        ),
        tuple(str(light) for light in identity.get("lights", [])),
        bool(identity.get("photographed", True)),
    )


def _identity(card: compiler.Card) -> dict[str, Any]:
    return {
        "traits": [dataclasses.asdict(trait) for trait in card.traits],
        "lights": list(card.lights),
        "photographed": card.photographed,
    }


def _manifest(production_id: int, **fields: Any) -> None:
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        assert production is not None
        store.update_production(connection, production_id, manifest={**production.manifest, **fields})


def _by_hand(production_id: int) -> tuple[list[store.Member], dict[str, Any]]:
    """The members edited by hand, and the production's manifest: what was dropped or detached by hand."""
    with store.connect() as connection:
        production = store.production(connection, production_id)
        assert production is not None
        return [m for m in store.members_of(connection, production_id) if m.pinned], production.manifest


def _overlay(
    members: dict[str, direct.Member], shows: list[tuple[str, ...]], pinned: list[store.Member], dropped: list[str]
) -> tuple[dict[str, direct.Member], list[tuple[str, ...]]]:
    """The survey's cast with the hand's edits over it: a pinned member as written, a dropped one gone,
    and a member decided in words in no photo, so nothing is cut or read of it."""
    out = {slug: member for slug, member in members.items() if slug not in dropped}
    for member in pinned:
        out[member.slug] = direct.Member(member.slug, member.kind, member.noun, member.possessive, member.find)
    unseen = {m.slug for m in pinned if not m.identity.get("photographed", True)} | set(dropped)
    return out, [tuple(m for m in listed if m not in unseen) for listed in shows]


# --- the stages ----------------------------------------------------------------------------------
def run_guess_reference(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    frame = _frame(inputs["path"])
    made = key("guess_reference", 1, photo=frame.pixels, **_fingerprint())
    outputs = _start("guess_reference", made) or {"path": str(inputs["path"]), **guess.guess(frame)}
    if inputs.get("production"):
        _style_words(int(inputs["production"]), str(inputs["path"]), str(outputs.get("look") or ""))
    return outputs


def _style_words(production_id: int, path: str, look: str) -> None:
    """A style member given no words at ingest takes its picture's look as the VLM saw it."""
    with store.connect() as connection, store.transaction(connection):
        for member in store.members_of(connection, production_id, style=True):
            pictured = any(r.photo == path for r in store.refs_of(connection, member.id))
            if member.kind == "style" and pictured and not member.appearance and look:
                store.update_member(connection, member.id, appearance=look)


def run_same_person(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    first, second = _frame(inputs["a"]), _frame(inputs["b"])
    made = key("same_person", 1, a=first.pixels, b=second.pixels, **_fingerprint())
    if (previous := _start("same_person", made)) is not None:
        return previous
    same, why = ruling.same_person(first, second)
    return {"ruled": {"pair": [int(i) for i in inputs["pair"]], "same": same, "why": why}}


def _labelled(inputs: dict[str, Any]) -> tuple[list[survey.Label], list[list[int]], list[dict[str, Any]]]:
    """Each photo's label with the face rulings applied: a photo ruled the same person as a labelled one
    takes its entity, and the others ruled alike are one group. Also each merge, for the page to say."""
    labels = [
        survey.Label(str(given.get("kind") or ""), str(given.get("entity") or ""))
        for given in inputs.get("labels") or []
    ]
    parent = list(range(len(labels)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for ruled in inputs.get("same") or []:
        if ruled and ruled.get("same"):
            a, b = (int(i) for i in ruled["pair"])
            parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for i in range(len(labels)):
        groups.setdefault(root(i), []).append(i)
    merged = []
    for group in groups.values():
        entity = next((labels[i].entity for i in group if labels[i].entity), "")
        for i in group:
            if entity and not labels[i].entity:
                labels[i] = survey.Label("character", entity)
                merged.append({"photo": i, "entity": entity})
    return labels, [g for g in groups.values() if len(g) > 1], merged


def _known(given: Any) -> survey.Surveyed | None:
    if not given:
        return None
    return survey.Surveyed(
        {str(m["id"]): direct.Member(**{k: str(v) for k, v in m.items()}) for m in given["members"]},
        [tuple(str(m) for m in shown) for shown in given["shows"]],
        {str(k): str(v) for k, v in (given.get("aliases") or {}).items()},
    )


def run_survey(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """One chapter read for its cast, told what the chapters before it found; the last one drops the rest."""
    production_id, idea = int(inputs["production"]), str(inputs["idea"])
    frames = [_frame(p) for p in inputs["photos"]]
    pinned, manifest = _by_hand(production_id)
    dropped = [str(m) for m in manifest.get("dropped", [])]
    labels, same, merged = _labelled(inputs)
    known = _known(inputs.get("known"))
    chapter = int(inputs.get("chapter_position") or 0)
    # Only a production edited by hand keys by the edits, so one never edited keeps the key it had.
    hand = [
        {**dataclasses.asdict(direct.Member(m.slug, m.kind, m.noun, m.possessive, m.find)),
         "photographed": m.identity.get("photographed", True)}
        for m in pinned
    ]  # fmt: skip
    edits = {"by_hand": hand, "dropped": dropped} if hand or dropped else {}
    told = {
        **(
            {"labels": [dataclasses.asdict(label) for label in labels]}
            if any(label.kind or label.entity for label in labels)
            else {}
        ),
        **({"same": same} if same else {}),
        **({"known": inputs["known"]} if known else {}),
    }
    made = key(
        "survey", 1, production=production_id, idea=idea, photos=[f.pixels for f in frames], **edits, **told,
        **_fingerprint(),
    )  # fmt: skip
    if (previous := _start("survey", made)) is not None:
        _announce(chapter, previous)
        return previous
    surveyed = survey.survey(idea, frames, known, labels, same)
    members, shows = _overlay(surveyed.members, surveyed.shows, pinned, dropped)
    kept = {m.slug for m in pinned}
    photos = [str(p) for p in inputs["photos"]]
    unmatched = [
        {"photo": photos[i], "about": about, "likely": likely} for i, (about, likely) in surveyed.unplaced.items() if i
    ]
    with store.connect() as connection, store.transaction(connection):
        for position, member in enumerate(members.values()):
            if member.id not in kept:
                store.put_member(
                    connection, production_id, member.id, member.kind, member.noun, member.possessive, member.find,
                    position,
                )  # fmt: skip
        if inputs.get("last", True):
            store.drop_members_except(connection, production_id, list(members))
        for row in store.members_of(connection, production_id):
            if surveyed.aliases.get(row.slug) and not row.aliases:
                store.update_member(connection, row.id, aliases=surveyed.aliases[row.slug])
    _manifest(production_id, unmatched=unmatched)
    listed = [dataclasses.asdict(m) for m in members.values()]
    cast = {"members": listed, "shows": [list(s) for s in shows], "aliases": surveyed.aliases}
    outputs = {
        **cast, "cast": cast, "unmatched": unmatched, "merged": merged,
        "new": [{**dataclasses.asdict(members[m]), "photos": sum(m in s for s in shows)} for m in surveyed.new
                if m in members],
    }  # fmt: skip
    _announce(chapter, outputs)
    return outputs


def _announce(chapter: int, outputs: dict[str, Any]) -> None:
    run = context.current().run
    if run is not None and outputs.get("new"):
        found = [
            {k: str(v) for k, v in m.items() if k in ("id", "kind", "noun", "find", "photos")} for m in outputs["new"]
        ]
        run.emit(protocol.Found(chapter=chapter, members=found))


def skip_survey(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """A chapter not read: the cast as the chapters before it left it. The first one has nothing to pass on."""
    known = inputs.get("known")
    return None if not known else {**known, "cast": known, "unmatched": [], "merged": [], "new": []}


# Per tool, the outputs a job skipped by a person hands on, from its inputs; a tool absent here is not skipped.
SKIPS: dict[str, Callable[[dict[str, Any]], dict[str, Any] | None]] = {
    "survey": skip_survey,
    "guess_reference": lambda inputs: {"path": str(inputs["path"]), "kind": "", "about": "", "look": ""},
    "same_person": lambda inputs: {"ruled": {"pair": list(inputs["pair"]), "same": None, "why": "skipped"}},
}


def run_ground(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production_id, index = int(inputs["production"]), int(inputs["index"])
    members = _members(inputs["members"])
    listed = [members[m] for m in _shows(inputs["shows"])[index] if m in members]
    frame = _frame(inputs["photo"])
    made = key(
        "ground", 1, production=production_id, photo=frame.pixels, members=[dataclasses.asdict(m) for m in listed],
        **_fingerprint(),
    )  # fmt: skip
    if (previous := _start("ground", made)) is not None:
        return previous
    found = ground.ground(frame, listed) if listed else []
    return {"found": _found_data(found)}


def run_cut_crops(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production_id = int(inputs["production"])
    photos = [str(p) for p in inputs["photos"]]
    frames = [_frame(p) for p in photos]
    members = _members(inputs["members"])
    shows = _shows(inputs["shows"])
    found = [_found(listed) for listed in inputs["found"]]
    kinds = {m.id: m.kind for m in members.values()}
    aspect = None if inputs.get("aspect") in (None, "") else float(inputs["aspect"])
    video = VideoSettings(**settings["video"])
    _, manifest = _by_hand(production_id)
    detached = [list(map(str, entry)) for entry in manifest.get("detached", [])]
    made = key(
        "cut_crops", 3, production=production_id, photos=[f.pixels for f in frames], shows=inputs["shows"],
        found=inputs["found"], kinds=kinds, aspect=aspect, video=video,
        **({"detached": detached} if detached else {}), **_fingerprint(),
    )  # fmt: skip
    if (previous := _start("cut_crops", made)) is not None:
        _keep_cut(production_id, photos, previous, str(made))
        return previous
    cut = cut_crops.opening_and_crops(frames, found, shows, kinds, aspect, video)
    cut = dataclasses.replace(cut, crops=[c for c in cut.crops if [c.member, photos[c.photo], c.role] not in detached])
    lost = {(member, role) for member, _, role in detached}
    gaps = crops.gaps(cut.crops, cut.dropped, found, shows, kinds, lost)
    for gap in gaps:
        ui.note(f"{gap.member}: no {gap.role} reference, {gap.why}")
    first = cache.path(made, ".png")
    cut.frame.pil().save(first)
    folder = cache.directory / "crops"
    folder.mkdir(parents=True, exist_ok=True)
    listed = []
    for crop in cut.crops:
        path = folder / f"{made.token}_{crop.key}.png"
        crop.cut(frames[crop.photo]).pil().save(path)
        listed.append({**dataclasses.asdict(crop), "box": list(crop.box.pixels()), "path": store.relative(path)})
    outputs = {
        "opening": store.relative(first),
        "cut": list(cut.cut.pixels()),
        "canvas": {**dataclasses.asdict(cut.canvas), "seconds": cut.canvas.duration},
        "seconds": cut.canvas.duration,
        "crops": listed,
        "gaps": [dataclasses.asdict(gap) for gap in gaps],
    }
    _keep_cut(production_id, photos, outputs, str(made))
    return outputs


def _keep_cut(production_id: int, photos: list[str], cut: dict[str, Any], made: str) -> None:
    """The cut's references and canvas, written again when the cut is reused by key: an aspect changed
    and changed back would otherwise leave the other aspect's crops and canvas in place."""
    canvas = dict(cut["canvas"])
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        cut_refs = [
            r for r in store.refs_of_production(connection, production_id) if r.provided is None and not r.generated
        ]
        current = production is not None and production.manifest.get("canvas") == canvas
        if current and cut_refs and all(r.wanted_key == made for r in cut_refs):
            return
        rows = {member.slug: member.id for member in store.members_of(connection, production_id)}
        store.clear_refs(connection, production_id)
        for crop in cut["crops"]:
            if crop["member"] not in rows:
                continue
            size = [int(v) for v in crop["size"]]
            store.add_ref(
                connection, rows[crop["member"]], str(crop["role"]), photos[int(crop["photo"])],
                [int(v) for v in crop["box"]], size, (size[0] // 32) * (size[1] // 32), [str(m) for m in crop["shows"]],
                float(crop["quality"]), artifact=str(crop["path"]), key=made,
            )  # fmt: skip
    _manifest(
        production_id,
        opening={"photo": photos[0], "box": list(cut["cut"]), "size": [canvas["width"], canvas["height"]]},
        canvas=canvas,
        seconds=float(cut["seconds"]),
        gaps=list(cut.get("gaps", [])),
    )


def run_cards(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production_id, index = int(inputs["production"]), int(inputs["index"])
    members = _members(inputs["members"])
    listed = [members[m] for m in _shows(inputs["shows"])[index] if m in members]
    frame = _frame(inputs["photo"])
    made = key(
        "cards", 1, production=production_id, photo=frame.pixels, members=[dataclasses.asdict(m) for m in listed],
        **_fingerprint(),
    )  # fmt: skip
    if (previous := _start("cards", made)) is not None:
        return previous
    return {"card": cards.identify([frame], [listed])[0]}


def run_describe_opening(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production_id = int(inputs["production"])
    members = _members(inputs["members"])
    shows = _shows(inputs["shows"])
    found = _found(inputs["found"])
    cut = crops.Box(*(float(v) for v in inputs["cut"]))
    card = dict(inputs["card"])
    frame = _frame(inputs["opening"])
    in_opening = [
        members[m]
        for m in shows[0]
        if m in members and members[m].kind != "environment" and cut_crops.inside(found, m, cut)
    ]
    layout = cut_crops.layout(found, in_opening, cut)
    made = key(
        "describe_opening", 1, production=production_id, opening=frame.pixels,
        members=[dataclasses.asdict(m) for m in in_opening], card=card,
        layout=[(m.id, part, where) for m, part, where in layout],
        **_fingerprint(),
    )  # fmt: skip
    if (previous := _start("describe_opening", made)) is not None:
        return previous
    seen = opening.describe_opening(frame, in_opening, card, layout)
    return {"seen": dataclasses.asdict(seen), "in_opening": [m.id for m in in_opening]}


def run_tidy(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production_id, idea = int(inputs["production"]), str(inputs["idea"])
    members = _members(inputs["members"])
    shows = _shows(inputs["shows"])
    seen = [dict(card) for card in inputs["cards"]]
    found = [_found(listed) for listed in inputs["found"]]
    pinned = {m.slug: m.identity for m in _by_hand(production_id)[0] if m.slug in members}
    with store.connect() as connection:
        production = store.production(connection, production_id)
        aliases = {m.slug: m.aliases for m in store.members_of(connection, production_id) if m.aliases}
    made = key(
        "tidy", 1, production=production_id, idea=idea, members=inputs["members"], shows=inputs["shows"], cards=seen,
        found=inputs["found"], **({"by_hand": pinned} if pinned else {}), **({"aliases": aliases} if aliases else {}),
        **_fingerprint(),
    )  # fmt: skip
    if (previous := _start("tidy", made)) is not None:
        return previous
    asked = {slug: member for slug, member in members.items() if slug not in pinned}
    merged, conflicts = cards.merge(idea, asked, seen, found, shows)
    identities = {slug: _identity(card) for slug, card in merged.items()}
    identities |= {
        slug: {"traits": [], "lights": [], "photographed": True, **identity} for slug, identity in pinned.items()
    }
    told = production is not None and production.story_kind in ("idea", "text")
    # A video's story is its captions' change, not a description: nothing in it argues with a photograph.
    texts = {
        slug: ruling.novel(idea, card, aliases.get(slug, ""))
        for slug, card in merged.items()
        if told and card.photographed
    }
    with store.connect() as connection, store.transaction(connection):
        for member in store.members_of(connection, production_id):
            if member.slug in identities and member.slug not in pinned:
                store.set_identity(connection, member.id, identities[member.slug])
                store.set_conflicts(connection, member.id, conflicts.get(member.slug, []), texts.get(member.slug, []))
    return {"cards": identities}


def run_describe_member(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """One member read again from each photograph it was cut from, tidied, and checked against the story."""
    production_id = int(inputs["production"])
    with store.connect() as connection:
        production = store.production(connection, production_id)
        member = store.member(connection, int(inputs["member"]))
        if production is None or member is None:
            raise LookupError(f"member {inputs['member']} was deleted before it was described")
        own = [r for r in store.refs_of(connection, member.id) if r.photo and not r.generated]
        story = next(
            (c.story for c in store.chapters_of(connection, production_id) if c.story.strip()), production.story
        )
    photos = [p for p in production.manifest.get("photos", []) if p in {r.photo for r in own}]
    frames = [_frame(p) for p in photos]
    asked = direct.Member(member.slug, member.kind, member.noun, member.possessive, member.find)
    largest = {photo: max((r for r in own if r.photo == photo), key=lambda r: r.quality) for photo in photos}
    found = [[crops.Found(member.slug, crops.Box(*map(float, largest[photo].box)))] for photo in photos]
    seen = cards.identify(frames, [[asked]] * len(frames))
    merged, conflicts = cards.merge(story, {member.slug: asked}, seen, found, [(member.slug,)] * len(frames))
    card = merged[member.slug]
    told = production.story_kind in ("idea", "text") and card.photographed
    texts = ruling.novel(story, card, member.aliases) if told else []
    with store.connect() as connection, store.transaction(connection):
        store.update_member(connection, member.id, identity=_identity(card), appearance="", pinned=False)
        store.set_conflicts(connection, member.id, conflicts.get(member.slug, []), texts)
        moved = production_module.stale_casting(connection, member.id)
    return {"card": _identity(card), "conflicts": len(conflicts.get(member.slug, [])) + len(texts), "stale": moved}


def run_arrange(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production_id = int(inputs["production"])
    members = _members(inputs["members"])
    identities = {str(k): dict(v) for k, v in inputs["cards"].items()}
    in_opening = [str(m) for m in inputs["in_opening"]]
    seen_data = dict(inputs["seen"])
    seen = direct.Seen(
        description=str(seen_data["description"]),
        layout=str(seen_data["layout"]),
        hands=str(seen_data["hands"]),
        sided=tuple(str(s) for s in seen_data["sided"]),
        mouths=str(seen_data["mouths"]),
    )
    hand = str(_by_hand(production_id)[1].get("opening_words") or "")
    made = key(
        "arrange", 1, production=production_id, seen=seen_data, in_opening=in_opening,
        cards={m: identities[m] for m in in_opening if m in identities},
        **({"by_hand": hand} if hand else {}), **_fingerprint(),
    )  # fmt: skip
    if (previous := _start("arrange", made)) is not None:
        return previous
    listed = [_card(m, members[m], identities[m]) for m in in_opening if m in members and m in identities]
    arrangement = hand or opening.arrange_opening(seen, listed)
    _manifest(production_id, opening_arrangement=arrangement)
    return {"arrangement": arrangement}


def run_plan_shots(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    outputs = _plan_shots(inputs, settings)
    _castings(inputs, outputs)
    _voices(int(inputs["production"]), dict(outputs.get("voices") or {}))
    _write_and_compile(inputs, len(outputs["chunks"]))
    return outputs


def _castings(inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
    """Each unpinned shot's castings from its plan's cast and the planner's lasting changes."""
    if "looks" not in outputs:  # a plan reused from before its looks were an output of their own
        return
    production_id, chapter_id = int(inputs["production"]), int(inputs["chapter"])
    story = [(str(words_), int(clip)) for words_, clip in outputs.get("story", [])]
    with store.connect() as connection, store.transaction(connection):
        chapter = store.chapter(connection, chapter_id)
        assert chapter is not None
        members = {m.slug: m.id for m in store.members_of(connection, production_id)}
        shots = store.shots_of(connection, chapter_id)
        said: dict[str, str] = {}
        for shot, plan, changed in zip(shots, outputs["chunks"], outputs["looks"], strict=False):
            if shot.pinned:
                continue
            existing = {c.member_id: c for c in store.castings_of(connection, shot.id)}
            cast = [str(m) for m in plan.get("cast", []) if str(m) in members]
            for position, slug in enumerate(cast):
                amend = " ".join(str((changed.get(slug) or {}).get("amend") or "").split())
                label = " ".join(str((changed.get(slug) or {}).get("label") or "").split())
                # Said again word for word in a later clip, a change is that clip carrying it, not a new look.
                amend = "" if amend and amend == said.get(slug) else amend
                said[slug] = amend or said.get(slug, "")
                where = _paragraph(str(inputs["idea"]), story, shot.position + 1)
                origin = f"from text · ch {chapter.position + 1}" + (f" ¶{where}" if where else "") if amend else ""
                found = existing.pop(members[slug], None)
                if found is None:
                    found = store.add_casting(connection, shot.id, members[slug], amend=amend, origin=origin)
                elif found.amend != amend or found.parent is not None:
                    store.set_made(connection, "casting", found.id, store.STALE if found.artifact else store.MISSING)
                    store.update_casting(connection, found.id, parent=None, amend=amend)
                store.update_casting(
                    connection, found.id, position=position, origin=origin, label=label if amend else ""
                )
            for gone in existing.values():
                store.delete_casting(connection, gone.id)


def _voices(production_id: int, voices: dict[str, str]) -> None:
    """The planner's voice for each speaker the bible has none for: a voice once set is the member's."""
    with store.connect() as connection, store.transaction(connection):
        for member in store.members_of(connection, production_id):
            if not member.voice and voices.get(member.slug):
                store.update_member(connection, member.id, voice=str(voices[member.slug]))


def _paragraph(idea: str, story: list[tuple[str, int]], clip: int) -> int | None:
    """The idea's paragraph a clip's first event comes from: its number when the idea numbers its lines."""
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", idea) if p.strip()]
    lines = [line.strip() for line in idea.splitlines()]
    numbered = {int(m[1]): m[2] for line in lines if (m := re.match(r"(\d+)[.)]\s+(.*)", line))}
    for event, at in story:
        wanted = event.lower().strip(" .")
        if at != clip or not wanted:
            continue
        for number, text in numbered.items():
            if wanted in text.lower():
                return number
        for number, text in enumerate(paragraphs, 1):
            if wanted in text.lower():
                return number
    return None


def _write_and_compile(inputs: dict[str, Any], count: int) -> None:
    """The run's write and compile jobs, one each per shot the plan chose: the run could not know how
    many when it was queued. A plan job reused by key still adds them, since its run has none."""
    ctx = context.current()
    production_id, chapter_id = int(inputs["production"]), int(inputs["chapter"])
    with store.connect() as connection, store.transaction(connection):
        if ctx.run_id is None or ctx.job_id is None:
            return
        if any(job.tool == "write_shot" for job in store.jobs_of(connection, ctx.run_id)):
            return
        common = {"production": production_id, "chapter": chapter_id}
        written = [
            store.add_job(
                connection, ctx.run_id, "write_shot",
                {**common, "position": i, "planned": {"from_job": ctx.job_id, "output": "chunks"}, **_taken(inputs)},
                label=f"write shot {i + 1}", settings=chosen.for_job("write_shot", production_id),
            ).id
            for i in range(count)
        ]  # fmt: skip
        for i in range(count):
            store.add_job(
                connection, ctx.run_id, "compile_shot",
                {**common, "position": i, "written": [{"from_job": job, "output": "summary"} for job in written]},
                label=f"compile shot {i + 1}", settings=chosen.for_job("compile_shot", production_id),
            )  # fmt: skip


def _plan_shots(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production_id, chapter_id = int(inputs["production"]), int(inputs["chapter"])
    idea, total = str(inputs["idea"]), float(inputs["take_seconds"])
    members = _members(inputs["members"])
    identities = {str(k): dict(v) for k, v in inputs["cards"].items()}
    arrangement = str(inputs["arrangement"])
    seconds, tail, budget = float(inputs["seconds"]), bool(inputs["tail"]), int(inputs["budget"])
    shortest = float(inputs.get("shortest") or knobs.DIRECTOR_CLIP_MIN)
    video = VideoSettings(**settings["video"])
    with store.connect() as connection:
        production = store.production(connection, production_id)
        assert production is not None
        references = store.refs_of_production(connection, production_id)
        rows = store.members_of(connection, production_id)
    photos = [str(p) for p in production.manifest.get("photos", [])]
    refs = compiler.refs_of(references, rows, photos)
    aliases = {m.slug: m.aliases for m in rows if m.aliases.strip() and m.slug in members}
    made = key(
        "plan_shots", 2, production=production_id, chapter=chapter_id, idea=idea, take_seconds=total,
        members=inputs["members"],
        cards=identities, refs={k: dataclasses.asdict(r) for k, r in refs.items()}, arrangement=arrangement,
        seconds=seconds, tail=tail, budget=budget, **({"aliases": aliases} if aliases else {}), **_taken(inputs),
        **({"shortest": shortest} if shortest != knobs.DIRECTOR_CLIP_MIN else {}),
        **_fingerprint(),
    )  # fmt: skip
    if (previous := _start("plan_shots", made)) is not None:
        return previous
    base = compiler.Take(
        title=production.title,
        members=tuple(members),
        cards={slug: _card(slug, member, identities[slug]) for slug, member in members.items() if slug in identities},
        refs=refs,
        opening=arrangement,
        ambience="",
        seconds=seconds,
        tail=tail,
        budget=budget,
    )
    with _take(inputs):
        planned = plan_shots.plan(base, idea, total, aliases, shortest)
    chunks = [compiler.plan_of(chunk) for chunk in planned.chunks]
    changed = [
        {m: {"amend": look, "label": chunk.look_names.get(m, "")} for m, look in chunk.looks.items()}
        for chunk in planned.chunks
    ]
    with store.connect() as connection, store.transaction(connection):
        now = store.production(connection, production_id)
        assert now is not None
        # Each chapter's own: one chapter planned after another would otherwise compile with its sound.
        own = {"ambience": planned.ambience, "story": [list(event) for event in planned.story]}
        plans = {**now.manifest.get("chapter_plans", {}), str(chapter_id): own}
        fields = {"chapter_plans": plans, "seconds": seconds, "tail": tail, "budget": budget}
        store.update_production(connection, production_id, manifest={**now.manifest, **fields})
    with store.connect() as connection, store.transaction(connection):
        store.set_cast_order(connection, production_id, list(planned.members))
        existing = store.shots_of(connection, chapter_id)
        for position, (chunk, plan) in enumerate(zip(planned.chunks, chunks, strict=True)):
            camera = f"{chunk.camera_move} → {chunk.camera_toward}"
            dialogue = " / ".join(f"{line.who}: {line.line}" for line in chunk.dialogue)
            if position < len(existing) and existing[position].pinned:
                shot = existing[position]  # edited by hand: its plan stays as written
            elif position < len(existing):
                shot = existing[position]
                if _bare(shot.plan) == _bare(plan):
                    plan = shot.plan  # the same beats: the prose written for them stays
                store.update_shot(
                    connection, shot.id, plan=plan, action=chunk.purpose, camera=camera, dialogue=dialogue,
                    seconds=chunk.seconds, scene=chunk.scene,
                )  # fmt: skip
            else:
                shot = store.add_shot(connection, chapter_id, chunk.seconds, action=chunk.purpose, camera=camera)
                store.update_shot(connection, shot.id, plan=plan, dialogue=dialogue, scene=chunk.scene)
                store.set_clip_seed(connection, shot.id, video.seed + shot.position)
            first = store.keyframe_of(connection, shot.id)
            by_hand = first.source != store.CHAINED and first.note not in (OPENING_NOTE, CUT_NOTE)
            if not by_hand:
                if position == 0 or chunk.cut == "back":
                    store.provide_keyframe(connection, shot.id, str(inputs["opening"]), note=OPENING_NOTE)
                elif chunk.cut == "new":
                    store.compose_keyframe(connection, shot.id, note=CUT_NOTE)
                elif first.source != store.CHAINED:
                    store.chain_keyframe(connection, shot.id)
                # A cut's keyframe is where the clip starts, not where the clip before it ends.
                store.land_keyframe(connection, shot.id, not chunk.cut)
        for extra in existing[len(chunks) :]:
            if extra.pinned:
                ui.note(f"shot {extra.position + 1} is pinned, so it stays after the plan's {len(chunks)} shots")
            else:
                store.delete_shot(connection, extra.id)
    return {
        "chunks": chunks,
        "ambience": planned.ambience,
        "story": [list(e) for e in planned.story],
        "order": list(planned.members),
        "looks": changed,
        "voices": dict(planned.voices),
    }


def _bare(plan: dict[str, Any]) -> dict[str, Any]:
    """A plan without the writer's words: what decides whether the words still fit."""
    beats = [{k: v for k, v in beat.items() if k != "prose"} for beat in plan.get("beats", [])]
    return {**{k: v for k, v in plan.items() if k not in ("summary", "sounds", "beats")}, "beats": beats}


def _chapter(inputs: dict[str, Any]) -> tuple[store.Production, store.Chapter, list[store.Shot]]:
    with store.connect() as connection:
        production = store.production(connection, int(inputs["production"]))
        chapter = store.chapter(connection, int(inputs["chapter"]))
        assert production is not None and chapter is not None
        return production, chapter, store.shots_of(connection, chapter.id)


def run_write_shot(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production, chapter, shots = _chapter(inputs)
    position = int(inputs["position"])
    shot = shots[position]
    with store.connect() as connection:
        take = compiler.take_of(connection, production, chapter)
    made = key(
        "write_shot", 1, production=production.id, shot=shot.id, plan=_bare(shot.plan), cards=take.cards,
        refs=take.refs, opening=take.opening, ambience=take.ambience,
        closing_before=take.chunks[position - 1].closing if position else "",
        **_taken(inputs), **_fingerprint(),
    )  # fmt: skip
    if (previous := _start("write_shot", made)) is not None and shot.plan.get("summary"):
        return previous
    with _take(inputs):
        written = write_shot.write_one(take, position)
    chunk = written.chunks[position]
    with store.connect() as connection, store.transaction(connection):
        store.update_shot(connection, shot.id, plan=compiler.plan_of(chunk))
    return {"summary": chunk.summary, "sounds": chunk.sounds, "beats": [beat.prose for beat in chunk.beats]}


def run_compile_shot(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production, chapter, shots = _chapter(inputs)
    position = int(inputs["position"])
    shot = shots[position]
    with store.connect() as connection:
        take = compiler.take_of(connection, production, chapter)
        pictures = compiler.pictures_of(connection, production)
    compiled = compiler.compile_chunk(take, position)
    problems = write_shot.chunk_problems(take, position)
    if position == len(shots) - 1:
        everything = [compiler.compile_chunk(take, index).prompt for index in range(len(shots))]
        problems += checks.cross_chunk_rules(take, everything) + checks.plan_rules(take)
    found = [dataclasses.asdict(problem) for problem in problems]
    made = key(
        "compile_shot", 1, production=production.id, shot=shot.id, prompt=compiled.prompt, pictures=compiled.references
    )
    handed = [pictures[name] for name in compiled.references if name in pictures]
    with store.connect() as connection, store.transaction(connection):
        prompt = store.prompt_of(connection, shot.id)
        if not prompt.pinned:
            store.set_compiled(connection, shot.id, compiled.prompt, handed, found, str(made))
    if problems:
        ui.panel(
            f"shot {position + 1}: the gate",
            {f"{p.severity} · {p.code} {i}": p.message for i, p in enumerate(problems)},
            color="red" if any(p.severity == "error" for p in problems) else "yellow",
        )
    return {"text": compiled.prompt, "pictures": handed, "problems": len(problems), "key": str(made)}


def run_refine_idea(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    production_id, round_ = int(inputs["production"]), int(inputs["round"])
    frames = [_frame(p) for p in inputs["photos"]]
    with _take({"take": round_}):  # every round samples anew; the same round asked again is cached
        draft = refine_idea.refine(
            str(inputs["idea"]), frames, float(inputs["take_seconds"]), float(inputs["seconds"]),
            str(inputs.get("previous") or ""), str(inputs.get("feedback") or ""),
        )  # fmt: skip
    story = draft.story()
    with store.connect() as connection, store.transaction(connection):
        store.write_idea(connection, production_id, round_, story, list(draft.changes), list(draft.questions))
    return {"story": story, "changes": list(draft.changes), "questions": list(draft.questions)}


PLAN_RUNNERS = {
    "guess_reference": run_guess_reference,
    "same_person": run_same_person,
    "survey": run_survey,
    "ground": run_ground,
    "cut_crops": run_cut_crops,
    "cards": run_cards,
    "describe_opening": run_describe_opening,
    "tidy": run_tidy,
    "describe_member": run_describe_member,
    "arrange": run_arrange,
    "plan_shots": run_plan_shots,
    "write_shot": run_write_shot,
    "compile_shot": run_compile_shot,
    "refine_idea": run_refine_idea,
}
