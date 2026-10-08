"""Trait cards: what the VLM saw of each member in each photo, tidied into one identity per member."""

from __future__ import annotations

import json
from typing import Any

from difflab.core import ui
from difflab.core.types import Frame
from difflab.models.minimax_h3 import caption
from difflab.screenplay import compiler as compose
from difflab.screenplay import config as h3
from difflab.screenplay import crops, words
from difflab.tools import direct, ruling


def identify(frames: list[Frame], listed: list[list[direct.Member]]) -> list[dict[str, dict[str, Any]]]:
    """One trait card per photo: {member id: {"traits": [...], "lights": [...]}}. A photo showing none of
    the cast gets an empty card and no question; a member the direct.VLM adds on its own is left out."""
    cards: list[dict[str, dict[str, Any]]] = []
    for frame, members in zip(frames, listed, strict=True):
        wanted = [m.id for m in members]
        if not wanted:
            cards.append({})
            continue
        question = words.CARD_USER.format(
            members="\n".join(words.CARD_MEMBER.format(id=m.id, find=m.find, kind=m.kind) for m in members)
        )

        def parse(data: Any, wanted: list[str] = wanted) -> dict[str, dict[str, Any]]:
            card = {str(entry["id"]): dict(entry) for entry in data["members"] if str(entry["id"]) in wanted}
            for entry in card.values():
                entry["lights"] = [str(x["phrase"]) if isinstance(x, dict) else str(x) for x in entry.get("lights", [])]
            return card

        def gate(
            data: Any, wanted: list[str] = wanted, members: list[direct.Member] = members
        ) -> list[caption.Problem]:
            card = parse(data)
            missing = [member for member in wanted if member not in card]
            out = [direct.problem(f"record every listed member; missing: {missing}")] if missing else []
            others = {m.id: m for m in members}
            for member in members:
                entry = card.get(member.id, {"traits": [], "lights": []})
                out += _trait_problems(member.id, entry["traits"]) or ruling.traits(member, others, entry["traits"])
                if member.kind == "environment" and not entry.get("lights"):
                    out.append(direct.problem(f'{member.id}: list every visible light source in "lights"'))
            return out

        image = direct.shrink(frame, h3.DIRECTOR_PHOTO_PIXELS)
        data = direct.ask_json(
            words.CARD_SYSTEM, question, [image], gate, f"card · {frame.origin}", h3.DIRECTOR_TOKENS["card"]
        )
        cards.append(parse(data))
    return cards


def _trait_problems(member: str, traits: Any) -> list[caption.Problem]:
    parts = h3.DIRECTOR_TRAIT_ORDER
    out = [] if len(traits) >= 2 else [direct.problem(f"{member}: record at least 2 traits")]
    for trait in traits:
        if trait.get("part") not in parts:
            out.append(direct.problem(f'{member}: "part" is one of {", ".join(parts)}'))
        if not all(isinstance(trait.get(field), str) and trait[field].strip() for field in ("phrase", "short", "name")):
            out.append(direct.problem(f'{member}: every trait has "part", "phrase", "short" and "name"'))
    return out


# --- writing: text only from here on ----------------------------------------------------------------------------
UNCOUNTABLE = {
    "hair",
    "fur",
    "skin",
    "polish",
    "nail polish",
    "makeup",
    "stubble",
    "clothing",
    "denim",
    "smoke",
    "haze",
}


DETERMINERS = ("a ", "an ", "the ", "his ", "her ", "its ", "their ", "one ", "two ", "three ", "some ")


def article(phrase: str, name: str) -> str:
    """ "black hookah hose" -> "a black hookah hose": every trait is said after "with". `name` is the
    thing itself, so a plural or uncountable one takes none."""
    phrase = phrase.strip()
    noun = name.strip().lower()
    if not phrase or phrase.lower().startswith(DETERMINERS) or noun in UNCOUNTABLE or noun.endswith("s"):
        return phrase
    return ("an " if phrase[0].lower() in "aeiou" else "a ") + phrase


def _traits(entries: Any) -> tuple[compose.Trait, ...]:
    return tuple(
        compose.Trait(
            str(t["part"]),
            article(str(t["phrase"]), str(t["name"])),
            article(str(t["short"]), str(t["name"])),
            str(t["name"]),
        )
        for t in entries
    )


def invent(idea: str, member: direct.Member, members: dict[str, direct.Member]) -> compose.Card:
    """The card of a member the idea needs and no photograph shows: its look decided, not observed.

    H3 renders a person it has no reference for from the words of the prompt, so "a man enters" needs
    a card, not a photograph. `photographed=False` is what keeps `compose.definition` from citing a
    picture for him and `lint.plan_rules` from asking which crop shows him.
    """
    question = words.INVENT_USER.format(
        idea=idea.strip(), member=member.id, kind=member.kind, noun=member.noun, find=member.find
    )

    def gate(data: Any) -> list[caption.Problem]:
        out = _trait_problems(member.id, data["traits"]) or ruling.traits(member, members, data["traits"])
        if member.kind == "environment" and not data.get("lights"):
            out.append(direct.problem('a place lists every source of its light in "lights"'))
        return out

    data = direct.ask_json(
        words.INVENT_SYSTEM, question, [], gate, f"card · {member.id} · invented", h3.DIRECTOR_TOKENS["tidy"]
    )
    lights = tuple(str(x) for x in data.get("lights", [])) if member.kind == "environment" else ()
    return compose.Card(
        member.id, member.kind, member.noun, member.possessive, _traits(data["traits"]), lights, photographed=False
    )


def merge(
    idea: str,
    members: dict[str, direct.Member],
    cards: list[dict[str, dict[str, Any]]],
    found: list[list[crops.Found]],
    shows: list[tuple[str, ...]],
) -> tuple[dict[str, compose.Card], dict[str, list[str]]]:
    """One clean card per member, tidied from what the model noted per photo and merged over them all.

    Photos taken under different light disagree on colour: the same cat is black under a lamp and
    brown at a window. The card says both, lighter first; any other disagreement goes the way of the
    photo with the largest view of the member.

    Whether a member was photographed is `shows`, what the survey decided, not whether the card stage
    happened to note it: a `card · <photo>` call that settles without a member it was asked about
    leaves no notes for someone the photographs do show, and its crops were already cut. Inventing a
    look there would present pictures that no subject line cites, which no later repair can mend.
    """
    photographed = {member for listed in shows for member in listed}
    merged: dict[str, compose.Card] = {}
    conflicts: dict[str, list[str]] = {}
    for member in members.values():
        seen = {f"photo {i + 1}": card[member.id] for i, card in enumerate(cards) if member.id in card}
        if member.id not in photographed:  # no photograph shows it: its look is decided, not merged
            merged[member.id] = invent(idea, member, members)
            continue
        if not seen:
            raise direct.DirectorError(
                f"tidy · {member.id}: every photograph that shows it was read without a card for it"
            )
        sizes = {
            f"photo {i + 1}": max((t.box.area for t in f if t.label == member.id), default=0.0)
            for i, f in enumerate(found)
        }
        notes = "\n".join(
            f"{key}: {json.dumps({'traits': entry['traits'], 'lights': entry['lights']}, ensure_ascii=False)}"
            for key, entry in seen.items()
        )
        question = words.TIDY_USER.format(
            member=member.id,
            kind=member.kind,
            noun=member.noun,
            primary=max(seen, key=lambda key: sizes.get(key, 0.0)),
            cards=notes,
        )

        def gate(data: Any, member: direct.Member = member) -> list[caption.Problem]:
            out = _trait_problems(member.id, data["traits"]) or ruling.traits(member, members, data["traits"])
            if member.kind == "environment" and not data.get("lights"):
                out.append(direct.problem('list every light source in "lights"'))
            return out

        data = direct.ask_json(words.TIDY_SYSTEM, question, [], gate, f"card · {member.id}", h3.DIRECTOR_TOKENS["tidy"])
        lights = tuple(str(x) for x in data.get("lights", [])) if member.kind == "environment" else ()
        conflicts[member.id] = [str(conflict) for conflict in data.get("conflicts", [])]
        for conflict in conflicts[member.id]:
            ui.note(f"{member.id}: {conflict}")
        merged[member.id] = compose.Card(
            member.id, member.kind, member.noun, member.possessive, _traits(data["traits"]), lights
        )
    rows = {card.id: card.full() for card in merged.values()}
    rows |= {f"{card.id} · lights": compose.join(card.lights) for card in merged.values() if card.lights}
    ui.panel("cards", rows)
    return merged, conflicts
