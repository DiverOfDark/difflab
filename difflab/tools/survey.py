"""One look at every photo: the members the idea needs, and which photo shows which of them.

A long story is surveyed a chapter at a time, each chapter told what the ones before it found, so the cast
grows chapter by chapter under ids that stay put. What the person said about a picture overrides the model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from difflab.core import ui
from difflab.core.types import Frame
from difflab.models.minimax_h3 import caption
from difflab.screenplay import config as h3
from difflab.screenplay import crops, words
from difflab.tools import direct

# A row's kind at ingest, as the member kinds it may be.
KINDS_OF = {"character": ("person", "animal"), "location": ("environment",), "prop": ("object",)}


@dataclass(frozen=True)
class Label:
    kind: str = ""  # character | location | prop, or "" when the person left it to the model
    entity: str = ""  # a member id, or ""


@dataclass(frozen=True)
class Surveyed:
    members: dict[str, direct.Member]
    shows: list[tuple[str, ...]]
    aliases: dict[str, str] = field(default_factory=dict)  # member -> what the idea calls it, comma-joined
    # photo index -> (what it shows, the member it most resembles or ""), for a photo that shows no member
    unplaced: dict[int, tuple[str, str]] = field(default_factory=dict)
    new: tuple[str, ...] = ()  # the members this chapter found that the ones before it had not


# --- vision: the direct.VLM sees the photos ---------------------------------------------------------------------
def survey(
    idea: str,
    frames: list[Frame],
    known: Surveyed | None = None,
    labels: list[Label] | None = None,
    same: list[list[int]] | None = None,
) -> Surveyed:
    """`labels` per photo, `same` groups of photo indices that show one person; `known` is the chapters before."""
    before = known or Surveyed({}, [() for _ in frames])
    given = list(labels or []) + [Label()] * (len(frames) - len(labels or []))

    def kind_of(member: direct.Member, label: Label) -> direct.Member:
        kinds = KINDS_OF.get(label.kind)
        if kinds is None or member.kind in kinds:
            return member
        possessive = member.possessive if kinds[0] in crops.LIVING else "its"
        return dataclasses.replace(member, kind=kinds[0], possessive=possessive)

    def parse(data: Any) -> tuple[dict[str, direct.Member], list[tuple[str, ...]]]:
        answered = {
            str(m["id"]): direct.Member(
                str(m["id"]), str(m["kind"]), str(m["noun"]), str(m["possessive"]), str(m["find"])
            )
            for m in data["members"]
        }
        members = {**before.members, **{k: m for k, m in answered.items() if k not in before.members}}
        for label in given:
            if label.entity in members:
                members[label.entity] = kind_of(members[label.entity], label)
        pictures = {int(p["picture"]): [str(i) for i in p["shows"]] for p in data["pictures"]}
        places = [m.id for m in members.values() if m.kind == "environment"]
        # The first place is Picture 1's whatever the answer lists; a further place is shown only where listed.
        shows = []
        for number in range(1, len(frames) + 1):
            named = given[number - 1].entity
            listed = [*before.shows[number - 1], *pictures.get(number, []), *([named] if named in members else [])]
            listed = [m for m in listed if m in members and m not in places[:1]]
            shows.append(tuple(dict.fromkeys([*listed, *(places[:1] if number == 1 else [])])))
        return members, shows

    def gate(data: Any) -> list[caption.Problem]:
        members, _ = parse(data)
        ids = [str(m["id"]) for m in data["members"]]
        out = [
            direct.problem(f'"{i}" is not a valid id: one lowercase word, or two joined by "_"')
            for i in ids
            if not direct.ID.match(i)
        ]
        if len(set(ids)) != len(ids):
            out.append(direct.problem("every member has a different id"))
        out += [
            direct.problem(f'{m.id}: "kind" is one of {", ".join(direct.KINDS)}')
            for m in members.values()
            if m.kind not in direct.KINDS
        ]
        out += [
            direct.problem(f'{m.id}: "possessive" is his, her, their or its')
            for m in members.values()
            if m.possessive not in ("his", "her", "their", "its")
        ]
        if not any(m.kind == "environment" for m in members.values()):
            out.append(direct.problem("the first environment member is the place in Picture 1"))
        if not any(m.kind in crops.LIVING for m in members.values()):
            out.append(direct.problem("the IDEA needs at least one person or animal"))
        if len(data["pictures"]) != len(frames):
            out.append(direct.problem(f'"pictures" has one entry per picture: {len(frames)}'))
        listed = {str(i) for p in data["pictures"] for i in p["shows"]}
        out += [
            direct.problem(f'"{i}" is in a picture but is not one of the members')
            for i in sorted(listed - set(ids) - set(before.members))
        ]
        shown = {int(p["picture"]): {str(i) for i in p["shows"]} for p in data["pictures"]}
        for number, label in enumerate(given, 1):
            kinds = KINDS_OF.get(label.kind, direct.KINDS)
            if label.entity and label.entity not in members:
                out.append(direct.problem(f'Picture {number} is labelled "{label.entity}": list a member with that '
                                          f"id, of kind {' or '.join(kinds)}"))  # fmt: skip
            elif (
                label.kind
                and not label.entity
                and not any(members[m].kind in kinds for m in shown.get(number, set()) if m in members)
            ):
                out.append(direct.problem(f"Picture {number} is labelled a {label.kind}: list it as a member of "
                                          f"kind {' or '.join(kinds)} in that picture's \"shows\""))  # fmt: skip
        # A member no picture shows is created from words alone (`cards.invent`) and drawn, an object the
        # action uses included: production 13's jian (2026-10-05), left out as in no picture, came back a
        # different length in every clip and a scabbard for a moment mid-guard.
        return out

    images = [direct.shrink(frame, h3.DIRECTOR_SURVEY_PIXELS) for frame in frames]
    question = words.SURVEY_USER.format(idea=idea.strip(), count=len(frames))
    if before.members:
        listed = "\n".join(f"{m.id} | {m.kind} | {m.noun} | {m.find}" for m in before.members.values())
        question += "\n\n" + words.SURVEY_KNOWN.format(known=listed)
    told = _told(given, same or [])
    if told:
        question += "\n\n" + words.SURVEY_LABELS.format(labels="\n".join(told))
    data = direct.ask_json(words.SURVEY_SYSTEM, question, images, gate, "survey", h3.DIRECTOR_TOKENS["survey"])
    members, shows = parse(data)
    aliases = dict(before.aliases)
    for m in data["members"]:
        said = [" ".join(str(a).split()) for a in m.get("aliases") or [] if str(a).strip()]
        had = [a for a in aliases.get(str(m["id"]), "").split(", ") if a]
        aliases[str(m["id"])] = ", ".join(dict.fromkeys([*had, *said]))
    about = {int(p["picture"]) - 1: p for p in data["pictures"]}
    unplaced: dict[int, tuple[str, str]] = {}
    for i, listed_ in enumerate(shows):
        if not listed_:
            entry = about.get(i, {})
            likely = str(entry.get("likely") or "")
            unplaced[i] = (" ".join(str(entry.get("about") or "").split()), likely if likely in members else "")
    rows = {f"{m.id}": f"{m.kind} · {m.noun} · {m.find}" for m in members.values()}
    rows |= {f"photo {i + 1}": ", ".join(listed_) or f"- {unplaced[i][0]}" for i, listed_ in enumerate(shows)}
    ui.panel("cast", rows)
    new = tuple(m for m in members if m not in before.members)
    return Surveyed(members, shows, {m: a for m, a in aliases.items() if a and m in members}, unplaced, new)


def _told(labels: list[Label], same: list[list[int]]) -> list[str]:
    out = []
    for number, label in enumerate(labels, 1):
        kinds = " or ".join(KINDS_OF.get(label.kind, direct.KINDS))
        if label.entity:
            out.append(
                words.SURVEY_LABEL_ENTITY.format(
                    picture=number, kind=label.kind or "member", entity=label.entity, kinds=kinds
                )
            )
        elif label.kind:
            out.append(words.SURVEY_LABEL_KIND.format(picture=number, kind=label.kind, kinds=kinds))
    for group in same:
        numbers = [str(i + 1) for i in sorted(group)]
        if len(numbers) > 1 and not any(labels[i].entity for i in group if i < len(labels)):
            out.append(words.SURVEY_LABEL_SAME.format(pictures=", ".join(numbers[:-1]) + " and " + numbers[-1]))
    return out
