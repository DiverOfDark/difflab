"""One look at every photo: the members the idea needs, and which photo shows which of them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from difflab.core import ui
from difflab.core.types import Frame
from difflab.models.minimax_h3 import caption
from difflab.screenplay import config as h3
from difflab.screenplay import crops, words
from difflab.tools import direct


@dataclass(frozen=True)
class Surveyed:
    members: dict[str, direct.Member]
    shows: list[tuple[str, ...]]
    aliases: dict[str, str] = field(default_factory=dict)  # member -> what the idea calls it, comma-joined
    # photo index -> (what it shows, the member it most resembles or ""), for a photo that shows no member
    unplaced: dict[int, tuple[str, str]] = field(default_factory=dict)


# --- vision: the direct.VLM sees the photos ---------------------------------------------------------------------
def survey(idea: str, frames: list[Frame]) -> Surveyed:
    """One look at every photo: the members the idea needs, and which photo shows which of them."""

    def parse(data: Any) -> tuple[dict[str, direct.Member], list[tuple[str, ...]]]:
        members = {
            str(m["id"]): direct.Member(
                str(m["id"]), str(m["kind"]), str(m["noun"]), str(m["possessive"]), str(m["find"])
            )
            for m in data["members"]
        }
        pictures = {int(p["picture"]): [str(i) for i in p["shows"]] for p in data["pictures"]}
        places = [m.id for m in members.values() if m.kind == "environment"]
        # The first place is Picture 1's whatever the answer lists; a further place is shown only where listed.
        shows = []
        for number in range(1, len(frames) + 1):
            listed = [m for m in pictures.get(number, []) if m in members and m not in places[:1]]
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
            direct.problem(f'"{i}" is in a picture but is not one of the members') for i in sorted(listed - set(ids))
        ]
        # A member no picture shows is created from words alone (`cards.invent`) and drawn, an object the
        # action uses included: production 13's jian (2026-10-05), left out as in no picture, came back a
        # different length in every clip and a scabbard for a moment mid-guard.
        return out

    images = [direct.shrink(frame, h3.DIRECTOR_SURVEY_PIXELS) for frame in frames]
    question = words.SURVEY_USER.format(idea=idea.strip(), count=len(frames))
    data = direct.ask_json(words.SURVEY_SYSTEM, question, images, gate, "survey", h3.DIRECTOR_TOKENS["survey"])
    members, shows = parse(data)
    aliases = {
        str(m["id"]): ", ".join(
            dict.fromkeys(" ".join(str(a).split()) for a in m.get("aliases") or [] if str(a).strip())
        )
        for m in data["members"]
    }
    about = {int(p["picture"]) - 1: p for p in data["pictures"]}
    unplaced: dict[int, tuple[str, str]] = {}
    for i, listed in enumerate(shows):
        if not listed:
            entry = about.get(i, {})
            likely = str(entry.get("likely") or "")
            unplaced[i] = (" ".join(str(entry.get("about") or "").split()), likely if likely in members else "")
    rows = {f"{m.id}": f"{m.kind} · {m.noun} · {m.find}" for m in members.values()}
    rows |= {f"photo {i + 1}": ", ".join(listed) or f"- {unplaced[i][0]}" for i, listed in enumerate(shows)}
    ui.panel("cast", rows)
    return Surveyed(members, shows, {m: a for m, a in aliases.items() if a and m in members}, unplaced)
