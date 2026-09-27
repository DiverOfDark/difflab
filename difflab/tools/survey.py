"""One look at every photo: the members the idea needs, and which photo shows which of them."""

from __future__ import annotations

from typing import Any

from difflab.core import ui
from difflab.core.types import Frame
from difflab.models.minimax_h3 import caption
from difflab.screenplay import config as h3
from difflab.screenplay import crops, words
from difflab.tools import direct


# --- vision: the direct.VLM sees the photos ---------------------------------------------------------------------
def survey(idea: str, frames: list[Frame]) -> tuple[dict[str, direct.Member], list[tuple[str, ...]]]:
    """One look at every photo: the members the idea needs, and which photo shows which of them."""

    def parse(data: Any) -> tuple[dict[str, direct.Member], list[tuple[str, ...]]]:
        members = {
            str(m["id"]): direct.Member(
                str(m["id"]), str(m["kind"]), str(m["noun"]), str(m["possessive"]), str(m["find"])
            )
            for m in data["members"]
        }
        pictures = {int(p["picture"]): [str(i) for i in p["shows"]] for p in data["pictures"]}
        environment = [m.id for m in members.values() if m.kind == "environment"]
        shows = []
        for number in range(1, len(frames) + 1):
            listed = [m for m in pictures.get(number, []) if m in members and m not in environment]
            shows.append(tuple(dict.fromkeys([*listed, *(environment if number == 1 else [])])))
        return members, shows

    def gate(data: Any) -> list[caption.Problem]:
        members, shows = parse(data)
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
            direct.problem(f'{m.id}: "possessive" is his, her or its')
            for m in members.values()
            if m.possessive not in ("his", "her", "its")
        ]
        if sum(m.kind == "environment" for m in members.values()) != 1:
            out.append(direct.problem("exactly one member is the environment: the place in Picture 1"))
        if not any(m.kind in crops.LIVING for m in members.values()):
            out.append(direct.problem("the IDEA needs at least one person or animal"))
        if len(data["pictures"]) != len(frames):
            out.append(direct.problem(f'"pictures" has one entry per picture: {len(frames)}'))
        listed = {str(i) for p in data["pictures"] for i in p["shows"]}
        out += [
            direct.problem(f'"{i}" is in a picture but is not one of the members') for i in sorted(listed - set(ids))
        ]
        # A person or animal no picture shows is created from words alone (`invent`). An object or a
        # place is there to be seen, or it is not a member.
        unseen = [m.id for m in members.values() if m.kind not in crops.LIVING and not any(m.id in s for s in shows)]
        only_living = "only a person or an animal can be a member no picture shows"
        out += [
            direct.problem(f"{m} is in no picture: list it where it is seen, or leave it out - {only_living}")
            for m in unseen
        ]
        return out

    images = [direct.shrink(frame, h3.DIRECTOR_SURVEY_PIXELS) for frame in frames]
    question = words.SURVEY_USER.format(idea=idea.strip(), count=len(frames))
    data = direct.ask_json(words.SURVEY_SYSTEM, question, images, gate, "survey", h3.DIRECTOR_TOKENS["survey"])
    members, shows = parse(data)
    rows = {f"{m.id}": f"{m.kind} · {m.noun} · {m.find}" for m in members.values()}
    rows |= {f"photo {i + 1}": ", ".join(listed) or "-" for i, listed in enumerate(shows)}
    ui.panel("cast", rows)
    return members, shows
