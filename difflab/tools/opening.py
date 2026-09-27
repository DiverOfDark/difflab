"""The opening frame in words, and the chapter's first sentence written from them."""

from __future__ import annotations

import re
from typing import Any

from difflab.core.types import Frame
from difflab.models.minimax_h3 import caption
from difflab.screenplay import checks as lint
from difflab.screenplay import compiler as compose
from difflab.screenplay import config as h3
from difflab.screenplay import crops, words
from difflab.tools import direct


def _holds(data: Any) -> list[caption.Problem]:
    if not all(isinstance(hold, dict) and "thing" in hold and "wrist" in hold for hold in data["holds"]):
        return [direct.problem('"holds" lists {"thing": ..., "wrist": ...} entries')]
    return []


def _mouth(data: Any) -> list[caption.Problem]:
    if not all(isinstance(data.get(key), str) for key in ("mouth", "lips", "smoke")):
        return [direct.problem('answer {"mouth": ..., "lips": ..., "smoke": ...}, each a short phrase')]
    return []


def _unsided(text: str) -> str:
    """ "to his left" -> "beside him": a side said from a person's point of view is one Qwen3-VL 8B mirrored."""

    def neutral(match: re.Match[str]) -> str:
        said = match.group(0).lower().split()
        beside = "Beside " if match.group(0)[0].isupper() else "beside "
        return beside + (direct.PRONOUN[said[1]] if said[0] in ("to", "on", "at") else said[2])

    return lint.SIDE.sub(neutral, text)


def _hand(member: direct.Member, thing: str, wrist: str, card: dict[str, dict[str, Any]]) -> tuple[str, str | None]:
    """Which hand holds `thing`: the model says "watch", a trait says "a black smartwatch on his left
    wrist", so it is his left hand. Without one, only what the wrist wears -> (HANDS line, the hand)."""
    worn = [word for word in lint.tokens(wrist) if len(word) >= 4]
    for trait in card.get(member.id, {}).get("traits", []):
        phrase = str(trait.get("phrase", ""))
        side = re.search(r"\b(left|right) wrist\b", phrase, re.I)
        if side and any(word in token for word in worn for token in lint.tokens(phrase)):
            hand = f"{member.possessive} {side.group(1).lower()} hand"
            return (
                f"the {member.noun} holds {thing} in {hand}: that wrist wears {wrist}, and a trait puts {phrase}",
                hand,
            )
    return f"the hand of the {member.noun} that holds {thing} wears on its wrist: {wrist}", None


def describe_opening(
    opening: Frame,
    members: list[direct.Member],
    card: dict[str, dict[str, Any]],
    layout: list[tuple[direct.Member, str, str]],
) -> direct.Seen:
    """The opening frame in plain words, its members' places from their boxes, each holding hand and
    each mouth: `arrange_opening` turns it all into the take's first sentence.

    direct.Seen face on, a person's left hand is on the picture's right, and Qwen3-VL 8B mirrored it: asked
    which hand holds the hose it says "right" even when told not to, asked what is on the wrist of the
    hand that holds it, "a watch". So the sides it gives are struck out, it is asked the second way,
    and the card's "smartwatch on his left wrist" names the hand (`_hand`). A hand wrong here is a hand
    wrong for the whole take, and the same goes for "the table to his left". The mouth is asked on its
    own because a first frame is a moment in the middle of something: an exhale under way is what the
    take's first beat has to finish.
    """
    (low, high), (floor, ceiling) = h3.DIRECTOR_OPENING_WORDS, h3.DIRECTOR_OPENING_WORDS_HARD

    def gate(data: Any) -> list[caption.Problem]:
        text = str(data["description"])
        count = len(text.split())
        out = (
            []
            if floor <= count <= ceiling
            else [direct.problem(f"the description has {count} words; describe the frame in {low} to {high}")]
        )
        if side := lint.SIDE.search(text):
            out.append(
                direct.problem(f"say sides as the picture's ('at the left of the picture'), not '{side.group(0)}'")
            )
        return out

    def traits(member: direct.Member) -> str:
        return "; ".join(str(t.get("phrase", "")) for t in card.get(member.id, {}).get("traits", []))

    question = words.OPENING_USER.format(
        members="\n".join(f"- {m.noun} ({m.kind}): {traits(m)}" for m in members),
        layout="\n".join(f"- {m.noun}{f' ({part})' if part else ''}: {place}" for m, part, place in layout),
    )
    system = words.OPENING_SYSTEM.format(low=low, high=high)
    image = direct.shrink(opening, h3.DIRECTOR_PHOTO_PIXELS)
    data = direct.ask_json(system, question, [image], gate, "opening frame", h3.DIRECTOR_TOKENS["opening"])
    description = re.sub(r"\b(?:left|right) (hand|paw|wrist)", r"\1", _unsided(str(data["description"]).strip()))

    hands: list[tuple[str, str | None]] = []
    mouths: list[str] = []
    for member in members:
        if member.kind == "person":
            question = words.HANDS_USER.format(noun=member.noun)
            answer = direct.ask_json(
                "", question, [image], _holds, f"hands · {member.id}", h3.DIRECTOR_TOKENS["opening"]
            )
            hands += [_hand(member, str(hold["thing"]), str(hold["wrist"]), card) for hold in answer["holds"]]
        if member.kind in crops.LIVING:
            question = words.MOUTH_USER.format(noun=member.noun)
            state = direct.ask_json(
                "", question, [image], _mouth, f"mouth · {member.id}", h3.DIRECTOR_TOKENS["opening"]
            )
            mouths.append(
                f"the {member.noun}: mouth {state.get('mouth', '?')}, touching the lips: {state.get('lips', '?')}, "
                f"smoke: {state.get('smoke', '?')}"
            )
    return direct.Seen(
        description=description,
        layout="\n".join(f"- {{{m.id}}}{f' ({part})' if part else ''}: {place}" for m, part, place in layout),
        hands="\n".join(line for line, _ in hands) or "no hand holds anything",
        sided=tuple(hand for _, hand in hands if hand),
        mouths="\n".join(mouths) or "no mouth to see",
    )


def arrange_opening(seen: direct.Seen, cards: list[compose.Card]) -> str:
    """The first frame as the take's first sentence, members as {id}: chunk 1 opens on these words."""
    ids = {card.id for card in cards}
    (low, high), (floor, ceiling) = h3.DIRECTOR_ARRANGE_WORDS, h3.DIRECTOR_ARRANGE_WORDS_HARD
    # A 96-word description does not become a 70-word sentence with nothing dropped: asking for that
    # got a 30-word overrun three times over. The budget is the config's until the description is
    # longer than it, and then it is the description.
    source = len(seen.description.split())
    high, ceiling = max(high, source), max(ceiling, round(source * 1.15))

    def gate(data: Any) -> list[caption.Problem]:
        text = str(data["arrangement"])
        named = set(re.findall(r"\{([a-z_]+)\}", text))
        out = [direct.problem(f"name {{{m}}} in braces") for m in sorted(ids - named)]
        out += [direct.problem(f"{{{m}}} is not one of the listed members") for m in sorted(named - ids)]
        if message := lint.possessive(text):
            out.append(direct.problem(message))
        if side := lint.SIDE.search(text):
            out.append(
                direct.problem(f"say sides as the frame's ('at the left edge of the frame'), not '{side.group(0)}'")
            )
        for hand in seen.sided:  # deduced from a trait, so certain: the plan keeps whatever this sentence says
            if hand.split(" ", 1)[1] not in text.lower():
                out.append(direct.problem(f"HANDS says it is {hand}: write '{hand}' for the hand that holds it"))
        count = len(text.split())
        if not floor <= count <= ceiling:
            out.append(direct.problem(f"the sentence has {count} words; write one sentence of {low} to {high}"))
        return out

    question = words.ARRANGE_USER.format(
        members="\n".join(f"{c.id} | {c.noun} | {c.full()}" for c in cards),
        layout=seen.layout,
        description=seen.description,
        hands=seen.hands,
        mouths=seen.mouths,
    )
    system = words.ARRANGE_SYSTEM.format(low=low, high=high)
    data = direct.ask_json(system, question, [], gate, "opening", h3.DIRECTOR_TOKENS["opening"])
    return str(data["arrangement"]).strip()
