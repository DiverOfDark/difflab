"""Boxes around the members one photo shows, and around anyone else in it."""

from __future__ import annotations

from typing import Any

from difflab.core.types import Frame
from difflab.models.minimax_h3 import caption
from difflab.screenplay import config as h3
from difflab.screenplay import crops, words
from difflab.tools import direct


def ground(frame: Frame, listed: list[direct.Member]) -> list[crops.Found]:
    """Boxes around the members a photo shows, and around anyone else in it (a bystander, "other")."""
    wanted = [m for m in listed if m.kind != "environment"]
    items = [words.GROUND_ITEM.format(id=m.id, find=m.find, kind=m.kind) for m in wanted] + [words.GROUND_OTHERS]
    labels = {m.id for m in wanted} | {"other"}
    kinds = {m.id: m.kind for m in wanted}

    def parse(data: Any) -> list[crops.Found]:
        found = []
        for entry in data:
            box = crops.Box.relative(entry["bbox_2d"], frame.width, frame.height)
            head = entry.get("head_bbox_2d")
            head_box = crops.Box.relative(head, frame.width, frame.height) if head and len(head) == 4 else None
            part = str(entry.get("part") or "") if kinds.get(str(entry["label"])) == "object" else ""
            found.append(
                crops.Found(str(entry["label"]), box, head_box if head_box and head_box.area > 0 else None, part)
            )
        return found

    def gate(data: Any) -> list[caption.Problem]:
        found = parse(data)
        out = [
            direct.problem(f'"{t.label}" is not a label here; use only {sorted(labels)}')
            for t in found
            if t.label not in labels
        ]
        out += [direct.problem(f'the box of "{t.label}" is empty') for t in found if t.box.area <= 0]
        out += [direct.problem(f'find "{m.id}": {m.find}') for m in wanted if not any(t.label == m.id for t in found)]
        return out

    question = words.GROUND_USER.format(items="\n".join(items))
    image = direct.shrink(frame, h3.DIRECTOR_PHOTO_PIXELS)
    data = direct.ask_json("", question, [image], gate, f"ground · {frame.origin}", h3.DIRECTOR_TOKENS["ground"])
    return [thing for thing in parse(data) if thing.label in labels and thing.box.area > 0]
