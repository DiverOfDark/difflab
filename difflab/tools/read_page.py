"""A comic page read into its panels and each panel's script: what is lettered, who is in it, what happens,
and whether the next panel carries on from it."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image, ImageDraw

from difflab.models.minimax_h3 import caption
from difflab.screenplay import config as knobs
from difflab.screenplay import words
from difflab.tools import direct, panels

KINDS = ("speech", "thought", "shout", "narration")
SPOKEN = ("speech", "shout")  # said aloud; a thought and a caption are subtitles alone
VOICED = ("speech", "thought", "shout")
JOINS = ("land", "cut")
NAME_WORDS = 4
ORDER = {"rtl": "right to left, then top to bottom", "ltr": "left to right, then top to bottom",
         "strip": "top to bottom"}  # fmt: skip


@dataclass(frozen=True)
class Balloon:
    kind: str
    speaker: str
    text: str


@dataclass(frozen=True)
class Script:
    balloons: tuple[Balloon, ...]
    sfx: tuple[str, ...]
    who: tuple[str, ...]
    action: str
    into_next: str
    why: str
    drawn: bool = True  # False: lettering on a blank ground, with no picture to draw from

    def data(self) -> dict[str, Any]:
        return {**asdict(self), "balloons": [asdict(b) for b in self.balloons], "sfx": list(self.sfx),
                "who": list(self.who)}  # fmt: skip

    @classmethod
    def of(cls, data: dict[str, Any]) -> Script:
        return cls(
            tuple(Balloon(str(b["kind"]), str(b["speaker"]), str(b["text"])) for b in data.get("balloons", [])),
            tuple(str(s) for s in data.get("sfx", [])),
            tuple(str(w) for w in data.get("who", [])),
            str(data.get("action", "")),
            str(data.get("into_next", "cut")),
            str(data.get("why", "")),
            bool(data.get("drawn", True)),
        )


def sentence_case(text: str) -> str:
    """Lettering in capitals as a sentence; anything already in mixed case as it is."""
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) < 0.8 * len(letters):
        return text
    lowered = text.lower()
    out = re.sub(r"(^|[.!?…]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), lowered)
    return re.sub(r"\bi\b", "I", out)


def _name(said: Any) -> str:
    return " ".join(str(said).lower().split())


def confirm(page: Image.Image, cut: panels.Page, reading: str) -> panels.Page:
    """The panels of a page the gutters did not settle, as the VLM sees them over the numbered boxes."""
    if cut.settled or reading == "strip":
        return cut
    width, height = cut.size
    shown = page.convert("RGB")
    draw = ImageDraw.Draw(shown)
    side = max(width, height)
    for number, panel in enumerate(cut.panels, 1):
        draw.rectangle(panel.box, outline=(255, 0, 0), width=max(3, side // 250))
        draw.text((panel.box[0] + side // 100, panel.box[1] + side // 100), str(number), fill=(255, 0, 0),
                  font_size=max(24, side // 25))  # fmt: skip
    image = _shrunk(shown, knobs.DIRECTOR_PHOTO_PIXELS)

    def boxes(data: Any) -> list[tuple[int, int, int, int]]:
        out = []
        for entry in data["panels"]:
            left, top, right, bottom = (float(v) for v in entry["box"])
            out.append((round(left * width / 1000), round(top * height / 1000), round(right * width / 1000),
                        round(bottom * height / 1000)))  # fmt: skip
        return out

    def gate(data: Any) -> list[caption.Problem]:
        found = boxes(data)
        out = []
        if not found:
            out.append(direct.problem('"panels" lists at least one panel'))
        for number, (left, top, right, bottom) in enumerate(found, 1):
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                out.append(direct.problem(f"panel {number}: a box is [left, top, right, bottom] within 0-1000"))
            elif (right - left) * (bottom - top) < panels.PANEL_MIN * width * height:
                out.append(direct.problem(f"panel {number} is too small to be a panel: drop it or join it"))
        return out

    data = direct.ask_json(
        words.MANGA_CONFIRM_SYSTEM.format(order=ORDER[reading]),
        words.MANGA_CONFIRM_USER.format(count=len(cut.panels)),
        [image], gate, "panels", knobs.MANGA_TOKENS["confirm"],
    )  # fmt: skip
    return panels.Page(cut.size, tuple(panels.Panel(box, True) for box in boxes(data)))


def confirm_bands(strip: Image.Image, cut: panels.Page) -> panels.Page:
    """A strip's bands split into their pictures; a band with none is lettering read with the picture before."""
    out: list[panels.Panel] = []
    pending: list[tuple[int, int, int, int]] = []
    for panel in cut.panels:
        left, top, right, bottom = panel.box
        w, h = right - left, bottom - top
        band = strip.crop(panel.box)
        least = 1000 * panels.PANEL_MIN * 4 * w / h  # of the band, as `panels._strip` counts a picture

        def gate(data: Any, least: float = least) -> list[caption.Problem]:
            wrong = []
            for number, values in enumerate(data["pictures"], 1):
                box = [float(v) for v in values]
                if len(box) != 4 or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
                    wrong.append(direct.problem(f"picture {number}: a box is [left, top, right, bottom] within 0-1000"))
                elif box[3] - box[1] < least:
                    wrong.append(direct.problem(f"picture {number} is too small to be a panel: drop it or join it"))
            return wrong

        data = direct.ask_json(
            words.MANGA_BAND_SYSTEM, words.MANGA_BAND_USER, [_shrunk(band, knobs.DIRECTOR_PHOTO_PIXELS)], gate,
            "band", knobs.MANGA_TOKENS["confirm"],
        )  # fmt: skip
        boxes = [
            (left + round(float(x0) * w / 1000), top + round(float(y0) * h / 1000),
             left + round(float(x1) * w / 1000), top + round(float(y1) * h / 1000))
            for x0, y0, x1, y1 in data["pictures"]
        ]  # fmt: skip
        if not boxes:
            if out:
                last = out[-1]
                out[-1] = panels.Panel(last.box, True, (*last.lettering, panel.box, *panel.lettering))
            else:
                pending += [panel.box, *panel.lettering]
            continue
        above = min(b[1] for b in boxes)
        below = max(b[3] for b in boxes)
        for index, box in enumerate(boxes):
            lettering = [*pending, *panel.lettering, (left, top, right, above)] if index == 0 else []
            if index == len(boxes) - 1:
                lettering.append((left, below, right, bottom))
            out.append(panels.Panel(box, True, tuple(b for b in lettering if b[3] - b[1] > 8)))
        pending = []
    if pending and out:
        last = out[-1]
        out[-1] = panels.Panel(last.box, True, (*last.lettering, *pending))
    return panels.Page(cut.size, tuple(out) or cut.panels)


def read(
    panel: Image.Image,
    page: Image.Image | None,
    after: Image.Image | None,
    names: list[str],
    reading: str,
) -> Script:
    """One panel's script; `after` is the panel read next, None for the last of the chapter."""
    images = [_shrunk(panel, knobs.MANGA_READ_PIXELS)]
    context = ""
    if page is not None:
        images.append(_shrunk(page, knobs.DIRECTOR_SURVEY_PIXELS))
        context += words.MANGA_READ_CONTEXT_PAGE
    if after is not None:
        images.append(_shrunk(after, knobs.DIRECTOR_SURVEY_PIXELS))
        context += words.MANGA_READ_CONTEXT_NEXT.format(picture=len(images))
        into = words.MANGA_READ_LAND.format(picture=len(images))
    else:
        into = words.MANGA_READ_LAST
    system = words.MANGA_READ_SYSTEM.format(
        context=context, order=ORDER[reading], into_next_rule=into,
        known_rule=words.MANGA_READ_KNOWN if names else "",
    )  # fmt: skip
    question = words.MANGA_READ_USER.format(
        known=words.MANGA_READ_NAMES.format(names=", ".join(names)) if names else ""
    )

    def gate(data: Any) -> list[caption.Problem]:
        out = []
        who = {_name(w) for w in data["who"]}
        for number, balloon in enumerate(data["balloons"], 1):
            kind, speaker, text = str(balloon["kind"]), _name(balloon["speaker"]), str(balloon["text"]).strip()
            if kind not in KINDS:
                out.append(direct.problem(f'balloon {number}: "kind" is one of {", ".join(KINDS)}'))
            if not text:
                out.append(direct.problem(f'balloon {number}: "text" is what is lettered in it, never empty'))
            if kind == "narration" and speaker:
                out.append(direct.problem(f'balloon {number} is narration: its "speaker" is ""'))
            if kind in VOICED and not speaker:
                out.append(direct.problem(f'balloon {number} is {kind}: name its "speaker"'))
            if kind in VOICED and speaker and speaker not in who:
                out.append(direct.problem(f'balloon {number}: its speaker "{speaker}" is missing from "who"'))
        for name in [_name(b["speaker"]) for b in data["balloons"]] + list(who):
            if name and (len(name.split()) > NAME_WORDS or not re.search(r"[a-z]", name)):
                out.append(direct.problem(f'"{name}" is not a NAME: the character\'s own name, or at most '
                                          f"{NAME_WORDS} plain words"))  # fmt: skip
        if str(data["into_next"]) not in JOINS:
            out.append(direct.problem('"into_next" is "land" or "cut"'))
        if not str(data["action"]).strip():
            out.append(direct.problem('"action" says in one sentence what happens in the panel'))
        return out

    data = direct.ask_json(system, question, images, gate, "read the panel", knobs.MANGA_TOKENS["read"])
    balloons = tuple(
        Balloon(str(b["kind"]), _name(b["speaker"]), sentence_case(" ".join(str(b["text"]).split())))
        for b in data["balloons"]
    )
    return Script(
        balloons,
        tuple(" ".join(str(s).split()) for s in data.get("sfx") or [] if str(s).strip()),
        tuple(dict.fromkeys(_name(w) for w in data["who"] if _name(w))),
        " ".join(str(data["action"]).split()),
        str(data["into_next"]) if after is not None else "cut",
        " ".join(str(data.get("why") or "").split()),
        data.get("drawn") is not False,
    )


def _shrunk(image: Image.Image, pixels: float) -> Image.Image:
    image = image.convert("RGB")
    scale = (pixels / (image.width * image.height)) ** 0.5
    if scale >= 1.0:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
