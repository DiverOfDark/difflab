"""The director's references, cut in code from the VLM's boxes by the rules the hand-cut first cast
was tuned with.

A tight face crop beside the photo is what holds identity over a long take, so a person or animal
with a big enough head gets a 4:5 portrait. A body or view crop has no margin below, where the
blanket or table under the member is. The first photo, whole, is the environment and already holds
whoever sits in it at about the same size, so they get no body crop from it. No crop may show a
bystander: it is cut tighter, or back past them, or dropped.

Every crop is resized with LANCZOS to a size on the 32 grid within the canvas's pixel area, so H3's
own reference sizing (`pipeline.reference_size(..., "match")`) passes it through untouched. The box
is widened to that size's aspect first, so nothing is stretched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from difflab import config as app_config
from difflab.core.types import Frame
from difflab.screenplay import config, words

ALIGN = app_config.CANVAS_ALIGN
LIVING = ("person", "animal")


@dataclass(frozen=True)
class Box:
    """`(left, top, right, bottom)` in the photo's pixels."""

    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def relative(cls, values: list[float], width: int, height: int) -> Box:
        """From the model's 0-1000 scale (asked for in GROUND_USER), clipped to the photo."""
        left, top, right, bottom = (float(v) for v in values)
        return cls(
            max(0.0, min(left, right) * width / 1000),
            max(0.0, min(top, bottom) * height / 1000),
            min(float(width), max(left, right) * width / 1000),
            min(float(height), max(top, bottom) * height / 1000),
        )

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def centre(self) -> tuple[float, float]:
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def inside(self, other: Box) -> float:
        """How much of this box lies inside `other`, as a fraction of this box."""
        width = min(self.right, other.right) - max(self.left, other.left)
        height = min(self.bottom, other.bottom) - max(self.top, other.top)
        return max(0.0, width) * max(0.0, height) / self.area if self.area > 0 else 0.0

    def union(self, other: Box) -> Box:
        return Box(
            min(self.left, other.left),
            min(self.top, other.top),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )

    def grown(self, margin: float, width: int, height: int, below: bool = True) -> Box:
        """`margin` of its own size added on every side (not below, unless `below`), inside the photo."""
        dx, dy = self.width * margin, self.height * margin
        return Box(
            max(0.0, self.left - dx),
            max(0.0, self.top - dy),
            min(width, self.right + dx),
            min(height, self.bottom + (dy if below else 0.0)),
        )

    def pixels(self) -> tuple[int, int, int, int]:
        return round(self.left), round(self.top), round(self.right), round(self.bottom)


@dataclass(frozen=True)
class Found:
    """One thing the VLM located in one photo."""

    label: str  # a member id, or "other" for a bystander
    box: Box
    head: Box | None = None
    part: str = ""  # an object is found part by part: "bowl", "hose"


@dataclass(frozen=True)
class Crop:
    key: str  # refs/<key>.png
    photo: int  # index into the photos
    member: str
    role: str  # face | body | view | object | environment
    box: Box
    size: tuple[int, int]
    shows: tuple[str, ...]  # members at least DIRECTOR_SHOWS inside, its own first
    quality: float  # the head's width for a face, the box's area otherwise: of two alike, the larger wins

    @property
    def rows(self) -> int:
        """Reference rows it costs a chunk: one latent frame of (height/16/2) x (width/16/2) patches."""
        return (self.size[0] // 32) * (self.size[1] // 32)

    def cut(self, photo: Frame) -> Frame:
        return photo.crop(self.box.pixels()).resized(*self.size)


def _place(width: float, height: float, centre: tuple[float, float], bounds: tuple[int, int]) -> Box:
    """A `width` x `height` box as close to `centre` as fits inside the photo."""
    left = min(max(0.0, centre[0] - width / 2), bounds[0] - width)
    top = min(max(0.0, centre[1] - height / 2), bounds[1] - height)
    return Box(left, top, left + width, top + height)


def reshape(box: Box, aspect: float, bounds: tuple[int, int]) -> Box:
    """`box` widened (or, against an edge, narrowed) to `aspect` about its centre, inside the photo."""
    width, height = box.width, box.height
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    shrink = min(1.0, bounds[0] / width, bounds[1] / height)
    return _place(width * shrink, height * shrink, box.centre(), bounds)


def fitted(box: Box, pixels: float, bounds: tuple[int, int], canvas: int) -> tuple[Box, tuple[int, int]]:
    """A size on the 32 grid near `pixels` (at most DIRECTOR_MAX_UPSCALE up, never above `canvas`), and
    `box` reshaped to its aspect so the resize does not stretch."""
    scale = min(math.sqrt(pixels / box.area), config.DIRECTOR_MAX_UPSCALE)
    width = max(ALIGN, round(box.width * scale / ALIGN) * ALIGN)
    height = max(ALIGN, round(box.height * scale / ALIGN) * ALIGN)
    while width * height > canvas:
        if width >= height:
            width -= ALIGN
        else:
            height -= ALIGN
    return reshape(box, width / height, bounds), (width, height)


def face(head: Box, body: Box, kind: str, bounds: tuple[int, int], canvas: int) -> tuple[Box, tuple[int, int]] | None:
    """The 4:5 portrait around a head, always tall enough for the whole of it.

    An animal's head box is often drawn from the eyes down with its ears above it, so the portrait
    reaches up to where the animal's own box starts - but no further than DIRECTOR_FACE_EARS, which is
    what keeps a raised tail from turning the portrait into a view.
    """
    if head.width < config.DIRECTOR_FACE_MIN:
        return None
    unit_width, unit_height = (side // 4 for side in config.DIRECTOR_FACE_SIZE)  # 128 x 160 steps
    headroom = head.height * config.DIRECTOR_FACE_HEADROOM
    ears = min(max(0.0, head.top - body.top), head.height * config.DIRECTOR_FACE_EARS) if kind == "animal" else 0.0
    top = head.top - max(headroom, ears + headroom / 4)
    needed = head.bottom + headroom - top  # the whole head, with as much again below the chin as above it
    width = max(head.width * config.DIRECTOR_FACE_WIDTH[kind], needed * unit_width / unit_height)
    height = width * unit_height / unit_width
    shrink = min(1.0, bounds[0] / width, bounds[1] / height)
    width, height = width * shrink, height * shrink
    box = _place(width, height, (head.centre()[0], max(0.0, top) + height / 2), bounds)
    steps = min(4, math.floor(box.width * config.DIRECTOR_MAX_UPSCALE / unit_width))
    while steps and unit_width * steps * unit_height * steps > canvas:
        steps -= 1
    return (box, (unit_width * steps, unit_height * steps)) if steps else None


def showing(box: Box, own: str, found: list[Found]) -> list[str]:
    """Who `box` shows: its own member, then everyone at least DIRECTOR_SHOWS inside it (bystanders as "other")."""
    shown = [own]
    for thing in found:
        if thing.label in shown and thing.label != "other":
            continue
        head = thing.head is not None and thing.head.inside(box) >= config.DIRECTOR_SHOWS_HEAD
        if thing.box.inside(box) >= config.DIRECTOR_SHOWS or head:
            shown.append(thing.label)
    return shown


def best(found: list[Found], member: str, kind: str) -> Found | None:
    """A member's box in one photo: the largest for a person or animal, the union of every part for an
    object (a hookah reported as its bowl and its hose is one hookah)."""
    mine = [thing for thing in found if thing.label == member]
    if not mine:
        return None
    if kind in LIVING:
        return max(mine, key=lambda thing: thing.box.area)
    union = mine[0].box
    for thing in mine[1:]:
        union = union.union(thing.box)
    return Found(member, union, union)


def bystanders(found: list[Found]) -> list[Found]:
    """Anyone reported as "other" who is not the same box as a member reported twice."""
    members = [thing for thing in found if thing.label != "other"]
    within, covers = config.DIRECTOR_SAME_THING
    return [
        thing
        for thing in found
        if thing.label == "other"
        and not any(thing.box.inside(m.box) > within and m.box.inside(thing.box) > covers for m in members)
    ]


@dataclass(frozen=True)
class _Option:
    """A crop to make, and the boxes to try for it in order: the first that shows no bystander is cut."""

    member: str
    role: str
    tries: list[tuple[Box, tuple[int, int]]]
    quality: float


def cut_all(
    sizes: list[tuple[int, int]],
    found: list[list[Found]],
    shows: list[tuple[str, ...]],
    kinds: dict[str, str],
    canvas: int,
    opening: Box,
) -> tuple[list[Crop], list[str]]:
    """Every crop of every photo, in presentation order, and a note for each one dropped.

    `canvas` is the render canvas's pixel area, which no crop may exceed. The environment is tried as
    the first photo whole, then cut back past any bystander, then the part the take opens on.
    """
    crops: list[Crop] = []
    notes: list[str] = []
    counts: dict[str, int] = {}

    def key(member: str, role: str) -> str:
        name = member if role == "environment" else f"{member}_{role}"
        counts[name] = counts.get(name, 0) + 1
        return name if counts[name] == 1 else f"{name}{counts[name]}"

    for photo, (bounds, things, listed) in enumerate(zip(sizes, found, shows, strict=True)):
        clean = [thing for thing in things if thing.label != "other"] + bystanders(things)
        others = [thing.box for thing in clean if thing.label == "other"]
        options: list[_Option] = []
        environment = next((m for m in listed if kinds[m] == "environment"), None)
        if environment is not None:
            whole = Box(0, 0, *bounds)
            trimmed = clear_of(whole, others, [thing.box for thing in clean if thing.label != "other"])
            boxes = (whole, trimmed, opening)
            tries = [fitted(box, config.DIRECTOR_ENVIRONMENT_PIXELS, bounds, canvas) for box in boxes]
            options.append(_Option(environment, "environment", tries, whole.area))
            living = [thing.box for thing in clean if thing.label == "other" or kinds.get(thing.label) in LIVING]
            room = clear_of(whole, living, [])
            if living and room.area >= config.DIRECTOR_ROOM_MIN * whole.area:
                room_try = fitted(room, config.DIRECTOR_ENVIRONMENT_PIXELS, bounds, canvas)
                options.append(_Option(environment, "environment", [room_try], room.area))
        for member in (m for m in listed if kinds[m] != "environment"):
            kind, thing = kinds[member], best(things, member, kinds[member])
            if thing is None:
                continue
            portrait = None
            if kind in LIVING and thing.head is not None:
                heads = (thing.head, _narrow(thing.head))
                tries = [f for f in (face(head, thing.box, kind, bounds, canvas) for head in heads) if f is not None]
                if tries:
                    options.append(_Option(member, "face", tries, thing.head.width))
                    portrait = tries[0][0]
            if environment is not None:
                continue  # the environment photo, whole, already holds everyone in it at this size
            role = {"person": "body", "animal": "view", "object": "object"}[kind]
            margin = config.DIRECTOR_OBJECT_MARGIN if role == "object" else config.DIRECTOR_BODY_MARGIN
            grown = [thing.box.grown(grow, *bounds, below=role == "object") for grow in (margin, 0.0)]
            cleared = clear_of(grown[0], others, [thing.box])  # past a bystander, keeping the most of it
            grown += [cleared] if thing.box.inside(cleared) >= config.DIRECTOR_CLEAR_KEEP else []
            tries = [fitted(box, config.DIRECTOR_VIEW_PIXELS, bounds, canvas) for box in grown]
            if portrait is not None and tries[0][0].inside(portrait) >= config.DIRECTOR_PORTRAIT_COVERS:
                continue  # the face crop already is this view: a small animal's portrait holds all of it
            options.append(_Option(member, role, tries, thing.box.area))
        for option in options:
            for box, size in option.tries:
                shown = showing(box, option.member, clean)
                if "other" not in shown:
                    members = tuple(dict.fromkeys(m for m in shown if m in kinds))
                    crops.append(
                        Crop(
                            key(option.member, option.role),
                            photo,
                            option.member,
                            option.role,
                            box,
                            size,
                            members,
                            option.quality,
                        )
                    )
                    break
            else:
                notes.append(f"photo {photo + 1}: no {option.role} crop of {option.member} without a bystander in it")
    return crops, notes


def clear_of(box: Box, bystanders: list[Box], members: list[Box]) -> Box:
    """`box` cut back past each bystander on whichever side keeps the most of the members (and then of
    the box): a far-off figure at the edge of the room costs a strip, not the whole view."""
    for other in bystanders:
        if other.inside(box) <= 0:
            continue
        sides = [
            Box(box.left, box.top, min(box.right, other.left), box.bottom),
            Box(max(box.left, other.right), box.top, box.right, box.bottom),
            Box(box.left, box.top, box.right, min(box.bottom, other.top)),
            Box(box.left, max(box.top, other.bottom), box.right, box.bottom),
        ]
        box = max(
            (side for side in sides if side.area > 0),
            key=lambda side: (round(sum(m.inside(side) for m in members), 2), side.area),
            default=box,
        )
    return box


def _narrow(head: Box) -> Box:
    """A second try for a face crop that caught a bystander: the same head, cut a third tighter."""
    cx, _ = head.centre()
    width = head.width * config.DIRECTOR_FACE_NARROW
    return Box(cx - width / 2, head.top, cx + width / 2, head.bottom)


def opening_cut(bounds: tuple[int, int], aspect: float, living: list[Box], heads: list[Box], objects: list[Box]) -> Box:
    """The first photo cut at the canvas's aspect, where it holds the most of the take's first frame.

    Heads whole first, then the people and animals, then the objects part by part (the hookah's bowl
    as well as its hose): a prop the frame cuts out is a prop the video invents again somewhere else.
    """
    width, height = (
        (bounds[1] * aspect, float(bounds[1]))
        if bounds[0] / bounds[1] > aspect
        else (float(bounds[0]), bounds[0] / aspect)
    )
    centre = (bounds[0] / 2, bounds[1] / 2)
    if living:
        union = living[0]
        for box in living[1:]:
            union = union.union(box)
        centre = union.centre()
    horizontal = width < bounds[0] - 0.5
    span = (bounds[0] - width) if horizontal else (bounds[1] - height)
    keep = config.DIRECTOR_OPENING_KEEP

    def score(offset: float) -> tuple[float, float, float]:
        cut = Box(offset, 0, offset + width, height) if horizontal else Box(0, offset, width, offset + height)
        whole = sum(min(1.0, head.inside(cut) + 1e-3) for head in heads)
        kept = config.DIRECTOR_OPENING_LIVING_WEIGHT * sum(box.inside(cut) for box in living)
        kept += sum(min(box.inside(cut), keep) / keep for box in objects)
        middle = (offset + (width if horizontal else height) / 2) - centre[0 if horizontal else 1]
        return round(whole, 2), round(kept, 3), -abs(middle)

    offset = max((float(step) for step in range(math.floor(span) + 1)), key=score, default=0.0)
    return Box(offset, 0, offset + width, height) if horizontal else Box(0, offset, width, offset + height)


def where(box: Box, frame: Box, head: Box | None = None) -> str:
    """Where `box` is in `frame`, in the frame's own sides: "right of centre, from the upper third to the
    bottom edge". Said to the VLM and the writer, so neither has to guess a side from a person's own
    point of view, which an 8B model mirrors."""

    def across(fraction: float) -> str:
        return next((name for limit, name in words.FRAME_ACROSS if fraction < limit), words.FRAME_ACROSS_LAST)

    def down(fraction: float) -> str:
        return next((name for limit, name in words.FRAME_DOWN if fraction < limit), words.FRAME_DOWN_LAST)

    top, bottom = ((edge - frame.top) / frame.height for edge in (box.top, box.bottom))
    x = ((head or box).centre()[0] - frame.left) / frame.width
    span = f"in {down(top)}" if down(top) == down(bottom) else f"from {down(top)} to {down(bottom)}"
    cut = words.FRAME_PARTLY if box.inside(frame) < config.DIRECTOR_SHOWS_WHOLE else ""
    return f"{across(x)}, {span}{cut}"


def sheet(photos: list[Frame], found: list[list[Found]], crops: list[Crop], cut: Box, height: int = 720) -> Image.Image:
    """Every photo with what the VLM found and what was cut from it, for a look before rendering.

    Yellow: a member's box. Green: its head. Red: a bystander. Cyan: a crop. Magenta: the opening cut.
    """
    font = ImageFont.load_default(size=18)
    tiles = []
    for index, (photo, things) in enumerate(zip(photos, found, strict=True)):
        scale = height / photo.height
        image = photo.pil().resize((round(photo.width * scale), height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)

        def rect(box: Box, colour: str, width: int, scale: float = scale, draw: ImageDraw.ImageDraw = draw) -> None:
            draw.rectangle(
                [box.left * scale, box.top * scale, box.right * scale, box.bottom * scale], outline=colour, width=width
            )

        for crop in (c for c in crops if c.photo == index):
            rect(crop.box, "cyan", 2)
            draw.text((crop.box.left * scale + 4, crop.box.bottom * scale - 22), crop.key, fill="cyan", font=font)
        for thing in things:
            rect(thing.box, "red" if thing.label == "other" else "yellow", 3)
            if thing.head is not None and thing.head != thing.box:
                rect(thing.head, "lime", 2)
            name = f"{thing.label} ({thing.part})" if thing.part else thing.label
            draw.text((thing.box.left * scale + 4, thing.box.top * scale + 2), name, fill="yellow", font=font)
        if index == 0:
            rect(cut, "magenta", 4)
        draw.text((6, 6), f"photo {index + 1}", fill="white", font=font, stroke_width=2, stroke_fill="black")
        tiles.append(image)
    width = sum(tile.width for tile in tiles) + 8 * (len(tiles) - 1)
    board = Image.new("RGB", (width, height), "black")
    left = 0
    for tile in tiles:
        board.paste(tile, (left, 0))
        left += tile.width + 8
    return board
