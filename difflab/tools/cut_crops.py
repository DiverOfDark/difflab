"""The references, cut in code from the survey's boxes: the opening frame and every crop, checked
to pass H3's reference sizing untouched."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageFilter

from difflab.core import ui
from difflab.core.types import Frame
from difflab.models.minimax_h3 import pipeline
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import config as h3
from difflab.screenplay import crops
from difflab.screenplay.crops import Dropped
from difflab.tools import direct, render_chunk


def inside(found: list[crops.Found], member: str, cut: crops.Box) -> bool:
    return any(t.label == member and t.box.inside(cut) >= h3.DIRECTOR_SHOWS for t in found)


def cut(
    frames: list[Frame],
    found: list[list[crops.Found]],
    shows: list[tuple[str, ...]],
    kinds: dict[str, str],
    canvas: pipeline.Geometry,
    cut: crops.Box,
) -> tuple[list[crops.Crop], list[Dropped]]:
    """Every crop, checked to pass H3's own reference sizing untouched, and each one not cut."""
    sizes = [(frame.width, frame.height) for frame in frames]
    made, dropped = crops.cut_all(sizes, found, shows, kinds, canvas.width * canvas.height, cut)
    for crop in made:
        if pipeline.reference_size(*crop.size, canvas, "match") != crop.size:
            raise RuntimeError(f"{crop.key} {crop.size} would be resampled by H3's reference sizing")
    ui.panel(
        "crops",
        {
            crop.key: f"photo {crop.photo + 1} {crop.box.pixels()} → {crop.size[0]}×{crop.size[1]} · "
            f"{crop.rows} rows · shows {', '.join(crop.shows)}"
            for crop in made
        },
    )
    return made, dropped


def layout(
    found: list[crops.Found], members: list[direct.Member], cut: crops.Box
) -> list[tuple[direct.Member, str, str]]:
    """Where each member of the opening frame is in it, by its boxes: (member, part, where)."""
    return [
        (member, thing.part, crops.where(thing.box, cut, thing.head if member.kind in crops.LIVING else None))
        for member in members
        for thing in found
        if thing.label == member.id and thing.box.inside(cut) > 0
    ]


def extends(photo: Frame, aspect: float) -> bool:
    own = photo.width / photo.height
    return max(own, aspect) / min(own, aspect) > h3.OPENING_EXTEND_BEYOND


def around(photo: Frame, aspect: float) -> crops.Box:
    """The canvas's shape centred on the whole photo, reaching past two of its sides."""
    if photo.width / photo.height < aspect:
        side = (photo.height * aspect - photo.width) / 2
        return crops.Box(-side, 0, photo.width + side, photo.height)
    side = (photo.width / aspect - photo.height) / 2
    return crops.Box(0, -side, photo.width, photo.height + side)


def canvas_for(photo: Frame, aspect: float, settings: VideoSettings, extend: bool = False) -> pipeline.Geometry:
    """The canvas the take renders at: the one `render_chunk.plan` gives opening.png itself."""
    if extend:
        box = around(photo, aspect)
        width, height = round(box.width), round(box.height)
    else:
        cut = photo.crop_to_aspect(aspect)
        width, height = cut.width, cut.height
    return pipeline.settling_canvas(width, height, settings.duration, settings.megapixels)


def padded(photo: Frame, width: int, height: int) -> Frame:
    """The whole photo fitted into `width`×`height` over a blurred cover of itself, the blur for the editor to
    replace."""
    image = photo.pil()
    cover = max(width / photo.width, height / photo.height)
    wide, tall = math.ceil(photo.width * cover), math.ceil(photo.height * cover)
    left, top = (wide - width) // 2, (tall - height) // 2
    back = image.resize((wide, tall), Image.Resampling.LANCZOS).crop((left, top, left + width, top + height))
    back = back.filter(ImageFilter.GaussianBlur(max(width, height) / 40))
    fit = min(width / photo.width, height / photo.height)
    inner = image.resize((round(photo.width * fit), round(photo.height * fit)), Image.Resampling.LANCZOS)
    back.paste(inner, ((width - inner.width) // 2, (height - inner.height) // 2))
    return Frame.from_pil(back, f"{photo.origin} (padded)" if photo.origin else "")


@dataclass(frozen=True)
class Opening:
    frame: Frame  # the first photo cut, or padded, to the canvas: what the chapter opens on
    cut: crops.Box  # where in the first photo; past its sides when extended
    canvas: pipeline.Geometry
    crops: list[crops.Crop]
    dropped: list[Dropped]
    extend: bool = False  # `frame` is the photo padded, for the editor to fill the sides of


def opening_and_crops(
    frames: list[Frame],
    found: list[list[crops.Found]],
    shows: list[tuple[str, ...]],
    kinds: dict[str, str],
    aspect: float | None,
    settings: VideoSettings,
    fit: str = "crop",
) -> Opening:
    """The take's first frame, cut where the first photo holds the most of its cast, or with `fit`
    "extend" the whole photo padded out to the canvas, and every reference crop of every photo.
    `aspect` defaults to the first photo's own."""
    ratio = frames[0].width / frames[0].height if aspect is None else aspect
    extend = fit == "extend" and extends(frames[0], ratio)
    canvas = canvas_for(frames[0], ratio, settings, extend)
    if extend:
        box = around(frames[0], canvas.width / canvas.height)
        opening = padded(frames[0], canvas.width, canvas.height)
    else:
        living = [thing for thing in found[0] if kinds.get(thing.label) in crops.LIVING]
        box = crops.opening_cut(
            (frames[0].width, frames[0].height),
            canvas.width / canvas.height,
            [thing.box for thing in living],
            [thing.head for thing in living if thing.head is not None],
            [thing.box for thing in found[0] if kinds.get(thing.label) == "object"],
        )
        opening = frames[0].crop(box.pixels()).resized(canvas.width, canvas.height)
    planned = render_chunk.plan(opening, settings)
    if (planned.width, planned.height) != (canvas.width, canvas.height):
        raise RuntimeError(f"the opening plans {planned.width}×{planned.height}, not {canvas.width}×{canvas.height}")
    within = crops.Box(0, 0, frames[0].width, frames[0].height) if extend else box
    return Opening(opening, box, canvas, *cut(frames, found, shows, kinds, canvas, within), extend=extend)
