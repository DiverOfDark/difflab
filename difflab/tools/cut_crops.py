"""The references, cut in code from the survey's boxes: the opening frame and every crop, checked
to pass H3's reference sizing untouched."""

from __future__ import annotations

from dataclasses import dataclass

from difflab.core import ui
from difflab.core.types import Frame
from difflab.models.minimax_h3 import pipeline
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import config as h3
from difflab.screenplay import crops
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
) -> list[crops.Crop]:
    """Every crop, checked to pass H3's own reference sizing untouched."""
    sizes = [(frame.width, frame.height) for frame in frames]
    made, notes = crops.cut_all(sizes, found, shows, kinds, canvas.width * canvas.height, cut)
    for note in notes:
        ui.note(note)
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
    return made


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


def canvas_for(photo: Frame, aspect: float, settings: VideoSettings) -> pipeline.Geometry:
    """The canvas the take renders at: the one `render_chunk.plan` gives opening.png itself."""
    cut = photo.crop_to_aspect(aspect)
    return pipeline.settling_canvas(cut.width, cut.height, settings.duration, settings.megapixels)


@dataclass(frozen=True)
class Opening:
    frame: Frame  # the first photo cut at the canvas's aspect and resized to it: what the chapter opens on
    cut: crops.Box  # where in the first photo
    canvas: pipeline.Geometry
    crops: list[crops.Crop]


def opening_and_crops(
    frames: list[Frame],
    found: list[list[crops.Found]],
    shows: list[tuple[str, ...]],
    kinds: dict[str, str],
    aspect: float | None,
    settings: VideoSettings,
) -> Opening:
    """The take's first frame, cut where the first photo holds the most of its cast, and every
    reference crop of every photo. `aspect` defaults to the first photo's own."""
    ratio = frames[0].width / frames[0].height if aspect is None else aspect
    canvas = canvas_for(frames[0], ratio, settings)
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
    return Opening(opening, box, canvas, cut(frames, found, shows, kinds, canvas, box))
