"""A comic page cut into its panels by code: gutters, borders and reading order, no model.

A page is cut recursively on blank rows and columns once the specks are gone (lettering in a gutter,
a scanlator's watermark), so the cuts come out in reading order. A webtoon strip is cut on bands of
flat background across its whole width, and a band that is mostly background is lettering between
pictures rather than a picture: it is read with the picture before it and drawn with none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage

READINGS = ("rtl", "ltr", "strip")
INK = 170  # luma: darker is a line or a fill; screentone greys are lighter and do not join panels
# A connected run of ink smaller than this share of the page is lettering or a watermark, not a panel.
SPECK = 0.004
GUTTER = 0.006  # of the page's side: the narrowest blank run that separates two panels
PANEL_MIN = 0.02  # of the page's area: anything smaller is a crop of nothing
# A panel whose own border inks this share of its outline is a drawn frame; below, it is borderless or was
# cut through a figure, and the reader confirms it. A side on the page's edge bleeds and counts as framed.
FRAMED = 0.75
EDGE = 0.02  # of the page's side
# Two panels with one shared border line and no white gutter: a line across the whole panel inked this
# fully, no thicker than LINE px, is cut on. A black fill is thicker.
LINE_INKED = 0.97
LINE = 8
LINE_SIDE = 0.2  # of the panel: neither side of such a cut is narrower
LINE_DARK = 0.5  # a panel this inked is a dark fill or a cover, where any column can look like a line
LARGE = 0.45  # of the page's area: a panel this big may be several the gutters did not separate
STRIP_FLAT = 6.0  # luma spread across a strip row that still counts as flat background
STRIP_GAP = 24  # px at the strip's own width: a flat run this tall separates two pictures
STRIP_TEXT = 0.7  # a band this much background is lettering between pictures
STRIP_LOOSE = 0.35  # a picture band this much background holds balloons too: the reader tightens its box


@dataclass(frozen=True)
class Panel:
    box: tuple[int, int, int, int]  # left, top, right, bottom on the page, in pixels
    framed: bool  # its border was found all round: code is sure of it
    lettering: tuple[tuple[int, int, int, int], ...] = ()  # a strip's lettering bands read with it

    @property
    def read_box(self) -> tuple[int, int, int, int]:
        """The panel with the lettering read with it."""
        boxes = [self.box, *self.lettering]
        return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


@dataclass(frozen=True)
class Page:
    size: tuple[int, int]
    panels: tuple[Panel, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def settled(self) -> bool:
        return all(p.framed for p in self.panels) and not self.notes


def cut(image: Image.Image, reading: str) -> Page:
    if reading not in READINGS:
        raise ValueError(f"reading is one of {', '.join(READINGS)}, not {reading!r}")
    grey = np.asarray(image.convert("L"), dtype=np.uint8)
    return _strip(grey) if reading == "strip" else _page(grey, reading == "rtl")


def _page(grey: np.ndarray, rtl: bool) -> Page:
    height, width = grey.shape
    ink = grey < INK
    labels, count = ndimage.label(ink, structure=np.ones((3, 3), dtype=bool))
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, count + 1))
    boxes = ndimage.find_objects(labels)
    keep = np.zeros(count + 1, dtype=bool)
    for index, (found, size) in enumerate(zip(boxes, sizes, strict=True), 1):
        area = (found[0].stop - found[0].start) * (found[1].stop - found[1].start)
        keep[index] = area >= SPECK * height * width and size > 0
    mask = keep[labels]
    out: list[tuple[int, int, int, int]] = []
    _xy(mask, ink, 0, 0, width, height, max(3, round(GUTTER * min(height, width))), rtl, out)
    out = [b for b in out if (b[2] - b[0]) * (b[3] - b[1]) >= PANEL_MIN * height * width]
    if not out:
        out = [(0, 0, width, height)]
    panels = tuple(Panel(box, _framed(ink, box)) for box in out)
    notes = []
    if any((p.box[2] - p.box[0]) * (p.box[3] - p.box[1]) >= LARGE * height * width for p in panels):
        notes.append("a panel covers most of the page")
    if any(_overlap(a.box, b.box) for i, a in enumerate(panels) for b in panels[i + 1 :]):
        notes.append("two panels overlap")
    return Page((width, height), panels, tuple(notes))


def _xy(
    mask: np.ndarray, ink: np.ndarray, left: int, top: int, right: int, bottom: int, gap: int, rtl: bool,
    out: list[tuple[int, int, int, int]],
) -> None:  # fmt: skip
    region = mask[top:bottom, left:right]
    rows, cols = region.any(axis=1), region.any(axis=0)
    if not rows.any():
        return
    top, bottom = top + int(rows.argmax()), top + len(rows) - int(rows[::-1].argmax())
    left, right = left + int(cols.argmax()), left + len(cols) - int(cols[::-1].argmax())
    region = mask[top:bottom, left:right]
    across = _runs(region.any(axis=1), gap)
    if len(across) > 1:
        for start, end in across:
            _xy(mask, ink, left, top + start, right, top + end, gap, rtl, out)
        return
    down = _runs(region.any(axis=0), gap)
    if len(down) > 1:
        for start, end in reversed(down) if rtl else down:
            _xy(mask, ink, left + start, top, left + end, bottom, gap, rtl, out)
        return
    inked = ink[top:bottom, left:right]
    if inked.mean() >= LINE_DARK:
        out.append((left, top, right, bottom))
        return
    if (at := _line(inked.mean(axis=1))) is not None:
        _xy(mask, ink, left, top, right, top + at, gap, rtl, out)
        _xy(mask, ink, left, top + at, right, bottom, gap, rtl, out)
        return
    if (at := _line(inked.mean(axis=0))) is not None:
        halves = [(left, left + at), (left + at, right)]
        for start, end in reversed(halves) if rtl else halves:
            _xy(mask, ink, start, top, end, bottom, gap, rtl, out)
        return
    out.append((left, top, right, bottom))


def _line(inked: np.ndarray) -> int | None:
    """Where a thin line runs fully across, away from the edges."""
    margin = max(LINE * 2, round(len(inked) * LINE_SIDE))
    full = inked >= LINE_INKED
    index = margin
    while index < len(inked) - margin:
        if not full[index]:
            index += 1
            continue
        end = index
        while end < len(inked) and full[end]:
            end += 1
        if (
            end - index <= LINE
            and inked[max(0, index - LINE)] < LINE_INKED
            and inked[min(len(inked) - 1, end + LINE)] < LINE_INKED
        ):
            return (index + end) // 2
        index = end
    return None


def _runs(filled: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """The stretches of `filled` separated by at least `gap` empty places."""
    out: list[tuple[int, int]] = []
    start, empty = None, 0
    for index, value in enumerate(filled):
        if value:
            if start is None:
                start = index
            elif empty >= gap:
                out.append((start, index - empty))
                start = index
            empty = 0
        else:
            empty += 1
    if start is not None:
        out.append((start, len(filled) - empty))
    return out


def _framed(ink: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Whether a line runs round the box: each side inked over FRAMED of its length within a few pixels."""
    left, top, right, bottom = box
    height, width = ink.shape
    band = 4
    sides = [
        (top <= EDGE * height, ink[top : top + band, left:right].any(axis=0)),
        (bottom >= (1 - EDGE) * height, ink[bottom - band : bottom, left:right].any(axis=0)),
        (left <= EDGE * width, ink[top:bottom, left : left + band].any(axis=1)),
        (right >= (1 - EDGE) * width, ink[top:bottom, right - band : right].any(axis=1)),
    ]
    return all(bleeds or (side.size and side.mean() >= FRAMED) for bleeds, side in sides)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _strip(grey: np.ndarray) -> Page:
    height, width = grey.shape
    pixels = grey.astype(np.int16)
    flat = (pixels.max(axis=1) - pixels.min(axis=1)) <= STRIP_FLAT
    gap = max(4, round(STRIP_GAP * width / 720))
    bands = _runs(~flat, gap)
    background = np.median(grey[flat]) if flat.any() else 255
    pictures: list[Panel] = []
    pending: list[tuple[int, int, int, int]] = []
    for start, end in bands:
        box = (0, start, width, end)
        share = float((np.abs(pixels[start:end] - background) <= STRIP_FLAT * 2).mean())
        if share >= STRIP_TEXT or (end - start) < PANEL_MIN * width * 4:
            if pictures and not pending:
                last = pictures[-1]
                pictures[-1] = Panel(last.box, last.framed, (*last.lettering, box))
            else:
                pending.append(box)
            continue
        pictures.append(Panel(_trimmed(grey, box, background), share < STRIP_LOOSE, tuple(pending)))
        pending = []
    if pending and pictures:
        last = pictures[-1]
        pictures[-1] = Panel(last.box, last.framed, (*last.lettering, *pending))
    if not pictures:
        return Page((width, height), (Panel((0, 0, width, height), False),), ("no picture found in the strip",))
    return Page((width, height), tuple(pictures))


def _trimmed(grey: np.ndarray, box: tuple[int, int, int, int], background: float) -> tuple[int, int, int, int]:
    """The band without the flat background either side of a picture narrower than the strip."""
    left, top, right, bottom = box
    cols = (np.abs(grey[top:bottom, left:right].astype(np.int16) - background) > STRIP_FLAT * 2).mean(axis=0) > 0.02
    if not cols.any():
        return box
    return left + int(cols.argmax()), top, right - int(cols[::-1].argmax()), bottom
