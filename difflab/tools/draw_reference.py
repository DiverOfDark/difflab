"""A member no photograph shows, drawn by the editor from its words. The editor has no text-to-image
mode, so it edits a blank grey plate: a plate holds nothing it could be asked to keep."""

from __future__ import annotations

from PIL import Image

from difflab.core.imaging import megapixel_size
from difflab.core.settings import EditSettings
from difflab.core.types import Frame
from difflab.screenplay import compiler, words
from difflab.tools import edit_image

PLATE = (128, 128, 128)
ALIGN = 16  # the editor floors its output to /16


def instruction(card: compiler.Card, fixes: list[str] | None = None) -> str:
    said = words.COMPOSE_REFERENCE_INSTRUCTION[card.kind].format(
        noun=card.noun,
        traits=card.full(),
        lights=compiler.join(list(card.lights)) or "soft natural daylight",
        look=words.COMPOSE_REFERENCE_LOOK,
    )
    return f"{said} {words.COMPOSE_FIXES.format(fixes='; '.join(fixes))}" if fixes else said


def draw(card: compiler.Card, settings: EditSettings, size: tuple[int, int], fixes: list[str] | None = None) -> Frame:
    """Drawn at the editor's own size in `size`'s aspect, then brought down to `size`."""
    width, height = megapixel_size(*size, settings.megapixels)
    work = (max(ALIGN, width // ALIGN * ALIGN), max(ALIGN, height // ALIGN * ALIGN))
    plate = Frame.from_pil(Image.new("RGB", work, PLATE), origin="plate")
    return edit_image.edit(plate, instruction(card, fixes), settings, size=work).resized(*size)
