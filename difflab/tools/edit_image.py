"""Qwen-Image-Edit block: a frame in, an edited frame out.

Cached by what went into it, which matters beyond not paying twice: a drawn reference must come
back bit for bit, or every chunk holding it keys differently and the take is rendered again.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from difflab import prompts
from difflab.core import models
from difflab.core.cache import cache as default_cache
from difflab.core.hub import HubFile
from difflab.core.imaging import megapixel_size, resize_to
from difflab.core.keys import key
from difflab.core.settings import EditSettings
from difflab.core.types import Frame
from difflab.models.qwen_edit import Editor


def spec(checkpoint: str) -> models.ModelSpec:
    """One registry entry per AIO file, so two checkpoints never share a slot."""
    weights = HubFile.parse(checkpoint)
    return models.ModelSpec(
        name=f"qwen_edit:{weights.filename}",
        build=lambda: Editor(weights),
        checkpoints=(weights.path,),
        unload=lambda editor: editor.unload(),
    )


def edit(
    frame: Frame,
    instruction: str,
    settings: EditSettings | None = None,
    negative_prompt: str = prompts.NEGATIVE_PROMPT,
    size: tuple[int, int] | None = None,
    donors: Sequence[Frame] = (),
) -> Frame:
    """Apply `instruction` to `frame`, delivered at `size` (default: the settings' megapixels).

    The input is resized to the output size before it is handed over. `donors` are Picture 2 onward,
    passed as they are: the size stays explicit, since diffusers otherwise takes the output geometry
    from the LAST image passed, which for a donor is the wrong picture.
    """
    settings = settings or EditSettings()
    target = size or megapixel_size(frame.width, frame.height, settings.megapixels)

    def produce() -> Frame:
        model: Any = models.registry.acquire(spec(settings.checkpoint))
        resized = resize_to(frame.pil(), target)
        edited = model.edit(
            resized,
            instruction,
            negative_prompt=negative_prompt,
            steps=settings.steps,
            cfg=settings.cfg,
            seed=settings.seed,
            donors=[donor.pil() for donor in donors],
        )
        return Frame.from_pil(edited, origin=f"edited({frame.origin})" if frame.origin else "edited")

    made = key(
        "edit_image",
        1,
        frame=frame.pixels,
        instruction=instruction,
        settings=settings,
        negative=negative_prompt,
        size=size,
        **({"donors": [donor.pixels for donor in donors]} if donors else {}),
    )
    return default_cache.frame(made, produce)
