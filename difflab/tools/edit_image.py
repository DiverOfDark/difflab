"""Qwen-Image-Edit block: a frame in, an edited frame out.

Cached by what went into it, which matters beyond not paying twice: a drawn reference must come
back bit for bit, or every chunk holding it keys differently and the take is rendered again.
"""

from __future__ import annotations

import math
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

VAE_REFERENCE_AREA = 1024 * 1024
"""The pixel area QwenImageEditPlus encodes every input at for its VAE reference latents. Not a knob:
a fact about the pipeline, which `vae_reference_size` exists only to agree with."""


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


def vae_reference_size(frame: Frame, area: int = VAE_REFERENCE_AREA) -> tuple[int, int]:
    """The size diffusers encodes `frame` at as its VAE reference: `area`, on the /32 grid.

    Output at exactly this size and an in-place edit lands on the reference latent 1:1, so the model
    replaces around the subject instead of re-composing it; a mismatched size is where a plate's zoom
    and crop drift come from. Each axis rounds to /32 on its own, so the result can sit up to ~1.5%
    off the input's aspect.

    Not `minimax_h3.pipeline.reference_size`, which is H3's reference encoding and a different thing.
    """
    scale = math.sqrt(area / (frame.width * frame.height))
    align = 32
    return (
        max(align, round(frame.width * scale / align) * align),
        max(align, round(frame.height * scale / align) * align),
    )


def plate(frame: Frame, instruction: str, settings: EditSettings | None = None) -> Frame:
    """A reference photo with ONE subject on ONE flat backdrop, back at the photo's own size.

    A reference picture lends H3 everything in it - the face, and also the sofa and the hose in someone's
    hand, which come back as a cut to that room or as a second hose. `settings.megapixels` does not apply:
    the size is the reference size, so the seed is the only knob a plate turns.
    """
    return edit(frame, instruction, settings, size=vae_reference_size(frame)).resized(frame.width, frame.height)
