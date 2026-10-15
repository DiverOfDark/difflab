"""Qwen-Image-Edit block: a frame in, an edited frame out.

Cached by what went into it, which matters beyond not paying twice: a drawn reference must come
back bit for bit, or every chunk holding it keys differently and the take is rendered again.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import torch

from difflab import prompts
from difflab.core import models, ui
from difflab.core.cache import cache as default_cache
from difflab.core.hub import HubFile, parse_file
from difflab.core.imaging import megapixel_size, resize_to
from difflab.core.keys import key
from difflab.core.settings import EditSettings
from difflab.core.types import Frame
from difflab.models.qwen_edit import Editor
from difflab.models.qwen_image21 import Editor21

VAE_REFERENCE_AREA = 1024 * 1024
"""The pixel area QwenImageEditPlus encodes every input at for its VAE reference latents. Not a knob:
a fact about the pipeline, which `vae_reference_size` exists only to agree with."""


_PICTURE = re.compile(r"\bPicture (\d+)\b")


def named(instruction: str, settings: EditSettings, pictures: int) -> str:
    """Pictures as `engine` was trained to read them: Edit-2511's "Picture 2" is Qwen-Image 2.1's
    `<image2>`, the tag its pipeline puts before each image, and a single image goes untagged."""
    if settings.engine != "qwen-image-2.1":
        return instruction
    if pictures == 1:
        return _PICTURE.sub("the image", instruction)
    return _PICTURE.sub(lambda match: f"<image{match[1]}>", instruction)


def spec(settings: EditSettings) -> models.ModelSpec:
    """One registry entry per model file and LoRA stack, so two never share a slot."""
    if settings.engine == "qwen-image-2.1":
        weights, text_encoder = parse_file(settings.model_21), parse_file(settings.text_encoder_21)
        loras = [(parse_file(ref), strength) for ref, strength in settings.loras_21()]
        stack = ",".join(f"{lora.filename}*{strength:g}" for lora, strength in loras)
        return models.ModelSpec(
            name=f"qwen_image21:{weights.filename}:{text_encoder.filename}:{stack}",
            build=lambda: Editor21(weights, text_encoder, loras),
            checkpoints=(weights.path, text_encoder.path),
            unload=lambda editor: editor.unload(),
        )
    aio = HubFile.parse(settings.checkpoint)
    return models.ModelSpec(
        name=f"qwen_edit:{aio.filename}",
        build=lambda: Editor(aio),
        checkpoints=(aio.path,),
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
        model: Any = models.registry.acquire(spec(settings))
        resized = resize_to(frame.pil(), target)
        edited = model.edit(
            resized,
            named(instruction, settings, 1 + len(donors)),
            negative_prompt=negative_prompt,
            steps=settings.steps,
            cfg=settings.cfg,
            seed=settings.seed,
            donors=[donor.pil() for donor in donors],
            **({"turbo": settings.turbo} if settings.engine == "qwen-image-2.1" else {}),
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


def carried_size(frame: Frame, megapixels: float) -> tuple[int, int]:
    """A size Qwen-Image 2.1 resizes its condition image to unchanged, so an edit's latent is the next one's
    condition latent cell for cell: /32 on both axes, and a fixed point of the pipeline's `calculate_dimensions`."""
    width, height = (max(32, round(side / 32) * 32) for side in megapixel_size(frame.width, frame.height, megapixels))
    for _ in range(8):
        area, ratio = round(math.sqrt(width * height)) ** 2, width / height
        again = round(math.sqrt(area * ratio) / 32) * 32, round(math.sqrt(area * ratio) / ratio / 32) * 32
        if again == (width, height):
            return width, height
        width, height = again
    raise ValueError(f"no size near {megapixel_size(frame.width, frame.height, megapixels)} holds its condition")


def edit_carried(
    frame: Frame, latent: torch.Tensor | None, instruction: str, settings: EditSettings, size: tuple[int, int]
) -> tuple[Frame, torch.Tensor]:
    """`edit` on Qwen-Image 2.1 that hands back the edit's latent, and takes `frame`'s as `latent` instead of
    encoding its pixels again. `size` is a `carried_size`; `frame` is already at it after the first edit."""
    made = key(
        "edit_carried", 1, frame=frame.pixels, latent=latent, instruction=instruction, settings=settings, size=size
    )
    picture, kept = default_cache.path(made, ".png"), default_cache.path(made, ".pt")
    if default_cache.enabled and picture.exists() and kept.exists():
        ui.note(f"cached {made}")
        return Frame.load_png(picture), torch.load(kept, map_location="cpu", weights_only=True)
    model: Any = models.registry.acquire(spec(settings))
    edited, own = model.edit_latent(
        resize_to(frame.pil(), size), named(instruction, settings, 1), steps=settings.steps, cfg=settings.cfg,
        seed=settings.seed, turbo=settings.turbo, carried=latent,
    )  # fmt: skip
    out = Frame.from_pil(edited, origin=f"edited({frame.origin})" if frame.origin else "edited")
    if default_cache.enabled:
        out.pil().save(picture)
        torch.save(own, kept)
    return out, own


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
