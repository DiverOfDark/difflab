"""The person an edit must keep, measured rather than asked: girths under the clothes (SAM 3D Body), who the face
is and the colour of the eyes (antelopev2).

On the 2026-10-17 probe (6 body and 5 face photos, 60 edits, planted warps) the VLM checklist caught no body
widened, slimmed or pinched by 10-20%; the girths caught 100% of a 20% widening and 89% of a 10% slimming with no
flag on a dress swap. ArcFace and the iris caught what the VLM let through on AIO face edits: eyes 20 -> 80%,
nose 0 -> 60%.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from difflab import config
from difflab.core import models
from difflab.core.cache import cache
from difflab.core.keys import key
from difflab.core.types import Frame

# What the edit may set out to change, in the words a ruling is asked about (`chain_edit.changed`).
TRAITS = {
    "bust": "the size of the person's bust",
    "waist": "the size of the person's waist",
    "hips": "the size of the person's hips",
    "identity": "the person's face and identity",
    "eyes": "the colour of the person's eyes",
}


@dataclass(frozen=True)
class Change:
    trait: str  # a key of TRAITS
    chance: float  # that the person in the edit differs here from the original
    measured: str


def changes(original: Frame, image: Frame) -> list[Change]:
    """Every trait both pictures let us measure, with the chance it changed."""
    out = []
    before, after = _girths(original), _girths(image)
    if before and after:
        for name in ("bust", "waist", "hips"):
            ratio = math.log(after[name] / before[name])
            chance = _odds(abs(ratio), config.LIKENESS_GIRTH_CUT, config.LIKENESS_GIRTH_SPREAD)
            out.append(Change(name, chance, f"its {name} girth measures {ratio:+.0%} against the original's"))
    was, now = _face(original), _face(image)
    if was and now:
        distance = 1 - float(np.dot(was["identity"], now["identity"]))
        chance = _odds(distance, config.LIKENESS_IDENTITY_CUT, config.LIKENESS_IDENTITY_SPREAD)
        out.append(Change("identity", chance, f"the face reads as another person's (ArcFace distance {distance:.2f})"))
        shifts = [
            math.dist(_against(a["colour"], was["skin"]), _against(b["colour"], now["skin"]))
            for a, b in zip(was["eyes"], now["eyes"], strict=True)
            if a and b and min(a["open"], b["open"]) >= config.LIKENESS_EYE_OPEN
        ]
        if shifts:
            chance = _odds(max(shifts), config.LIKENESS_EYE_CUT, config.LIKENESS_EYE_SPREAD)
            out.append(Change("eyes", chance, f"the irises' colour moved by {max(shifts):.0f} against the skin's"))
    return out


def _against(colour: list[float], skin: list[float]) -> tuple[float, ...]:
    """An iris's colour as its RGB over the skin's, in log units: a colour cast multiplies both by the same gains.
    In CIELAB, alone or less the skin's, an amber relight moved it past the cut on four clean edits of six."""
    ratio = [math.log(c / k) for c, k in zip(colour, skin, strict=True)]
    mean = sum(ratio) / 3
    return tuple(100 * (r - mean) for r in ratio)


def _odds(value: float, cut: float, spread: float) -> float:
    return 1 / (1 + math.exp(-(value - cut) / spread))


def _girths(frame: Frame) -> dict[str, float] | None:
    def measure() -> str:
        model: Any = models.registry.acquire(_body_spec())
        return json.dumps(model.girths(frame.pil()))

    loaded: dict[str, float] | None = json.loads(cache.text(key("likeness_body", 2, frame=frame.pixels), measure))
    return loaded


def _face(frame: Frame) -> dict[str, Any] | None:
    def measure() -> str:
        found = _faces().read(frame.pil())
        if found is None:
            return "null"
        eyes = [None if eye is None else {"colour": eye.colour, "open": eye.open} for eye in found.eyes]
        return json.dumps(
            {"identity": found.identity.tolist(), "eyes": eyes, "skin": found.skin, "height": found.height}
        )

    loaded: dict[str, Any] | None = json.loads(cache.text(key("likeness_face", 3, frame=frame.pixels), measure))
    return loaded


def _body_spec() -> models.ModelSpec:
    from difflab.models.body_shape import BodyShape

    return models.ModelSpec(
        name="sam3d_body",
        build=BodyShape,
        checkpoints=(config.BODY_CHECKPOINT.path,),
        unload=lambda model: model.unload(),
    )


_FACE_ID: Any = None


def _faces() -> Any:
    global _FACE_ID
    if _FACE_ID is None:
        from difflab.models.face_id import FaceId

        _FACE_ID = FaceId()
    return _FACE_ID
