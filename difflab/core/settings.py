"""Per-call parameters, as values rather than module globals, so two jobs in one process can differ.

Every field is declared with `setting()`, which carries what a page needs to draw it and what a
key needs to know about it. Three layers resolve to one value: the default written here, a saved
override for the machine or a production, and the call. A tool records the resolved values it ran
with, so nothing points at a profile that has since changed.

They are frozen because keys are computed from them. Only `keyed` fields join a key; renaming a
class or a field moves nothing, since `key_part` is a dict of names and values and the tool that
owns it is what carries the version.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, Self, get_args, get_origin, get_type_hints


def setting(
    default: Any,
    label: str = "",
    hint: str = "",
    unit: str = "",
    choices: tuple[Any, ...] = (),
    low: float | None = None,
    high: float | None = None,
    keyed: bool = True,
    advanced: bool = False,
    role: str = "",
    multiple: bool = False,
) -> Any:
    """A dataclass field with the metadata the page and the keys read. `keyed=False` for a knob that
    changes how a run is shown rather than what it makes. Fields of one `role` name the same model in
    different tools, and the page sets them together; a `multiple` one holds a list of them, one
    `reference*strength` per line."""
    return dataclasses.field(
        default=default,
        metadata={
            "label": label,
            "hint": hint,
            "unit": unit,
            "choices": choices,
            "low": low,
            "high": high,
            "keyed": keyed,
            "advanced": advanced,
            "role": role,
            "multiple": multiple,
        },
    )


@dataclass(frozen=True)
class Settings:
    def with_(self, **changes: Any) -> Self:
        return replace(self, **changes)

    def data(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def key_part(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.metadata.get("keyed", True)}

    @classmethod
    def schema(cls) -> list[dict[str, Any]]:
        """One entry per field, in declaration order: what a form draws and what a key includes."""
        hints = get_type_hints(cls)
        out = []
        for f in dataclasses.fields(cls):
            meta = dict(f.metadata)
            kind, choices = _kind(hints.get(f.name, str), tuple(meta.pop("choices", ())))
            out.append(
                {
                    "name": f.name,
                    "type": kind,
                    "default": f.default,
                    "choices": list(choices),
                    **{k: v for k, v in meta.items() if k != "choices"},
                }
            )
        return out

    @classmethod
    def resolve(cls, *layers: Mapping[str, Any] | None) -> tuple[Self, list[str]]:
        """The defaults with each layer applied over the last. A key the class does not have, or a
        value outside a field's range or choices, is dropped and reported rather than raised: a saved
        profile is allowed to be stale. A caller that wants a refusal instead constructs the class
        directly and lets it raise."""
        hints = get_type_hints(cls)
        fields = {f.name: f for f in dataclasses.fields(cls)}
        values: dict[str, Any] = {}
        notes: list[str] = []
        for layer in layers:
            for name, value in (layer or {}).items():
                f = fields.get(name)
                if f is None:
                    notes.append(f"{cls.__name__}: no setting called {name!r}, dropped")
                    continue
                kind, choices = _kind(hints.get(name, str), tuple(f.metadata.get("choices", ())))
                try:
                    value = _coerce(value, kind)
                except (TypeError, ValueError):
                    notes.append(f"{cls.__name__}.{name}: {value!r} is not a {kind}, dropped")
                    continue
                low, high = f.metadata.get("low"), f.metadata.get("high")
                if choices and value not in choices:
                    notes.append(f"{cls.__name__}.{name}: {value!r} is not one of {list(choices)}, dropped")
                    continue
                if (low is not None and value < low) or (high is not None and value > high):
                    notes.append(f"{cls.__name__}.{name}: {value!r} is outside {low}..{high}, dropped")
                    continue
                values[name] = value
        return cls(**values), notes


def _kind(hint: Any, choices: tuple[Any, ...]) -> tuple[str, tuple[Any, ...]]:
    if get_origin(hint) is Literal:
        return "str", tuple(get_args(hint))
    optional = tuple(a for a in get_args(hint) if a is not type(None)) if get_origin(hint) is not None else ()
    base = optional[0] if len(optional) == 1 else hint
    return {int: "int", float: "float", bool: "bool"}.get(base, "str"), choices


def _coerce(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if kind == "int":
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(value)
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "bool":
        return bool(value)
    return str(value)


# The one language model, which sees and writes: Qwen3.8-27B, vision tower and MTP head kept,
# as turboderp's 4.0 bpw EXL3 (16.5 GB).
VLM = "turboderp/Qwen3.8-27B-exl3@113cf7ab958054860e43fb7f3063b1af19171095"


@dataclass(frozen=True)
class VlmSettings(Settings):
    """Qwen-VL, as `AILab_QwenVL_Advanced` ran it: 4-bit, max_tokens 2048, keep_model_loaded."""

    checkpoint: str = setting(VLM, "model", "a Hugging Face repo at a pinned commit", role="vlm")
    max_new_tokens: int = setting(2048, "max new tokens", "a cap, not a target", low=1, advanced=True)
    temperature: float = setting(0.7, "temperature", "lower repeats, higher invents", low=0, high=2, advanced=True)
    top_p: float = setting(
        0.8, "top p", "Qwen's own recommendation is 0.8 with temperature 0.7", low=0, high=1, advanced=True
    )
    top_k: int = setting(20, "top k", "rarely worth moving", low=0, advanced=True)
    seed: int = setting(-1, "seed", "-1 draws a new one each call; a fixed one repeats the words", advanced=True)

    def generation(self) -> dict[str, Any]:
        """What `Exl3VL.ask` takes: everything but the checkpoint, which chose the model."""
        chosen = {name: value for name, value in self.data().items() if name != "checkpoint"}
        if chosen["seed"] < 0:
            chosen["seed"] = random.randrange(2**32)
        return chosen


@dataclass(frozen=True)
class DirectorSettings(Settings):
    """The VLM the director asks with. Its sampling stays the code's, tuned with its prompts (`tools/direct`)."""

    checkpoint: str = setting(VLM, "model", "a Hugging Face repo at a pinned commit", role="vlm")


@dataclass(frozen=True)
class JudgeSettings(Settings):
    """The VLM as a continuity supervisor over every keyframe, clip and drawn reference."""

    checkpoint: str = setting(VLM, "model", "a Hugging Face repo at a pinned commit", role="vlm")
    # Measured on production 9's five clips (2026-10-04): 0.60-0.85, the same before and after its colour
    # was fixed, and 0.6 for two clips that show their shots. Above that, a blind re-seed re-renders
    # good clips; this catches a broken one. Never 0: every render is judged.
    gate: float = setting(0.5, "gate", "the score a keyframe or clip needs before it is made again", low=0.1, high=1.0)
    retries: int = setting(
        2,
        "retries",
        "keyframe attempts, told the judge's fixes, before a failing one goes to review; a clip goes there at once",
        low=0,
        high=5,
        keyed=False,
        advanced=True,
    )
    # 512 cut a clip's verdict off mid-string once it named strangers and fixes (production 14, 2026-10-04).
    max_new_tokens: int = setting(
        1536, "answer length", "a cap, in tokens", low=64, high=4096, keyed=False, advanced=True
    )


@dataclass(frozen=True)
class EditSettings(Settings):
    """Qwen-Image-Edit, as ComfyUI's ClownsharKSampler ran it: 8 steps, cfg 1, denoise 1, euler.
    `cfg <= 1` turns CFG off, and then the negative prompt is unused."""

    # The single-file AIO: transformer, text encoder and VAE. The repo also holds
    # "v23/Qwen-Rapid-AIO-SFW-v23.safetensors", the newer revision.
    checkpoint: str = setting(
        "Phr00t/Qwen-Image-Edit-Rapid-AIO@691024f438640508f8aa86414863fc15edfb8a84:v18/Qwen-Rapid-AIO-SFW-v18.safetensors",
        "model",
        "the AIO file: repo@commit:path",
        role="editor",
    )
    steps: int = setting(8, "steps", "the editor is distilled; eight is the whole of it", low=1, high=60, advanced=True)
    cfg: float = setting(1.0, "cfg", "at 1 the negative prompt is inactive", low=1, high=8, advanced=True)
    seed: int = setting(42, "seed", "the only thing that changes between two runs of one instruction", advanced=True)
    megapixels: float = setting(
        2.0, "size", "the picture is resized to this many pixels before editing", unit="MP", low=0.25, high=4
    )


@dataclass(frozen=True)
class EnlargePixelsSettings(Settings):
    """An ESRGAN over finished frames."""

    # 4xLSDIR measured best on H3's output on every axis - detail, fidelity and temporal steadiness - see
    # `models/esrgan.py` for the table. 4xNMKDSuperscale shimmers, avoid it. Unported: SEEDVR2, a 7B
    # diffusion video restorer.
    checkpoint: str = setting(
        "nuriyoo/openmodeldb-mirror@1d24ff21e7f7a56d578ba5af0fe24f0264e413dd:models/4x-LSDIR/4xLSDIR.pth",
        "model",
        "an ESRGAN-family .pth: repo@commit:path",
        advanced=True,
        role="pixel-upscaler",
    )


@dataclass(frozen=True)
class RetimeSettings(Settings):
    """RIFE, for 48 or 60 fps out of a 24 fps render."""

    # ComfyUI's conversion of `RIFEv4.26_0921`, because the official release is a zip around a pickle.
    # 4.25 and 4.26 are the same architecture and the authors still recommend 4.25 for most scenes; 4.26
    # is what the reference workflow's node downloads, so it is what this reproduces.
    checkpoint: str = setting(
        "Comfy-Org/frame_interpolation@9bca6366a22473ccee25602fa82b224d78413960:frame_interpolation/rife_v4.26.safetensors",
        "model",
        "a RIFE 4.x safetensors: repo@commit:path",
        advanced=True,
        role="frame-interpolation",
    )


@dataclass(frozen=True)
class Continuity(Settings):
    """How one chunk of a continuous shot is joined to the next.

    Two mechanisms: the **handoff frame** decides where the next chunk opens (the sharpest of the last
    few beats always taking the final one, which is often mid-blur), and the **tail** hands over the
    frames leading up to it, which is the only way the model learns what was *moving* - a single frame
    carries pose but no velocity. The tail lands exactly one inter-frame gap before the new chunk's
    first frame, so it reads as the immediate past.

    The tail is priced in rows, and rows cost quadratic attention: at 15 s / 0.8 MP a 0.21 s tail is
    +1.8% of the packed sequence and about +3% of a render, 1 s is +6.5% and +12%, 2.3 s is +16% and
    +30%. 0.21 s is the grid's minimum (frame counts must satisfy `n % 17 == 5`) and five frames is
    already plenty to establish velocity - the community's LongTake measured 5 frames *better* than
    22 for reference adherence, while 22 is smoother and repeats the previous motion.

    What a chain loses anyway, measured by the same people: about +4% contrast and x1.3 texture per
    join, and audio treble. The countermeasures are a colour anchor to chunk 1 (holding its frame as
    a reference), one LUT and a little grain over the finished film, and short chains that restart at
    natural cuts.
    """

    tail_seconds: float = setting(
        0.25,
        "tail",
        "seconds of the previous chunk shown to the next; 0 disables it",
        unit="s",
        low=0,
        high=3,
        advanced=True,
    )
    handoff_window: int = setting(
        6,
        "handoff window",
        "frames searched at the end of a chunk for the sharpest one; 1 means the last",
        low=1,
        advanced=True,
    )
    crossfade: float = setting(
        0.0, "crossfade", "seconds of overlap blended rather than cut; at most the handoff window", unit="s", low=0
    )
    # A chain inherits its own colour. Chunk k+1 opens on chunk k's pixels, so a cast that enters once is
    # carried forward rather than corrected, and the reference crops cannot undo it: they are semantic
    # conditioning, while the opening frame is the pixels themselves. Measured over the five-chunk shisha
    # take, the red-minus-blue spread of the handoff frames ran 15.5, 18.0, 20.7, 20.9, 23.9 - up at every
    # seam, down at none, so the delivered take ends warmer and darker than the photograph it opened on.
    #
    # Keeping the finish out of that loop is the larger and more honest fix, because it removes lossy
    # steps rather than compensating for them. This is the belt to those braces: reach for it when a
    # take still drifts, and prefer a half to a whole, because the light in a scene is allowed to change
    # and a full match argues with it every time it does.
    color_anchor: float = setting(
        0.0,
        "colour anchor",
        "each handoff frame pulled toward the take's opening colour: 0.5 closes half the gap, 1 matches",
        low=0,
        high=1,
    )

    # The cause of the drift above, measured on production 9: each chunk inflates its own cast and
    # chroma as it goes, and the next opens on that. Held at the source, a chunk hands on the colour it
    # was given (`media.hold_colour`), capped so a real change of light or place survives.
    color_hold: float = setting(
        1.0,
        "colour hold",
        "each chunk's colour cast and contrast held to the frame it opened on: 0 off, 1 fully",
        low=0,
        high=1,
    )

    @property
    def tail_frames(self) -> int:
        """Frames of motion carried into the next chunk: 0 below the grid's five-frame minimum."""
        from difflab.models.minimax_h3 import pipeline  # here, not at the top: it imports torch

        return pipeline.tail_frames(self.tail_seconds)
