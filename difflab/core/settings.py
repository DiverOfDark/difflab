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
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
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
    `reference*strength` per line. `DIFFLAB_MODEL_<ROLE>` in the environment or `.env` replaces a role's
    default on one machine."""
    if role and (chosen := os.environ.get(f"DIFFLAB_MODEL_{role.upper().replace('-', '_')}")) is not None:
        default = chosen
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


def lora_stack(text: str) -> list[tuple[str, float]]:
    """A `multiple` field's lines as (reference, strength), without the ones at strength 0."""
    out = []
    for line in text.splitlines():
        reference, _, strength = line.strip().partition("*")
        if reference and (value := float(strength or 1.0)) != 0.0:
            out.append((reference.strip(), value))
    return out


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
# as turboderp's 4.0 bpw EXL3 (16.9 GB).
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
    # was fixed, and 0.6 for two clips that show their shots. Above that, a retake re-renders
    # good clips; this catches a broken one. Never 0: every render is judged.
    gate: float = setting(0.5, "gate", "the score a keyframe or clip needs before it is made again", low=0.1, high=1.0)
    retries: int = setting(
        2,
        "retries",
        "keyframe, look and drawn-reference attempts, told the judge's fixes, before a failing one goes to review",
        low=0,
        high=5,
        keyed=False,
        advanced=True,
    )
    clip_retry: bool = setting(
        True,
        "auto-retry clips",
        "a clip below the gate is rendered again on new seeds and the best take kept; off, it pauses the run",
        keyed=False,
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
    engine: Literal["qwen-image-edit", "qwen-image-2.1"] = setting(
        "qwen-image-edit",
        "engine",
        "qwen-image-2.1 is experimental: its own model, text encoder and LoRAs below; the AIO model is unused",
    )
    # Qwen's 8-step Turbo as Comfy-Org repacked it. The base model is
    # "diffusion_models/qwen_image_2.1_int8_convrot.safetensors" in the same repo, with `turbo` off.
    model_21: str = setting(
        "Comfy-Org/Qwen-Image-2.1@df94239739eef7205973a145ffa6f441f7b64e84"
        ":diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
        "model",
        "the int8 ConvRot Qwen-Image 2.1 transformer, a fine-tune's included: repo@commit:path or civitai:…",
        role="editor-21",
    )
    text_encoder_21: str = setting(
        "Comfy-Org/Qwen-Image-2.1@df94239739eef7205973a145ffa6f441f7b64e84"
        ":text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "text encoder",
        "the int8 ConvRot Qwen3-VL 8B: repo@commit:path",
        advanced=True,
        role="editor-21-text-encoder",
    )
    loras: str = setting(
        "",
        "LoRAs",
        "stacked on the Qwen-Image 2.1 model in order, as reference*strength",
        role="editor-21-loras",
        multiple=True,
    )
    # The sigmas are Qwen-Image-2.1-Turbo's `sample_sigmas`, the only schedule Qwen evaluated it on.
    turbo: bool = setting(
        True, "turbo", "Qwen-Image 2.1 on the Turbo's own 8 sigmas, and steps is unused; off for the base model"
    )

    @property
    def model_stem(self) -> str:
        reference = self.model_21 if self.engine == "qwen-image-2.1" else self.checkpoint
        return Path(reference.rsplit(":", 1)[-1]).stem  # repo@commit:path and civitai:version@sha:file alike

    def loras_21(self) -> list[tuple[str, float]]:
        return lora_stack(self.loras)

    def key_part(self) -> dict[str, Any]:
        # The AIO engine keys as it did before the 2.1 fields existed, so nothing it drew is drawn again.
        part = super().key_part()
        mine = ("engine", "model_21", "text_encoder_21", "loras", "turbo")
        if self.engine == "qwen-image-edit":
            return {name: value for name, value in part.items() if name not in mine}
        part.pop("checkpoint")
        part["loras"] = self.loras_21()
        if self.turbo:
            part.pop("steps")
        return part


@dataclass(frozen=True)
class ChainSettings(Settings):
    """A quick edit made one logical change at a time, each judged against the original photo."""

    retries: int = setting(
        2,
        "retries",
        "redos of an edit the judge fails, each told what it found, before the chain carries on with it flagged",
        low=0,
        high=5,
        keyed=False,
    )
    # Beach test, 4 edits on Qwen-Image 2.1 (2026-10-15): from decoded pictures the sky turned to painterly noise by
    # the third edit and two edits stayed flagged; carried, every edit passed and the sky stayed clean, in the same
    # 660 s.
    carry: bool = setting(
        True,
        "carry latents",
        "Qwen-Image 2.1: each edit is conditioned on the last one's latent, never on its decoded and re-encoded "
        "picture; the size snaps to /32. The AIO always edits from pictures",
        advanced=True,
    )
    body: bool = setting(
        True,
        "body check",
        "the people's traits read from the photo (face, eyes, hair, build, bust, waist, hips, marks) and checked on "
        "every picture; a fifth longer or wider body still passes",
        advanced=True,
    )


@dataclass(frozen=True)
class IngestSettings(Settings):
    """How a story becomes chapters, and a reference video becomes pictures."""

    heading_level: int = setting(
        1, "chapter split", "a heading of this level or above starts a chapter (1 is #, 2 is ##)", low=1, high=6
    )
    frames: int = setting(12, "frames per video", "frames pulled from each reference video, evenly", low=1, high=48)
    # Variance of the Laplacian at 512 px wide (`ingest.sharpness`), measured 2026-10-09: an H3 frame 2140,
    # Gaussian-blurred sigma 1/2/3 750/130/36; a sharp studio portrait 809, blurred 96/12/5; a soft phone
    # portrait 91. 50 drops a portrait blurred at sigma 2 and a detailed frame at sigma 3. The texture of
    # the picture moves it as much as the blur does, so a video's sharpest frame is kept whatever it measures.
    min_sharpness: float = setting(50.0, "min sharpness", "frames blurrier than this are dropped", low=0, high=5000)


@dataclass(frozen=True)
class EnlargePixelsSettings(Settings):
    """An ESRGAN over finished frames."""

    # 4xLSDIR measured best on H3's output on every axis - detail, fidelity and temporal steadiness - see
    # `models/esrgan.py` for the table. 4xNMKDSuperscale shimmers, avoid it.
    checkpoint: str = setting(
        "nuriyoo/openmodeldb-mirror@1d24ff21e7f7a56d578ba5af0fe24f0264e413dd:models/4x-LSDIR/4xLSDIR.pth",
        "model",
        "an ESRGAN-family .pth: repo@commit:path",
        advanced=True,
        role="pixel-upscaler",
    )


@dataclass(frozen=True)
class RestoreSettings(Settings):
    """SeedVR2 7B to half the finish's size, then a 2x ESRGAN the rest of the way."""

    checkpoint: str = setting(
        "Comfy-Org/SeedVR2@df48879708206a403d2a61acd55578c2e80fd233:diffusion_models/seedvr2_7b_int8_convrot.safetensors",
        "model",
        "SeedVR2 7B as int8 ConvRot: repo@commit:path",
        advanced=True,
        role="restorer",
    )
    vae: str = setting(
        "Comfy-Org/SeedVR2@df48879708206a403d2a61acd55578c2e80fd233:vae/seedvr2_ema_vae_fp16.safetensors",
        "VAE",
        "SeedVR2's causal 3D VAE: repo@commit:path",
        advanced=True,
        role="restorer-vae",
    )
    # RealESRGAN x2 rather than 4xLSDIR: from 1080p a 2x lands on 3840x2160 exactly, at 0.14 s a frame
    # against LSDIR's 0.63 for 0.05 less sharpness (`models/esrgan.py`).
    upscaler: str = setting(
        "ai-forever/Real-ESRGAN@8110204ebf8d25c031b66c26c2d1098aa831157e:RealESRGAN_x2.pth",
        "upscaler",
        "a 2x ESRGAN-family .pth: repo@commit:path",
        advanced=True,
        role="pixel-upscaler",
    )
    frames: int = setting(33, "window", "frames restored together; neighbours share the overlap", low=5, high=81)
    overlap: int = setting(4, "overlap", "frames two windows share, cross-faded", low=0, high=16)
    seed: int = setting(42, "seed", "the restorer's noise; the same for every window", advanced=True)


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
    +30%. 0.21 s (five frames, the grid's minimum: `n % 17 == 5`) carries pose and a hint of velocity,
    and with it every chunk eased in from near-stillness. 0.92 s is 22 frames, which the community's
    LongTake measured smoother than 5, at the cost of some reference adherence and a tendency to repeat
    the previous motion; the plan's carry joins need the speed to come across.

    What a chain loses anyway, measured by the same people: about +4% contrast and x1.3 texture per
    join, and audio treble. The countermeasures are a colour anchor to chunk 1 (holding its frame as
    a reference), one LUT and a little grain over the finished film, and short chains that restart at
    natural cuts.
    """

    tail_seconds: float = setting(
        0.92,
        "tail",
        "seconds of the previous chunk shown to the next; 0 disables it",
        unit="s",
        low=0,
        high=3,
        advanced=True,
    )
    # A shot that continues the one before opens on that shot's last latent steps and sound, pinned,
    # instead of a re-encoded handoff frame shown its tail as a reference. Measured over five chained
    # runs (2026-10-05..12), the re-encoded handoff raised the latents' high band 2-4% at every join and
    # never within a chunk: x1.21 after ten shots.
    latent_carry: bool = setting(
        True,
        "carry latents",
        "a continuing shot opens on the last steps and sound of the one before, pinned, not on a re-encoded frame",
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
        0.1,
        "seam blend",
        "seconds a chained shot blends over the frames it shares with the shot before; at most the handoff window",
        unit="s",
        low=0,
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
        0.5,
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


Subtitles = Literal["separate track", "sidecar files", "off"]


@dataclass(frozen=True)
class AssembleSettings(Settings):
    """A chapter's joins, loudness and subtitles, and the film's chapter markers."""

    lufs: float = setting(-16.0, "loudness", "target per chapter, LUFS", unit="LUFS", low=-36, high=-6)
    crossfade: float = setting(0.25, "crossfade", "seconds at scene boundaries", unit="s", low=0, high=2)
    subtitles: Subtitles = setting("separate track", "subtitles", "dialogue lines")
    markers: bool = setting(True, "chapter markers", "in the full film")
