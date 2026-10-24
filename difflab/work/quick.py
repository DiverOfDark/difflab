"""The quick modes, as a value that fits in a request: a picture, one clip, an upscale.

Nothing here imports torch: the server checks a job without a model package, with the same `check`
the worker runs, so the Render button is disabled while anything is refused.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from difflab import config
from difflab.core.settings import EditSettings, VlmSettings
from difflab.models.ltx25 import config as ltx_config
from difflab.models.ltx25.settings import LtxSettings
from difflab.models.minimax_h3 import config as h3_config
from difflab.models.minimax_h3.settings import ReferenceSize, VideoPromptSettings, VideoSettings

# picture -> tools.edit_image, video -> tools.render_chunk, upscale -> tools.enlarge_pixels. A chain of
# shots is a production (`difflab.screenplay`), not a mode.
Mode = Literal["picture", "video", "upscale"]
# Mirrors `screenplay.production.FITS`: the photo centre-cropped to `aspect`, or kept whole and drawn out to it.
Fit = Literal["crop", "extend"]
VideoEngine = Literal["minimax-h3", "ltx-2.5"]
RETIRED = ("chain",)  # the chained, judged edit, removed 2026-10-17
# "create" and "edit" became one "picture" on 2026-10-17; every run before it recorded one of them.
MERGED = {"create": False, "edit": True}


@dataclass(frozen=True)
class Job:
    """One render, described. Give `idea` (the VLM writes the words) or `prompt` (they are written).
    A clip opens on `photos[0]`; a picture starts from it only when `start` is set. The rest are
    references, in the order given."""

    mode: Mode
    idea: str = ""
    prompt: str = ""
    photos: tuple[Path, ...] = ()
    start: bool = False
    clip: Path | None = None  # the mp4 to enlarge
    # The finish an upscale delivers at, in megapixels.
    finish_megapixels: float | None = None
    # `tools.grade` over the delivered file: how hard it is pulled back to the colour it opened in,
    # or `None` to leave it as rendered.
    grade_strength: float | None = None
    # Frames per second to deliver at, or `None` to leave the render at H3's own 24. A number above
    # it runs `tools.retime` over the finished file.
    retime_fps: int | None = None
    # "W:H". A created picture's: empty, the writer picks one for an idea and a prompt is drawn square. An
    # edit's or a take's: the photo is brought to it by `fit`; empty keeps the photo's own.
    aspect: str = ""
    fit: Fit = "crop"
    video_engine: VideoEngine = "minimax-h3"
    video: VideoSettings = field(default_factory=VideoSettings)
    ltx: LtxSettings = field(default_factory=LtxSettings)
    edit: EditSettings = field(default_factory=EditSettings)
    vlm: VlmSettings = field(default_factory=VlmSettings)  # writes the edit instruction
    video_vlm: VideoPromptSettings = field(default_factory=VideoPromptSettings)  # writes the H3 caption

    def __post_init__(self) -> None:
        """The idea as one paragraph; the prompt exactly as written, since a finished H3 caption has
        sections and its line breaks are part of it."""
        object.__setattr__(self, "idea", " ".join(self.idea.split()))

    # --- what can be known without a GPU ---------------------------------------------------------
    def check(self) -> list[str]:
        """Everything wrong that a file and a folder can say; empty means the worker reaches a model."""
        chosen = (
            ("mode", self.mode, get_args(Mode)),
            ("fit", self.fit, get_args(Fit)),
            ("video_engine", self.video_engine, get_args(VideoEngine)),
            ("video.reference_size", self.video.reference_size, get_args(ReferenceSize)),
        )
        unknown = [
            f"{name}={value!r}: one of {', '.join(allowed)}" for name, value, allowed in chosen if value not in allowed
        ]
        if unknown:
            return unknown

        problems = []
        if self.mode in ("picture", "video") and bool(self.idea) == bool(self.prompt):
            problems.append("give exactly one of idea / prompt")
        if self.aspect and not re.fullmatch(r"[1-9]\d?:[1-9]\d?", self.aspect):
            problems.append(f"aspect={self.aspect!r}: width:height, such as 3:2, or empty")
        if self.mode == "picture":
            if self.start and not self.photos:
                problems.append(
                    "no photo to edit: add one, or pick qwen-image-2.1 to draw from the words alone"
                    if self.edit.engine != "qwen-image-2.1"
                    else "no start photo: add one, or draw from the words and references alone"
                )
            elif not self.start and self.edit.engine == "minimax-h3":
                problems.append("minimax-h3 plays a clip out from a start photo: mark one, or pick a Qwen engine")
        elif self.mode == "upscale":
            if self.clip is None:
                problems.append("upscale needs a clip to enlarge")
            elif not self.clip.is_file():
                problems.append(f"no such video: {self.clip}")
            if self.finish_megapixels is None:
                problems.append("upscale needs finish_megapixels: the size to deliver at")
        elif not self.photos:
            problems.append("no photos: the first one is what the result opens on")
        elif self.video_engine == "ltx-2.5" and len(self.photos) - 1 > ltx_config.MSR_MAX_REFERENCES:
            problems.append(
                f"ltx-2.5 places at most {ltx_config.MSR_MAX_REFERENCES} references after the opening photo"
            )
        elif self.video_engine == "ltx-2.5" and len(self.photos) > 1 and self.ltx.version() == ltx_config.V23:
            problems.append("the LTX model is 2.3, which has no MSR: references need a 2.5 model")
        problems += [f"no such photo: {photo}" for photo in self.photos if not photo.is_file()]
        if self.finish_megapixels is not None and self.finish_megapixels <= 0:
            problems.append(f"finish_megapixels={self.finish_megapixels:g}: leave it unset to deliver as sampled")
        if self.grade_strength is not None:
            if self.mode == "picture":
                problems.append("grade_strength: a picture has no chain to drift along")
            elif not 0.0 < self.grade_strength <= 1.0:
                problems.append(
                    f"grade_strength={self.grade_strength:g}: between 0 and 1, or leave it unset - "
                    "it is the exponent, so 0.5 closes half the gap and 1 matches the opening outright"
                )
        if self.retime_fps is not None:
            if self.mode == "picture":
                problems.append("retime_fps: a picture has no frame rate")
            elif self.retime_fps <= h3_config.FPS:
                # Retiming only adds frames, and asking for 24 out of 24 is a whole pass over the
                # video to write it back unchanged.
                problems.append(f"retime_fps={self.retime_fps}: deliver above {h3_config.FPS}, or leave it unset")
        return problems

    @property
    def title(self) -> str:
        """What the run is called in the terminal and the page's list."""
        named = self.clip or (self.photos[0] if self.photos else None)
        if named:
            return f"{self.mode} · {named.stem}"
        words = " ".join((self.idea or self.prompt).split()[:8])
        return f"{self.mode} · {words}" if words else self.mode

    # --- on the wire -----------------------------------------------------------------------------
    def data(self) -> dict[str, Any]:
        """The job as JSON-able data. Nested settings become objects of their own fields."""
        data = asdict(self)
        data["photos"] = [str(photo) for photo in self.photos]
        data["clip"] = str(self.clip) if self.clip else None
        return data

    @classmethod
    def of(cls, data: Mapping[str, Any]) -> Job:
        """A job from data, refusing keys it does not know: a typo, or a field since renamed, either of
        which would otherwise render something other than what was asked for. A retired field is dropped: every
        run before 2026-10-17 recorded `chain`, and "again" on one sends its spec back."""
        data = {name: value for name, value in data.items() if name not in RETIRED}
        if data.get("mode") in MERGED:
            data = {**data, "start": MERGED[data["mode"]], "mode": "picture"}
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"a job has no {', '.join(unknown)} (known: {', '.join(cls.__dataclass_fields__)})")
        fields = dict(data)
        fields["photos"] = tuple(_under_output(photo) for photo in fields.get("photos", ()))
        if "clip" in fields:
            # The form writes "" into a cleared clip, and a `str` there is an AttributeError inside
            # `check` rather than the problem string the caller is waiting for.
            fields["clip"] = _under_output(fields["clip"]) if fields["clip"] else None
        for name, kind in (
            ("video", VideoSettings),
            ("ltx", LtxSettings),
            ("edit", EditSettings),
            ("vlm", VlmSettings),
            ("video_vlm", VideoPromptSettings),
        ):
            if name in fields:
                fields[name] = kind(**fields[name])
        return cls(**fields)


def _under_output(path: str | Path) -> Path:
    """Relative paths are under output/: that is how the run record names what it made."""
    given = Path(path)
    return given if given.is_absolute() else config.OUTPUT_DIR / given
