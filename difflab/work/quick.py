"""The three quick modes, as a value that fits in a request: a photo restyled, one clip, an upscale.

Nothing here imports torch: the server checks a job without a model package, with the same `check`
the worker runs, so the Render button is disabled while anything is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from difflab import config
from difflab.core.settings import EditSettings, VlmSettings
from difflab.models.minimax_h3 import config as h3_config
from difflab.models.minimax_h3.settings import ReferenceSize, VideoPromptSettings, VideoSettings

# edit -> tools.edit_image, video -> tools.render_chunk, upscale -> tools.enlarge_pixels. A chain of
# shots is a production (`difflab.screenplay`), not a mode.
Mode = Literal["edit", "video", "upscale"]


@dataclass(frozen=True)
class Job:
    """One render, described. Give `idea` (the VLM writes the words) or `prompt` (they are written).
    `photos[0]` is what the result opens on; the rest are references, in the order given."""

    mode: Mode
    idea: str = ""
    prompt: str = ""
    photos: tuple[Path, ...] = ()
    clip: Path | None = None  # the mp4 to enlarge
    # The finish an upscale delivers at, in megapixels.
    finish_megapixels: float | None = None
    # `tools.grade` over the delivered file: how hard it is pulled back to the colour it opened in,
    # or `None` to leave it as rendered.
    grade_strength: float | None = None
    # Frames per second to deliver at, or `None` to leave the render at H3's own 24. A number above
    # it runs `tools.retime` over the finished file.
    retime_fps: int | None = None
    video: VideoSettings = field(default_factory=VideoSettings)
    # An edit's idea split into its logical edits, made one at a time and each judged (`tools/chain_edit`).
    chain: bool = False
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
            ("video.reference_size", self.video.reference_size, get_args(ReferenceSize)),
        )
        unknown = [
            f"{name}={value!r}: one of {', '.join(allowed)}" for name, value, allowed in chosen if value not in allowed
        ]
        if unknown:
            return unknown

        problems = []
        if self.mode in ("edit", "video") and bool(self.idea) == bool(self.prompt):
            problems.append("give exactly one of idea / prompt")
        if self.chain and self.mode != "edit":
            problems.append("chain: only an edit is made in steps")
        elif self.chain and not self.idea:
            problems.append("chain needs an idea: the VLM splits it into edits")
        if self.mode == "upscale":
            if self.clip is None:
                problems.append("upscale needs a clip to enlarge")
            elif not self.clip.is_file():
                problems.append(f"no such video: {self.clip}")
            if self.finish_megapixels is None:
                problems.append("upscale needs finish_megapixels: the size to deliver at")
        elif not self.photos:
            problems.append("no photos: the first one is what the result opens on")
        problems += [f"no such photo: {photo}" for photo in self.photos if not photo.is_file()]
        if self.finish_megapixels is not None and self.finish_megapixels <= 0:
            problems.append(f"finish_megapixels={self.finish_megapixels:g}: leave it unset to deliver as sampled")
        if self.grade_strength is not None:
            if self.mode == "edit":
                problems.append("grade_strength: a photo has no chain to drift along")
            elif not 0.0 < self.grade_strength <= 1.0:
                problems.append(
                    f"grade_strength={self.grade_strength:g}: between 0 and 1, or leave it unset - "
                    "it is the exponent, so 0.5 closes half the gap and 1 matches the opening outright"
                )
        if self.retime_fps is not None:
            if self.mode == "edit":
                problems.append("retime_fps: an edited photo has no frame rate")
            elif self.retime_fps <= h3_config.FPS:
                # Retiming only adds frames, and asking for 24 out of 24 is a whole pass over the
                # video to write it back unchanged.
                problems.append(f"retime_fps={self.retime_fps}: deliver above {h3_config.FPS}, or leave it unset")
        return problems

    @property
    def title(self) -> str:
        """What the run is called in the terminal and the page's list."""
        named = self.clip or (self.photos[0] if self.photos else None)
        return f"{self.mode} · {named.stem}" if named else self.mode

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
        which would otherwise render something other than what was asked for."""
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
