"""A rough idea and a frame into the finished words a model is handed: an H3 caption or an edit
instruction."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from difflab import prompts as edit_prompts
from difflab.core.settings import VlmSettings
from difflab.core.types import Frame
from difflab.models.exl3vl import ANY_JSON
from difflab.models.minimax_h3 import prompts as video_prompts
from difflab.models.minimax_h3.settings import VideoPromptSettings
from difflab.tools.ask_vlm import _vlm

VIDEO_VLM = VideoPromptSettings()
RATIO = re.compile(r"^\s*(\d{1,2})\s*:\s*(\d{1,2})\s*$")
DESCRIBED = {
    **ANY_JSON,
    "type": "object",
    "properties": {"rewritten_prompt": {"type": "string", "minLength": 1}, "wh_ratio": {"type": "string"}},
    "required": ["rewritten_prompt", "wh_ratio"],
    "additionalProperties": False,
}
FOLLOWED = {
    **ANY_JSON,
    "type": "object",
    "properties": {
        "rewritten_prompt": {"type": "string", "minLength": 1},
        "wh_ratio": {"type": "string"},
        "ratio_follow": {"type": "string"},
    },
    "required": ["rewritten_prompt", "wh_ratio", "ratio_follow"],
    "additionalProperties": False,
}
TAG = re.compile(r"<image(\d+)>")


def write_video_prompt(
    frame: Frame, idea: str, settings: VlmSettings | None = None, system: str = video_prompts.VLM_SYSTEM_PROMPT
) -> str:
    """A rough idea plus the frame -> one finished MiniMax H3 prompt."""
    settings = settings or VIDEO_VLM
    written = _vlm(settings).ask(
        frame.pil(),
        idea,
        system=system,
        title="H3 prompt",
        **settings.generation(),
    )
    if not written:
        raise RuntimeError("the VLM returned an empty prompt")
    return written


def write_edit_instruction(
    frame: Frame,
    idea: str,
    settings: VlmSettings | None = None,
    engine: str = "qwen-image-edit",
    references: Sequence[Frame] = (),
) -> str:
    """A rough idea, the frame and its references -> one finished instruction, in the words `engine` was trained on."""
    settings = settings or VlmSettings()
    if engine != "qwen-image-2.1":
        system = edit_prompts.VLM_SYSTEM_PROMPT
    else:
        system = edit_prompts.VLM_SYSTEM_PROMPT_21_MANY if references else edit_prompts.VLM_SYSTEM_PROMPT_21
    written = _vlm(settings).ask(
        [frame.pil(), *(reference.pil() for reference in references)],
        idea,
        system=system,
        title="edit instruction",
        **settings.generation(),
    )
    if not written:
        raise RuntimeError("the VLM returned an empty instruction")
    return written


def write_image_prompt(
    idea: str, settings: VlmSettings | None = None, references: Sequence[Frame] = ()
) -> tuple[str, str]:
    """A rough idea and any references -> the description Qwen-Image 2.1 draws from, and the aspect ratio it was
    written for."""
    settings = settings or VlmSettings()
    raw = _vlm(settings).ask(
        [reference.pil() for reference in references] or None,
        idea,
        system=edit_prompts.VLM_SYSTEM_PROMPT_REFS if references else edit_prompts.VLM_SYSTEM_PROMPT_CREATE,
        title="picture description",
        as_json=True,
        schema=FOLLOWED if references else DESCRIBED,
        **settings.generation(),
    )
    data = json.loads(raw)
    written = str(data["rewritten_prompt"]).strip()
    if not written:
        raise RuntimeError("the VLM returned an empty description")
    ratio = str(data["wh_ratio"])
    if RATIO.match(ratio):
        return written, ratio
    followed = TAG.fullmatch(str(data.get("ratio_follow", "")).strip())
    if followed and 1 <= int(followed[1]) <= len(references):
        image = references[int(followed[1]) - 1]
        return written, f"{image.width}:{image.height}"
    return written, "1:1"
