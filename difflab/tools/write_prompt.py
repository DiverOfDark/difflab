"""A rough idea and a frame into the finished words a model is handed: an H3 caption or an edit
instruction."""

from __future__ import annotations

import json
import re

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


def write_video_prompt(frame: Frame, idea: str, settings: VlmSettings | None = None) -> str:
    """A rough idea plus the frame -> one finished MiniMax H3 prompt."""
    settings = settings or VIDEO_VLM
    written = _vlm(settings).ask(
        frame.pil(),
        idea,
        system=video_prompts.VLM_SYSTEM_PROMPT,
        title="H3 prompt",
        **settings.generation(),
    )
    if not written:
        raise RuntimeError("the VLM returned an empty prompt")
    return written


def write_edit_instruction(
    frame: Frame, idea: str, settings: VlmSettings | None = None, engine: str = "qwen-image-edit"
) -> str:
    """A rough idea plus the frame -> one finished instruction, in the words `engine` was trained on."""
    settings = settings or VlmSettings()
    written = _vlm(settings).ask(
        frame.pil(),
        idea,
        system=edit_prompts.VLM_SYSTEM_PROMPT_21 if engine == "qwen-image-2.1" else edit_prompts.VLM_SYSTEM_PROMPT,
        title="edit instruction",
        **settings.generation(),
    )
    if not written:
        raise RuntimeError("the VLM returned an empty instruction")
    return written


def write_image_prompt(idea: str, settings: VlmSettings | None = None) -> tuple[str, str]:
    """A rough idea -> the description Qwen-Image 2.1 draws from, and the aspect ratio it was written for."""
    settings = settings or VlmSettings()
    raw = _vlm(settings).ask(
        None,
        idea,
        system=edit_prompts.VLM_SYSTEM_PROMPT_CREATE,
        title="picture description",
        as_json=True,
        schema=DESCRIBED,
        **settings.generation(),
    )
    data = json.loads(raw)
    written = str(data["rewritten_prompt"]).strip()
    if not written:
        raise RuntimeError("the VLM returned an empty description")
    ratio = str(data["wh_ratio"])
    return written, ratio if RATIO.match(ratio) else "1:1"
