"""A rough idea and a frame into the finished words a model is handed: an H3 caption or an edit
instruction."""

from __future__ import annotations

from difflab import prompts as edit_prompts
from difflab.core.settings import VlmSettings
from difflab.core.types import Frame
from difflab.models.minimax_h3 import prompts as video_prompts
from difflab.models.minimax_h3.settings import VideoPromptSettings
from difflab.tools.ask_vlm import _vlm

VIDEO_VLM = VideoPromptSettings()


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


def write_edit_instruction(frame: Frame, idea: str, settings: VlmSettings | None = None) -> str:
    """A rough idea plus the frame -> one finished Qwen-Image-Edit instruction."""
    settings = settings or VlmSettings()
    written = _vlm(settings).ask(
        frame.pil(),
        idea,
        system=edit_prompts.VLM_SYSTEM_PROMPT,
        title="edit instruction",
        **settings.generation(),
    )
    if not written:
        raise RuntimeError("the VLM returned an empty instruction")
    return written
