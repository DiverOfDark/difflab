"""A rough idea and a frame into the finished words a model is handed: an H3 caption, an LTX prompt or an
edit instruction."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from difflab import config
from difflab import prompts as edit_prompts
from difflab.core.settings import VlmSettings
from difflab.core.types import Frame
from difflab.models.exl3vl import ANY_JSON
from difflab.models.ltx25 import config as ltx_config
from difflab.models.ltx25 import prompts as ltx_prompts
from difflab.models.minimax_h3 import caption
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
        effort=config.VLM_WRITING_EFFORT,
        **settings.generation(),
    )
    if not written:
        raise RuntimeError("the VLM returned an empty prompt")
    return written


def write_ltx_prompt(
    frame: Frame,
    idea: str,
    references: Sequence[Frame] = (),
    settings: VlmSettings | None = None,
    version: str = ltx_config.V25,
) -> str:
    """A rough idea, the opening frame and any MSR references -> one finished LTX prompt, in its version's style."""
    settings = settings or VIDEO_VLM
    if version == ltx_config.V23:
        system = ltx_prompts.VLM_SYSTEM_PROMPT_23_REFERENCES if references else ltx_prompts.VLM_SYSTEM_PROMPT_23
    else:
        system = ltx_prompts.VLM_SYSTEM_PROMPT_REFERENCES if references else ltx_prompts.VLM_SYSTEM_PROMPT
    written = _vlm(settings).ask(
        [frame.pil(), *(reference.pil() for reference in references)],
        idea,
        system=system,
        title="LTX prompt",
        effort=config.VLM_WRITING_EFFORT,
        **settings.generation(),
    )
    if not written:
        raise RuntimeError("the VLM returned an empty prompt")
    return written


def _parts(pictures: int) -> dict[str, Any]:
    each = {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": pictures, "maxItems": pictures}
    text = {"type": "string", "minLength": 1}
    fields = {"pictures": each, "kept": each, "summary": text, "style": text, "shot": text, "sound": text}
    return {**ANY_JSON, "type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def write_reference_caption(frames: Sequence[Frame], idea: str, settings: VlmSettings | None = None) -> str:
    """The photos and an edit -> the six-section ref2va caption of the photo already edited."""
    settings = settings or VIDEO_VLM
    raw = _vlm(settings).ask(
        [frame.pil() for frame in frames],
        idea,
        system=video_prompts.VLM_SYSTEM_PROMPT_EDIT_REF,
        title="H3 caption",
        as_json=True,
        schema=_parts(len(frames)),
        effort=config.VLM_WRITING_EFFORT,
        **settings.generation(),
    )
    data = json.loads(raw)
    labels = caption.labels(len(frames), tail=False)
    retention = [
        f"{label} (appears in [Shot 1]): partially_preserved - {kept.strip()}"
        for label, kept in zip(labels, data["kept"], strict=True)
    ]
    text = (
        "subject_definitions:\n" + "\n".join(line.strip() for line in data["pictures"]) + "\n\n"
        f"summary:\n{caption.directive(tail=False, framed=False)} {data['summary'].strip()}\n\n"
        "retention_analysis:\n" + "\n".join(retention) + "\n\n"
        f"detailed_description:\n{data['style'].strip()}\n[Shot 1] {data['shot'].strip()}\n\n"
        f"overall_soundscape:\n{data['sound'].strip()}\n\n"
        "non_diegetic_music:\nnone\n"
    )
    wrong = caption.errors(caption.ide_rules(text) + caption.label_rules(text, labels))
    if wrong:
        raise RuntimeError("the VLM's caption breaks H3's format: " + "; ".join(p.message for p in wrong))
    return text


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
        effort=config.VLM_WRITING_EFFORT,
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
        effort=config.VLM_WRITING_EFFORT,
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
