"""A rough idea and a frame into the finished words a model is handed: an H3 caption, an edit
instruction, or a storyboard of beats."""

from __future__ import annotations

from difflab import prompts as edit_prompts
from difflab.core.settings import VlmSettings
from difflab.core.types import Frame, Storyboard
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


# The ComfyUI i2v writer difflab started from, and it writes the format H3 does *not* want - one
# 60-130 word paragraph per beat, no headings, no picture named, split on `---` without checking the
# count. the director tools write MiniMax's own caption format and are what a production uses;
# this is kept because an idea and one line is the shortest thing that renders.
def write_storyboard(
    frame: Frame,
    idea: str,
    beats: int,
    seconds_per_beat: float = 15.0,
    settings: VlmSettings | None = None,
    continuous: bool = False,
) -> Storyboard:
    """Plan a long video up front: `beats` prompts that continue one another.

    Written in one call rather than one per chunk, so the VLM can see the whole arc while it decides
    where each beat starts and ends. Each beat is still a self-contained H3 prompt, because the
    renderer hands them over one at a time with only the previous chunk's last frame for continuity.
    """
    settings = settings or VIDEO_VLM
    if continuous:
        instruction = (
            f"{idea}\n\n"
            f"This is ONE CONTINUOUS SHOT of about {beats * seconds_per_beat:g} seconds, with no cuts. "
            f"Break it into {beats} consecutive segments of about {seconds_per_beat:g} seconds each. "
            f"Each segment picks up at the exact instant the previous one ended, mid-motion. "
            f"Keep one camera move running across the whole take and do not restate it each time. "
            f"Do not re-describe the location, wardrobe or lighting after the first segment - say only "
            f"what changes. Do not start a segment with the subject at rest if the previous one left "
            f"them moving. Write each segment as a complete prompt following the rules above, and give "
            f"each one its own audio line. Separate them with a line containing only ---, nothing else."
        )
    else:
        instruction = (
            f"{idea}\n\n"
            f"Plan this as {beats} consecutive shots of about {seconds_per_beat:g} seconds each, in order. "
            f"Each shot continues from the last frame of the one before it, so keep the subject, wardrobe, "
            f"location and light consistent and let only the action and camera move on. "
            f"Write each shot as a complete standalone prompt following the rules above. "
            f"Separate them with a line containing only ---, and write nothing else."
        )
    written = _vlm(settings).ask(
        frame.pil(),
        instruction,
        system=video_prompts.VLM_SYSTEM_PROMPT,
        title=f"storyboard ({beats} beats{', continuous' if continuous else ''})",
        **settings.generation(),
    )
    parts = [part.strip() for part in written.split("---") if part.strip()]
    if not parts:
        raise RuntimeError("the VLM returned an empty storyboard")
    return Storyboard(beats=parts, title=idea[:60])
