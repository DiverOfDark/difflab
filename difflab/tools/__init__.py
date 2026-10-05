"""One module per tool: one model applied once, with settings that describe themselves.

This table is what the page and the store see of a tool: its name, the version of its key, the
tasks a job of it runs through, and the settings class whose schema draws its form. The modules
themselves import the models; this file imports none, so a server can list tools without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from difflab.core.settings import (
    Continuity,
    DirectorSettings,
    EditSettings,
    EnlargePixelsSettings,
    JudgeSettings,
    RetimeSettings,
    Settings,
    VlmSettings,
)
from difflab.models.minimax_h3.settings import VideoPromptSettings, VideoSettings


@dataclass(frozen=True)
class Tool:
    name: str
    key_version: int
    tasks: tuple[str, ...]
    settings: type[Settings] | None = None
    hint: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "key_version": self.key_version,
            "tasks": list(self.tasks),
            "hint": self.hint,
            "settings": self.settings.schema() if self.settings else [],
        }


TOOLS: tuple[Tool, ...] = (
    Tool("edit_image", 1, ("load editor", "encode", "denoise", "decode", "save"), EditSettings, "Qwen-Image-Edit"),
    Tool("ask_vlm", 1, ("load VLM", "answer"), VlmSettings, "one question about a picture"),
    Tool("write_instruction", 1, ("load VLM", "write"), VlmSettings, "an idea into an edit instruction"),
    Tool("write_video_prompt", 1, ("load VLM", "write"), VideoPromptSettings, "an idea into an H3 caption"),
    Tool("render_chunk", 1, ("load", "condition", "encode", "sample", "decode"), VideoSettings, "MiniMax H3"),
    Tool("enlarge_pixels", 1, ("load", "upscale"), EnlargePixelsSettings, "an ESRGAN over finished frames"),
    Tool("retime", 1, ("load", "interpolate"), RetimeSettings, "RIFE draws the frames between"),
    Tool("grade", 1, ("measure", "grade"), None, "a take pulled back to the colour it opened in"),
    Tool("handoff", 1, ("handoff",), Continuity, "the frame the next shot opens on, and the motion into it"),
    Tool("assemble", 1, ("join",), None, "a chapter's chunks as one video"),
    Tool("judge", 1, ("load VLM", "judge"), JudgeSettings, "does it show the shot, with the cast looking right"),
    Tool("compose_keyframe", 1, ("load editor", "compose"), EditSettings, "a first frame from the plan's words"),
    Tool("draw_reference", 1, ("load editor", "draw"), None, "a member no photo shows, drawn from its words"),
    Tool(
        "refine_idea",
        1,
        ("load VLM", "rewrite"),
        DirectorSettings,
        "an idea rewritten into a story the planner renders well",
    ),
    Tool("survey", 1, ("load VLM", "survey"), DirectorSettings, "the cast an idea needs, and which photo shows whom"),
    Tool("ground", 1, ("load VLM", "ground"), DirectorSettings, "boxes around the cast in one photo"),
    Tool("cut_crops", 1, ("cut",), None, "the opening frame and every reference crop"),
    Tool("cards", 1, ("load VLM", "read"), DirectorSettings, "what one photo shows of each member"),
    Tool("describe_opening", 1, ("load VLM", "describe"), DirectorSettings, "the opening frame in words"),
    Tool("tidy", 1, ("load VLM", "tidy"), DirectorSettings, "one identity per member, merged over its photos"),
    Tool("arrange", 1, ("load VLM", "arrange"), DirectorSettings, "the chapter's first sentence"),
    Tool("plan_shots", 1, ("load VLM", "plan"), DirectorSettings, "the whole chapter, shot by shot"),
    Tool("write_shot", 1, ("load VLM", "write"), DirectorSettings, "one shot's prose"),
    Tool("compile_shot", 1, ("compile",), None, "the caption H3 is handed, and the pictures with it"),
)
