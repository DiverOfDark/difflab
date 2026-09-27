"""Captions written without a director: an instruction wrapped in the six sections H3 reads, and the
declarations of what a shot is handed - a source excerpt, the previous shot's tail - which differ
between the first shot and the rest and so are said per shot, not once."""

from __future__ import annotations

import re

from difflab.models.minimax_h3 import caption

SOUNDSCAPE = "The location sound of <Video 1> continues unchanged: its room tone, movement and voices."
EDITING = "[video editing + keyframe completion + reference generation]"
CONTINUING = "[video continuation + keyframe completion + reference generation + audio reference]"


def renumbered(said: str, keyframe: bool) -> str:
    """Picture numbers moved down one when no keyframe holds <Picture 1>. <Picture 1> itself is left,
    so a caption naming it in a keyframe-free shot is refused by name rather than quietly moved."""
    if keyframe:
        return said
    return re.sub(
        r"<Picture (\d+)>", lambda m: f"<Picture {int(m.group(1)) - 1}>" if int(m.group(1)) > 1 else m.group(0), said
    )


def caption_for(instruction: str, pictures: int, keyframe: bool = True) -> str:
    """A one-line instruction wrapped in the six sections, or passed through if it already is one."""
    if any(name == "subject_definitions" for name, *_ in caption.sections(instruction)):
        return instruction.strip() + "\n"
    said = renumbered(instruction.strip().rstrip("."), keyframe)
    first = 2 if keyframe else 1
    subjects = ["<Picture 1> is this shot's first frame: the video starts exactly from it."] if keyframe else []
    subjects += [
        f"<Picture {index}> is a reference photograph; keep what it shows of its subject."
        for index in range(first, first + pictures - (1 if keyframe else 0))
    ]
    retention = (
        ["<Picture 1> ([Shot 1] first frame): fully_preserved - the shot starts from its framing and light."]
        if keyframe
        else []
    )
    retention += [
        f"<Picture {index}> (appears in [Shot 1]): partially_preserved - what it shows of its subject is kept."
        for index in range(first, first + pictures - (1 if keyframe else 0))
    ]
    return (
        "subject_definitions:\n" + "\n".join(subjects) + "\n\n"
        f"summary:\n{EDITING} {said}.\n\n"
        "retention_analysis:\n" + "\n".join(retention) + "\n\n"
        f"detailed_description:\n[Shot 1] The shot re-creates what the source footage does, beat for beat, "
        f"with one difference: {said}.\n\n"
        f"overall_soundscape:\n{SOUNDSCAPE}\n\n"
        "non_diegetic_music:\nnone\n"
    )


def _declares(prompt: str, label: str) -> bool:
    return bool(re.search(rf"^\s*{re.escape(label)}", caption.body(prompt, "subject_definitions"), re.M))


def _declare(prompt: str, label: str, definition: str, retention: str) -> str:
    if _declares(prompt, label):
        return prompt
    prompt = caption.add_to_section(prompt, "subject_definitions", definition)
    return caption.add_to_section(prompt, "retention_analysis", retention)


def cite_source(prompt: str, start: float, end: float, tail: bool, change: str = "") -> str:
    """Declare the source excerpt this shot is handed as <Video 1>, and the previous shot's tail as
    <Video 2> with <Audio 1> when there is one - the exclusive H3 tasks make the tail the declared
    task and the source a reference. `change` is what is different, said where the picture is read."""
    prompt = _declare(
        prompt,
        "<Video 1>",
        f"<Video 1> is the source footage this shot re-creates, from {start:.1f}s to {end:.1f}s of it: "
        "reproduce its action, its camera and its timing beat for beat, changing only what the "
        "description below changes.",
        "<Video 1> (appears in [Shot 1]): partially_preserved - its action, timing, camera and framing "
        "are retained; only what the description changes is different.",
    )
    if tail:
        prompt = _declare(
            prompt,
            "<Video 2>",
            "<Video 2> is the end of the previous shot; this one continues from its final frame.",
            "<Video 2> (appears in [Shot 1]): fully_preserved - this shot continues its motion.",
        )
        prompt = _declare(
            prompt,
            "<Audio 1>",
            "<Audio 1> is the soundtrack of <Video 2>, whose room tone continues into this shot.",
            "<Audio 1>: reference - its room tone continues.",
        )
        prompt = prompt.replace(EDITING, CONTINUING)
    if change:
        prompt = caption.add_to_section(
            prompt,
            "detailed_description",
            f"Changed from <Video 1>: {change.strip().rstrip('.')}. Everything else in this shot - the "
            "action, the camera, the framing and who is in it - matches <Video 1> exactly.",
        )
    return prompt
