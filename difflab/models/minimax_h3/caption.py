"""MiniMax's Ref2VA caption format: its sections, its labels, and whether a text obeys them.

This is the format itself, not anyone's policy about what to say in it. `ide_rules` is the port of
ComfyUI-H3-Prompt-IDE's `analyzeH3Prompt` (the Ref2VA path of `h3_prompt_schema_core.mjs`), so a
caption clean here is clean in the IDE; `label_rules` checks one against the labels difflab will
really present. `labels` is where that presentation order is written down once: a caption citing
<Picture 4> when only three pictures ride along binds nothing at all.

It imports nothing, not torch and not difflab, so any layer can check a caption - including a
hand-written one - without loading a model.

Where the format comes from: MiniMax-AI/MiniMax-H3 @d21241f,
`skills/h3-prompt-writing/references/{base-en,ref-en}.txt` (the same documents as HF
`docs/VIDEO_PROMPT_WRITING_GUIDE_{base,ref}_en.md`), its README's Context-IR examples and
`tokenizer/tokenizer_config.json`; the validator is a port of ComfyUI-H3-Prompt-IDE. MiniMax runs a
prompt rewriter, H3-Context-IR, in front of the model and calls it critical to the result; it is not
released, and ComfyUI and diffusers both pass text straight through, so what you write IS the
caption. A 100-word paragraph is far outside what the model was trained on. There is no practical
length limit locally (the hosted API caps prompts at 7,000 characters).

The other mode: base captions (T2VA / I2VA / FL2VA / L2VA) write
`integrated_multimodal_description:` / `overall_soundscape:` / `non_diegetic_music:` instead of these
six sections, and every mode but T2VA opens with an alignment line, exactly:

    I2VA:  For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1])
           is fully referenced.
    FL2VA: How the reference pictures align with the target video - Picture 1 (from Shot 1) aligns
           with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the
           S.SS-second mark of the target video.
    L2VA:  How the reference pictures align with the target video - <Picture 1> (from [Shot N])
           aligns with the S.SS-second mark of the target video.

Ref2VA must NOT begin with one, which is what `ALIGNMENT` checks. difflab ships an `fl2va_8step`
preset, so a base caption is reachable; nothing here compiles one.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

REF_SECTIONS = (
    "subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape",
    "non_diegetic_music",
)  # fmt: skip
ALL_SECTIONS = ("integrated_multimodal_description", *REF_SECTIONS)
SPECIAL_TOKENS = ("<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>", "<|caption_start|>",
                  "<|caption_end|>")  # fmt: skip
TASK_ORDER = {
    "video editing": 0, "video continuation": 0, "keyframe completion": 1, "reference generation": 2,
    "audio reference": 3, "audio reuse": 3,
}  # fmt: skip
LABEL = re.compile(r"<(?:Subject|Picture|Video|Audio)\s+\d+>")
SECTION = re.compile(rf"^({'|'.join(ALL_SECTIONS)}):", re.M)
ALIGNMENT = re.compile(
    r"^(?:For the target video, at 0\.00 seconds into the target video,|How the reference pictures align with)"
)
SECTION = re.compile(rf"^({'|'.join(ALL_SECTIONS)}):", re.M)
ALIGNMENT = re.compile(
    r"^(?:For the target video, at 0\.00 seconds into the target video,|How the reference pictures align with)"
)


def labels(pictures: int, tail: bool, videos: int = 0) -> list[str]:
    """What `text_encoder.encode` announces for a chunk holding `pictures` stills, in its order.

    <Picture 1> is the chunk's own opening frame and the references follow it; a carried tail adds its
    soundtrack before the clip, which is why <Audio 1> comes before <Video 1>.

    `videos` counts reference clips held *before* the tail - a take built from a source video hands
    each chunk that stretch of the source. They come first so that what steers a chunk is <Video 1>
    whether or not it has a tail: "do what <Video 1> does" has to mean the same picture in chunk 1,
    which has none, and in chunk 4, which has one.
    """
    numbered = [f"<Picture {index + 1}>" for index in range(pictures)]
    held = [f"<Video {index + 1}>" for index in range(videos)]
    return numbered + held + (["<Audio 1>", f"<Video {videos + 1}>"] if tail else [])


def directive(tail: bool, framed: bool = True) -> str:
    """The H3 task list a chunk's summary opens with; `framed` is false for one that opens on no picture."""
    if tail:
        return "[video continuation + keyframe completion + reference generation + audio reference]"
    return "[keyframe completion + reference generation]" if framed else "[reference generation]"


@dataclass(frozen=True)
class Problem:
    severity: str  # error | warning
    code: str
    message: str
    repairable: bool = False


def errors(problems: list[Problem]) -> list[Problem]:
    return [problem for problem in problems if problem.severity == "error"]


def sections(text: str) -> list[tuple[str, int, int, int]]:
    """`(name, start, content_start, end)`, as the IDE's parseH3Sections finds them."""
    found = [(match.group(1), match.start(), match.end()) for match in SECTION.finditer(text)]
    records = []
    for index, (name, start, cursor) in enumerate(found):
        end = found[index + 1][1] if index + 1 < len(found) else len(text)
        while cursor < end and text[cursor] in " \t":
            cursor += 1
        cursor += 2 if text[cursor : cursor + 2] == "\r\n" else 1 if text[cursor : cursor + 1] == "\n" else 0
        while cursor < end and text[cursor] in " \t":
            cursor += 1
        records.append((name, start, cursor, end))
    return records


def body(text: str, name: str) -> str:
    return next((text[c:e].strip() for n, _, c, e in sections(text) if n == name), "")


def add_to_section(text: str, name: str, line: str) -> str:
    """A reference added after the caption was compiled has to be announced inside the section that
    cites it, not tacked on the end, or `label_rules` refuses the chunk for presenting a picture its
    words never mention."""
    for section, _, content, end in sections(text):
        if section != name:
            continue
        body_text = text[content:end].rstrip()
        if line in body_text:
            return text
        return text[:content] + f"{body_text}\n{line}\n\n" + text[end:]
    raise ValueError(f"the caption has no {name}: section to add to")


def directive_problem(summary: str) -> str | None:
    match = re.match(r"^\[([^\]]+)\]", summary)
    if not match:
        return "summary must begin with a square-bracketed H3 task type"
    parts = [part.strip() for part in match.group(1).split("+") if part.strip()]
    if not parts or any(part not in TASK_ORDER for part in parts):
        return "summary contains an unknown H3 task type"
    if len(set(parts)) != len(parts):
        return "summary repeats an H3 task type"
    if {"video editing", "video continuation"} <= set(parts) or {"audio reference", "audio reuse"} <= set(parts):
        return "summary combines exclusive H3 task types"
    if any(TASK_ORDER[b] < TASK_ORDER[a] for a, b in itertools.pairwise(parts)):
        return "summary task types are not in canonical H3 order"
    return None


def ide_rules(text: str) -> list[Problem]:
    """analyzeH3Prompt(text, "ref2va"), rule for rule."""
    out: list[Problem] = []
    records = sections(text)
    names = [record[0] for record in records]
    out += [Problem("error", "missing", f"Missing {s}:") for s in REF_SECTIONS if s not in names]
    out += [Problem("error", "duplicate", f"Duplicate {s}:") for s in sorted({n for n in names if names.count(n) > 1})]
    out += [Problem("error", "unexpected", f"{s}: is not Ref2VA") for s in sorted(set(names) - set(REF_SECTIONS))]
    order = [REF_SECTIONS.index(n) for n in names if n in REF_SECTIONS]
    if any(b < a for a, b in itertools.pairwise(order)):
        out.append(Problem("error", "order", "H3 sections are not in their required order"))
    for number, line in enumerate(text.split("\n"), start=1):
        match = re.match(r"^([A-Za-z][A-Za-z _-]+)\s*:", line)
        name = re.sub(r"[ -]+", "_", match.group(1).strip().lower()) if match else ""
        if name in ALL_SECTIONS and not line.startswith(f"{name}:"):
            out.append(Problem("error", "format", f"Line {number}: use exact {name}:"))
    if ALIGNMENT.match(text.lstrip().split("\n", 1)[0]):
        out.append(Problem("error", "alignment", "Ref2VA must not begin with a keyframe-alignment instruction"))
    if "summary" in names and (problem := directive_problem(body(text, "summary"))):
        out.append(Problem("error", "summary", problem))
    for name in REF_SECTIONS:
        content = body(text, name)
        if name in names and not content:
            out.append(Problem("warning", "empty", f"{name}: still needs content"))
    main = body(text, "detailed_description")
    shots = list(re.finditer(r"\[Shot\s+(\d+)\]", main))
    if not shots or shots[0].group(1) != "1":
        out.append(Problem("error", "shots", "The description must begin its shot sequence with [Shot 1]"))
    else:
        if any(int(match.group(1)) != index + 1 for index, match in enumerate(shots)):
            out.append(Problem("error", "shots", "Shot markers must be sequential"))
        previous = -1.0
        for match in shots[1:]:
            stamp = re.match(r"^ At (\d{2}):(\d{2})\.(\d{3}),", main[match.end() :])
            if not stamp:
                out.append(Problem("error", "timestamp", f"{match.group(0)} must immediately use At MM:SS.mmm,"))
                continue
            seconds = int(stamp.group(1)) * 60 + int(stamp.group(2)) + int(stamp.group(3)) / 1000
            if seconds <= previous:
                out.append(Problem("error", "timestamp", f"{match.group(0)} cut time must be strictly later"))
            previous = seconds
        if not main[: main.find("[Shot 1]")].strip():
            out.append(Problem("warning", "style", "detailed_description needs 1-2 style sentences before [Shot 1]"))
    if len(re.findall(r"<d>", text, re.I)) != len(re.findall(r"</d>", text, re.I)):
        out.append(Problem("error", "dialogue", "Unbalanced <d> and </d> dialogue tags"))
    for match in re.finditer(r"</?d>|<\|(?:cutoff|lyrics_start|lyrics_end|caption_start|caption_end)\|>", text, re.I):
        canonical = next((token for token in SPECIAL_TOKENS if token.lower() == match.group(0).lower()), None)
        if canonical and match.group(0) != canonical:
            out.append(Problem("error", "special_token", f"Use exact case-sensitive MiniMax token {canonical}"))
    spans = [(match.start(), match.end(), match.group(1)) for match in re.finditer(r"<d>([\s\S]*?)</d>", text, re.I)]
    for _, _, inner in spans:
        language = re.match(r"^\s*\[([A-Za-z][^\]]*)\]\s+", inner)
        if not language or language.group(1).lower() == "unclear":
            out.append(Problem("warning", "language", "Each <d> span should begin with a [Language] marker"))
    for match in re.finditer(r"<(?:scenetrans|cutoff)>|<\|cutoff\|>", text, re.I):
        if not any(start < match.start() < end for start, end, _ in spans):
            out.append(Problem("error", "dialogue_flow", f"{match.group(0)} must be inside a <d> dialogue span"))
    if re.search(r"<cutoff>", text, re.I):
        out.append(Problem("warning", "legacy_token", "Use tokenizer-native <|cutoff|> instead of legacy <cutoff>"))
    definitions = body(text, "subject_definitions")
    after = text[next((e for n, _, _, e in records if n == "subject_definitions"), 0) :]
    lines = re.findall(r"^\s*(<(?:Subject|Picture|Video|Audio)\s+\d+>)", definitions, re.M)
    defined = list(dict.fromkeys(lines))
    out += [
        Problem("error", "definition", f"{x} is defined more than once")
        for x in sorted(set(lines))
        if lines.count(x) > 1
    ]
    for label in dict.fromkeys(LABEL.findall(after)):
        if label not in defined:
            out.append(Problem("warning", "definition", f"{label} is used but not defined in subject_definitions"))
    retained = set(LABEL.findall(body(text, "retention_analysis")))
    out += [
        Problem("warning", "retention", f"{x} has no retention_analysis entry") for x in defined if x not in retained
    ]
    return out


def label_rules(text: str, labels: list[str]) -> list[Problem]:
    """The labels a chunk cites against the ones difflab presents for it: a stray one binds nothing."""
    out = [
        Problem("error", "label", f"{label} is not in this chunk's presentation {labels}")
        for label in sorted(set(re.findall(r"<(?:Picture|Video|Audio)\s+\d+>", text)) - set(labels))
    ]
    definitions = body(text, "subject_definitions")
    out += [
        Problem("error", "label", f"{x} is presented but never cited in subject_definitions")
        for x in labels
        if x not in definitions
    ]
    return out
