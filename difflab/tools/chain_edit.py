"""A photo edited one logical change at a time. Every picture is judged against the original, never
the step before it, so drift that each step adds a little of still fails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from difflab import prompts
from difflab.core import ui
from difflab.core.settings import EditSettings
from difflab.core.types import Frame
from difflab.models.exl3vl import ANY_JSON
from difflab.models.minimax_h3.caption import Problem
from difflab.screenplay import config as h3
from difflab.tools import direct, edit_image

INSTRUCTION = {"type": "string", "minLength": 1, "maxLength": 1500}
QUOTE = {"type": "string", "maxLength": 300}


def _shape(**properties: Any) -> dict[str, Any]:
    return {
        **ANY_JSON, "type": "object", "properties": properties, "required": list(properties),
        "additionalProperties": False,
    }  # fmt: skip


EDITS = _shape(edits={"type": "array", "items": INSTRUCTION})
EDIT = _shape(edit=INSTRUCTION)
DONE = _shape(seen=QUOTE, done={"type": "boolean"}, fix=QUOTE)
UNASKED = _shape(changes={"type": "array", "items": QUOTE})
UNASKED_PREFIX = "not asked for: "


@dataclass(frozen=True)
class Chained:
    edits: list[str]
    images: list[Frame]  # images[k] is the photo after the first k edits; images[0] is the photo
    flagged: list[int]  # the edits no retry got right, 0-based; the chain carries on past each
    issues: list[str]  # what the judge finds in the delivered picture, against the photo


def chain(photo: Frame, idea: str, settings: EditSettings, retries: int, carry: bool = False) -> Chained:
    """`carry` conditions each edit on the last one's latent, never on its picture decoded and encoded again."""
    if carry and settings.engine != "qwen-image-2.1":
        ui.note(f"chain · carrying latents is Qwen-Image 2.1's alone; {settings.engine} edits from pictures")
        carry = False
    size = edit_image.carried_size(photo, settings.megapixels) if carry else None
    edits = plan(photo, photo, idea, [], settings.engine)
    images: list[Frame] = [photo]
    latents: list[torch.Tensor | None] = [None]
    flagged: list[int] = []
    # Pictures are judged against images[base]. A flagged picture becomes the base: judged against the photo,
    # what it got wrong would fail every picture after it and leave no retry for their own edits.
    base = 0

    def made(source: int, edit: str, chosen: EditSettings) -> tuple[Frame, torch.Tensor | None]:
        if size is None:
            return edit_image.edit(images[source], edit, chosen), None
        return edit_image.edit_carried(images[source], latents[source], edit, chosen, size)

    while True:
        while len(images) <= len(edits):
            step = len(images)
            ui.note(f"chain · edit {step} of {len(edits)}: {edits[step - 1]}")
            image, latent = made(step - 1, edits[step - 1], settings)
            images.append(image)
            latents.append(latent)
        held, issues = _held(images, edits, base)
        if held == len(edits):
            break
        # A picture that held stays held: the judge is greedy and kept by its question, so `held` only grows.
        tries = [(issues, images[held + 1], edits[held], latents[held + 1])]
        for attempt in range(1, retries + 1):
            edit = revise(images[held], tries[-1][1], tries[-1][2], tries[-1][0], settings.engine)
            image, latent = made(held, edit, settings.with_(seed=settings.seed + attempt))
            tries.append((faults(images[base], image, [*edits[base:held], edit]), image, edit, latent))
            if not tries[-1][0]:
                break
        # By count alone a tie kept the first take: on the beach test (2026-10-14) a noisy sky without footprints
        # over a retry that drew them and moved the palm.
        left, image, edit, latent = min(
            tries, key=lambda t: (sum(not i.startswith(UNASKED_PREFIX) for i in t[0]), len(t[0]))
        )
        if left:
            flagged.append(held)
            base = held + 1
            ui.note(f"chain · edit {held + 1} still fails after {retries} retries; carrying on with it flagged")
        edits[held], images[held + 1 :], latents[held + 1 :] = edit, [image], [latent]
        edits[held + 1 :] = plan(photo, image, idea, edits[: held + 1], settings.engine)
    return Chained(edits, images, flagged, faults(photo, images[-1], edits))


def plan(original: Frame, current: Frame, idea: str, done: list[str], engine: str) -> list[str]:
    """The edits still to make: all of them from the photo, or what is left after `done`, from `current`."""
    if done:
        question = prompts.CHAIN_REPLAN_USER.format(idea=idea, done=_listed(done))
        images = [_view(original), _view(current)]
    else:
        question, images = prompts.CHAIN_SPLIT_USER.format(idea=idea), [_view(original)]

    def gate(data: Any) -> list[Problem]:
        written = [str(edit).strip() for edit in data["edits"]]
        out = [] if written or done else [direct.problem("the idea needs at least one edit")]
        return out + [direct.problem(f"edit {n} is empty") for n, edit in enumerate(written, 1) if not edit]

    title = "chain · re-plan" if done else "chain · split"
    data = direct.ask_json(_system(engine), question, images, gate, title, h3.DIRECTOR_WRITE_TOKENS, schema=EDITS)
    edits = [str(edit).strip() for edit in data["edits"]]
    ui.panel(title, {str(n): edit for n, edit in enumerate(edits, len(done) + 1)} or {"-": "nothing left"})
    return edits


def revise(before: Frame, after: Frame, edit: str, issues: list[str], engine: str) -> str:
    question = prompts.CHAIN_REVISE_USER.format(edit=edit, issues="\n".join(f"- {issue}" for issue in issues))
    data = direct.ask_json(
        _system(engine), question, [_view(before), _view(after)], lambda _: [], "chain · revise",
        h3.DIRECTOR_WRITE_TOKENS, schema=EDIT,
    )  # fmt: skip
    return str(data["edit"]).strip()


def faults(original: Frame, image: Frame, edits: list[str]) -> list[str]:
    """What is wrong with `image` as `original` after `edits`; empty when it holds them all.
    One question per edit: in the screenplay's rulings (2026-10-03) a joint question let one item's verdict
    flip with a change to another."""
    if not edits:
        return []
    seen, listed, out = [_view(original), _view(image)], _listed(edits), []
    for number in range(1, len(edits) + 1):
        data = direct.ask_json(
            prompts.CHAIN_JUDGE_SYSTEM, prompts.CHAIN_DONE_USER.format(edits=listed, number=number), seen,
            lambda _: [], f"chain · judge edit {number}", h3.RULING_TOKENS, schema=DONE,
        )  # fmt: skip
        if not data["done"]:
            fix = str(data["fix"]).strip()
            out.append(f"instruction {number} is not carried out: {data['seen']}" + (f"; {fix}" if fix else ""))
    data = direct.ask_json(
        prompts.CHAIN_JUDGE_SYSTEM, prompts.CHAIN_UNASKED_USER.format(edits=listed), seen, lambda _: [],
        "chain · judge the rest", h3.RULING_TOKENS, schema=UNASKED,
    )  # fmt: skip
    out += [f"{UNASKED_PREFIX}{change}" for change in map(str, data["changes"]) if change.strip()]
    ui.panel(
        f"chain · after edit {len(edits)}",
        {str(n): issue for n, issue in enumerate(out, 1)} or {"holds": "every edit, nothing else changed"},
        color="yellow" if out else "green",
    )
    return out


def _held(images: list[Frame], edits: list[str], base: int) -> tuple[int, list[str]]:
    """The most edits a picture holds, walking back from the last to `base`, and what is wrong with the one after it."""
    after: list[str] = []
    for count in range(len(edits), base, -1):
        found = faults(images[base], images[count], edits[base:count])
        if not found:
            return count, after
        after = found
    return base, after


def _system(engine: str) -> str:
    rules = prompts.VLM_SYSTEM_PROMPT_21 if engine == "qwen-image-2.1" else prompts.VLM_SYSTEM_PROMPT
    return f"{rules}\n\n{prompts.CHAIN_SYSTEM}"


def _listed(edits: list[str]) -> str:
    return "\n".join(f"{n}. {edit}" for n, edit in enumerate(edits, 1))


def _view(frame: Frame) -> Image.Image:
    return direct.shrink(frame, h3.DIRECTOR_PHOTO_PIXELS)
