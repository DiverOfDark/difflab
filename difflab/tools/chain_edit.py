"""A photo edited one logical change at a time. Every picture is judged against the original, never
the step before it, so drift that each step adds a little of still fails."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from difflab import config, prompts
from difflab.core import events, protocol, ui
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
UNASKED = _shape(changed={"type": "boolean"}, changes={"type": "array", "items": QUOTE})
UNASKED_PREFIX = "not asked for: "
TRAITS = _shape(traits={"type": "array", "items": QUOTE})
TARGETED = _shape(changes={"type": "boolean"})
HOLDS = {
    "type": "object", "properties": {"seen": QUOTE, "holds": {"type": "boolean"}}, "required": ["seen", "holds"],
    "additionalProperties": False,
}  # fmt: skip
BODY_PREFIX = "the person changed: "


@dataclass(frozen=True)
class Fault:
    issue: str
    chance: float  # the judge's probability that the fault is real

    def __str__(self) -> str:
        return f"{self.issue} ({self.chance:.0%})"


@dataclass(frozen=True)
class Chained:
    edits: list[str]
    images: list[Frame]  # images[k] is the photo after the first k edits; images[0] is the photo
    flagged: list[int]  # the edits no amendment got right, 0-based; the chain carries on past each
    issues: list[str]  # what the judge finds in the delivered picture, against the photo


def chain(
    photo: Frame, idea: str, settings: EditSettings, retries: int, carry: bool = False, body: bool = True
) -> Chained:
    """`carry` conditions each edit on the last one's latent, never on its picture decoded and encoded again.
    A picture that fails is amended, not redrawn: the editor gets it with the judge's base as a second picture, so
    each edit costs at most 1 + `retries` pictures."""
    # Carried on the AIO, the sky turned to noise by the third edit, worse than from decoded pictures (beach test,
    # 2026-10-15).
    carry = carry and settings.engine == "qwen-image-2.1"
    size = edit_image.carried_size(photo, settings.megapixels) if carry else None
    edits = plan(photo, photo, idea, [], settings.engine)
    images: list[Frame] = [photo]
    latents: list[torch.Tensor | None] = [None]
    flagged: list[int] = []
    # Pictures are judged against images[base]. A flagged picture becomes the base: judged against the photo,
    # what it got wrong would fail every picture after it and leave no amendment for their own edits.
    base = 0
    shown = _Shown()
    shown(photo, "the photo")
    while len(images) <= len(edits):
        step = len(images)
        ui.note(f"chain · edit {step} of {len(edits)}: {edits[step - 1]}")
        if size is None:
            image, latent = edit_image.edit(images[-1], edits[step - 1], settings), None
        else:
            image, latent = edit_image.edit_carried(images[-1], latents[-1], edits[step - 1], settings, size)
        shown(image, f"edit {step} of {len(edits)} · {edits[step - 1]}")
        judged = edits[base:step]
        tries = [(faults(images[base], image, judged, body), image, latent)]
        for attempt in range(1, retries + 1):
            found, best, _ = max(tries, key=lambda t: _holds(t[0]))
            if not _failing(found):
                break
            fix = amend(best, images[base], edits[step - 1], _failing(found), settings.engine)
            image = edit_image.edit(
                best, fix, settings.with_(seed=settings.seed + attempt), size=size, donors=[images[base]]
            )
            shown(image, f"edit {step} of {len(edits)} · amendment {attempt} · {fix}")
            tries.append((faults(images[base], image, judged, body), image, None))
        found, image, latent = max(tries, key=lambda t: _holds(t[0]))
        images.append(image)
        latents.append(latent)
        if _failing(found):
            flagged.append(step - 1)
            base = step
            ui.note(f"chain · edit {step} still fails after {retries} amendments; carrying on with it flagged")
            # Never longer than what was left: a re-plan that grew the chain on every flag could run without end.
            left = len(edits) - step
            edits[step:] = plan(photo, image, idea, edits[:step], settings.engine)[:left]
    return Chained(edits, images, flagged, [str(f) for f in _failing(faults(photo, images[-1], edits, body))])


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


def amend(made: Frame, reference: Frame, edit: str, found: list[Fault], engine: str) -> str:
    """The instruction that corrects `made`, the editor's take on `edit`, with `reference` as its Picture 2."""
    question = prompts.CHAIN_AMEND_USER.format(edit=edit, issues="\n".join(f"- {f.issue}" for f in found))
    data = direct.ask_json(
        _system(engine), question, [_view(made), _view(reference)], lambda _: [], "chain · amend",
        h3.DIRECTOR_WRITE_TOKENS, schema=EDIT,
    )  # fmt: skip
    return str(data["edit"]).strip()


def faults(original: Frame, image: Frame, edits: list[str], body: bool = True) -> list[Fault]:
    """Every check of `image` as `original` after `edits`, each with the chance that it fails.
    One question per edit: in the screenplay's rulings (2026-10-03) a joint question let one item's verdict
    flip with a change to another."""
    if not edits:
        return []
    seen, listed, out = [_view(original), _view(image)], _listed(edits), []
    for number in range(1, len(edits) + 1):
        data = direct.ask_json(
            prompts.CHAIN_JUDGE_SYSTEM, prompts.CHAIN_DONE_USER.format(edits=listed, number=number), seen,
            lambda _: [], f"chain · judge edit {number}", h3.RULING_TOKENS, schema=DONE, odds=True,
        )  # fmt: skip
        fix = str(data["fix"]).strip()
        out.append(Fault(f"instruction {number} is not carried out: {data['seen']}" + (f"; {fix}" if fix else ""),
                         1 - float(data["done"])))  # fmt: skip
    data = direct.ask_json(
        prompts.CHAIN_JUDGE_SYSTEM, prompts.CHAIN_UNASKED_USER.format(edits=listed), seen, lambda _: [],
        "chain · judge the rest", h3.RULING_TOKENS, schema=UNASKED, odds=True,
    )  # fmt: skip
    told = "; ".join(change.strip() for change in map(str, data["changes"]) if change.strip())
    out.append(Fault(f"{UNASKED_PREFIX}{told or 'something'}", float(data["changed"])))
    out += changed(original, image, edits, body)
    likely = _failing(out)
    ui.panel(
        f"chain · after edit {len(edits)}",
        {str(n): str(f) for n, f in enumerate(likely, 1)} or {"holds": f"{_holds(out):.0%} that every check holds"},
        color="yellow" if likely else "green",
    )
    return out


def traits(original: Frame) -> list[str]:
    """The lasting physical traits of the people `original` shows, as checkable statements."""
    data = direct.ask_json(
        prompts.CHAIN_TRAITS_SYSTEM, prompts.CHAIN_TRAITS_USER, [_view(original)], lambda _: [], "chain · traits",
        h3.DIRECTOR_WRITE_TOKENS, schema=TRAITS,
    )  # fmt: skip
    return list(dict.fromkeys(trait.strip() for trait in map(str, data["traits"]) if trait.strip()))


def changed(original: Frame, image: Frame, edits: list[str], body: bool) -> list[Fault]:
    """The traits of `original`'s people that `image` lost, leaving out those `edits` set out to change. All in one
    question: on 26 planted and clean pictures (2026-10-15) it caught what one question per trait caught, 13 of 20
    with no false alarm, at 9 s a picture against 37. Neither caught a body stretched or widened by a fifth."""
    if not body:
        return []
    listed = _listed(edits)
    kept = [
        trait for trait in traits(original)
        if not direct.ask_json(
            prompts.CHAIN_TRAITS_SYSTEM, prompts.CHAIN_TARGETED_USER.format(edits=listed, trait=trait), [],
            lambda _: [], "chain · targeted", h3.RULING_TOKENS, schema=TARGETED,
        )["changes"]
    ]  # fmt: skip
    if not kept:
        return []
    data = direct.ask_json(
        prompts.CHAIN_JUDGE_SYSTEM, prompts.CHAIN_TRAIT_LIST_USER.format(traits=_listed(kept)),
        [_view(original), _view(image)], lambda _: [], "chain · judge the person", h3.RULING_TOKENS * 2,
        schema=_shape(traits={"type": "array", "items": HOLDS, "minItems": len(kept), "maxItems": len(kept)}),
        odds=True,
    )  # fmt: skip
    return [
        Fault(f"{BODY_PREFIX}{trait}; it now shows {verdict['seen']}", 1 - float(verdict["holds"]))
        for trait, verdict in zip(kept, data["traits"], strict=True)
    ]


class _Shown:
    """Each picture on the job's page as it is made, before the judge has seen it."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, image: Frame, label: str) -> None:
        run = events.active()
        if run is None:
            return
        self.count += 1
        path = run.path(f"chain/{self.count:03d}.jpg")
        image.pil().convert("RGB").save(path, quality=90)
        run.emit(protocol.Picture(label=label, image=run.rel(path)))


def _failing(found: list[Fault]) -> list[Fault]:
    return [f for f in found if f.chance >= config.CHAIN_FAULT_ODDS]


def _holds(found: list[Fault]) -> float:
    """The chance that every check holds, as if they were independent."""
    return math.prod(1 - f.chance for f in found)


def _system(engine: str) -> str:
    rules = prompts.VLM_SYSTEM_PROMPT_21 if engine == "qwen-image-2.1" else prompts.VLM_SYSTEM_PROMPT
    return f"{rules}\n\n{prompts.CHAIN_SYSTEM}"


def _listed(edits: list[str]) -> str:
    return "\n".join(f"{n}. {edit}" for n, edit in enumerate(edits, 1))


def _view(frame: Frame) -> Image.Image:
    return direct.shrink(frame, h3.DIRECTOR_PHOTO_PIXELS)
