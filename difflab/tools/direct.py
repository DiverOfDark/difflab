"""Asking the VLM for JSON that passes a gate, and the cast as the survey names it.

A stand-in for MiniMax's H3-Context-IR, the prompt rewriter MiniMax runs in front of H3 and has not
released, split so that no model is trusted with anything a program can do exactly. The stages, each
a tool of its own: survey (one call, all photos), ground (one per photo), cut_crops (code), cards
(one per photo), opening (the frame, then one call per pair of hands and one per mouth), tidy (one
per member), arrange (one call), plan_shots (one call for the whole chapter), write_shot (one per
shot), and the compiler and its checks (code; what the gate finds goes back to the model that wrote
it). One model does every step: Qwen3.8 27B (EXL3, `models/exl3vl.py`), shown the photos
for the first four stages and only text after them, so every visual fact a later stage uses is one
it wrote down.

Every answer that passed its gate is kept under output/blobs/director by its exact question, so a
plan run asked again with the same words looks at nothing twice; an answer that failed is never
kept, and a question that failed is asked afresh next time.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from difflab.core import context, models, ui
from difflab.core.cache import cache
from difflab.core.keys import key
from difflab.core.settings import VlmSettings
from difflab.core.types import Frame
from difflab.models.exl3vl import Exl3VL
from difflab.models.minimax_h3 import caption
from difflab.screenplay import compiler as compose
from difflab.screenplay import config as h3
from difflab.screenplay import words
from difflab.tools import ask_vlm

# The director runs on the defaults, not a profile: its answers are cached under this model's name.
DIRECTOR = VlmSettings()
# The checkpoint a director or judge job asks with, as its settings chose it; the default outside one.
MODEL: ContextVar[str] = ContextVar("director_model", default=DIRECTOR.checkpoint)


@contextlib.contextmanager
def using(checkpoint: str | None) -> Iterator[None]:
    token = MODEL.set(checkpoint or DIRECTOR.checkpoint)
    try:
        yield
    finally:
        MODEL.reset(token)


# A fresh re-plan's number: nonzero asks every question again, warm and on its own seeds, under its own
# cache key. Zero leaves the key as it was, so the answers already kept stay found.
TAKE: ContextVar[int] = ContextVar("director_take", default=0)


KINDS = ("person", "animal", "object", "environment")


# In the order `ui.tally` counts them on the progress line.
STAGES = ("survey", "ground", "cards", "opening", "tidy", "arrange", "plan", "write")


ID = re.compile(rf"^[a-z][a-z_]{{0,{h3.DIRECTOR_ID_LENGTH - 1}}}$")


PRONOUN = {"his": "him", "her": "her", "its": "it", "their": "them"}


@dataclass(frozen=True)
class Member:
    id: str  # "man": how every model and every plan refers to it, as {man}
    kind: str  # person | animal | object | environment
    noun: str
    possessive: str  # his | her | its
    find: str  # how to find it in a photo: "the man in the black t-shirt holding the hookah hose"


@dataclass(frozen=True)
class Seen:
    """The opening frame as the model saw it, for it to turn into the take's first sentence (text only)."""

    description: str  # plain words, sides as the picture's own
    layout: str  # where each member was found in the frame, from its box: code, not the VLM
    hands: str  # what is on the wrist of each hand that holds something
    sided: tuple[str, ...]  # the holding hands a trait makes certain: "his left hand"
    mouths: str  # each person's and animal's mouth, what touches the lips, any smoke


# --- asking, answering, checking ------------------------------------------------------------------------
class DirectorError(RuntimeError):
    """A gate could not read the director's own state: not something the model can be asked to fix.

    A plan that casts an id the survey never produced leaves `take.cards[id]` missing; asking again
    spends every repair on a question the model already answered correctly, under a message that lies
    about its JSON. That lookup goes through `member_card`, and no `except` here catches this.
    """


def member_card(cards: dict[str, compose.Card], member: str, where: str) -> compose.Card:
    """The card of `member`, or a DirectorError naming where a cast id came from that has no card."""
    try:
        return cards[member]
    except KeyError:
        raise DirectorError(f"{where}: '{member}' is cast here but is not one of {sorted(cards)}") from None


def ask(
    system: str,
    question: str,
    images: Sequence[Image.Image],
    title: str,
    tokens: int,
    temperature: float = 0.0,
    seed: int = DIRECTOR.seed,
) -> str:
    """One answer, uncached and unchecked: `ask_json` is the one that gates and keeps."""
    vlm: Exl3VL = models.registry.acquire(ask_vlm.spec(MODEL.get()))
    return vlm.ask(
        images,
        question,
        system=system,
        title=title,
        max_new_tokens=tokens,
        temperature=temperature,
        top_p=DIRECTOR.top_p,
        top_k=DIRECTOR.top_k,
        seed=seed,
        echo=False,  # JSON for the gate, not for reading: the tally line shows where we are
        as_json=True,
    )


def parse_json(raw: str) -> Any:
    """The JSON in an answer, fenced in ```json or not, ignoring anything around it.

    Qwen3-VL 8B tended to stop one bracket short at the end of a long answer and, asked again greedily,
    repeated it; the brackets left open are closed here.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    text = fenced.group(1) if fenced else raw.replace("```json", "").replace("```", "")
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise ValueError("no JSON in the answer")
    text = text[min(starts) :].strip()
    try:
        data, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        data, _ = json.JSONDecoder().raw_decode(closed(text))
    return data


def closed(text: str) -> str:
    """`text` with every bracket it leaves open closed, in order (not inside a string)."""
    stack: list[str] = []
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            stack.append("]" if char == "[" else "}")
        elif char in "]}" and stack:
            stack.pop()
    return text + "".join(reversed(stack))


def ask_json(
    system: str,
    question: str,
    images: Sequence[Image.Image],
    gate: Callable[[Any], list[caption.Problem]],
    title: str,
    tokens: int,
    temperature: float = 0.0,
    settle: bool = True,
    repairs: int = h3.DIRECTOR_REPAIRS,
) -> Any:
    """Ask, check, and on failure ask again with the gate's own messages as the instructions.

    Keeps the answer with the fewest errors and, after `repairs` repairs, returns it anyway, saying
    loudly what is still wrong: one beat that runs long is worth more than throwing away the six
    stages before it. `settle=False` for a question whose answer would be unusable, not imperfect.

    The cache token is the model, the system prompt, the question text, the images' bytes, the token
    budget and the temperature - not the title, so two members with the same noun ask the same
    question and share one answer. Only the answer returned is kept, and a settled one is marked
    beside itself so the next run takes it as it is; an answer kept clean is re-checked, since the
    rules may have changed since it was written. A question that stopped the director outright is
    counted there instead, and the next run samples it afresh with new seeds.

    Every question asks for minified JSON (`words.COMPACT`), here rather than in each of the eleven
    prompts so a repair carries it too: the director is decode-bound, and the whitespace of a
    pretty-printed answer is a third of what it waits for.
    """
    question = f"{question}\n\n{words.COMPACT}"
    seen = [hashlib.sha256(image.tobytes()).hexdigest() + f"{image.size}" for image in images]
    take = TAKE.get()
    asked_key = key(
        "director", 1, model=MODEL.get(), system=system, question=question, images=seen, tokens=tokens,
        temperature=temperature, **({"take": take} if take else {}),
    )  # fmt: skip
    kept, failed, marked = (cache.path(asked_key, suffix) for suffix in (".txt", ".failed", ".settled"))
    was_settled = cache.enabled and marked.exists()
    failures = int(failed.read_text(encoding="utf-8")) if cache.enabled and failed.exists() else 0
    best: tuple[int, Any, list[caption.Problem], str] | None = None
    problems: list[caption.Problem] = []
    asked = question
    for attempt in range(repairs + 1):
        heading = title if not attempt else f"{title} · repair {attempt}"
        cached = not attempt and cache.enabled and kept.exists()
        if cached:
            raw = kept.read_text(encoding="utf-8")
            ui.answered(title, "cached", 0.0)
        else:
            fresh = attempt or failures or take
            warmth = max(temperature, h3.DIRECTOR_REPAIR_TEMPERATURE) if fresh else temperature
            seed = (DIRECTOR.seed + attempt + failures * (repairs + 1) + take * 1000) % 2**32  # the sampler's range
            raw = ask(system, asked, images, heading, tokens, warmth, seed)
        try:
            data = parse_json(raw)
        except ValueError as error:
            data, problems = (
                None,
                [caption.Problem("error", "json", f"answer only JSON in the shape shown ({error})", True)],
            )
        else:
            # A gate coerces as it reads - a box to four floats, a clip to an int - so a field of the
            # wrong shape is the model's to fix. A gate tripping over the director's own state raises
            # DirectorError, which no repair can answer and which passes straight out of here.
            try:
                problems = caption.errors(gate(data))
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                data, problems = (
                    None,
                    [caption.Problem("error", "shape", f"the answer does not fit the shape shown ({error})", True)],
                )
        if data is not None and (best is None or len(problems) < best[0]):
            best = (len(problems), data, problems, raw)
        if data is not None and (not problems or (cached and was_settled)):
            keep(kept, failed, raw, settled=was_settled and bool(problems))
            ui.called()
            return data
        fixable = list(dict.fromkeys(p.message for p in problems if p.repairable))
        if not fixable or attempt == repairs:
            break
        issues = "\n".join(f"- {message}" for message in fixable)
        ui.note(f"{title}: asking again - " + "; ".join(fixable))
        asked = words.REPAIR.format(question=question, answer=raw.strip(), issues=issues)
    left = best[2] if best else problems
    if best is not None and settle:
        context.current().settled.append(f"{title}: " + "; ".join(p.message for p in left))
        ui.panel(
            f"{title}: kept after {repairs} repairs, still {len(left)} problem(s)",
            {f"{p.code} {i}": p.message for i, p in enumerate(left)},
            color="yellow",
        )
        keep(kept, failed, best[3], settled=True)
        ui.called()
        return best[1]
    if cache.enabled:
        failed.write_text(str(failures + 1), encoding="utf-8")
    ui.panel(f"{title}: still failing", {f"{p.code} {i}": p.message for i, p in enumerate(left)}, color="red")
    raise RuntimeError(
        f"{title}: {len(left)} problem(s) left after {repairs} repairs. None of its answers was kept "
        f"({failed} counts {failures + 1} failed run(s)); run again and it is asked afresh"
    )


def keep(kept: Path, failed: Path, raw: str, settled: bool = False) -> None:
    """`settled` marks an answer the gate still complains about, so the next run takes it as it is
    instead of paying for repairs that already did not work."""
    if cache.enabled:
        kept.write_text(raw, encoding="utf-8")
        failed.unlink(missing_ok=True)
        marked = kept.with_suffix(".settled")
        if settled:
            marked.write_text("", encoding="utf-8")
        else:
            marked.unlink(missing_ok=True)


def shrink(frame: Frame, pixels: float) -> Image.Image:
    """The photo at most `pixels` large: a phone photo is ~12k vision tokens at full size."""
    image = frame.pil()
    scale = (pixels / (image.width * image.height)) ** 0.5
    if scale >= 1.0:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def problem(message: str) -> caption.Problem:
    return caption.Problem("error", "answer", message, True)
