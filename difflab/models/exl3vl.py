"""Qwen3.8-27B abliterated as turboderp's EXL3 on ExLlamaV3, its MTP head (kept at `mtp_bits: 4`) drafting."""

import gc
import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from exllamav3 import Cache, Config, Generator, Job, LLGuidanceFilter, Model, Tokenizer
from exllamav3.cache import CacheLayer_quant
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from PIL import Image
from transformers import AutoTokenizer

from difflab import config
from difflab.core import ui
from difflab.core.hub import Checkpoint
from difflab.models import kvdisk

# The chat template writes this for every image; the vision model's alias brings its own start and end.
IMAGE_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"

# Sampled free at 0.7, 3 to 6 answers in 20 ran on past their closing bracket with ", {" to the token
# cap; masked to JSON, 0 in the same 20, since stopping is the only token left once the value closes.
# Either separator, as the model chooses: forced minified, every seed answered `{"members":[]}`, the
# prefix `"members":[` being one it never writes unthinking; forced `": "`, a model that had thought "let
# me format the JSON on one line" wrote `{"traits [{": ...`, its `":[{"` masked out (2026-10-11).
ANY_JSON = {"type": ["object", "array"], "x-guidance": {"whitespace_pattern": " ?"}}

OPEN_CALL = "<tool_call>"
CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)

# Hashes of every full page any prompt has had since the server started, reloads included: how much the
# disk cache could restore at most, against what it did.
_PAGES_SEEN: set[bytes] = set()


class Turn(NamedTuple):
    role: str  # "user", "assistant" or "tool"
    said: Sequence[str | Image.Image]
    # An answer's own thinking, given back as it was written so the prompt still matches the cached pages.
    thought: str = ""
    calls: Sequence[dict[str, Any]] = ()  # an answer's tool calls, `{"name", "arguments": {...}}`


class Exl3VL:
    """Resident from the first call until `unload`, like `QwenVL`, and asked the same way."""

    def __init__(self, checkpoint: Checkpoint) -> None:
        self.checkpoint = checkpoint
        self._parts: dict[str, Any] | None = None
        self._embedded: OrderedDict[str, Any] = OrderedDict()

    def load(self) -> dict[str, Any]:
        if self._parts is None:
            with ui.loading(f"Loading VLM ({self.checkpoint.repo.rsplit('/', 1)[-1]})") as detail:
                self._parts = self._load(self.checkpoint.fetch(), detail)
        return self._parts

    def _load(self, path: Path, detail: Callable[[str], None]) -> dict[str, Any]:
        detail(f"from {ui.link(path)}")
        exl_config = Config.from_directory(str(path))
        vision = Model.from_config(exl_config, component="vision")
        vision.load()
        model = Model.from_config(exl_config)
        # One job at a time: every recurrent slot is 48 layers of linear-attention state, ~150 MB for
        # each draft position, and the default 16 slots do not fit beside the weights.
        bits = config.VLM_KV_BITS
        quant: dict[str, Any] = {"layer_type": CacheLayer_quant, "k_bits": bits, "v_bits": bits} if bits else {}
        cache = Cache(
            model,
            max_num_tokens=config.VLM_CONTEXT_TOKENS,
            max_batch_size=1,
            max_history=config.VLM_DRAFT_TOKENS,
            **quant,
        )
        model.load()
        drafts: dict[str, Any] = {}
        if config.VLM_DRAFT_TOKENS:
            mtp = Model.from_config(exl_config, component="mtp")
            drafts = {
                "draft_model": mtp,
                "draft_cache": Cache(mtp, max_num_tokens=config.VLM_CONTEXT_TOKENS, max_batch_size=1, **quant),
                "num_draft_tokens": config.VLM_DRAFT_TOKENS,
            }
            mtp.load()
        tokenizer = Tokenizer.from_config(exl_config)
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            max_batch_size=1,
            recurrent_cache_size=config.VLM_RECURRENT_CACHE_BYTES,
            **drafts,
        )
        root = config.VLM_DISK_CACHE_DIR / f"{self.checkpoint.repo.replace('/', '--')}@{self.checkpoint.revision}"
        detail(f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        return {
            "disk": kvdisk.DiskCache(root, config.VLM_DISK_CACHE_BYTES, generator),
            "vision": vision,
            "models": [m for m in (drafts.get("draft_model"), model, vision) if m is not None],
            "tokenizer": tokenizer,
            "template": AutoTokenizer.from_pretrained(path, local_files_only=True),
            "generator": generator,
            "stops": [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")],
            "thought": tokenizer.single_id("</think>"),
        }

    def ask(
        self,
        images: Image.Image | Sequence[Image.Image] | None,
        question: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        system: str | None = None,
        title: str = "Qwen3.8",
        echo: bool = True,
        as_json: bool = False,
        schema: dict[str, Any] | None = None,
        steady: int = 0,
        effort: str | None = config.VLM_REASONING_EFFORT,
    ) -> str:
        """Several images are announced as "Picture 1: ", "Picture 2: " in order; temperature=0 is greedy.
        `schema` narrows `as_json` from any JSON to that shape. The first `steady` images open other prompts
        too, and are kept on disk with the system prompt before them."""
        shown = [images] if isinstance(images, Image.Image) else list(images or ())
        return self.chat(
            [Turn("user", [*shown, question])],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            system=system,
            title=title,
            echo=echo,
            as_json=as_json,
            schema=schema,
            steady=steady,
            effort=effort,
        )

    def chat(
        self,
        turns: Sequence[Turn],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        system: str | None = None,
        title: str = "Qwen3.8",
        echo: bool = True,
        as_json: bool = False,
        schema: dict[str, Any] | None = None,
        steady: int = 0,
        heard: Callable[[str, str], None] | None = None,
        numbered: bool | None = None,
        effort: str | None = config.VLM_REASONING_EFFORT,
        tools: Sequence[dict[str, Any]] = (),
        usage: dict[str, int] | None = None,
        kept: bool = False,
    ) -> str:
        """`ask` over a whole conversation; `heard` is handed the thought and the answer so far as they arrive.
        `numbered` announces pictures even when there is one, so a conversation's prompt does not change under
        the cache once a second arrives; unset, only several are. `effort=None` answers without thinking.
        `tools` are OpenAI function definitions, called in the template's own `<tool_call>` markup; `usage`
        is filled with the prompt's and the answer's token counts. `kept` writes the whole prompt to disk, for
        a conversation that goes on after something else has had the card."""

        def asked() -> str:
            return self._chat(
                turns, max_new_tokens, temperature, top_p, top_k, seed, system, title, echo, as_json, schema,
                steady, heard, numbered, effort, tools, usage, kept,
            )  # fmt: skip

        try:
            return asked()
        except torch.OutOfMemoryError:
            # Weights and a 4.25 GiB cache left ~0.4 GB of a 3090 once the desktop held 3.4 GB: a clip judge ran
            # out there, and the pages it stranded ran the next question out too (2026-10-25). A fresh load
            # gives the card back whole.
            ui.note("VLM out of memory: loading it again and asking once more")
            self.unload()
        return asked()

    def _chat(
        self,
        turns: Sequence[Turn],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        system: str | None,
        title: str,
        echo: bool,
        as_json: bool,
        schema: dict[str, Any] | None,
        steady: int,
        heard: Callable[[str, str], None] | None,
        numbered: bool | None,
        effort: str | None,
        tools: Sequence[dict[str, Any]],
        usage: dict[str, int] | None,
        kept: bool,
    ) -> str:
        parts = self.load()
        shown = [part for turn in turns for part in turn.said if isinstance(part, Image.Image)]
        tokenizer = parts["tokenizer"]
        seen = [hashlib.sha256(image.tobytes()).hexdigest() + f"{image.mode}{image.size}" for image in shown]
        embeddings = [self._embedding(parts, image, name) for image, name in zip(shown, seen, strict=True)]
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        for turn in turns:
            content = [
                {"type": "image"} if isinstance(part, Image.Image) else {"type": "text", "text": part}
                for part in turn.said
            ]
            message: dict[str, Any] = {"role": turn.role, "content": content, "reasoning_content": turn.thought}
            if turn.calls:
                message["tool_calls"] = [{"function": call} for call in turn.calls]
            messages.append(message)
        prompt = parts["template"].apply_chat_template(
            messages,
            tools=list(tools) or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=effort is not None,
            **({"reasoning_effort": effort} if effort else {}),
            add_vision_id=len(shown) > 1 if numbered is None else numbered,
        )
        for embedding in embeddings:
            prompt = prompt.replace(IMAGE_SLOT, embedding.text_alias, 1)
        sampler = (
            ComboSampler(temperature=temperature, top_p=top_p, top_k=top_k) if temperature > 0 else ArgmaxSampler()
        )
        masked = schema or ANY_JSON
        generator = parts["generator"]
        counted = ("prompt_tokens", "cached_tokens", "seen_tokens", "prefill_seconds", "new_tokens", "decode_seconds")
        numbers = dict.fromkeys(counted, 0.0)

        def drain(
            show: Callable[[str], None], tallied: tuple[str, ...] = ("prompt_tokens", "cached_tokens", "new_tokens")
        ) -> str:
            said, timed = "", (("prefill_seconds", "time_prefill"), ("decode_seconds", "time_generate"))
            try:
                while generator.num_remaining_jobs():
                    for result in generator.iterate():
                        if chunk := result.get("text"):
                            said += chunk
                            show(said)
                        if result.get("stage") == "error":
                            raise result["error"]
                        if result.get("eos"):
                            for name, field in timed:
                                numbers[name] += result[field]
                            for name in tallied:
                                numbers[name] += result[name]
            except BaseException:
                # A cancelled answer left in the queue would be drained as the next question's.
                generator.clear_queue()
                raise
            return said

        def run(text: str, cap: int, stops: list[int], shape: list[Any] | None, show: Callable[[str], None]) -> str:
            ids = tokenizer.encode(text, encode_special_tokens=True, embeddings=embeddings)
            numbers["seen_tokens"] += _seen_tokens(ids)
            generator.enqueue(
                Job(
                    input_ids=ids,
                    max_new_tokens=cap,
                    stop_conditions=stops,
                    sampler=sampler,
                    seed=seed,
                    embeddings=embeddings,
                    filters=shape,
                )
            )
            return drain(show)

        def keep(ids: torch.Tensor, ends: list[int]) -> None:
            """A prompt is checkpointed only at the last page it fills and every 2,048 tokens before its end, so
            each shared opening is put through alone first, its checkpoint and pages then written to disk."""
            for end in ends:
                chain = kvdisk.anchors(ids, end // PAGE_SIZE)
                if not chain or chain[-1] in parts["disk"].states.on_disk:
                    continue
                # Two past the cut: with an MTP draft, ExLlamaV3 caches only (len - 2) // PAGE_SIZE pages of a
                # prompt (`job.py`, max_cached_pages). At one past, the cut's own page was never live: five
                # openings were ever kept, and every clip judge prefilled its ~20k tokens afresh (2026-10-23).
                opening = ids[:, : len(chain) * PAGE_SIZE + 2].clone()
                # The MTP draft embeds the last token itself, and a picture's placeholder ids are past its
                # vocabulary: an opening cut inside a picture failed the judge (IndexError, 2026-10-14). That
                # token is past the checkpoint, so a text one stands in; the draft's KV at the cut is the only
                # thing it moves, and the model still checks every draft.
                if opening[0, -1] >= FIRST_MM_EMBEDDING_INDEX:
                    opening[0, -1] = parts["thought"]
                generator.enqueue(
                    Job(input_ids=opening, max_new_tokens=1, sampler=ArgmaxSampler(), embeddings=embeddings)
                )
                drain(lambda said: None, ())
                if missed := parts["disk"].keep(chain):
                    ui.note(f"VLM opening not kept on disk: {missed}")

        ids, ends = tokenizer.encode(prompt, encode_special_tokens=True, embeddings=embeddings), []
        if ids.shape[-1] >= config.VLM_CONTEXT_TOKENS:
            raise ValueError(f"the prompt is {ids.shape[-1]:,} tokens and the VLM holds {config.VLM_CONTEXT_TOKENS:,}")
        if system:
            ends.append(
                tokenizer.encode(prompt[: prompt.index("<|im_start|>user")], encode_special_tokens=True).shape[-1]
            )
        if steady:
            ends.append(int((ids[0] == embeddings[steady - 1].last_index - 1).nonzero()[0]) + 1)
        if kept:
            ends.append(ids.shape[-1] - 2)  # the two tokens `keep` puts past its cut stay inside the prompt

        with ui.live_answer(title, echo) as show:
            keep(ids, ends)

            def thinking(said: str) -> None:
                show(said)
                if heard:
                    heard(said, "")

            def answering(said: str) -> None:
                show(f"{thought}\n\n{said}" if thought else said)
                if heard:
                    heard(thought, said)

            thought = ""
            if effort:
                thought = run(prompt, config.VLM_THINK_TOKENS, [*parts["stops"], parts["thought"]], None, thinking)
                # Closed here whether the model closed it or the budget ran out, so the answer always has its own cap.
                prompt += f"{thought.strip()}\n</think>\n\n"
            shape = [LLGuidanceFilter(tokenizer, eos_after_completed=True, json_schema=masked)] if as_json else None
            answer = run(prompt, max_new_tokens, parts["stops"], shape, answering)
            show(answer.strip(), **{name: round(value, 2) for name, value in numbers.items()})
        if usage is not None:
            usage.update(prompt_tokens=int(ids.shape[-1]), completion_tokens=int(numbers["new_tokens"]))
        return answer.strip()

    def _embedding(self, parts: dict[str, Any], image: Image.Image, seen: str) -> Any:
        if (embedding := self._embedded.get(seen)) is not None:
            self._embedded.move_to_end(seen)
            return embedding
        embedding = parts["vision"].get_image_embeddings(tokenizer=parts["tokenizer"], image=image)
        # Its placeholder ids come from a process-wide counter; from its bytes, a prompt holding it hashes to the
        # same pages after a restart, which is what the disk cache is keyed by.
        first = FIRST_MM_EMBEDDING_INDEX + 2**40 + int(seen[:12], 16) % 2**30 * 2**20
        placed = (embedding.token_string >= embedding.first_index) & (embedding.token_string < embedding.last_index)
        embedding.token_string[placed] += first - embedding.first_index
        embedding.token_list = embedding.token_string[0].tolist()
        embedding.first_index, embedding.last_index = first, first + embedding.mm_length
        self._embedded[seen] = embedding
        held = sum(e.embeddings.nbytes for e in self._embedded.values())
        while held > config.VLM_EMBEDDING_CACHE_BYTES and len(self._embedded) > 1:
            held -= self._embedded.popitem(last=False)[1].embeddings.nbytes
        return embedding

    def unload(self) -> None:
        if self._parts is not None:
            for model in self._parts["models"]:
                model.unload()
        self._parts = None
        self._embedded.clear()
        gc.collect()
        torch.cuda.empty_cache()


def _seen_tokens(ids: torch.Tensor) -> int:
    found, matching = 0, True
    for phash in kvdisk.anchors(ids, ids.shape[-1] // PAGE_SIZE):
        if matching and phash in _PAGES_SEEN:
            found += PAGE_SIZE
        else:
            matching = False
            _PAGES_SEEN.add(phash)
    return found


def tool_calls(text: str, tools: Sequence[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """The answer's words before its first tool call, and its calls as `{"name", "arguments": {...}}`, each
    argument typed by its tool's schema."""
    found = list(CALL.finditer(text))
    if not found:
        return text, []
    kinds = {tool["name"]: (tool.get("parameters") or {}).get("properties") or {} for tool in tools}
    made = []
    for match in found:
        name = match.group(1).strip()
        arguments = {
            key.strip(): _typed(value, kinds.get(name, {}).get(key.strip(), {}))
            for key, value in PARAMETER.findall(match.group(2))
        }
        made.append({"name": name, "arguments": arguments})
    return text[: found[0].start()].rstrip(), made


def before_calls(text: str, settled: bool) -> str:
    """What of an answer so far can be shown: nothing from a tool call on, nor what may be the start of one."""
    if OPEN_CALL in text:
        return text[: text.index(OPEN_CALL)].rstrip()
    if settled:
        return text
    for cut in range(min(len(OPEN_CALL) - 1, len(text)), 0, -1):
        if OPEN_CALL.startswith(text[-cut:]):
            return text[:-cut]
    return text


def _typed(value: str, schema: dict[str, Any]) -> Any:
    """A string parameter as written; any other as the JSON it should be, or as written when it is not."""
    if schema.get("type", "string") == "string":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
