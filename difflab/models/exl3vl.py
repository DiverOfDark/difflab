"""Qwen3.8-27B as turboderp's EXL3 on ExLlamaV3, its MTP head (kept at `mtp_bits: 4`) drafting."""

import gc
import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from exllamav3 import Cache, Config, Generator, Job, LLGuidanceFilter, Model, Tokenizer
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
from PIL import Image
from transformers import AutoTokenizer

from difflab import config
from difflab.core import ui
from difflab.core.hub import Checkpoint

# The chat template writes this for every image; the vision model's alias brings its own start and end.
IMAGE_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"

# Sampled free at 0.7, 3 to 6 answers in 20 ran on past their closing bracket with ", {" to the token
# cap; masked to JSON, 0 in the same 20, since stopping is the only token left once the value closes.
# Either separator, as the model chooses: forced minified, every seed answered `{"members":[]}`, the
# prefix `"members":[` being one it never writes unthinking; forced `": "`, a model that had thought "let
# me format the JSON on one line" wrote `{"traits [{": ...`, its `":[{"` masked out (2026-10-11).
ANY_JSON = {"type": ["object", "array"], "x-guidance": {"whitespace_pattern": " ?"}}

# Hashes of every full page any prompt has had since the server started, reloads included: how much an
# on-disk KV cache could restore at most, measured before building one. An image counts by its bytes.
_PAGES_SEEN: set[bytes] = set()


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
        cache = Cache(
            model, max_num_tokens=config.VLM_CONTEXT_TOKENS, max_batch_size=1, max_history=config.VLM_DRAFT_TOKENS
        )
        model.load()
        drafts: dict[str, Any] = {}
        if config.VLM_DRAFT_TOKENS:
            mtp = Model.from_config(exl_config, component="mtp")
            drafts = {
                "draft_model": mtp,
                "draft_cache": Cache(mtp, max_num_tokens=config.VLM_CONTEXT_TOKENS, max_batch_size=1),
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
        detail(f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        return {
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
    ) -> str:
        """Several images are announced as "Picture 1: ", "Picture 2: " in order; temperature=0 is greedy.
        `schema` narrows `as_json` from any JSON to that shape."""
        parts = self.load()
        shown = [images] if isinstance(images, Image.Image) else list(images or ())
        tokenizer = parts["tokenizer"]
        seen = [hashlib.sha256(image.tobytes()).hexdigest() + f"{image.mode}{image.size}" for image in shown]
        embeddings = [self._embedding(parts, image, name) for image, name in zip(shown, seen, strict=True)]
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        content = [*({"type": "image"} for _ in shown), {"type": "text", "text": question}]
        messages.append({"role": "user", "content": content})
        prompt = parts["template"].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort=config.VLM_REASONING_EFFORT,
            add_vision_id=len(shown) > 1,
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

        def run(text: str, cap: int, stops: list[int], shape: list[Any] | None, show: Callable[[str], None]) -> str:
            ids = tokenizer.encode(text, encode_special_tokens=True, embeddings=embeddings)
            numbers["seen_tokens"] += _seen_tokens(ids, embeddings, seen)
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
            said = ""
            while generator.num_remaining_jobs():
                for result in generator.iterate():
                    if chunk := result.get("text"):
                        said += chunk
                        show(said)
                    if result.get("eos"):
                        for name, field in (("prefill_seconds", "time_prefill"), ("decode_seconds", "time_generate")):
                            numbers[name] += result[field]
                        for name in ("prompt_tokens", "cached_tokens", "new_tokens"):
                            numbers[name] += result[name]
            return said

        with ui.live_answer(title, echo) as show:
            thought = run(prompt, config.VLM_THINK_TOKENS, [*parts["stops"], parts["thought"]], None, show)
            # Closed here whether the model closed it or the budget ran out, so the answer always has its own cap.
            prompt += f"{thought.strip()}\n</think>\n\n"
            shape = [LLGuidanceFilter(tokenizer, eos_after_completed=True, json_schema=masked)] if as_json else None
            answer = run(prompt, max_new_tokens, parts["stops"], shape, lambda said: show(f"{thought}\n\n{said}"))
            show(answer.strip(), **{name: round(value, 2) for name, value in numbers.items()})
        return answer.strip()

    def _embedding(self, parts: dict[str, Any], image: Image.Image, seen: str) -> Any:
        if (embedding := self._embedded.get(seen)) is not None:
            self._embedded.move_to_end(seen)
            return embedding
        embedding = parts["vision"].get_image_embeddings(tokenizer=parts["tokenizer"], image=image)
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


def _seen_tokens(ids: torch.Tensor, embeddings: Sequence[Any], seen: Sequence[str]) -> int:
    tokens: list[Any] = ids[0].tolist()
    for embedding, name in zip(embeddings, seen, strict=True):
        tokens = [f"{name}:{t - embedding.first_index}" if embedding.first_index <= t < embedding.last_index else t
                  for t in tokens]  # fmt: skip
    found, chain, matching = 0, b"", True
    for start in range(0, len(tokens) - PAGE_SIZE + 1, PAGE_SIZE):
        chain = hashlib.blake2b(chain + repr(tokens[start : start + PAGE_SIZE]).encode(), digest_size=16).digest()
        if matching and chain in _PAGES_SEEN:
            found += PAGE_SIZE
        else:
            matching = False
            _PAGES_SEEN.add(chain)
    return found
