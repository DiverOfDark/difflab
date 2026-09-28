"""Qwen3.8-27B as turboderp's EXL3 on ExLlamaV3, its MTP head (kept at `mtp_bits: 4`) drafting.

Transformers over bitsandbytes NF4 decoded it at 17.7 tok/s on a 3090, a third of what reading 15 GB of
weights per token allows; `config.VLM_DRAFT_TOKENS` has what this does.
"""

import gc
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from exllamav3 import Cache, Config, Generator, Job, LLGuidanceFilter, Model, Tokenizer
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
# The separators are the model's own: forced minified (no whitespace at all), every seed answered
# `{"members":[]}`, the prefix `"members":[` being one it never writes and `]` its likeliest sequel.
ANY_JSON = {
    "type": ["object", "array"],
    "x-guidance": {"whitespace_flexible": False, "item_separator": ", ", "key_separator": ": "},
}


class Exl3VL:
    """Resident from the first call until `unload`, like `QwenVL`, and asked the same way."""

    def __init__(self, checkpoint: Checkpoint) -> None:
        self.checkpoint = checkpoint
        self._parts: dict[str, Any] | None = None

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
    ) -> str:
        """Several images are announced as "Picture 1: ", "Picture 2: " in order; temperature=0 is greedy."""
        parts = self.load()
        shown = [images] if isinstance(images, Image.Image) else list(images or ())
        tokenizer = parts["tokenizer"]
        embeddings = [parts["vision"].get_image_embeddings(tokenizer=tokenizer, image=image) for image in shown]
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        content = [*({"type": "image"} for _ in shown), {"type": "text", "text": question}]
        messages.append({"role": "user", "content": content})
        prompt = parts["template"].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            add_vision_id=len(shown) > 1,
        )
        for embedding in embeddings:
            prompt = prompt.replace(IMAGE_SLOT, embedding.text_alias, 1)
        sampler = (
            ComboSampler(temperature=temperature, top_p=top_p, top_k=top_k) if temperature > 0 else ArgmaxSampler()
        )
        job = Job(
            input_ids=tokenizer.encode(prompt, encode_special_tokens=True, embeddings=embeddings),
            max_new_tokens=max_new_tokens,
            stop_conditions=parts["stops"],
            sampler=sampler,
            seed=seed,
            embeddings=embeddings,
            filters=[LLGuidanceFilter(tokenizer, eos_after_completed=True, json_schema=ANY_JSON)] if as_json else None,
        )
        generator = parts["generator"]
        generator.enqueue(job)
        text = ""
        with ui.live_answer(title, echo) as show:
            while generator.num_remaining_jobs():
                for result in generator.iterate():
                    if chunk := result.get("text"):
                        text += chunk
                        show(text)
        return text.strip()

    def unload(self) -> None:
        if self._parts is not None:
            for model in self._parts["models"]:
                model.unload()
        self._parts = None
        gc.collect()
        torch.cuda.empty_cache()
