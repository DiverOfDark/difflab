"""The one language model: Qwen3.8-27B, which sees photographs and writes every prompt.

`Qwen/Qwen3.8-27B` is Qwen3.8-27B with its vision tower. 52 GB of bf16 on disk and 15.1 GiB of NF4 on the card, so it cannot share the 24 GB card with
H3 or the editor, and the quantized weights are kept rather than made again (`config.VLM_4BIT_CACHE`).

Qwen3.8 thinks before it answers unless told not to; every call here turns thinking off, so the token
budget is spent on the answer alone.
"""

import gc
import shutil
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
    TextIteratorStreamer,
    set_seed,
)
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

from difflab import config
from difflab.core import ui
from difflab.core.hub import Checkpoint
from difflab.models.processors import ImageOnlyQwen3VLProcessor

NAME = "Qwen3.8-27B (NF4)"


def compile_generation(model: Any) -> None:
    """Let `generate` capture CUDA graphs for this 4-bit model, by answering the two checks that refuse it.

    `generation/utils.py` compiles only when `using_compilable_cache` holds and then
    `can_compile &= hf_quantizer.is_compileable`. The cache refuses because this model builds a
    `DynamicCache`, excluded by type; asking for a static one is enough, since `STATIC_LAYER_TYPE_MAPPING`
    reads the config's own `layer_types` and gives the 48 linear-attention layers a `LinearAttentionLayer`
    and the 16 full-attention layers a `StaticLayer`, both compilable. The quantizer refuses because
    `HfQuantizer.is_compileable` is False by default and `Bnb4BitHfQuantizer` never overrides it - a
    default nobody went back to rather than a limitation, since bitsandbytes 0.50.2 defines
    `dequantize_4bit` and `gemv_4bit` through `torch.library` with a registered fake. Should a release
    make that untrue, the symptom is a compilation error on the first answer, not a wrong answer.

    It works - zero graph breaks, 22.5 tok/s against 18.4 eager - and is off anyway, because the static
    cache it needs is read whole on every step and costs more than the graph saves at the lengths the
    director asks at. `config.VLM_COMPILE` carries the numbers and the condition for flipping it back.
    """
    Bnb4BitHfQuantizer.is_compileable = property(lambda _: True)
    model.generation_config.cache_implementation = "static"
    model.generation_config.max_cache_len = config.VLM_CACHE_TOKENS


def _keep(model: Qwen3_5ForConditionalGeneration, kept: Path, detail: Callable[[str], None]) -> None:
    """Never fatal: the model is already loaded, so failing here costs one slow load next time."""
    staging = kept.with_name(f"{kept.name}.partial")
    try:
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(staging)
        staging.rename(kept)
        size = sum(f.stat().st_size for f in kept.rglob("*") if f.is_file())
        detail(f"kept 4-bit for next time: {size / 1024**3:.1f} GB in {ui.link(kept)}")
    except (OSError, RuntimeError, ValueError) as error:
        shutil.rmtree(staging, ignore_errors=True)
        ui.note(f"could not keep the 4-bit weights ({type(error).__name__}: {error}); loading stays slow")


def load_processor(path: Path) -> ImageOnlyQwen3VLProcessor:
    return ImageOnlyQwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorPil.from_pretrained(path, local_files_only=True),
        tokenizer=AutoTokenizer.from_pretrained(path, local_files_only=True),
        video_processor=None,
        chat_template=(path / "chat_template.jinja").read_text(encoding="utf-8"),
    )


class QwenVL:
    """4-bit Qwen3.8-27B, resident from the first call until `unload`."""

    def __init__(self, checkpoint: Checkpoint, device: str = config.DEVICE) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._model: Qwen3_5ForConditionalGeneration | None = None
        self._processor: ImageOnlyQwen3VLProcessor | None = None

    def load(self) -> tuple[Qwen3_5ForConditionalGeneration, ImageOnlyQwen3VLProcessor]:
        if self._model is None or self._processor is None:
            path = self.checkpoint.fetch()
            with ui.loading("Loading VLM (Qwen3.8 27B, 4-bit)") as detail:
                self._model = self._quantized(path, detail)
                self._processor = load_processor(path)
                if config.VLM_COMPILE:
                    compile_generation(self._model)
                detail(f"{self._model.get_memory_footprint() / 1024**3:.1f} GB")
        return self._model, self._processor

    def _quantized(self, path: Path, detail: Callable[[str], None]) -> Qwen3_5ForConditionalGeneration:
        """The 4-bit model: the kept copy if there is one, otherwise made now and kept.

        Keyed by the snapshot's own commit, so re-pinning the checkpoint makes a new one rather than
        quietly loading the old model. Written through a temporary directory and renamed, because a save
        interrupted half way through would otherwise be indistinguishable from a finished one.
        """
        kept = config.VLM_4BIT_DIR.with_name(f"{config.VLM_4BIT_DIR.name}-{self.checkpoint.revision[:12]}")
        if config.VLM_4BIT_CACHE and (kept / "config.json").is_file():
            detail(f"from {ui.link(kept)}")
            model: Qwen3_5ForConditionalGeneration = Qwen3_5ForConditionalGeneration.from_pretrained(
                kept, dtype=torch.bfloat16, device_map=self.device, local_files_only=True
            ).eval()
            return model

        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map=self.device,
            local_files_only=True,
        ).eval()
        if config.VLM_4BIT_CACHE:
            _keep(model, kept, detail)
        return model

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
    ) -> str:
        """Ask anything about one image, several, or none; the answer streams into a panel.

        Several images are announced as "Picture 1: ", "Picture 2: " in order (the chat template's
        add_vision_id), so a question can point at them by number. temperature=0 decodes greedily.
        """
        model, processor = self.load()
        shown = [images] if isinstance(images, Image.Image) else list(images or ())
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in shown]
        messages.append({"role": "user", "content": [*content, {"type": "text", "text": question}]})
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            add_vision_id=len(shown) > 1,
            enable_thinking=False,
        ).to(model.device)
        return answer(model, processor.tokenizer, inputs, title, max_new_tokens, temperature, top_p, top_k, seed, echo)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        torch.cuda.empty_cache()


def answer(
    model: Any,
    tokenizer: Any,
    inputs: Any,
    title: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    echo: bool = True,
) -> str:
    """`generate` on a worker thread, streamed into a panel titled `title`.

    Greedy decoding still names the sampling knobs, as `None`: the checkpoint's generation_config sets
    its own (do_sample=True, temperature 1.0), and transformers warns when they sit unused.
    """
    sampling: dict[str, Any] = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    if temperature > 0:
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=max_new_tokens, streamer=streamer, **sampling)
        except BaseException as e:  # re-raised in the caller's thread below
            errors.append(e)
            streamer.end()

    set_seed(seed)
    worker = threading.Thread(target=generate, daemon=True)
    text = ""
    with ui.live_answer(title, echo) as show:
        worker.start()
        for chunk in streamer:
            text += chunk
            show(text)
        worker.join()
        if errors:
            raise errors[0]
    return text.strip()
