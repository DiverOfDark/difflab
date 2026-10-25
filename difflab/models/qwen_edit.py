"""Edit stage: QwenImageEditPlusPipeline assembled from the AIO checkpoint."""

import gc
from collections.abc import Sequence
from typing import Any, cast

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
from torch import nn

from difflab import config, prompts
from difflab.core import ui
from difflab.core.hub import HubFile
from difflab.models import offload
from difflab.models import qwen_edit_aio as aio


class Editor:
    """Qwen-Image-Edit (2511 generation) on the AIO fine-tune."""

    def __init__(self, weights: HubFile, device: str = config.DEVICE) -> None:
        self.weights = weights
        self.device = device
        self._pipe: QwenImageEditPlusPipeline | None = None

    @property
    def model_name(self) -> str:
        return self.weights.path.stem

    @property
    def pipe(self) -> QwenImageEditPlusPipeline:
        if self._pipe is None:
            self._pipe = self._build()
        return self._pipe

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        steps: int,
        cfg: float,
        seed: int,
        negative_prompt: str = prompts.NEGATIVE_PROMPT,
        donors: Sequence[Image.Image] = (),
    ) -> Image.Image:
        """`image` is already at the target size; the output is that size floored to /16. `donors` follow it
        as Picture 2, 3, ...: the prompt names them, the output keeps `image`'s geometry."""
        pipe = self.pipe
        blocks = pipe.transformer.transformer_blocks

        def run() -> Any:
            with ui.EditProgress(pipe.transformer, steps) as progress:
                return pipe(
                    image=[image, *donors] if donors else image,
                    prompt=prompt,
                    # CFG off at cfg<=1: the negative prompt would be ignored anyway (ComfyUI skips it too).
                    negative_prompt=negative_prompt if cfg > 1 else None,
                    true_cfg_scale=cfg,
                    num_inference_steps=steps,
                    height=image.height,
                    width=image.width,
                    # CPU generator: identical noise regardless of where offloading puts the weights.
                    generator=torch.Generator("cpu").manual_seed(seed),
                    callback_on_step_end=progress.on_step_end,
                )

        settled = offload.fit(blocks, config.EDIT_RESERVE_BYTES)
        ui.line(f"{settled}/{len(blocks)} transformer blocks resident")
        try:
            result = run()
        except torch.OutOfMemoryError:
            # Retried outside the handler, so its traceback lets go of the first attempt's activations.
            if not offload.release():
                raise
            result = None
        if result is None:
            ui.note("out of memory with transformer blocks resident: editing again with every block streaming")
            result = run()
        edited: Image.Image = result.images[0]
        return edited

    def unload(self) -> None:
        self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def _build(self) -> QwenImageEditPlusPipeline:
        weights = self.weights.fetch()
        tokenizer, processor = aio.load_processor()
        pipe = QwenImageEditPlusPipeline(
            scheduler=aio.load_scheduler(),
            vae=aio.load_vae(weights),
            text_encoder=aio.load_text_encoder(weights),
            tokenizer=tokenizer,
            processor=processor,
            transformer=aio.load_transformer(weights),
        )
        device = torch.device(self.device)
        for name, child in pipe.transformer.named_children():
            if name != "transformer_blocks":
                child.to(device)
        offload.stream_blocks(pipe.transformer.transformer_blocks, device)
        inner = cast(nn.Module, pipe.text_encoder.model)
        language_model = cast(nn.Module, inner.language_model)
        for name, child in language_model.named_children():
            if name != "layers":
                child.to(device)
        cast(nn.Module, inner.visual).to(device)
        offload.stream_blocks(cast(nn.ModuleList, language_model.layers), device)
        pipe.vae.to(self.device)
        pipe.set_progress_bar_config(disable=True)  # difflab.ui draws the progress
        return pipe
