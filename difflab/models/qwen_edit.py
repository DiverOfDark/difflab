"""Edit stage: QwenImageEditPlusPipeline assembled from the AIO checkpoint."""

import gc
from collections.abc import Sequence

import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from PIL import Image

from difflab import config, prompts
from difflab.core import ui
from difflab.core.hub import HubFile
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
        with ui.EditProgress(pipe.transformer, steps) as progress:
            result = pipe(
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
        # Fits a 24 GB GPU next to the resident 4-bit VLM: the transformer streams block by block and the
        # text encoder layer by layer from pinned RAM (CUDA-stream prefetch); the small VAE stays on the GPU.
        onload = torch.device(self.device)
        pipe.transformer.enable_group_offload(
            onload_device=onload,
            offload_type="block_level",
            num_blocks_per_group=1,  # CUDA-stream prefetch requires one block per group
            use_stream=True,
            low_cpu_mem_usage=True,
        )
        apply_group_offloading(
            pipe.text_encoder,
            onload_device=onload,
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
        )
        pipe.vae.to(self.device)
        pipe.set_progress_bar_config(disable=True)  # difflab.ui draws the progress
        return pipe
