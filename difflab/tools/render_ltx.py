"""LTX-2.5 stages: a frame, its references and a prompt in, a clip out."""

# One model resident at a time: the Gemma encoder is 15.4 GB of int8 and the DiT 21.5 GB. Each stage is a
# function of its own so its model leaves with it: the encoder held in a local of `render` sat beside the
# DiT and ran the card out at its eleventh block (2026-10-24).

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoTokenizer

from difflab import config as app_config
from difflab.core import models, ui
from difflab.core.hub import HubFile, parse_file
from difflab.core.types import Clip, Frame
from difflab.models import offload
from difflab.models.ltx25 import config, pipeline, weights
from difflab.models.ltx25.settings import LtxSettings
from difflab.tools import previews


def _video_vae_spec(settings: LtxSettings) -> models.ModelSpec:
    file = HubFile.parse(settings.video_vae)

    def build() -> nn.Module:
        vae = weights.load_video_vae(file.fetch())
        vae.enable_tiling()
        built: nn.Module = vae.to(app_config.DEVICE)
        return built

    return models.ModelSpec(name=f"ltx_video_vae:{file.filename}", build=build, checkpoints=(file.path,), heavy=False)


def _audio_vae_spec(settings: LtxSettings) -> models.ModelSpec:
    file = HubFile.parse(settings.audio_vae)

    def build() -> nn.Module:
        vae, vocoder = weights.load_audio_vae(file.fetch())
        return nn.ModuleDict({"vae": vae, "vocoder": vocoder}).to(app_config.DEVICE)

    return models.ModelSpec(name=f"ltx_audio_vae:{file.filename}", build=build, checkpoints=(file.path,), heavy=False)


def _text_encoder_spec(settings: LtxSettings) -> models.ModelSpec:
    file = HubFile.parse(settings.text_encoder)
    return models.ModelSpec(
        name=f"ltx_text_encoder:{file.filename}",
        build=lambda: weights.load_text_encoder(file.fetch()).to(app_config.DEVICE),
        checkpoints=(file.path,),
    )


def _connectors_spec(settings: LtxSettings) -> models.ModelSpec:
    dit, encoder = HubFile.parse(settings.checkpoint), HubFile.parse(settings.text_encoder)
    return models.ModelSpec(
        name=f"ltx_connectors:{dit.filename}:{encoder.filename}",
        build=lambda: weights.load_connectors(dit.fetch(), encoder.fetch()).to(app_config.DEVICE),
        heavy=False,
    )


def _dit_spec(settings: LtxSettings, loras: Sequence[tuple[str, float]]) -> models.ModelSpec:
    file = HubFile.parse(settings.checkpoint)
    stack = [(parse_file(reference), strength) for reference, strength in loras]

    def build() -> nn.Module:
        with ui.step(f"Loading DiT ({file.path.stem})") as detail:
            dit = weights.load_dit(file.fetch(), [(lora.fetch(), strength) for lora, strength in stack])
            detail(", ".join(f"{lora.filename}*{strength:g}" for lora, strength in stack) or "no LoRA")
        offload.stream_blocks(dit.transformer_blocks, app_config.DEVICE)
        for name, child in dit.named_children():
            if name != "transformer_blocks":
                child.to(app_config.DEVICE)
        for parameter in dit.parameters(recurse=False):
            parameter.data = parameter.data.to(app_config.DEVICE)
        built: nn.Module = dit
        return built

    named = ",".join(f"{lora.filename}*{strength:g}" for lora, strength in stack)
    return models.ModelSpec(name=f"ltx_dit:{file.filename}:{named}", build=build, checkpoints=(file.path,))


def condition(
    frame: Frame,
    references: Sequence[Frame],
    settings: LtxSettings,
    geometry: pipeline.Geometry,
    slots: weights.SlotEmbedding | None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The opening frame's latent and each reference's, its slot identity added."""
    vae = models.registry.acquire(_video_vae_spec(settings))
    with ui.step("Encoding first frame and references") as detail:
        opening = pipeline.encode_opening(vae, frame, geometry)
        guides = [
            pipeline.encode_reference(vae, reference, geometry, slots, slot)
            for slot, reference in enumerate(references, start=1)
            if slots is not None
        ]
        detail(f"{1 + len(guides)} encoded")
    return opening, guides


def encode(prompt: str, settings: LtxSettings) -> pipeline.Conditioning:
    ui.stage("encoding")
    gemma = models.registry.acquire(_text_encoder_spec(settings))
    connectors = models.registry.acquire(_connectors_spec(settings))
    tokenizer = AutoTokenizer.from_pretrained(config.LAYOUT.fetch() / "tokenizer")
    with ui.step("Encoding prompt") as detail, torch.no_grad():
        conditioning = pipeline.encode_prompt(gemma, connectors, tokenizer, prompt)
        detail(f"{len(tokenizer(prompt).input_ids)} tokens")
    return conditioning


def sample(
    conditioning: pipeline.Conditioning,
    geometry: pipeline.Geometry,
    settings: LtxSettings,
    loras: Sequence[tuple[str, float]],
    opening: torch.Tensor,
    guides: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    ui.stage("sampling")
    dit = models.registry.acquire(_dit_spec(settings, loras))
    blocks = dit.transformer_blocks
    rows = geometry.latent_frames * geometry.tokens_per_frame + sum(guide[0, 0].numel() for guide in guides)
    settled = offload.fit(blocks, config.SAMPLE_RESERVE + rows * config.SAMPLE_RESERVE_PER_ROW)
    ui.line(f"{settled}/{len(blocks)} DiT blocks resident for {rows} rows")
    previewer = previews.for_ltx(settings)
    with ui.sampling("Sampling", len(config.SIGMAS) - 1, len(blocks)) as tick:
        hooks = [block.register_forward_hook(lambda *_: tick()) for block in blocks]
        try:
            return pipeline.sample(
                dit, conditioning, geometry, settings.seed, opening, guides, settings.reference_strength, previewer
            )
        finally:
            for hook in hooks:
                hook.remove()
            if previewer is not None:
                previewer.close()


def decode(video: torch.Tensor, audio: torch.Tensor, settings: LtxSettings, prompt: str) -> Clip:
    ui.stage("decoding")
    offload.release()
    vae = models.registry.acquire(_video_vae_spec(settings))
    with ui.step(
        f"Decoding video {video.shape[-1] * config.SPATIAL_COMPRESSION}x{video.shape[-2] * config.SPATIAL_COMPRESSION}"
    ):
        frames = pipeline.decode_video(vae, video)
    sound = models.registry.acquire(_audio_vae_spec(settings))
    with ui.step("Decoding audio") as detail:
        waveform = pipeline.decode_audio(sound["vae"], sound["vocoder"], audio)
        detail(f"{waveform.shape[-1] / config.AUDIO_SAMPLE_RATE:.2f}s")
    return Clip(frames=frames, audio=waveform, fps=config.FPS, sample_rate=config.AUDIO_SAMPLE_RATE, prompt=prompt)


def render(frame: Frame, prompt: str, settings: LtxSettings | None = None, references: Sequence[Frame] = ()) -> Clip:
    """The clip opens on `frame`; `references` are MSR's Image 1, Image 2... in order."""
    settings = settings or LtxSettings()
    if len(references) > config.MSR_MAX_REFERENCES:
        raise ValueError(f"MSR places at most {config.MSR_MAX_REFERENCES} references, got {len(references)}")
    loras = settings.stack() if references else []
    slots = None
    if references:
        slots = next(
            (found for ref, _ in loras if (found := weights.read_slots(parse_file(ref).fetch())) is not None), None
        )
        if slots is None:
            raise ValueError("references need an MSR LoRA in the stack: none of them carries slot embeddings")
    geometry = pipeline.plan(frame.width, frame.height, settings.duration, settings.megapixels)
    ui.line(f"{geometry.width}x{geometry.height}, {geometry.frame_count} frames, {len(references)} reference(s)")
    opening, guides = condition(frame, references, settings, geometry, slots)
    conditioning = encode(prompt, settings)
    video, audio = sample(conditioning, geometry, settings, loras, opening, guides)
    return decode(video, audio, settings, prompt)
