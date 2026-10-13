"""MiniMax H3 blocks: a frame and a prompt in, a clip out.

Split into four stages rather than one `render()` because each wants a different model resident, and
that is what the farm scheduler reorders around:

    condition   video VAE     frame  -> first-frame latent
    encode      text encoder  prompt -> conditioning rows
    sample      DiT           both   -> video + audio latents
    decode      both VAEs     latents-> Clip

Rendering twenty independent clips loads each model once instead of twenty times. A chained long
video cannot be reordered that way - chunk N needs chunk N-1's last frame - but it still keeps the
DiT resident across chunks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from difflab import config
from difflab.core import models, ui
from difflab.core.audio import trim_to_frames
from difflab.core.cache import Cache
from difflab.core.cache import cache as default_cache
from difflab.core.hub import HubFile, parse_file
from difflab.core.keys import Key, key
from difflab.core.types import Clip, Frame
from difflab.models import offload
from difflab.models.minimax_h3 import config as h3_config
from difflab.models.minimax_h3 import pipeline, text_encoder, transformer, vae
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.tools import previews as preview


def video_vae_spec(settings: VideoSettings) -> models.ModelSpec:
    file = HubFile.parse(settings.video_vae)
    return models.ModelSpec(
        name=f"h3_video_vae:{file.filename}",
        build=lambda: vae.load_video_vae(file.fetch()).to(config.DEVICE),
        checkpoints=(file.path,),
        heavy=False,  # 2.6 GB, and the decode stage wants it next to the audio VAE
    )


def audio_vae_spec(settings: VideoSettings) -> models.ModelSpec:
    file = HubFile.parse(settings.audio_vae)
    return models.ModelSpec(
        name=f"h3_audio_vae:{file.filename}",
        build=lambda: vae.load_audio_vae(file.fetch()).to(config.DEVICE),
        checkpoints=(file.path,),
        heavy=False,
    )


def text_encoder_spec(settings: VideoSettings) -> models.ModelSpec:
    file = HubFile.parse(settings.text_encoder)
    return models.ModelSpec(
        name=f"h3_text_encoder:{file.filename}",
        build=lambda: text_encoder.to_device(text_encoder.load(file.fetch())),
        checkpoints=(file.path,),
    )


def dit_spec(settings: VideoSettings) -> models.ModelSpec:
    """One registry entry per checkpoint and LoRA stack, so two never share a slot."""
    weights = HubFile.parse(settings.checkpoint)
    loras = [(parse_file(ref), strength) for ref, strength in settings.stack()]
    stack = ",".join(f"{lora.filename}*{strength:g}" for lora, strength in loras)
    return models.ModelSpec(
        name=f"h3_dit:{weights.filename}:{stack}",
        build=lambda: transformer.load(weights, loras),
        checkpoints=(weights.path,),
    )


StepHook = Callable[[int, torch.Tensor, torch.Tensor], None]


@dataclass(frozen=True)
class Latents:
    video: torch.Tensor
    audio: torch.Tensor
    geometry: pipeline.Geometry
    settings: VideoSettings
    prompt: str = ""


def plan(frame: Frame, settings: VideoSettings | None = None) -> pipeline.Geometry:
    """The canvas and frame count this frame would render at. Loads nothing."""
    settings = settings or VideoSettings()
    return pipeline.plan(frame.width, frame.height, settings.duration, settings.megapixels)


def condition(
    frame: Frame, settings: VideoSettings | None = None, geometry: pipeline.Geometry | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The frame comes back beside its latent because the text encoder needs the same rescaled pixels
    the latent was made from."""
    settings = settings or VideoSettings()
    geometry = geometry or plan(frame, settings)
    canvas = pipeline.resize_to_canvas(frame.batched(), geometry)
    model = models.registry.acquire(video_vae_spec(settings))
    with ui.step("Encoding first frame") as detail:
        latent = vae.encode_frame(model, canvas.permute(0, 3, 1, 2))
        detail(f"{tuple(latent.shape)}")
    return latent, canvas


def condition_last(
    frame: Frame, settings: VideoSettings, geometry: pipeline.Geometry
) -> tuple[torch.Tensor, torch.Tensor]:
    """The frame a clip ends on, cover-cropped onto the canvas as diffusers crops `last_image`: a
    stretched one would be a different arrangement from the picture the next shot opens on."""
    return condition(frame.crop_to_aspect(geometry.width / geometry.height), settings, geometry)


def encode_references(
    frames: Sequence[Frame],
    settings: VideoSettings | None = None,
    geometry: pipeline.Geometry | None = None,
    clips: Sequence[Clip] = (),
    soundtracks: Sequence[torch.Tensor] = (),
    clip_scale: float = 1.0,
) -> tuple[list[pipeline.Reference], list[text_encoder.PresentedReference]]:
    """References -> `(latents for the DiT, the presentation for the text encoder)`.

    A reference image keeps its own aspect ratio and is sized by `settings.reference_size`: it is not
    a frame of the clip, so it is not fitted to the canvas. A reference clip is, because the VAE has
    to encode it the way it encodes video.

    The two halves differ on purpose: the DiT gets every latent frame the VAE produced, while Qwen is
    shown the clip at 2 fps.
    """
    settings = settings or VideoSettings()
    if geometry is None:
        geometry = plan(frames[0] if frames else clips[0].frame(0), settings)
    model = models.registry.acquire(video_vae_spec(settings))
    references: list[pipeline.Reference] = []
    presented: list[text_encoder.PresentedReference] = []
    described: list[str] = []

    with ui.step(f"Encoding {len(frames) + len(clips) + len(soundtracks)} reference(s)") as detail:
        for frame in frames:
            width, height = pipeline.reference_size(frame.width, frame.height, geometry, settings.reference_size)
            resized = pipeline.resize_to_canvas(frame.batched(), pipeline.Geometry(width, height, geometry.frame_count))
            references.append(pipeline.Reference(vae.encode_frame(model, resized.permute(0, 3, 1, 2)), "image"))
            presented.append(text_encoder.PresentedReference.image(resized))
            described.append(f"image {width}×{height}")

        for clip in clips:
            pixels, count = _fit_reference_clip(clip, geometry, clip_scale)
            latent = vae.encode_clip(model, pixels.permute(0, 3, 1, 2))
            sound = _encode_reference_audio(clip, count, settings) if clip.audio is not None else None
            references.append(pipeline.Reference(latent, "video", sound))
            # Qwen sees it at 2 fps: one frame every FPS//2, timestamped half a second apart.
            sampled = pixels[:: h3_config.FPS // 2]
            presented.append(text_encoder.PresentedReference.video(sampled, has_audio=sound is not None))
            heard = f", {sound.shape[-1]} audio latents" if sound is not None else ""
            described.append(f"video {count}f → {latent.shape[2]} latent, {len(sampled)} shown{heard}")

        for waveform in soundtracks:
            sound = vae.encode_audio(models.registry.acquire(audio_vae_spec(settings)), waveform)
            references.append(pipeline.Reference(None, "audio", sound))
            presented.append(text_encoder.PresentedReference.audio())
            described.append(f"audio {sound.shape[-1]} latents")
        detail(" · ".join(described))
    return references, presented


def _fit_reference_clip(clip: Clip, geometry: pipeline.Geometry, scale: float = 1.0) -> tuple[torch.Tensor, int]:
    """Cut and rescale a reference clip to something the VAE and the layout both accept.

    The frame count lands on the `17k+5` grid the VAE encodes in by trimming rather than padding — a
    repeated tail would read as the subject freezing — and is capped at the target's length, since a
    reference longer than the clip it steers costs rows for nothing.

    `scale` shrinks it in space, as a fraction of the canvas's own edges, and it is the only knob that
    makes a long reference affordable: a clip costs its latent frames times the canvas's rows, so a
    12 s reference beside a 12 s chunk is 82,656 rows against a budget of about 101,000 for the whole
    sequence. At 0.5 it is a quarter of that, for the same seconds in less detail.
    """
    if min(len(clip), geometry.frame_count) < pipeline.FRAME_PHASE:
        raise ValueError(
            f"a reference clip needs at least {pipeline.FRAME_PHASE} frames "
            f"(~{pipeline.FRAME_PHASE / h3_config.FPS:.2f}s), got {len(clip)}"
        )
    count, width, height = pipeline.fitted_clip(len(clip), geometry, scale)
    frames = clip.frames[:count]
    return pipeline.resize_to_canvas(frames, pipeline.Geometry(width, height, count)), count


def _encode_reference_audio(clip: Clip, frames: int, settings: VideoSettings) -> torch.Tensor:
    """Encode a reference clip's soundtrack, cut to the frames that survived the grid trim: the layout
    gives a block one cursor for both halves, so a longer soundtrack would claim time its own frames
    never cover."""
    assert clip.audio is not None
    trimmed = trim_to_frames(clip.audio, frames, clip.fps, clip.sample_rate)
    return vae.encode_audio(models.registry.acquire(audio_vae_spec(settings)), trimmed)


def encode(
    prompt: str,
    canvas: torch.Tensor | None,
    references: Sequence[text_encoder.PresentedReference] = (),
    settings: VideoSettings | None = None,
    last_canvas: torch.Tensor | None = None,
) -> text_encoder.Conditioning:
    """References are presented after the keyframes as `<Picture 2>` (or 3, after a last frame),
    `<Video 1>` and so on, so a prompt can name them."""
    model = models.registry.acquire(text_encoder_spec(settings or VideoSettings()))
    with ui.step("Encoding prompt") as detail:
        encoded = text_encoder.encode(
            model, text_encoder.load_tokenizer(), prompt, canvas, references=references, last_image=last_canvas
        )
        detail(f"{encoded.embeds.shape[1]} rows")
    return encoded


def sample(
    conditioning: text_encoder.Conditioning,
    first_latent: torch.Tensor | None,
    geometry: pipeline.Geometry,
    settings: VideoSettings | None = None,
    prompt: str = "",
    references: Sequence[pipeline.Reference] = (),
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    last_latent: torch.Tensor | None = None,
    carry: pipeline.Carry | None = None,
) -> Latents:
    """The denoising loop. ~78% of a render.

    Each step's guess at the finished clip goes to the dashboard as a preview (`tools.previews`), and
    to `on_step` if given - which may raise to abandon the loop.
    """
    settings = settings or VideoSettings()
    # A no-op unless a recipe opened a tally naming this phase, so a take - which runs under the
    # director's stages, or under none - is unaffected.
    ui.stage("sampling")
    dit = models.registry.acquire(dit_spec(settings))
    steps = settings.steps
    previewer = preview.for_sampling(settings)

    def step_done(index: int, video: torch.Tensor, audio: torch.Tensor) -> None:
        if previewer is not None:
            previewer(index, video, audio)
        if on_step is not None:
            on_step(index, video, audio)

    blocks = dit.transformer_blocks

    def run(resident: bool) -> tuple[torch.Tensor, torch.Tensor]:
        with ui.sampling("Sampling", steps, len(blocks)) as tick:
            hooks = [block.register_forward_hook(lambda *_: tick()) for block in blocks]
            try:
                return pipeline.sample(
                    dit,
                    conditioning,
                    first_latent,
                    geometry,
                    steps=steps,
                    seed=settings.seed,
                    shift_video=settings.shift_video,
                    shift_audio=settings.shift_audio,
                    sampler=settings.sampler,
                    schedule=settings.schedule,
                    references=references,
                    on_step=step_done if previewer is not None or on_step is not None else None,
                    last_frame_latent=last_latent,
                    resident=resident,
                    carry=carry,
                )
            finally:
                for hook in hooks:
                    hook.remove()

    try:
        try:
            video, audio = run(resident=True)
        except torch.OutOfMemoryError:
            if not offload.release():
                raise
            ui.note("out of memory with DiT blocks resident: sampling again with every block streaming")
            video, audio = run(resident=False)
    finally:
        if previewer is not None:
            previewer.close()
    return Latents(video, audio, geometry, settings, prompt)


def decode(latents: Latents) -> Clip:
    ui.stage("decoding")
    offload.release()
    video_model = models.registry.acquire(video_vae_spec(latents.settings))
    clips = vae.decode_clip_count(video_model, int(latents.video.shape[2]))
    # From the latent itself rather than the geometry: an enlarged latent decodes bigger than it sampled.
    height, width = (int(side) * pipeline.SPATIAL_COMPRESSION for side in latents.video.shape[-2:])
    label = f"Decoding video {width}×{height}"
    with ui.progress(label, clips, unit="clip") as advance:
        frames = vae.decode_video(video_model, latents.video, on_clip=advance)
    audio_model = models.registry.acquire(audio_vae_spec(latents.settings))
    with ui.step("Decoding audio") as detail:
        waveform = vae.decode_audio(audio_model, latents.audio)
        detail(f"{waveform.shape[1] / h3_config.AUDIO_SAMPLE_RATE:.2f}s stereo")
    return Clip(
        frames=frames,
        audio=waveform,
        fps=h3_config.FPS,
        sample_rate=h3_config.AUDIO_SAMPLE_RATE,
        prompt=latents.prompt,
    )


def render(
    frame: Frame | None,
    prompt: str,
    settings: VideoSettings | None = None,
    references: Sequence[Frame] = (),
    reference_clips: Sequence[Clip] = (),
    reference_audio: Sequence[torch.Tensor] = (),
    canvas_from: Frame | None = None,
    closing: Frame | None = None,
) -> Clip:
    """All the stages, for when you just want the clip.

    `frame` alone is `fl2va`: the clip opens on it. `references` (stills) and `reference_clips` (motion)
    are `ref2va`: they steer the clip while leaving it free to open anywhere, and the canvas comes from
    `canvas_from` or the first reference. Any combination is legal, and a first frame plus a reference
    is what a chained long video wants. `closing` is the frame the clip ends on.
    """
    settings = settings or VideoSettings()
    if frame is None and not references and not reference_clips:
        raise ValueError("give a frame (fl2va), references (ref2va), or both")
    shape_source = canvas_from or frame or (references[0] if references else reference_clips[0].frame(0))
    geometry = plan(shape_source, settings)

    first_latent, canvas = (None, None)
    if frame is not None:
        first_latent, canvas = condition(frame, settings, geometry)
    encoded_references: list[pipeline.Reference] = []
    presented: list[text_encoder.PresentedReference] = []
    if references or reference_clips or reference_audio:
        encoded_references, presented = encode_references(
            references, settings, geometry, reference_clips, reference_audio
        )

    last_latent, last_canvas = condition_last(closing, settings, geometry) if closing is not None else (None, None)

    ui.stage("encoding")
    conditioning = encode(prompt, canvas, presented, settings, last_canvas)
    latents = sample(
        conditioning, first_latent, geometry, settings, prompt, encoded_references, last_latent=last_latent
    )
    return decode(latents)


def chunk_key(
    opening: Frame | None,
    prompt: str,
    settings: VideoSettings,
    references: Sequence[Frame] = (),
    tail: Clip | None = None,
    clips: Sequence[Clip] = (),
    clip_scale: float = 1.0,
    closing: Frame | None = None,
    canvas_from: Frame | None = None,
    carry: pipeline.Carry | None = None,
) -> Key:
    """What identifies a chunk: its opening, prompt, settings, references, tail and any reference
    clips - pictures by their pixels, never by their paths, so renaming a photo renders nothing again.
    A part added here moves every chunk, which is what the version is for; `closing` and `canvas` are
    keyed only when given, so they moved none. The canvas is keyed only without an opening, which
    otherwise decides the shape itself."""
    canvas = (canvas_from.width, canvas_from.height) if opening is None and canvas_from is not None else None
    return key(
        "render_chunk",
        1,
        opening=opening.pixels if opening is not None else None,
        prompt=prompt,
        settings=settings,
        references=[r.pixels for r in references],
        tail=tail.frames if tail is not None else None,
        clips=[clip.frames for clip in clips],
        clip_scale=clip_scale,
        **({"closing": closing.pixels} if closing is not None else {}),
        **({"canvas": canvas} if canvas is not None else {}),
        **(
            {"carry": [carry.video, carry.audio, float(carry.audio_start), carry.audio_late], "held": 1}
            if carry is not None
            else {}
        ),
    )


def cached_latents(
    opening: Frame | None,
    prompt: str,
    settings: VideoSettings,
    references: Sequence[Frame] = (),
    tail: Clip | None = None,
    on_step: StepHook | None = None,
    cache: Cache | None = None,
    clips: Sequence[Clip] = (),
    clip_scale: float = 1.0,
    canvas_from: Frame | None = None,
    closing: Frame | None = None,
    carry: pipeline.Carry | None = None,
) -> Latents:
    """One chunk's latents, read back when this exact opening, prompt and history already ran - so a
    workflow that samples the same thing twice pays once, and a run that dies resumes where it stopped.

    Latents are what gets cached, not frames. They are the expensive half - decoding is cheap and
    deterministic - and they are small: a 15 s chunk is about 17 MB against gigabytes once it has been
    decoded at 2 MP.
    """
    made = chunk_key(opening, prompt, settings, references, tail, clips, clip_scale, closing, canvas_from, carry)
    # Without a keyframe there is nothing to plan the canvas from, so the shape comes from a frame
    # that is only ever measured: ref2va fixes no opening, and that is the whole point of using it.
    shape = opening or canvas_from
    if shape is None:
        raise ValueError("give an opening frame, or a canvas_from to take the shape from")
    geometry = plan(shape, settings)
    if carry is not None and carry.video.shape[-2:] != (geometry.latent_height, geometry.latent_width):
        raise ValueError(
            f"the carried steps are {tuple(carry.video.shape[-2:])} latents, "
            f"this canvas is {geometry.latent_height}x{geometry.latent_width}"
        )

    def produce() -> dict[str, torch.Tensor]:
        first_latent, conditioning, encoded, last_latent = _conditions(
            opening, prompt, settings, geometry, references, held, clip_scale, closing, carry
        )
        sampled = sample(conditioning, first_latent, geometry, settings, prompt, encoded, on_step, last_latent, carry)
        video = held_to(sampled.video, carry.video) if carry is not None else sampled.video
        return {"video": video, "audio": sampled.audio}

    # The order is the labelling: what a chunk is steered by is <Video 1>, and the tail carrying the
    # previous chunk's motion follows it. A caption written for one chunk then means the same thing
    # in every chunk, which is what `Reel.check` refuses to let drift.
    held = [*clips, *([tail] if tail else [])]
    store = cache if cache is not None else default_cache
    stored = store.tensors(made, produce)
    return Latents(stored["video"], stored["audio"], geometry, settings, prompt)


def _conditions(
    opening: Frame | None,
    prompt: str,
    settings: VideoSettings,
    geometry: pipeline.Geometry,
    references: Sequence[Frame],
    held: list[Clip],
    clip_scale: float,
    closing: Frame | None,
    carry: pipeline.Carry | None,
) -> tuple[torch.Tensor | None, text_encoder.Conditioning, list[pipeline.Reference], torch.Tensor | None]:
    """What a chunk is sampled under: its first frame's latent, the prompt, the references and the last frame."""
    if carry is not None:
        # The text encoder still sees the opening as <Picture 1>; the DiT has the carried steps instead.
        first_latent, canvas = None, pipeline.resize_to_canvas(opening.batched(), geometry) if opening else None
    elif opening is not None:
        first_latent, canvas = condition(opening, settings, geometry)
    else:
        first_latent, canvas = None, None
    encoded, presented = models.registry.roomy(
        video_vae_spec(settings),
        lambda: encode_references(references, settings, geometry, held, clip_scale=clip_scale),
    )
    last_latent, last_canvas = condition_last(closing, settings, geometry) if closing is not None else (None, None)
    conditioning = encode(prompt, canvas, presented, settings, last_canvas)
    return first_latent, conditioning, encoded, last_latent


# The band split of noEmbryo's H3 Clip Refiner (ComfyUI-noEmbryo).
BAND_SIGMA = 2.0


# The most a carried chunk is corrected by, either way. Measured drift was 3-4% a join; a tenfold margin.
HELD_LIMIT = (0.8, 1.25)


def held_to(video: torch.Tensor, carried: torch.Tensor) -> torch.Tensor:
    """`video` corrected by how far its redrawn opening steps drifted from the `carried` steps they were
    pinned to: per channel its mean and spread, then its fine detail.

    The two are the same frames, so whatever differs is H3's own drift, not the scene. Uncorrected, the
    redrawn steps came back with 3.2-3.6% more high band at every join of production 18, and the next
    shot carries that on: x1.16 after six shots. noEmbryo's Clip Refiner matches the same statistics to
    the first clip's, where the scene has moved on.
    """
    steps = carried.shape[2]
    head, carried = video[:, :, :steps].float(), carried.float()
    dims = (0, 2, 3, 4)
    scale = (carried.std(dim=dims) / head.std(dim=dims).clamp_min(1e-6)).clamp(*HELD_LIMIT)
    shift = carried.mean(dim=dims) - scale * head.mean(dim=dims)
    video = video.float() * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)
    gain = _bands(carried)[1].std() / _bands(video[:, :, :steps])[1].std().clamp_min(1e-6)
    low, high = _bands(video)
    return low + float(gain.clamp(*HELD_LIMIT)) * high


def _bands(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(low, high)` of each latent frame: a separable Gaussian blur, replicate-padded, and the rest."""
    radius = round(3 * BAND_SIGMA)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2 * BAND_SIGMA**2))
    kernel = kernel / kernel.sum()
    flat = video.float().reshape(-1, 1, *video.shape[-2:])
    low = torch.nn.functional.conv2d(
        torch.nn.functional.pad(flat, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1)
    )
    low = torch.nn.functional.conv2d(
        torch.nn.functional.pad(low, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1)
    )
    low = low.reshape(video.shape)
    return low, video.float() - low
