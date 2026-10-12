"""Canvas and frame arithmetic, the packed sequence, and the few-step sampling loop.

The sequence the DiT sees is `[text | keyframe rows | reference rows | audio | video]`. Each row carries a modality
tag (video 0, text 1, audio 2), a `(t, h, w)` rotary coordinate and a timestep index, which is how one
forward pass serves rows at different noise levels: the first-frame rows are pinned just short of
clean at t=0.999 and are never denoised, the video and audio rows follow their own schedules (flow
shift 12 and 3), and the text rows ride the video one.

diffusers' `build_packed_sequence` produces the layout; the loop, the frame arithmetic and the
conditioning rows are here.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.stats
import torch
from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    MiniMaxH3PrepareLayoutStep,
    _frame_position_grid,
    _temporal_position_grid,
    patchify_video_latents,
)

from difflab import config as app_config
from difflab.core import profiling, ui
from difflab.models import offload
from difflab.models.minimax_h3 import config
from difflab.models.minimax_h3.text_encoder import Conditioning

PATCH_SIZE = (1, 2, 2)
SPATIAL_COMPRESSION = 16
AUDIO_CHANNELS = 2
VIDEO_TAG, AUDIO_TAG = 0, 2  # text is 1
# The first frame is held here rather than at a clean 0: "just short of clean", as the model was trained.
CONDITION_TIMESTEP = 0.999
# A reference soundtrack is pinned at exactly clean, which is also why it is handed over un-noised.
AUDIO_CONDITION_TIMESTEP = 1.0
FRAME_CYCLE, FRAME_PHASE = config.FRAME_CYCLE, config.FRAME_PHASE
# The VAE's chunked decoder needs at least this many latent frames (clip_length 17, token_drop 3).
MIN_LATENT_FRAMES = 7
# ref2va's "max" sizing: the reference pipeline's short edge, for identity fidelity over speed.
REFERENCE_SHORT_EDGE = 2048


@dataclass(frozen=True)
class Geometry:
    """Everything the layout needs, derived once from the input image and the requested duration."""

    width: int
    height: int
    frame_count: int

    @property
    def latent_height(self) -> int:
        return self.height // SPATIAL_COMPRESSION

    @property
    def latent_width(self) -> int:
        return self.width // SPATIAL_COMPRESSION

    @property
    def latent_frames(self) -> int:
        if self.frame_count <= FRAME_PHASE:
            return 2
        return ((self.frame_count - FRAME_PHASE) // FRAME_CYCLE) * 5 + 2

    @property
    def audio_latents(self) -> int:
        return round(self.frame_count / config.FPS * config.AUDIO_LATENT_FPS)

    @property
    def rows_per_frame(self) -> int:
        return (self.latent_height // PATCH_SIZE[1]) * (self.latent_width // PATCH_SIZE[2])

    @property
    def duration(self) -> float:
        return self.frame_count / config.FPS


def align_frame_count(frames: int) -> int:
    """The smallest allowed frame count at or above `frames`: `max(5, n)` rounded up to `n % 17 == 5`."""
    count = max(FRAME_PHASE, frames)
    return count + (FRAME_PHASE - (count % FRAME_CYCLE)) % FRAME_CYCLE


def canvas_size(width: int, height: int, megapixels: float) -> tuple[int, int]:
    """ComfyUI's ImageScaleToTotalPixels: scale to the pixel budget, then round both axes to 32."""
    scale = math.sqrt(megapixels * 1024 * 1024 / (width * height))
    align = app_config.CANVAS_ALIGN
    return max(align, round(width * scale / align) * align), max(align, round(height * scale / align) * align)


def settling_canvas(width: int, height: int, duration: float, megapixels: float) -> Geometry:
    """The canvas a region of this size renders at, re-planned until it plans to itself.

    `canvas_size` rounds each side to 32 on its own, so a frame resized to the canvas it planned can
    plan a different one (672x1216 plans 672x1248 at 0.8 MP) and the render would stretch a take's
    first frame to it. Planned again from the canvas, it settles - usually at once, and this raises
    rather than looping if it does not.
    """
    canvas = plan(width, height, duration, megapixels)
    for _ in range(4):
        again = plan(canvas.width, canvas.height, duration, megapixels)
        if again == canvas:
            return canvas
        canvas = again
    raise RuntimeError(f"no canvas near {width}x{height} plans itself again at {megapixels} MP")


def fitted_clip(frames: int, geometry: Geometry, scale: float = 1.0) -> tuple[int, int, int]:
    """How a reference clip is encoded: its frame count on the VAE's grid, and the size it is drawn at.

    One function because two callers need the same answer and must not disagree: the encoder, which
    does it, and `Reel.rows`, which prices it before a model is loaded. A clip that costs more rows
    than the card has should be refused on the CPU, not discovered at the first sampling step.
    """
    count = min(frames, geometry.frame_count)
    while count % FRAME_CYCLE != FRAME_PHASE:
        count -= 1
    align = app_config.CANVAS_ALIGN
    width = max(align, round(geometry.width * scale / align) * align)
    height = max(align, round(geometry.height * scale / align) * align)
    return count, width, height


def tail_frames(seconds: float) -> int:
    """How many frames of motion a chunk carries into the next, rounded down to a count the VAE can
    encode (`n % 17 == 5`). Anything under the five-frame minimum is no tail at all rather than a
    silently shortened one."""
    wanted = int(seconds * config.FPS)
    if wanted < FRAME_PHASE:
        return 0
    while wanted % FRAME_CYCLE != FRAME_PHASE:
        wanted -= 1
    return wanted


def plan(width: int, height: int, duration: float, megapixels: float) -> Geometry:
    canvas_width, canvas_height = canvas_size(width, height, megapixels)
    geometry = Geometry(canvas_width, canvas_height, align_frame_count(round(duration * config.FPS)))
    if geometry.latent_frames < MIN_LATENT_FRAMES:
        raise ValueError(
            f"{duration}s gives {geometry.frame_count} frames ({geometry.latent_frames} latent), below the "
            f"{MIN_LATENT_FRAMES} the VAE decoder needs. Ask for at least {(MIN_LATENT_FRAMES - 2) // 5 * 17 + 5} "
            f"frames (~{((MIN_LATENT_FRAMES - 2) // 5 * 17 + 5) / config.FPS:.2f}s)."
        )
    return geometry


def resize_to_canvas(image: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    """`[N, H, W, 3]` -> `[N, canvas_h, canvas_w, 3]`: the first frame, a reference, or a tail's frames.

    Antialiased bicubic, as ComfyUI's H3 nodes resize all three (they use lanczos, which torch lacks).
    Nearest-exact aliased hair, fur and faces at every join of a long take, where each handoff frame
    and tail comes from a 2 MP decode and shrinks 1.4x to the canvas. A picture already at the canvas
    size comes back unchanged.
    """
    pixels = image.permute(0, 3, 1, 2)
    size = (geometry.height, geometry.width)
    resized = torch.nn.functional.interpolate(pixels, size=size, mode="bicubic", antialias=True).clamp(0, 1)
    return resized.permute(0, 2, 3, 1)


def unpatchify(rows: torch.Tensor, geometry: Geometry, channels: int = 24) -> torch.Tensor:
    """Inverse of `patchify_video_latents`: rows `[N, 96]` -> latents `[1, 24, T, h, w]`."""
    patch_t, patch_h, patch_w = PATCH_SIZE
    frames, height, width = geometry.latent_frames, geometry.latent_height, geometry.latent_width
    latents = rows.reshape(
        1, frames // patch_t, height // patch_h, width // patch_w, channels, patch_t, patch_h, patch_w
    )
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return latents.reshape(1, channels, frames, height, width)


def unpack_audio(rows: torch.Tensor, geometry: Geometry, channels: int = 32) -> torch.Tensor:
    """Rows `[2 * T, 32]`, channel-major -> latents `[1, 32, 2, T]`."""
    stereo = rows.reshape(AUDIO_CHANNELS, geometry.audio_latents, channels)
    return stereo.permute(2, 0, 1).unsqueeze(0)


def last_anchor_time(origin: float, latent_frames: int) -> float:
    """Where a last keyframe sits: the final pixel frame's time, three rescales past the last latent's own.

    Summed pairwise by numpy, as diffusers' `build_packed_sequence` does for `"last"`; a sequential sum
    differs in the last ulp from 16 latent frames on.
    """
    spans = np.ones(latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
    return float(origin + spans.sum() - _ROPE_FRAME_RESCALE)


def video_time_spans(latent_frames: int) -> list[float]:
    """The rotary time each latent frame of a clip occupies: `5/3 * (1, 4, 4, 4, 4)`, cycling."""
    return [
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
        for index in range(latent_frames)
    ]


@dataclass(frozen=True)
class Reference:
    """One encoded reference: its latent, and the grid that latent sits on.

    A reference keeps its own size — it is not resized to the target canvas — so it carries its own
    latent height and width and gets its own spatial rotary grid. A `"video"` reference is the same
    thing with more than one latent frame, and its rows lie along the time axis the way the target's
    do rather than stacking at a single instant.
    """

    latent: torch.Tensor | None  # [1, 24, t, h, w], normalized, as the video VAE produced it
    kind: str = "image"  # "image" | "video" | "audio"
    audio_latent: torch.Tensor | None = None  # [1, 32, 2, T], a clip's own soundtrack

    @property
    def latent_frames(self) -> int:
        return 0 if self.latent is None else int(self.latent.shape[2])

    @property
    def audio_latents(self) -> int:
        return 0 if self.audio_latent is None else int(self.audio_latent.shape[-1])

    def audio_rows(self, audio_channels: int = AUDIO_CHANNELS) -> int:
        return self.audio_latents * audio_channels

    @property
    def latent_height(self) -> int:
        return 0 if self.latent is None else int(self.latent.shape[-2])

    @property
    def latent_width(self) -> int:
        return 0 if self.latent is None else int(self.latent.shape[-1])

    def rows(self, patch_h: int, patch_w: int) -> int:
        if self.latent is None:
            return 0
        return self.latent_frames * (self.latent_height // patch_h) * (self.latent_width // patch_w)

    @property
    def span(self) -> float:
        """The time axis this reference occupies ahead of the target streams.

        A still is worth exactly one unit however large it is; a clip is worth what its frames span,
        summed sequentially — the order the reference sums it, which differs from a pairwise sum in
        the last ulp once a clip runs past 16 latent frames. A soundtrack is worth one unit per audio
        latent, and a clip that has one claims whichever of the two reaches further: the streams share
        a clock, so the block ends when its slower half does.
        """
        if self.kind == "image":
            return 1.0
        if self.kind == "audio":
            return float(self.audio_latents)
        return max(float(self.audio_latents), sum(video_time_spans(self.latent_frames)))


@dataclass(frozen=True)
class Carry:
    """The previous clip's last frames and sound, pinned on this clip's own timeline: the frames over its
    first latent steps, the sound ending where the previous clip's picture ended.

    Pinned rather than shown as a reference clip: as a reference the model imitates the sound rather
    than continuing it, which NikoDemon80's H3-Motion-Context measured as a join cross-correlation of
    about 0.45 against 0.95 pinned, and the picture would go through the VAE a second time.
    """

    video: torch.Tensor  # [1, 24, k, h, w], k latent steps from a cycle start
    audio: torch.Tensor  # [1, 32, 2, n]
    audio_start: int  # where the first audio latent sits, in audio latents after the target's origin
    audio_late: float = 0.0  # latents the carried sound lands after where its picture puts it, under 1/2

    @property
    def latent_frames(self) -> int:
        return int(self.video.shape[2])

    @property
    def audio_latents(self) -> int:
        return int(self.audio.shape[-1])


def carry_of(video: torch.Tensor, audio: torch.Tensor, frames: int, audio_latents: int) -> Carry:
    """The last `frames` pixel frames of a clip's latents (`n % 17 == 5`) and its last `audio_latents`.

    The window starts on a cycle's one-frame latent, so its steps are the ones the next clip's first
    steps are: copied, never decoded and encoded again. The sound sits on the audio grid nearest to its
    offset from the picture, which the two grids only agree on every three frames: placed at a fraction,
    H3 redrew it on the grid anyway (production 18, a third of a latent: 268 samples late at 0.925
    correlation, -0.56 at the cut's own sample). `audio_late` is what the soundtrack is moved back by.
    """
    steps = Geometry(0, 0, frames).latent_frames
    total = int(video.shape[2])
    if frames % FRAME_CYCLE != FRAME_PHASE or (total - steps) % 5 or steps > total:
        raise ValueError(f"cannot carry {frames} frames of a {total}-step clip")
    clip_frames = (total - 2) // 5 * FRAME_CYCLE + FRAME_PHASE
    count = min(audio_latents, int(audio.shape[-1]))
    exact = int(audio.shape[-1]) - count - (clip_frames - frames) * _ROPE_FRAME_RESCALE
    start = round(exact)
    return Carry(video[:, :, total - steps :].clone(), audio[..., -count:].clone(), start, start - exact)


def reference_size(width: int, height: int, geometry: Geometry, mode: str = "match") -> tuple[int, int]:
    """How large a reference image is encoded, rounded to the canvas multiple. Never upscales.

    `"match"` scales the reference to the generation's pixel area, which keeps its token count in the
    same range as the target's. `"max"` uses the reference pipeline's 2048-pixel short edge instead:
    better identity fidelity, but reference rows ride through every sampling step, so a `max`
    reference can cost more than the clip it is steering.
    """
    if mode == "match":
        scale = min(1.0, math.sqrt((geometry.width * geometry.height) / (width * height)))
    elif mode == "max":
        scale = min(1.0, REFERENCE_SHORT_EDGE / min(width, height))
    else:
        raise ValueError(f"reference size mode must be 'match' or 'max', got {mode!r}")
    align = app_config.CANVAS_ALIGN
    return (
        max(align, round(width * scale / align) * align),
        max(align, round(height * scale / align) * align),
    )


def build_reference_layout(
    text_token_tags: torch.Tensor,
    geometry: Geometry,
    references: Sequence[Reference],
    first_frame: bool = False,
    last_frame: bool = False,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
    audio_channels: int = AUDIO_CHANNELS,
    carry: Carry | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """The `ref2va` layout: `[text | first frame | reference blocks | target audio | target video]`.

    diffusers builds `t2va` and `fl2va` only, so this is the third arrangement, and the shape of it is
    what makes references work: each block advances a cursor along the rotary time axis, and the
    target streams begin *after* every reference has had its turn. A reference is therefore not a
    frame of the video being made — it sits in the model's past, which is why it steers identity
    without pinning the clip's opening to it.

    `first_frame` adds an `fl2va` keyframe alongside the references. The DiT's layout has always
    allowed both — ComfyUI simply exposes them as two nodes — and together they are what a chained
    long video wants: the previous chunk's last frame for continuity, plus a fixed reference so the
    subject stops drifting. The keyframe shares the target streams' origin, which is why it is placed
    only once every reference span has been counted.

    `last_frame` pins the clip's end the same way, at the target's final frame; its rows follow the
    first frame's, the order diffusers packs `fl2va`'s two keyframes in.

    `carry` pins the previous clip's last steps over the target's first ones, ahead of every other
    keyframe, and its sound beside them.

    Reference and keyframe rows alike are video rows that are never denoised, so the returned
    condition-row count covers both and the sampler leaves them alone.
    """
    _, patch_h, patch_w = patch_size
    rows_per_frame = (geometry.latent_height // patch_h) * (geometry.latent_width // patch_w)
    num_text_tokens = int(text_token_tags.shape[0])
    reference_rows = [reference.rows(patch_h, patch_w) for reference in references]
    reference_audio_rows = [reference.audio_rows(audio_channels) for reference in references]
    carried = carry.latent_frames if carry is not None else 0
    carried_audio = carry.audio_latents * audio_channels if carry is not None else 0
    keyframe_rows = rows_per_frame * (carried + int(first_frame) + int(last_frame))
    num_condition_rows = keyframe_rows + sum(reference_rows)
    num_audio_condition_rows = carried_audio + sum(reference_audio_rows)
    num_audio_rows = geometry.audio_latents * audio_channels
    num_video_rows = geometry.latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_condition_rows + num_audio_rows + num_video_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    target_grid, target_width_grid = _frame_position_grid(
        geometry.latent_height, geometry.latent_width, patch_h, patch_w
    )

    def fill_audio(start: int, latents: int, origin: float, low: float, high: float) -> None:
        """Channel-major stereo rows: time advances per latent, width pins to the grid's extremes."""
        position_ids[start : start + latents * audio_channels, 0] = (
            origin + torch.arange(latents, dtype=torch.float64)
        ).repeat(audio_channels)
        position_ids[start : start + latents, 2] = low
        position_ids[start + latents : start + latents * audio_channels, 2] = high

    # The target timeline starts after every reference has had its span.
    origin = float(num_text_tokens) + sum(reference.span for reference in references)
    row = num_text_tokens

    # A keyframe is one of the clip's own frames, so it sits on the target's clock, not before it.
    if carry is not None:
        times = _temporal_position_grid(carried, origin)
        for time in times:
            position_ids[row : row + rows_per_frame, 0] = time
            position_ids[row : row + rows_per_frame, 1:] = target_grid
            row += rows_per_frame
    for anchored, time in ((first_frame, origin), (last_frame, last_anchor_time(origin, geometry.latent_frames))):
        if anchored:
            position_ids[row : row + rows_per_frame, 0] = time
            position_ids[row : row + rows_per_frame, 1:] = target_grid
            row += rows_per_frame

    # Reference rows in sequence order. A block's soundtrack packs immediately before its own video
    # rows and shares their origin, so the two halves of one reference describe the same moment.
    video_condition_spans: list[torch.Tensor] = [torch.arange(num_text_tokens, num_text_tokens + keyframe_rows)]
    audio_condition_spans: list[torch.Tensor] = []
    if carry is not None:
        low, high = float(target_width_grid[0]), float(target_width_grid[-1])
        fill_audio(row, carry.audio_latents, origin + carry.audio_start, low, high)
        audio_condition_spans.append(torch.arange(row, row + carried_audio))
        row += carried_audio
    cursor = float(num_text_tokens)
    for reference, rows, audio_rows in zip(references, reference_rows, reference_audio_rows, strict=True):
        grid, width_grid = (
            _frame_position_grid(reference.latent_height, reference.latent_width, patch_h, patch_w)
            if reference.latent is not None
            else (target_grid, target_width_grid)
        )
        if audio_rows:
            fill_audio(row, reference.audio_latents, cursor, float(width_grid[0]), float(width_grid[-1]))
            audio_condition_spans.append(torch.arange(row, row + audio_rows))
            row += audio_rows
        if rows:
            # A still is one instant, a clip walks the same non-uniform frame clock the target does.
            times = _temporal_position_grid(reference.latent_frames, cursor)
            block = torch.empty(reference.latent_frames, rows // reference.latent_frames, 3, dtype=torch.float64)
            block[:, :, 0] = times[:, None]
            block[:, :, 1:] = grid[None]
            position_ids[row : row + rows] = block.reshape(-1, 3)
            video_condition_spans.append(torch.arange(row, row + rows))
            row += rows
        cursor += reference.span

    audio_start, video_start = row, row + num_audio_rows
    fill_audio(audio_start, geometry.audio_latents, origin, float(target_width_grid[0]), float(target_width_grid[-1]))

    video_position_ids = torch.empty(geometry.latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(geometry.latent_frames, origin)[:, None]
    video_position_ids[:, :, 1:] = target_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    # Conditioning rows come first within each modality, so the sampler can leave a leading run alone.
    video_indices = torch.cat([*video_condition_spans, torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat([*audio_condition_spans, torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_rows,
        num_audio_condition_rows,
    )


def build_row_timesteps(
    sequence_length: int,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    condition_rows: int,
    video_timestep: float,
    audio_timestep: float,
    audio_condition_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One timestep per row, deduplicated.

    Every row starts at the video timestep — text rows never reach an output head and inherit it, as
    the reference does — then the audio rows take their own schedule and the conditioning rows are
    pinned. The two modalities pin to different values: visual conditioning sits just short of clean
    at 0.999, while audio conditioning is pinned *at* 1.0, which is also why it is never noised.

    Conditioning rows lead their own modality's index list, so a count is enough to find them even
    when a reference's audio and video rows are interleaved in the packed sequence.
    """
    row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
    row_timesteps[audio_indices] = audio_timestep
    if condition_rows:
        row_timesteps[video_indices[:condition_rows]] = max(video_timestep, CONDITION_TIMESTEP)
    if audio_condition_rows:
        row_timesteps[audio_indices[:audio_condition_rows]] = max(audio_timestep, AUDIO_CONDITION_TIMESTEP)
    unique, inverse = torch.unique(row_timesteps, sorted=True, return_inverse=True)
    return unique, inverse


def beta_sigmas(steps: int, shift: float, alpha: float = 0.5, beta: float = 0.7) -> torch.Tensor:
    """ComfyUI's `beta_scheduler` over `ModelSamplingDiscreteFlow`'s 1000-entry table; RES4LYF's beta57 is
    alpha 0.5, beta 0.7."""
    ts = np.rint(scipy.stats.beta.ppf(1 - np.linspace(0, 1, steps, endpoint=False), alpha, beta) * 999)
    t = torch.unique_consecutive(torch.from_numpy(ts)).double().add(1).div(1000)
    return torch.cat([shift * t / (1 + (shift - 1) * t), t.new_zeros(1)]).float()


def make_schedulers(
    steps: int, shift_video: float, shift_audio: float, schedule: str = "beta57"
) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler]:
    """Two rectified-flow schedules over the same grid, one per modality.

    `set_timesteps(n)` lays out `n` sigma *grid points* and exposes `n - 1` timesteps, so an 8-step
    schedule needs 9 points to give 8 model evaluations.
    """
    video = MiniMaxH3Scheduler(shift=shift_video)
    audio = MiniMaxH3Scheduler(shift=shift_audio)
    if schedule == "beta57":
        video.set_timesteps(sigmas=beta_sigmas(steps, shift_video))
        audio.set_timesteps(sigmas=beta_sigmas(steps, shift_audio))
    else:
        video.set_timesteps(steps + 1)
        audio.set_timesteps(steps + 1)
    return video, audio


class ErSde:
    """ComfyUI's `sample_er_sde` (VP ER-SDE-Solver-3, arXiv 2309.06169) on a rectified-flow schedule."""

    INTEGRATION_POINTS = 200

    def __init__(self, sigmas: torch.Tensor, generator: torch.Generator, max_stage: int = 3):
        # ComfyUI's offset_first_sigma_for_snr: sigma 1 has an infinite lambda.
        self.sigmas = sigmas.double().cpu().clamp(max=1 - 1e-4)
        self.sigmas[-1] = 0.0
        self.lambdas = self.sigmas / (1 - self.sigmas)
        self.generator = generator
        self.max_stage = max_stage
        self.old_denoised: torch.Tensor | None = None
        self.old_denoised_d: torch.Tensor | None = None

    @staticmethod
    def noise_scaler(x: torch.Tensor) -> torch.Tensor:
        return x * ((x**0.3).exp() + 10.0)

    def step(self, index: int, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        sigma_s, sigma_t = self.sigmas[index], self.sigmas[index + 1]
        if sigma_t == 0:
            return denoised
        lambdas, scaler = self.lambdas, self.noise_scaler
        lambda_s, lambda_t = lambdas[index], lambdas[index + 1]
        alpha_s, alpha_t = 1 - sigma_s, 1 - sigma_t
        r = scaler(lambda_t) / scaler(lambda_s)
        x = float(alpha_t / alpha_s * r) * x + float(alpha_t * (1 - r)) * denoised

        stage = min(self.max_stage, index + 1)
        if stage >= 2:
            dt = lambda_t - lambda_s
            step = -dt / self.INTEGRATION_POINTS
            positions = lambda_t + torch.arange(self.INTEGRATION_POINTS, dtype=torch.float64) * step
            scaled = scaler(positions)
            denoised_d = (denoised - self.old_denoised) / float(lambda_s - lambdas[index - 1])
            x = x + float(alpha_t * (dt + (1 / scaled).sum() * step * scaler(lambda_t))) * denoised_d
            if stage >= 3:
                s_u = ((positions - lambda_s) / scaled).sum() * step
                denoised_u = (denoised_d - self.old_denoised_d) / float((lambda_s - lambdas[index - 2]) / 2)
                x = x + float(alpha_t * (dt**2 / 2 + s_u * scaler(lambda_t))) * denoised_u
            self.old_denoised_d = denoised_d
        self.old_denoised = denoised

        spread = (lambda_t**2 - lambda_s**2 * r**2).clamp(min=0).sqrt()
        noise = torch.randn(x.shape, generator=self.generator, dtype=torch.float32).to(x.device)
        return x + float(alpha_t * spread) * noise


def pack_audio(latents: torch.Tensor) -> torch.Tensor:
    """Inverse of `unpack_audio`: `[1, 32, 2, T]` -> rows `[2 * T, 32]`, channel-major."""
    return latents[0].permute(1, 2, 0).reshape(-1, latents.shape[1]).contiguous()


def condition_rows_from_latent(
    latents: Sequence[torch.Tensor], seed: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Hold each conditioning latent at t=0.999 (`0.999 * latent + 0.001 * noise`).

    Patchify first, then noise: the reference draws its noise in *row* shape, and patchifying is a
    permutation, so drawing it in latent shape instead would place the same seed's values differently.
    Every conditioning restarts the same RNG stream, which is also what the reference does — several
    references therefore share one noise pattern rather than continuing a single stream.
    """
    rows = []
    for latent in latents:
        patched: torch.Tensor = patchify_video_latents(latent.float().cpu(), PATCH_SIZE)
        generator = torch.Generator("cpu").manual_seed(seed)
        noise = torch.randn(patched.shape, generator=generator, dtype=torch.float32)
        rows.append(CONDITION_TIMESTEP * patched + (1.0 - CONDITION_TIMESTEP) * noise)
    return torch.cat(rows, dim=0).to(dtype) if rows else torch.zeros(0, dtype=dtype)


def sample(
    transformer: MiniMaxH3Transformer3DModel,
    conditioning: Conditioning,
    first_frame_latent: torch.Tensor | None,
    geometry: Geometry,
    steps: int,
    seed: int,
    shift_video: float,
    shift_audio: float,
    sampler: str = "er_sde",
    schedule: str = "beta57",
    device: str | torch.device = app_config.DEVICE,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    references: Sequence[Reference] = (),
    last_frame_latent: torch.Tensor | None = None,
    resident: bool = True,
    carry: Carry | None = None,
    init: tuple[torch.Tensor, torch.Tensor] | None = None,
    denoise: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the loop and return `(video latents [1, 24, T, h, w], audio latents [1, 32, 2, T40])`.

    With a `first_frame_latent` alone this is `fl2va`: the clip opens on that exact frame. With
    `references` it is `ref2va`, where the references sit in the model's past and steer the clip
    without fixing its opening. Both together is legal and is what a chained long video wants. A
    `last_frame_latent` pins the clip's end to that frame as well, as diffusers' `last_image` does.
    `carry` pins the clip's opening steps to the previous clip's last ones; it replaces a first frame.
    `init` is a finished `(video, audio)` sampled again from `denoise` of the way: ComfyUI's meaning, the
    last `steps` of a `steps / denoise` schedule.

    `on_step(index, video, audio)` is called after every step with the model's current guess at the
    finished clip - the denoised estimate, in the same shapes as the result. That is what a preview
    shows; raising from it abandons the loop.
    """
    target = torch.device(device)
    generator = torch.Generator("cpu").manual_seed(seed)
    if first_frame_latent is None and last_frame_latent is None and not references and carry is None:
        raise ValueError("give a first or last frame latent (fl2va), references (ref2va), or both")
    if carry is not None and first_frame_latent is not None:
        raise ValueError("a carried clip opens on the carried steps, not on a first frame as well")

    keyframes = [latent for latent in (first_frame_latent, last_frame_latent) if latent is not None]
    if carry is not None:
        keyframes.insert(0, carry.video)
    if references or carry is not None:
        layout = build_reference_layout(
            conditioning.tags,
            geometry,
            references,
            first_frame=first_frame_latent is not None,
            last_frame=last_frame_latent is not None,
            carry=carry,
        )
        # Order matters: the layout puts the keyframes first, then each reference's audio before its
        # own video rows, so the latents have to be gathered in exactly that order.
        condition_latents = [
            *keyframes,
            *(reference.latent for reference in references if reference.latent is not None),
        ]
        audio_condition_latents = [
            *([carry.audio] if carry is not None else []),
            *(reference.audio_latent for reference in references if reference.audio_latent is not None),
        ]
    else:
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=conditioning.tags,
            num_latent_frames=geometry.latent_frames,
            latent_height=geometry.latent_height,
            latent_width=geometry.latent_width,
            num_audio_latents=geometry.audio_latents,
            patch_size=PATCH_SIZE,
            audio_channels=AUDIO_CHANNELS,
            audio_tag=AUDIO_TAG,
            video_tag=VIDEO_TAG,
            keyframe_anchors=tuple(
                anchor
                for anchor, latent in (("first", first_frame_latent), ("last", last_frame_latent))
                if latent is not None
            ),
        )
        condition_latents = keyframes
        audio_condition_latents = []
    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        condition_rows,
        audio_condition_rows,
    ) = layout
    sequence_length = position_ids.shape[0]
    if resident:
        blocks = transformer.transformer_blocks
        settled = offload.fit(blocks, sequence_length * config.SAMPLE_RESERVE_PER_ROW)
        ui.line(f"{settled}/{len(blocks)} DiT blocks resident for {sequence_length} rows")

    # Noise is drawn in latent shape and then packed, the way the reference does: packing is a
    # permutation, so drawing straight into rows would scatter the seed's values differently.
    video_noise = torch.randn(
        (1, transformer.config.in_channels, geometry.latent_frames, geometry.latent_height, geometry.latent_width),
        generator=generator,
        dtype=torch.float32,
    )
    audio_noise = torch.randn(
        (1, transformer.config.audio_in_channels, AUDIO_CHANNELS, geometry.audio_latents),
        generator=generator,
        dtype=torch.float32,
    )
    condition = condition_rows_from_latent(condition_latents, seed)
    latents = torch.cat([condition, patchify_video_latents(video_noise, PATCH_SIZE)], dim=0).to(target)
    # Audio conditioning is pinned at exactly 1.0, so it goes in clean: no noise to mix.
    audio_condition = [pack_audio(latent.float().cpu()) for latent in audio_condition_latents]
    audio_latents = torch.cat([*audio_condition, pack_audio(audio_noise)], dim=0).to(target)

    total = steps if init is None else max(steps, round(steps / denoise))
    skip = total - steps
    video_scheduler, audio_scheduler = make_schedulers(total, shift_video, shift_audio, schedule)
    if init is not None:
        video_sigma, audio_sigma = float(video_scheduler.sigmas[skip]), float(audio_scheduler.sigmas[skip])
        ui.line(f"resampling from sigma {video_sigma:.3f} (video), {audio_sigma:.3f} (audio)")
        start_video = patchify_video_latents(init[0].float().cpu(), PATCH_SIZE)
        latents[condition.shape[0] :] = (
            (1 - video_sigma) * start_video + video_sigma * latents[condition.shape[0] :].cpu()
        ).to(target)
        start_audio = pack_audio(init[1].float().cpu()).to(target)
        audio_latents[-start_audio.shape[0] :] = (1 - audio_sigma) * start_audio + audio_sigma * audio_latents[
            -start_audio.shape[0] :
        ]
        video_scheduler.set_begin_index(skip)
        audio_scheduler.set_begin_index(skip)
    if sampler == "er_sde":
        video_sde = ErSde(video_scheduler.sigmas[skip:], generator)
        audio_sde = ErSde(audio_scheduler.sigmas[skip:], generator)
    prompt_embeds = conditioning.embeds.to(target)
    layout_kwargs = {
        "token_tags": token_tags.to(target),
        "position_ids": position_ids.to(target),
        "video_indices": video_indices.to(target),
        "audio_indices": audio_indices.to(target),
        "text_indices": text_indices.to(target),
    }

    for index, (timestep, audio_timestep) in enumerate(
        zip(video_scheduler.timesteps[skip:], audio_scheduler.timesteps[skip:], strict=True)
    ):
        profiling.step()
        unique, row_indices = build_row_timesteps(
            sequence_length,
            video_indices,
            audio_indices,
            condition_rows,
            float(timestep),
            float(audio_timestep),
            audio_condition_rows,
        )
        with torch.inference_mode():
            noise_pred, audio_noise_pred = transformer(
                hidden_states=latents[None],
                audio_hidden_states=audio_latents[None],
                encoder_hidden_states=prompt_embeds,
                timestep=unique.to(target),
                timestep_indices=row_indices.to(target),
                return_dict=False,
                **layout_kwargs,
            )
            # The velocity points data-ward, so the clean estimate is x_t + (1 - t) * v (see the scheduler).
            video_x0 = latents[condition_rows:] + (1 - float(timestep)) * noise_pred[0, condition_rows:].float()
            audio_x0 = (
                audio_latents[audio_condition_rows:]
                + (1 - float(audio_timestep)) * audio_noise_pred[0, audio_condition_rows:].float()
            )
            if on_step is not None:
                estimate = (
                    unpatchify(video_x0.float().cpu(), geometry),
                    unpack_audio(audio_x0.float().cpu(), geometry),
                )
            if sampler == "er_sde":
                latents[condition_rows:] = video_sde.step(index, latents[condition_rows:].float(), video_x0)
                audio_latents[audio_condition_rows:] = audio_sde.step(
                    index, audio_latents[audio_condition_rows:].float(), audio_x0
                )
            else:
                latents[condition_rows:] = video_scheduler.step(
                    noise_pred[0, condition_rows:].float(), timestep, latents[condition_rows:], return_dict=False
                )[0]
                audio_latents[audio_condition_rows:] = audio_scheduler.step(
                    audio_noise_pred[0, audio_condition_rows:].float(),
                    audio_timestep,
                    audio_latents[audio_condition_rows:],
                    return_dict=False,
                )[0]
        if on_step is not None:
            on_step(index, *estimate)
    profiling.done()

    # Both streams carry their conditioning rows at the front; only the generated tail comes back.
    video = unpatchify(latents[condition_rows:].float().cpu(), geometry)
    audio = unpack_audio(audio_latents[audio_condition_rows:].float().cpu(), geometry)
    return video, audio
