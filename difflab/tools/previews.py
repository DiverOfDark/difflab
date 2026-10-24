"""Previews while a clip samples: each step's guess at the finished take, as a small video with sound.

The video half goes through a tiny decoder on the card at half size - taeh3 for H3, taeltx2_3_wide for
LTX-2.5 - seconds, between two steps that each take longer. The audio half goes through the model's own
audio VAE on the CPU, in a background thread, so sampling never waits for it.

A preview is a courtesy. Anything that goes wrong in one - a missing decoder, a tight card - is
reported once and switches previews off for the rest of the process; it never costs the render.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from PIL import Image

from difflab import config
from difflab.core import context, events, models, protocol, ui, video_io
from difflab.core.hub import HubFile, parse_file
from difflab.models import tae, taeltx
from difflab.models.ltx25 import config as ltx_config
from difflab.models.ltx25 import pipeline as ltx_pipeline
from difflab.models.ltx25 import weights as ltx_weights
from difflab.models.ltx25.settings import LtxSettings
from difflab.models.minimax_h3 import config as h3_config
from difflab.models.minimax_h3 import vae
from difflab.models.minimax_h3.settings import VideoSettings


def tae_spec(file: HubFile) -> models.ModelSpec:
    return models.ModelSpec(
        name=f"taeh3:{file.filename}",
        build=lambda: tae.load(file.fetch(), config.DEVICE),
        checkpoints=(file.path,),
        heavy=False,  # 9 MB
    )


def taeltx_spec() -> models.ModelSpec:
    return models.ModelSpec(
        name="taeltx2_3_wide",
        build=lambda: taeltx.load(taeltx.WEIGHTS.fetch(), config.DEVICE),
        checkpoints=(taeltx.WEIGHTS.path,),
        heavy=False,  # 58 MB
    )


LATENT_FPS = h3_config.FPS // 4  # taeh3 draws one picture per latent frame, and a latent frame is 4 frames

# Our own CPU copies, by checkpoint: the registry's are parked and must stay untouched.
_audio_vae: tuple[str, torch.nn.Module] | None = None
_ltx_audio: tuple[str, torch.nn.Module, torch.nn.Module] | None = None
_audio_lock = threading.Lock()


def for_sampling(settings: VideoSettings) -> Previewer | None:
    """An H3 previewer for the loop about to run, or `None` when previews are off or cannot work."""
    if not _wanted(settings.previews):
        return None
    try:
        decoder = HubFile.parse(settings.preview_decoder)
        decoder.fetch()
    except Exception as e:  # previews are optional: a failed download must not stop the render
        _give_up(f"no preview decoder ({settings.preview_decoder}): {e}")
        return None
    ctx = context.current()
    return Previewer(
        settings.steps,
        ctx.preview_item,
        ctx.preview_name,
        lambda video: tae.decode(models.registry.acquire(tae_spec(decoder)), video, scale=settings.preview_scale),
        lambda audio: _decode_audio(audio, settings.audio_vae),
        LATENT_FPS,
        h3_config.AUDIO_SAMPLE_RATE,
    )


def for_ltx(settings: LtxSettings) -> Previewer | None:
    """An LTX previewer, or `None` when previews are off or cannot work."""
    if not _wanted(settings.previews):
        return None
    try:
        taeltx.WEIGHTS.fetch()
    except Exception as e:  # previews are optional: a failed download must not stop the render
        _give_up(f"no preview decoder (taeltx2_3_wide): {e}")
        return None
    ctx = context.current()
    return Previewer(
        len(settings.schedule()) - 1,
        ctx.preview_item,
        ctx.preview_name,
        lambda video: taeltx.decode(models.registry.acquire(taeltx_spec()), video, scale=settings.preview_scale),
        lambda audio: _decode_ltx_audio(audio, settings.audio_vae_source()),
        ltx_config.FPS,
        ltx_config.AUDIO_SAMPLE_RATE,
    )


def _wanted(enabled: bool) -> bool:
    return enabled and not context.current().previews_broken and events.active() is not None


def target(item: int | None, name: str) -> None:
    """Which clip the next previews belong to. A long video sets it per chunk; a single clip leaves it."""
    ctx = context.current()
    ctx.preview_item, ctx.preview_name = item, name


class Previewer:
    def __init__(
        self,
        steps: int,
        item: int | None,
        name: str,
        picture: Callable[[torch.Tensor], torch.Tensor],
        sound: Callable[[torch.Tensor], torch.Tensor | None],
        fps: int,
        sample_rate: int,
    ) -> None:
        self.steps, self.item, self.name = steps, item, name
        self.picture, self.sound, self.fps, self.sample_rate = picture, sound, fps, sample_rate
        self._ctx = context.current()  # a pool thread has no context of its own, and an unbound one would open a run
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview")
        self._pending: list[Future[None]] = []

    def __call__(self, index: int, video: torch.Tensor, audio: torch.Tensor) -> None:
        if context.current().previews_broken:
            return
        try:
            frames = self.picture(video)
        except Exception as error:
            _give_up(f"preview decode failed ({type(error).__name__}: {error})")
            return
        self._pending.append(self._pool.submit(self._finish, index, frames, audio))

    def close(self) -> None:
        """Wait for the last previews to be written - the audio VAE copy must be idle before we go on."""
        self._pool.shutdown(wait=True)

    def _finish(self, index: int, frames: torch.Tensor, audio: torch.Tensor) -> None:
        with context.bind(self._ctx):
            self._write(index, frames, audio)

    def _write(self, index: int, frames: torch.Tensor, audio: torch.Tensor) -> None:
        try:
            run = events.current()
            stem = f"previews/{self.name}_step{index + 1}"
            waveform = self.sound(audio)
            path = run.path(f"{stem}.mp4")
            writer = video_io.VideoWriter(
                path, int(frames.shape[2]), int(frames.shape[1]), self.fps, self.sample_rate, crf=28
            )
            writer.write(frames)
            writer.close(waveform)
            poster = run.path(f"{stem}.jpg")
            Image.fromarray(frames[len(frames) // 2].numpy()).save(poster, quality=85)
            events.emit(
                protocol.Preview(
                    item=self.item,
                    name=self.name,
                    step=index + 1,
                    steps=self.steps,
                    video=run.rel(path),
                    poster=run.rel(poster),
                )
            )
        except Exception as error:
            _give_up(f"preview write failed ({type(error).__name__}: {error})")


def _decode_audio(latents: torch.Tensor, checkpoint: str) -> torch.Tensor | None:
    """The estimate's soundtrack, on the CPU: 15 s decodes in about 17 s there, beside a 2-minute step."""
    global _audio_vae
    with _audio_lock:
        if _audio_vae is None or _audio_vae[0] != checkpoint:
            _audio_vae = checkpoint, vae.load_audio_vae(HubFile.parse(checkpoint).fetch()).float()
        return vae.decode_audio(_audio_vae[1], latents)


def _decode_ltx_audio(latents: torch.Tensor, checkpoint: str) -> torch.Tensor:
    """The estimate's soundtrack through LTX's audio VAE and vocoder, on the CPU."""
    global _ltx_audio
    with _audio_lock:
        if _ltx_audio is None or _ltx_audio[0] != checkpoint:
            audio_vae, vocoder = ltx_weights.load_audio_vae(parse_file(checkpoint).fetch())
            _ltx_audio = checkpoint, audio_vae.float(), vocoder.float()
        return ltx_pipeline.decode_audio(_ltx_audio[1], _ltx_audio[2], latents)


def _give_up(reason: str) -> None:
    ctx = context.current()
    if not ctx.previews_broken:
        ctx.previews_broken = True
        ui.note(f"previews off: {reason}")
