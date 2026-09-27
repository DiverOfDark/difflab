"""Previews while a clip samples: each step's guess at the finished take, as a small video with sound.

The video half goes through taeh3, the 9 MB preview decoder, on the card and at half size - seconds,
between two steps that each take minutes. The audio half goes through the audio VAE on the CPU, in a
background thread, so sampling never waits for it.

A preview is a courtesy. Anything that goes wrong in one - a missing decoder, a tight card - is
reported once and switches previews off for the rest of the process; it never costs the render.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from PIL import Image

from difflab import config
from difflab.core import context, events, models, protocol, ui, video_io
from difflab.core.hub import HubFile
from difflab.models import tae
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


LATENT_FPS = h3_config.FPS // 4  # taeh3 draws one picture per latent frame, and a latent frame is 4 frames

# Our own CPU copy, by checkpoint: the registry's is parked and must stay untouched.
_audio_vae: tuple[str, torch.nn.Module] | None = None
_audio_lock = threading.Lock()


def for_sampling(settings: VideoSettings) -> Previewer | None:
    """A previewer for the loop about to run, or `None` when previews are off or cannot work."""
    ctx = context.current()
    if ctx.previews_broken or not settings.previews or events.active() is None:
        return None
    try:
        decoder = HubFile.parse(settings.preview_decoder)
        decoder.fetch()
    except Exception as e:  # previews are optional: a failed download must not stop the render
        _give_up(f"no preview decoder ({settings.preview_decoder}): {e}")
        return None
    return Previewer(settings, decoder, ctx.preview_item, ctx.preview_name)


def target(item: int | None, name: str) -> None:
    """Which clip the next previews belong to. A long video sets it per chunk; a single clip leaves it."""
    ctx = context.current()
    ctx.preview_item, ctx.preview_name = item, name


class Previewer:
    def __init__(self, settings: VideoSettings, decoder: HubFile, item: int | None, name: str) -> None:
        self.steps, self.item, self.name = settings.steps, item, name
        self.settings, self.decoder = settings, decoder
        self._ctx = context.current()  # a pool thread has no context of its own, and an unbound one would open a run
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview")
        self._pending: list[Future[None]] = []

    def __call__(self, index: int, video: torch.Tensor, audio: torch.Tensor) -> None:
        if context.current().previews_broken:
            return
        try:
            decoder = models.registry.acquire(tae_spec(self.decoder))
            frames = tae.decode(decoder, video, scale=self.settings.preview_scale)
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
            waveform = _decode_audio(audio, self.settings.audio_vae)
            path = run.path(f"{stem}.mp4")
            writer = video_io.VideoWriter(path, int(frames.shape[2]), int(frames.shape[1]), LATENT_FPS, crf=28)
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


def _give_up(reason: str) -> None:
    ctx = context.current()
    if not ctx.previews_broken:
        ctx.previews_broken = True
        ui.note(f"previews off: {reason}")
