"""LTX-2.5's fixed facts. Per-call knobs and the model files are in `settings.py`."""

from difflab.core.hub import Checkpoint

# The split pack ships weights only. unsloth's diffusers-layout repack of the same release is ungated and
# carries every component's config and the Gemma tokenizer; its weights are never fetched.
LAYOUT = Checkpoint(
    "unsloth/LTX-2.5-FP8",
    "7b40724fc39bbb01d9b4d72638c0d263531c88c2",
    ("*/config.json", "tokenizer/*"),
)

ENGINE = "ltx-2.5"  # what a quick job or a production names this model by
FPS = 24
# Frame counts are 8k+1; the VAE turns them into k+1 latent frames, the first a single pixel frame.
FRAME_CYCLE = 8
SPATIAL_COMPRESSION = 32
LATENT_CHANNELS = 128
AUDIO_LATENTS_PER_SECOND = 25  # 16 kHz mel, hop 160, 4x in the audio VAE
AUDIO_SAMPLE_RATE = 48000  # the vocoder's bandwidth-extended output

# The distilled checkpoint's fixed schedule, from ltx-pipelines' constants; CFG 1.
SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
# H.264 CRF the opening frame is re-compressed at before encoding: the compression 2.5 was trained against
# (33 through 2.3).
IMAGE_CRF = 18
# Gemma is padded to this, and the connectors' 128 learnable registers have to divide it.
PROMPT_TOKENS = 1024

# Licon MSR: each reference is one still repeated over 33 frames (25 is the other trained length), five
# latent frames, and the slots sit at pixel frames -N..-1 in reference order.
MSR_FRAMES = 33
MSR_MAX_REFERENCES = 5

# What sampling keeps free beside the DiT blocks it holds resident. Measured 2026-10-24: a 4.60 GB peak over
# the held weights at 13,260 rows (960x544, 121 frames, two references), with 32 of 48 blocks resident.
SAMPLE_RESERVE = 1024**3
SAMPLE_RESERVE_PER_ROW = 330 * 1024
