"""MiniMax H3's fixed facts: the frame grid, the audio rates, and what is loaded once per process.
Per-call knobs and the model files are in `settings.py`."""

from difflab.core.hub import Checkpoint, UrlFile

# The checkpoint ships no tokenizer files: MiniMax's own (Qwen2 vocab plus H3's special tokens) gives
# the same ids as ComfyUI's qwen25_tokenizer with those tokens registered.
TEXT_TOKENIZER = Checkpoint("MiniMaxAI/MiniMax-H3", "42ed227ee7df40d41602854ae760620d6eb651fe", ("tokenizer/*",))
# silu(time_embedder(t)) grid, needed to apply a full-width AdaLN LoRA to a pruned checkpoint. It ships
# with larryvrh's ComfyUI node on GitHub, not on the Hub.
SILU_TEMB_GRID = UrlFile(
    "https://raw.githubusercontent.com/larryvrh/ComfyUI-MiniMax-H3-Turbo/"
    "4274783a23afcfdbea3b4876cb79effd6c510785/h3_silu_temb_grid.safetensors",
    "30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8",
)

FPS = 24
# Frame counts must satisfy `n % 17 == 5`; the VAE turns 17k+5 frames into 5k+2 latent frames. Here
# rather than in `pipeline`, which cannot be imported without torch - `work.quick.Job.check` runs in
# the web process and has to know how long a chunk really is before anything is spawned.
FRAME_CYCLE, FRAME_PHASE = 17, 5
AUDIO_LATENT_FPS = 40  # audio latent rate; the audio VAE decodes to 32 kHz stereo
AUDIO_SAMPLE_RATE = 32000

# --- Memory -------------------------------------------------------------------------------------
# The packed sequence 24 GB holds, from three measurements, and they are closer together than is
# comfortable - this is a warning threshold, not a wall, and what actually fits depends on the shape
# of the sequence as well as its length:
#
#   15 s chunk at 0.98 MP, no reference clip      ~115k rows   did not fit
#   12 s chunk at 0.98 MP, no reference clip       ~94k rows   fits (the cat take: 12m 28s sampled)
#   12 s chunk at 0.98 MP + 12 s reference at 0.5  113.5k rows fits, 17.3 GB peak, 178 s/step
#
# The third is the video-editing shape and it is why this moved off 105,000: warning about a
# configuration that has now rendered twice is noise. Deliberately not DIRECTOR_-prefixed: that
# prefix is a fingerprint.
MAX_ROWS = 114_000

# Both fixed when a model is built, and a built model is shared by every call after it.
MATMUL_CHUNK_TOKENS = 8192  # token chunk for the int8 linears and the SwiGLU intermediate
VAE_TILE_BATCH = 24  # 256x256 decoder tiles run at once; halved automatically if one group cannot fit
