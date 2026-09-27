"""Paths, the machine, and what is loaded once per process; per-call knobs are in `core/settings.py`."""

import os
from pathlib import Path

from difflab.core.hub import Checkpoint

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

# Qwen2.5-VL tokenizer for the editor's text encoder (not stored in the AIO file).
TOKENIZER = Checkpoint("Qwen/Qwen-Image", "75e0b4be04f60ec59a75f475837eced720f823b6", ("tokenizer/*",))

DEVICE = "cuda"

# Both video models compress space by 32, so a side that is not a multiple of it cannot be encoded; a
# model that ever wants coarser rounds again in its own pipeline.
CANVAS_ALIGN = 32

INPUT_IMAGE = PROJECT_DIR / "samples/input.jpg"
INPUT_VIDEO = PROJECT_DIR / "samples/input.mp4"

# The VLM is 52 GB of bf16 on disk and 15.1 GiB of NF4 in VRAM, and `from_pretrained` quantizes it on
# the way in - so every director pass reads 52 GB to produce 15, at about 1.5 GB/s, and none of it can
# stay in the page cache because the box has 58 GB and a render wants most of them. Saving the
# quantized model once and loading that instead reads a third as much and quantizes nothing.
#
# It is a cache, not a conversion step: written automatically the first time, keyed by the snapshot's
# own commit so a new pin makes a new one, and deletable at any time for the cost of one slow load.
VLM_4BIT_CACHE = True
VLM_4BIT_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "difflab" / "qwenvl-nf4"

# CUDA graphs, measured and turned down. Decoding looks launch-bound - 54 ms a token
# against the ~16 ms the 15.1 GiB of weights each token reads would take at the card's bandwidth - and
# CUDA graphs do capture that: a bench on the 3090 measured 18.4 tok/s eager against
# 22.5 tok/s compiled, with zero graph breaks. bitsandbytes traces cleanly; that was never the problem.
#
# The problem is the static cache a captured graph requires. It is read whole on every step
# (`StaticLayer.get_mask_sizes` returns `max_cache_len`, not the length in use) and costs a measured
# 2.56 us per allocated token - dead linear from 2k to 24k, and about 25 GB/s effective, so it is the
# GQA expansion and a materialized mask dropping SDPA onto its math backend rather than bandwidth:
#
#     dynamic, eager   18.4 tok/s      static 8k, compiled   16.6 tok/s
#     static 2k, eager 17.9 tok/s      static 24k, compiled   9.8 tok/s
#     static 2k, compiled 22.5 tok/s
#
# The two cross at about 5,900 allocated tokens and the director lives above that: one card call is a
# thousand image tokens plus its system prompt plus 2,048 new, and a plan repair carries an 8k answer
# back into its question. Sized for the worst call, compiling halves the director's speed; sized at
# 4,096 it gains 9%; the 2,048 that gains 22% does not hold a single real call. So: off. Turn it on
# only alongside a cache that reads its used length, or a stage that is known to stay short.
VLM_COMPILE = False
VLM_CACHE_TOKENS = 4096  # only read when VLM_COMPILE is on; the largest size that still gains

# ComfyUI Image Saver Simple.
OUT_DIR = OUTPUT_DIR / "images"
VIDEO_DIR = OUTPUT_DIR / "videos"

DASHBOARD = True  # serve it whenever something runs; runs are recorded under RUNS_DIR either way
# These two read the environment, because a container is configured by its environment and nothing else -
# but the defaults are the same values, so pressing Run changes nothing.
DASHBOARD_HOST = os.environ.get("DIFFLAB_HOST", "0.0.0.0")  # LAN-reachable; "127.0.0.1" for this machine only
DASHBOARD_PORT = int(os.environ.get("DIFFLAB_PORT", "8765"))
# Whether a long video keeps its process up so a weak chunk can be re-rolled from the browser. Off,
# because the wait has no end: it holds the card until someone presses "finish". Re-rolling afterwards
# means rendering again, which the chunk cache makes cheap - only the changed chunks are sampled.
DASHBOARD_LINGER = False
RUNS_DIR = OUTPUT_DIR / "runs"
RUNS_KEEP = 10  # older run directories are deleted at startup; final images and videos are kept
