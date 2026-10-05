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


# The VLM (`models/exl3vl.py`). 16 full-attention layers of 4 KV heads at 256 cost 64 KiB a token, so
# this is 2 GiB on the card. Measured 2026-10-03: a plan repair peaks near 11k tokens on a 60 s take of
# five clips, and its answer grows ~640 tokens a clip, so 16k ran out at about nine clips (~100 s).
VLM_CONTEXT_TOKENS = 32768
# MTP proposals per step. Measured on the 3090 (tok/s, sampled at 0.7/0.8/20; a card call with one
# photo, a survey with two, 2k tokens of prose): none 36 / 36 / 36, 2 drafts 71 / 75 / 67, 3 drafts
# 80 / 88 / 74, 4 drafts 85 / 97 / 70, 5 drafts 96 / 114 / 68. JSON, which is all the director
# asks for, keeps accepting further out; prose stops gaining at 3. Each draft is another
# linear-attention state held per slot, ~150 MB.
VLM_DRAFT_TOKENS = 4
VLM_RECURRENT_CACHE_BYTES = 1024**3  # host RAM, for reusing a prefix's linear-attention state

# ComfyUI Image Saver Simple.
OUT_DIR = OUTPUT_DIR / "images"
VIDEO_DIR = OUTPUT_DIR / "videos"

DASHBOARD = True  # serve it whenever something runs; runs are recorded under RUNS_DIR either way
# These two read the environment, because a container is configured by its environment and nothing else -
# but the defaults are the same values, so pressing Run changes nothing.
DASHBOARD_HOST = os.environ.get("DIFFLAB_HOST", "0.0.0.0")  # LAN-reachable; "127.0.0.1" for this machine only
DASHBOARD_PORT = int(os.environ.get("DIFFLAB_PORT", "8765"))
RUNS_DIR = OUTPUT_DIR / "runs"
RUNS_KEEP = 10  # older run directories are deleted at startup; final images and videos are kept
