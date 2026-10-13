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

# What an edit keeps free beside the transformer blocks it holds resident: the highest card use seen
# across 25 edits run with every block streaming (2026-10-10/11), 7.41 GB in the VAE decode against
# 4.4-6.3 GB while denoising. CUDA context and VAE weights included, so it errs high by those.
EDIT_RESERVE_BYTES = int(7.5 * 1024**3)

FINISH_FPS = 60  # what RIFE retimes a finished chapter to; H3 draws 24
FINISH_MEGAPIXELS = 3840 * 2160 / 1e6  # what the ESRGAN enlarges it to


# The VLM (`models/exl3vl.py`). 16 full-attention layers of 4 KV heads at 256 cost 64 KiB a token, so
# this is 2 GiB on the card. Measured 2026-10-03: a plan repair peaks near 11k tokens on a 60 s take of
# five clips, and its answer grows ~640 tokens a clip, so 16k ran out at about nine clips (~100 s).
# 64k (4 GiB) is for the clip judge's frame every half second (`judge.SAMPLED_EVERY`). Unquantized: a
# quantized cache would move every ruling measured against this one.
VLM_CONTEXT_TOKENS = 65536
# MTP proposals per step. Measured on the 3090 (tok/s, sampled at 0.7/0.8/20; a card call with one
# photo, a survey with two, 2k tokens of prose): none 36 / 36 / 36, 2 drafts 71 / 75 / 67, 3 drafts
# 80 / 88 / 74, 4 drafts 85 / 97 / 70, 5 drafts 96 / 114 / 68. JSON, which is all the director
# asks for, keeps accepting further out; prose stops gaining at 3. Each draft is another
# linear-attention state held per slot, ~150 MB.
VLM_DRAFT_TOKENS = 4
# The template's default is xhigh; low keeps the many one-question rulings cheap. A thought that reaches
# VLM_THINK_TOKENS is closed there: asked for 60-80 words, it counted its draft word by word past 4,096
# tokens and never answered (production 16's opening, 2026-10-11).
VLM_REASONING_EFFORT = "low"
VLM_THINK_TOKENS = 4096
VLM_RECURRENT_CACHE_BYTES = 1024**3  # host RAM, for reusing a prefix's linear-attention state
# Host RAM for images already embedded. A picture embedded afresh gets new placeholder token ids, so no
# prompt holding it ever matches the paged cache past it; a repair re-read the whole plan or clip. A
# 1280x736 frame is 920 tokens, ~9.4 MB: this holds one judged clip of 26 frames twice over.
VLM_EMBEDDING_CACHE_BYTES = 512 * 1024**2

# ComfyUI Image Saver Simple.
OUT_DIR = OUTPUT_DIR / "images"
VIDEO_DIR = OUTPUT_DIR / "videos"

DASHBOARD = True  # serve it whenever something runs; runs are recorded under RUNS_DIR either way
# These two read the environment, because a container is configured by its environment and nothing else -
# but the defaults are the same values, so pressing Run changes nothing.
DASHBOARD_HOST = os.environ.get("DIFFLAB_HOST", "0.0.0.0")  # LAN-reachable; "127.0.0.1" for this machine only
DASHBOARD_PORT = int(os.environ.get("DIFFLAB_PORT", "8765"))
RUNS_DIR = OUTPUT_DIR / "runs"
RUNS_KEEP = 10  # newest run directories kept whatever their age; final images and videos are kept
# `work/cleanup.py`, run by the worker between jobs. Blobs on 2026-10-11 were 22 G, 12 G of it handoff
# tails at ~37 MB a shot; a cache hit refreshes a blob's mtime, so the age is since last use.
CLEANUP_EVERY_HOURS = 6
BLOB_KEEP_DAYS = 7
BLOB_KEEP_TOOLS = ("given", "director", "ruling")  # uploads, and VLM answers: small and slow to ask again
RUNS_KEEP_DAYS = 14
PRODUCTIONS_KEEP = 15  # newest by last edit or run, whatever their age
PRODUCTIONS_IDLE_DAYS = 7
BACKUPS_KEEP = 5
