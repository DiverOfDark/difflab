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
# The same for Qwen-Image 2.1. A 2 MP edit with a donor held 8.4 GB before the call (26 of 32 blocks resident
# at 10.5 GiB) and still ran out; streaming every block it peaked at 18.5 GB beside a 3.4 GB desktop
# (2026-10-25). Taken from that one edit, not a sweep: the all-streaming retry covers what it misses.
EDIT21_RESERVE_BYTES = 12 * 1024**3

FINISH_FPS = 60  # what RIFE retimes a finished chapter to; H3 draws 24
FINISH_MEGAPIXELS = 3840 * 2160 / 1e6  # what the ESRGAN enlarges it to


# The VLM (`models/exl3vl.py`). 16 full-attention layers of 4 KV heads at 256 cost 64 KiB a token, so
# this is 2 GiB on the card. Measured 2026-10-03: a plan repair peaks near 11k tokens on a 60 s take of
# five clips, and its answer grows ~640 tokens a clip, so 16k ran out at about nine clips (~100 s).
# 64k (4 GiB) is for the clip judge's frame every half second (`judge.SAMPLED_EVERY`). Unquantized: a
# quantized cache would move every ruling measured against this one.
VLM_CONTEXT_TOKENS = 65536
# The chat's own load (`tools/chat.py`), which rulings never use. At 8 bits a token is 34 KiB of cache and
# 2.1 KiB of draft cache, and ExLlamaV3 stages the window as fp16 for a long prefill, 4 KiB a token more
# (`Attention.autosplit_extra_measure`): 432 pages fill the 4.25 GiB the 64k fp16 cache and its draft do.
CHAT_KV_BITS = 8
CHAT_CONTEXT_TOKENS = 432 * 256
# MTP proposals per step. Measured on the 3090 (tok/s, sampled at 0.7/0.8/20; a card call with one
# photo, a survey with two, 2k tokens of prose): none 36 / 36 / 36, 2 drafts 71 / 75 / 67, 3 drafts
# 80 / 88 / 74, 4 drafts 85 / 97 / 70, 5 drafts 96 / 114 / 68. JSON, which is all the director
# asks for, keeps accepting further out; prose stops gaining at 3. Each draft is another
# linear-attention state held per slot, ~150 MB.
VLM_DRAFT_TOKENS = 4
# The template's default is xhigh. Chat and free questions think at low; the director's JSON and every
# ruling do not think at all (`direct.ask`), and the writers of prose a model draws from think at medium.
# A thought that reaches VLM_THINK_TOKENS is closed there: asked for 60-80 words, it counted its draft
# word by word past 4,096 tokens and never answered (an opening frame, 2026-10-11).
VLM_REASONING_EFFORT = "low"
VLM_WRITING_EFFORT = "medium"
VLM_THINK_TOKENS = 4096
VLM_RECURRENT_CACHE_BYTES = 1024**3  # host RAM, for reusing a prefix's linear-attention state
# Host RAM for images already embedded. A picture embedded afresh gets new placeholder token ids, so no
# prompt holding it ever matches the paged cache past it; a repair re-read the whole plan or clip. A
# 1280x736 frame is 920 tokens, ~9.4 MB: this holds one judged clip of 26 frames twice over.
VLM_EMBEDDING_CACHE_BYTES = 512 * 1024**2
# On NVMe, not RAM: only the openings prompts share (`Exl3VL.ask`'s system and steady images), each a
# ~150 MB linear-attention checkpoint plus 16 MB of KV a page. Read through the page cache, dropped after.
VLM_DISK_CACHE_DIR = OUTPUT_DIR / "kvcache"
VLM_DISK_CACHE_BYTES = 40 * 1024**3

# What a chat hands the VLM of its attachments. A photo at 1 MP is ~1k vision tokens; a video is a frame
# every CHAT_VIDEO_EVERY seconds, at most CHAT_VIDEO_FRAMES of them at ~400 tokens each.
CHAT_PHOTO_PIXELS = 1024 * 1024
CHAT_FRAME_PIXELS = 400_000
CHAT_VIDEO_EVERY = 1.0
CHAT_VIDEO_FRAMES = 24

# ComfyUI Image Saver Simple.
OUT_DIR = OUTPUT_DIR / "images"
VIDEO_DIR = OUTPUT_DIR / "videos"

DASHBOARD = True  # serve it whenever something runs; runs are recorded under RUNS_DIR either way
# These two read the environment, because a container is configured by its environment and nothing else -
# but the defaults are the same values, so pressing Run changes nothing.
DASHBOARD_HOST = os.environ.get("DIFFLAB_HOST", "0.0.0.0")  # LAN-reachable; "127.0.0.1" for this machine only
DASHBOARD_PORT = int(os.environ.get("DIFFLAB_PORT", "8765"))
RUNS_DIR = OUTPUT_DIR / "runs"
# Every request needs it once set, as a bearer header or the cookie `/auth` hands out: a pod's proxy URL
# is public.
TOKEN = os.environ.get("DIFFLAB_TOKEN", "")

# `work/watchdog.py`. RunPod sets RUNPOD_* on every pod; the desktop app sets the rest on the pod it
# creates. Off RunPod POD_ID is empty and nothing watches. The app passes the user's own key as well:
# RunPod does not document whether the pod-scoped key it injects may delete its own pod.
POD_ID = os.environ.get("RUNPOD_POD_ID", "")
POD_API_KEY = os.environ.get("DIFFLAB_RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY", "")
POD_IDLE_MINUTES = float(os.environ.get("DIFFLAB_IDLE_MINUTES", "5"))
POD_STUCK_MINUTES = float(os.environ.get("DIFFLAB_STUCK_MINUTES", "15"))
POD_USD_PER_HOUR = float(os.environ.get("DIFFLAB_USD_PER_HOUR", "0"))  # 0: unknown, so no spend limit
POD_SPEND_LIMIT_USD = float(os.environ.get("DIFFLAB_SPEND_LIMIT_USD", "5"))
POD_EXIT_WHEN_DONE = os.environ.get("DIFFLAB_EXIT_WHEN_DONE", "") == "1"
POD_ENDED = OUTPUT_DIR / "pod-ended.json"  # on the network volume, where the app reads why the last pod went
RUNS_KEEP = 10  # newest run directories kept whatever their age; final images and videos are kept
# `work/cleanup.py`, run by the worker between jobs. Blobs on 2026-10-11 were 22 G, 12 G of it handoff
# tails at ~37 MB a shot, and 28 G on 2026-10-23, all of it under a week old; a cache hit, or a job naming
# a blob, refreshes its mtime, so the age is since last use and the budget evicts least recently used.
CLEANUP_EVERY_HOURS = 6
BLOB_KEEP_DAYS = 7
BLOB_KEEP_GB = float(os.environ.get("DIFFLAB_BLOB_KEEP_GB", "15"))
BLOB_KEEP_TOOLS = ("given", "director", "ruling")  # uploads, and VLM answers: small and slow to ask again
RUNS_KEEP_DAYS = 14
PRODUCTIONS_KEEP = 15  # newest by last edit or run, whatever their age
PRODUCTIONS_IDLE_DAYS = 7
BACKUPS_KEEP = 5
