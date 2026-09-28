"""Headless diffusers ports of two ComfyUI workflows: Qwen Image Edit, and MiniMax H3 image-to-video."""

import logging
import os

# Block streaming allocates and frees the same large shapes every step; without expandable segments
# the allocator strands gigabytes between them (3 GB reserved-but-unused at the 0.8 MP OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# difflab.ui is the only terminal output: silence library logs, warnings and progress bars.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
# In code rather than by HF_HUB_DISABLE_PROGRESS_BARS, which would also stop `core.hub` from showing a bar
# for the one thing worth watching: a checkpoint's first download.
from huggingface_hub.utils.tqdm import disable_progress_bars

disable_progress_bars()
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)  # torchao's enum registration chatter
