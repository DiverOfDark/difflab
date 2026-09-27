"""MiniMax H3 image-to-video (+ native audio), ported from the ComfyUI workflow.

Independent of the image-edit pipeline in `difflab/`, but reuses its UI, image helpers and VLM.
"""

import difflab  # noqa: F401  (sets HF_HUB_OFFLINE and silences library logs before any HF import)
