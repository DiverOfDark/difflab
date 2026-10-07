"""SeedVR2 7B, the one-step diffusion video restorer, for enlarging a finished chapter.

Ported from ByteDance's SeedVR (Apache-2.0) as carried by numz/ComfyUI-SeedVR2_VideoUpscaler
(Apache-2.0), cut to one sample, one step and the 7B; ComfyUI's own port is GPL-3.0 and was not used.
The weights are Comfy-Org's int8 ConvRot repack, which also carries the positive and negative text
embeddings, so no text encoder is loaded.
"""
