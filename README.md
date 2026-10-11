# difflab

A video factory on one card, built on diffusers and transformers: an idea and a few photographs go
in, a chapter of shots comes out as one continuous video with its own sound. Three models do all of
it - Qwen-Image-Edit for pictures, MiniMax H3 for video, one Qwen3.8 VLM for every word - and there
is no ComfyUI and no custom node.

What it makes:

1. **Quick renders** - a photo restyled, one clip from a photo and words, a finished mp4 enlarged.
2. **Productions** - from an idea and photos, from a text split into chapters, or from a video to
   redo shot by shot. A production is a cast, chapters of shots, and for every shot a keyframe, a
   prompt and a clip, each made by a job and remade only when something it was made of changes.

## Layout

```
difflab/
  core/        types (Frame, Clip) · settings kit · keys · cache · context · events · protocol · record · ui · hub · media
  models/      exl3vl · qwen_edit_aio · convrot · offload · video_io · esrgan · rife · tae
               seedvr2/ (SeedVR2 7B: dit, vae, color, pipeline)
               minimax_h3/ (the port, its settings, and caption: the Ref2VA format)
  tools/       edit_image · ask_vlm · write_prompt · render_chunk · handoff · assemble · restore
               enlarge_pixels · retime · grade · judge · compose_keyframe
               direct · survey · ground · cut_crops · cards · opening · plan_shots · write_shot   (the director)
  screenplay/  production · ingest · planner · made · settings · compiler · checks · crops · words · captions
  work/        quick (the three quick modes as data) · queue · runners · plan_runners · worker
  store/       db (migrations) · models · repo (runs, jobs, tasks, artifacts, events, commands) · screenplay
  api/         app · serve · runs · assets            the HTTP interface, and the worker thread beside it
frontend/      the page (Vue 3 + Vite), built into difflab/api/static
```

A tool takes and returns `core.types`, so anything composes:

```python
frame = media.frame_from_image("photo.jpg")
restyled = edit_image.edit(frame, "make it golden hour")  # Qwen Image Edit
prompt = write_prompt.write_video_prompt(restyled, "she turns and smiles")  # Qwen-VL
clip = render_chunk.render(restyled, prompt)  # MiniMax H3
```

The dependency rule is one line: `core/` imports no model, `models/` imports no tool, `tools/` import
no screenplay, and `store/` and `api/` import no model package, so the server starts without torch
and the worker thread beside it is what holds the weights.

Five things make that more than a call chain:

* **`core/models` - one registry** decides which heavy model is resident, instead of each pipeline
  deciding for itself. It evicts on a budget and drops the evicted checkpoint's page cache, which
  matters because the text encoder maps 25 GB and the DiT 18 GB against 58 GB of RAM.
* **`core/settings` - parameters are values**, not module globals. Every field says what it is - a
  label, a hint, a range, whether it joins a key - and resolves in three layers: the code default,
  this machine's overrides, one production's. A job records the values it ran with.
* **`core/keys` - a made thing is named by what it was made of**: the tool, the tool's key version,
  and its inputs by name, pictures by their pixels. Two calls with equal parts share one artifact, so
  a run that dies at shot 7 resumes at shot 7, a re-roll of shot 2 re-renders shots 2 and 3 and
  nothing else, and rendering a finished chapter again samples nothing.
* **`store/` - a run is rows.** A run is what was asked for; a job is one tool applied once with its
  inputs, outputs and key; a task is a phase of that tool and the only thing with a bar. Events and
  commands are rows too, so a page can join a render halfway through or replay a finished one.
* **`screenplay/` - what you intend is separate from what ran.** Productions, chapters, shots and the
  made things a shot owns share one lifecycle - missing, queued, running, made, approved, stale,
  pinned - and a run is built from whatever in its scope is missing or stale.

### A chapter is a chain

H3 renders about 15 s at a time, so a chapter is shots, each opening on the one before. Three things
cross every seam, and they fix different failures (`core.settings.Continuity` has their prices): the
**handoff frame** - the sharpest of the previous shot's last few, not its final one, which is often
mid-blur; the **tail** - the frames running into it, since a single frame carries pose but no
velocity and without it the subject visibly restarts at every join; and the **identity references**,
the same crops in every shot, the only one of the three that fights drift. `tools/handoff` takes the
first two right after a shot decodes, `tools/assemble` joins the chunks at the handoff frames, and
`screenplay/planner` writes the jobs in that order.

A shot whose keyframe is composed or given is not a chain link but a picture of its own, and by
default the shot before **lands** on it: H3's `last_image` keyframe pins that shot's final frame to
the picture, its caption says so as `<Picture 2>`, and its tail crosses into the new shot as a
handoff's would. The planner composes every keyframe before any clip for it. Turned off on the
Keyframes screen, the join is a cut: the shot before ends where the model takes it, no tail crosses.

### The director

The VLM is never asked to write a whole caption: a local model gets the prose right and the
bookkeeping wrong. It is asked one question at a time - survey the photos, box the cast in each,
read their traits, describe the opening frame, tidy the cards, write the opening sentence, plan the
chapter, write each shot's prose - and every answer passes a gate written in code, which sends its
own messages back as the repair instructions. The compiler (`screenplay/compiler`) then writes
MiniMax's six-section Ref2VA caption around the prose: which picture each `<Subject N>` comes from,
the retention lines, the style sentence, the `<Video 1>`/`<Audio 1>` bookkeeping. Every question's
answer is kept by its exact text, so a plan run asked again with nothing changed asks the model
nothing.

Before any of that, the same model can rewrite a rough idea into a story the planner renders well
(`tools/refine_idea`, one job per round, the Ingest stage's "Improve with director"). Its rules in
`words.IDEA_SYSTEM` are each a failure of the planner on a rough idea: the place taken from the
first photo, a roasted pigeon alive again in the next shot, every animal ending turned to the
camera, "fires flame" rendered as a stream. A round takes 25-50 s; a round with feedback revises
the version it was given rather than starting over, and the user adopts one as the story.

### What is in a chunk key

A chunk is cached by its opening frame's pixels, its caption's exact text, its `VideoSettings`, its
reference pixels, its motion tail, any source excerpt and the frame it lands on, which is keyed only
when there is one. Shot k+1 opens on shot k's handoff frame,
so one changed byte in a caption re-renders that shot and every shot chained after it. Nothing
checks that automatically; when editing `chunk_key`, watch the cache.

## Image editing

A headless Python port of the ComfyUI **"Qwen Image Edit"** workflow.

Given a photo (or one video frame) and a rough idea, a 4-bit Qwen3.8-27B (vision kept) writes one precise edit instruction. Qwen-Image-Edit (the 2511-generation *Qwen-Rapid-AIO* fine-tune) then renders the edited photo at 2 MP in 8 steps.

## Install

```bash
pixi run serve   # install everything into .pixi/, build the page, serve it at http://localhost:8765
```

Needs [pixi](https://pixi.sh), an NVIDIA driver, a CUDA GPU (developed on an RTX 3090, 24 GB) and ~32 GB of free RAM. Nothing else is taken from the system.
`exllamav3` has no wheel for this torch, so it builds from source, and `comfy-kitchen` is built for this machine's card only (see [Speed](#speed)). They need a CUDA 13.3 toolkit and a gcc nvcc 13.3 accepts (14; it rejects 16), so pixi installs both from conda-forge, with Python, node for the page and every PyPI package, into `.pixi/`, locked together in `pixi.lock` (the `[tool.pixi.*]` tables in `pyproject.toml`). `toolchain.sh` points the builds at the toolchain and reads the card's architecture from `nvidia-smi`. For PyCharm, the interpreter is `.pixi/envs/default/bin/python`. pixi prints nothing while a source build runs unless asked, with `RUST_LOG=uv_build_frontend=debug pixi install`.

## Development

```bash
pixi run ruff format   # format (also the Python blocks in Markdown files)
pixi run ruff check    # lint
pixi run mypy          # strict type check of difflab/
```

mypy treats the large, dynamically typed ML libraries (diffusers, transformers, accelerate, safetensors, av) as `Any`; see `pyproject.toml`.

## Models

| Role | Where |
|---|---|
| Editor: transformer + text encoder + VAE (single-file AIO, v18 = the original graph's file) | HF `Phr00t/Qwen-Image-Edit-Rapid-AIO` @ `691024f4`, `v18/Qwen-Rapid-AIO-SFW-v18.safetensors` |
| Newer AIO revision (set `edit_image`'s model setting to it) | the same repo, `v23/Qwen-Rapid-AIO-SFW-v23.safetensors` |
| The VLM, which sees and writes every prompt (4.0 bpw EXL3 of HF `Qwen/Qwen3.8-27B`, MTP head kept) | HF `turboderp/Qwen3.8-27B-exl3` @ `113cf7ab` |
| Editor tokenizer (not stored in the AIO) | HF `Qwen/Qwen-Image` @ `75e0b4be`, `tokenizer/*` |

Every model is named by where it comes from, pinned to a commit (`core/hub.py`): a Hugging Face repo (`Checkpoint`, `HubFile`), or, for the one file not on the Hub, a GitHub raw URL checked by sha256 (`UrlFile`, cached under `~/.cache/difflab`). The first fetch of everything is about 96 GB. The VLM is a 4-bit EXL3, which ExLlamaV3 runs in-process with its MTP head drafting ahead, at 70-100 tok/s on a 3090 (`config.VLM_DRAFT_TOKENS`); its checkpoint setting takes EXL3 checkpoints only. Each of them downloads into the HF cache the first time its model loads, then leaves a marker file there, so later loads never touch the network. That fetch is the only download difflab does: every `from_pretrained` call gets a local path plus `local_files_only=True`.

## Usage

Start the page with `pixi run serve` and render from it. Paths and machine-level switches live in `difflab/config.py`; every per-call knob and every model file is a field of a frozen settings dataclass you pass in (and can override per machine or production on the Settings page) — `EditSettings`, `VlmSettings`, `Continuity`, `EnlargePixelsSettings` and `RetimeSettings` in `core.settings`, `VideoSettings` beside the model it configures.

A render writes one line to the terminal:

```
difflab · watching at http://localhost:8765/runs/2026-09-20-2117
```

Everything else — the settings it started with, the VLM's answer streaming in, a ✓ line per stage
with its sizes and timings, the denoising bar and its ETA, the previews, the finished chunks and the
paths it wrote — is on that page, drawn from the run's events in `output/factory.db` (see [the page](#the-page)).

`core/ui`'s calls — `step`, `progress`, `sampling`, `panel` — are each purely an event. **A process that is recording nothing prints them instead**: a check that
loads no model opens `events.quiet()` because a report with no reader is useless, so its own goes
to stdout. Library logs and progress bars are silenced in
`difflab/__init__.py` either way.

The pieces can also be used on their own:

```python
from difflab.core import media as io
from difflab.tools import ask_vlm, edit_image, write_prompt
from difflab.core.settings import EditSettings

frame = io.frame_from_image("photo.jpg")
print(ask_vlm.ask(frame, "What is the man on the left wearing?"))
instruction = write_prompt.write_edit_instruction(frame, "give him a green hoodie, same pose")

out = edit_image.edit(frame, instruction, EditSettings(seed=42))
io.save(out, "edit", seed=42)
```

Outputs go to `output/images/<YYYY-MM-DD>/<YYYY-MM-DD-HHMMSS>_<model_name>_<seed>.png`, plus a lossless WebP copy.

## How the AIO file is loaded (no conversion, no bf16 copy)

`difflab/models/qwen_edit_aio.py` reads the single `.safetensors` file directly:

1. Each model is built on the meta device, so the skeleton costs no memory.
2. Only the tensors under one component's prefix are read (`model.diffusion_model.`, `text_encoders.qwen25_7b.transformer.`, `vae.`). Keys are renamed where needed, and the tensors are assigned as-is with `load_state_dict(assign=True)`, with no dtype change.
3. diffusers' `apply_layerwise_casting` keeps Linear/Conv/Embedding weights in **fp8** and upcasts each layer to bf16 only while it runs. ComfyUI does the same on GPUs without fp8 matmul.

The only tensors upcast to bf16 at load are the norms and in/out projections (0.04 GB). The VAE is bf16 on disk (0.24 GB). Sizes: transformer 19.0 GB fp8, text encoder 7.2 GB fp8. Because the tensors stay memory-mapped, loading takes 1–5 s. The configs (Qwen-Image-Edit-2511 transformer / Qwen2.5-VL-7B text encoder / scheduler / image processor) are dicts in `qwen_edit_aio.py`.

## Memory and speed

Built for one machine: an RTX 3090 (24 GB). The 4-bit VLM (~17 GB) is evicted before the editor loads. The transformer streams one block at a time and the text encoder one layer at a time from pinned CPU RAM, prefetched on a CUDA stream. The VAE stays on the GPU.

Measured on that machine (2026-09-16):

- 2 MP edit, 8 steps: ~62 s (7.1 s/step; the first step is slower at ~13 s)
- VLM instruction: ~11 s including the 4-bit load (Qwen3-VL 8B, the VLM before Qwen3.8 27B; not re-measured)
- Peak VRAM: 14.4 GB
- Process RAM: ~30 GB (pinned copies of the streamed weights)
- A whole edit run from a fresh process (VLM load → instruction → editor load → edit → save): 90–105 s. About 5 s of that is saving the optimized PNG and lossless WebP.

## ComfyUI node → Python mapping

| ComfyUI node | Here |
|---|---|
| `LoadImage` / `ImagePass` | `imaging.load_image`: PIL, EXIF-transposed, RGB |
| `LoadVideo` → `GetVideoComponents` → `Frame Select(0)` (dangling in the graph) | `imaging.load_video_frame(path, 0)` via `av` |
| `AILab_QwenVL` (Qwen-VL, 4-bit, flash_attention_2, max_tokens 2048, keep_model_loaded) | `exl3vl.Exl3VL`: Qwen3.8-27B as 4.0 bpw EXL3 on ExLlamaV3 with MTP drafting, `max_new_tokens=2048`, temperature 0.7 / top_p 0.8 / top_k 20, seeded. Stays resident. |
| `StringConcatenate` (system prompt + rough idea) | Chat template: system = `prompts.VLM_SYSTEM_PROMPT`, user = image + idea |
| `CheckpointLoaderSimple` (AIO) | `qwen_edit_aio.load_transformer / load_text_encoder / load_vae`, directly from the `.safetensors` |
| `ImageScaleToTotalPixels` (2 MP, lanczos, steps 1) | `imaging.resize_to_megapixels`: `scale = sqrt(2e6 / (w*h))`, LANCZOS |
| `VAEEncode` + `TextEncodeQwenImageEditPlus` / `TextEncodeQwenImageEdit` | Inside `QwenImageEditPlusPipeline` |
| `ClownsharKSampler_Beta` (8 steps, cfg 1, denoise 1, euler, eta 0.5) | Pipeline: `num_inference_steps=8`, `true_cfg_scale=1.0`, default `FlowMatchEulerDiscreteScheduler`, CPU `torch.Generator` seeded |
| `VAEDecode` | Pipeline output |
| `Image Saver Simple` (png optimize, lossless webp q100, `%time_%basemodelname_%seed`, `qwen_edit/%date`) | `imaging.save_result`, into `output/images/%date`. "Embed workflow" is skipped (ComfyUI-only). |

## Known simplifications and deviations

1. **Automated hand-off.** In ComfyUI a person copied the VLM output into a string node. Here it feeds the edit stage directly. Manual mode is `main.manual_prompt()`, which hands `edit_image.edit` a written instruction.
2. **Default flow-match Euler scheduler** instead of the custom "bongmath/beta" sampler (its eta 0.5 has no equivalent). There's no hand-rolled sampler.
3. **One image for both conditionings.** The pipeline derives both the VL-encoder image (384² area) and the VAE reference latent (1 MP) from the 2 MP-resized image. In ComfyUI, the CLIP side saw the original-resolution image.
4. **No AIO→diffusers conversion step.** The AIO file is loaded directly (see above) instead of being split into a diffusers folder on disk.
5. **`QwenImageEditPlusPipeline`**, not `QwenImageEditPipeline`. The checkpoint is a 2511-generation model and the graph used `TextEncodeQwenImageEditPlus`. The Plus pipeline is its diffusers counterpart ("Picture 1:" template, multi-image-style conditioning).
6. **cfg → `true_cfg_scale`.** In diffusers' Qwen pipelines, `guidance_scale` is only for guidance-embedding models, and this one has none. At cfg 1.0 the negative prompt is inactive (ComfyUI also skips the uncond pass at cfg 1). It becomes active when `CFG > 1`.
7. **No `strength` argument.** The Plus pipeline always starts from pure noise, which is `denoise=1.0`.
8. **Output size is floored to a multiple of 16** (the pipeline's latent packing). For example, 2 MP of a 4:3 photo gives 1632×1216.
9. **No CLIP vision.** The AIO has none, and Qwen-Image-Edit doesn't use one: image understanding is the Qwen2.5-VL vision tower inside the text encoder. `clip_vision_h.safetensors` is unused.
10. **Tokenizer from the local Qwen-Image HF cache.** The AIO has no tokenizer files.
11. **Text encoder `lm_head`** is absent from the AIO. It's replaced by `Identity` because the pipeline only reads hidden states.
12. **PIL image-processor backend.** torchvision is not a dependency, so video preprocessing in the processors is unavailable. Only still images are ever passed.

## Determinism

The same seed and the same input give the same output. The edit noise comes from a seeded CPU generator, and the VLM by its job's seed.

Verified 2026-09-16:
- A manual-prompt edit repeated in a separate process, with a different offload setup, was **pixel-identical**.
- The full workflow run twice in separate processes gave the same VLM instruction. The two images differ by at most 1/255 on a few pixels (mean 0.01/255) and look identical. The first run decoded under VRAM pressure (before the text encoder was streamed), so cuDNN chose different conv algorithms, which round differently.

Caveats: expect last-bit differences whenever memory pressure or a different GPU/driver changes the cuDNN algorithm choice. ExLlamaV3's kernels are not guaranteed bit-exact across GPUs or drivers either. The runs above used SDPA on one GPU.

## Offline proof

Once every checkpoint has been fetched, nothing touches the network. The strongest check is to run with no network at all:

```bash
unshare -rn .pixi/envs/default/bin/python -c 'import main; main.full_workflow()'
```

Result (2026-09-16): inside a network namespace with no interfaces up, the full workflow (VLM load → instruction → editor load → 2 MP edit → PNG + WebP) completed with exit code 0.

# Video: MiniMax H3 image-to-video + native audio

A port of the ComfyUI **"MiniMax H3 (Hailuo 3.0) image-to-video turbo"** workflow: a photo becomes a 24 fps clip whose **first frame is that photo**, with matching ambience and sound effects generated in the same pass. Code lives in `difflab/models/minimax_h3/`, driven by `difflab/tools/render_chunk.py`.

## Usage

```python
from difflab.core import media
from difflab.tools import render_chunk, write_prompt

frame = media.frame_from_image("photo.jpg")
clip = render_chunk.render(frame, "0-5s: she turns toward the camera and smiles.")  # written prompt, no VLM
clip = render_chunk.render(frame, write_prompt.write_video_prompt(frame, "she turns and smiles"))
media.save(clip, "clip", seed)
```

Geometry, sampling knobs and model files are `VideoSettings` (`duration`, `megapixels`, `seed`, `steps`, `reference_size`, the shifts, the DiT, LoRA, text encoder and both VAEs). `difflab/models/minimax_h3/config.py` keeps only the model's fixed facts. Output goes to `output/videos/<date>/<time>_<model>_<seed>.mp4`.

## Models (fetched on first use, see `core/hub.py`)

| Role | Source |
|---|---|
| DiT, 20 B packed video+audio (`fl2va_8step` preset) | `Comfy-Org/MiniMax-H3`: `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Turbo LoRA, 8 steps | `Comfy-Org/MiniMax-H3`: `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` |
| DiT + LoRA for the `ref2va_4step` preset | `Comfy-Org/MiniMax-H3`: `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` + `larryvrh/MiniMax-H3-Turbo-Lora`: `minimax_h3_turbo_v4_step600_ema.safetensors` |
| Text encoder, Qwen3-VL-32B truncated to 50 layers | `Comfy-Org/MiniMax-H3` |
| Its tokenizer (H3's special tokens included) | `MiniMaxAI/MiniMax-H3`: `tokenizer/*` |
| Video VAE / audio VAE | `Comfy-Org/MiniMax-H3`: `vae/minimax_h3_video_vae_int8_convrot.safetensors` (decoder linears int8), `vae/minimax_h3_audio_vae_fp32.safetensors` |
| AdaLN silu grid (for the v4 LoRA on a pruned base) | GitHub `larryvrh/ComfyUI-MiniMax-H3-Turbo` @ `4274783a`: `h3_silu_temb_grid.safetensors` |
| Upscalers: 4xLSDIR / RealESRGAN x2 | `nuriyoo/openmodeldb-mirror` / `ai-forever/Real-ESRGAN` |
| Frame interpolation, RIFE 4.26 | `Comfy-Org/frame_interpolation`: `frame_interpolation/rife_v4.26.safetensors` |
| Preview decoder, taeh3 | `t8star/Taeh3-Comfy`: `models/vae_approx/taeh3_2d_kijai.safetensors` |

## How it fits in 24 GB

The DiT is 19.5 GB of int8 weights and the text encoder 22.7 GB, so nothing is resident for long:

1. **int8 "ConvRot"** (`models/convrot.py`): weights stay int8 with per-row scales and a 256-wide Hadamard rotation; rotating, quantizing and multiplying the activation is one fused `comfy_kitchen.int8_linear` call — the same Apache-2.0 kernel ComfyUI runs. Dequantization matches ComfyUI's own to 2.4e-07.
2. **Chunked feed-forwards** (`transformer.ChunkedFeedForward`): the SwiGLU intermediate is 28672 wide, so a full-length clip would materialize 4.7 GB of it per block. Slicing the token axis is exact and keeps the transient at a few hundred MB.
3. **Block streaming** (`models/offload.py`): each block's tensors are copied in for its forward and released after — 15 ms per 385 MB block, so ~0.8 s per step for all 50.
4. **Stage-by-stage loading**: VAE encode → text encoder → DiT → VAEs decode, each freed before the next. Freeing a stage also drops its checkpoint's page cache (`core.models.Registry.evict`): safetensors maps rather than copies, so those 25 GB and 18 GB would otherwise sit there while the next stage reads its own.

When a clip is long enough that something still does not fit, the run backs off instead of raising:
attention retries in head groups (1.5% slower, 23% smaller) and the VAE decoder halves its tile
group. Both say so on stdout, and both are reached only after the allocation has actually failed.

Host RAM matters as much as VRAM here: a render maps ~26 GB of text encoder and ~18 GB of DiT, so a
machine with 32 GB or less will swap even though the card is fine.

## Upscaling

Generating large is expensive and enlarging afterwards is not: sampling cost grows with the square of
the packed sequence, so pixels are the costly thing to *generate* and a cheap thing to *add*.

`tools.enlarge_pixels` runs an ESRGAN (4xLSDIR) over finished frames. It can only sharpen what the decoder
already drew, which is a ceiling rather than a tuning problem.

The latent route, `LBH-123-AI/Minimax_h3_latent_Upscaler` enlarging the latent before H3's own decoder
renders it, measured better on a synthetic test - reconstructing a native render from a half-size one:

| route | detail vs native | shimmer vs native | time |
|---|---|---|---|
| bicubic (floor) | 0.236 | 1.057 | — |
| decode → 4xLSDIR | 1.053 | 1.173 | 15.4 s |
| **upscale latent → decode** | **1.261** | **1.054** | **9.3 s** |

On a real chapter it lost. Taking production 13's 1056×608 shots to 4K (2026-10-06), at 2× and at 3.6×
alike, it redrew the face - another eye shape, an older man, colour fringes on the hair - while the
ESRGAN kept the render's own face and only sharpened it, at the same 2.6 s a 4K frame. It was removed;
`git log -- difflab/tools/enlarge_latents.py` has it.

Every chapter render ends with a finish (`tools.restore`). After the native join, each shot's cached
latents are decoded and given the colour hold its native chunk got, filled to half of
`config.FINISH_MEGAPIXELS` (1920×1080; a canvas within a tenth of 16:9 is filled and cropped), restored by
SeedVR2 7B (`models/seedvr2`) in 33-frame windows sharing 4, and doubled by RealESRGAN x2 to 3840×2160
straight into the file. The 4K chunks are joined at the native cuts and retimed to `config.FINISH_FPS` by
RIFE (`finish`). Every re-encode after the chunks is at crf 16 (`media.JOIN_CRF`): x264's default 23 on
the join and the retime halved the finish's fine detail.

SeedVR2 against the ESRGAN alone, on production 13 (2026-10-06): the ESRGAN left ladders and ringing on
every edge; SeedVR2 drew clean ones, and at 4K kept the render's face. At 1080p it redraws more of a small
face - ComfyUI's own port does the same at that size, so it is the model, not the port. On a 3090 at
1080p a frame costs 0.17 s of DiT, 1.8 s of VAE (each 3D conv already at 62 TFLOPS: arithmetic, not
overhead) and 0.63 s of fp16 ESRGAN, about 3 s with the rest; restoring at 4K would quadruple the VAE.

Among the pixel upscalers, 4xLSDIR measured best of the three on disk and is the default;
4xNMKDSuperscale hallucinates detail that was never there (sharpness 1.113) and shimmers at 1.705 —
keep it away from video. See `models/esrgan.py` for the full table.

## Frame rate

H3 draws 24 fps and nothing changes that: the rate is baked into the VAE's temporal packing
(`n % 17 == 5`) and into the model's sense of how fast a thing moves. 48 or 60 is therefore a
finishing step, and `tools.retime` is RIFE 4.26 doing it — a flow network that works out where each
pixel went between two frames and draws the picture partway, rather than blending two frames and
calling the smear a frame.

```python
clip = retime.retime(clip, fps=60)  # the same seconds, the same soundtrack, 2.5x the frames
```

The take is not touched: the clip is exactly as long as it was, so the audio passes through
unchanged. What it costs is one forward pass of a 22 MB network per *made* frame, with the per-image
encoder's output shared by every frame made between the same pair — at 60 fps that is two encodes
serving four interpolations. A frame that lands exactly on a source frame is that frame, copied.

Three things worth knowing about this version, because the internet still gets them wrong:

* **4.25 and 4.26 are the same architecture** — five flow blocks, not the four the upstream file's own
  (stale) default argument suggests. The checkpoint here is ComfyUI's safetensors conversion of
  `RIFEv4.26_0921`, which is what the reference workflow's node downloads; the official release is a
  zip around a pickle.
* **`ensemble` and `fastmode` do not exist here.** The refine U-net went after 4.3 and the
  swap-and-average ensemble after 4.17. Anything offering them is describing an older RIFE.
* **`t` is a channel, not a recursion depth.** It arrives as a full-resolution constant plane, so any
  timestep is one forward pass: 24 → 60 (t cycling 0, 0.4, 0.8, 0.2, 0.6) costs the same per output
  frame as 24 → 48, with no power-of-two ladder.

Like upscaling, it is a separate tool and never something a render does on your behalf — and for the same
reason a cut needs different treatment: RIFE morphs across one instead of holding the frame, so this
belongs on a take, or on a film whose joins are crossfades.

A job asks for it with `retime_fps` — the **delivery** group in the app, which every mode that makes
a video shows. There `retime_file` is what runs: the finished mp4 read a frame at a time and written
out beside itself as `<name>_60fps.mp4`, so a minute at 2 MP costs two frames of memory rather than
the tens of gigabytes the in-memory route would ask for. The 24 fps render is kept, because it is
what every chunk on disk agrees with.

## Colour drift

A chain inherits its own colour. Chunk k+1 opens on chunk k's pixels, so whatever the model does to
the palette is carried forward rather than corrected — and what it does is regress toward its own
prior. Measured over a six-chunk take that opens on coals:

| | G/R | B/R |
|---|---|---|
| first sixth | 0.464 | 0.168 |
| fifth sixth | 0.641 | 0.574 |

Green rose 38% against red and blue more than tripled: deep orange became a yellow-grey, at every
join and at none of them backwards. There are two fixes and they are not alternatives.

**`Continuity.color_anchor`** (the handoff tool's colour anchor setting) fixes the cause. It pulls the frame each chunk opens on back toward
the take's own opening (`media.color_gain`, an exponent: 0.5 closes half the gap), so the model starts
each chunk from the right palette instead of inheriting the last one's. It is off by default because
it is expensive: the opening frame's pixels are part of `render_chunk.chunk_key`, so turning it on
re-renders every chunk after the first.

**`tools.grade`** fixes the symptom, over the finished file, in one pass and with no sampling:

```python
grade.grade(Path("output/videos/2026-09-20/take.mp4"), strength=0.5)
```

Measured on the take above — the spread across sixths of the video, which is the drift itself:

| strength | G/R spread | B/R spread |
|---|---|---|
| none | 0.177 | 0.406 |
| 0.5 | 0.084 | 0.137 |
| 0.75 | 0.042 | 0.060 |
| 1.0 | 0.004 | 0.023 |

It is a grade and not a filter because of two choices. The **reference is the take's own opening** —
the first second, the frame a person picked and the one chunk 1 was conditioned on — so a graded take
begins exactly as it did and only what left it is pulled back. And the correction follows a
**smoothed trajectory** rather than each frame: drift is slow by definition, so the channel means are
averaged over about a second first, and a cloud crossing the sun or a lamp switched on moves faster
than the window and survives. Correcting frame by frame would flatten those and add its own flicker.

Prefer a half. A take is allowed to change its light, and a full match argues with it every time it
does — the 1.0 row above is the drift gone and the scene's own changes gone with it.

A job asks for it with `grade_strength`, in the app's **delivery** group beside the frame rate. There
the two finishes chain: grading runs first, because a colour correction belongs on the frames the
model actually drew rather than on the ones RIFE invented between them — and because it is a fifth of
the work before the frame count goes up. Each writes beside its input (`take.mp4` →
`take_graded.mp4` → `take_graded_60fps.mp4`), so the plain render, which is what every chunk on disk
agrees with, is still there when a grade turns out to have gone too far.

## Redoing a video

A source video and a sentence about what should be different (`POST /api/ingest/video`). The video
is cut into shots of the render length - a remainder under half a shot is left alone - and each shot
is handed three things: the source's own frame at that moment to open on, that stretch of the source
as `<Video 1>`, and a caption saying to do what the reference does with the change applied. H3 has no
video-to-video path; this is built out of ref2va, which is the part of it that does take a clip.

Anchoring every shot to the source is what keeps a long edit honest - shot k opens on the source at
shot k's start rather than on wherever shot k-1 drifted to - and the motion tail still crosses the
join, so the anchor arrives with velocity and the crossfade softens what is otherwise a cut. The tail
is `<Video 2>`, because the source is `<Video 1>` in every shot whether that shot has a tail or not.

**What fits is the whole design.** A reference clip costs its latent frames times the canvas's rows,
on every sampling step, and the card holds about 105,000 rows:

| chunk | reference | rows | sampling | |
|---|---|---|---|---|
| 12 s at 0.4 MP | 12 s, full size | 70,640 | ~6 min | estimated |
| 12 s at 0.98 MP | none (a plain take) | 93,716 | 12 min 28 s | **measured** |
| 12 s at 0.98 MP | 12 s, half scale | 113,482 | 17–20 min, 17.3 GB peak | **measured** |
| 12 s at 0.98 MP | 12 s, full size | 174,356 | ~31 min | estimated |

0.98 MP with `clip_scale` 0.5 is the one to reach for: it renders, at about 1.4× a plain take, and
the whole 12 s of source is still shown - at 384×672 rather than 768×1344, which is conditioning
rather than output. Attention is about 75% of a step and quadratic in rows, so rows are time as much
as memory; the weights already stream from RAM a block at a time (`models/offload.py`), so what is
resident is activations.

## A run as data

Every run is rows in `output/factory.db`: the run, its jobs with their inputs, outputs and keys, each
job's tasks with a bar, every event the render emits (with VRAM and RAM samples every 2 s), and every
command a page sent. The worker is a thread inside the server process, one job at a time in the order
asked, so a Render click never waits and never refuses. A run left `running` by a process that died
is failed when the worker next starts; queued runs survive a restart.

- **Overall** - what is running, the job and the task it is on, VRAM and RAM.
- **Shots** - a card per shot (done / running / pending), its prompt, seed, opening frame and references.
- **Previews while sampling** - after every step the pipeline hands over its denoised estimate
  (`x0 = x_t + (1 - t) v`), which `taeh3` - ComfyUI's 9 MB preview decoder - draws at half size.
- **Finished chunks** play as soon as they are written; the joined video when it is assembled.
- **Controls** - pause after this job, cancel now. A shot is re-rolled on its production: a new seed
  and the chapter queued again, which skips every shot whose files are already there.

```
sqlite3 output/factory.db "select tool, state, key from jobs where run_id = '<id>'"
```

## The page

```
pixi run page                  # once, and after a page change
pixi run python -m difflab.api   # http://localhost:8765
```

Four screens. **Productions** lists them and begins one - an idea and photographs, a text, or a
video - and a production's own screen shows its cast with their reference crops, its shots with
their keyframe, prompt and clip, and the two buttons that make a run: *Plan* (the director) and
*Render* (the chain). A prompt can be read and edited there, which pins it; a keyframe can be
chained, composed by the editor from the plan's words, or a picture you give; a shot re-rolls with a
new seed. **Quick** is the three quick modes as a form the server checks on every edit. **Runs** lists
every run, and one run's screen is the render as it happens or its delivery once it ended.
**Settings** is the models first - one picker per role, set on every tool that loads it, remembering
what was pasted before until it is forgotten - then the knobs that matter, grouped by what they
change, with the rest behind Advanced. For this machine or one production, and priced as it is edited:
the keyed fields that move, and the made things in scope that go stale for it.

Photos are uploaded and kept under `output/assets/`, named by their own content hash, so the same
picture sent twice is one file. For working on the page itself, `npm --prefix frontend run dev`
serves it with hot reload and passes the API through to the Python process.

### Serving it

**There is no authentication.** On a LAN that is a choice; on a pod's public URL it is an open card
and a public gallery of everything it makes, and the server says so on every start that is not bound
to localhost. Put it behind something, or set `DIFFLAB_HOST=127.0.0.1` and reach it through a tunnel.

`DIFFLAB_HOST` and `DIFFLAB_PORT` are the only two knobs that read the environment, so a container
can be configured without editing `config.py`. The `Dockerfile` builds the page and the package with the
same `pyproject.toml` and `pixi.lock` and runs the server. One image serves every card from Ampere to Blackwell (`ghcr.io/diverofdark/difflab:latest`): comfy-kitchen
is compiled for its own release list less Turing plus `86-real`, and exllamav3 for the same cards, so each card runs
code built for it. `--build-arg COMFY_CUDA_ARCHS=89-real --build-arg TORCH_CUDA_ARCH_LIST=8.9` builds for one card only,
in a sixth of the kernel compile.

Everything persistent - weights, downloads, the database and every output - lives under one volume,
`/data` (`-v difflab:/data`); without it each new pod downloads ~96 GB of weights again.

## Speed

Attention is ~75% of a step at full length, and **comfy-kitchen's int8 attention** (`attention.py`)
is what makes that affordable. On real H3 Q/K/V at 95849 packed rows, one block takes 4321 ms in SDPA,
2193 ms in SageAttention 1.0.6, 1851 ms in `comfy_kitchen.int8_attention` from PyPI and **1585 ms**
with the same kernel built for sm_86 — a relative error of 0.7–1.2% against bf16 SDPA, which is Sage's
own level and the same order as the int8 weights the model already runs on. Kitchen quantizes V and
the softmax probabilities too, and holds the error there by Hadamard-rotating Q and K first. It is the
same kernel pack as the int8 linears, so it costs no extra dependency.

The sm_86 build is the reason `pyproject.toml` takes comfy-kitchen from its git tag: the published
wheel ships cubins for sm_75/80/89/90a/100/120, so a 3090 runs the sm_80 code, where each int→float in
the softmax loop is an `I2F` on the quarter-rate special-function unit. Built for `86-real`, they are
`I2FP` on the full-rate ALU — 17% off attention for bit-identical output. `pixi install` builds it in
about four minutes (four compiles at a time via `ck-jobs.cmake`) for whatever card `nvidia-smi`
reports; `COMFY_CUDA_ARCHS` set by hand wins. int8 *linears* come out unchanged, at 140–205 TOPS either way.

## Measured (RTX 3090, 2026-09-16)

| | 2 s @ 0.4 MP (default) | 15 s @ 0.8 MP, 4 steps |
|---|---|---|
| Sampling | 62 s for 8 steps (7.8 s/step) | 488 s (**121 s/step**) |
| Text conditioning | 21 s | 12 s |
| Decode (video + audio), fp16 VAE | 11 s | 105 s |
| Peak VRAM | 5.7 GB | 8.0 GB while sampling |

ComfyUI renders the same 15 s clip — same checkpoint, LoRA, step count and geometry — at **119–136
s/step** (typically ~126), so the two are at parity. Getting there took two changes: the fused `int8_linear` kernel (1.8–3.5× over the hand-written
PyTorch path) and SageAttention (1.85× on the dominant term), since replaced by kitchen's int8
attention, built for sm_86 (a further 1.38×).

## ComfyUI node → Python mapping

| ComfyUI node | Here |
|---|---|
| `LoadImage` / `LoadVideo` → frame 0 | `imaging.load_image` / `imaging.load_video_frame` |
| `ImageScaleToTotalPixels` (0.4 MP, nearest-exact, steps 32) | `pipeline.canvas_size` + `pipeline.resize_to_canvas` (antialiased bicubic, as the H3 nodes' own lanczos) |
| `AILab_QwenVL_Advanced` (4-bit, 4096 tokens, temp 0.6, top_p 0.9) | `tools.write_prompt.write_video_prompt` |
| `UNETLoader` + `LoraLoaderModelOnly` | `transformer.load` (int8, curve AdaLN, both LoRA formats) |
| `CLIPLoader` type `minimax` | `text_encoder.load` + `encode` (50 layers, `<Picture 1>:` presentation, modality tags) |
| `MiniMaxH3ImageToVideo` | `pipeline.plan` + `build_packed_sequence` + the conditioning rows |
| `MiniMaxH3SigmaShift` (12 / 3) | `pipeline.make_schedulers` |
| `BasicScheduler` + `KSamplerSelect(euler)` + `SamplerCustomAdvanced` | `pipeline.sample` |
| `VAEDecode` / `VAEDecodeAudio` | `vae.decode_video` / `vae.decode_audio` |
| `CreateVideo` + `SaveVideo` | `video_io.save_video` (H.264 + AAC at 32 kHz) |
| dangling video-measuring chain | `video_io.probe_duration` |

## Known simplifications and deviations

1. **Int8 attention, not Sage**: the reference workflow patches every block onto SageAttention (KJNodes' *MiniMax H3 Memory Efficient Sage Attention Patch*); `attention.py` runs `comfy_kitchen.int8_attention` instead - 1.38× faster at the same error against SDPA (0.7–1.2%), with the kernel pack built for this GPU's sm_86.
2. **diffusers' H3 model code** is the base; only what it lacks is hand-written (ComfyUI → diffusers conversion, int8 ConvRot, the pruned curve AdaLN, and the turbo LoRA's AdaLN injection).
3. **Two presets**, not one: `fl2va_8step` matches the saved ComfyUI workflow, `ref2va_4step` matches the written spec.
4. **ref2va reference mode** is implemented in full: reference **images**, **videos** with their soundtracks, and **standalone audio** (`reference_audio`), each soundtrack packed just before its own video rows. The DiT's layout allows a first frame and references together, so `render_chunk.render(frame, prompt, references=[...], reference_clips=[...], reference_audio=[...])` keeps the opening *and* holds the subject — the two ComfyUI nodes are either/or only because they are two nodes.

   A reference video is expensive: its rows ride through every sampling step, and 7 latent frames at the dev canvas cost 2975 rows against the target's 7225. Attention is quadratic in the packed length, so budget for it.
5. The VAE decoder's fused QKV is grouped **per head** — splitting it by row thirds concatenates back to the same matrix and silently scrambles attention, so it is split per head (`vae.split_interleaved_qkv`).
6. The text encoder runs in bf16 where ComfyUI runs fp32, so expect small numeric drift.
7. The video VAE is Comfy-Org's **int8 ConvRot** file, where the saved workflow loads the fp16 one: its decoder linears run W8A8 through the same kernel as the DiT, 1.7x faster (6.96 against 11.97 s per 2 MP clip). Against the fp16 VAE: 58.9 dB PSNR, 0.13/255 mean difference. Its latent mean/std are the fp32 originals, ComfyUI's own constants.
