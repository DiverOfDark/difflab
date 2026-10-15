# syntax=docker/dockerfile:1
# difflab on a GPU host, as one process: uvicorn serves the page and the worker thread beside it renders.
#
# The environment is pixi's, from pyproject.toml and pixi.lock exactly as on a workstation. The build
# stage holds it whole, compilers included; the image gets it without the 2.7 GB only the kernel builds
# use, in four layers - nvidia's libraries, torch, triton, the rest - so a lock change re-pushes and
# re-pulls only the ones whose files changed. That needs the same bytes for the same files: CI builds
# with SOURCE_DATE_EPOCH and rewrite-timestamp, without which every rebuild stamps new mtimes on all four.
#
# One image for every card from Ampere to Blackwell: comfy-kitchen's own release list (its setup.py's Linux
# default, with PTX at 89 for anything newer) less Turing, where exllamav3 is experimental, plus 86-real,
# which the release leaves on sm_80 code: 17% slower attention on a 3090. --build-arg COMFY_CUDA_ARCHS=<cc>-real with
# TORCH_CUDA_ARCH_LIST=<c.c> builds for one card only, in a sixth of the kernel compile.

FROM node:22-slim AS page
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN npm --prefix frontend ci
COPY frontend ./frontend
RUN npm --prefix frontend run build

FROM ghcr.io/prefix-dev/pixi:0.81.0 AS env

ARG COMFY_CUDA_ARCHS=80-real;86-real;89;90a-real;100f;120f
# The same cards for exllamav3, in torch's spelling, which has no family ("f") targets.
ARG TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9+PTX;9.0;10.0;12.0
ARG MAX_JOBS=4
ARG NVCC_THREADS=4
ENV COMFY_CUDA_ARCHS=${COMFY_CUDA_ARCHS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS}

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml pixi.lock toolchain.sh ck-jobs.cmake ./
# The mount is pixi's whole cache, since uv moves a built wheel into place from elsewhere in it; what
# survives the step is the compiled wheels and comfy-kitchen's checkout. A changed lock re-downloads the
# rest in minutes, where the kernels are hours on a CI runner. One per architecture, because the wheel
# cache does not key on it. pixi logs a source build's output only at uv's debug level, and without
# it the hours of kernel builds print nothing.
RUN --mount=type=cache,id=difflab-kernels-${COMFY_CUDA_ARCHS},target=/root/.cache/rattler/cache \
    RUST_LOG=uv_build_frontend=debug pixi install --locked -e prod \
    && rm -rf .pixi/tmp \
    && find /root/.cache/rattler/cache -mindepth 1 -maxdepth 1 ! -name uv-cache -exec rm -rf {} + \
    && find /root/.cache/rattler/cache/uv-cache -mindepth 1 -maxdepth 1 ! -name sdists-v9 ! -name git-v0 -exec rm -rf {} +

# Conda packages are kept when python or ffmpeg needs them, or when they put files in site-packages,
# since a PyPI dependency they satisfy was not installed again; cuda-cudart because exllamav3_ext's
# rpath finds libcudart there. Everything else is the compilers, nvcc, cmake, node and conda's copies
# of cuBLAS and friends - the extensions and comfy-kitchen's ctypes load nvidia/cu13's, which torch
# needs anyway.
RUN .pixi/envs/prod/bin/python <<'EOF'
import json, pathlib, shutil
env = pathlib.Path(".pixi/envs/prod")
meta = {}
for f in (env / "conda-meta").glob("*.json"):
    m = json.loads(f.read_text())
    meta[m["name"]] = (f, m)
todo = [n for n, (_, m) in meta.items()
        if n in {"python", "ffmpeg", "cuda-cudart", "cuda-version"} or any("site-packages/" in p for p in m["files"])]
keep = set()
while todo:
    n = todo.pop()
    if n in meta and n not in keep:
        keep.add(n)
        todo += [d.split()[0] for d in meta[n][1]["depends"]]
for n, (f, m) in meta.items():
    if n not in keep:
        for p in m["files"]:
            (env / p).unlink(missing_ok=True)
        f.unlink()
site = env / "lib/python3.13/site-packages"
split = pathlib.Path("/split")
split.mkdir()
for d in ("nvidia", "torch", "triton"):
    shutil.move(site / d, split / d)
EOF

# pixi's activation, written out once so the container needs no pixi at run time: PATH to the
# environment's ffmpeg, CUDA_HOME. Written after the pruning, so it sources no removed compiler's
# activation and leaves CC unset: triton then compiles its C shim with the image's gcc.
RUN pixi shell-hook -e prod --as-is --change-ps1 false > /app/activate.sh && echo 'exec "$@"' >> /app/activate.sh

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# --link: each is a layer of its own that does not need anything under it unpacked, so an unchanged one
# is referred to in the registry, not moved, and a host that has it pulls nothing.
COPY --link --from=env /split/nvidia ./.pixi/envs/prod/lib/python3.13/site-packages/nvidia
COPY --link --from=env /split/torch ./.pixi/envs/prod/lib/python3.13/site-packages/torch
COPY --link --from=env /split/triton ./.pixi/envs/prod/lib/python3.13/site-packages/triton
COPY --link --from=env /app/.pixi/envs/prod ./.pixi/envs/prod
COPY --link --from=env /app/activate.sh /app/toolchain.sh ./

# Everything that outlives a pod is under output/: Hugging Face weights, Civitai and URL downloads (and
# triton's compiled kernels) beside the database, blobs and finals.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/output/.models \
    XDG_CACHE_HOME=/app/output/.cache \
    DIFFLAB_HOST=0.0.0.0

COPY --link difflab ./difflab
COPY --link --from=page /app/difflab/api/static ./difflab/api/static

# Weights are fetched on first use; without a volume here the first render downloads ~96 GB again on
# every new pod.
VOLUME ["/app/output"]

EXPOSE 8765
# DIFFLAB_HOST above makes it reachable from outside, open to anyone unless DIFFLAB_TOKEN is set: see README, "Serving it".
ENTRYPOINT ["/bin/bash", "/app/activate.sh"]
CMD ["python", "-m", "difflab.api"]
