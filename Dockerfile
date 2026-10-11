# syntax=docker/dockerfile:1
# difflab on a GPU host, as one process: uvicorn serves the page and the worker thread beside it renders.
#
# The environment is pixi's, from pyproject.toml and pixi.lock exactly as on a workstation, and the image
# is that stage itself rather than a copy of it: every layer is in the pushed image, so the image is its
# own build cache, and the environment layer is only rebuilt when the manifest, the lock or the toolchain
# script changes.
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

FROM ghcr.io/prefix-dev/pixi:0.81.0

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

# Everything that outlives a pod is under output/: Hugging Face weights, Civitai and URL downloads (and
# triton's compiled kernels) beside the database, blobs and finals.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/output/.models \
    XDG_CACHE_HOME=/app/output/.cache \
    DIFFLAB_HOST=0.0.0.0

# --link: the code is a layer of its own that does not need the environment under it unpacked. Without
# it a code-only build pulled and extracted the 4.9 GB environment layer to put 1 MB on top (5.4 of its
# 6.5 minutes on the home runner); with it the environment stays in the registry, referred to, not moved.
COPY --link difflab ./difflab
COPY --link --from=page /app/difflab/api/static ./difflab/api/static

# Weights are fetched on first use; without a volume here the first render downloads ~96 GB again on
# every new pod.
VOLUME ["/app/output"]

EXPOSE 8765
# DIFFLAB_HOST above makes it reachable from outside, with no authentication: see README, "Serving it".
CMD ["/app/.pixi/envs/prod/bin/python", "-m", "difflab.api"]
