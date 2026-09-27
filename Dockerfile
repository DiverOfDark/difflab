# syntax=docker/dockerfile:1
# difflab on a GPU host, as one process: uvicorn serves the page and the worker thread beside it renders.
#
# The environment is pixi's, from pyproject.toml and pixi.lock exactly as on a workstation, and the image
# is that stage itself rather than a copy of it: every layer is in the pushed image, so the image is its
# own build cache, and the environment layer is only rebuilt when the manifest, the lock or the toolchain
# script changes.
#
# comfy-kitchen is compiled for one architecture: 86-real is a 3090, and another card needs
# --build-arg COMFY_CUDA_ARCHS=<its compute capability without the dot>-real, or its attention kernels
# will not load. An empty value builds comfy-kitchen's own list for every card, sm_86 on sm_80 code.

FROM node:22-slim AS page
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN npm --prefix frontend ci
COPY frontend ./frontend
RUN npm --prefix frontend run build

FROM ghcr.io/prefix-dev/pixi:0.81.0

ARG COMFY_CUDA_ARCHS=86-real
ARG MAX_JOBS=4
ARG NVCC_THREADS=4
ENV COMFY_CUDA_ARCHS=${COMFY_CUDA_ARCHS} \
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

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    DIFFLAB_HOST=0.0.0.0

COPY main.py ./
COPY difflab ./difflab
COPY --from=page /app/difflab/api/static ./difflab/api/static

# Weights are fetched on first use into HF_HOME; mount a volume there or the first render downloads
# ~160 GB again on every new pod.
VOLUME ["/models", "/app/output"]

EXPOSE 8765
# DIFFLAB_HOST above makes it reachable from outside, with no authentication: see README, "Serving it".
CMD ["/app/.pixi/envs/prod/bin/python", "-m", "difflab.api"]
