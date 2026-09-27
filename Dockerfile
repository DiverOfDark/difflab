# difflab on a GPU host, as one process: uvicorn serves the page and the worker thread beside it renders.
#
# The environment is pixi's, from pyproject.toml and pixi.lock exactly as on a workstation, and comes over
# to the runtime stage whole: its toolchain rides along, because causal-conv1d is built inside it.
# comfy-kitchen is compiled for one architecture: 86-real is a 3090, and another card needs
# --build-arg COMFY_CUDA_ARCHS=<its compute capability without the dot>-real, or its attention kernels
# will not load. An empty value builds comfy-kitchen's own list for every card, sm_86 on sm_80 code.
FROM ghcr.io/prefix-dev/pixi:0.81.0 AS build

ARG COMFY_CUDA_ARCHS=86-real
ENV COMFY_CUDA_ARCHS=${COMFY_CUDA_ARCHS}

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml pixi.lock toolchain.sh ck-jobs.cmake README.md ./
RUN pixi install --locked -e prod

COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN pixi run -e prod npm --prefix frontend ci
COPY frontend ./frontend
COPY difflab ./difflab
RUN pixi run -e prod npm --prefix frontend run build
COPY main.py ./

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    DIFFLAB_HOST=0.0.0.0

COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY --from=build /app/difflab /app/difflab
COPY --from=build /app/main.py /app/
WORKDIR /app

# Weights are fetched on first use into HF_HOME; mount a volume there or the first render downloads
# ~160 GB again on every new pod.
VOLUME ["/models", "/app/output"]

EXPOSE 8765
# DIFFLAB_HOST above makes it reachable from outside, with no authentication: see README, "Serving it".
CMD ["/app/.pixi/envs/prod/bin/python", "-m", "difflab.api"]
