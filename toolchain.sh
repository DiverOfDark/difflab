export CUDA_HOME="$CONDA_PREFIX"
export NVCC_CCBIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
# gxx's own activation prepends a bare -ccbin, which outranks NVCC_CCBIN and which nvcc cannot resolve
# inside an isolated build environment ("Failed to preprocess host compiler properties").
export NVCC_PREPEND_FLAGS="-ccbin=$NVCC_CCBIN"
export CUDAHOSTCXX="$NVCC_CCBIN"
export CMAKE_TOOLCHAIN_FILE="$PIXI_PROJECT_ROOT/ck-jobs.cmake"
# nvcc's intermediates for comfy-kitchen's CUTLASS kernels run to gigabytes each, four at a time: more
# than a tmpfs /tmp holds.
export TMPDIR="$PIXI_PROJECT_ROOT/.pixi/tmp"
mkdir -p "$TMPDIR"

# comfy-kitchen is built for this machine's card only. With no card to ask (a docker build), it is
# left unset and comfy-kitchen builds its own default list: every card, and sm_86 runs its sm_80 code.
if [ -z "$COMFY_CUDA_ARCHS" ] && command -v nvidia-smi >/dev/null 2>&1; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '. ')
    [ -n "$cap" ] && export COMFY_CUDA_ARCHS="${cap}-real"
fi
