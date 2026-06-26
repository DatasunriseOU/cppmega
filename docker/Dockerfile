# syntax=docker/dockerfile:1.7
# cppmega runtime image.
#
# Wheels are built upstream by .github/workflows/build-wheels.yml and uploaded
# as GH Release assets. This Dockerfile assumes a sibling `wheels/` directory
# in the build context populated from that release; the image-build workflow
# (.github/workflows/build-image.yml) does that download before `docker build`.
#
# Layers are ordered cheapest → most-volatile so buildx registry cache reuses
# everything below the changed wheel. Push/pull cache via:
#   --cache-from=type=registry,ref=ghcr.io/datasunriseou/cppmega:buildcache
#   --cache-to=type=registry,ref=ghcr.io/datasunriseou/cppmega:buildcache,mode=max

ARG CUDA_IMAGE=nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

# --- system tooling: gcc-15 + python 3.13 -----------------------------------
# Skip the system python3-pip apt package — it installs pip without a RECORD
# file, so 'pip install --upgrade pip' later fails with 'Cannot uninstall'.
# Use ensurepip on python3.13 directly instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg curl wget \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        gcc-15 g++-15 \
        python3.13 python3.13-dev python3.13-venv \
        git build-essential cmake ninja-build pkg-config libnuma-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 90 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-15 90 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 90 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 90 \
    && python3.13 -m ensurepip --upgrade \
    && rm -rf /var/lib/apt/lists/*

# nvcc must be told to accept gcc-15 (CUDA 13.2 officially supports up to gcc-14).
ENV NVCC_PREPEND_FLAGS="--compiler-bindir=/usr/bin/g++-15 --allow-unsupported-compiler"

# --- python deps from PyPI (stable, slow-changing layer) --------------------
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

ARG TORCH_INDEX=https://download.pytorch.org/whl/nightly/cu132
RUN python -m pip install --pre --extra-index-url ${TORCH_INDEX} \
        "torch==2.13.*" \
        numpy packaging ninja einops pybind11 pyyaml regex \
        sentencepiece tiktoken six scipy filelock typing-extensions \
        cloudpickle psutil pynvml \
        huggingface_hub transformers tokenizers datasets accelerate \
        tensorboard wandb tqdm pytest \
        liger-kernel z3-solver \
        onnx onnxscript \
        apache-tvm-ffi==0.1.9 \
        nvidia-nccl-cu13

# --- FA4 CuTe DSL runtime (Python-only, JIT-compiled kernels) ---------------
# FA4 is intentionally separate from the flash_attn wheel. The flash-attn-4
# package owns the flash_attn.cute namespace and pins the compatible
# nvidia-cutlass-dsl[cu13] / quack-kernels dependency stack.
RUN python -m pip install --no-cache-dir --pre \
        --extra-index-url https://pypi.nvidia.com \
        "flash-attn-4[cu13]==4.0.0b19"

# --- prebuilt wheels (this layer changes whenever any wheel is rebuilt) -----
COPY wheels/ /tmp/wheels/
RUN python -m pip install --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels

# --- megatron-lm with cppmega patches (editable so runtime patches resolve) -
ARG MEGATRON_REPO=https://github.com/NVIDIA/Megatron-LM.git
ARG MEGATRON_REF=core_v0.16.0
RUN set -eu; \
    git clone --depth=1 --branch "${MEGATRON_REF}" "${MEGATRON_REPO}" /opt/megatron-lm; \
    cd /opt/megatron-lm; \
    python -m pip install -e . --no-build-isolation --no-deps

# --- cppmega itself (source of cppmega.megatron.* runtime patches) ----------
COPY . /opt/cppmega
RUN python -m pip install -e /opt/cppmega --no-build-isolation --no-deps

# --- runtime env (mirrors modal_cppmega_base.py) ----------------------------
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/sbin \
    LD_LIBRARY_PATH=/usr/local/lib/python3.13/dist-packages/z3/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat \
    TORCH_CUDA_ARCH_LIST="9.0;10.0;12.1" \
    PYTHONPATH=/opt/megatron-lm:/opt/cppmega

# Smoke-import to fail fast on ABI mismatches at build time.
# Note: flash_attn ships FA2 kernels, flash_attn_3 is the separate FA3 Hopper
# package, and flash-attn-4 owns flash_attn.cute for CuTe DSL kernels.
RUN python -c "\
import torch; print('torch', torch.__version__); \
import transformer_engine; print('te', transformer_engine.__version__); \
import transformer_engine.pytorch; \
import flash_attn; \
from flash_attn.cute.interface import flash_attn_func as fa4_flash_attn_func; \
import flash_attn.cute.block_sparsity as _fa4_block_sparsity; \
import flash_attn_3; \
import mamba_ssm, causal_conv1d, fast_hadamard_transform, tilelang, qoptim_cuda; \
import cutlass; import cutlass.cute as cute; \
from cutlass.utils import LayoutEnum, SmemAllocator; \
import quack; from quack import sm90_utils, copy_utils, layout_utils, mx_utils; \
print('OK — cutlass', cutlass.__version__ if hasattr(cutlass,'__version__') else '(no __version__)', \
      'fa', flash_attn.__version__, 'fa3', flash_attn_3.__version__ if hasattr(flash_attn_3,'__version__') else '(no __version__)')"

WORKDIR /workspace
CMD ["python"]
