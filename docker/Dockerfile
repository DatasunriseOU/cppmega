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

ARG CUDA_IMAGE=nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04@sha256:6435dc5a825b0095648d87a3c91240fd7788a85fafaf215739544d389ab74366
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
        zstd \
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

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu132
ARG TORCH_VERSION=2.13.0+cu132
RUN python -m pip install --extra-index-url ${TORCH_INDEX} \
        "torch==${TORCH_VERSION}" \
        numpy packaging ninja einops pybind11 pyyaml regex \
        sentencepiece tiktoken six scipy filelock typing-extensions \
        cloudpickle psutil pynvml \
        huggingface_hub transformers tokenizers datasets accelerate \
        tensorboard wandb tqdm pytest \
        liger-kernel "z3-solver==4.15.4.0" \
        onnx onnxscript \
        apache-tvm-ffi==0.1.13 \
        nvidia-nccl-cu13

# --- FA4 CuTe DSL runtime (Python-only, JIT-compiled kernels) ---------------
# FA4 is intentionally separate from the flash_attn wheel. The flash-attn-4
# package owns the flash_attn.cute namespace and pins the compatible
# nvidia-cutlass-dsl[cu13] / quack-kernels dependency stack.
RUN python -m pip install --no-cache-dir --pre \
        --extra-index-url https://pypi.nvidia.com \
        "flash-attn-4[cu13]==4.0.0b23"

# --- prebuilt wheels (this layer changes whenever any wheel is rebuilt) -----
COPY wheels/ /tmp/wheels/
RUN python -m pip install --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels

# --- megatron-lm with cppmega patches (editable so runtime patches resolve) -
ARG MEGATRON_REPO=https://github.com/NVIDIA/Megatron-LM.git
ARG MEGATRON_REF=ba7b5ebce12af60627a80985792a1449ce45f46c
RUN set -eu; \
    git init /opt/megatron-lm; \
    git -C /opt/megatron-lm remote add origin "${MEGATRON_REPO}"; \
    git -C /opt/megatron-lm fetch --depth=1 origin "${MEGATRON_REF}"; \
    git -C /opt/megatron-lm checkout --detach FETCH_HEAD; \
    test "$(git -C /opt/megatron-lm rev-parse HEAD)" = "${MEGATRON_REF}"; \
    cd /opt/megatron-lm; \
    python -m pip install -e . --no-build-isolation --no-deps

# --- cppmega itself (source of cppmega.megatron.* runtime patches) ----------
ARG CPPMEGA_SOURCE_SHA
ARG CPPMEGA_SOURCE_TREE
ARG CPPMEGA_SOURCE_MANIFEST_SHA256
ARG CPPMEGA_SOURCE_MANIFEST_FILE_COUNT
COPY . /opt/cppmega
RUN set -eu; \
    printf '%s' "${CPPMEGA_SOURCE_SHA}" | grep -Eq '^[0-9a-f]{40}$'; \
    printf '%s' "${CPPMEGA_SOURCE_TREE}" | grep -Eq '^[0-9a-f]{40}$'; \
    printf '%s' "${CPPMEGA_SOURCE_MANIFEST_SHA256}" | grep -Eq '^[0-9a-f]{64}$'; \
    printf '%s' "${CPPMEGA_SOURCE_MANIFEST_FILE_COUNT}" | grep -Eq '^[1-9][0-9]*$'; \
    printf '{"cppmega_sha":"%s","source_tree":"%s","source_manifest_sha256":"%s","source_manifest_file_count":%s}\n' \
      "${CPPMEGA_SOURCE_SHA}" \
      "${CPPMEGA_SOURCE_TREE}" \
      "${CPPMEGA_SOURCE_MANIFEST_SHA256}" \
      "${CPPMEGA_SOURCE_MANIFEST_FILE_COUNT}" \
      > /opt/cppmega-image-source.json
LABEL org.opencontainers.image.revision="${CPPMEGA_SOURCE_SHA}" \
      io.cppmega.source-tree="${CPPMEGA_SOURCE_TREE}" \
      io.cppmega.source-manifest-sha256="${CPPMEGA_SOURCE_MANIFEST_SHA256}"
RUN python -m pip install -e /opt/cppmega --no-build-isolation --no-deps

# Apply the repository-bound Mamba3 Stage2 safety patch while building the
# immutable image. Runtime gates verify this state and never mutate it.
RUN cd /opt/cppmega \
    && CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1 \
       MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1 \
       python -m \
       cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches \
    && python -c "\
from cppmega.megatron.upstream_patches import apply_mamba3_stage2_force_nontma_patches as stage2; \
assert stage2._is_stage2_patch_applied(); \
assert not stage2._is_stage2_patch_absent()"

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
from importlib import metadata; \
assert metadata.version('apache-tvm-ffi') == '0.1.13.post5'; \
assert metadata.version('flash-attn') == '2.8.3'; \
assert metadata.version('flash-attn-4') == '4.0.0b23'; \
import torch; print('torch', torch.__version__); \
import transformer_engine; print('te', transformer_engine.__version__); \
import transformer_engine.pytorch; \
import flash_attn; \
from flash_attn.cute.interface import flash_attn_func as fa4_flash_attn_func; \
import flash_attn.cute.block_sparsity as _fa4_block_sparsity; \
import flash_attn_3; \
import mamba_ssm, causal_conv1d, fast_hadamard_transform, tilelang, qoptim_cuda; \
from mamba_ssm.modules import mamba3 as _mamba3_module; \
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import mamba_mimo_bwd_combined as _mamba3_mimo_bwd; \
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd_varlen import mamba_mimo_bwd_combined_varlen as _mamba3_mimo_bwd_varlen; \
assert _mamba3_module.mamba3_mimo_combined is not None; \
import cutlass; import cutlass.cute as cute; \
from cutlass.utils import LayoutEnum, SmemAllocator; \
import quack; from quack import sm90_utils, copy_utils, layout_utils, mx_utils; \
print('OK — cutlass', cutlass.__version__ if hasattr(cutlass,'__version__') else '(no __version__)', \
      'fa', flash_attn.__version__, 'fa3', flash_attn_3.__version__ if hasattr(flash_attn_3,'__version__') else '(no __version__)')"

WORKDIR /workspace
CMD ["python"]
