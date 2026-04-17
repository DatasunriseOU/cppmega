"""Modal base image for cppmega stack — bench3/GB10 parity via prebuilt wheels.

Stack (all cp313 prebuilt, no source builds):
  - torch 2.12.* nightly cu132 (from pytorch nightly index)
  - transformer_engine_torch 2.13.0   (local wheel)
  - transformer_engine_cu13 + meta     (pypi, pulled by TE torch bindings)
  - mamba_ssm 2.3.1 (local wheel — @31f3d7b + bench patches baked in)
  - causal_conv1d 1.6.1 (local wheel)
  - flash_attn 2.8.3 (local wheel)
  - tilelang 0.1.8+cuda.gitf309d814 (local wheel, abi3)
  - qoptim_cuda 0.0.0 (local wheel)
  - fast_hadamard_transform 1.1.0 (local wheel)
  - apache-tvm-ffi 0.1.9 (pypi, <0.1.10 for TileLang)
  - megatron-core 0.18 from origin/dev HEAD (editable)

Wheels are downloaded once from HETZ_1_IP:/data/gs-* to
/tmp/cppmega_wheels/ and baked into the image via add_local_dir(copy=True).
Total: ~700 MB, caches once in Modal's layer store, reused forever.
"""
# ruff: noqa: E402

from __future__ import annotations

import pathlib
from typing import Any

import modal

PYTHON_VERSION = "3.13"
CUDA_BASE = "nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04"
TORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu132"
MEGATRON_COMMIT = "980211ae"  # bench3's pin (2026-04-09), before output_cross_entropy_loss kwarg

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_WHEELS_DIR = "/tmp/cppmega_wheels"

_WHEEL_FILES = [
    "transformer_engine_torch-2.13.0-cp313-cp313-linux_x86_64.whl",
    # Pristine wheel includes pure-python tilelang/cute subpackages that
    # plain 2.3.1 wheel drops due to pre-PR-#861 find_packages bug.
    "mamba_ssm-2.3.1+pristine-cp313-cp313-linux_x86_64.whl",
    "causal_conv1d-1.6.1-cp313-cp313-linux_x86_64.whl",
    "flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl",
    "qoptim_cuda-0.0.0-cp313-cp313-linux_x86_64.whl",
    "tilelang-0.1.8+cuda.gitf309d814-cp38-abi3-linux_x86_64.whl",
    "fast_hadamard_transform-1.1.0-cp313-cp313-linux_x86_64.whl",
]


def cppmega_base_image() -> modal.Image:
    base: Any = modal.Image.from_registry(CUDA_BASE, add_python=PYTHON_VERSION)
    img = (
        base.apt_install(
            "git", "build-essential", "curl", "ca-certificates",
            "pkg-config", "libnuma-dev",
        )
        .env({
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/sbin",
            # TileLang dev needs libz3.so.4.15 from z3-solver pip package.
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.13/site-packages/z3/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "PYTHONPATH": "/opt/megatron-lm",
        })
        # torch 2.12 nightly cu132 + pure-Python deps (no source builds needed).
        .pip_install(
            "torch==2.12.*",
            "numpy>=1.26", "packaging", "wheel", "setuptools", "ninja",
            "einops", "pybind11", "pyyaml", "regex",
            "sentencepiece", "tiktoken", "six", "scipy",
            extra_index_url=TORCH_NIGHTLY_INDEX,
            pre=True,
        )
        # TE + wheel-pkg declared deps (must be present BEFORE installing
        # our --no-deps wheels, because mamba_ssm/TE import time needs them).
        .pip_install(
            "apache-tvm-ffi==0.1.9",
            "transformer-engine-cu13==2.13.0",
            "transformer-engine==2.13.0",
            "nvidia-nccl-cu13",
            # TE extra deps
            "onnxscript", "onnx",
            "pydantic", "nvdlfw-inspect",
            # TileLang dev wheel was linked against libz3.so.4.15 specifically.
            # Pin to 4.15.x (latest release branch that ships .4.15 file).
            "z3-solver==4.15.*",
            # TileLang dev runtime deps (cloudpickle, psutil, pynvml etc.)
            "cloudpickle", "psutil", "pynvml", "typing-extensions",
            # mamba_ssm / flash_attn import-time deps
            "huggingface_hub", "transformers", "tokenizers",
            # Training ecosystem
            "datasets", "accelerate", "tensorboard",
            "wandb", "tqdm", "pytest", "filelock",
            "liger-kernel",
        )
    )
    # Add wheels from local scp'd /tmp/cppmega_wheels/ as a single big layer.
    wheels_path = pathlib.Path(_WHEELS_DIR)
    for whl in _WHEEL_FILES:
        p = wheels_path / whl
        if p.exists():
            img = img.add_local_file(str(p), f"/wheels/{whl}", copy=True)
    img = img.run_commands(
        # Ensure libz3.so.4.15 symlink exists even if z3-solver ships a
        # slightly different minor version.
        "Z3LIB=/usr/local/lib/python3.13/site-packages/z3/lib && "
        "ls $Z3LIB && "
        "if [ ! -f $Z3LIB/libz3.so.4.15 ]; then "
        "  ln -sf $Z3LIB/libz3.so $Z3LIB/libz3.so.4.15; fi && "
        "ls $Z3LIB/libz3*",
    ).run_commands(
        # Install all local wheels with --no-deps (torch/TE already installed).
        "pip install --no-deps /wheels/*.whl && "
        "python -c 'import transformer_engine.pytorch as te; print(\"TE Linear ok:\", te.Linear)' && "
        "python -c 'import mamba_ssm, flash_attn, tilelang; "
        "print(\"mamba_ssm\", mamba_ssm.__version__, "
        "\"flash_attn\", flash_attn.__version__, "
        "\"tilelang\", tilelang.__version__)' && "
        "python -c 'from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo; print(\"mamba3_mimo ok\")'",
    ).run_commands(
        # Megatron-core from dev branch (editable).
        f"cd /opt && git clone https://github.com/NVIDIA/Megatron-LM.git megatron-lm && "
        f"cd megatron-lm && git checkout -q {MEGATRON_COMMIT} && "
        "pip install --no-deps -e .",
        "python -c 'import megatron.core; print(\"megatron-core\", megatron.core.__version__)'",
    )
    return img


# ---------------------------------------------------------------------------
# Deploy-able app — pins base image in Modal's cache.
# ---------------------------------------------------------------------------

app = modal.App("cppmega-base")
base_image = cppmega_base_image()


@app.function(image=base_image, timeout=3600)
def pin() -> str:
    return "pinned"


@app.local_entrypoint()
def build():
    print(f"cppmega-base: {pin.remote()}")
