"""Modal base image for cppmega stack — bench3/GB10 parity via prebuilt wheels.

Stack (all cp313 prebuilt, no source builds):
  - torch 2.13.0 stable cu132
  - transformer_engine_torch 2.13.0   (local wheel)
  - transformer_engine_cu13 + meta     (pypi, pulled by TE torch bindings)
  - mamba_ssm 2.3.1 (local wheel — @31f3d7b + bench patches baked in)
  - causal_conv1d 1.6.1 (local wheel)
  - flash_attn 2.8.3 (local wheel)
  - flash-attn-4 4.0.0b23 (NVIDIA PyPI)
  - tilelang 0.1.9 from DatasunriseOU/tilelang@334266af (local abi3 wheel;
    carries the TVM __slots__ fix, restored nvbench CUDA header, and removes
    the apache-tvm-ffi<0.1.10 cap)
  - qoptim_cuda 0.0.0 (local wheel)
  - fast_hadamard_transform 1.1.0 (local wheel)
  - apache-tvm-ffi 0.1.13.post5 (local wheel matching TileLang's linked TVM ABI)
  - megatron-core from commit 980211ae (editable)

Wheels are downloaded once into the repository-owned wheels/ directory (or
CPPMEGA_WHEELS_DIR) and baked into the image via add_local_file(copy=True).
The local compressed inputs therefore survive a host restart; Modal caches the
resulting image layer separately.
"""
# ruff: noqa: E402

from __future__ import annotations

import os
import pathlib
from typing import Any

import modal

PYTHON_VERSION = "3.13"
CUDA_BASE = "nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04"
TORCH_VERSION = "2.13.0+cu132"
TORCH_INDEX = "https://download.pytorch.org/whl/cu132"
MEGATRON_COMMIT = "980211ae"  # bench3's pin (2026-04-09), before output_cross_entropy_loss kwarg

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_WHEELS_DIR = pathlib.Path(
    os.environ.get("CPPMEGA_WHEELS_DIR", _REPO_ROOT / "wheels")
).expanduser().resolve()

_WHEEL_FILES = [
    "transformer_engine_torch-2.13.0-cp313-cp313-linux_x86_64.whl",
    # Pristine wheel includes pure-python tilelang/cute subpackages that
    # plain 2.3.1 wheel drops due to pre-PR-#861 find_packages bug.
    "mamba_ssm-2.3.1+pristine-cp313-cp313-linux_x86_64.whl",
    "causal_conv1d-1.6.1-cp313-cp313-linux_x86_64.whl",
    "flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl",
    "qoptim_cuda-0.0.0-cp313-cp313-linux_x86_64.whl",
    "apache_tvm_ffi-0.1.13.post5-cp313-cp313-linux_x86_64.whl",
    "tilelang-0.1.9-cp38-abi3-linux_x86_64.whl",
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
        # Stable torch 2.13 cu132 + pure-Python deps (no source builds needed).
        .pip_install(
            f"torch=={TORCH_VERSION}",
            "numpy>=1.26", "packaging", "wheel", "setuptools", "ninja",
            "einops", "pybind11", "pyyaml", "regex",
            "sentencepiece", "tiktoken", "six", "scipy",
            extra_index_url=TORCH_INDEX,
        )
        # TE + wheel-pkg declared deps (must be present BEFORE installing
        # our --no-deps wheels, because mamba_ssm/TE import time needs them).
        .pip_install(
            # Bootstrap FA4's dependency; the matching local post5 wheel
            # replaces it in the compressed wheel layer below.
            "apache-tvm-ffi==0.1.13",
            "transformer-engine-cu13==2.13.0",
            "transformer-engine==2.13.0",
            "nvidia-nccl-cu13",
            # TE extra deps
            "onnxscript", "onnx",
            "pydantic", "nvdlfw-inspect",
            # TileLang dev wheel was linked against libz3.so.4.15 specifically.
            # Match the exact ABI used while building the TileLang wheel.
            "z3-solver==4.15.4.0",
            # TileLang dev runtime deps (cloudpickle, psutil, pynvml etc.)
            "cloudpickle", "psutil", "pynvml", "typing-extensions",
            # mamba_ssm / flash_attn import-time deps
            "huggingface_hub", "transformers", "tokenizers",
            # Training ecosystem
            "datasets", "accelerate", "tensorboard",
            "wandb", "tqdm", "pytest", "filelock",
            "liger-kernel",
        )
        .pip_install(
            "flash-attn-4[cu13]==4.0.0b23",
            extra_index_url="https://pypi.nvidia.com",
        )
    )
    # Add durable, compressed local wheel inputs as one image layer. Missing
    # pins fail locally instead of producing a partially populated image.
    wheels_path = _WHEELS_DIR
    missing_wheels = [
        name for name in _WHEEL_FILES if not (wheels_path / name).is_file()
    ]
    if missing_wheels:
        raise FileNotFoundError(
            f"missing pinned wheels below {wheels_path}: {missing_wheels}"
        )
    for whl in _WHEEL_FILES:
        p = wheels_path / whl
        img = img.add_local_file(str(p), f"/wheels/{whl}", copy=True)
    img = img.run_commands(
        "test -f /usr/local/lib/python3.13/site-packages/z3/lib/libz3.so.4.15",
    ).run_commands(
        # Install all local wheels with --no-deps (torch/TE already installed).
        "pip install --no-deps /wheels/*.whl && "
        "python -c 'import transformer_engine.pytorch as te; print(\"TE Linear ok:\", te.Linear)' && "
        "python -c 'from importlib import metadata; "
        "assert metadata.version(\"apache-tvm-ffi\") == \"0.1.13.post5\"; "
        "assert metadata.version(\"flash-attn-4\") == \"4.0.0b23\"; "
        "import mamba_ssm, flash_attn, tilelang; "
        "print(\"mamba_ssm\", mamba_ssm.__version__, "
        "\"flash-attn-4\", metadata.version(\"flash-attn-4\"), "
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
