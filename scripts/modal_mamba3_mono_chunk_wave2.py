"""Modal Wave 2 harness for Mamba3 monolithic CuTe/WMMA chunk work.

Modes:
  * ``cute-check`` builds a bounded overlay with NVIDIA CuTe DSL and
    quack-kernels, then reports import viability.
  * ``cute-gemm`` runs the existing single-GEMM CuTe DSL WGMMA smoke on H200.
  * ``wmma-smoke`` runs the CUDA WMMA fallback correctness and timing probe.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import modal


GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200")
CPPMEGA_ROOT = "/opt/cppmega"
APP_NAME = "cppmega-mamba3-mono-chunk-wave2-" + re.sub(r"[^0-9A-Za-z]+", "-", GPU_SPEC).lower()


def _repo_overlay(image: modal.Image) -> modal.Image:
    return (
        image.add_local_dir("cppmega", f"{CPPMEGA_ROOT}/cppmega", copy=True)
        .add_local_dir("tools", f"{CPPMEGA_ROOT}/tools", copy=True)
        .env(
            {
                "PYTHONPATH": CPPMEGA_ROOT,
                "CPPMEGA_IMAGE_REF": GHCR_REF,
                "CUTE_DSL_ARCH": "sm_90a",
            }
        )
    )


def _base_image() -> modal.Image:
    image: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    return _repo_overlay(image)


def _cute_image() -> modal.Image:
    image: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    image = image.pip_install(
        "nvidia-cutlass-dsl==4.4.2",
        "quack-kernels==0.3.10",
        extra_index_url="https://pypi.nvidia.com",
    )
    return _repo_overlay(image)


app = modal.App(APP_NAME)


def _package_version(name: str) -> str | None:
    import importlib.metadata as md

    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _module_spec(module: str) -> str | None:
    import importlib.util

    try:
        spec = importlib.util.find_spec(module)
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None if spec is None else str(spec.origin)


@app.function(image=_cute_image(), timeout=20 * 60)
def cute_stack_check() -> dict[str, Any]:
    import os
    import sys

    modules = [
        "cutlass",
        "cutlass.cute",
        "cutlass.cute.runtime",
        "cuda.bindings.driver",
        "quack",
        "quack.sm90_utils",
        "quack.copy_utils",
        "quack.layout_utils",
        "torch",
    ]
    specs = {module: _module_spec(module) for module in modules}
    import_error = None
    api_error = None
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack, make_fake_tensor
        from quack import copy_utils, layout_utils, sm90_utils

        _ = (
            cutlass,
            cute,
            from_dlpack,
            make_fake_tensor,
            sm90_utils,
            copy_utils,
            layout_utils,
        )
    except Exception as exc:  # noqa: BLE001
        import_error = f"{type(exc).__name__}: {exc}"
    try:
        import cutlass.cute as cute

        required = ("kernel", "jit", "compile")
        missing = [name for name in required if not hasattr(cute, name)]
        api_error = None if not missing else f"missing cutlass.cute APIs: {missing}"
    except Exception as exc:  # noqa: BLE001
        api_error = f"{type(exc).__name__}: {exc}"

    return {
        "image_ref": GHCR_REF,
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "specs": specs,
        "versions": {
            "nvidia-cutlass-dsl": _package_version("nvidia-cutlass-dsl"),
            "nvidia-cutlass-dsl-libs-base": _package_version("nvidia-cutlass-dsl-libs-base"),
            "quack-kernels": _package_version("quack-kernels"),
            "cuda-python": _package_version("cuda-python"),
            "cuda-bindings": _package_version("cuda-bindings"),
            "torch": _package_version("torch"),
        },
        "import_error": import_error,
        "api_error": api_error,
        "cute_viable": import_error is None and api_error is None,
    }


@app.function(image=_cute_image(), gpu=GPU_SPEC, timeout=30 * 60)
def cute_single_gemm_h200() -> dict[str, Any]:
    import contextlib
    import io
    import os
    import traceback

    os.environ["CUTE_DSL_ARCH"] = "sm_90a"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            from cppmega.megatron.cute_dsl_mimo.single_gemm_test import run_phase1

            passed, per_iter_us = run_phase1()
        error = None
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc(file=stdout)
        passed = False
        per_iter_us = None
        error = f"{type(exc).__name__}: {exc}"
    return {
        "image_ref": GHCR_REF,
        "gpu_spec": GPU_SPEC,
        "passed": bool(passed),
        "per_iter_us": per_iter_us,
        "error": error,
        "output": stdout.getvalue()[-12000:],
    }


@app.function(image=_base_image(), gpu=GPU_SPEC, timeout=30 * 60)
def wmma_smoke_h200(shape: str) -> dict[str, Any]:
    import os
    import shlex
    import subprocess
    import sys

    env = os.environ.copy()
    env.setdefault("MAX_JOBS", "2")
    env.setdefault("CPPMEGA_VERBOSE_EXT_BUILD", "1")
    env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/cppmega_mamba3_mono_chunk_ext")
    cmd = [
        sys.executable,
        "tools/probes/mamba3_mono_chunk_smoke.py",
        *shlex.split(shape),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=CPPMEGA_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
        if len(lines) > 1200:
            lines = lines[-1200:]
    returncode = proc.wait()
    return {
        "image_ref": GHCR_REF,
        "gpu_spec": GPU_SPEC,
        "returncode": returncode,
        "shape": shape,
        "output": "".join(lines)[-16000:],
    }


@app.local_entrypoint()
def main(
    mode: str = "cute-check",
    shape: str = "--B 1 --S 64 --H 4 --P 64 --bench-iters 100 --bench-warmup 20",
) -> None:
    if mode == "cute-check":
        result: Any = cute_stack_check.remote()
    elif mode == "cute-gemm":
        result = cute_single_gemm_h200.remote()
    elif mode == "wmma-smoke":
        result = wmma_smoke_h200.remote(shape)
    elif mode == "all":
        result = {
            "cute_check": cute_stack_check.remote(),
            "cute_gemm": cute_single_gemm_h200.remote(),
            "wmma_smoke": wmma_smoke_h200.remote(shape),
        }
    else:
        raise ValueError("mode must be one of: cute-check, cute-gemm, wmma-smoke, all")
    print(json.dumps(result, indent=2, sort_keys=True))
