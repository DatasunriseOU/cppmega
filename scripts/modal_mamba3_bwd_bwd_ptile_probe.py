"""Modal Hopper probe for the Mamba3 bwd_bwd P_TILE compile-only prototype.

The harness applies the existing stage2 force-non-TMA patch, then applies the
compile-only P_TILE patch. The new TileLang function intentionally does not
replace production bwd_bwd; it checks whether the pressure-heavy P-scaled
dPhiO/PsiV/dPsiV/dstates chain can be represented as P_TILE=64 structural
tiling.

Run examples:

    python -m py_compile scripts/modal_mamba3_bwd_bwd_ptile_probe.py
    CPPMEGA_MODAL_GPU=H200:2 modal run scripts/modal_mamba3_bwd_bwd_ptile_probe.py
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")

APP_NAME = "cppmega-mamba3-bwd-bwd-ptile-probe"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"
PATCH_BASENAMES = (
    "mamba3_bwd_stage2_force_nontma.patch",
    "mamba3_bwd_bwd_ptile_compile_only.patch",
)


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    img = img.add_local_dir("cppmega", f"{CPPMEGA_ROOT}/cppmega", copy=True)
    img = img.add_local_dir(
        "upstream_prs/examples/13_tilelang_floormod_dbz",
        f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz",
        copy=True,
    )
    img = img.add_local_dir(
        "/home/dave/state-spaces-mamba/mamba_ssm",
        f"{SOURCE_ROOT}/mamba_ssm",
        copy=True,
    )
    return img


app = modal.App(APP_NAME)


def _install_source_paths() -> None:
    import sys

    for path in (CPPMEGA_ROOT, SOURCE_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)


def _reset_mamba_imports() -> None:
    import sys

    for name in list(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]


def _device_report(requested_gpu: str) -> dict[str, Any]:
    import torch

    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "requested_gpu_spec": requested_gpu,
        "image_ref": GHCR_REF,
    }


def _tilelang_report() -> dict[str, Any]:
    import importlib.metadata
    import os
    import subprocess

    import tilelang

    report: dict[str, Any] = {
        "module_file": getattr(tilelang, "__file__", None),
        "module_version": getattr(tilelang, "__version__", None),
    }
    try:
        report["package_version"] = importlib.metadata.version("tilelang")
    except importlib.metadata.PackageNotFoundError:
        report["package_version"] = None

    module_file = report["module_file"]
    if isinstance(module_file, str):
        probe_dir = os.path.dirname(os.path.abspath(module_file))
        proc = subprocess.run(
            ["git", "-C", probe_dir, "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        report["git_head"] = proc.stdout.strip() if proc.returncode == 0 else None
    return report


def _snippet(text: str, needle: str, radius: int = 220) -> str | None:
    index = text.find(needle)
    if index < 0:
        return None
    lo = max(0, index - radius)
    hi = min(len(text), index + len(needle) + radius)
    return text[lo:hi]


def _patched_source_report(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    return {
        "patches": list(PATCH_BASENAMES),
        "disable_tma_count": text.count("disable_tma=True"),
        "ptile_function_count": text.count("mamba_mimo_bwd_bwd_ptile_compile_only"),
        "p_tile_64_default": "p_tile: int = 64" in text,
        "ptile_dPhiO_shared": "dPhiO_shared_tile = T.alloc_shared([fused_chunk_size, P_TILE]" in text,
        "ptile_PsiV_shared": "PsiV_shared_tile = T.alloc_shared([fused_chunk_size, P_TILE]" in text,
        "ptile_dPsiV_frag": "dPsiV_frag_tile = T.alloc_fragment([fused_chunk_size, P_TILE]" in text,
        "ptile_dstates": "dstates_frag_tile = T.alloc_fragment([N, P_TILE]" in text,
        "required_cross_p_accumulator_comment": "cross-P accumulators" in text,
    }


def _prepare_module() -> tuple[str, dict[str, Any]]:
    import shutil
    import subprocess
    import tempfile

    src = f"{SOURCE_ROOT}/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"
    work = tempfile.mkdtemp(prefix="cppmega_mamba3_bwd_bwd_ptile_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src, dst)

    applications: list[dict[str, Any]] = []
    for patch_basename in PATCH_BASENAMES:
        patch_file = (
            f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz/"
            f"{patch_basename}"
        )
        with open(patch_file, "rb") as handle:
            patch_bytes = handle.read()
        proc = subprocess.run(
            ["patch", "-p4", dst],
            input=patch_bytes,
            capture_output=True,
            cwd=work,
            check=False,
        )
        applications.append(
            {
                "patch": patch_basename,
                "rc": proc.returncode,
                "stdout_tail": proc.stdout.decode(errors="replace")[-2000:],
                "stderr_tail": proc.stderr.decode(errors="replace")[-2000:],
            }
        )
        if proc.returncode != 0:
            return dst, {"work": work, "patches": applications}

    return dst, {
        "work": work,
        "patches": applications,
        "patched_source": _patched_source_report(dst),
    }


def _source_markers(source: str, consumer_threads: int) -> dict[str, Any]:
    launch_bounds = sorted(set(re.findall(r"__launch_bounds__\((\d+),\s*(\d+)\)", source)))
    launch_bound_threads = {int(item[0]) for item in launch_bounds}
    producer_guard = f"if ({consumer_threads} <= ((int)threadIdx.x))"
    return {
        "source_chars": len(source),
        "tma_load_count": source.count("tl::tma_load"),
        "tma_store_count": source.count("tl::tma_store"),
        "launch_bounds": launch_bounds,
        "producer_guard": producer_guard in source,
        "expected_ws_launch_bound": any(bound > consumer_threads for bound in launch_bound_threads),
        "barrier_wait_count": source.count("mbarrier_wait"),
        "p_tile_symbol_count": source.count("P_TILE"),
        "producer_guard_snippet": _snippet(source, producer_guard),
    }


def _classify_exception(exc: BaseException, traceback_text: str) -> dict[str, Any]:
    import textwrap

    combined = (str(exc) + "\n" + traceback_text).lower()
    return {
        "exception_type": type(exc).__name__,
        "exception_short": textwrap.shorten(str(exc), width=1000),
        "traceback_tail": traceback_text[-5000:],
        "is_floormod_dbz": (
            "divide by zero" in combined
            and ("floormod" in combined or "layoutinference" in combined or "tryconstfold" in combined)
        ),
        "is_loop_layout_injective": "loop layout is not injective" in combined,
        "is_tma_inputdim": "inputdim() == 2" in combined or "cannot detect tma layout" in combined,
        "is_tma_descriptor_716": "failed to initialize the tma descriptor 716" in combined,
        "is_misaligned_address": "cuda_error_misaligned_address" in combined or "misaligned address" in combined,
        "is_ws_warning": "[ws]" in combined,
    }


def _load_temp_module(path: str, suffix: str) -> Any:
    import importlib.util
    import sys

    name = f"cppmega_mamba3_bwd_bwd_ptile_{suffix}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _compile_one(p_tile: int, bb_stages: int) -> dict[str, Any]:
    import time
    import traceback

    _install_source_paths()
    _reset_mamba_imports()
    import mamba_ssm.ops.tilelang.mamba3  # noqa: F401

    path, prep = _prepare_module()
    result: dict[str, Any] = {
        "p_tile": p_tile,
        "bb_num_stages": bb_stages,
        "prepare": prep,
    }
    if any(item.get("rc") != 0 for item in prep.get("patches", [])):
        result["status"] = "patch_failed"
        return result

    t0 = time.time()
    try:
        mod = _load_temp_module(path, f"compile_{p_tile}_{bb_stages}")
        kernel = mod.mamba_mimo_bwd_bwd_ptile_compile_only(
            1, 64, 4, 1, 64, 128, 4, False, True, True, 16, 4, "bfloat16", 256, bb_stages, p_tile
        )
        source = kernel.get_kernel_source()
        result.update(
            {
                "status": "compiled",
                "bwd_bwd_ptile": _source_markers(source, 256),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["status"] = "crashed"
        result.update(_classify_exception(exc, traceback.format_exc()))
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 3)
    return result


def _compile_matrix() -> list[tuple[int, int]]:
    raw = os.environ.get("CPPMEGA_MAMBA3_PTILE_MATRIX", "64,0")
    matrix: list[tuple[int, int]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        p_tile_raw, bb_raw = item.split(",", maxsplit=1)
        matrix.append((int(p_tile_raw), int(bb_raw)))
    return matrix


@app.function(image=_image(), gpu=GPU_SPEC, timeout=600)
def run_probe(requested_gpu: str) -> dict[str, Any]:
    matrix = _compile_matrix()
    compile_results = [_compile_one(p_tile, bb) for p_tile, bb in matrix]
    return {
        "device": _device_report(requested_gpu),
        "tilelang": _tilelang_report(),
        "app_name": APP_NAME,
        "compile_only": True,
        "correctness_supported": False,
        "reason": (
            "The prototype leaves DK/DQ/DSSDA/DGAMMA/DDA* without required "
            "cross-P accumulator writes, so it is a structural compile probe only."
        ),
        "matrix": matrix,
        "compile": compile_results,
    }


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote(GPU_SPEC)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
