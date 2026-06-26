"""Modal H200 probe for a bounded Mamba3 bwd_bwd structural split.

This harness does not patch production code. It copies upstream
``mamba3_mimo_bwd.py`` into a tempdir, applies the current best
``stage2_force_nontma`` patch, then appends a reduced-output
``mamba_mimo_bwd_bwd_dgamma_diag_probe`` kernel.

The probe isolates the chunk-local dPhiO/PsiV/qk_dot -> DGAMMA_DIAG subgraph
from the reverse-state path. It answers a narrow question: can that subgraph
compile, match the full bwd_bwd DGAMMA_DIAG output, and run cheaply enough to
make a real split worth considering?

Examples:

    python scripts/modal_mamba3_bwd_bwd_split_probe.py --local-dry-run

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 12m \
        modal run scripts/modal_mamba3_bwd_bwd_split_probe.py \
        --shape-csv smoke --iters 5 --warmup 2

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 18m \
        modal run scripts/modal_mamba3_bwd_bwd_split_probe.py \
        --shape-csv productionish --iters 5 --warmup 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")
BENCH_VOLUME_NAME = os.environ.get("CPPMEGA_MODAL_BENCH_VOLUME", "cppmega-mamba3-benchmarks")

APP_NAME = "cppmega-mamba3-bwd-bwd-split-probe"
_DEFAULT_SOURCE_ROOT = "/opt/state-spaces-mamba" if os.path.exists("/opt/state-spaces-mamba") else "/home/dave/state-spaces-mamba"
SOURCE_ROOT = os.environ.get("CPPMEGA_MAMBA3_SOURCE_ROOT", _DEFAULT_SOURCE_ROOT)
CPPMEGA_ROOT = os.environ.get("CPPMEGA_ROOT", "/opt/cppmega")
LOCAL_CPPMEGA_ROOT = os.environ.get("CPPMEGA_LOCAL_ROOT", os.getcwd())
BENCH_ROOT = "/benchmarks"
BENCH_PREFIX = "mamba3_bwd_bwd_split_probe"
PATCH_DIR = "upstream_prs/examples/13_tilelang_floormod_dbz"
STAGE2_PATCH = "mamba3_bwd_stage2_force_nontma.patch"
SPLIT_PATCH = "mamba3_bwd_bwd_split_probe_dgamma.patch"

bench_volume = modal.Volume.from_name(BENCH_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


@dataclass(frozen=True)
class Shape:
    name: str
    B: int
    S: int
    H: int
    G: int
    N: int
    P: int
    R: int
    chunk: int = 16
    rotary_dim_divisor: int = 4


SHAPES: dict[str, Shape] = {
    "tiny": Shape("tiny", B=1, S=64, H=4, G=1, N=64, P=64, R=4),
    "smoke": Shape("smoke", B=1, S=256, H=4, G=1, N=64, P=64, R=4),
    "representative": Shape("representative", B=2, S=1024, H=16, G=1, N=64, P=64, R=4),
    "productionish": Shape("productionish", B=4, S=4096, H=32, G=1, N=64, P=128, R=4),
}


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    img = img.env(
        {
            "GHCR_REPO": GHCR_REPO,
            "GHCR_TAG": GHCR_TAG,
            "CPPMEGA_IMAGE_REF": GHCR_REF,
        }
    )
    img = img.add_local_dir("cppmega", f"{CPPMEGA_ROOT}/cppmega", copy=True)
    img = img.add_local_dir(PATCH_DIR, f"{CPPMEGA_ROOT}/{PATCH_DIR}", copy=True)
    img = img.add_local_dir(
        "/home/dave/state-spaces-mamba/mamba_ssm",
        f"{SOURCE_ROOT}/mamba_ssm",
        copy=True,
    )
    return img


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


def _patch_path(name: str) -> str:
    remote_path = os.path.join(CPPMEGA_ROOT, PATCH_DIR, name)
    if os.path.exists(remote_path):
        return remote_path
    return os.path.join(LOCAL_CPPMEGA_ROOT, PATCH_DIR, name)


def _source_path() -> str:
    remote_path = os.path.join(SOURCE_ROOT, "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py")
    if os.path.exists(remote_path):
        return remote_path
    return "/home/dave/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"


def _apply_patch_file(dst: str, patch_file: str, *, dry_run: bool = False) -> dict[str, Any]:
    import subprocess

    args = ["patch", "-p4"]
    if dry_run:
        args.append("--dry-run")
    args.append(dst)
    with open(patch_file, "rb") as handle:
        patch_bytes = handle.read()
    proc = subprocess.run(
        args,
        input=patch_bytes,
        capture_output=True,
        cwd=os.path.dirname(dst),
        check=False,
    )
    return {
        "patch": os.path.basename(patch_file),
        "dry_run": dry_run,
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.decode(errors="replace")[-2000:],
        "stderr_tail": proc.stderr.decode(errors="replace")[-2000:],
    }


def _patched_source_report(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    split_name = "mamba_mimo_bwd_bwd_dgamma_diag_probe"
    split_body = _snippet_until(text, f"def {split_name}(", "\ndef mamba_mimo_bwd_combined(")
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chars": len(text),
        "disable_tma_count": text.count("disable_tma=True"),
        "stage2_flat_qk_count": text.count("QK_DOT: T.Tensor([B, H, S, R * R]"),
        "split_probe_defs": text.count(f"def {split_name}("),
        "split_probe_kernel_defs": text.count(f"def {split_name}_kernel("),
        "split_probe_uses_dstates": "dstates" in split_body,
        "split_probe_uses_states": "STATES" in split_body,
        "split_probe_uses_q_or_k": bool(re.search(r"\\b[QK]: T\\.Tensor", split_body)),
    }


def _prepare_module() -> tuple[str, dict[str, Any]]:
    import shutil
    import tempfile

    src = _source_path()
    work = tempfile.mkdtemp(prefix="cppmega_mamba3_bwd_bwd_split_probe_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src, dst)

    applied = []
    for patch_name in (STAGE2_PATCH, SPLIT_PATCH):
        item = _apply_patch_file(dst, _patch_path(patch_name), dry_run=False)
        applied.append(item)
        if item["rc"] != 0:
            return dst, {"work": work, "source": src, "patches": applied, "status": "patch_failed"}

    return dst, {
        "work": work,
        "source": src,
        "patches": applied,
        "status": "patched",
        "patched_source": _patched_source_report(dst),
    }


def _snippet(text: str, needle: str, radius: int = 220) -> str:
    index = text.find(needle)
    if index < 0:
        return ""
    lo = max(0, index - radius)
    hi = min(len(text), index + len(needle) + radius)
    return text[lo:hi]


def _snippet_until(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        return ""
    end_index = text.find(end, start_index)
    if end_index < 0:
        end_index = len(text)
    return text[start_index:end_index]


def _load_temp_module(path: str, suffix: str) -> Any:
    import importlib.util
    import sys

    import mamba_ssm.ops.tilelang.mamba3  # noqa: F401

    name = f"cppmega_mamba3_bwd_bwd_split_probe_{suffix}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _source_markers(source: str, consumer_threads: int) -> dict[str, Any]:
    launch_bounds = sorted(set(re.findall(r"__launch_bounds__\\((\\d+),\\s*(\\d+)\\)", source)))
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
    }


def _classify_exception(exc: BaseException, traceback_text: str) -> dict[str, Any]:
    import textwrap

    combined = (str(exc) + "\n" + traceback_text).lower()
    return {
        "exception_type": type(exc).__name__,
        "exception_short": textwrap.shorten(str(exc), width=1200),
        "traceback_tail": traceback_text[-5000:],
        "is_floormod_dbz": "divide by zero" in combined and "floormod" in combined,
        "is_tma_descriptor_716": "failed to initialize the tma descriptor 716" in combined,
        "is_tma_inputdim": "inputdim() == 2" in combined or "cannot detect tma layout" in combined,
        "is_misaligned_address": "misaligned address" in combined,
        "is_ws_warning": "[ws]" in combined,
    }


def _compile_one(shape: Shape) -> dict[str, Any]:
    import time
    import traceback

    _install_source_paths()
    _reset_mamba_imports()
    path, prep = _prepare_module()
    result: dict[str, Any] = {"shape": asdict(shape), "prepare": prep}
    if prep.get("status") != "patched":
        result["status"] = "patch_failed"
        return result

    t0 = time.time()
    try:
        mod = _load_temp_module(path, f"compile_{shape.name}")
        bf_kernel = mod.mamba_mimo_bwd_fwd(
            shape.B, shape.S, shape.H, shape.G, shape.N, shape.P, shape.R,
            False, False, True, shape.chunk, shape.rotary_dim_divisor, "bfloat16", 128, 1,
        )
        bb_kernel = mod.mamba_mimo_bwd_bwd(
            shape.B, shape.S, shape.H, shape.G, shape.N, shape.P, shape.R,
            False, False, True, shape.chunk, shape.rotary_dim_divisor, "bfloat16", 256, 0,
        )
        dg_kernel = mod.mamba_mimo_bwd_bwd_dgamma_diag_probe(
            shape.B, shape.S, shape.H, shape.N, shape.P, shape.R,
            True, shape.chunk, "bfloat16", 128, 0,
        )
        result.update(
            {
                "status": "compiled",
                "bwd_fwd": _source_markers(bf_kernel.get_kernel_source(), 128),
                "bwd_bwd": _source_markers(bb_kernel.get_kernel_source(), 256),
                "dgamma_probe": _source_markers(dg_kernel.get_kernel_source(), 128),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["status"] = "crashed"
        result.update(_classify_exception(exc, traceback.format_exc()))
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 3)
    return result


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
        "image_ref": os.environ.get("CPPMEGA_IMAGE_REF", GHCR_REF),
    }


def _tilelang_report() -> dict[str, Any]:
    import importlib.metadata

    import tilelang

    report: dict[str, Any] = {
        "module_file": getattr(tilelang, "__file__", None),
        "module_version": getattr(tilelang, "__version__", None),
    }
    try:
        report["package_version"] = importlib.metadata.version("tilelang")
    except importlib.metadata.PackageNotFoundError:
        report["package_version"] = None
    return report


def _empty_outputs(shape: Shape) -> dict[str, Any]:
    import math

    import torch

    device = torch.device("cuda")
    dtype = torch.bfloat16
    nchunks = math.ceil(shape.S / shape.chunk)
    return {
        "dmimo_o": torch.zeros(shape.B, shape.H, shape.R, shape.P, dtype=torch.float32, device=device),
        "states": torch.zeros(shape.B, shape.H, nchunks, shape.N, shape.P, dtype=dtype, device=device),
        "qk_dot": torch.zeros(shape.B, shape.H, shape.S, shape.R * shape.R, dtype=dtype, device=device),
        "dk": torch.zeros(shape.B, shape.S * shape.R, shape.H, shape.N, dtype=dtype, device=device),
        "dv": torch.zeros(shape.B, shape.S, shape.H, shape.P, dtype=dtype, device=device),
        "dmimo_v": torch.zeros(shape.B, shape.H, shape.R, shape.P, dtype=torch.float32, device=device),
        "dq": torch.zeros(shape.B, shape.S * shape.R, shape.H, shape.N, dtype=dtype, device=device),
        "dfactor": torch.zeros(shape.B, shape.H, shape.S, dtype=torch.float32, device=device),
        "dgamma_diag": torch.zeros(shape.B, shape.H, shape.S, dtype=torch.float32, device=device),
        "dangles": torch.zeros(shape.B, shape.S, shape.H, shape.N // shape.rotary_dim_divisor, dtype=torch.float32, device=device),
        "dd": torch.zeros(shape.B, shape.H, dtype=torch.float32, device=device),
        "dda": torch.zeros(shape.B, shape.H, shape.S, dtype=torch.float32, device=device),
        "dssda": torch.zeros(shape.B, shape.H, nchunks, shape.chunk, shape.chunk, dtype=torch.float32, device=device),
        "dda_cs_rev": torch.zeros(shape.B, shape.H, shape.S, dtype=torch.float32, device=device),
        "dda_cs": torch.zeros(shape.B, shape.H, shape.S, dtype=torch.float32, device=device),
    }


def _make_inputs(shape: Shape) -> dict[str, Any]:
    import torch
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(123)
    q = torch.randn(shape.B, shape.S, shape.R, shape.G, shape.N, device=device, dtype=dtype) * 0.01
    k = torch.randn(shape.B, shape.S, shape.R, shape.G, shape.N, device=device, dtype=dtype) * 0.01
    adt = -torch.abs(torch.randn(shape.B, shape.H, shape.S, device=device, dtype=torch.float32) * 0.01)
    da_cs, da_cs_rev, segsum = compute_dacs_segsum_triton(adt, shape.chunk)
    return {
        "dout": torch.randn(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype) * 0.01,
        "q_flat": q.view(shape.B, shape.S * shape.R, shape.G, shape.N),
        "k_flat": k.view(shape.B, shape.S * shape.R, shape.G, shape.N),
        "v": torch.randn(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype) * 0.01,
        "q_bias": torch.randn(shape.H, shape.R, shape.N, device=device, dtype=torch.float32) * 0.01,
        "k_bias": torch.randn(shape.H, shape.R, shape.N, device=device, dtype=torch.float32) * 0.01,
        "mimo_v": torch.randn(shape.H, shape.R, shape.P, device=device, dtype=torch.float32) * 0.01,
        "mimo_o": torch.randn(shape.H, shape.R, shape.P, device=device, dtype=torch.float32) * 0.01,
        "angles": torch.randn(
            shape.B, shape.S, shape.H, shape.N // shape.rotary_dim_divisor,
            device=device, dtype=torch.float32,
        ) * 0.01,
        "dt": torch.randn(shape.B, shape.H, shape.S, device=device, dtype=torch.float32) * 0.01,
        "trap": torch.randn(shape.B, shape.H, shape.S, device=device, dtype=dtype) * 0.01,
        "da_cs": da_cs,
        "da_cs_rev": da_cs_rev,
        "segsum": segsum,
        "z": torch.zeros(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype),
        "dz": torch.zeros(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype),
        "mimo_z": torch.zeros(shape.H, shape.R, shape.P, device=device, dtype=torch.float32),
        "dmimo_z": torch.zeros(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32),
        "d": torch.zeros(shape.H, device=device, dtype=torch.float32),
    }


def _call_bwd_fwd(kernel: Any, x: dict[str, Any], o: dict[str, Any]) -> None:
    kernel(
        x["dout"], x["q_flat"], x["k_flat"], x["v"], x["q_bias"], x["k_bias"],
        x["mimo_v"], x["mimo_o"], o["dmimo_o"], o["states"], x["z"],
        x["mimo_z"], x["dz"], x["dmimo_z"], x["angles"], x["da_cs"],
        x["da_cs_rev"], x["dt"], x["trap"], x["d"], o["qk_dot"], x["segsum"],
    )


def _call_bwd_bwd(kernel: Any, x: dict[str, Any], o: dict[str, Any]) -> None:
    kernel(
        x["dout"], x["q_flat"], x["k_flat"], x["v"], x["q_bias"], x["k_bias"],
        x["mimo_v"], x["mimo_o"], o["dk"], o["dv"], o["dmimo_v"],
        o["states"], o["dq"], x["z"], x["mimo_z"], x["angles"], x["da_cs"],
        x["da_cs_rev"], x["dt"], x["trap"], o["dfactor"], o["dgamma_diag"],
        o["dangles"], x["d"], o["dd"], o["qk_dot"], o["dda"], o["dssda"],
        o["dda_cs_rev"], o["dda_cs"], x["segsum"],
    )


def _call_dgamma(kernel: Any, x: dict[str, Any], qk_dot: Any, dgamma_diag: Any) -> None:
    kernel(x["dout"], x["v"], x["mimo_v"], x["mimo_o"], qk_dot, dgamma_diag)


def _time_cuda_events(fn: Any, *, warmup: int, iters: int) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _stats(values: list[float]) -> dict[str, float]:
    import statistics

    if not values:
        return {"mean_ms": float("nan"), "min_ms": float("nan"), "max_ms": float("nan")}
    return {
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _smoke_and_perf_one(shape: Shape, warmup: int, iters: int) -> dict[str, Any]:
    import time
    import traceback

    import torch

    _install_source_paths()
    _reset_mamba_imports()
    path, prep = _prepare_module()
    result: dict[str, Any] = {"shape": asdict(shape), "prepare": prep}
    if prep.get("status") != "patched":
        result["status"] = "patch_failed"
        return result

    t0 = time.time()
    try:
        mod = _load_temp_module(path, f"run_{shape.name}")
        bf_kernel = mod.mamba_mimo_bwd_fwd(
            shape.B, shape.S, shape.H, shape.G, shape.N, shape.P, shape.R,
            False, False, True, shape.chunk, shape.rotary_dim_divisor, "bfloat16", 128, 1,
        )
        bb_kernel = mod.mamba_mimo_bwd_bwd(
            shape.B, shape.S, shape.H, shape.G, shape.N, shape.P, shape.R,
            False, False, True, shape.chunk, shape.rotary_dim_divisor, "bfloat16", 256, 0,
        )
        dg_kernel = mod.mamba_mimo_bwd_bwd_dgamma_diag_probe(
            shape.B, shape.S, shape.H, shape.N, shape.P, shape.R,
            True, shape.chunk, "bfloat16", 128, 0,
        )
        x = _make_inputs(shape)

        full = _empty_outputs(shape)
        _call_bwd_fwd(bf_kernel, x, full)
        _call_bwd_bwd(bb_kernel, x, full)

        dgamma_probe = torch.zeros_like(full["dgamma_diag"])
        _call_dgamma(dg_kernel, x, full["qk_dot"], dgamma_probe)
        torch.cuda.synchronize()
        diff = (dgamma_probe - full["dgamma_diag"]).abs()
        ref_abs = full["dgamma_diag"].abs()

        chain_out = _empty_outputs(shape)
        bwd_bwd_out = _empty_outputs(shape)
        bwd_bwd_out["states"].copy_(full["states"])
        bwd_bwd_out["qk_dot"].copy_(full["qk_dot"])
        dgamma_timed = torch.empty_like(full["dgamma_diag"])

        def run_chain() -> None:
            _call_bwd_fwd(bf_kernel, x, chain_out)
            _call_bwd_bwd(bb_kernel, x, chain_out)

        def run_bwd_bwd() -> None:
            _call_bwd_bwd(bb_kernel, x, bwd_bwd_out)

        def run_dgamma_probe() -> None:
            _call_dgamma(dg_kernel, x, full["qk_dot"], dgamma_timed)

        result.update(
            {
                "status": "ok",
                "correctness": {
                    "dgamma_diag_absmax": float(diff.max().item()),
                    "dgamma_diag_ref_absmax": float(ref_abs.max().item()),
                    "dgamma_diag_allclose_rtol_1e_3_atol_1e_5": bool(
                        torch.allclose(dgamma_probe, full["dgamma_diag"], rtol=1e-3, atol=1e-5)
                    ),
                },
                "elapsed": {
                    "chain": _stats(_time_cuda_events(run_chain, warmup=warmup, iters=iters)),
                    "bwd_bwd": _stats(_time_cuda_events(run_bwd_bwd, warmup=warmup, iters=iters)),
                    "dgamma_probe": _stats(_time_cuda_events(run_dgamma_probe, warmup=warmup, iters=iters)),
                },
                "output_absmax": {
                    "qk_dot": float(full["qk_dot"].abs().max().item()),
                    "dgamma_diag": float(full["dgamma_diag"].abs().max().item()),
                    "dq": float(full["dq"].abs().max().item()),
                    "dk": float(full["dk"].abs().max().item()),
                    "dv": float(full["dv"].abs().max().item()),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["status"] = "crashed"
        result.update(_classify_exception(exc, traceback.format_exc()))
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 3)
    return result


def _shape_list(shape_csv: str) -> list[Shape]:
    names = [item.strip() for item in shape_csv.split(",") if item.strip()]
    unknown = [name for name in names if name not in SHAPES]
    if unknown:
        raise ValueError(f"unknown shapes: {unknown}; valid={sorted(SHAPES)}")
    return [SHAPES[name] for name in names]


def _write_artifacts(result: dict[str, Any]) -> dict[str, str]:
    os.makedirs(BENCH_ROOT, exist_ok=True)
    run_id = result["run_id"]
    json_path = os.path.join(BENCH_ROOT, f"{BENCH_PREFIX}_{run_id}.json")
    csv_path = os.path.join(BENCH_ROOT, f"{BENCH_PREFIX}_{run_id}.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=str)
    rows = []
    for item in result.get("runs", []):
        shape = item.get("shape", {})
        elapsed = item.get("elapsed", {})
        correctness = item.get("correctness", {})
        rows.append(
            {
                "shape": shape.get("name"),
                "status": item.get("status"),
                "chain_mean_ms": elapsed.get("chain", {}).get("mean_ms"),
                "bwd_bwd_mean_ms": elapsed.get("bwd_bwd", {}).get("mean_ms"),
                "dgamma_probe_mean_ms": elapsed.get("dgamma_probe", {}).get("mean_ms"),
                "dgamma_vs_bwd_bwd_pct": (
                    100.0 * elapsed.get("dgamma_probe", {}).get("mean_ms", float("nan"))
                    / elapsed.get("bwd_bwd", {}).get("mean_ms", float("nan"))
                ),
                "dgamma_absmax": correctness.get("dgamma_diag_absmax"),
                "dgamma_allclose": correctness.get("dgamma_diag_allclose_rtol_1e_3_atol_1e_5"),
            }
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["shape", "status"])
        writer.writeheader()
        writer.writerows(rows)
    return {"json": json_path, "csv": csv_path}


@app.function(image=_image(), gpu=GPU_SPEC, timeout=1200, volumes={BENCH_ROOT: bench_volume})
def run_probe(requested_gpu: str, shape_csv: str, warmup: int, iters: int) -> dict[str, Any]:
    import time

    shapes = _shape_list(shape_csv)
    result: dict[str, Any] = {
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "app_name": APP_NAME,
        "device": _device_report(requested_gpu),
        "tilelang": _tilelang_report(),
        "patches": [STAGE2_PATCH, SPLIT_PATCH],
        "warmup": warmup,
        "iters": iters,
        "compile": [_compile_one(SHAPES["tiny"])],
        "runs": [_smoke_and_perf_one(shape, warmup, iters) for shape in shapes],
    }
    result["artifacts"] = _write_artifacts(result)
    bench_volume.commit()
    return result


def local_dry_run() -> dict[str, Any]:
    import shutil
    import tempfile

    src = _source_path()
    work = tempfile.mkdtemp(prefix="cppmega_mamba3_bwd_bwd_split_probe_dryrun_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src, dst)
    patches = []
    for patch_name in (STAGE2_PATCH, SPLIT_PATCH):
        patch_file = _patch_path(patch_name)
        dry = _apply_patch_file(dst, patch_file, dry_run=True)
        real = _apply_patch_file(dst, patch_file, dry_run=False) if dry["rc"] == 0 else None
        patches.append({"dry_run": dry, "apply": real})
        if dry["rc"] != 0 or real is None or real["rc"] != 0:
            break
    last_apply = patches[-1].get("apply") if patches else None
    return {
        "source": src,
        "work": work,
        "patches": patches,
        "patched_source": _patched_source_report(dst) if last_apply and last_apply.get("rc") == 0 else None,
    }


def _print_summary(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    for item in result.get("runs", []):
        elapsed = item.get("elapsed", {})
        corr = item.get("correctness", {})
        shape = item.get("shape", {}).get("name")
        print(
            "SUMMARY "
            f"shape={shape} status={item.get('status')} "
            f"chain_ms={elapsed.get('chain', {}).get('mean_ms')} "
            f"bwd_bwd_ms={elapsed.get('bwd_bwd', {}).get('mean_ms')} "
            f"dgamma_probe_ms={elapsed.get('dgamma_probe', {}).get('mean_ms')} "
            f"dgamma_absmax={corr.get('dgamma_diag_absmax')} "
            f"allclose={corr.get('dgamma_diag_allclose_rtol_1e_3_atol_1e_5')}"
        )


@app.local_entrypoint()
def main(
    shape_csv: str = "smoke",
    warmup: int = 2,
    iters: int = 5,
) -> None:
    result = run_probe.remote(GPU_SPEC, shape_csv, warmup, iters)
    _print_summary(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dry-run", action="store_true")
    args = parser.parse_args()
    if not args.local_dry_run:
        raise SystemExit("Use `modal run ...` for GPU probing or pass --local-dry-run.")
    _print_summary(local_dry_run())
