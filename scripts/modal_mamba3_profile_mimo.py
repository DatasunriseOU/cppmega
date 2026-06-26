"""Modal profiler harness for Mamba3 MIMO baseline vs qk_shared_direct.

The harness is non-production and temp-only:
  * overlays the local cppmega tree and state-spaces/mamba source into Modal,
  * compares upstream TileLang MIMO backward against the Hopper
    qk_shared_direct patch artifact,
  * records CUDA-event latency distributions for bwd_fwd, bwd_bwd, and chain,
  * emits TileLang generated-source metadata and source snapshots,
  * writes JSON, CSV, source, and torch profiler artifacts into a Modal Volume.

Run examples:

    CPPMEGA_MODAL_GPU=H200:2 modal run scripts/modal_mamba3_profile_mimo.py
    CPPMEGA_MODAL_GPU=H100:2 CPPMEGA_MAMBA3_PROFILE_SHAPES=small modal run scripts/modal_mamba3_profile_mimo.py

Retrieve artifacts with the Modal CLI, for example:

    modal volume get cppmega-mamba3-profiles /mamba3_mimo_profile/<run_id> ./mamba3_mimo_profile_<run_id>
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")
PROFILE_VOLUME_NAME = os.environ.get("CPPMEGA_MODAL_PROFILE_VOLUME", "cppmega-mamba3-profiles")

APP_NAME = "cppmega-mamba3-profile-mimo"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"
PROFILE_ROOT = "/profiles"
PROFILE_PREFIX = "mamba3_mimo_profile"

profile_volume = modal.Volume.from_name(PROFILE_VOLUME_NAME, create_if_missing=True)


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


def _tool_report() -> dict[str, Any]:
    import shutil
    import subprocess

    out: dict[str, Any] = {}
    for name in ("ncu", "nsys"):
        path = shutil.which(name)
        info: dict[str, Any] = {"path": path}
        if path:
            proc = subprocess.run([path, "--version"], capture_output=True, text=True, check=False)
            info.update(
                {
                    "version_rc": proc.returncode,
                    "version_stdout_tail": proc.stdout[-1000:],
                    "version_stderr_tail": proc.stderr[-1000:],
                }
            )
        out[name] = info
    out["profiler_used"] = "torch_profiler_nvtx"
    out["ncu_note"] = (
        "This harness records ncu/nsys availability. CUDA-event distributions and "
        "torch profiler traces are always collected; run the saved source/shape metadata "
        "under ncu separately if the Modal image exposes ncu with permission to launch profiled children."
    )
    return out


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


def _shape_catalog() -> dict[str, dict[str, int]]:
    return {
        "small": {"B": 1, "S": 256, "H": 4, "G": 1, "N": 64, "P": 64, "R": 4, "chunk": 16},
        "prodish": {"B": 2, "S": 2048, "H": 16, "G": 1, "N": 64, "P": 64, "R": 4, "chunk": 16},
        "fullprod": {"B": 4, "S": 4096, "H": 32, "G": 1, "N": 128, "P": 128, "R": 4, "chunk": 16},
    }


def _selected_shapes(shape_csv: str) -> list[dict[str, int | str]]:
    names = [name.strip() for name in shape_csv.split(",")]
    catalog = _shape_catalog()
    shapes: list[dict[str, int | str]] = []
    for name in names:
        if not name:
            continue
        if name not in catalog:
            raise ValueError(f"unknown shape {name!r}; choose one of {sorted(catalog)}")
        entry: dict[str, int | str] = {"name": name}
        entry.update(catalog[name])
        shapes.append(entry)
    return shapes


def _prepare_variant(variant: str) -> tuple[str, dict[str, Any]]:
    import shutil
    import subprocess
    import tempfile

    if variant not in {"baseline", "qk_shared_direct"}:
        raise ValueError(f"unknown variant: {variant}")

    src = f"{SOURCE_ROOT}/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"
    work = tempfile.mkdtemp(prefix=f"cppmega_mamba3_profile_{variant}_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src, dst)

    meta: dict[str, Any] = {"variant": variant, "work": work, "source_path": dst, "patch": None}
    if variant == "baseline":
        return dst, meta

    patch_file = (
        f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz/"
        "mamba3_bwd_hopper_tma_ws_fix.patch"
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
    meta.update(
        {
            "patch": patch_file,
            "patch_rc": proc.returncode,
            "patch_stdout_tail": proc.stdout.decode(errors="replace")[-2000:],
            "patch_stderr_tail": proc.stderr.decode(errors="replace")[-2000:],
        }
    )
    return dst, meta


def _import_variant(path: str, variant: str):
    import importlib.util
    import sys
    import time

    name = f"cppmega_mamba3_profile_{variant}_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import variant module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_kernels(mod, shape: dict[str, int | str]) -> tuple[Any, Any]:
    B, S, H, G, N, P, R = (int(shape[k]) for k in ("B", "S", "H", "G", "N", "P", "R"))
    chunk = int(shape["chunk"])
    bf = mod.mamba_mimo_bwd_fwd(B, S, H, G, N, P, R, False, False, True, chunk, 4, "bfloat16", 128, 0)
    bb = mod.mamba_mimo_bwd_bwd(B, S, H, G, N, P, R, False, False, True, chunk, 4, "bfloat16", 256, 0)
    return bf, bb


def _source_meta(kernel: Any, out_path: str) -> dict[str, Any]:
    import hashlib
    import os

    meta: dict[str, Any] = {"has_get_kernel_source": hasattr(kernel, "get_kernel_source")}
    if not hasattr(kernel, "get_kernel_source"):
        return meta
    source = kernel.get_kernel_source()
    if not isinstance(source, str):
        source = str(source)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(source)
    meta.update(
        {
            "artifact": out_path,
            "basename": os.path.basename(out_path),
            "chars": len(source),
            "lines": source.count("\n") + 1,
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "contains_tma": "tma" in source.lower(),
            "contains_qk_shared_direct": "qk_dot_shared" in source and "qk_dot_frag[cs, r_out" not in source,
        }
    )
    return meta


def _stats(values: list[float]) -> dict[str, Any]:
    import math

    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
        return ordered[idx]

    mean = sum(ordered) / len(ordered)
    var = sum((x - mean) ** 2 for x in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean_ms": mean,
        "min_ms": ordered[0],
        "p50_ms": pct(50),
        "p90_ms": pct(90),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": ordered[-1],
        "std_ms": math.sqrt(var),
        "samples_ms": values,
    }


def _time_cuda_events(fn, *, warmup: int, iters: int) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings


def _make_tensors(shape: dict[str, int | str], *, flattened_inputs: bool) -> dict[str, Any]:
    import math

    import torch
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, S, H, G, N, P, R = (int(shape[k]) for k in ("B", "S", "H", "G", "N", "P", "R"))
    chunk = int(shape["chunk"])
    nchunks = math.ceil(S / chunk)

    torch.manual_seed(20260429)
    q = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.01
    k = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.01
    v = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.01
    dout = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.01
    q_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.01
    k_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.01
    mimo_v = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.01
    mimo_o = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.01
    angles = torch.randn(B, S, H, N // 4, device=device, dtype=torch.float32) * 0.01
    dt = torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.01
    trap = torch.randn(B, H, S, device=device, dtype=dtype) * 0.01
    adt = -torch.abs(torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.01)
    da_cs, da_cs_rev, segsum = compute_dacs_segsum_triton(adt, chunk)

    tensors: dict[str, Any] = {
        "dout": dout,
        "q": q.view(B, S * R, G, N) if flattened_inputs else q,
        "k": k.view(B, S * R, G, N) if flattened_inputs else k,
        "v": v,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "mimo_v": mimo_v,
        "mimo_o": mimo_o,
        "z": torch.zeros(B, S, H, P, device=device, dtype=dtype),
        "dz": torch.zeros(B, S, H, P, device=device, dtype=dtype),
        "mimo_z": torch.zeros(H, R, P, device=device, dtype=torch.float32),
        "dmimo_z": torch.zeros(B, H, R, P, device=device, dtype=torch.float32),
        "angles": angles,
        "da_cs": da_cs,
        "da_cs_rev": da_cs_rev,
        "segsum": segsum,
        "dt": dt,
        "trap": trap,
        "d": torch.zeros(H, device=device, dtype=torch.float32),
        "dmimo_o": torch.zeros(B, H, R, P, dtype=torch.float32, device=device),
        "states": torch.zeros(B, H, nchunks, N, P, dtype=dtype, device=device),
        "qk_dot": torch.zeros(B, H, S, R * R, dtype=dtype, device=device)
        if flattened_inputs
        else torch.zeros(B, H, S, R, R, dtype=dtype, device=device),
        "dk": torch.zeros(B, S * R, H, N, dtype=dtype, device=device),
        "dv": torch.zeros(B, S, H, P, dtype=dtype, device=device),
        "dmimo_v": torch.zeros(B, H, R, P, dtype=torch.float32, device=device),
        "dq": torch.zeros(B, S * R, H, N, dtype=dtype, device=device),
        "dfactor": torch.zeros(B, H, S, dtype=torch.float32, device=device),
        "dgamma_diag": torch.zeros(B, H, S, dtype=torch.float32, device=device),
        "dangles": torch.zeros(B, S, H, N // 4, dtype=torch.float32, device=device),
        "dd": torch.zeros(B, H, dtype=torch.float32, device=device),
        "dda": torch.zeros(B, H, S, dtype=torch.float32, device=device),
        "dssda": torch.zeros(B, H, nchunks, chunk, chunk, dtype=torch.float32, device=device),
        "dda_cs_rev": torch.zeros(B, H, S, dtype=torch.float32, device=device),
        "dda_cs": torch.zeros(B, H, S, dtype=torch.float32, device=device),
    }
    return tensors


def _call_bwd_fwd(kernel: Any, t: dict[str, Any]) -> None:
    kernel(
        t["dout"],
        t["q"],
        t["k"],
        t["v"],
        t["q_bias"],
        t["k_bias"],
        t["mimo_v"],
        t["mimo_o"],
        t["dmimo_o"],
        t["states"],
        t["z"],
        t["mimo_z"],
        t["dz"],
        t["dmimo_z"],
        t["angles"],
        t["da_cs"],
        t["da_cs_rev"],
        t["dt"],
        t["trap"],
        t["d"],
        t["qk_dot"],
        t["segsum"],
    )


def _call_bwd_bwd(kernel: Any, t: dict[str, Any]) -> None:
    kernel(
        t["dout"],
        t["q"],
        t["k"],
        t["v"],
        t["q_bias"],
        t["k_bias"],
        t["mimo_v"],
        t["mimo_o"],
        t["dk"],
        t["dv"],
        t["dmimo_v"],
        t["states"],
        t["dq"],
        t["z"],
        t["mimo_z"],
        t["angles"],
        t["da_cs"],
        t["da_cs_rev"],
        t["dt"],
        t["trap"],
        t["dfactor"],
        t["dgamma_diag"],
        t["dangles"],
        t["d"],
        t["dd"],
        t["qk_dot"],
        t["dda"],
        t["dssda"],
        t["dda_cs_rev"],
        t["dda_cs"],
        t["segsum"],
    )


def _output_summary(t: dict[str, Any]) -> dict[str, float]:
    import torch

    torch.cuda.synchronize()
    return {
        "qk_dot_absmax": float(t["qk_dot"].abs().max().item()),
        "dq_absmax": float(t["dq"].abs().max().item()),
        "dk_absmax": float(t["dk"].abs().max().item()),
        "dv_absmax": float(t["dv"].abs().max().item()),
        "dmimo_o_absmax": float(t["dmimo_o"].abs().max().item()),
        "dmimo_v_absmax": float(t["dmimo_v"].abs().max().item()),
        "dfactor_absmax": float(t["dfactor"].abs().max().item()),
    }


def _profile_with_torch_profiler(
    variant: str,
    shape_name: str,
    bf_kernel: Any,
    bb_kernel: Any,
    tensors: dict[str, Any],
    artifact_dir: str,
) -> dict[str, Any]:
    import os
    import traceback

    import torch

    trace_path = os.path.join(artifact_dir, f"{shape_name}_{variant}_torch_trace.json")
    table_path = os.path.join(artifact_dir, f"{shape_name}_{variant}_torch_cuda_table.txt")
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for _ in range(3):
                with torch.profiler.record_function(f"{variant}.bwd_fwd"):
                    torch.cuda.nvtx.range_push(f"{variant}.bwd_fwd")
                    _call_bwd_fwd(bf_kernel, tensors)
                    torch.cuda.nvtx.range_pop()
                with torch.profiler.record_function(f"{variant}.bwd_bwd"):
                    torch.cuda.nvtx.range_push(f"{variant}.bwd_bwd")
                    _call_bwd_bwd(bb_kernel, tensors)
                    torch.cuda.nvtx.range_pop()
                prof.step()
        torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
        with open(table_path, "w", encoding="utf-8") as handle:
            handle.write(table)
        return {"status": "ok", "trace": trace_path, "table": table_path}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback_tail": traceback.format_exc()[-4000:],
        }


def _benchmark_variant(
    variant: str,
    shape: dict[str, int | str],
    run_dir: str,
    *,
    warmup: int,
    iters: int,
    torch_profile: bool,
) -> dict[str, Any]:
    import os
    import time
    import traceback

    import torch

    _install_source_paths()
    _reset_mamba_imports()
    t0 = time.time()
    result: dict[str, Any] = {"variant": variant, "shape": shape}
    try:
        variant_dir = os.path.join(run_dir, str(shape["name"]), variant)
        os.makedirs(variant_dir, exist_ok=True)
        path, prep_meta = _prepare_variant(variant)
        result["prepare"] = prep_meta
        if prep_meta.get("patch_rc", 0) != 0:
            result["status"] = "patch_failed"
            return result

        mod = _import_variant(path, variant)
        bf_kernel, bb_kernel = _make_kernels(mod, shape)
        result["tilelang_source"] = {
            "bwd_fwd": _source_meta(bf_kernel, os.path.join(variant_dir, "bwd_fwd_kernel_source.cu")),
            "bwd_bwd": _source_meta(bb_kernel, os.path.join(variant_dir, "bwd_bwd_kernel_source.cu")),
        }

        tensors = _make_tensors(shape, flattened_inputs=(variant == "qk_shared_direct"))
        _call_bwd_fwd(bf_kernel, tensors)
        _call_bwd_bwd(bb_kernel, tensors)
        torch.cuda.synchronize()

        bf_samples = _time_cuda_events(lambda: _call_bwd_fwd(bf_kernel, tensors), warmup=warmup, iters=iters)
        bb_samples = _time_cuda_events(lambda: _call_bwd_bwd(bb_kernel, tensors), warmup=warmup, iters=iters)
        chain_samples = _time_cuda_events(
            lambda: (_call_bwd_fwd(bf_kernel, tensors), _call_bwd_bwd(bb_kernel, tensors)),
            warmup=warmup,
            iters=iters,
        )
        result.update(
            {
                "status": "ok",
                "elapsed": {
                    "bwd_fwd": _stats(bf_samples),
                    "bwd_bwd": _stats(bb_samples),
                    "chain": _stats(chain_samples),
                },
                "output": _output_summary(tensors),
                "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            }
        )
        if torch_profile:
            result["torch_profiler"] = _profile_with_torch_profiler(
                variant, str(shape["name"]), bf_kernel, bb_kernel, tensors, variant_dir
            )
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "crashed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-6000:],
            }
        )
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 3)
    return result


def _write_distribution_csv(report: dict[str, Any], csv_path: str) -> None:
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["shape", "variant", "phase", "sample_index", "elapsed_ms"],
        )
        writer.writeheader()
        for shape_result in report["profiles"]:
            shape_name = shape_result["shape"]["name"]
            for variant_result in shape_result["variants"]:
                if variant_result.get("status") != "ok":
                    continue
                variant = variant_result["variant"]
                for phase, stats in variant_result["elapsed"].items():
                    for index, sample in enumerate(stats["samples_ms"]):
                        writer.writerow(
                            {
                                "shape": shape_name,
                                "variant": variant,
                                "phase": phase,
                                "sample_index": index,
                                "elapsed_ms": sample,
                            }
                        )


def _compare_variants(shape_result: dict[str, Any]) -> dict[str, Any]:
    variants = {entry["variant"]: entry for entry in shape_result["variants"]}
    base = variants.get("baseline")
    qk = variants.get("qk_shared_direct")
    if not base or not qk or base.get("status") != "ok" or qk.get("status") != "ok":
        return {"status": "missing_ok_variants"}
    out: dict[str, Any] = {"status": "ok", "speedup_qk_over_baseline": {}}
    for phase in ("bwd_fwd", "bwd_bwd", "chain"):
        base_mean = base["elapsed"][phase]["mean_ms"]
        qk_mean = qk["elapsed"][phase]["mean_ms"]
        out["speedup_qk_over_baseline"][phase] = base_mean / qk_mean if qk_mean else None
    return out


@app.function(image=_image(), gpu=GPU_SPEC, timeout=3600, volumes={PROFILE_ROOT: profile_volume})
def run_profile(
    requested_gpu: str,
    run_id: str | None = None,
    shape_csv: str = "small,prodish",
    warmup: int = 5,
    iters: int = 30,
    torch_profile: bool = True,
) -> dict[str, Any]:
    import os
    import time

    import torch

    _install_source_paths()
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_rel = f"{PROFILE_PREFIX}/{run_id}"
    run_dir = os.path.join(PROFILE_ROOT, run_rel)
    os.makedirs(run_dir, exist_ok=True)

    report: dict[str, Any] = {
        "run_id": run_id,
        "volume": PROFILE_VOLUME_NAME,
        "volume_relpath": f"/{run_rel}",
        "artifact_dir": run_dir,
        "device": _device_report(requested_gpu),
        "tilelang": _tilelang_report(),
        "tools": _tool_report(),
        "settings": {
            "shape_csv": shape_csv,
            "warmup": warmup,
            "iters": iters,
            "torch_profile": torch_profile,
        },
        "profiles": [],
    }

    for shape in _selected_shapes(shape_csv):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        shape_result: dict[str, Any] = {"shape": shape, "variants": []}
        for variant in ("baseline", "qk_shared_direct"):
            shape_result["variants"].append(
                _benchmark_variant(
                    variant,
                    shape,
                    run_dir,
                    warmup=warmup,
                    iters=iters,
                    torch_profile=torch_profile,
                )
            )
            torch.cuda.empty_cache()
        shape_result["comparison"] = _compare_variants(shape_result)
        report["profiles"].append(shape_result)

    json_path = os.path.join(run_dir, "report.json")
    csv_path = os.path.join(run_dir, "elapsed_samples.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    _write_distribution_csv(report, csv_path)
    report["artifacts"] = {"report_json": json_path, "elapsed_samples_csv": csv_path}
    profile_volume.commit()
    return report


@app.local_entrypoint()
def main(run_id: str | None = None) -> None:
    shape_csv = os.environ.get("CPPMEGA_MAMBA3_PROFILE_SHAPES", "small,prodish")
    warmup = int(os.environ.get("CPPMEGA_MAMBA3_PROFILE_WARMUP", "5"))
    iters = int(os.environ.get("CPPMEGA_MAMBA3_PROFILE_ITERS", "30"))
    torch_profile = os.environ.get("CPPMEGA_MAMBA3_TORCH_PROFILE", "1") != "0"
    result = run_profile.remote(GPU_SPEC, run_id, shape_csv, warmup, iters, torch_profile)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
