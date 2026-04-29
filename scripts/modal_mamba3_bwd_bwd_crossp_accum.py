"""Modal H200 probe for a Mamba3 bwd_bwd P_TILE cross-P accumulator prototype.

The script applies the existing stage2 force-nonTMA patch to a temporary copy of
state-spaces/mamba, appends a TileLang `mamba_mimo_bwd_bwd_ptile_crossp_accum`
kernel, and compares it with the stage2 `bf=1, bb=0` baseline where that
baseline compiles.

Run examples:

    python -m py_compile scripts/modal_mamba3_bwd_bwd_crossp_accum.py
    CPPMEGA_MODAL_GPU=H200:2 timeout 20m modal run \
        scripts/modal_mamba3_bwd_bwd_crossp_accum.py --shape-csv smoke --iters 2 --warmup 1
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")

APP_NAME = "cppmega-mamba3-bwd-bwd-crossp-accum"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"


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
    "tiny": Shape("tiny", B=1, S=64, H=2, G=1, N=64, P=64, R=4),
    "smoke": Shape("smoke", B=1, S=256, H=4, G=1, N=64, P=64, R=4),
    "smoke_p128": Shape("smoke_p128", B=1, S=256, H=4, G=1, N=64, P=128, R=4),
    "productionish": Shape("productionish", B=4, S=4096, H=32, G=1, N=64, P=128, R=4),
}


PATCH_BASENAMES = (
    "mamba3_bwd_stage2_force_nontma.patch",
    "mamba3_bwd_bwd_crossp_accum_prototype.patch",
)


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    img = img.env({"CPPMEGA_IMAGE_REF": GHCR_REF})
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
        "image_ref": os.environ.get("CPPMEGA_IMAGE_REF", GHCR_REF),
    }


def _prepare_module() -> tuple[str, dict[str, Any]]:
    import shutil
    import subprocess
    import tempfile

    src = f"{SOURCE_ROOT}/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"
    work = tempfile.mkdtemp(prefix="cppmega_mamba3_crossp_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src, dst)

    applications: list[dict[str, Any]] = []
    for patch_basename in PATCH_BASENAMES:
        patch_file = f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz/{patch_basename}"
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
                "patch_file": patch_file,
                "patch_rc": proc.returncode,
                "patch_stdout_tail": proc.stdout.decode(errors="replace")[-2000:],
                "patch_stderr_tail": proc.stderr.decode(errors="replace")[-2000:],
            }
        )
        if proc.returncode != 0:
            break

    with open(dst, "r", encoding="utf-8") as handle:
        patched_text = handle.read()
    return dst, {
        "work": work,
        "patches": applications,
        "patch_rc": applications[-1]["patch_rc"] if applications else 1,
        "crossp_function_count": patched_text.count("mamba_mimo_bwd_bwd_ptile_crossp_accum"),
    }


def _import_module(path: str, suffix: str) -> Any:
    import importlib.util
    import sys
    import time

    name = f"cppmega_mamba3_crossp_{suffix}_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(shape: Shape) -> dict[str, Any]:
    import torch
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(20260429)
    q = torch.randn(shape.B, shape.S, shape.R, shape.G, shape.N, device=device, dtype=dtype) * 0.01
    k = torch.randn(shape.B, shape.S, shape.R, shape.G, shape.N, device=device, dtype=dtype) * 0.01
    v = torch.randn(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype) * 0.01
    dout = torch.randn(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype) * 0.01
    q_bias = torch.randn(shape.H, shape.R, shape.N, device=device, dtype=torch.float32) * 0.01
    k_bias = torch.randn(shape.H, shape.R, shape.N, device=device, dtype=torch.float32) * 0.01
    mimo_v = torch.randn(shape.H, shape.R, shape.P, device=device, dtype=torch.float32) * 0.01
    mimo_o = torch.randn(shape.H, shape.R, shape.P, device=device, dtype=torch.float32) * 0.01
    angles = torch.randn(
        shape.B,
        shape.S,
        shape.H,
        shape.N // shape.rotary_dim_divisor,
        device=device,
        dtype=torch.float32,
    ) * 0.01
    dt = torch.randn(shape.B, shape.H, shape.S, device=device, dtype=torch.float32) * 0.01
    trap = torch.randn(shape.B, shape.H, shape.S, device=device, dtype=dtype) * 0.01
    adt = -torch.abs(torch.randn(shape.B, shape.H, shape.S, device=device, dtype=torch.float32) * 0.01)
    da_cs, da_cs_rev, segsum = compute_dacs_segsum_triton(adt, shape.chunk)
    return {
        "q": q,
        "k": k,
        "q_flat": q.view(shape.B, shape.S * shape.R, shape.G, shape.N),
        "k_flat": k.view(shape.B, shape.S * shape.R, shape.G, shape.N),
        "v": v,
        "dout": dout,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "mimo_v": mimo_v,
        "mimo_o": mimo_o,
        "angles": angles,
        "da_cs": da_cs,
        "da_cs_rev": da_cs_rev,
        "dt": dt,
        "trap": trap,
        "d": torch.zeros(shape.H, device=device, dtype=torch.float32),
        "segsum": segsum,
    }


def _empty_outputs(shape: Shape) -> dict[str, Any]:
    import torch

    device = torch.device("cuda")
    dtype = torch.bfloat16
    nchunks = math.ceil(shape.S / shape.chunk)
    return {
        "z": torch.zeros(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype),
        "dz": torch.zeros(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype),
        "mimo_z": torch.zeros(shape.H, shape.R, shape.P, device=device, dtype=torch.float32),
        "dmimo_z": torch.zeros(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32),
        "dmimo_o": torch.zeros(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32),
        "states": torch.zeros(shape.B, shape.H, nchunks, shape.N, shape.P, device=device, dtype=dtype),
        "qk_dot": torch.zeros(shape.B, shape.H, shape.S, shape.R * shape.R, device=device, dtype=dtype),
        "dk": torch.zeros(shape.B, shape.S * shape.R, shape.H, shape.N, device=device, dtype=dtype),
        "dv": torch.zeros(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype),
        "dmimo_v": torch.zeros(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32),
        "dq": torch.zeros(shape.B, shape.S * shape.R, shape.H, shape.N, device=device, dtype=dtype),
        "dfactor": torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32),
        "dgamma_diag": torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32),
        "dangles": torch.zeros(
            shape.B,
            shape.S,
            shape.H,
            shape.N // shape.rotary_dim_divisor,
            device=device,
            dtype=torch.float32,
        ),
        "dd": torch.zeros(shape.B, shape.H, device=device, dtype=torch.float32),
        "dda": torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32),
        "dssda": torch.zeros(
            shape.B,
            shape.H,
            nchunks,
            shape.chunk,
            shape.chunk,
            device=device,
            dtype=torch.float32,
        ),
        "dda_cs_rev": torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32),
        "dda_cs": torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32),
    }


def _bf_args(inputs: dict[str, Any], outputs: dict[str, Any]) -> tuple[Any, ...]:
    return (
        inputs["dout"],
        inputs["q_flat"],
        inputs["k_flat"],
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        inputs["mimo_o"],
        outputs["dmimo_o"],
        outputs["states"],
        outputs["z"],
        outputs["mimo_z"],
        outputs["dz"],
        outputs["dmimo_z"],
        inputs["angles"],
        inputs["da_cs"],
        inputs["da_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        inputs["d"],
        outputs["qk_dot"],
        inputs["segsum"],
    )


def _bb_args(inputs: dict[str, Any], outputs: dict[str, Any]) -> tuple[Any, ...]:
    return (
        inputs["dout"],
        inputs["q_flat"],
        inputs["k_flat"],
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        inputs["mimo_o"],
        outputs["dk"],
        outputs["dv"],
        outputs["dmimo_v"],
        outputs["states"],
        outputs["dq"],
        outputs["z"],
        outputs["mimo_z"],
        inputs["angles"],
        inputs["da_cs"],
        inputs["da_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        outputs["dfactor"],
        outputs["dgamma_diag"],
        outputs["dangles"],
        inputs["d"],
        outputs["dd"],
        outputs["qk_dot"],
        outputs["dda"],
        outputs["dssda"],
        outputs["dda_cs_rev"],
        outputs["dda_cs"],
        inputs["segsum"],
    )


def _compare_outputs(shape: Shape, ref: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
    import torch

    names = [
        "dk",
        "dv",
        "dmimo_v",
        "dq",
        "dfactor",
        "dgamma_diag",
        "dangles",
        "dd",
        "dda",
        "dssda",
        "dda_cs_rev",
        "dda_cs",
    ]
    diffs: dict[str, Any] = {}
    for name in names:
        diff = (ref[name].float() - got[name].float()).abs()
        ref_abs = ref[name].float().abs()
        max_abs = float(diff.max().item())
        ref_max = float(ref_abs.max().item())
        diffs[name] = {
            "max_abs": max_abs,
            "ref_absmax": ref_max,
            "rel_to_ref_absmax": max_abs / max(ref_max, 1.0e-12),
            "allclose_1e_2": bool(torch.allclose(ref[name].float(), got[name].float(), rtol=1.0e-2, atol=1.0e-2)),
        }
    diffs["allclose_count"] = sum(1 for name in names if diffs[name]["allclose_1e_2"])
    diffs["allclose_total"] = len(names)
    return diffs


def _time_cuda_events(fn: Any, *, warmup: int, iters: int) -> dict[str, Any]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "count": len(samples),
        "mean_ms": sum(samples) / len(samples) if samples else None,
        "min_ms": min(samples) if samples else None,
        "max_ms": max(samples) if samples else None,
        "samples_ms": samples,
    }


def _source_markers(kernel: Any, consumer_threads: int) -> dict[str, Any]:
    source = kernel.get_kernel_source()
    launch_bounds = sorted(set(re.findall(r"__launch_bounds__\((\d+),\s*(\d+)\)", source)))
    launch_bound_threads = {int(item[0]) for item in launch_bounds}
    producer_guard = f"if ({consumer_threads} <= ((int)threadIdx.x))"
    return {
        "source_chars": len(source),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "tma_load_count": source.count("tl::tma_load"),
        "tma_store_count": source.count("tl::tma_store"),
        "mbarrier_wait_count": source.count("mbarrier_wait"),
        "launch_bounds": launch_bounds,
        "producer_guard": producer_guard in source,
        "expected_ws_launch_bound": any(bound > consumer_threads for bound in launch_bound_threads),
        "has_crossp_name": "ptile_crossp_accum" in source,
        "has_scratch": "DSTATES_PTILE" in source,
        "dynamic_scratch_tma_guarded": "DSTATES_PTILE[i_b, i_h, p_block, :, :], dstates_shared_tile, disable_tma=True" in source,
    }


def _run_shape(shape: Shape, warmup: int, iters: int, p_tile: int, crossp_num_stages: int) -> dict[str, Any]:
    import time
    import traceback

    import torch

    print(f"[crossp] start shape={shape.name}", flush=True)
    t0 = time.time()
    path, prep = _prepare_module()
    print(f"[crossp] prepared shape={shape.name} patch_rc={prep['patch_rc']}", flush=True)
    result: dict[str, Any] = {
        "shape": asdict(shape),
        "prepare": prep,
        "p_tile": p_tile,
        "crossp_num_stages": crossp_num_stages,
    }
    if prep["patch_rc"] != 0:
        result["status"] = "patch_failed"
        return result
    try:
        mod = _import_module(path, shape.name)
        print(f"[crossp] imported shape={shape.name}", flush=True)
        inputs = _make_inputs(shape)
        print(f"[crossp] inputs shape={shape.name}", flush=True)

        common = (
            shape.B,
            shape.S,
            shape.H,
            shape.G,
            shape.N,
            shape.P,
            shape.R,
            False,
            False,
            True,
            shape.chunk,
            shape.rotary_dim_divisor,
            "bfloat16",
        )
        bf_kernel = mod.mamba_mimo_bwd_fwd(*common, 128, 1)
        print(f"[crossp] compiled bf shape={shape.name}", flush=True)
        baseline_bb_status: dict[str, Any] = {"status": "not_run"}
        baseline_outputs: dict[str, Any] | None = None
        try:
            baseline_bb = mod.mamba_mimo_bwd_bwd(*common, 256, 0)
            print(f"[crossp] compiled baseline bb shape={shape.name}", flush=True)
            baseline_outputs = _empty_outputs(shape)
            bf_kernel(*_bf_args(inputs, baseline_outputs))
            baseline_bb(*_bb_args(inputs, baseline_outputs))
            torch.cuda.synchronize()
            baseline_bb_status = {
                "status": "ok",
                "source": _source_markers(baseline_bb, 256),
                "elapsed": _time_cuda_events(lambda: baseline_bb(*_bb_args(inputs, baseline_outputs)), warmup=warmup, iters=iters),
            }
        except Exception as exc:  # noqa: BLE001
            baseline_bb_status = {
                "status": "crashed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-3000:],
            }

        print(f"[crossp] compiling crossp bb shape={shape.name}", flush=True)
        crossp_kernel = mod.mamba_mimo_bwd_bwd_ptile_crossp_accum(*common, 256, crossp_num_stages, p_tile)
        print(f"[crossp] compiled crossp bb shape={shape.name}", flush=True)
        candidate_outputs = _empty_outputs(shape)
        scratch = torch.empty(
            shape.B,
            shape.H,
            shape.P // p_tile,
            shape.N,
            p_tile,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        bf_kernel(*_bf_args(inputs, candidate_outputs))
        print(f"[crossp] ran bf for candidate shape={shape.name}", flush=True)
        crossp_kernel(*_bb_args(inputs, candidate_outputs), scratch)
        torch.cuda.synchronize()
        print(f"[crossp] ran crossp shape={shape.name}", flush=True)

        comparison = None
        if baseline_outputs is not None:
            comparison = _compare_outputs(shape, baseline_outputs, candidate_outputs)

        result.update(
            {
                "status": "ok",
                "baseline_bwd_bwd": baseline_bb_status,
                "crossp_bwd_bwd": {
                    "source": _source_markers(crossp_kernel, 256),
                    "elapsed": _time_cuda_events(
                        lambda: crossp_kernel(*_bb_args(inputs, candidate_outputs), scratch),
                        warmup=warmup,
                        iters=iters,
                    ),
                    "scratch_bytes": scratch.numel() * scratch.element_size(),
                },
                "comparison_vs_stage2": comparison,
            }
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


def _selected_shapes(shape_csv: str) -> list[Shape]:
    selected: list[Shape] = []
    for raw in shape_csv.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in SHAPES:
            raise ValueError(f"unknown shape {name!r}; choose one of {sorted(SHAPES)}")
        selected.append(SHAPES[name])
    if not selected:
        raise ValueError("at least one shape required")
    return selected


@app.function(image=_image(), gpu=GPU_SPEC, timeout=1800)
def run_probe(
    requested_gpu: str,
    shape_csv: str,
    warmup: int,
    iters: int,
    p_tile: int,
    crossp_num_stages: int,
) -> dict[str, Any]:
    import traceback

    try:
        _install_source_paths()
        _reset_mamba_imports()
        print(f"[crossp] run_probe start shape_csv={shape_csv}", flush=True)
        device = _device_report(requested_gpu)
        print(f"[crossp] device ready {device.get('device')}", flush=True)
        return {
            "app_name": APP_NAME,
            "device": device,
            "settings": {
                "shape_csv": shape_csv,
                "warmup": warmup,
                "iters": iters,
                "p_tile": p_tile,
                "crossp_num_stages": crossp_num_stages,
            },
            "results": [
                _run_shape(shape, warmup, iters, p_tile, crossp_num_stages)
                for shape in _selected_shapes(shape_csv)
            ],
        }
    except BaseException as exc:  # noqa: BLE001
        return {
            "app_name": APP_NAME,
            "top_level_status": "crashed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback_tail": traceback.format_exc()[-8000:],
        }


@app.local_entrypoint()
def main(
    shape_csv: str = "tiny",
    warmup: int = 1,
    iters: int = 2,
    p_tile: int = 64,
    crossp_num_stages: int = 0,
) -> None:
    result = run_probe.remote(GPU_SPEC, shape_csv, warmup, iters, p_tile, crossp_num_stages)
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
