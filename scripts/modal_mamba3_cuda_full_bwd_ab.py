"""Modal AB harness for the Mamba3 full-CUDA bwd_bwd candidate.

This intentionally uses CUDA events, source metadata, and torch memory counters
only.  It does not require NCU.

Run examples:

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 timeout 1800s \
        modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
        --run-id wave8_h200_prod_20260430_1 \
        --shape-csv smoke,productionish \
        --iters 6 --warmup 2 --cuda-iters 10 --cuda-warmup 3

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100:2 timeout 900s \
        modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
        --run-id wave8_h100_smoke_20260430_1 \
        --shape-csv smoke \
        --iters 2 --warmup 1 --cuda-iters 5 --cuda-warmup 1

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=B200:1 timeout 900s \
        modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
        --run-id wave8_b200_smoke_20260430_1 \
        --shape-csv smoke \
        --iters 2 --warmup 1 --cuda-iters 5 --cuda-warmup 1
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Any

import modal


GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200")
BENCH_VOLUME_NAME = os.environ.get("CPPMEGA_MODAL_BENCH_VOLUME", "cppmega-mamba3-benchmarks")

APP_NAME = "cppmega-mamba3-cuda-full-bwd-ab"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"
BENCH_ROOT = "/benchmarks"
BENCH_PREFIX = "mamba3_cuda_full_bwd_ab"

bench_volume = modal.Volume.from_name(BENCH_VOLUME_NAME, create_if_missing=True)


def _image() -> modal.Image:
    image: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    env = {
        "GHCR_REPO": GHCR_REPO,
        "GHCR_TAG": GHCR_TAG,
        "CPPMEGA_IMAGE_REF": GHCR_REF,
    }
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        env["TORCH_CUDA_ARCH_LIST"] = os.environ["TORCH_CUDA_ARCH_LIST"]
    image = image.env(env)
    image = image.add_local_dir("cppmega", f"{CPPMEGA_ROOT}/cppmega", copy=True)
    image = image.add_local_dir("scripts", f"{CPPMEGA_ROOT}/scripts", copy=True)
    image = image.add_local_dir(
        "upstream_prs/examples/13_tilelang_floormod_dbz",
        f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz",
        copy=True,
    )
    image = image.add_local_dir(
        "/home/dave/state-spaces-mamba/mamba_ssm",
        f"{SOURCE_ROOT}/mamba_ssm",
        copy=True,
    )
    return image


app = modal.App(APP_NAME)


def _load_module_from_path(module_name: str, path: str) -> Any:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_stage2_benchlib() -> Any:
    """Load helper functions from the existing Modal stage2 script.

    The source is executed only through the helper-function region so this AB
    app does not register the stage2 Modal app while already running remotely.
    """

    import pathlib
    import sys
    import types

    path = pathlib.Path(CPPMEGA_ROOT) / "scripts/modal_mamba3_stage2_force_nontma_benchmark.py"
    source = path.read_text(encoding="utf-8")
    source = source.replace(
        "bench_volume = modal.Volume.from_name(BENCH_VOLUME_NAME, create_if_missing=True)",
        "bench_volume = None",
    )
    source = source.replace("app = modal.App(APP_NAME)", "app = None")
    source = source.split("\n@app.function", 1)[0]
    mod = types.ModuleType("mamba3_stage2_benchlib")
    mod.__file__ = str(path)
    mod.__dict__["__name__"] = "mamba3_stage2_benchlib"
    sys.modules[mod.__name__] = mod
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def _load_wave7_cuda() -> Any:
    import pathlib
    import sys

    bench_dir = pathlib.Path(CPPMEGA_ROOT) / "upstream_prs/examples/13_tilelang_floormod_dbz"
    bench_dir_str = str(bench_dir)
    if bench_dir_str not in sys.path:
        sys.path.insert(0, bench_dir_str)
    return _load_module_from_path("rr_diag_wave7_chunk_owner_cuda_ab", str(bench_dir / "rr_diag_wave7_chunk_owner_cuda.py"))


def _set_cuda_arch_from_device() -> str | None:
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    return os.environ.get("TORCH_CUDA_ARCH_LIST")


def _mean_ms(variant: dict[str, Any] | None, phase: str) -> float | None:
    if not variant or variant.get("status") != "ok":
        return None
    return variant.get("elapsed", {}).get(phase, {}).get("mean_ms")


def _compact_stats(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    if not stats:
        return None
    return {
        key: stats.get(key)
        for key in ("count", "mean_ms", "min_ms", "p50_ms", "p90_ms", "p95_ms", "max_ms", "std_ms")
        if key in stats
    }


def _tree_max_abs(value: Any) -> float | None:
    values: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            values.append(abs(float(node)))
        elif isinstance(node, dict):
            for child in node.values():
                walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(value)
    return max(values) if values else None


def _cuda_correctness_summary(cuda_result: dict[str, Any]) -> dict[str, Any]:
    correctness = cuda_result.get("correctness", {})
    return {
        name: {
            "max_abs": _tree_max_abs(values),
            "details": values,
        }
        for name, values in correctness.items()
    }


def _cuda_timing_summary(cuda_result: dict[str, Any]) -> dict[str, Any]:
    timings = cuda_result.get("timings", {})
    return {
        "wave6_diag_ms": _compact_stats(timings.get("wave6_chunk_warp_owner_diag_slice")),
        "wave7_qk_dv_ms": _compact_stats(timings.get("wave7_chunk_warp_qk_dv_consumer_slice")),
        "wave7_diag_plus_qk_dv_ms": _compact_stats(
            timings.get("wave7_chunk_warp_diag_plus_qk_dv_total_slice")
        ),
    }


def _replacement_estimate(
    tilelang_variants: dict[str, dict[str, Any]],
    cuda_result: dict[str, Any],
) -> dict[str, Any]:
    baseline = tilelang_variants.get("baseline")
    stage2 = tilelang_variants.get("stage2_bf1_bb0")
    stage2_ref = stage2 if stage2 and stage2.get("status") == "ok" else baseline
    timings = cuda_result.get("timings", {})
    diag = timings.get("wave6_chunk_warp_owner_diag_slice", {})
    qk_dv = timings.get("wave7_chunk_warp_qk_dv_consumer_slice", {})
    combined = timings.get("wave7_chunk_warp_diag_plus_qk_dv_total_slice", {})

    diag_ms = diag.get("mean_ms")
    qk_dv_ms = qk_dv.get("mean_ms")
    combined_ms = combined.get("mean_ms")
    component_sum_ms = combined.get("component_sum_ms")
    if component_sum_ms is None and diag_ms is not None and qk_dv_ms is not None:
        component_sum_ms = diag_ms + qk_dv_ms

    stage2_bf_ms = _mean_ms(stage2_ref, "bwd_fwd")
    stage2_bb_ms = _mean_ms(stage2_ref, "bwd_bwd")
    stage2_chain_ms = _mean_ms(stage2_ref, "chain")
    floor_chain_ms = None
    if stage2_bf_ms is not None and combined_ms is not None:
        floor_chain_ms = stage2_bf_ms + combined_ms

    estimate: dict[str, Any] = {
        "validity": (
            "incomplete floor: current CUDA candidate covers DGAMMA_DIAG, DK/DQ diagonal "
            "contributions, and qk_dot->dPsiV->DV only"
        ),
        "tilelang_reference_variant": stage2_ref["variant"] if stage2_ref else None,
        "tilelang_baseline_bwd_bwd_ms": _mean_ms(baseline, "bwd_bwd"),
        "tilelang_stage2_bf1_bb0_bwd_fwd_ms": _mean_ms(stage2, "bwd_fwd"),
        "tilelang_stage2_bf1_bb0_bwd_bwd_ms": _mean_ms(stage2, "bwd_bwd"),
        "tilelang_stage2_bf1_bb0_chain_ms": _mean_ms(stage2, "chain"),
        "component_timings_ms": {
            "wave6_diag": diag_ms,
            "wave7_qk_dv": qk_dv_ms,
            "component_sum_two_launches": component_sum_ms,
            "combined_one_launch_current_candidate": combined_ms,
        },
        "launch_counts": {
            "tilelang_bwd_bwd": 1,
            "cuda_component_sum": 2,
            "cuda_combined_current_candidate": 1,
            "stage2_chain": 2,
            "stage2_chain_with_current_incomplete_cuda_floor": 2,
        },
        "memory_peak_gib": {
            "tilelang_baseline_allocated": baseline.get("max_memory_allocated_gib") if baseline else None,
            "tilelang_baseline_reserved": baseline.get("max_memory_reserved_gib") if baseline else None,
            "tilelang_stage2_bf1_bb0_allocated": stage2.get("max_memory_allocated_gib") if stage2 else None,
            "tilelang_stage2_bf1_bb0_reserved": stage2.get("max_memory_reserved_gib") if stage2 else None,
            "cuda_components_allocated": cuda_result.get("memory", {}).get("max_memory_allocated_gib"),
            "cuda_components_reserved": cuda_result.get("memory", {}).get("max_memory_reserved_gib"),
        },
        "missing_for_full_replacement": [
            "off-time intra-chunk/state work",
            "full DK/DQ/DV accumulation, not only same-time diagonal/qk-dV slices",
            "DMIMO_V cross-chunk reduction or alternate ownership",
            "dfactor/dangles/dd/dda/dssda/dda_cs_rev/dda_cs outputs",
            "production integration in the real bwd_bwd call boundary",
        ],
    }
    if stage2_bb_ms is not None and combined_ms is not None:
        estimate["current_candidate_ratio_vs_tilelang_bwd_bwd"] = combined_ms / stage2_bb_ms
        estimate["current_candidate_speedup_floor_vs_tilelang_bwd_bwd"] = stage2_bb_ms / combined_ms
        estimate["remaining_budget_ms_to_equal_tilelang_bwd_bwd"] = stage2_bb_ms - combined_ms
    if floor_chain_ms is not None and stage2_chain_ms is not None:
        estimate["stage2_chain_with_current_incomplete_cuda_floor_ms"] = floor_chain_ms
        estimate["stage2_chain_floor_speedup"] = stage2_chain_ms / floor_chain_ms
    return estimate


def _summarize_shape(shape_result: dict[str, Any]) -> dict[str, Any]:
    tilelang = {
        item["variant"]: item
        for item in shape_result.get("tilelang_variants", [])
    }
    tilelang_summary = {}
    for name, result in tilelang.items():
        tilelang_summary[name] = {
            "status": result.get("status"),
            "bwd_fwd_ms": _compact_stats(result.get("elapsed", {}).get("bwd_fwd")),
            "bwd_bwd_ms": _compact_stats(result.get("elapsed", {}).get("bwd_bwd")),
            "chain_ms": _compact_stats(result.get("elapsed", {}).get("chain")),
            "max_memory_allocated_gib": result.get("max_memory_allocated_gib"),
            "max_memory_reserved_gib": result.get("max_memory_reserved_gib"),
            "bwd_bwd_source": result.get("tilelang_source", {}).get("bwd_bwd"),
        }
    return {
        "shape": shape_result["shape"]["name"],
        "status": shape_result["status"],
        "estimated_tensor_bytes": shape_result.get("estimated_tensor_bytes"),
        "tilelang": tilelang_summary,
        "tilelang_comparison": shape_result.get("tilelang_comparison"),
        "cuda": {
            "device": shape_result.get("cuda_result", {}).get("cuda_device"),
            "timings": _cuda_timing_summary(shape_result.get("cuda_result", {})),
            "correctness": _cuda_correctness_summary(shape_result.get("cuda_result", {})),
            "memory": shape_result.get("cuda_result", {}).get("memory"),
            "metadata": shape_result.get("cuda_result", {}).get("metadata"),
        },
        "replacement_estimate": shape_result.get("replacement_estimate"),
    }


def _summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": report["run_id"],
        "volume": report["volume"],
        "volume_relpath": report["volume_relpath"],
        "device": report["device"],
        "settings": report["settings"],
        "artifacts": report["artifacts"],
        "shapes": [_summarize_shape(shape) for shape in report["shapes"]],
    }


def _write_summary_csv(summary: dict[str, Any], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "shape",
                "tilelang_baseline_bwd_bwd_ms",
                "tilelang_stage2_bf1_bb0_bwd_bwd_ms",
                "cuda_wave6_diag_ms",
                "cuda_wave7_qk_dv_ms",
                "cuda_combined_ms",
                "cuda_component_sum_ms",
                "cuda_ratio_vs_stage2_bwd_bwd",
                "remaining_budget_ms_to_equal_stage2_bwd_bwd",
                "stage2_chain_floor_speedup",
            ],
        )
        writer.writeheader()
        for shape in summary["shapes"]:
            estimate = shape.get("replacement_estimate", {})
            components = estimate.get("component_timings_ms", {})
            writer.writerow(
                {
                    "shape": shape["shape"],
                    "tilelang_baseline_bwd_bwd_ms": estimate.get("tilelang_baseline_bwd_bwd_ms"),
                    "tilelang_stage2_bf1_bb0_bwd_bwd_ms": estimate.get(
                        "tilelang_stage2_bf1_bb0_bwd_bwd_ms"
                    ),
                    "cuda_wave6_diag_ms": components.get("wave6_diag"),
                    "cuda_wave7_qk_dv_ms": components.get("wave7_qk_dv"),
                    "cuda_combined_ms": components.get("combined_one_launch_current_candidate"),
                    "cuda_component_sum_ms": components.get("component_sum_two_launches"),
                    "cuda_ratio_vs_stage2_bwd_bwd": estimate.get(
                        "current_candidate_ratio_vs_tilelang_bwd_bwd"
                    ),
                    "remaining_budget_ms_to_equal_stage2_bwd_bwd": estimate.get(
                        "remaining_budget_ms_to_equal_tilelang_bwd_bwd"
                    ),
                    "stage2_chain_floor_speedup": estimate.get("stage2_chain_floor_speedup"),
                }
            )


@app.function(image=_image(), gpu=GPU_SPEC, timeout=30 * 60, volumes={BENCH_ROOT: bench_volume})
def run_ab_remote(
    requested_gpu: str,
    run_id: str | None,
    shape_csv: str,
    tilelang_variant_csv: str,
    warmup: int,
    iters: int,
    cuda_warmup: int,
    cuda_iters: int,
    seed: int,
) -> dict[str, Any]:
    import os
    import time
    import traceback

    import torch

    stage2 = _load_stage2_benchlib()
    stage2._install_source_paths()
    wave7 = _load_wave7_cuda()

    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_rel = f"{BENCH_PREFIX}/{run_id}"
    run_dir = os.path.join(BENCH_ROOT, run_rel)
    os.makedirs(run_dir, exist_ok=True)

    report: dict[str, Any] = {
        "run_id": run_id,
        "volume": BENCH_VOLUME_NAME,
        "volume_relpath": f"/{run_rel}",
        "artifact_dir": run_dir,
        "device": stage2._device_report(requested_gpu),
        "tilelang": stage2._tilelang_report(),
        "settings": {
            "shape_csv": shape_csv,
            "tilelang_variant_csv": tilelang_variant_csv,
            "warmup": warmup,
            "iters": iters,
            "cuda_warmup": cuda_warmup,
            "cuda_iters": cuda_iters,
            "seed": seed,
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        },
        "shapes": [],
    }

    for shape in stage2._selected_shapes(shape_csv):
        shape_dir = os.path.join(run_dir, shape.name)
        os.makedirs(shape_dir, exist_ok=True)
        inputs = stage2._make_inputs(shape)
        tilelang_results: list[dict[str, Any]] = []
        shape_status = "ok"

        for variant in stage2._selected_variants(tilelang_variant_csv):
            stage2._reset_mamba_imports()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            variant_result = stage2._benchmark_variant(
                variant,
                shape,
                inputs,
                shape_dir,
                warmup=warmup,
                iters=iters,
                torch_profile=False,
            )
            tilelang_results.append(variant_result)
            if variant_result.get("status") != "ok":
                shape_status = "tilelang_variant_failed"

        compare_input = {
            "shape": asdict(shape),
            "variants": tilelang_results,
        }
        try:
            tilelang_comparison = stage2._compare_shape(compare_input)
        except Exception as exc:  # noqa: BLE001
            tilelang_comparison = {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-4000:],
            }
            shape_status = "tilelang_comparison_failed"

        stripped_tilelang = [stage2._strip_tensors(item) for item in tilelang_results]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        arch_list = _set_cuda_arch_from_device()
        try:
            cuda_args = wave7.argparse.Namespace(
                shape=shape.name,
                B=shape.B,
                S=shape.S,
                H=shape.H,
                G=shape.G,
                N=shape.N,
                P=shape.P,
                R=shape.R,
                chunk=shape.chunk,
                dtype="bf16",
                device="cuda",
                seed=seed,
                warmup=cuda_warmup,
                iters=cuda_iters,
            )
            cuda_result = wave7.run(cuda_args)
            torch.cuda.synchronize()
            cuda_result["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            cuda_result = {
                "status": "crashed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-6000:],
            }
            shape_status = "cuda_failed"
        cuda_result["memory"] = {
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        }
        cuda_result["torch_cuda_arch_list"] = arch_list

        tilelang_by_name = {item["variant"]: item for item in stripped_tilelang}
        shape_result: dict[str, Any] = {
            "shape": asdict(shape),
            "status": shape_status,
            "estimated_tensor_bytes": stage2._shape_bytes_estimate(shape),
            "tilelang_variants": stripped_tilelang,
            "tilelang_comparison": tilelang_comparison,
            "cuda_result": cuda_result,
        }
        shape_result["replacement_estimate"] = _replacement_estimate(tilelang_by_name, cuda_result)
        report["shapes"].append(shape_result)

    report["artifacts"] = {
        "report_json": os.path.join(run_dir, "report.json"),
        "summary_json": os.path.join(run_dir, "summary.json"),
        "summary_csv": os.path.join(run_dir, "summary.csv"),
    }
    summary = _summarize_report(report)
    with open(report["artifacts"]["report_json"], "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    with open(report["artifacts"]["summary_json"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    _write_summary_csv(summary, report["artifacts"]["summary_csv"])
    bench_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_id: str | None = None,
    shape_csv: str = "smoke,productionish",
    tilelang_variant_csv: str = "baseline,stage2_bf1_bb0",
    warmup: int = 2,
    iters: int = 6,
    cuda_warmup: int = 3,
    cuda_iters: int = 10,
    seed: int = 20260430,
) -> None:
    result = run_ab_remote.remote(
        GPU_SPEC,
        run_id,
        shape_csv,
        tilelang_variant_csv,
        warmup,
        iters,
        cuda_warmup,
        cuda_iters,
        seed,
    )
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
