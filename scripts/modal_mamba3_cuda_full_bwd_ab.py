"""Modal AB harness for Mamba3 monolithic bwd_bwd production candidates.

This intentionally uses CUDA events, source metadata, and torch memory counters
only.  It does not require NCU.  The report schema compares the main guarded
stage2 production reference, the prior covered CUDA subset, and named future
monolithic candidates against the same shape/config/output-slot contract.

Run examples:

    CPPMEGA_MODAL_GPU=H200 modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
        --dry-run-schema \
        --shape-csv smoke \
        --monolithic-candidate-csv mono_chunk_v0 \
        --candidate-record-path-csv docs/status/lane_a_component_record.md \
        --modal-hygiene-enforcement fail

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 timeout 900s \
        modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
        --run-id mono_ab_wave1_h200_smoke_20260430_1 \
        --shape-csv smoke \
        --iters 1 --warmup 0 --cuda-iters 1 --cuda-warmup 0

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 timeout 1800s \
        modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
        --run-id wave9_h200_prod_20260430_1 \
        --shape-csv smoke,productionish \
        --iters 6 --warmup 2 --cuda-iters 10 --cuda-warmup 3

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100:2 timeout 900s \
        modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
        --run-id wave9_h100_smoke_20260430_1 \
        --shape-csv smoke \
        --iters 2 --warmup 1 --cuda-iters 5 --cuda-warmup 1

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=B200:1 timeout 900s \
        modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
        --run-id wave9_b200_smoke_20260430_1 \
        --shape-csv smoke \
        --iters 2 --warmup 1 --cuda-iters 5 --cuda-warmup 1
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import asdict
from typing import Any

import modal

from cppmega.megatron.mamba3_mono_ab_schema import (
    BWD_BWD_OUTPUT_NAMES,
    MAIN_GUARDED_STAGE2_COMMIT,
    SCHEMA_VERSION,
    candidate_configs,
    component_record_projection,
    coerce_shape,
    cuda_subset_slot_results,
    empty_slot_results,
    filter_candidate_component_records_for_shape,
    guarded_stage2_training_ab_stub,
    load_candidate_component_records,
    memory_accounting,
    normalize_candidate_component_records,
    readiness_gates,
    selected_shapes as selected_schema_shapes,
    slot_results_from_diffs,
    slot_schema,
    summarize_slot_results,
)


GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200")
BENCH_VOLUME_NAME = os.environ.get("CPPMEGA_MODAL_BENCH_VOLUME", "cppmega-mamba3-benchmarks")

APP_NAME = "cppmega-mamba3-cuda-full-bwd-ab"
DEFAULT_MODAL_CAMPAIGN_PREFIX_CSV = "cppmega-mamba3-"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"
BENCH_ROOT = "/benchmarks"
BENCH_PREFIX = "mamba3_cuda_full_bwd_ab"

bench_volume = modal.Volume.from_name(BENCH_VOLUME_NAME, create_if_missing=True)


DMIMOV_SIDECAR_SOURCE: dict[str, Any] = {
    "source_worktree": "worker/mamba3-cuda-dmimo-reduce",
    "source_commit": "9308289",
    "doc": "docs/status/mamba3_cuda_dmimo_reduce_wave8_2026_04_30.md",
    "scope": (
        "qk_dot same-time DMIMO_V contribution only; full state/intra-chunk "
        "DMIMO_V terms are still outside this sidecar"
    ),
    "h200_productionish": {
        "device": "NVIDIA H200",
        "shape": "productionish",
        "tilelang_stage2_bf1_bb0_bwd_bwd_ms": 3.70674,
        "wave7_diag_plus_qk_dv_ms": 1.91459,
        "wave7_refreshed_in_sidecar_ms": 1.92434,
        "qk_dmimov_output_owner_all_r_ms": 0.53634,
        "projected_wave7_plus_qk_dmimov_ms": 2.45093,
        "projected_ratio_vs_tilelang_stage2": 0.661,
        "correctness_max_abs": 1.066e-13,
        "metadata": {
            "regs_per_thread": 40,
            "static_smem_bytes": 2048,
            "active_blocks_per_sm": 12,
            "occupancy": 0.75,
        },
    },
    "h100_smoke": {
        "device": "NVIDIA H100 80GB HBM3",
        "shape": "smoke",
        "wave7_diag_plus_qk_dv_ms": 0.03193600028753281,
        "qk_dmimov_output_owner_all_r_ms": 0.026003200188279153,
        "two_pass_total_ms": 0.016627199575304986,
        "atomic_chunk_ms": 0.020524800196290015,
        "output_owner_single_r_ms": 0.023174399882555007,
        "correctness_max_abs": 2.665e-15,
        "metadata": {
            "regs_per_thread": 40,
            "static_smem_bytes": 2048,
            "active_blocks_per_sm": 12,
            "occupancy": 0.75,
        },
    },
}


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


def _shape_memory_model(shape: dict[str, Any]) -> dict[str, Any]:
    b = int(shape["B"])
    s = int(shape["S"])
    h = int(shape["H"])
    r = int(shape["R"])
    p = int(shape["P"])
    chunk = int(shape["chunk"])
    nchunks = (s + chunk - 1) // chunk
    output_bytes = b * h * r * p * 4
    partial_bytes = b * h * nchunks * r * p * 4
    atomic_adds = b * h * nchunks * r * p
    timestep_contributions = b * h * s * r * p
    return {
        "dmimo_v_output_bytes": output_bytes,
        "dmimo_v_output_mib": output_bytes / (1024**2),
        "atomic_chunk_extra_temp_bytes": 0,
        "atomic_chunk_global_atomic_adds": atomic_adds,
        "two_pass_partial_bytes": partial_bytes,
        "two_pass_partial_mib": partial_bytes / (1024**2),
        "two_pass_extra_global_rw_bytes": partial_bytes * 2 + output_bytes,
        "two_pass_extra_global_rw_mib": (partial_bytes * 2 + output_bytes) / (1024**2),
        "output_owner_all_r_extra_temp_bytes": 0,
        "raw_timestep_dmimov_contributions": timestep_contributions,
    }


def _dmimov_sidecar_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    measured = None
    if shape.get("name") == "productionish":
        measured = DMIMOV_SIDECAR_SOURCE["h200_productionish"]
    return {
        "source": {
            key: DMIMOV_SIDECAR_SOURCE[key]
            for key in ("source_worktree", "source_commit", "doc", "scope")
        },
        "available_receipts": {
            key: DMIMOV_SIDECAR_SOURCE[key]
            for key in ("h200_productionish", "h100_smoke")
        },
        "measured": measured,
        "memory_model": _shape_memory_model(shape),
        "launch_accounting": {
            "standalone_output_owner_all_r_launches": 1,
            "wave7_plus_dmimov_sidecar_bwd_bwd_launches": 2,
            "stage2_bwd_fwd_plus_wave7_plus_dmimov_sidecar_chain_launches": 3,
            "production_single_bwd_bwd_replacement_target_launches": 1,
        },
    }


def _compact_stats(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    if not stats:
        return None
    return {
        key: stats.get(key)
        for key in ("count", "mean_ms", "min_ms", "p50_ms", "p90_ms", "p95_ms", "max_ms", "std_ms")
        if key in stats
    }


def _ab_schema_for_shape(
    shape: dict[str, Any],
    monolithic_candidate_csv: str,
    component_records: list[dict[str, Any]] | None = None,
    projection_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shape_obj = coerce_shape(shape)
    shape_component_records = filter_candidate_component_records_for_shape(
        component_records or [],
        shape_obj,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "shape": shape_obj.to_dict(),
        "candidate_configs": candidate_configs(
            monolithic_candidate_csv,
            shape_component_records,
        ),
        "candidate_component_records": shape_component_records,
        "candidate_component_projections": [
            component_record_projection(record, reference=projection_reference)
            for record in shape_component_records
        ],
        "boundary_slots": slot_schema(shape_obj),
        "memory_accounting": memory_accounting(shape_obj),
        "readiness_gates": readiness_gates(),
    }


def _find_tilelang_variant(
    tilelang_variants: dict[str, dict[str, Any]],
    names: tuple[str, ...],
) -> dict[str, Any] | None:
    for name in names:
        variant = tilelang_variants.get(name)
        if variant:
            return variant
    return None


def _timing_for_variant(variant: dict[str, Any] | None) -> dict[str, Any]:
    if not variant:
        return {}
    elapsed = variant.get("elapsed", {})
    return {
        "bwd_fwd_ms": _compact_stats(elapsed.get("bwd_fwd")),
        "bwd_bwd_ms": _compact_stats(elapsed.get("bwd_bwd")),
        "chain_ms": _compact_stats(elapsed.get("chain")),
    }


def _projection_reference_for_stage2(
    tilelang_variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stage2 = _find_tilelang_variant(tilelang_variants, ("stage2_bf1_bb0", "stage2_force_nontma"))
    if not stage2 or stage2.get("status") != "ok":
        stage2 = tilelang_variants.get("baseline")
    if not stage2:
        return {}
    return {
        "stage2_bwd_fwd_ms": _mean_ms(stage2, "bwd_fwd"),
        "stage2_bwd_bwd_ms": _mean_ms(stage2, "bwd_bwd"),
        "stage2_chain_ms": _mean_ms(stage2, "chain"),
        "stage2_max_memory_allocated_gib": stage2.get("max_memory_allocated_gib"),
        "stage2_max_memory_reserved_gib": stage2.get("max_memory_reserved_gib"),
    }


def _memory_for_variant(variant: dict[str, Any] | None) -> dict[str, Any]:
    if not variant:
        return {}
    return {
        "max_memory_allocated_gib": variant.get("max_memory_allocated_gib"),
        "max_memory_reserved_gib": variant.get("max_memory_reserved_gib"),
    }


def _stage2_candidate_report(
    tilelang_variants: dict[str, dict[str, Any]],
    tilelang_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    stage2 = _find_tilelang_variant(tilelang_variants, ("stage2_bf1_bb0", "stage2_force_nontma"))
    variant_name = stage2.get("variant") if stage2 else "stage2_bf1_bb0"
    compare = (tilelang_comparison or {}).get("vs_baseline", {}).get(variant_name, {})
    diffs = compare.get("diffs") or {}
    if diffs:
        slot_results = slot_results_from_diffs(diffs, atol=0.0)
    else:
        slot_results = empty_slot_results(
            "not_reported",
            f"{variant_name} was not compared against baseline in this run",
        )
    return {
        "candidate_id": "main_guarded_stage2",
        "role": "production_reference",
        "source_commit": MAIN_GUARDED_STAGE2_COMMIT,
        "variant": variant_name,
        "status": stage2.get("status") if stage2 else "missing",
        "timings": _timing_for_variant(stage2),
        "memory_peak_gib": _memory_for_variant(stage2),
        "slot_results": slot_results,
        "slot_summary": summarize_slot_results(slot_results),
        "comparison_vs_baseline": {
            "status": compare.get("status"),
            "speedup": compare.get("speedup"),
            "max_main_grad_abs_diff": compare.get("max_main_grad_abs_diff"),
        },
    }


def _cuda_subset_candidate_report(
    cuda_result: dict[str, Any],
    replacement_estimate: dict[str, Any],
) -> dict[str, Any]:
    dmimov_sidecar = replacement_estimate.get("dmimov_sidecar", {})
    slot_results = cuda_subset_slot_results(
        cuda_result.get("correctness", {}),
        dmimov_sidecar_receipt=bool(dmimov_sidecar.get("measured")),
    )
    return {
        "candidate_id": "cuda_covered_subset_wave9",
        "role": "prior_component_floor",
        "status": cuda_result.get("status"),
        "boundary_status": "partial_only",
        "timings": _cuda_timing_summary(cuda_result),
        "component_timings_ms": replacement_estimate.get("component_timings_ms"),
        "replacement_estimate": {
            key: replacement_estimate.get(key)
            for key in (
                "current_candidate_ratio_vs_tilelang_bwd_bwd",
                "current_candidate_speedup_floor_vs_tilelang_bwd_bwd",
                "remaining_budget_ms_to_equal_tilelang_bwd_bwd",
                "candidate_plus_qk_dmimov_ratio_vs_tilelang_bwd_bwd",
                "candidate_plus_qk_dmimov_speedup_floor_vs_tilelang_bwd_bwd",
                "remaining_budget_after_qk_dmimov_to_equal_tilelang_bwd_bwd",
                "stage2_chain_floor_speedup",
                "stage2_chain_with_current_plus_qk_dmimov_speedup",
            )
            if key in replacement_estimate
        },
        "memory_peak_gib": replacement_estimate.get("memory_peak_gib"),
        "slot_results": slot_results,
        "slot_summary": summarize_slot_results(slot_results),
        "missing_for_full_replacement": replacement_estimate.get("missing_for_full_replacement"),
    }


def _component_record_slot_results(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = empty_slot_results(
        "missing",
        "not covered by the candidate component record",
    )
    correctness = record.get("correctness") or {}
    for slot_name in record.get("covered_slots") or []:
        results[slot_name] = {
            "status": "partial_component_record",
            "max_abs": correctness.get("max_abs"),
            "ref_absmax": correctness.get("ref_absmax"),
            "rel_to_ref_absmax": correctness.get("rel_to_ref_absmax"),
            "full_boundary_pass": False,
            "coverage": "reported by external Lane A/B/C/D component record",
            "note": "component coverage must be rechecked at the integrated call boundary",
        }
    return results


def _component_candidate_reports(
    component_records: list[dict[str, Any]],
    shape: dict[str, Any],
    projection_reference: dict[str, Any],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for record in filter_candidate_component_records_for_shape(component_records, shape):
        slot_results = _component_record_slot_results(record)
        projection = component_record_projection(
            record,
            reference=projection_reference,
        )
        reports.append(
            {
                "candidate_id": record["candidate_id"],
                "role": record["role"],
                "status": record["status"],
                "boundary_status": (
                    "full_boundary_claim"
                    if not record.get("missing_slots")
                    else "partial_component_record"
                ),
                "source": record.get("source"),
                "components": record.get("components"),
                "projection": projection,
                "production_gate": projection["production_gate"],
                "memory_peak_gib": record.get("memory_peak_gib"),
                "hardware_tags": record.get("hardware_tags") or [],
                "modal_hygiene": record.get("modal_hygiene") or {},
                "slot_results": slot_results,
                "slot_summary": summarize_slot_results(slot_results),
                "missing_for_full_replacement": record.get("missing_slots"),
                "note": record.get("note"),
            }
        )
    return reports


def _future_monolithic_candidate_reports(
    monolithic_candidate_csv: str,
    component_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    component_ids = {
        record["candidate_id"]
        for record in normalize_candidate_component_records(component_records or [])
    }
    for config in candidate_configs(monolithic_candidate_csv):
        if config.get("role") != "future_monolithic_candidate":
            continue
        if config["candidate_id"] in component_ids:
            continue
        slot_results = empty_slot_results(
            "not_run",
            "reserved schema slot for a future monolithic mamba_mimo_bwd_bwd candidate",
        )
        reports.append(
            {
                "candidate_id": config["candidate_id"],
                "role": config["role"],
                "status": "pending_integration",
                "boundary_status": "not_run",
                "expected_output_slots": list(BWD_BWD_OUTPUT_NAMES),
                "slot_results": slot_results,
                "slot_summary": summarize_slot_results(slot_results),
            }
        )
    return reports


def _candidate_reports_for_shape(
    tilelang_variants: dict[str, dict[str, Any]],
    tilelang_comparison: dict[str, Any] | None,
    cuda_result: dict[str, Any],
    replacement_estimate: dict[str, Any],
    monolithic_candidate_csv: str,
    component_records: list[dict[str, Any]] | None = None,
    shape: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    projection_reference = _projection_reference_for_stage2(tilelang_variants)
    return [
        _stage2_candidate_report(tilelang_variants, tilelang_comparison),
        _cuda_subset_candidate_report(cuda_result, replacement_estimate),
        *_component_candidate_reports(
            component_records or [],
            shape or {},
            projection_reference,
        ),
        *_future_monolithic_candidate_reports(monolithic_candidate_csv, component_records),
    ]


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
    shape: dict[str, Any],
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
    dmimov_sidecar = _dmimov_sidecar_for_shape(shape)
    dmimov_measured = dmimov_sidecar.get("measured")
    dmimov_all_r_ms = (
        dmimov_measured.get("qk_dmimov_output_owner_all_r_ms") if dmimov_measured else None
    )
    combined_plus_dmimov_ms = None
    if combined_ms is not None and dmimov_all_r_ms is not None:
        combined_plus_dmimov_ms = combined_ms + dmimov_all_r_ms

    estimate: dict[str, Any] = {
        "validity": (
            "incomplete floor: current CUDA candidate covers DGAMMA_DIAG, DK/DQ diagonal "
            "contributions, qk_dot->dPsiV->DV, and sidecar qk_dot->DMIMO_V only"
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
            "qk_dmimov_output_owner_all_r_sidecar": dmimov_all_r_ms,
            "combined_plus_qk_dmimov_sidecar": combined_plus_dmimov_ms,
        },
        "launch_counts": {
            "tilelang_bwd_bwd": 1,
            "cuda_component_sum": 2,
            "cuda_combined_current_candidate": 1,
            "cuda_combined_plus_qk_dmimov_sidecar": (
                dmimov_sidecar["launch_accounting"]["wave7_plus_dmimov_sidecar_bwd_bwd_launches"]
            ),
            "stage2_chain": 2,
            "stage2_chain_with_current_incomplete_cuda_floor": 2,
            "stage2_chain_with_current_plus_qk_dmimov_sidecar": (
                dmimov_sidecar["launch_accounting"][
                    "stage2_bwd_fwd_plus_wave7_plus_dmimov_sidecar_chain_launches"
                ]
            ),
        },
        "memory_peak_gib": {
            "tilelang_baseline_allocated": baseline.get("max_memory_allocated_gib") if baseline else None,
            "tilelang_baseline_reserved": baseline.get("max_memory_reserved_gib") if baseline else None,
            "tilelang_stage2_bf1_bb0_allocated": stage2.get("max_memory_allocated_gib") if stage2 else None,
            "tilelang_stage2_bf1_bb0_reserved": stage2.get("max_memory_reserved_gib") if stage2 else None,
            "cuda_components_allocated": cuda_result.get("memory", {}).get("max_memory_allocated_gib"),
            "cuda_components_reserved": cuda_result.get("memory", {}).get("max_memory_reserved_gib"),
        },
        "dmimov_sidecar": dmimov_sidecar,
        "missing_for_full_replacement": [
            "off-time intra-chunk/state work",
            "full DK/DQ/DV accumulation, not only same-time diagonal/qk-dV slices",
            "full DMIMO_V accumulation, not only the qk_dot same-time sidecar slice",
            "dfactor/dangles/dd/dda/dssda/dda_cs_rev/dda_cs outputs",
            "production integration in the real bwd_bwd call boundary",
        ],
    }
    if stage2_bb_ms is not None and combined_ms is not None:
        estimate["current_candidate_ratio_vs_tilelang_bwd_bwd"] = combined_ms / stage2_bb_ms
        estimate["current_candidate_speedup_floor_vs_tilelang_bwd_bwd"] = stage2_bb_ms / combined_ms
        estimate["remaining_budget_ms_to_equal_tilelang_bwd_bwd"] = stage2_bb_ms - combined_ms
    if stage2_bb_ms is not None and combined_plus_dmimov_ms is not None:
        estimate["candidate_plus_qk_dmimov_ratio_vs_tilelang_bwd_bwd"] = (
            combined_plus_dmimov_ms / stage2_bb_ms
        )
        estimate["candidate_plus_qk_dmimov_speedup_floor_vs_tilelang_bwd_bwd"] = (
            stage2_bb_ms / combined_plus_dmimov_ms
        )
        estimate["remaining_budget_after_qk_dmimov_to_equal_tilelang_bwd_bwd"] = (
            stage2_bb_ms - combined_plus_dmimov_ms
        )
    if floor_chain_ms is not None and stage2_chain_ms is not None:
        estimate["stage2_chain_with_current_incomplete_cuda_floor_ms"] = floor_chain_ms
        estimate["stage2_chain_floor_speedup"] = stage2_chain_ms / floor_chain_ms
    if (
        stage2_bf_ms is not None
        and combined_plus_dmimov_ms is not None
        and stage2_chain_ms is not None
    ):
        dmimov_floor_chain_ms = stage2_bf_ms + combined_plus_dmimov_ms
        estimate["stage2_chain_with_current_plus_qk_dmimov_floor_ms"] = dmimov_floor_chain_ms
        estimate["stage2_chain_with_current_plus_qk_dmimov_speedup"] = (
            stage2_chain_ms / dmimov_floor_chain_ms
        )
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
        "ab_schema": shape_result.get("ab_schema"),
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
        "candidate_reports": shape_result.get("candidate_reports"),
    }


def _summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "run_id": report["run_id"],
        "volume": report["volume"],
        "volume_relpath": report["volume_relpath"],
        "device": report["device"],
        "settings": report["settings"],
        "artifacts": report["artifacts"],
        "candidate_configs": report.get("candidate_configs"),
        "candidate_component_records": report.get("candidate_component_records"),
        "readiness_gates": report.get("readiness_gates"),
        "modal_hygiene_policy": report.get("modal_hygiene_policy"),
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
                "qk_dmimov_output_owner_all_r_sidecar_ms",
                "cuda_combined_plus_qk_dmimov_ms",
                "cuda_component_sum_ms",
                "cuda_ratio_vs_stage2_bwd_bwd",
                "remaining_budget_ms_to_equal_stage2_bwd_bwd",
                "cuda_plus_qk_dmimov_ratio_vs_stage2_bwd_bwd",
                "remaining_budget_after_qk_dmimov_ms",
                "stage2_chain_floor_speedup",
                "stage2_chain_plus_qk_dmimov_floor_speedup",
                "bwd_bwd_output_mib",
                "main_guarded_stage2_full_boundary_pass",
                "main_guarded_stage2_full_boundary_pass_count",
                "cuda_subset_partial_slots",
                "cuda_subset_missing_slots",
                "component_candidate_ids",
                "component_candidate_projected_bwd_bwd_ms",
                "component_candidate_remaining_budget_ms",
                "component_candidate_production_credit",
                "future_monolithic_candidate_ids",
            ],
        )
        writer.writeheader()
        for shape in summary["shapes"]:
            estimate = shape.get("replacement_estimate", {})
            components = estimate.get("component_timings_ms", {})
            memory = shape.get("ab_schema", {}).get("memory_accounting", {})
            reports = {
                item.get("candidate_id"): item
                for item in (shape.get("candidate_reports") or [])
            }
            stage2_slots = reports.get("main_guarded_stage2", {}).get("slot_summary", {})
            cuda_slots = reports.get("cuda_covered_subset_wave9", {}).get("slot_summary", {})
            future_ids = [
                item.get("candidate_id")
                for item in (shape.get("candidate_reports") or [])
                if item.get("role") == "future_monolithic_candidate"
            ]
            component_reports = [
                item
                for item in (shape.get("candidate_reports") or [])
                if item.get("role") == "external_component_candidate"
            ]
            component_ids = [item.get("candidate_id") for item in component_reports]
            component_projected = [
                f"{item.get('candidate_id')}={item.get('projection', {}).get('projected_bwd_bwd_ms')}"
                for item in component_reports
                if item.get("candidate_id")
            ]
            component_budget = [
                (
                    f"{item.get('candidate_id')}="
                    f"{item.get('projection', {}).get('remaining_budget_ms_to_equal_stage2_bwd_bwd')}"
                )
                for item in component_reports
                if item.get("candidate_id")
            ]
            component_credit = [
                (
                    f"{item.get('candidate_id')}="
                    f"{item.get('production_gate', {}).get('production_credit')}"
                )
                for item in component_reports
                if item.get("candidate_id")
            ]
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
                    "qk_dmimov_output_owner_all_r_sidecar_ms": components.get(
                        "qk_dmimov_output_owner_all_r_sidecar"
                    ),
                    "cuda_combined_plus_qk_dmimov_ms": components.get(
                        "combined_plus_qk_dmimov_sidecar"
                    ),
                    "cuda_component_sum_ms": components.get("component_sum_two_launches"),
                    "cuda_ratio_vs_stage2_bwd_bwd": estimate.get(
                        "current_candidate_ratio_vs_tilelang_bwd_bwd"
                    ),
                    "remaining_budget_ms_to_equal_stage2_bwd_bwd": estimate.get(
                        "remaining_budget_ms_to_equal_tilelang_bwd_bwd"
                    ),
                    "cuda_plus_qk_dmimov_ratio_vs_stage2_bwd_bwd": estimate.get(
                        "candidate_plus_qk_dmimov_ratio_vs_tilelang_bwd_bwd"
                    ),
                    "remaining_budget_after_qk_dmimov_ms": estimate.get(
                        "remaining_budget_after_qk_dmimov_to_equal_tilelang_bwd_bwd"
                    ),
                    "stage2_chain_floor_speedup": estimate.get("stage2_chain_floor_speedup"),
                    "stage2_chain_plus_qk_dmimov_floor_speedup": estimate.get(
                        "stage2_chain_with_current_plus_qk_dmimov_speedup"
                    ),
                    "bwd_bwd_output_mib": memory.get("bwd_bwd_output_mib"),
                    "main_guarded_stage2_full_boundary_pass": stage2_slots.get(
                        "full_boundary_pass"
                    ),
                    "main_guarded_stage2_full_boundary_pass_count": stage2_slots.get(
                        "full_boundary_pass_count"
                    ),
                    "cuda_subset_partial_slots": ",".join(cuda_slots.get("partial") or []),
                    "cuda_subset_missing_slots": ",".join(cuda_slots.get("missing") or []),
                    "component_candidate_ids": ",".join(
                        str(item) for item in component_ids if item
                    ),
                    "component_candidate_projected_bwd_bwd_ms": ";".join(component_projected),
                    "component_candidate_remaining_budget_ms": ";".join(component_budget),
                    "component_candidate_production_credit": ";".join(component_credit),
                    "future_monolithic_candidate_ids": ",".join(
                        str(item) for item in future_ids if item
                    ),
                }
            )


@app.function(image=_image(), gpu=GPU_SPEC, timeout=30 * 60, volumes={BENCH_ROOT: bench_volume})
def run_ab_remote(
    requested_gpu: str,
    run_id: str | None,
    shape_csv: str,
    tilelang_variant_csv: str,
    monolithic_candidate_csv: str,
    candidate_records_json: str,
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
    component_records = normalize_candidate_component_records(
        json.loads(candidate_records_json or "[]")
    )

    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_rel = f"{BENCH_PREFIX}/{run_id}"
    run_dir = os.path.join(BENCH_ROOT, run_rel)
    os.makedirs(run_dir, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "volume": BENCH_VOLUME_NAME,
        "volume_relpath": f"/{run_rel}",
        "artifact_dir": run_dir,
        "device": stage2._device_report(requested_gpu),
        "tilelang": stage2._tilelang_report(),
        "candidate_configs": candidate_configs(monolithic_candidate_csv, component_records),
        "candidate_component_records": component_records,
        "readiness_gates": readiness_gates(),
        "modal_hygiene_policy": {
            "safe_auto_stop_scope": (
                "local entrypoint stops only exact-name apps with zero tasks; "
                "same-campaign active apps are reported for warn/fail gating"
            ),
            "app_name": APP_NAME,
            "same_campaign_prefix_csv": DEFAULT_MODAL_CAMPAIGN_PREFIX_CSV,
        },
        "settings": {
            "shape_csv": shape_csv,
            "tilelang_variant_csv": tilelang_variant_csv,
            "monolithic_candidate_csv": monolithic_candidate_csv,
            "candidate_component_record_count": len(component_records),
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
        shape_result["replacement_estimate"] = _replacement_estimate(
            tilelang_by_name,
            cuda_result,
            shape_result["shape"],
        )
        projection_reference = _projection_reference_for_stage2(tilelang_by_name)
        shape_result["ab_schema"] = _ab_schema_for_shape(
            shape_result["shape"],
            monolithic_candidate_csv,
            component_records,
            projection_reference,
        )
        shape_result["candidate_reports"] = _candidate_reports_for_shape(
            tilelang_by_name,
            tilelang_comparison,
            cuda_result,
            shape_result["replacement_estimate"],
            monolithic_candidate_csv,
            component_records,
            shape_result["shape"],
        )
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


def _modal_tasks(entry: dict[str, Any]) -> int | None:
    raw = str(entry.get("Tasks", "")).strip()
    if raw.isdigit():
        return int(raw)
    return None


def _compact_modal_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "app_id": entry.get("App ID"),
        "description": entry.get("Description"),
        "state": entry.get("State"),
        "tasks": _modal_tasks(entry),
        "created_at": entry.get("Created at"),
        "stopped_at": entry.get("Stopped at"),
    }


def _modal_campaign_prefixes(prefix_csv: str) -> tuple[str, ...]:
    prefixes = tuple(item.strip() for item in prefix_csv.split(",") if item.strip())
    return prefixes or tuple(
        item.strip()
        for item in DEFAULT_MODAL_CAMPAIGN_PREFIX_CSV.split(",")
        if item.strip()
    )


def _modal_entry_active(entry: dict[str, Any]) -> bool:
    return str(entry.get("State", "")).lower() != "stopped"


def _modal_entry_matches_campaign(
    entry: dict[str, Any],
    campaign_prefixes: tuple[str, ...],
) -> bool:
    description = str(entry.get("Description") or "")
    return any(description.startswith(prefix) for prefix in campaign_prefixes)


def _modal_app_list() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["modal", "app", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
    if proc.returncode != 0:
        return {
            "status": "failed",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    try:
        entries = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "stdout_tail": proc.stdout[-2000:],
        }
    return {"status": "ok", "entries": entries}


def _modal_hygiene_check(
    phase: str,
    *,
    auto_stop: bool,
    campaign_prefixes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    campaign_prefixes = campaign_prefixes or _modal_campaign_prefixes(
        DEFAULT_MODAL_CAMPAIGN_PREFIX_CSV
    )
    listed = _modal_app_list()
    result: dict[str, Any] = {
        "phase": phase,
        "app_name": APP_NAME,
        "same_campaign_prefixes": list(campaign_prefixes),
        "auto_stop_requested": auto_stop,
        "list_status": listed.get("status"),
    }
    if listed.get("status") != "ok":
        result["list_error"] = listed
        return result

    entries = listed.get("entries") or []
    owned = [
        entry
        for entry in entries
        if entry.get("Description") == APP_NAME
    ]
    non_stopped = [
        entry
        for entry in entries
        if _modal_entry_active(entry)
    ]
    same_campaign_active = [
        entry
        for entry in entries
        if _modal_entry_active(entry)
        and _modal_entry_matches_campaign(entry, campaign_prefixes)
    ]
    safe_to_stop = [
        entry
        for entry in owned
        if _modal_entry_active(entry)
        and _modal_tasks(entry) == 0
        and entry.get("App ID")
    ]
    blocked = [
        entry
        for entry in owned
        if _modal_entry_active(entry)
        and _modal_tasks(entry) != 0
    ]
    result.update(
        {
            "owned_entries": [_compact_modal_entry(entry) for entry in owned],
            "non_stopped_entries": [_compact_modal_entry(entry) for entry in non_stopped],
            "same_campaign_active_entries": [
                _compact_modal_entry(entry) for entry in same_campaign_active
            ],
            "safe_to_stop": [_compact_modal_entry(entry) for entry in safe_to_stop],
            "blocked_owned_entries": [_compact_modal_entry(entry) for entry in blocked],
            "stopped": [],
        }
    )
    if not auto_stop:
        return result

    for entry in safe_to_stop:
        app_id = str(entry["App ID"])
        proc = subprocess.run(
            ["modal", "app", "stop", "--yes", app_id],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        result["stopped"].append(
            {
                "app_id": app_id,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-1000:],
                "stderr_tail": proc.stderr[-1000:],
            }
        )
    if result["stopped"]:
        result["after_stop"] = _modal_hygiene_check(
            f"{phase}_after_stop",
            auto_stop=False,
            campaign_prefixes=campaign_prefixes,
        )
    return result


def _modal_hygiene_terminal_check(check: dict[str, Any]) -> dict[str, Any]:
    terminal = check
    while isinstance(terminal.get("after_stop"), dict):
        terminal = terminal["after_stop"]
    return terminal


def _modal_hygiene_verdict(
    check: dict[str, Any],
    enforcement: str,
) -> dict[str, Any]:
    enforcement = enforcement.lower().strip()
    if enforcement not in {"off", "warn", "fail"}:
        raise ValueError("modal_hygiene_enforcement must be one of off, warn, fail")
    terminal = _modal_hygiene_terminal_check(check)
    list_status = terminal.get("list_status")
    active = terminal.get("same_campaign_active_entries") or []
    if enforcement == "off":
        status = "off"
    elif list_status != "ok":
        status = "fail" if enforcement == "fail" else "warn"
    elif active:
        status = "fail" if enforcement == "fail" else "warn"
    else:
        status = "pass"

    if list_status != "ok":
        message = "Modal hygiene could not list apps after run"
    elif active:
        descriptions = [
            str(entry.get("description") or entry.get("app_id"))
            for entry in active
        ]
        message = (
            "Modal hygiene found active same-campaign app(s): "
            + ", ".join(descriptions)
        )
    else:
        message = "Modal hygiene passed: no active same-campaign apps remain"

    return {
        "status": status,
        "enforcement": enforcement,
        "phase": terminal.get("phase"),
        "active_same_campaign_count": len(active),
        "active_same_campaign_entries": active,
        "message": message,
    }


def _local_schema_dry_run(
    requested_gpu: str,
    run_id: str | None,
    shape_csv: str,
    tilelang_variant_csv: str,
    monolithic_candidate_csv: str,
    component_records: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    configs = candidate_configs(monolithic_candidate_csv, component_records)
    shapes: list[dict[str, Any]] = []
    for shape in selected_schema_shapes(shape_csv):
        reports = []
        for config in configs:
            if config["candidate_id"] in {
                record["candidate_id"] for record in component_records
            }:
                matching = [
                    record
                    for record in filter_candidate_component_records_for_shape(
                        component_records,
                        shape,
                    )
                    if record["candidate_id"] == config["candidate_id"]
                ]
                if not matching:
                    continue
                projection = component_record_projection(matching[0])
                slot_results = _component_record_slot_results(matching[0])
                reports.append(
                    {
                        "candidate_id": config["candidate_id"],
                        "role": config["role"],
                        "status": "schema_dry_run",
                        "projection": projection,
                        "production_gate": projection["production_gate"],
                        "slot_results": slot_results,
                        "slot_summary": summarize_slot_results(slot_results),
                    }
                )
                continue
            slot_results = empty_slot_results(
                "not_run",
                "schema dry-run only; no Modal remote function was started",
            )
            reports.append(
                {
                    "candidate_id": config["candidate_id"],
                    "role": config["role"],
                    "status": "schema_dry_run",
                    "slot_results": slot_results,
                    "slot_summary": summarize_slot_results(slot_results),
                }
            )
        shapes.append(
            {
                "shape": shape.name,
                "status": "schema_dry_run",
                "ab_schema": _ab_schema_for_shape(
                    shape.to_dict(),
                    monolithic_candidate_csv,
                    component_records,
                ),
                "candidate_reports": reports,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or "schema_dry_run",
        "volume": BENCH_VOLUME_NAME,
        "volume_relpath": None,
        "device": {
            "requested_gpu_spec": requested_gpu,
            "dry_run_schema": True,
        },
        "settings": {
            "shape_csv": shape_csv,
            "tilelang_variant_csv": tilelang_variant_csv,
            "monolithic_candidate_csv": monolithic_candidate_csv,
            "candidate_component_record_count": len(component_records),
            "seed": seed,
        },
        "artifacts": {},
        "candidate_configs": configs,
        "candidate_component_records": component_records,
        "readiness_gates": readiness_gates(),
        "modal_hygiene_policy": {
            "safe_auto_stop_scope": (
                "dry-run does not invoke the remote GPU function; local hygiene "
                "may still stop the modal run ephemeral app"
            ),
            "app_name": APP_NAME,
            "same_campaign_prefix_csv": DEFAULT_MODAL_CAMPAIGN_PREFIX_CSV,
        },
        "shapes": shapes,
    }


@app.local_entrypoint()
def main(
    run_id: str | None = None,
    shape_csv: str = "smoke,productionish",
    tilelang_variant_csv: str = "baseline,stage2_bf1_bb0",
    monolithic_candidate_csv: str = "monolithic_chunk_candidate",
    candidate_record_path_csv: str = "",
    candidate_record_json: str = "",
    warmup: int = 2,
    iters: int = 6,
    cuda_warmup: int = 3,
    cuda_iters: int = 10,
    seed: int = 20260430,
    dry_run_schema: bool = False,
    print_training_ab_stub: bool = False,
    modal_hygiene: bool = True,
    modal_auto_stop: bool = True,
    modal_hygiene_enforcement: str = "warn",
    modal_campaign_prefix_csv: str = DEFAULT_MODAL_CAMPAIGN_PREFIX_CSV,
) -> None:
    component_records = load_candidate_component_records(
        candidate_record_path_csv,
        candidate_record_json,
    )
    candidate_records_json = json.dumps(component_records, sort_keys=True)

    hygiene: dict[str, Any] = {}
    campaign_prefixes = _modal_campaign_prefixes(modal_campaign_prefix_csv)
    if modal_hygiene:
        hygiene["before"] = _modal_hygiene_check(
            "before",
            auto_stop=False,
            campaign_prefixes=campaign_prefixes,
        )

    if print_training_ab_stub:
        result = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id or "training_ab_stub",
            "training_ab_stub": guarded_stage2_training_ab_stub(
                run_id=run_id or "mamba3_stage2_guarded_train_ab",
            ),
            "candidate_component_records": component_records,
        }
    elif dry_run_schema:
        result = _local_schema_dry_run(
            GPU_SPEC,
            run_id,
            shape_csv,
            tilelang_variant_csv,
            monolithic_candidate_csv,
            component_records,
            seed,
        )
    else:
        result = run_ab_remote.remote(
            GPU_SPEC,
            run_id,
            shape_csv,
            tilelang_variant_csv,
            monolithic_candidate_csv,
            candidate_records_json,
            warmup,
            iters,
            cuda_warmup,
            cuda_iters,
            seed,
        )

    hygiene_exit_message = None
    if modal_hygiene:
        hygiene["after"] = _modal_hygiene_check(
            "after",
            auto_stop=modal_auto_stop,
            campaign_prefixes=campaign_prefixes,
        )
        hygiene["verdict"] = _modal_hygiene_verdict(
            hygiene["after"],
            modal_hygiene_enforcement,
        )
        result["local_modal_hygiene"] = hygiene
        if hygiene["verdict"]["status"] == "fail":
            hygiene_exit_message = hygiene["verdict"]["message"]

    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
    if hygiene_exit_message:
        raise SystemExit(hygiene_exit_message)
