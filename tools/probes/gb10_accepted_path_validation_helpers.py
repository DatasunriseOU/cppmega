"""Pure helpers for the GB10 accepted-path validation probe.

This module is intentionally side-effect free: it must not import CUDA/TE
packages, mutate env vars, or launch subprocesses.  The executable probe owns
orchestration; unit tests import this stable parser/validator surface.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any


FALLBACK_STAT_KEYS = ("bf16_fallback_dgrad", "bf16_fallback_wgrad")
ADAPTER_STAT_KEYS = ("mxfp8_tn_adapter_dgrad", "mxfp8_tn_adapter_wgrad")
CUTLASS_STAT_KEYS = ("mxfp8_cutlass_native_dgrad", "mxfp8_cutlass_native_wgrad")
FLASHINFER_STAT_KEYS = ("mxfp8_flashinfer_dgrad", "mxfp8_flashinfer_wgrad")
FLASHINFER_FPROP_STAT_KEYS = ("mxfp8_flashinfer_fprop",)
PASSTHROUGH_STAT_KEYS = ("native_passthrough_dgrad", "native_passthrough_wgrad")
GROUPED_DIRECT_STAT_KEYS = (
    "mxfp8_grouped_cutlass_native_dgrad",
    "mxfp8_grouped_cutlass_native_wgrad",
    "mxfp8_cutlass_native_grouped_dgrad",
    "mxfp8_cutlass_native_grouped_wgrad",
    "mxfp8_grouped_flashinfer_dgrad",
    "mxfp8_grouped_flashinfer_wgrad",
    "mxfp8_flashinfer_grouped_dgrad",
    "mxfp8_flashinfer_grouped_wgrad",
    "mxfp8_grouped_direct_dgrad",
    "mxfp8_grouped_direct_wgrad",
    "mxfp8_grouped_direct_miss_dgrad",
    "mxfp8_grouped_direct_miss_wgrad",
    "mxfp8_grouped_gemm_ready_dgrad",
    "mxfp8_grouped_gemm_ready_wgrad",
    "mxfp8_grouped_gemm_ready_miss_dgrad",
    "mxfp8_grouped_gemm_ready_miss_wgrad",
    "mxfp8_grouped_transpose_copy_fallback_dgrad",
    "mxfp8_grouped_transpose_copy_fallback_wgrad",
    "mxfp8_dense_gemm_ready_dgrad",
    "mxfp8_dense_gemm_ready_wgrad",
    "mxfp8_dense_copy_fallback_dgrad",
    "mxfp8_dense_copy_fallback_wgrad",
)
MATERIALIZATION_STAT_KEYS = (
    "mxfp8_tn_adapter_te_emit",
    "mxfp8_tn_adapter_te_emit_deferred",
    "mxfp8_tn_adapter_saved_transpose_operand",
    "mxfp8_tn_adapter_te_emit_swizzled",
    "mxfp8_tn_adapter_te_emit_swizzled_unavailable",
    "mxfp8_dense_grad_output_transpose_emit",
    "mxfp8_dense_grad_output_transpose_emit_failed",
    "mxfp8_tn_adapter_copy_transpose",
    "mxfp8_tn_adapter_missing_sidecar_copy",
    "mxfp8_tn_adapter_missing_sidecar_strict",
    "mxfp8_norm_quantize_sidecar_bridge",
)
SIDECAR_LIVE_ZERO_KEYS = (
    "mxfp8_tn_sidecar_registry_size",
    "mxfp8_tn_sidecar_registry_persistent",
    "mxfp8_tn_sidecar_registry_current_bytes",
    "mxfp8_tn_sidecar_tracked_attr_current_bytes",
)
SIDECAR_REGISTRY_ZERO_KEYS = SIDECAR_LIVE_ZERO_KEYS
SIDECAR_REGISTRY_STAT_KEYS = SIDECAR_LIVE_ZERO_KEYS + (
    "mxfp8_tn_sidecar_registry_peak",
    "mxfp8_tn_sidecar_registry_peak_bytes",
    "mxfp8_tn_sidecar_tracked_attr_peak_bytes",
    "mxfp8_tn_sidecar_attr_attached",
    "mxfp8_tn_sidecar_attr_cleared",
    "mxfp8_tn_sidecar_consumed",
    "mxfp8_tn_sidecar_attr_attached_bytes",
)
ALL_STAT_KEYS = (
    ADAPTER_STAT_KEYS
    + CUTLASS_STAT_KEYS
    + FLASHINFER_STAT_KEYS
    + FLASHINFER_FPROP_STAT_KEYS
    + GROUPED_DIRECT_STAT_KEYS
    + FALLBACK_STAT_KEYS
    + PASSTHROUGH_STAT_KEYS
    + MATERIALIZATION_STAT_KEYS
    + SIDECAR_REGISTRY_STAT_KEYS
)

NUMBER_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
GIB = 1024**3
MIB = 1024**2

GROUPED_DGRAD_RE = re.compile(
    r"\bMXFP8 TN adapter grouped dgrad\b.*?\bconverted_A=(?P<converted_a>\d+)/(?P<total_a>\d+)"
)
GROUPED_WGRAD_RE = re.compile(
    r"\bMXFP8 TN adapter grouped wgrad\b.*?"
    r"\bconverted_A=(?P<converted_a>\d+)/(?P<total_a>\d+)\s+"
    r"\bconverted_B=(?P<converted_b>\d+)/(?P<total_b>\d+)"
)


def _counter(counters: dict[str, int], key: str, default: int = 0) -> int:
    try:
        return int(counters.get(key, default))
    except (TypeError, ValueError):
        return default


def extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in probe output")


def validate_probe_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    stats = report.get("shim_stats")
    if not isinstance(stats, dict):
        return ["probe report is missing shim_stats"]

    for key in FALLBACK_STAT_KEYS:
        if int(stats.get(key, -1)) != 0:
            errors.append(f"{key}={stats.get(key)}; expected 0")
    adapter_used = False
    direct_cutlass_used = False
    flashinfer_used = False
    for adapter_key, cutlass_key in zip(ADAPTER_STAT_KEYS, CUTLASS_STAT_KEYS):
        adapter_count = int(stats.get(adapter_key, 0))
        flashinfer_key = FLASHINFER_STAT_KEYS[ADAPTER_STAT_KEYS.index(adapter_key)]
        cutlass_count = int(stats.get(cutlass_key, 0))
        flashinfer_count = int(stats.get(flashinfer_key, 0))
        adapter_used = adapter_used or adapter_count > 0
        direct_cutlass_used = direct_cutlass_used or cutlass_count > 0
        flashinfer_used = flashinfer_used or flashinfer_count > 0
        if adapter_count <= 0 and cutlass_count <= 0 and flashinfer_count <= 0:
            errors.append(
                f"{adapter_key}={stats.get(adapter_key)} and "
                f"{cutlass_key}={stats.get(cutlass_key)} and "
                f"{flashinfer_key}={stats.get(flashinfer_key)}; expected one >0"
            )
    for key in PASSTHROUGH_STAT_KEYS:
        if int(stats.get(key, -1)) != 0:
            errors.append(f"{key}={stats.get(key)}; expected 0")
    if stats.get("fallback_reasons", {}) not in ({}, None):
        errors.append(f"fallback_reasons={stats.get('fallback_reasons')!r}; expected empty")
    for key in SIDECAR_LIVE_ZERO_KEYS:
        if key not in stats:
            errors.append(f"{key} missing; expected 0")
        elif int(stats.get(key, -1)) != 0:
            errors.append(f"{key}={stats.get(key)}; expected 0")
    if direct_cutlass_used and not adapter_used and not flashinfer_used:
        for key in MATERIALIZATION_STAT_KEYS:
            if int(stats.get(key, 0)) != 0:
                errors.append(f"{key}={stats.get(key)}; expected 0 for cutlass_native")
        if int(stats.get("mxfp8_tn_sidecar_registry_peak", -1)) != 0:
            errors.append(
                "mxfp8_tn_sidecar_registry_peak="
                f"{stats.get('mxfp8_tn_sidecar_registry_peak')}; expected 0 for cutlass_native"
            )

    results = {
        row.get("name"): row
        for row in report.get("results", [])
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    for name in ("mxfp8_dgrad_shim_NN_to_TN", "mxfp8_wgrad_shim_NT_to_TN"):
        row = results.get(name)
        if row is None:
            errors.append(f"missing probe result {name}")
        elif row.get("status") != "pass":
            errors.append(f"{name} status={row.get('status')!r}; expected pass")
        elif row.get("finite") is False:
            errors.append(f"{name} finite=False; expected finite output")
    return errors


def _parse_grouped_adapter_events(text: str) -> list[dict[str, int | str]]:
    events: list[dict[str, int | str]] = []
    for line in text.splitlines():
        match = GROUPED_DGRAD_RE.search(line)
        if match:
            converted_a = int(match.group("converted_a"))
            total_a = int(match.group("total_a"))
            events.append(
                {
                    "kind": "dgrad",
                    "converted_a": converted_a,
                    "total_a": total_a,
                    "converted_b": 0,
                    "total_b": 0,
                    "fallback_copies": converted_a,
                }
            )
            continue
        match = GROUPED_WGRAD_RE.search(line)
        if match:
            converted_a = int(match.group("converted_a"))
            converted_b = int(match.group("converted_b"))
            events.append(
                {
                    "kind": "wgrad",
                    "converted_a": converted_a,
                    "total_a": int(match.group("total_a")),
                    "converted_b": converted_b,
                    "total_b": int(match.group("total_b")),
                    "fallback_copies": converted_a + converted_b,
                }
            )
    return events


def _discover_experts(text: str, grouped_events: list[dict[str, int | str]]) -> dict[str, Any]:
    declared: int | None = None
    declared_patterns = (
        r"--num-experts\s+(\d+)",
        r"\bnum[_-]?experts\s*[:=]\s*(\d+)",
        r"\bnum[_-]?moe[_-]?experts\s*[:=]\s*(\d+)",
        r"\bnum\s+experts\s*[:=]\s*(\d+)",
    )
    for pattern in declared_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            declared = int(matches[-1])
    observed = sorted(
        {
            int(value)
            for event in grouped_events
            for value in (event.get("total_a"), event.get("total_b"))
            if isinstance(value, int) and value > 0
        }
    )
    used_for_inference: int | None = None
    if len(observed) == 1:
        used_for_inference = observed[0]
    elif declared is not None:
        used_for_inference = declared
    return {
        "declared": declared,
        "observed_group_sizes": observed,
        "used_for_inference": used_for_inference,
    }


def _parse_max_cuda_allocated_bytes(text: str) -> int | None:
    values: list[int] = []
    mem_profile_re = re.compile(rf"\bmax_alloc=(?P<max_alloc>{NUMBER_RE})\s+GiB")
    rank_memory_re = re.compile(
        rf"\bmemory \(MB\).*?\bmax allocated:\s*(?P<max_alloc>{NUMBER_RE})\b"
    )
    byte_re = re.compile(
        r"\b(?:max_cuda_allocated|max_memory_allocated)"
        r"(?:_bytes)?\s*[:=]\s*(?P<bytes>\d+)\b"
    )
    for match in mem_profile_re.finditer(text):
        values.append(int(round(float(match.group("max_alloc")) * GIB)))
    for match in rank_memory_re.finditer(text):
        values.append(int(round(float(match.group("max_alloc")) * MIB)))
    for match in byte_re.finditer(text):
        values.append(int(match.group("bytes")))
    return max(values) if values else None


def _build_mxfp8_copy_breakdown(text: str, counters: dict[str, int]) -> dict[str, Any]:
    grouped_events = _parse_grouped_adapter_events(text)
    experts = _discover_experts(text, grouped_events)

    grouped_dgrad_calls = sum(1 for event in grouped_events if event["kind"] == "dgrad")
    grouped_wgrad_calls = sum(1 for event in grouped_events if event["kind"] == "wgrad")
    if not grouped_events:
        grouped_dgrad_calls = _counter(
            counters,
            "mxfp8_grouped_transpose_copy_fallback_dgrad",
        )
        grouped_wgrad_calls = _counter(
            counters,
            "mxfp8_grouped_transpose_copy_fallback_wgrad",
        )
    grouped_dgrad_copies = sum(
        int(event["fallback_copies"]) for event in grouped_events if event["kind"] == "dgrad"
    )
    grouped_wgrad_copies = sum(
        int(event["fallback_copies"]) for event in grouped_events if event["kind"] == "wgrad"
    )

    estimate_source = "grouped_debug_converted_counts" if grouped_events else "none"
    if not grouped_events and experts["used_for_inference"] is not None:
        estimate_source = "experts_inferred"
        grouped_dgrad_copies = grouped_dgrad_calls * int(experts["used_for_inference"])
        grouped_wgrad_copies = grouped_wgrad_calls * int(experts["used_for_inference"]) * 2

    adapter_dgrad_hits = _counter(counters, "mxfp8_tn_adapter_dgrad")
    adapter_wgrad_hits = _counter(counters, "mxfp8_tn_adapter_wgrad")
    dense_adapter_dgrad_hits = max(adapter_dgrad_hits - grouped_dgrad_calls, 0)
    dense_adapter_wgrad_hits = max(adapter_wgrad_hits - grouped_wgrad_calls, 0)
    dense_current_hits = dense_adapter_dgrad_hits + dense_adapter_wgrad_hits
    dense_expected_fallback_copies = dense_adapter_dgrad_hits + (2 * dense_adapter_wgrad_hits)

    grouped_fallback_copies = grouped_dgrad_copies + grouped_wgrad_copies
    total_copy_transpose = (
        _counter(counters, "mxfp8_tn_adapter_copy_transpose")
        if "mxfp8_tn_adapter_copy_transpose" in counters
        else None
    )
    if total_copy_transpose is None:
        dense_fallback_copies = dense_expected_fallback_copies
        unattributed_copies: int | None = None
        dense_copy_source = "adapter_hit_estimate"
    else:
        dense_fallback_copies = max(total_copy_transpose - grouped_fallback_copies, 0)
        unattributed_copies = total_copy_transpose - grouped_fallback_copies - dense_expected_fallback_copies
        dense_copy_source = "copy_transpose_counter_minus_grouped"

    dense_dgrad_direct_hits = _counter(counters, "mxfp8_cutlass_native_dgrad") + _counter(
        counters, "mxfp8_flashinfer_dgrad"
    )
    dense_wgrad_direct_hits = _counter(counters, "mxfp8_cutlass_native_wgrad") + _counter(
        counters, "mxfp8_flashinfer_wgrad"
    )
    grouped_dgrad_direct_hits = sum(
        _counter(counters, key)
        for key in (
            "mxfp8_grouped_cutlass_native_dgrad",
            "mxfp8_cutlass_native_grouped_dgrad",
            "mxfp8_grouped_flashinfer_dgrad",
            "mxfp8_flashinfer_grouped_dgrad",
            "mxfp8_grouped_direct_dgrad",
            "mxfp8_grouped_gemm_ready_dgrad",
        )
    )
    grouped_wgrad_direct_hits = sum(
        _counter(counters, key)
        for key in (
            "mxfp8_grouped_cutlass_native_wgrad",
            "mxfp8_cutlass_native_grouped_wgrad",
            "mxfp8_grouped_flashinfer_wgrad",
            "mxfp8_flashinfer_grouped_wgrad",
            "mxfp8_grouped_direct_wgrad",
            "mxfp8_grouped_gemm_ready_wgrad",
        )
    )

    sidecar_peak_bytes = (
        _counter(counters, "mxfp8_tn_sidecar_registry_peak_bytes")
        if "mxfp8_tn_sidecar_registry_peak_bytes" in counters
        else None
    )
    return {
        "total_copy_transpose": total_copy_transpose,
        "copy_counts_by_source": {
            "grouped_dgrad": grouped_dgrad_copies,
            "grouped_wgrad": grouped_wgrad_copies,
            "grouped_total": grouped_fallback_copies,
            "dense_likely": dense_fallback_copies,
            "dense_expected_from_adapter_hits": dense_expected_fallback_copies,
            "unattributed_vs_adapter_hit_model": unattributed_copies,
        },
        "grouped": {
            "current_hits": grouped_dgrad_calls + grouped_wgrad_calls,
            "direct_hits": grouped_dgrad_direct_hits + grouped_wgrad_direct_hits,
            "fallback_copies": grouped_fallback_copies,
            "dgrad_current_hits": grouped_dgrad_calls,
            "wgrad_current_hits": grouped_wgrad_calls,
            "dgrad_direct_hits": grouped_dgrad_direct_hits,
            "wgrad_direct_hits": grouped_wgrad_direct_hits,
            "debug_events": grouped_events,
        },
        "dense": {
            "current_hits": dense_current_hits,
            "direct_hits": dense_dgrad_direct_hits + dense_wgrad_direct_hits,
            "fallback_copies": dense_fallback_copies,
            "dgrad_current_hits": dense_adapter_dgrad_hits,
            "wgrad_current_hits": dense_adapter_wgrad_hits,
            "dgrad_direct_hits": dense_dgrad_direct_hits,
            "wgrad_direct_hits": dense_wgrad_direct_hits,
        },
        "experts": experts,
        "sidecar_registry_peak_bytes": sidecar_peak_bytes,
        "max_cuda_allocated_bytes": _parse_max_cuda_allocated_bytes(text),
        "estimate_source": estimate_source,
        "dense_copy_source": dense_copy_source,
    }


def _build_profile_readiness(
    counters: dict[str, int], copy_breakdown: dict[str, Any]
) -> dict[str, Any]:
    missing: list[str] = []
    if not counters:
        missing.append("mxfp8_counters")
    if copy_breakdown.get("total_copy_transpose") is None:
        missing.append("mxfp8_tn_adapter_copy_transpose")
    if copy_breakdown.get("sidecar_registry_peak_bytes") is None:
        missing.append("mxfp8_tn_sidecar_registry_peak_bytes")
    if copy_breakdown.get("max_cuda_allocated_bytes") is None:
        missing.append("max_cuda_allocated")
    return {
        "ready": not missing,
        "missing": missing,
        "direct_backend_observed": (
            int(copy_breakdown["dense"]["direct_hits"]) > 0
            or int(copy_breakdown["grouped"]["direct_hits"]) > 0
        ),
        "grouped_breakdown_observed": int(copy_breakdown["grouped"]["current_hits"]) > 0,
        "copy_breakdown_estimate_source": copy_breakdown.get("estimate_source"),
    }


def parse_training_log(text: str) -> dict[str, Any]:
    losses: dict[str, list[float]] = {"lm": [], "mtp_1": []}
    for key in list(losses):
        for pattern in (
            rf"\b{re.escape(key)}\s+loss\s*[:=]\s*({NUMBER_RE})",
            rf"\bloss:\s*.*?\b{re.escape(key)}\s+({NUMBER_RE})",
        ):
            losses[key].extend(float(match) for match in re.findall(pattern, text))
    losses = {key: values for key, values in losses.items() if values}

    counters: dict[str, int] = {}
    fallback_reasons: dict[str, Any] | None = None
    marker = "TE block-scaled backward stats:"
    for line in text.splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            parsed = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        for key in ALL_STAT_KEYS:
            if key in parsed:
                counters[key] = int(parsed[key])
        reasons = parsed.get("fallback_reasons")
        if isinstance(reasons, dict):
            fallback_reasons = reasons

    for key in ALL_STAT_KEYS:
        matches = re.findall(rf"\b{re.escape(key)}\s*=\s*(\d+)\b", text)
        if matches:
            counters[key] = int(matches[-1])
    if fallback_reasons is None and re.search(r"\bfallback_reasons\s*=\s*\{\s*\}", text):
        fallback_reasons = {}

    copy_breakdown = _build_mxfp8_copy_breakdown(text, counters)
    profile_readiness = _build_profile_readiness(counters, copy_breakdown)

    return {
        "losses": losses,
        "counters": counters,
        "fallback_reasons": fallback_reasons,
        "mxfp8_copy_breakdown": copy_breakdown,
        "profile_readiness": profile_readiness,
    }
