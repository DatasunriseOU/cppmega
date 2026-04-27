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
MATERIALIZATION_STAT_KEYS = (
    "mxfp8_tn_adapter_te_emit",
    "mxfp8_tn_adapter_te_emit_swizzled",
    "mxfp8_tn_adapter_te_emit_swizzled_unavailable",
    "mxfp8_tn_adapter_copy_transpose",
    "mxfp8_tn_adapter_missing_sidecar_copy",
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
    + FALLBACK_STAT_KEYS
    + PASSTHROUGH_STAT_KEYS
    + MATERIALIZATION_STAT_KEYS
    + SIDECAR_REGISTRY_STAT_KEYS
)

NUMBER_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


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
    elif int(stats.get("mxfp8_tn_sidecar_registry_peak", 0)) <= 0:
        errors.append(
            "mxfp8_tn_sidecar_registry_peak="
            f"{stats.get('mxfp8_tn_sidecar_registry_peak')}; expected >0"
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

    return {"losses": losses, "counters": counters, "fallback_reasons": fallback_reasons}
