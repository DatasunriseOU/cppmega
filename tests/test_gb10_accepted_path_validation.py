from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from tools.probes.gb10_accepted_path_validation_helpers import (
    extract_first_json_object,
    parse_training_log,
    validate_probe_report,
)


ROOT = Path(__file__).resolve().parents[1]
PROBE_CLI = ROOT / "tools" / "probes" / "gb10_accepted_path_validation.py"


def _run_probe_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = [str(ROOT), str(ROOT / "scripts"), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath if part)
    return subprocess.run(
        [sys.executable, str(PROBE_CLI), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def test_probe_json_parser_validates_zero_bf16_fallbacks():
    stdout = """
[cppmega_fp8_shim] TE block-scaled backward override installed
{
  "results": [
    {"name": "mxfp8_dgrad_shim_NN_to_TN", "status": "pass"},
    {"name": "mxfp8_wgrad_shim_NT_to_TN", "status": "pass"}
  ],
  "shim_stats": {
    "mxfp8_tn_adapter_dgrad": 1,
	    "mxfp8_tn_adapter_wgrad": 1,
	    "bf16_fallback_dgrad": 0,
	    "bf16_fallback_wgrad": 0,
	    "native_passthrough_dgrad": 0,
	    "native_passthrough_wgrad": 0,
	    "mxfp8_tn_sidecar_registry_size": 0,
	    "mxfp8_tn_sidecar_registry_persistent": 0,
	    "mxfp8_tn_sidecar_registry_peak": 3,
	    "mxfp8_tn_sidecar_registry_current_bytes": 0,
	    "mxfp8_tn_sidecar_registry_peak_bytes": 49152,
	    "mxfp8_tn_sidecar_tracked_attr_current_bytes": 0,
	    "mxfp8_tn_sidecar_tracked_attr_peak_bytes": 49152,
	    "mxfp8_tn_sidecar_attr_attached": 3,
	    "mxfp8_tn_sidecar_attr_cleared": 3,
	    "mxfp8_tn_sidecar_consumed": 3,
	    "mxfp8_tn_sidecar_attr_attached_bytes": 49152,
	    "fallback_reasons": {}
	  }
	}
[cppmega_fp8_shim] TE block-scaled backward stats: {'mxfp8_tn_adapter_dgrad': 1}
"""
    report = extract_first_json_object(stdout)

    assert validate_probe_report(report) == []


def test_probe_json_parser_accepts_cutlass_native_backend():
    stdout = """
{
  "results": [
    {"name": "mxfp8_dgrad_shim_NN_to_TN", "status": "pass"},
    {"name": "mxfp8_wgrad_shim_NT_to_TN", "status": "pass"}
  ],
	  "shim_stats": {
	    "mxfp8_tn_adapter_dgrad": 0,
	    "mxfp8_tn_adapter_wgrad": 0,
            "mxfp8_cutlass_native_dgrad": 1,
            "mxfp8_cutlass_native_wgrad": 1,
            "mxfp8_tn_adapter_te_emit": 0,
            "mxfp8_tn_adapter_te_emit_swizzled": 0,
            "mxfp8_tn_adapter_te_emit_swizzled_unavailable": 0,
            "mxfp8_tn_adapter_copy_transpose": 0,
	    "mxfp8_tn_adapter_missing_sidecar_copy": 0,
	    "mxfp8_norm_quantize_sidecar_bridge": 0,
	    "bf16_fallback_dgrad": 0,
	    "bf16_fallback_wgrad": 0,
	    "native_passthrough_dgrad": 0,
	    "native_passthrough_wgrad": 0,
	    "mxfp8_tn_sidecar_registry_size": 0,
	    "mxfp8_tn_sidecar_registry_persistent": 0,
	    "mxfp8_tn_sidecar_registry_peak": 0,
	    "mxfp8_tn_sidecar_registry_current_bytes": 0,
	    "mxfp8_tn_sidecar_registry_peak_bytes": 0,
	    "mxfp8_tn_sidecar_tracked_attr_current_bytes": 0,
	    "mxfp8_tn_sidecar_tracked_attr_peak_bytes": 0,
	    "mxfp8_tn_sidecar_attr_attached": 0,
	    "mxfp8_tn_sidecar_attr_cleared": 0,
	    "mxfp8_tn_sidecar_consumed": 0,
	    "mxfp8_tn_sidecar_attr_attached_bytes": 0,
	    "fallback_reasons": {}
	  }
	}
"""
    report = extract_first_json_object(stdout)

    assert validate_probe_report(report) == []


def test_probe_json_parser_accepts_flashinfer_cutlass_backend():
    stdout = """
{
  "results": [
    {"name": "mxfp8_dgrad_shim_NN_to_TN", "status": "pass"},
    {"name": "mxfp8_wgrad_shim_NT_to_TN", "status": "pass"}
  ],
	  "shim_stats": {
	    "mxfp8_tn_adapter_dgrad": 0,
	    "mxfp8_tn_adapter_wgrad": 0,
            "mxfp8_cutlass_native_dgrad": 0,
            "mxfp8_cutlass_native_wgrad": 0,
            "mxfp8_flashinfer_dgrad": 1,
            "mxfp8_flashinfer_wgrad": 1,
            "mxfp8_flashinfer_fprop": 1,
            "mxfp8_tn_adapter_te_emit": 3,
            "mxfp8_tn_adapter_te_emit_swizzled": 3,
            "mxfp8_tn_adapter_te_emit_swizzled_unavailable": 0,
            "mxfp8_tn_adapter_copy_transpose": 0,
	    "mxfp8_tn_adapter_missing_sidecar_copy": 0,
	    "mxfp8_norm_quantize_sidecar_bridge": 0,
	    "bf16_fallback_dgrad": 0,
	    "bf16_fallback_wgrad": 0,
	    "native_passthrough_dgrad": 0,
	    "native_passthrough_wgrad": 0,
	    "mxfp8_tn_sidecar_registry_size": 0,
	    "mxfp8_tn_sidecar_registry_persistent": 0,
	    "mxfp8_tn_sidecar_registry_peak": 3,
	    "mxfp8_tn_sidecar_registry_current_bytes": 0,
	    "mxfp8_tn_sidecar_registry_peak_bytes": 1402368,
	    "mxfp8_tn_sidecar_tracked_attr_current_bytes": 0,
	    "mxfp8_tn_sidecar_tracked_attr_peak_bytes": 1402368,
	    "mxfp8_tn_sidecar_attr_attached": 3,
	    "mxfp8_tn_sidecar_attr_cleared": 3,
	    "mxfp8_tn_sidecar_consumed": 3,
	    "mxfp8_tn_sidecar_attr_attached_bytes": 1402368,
	    "fallback_reasons": {}
	  }
	}
"""
    report = extract_first_json_object(stdout)

    assert validate_probe_report(report) == []


def test_probe_json_parser_accepts_flashinfer_zero_sidecar_backend():
    stdout = """
{
  "results": [
    {"name": "mxfp8_dgrad_shim_NN_to_TN", "status": "pass"},
    {"name": "mxfp8_wgrad_shim_NT_to_TN", "status": "pass"}
  ],
  "shim_stats": {
    "mxfp8_tn_adapter_dgrad": 0,
    "mxfp8_tn_adapter_wgrad": 0,
    "mxfp8_cutlass_native_dgrad": 0,
    "mxfp8_cutlass_native_wgrad": 0,
    "mxfp8_flashinfer_dgrad": 1,
    "mxfp8_flashinfer_wgrad": 1,
    "mxfp8_tn_adapter_te_emit": 0,
    "mxfp8_tn_adapter_te_emit_deferred": 4,
    "mxfp8_tn_adapter_saved_transpose_operand": 3,
    "mxfp8_tn_adapter_te_emit_swizzled": 0,
    "mxfp8_tn_adapter_te_emit_swizzled_unavailable": 0,
    "mxfp8_tn_adapter_copy_transpose": 0,
    "mxfp8_tn_adapter_missing_sidecar_copy": 0,
    "mxfp8_norm_quantize_sidecar_bridge": 0,
    "bf16_fallback_dgrad": 0,
    "bf16_fallback_wgrad": 0,
    "native_passthrough_dgrad": 0,
    "native_passthrough_wgrad": 0,
    "mxfp8_tn_sidecar_registry_size": 0,
    "mxfp8_tn_sidecar_registry_persistent": 0,
    "mxfp8_tn_sidecar_registry_peak": 0,
    "mxfp8_tn_sidecar_registry_current_bytes": 0,
    "mxfp8_tn_sidecar_registry_peak_bytes": 0,
    "mxfp8_tn_sidecar_tracked_attr_current_bytes": 0,
    "mxfp8_tn_sidecar_tracked_attr_peak_bytes": 0,
    "mxfp8_tn_sidecar_attr_attached": 0,
    "mxfp8_tn_sidecar_attr_cleared": 0,
    "mxfp8_tn_sidecar_consumed": 0,
    "mxfp8_tn_sidecar_attr_attached_bytes": 0,
    "fallback_reasons": {}
  }
}
"""
    report = extract_first_json_object(stdout)

    assert validate_probe_report(report) == []


def test_training_log_parser_accepts_loss_and_counter_formats():
    log = """
	iteration 1 | lm loss: 1.165876E+01 | mtp_1 loss: 1.164849E+01
		[cppmega_fp8_shim] TE block-scaled backward stats: {'mxfp8_tn_adapter_dgrad': 6, 'mxfp8_tn_adapter_wgrad': 6, 'bf16_fallback_dgrad': 0, 'bf16_fallback_wgrad': 0, 'native_passthrough_dgrad': 0, 'native_passthrough_wgrad': 0, 'mxfp8_tn_sidecar_registry_size': 0, 'mxfp8_tn_sidecar_registry_persistent': 0, 'mxfp8_tn_sidecar_registry_peak': 18, 'mxfp8_tn_sidecar_registry_current_bytes': 0, 'mxfp8_tn_sidecar_registry_peak_bytes': 294912, 'mxfp8_tn_sidecar_tracked_attr_current_bytes': 0, 'mxfp8_tn_sidecar_tracked_attr_peak_bytes': 294912, 'mxfp8_tn_sidecar_attr_attached': 18, 'mxfp8_tn_sidecar_attr_cleared': 18, 'mxfp8_tn_sidecar_consumed': 18, 'mxfp8_tn_sidecar_attr_attached_bytes': 294912, 'fallback_reasons': {}}
		mxfp8_tn_adapter_dgrad=6
		mxfp8_tn_adapter_wgrad=6
		bf16_fallback_dgrad=0
		bf16_fallback_wgrad=0
		mxfp8_tn_sidecar_registry_size=0
		mxfp8_tn_sidecar_registry_current_bytes=0
		mxfp8_tn_sidecar_tracked_attr_current_bytes=0
		fallback_reasons={}
	"""

    parsed = parse_training_log(log)

    assert parsed["losses"]["lm"] == [11.65876]
    assert parsed["losses"]["mtp_1"] == [11.64849]
    assert parsed["counters"]["mxfp8_tn_adapter_dgrad"] == 6
    assert parsed["counters"]["mxfp8_tn_adapter_wgrad"] == 6
    assert parsed["counters"]["bf16_fallback_dgrad"] == 0
    assert parsed["counters"]["bf16_fallback_wgrad"] == 0
    assert parsed["counters"]["mxfp8_tn_sidecar_registry_size"] == 0
    assert parsed["counters"]["mxfp8_tn_sidecar_registry_peak"] == 18
    assert parsed["counters"]["mxfp8_tn_sidecar_registry_current_bytes"] == 0
    assert parsed["counters"]["mxfp8_tn_sidecar_tracked_attr_current_bytes"] == 0
    assert parsed["fallback_reasons"] == {}


def test_training_log_parser_reports_mxfp8_copy_breakdown_and_profile_readiness():
    log = """
        args: --num-experts 4 --moe-grouped-gemm
        [cppmega_fp8_shim] MXFP8 TN adapter grouped dgrad layout=NN->TN converted_A=4/4
        [cppmega_fp8_shim] MXFP8 TN adapter grouped wgrad layout=NT->TN converted_A=4/4 converted_B=4/4
        [cppmega_fp8_shim] TE block-scaled backward stats: {'mxfp8_tn_adapter_dgrad': 3, 'mxfp8_tn_adapter_wgrad': 2, 'mxfp8_flashinfer_dgrad': 5, 'mxfp8_flashinfer_wgrad': 5, 'mxfp8_tn_adapter_copy_transpose': 16, 'mxfp8_tn_sidecar_registry_peak_bytes': 123456, 'fallback_reasons': {}}
        [Rank 0] (after 1 iterations) memory (MB) | allocated: 1024.00 | max allocated: 2048.00 | reserved: 3072.00 | max reserved: 4096.00
    """

    parsed = parse_training_log(log)
    breakdown = parsed["mxfp8_copy_breakdown"]

    assert breakdown["total_copy_transpose"] == 16
    assert breakdown["copy_counts_by_source"]["grouped_dgrad"] == 4
    assert breakdown["copy_counts_by_source"]["grouped_wgrad"] == 8
    assert breakdown["grouped"]["current_hits"] == 2
    assert breakdown["grouped"]["fallback_copies"] == 12
    assert breakdown["grouped"]["direct_hits"] == 0
    assert breakdown["dense"]["current_hits"] == 3
    assert breakdown["dense"]["direct_hits"] == 10
    assert breakdown["dense"]["fallback_copies"] == 4
    assert breakdown["sidecar_registry_peak_bytes"] == 123456
    assert breakdown["max_cuda_allocated_bytes"] == 2048 * 1024 * 1024
    assert breakdown["experts"]["declared"] == 4
    assert breakdown["experts"]["observed_group_sizes"] == [4]
    assert parsed["profile_readiness"] == {
        "ready": True,
        "missing": [],
        "direct_backend_observed": True,
        "grouped_breakdown_observed": True,
        "copy_breakdown_estimate_source": "grouped_debug_converted_counts",
    }


def test_training_log_parser_reports_grouped_direct_routing_counters():
    stats = {
        "mxfp8_grouped_direct_dgrad": 2,
        "mxfp8_grouped_direct_wgrad": 3,
        "mxfp8_grouped_direct_miss_dgrad": 1,
        "mxfp8_grouped_transpose_copy_fallback_dgrad": 1,
        "mxfp8_tn_adapter_copy_transpose": 4,
        "mxfp8_tn_sidecar_registry_peak_bytes": 0,
        "fallback_reasons": {"unsupported_shape": 1},
    }
    log = """
        [cppmega_fp8_shim] TE block-scaled backward stats: {stats}
    """.format(stats=stats)

    parsed = parse_training_log(log)

    assert parsed["counters"]["mxfp8_grouped_direct_dgrad"] == 2
    assert parsed["counters"]["mxfp8_grouped_direct_wgrad"] == 3
    assert parsed["counters"]["mxfp8_grouped_direct_miss_dgrad"] == 1
    assert parsed["fallback_reasons"] == {"unsupported_shape": 1}
    assert parsed["mxfp8_copy_breakdown"]["grouped"]["direct_hits"] == 5
    assert parsed["mxfp8_copy_breakdown"]["grouped"]["current_hits"] == 1


def test_probe_cli_smoke_checks_deprecated_paths_fail_closed_without_ack():
    proc = _run_probe_cli("--skip-mxfp8-probe")
    output = f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    assert proc.returncode == 0, output
    report = extract_first_json_object(proc.stdout)
    checks = {check["name"]: check for check in report["checks"]}
    fail_closed_checks = {
        "old_mxfp8_bf16_bridge_requires_ack",
        "dsa_gather_scatter_requires_ack",
        "fp8_activation_legacy_requires_ack",
    }

    assert report["status"] == "pass"
    assert fail_closed_checks <= checks.keys()
    assert {checks[name]["status"] for name in fail_closed_checks} == {"pass"}
    assert checks["mxfp8_tn_adapter_probe"]["status"] == "skip"
