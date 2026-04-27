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
