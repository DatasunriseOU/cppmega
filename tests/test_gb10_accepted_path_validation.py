from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "tools" / "probes" / "gb10_accepted_path_validation.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("gb10_accepted_path_validation", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe_module()


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
    "fallback_reasons": {}
  }
}
[cppmega_fp8_shim] TE block-scaled backward stats: {'mxfp8_tn_adapter_dgrad': 1}
"""
    report = probe.extract_first_json_object(stdout)

    assert probe.validate_probe_report(report) == []


def test_training_log_parser_accepts_loss_and_counter_formats():
    log = """
iteration 1 | lm loss: 1.165876E+01 | mtp_1 loss: 1.164849E+01
[cppmega_fp8_shim] TE block-scaled backward stats: {'mxfp8_tn_adapter_dgrad': 6, 'mxfp8_tn_adapter_wgrad': 6, 'bf16_fallback_dgrad': 0, 'bf16_fallback_wgrad': 0, 'native_passthrough_dgrad': 0, 'native_passthrough_wgrad': 0, 'fallback_reasons': {}}
mxfp8_tn_adapter_dgrad=6
mxfp8_tn_adapter_wgrad=6
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
fallback_reasons={}
"""

    parsed = probe.parse_training_log(log)

    assert parsed["losses"]["lm"] == [11.65876]
    assert parsed["losses"]["mtp_1"] == [11.64849]
    assert parsed["counters"]["mxfp8_tn_adapter_dgrad"] == 6
    assert parsed["counters"]["mxfp8_tn_adapter_wgrad"] == 6
    assert parsed["counters"]["bf16_fallback_dgrad"] == 0
    assert parsed["counters"]["bf16_fallback_wgrad"] == 0
    assert parsed["fallback_reasons"] == {}


def test_deprecated_paths_fail_closed_without_ack():
    checks = [
        probe.check_mxfp8_bf16_bridge_gate(),
        probe.check_dsa_gather_scatter_gate(),
        probe.check_fp8_activation_legacy_gate(),
    ]

    assert [check.status for check in checks] == ["pass", "pass", "pass"]
