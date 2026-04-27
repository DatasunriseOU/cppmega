from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture(scope="module")
def probe() -> Any:
    pytest.importorskip("transformer_engine")
    try:
        import transformer_engine_torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"transformer_engine_torch unavailable: {type(exc).__name__}: {exc}")

    probe_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "probes"
        / "te_blockscaled_backward_probe.py"
    )
    spec = importlib.util.spec_from_file_location("test_probe_helpers_probe", probe_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rel_l2_and_max_abs_known_values(probe: Any) -> None:
    ref = torch.ones(4, 8)
    out = ref + 0.1

    assert probe._rel_l2(out, ref) == pytest.approx(0.1, rel=1e-4)
    assert probe._max_abs(out, ref) == pytest.approx(0.1, rel=1e-4)


def test_record_success_marks_bad_math_and_nonfinite(probe: Any) -> None:
    ref = torch.ones(4, 8)
    bad = ref + 1.0
    nonfinite = ref.clone()
    nonfinite[0, 0] = float("nan")

    bad_row = probe._record_success("bad", bad, ref, rel_l2_limit=0.01)
    nonfinite_row = probe._record_success("nan", nonfinite, ref)

    assert bad_row["status"] == "bad_math"
    assert bad_row["shape"] == [4, 8]
    assert nonfinite_row["finite"] is False


def test_record_failure_keeps_type_and_first_line(probe: Any) -> None:
    row = probe._record_failure("boom", RuntimeError("first line\nsecond line"))

    assert row == {
        "name": "boom",
        "status": "fail",
        "finite": None,
        "error_type": "RuntimeError",
        "error": "first line",
    }


def test_mxfp8_transpose_copy_bytes_counts_existing_payloads(probe: Any) -> None:
    tensor = Mock()
    tensor._columnwise_data.numel.return_value = 100
    tensor._columnwise_scale_inv.numel.return_value = 50

    assert probe._mxfp8_transpose_copy_bytes(tensor) == 150

    tensor._columnwise_data = None
    assert probe._mxfp8_transpose_copy_bytes(tensor) == 0


def test_mxfp8_transpose_emit_bytes_reports_bf16_read_and_scale(probe: Any) -> None:
    source = torch.randn(4, 8, dtype=torch.bfloat16)
    tensor = Mock()
    tensor._columnwise_scale_inv.numel.return_value = 12

    row = probe._mxfp8_transpose_emit_bytes(source, tensor)

    assert row == {
        "bf16_source_read_bytes": 64,
        "emitted_payload_bytes": 32,
        "scale_transpose_bytes": 12,
        "existing_mxfp8_payload_copy_bytes": 0,
    }
