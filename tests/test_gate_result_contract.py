"""Fail-closed contracts shared by the Wave32 remote gate harnesses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cppmega.megatron.gate_result_contract import (
    require_successful_steps,
    require_variant_rows,
)


def test_historical_empty_gate_receipt_is_rejected() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "mamba3_wave32_h200_20step_gate"
        / "wave32_h200_fa3_prod_gate_v2_20260430"
        / "result.json"
    )
    result = json.loads(artifact.read_text())
    assert result["status"] == "IMAGE_BACKEND_BLOCKED"
    assert result["variants"] == []

    with pytest.raises(RuntimeError, match="empty summary.*IMAGE_BACKEND_BLOCKED"):
        require_variant_rows(result)


def test_nonempty_gate_receipt_is_accepted() -> None:
    require_variant_rows(
        {
            "run_id": "ok_run",
            "variants": [{"variant": "grouped_head_bwd_baseline"}],
        }
    )


def test_failed_applier_step_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="'gated': 1"):
        require_successful_steps({"noop": 0, "gated": 1, "rollback": 0})


def test_successful_applier_steps_are_accepted() -> None:
    require_successful_steps({"noop": 0, "gated": 0, "rollback": 0})
