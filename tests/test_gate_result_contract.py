"""Fail-closed contracts shared by the Wave32 remote gate harnesses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cppmega.megatron.gate_result_contract import (
    require_successful_steps,
    require_successful_training_variants,
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


def _training_variant(name: str, *, steps: int = 20, grad_norm: float = 1.0) -> dict:
    return {
        "variant": name,
        "run": {"status": "ok", "returncode": 0},
        "metrics": {
            "iterations_seen": steps,
            "lm_losses": [1.0],
            "grad_norms": [grad_norm],
            "nonfinite_lm_loss_count": 0,
            "nonfinite_mtp_loss_count": 0,
            "nonfinite_grad_norm_count": 0,
            "max_nan_iterations": 0,
        },
    }


def test_complete_finite_training_receipt_is_accepted() -> None:
    names = ("baseline", "candidate")
    require_successful_training_variants(
        {"variants": [_training_variant(name) for name in names]},
        expected_variants=names,
        minimum_steps=20,
    )


@pytest.mark.parametrize(
    ("variants", "expected_variants", "message"),
    [
        (
            [_training_variant("baseline")],
            ("baseline", "candidate"),
            "candidate: missing",
        ),
        ([_training_variant("baseline", steps=19)], ("baseline",), "steps=19 < 20"),
        (
            [
                {
                    **_training_variant("baseline", grad_norm=float("nan")),
                    "metrics": {
                        **_training_variant("baseline")["metrics"],
                        "grad_norms": [float("nan")],
                    },
                }
            ],
            ("baseline",),
            "non-finite grad_norms",
        ),
        (
            [
                {
                    **_training_variant("baseline"),
                    "metrics": {
                        **_training_variant("baseline")["metrics"],
                        "max_nan_iterations": None,
                    },
                }
            ],
            ("baseline",),
            "max_nan_iterations=None",
        ),
    ],
)
def test_incomplete_or_nonfinite_training_receipt_is_rejected(
    variants: list[dict], expected_variants: tuple[str, ...], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        require_successful_training_variants(
            {"variants": variants},
            expected_variants=expected_variants,
            minimum_steps=20,
        )


def test_duplicate_training_variant_cannot_overwrite_failure() -> None:
    with pytest.raises(RuntimeError, match="baseline: duplicate"):
        require_successful_training_variants(
            {
                "variants": [
                    {
                        **_training_variant("baseline"),
                        "run": {"status": "failed", "returncode": 1},
                    },
                    _training_variant("baseline"),
                ]
            },
            expected_variants=("baseline",),
            minimum_steps=20,
        )
