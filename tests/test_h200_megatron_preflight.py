from __future__ import annotations

import json

import pytest

from scripts.h200_megatron_preflight import (
    _profile_environment,
    _write_wrappers,
    write_training_loss_receipt,
)


def test_dense_preflight_uses_derived_capacity_and_zero_dsa_auxiliary():
    environment = _profile_environment(
        sequence_length=8192,
        micro_batch_size=1,
        fp8_recipe="off",
        graph_max_edges=713,
        graph_max_chunks=419,
        enable_dsa_patch=False,
    )

    assert environment["CPPMEGA_GRAPH_MAX_EDGES"] == "713"
    assert environment["CPPMEGA_GRAPH_MAX_CHUNKS"] == "419"
    assert environment["CPPMEGA_DSA_PATCH_ENABLED"] == "0"
    assert environment["CPPMEGA_DSA_GRAPH_AUX_ENABLED"] == "0"
    assert environment["CPPMEGA_DSA_GRAPH_AUX_WEIGHT"] == "0"
    assert environment["CPPMEGA_DSA_INDEXER_LOSS_COEFF"] == "0"
    assert environment["CPPMEGA_DSA_SKIP_INDEXER_LOSS"] == "1"


def test_preflight_wrapper_only_applies_dsa_patch_after_explicit_opt_in(tmp_path):
    wrapper = _write_wrappers(tmp_path)

    source = wrapper.read_text(encoding="utf-8")
    gate = "os.environ.get('CPPMEGA_DSA_PATCH_ENABLED', '0') == '1'"
    assert gate in source
    assert source.index(gate) < source.index("apply_dsa_indexer_fused_patch()")


def test_training_loss_gate_writes_hash_bound_finite_receipt(tmp_path):
    log = tmp_path / "stage.log"
    log.write_text(
        "iteration 9/ 9 | lm loss: 6.5 | grad norm: 0.25 | "
        "number of skipped iterations: 0 | number of nan iterations: 0 |\n",
        encoding="utf-8",
    )
    output = tmp_path / "loss-receipt.json"

    receipt = write_training_loss_receipt(
        log,
        expected_iteration=9,
        output=output,
    )

    assert receipt["schema"] == "cppmega_h200_training_loss_gate_v1"
    assert receipt["status"] == "verified"
    assert receipt["evidence"]["lm_loss"] == 6.5
    assert len(receipt["log_sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == receipt


def test_training_loss_gate_rejects_nan_or_skipped_stage(tmp_path):
    log = tmp_path / "stage.log"
    log.write_text(
        "iteration 2/ 2 | lm loss: nan | grad norm: 0.25 | "
        "number of skipped iterations: 0 | number of nan iterations: 1 |\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="finite positive LM loss"):
        write_training_loss_receipt(
            log,
            expected_iteration=2,
            output=tmp_path / "loss-receipt.json",
        )
