from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cppmega.megatron.h200_preflight import observe_production_batch
from scripts.h200_megatron_preflight import _iteration_evidence, build_megatron_command


def _production_batch():
    return {
        "tokens": torch.tensor([[11, 12, 13, 14]], dtype=torch.int64),
        "labels": torch.tensor([[12, 13, 14, 0]], dtype=torch.int64),
        "loss_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
    }


def _structure_batch():
    return {
        "structure_ids": torch.tensor([[1, 2, 2, 1]], dtype=torch.int64),
        "graph_chunk_starts": torch.tensor([[0, 2]], dtype=torch.int64),
        "graph_chunk_ends": torch.tensor([[2, 4]], dtype=torch.int64),
        "graph_chunk_kinds": torch.tensor([[1, 2]], dtype=torch.int64),
        "graph_chunk_dep_levels": torch.tensor([[0, 1]], dtype=torch.int64),
        "graph_chunk_counts": torch.tensor([2], dtype=torch.int64),
    }


def test_observe_production_batch_records_nonzero_structure_and_graph(tmp_path):
    receipt_path = tmp_path / "batch.json"

    receipt = observe_production_batch(
        batch=_production_batch(),
        structure_batch=_structure_batch(),
        receipt_path=receipt_path,
    )

    assert receipt["schema"] == "cppmega_h200_production_batch_v1"
    assert receipt["status"] == "verified"
    assert receipt["batch"]["tokens"]["shape"] == [1, 4]
    assert receipt["structure"]["structure_ids"]["nonzero"] == 4
    assert receipt["structure"]["graph_chunk_counts"]["sum"] == 2
    assert receipt["active_graph"] == {"chunk_count": 2, "max_chunk_end": 4}
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_observe_production_batch_rejects_zero_structure_before_receipt(tmp_path):
    structure = _structure_batch()
    structure["structure_ids"].zero_()
    receipt_path = tmp_path / "batch.json"

    with pytest.raises(RuntimeError, match="structure_ids.*nonzero"):
        observe_production_batch(
            batch=_production_batch(),
            structure_batch=structure,
            receipt_path=receipt_path,
        )

    assert not receipt_path.exists()


def test_observe_production_batch_rejects_missing_graph_sidecars(tmp_path):
    structure = _structure_batch()
    structure.pop("graph_chunk_counts")

    with pytest.raises(RuntimeError, match="missing graph batch fields"):
        observe_production_batch(
            batch=_production_batch(),
            structure_batch=structure,
            receipt_path=tmp_path / "batch.json",
        )


def test_observe_production_batch_rejects_invalid_active_graph_span(tmp_path):
    structure = _structure_batch()
    structure["graph_chunk_ends"][0, 0] = 0

    with pytest.raises(RuntimeError, match="active graph chunk spans"):
        observe_production_batch(
            batch=_production_batch(),
            structure_batch=structure,
            receipt_path=tmp_path / "batch.json",
        )


def test_observe_production_batch_rejects_batch_shape_drift(tmp_path):
    structure = _structure_batch()
    structure["structure_ids"] = torch.tensor([[1, 2, 3]])

    with pytest.raises(RuntimeError, match="structure_ids shape"):
        observe_production_batch(
            batch=_production_batch(),
            structure_batch=structure,
            receipt_path=tmp_path / "batch.json",
        )


def test_h200_commands_save_and_restore_full_optimizer_state(tmp_path):
    environment = {
        "HYBRID_LAYER_PATTERN": "M*2",
        "CPPMEGA_HIDDEN_SIZE": "128",
        "CPPMEGA_FFN_HIDDEN_SIZE": "256",
        "CPPMEGA_NUM_ATTN_HEADS": "4",
        "CPPMEGA_NUM_QUERY_GROUPS": "2",
        "CPPMEGA_KV_CHANNELS": "32",
        "CPPMEGA_LR": "1e-4",
        "CPPMEGA_MIN_LR": "1e-5",
        "CPPMEGA_ATTN_BACKEND": "auto",
        "CPPMEGA_OPTIMIZER": "adam",
        "CPPMEGA_USE_FLASH_ATTN": "1",
        "CPPMEGA_FP8_RECIPE": "off",
    }
    common = {
        "wrapper": tmp_path / "pretrain_mamba.py",
        "data_prefix": Path("/data/cppmega_sidecar/train"),
        "tokenizer_model": Path("/data/cpp_tokenizer_hf"),
        "checkpoint_root": Path("/data/preflight-checkpoint"),
        "sequence_length": 1024,
        "micro_batch_size": 1,
        "environment": environment,
    }

    save = build_megatron_command(train_iters=1, load_checkpoint=False, **common)
    restore = build_megatron_command(train_iters=2, load_checkpoint=True, **common)

    assert save[save.index("--train-iters") + 1] == "1"
    assert restore[restore.index("--train-iters") + 1] == "2"
    assert "--load" not in save
    assert restore[restore.index("--load") + 1] == "/data/preflight-checkpoint"
    assert "--save" in save and "--save" in restore
    assert "--no-save-optim" not in save
    assert "--no-save-rng" not in save
    assert "--no-load-optim" not in restore
    assert "--no-load-rng" not in restore


def test_iteration_evidence_requires_finite_forward_and_backward_metrics():
    evidence = _iteration_evidence(
        "iteration 2/ 2 | lm loss: 7.25E+00 | grad norm: 1.5E-01 | "
        "number of skipped iterations: 0 | number of nan iterations: 0 |",
        expected_iteration=2,
    )

    assert evidence == {
        "iteration": 2,
        "lm_loss": 7.25,
        "grad_norm": 0.15,
        "skipped_iterations": 0,
        "nan_iterations": 0,
    }


@pytest.mark.parametrize(
    ("log", "message"),
    [
        (
            "iteration 1/ 1 | lm loss: 7.0 | grad norm: nan | "
            "number of skipped iterations: 0 | number of nan iterations: 1 |",
            "finite positive grad norm",
        ),
        (
            "iteration 1/ 1 | lm loss: 7.0 | grad norm: 0.2 | "
            "number of skipped iterations: 1 | number of nan iterations: 0 |",
            "skipped iterations",
        ),
    ],
)
def test_iteration_evidence_rejects_unproven_optimizer_step(log, message):
    with pytest.raises(RuntimeError, match=message):
        _iteration_evidence(log, expected_iteration=1)
