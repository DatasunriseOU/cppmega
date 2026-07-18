from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cppmega.receipt_binding import build_implementation_binding  # noqa: E402

from cppmega.megatron.h200_preflight import (  # noqa: E402
    GRAPH_CHUNK_KIND_COUNT,
    GraphChunkKind,
    observe_dsa_selector,
    observe_graph_prior,
    observe_production_batch,
)
from cppmega.megatron.checkpoint_restore_preflight import state_fingerprint  # noqa: E402
from cppmega.megatron.graph_objective_loss import (  # noqa: E402
    validate_runtime_graph_contract,
)
from cppmega.megatron.graph_recipe import (  # noqa: E402
    stage1_graph_recipe_binding,
    stage1_graph_recipe_payload,
)
from scripts.h200_megatron_preflight import (  # noqa: E402
    _claimed_backend_modules,
    _checkpoint_load_evidence,
    _checkpoint_tree_sha256,
    _iteration_evidence,
    _profile_environment,
    _stage_cold_checkpoint,
    _validate_backend_dispatch_receipt,
    _validate_checkpoint_state_restore,
    _validate_graph_prior_receipt,
    build_arg_parser,
    build_megatron_command,
)


def _production_batch():
    return {
        "tokens": torch.tensor([[11, 12, 13, 14]], dtype=torch.int64),
        "labels": torch.tensor([[12, 13, 14, 0]], dtype=torch.int64),
        "loss_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
    }


def _structure_batch():
    batch = {
        "source_doc_ids": torch.tensor([[7, 7, 7, 7]], dtype=torch.int64),
        "structure_ids": torch.tensor([[1, 2, 2, 1]], dtype=torch.int64),
        "graph_chunk_starts": torch.tensor([[0, 2]], dtype=torch.int64),
        "graph_chunk_ends": torch.tensor([[2, 4]], dtype=torch.int64),
        "graph_chunk_kinds": torch.tensor([[1, 2]], dtype=torch.int64),
        "graph_chunk_dep_levels": torch.tensor([[0, 1]], dtype=torch.int64),
        "graph_chunk_counts": torch.tensor([2], dtype=torch.int64),
        "objective_ids": torch.tensor([[2, 4, 5, 5]], dtype=torch.int64),
    }
    for family in ("call", "type"):
        batch[f"graph_{family}_edges"] = torch.empty((1, 0, 2), dtype=torch.int64)
        batch[f"graph_{family}_edge_counts"] = torch.tensor([0], dtype=torch.int64)
    for family in ("domain", "build", "shell", "diagnostic", "cross_domain"):
        batch[f"graph_{family}_edges"] = torch.empty((1, 0, 3), dtype=torch.int64)
        batch[f"graph_{family}_edge_counts"] = torch.tensor([0], dtype=torch.int64)
    batch["graph_domain_edges"] = torch.tensor([[[1, 2, 5]]], dtype=torch.int64)
    batch["graph_domain_edge_counts"] = torch.tensor([1], dtype=torch.int64)
    return batch


def _receipt_binding():
    return {
        "schema": "cppmega_case6_receipt_binding_v2",
        "bundle_id": "bundle-1",
        "artifact_set_sha256": "a" * 64,
        "prefix_manifest_sha256s": {"data/train.json": "b" * 64},
        "checkpoint_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "command_sha256": "e" * 64,
        "run_id": "run-1",
        "implementation": build_implementation_binding(
            cppmega_commit="1" * 40,
            cppmega_tree_sha256="2" * 64,
            megatron_commit="3" * 40,
            cppmega_mlx_commit="4" * 40,
            cppmega_mlx_tree_sha256="5" * 64,
            clang_indexer_sha256="6" * 64,
            clang_indexer_dependency_closure_sha256="7" * 64,
        ),
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
    assert receipt["active_graph"]["route_edge_count"] == 1
    assert receipt["source_provenance"]["minimum_source_doc_id"] == 7
    assert receipt["batch"]["tokens"]["shape"] == [1, 4]
    assert receipt["structure"]["structure_ids"]["nonzero"] == 4
    assert receipt["structure"]["graph_chunk_counts"]["sum"] == 2
    assert receipt["active_graph"] == {
        "chunk_count": 2,
        "max_chunk_end": 4,
        "route_edge_count": 1,
        "route_edge_counts": {"domain": 1},
    }
    assert receipt["objective_mix"] == {
        "input_tokens_by_objective": {
            "fim": 1,
            "ifim": 1,
            "commit_diff": 2,
        },
        "loss_tokens_by_objective": {
            "fim": 1,
            "ifim": 1,
            "commit_diff": 1,
        },
        "observed_objective_ids": [2, 4, 5],
    }
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_observe_production_batch_requires_objective_ids_for_contract_mode(tmp_path):
    structure = _structure_batch()
    structure.pop("objective_ids")

    with pytest.raises(RuntimeError, match="objective_ids"):
        observe_production_batch(
            batch=_production_batch(),
            structure_batch=structure,
            receipt_path=tmp_path / "batch.json",
            environment={"CPPMEGA_OBJECTIVE_CONTRACT_REQUIRED": "1"},
        )


def test_observe_production_batch_accepts_canonical_other_chunk_kind_zero(tmp_path):
    structure = _structure_batch()
    structure["graph_chunk_kinds"].fill_(int(GraphChunkKind.OTHER))

    receipt = observe_production_batch(
        batch=_production_batch(),
        structure_batch=structure,
        receipt_path=tmp_path / "batch.json",
    )

    assert int(GraphChunkKind.OTHER) == 0
    assert receipt["structure"]["graph_chunk_kinds"]["nonzero"] == 0
    assert receipt["active_graph"]["chunk_count"] == 2


def test_observe_production_batch_rejects_out_of_range_chunk_kind(tmp_path):
    structure = _structure_batch()
    structure["graph_chunk_kinds"][0, 0] = GRAPH_CHUNK_KIND_COUNT

    with pytest.raises(RuntimeError, match="chunk kind.*canonical range"):
        observe_production_batch(
            batch=_production_batch(),
            structure_batch=structure,
            receipt_path=tmp_path / "batch.json",
        )


@pytest.mark.parametrize(
    ("starts", "ends"),
    (
        ([0, 1], [2, 4]),
        ([2, 0], [4, 2]),
    ),
)
def test_observe_production_batch_rejects_unordered_or_overlapping_chunks(
    tmp_path, starts, ends
):
    structure = _structure_batch()
    structure["graph_chunk_starts"][0] = torch.tensor(starts)
    structure["graph_chunk_ends"][0] = torch.tensor(ends)

    with pytest.raises(RuntimeError, match="ordered and nonoverlapping"):
        observe_production_batch(
            batch=_production_batch(),
            structure_batch=structure,
            receipt_path=tmp_path / "batch.json",
        )


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


def test_observe_production_batch_rejects_zero_route_edges(tmp_path):
    structure = _structure_batch()
    structure["graph_domain_edges"].zero_()
    structure["graph_domain_edge_counts"].zero_()

    with pytest.raises(RuntimeError, match="no active route edges"):
        observe_production_batch(
            batch=_production_batch(),
            structure_batch=structure,
            receipt_path=tmp_path / "batch.json",
        )


def test_observe_graph_prior_requires_nonzero_consumer_input(tmp_path):
    receipt_path = tmp_path / "prior.json"
    receipt = observe_graph_prior(
        prior=torch.tensor([[[[0.0, 2.0], [0.0, 0.0]]]]),
        consumer="dense_attention",
        receipt_path=receipt_path,
    )

    assert receipt["status"] == "verified"
    assert receipt["consumer"] == "dense_attention"
    assert receipt["bias_beta"]["value"] == "1"
    assert receipt["prior"]["nonzero"] == 1

    with pytest.raises(RuntimeError, match="graph prior.*nonzero"):
        observe_graph_prior(
            prior=torch.zeros((1, 1, 2, 2)),
            consumer="dsa_indexer",
            receipt_path=tmp_path / "zero.json",
        )


def test_observe_graph_prior_rejects_stale_beta_receipt(tmp_path):
    receipt_path = tmp_path / "prior.json"
    observe_graph_prior(
        prior=torch.tensor([[[[0.0, 2.0], [0.0, 0.0]]]]),
        consumer="dsa_indexer",
        receipt_path=receipt_path,
        bias_beta=1.0,
    )

    with pytest.raises(RuntimeError, match="receipt beta"):
        observe_graph_prior(
            prior=torch.tensor([[[[0.0, 2.0], [0.0, 0.0]]]]),
            consumer="dsa_indexer",
            receipt_path=receipt_path,
            bias_beta=2.0,
        )


def test_observe_dsa_selector_records_equation_mask_and_topk(tmp_path):
    receipt_path = tmp_path / "prior.json"
    neural = torch.zeros((1, 2, 4), dtype=torch.float32)
    graph = torch.zeros_like(neural)
    graph[0, 0, 3] = 2.0
    graph[0, 1, 1] = 3.0
    mask = torch.zeros_like(neural)
    mask[0, 0, 3] = float("-inf")
    post_add = neural + graph
    post_mask = post_add + mask
    topk = post_mask.topk(2, dim=-1).indices

    observe_graph_prior(
        prior=graph,
        consumer="dsa_indexer",
        receipt_path=receipt_path,
        bias_beta=1.0,
    )
    receipt = observe_dsa_selector(
        neural_scores=neural,
        graph_prior=graph,
        beta=1.0,
        mask=mask,
        actual_post_add_scores=post_add,
        actual_post_mask_scores=post_mask,
        actual_topk_indices=topk,
        index_topk=2,
        receipt_path=receipt_path,
        layer_number=3,
    )

    assert receipt["selector"]["status"] == "verified"
    assert receipt["selector"]["formula"] == "I_neural + beta*S_graph -> mask -> topk"
    observation = receipt["selector"]["observations"][0]
    assert observation["indices_match"] is True
    assert observation["topk_indices"]["sample"] == [
        int(value) for value in topk.reshape(-1)
    ]
    assert observation["equation_max_abs_error"] == 0.0
    assert observation["mask_max_abs_error"] == 0.0
    assert observation["post_mask"]["negative_infinity_count"] == 1
    assert "-Infinity" not in receipt_path.read_text(encoding="utf-8")
    _validate_graph_prior_receipt(receipt, expected_beta=1.0, require_selector=True)


def test_h200_graph_preflight_contract_rejects_tensor_only_without_gpu(tmp_path):
    environment = _profile_environment(
        sequence_length=1024,
        micro_batch_size=1,
        fp8_recipe="off",
        graph_max_edges=8,
        graph_max_chunks=4,
        enable_dsa_patch=True,
    )
    graph_contract = {
        **stage1_graph_recipe_payload(),
        "recipe": stage1_graph_recipe_binding(),
        "included_in_total_loss": True,
    }

    assert environment["CPPMEGA_GRAPH_ROUTES_ENABLED"] == "1"
    assert environment["CPPMEGA_STRUCTURE_ENABLED"] == "1"
    assert environment["CPPMEGA_DSA_GRAPH_AUX_ENABLED"] == "1"
    assert environment["CPPMEGA_DSA_PATCH_ENABLED"] == "1"
    assert "CPPMEGA_GRAPH_ROUTES_ABLATION" not in environment
    validate_runtime_graph_contract(
        graph_contract,
        environment=environment,
        require_included_auxiliary=True,
    )

    tensor_only = {
        **environment,
        "CPPMEGA_GRAPH_ROUTES_ENABLED": "0",
        "CPPMEGA_GRAPH_ROUTES_ABLATION": "1",
    }
    with pytest.raises(ValueError, match="requires CPPMEGA_GRAPH_ROUTES_ENABLED=1"):
        validate_runtime_graph_contract(
            graph_contract,
            environment=tensor_only,
            require_included_auxiliary=True,
        )

    receipt_path = tmp_path / "graph-prior-without-selector.json"
    graph_prior = observe_graph_prior(
        prior=torch.tensor([[[0.0, 2.0], [0.0, 0.0]]]),
        consumer="dsa_indexer",
        receipt_path=receipt_path,
        bias_beta=1.0,
    )
    with pytest.raises(RuntimeError, match="selector-level top-k evidence"):
        _validate_graph_prior_receipt(
            graph_prior,
            expected_beta=1.0,
            require_selector=True,
        )


def test_observe_production_batch_rejects_stale_binding(tmp_path):
    receipt_path = tmp_path / "batch.json"
    observe_production_batch(
        batch=_production_batch(),
        structure_batch=_structure_batch(),
        receipt_path=receipt_path,
        receipt_binding=_receipt_binding(),
    )
    expected = _receipt_binding()
    expected["run_id"] = "run-2"

    with pytest.raises(RuntimeError, match="run_id"):
        observe_production_batch(
            batch=_production_batch(),
            structure_batch=_structure_batch(),
            receipt_path=receipt_path,
            receipt_binding=expected,
        )


def test_dense_profile_does_not_claim_unselected_kernel_backends():
    environment = {
        "CPPMEGA_DENSE_GQA": "1",
        "CPPMEGA_DSA_SPARSE_MODE": "tilelang",
    }

    assert _claimed_backend_modules(environment) == ()


def test_tilelang_profile_requires_actual_dispatch_and_numerical_evidence():
    claims = _claimed_backend_modules(
        {"CPPMEGA_DENSE_GQA": "0", "CPPMEGA_DSA_SPARSE_MODE": "tilelang"}
    )
    assert claims == ("tilelang",)
    receipt = {
        "schema": "cppmega_backend_dispatch_v1",
        "selected_backend": "tilelang",
        "forward": {"status": "passed", "finite": True},
        "backward": {"status": "passed", "finite": True},
        "numerical": {"status": "passed", "max_abs_error": 0.001},
    }

    assert _validate_backend_dispatch_receipt(receipt, claims=claims) == receipt

    del receipt["backward"]
    with pytest.raises(RuntimeError, match="backward"):
        _validate_backend_dispatch_receipt(receipt, claims=claims)


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
        "NATIVE_ARGS": (
            "--experimental-attention-variant dsa "
            "--dsa-indexer-loss-coeff 0.001"
        ),
        "CPPMEGA_DSA_INDEXER_LOSS_COEFF": "0.001",
        "CPPMEGA_SPEC_MODULE": "cppmega.megatron.nam56r_full_spec",
        "CPPMEGA_SPEC_FUNCTION": "build_cppmega_nam56r_full_stack_spec",
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


def test_checkpoint_load_evidence_requires_explicit_megatron_load_at_iteration_one():
    log = (
        "successfully loaded checkpoint from /data/preflight "
        "[ t 1/1, p 1/1 ] at iteration 1\n"
        "iteration 2/ 2 | lm loss: 7.25 | grad norm: 0.15 |"
    )

    assert _checkpoint_load_evidence(log, expected_iteration=1)["iteration"] == 1

    with pytest.raises(RuntimeError, match="successful checkpoint load.*iteration 1"):
        _checkpoint_load_evidence(
            "iteration 2/ 2 | lm loss: 7.25 | grad norm: 0.15 |",
            expected_iteration=1,
        )


@pytest.mark.parametrize("component", ["model", "optimizer", "rng"])
def test_checkpoint_restore_requires_matching_runtime_state_fingerprints(component):
    saved = {
        "schema": "cppmega_h200_checkpoint_state_v1",
        "status": "verified",
        "mode": "save",
        "iteration": 1,
        "fingerprints": {"model": "a" * 64, "optimizer": "b" * 64, "rng": "c" * 64},
    }
    loaded = {
        **saved,
        "mode": "load",
        "fingerprints": dict(saved["fingerprints"]),
    }

    assert _validate_checkpoint_state_restore(saved, loaded)["matched"] == [
        "model",
        "optimizer",
        "rng",
    ]

    loaded["fingerprints"][component] = "d" * 64
    with pytest.raises(RuntimeError, match=component):
        _validate_checkpoint_state_restore(saved, loaded)


def test_checkpoint_state_fingerprint_is_content_and_structure_bound():
    first = {
        "weight": torch.tensor([[1.0, 2.0]]),
        "step": 3,
        "nested": [True, None],
    }
    same = {
        "nested": [True, None],
        "step": 3,
        "weight": torch.tensor([[1.0, 2.0]]),
    }
    changed = {**same, "weight": torch.tensor([[1.0, 2.5]])}

    assert state_fingerprint(first) == state_fingerprint(same)
    assert state_fingerprint(first) != state_fingerprint(changed)


def test_h200_cli_requires_explicit_bundle_and_run_identity():
    required = {
        action.dest
        for action in build_arg_parser()._actions
        if getattr(action, "required", False)
    }

    assert {"bundle_root", "data_prefix", "tokenizer_model", "run_id"} <= required


def test_cold_checkpoint_restore_uses_distinct_verified_tree(tmp_path):
    source = tmp_path / "save"
    iteration = source / "iter_0000001"
    iteration.mkdir(parents=True)
    (iteration / "state.pt").write_bytes(b"checkpoint-state")
    (source / "latest_checkpointed_iteration.txt").write_text(
        "1\n", encoding="utf-8"
    )
    checkpoint_sha256 = _checkpoint_tree_sha256(source)
    binding = _receipt_binding()
    binding["checkpoint_sha256"] = checkpoint_sha256
    destination = tmp_path / "cold"
    receipt_path = tmp_path / "cold-receipt.json"

    receipt = _stage_cold_checkpoint(
        source=source,
        destination=destination,
        receipt_path=receipt_path,
        receipt_binding=binding,
    )

    assert receipt["status"] == "verified"
    assert receipt["checkpoint_sha256"] == checkpoint_sha256
    assert receipt["binding"] == binding
    assert _checkpoint_tree_sha256(destination) == checkpoint_sha256
    assert destination.resolve() != source.resolve()
    with pytest.raises(RuntimeError, match="stale cold checkpoint"):
        _stage_cold_checkpoint(
            source=source,
            destination=destination,
            receipt_path=tmp_path / "second.json",
            receipt_binding=binding,
        )


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
