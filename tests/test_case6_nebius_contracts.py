from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import struct

import pytest

from cppmega.megatron.graph_recipe import (
    STAGE1_GRAPH_RELATIONS,
    STAGE1_GRAPH_TOPK,
    stage1_graph_recipe_binding,
)
from cppmega.megatron.objective_contract import (
    LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1,
    OBJECTIVE_CONTRACT_SCHEMA,
    OBJECTIVE_IDS,
    validate_objective_contract,
)
from scripts.data.publish_megatron_bundle_to_nebius_s3 import (
    _head_matches,
    _resolve_s3_env,
    _validate_logical_manifest_contract,
)
from scripts.data.restore_megatron_bundle_from_nebius_s3 import (
    _acquire_restore_lock,
    _remove_partial_tree,
    _validate_run_id,
)
from scripts.h200_megatron_preflight import (
    STACK_REQUIRED_IMPORTS,
    _derive_graph_capacity_from_manifest,
    _profile_environment,
    build_megatron_command,
    validate_stack_compatibility,
)


TASKS = (
    "causal_lm",
    "fim",
    "ast_fim",
    "ifim",
    "commit_diff",
    "pre_to_post",
)


def _valid_objective_contract() -> dict[str, object]:
    return {
        "schema": OBJECTIVE_CONTRACT_SCHEMA,
        "algorithm": "hamilton_eligibility_bipartite_v1",
        "seed": 17,
        "quota_window_samples": 6,
        "task_order": list(TASKS),
        "objective_ids": {task: OBJECTIVE_IDS[task] for task in TASKS},
        "configured_rates": {task: "1/6" for task in TASKS},
        "planned_samples": {task: 1 for task in TASKS},
        "realized": {
            task: {
                "samples": 1,
                "input_tokens": 3,
                "loss_tokens": 3 if task == "causal_lm" else 2,
            }
            for task in TASKS
        },
        "totals": {"samples": 6, "input_tokens": 18, "loss_tokens": 13},
        "typed_sources": {
            "ifim_instruction": "ifim_instruction_token_ids",
            "commit_message": "commit_msg_token_ids",
            "diff": "diff_token_ids",
            "pre": "pre_token_ids",
            "post": "post_token_ids",
            "missing_fields": "ineligible",
            "rendered_text_parsing": False,
        },
        "graph_auxiliary": {
            "recipe": stage1_graph_recipe_binding(),
            "relations": list(STAGE1_GRAPH_RELATIONS),
            "bias_beta": "1",
            "score_formula": "i_neural_plus_beta_s_graph_v1",
            "score_stage": "before_topk",
            "eligible_samples": 1,
            "positive_edges": 5,
            "global_weight": "1",
            "indexer_weight": "1/1000",
            "layer_weight": "1",
            "layer_reduction": "sum",
            "bce_weight": "1/10",
            "coverage_weight": "1/20",
            "topk": STAGE1_GRAPH_TOPK,
            "pos_weight": "1",
            "margin": "1",
            "included_in_total_loss": True,
            "runtime": "megatron_dsa_indexer_v1",
            "pair_mask": "causal_same_document_upstream_v1",
            "chunk_edge_expansion": "cartesian_token_spans_v1",
        },
        "materialization": {
            "format": "shifted_lm_document_v1",
            "token_column": "input_ids",
            "loss_mask_column": "loss_mask",
            "loss_mask_alignment": (
                LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1
            ),
            "length_column": "valid_token_count",
            "objective_column": "objective_kind",
            "document_id_column": "doc_ids",
            "source_document_id_column": "token_source_doc_ids",
        },
    }


def test_remote_logical_manifest_rejects_legacy_training_before_archive() -> None:
    manifest = {
        "schema": "cppmega_megatron_bundle_v1",
        "tokenizer_contract": "megacpp-vocab-65536",
        "vocab_size": 65536,
        "training_contract": "legacy_causal",
    }

    with pytest.raises(ValueError, match="training_contract"):
        _validate_logical_manifest_contract(manifest)


def test_s3_credentials_do_not_mix_nebius_and_aws_families() -> None:
    resolved = _resolve_s3_env(
        {
            "NEBIUS_S3_ACCESS_KEY_ID": "nebius-a",
            "NEBIUS_S3_SECRET_ACCESS_KEY": "nebius-s",
            "AWS_ACCESS_KEY_ID": "aws-a",
            "AWS_SECRET_ACCESS_KEY": "aws-s",
            "AWS_SESSION_TOKEN": "stale",
        }
    )
    assert resolved["AWS_ACCESS_KEY_ID"] == "nebius-a"
    assert resolved["AWS_SECRET_ACCESS_KEY"] == "nebius-s"
    assert "AWS_SESSION_TOKEN" not in resolved

    with pytest.raises(SystemExit, match="complete Nebius"):
        _resolve_s3_env(
            {
                "NEBIUS_S3_ACCESS_KEY_ID": "nebius-a",
                "AWS_SECRET_ACCESS_KEY": "aws-s",
            }
        )


def test_s3_already_verified_requires_exact_full_object_checksum() -> None:
    digest = hashlib.sha256(b"remote-bytes").hexdigest()
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    common = {
        "ContentLength": 12,
        "Metadata": {"sha256": digest},
        "ChecksumSHA256": checksum,
    }

    assert _head_matches(
        {**common, "ChecksumType": "FULL_OBJECT"}, size=12, sha256=digest
    )
    assert not _head_matches(common, size=12, sha256=digest)
    assert not _head_matches(
        {
            **common,
            "ChecksumSHA256": checksum + "-2",
            "ChecksumType": "COMPOSITE",
        },
        size=12,
        sha256=digest,
    )


def test_restore_lock_is_scoped_to_bundle_not_run_id(tmp_path: Path) -> None:
    first = _acquire_restore_lock(tmp_path, bundle_id="bundle-1", run_id="run-1")
    try:
        with pytest.raises(RuntimeError, match="restore already active"):
            _acquire_restore_lock(tmp_path, bundle_id="bundle-1", run_id="run-2")
    finally:
        first.close()


def test_restore_refuses_symlink_lock_path(tmp_path: Path) -> None:
    target = tmp_path / "target.lock"
    target.write_text("do not touch", encoding="utf-8")
    (tmp_path / ".bundle-1.restore.lock").symlink_to(target)

    with pytest.raises(ValueError, match="restore lock path"):
        _acquire_restore_lock(tmp_path, bundle_id="bundle-1", run_id="run-1")

    assert target.read_text(encoding="utf-8") == "do not touch"


@pytest.mark.parametrize("run_id", ["../escape", "bad/id", "x" * 129, "space id"])
def test_restore_run_id_is_validated_before_path_use(run_id: str) -> None:
    with pytest.raises(ValueError, match="restore run_id"):
        _validate_run_id(run_id)


def test_restore_refuses_symlink_partial_tree(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    partial = tmp_path / ".bundle.run.partial"
    partial.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="partial path"):
        _remove_partial_tree(partial)

    assert target.is_dir()


def test_profile_requires_derived_graph_capacity() -> None:
    with pytest.raises(ValueError, match="capacities derived"):
        _profile_environment(
            sequence_length=1024,
            micro_batch_size=1,
            fp8_recipe="off",
            graph_max_edges=None,
            graph_max_chunks=None,
            enable_dsa_patch=True,
        )
    environment = _profile_environment(
        sequence_length=1024,
        micro_batch_size=1,
        fp8_recipe="off",
        graph_max_edges=7,
        graph_max_chunks=5,
        enable_dsa_patch=True,
    )
    assert environment["CPPMEGA_GRAPH_MAX_EDGES"] == "7"
    assert environment["CPPMEGA_GRAPH_MAX_CHUNKS"] == "5"


def test_h200_training_command_keeps_nan_checks_enabled(tmp_path: Path) -> None:
    environment = _profile_environment(
        sequence_length=1024,
        micro_batch_size=1,
        fp8_recipe="off",
        graph_max_edges=7,
        graph_max_chunks=5,
        enable_dsa_patch=True,
    )
    command = build_megatron_command(
        wrapper=tmp_path / "wrapper.py",
        data_prefix=tmp_path / "train",
        tokenizer_model=tmp_path / "tokenizer",
        checkpoint_root=tmp_path / "checkpoint",
        sequence_length=1024,
        micro_batch_size=1,
        train_iters=1,
        environment=environment,
        load_checkpoint=False,
    )
    assert "--no-check-for-nan-in-loss-and-grad" not in command


def test_graph_capacity_receipt_comes_from_csr_offsets(tmp_path: Path) -> None:
    data_prefix = tmp_path / "train"
    data_prefix.with_suffix(".json").write_text("{}", encoding="utf-8")
    offsets = (0, 2, 5)
    graph_paths: dict[str, dict[str, object]] = {}
    for name in (
        "token_chunk_starts",
        "token_chunk_ends",
        "token_chunk_kinds",
        "token_chunk_dep_levels",
    ):
        path = tmp_path / f"{name}.offsets"
        path.write_bytes(struct.pack("<3q", *offsets))
        graph_paths[name] = {
            "kind": "ragged_1d",
            "offsets_path": path.name,
            "offset_dtype": "int64",
            "item_count": offsets[-1],
        }

    edge_path = tmp_path / "domain.offsets"
    edge_path.write_bytes(struct.pack("<3q", 0, 1, 1))
    graph_paths["token_domain_edges"] = {
        "kind": "edge_triples",
        "offsets_path": edge_path.name,
        "offset_dtype": "int64",
        "item_count": 1,
    }

    receipt = _derive_graph_capacity_from_manifest(
        data_prefix,
        manifest={
            "document_count": 2,
            "source_capacity_token_count": 2048,
            "graph_sidecar_paths": graph_paths,
        },
        sequence_length=1024,
    )

    assert receipt["status"] == "verified"
    assert receipt["graph_max_edges"] == 1
    assert receipt["graph_max_chunks"] == 3
    assert len(receipt["sidecars"]) == 5


def test_stack_contract_matches_pinned_runtime_and_import_set() -> None:
    lock = {
        "base": {
            "cuda_image": "nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04",
            "python": "3.13",
            "torch": "2.13.0.dev20260611+cu132",
        },
        "wheels": {"transformer_engine": {"version": 2.16}},
    }
    contract = validate_stack_compatibility(
        lock,
        python_version=(3, 13),
        torch_version="2.13.0.dev20260611+cu132",
        cuda_runtime="13.2",
        transformer_engine_version="2.16.0.dev0+local",
        imported_modules=STACK_REQUIRED_IMPORTS,
    )
    assert contract["status"] == "verified"

    with pytest.raises(RuntimeError, match="torch version mismatch"):
        validate_stack_compatibility(
            lock,
            python_version=(3, 13),
            torch_version="2.13.0.dev20260612+cu132",
            cuda_runtime="13.2",
            transformer_engine_version="2.16.0.dev0+local",
            imported_modules=STACK_REQUIRED_IMPORTS,
        )

    with pytest.raises(RuntimeError, match="required H200 extension imports"):
        validate_stack_compatibility(
            lock,
            python_version=(3, 13),
            torch_version="2.13.0.dev20260611+cu132",
            cuda_runtime="13.2",
            transformer_engine_version="2.16.0.dev0+local",
            imported_modules=STACK_REQUIRED_IMPORTS[:-1],
        )


def test_objective_graph_relation_contract_rejects_unknown_relation() -> None:
    contract = _valid_objective_contract()
    graph = contract["graph_auxiliary"]
    assert isinstance(graph, dict)
    graph["relations"] = ["unknown"]

    with pytest.raises(ValueError, match="unknown relations"):
        validate_objective_contract(contract)


def test_objective_graph_eligible_samples_cannot_exceed_total() -> None:
    contract = _valid_objective_contract()
    graph = contract["graph_auxiliary"]
    assert isinstance(graph, dict)
    graph["eligible_samples"] = 7

    with pytest.raises(ValueError, match="cannot exceed totals.samples"):
        validate_objective_contract(contract)
