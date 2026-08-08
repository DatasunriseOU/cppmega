from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cppmega.megatron.objective_contract as objective_contract
from cppmega.megatron.graph_recipe import (
    STAGE1_GRAPH_RELATIONS,
    STAGE1_GRAPH_TOPK,
    stage1_graph_recipe_binding,
)
from cppmega.megatron.objective_contract import (
    GRAPH_ELIGIBILITY_RECEIPT_SCHEMA,
    OBJECTIVE_REALIZATION_RECEIPT_SCHEMA,
    LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1,
    OBJECTIVE_CONTRACT_SCHEMA,
    OBJECTIVE_IDS,
    OBJECTIVE_SCHEDULE_ALGORITHM,
    OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA,
    OBJECTIVE_SCHEDULE_WINDOW_SCHEMA,
    OBJECTIVE_SOURCE_RESUME_SCHEMA,
    OBJECTIVE_SOURCE_SELECTION_SCHEMA,
    OBJECTIVE_GRAPH_SIDECARS,
    OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA,
    OBJECTIVE_TOKEN_SIDE_CHANNELS,
    ObjectiveMaterializationTracker,
    load_objective_materialization_artifact,
    materialized_objective_artifact_manifest,
    validate_materialized_objective_contract,
    validate_objective_contract,
    validate_production_objective_contract,
    verify_objective_materialization_shard,
)

TASKS = (
    "causal_lm",
    "fim",
    "ast_fim",
    "ifim",
    "commit_diff",
    "pre_to_post",
)


def _source_selection_receipt() -> dict[str, object]:
    relations = list(STAGE1_GRAPH_RELATIONS)
    assignments = []
    for index, task in enumerate(TASKS):
        eligible = index == 0
        route_mode = "identity" if task == "causal_lm" else (
            "excluded" if task in {"commit_diff", "pre_to_post"} else "source_token_remap"
        )
        reason = None if eligible else (
            "exact_source_route_map_unavailable"
            if route_mode == "excluded"
            else "no_configured_graph_positive_causal_same_document_pair"
        )
        assignments.append(
            {
                "source_index": index,
                "source_pool_index": index,
                "task": task,
                "realization": {
                    "schema": OBJECTIVE_REALIZATION_RECEIPT_SCHEMA,
                    "task": task,
                    "selected_packet_index": 0,
                    "example_sha256": hashlib.sha256(
                        f"{index}:{task}".encode("ascii")
                    ).hexdigest(),
                    "input_tokens": 3,
                    "loss_tokens": 2,
                },
                "graph_eligibility": {
                    "schema": GRAPH_ELIGIBILITY_RECEIPT_SCHEMA,
                    "objective": task,
                    "eligible": eligible,
                    "reason": reason,
                    "positive_edges": 5 if eligible else 0,
                    "relations": relations,
                    "route_mode": route_mode,
                    "route_receipt": {"mode": route_mode},
                },
            }
        )
    window = {
        "schema": OBJECTIVE_SCHEDULE_WINDOW_SCHEMA,
        "algorithm": OBJECTIVE_SCHEDULE_ALGORITHM,
        "start_step": 0,
        "output_samples": len(TASKS),
        "source_pool_samples": len(TASKS),
        "source_pool_source_indices": list(range(len(TASKS))),
        "source_rows_consumed": len(TASKS),
        "selected_source_indices": list(range(len(TASKS))),
        "task_counts": {task: 1 for task in TASKS},
        "assignments": assignments,
        "graph_positive_assignments": 1,
        "graph_positive_edges": 5,
    }
    digest = hashlib.sha256(
        json.dumps([window], sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "schema": OBJECTIVE_SOURCE_SELECTION_SCHEMA,
        "algorithm": OBJECTIVE_SCHEDULE_ALGORITHM,
        "output_samples": len(TASKS),
        "source_rows_consumed": len(TASKS),
        "unused_buffered_sources": 0,
        "quota_window_samples": len(TASKS),
        "quota_lookahead_samples": 0,
        "max_source_pool_samples": len(TASKS),
        "max_source_pool_observed": len(TASKS),
        "required_graph_relations": relations,
        "windows": [window],
        "windows_sha256": digest,
        "resume": {
            "schema": OBJECTIVE_SOURCE_RESUME_SCHEMA,
            "cursor_semantics": (
                "replay_buffered_rows_then_continue_after_last_yielded_v1"
            ),
            "last_yielded_cursor": {"source_index": len(TASKS) - 1},
            "buffered_source_cursors": [],
        },
        "schedule": {
            "schema": OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA,
            "algorithm": OBJECTIVE_SCHEDULE_ALGORITHM,
            "windows_sha256": digest,
        },
    }


def _refresh_source_selection_digest(receipt: dict[str, object]) -> None:
    digest = hashlib.sha256(
        json.dumps(
            receipt["windows"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    receipt["windows_sha256"] = digest
    receipt["schedule"]["windows_sha256"] = digest  # type: ignore[index]


def _valid_contract() -> dict[str, object]:
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
        "totals": {
            "samples": 6,
            "input_tokens": 18,
            "loss_tokens": 13,
        },
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
            "eligible_samples": 1,
            "positive_edges": 5,
            "global_weight": "1",
            "indexer_weight": "1/1000",
            "layer_weight": "1",
            "layer_reduction": "sum",
            "bce_weight": "1/10",
            "coverage_weight": "1/20",
            "bias_beta": "1",
            "topk": STAGE1_GRAPH_TOPK,
            "score_formula": "i_neural_plus_beta_s_graph_v1",
            "score_stage": "before_topk",
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
        "source_selection": _source_selection_receipt(),
    }


def _write_materialization_artifact(
    tmp_path: Path, *, contract: dict[str, object] | None = None
) -> Path:
    contract = _valid_contract() if contract is None else contract
    contract_path = tmp_path / "objective_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    shard = tmp_path / "objectives_00000.parquet"
    shard.write_bytes(b"bound parquet bytes")
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    artifact = {
        "schema": OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA,
        "graph_recipe": stage1_graph_recipe_binding(),
        "documents": 6,
        "objective_contract": {
            "path": contract_path.name,
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "size_bytes": contract_path.stat().st_size,
            "file_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        },
        "parquet_shards": [
            {
                "path": shard.name,
                "size_bytes": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
        "converter": {
            "split": "all",
            "token_column": "input_ids",
            "length_column": "valid_token_count",
            "side_channels": [
                {"column": column, "dtype": dtype}
                for column, dtype in OBJECTIVE_TOKEN_SIDE_CHANNELS
            ],
            "graph_sidecars": [
                {"column": column, "kind": kind, "dtype": dtype}
                for column, kind, dtype in OBJECTIVE_GRAPH_SIDECARS
            ],
            "source_platform_sidecar": "require",
            "loss_mask_alignment": (
                LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1
            ),
            "graph_relations": list(STAGE1_GRAPH_RELATIONS),
            "graph_pair_mask": "causal_same_document_upstream_v1",
            "chunk_edge_expansion": "cartesian_token_spans_v1",
        },
    }
    artifact["artifact_set_sha256"] = hashlib.sha256(
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    artifact_path = tmp_path / "objective_materialization.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path


def _source_record(path: str, *, rows: int = 3) -> dict[str, object]:
    return {
        "path": path,
        "rows": rows,
        "size_bytes": 123,
        "sha256": hashlib.sha256(path.encode("ascii")).hexdigest(),
    }


def _source_artifact_digest(records: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                {
                    "path": record["path"],
                    "size": record["size_bytes"],
                    "sha256": record["sha256"],
                }
                for record in records
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _source_pool_snapshot(
    record: dict[str, object],
    *,
    sequence_length: int = 1024,
) -> dict[str, object]:
    records = [record]
    return {
        "schema": "cppmega_objective_source_snapshot_v1",
        "sequence_length": sequence_length,
        "file_count": 1,
        "row_count": 3,
        "files": records,
        "sampling": {
            "mode": "deterministic_shard_row_group_record_batch_shuffle_v2",
            "seed": 17,
            "requested_samples": 3,
            "full_passes": 1,
            "tail_rows": 0,
            "min_row_reuse": 1,
            "max_row_reuse": 1,
            "record_batch_rows": 64,
            "producer": {
                "name": "pyarrow.parquet.ParquetFile.iter_batches",
                "version": 1,
                "row_group_rows": [[3]],
            },
            "ordering": {
                "permutation": "sha256_sort_key_v1",
                "epochs": "ascending",
                "shards": "seeded_permutation_per_epoch",
                "row_groups": "seeded_permutation_per_shard_epoch",
                "record_batches": "physical_order_within_row_group",
                "rows": "seeded_permutation_within_record_batch",
            },
            "cursor_semantics": "last_yielded_row_v1",
            "final_cursor": {
                "epoch": 0,
                "shard_position": 0,
                "shard_index": 0,
                "row_group_position": 0,
                "row_group_index": 0,
                "record_batch_index": 0,
                "row_shuffle_position": 2,
                "row_index_in_record_batch": 2,
                "source_index": 2,
            },
        },
        "artifact_set_sha256": _source_artifact_digest(records),
    }


def _attach_two_pool_source_snapshot(
    tmp_path: Path,
    contract: dict[str, object],
    *,
    buckets: tuple[int, ...] = objective_contract.OBJECTIVE_SOURCE_BUCKETS,
    sequence_length: int = 1024,
) -> None:
    if sequence_length not in buckets:
        raise ValueError("fixture sequence length must be in buckets")
    primary = _source_record(f"{sequence_length}/ci.parquet")
    seed = _source_record(f"commits/{sequence_length}/seed.parquet")
    receipt = {
        "schema": "cppmega_ci_content_store_case5_export_v2",
        "status": "complete",
    }
    receipt_raw = json.dumps(receipt, sort_keys=True).encode("utf-8")
    (tmp_path / "ci_export_receipt.json").write_bytes(receipt_raw)
    primary_by_length = {
        str(bucket): [
            primary
            if bucket == 1024
            else _source_record(f"{bucket}/ci.parquet")
        ]
        for bucket in buckets
    }
    manifest = {
        "schema": "cppmega_ci_objective_pool_manifest_v1",
        "algorithm": "alternate_primary_seed_v1",
        "sequence_lengths": list(buckets),
        "ci_export": {
            "path": "export_receipt.json",
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "schema": receipt["schema"],
            "status": "complete",
            "source_completion": {
                "schema": receipt["schema"],
                "status": "complete",
            },
        },
        "primary_ci": {"files_by_sequence_length": primary_by_length},
        "objective_seed": {"files": [seed]},
        "producer": {
            "repository": "cppmega",
            "git_commit": "a" * 40,
            "script": "scripts/data/prepare_ci_objective_source_manifest.py",
            "script_sha256": "b" * 64,
        },
    }
    manifest_raw = json.dumps(manifest, sort_keys=True).encode("utf-8")
    (tmp_path / "objective_source_pool_manifest.json").write_bytes(manifest_raw)
    contract["source_snapshot"] = {
        "schema": "cppmega_objective_source_snapshot_v2",
        "sequence_length": sequence_length,
        "algorithm": "alternate_primary_seed_v1",
        "pool_order": ["primary_ci", "objective_seed"],
        "source_pool_manifest": {
            "path": "objective_source_pool_manifest.json",
            "size_bytes": len(manifest_raw),
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        },
        "ci_export_receipt": {
            "path": "ci_export_receipt.json",
            "size_bytes": len(receipt_raw),
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
        },
        "pools": {
            "primary_ci": _source_pool_snapshot(
                primary,
                sequence_length=sequence_length,
            ),
            "objective_seed": _source_pool_snapshot(
                seed,
                sequence_length=sequence_length,
            ),
        },
    }
    resume = contract["source_selection"]["resume"]  # type: ignore[index]
    resume["last_yielded_cursor"].update(  # type: ignore[index]
        {
            "pool_index": 1,
            "pool_source_index": 2,
            "primary_rows_yielded": 3,
            "objective_seed_rows_yielded": 3,
            "next_pool_index": 0,
        }
    )


def test_valid_contract_has_stable_digest_and_distinct_objective_ids() -> None:
    contract = _valid_contract()

    first = validate_objective_contract(contract)
    second = validate_objective_contract(copy.deepcopy(contract))

    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert set(first.planned_samples) == set(TASKS)
    assert OBJECTIVE_IDS["ifim"] != OBJECTIVE_IDS["ast_fim"]


def test_contract_validates_canonical_bounded_schedule_receipt() -> None:
    contract = _valid_contract()
    contract["source_selection"] = _source_selection_receipt()

    validated = validate_objective_contract(contract)

    assert validated.payload["source_selection"] == contract["source_selection"]


def test_production_contract_accepts_one_schedule_for_all_required_objectives() -> None:
    contract = _valid_contract()

    validated = validate_production_objective_contract(contract)

    assert set(TASKS).issubset(set(validated.task_order))
    assert validated.payload["source_selection"]["schedule"]["schema"] == (
        OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA
    )


def test_production_contract_rejects_missing_schedule_receipt() -> None:
    contract = _valid_contract()
    contract.pop("source_selection")

    with pytest.raises(ValueError, match="canonical.*schedule receipt"):
        validate_production_objective_contract(contract)


def test_production_contract_rejects_empty_schedule_receipt() -> None:
    contract = _valid_contract()
    source_selection = contract["source_selection"]
    assert isinstance(source_selection, dict)
    source_selection["schedule"] = {}

    with pytest.raises(ValueError, match="source_selection.schedule binding"):
        validate_production_objective_contract(contract)


def test_production_contract_rejects_zero_schedule_digest() -> None:
    contract = _valid_contract()
    source_selection = contract["source_selection"]
    assert isinstance(source_selection, dict)
    schedule = source_selection["schedule"]
    assert isinstance(schedule, dict)
    schedule["windows_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source_selection.schedule binding"):
        validate_production_objective_contract(contract)


def test_production_contract_rejects_ambiguous_schedule_receipts() -> None:
    contract = _valid_contract()
    source_selection = contract["source_selection"]
    assert isinstance(source_selection, dict)
    schedule = source_selection["schedule"]
    assert isinstance(schedule, dict)
    contract["alternate_schedule_receipt"] = copy.deepcopy(schedule)

    with pytest.raises(ValueError, match="ambiguous.*schedule receipt"):
        validate_production_objective_contract(contract)


def test_materialized_production_contract_rejects_missing_schedule_receipt() -> None:
    contract = _valid_contract()
    contract.pop("source_selection")
    legacy = validate_objective_contract(contract)
    wrapper = {
        "schema": OBJECTIVE_CONTRACT_SCHEMA,
        "sha256": legacy.sha256,
        "payload": legacy.payload,
        "objective_id_sidecar": {
            "path": "objective_ids.bin",
            "dtype": "uint8",
            "document_aligned": True,
        },
    }

    with pytest.raises(ValueError, match="canonical.*schedule receipt"):
        validate_materialized_objective_contract(
            wrapper,
            require_schedule_receipt=True,
        )


def test_contract_rejects_materializer_runner_assignment_divergence() -> None:
    contract = _valid_contract()
    source_selection = _source_selection_receipt()
    source_selection["windows"][0]["selected_source_indices"] = [  # type: ignore[index]
        1,
        0,
        2,
        3,
        4,
        5,
    ]
    _refresh_source_selection_digest(source_selection)
    contract["source_selection"] = source_selection

    with pytest.raises(ValueError, match="selected_source_indices drifted"):
        validate_objective_contract(contract)


def test_contract_requires_explicit_commit_graph_ineligibility_receipt() -> None:
    contract = _valid_contract()
    source_selection = _source_selection_receipt()
    assignments = source_selection["windows"][0]["assignments"]  # type: ignore[index]
    commit = next(row for row in assignments if row["task"] == "commit_diff")
    commit["graph_eligibility"]["route_mode"] = "source_token_remap"
    commit["graph_eligibility"]["route_receipt"]["mode"] = (
        "source_token_remap"
    )
    _refresh_source_selection_digest(source_selection)
    contract["source_selection"] = source_selection

    with pytest.raises(ValueError, match="commit objectives without exact route maps"):
        validate_objective_contract(contract)


def test_contract_rejects_source_pool_binding_drift() -> None:
    contract = _valid_contract()
    source_selection = _source_selection_receipt()
    window = source_selection["windows"][0]  # type: ignore[index]
    window["source_pool_source_indices"][:2] = [1, 0]
    _refresh_source_selection_digest(source_selection)
    contract["source_selection"] = source_selection

    with pytest.raises(ValueError, match="source pool binding drifted"):
        validate_objective_contract(contract)


def test_contract_rejects_objective_realization_drift() -> None:
    contract = _valid_contract()
    source_selection = _source_selection_receipt()
    assignments = source_selection["windows"][0]["assignments"]  # type: ignore[index]
    assignments[0]["realization"]["task"] = "fim"
    _refresh_source_selection_digest(source_selection)
    contract["source_selection"] = source_selection

    with pytest.raises(ValueError, match="realization.task differs"):
        validate_objective_contract(contract)


def test_objective_materialization_artifact_opens_exact_bound_inputs(
    tmp_path: Path,
) -> None:
    artifact_path = _write_materialization_artifact(tmp_path)

    artifact = load_objective_materialization_artifact(artifact_path)

    assert artifact.input_dir == tmp_path.resolve()
    assert artifact.contract_path == (tmp_path / "objective_contract.json").resolve()
    assert artifact.parquet_paths == (
        (tmp_path / "objectives_00000.parquet").resolve(),
    )
    assert artifact.contract.sha256 == artifact.payload["objective_contract"]["sha256"]


def test_objective_artifact_opens_bound_two_pool_source_snapshot(
    tmp_path: Path,
) -> None:
    contract = _valid_contract()
    _attach_two_pool_source_snapshot(tmp_path, contract)

    artifact = load_objective_materialization_artifact(
        _write_materialization_artifact(tmp_path, contract=contract)
    )

    assert artifact.contract.payload["source_snapshot"]["schema"] == (
        "cppmega_objective_source_snapshot_v2"
    )


def test_objective_artifact_opens_bound_large_context_source_snapshot(
    tmp_path: Path,
) -> None:
    contract = _valid_contract()
    _attach_two_pool_source_snapshot(
        tmp_path,
        contract,
        buckets=objective_contract.SUPPORTED_OBJECTIVE_SOURCE_BUCKETS,
        sequence_length=65536,
    )

    artifact = load_objective_materialization_artifact(
        _write_materialization_artifact(tmp_path, contract=contract)
    )

    assert artifact.contract.payload["source_snapshot"]["sequence_length"] == 65536


def test_objective_artifact_rejects_two_pool_manifest_byte_drift(
    tmp_path: Path,
) -> None:
    contract = _valid_contract()
    _attach_two_pool_source_snapshot(tmp_path, contract)
    artifact_path = _write_materialization_artifact(tmp_path, contract=contract)
    (tmp_path / "objective_source_pool_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_pool_manifest byte binding drifted"):
        load_objective_materialization_artifact(artifact_path)


def test_two_pool_source_snapshot_rejects_seed_only_schedule_window(
    tmp_path: Path,
) -> None:
    contract = _valid_contract()
    _attach_two_pool_source_snapshot(tmp_path, contract)
    contract["source_selection"]["windows"][0]["selected_source_indices"] = [  # type: ignore[index]
        1,
        3,
        5,
    ]

    with pytest.raises(ValueError, match="must use both source pools"):
        objective_contract._validate_two_pool_source_snapshot(contract)


def test_objective_materialization_artifact_rejects_missing_schedule(
    tmp_path: Path,
) -> None:
    contract = _valid_contract()
    contract.pop("source_selection")

    with pytest.raises(ValueError, match="canonical.*schedule receipt"):
        load_objective_materialization_artifact(
            _write_materialization_artifact(tmp_path, contract=contract)
        )


def test_objective_materialization_artifact_rejects_shard_byte_drift(
    tmp_path: Path,
) -> None:
    artifact_path = _write_materialization_artifact(tmp_path)
    shard = tmp_path / "objectives_00000.parquet"
    shard.write_bytes(b"x" * shard.stat().st_size)

    with pytest.raises(ValueError, match="parquet_shards.*sha256"):
        load_objective_materialization_artifact(artifact_path)


def test_objective_materialization_artifact_rejects_unlisted_parquet(
    tmp_path: Path,
) -> None:
    artifact_path = _write_materialization_artifact(tmp_path)
    (tmp_path / "unlisted.parquet").write_bytes(b"not in artifact")

    with pytest.raises(ValueError, match="unlisted parquet"):
        load_objective_materialization_artifact(artifact_path)


def test_objective_materialization_shard_reverification_detects_replacement(
    tmp_path: Path,
) -> None:
    artifact = load_objective_materialization_artifact(
        _write_materialization_artifact(tmp_path)
    )
    shard = artifact.parquet_paths[0]
    before = verify_objective_materialization_shard(artifact, shard)
    replacement = tmp_path / "replacement.parquet"
    replacement.write_bytes(shard.read_bytes())
    replacement.replace(shard)

    with pytest.raises(ValueError, match="stat changed while the shard was consumed"):
        verify_objective_materialization_shard(
            artifact,
            shard,
            previous_stat=before,
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("diff", "post_token_ids", "typed_sources.diff"),
        ("commit_message", "source_text", "typed_sources.commit_message"),
        ("ifim_instruction", "source_text", "typed_sources.ifim_instruction"),
        ("rendered_text_parsing", True, "rendered_text_parsing"),
    ],
)
def test_contract_rejects_fake_or_rendered_objective_sources(
    field: str, bad_value: object, match: str
) -> None:
    contract = _valid_contract()
    contract["typed_sources"][field] = bad_value  # type: ignore[index]

    with pytest.raises(ValueError, match=match):
        validate_objective_contract(contract)


def test_contract_rejects_realized_mix_drift() -> None:
    contract = _valid_contract()
    contract["realized"]["fim"]["samples"] = 0  # type: ignore[index]

    with pytest.raises(ValueError, match="realized.*fim.*planned"):
        validate_objective_contract(contract)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("token_column", "token_ids"),
        ("loss_mask_column", "token_loss_mask"),
        ("loss_mask_alignment", "target_token_v0"),
        ("length_column", None),
        ("document_id_column", "token_source_doc_ids"),
    ],
)
def test_contract_binds_materialized_parquet_columns(
    field: str, bad_value: object
) -> None:
    contract = _valid_contract()
    contract["materialization"][field] = bad_value  # type: ignore[index]

    with pytest.raises(ValueError, match=f"materialization.{field}"):
        validate_objective_contract(contract)


def test_contract_rejects_zero_required_quota_in_small_window() -> None:
    contract = _valid_contract()
    contract["quota_window_samples"] = 5
    contract["planned_samples"]["pre_to_post"] = 0  # type: ignore[index]
    contract["realized"]["pre_to_post"] = {  # type: ignore[index]
        "samples": 0,
        "input_tokens": 0,
        "loss_tokens": 0,
    }
    contract["totals"] = {
        "samples": 5,
        "input_tokens": 15,
        "loss_tokens": 11,
    }

    with pytest.raises(ValueError, match="nonzero planned quota.*Hamilton window"):
        validate_objective_contract(contract)


def test_contract_rejects_whole_dataset_hamilton_instead_of_window_schedule() -> None:
    contract = _valid_contract()
    contract["quota_window_samples"] = 7
    whole_dataset_quotas = {
        "causal_lm": 3,
        "fim": 3,
        "ast_fim": 2,
        "ifim": 2,
        "commit_diff": 2,
        "pre_to_post": 2,
    }
    contract["planned_samples"] = whole_dataset_quotas
    contract["realized"] = {
        task: {
            "samples": samples,
            "input_tokens": samples * 3,
            "loss_tokens": samples * (3 if task == "causal_lm" else 2),
        }
        for task, samples in whole_dataset_quotas.items()
    }
    contract["totals"] = {
        "samples": 14,
        "input_tokens": 42,
        "loss_tokens": 31,
    }

    with pytest.raises(ValueError, match="identical Hamilton quota windows"):
        validate_objective_contract(contract)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("included_in_total_loss", False),
        ("global_weight", "0"),
        ("bce_weight", "0"),
        ("eligible_samples", 0),
        ("positive_edges", 0),
    ],
)
def test_contract_rejects_declared_but_inactive_graph_objective(
    field: str, bad_value: object
) -> None:
    contract = _valid_contract()
    contract["graph_auxiliary"][field] = bad_value  # type: ignore[index]

    with pytest.raises(ValueError, match=f"graph_auxiliary.{field}"):
        validate_objective_contract(contract)


def test_contract_requires_the_bound_pre_topk_graph_selector_beta() -> None:
    contract = _valid_contract()
    contract["graph_auxiliary"].pop("bias_beta")  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="graph_auxiliary.bias_beta"):
        validate_objective_contract(contract)


def test_objective_artifact_is_v2_and_binds_the_canonical_graph_recipe(
    tmp_path: Path,
) -> None:
    artifact = json.loads(
        _write_materialization_artifact(tmp_path).read_text(encoding="utf-8")
    )

    assert artifact["schema"] == "cppmega_objective_materialization_artifact_v2"
    assert artifact["graph_recipe"] == stage1_graph_recipe_binding()


def test_legacy_objective_artifact_requires_regeneration(tmp_path: Path) -> None:
    artifact_path = _write_materialization_artifact(tmp_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["schema"] = "cppmega_objective_materialization_artifact_v1"
    artifact.pop("graph_recipe")
    artifact_payload = dict(artifact)
    artifact_payload.pop("artifact_set_sha256")
    artifact["artifact_set_sha256"] = hashlib.sha256(
        json.dumps(
            artifact_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy.*migration required.*regenerate"):
        load_objective_materialization_artifact(artifact_path)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("pair_mask", "global_causal_v0"),
        ("chunk_edge_expansion", "chunk_starts_only_v0"),
    ],
)
def test_contract_rejects_graph_mask_or_expansion_semantic_drift(
    field: str, bad_value: object
) -> None:
    contract = _valid_contract()
    contract["graph_auxiliary"][field] = bad_value  # type: ignore[index]

    with pytest.raises(ValueError, match=f"graph_auxiliary.{field}"):
        validate_objective_contract(contract)


def test_materialization_tracker_writes_ids_and_exact_counts(tmp_path: Path) -> None:
    contract = validate_objective_contract(_valid_contract())
    prefix = tmp_path / "objective_train"
    tracker = ObjectiveMaterializationTracker(contract, str(prefix))

    for task in TASKS:
        tracker.append(
            task,
            input_tokens=3,
            loss_tokens=3 if task == "causal_lm" else 2,
            graph_edges=5 if task == "causal_lm" else 0,
        )
    manifest = tracker.close()

    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "objective_train_objective_ids.bin", dtype=np.uint8),
        np.array([OBJECTIVE_IDS[task] for task in TASKS], dtype=np.uint8),
    )
    assert manifest["schema"] == OBJECTIVE_CONTRACT_SCHEMA
    assert manifest["sha256"] == contract.sha256
    assert manifest["objective_id_sidecar"] == {
        "path": "objective_train_objective_ids.bin",
        "dtype": "uint8",
        "document_aligned": True,
    }
    json.dumps(manifest)


def test_materialization_tracker_fails_on_row_accounting_drift(
    tmp_path: Path,
) -> None:
    contract = validate_objective_contract(_valid_contract())
    tracker = ObjectiveMaterializationTracker(
        contract, str(tmp_path / "objective_train")
    )
    for task in TASKS:
        tracker.append(
            task,
            input_tokens=4 if task == "fim" else 3,
            loss_tokens=3 if task == "causal_lm" else 2,
            graph_edges=5 if task == "causal_lm" else 0,
        )

    with pytest.raises(ValueError, match="input_tokens.*fim"):
        tracker.close()


def test_graph_enabled_dataset_ingress_requires_objective_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cppmega.megatron import structure_dataset_patch as dataset_patch

    prefix = tmp_path / "legacy_graph_train"
    prefix.with_suffix(".json").write_text(
        json.dumps({"document_count": 1}), encoding="utf-8"
    )
    dataset = SimpleNamespace(dataset=SimpleNamespace(bin_path=str(prefix) + ".bin"))
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", "1")

    with pytest.raises(KeyError, match="objective_contract"):
        dataset_patch._load_sidecar_manifest(dataset)


def test_graph_enabled_boolean_alias_still_requires_objective_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cppmega.megatron import structure_dataset_patch as dataset_patch

    prefix = tmp_path / "legacy_graph_train"
    prefix.with_suffix(".json").write_text(
        json.dumps({"document_count": 1}), encoding="utf-8"
    )
    dataset = SimpleNamespace(dataset=SimpleNamespace(bin_path=str(prefix) + ".bin"))
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "true")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", "1")

    with pytest.raises(KeyError, match="objective_contract"):
        dataset_patch._load_sidecar_manifest(dataset)


@pytest.mark.parametrize(
    ("graph_enabled", "structure_enabled", "message"),
    (
        ("0", "1", "GRAPH_ROUTES_ENABLED"),
        ("1", "0", "STRUCTURE_ENABLED"),
    ),
)
def test_production_objective_ingress_rejects_disabled_routes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    graph_enabled: str,
    structure_enabled: str,
    message: str,
) -> None:
    from cppmega.megatron import structure_dataset_patch as dataset_patch

    prefix = tmp_path / "objective_train"
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "document_count": 1,
                "objective_contract": {},
                "objective_materialization": {},
            }
        ),
        encoding="utf-8",
    )
    dataset = SimpleNamespace(dataset=SimpleNamespace(bin_path=str(prefix) + ".bin"))
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", graph_enabled)
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", structure_enabled)

    with pytest.raises(RuntimeError, match=message):
        dataset_patch._load_sidecar_manifest(dataset)


def test_graph_enabled_dataset_ingress_validates_bound_objective_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cppmega.megatron import structure_dataset_patch as dataset_patch

    prefix = tmp_path / "objective_train"
    contract = validate_objective_contract(_valid_contract())
    tracker = ObjectiveMaterializationTracker(contract, str(prefix))
    for task in TASKS:
        tracker.append(
            task,
            input_tokens=3,
            loss_tokens=3 if task == "causal_lm" else 2,
            graph_edges=5 if task == "causal_lm" else 0,
        )
    embedded = tracker.close()
    objective_artifact = load_objective_materialization_artifact(
        _write_materialization_artifact(tmp_path)
    )
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "document_count": 6,
                "objective_contract": embedded,
                "objective_materialization": materialized_objective_artifact_manifest(
                    objective_artifact
                ),
            }
        ),
        encoding="utf-8",
    )
    dataset = SimpleNamespace(dataset=SimpleNamespace(bin_path=str(prefix) + ".bin"))
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", "1")

    _path, manifest = dataset_patch._load_sidecar_manifest(dataset)

    assert manifest["objective_contract"]["sha256"] == contract.sha256


@pytest.mark.parametrize(
    ("env_name", "env_value", "field"),
    (
        ("CPPMEGA_DSA_GRAPH_AUX_WEIGHT", "2.0", "global_weight"),
        ("CPPMEGA_DSA_INDEXER_LOSS_COEFF", "0.002", "indexer_weight"),
        ("CPPMEGA_DSA_GRAPH_LAYER_WEIGHT", "0.5", "layer_weight"),
    ),
)
def test_graph_enabled_dataset_ingress_rejects_runtime_weight_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_name: str,
    env_value: str,
    field: str,
) -> None:
    from cppmega.megatron import structure_dataset_patch as dataset_patch

    prefix = tmp_path / "objective_train"
    contract = validate_objective_contract(_valid_contract())
    tracker = ObjectiveMaterializationTracker(contract, str(prefix))
    for task in TASKS:
        tracker.append(
            task,
            input_tokens=3,
            loss_tokens=3 if task == "causal_lm" else 2,
            graph_edges=5 if task == "causal_lm" else 0,
        )
    objective_artifact = load_objective_materialization_artifact(
        _write_materialization_artifact(tmp_path)
    )
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "document_count": 6,
                "objective_contract": tracker.close(),
                "objective_materialization": materialized_objective_artifact_manifest(
                    objective_artifact
                ),
            }
        ),
        encoding="utf-8",
    )
    dataset = SimpleNamespace(dataset=SimpleNamespace(bin_path=str(prefix) + ".bin"))
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", "1")
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValueError, match=field + ".*contract"):
        dataset_patch._load_sidecar_manifest(dataset)
