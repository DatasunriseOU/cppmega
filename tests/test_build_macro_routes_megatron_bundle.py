from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega.data.nanochat_pipeline.packed_rows_schema import (
    NUM_DOCS_COLUMN,
    SOURCE_COMMIT_HASHES_COLUMN,
    SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
    SOURCE_PR_NUMBERS_COLUMN,
)
from cppmega.data.pr_primary_membership import (
    PRIMARY_PR_MEMBERSHIP_TABLE,
    PRIMARY_PR_MEMBERSHIP_POLICY,
    PRIMARY_PR_MEMBERSHIP_SCHEMA,
    primary_commit_artifact_binding,
    publish_primary_pr_membership_inputs,
)
from cppmega.data.source_conveyor_composition import SourceComposition
from scripts.ci_source_binding_projection import (
    REVIEWED_FROZEN_PARSER_FROM_SHA256,
    REVIEWED_FROZEN_PARSER_LINEAGE,
    REVIEWED_FROZEN_PARSER_SINK_SHA256,
    REVIEWED_FROZEN_PARSER_UPGRADE_REASON,
    REVIEWED_PRIMARY_EQUIVALENT_PARSER_SHA256,
    REVIEWED_PRIMARY_EQUIVALENT_PARSER_UPGRADE_REASON,
)
import scripts.data.build_macro_routes_megatron_bundle as builder
import scripts.data.prepare_ci_objective_source_manifest as objective_manifest
import scripts.data.publish_megatron_bundle_to_nebius_s3 as publisher
from scripts.canonical_parquet_ledger import CanonicalParquetLedgerWriter
from scripts.data.build_macro_routes_megatron_bundle import (
    BUNDLE_KNOWN_LIMITATIONS,
    _acquire_build_lock,
    _artifact_set_sha256,
    build_arg_parser,
    _canonical_sha256,
    _ensure_partial_build_plan,
    _ensure_partial_snapshot_plan,
    _load_ci_manifest_allowlist,
    _load_pr_export_allowlist,
    _parse_objective_artifacts,
    _portable_bucket_results,
    _producer_binding_from_local_revision,
    _publish_validated_bundle,
    _run_snapshot_audit,
    _snapshot_sources,
    _stage_data_contracts,
    _stage_tokenizer,
    _verify_local_cppmega_revision,
    _validate_objective_source_binding,
    _write_repaired_snapshot_manifest,
)
from tests.test_megatron_objective_contract import (
    _valid_contract,
    _write_materialization_artifact,
)

_CI_PRODUCER_COMMIT = "c" * 40
_CI_PRODUCER_TREE_SHA256 = "d" * 64


def _write_ci_generation(
    root: Path,
    *,
    buckets: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384),
) -> Path:
    source_inventory = [
        {
            "name": "ci_logs_enriched.jsonl",
            "path": "/corpus/ci_logs_enriched.jsonl",
            "size": 123,
            "mtime_ns": 456,
            "sha256": "a" * 64,
        },
        {
            "name": "ci_paired_enriched.jsonl",
            "path": "/corpus/ci_paired_enriched.jsonl",
            "size": 789,
            "mtime_ns": 987,
            "sha256": "e" * 64,
        },
    ]
    bucket_receipts: dict[str, dict[str, object]] = {}
    for bucket in buckets:
        bucket_dir = root / str(bucket)
        bucket_dir.mkdir(parents=True, exist_ok=True)
        parquet = bucket_dir / f"ci_packed_{bucket}.parquet"
        parquet.write_bytes(f"ci-{bucket}".encode("ascii"))
        persisted = {
            "schema": builder.CI_BUCKET_MANIFEST_SCHEMA,
            "kind": "ci",
            "bucket_seq_length": bucket,
            "fragments": 1,
            "packed_rows": 1,
            "valid_tokens": bucket,
            "trained_tokens": bucket - 1,
            "capacity_tokens": bucket,
            "packing_overflow_docs": 0,
            "parquet": {
                "path": parquet.name,
                "size": parquet.stat().st_size,
                "sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
            },
            "domain_kind_counts": {"BUILD_DIAGNOSTIC": 1},
            "fixed_width_verified": True,
        }
        bucket_manifest = bucket_dir / "manifest.json"
        bucket_manifest.write_text(
            json.dumps(persisted, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bucket_receipts[str(bucket)] = {
            **persisted,
            "manifest": {
                "path": f"{bucket}/manifest.json",
                "sha256": hashlib.sha256(bucket_manifest.read_bytes()).hexdigest(),
            },
            "parquet": {
                **persisted["parquet"],
                "path": f"{bucket}/{parquet.name}",
            },
        }
    total_tokens = sum(buckets)
    manifest = {
        "schema": builder.CI_MANIFEST_SCHEMA,
        "kind": "ci",
        "seq_lengths": list(buckets),
        "source_inventory": source_inventory,
        "source_inventory_sha256": _canonical_sha256(source_inventory),
        "source_completion": {
            "schema": builder.CI_LOG_COMPLETION_SCHEMA,
            "status": "complete",
            "receipt_sha256": "f" * 64,
            "unique_job_count": len(buckets),
            "fetched_count": len(buckets),
            "expired_count": 0,
            "too_short_count": 0,
            "unresolved_count": 0,
            "output": {
                "row_count": len(buckets),
                "size": source_inventory[0]["size"],
                "sha256": source_inventory[0]["sha256"],
            },
            "state": {
                "row_count": len(buckets),
                "sha256": "9" * 64,
            },
            "expired_jobs": [],
        },
        "counters": {
            "input_docs": len(buckets),
            "tokenized_docs": len(buckets),
            "source_tokens": total_tokens,
            "fragment_tokens": total_tokens,
            "fragments": len(buckets),
            "split_source_docs": 0,
            "cross_boundary_chunk_edges": 2,
            "cross_boundary_token_edges": 3,
            "malformed_json_rows": 0,
            "empty_text_docs": 0,
            "zero_token_docs": 0,
            "normalization_rejects": 0,
            "packing_overflow_docs": 0,
            "unexpected_rejects": 0,
        },
        "split_policy": {
            "schema": "cppmega_ci_lossless_token_fragmentation_v1",
            "token_loss": 0,
            "cross_boundary_edges_are_counted": True,
        },
        "producer": {
            "script": "tokenize_ci_enriched.py",
            "script_sha256": "b" * 64,
            "code_revision": {
                "schema": "cppmega_ci_code_revision_v2",
                "schema_version": 2,
                "repository_identity": "cppmega.mlx",
                "git_commit": _CI_PRODUCER_COMMIT,
                "source_tree_sha256": _CI_PRODUCER_TREE_SHA256,
                "dirty": False,
                "status_sha256": hashlib.sha256(b"").hexdigest(),
            },
        },
        "buckets": bucket_receipts,
        "verification": {
            "fixed_width_all_rows": True,
            "source_tokens_equal_fragment_tokens": True,
            "unexpected_rejects": 0,
            "packing_overflow_docs": 0,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_content_store_ci_export(
    root: Path,
    *,
    buckets: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384),
    schema: str = builder.CI_CONTENT_STORE_EXPORT_SCHEMA,
) -> Path:
    artifacts: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    fragments = 0
    capacity_tokens = 0
    valid_tokens = 0
    trained_tokens = 0
    for bucket in buckets:
        bucket_dir = root / str(bucket)
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for shard, split in enumerate(("train", "validation")):
            parquet = (
                bucket_dir
                / f"ci-case5-{split}-{bucket}-{shard:06d}.parquet"
            )
            pq.write_table(
                pa.table({"fixture_row": [bucket]}),
                parquet,
                compression="zstd",
                compression_level=9,
            )
            row_valid_tokens = bucket - shard - 1
            row_trained_tokens = row_valid_tokens - 1
            relative = parquet.relative_to(root).as_posix()
            artifacts.append(
                {
                    "path": relative,
                    "kind": "case5_parquet",
                    "split": split,
                    "bucket": bucket,
                    "rows": 1,
                    "byte_size": parquet.stat().st_size,
                    "sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
                }
            )
            audits.append(
                {
                    "path": relative,
                    "rows": 1,
                    "capacity_tokens": bucket,
                    "valid_tokens": row_valid_tokens,
                    "trained_tokens": row_trained_tokens,
                    "bad_files": 0,
                    "bad_rows": 0,
                }
            )
            fragments += 1
            capacity_tokens += bucket
            valid_tokens += row_valid_tokens
            trained_tokens += row_trained_tokens

    for kind, filename, domain in (
        (
            "representative_ledger",
            "representative_ledger.parquet",
            "cppmega-ci-case5-representative-ledger-v1",
        ),
        ("fragment_ledger", "fragment_ledger.parquet", None),
        ("dropped_graph_edges", "dropped_graph_edges.parquet", None),
        ("representative_metadata", "representative_metadata.parquet", None),
        (
            "excluded_opaque_artifacts",
            "excluded_opaque_artifacts.parquet",
            "cppmega-ci-case5-excluded-opaque-artifact-ledger-v1",
        ),
        (
            "excluded_training_scope",
            "excluded_training_scope.parquet",
            "cppmega-ci-case5-excluded-training-scope-ledger-v1",
        ),
        (
            "source_binding_projection",
            "source_binding_projection.parquet",
            "cppmega-ci-source-binding-projection-ledger-v1",
        ),
    ):
        ledger = root / filename
        ledger.unlink(missing_ok=True)
        ledger_writer = CanonicalParquetLedgerWriter(
            ledger,
            domain=domain,
        )
        ledger_writer.append({})
        ledger_writer.close()
        artifacts.append(
            {
                "path": filename,
                "kind": kind,
                "rows": ledger_writer.count,
                "byte_size": ledger.stat().st_size,
                "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            }
        )
    occurrence_metadata = root / "occurrence_metadata.parquet"
    occurrence_metadata.unlink(missing_ok=True)
    occurrence_writer = CanonicalParquetLedgerWriter(
        occurrence_metadata,
        domain="cppmega-ci-case5-occurrence-metadata-ledger-v1",
    )
    occurrence_writer.append(
        {"schema": "cppmega_ci_case5_occurrence_metadata_v1"}
    )
    occurrence_writer.close()
    artifacts.append(
        {
            "path": occurrence_metadata.name,
            "kind": "occurrence_metadata",
            "rows": occurrence_writer.count,
            "byte_size": occurrence_metadata.stat().st_size,
            "sha256": hashlib.sha256(
                occurrence_metadata.read_bytes()
            ).hexdigest(),
        }
    )

    payload_tokens = 20_000_000_123
    current_parser_sha256 = builder.target_parser_script_sha256()
    excluded_scope_artifact = next(
        artifact
        for artifact in artifacts
        if artifact["kind"] == "excluded_training_scope"
    )
    exporter_path = (
        builder.REPO_ROOT / "scripts/export_ci_content_store_case5.py"
    )
    manifest = {
        "schema": schema,
        "status": "complete",
        "exporter_script_sha256": hashlib.sha256(
            exporter_path.read_bytes()
        ).hexdigest(),
        "input_store": {
            "schema": "cppmega_ci_content_store_v1",
            "receipt_schema": "cppmega_ci_content_store_receipt_v1",
            "receipt_sha256": "5" * 64,
            "policy_sha256": "6" * 64,
            "sqlite_schema_sha256": "7" * 64,
            "logical_content_set_sha256": "8" * 64,
            "logical_token_sequence_set_sha256": "9" * 64,
            "occurrence_set_sha256": "a" * 64,
            "sqlite_logical_sha256": "b" * 64,
            "pack_hashes": [{"path": "pack-00000000.cicp", "sha256": "c" * 64}],
            "verified_before_export": True,
            "unchanged_after_export": True,
        },
        "input_fetch_state": {
            "schema": builder.CI_FETCH_STATE_SCHEMA,
            "artifact": {
                "path": str(root / "fetch_state.sqlite3"),
                "sha256": "1" * 64,
            },
            "sqlite_schema_sha256": "d" * 64,
            "sqlite_logical_sha256": "2" * 64,
            "sidecar_set_sha256": "3" * 64,
            "settings": {
                "fetcher_script_sha256": "e" * 64,
                "parser_script_sha256": current_parser_sha256,
                "content_store_script_sha256": "0" * 64,
            },
        },
        "parser_generation_policy": {
            "mode": "current-singleton-required",
            "expected_current_parser_script_sha256": (
                current_parser_sha256
            ),
            "observed_parser_lineage": [current_parser_sha256],
            "current_singleton": True,
        },
        "case5_contract": {
            "buckets": list(buckets),
            "overflow_rows": 0,
            "parquet_shard_max_rows": 512,
            "parquet_layout": "bucket-first-split-in-filename-v1",
            "parquet_compression": {"codec": "zstd", "level": 9},
        },
        "eligibility": {
            "policy": {
                "schema": (
                    "cppmega_ci_primary_training_eligibility_policy_v1"
                ),
                "primary_route": "primary_cpp_sql_build_test",
                "training_scope": builder.training_scope_policy(),
                "exact_step_propagation": {
                    "schema": (
                        "cppmega_ci_exact_step_scope_propagation_v1"
                    ),
                    "key": ["repo", "run_attempt", "job", "step"],
                    "primary_priority": True,
                    "opaque_members_never_inherit": True,
                    "cross_step_propagation": False,
                    "cross_job_propagation": False,
                    "cross_attempt_propagation": False,
                },
            },
            "target_exact_unique_payload_tokens": 20_000_000_000,
            "target_source": "explicit_export_requirement",
            "cas_acquisition_target_exact_unique_payload_tokens": 20_200_000_000,
            "cas_reserve_exact_unique_payload_tokens": 200_000_000,
            "target_met": True,
            "eligible": {
                "unique_token_sequences": len(buckets),
                "exact_unique_payload_tokens": payload_tokens,
            },
            "conservation": {
                "exact_unique_payload_tokens": True,
                "unique_token_sequences": True,
            },
            "excluded_training_scope_occurrences": {
                "members": 1,
                "occurrences": int(excluded_scope_artifact["rows"]),
                "summed_exact_tokens_with_occurrence_multiplicity": 1,
                "ledger_schema": (
                    "cppmega_ci_case5_excluded_training_scope_v1"
                ),
                "ledger": str(excluded_scope_artifact["path"]),
                "ledger_sha256": "7" * 64,
                "ledger_artifact_sha256": str(
                    excluded_scope_artifact["sha256"]
                ),
            },
        },
        "representatives": {
            "count": len(buckets),
            "ledger_sha256": "4" * 64,
        },
        "source_binding_projection": {
            "mode": "current_audit",
            "projection_script_sha256": (
                builder.projection_script_sha256()
            ),
            "input_parser_script_sha256": current_parser_sha256,
            "target_parser_script_sha256": current_parser_sha256,
            "coverage": {"occurrence_count": occurrence_writer.count},
        },
        "occurrence_metadata": {
            "schema": "cppmega_ci_case5_occurrence_metadata_v1",
            "scope": "one-record-per-frozen-cas-occurrence",
            "count": occurrence_writer.count,
            "input_occurrence_set_sha256": "a" * 64,
            "artifact": occurrence_metadata.name,
            "artifact_sha256": hashlib.sha256(
                occurrence_metadata.read_bytes()
            ).hexdigest(),
            "physical_format": {
                "container": "parquet",
                "compression": "zstd",
                "record_encoding": "canonical-json",
            },
        },
        "counts": {
            "representatives": len(buckets),
            "fragments": fragments,
            "payload_tokens": payload_tokens,
            "valid_tokens": valid_tokens,
            "trained_tokens": trained_tokens,
            "capacity_tokens": capacity_tokens,
        },
        "graph_accounting": {
            "cross_chunk_outbound_edges_dropped": 2,
            "cross_fragment_edges_dropped": 3,
        },
        "artifacts": artifacts,
        "validation": {
            "all_passed": True,
            "fixed_widths": True,
            "zero_overflow": True,
            "payload_conserved": True,
            "payload_identity_and_order_verified": True,
            "all_case5_parquet_zstd": True,
            "post_normalize_pack_sidecars_and_edges_verified": True,
            "case5_audit": audits,
        },
    }
    if schema == builder.PRODUCTION_CI_CONTENT_STORE_EXPORT_SCHEMA:
        manifest.update(
            {
                "completion_mode": "inventory-exhaustive",
                "production_complete": True,
                "acquisition_provenance": {
                    "completion_mode": "inventory-exhaustive",
                    "production_complete": True,
                    "inventory": {
                        "path": str(root / "inventory.sqlite3"),
                        "sha256": "d" * 64,
                        "logical_sha256": "e" * 64,
                        "receipt_path": str(
                            root / "inventory_receipt.json"
                        ),
                        "receipt_sha256": "f" * 64,
                    },
                    "fetch": {
                        "state_path": str(root / "fetch_state.sqlite3"),
                        "state_sha256": "1" * 64,
                        "receipt_path": str(root / "fetch_receipt.json"),
                        "receipt_sha256": "2" * 64,
                        "attempt_set_sha256": "3" * 64,
                        "terminal_proof_sha256": "4" * 64,
                    },
                    "store": {
                        "path": str(root / "content_store"),
                        "receipt_path": str(root / "store_receipt.json"),
                        "receipt_sha256": "5" * 64,
                    },
                    "merge": {
                        "receipt_path": str(root / "merge_receipt.json"),
                        "receipt_sha256": "6" * 64,
                        "schema": (
                            "cppmega_ci_stream_shard_union_receipt_v3"
                        ),
                    },
                },
            }
        )
    manifest_path = root / "export_receipt.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_repaired_ci_snapshot(
    receipt_path: Path,
    *,
    snapshot_root: Path,
) -> Path:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for artifact in receipt["artifacts"]:
        if artifact["kind"] != "case5_parquet":
            continue
        relative = Path(artifact["path"])
        source = receipt_path.parent / relative
        target = snapshot_root / "ci" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source)
        if artifact["split"] == "train":
            table = table.replace_schema_metadata({b"boundary-repaired": b"true"})
        pq.write_table(
            table,
            target,
            compression="zstd",
            compression_level=9,
        )
        snapshot_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        boundary_repaired = (
            target.stat().st_size != artifact["byte_size"]
            or snapshot_sha256 != artifact["sha256"]
        )
        records.append(
            {
                "kind": "ci",
                "bucket": artifact["bucket"],
                "snapshot": f"ci/{relative.as_posix()}",
                "size": target.stat().st_size,
                "rows": artifact["rows"],
                "source_sha256": artifact["sha256"],
                "snapshot_sha256": snapshot_sha256,
                "boundary_repaired": boundary_repaired,
            }
        )
    repaired = {
        "schema": "cppmega_repaired_parquet_snapshot_v1",
        "created_at": "2026-08-02T00:00:00+00:00",
        "source_manifest_sha256": "1" * 64,
        "file_count": len(records),
        "changed_files": sum(
            int(record["boundary_repaired"]) for record in records
        ),
        "files": records,
    }
    repaired_path = snapshot_root / "repaired_manifest.json"
    repaired_path.write_text(json.dumps(repaired), encoding="utf-8")
    return repaired_path


def test_ci_objective_source_manifest_uses_repaired_train_shards_only(
    tmp_path: Path,
) -> None:
    original_ci_root = tmp_path / "original-ci"
    receipt_path = _write_content_store_ci_export(original_ci_root)
    snapshot_root = tmp_path / "snapshot"
    repaired_manifest_path = _write_repaired_ci_snapshot(
        receipt_path,
        snapshot_root=snapshot_root,
    )
    seed_root = snapshot_root
    code = seed_root / "code" / "1024" / "code.parquet"
    commit = seed_root / "commits" / "1024" / "commit.parquet"
    for path in (code, commit):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"value": [1]}), path, compression="zstd")

    manifest = objective_manifest.build_source_pool_manifest(
        ci_root=snapshot_root / "ci",
        ci_receipt_path=receipt_path,
        repaired_manifest_path=repaired_manifest_path,
        objective_seed_root=seed_root,
        seed_globs=("code/1024/*.parquet", "commits/1024/*.parquet"),
        buckets=builder.DEFAULT_BUCKETS,
        producer={"script": "fixture"},
    )

    assert manifest["schema"] == objective_manifest.MANIFEST_SCHEMA
    assert manifest["algorithm"] == objective_manifest.SCHEDULE
    assert manifest["sequence_lengths"] == list(builder.DEFAULT_BUCKETS)
    assert [
        record["path"] for record in manifest["objective_seed"]["files"]
    ] == ["code/1024/code.parquet", "commits/1024/commit.parquet"]
    primary = manifest["primary_ci"]["files_by_sequence_length"]["1024"]
    assert [record["path"] for record in primary] == [
        "1024/ci-case5-train-1024-000000.parquet"
    ]
    repaired_train = (
        snapshot_root / "ci/1024/ci-case5-train-1024-000000.parquet"
    )
    original_train = (
        original_ci_root / "1024/ci-case5-train-1024-000000.parquet"
    )
    assert primary[0]["size_bytes"] == repaired_train.stat().st_size
    assert primary[0]["sha256"] == hashlib.sha256(
        repaired_train.read_bytes()
    ).hexdigest()
    assert primary[0]["sha256"] != hashlib.sha256(
        original_train.read_bytes()
    ).hexdigest()
    completion = manifest["ci_export"]["source_completion"]
    assert completion["status"] == "complete"
    assert "production_complete" not in completion


def test_ci_objective_source_manifest_accepts_full_large_context_ladder(
    tmp_path: Path,
) -> None:
    buckets = builder.SUPPORTED_CI_BUCKETS
    original_ci_root = tmp_path / "original-ci"
    receipt_path = _write_content_store_ci_export(
        original_ci_root,
        buckets=buckets,
    )
    snapshot_root = tmp_path / "snapshot"
    repaired_manifest_path = _write_repaired_ci_snapshot(
        receipt_path,
        snapshot_root=snapshot_root,
    )
    seed = snapshot_root / "code" / "1024" / "code.parquet"
    seed.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"value": [1]}), seed, compression="zstd")

    manifest = objective_manifest.build_source_pool_manifest(
        ci_root=snapshot_root / "ci",
        ci_receipt_path=receipt_path,
        repaired_manifest_path=repaired_manifest_path,
        objective_seed_root=snapshot_root,
        seed_globs=("code/1024/*.parquet",),
        buckets=buckets,
        producer={"script": "fixture"},
    )

    assert manifest["sequence_lengths"] == list(buckets)
    assert set(manifest["primary_ci"]["files_by_sequence_length"]) == {
        str(bucket) for bucket in buckets
    }
    assert objective_manifest._parse_buckets(
        ",".join(str(bucket) for bucket in buckets)
    ) == buckets
    with pytest.raises(ValueError, match="exactly one of"):
        objective_manifest._parse_buckets("1024,2048,4096,8192,16384,32768")


def test_ci_objective_source_manifest_rejects_repaired_byte_drift(
    tmp_path: Path,
) -> None:
    original_ci_root = tmp_path / "original-ci"
    receipt_path = _write_content_store_ci_export(original_ci_root)
    snapshot_root = tmp_path / "snapshot"
    repaired_manifest_path = _write_repaired_ci_snapshot(
        receipt_path,
        snapshot_root=snapshot_root,
    )
    train = snapshot_root / "ci/1024/ci-case5-train-1024-000000.parquet"
    table = pq.read_table(train).replace_schema_metadata(
        {b"boundary-repaired": b"drifted"}
    )
    pq.write_table(table, train, compression="zstd", compression_level=9)
    seed = snapshot_root / "code/1024/code.parquet"
    seed.parent.mkdir(parents=True)
    pq.write_table(pa.table({"value": [1]}), seed, compression="zstd")

    with pytest.raises(RuntimeError, match="snapshot bytes drifted"):
        objective_manifest.build_source_pool_manifest(
            ci_root=snapshot_root / "ci",
            ci_receipt_path=receipt_path,
            repaired_manifest_path=repaired_manifest_path,
            objective_seed_root=snapshot_root,
            seed_globs=("code/1024/*.parquet",),
            buckets=builder.DEFAULT_BUCKETS,
            producer={"script": "fixture"},
        )


def _write_pr_export(
    root: Path,
    *,
    buckets: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384),
) -> tuple[Path, SourceComposition, Path]:
    scan_id = "1" * 64
    commit_root = root.parent / "commits"
    composition_allowlist: dict[tuple[str, int], dict[str, int]] = {}
    for bucket in buckets:
        commit_bucket = commit_root / str(bucket)
        commit_bucket.mkdir(parents=True, exist_ok=True)
        commit_parquet = commit_bucket / "primary.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "repo": "owner/repo",
                        NUM_DOCS_COLUMN: 1,
                        SOURCE_PR_NUMBERS_COLUMN: [1],
                        SOURCE_COMMIT_HASHES_COLUMN: ["6" * 40],
                        SOURCE_HAS_PR_DISCUSSIONS_COLUMN: [True],
                    }
                ],
                schema=pa.schema(
                    [
                        pa.field("repo", pa.string()),
                        pa.field(NUM_DOCS_COLUMN, pa.int32()),
                        pa.field(
                            SOURCE_PR_NUMBERS_COLUMN,
                            pa.list_(pa.int64()),
                        ),
                        pa.field(
                            SOURCE_COMMIT_HASHES_COLUMN,
                            pa.list_(pa.string()),
                        ),
                        pa.field(
                            SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
                            pa.list_(pa.bool_()),
                        ),
                    ]
                ),
            ),
            commit_parquet,
            compression="zstd",
        )
        composition_allowlist[("code", bucket)] = {"unused.parquet": 1}
        composition_allowlist[("commits", bucket)] = {
            commit_parquet.name: 1
        }
    composition = SourceComposition(
        allowlist=composition_allowlist,
        receipt={
            "schema": "cppmega_source_conveyor_composition_v1",
            "status": "complete",
            "plan_sha256": "7" * 64,
        },
        plan_path=root.parent / "source_composition.json",
        dedup_receipt_path=root.parent / "dedup.json",
        run_files=(),
    )
    membership_digest = hashlib.sha256()
    membership_key = b"owner/repo\x001"
    membership_digest.update(len(membership_key).to_bytes(8, "big"))
    membership_digest.update(membership_key)
    membership = {
        "schema": PRIMARY_PR_MEMBERSHIP_SCHEMA,
        "policy": PRIMARY_PR_MEMBERSHIP_POLICY,
        "scan_id": scan_id,
        "commit_artifacts": primary_commit_artifact_binding(
            source_composition=composition,
            commit_root=commit_root,
            buckets=buckets,
        ),
        "rows": len(buckets),
        "source_docs": len(buckets),
        "source_docs_with_pr_number": len(buckets),
        "source_docs_with_commit_sha": len(buckets),
        "selected_pr_count": 1,
        "sha_only_matched_source_docs": 0,
        "unmatched_commit_sha_source_docs": 0,
        "selected_membership_sha256": membership_digest.hexdigest(),
        "validation": {
            "source_composition_complete": True,
            "exact_allowlisted_commit_artifacts": True,
            "exact_source_doc_shapes": True,
            "exact_scan_membership": True,
            "direct_pr_sha_conflicts": 0,
        },
    }
    membership_conn = sqlite3.connect(":memory:")
    membership_conn.row_factory = sqlite3.Row
    try:
        membership_conn.execute(
            f"""
            CREATE TEMP TABLE {PRIMARY_PR_MEMBERSHIP_TABLE} (
                repo TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                PRIMARY KEY (repo, pr_number)
            )
            """
        )
        membership_conn.execute(
            f"""
            INSERT INTO {PRIMARY_PR_MEMBERSHIP_TABLE}(repo, pr_number)
            VALUES (?, ?)
            """,
            ("owner/repo", 1),
        )
        membership, membership_receipt = (
            publish_primary_pr_membership_inputs(
                membership_conn,
                output_root=root,
                membership=membership,
            )
        )
    finally:
        membership_conn.close()
    artifacts: list[dict[str, object]] = []
    for bucket in buckets:
        bucket_root = root / str(bucket)
        bucket_root.mkdir(parents=True, exist_ok=True)
        parquet = (
            bucket_root
            / f"pr_discussions_all_{scan_id[:12]}_{bucket:08d}.parquet"
        )
        parquet.write_bytes(f"pr-{bucket}".encode("ascii"))
        artifacts.append(
            {
                "path": parquet.relative_to(root).as_posix(),
                "bucket": bucket,
                "rows": 1,
                "valid_tokens": bucket - 1,
                "pad_tokens": 1,
                "capacity_tokens": bucket,
                "byte_size": parquet.stat().st_size,
                "sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
            }
        )
    completion = {
        "schema": "cppmega_pr_completion_v2",
        "status": "verified",
        "receipt_sha256": "2" * 64,
        "pr_store_sha256": "3" * 64,
        "repo_list_sha256": "4" * 64,
        "expected_repos_sha256": "5" * 64,
        "scan_id": scan_id,
        "expected_repo_count": 1,
        "stored_pr_count": 1,
        "unverified_store_pr_count": 0,
    }
    done_manifest = root / "_done.json"
    done_manifest.write_text(
        json.dumps(
            {
                "schema": "cppmega_pr_parquet_export_manifest_v3",
                "status": "complete",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    exporter = builder.REPO_ROOT / "scripts/pr_ingest/export_pr_parquet.py"
    receipt = {
        "schema": builder.PR_EXPORT_SCHEMA,
        "status": "complete",
        "source": "pr",
        "scan_id": scan_id,
        "pr_completion": completion,
        "primary_membership": membership,
        "primary_membership_receipt": membership_receipt,
        "exporter_script_sha256": hashlib.sha256(exporter.read_bytes()).hexdigest(),
        "target_lengths": list(buckets),
        "selected_pr_count": 1,
        "rendered_docs": 1,
        "manifest": {
            "path": str(done_manifest.resolve()),
            "sha256": hashlib.sha256(done_manifest.read_bytes()).hexdigest(),
        },
        "artifacts": artifacts,
        "validation": {
            "exact_scan_membership": True,
            "exact_primary_commit_membership": True,
            "portable_primary_membership_verified": True,
            "input_revalidated_after_export": True,
            "document_conservation": True,
            "all_requested_buckets_present": True,
            "artifact_hashes_verified": True,
        },
    }
    receipt_path = root / "export_receipt.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    return receipt_path, composition, commit_root


def test_ci_manifest_allowlist_binds_five_buckets_and_lossless_counters(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_ci_generation(ci_root)

    allowed, metadata = _load_ci_manifest_allowlist(
        manifest_path,
        ci_root,
        builder.DEFAULT_BUCKETS,
        cppmega_mlx_commit=_CI_PRODUCER_COMMIT,
        cppmega_mlx_tree_sha256=_CI_PRODUCER_TREE_SHA256,
    )

    assert set(allowed) == {
        ("ci", bucket) for bucket in builder.DEFAULT_BUCKETS
    }
    assert metadata["valid_tokens"] == sum(builder.DEFAULT_BUCKETS)
    assert metadata["cross_boundary_chunk_edges"] == 2
    assert metadata["cross_boundary_token_edges"] == 3


def test_content_store_export_rejects_inconsistent_cas_reserve(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_content_store_ci_export(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["eligibility"]["cas_reserve_exact_unique_payload_tokens"] += 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="representative/token conservation drifted",
    ):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )


def test_pr_export_allowlist_binds_exact_scan_and_every_artifact(
    tmp_path: Path,
) -> None:
    pr_root = tmp_path / "pr"
    receipt_path, composition, commit_root = _write_pr_export(pr_root)

    allowed, metadata = _load_pr_export_allowlist(
        receipt_path,
        pr_root,
        (1024, 2048, 4096, 8192, 16384),
        source_composition=composition,
        commit_root=commit_root,
    )

    assert metadata["schema"] == builder.PR_EXPORT_SCHEMA
    assert metadata["input_docs"] == 1
    assert metadata["fragments"] == 5
    assert metadata["source_binding"]["scan_id"] == "1" * 64
    assert set(allowed) == {
        ("pr", 1024),
        ("pr", 2048),
        ("pr", 4096),
        ("pr", 8192),
        ("pr", 16384),
    }
    assert all(len(files) == 1 for files in allowed.values())

    (pr_root / "1024" / "orphan.parquet").write_bytes(b"orphan")
    with pytest.raises(RuntimeError, match="inventory differs"):
        _load_pr_export_allowlist(
            receipt_path,
            pr_root,
            (1024, 2048, 4096, 8192, 16384),
            source_composition=composition,
            commit_root=commit_root,
        )


def test_pr_export_allowlist_rejects_portable_membership_receipt_drift(
    tmp_path: Path,
) -> None:
    pr_root = tmp_path / "pr"
    receipt_path, composition, commit_root = _write_pr_export(pr_root)
    membership_receipt = pr_root / "primary_pr_membership_receipt.json"
    membership_receipt.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="primary PR membership receipt binding drifted",
    ):
        _load_pr_export_allowlist(
            receipt_path,
            pr_root,
            builder.DEFAULT_BUCKETS,
            source_composition=composition,
            commit_root=commit_root,
        )


def test_pr_export_allowlist_rejects_portable_membership_artifact_drift(
    tmp_path: Path,
) -> None:
    pr_root = tmp_path / "pr"
    receipt_path, composition, commit_root = _write_pr_export(pr_root)
    artifact = pr_root / "primary_pr_membership.parquet"
    parquet = pq.ParquetFile(artifact)
    mutated = pa.Table.from_pylist(
        [{"repo": "owner/repo", "pr_number": 2}],
        schema=parquet.schema_arrow,
    )
    pq.write_table(mutated, artifact, compression="zstd")

    with pytest.raises(
        RuntimeError,
        match="primary PR membership receipt binding drifted",
    ):
        _load_pr_export_allowlist(
            receipt_path,
            pr_root,
            builder.DEFAULT_BUCKETS,
            source_composition=composition,
            commit_root=commit_root,
        )


def test_content_store_export_allowlist_binds_all_split_shards(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_content_store_ci_export(ci_root)

    allowed, metadata = _load_ci_manifest_allowlist(
        manifest_path,
        ci_root,
        builder.DEFAULT_BUCKETS,
        cppmega_mlx_commit="unused-for-content-store-export",
        cppmega_mlx_tree_sha256="unused-for-content-store-export",
    )

    assert {
        key: len(value) for key, value in allowed.items()
    } == {
        ("ci", bucket): 2 for bucket in builder.DEFAULT_BUCKETS
    }
    assert metadata["schema"] == builder.CI_CONTENT_STORE_EXPORT_SCHEMA
    assert metadata["valid_tokens"] == sum(
        2 * bucket - 3 for bucket in builder.DEFAULT_BUCKETS
    )
    assert metadata["cross_boundary_chunk_edges"] == 2
    assert metadata["cross_boundary_token_edges"] == 3
    assert metadata["source_completion"][
        "cas_reserve_exact_unique_payload_tokens"
    ] == 200_000_000


def test_content_store_export_allowlist_accepts_explicit_32k_64k_ladder(
    tmp_path: Path,
) -> None:
    buckets = builder.SUPPORTED_CI_BUCKETS
    ci_root = tmp_path / "ci-large-context"
    manifest_path = _write_content_store_ci_export(ci_root, buckets=buckets)

    allowed, metadata = _load_ci_manifest_allowlist(
        manifest_path,
        ci_root,
        buckets,
        cppmega_mlx_commit="unused-for-content-store-export",
        cppmega_mlx_tree_sha256="unused-for-content-store-export",
    )

    assert set(allowed) == {("ci", bucket) for bucket in buckets}
    assert all(len(files) == 2 for files in allowed.values())
    assert metadata["allowlist_counts"] == {
        f"ci/{bucket}": 2 for bucket in buckets
    }

    with pytest.raises(RuntimeError, match="unsupported CI export CASE5 bucket contract"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused-for-content-store-export",
            cppmega_mlx_tree_sha256="unused-for-content-store-export",
        )


def test_content_store_export_accepts_only_reviewed_primary_equivalent_transition(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_content_store_ci_export(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = builder.target_parser_script_sha256()
    upgrade = {
        "binding_key": "parser_script_sha256",
        "from_sha256": REVIEWED_PRIMARY_EQUIVALENT_PARSER_SHA256,
        "to_sha256": current,
        "reason": REVIEWED_PRIMARY_EQUIVALENT_PARSER_UPGRADE_REASON,
        "upgraded_at": "2026-07-30T13:38:03Z",
    }
    manifest["input_fetch_state"]["summary"] = {
        "binding_upgrades": [upgrade]
    }
    manifest["parser_generation_policy"] = {
        "mode": "reviewed-primary-equivalent-transition",
        "expected_current_parser_script_sha256": current,
        "observed_parser_lineage": [
            REVIEWED_PRIMARY_EQUIVALENT_PARSER_SHA256,
            current,
        ],
        "current_singleton": False,
    }
    manifest["source_binding_projection"].update(
        {
            "mode": "mixed_lineage_projection",
            "parser_lineage": [
                REVIEWED_PRIMARY_EQUIVALENT_PARSER_SHA256,
                current,
            ],
            "selection_policy": (
                "stored-binding-semantics-current-first-v1"
            ),
            "selection_counts": {},
        }
    )
    manifest["source_binding_projection"]["coverage"][
        "source_input_count"
    ] = 0
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    allowed, _metadata = _load_ci_manifest_allowlist(
        manifest_path,
        ci_root,
        builder.DEFAULT_BUCKETS,
        cppmega_mlx_commit="unused",
        cppmega_mlx_tree_sha256="unused",
    )
    assert set(allowed) == {
        ("ci", bucket) for bucket in builder.DEFAULT_BUCKETS
    }

    upgrade["reason"] = "unreviewed parser transition"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="parser generation is not reviewed"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )


def test_content_store_export_accepts_exact_reviewed_frozen_parser_transition(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_content_store_ci_export(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = builder.target_parser_script_sha256()
    upgrade = {
        "binding_key": "parser_script_sha256",
        "from_sha256": REVIEWED_FROZEN_PARSER_FROM_SHA256,
        "to_sha256": REVIEWED_FROZEN_PARSER_SINK_SHA256,
        "reason": REVIEWED_FROZEN_PARSER_UPGRADE_REASON,
        "upgraded_at": "2026-07-31T13:01:16Z",
    }
    manifest["input_fetch_state"]["settings"]["parser_script_sha256"] = (
        REVIEWED_FROZEN_PARSER_SINK_SHA256
    )
    manifest["input_fetch_state"]["summary"] = {
        "binding_upgrades": [upgrade]
    }
    manifest["parser_generation_policy"] = {
        "mode": "reviewed-frozen-parser-transition",
        "expected_current_parser_script_sha256": current,
        "observed_parser_lineage": list(REVIEWED_FROZEN_PARSER_LINEAGE),
        "current_singleton": False,
        "authorized_projection_from_parser_script_sha256": (
            REVIEWED_FROZEN_PARSER_FROM_SHA256
        ),
    }
    manifest["source_binding_projection"].update(
        {
            "mode": "mixed_lineage_projection",
            "parser_lineage": list(REVIEWED_FROZEN_PARSER_LINEAGE),
            "selection_policy": "stored-binding-semantics-current-first-v1",
            "selection_counts": {},
        }
    )
    manifest["source_binding_projection"]["coverage"][
        "source_input_count"
    ] = 0
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    allowed, _metadata = _load_ci_manifest_allowlist(
        manifest_path,
        ci_root,
        builder.DEFAULT_BUCKETS,
        cppmega_mlx_commit="unused",
        cppmega_mlx_tree_sha256="unused",
    )
    assert set(allowed) == {
        ("ci", bucket) for bucket in builder.DEFAULT_BUCKETS
    }

    manifest["parser_generation_policy"][
        "authorized_projection_from_parser_script_sha256"
    ] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="parser generation is not reviewed"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )


def test_content_store_export_rejects_legacy_fetch_state_schema(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_content_store_ci_export(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_fetch_state"]["schema"] = "cppmega_ci_stream_fetch_v3"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="immutable producer binding is incomplete",
    ):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )


def test_production_content_store_export_rejects_nonexistent_artifact_chain(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci-production"
    manifest_path = _write_content_store_ci_export(
        ci_root,
        schema=builder.PRODUCTION_CI_CONTENT_STORE_EXPORT_SCHEMA,
    )

    with pytest.raises(
        RuntimeError,
        match="inventory database is missing or unsafe",
    ):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused-for-content-store-export",
            cppmega_mlx_tree_sha256="unused-for-content-store-export",
        )


def test_production_export_schema_label_cannot_masquerade_as_exhaustive(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci-masquerade"
    manifest_path = _write_content_store_ci_export(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = (
        builder.PRODUCTION_CI_CONTENT_STORE_EXPORT_SCHEMA
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="lacks inventory-exhaustive semantics",
    ):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )


def test_content_store_export_allowlist_rejects_drift_and_orphans(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_content_store_ci_export(ci_root)
    first = (
        ci_root
        / "1024"
        / "ci-case5-train-1024-000000.parquet"
    )
    first.write_bytes(b"drifted")

    with pytest.raises(RuntimeError, match="artifact binding drifted"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )

    manifest_path = _write_content_store_ci_export(ci_root)
    (ci_root / "1024" / "orphan.parquet").write_bytes(b"orphan")
    with pytest.raises(RuntimeError, match="inventory differs"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )


def test_content_store_export_requires_physical_zstd_training_parquet(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_content_store_ci_export(ci_root)
    parquet_path = (
        ci_root
        / "1024"
        / "ci-case5-train-1024-000000.parquet"
    )
    table = pq.read_table(parquet_path)
    pq.write_table(table, parquet_path, compression=None)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item["path"] == "1024/ci-case5-train-1024-000000.parquet"
    )
    artifact["byte_size"] = parquet_path.stat().st_size
    artifact["sha256"] = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="CI training Parquet is not receipt-bound ZSTD",
    ):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )


def test_content_store_export_requires_zstd_receipt_contract(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_content_store_ci_export(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["case5_contract"]["parquet_compression"] = {
        "codec": "snappy",
        "level": None,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="unsupported CI export CASE5 bucket contract",
    ):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )


def test_content_store_export_rejects_symlinked_bucket_parent(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_content_store_ci_export(ci_root)
    outside_bucket = tmp_path / "outside-1024"
    (ci_root / "1024").rename(outside_bucket)
    (ci_root / "1024").symlink_to(outside_bucket, target_is_directory=True)

    with pytest.raises(RuntimeError, match="contains a symlink"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )


def test_content_store_export_rejects_unlisted_symlink_directory(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_content_store_ci_export(ci_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "not-in-export.parquet").write_bytes(b"outside")
    (ci_root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="contains a symlink"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit="unused",
            cppmega_mlx_tree_sha256="unused",
        )


def test_ci_manifest_allowlist_rejects_hash_drift_rejects_and_orphans(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_ci_generation(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["counters"]["normalization_rejects"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="reject counters"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit=_CI_PRODUCER_COMMIT,
            cppmega_mlx_tree_sha256=_CI_PRODUCER_TREE_SHA256,
        )

    manifest_path = _write_ci_generation(ci_root)
    (ci_root / "1024" / "ci_packed_1024.parquet").write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match="artifact binding drifted"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit=_CI_PRODUCER_COMMIT,
            cppmega_mlx_tree_sha256=_CI_PRODUCER_TREE_SHA256,
        )

    manifest_path = _write_ci_generation(ci_root)
    (ci_root / "1024" / "orphan.parquet").write_bytes(b"orphan")
    with pytest.raises(RuntimeError, match="inventory differs"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit=_CI_PRODUCER_COMMIT,
            cppmega_mlx_tree_sha256=_CI_PRODUCER_TREE_SHA256,
        )


def test_ci_manifest_rejects_incomplete_or_unbound_log_extraction(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"
    manifest_path = _write_ci_generation(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_completion"]["unresolved_count"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="completion is missing or incomplete"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit=_CI_PRODUCER_COMMIT,
            cppmega_mlx_tree_sha256=_CI_PRODUCER_TREE_SHA256,
        )

    manifest_path = _write_ci_generation(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_completion"]["output"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact binding drifted"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit=_CI_PRODUCER_COMMIT,
            cppmega_mlx_tree_sha256=_CI_PRODUCER_TREE_SHA256,
        )


def test_ci_manifest_rejects_stale_or_forged_mlx_revision_and_inventory(
    tmp_path: Path,
) -> None:
    ci_root = tmp_path / "ci"

    manifest_path = _write_ci_generation(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["producer"]["code_revision"]["git_commit"] = "f" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="producer commit"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit=_CI_PRODUCER_COMMIT,
            cppmega_mlx_tree_sha256=_CI_PRODUCER_TREE_SHA256,
        )

    manifest_path = _write_ci_generation(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["producer"]["code_revision"]["source_tree_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="producer source tree"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit=_CI_PRODUCER_COMMIT,
            cppmega_mlx_tree_sha256=_CI_PRODUCER_TREE_SHA256,
        )

    manifest_path = _write_ci_generation(ci_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_inventory"][1]["name"] = "ambient_fallback.jsonl"
    manifest["source_inventory_sha256"] = _canonical_sha256(
        manifest["source_inventory"]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical ordered pair"):
        _load_ci_manifest_allowlist(
            manifest_path,
            ci_root,
            builder.DEFAULT_BUCKETS,
            cppmega_mlx_commit=_CI_PRODUCER_COMMIT,
            cppmega_mlx_tree_sha256=_CI_PRODUCER_TREE_SHA256,
        )


def test_bundle_known_limitations_do_not_claim_retired_qname_or_domain_gaps() -> None:
    assert BUNDLE_KNOWN_LIMITATIONS == ()
    text = " ".join(BUNDLE_KNOWN_LIMITATIONS).lower()
    assert "qname" not in text
    assert "no observed shell" not in text


def _builder_revision_binding() -> dict[str, object]:
    return {
        "schema_version": 2,
        "git_commit": "a" * 40,
        "dirty": False,
        "source_tree_sha256": "b" * 64,
        "producer_role": "canonical_source_conveyor",
        "repository_identity": "cppmega",
        "indexer_dependency_closure_sha256": "d" * 64,
        "indexer_provenance": {
            "schema": "cppmega_indexer_dependency_binding_v1",
            "path": "tools/clang_indexer/index_project.py",
            "source_sha256": "c" * 64,
            "dependency_closure_sha256": "d" * 64,
            "dependency_manifest": {
                "tools/clang_indexer/index_project.py": "c" * 64,
            },
        },
    }


def test_bundle_producer_binding_covers_cppmega_mlx_and_indexer_closure() -> None:
    binding = _producer_binding_from_local_revision(
        _builder_revision_binding(),
        cppmega_commit="a" * 40,
        cppmega_tree_sha256="b" * 64,
        cppmega_mlx_commit="e" * 40,
        cppmega_mlx_tree_sha256="f" * 64,
    )

    assert set(binding["components"]) == {
        "cppmega",
        "cppmega_mlx",
        "clang_indexer",
    }
    assert binding["components"]["clang_indexer"][
        "dependency_closure_sha256"
    ] == "d" * 64
    assert binding["components"]["cppmega"] == {
        "commit": "a" * 40,
        "tree_sha256": "b" * 64,
    }
    assert binding["components"]["cppmega_mlx"] == {
        "commit": "e" * 40,
        "tree_sha256": "f" * 64,
    }


def test_bundle_builder_verifies_its_live_clean_cppmega_revision(
    tmp_path: Path,
) -> None:
    from scripts import streaming_conveyor

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    indexer_root = repo / "tools" / "clang_indexer"
    indexer_root.mkdir(parents=True)
    (indexer_root / "index_project.py").write_text(
        "INDEXER_VALUE = 1\n",
        encoding="utf-8",
    )
    for command in (
        ("init", "-q"),
        ("config", "user.name", "Bundle Test"),
        ("config", "user.email", "bundle@example.test"),
        ("add", "."),
        ("commit", "-q", "-m", "initial"),
    ):
        builder.subprocess.run(
            ["git", "-C", str(repo), *command],
            check=True,
            capture_output=True,
            text=True,
        )
    revision = streaming_conveyor.capture_code_revision(repo)

    verified = _verify_local_cppmega_revision(
        expected_commit=revision["git_commit"],
        expected_tree_sha256=revision["source_tree_sha256"],
        repo_root=repo,
    )
    assert verified["git_commit"] == revision["git_commit"]

    with pytest.raises(RuntimeError, match="--cppmega-commit"):
        _verify_local_cppmega_revision(
            expected_commit="f" * 40,
            expected_tree_sha256=revision["source_tree_sha256"],
            repo_root=repo,
        )

    (repo / "scripts" / "worker.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkout is dirty"):
        _verify_local_cppmega_revision(
            expected_commit=revision["git_commit"],
            expected_tree_sha256=revision["source_tree_sha256"],
            repo_root=repo,
        )


def test_bundle_producer_binding_rejects_legacy_revision_receipt() -> None:
    with pytest.raises(RuntimeError, match="schema v2"):
        _producer_binding_from_local_revision(
            {"schema_version": 1, "dirty": False},
            cppmega_commit="e" * 40,
            cppmega_tree_sha256="f" * 64,
            cppmega_mlx_commit="a" * 40,
            cppmega_mlx_tree_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("producer_role", "unknown", "producer role"),
        ("repository_identity", "cppmega_mlx", "not bound to cppmega"),
        ("git_commit", "9" * 40, "cppmega commit"),
        ("source_tree_sha256", "8" * 64, "cppmega source tree"),
    ],
)
def test_bundle_producer_binding_rejects_wrong_repository_provenance(
    field: str, value: str, error: str
) -> None:
    revision = _builder_revision_binding()
    revision[field] = value

    with pytest.raises(RuntimeError, match=error):
        _producer_binding_from_local_revision(
            revision,
            cppmega_commit="a" * 40,
            cppmega_tree_sha256="b" * 64,
            cppmega_mlx_commit="e" * 40,
            cppmega_mlx_tree_sha256="f" * 64,
        )


def test_artifact_set_fingerprint_is_order_independent_and_content_bound() -> None:
    first = {"path": "b.bin", "size": 2, "sha256": "bb"}
    second = {"path": "a.bin", "size": 1, "sha256": "aa"}

    digest = _artifact_set_sha256([first, second])

    assert digest == _artifact_set_sha256([second, first])
    assert digest != _artifact_set_sha256(
        [first, {"path": "a.bin", "size": 1, "sha256": "changed"}]
    )


def test_builder_discards_intermediate_snapshot_by_default() -> None:
    parser = build_arg_parser()

    assert parser.parse_args([]).keep_snapshot is False
    assert parser.parse_args(["--keep-snapshot"]).keep_snapshot is True


def test_builder_accepts_explicit_bucketed_objective_artifacts() -> None:
    args = build_arg_parser().parse_args(
        ["--objective-artifact", "1024=/checked-out/objective_materialization.json"]
    )

    assert args.objective_artifact == [
        "1024=/checked-out/objective_materialization.json"
    ]


def test_builder_requires_exactly_one_objective_artifact_per_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "objective_materialization.json"
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        builder, "load_objective_materialization_artifact", lambda _path: object()
    )

    assert _parse_objective_artifacts(
        [f"1024={artifact}", f"2048={artifact}"], (1024, 2048)
    ) == {1024: artifact.resolve(), 2048: artifact.resolve()}
    with pytest.raises(ValueError, match="exactly match"):
        _parse_objective_artifacts([f"1024={artifact}"], (1024, 2048))


def test_production_objective_target_is_exact_hash_bound_and_enforced() -> None:
    target = builder._load_production_objective_target()
    sample_targets = {
        "1024": 281580,
        "2048": 167040,
        "4096": 95400,
        "8192": 45540,
        "16384": 19740,
    }
    policy = target["materialization"]

    assert target["sample_targets"] == sample_targets
    assert policy == {
        "seed": 17,
        "quota_window_samples": 60,
        "quota_lookahead_samples": 180,
        "objective_seed_kinds": ["code", "commits"],
        "graph_relations": [
            "call",
            "type",
            "domain",
            "build",
            "shell",
            "diagnostic",
            "cross_domain",
        ],
    }
    assert target["megatron_split"] == {
        "value": "969,30,1",
        "fractions": ["0.969", "0.030", "0.001"],
        "rounding": "python_round_on_cumulative_document_counts",
    }
    assert builder._sha256(builder.PRODUCTION_OBJECTIVE_TARGET_PATH) == (
        builder.PRODUCTION_OBJECTIVE_TARGET_SHA256
    )
    contract = {
        "seed": 17,
        "quota_window_samples": 60,
        "totals": {"samples": 0},
        "source_selection": {"quota_lookahead_samples": 180},
        "graph_auxiliary": {"relations": policy["graph_relations"]},  # type: ignore[index]
        "source_snapshot": {"pools": {"objective_seed": {"files": []}}},
    }
    for bucket, samples in zip(
        builder.DEFAULT_BUCKETS, sample_targets.values(), strict=True
    ):
        contract["totals"]["samples"] = samples  # type: ignore[index]
        snapshot = contract["source_snapshot"]  # type: ignore[assignment]
        snapshot["sequence_length"] = bucket
        snapshot["pools"]["objective_seed"]["files"] = [  # type: ignore[index]
            {"path": f"code/{bucket}/code.parquet"},
            {"path": f"commits/{bucket}/commits.parquet"},
        ]
        builder._validate_production_objective_contract(
            contract=contract,
            bucket=bucket,
            target=target,
        )
    seed_files = contract["source_snapshot"]["pools"]["objective_seed"][  # type: ignore[index]
        "files"
    ]
    seed_files.append({"path": "pr/16384/pr.parquet"})
    with pytest.raises(RuntimeError, match="production objective|production target"):
        builder._validate_production_objective_contract(
            contract=contract,
            bucket=16384,
            target=target,
        )
    with pytest.raises(RuntimeError, match="exactly cover buckets"):
        builder._validate_production_objective_artifacts({}, (1024,))


def test_every_bucket_conversion_receives_hash_bound_objective_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    objective_artifact = tmp_path / "objective_materialization.json"
    objective_artifact.write_text("{}", encoding="utf-8")

    def fake_convert(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(builder, "convert_parquet_to_megatron", fake_convert)
    monkeypatch.setattr(
        builder,
        "_current_generation_directory",
        lambda _prefix: generation_dir,
    )
    monkeypatch.setattr(
        builder,
        "load_objective_materialization_artifact",
        lambda _path: SimpleNamespace(
            artifact_set_sha256="a" * 64,
            file_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        builder,
        "_objective_expected_counts",
        lambda _path: {"rows": 1, "valid_tokens": 3, "trained_tokens": 2},
    )
    monkeypatch.setattr(
        builder,
        "_verify_prefix",
        lambda _prefix, _expected: {
            "objective_contract": {
                "sha256": "a" * 64,
                "objective_id_sidecar": {
                    "path": "objective_ids.bin",
                    "dtype": "uint8",
                    "document_aligned": True,
                },
            },
            "objective_materialization": {
                "artifact_set_sha256": "a" * 64,
                "artifact_file_sha256": "b" * 64,
            },
        },
    )

    builder._build_bucket(
        bucket=1024,
        data_root=tmp_path / "data",
        objective_artifact_path=objective_artifact,
    )

    assert captured["objective_artifact_path"] == str(objective_artifact.resolve())
    assert captured["input_dir"] is None
    assert captured["token_column"] == "input_ids"


def test_build_bucket_seals_pointer_generation_as_regular_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objective_artifact = tmp_path / "objective_materialization.json"
    objective_artifact.write_text("{}", encoding="utf-8")
    verified: list[Path] = []

    monkeypatch.setattr(
        builder,
        "load_objective_materialization_artifact",
        lambda _path: SimpleNamespace(
            artifact_set_sha256="a" * 64,
            file_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        builder,
        "_objective_expected_counts",
        lambda _path: {"rows": 1, "valid_tokens": 3, "trained_tokens": 2},
    )

    def fake_convert(**kwargs: object) -> None:
        prefix = Path(str(kwargs["output_prefix"]))
        generation = (
            prefix.parent
            / "snapshot"
            / "megatron_generations"
            / prefix.name
            / "generation-test"
        )
        generation.mkdir(parents=True)
        for suffix in (".bin", ".idx", ".json"):
            (generation / f"{prefix.name}{suffix}").write_bytes(suffix.encode())
        current = prefix.parent / f".{prefix.name}.current"
        current.symlink_to(os.path.relpath(generation, prefix.parent))
        for suffix in (".bin", ".idx", ".json"):
            alias = prefix.with_suffix(suffix)
            alias.symlink_to(f"{current.name}/{alias.name}")

    manifest = {
        "objective_materialization": {
            "artifact_set_sha256": "a" * 64,
            "artifact_file_sha256": "b" * 64,
        }
    }

    def fake_verify(prefix: Path, _expected: dict[str, int]) -> dict[str, object]:
        verified.append(prefix)
        return manifest

    monkeypatch.setattr(builder, "convert_parquet_to_megatron", fake_convert)
    monkeypatch.setattr(builder, "_verify_prefix", fake_verify)

    result = builder._build_bucket(
        bucket=1024,
        data_root=tmp_path / "data",
        objective_artifact_path=objective_artifact,
    )

    final_dir = tmp_path / "data" / "seq_1024"
    final_prefix = final_dir / "cppmega_macro_routes_seq1024_train"
    assert result["prefix"] == str(final_prefix)
    assert verified[-1] == final_prefix
    assert not (tmp_path / "data" / ".seq_1024.building").exists()
    assert all(path.is_file() and not path.is_symlink() for path in final_dir.iterdir())


def test_objective_sources_must_match_repaired_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = b"code"
    second = b"commit"
    files = [
        {
            "path": "code/1024/repo.parquet",
            "size_bytes": len(first),
            "sha256": hashlib.sha256(first).hexdigest(),
            "rows": 2,
        },
        {
            "path": "commits/1024/repo_r0.parquet",
            "size_bytes": len(second),
            "sha256": hashlib.sha256(second).hexdigest(),
            "rows": 3,
        },
    ]
    digest = _artifact_set_sha256(
        [
            {"path": row["path"], "size": row["size_bytes"], "sha256": row["sha256"]}
            for row in files
        ]
    )
    source_snapshot = {
        "schema": "cppmega_objective_source_snapshot_v1",
        "sequence_length": 1024,
        "file_count": 2,
        "row_count": 5,
        "files": files,
        "sampling": {
            "mode": "deterministic_epoch_shuffle_v1",
            "seed": 17,
            "requested_samples": 60,
            "full_passes": 12,
            "tail_rows": 0,
            "min_row_reuse": 12,
            "max_row_reuse": 12,
        },
        "artifact_set_sha256": digest,
    }
    monkeypatch.setattr(
        builder,
        "load_objective_materialization_artifact",
        lambda _path: SimpleNamespace(
            contract=SimpleNamespace(payload={"source_snapshot": source_snapshot})
        ),
    )
    repaired = {
        "files": [
            {
                "kind": Path(row["path"]).parts[0],
                "bucket": 1024,
                "snapshot": row["path"],
                "size": row["size_bytes"],
                "snapshot_sha256": row["sha256"],
                "rows": row["rows"],
            }
            for row in files
        ]
    }

    result = _validate_objective_source_binding(
        objective_artifact_path=tmp_path / "objective.json",
        repaired_snapshot_manifest=repaired,
        bucket=1024,
    )
    assert result["artifact_set_sha256"] == digest

    for field, value in (
        ("kind", "commits"),
        ("bucket", 2048),
        ("snapshot", "code/1024/other.parquet"),
        ("size", 999),
        ("snapshot_sha256", "0" * 64),
        ("rows", 999),
    ):
        drifted = json.loads(json.dumps(repaired))
        drifted["files"][0][field] = value
        with pytest.raises(RuntimeError, match="do not match repaired snapshot"):
            _validate_objective_source_binding(
                objective_artifact_path=tmp_path / "objective.json",
                repaired_snapshot_manifest=drifted,
                bucket=1024,
            )

    reversed_snapshot = {"files": list(reversed(repaired["files"]))}
    with pytest.raises(RuntimeError, match="do not match repaired snapshot"):
        _validate_objective_source_binding(
            objective_artifact_path=tmp_path / "objective.json",
            repaired_snapshot_manifest=reversed_snapshot,
            bucket=1024,
        )


def _bounded_v2_sampling() -> dict[str, object]:
    return {
        "mode": "deterministic_shard_row_group_record_batch_shuffle_v2",
        "seed": 31,
        "requested_samples": 5,
        "full_passes": 2,
        "tail_rows": 1,
        "min_row_reuse": 2,
        "max_row_reuse": 3,
        "record_batch_rows": 2,
        "ordering": {
            "permutation": "sha256_sort_key_v1",
            "epochs": "ascending",
            "shards": "seeded_permutation_per_epoch",
            "row_groups": "seeded_permutation_per_shard_epoch",
            "record_batches": "physical_order_within_row_group",
            "rows": "seeded_permutation_within_record_batch",
        },
        "cursor_semantics": "last_yielded_row_v1",
        "producer": {
            "name": "pyarrow.parquet.ParquetFile.iter_batches",
            "version": 1,
            "row_group_rows": [[2]],
        },
        "final_cursor": {
            "epoch": 2,
            "shard_position": 0,
            "shard_index": 0,
            "row_group_position": 0,
            "row_group_index": 0,
            "record_batch_index": 0,
            "row_shuffle_position": 0,
            "row_index_in_record_batch": 0,
            "source_index": 4,
        },
    }


def _bounded_v2_source_binding() -> tuple[dict[str, object], dict[str, object]]:
    payload = b"code"
    files = [
        {
            "path": "code/1024/repo.parquet",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": 2,
        }
    ]
    digest = _artifact_set_sha256(
        [
            {
                "path": files[0]["path"],
                "size": files[0]["size_bytes"],
                "sha256": files[0]["sha256"],
            }
        ]
    )
    source_snapshot = {
        "schema": "cppmega_objective_source_snapshot_v1",
        "sequence_length": 1024,
        "file_count": 1,
        "row_count": 2,
        "files": files,
        "sampling": _bounded_v2_sampling(),
        "artifact_set_sha256": digest,
    }
    repaired_snapshot = {
        "files": [
            {
                "kind": "code",
                "bucket": 1024,
                "snapshot": files[0]["path"],
                "size": files[0]["size_bytes"],
                "snapshot_sha256": files[0]["sha256"],
                "rows": files[0]["rows"],
            }
        ]
    }
    return source_snapshot, repaired_snapshot


def _two_pool_source_binding_v2(
    seed_kind: str,
) -> tuple[dict[str, object], dict[str, object]]:
    pool_specs = (
        (
            "primary_ci",
            "1024/ci-case5-train-1024-000000.parquet",
            b"ci",
        ),
        ("objective_seed", f"{seed_kind}/1024/seed.parquet", b"seed"),
    )
    pools: dict[str, dict[str, object]] = {}
    repaired_files: list[dict[str, object]] = []
    for name, path, payload in pool_specs:
        record = {
            "path": path,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": 2,
        }
        digest = _artifact_set_sha256(
            [
                {
                    "path": path,
                    "size": record["size_bytes"],
                    "sha256": record["sha256"],
                }
            ]
        )
        pools[name] = {
            "schema": "cppmega_objective_source_snapshot_v1",
            "sequence_length": 1024,
            "file_count": 1,
            "row_count": 2,
            "files": [record],
            "sampling": {
                **_bounded_v2_sampling(),
                "requested_samples": 3,
                "full_passes": 1,
                "tail_rows": 1,
                "min_row_reuse": 1,
                "max_row_reuse": 2,
                "final_cursor": publisher._objective_sampling_v2_final_cursor(
                    seed=31,
                    requested_samples=3,
                    record_batch_rows=2,
                    row_group_rows=((2,),),
                ),
            },
            "artifact_set_sha256": digest,
        }
        canonical = f"ci/{path}" if name == "primary_ci" else path
        repaired_files.append(
            {
                "kind": "ci" if name == "primary_ci" else seed_kind,
                "bucket": 1024,
                "snapshot": canonical,
                "size": record["size_bytes"],
                "snapshot_sha256": record["sha256"],
                "rows": record["rows"],
            }
        )
    repaired_files.append(
        {
            "kind": "ci",
            "bucket": 1024,
            "snapshot": "ci/1024/ci-case5-validation-1024-000001.parquet",
            "size": 1,
            "snapshot_sha256": hashlib.sha256(b"validation").hexdigest(),
            "rows": 1,
        }
    )
    return (
        {
            "schema": "cppmega_objective_source_snapshot_v2",
            "sequence_length": 1024,
            "algorithm": "alternate_primary_seed_v1",
            "pool_order": ["primary_ci", "objective_seed"],
            "source_pool_manifest": {
                "path": "objective_source_pool_manifest.json",
                "size_bytes": 17,
                "sha256": "a" * 64,
            },
            "ci_export_receipt": {
                "path": "ci_export_receipt.json",
                "size_bytes": 19,
                "sha256": "b" * 64,
            },
            "pools": pools,
        },
        {"files": list(reversed(repaired_files))},
    )


def _write_two_pool_objective_artifact(
    tmp_path: Path, *, seed_kind: str
) -> tuple[Path, dict[str, object]]:
    source_snapshot, repaired_snapshot = _two_pool_source_binding_v2(seed_kind)
    pools = source_snapshot["pools"]
    primary_record = pools["primary_ci"]["files"][0]
    seed_record = pools["objective_seed"]["files"][0]

    ci_receipt = {
        "schema": "cppmega_ci_content_store_case5_export_v2",
        "status": "complete",
    }
    ci_receipt_path = tmp_path / "ci_export_receipt.json"
    ci_receipt_path.write_text(json.dumps(ci_receipt), encoding="utf-8")
    ci_receipt_sha256 = hashlib.sha256(ci_receipt_path.read_bytes()).hexdigest()
    source_completion = {
        "schema": ci_receipt["schema"],
        "status": "complete",
    }
    primary_by_bucket = {
        str(bucket): [
            (
                primary_record
                if bucket == 1024
                else {
                    "path": f"{bucket}/ci-case5-train-{bucket}-000000.parquet",
                    "rows": 1,
                    "size_bytes": 1,
                    "sha256": hashlib.sha256(
                        f"ci:{bucket}".encode("ascii")
                    ).hexdigest(),
                }
            )
        ]
        for bucket in (1024, 2048, 4096, 8192, 16384)
    }
    source_pool_manifest = {
        "schema": "cppmega_ci_objective_pool_manifest_v1",
        "algorithm": "alternate_primary_seed_v1",
        "sequence_lengths": [1024, 2048, 4096, 8192, 16384],
        "ci_export": {
            "path": "export_receipt.json",
            "sha256": ci_receipt_sha256,
            "schema": ci_receipt["schema"],
            "status": "complete",
            "source_completion": source_completion,
        },
        "primary_ci": {"files_by_sequence_length": primary_by_bucket},
        "objective_seed": {"files": [seed_record]},
        "producer": {
            "repository": "cppmega",
            "git_commit": "c" * 40,
            "script": "scripts/data/prepare_ci_objective_source_manifest.py",
            "script_sha256": "d" * 64,
        },
    }
    source_pool_manifest_path = tmp_path / "objective_source_pool_manifest.json"
    source_pool_manifest_path.write_text(
        json.dumps(source_pool_manifest), encoding="utf-8"
    )
    source_snapshot["source_pool_manifest"].update(
        {
            "size_bytes": source_pool_manifest_path.stat().st_size,
            "sha256": hashlib.sha256(
                source_pool_manifest_path.read_bytes()
            ).hexdigest(),
        }
    )
    source_snapshot["ci_export_receipt"].update(
        {
            "size_bytes": ci_receipt_path.stat().st_size,
            "sha256": ci_receipt_sha256,
        }
    )

    contract = _valid_contract()
    contract["source_snapshot"] = source_snapshot
    cursor = contract["source_selection"]["resume"]["last_yielded_cursor"]
    cursor.update(
        {
            "pool_index": 1,
            "pool_source_index": 2,
            "primary_rows_yielded": 3,
            "objective_seed_rows_yielded": 3,
            "next_pool_index": 0,
        }
    )
    return (
        _write_materialization_artifact(tmp_path, contract=contract),
        repaired_snapshot,
    )


def test_builder_accepts_exact_bounded_v2_sampling_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_snapshot, repaired_snapshot = _bounded_v2_source_binding()
    monkeypatch.setattr(
        builder,
        "load_objective_materialization_artifact",
        lambda _path: SimpleNamespace(
            contract=SimpleNamespace(payload={"source_snapshot": source_snapshot})
        ),
    )

    result = _validate_objective_source_binding(
        objective_artifact_path=tmp_path / "objective.json",
        repaired_snapshot_manifest=repaired_snapshot,
        bucket=1024,
    )

    assert result["sampling"] == _bounded_v2_sampling()


@pytest.mark.parametrize("seed_kind", ("code", "commits", "pr"))
def test_builder_accepts_two_pool_source_snapshot_v2(
    tmp_path: Path, seed_kind: str
) -> None:
    objective_artifact, repaired_snapshot = _write_two_pool_objective_artifact(
        tmp_path, seed_kind=seed_kind
    )

    result = _validate_objective_source_binding(
        objective_artifact_path=objective_artifact,
        repaired_snapshot_manifest=repaired_snapshot,
        bucket=1024,
    )

    assert result["schema"] == "cppmega_objective_source_snapshot_v2"
    assert set(result["pools"]) == {"primary_ci", "objective_seed"}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("snapshot_sha256", "0" * 64),
        ("size", 99),
        ("rows", 99),
        ("snapshot", "code/1024/other.parquet"),
    ),
)
def test_builder_rejects_two_pool_source_snapshot_v2_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    objective_artifact, repaired_snapshot = _write_two_pool_objective_artifact(
        tmp_path, seed_kind="code"
    )
    seed_record = next(
        record for record in repaired_snapshot["files"] if record["kind"] == "code"
    )
    seed_record[field] = value

    with pytest.raises(RuntimeError, match="do not match repaired snapshot"):
        _validate_objective_source_binding(
            objective_artifact_path=objective_artifact,
            repaired_snapshot_manifest=repaired_snapshot,
            bucket=1024,
        )


def test_builder_rejects_two_pool_ci_seed_kind(tmp_path: Path) -> None:
    objective_artifact, repaired_snapshot = _write_two_pool_objective_artifact(
        tmp_path, seed_kind="ci"
    )

    with pytest.raises(RuntimeError, match="source path is not canonical"):
        _validate_objective_source_binding(
            objective_artifact_path=objective_artifact,
            repaired_snapshot_manifest=repaired_snapshot,
            bucket=1024,
        )


def test_builder_rejects_unknown_two_pool_ci_split(tmp_path: Path) -> None:
    objective_artifact, repaired_snapshot = _write_two_pool_objective_artifact(
        tmp_path, seed_kind="code"
    )
    repaired_snapshot["files"].append(
        {
            "kind": "ci",
            "bucket": 1024,
            "snapshot": "ci/1024/ci-case5-dev-1024-000002.parquet",
            "size": 1,
            "snapshot_sha256": hashlib.sha256(b"dev").hexdigest(),
            "rows": 1,
        }
    )

    with pytest.raises(RuntimeError, match="CI snapshot path is not canonical"):
        _validate_objective_source_binding(
            objective_artifact_path=objective_artifact,
            repaired_snapshot_manifest=repaired_snapshot,
            bucket=1024,
        )


@pytest.mark.parametrize(
    "malformation",
    (
        "missing_record_batch_rows",
        "record_batch_size_alias",
        "ordering_drift",
        "missing_cursor_coordinate",
        "wrong_source_index",
    ),
)
def test_builder_rejects_malformed_bounded_v2_sampling_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    source_snapshot, repaired_snapshot = _bounded_v2_source_binding()
    sampling = source_snapshot["sampling"]
    assert isinstance(sampling, dict)
    if malformation == "missing_record_batch_rows":
        sampling.pop("record_batch_rows")
    elif malformation == "record_batch_size_alias":
        sampling["record_batch_size"] = sampling.pop("record_batch_rows")
    elif malformation == "ordering_drift":
        sampling["ordering"]["rows"] = "physical_order_within_record_batch"
    elif malformation == "missing_cursor_coordinate":
        sampling["final_cursor"].pop("row_index_in_record_batch")
    else:
        sampling["final_cursor"]["source_index"] = 3
    monkeypatch.setattr(
        builder,
        "load_objective_materialization_artifact",
        lambda _path: SimpleNamespace(
            contract=SimpleNamespace(payload={"source_snapshot": source_snapshot})
        ),
    )

    with pytest.raises(RuntimeError, match="objective source"):
        _validate_objective_source_binding(
            objective_artifact_path=tmp_path / "objective.json",
            repaired_snapshot_manifest=repaired_snapshot,
            bucket=1024,
        )


def test_builder_stages_and_hashes_the_production_tokenizer(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "data/tokenizer_v2"
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    descriptor = _stage_tokenizer(source, bundle)
    resumed = _stage_tokenizer(source, bundle)

    assert resumed == descriptor
    assert descriptor["path"] == "tokenizer"
    assert descriptor["vocab_size"] == 65536
    assert {record["path"] for record in descriptor["files"]} == {
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_contract_v1.json",
        "tokenizer/tokenizer_config.json",
    }
    for record in descriptor["files"]:
        staged = bundle / record["path"]
        assert staged.stat().st_size == record["size"]
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == record["sha256"]


def test_builder_stages_all_frozen_data_contracts(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    descriptors = _stage_data_contracts(bundle)
    resumed = _stage_data_contracts(bundle)

    assert resumed == descriptors
    assert set(descriptors) == {
        "domain_schema",
        "tokenizer_contract",
        "production_objective_target",
    }
    for descriptor in descriptors.values():
        staged = bundle / str(descriptor["path"])
        assert staged.stat().st_size == descriptor["size"]
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == descriptor["sha256"]
    assert descriptors["production_objective_target"]["sha256"] == (
        builder.PRODUCTION_OBJECTIVE_TARGET_SHA256
    )


def test_bucket_prefixes_are_bundle_relative_and_cannot_escape(tmp_path: Path) -> None:
    bundle = tmp_path / ".bundle.partial"
    prefix = bundle / "data/seq_1024/cppmega_train"

    results = _portable_bucket_results(
        bundle,
        [{"bucket": 1024, "prefix": str(prefix), "manifest": {}}],
    )

    assert results[0]["prefix"] == "data/seq_1024/cppmega_train"
    with pytest.raises(RuntimeError, match="escapes bundle root"):
        _portable_bucket_results(
            bundle,
            [{"bucket": 1024, "prefix": str(tmp_path / "outside"), "manifest": {}}],
        )


def test_builder_rejects_unbound_existing_audit_receipt(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    (audit_root / "sidecar_parquet_audit.json").write_text(
        json.dumps({"total": {"bad_files": 0, "bad_rows": 0}, "bad_files": []}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not bound"):
        _run_snapshot_audit(
            snapshot_root=tmp_path / "snapshot",
            audit_script=tmp_path / "audit.py",
            audit_root=audit_root,
            buckets=(1024,),
            workers=1,
            snapshot_manifest_sha256="abc",
        )


def test_snapshot_audit_passes_explicit_empty_pr_root(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, list[str]] = {}
    audit_root = tmp_path / "audit"

    def fake_run(cmd: list[str], *, check: bool) -> None:
        assert check is True
        captured["cmd"] = cmd
        audit_root.mkdir(parents=True, exist_ok=True)
        (audit_root / "sidecar_parquet_audit.json").write_text(
            json.dumps(
                {
                    "total": {"bad_files": 0, "bad_rows": 0},
                    "bad_files": [],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    _run_snapshot_audit(
        snapshot_root=tmp_path / "snapshot",
        audit_script=tmp_path / "audit.py",
        audit_root=audit_root,
        buckets=(1024,),
        workers=1,
        snapshot_manifest_sha256="abc",
    )

    cmd = captured["cmd"]
    pr_root = Path(cmd[cmd.index("--pr-root") + 1])
    assert pr_root == audit_root / "empty_standalone_pr_root"
    assert pr_root.is_dir()
    assert "outputs/reindexed_pr" not in " ".join(cmd)


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_source_composition_allowlist_excludes_uncommitted_parquet_orphans(
    tmp_path: Path,
) -> None:
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    _write(code_root / "1024" / "repo.parquet", b"code")
    _write(code_root / "1024" / "orphan.parquet", b"orphan")
    _write(commit_root / "1024" / "repo_r0.parquet", b"commit")
    _write(commit_root / "1024" / "orphan_r0.parquet", b"orphan")
    allowed = {
        ("code", 1024): {"repo.parquet": 1},
        ("commits", 1024): {"repo_r0.parquet": 1},
    }
    source_composition = {
        "schema": "cppmega_source_conveyor_composition_v1",
        "status": "complete",
    }
    snapshot = tmp_path / "snapshot"
    receipt = _snapshot_sources(
        code_root=code_root,
        commit_root=commit_root,
        snapshot_root=snapshot,
        buckets=(1024,),
        min_age_seconds=0,
        hash_jobs=1,
        allowed=allowed,
        source_composition=source_composition,
    )

    assert receipt["by_kind_bucket"] == {"code/1024": 1, "commits/1024": 1}
    assert sorted(path.name for path in (snapshot / "code/1024").glob("*.parquet")) == [
        "repo.parquet"
    ]
    assert sorted(
        path.name for path in (snapshot / "commits/1024").glob("*.parquet")
    ) == ["repo_r0.parquet"]
    assert receipt["files"][0]["rows"] == 1
    assert (
        (snapshot / "code/1024/repo.parquet").stat().st_ino
        != (code_root / "1024/repo.parquet").stat().st_ino
    )


def test_snapshot_rejects_routed_source_hash_drift(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    _write(code_root / "1024/repo.parquet", b"code")
    _write(commit_root / "1024/repo_r0.parquet", b"commit")

    with pytest.raises(RuntimeError, match="routed source SHA-256 drifted"):
        _snapshot_sources(
            code_root=code_root,
            commit_root=commit_root,
            snapshot_root=tmp_path / "snapshot",
            buckets=(1024,),
            min_age_seconds=0,
            hash_jobs=1,
            allowed={
                ("code", 1024): {"repo.parquet": 1},
                ("commits", 1024): {"repo_r0.parquet": 1},
            },
            expected_source_sha256={
                ("code", 1024, "repo.parquet"): "0" * 64,
                ("commits", 1024, "repo_r0.parquet"): hashlib.sha256(
                    b"commit"
                ).hexdigest(),
            },
            source_composition={
                "schema": "cppmega_source_conveyor_composition_v1"
            },
        )


def test_repaired_snapshot_manifest_hashes_and_binds_replaced_files(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    unchanged = snapshot / "code/1024/a.parquet"
    changed = snapshot / "commits/1024/b.parquet"
    _write(unchanged, b"same")
    _write(changed, b"before")
    source_manifest = {
        "files": [
            {
                "kind": "code",
                "bucket": 1024,
                "snapshot": "code/1024/a.parquet",
                "size": len(b"same"),
                "rows": 2,
                "sha256": hashlib.sha256(b"same").hexdigest(),
            },
            {
                "kind": "commits",
                "bucket": 1024,
                "snapshot": "commits/1024/b.parquet",
                "size": len(b"before"),
                "rows": 3,
                "sha256": hashlib.sha256(b"before").hexdigest(),
            },
        ]
    }
    replacement = changed.with_suffix(".new")
    replacement.write_bytes(b"after")
    os.replace(replacement, changed)
    repaired = _write_repaired_snapshot_manifest(
        snapshot_root=snapshot,
        source_manifest=source_manifest,
        repair_receipt={"file_scans": [{"path": str(changed), "rows": 3}]},
        hash_jobs=1,
    )

    by_path = {record["snapshot"]: record for record in repaired["files"]}
    assert by_path["code/1024/a.parquet"]["boundary_repaired"] is False
    assert (
        by_path["code/1024/a.parquet"]["snapshot_sha256"]
        == by_path["code/1024/a.parquet"]["source_sha256"]
    )
    assert by_path["commits/1024/b.parquet"]["boundary_repaired"] is True
    assert (
        by_path["commits/1024/b.parquet"]["snapshot_sha256"]
        != by_path["commits/1024/b.parquet"]["source_sha256"]
    )
    assert by_path["code/1024/a.parquet"]["rows"] == 2
    assert by_path["commits/1024/b.parquet"]["rows"] == 3


def test_snapshot_rejects_source_mutation_during_private_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    code_source = code_root / "1024/repo.parquet"
    _write(code_source, b"code-before")
    _write(commit_root / "1024/repo_r0.parquet", b"commit")
    real_copy = builder._copy_private

    def copy_then_mutate(source: Path, target: Path) -> None:
        real_copy(source, target)
        if source == code_source:
            source.write_bytes(b"code-after")

    monkeypatch.setattr(builder, "_copy_private", copy_then_mutate)
    snapshot = tmp_path / "snapshot"
    with pytest.raises(RuntimeError, match="source changed while snapshotting"):
        _snapshot_sources(
            code_root=code_root,
            commit_root=commit_root,
            snapshot_root=snapshot,
            buckets=(1024,),
            min_age_seconds=0,
            hash_jobs=1,
            allowed={
                ("code", 1024): {"repo.parquet": 2},
                ("commits", 1024): {"repo_r0.parquet": 3},
            },
            source_composition={"schema": "cppmega_source_conveyor_composition_v1"},
        )

    assert not (snapshot / "source_manifest.json").exists()
    assert not (snapshot / "code/1024/repo.parquet").exists()


def _test_build_plan(objective_digest: str) -> dict[str, object]:
    objectives = [
        {
            "bucket": 1024,
            "artifact_path": "/objective.json",
            "artifact_set_sha256": objective_digest,
            "artifact_file_sha256": "b" * 64,
            "contract_path": "/contract.json",
            "contract_sha256": "c" * 64,
            "contract_file_sha256": "d" * 64,
        }
    ]
    plan = {"buckets": [1024], "objective_artifacts": objectives}
    return {
        "schema": builder.BUILD_PLAN_SCHEMA,
        "objective_artifacts_sha256": _canonical_sha256(objectives),
        "build_plan_sha256": _canonical_sha256(plan),
        "plan": plan,
    }


def _test_snapshot_plan(source_digest: str) -> dict[str, object]:
    plan = {
        "buckets": [1024],
        "source_composition_sha256": source_digest,
    }
    return {
        "schema": builder.SNAPSHOT_PLAN_SCHEMA,
        "snapshot_plan_sha256": _canonical_sha256(plan),
        "plan": plan,
    }


def test_parser_exposes_explicit_objective_snapshot_preparation() -> None:
    args = build_arg_parser().parse_args(["--prepare-objective-snapshot"])

    assert args.prepare_objective_snapshot is True
    assert args.objective_artifact == []


def test_prepared_snapshot_plan_is_exactly_resume_bound(tmp_path: Path) -> None:
    partial = tmp_path / ".bundle.partial"
    original = _test_snapshot_plan("a" * 64)
    _ensure_partial_snapshot_plan(partial, original)
    _ensure_partial_snapshot_plan(partial, original)

    with pytest.raises(RuntimeError, match="stale partial snapshot plan mismatch"):
        _ensure_partial_snapshot_plan(
            partial,
            _test_snapshot_plan("b" * 64),
        )


def test_prepared_snapshot_can_adopt_one_exact_objective_build_plan(
    tmp_path: Path,
) -> None:
    partial = tmp_path / ".bundle.partial"
    snapshot_plan = _test_snapshot_plan("a" * 64)
    build_plan = _test_build_plan("b" * 64)
    _ensure_partial_snapshot_plan(partial, snapshot_plan)

    _ensure_partial_build_plan(partial, build_plan)
    _ensure_partial_build_plan(partial, build_plan)

    assert json.loads(
        (partial / "snapshot_plan.json").read_text(encoding="utf-8")
    ) == snapshot_plan
    assert json.loads(
        (partial / "build_plan.json").read_text(encoding="utf-8")
    ) == build_plan


def test_prepared_snapshot_rejects_symlinked_objective_build_plan(
    tmp_path: Path,
) -> None:
    partial = tmp_path / ".bundle.partial"
    _ensure_partial_snapshot_plan(partial, _test_snapshot_plan("a" * 64))
    (partial / "build_plan.json").symlink_to(tmp_path / "missing.json")

    with pytest.raises(RuntimeError, match="invalid canonical build plan"):
        _ensure_partial_build_plan(partial, _test_build_plan("b" * 64))


def test_stale_partial_cannot_reuse_different_objective_build_plan(
    tmp_path: Path,
) -> None:
    partial = tmp_path / ".bundle.partial"
    original = _test_build_plan("a" * 64)
    _ensure_partial_build_plan(partial, original)
    _ensure_partial_build_plan(partial, original)

    with pytest.raises(RuntimeError, match="stale partial build plan mismatch"):
        _ensure_partial_build_plan(partial, _test_build_plan("e" * 64))

    changed_plan = json.loads(json.dumps(original))
    changed_plan["plan"]["keep_snapshot"] = True
    changed_plan["build_plan_sha256"] = _canonical_sha256(changed_plan["plan"])
    assert (
        changed_plan["objective_artifacts_sha256"]
        == original["objective_artifacts_sha256"]
    )
    with pytest.raises(RuntimeError, match="stale partial build plan mismatch"):
        _ensure_partial_build_plan(partial, changed_plan)


def test_unbound_partial_is_rejected_as_stale(tmp_path: Path) -> None:
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()

    with pytest.raises(RuntimeError, match="no canonical build plan"):
        _ensure_partial_build_plan(partial, _test_build_plan("a" * 64))


def test_concurrent_build_lock_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    first = _acquire_build_lock(output)
    try:
        with pytest.raises(RuntimeError, match="bundle build already active"):
            _acquire_build_lock(output)
    finally:
        first.close()

    second = _acquire_build_lock(output)
    second.close()


def test_strict_validation_runs_before_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    partial = tmp_path / ".bundle.partial"
    output = tmp_path / "bundle"

    def validate(bundle: Path, hash_jobs: int) -> None:
        assert bundle == partial
        assert hash_jobs == 3
        events.append("validate")

    def publish(source: Path, target: Path) -> None:
        assert source == partial
        assert target == output
        events.append("publish")

    monkeypatch.setattr(builder, "_validate_bundle", validate)
    monkeypatch.setattr(builder.os, "replace", publish)

    _publish_validated_bundle(partial, output, hash_jobs=3)

    assert events == ["validate", "publish"]


def test_artifact_records_include_nested_source_run_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
    run_manifest = (
        bundle
        / "provenance"
        / "source_composition"
        / "runs"
        / "full"
        / "manifest.json"
    )
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text('{"schema":"fixture"}\n', encoding="utf-8")

    records = builder._artifact_records(bundle, hash_jobs=1)
    paths = {record["path"] for record in records}

    assert "manifest.json" not in paths
    assert "provenance/source_composition/runs/full/manifest.json" in paths


def test_stage_source_composition_binds_salvage_and_pr_artifacts(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"

    def write(name: str, payload: bytes) -> Path:
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def binding(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    plan_path = write("plan.json", b'{"schema":"fixture"}\n')
    dedup_path = write("dedup.json", b'{"schema":"fixture"}\n')
    input_artifacts = {
        name: write(f"inputs/{name}.json", name.encode("ascii"))
        for name in (
            "archive_sha256_receipt",
            "archive_inventory_receipt",
            "repo_list",
            "source_quarantine_manifest",
            "tokenizer",
        )
    }
    artifacts = {
        "launch": write("run/launch.json", b"launch"),
        "exit": write("run/exit.json", b"salvaged-exit"),
        "manifest": write("run/manifest.json", b"manifest"),
        "archive_sha256_receipt": input_artifacts["archive_sha256_receipt"],
        "archive_inventory": input_artifacts["archive_inventory_receipt"],
        "repo_list": input_artifacts["repo_list"],
        "source_quarantine_manifest": input_artifacts[
            "source_quarantine_manifest"
        ],
        "tokenizer": input_artifacts["tokenizer"],
        "original_exit": write("run/original_exit.json", b"original-exit"),
        "pr_completion": write("run/pr_completion.json", b"pr-completion"),
        "pr_repo_list": write("run/pr_repo_list.json", b"pr-repos"),
    }
    verifier = builder.REPO_ROOT / "scripts/data/verify_global_dedup_store.py"
    assert verifier.is_file()
    run_receipt = {
        "run_id": "repair",
        "launch": {"sha256": binding(artifacts["launch"])},
        "exit": {
            "sha256": binding(artifacts["exit"]),
            "salvage": {
                "original_exit_receipt_sha256": binding(artifacts["original_exit"]),
                "original_exit_receipt_size_bytes": artifacts["original_exit"].stat().st_size,
            },
        },
        "manifest": {"sha256": binding(artifacts["manifest"])},
        "input_artifacts": {
            name: binding(path) for name, path in input_artifacts.items()
        },
        "pr_completion": {
            "receipt_sha256": binding(artifacts["pr_completion"]),
            "repo_list_sha256": binding(artifacts["pr_repo_list"]),
        },
    }
    composition = SourceComposition(
        allowlist={},
        receipt={
            "schema": "cppmega_source_conveyor_composition_v1",
            "plan_sha256": binding(plan_path),
            "dedup": {
                "receipt_sha256": binding(dedup_path),
                "verifier": {
                    "script": "scripts/data/verify_global_dedup_store.py",
                    "script_sha256": binding(verifier),
                },
            },
            "runs": [run_receipt],
        },
        plan_path=plan_path,
        dedup_receipt_path=dedup_path,
        run_files=(artifacts,),
    )

    staged = builder._stage_source_composition(
        composition,
        partial_dir=tmp_path / "partial",
        provenance_root=tmp_path / "partial" / "provenance",
    )

    staged_artifacts = staged["runs"][0]["artifacts"]
    assert set(staged_artifacts) == set(artifacts)
    assert staged_artifacts["original_exit"]["sha256"] == binding(
        artifacts["original_exit"]
    )
    assert staged_artifacts["pr_completion"]["sha256"] == binding(
        artifacts["pr_completion"]
    )

    original_exit = artifacts.pop("original_exit")
    missing_partial = tmp_path / "missing.partial"
    with pytest.raises(RuntimeError, match="artifact set drifted"):
        builder._stage_source_composition(
            composition,
            partial_dir=missing_partial,
            provenance_root=missing_partial / "provenance",
        )
    artifacts["original_exit"] = original_exit

    salvage = run_receipt["exit"]["salvage"]
    original_size = salvage["original_exit_receipt_size_bytes"]
    salvage["original_exit_receipt_size_bytes"] = original_size + 1
    wrong_size_partial = tmp_path / "wrong-size.partial"
    with pytest.raises(RuntimeError, match="original exit size drifted"):
        builder._stage_source_composition(
            composition,
            partial_dir=wrong_size_partial,
            provenance_root=wrong_size_partial / "provenance",
        )
    salvage["original_exit_receipt_size_bytes"] = original_size
