from __future__ import annotations

import hashlib
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import tarfile
import threading
from types import SimpleNamespace

import pytest

from cppmega.megatron.graph_recipe import (
    STAGE1_GRAPH_RELATIONS,
    STAGE1_GRAPH_TOPK,
    stage1_graph_recipe_binding,
)
from cppmega.receipt_binding import build_data_producer_binding
import scripts.data.publish_megatron_bundle_to_nebius_s3 as publisher
from scripts.data.publish_megatron_bundle_to_nebius_s3 import (
    _head,
    _head_matches,
    _stable_upload_snapshot,
    _upload_file,
    _validate_archive,
    _validate_archive_member_names,
    _validate_bundle,
    _validate_tokenizer_directory,
    main,
)


def _bundle(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    artifact = prefix.with_suffix(".bin")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return artifact, digest


def _force_tiny_multipart(monkeypatch, *, single_put_max=5, part_size=4):
    monkeypatch.setattr(publisher, "S3_SINGLE_PUT_MAX_BYTES", single_put_max)
    monkeypatch.setattr(publisher, "S3_MIN_MULTIPART_PART_BYTES", part_size)
    monkeypatch.setattr(publisher, "MULTIPART_DEFAULT_PART_BYTES", part_size)
    monkeypatch.setattr(publisher, "MULTIPART_PART_ALIGNMENT_BYTES", 1)

    @contextmanager
    def stable_snapshot(local, **_kwargs):
        yield local

    monkeypatch.setattr(publisher, "_stable_upload_snapshot", stable_snapshot)


def _command_option(command, name):
    return command[command.index(name) + 1]


def _clear_s3_credential_env(monkeypatch):
    for name in (
        "NEBIUS_S3_ACCESS_KEY_ID",
        "NEBIUS_S3_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


def _archive(tmp_path, artifact):
    raw_archive = tmp_path / "bundle.tar"
    archive = tmp_path / "bundle.tar.zst"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    with tarfile.open(raw_archive, "w") as tar:
        for record in manifest["artifacts"]:
            tar.add(tmp_path / record["path"], arcname=record["path"])
        tar.add(tmp_path / "manifest.json", arcname="manifest.json")
    subprocess.run(
        ["zstd", "-q", "-1", str(raw_archive), "-o", str(archive)], check=True
    )
    return archive


def _write_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _source_identity(source: str) -> tuple[int, str]:
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    identity_id = int.from_bytes(digest[:8], "big", signed=False)
    if identity_id == 0:
        identity_id = int.from_bytes(digest[8:16], "big", signed=False)
    assert identity_id > 0
    return identity_id, digest.hex()


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


def _bounded_v2_source_snapshot() -> dict[str, object]:
    source_snapshot = _objective_payload()["source_snapshot"]
    source_snapshot["row_count"] = 2
    source_snapshot["files"][0]["rows"] = 2
    source_snapshot["sampling"] = _bounded_v2_sampling()
    return source_snapshot


def _malform_bounded_v2_sampling(
    sampling: dict[str, object], malformation: str
) -> None:
    if malformation == "missing_record_batch_rows":
        sampling.pop("record_batch_rows")
    elif malformation == "record_batch_size_alias":
        sampling["record_batch_size"] = sampling.pop("record_batch_rows")
    elif malformation == "invalid_record_batch_rows":
        sampling["record_batch_rows"] = True
    elif malformation == "ordering_drift":
        ordering = sampling["ordering"]
        assert isinstance(ordering, dict)
        ordering["rows"] = "physical_order_within_record_batch"
    elif malformation == "missing_cursor_coordinate":
        cursor = sampling["final_cursor"]
        assert isinstance(cursor, dict)
        cursor.pop("row_index_in_record_batch")
    elif malformation == "missing_producer":
        sampling.pop("producer")
    elif malformation == "invalid_producer_version":
        producer = sampling["producer"]
        assert isinstance(producer, dict)
        producer["version"] = 2
    elif malformation == "invalid_producer_name":
        producer = sampling["producer"]
        assert isinstance(producer, dict)
        producer["name"] = "unknown"
    elif malformation == "extra_producer_field":
        producer = sampling["producer"]
        assert isinstance(producer, dict)
        producer["unbound"] = True
    elif malformation == "producer_layout_mismatch":
        producer = sampling["producer"]
        assert isinstance(producer, dict)
        producer["row_group_rows"] = [[1]]
    elif malformation == "wrong_source_index":
        cursor = sampling["final_cursor"]
        assert isinstance(cursor, dict)
        cursor["source_index"] = 3
    else:
        cursor = sampling["final_cursor"]
        assert isinstance(cursor, dict)
        cursor["epoch"] = 1


def _objective_payload():
    tasks = ("causal_lm", "fim", "ast_fim", "ifim", "commit_diff", "pre_to_post")
    source_files = [
        {
            "path": "/frozen/code/1024/code.parquet",
            "size_bytes": 17,
            "sha256": "2" * 64,
            "rows": 6,
        }
    ]
    source_digest = publisher._artifact_set_sha256(
        [
            {
                "path": record["path"],
                "size": record["size_bytes"],
                "sha256": record["sha256"],
            }
            for record in source_files
        ]
    )
    return {
        "schema": "cppmega_pre_materialized_objectives_v1",
        "algorithm": "hamilton_eligibility_bipartite_v1",
        "seed": 17,
        "quota_window_samples": 6,
        "task_order": list(tasks),
        "objective_ids": {task: index + 1 for index, task in enumerate(tasks)},
        "configured_rates": {task: "1/6" for task in tasks},
        "planned_samples": {task: 1 for task in tasks},
        "realized": {
            task: {
                "samples": 1,
                "input_tokens": 3,
                "loss_tokens": 3 if task == "causal_lm" else 2,
            }
            for task in tasks
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
            "eligible_samples": 1,
            "positive_edges": 1,
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
            "loss_mask_alignment": "source_token_predicts_next_v1",
            "length_column": "valid_token_count",
            "objective_column": "objective_kind",
            "document_id_column": "doc_ids",
            "source_document_id_column": "token_source_doc_ids",
        },
        "source_snapshot": {
            "schema": "cppmega_objective_source_snapshot_v1",
            "sequence_length": 1024,
            "file_count": 1,
            "row_count": 6,
            "files": source_files,
            "sampling": {
                "mode": "deterministic_epoch_shuffle_v1",
                "seed": 17,
                "requested_samples": 6,
                "full_passes": 1,
                "tail_rows": 0,
                "min_row_reuse": 1,
                "max_row_reuse": 1,
            },
            "artifact_set_sha256": source_digest,
        },
    }


def _write_source_composition_provenance(root: Path) -> dict[str, object]:
    provenance = root / "provenance" / "source_composition"
    provenance.mkdir(parents=True, exist_ok=True)

    def write(name: str, payload: object) -> dict[str, object]:
        path = provenance / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    plan = write(
        "plan.json",
        {"schema": "cppmega_source_conveyor_composition_plan_v1"},
    )
    verifier = write("verify_global_dedup_store.py", b"# fixture verifier\n")
    dedup_payload = {
        "schema": "cppmega_global_dedup_store_receipt_v1",
        "status": "verified",
        "created_at": "2026-07-28T00:00:00Z",
        "database": {
            "path": "/fixture/dedup.sqlite",
            "size_bytes": 1,
            "sha256": "1" * 64,
        },
        "checkpoint": {
            "mode": "TRUNCATE",
            "busy": 0,
            "log_frames": 0,
            "checkpointed_frames": 0,
            "wal_size_bytes": 0,
        },
        "integrity_check": "ok",
        "sqlite_schema_sha256": "2" * 64,
        "logical_hash_algorithm": "cppmega_sqlite_rows_lenprefixed_v1",
        "logical_sha256": "3" * 64,
        "tables": {
            name: {
                "rows": (
                    0
                    if name.endswith("_stage") or name == "dedup_stages"
                    else 1
                ),
                "logical_sha256": hashlib.sha256(b"").hexdigest(),
            }
            for name in (
                "exact",
                "lsh",
                "minhash",
                "dedup_meta",
                "chunk_claims",
                "dedup_stages",
                "exact_stage",
                "minhash_stage",
                "lsh_stage",
                "chunk_claims_stage",
            )
        },
        "policy": {
            "exact": "sha1_token_ids_v1",
            "chunk": "tokenized_chunk_claims_v1",
            "near": {
                "enabled": True,
                "threshold": 0.7,
                "num_perm": 256,
                "shingle_k": 5,
            },
        },
        "verifier": {
            "repository_identity": "cppmega",
            "script": "scripts/data/verify_global_dedup_store.py",
            "script_sha256": verifier["sha256"],
        },
    }
    dedup = write("global_dedup_receipt.json", dedup_payload)
    input_artifacts = {
        "archive_sha256_receipt": write(
            "runs/full/archive_sha256_receipt.json", {"fixture": "archive"}
        ),
        "archive_inventory_receipt": write(
            "runs/full/archive_inventory.json", {"fixture": "inventory"}
        ),
        "repo_list": write("runs/full/repo_list.json", {"fixture": "repos"}),
        "source_quarantine_manifest": write(
            "runs/full/source_quarantine_manifest.json",
            {"fixture": "quarantine"},
        ),
        "tokenizer": write("runs/full/tokenizer.json", {"fixture": "tokenizer"}),
    }
    run_artifacts = {
        "launch": write("runs/full/launch.json", {"fixture": "launch"}),
        "exit": write("runs/full/exit.json", {"fixture": "exit"}),
        "manifest": write("runs/full/manifest.json", {"fixture": "manifest"}),
        "archive_sha256_receipt": input_artifacts["archive_sha256_receipt"],
        "archive_inventory": input_artifacts["archive_inventory_receipt"],
        "repo_list": input_artifacts["repo_list"],
        "source_quarantine_manifest": input_artifacts[
            "source_quarantine_manifest"
        ],
        "tokenizer": input_artifacts["tokenizer"],
    }
    producer = {
        "cppmega": {"commit": "a" * 40, "tree_sha256": "b" * 64},
        "clang_indexer": {
            "source_sha256": "c" * 64,
            "dependency_closure_sha256": "d" * 64,
        },
    }
    portable_dedup = dict(dedup_payload)
    portable_database = dict(dedup_payload["database"])
    portable_database.pop("path")
    portable_dedup["database"] = portable_database
    portable_dedup["receipt_sha256"] = dedup["sha256"]
    run_receipt = {
        "run_id": "full",
        "launch": {"schema": "fixture", "sha256": run_artifacts["launch"]["sha256"]},
        "exit": {
            "schema": "fixture",
            "sha256": run_artifacts["exit"]["sha256"],
            "exit_code": 0,
        },
        "manifest": {
            "sha256": run_artifacts["manifest"]["sha256"],
            "done_units": 2,
            "failed_units": 0,
            "done_unit_set_sha256": "4" * 64,
            "failed_unit_set_sha256": "5" * 64,
        },
        "streams": "both",
        "selected_repositories": [],
        "terminal_repositories": ["repo"],
        "terminal_repository_set_sha256": "6" * 64,
        "input_artifacts": {
            name: descriptor["sha256"]
            for name, descriptor in input_artifacts.items()
        },
        "code_revision": producer,
        "allowlist_counts": {"code/1024": 1, "commits/1024": 1},
    }
    receipt_payload = {
        "schema": "cppmega_source_conveyor_composition_v1",
        "status": "complete",
        "plan_sha256": plan["sha256"],
        "buckets": [1024],
        "archive": {
            "repository_count": 1,
            "repository_names_sha256": "7" * 64,
            "input_binding_sha256": "8" * 64,
            "archive_identity_sha256": "9" * 64,
        },
        "dedup": portable_dedup,
        "runs": [run_receipt],
        "source_producers": [producer],
        "source_producer_set_sha256": hashlib.sha256(
            json.dumps(
                [producer],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest(),
        "coverage": {
            "expected_repositories": 1,
            "code_success_repositories": 1,
            "commit_success_repositories": 1,
            "failed_repositories_observed": 0,
            "failed_units_observed": 0,
            "unresolved_failed_units": 0,
            "repository_set_sha256": "a" * 64,
            "allowlist_counts": {"code/1024": 1, "commits/1024": 1},
        },
    }
    receipt = write("receipt.json", receipt_payload)
    return {
        "schema": receipt_payload["schema"],
        "receipt": {key: receipt[key] for key in ("path", "sha256")},
        "plan": {key: plan[key] for key in ("path", "sha256")},
        "dedup_receipt": {key: dedup[key] for key in ("path", "sha256")},
        "dedup_verifier": {key: verifier[key] for key in ("path", "sha256")},
        "runs": [{"run_id": "full", "artifacts": run_artifacts}],
    }


def _prefix_bundle(tmp_path):
    prefix = tmp_path / "data" / "seq_1024" / "cppmega_train"
    tokens_per_document = 4
    document_count = 6
    token_count = tokens_per_document * document_count
    token_values = [
        value
        for document in range(document_count)
        for value in (10 + document, 20 + document, 30 + document, 40 + document)
    ]
    source_identities = [
        (
            *_source_identity(f'{{"source_path":"src/doc-{document}.cc"}}'),
            f'{{"source_path":"src/doc-{document}.cc"}}',
        )
        for document in range(document_count)
    ]
    _write_bytes(
        prefix.with_suffix(".bin"),
        struct.pack(f"<{token_count}H", *token_values),
    )
    _write_bytes(
        prefix.with_suffix(".idx"),
        b"MMIDIDX\x00\x00"
        + struct.pack("<QBQQ", 1, 8, document_count, document_count + 1)
        + struct.pack(f"<{document_count}i", *([tokens_per_document] * document_count))
        + struct.pack(
            f"<{document_count}q",
            *(index * tokens_per_document * 2 for index in range(document_count)),
        )
        + struct.pack(f"<{document_count + 1}q", *range(document_count + 1)),
    )

    side_channel_paths = {}
    required_token_sidecars = set(publisher.REQUIRED_TOKEN_SIDECARS) | {
        "token_source_doc_ids"
    }
    for name in sorted(required_token_sidecars):
        dtype = publisher.TOKEN_SIDECAR_DTYPES[name]
        rel = f"{prefix.name}_{name}.bin"
        payload = bytearray(token_count * publisher.DTYPE_SIZES[dtype])
        if name == "token_structure_ids":
            payload[0] = 1
        if name == "loss_mask":
            payload[:] = bytes(
                value
                for document in range(document_count)
                for value in ((1, 1, 1, 0) if document == 0 else (0, 1, 1, 0))
            )
        if name == "doc_ids":
            payload[:] = struct.pack(
                f"<{token_count}I",
                *(
                    value
                    for _document in range(document_count)
                    for value in (1, 1, 1, 1)
                ),
            )
        if name == "token_source_identity_ids":
            payload[:] = struct.pack(
                f"<{token_count}Q",
                *(
                    source_identities[document][0]
                    for document in range(document_count)
                    for _token in range(tokens_per_document)
                ),
            )
        if name == "token_source_doc_ids":
            payload[:] = struct.pack(
                f"<{token_count}I",
                *(
                    document + 1
                    for document in range(document_count)
                    for _token in range(tokens_per_document)
                ),
            )
        _write_bytes(prefix.parent / rel, payload)
        side_channel_paths[name] = {"path": rel, "dtype": dtype}

    graph_sidecar_paths = {}
    for name in sorted(publisher.REQUIRED_GRAPH_SIDECARS):
        if name in {"token_call_edges", "token_type_edges"}:
            kind = "edge_pairs"
            dtype = "int32"
            shape_tail = [2]
            item_count = 0
        elif name.endswith("_edges"):
            kind = "edge_triples"
            dtype = "int32"
            shape_tail = [3]
            item_count = 1 if name == "token_domain_edges" else 0
        else:
            kind = "ragged_1d"
            if name in {"token_chunk_starts", "token_chunk_ends"}:
                dtype = "uint32"
            elif name == "token_chunk_kinds":
                dtype = "uint8"
            else:
                dtype = "uint16"
            shape_tail = [1]
            item_count = 1
        offsets_rel = f"{prefix.name}_{name}_offsets.bin"
        data_rel = f"{prefix.name}_{name}_data.bin"
        _write_bytes(
            prefix.parent / offsets_rel,
            struct.pack(
                f"<{document_count + 1}q",
                0,
                *([item_count] * document_count),
            ),
        )
        payload = b"\x00" * (item_count * shape_tail[0] * publisher.DTYPE_SIZES[dtype])
        if name == "token_domain_edges":
            payload = struct.pack("<3i", 1, 0, 5)
        elif name == "token_chunk_ends":
            payload = struct.pack("<I", tokens_per_document)
        elif name == "token_chunk_kinds":
            payload = struct.pack("<B", 1)
        _write_bytes(prefix.parent / data_rel, payload)
        graph_sidecar_paths[name] = {
            "kind": kind,
            "coordinate_space": publisher.GRAPH_ROUTE_COORDINATE_SPACES[name],
            "offsets_path": offsets_rel,
            "data_path": data_rel,
            "offset_dtype": "int64",
            "dtype": dtype,
            "item_count": item_count,
            "shape_tail": shape_tail,
        }

    source_platform = {
        "schema": "cppmega_source_platform_v1",
        "sequence_doc_offsets_path": f"{prefix.name}_source_platform_sequence_doc_offsets.bin",
        "doc_platform_offsets_path": f"{prefix.name}_source_platform_doc_id_offsets.bin",
        "platform_ids_path": f"{prefix.name}_source_platform_ids.bin",
        "source_document_count": document_count,
        "platform_id_count": document_count,
    }
    _write_bytes(
        prefix.parent / source_platform["sequence_doc_offsets_path"],
        struct.pack(f"<{document_count + 1}q", *range(document_count + 1)),
    )
    _write_bytes(
        prefix.parent / source_platform["doc_platform_offsets_path"],
        struct.pack(f"<{document_count + 1}q", *range(document_count + 1)),
    )
    _write_bytes(
        prefix.parent / source_platform["platform_ids_path"],
        struct.pack(f"<{document_count}H", *range(1, document_count + 1)),
    )
    registry_path = prefix.parent / f"{prefix.name}_source_identity_registry.sqlite"
    registry = sqlite3.connect(registry_path)
    try:
        registry.executescript(
            """
            CREATE TABLE source_identities (
                source_identity_id BLOB PRIMARY KEY,
                canonical_sha256 TEXT NOT NULL,
                source TEXT NOT NULL
            );
            CREATE TABLE sequence_source_identities (
                sequence_index INTEGER NOT NULL,
                source_identity_id BLOB NOT NULL,
                PRIMARY KEY(sequence_index, source_identity_id)
            );
            """
        )
        for sequence_index, (identity_id, digest, source) in enumerate(
            source_identities
        ):
            key = identity_id.to_bytes(8, "big", signed=False)
            registry.execute(
                "INSERT INTO source_identities VALUES (?, ?, ?)",
                (key, digest, source),
            )
            registry.execute(
                "INSERT INTO sequence_source_identities VALUES (?, ?)",
                (sequence_index, key),
            )
        registry.commit()
    finally:
        registry.close()
    objective_payload = _objective_payload()
    objective_sha256 = hashlib.sha256(
        json.dumps(
            objective_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    provenance_dir = tmp_path / "provenance"
    provenance_dir.mkdir()
    objective_source = provenance_dir / "objective_contract_seq1024.json"
    objective_source.write_text(json.dumps(objective_payload), encoding="utf-8")
    artifact_payload = {
        "schema": "cppmega_objective_materialization_artifact_v2",
        "graph_recipe": stage1_graph_recipe_binding(),
        "documents": document_count,
        "objective_contract": {
            "path": objective_source.name,
            "sha256": objective_sha256,
            "size_bytes": objective_source.stat().st_size,
            "file_sha256": hashlib.sha256(objective_source.read_bytes()).hexdigest(),
        },
        "parquet_shards": [
            {
                "path": "objectives_00000.parquet",
                "size_bytes": 1,
                "sha256": "1" * 64,
            }
        ],
        "converter": {
            "split": "all",
            "token_column": "input_ids",
            "length_column": "valid_token_count",
            "side_channels": [],
            "graph_sidecars": [],
            "source_platform_sidecar": "require",
            "loss_mask_alignment": "source_token_predicts_next_v1",
            "graph_relations": list(STAGE1_GRAPH_RELATIONS),
            "graph_pair_mask": "causal_same_document_upstream_v1",
            "chunk_edge_expansion": "cartesian_token_spans_v1",
        },
    }
    artifact_payload["artifact_set_sha256"] = hashlib.sha256(
        json.dumps(
            artifact_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    objective_artifact = provenance_dir / "objective_artifact_seq1024.json"
    objective_artifact.write_text(json.dumps(artifact_payload), encoding="utf-8")
    artifact_file_sha256 = hashlib.sha256(objective_artifact.read_bytes()).hexdigest()
    objective_ids_rel = f"{prefix.name}_objective_ids.bin"
    _write_bytes(
        prefix.parent / objective_ids_rel,
        bytes(range(1, document_count + 1)),
    )
    objective_contract = {
        "schema": "cppmega_pre_materialized_objectives_v1",
        "sha256": objective_sha256,
        "payload": objective_payload,
        "objective_id_sidecar": {
            "path": objective_ids_rel,
            "dtype": "uint8",
            "document_aligned": True,
        },
    }
    prefix_manifest = {
        "vocab_size": 65536,
        "tokenizer_contract": "megacpp",
        "dtype": "uint16",
        "token_count": token_count,
        "trained_token_count": 13,
        "document_count": document_count,
        "graph_sidecar_schema": "cppmega_graph_routes_v2",
        "loss_mask_alignment": "source_token_predicts_next_v1",
        "side_channel_paths": side_channel_paths,
        "graph_sidecar_paths": graph_sidecar_paths,
        "source_platform_sidecar": source_platform,
        "symbol_identity_schema_version": 3,
        "source_identity_registry": {
            "schema": publisher.SOURCE_IDENTITY_REGISTRY_SCHEMA,
            "path": registry_path.name,
            "id_encoding": "uint64_be",
            "canonical_digest": "sha256",
            "sequence_count": document_count,
            "identity_count": document_count,
            "sequence_identity_reference_count": document_count,
            "token_foreign_key_sidecar": "token_source_identity_ids",
        },
        publisher.CASE5_RECEIPT_KEY: {
            "status": "success",
            "schema": publisher.CASE5_SCHEMA_VERSION,
            "validated_shard_count": 1,
            "delimiter_contract_sha256": (publisher.DOMAIN_DELIMITER_CONTRACT_SHA256),
            "domain_schema_sha256": publisher.DOMAIN_SCHEMA_SHA256,
            "tokenizer_contract_sha256": publisher.TOKENIZER_CONTRACT_SHA256,
            "domain_route_columns": list(publisher.DOMAIN_ROUTE_COLUMNS),
            "graph_route_columns": list(publisher.GRAPH_ROUTE_COLUMNS),
            "graph_sidecars_written": True,
            "source_identity_registry_schema": (
                publisher.SOURCE_IDENTITY_REGISTRY_SCHEMA
            ),
        },
        "objective_contract": objective_contract,
        "objective_materialization": {
            **artifact_payload,
            "artifact_file_sha256": artifact_file_sha256,
        },
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(prefix_manifest), encoding="utf-8"
    )

    tokenizer_dir = tmp_path / "tokenizer"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "data/tokenizer_v2",
        tokenizer_dir,
    )
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    domain_contract = contracts_dir / "domain_schema_v1.json"
    tokenizer_contract = contracts_dir / "tokenizer_contract_v1.json"
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "data/domain_schema_v1.json",
        domain_contract,
    )
    shutil.copy2(
        Path(__file__).resolve().parents[1]
        / "data/tokenizer_v2/tokenizer_contract_v1.json",
        tokenizer_contract,
    )
    source_composition = _write_source_composition_provenance(tmp_path)
    paths = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    records = [
        {
            "path": str(path.relative_to(tmp_path)),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    artifact_set_sha256 = hashlib.sha256(
        json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    tokenizer_records = [
        record for record in records if str(record["path"]).startswith("tokenizer/")
    ]
    tokenizer_set_sha256 = hashlib.sha256(
        json.dumps(tokenizer_records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema": "cppmega_megatron_bundle_v3",
        "bundle_id": f"test-bundle-{artifact_set_sha256[:16]}",
        "tokenizer_contract": "megacpp-vocab-65536",
        "vocab_size": 65536,
        "training_contract": "objective_materialized",
        "known_limitations": [],
        "source_snapshot": {"source_composition": source_composition},
        "implementation": build_data_producer_binding(
            cppmega_commit="a" * 40,
            cppmega_tree_sha256="b" * 64,
            cppmega_mlx_commit="c" * 40,
            cppmega_mlx_tree_sha256="d" * 64,
            clang_indexer_sha256="e" * 64,
            clang_indexer_dependency_closure_sha256="f" * 64,
        ),
        "objective_materialization": {
            "schema": "cppmega_bucketed_objective_materializations_v1",
            "buckets": {
                "1024": {
                    "artifact_path": "provenance/objective_artifact_seq1024.json",
                    "artifact_schema": "cppmega_objective_materialization_artifact_v2",
                    "artifact_set_sha256": artifact_payload["artifact_set_sha256"],
                    "artifact_file_sha256": artifact_file_sha256,
                    "contract_path": "provenance/objective_contract_seq1024.json",
                    "contract_schema": "cppmega_pre_materialized_objectives_v1",
                    "contract_sha256": objective_sha256,
                    "contract_file_sha256": hashlib.sha256(
                        objective_source.read_bytes()
                    ).hexdigest(),
                    "source_snapshot": {
                        key: objective_payload["source_snapshot"][key]
                        for key in (
                            "schema",
                            "artifact_set_sha256",
                            "file_count",
                            "row_count",
                            "sampling",
                        )
                    },
                }
            },
        },
        "buckets": [1024],
        "tokenizer": {
            "path": "tokenizer",
            "contract": "megacpp-vocab-65536",
            "vocab_size": 65536,
            "files": tokenizer_records,
            "artifact_set_sha256": tokenizer_set_sha256,
        },
        "data_contracts": {
            "domain_schema": {
                "path": "contracts/domain_schema_v1.json",
                "size": domain_contract.stat().st_size,
                "sha256": hashlib.sha256(domain_contract.read_bytes()).hexdigest(),
            },
            "tokenizer_contract": {
                "path": "contracts/tokenizer_contract_v1.json",
                "size": tokenizer_contract.stat().st_size,
                "sha256": hashlib.sha256(tokenizer_contract.read_bytes()).hexdigest(),
            },
        },
        "bucket_results": [
            {
                "bucket": 1024,
                "prefix": str(prefix.relative_to(tmp_path)),
                "manifest": prefix_manifest,
            }
        ],
        "artifact_count": len(records),
        "artifact_bytes": sum(int(record["size"]) for record in records),
        "artifact_set_sha256": artifact_set_sha256,
        "artifacts": records,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return prefix


def _set_graph_chunks(
    prefix: Path,
    *,
    starts: list[int],
    ends: list[int],
    kinds: list[int] | None = None,
    dep_levels: list[int] | None = None,
) -> None:
    if len(starts) != len(ends):
        raise AssertionError("test graph chunk spans must be aligned")
    count = len(starts)
    kinds = [1] * count if kinds is None else kinds
    dep_levels = [0] * count if dep_levels is None else dep_levels
    if not len(kinds) == len(dep_levels) == count:
        raise AssertionError("test graph chunk metadata must be aligned")
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = {
        "token_chunk_starts": (starts, "I"),
        "token_chunk_ends": (ends, "I"),
        "token_chunk_kinds": (kinds, "B"),
        "token_chunk_dep_levels": (dep_levels, "H"),
    }
    document_count = int(manifest["document_count"])
    for name, (items, code) in values.items():
        spec = manifest["graph_sidecar_paths"][name]
        spec["item_count"] = count
        _write_bytes(
            prefix.parent / spec["offsets_path"],
            struct.pack(f"<{document_count + 1}q", 0, *([count] * document_count)),
        )
        _write_bytes(
            prefix.parent / spec["data_path"],
            struct.pack(f"<{count}{code}", *items),
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _rehash_bundle_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in manifest["artifacts"]:
        path = tmp_path / record["path"]
        records.append(
            {
                "path": record["path"],
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    records.sort(key=lambda record: record["path"])
    artifact_set_sha256 = hashlib.sha256(
        json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest["artifacts"] = records
    manifest["artifact_count"] = len(records)
    manifest["artifact_bytes"] = sum(record["size"] for record in records)
    manifest["artifact_set_sha256"] = artifact_set_sha256
    manifest["bundle_id"] = f"test-bundle-{artifact_set_sha256[:16]}"
    for result in manifest["bucket_results"]:
        prefix_manifest_path = tmp_path / (str(result["prefix"]) + ".json")
        result["manifest"] = json.loads(
            prefix_manifest_path.read_text(encoding="utf-8")
        )
    tokenizer_records = [
        record for record in records if record["path"].startswith("tokenizer/")
    ]
    manifest["tokenizer"]["files"] = tokenizer_records
    manifest["tokenizer"]["artifact_set_sha256"] = hashlib.sha256(
        json.dumps(tokenizer_records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_validate_bundle_rehashes_every_manifest_artifact(tmp_path):
    artifact, digest = _bundle(tmp_path)

    manifest, records = _validate_bundle(tmp_path, hash_jobs=2)

    assert manifest["artifact_bytes"] == sum(record["size"] for record in records)
    artifact_record = next(
        record for record in records if record["local_path"] == str(artifact)
    )
    assert artifact_record["sha256"] == digest


def test_validate_bundle_requires_source_composition_descriptor(tmp_path) -> None:
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_snapshot"].pop("source_composition")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source composition descriptor"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_incomplete_source_composition_coverage(
    tmp_path,
) -> None:
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = manifest["source_snapshot"]["source_composition"]
    receipt_path = tmp_path / descriptor["receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["coverage"]["code_success_repositories"] = 0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    descriptor["receipt"]["sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="full repository coverage"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_requires_objective_source_snapshot_summary(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = manifest["objective_materialization"]["buckets"]["1024"]
    descriptor.pop("source_snapshot")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="descriptor is invalid"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_publisher_accepts_exact_bounded_v2_sampling_contract():
    summary = publisher._objective_source_snapshot_summary(
        _bounded_v2_source_snapshot(), bucket=1024
    )

    publisher._validate_objective_source_summary(summary, bucket=1024)

    assert summary["sampling"] == _bounded_v2_sampling()


def test_publisher_validates_seeded_shard_cursor_permutation():
    sampling = _bounded_v2_sampling()
    sampling.update(
        {
            "seed": 17,
            "requested_samples": 1,
            "full_passes": 0,
            "tail_rows": 1,
            "min_row_reuse": 0,
            "max_row_reuse": 1,
        }
    )
    cursor = sampling["final_cursor"]
    assert isinstance(cursor, dict)
    cursor.update({"epoch": 0, "shard_index": 1, "source_index": 0})
    producer = sampling["producer"]
    assert isinstance(producer, dict)
    producer["row_group_rows"] = [[2], [1, 2], [4]]

    publisher._validate_objective_source_sampling(
        sampling,
        bucket=1024,
        total_rows=9,
        file_count=3,
        source_rows=(2, 3, 4),
    )

    cursor["shard_index"] = 0
    with pytest.raises(ValueError, match="final_cursor.*replay"):
        publisher._validate_objective_source_sampling(
            sampling,
            bucket=1024,
            total_rows=9,
            file_count=3,
            source_rows=(2, 3, 4),
        )


@pytest.mark.parametrize(
    "coordinate",
    tuple(sorted(publisher.OBJECTIVE_SAMPLING_V2_CURSOR_KEYS)),
)
def test_publisher_rejects_each_inexact_v2_replay_cursor_coordinate(coordinate):
    sampling = _bounded_v2_sampling()
    sampling.update(
        {
            "seed": 1,
            "requested_samples": 13,
            "full_passes": 0,
            "tail_rows": 13,
            "min_row_reuse": 0,
            "max_row_reuse": 1,
        }
    )
    producer = sampling["producer"]
    assert isinstance(producer, dict)
    producer["row_group_rows"] = [[3, 4], [2, 5], [1, 3, 2]]
    cursor = sampling["final_cursor"]
    assert isinstance(cursor, dict)
    cursor.update(
        {
            "epoch": 0,
            "shard_position": 1,
            "shard_index": 0,
            "row_group_position": 1,
            "row_group_index": 1,
            "record_batch_index": 1,
            "row_shuffle_position": 1,
            "row_index_in_record_batch": 0,
            "source_index": 12,
        }
    )

    publisher._validate_objective_source_sampling(
        sampling,
        bucket=1024,
        total_rows=20,
        file_count=3,
        source_rows=(7, 7, 6),
    )

    tampered = json.loads(json.dumps(sampling))
    tampered_cursor = tampered["final_cursor"]
    tampered_cursor[coordinate] = 0 if cursor[coordinate] != 0 else 1
    with pytest.raises(ValueError, match="final_cursor.*replay"):
        publisher._validate_objective_source_sampling(
            tampered,
            bucket=1024,
            total_rows=20,
            file_count=3,
            source_rows=(7, 7, 6),
        )


@pytest.mark.parametrize(
    "malformation",
    (
        "missing_record_batch_rows",
        "record_batch_size_alias",
        "invalid_record_batch_rows",
        "ordering_drift",
        "missing_cursor_coordinate",
        "missing_producer",
        "invalid_producer_version",
        "invalid_producer_name",
        "extra_producer_field",
        "producer_layout_mismatch",
        "wrong_source_index",
        "wrong_epoch",
    ),
)
def test_publisher_rejects_malformed_bounded_v2_sampling_contract(malformation):
    source_snapshot = _bounded_v2_source_snapshot()
    sampling = source_snapshot["sampling"]
    assert isinstance(sampling, dict)
    _malform_bounded_v2_sampling(sampling, malformation)

    with pytest.raises(ValueError, match="objective source"):
        publisher._objective_source_snapshot_summary(source_snapshot, bucket=1024)

    valid_summary = publisher._objective_source_snapshot_summary(
        _bounded_v2_source_snapshot(), bucket=1024
    )
    summary_sampling = valid_summary["sampling"]
    assert isinstance(summary_sampling, dict)
    _malform_bounded_v2_sampling(summary_sampling, malformation)
    with pytest.raises(ValueError, match="objective source"):
        publisher._validate_objective_source_summary(valid_summary, bucket=1024)


def test_publisher_preserves_fail_closed_v1_sampling_support():
    source_snapshot = _objective_payload()["source_snapshot"]
    summary = publisher._objective_source_snapshot_summary(source_snapshot, bucket=1024)
    publisher._validate_objective_source_summary(summary, bucket=1024)

    sampling = source_snapshot["sampling"]
    assert isinstance(sampling, dict)
    sampling.pop("seed")
    with pytest.raises(ValueError, match="sampling fields drifted"):
        publisher._objective_source_snapshot_summary(source_snapshot, bucket=1024)


def test_validate_bundle_matches_source_summary_to_staged_contract(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = manifest["objective_materialization"]["buckets"]["1024"]
    descriptor["source_snapshot"]["artifact_set_sha256"] = "3" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source_snapshot summary mismatch"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_manifest_path_escape(tmp_path):
    artifact, digest = _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "../sample.bin"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe artifact path"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_artifact_count_mismatch(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_count"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_missing_per_file_sha256(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0].pop("sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="valid sha256"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_wrong_tokenizer_contract(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer_contract"] = "wrong-tokenizer"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer_contract"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_tokenizer_rejects_case5_semantic_reserved_id_drift(tmp_path):
    tokenizer_root = tmp_path / "tokenizer"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "data/tokenizer_v2",
        tokenizer_root,
    )
    tokenizer_path = tokenizer_root / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    del vocab["<RESERVED_237>"]
    vocab["<DRIFTED_237>"] = 237
    entry = next(item for item in tokenizer["added_tokens"] if item["id"] == 237)
    entry["content"] = "<DRIFTED_237>"
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")

    with pytest.raises(ValueError, match="reserved.*237|ID 237"):
        _validate_tokenizer_directory(tokenizer_root)


def test_validate_bundle_rejects_zero_route_edges(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest_path = prefix.with_suffix(".json")
    prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
    spec = prefix_manifest["graph_sidecar_paths"]["token_domain_edges"]
    spec["item_count"] = 0
    (prefix.parent / spec["offsets_path"]).write_bytes(struct.pack("<7q", *([0] * 7)))
    (prefix.parent / spec["data_path"]).write_bytes(b"")
    prefix_manifest_path.write_text(json.dumps(prefix_manifest), encoding="utf-8")
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="nonempty route edge"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_prefix_rejects_graph_routes_without_case1_objective_contract(
    tmp_path,
):
    prefix = _prefix_bundle(tmp_path)
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("objective_contract")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="objective_contract"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_bundle_requires_positive_uint32_source_doc_ids(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest = json.loads(
        prefix.with_suffix(".json").read_text(encoding="utf-8")
    )
    spec = prefix_manifest["side_channel_paths"]["token_source_doc_ids"]
    (prefix.parent / spec["path"]).write_bytes(struct.pack("<24I", *([0] * 24)))
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="token_source_doc_ids.*positive"):
        _validate_bundle(tmp_path, hash_jobs=1)


@pytest.mark.parametrize(
    "first_sequence_doc_ids",
    (
        [0, 1, 1, 1],
        [2, 2, 2, 2],
        [1, 3, 3, 3],
        [1, 2, 1, 1],
    ),
)
def test_validate_prefix_rejects_noncanonical_attention_doc_ids(
    tmp_path, first_sequence_doc_ids
):
    prefix = _prefix_bundle(tmp_path)
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    spec = manifest["side_channel_paths"]["doc_ids"]
    path = prefix.parent / spec["path"]
    values = list(struct.unpack("<24I", path.read_bytes()))
    values[:4] = first_sequence_doc_ids
    path.write_bytes(struct.pack("<24I", *values))

    with pytest.raises(ValueError, match="doc_ids.*contiguous.*1..N"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_prefix_rejects_missing_loss_mask_alignment(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    manifest.pop("loss_mask_alignment")
    prefix.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="loss_mask_alignment"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_prefix_rejects_trained_cross_document_label(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    doc_spec = manifest["side_channel_paths"]["doc_ids"]
    doc_path = prefix.parent / doc_spec["path"]
    doc_values = list(struct.unpack("<24I", doc_path.read_bytes()))
    doc_values[:4] = [1, 1, 2, 2]
    doc_path.write_bytes(struct.pack("<24I", *doc_values))

    with pytest.raises(ValueError, match="cross-document transitions"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_prefix_matches_attention_segments_to_source_platform_csr(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    spec = manifest["side_channel_paths"]["doc_ids"]
    path = prefix.parent / spec["path"]
    values = list(struct.unpack("<24I", path.read_bytes()))
    values[:4] = [1, 1, 2, 2]
    path.write_bytes(struct.pack("<24I", *values))
    loss_spec = manifest["side_channel_paths"]["loss_mask"]
    loss_path = prefix.parent / loss_spec["path"]
    loss_values = list(loss_path.read_bytes())
    loss_values[:4] = [1, 0, 1, 0]
    loss_path.write_bytes(bytes(loss_values))
    manifest["trained_token_count"] = 12
    prefix.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="doc_ids cover 2.*source platform.*1"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_bundle_rejects_source_identity_sidecar_registry_mismatch(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest = json.loads(
        prefix.with_suffix(".json").read_text(encoding="utf-8")
    )
    spec = prefix_manifest["side_channel_paths"]["token_source_identity_ids"]
    path = prefix.parent / spec["path"]
    values = list(struct.unpack("<24Q", path.read_bytes()))
    values[0] ^= 1
    path.write_bytes(struct.pack("<24Q", *values))
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="registry/token mismatch"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_tampered_source_identity_witness(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest = json.loads(
        prefix.with_suffix(".json").read_text(encoding="utf-8")
    )
    registry_path = prefix.parent / prefix_manifest["source_identity_registry"]["path"]
    registry = sqlite3.connect(registry_path)
    try:
        registry.execute(
            "UPDATE source_identities SET source = ? WHERE rowid = 1",
            ('{"source_path":"src/tampered.cc"}',),
        )
        registry.commit()
    finally:
        registry.close()
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="invalid source identity witness"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_source_identity_reference_without_witness(
    tmp_path,
):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest = json.loads(
        prefix.with_suffix(".json").read_text(encoding="utf-8")
    )
    registry_path = prefix.parent / prefix_manifest["source_identity_registry"]["path"]
    registry = sqlite3.connect(registry_path)
    try:
        key = registry.execute(
            "SELECT source_identity_id FROM sequence_source_identities LIMIT 1"
        ).fetchone()[0]
        registry.execute(
            "DELETE FROM source_identities WHERE source_identity_id = ?", (key,)
        )
        registry.commit()
    finally:
        registry.close()
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="without canonical witnesses"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_edge_across_attention_document_boundary(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest = json.loads(
        prefix.with_suffix(".json").read_text(encoding="utf-8")
    )
    doc_spec = prefix_manifest["side_channel_paths"]["doc_ids"]
    doc_ids = [1] * 24
    doc_ids[1:4] = [2, 2, 2]
    (prefix.parent / doc_spec["path"]).write_bytes(struct.pack("<24I", *doc_ids))
    loss_spec = prefix_manifest["side_channel_paths"]["loss_mask"]
    loss_path = prefix.parent / loss_spec["path"]
    loss_values = list(loss_path.read_bytes())
    loss_values[:4] = [0, 1, 1, 0]
    loss_path.write_bytes(bytes(loss_values))
    prefix_manifest["trained_token_count"] = 12

    platform = prefix_manifest["source_platform_sidecar"]
    (prefix.parent / platform["sequence_doc_offsets_path"]).write_bytes(
        struct.pack("<7q", 0, 2, 3, 4, 5, 6, 7)
    )
    (prefix.parent / platform["doc_platform_offsets_path"]).write_bytes(
        struct.pack("<8q", *range(8))
    )
    (prefix.parent / platform["platform_ids_path"]).write_bytes(
        struct.pack("<7H", *([1] * 7))
    )
    platform["source_document_count"] = 7
    platform["platform_id_count"] = 7
    prefix.with_suffix(".json").write_text(
        json.dumps(prefix_manifest), encoding="utf-8"
    )
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="attention-document boundary"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_wrong_domain_edge_family_kind_26(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest = json.loads(
        prefix.with_suffix(".json").read_text(encoding="utf-8")
    )
    spec = prefix_manifest["graph_sidecar_paths"]["token_domain_edges"]
    (prefix.parent / spec["data_path"]).write_bytes(struct.pack("<3i", 1, 0, 26))
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="kind 26.*token_domain_edges"):
        _validate_bundle(tmp_path, hash_jobs=1)


@pytest.mark.parametrize(
    ("tokens", "domains", "roles", "confidences", "message"),
    [
        (
            [191, 195, 196, 192],
            [1, 1, 2, 1],
            [1, 1, 1, 1],
            [4, 4, 4, 4],
            "wrong domain",
        ),
        (
            [192, 191, 10, 11],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [4, 4, 0, 0],
            "unmatched",
        ),
        (
            [191, 195, 192, 196],
            [1, 2, 1, 2],
            [1, 1, 1, 1],
            [4, 4, 4, 4],
            "crossing",
        ),
        (
            [191, 195, 196, 10],
            [1, 2, 2, 1],
            [1, 1, 1, 0],
            [4, 4, 4, 0],
            "unclosed",
        ),
        (
            [191, 10, 192, 11],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [4, 4, 4, 0],
            "active delimiter scope",
        ),
    ],
)
def test_validate_bundle_rejects_invalid_domain_delimiter_mapping_and_balance(
    tmp_path, tokens, domains, roles, confidences, message
):
    prefix = _prefix_bundle(tmp_path)
    token_values = list(struct.unpack("<24H", prefix.with_suffix(".bin").read_bytes()))
    token_values[:4] = tokens
    prefix.with_suffix(".bin").write_bytes(struct.pack("<24H", *token_values))
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    for sidecar, values, dtype in (
        ("token_domain_ids", domains, "H"),
        ("token_role_ids", roles, "H"),
        ("token_confidence_ids", confidences, "B"),
    ):
        spec = manifest["side_channel_paths"][sidecar]
        path = prefix.parent / spec["path"]
        width = publisher.DTYPE_SIZES[spec["dtype"]]
        payload = bytearray(path.read_bytes())
        payload[: 4 * width] = struct.pack(f"<4{dtype}", *values)
        path.write_bytes(payload)
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match=message):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_missing_staged_tokenizer_contract(tmp_path):
    _prefix_bundle(tmp_path)
    contract_path = tmp_path / "tokenizer/tokenizer_contract_v1.json"
    contract_path.unlink()
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        record
        for record in manifest["artifacts"]
        if record["path"] != "tokenizer/tokenizer_contract_v1.json"
    ]
    manifest["tokenizer"]["files"] = [
        record
        for record in manifest["tokenizer"]["files"]
        if record["path"] != "tokenizer/tokenizer_contract_v1.json"
    ]
    manifest["tokenizer"]["artifact_set_sha256"] = publisher._artifact_set_sha256(
        manifest["tokenizer"]["files"]
    )
    manifest["artifact_count"] = len(manifest["artifacts"])
    manifest["artifact_bytes"] = sum(record["size"] for record in manifest["artifacts"])
    manifest["artifact_set_sha256"] = publisher._artifact_set_sha256(
        manifest["artifacts"]
    )
    manifest["bundle_id"] = f"test-bundle-{manifest['artifact_set_sha256'][:16]}"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required regular artifacts"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_requires_hashed_tokenizer_artifacts(tmp_path):
    _prefix_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("tokenizer")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer descriptor"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_tokenizer_vocab_drift_even_when_rehashed(tmp_path):
    _prefix_bundle(tmp_path)
    tokenizer_path = tmp_path / "tokenizer/tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"].pop(next(iter(tokenizer["model"]["vocab"])))
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="tokenizer vocab size"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_tokenizer_rejects_same_size_special_token_id_drift(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "data/tokenizer_v2",
        tokenizer_dir,
    )
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"]["<BOS>"] = 3
    tokenizer["model"]["vocab"]["<EOS>"] = 2
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")

    with pytest.raises(ValueError, match="token.*<BOS>.*(disagrees|must remain)"):
        publisher._validate_tokenizer_directory(tokenizer_dir)


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (9, struct.pack("<Q", 2), "MMIDIDX version"),
        (17, struct.pack("<B", 4), "MMIDIDX dtype"),
        (66, struct.pack("<q", 2), "sequence pointers"),
        (114, struct.pack("<q", 2), "document indices"),
    ],
)
def test_validate_prefix_rejects_mmididx_contract_drift(
    tmp_path, offset, replacement, message
):
    prefix = _prefix_bundle(tmp_path)
    index_path = prefix.with_suffix(".idx")
    payload = bytearray(index_path.read_bytes())
    payload[offset : offset + len(replacement)] = replacement
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_prefix_rejects_in_tree_sidecar_symlink(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    sidecar = prefix.parent / manifest["side_channel_paths"]["loss_mask"]["path"]
    sidecar.unlink()
    sidecar.symlink_to(prefix.with_suffix(".bin"))

    with pytest.raises(ValueError, match="regular file.*symlink"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_prefix_accepts_canonical_other_chunk_kind_zero(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    spec = manifest["graph_sidecar_paths"]["token_chunk_kinds"]
    (prefix.parent / spec["data_path"]).write_bytes(struct.pack("<B", 0))

    publisher._validate_prefix_manifest_contract(prefix)


def test_validate_prefix_rejects_out_of_range_chunk_kind(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    spec = manifest["graph_sidecar_paths"]["token_chunk_kinds"]
    (prefix.parent / spec["data_path"]).write_bytes(
        struct.pack("<B", publisher.GRAPH_CHUNK_KIND_COUNT)
    )

    with pytest.raises(ValueError, match="chunk kind.*canonical range"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_prefix_accepts_ordered_touching_graph_chunks(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    _set_graph_chunks(prefix, starts=[0, 2], ends=[2, 4], kinds=[0, 11])

    publisher._validate_prefix_manifest_contract(prefix)


@pytest.mark.parametrize(
    ("starts", "ends"),
    (
        ([0, 1], [2, 4]),
        ([2, 0], [4, 2]),
    ),
)
def test_validate_prefix_rejects_overlapping_or_unordered_graph_chunks(
    tmp_path, starts, ends
):
    prefix = _prefix_bundle(tmp_path)
    _set_graph_chunks(prefix, starts=starts, ends=ends)

    with pytest.raises(ValueError, match="ordered and nonoverlapping"):
        publisher._validate_prefix_manifest_contract(prefix)


@pytest.mark.parametrize("mutation", ("missing", "unexpected"))
def test_validate_prefix_requires_exact_graph_sidecar_key_set(tmp_path, mutation):
    prefix = _prefix_bundle(tmp_path)
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph_paths = manifest["graph_sidecar_paths"]
    if mutation == "missing":
        graph_paths.pop("token_call_edges")
    else:
        graph_paths["token_unknown_edges"] = dict(graph_paths["token_call_edges"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=f"graph sidecar key set.*{mutation}"):
        publisher._validate_prefix_manifest_contract(prefix)


@pytest.mark.parametrize("coordinate_space", (None, "token_index"))
def test_validate_prefix_requires_exact_graph_coordinate_space(
    tmp_path, coordinate_space
):
    prefix = _prefix_bundle(tmp_path)
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["graph_sidecar_paths"]["token_call_edges"]
    if coordinate_space is None:
        spec.pop("coordinate_space")
    else:
        spec["coordinate_space"] = coordinate_space
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="coordinate_space"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_prefix_accepts_graph_route_across_source_constituents(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    spec = manifest["side_channel_paths"]["token_source_doc_ids"]
    source_doc_path = prefix.parent / spec["path"]
    source_docs = bytearray(source_doc_path.read_bytes())
    source_docs[4:8] = struct.pack("<I", 2)
    source_doc_path.write_bytes(source_docs)

    publisher._validate_prefix_manifest_contract(prefix)


def test_validate_bundle_rejects_embedded_prefix_manifest_drift(tmp_path):
    _prefix_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bucket_results"][0]["manifest"]["vocab_size"] = 42
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="embedded prefix manifest"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_logical_manifest_preflight_rejects_nested_graph_contract_before_archive(
    tmp_path,
):
    _prefix_bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["bucket_results"][0]["manifest"]["graph_sidecar_schema"] = (
        "stale_graph_schema"
    )

    with pytest.raises(ValueError, match="graph_sidecar_schema"):
        publisher._validate_logical_manifest_contract(manifest)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    objective = manifest["bucket_results"][0]["manifest"]["objective_contract"]
    objective["payload"]["graph_auxiliary"]["relations"] = ["unknown"]
    with pytest.raises(ValueError, match="unknown relations"):
        publisher._validate_logical_manifest_contract(manifest)


def test_validate_bundle_rejects_wrong_prefix_graph_schema(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["graph_sidecar_schema"] = "flat_tokens_only"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="graph_sidecar_schema"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_bad_graph_csr_offsets(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    bad_offsets = prefix.parent / f"{prefix.name}_token_chunk_starts_offsets.bin"
    bad_offsets.write_bytes(struct.pack("<7q", 1, 1, 1, 1, 1, 1, 1))

    with pytest.raises(ValueError, match="CSR offsets"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_prefix_rejects_misaligned_graph_chunk_csr_counts(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["graph_sidecar_paths"]["token_chunk_ends"]
    spec["item_count"] = 2
    (prefix.parent / spec["offsets_path"]).write_bytes(
        struct.pack("<7q", 0, 2, 2, 2, 2, 2, 2)
    )
    (prefix.parent / spec["data_path"]).write_bytes(struct.pack("<2I", 1, 2))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="chunk CSR item counts"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_s3_env_selects_nebius_credentials_and_clears_aws_tokens(monkeypatch):
    _clear_s3_credential_env(monkeypatch)
    monkeypatch.setenv("NEBIUS_S3_ACCESS_KEY_ID", "nebius-access")
    monkeypatch.setenv("NEBIUS_S3_SECRET_ACCESS_KEY", "nebius-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "stale-aws-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "stale-aws-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "stale-session-token")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "stale-security-token")

    env = publisher._s3_env()

    assert env["AWS_ACCESS_KEY_ID"] == "nebius-access"
    assert env["AWS_SECRET_ACCESS_KEY"] == "nebius-secret"
    assert "AWS_SESSION_TOKEN" not in env
    assert "AWS_SECURITY_TOKEN" not in env


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("NEBIUS_S3_ACCESS_KEY_ID", "partial-nebius-access"),
        ("NEBIUS_S3_SECRET_ACCESS_KEY", "partial-nebius-secret"),
    ),
)
def test_s3_env_rejects_partial_nebius_family_even_with_complete_aws(
    monkeypatch, name, value
):
    _clear_s3_credential_env(monkeypatch)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "aws-session-token")
    monkeypatch.setenv(name, value)

    with pytest.raises(SystemExit) as error:
        publisher._s3_env()

    message = str(error.value)
    assert "complete Nebius S3 credential pair" in message
    assert value not in message
    assert "aws-secret" not in message
    assert "aws-session-token" not in message


def test_s3_env_preserves_session_token_only_for_complete_aws_family(monkeypatch):
    _clear_s3_credential_env(monkeypatch)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "aws-session-token")

    env = publisher._s3_env()

    assert env["AWS_ACCESS_KEY_ID"] == "aws-access"
    assert env["AWS_SECRET_ACCESS_KEY"] == "aws-secret"
    assert env["AWS_SESSION_TOKEN"] == "aws-session-token"


def test_head_contract_requires_size_metadata_and_exact_server_sha256():
    digest = hashlib.sha256(b"remote-bytes").hexdigest()
    expected_checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")

    assert _head_matches(
        {
            "ContentLength": 8,
            "Metadata": {"sha256": digest},
            "ChecksumSHA256": expected_checksum,
            "ChecksumType": "FULL_OBJECT",
        },
        size=8,
        sha256=digest,
    )
    assert not _head_matches(
        {
            "ContentLength": 7,
            "Metadata": {"sha256": digest},
            "ChecksumSHA256": expected_checksum,
        },
        size=8,
        sha256=digest,
    )
    assert not _head_matches(
        {
            "ContentLength": 8,
            "Metadata": {"sha256": "0" * 64},
            "ChecksumSHA256": expected_checksum,
        },
        size=8,
        sha256=digest,
    )
    assert not _head_matches(
        {"ContentLength": 8, "Metadata": {"sha256": digest}},
        size=8,
        sha256=digest,
    )
    assert not _head_matches(
        {"ContentLength": 8, "Metadata": {}, "ChecksumSHA256": expected_checksum},
        size=8,
        sha256=digest,
    )


def test_head_contract_accepts_nebius_metadata_key_casing():
    digest = hashlib.sha256(b"remote-bytes").hexdigest()
    assert _head_matches(
        {
            "ContentLength": 8,
            "Metadata": {"Sha256": digest},
            "ChecksumSHA256": base64.b64encode(bytes.fromhex(digest)).decode(
                "ascii"
            ),
            "ChecksumType": "FULL_OBJECT",
        },
        size=8,
        sha256=digest,
    )


def test_head_contract_rejects_arbitrary_multipart_composite_checksum():
    digest = hashlib.sha256(b"remote-bytes").hexdigest()
    arbitrary_composite = (
        base64.b64encode(hashlib.sha256(b"other-part").digest()).decode("ascii")
        + "-2"
    )
    assert not _head_matches(
        {
            "ContentLength": 8,
            "Metadata": {"sha256": digest},
            "ChecksumSHA256": arbitrary_composite,
            "ChecksumType": "COMPOSITE",
        },
        size=8,
        sha256=digest,
    )


def test_head_distinguishes_missing_object_from_transport_failure(monkeypatch):
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=254,
            stdout="",
            stderr="An error occurred (404) when calling HeadObject",
        ),
    )
    assert (
        _head(endpoint="https://s3.invalid", bucket="b", key="missing", env={}) is None
    )

    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=255, stdout="", stderr="Could not connect to endpoint"
        ),
    )
    with pytest.raises(RuntimeError, match="remote HEAD failed"):
        _head(endpoint="https://s3.invalid", bucket="b", key="unknown", env={})


def test_dry_run_upload_never_calls_aws(tmp_path):
    artifact, digest = _bundle(tmp_path)

    receipt = _upload_file(
        local=artifact,
        endpoint="https://example.invalid",
        bucket="bucket",
        key="prefix/sample.bin",
        size=artifact.stat().st_size,
        sha256=digest,
        env={},
        dry_run=True,
    )

    assert receipt["status"] == "dry_run"
    assert receipt["sha256"] == digest


def test_immutable_bundle_object_rejects_existing_remote_mismatch(
    tmp_path, monkeypatch
):
    artifact, digest = _bundle(tmp_path)
    monkeypatch.setattr(
        publisher,
        "_head",
        lambda **_kwargs: {
            "ContentLength": artifact.stat().st_size,
            "Metadata": {"sha256": "different"},
        },
    )

    def forbidden_upload(*_args, **_kwargs):
        raise AssertionError("immutable mismatch must fail before aws s3 cp")

    monkeypatch.setattr(publisher.subprocess, "run", forbidden_upload)
    with pytest.raises(RuntimeError, match="immutable remote object mismatch"):
        _upload_file(
            local=artifact,
            endpoint="https://example.invalid",
            bucket="bucket",
            key="bundles/test-bundle/data/sample.bin",
            size=artifact.stat().st_size,
            sha256=digest,
            env={},
            dry_run=False,
        )


def test_small_existing_object_with_forged_metadata_and_no_checksum_fails_closed(
    tmp_path, monkeypatch
):
    artifact, digest = _bundle(tmp_path)
    monkeypatch.setattr(
        publisher,
        "_head",
        lambda **_kwargs: {
            "ContentLength": artifact.stat().st_size,
            "Metadata": {"sha256": digest},
            "ETag": '"forged-metadata"',
        },
    )

    def forbidden_upload(*_args, **_kwargs):
        raise AssertionError("unverified immutable object must not be uploaded")

    monkeypatch.setattr(publisher.subprocess, "run", forbidden_upload)
    with pytest.raises(RuntimeError, match="immutable remote object mismatch"):
        _upload_file(
            local=artifact,
            endpoint="https://example.invalid",
            bucket="bucket",
            key="bundles/test-bundle/data/sample.bin",
            size=artifact.stat().st_size,
            sha256=digest,
            env={},
            dry_run=False,
        )


def test_small_existing_object_requires_exact_server_sha256_for_already_verified(
    tmp_path, monkeypatch
):
    artifact, digest = _bundle(tmp_path)
    expected_checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    monkeypatch.setattr(
        publisher,
        "_head",
        lambda **_kwargs: {
            "ContentLength": artifact.stat().st_size,
            "Metadata": {"sha256": digest},
            "ChecksumSHA256": expected_checksum,
            "ChecksumType": "FULL_OBJECT",
            "ETag": '"verified"',
        },
    )

    def forbidden_upload(*_args, **_kwargs):
        raise AssertionError("exactly verified object must not be uploaded")

    monkeypatch.setattr(publisher.subprocess, "run", forbidden_upload)
    receipt = _upload_file(
        local=artifact,
        endpoint="https://example.invalid",
        bucket="bucket",
        key="bundles/test-bundle/data/sample.bin",
        size=artifact.stat().st_size,
        sha256=digest,
        env={},
        dry_run=False,
    )

    assert receipt["status"] == "already_verified"


def test_small_upload_is_checksum_bound_and_create_only(tmp_path, monkeypatch):
    artifact, digest = _bundle(tmp_path)
    expected_checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    heads = iter(
        [
            None,
            {
                "ContentLength": artifact.stat().st_size,
                "Metadata": {"sha256": digest},
                "ChecksumSHA256": expected_checksum,
                "ChecksumType": "FULL_OBJECT",
                "ETag": '"etag"',
            },
        ]
    )
    monkeypatch.setattr(publisher, "_head", lambda **_kwargs: next(heads))

    @contextmanager
    def stable_snapshot(local, **_kwargs):
        yield local

    monkeypatch.setattr(publisher, "_stable_upload_snapshot", stable_snapshot)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    receipt = _upload_file(
        local=artifact,
        endpoint="https://example.invalid",
        bucket="bucket",
        key="bundles/test-bundle/data/sample.bin",
        size=artifact.stat().st_size,
        sha256=digest,
        env={},
        dry_run=False,
    )

    command = commands[0]
    assert command[:3] == ["aws", "s3api", "put-object"]
    assert command[command.index("--checksum-sha256") + 1] == expected_checksum
    assert command[command.index("--if-none-match") + 1] == "*"
    assert receipt["status"] == "uploaded_verified"


def test_over_5gib_selects_multipart_without_allocating_5gib(tmp_path, monkeypatch):
    artifact = tmp_path / "tiny-large-object"
    artifact.write_bytes(b"selection-only")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    requested_size = 5 * 1024**3 + 1
    monkeypatch.setattr(publisher, "_head", lambda **_kwargs: None)
    calls = []

    def fake_multipart(**kwargs):
        calls.append(kwargs)
        return {
            "key": kwargs["key"],
            "size": kwargs["size"],
            "sha256": kwargs["sha256"],
            "status": "uploaded_verified",
        }

    monkeypatch.setattr(publisher, "_upload_multipart_file", fake_multipart)

    receipt = _upload_file(
        local=artifact,
        endpoint="https://example.invalid",
        bucket="bucket",
        key="bundles/test-bundle/large.bin",
        size=requested_size,
        sha256=digest,
        env={},
        dry_run=False,
    )

    assert receipt["status"] == "uploaded_verified"
    assert calls[0]["size"] == requested_size
    assert calls[0]["initial_head"] is None


def test_multipart_part_failure_aborts_and_removes_temporary_part(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "multipart.bin"
    artifact.write_bytes(b"part-failure")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _force_tiny_multipart(monkeypatch)
    monkeypatch.setattr(publisher, "_head", lambda **_kwargs: None)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        operation = command[2]
        if operation == "create-multipart-upload":
            return SimpleNamespace(
                returncode=0, stdout=json.dumps({"UploadId": "upload-1"}), stderr=""
            )
        if operation == "upload-part":
            return SimpleNamespace(
                returncode=1, stdout="", stderr="injected part failure"
            )
        if operation == "abort-multipart-upload":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="upload-part 1 failed"):
        _upload_file(
            local=artifact,
            endpoint="https://example.invalid",
            bucket="bucket",
            key="bundles/test-bundle/large.bin",
            size=artifact.stat().st_size,
            sha256=digest,
            env={},
            dry_run=False,
        )

    assert [command[2] for command in commands] == [
        "create-multipart-upload",
        "upload-part",
        "abort-multipart-upload",
    ]
    assert _command_option(commands[-1], "--upload-id") == "upload-1"
    assert not list(tmp_path.glob(".cppmega-multipart-part-*"))


def test_existing_multipart_requires_locally_derived_exact_composite_checksum(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "multipart.bin"
    artifact.write_bytes(b"existing-destination")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _force_tiny_multipart(monkeypatch)
    part_size, part_count = publisher._multipart_layout(artifact.stat().st_size)
    part_checksums = publisher._multipart_part_checksums(
        artifact,
        size=artifact.stat().st_size,
        part_size=part_size,
        part_count=part_count,
    )
    composite = publisher._multipart_composite_sha256(part_checksums)
    metadata = publisher._multipart_metadata(
        sha256=digest, part_size=part_size, part_count=part_count
    )
    monkeypatch.setattr(
        publisher,
        "_head",
        lambda **_kwargs: {
            "ContentLength": artifact.stat().st_size,
            "Metadata": metadata,
            "ChecksumSHA256": composite,
            "ChecksumType": "COMPOSITE",
            "ETag": '"existing"',
        },
    )

    def forbidden_command(*_args, **_kwargs):
        raise AssertionError("an exactly verified destination must not be uploaded")

    monkeypatch.setattr(publisher.subprocess, "run", forbidden_command)

    receipt = _upload_file(
        local=artifact,
        endpoint="https://example.invalid",
        bucket="bucket",
        key="bundles/test-bundle/large.bin",
        size=artifact.stat().st_size,
        sha256=digest,
        env={},
        dry_run=False,
    )

    assert receipt["status"] == "already_verified"
    assert receipt["checksum_sha256"] == composite
    assert receipt["verification"]["metadata"] == metadata


@pytest.mark.parametrize("checksum_state", ("missing", "arbitrary"))
def test_multipart_existing_destination_without_exact_checksum_is_rejected(
    tmp_path, monkeypatch, checksum_state
):
    artifact = tmp_path / "multipart.bin"
    artifact.write_bytes(b"existing-destination")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _force_tiny_multipart(monkeypatch)
    part_size, part_count = publisher._multipart_layout(artifact.stat().st_size)
    metadata = publisher._multipart_metadata(
        sha256=digest, part_size=part_size, part_count=part_count
    )
    head = {
        "ContentLength": artifact.stat().st_size,
        "Metadata": metadata,
        "ChecksumType": "COMPOSITE",
        "ETag": '"existing"',
    }
    if checksum_state == "arbitrary":
        arbitrary = base64.b64encode(hashlib.sha256(b"forged-parts").digest()).decode(
            "ascii"
        )
        head["ChecksumSHA256"] = f"{arbitrary}-{part_count}"
    monkeypatch.setattr(publisher, "_head", lambda **_kwargs: head)

    def forbidden_command(*_args, **_kwargs):
        raise AssertionError("an existing immutable destination must not be replaced")

    monkeypatch.setattr(publisher.subprocess, "run", forbidden_command)

    with pytest.raises(RuntimeError, match="cannot be verified exactly"):
        _upload_file(
            local=artifact,
            endpoint="https://example.invalid",
            bucket="bucket",
            key="bundles/test-bundle/large.bin",
            size=artifact.stat().st_size,
            sha256=digest,
            env={},
            dry_run=False,
        )


def test_concurrent_multipart_publishers_preserve_immutable_destination(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "multipart.bin"
    artifact.write_bytes(b"concurrent-publication")
    size = artifact.stat().st_size
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _force_tiny_multipart(monkeypatch)
    lock = threading.Lock()
    initial_head_barrier = threading.Barrier(2)
    complete_barrier = threading.Barrier(2)
    state = {
        "initial_heads": 0,
        "next_upload": 0,
        "uploads": {},
        "object_head": None,
        "commands": [],
        "aborts": [],
    }

    def response(payload=None, *, returncode=0, stderr=""):
        return SimpleNamespace(
            returncode=returncode,
            stdout=json.dumps(payload or {}) if returncode == 0 else "",
            stderr=stderr,
        )

    def fake_run(command, **_kwargs):
        operation = command[2]
        with lock:
            state["commands"].append(list(command))

        if operation == "head-object":
            with lock:
                current = state["object_head"]
                is_initial = current is None and state["initial_heads"] < 2
                if is_initial:
                    state["initial_heads"] += 1
            if is_initial:
                initial_head_barrier.wait(timeout=10)
                return response(returncode=254, stderr="404 Not Found")
            if current is None:
                return response(returncode=254, stderr="404 Not Found")
            return response(current)

        if operation == "create-multipart-upload":
            with lock:
                state["next_upload"] += 1
                upload_id = f"upload-{state['next_upload']}"
                state["uploads"][upload_id] = {
                    "metadata": json.loads(_command_option(command, "--metadata")),
                    "parts": {},
                }
            return response({"UploadId": upload_id})

        if operation == "upload-part":
            upload_id = _command_option(command, "--upload-id")
            part_number = int(_command_option(command, "--part-number"))
            body = Path(_command_option(command, "--body")).read_bytes()
            checksum = base64.b64encode(hashlib.sha256(body).digest()).decode("ascii")
            assert checksum == _command_option(command, "--checksum-sha256")
            etag = f'"{upload_id}-part-{part_number}"'
            with lock:
                state["uploads"][upload_id]["parts"][part_number] = {
                    "length": len(body),
                    "checksum": checksum,
                    "etag": etag,
                }
            return response({"ETag": etag, "ChecksumSHA256": checksum})

        if operation == "complete-multipart-upload":
            upload_id = _command_option(command, "--upload-id")
            manifest_path = Path(
                _command_option(command, "--multipart-upload").removeprefix("file://")
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            part_checksums = [part["ChecksumSHA256"] for part in manifest["Parts"]]
            composite = publisher._multipart_composite_sha256(part_checksums)
            assert _command_option(command, "--checksum-sha256") == composite.rsplit(
                "-", 1
            )[0]
            assert _command_option(command, "--if-none-match") == "*"
            assert int(_command_option(command, "--mpu-object-size")) == size
            complete_barrier.wait(timeout=10)
            with lock:
                if state["object_head"] is not None:
                    return response(
                        returncode=255,
                        stderr="PreconditionFailed: 412 If-None-Match",
                    )
                upload = state["uploads"][upload_id]
                etag = f'"{upload_id}-complete"'
                state["object_head"] = {
                    "ContentLength": sum(
                        part["length"] for part in upload["parts"].values()
                    ),
                    "Metadata": upload["metadata"],
                    "ChecksumSHA256": composite,
                    "ChecksumType": "COMPOSITE",
                    "ETag": etag,
                }
            return response(
                {
                    "ETag": etag,
                    "ChecksumSHA256": composite,
                    "ChecksumType": "COMPOSITE",
                }
            )

        if operation == "abort-multipart-upload":
            with lock:
                state["aborts"].append(_command_option(command, "--upload-id"))
            return response()

        raise AssertionError(command)

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    def publish():
        return _upload_file(
            local=artifact,
            endpoint="https://example.invalid",
            bucket="bucket",
            key="bundles/test-bundle/large.bin",
            size=size,
            sha256=digest,
            env={},
            dry_run=False,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish) for _ in range(2)]
        receipts = [future.result(timeout=20) for future in futures]

    assert sorted(receipt["status"] for receipt in receipts) == [
        "already_verified",
        "uploaded_verified",
    ]
    concurrent_receipt = next(
        receipt for receipt in receipts if receipt["status"] == "already_verified"
    )
    assert concurrent_receipt["race_resolution"] == "matching_concurrent_publisher"
    assert concurrent_receipt["verification"]["metadata"]["sha256"] == digest
    assert len(state["aborts"]) == 1
    completes = [
        command
        for command in state["commands"]
        if command[2] == "complete-multipart-upload"
    ]
    assert len(completes) == 2
    assert all(_command_option(command, "--if-none-match") == "*" for command in completes)


def test_latest_pointer_update_uses_remote_etag_compare_and_swap(tmp_path, monkeypatch):
    artifact, digest = _bundle(tmp_path)
    expected_checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    heads = iter(
        [
            {
                "ContentLength": artifact.stat().st_size,
                "Metadata": {"sha256": "0" * 64},
                "ETag": '"old-etag"',
            },
            {
                "ContentLength": artifact.stat().st_size,
                "Metadata": {"sha256": digest},
                "ChecksumSHA256": expected_checksum,
                "ChecksumType": "FULL_OBJECT",
                "ETag": '"new-etag"',
            },
        ]
    )
    monkeypatch.setattr(publisher, "_head", lambda **_kwargs: next(heads))

    @contextmanager
    def stable_snapshot(local, **_kwargs):
        yield local

    monkeypatch.setattr(publisher, "_stable_upload_snapshot", stable_snapshot)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    _upload_file(
        local=artifact,
        endpoint="https://example.invalid",
        bucket="bucket",
        key="latest.json",
        size=artifact.stat().st_size,
        sha256=digest,
        env={},
        dry_run=False,
        allow_overwrite=True,
    )

    command = commands[0]
    assert command[command.index("--if-match") + 1] == '"old-etag"'


def test_upload_snapshot_isolated_from_source_mutation(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable")
    digest = hashlib.sha256(b"stable").hexdigest()

    with _stable_upload_snapshot(source, size=6, sha256=digest) as snapshot:
        source.write_bytes(b"changed")
        assert snapshot.read_bytes() == b"stable"


def test_archive_member_set_must_be_exact_and_unique():
    _validate_archive_member_names(
        ["data/sample.bin", "manifest.json"],
        {"data/sample.bin", "manifest.json"},
    )
    with pytest.raises(ValueError, match="duplicate"):
        _validate_archive_member_names(
            ["manifest.json", "manifest.json"], {"manifest.json"}
        )
    with pytest.raises(ValueError, match="member set mismatch"):
        _validate_archive_member_names(
            ["manifest.json", "unexpected.bin"],
            {"manifest.json", "data/sample.bin"},
        )


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_validate_archive_binds_exact_members_and_manifest(tmp_path):
    artifact, _digest = _bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    archive = _archive(tmp_path, artifact)

    size, digest = _validate_archive(
        bundle=tmp_path, archive=archive, manifest=manifest
    )

    assert size == archive.stat().st_size
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_validate_archive_rejects_payload_that_disagrees_with_manifest(tmp_path):
    artifact, _digest = _bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    payload = bytearray(artifact.read_bytes())
    payload[0] ^= 1
    artifact.write_bytes(payload)
    archive = _archive(tmp_path, artifact)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate_archive(bundle=tmp_path, archive=archive, manifest=manifest)


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_archive_publish_runs_full_prefix_contract_validation(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest_path = prefix.with_suffix(".json")
    prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
    prefix_manifest["graph_sidecar_schema"] = "stale_graph_schema"
    prefix_manifest_path.write_text(json.dumps(prefix_manifest), encoding="utf-8")
    manifest = _rehash_bundle_manifest(tmp_path)
    manifest["bucket_results"][0]["manifest"] = prefix_manifest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive = _archive(tmp_path, prefix.with_suffix(".bin"))

    with pytest.raises(ValueError, match="graph_sidecar_schema"):
        main(["--bundle", str(tmp_path), "--archive", str(archive), "--dry-run"])


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_archive_transport_dry_run_writes_commit_order_receipt(tmp_path):
    artifact, _digest = _bundle(tmp_path)
    archive = _archive(tmp_path, artifact)

    assert (
        main(["--bundle", str(tmp_path), "--archive", str(archive), "--dry-run"]) == 0
    )

    receipt = json.loads(
        (tmp_path / "archive_publish_dry_run_receipt.json").read_text(encoding="utf-8")
    )
    bundle_id = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[
        "bundle_id"
    ]
    assert receipt["archive"]["key"].startswith(
        f"cppmega-megatron/macro-routes/transports/{bundle_id}/bundle-"
    )
    assert receipt["archive"]["key"].endswith(".tar.zst")
    assert receipt["logical_manifest"]["key"].endswith(
        f"/{bundle_id}/logical_manifest.json"
    )
    assert receipt["transport"]["key"].endswith(f"/{bundle_id}/transport.json")
    assert receipt["latest_transport"]["key"].endswith("/latest_transport.json")
    assert receipt["archive"]["status"] == "dry_run"
    assert receipt["status"] == "complete"
    assert receipt["archive_validation"] == {
        "status": "verified",
        "member_count": len(
            json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[
                "artifacts"
            ]
        )
        + 1,
        "artifact_set_sha256": json.loads(
            (tmp_path / "manifest.json").read_text(encoding="utf-8")
        )["artifact_set_sha256"],
        "logical_manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.json").read_bytes()
        ).hexdigest(),
    }


def test_loose_publish_receipt_is_incremental_and_bundle_bound(tmp_path):
    _prefix_bundle(tmp_path)

    assert main(["--bundle", str(tmp_path), "--dry-run", "--jobs", "2"]) == 0

    receipt_path = tmp_path / "publish_dry_run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["bundle_id"] == manifest["bundle_id"]
    assert receipt["artifact_set_sha256"] == manifest["artifact_set_sha256"]
    assert len(receipt["artifacts"]) == manifest["artifact_count"]

    receipt["bundle_id"] = "different-bundle"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="publish receipt binding"):
        main(["--bundle", str(tmp_path), "--dry-run"])


def test_loose_publish_rejects_stale_artifact_receipt_entry(tmp_path):
    _prefix_bundle(tmp_path)
    assert main(["--bundle", str(tmp_path), "--dry-run"]) == 0
    receipt_path = tmp_path / "publish_dry_run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"][0]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="publish receipt artifact mismatch"):
        main(["--bundle", str(tmp_path), "--dry-run"])
