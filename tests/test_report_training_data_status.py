from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.report_training_data_status import (
    STATUS_SCHEMA,
    publish_status,
    scan_parquet_snapshot,
)


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True)
    table = pa.table(
        {
            "valid_token_count": pa.array([15], type=pa.int32()),
            "trained_token_count": pa.array([13], type=pa.int32()),
            "num_docs": pa.array([2], type=pa.int32()),
            "source_doc_types": pa.array(
                [["code", "code"]], type=pa.list_(pa.string())
            ),
            "source_build_kinds": pa.array(
                [[None, "python"]], type=pa.list_(pa.string())
            ),
            "source_doc_token_lengths": pa.array(
                [[10, 5]], type=pa.list_(pa.int32())
            ),
        }
    )
    pq.write_table(table, path, compression="zstd")


def test_parquet_snapshot_counts_batches_schema_and_logical_routes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "packed"
    _write_parquet(root / "1024" / "one.parquet")

    result = scan_parquet_snapshot(
        root,
        batch_size=192,
        jobs=1,
        classify_documents=True,
    )

    assert result["files"] == 1
    assert result["rows"] == 1
    assert result["valid_tokens"] == 15
    assert result["trained_tokens"] == 13
    assert result["capacity_tokens"] == 1024
    assert result["compression"]["all_zstd"] is True
    assert result["schema"]["uniform"] is True
    assert result["classification"]["conserved"] is True
    assert result["classification"]["by_category"]["c_cpp_source"] == {
        "documents": 1,
        "valid_tokens": 10,
        "trained_tokens": 9,
    }
    assert result["classification"]["by_category"]["python_aux"] == {
        "documents": 1,
        "valid_tokens": 5,
        "trained_tokens": 4,
    }
    assert result["buckets"]["1024"]["batch"]["full_batches"] == 0
    assert result["buckets"]["1024"]["batch"]["remainder_rows"] == 1


def _minimal_status(*, sha: str, live_tokens: int) -> dict[str, object]:
    bucket = {
        "files": 1,
        "rows": 1,
        "valid_tokens": live_tokens,
        "trained_tokens": live_tokens - 1,
        "pad_tokens": 1024 - live_tokens,
        "batch": {"full_batches": 0, "remainder_rows": 1},
    }
    return {
        "schema": STATUS_SCHEMA,
        "generated_at": "2026-07-31T00:00:00+00:00",
        "status_sha256": sha,
        "batch_size": 192,
        "datasets": {
            "live_source": {
                "state": "packed_unsealed",
                "release_ready": False,
                "blockers": ["test"],
                "version": {
                    "source_repo_list": {"sha256": "source-list"},
                },
                "parquet": {
                    "root": "/data/source",
                    "files": 1,
                    "rows": 1,
                    "valid_tokens": live_tokens,
                    "trained_tokens": live_tokens - 1,
                    "buckets": {"1024": bucket},
                    "classification": {"by_category": {}},
                    "schema": {
                        "counts": {"source-schema": 1},
                        "metadata_by_sha256": {
                            "source-schema": {
                                "cppmega.tokenizer_contract_sha256": "tokenizer"
                            }
                        },
                    },
                },
            },
            "sealed_megatron": {
                "state": "sealed_megatron",
                "release_ready": True,
                "manifest": "/data/sealed/manifest.json",
                "version": {
                    "bundle_id": "sealed",
                    "artifact_set_sha256": "artifact-set",
                },
                "totals": {
                    "rows": 1,
                    "valid_tokens": 10,
                    "trained_tokens": 9,
                },
                "buckets": {"1024": bucket},
                "sidecars": {"dense": ["loss_mask"], "ragged_graph": []},
            },
            "validation_bundle": {
                "version": {"bundle_id": "mini"},
                "totals": {"valid_tokens": 2, "trained_tokens": 1},
            },
            "pr_mr": {
                "state": "verified_store_not_materialized",
                "release_ready": False,
                "version": {"scan_id": "scan", "store_sha256": "store"},
                "records": {"stored_prs": 3},
            },
            "ci": {
                "state": "cas_staged_not_exported",
                "release_ready": False,
                "token_accounting": {"store_local_unique_upper_bound": 20},
                "stores": [
                    {
                        "interval": {"start": "a", "end": "b"},
                        "sidecar_set_sha256": "sidecars",
                        "tokenizer": {
                            "tokenizer_contract_sha256": "tokenizer"
                        },
                    }
                ],
                "legacy_sample": {
                    "parquet": {
                        "valid_tokens": 4,
                        "buckets": {"1024": bucket},
                    }
                },
            },
        },
    }


def test_publish_status_appends_changelog_only_for_semantic_change(
    tmp_path: Path,
) -> None:
    output = tmp_path / "status"
    first = _minimal_status(sha="1" * 64, live_tokens=15)
    publish_status(first, output)
    publish_status(first, output)
    second = _minimal_status(sha="2" * 64, live_tokens=20)
    publish_status(second, output)

    entries = [
        json.loads(line)
        for line in (output / "changelog.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(entries) == 2
    assert entries[0]["previous_status_sha256"] is None
    assert entries[1]["previous_status_sha256"] == "1" * 64
    assert entries[1]["numeric_delta"]["live_source"]["valid_tokens"] == 5
    current = json.loads((output / "current.json").read_text(encoding="utf-8"))
    assert current["status_sha256"] == "2" * 64
