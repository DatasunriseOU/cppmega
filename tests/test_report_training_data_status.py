from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.report_training_data_status import (
    HEARTBEAT_SCHEMA,
    STATUS_SCHEMA,
    _sha256,
    _without_volatile,
    check_heartbeat,
    collect_freshness,
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


def _freshness_config(
    source_receipt: Path, ci_receipt: Path
) -> dict[str, object]:
    return {
        "source": {"completion_receipt": str(source_receipt)},
        "ci": {"progress_receipts": [str(ci_receipt)]},
    }


def test_collect_freshness_flags_inputs_older_than_threshold(
    tmp_path: Path,
) -> None:
    fresh = tmp_path / "done.json"
    old = tmp_path / "fetch.progress.json"
    fresh.write_text("{}", encoding="utf-8")
    old.write_text("{}", encoding="utf-8")
    now = time.time()
    os.utime(fresh, (now - 60, now - 60))
    os.utime(old, (now - 7200, now - 7200))

    result = collect_freshness(
        _freshness_config(fresh, old), stale_minutes=30.0, now=now
    )

    old_name = f"ci_progress_receipt:{old}"
    assert result["stale"] == [old_name]
    assert result["stale_minutes"] == 30.0
    source = result["upstreams"]["source_completion_receipt"]
    assert source["stale"] is False
    assert source["missing"] is False
    assert source["age_seconds"] == 60
    assert result["upstreams"][old_name]["age_seconds"] == 7200


def test_collect_freshness_treats_missing_input_as_stale(
    tmp_path: Path,
) -> None:
    present = tmp_path / "fetch.progress.json"
    present.write_text("{}", encoding="utf-8")
    missing = tmp_path / "done.json"

    result = collect_freshness(
        _freshness_config(missing, present),
        stale_minutes=30.0,
        now=time.time(),
    )

    assert result["stale"] == ["source_completion_receipt"]
    assert result["upstreams"]["source_completion_receipt"]["missing"] is True


def test_freshness_block_is_excluded_from_status_hash() -> None:
    first = _minimal_status(sha="0" * 64, live_tokens=15)
    second = _minimal_status(sha="0" * 64, live_tokens=15)
    first["freshness"] = {"stale": [], "upstreams": {"a": {"age_seconds": 1}}}
    second["freshness"] = {
        "stale": ["a"],
        "upstreams": {"a": {"age_seconds": 9999}},
    }

    assert _sha256(_without_volatile(first)) == _sha256(_without_volatile(second))


def test_publish_status_writes_heartbeat_and_check_detects_staleness(
    tmp_path: Path,
) -> None:
    output = tmp_path / "status"
    status = _minimal_status(sha="1" * 64, live_tokens=15)
    paths = publish_status(status, output)

    heartbeat = json.loads(paths["heartbeat"].read_text(encoding="utf-8"))
    assert heartbeat["schema"] == HEARTBEAT_SCHEMA
    assert heartbeat["pid"] == os.getpid()
    assert heartbeat["status_sha256"] == "1" * 64

    recorded = datetime.fromisoformat(heartbeat["recorded_at"]).timestamp()
    fresh = check_heartbeat(
        paths["heartbeat"], stale_after_seconds=600.0, now=recorded + 60
    )
    assert fresh["stale"] is False
    assert fresh["pid"] == os.getpid()
    stale = check_heartbeat(
        paths["heartbeat"], stale_after_seconds=600.0, now=recorded + 3600
    )
    assert stale["stale"] is True
    assert stale["age_seconds"] == 3600
