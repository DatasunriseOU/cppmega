from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.report_training_data_status import (
    HEARTBEAT_SCHEMA,
    STATUS_SCHEMA,
    _collect_frozen_ci_case5,
    _sha256,
    _utc_now,
    _without_volatile,
    build_status,
    check_heartbeat,
    collect_ci_status,
    collect_gitlab_mr_status,
    collect_live_source,
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
            "source_doc_token_lengths": pa.array([[10, 5]], type=pa.list_(pa.int32())),
        }
    )
    pq.write_table(table, path, compression="zstd")


def _write_frozen_ci_receipts(
    root: Path,
    source: Path,
    *,
    parser_script_sha256: str = "6" * 64,
) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
    root.mkdir(exist_ok=True)
    source.mkdir(exist_ok=True)
    tokenizer_path = root / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    tokenizer = {
        "artifact_sha256": tokenizer_sha256,
        "tokenizer_contract_sha256": "1" * 64,
    }
    store = {
        "schema": "cppmega_ci_content_store_receipt_v1",
        "status": "complete",
        "verification": {"ok": True},
        "exact_unique_payload_tokens": 61_311_228_208,
        "counters": {"exact_unique_payload_tokens": 61_311_228_208},
        "policy_sha256": "b" * 64,
        "sqlite_schema_sha256": "c" * 64,
        "sqlite_logical_sha256": "d" * 64,
        "logical_content_set_sha256": "2" * 64,
        "logical_token_sequence_set_sha256": "3" * 64,
        "occurrence_set_sha256": "4" * 64,
    }
    summary = {
        "attempt_statuses": {"done": 1},
        "sidecar_set_sha256": "5" * 64,
    }
    settings = {
        "tokenizer_contract": json.dumps(tokenizer),
        "tokenizer_fingerprint": "fingerprint",
        "parser_script_sha256": parser_script_sha256,
    }
    fetch = {
        "schema": "cppmega_ci_stream_fetch_receipt_v3",
        "completed_at": "2026-08-02T08:08:49Z",
        "content_store_receipt": store,
        "fetch_state": summary,
        "frozen_fetch_state": {
            "artifact": {"sha256": "7" * 64},
            "schema": "cppmega_ci_stream_fetch_v4",
            "sqlite_schema_sha256": "a" * 64,
            "sqlite_logical_sha256": "8" * 64,
            "settings": settings,
            "summary": summary,
            "sidecar_set_sha256": summary["sidecar_set_sha256"],
        },
        "tokenizer_contract": tokenizer,
        "tokenizer_fingerprint": settings["tokenizer_fingerprint"],
    }
    cut = {
        "schema": "cppmega_ci_threshold_cow_cut_receipt_v1",
        "status": "complete",
        "cut": {
            "snapshot_root": str(root),
            "source_root": str(source),
            "source_code_commit": "9" * 40,
        },
        "semantics": {
            "scope": "store-local-threshold-snapshot",
            "cross_store_global_dedup": False,
            "exhaustive_complete": False,
            "production_complete": False,
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": tokenizer_sha256,
        },
    }
    for path, value in (
        (root / "threshold_cut.receipt.json", cut),
        (root / "fetch.receipt.json", fetch),
        (root / "store.receipt.json", store),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")
    return source / "fetch.progress.json", tokenizer, store, fetch


def test_frozen_ci_threshold_is_staged_until_export_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frozen"
    progress, _tokenizer, _store, _fetch = _write_frozen_ci_receipts(
        root, tmp_path / "live"
    )

    result = _collect_frozen_ci_case5(root, progress_paths=[progress])

    assert result["staged_exact_unique_payload_tokens"] == 61_311_228_208
    assert result["ready_valid_tokens"] == 0
    assert result["ready_trained_tokens"] == 0
    assert result["export"] is None
    assert result["overlaps_progress_receipts"] == [str(progress)]

    status = _minimal_status(sha="a" * 64, live_tokens=15)
    ci = status["datasets"]["ci"]
    ci["frozen_case5_snapshot"] = result
    ci["token_accounting"].update(
        {
            "frozen_snapshot_exact_unique_payload_tokens": 61_311_228_208,
            "ready_valid_tokens": 0,
            "ready_trained_tokens": 0,
        }
    )
    paths = publish_status(status, tmp_path / "status")
    changelog = json.loads(paths["changelog"].read_text(encoding="utf-8"))
    assert changelog["summary"]["ci"]["frozen_case5_snapshot"]["export"] is None


def test_frozen_ci_ready_tokens_include_only_train_split(
    tmp_path: Path,
) -> None:
    from tests.test_build_macro_routes_megatron_bundle import (
        _write_content_store_ci_export,
    )

    root = tmp_path / "frozen"
    export_path = _write_content_store_ci_export(root / "case5")
    export = json.loads(export_path.read_text(encoding="utf-8"))
    parser_sha256 = export["input_fetch_state"]["settings"]["parser_script_sha256"]
    progress, tokenizer, store, fetch = _write_frozen_ci_receipts(
        root,
        tmp_path / "live",
        parser_script_sha256=parser_sha256,
    )

    input_store = export["input_store"]
    input_store["receipt_sha256"] = hashlib.sha256(
        (root / "store.receipt.json").read_bytes()
    ).hexdigest()
    for name in (
        "policy_sha256",
        "sqlite_schema_sha256",
        "sqlite_logical_sha256",
        "logical_content_set_sha256",
        "logical_token_sequence_set_sha256",
        "occurrence_set_sha256",
    ):
        input_store[name] = store[name]
    export["occurrence_metadata"]["input_occurrence_set_sha256"] = store[
        "occurrence_set_sha256"
    ]
    frozen_fetch = fetch["frozen_fetch_state"]
    export["input_fetch_state"].update(
        {
            name: frozen_fetch[name]
            for name in (
                "schema",
                "artifact",
                "sqlite_schema_sha256",
                "sqlite_logical_sha256",
                "sidecar_set_sha256",
            )
        }
    )
    export["tokenizer"] = {"contract": tokenizer}
    export_path.write_text(json.dumps(export), encoding="utf-8")

    result = _collect_frozen_ci_case5(root, progress_paths=[progress])

    assert result["ready_valid_tokens"] == 31_739
    assert result["ready_trained_tokens"] == 31_734
    assert (
        result["ready_valid_tokens"]
        == result["export"]["splits"]["train"]["valid_tokens"]
    )
    assert (
        result["ready_valid_tokens"]
        != result["export"]["all_split_counts"]["valid_tokens"]
    )
    assert result["export"]["splits"]["validation"]["valid_tokens"] == 31_734
    zero_counts = {
        "files": 0,
        "rows": 0,
        "bytes": 0,
        "capacity_tokens": 0,
        "valid_tokens": 0,
        "trained_tokens": 0,
    }
    assert set(result["export"]["splits"]) == {"train", "validation", "test"}
    assert all(
        set(counts) == set(zero_counts)
        for counts in result["export"]["splits"].values()
    )
    assert result["export"]["splits"]["test"] == zero_counts
    assert (
        result["export"]["buckets"]["1024"]["splits"]["train"]["valid_tokens"] == 1_023
    )
    assert (
        result["export"]["buckets"]["1024"]["splits"]["validation"]["valid_tokens"]
        == 1_022
    )
    assert set(result["export"]["buckets"]) == {
        "1024",
        "2048",
        "4096",
        "8192",
        "16384",
    }
    for bucket in result["export"]["buckets"].values():
        assert set(bucket["splits"]) == {"train", "validation", "test"}
        assert all(
            set(counts) == set(zero_counts) for counts in bucket["splits"].values()
        )
        assert bucket["splits"]["test"] == zero_counts


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


def test_parquet_snapshot_ignores_atomic_staging_files(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    _write_parquet(root / "1024" / "one.parquet")
    (root / "1024" / ".one.partial.staged.parquet").write_bytes(b"partial")

    result = scan_parquet_snapshot(
        root,
        batch_size=192,
        jobs=1,
        classify_documents=True,
    )

    assert result["files"] == 1
    assert result["valid_tokens"] == 15


def test_parquet_snapshot_retries_atomic_publish_window(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    target = root / "1024" / "one.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"partial")
    replacement = tmp_path / "replacement" / "one.parquet"
    _write_parquet(replacement)

    def publish() -> None:
        time.sleep(0.02)
        os.replace(replacement, target)

    publisher = Thread(target=publish)
    publisher.start()
    result = scan_parquet_snapshot(
        root,
        batch_size=192,
        jobs=1,
        classify_documents=True,
    )
    publisher.join(timeout=1)

    assert not publisher.is_alive()
    assert result["files"] == 1
    assert result["valid_tokens"] == 15


def test_parquet_snapshot_waits_between_failed_retries(tmp_path: Path) -> None:
    target = tmp_path / "packed" / "1024" / "one.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"partial")

    started = time.monotonic()
    with pytest.raises(pa.ArrowException):
        scan_parquet_snapshot(
            tmp_path / "packed",
            batch_size=192,
            jobs=1,
            classify_documents=True,
            snapshot_retries=1,
        )

    assert time.monotonic() - started >= 0.09


def test_live_source_progress_uses_archive_scope_not_mapping_superset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    (root / "reindexed").mkdir(parents=True)
    completion = root / "conveyor" / "_done.json"
    completion.parent.mkdir(parents=True)
    completion.write_text(
        json.dumps(
            {
                "code_revision": {"git_commit": "a" * 40},
                "done": {},
                "failed": {},
                "source_repo_list": {
                    "mapping_count": 2,
                    "sha256": "source-list",
                },
            }
        ),
        encoding="utf-8",
    )
    launch = root / "launch.json"
    launch.write_text(
        json.dumps(
            {
                "schema": "cppmega.canonical_source_launch_v1",
                "expected_repository_count": 1,
                "inputs": {
                    "archive_inventory_receipt": {"repository_count": 1},
                    "repo_list": {"sha256": "source-list"},
                },
                "outputs": {
                    "conveyor_manifest": str(completion),
                    "run_root": str(root),
                },
            }
        ),
        encoding="utf-8",
    )

    result = collect_live_source(
        {
            "root": str(root),
            "completion_receipt": str(completion),
            "launch_receipt": str(launch),
        },
        batch_size=192,
        jobs=1,
    )

    assert result["parquet"]["files"] == 0
    assert result["training_readable"] is False
    assert "source conveyor has not produced Parquet yet" in result["blockers"]
    assert result["conveyor"]["not_terminal"] == 1

    _write_parquet(root / "reindexed" / "1024" / "repo.parquet")
    completion.write_text(
        json.dumps(
            {
                "code_revision": {"git_commit": "a" * 40},
                "done": {
                    "repo::code": {
                        "source": "code",
                        "lengths": {
                            "1024": {
                                "rows": 1,
                                "capacity_tokens": 1024,
                                "valid_tokens": 15,
                            }
                        },
                    }
                },
                "failed": {},
                "source_repo_list": {
                    "mapping_count": 2,
                    "sha256": "source-list",
                },
            }
        ),
        encoding="utf-8",
    )
    result = collect_live_source(
        {
            "root": str(root),
            "completion_receipt": str(completion),
            "launch_receipt": str(launch),
        },
        batch_size=192,
        jobs=1,
    )

    assert result["conveyor"] == {
        "mapping_count": 2,
        "expected_repository_count": 1,
        "done": 1,
        "failed": 0,
        "not_terminal": 0,
    }
    assert "source conveyor is still incomplete" not in result["blockers"]


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
            "github_pr": {
                "state": "verified_store_not_materialized",
                "release_ready": False,
                "version": {"scan_id": "scan", "store_sha256": "store"},
                "records": {"stored_prs": 3},
            },
            "gitlab_mr": {
                "state": "verified_inventory_not_materialized",
                "release_ready": False,
                "blockers": ["exact primary GitLab MR membership is not verified"],
                "version": {"scan_id": "gitlab-scan"},
                "records": {
                    "declared_mrs": 7,
                    "candidate_mrs": 2,
                    "primary_stored_mrs": 0,
                    "ancillary_stored_mrs": 1,
                },
                "stores": {
                    "primary": {"sha256": "primary-store"},
                    "ancillary": {"sha256": "ancillary-store"},
                },
                "sidecars": {"files": 8},
            },
            "ci": {
                "state": "cas_staged_not_exported",
                "release_ready": False,
                "token_accounting": {"store_local_unique_upper_bound": 20},
                "stores": [
                    {
                        "interval": {"start": "a", "end": "b"},
                        "sidecar_set_sha256": "sidecars",
                        "tokenizer": {"tokenizer_contract_sha256": "tokenizer"},
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


def test_publish_status_accepts_live_source_before_first_parquet(
    tmp_path: Path,
) -> None:
    status = _minimal_status(sha="1" * 64, live_tokens=0)
    source = status["datasets"]["live_source"]["parquet"]
    source.update(
        {
            "files": 0,
            "rows": 0,
            "buckets": {},
            "schema": {"counts": {}, "metadata_by_sha256": {}},
        }
    )

    paths = publish_status(status, tmp_path / "status")

    entry = json.loads(paths["changelog"].read_text(encoding="utf-8"))
    assert entry["summary"]["live_source"]["tokenizer_contract_sha256"] is None


def _freshness_config(source_receipt: Path, ci_receipt: Path) -> dict[str, object]:
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


def test_collect_ci_status_reports_missing_progress_receipt(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    _write_parquet(legacy / "1024" / "part.parquet")
    (legacy / "manifest.json").write_text(
        json.dumps({"source_completion": {}, "domain_kind_counts": {}}),
        encoding="utf-8",
    )
    missing = tmp_path / "missing-progress.json"

    result = collect_ci_status(
        {
            "progress_receipts": [str(missing)],
            "legacy_parquet_root": str(legacy),
            "batch_size": 192,
            "jobs": 1,
        }
    )

    assert result["stores"] == []
    assert result["missing_progress_receipts"] == [str(missing)]
    assert "configured CI progress receipt is missing" in result["blockers"]
    assert result["token_accounting"]["store_local_unique_upper_bound"] == 0


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


def test_collect_gitlab_mr_status_keeps_stores_separate_and_unready(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.sqlite"
    ancillary = tmp_path / "ancillary.sqlite"
    manifest = tmp_path / "manifest.json"
    repo_list = tmp_path / "repo_list.json"
    primary.write_bytes(b"primary")
    ancillary.write_bytes(b"ancillary")
    manifest.write_text("{}\n", encoding="utf-8")
    repo_list.write_text("{}\n", encoding="utf-8")
    (tmp_path / "sidecars").mkdir()

    def binding(path: Path, *, include_size: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if include_size:
            result["size"] = path.stat().st_size
        return result

    receipt = {
        "schema": "cppmega_gitlab_mr_completion_v1",
        "status": "verified",
        "platform": "gitlab",
        "scan_id": "scan",
        "contract_sha256": "1" * 64,
        "manifest": binding(manifest, include_size=False),
        "repo_list": binding(repo_list, include_size=False),
        "expected_host_count": 1,
        "expected_hosts": ["gitlab.example"],
        "expected_repo_count": 1,
        "expected_repos": ["gitlab.example/acme%2Frepo"],
        "declared_mr_count": 5,
        "candidate_mr_count": 2,
        "noncandidate_mr_count": 3,
        "route_counts": {
            "primary": 0,
            "ancillary": 1,
            "terminal": 1,
            "excluded": 0,
        },
        "pr_store": binding(primary, include_size=True),
        "stored_pr_count": 0,
        "unverified_store_pr_count": 0,
        "ancillary_store": binding(ancillary, include_size=True),
        "ancillary_stored_count": 1,
        "ancillary_unverified_store_count": 0,
        "sidecars": {
            "root": str(tmp_path / "sidecars"),
            "files": 5,
            "format": "canonical-json-gzip",
            "logical_byte_size": 10,
            "physical_byte_size": 8,
            "logical_set_sha256": "2" * 64,
            "physical_set_sha256": "3" * 64,
        },
        "validation": {
            "candidate_route_conservation": True,
            "deterministic_gzip_sidecars": True,
            "exact_primary_membership_verified": False,
            "immutable_artifact_hashes": True,
            "inventory_complete": True,
            "primary_ancillary_physical_separation": True,
            "store_counts_match_manifest": True,
            "terminal_http_statuses_preserved": True,
        },
        "required_training_gate": "exact_primary_pr_membership_receipt",
    }
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps(receipt), encoding="utf-8")

    result = collect_gitlab_mr_status({"completion_receipt": str(completion)})

    assert result["state"] == "verified_inventory_not_materialized"
    assert result["release_ready"] is False
    assert result["records"]["declared_mrs"] == 5
    assert result["records"]["primary_stored_mrs"] == 0
    assert result["records"]["ancillary_stored_mrs"] == 1
    assert result["stores"]["primary"]["path"] == str(primary)
    assert result["stores"]["ancillary"]["path"] == str(ancillary)
    assert "exact primary GitLab MR membership is not verified" in result["blockers"]

    primary.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size drifted"):
        collect_gitlab_mr_status({"completion_receipt": str(completion)})


def _minimal_config(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": "cppmega_training_data_status_config_v2",
        "batch_size": 192,
        "output_dir": str(tmp_path / "out"),
        "source": {
            "root": str(tmp_path / "source"),
            "completion_receipt": str(tmp_path / "done.json"),
            "launch_receipt": str(tmp_path / "launch.json"),
        },
        "sealed_megatron": {"manifest": str(tmp_path / "sealed.json")},
        "validation_bundle": {"manifest": str(tmp_path / "validation.json")},
        "github_pr": {
            "completion_receipt": str(tmp_path / "pr_completion.json"),
            "gap_completion_receipt": str(tmp_path / "pr_gap.json"),
            "export_launch_receipt": str(tmp_path / "pr_launch.json"),
            "export_cancellation_receipt": str(tmp_path / "pr_cancel.json"),
            "quarantine_receipt": str(tmp_path / "pr_quarantine.json"),
        },
        "gitlab_mr": {"completion_receipt": str(tmp_path / "gitlab_completion.json")},
        "ci": {
            "progress_receipts": [str(tmp_path / "ci_progress.json")],
            "legacy_parquet_root": str(tmp_path / "ci_legacy"),
        },
    }


def _write_minimal_config_files(tmp_path: Path) -> None:
    (tmp_path / "done.json").write_text(
        json.dumps(
            {
                "code_revision": {"git_commit": "a" * 40},
                "done": {},
                "failed": {},
                "source_repo_list": {"mapping_count": 0, "sha256": "source-list"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "launch.json").write_text(
        json.dumps(
            {
                "schema": "cppmega.canonical_source_launch_v1",
                "expected_repository_count": 0,
                "inputs": {"archive_inventory_receipt": {"repository_count": 0}},
                "outputs": {
                    "conveyor_manifest": str(tmp_path / "done.json"),
                    "run_root": str(tmp_path / "source"),
                },
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "pr_completion.json",
        "pr_gap.json",
        "pr_launch.json",
        "pr_cancel.json",
        "pr_quarantine.json",
    ):
        (tmp_path / name).write_text(
            json.dumps({"status": "verified"}), encoding="utf-8"
        )
    (tmp_path / "pr_launch.json").write_text(
        json.dumps({"status": "verified", "output_root": str(tmp_path / "pr_out")}),
        encoding="utf-8",
    )
    (tmp_path / "ci_progress.json").write_text(
        json.dumps(
            {
                "schema": "cppmega.ci_stream_progress_v1",
                "generated_at": _utc_now(),
                "inventory": {},
                "fetch": {"occurrence_tokens": 0},
                "content_store": {"counters": {"exact_unique_payload_tokens": 0}},
                "token_accounting": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "sealed.json").write_text(
        json.dumps({"schema": "cppmega.megatron_bundle_v1", "bundles": []}),
        encoding="utf-8",
    )
    (tmp_path / "validation.json").write_text(
        json.dumps({"schema": "cppmega.megatron_bundle_v1", "bundles": []}),
        encoding="utf-8",
    )


def test_build_status_includes_heartbeat_freshness_when_path_provided(
    monkeypatch, tmp_path: Path
) -> None:
    _write_minimal_config_files(tmp_path)
    config = _minimal_config(tmp_path)
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_live_source",
        lambda _s, **_: {"state": "packed_unsealed", "release_ready": False},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_megatron_bundle",
        lambda _m, **_: {"state": "sealed_megatron", "release_ready": True},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_pr_status",
        lambda _p: {"state": "verified_store_not_materialized"},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_gitlab_mr_status",
        lambda _p: {"state": "verified_inventory_not_materialized"},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_ci_status",
        lambda _c: {"state": "cas_staged_not_exported"},
    )

    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text(
        json.dumps(
            {
                "schema": HEARTBEAT_SCHEMA,
                "pid": 12345,
                "recorded_at": datetime.fromtimestamp(time.time() - 7200)
                .astimezone()
                .isoformat(),
                "status_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    status = build_status(config, jobs=1, stale_minutes=30.0, heartbeat_path=heartbeat)
    assert "heartbeat" in status["freshness"]
    assert status["freshness"]["heartbeat"]["stale"] is True
    assert status["freshness"]["heartbeat"]["pid"] == 12345


def test_build_status_omits_heartbeat_when_path_not_provided(
    monkeypatch, tmp_path: Path
) -> None:
    _write_minimal_config_files(tmp_path)
    config = _minimal_config(tmp_path)
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_live_source",
        lambda _s, **_: {"state": "packed_unsealed", "release_ready": False},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_megatron_bundle",
        lambda _m, **_: {"state": "sealed_megatron", "release_ready": True},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_pr_status",
        lambda _p: {"state": "verified_store_not_materialized"},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_gitlab_mr_status",
        lambda _p: {"state": "verified_inventory_not_materialized"},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_ci_status",
        lambda _c: {"state": "cas_staged_not_exported"},
    )

    status = build_status(config, jobs=1, stale_minutes=30.0)
    assert "heartbeat" not in status["freshness"]


def test_build_status_allows_first_publish_before_heartbeat_exists(
    monkeypatch, tmp_path: Path
) -> None:
    _write_minimal_config_files(tmp_path)
    config = _minimal_config(tmp_path)
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_live_source",
        lambda _s, **_: {"state": "packed_unsealed", "release_ready": False},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_megatron_bundle",
        lambda _m, **_: {"state": "sealed_megatron", "release_ready": True},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_pr_status",
        lambda _p: {"state": "verified_store_not_materialized"},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_gitlab_mr_status",
        lambda _p: {"state": "verified_inventory_not_materialized"},
    )
    monkeypatch.setattr(
        "scripts.report_training_data_status.collect_ci_status",
        lambda _c: {"state": "cas_staged_not_exported"},
    )

    status = build_status(
        config,
        jobs=1,
        stale_minutes=30.0,
        heartbeat_path=tmp_path / "missing-heartbeat.json",
    )

    assert "heartbeat" not in status["freshness"]
