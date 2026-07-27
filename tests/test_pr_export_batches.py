from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from cppmega.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
)
from scripts.nanochat_data.token_budget import count_tokens, load_tokenizer


MLX_ROOT = Path(__file__).resolve().parents[1]
PR_INGEST = MLX_ROOT / "scripts" / "pr_ingest"
if str(PR_INGEST) not in sys.path:
    sys.path.insert(0, str(PR_INGEST))


def _verified_pr_inputs(
    tmp_path: Path,
    records: list[dict],
    *,
    stale_records: list[dict] | None = None,
) -> tuple[Path, Path, Path, str]:
    import pr_store
    from scripts.pr_ingest.graphql_pr_stream import (
        GRAPHQL_MANIFEST_SCHEMA,
        GRAPHQL_QUERY_CONTRACT_SHA256,
    )
    from scripts.pr_ingest.verify_pr_completion import verify_pr_completion

    scan_id = "1" * 64
    store = tmp_path / "prs.sqlite"
    conn = pr_store.connect(str(store), create=True)
    try:
        for record in records:
            pr_store.upsert_record(conn, record, scan_id=scan_id)
        for record in stale_records or []:
            pr_store.upsert_record(conn, record, scan_id="2" * 64)
    finally:
        conn.close()

    repo_counts: dict[str, int] = {}
    for record in records:
        repo = str(record["repo"])
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
    repo_list = tmp_path / "repo_list.json"
    repo_list.write_text(
        json.dumps(
            {
                "repos": [
                    {"owner_repo": repo}
                    for repo in sorted(repo_counts)
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "graphql_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": GRAPHQL_MANIFEST_SCHEMA,
                "query_contract_sha256": GRAPHQL_QUERY_CONTRACT_SHA256,
                "scan_id": scan_id,
                "repos": {
                    repo: {
                        "status": "done",
                        "cursor": None,
                        "prs": count,
                        "initial_total_count": count,
                        "total_count": count,
                        "source_growth_count": 0,
                        "truncated": 0,
                    }
                    for repo, count in sorted(repo_counts.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "pr_completion.json"
    verify_pr_completion(
        repo_list_path=repo_list,
        graphql_manifest_path=manifest,
        store_path=store,
        output_path=receipt,
    )
    return store, repo_list, receipt, scan_id


def _record(pr_number: int, *, repo: str = "owner/repo") -> dict:
    return {
        "repo": repo,
        "pr_number": pr_number,
        "merge_commit_sha": f"sha{pr_number}",
        "pr_title": f"title {pr_number}",
        "pr_body": "body",
        "comments": [
            {
                "user": "alice",
                "body": f"comment {pr_number}",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
        "reviews": [],
        "linked_issues": [],
    }


def test_pr_export_all_batches_writes_manifest_and_shard(tmp_path):
    import export_pr_parquet

    store, repo_list, receipt, scan_id = _verified_pr_inputs(
        tmp_path,
        [_record(1), _record(2)],
        stale_records=[_record(3, repo="stale/repo")],
    )

    out = tmp_path / "out"
    manifest = out / "_done.json"
    args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
        output_root=str(out),
        target_lengths="1024",
        repo=None,
        offset=0,
        limit=10_000,
        all=True,
        batch_size=1,
        max_shards=None,
        manifest=str(manifest),
        no_resume=False,
        memory_limit_gb=4.0,
    )

    result = export_pr_parquet.export_pr_parquet_batches(args)

    assert result["n_shards"] == 2
    assert result["next_offset"] == 2
    assert result["selected_pr_count"] == 2
    assert result["scan_id"] == scan_id
    shard = (
        out
        / "1024"
        / f"pr_discussions_all_{scan_id[:12]}_00000000.parquet"
    )
    assert shard.exists()
    schema = pq.read_schema(shard)
    assert schema.metadata is not None
    assert schema.metadata[
        SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")
    ] == str(SYMBOL_IDENTITY_SCHEMA_VERSION).encode("ascii")
    identities = pq.read_table(shard, columns=[SYMBOL_IDENTITIES_COLUMN]).column(0)
    assert identities.to_pylist() == [[]]
    blob = json.loads(manifest.read_text(encoding="utf-8"))
    assert blob["schema"] == export_pr_parquet.EXPORT_MANIFEST_SCHEMA
    assert blob["status"] == "complete"
    assert blob["input"]["pr_completion"]["scan_id"] == scan_id
    assert "all:0" in blob["done"]
    receipt_blob = json.loads(
        (out / "export_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt_blob["schema"] == export_pr_parquet.EXPORT_RECEIPT_SCHEMA
    assert receipt_blob["selected_pr_count"] == 2
    assert receipt_blob["rendered_docs"] == 2


def test_pr_export_partial_runs_cannot_publish_global_receipt(tmp_path):
    import export_pr_parquet

    store, repo_list, receipt, _scan_id = _verified_pr_inputs(
        tmp_path / "inputs",
        [_record(1), _record(2)],
    )

    subset_out = tmp_path / "subset"
    subset_args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
        output_root=str(subset_out),
        target_lengths="1024",
        repo=None,
        offset=1,
        limit=10_000,
        all=True,
        batch_size=1,
        max_shards=None,
        manifest=None,
        no_resume=False,
        memory_limit_gb=4.0,
    )
    subset_result = export_pr_parquet.export_pr_parquet_batches(subset_args)

    assert "completion_receipt" not in subset_result
    assert not (subset_out / "export_receipt.json").exists()
    subset_manifest = json.loads(
        (subset_out / "_done.json").read_text(encoding="utf-8")
    )
    assert subset_manifest["status"] == "selection_complete"
    assert subset_manifest["completed_pr_count"] == 1

    bounded_out = tmp_path / "bounded"
    bounded_args = argparse.Namespace(
        **{
            **vars(subset_args),
            "output_root": str(bounded_out),
            "offset": 0,
            "max_shards": 1,
        }
    )
    bounded_result = export_pr_parquet.export_pr_parquet_batches(bounded_args)

    assert "completion_receipt" not in bounded_result
    assert not (bounded_out / "export_receipt.json").exists()
    bounded_manifest = json.loads(
        (bounded_out / "_done.json").read_text(encoding="utf-8")
    )
    assert "status" not in bounded_manifest
    assert "completed_pr_count" not in bounded_manifest


def test_pr_export_losslessly_splits_discussion_larger_than_16k(tmp_path):
    import export_pr_parquet
    import pr_store

    body = "\n".join(
        (
            f"Review finding {index}: rename parser_state_{index}, preserve "
            f"diagnostic_{index}, and add regression_case_{index}."
        )
        for index in range(5_000)
    )
    store = tmp_path / "prs.sqlite"
    conn = pr_store.connect(str(store), create=True)
    try:
        pr_store.upsert_record(
            conn,
            {
                "repo": "owner/repo",
                "pr_number": 99,
                "merge_commit_sha": "sha99",
                "pr_title": "Large parser review",
                "pr_body": body,
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
        )
        record = pr_store.get_by_pr(conn, "owner/repo", 99)
        assert record is not None
        tokenizer = load_tokenizer(
            str(MLX_ROOT / "cppmega" / "tokenizer" / "tokenizer.json")
        )
        assert count_tokens(
            export_pr_parquet._render_training_doc(record), tokenizer
        ) > 16_384
    finally:
        conn.close()
    store, repo_list, receipt, _scan_id = _verified_pr_inputs(
        tmp_path / "verified",
        [
            {
                "repo": "owner/repo",
                "pr_number": 99,
                "merge_commit_sha": "sha99",
                "pr_title": "Large parser review",
                "pr_body": body,
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            }
        ],
    )

    out = tmp_path / "out"
    args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
        output_root=str(out),
        target_lengths="1024,2048,4096,8192,16384",
        repo=None,
        offset=0,
        limit=1,
        all=False,
        batch_size=10_000,
        max_shards=None,
        manifest=None,
        no_resume=False,
        memory_limit_gb=4.0,
    )

    result = export_pr_parquet.export_pr_parquet(args)

    stats = result["materialize_stats"]
    assert stats["docs_in"] == 1
    assert stats["split_input_docs"] == 1
    assert stats["docs_out"] > 1
    assert stats["dropped_input_docs"] == 0
    assert stats["max_materialized_tokens"] <= 16_384
    assert sum(item["rows"] for item in result["lengths"].values()) > 1
    assert "16384" in result["lengths"]


def test_pr_export_refuses_unverified_store(tmp_path):
    import export_pr_parquet
    import pr_store

    store = tmp_path / "prs.sqlite"
    conn = pr_store.connect(str(store), create=True)
    try:
        pr_store.upsert_record(conn, _record(1), scan_id="1" * 64)
    finally:
        conn.close()
    args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(tmp_path / "missing.json"),
        repo_list=str(tmp_path / "missing-repos.json"),
        output_root=str(tmp_path / "out"),
        target_lengths="1024",
        repo=None,
        offset=0,
        limit=1,
        all=False,
        batch_size=10_000,
        max_shards=None,
        manifest=None,
        no_resume=False,
        memory_limit_gb=4.0,
    )

    with pytest.raises(Exception, match="PR completion receipt is missing"):
        export_pr_parquet.export_pr_parquet(args)
