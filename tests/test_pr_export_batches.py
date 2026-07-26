from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

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


def test_pr_export_all_batches_writes_manifest_and_shard(tmp_path):
    import export_pr_parquet
    import pr_store

    store = tmp_path / "prs.sqlite"
    conn = pr_store.connect(str(store), create=True)
    try:
        for pr_number in (1, 2):
            pr_store.upsert_record(
                conn,
                {
                    "repo": "owner/repo",
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
                },
            )
    finally:
        conn.close()

    out = tmp_path / "out"
    manifest = out / "_done.json"
    args = argparse.Namespace(
        store=str(store),
        output_root=str(out),
        target_lengths="1024",
        repo=None,
        offset=0,
        limit=10_000,
        all=True,
        batch_size=1,
        max_shards=1,
        manifest=str(manifest),
        no_resume=False,
        memory_limit_gb=4.0,
    )

    result = export_pr_parquet.export_pr_parquet_batches(args)

    assert result["n_shards"] == 1
    assert result["next_offset"] == 1
    shard = out / "1024" / "pr_discussions_all_00000000.parquet"
    assert shard.exists()
    schema = pq.read_schema(shard)
    assert schema.metadata is not None
    assert schema.metadata[
        SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")
    ] == str(SYMBOL_IDENTITY_SCHEMA_VERSION).encode("ascii")
    identities = pq.read_table(shard, columns=[SYMBOL_IDENTITIES_COLUMN]).column(0)
    assert identities.to_pylist() == [[]]
    blob = json.loads(manifest.read_text(encoding="utf-8"))
    assert "all:0" in blob["done"]


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

    out = tmp_path / "out"
    args = argparse.Namespace(
        store=str(store),
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
