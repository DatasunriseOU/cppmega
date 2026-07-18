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
