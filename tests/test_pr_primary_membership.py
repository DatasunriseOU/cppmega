from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega.data.nanochat_pipeline.packed_rows_schema import (
    NUM_DOCS_COLUMN,
    SOURCE_COMMIT_HASHES_COLUMN,
    SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
    SOURCE_PR_NUMBERS_COLUMN,
)
from cppmega.data.source_conveyor_composition import SourceComposition
from scripts.pr_ingest.pr_store import connect, upsert_record


def _sha(number: int) -> str:
    return f"{number:040x}"


def _composition(
    tmp_path: Path,
    rows: list[dict[str, object]],
) -> tuple[SourceComposition, Path]:
    commit_root = tmp_path / "commits"
    bucket_root = commit_root / "1024"
    bucket_root.mkdir(parents=True)
    parquet = bucket_root / "primary.parquet"
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("repo", pa.string()),
                pa.field(NUM_DOCS_COLUMN, pa.int32()),
                pa.field(SOURCE_PR_NUMBERS_COLUMN, pa.list_(pa.int64())),
                pa.field(SOURCE_COMMIT_HASHES_COLUMN, pa.list_(pa.string())),
                pa.field(
                    SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
                    pa.list_(pa.bool_()),
                ),
            ]
        ),
    )
    pq.write_table(table, parquet, compression="zstd")
    composition = SourceComposition(
        allowlist={
            ("code", 1024): {"unused.parquet": 1},
            ("commits", 1024): {parquet.name: table.num_rows},
        },
        receipt={
            "schema": "cppmega_source_conveyor_composition_v1",
            "status": "complete",
            "plan_sha256": "a" * 64,
        },
        plan_path=tmp_path / "plan.json",
        dedup_receipt_path=tmp_path / "dedup.json",
        run_files=(),
    )
    return composition, commit_root


def _store(tmp_path: Path) -> tuple[Path, str]:
    store = tmp_path / "prs.sqlite"
    scan_id = "9" * 64
    conn = connect(str(store), create=True)
    try:
        for number in (1, 2, 3):
            upsert_record(
                conn,
                {
                    "repo": "owner/repo",
                    "pr_number": number,
                    "merge_commit_sha": _sha(number),
                    "pr_title": f"PR {number}",
                    "pr_body": "body",
                    "comments": [],
                    "reviews": [],
                    "linked_issues": [],
                },
                scan_id=scan_id,
            )
    finally:
        conn.close()
    return store, scan_id


def test_primary_membership_keeps_only_commit_attached_prs(tmp_path: Path) -> None:
    from cppmega.data.pr_primary_membership import (
        build_primary_pr_membership,
        iter_primary_pr_keys,
    )

    store, scan_id = _store(tmp_path)
    composition, commit_root = _composition(
        tmp_path,
        [
            {
                "repo": "owner/repo",
                NUM_DOCS_COLUMN: 2,
                SOURCE_PR_NUMBERS_COLUMN: [1, None],
                SOURCE_COMMIT_HASHES_COLUMN: [_sha(1), _sha(2)],
                SOURCE_HAS_PR_DISCUSSIONS_COLUMN: [True, False],
            },
            {
                "repo": "owner/repo",
                NUM_DOCS_COLUMN: 1,
                SOURCE_PR_NUMBERS_COLUMN: [3],
                SOURCE_COMMIT_HASHES_COLUMN: [_sha(999)],
                SOURCE_HAS_PR_DISCUSSIONS_COLUMN: [False],
            },
        ],
    )
    conn = connect(str(store), create=False, readonly=True)
    try:
        receipt = build_primary_pr_membership(
            conn,
            source_composition=composition,
            commit_root=commit_root,
            buckets=(1024,),
            scan_id=scan_id,
        )
        keys = list(
            iter_primary_pr_keys(
                conn,
                repo=None,
                scan_id=scan_id,
                offset=0,
                limit=None,
            )
        )
    finally:
        conn.close()

    assert [(row["repo"], row["pr_number"]) for row in keys] == [
        ("owner/repo", 1),
        ("owner/repo", 2),
    ]
    assert receipt["selected_pr_count"] == 2
    assert receipt["source_docs"] == 3
    assert receipt["unmatched_commit_sha_source_docs"] == 1
    assert receipt["ignored_unverified_pr_number_source_docs"] == 1
    assert receipt["validation"]["exact_source_doc_shapes"] is True
    assert receipt["validation"]["exact_scan_membership"] is True


def test_primary_membership_rejects_per_document_shape_drift(
    tmp_path: Path,
) -> None:
    from cppmega.data.pr_primary_membership import build_primary_pr_membership

    store, scan_id = _store(tmp_path)
    composition, commit_root = _composition(
        tmp_path,
        [
            {
                "repo": "owner/repo",
                NUM_DOCS_COLUMN: 2,
                SOURCE_PR_NUMBERS_COLUMN: [1],
                SOURCE_COMMIT_HASHES_COLUMN: [_sha(1), _sha(2)],
                SOURCE_HAS_PR_DISCUSSIONS_COLUMN: [True, False],
            }
        ],
    )
    conn = connect(str(store), create=False, readonly=True)
    try:
        with pytest.raises(ValueError, match="per-document provenance shape"):
            build_primary_pr_membership(
                conn,
                source_composition=composition,
                commit_root=commit_root,
                buckets=(1024,),
                scan_id=scan_id,
            )
    finally:
        conn.close()


def test_primary_membership_binding_rejects_commit_artifact_mutation(
    tmp_path: Path,
) -> None:
    from cppmega.data.pr_primary_membership import (
        build_primary_pr_membership,
        verify_primary_pr_membership_binding,
    )

    store, scan_id = _store(tmp_path)
    composition, commit_root = _composition(
        tmp_path,
        [
            {
                "repo": "owner/repo",
                NUM_DOCS_COLUMN: 1,
                SOURCE_PR_NUMBERS_COLUMN: [1],
                SOURCE_COMMIT_HASHES_COLUMN: [_sha(1)],
                SOURCE_HAS_PR_DISCUSSIONS_COLUMN: [True],
            }
        ],
    )
    conn = connect(str(store), create=False, readonly=True)
    try:
        receipt = build_primary_pr_membership(
            conn,
            source_composition=composition,
            commit_root=commit_root,
            buckets=(1024,),
            scan_id=scan_id,
        )
    finally:
        conn.close()

    parquet = commit_root / "1024" / "primary.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"mutation")
    with pytest.raises(RuntimeError, match="commit artifact binding drifted"):
        verify_primary_pr_membership_binding(
            receipt,
            source_composition=composition,
            commit_root=commit_root,
            buckets=(1024,),
        )
