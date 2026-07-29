from __future__ import annotations

from cppmega.data.nanochat_pipeline.packed_rows_schema import (
    SOURCE_COMMIT_HASHES_COLUMN,
)
from scripts.nanochat_data.pack_enriched_rows import (
    normalize_document_record,
    pack_documents,
    rows_to_table,
)


def _commit_doc(
    *,
    source_doc_index: int,
    commit_hash: str,
    file_local_commit_index: int,
):
    return normalize_document_record(
        {
            "token_ids": [source_doc_index + 1, source_doc_index + 2],
            "repo": "owner/repo",
            "filepath": "tests/parser.cpp",
            "commit_hash": commit_hash,
            "file_local_commit_index": file_local_commit_index,
            "doc_type": "code",
        },
        source_doc_index=source_doc_index,
    )


def test_packed_rows_preserve_exact_commit_hash_per_source_document() -> None:
    first_sha = "1" * 40
    second_sha = "2" * 40
    rows, overflow = pack_documents(
        [
            _commit_doc(
                source_doc_index=0,
                commit_hash=first_sha,
                file_local_commit_index=0,
            ),
            _commit_doc(
                source_doc_index=1,
                commit_hash=second_sha,
                file_local_commit_index=1,
            ),
        ],
        target_length=16,
        strategy="sequential",
    )

    assert overflow == []
    assert len(rows) == 1
    assert rows[0][SOURCE_COMMIT_HASHES_COLUMN] == [first_sha, second_sha]
    table = rows_to_table(rows)
    assert table.column(SOURCE_COMMIT_HASHES_COLUMN).to_pylist() == [
        [first_sha, second_sha]
    ]
