from __future__ import annotations

import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest


MLX_ROOT = Path(__file__).resolve().parents[1]
PR_INGEST = MLX_ROOT / "scripts" / "pr_ingest"
if str(PR_INGEST) not in sys.path:
    sys.path.insert(0, str(PR_INGEST))


def test_pr_store_uses_wal_and_supports_page_commits(tmp_path):
    import pr_store

    db = tmp_path / "prs.sqlite"
    first = pr_store.connect(str(db), create=True)
    second = pr_store.connect(str(db), create=True)
    try:
        assert first.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert second.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

        pr_store.upsert_record(
            first,
            {
                "repo": "owner/repo",
                "pr_number": 7,
                "merge_commit_sha": "abc",
                "pr_title": "title",
                "pr_body": "body",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
            commit=False,
        )

        assert pr_store.get_by_pr(second, "owner/repo", 7) is None
        first.commit()
        assert pr_store.get_by_pr(second, "owner/repo", 7)["merge_commit_sha"] == "abc"

        first.execute("BEGIN IMMEDIATE")
        worker = pr_store.connect(
            str(db),
            create=False,
            initialize=False,
        )
        try:
            assert worker.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            worker.close()
            first.rollback()
    finally:
        first.close()
        second.close()


def test_pr_store_backfills_aliases_only_when_legacy_columns_are_added(tmp_path):
    import pr_store

    db = tmp_path / "legacy.sqlite"
    legacy = sqlite3.connect(db)
    legacy.executescript(
        """
        CREATE TABLE prs (
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            pr_title TEXT,
            pr_body TEXT,
            PRIMARY KEY (repo, pr_number)
        );
        INSERT INTO prs (repo, pr_number, pr_title, pr_body)
        VALUES ('owner/repo', 1, 'legacy title', 'legacy body');
        """
    )
    legacy.commit()
    legacy.close()

    migrated = pr_store.connect(str(db), create=True)
    try:
        row = migrated.execute(
            "SELECT title, body FROM prs WHERE repo='owner/repo' AND pr_number=1"
        ).fetchone()
        assert tuple(row) == ("legacy title", "legacy body")
        columns = {
            str(row["name"])
            for row in migrated.execute("PRAGMA table_info(prs)")
        }
        indexes = {
            str(row["name"])
            for row in migrated.execute("PRAGMA index_list(prs)")
        }
        assert "scan_id" in columns
        assert "idx_prs_scan_repo" in indexes
    finally:
        migrated.close()


def test_readonly_store_rejects_legacy_schema_with_clear_migration_error(tmp_path):
    import pr_store

    db = tmp_path / "legacy-readonly.sqlite"
    legacy = sqlite3.connect(db)
    legacy.executescript(
        """
        CREATE TABLE prs (
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            pr_title TEXT,
            pr_body TEXT,
            PRIMARY KEY (repo, pr_number)
        );
        """
    )
    legacy.commit()
    legacy.close()

    with pytest.raises(
        RuntimeError,
        match=r"readonly PR store schema is incompatible.*scan_id",
    ):
        pr_store.connect(str(db), create=False, readonly=True)

    migrated = pr_store.connect(str(db), create=True)
    migrated.close()
    readonly = pr_store.connect(str(db), create=False, readonly=True)
    readonly.close()


class _BoundedReadStringIO(io.StringIO):
    def __init__(self, value: str, *, max_read: int) -> None:
        super().__init__(value)
        self.max_read = max_read
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> str:
        assert 0 < size <= self.max_read
        self.read_sizes.append(size)
        return super().read(size)


def test_prettyjson_array_is_consumed_incrementally_one_row_at_a_time():
    import pr_store

    payload = json.dumps(
        [
            {"id": 1, "payload": "first"},
            {"id": 2, "payload": "x" * 2_000},
        ],
        indent=2,
    )
    handle = _BoundedReadStringIO(payload, max_read=17)
    rows = pr_store._iter_json_array(
        handle,
        "fixture.prettyjson",
        chunk_size=17,
    )

    assert next(rows) == {"id": 1, "payload": "first"}
    assert handle.tell() < len(payload)
    assert next(rows) == {"id": 2, "payload": "x" * 2_000}
    with pytest.raises(StopIteration):
        next(rows)
    assert handle.read_sizes


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ('[{"id":1},]', "trailing comma"),
        ('[{"id":1}] garbage', "trailing content"),
        ('[1]', "row 0 must be a JSON object"),
    ],
)
def test_prettyjson_array_rejects_ambiguous_or_non_object_rows(
    payload: str,
    match: str,
) -> None:
    import pr_store

    with pytest.raises(SystemExit, match=match):
        list(
            pr_store._iter_json_array(
                _BoundedReadStringIO(payload, max_read=3),
                "fixture.prettyjson",
                chunk_size=3,
            )
        )
