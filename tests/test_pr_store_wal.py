from __future__ import annotations

import sys
from pathlib import Path


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
    finally:
        first.close()
        second.close()
