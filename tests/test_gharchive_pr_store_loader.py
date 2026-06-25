from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PR_INGEST_DIR = _REPO_ROOT / "scripts" / "pr_ingest"


def _load_loader():
    if str(_PR_INGEST_DIR) not in sys.path:
        sys.path.insert(0, str(_PR_INGEST_DIR))
    module_path = _PR_INGEST_DIR / "gharchive_load_pr_store.py"
    spec = importlib.util.spec_from_file_location("gharchive_load_pr_store", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gharchive_loader_materializes_raw_pull_request_event_into_pr_store(
    tmp_path: Path,
) -> None:
    loader = _load_loader()
    from pr_store import PRStore

    events_path = tmp_path / "events.json"
    events_path.write_text(
        json.dumps(
            [
                {
                    "type": "PullRequestEvent",
                    "repo_name": "owner/repo",
                    "actor_login": "alice",
                    "created_at": "2026-01-02 03:04:05 UTC",
                    "id": "event-1",
                    "payload": json.dumps(
                        {
                            "action": "closed",
                            "pull_request": {
                                "number": 42,
                                "title": "Fix parser",
                                "body": "details",
                                "state": "closed",
                                "user": {"login": "alice"},
                                "created_at": "2026-01-01T00:00:00Z",
                                "merged_at": "2026-01-02T03:04:00Z",
                                "merge_commit_sha": "abc123",
                            },
                        }
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )

    db_path = tmp_path / "prs.sqlite"
    count = loader.load_gharchive_events(events_path, db_path)

    assert count == 1
    with PRStore(str(db_path)) as store:
        row = store.get_by_number("owner/repo", 42)
        assert row is not None
        assert row["title"] == "Fix parser"
        assert row["merge_commit_sha"] == "abc123"
        assert store.get_by_sha("owner/repo", "abc123")["pr_number"] == 42
