from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_graphql_module():
    pr_ingest_dir = Path(__file__).resolve().parents[1] / "scripts" / "pr_ingest"
    if str(pr_ingest_dir) not in sys.path:
        sys.path.insert(0, str(pr_ingest_dir))
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "pr_ingest"
        / "graphql_pr_stream.py"
    )
    spec = importlib.util.spec_from_file_location("graphql_pr_stream", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}
    text = ""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeStore:
    def __init__(self) -> None:
        self.rows: list[int] = []
        self.cursor_updates: list[tuple[str | None, bool]] = []
        self.commits = 0

    def get_cursor(self, repo: str, stream: str) -> None:
        assert repo == "owner/repo"
        assert stream == "pr"
        return None

    def upsert_pr(self, repo: str, number: int, **_: object) -> None:
        assert repo == "owner/repo"
        self.rows.append(number)

    def set_cursor(
        self,
        repo: str,
        stream: str,
        cursor: str | None,
        page_count: int,
        pr_count: int,
        done: bool,
        updated_at: str,
    ) -> None:
        assert repo == "owner/repo"
        assert stream == "pr"
        assert page_count >= 0
        assert pr_count >= 0
        assert updated_at
        self.cursor_updates.append((cursor, done))

    def commit(self) -> None:
        self.commits += 1

    def count(self, repo: str) -> int:
        assert repo == "owner/repo"
        return len(self.rows)


def _pr_node(number: int) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"PR {number}",
        "body": "",
        "state": "MERGED",
        "createdAt": "2026-01-01T00:00:00Z",
        "mergedAt": "2026-01-01T00:00:01Z",
        "mergeCommit": {"oid": f"sha{number}"},
        "author": {"login": "dev"},
        "comments": {"nodes": []},
        "reviews": {"nodes": []},
    }


def test_max_prs_does_not_advance_cursor_after_partial_page() -> None:
    graphql = _load_graphql_module()
    store = _FakeStore()
    rotator = graphql.TokenRotator(["token"])

    payload = {
        "data": {
            "rateLimit": {"remaining": 4999, "resetAt": "2026-01-01T01:00:00Z"},
            "repository": {
                "pullRequests": {
                    "nodes": [_pr_node(1), _pr_node(2), _pr_node(3)],
                    "pageInfo": {"hasNextPage": True, "endCursor": "cursor-after-page"},
                }
            },
        }
    }

    def fake_post(token: str, query: str, variables: dict[str, object]) -> _FakeResponse:
        assert token == "token"
        assert query
        assert variables == {"owner": "owner", "name": "repo", "after": None}
        return _FakeResponse(payload)

    result = graphql.fetch_repo(
        store,
        rotator,
        "owner",
        "repo",
        max_pages=None,
        max_prs=2,
        comment_cap=20,
        verbose=False,
        graphql_post=fake_post,
    )

    assert result["fetched"] == 2
    assert store.rows == [1, 2]
    assert store.commits == 1
    assert store.cursor_updates == []


def test_max_prs_zero_does_not_issue_graphql_request() -> None:
    graphql = _load_graphql_module()
    store = _FakeStore()
    rotator = graphql.TokenRotator(["token"])

    def fake_post(token: str, query: str, variables: dict[str, object]) -> _FakeResponse:
        raise AssertionError("GraphQL request should not be issued when max_prs=0")

    result = graphql.fetch_repo(
        store,
        rotator,
        "owner",
        "repo",
        max_pages=None,
        max_prs=0,
        comment_cap=20,
        verbose=False,
        graphql_post=fake_post,
    )

    assert result["fetched"] == 0
    assert store.rows == []
    assert store.cursor_updates == []
