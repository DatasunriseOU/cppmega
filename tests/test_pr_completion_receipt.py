from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pr_ingest import pr_store
from scripts.pr_ingest.github_graphql_fallback import (
    TokenPool,
    _persist_gap_progress,
    build_gap_completion_receipt,
    fetch_pr,
    load_gap_resume_state,
)
from scripts.pr_ingest.graphql_pr_stream import (
    GRAPHQL_MANIFEST_SCHEMA,
    GRAPHQL_QUERY_CONTRACT_SHA256,
)
from scripts.pr_ingest.verify_pr_completion import (
    PRCompletionError,
    canonical_target_set_sha256,
    verify_pr_completion,
)

SCAN_ID = "1" * 64


def _write_repo_list(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "repos": [
                    {
                        "project_identity": "owner/one",
                        "owner_repo": "owner/one",
                    },
                    {
                        "project_identity": "owner/empty",
                        "owner_repo": "owner/empty",
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    *,
    first_status: str = "done",
    first_total: int = 1,
    first_truncated: int = 0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": GRAPHQL_MANIFEST_SCHEMA,
                "query_contract_sha256": GRAPHQL_QUERY_CONTRACT_SHA256,
                "scan_id": SCAN_ID,
                "repos": {
                    "owner/one": {
                        "status": first_status,
                        "cursor": None,
                        "initial_total_count": first_total,
                        "total_count": first_total,
                        "source_growth_count": 0,
                        "truncated": first_truncated,
                    },
                    "owner/empty": {
                        "status": "done",
                        "cursor": None,
                        "initial_total_count": 0,
                        "total_count": 0,
                        "source_growth_count": 0,
                        "truncated": 0,
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_store(path: Path) -> None:
    conn = pr_store.connect(str(path), create=True)
    try:
        pr_store.upsert_record(
            conn,
            {
                "repo": "owner/one",
                "pr_number": 7,
                "pr_title": "Complete",
                "pr_body": "Body",
                "merge_commit_sha": "abc7",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
            scan_id=SCAN_ID,
        )
    finally:
        conn.close()


def test_verified_pr_completion_binds_repo_manifest_and_store(tmp_path: Path) -> None:
    repo_list = tmp_path / "repo_list.json"
    manifest = tmp_path / "graphql_manifest.json"
    store = tmp_path / "prs.sqlite"
    output = tmp_path / "completion.json"
    _write_repo_list(repo_list)
    _write_manifest(manifest)
    _write_store(store)

    receipt = verify_pr_completion(
        repo_list_path=repo_list,
        graphql_manifest_path=manifest,
        store_path=store,
        output_path=output,
    )

    assert receipt["schema"] == "cppmega_pr_completion_v2"
    assert receipt["status"] == "verified"
    assert receipt["expected_repo_count"] == 2
    assert receipt["stored_pr_count"] == 1
    assert receipt["declared_pr_count"] == 1
    assert receipt["source_growth_during_scan"] == 0
    assert receipt["scan_id"] == SCAN_ID
    assert receipt["unverified_store_pr_count"] == 0
    assert receipt["repo_list"]["sha256"]
    assert receipt["graphql_manifest"]["sha256"]
    assert receipt["pr_store"]["sha256"]
    assert json.loads(output.read_text(encoding="utf-8")) == receipt


def test_pr_completion_rejects_fallback_or_in_progress_repo(tmp_path: Path) -> None:
    repo_list = tmp_path / "repo_list.json"
    manifest = tmp_path / "graphql_manifest.json"
    store = tmp_path / "prs.sqlite"
    _write_repo_list(repo_list)
    _write_manifest(manifest, first_status="fallback")
    _write_store(store)

    with pytest.raises(PRCompletionError, match="not terminal GraphQL-done"):
        verify_pr_completion(
            repo_list_path=repo_list,
            graphql_manifest_path=manifest,
            store_path=store,
        )


def test_pr_completion_rejects_store_count_drift(tmp_path: Path) -> None:
    repo_list = tmp_path / "repo_list.json"
    manifest = tmp_path / "graphql_manifest.json"
    store = tmp_path / "prs.sqlite"
    _write_repo_list(repo_list)
    _write_manifest(manifest, first_total=2)
    _write_store(store)

    with pytest.raises(PRCompletionError, match="stored PR count mismatch"):
        verify_pr_completion(
            repo_list_path=repo_list,
            graphql_manifest_path=manifest,
            store_path=store,
        )


def test_pr_completion_binds_monotonic_source_growth(tmp_path: Path) -> None:
    repo_list = tmp_path / "repo_list.json"
    manifest = tmp_path / "graphql_manifest.json"
    store = tmp_path / "prs.sqlite"
    _write_repo_list(repo_list)
    _write_manifest(manifest)
    _write_store(store)

    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["repos"]["owner/one"]["initial_total_count"] = 0
    value["repos"]["owner/one"]["source_growth_count"] = 1
    manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")

    receipt = verify_pr_completion(
        repo_list_path=repo_list,
        graphql_manifest_path=manifest,
        store_path=store,
    )
    assert receipt["source_growth_during_scan"] == 1

    value["repos"]["owner/one"]["source_growth_count"] = 0
    manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(PRCompletionError, match="source_growth_count"):
        verify_pr_completion(
            repo_list_path=repo_list,
            graphql_manifest_path=manifest,
            store_path=store,
        )


def test_pr_completion_does_not_let_stale_rows_mask_missing_scan_membership(
    tmp_path: Path,
) -> None:
    repo_list = tmp_path / "repo_list.json"
    manifest = tmp_path / "graphql_manifest.json"
    store = tmp_path / "prs.sqlite"
    _write_repo_list(repo_list)
    _write_manifest(manifest)

    conn = pr_store.connect(str(store), create=True)
    try:
        pr_store.upsert_record(
            conn,
            {
                "repo": "owner/one",
                "pr_number": 999,
                "pr_title": "stale row from an older scan",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
        )
    finally:
        conn.close()

    with pytest.raises(PRCompletionError, match="stored PR count mismatch"):
        verify_pr_completion(
            repo_list_path=repo_list,
            graphql_manifest_path=manifest,
            store_path=store,
        )


def test_pr_completion_ignores_stale_rows_outside_exact_scan(tmp_path: Path) -> None:
    repo_list = tmp_path / "repo_list.json"
    manifest = tmp_path / "graphql_manifest.json"
    store = tmp_path / "prs.sqlite"
    _write_repo_list(repo_list)
    _write_manifest(manifest)
    _write_store(store)

    conn = pr_store.connect(str(store), create=False)
    try:
        pr_store.upsert_record(
            conn,
            {
                "repo": "owner/one",
                "pr_number": 999,
                "pr_title": "stale row from an older scan",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
        )
    finally:
        conn.close()

    receipt = verify_pr_completion(
        repo_list_path=repo_list,
        graphql_manifest_path=manifest,
        store_path=store,
    )

    assert receipt["stored_pr_count"] == 1
    assert receipt["unverified_store_pr_count"] == 1


def test_truncated_prs_require_exact_gap_completion_receipt(tmp_path: Path) -> None:
    repo_list = tmp_path / "repo_list.json"
    manifest = tmp_path / "graphql_manifest.json"
    store = tmp_path / "prs.sqlite"
    targets = tmp_path / "truncated.jsonl"
    gap = tmp_path / "gap_completion.json"
    _write_repo_list(repo_list)
    _write_manifest(manifest, first_truncated=1)
    _write_store(store)
    targets.write_text(
        json.dumps({"repo": "owner/one", "pr_number": 7}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PRCompletionError, match="gap completion receipt is required"):
        verify_pr_completion(
            repo_list_path=repo_list,
            graphql_manifest_path=manifest,
            store_path=store,
            truncated_targets_path=targets,
        )

    conn = pr_store.connect(str(store), create=False)
    try:
        stored = pr_store.get_by_pr(conn, "owner/one", 7)
        assert stored is not None
        record_sha256 = pr_store.record_content_sha256(stored)
    finally:
        conn.close()

    gap.write_text(
        json.dumps(
            {
                "schema": "cppmega_pr_gap_completion_v1",
                "status": "verified",
                "targets_sha256": canonical_target_set_sha256(
                    (("owner/one", 7),)
                ),
                "target_count": 1,
                "completed_count": 1,
                "miss_count": 0,
                "completed": [
                    {
                        "repo": "owner/one",
                        "pr_number": 7,
                        "record_sha256": record_sha256,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = verify_pr_completion(
        repo_list_path=repo_list,
        graphql_manifest_path=manifest,
        store_path=store,
        truncated_targets_path=targets,
        gap_completion_path=gap,
    )
    assert receipt["truncated_target_count"] == 1
    assert receipt["gap_completion"]["sha256"]


def test_gap_completion_rejects_store_content_drift(tmp_path: Path) -> None:
    repo_list = tmp_path / "repo_list.json"
    manifest = tmp_path / "graphql_manifest.json"
    store = tmp_path / "prs.sqlite"
    targets = tmp_path / "truncated.jsonl"
    gap = tmp_path / "gap_completion.json"
    _write_repo_list(repo_list)
    _write_manifest(manifest, first_truncated=1)
    _write_store(store)
    targets.write_text(
        json.dumps({"repo": "owner/one", "pr_number": 7}) + "\n",
        encoding="utf-8",
    )

    conn = pr_store.connect(str(store), create=False)
    try:
        stored = pr_store.get_by_pr(conn, "owner/one", 7)
        assert stored is not None
        original_sha256 = pr_store.record_content_sha256(stored)
        pr_store.upsert_record(
            conn,
            {
                "repo": "owner/one",
                "pr_number": 7,
                "comments": [
                    {
                        "user": "late",
                        "body": "changed after receipt",
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ],
            },
        )
    finally:
        conn.close()

    gap.write_text(
        json.dumps(
            {
                "schema": "cppmega_pr_gap_completion_v1",
                "status": "verified",
                "targets_sha256": canonical_target_set_sha256(
                    (("owner/one", 7),)
                ),
                "target_count": 1,
                "completed_count": 1,
                "miss_count": 0,
                "completed": [
                    {
                        "repo": "owner/one",
                        "pr_number": 7,
                        "record_sha256": original_sha256,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PRCompletionError, match="content hash mismatch"):
        verify_pr_completion(
            repo_list_path=repo_list,
            graphql_manifest_path=manifest,
            store_path=store,
            truncated_targets_path=targets,
            gap_completion_path=gap,
        )


def test_authoritative_gap_upsert_replaces_rows_and_preserves_duplicates(
    tmp_path: Path,
) -> None:
    store = tmp_path / "prs.sqlite"
    conn = pr_store.connect(str(store), create=True)
    try:
        base = {
            "repo": "owner/one",
            "pr_number": 7,
            "pr_title": "PR",
            "pr_body": "Body",
            "comments": [
                {
                    "user": "old",
                    "body": "truncated first page",
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
            "reviews": [],
            "linked_issues": [],
        }
        pr_store.upsert_record(conn, base)
        complete = {
            **base,
            "comments": [
                {
                    "user": "new",
                    "body": "authoritative full page",
                    "created_at": "2026-01-02T00:00:00Z",
                },
                {
                    "user": "new",
                    "body": "authoritative full page",
                    "created_at": "2026-01-02T00:00:00Z",
                },
            ],
        }
        pr_store.upsert_record(conn, complete, replace_children=True)
        stored = pr_store.get_by_pr(conn, "owner/one", 7)
        assert stored is not None
        assert [item["body"] for item in stored["comments"]] == [
            "authoritative full page",
            "authoritative full page",
        ]
        assert pr_store.record_content_sha256(stored) == (
            pr_store.record_content_sha256(complete)
        )
    finally:
        conn.close()


def test_gap_completion_receipt_is_verified_only_for_exact_fetched_set() -> None:
    targets = (("owner/one", 7), ("owner/two", 9))
    verified = build_gap_completion_receipt(
        targets=targets,
        completed=targets,
        completed_record_sha256s={
            ("owner/one", 7): "a" * 64,
            ("owner/two", 9): "b" * 64,
        },
        misses=(),
        skipped=(),
    )
    assert verified["status"] == "verified"
    assert verified["target_count"] == 2
    assert verified["completed_count"] == 2
    assert verified["miss_count"] == 0

    incomplete = build_gap_completion_receipt(
        targets=targets,
        completed=(("owner/one", 7),),
        completed_record_sha256s={("owner/one", 7): "a" * 64},
        misses=(("owner/two", 9),),
        skipped=(),
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["miss_count"] == 1


def test_gap_completion_resume_trusts_only_receipt_bound_store_hashes(
    tmp_path: Path,
) -> None:
    store = tmp_path / "prs.sqlite"
    receipt_path = tmp_path / "gap_completion.json"
    misses_path = tmp_path / "misses.jsonl"
    targets = (("owner/one", 7), ("owner/two", 9))
    _write_store(store)

    conn = pr_store.connect(str(store), create=False)
    try:
        stored = pr_store.get_by_pr(conn, "owner/one", 7)
        assert stored is not None
        record_sha256 = pr_store.record_content_sha256(stored)
        partial = _persist_gap_progress(
            completion_receipt=str(receipt_path),
            misses_path=str(misses_path),
            targets=targets,
            completed=[("owner/one", 7)],
            completed_record_sha256s={
                ("owner/one", 7): record_sha256,
            },
            misses=[],
            skipped=[],
        )
        assert partial["status"] == "incomplete"
    finally:
        conn.close()

    resumed_conn = pr_store.connect(str(store), create=False)
    try:
        completed, hashes = load_gap_resume_state(
            completion_receipt=str(receipt_path),
            targets=targets,
            conn=resumed_conn,
        )
    finally:
        resumed_conn.close()

    assert completed == [("owner/one", 7)]
    assert hashes == {("owner/one", 7): record_sha256}
    assert misses_path.read_text(encoding="utf-8") == ""


def test_gap_completion_resume_rejects_target_or_store_drift(
    tmp_path: Path,
) -> None:
    store = tmp_path / "prs.sqlite"
    receipt_path = tmp_path / "gap_completion.json"
    misses_path = tmp_path / "misses.jsonl"
    targets = (("owner/one", 7),)
    _write_store(store)

    conn = pr_store.connect(str(store), create=False)
    try:
        stored = pr_store.get_by_pr(conn, "owner/one", 7)
        assert stored is not None
        record_sha256 = pr_store.record_content_sha256(stored)
        _persist_gap_progress(
            completion_receipt=str(receipt_path),
            misses_path=str(misses_path),
            targets=targets,
            completed=[("owner/one", 7)],
            completed_record_sha256s={
                ("owner/one", 7): record_sha256,
            },
            misses=[],
            skipped=[],
        )

        with pytest.raises(RuntimeError, match="different target set"):
            load_gap_resume_state(
                completion_receipt=str(receipt_path),
                targets=(("owner/two", 9),),
                conn=conn,
            )

        pr_store.upsert_record(
            conn,
            {
                "repo": "owner/one",
                "pr_number": 7,
                "pr_title": "changed after checkpoint",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
            replace_children=True,
        )
        with pytest.raises(RuntimeError, match="receipt/store content hash mismatch"):
            load_gap_resume_state(
                completion_receipt=str(receipt_path),
                targets=targets,
                conn=conn,
            )
    finally:
        conn.close()


def test_gap_fetch_paginates_nested_review_thread_comments() -> None:
    def response(_token: str, _variables: dict):
        return (
            200,
            {},
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "number": 7,
                            "title": "PR",
                            "body": "body",
                            "mergeCommit": {"oid": "abc"},
                            "comments": {
                                "totalCount": 0,
                                "pageInfo": {
                                    "hasNextPage": False,
                                    "endCursor": None,
                                },
                                "nodes": [],
                            },
                            "reviews": {
                                "totalCount": 0,
                                "pageInfo": {
                                    "hasNextPage": False,
                                    "endCursor": None,
                                },
                                "nodes": [],
                            },
                            "reviewThreads": {
                                "totalCount": 1,
                                "pageInfo": {
                                    "hasNextPage": False,
                                    "endCursor": None,
                                },
                                "nodes": [
                                    {
                                        "id": "thread-1",
                                        "comments": {
                                            "totalCount": 101,
                                            "pageInfo": {
                                                "hasNextPage": True,
                                                "endCursor": "cursor",
                                            },
                                            "nodes": [
                                                {
                                                    "id": f"comment-{index}",
                                                    "author": {"login": "reviewer"},
                                                    "body": f"comment {index}",
                                                    "path": "src/file.cc",
                                                    "createdAt": (
                                                        "2026-01-01T00:00:00Z"
                                                    ),
                                                }
                                                for index in range(100)
                                            ],
                                        },
                                    }
                                ],
                            },
                            "closingIssuesReferences": {
                                "totalCount": 0,
                                "nodes": [],
                            },
                        }
                    }
                }
            },
        )

    thread_calls: list[dict] = []

    def thread_response(_token: str, variables: dict):
        thread_calls.append(dict(variables))
        return (
            200,
            {},
            {
                "data": {
                    "node": {
                        "id": "thread-1",
                        "comments": {
                            "totalCount": 101,
                            "pageInfo": {
                                "hasNextPage": False,
                                "endCursor": None,
                            },
                            "nodes": [
                                {
                                    "id": "comment-100",
                                    "author": {"login": "reviewer"},
                                    "body": "comment 100",
                                    "path": "src/file.cc",
                                    "createdAt": "2026-01-01T00:00:01Z",
                                }
                            ],
                        },
                    }
                }
            },
        )

    record = fetch_pr(
        TokenPool(["token"]),
        "owner",
        "repo",
        7,
        post_fn=response,
        thread_post_fn=thread_response,
    )

    assert thread_calls == [{"threadId": "thread-1", "cursor": "cursor"}]
    review_comments = [
        comment
        for comment in record["comments"]
        if comment["kind"] == "review_comment"
    ]
    assert len(review_comments) == 101
    assert review_comments[-1]["body"] == "comment 100"


def test_gap_fetch_paginates_connections_independently_without_duplicates() -> None:
    calls: list[dict] = []

    def response(_token: str, variables: dict):
        calls.append(dict(variables))
        second = variables["cReviews"] is not None
        return (
            200,
            {},
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "number": 7,
                            "title": "PR",
                            "body": "body",
                            "mergeCommit": {"oid": "abc"},
                            "comments": {
                                "totalCount": 1,
                                "pageInfo": {
                                    "hasNextPage": False,
                                    "endCursor": "comments-final",
                                },
                                "nodes": [
                                    {
                                        "author": {"login": "commenter"},
                                        "body": "only comment",
                                        "createdAt": "2026-01-01T00:00:00Z",
                                    }
                                ],
                            },
                            "reviews": {
                                "totalCount": 2,
                                "pageInfo": {
                                    "hasNextPage": not second,
                                    "endCursor": "reviews-next" if not second else None,
                                },
                                "nodes": [
                                    {
                                        "author": {"login": f"reviewer-{int(second)}"},
                                        "state": "APPROVED",
                                        "body": f"review-{int(second)}",
                                        "submittedAt": (
                                            "2026-01-02T00:00:00Z"
                                            if not second
                                            else "2026-01-03T00:00:00Z"
                                        ),
                                    }
                                ],
                            },
                            "reviewThreads": {
                                "totalCount": 0,
                                "pageInfo": {
                                    "hasNextPage": False,
                                    "endCursor": None,
                                },
                                "nodes": [],
                            },
                            "closingIssuesReferences": {
                                "totalCount": 0,
                                "pageInfo": {
                                    "hasNextPage": False,
                                    "endCursor": None,
                                },
                                "nodes": [],
                            },
                        }
                    }
                }
            },
        )

    record = fetch_pr(
        TokenPool(["token"]),
        "owner",
        "repo",
        7,
        post_fn=response,
    )

    assert len(calls) == 2
    assert calls[1]["cComments"] is None
    assert calls[1]["cReviews"] == "reviews-next"
    assert [item["body"] for item in record["comments"]] == ["only comment"]
    assert [item["body"] for item in record["reviews"]] == [
        "review-0",
        "review-1",
    ]
