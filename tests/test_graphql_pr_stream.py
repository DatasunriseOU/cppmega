from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


MLX_ROOT = Path(__file__).resolve().parents[1]
PR_INGEST = MLX_ROOT / "scripts" / "pr_ingest"
if str(PR_INGEST) not in sys.path:
    sys.path.insert(0, str(PR_INGEST))


def test_load_repo_list_deduplicates_preserving_order(tmp_path):
    from graphql_pr_stream import load_repo_list

    repo_list = tmp_path / "repo_list.json"
    repo_list.write_text(
        json.dumps(
            {
                "repos": [
                    {"owner_repo": "a/one"},
                    {"owner_repo": "b/two"},
                    {"owner_repo": "a/one"},
                    {"owner_repo": "c/three"},
                    {"owner_repo": "b/two"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_repo_list(str(repo_list)) == ["a/one", "b/two", "c/three"]


def test_load_repo_list_excludes_non_github_project_identities(tmp_path):
    from graphql_pr_stream import load_repo_list

    repo_list = tmp_path / "repo_list.json"
    repo_list.write_text(
        json.dumps(
            {
                "repos": [
                    {
                        "project_identity": (
                            "android.googlesource.com/platform%2Fframeworks%2Fav"
                        )
                    },
                    {
                        "project_identity": "llvm/llvm-project",
                        "owner_repo": "llvm/llvm-project",
                    },
                    {"owner_repo": "legacy/repo"},
                    {
                        "project_identity": "sourceware.org/git%2Fbinutils-gdb"
                    },
                    {
                        "project_identity": "llvm/llvm-project",
                        "owner_repo": "llvm/llvm-project",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_repo_list(str(repo_list)) == [
        "llvm/llvm-project",
        "legacy/repo",
    ]


def test_load_repo_list_rejects_conflicting_github_identity(tmp_path):
    from graphql_pr_stream import load_repo_list

    repo_list = tmp_path / "repo_list.json"
    repo_list.write_text(
        json.dumps(
            {
                "repos": [
                    {
                        "project_identity": "wrong/repo",
                        "owner_repo": "right/repo",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="conflicting project_identity"):
        load_repo_list(str(repo_list))


def test_manifest_completion_summary_rejects_fallback_and_in_progress(tmp_path):
    from graphql_pr_stream import Manifest, manifest_completion_summary

    manifest = Manifest(str(tmp_path / "manifest.json"))
    manifest.update("a/one", status="done", cursor=None, total_count=3)
    manifest.update("b/two", status="fallback", cursor="cursor-b", total_count=9)
    manifest.update("c/three", status="in_progress", cursor="cursor-c")

    summary = manifest_completion_summary(
        manifest,
        ["a/one", "b/two", "c/three"],
    )
    assert summary["status"] == "incomplete"
    assert summary["done"] == 1
    assert summary["fallback"] == 1
    assert summary["in_progress"] == 1
    assert summary["pending"] == 0
    assert summary["incomplete_repos"] == [
        {"repo": "b/two", "status": "fallback"},
        {"repo": "c/three", "status": "in_progress"},
    ]


def test_manifest_completion_summary_is_complete_only_when_every_repo_done(tmp_path):
    from graphql_pr_stream import Manifest, manifest_completion_summary

    manifest = Manifest(str(tmp_path / "manifest.json"))
    manifest.update("a/one", status="done", cursor=None, total_count=0)
    manifest.update("b/two", status="done", cursor=None, total_count=1)

    summary = manifest_completion_summary(manifest, ["a/one", "b/two"])
    assert summary == {
        "status": "complete",
        "expected": 2,
        "done": 2,
        "fallback": 0,
        "in_progress": 0,
        "pending": 0,
        "other": 0,
        "incomplete_repos": [],
    }


def test_manifest_persists_and_restores_exact_resume_cursor(tmp_path):
    from graphql_pr_stream import Manifest

    path = tmp_path / "manifest.json"
    manifest = Manifest(str(path))
    manifest.update(
        "owner/repo",
        status="in_progress",
        cursor="opaque-end-cursor",
    )

    restored = Manifest(str(path))

    assert restored.cursor("owner/repo") == "opaque-end-cursor"


def test_graphql_rate_limited_error_rotates_to_another_token():
    from graphql_pr_stream import SharedTokenPool, _post_with_rotation

    responses = iter(
        [
            (
                200,
                {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "4102444800"},
                {
                    "errors": [
                        {
                            "type": "RATE_LIMITED",
                            "message": "API rate limit exceeded",
                        }
                    ]
                },
            ),
            (200, {"X-RateLimit-Remaining": "10"}, {"data": {"repository": {}}}),
        ]
    )
    used_tokens: list[str] = []

    def post(token: str, _variables: dict) -> tuple[int, dict, dict]:
        used_tokens.append(token)
        return next(responses)

    pool = SharedTokenPool(["first", "second"])

    result = _post_with_rotation(
        pool,
        {"owner": "a", "name": "one"},
        "a",
        "one",
        max_retries=2,
        post_fn=post,
    )

    assert result == {"data": {"repository": {}}}
    assert used_tokens == ["first", "second"]


def test_manifest_rejects_stale_query_contract_and_archives_on_explicit_restart(
    tmp_path,
):
    from graphql_pr_stream import (
        GRAPHQL_MANIFEST_SCHEMA,
        GRAPHQL_QUERY_CONTRACT_SHA256,
        Manifest,
        archive_query_bound_side_files,
    )

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"repos": {"owner/repo": {"status": "done"}}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="query contract is missing or stale"):
        Manifest(str(path))

    restarted = Manifest(str(path), restart_query_contract=True)
    assert restarted.data == {
        "schema": GRAPHQL_MANIFEST_SCHEMA,
        "query_contract_sha256": GRAPHQL_QUERY_CONTRACT_SHA256,
        "repos": {},
    }
    backups = list(tmp_path.glob("manifest.json.pre-*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8"))["repos"][
        "owner/repo"
    ]["status"] == "done"
    targets = tmp_path / "targets.jsonl"
    fallback = tmp_path / "fallback.jsonl"
    targets.write_text('{"repo":"owner/repo","pr_number":7}\n')
    fallback.write_text('{"repo":"owner/repo"}\n')
    archived = archive_query_bound_side_files(
        restarted,
        (str(targets), str(fallback)),
    )
    assert len(archived) == 2
    assert not targets.exists()
    assert not fallback.exists()
    assert all(Path(item).is_file() for item in archived)


def test_truncated_target_resume_accounting_is_unique_and_fail_closed(tmp_path):
    from graphql_pr_stream import load_truncated_target_keys

    targets = tmp_path / "truncated.jsonl"
    targets.write_text(
        "\n".join(
            [
                json.dumps({"repo": "a/one", "pr_number": 7}),
                json.dumps({"repo": "a/one", "pr_number": 7}),
                json.dumps({"repo": "b/two", "pr_number": 9}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_truncated_target_keys(str(targets)) == {
        ("a/one", 7),
        ("b/two", 9),
    }

    targets.write_text('{"repo":"a/one","pr_number":false}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid target"):
        load_truncated_target_keys(str(targets))


@pytest.mark.parametrize(
    "node",
    [
        {
            "number": 7,
            "comments": {"totalCount": 0, "nodes": []},
            "reviews": {"totalCount": 0, "nodes": []},
            "reviewThreads": {
                "totalCount": 21,
                "nodes": [
                    {
                        "id": f"thread-{index}",
                        "comments": {"totalCount": 0, "nodes": []},
                    }
                    for index in range(20)
                ],
            },
            "closingIssuesReferences": {"totalCount": 0, "nodes": []},
        },
        {
            "number": 8,
            "comments": {"totalCount": 0, "nodes": []},
            "reviews": {"totalCount": 0, "nodes": []},
            "reviewThreads": {"totalCount": 0, "nodes": []},
            "closingIssuesReferences": {
                "totalCount": 21,
                "nodes": [
                    {"number": index, "title": "", "body": ""}
                    for index in range(20)
                ],
            },
        },
    ],
)
def test_pr_node_routes_review_threads_and_linked_issue_overflow_to_gap_fill(
    node,
):
    from graphql_pr_stream import _pr_node_to_record

    _record, truncated = _pr_node_to_record("owner/repo", node)
    assert truncated is True


def test_pr_node_keeps_inline_review_thread_comments_without_gap_target():
    from graphql_pr_stream import _pr_node_to_record

    node = {
        "number": 7,
        "comments": {"totalCount": 0, "nodes": []},
        "reviews": {"totalCount": 0, "nodes": []},
        "reviewThreads": {
            "totalCount": 1,
            "nodes": [
                {
                    "id": "thread-1",
                    "comments": {
                        "totalCount": 1,
                        "nodes": [
                            {
                                "id": "comment-1",
                                "author": {"login": "reviewer"},
                                "body": "inline review",
                                "path": "src/file.cc",
                                "createdAt": "2026-01-01T00:00:00Z",
                            }
                        ],
                    },
                }
            ],
        },
        "closingIssuesReferences": {"totalCount": 0, "nodes": []},
    }

    record, truncated = _pr_node_to_record("owner/repo", node)

    assert truncated is False
    assert record["comments"] == [
        {
            "user": "reviewer",
            "body": "inline review",
            "path": "src/file.cc",
            "created_at": "2026-01-01T00:00:00Z",
            "kind": "review_comment",
        }
    ]
