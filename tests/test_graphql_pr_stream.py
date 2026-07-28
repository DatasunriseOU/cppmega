from __future__ import annotations

import json
import sys
import threading
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


def test_explicit_rate_limit_wait_retries_only_all_token_cooling():
    from graphql_pr_stream import (
        AllTokensExhausted,
        run_with_optional_rate_limit_wait,
    )

    attempts = 0
    sleeps: list[float] = []
    notices: list[tuple[float, int]] = []

    def run_once() -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AllTokensExhausted(1.2)
        return {"status": "done"}

    result = run_with_optional_rate_limit_wait(
        run_once,
        wait=True,
        sleep_fn=sleeps.append,
        on_wait=lambda exc, seconds: notices.append(
            (exc.soonest_s, seconds)
        ),
    )

    assert result == {"status": "done"}
    assert attempts == 2
    assert sleeps == [3.0]
    assert notices == [(1.2, 3)]


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
    assert restarted.data["schema"] == GRAPHQL_MANIFEST_SCHEMA
    assert (
        restarted.data["query_contract_sha256"]
        == GRAPHQL_QUERY_CONTRACT_SHA256
    )
    assert restarted.data["scan_id"] == restarted.scan_id
    assert len(restarted.scan_id) == 64
    assert restarted.data["repos"] == {}
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


def test_manifest_can_explicitly_restart_an_invalid_scan_identity(tmp_path):
    from graphql_pr_stream import (
        GRAPHQL_MANIFEST_SCHEMA,
        GRAPHQL_QUERY_CONTRACT_SHA256,
        Manifest,
    )

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": GRAPHQL_MANIFEST_SCHEMA,
                "query_contract_sha256": GRAPHQL_QUERY_CONTRACT_SHA256,
                "scan_id": "invalid",
                "repos": {"owner/repo": {"status": "done"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="invalid scan_id"):
        Manifest(str(path))
    restarted = Manifest(str(path), restart_query_contract=True)

    assert len(restarted.scan_id) == 64
    assert restarted.data["repos"] == {}
    assert len(list(tmp_path.glob("manifest.json.pre-*.json"))) == 1


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


def test_max_prs_finishes_page_and_advances_exact_scan_cursor(tmp_path):
    from graphql_pr_stream import Manifest, SharedTokenPool, stream_repo
    from pr_store import connect

    manifest = Manifest(str(tmp_path / "manifest.json"))
    store = connect(str(tmp_path / "prs.sqlite"), create=True)
    node = {
        "title": "title",
        "body": "body",
        "comments": {"totalCount": 0, "nodes": []},
        "reviews": {"totalCount": 0, "nodes": []},
        "reviewThreads": {"totalCount": 0, "nodes": []},
        "closingIssuesReferences": {"totalCount": 0, "nodes": []},
    }

    def post(_token: str, variables: dict) -> tuple[int, dict, dict]:
        assert variables["cursor"] is None
        return (
            200,
            {"X-RateLimit-Remaining": "100"},
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "totalCount": 3,
                            "pageInfo": {
                                "hasNextPage": True,
                                "endCursor": "page-one-end",
                            },
                            "nodes": [
                                {**node, "number": 1},
                                {**node, "number": 2},
                            ],
                        }
                    }
                }
            },
        )

    try:
        stats = stream_repo(
            SharedTokenPool(["token"]),
            store,
            manifest,
            "owner/repo",
            fallback_pr_threshold=0,
            fallback_ratelimit_trips=0,
            fallback_list_path=str(tmp_path / "fallback.jsonl"),
            max_prs=1,
            truncated_targets_path=str(tmp_path / "targets.jsonl"),
            post_fn=post,
        )
        assert stats["prs"] == 2
        assert manifest.status("owner/repo") == "in_progress"
        assert manifest.cursor("owner/repo") == "page-one-end"
        assert manifest.get("owner/repo")["prs"] == 2
        assert manifest.get("owner/repo")["truncated"] == 0
        assert store.execute(
            "SELECT COUNT(*) FROM prs WHERE scan_id=?",
            (manifest.scan_id,),
        ).fetchone()[0] == 2
    finally:
        store.close()


def test_truncated_target_count_is_snapshotted_once_per_repo(tmp_path):
    from graphql_pr_stream import Manifest, SharedTokenPool, stream_repo
    from pr_store import connect

    class CountingTargetSet(set):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    manifest = Manifest(str(tmp_path / "manifest.json"))
    store = connect(str(tmp_path / "prs.sqlite"), create=True)
    target_keys = CountingTargetSet(
        {("other/repo", number) for number in range(1, 101)}
    )
    targets_path = tmp_path / "targets.jsonl"
    targets_path.write_text(
        "".join(
            json.dumps(
                {"repo": repo, "pr_number": number},
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for repo, number in sorted(target_keys)
        ),
        encoding="utf-8",
    )
    target_keys.iterations = 0
    base = {
        "title": "title",
        "body": "body",
        "reviews": {"totalCount": 0, "nodes": []},
        "reviewThreads": {"totalCount": 0, "nodes": []},
        "closingIssuesReferences": {"totalCount": 0, "nodes": []},
    }

    def post(_token: str, variables: dict) -> tuple[int, dict, dict]:
        first = variables["cursor"] is None
        return (
            200,
            {"X-RateLimit-Remaining": "100"},
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "totalCount": 2,
                            "pageInfo": {
                                "hasNextPage": first,
                                "endCursor": "page-one" if first else None,
                            },
                            "nodes": [
                                {
                                    **base,
                                    "number": 1 if first else 2,
                                    "comments": {
                                        "totalCount": 1 if first else 0,
                                        "nodes": [],
                                    },
                                }
                            ],
                        }
                    }
                }
            },
        )

    try:
        stream_repo(
            SharedTokenPool(["token"]),
            store,
            manifest,
            "owner/repo",
            fallback_pr_threshold=0,
            fallback_ratelimit_trips=0,
            fallback_list_path=str(tmp_path / "fallback.jsonl"),
            truncated_targets_path=str(targets_path),
            truncated_target_keys=target_keys,
            append_lock=threading.Lock(),
            post_fn=post,
        )
        assert manifest.get("owner/repo")["truncated"] == 1
        assert ("owner/repo", 1) in target_keys
        assert target_keys.iterations == 1
    finally:
        store.close()


def test_truncated_target_set_and_file_update_share_one_lock(tmp_path):
    from graphql_pr_stream import (
        Manifest,
        SharedTokenPool,
        reset_repo_truncated_targets,
        stream_repo,
    )
    from pr_store import connect

    targets_path = tmp_path / "targets.jsonl"
    targets_path.write_text(
        '{"pr_number":9,"repo":"other/repo"}\n',
        encoding="utf-8",
    )
    target_keys = {("other/repo", 9)}

    class InterleavingLock:
        def __init__(self):
            self.acquisitions = 0

        def __enter__(self):
            self.acquisitions += 1
            if self.acquisitions == 3:
                reset_repo_truncated_targets(
                    str(targets_path),
                    target_keys,
                    "other/repo",
                )
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    append_lock = InterleavingLock()
    manifest = Manifest(str(tmp_path / "manifest.json"))
    store = connect(str(tmp_path / "prs.sqlite"), create=True)
    node = {
        "number": 1,
        "title": "title",
        "body": "body",
        "comments": {"totalCount": 1, "nodes": []},
        "reviews": {"totalCount": 0, "nodes": []},
        "reviewThreads": {"totalCount": 0, "nodes": []},
        "closingIssuesReferences": {"totalCount": 0, "nodes": []},
    }

    def post(_token: str, _variables: dict) -> tuple[int, dict, dict]:
        return (
            200,
            {"X-RateLimit-Remaining": "100"},
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "totalCount": 1,
                            "pageInfo": {
                                "hasNextPage": False,
                                "endCursor": None,
                            },
                            "nodes": [node],
                        }
                    }
                }
            },
        )

    try:
        stream_repo(
            SharedTokenPool(["token"]),
            store,
            manifest,
            "owner/repo",
            fallback_pr_threshold=0,
            fallback_ratelimit_trips=0,
            fallback_list_path=str(tmp_path / "fallback.jsonl"),
            truncated_targets_path=str(targets_path),
            truncated_target_keys=target_keys,
            append_lock=append_lock,
            post_fn=post,
        )
    finally:
        store.close()

    persisted = {
        (item["repo"], item["pr_number"])
        for item in (
            json.loads(line)
            for line in targets_path.read_text(encoding="utf-8").splitlines()
        )
    }
    assert append_lock.acquisitions == 2
    assert persisted == target_keys == {
        ("other/repo", 9),
        ("owner/repo", 1),
    }
    assert len(targets_path.read_text(encoding="utf-8").splitlines()) == 2


def test_stream_accepts_append_only_membership_growth_and_binds_final_count(
    tmp_path,
):
    from graphql_pr_stream import Manifest, SharedTokenPool, stream_repo
    from pr_store import connect

    manifest = Manifest(str(tmp_path / "manifest.json"))
    store = connect(str(tmp_path / "prs.sqlite"), create=True)
    base = {
        "title": "title",
        "body": "body",
        "comments": {"totalCount": 0, "nodes": []},
        "reviews": {"totalCount": 0, "nodes": []},
        "reviewThreads": {"totalCount": 0, "nodes": []},
        "closingIssuesReferences": {"totalCount": 0, "nodes": []},
    }

    def post(_token: str, variables: dict) -> tuple[int, dict, dict]:
        if variables["cursor"] is None:
            total_count = 2
            page_info = {
                "hasNextPage": True,
                "endCursor": "page-one",
            }
            nodes = [{**base, "number": 1}]
        else:
            assert variables["cursor"] == "page-one"
            total_count = 3
            page_info = {
                "hasNextPage": False,
                "endCursor": None,
            }
            nodes = [
                {**base, "number": 2},
                {**base, "number": 3},
            ]
        return (
            200,
            {"X-RateLimit-Remaining": "100"},
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "totalCount": total_count,
                            "pageInfo": page_info,
                            "nodes": nodes,
                        }
                    }
                }
            },
        )

    try:
        stream_repo(
            SharedTokenPool(["token"]),
            store,
            manifest,
            "owner/repo",
            fallback_pr_threshold=0,
            fallback_ratelimit_trips=0,
            fallback_list_path=str(tmp_path / "fallback.jsonl"),
            truncated_targets_path=str(tmp_path / "targets.jsonl"),
            truncated_target_keys=set(),
            post_fn=post,
        )
        record = manifest.get("owner/repo")
        assert record["status"] == "done"
        assert record["initial_total_count"] == 2
        assert record["total_count"] == 3
        assert record["source_growth_count"] == 1
        assert record["prs"] == 3
        assert store.execute(
            "SELECT COUNT(*) FROM prs WHERE scan_id=?",
            (manifest.scan_id,),
        ).fetchone()[0] == 3
    finally:
        store.close()


def test_stream_rejects_membership_shrink_and_resets_cursor(tmp_path):
    from graphql_pr_stream import Manifest, SharedTokenPool, stream_repo
    from pr_store import connect

    manifest = Manifest(str(tmp_path / "manifest.json"))
    store = connect(str(tmp_path / "prs.sqlite"), create=True)
    base = {
        "title": "title",
        "body": "body",
        "comments": {"totalCount": 0, "nodes": []},
        "reviews": {"totalCount": 0, "nodes": []},
        "reviewThreads": {"totalCount": 0, "nodes": []},
        "closingIssuesReferences": {"totalCount": 0, "nodes": []},
    }

    def post(_token: str, variables: dict) -> tuple[int, dict, dict]:
        first = variables["cursor"] is None
        return (
            200,
            {"X-RateLimit-Remaining": "100"},
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "totalCount": 3 if first else 2,
                            "pageInfo": {
                                "hasNextPage": first,
                                "endCursor": "page-one" if first else None,
                            },
                            "nodes": [{**base, "number": 1 if first else 2}],
                        }
                    }
                }
            },
        )

    try:
        with pytest.raises(SystemExit, match="membership shrank"):
            stream_repo(
                SharedTokenPool(["token"]),
                store,
                manifest,
                "owner/repo",
                fallback_pr_threshold=0,
                fallback_ratelimit_trips=0,
                fallback_list_path=str(tmp_path / "fallback.jsonl"),
                truncated_targets_path=str(tmp_path / "targets.jsonl"),
                truncated_target_keys=set(),
                post_fn=post,
            )
        record = manifest.get("owner/repo")
        assert record["status"] == "in_progress"
        assert record["cursor"] is None
        assert "membership shrank" in record["note"]
    finally:
        store.close()


def test_terminal_count_mismatch_restarts_exact_repo_membership(tmp_path):
    from graphql_pr_stream import Manifest, SharedTokenPool, stream_repo
    from pr_store import connect

    manifest = Manifest(str(tmp_path / "manifest.json"))
    store = connect(str(tmp_path / "prs.sqlite"), create=True)
    node = {
        "title": "title",
        "body": "body",
        "comments": {"totalCount": 1, "nodes": []},
        "reviews": {"totalCount": 0, "nodes": []},
        "reviewThreads": {"totalCount": 0, "nodes": []},
        "closingIssuesReferences": {"totalCount": 0, "nodes": []},
    }

    def incomplete_post(
        _token: str, _variables: dict
    ) -> tuple[int, dict, dict]:
        return (
            200,
            {"X-RateLimit-Remaining": "100"},
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "totalCount": 2,
                            "pageInfo": {
                                "hasNextPage": False,
                                "endCursor": None,
                            },
                            "nodes": [{**node, "number": 1}],
                        }
                    }
                }
            },
        )

    def complete_post(
        _token: str, variables: dict
    ) -> tuple[int, dict, dict]:
        assert variables["cursor"] is None
        return (
            200,
            {"X-RateLimit-Remaining": "100"},
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "totalCount": 1,
                            "pageInfo": {
                                "hasNextPage": False,
                                "endCursor": None,
                            },
                            "nodes": [
                                {
                                    **node,
                                    "number": 2,
                                    "comments": {
                                        "totalCount": 0,
                                        "nodes": [],
                                    },
                                }
                            ],
                        }
                    }
                }
            },
        )

    targets_path = tmp_path / "targets.jsonl"
    targets_path.write_text(
        '{"pr_number":9,"repo":"other/repo"}\n',
        encoding="utf-8",
    )
    target_keys = {("other/repo", 9)}
    kwargs = {
        "fallback_pr_threshold": 0,
        "fallback_ratelimit_trips": 0,
        "fallback_list_path": str(tmp_path / "fallback.jsonl"),
        "truncated_targets_path": str(targets_path),
        "truncated_target_keys": target_keys,
    }
    try:
        with pytest.raises(SystemExit, match="membership count mismatch"):
            stream_repo(
                SharedTokenPool(["token"]),
                store,
                manifest,
                "owner/repo",
                post_fn=incomplete_post,
                **kwargs,
            )
        assert manifest.status("owner/repo") == "in_progress"
        assert manifest.cursor("owner/repo") is None
        assert store.execute(
            "SELECT COUNT(*) FROM prs WHERE scan_id=?",
            (manifest.scan_id,),
        ).fetchone()[0] == 1
        assert target_keys == {
            ("other/repo", 9),
            ("owner/repo", 1),
        }

        stream_repo(
            SharedTokenPool(["token"]),
            store,
            manifest,
            "owner/repo",
            post_fn=complete_post,
            **kwargs,
        )

        assert manifest.status("owner/repo") == "done"
        assert store.execute("SELECT COUNT(*) FROM prs").fetchone()[0] == 2
        assert store.execute(
            "SELECT COUNT(*) FROM prs WHERE scan_id=?",
            (manifest.scan_id,),
        ).fetchone()[0] == 1
        assert store.execute(
            "SELECT scan_id FROM prs WHERE repo=? AND pr_number=1",
            ("owner/repo",),
        ).fetchone()[0] is None
        assert target_keys == {("other/repo", 9)}
        assert targets_path.read_text(encoding="utf-8") == (
            '{"pr_number":9,"repo":"other/repo"}\n'
        )
        assert manifest.get("owner/repo")["truncated"] == 0
    finally:
        store.close()


@pytest.mark.parametrize(
    ("page_info", "match"),
    [
        ({"hasNextPage": True, "endCursor": None}, "lacks a valid endCursor"),
        ({"hasNextPage": "yes", "endCursor": "cursor"}, "hasNextPage is invalid"),
    ],
)
def test_stream_rejects_invalid_page_contract_before_store_write(
    tmp_path, page_info, match
):
    from graphql_pr_stream import Manifest, SharedTokenPool, stream_repo
    from pr_store import connect

    manifest = Manifest(str(tmp_path / "manifest.json"))
    store = connect(str(tmp_path / "prs.sqlite"), create=True)

    def post(_token: str, _variables: dict) -> tuple[int, dict, dict]:
        return (
            200,
            {"X-RateLimit-Remaining": "100"},
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "totalCount": 1,
                            "pageInfo": page_info,
                            "nodes": [
                                {
                                    "number": 1,
                                    "comments": {"totalCount": 0, "nodes": []},
                                    "reviews": {"totalCount": 0, "nodes": []},
                                    "reviewThreads": {
                                        "totalCount": 0,
                                        "nodes": [],
                                    },
                                    "closingIssuesReferences": {
                                        "totalCount": 0,
                                        "nodes": [],
                                    },
                                }
                            ],
                        }
                    }
                }
            },
        )

    try:
        with pytest.raises(SystemExit, match=match):
            stream_repo(
                SharedTokenPool(["token"]),
                store,
                manifest,
                "owner/repo",
                fallback_pr_threshold=0,
                fallback_ratelimit_trips=0,
                fallback_list_path=str(tmp_path / "fallback.jsonl"),
                truncated_targets_path=str(tmp_path / "targets.jsonl"),
                truncated_target_keys=set(),
                post_fn=post,
            )
        assert store.execute("SELECT COUNT(*) FROM prs").fetchone()[0] == 0
    finally:
        store.close()
