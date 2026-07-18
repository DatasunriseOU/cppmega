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
