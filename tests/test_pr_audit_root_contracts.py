from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULES = (
    "scripts.pr_ingest.build_repo_list",
    "scripts.pr_ingest.export_pr_parquet",
    "scripts.pr_ingest.render_discussion",
    "scripts.pr_ingest.pr_store",
    "scripts.pr_ingest.graphql_pr_stream",
    "scripts.pr_ingest.github_graphql_fallback",
    "scripts.audit_sidecar_parquet",
    "scripts.verify_domain_routed_dataset",
)
TARGET_FILES = tuple(
    REPO_ROOT / relative
    for relative in (
        "scripts/pr_ingest/build_repo_list.py",
        "scripts/pr_ingest/export_pr_parquet.py",
        "scripts/pr_ingest/render_discussion.py",
        "scripts/pr_ingest/pr_store.py",
        "scripts/pr_ingest/graphql_pr_stream.py",
        "scripts/pr_ingest/github_graphql_fallback.py",
        "scripts/pr_ingest/gharchive_query.sql",
        "scripts/pr_ingest/gharchive_run.sh",
        "scripts/audit_sidecar_parquet.py",
        "scripts/verify_domain_routed_dataset.py",
    )
)


def test_owned_pr_audit_files_are_root_namespace_only() -> None:
    missing = [str(path.relative_to(REPO_ROOT)) for path in TARGET_FILES if not path.is_file()]
    assert missing == []

    violations: list[str] = []
    for path in TARGET_FILES:
        text = path.read_text(encoding="utf-8")
        if "cppmega_mlx" in text:
            violations.append(str(path.relative_to(REPO_ROOT)))
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
    assert violations == []


def test_owned_modules_import_when_cppmega_mlx_is_unavailable() -> None:
    script = f"""
import importlib
import importlib.abc
import sys

class BlockCppMegaMlx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cppmega_mlx' or fullname.startswith('cppmega_mlx.'):
            raise ImportError('blocked sibling import: ' + fullname)
        return None

sys.meta_path.insert(0, BlockCppMegaMlx())
sys.path.insert(0, {str(REPO_ROOT)!r})
for module_name in {TARGET_MODULES!r}:
    importlib.import_module(module_name)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_graphql_repo_list_keeps_github_lookup_keys_and_rejects_conflicts(tmp_path):
    graphql = importlib.import_module("scripts.pr_ingest.graphql_pr_stream")
    repo_list = tmp_path / "repo_list.json"
    repo_list.write_text(
        json.dumps(
            {
                "repos": [
                    {"project_identity": "forge.example/a%2Fb"},
                    {"project_identity": "llvm/llvm-project", "owner_repo": "llvm/llvm-project"},
                    {"owner_repo": "legacy/repo"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert graphql.load_repo_list(str(repo_list)) == ["llvm/llvm-project", "legacy/repo"]

    repo_list.write_text(
        json.dumps(
            {"repos": [{"project_identity": "wrong/repo", "owner_repo": "right/repo"}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="conflicting project_identity"):
        graphql.load_repo_list(str(repo_list))


def test_pr_store_preserves_root_metadata_and_normalizes_new_discussion_rows(tmp_path):
    store_mod = importlib.import_module("scripts.pr_ingest.pr_store")
    db = tmp_path / "prs.sqlite"
    with store_mod.PRStore(str(db)) as store:
        store.upsert_pr(
            "owner/repo",
            7,
            title="Root title",
            body="Root body",
            state="merged",
            author="alice",
            created_at="2026-07-15T00:00:00Z",
            merged_at="2026-07-15T01:00:00Z",
            merge_commit_sha="sha7",
            comments=[{"author": "reviewer", "body": "Approved."}],
            reviews=[],
            raw={"source": "root"},
            fetched_at="2026-07-15T01:05:00Z",
        )
        store.commit()

    conn = store_mod.connect(str(db), create=False, readonly=True)
    try:
        record = store_mod.get_by_pr(conn, "owner/repo", 7)
        assert record is not None
        assert record["pr_title"] == "Root title"
        assert record["pr_body"] == "Root body"
        assert record["merge_commit_sha"] == "sha7"
        assert record["comments"][0]["user"] == "reviewer"
        assert store_mod.get_by_sha(conn, "owner/repo", "sha7")["pr_number"] == 7
    finally:
        conn.close()


def test_pr_rendering_keeps_metadata_and_exact_constituent_provenance(tmp_path):
    store_mod = importlib.import_module("scripts.pr_ingest.pr_store")
    render_mod = importlib.import_module("scripts.pr_ingest.render_discussion")
    export_mod = importlib.import_module("scripts.pr_ingest.export_pr_parquet")

    db = tmp_path / "prs.sqlite"
    conn = store_mod.connect(str(db), create=True)
    try:
        store_mod.upsert_record(
            conn,
            {
                "repo": "owner/repo",
                "pr_number": 11,
                "merge_commit_sha": "merge11",
                "pr_title": "Fix parser",
                "pr_body": "Details",
                "comments": [{"user": "bob", "body": "Looks good.", "created_at": "1"}],
                "reviews": [{"user": "carol", "state": "APPROVED", "body": "Ship it.", "created_at": "2"}],
                "linked_issues": [{"number": 99, "title": "Parser bug", "body": "Repro"}],
            },
        )
        record = store_mod.get_by_pr(conn, "owner/repo", 11)
        assert record is not None
        discussion = render_mod.render_discussion(record)
        assert "PR #11: Fix parser" in discussion
        assert "Looks good." in discussion
        assert "APPROVED" in discussion
        assert "#99 Parser bug" in discussion

        output = tmp_path / "pr.jsonl"
        assert export_mod._write_pr_jsonl(
            conn, output, repo="owner/repo", offset=0, limit=None
        ) == 1
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["doc_type"] == "pr_discussion"
        assert payload["commit_hash"] == "merge11"
        assert payload["symbol_identities"] == []
        assert json.loads(payload["constituent_provenance_json"]) == [
            {
                "kind": "pr_discussion",
                "repo": "owner/repo",
                "pr_number": 11,
                "merge_commit_sha": "merge11",
            }
        ]
    finally:
        conn.close()


def test_graphql_node_projection_preserves_pr_metadata() -> None:
    graphql = importlib.import_module("scripts.pr_ingest.graphql_pr_stream")
    record, truncated = graphql._pr_node_to_record(
        "owner/repo",
        {
            "number": 12,
            "title": "Metadata",
            "body": "Body",
            "mergeCommit": {"oid": "sha12"},
            "comments": {
                "totalCount": 1,
                "nodes": [{"author": {"login": "d"}, "body": "c", "createdAt": "1"}],
            },
            "reviews": {
                "totalCount": 1,
                "nodes": [{"author": {"login": "r"}, "state": "APPROVED", "body": "ok", "submittedAt": "2"}],
            },
            "closingIssuesReferences": {
                "nodes": [{"number": 4, "title": "Issue", "body": "link"}]
            },
        },
    )
    assert truncated is False
    assert record["pr_title"] == "Metadata"
    assert record["merge_commit_sha"] == "sha12"
    assert record["reviews"][0]["state"] == "APPROVED"
    assert record["linked_issues"] == [{"number": 4, "title": "Issue", "body": "link"}]


def test_audit_and_verifier_argument_contracts_are_fail_closed(capsys):
    audit = importlib.import_module("scripts.audit_sidecar_parquet")
    with pytest.raises(SystemExit) as exc:
        audit.build_arg_parser().parse_args([])
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "--code-root" in stderr
    assert "--commit-root" in stderr
    assert "--pr-root" in stderr
