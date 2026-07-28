from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLANG_INDEXER = ROOT / "tools" / "clang_indexer"
if str(CLANG_INDEXER) not in sys.path:
    sys.path.insert(0, str(CLANG_INDEXER))

import process_commits  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verified_pr_completion(
    *,
    scan_id: str = "a" * 64,
    receipt_sha256: str = "3" * 64,
    pr_store_sha256: str = "4" * 64,
    repo_list_sha256: str = "1" * 64,
) -> dict[str, object]:
    return {
        "schema": "cppmega_pr_completion_v2",
        "status": "verified",
        "receipt_sha256": receipt_sha256,
        "pr_store_sha256": pr_store_sha256,
        "repo_list_sha256": repo_list_sha256,
        "expected_repos_sha256": "6" * 64,
        "scan_id": scan_id,
        "expected_repo_count": 1,
        "stored_pr_count": 7,
        "unverified_store_pr_count": 0,
    }


def test_extract_git_history_uses_explicit_identity_for_staged_src(tmp_path: Path) -> None:
    repo = tmp_path / "_src"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Identity Test")
    _git(repo, "config", "user.email", "identity@example.invalid")
    source = repo / "main.cpp"
    source.write_text("int value() { return 1; }\n", encoding="utf-8")
    _git(repo, "add", "main.cpp")
    _git(repo, "commit", "-qm", "initial")
    source.write_text("int value() { return 2; }\n", encoding="utf-8")
    _git(repo, "add", "main.cpp")
    _git(repo, "commit", "-qm", "modify")

    output = tmp_path / "commits.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "nanochat_data" / "extract_git_history.py"),
            "--repo",
            str(repo),
            "--repo-name",
            "tests/staged-src",
            "--output",
            str(output),
            "--notes",
            "off",
            "--checkpoint-commits",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records
    assert {record["repo"] for record in records} == {"tests/staged-src"}


def test_process_commits_repairs_legacy_identity_and_stable_ids() -> None:
    record = {
        "repo": "_src",
        "repo_stable_id": "stale",
        "filepath": "source/blender/editors/object/object.cc",
        "filepath_stable_id": "stale",
        "commit_hash": "abc123",
    }

    changed = process_commits.normalize_record_project_identity(
        record,
        project_id="blender/blender",
    )

    assert changed is True
    assert record["repo"] == "blender/blender"
    assert record["repo_stable_id"] == hashlib.sha1(
        b"blender/blender"
    ).hexdigest()[:16]
    assert record["filepath_stable_id"] == hashlib.sha1(
        b"blender/blender\0source/blender/editors/object/object.cc"
    ).hexdigest()[:16]


def test_process_commits_rejects_conflicting_canonical_identity() -> None:
    record = {
        "repo": "other/project",
        "filepath": "src/file.cc",
        "commit_hash": "abc123",
    }

    with pytest.raises(process_commits.SymbolIdentityError, match="conflicts"):
        process_commits.normalize_record_project_identity(
            record,
            project_id="blender/blender",
        )


def test_commit_command_keeps_project_identity_independent_from_pr_key(
    tmp_path: Path,
) -> None:
    from scripts import streaming_reindex_commits

    commit_input = tmp_path / "input.jsonl"
    command = streaming_reindex_commits.build_process_commits_command(
        [commit_input],
        tmp_path / "output.jsonl",
        tmp_path / "_src",
        None,
        pr_store=tmp_path / "prs.sqlite",
        project_id="source/project",
        pr_owner_repo="pr-owner/pr-project",
        pr_scan_id="a" * 64,
    )

    command = [str(value) for value in command]
    assert command[command.index("--project-id") + 1] == "source/project"
    assert command[command.index("--pr-repo") + 1] == "pr-owner/pr-project"
    assert command[command.index("--pr-scan-id") + 1] == "a" * 64
    assert "--repo-list" not in command


def test_source_only_commit_command_omits_every_pr_argument(
    tmp_path: Path,
) -> None:
    from scripts import streaming_reindex_commits

    command = streaming_reindex_commits.build_process_commits_command(
        [tmp_path / "input.jsonl"],
        tmp_path / "output.jsonl",
        tmp_path / "_src",
        None,
        pr_store=tmp_path / "prs.sqlite",
        project_id="corpus.local/local-source",
        pr_owner_repo=None,
    )
    command = [str(value) for value in command]

    assert command[command.index("--project-id") + 1] == (
        "corpus.local/local-source"
    )
    assert "--pr-store" not in command
    assert "--pr-repo" not in command
    assert "--pr-scan-id" not in command
    assert "--repo-list" not in command
    receipt = streaming_reindex_commits.empty_after_dedup_info(
        "local-source",
        0,
        1,
        1,
        project_id="corpus.local/local-source",
        pr_eligible=False,
    )
    assert receipt["project_id"] == "corpus.local/local-source"
    assert receipt["pr_eligible"] is False


def test_verified_standalone_rejects_legacy_list_but_direct_loader_keeps_it(
    tmp_path: Path,
) -> None:
    from scripts import streaming_reindex_commits

    legacy_list = tmp_path / "legacy.json"
    legacy_list.write_text(
        json.dumps(
            {
                "repos": [
                    {
                        "name": "project",
                        "owner_repo": "owner/project",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert streaming_reindex_commits.load_pr_owner_repo_map(legacy_list) == {
        "project": "owner/project"
    }
    with pytest.raises(
        streaming_reindex_commits.RepoListBindingError,
        match="schema_version",
    ):
        streaming_reindex_commits.load_repo_list_snapshot(
            legacy_list,
            role="source",
        )


def test_verified_standalone_manifest_resume_is_input_bound(
    tmp_path: Path,
) -> None:
    from scripts import streaming_reindex_commits

    snapshot = streaming_reindex_commits.RepoListSnapshot(
        path=tmp_path / "repos.json",
        sha256="1" * 64,
        canonical_mapping_sha256="2" * 64,
        mapping_count=1,
        project_id_by_bare_name=MappingProxyType(
            {"project": "owner/project"}
        ),
        owner_repo_by_bare_name=MappingProxyType(
            {"project": "owner/project"}
        ),
        github_repos=("owner/project",),
    )
    manifest_path = tmp_path / "done.json"
    manifest = streaming_reindex_commits.Manifest.load(manifest_path)
    completion = _verified_pr_completion()
    inputs = {
        "source": snapshot,
        "pr": snapshot,
        "pr_completion": completion,
    }

    with pytest.raises(
        streaming_reindex_commits.RepoListBindingError,
        match="repo-list hash does not match",
    ):
        streaming_reindex_commits.bind_verified_manifest_inputs(
            manifest,
            **{
                **inputs,
                "pr_completion": {
                    **completion,
                    "repo_list_sha256": "9" * 64,
                },
            },
        )
    streaming_reindex_commits.bind_verified_manifest_inputs(manifest, **inputs)
    streaming_reindex_commits.bind_verified_manifest_inputs(
        streaming_reindex_commits.Manifest.load(manifest_path),
        **inputs,
    )
    with pytest.raises(
        streaming_reindex_commits.RepoListBindingError,
        match="input mismatch",
    ):
        streaming_reindex_commits.bind_verified_manifest_inputs(
            streaming_reindex_commits.Manifest.load(manifest_path),
            **{
                **inputs,
                "pr_completion": {
                    **completion,
                    "receipt_sha256": "7" * 64,
                },
            },
        )
    with pytest.raises(
        streaming_reindex_commits.RepoListBindingError,
        match="input mismatch",
    ):
        streaming_reindex_commits.bind_verified_manifest_inputs(
            streaming_reindex_commits.Manifest.load(manifest_path),
            **{
                **inputs,
                "pr_completion": {
                    **completion,
                    "pr_store_sha256": "8" * 64,
                },
            },
        )


def test_standalone_scan_is_derived_from_receipt_and_never_self_verified() -> None:
    from scripts import streaming_reindex_commits

    completion = _verified_pr_completion(scan_id="a" * 64)

    assert streaming_reindex_commits.resolve_verified_pr_scan_id(
        completion,
        None,
    ) == "a" * 64
    assert streaming_reindex_commits.resolve_verified_pr_scan_id(
        completion,
        "a" * 64,
    ) == "a" * 64
    with pytest.raises(
        streaming_reindex_commits.PRCompletionBindingError,
        match="requires --pr-completion-receipt",
    ):
        streaming_reindex_commits.resolve_verified_pr_scan_id(
            None,
            "a" * 64,
        )
    with pytest.raises(
        streaming_reindex_commits.PRCompletionBindingError,
        match="does not match",
    ):
        streaming_reindex_commits.resolve_verified_pr_scan_id(
            completion,
            "b" * 64,
        )


def test_standalone_rejects_missing_receipt_before_initializing_dedup(
    tmp_path: Path,
) -> None:
    row = {
        "bare_name": "project",
        "project_identity": "owner/project",
        "owner_repo": "owner/project",
    }
    repo_list_document = {
        "schema_version": 2,
        "repos": [row],
        "by_bare_name": {"project": "owner/project"},
        "project_identities": ["owner/project"],
        "repo_names": ["owner/project"],
        "unresolved": [],
    }
    source_repo_list = tmp_path / "source_repo_list.json"
    pr_repo_list = tmp_path / "pr_repo_list.json"
    for path in (source_repo_list, pr_repo_list):
        path.write_text(
            json.dumps(repo_list_document, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    pr_store = tmp_path / "prs.sqlite"
    pr_store.write_bytes(b"immutable store")
    dedup_db = tmp_path / "dedup" / "global.sqlite"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "streaming_reindex_commits.py"),
            "--repo-list",
            str(source_repo_list),
            "--pr-repo-list",
            str(pr_repo_list),
            "--pr-store",
            str(pr_store),
            "--pr-completion-receipt",
            str(tmp_path / "missing-completion.json"),
            "--dedup-db",
            str(dedup_db),
            "--max-repos",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "PR completion receipt is missing" in completed.stderr
    assert not dedup_db.exists()
    assert not dedup_db.parent.exists()


def test_process_commits_verified_scan_requires_explicit_pr_repo(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "clang_indexer" / "process_commits.py"),
            "--inputs",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--pr-store",
            str(tmp_path / "prs.sqlite"),
            "--pr-scan-id",
            "a" * 64,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "--pr-scan-id requires --pr-store and --pr-repo" in completed.stderr


def test_process_commits_rejects_invalid_explicit_pr_repo(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "clang_indexer" / "process_commits.py"),
            "--inputs",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--pr-store",
            str(tmp_path / "prs.sqlite"),
            "--pr-repo",
            "not-a-key",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "project identity must contain exactly one slash" in completed.stderr
