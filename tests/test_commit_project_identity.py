from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

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


def test_commit_stages_forward_canonical_identity(monkeypatch, tmp_path: Path) -> None:
    from scripts import streaming_reindex
    from scripts import streaming_reindex_commits

    index_command: list[str] = []

    def fake_index(_repo, _stage, command, *, log_path, **_kwargs):
        del log_path
        index_command.extend(str(value) for value in command)
        (tmp_path / "blender.enriched.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(streaming_reindex, "run_checked", fake_index)
    commit_input = tmp_path / "input.jsonl"
    commit_input.write_text("{}\n", encoding="utf-8")
    streaming_reindex.stage_index_commits(
        "blender",
        [commit_input],
        tmp_path,
        tmp_path / "_src",
        None,
        project_id="blender/blender",
    )
    assert index_command[index_command.index("--project-id") + 1] == "blender/blender"

    extract_command: list[str] = []

    def fake_extract(_repo, _stage, command, *, log_path, **_kwargs):
        del log_path
        extract_command.extend(str(value) for value in command)
        (tmp_path / "blender_commits.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(streaming_reindex_commits, "run_checked", fake_extract)
    streaming_reindex_commits.stage_extract_commits(
        "blender",
        tmp_path / "_src",
        tmp_path,
        project_id="blender/blender",
    )
    assert extract_command[extract_command.index("--repo-name") + 1] == "blender/blender"
