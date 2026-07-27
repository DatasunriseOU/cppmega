from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import streaming_conveyor as conveyor


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    indexer_root = repo / "tools" / "clang_indexer"
    indexer_root.mkdir(parents=True)
    (indexer_root / "index_project.py").write_text(
        "from indexer_helper import INDEXER_VALUE\n",
        encoding="utf-8",
    )
    (indexer_root / "indexer_helper.py").write_text(
        "INDEXER_VALUE = 1\n",
        encoding="utf-8",
    )
    (repo / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Revision Test")
    _git(repo, "config", "user.email", "revision@example.test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def test_revision_receipt_is_schema_v2_and_binds_indexer_dependency_closure(
    source_repo: Path,
) -> None:
    receipt = conveyor.capture_code_revision(source_repo)

    assert receipt["schema_version"] == 2
    assert receipt["producer_role"] == "canonical_source_conveyor"
    assert receipt["repository_identity"] == "cppmega"
    assert receipt["dirty"] is False
    assert len(receipt["source_tree_sha256"]) == 64
    provenance = receipt["indexer_provenance"]
    assert provenance["schema"] == "cppmega_indexer_dependency_binding_v1"
    assert provenance["path"] == "tools/clang_indexer/index_project.py"
    assert set(provenance["dependency_manifest"]) == {
        "tools/clang_indexer/index_project.py",
        "tools/clang_indexer/indexer_helper.py",
    }
    assert receipt["indexer_dependency_closure_sha256"] == (
        provenance["dependency_closure_sha256"]
    )


def test_production_revision_requires_same_checkout_indexer(
    source_repo: Path,
) -> None:
    indexer = source_repo / "tools" / "clang_indexer" / "index_project.py"
    indexer.unlink()
    _git(source_repo, "add", "-u")
    _git(source_repo, "commit", "-q", "-m", "remove indexer")

    with pytest.raises(
        conveyor.CodeRevisionMismatchError,
        match="clang indexer",
    ):
        conveyor.CodeRevisionGuard.for_production(
            _git(source_repo, "rev-parse", "HEAD"),
            repo_root=source_repo,
        )


def test_manifest_identity_rejects_changed_indexer_dependency_closure(
    source_repo: Path,
    tmp_path: Path,
) -> None:
    manifest = conveyor.ConcurrentManifest.load(tmp_path / "_done.json")
    manifest.bind_code_revision(conveyor.capture_code_revision(source_repo))

    helper = source_repo / "tools" / "clang_indexer" / "indexer_helper.py"
    helper.write_text("INDEXER_VALUE = 2\n", encoding="utf-8")
    _git(source_repo, "add", str(helper.relative_to(source_repo)))
    _git(source_repo, "commit", "-q", "-m", "change indexer dependency")

    with pytest.raises(
        conveyor.CodeRevisionMismatchError,
        match="manifest code revision mismatch",
    ):
        manifest.bind_code_revision(conveyor.capture_code_revision(source_repo))


def test_code_revision_ignores_generated_outputs_but_rejects_source_drift(
    source_repo: Path,
) -> None:
    guard = conveyor.CodeRevisionGuard.for_production(
        _git(source_repo, "rev-parse", "HEAD"),
        repo_root=source_repo,
    )
    output = source_repo / "outputs" / "corpus" / "part.parquet"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"generated")
    guard.verify("generated output")

    (source_repo / "scripts" / "worker.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )
    with pytest.raises(conveyor.CodeRevisionDriftError, match="worktree"):
        guard.verify("source edit")


def _write_pr_completion_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    pr_store = tmp_path / "prs.sqlite"
    repo_list = tmp_path / "repo_list.json"
    receipt_path = tmp_path / "pr_completion.json"
    pr_store.write_bytes(b"immutable sqlite fixture")
    repo_list.write_text('{"repos":["owner/repo"]}\n', encoding="utf-8")

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    receipt_path.write_text(
        json.dumps(
            {
                "schema": "cppmega_pr_completion_v2",
                "status": "verified",
                "pr_store": {
                    "path": str(pr_store.resolve()),
                    "sha256": sha256(pr_store),
                },
                "repo_list": {
                    "path": str(repo_list.resolve()),
                    "sha256": sha256(repo_list),
                },
                "expected_repos_sha256": "a" * 64,
                "scan_id": "1" * 64,
                "expected_repo_count": 1,
                "declared_pr_count": 7,
                "stored_pr_count": 7,
                "unverified_store_pr_count": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt_path, pr_store, repo_list


def test_pr_completion_binding_hashes_explicit_store_and_repo_list(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    wal = Path(f"{pr_store}-wal")
    wal.write_bytes(b"uncheckpointed")
    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="uncheckpointed WAL",
    ):
        conveyor.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=repo_list,
        )
    wal.unlink()

    binding = conveyor.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )

    assert binding == {
        "schema": "cppmega_pr_completion_v2",
        "status": "verified",
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "pr_store_sha256": hashlib.sha256(pr_store.read_bytes()).hexdigest(),
        "repo_list_sha256": hashlib.sha256(repo_list.read_bytes()).hexdigest(),
        "expected_repos_sha256": "a" * 64,
        "scan_id": "1" * 64,
        "expected_repo_count": 1,
        "stored_pr_count": 7,
        "unverified_store_pr_count": 0,
    }

    pr_store.write_bytes(b"changed after verification")
    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="pr_store hash mismatch",
    ):
        conveyor.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=repo_list,
        )


def test_pr_completion_binding_rejects_legacy_receipt_without_scan_membership(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema"] = "cppmega_pr_completion_v1"
    receipt.pop("scan_id")
    receipt.pop("unverified_store_pr_count")
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="unsupported PR completion schema",
    ):
        conveyor.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=repo_list,
        )


def test_pr_completion_binding_rejects_receipt_over_metadata_bound(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    receipt_path.write_bytes(b"x" * (4 * 1024 * 1024 + 1))

    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="exceeds the 4 MiB metadata bound",
    ):
        conveyor.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=repo_list,
        )


def test_pr_completion_finish_revalidation_detects_input_drift(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    binding = conveyor.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )
    conveyor.revalidate_pr_completion_binding(
        binding,
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )

    repo_list.write_text('{"repos":["other/repo"]}\n', encoding="utf-8")
    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="repo_list hash mismatch",
    ):
        conveyor.revalidate_pr_completion_binding(
            binding,
            receipt_path,
            pr_store=pr_store,
            repo_list=repo_list,
        )


def test_manifest_pr_completion_binding_is_preserved_and_resume_bound(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    binding = conveyor.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )
    manifest_path = tmp_path / "conveyor" / "_done.json"
    manifest = conveyor.ConcurrentManifest.load(manifest_path)
    manifest.bind_pr_completion(binding)
    manifest.mark_done("owner_repo::code", {"rows": 1})

    reloaded = conveyor.ConcurrentManifest.load(manifest_path)
    assert reloaded.pr_completion == binding

    changed = dict(binding)
    changed["receipt_sha256"] = "b" * 64
    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="PR completion mismatch",
    ):
        reloaded.bind_pr_completion(changed)


@pytest.mark.parametrize(
    "legacy_key",
    (
        "owner_repo::r0",
        "owner_repo::commits",
        "owner_repo::commit_plan",
        "owner_repo::no_git",
        "owner_repo::repo",
    ),
)
def test_manifest_rejects_legacy_commit_receipts_without_pr_binding(
    tmp_path: Path,
    legacy_key: str,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    binding = conveyor.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )
    manifest = conveyor.ConcurrentManifest.load(tmp_path / "_done.json")
    manifest.mark_done(legacy_key, {"rows": 1})

    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="commit work receipts",
    ):
        manifest.bind_pr_completion(binding)


def test_commit_stream_requires_explicit_verified_pr_inputs() -> None:
    with pytest.raises(
        SystemExit,
        match=(
            r"commits/both requires explicit immutable PR inputs: "
            r"--pr-store, --repo-list, --pr-completion-receipt"
        ),
    ):
        conveyor.main(["--streams", "commits"])
