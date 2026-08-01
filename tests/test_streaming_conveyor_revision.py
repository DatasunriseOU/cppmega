from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import streaming_conveyor as conveyor


def test_source_stream_read_error_is_only_suppressed_after_checkpoint_signal() -> None:
    error = tarfile.ReadError("unexpected end of data")

    with pytest.raises(tarfile.ReadError, match="unexpected end of data"):
        conveyor._handle_source_stream_read_error(error, interrupted=False)

    conveyor._handle_source_stream_read_error(error, interrupted=True)


def test_source_archive_argument_configures_both_streaming_producers(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.tar.zst"
    original_source_archive = conveyor.sr.TARBALL
    original_commit_archive = conveyor.src.TARBALL
    try:
        args = conveyor.parse_args(["--source-archive", str(archive)])
        conveyor.configure_runtime_paths_from_args(args)
        assert conveyor.sr.TARBALL == archive
        assert conveyor.src.TARBALL == archive
    finally:
        conveyor.sr.TARBALL = original_source_archive
        conveyor.src.TARBALL = original_commit_archive


@pytest.mark.parametrize("dedup_near", (True, False))
def test_adaptive_code_retry_preserves_configured_dedup_policy(
    tmp_path: Path,
    dedup_near: bool,
) -> None:
    calls: list[tuple[int, bool]] = []

    def runner(*args: object) -> dict[str, object]:
        calls.append((int(args[9]), bool(args[6])))
        if len(calls) == 1:
            raise conveyor.RepoFailure("repo", "index_project", "exit code 137")
        return {"status": "done"}

    result = conveyor.run_code_half_adaptive(
        "repo",
        "owner/repo",
        tmp_path / "repo",
        (1024,),
        tmp_path / "work",
        tmp_path / "dedup.sqlite",
        dedup_near,
        parse_workers=4,
        runner=runner,
    )

    assert result == {"status": "done"}
    assert calls == [(4, dedup_near), (1, dedup_near)]


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


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_repo_list(path: Path, rows: list[dict[str, str]]) -> None:
    by_bare_name = {
        row["bare_name"]: row["project_identity"] for row in rows
    }
    document = {
        "schema_version": 2,
        "repos": rows,
        "by_bare_name": dict(sorted(by_bare_name.items())),
        "project_identities": sorted(set(by_bare_name.values())),
        "repo_names": sorted(
            {row["owner_repo"] for row in rows if "owner_repo" in row}
        ),
        "unresolved": [],
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _github_row(
    bare_name: str = "project",
    owner_repo: str = "owner/project",
) -> dict[str, str]:
    return {
        "bare_name": bare_name,
        "project_identity": owner_repo,
        "owner_repo": owner_repo,
    }


def _local_row(
    bare_name: str = "local-source",
    project_identity: str = "corpus.local/local-source",
) -> dict[str, str]:
    return {
        "bare_name": bare_name,
        "project_identity": project_identity,
    }


def _write_pr_completion_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    pr_store = tmp_path / "prs.sqlite"
    source_repo_list = tmp_path / "source_repo_list.json"
    pr_repo_list = tmp_path / "pr_repo_list.json"
    receipt_path = tmp_path / "pr_completion.json"
    pr_store.write_bytes(b"immutable sqlite fixture")
    _write_repo_list(
        source_repo_list,
        [
            _github_row(),
            _local_row(),
            _local_row("source-only", "corpus.local/source-only"),
        ],
    )
    _write_repo_list(pr_repo_list, [_github_row(), _local_row()])

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
                    "path": str(pr_repo_list.resolve()),
                    "sha256": sha256(pr_repo_list),
                },
                "expected_repos_sha256": _canonical_json_sha256(
                    ["owner/project"]
                ),
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
    return receipt_path, pr_store, source_repo_list, pr_repo_list


def test_pr_completion_binding_hashes_explicit_store_and_repo_list(
    tmp_path: Path,
) -> None:
    from scripts import streaming_reindex_commits

    assert (
        conveyor.PRCompletionBindingError
        is streaming_reindex_commits.PRCompletionBindingError
    )
    assert (
        conveyor.load_pr_completion_binding
        is streaming_reindex_commits.load_pr_completion_binding
    )
    assert (
        conveyor.revalidate_pr_completion_binding
        is streaming_reindex_commits.revalidate_pr_completion_binding
    )
    receipt_path, pr_store, source_repo_list, pr_repo_list = (
        _write_pr_completion_fixture(tmp_path)
    )
    wal = Path(f"{pr_store}-wal")
    wal.write_bytes(b"uncheckpointed")
    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="uncheckpointed WAL",
    ):
        conveyor.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=pr_repo_list,
        )
    wal.unlink()

    binding = conveyor.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=pr_repo_list,
    )

    assert binding == {
        "schema": "cppmega_pr_completion_v2",
        "status": "verified",
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "pr_store_sha256": hashlib.sha256(pr_store.read_bytes()).hexdigest(),
        "repo_list_sha256": hashlib.sha256(
            pr_repo_list.read_bytes()
        ).hexdigest(),
        "expected_repos_sha256": _canonical_json_sha256(["owner/project"]),
        "scan_id": "1" * 64,
        "expected_repo_count": 1,
        "stored_pr_count": 7,
        "unverified_store_pr_count": 0,
    }
    assert binding["repo_list_sha256"] != hashlib.sha256(
        source_repo_list.read_bytes()
    ).hexdigest()

    pr_store.write_bytes(b"changed after verification")
    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="pr_store hash mismatch",
    ):
        conveyor.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=pr_repo_list,
        )


def test_pr_completion_repo_membership_hash_preserves_repo_list_row_order(
    tmp_path: Path,
) -> None:
    pr_store = tmp_path / "prs.sqlite"
    pr_repo_list = tmp_path / "pr_repo_list.json"
    receipt_path = tmp_path / "pr_completion.json"
    pr_store.write_bytes(b"immutable sqlite fixture")
    _write_repo_list(
        pr_repo_list,
        [
            _github_row("z-project", "z-owner/project"),
            _github_row("a-project", "a-owner/project"),
        ],
    )
    expected_repos = ["z-owner/project", "a-owner/project"]
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "cppmega_pr_completion_v2",
                "status": "verified",
                "pr_store": {
                    "path": str(pr_store.resolve()),
                    "sha256": hashlib.sha256(pr_store.read_bytes()).hexdigest(),
                },
                "repo_list": {
                    "path": str(pr_repo_list.resolve()),
                    "sha256": hashlib.sha256(
                        pr_repo_list.read_bytes()
                    ).hexdigest(),
                },
                "expected_repos_sha256": _canonical_json_sha256(
                    expected_repos
                ),
                "scan_id": "1" * 64,
                "expected_repo_count": 2,
                "declared_pr_count": 0,
                "stored_pr_count": 0,
                "unverified_store_pr_count": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    binding = conveyor.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=pr_repo_list,
    )

    assert binding["expected_repos_sha256"] == _canonical_json_sha256(
        expected_repos
    )


def test_pr_completion_binding_rejects_legacy_receipt_without_scan_membership(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, _source_repo_list, pr_repo_list = (
        _write_pr_completion_fixture(tmp_path)
    )
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
            repo_list=pr_repo_list,
        )


def test_pr_completion_binding_rejects_repo_set_receipt_mismatch(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, _source_repo_list, pr_repo_list = (
        _write_pr_completion_fixture(tmp_path)
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["expected_repo_count"] = 2
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="expected_repo_count does not match",
    ):
        conveyor.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=pr_repo_list,
        )


def test_pr_completion_binding_rejects_receipt_over_metadata_bound(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, _source_repo_list, pr_repo_list = (
        _write_pr_completion_fixture(tmp_path)
    )
    receipt_path.write_bytes(b"x" * (4 * 1024 * 1024 + 1))

    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="exceeds the 4 MiB metadata bound",
    ):
        conveyor.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=pr_repo_list,
        )


def test_pr_completion_finish_revalidation_detects_input_drift(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, _source_repo_list, pr_repo_list = (
        _write_pr_completion_fixture(tmp_path)
    )
    binding = conveyor.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=pr_repo_list,
    )
    conveyor.revalidate_pr_completion_binding(
        binding,
        receipt_path,
        pr_store=pr_store,
        repo_list=pr_repo_list,
    )

    _write_repo_list(pr_repo_list, [_github_row(owner_repo="other/repo")])
    with pytest.raises(
        conveyor.PRCompletionBindingError,
        match="repo_list hash mismatch",
    ):
        conveyor.revalidate_pr_completion_binding(
            binding,
            receipt_path,
            pr_store=pr_store,
            repo_list=pr_repo_list,
        )


def test_manifest_pr_completion_binding_is_preserved_and_resume_bound(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, _source_repo_list, pr_repo_list = (
        _write_pr_completion_fixture(tmp_path)
    )
    binding = conveyor.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=pr_repo_list,
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
    receipt_path, pr_store, _source_repo_list, pr_repo_list = (
        _write_pr_completion_fixture(tmp_path)
    )
    binding = conveyor.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=pr_repo_list,
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
            r"--repo-list, --pr-store, --pr-repo-list, "
            r"--pr-completion-receipt"
        ),
    ):
        conveyor.main(["--streams", "commits"])


def test_dual_repo_lists_accept_source_only_corpus_local_mapping(
    tmp_path: Path,
) -> None:
    _receipt, _store, source_repo_list, pr_repo_list = (
        _write_pr_completion_fixture(tmp_path)
    )

    source, pr = conveyor.load_repo_list_contracts(
        source_repo_list,
        pr_repo_list,
    )

    assert dict(source.project_id_by_bare_name) == {
        "local-source": "corpus.local/local-source",
        "project": "owner/project",
        "source-only": "corpus.local/source-only",
    }
    assert dict(pr.project_id_by_bare_name) == {
        "local-source": "corpus.local/local-source",
        "project": "owner/project",
    }
    assert pr.owner_repo_by_bare_name["project"] == "owner/project"
    assert pr.owner_repo_by_bare_name["local-source"] is None
    assert source.owner_repo_by_bare_name["local-source"] is None
    with pytest.raises(TypeError):
        source.project_id_by_bare_name["project"] = "other/project"


@pytest.mark.parametrize(
    ("source_rows", "pr_rows"),
    [
        (
            [_github_row(), _github_row("other", "owner/other")],
            [_github_row()],
        ),
        (
            [_github_row(), _local_row()],
            [_github_row(), _github_row("unexpected", "owner/unexpected")],
        ),
        (
            [_github_row()],
            [_github_row("project", "different/project")],
        ),
    ],
)
def test_dual_repo_lists_reject_scope_inconsistency(
    tmp_path: Path,
    source_rows: list[dict[str, str]],
    pr_rows: list[dict[str, str]],
) -> None:
    source_repo_list = tmp_path / "source.json"
    pr_repo_list = tmp_path / "pr.json"
    _write_repo_list(source_repo_list, source_rows)
    _write_repo_list(pr_repo_list, pr_rows)

    with pytest.raises(
        conveyor.RepoListBindingError,
        match="PR repo list does not match",
    ):
        conveyor.load_repo_list_contracts(source_repo_list, pr_repo_list)


def test_repo_list_validation_rejects_inconsistent_derived_map(
    tmp_path: Path,
) -> None:
    source_repo_list = tmp_path / "source.json"
    _write_repo_list(source_repo_list, [_github_row()])
    document = json.loads(source_repo_list.read_text(encoding="utf-8"))
    document["by_bare_name"]["project"] = "different/project"
    source_repo_list.write_text(
        json.dumps(document, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        conveyor.RepoListBindingError,
        match="by_bare_name does not match",
    ):
        conveyor.load_repo_list_snapshot(source_repo_list, role="source")


def test_repo_list_validation_rejects_duplicate_bare_name(
    tmp_path: Path,
) -> None:
    source_repo_list = tmp_path / "source.json"
    rows = [
        _github_row(),
        _github_row("project", "owner/project"),
    ]
    _write_repo_list(source_repo_list, rows)

    with pytest.raises(
        conveyor.RepoListBindingError,
        match="duplicate bare_name",
    ):
        conveyor.load_repo_list_snapshot(source_repo_list, role="source")


def test_repo_filesystem_keys_isolate_case_variants() -> None:
    upper = conveyor.sr.filesystem_repo_key("DirectXTK")
    lower = conveyor.sr.filesystem_repo_key("directxtk")

    assert lower == "directxtk"
    assert upper == "%44irect%58%54%4b"
    assert upper.casefold() != lower.casefold()
    assert (
        conveyor.sr.code_output_filename("DirectXTK").casefold()
        != conveyor.sr.code_output_filename("directxtk").casefold()
    )
    assert (
        conveyor.sr.commit_output_filename("DirectXTK", 0).casefold()
        != conveyor.sr.commit_output_filename("directxtk", 0).casefold()
    )


def test_source_repo_list_finish_revalidation_detects_drift(
    tmp_path: Path,
) -> None:
    source_repo_list = tmp_path / "source.json"
    _write_repo_list(source_repo_list, [_github_row(), _local_row()])
    snapshot = conveyor.load_repo_list_snapshot(
        source_repo_list,
        role="source",
    )
    binding = conveyor.build_source_repo_list_binding(snapshot)
    conveyor.revalidate_source_repo_list_binding(binding, source_repo_list)

    _write_repo_list(
        source_repo_list,
        [
            _github_row(),
            _local_row(project_identity="corpus.local/different-source"),
        ],
    )
    with pytest.raises(
        conveyor.RepoListBindingError,
        match="changed while the conveyor was running",
    ):
        conveyor.revalidate_source_repo_list_binding(
            binding,
            source_repo_list,
        )


def test_manifest_source_repo_list_binding_is_resume_bound(
    tmp_path: Path,
) -> None:
    source_repo_list = tmp_path / "source.json"
    _write_repo_list(source_repo_list, [_github_row(), _local_row()])
    binding = conveyor.build_source_repo_list_binding(
        conveyor.load_repo_list_snapshot(source_repo_list, role="source")
    )
    manifest_path = tmp_path / "conveyor" / "_done.json"
    manifest = conveyor.ConcurrentManifest.load(manifest_path)
    manifest.bind_source_repo_list(binding)
    manifest.mark_done("project::code", {"rows": 1})

    reloaded = conveyor.ConcurrentManifest.load(manifest_path)
    assert reloaded.source_repo_list == binding
    changed = dict(binding)
    changed["sha256"] = "f" * 64
    with pytest.raises(
        conveyor.RepoListBindingError,
        match="source repo-list mismatch",
    ):
        reloaded.bind_source_repo_list(changed)


def test_source_completion_receipt_is_bounded_and_binds_manifest_bytes(
    source_repo: Path,
    tmp_path: Path,
) -> None:
    source_repo_list = tmp_path / "source.json"
    _write_repo_list(
        source_repo_list,
        [_github_row(), _local_row()],
    )
    source_binding = conveyor.build_source_repo_list_binding(
        conveyor.load_repo_list_snapshot(source_repo_list, role="source")
    )
    manifest_path = tmp_path / "conveyor" / "_done.json"
    manifest = conveyor.ConcurrentManifest.load(manifest_path)
    revision = conveyor.capture_code_revision(source_repo)
    manifest.bind_code_revision(revision)
    manifest.bind_source_repo_list(source_binding)
    manifest.mark_done("local-source::code", {"rows": 1})
    manifest.mark_done("project::code", {"rows": 2})

    receipt_path = tmp_path / "conveyor" / "completion_receipt.json"
    receipt = conveyor.write_source_completion_receipt(
        receipt_path,
        manifest=manifest,
        streams="code",
        source_repo_list_reverified_at_finish=True,
        interrupted=False,
    )

    assert receipt["schema"] == conveyor.SOURCE_COMPLETION_SCHEMA
    assert receipt["status"] == "success"
    assert receipt["code_repositories"] == ["local-source", "project"]
    assert receipt["code_repository_names_sha256"] == (
        _canonical_json_sha256(["local-source", "project"])
    )
    assert receipt["failed_unit_count"] == 0
    assert receipt["non_code_done_unit_count"] == 0
    assert receipt["code_revision"] == revision
    assert receipt["source_repo_list"] == source_binding
    assert receipt["manifest"] == {
        "path": str(manifest_path.resolve()),
        "size_bytes": manifest_path.stat().st_size,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    assert receipt_path.stat().st_size < 4 * 1024 * 1024


def test_manifest_rejects_legacy_work_without_source_repo_binding(
    tmp_path: Path,
) -> None:
    source_repo_list = tmp_path / "source.json"
    _write_repo_list(source_repo_list, [_github_row()])
    binding = conveyor.build_source_repo_list_binding(
        conveyor.load_repo_list_snapshot(source_repo_list, role="source")
    )
    manifest = conveyor.ConcurrentManifest.load(tmp_path / "_done.json")
    manifest.mark_done("project::code", {"rows": 1})

    with pytest.raises(
        conveyor.RepoListBindingError,
        match="work receipts",
    ):
        manifest.bind_source_repo_list(binding)


def test_commit_stream_does_not_alias_missing_pr_repo_list(
    tmp_path: Path,
) -> None:
    source_repo_list = tmp_path / "source.json"
    _write_repo_list(source_repo_list, [_github_row()])

    with pytest.raises(
        SystemExit,
        match=(
            r"commits/both requires explicit immutable PR inputs: "
            r"--pr-repo-list"
        ),
    ):
        conveyor.main(
            [
                "--streams",
                "commits",
                "--repo-list",
                str(source_repo_list),
                "--pr-store",
                str(tmp_path / "prs.sqlite"),
                "--pr-completion-receipt",
                str(tmp_path / "completion.json"),
            ]
        )


def test_code_stream_accepts_source_repo_list_without_pr_inputs(
    tmp_path: Path,
) -> None:
    source_repo_list = tmp_path / "source.json"
    _write_repo_list(source_repo_list, [_github_row()])

    with pytest.raises(SystemExit, match="expected-code-revision"):
        conveyor.main(
            [
                "--streams",
                "code",
                "--repo-list",
                str(source_repo_list),
            ]
        )


def test_process_one_repo_commits_path_records_git_log_failure_not_typeerror(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "_src"
    repo_dir.mkdir()
    work_root = tmp_path / "work"
    work_parent = tmp_path / "work-parent"
    work_parent.mkdir()
    manifest = conveyor.Manifest.load(tmp_path / "_done.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = conveyor.process_one_repo(
            "empty",
            repo_dir,
            (1024,),
            (1024,),
            1,
            0,
            work_root,
            work_parent,
            pool,
            manifest,
            threading.Lock(),
            False,
            {"valid": 0},
            False,
            None,
            True,
            tmp_path / "prs.sqlite",
            streams="commits",
            project_id="owner/empty",
            pr_owner_repo="owner/empty",
            pr_scan_id="a" * 64,
            source_quarantine_manifest=tmp_path / "code-only-quarantine.json",
        )

    assert result["commits_done"] == 0
    assert result["commits_failed"] == 0
    assert manifest.failed["empty::commits"]["stage"] == "git_log"
