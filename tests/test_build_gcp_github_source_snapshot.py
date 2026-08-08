from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.build_gcp_github_source_snapshot import (
    SOURCE_CHECKPOINT_SCHEMA,
    SOURCE_SCOPE_SCHEMA,
    SOURCE_SNAPSHOT_SCHEMA,
    SourceSnapshotError,
    build_public_source_scope,
    canonical_projects,
    freeze_github_source_snapshot,
    parse_ls_remote,
    resolve_head,
)


def _repo_list() -> dict[str, object]:
    return {
        "schema_version": 1,
        "project_identities": ["acme/alpha", "acme/beta"],
        "repos": [
            {
                "bare_name": "alpha.bare",
                "name": "alpha.bare",
                "owner_repo": "acme/alpha",
                "project_identity": "acme/alpha",
                "url": "https://github.com/acme/alpha.git",
            },
            {
                "bare_name": "alpha",
                "name": "alpha",
                "owner_repo": "acme/alpha",
                "project_identity": "acme/alpha",
                "url": "https://github.com/acme/alpha",
            },
            {
                "bare_name": "beta",
                "name": "beta",
                "owner_repo": "acme/beta",
                "project_identity": "acme/beta",
                "url": "https://github.com/acme/beta.git",
            },
        ],
    }


def _mixed_forge_repo_list() -> dict[str, object]:
    return {
        "schema_version": 1,
        "project_identities": ["android.googlesource.com/platform%2Fframeworks%2Fav"],
        "repos": [
            {
                "bare_name": "aosp-frameworks-av.bare",
                "project_identity": "android.googlesource.com/platform%2Fframeworks%2Fav",
                "remote_url": "https://android.googlesource.com/platform/frameworks/av",
            },
            {
                "bare_name": "aosp-frameworks-av",
                "project_identity": "android.googlesource.com/platform%2Fframeworks%2Fav",
                "remote_url": "https://android.googlesource.com/platform/frameworks/av",
            },
        ],
    }


def _public_scope_repo_list() -> dict[str, object]:
    return {
        "schema_version": 2,
        "project_identities": [
            "acme/alpha",
            "corpus.local/private-kit",
            "git.musl-libc.org/musl",
        ],
        "repos": [
            {
                "bare_name": "alpha",
                "owner_repo": "acme/alpha",
                "project_identity": "acme/alpha",
                "url": "https://github.com/acme/alpha",
            },
            {
                "bare_name": "private-kit",
                "project_identity": "corpus.local/private-kit",
                "remote_url": "https://corpus.local/private-kit",
            },
            {
                "bare_name": "musl",
                "project_identity": "git.musl-libc.org/musl",
                "remote_url": "git://git.musl-libc.org/musl",
            },
        ],
    }


def _completed(commit: str, ref: str = "main") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        ["git"],
        0,
        stdout=f"ref: refs/heads/{ref}\tHEAD\n{commit}\tHEAD\n",
        stderr="",
    )


def test_canonical_projects_deduplicates_aliases_and_uses_full_github_mirrors() -> None:
    projects = canonical_projects(_repo_list())

    assert projects == [
        {
            "repo": "acme__alpha",
            "project_id": "acme/alpha",
            "remote_url": "https://github.com/acme/alpha.git",
        },
        {
            "repo": "acme__beta",
            "project_id": "acme/beta",
            "remote_url": "https://github.com/acme/beta.git",
        },
    ]


def test_canonical_projects_rejects_url_project_drift() -> None:
    repo_list = _repo_list()
    repo_list["repos"][0]["url"] = "https://github.com/other/alpha.git"  # type: ignore[index]

    with pytest.raises(SourceSnapshotError, match="differs from owner_repo"):
        canonical_projects(repo_list)


def test_canonical_projects_preserves_non_github_authoritative_remote() -> None:
    assert canonical_projects(_mixed_forge_repo_list()) == [
        {
            "repo": "aosp-frameworks-av",
            "project_id": "android.googlesource.com/platform%2Fframeworks%2Fav",
            "remote_url": "https://android.googlesource.com/platform/frameworks/av",
        }
    ]


def test_public_scope_excludes_private_and_overrides_exact_musl_remote() -> None:
    projects, scope = build_public_source_scope(
        _public_scope_repo_list(), repo_list_sha256="1" * 64
    )

    assert [project["project_id"] for project in projects] == [
        "acme/alpha",
        "git.musl-libc.org/musl",
    ]
    assert projects[1]["remote_url"] == "https://git.musl-libc.org/git/musl"
    assert scope["schema"] == SOURCE_SCOPE_SCHEMA
    assert scope["status"] == "partial"
    assert scope["training_ready"] is False
    assert scope["canonical_project_count"] == 3
    assert scope["included_project_count"] == 2
    assert scope["excluded_project_count"] == 1
    assert scope["remote_override_count"] == 1
    assert scope["canonical_coverage_complete"] is False
    assert scope["exclusions"] == [
        {
            "project_id": "corpus.local/private-kit",
            "source_remote_url": "https://corpus.local/private-kit",
            "reason": "private_corpus_local_not_authorized_for_public_cloud",
            "public_cloud_allowed": False,
            "provenance_authorization_required": True,
        }
    ]
    assert scope["remote_overrides"] == [
        {
            "project_id": "git.musl-libc.org/musl",
            "source_remote_url": "git://git.musl-libc.org/musl",
            "remote_url": "https://git.musl-libc.org/git/musl",
            "reason": "official_https_transport_override",
        }
    ]
    with pytest.raises(SourceSnapshotError, match="explicit public source scope"):
        canonical_projects(_public_scope_repo_list())


def test_public_scope_fails_closed_when_musl_override_input_drifts() -> None:
    repo_list = _public_scope_repo_list()
    repo_list["repos"][2]["remote_url"] = "git://git.musl-libc.org/musl.git"  # type: ignore[index]

    with pytest.raises(SourceSnapshotError, match="override input drifted"):
        build_public_source_scope(repo_list, repo_list_sha256="1" * 64)


def test_parse_ls_remote_requires_one_valid_head() -> None:
    assert parse_ls_remote("ref: refs/heads/main\tHEAD\n" + "a" * 40 + "\tHEAD\n") == (
        "a" * 40,
        "refs/heads/main",
    )
    with pytest.raises(SourceSnapshotError, match="no HEAD"):
        parse_ls_remote("a" * 40 + "\trefs/heads/main\n")


def test_resolve_head_retries_only_transient_rate_limit() -> None:
    calls = 0
    sleeps: list[float] = []

    def runner(_command: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                ["git"], 128, stdout="", stderr="HTTP 429 Too Many Requests"
            )
        return _completed("b" * 40)

    assert resolve_head(
        "https://github.com/acme/alpha.git",
        max_retries=2,
        retry_delay_seconds=0.25,
        runner=runner,
        sleeper=sleeps.append,
    ) == ("b" * 40, "refs/heads/main")
    assert calls == 2
    assert sleeps == [0.25]


def test_freeze_snapshot_resumes_checkpoint_and_writes_immutable_complete_output(
    tmp_path: Path,
) -> None:
    repo_list_path = tmp_path / "repo-list.json"
    repo_list_path.write_text(json.dumps(_repo_list()), encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.json"
    output_path = tmp_path / "snapshot.json"
    calls: list[str] = []

    def runner(command: object) -> subprocess.CompletedProcess[str]:
        remote = list(command)[3]  # type: ignore[arg-type]
        calls.append(remote)
        return _completed("a" * 40 if remote.endswith("alpha.git") else "b" * 40)

    snapshot = freeze_github_source_snapshot(
        repo_list_path=repo_list_path,
        checkpoint_path=checkpoint_path,
        scope_receipt_path=tmp_path / "scope.json",
        output_path=output_path,
        runner=runner,
        sleeper=lambda _: None,
    )

    assert snapshot["schema"] == SOURCE_SNAPSHOT_SCHEMA
    assert snapshot["status"] == "complete"
    assert snapshot["training_ready"] is False
    assert len(snapshot["repositories"]) == 2
    assert len(calls) == 2
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["schema"] == SOURCE_CHECKPOINT_SCHEMA
    assert all(entry["state"] == "complete" for entry in checkpoint["entries"])
    scope = json.loads((tmp_path / "scope.json").read_text())
    assert scope["schema"] == SOURCE_SCOPE_SCHEMA
    assert scope["status"] == "complete"
    assert scope["canonical_coverage_complete"] is True
    assert checkpoint["scope_sha256"] == scope["scope_sha256"]

    def fail_runner(_command: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("a complete checkpoint must skip remote lookups")

    assert (
        freeze_github_source_snapshot(
            repo_list_path=repo_list_path,
            checkpoint_path=checkpoint_path,
            scope_receipt_path=tmp_path / "scope.json",
            output_path=output_path,
            runner=fail_runner,
            sleeper=lambda _: None,
        )
        == snapshot
    )


def test_freeze_partial_scope_never_resolves_excluded_project(tmp_path: Path) -> None:
    repo_list_path = tmp_path / "repo-list.json"
    repo_list_path.write_text(json.dumps(_public_scope_repo_list()), encoding="utf-8")
    calls: list[str] = []

    def runner(command: object) -> subprocess.CompletedProcess[str]:
        remote = list(command)[3]  # type: ignore[arg-type]
        calls.append(remote)
        return _completed("c" * 40)

    snapshot = freeze_github_source_snapshot(
        repo_list_path=repo_list_path,
        checkpoint_path=tmp_path / "checkpoint.json",
        scope_receipt_path=tmp_path / "scope.json",
        output_path=tmp_path / "snapshot.json",
        runner=runner,
        sleeper=lambda _: None,
    )

    assert snapshot["status"] == "partial"
    assert snapshot["training_ready"] is False
    assert snapshot["canonical_project_count"] == 3
    assert snapshot["repository_count"] == 2
    assert snapshot["excluded_project_count"] == 1
    assert snapshot["canonical_coverage_complete"] is False
    assert calls == [
        "https://github.com/acme/alpha.git",
        "https://git.musl-libc.org/git/musl",
    ]
    assert all("corpus.local" not in remote for remote in calls)
    scope = json.loads((tmp_path / "scope.json").read_text())
    assert snapshot["scope_sha256"] == scope["scope_sha256"]


def test_freeze_snapshot_rejects_checkpoint_bound_to_other_input(
    tmp_path: Path,
) -> None:
    repo_list_path = tmp_path / "repo-list.json"
    repo_list_path.write_text(json.dumps(_repo_list()), encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema": SOURCE_CHECKPOINT_SCHEMA,
                "status": "in_progress",
                "repo_list_sha256": "f" * 64,
                "repository_count": 0,
                "repository_order_sha256": "e" * 64,
                "entries": [],
                "checkpoint_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SourceSnapshotError, match="different repo list"):
        freeze_github_source_snapshot(
            repo_list_path=repo_list_path,
            checkpoint_path=checkpoint_path,
            scope_receipt_path=tmp_path / "scope.json",
            output_path=tmp_path / "snapshot.json",
            runner=lambda _command: _completed("a" * 40),
            sleeper=lambda _: None,
        )
