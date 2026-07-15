from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cppmega.data.symbol_identity import (
    SymbolIdentityError,
    require_project_identity,
    resolve_remote_project_identity,
)
from scripts.pr_ingest import build_repo_list


def _init_clone(parent: Path, name: str, remote_url: str | None) -> Path:
    clone = parent / name
    subprocess.run(
        ["git", "init", "--quiet", str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    if remote_url is not None:
        subprocess.run(
            ["git", "-C", str(clone), "remote", "add", "origin", remote_url],
            check=True,
            capture_output=True,
            text=True,
        )
    return clone


@pytest.mark.parametrize(
    ("remote_url", "project_identity", "owner_repo"),
    [
        (
            "https://android.googlesource.com/platform/frameworks/av",
            "android.googlesource.com/platform%2Fframeworks%2Fav",
            None,
        ),
        (
            "ssh://android.googlesource.com/platform/hardware/interfaces.git/",
            "android.googlesource.com/platform%2Fhardware%2Finterfaces",
            None,
        ),
        (
            "https://android.googlesource.com/platform/system/core.git",
            "android.googlesource.com/platform%2Fsystem%2Fcore",
            None,
        ),
        (
            "git://sourceware.org/git/binutils-gdb.git",
            "sourceware.org/git%2Fbinutils-gdb",
            None,
        ),
        (
            "git@github.com:llvm/llvm-project.git",
            "llvm/llvm-project",
            "llvm/llvm-project",
        ),
    ],
)
def test_remote_project_identity_fixtures(
    remote_url: str,
    project_identity: str,
    owner_repo: str | None,
) -> None:
    resolved = resolve_remote_project_identity(remote_url, source="fixture")

    assert resolved.project_identity == project_identity
    assert resolved.owner_repo == owner_repo
    assert project_identity.count("/") == 1
    assert require_project_identity(project_identity, source="fixture") == project_identity


def test_remote_project_identity_canonicalizes_transport_and_host() -> None:
    https = resolve_remote_project_identity(
        "https://ANDROID.GOOGLESOURCE.COM/platform/frameworks/av.git/",
        source="https fixture",
    )
    ssh = resolve_remote_project_identity(
        "git@android.googlesource.com:platform/frameworks/av",
        source="ssh fixture",
    )

    assert https == ssh


@pytest.mark.parametrize(
    "remote_url",
    [
        "file:///srv/source/repo.git",
        "/srv/source/repo.git",
        "C:/source/repo.git",
        "https://example.com/org/../repo.git",
        "https://example.com/org/%2e%2e/repo.git",
        "https://example.com/org//repo.git",
        "https://example.com/org/repo.git?identity=other",
        "https://example.com/org/repo%ZZ.git",
        "https://example.com/org%2Frepo.git",
        "https://example.com/org/back\\slash.git",
    ],
)
def test_remote_project_identity_rejects_unsafe_or_non_forge_paths(
    remote_url: str,
) -> None:
    with pytest.raises(SymbolIdentityError):
        resolve_remote_project_identity(remote_url, source="unsafe fixture")


def test_remote_path_projection_does_not_flatten_distinct_paths() -> None:
    left = resolve_remote_project_identity(
        "https://example.com/a-b/c.git",
        source="left fixture",
    )
    right = resolve_remote_project_identity(
        "https://example.com/a/b-c.git",
        source="right fixture",
    )

    assert left.project_identity == "example.com/a-b%2Fc"
    assert right.project_identity == "example.com/a%2Fb-c"
    assert left.project_identity != right.project_identity


def test_build_emits_all_project_identities_but_only_github_pr_names(
    tmp_path: Path,
) -> None:
    clones = tmp_path / "clones"
    clones.mkdir()
    github = _init_clone(clones, "llvm-project", "git@github.com:llvm/llvm-project.git")
    android = _init_clone(
        clones,
        "aosp-frameworks-av",
        "https://android.googlesource.com/platform/frameworks/av",
    )
    android_hardware = _init_clone(
        clones,
        "aosp-hardware-interfaces",
        "https://android.googlesource.com/platform/hardware/interfaces",
    )
    android_system = _init_clone(
        clones,
        "aosp-system-core",
        "https://android.googlesource.com/platform/system/core",
    )
    sourceware = _init_clone(
        clones,
        "binutils-gdb",
        "git://sourceware.org/git/binutils-gdb.git",
    )
    xbox = _init_clone(clones, "xbox-leaked-source", None)

    result = build_repo_list.build(
        repo_dirs=[],
        repos=[
            str(github),
            str(android),
            str(android_hardware),
            str(android_system),
            str(sourceware),
            str(xbox),
        ],
        stored_map_path=None,
        allow_unresolved=True,
    )

    assert result["schema_version"] == 2
    entries = {entry["bare_name"]: entry for entry in result["repos"]}
    assert entries["llvm-project"]["project_identity"] == "llvm/llvm-project"
    assert entries["llvm-project"]["owner_repo"] == "llvm/llvm-project"
    assert entries["aosp-frameworks-av"]["project_identity"] == (
        "android.googlesource.com/platform%2Fframeworks%2Fav"
    )
    assert "owner_repo" not in entries["aosp-frameworks-av"]
    assert entries["aosp-hardware-interfaces"]["project_identity"] == (
        "android.googlesource.com/platform%2Fhardware%2Finterfaces"
    )
    assert entries["aosp-system-core"]["project_identity"] == (
        "android.googlesource.com/platform%2Fsystem%2Fcore"
    )
    assert entries["binutils-gdb"]["project_identity"] == (
        "sourceware.org/git%2Fbinutils-gdb"
    )
    assert result["repo_names"] == ["llvm/llvm-project"]
    assert set(result["project_identities"]) == {
        "android.googlesource.com/platform%2Fframeworks%2Fav",
        "android.googlesource.com/platform%2Fhardware%2Finterfaces",
        "android.googlesource.com/platform%2Fsystem%2Fcore",
        "llvm/llvm-project",
        "sourceware.org/git%2Fbinutils-gdb",
    }
    assert result["by_bare_name"]["aosp-frameworks-av"] == (
        "android.googlesource.com/platform%2Fframeworks%2Fav"
    )
    assert [entry["bare_name"] for entry in result["unresolved"]] == [
        "xbox-leaked-source"
    ]
    assert "project_identity" not in result["unresolved"][0]


def test_build_deduplicates_same_bare_name_and_identity(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _init_clone(
        first_root,
        "frameworks-av",
        "https://android.googlesource.com/platform/frameworks/av",
    )
    second = _init_clone(
        second_root,
        "frameworks-av",
        "git@android.googlesource.com:platform/frameworks/av.git",
    )

    result = build_repo_list.build(
        repo_dirs=[],
        repos=[str(first), str(second), str(first)],
        stored_map_path=None,
        allow_unresolved=False,
    )

    assert len(result["repos"]) == 1
    assert result["repos"][0]["project_identity"] == (
        "android.googlesource.com/platform%2Fframeworks%2Fav"
    )


def test_build_rejects_bare_name_identity_collision(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _init_clone(
        first_root,
        "system-core",
        "https://android.googlesource.com/platform/system/core",
    )
    second = _init_clone(second_root, "system-core", "https://example.com/vendor/system/core")

    with pytest.raises(SystemExit, match="project identity collision"):
        build_repo_list.build(
            repo_dirs=[],
            repos=[str(first), str(second)],
            stored_map_path=None,
            allow_unresolved=False,
        )


def test_stored_map_migrates_legacy_owner_repo_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    stored = tmp_path / "repo_list.json"
    stored.write_text(
        json.dumps(
            {
                "repos": [
                    {"bare_name": "cjson", "owner_repo": "DaveGamble/cJSON"}
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_repo_list.build([], [], str(stored), allow_unresolved=False)
    assert result["repos"][0]["project_identity"] == "DaveGamble/cJSON"
    assert result["repos"][0]["owner_repo"] == "DaveGamble/cJSON"

    stored.write_text(
        json.dumps(
            {
                "repos": [
                    {
                        "bare_name": "cjson",
                        "project_identity": "other/cjson",
                        "owner_repo": "DaveGamble/cJSON",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        SystemExit,
        match="conflicting project_identity .* and owner_repo",
    ):
        build_repo_list.load_stored_map(str(stored))


def test_stored_map_migrates_resolvable_non_github_unresolved_remote(
    tmp_path: Path,
) -> None:
    stored = tmp_path / "repo_list.json"
    stored.write_text(
        json.dumps(
            {
                "repos": [],
                "unresolved": [
                    {
                        "name": "blender",
                        "url": "https://projects.blender.org/blender/blender.git",
                        "reason": "remote-url-not-github",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_repo_list.build([], [], str(stored), allow_unresolved=False)

    assert result["unresolved"] == []
    assert result["by_bare_name"]["blender"] == (
        "projects.blender.org/blender%2Fblender"
    )
    assert result["repos"][0]["source"] == "stored_map"
