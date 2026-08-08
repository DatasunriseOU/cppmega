#!/usr/bin/env python3
"""Freeze direct full-mirror inputs from a canonical cppmega repo list.

The distributed source worker accepts only a concrete commit.  This utility
turns the canonical repository mapping into a resume-safe, immutable source
specification: every project is resolved once through ``git ls-remote`` and
the workers later clone the full mirror and verify that exact commit. GitHub
projects are normalized to direct HTTPS remotes; public non-GitHub projects
preserve their authoritative network forge remote except for exact, audited
transport overrides. Private ``corpus.local`` identities are recorded in a
separate immutable exclusion receipt and never sent to public cloud. The
utility never uses the local corpus archive or a shallow clone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_sha256,
    load_json_object,
    sha256_file,
)

SOURCE_SNAPSHOT_SCHEMA = "cppmega.gcp_github_source_snapshot_v1"
SOURCE_CHECKPOINT_SCHEMA = "cppmega.gcp_github_source_checkpoint_v1"
SOURCE_SCOPE_SCHEMA = "cppmega.gcp_public_source_scope_v1"
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}")
_OWNER_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_PROJECT_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._+%:-]*/[A-Za-z0-9][A-Za-z0-9._+%/-]*"
)
_REPO_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,255}")
_PUBLIC_SCOPE_POLICY = "public_network_only_v1"
_CORPUS_LOCAL_HOST = "corpus.local"
_REMOTE_OVERRIDES = {
    "git.musl-libc.org/musl": {
        "source_remote_url": "git://git.musl-libc.org/musl",
        "remote_url": "https://git.musl-libc.org/git/musl",
        "reason": "official_https_transport_override",
    }
}


class SourceSnapshotError(ContractError):
    """The direct public source snapshot cannot be safely frozen."""


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Sleeper = Callable[[float], None]


def _json_file(path: Path, *, where: str) -> tuple[bytes, dict[str, Any]]:
    raw, value = load_json_object(path, where=where)
    if not isinstance(value, dict):
        raise SourceSnapshotError(f"{where} must be a JSON object")
    return raw, dict(value)


def _canonical_owner_repo(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _OWNER_REPO_RE.fullmatch(value) is None:
        raise SourceSnapshotError(f"{where} must be a canonical GitHub owner/repo")
    return value


def _canonical_project_id(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _PROJECT_ID_RE.fullmatch(value) is None:
        raise SourceSnapshotError(f"{where} must be a canonical project identity")
    return value


def _github_remote(owner_repo: str) -> str:
    return f"https://github.com/{owner_repo}.git"


def _parsed_remote(value: object, *, where: str):
    if not isinstance(value, str) or not value:
        raise SourceSnapshotError(f"{where} must be a non-empty network remote")
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        raise SourceSnapshotError(f"{where} is not an absolute network remote")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SourceSnapshotError(f"{where} is not a canonical network remote")
    if not parsed.path or parsed.path == "/":
        raise SourceSnapshotError(f"{where} has no repository path")
    return parsed


def _network_remote(value: object, *, where: str) -> str:
    parsed = _parsed_remote(value, where=where)
    assert isinstance(value, str)
    if parsed.scheme not in {"https", "ssh"}:
        raise SourceSnapshotError(
            f"{where} must use an authoritative https:// or ssh:// remote"
        )
    if parsed.hostname is not None and parsed.hostname.lower() == _CORPUS_LOCAL_HOST:
        raise SourceSnapshotError(f"{where} must not name corpus.local")
    return value


def _validate_github_url(value: object, *, owner_repo: str, where: str) -> None:
    remote = _network_remote(value, where=where)
    parsed = urlsplit(remote)
    if parsed.hostname is None or parsed.hostname.lower() != "github.com":
        raise SourceSnapshotError(f"{where} is not a canonical GitHub remote")
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if path != owner_repo:
        raise SourceSnapshotError(
            f"{where} project differs from owner_repo: {path!r} != {owner_repo!r}"
        )


def _repo_label(owner_repo: str, occupied: set[str]) -> str:
    candidate = owner_repo.replace("/", "__")
    if _REPO_LABEL_RE.fullmatch(candidate) is None:
        digest = hashlib.sha256(owner_repo.encode("utf-8")).hexdigest()[:16]
        candidate = f"repo_{digest}"
    if candidate not in occupied:
        occupied.add(candidate)
        return candidate
    digest = hashlib.sha256(owner_repo.encode("utf-8")).hexdigest()[:16]
    candidate = f"{candidate}__{digest}"
    if _REPO_LABEL_RE.fullmatch(candidate) is None or candidate in occupied:
        raise SourceSnapshotError(f"cannot derive a unique repo label for {owner_repo}")
    occupied.add(candidate)
    return candidate


def build_public_source_scope(
    repo_list: Mapping[str, object], *, repo_list_sha256: str
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Route every canonical project into public GCP scope or an explicit exclusion."""

    if re.fullmatch(r"[0-9a-f]{64}", repo_list_sha256) is None:
        raise SourceSnapshotError("repo_list_sha256 is invalid")
    raw_repos = repo_list.get("repos")
    if not isinstance(raw_repos, list) or not raw_repos:
        raise SourceSnapshotError("canonical repo list has no non-empty repos list")
    candidates: dict[str, list[dict[str, str]]] = {}
    for index, raw in enumerate(raw_repos):
        if not isinstance(raw, Mapping):
            raise SourceSnapshotError(f"repo_list.repos[{index}] must be an object")
        project_id = _canonical_project_id(
            raw.get("project_identity"),
            where=f"repo_list.repos[{index}].project_identity",
        )
        bare_name = raw.get("bare_name")
        if not isinstance(bare_name, str) or not bare_name:
            raise SourceSnapshotError(f"repo_list.repos[{index}].bare_name is invalid")
        name = raw.get("name", bare_name)
        if not isinstance(name, str) or not name:
            raise SourceSnapshotError(f"repo_list.repos[{index}].name is invalid")
        candidate: dict[str, str] = {
            "bare_name": bare_name,
            "name": name,
            "project_id": project_id,
        }
        if "owner_repo" in raw:
            owner_repo = _canonical_owner_repo(
                raw.get("owner_repo"), where=f"repo_list.repos[{index}].owner_repo"
            )
            if owner_repo != project_id:
                raise SourceSnapshotError(
                    f"repo_list.repos[{index}] owner_repo/project_identity drift"
                )
            _validate_github_url(
                raw.get("url"),
                owner_repo=owner_repo,
                where=f"repo_list.repos[{index}].url",
            )
            candidate.update(
                {
                    "label_hint": owner_repo,
                    "route": "included",
                    "remote_url": _github_remote(owner_repo),
                    "source_remote_url": str(raw.get("url")),
                }
            )
        else:
            source_remote = raw.get("remote_url", raw.get("url"))
            parsed = _parsed_remote(
                source_remote, where=f"repo_list.repos[{index}].remote_url"
            )
            assert isinstance(source_remote, str)
            host = str(parsed.hostname).lower()
            candidate.update(
                {
                    "label_hint": bare_name,
                    "source_remote_url": source_remote,
                }
            )
            if host == _CORPUS_LOCAL_HOST:
                if not project_id.startswith(f"{_CORPUS_LOCAL_HOST}/"):
                    raise SourceSnapshotError(
                        f"repo_list.repos[{index}] corpus.local identity drift"
                    )
                if parsed.scheme != "https":
                    raise SourceSnapshotError(
                        f"repo_list.repos[{index}] corpus.local source scheme drift"
                    )
                candidate["route"] = "excluded"
            else:
                if project_id.startswith(f"{_CORPUS_LOCAL_HOST}/"):
                    raise SourceSnapshotError(
                        f"repo_list.repos[{index}] corpus.local route escaped exclusion"
                    )
                override = _REMOTE_OVERRIDES.get(project_id)
                if override is not None:
                    if source_remote != override["source_remote_url"]:
                        raise SourceSnapshotError(
                            f"repo_list.repos[{index}] approved remote override input drifted"
                        )
                    candidate["remote_url"] = _network_remote(
                        override["remote_url"],
                        where=f"repo_list.repos[{index}].override_remote_url",
                    )
                    candidate["override_reason"] = override["reason"]
                else:
                    candidate["remote_url"] = _network_remote(
                        source_remote,
                        where=f"repo_list.repos[{index}].remote_url",
                    )
                candidate["route"] = "included"
        candidates.setdefault(project_id, []).append(candidate)

    expected_projects = repo_list.get("project_identities")
    if not isinstance(expected_projects, list) or not expected_projects:
        raise SourceSnapshotError("canonical repo list has no project_identities list")
    expected = [
        _canonical_project_id(item, where="project_identities item")
        for item in expected_projects
    ]
    if len(expected) != len(set(expected)):
        raise SourceSnapshotError(
            "canonical repo list has duplicate project identities"
        )
    if set(expected) != set(candidates):
        missing = sorted(set(expected) - set(candidates))
        unexpected = sorted(set(candidates) - set(expected))
        raise SourceSnapshotError(
            f"canonical repo/project coverage drift: missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    occupied: set[str] = set()
    projects: list[dict[str, str]] = []
    exclusions: list[dict[str, object]] = []
    overrides: list[dict[str, str]] = []
    for project_id in sorted(expected):
        project_candidates = candidates[project_id]
        routes = {item["route"] for item in project_candidates}
        if len(routes) != 1:
            raise SourceSnapshotError(
                f"project aliases cross public scope routes: {project_id}"
            )
        chosen = min(
            project_candidates,
            key=lambda item: (
                item["bare_name"].endswith(".bare"),
                item["name"].endswith(".bare"),
                item["bare_name"],
                item["name"],
            ),
        )
        if chosen["route"] == "excluded":
            if any(
                str(
                    _parsed_remote(item["source_remote_url"], where=project_id).hostname
                ).lower()
                != _CORPUS_LOCAL_HOST
                for item in project_candidates
            ):
                raise SourceSnapshotError(
                    f"excluded project aliases escaped corpus.local: {project_id}"
                )
            exclusions.append(
                {
                    "project_id": project_id,
                    "source_remote_url": chosen["source_remote_url"],
                    "reason": "private_corpus_local_not_authorized_for_public_cloud",
                    "public_cloud_allowed": False,
                    "provenance_authorization_required": True,
                }
            )
            continue
        remote_urls = {item["remote_url"] for item in project_candidates}
        if len(remote_urls) != 1:
            raise SourceSnapshotError(
                f"project aliases resolve to different remotes: {project_id}"
            )
        project = {
            "repo": _repo_label(chosen["label_hint"], occupied),
            "project_id": project_id,
            "remote_url": chosen["remote_url"],
        }
        projects.append(project)
        if "override_reason" in chosen:
            overrides.append(
                {
                    "project_id": project_id,
                    "source_remote_url": chosen["source_remote_url"],
                    "remote_url": chosen["remote_url"],
                    "reason": chosen["override_reason"],
                }
            )

    scope: dict[str, object] = {
        "schema": SOURCE_SCOPE_SCHEMA,
        "status": "partial" if exclusions else "complete",
        "training_ready": False,
        "policy": {
            "name": _PUBLIC_SCOPE_POLICY,
            "allow_private_sources": False,
            "excluded_hostname": _CORPUS_LOCAL_HOST,
            "remote_override_project_ids": sorted(_REMOTE_OVERRIDES),
        },
        "repo_list_sha256": repo_list_sha256,
        "canonical_project_count": len(expected),
        "included_project_count": len(projects),
        "excluded_project_count": len(exclusions),
        "remote_override_count": len(overrides),
        "canonical_coverage_complete": not exclusions,
        "included_project_order_sha256": canonical_sha256(
            [project["project_id"] for project in projects]
        ),
        "included_projects": projects,
        "exclusions": exclusions,
        "remote_overrides": overrides,
    }
    scope["scope_sha256"] = canonical_sha256(scope)
    return projects, scope


def canonical_projects(repo_list: Mapping[str, object]) -> list[dict[str, str]]:
    """Reduce a fully public, override-free repo list to canonical projects."""

    projects, scope = build_public_source_scope(
        repo_list, repo_list_sha256=canonical_sha256(repo_list)
    )
    if scope["excluded_project_count"] or scope["remote_override_count"]:
        raise SourceSnapshotError(
            "canonical repo list requires an explicit public source scope receipt"
        )
    return projects


def parse_ls_remote(stdout: str) -> tuple[str, str | None]:
    """Extract a single HEAD commit and optional symbolic default ref."""

    commit: str | None = None
    default_ref: str | None = None
    for line in stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            continue
        left, right = fields
        if right != "HEAD":
            continue
        if left.startswith("ref: "):
            candidate = left.removeprefix("ref: ")
            if not candidate.startswith("refs/heads/") or len(candidate) <= len(
                "refs/heads/"
            ):
                raise SourceSnapshotError(
                    "git ls-remote returned an invalid symbolic HEAD ref"
                )
            default_ref = candidate
            continue
        if _GIT_OBJECT_RE.fullmatch(left) is None:
            raise SourceSnapshotError("git ls-remote returned an invalid HEAD commit")
        if commit is not None and commit != left:
            raise SourceSnapshotError("git ls-remote returned conflicting HEAD commits")
        commit = left
    if commit is None:
        raise SourceSnapshotError("git ls-remote returned no HEAD commit")
    return commit, default_ref


def _run_git(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def _transient_failure(completed: subprocess.CompletedProcess[str]) -> bool:
    message = f"{completed.stdout}\n{completed.stderr}".lower()
    markers = (
        "429",
        "too many requests",
        "rate limit",
        "http 5",
        "connection reset",
        "connection timed out",
        "could not resolve host",
        "temporary failure",
        "remote end hung up",
    )
    return any(marker in message for marker in markers)


def resolve_head(
    remote_url: str,
    *,
    git: str = "git",
    max_retries: int = 4,
    retry_delay_seconds: float = 2.0,
    runner: Runner = _run_git,
    sleeper: Sleeper = time.sleep,
) -> tuple[str, str | None]:
    """Resolve a public network HEAD with bounded transient retry."""

    if max_retries < 0:
        raise SourceSnapshotError("max_retries must be >= 0")
    for attempt in range(max_retries + 1):
        completed = runner((git, "ls-remote", "--symref", remote_url, "HEAD"))
        if completed.returncode == 0:
            return parse_ls_remote(completed.stdout)
        if not _transient_failure(completed) or attempt == max_retries:
            detail = (completed.stderr or completed.stdout).strip().replace("\n", " ")
            raise SourceSnapshotError(
                f"git ls-remote failed for {remote_url} (exit {completed.returncode}): {detail[-800:]}"
            )
        sleeper(retry_delay_seconds * (2**attempt))
    raise AssertionError("unreachable")


def _checkpoint_payload(
    *,
    repo_list_sha256: str,
    scope_sha256: str,
    projects: Sequence[Mapping[str, str]],
    resolutions: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for project in projects:
        resolution = resolutions.get(project["project_id"])
        if resolution is None:
            entries.append({**project, "state": "pending"})
            continue
        commit = resolution.get("expected_commit")
        default_ref = resolution.get("default_ref")
        if not isinstance(commit, str) or _GIT_OBJECT_RE.fullmatch(commit) is None:
            raise SourceSnapshotError("checkpoint has invalid resolved commit")
        if default_ref is not None and not isinstance(default_ref, str):
            raise SourceSnapshotError("checkpoint has invalid default ref")
        entries.append(
            {
                **project,
                "state": "complete",
                "expected_commit": commit,
                "default_ref": default_ref,
            }
        )
    payload: dict[str, object] = {
        "schema": SOURCE_CHECKPOINT_SCHEMA,
        "status": "in_progress",
        "repo_list_sha256": repo_list_sha256,
        "scope_sha256": scope_sha256,
        "repository_count": len(projects),
        "repository_order_sha256": canonical_sha256(
            [project["project_id"] for project in projects]
        ),
        "entries": entries,
    }
    payload["checkpoint_sha256"] = canonical_sha256(payload)
    return payload


def _load_checkpoint(
    path: Path,
    *,
    repo_list_sha256: str,
    scope_sha256: str,
    projects: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    raw, checkpoint = _json_file(path, where="GitHub source checkpoint")
    if checkpoint.get("schema") != SOURCE_CHECKPOINT_SCHEMA:
        raise SourceSnapshotError("GitHub source checkpoint schema drifted")
    if checkpoint.get("status") != "in_progress":
        raise SourceSnapshotError("GitHub source checkpoint is not resumable")
    if checkpoint.get("repo_list_sha256") != repo_list_sha256:
        raise SourceSnapshotError(
            "GitHub source checkpoint binds a different repo list"
        )
    if checkpoint.get("scope_sha256") != scope_sha256:
        raise SourceSnapshotError(
            "GitHub source checkpoint binds a different public scope"
        )
    expected_digest = checkpoint.get("checkpoint_sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_digest
    ):
        raise SourceSnapshotError("GitHub source checkpoint digest is invalid")
    digest_payload = dict(checkpoint)
    digest_payload.pop("checkpoint_sha256", None)
    if canonical_sha256(digest_payload) != expected_digest:
        raise SourceSnapshotError("GitHub source checkpoint digest drifted")
    entries = checkpoint.get("entries")
    if not isinstance(entries, list) or len(entries) != len(projects):
        raise SourceSnapshotError("GitHub source checkpoint entries drifted")
    expected_projects = [
        (row["repo"], row["project_id"], row["remote_url"]) for row in projects
    ]
    actual_projects = [
        (entry.get("repo"), entry.get("project_id"), entry.get("remote_url"))
        for entry in entries
        if isinstance(entry, Mapping)
    ]
    if actual_projects != expected_projects:
        raise SourceSnapshotError("GitHub source checkpoint project order drifted")
    result: dict[str, dict[str, object]] = {}
    for entry in entries:
        assert isinstance(entry, Mapping)
        if entry.get("state") == "pending":
            continue
        if entry.get("state") != "complete":
            raise SourceSnapshotError("GitHub source checkpoint entry state is invalid")
        project_id = entry.get("project_id")
        commit = entry.get("expected_commit")
        if not isinstance(project_id, str) or not isinstance(commit, str):
            raise SourceSnapshotError("GitHub source checkpoint completion is invalid")
        result[project_id] = {
            "expected_commit": commit,
            "default_ref": entry.get("default_ref"),
        }
    if hashlib.sha256(raw).hexdigest() != sha256_file(path):
        raise SourceSnapshotError("GitHub source checkpoint changed while loading")
    return result


def _frozen_snapshot(
    *,
    repo_list_sha256: str,
    scope: Mapping[str, object],
    projects: Sequence[Mapping[str, str]],
    resolutions: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    repositories: list[dict[str, object]] = []
    for project in projects:
        resolution = resolutions.get(project["project_id"])
        if resolution is None:
            raise SourceSnapshotError("cannot freeze incomplete GitHub source snapshot")
        expected_commit = resolution.get("expected_commit")
        if (
            not isinstance(expected_commit, str)
            or _GIT_OBJECT_RE.fullmatch(expected_commit) is None
        ):
            raise SourceSnapshotError("cannot freeze invalid GitHub commit")
        repositories.append(
            {
                "repo": project["repo"],
                "project_id": project["project_id"],
                "source": {
                    "kind": "git_mirror",
                    "remote_url": project["remote_url"],
                    "expected_commit": expected_commit,
                    "expected_tree": None,
                },
            }
        )
    snapshot: dict[str, object] = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "status": scope["status"],
        "training_ready": False,
        "repo_list_sha256": repo_list_sha256,
        "scope_sha256": scope["scope_sha256"],
        "canonical_project_count": scope["canonical_project_count"],
        "excluded_project_count": scope["excluded_project_count"],
        "canonical_coverage_complete": scope["canonical_coverage_complete"],
        "repository_count": len(repositories),
        "repository_order_sha256": canonical_sha256(
            [str(row["project_id"]) for row in repositories]
        ),
        "repositories": repositories,
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    return snapshot


def _write_immutable_json(
    path: Path, payload: Mapping[str, object], *, where: str
) -> None:
    if not path.exists():
        atomic_write_json(path, dict(payload))
        return
    _raw, existing = _json_file(path, where=where)
    if existing != payload:
        raise SourceSnapshotError(f"immutable {where} already differs: {path}")


def freeze_github_source_snapshot(
    *,
    repo_list_path: Path,
    checkpoint_path: Path,
    scope_receipt_path: Path,
    output_path: Path,
    git: str = "git",
    max_retries: int = 4,
    retry_delay_seconds: float = 2.0,
    runner: Runner = _run_git,
    sleeper: Sleeper = time.sleep,
) -> dict[str, object]:
    """Resume direct HEAD resolution and publish a scope-bound frozen spec."""

    raw, repo_list = _json_file(repo_list_path, where="canonical repo list")
    repo_list_sha256 = hashlib.sha256(raw).hexdigest()
    if sha256_file(repo_list_path) != repo_list_sha256:
        raise SourceSnapshotError("canonical repo list changed while loading")
    projects, scope = build_public_source_scope(
        repo_list, repo_list_sha256=repo_list_sha256
    )
    _write_immutable_json(
        scope_receipt_path,
        scope,
        where="public source scope receipt",
    )
    scope_sha256 = str(scope["scope_sha256"])
    resolutions = _load_checkpoint(
        checkpoint_path,
        repo_list_sha256=repo_list_sha256,
        scope_sha256=scope_sha256,
        projects=projects,
    )
    for project in projects:
        project_id = project["project_id"]
        if project_id in resolutions:
            continue
        expected_commit, default_ref = resolve_head(
            project["remote_url"],
            git=git,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            runner=runner,
            sleeper=sleeper,
        )
        resolutions[project_id] = {
            "expected_commit": expected_commit,
            "default_ref": default_ref,
        }
        atomic_write_json(
            checkpoint_path,
            _checkpoint_payload(
                repo_list_sha256=repo_list_sha256,
                scope_sha256=scope_sha256,
                projects=projects,
                resolutions=resolutions,
            ),
        )
    snapshot = _frozen_snapshot(
        repo_list_sha256=repo_list_sha256,
        scope=scope,
        projects=projects,
        resolutions=resolutions,
    )
    _write_immutable_json(
        output_path,
        snapshot,
        where="frozen GitHub source snapshot",
    )
    return snapshot


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-list", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scope-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--git", default="git")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.retry_delay_seconds < 0:
        parser.error("--retry-delay-seconds must be >= 0")
    try:
        snapshot = freeze_github_source_snapshot(
            repo_list_path=args.repo_list,
            checkpoint_path=args.checkpoint,
            scope_receipt_path=args.scope_receipt,
            output_path=args.output,
            git=args.git,
            max_retries=args.max_retries,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    except (SourceSnapshotError, OSError, ValueError) as exc:
        parser.exit(2, f"GitHub source snapshot failed: {exc}\n")
    print(snapshot["snapshot_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "SOURCE_CHECKPOINT_SCHEMA",
    "SOURCE_SCOPE_SCHEMA",
    "SOURCE_SNAPSHOT_SCHEMA",
    "SourceSnapshotError",
    "build_public_source_scope",
    "canonical_projects",
    "freeze_github_source_snapshot",
    "parse_ls_remote",
    "resolve_head",
]
