#!/usr/bin/env python3
"""Publish canonical primary-MR membership from a verified GitLab scan.

This bridge is deliberately read-only with respect to the GitLab stores and
source composition.  It executes the authoritative ``cppmega.mlx`` completion
verifier directly (never its writer entrypoint), derives exact membership with
``cppmega.data.pr_primary_membership``, and publishes the existing portable
ZSTD Parquet plus its canonical receipt.  A second immutable bridge receipt
binds those outputs to the GitLab completion/store and source-composition
proofs without changing the consumer-facing membership schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cppmega.data.pr_primary_membership import (  # noqa: E402
    build_primary_pr_membership,
    publish_primary_pr_membership_inputs,
    verify_primary_pr_membership_binding,
    verify_primary_pr_membership_receipt,
)
from cppmega.data.source_conveyor_composition import (  # noqa: E402
    SourceComposition,
    load_source_composition,
    source_composition_receipt_sha256,
)
from scripts.pr_ingest.pr_store import connect  # noqa: E402


GITLAB_COMPLETION_SCHEMA = "cppmega_gitlab_mr_completion_v1"
BRIDGE_RECEIPT_SCHEMA = "cppmega_gitlab_primary_pr_membership_bridge_v1"
BRIDGE_RECEIPT_NAME = "gitlab_primary_pr_membership_bridge_receipt.json"
VERIFIER_SOURCE_SCHEMA = "cppmega_gitlab_completion_verifier_source_v1"
SOURCE_BINDING_SCHEMA = "cppmega_gitlab_membership_source_binding_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_COMPLETION_BINDING_FIELDS = {
    "schema",
    "status",
    "platform",
    "receipt_sha256",
    "pr_store_sha256",
    "repo_list_sha256",
    "expected_repos_sha256",
    "scan_id",
    "expected_repo_count",
    "stored_pr_count",
    "unverified_store_pr_count",
    "training_ready_without_membership",
}
_VERIFY_WRAPPER = r"""
import json
from pathlib import Path
import runpy
import sys

script = Path(sys.argv[1]).resolve()
root = script.parents[2]
sys.path.insert(0, str(root))
namespace = runpy.run_path(str(script), run_name="_cppmega_gitlab_verifier")
verify = namespace.get("verify_gitlab_completion_receipt")
if not callable(verify):
    raise RuntimeError("GitLab verifier entrypoint is missing")
binding = verify(
    Path(sys.argv[2]),
    pr_store=Path(sys.argv[3]),
    repo_list=Path(sys.argv[4]),
)
print(json.dumps(binding, separators=(",", ":"), sort_keys=True))
"""


class GitLabMembershipBridgeError(RuntimeError):
    """One of the immutable bridge inputs or outputs failed verification."""


CompletionVerifier = Callable[[Path, Path, Path, Path], dict[str, object]]
VerifierBindingLoader = Callable[[Path], dict[str, object]]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GitLabMembershipBridgeError(
                f"JSON input contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, role: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise GitLabMembershipBridgeError(f"{role} is symlinked: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise GitLabMembershipBridgeError(f"{role} is missing: {resolved}")
    return resolved


def _file_descriptor(path: Path, *, role: str) -> dict[str, object]:
    resolved = _regular_file(path, role=role)
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    identity = lambda stat: (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise GitLabMembershipBridgeError(
            f"{role} changed while it was hashed: {resolved}"
        )
    return {
        "path": str(resolved),
        "byte_size": after.st_size,
        "sha256": digest,
    }


def _read_json_object(
    path: Path,
    *,
    role: str,
    canonical: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    descriptor = _file_descriptor(path, role=role)
    if int(descriptor["byte_size"]) < 1 or int(descriptor["byte_size"]) > _MAX_JSON_BYTES:
        raise GitLabMembershipBridgeError(
            f"{role} exceeds its {_MAX_JSON_BYTES}-byte metadata bound"
        )
    resolved = Path(str(descriptor["path"]))
    try:
        value = json.loads(
            resolved.read_bytes(),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitLabMembershipBridgeError(
            f"{role} is invalid UTF-8 JSON: {resolved}"
        ) from exc
    if not isinstance(value, dict):
        raise GitLabMembershipBridgeError(f"{role} must be a JSON object")
    if canonical and resolved.read_bytes() != _canonical_bytes(value):
        raise GitLabMembershipBridgeError(f"{role} is not canonical JSON")
    if _file_descriptor(resolved, role=role) != descriptor:
        raise GitLabMembershipBridgeError(f"{role} changed while it was read")
    return value, descriptor


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GitLabMembershipBridgeError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GitLabMembershipBridgeError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[:2000]
        raise GitLabMembershipBridgeError(
            f"cannot bind GitLab verifier source: git {' '.join(args)}: {detail}"
        )
    return result.stdout


def gitlab_verifier_source_binding(script_path: Path) -> dict[str, object]:
    """Bind the clean cppmega.mlx verifier revision and dependency subtree."""

    script = _regular_file(script_path, role="GitLab completion verifier")
    root = script.parents[2]
    expected = root / "scripts" / "pr_ingest" / "gitlab_mr_stream.py"
    if script != expected:
        raise GitLabMembershipBridgeError(
            "GitLab verifier must be scripts/pr_ingest/gitlab_mr_stream.py "
            f"inside its cppmega.mlx checkout: {script}"
        )
    top = Path(_run_git(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != root:
        raise GitLabMembershipBridgeError(
            f"GitLab verifier checkout root drifted: expected={root} git={top}"
        )
    commit = _run_git(root, "rev-parse", "HEAD").strip()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise GitLabMembershipBridgeError("GitLab verifier git commit is invalid")
    tracked_paths = (
        "scripts/pr_ingest/gitlab_mr_stream.py",
        "scripts/pr_ingest/pr_store.py",
        "cppmega_mlx/data",
    )
    dirty = _run_git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *tracked_paths,
    ).strip()
    if dirty:
        raise GitLabMembershipBridgeError(
            "GitLab verifier dependency subtree is not clean"
        )
    tree_listing = _run_git(root, "ls-tree", "-r", "HEAD", "--", *tracked_paths)
    if not tree_listing.strip():
        raise GitLabMembershipBridgeError(
            "GitLab verifier dependency subtree is absent from its commit"
        )
    return {
        "schema": VERIFIER_SOURCE_SCHEMA,
        "repository_identity": "cppmega.mlx",
        "git_commit": commit,
        "dependency_tree_sha256": hashlib.sha256(
            tree_listing.encode("utf-8")
        ).hexdigest(),
        "script": _file_descriptor(script, role="GitLab completion verifier"),
    }


def verify_gitlab_completion_external(
    script_path: Path,
    receipt_path: Path,
    pr_store: Path,
    repo_list: Path,
) -> dict[str, object]:
    """Run only cppmega.mlx's hardened read-only completion verifier."""

    script = _regular_file(script_path, role="GitLab completion verifier")
    receipt = _regular_file(receipt_path, role="GitLab completion receipt")
    store = _regular_file(pr_store, role="GitLab primary store")
    repos = _regular_file(repo_list, role="GitLab repo list")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _VERIFY_WRAPPER,
            str(script),
            str(receipt),
            str(store),
            str(repos),
        ],
        cwd=script.parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-4000:]
        raise GitLabMembershipBridgeError(
            f"hardened GitLab completion verification failed: {detail}"
        )
    if len(result.stdout.encode("utf-8")) > _MAX_JSON_BYTES:
        raise GitLabMembershipBridgeError(
            "GitLab completion verifier output exceeds the metadata bound"
        )
    try:
        binding = json.loads(
            result.stdout,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitLabMembershipBridgeError(
            "GitLab completion verifier returned invalid JSON"
        ) from exc
    if not isinstance(binding, dict):
        raise GitLabMembershipBridgeError(
            "GitLab completion verifier returned a non-object binding"
        )
    return binding


def _validated_completion_bundle(
    *,
    script_path: Path,
    receipt_path: Path,
    pr_store: Path,
    repo_list: Path,
    verifier: CompletionVerifier,
) -> dict[str, object]:
    binding = verifier(script_path, receipt_path, pr_store, repo_list)
    if set(binding) != _COMPLETION_BINDING_FIELDS:
        raise GitLabMembershipBridgeError(
            "GitLab completion binding fields drifted: "
            f"missing={sorted(_COMPLETION_BINDING_FIELDS - set(binding))} "
            f"extra={sorted(set(binding) - _COMPLETION_BINDING_FIELDS)}"
        )
    if (
        binding.get("schema") != GITLAB_COMPLETION_SCHEMA
        or binding.get("status") != "verified"
        or binding.get("platform") != "gitlab"
        or binding.get("training_ready_without_membership") is not False
    ):
        raise GitLabMembershipBridgeError(
            "GitLab completion binding is unsupported or overclaims readiness"
        )
    for field in (
        "receipt_sha256",
        "pr_store_sha256",
        "repo_list_sha256",
        "expected_repos_sha256",
        "scan_id",
    ):
        _require_sha256(binding.get(field), field=f"completion.{field}")
    for field, minimum in (
        ("expected_repo_count", 1),
        ("stored_pr_count", 0),
        ("unverified_store_pr_count", 0),
    ):
        _require_int(binding.get(field), field=f"completion.{field}", minimum=minimum)

    receipt, receipt_descriptor = _read_json_object(
        receipt_path,
        role="GitLab completion receipt",
        canonical=True,
    )
    store_descriptor = _file_descriptor(pr_store, role="GitLab primary store")
    repo_list_descriptor = _file_descriptor(repo_list, role="GitLab repo list")
    if (
        receipt.get("schema") != GITLAB_COMPLETION_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("platform") != "gitlab"
        or receipt.get("scan_id") != binding["scan_id"]
        or receipt.get("training_ready_without_membership") is not False
        or receipt.get("required_training_gate")
        != "exact_primary_pr_membership_receipt"
        or receipt_descriptor["sha256"] != binding["receipt_sha256"]
        or store_descriptor["sha256"] != binding["pr_store_sha256"]
        or repo_list_descriptor["sha256"] != binding["repo_list_sha256"]
    ):
        raise GitLabMembershipBridgeError(
            "GitLab completion receipt/store binding disagrees with its verifier"
        )
    contract_sha256 = _require_sha256(
        receipt.get("contract_sha256"),
        field="completion.contract_sha256",
    )
    for field in ("manifest", "ancillary_store", "sidecars"):
        if not isinstance(receipt.get(field), dict):
            raise GitLabMembershipBridgeError(
                f"GitLab completion receipt lacks {field} proof"
            )
    return {
        "contract_sha256": contract_sha256,
        "binding": binding,
        "receipt": receipt_descriptor,
        "store": store_descriptor,
        "repo_list": repo_list_descriptor,
        "routed_artifacts": {
            "manifest": dict(receipt["manifest"]),
            "ancillary_store": dict(receipt["ancillary_store"]),
            "sidecars": dict(receipt["sidecars"]),
        },
    }


def _source_binding(
    *,
    composition: SourceComposition,
    commit_root: Path,
    buckets: tuple[int, ...],
    membership: dict[str, object],
) -> dict[str, object]:
    plan = _file_descriptor(
        composition.plan_path,
        role="source composition plan",
    )
    receipt_sha256 = source_composition_receipt_sha256(composition.receipt)
    commit_artifacts = membership.get("commit_artifacts")
    if (
        composition.receipt.get("status") != "complete"
        or composition.receipt.get("plan_sha256") != plan["sha256"]
        or not isinstance(commit_artifacts, dict)
        or commit_artifacts.get("source_composition_sha256") != receipt_sha256
        or commit_artifacts.get("source_composition_plan_sha256")
        != plan["sha256"]
        or commit_artifacts.get("buckets") != list(buckets)
    ):
        raise GitLabMembershipBridgeError(
            "canonical membership does not bind the exact source composition"
        )
    verify_primary_pr_membership_binding(
        membership,
        source_composition=composition,
        commit_root=commit_root,
        buckets=buckets,
    )
    raw_commit_root = commit_root.expanduser()
    if raw_commit_root.is_symlink():
        raise GitLabMembershipBridgeError(
            f"commit root is symlinked: {raw_commit_root}"
        )
    resolved_commit_root = raw_commit_root.resolve()
    if not resolved_commit_root.is_dir():
        raise GitLabMembershipBridgeError(
            f"commit root is missing: {resolved_commit_root}"
        )
    return {
        "schema": SOURCE_BINDING_SCHEMA,
        "plan": plan,
        "composition_receipt_sha256": receipt_sha256,
        "commit_root": str(resolved_commit_root),
        "buckets": list(buckets),
        "commit_artifacts": dict(commit_artifacts),
    }


def _atomic_publish_json(path: Path, value: dict[str, object]) -> None:
    payload = _canonical_bytes(value)
    if path.is_symlink():
        raise GitLabMembershipBridgeError(
            f"bridge receipt is symlinked: {path}"
        )
    if path.exists():
        existing, _descriptor = _read_json_object(
            path,
            role="GitLab membership bridge receipt",
            canonical=True,
        )
        if existing != value:
            raise GitLabMembershipBridgeError(
                f"existing GitLab membership bridge receipt differs: {path}"
            )
        return
    staged = path.with_name(f".{path.name}.staging-{os.getpid()}")
    if staged.exists() or staged.is_symlink():
        raise GitLabMembershipBridgeError(
            f"stale GitLab membership bridge staging file: {staged}"
        )
    try:
        with staged.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        staged.unlink(missing_ok=True)


def _load_composition(
    args: argparse.Namespace,
    *,
    buckets: tuple[int, ...],
    supplied: SourceComposition | None,
) -> SourceComposition:
    if supplied is not None:
        return supplied
    return load_source_composition(
        Path(args.source_composition),
        buckets=buckets,
        code_root=Path(args.code_root),
        commit_root=Path(args.commit_root),
    )


def _parse_buckets(value: str) -> tuple[int, ...]:
    try:
        buckets = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise GitLabMembershipBridgeError(
            "--target-lengths must contain comma-separated integers"
        ) from exc
    if (
        not buckets
        or any(bucket < 1 for bucket in buckets)
        or buckets != tuple(sorted(set(buckets)))
    ):
        raise GitLabMembershipBridgeError(
            "--target-lengths must be unique, positive, and sorted"
        )
    return buckets


def build_gitlab_primary_membership(
    args: argparse.Namespace,
    *,
    source_composition: SourceComposition | None = None,
    completion_verifier: CompletionVerifier | None = None,
    verifier_binding_loader: VerifierBindingLoader | None = None,
) -> dict[str, object]:
    """Verify all inputs, build canonical membership, and seal the bridge."""

    verifier = completion_verifier or verify_gitlab_completion_external
    load_verifier_binding = (
        verifier_binding_loader or gitlab_verifier_source_binding
    )
    script_path = Path(args.gitlab_verifier_script)
    completion_path = Path(args.gitlab_completion_receipt)
    pr_store = Path(args.store)
    repo_list = Path(args.repo_list)
    commit_root = Path(args.commit_root)
    output_root = Path(args.output_root)
    buckets = _parse_buckets(str(args.target_lengths))

    verifier_source = load_verifier_binding(script_path)
    completion = _validated_completion_bundle(
        script_path=script_path,
        receipt_path=completion_path,
        pr_store=pr_store,
        repo_list=repo_list,
        verifier=verifier,
    )
    scan_id = str(completion["binding"]["scan_id"])
    composition = _load_composition(
        args,
        buckets=buckets,
        supplied=source_composition,
    )

    conn: sqlite3.Connection = connect(
        str(_regular_file(pr_store, role="GitLab primary store")),
        create=False,
        readonly=True,
    )
    try:
        membership = build_primary_pr_membership(
            conn,
            source_composition=composition,
            commit_root=commit_root,
            buckets=buckets,
            scan_id=scan_id,
        )
        source = _source_binding(
            composition=composition,
            commit_root=commit_root,
            buckets=buckets,
            membership=membership,
        )

        # Revalidate every mutable path immediately before any output exists.
        if load_verifier_binding(script_path) != verifier_source:
            raise GitLabMembershipBridgeError(
                "GitLab verifier source changed before membership publication"
            )
        before_publish = _validated_completion_bundle(
            script_path=script_path,
            receipt_path=completion_path,
            pr_store=pr_store,
            repo_list=repo_list,
            verifier=verifier,
        )
        if before_publish != completion:
            raise GitLabMembershipBridgeError(
                "GitLab completion changed before membership publication"
            )
        if _source_binding(
            composition=composition,
            commit_root=commit_root,
            buckets=buckets,
            membership=membership,
        ) != source:
            raise GitLabMembershipBridgeError(
                "source composition changed before membership publication"
            )

        published, membership_receipt = publish_primary_pr_membership_inputs(
            conn,
            output_root=output_root,
            membership=membership,
        )
    finally:
        conn.close()

    # A third verification closes the build/publish TOCTOU window.  If it
    # fails, no bridge receipt is emitted and the canonical files remain
    # non-authoritative orphan candidates rather than a false success.
    if load_verifier_binding(script_path) != verifier_source:
        raise GitLabMembershipBridgeError(
            "GitLab verifier source changed during membership publication"
        )
    after_publish = _validated_completion_bundle(
        script_path=script_path,
        receipt_path=completion_path,
        pr_store=pr_store,
        repo_list=repo_list,
        verifier=verifier,
    )
    if after_publish != completion:
        raise GitLabMembershipBridgeError(
            "GitLab completion changed during membership publication"
        )
    if _source_binding(
        composition=composition,
        commit_root=commit_root,
        buckets=buckets,
        membership=published,
    ) != source:
        raise GitLabMembershipBridgeError(
            "source composition changed during membership publication"
        )
    verify_primary_pr_membership_receipt(
        published,
        membership_receipt,
        output_root=output_root,
    )

    bridge_receipt: dict[str, object] = {
        "schema": BRIDGE_RECEIPT_SCHEMA,
        "status": "complete",
        "training_ready_without_export": False,
        "required_next_gate": "lossless_pr_parquet_export_receipt",
        "gitlab_verifier_source": verifier_source,
        "gitlab_completion": completion,
        "source": source,
        "primary_membership": published,
        "primary_membership_receipt": membership_receipt,
        "output_root": str(output_root.expanduser().resolve()),
        "validation": {
            "hardened_gitlab_completion_verified": True,
            "gitlab_completion_revalidated_before_publish": True,
            "gitlab_completion_revalidated_after_publish": True,
            "source_composition_revalidated_before_publish": True,
            "source_composition_revalidated_after_publish": True,
            "canonical_membership_receipt_verified": True,
            "exact_scan_store_membership": True,
        },
    }
    resolved_output = output_root.expanduser().resolve()
    bridge_path = resolved_output / BRIDGE_RECEIPT_NAME
    _atomic_publish_json(bridge_path, bridge_receipt)
    verified, bridge_binding = _read_json_object(
        bridge_path,
        role="GitLab membership bridge receipt",
        canonical=True,
    )
    if verified != bridge_receipt:
        raise GitLabMembershipBridgeError(
            "published GitLab membership bridge receipt drifted"
        )
    return {
        "status": "complete",
        "scan_id": scan_id,
        "selected_pr_count": int(published["selected_pr_count"]),
        "primary_membership": published,
        "primary_membership_receipt": membership_receipt,
        "bridge_receipt": bridge_binding,
        "training_ready": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="Verified GitLab primary SQLite store.")
    parser.add_argument(
        "--gitlab-completion-receipt",
        required=True,
        help="Hardened cppmega_gitlab_mr_completion_v1 receipt.",
    )
    parser.add_argument("--repo-list", required=True, help="Exact GitLab repo scope.")
    parser.add_argument(
        "--gitlab-verifier-script",
        required=True,
        help="Clean cppmega.mlx scripts/pr_ingest/gitlab_mr_stream.py.",
    )
    parser.add_argument(
        "--source-composition",
        required=True,
        help="Sealed cppmega source composition plan.",
    )
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--commit-root", required=True)
    parser.add_argument(
        "--target-lengths",
        required=True,
        help="Sorted comma-separated composed commit buckets, e.g. 1024,...,65536.",
    )
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_gitlab_primary_membership(args)
    except (GitLabMembershipBridgeError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"GITLAB_PRIMARY_MEMBERSHIP_FAILED: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
