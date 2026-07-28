#!/usr/bin/env python3
"""Unified per-repo CONVEYOR: ONE .git-preserving tarball pass, BOTH streams per repo.

This is the single driver that packages the manually-proven orchestration
(apex / abseil-cpp / amrex were run stage-by-stage by hand) into ONE resumable
conveyor. It does NOT reimplement any stage logic -- it imports and orchestrates
the EXACT existing stages from ``streaming_reindex`` (code) and
``streaming_reindex_commits`` (commits + PR-discussion injection).

Per repo, in ONE pass over the ``.git``-preserving tar stream:

  extract cpp_all/<repo>/ (incl .git) -> work/<repo>/_src   (ONE extraction)
    -> CODE pipeline (streaming_reindex.process_one_repo):
         index_project.py --enriched [C/C++ + build files]
           -> clang_enriched_to_parquet (tokenize @65536, materialize)
           -> route_by_fit (--target-lengths-code, default
              1024,2048,4096,8192,16384)
           -> pack_enriched_rows per bucket
           -> outputs/reindexed/<L>/<repo>.parquet   (recompressed zstd-max)
       The SHARED --dedup-db is threaded in (function-level tokenized-hash exact
       + MinHash-LSH near).
    -> COMMITS pipeline (streaming_reindex_commits stages):
         extract_git_history.py once -> <repo>_commits.jsonl (records)
           -> DELETE .git immediately (records captured; bounded disk)
           -> split records into ranges of --range-size, fan to the SAME pool:
                process_commits.py --format both --pr-store --repo-list --dedup-db
                  (PR @discussion injected via PRDiscussionLookup, commit 36c717a)
                -> materialize -> route_by_fit (--target-lengths-commits,
                   default 1024,2048,4096,8192,16384) -> pack
                -> outputs/reindexed_commits/<L>/<repo>_r<start>.parquet (zstd-max)
       The SAME SHARED --dedup-db is threaded in (commit-DOC tokenized-hash).
    -> DELETE the whole repo work dir (bounded disk: ~1 repo on disk at a time).

Resumable: ONE manifest at outputs/conveyor/_done.json. The CODE half is keyed
``<repo>::code``; each COMMIT range is keyed ``<repo>::r<start>`` (exactly the
commit driver's range key). Resume skips completed code-halves and completed
ranges exactly; a repo whose code is done but ranges remain is resumed mid-way.

Fail-loud per repo (RULE #1): every stage runs ONE clear subprocess path and
RAISES RepoFailure on any non-zero exit / missing / empty output. A failure is
recorded in the manifest (``failed``) and the conveyor moves to the next unit --
no degraded / best-effort / silent-fallback path anywhere.

SHARED dedup: ONE --dedup-db (default outputs/dedup_seen.sqlite) is passed to
BOTH the code stage AND every commit range stage, so the user's clever dedup
(function-level for code, doc-level for commits; exact + near) is global across
repos AND across the two streams.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Callable, Iterable, Sequence

# Reuse the proven machinery. streaming_reindex_commits already re-exports the
# shared primitives from streaming_reindex (ROOT, Manifest, RepoFailure, ...)
# and adds the .git-preserving stream + range pipeline. Import BOTH, do not
# duplicate a single line of stage logic.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if __package__:
    from scripts import streaming_reindex as sr  # noqa: E402
    from scripts import streaming_reindex_commits as src  # noqa: E402
else:  # pragma: no cover - exercised by direct file-mode execution.
    import streaming_reindex as sr  # noqa: E402
    import streaming_reindex_commits as src  # noqa: E402
from nanochat_data import extract_git_history as extract_history  # noqa: E402

SymbolIdentityError = sr.SymbolIdentityError

if src.RepoFailure is not sr.RepoFailure or src.Manifest is not sr.Manifest:
    raise RuntimeError(
        "streaming pipeline imported duplicate streaming_reindex module identities"
    )

ROOT = src.ROOT
VENV_PYTHON = src.VENV_PYTHON
RepoFailure = src.RepoFailure
Manifest = src.Manifest
EXTRACT_GIT = src.EXTRACT_GIT
DEFAULT_RANGE_SIZE = src.DEFAULT_RANGE_SIZE
stream_repo_subtrees_with_git = src.stream_repo_subtrees_with_git
get_commit_list = src.get_commit_list
stage_extract_commits = src.stage_extract_commits
process_range = src.process_range
range_key = src.range_key

# Unified conveyor manifest lives in its OWN root so it never collides with the
# per-stream manifests (the code/commit outputs themselves still land in the
# existing outputs/reindexed and outputs/reindexed_commits trees, so already-run
# parquet is reused as-is).
CONVEYOR_ROOT = ROOT / "outputs" / "conveyor"
CONVEYOR_MANIFEST = CONVEYOR_ROOT / "_done.json"

DEFAULT_DEDUP_DB = ROOT / "outputs" / "dedup_seen.sqlite"

DEFAULT_TARGET_LENGTHS_CODE = (1024, 2048, 4096, 8192, 16384)
DEFAULT_TARGET_LENGTHS_COMMITS = (1024, 2048, 4096, 8192, 16384)
DEFAULT_PROGRESS_JSONL = CONVEYOR_ROOT / "progress.jsonl"
DEFAULT_DEDUP_CHECKPOINT_TOKENS = 25_000_000
DEFAULT_RUN_LOCK_DIR = CONVEYOR_ROOT / "locks"
DEFAULT_WORK_PARENT = CONVEYOR_ROOT / "tmp"
DEFAULT_RESERVATION_FILE = CONVEYOR_ROOT / "_reservations.json"
DEFAULT_RANGE_SUBMIT_WINDOW_MULTIPLIER = 2
DEFAULT_MEMORY_BUDGET_FRACTION = 0.55
DEFAULT_MEMORY_BUDGET_FALLBACK_GB = 48.0
DEFAULT_MIN_RETRY_RANGE_SIZE = 25
DEFAULT_RANGE_TARGET_BYTES = 32 * 1024 * 1024
DEFAULT_DEDUP_PROMOTE_BATCH_SIZE = 8
DEFAULT_MIN_FREE_DISK_GB = 50.0

CODE_REVISION_SCHEMA_VERSION = 2
CODE_REVISION_PRODUCER_ROLE = "canonical_source_conveyor"
CODE_REVISION_REPOSITORY_IDENTITY = "cppmega"
CODE_REVISION_DRIFT_MARKER = "CPPMEGA_CODE_REVISION_DRIFT"
CODE_REVISION_DRIFT_EXIT_CODE = 86
CODE_REVISION_MAX_GIT_OUTPUT_BYTES = 64 * 1024 * 1024
CODE_REVISION_MAX_STATUS_BYTES = 4 * 1024 * 1024
CODE_REVISION_MAX_UNTRACKED_FILES = 4096
CODE_REVISION_MAX_UNTRACKED_FILE_BYTES = 8 * 1024 * 1024
CODE_REVISION_MAX_UNTRACKED_TOTAL_BYTES = 32 * 1024 * 1024

# Git pathspecs are deliberately limited to executable source and configuration.
# Corpus data, generated parquet, caches, node_modules, and outputs are never
# traversed or hashed by the revision guard.
CODE_REVISION_PATHS = (
    ":(top)cppmega",
    ":(top)scripts",
    ":(top)tools",
    ":(top)configs",
    ":(top,glob)*.py",
    ":(top,glob)*.pyi",
    ":(top,glob)*.toml",
    ":(top,glob)*.yaml",
    ":(top,glob)*.yml",
    ":(top,glob)requirements*.txt",
    ":(top)uv.lock",
    ":(top)setup.cfg",
    ":(top)setup.py",
    ":(top)Makefile",
    ":(top)CMakeLists.txt",
)

# Run-local, repo-keyed cache for the EXPENSIVE git-history extraction output.
# An explicit --extract-cache-root changes ownership to EXTERNAL: the same
# extractor checkpoint/publication can then be reused across independent
# conveyor roots and is never deleted by conveyor cleanup.
EXTRACT_CACHE_ROOT = CONVEYOR_ROOT / "extract_cache"
EXTRACT_CACHE_MODE_RUN_LOCAL = "run_local"
EXTRACT_CACHE_MODE_EXTERNAL = "external"
EXTRACT_CACHE_MODE = EXTRACT_CACHE_MODE_RUN_LOCAL

_PRINT_LOCK = threading.Lock()


def _handle_source_stream_read_error(
    error: tarfile.ReadError,
    *,
    interrupted: bool,
) -> None:
    if not interrupted:
        raise error
    _log(f"Source stream closed after checkpoint signal: {error}")


class CodeRevisionError(RuntimeError):
    """The production code revision cannot be captured or enforced."""


class CodeRevisionMismatchError(CodeRevisionError):
    """The requested/resumed revision does not match the live checkout."""


class CodeRevisionDriftError(CodeRevisionError):
    """The checkout changed after the conveyor captured its revision."""


class PRCompletionBindingError(RuntimeError):
    """The commit stream is not bound to one verified PR snapshot."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        before = path.stat()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise PRCompletionBindingError(
            f"cannot hash PR completion input {path}: {exc}"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise PRCompletionBindingError(
            f"PR completion input changed while hashing: {path}"
        )
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise PRCompletionBindingError(
            f"PR completion receipt has invalid {field}: {value!r}"
        )
    return value


def load_pr_completion_binding(
    receipt_path: Path,
    *,
    pr_store: Path,
    repo_list: Path,
) -> dict[str, object]:
    """Validate and normalize an immutable, already-verified PR receipt."""

    receipt_path = receipt_path.expanduser().resolve()
    pr_store = pr_store.expanduser().resolve()
    repo_list = repo_list.expanduser().resolve()
    for label, path in (
        ("PR completion receipt", receipt_path),
        ("PR store", pr_store),
        ("PR repo list", repo_list),
    ):
        if not path.is_file():
            raise PRCompletionBindingError(f"{label} is missing: {path}")
    pr_store_wal = Path(f"{pr_store}-wal")
    if pr_store_wal.exists() and pr_store_wal.stat().st_size:
        raise PRCompletionBindingError(
            "PR store has an uncheckpointed WAL and is not the immutable snapshot "
            f"bound by the receipt: {pr_store_wal}"
        )
    receipt_max_bytes = 4 * 1024 * 1024
    try:
        with receipt_path.open("rb") as handle:
            receipt_bytes = handle.read(receipt_max_bytes + 1)
    except OSError as exc:
        raise PRCompletionBindingError(
            f"cannot read PR completion receipt {receipt_path}: {exc}"
        ) from exc
    if len(receipt_bytes) > receipt_max_bytes:
        raise PRCompletionBindingError(
            "PR completion receipt exceeds the 4 MiB metadata bound: "
            f"{receipt_path}"
        )
    try:
        receipt = json.loads(receipt_bytes)
    except json.JSONDecodeError as exc:
        raise PRCompletionBindingError(
            f"PR completion receipt is invalid JSON: {receipt_path}: {exc}"
        ) from exc
    if not isinstance(receipt, dict):
        raise PRCompletionBindingError("PR completion receipt must be an object")
    if receipt.get("schema") != "cppmega_pr_completion_v2":
        raise PRCompletionBindingError(
            f"unsupported PR completion schema: {receipt.get('schema')!r}"
        )
    if receipt.get("status") != "verified":
        raise PRCompletionBindingError(
            f"PR completion status is not verified: {receipt.get('status')!r}"
        )

    def require_bound_artifact(field: str, expected_path: Path) -> str:
        artifact = receipt.get(field)
        if not isinstance(artifact, dict):
            raise PRCompletionBindingError(
                f"PR completion receipt lacks {field} binding"
            )
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str):
            raise PRCompletionBindingError(
                f"PR completion receipt has invalid {field}.path"
            )
        try:
            bound_path = Path(raw_path).expanduser().resolve()
        except OSError as exc:
            raise PRCompletionBindingError(
                f"cannot resolve PR completion {field}.path: {raw_path}: {exc}"
            ) from exc
        if bound_path != expected_path:
            raise PRCompletionBindingError(
                f"PR completion {field} path mismatch: "
                f"receipt={bound_path} requested={expected_path}"
            )
        expected_sha256 = _require_sha256(
            artifact.get("sha256"),
            field=f"{field}.sha256",
        )
        actual_sha256 = _sha256_file(expected_path)
        if actual_sha256 != expected_sha256:
            raise PRCompletionBindingError(
                f"PR completion {field} hash mismatch: "
                f"receipt={expected_sha256} current={actual_sha256}"
            )
        return actual_sha256

    pr_store_sha256 = require_bound_artifact("pr_store", pr_store)
    if pr_store_wal.exists() and pr_store_wal.stat().st_size:
        raise PRCompletionBindingError(
            "PR store WAL appeared while validating the immutable snapshot: "
            f"{pr_store_wal}"
        )
    repo_list_sha256 = require_bound_artifact("repo_list", repo_list)
    expected_repos_sha256 = _require_sha256(
        receipt.get("expected_repos_sha256"),
        field="expected_repos_sha256",
    )
    scan_id = _require_sha256(
        receipt.get("scan_id"),
        field="scan_id",
    )
    expected_repo_count = receipt.get("expected_repo_count")
    stored_pr_count = receipt.get("stored_pr_count")
    declared_pr_count = receipt.get("declared_pr_count")
    unverified_store_pr_count = receipt.get("unverified_store_pr_count")
    if (
        isinstance(expected_repo_count, bool)
        or not isinstance(expected_repo_count, int)
        or expected_repo_count <= 0
    ):
        raise PRCompletionBindingError(
            "PR completion expected_repo_count must be a positive integer"
        )
    if (
        isinstance(stored_pr_count, bool)
        or not isinstance(stored_pr_count, int)
        or stored_pr_count < 0
    ):
        raise PRCompletionBindingError(
            "PR completion stored_pr_count must be a non-negative integer"
        )
    if declared_pr_count != stored_pr_count:
        raise PRCompletionBindingError(
            "PR completion declared_pr_count does not match stored_pr_count"
        )
    if (
        isinstance(unverified_store_pr_count, bool)
        or not isinstance(unverified_store_pr_count, int)
        or unverified_store_pr_count < 0
    ):
        raise PRCompletionBindingError(
            "PR completion unverified_store_pr_count must be a "
            "non-negative integer"
        )
    return {
        "schema": "cppmega_pr_completion_v2",
        "status": "verified",
        "receipt_sha256": _sha256_bytes(receipt_bytes),
        "pr_store_sha256": pr_store_sha256,
        "repo_list_sha256": repo_list_sha256,
        "expected_repos_sha256": expected_repos_sha256,
        "scan_id": scan_id,
        "expected_repo_count": expected_repo_count,
        "stored_pr_count": stored_pr_count,
        "unverified_store_pr_count": unverified_store_pr_count,
    }


def _pr_completion_identity(binding: dict) -> tuple:
    required = (
        "schema",
        "status",
        "receipt_sha256",
        "pr_store_sha256",
        "repo_list_sha256",
        "expected_repos_sha256",
        "scan_id",
        "expected_repo_count",
        "stored_pr_count",
        "unverified_store_pr_count",
    )
    missing = [field for field in required if field not in binding]
    if missing:
        raise PRCompletionBindingError(
            "manifest PR completion binding is missing: " + ", ".join(missing)
        )
    if binding["schema"] != "cppmega_pr_completion_v2":
        raise PRCompletionBindingError(
            f"manifest PR completion schema is unsupported: {binding['schema']!r}"
        )
    if binding["status"] != "verified":
        raise PRCompletionBindingError(
            f"manifest PR completion status is not verified: {binding['status']!r}"
        )
    for field in (
        "receipt_sha256",
        "pr_store_sha256",
        "repo_list_sha256",
        "expected_repos_sha256",
        "scan_id",
    ):
        _require_sha256(binding[field], field=field)
    if (
        isinstance(binding["expected_repo_count"], bool)
        or not isinstance(binding["expected_repo_count"], int)
        or binding["expected_repo_count"] <= 0
    ):
        raise PRCompletionBindingError(
            "manifest PR completion expected_repo_count is invalid"
        )
    if (
        isinstance(binding["stored_pr_count"], bool)
        or not isinstance(binding["stored_pr_count"], int)
        or binding["stored_pr_count"] < 0
    ):
        raise PRCompletionBindingError(
            "manifest PR completion stored_pr_count is invalid"
        )
    if (
        isinstance(binding["unverified_store_pr_count"], bool)
        or not isinstance(binding["unverified_store_pr_count"], int)
        or binding["unverified_store_pr_count"] < 0
    ):
        raise PRCompletionBindingError(
            "manifest PR completion unverified_store_pr_count is invalid"
        )
    return tuple(binding[field] for field in required)


def revalidate_pr_completion_binding(
    binding: dict,
    receipt_path: Path,
    *,
    pr_store: Path,
    repo_list: Path,
) -> None:
    """Prove the commit stream's external PR inputs stayed unchanged."""

    current = load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )
    if _pr_completion_identity(current) != _pr_completion_identity(binding):
        raise PRCompletionBindingError(
            "PR completion binding changed while the commit stream was running"
        )


def _bounded_git_output(
    repo_root: Path,
    args: Sequence[str],
    *,
    max_bytes: int,
) -> bytes:
    """Run one deterministic Git query with bounded captured stdout."""
    env = dict(os.environ)
    env.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    command = [
        "git",
        "-c",
        "color.ui=false",
        "-C",
        str(repo_root),
        *args,
    ]
    with tempfile.TemporaryFile() as stdout_fh:
        proc = subprocess.run(
            command,
            stdout=stdout_fh,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
        size = stdout_fh.tell()
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")[-2000:]
            raise CodeRevisionError(
                f"git revision query failed ({proc.returncode}): "
                f"{' '.join(args)}: {stderr}"
            )
        if size > max_bytes:
            raise CodeRevisionError(
                f"git revision query exceeded {max_bytes} bytes "
                f"({size} bytes): {' '.join(args)}"
            )
        stdout_fh.seek(0)
        return stdout_fh.read()


def _hash_untracked_source_files(repo_root: Path, paths_blob: bytes) -> str:
    """Hash bounded untracked source files, never corpus/output trees."""
    raw_paths = sorted(path for path in paths_blob.split(b"\0") if path)
    if len(raw_paths) > CODE_REVISION_MAX_UNTRACKED_FILES:
        raise CodeRevisionError(
            "relevant untracked source count exceeds revision guard bound: "
            f"{len(raw_paths)} > {CODE_REVISION_MAX_UNTRACKED_FILES}"
        )

    total_bytes = 0
    digest = hashlib.sha256()
    for raw_path in raw_paths:
        relative = Path(os.fsdecode(raw_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise CodeRevisionError(f"unsafe untracked Git path: {relative}")
        path = repo_root / relative
        try:
            file_stat = path.lstat()
        except FileNotFoundError as exc:
            raise CodeRevisionError(
                f"untracked source changed while fingerprinting: {relative}"
            ) from exc

        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(f"{file_stat.st_mode & 0o7777:o}".encode("ascii"))
        digest.update(b"\0")
        if path.is_symlink():
            payload = os.fsencode(os.readlink(path))
            if len(payload) > CODE_REVISION_MAX_UNTRACKED_FILE_BYTES:
                raise CodeRevisionError(
                    f"untracked symlink target exceeds revision guard bound: {relative}"
                )
            total_bytes += len(payload)
            digest.update(b"symlink\0")
            digest.update(payload)
        elif path.is_file():
            if file_stat.st_size > CODE_REVISION_MAX_UNTRACKED_FILE_BYTES:
                raise CodeRevisionError(
                    "untracked source exceeds per-file revision guard bound: "
                    f"{relative} ({file_stat.st_size} bytes)"
                )
            total_bytes += file_stat.st_size
            if total_bytes > CODE_REVISION_MAX_UNTRACKED_TOTAL_BYTES:
                raise CodeRevisionError(
                    "untracked sources exceed total revision guard bound: "
                    f"{total_bytes} bytes"
                )
            digest.update(b"file\0")
            with path.open("rb") as fh:
                while chunk := fh.read(1024 * 1024):
                    digest.update(chunk)
        else:
            raise CodeRevisionError(
                f"unsupported untracked source file type: {relative}"
            )
        digest.update(b"\0")
    return digest.hexdigest()


def _capture_indexer_provenance(repo_root: Path) -> dict[str, object] | None:
    """Capture the exact clang-indexer source and imported local closure.

    Small unit-test repositories may omit the production indexer so the generic
    revision helper remains usable. Production guards reject that omission.
    """

    indexer_path = repo_root / "tools" / "clang_indexer" / "index_project.py"
    if not indexer_path.is_file() or indexer_path.is_symlink():
        return None
    try:
        from cppmega.prompt_graph_provenance import indexer_dependency_hash
    except ImportError as exc:
        raise CodeRevisionError(
            "cannot import the authoritative clang indexer provenance helper"
        ) from exc
    dependency_manifest, dependency_sha256 = indexer_dependency_hash(
        indexer_path,
        repo_root,
    )
    return {
        "schema": "cppmega_indexer_dependency_binding_v1",
        "path": "tools/clang_indexer/index_project.py",
        "source_sha256": _sha256_bytes(indexer_path.read_bytes()),
        "dependency_closure_sha256": dependency_sha256,
        "dependency_manifest": dict(sorted(dependency_manifest.items())),
    }


def capture_code_revision(repo_root: Path = ROOT) -> dict:
    """Capture HEAD plus a bounded source/config-only dirty-tree identity."""
    repo_root = repo_root.resolve()
    head_before = _bounded_git_output(
        repo_root,
        ("rev-parse", "--verify", "HEAD^{commit}"),
        max_bytes=256,
    ).decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40}", head_before) is None:
        raise CodeRevisionError(f"git returned a non-exact HEAD: {head_before!r}")

    path_args = ("--", *CODE_REVISION_PATHS)
    status = _bounded_git_output(
        repo_root,
        (
            "status",
            "--porcelain=v2",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
            *path_args,
        ),
        max_bytes=CODE_REVISION_MAX_STATUS_BYTES,
    )
    index_diff = _bounded_git_output(
        repo_root,
        (
            "diff",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            "HEAD",
            *path_args,
        ),
        max_bytes=CODE_REVISION_MAX_GIT_OUTPUT_BYTES,
    )
    worktree_diff = _bounded_git_output(
        repo_root,
        (
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            *path_args,
        ),
        max_bytes=CODE_REVISION_MAX_GIT_OUTPUT_BYTES,
    )
    untracked_paths = _bounded_git_output(
        repo_root,
        ("ls-files", "--others", "--exclude-standard", "-z", *path_args),
        max_bytes=CODE_REVISION_MAX_STATUS_BYTES,
    )
    tracked_source_tree = _bounded_git_output(
        repo_root,
        (
            "ls-files",
            "--cached",
            "--stage",
            "-z",
            *path_args,
        ),
        max_bytes=CODE_REVISION_MAX_GIT_OUTPUT_BYTES,
    )
    untracked_sha256 = _hash_untracked_source_files(repo_root, untracked_paths)
    head_after = _bounded_git_output(
        repo_root,
        ("rev-parse", "--verify", "HEAD^{commit}"),
        max_bytes=256,
    ).decode("ascii").strip()
    if head_after != head_before:
        raise CodeRevisionError(
            "HEAD changed while the conveyor captured its code revision: "
            f"{head_before} -> {head_after}"
        )
    indexer_provenance = _capture_indexer_provenance(repo_root)

    components = {
        "status_sha256": _sha256_bytes(status),
        "index_diff_sha256": _sha256_bytes(index_diff),
        "worktree_diff_sha256": _sha256_bytes(worktree_diff),
        "untracked_sources_sha256": untracked_sha256,
        "source_tree_sha256": _sha256_bytes(tracked_source_tree),
        "indexer_dependency_closure_sha256": (
            None
            if indexer_provenance is None
            else indexer_provenance["dependency_closure_sha256"]
        ),
    }
    dirty_fingerprint = _sha256_bytes(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    relevant_scope_sha256 = _sha256_bytes(
        b"\0".join(path.encode("utf-8") for path in CODE_REVISION_PATHS)
    )
    return {
        "schema_version": CODE_REVISION_SCHEMA_VERSION,
        "producer_role": CODE_REVISION_PRODUCER_ROLE,
        "repository_identity": CODE_REVISION_REPOSITORY_IDENTITY,
        "git_commit": head_before,
        "dirty": bool(status),
        "dirty_fingerprint": dirty_fingerprint,
        "index_dirty": bool(index_diff),
        "worktree_dirty": bool(worktree_diff),
        "untracked_source_dirty": bool(untracked_paths),
        "relevant_scope_sha256": relevant_scope_sha256,
        "relevant_pathspecs": list(CODE_REVISION_PATHS),
        "bounds": {
            "max_git_output_bytes": CODE_REVISION_MAX_GIT_OUTPUT_BYTES,
            "max_status_bytes": CODE_REVISION_MAX_STATUS_BYTES,
            "max_untracked_files": CODE_REVISION_MAX_UNTRACKED_FILES,
            "max_untracked_file_bytes": CODE_REVISION_MAX_UNTRACKED_FILE_BYTES,
            "max_untracked_total_bytes": CODE_REVISION_MAX_UNTRACKED_TOTAL_BYTES,
        },
        "indexer_provenance": indexer_provenance,
        **components,
    }


def _code_revision_identity(receipt: dict) -> tuple:
    required = (
        "schema_version",
        "producer_role",
        "repository_identity",
        "git_commit",
        "dirty",
        "dirty_fingerprint",
        "relevant_scope_sha256",
        "source_tree_sha256",
        "indexer_dependency_closure_sha256",
    )
    missing = [key for key in required if key not in receipt]
    if missing:
        raise CodeRevisionMismatchError(
            "manifest code revision receipt is missing: " + ", ".join(missing)
        )
    return tuple(receipt[key] for key in required)


def _dirty_component_names(snapshot: dict) -> list[str]:
    return [
        name
        for name, key in (
            ("index", "index_dirty"),
            ("worktree", "worktree_dirty"),
            ("untracked source", "untracked_source_dirty"),
        )
        if snapshot.get(key)
    ]


class CodeRevisionGuard:
    """Fail-closed production pin for every conveyor stage submission."""

    def __init__(
        self,
        repo_root: Path,
        snapshot: dict,
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.snapshot = json.loads(json.dumps(snapshot))
        self.stop_event = stop_event
        self._verify_lock = threading.Lock()

    @classmethod
    def for_production(
        cls,
        expected_code_revision: str | None,
        *,
        repo_root: Path = ROOT,
        stop_event: threading.Event | None = None,
    ) -> "CodeRevisionGuard":
        if expected_code_revision is None:
            raise CodeRevisionMismatchError(
                "--expected-code-revision is required for production conveyor runs"
            )
        expected = expected_code_revision.strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
            raise CodeRevisionMismatchError(
                "--expected-code-revision must be an exact 40-character Git commit"
            )
        snapshot = capture_code_revision(repo_root)
        if snapshot["git_commit"] != expected:
            raise CodeRevisionMismatchError(
                "expected code revision does not match HEAD: "
                f"expected={expected} actual={snapshot['git_commit']}"
            )
        if snapshot.get("indexer_provenance") is None:
            raise CodeRevisionMismatchError(
                "production conveyor requires the authoritative clang indexer "
                "dependency provenance under tools/clang_indexer"
            )
        if snapshot["dirty"]:
            dirty_parts = ", ".join(_dirty_component_names(snapshot)) or "unknown"
            raise CodeRevisionMismatchError(
                "production conveyor requires a clean executable source/config "
                f"worktree; dirty components: {dirty_parts}; "
                f"fingerprint={snapshot['dirty_fingerprint']}"
            )
        return cls(repo_root, snapshot, stop_event=stop_event)

    @property
    def git_commit(self) -> str:
        return str(self.snapshot["git_commit"])

    @property
    def receipt(self) -> dict:
        return {
            **json.loads(json.dumps(self.snapshot)),
            "repo_root": str(self.repo_root),
            "clean_worktree_required": True,
            "expected_code_revision": self.git_commit,
            "child_python_preflight": "sitecustomize",
        }

    def verify(self, submission: str) -> None:
        """Verify the pinned clean state immediately before one submission."""
        try:
            with self._verify_lock:
                head = _bounded_git_output(
                    self.repo_root,
                    ("rev-parse", "--verify", "HEAD^{commit}"),
                    max_bytes=256,
                ).decode("ascii").strip()
                status = _bounded_git_output(
                    self.repo_root,
                    (
                        "status",
                        "--porcelain=v2",
                        "-z",
                        "--untracked-files=all",
                        "--ignore-submodules=none",
                        "--",
                        *CODE_REVISION_PATHS,
                    ),
                    max_bytes=CODE_REVISION_MAX_STATUS_BYTES,
                )
        except CodeRevisionError as exc:
            self._raise_drift(submission, f"verification failed: {exc}")
        if head == self.git_commit and not status:
            return

        details = [f"HEAD {self.git_commit} -> {head}"] if head != self.git_commit else []
        if status:
            try:
                current = capture_code_revision(self.repo_root)
                details.extend(_dirty_component_names(current))
                details.append(f"fingerprint={current['dirty_fingerprint']}")
            except CodeRevisionError as exc:
                details.append(f"dirty-state capture failed: {exc}")
        self._raise_drift(submission, "; ".join(details) or "unknown revision drift")

    def _raise_drift(self, submission: str, detail: str) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        raise CodeRevisionDriftError(
            f"{CODE_REVISION_DRIFT_MARKER}: code revision drift before "
            f"{submission}: {detail}"
        )


def _write_atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def install_code_revision_child_guard(
    guard: CodeRevisionGuard,
    conveyor_root: Path,
) -> Path:
    """Install an immutable child-start check and prepend it to PYTHONPATH."""
    shadow = guard.repo_root / "sitecustomize.py"
    if shadow.exists():
        raise CodeRevisionError(
            f"repository sitecustomize.py would shadow revision guard: {shadow}"
        )
    guard_dir = (
        conveyor_root
        / "code_revision_guard"
        / f"{guard.git_commit}-{guard.snapshot['dirty_fingerprint'][:16]}"
    )
    sitecustomize_path = guard_dir / "sitecustomize.py"
    source = f'''# Generated by streaming_conveyor.py; do not edit during a run.
import hashlib
import os
import subprocess
import sys

_EXPECTED = {guard.git_commit!r}
_REPO_ROOT = {str(guard.repo_root)!r}
_PATHS = {CODE_REVISION_PATHS!r}
_MARKER = {CODE_REVISION_DRIFT_MARKER!r}
_EXIT_CODE = {CODE_REVISION_DRIFT_EXIT_CODE}
_MAX_STATUS = {CODE_REVISION_MAX_STATUS_BYTES}


def _fail(detail):
    sys.stderr.write(f"{{_MARKER}}: {{detail}}\\n")
    sys.stderr.flush()
    os._exit(_EXIT_CODE)


def _git(args, limit):
    env = dict(os.environ)
    env.update({{"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C", "LANG": "C"}})
    proc = subprocess.run(
        ["git", "-c", "color.ui=false", "-C", _REPO_ROOT, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        _fail("git preflight failed: " + proc.stderr.decode("utf-8", "replace")[-1000:])
    if len(proc.stdout) > limit:
        _fail(f"git preflight output exceeded {{limit}} bytes")
    return proc.stdout


_head = _git(("rev-parse", "--verify", "HEAD^{{commit}}"), 256).decode("ascii").strip()
_status = _git(
    (
        "status",
        "--porcelain=v2",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
        "--",
        *_PATHS,
    ),
    _MAX_STATUS,
)
if _head != _EXPECTED:
    _fail(f"HEAD drift: expected={{_EXPECTED}} actual={{_head}}")
if _status:
    _fail(
        "tracked/index/untracked source drift: status_sha256="
        + hashlib.sha256(_status).hexdigest()
    )
'''
    _write_atomic_text(sitecustomize_path, source)
    launch_receipt = {
        "schema_version": 1,
        "code_revision": guard.receipt,
        "sitecustomize": str(sitecustomize_path),
        "sitecustomize_sha256": _sha256_bytes(source.encode("utf-8")),
        "drift_marker": CODE_REVISION_DRIFT_MARKER,
        "drift_exit_code": CODE_REVISION_DRIFT_EXIT_CODE,
    }
    _write_atomic_text(
        guard_dir / "launch_receipt.json",
        json.dumps(launch_receipt, indent=2, sort_keys=True) + "\n",
    )
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        str(guard_dir)
        if not existing
        else os.pathsep.join((str(guard_dir), existing))
    )
    return sitecustomize_path


def _raise_if_revision_subprocess_failure(exc: RepoFailure) -> None:
    if CODE_REVISION_DRIFT_MARKER in exc.detail:
        raise CodeRevisionDriftError(exc.detail) from exc


def configure_output_roots(
    *,
    code_output_root: str | os.PathLike[str] | None = None,
    commit_output_root: str | os.PathLike[str] | None = None,
    conveyor_root: str | os.PathLike[str] | None = None,
    extract_cache_root: str | os.PathLike[str] | None = None,
) -> None:
    """Rebase all runtime output roots used by the conveyor.

    The conveyor orchestrates code and commit drivers that historically kept
    module-level output constants. This explicit production seam lets a full
    regeneration run write to a fresh tree instead of mixing new packed shards
    with legacy ``outputs/reindexed*`` files.
    """

    global CONVEYOR_ROOT
    global CONVEYOR_MANIFEST
    global DEFAULT_PROGRESS_JSONL
    global DEFAULT_RUN_LOCK_DIR
    global DEFAULT_WORK_PARENT
    global DEFAULT_RESERVATION_FILE
    global EXTRACT_CACHE_ROOT
    global EXTRACT_CACHE_MODE

    if code_output_root is not None:
        sr.OUTPUT_ROOT = Path(code_output_root)
        sr.MANIFEST_PATH = sr.OUTPUT_ROOT / "_done.json"
    if commit_output_root is not None:
        src.COMMIT_OUTPUT_ROOT = Path(commit_output_root)
        src.COMMIT_MANIFEST = src.COMMIT_OUTPUT_ROOT / "_done.json"
    if conveyor_root is not None:
        CONVEYOR_ROOT = Path(conveyor_root)
        CONVEYOR_MANIFEST = CONVEYOR_ROOT / "_done.json"
        DEFAULT_PROGRESS_JSONL = CONVEYOR_ROOT / "progress.jsonl"
        DEFAULT_RUN_LOCK_DIR = CONVEYOR_ROOT / "locks"
        DEFAULT_WORK_PARENT = CONVEYOR_ROOT / "tmp"
        DEFAULT_RESERVATION_FILE = CONVEYOR_ROOT / "_reservations.json"
        EXTRACT_CACHE_ROOT = CONVEYOR_ROOT / "extract_cache"
        EXTRACT_CACHE_MODE = EXTRACT_CACHE_MODE_RUN_LOCAL
    if extract_cache_root is not None:
        EXTRACT_CACHE_ROOT = Path(extract_cache_root).expanduser().resolve()
        EXTRACT_CACHE_MODE = EXTRACT_CACHE_MODE_EXTERNAL


def configure_runtime_paths_from_args(args: argparse.Namespace) -> None:
    old_defaults = {
        "progress_jsonl": DEFAULT_PROGRESS_JSONL,
        "run_lock_dir": DEFAULT_RUN_LOCK_DIR,
        "work_parent_dir": DEFAULT_WORK_PARENT,
        "reservation_file": DEFAULT_RESERVATION_FILE,
    }
    configure_output_roots(
        code_output_root=args.code_output_root,
        commit_output_root=args.commit_output_root,
        conveyor_root=args.conveyor_root,
        extract_cache_root=args.extract_cache_root,
    )

    if Path(args.progress_jsonl) == old_defaults["progress_jsonl"]:
        args.progress_jsonl = str(DEFAULT_PROGRESS_JSONL)
    if Path(args.run_lock_dir) == old_defaults["run_lock_dir"]:
        args.run_lock_dir = str(DEFAULT_RUN_LOCK_DIR)
    if Path(args.work_parent_dir) == old_defaults["work_parent_dir"]:
        args.work_parent_dir = str(DEFAULT_WORK_PARENT)
    if Path(args.reservation_file) == old_defaults["reservation_file"]:
        args.reservation_file = str(DEFAULT_RESERVATION_FILE)


class RepoNoCommitRecords(RuntimeError):
    """Commit stream completed discovery but has no trainable commit records."""

    def __init__(self, repo: str, *, reason: str, detail: str):
        super().__init__(f"[{repo}] no commit records: {reason}: {detail}")
        self.repo = repo
        self.reason = reason
        self.detail = detail


# Cooperative shutdown flag set by the SIGINT/SIGTERM handlers in main(). When
# set, the conveyor stops SUBMITTING new repos/ranges, lets in-flight subprocess
# tasks drain, and cancels queued-but-unstarted range futures (their units stay
# un-marked so resume re-runs them). A SECOND signal forces an immediate exit.
STOP_EVENT = threading.Event()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, file=sys.stderr, flush=True)


class BackgroundRecompressor:
    """Track deferred parquet recompress jobs and surface failures at shutdown."""

    def __init__(
        self,
        max_workers: int,
        revision_guard: CodeRevisionGuard | None = None,
    ) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._lock = threading.Lock()
        self._futures: list[tuple[Path, Future]] = []
        self._handled: set[int] = set()
        self._revision_guard = revision_guard

    def _recompress(self, path: Path) -> None:
        if self._revision_guard is not None:
            self._revision_guard.verify(f"background recompress {path}")
        src.recompress_zstd_max(path)

    def submit(self, path: Path) -> Future:
        if self._revision_guard is not None:
            self._revision_guard.verify(f"submit background recompress {path}")
        fut = self._pool.submit(self._recompress, path)
        with self._lock:
            self._futures.append((path, fut))
        return fut

    def wait(self, jobs: Sequence[tuple[Path, Future]]) -> None:
        failures: list[str] = []
        try:
            for path, fut in jobs:
                try:
                    fut.result()
                except SymbolIdentityError:
                    raise
                except Exception as exc:
                    failures.append(f"{path}: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._handled.update(id(fut) for _path, fut in jobs)
        if failures:
            raise RuntimeError(
                "background code parquet recompress failed:\n"
                + "\n".join(failures[:20])
            )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        failures: list[str] = []
        with self._lock:
            futures = [
                (path, fut)
                for path, fut in self._futures
                if id(fut) not in self._handled
            ]
        for path, fut in futures:
            try:
                fut.result()
            except SymbolIdentityError:
                raise
            except Exception as exc:
                failures.append(f"{path}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError(
                "background code parquet recompress failed:\n"
                + "\n".join(failures[:20])
            )


def disk_free_gb(path: Path) -> float:
    """Return free space on the filesystem containing ``path`` in GiB."""
    probe = path if path.exists() else path.parent
    usage = shutil.disk_usage(probe)
    return float(usage.free) / (1024**3)


def ensure_min_free_disk(path: Path, min_free_gb: float, *, context: str) -> None:
    """Fail loud before staging more raw source when the work disk is too full."""
    threshold = float(min_free_gb or 0.0)
    if threshold <= 0.0:
        return
    free = disk_free_gb(path)
    if free < threshold:
        raise SystemExit(
            f"unsafe conveyor disk state before {context}: "
            f"{free:.2f} GiB free under {path}, "
            f"minimum required is {threshold:.2f} GiB. "
            "Clean outputs/conveyor/tmp or lower --min-free-disk-gb explicitly."
        )


def remove_tree(path: Path, *, reason: str) -> None:
    """Best-effort tree removal with a single useful log line."""
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)
    _log(f"CLEANUP {reason}: removed {path}")


def should_remove_owned_work_root(
    *, own_work_root: bool, keep_temp: bool, retain_partial_work: bool
) -> bool:
    """Honor both explicit retention modes at the parent work-root boundary."""
    return own_work_root and not keep_temp and not retain_partial_work


def physical_memory_gb() -> float | None:
    """Best-effort physical RAM in GiB using only stdlib/system tools."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and pages > 0:
            return float(page_size * pages) / (1024**3)
    except (AttributeError, OSError, ValueError):
        pass
    if sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(int(proc.stdout.strip())) / (1024**3)
        except (OSError, subprocess.CalledProcessError, ValueError):
            return None
    return None


def default_memory_budget_gb() -> float:
    total = physical_memory_gb()
    if total is None:
        return DEFAULT_MEMORY_BUDGET_FALLBACK_GB
    return max(8.0, total * DEFAULT_MEMORY_BUDGET_FRACTION)


def heavy_subprocess_slots(*, streams: str, workers: int, repo_workers: int) -> int:
    """Conservative count of concurrently alive heavy stage subprocesses.

    Commit range workers spawn process_commits/materialize/pack. Code repo workers
    spawn index_project/materialize/pack. The exact stage mix changes over time,
    but this bound catches configurations like 20 repo workers + 16 range workers
    before they can launch hundreds of GiB worth of native clang processes.
    """
    slots = 0
    if streams in {"both", "commits"}:
        slots += max(1, int(workers))
    if streams in {"both", "code"}:
        slots += max(1, int(repo_workers))
    return max(1, slots)


def validate_memory_plan(
    *,
    streams: str,
    workers: int,
    repo_workers: int,
    memory_limit_gb: float,
    code_memory_limit_gb: float | None = None,
    commit_memory_limit_gb: float | None = None,
    memory_budget_gb: float,
    allow_oversubscription: bool,
) -> dict[str, float | int | bool]:
    """Fail before extraction when requested parallelism cannot fit RAM."""
    stage_limit = max(0.0, float(memory_limit_gb))
    code_limit = (
        stage_limit if code_memory_limit_gb is None
        else max(0.0, float(code_memory_limit_gb))
    )
    commit_limit = (
        stage_limit if commit_memory_limit_gb is None
        else max(0.0, float(commit_memory_limit_gb))
    )
    commit_slots = max(1, int(workers)) if streams in {"both", "commits"} else 0
    code_slots = max(1, int(repo_workers)) if streams in {"both", "code"} else 0
    slots = max(1, commit_slots + code_slots)
    budget = max(0.0, float(memory_budget_gb))
    code_reserved = code_slots * code_limit
    commit_reserved = commit_slots * commit_limit
    reserved = code_reserved + commit_reserved
    plan = {
        "heavy_slots": slots,
        "code_heavy_slots": code_slots,
        "commit_heavy_slots": commit_slots,
        "memory_limit_gb": stage_limit,
        "code_memory_limit_gb": code_limit,
        "commit_memory_limit_gb": commit_limit,
        "code_reserved_gb": code_reserved,
        "commit_reserved_gb": commit_reserved,
        "memory_budget_gb": budget,
        "reserved_gb": reserved,
        "allow_oversubscription": bool(allow_oversubscription),
    }
    if budget > 0 and reserved > budget and not allow_oversubscription:
        raise SystemExit(
            "unsafe conveyor memory plan: "
            f"heavy_slots={slots} * --memory-limit-gb={stage_limit:.2f} "
            f"= {reserved:.2f} GiB exceeds --memory-budget-gb={budget:.2f}. "
            "Lower --workers/--repo-workers/--memory-limit-gb, increase the "
            "budget, or pass --allow-memory-oversubscription explicitly."
        )
    return plan


# --------------------------------------------------------------------------- #
# Extract checkpoint: never re-run the (~6h) extract_git_history for a repo whose
# commit extraction already completed. Run-local entries keep the legacy .done
# sentinel; external entries use the extractor's source-bound publication state.
# --------------------------------------------------------------------------- #
def extract_cache_dir(repo: str) -> Path:
    """Repo-keyed directory holding commit JSONL plus validation metadata."""
    return EXTRACT_CACHE_ROOT / repo


def extract_cache_config_receipt() -> dict:
    """Serializable cache ownership recorded in run and repository receipts."""
    return {
        "root": str(EXTRACT_CACHE_ROOT.resolve()),
        "mode": EXTRACT_CACHE_MODE,
    }


def extract_cache_is_external() -> bool:
    return EXTRACT_CACHE_MODE == EXTRACT_CACHE_MODE_EXTERNAL


def extract_cache_access_receipt(status: str) -> dict:
    reused = status in {
        "hit",
        "checkpoint_resume",
        "adopt",
        "back_compat",
        "orphan_adopt",
        "hit_legacy_identity_override",
    }
    receipt = {
        **extract_cache_config_receipt(),
        "status": status,
        "hit": status == "hit",
        "reused": reused,
    }
    if status == "hit_legacy_identity_override":
        receipt["legacy_identity_override"] = True
    return receipt


@contextmanager
def extract_cache_repo_lock(repo: str):
    """Serialize validation/publication of one external entry across processes."""
    path = extract_cache_dir(repo) / ".conveyor-cache.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _extract_sentinel_path(jsonl: Path) -> Path:
    return Path(str(jsonl) + ".done")


def _extract_transaction_checkpoint_path(jsonl: Path) -> Path:
    return Path(str(jsonl) + ".extract-checkpoint")


def has_resumable_extract_checkpoint(repo: str) -> bool:
    """True when the extractor has durable range state worth retaining."""
    jsonl = extract_cache_dir(repo) / f"{repo}_commits.jsonl"
    root = _extract_transaction_checkpoint_path(jsonl)
    if not root.is_dir():
        return False
    if (root / "publication.json").is_file():
        return True
    return any(root.glob("repo-*/checkpoint.sqlite3"))


def _count_jsonl_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for _ in fh:
            n += 1
    return n


def _write_extract_sentinel(jsonl: Path, n_records: int) -> Path:
    """Atomically stamp the completion sentinel next to ``jsonl``.

    Records line_count plus the jsonl's size+mtime so a later resume can validate
    the cache with a cheap stat() instead of re-reading the (10GB) file.
    """
    st = jsonl.stat()
    sentinel = _extract_sentinel_path(jsonl)
    payload = {
        "jsonl": str(jsonl),
        "line_count": int(n_records),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = sentinel.with_name(f"{sentinel.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(sentinel)  # atomic on POSIX
    return sentinel


def _read_valid_sentinel(jsonl: Path) -> int | None:
    """Return the recorded record count iff a sentinel matches ``jsonl`` exactly.

    A match requires the jsonl to exist non-empty AND its current size+mtime to
    equal the stamped values. Any mismatch (truncated by a kill mid-extract,
    corrupt sentinel, missing file) returns None -> the caller re-extracts the
    full records (the one clear path; NOT a degraded/partial output).
    """
    sentinel = _extract_sentinel_path(jsonl)
    if not jsonl.exists() or not sentinel.exists():
        return None
    st = jsonl.stat()
    if st.st_size == 0:
        return None
    try:
        meta = json.loads(sentinel.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if int(meta.get("size_bytes", -1)) != int(st.st_size):
        return None
    if int(meta.get("mtime_ns", -1)) != int(st.st_mtime_ns):
        return None
    lc = meta.get("line_count")
    return int(lc) if lc is not None else None


def _external_cache_state(repo: str, jsonl: Path) -> str:
    """Classify metadata state without accepting a JSONL by name or stat alone."""
    checkpoint_root = _extract_transaction_checkpoint_path(jsonl)
    publication_path = checkpoint_root / "publication.json"
    if publication_path.exists():
        try:
            publication = json.loads(publication_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RepoFailure(
                repo,
                "extract_cache_validate",
                f"invalid external extraction publication {publication_path}: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if not isinstance(publication, dict):
            raise RepoFailure(
                repo,
                "extract_cache_validate",
                f"external extraction publication must be an object: "
                f"{publication_path}",
            )
        publication_status = publication.get("status")
        if publication_status == "corrupt":
            raise RepoFailure(
                repo,
                "extract_cache_validate",
                f"external extraction publication is marked corrupt: "
                f"{publication_path}: {publication.get('corruption', '<no detail>')}",
            )
        if publication_status not in {"done", "failed_partial"}:
            raise RepoFailure(
                repo,
                "extract_cache_validate",
                f"invalid external extraction publication status in "
                f"{publication_path}: {publication_status!r}",
            )
        return "published" if publication_status == "done" else "checkpoint"

    checkpoint_dbs = (
        list(checkpoint_root.glob("repo-*/checkpoint.sqlite3"))
        if checkpoint_root.is_dir()
        else []
    )
    if checkpoint_dbs:
        return "checkpoint"

    unvalidated = [
        path
        for path in (jsonl, _extract_sentinel_path(jsonl))
        if path.exists()
    ]
    if unvalidated:
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            "external cache contains output without extractor checkpoint/publication "
            "metadata; refusing filename-only reuse or automatic replacement: "
            + ", ".join(str(path) for path in unvalidated),
        )
    if checkpoint_root.exists() and not checkpoint_root.is_dir():
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"external extraction checkpoint root is not a directory: "
            f"{checkpoint_root}",
        )
    return "miss"


def _read_completed_external_publication(repo: str, jsonl: Path) -> int:
    """Read count/path/size from the publication validated by the extractor."""
    publication_path = (
        _extract_transaction_checkpoint_path(jsonl) / "publication.json"
    )
    try:
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"missing or invalid completed external extraction publication "
            f"{publication_path}: {type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(publication, dict):
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"external extraction publication must be an object: {publication_path}",
        )
    output = publication.get("output")
    if publication.get("status") != "done" or not isinstance(output, dict):
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"external extraction did not publish a completed receipt: "
            f"{publication_path}",
        )
    try:
        line_count = int(output["line_count"])
        size_bytes = int(output["size_bytes"])
        output_path = Path(str(publication["output_path"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"incomplete external extraction publication {publication_path}: {exc}",
        ) from exc
    if line_count <= 0 or size_bytes <= 0:
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"invalid external extraction counts in {publication_path}: {output}",
        )
    if output_path.resolve() != jsonl.resolve():
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"external extraction publication path mismatch: metadata={output_path} "
            f"requested={jsonl}",
        )
    try:
        actual_size = jsonl.stat().st_size
    except OSError as exc:
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"cannot stat published external extraction {jsonl}: {exc}",
        ) from exc
    if actual_size != size_bytes:
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"external extraction size changed after validation: {jsonl}: "
            f"metadata={size_bytes} actual={actual_size}",
        )
    return line_count


def _validate_completed_external_cache(
    repo: str,
    repo_dir: Path,
    jsonl: Path,
    project_id: str,
) -> int:
    """Validate a hit with the canonical or explicitly legacy source contract."""
    policy = os.environ.get("CPPMEGA_EXTRACT_BAD_UNIT_POLICY", "fail")
    max_bad_units_raw = os.environ.get("CPPMEGA_EXTRACT_MAX_BAD_UNITS", "0")
    try:
        max_bad_units = int(max_bad_units_raw)
    except ValueError as exc:
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            "CPPMEGA_EXTRACT_MAX_BAD_UNITS must be an integer, got "
            f"{max_bad_units_raw!r}",
        ) from exc
    if (
        policy not in {"fail", "quarantine"}
        or (policy == "fail" and max_bad_units != 0)
        or (policy == "quarantine" and max_bad_units <= 0)
    ):
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"invalid extraction failure contract: policy={policy!r} "
            f"max_bad_units={max_bad_units}",
        )
    try:
        canonical_source = extract_history._repo_source_context(
            str(repo_dir),
            max_commits=0,
            repo_name=project_id,
            notes="auto",
        )
        canonical_fingerprint = extract_history._job_fingerprint(
            [canonical_source],
            checkpoint_commits=extract_history.DEFAULT_CHECKPOINT_COMMITS,
            bad_unit_policy=policy,
            max_bad_units=max_bad_units,
        )
        checkpoint_root = _extract_transaction_checkpoint_path(jsonl)
        publication_state = extract_history._load_publication_state(checkpoint_root)
        if publication_state is None:
            completed = None
        elif publication_state.get("job_fingerprint") == canonical_fingerprint:
            completed = extract_history._completed_publication(
                output_path=jsonl,
                checkpoint_root=checkpoint_root,
                job_fingerprint=canonical_fingerprint,
            )
        else:
            legacy_source = extract_history._repo_source_context(
                str(repo_dir),
                max_commits=0,
                repo_name=repo_dir.name,
                notes="auto",
            )
            legacy_fingerprint = extract_history._job_fingerprint(
                [legacy_source],
                checkpoint_commits=extract_history.DEFAULT_CHECKPOINT_COMMITS,
                bad_unit_policy=policy,
                max_bad_units=max_bad_units,
            )
            if publication_state.get("job_fingerprint") != legacy_fingerprint:
                completed = extract_history._completed_publication(
                    output_path=jsonl,
                    checkpoint_root=checkpoint_root,
                    job_fingerprint=canonical_fingerprint,
                )
            else:
                completed = extract_history._completed_publication(
                    output_path=jsonl,
                    checkpoint_root=checkpoint_root,
                    job_fingerprint=legacy_fingerprint,
                )
    except Exception as exc:
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"external cache source/publication validation failed for {jsonl}: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if completed is None:
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"external cache publication is not complete: {jsonl}",
        )
    n_records = int(completed["output"]["line_count"])
    if n_records <= 0:
        raise RepoFailure(
            repo,
            "extract_cache_validate",
            f"completed external cache has no records: {jsonl}",
        )
    return n_records


def _cache_first_record_needs_identity_override(
    jsonl: Path,
    *,
    project_id: str,
) -> bool:
    """Inspect one record so a legacy cache hit is visible in the receipt."""
    try:
        with jsonl.open("r", encoding="utf-8", errors="replace") as handle:
            line = next((raw for raw in handle if raw.strip()), "")
        record = json.loads(line)
    except (OSError, json.JSONDecodeError) as exc:
        raise RepoFailure(
            jsonl.parent.name,
            "extract_cache_validate",
            f"cannot inspect first cached commit identity in {jsonl}: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(record, dict):
        raise RepoFailure(
            jsonl.parent.name,
            "extract_cache_validate",
            f"first cached commit record is not an object: {jsonl}",
        )
    raw_repo = record.get("repo")
    if isinstance(raw_repo, str) and "/" in raw_repo:
        parsed = sr.require_project_identity(
            raw_repo,
            source=f"cached commit record {jsonl}",
        )
        if parsed != project_id:
            raise RepoFailure(
                jsonl.parent.name,
                "project_identity",
                f"cached commit project identity {parsed!r} conflicts with "
                f"resolved {project_id!r}",
            )
    filepath = record.get("filepath")
    expected_repo_id = extract_history.stable_repo_id(project_id)
    expected_filepath_id = (
        extract_history.stable_filepath_id(project_id, filepath)
        if isinstance(filepath, str) and filepath
        else None
    )
    return (
        raw_repo != project_id
        or record.get("repo_stable_id") != expected_repo_id
        or record.get("filepath_stable_id") != expected_filepath_id
    )


def _resolve_commit_project_id(
    repo: str,
    repo_dir: Path,
    project_id: str | None,
) -> str | None:
    if project_id is not None:
        return sr.require_project_identity(
            project_id,
            source=f"commit conveyor project identity for {repo}",
        )
    if "/" in repo:
        return sr.require_project_identity(repo, source=f"commit repo {repo}")
    remote_identity = extract_history.resolve_owner_repo(str(repo_dir))
    if remote_identity is not None:
        return sr.require_project_identity(
            remote_identity,
            source=f"git remote project identity for {repo}",
        )
    return None


def _ensure_external_commit_records(
    repo: str,
    repo_dir: Path,
    *,
    revision_guard: CodeRevisionGuard | None,
    project_id: str | None = None,
) -> tuple[Path, int, str]:
    """Validate/reuse or atomically publish one externally owned cache entry."""
    cache_dir = extract_cache_dir(repo)
    cache_jsonl = cache_dir / f"{repo}_commits.jsonl"
    canonical_project_id = _resolve_commit_project_id(repo, repo_dir, project_id)
    with extract_cache_repo_lock(repo):
        state = _external_cache_state(repo, cache_jsonl)
        if revision_guard is not None:
            revision_guard.verify(f"external extract cache stage for {repo}")
        if state == "published":
            n_records = _validate_completed_external_cache(
                repo,
                repo_dir,
                cache_jsonl,
                canonical_project_id or repo_dir.name,
            )
            legacy_identity = (
                canonical_project_id is not None
                and _cache_first_record_needs_identity_override(
                    cache_jsonl,
                    project_id=canonical_project_id,
                )
            )
            _log(
                f"EXTRACT-CACHE HIT {repo}: source/publication validated "
                f"{n_records} records at {cache_jsonl}"
                + (" (legacy identity override recorded)" if legacy_identity else "")
            )
            return (
                cache_jsonl,
                n_records,
                "hit_legacy_identity_override" if legacy_identity else "hit",
            )
        if state == "miss":
            _log(
                f"EXTRACT-CACHE MISS {repo}: no published/checkpointed entry at "
                f"{cache_jsonl}; extracting explicitly"
            )
        if canonical_project_id is None:
            records_jsonl = stage_extract_commits(repo, repo_dir, cache_dir)
        else:
            records_jsonl = stage_extract_commits(
                repo,
                repo_dir,
                cache_dir,
                project_id=canonical_project_id,
            )
        if records_jsonl.resolve() != cache_jsonl.resolve():
            raise RepoFailure(
                repo,
                "extract_cache_validate",
                f"extractor returned unexpected external cache path: "
                f"{records_jsonl} != {cache_jsonl}",
            )
        n_records = _read_completed_external_publication(repo, cache_jsonl)
        status = "checkpoint_resume" if state == "checkpoint" else "fresh"
        _log(
            f"EXTRACT-CACHE {status.upper()} {repo}: validated "
            f"{n_records} records at {cache_jsonl}"
        )
        return cache_jsonl, n_records, status


def _discover_existing_jsonl(repo: str, work_root: Path, work_parent: Path) -> Path | None:
    """Locate a pre-existing <repo>_commits.jsonl from this or a prior run.

    Deterministic search order: the current work_root, then prior randomized
    conveyor work dirs under work_parent / DEFAULT_WORK_PARENT (back-compat:
    php-src was extracted by older code into a now-orphaned random work_root).
    The stable cache is handled only by its validated sentinel and is never
    adopted through this legacy path. Returns the first existing non-empty path,
    else None. Safe under the per-stream RunLock, which guarantees no OTHER live
    conveyor owns these dirs while we adopt from them.
    """
    # The stable cache candidate is intentionally absent. It is authoritative
    # only through _read_valid_sentinel(); adopting the same cache file after a
    # missing/mismatched sentinel would turn a truncated or corrupt extract into
    # a freshly stamped "done" corpus. Legacy work roots remain eligible because
    # they are outside the stable cache and predate its sentinel protocol.
    candidates: list[Path] = [work_root / repo / f"{repo}_commits.jsonl"]
    parents = {p for p in (work_parent, DEFAULT_WORK_PARENT) if p is not None}
    for parent in parents:
        if parent.exists():
            candidates.extend(
                sorted(parent.glob(f"streaming_conveyor_*/{repo}/{repo}_commits.jsonl"))
            )
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return None


def ensure_commit_records(
    repo: str,
    repo_dir: Path,
    work_root: Path,
    work_parent: Path,
    manifest: Manifest,
    resume: bool,
    *,
    revision_guard: CodeRevisionGuard | None = None,
    project_id: str | None = None,
) -> tuple[Path, int, str]:
    """Return (records_jsonl, n_records), running extract_git_history ONLY if needed.

    Resolution order (all writing/reading the STABLE EXTRACT_CACHE_ROOT/<repo>):
      (a) HIT: a valid .done sentinel matches the cached jsonl -> reuse instantly.
      (b) BACK-COMPAT: a jsonl is discoverable from this/a prior run AND the
          manifest already has a <repo>::r<...> range unit proving the prior
          extract finished -> adopt it into the cache, count lines, stamp the
          sentinel retroactively, reuse. Unproven JSONL is never marked done.
      (c) FRESH: nothing reusable -> run extract_git_history into the cache and
          stamp the sentinel on success.
    RAISES (RULE #1) on empty git log / empty extract; never returns a partial set.
    """
    if EXTRACT_CACHE_MODE == EXTRACT_CACHE_MODE_EXTERNAL:
        return _ensure_external_commit_records(
            repo,
            repo_dir,
            revision_guard=revision_guard,
            project_id=project_id,
        )

    canonical_project_id = _resolve_commit_project_id(repo, repo_dir, project_id)

    cache_dir = extract_cache_dir(repo)
    cache_jsonl = cache_dir / f"{repo}_commits.jsonl"
    repo_has_range_done = any(k.startswith(f"{repo}::r") for k in manifest.done)

    if resume:
        # (a) Cheap stat-validated cache hit.
        lc = _read_valid_sentinel(cache_jsonl)
        if lc is not None:
            legacy_identity = (
                canonical_project_id is not None
                and _cache_first_record_needs_identity_override(
                    cache_jsonl,
                    project_id=canonical_project_id,
                )
            )
            _log(f"EXTRACT-CKPT HIT {repo}: reuse {cache_jsonl} ({lc} records); "
                 f"skip ~extract_git_history"
                 + (" (legacy identity override recorded)" if legacy_identity else ""))
            return (
                cache_jsonl,
                lc,
                "hit_legacy_identity_override" if legacy_identity else "hit",
            )

        # (b) Adopt an existing jsonl (stable cache miss but a prior extract exists).
        existing = _discover_existing_jsonl(repo, work_root, work_parent)
        if existing is not None and not repo_has_range_done:
            _log(
                f"EXTRACT-CKPT UNPROVEN {repo}: ignoring {existing}; no completed "
                "range receipt proves this legacy JSONL reached EOF"
            )
            existing = None
        if existing is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if existing.resolve() != cache_jsonl.resolve():
                cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(existing), str(cache_jsonl))  # rename within fs / copy across
            n = _count_jsonl_lines(cache_jsonl)
            if n == 0:
                raise RepoFailure(repo, "extract_git_history",
                                  f"adopted commit jsonl is empty: {cache_jsonl}")
            _write_extract_sentinel(cache_jsonl, n)
            tag = "BACK-COMPAT"
            _log(f"EXTRACT-CKPT {tag} {repo}: adopted {existing} -> {cache_jsonl} "
                 f"({n} records); stamped sentinel; skip ~extract_git_history")
            legacy_identity = (
                canonical_project_id is not None
                and _cache_first_record_needs_identity_override(
                    cache_jsonl,
                    project_id=canonical_project_id,
                )
            )
            return (
                cache_jsonl,
                n,
                "hit_legacy_identity_override"
                if legacy_identity
                else tag.lower().replace("-", "_"),
            )

        cache_status = "fresh"
        if repo_has_range_done:
            # Manifest proves the extract finished, but no jsonl survived anywhere.
            # extract_git_history is deterministic (git log order), so a fresh
            # re-extract reproduces the SAME record order and already-done ranges
            # still resume-skip. Loud, not silent.
            _log(f"EXTRACT-CKPT MISS {repo}: manifest has done ranges but NO jsonl "
                 f"found on disk; re-extracting (deterministic order preserves "
                 f"done-range alignment)")
            cache_status = "miss_reextract"
    else:
        cache_status = "fresh"

    # (c) Fresh extract into the stable cache. FAIL LOUD on empty git log / output.
    if revision_guard is not None:
        revision_guard.verify(f"git commit-list stage for {repo}")
    commit_list = get_commit_list(repo_dir)
    if not commit_list:
        raise RepoNoCommitRecords(
            repo,
            reason="no_matching_commits",
            detail="no --no-merges --diff-filter=M commits",
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    if revision_guard is not None:
        revision_guard.verify(f"extract_git_history stage for {repo}")
    if canonical_project_id is None:
        records_jsonl = stage_extract_commits(repo, repo_dir, cache_dir)
    else:
        records_jsonl = stage_extract_commits(
            repo,
            repo_dir,
            cache_dir,
            project_id=canonical_project_id,
        )
    n = _count_jsonl_lines(records_jsonl)
    if n == 0:
        raise RepoNoCommitRecords(
            repo,
            reason="no_cpp_commit_records",
            detail=f"zero records after extract for {repo}",
        )
    _write_extract_sentinel(records_jsonl, n)
    _log(f"EXTRACT-CKPT FRESH {repo}: extracted {n} records -> {records_jsonl}; "
         f"stamped sentinel")
    return records_jsonl, n, cache_status


def repo_fully_done(
    repo: str, manifest: Manifest, streams: str, all_ranges_done: bool
) -> bool:
    """True iff EVERY unit this run is responsible for is marked done in the manifest.

    Used to gate temp + extract-cache deletion: a mid-repo kill, a failed unit,
    or a signal-cancelled range leaves this False -> temp is RETAINED for resume.
    """
    code_ok = (streams not in {"both", "code"}) or manifest.is_done(code_key(repo))
    commits_ok = (streams not in {"both", "commits"}) or all_ranges_done
    return code_ok and commits_ok


def commit_plan_key(repo: str) -> str:
    return f"{repo}::commit_plan"


def manifest_complete_commit_ranges(
    repo: str, manifest: Manifest, range_size: int
) -> tuple[tuple[int, int], ...] | None:
    """Return done ranges only with an authoritative extracted record count.

    A range shorter than ``range_size`` is not EOF evidence because adaptive
    planning also cuts ranges at a byte target. Completion is proven only by a
    persisted commit-plan record and exact coverage of ``[0, n_records)``.
    """
    if range_size <= 0:
        raise ValueError(f"range_size must be positive, got {range_size}")
    # A repo-level failure records an earlier extraction attempt, not a unit of
    # range coverage. Once the authoritative plan and every range prove exact
    # coverage it is stale. A failed range remains authoritative and must keep
    # completion fail-closed until that same range key is marked done.
    if any(k.startswith(f"{repo}::r") for k in manifest.failed):
        return None

    plan = manifest.done.get(commit_plan_key(repo))
    if not isinstance(plan, dict) or plan.get("source") != "commit_plan":
        return None
    try:
        n_records = int(plan["n_records"])
    except (KeyError, TypeError, ValueError):
        return None
    if n_records <= 0:
        return None

    ranges: list[tuple[int, int]] = []
    prefix = f"{repo}::r"
    for key, info in manifest.done.items():
        if not key.startswith(prefix):
            continue
        raw = info.get("range") if isinstance(info, dict) else None
        if not isinstance(raw, list | tuple) or len(raw) != 2:
            return None
        try:
            start = int(raw[0])
            end = int(raw[1])
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start:
            return None
        # The key and payload must agree; otherwise the manifest is not a
        # reliable source of coverage truth.
        try:
            key_start = int(key.rsplit("::r", 1)[1])
        except (IndexError, ValueError):
            return None
        if key_start != start:
            return None
        ranges.append((start, end))

    if not ranges:
        return None
    ranges.sort()
    expected_start = 0
    for start, end in ranges:
        if start != expected_start:
            return None
        if end - start > range_size:
            return None
        expected_start = end
    if expected_start != n_records:
        return None
    return tuple(ranges)


def mark_commit_stream_complete(
    repo: str,
    manifest: Manifest,
    manifest_lock: threading.Lock | None,
    complete_ranges: Sequence[tuple[int, int]],
) -> dict:
    """Persist aggregate completion proven by the plan and exact range coverage.

    ``Manifest.mark_done`` also removes an earlier aggregate failure. Keeping
    this as a derived summary avoids treating ``repo::commits`` as a second,
    independent source of completion truth.
    """
    if not complete_ranges:
        raise ValueError(f"cannot mark {repo}::commits complete without ranges")
    plan = manifest.done.get(commit_plan_key(repo))
    if not isinstance(plan, dict) or plan.get("source") != "commit_plan":
        raise RuntimeError(
            f"cannot mark {repo}::commits complete without authoritative commit_plan"
        )
    try:
        n_records = int(plan["n_records"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot mark {repo}::commits complete: invalid commit_plan n_records"
        ) from exc
    if complete_ranges[0][0] != 0 or complete_ranges[-1][1] != n_records:
        raise RuntimeError(
            f"cannot mark {repo}::commits complete: ranges do not cover "
            f"[0, {n_records})"
        )
    info = {
        "source": "commits",
        "repo": repo,
        "complete": True,
        "completion_proof": "commit_plan_exact_range_coverage",
        "n_records": n_records,
        "range_count": len(complete_ranges),
    }
    cache_receipt = plan.get("extract_cache")
    if (
        isinstance(cache_receipt, dict)
        and cache_receipt.get("mode") == EXTRACT_CACHE_MODE_EXTERNAL
    ):
        info["extract_cache"] = dict(cache_receipt)
    if manifest_lock is None:
        manifest.mark_done(f"{repo}::commits", info)
    else:
        with manifest_lock:
            manifest.mark_done(f"{repo}::commits", info)
    return info


def manifest_done_commit_intervals(repo: str, manifest: Manifest) -> tuple[tuple[int, int], ...]:
    """Return validated done commit intervals for ``repo``.

    Adaptive range splitting means a manifest key such as ``repo::r23000`` may
    cover only ``[23000, 23250)`` instead of the original 500-record range. All
    resume and cleanup decisions must therefore use the payload interval, not
    only the key.
    """
    intervals: list[tuple[int, int]] = []
    prefix = f"{repo}::r"
    for key, info in manifest.done.items():
        if not key.startswith(prefix):
            continue
        raw = info.get("range") if isinstance(info, dict) else None
        if not isinstance(raw, list | tuple) or len(raw) != 2:
            continue
        try:
            start = int(raw[0])
            end = int(raw[1])
            key_start = int(key.rsplit("::r", 1)[1])
        except (TypeError, ValueError, IndexError):
            continue
        if start < 0 or end <= start or key_start != start:
            continue
        intervals.append((start, end))
    return tuple(sorted(intervals))


def missing_commit_subranges(
    start: int, end: int, done_intervals: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """Return uncovered subranges inside ``[start, end)``.

    Used on resume so an adaptively split completed half does not cause the
    uncompleted half to be skipped.
    """
    if end <= start:
        raise ValueError(f"invalid range [{start}:{end}]")
    missing: list[tuple[int, int]] = []
    cursor = start
    for done_start, done_end in done_intervals:
        if done_end <= cursor:
            continue
        if done_start >= end:
            break
        if done_start > cursor:
            missing.append((cursor, min(done_start, end)))
        cursor = max(cursor, min(done_end, end))
        if cursor >= end:
            break
    if cursor < end:
        missing.append((cursor, end))
    return tuple(missing)


def plan_commit_ranges(
    records_jsonl: Path,
    n_records: int,
    *,
    max_records: int,
    target_bytes: int = DEFAULT_RANGE_TARGET_BYTES,
) -> tuple[tuple[int, int], ...]:
    """Plan commit ranges bounded by both record count and raw JSONL bytes.

    Fixed-size 500-record ranges are badly imbalanced on repos with large diffs:
    one range can produce 2-3x the JSONL/materialization work of another. This
    keeps the old ``max_records`` ceiling while cutting earlier when the raw
    extracted commit-record bytes cross ``target_bytes``. Manifest keys remain
    ``repo::r<start>`` and payloads carry the exact ``[start, end)`` interval, so
    resume compatibility is preserved.
    """
    if n_records < 0:
        raise ValueError(f"n_records must be non-negative, got {n_records}")
    if max_records <= 0:
        raise ValueError(f"max_records must be positive, got {max_records}")
    if n_records == 0:
        return ()
    if target_bytes <= 0:
        return tuple(
            (s, min(s + max_records, n_records))
            for s in range(0, n_records, max_records)
        )

    ranges: list[tuple[int, int]] = []
    start = 0
    current_bytes = 0
    current_records = 0
    with records_jsonl.open("rb") as fh:
        for idx, line in enumerate(fh):
            if idx >= n_records:
                break
            line_bytes = len(line)
            if current_records and (
                current_records >= max_records
                or current_bytes + line_bytes > target_bytes
            ):
                ranges.append((start, idx))
                start = idx
                current_bytes = 0
                current_records = 0
            current_bytes += line_bytes
            current_records += 1

    if current_records:
        ranges.append((start, start + current_records))

    if not ranges or ranges[-1][1] != n_records:
        # This means the caller supplied a bad n_records for the file currently on
        # disk. Fail here instead of silently skipping or duplicating records.
        raise RuntimeError(
            f"range planning covered {ranges[-1][1] if ranges else 0} records "
            f"but n_records={n_records} for {records_jsonl}"
        )
    return tuple(ranges)


def manifest_covers_commit_span(repo: str, manifest: Manifest, n_records: int) -> bool:
    """True iff manifest done intervals cover ``[0, n_records)`` exactly enough."""
    if n_records <= 0:
        return False
    cursor = 0
    for start, end in manifest_done_commit_intervals(repo, manifest):
        if end <= cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= n_records:
            return True
    return False


def _resolved_failure_keys_for_success(key: str) -> tuple[str, ...]:
    """Return stale failure keys resolved by one terminal unit success.

    ``<repo>::repo`` is the outer worker's fallback key when a failure escapes
    before it can be recorded against the concrete code/range unit. A later
    terminal success for that repo makes the fallback receipt stale. Commit-plan
    publication is metadata rather than terminal work, so it cannot resolve the
    fallback. Sibling ranges and aggregate commit failures remain authoritative.
    """
    repo, separator, unit = key.rpartition("::")
    if not separator:
        return (key,)
    terminal_unit = unit in {"code", "commits", "no_git"} or (
        re.fullmatch(r"r\d+", unit) is not None
    )
    if terminal_unit:
        return key, f"{repo}::repo"
    return (key,)


class ConcurrentManifest(Manifest):
    """Cross-process-safe conveyor resume manifest (fixes the H4 clobber).

    The base ``streaming_reindex.Manifest.save()`` rewrites the whole file from
    its in-memory ``done``/``failed`` dicts with NO disk re-read and NO OS-level
    lock (``manifest_lock`` is only a ``threading.Lock`` -- useless across
    processes). Two conveyor processes that legitimately run at the same time --
    a ``--streams code`` run and a ``--streams commits`` run, each holding only
    its OWN per-stream :class:`RunLock` -- therefore CLOBBER the single shared
    ``outputs/conveyor/_done.json``: whichever writes last overwrites the file
    with its own snapshot and drops the other stream's keys (lost update),
    silently destroying that stream's resume + token accounting even though the
    two key spaces are disjoint (``<repo>::code`` vs ``<repo>::r<start>``).

    This subclass makes every manifest mutation atomic ACROSS processes: under an
    exclusive ``flock`` on a sibling ``.lock`` file it RE-READS the on-disk
    manifest, applies one logical mutation to it, then atomically replaces the
    file. The in-memory dicts are refreshed to the merged on-disk state so
    in-process resume checks (``is_done``) also observe the other process's
    committed keys. ONE clear write path; any error RAISES (RULE #1). A blind
    full-file ``save()`` is disabled so the clobber cannot be reintroduced.
    """

    def __init__(
        self,
        path: Path,
        done: dict | None = None,
        failed: dict | None = None,
        code_revision: dict | None = None,
        pr_completion: dict | None = None,
    ):
        # Bypass the dataclass __init__ so we can attach lock state.
        self.path = path
        self.done = dict(done or {})
        self.failed = dict(failed or {})
        self.code_revision = (
            json.loads(json.dumps(code_revision)) if code_revision is not None else None
        )
        self.pr_completion = (
            json.loads(json.dumps(pr_completion)) if pr_completion is not None else None
        )
        self._lock_path = Path(str(path) + ".lock")
        self._thread_lock = threading.Lock()

    @classmethod
    def load(cls, path: Path) -> "ConcurrentManifest":
        if not path.exists():
            return cls(path=path)
        blob = json.loads(path.read_text())
        return cls(
            path=path,
            done=blob.get("done", {}),
            failed=blob.get("failed", {}),
            code_revision=blob.get("code_revision"),
            pr_completion=blob.get("pr_completion"),
        )

    def _read_disk(self) -> dict:
        if self.path.exists():
            blob = json.loads(self.path.read_text())
            return {
                "done": blob.get("done", {}),
                "failed": blob.get("failed", {}),
                "code_revision": blob.get("code_revision"),
                "pr_completion": blob.get("pr_completion"),
            }
        return {
            "done": {},
            "failed": {},
            "code_revision": None,
            "pr_completion": None,
        }

    def _atomic_replace(self, done: dict, failed: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp.{os.getpid()}")
        code_revision = getattr(
            self,
            "_pending_code_revision",
            self.code_revision,
        )
        pr_completion = getattr(
            self,
            "_pending_pr_completion",
            self.pr_completion,
        )
        payload = {"done": done, "failed": failed}
        if code_revision is not None:
            payload["code_revision"] = code_revision
        if pr_completion is not None:
            payload["pr_completion"] = pr_completion
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )
        tmp.replace(self.path)  # atomic on POSIX

    def _merge_under_lock(self, apply_change) -> None:
        """Apply ``apply_change(done, failed)`` to the freshest on-disk state
        under an exclusive cross-process flock, then atomically persist and
        refresh the in-memory dicts to the merged result."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            fh = self._lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                state = self._read_disk()
                apply_change(state)
                self._pending_code_revision = state["code_revision"]
                self._pending_pr_completion = state["pr_completion"]
                try:
                    self._atomic_replace(state["done"], state["failed"])
                finally:
                    del self._pending_code_revision
                    del self._pending_pr_completion
                self.done = state["done"]
                self.failed = state["failed"]
                self.code_revision = state["code_revision"]
                self.pr_completion = state["pr_completion"]
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()

    def mark_done(self, key: str, info: dict) -> None:
        resolved_failures = _resolved_failure_keys_for_success(key)

        def apply(state: dict) -> None:
            done = state["done"]
            failed = state["failed"]
            done[key] = info
            for failure_key in resolved_failures:
                failed.pop(failure_key, None)

        self._merge_under_lock(apply)

    def mark_started(self, key: str) -> None:
        """Invalidate stale terminal state before a real retry starts."""

        def apply(state: dict) -> None:
            done = state["done"]
            failed = state["failed"]
            done.pop(key, None)
            failed.pop(key, None)

        self._merge_under_lock(apply)

    def mark_started_prefix(self, prefix: str) -> None:
        """Invalidate every stale range state for one logical parent."""

        def apply(state: dict) -> None:
            done = state["done"]
            failed = state["failed"]
            for key in tuple(done):
                if key.startswith(prefix):
                    done.pop(key)
            for key in tuple(failed):
                if key.startswith(prefix):
                    failed.pop(key)

        self._merge_under_lock(apply)

    def mark_failed(self, key: str, stage: str, detail: str) -> None:
        rec = {
            "stage": stage,
            "detail": detail[:2000],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        def apply(state: dict) -> None:
            done = state["done"]
            failed = state["failed"]
            done.pop(key, None)
            failed[key] = rec

        self._merge_under_lock(apply)

    def bind_code_revision(self, receipt: dict) -> None:
        """Atomically bind resume state to one exact code identity."""
        requested_identity = _code_revision_identity(receipt)

        def apply(state: dict) -> None:
            existing = state["code_revision"]
            if existing is None:
                if state["done"] or state["failed"]:
                    raise CodeRevisionMismatchError(
                        "existing conveyor manifest has work receipts but no code "
                        "revision; use a new conveyor/output root"
                    )
                state["code_revision"] = json.loads(json.dumps(receipt))
                return
            if _code_revision_identity(existing) != requested_identity:
                raise CodeRevisionMismatchError(
                    "conveyor manifest code revision mismatch: "
                    f"manifest={existing.get('git_commit')}:"
                    f"{existing.get('dirty_fingerprint')} requested="
                    f"{receipt.get('git_commit')}:{receipt.get('dirty_fingerprint')}; "
                    "resume requires the same revision or a new conveyor/output root"
                )

        self._merge_under_lock(apply)

    def bind_pr_completion(self, binding: dict) -> None:
        """Atomically bind every commit receipt to one verified PR snapshot."""
        requested_identity = _pr_completion_identity(binding)

        def apply(state: dict) -> None:
            existing = state["pr_completion"]
            if existing is None:
                commit_keys = [
                    key
                    for key in (*state["done"], *state["failed"])
                    if re.search(
                        r"::(?:r\d+|commits|commit_plan|no_git|repo)$",
                        key,
                    )
                ]
                if commit_keys:
                    raise PRCompletionBindingError(
                        "existing conveyor manifest has commit work receipts but "
                        "no PR completion binding; use a new conveyor/output root"
                    )
                state["pr_completion"] = json.loads(json.dumps(binding))
                return
            if _pr_completion_identity(existing) != requested_identity:
                raise PRCompletionBindingError(
                    "conveyor manifest PR completion mismatch: "
                    f"manifest={existing.get('receipt_sha256')} requested="
                    f"{binding.get('receipt_sha256')}; resume requires the same "
                    "PR snapshot or a new conveyor/output root"
                )

        self._merge_under_lock(apply)

    def save(self) -> None:  # pragma: no cover - guard against accidental reuse
        raise RuntimeError(
            "ConcurrentManifest persists via mark_done/mark_failed (atomic "
            "cross-process merge under flock); a blind save() would re-introduce "
            "the lost-update clobber this class exists to prevent."
        )


class ProgressWriter:
    """Append-only machine-readable progress events for live throughput tracking."""

    def __init__(self, path: Path | None):
        self.path = path
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.extract_cache_seen = 0
        self.extract_cache_hits = 0
        self.extract_cache_reused = 0
        self.extract_cache_status_counts: dict[str, int] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def extract_cache_metrics(self) -> dict:
        with self.lock:
            return {
                "seen": self.extract_cache_seen,
                "hits": self.extract_cache_hits,
                "hit_rate": (
                    self.extract_cache_hits / self.extract_cache_seen
                    if self.extract_cache_seen else 0.0
                ),
                "reused": self.extract_cache_reused,
                "reuse_rate": (
                    self.extract_cache_reused / self.extract_cache_seen
                    if self.extract_cache_seen else 0.0
                ),
                "status_counts": dict(self.extract_cache_status_counts),
            }

    def emit(self, event: str, **payload) -> None:
        if self.path is None:
            return
        now = time.time()
        with self.lock:
            if event == "extract_cache":
                status = str(payload.get("status") or "unknown")
                self.extract_cache_seen += 1
                self.extract_cache_status_counts[status] = (
                    self.extract_cache_status_counts.get(status, 0) + 1
                )
                cache_hit = bool(payload.get("hit", status == "hit"))
                cache_reused = bool(
                    payload.get(
                        "reused",
                        status in {
                            "hit",
                            "checkpoint_resume",
                            "adopt",
                            "back_compat",
                            "orphan_adopt",
                        },
                    )
                )
                if cache_hit:
                    self.extract_cache_hits += 1
                if cache_reused:
                    self.extract_cache_reused += 1
                payload = {
                    **payload,
                    "extract_cache_seen": self.extract_cache_seen,
                    "extract_cache_hits": self.extract_cache_hits,
                    "extract_cache_hit_rate": (
                        self.extract_cache_hits / self.extract_cache_seen
                    ),
                    "extract_cache_reused": self.extract_cache_reused,
                    "extract_cache_reuse_rate": (
                        self.extract_cache_reused / self.extract_cache_seen
                    ),
                    "extract_cache_status_counts": dict(
                        self.extract_cache_status_counts
                    ),
                }
            row = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_s": round(now - self.started_at, 3),
                "event": event,
                **payload,
            }
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")


class RunLock:
    """Fail-loud per-stream process lock.

    The conveyor writes per-stream outputs and uses stream-local temp filenames.
    Two live conveyors for the same stream can therefore race on repo outputs.
    This lock prevents that class of bug before any repo is extracted.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.seek(0)
            holder = self._fh.read().strip()
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"conveyor stream lock is already held: {self.path}\n"
                f"holder:\n{holder}"
            ) from exc
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "argv": sys.argv,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


class WorkReservation:
    """Handle for one active conveyor unit reservation."""

    def __init__(
        self,
        ledger: "UnitReservationLedger",
        key: str,
        token: str | None,
        holder: dict | None,
    ):
        self.ledger = ledger
        self.key = key
        self.token = token
        self.holder = holder
        self.acquired = token is not None
        self._released = False

    def release(self) -> None:
        if not self.acquired or self._released:
            return
        assert self.token is not None
        self.ledger.release(self.key, self.token)
        self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class UnitReservationLedger:
    """Small cross-process active-unit ledger for high-parallel conveyor runs.

    The dedup DB no longer accepts staging writes from subprocesses. This ledger
    solves a different problem: do not let two parent conveyor workers process
    the same output unit at once when repo/range parallelism is high or when
    separate stream-specific conveyors share the manifest. It is a tiny JSON
    file under an OS flock; completed history still lives in the manifest.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock_path = Path(str(self.path) + ".lock")
        self._thread_lock = threading.Lock()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _read(self) -> dict:
        if not self.path.exists():
            return {"active": {}}
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"reservation ledger is corrupt: {self.path}") from exc
        active = blob.get("active", {})
        if not isinstance(active, dict):
            raise RuntimeError(f"reservation ledger has invalid active map: {self.path}")
        return {"active": active}

    def _atomic_replace(self, blob: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(blob, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def _with_lock(self, fn):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            fh = self._lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                blob = self._read()
                result = fn(blob)
                self._atomic_replace(blob)
                return result
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()

    def cleanup_stale(self) -> int:
        """Drop reservations whose owning parent process no longer exists."""

        def apply(blob: dict) -> int:
            active = blob["active"]
            stale = [
                key for key, rec in active.items()
                if not self._pid_alive(int(rec.get("pid", -1)))
            ]
            for key in stale:
                active.pop(key, None)
            return len(stale)

        return int(self._with_lock(apply))

    def acquire(self, key: str, *, stream: str, repo: str) -> WorkReservation:
        if not key:
            raise ValueError("reservation key must be non-empty")
        token = f"{os.getpid()}:{threading.get_ident()}:{time.time_ns()}"

        def apply(blob: dict) -> WorkReservation:
            active = blob["active"]
            stale = [
                item_key for item_key, rec in active.items()
                if not self._pid_alive(int(rec.get("pid", -1)))
            ]
            for item_key in stale:
                active.pop(item_key, None)
            holder = active.get(key)
            if holder is not None:
                return WorkReservation(self, key, None, dict(holder))
            active[key] = {
                "key": key,
                "stream": stream,
                "repo": repo,
                "pid": os.getpid(),
                "thread": threading.get_ident(),
                "token": token,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "argv": sys.argv,
            }
            return WorkReservation(self, key, token, None)

        return self._with_lock(apply)

    def release(self, key: str, token: str) -> None:
        def apply(blob: dict) -> None:
            active = blob["active"]
            rec = active.get(key)
            if rec is not None and rec.get("token") == token:
                active.pop(key, None)

        self._with_lock(apply)


class DedupCheckpointController:
    """Token-milestone SQLite WAL checkpoints for the shared dedup store."""

    def __init__(
        self,
        *,
        dedup_db: Path | None,
        interval_tokens: int,
        mode: str,
        busy_timeout_ms: int,
        progress: ProgressWriter,
    ):
        self.dedup_db = dedup_db
        self.interval_tokens = int(interval_tokens)
        self.mode = mode.upper()
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.progress = progress
        self.lock = threading.Lock()
        self.next_threshold = (
            self.interval_tokens if self.dedup_db and self.interval_tokens > 0 else None
        )

    @staticmethod
    def _wal_path(db_path: Path) -> Path:
        return Path(str(db_path) + "-wal")

    def maybe_checkpoint(self, cumulative_valid_tokens: int) -> None:
        if self.next_threshold is None:
            return
        with self.lock:
            if self.next_threshold is None or cumulative_valid_tokens < self.next_threshold:
                return
            threshold = self.next_threshold
            while self.next_threshold <= cumulative_valid_tokens:
                self.next_threshold += self.interval_tokens
        self._checkpoint(threshold, cumulative_valid_tokens)

    def _checkpoint(self, threshold: int, cumulative_valid_tokens: int) -> None:
        assert self.dedup_db is not None
        wal = self._wal_path(self.dedup_db)
        wal_before = wal.stat().st_size if wal.exists() else 0
        started = time.time()
        try:
            conn = sqlite3.connect(
                str(self.dedup_db),
                timeout=max(self.busy_timeout_ms / 1000.0, 0.001),
            )
            try:
                conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
                row = conn.execute(f"PRAGMA wal_checkpoint({self.mode})").fetchone()
            finally:
                conn.close()
            wal_after = wal.stat().st_size if wal.exists() else 0
            busy, log_pages, checkpointed_pages = [int(x) for x in row]
            self.progress.emit(
                "dedup_checkpoint",
                dedup_db=str(self.dedup_db),
                mode=self.mode,
                threshold_tokens=threshold,
                cumulative_valid_tokens=cumulative_valid_tokens,
                busy=busy,
                log_pages=log_pages,
                checkpointed_pages=checkpointed_pages,
                wal_bytes_before=wal_before,
                wal_bytes_after=wal_after,
                checkpoint_elapsed_s=round(time.time() - started, 3),
            )
        except SymbolIdentityError:
            raise
        except Exception as exc:  # noqa: BLE001 - checkpoint failure is telemetry, not data loss
            wal_after = wal.stat().st_size if wal.exists() else 0
            self.progress.emit(
                "dedup_checkpoint_failed",
                dedup_db=str(self.dedup_db),
                mode=self.mode,
                threshold_tokens=threshold,
                cumulative_valid_tokens=cumulative_valid_tokens,
                wal_bytes_before=wal_before,
                wal_bytes_after=wal_after,
                detail=str(exc)[:2000],
                checkpoint_elapsed_s=round(time.time() - started, 3),
            )


def _length_totals(info: dict) -> dict[str, int]:
    totals = {"rows": 0, "valid_tokens": 0, "pad_tokens": 0, "capacity_tokens": 0}
    for bucket, st in info.get("lengths", {}).items():
        for key in totals:
            # Fail-loud (RULE #1): a malformed stat dict must crash here rather
            # than silently default to 0 -- cumulative['valid'] drives the
            # --token-budget stop gate and the dedup WAL checkpoint thresholds,
            # so an undercount overshoots the budget / skips checkpoints. Mirror
            # the fail-loud st[...] reads in the run report aggregation below.
            if key not in st:
                raise KeyError(
                    f"_length_totals: malformed stat dict for length bucket "
                    f"{bucket!r} (source={info.get('source')!r}): missing field "
                    f"{key!r}; present fields = {sorted(st)}"
                )
            totals[key] += int(st[key])
    return totals


def _primary_bucket_progress(info: dict, lengths: Sequence[int]) -> dict[str, int]:
    """Return the smallest configured bucket's real-token count for live logs."""
    if not lengths:
        return {"primary_bucket_length": 0, "primary_bucket_valid_tokens": 0}
    primary = min(int(x) for x in lengths)
    stats = (info.get("lengths") or {}).get(str(primary), {})
    return {
        "primary_bucket_length": primary,
        "primary_bucket_valid_tokens": int(stats.get("valid_tokens", 0)),
    }


def code_key(repo: str) -> str:
    """Checkpoint key for one repo's CODE half."""
    return f"{repo}::code"


def no_git_key(repo: str) -> str:
    """Checkpoint key proving the .git-preserving stream saw no git metadata."""
    return f"{repo}::no_git"


def manifest_repo_known_no_git(repo: str, manifest: Manifest) -> bool:
    info = manifest.done.get(no_git_key(repo))
    return isinstance(info, dict) and info.get("no_git") is True


def should_stage_repo_from_manifest(
    repo: str,
    *,
    streams: str,
    resume: bool,
    manifest: Manifest,
    range_size: int,
    only_repos: set[str] | None,
    manifest_lock: threading.Lock | None = None,
) -> bool:
    """Return False when the manifest proves this repo needs no extraction.

    The conservative commit proof comes from ``manifest_complete_commit_ranges``:
    if it cannot prove full coverage without reading the repo's commit JSONL, we
    keep the old safe behavior and stage the repo.
    """
    if only_repos is not None:
        return repo in only_repos
    if not resume:
        return True
    if streams in {"both", "commits"} and manifest_repo_known_no_git(repo, manifest):
        return False

    code_needed = streams in {"both", "code"} and not manifest.is_done(code_key(repo))
    commits_needed = False
    if streams in {"both", "commits"}:
        complete_ranges = manifest_complete_commit_ranges(repo, manifest, range_size)
        commits_needed = complete_ranges is None
        if complete_ranges is not None:
            # This callback can skip extraction entirely, so reconcile the
            # derived aggregate sentinel here instead of waiting for
            # run_commits_half, which will never be called for this repo.
            mark_commit_stream_complete(
                repo,
                manifest,
                manifest_lock,
                complete_ranges,
            )
    return code_needed or commits_needed


def stream_lock_names(streams: str) -> tuple[str, ...]:
    if streams == "both":
        return ("code", "commits")
    return (streams,)


def populate_code_source_cache(
    work_root: Path,
    source_cache_dir: Path,
    should_process,
    progress: ProgressWriter,
    *,
    max_repos: int | None = None,
) -> dict:
    """Populate the code source cache without running index/tokenize stages."""
    started = time.monotonic()

    def emit_repo_ready(repo: str, repo_dir: Path, repo_count: int) -> None:
        progress.emit(
            "source_cache_repo_ready",
            repo=repo,
            repo_dir=str(repo_dir),
            source_cache_dir=str(source_cache_dir),
            repo_count=repo_count,
        )

    report = sr.populate_source_cache(
        work_root,
        should_process,
        source_cache_dir,
        max_repos=max_repos,
        on_repo_ready=emit_repo_ready,
    )
    report = {
        **report,
        "elapsed_s": round(time.monotonic() - started, 6),
    }
    progress.emit("source_cache_populated", **report)
    return report


# --------------------------------------------------------------------------- #
# CODE half: orchestrate streaming_reindex.process_one_repo (no reimplementation).
# Its append_output already lands outputs/reindexed/<L>/<repo>.parquet; we then
# recompress those per-length files with MAX zstd to match the commit stream.
# --------------------------------------------------------------------------- #
def run_code_half(
    repo: str,
    project_id: str,
    repo_dir: Path,
    lengths_code: Sequence[int],
    work_root: Path,
    dedup_db: Path | None,
    dedup_near: bool,
    global_symbol_index: Path | None = None,
    memory_limit_gb: float = 10.0,
    parse_workers: int = 2,
    index_timeout_s: int | None = None,
    index_stall_timeout_s: int | None = None,
    recompressor: BackgroundRecompressor | None = None,
    revision_guard: CodeRevisionGuard | None = None,
    source_quarantine_manifest: Path | None = None,
) -> dict:
    """index+route+pack the repo's source via the EXISTING code stage, zstd-max.

    Calls streaming_reindex.process_one_repo verbatim (it threads the SHARED
    dedup_db into index_project). Then recompresses each produced
    outputs/reindexed/<L>/<repo>.parquet with MAX zstd (the code stage writes
    plain parquet; the commit stage already recompresses, so we make code match).
    RAISES RepoFailure on any stage failure (no fallback).
    """
    stage_id = sr.code_stage_id(repo) if dedup_db is not None else None
    stage_db = sr.code_stage_db(work_root / repo, repo) if dedup_db is not None else None
    promoted = False
    try:
        try:
            if revision_guard is not None:
                revision_guard.verify(f"code pipeline stage for {repo}")
            info = sr.process_one_repo(
                repo, repo_dir, lengths_code, work_root, dedup_db, dedup_near,
                global_symbol_index, memory_limit_gb, parse_workers, index_timeout_s,
                index_stall_timeout_s,
                project_id=project_id,
                promote_dedup_on_success=False,
                source_quarantine_manifest=source_quarantine_manifest,
            )
        except RepoFailure as exc:
            _raise_if_revision_subprocess_failure(exc)
            skip_reason = code_skip_reason(exc)
            if skip_reason is not None:
                return {
                    "source": "code",
                    "repo": repo,
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "lengths": {},
                    "stage_timings_s": {},
                    "detail": exc.detail,
                }
            raise
        if info.get("skipped"):
            return info
        timings = dict(info.get("stage_timings_s", {}))
        # Recompression is part of publication, not deferred cleanup. Multiple
        # bucket files can still run concurrently, but this unit cannot promote
        # dedup or become manifest-done until every recompress future succeeds.
        started = time.monotonic()
        jobs: list[tuple[Path, Future]] = []
        try:
            for L in info.get("lengths", {}):
                dest = sr.OUTPUT_ROOT / str(L) / f"{repo}.parquet"
                if dest.exists():
                    if recompressor is None:
                        if revision_guard is not None:
                            revision_guard.verify(f"recompress stage for {repo}:{L}")
                        src.recompress_zstd_max(dest)
                    else:
                        jobs.append((dest, recompressor.submit(dest)))
            if recompressor is not None:
                recompressor.wait(jobs)
        except SymbolIdentityError:
            remove_code_outputs(repo, info.get("lengths", {}).keys())
            raise
        except Exception as exc:
            remove_code_outputs(repo, info.get("lengths", {}).keys())
            raise RepoFailure(repo, "recompress", f"{type(exc).__name__}: {exc}") from exc
        timings["recompress_s"] = round(time.monotonic() - started, 6)

        try:
            timings.update(sr.promote_dedup_stage(dedup_db, stage_id, stage_db))
        except SymbolIdentityError:
            remove_code_outputs(repo, info.get("lengths", {}).keys())
            raise
        except Exception as exc:
            remove_code_outputs(repo, info.get("lengths", {}).keys())
            raise RepoFailure(
                repo,
                "dedup_promote",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        promoted = True
        info["stage_timings_s"] = timings
        return info
    finally:
        if not promoted:
            sr.discard_dedup_stage(dedup_db, stage_id, stage_db)


CodeRunner = Callable[
    [
        str,
        str,
        Path,
        Sequence[int],
        Path,
        Path | None,
        bool,
        Path | None,
        float,
        int,
        int | None,
        int | None,
        BackgroundRecompressor | None,
    ],
    dict,
]


def remove_code_outputs(repo: str, lengths: Iterable[str | int]) -> None:
    for length in lengths:
        dest = sr.OUTPUT_ROOT / str(length) / f"{repo}.parquet"
        if dest.exists():
            dest.unlink()


def is_no_trainable_source_failure(exc: RepoFailure) -> bool:
    return (
        exc.stage == "index_project"
        and "no training docs (no_trainable_source)" in exc.detail.lower()
    )


def code_skip_reason(exc: RepoFailure) -> str | None:
    if is_no_trainable_source_failure(exc):
        return "no_trainable_source"
    if (
        exc.stage == "index_project"
        and "no training docs (dedup_exhausted)" in exc.detail.lower()
    ):
        return "dedup_exhausted"
    return None


def is_retryable_index_project_failure(exc: RepoFailure) -> bool:
    detail = exc.detail.lower()
    return (
        exc.stage == "index_project"
        and (
            "exceeded memory limit" in detail
            or "exit code 137" in detail
            or "killed: 9" in detail
            or "timed out after" in detail
            or "stalled after" in detail
        )
    )


def is_index_project_memory_failure(exc: RepoFailure) -> bool:
    return is_retryable_index_project_failure(exc)


_PARSE_WORKERS_RE = re.compile(r"\bUsing\s+(\d+)\s+parse workers\b", re.IGNORECASE)


def _failure_parse_workers(detail: str) -> int | None:
    match = _PARSE_WORKERS_RE.search(detail)
    if match is None:
        return None
    return int(match.group(1))


def failed_code_unit_was_index_memory(
    repo: str,
    manifest: Manifest,
    *,
    parse_workers: int | None = None,
) -> bool:
    rec = manifest.failed.get(code_key(repo))
    if not isinstance(rec, dict):
        return False
    detail = str(rec.get("detail", "")).lower()
    is_retryable_failure = (
        rec.get("stage") == "index_project"
        and (
            "exceeded memory limit" in detail
            or "exit code 137" in detail
            or "killed: 9" in detail
            or "timed out after" in detail
            or "stalled after" in detail
        )
    )
    if not is_retryable_failure:
        return False
    if parse_workers is not None:
        prior_parse_workers = _failure_parse_workers(detail)
        if prior_parse_workers is not None and prior_parse_workers > parse_workers:
            return False
    return True


def code_index_retry_policy(dedup_near: bool) -> tuple[int, bool]:
    """Reduce parser concurrency without weakening the configured dedup policy."""

    return 1, bool(dedup_near)


def run_code_half_adaptive(
    repo: str,
    project_id: str,
    repo_dir: Path,
    lengths_code: Sequence[int],
    work_root: Path,
    dedup_db: Path | None,
    dedup_near: bool,
    global_symbol_index: Path | None = None,
    memory_limit_gb: float = 10.0,
    parse_workers: int = 2,
    index_timeout_s: int | None = None,
    index_stall_timeout_s: int | None = None,
    *,
    runner: CodeRunner | None = None,
    recompressor: BackgroundRecompressor | None = None,
    revision_guard: CodeRevisionGuard | None = None,
    source_quarantine_manifest: Path | None = None,
) -> dict:
    """Run the code half, retrying index_project peaks/stalls with one parser.

    Large repos can peak while merging multiprocessing parse payloads into the
    repo-wide ProjectIndex; some also stall in parser-heavy paths. Retrying with
    one parser preserves the same enriched output contract and graph routes
    while removing avoidable IPC/parser concurrency pressure.
    """
    active_runner = run_code_half if runner is None else runner
    def invoke(active_parse_workers: int, active_dedup_near: bool) -> dict:
        if revision_guard is not None:
            revision_guard.verify(f"submit code pipeline for {repo}")
        args = (
            repo,
            project_id,
            repo_dir,
            lengths_code,
            work_root,
            dedup_db,
            active_dedup_near,
            global_symbol_index,
            memory_limit_gb,
            active_parse_workers,
            index_timeout_s,
            index_stall_timeout_s,
            recompressor,
        )
        if active_runner is run_code_half:
            return active_runner(
                *args,
                revision_guard=revision_guard,
                source_quarantine_manifest=source_quarantine_manifest,
            )
        return active_runner(*args)

    try:
        return invoke(parse_workers, dedup_near)
    except RepoFailure as exc:
        _raise_if_revision_subprocess_failure(exc)
        if parse_workers <= 1 or not is_retryable_index_project_failure(exc):
            raise
        retry_parse_workers, retry_dedup_near = code_index_retry_policy(dedup_near)
        _log(
            f"RETRY {code_key(repo)}: {exc.stage} retryable failure with "
            f"parse_workers={parse_workers}; retrying "
            f"parse_workers={retry_parse_workers} with the configured "
            "global exact/chunk and near-dedup policy preserved"
        )
        return invoke(retry_parse_workers, retry_dedup_near)


RangeRunner = Callable[
    [
        str,
        Path,
        Path,
        int,
        int,
        Sequence[int],
        Path,
        Path | None,
        bool,
        Path | None,
        Path | None,
        float,
        int,
    ],
    dict,
]

CommitRecordsProvider = Callable[
    [
        str,
        Path,
        Path,
        Path,
        Manifest,
        bool,
    ],
    tuple[Path, int, str],
]


def is_splitworthy_range_failure(exc: RepoFailure) -> bool:
    """Return True for range-local peak failures that may shrink by splitting."""
    detail = exc.detail.lower()
    return (
        exc.stage in {"process_commits", "materialize", "pack"}
        and (
            "exceeded memory limit" in detail
            or "exit code 137" in detail
            or "killed: 9" in detail
        )
    )


def process_range_adaptive(
    repo: str,
    repo_dir: Path,
    records_jsonl: Path,
    start: int,
    end: int,
    lengths_sorted: Sequence[int],
    repo_work: Path,
    dedup_db: Path | None,
    dedup_near: bool,
    pr_store: Path | None,
    repo_list: Path | None,
    memory_limit_gb: float,
    *,
    analysis_cache_entries: int = 128,
    min_range_size: int = DEFAULT_MIN_RETRY_RANGE_SIZE,
    runner: RangeRunner | None = None,
    revision_guard: CodeRevisionGuard | None = None,
) -> dict[str, list[tuple[int, int, dict]] | list[tuple[int, int, RepoFailure]]]:
    """Run a commit range, recursively splitting only peak-OOM failures.

    Large C++ repos contain pathological commits/ranges whose enriched sidecars
    transiently peak above the per-stage RSS guard. Splitting the affected range
    preserves the exact same downstream processors and manifest accounting while
    keeping retry work transactional.
    """
    if end <= start:
        raise ValueError(f"invalid range [{start}:{end}]")
    if min_range_size <= 0:
        raise ValueError(f"min_range_size must be positive, got {min_range_size}")
    active_runner = process_range if runner is None else runner

    try:
        if revision_guard is not None:
            revision_guard.verify(
                f"commit range stage for {repo}[{start}:{end}]"
            )
        info = active_runner(
            repo,
            repo_dir,
            records_jsonl,
            start,
            end,
            lengths_sorted,
            repo_work,
            dedup_db,
            dedup_near,
            pr_store,
            repo_list,
            memory_limit_gb,
            analysis_cache_entries,
        )
        return {"done": [(start, end, info)], "failed": []}
    except RepoFailure as exc:
        _raise_if_revision_subprocess_failure(exc)
        if (end - start) <= min_range_size or not is_splitworthy_range_failure(exc):
            return {"done": [], "failed": [(start, end, exc)]}
        mid = start + ((end - start) // 2)
        _log(
            f"SPLIT {range_key(repo, start)} [{start}:{end}] after "
            f"{exc.stage} peak failure -> [{start}:{mid}] + [{mid}:{end}]"
        )
        left = process_range_adaptive(
            repo,
            repo_dir,
            records_jsonl,
            start,
            mid,
            lengths_sorted,
            repo_work,
            dedup_db,
            dedup_near,
            pr_store,
            repo_list,
            memory_limit_gb,
            analysis_cache_entries=analysis_cache_entries,
            min_range_size=min_range_size,
            runner=active_runner,
            revision_guard=revision_guard,
        )
        right = process_range_adaptive(
            repo,
            repo_dir,
            records_jsonl,
            mid,
            end,
            lengths_sorted,
            repo_work,
            dedup_db,
            dedup_near,
            pr_store,
            repo_list,
            memory_limit_gb,
            analysis_cache_entries=analysis_cache_entries,
            min_range_size=min_range_size,
            runner=active_runner,
            revision_guard=revision_guard,
        )
        return {
            "done": [*left["done"], *right["done"]],
            "failed": [*left["failed"], *right["failed"]],
        }


def run_bounded_future_queue(
    items,
    *,
    max_pending: int,
    submit: Callable,
    handle_done: Callable,
    stop_event: threading.Event | None = None,
    cancel_pending_on_stop: bool = False,
) -> int:
    """Submit futures lazily and keep at most ``max_pending`` outstanding.

    ``submit(item)`` returns ``(future, state)`` or ``None`` to skip an item.
    ``handle_done(item, future, state)`` is called for every submitted future,
    including futures cancelled by ``cancel_pending_on_stop``.
    """
    if max_pending < 1:
        raise ValueError("max_pending must be >= 1")

    iterator = iter(items)
    futures = {}
    submitted = 0
    exhausted = False
    cancelled_pending = False

    def fill_window() -> None:
        nonlocal exhausted, submitted
        while len(futures) < max_pending and not exhausted:
            if stop_event is not None and stop_event.is_set():
                return
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                return
            submitted_future = submit(item)
            if submitted_future is None:
                continue
            future, state = submitted_future
            futures[future] = (item, state)
            submitted += 1

    fill_window()
    while futures:
        if (
            stop_event is not None
            and stop_event.is_set()
            and cancel_pending_on_stop
            and not cancelled_pending
        ):
            for future in list(futures):
                if not future.done():
                    future.cancel()
            cancelled_pending = True
        done, _pending = wait(futures.keys(), return_when=FIRST_COMPLETED)
        for future in done:
            item, state = futures.pop(future)
            handle_done(item, future, state)
        fill_window()

    return submitted


def handle_repo_future_result(
    future: Future,
    repo: str,
    manifest: Manifest,
    manifest_lock: threading.Lock,
) -> tuple[int, int, int, int]:
    """Consume one repo future and persist any escaped failure immediately."""
    try:
        result = future.result()
    except CodeRevisionDriftError:
        raise
    except RepoFailure as exc:
        _raise_if_revision_subprocess_failure(exc)
        _log(f"FAIL {repo}::repo: {exc}")
        with manifest_lock:
            manifest.mark_failed(f"{repo}::repo", exc.stage, exc.detail)
        return 1, 0, 0, 1
    except SymbolIdentityError:
        raise
    except Exception as exc:  # surface unexpected worker failures loudly
        _log(f"FAIL {repo}::repo: unexpected {type(exc).__name__}: {exc}")
        with manifest_lock:
            manifest.mark_failed(f"{repo}::repo", "unexpected", str(exc))
        return 1, 0, 0, 1
    return (
        1,
        1 if isinstance(result.get("code"), dict) else 0,
        int(result.get("commits_done", 0)),
        int(result.get("commits_failed", 0)),
    )


def guard_source_submissions(source, revision_guard: CodeRevisionGuard):
    """Verify immediately before advancing a source iterator that may spawn."""
    iterator = iter(source)
    try:
        while True:
            revision_guard.verify("source stream submission")
            try:
                yield next(iterator)
            except StopIteration:
                return
    finally:
        if hasattr(iterator, "close"):
            iterator.close()


def stream_source_with_repo_future_drain(
    source,
    inflight: dict[Future, str],
    handle_repo_done: Callable[[Future, str], None],
    *,
    poll_interval_s: float = 0.25,
):
    """Yield source items while observing repo futures during blocking extraction.

    Only ``next(source)`` runs in the producer thread. The caller remains in this
    polling loop, so a long tar extraction cannot hide a completed repo failure.
    """
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    iterator = iter(source)
    try:
        with ThreadPoolExecutor(max_workers=1) as source_pool:
            while True:
                source_future = source_pool.submit(next, iterator)
                while True:
                    watched = (source_future, *tuple(inflight))
                    completed, _pending = wait(
                        watched,
                        timeout=poll_interval_s,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in completed:
                        if future is source_future:
                            continue
                        repo = inflight.pop(future)
                        handle_repo_done(future, repo)
                    if source_future.done():
                        break
                try:
                    yield source_future.result()
                except StopIteration:
                    return
    finally:
        if hasattr(iterator, "close"):
            iterator.close()


# --------------------------------------------------------------------------- #
# COMMITS half: orchestrate the commit range pipeline (no reimplementation).
# extract_git_history once -> delete .git -> fan ranges to the pool via the
# EXISTING process_range (which runs process_commits --pr-store/--repo-list with
# the SHARED dedup_db, then materialize/route/pack/zstd-max).
# --------------------------------------------------------------------------- #
def run_commits_half(
    repo: str,
    repo_dir: Path,
    repo_work: Path,
    work_root: Path,
    work_parent: Path,
    lengths_commits: Sequence[int],
    range_size: int,
    pool: ThreadPoolExecutor,
    manifest: Manifest,
    manifest_lock: threading.Lock,
    resume: bool,
    cumulative: dict,
    dedup_db: Path | None,
    dedup_near: bool,
    pr_store: Path | None,
    repo_list: Path | None,
    memory_limit_gb: float = 10.0,
    progress: ProgressWriter | None = None,
    checkpoint: DedupCheckpointController | None = None,
    reservations: UnitReservationLedger | None = None,
    range_submit_window: int | None = None,
    analysis_cache_entries: int = 128,
    range_target_bytes: int = DEFAULT_RANGE_TARGET_BYTES,
    dedup_promote_batch_size: int = DEFAULT_DEDUP_PROMOTE_BATCH_SIZE,
    range_runner_override: RangeRunner | None = None,
    commit_records_override: CommitRecordsProvider | None = None,
    revision_guard: CodeRevisionGuard | None = None,
    project_id: str | None = None,
    pr_scan_id: str | None = None,
) -> tuple[int, int, bool]:
    """Extract commit records once (checkpointed), fan ranges to the pool.

    Returns ``(done, failed, all_ranges_done)``. The records jsonl comes from
    :func:`ensure_commit_records`, which REUSES a stable cached extract whenever
    possible (never re-running the ~6h extract_git_history on resume). .git is
    deleted immediately after the records are available, keeping disk bounded.
    Each range is checkpointed exactly as ``<repo>::r<start>`` so resume skips
    finished ranges. On STOP_EVENT we stop submitting new ranges and cancel any
    queued-but-unstarted ones (their units stay un-marked -> resume re-runs them),
    while in-flight ranges drain. RAISES RepoFailure only for the up-front
    git-log / extract steps; per-range failures are recorded and skipped.
    ``all_ranges_done`` is True iff EVERY range key is marked done afterwards
    (drives temp/extract-cache retention).
    """
    lengths_sorted = tuple(sorted(int(x) for x in lengths_commits))
    smallest = lengths_sorted[0]
    canonical_project_id: str | None = None
    if project_id is not None:
        canonical_project_id = _resolve_commit_project_id(repo, repo_dir, project_id)
    elif commit_records_override is None:
        mapped_project_id = (
            sr.resolve_project_identity(repo, repo_list)
            if repo_list is not None
            else None
        )
        canonical_project_id = _resolve_commit_project_id(
            repo,
            repo_dir,
            mapped_project_id,
        )

    if not resume:
        with manifest_lock:
            manifest.mark_started_prefix(f"{repo}::r")
            manifest.mark_started(commit_plan_key(repo))
            manifest.mark_started(f"{repo}::commits")

    if resume:
        complete_ranges = manifest_complete_commit_ranges(repo, manifest, range_size)
        if complete_ranges is not None:
            first = complete_ranges[0][0]
            last = complete_ranges[-1][1]
            mark_commit_stream_complete(
                repo,
                manifest,
                manifest_lock,
                complete_ranges,
            )
            _log(
                f"SKIP (done) {repo}::commits: manifest covers "
                f"{len(complete_ranges)} range(s) [{first}:{last}); "
                "skip ~extract_git_history"
            )
            return 0, 0, True

    # Reuse a cached/adopted extract when available; never re-run the ~6h extract
    # for a repo whose commit extraction already completed (sentinel or manifest).
    try:
        if commit_records_override is None:
            records_jsonl, n_records, extract_cache_status = ensure_commit_records(
                repo,
                repo_dir,
                work_root,
                work_parent,
                manifest,
                resume,
                revision_guard=revision_guard,
                project_id=canonical_project_id,
            )
        else:
            if revision_guard is not None:
                revision_guard.verify(f"commit records provider for {repo}")
            records_jsonl, n_records, extract_cache_status = commit_records_override(
                repo, repo_dir, work_root, work_parent, manifest, resume,
            )
    except RepoNoCommitRecords as exc:
        raise RepoFailure(
            repo,
            "extract_git_history",
            f"no commit records ({exc.reason}): {exc.detail}",
        ) from exc
    extract_cache_receipt = extract_cache_access_receipt(extract_cache_status)
    if progress is not None:
        progress.emit(
            "extract_cache",
            stream="commits",
            repo=repo,
            **extract_cache_receipt,
            n_records=n_records,
            records_jsonl=str(records_jsonl),
        )

    # This is the only safe EOF proof. Old manifests without it are deliberately
    # re-staged and counted from the extraction cache before ranges are skipped.
    records_stat = records_jsonl.stat()
    with manifest_lock:
        manifest.mark_done(
            commit_plan_key(repo),
            {
                "source": "commit_plan",
                "repo": repo,
                "n_records": int(n_records),
                "records_size_bytes": int(records_stat.st_size),
                "extract_cache": extract_cache_receipt,
            },
        )

    # .git no longer needed -> free disk now (records already captured/cached).
    git_dir = repo_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)

    # Range over ACTUAL emitted record count (extract keeps only C/C++-touching
    # commits, so the records JSONL is a subset; slicing by line index aligns).
    if n_records == 0:
        raise RepoFailure(repo, "extract_git_history",
                          f"zero records after extract for {repo}")
    ranges = plan_commit_ranges(
        records_jsonl,
        n_records,
        max_records=range_size,
        target_bytes=range_target_bytes,
    )
    if progress is not None:
        range_lengths = [end - start for start, end in ranges]
        progress.emit(
            "commit_range_plan",
            stream="commits",
            repo=repo,
            n_records=n_records,
            range_count=len(ranges),
            range_size=range_size,
            range_target_bytes=range_target_bytes,
            min_range_records=min(range_lengths) if range_lengths else 0,
            max_range_records=max(range_lengths) if range_lengths else 0,
        )
    if ranges:
        range_lengths = [end - start for start, end in ranges]
        _log(
            f"Commit range plan for {repo}: {len(ranges)} ranges, "
            f"records={n_records}, max_records={range_size}, "
            f"target_bytes={range_target_bytes}, "
            f"range_records=[{min(range_lengths)}..{max(range_lengths)}]"
        )

    done_intervals = manifest_done_commit_intervals(repo, manifest) if resume else ()
    pending_ranges: list[tuple[int, int]] = []
    for (start, end) in ranges:
        if STOP_EVENT.is_set():
            _log(f"STOP: not submitting further ranges for {repo} (from r{start})")
            break
        pending_subranges = (
            missing_commit_subranges(start, end, done_intervals)
            if resume
            else ((start, end),)
        )
        if not pending_subranges:
            _log(f"SKIP (done) {range_key(repo, start)}")
            continue
        for sub_start, sub_end in pending_subranges:
            pending_ranges.append((sub_start, sub_end))

    submit_window = max(
        1,
        int(range_submit_window or len(pending_ranges) or 1),
    )
    if pending_ranges and submit_window < len(pending_ranges):
        _log(
            f"Commit range queue for {repo}: {len(pending_ranges)} pending, "
            f"submit_window={submit_window}"
        )

    done = 0
    failed = 0
    promote_batch_size = max(1, int(dedup_promote_batch_size or 1))
    defer_range_promotes = dedup_db is not None and promote_batch_size > 1
    deferred_stage_dir = repo_work / "_deferred_promote"
    deferred_promotions: list[
        tuple[list[tuple[int, int, dict]], WorkReservation | None]
    ] = []

    def unlink_sqlite_family(db_path: Path) -> None:
        for path in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def range_runner(
        range_repo,
        range_repo_dir,
        range_records_jsonl,
        range_start,
        range_end,
        range_lengths_sorted,
        range_repo_work,
        range_dedup_db,
        range_dedup_near,
        range_pr_store,
        range_repo_list,
        range_memory_limit_gb,
        range_analysis_cache_entries,
    ):
        args = (
            range_repo,
            range_repo_dir,
            range_records_jsonl,
            range_start,
            range_end,
            range_lengths_sorted,
            range_repo_work,
            range_dedup_db,
            range_dedup_near,
            range_pr_store,
            range_repo_list,
            range_memory_limit_gb,
            range_analysis_cache_entries,
        )
        active_process_range = (
            process_range if range_runner_override is None else range_runner_override
        )
        scan_kwargs = (
            {"pr_scan_id": pr_scan_id}
            if active_process_range is process_range
            else {}
        )
        if defer_range_promotes:
            return active_process_range(
                *args,
                defer_promote=True,
                deferred_stage_dir=deferred_stage_dir,
                **scan_kwargs,
            )
        return active_process_range(*args, **scan_kwargs)

    def mark_failed_range(
        failed_start: int,
        failed_end: int,
        exc: RepoFailure,
    ) -> None:
        nonlocal failed
        frkey = range_key(repo, failed_start)
        _log(f"FAIL {frkey}: {exc}")
        failed += 1
        with manifest_lock:
            manifest.mark_failed(frkey, exc.stage, exc.detail)
        if progress is not None:
            progress.emit(
                "unit_failed",
                stream="commits",
                repo=repo,
                unit=frkey,
                range=[failed_start, failed_end],
                stage=exc.stage,
                detail=exc.detail[:2000],
            )

    def mark_done_range(done_start: int, done_end: int, rinfo: dict) -> None:
        nonlocal done
        drkey = range_key(repo, done_start)
        rinfo["extract_cache_status"] = extract_cache_status
        rinfo["extract_cache"] = extract_cache_receipt
        with manifest_lock:
            manifest.mark_done(drkey, rinfo)
        done += 1
        totals = _length_totals(rinfo)
        added = rinfo["lengths"].get(str(smallest), {}).get("valid_tokens", 0)
        with manifest_lock:
            cumulative["valid"] += totals["valid_tokens"]
            cumulative_valid = cumulative["valid"]
        if progress is not None:
            progress.emit(
                "unit_done",
                stream="commits",
                repo=repo,
                unit=drkey,
                range=[done_start, done_end],
                rows=totals["rows"],
                valid_tokens=totals["valid_tokens"],
                capacity_tokens=totals["capacity_tokens"],
                **_primary_bucket_progress(rinfo, lengths_sorted),
                lengths=rinfo.get("lengths", {}),
                stage_timings_s=rinfo.get("stage_timings_s", {}),
                extract_cache_status=extract_cache_status,
                extract_cache=extract_cache_receipt,
                cumulative_valid_tokens=cumulative_valid,
            )
        if checkpoint is not None:
            checkpoint.maybe_checkpoint(cumulative_valid)
        _log(
            f"DONE {drkey}: ranges [{done_start}:{done_end}] "
            f"buckets={sorted(rinfo['lengths'].keys())} "
            f"(+{added} @ {smallest}, cum_all={cumulative['valid']})"
        )

    def flush_deferred_promotions() -> None:
        if not deferred_promotions:
            return
        batch = list(deferred_promotions)
        deferred_promotions.clear()
        stages: list[tuple[str, Path | None]] = []
        stage_paths: list[Path] = []
        try:
            # Batch promotion is the commit barrier: publish no range receipt
            # unless every staged dedup transaction promotes successfully.
            try:
                for done_items, _reservation in batch:
                    for _done_start, _done_end, rinfo in done_items:
                        stage = rinfo.get("dedup_stage")
                        if not stage:
                            continue
                        stage_id = stage["stage_id"]
                        stage_db = Path(stage["stage_db"])
                        stages.append((stage_id, stage_db))
                        stage_paths.append(stage_db)
                metrics = sr.promote_dedup_stages(dedup_db, stages)
            except Exception as exc:
                failure = (
                    exc
                    if isinstance(exc, RepoFailure)
                    else RepoFailure(
                        repo,
                        "dedup_promote",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                for done_items, _reservation in batch:
                    for failed_start, failed_end, _rinfo in done_items:
                        mark_failed_range(failed_start, failed_end, failure)
                if isinstance(exc, SymbolIdentityError):
                    raise
                return

            stage_count = max(
                1,
                int(metrics.get("promote_batch_size") or len(stages) or 1),
            )
            per_wait = float(metrics.get("promote_wait_s", 0.0)) / stage_count
            per_duration = float(metrics.get("promote_duration_s", 0.0)) / stage_count
            for done_items, _reservation in batch:
                for done_start, done_end, rinfo in done_items:
                    stage = rinfo.pop("dedup_stage", None)
                    if stage:
                        timings = dict(rinfo.get("stage_timings_s", {}))
                        timings["promote_wait_s"] = round(per_wait, 6)
                        timings["promote_duration_s"] = round(per_duration, 6)
                        timings["promote_batch_size"] = stage_count
                        timings["promote_batch_wait_s"] = metrics.get(
                            "promote_wait_s", 0.0
                        )
                        timings["promote_batch_duration_s"] = metrics.get(
                            "promote_duration_s", 0.0
                        )
                        timings["promote_deferred"] = 0.0
                        rinfo["stage_timings_s"] = timings
                        rinfo["dedup_stage_promoted"] = {
                            "stage_id": stage["stage_id"],
                        }
                    mark_done_range(done_start, done_end, rinfo)
        finally:
            try:
                for _done_items, reservation in batch:
                    if reservation is not None:
                        reservation.release()
            finally:
                for stage_db in stage_paths:
                    unlink_sqlite_family(stage_db)

    def submit_commit_range(item: tuple[int, int]):
        sub_start, sub_end = item
        rkey = range_key(repo, sub_start)
        if STOP_EVENT.is_set():
            _log(f"STOP: not submitting further ranges for {repo} (from r{sub_start})")
            return None
        reservation: WorkReservation | None = None
        if reservations is not None:
            reservation = reservations.acquire(rkey, stream="commits", repo=repo)
            if not reservation.acquired:
                _log(
                    f"SKIP (reserved) {rkey}: holder="
                    f"{reservation.holder}"
                )
                return None
        try:
            if revision_guard is not None:
                revision_guard.verify(
                    f"submit commit range for {repo}[{sub_start}:{sub_end}]"
                )
            with manifest_lock:
                manifest.mark_started(rkey)
            if progress is not None:
                progress.emit(
                    "unit_started",
                    stream="commits",
                    repo=repo,
                    unit=rkey,
                    range=[sub_start, sub_end],
                    extract_cache_status=extract_cache_status,
                    extract_cache=extract_cache_receipt,
                )
            fut = pool.submit(
                process_range_adaptive, repo, repo_dir, records_jsonl,
                sub_start, sub_end, lengths_sorted, repo_work,
                dedup_db, dedup_near, pr_store, repo_list,
                memory_limit_gb,
                analysis_cache_entries=analysis_cache_entries,
                runner=range_runner,
                revision_guard=revision_guard,
            )
        except Exception:
            if reservation is not None:
                reservation.release()
            raise
        return fut, reservation

    def handle_commit_range(
        item: tuple[int, int],
        fut,
        reservation: WorkReservation | None,
    ) -> None:
        nonlocal done, failed
        start, end = item
        try:
            adaptive_result = fut.result()
        except CodeRevisionDriftError:
            raise
        except CancelledError:
            # Never started -> leave un-marked so resume re-runs this range.
            return
        except RepoFailure as exc:
            _raise_if_revision_subprocess_failure(exc)
            mark_failed_range(start, end, exc)
            return
        except SymbolIdentityError:
            raise
        except Exception as exc:  # surface unexpected failures loud
            mark_failed_range(
                start,
                end,
                RepoFailure(
                    repo,
                    "unexpected",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            return
        else:
            for failed_start, failed_end, exc in adaptive_result["failed"]:
                mark_failed_range(failed_start, failed_end, exc)

            done_items = list(adaptive_result["done"])
            if done_items and defer_range_promotes and any(
                bool(rinfo.get("dedup_stage"))
                for _done_start, _done_end, rinfo in done_items
            ):
                deferred_promotions.append((done_items, reservation))
                reservation = None
                if len(deferred_promotions) >= promote_batch_size:
                    flush_deferred_promotions()
            else:
                for done_start, done_end, rinfo in done_items:
                    mark_done_range(done_start, done_end, rinfo)
        finally:
            if reservation is not None:
                reservation.release()

    run_bounded_future_queue(
        pending_ranges,
        max_pending=submit_window,
        submit=submit_commit_range,
        handle_done=handle_commit_range,
        stop_event=STOP_EVENT,
        cancel_pending_on_stop=True,
    )
    flush_deferred_promotions()

    # True iff EVERY range for this repo is now marked done in the manifest
    # (covers resume-skipped + newly-done; excludes cancelled/failed). Drives
    # temp + extract-cache retention in process_one_repo.
    complete_ranges = manifest_complete_commit_ranges(repo, manifest, range_size)
    all_ranges_done = complete_ranges is not None
    if complete_ranges is not None:
        mark_commit_stream_complete(
            repo,
            manifest,
            manifest_lock,
            complete_ranges,
        )
    return done, failed, all_ranges_done


# --------------------------------------------------------------------------- #
# Per-repo conveyor unit: ONE extraction, BOTH halves, then delete the repo.
# --------------------------------------------------------------------------- #
def process_one_repo(
    repo: str,
    repo_dir: Path,
    lengths_code: Sequence[int],
    lengths_commits: Sequence[int],
    range_size: int,
    range_target_bytes: int,
    work_root: Path,
    work_parent: Path,
    pool: ThreadPoolExecutor,
    manifest: Manifest,
    manifest_lock: threading.Lock,
    resume: bool,
    cumulative: dict,
    keep_temp: bool,
    dedup_db: Path | None,
    dedup_near: bool,
    pr_store: Path | None,
    repo_list: Path | None,
    streams: str = "both",
    global_symbol_index: Path | None = None,
    memory_limit_gb: float = 10.0,
    parse_workers: int = 2,
    progress: ProgressWriter | None = None,
    checkpoint: DedupCheckpointController | None = None,
    *,
    code_memory_limit_gb: float | None = None,
    commit_memory_limit_gb: float | None = None,
    code_index_timeout_s: int | None = None,
    code_index_stall_timeout_s: int | None = None,
    code_recompressor: BackgroundRecompressor | None = None,
    reservations: UnitReservationLedger | None = None,
    range_submit_window: int = 1,
    analysis_cache_entries: int = 128,
    dedup_promote_batch_size: int = DEFAULT_DEDUP_PROMOTE_BATCH_SIZE,
    retain_partial_work: bool = False,
    revision_guard: CodeRevisionGuard | None = None,
    pr_scan_id: str | None = None,
    source_quarantine_manifest: Path | None = None,
) -> dict:
    """Run BOTH halves for one already-extracted repo subtree, then delete it.

    The repo was extracted ONCE (incl .git) by the .git-preserving stream into
    ``repo_dir`` (== work_root/<repo>/_src). The CODE half runs first (it does
    NOT touch .git); then the COMMITS half consumes + deletes .git. The repo work
    dir (and a run-local extract cache) are removed at the end. An explicitly
    configured external extract cache is never removed. Fully-done repos are
    always cleaned. Interrupted / failed / partial repos are also cleaned by
    default so the conveyor cannot fill the disk with raw source clones; use
    --retain-partial-work for the older zero-rework resume mode. RULE #1: a
    failure in one half is recorded; the other half still runs.
    """
    repo_work = work_root / repo
    repo_work.mkdir(parents=True, exist_ok=True)
    result = {"repo": repo, "code": None, "commits_done": 0, "commits_failed": 0}
    all_ranges_done = streams not in {"both", "commits"}  # True when commits disabled
    code_limit = memory_limit_gb if code_memory_limit_gb is None else code_memory_limit_gb
    commit_limit = (
        memory_limit_gb if commit_memory_limit_gb is None else commit_memory_limit_gb
    )
    try:
        # ---- CODE half (skip if already done) ----
        if streams in {"both", "code"}:
            ck = code_key(repo)
            if not sr.is_code_worktree_repo(repo):
                info = {
                    "source": "code",
                    "repo": repo,
                    "skipped": True,
                    "reason": "bare git repository, not a source worktree",
                    "lengths": {},
                    "stage_timings_s": {},
                }
                with manifest_lock:
                    manifest.mark_done(ck, info)
                if progress is not None:
                    progress.emit(
                        "unit_skipped",
                        stream="code",
                        repo=repo,
                        unit=ck,
                        reason=info["reason"],
                    )
                result["code"] = "skipped_non_worktree"
            elif resume and manifest.is_done(ck):
                _log(f"SKIP (done) {ck}")
                result["code"] = "skipped"
            else:
                code_reservation: WorkReservation | None = None
                run_code_unit = True
                try:
                    if reservations is not None:
                        code_reservation = reservations.acquire(
                            ck,
                            stream="code",
                            repo=repo,
                        )
                        if not code_reservation.acquired:
                            _log(
                                f"SKIP (reserved) {ck}: holder="
                                f"{code_reservation.holder}"
                            )
                            result["code"] = "reserved"
                            code_reservation = None
                            run_code_unit = False
                    if run_code_unit:
                        code_parse_workers = parse_workers
                        code_dedup_db = dedup_db
                        code_dedup_near = dedup_near
                        if (
                            parse_workers > 1
                            and resume
                            and failed_code_unit_was_index_memory(
                                repo,
                                manifest,
                                parse_workers=parse_workers,
                            )
                        ):
                            code_parse_workers, code_dedup_near = (
                                code_index_retry_policy(dedup_near)
                            )
                            code_dedup_db = dedup_db
                            _log(
                                f"RETRY {ck}: prior manifest failure was "
                                "retryable index_project failure; starting "
                                f"parse_workers={code_parse_workers} with the configured "
                                "global exact/chunk and near-dedup policy preserved"
                            )
                        with manifest_lock:
                            manifest.mark_started(ck)
                        if progress is not None:
                            progress.emit(
                                "unit_started",
                                stream="code",
                                repo=repo,
                                unit=ck,
                                parse_workers=code_parse_workers,
                                memory_limit_gb=code_limit,
                                index_timeout_s=code_index_timeout_s,
                                index_stall_timeout_s=code_index_stall_timeout_s,
                            )
                        try:
                            project_identity = sr.resolve_project_identity(repo, repo_list)
                        except SymbolIdentityError as exc:
                            raise RepoFailure(repo, "project_identity", str(exc)) from exc
                        cinfo = run_code_half_adaptive(
                            repo,
                            project_identity,
                            repo_dir, lengths_code, work_root,
                            code_dedup_db, code_dedup_near,
                            global_symbol_index, code_limit, code_parse_workers,
                            code_index_timeout_s,
                            code_index_stall_timeout_s,
                            recompressor=code_recompressor,
                            revision_guard=revision_guard,
                            source_quarantine_manifest=(
                                source_quarantine_manifest
                            ),
                        )
                        if cinfo.get("skipped"):
                            with manifest_lock:
                                manifest.mark_done(ck, cinfo)
                            if progress is not None:
                                progress.emit(
                                    "unit_skipped",
                                    stream="code",
                                    repo=repo,
                                    unit=ck,
                                    reason=cinfo.get("skip_reason", "skipped"),
                                    detail=str(cinfo.get("detail", ""))[:2000],
                                )
                            result["code"] = cinfo
                            _log(
                                f"SKIP {ck}: {cinfo.get('skip_reason', 'skipped')}"
                            )
                        else:
                            with manifest_lock:
                                manifest.mark_done(ck, cinfo)
                            totals = _length_totals(cinfo)
                            with manifest_lock:
                                cumulative["valid"] += totals["valid_tokens"]
                                cumulative_valid = cumulative["valid"]
                            if progress is not None:
                                progress.emit(
                                    "unit_done",
                                    stream="code",
                                    repo=repo,
                                    unit=ck,
                                    rows=totals["rows"],
                                    valid_tokens=totals["valid_tokens"],
                                    capacity_tokens=totals["capacity_tokens"],
                                    **_primary_bucket_progress(cinfo, lengths_code),
                                    lengths=cinfo.get("lengths", {}),
                                    stage_timings_s=cinfo.get("stage_timings_s", {}),
                                    cumulative_valid_tokens=cumulative_valid,
                                )
                            if checkpoint is not None:
                                checkpoint.maybe_checkpoint(cumulative_valid)
                            result["code"] = cinfo
                            _log(
                                f"DONE {ck}: buckets={sorted(cinfo['lengths'].keys())} "
                                f"(cum_all={cumulative['valid']})"
                            )
                except CodeRevisionDriftError:
                    raise
                except RepoFailure as exc:
                    _raise_if_revision_subprocess_failure(exc)
                    _log(f"FAIL {ck}: {exc}")
                    with manifest_lock:
                        manifest.mark_failed(ck, exc.stage, exc.detail)
                    if progress is not None:
                        progress.emit(
                            "unit_failed",
                            stream="code",
                            repo=repo,
                            unit=ck,
                            stage=exc.stage,
                            detail=exc.detail[:2000],
                    )
                    result["code"] = "failed"
                finally:
                    if code_reservation is not None:
                        code_reservation.release()
        else:
            result["code"] = "disabled"

        # ---- COMMITS half (per-range resume + checkpoint inside) ----
        if streams in {"both", "commits"}:
            try:
                done, failed, all_ranges_done = run_commits_half(
                    repo, repo_dir, repo_work, work_root, work_parent,
                    lengths_commits, range_size,
                    pool, manifest, manifest_lock, resume, cumulative,
                    dedup_db, dedup_near, pr_store, repo_list, commit_limit,
                    progress, checkpoint, reservations, range_submit_window,
                    analysis_cache_entries,
                    range_target_bytes=range_target_bytes,
                    dedup_promote_batch_size=dedup_promote_batch_size,
                    revision_guard=revision_guard,
                    pr_scan_id=pr_scan_id,
                    source_quarantine_manifest=source_quarantine_manifest,
                )
                result["commits_done"] = done
                result["commits_failed"] = failed
            except CodeRevisionDriftError:
                raise
            except RepoFailure as exc:
                _raise_if_revision_subprocess_failure(exc)
                _log(f"FAIL {repo}::commits: {exc}")
                all_ranges_done = False
                with manifest_lock:
                    manifest.mark_failed(f"{repo}::commits", exc.stage, exc.detail)
        return result
    finally:
        # Production default is disk-bounded: keep completed parquet + manifest,
        # but delete raw source / JSONL / stage scratch even when a repo is only
        # partially done. Resume re-extracts unfinished units instead of pinning
        # hundreds of GiB in outputs/conveyor/tmp. The old zero-rework behavior is
        # still explicit via --retain-partial-work or --keep-temp.
        fully_done = repo_fully_done(repo, manifest, streams, all_ranges_done)
        if keep_temp:
            pass
        elif fully_done:
            remove_tree(repo_work, reason=f"{repo} fully done work")
            if extract_cache_is_external():
                _log(
                    f"RETAIN external extract cache for completed {repo}: "
                    f"{extract_cache_dir(repo)}"
                )
            else:
                remove_tree(
                    extract_cache_dir(repo),
                    reason=f"{repo} fully done extract cache",
                )
        elif retain_partial_work:
            _log(f"RETAIN temp for {repo}: not all units marked done "
                 f"(interrupted/failed/partial); kept {repo_work} + extract cache "
                 f"{extract_cache_dir(repo)} for zero-rework resume")
        elif STOP_EVENT.is_set():
            remove_tree(repo_work, reason=f"{repo} interrupted partial work")
            _log(f"RETAIN extract cache for interrupted {repo}: "
                 f"{extract_cache_dir(repo)} kept for checkpoint resume")
        else:
            remove_tree(repo_work, reason=f"{repo} partial work")
            if extract_cache_is_external():
                _log(
                    f"RETAIN external extract cache for failed/partial {repo}: "
                    f"{extract_cache_dir(repo)}"
                )
            elif has_resumable_extract_checkpoint(repo):
                _log(
                    f"RETAIN extract checkpoint for failed/partial {repo}: "
                    f"{extract_cache_dir(repo)} kept; committed extraction chunks "
                    "will resume without reprocessing"
                )
            else:
                remove_tree(
                    extract_cache_dir(repo),
                    reason=f"{repo} partial extract cache",
                )
                _log(
                    f"CLEANUP partial for {repo}: unfinished units remain "
                    "unmarked and no extraction checkpoint exists; resume will "
                    "re-extract/re-run only unfinished work from manifest"
                )


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--target-lengths-code",
        default=",".join(str(length) for length in DEFAULT_TARGET_LENGTHS_CODE),
        help="Route-by-fit ladder for CODE "
             "(default 1024,2048,4096,8192,16384).",
    )
    p.add_argument("--target-lengths-commits", default="1024,2048,4096,8192,16384",
                   help="Route-by-fit ladder for COMMITS "
                        "(default 1024,2048,4096,8192,16384).")
    p.add_argument("--range-size", type=int, default=DEFAULT_RANGE_SIZE,
                   help="Commits per checkpointed range (default 500).")
    p.add_argument("--range-target-bytes", type=int,
                   default=DEFAULT_RANGE_TARGET_BYTES,
                   help="Soft cap for raw extracted commit-record JSONL bytes per "
                        "range. The conveyor still respects --range-size as a hard "
                        "record-count cap. Use 0 to restore fixed --range-size "
                        f"ranges. Default {DEFAULT_RANGE_TARGET_BYTES}.")
    p.add_argument("--range-submit-window", type=int, default=0,
                   help="Maximum outstanding commit range futures per repo. "
                        "Default 0 means 2 * --workers. Bounds reservation "
                        "claims and ThreadPoolExecutor queue size for huge repos.")
    p.add_argument("--dedup-promote-batch-size", type=int,
                   default=DEFAULT_DEDUP_PROMOTE_BATCH_SIZE,
                   help="Commit ranges defer local dedup stage promotion and the "
                        "parent promotes this many completed stages while holding "
                        "the global SQLite writer lock once. Use 1 to restore "
                        "per-range promotion. "
                        f"Default {DEFAULT_DEDUP_PROMOTE_BATCH_SIZE}.")
    p.add_argument("--analysis-cache-entries", type=int, default=128,
                   help="Bounded per-process LRU entries passed to "
                        "process_commits.py for repeated old/new file analyses. "
                        "Default 128; use 0 to disable.")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="ThreadPoolExecutor size for commit ranges "
                        "(default = os.cpu_count()).")
    p.add_argument("--repo-workers", type=int, default=1,
                   help="Number of repos to process concurrently. Default 1 "
                        "preserves the legacy one-repo-at-a-time conveyor.")
    p.add_argument("--max-active-repos", type=int, default=None,
                   help="Bound staged, not-yet-finished repos on disk. Default "
                        "= --repo-workers. Use 20-30 for a wide streaming window "
                        "when disk has room.")
    p.add_argument("--streams", choices=("both", "code", "commits"), default="both",
                   help="Which streams to emit. 'code' uses the source-only tar "
                        "stream and does not extract .git / run PR/commit stages.")
    p.add_argument("--max-repos", type=int, default=None,
                   help="Process at most N repos this run (after resume filtering).")
    p.add_argument("--token-budget", type=int, default=None,
                   help="Stop after cumulative valid tokens (code + all commit "
                        "buckets) reaches this.")
    p.add_argument(
        "--expected-code-revision",
        default=None,
        help="Required exact 40-character Git commit for a production run. "
             "The executable source/config worktree must also be clean.",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--retain-partial-work", action="store_true",
                   help="Retain raw source clones, range scratch and extracted "
                        "commit JSONL for interrupted/failed/partial repos. "
                        "Default is OFF: completed parquet + manifest are kept, "
                        "but intermediates are deleted so disk stays bounded.")
    p.add_argument("--min-free-disk-gb", type=float,
                   default=DEFAULT_MIN_FREE_DISK_GB,
                   help="Fail loud before staging a new repo when the filesystem "
                        "under --work-parent-dir has less free space than this. "
                        "Use 0 to disable. "
                        f"Default {DEFAULT_MIN_FREE_DISK_GB:.0f} GiB.")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--work-parent-dir", default=str(DEFAULT_WORK_PARENT),
                   help="Parent directory for conveyor temporary work dirs when "
                        "--work-dir is not set. Default lives under outputs/ on "
                        "the external corpus disk, not the macOS /var/folders "
                        f"temp volume: {DEFAULT_WORK_PARENT}.")
    p.add_argument("--code-output-root", default=str(sr.OUTPUT_ROOT),
                   help="Output root for packed CODE parquet buckets. "
                        f"Default {sr.OUTPUT_ROOT}.")
    p.add_argument("--commit-output-root", default=str(src.COMMIT_OUTPUT_ROOT),
                   help="Output root for packed COMMIT parquet buckets. "
                        f"Default {src.COMMIT_OUTPUT_ROOT}.")
    p.add_argument("--conveyor-root", default=str(CONVEYOR_ROOT),
                   help="Root for conveyor manifest, locks, extract cache, "
                        f"progress and tmp defaults. Default {CONVEYOR_ROOT}.")
    p.add_argument(
        "--extract-cache-root",
        default=None,
        help="Shared persistent root for per-repository commit extraction "
             "checkpoints and published JSONL. When omitted, the cache remains "
             "run-local at <conveyor-root>/extract_cache and keeps the existing "
             "cleanup behavior. An explicit root is externally owned, validated "
             "through extract_git_history publication metadata, and never deleted "
             "by this conveyor.",
    )
    p.add_argument("--only-repo", action="append", default=[],
                   help="Restrict the conveyor to these repo names (repeatable). "
                        "Other repos are DRAINED from the .git-preserving stream "
                        "without touching disk. Useful for targeted (re)runs.")
    p.add_argument("--dedup-db", default=str(DEFAULT_DEDUP_DB),
                   help="SHARED global dedup SQLite store threaded into BOTH the "
                        "code stage (function-level tokenized-hash) AND every "
                        "commit range stage (commit-doc tokenized-hash). "
                        f"Default {DEFAULT_DEDUP_DB}.")
    p.add_argument("--no-near-dedup", action="store_true",
                   help="Disable MinHash-LSH near dedup (exact-only) for both.")
    p.add_argument(
        "--pr-store",
        default=None,
        help="Explicit Tier-2 PR-discussion SQLite store passed to "
             "process_commits. Required for commits/both together with "
             "--pr-completion-receipt; there is no implicit production default.",
    )
    p.add_argument(
        "--repo-list",
        default=None,
        help="Explicit repo_list.json for resolving PR-store keys. Required for "
             "commits/both together with --pr-completion-receipt.",
    )
    p.add_argument(
        "--source-quarantine-manifest",
        default=str(sr.DEFAULT_SOURCE_QUARANTINE_MANIFEST),
        help="Exact path+size+SHA manifest for verified non-C++ files stored "
             "under C/C++ suffixes. Passed to every CODE indexer; "
             f"default {sr.DEFAULT_SOURCE_QUARANTINE_MANIFEST}.",
    )
    p.add_argument(
        "--pr-completion-receipt",
        default=None,
        help="Verified cppmega_pr_completion_v2 JSON receipt. Required for "
             "commits/both and hash-checked against the explicit --pr-store and "
             "--repo-list before any work starts.",
    )
    p.add_argument("--global-symbol-index", default=None,
                   help="Path to the GLOBAL cross-repo base-lib symbol SQLite store "
                        "(built by scripts/crossrepo/build_global_symbol_index.py). "
                        "When set, the CODE half threads it into index_project so "
                        "unresolved base-lib callees are pulled as bounded depth-1 "
                        "deps tagged crosslib:<repo>. DEFAULT off (unchanged).")
    p.add_argument("--memory-limit-gb", type=float, default=10.0,
                   help="Per-stage fail-loud RSS limit passed to index/materialize/"
                        "commit processors (default 10.0).")
    p.add_argument("--code-memory-limit-gb", type=float, default=None,
                   help="CODE-stage RSS limit passed to index_project/materialize/"
                        "pack. Default: --memory-limit-gb.")
    p.add_argument("--commit-memory-limit-gb", type=float, default=None,
                   help="COMMITS-stage RSS limit passed to process_commits/"
                        "materialize/pack. Default: --memory-limit-gb. Use this "
                        "to raise --workers without over-reserving for code.")
    p.add_argument("--parse-workers", type=int, default=2,
                   help="Parse workers passed to index_project for the CODE half "
                        "(default 2; avoid multiplying --repo-workers by the old "
                        "index_project default of 8 clang workers).")
    p.add_argument("--code-index-timeout-s", type=int, default=0,
                   help="Optional fail-loud timeout for each CODE index_project "
                        "stage. 0 disables the timeout (default).")
    p.add_argument("--code-index-stall-timeout-s", type=int, default=0,
                   help="Optional fail-loud CODE index_project stall watchdog: "
                        "kill when the index log has no size/mtime progress for "
                        "this many seconds. 0 disables the watchdog (default).")
    p.add_argument("--background-code-recompress", action="store_true",
                   help="Defer code parquet zstd-max recompress to a background "
                        "pool so repo processing can continue after valid parquet "
                        "has been written. The pool is drained before exit.")
    p.add_argument("--code-recompress-workers", type=int, default=2,
                   help="Background code parquet recompress workers when "
                        "--background-code-recompress is set (default 2).")
    p.add_argument("--source-cache-dir", default=None,
                   help="Optional repo-level source cache for --streams code. "
                        "Complete cached repos are processed before opening the "
                        "source tarball; uncached repos are materialized into the "
                        "cache with a completion sentinel.")
    p.add_argument("--source-cache-only", action="store_true",
                   help="For --streams code, only process complete repos already "
                        "in --source-cache-dir; do not open/decompress the source "
                        "tarball.")
    p.add_argument("--source-cache-populate-only", action="store_true",
                   help="For --streams code, populate --source-cache-dir from "
                        "the source tarball and exit without indexing/tokenizing. "
                        "Follow with --source-cache-only for hot code runs.")
    p.add_argument("--source-dir-root", action="append", default=[],
                   help="For --streams code, process already-extracted repo dirs "
                        "directly without opening the source tarball. May be passed "
                        "multiple times; children named <repo>/_src or <repo> are "
                        "treated as repo dirs.")
    p.add_argument("--memory-budget-gb", type=float, default=None,
                   help="Global conveyor memory budget for heavy subprocesses. "
                        "Default is 55%% of physical RAM (or 48 GiB if RAM size "
                        "cannot be detected). Use 0 to disable the preflight.")
    p.add_argument("--allow-memory-oversubscription", action="store_true",
                   help="Allow heavy_slots * --memory-limit-gb to exceed "
                        "--memory-budget-gb. This is intentionally explicit.")
    p.add_argument("--progress-jsonl", default=str(DEFAULT_PROGRESS_JSONL),
                   help="Append unit-level progress events here for live "
                        "throughput monitoring. Use empty string to disable.")
    p.add_argument("--dedup-checkpoint-tokens", type=int,
                   default=DEFAULT_DEDUP_CHECKPOINT_TOKENS,
                   help="Run a SQLite WAL checkpoint on --dedup-db after each "
                        "N valid tokens produced by this conveyor run. Use 0 "
                        "to disable. Default 25,000,000.")
    p.add_argument("--dedup-checkpoint-mode",
                   choices=("PASSIVE", "FULL", "RESTART", "TRUNCATE"),
                   default="TRUNCATE",
                   help="SQLite wal_checkpoint mode for token milestones. "
                        "Default TRUNCATE caps WAL growth when no writer holds it.")
    p.add_argument("--dedup-checkpoint-busy-timeout-ms", type=int, default=5000,
                   help="Busy timeout for token-milestone dedup WAL checkpoints.")
    p.add_argument("--run-lock-dir", default=str(DEFAULT_RUN_LOCK_DIR),
                   help="Directory for fail-loud per-stream conveyor locks. "
                        f"Default {DEFAULT_RUN_LOCK_DIR}.")
    p.add_argument("--no-run-lock", action="store_true",
                   help="Disable per-stream conveyor process locks. Intended "
                        "only for controlled tests; normal runs should keep the "
                        "lock so duplicate stream writers fail before extraction.")
    p.add_argument("--reservation-file", default=str(DEFAULT_RESERVATION_FILE),
                   help="Cross-process active-unit reservation JSON file. Prevents "
                        "duplicate repo/range workers from processing the same "
                        f"output unit. Default {DEFAULT_RESERVATION_FILE}.")
    p.add_argument("--no-reservation-ledger", action="store_true",
                   help="Disable active-unit reservations. Intended only for "
                        "controlled tests.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    configure_runtime_paths_from_args(args)
    pr_store = (
        Path(args.pr_store).expanduser().resolve()
        if args.pr_store
        else None
    )
    repo_list = (
        Path(args.repo_list).expanduser().resolve()
        if args.repo_list
        else None
    )
    pr_completion_receipt_path = (
        Path(args.pr_completion_receipt).expanduser().resolve()
        if args.pr_completion_receipt
        else None
    )
    pr_completion_binding: dict[str, object] | None = None
    if args.streams in {"both", "commits"}:
        missing_pr_options = [
            option
            for option, value in (
                ("--pr-store", args.pr_store),
                ("--repo-list", args.repo_list),
                ("--pr-completion-receipt", args.pr_completion_receipt),
            )
            if not value
        ]
        if missing_pr_options:
            raise SystemExit(
                "commits/both requires explicit immutable PR inputs: "
                + ", ".join(missing_pr_options)
            )
        assert pr_store is not None
        assert repo_list is not None
        assert pr_completion_receipt_path is not None
        try:
            pr_completion_binding = load_pr_completion_binding(
                pr_completion_receipt_path,
                pr_store=pr_store,
                repo_list=repo_list,
            )
        except PRCompletionBindingError as exc:
            raise SystemExit(f"PR_COMPLETION_FAILED: {exc}") from exc
        _log(
            "PR-store: verified immutable completion receipt "
            f"{pr_completion_binding['receipt_sha256']} for {pr_store}; "
            f"repos={pr_completion_binding['expected_repo_count']} "
            f"prs={pr_completion_binding['stored_pr_count']} "
            f"(repo_list={repo_list})"
        )
    pr_scan_id = (
        str(pr_completion_binding["scan_id"])
        if pr_completion_binding is not None
        else None
    )
    try:
        revision_guard = CodeRevisionGuard.for_production(
            args.expected_code_revision,
            repo_root=ROOT,
            stop_event=STOP_EVENT,
        )
    except CodeRevisionError as exc:
        raise SystemExit(str(exc)) from exc
    lengths_code = tuple(sorted(
        int(x) for x in args.target_lengths_code.split(",") if x.strip()))
    lengths_commits = tuple(sorted(
        int(x) for x in args.target_lengths_commits.split(",") if x.strip()))
    if not lengths_code:
        raise SystemExit("--target-lengths-code produced no lengths")
    if not lengths_commits:
        raise SystemExit("--target-lengths-commits produced no lengths")
    workers = max(1, int(args.workers or 1))
    repo_workers = max(1, int(args.repo_workers or 1))
    max_active_repos = int(args.max_active_repos or repo_workers)
    max_active_repos = max(1, max_active_repos)
    if max_active_repos < repo_workers:
        raise SystemExit("--max-active-repos must be >= --repo-workers")
    parse_workers = max(1, int(args.parse_workers or 1))
    args.parse_workers = parse_workers
    code_index_stall_timeout_s = int(args.code_index_stall_timeout_s or 0)
    source_cache_dir = Path(args.source_cache_dir) if args.source_cache_dir else None
    source_dir_roots = [Path(p) for p in args.source_dir_root]
    source_quarantine_manifest = (
        Path(args.source_quarantine_manifest).expanduser().resolve()
        if args.source_quarantine_manifest
        else None
    )
    if (
        source_quarantine_manifest is not None
        and not source_quarantine_manifest.is_file()
    ):
        raise SystemExit(
            "--source-quarantine-manifest not found: "
            f"{source_quarantine_manifest}"
        )
    if args.source_cache_only and source_cache_dir is None:
        raise SystemExit("--source-cache-only requires --source-cache-dir")
    if args.source_cache_populate_only and source_cache_dir is None:
        raise SystemExit("--source-cache-populate-only requires --source-cache-dir")
    if args.source_cache_populate_only and args.source_cache_only:
        raise SystemExit("--source-cache-populate-only cannot be combined with --source-cache-only")
    if args.source_cache_populate_only and args.streams != "code":
        raise SystemExit("--source-cache-populate-only is only valid with --streams code")
    if args.source_cache_only and args.streams != "code":
        raise SystemExit("--source-cache-only is only valid with --streams code")
    if source_cache_dir is not None and args.streams != "code":
        raise SystemExit("--source-cache-dir is currently only supported with --streams code")
    if source_dir_roots and args.streams != "code":
        raise SystemExit("--source-dir-root is currently only supported with --streams code")
    if source_dir_roots and (source_cache_dir is not None or args.source_cache_only):
        raise SystemExit("--source-dir-root cannot be combined with source cache flags")
    range_submit_window = (
        max(1, workers * DEFAULT_RANGE_SUBMIT_WINDOW_MULTIPLIER)
        if int(args.range_submit_window or 0) <= 0
        else max(1, int(args.range_submit_window))
    )
    dedup_promote_batch_size = max(1, int(args.dedup_promote_batch_size or 1))
    code_memory_limit_gb = (
        float(args.memory_limit_gb)
        if args.code_memory_limit_gb is None
        else float(args.code_memory_limit_gb)
    )
    commit_memory_limit_gb = (
        float(args.memory_limit_gb)
        if args.commit_memory_limit_gb is None
        else float(args.commit_memory_limit_gb)
    )
    memory_budget_gb = (
        default_memory_budget_gb()
        if args.memory_budget_gb is None
        else float(args.memory_budget_gb)
    )
    memory_plan = validate_memory_plan(
        streams=args.streams,
        workers=workers,
        repo_workers=repo_workers,
        memory_limit_gb=args.memory_limit_gb,
        code_memory_limit_gb=code_memory_limit_gb,
        commit_memory_limit_gb=commit_memory_limit_gb,
        memory_budget_gb=memory_budget_gb,
        allow_oversubscription=args.allow_memory_oversubscription,
    )

    # SIGNAL HANDLER (requirement 1): cooperative, fail-loud shutdown. The first
    # SIGINT/SIGTERM sets STOP_EVENT -> the submission loops stop staging new
    # repos and run_commits_half stops/cancels new ranges, while in-flight
    # subprocess tasks drain and only COMPLETED units are marked done (the
    # manifest is already persisted atomically per mark, so nothing partial can be
    # reported as done). In-progress repo temp + extract caches are RETAINED. A
    # SECOND signal forces an immediate os._exit (skips all cleanup, so temp is
    # preserved). Installed in the main thread before any streaming begins.
    def _on_signal(signum, _frame):
        name = signal.Signals(signum).name
        if STOP_EVENT.is_set():
            _log(f"Signal {name} again: FORCING immediate exit (130). Completed "
                 "units already persisted; in-progress temp may be retained.")
            os._exit(130)
        STOP_EVENT.set()
        _log(f"Signal {name} received: CHECKPOINTING -- no new repos/ranges "
             "submitted; draining in-flight tasks; manifest is atomic. Send the "
             "signal again to force-exit.")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    run_locks: list[RunLock] = []
    if not args.no_run_lock:
        lock_dir = Path(args.run_lock_dir)
        try:
            for name in stream_lock_names(args.streams):
                lock = RunLock(lock_dir / f"{name}.lock")
                lock.acquire()
                run_locks.append(lock)
            _log(
                "Run-lock: acquired "
                + ", ".join(str(lock.path) for lock in run_locks)
            )
        except Exception:
            for lock in run_locks:
                lock.close()
            raise

    # FAIL LOUD up front (RULE #1): every required stage binary must exist.
    required_paths = [VENV_PYTHON, sr.TOKENIZER_PATH, sr.MATERIALIZER, sr.PACKER]
    if args.streams in {"both", "code"}:
        required_paths.append(sr.INDEX_PROJECT)
    if args.streams in {"both", "commits"}:
        required_paths.extend([sr.PROCESS_COMMITS, EXTRACT_GIT])
    for path in required_paths:
        if not Path(path).exists():
            raise SystemExit(f"required path missing: {path}")

    # Pre-create output trees for BOTH streams.
    CONVEYOR_ROOT.mkdir(parents=True, exist_ok=True)
    EXTRACT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    _log(
        "Extract cache: "
        f"mode={EXTRACT_CACHE_MODE} root={EXTRACT_CACHE_ROOT.resolve()}"
    )
    sr.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for L in lengths_code:
        (sr.OUTPUT_ROOT / str(L)).mkdir(parents=True, exist_ok=True)
    src.COMMIT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for L in lengths_commits:
        (src.COMMIT_OUTPUT_ROOT / str(L)).mkdir(parents=True, exist_ok=True)

    # SHARED dedup db for BOTH streams. FAIL LOUD: open once now so a bad path /
    # missing datasketch crashes before any work (RULE #1).
    dedup_db = Path(args.dedup_db) if args.dedup_db else None
    dedup_near = not args.no_near_dedup
    if dedup_db is not None:
        sys.path.insert(0, str(ROOT / "tools" / "clang_indexer"))
        from dedup_store import DedupStore  # noqa: E402
        dedup_db.parent.mkdir(parents=True, exist_ok=True)
        # Path/schema validation only. Do not rebuild the persisted MinHash/LSH
        # index in the driver parent; real worker stages open near-dedup only
        # when requested. On full corpora the LSH reload dominates startup.
        DedupStore(str(dedup_db), near=False, commit_every=1000).close()
        _log(f"Dedup: SHARED global store at {dedup_db} threaded into BOTH "
             f"code + commit stages (exact{'+near' if dedup_near else ''}, "
             f"tokenized hash)")

    # PR-discussion live lookup (fail-loud on bad paths up front).
    if pr_store is not None and not pr_store.exists():
        raise SystemExit(f"--pr-store does not exist: {pr_store}")
    if repo_list is not None and not repo_list.exists():
        raise SystemExit(f"--repo-list does not exist: {repo_list}")

    # Optional cross-repo base-lib symbol index. FAIL LOUD if given but missing.
    global_symbol_index = (
        Path(args.global_symbol_index) if args.global_symbol_index else None
    )
    if global_symbol_index is not None:
        if not global_symbol_index.exists():
            raise SystemExit(f"--global-symbol-index not found: {global_symbol_index}")
        if args.streams in {"both", "code"}:
            _log(f"Cross-lib: GLOBAL base-lib symbol index at {global_symbol_index} "
                 f"threaded into CODE half (bounded depth-1 pulls, crosslib:<repo>).")

    # Cross-process-safe manifest: a concurrent --streams code run and
    # --streams commits run (each holding only its own per-stream RunLock) share
    # CONVEYOR_MANIFEST; ConcurrentManifest merges + flocks every write so they
    # cannot clobber each other's resume/accounting keys (H4 fix).
    manifest = ConcurrentManifest.load(CONVEYOR_MANIFEST)
    try:
        manifest.bind_code_revision(revision_guard.receipt)
        if pr_completion_binding is not None:
            manifest.bind_pr_completion(pr_completion_binding)
        child_guard_path = install_code_revision_child_guard(
            revision_guard,
            CONVEYOR_ROOT,
        )
    except (CodeRevisionError, PRCompletionBindingError) as exc:
        raise SystemExit(str(exc)) from exc
    manifest_lock = threading.Lock()
    resume = not args.no_resume
    progress = ProgressWriter(Path(args.progress_jsonl) if args.progress_jsonl else None)
    code_recompressor = (
        BackgroundRecompressor(
            args.code_recompress_workers,
            revision_guard=revision_guard,
        )
        if args.background_code_recompress and args.streams in {"both", "code"}
        else None
    )
    reservations = None
    if not args.no_reservation_ledger:
        reservations = UnitReservationLedger(Path(args.reservation_file))
        stale = reservations.cleanup_stale()
        if stale:
            _log(f"Reservations: cleaned {stale} stale active unit claim(s)")
    checkpoint = DedupCheckpointController(
        dedup_db=dedup_db,
        interval_tokens=args.dedup_checkpoint_tokens,
        mode=args.dedup_checkpoint_mode,
        busy_timeout_ms=args.dedup_checkpoint_busy_timeout_ms,
        progress=progress,
    )
    progress.emit(
        "run_started",
        code_revision=revision_guard.receipt,
        pr_completion=pr_completion_binding,
        code_revision_child_guard=str(child_guard_path),
        streams=args.streams,
        workers=workers,
        repo_workers=repo_workers,
        max_active_repos=max_active_repos,
        range_size=args.range_size,
        range_target_bytes=args.range_target_bytes,
        range_submit_window=range_submit_window,
        dedup_promote_batch_size=dedup_promote_batch_size,
        target_lengths_code=list(lengths_code),
        target_lengths_commits=list(lengths_commits),
        manifest=str(CONVEYOR_MANIFEST),
        extract_cache=extract_cache_config_receipt(),
        retain_partial_work=args.retain_partial_work,
        keep_temp=args.keep_temp,
        min_free_disk_gb=args.min_free_disk_gb,
        dedup_checkpoint_tokens=args.dedup_checkpoint_tokens,
        dedup_checkpoint_mode=args.dedup_checkpoint_mode,
        parse_workers=parse_workers,
        code_index_timeout_s=args.code_index_timeout_s,
        code_index_stall_timeout_s=code_index_stall_timeout_s,
        code_memory_limit_gb=code_memory_limit_gb,
        commit_memory_limit_gb=commit_memory_limit_gb,
        background_code_recompress=code_recompressor is not None,
        code_recompress_workers=args.code_recompress_workers if code_recompressor else 0,
        source_cache_dir=str(source_cache_dir) if source_cache_dir is not None else None,
        source_cache_only=args.source_cache_only,
        source_cache_populate_only=args.source_cache_populate_only,
        source_dir_roots=[str(p) for p in source_dir_roots],
        reservation_file=str(args.reservation_file)
        if reservations is not None else None,
        memory_plan=memory_plan,
    )
    _log(
        "Memory plan: "
        f"heavy_slots={memory_plan['heavy_slots']} "
        f"code_slots={memory_plan['code_heavy_slots']} "
        f"commit_slots={memory_plan['commit_heavy_slots']} "
        f"code_limit={memory_plan['code_memory_limit_gb']:.2f}GiB "
        f"commit_limit={memory_plan['commit_memory_limit_gb']:.2f}GiB "
        f"reserved={memory_plan['reserved_gb']:.2f}GiB "
        f"budget={memory_plan['memory_budget_gb']:.2f}GiB "
        f"parse_workers={parse_workers} "
        f"range_submit_window={range_submit_window} "
        f"dedup_promote_batch_size={dedup_promote_batch_size}"
    )

    # work_parent is always the configured parent: it both hosts the randomized
    # per-run work_root (when --work-dir is unset) AND is scanned by the extract
    # checkpoint to ADOPT a prior run's orphaned <repo>_commits.jsonl on resume.
    work_parent = Path(args.work_parent_dir)
    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        own_work_root = False
    else:
        work_parent.mkdir(parents=True, exist_ok=True)
        ensure_min_free_disk(
            work_parent,
            args.min_free_disk_gb,
            context="creating conveyor work root",
        )
        os.environ.setdefault("TMPDIR", str(work_parent))
        os.environ.setdefault("TMP", str(work_parent))
        os.environ.setdefault("TEMP", str(work_parent))
        work_root = Path(tempfile.mkdtemp(prefix="streaming_conveyor_", dir=str(work_parent)))
        own_work_root = True
    ensure_min_free_disk(
        work_parent,
        args.min_free_disk_gb,
        context="starting conveyor",
    )
    progress.emit("work_root_ready", work_root=str(work_root), own_work_root=own_work_root)

    cumulative = {"valid": 0}
    processed_repos = 0
    code_done = 0
    ranges_done = 0
    ranges_failed = 0
    submitted_repo_names: set[str] = set()

    only_repos = set(args.only_repo) if args.only_repo else None

    def should_process(repo: str) -> bool:
        if args.streams == "code" and not sr.is_code_worktree_repo(repo):
            return False
        # Restrict to --only-repo when given (others drained without extraction).
        # Otherwise skip extraction when the manifest itself proves every stream
        # for this repo is complete; partial/ambiguous repos still stage and use
        # the existing per-half/per-range resume below.
        should_stage = should_stage_repo_from_manifest(
            repo,
            streams=args.streams,
            resume=resume,
            manifest=manifest,
            range_size=args.range_size,
            only_repos=only_repos,
            manifest_lock=manifest_lock,
        )
        if should_stage:
            ensure_min_free_disk(
                work_parent,
                args.min_free_disk_gb,
                context=f"staging repo {repo}",
            )
        return should_stage

    if args.source_cache_populate_only:
        try:
            revision_guard.verify("source cache population stage")
            report = populate_code_source_cache(
                work_root,
                source_cache_dir,
                should_process,
                progress,
                max_repos=args.max_repos,
            )
            summary = {
                "repos_this_run": 0,
                "code_halves_this_run": 0,
                "commit_ranges_this_run": 0,
                "commit_ranges_failed_this_run": 0,
                "workers": workers,
                "repo_workers": repo_workers,
                "max_active_repos": max_active_repos,
                "parse_workers": parse_workers,
                "memory_plan": memory_plan,
                "streams": args.streams,
                "source_cache_populate_only": True,
                "source_cache_report": report,
                "manifest": str(CONVEYOR_MANIFEST),
                "code_revision": revision_guard.receipt,
                "pr_completion": pr_completion_binding,
                "extract_cache": extract_cache_config_receipt(),
                "interrupted": STOP_EVENT.is_set(),
            }
            progress.emit("run_finished", **summary)
            print(json.dumps(summary, indent=2))
            return 130 if STOP_EVENT.is_set() else 0
        finally:
            if code_recompressor is not None:
                code_recompressor.shutdown()
            if should_remove_owned_work_root(
                own_work_root=own_work_root,
                keep_temp=args.keep_temp,
                retain_partial_work=args.retain_partial_work,
            ):
                remove_tree(work_root, reason="source-cache populate work_root")
            for lock in reversed(run_locks):
                lock.close()

    range_pool = ThreadPoolExecutor(max_workers=workers)
    repo_pool = ThreadPoolExecutor(max_workers=repo_workers) if repo_workers > 1 else None

    def claim_repo_once(repo: str) -> None:
        if repo in submitted_repo_names:
            raise RepoFailure(
                repo,
                "duplicate_repo",
                f"repo {repo!r} was yielded/submitted twice in one conveyor run",
            )
        submitted_repo_names.add(repo)

    def mark_no_git_repo(repo: str) -> None:
        info = {
            "source": "commits",
            "no_git": True,
            "reason": "missing .git metadata",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with manifest_lock:
            manifest.mark_done(no_git_key(repo), info)
        progress.emit("repo_no_git", repo=repo, unit=no_git_key(repo), **info)

    try:
        gen = (
            sr.stream_repo_dirs(source_dir_roots, should_process)
            if source_dir_roots
            else sr.stream_repo_subtrees(
                work_root,
                should_process,
                source_cache_dir=source_cache_dir,
                source_cache_only=args.source_cache_only,
            )
            if args.streams == "code"
            else stream_repo_subtrees_with_git(
                work_root,
                should_process,
                on_no_git=mark_no_git_repo,
            )
        )
        gen = guard_source_submissions(gen, revision_guard)

        if repo_pool is None:
            for repo, repo_dir in gen:
                # Signal-driven stop: do not stage a NEW repo. The repo currently
                # in process_one_repo (if any) already drained its own ranges.
                if STOP_EVENT.is_set():
                    if hasattr(gen, "close"):
                        gen.close()
                    break
                claim_repo_once(repo)
                revision_guard.verify(f"submit repo worker for {repo}")
                res = process_one_repo(
                    repo, repo_dir, lengths_code, lengths_commits, args.range_size,
                    args.range_target_bytes,
                    work_root, work_parent, range_pool, manifest, manifest_lock,
                    resume, cumulative,
                    args.keep_temp, dedup_db, dedup_near, pr_store, repo_list,
                    args.streams, global_symbol_index, args.memory_limit_gb,
                    args.parse_workers,
                    progress, checkpoint,
                    code_memory_limit_gb=code_memory_limit_gb,
                    commit_memory_limit_gb=commit_memory_limit_gb,
                    code_index_timeout_s=args.code_index_timeout_s,
                    code_index_stall_timeout_s=code_index_stall_timeout_s,
                    code_recompressor=code_recompressor,
                    reservations=reservations,
                    range_submit_window=range_submit_window,
                    analysis_cache_entries=args.analysis_cache_entries,
                    dedup_promote_batch_size=dedup_promote_batch_size,
                    retain_partial_work=args.retain_partial_work,
                    revision_guard=revision_guard,
                    pr_scan_id=pr_scan_id,
                    source_quarantine_manifest=source_quarantine_manifest,
                )
                processed_repos += 1
                if isinstance(res.get("code"), dict):
                    code_done += 1
                ranges_done += res.get("commits_done", 0)
                ranges_failed += res.get("commits_failed", 0)

                stop = STOP_EVENT.is_set()
                if args.max_repos is not None and processed_repos >= args.max_repos:
                    stop = True
                if args.token_budget is not None and cumulative["valid"] >= args.token_budget:
                    _log(f"Token budget {args.token_budget} reached.")
                    stop = True
                if stop:
                    if hasattr(gen, "close"):
                        gen.close()
                    break
        else:
            inflight: dict = {}
            submitted_repos = 0
            stop_submitting = False

            def account_repo_result(fut: Future, repo: str) -> None:
                nonlocal processed_repos, code_done, ranges_done, ranges_failed
                pr, cd, rd, rf = handle_repo_future_result(
                    fut,
                    repo,
                    manifest,
                    manifest_lock,
                )
                processed_repos += pr
                code_done += cd
                ranges_done += rd
                ranges_failed += rf

            def drain_one_or_more(block: bool = True) -> None:
                if not inflight:
                    return
                timeout = None if block else 0
                done, _pending = wait(
                    inflight.keys(),
                    timeout=timeout,
                    return_when=FIRST_COMPLETED,
                )
                for fut in done:
                    repo = inflight.pop(fut)
                    account_repo_result(fut, repo)

            gen = stream_source_with_repo_future_drain(
                gen,
                inflight,
                account_repo_result,
            )
            for repo, repo_dir in gen:
                # Signal-driven stop: stop submitting NEW repos; already-inflight
                # repos drain (and self-cancel their queued ranges) in the loop
                # below. Their un-finished units stay un-marked -> resume re-runs.
                if STOP_EVENT.is_set():
                    if hasattr(gen, "close"):
                        gen.close()
                    break
                claim_repo_once(repo)
                while len(inflight) >= max_active_repos:
                    drain_one_or_more(block=True)
                    # Worker threads mutate cumulative['valid'] under manifest_lock
                    # (see process_one_repo / run_commits_half); snapshot under the
                    # same lock before the budget comparison so this drain-path read
                    # is not a concurrent unlocked read while writers are in flight.
                    with manifest_lock:
                        cumulative_valid_snapshot = cumulative["valid"]
                    if args.token_budget is not None and cumulative_valid_snapshot >= args.token_budget:
                        _log(f"Token budget {args.token_budget} reached.")
                        stop_submitting = True
                        break
                    if STOP_EVENT.is_set():
                        stop_submitting = True
                        break
                if stop_submitting or STOP_EVENT.is_set():
                    if hasattr(gen, "close"):
                        gen.close()
                    break
                if args.max_repos is not None and submitted_repos >= args.max_repos:
                    if hasattr(gen, "close"):
                        gen.close()
                    break
                revision_guard.verify(f"submit repo worker for {repo}")
                fut = repo_pool.submit(
                    process_one_repo,
                    repo, repo_dir, lengths_code, lengths_commits, args.range_size,
                    args.range_target_bytes,
                    work_root, work_parent, range_pool, manifest, manifest_lock,
                    resume, cumulative,
                    args.keep_temp, dedup_db, dedup_near, pr_store, repo_list,
                    args.streams, global_symbol_index, args.memory_limit_gb,
                    args.parse_workers,
                    progress, checkpoint,
                    code_memory_limit_gb=code_memory_limit_gb,
                    commit_memory_limit_gb=commit_memory_limit_gb,
                    code_index_timeout_s=args.code_index_timeout_s,
                    code_index_stall_timeout_s=code_index_stall_timeout_s,
                    code_recompressor=code_recompressor,
                    reservations=reservations,
                    range_submit_window=range_submit_window,
                    analysis_cache_entries=args.analysis_cache_entries,
                    dedup_promote_batch_size=dedup_promote_batch_size,
                    retain_partial_work=args.retain_partial_work,
                    revision_guard=revision_guard,
                    pr_scan_id=pr_scan_id,
                    source_quarantine_manifest=source_quarantine_manifest,
                )
                inflight[fut] = repo
                submitted_repos += 1
                drain_one_or_more(block=False)

            while inflight:
                drain_one_or_more(block=True)
    except tarfile.ReadError as exc:
        _handle_source_stream_read_error(exc, interrupted=STOP_EVENT.is_set())
    finally:
        recompress_error: Exception | None = None
        if repo_pool is not None:
            repo_pool.shutdown(wait=True)
        range_pool.shutdown(wait=True)
        if code_recompressor is not None:
            try:
                code_recompressor.shutdown()
            except SymbolIdentityError:
                raise
            except Exception as exc:
                recompress_error = exc
        interrupted = STOP_EVENT.is_set()
        # Reclaim the randomized work_root whenever production cleanup is active.
        # --retain-partial-work keeps the older zero-rework signal/debug mode.
        if should_remove_owned_work_root(
            own_work_root=own_work_root,
            keep_temp=args.keep_temp,
            retain_partial_work=args.retain_partial_work,
        ):
            remove_tree(
                work_root,
                reason="interrupted run work_root" if interrupted else "clean run work_root",
            )
        for lock in reversed(run_locks):
            lock.close()
        if recompress_error is not None:
            raise recompress_error

    pr_completion_reverified_at_finish = False
    if pr_completion_binding is not None and not interrupted:
        assert pr_store is not None
        assert repo_list is not None
        assert pr_completion_receipt_path is not None
        try:
            revalidate_pr_completion_binding(
                pr_completion_binding,
                pr_completion_receipt_path,
                pr_store=pr_store,
                repo_list=repo_list,
            )
        except PRCompletionBindingError as exc:
            raise SystemExit(f"PR_COMPLETION_DRIFT: {exc}") from exc
        pr_completion_reverified_at_finish = True

    # ----- cumulative per-length report from the manifest (both streams) -----
    def _empty_totals(lengths):
        return {str(L): {"rows": 0, "valid_tokens": 0, "pad_tokens": 0,
                         "capacity_tokens": 0} for L in lengths}

    code_totals = _empty_totals(lengths_code)
    commit_totals = _empty_totals(lengths_commits)
    for key, info in manifest.done.items():
        is_code = key.endswith("::code")
        totals = code_totals if is_code else commit_totals
        for tl_s, st in info.get("lengths", {}).items():
            if tl_s not in totals:
                continue
            agg = totals[tl_s]
            agg["rows"] += st["rows"]
            agg["valid_tokens"] += st["valid_tokens"]
            agg["pad_tokens"] += st["pad_tokens"]
            agg["capacity_tokens"] += st["capacity_tokens"]

    summary = {
        "repos_this_run": processed_repos,
        "code_halves_this_run": code_done,
        "commit_ranges_this_run": ranges_done,
        "commit_ranges_failed_this_run": ranges_failed,
        "workers": workers,
        "repo_workers": repo_workers,
        "max_active_repos": max_active_repos,
        "parse_workers": parse_workers,
        "memory_plan": memory_plan,
        "streams": args.streams,
        "range_size": args.range_size,
        "target_lengths_code": list(lengths_code),
        "target_lengths_commits": list(lengths_commits),
        "cumulative_valid_tokens_this_run": cumulative["valid"],
        "total_done_units": len(manifest.done),
        "total_failed_units": len(manifest.failed),
        "code_per_length_totals": code_totals,
        "commit_per_length_totals": commit_totals,
        "dedup_db": str(dedup_db) if dedup_db else None,
        "pr_store": str(pr_store) if pr_store else None,
        "manifest": str(CONVEYOR_MANIFEST),
        "code_revision": revision_guard.receipt,
        "pr_completion": pr_completion_binding,
        "pr_completion_reverified_at_finish": (
            pr_completion_reverified_at_finish
        ),
        "extract_cache": extract_cache_config_receipt(),
        "extract_cache_metrics": progress.extract_cache_metrics(),
        "interrupted": interrupted,
    }
    progress.emit("run_finished", **summary)
    print(json.dumps(summary, indent=2))
    if interrupted:
        _log("CHECKPOINTED on signal: every completed unit is recorded in the "
             f"manifest ({CONVEYOR_MANIFEST}); in-progress repo temp + the "
             "persistent extract cache were retained. RESUME WITH THE SAME ARGS "
             "to continue exactly where this left off (zero work lost).")
        return 130
    return 0 if not manifest.failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
