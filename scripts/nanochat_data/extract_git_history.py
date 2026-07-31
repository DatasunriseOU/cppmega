"""Extract git commit history as raw JSONL for clang commit enrichment.

Extracts per-file commit diffs from git repos and outputs JSONL with:
  {old_content, new_content, diff, subject, body, filepath, repo}

The clang commit processor consumes these records, parses old/new source with
libclang, and produces enriched training documents.

Usage:
    # Extract raw commit data
    python3 scripts/nanochat_data/extract_git_history.py \
        --repo ~/data/cpp_raw/opencv \
        --output /mnt/nvme/nanochat_data/opencv_commits.jsonl \
        --max_commits 50000

    # Explicitly quarantine at most one reproducibly bad commit/path unit
    python3 scripts/nanochat_data/extract_git_history.py \
        --repo ~/data/cpp_raw/php-src \
        --output /mnt/nvme/nanochat_data/php-src_commits.jsonl \
        --bad-unit-policy quarantine --max-bad-units 1

    # Process with clang wrapper
    python3 tools/clang_indexer/process_commits.py \
        --inputs opencv_commits.jsonl \
        --output opencv_training.jsonl --max-tokens 4096 --format both
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import BinaryIO, Optional, TypedDict

# "Merge pull request #N from owner/source-branch" — GitHub merge-commit subject.
_MERGE_PR_RE = re.compile(
    r'Merge pull request #(\d+) from (\S+)',
)
# Trailing "(#N)" PR marker in a squash/merge subject.
_SUBJECT_PR_RE = re.compile(r'\(#(\d+)\)\s*$')
# Body trailers referencing a PR/issue.
_BODY_PR_RE = re.compile(
    r'(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)[ :]+#(\d+)',
    re.IGNORECASE,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.nanochat_data.atomic_publish import atomic_output_file  # noqa: E402
from scripts.nanochat_data.memory_guard import (  # noqa: E402
    check_memory_limit,
    start_memory_guard,
)
from cppmega.data.commit_scope import classify_primary_commit_path  # noqa: E402
from cppmega.symbol_identity import (  # noqa: E402
    SymbolIdentityError,
    require_project_identity,
)


CHECKPOINT_SCHEMA_VERSION = 1
EXTRACTION_CONTRACT_VERSION = 2
DEFAULT_CHECKPOINT_COMMITS = 250
CHECKPOINT_SUFFIX = ".extract-checkpoint"


class CheckpointCorruptionError(RuntimeError):
    """A committed extraction checkpoint no longer matches its durable bytes."""


class UnitExtractionError(RuntimeError):
    """A specific commit/path operation failed and is eligible for policy handling."""

    def __init__(
        self,
        *,
        repo_path: str,
        commit_hash: str,
        filepath: Optional[str],
        operation: str,
        error_type: str,
        detail: str,
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.commit_hash = commit_hash
        self.filepath = filepath
        self.operation = operation
        self.error_type = error_type
        self.detail = detail
        location = filepath if filepath is not None else "<commit>"
        super().__init__(
            f"{operation} failed for {self.repo_path}@{commit_hash}:{location}: "
            f"{error_type}: {detail}"
        )

class _ExtractionStats(TypedDict):
    repo: str
    commits_checked: int
    records_written: int


# Files/paths to skip
SKIP_PATTERNS = {
    "third_party/",
    "3rdparty/",
    "vendor/",
    "external/",
    "deps/",
    "generated/",
    "auto_generated/",
    "cmake-build",
    ".pb.h",
    ".pb.cc",
    "_generated.h",
    ".gen.cc",
    ".gen.h",
}

MAX_DIFF_CHARS = 50000
MIN_DIFF_CHARS = 50


class OutputFileLock:
    """Exclusive lock for one output JSONL writer.

    The conveyor may run many repos in parallel, but a single JSONL path must
    have exactly one writer. Lock before opening the output file so a duplicate
    invocation cannot truncate or race the active writer.
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.lock_path = Path(str(self.output_path) + ".lock")
        self._fh = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.seek(0)
            holder = self._fh.read().strip()
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"output JSONL lock is already held: {self.lock_path}\n"
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
                    "output": str(self.output_path),
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


def stable_repo_id(repo_name: str) -> str:
    return hashlib.sha1(repo_name.encode("utf-8")).hexdigest()[:16]


def stable_filepath_id(repo_name: str, filepath: str) -> str:
    key = f"{repo_name}\0{filepath}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def is_primary_commit_file(path: str) -> bool:
    return classify_primary_commit_path(path) is not None


def should_skip_path(path: str) -> bool:
    path_lower = path.lower()
    return any(p in path_lower for p in SKIP_PATTERNS)


def run_git(repo_path: str, args: list[str], timeout: int = 60) -> Optional[str]:
    cmd = ["git", "-C", repo_path] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, errors="replace"
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


def run_git_required(
    repo_path: str,
    args: list[str],
    *,
    timeout: int = 60,
    operation: str,
) -> str:
    """Run a repository-level git command and fail with command context."""
    cmd = ["git", "-C", repo_path] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{operation} timed out after {timeout}s in {os.path.abspath(repo_path)}: "
            f"{cmd!r}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"{operation} could not execute in {os.path.abspath(repo_path)}: "
            f"{cmd!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "<no git output>").strip()
        raise RuntimeError(
            f"{operation} failed in {os.path.abspath(repo_path)} with exit "
            f"{result.returncode}: {cmd!r}: {detail[:4000]}"
        )
    return result.stdout


def run_git_unit(
    repo_path: str,
    args: list[str],
    *,
    commit_hash: str,
    filepath: Optional[str],
    operation: str,
    timeout: int = 60,
) -> str:
    """Run git for one commit/path and preserve exact failure provenance."""
    cmd = ["git", "-C", repo_path] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise UnitExtractionError(
            repo_path=repo_path,
            commit_hash=commit_hash,
            filepath=filepath,
            operation=operation,
            error_type="TimeoutExpired",
            detail=f"timed out after {timeout}s; command={cmd!r}",
        ) from exc
    except OSError as exc:
        raise UnitExtractionError(
            repo_path=repo_path,
            commit_hash=commit_hash,
            filepath=filepath,
            operation=operation,
            error_type=type(exc).__name__,
            detail=f"command={cmd!r}; error={exc}",
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "<no git output>").strip()
        raise UnitExtractionError(
            repo_path=repo_path,
            commit_hash=commit_hash,
            filepath=filepath,
            operation=operation,
            error_type="GitCommandError",
            detail=(
                f"exit={result.returncode}; command={cmd!r}; "
                f"output={detail[:4000]}"
            ),
        )
    return result.stdout


# GitHub remote URL -> owner/repo. Matches https://github.com/owner/repo(.git)
# and git@github.com:owner/repo(.git) forms.
_GH_REMOTE_RE = re.compile(
    r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


def resolve_repo_url(repo_path: str) -> Optional[str]:
    """Return the clone's raw ``remote.origin.url`` (or None when unset)."""
    url = run_git(repo_path, ["config", "--get", "remote.origin.url"])
    if not url:
        return None
    url = url.strip()
    return url or None


def resolve_owner_repo(repo_path: str) -> Optional[str]:
    """Resolve canonical ``owner/repo`` from the clone's git remote.origin.url.

    The authoritative owner/repo is NOT in git history — it lives only in the
    clone's remote. Returns ``owner/repo`` or None when the remote is missing /
    not a GitHub URL (caller decides whether to fall back to the bare name).
    """
    url = resolve_repo_url(repo_path)
    if not url:
        return None
    m = _GH_REMOTE_RE.search(url)
    if not m:
        return None
    return f"{m.group('owner')}/{m.group('repo')}"


def get_commit_list(repo_path: str, max_commits: int = 0) -> list[str]:
    args = ["log", "--format=%H", "--no-merges", "--diff-filter=M"]
    if max_commits > 0:
        args.extend(["-n", str(max_commits)])
    output = run_git_required(
        repo_path,
        args,
        timeout=120,
        operation="list non-merge modified commits",
    )
    if not output.strip():
        return []
    return output.strip().splitlines()


def get_commit_info(repo_path: str, commit_hash: str) -> Optional[dict]:
    # NUL-delimit fields so multi-line trailers cannot collide with the
    # newline-delimited header fields. Order: hash, subject, body, parents,
    # author-date, commit-date, trailers(unfolded).
    fmt = (
        "%H%x00%s%x00%b%x00%P%x00%aI%x00%cI%x00"
        "%(trailers:only=true,unfold=true)"
    )
    output = run_git_unit(
        repo_path,
        ["show", "-s", f"--format={fmt}", commit_hash],
        commit_hash=commit_hash,
        filepath=None,
        operation="commit_info",
    )
    fields = output.split("\x00")
    if len(fields) < 7:
        raise UnitExtractionError(
            repo_path=repo_path,
            commit_hash=commit_hash,
            filepath=None,
            operation="commit_info_parse",
            error_type="MalformedGitOutput",
            detail=f"expected 7 NUL-delimited fields, got {len(fields)}",
        )
    commit_hash_out = fields[0]
    subject = fields[1]
    body = fields[2].strip()
    parent_hashes = [item for item in fields[3].strip().split() if item]
    author_timestamp = fields[4].strip() or None
    commit_timestamp = fields[5].strip() or None
    trailers = fields[6].strip()
    return {
        "hash": commit_hash_out,
        "subject": subject,
        "body": body,
        "trailers": trailers,
        "parent_hashes": parent_hashes,
        "parent_count": len(parent_hashes),
        "is_merge_commit": len(parent_hashes) > 1,
        "author_timestamp": author_timestamp,
        "commit_timestamp": commit_timestamp,
        "timestamp": commit_timestamp or author_timestamp,
    }


def parse_pr_number_from_text(subject: str, body: str) -> Optional[int]:
    """Parse a PR/issue number from the subject ``(#N)`` trailer or body trailers."""
    m = _SUBJECT_PR_RE.search(subject or "")
    if m:
        return int(m.group(1))
    m = _BODY_PR_RE.search(body or "")
    if m:
        return int(m.group(1))
    return None


def build_merge_pr_map(repo_path: str) -> dict[int, dict[str, str]]:
    """Mine merge commits for {pr_number -> {pr_title, source_branch}}.

    A SEPARATE ``git log --merges`` pass over 'Merge pull request #N from ...'
    subjects. Merge commits are NOT added as training records; this map lets a
    non-merge record carry the real PR title/source branch when its parsed
    pr_number matches.
    """
    out: dict[int, dict[str, str]] = {}
    fmt = "%x00%s%x00%b%x00"
    output = run_git_required(
        repo_path,
        ["log", "--merges", f"--format={fmt}"],
        timeout=180,
        operation="mine merge PR metadata",
    )
    if not output.strip():
        return out
    # Records are NUL-delimited triples (leading empty, subject, body) per commit.
    fields = output.split("\x00")
    # Walk in (subject, body) pairs: fields[1::3]=subject, fields[2::3]=body.
    for idx in range(1, len(fields) - 1, 3):
        subject = fields[idx]
        body = fields[idx + 1] if idx + 1 < len(fields) else ""
        m = _MERGE_PR_RE.search(subject)
        if not m:
            continue
        pr_number = int(m.group(1))
        source_branch = m.group(2)
        # The first non-empty body line is the squashed PR title in GitHub merges.
        pr_title = ""
        for line in body.splitlines():
            if line.strip():
                pr_title = line.strip()
                break
        out[pr_number] = {"pr_title": pr_title, "source_branch": source_branch}
    return out


def get_commit_note(repo_path: str, commit_hash: str) -> Optional[str]:
    """Read Gerrit/code-review note text, failing on git errors.

    Reads from all note refs (``--notes=*``). A successful empty result means
    that no note exists and returns None; command failures retain commit-level
    provenance through :class:`UnitExtractionError`.
    """
    output = run_git_unit(
        repo_path,
        ["log", "-1", "--notes=*", "--format=%N", commit_hash],
        commit_hash=commit_hash,
        filepath=None,
        operation="commit_note",
        timeout=30,
    )
    text = output.strip()
    return text or None


def repo_has_notes(repo_path: str) -> bool:
    """True when the repo has any refs/notes/* (so note extraction is meaningful)."""
    output = run_git_required(
        repo_path,
        ["for-each-ref", "--format=%(refname)", "refs/notes/"],
        operation="list git note refs",
    )
    return bool(output and output.strip())


def compute_file_local_commit_indices(
    repo_path: str,
    commit_hashes: list[str],
) -> dict[tuple[str, str], int]:
    indices, _files_by_commit = precompute_cpp_file_changes(repo_path, commit_hashes)
    return indices


def precompute_cpp_file_changes(
    repo_path: str,
    commit_hashes: list[str],
) -> tuple[dict[tuple[str, str], int], dict[str, list[str]]]:
    """Precompute per-file temporal commit indices over the EMITTED set.

    Contract: ``file_local_commit_index`` for ``(commit, filepath)`` is the
    0-based count of EARLIER commits (chronological, oldest-first) that modified
    the same file AND were ACCEPTED by :func:`get_commit_diffs` -- i.e. that
    passed the identical None-blob / diff-length
    (``MIN_DIFF_CHARS``..``MAX_DIFF_CHARS``) / content-size (``200000``)
    acceptance gate used when records are written. A commit whose diff is
    rejected by that gate produces no record and therefore MUST NOT advance the
    counter, so the indices carried on emitted records are contiguous
    (0, 1, 2, ...) and byte-identical to the values the original
    ``compute_file_local_commit_indices`` produced before the precompute
    refactor.

    Counting over the cheaper name-status-only set (:func:`get_commit_cpp_files`:
    ``is_primary_commit_file`` / ``should_skip_path`` / ``status == 'M'``) is
    NOT equivalent: it counts diff-filtered commits
    that are never emitted, silently inflating and gapping the indices for any
    file with an interspersed rejected-diff commit (e.g. 0, 2, 3 instead of
    0, 1, 2). That is a semantic change that makes corpora built before/after
    the refactor non-comparable, which is why the index must use the same diff
    gate as emission.

    Deciding that gate requires the blob/diff content, so this pass runs
    :func:`get_commit_diffs`. The per-commit ACCEPTED filepaths are carried
    forward in ``files_by_commit`` so the emit pass can request exactly those
    files (skipping name-status enumeration and the blobs of rejected files).

    Iteration order is deterministic (``reversed(commit_hashes)``, oldest-first).
    """
    counters: dict[str, int] = {}
    indices: dict[tuple[str, str], int] = {}
    files_by_commit: dict[str, list[str]] = {}
    for commit_hash in reversed(commit_hashes):
        file_diffs = get_commit_diffs(repo_path, commit_hash)
        if not file_diffs:
            continue
        accepted = [item["filepath"] for item in file_diffs]
        files_by_commit[commit_hash] = accepted
        for filepath in accepted:
            next_index = counters.get(filepath, 0)
            indices[(commit_hash, filepath)] = next_index
            counters[filepath] = next_index + 1
    return indices, files_by_commit


def get_commit_cpp_files(repo_path: str, commit_hash: str) -> Optional[list[str]]:
    """Return modified primary native-domain paths without reading blobs/diffs."""
    name_status = run_git_unit(
        repo_path,
        ["diff-tree", "--no-commit-id", "-r", "--name-status", commit_hash],
        commit_hash=commit_hash,
        filepath=None,
        operation="commit_paths",
    )
    if not name_status.strip():
        return None

    files = []
    for line in name_status.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0]:
            raise UnitExtractionError(
                repo_path=repo_path,
                commit_hash=commit_hash,
                filepath=None,
                operation="commit_paths_parse",
                error_type="MalformedGitOutput",
                detail=f"invalid --name-status line: {line!r}",
            )
        status = parts[0][0]
        filepath = parts[-1]
        if not is_primary_commit_file(filepath):
            continue
        if should_skip_path(filepath):
            continue
        if status not in ("M",):
            continue
        files.append(filepath)

    return files or None


def get_file_diff(
    repo_path: str,
    commit_hash: str,
    filepath: str,
) -> Optional[dict[str, str]]:
    """Read and gate one modified file, raising with exact unit provenance."""
    old_content = run_git_unit(
        repo_path,
        ["show", f"{commit_hash}^:{filepath}"],
        commit_hash=commit_hash,
        filepath=filepath,
        operation="old_blob",
        timeout=30,
    )
    new_content = run_git_unit(
        repo_path,
        ["show", f"{commit_hash}:{filepath}"],
        commit_hash=commit_hash,
        filepath=filepath,
        operation="new_blob",
        timeout=30,
    )
    diff = run_git_unit(
        repo_path,
        ["diff", f"{commit_hash}^", commit_hash, "--", filepath],
        commit_hash=commit_hash,
        filepath=filepath,
        operation="file_diff",
        timeout=30,
    )

    diff_len = len(diff)
    if diff_len < MIN_DIFF_CHARS or diff_len > MAX_DIFF_CHARS:
        return None
    if len(old_content) > 200000 or len(new_content) > 200000:
        return None
    return {
        "filepath": filepath,
        "old_content": old_content,
        "new_content": new_content,
        "diff": diff,
    }


def get_commit_diffs(
    repo_path: str,
    commit_hash: str,
    files: Optional[list[str]] = None,
) -> Optional[list[dict]]:
    """Get per-file diffs for primary native-domain files in a commit."""
    if files is None:
        files = get_commit_cpp_files(repo_path, commit_hash)
    if not files:
        return None
    results = []
    for filepath in files:
        file_diff = get_file_diff(repo_path, commit_hash, filepath)
        if file_diff is not None:
            results.append(file_diff)

    return results if results else None


def checkpoint_root_for_output(output_path: str | Path) -> Path:
    return Path(f"{output_path}{CHECKPOINT_SUFFIX}")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _hash_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> tuple[int, int, str]:
    size = 0
    lines = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            lines += block.count(b"\n")
            digest.update(block)
    return size, lines, digest.hexdigest()


def _copy_exact(
    source: BinaryIO,
    destination: BinaryIO,
    length: int,
) -> None:
    remaining = length
    while remaining:
        block = source.read(min(1024 * 1024, remaining))
        if not block:
            raise CheckpointCorruptionError(
                f"scratch block ended {remaining} bytes before its recorded boundary"
            )
        destination.write(block)
        remaining -= len(block)


def _notes_refs_digest(repo_path: str) -> str:
    output = run_git_required(
        repo_path,
        [
            "for-each-ref",
            "--format=%(refname)%00%(objectname)",
            "refs/notes/",
        ],
        operation="fingerprint git note refs",
    )
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def _repo_source_context(
    repo_path: str,
    *,
    max_commits: int,
    repo_name: str,
    notes: str,
) -> dict:
    requested_repo_name = repo_name or Path(repo_path).name
    repo_url = resolve_repo_url(repo_path)
    owner_repo = resolve_owner_repo(repo_path)
    explicit_project_id = "/" in requested_repo_name
    if explicit_project_id:
        canonical_requested = require_project_identity(
            requested_repo_name,
            source="extract_git_history --repo-name",
        )
        if owner_repo is not None and owner_repo != canonical_requested:
            raise RuntimeError(
                f"explicit project identity {canonical_requested!r} conflicts "
                f"with clone remote identity {owner_repo!r}"
            )
        canonical_repo_name = canonical_requested
    else:
        canonical_repo_name = owner_repo or requested_repo_name

    if notes == "off":
        use_notes = False
    elif notes == "on":
        if not repo_has_notes(repo_path):
            raise RuntimeError(
                f"[{canonical_repo_name}] --notes on requested but repo has no "
                "refs/notes/*"
            )
        use_notes = True
    elif notes == "auto":
        use_notes = repo_has_notes(repo_path)
    else:
        raise ValueError(f"Invalid --notes value: {notes!r}")

    depth_output = run_git_required(
        repo_path,
        ["rev-list", "--count", "HEAD"],
        operation="count repository commits",
    )
    try:
        commit_count = int(depth_output.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"invalid git rev-list count for {os.path.abspath(repo_path)}: "
            f"{depth_output!r}"
        ) from exc
    commits = [] if commit_count <= 1 else get_commit_list(repo_path, max_commits)
    head = run_git_required(
        repo_path,
        ["rev-parse", "HEAD"],
        operation="resolve repository HEAD",
    ).strip()
    source_payload = {
        "contract_version": EXTRACTION_CONTRACT_VERSION,
        "repo": canonical_repo_name,
        "repo_url": repo_url or "",
        "head": head,
        "commit_count": commit_count,
        "selected_commit_count": len(commits),
        "selected_commits_sha256": _hash_strings(commits),
        "max_commits": int(max_commits),
        "notes": notes,
        "notes_enabled": use_notes,
        "notes_refs_sha256": _notes_refs_digest(repo_path),
    }
    source_fingerprint = hashlib.sha256(
        json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        **source_payload,
        "source_fingerprint": source_fingerprint,
        "repo_path": os.path.abspath(repo_path),
        "commits": commits,
    }


class RepoExtractionCheckpoint:
    """SQLite-backed transaction boundary for one repository extraction."""

    def __init__(
        self,
        root: Path,
        *,
        source: dict,
        checkpoint_commits: int,
    ):
        if checkpoint_commits <= 0:
            raise ValueError("checkpoint_commits must be positive")
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "checkpoint.sqlite3"
        self.bad_units_path = self.root / "bad_units.jsonl"
        self.source = source
        self.checkpoint_commits = int(checkpoint_commits)
        self.conn = self._connect()
        try:
            self._initialize_or_validate()
            self._cleanup_uncommitted_temps()
        except BaseException:
            self.conn.close()
            raise

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except sqlite3.DatabaseError as exc:
            raise CheckpointCorruptionError(
                f"cannot open extraction checkpoint {self.db_path}: {exc}"
            ) from exc

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RepoExtractionCheckpoint":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _initialize_or_validate(self) -> None:
        try:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_index INTEGER PRIMARY KEY,
                    start_index INTEGER NOT NULL,
                    end_index INTEGER NOT NULL,
                    commits_sha256 TEXT NOT NULL,
                    artifact_name TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    line_count INTEGER NOT NULL,
                    sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS file_counters (
                    filepath TEXT PRIMARY KEY,
                    next_index INTEGER NOT NULL CHECK(next_index >= 0)
                );
                CREATE TABLE IF NOT EXISTS bad_units (
                    unit_key TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    repo_path TEXT NOT NULL,
                    commit_hash TEXT NOT NULL,
                    filepath TEXT,
                    operation TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    attempts INTEGER NOT NULL CHECK(attempts > 0)
                );
                """
            )
            integrity = self.conn.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as exc:
            raise CheckpointCorruptionError(
                f"invalid extraction checkpoint database {self.db_path}: {exc}"
            ) from exc
        if integrity is None or integrity[0] != "ok":
            detail = "<no result>" if integrity is None else str(integrity[0])
            raise CheckpointCorruptionError(
                f"SQLite integrity check failed for {self.db_path}: {detail}"
            )

        existing = self._meta_values()
        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "contract_version": EXTRACTION_CONTRACT_VERSION,
            "source_fingerprint": self.source["source_fingerprint"],
            "repo": self.source["repo"],
            "repo_url": self.source["repo_url"],
            "head": self.source["head"],
            "selected_commit_count": self.source["selected_commit_count"],
            "selected_commits_sha256": self.source["selected_commits_sha256"],
            "max_commits": self.source["max_commits"],
            "notes": self.source["notes"],
            "notes_enabled": self.source["notes_enabled"],
            "notes_refs_sha256": self.source["notes_refs_sha256"],
            "checkpoint_commits": self.checkpoint_commits,
        }
        if not existing:
            initial = {
                **expected,
                "record_repo_path": self.source["repo_path"],
                "status": "in_progress",
            }
            with self.conn:
                self.conn.executemany(
                    "INSERT INTO meta(key, value_json) VALUES (?, ?)",
                    [
                        (key, json.dumps(value, sort_keys=True))
                        for key, value in initial.items()
                    ],
                )
            return

        mismatches = {
            key: {"checkpoint": existing.get(key), "current": value}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise CheckpointCorruptionError(
                f"checkpoint source/config mismatch at {self.db_path}: "
                f"{json.dumps(mismatches, sort_keys=True)}; use "
                "--reset-checkpoint for an explicit restart"
            )
        if existing.get("status") == "corrupt":
            raise CheckpointCorruptionError(
                f"checkpoint is marked corrupt: {self.db_path}; use "
                "--reset-checkpoint after inspecting the artifacts"
            )

    def _cleanup_uncommitted_temps(self) -> None:
        for pattern in ("chunk-*.spool.tmp.*", "chunk-*.jsonl.tmp.*"):
            for path in self.root.glob(pattern):
                path.unlink(missing_ok=True)

    def _meta_values(self) -> dict:
        try:
            rows = self.conn.execute("SELECT key, value_json FROM meta").fetchall()
            return {row["key"]: json.loads(row["value_json"]) for row in rows}
        except (json.JSONDecodeError, sqlite3.DatabaseError) as exc:
            raise CheckpointCorruptionError(
                f"cannot decode checkpoint metadata in {self.db_path}: {exc}"
            ) from exc

    def meta(self, key: str):
        row = self.conn.execute(
            "SELECT value_json FROM meta WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise CheckpointCorruptionError(
                f"checkpoint metadata {key!r} is missing from {self.db_path}"
            )
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError as exc:
            raise CheckpointCorruptionError(
                f"checkpoint metadata {key!r} is invalid in {self.db_path}: {exc}"
            ) from exc

    def set_status(self, status: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value_json) VALUES ('status', ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (json.dumps(status),),
            )

    @property
    def record_repo_path(self) -> str:
        return str(self.meta("record_repo_path"))

    def chunk_rows(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunks ORDER BY chunk_index"
        ).fetchall()

    def _mark_corrupt_and_raise(self, detail: str) -> None:
        self.set_status("corrupt")
        raise CheckpointCorruptionError(
            f"corrupt extraction checkpoint {self.db_path}: {detail}"
        )

    def validate_committed_chunks(
        self,
        *,
        verify_artifacts: bool = True,
    ) -> list[sqlite3.Row]:
        rows = self.chunk_rows()
        commits = self.source["commits"]
        chronological = list(reversed(commits))
        for expected_index, row in enumerate(rows):
            start = expected_index * self.checkpoint_commits
            end = min(start + self.checkpoint_commits, len(chronological))
            if row["chunk_index"] != expected_index:
                self._mark_corrupt_and_raise(
                    f"chunk index gap: expected {expected_index}, "
                    f"got {row['chunk_index']}"
                )
            if row["start_index"] != start or row["end_index"] != end:
                self._mark_corrupt_and_raise(
                    f"chunk {expected_index} range mismatch: checkpoint="
                    f"[{row['start_index']}:{row['end_index']}], "
                    f"expected=[{start}:{end}]"
                )
            expected_digest = _hash_strings(chronological[start:end])
            if row["commits_sha256"] != expected_digest:
                self._mark_corrupt_and_raise(
                    f"chunk {expected_index} commit digest mismatch"
                )
            artifact = self.root / row["artifact_name"]
            if artifact.parent != self.root or not artifact.is_file():
                self._mark_corrupt_and_raise(
                    f"chunk {expected_index} artifact is missing: {artifact}"
                )
            if verify_artifacts:
                size, lines, digest = _file_fingerprint(artifact)
                if (
                    size != row["size_bytes"]
                    or lines != row["line_count"]
                    or digest != row["sha256"]
                ):
                    self._mark_corrupt_and_raise(
                        f"chunk {expected_index} artifact fingerprint mismatch: "
                        f"actual=(size={size}, lines={lines}, sha256={digest}), "
                        f"checkpoint=(size={row['size_bytes']}, "
                        f"lines={row['line_count']}, sha256={row['sha256']})"
                    )
        expected_chunks = (
            (len(chronological) + self.checkpoint_commits - 1)
            // self.checkpoint_commits
        )
        if len(rows) > expected_chunks:
            self._mark_corrupt_and_raise(
                f"checkpoint has {len(rows)} chunks for {expected_chunks} ranges"
            )
        if self.meta("status") == "chunks_complete" and len(rows) != expected_chunks:
            self._mark_corrupt_and_raise(
                f"status is chunks_complete but only {len(rows)}/{expected_chunks} "
                "chunks are committed"
            )
        return rows

    def next_file_index(self, filepath: str, pending: dict[str, int]) -> int:
        if filepath in pending:
            next_index = pending[filepath]
        else:
            row = self.conn.execute(
                "SELECT next_index FROM file_counters WHERE filepath = ?",
                (filepath,),
            ).fetchone()
            next_index = 0 if row is None else int(row["next_index"])
        pending[filepath] = next_index + 1
        return next_index

    def record_bad_unit(
        self,
        error: UnitExtractionError,
        *,
        chunk_index: int,
        policy: str,
        max_bad_units: int,
    ) -> None:
        unit_key = hashlib.sha256(
            "\0".join(
                [
                    self.source["repo"],
                    error.commit_hash,
                    error.filepath or "",
                    error.operation,
                ]
            ).encode("utf-8")
        ).hexdigest()
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO bad_units(
                    unit_key, repo, repo_path, commit_hash, filepath, operation,
                    error_type, error, chunk_index, first_seen, last_seen, attempts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(unit_key) DO UPDATE SET
                    repo_path = excluded.repo_path,
                    error_type = excluded.error_type,
                    error = excluded.error,
                    chunk_index = excluded.chunk_index,
                    last_seen = excluded.last_seen,
                    attempts = bad_units.attempts + 1
                """,
                (
                    unit_key,
                    self.source["repo"],
                    error.repo_path,
                    error.commit_hash,
                    error.filepath,
                    error.operation,
                    error.error_type,
                    error.detail,
                    int(chunk_index),
                    now,
                    now,
                ),
            )
        self._export_bad_units()
        count = int(self.conn.execute("SELECT COUNT(*) FROM bad_units").fetchone()[0])
        location = error.filepath if error.filepath is not None else "<commit>"
        policy_detail = (
            f"policy={policy}, distinct_bad_units={count}, "
            f"max_bad_units={max_bad_units}"
        )
        if policy == "fail" or count > max_bad_units:
            self.set_status("failed")
            raise RuntimeError(
                f"bad extraction unit rejected ({policy_detail}): "
                f"repo={self.source['repo']} commit={error.commit_hash} "
                f"path={location} operation={error.operation} "
                f"error={error.error_type}: {error.detail}; ledger="
                f"{self.bad_units_path}"
            ) from error
        print(
            f"  [{self.source['repo']}] QUARANTINE commit={error.commit_hash} "
            f"path={location} operation={error.operation} "
            f"error={error.error_type}: {error.detail} ({policy_detail})",
            file=sys.stderr,
        )

    def _export_bad_units(self) -> None:
        rows = self.conn.execute(
            "SELECT * FROM bad_units ORDER BY first_seen, unit_key"
        ).fetchall()
        with atomic_output_file(self.bad_units_path) as staged:
            with staged.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(dict(row), sort_keys=True) + "\n")

    def bad_unit_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM bad_units").fetchone()[0])

    def commit_chunk(
        self,
        *,
        chunk_index: int,
        start_index: int,
        end_index: int,
        commit_hashes: list[str],
        staged_artifact: Path,
        pending_counters: dict[str, int],
        line_count: int,
    ) -> Path:
        artifact_name = f"chunk-{chunk_index:08d}.jsonl"
        artifact = self.root / artifact_name
        size, actual_lines, digest = _file_fingerprint(staged_artifact)
        if actual_lines != line_count:
            raise CheckpointCorruptionError(
                f"chunk {chunk_index} staged line count changed: "
                f"expected={line_count}, actual={actual_lines}"
            )
        os.replace(staged_artifact, artifact)
        _fsync_directory(self.root)
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            expected = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if int(expected) != chunk_index:
                raise CheckpointCorruptionError(
                    f"chunk transaction out of order: expected {expected}, "
                    f"got {chunk_index}"
                )
            self.conn.executemany(
                """
                INSERT INTO file_counters(filepath, next_index) VALUES (?, ?)
                ON CONFLICT(filepath) DO UPDATE SET
                    next_index = excluded.next_index
                """,
                sorted(pending_counters.items()),
            )
            self.conn.execute(
                """
                INSERT INTO chunks(
                    chunk_index, start_index, end_index, commits_sha256,
                    artifact_name, size_bytes, line_count, sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_index,
                    start_index,
                    end_index,
                    _hash_strings(commit_hashes),
                    artifact_name,
                    size,
                    line_count,
                    digest,
                ),
            )
            self.conn.execute(
                "UPDATE meta SET value_json = ? WHERE key = 'status'",
                (json.dumps("in_progress"),),
            )
            self.conn.commit()
        except (sqlite3.DatabaseError, CheckpointCorruptionError):
            self.conn.rollback()
            raise
        return artifact

    def mark_chunks_complete(self) -> None:
        rows = self.validate_committed_chunks(verify_artifacts=False)
        total = len(self.source["commits"])
        expected = (
            (total + self.checkpoint_commits - 1) // self.checkpoint_commits
        )
        if len(rows) != expected:
            raise CheckpointCorruptionError(
                f"cannot complete checkpoint with {len(rows)}/{expected} chunks"
            )
        self.set_status("chunks_complete")

    def cleanup_artifacts(self) -> None:
        for row in self.chunk_rows():
            (self.root / row["artifact_name"]).unlink(missing_ok=True)
        for scratch in self.root.glob("*.tmp.*"):
            scratch.unlink(missing_ok=True)


def _subject_is_skipped(subject: str) -> bool:
    lowered = subject.lower()
    return any(
        skip in lowered
        for skip in [
            "merge branch",
            "merge pull request",
            "update submodule",
            "bump version",
            "auto-generated",
            "clang-format",
            "fix whitespace",
            "fix typo in comment",
        ]
    )


def _extract_commit_records(
    *,
    repo_path: str,
    commit_hash: str,
    checkpoint: RepoExtractionCheckpoint,
    pending_counters: dict[str, int],
    chunk_index: int,
    policy: str,
    max_bad_units: int,
    merge_pr_map: dict[int, dict[str, str]],
    use_notes: bool,
) -> list[bytes]:
    try:
        files = get_commit_cpp_files(repo_path, commit_hash)
    except UnitExtractionError as exc:
        checkpoint.record_bad_unit(
            exc,
            chunk_index=chunk_index,
            policy=policy,
            max_bad_units=max_bad_units,
        )
        return []
    if not files:
        return []

    accepted: list[tuple[dict[str, str], int]] = []
    for filepath in files:
        try:
            file_diff = get_file_diff(repo_path, commit_hash, filepath)
        except UnitExtractionError as exc:
            checkpoint.record_bad_unit(
                exc,
                chunk_index=chunk_index,
                policy=policy,
                max_bad_units=max_bad_units,
            )
            continue
        if file_diff is None:
            continue
        file_index = checkpoint.next_file_index(filepath, pending_counters)
        accepted.append((file_diff, file_index))
    if not accepted:
        return []

    # The historical two-pass implementation advanced file counters before
    # subject/metadata filtering. Keep that exact behavior for corpus identity.
    try:
        commit_info = get_commit_info(repo_path, commit_hash)
    except UnitExtractionError as exc:
        checkpoint.record_bad_unit(
            exc,
            chunk_index=chunk_index,
            policy=policy,
            max_bad_units=max_bad_units,
        )
        return []
    if commit_info is None:
        raise AssertionError("get_commit_info returned None without an error")
    if _subject_is_skipped(commit_info["subject"]):
        return []

    pr_number = parse_pr_number_from_text(
        commit_info["subject"], commit_info["body"]
    )
    pr_title = ""
    source_branch = ""
    if pr_number is not None and pr_number in merge_pr_map:
        pr_title = merge_pr_map[pr_number]["pr_title"]
        source_branch = merge_pr_map[pr_number]["source_branch"]

    note_text = ""
    if use_notes:
        try:
            fetched_note = get_commit_note(repo_path, commit_hash)
        except UnitExtractionError as exc:
            checkpoint.record_bad_unit(
                exc,
                chunk_index=chunk_index,
                policy=policy,
                max_bad_units=max_bad_units,
            )
            return []
        if fetched_note:
            note_text = fetched_note

    records: list[bytes] = []
    repo_name = checkpoint.source["repo"]
    repo_url = checkpoint.source["repo_url"]
    for file_diff, file_index in accepted:
        filepath = file_diff["filepath"]
        record = {
            "old_content": file_diff["old_content"],
            "new_content": file_diff["new_content"],
            "diff": file_diff["diff"],
            "subject": commit_info["subject"],
            "body": commit_info["body"],
            "filepath": filepath,
            "repo": repo_name,
            "repo_url": repo_url,
            "repo_path": checkpoint.record_repo_path,
            "commit_hash": commit_info["hash"],
            "timestamp": commit_info["timestamp"],
            "pr_number": pr_number,
            "pr_title": pr_title,
            "source_branch": source_branch,
            "trailers": commit_info.get("trailers", ""),
            "note_text": note_text,
            "parent_hashes": list(commit_info["parent_hashes"]),
            "parent_count": int(commit_info["parent_count"]),
            "is_merge_commit": bool(commit_info["is_merge_commit"]),
            "author_timestamp": commit_info["author_timestamp"],
            "commit_timestamp": commit_info["commit_timestamp"],
            "repo_stable_id": stable_repo_id(repo_name),
            "filepath_stable_id": stable_filepath_id(repo_name, filepath),
            "file_local_commit_index": int(file_index),
            "has_ambiguous_reconstruction": False,
            "has_rename_ambiguity": False,
        }
        records.append((json.dumps(record) + "\n").encode("utf-8"))
    return records


def _process_checkpoint_chunk(
    *,
    repo_path: str,
    checkpoint: RepoExtractionCheckpoint,
    chunk_index: int,
    commit_hashes: list[str],
    start_index: int,
    policy: str,
    max_bad_units: int,
    merge_pr_map: dict[int, dict[str, str]],
    use_notes: bool,
    memory_limit_gb: float,
) -> int:
    spool = checkpoint.root / f"chunk-{chunk_index:08d}.spool.tmp.{os.getpid()}"
    staged = checkpoint.root / f"chunk-{chunk_index:08d}.jsonl.tmp.{os.getpid()}"
    spool.unlink(missing_ok=True)
    staged.unlink(missing_ok=True)
    pending_counters: dict[str, int] = {}
    blocks: list[tuple[int, int]] = []
    line_count = 0
    try:
        with spool.open("w+b") as spool_handle:
            for offset, commit_hash in enumerate(commit_hashes):
                absolute_index = start_index + offset
                if absolute_index > 0 and absolute_index % 1000 == 0:
                    check_memory_limit(
                        memory_limit_gb,
                        label="extract_git_history",
                    )
                records = _extract_commit_records(
                    repo_path=repo_path,
                    commit_hash=commit_hash,
                    checkpoint=checkpoint,
                    pending_counters=pending_counters,
                    chunk_index=chunk_index,
                    policy=policy,
                    max_bad_units=max_bad_units,
                    merge_pr_map=merge_pr_map,
                    use_notes=use_notes,
                )
                if not records:
                    continue
                block_start = spool_handle.tell()
                for record in records:
                    spool_handle.write(record)
                block_end = spool_handle.tell()
                blocks.append((block_start, block_end - block_start))
                line_count += len(records)
            spool_handle.flush()
            os.fsync(spool_handle.fileno())

            with staged.open("wb") as staged_handle:
                for block_start, block_length in reversed(blocks):
                    spool_handle.seek(block_start)
                    _copy_exact(spool_handle, staged_handle, block_length)
                staged_handle.flush()
                os.fsync(staged_handle.fileno())

        checkpoint.commit_chunk(
            chunk_index=chunk_index,
            start_index=start_index,
            end_index=start_index + len(commit_hashes),
            commit_hashes=commit_hashes,
            staged_artifact=staged,
            pending_counters=pending_counters,
            line_count=line_count,
        )
        return line_count
    finally:
        spool.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)


def extract_repo_to_checkpoint(
    repo_path: str,
    checkpoint: RepoExtractionCheckpoint,
    *,
    memory_limit_gb: float,
    bad_unit_policy: str,
    max_bad_units: int,
) -> _ExtractionStats:
    if bad_unit_policy not in {"fail", "quarantine"}:
        raise ValueError(f"invalid bad unit policy: {bad_unit_policy!r}")
    if bad_unit_policy == "fail" and max_bad_units != 0:
        raise ValueError("bad-unit-policy=fail requires --max-bad-units=0")
    if bad_unit_policy == "quarantine" and max_bad_units <= 0:
        raise ValueError("bad-unit-policy=quarantine requires --max-bad-units > 0")

    existing_bad_units = checkpoint.bad_unit_count()
    allowed_bad_units = 0 if bad_unit_policy == "fail" else max_bad_units
    if existing_bad_units > allowed_bad_units:
        checkpoint.set_status("failed")
        raise RuntimeError(
            f"checkpoint already contains {existing_bad_units} distinct bad units, "
            f"exceeding policy={bad_unit_policy} max_bad_units="
            f"{allowed_bad_units}; ledger={checkpoint.bad_units_path}"
        )

    committed = checkpoint.validate_committed_chunks()
    commits = checkpoint.source["commits"]
    chronological = list(reversed(commits))
    expected_chunks = (
        (len(chronological) + checkpoint.checkpoint_commits - 1)
        // checkpoint.checkpoint_commits
    )
    if checkpoint.meta("status") == "chunks_complete":
        return {
            "repo": checkpoint.source["repo"],
            "commits_checked": len(commits),
            "records_written": sum(int(row["line_count"]) for row in committed),
        }

    checkpoint.set_status("in_progress")
    merge_pr_map = build_merge_pr_map(repo_path) if len(committed) < expected_chunks else {}
    if merge_pr_map:
        print(
            f"  [{checkpoint.source['repo']}] Mined {len(merge_pr_map):,} "
            "PR merge commits"
        )
    if checkpoint.source["notes_enabled"]:
        print(f"  [{checkpoint.source['repo']}] Gerrit/code-review notes enabled")

    records_written = sum(int(row["line_count"]) for row in committed)
    for chunk_index in range(len(committed), expected_chunks):
        start = chunk_index * checkpoint.checkpoint_commits
        end = min(start + checkpoint.checkpoint_commits, len(chronological))
        line_count = _process_checkpoint_chunk(
            repo_path=repo_path,
            checkpoint=checkpoint,
            chunk_index=chunk_index,
            commit_hashes=chronological[start:end],
            start_index=start,
            policy=bad_unit_policy,
            max_bad_units=max_bad_units,
            merge_pr_map=merge_pr_map,
            use_notes=bool(checkpoint.source["notes_enabled"]),
            memory_limit_gb=memory_limit_gb,
        )
        records_written += line_count
        print(
            f"  [{checkpoint.source['repo']}] Checkpointed chunk "
            f"{chunk_index + 1:,}/{expected_chunks:,} "
            f"({end:,}/{len(chronological):,} commits, "
            f"{records_written:,} records)"
        )
    checkpoint.mark_chunks_complete()
    return {
        "repo": checkpoint.source["repo"],
        "commits_checked": len(commits),
        "records_written": records_written,
    }


def _copy_verified_chunk(
    checkpoint: RepoExtractionCheckpoint,
    row: sqlite3.Row,
    destination: BinaryIO,
    output_digest,
) -> tuple[int, int]:
    artifact = checkpoint.root / row["artifact_name"]
    size = 0
    lines = 0
    digest = hashlib.sha256()
    with artifact.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            lines += block.count(b"\n")
            digest.update(block)
            output_digest.update(block)
            destination.write(block)
    actual_digest = digest.hexdigest()
    if (
        size != row["size_bytes"]
        or lines != row["line_count"]
        or actual_digest != row["sha256"]
    ):
        checkpoint._mark_corrupt_and_raise(
            f"chunk {row['chunk_index']} changed during publication: "
            f"actual=(size={size}, lines={lines}, sha256={actual_digest}), "
            f"checkpoint=(size={row['size_bytes']}, lines={row['line_count']}, "
            f"sha256={row['sha256']})"
        )
    return size, lines


def publish_checkpoints(
    output_path: str | Path,
    checkpoints: list[RepoExtractionCheckpoint],
) -> dict[str, int | str]:
    """Verify every committed chunk and atomically publish ordered JSONL."""
    rows_by_checkpoint: list[tuple[RepoExtractionCheckpoint, list[sqlite3.Row]]] = []
    for checkpoint in checkpoints:
        rows = checkpoint.validate_committed_chunks(verify_artifacts=False)
        if checkpoint.meta("status") != "chunks_complete":
            raise CheckpointCorruptionError(
                f"cannot publish incomplete checkpoint: {checkpoint.db_path}"
            )
        rows_by_checkpoint.append((checkpoint, rows))

    output_digest = hashlib.sha256()
    size = 0
    lines = 0
    with atomic_output_file(output_path) as staged_output:
        with staged_output.open("wb") as destination:
            for checkpoint, rows in rows_by_checkpoint:
                for row in reversed(rows):
                    chunk_size, chunk_lines = _copy_verified_chunk(
                        checkpoint,
                        row,
                        destination,
                        output_digest,
                    )
                    size += chunk_size
                    lines += chunk_lines
            destination.flush()
            os.fsync(destination.fileno())
    return {
        "size_bytes": size,
        "line_count": lines,
        "sha256": output_digest.hexdigest(),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    with atomic_output_file(path) as staged:
        staged.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _publication_state_path(checkpoint_root: Path) -> Path:
    return checkpoint_root / "publication.json"


def _job_fingerprint(
    sources: list[dict],
    *,
    checkpoint_commits: int,
    bad_unit_policy: str,
    max_bad_units: int,
) -> str:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "contract_version": EXTRACTION_CONTRACT_VERSION,
        "checkpoint_commits": checkpoint_commits,
        "bad_unit_policy": bad_unit_policy,
        "max_bad_units": max_bad_units,
        "repos": [source["source_fingerprint"] for source in sources],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_publication_state(checkpoint_root: Path) -> Optional[dict]:
    path = _publication_state_path(checkpoint_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError(
            f"invalid publication state {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointCorruptionError(
            f"publication state must be a JSON object: {path}"
        )
    return payload


def _completed_publication(
    *,
    output_path: Path,
    checkpoint_root: Path,
    job_fingerprint: str,
) -> Optional[dict]:
    state = _load_publication_state(checkpoint_root)
    if state is None or state.get("status") != "done":
        if state is not None and state.get("status") == "corrupt":
            raise CheckpointCorruptionError(
                f"publication is marked corrupt in "
                f"{_publication_state_path(checkpoint_root)}; use "
                "--reset-checkpoint after inspecting it"
            )
        return None
    if state.get("job_fingerprint") != job_fingerprint:
        raise CheckpointCorruptionError(
            "completed extraction checkpoint does not match the current "
            "repository/config; use --reset-checkpoint for an explicit restart"
        )
    if not output_path.is_file():
        state["status"] = "corrupt"
        state["corruption"] = f"published output is missing: {output_path}"
        _write_json_atomic(_publication_state_path(checkpoint_root), state)
        raise CheckpointCorruptionError(state["corruption"])
    size, lines, digest = _file_fingerprint(output_path)
    expected = state.get("output", {})
    if (
        size != expected.get("size_bytes")
        or lines != expected.get("line_count")
        or digest != expected.get("sha256")
    ):
        state["status"] = "corrupt"
        state["corruption"] = (
            f"published output fingerprint mismatch for {output_path}: "
            f"actual=(size={size}, lines={lines}, sha256={digest}), "
            f"expected={expected}"
        )
        _write_json_atomic(_publication_state_path(checkpoint_root), state)
        raise CheckpointCorruptionError(state["corruption"])
    return state


def process_repo(
    repo_path: str,
    output_file,
    max_commits: int = 0,
    repo_name: str = "",
    memory_limit_gb: float = 10.0,
    *,
    notes: str = "auto",
) -> _ExtractionStats:
    if not repo_name:
        repo_name = Path(repo_path).name

    # Canonical owner/repo from the clone's git remote (authoritative; not in
    # git history). When resolvable it overrides the bare directory name so each
    # record carries the same key the Tier-2 PR store is keyed by. repo_url keeps
    # the raw remote so downstream consumers can reconstruct the canonical URL.
    repo_url = resolve_repo_url(repo_path)
    owner_repo = resolve_owner_repo(repo_path)
    if owner_repo:
        repo_name = owner_repo

    stats: _ExtractionStats = {
        "repo": repo_name,
        "commits_checked": 0,
        "records_written": 0,
    }

    # Merge-commit mining: per-repo {pr_number -> {pr_title, source_branch}}.
    merge_pr_map = build_merge_pr_map(repo_path)
    if merge_pr_map:
        print(f"  [{repo_name}] Mined {len(merge_pr_map):,} PR merge commits")

    # Gerrit/code-review note extraction. notes="off" disables; "on" requires
    # refs/notes to exist (fail loud if requested-but-missing); "auto" enables
    # only when the repo actually has notes.
    if notes == "off":
        use_notes = False
    elif notes == "on":
        if not repo_has_notes(repo_path):
            raise RuntimeError(
                f"[{repo_name}] --notes on requested but repo has no refs/notes/*"
            )
        use_notes = True
    elif notes == "auto":
        use_notes = repo_has_notes(repo_path)
    else:
        raise ValueError(f"Invalid --notes value: {notes!r}")
    if use_notes:
        print(f"  [{repo_name}] Gerrit/code-review notes enabled")

    depth_output = run_git(repo_path, ["rev-list", "--count", "HEAD"])
    if depth_output:
        commit_count = int(depth_output.strip())
        if commit_count <= 1:
            print(f"  [{repo_name}] Shallow clone (1 commit), skipping")
            return stats
        print(f"  [{repo_name}] {commit_count:,} commits available")
    else:
        print(f"  [{repo_name}] Cannot count commits, skipping")
        return stats

    commits = get_commit_list(repo_path, max_commits)
    if not commits:
        print(f"  [{repo_name}] No commits found")
        return stats

    file_local_commit_indices, cpp_files_by_commit = precompute_cpp_file_changes(
        repo_path, commits
    )

    for i, commit_hash in enumerate(commits):
        if i > 0 and i % 1000 == 0:
            check_memory_limit(memory_limit_gb, label="extract_git_history")
            print(
                f"  [{repo_name}] Processed {i:,}/{len(commits):,} commits, "
                f"{stats['records_written']:,} records"
            )

        stats["commits_checked"] += 1

        commit_info = get_commit_info(repo_path, commit_hash)
        if not commit_info:
            continue

        subject = commit_info["subject"].lower()
        if any(
            skip in subject
            for skip in [
                "merge branch",
                "merge pull request",
                "update submodule",
                "bump version",
                "auto-generated",
                "clang-format",
                "fix whitespace",
                "fix typo in comment",
            ]
        ):
            continue

        file_diffs = get_commit_diffs(
            repo_path,
            commit_hash,
            files=cpp_files_by_commit.get(commit_hash),
        )
        if not file_diffs:
            continue

        # Resolve PR provenance: parse pr_number from subject/body, then attach
        # the real PR title / source branch from the merge map when known.
        pr_number = parse_pr_number_from_text(
            commit_info["subject"], commit_info["body"]
        )
        pr_title = ""
        source_branch = ""
        if pr_number is not None and pr_number in merge_pr_map:
            pr_title = merge_pr_map[pr_number]["pr_title"]
            source_branch = merge_pr_map[pr_number]["source_branch"]

        note_text = ""
        if use_notes:
            fetched_note = get_commit_note(repo_path, commit_hash)
            if fetched_note:
                note_text = fetched_note

        for fd in file_diffs:
            record = {
                "old_content": fd["old_content"],
                "new_content": fd["new_content"],
                "diff": fd["diff"],
                "subject": commit_info["subject"],
                "body": commit_info["body"],
                "filepath": fd["filepath"],
                "repo": repo_name,
                "repo_url": repo_url or "",
                "repo_path": os.path.abspath(repo_path),
                "commit_hash": commit_info["hash"],
                "timestamp": commit_info["timestamp"],
                "pr_number": pr_number,
                "pr_title": pr_title,
                "source_branch": source_branch,
                "trailers": commit_info.get("trailers", ""),
                "note_text": note_text,
                "parent_hashes": list(commit_info["parent_hashes"]),
                "parent_count": int(commit_info["parent_count"]),
                "is_merge_commit": bool(commit_info["is_merge_commit"]),
                "author_timestamp": commit_info["author_timestamp"],
                "commit_timestamp": commit_info["commit_timestamp"],
                "repo_stable_id": stable_repo_id(repo_name),
                "filepath_stable_id": stable_filepath_id(repo_name, fd["filepath"]),
                "file_local_commit_index": int(
                    file_local_commit_indices[(commit_hash, fd["filepath"])]
                ),
                "has_ambiguous_reconstruction": False,
                "has_rename_ambiguity": False,
            }
            output_file.write(json.dumps(record) + "\n")
            stats["records_written"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw git commit data as JSONL for clang commit enrichment"
    )
    env_bad_unit_policy = os.environ.get(
        "CPPMEGA_EXTRACT_BAD_UNIT_POLICY",
        "fail",
    )
    env_max_bad_units_raw = os.environ.get("CPPMEGA_EXTRACT_MAX_BAD_UNITS", "0")
    try:
        env_max_bad_units = int(env_max_bad_units_raw)
    except ValueError:
        parser.error(
            "CPPMEGA_EXTRACT_MAX_BAD_UNITS must be an integer, got "
            f"{env_max_bad_units_raw!r}"
        )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo", help="Path to a single git repository")
    group.add_argument("--repo_dir", help="Directory containing multiple repos")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--max_commits", type=int, default=0, help="Max commits per repo (0 = all)"
    )
    parser.add_argument(
        "--repo-name",
        default=None,
        help=(
            "Explicit canonical project identity (owner/repo) for --repo. "
            "Use this when the staged checkout directory is a synthetic name "
            "such as _src."
        ),
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=10.0,
        help="Abort if this Python wrapper exceeds this max RSS in GiB (default: 10).",
    )
    parser.add_argument(
        "--notes",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Gerrit/code-review note extraction: 'auto' (only when refs/notes "
            "exist), 'on' (require notes, fail loud if absent), 'off' (disable)."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Durable extraction checkpoint directory (default: "
            "<output>.extract-checkpoint)."
        ),
    )
    parser.add_argument(
        "--checkpoint-commits",
        type=int,
        default=DEFAULT_CHECKPOINT_COMMITS,
        help=(
            "Commits per transactional checkpoint chunk "
            f"(default: {DEFAULT_CHECKPOINT_COMMITS})."
        ),
    )
    parser.add_argument(
        "--bad-unit-policy",
        choices=("fail", "quarantine"),
        default=env_bad_unit_policy,
        help=(
            "Policy for a git commit/path command failure. 'fail' stops after "
            "durably recording the unit; 'quarantine' skips only recorded units "
            "up to --max-bad-units. Default: fail; conveyor subprocesses can "
            "set CPPMEGA_EXTRACT_BAD_UNIT_POLICY explicitly."
        ),
    )
    parser.add_argument(
        "--max-bad-units",
        type=int,
        default=env_max_bad_units,
        help=(
            "Maximum distinct bad commit/path units allowed by quarantine; "
            "must be 0 with --bad-unit-policy=fail. Default: 0; conveyor "
            "subprocesses can set CPPMEGA_EXTRACT_MAX_BAD_UNITS explicitly."
        ),
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Explicitly discard the durable extraction checkpoint before starting.",
    )
    args = parser.parse_args()
    if args.repo_name and not args.repo:
        parser.error("--repo-name requires --repo")
    if args.repo_name:
        try:
            require_project_identity(
                args.repo_name,
                source="extract_git_history --repo-name",
            )
        except SymbolIdentityError as exc:
            parser.error(str(exc))
    if args.checkpoint_commits <= 0:
        parser.error("--checkpoint-commits must be positive")
    if args.bad_unit_policy not in {"fail", "quarantine"}:
        parser.error(
            "--bad-unit-policy/CPPMEGA_EXTRACT_BAD_UNIT_POLICY must be "
            "'fail' or 'quarantine'"
        )
    if args.bad_unit_policy == "fail" and args.max_bad_units != 0:
        parser.error("--bad-unit-policy=fail requires --max-bad-units=0")
    if args.bad_unit_policy == "quarantine" and args.max_bad_units <= 0:
        parser.error("--bad-unit-policy=quarantine requires --max-bad-units > 0")
    start_memory_guard(args.memory_limit_gb, label="extract_git_history")

    repos = []
    if args.repo:
        repos.append(args.repo)
    else:
        for entry in sorted(os.listdir(args.repo_dir)):
            path = os.path.join(args.repo_dir, entry)
            if os.path.isdir(os.path.join(path, ".git")):
                repos.append(path)

    print(f"Found {len(repos)} repositories")
    print(f"Max commits per repo: {args.max_commits or 'all'}")
    print(f"Memory limit: {args.memory_limit_gb} GiB")
    output_path = Path(args.output)
    checkpoint_root = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else checkpoint_root_for_output(output_path)
    )
    print(f"Output: {output_path}")
    print(f"Checkpoint: {checkpoint_root}")
    print(f"Checkpoint commits: {args.checkpoint_commits}")
    print(
        f"Bad unit policy: {args.bad_unit_policy} "
        f"(max={args.max_bad_units})"
    )
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    failed_repos: list[dict[str, str]] = []
    start_time = time.time()
    output_lock = OutputFileLock(output_path)
    output_lock.acquire()
    open_checkpoints: list[RepoExtractionCheckpoint] = []
    successful_checkpoints: list[RepoExtractionCheckpoint] = []
    successful_stats: list[_ExtractionStats] = []
    sources: list[Optional[dict]] = []
    publication: Optional[dict[str, int | str]] = None

    try:
        if args.reset_checkpoint and checkpoint_root.exists():
            shutil.rmtree(checkpoint_root)
            print(f"Reset checkpoint: removed {checkpoint_root}")
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        for repo_path in repos:
            repo_name = args.repo_name or Path(repo_path).name
            try:
                source = _repo_source_context(
                    repo_path,
                    max_commits=args.max_commits,
                    repo_name=repo_name,
                    notes=args.notes,
                )
                sources.append(source)
            except Exception as exc:
                sources.append(None)
                print(f"  [{repo_name}] ERROR: {exc}")
                failed_repos.append(
                    {
                        "repo_path": os.path.abspath(repo_path),
                        "repo_name": repo_name,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        complete_sources = [source for source in sources if source is not None]
        job_fingerprint = _job_fingerprint(
            complete_sources,
            checkpoint_commits=args.checkpoint_commits,
            bad_unit_policy=args.bad_unit_policy,
            max_bad_units=args.max_bad_units,
        )
        if not failed_repos and len(complete_sources) == len(repos):
            completed = _completed_publication(
                output_path=output_path,
                checkpoint_root=checkpoint_root,
                job_fingerprint=job_fingerprint,
            )
            if completed is not None:
                publication = completed["output"]
                successful_stats = completed.get("repos", [])
                total_records = int(publication["line_count"])
                for artifact in checkpoint_root.glob("repo-*/chunk-*.jsonl"):
                    artifact.unlink(missing_ok=True)
                print(
                    f"EXTRACT-CKPT HIT: validated published output "
                    f"({total_records:,} records); no commit ranges reprocessed"
                )

        if publication is None:
            for i, (repo_path, source) in enumerate(zip(repos, sources, strict=True)):
                repo_name = Path(repo_path).name
                if source is None:
                    continue
                print(f"[{i + 1}/{len(repos)}] {source['repo']}...")
                print(
                    f"  [{source['repo']}] {source['commit_count']:,} commits "
                    f"available; {len(source['commits']):,} selected"
                )
                checkpoint_path = (
                    checkpoint_root
                    / f"repo-{stable_repo_id(source['repo'])}"
                )
                checkpoint: Optional[RepoExtractionCheckpoint] = None
                try:
                    checkpoint = RepoExtractionCheckpoint(
                        checkpoint_path,
                        source=source,
                        checkpoint_commits=args.checkpoint_commits,
                    )
                    open_checkpoints.append(checkpoint)
                    stats = extract_repo_to_checkpoint(
                        repo_path,
                        checkpoint,
                        memory_limit_gb=args.memory_limit_gb,
                        bad_unit_policy=args.bad_unit_policy,
                        max_bad_units=args.max_bad_units,
                    )
                    total_records += stats["records_written"]
                    successful_stats.append(stats)
                    successful_checkpoints.append(checkpoint)
                    print(
                        f"  [{source['repo']}] {stats['records_written']:,} "
                        "records checkpointed"
                    )
                except Exception as exc:
                    print(f"  [{repo_name}] ERROR: {exc}")
                    failed_repos.append(
                        {
                            "repo_path": os.path.abspath(repo_path),
                            "repo_name": repo_name,
                            "checkpoint": str(checkpoint_path),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

            if successful_checkpoints:
                try:
                    publication = publish_checkpoints(
                        output_path,
                        successful_checkpoints,
                    )
                except Exception as exc:
                    failed_repos.append(
                        {
                            "repo_path": "",
                            "repo_name": "<publication>",
                            "checkpoint": str(checkpoint_root),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"  [publication] ERROR: {exc}")

            if publication is not None:
                state = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "contract_version": EXTRACTION_CONTRACT_VERSION,
                    "status": "failed_partial" if failed_repos else "done",
                    "job_fingerprint": job_fingerprint,
                    "output_path": str(output_path),
                    "output": publication,
                    "repos": successful_stats,
                    "failed_count": len(failed_repos),
                    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                _write_json_atomic(_publication_state_path(checkpoint_root), state)
                if not failed_repos:
                    for checkpoint in successful_checkpoints:
                        checkpoint.cleanup_artifacts()
    finally:
        for checkpoint in open_checkpoints:
            checkpoint.close()
        output_lock.close()

    elapsed = time.time() - start_time
    output_size = output_path.stat().st_size if output_path.exists() else 0
    print("\n=== SUMMARY ===")
    print(f"Repos: {len(repos)}")
    print(f"Failed repos: {len(failed_repos)}")
    print(f"Total records: {total_records:,}")
    print(f"Time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"Output: {output_path} ({output_size / (1024**3):.2f} GB)")

    if failed_repos:
        # One clear path: any repo that fails to extract fails the whole run.
        # Persist a durable failure manifest next to the output BEFORE raising so
        # the failing repos survive even when stdout is lost, then raise to exit
        # non-zero (no silent exit-0 over a partially-extracted corpus).
        manifest_path = Path(str(output_path) + ".failures.json")
        _write_json_atomic(
            manifest_path,
            {
                "output": str(output_path),
                "checkpoint_root": str(checkpoint_root),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "total_repos": len(repos),
                "failed_count": len(failed_repos),
                "failed": failed_repos,
            },
        )
        failed_names = ", ".join(item["repo_name"] for item in failed_repos)
        raise RuntimeError(
            f"{len(failed_repos)}/{len(repos)} repos failed extraction "
            f"({failed_names}); see failure manifest: {manifest_path}"
        )
    Path(str(output_path) + ".failures.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
