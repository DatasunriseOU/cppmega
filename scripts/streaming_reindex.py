#!/usr/bin/env python3
"""Resumable per-repo streaming re-indexer for the cpp_all corpus.

Pipeline (one repo at a time, fail-loud per repo, continue on failure):

    extract cpp_all/<repo> subtree from the zstd tarball  ->  temp dir
      -> index_project.py  (--enriched; channel A build_info from the repo's
         own compile_commands.json when present, empty otherwise -- NO boilerplate)
      -> clang_enriched_to_parquet.py (--materialize-tokenized-enriched,
         --overflow-policy split at the largest fixed bucket)
      -> pack_enriched_rows.py once per --target-length (best_fit, no-crop)
      -> append packed parquet to
         outputs/reindexed/{1024,2048,4096,8192,16384}/<repo>.parquet
      -> mark <repo> done in outputs/reindexed/_done.json
      -> rm temp

Commit sources (--commit-source ...) are re-indexed the SAME way via
process_commits.py instead of index_project.py; everything downstream is
identical. Commits average ~1986 tokens so they mostly land in the 2048/4096
buckets -- that is expected, not an error.

NO automated/silent fallbacks (RULE #1): every step runs through ONE clear
subprocess path; any non-zero exit, missing output, or empty result RAISES a
RepoFailure with WHERE+WHAT, which is recorded and the driver moves to the next
repo. There is no degraded / best-effort / zero-output path.

Source: /Users/dave/sources/parquet/data-cpp_all/data-cpp_all.tar.zst
  members: cpp_all/<repo>/...
  decompress: zstd -dc --long=31
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from cppmega.data.symbol_identity import (  # noqa: E402
    SymbolIdentityError,
    require_project_identity,
)

# --------------------------------------------------------------------------- #
# Fixed environment contract (verified by the task brief).
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = Path(sys.executable)
TOKENIZER_PATH = ROOT / "cppmega" / "tokenizer" / "tokenizer.json"

INDEX_PROJECT = ROOT / "tools" / "clang_indexer" / "index_project.py"
PROCESS_COMMITS = ROOT / "tools" / "clang_indexer" / "process_commits.py"
MATERIALIZER = ROOT / "scripts" / "nanochat_data" / "clang_enriched_to_parquet.py"
PACKER = ROOT / "scripts" / "nanochat_data" / "pack_enriched_rows.py"

TARBALL = Path("/Users/dave/sources/parquet/data-cpp_all/data-cpp_all.tar.zst")
TAR_MEMBER_ROOT = "cpp_all"

OUTPUT_ROOT = ROOT / "outputs" / "reindexed"
MANIFEST_PATH = OUTPUT_ROOT / "_done.json"
DEFAULT_REPO_LIST = ROOT / "outputs" / "pr_ingest" / "repo_list.json"
DEFAULT_SOURCE_QUARANTINE_MANIFEST = (
    ROOT / "configs" / "source_quarantine_manifest.json"
)

# Compatibility default for direct stage_materialize callers. Production
# route-by-fit callers pass a target-derived lossless split budget explicitly.
TOKENIZE_BUDGET = 65536
# Upstream indexers historically discarded docs above --max-tokens before the
# lossless materializer could see them. Use an effectively unbounded signed
# 64-bit value so token length alone never filters an enriched source document.
LOSSLESS_INDEX_MAX_TOKENS = (1 << 63) - 1
# materialize_tokenized_enriched adds BOS and, for routed domains, a pair of
# domain delimiter tokens. Further embedded-domain expansion is checked against
# the actual token_ids length and fails closed before routing.
MATERIALIZED_ROW_RESERVED_TOKENS = 3
MATERIALIZE_SPLIT_STATS_SCHEMA = "cppmega.materialize_split_stats_v1"
DEFAULT_TARGET_LENGTHS = (1024, 2048, 4096, 8192, 16384)

# Directories never worth indexing (VCS / build artifacts). index_project has its
# own excludes; we additionally avoid extracting .git to keep staging small for
# pure-source repos. For commit re-indexing the .git history is needed, so we do
# NOT strip it there.
SOURCE_EXTRA_EXCLUDE = ".git,.svn,node_modules,build,_build,cmake-build-debug"
EXCLUDE_PARTS = frozenset(SOURCE_EXTRA_EXCLUDE.split(","))
_PUBLICATION_LOCK = threading.Lock()


def iter_tar_members_without_cache(tar):
    """Yield streaming tar members without retaining all prior metadata.

    Python's ``tarfile`` keeps appending ``TarInfo`` objects to ``members`` even
    in ``r|`` mode.  Clearing that lookup cache after each header keeps archive
    traversal bounded on multi-million-file corpora.  The caller receives the
    current ``TarInfo`` object directly, so extracting that member remains safe.
    """

    for member in tar:
        tar.members.clear()
        yield member


def load_project_identity_map(repo_list: Path) -> dict[str, str]:
    if not repo_list.exists():
        raise FileNotFoundError(f"repo identity map does not exist: {repo_list}")
    with repo_list.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data.get("repos")
    if not isinstance(entries, list):
        raise SymbolIdentityError(f"{repo_list}: expected a repos list")
    identities: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SymbolIdentityError(f"{repo_list}: repos[{index}] must be an object")
        name = entry.get("name") or entry.get("bare_name")
        project_identity = entry.get("project_identity")
        owner_repo = entry.get("owner_repo")
        if not isinstance(name, str) or not name:
            continue
        if project_identity is None:
            project_identity = owner_repo
        project_id = require_project_identity(
            project_identity,
            source=f"{repo_list}:repos[{index}].project_identity",
        )
        if owner_repo is not None:
            github_project_id = require_project_identity(
                owner_repo,
                source=f"{repo_list}:repos[{index}].owner_repo",
            )
            if github_project_id != project_id:
                raise SymbolIdentityError(
                    f"{repo_list}: repos[{index}] has conflicting project_identity "
                    f"{project_id!r} and owner_repo {github_project_id!r}"
                )
        previous = identities.get(name)
        if previous is not None and previous != project_id:
            raise SymbolIdentityError(
                f"{repo_list}: bare repo {name!r} maps to both "
                f"{previous!r} and {project_id!r}"
            )
        identities[name] = project_id
    return identities


def resolve_project_identity(repo: str, repo_list: Path | None) -> str:
    if "/" in repo:
        return require_project_identity(repo, source="streaming repo")
    if repo_list is None:
        raise SymbolIdentityError(
            f"repo {repo!r} has no canonical project identity; provide --repo-list"
        )
    identities = load_project_identity_map(repo_list)
    try:
        return identities[repo]
    except KeyError as exc:
        raise SymbolIdentityError(
            f"{repo_list}: no canonical project identity for bare repo {repo!r}"
        ) from exc


def _route_by_fit_impl():
    """Import bucket_for/route_by_fit from the commits driver (the ONE shared
    implementation), deferred to avoid a module-load cycle (streaming_reindex_commits
    imports this module at top). Both streams thus route identically."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from streaming_reindex_commits import bucket_for, route_by_fit  # noqa: F401
    return bucket_for, route_by_fit


def lossless_materialize_budget(target_lengths: Sequence[int]) -> int:
    """Raw-text budget that leaves room for mandatory tokenized-row markers."""

    lengths = sorted({int(length) for length in target_lengths})
    if not lengths or lengths[0] <= 0:
        raise ValueError(f"target lengths must be positive and non-empty: {lengths}")
    budget = lengths[-1] - MATERIALIZED_ROW_RESERVED_TOKENS
    if budget <= 0:
        raise ValueError(
            "largest target length is too small for BOS/domain delimiters: "
            f"{lengths[-1]} <= {MATERIALIZED_ROW_RESERVED_TOKENS}"
        )
    return budget


class RepoFailure(RuntimeError):
    """A single repo failed at a specific stage. Recorded; driver continues."""

    def __init__(self, repo: str, stage: str, detail: str):
        super().__init__(f"[{repo}] stage={stage}: {detail}")
        self.repo = repo
        self.stage = stage
        self.detail = detail


class RepoNoTrainingDocs(RuntimeError):
    """index_project completed successfully but produced no new trainable docs."""

    def __init__(self, repo: str, *, reason: str, detail: str):
        super().__init__(f"[{repo}] no training docs: {reason}: {detail}")
        self.repo = repo
        self.reason = reason
        self.detail = detail


_FOUND_CPP_RE = re.compile(r"\bFound\s+(\d+)\s+C/C\+\+ source files\b")
_FOUND_BUILD_RE = re.compile(r"\bFound\s+(\d+)\s+build/compilation files\b")
_TOTAL_ZERO_RE = re.compile(r"\bGenerated\s+0\s+total training documents\b")
_FUNCTION_DEDUP_RE = re.compile(
    r"\bFunction-level dedup:\s+kept_roots=(\d+)\s+"
    r"dropped_exact=(\d+)\s+dropped_near=(\d+)\b"
)
_BUILD_DOCS_RE = re.compile(
    r"\bBuild docs:\s+emitted=(\d+)\s+"
    r"source_chars_in=(\d+)\s+source_chars_out=(\d+)\s+"
    r"skipped_zero_length=(\d+)\s+"
    r"source_chunk_dedup=disabled_for_lossless_spans\b"
)


def _int_match(pattern: re.Pattern[str], text: str, default: int = 0) -> int:
    match = pattern.search(text)
    return int(match.group(1)) if match else default


def _classify_empty_index_project_log(text: str) -> str | None:
    """Classify a successful zero-doc index_project run.

    Empty output after a successful indexer run is not always a pipeline error:
    aliases can be fully exhausted by global dedup, and some source-cache entries
    simply contain no C/C++/build training signal.  Non-zero exit/stall paths
    still fail through run_checked before this classifier is consulted.
    """

    if not _TOTAL_ZERO_RE.search(text):
        return None

    cpp_files = _int_match(_FOUND_CPP_RE, text)
    build_files = _int_match(_FOUND_BUILD_RE, text)
    dedup = _FUNCTION_DEDUP_RE.search(text)
    build_docs = _BUILD_DOCS_RE.search(text)
    dropped_functions = 0
    if dedup is not None:
        _kept, dropped_exact, dropped_near = (int(dedup.group(i)) for i in (1, 2, 3))
        dropped_functions = dropped_exact + dropped_near
    if build_docs is not None:
        source_chars_in = int(build_docs.group(2))
        source_chars_out = int(build_docs.group(3))
        if source_chars_in != source_chars_out:
            return "domain_source_loss"

    if dropped_functions > 0:
        return "dedup_exhausted"
    if cpp_files == 0 and build_files == 0:
        return "no_trainable_source"
    return "no_training_documents"


def code_stage_id(repo: str) -> str:
    return f"code:{repo}"


def commit_stage_id(key: str) -> str:
    return f"commit:{key}"


def _safe_stage_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def filesystem_repo_key(repo: str) -> str:
    """Encode a logical repo name into one case-insensitive-safe path component."""
    if not isinstance(repo, str) or not repo:
        raise ValueError("repo must be a non-empty string")
    return "".join(
        chr(byte)
        if (
            ord("a") <= byte <= ord("z")
            or ord("0") <= byte <= ord("9")
            or byte in (ord("_"), ord("-"))
        )
        else f"%{byte:02x}"
        for byte in repo.encode("utf-8")
    )


def code_output_filename(repo: str) -> str:
    return f"{filesystem_repo_key(repo)}.parquet"


def commit_output_filename(repo: str, start_idx: int) -> str:
    return f"{filesystem_repo_key(repo)}_r{start_idx}.parquet"


def code_stage_db(work: Path, repo: str) -> Path:
    return work / f"{_safe_stage_name(repo)}.dedup_stage.sqlite"


def commit_stage_db(work: Path, key: str) -> Path:
    return work / f"{_safe_stage_name(key)}.dedup_stage.sqlite"


def is_code_worktree_repo(repo: str) -> bool:
    """Exclude bare Git object stores, but retain extracted source dumps."""
    return not repo.endswith(".bare")


def _dedup_store_cls():
    sys.path.insert(0, str(ROOT / "tools" / "clang_indexer"))
    from dedup_store import DedupStore
    return DedupStore


def dedup_promote_lock_path(dedup_db: Path) -> Path:
    return Path(str(dedup_db) + ".promote.lock")


@contextmanager
def dedup_promote_lock(dedup_db: Path):
    """Serialize parent-stage promotion before touching the global SQLite DB."""
    import fcntl

    lock_path = dedup_promote_lock_path(dedup_db)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    timeout_s = float(os.environ.get("CPPMEGA_DEDUP_PROMOTE_LOCK_TIMEOUT_SECONDS", "600"))
    deadline = time.monotonic() + timeout_s
    sleep_s = 0.05
    started = time.monotonic()
    with lock_path.open("a+b") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "DedupStore: global promote lock remained held after "
                        f"{timeout_s:.0f}s at {str(lock_path)!r}"
                    ) from exc
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 1.5, 2.0)
        wait_s = time.monotonic() - started
        try:
            yield wait_s
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def promote_dedup_stage(
    dedup_db: Path | None,
    stage_id: str | None,
    stage_db: Path | None = None,
) -> dict[str, float]:
    if dedup_db is None or stage_id is None:
        return {"promote_wait_s": 0.0, "promote_duration_s": 0.0}
    with dedup_promote_lock(dedup_db) as wait_s:
        started = time.monotonic()
        DedupStore = _dedup_store_cls()
        if stage_db is not None:
            DedupStore.promote_stage_from_db(str(dedup_db), str(stage_db), stage_id)
        else:
            DedupStore.promote_stage_in_db(str(dedup_db), stage_id)
        duration_s = time.monotonic() - started
    return {
        "promote_wait_s": round(wait_s, 6),
        "promote_duration_s": round(duration_s, 6),
    }


def promote_dedup_stages(
    dedup_db: Path | None,
    stages: Sequence[tuple[str, Path | None]],
) -> dict[str, float | int]:
    """Promote multiple dedup stages while holding the global writer lock once."""
    real_stages = [(sid, sdb) for sid, sdb in stages if dedup_db is not None and sid]
    if dedup_db is None or not real_stages:
        return {
            "promote_wait_s": 0.0,
            "promote_duration_s": 0.0,
            "promote_batch_size": 0,
        }
    with dedup_promote_lock(dedup_db) as wait_s:
        started = time.monotonic()
        DedupStore = _dedup_store_cls()
        for stage_id, stage_db in real_stages:
            if stage_db is not None:
                DedupStore.promote_stage_from_db(str(dedup_db), str(stage_db), stage_id)
            else:
                DedupStore.promote_stage_in_db(str(dedup_db), stage_id)
        duration_s = time.monotonic() - started
    return {
        "promote_wait_s": round(wait_s, 6),
        "promote_duration_s": round(duration_s, 6),
        "promote_batch_size": len(real_stages),
    }


def discard_dedup_stage(
    dedup_db: Path | None,
    stage_id: str | None,
    stage_db: Path | None = None,
) -> None:
    if dedup_db is None or stage_id is None:
        return
    DedupStore = _dedup_store_cls()
    DedupStore.discard_stage(
        str(dedup_db),
        stage_id,
        stage_db_path=str(stage_db) if stage_db is not None else None,
    )


# --------------------------------------------------------------------------- #
# Subprocess helper -- fail loud, no swallowed errors.
# --------------------------------------------------------------------------- #
def _subprocess_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    parts = [str(ROOT)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


_HEARTBEAT_PROGRESS_RE = re.compile(
    r"^\s*(?P<label>[^:\r\n]*\bheartbeat):\s*(?P<state>.*)\s*$",
    re.IGNORECASE,
)


def _consume_log_progress(
    log_path: Path,
    offset: int,
    heartbeat_states: dict[str, str],
) -> tuple[int, bool]:
    """Read appended log bytes and ignore duplicate semantic heartbeats."""

    size = log_path.stat().st_size
    if size < offset:
        offset = 0
        heartbeat_states.clear()
    if size == offset:
        return offset, False

    with log_path.open("rb") as log_reader:
        log_reader.seek(offset)
        appended = log_reader.read(size - offset)

    progressed = False
    for raw_line in appended.splitlines():
        if not raw_line.strip():
            continue
        line = raw_line.decode("utf-8", errors="replace")
        heartbeat = _HEARTBEAT_PROGRESS_RE.fullmatch(line)
        if heartbeat is None:
            progressed = True
            continue
        label = heartbeat.group("label").strip().casefold()
        state = heartbeat.group("state").strip()
        if heartbeat_states.get(label) != state:
            heartbeat_states[label] = state
            progressed = True
    return size, progressed


def run_checked(
    repo: str,
    stage: str,
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    log_path: Path | None = None,
    timeout: int | None = None,
    stall_timeout: int | None = None,
) -> None:
    """Run a command; RAISE RepoFailure on any non-zero exit. No fallback."""
    printable = " ".join(str(c) for c in cmd)
    print(f"  [{repo}] {stage}: {printable}", file=sys.stderr, flush=True)
    log_fh = open(log_path, "wb") if log_path else None
    stdout_data: bytes | None = None
    proc: subprocess.Popen[bytes] | None = None

    def terminate_process(reason: str) -> None:
        if proc is None:
            return
        try:
            if timeout or stall_timeout:
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if timeout or stall_timeout:
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
            proc.wait()
        print(f"  [{repo}] {stage}: killed after {reason}", file=sys.stderr, flush=True)

    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            env=_subprocess_env(),
            stdout=log_fh if log_fh else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=bool(timeout or stall_timeout),
        )
        if stall_timeout and log_path is not None:
            deadline = time.monotonic() + timeout if timeout and timeout > 0 else None
            last_activity = time.monotonic()
            log_offset = 0
            heartbeat_states: dict[str, str] = {}
            while proc.poll() is None:
                now = time.monotonic()
                if deadline is not None and now > deadline:
                    terminate_process(f"timeout {timeout}s")
                    raise RepoFailure(
                        repo,
                        stage,
                        f"timed out after {timeout}s",
                    )
                if log_path.exists():
                    log_offset, progressed = _consume_log_progress(
                        log_path,
                        log_offset,
                        heartbeat_states,
                    )
                    if progressed:
                        last_activity = now
                if now - last_activity > stall_timeout:
                    terminate_process(f"no log progress for {stall_timeout}s")
                    raise RepoFailure(
                        repo,
                        stage,
                        f"stalled after {stall_timeout}s without log progress",
                    )
                time.sleep(1.0)
            stdout_data, _ = proc.communicate(timeout=1)
        else:
            stdout_data, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if proc is not None:
            terminate_process(f"timeout {timeout}s")
        raise RepoFailure(repo, stage, f"timed out after {timeout}s: {exc}") from exc
    finally:
        if log_fh:
            log_fh.close()
    if proc is None:
        raise RepoFailure(repo, stage, "subprocess did not start")
    if proc.returncode != 0:
        tail = ""
        if log_path and log_path.exists():
            data = log_path.read_bytes()[-4000:]
            tail = data.decode("utf-8", errors="replace")
        elif stdout_data:
            tail = stdout_data.decode("utf-8", errors="replace")[-4000:]
        raise RepoFailure(
            repo, stage, f"exit code {proc.returncode}\n--- last output ---\n{tail}"
        )


# --------------------------------------------------------------------------- #
# Manifest (resume) helpers.
# --------------------------------------------------------------------------- #
@dataclass
class Manifest:
    path: Path
    done: dict = field(default_factory=dict)
    failed: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if path.exists():
            blob = json.loads(path.read_text())
            return cls(path=path, done=blob.get("done", {}), failed=blob.get("failed", {}))
        return cls(path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"done": self.done, "failed": self.failed},
                indent=2,
                sort_keys=True,
            )
        )
        tmp.replace(self.path)

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_started(self, key: str) -> None:
        """Invalidate stale terminal state before starting a new attempt."""
        changed = self.done.pop(key, None) is not None
        changed = self.failed.pop(key, None) is not None or changed
        if changed:
            self.save()

    def mark_started_prefix(self, prefix: str) -> None:
        """Invalidate all stale range states for a logical parent item."""
        done_keys = [key for key in self.done if key.startswith(prefix)]
        failed_keys = [key for key in self.failed if key.startswith(prefix)]
        for key in done_keys:
            self.done.pop(key)
        for key in failed_keys:
            self.failed.pop(key)
        if done_keys or failed_keys:
            self.save()

    def mark_done(self, key: str, info: dict) -> None:
        self.done[key] = info
        self.failed.pop(key, None)
        self.save()

    def mark_failed(self, key: str, stage: str, detail: str) -> None:
        self.done.pop(key, None)
        self.failed[key] = {
            "stage": stage,
            "detail": detail[:2000],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.save()


# --------------------------------------------------------------------------- #
# Tarball streaming (TRUE single-pass, one repo at a time -- no enumeration,
# no /tmp bulk extract).
# --------------------------------------------------------------------------- #
def _is_excluded(within: str) -> bool:
    """True if any path component of the in-repo path is an excluded dir."""
    return any(part in EXCLUDE_PARTS for part in within.split("/"))


SOURCE_CACHE_SENTINEL = ".cppmega_source_cache_complete.json"
TAR_COLLIDING_FILE_BASENAME = "__cppmega_tar_file__"


def _tar_collision_target(directory: Path) -> Path:
    return directory / (TAR_COLLIDING_FILE_BASENAME + directory.suffix)


def _ensure_tar_member_directory(
    directory: Path,
    *,
    repo: str,
    member_name: str,
) -> None:
    """Create a member parent, relocating an earlier colliding regular file."""
    if directory.is_dir():
        return
    if os.path.lexists(directory):
        if not directory.is_file() or directory.is_symlink():
            raise RepoFailure(
                repo,
                "stream",
                f"tar member {member_name!r} has unsupported non-directory parent "
                f"{directory}; refusing to drop source content",
            )
        try:
            staging_dir = Path(
                tempfile.mkdtemp(
                    prefix=".cppmega-tar-relocate-",
                    dir=directory.parent,
                )
            )
        except OSError as exc:
            raise RepoFailure(
                repo,
                "stream",
                f"cannot stage file/directory tar collision at {directory}: {exc}",
            ) from exc
        staged_file = staging_dir / _tar_collision_target(directory).name
        moved = False
        rollback_error: OSError | None = None
        try:
            directory.rename(staged_file)
            moved = True
            staging_dir.rename(directory)
        except OSError as exc:
            if moved and os.path.lexists(staged_file) and not os.path.lexists(directory):
                try:
                    staged_file.rename(directory)
                except OSError as rollback_exc:
                    rollback_error = rollback_exc
            try:
                staging_dir.rmdir()
            except OSError:
                pass
            detail = (
                f"cannot preserve file/directory tar collision at {directory}: {exc}"
            )
            if rollback_error is not None:
                detail += (
                    f"; rollback failed ({rollback_error}); source remains at "
                    f"{staged_file}"
                )
            raise RepoFailure(repo, "stream", detail) from exc
        print(
            f"[{repo}] preserving prior file/directory tar collision "
            f"at {directory} as {_tar_collision_target(directory)}",
            file=sys.stderr,
        )
        return
    _ensure_tar_member_directory(
        directory.parent,
        repo=repo,
        member_name=member_name,
    )
    directory.mkdir()


def _source_cache_repo_dir(source_cache_dir: Path, repo: str) -> Path:
    return source_cache_dir / filesystem_repo_key(repo)


def _source_cache_staging_dir(source_cache_dir: Path, repo: str) -> Path:
    staging = source_cache_dir / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging / f"{filesystem_repo_key(repo)}.{os.getpid()}.{time.time_ns()}"


def _source_cache_complete(repo_dir: Path) -> bool:
    return (repo_dir / SOURCE_CACHE_SENTINEL).exists()


def _source_cache_repo_name(repo_dir: Path) -> str:
    sentinel = repo_dir / SOURCE_CACHE_SENTINEL
    try:
        document = json.loads(sentinel.read_text(encoding="utf-8"))
        repo = document["repo"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RepoFailure(
            repo_dir.name,
            "source_cache",
            f"invalid source-cache sentinel {sentinel}: {exc}",
        ) from exc
    if not isinstance(repo, str) or not repo:
        raise RepoFailure(
            repo_dir.name,
            "source_cache",
            f"invalid repo identity in source-cache sentinel {sentinel}",
        )
    if repo_dir.name not in {repo, filesystem_repo_key(repo)}:
        raise RepoFailure(
            repo,
            "source_cache",
            f"source-cache path {repo_dir} does not match sentinel repo {repo!r}",
        )
    return repo


def _mark_source_cache_complete(repo_dir: Path, repo: str) -> None:
    sentinel = repo_dir / SOURCE_CACHE_SENTINEL
    tmp = sentinel.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "repo": repo,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": str(TARBALL),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(sentinel)


def _copy_tar_member_file(src, target: Path, *, repo: str, member_name: str) -> bool:
    """Copy one tar member without silently dropping path-type collisions."""
    if target.exists() and target.is_dir():
        collision_target = _tar_collision_target(target)
        if os.path.lexists(collision_target):
            raise RepoFailure(
                repo,
                "stream",
                f"tar member {member_name!r} collides with an existing directory "
                f"at {target} and reserved file {collision_target} already exists; "
                "refusing to overwrite source content",
            )
        print(
            f"[{repo}] preserving file/directory tar collision "
            f"{member_name!r} as {collision_target}",
            file=sys.stderr,
        )
        target = collision_target
    _ensure_tar_member_directory(
        target.parent,
        repo=repo,
        member_name=member_name,
    )
    reserved_name = TAR_COLLIDING_FILE_BASENAME + target.parent.suffix
    if target.name == reserved_name and os.path.lexists(target):
        raise RepoFailure(
            repo,
            "stream",
            f"tar member {member_name!r} collides with reserved tar path "
            f"{target}; refusing to overwrite source content",
        )
    with open(target, "wb") as out:
        shutil.copyfileobj(src, out)
    return True


def _iter_complete_source_cache(
    source_cache_dir: Path,
    should_process: Callable[[str], bool],
) -> dict[str, Path]:
    yielded: dict[str, Path] = {}
    if not source_cache_dir.exists():
        return yielded
    for repo_dir in sorted(p for p in source_cache_dir.iterdir() if p.is_dir()):
        if repo_dir.name == ".staging":
            continue
        if not _source_cache_complete(repo_dir):
            continue
        repo = _source_cache_repo_name(repo_dir)
        if not should_process(repo):
            continue
        yielded[repo] = repo_dir
    return yielded


def stream_repo_subtrees(
    work_root: Path,
    should_process,
    *,
    source_cache_dir: Path | None = None,
    source_cache_only: bool = False,
):
    """Yield (repo, repo_dir) for every cpp_all/<repo>/ subtree, in ONE pass.

    A single `zstd -dc --long=31 TARBALL` is piped into a Python
    `tarfile.open(fileobj=..., mode="r|")` stream and read sequentially. Members
    arrive in archive order and each repo's subtree is contiguous (the archive
    was built by a directory walk), so a change in the `<repo>` path component
    marks the end of the previous repo -- which is then yielded, fully extracted
    under work_root/<repo>/_src (the cpp_all/<repo>/ prefix stripped). Repos for
    which should_process(repo) is False are DRAINED from the stream without
    touching disk (resume skip). Only ONE repo is ever on disk at a time.

    RULE #1: a non-contiguous repo subtree (a repo name reappearing after it was
    closed) RAISES -- it would mean a partial re-extract, never silently
    tolerated. There is no separate header/enumeration pass and no /tmp bulk.
    """
    import tarfile

    cached_yielded: dict[str, Path] = {}
    if source_cache_dir is not None:
        source_cache_dir.mkdir(parents=True, exist_ok=True)
        cached_yielded = _iter_complete_source_cache(source_cache_dir, should_process)
        for repo, repo_dir in sorted(cached_yielded.items()):
            yield repo, repo_dir
        if source_cache_only:
            return

    if not TARBALL.exists():
        raise FileNotFoundError(f"source tarball missing: {TARBALL}")
    zstd = subprocess.Popen(
        ["zstd", "-dc", "--long=31", str(TARBALL)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert zstd.stdout is not None
    tar = tarfile.open(fileobj=zstd.stdout, mode="r|")
    prefix = TAR_MEMBER_ROOT + "/"
    finalized: set[str] = set()
    cur_repo: str | None = None
    cur_dir: Path | None = None
    cur_final_dir: Path | None = None
    active = False
    try:
        for member in iter_tar_members_without_cache(tar):
            name = member.name
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            slash = rest.find("/")
            if slash <= 0:
                continue  # a bare entry directly under cpp_all/ -- not a repo
            repo = rest[:slash]
            within = rest[slash + 1:]
            if repo != cur_repo:
                if cur_repo is not None and active:
                    assert cur_dir is not None
                    yield_dir = cur_dir
                    if cur_final_dir is not None:
                        _mark_source_cache_complete(cur_dir, cur_repo)
                        if cur_final_dir.exists() and _source_cache_complete(cur_final_dir):
                            shutil.rmtree(cur_dir, ignore_errors=True)
                            yield_dir = cur_final_dir
                        else:
                            if cur_final_dir.exists():
                                shutil.rmtree(cur_final_dir)
                            cur_dir.rename(cur_final_dir)
                            yield_dir = cur_final_dir
                    yield cur_repo, yield_dir
                    finalized.add(cur_repo)
                if repo in finalized:
                    raise RepoFailure(
                        repo, "stream",
                        "non-contiguous repo subtree in tarball "
                        "(would re-extract an already-closed repo)",
                    )
                cur_repo = repo
                active = repo not in cached_yielded and should_process(repo)
                cur_final_dir = None
                if active and source_cache_dir is not None:
                    cur_final_dir = _source_cache_repo_dir(source_cache_dir, repo)
                    if _source_cache_complete(cur_final_dir):
                        active = False
                        cached_yielded.add(repo)
                        cur_dir = None
                    else:
                        cur_dir = _source_cache_staging_dir(source_cache_dir, repo)
                else:
                    cur_dir = work_root / filesystem_repo_key(repo) / "_src"
                if active:
                    assert cur_dir is not None
                    cur_dir.mkdir(parents=True, exist_ok=True)
            if not active or not within:
                continue
            if not member.isfile() or _is_excluded(within):
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            target = cur_dir / within
            _copy_tar_member_file(src, target, repo=repo, member_name=name)
        if cur_repo is not None and active:
            assert cur_dir is not None
            yield_dir = cur_dir
            if cur_final_dir is not None:
                _mark_source_cache_complete(cur_dir, cur_repo)
                if cur_final_dir.exists() and _source_cache_complete(cur_final_dir):
                    shutil.rmtree(cur_dir, ignore_errors=True)
                    yield_dir = cur_final_dir
                else:
                    if cur_final_dir.exists():
                        shutil.rmtree(cur_final_dir)
                    cur_dir.rename(cur_final_dir)
                    yield_dir = cur_final_dir
            yield cur_repo, yield_dir
            finalized.add(cur_repo)
    finally:
        try:
            tar.close()
        except SymbolIdentityError:
            raise
        except Exception:
            pass
        if zstd.poll() is None:
            zstd.send_signal(signal.SIGTERM)
        try:
            zstd.wait(timeout=10)
        except subprocess.TimeoutExpired:
            zstd.kill()


def stream_repo_dirs(source_roots: Sequence[Path], should_process):
    """Yield already-extracted repo directories without opening the tarball."""
    seen: set[str] = set()
    for root in source_roots:
        if not root.exists():
            raise FileNotFoundError(f"source dir root missing: {root}")
        for entry in sorted(p for p in root.iterdir() if p.is_dir()):
            repo = entry.name
            if repo in seen or not is_code_worktree_repo(repo) or not should_process(repo):
                continue
            repo_dir = entry / "_src" if (entry / "_src").is_dir() else entry
            seen.add(repo)
            yield repo, repo_dir


def populate_source_cache(
    work_root: Path,
    should_process,
    source_cache_dir: Path,
    *,
    max_repos: int | None = None,
    on_repo_ready: Callable[[str, Path, int], None] | None = None,
) -> dict:
    """Materialize source-cache repos without running the tokenization pipeline.

    This is the explicit "cold source-store build" phase. It may read the
    monolithic source tar once, but callers can then run the hot code conveyor
    with --source-cache-only and avoid any sequential tar drain in production.
    """
    if source_cache_dir is None:
        raise ValueError("source_cache_dir is required")
    repos: list[dict[str, str]] = []
    gen = stream_repo_subtrees(
        work_root,
        should_process,
        source_cache_dir=source_cache_dir,
        source_cache_only=False,
    )
    try:
        for repo, repo_dir in gen:
            repos.append({"repo": repo, "path": str(repo_dir)})
            if on_repo_ready is not None:
                on_repo_ready(repo, repo_dir, len(repos))
            if max_repos is not None and len(repos) >= max_repos:
                break
    finally:
        if hasattr(gen, "close"):
            gen.close()
    return {
        "source_cache_dir": str(source_cache_dir),
        "repos": repos,
        "repo_count": len(repos),
    }


# --------------------------------------------------------------------------- #
# Per-repo pipeline stages.
# --------------------------------------------------------------------------- #
def stage_index_source(
    repo: str,
    project_id: str,
    repo_dir: Path,
    work: Path,
    dedup_db: Path | None = None,
    dedup_near: bool = True,
    dedup_stage_id: str | None = None,
    dedup_stage_db: Path | None = None,
    global_symbol_index: Path | None = None,
    memory_limit_gb: float = 10.0,
    parse_workers: int = 2,
    index_timeout_s: int | None = None,
    index_stall_timeout_s: int | None = None,
    source_quarantine_manifest: Path | None = None,
) -> Path:
    """index_project.py --enriched -> <repo>.enriched.jsonl.gz.

    Passes --tokenizer-path so index_project runs FUNCTION-LEVEL tokenized-hash
    dedup before grouping; --dedup-db makes that dedup GLOBAL + resumable +
    cross-stream (shared with commits) when given.

    --global-symbol-index, when given, enables bounded cross-repo base-lib symbol
    linking inside index_project (depth-1 pulls tagged crosslib:<repo>). None ->
    behavior unchanged.
    """
    project_id = require_project_identity(
        project_id, source=f"stage_index_source({repo})"
    )
    enriched = work / f"{repo}.enriched.jsonl.gz"
    cmd = [
        VENV_PYTHON, INDEX_PROJECT,
        "--project-dir", repo_dir,
        "--project-id", project_id,
        "--output", enriched,
        "--enriched",
        "--max-tokens", str(LOSSLESS_INDEX_MAX_TOKENS),
        "--exclude-dirs", SOURCE_EXTRA_EXCLUDE,
        "--tokenizer-path", TOKENIZER_PATH,
        "--memory-limit-gb", str(memory_limit_gb),
        "--parse-workers", str(max(1, int(parse_workers))),
    ]
    if dedup_db is not None:
        cmd += ["--dedup-db", str(dedup_db)]
        if dedup_stage_id is not None:
            cmd += ["--dedup-stage-id", dedup_stage_id]
            if dedup_stage_db is not None:
                cmd += ["--dedup-stage-db", str(dedup_stage_db)]
    if not dedup_near:
        cmd += ["--no-near-dedup"]
    if global_symbol_index is not None:
        cmd += ["--global-symbol-index", str(global_symbol_index)]
    quarantine_receipt = source_quarantine_receipt_path(work, repo)
    if source_quarantine_manifest is not None:
        cmd += [
            "--source-quarantine-manifest",
            str(source_quarantine_manifest),
            "--source-quarantine-receipt",
            str(quarantine_receipt),
        ]
    log_path = work / f"{repo}.index.log"
    run_checked(
        repo,
        "index_project",
        cmd,
        log_path=log_path,
        timeout=index_timeout_s if index_timeout_s and index_timeout_s > 0 else None,
        stall_timeout=(
            index_stall_timeout_s
            if index_stall_timeout_s and index_stall_timeout_s > 0
            else None
        ),
    )
    if not enriched.exists() or enriched.stat().st_size == 0:
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        reason = _classify_empty_index_project_log(log_text)
        if reason is not None:
            raise RepoNoTrainingDocs(
                repo,
                reason=reason,
                detail=f"empty enriched jsonl after successful index_project: {enriched}",
            )
        raise RepoFailure(repo, "index_project", f"empty enriched jsonl: {enriched}")
    if source_quarantine_manifest is not None and not quarantine_receipt.is_file():
        raise RepoFailure(
            repo,
            "index_project",
            f"missing source quarantine receipt: {quarantine_receipt}",
        )
    return enriched


def source_quarantine_receipt_path(work: Path, repo: str) -> Path:
    return work / f"{repo}.source_quarantine_receipt.json"


def read_source_quarantine_receipt(work: Path, repo: str) -> dict[str, object]:
    path = source_quarantine_receipt_path(work, repo)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepoFailure(
            repo,
            "source_quarantine_receipt",
            f"cannot read {path}: {exc}",
        ) from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema")
        not in {
            "cppmega.source_quarantine_receipt_v1",
            "cppmega.source_quarantine_receipt_v2",
        }
    ):
        raise RepoFailure(
            repo,
            "source_quarantine_receipt",
            f"unexpected receipt schema in {path}: {receipt!r}",
        )
    return receipt


def stage_index_commits(repo: str, commit_inputs: Sequence[Path], work: Path,
                        repo_root: Path | None, repo_dir: Path | None,
                        dedup_db: Path | None = None,
                        dedup_near: bool = True,
                        dedup_stage_id: str | None = None,
                        dedup_stage_db: Path | None = None,
                        pr_store: Path | None = None,
                        repo_list: Path | None = None,
                        memory_limit_gb: float = 10.0,
                        analysis_cache_entries: int = 128,
                        allow_empty: bool = False,
                        *,
                        pr_scan_id: str | None = None,
                        project_id: str) -> Path | None:
    """process_commits.py -> <repo>.enriched.jsonl (commit edit-signal docs).

    A commit is an ATOMIC change-unit: process_commits dedups whole commit DOCS
    by the tokenized hash of the doc (drops identical commits, e.g. cherry-picks)
    while keeping route-by-fit. --dedup-db makes that dedup share the SAME global
    store as the code stream.
    """
    enriched = work / f"{repo}.enriched.jsonl"
    cmd = [
        VENV_PYTHON, PROCESS_COMMITS,
        "--inputs", *[str(p) for p in commit_inputs],
        "--output", enriched,
        "--max-tokens", str(LOSSLESS_INDEX_MAX_TOKENS),
        "--tokenizer-path", TOKENIZER_PATH,
        "--format", "both",
        "--memory-limit-gb", str(memory_limit_gb),
        "--analysis-cache-entries", str(max(0, int(analysis_cache_entries))),
        "--project-id", require_project_identity(
            project_id,
            source=f"stage_index_commits({repo})",
        ),
    ]
    if dedup_db is not None:
        cmd += ["--dedup-db", str(dedup_db)]
        if dedup_stage_id is not None:
            cmd += ["--dedup-stage-id", dedup_stage_id]
            if dedup_stage_db is not None:
                cmd += ["--dedup-stage-db", str(dedup_stage_db)]
    if not dedup_near:
        cmd += ["--no-near-dedup"]
    if repo_root is not None:
        cmd += ["--repo-root", str(repo_root)]
    if repo_dir is not None:
        cmd += ["--repo-dir", str(repo_dir)]
    if pr_store is not None:
        cmd += ["--pr-store", str(pr_store)]
    if pr_scan_id is not None:
        if pr_store is None:
            raise ValueError("pr_scan_id requires pr_store")
        cmd += ["--pr-scan-id", pr_scan_id]
    if repo_list is not None:
        cmd += ["--repo-list", str(repo_list)]
    run_checked(repo, "process_commits", cmd, log_path=work / f"{repo}.commits.log")
    if not enriched.exists():
        raise RepoFailure(repo, "process_commits", f"empty enriched jsonl: {enriched}")
    if enriched.stat().st_size == 0:
        if allow_empty:
            return None
        raise RepoFailure(repo, "process_commits", f"empty enriched jsonl: {enriched}")
    return enriched


def stage_materialize(
    repo: str,
    enriched: Path,
    work: Path,
    memory_limit_gb: float = 10.0,
    *,
    project_id: str | None = None,
    max_tokens: int = TOKENIZE_BUDGET,
    fixed_shape_max_tokens: int | None = None,
) -> Path:
    """clang_enriched_to_parquet.py -> tokenized enriched parquet (single file).

    ``project_id`` is keyword-only so adding canonical identity cannot shift the
    long-standing ``(repo, enriched, work, memory_limit_gb)`` stage ABI. Sources
    that omit it must carry canonical repo identity in every input row; the
    materializer validates that contract fail-closed.
    """
    if max_tokens <= 0:
        raise RepoFailure(repo, "materialize", f"max_tokens must be positive: {max_tokens}")
    if fixed_shape_max_tokens is not None and fixed_shape_max_tokens <= 0:
        raise RepoFailure(
            repo,
            "materialize",
            f"fixed_shape_max_tokens must be positive: {fixed_shape_max_tokens}",
        )
    tok = work / f"{repo}.tok.parquet"
    stats_path = materialize_stats_path(tok)
    cmd = [
        VENV_PYTHON, MATERIALIZER,
        "--input-file", enriched,
        "--output-file", tok,
        "--tokenizer-path", TOKENIZER_PATH,
        "--materialize-tokenized-enriched",
        "--overflow-policy", "split",
        "--size", str(max_tokens),
        "--stats-file", stats_path,
        "--memory-limit-gb", str(memory_limit_gb),
    ]
    if project_id is not None:
        project_id = require_project_identity(
            project_id, source=f"stage_materialize({repo})"
        )
        cmd += ["--default-repo", project_id]
    if fixed_shape_max_tokens is not None:
        cmd += ["--fixed-shape-max-tokens", str(fixed_shape_max_tokens)]
    # A failed/retried materialization must never validate a newly replaced
    # parquet against a receipt left by an earlier attempt.
    stats_path.unlink(missing_ok=True)
    run_checked(
        repo,
        "materialize",
        cmd,
        log_path=work / f"{repo}.materialize.log",
    )
    if not tok.exists() or tok.stat().st_size == 0:
        raise RepoFailure(repo, "materialize", f"empty tokenized parquet: {tok}")
    try:
        read_materialize_stats(
            tok,
            fixed_shape_max_tokens=fixed_shape_max_tokens,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RepoFailure(
            repo,
            "materialize_stats",
            f"invalid lossless materialization receipt {stats_path}: {exc}",
        ) from exc
    return tok


def materialize_stats_path(tok: Path) -> Path:
    return Path(f"{tok}.materialize_stats.json")


def read_materialize_stats(
    tok: Path,
    *,
    fixed_shape_max_tokens: int | None = None,
) -> dict[str, object]:
    """Load and validate the exact split receipt paired with tokenized parquet."""

    path = materialize_stats_path(tok)
    if not path.is_file():
        raise ValueError(f"missing materialization stats receipt: {path}")
    stats = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stats, dict):
        raise ValueError("materialization stats must be a JSON object")
    if stats.get("schema") != MATERIALIZE_SPLIT_STATS_SCHEMA:
        raise ValueError(
            f"unsupported materialization stats schema: {stats.get('schema')!r}"
        )
    if stats.get("overflow_policy") != "split":
        raise ValueError(
            "production materialization receipt must use overflow_policy='split'"
        )

    integer_fields = (
        "max_tokens",
        "docs_in",
        "docs_out",
        "source_docs_emitted",
        "split_input_docs",
        "split_output_docs",
        "dropped_input_docs",
        "emitted_valid_tokens",
        "split_emitted_tokens",
        "emitted_chars",
        "split_emitted_chars",
        "max_emitted_tokens",
        "max_emitted_chars",
        "materialized_rows",
        "materialized_output_tokens",
        "max_materialized_tokens",
        "output_file_size",
        "cross_piece_call_edges",
        "cross_piece_type_edges",
        "cross_piece_domain_edges",
        "cross_piece_build_edges",
        "cross_piece_shell_edges",
        "cross_piece_diagnostic_edges",
        "cross_piece_cross_domain_edges",
    )
    normalized: dict[str, object] = {
        "schema": stats["schema"],
        "overflow_policy": stats["overflow_policy"],
    }
    receipt_fixed_shape_max_tokens = stats.get("fixed_shape_max_tokens")
    if receipt_fixed_shape_max_tokens is not None:
        receipt_fixed_shape_max_tokens = int(receipt_fixed_shape_max_tokens)
        if receipt_fixed_shape_max_tokens <= 0:
            raise ValueError(
                "fixed_shape_max_tokens must be positive, got "
                f"{receipt_fixed_shape_max_tokens}"
            )
        normalized["fixed_shape_max_tokens"] = receipt_fixed_shape_max_tokens
    for counter_name in integer_fields:
        value = int(stats[counter_name])
        if value < 0:
            raise ValueError(f"{counter_name} must be non-negative, got {value}")
        normalized[counter_name] = value
    output_file_sha256 = stats.get("output_file_sha256")
    if (
        not isinstance(output_file_sha256, str)
        or len(output_file_sha256) != 64
        or any(char not in "0123456789abcdef" for char in output_file_sha256)
    ):
        raise ValueError("output_file_sha256 must be a lowercase SHA-256")
    normalized["output_file_sha256"] = output_file_sha256
    actual_size = tok.stat().st_size
    if int(normalized["output_file_size"]) != actual_size:
        raise ValueError(
            "materialized parquet size does not match its receipt: "
            f"{actual_size} != {normalized['output_file_size']}"
        )
    digest = hashlib.sha256()
    with tok.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != output_file_sha256:
        raise ValueError(
            "materialized parquet hash does not match its receipt: "
            f"{actual_sha256} != {output_file_sha256}"
        )

    docs_in = int(normalized["docs_in"])
    docs_out = int(normalized["docs_out"])
    emitted = int(normalized["source_docs_emitted"])
    dropped = int(normalized["dropped_input_docs"])
    if docs_in != emitted + dropped:
        raise ValueError(
            "docs_in must equal source_docs_emitted + dropped_input_docs: "
            f"{docs_in} != {emitted} + {dropped}"
        )
    if dropped:
        raise ValueError(f"lossless split dropped {dropped} input document(s)")
    if docs_out < emitted:
        raise ValueError(f"docs_out {docs_out} is smaller than emitted inputs {emitted}")
    if int(normalized["materialized_rows"]) != docs_out:
        raise ValueError(
            "materialized_rows must equal docs_out: "
            f"{normalized['materialized_rows']} != {docs_out}"
        )
    if int(normalized["max_emitted_tokens"]) > int(normalized["max_tokens"]):
        raise ValueError(
            "max_emitted_tokens exceeds the requested raw split budget: "
            f"{normalized['max_emitted_tokens']} > {normalized['max_tokens']}"
        )
    if (
        fixed_shape_max_tokens is not None
        and int(normalized["max_materialized_tokens"]) > fixed_shape_max_tokens
    ):
        raise ValueError(
            "materialized token row exceeds the largest fixed bucket: "
            f"{normalized['max_materialized_tokens']} > {fixed_shape_max_tokens}"
        )
    if (
        receipt_fixed_shape_max_tokens is not None
        and int(normalized["max_materialized_tokens"])
        > receipt_fixed_shape_max_tokens
    ):
        raise ValueError(
            "materialized token row exceeds its receipted fixed-shape limit: "
            f"{normalized['max_materialized_tokens']} "
            f"> {receipt_fixed_shape_max_tokens}"
        )
    return normalized


_MATERIALIZE_SUM_FIELDS = (
    "docs_in",
    "docs_out",
    "source_docs_emitted",
    "split_input_docs",
    "split_output_docs",
    "dropped_input_docs",
    "emitted_valid_tokens",
    "split_emitted_tokens",
    "emitted_chars",
    "split_emitted_chars",
    "materialized_rows",
    "materialized_output_tokens",
    "output_file_size",
    "cross_piece_call_edges",
    "cross_piece_type_edges",
    "cross_piece_domain_edges",
    "cross_piece_build_edges",
    "cross_piece_shell_edges",
    "cross_piece_diagnostic_edges",
    "cross_piece_cross_domain_edges",
)
_MATERIALIZE_MAX_FIELDS = (
    "max_tokens",
    "max_emitted_tokens",
    "max_emitted_chars",
    "max_materialized_tokens",
)


def summarize_materialize_stats(entries: Sequence[dict]) -> dict[str, int]:
    """Aggregate durable per-unit split receipts; old entries are counted absent."""

    totals = {
        "receipts": 0,
        **{field: 0 for field in _MATERIALIZE_SUM_FIELDS},
        **{field: 0 for field in _MATERIALIZE_MAX_FIELDS},
    }
    for info in entries:
        stats = info.get("materialize_stats")
        if not isinstance(stats, dict):
            continue
        if stats.get("schema") != MATERIALIZE_SPLIT_STATS_SCHEMA:
            raise ValueError(
                f"unexpected materialize_stats schema: {stats.get('schema')!r}"
            )
        totals["receipts"] += 1
        for counter_name in _MATERIALIZE_SUM_FIELDS:
            totals[counter_name] += int(stats[counter_name])
        for counter_name in _MATERIALIZE_MAX_FIELDS:
            totals[counter_name] = max(
                totals[counter_name], int(stats[counter_name])
            )
    return totals


def stage_pack(repo: str, tok: Path, target_length: int, work: Path) -> Path:
    """pack_enriched_rows.py -> packed parquet for one target length."""
    packed = work / f"{repo}.packed.{target_length}.parquet"
    run_checked(
        repo,
        f"pack_{target_length}",
        [
            VENV_PYTHON, PACKER,
            "--input", tok,
            "--output", packed,
            "--target-length", str(target_length),
            "--strategy", "best_fit",
        ],
        log_path=work / f"{repo}.pack.{target_length}.log",
    )
    if not packed.exists() or packed.stat().st_size == 0:
        raise RepoFailure(repo, f"pack_{target_length}", f"empty packed parquet: {packed}")
    return packed


def _replace_publication_path(source: Path, destination: Path) -> None:
    os.replace(source, destination)


@contextmanager
def _publication_lock(output_root: Path):
    import fcntl

    lock_path = output_root / ".publish.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _PUBLICATION_LOCK, lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def publish_bucket_outputs_atomically(
    publication_key: str,
    packed_by_length: dict[int, Path],
    *,
    output_root: Path,
    filename: str,
    prepare_staged: Callable[[Path], None] | None = None,
    stats_reader: Callable[[Path, int], dict] | None = None,
    remove_lengths: Sequence[int] = (),
) -> dict[str, dict]:
    """Failure-atomically publish one logical item across all routed buckets."""
    if not packed_by_length:
        raise ValueError(f"[{publication_key}] no bucket outputs to publish")
    if Path(filename).name != filename:
        raise ValueError(f"[{publication_key}] invalid output filename: {filename!r}")

    read_stats = stats_reader or _parquet_stats
    transaction_id = f"{os.getpid()}.{time.time_ns()}"
    transaction_root = output_root / ".transactions" / transaction_id
    transaction_root.mkdir(parents=True, exist_ok=False)
    destinations: dict[int, Path] = {}
    staged: dict[int, Path] = {}
    stats: dict[str, dict] = {}

    try:
        publication_lengths = sorted(
            set(packed_by_length) | {int(length) for length in remove_lengths}
        )
        for target_length in publication_lengths:
            out_dir = output_root / str(target_length)
            out_dir.mkdir(parents=True, exist_ok=True)
            destinations[target_length] = out_dir / filename
        for target_length, packed in sorted(packed_by_length.items()):
            destination = destinations[target_length]
            staged_path = transaction_root / "staged" / str(target_length) / filename
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged[target_length] = staged_path
            shutil.copyfile(packed, staged_path)
            if prepare_staged is not None:
                prepare_staged(staged_path)
            stats[str(target_length)] = read_stats(staged_path, target_length)
    except BaseException:
        shutil.rmtree(transaction_root, ignore_errors=True)
        raise

    backups: dict[int, Path] = {}
    published: list[int] = []
    rollback_incomplete = False
    try:
        with _publication_lock(output_root):
            try:
                for target_length, destination in sorted(destinations.items()):
                    if not destination.exists():
                        continue
                    backup = (
                        transaction_root
                        / "backups"
                        / str(target_length)
                        / filename
                    )
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    _replace_publication_path(destination, backup)
                    backups[target_length] = backup

                for target_length, staged_path in sorted(staged.items()):
                    destination = destinations[target_length]
                    _replace_publication_path(staged_path, destination)
                    published.append(target_length)
            except BaseException as publication_error:
                rollback_errors: list[str] = []
                for target_length in reversed(published):
                    destination = destinations[target_length]
                    try:
                        destination.unlink(missing_ok=True)
                    except Exception as exc:
                        rollback_errors.append(f"remove {destination}: {exc}")
                for target_length, backup in reversed(tuple(backups.items())):
                    try:
                        _replace_publication_path(
                            backup, destinations[target_length]
                        )
                    except Exception as exc:
                        rollback_errors.append(
                            f"restore {destinations[target_length]}: {exc}"
                        )
                if rollback_errors:
                    rollback_incomplete = True
                    raise RuntimeError(
                        f"[{publication_key}] bucket publication failed and rollback "
                        f"was incomplete; recovery files remain in {transaction_root}: "
                        f"{'; '.join(rollback_errors)}"
                    ) from publication_error
                raise
    finally:
        if not rollback_incomplete:
            shutil.rmtree(transaction_root)

    return stats


def publish_outputs(
    repo: str,
    packed_by_length: dict[int, Path],
    target_lengths: Sequence[int] | None = None,
) -> dict[str, dict]:
    all_lengths = set(packed_by_length if target_lengths is None else target_lengths)
    try:
        return publish_bucket_outputs_atomically(
            repo,
            packed_by_length,
            output_root=OUTPUT_ROOT,
            filename=code_output_filename(repo),
            remove_lengths=sorted(all_lengths - set(packed_by_length)),
        )
    except RepoFailure:
        raise
    except Exception as exc:
        raise RepoFailure(
            repo,
            "publish",
            f"{type(exc).__name__}: {exc}",
        ) from exc


def append_output(repo: str, packed: Path, target_length: int) -> dict:
    """Publish one packed parquet through the atomic bucket publisher."""
    return publish_outputs(repo, {target_length: packed})[str(target_length)]


def _parquet_stats(path: Path, target_length: int) -> dict:
    """Read rows/tokens/padding from a packed parquet via the venv python."""
    code = (
        "import json,sys; import pyarrow.parquet as pq;"
        "t=pq.read_table(sys.argv[1]);"
        "n=t.num_rows;"
        "vtc=t.column('valid_token_count').to_pylist() if 'valid_token_count' in t.column_names else [];"
        "tl=int(sys.argv[2]);"
        "tot=n*tl;"
        "valid=sum(vtc);"
        "pad=tot-valid;"
        "print(json.dumps({'rows':n,'capacity_tokens':tot,'valid_tokens':valid,"
        "'pad_tokens':pad,'pad_frac':(pad/tot if tot else 0.0)}))"
    )
    proc = subprocess.run(
        [str(VENV_PYTHON), "-c", code, str(path), str(target_length)],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"stats read failed for {path}: {proc.stderr}")
    return json.loads(proc.stdout.strip())


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def process_one_repo(
    repo: str,
    repo_dir: Path,
    target_lengths: Sequence[int],
    work_root: Path,
    dedup_db: Path | None = None,
    dedup_near: bool = True,
    global_symbol_index: Path | None = None,
    memory_limit_gb: float = 10.0,
    parse_workers: int = 2,
    index_timeout_s: int | None = None,
    index_stall_timeout_s: int | None = None,
    *,
    project_id: str,
    promote_dedup_on_success: bool = True,
    source_quarantine_manifest: Path | None = None,
) -> dict:
    """Index a repo, then ROUTE each code doc to exactly ONE length bucket.

    Corrected design (route-by-fit, carried over from the commits driver):
    losslessly split and materialize the repo's docs ONCE, then route each
    aligned split piece by token count into the smallest fitting length bucket.
    Any residual row longer than the largest bucket fails the repo before
    publication. Each routed bucket is packed independently and landed at
    outputs/reindexed/<L>/<repo>.parquet. A given doc therefore appears in
    exactly one bucket -- never replicated across target lengths.
    """
    work = work_root / filesystem_repo_key(repo)
    project_id = require_project_identity(
        project_id, source=f"process_one_repo({repo})"
    )
    work.mkdir(parents=True, exist_ok=True)
    lengths_sorted = sorted(int(x) for x in target_lengths)
    materialize_budget = lossless_materialize_budget(lengths_sorted)
    stage_id = code_stage_id(repo) if dedup_db is not None else None
    stage_db = code_stage_db(work, repo) if dedup_db is not None else None
    promoted = False
    success_without_promote = False
    try:
        timings: dict[str, float] = {}
        started = time.monotonic()
        try:
            enriched = stage_index_source(
                repo, project_id, repo_dir, work, dedup_db, dedup_near,
                stage_id, stage_db, global_symbol_index, memory_limit_gb,
                parse_workers, index_timeout_s, index_stall_timeout_s,
                source_quarantine_manifest,
            )
        except RepoNoTrainingDocs as exc:
            timings["index_project_s"] = round(time.monotonic() - started, 6)
            raise RepoFailure(
                repo,
                "index_project",
                f"no training docs ({exc.reason}): {exc.detail}",
            ) from exc
        timings["index_project_s"] = round(time.monotonic() - started, 6)
        source_quarantine_receipt = (
            read_source_quarantine_receipt(work, repo)
            if source_quarantine_manifest is not None
            else None
        )
        started = time.monotonic()
        tok = stage_materialize(
            repo=repo,
            enriched=enriched,
            work=work,
            memory_limit_gb=memory_limit_gb,
            project_id=project_id,
            max_tokens=materialize_budget,
            fixed_shape_max_tokens=lengths_sorted[-1],
        )
        materialize_stats = read_materialize_stats(tok)
        timings["materialize_s"] = round(time.monotonic() - started, 6)

        started = time.monotonic()
        _bucket_for, route_by_fit = _route_by_fit_impl()
        route_dir = work / "routed"
        routed = route_by_fit(tok, lengths_sorted, route_dir, repo=repo)
        if not routed:
            raise RepoFailure(repo, "route_by_fit", f"no docs routed for {repo}")
        timings["route_by_fit_s"] = round(time.monotonic() - started, 6)

        started = time.monotonic()
        packed_by_length: dict[int, Path] = {}
        for L, route_parquet in sorted(routed.items()):
            packed_by_length[L] = stage_pack(repo, route_parquet, L, work)
        per_length = publish_outputs(repo, packed_by_length, lengths_sorted)
        timings["pack_s"] = round(time.monotonic() - started, 6)
        if promote_dedup_on_success:
            timings.update(promote_dedup_stage(dedup_db, stage_id, stage_db))
            promoted = True
        else:
            success_without_promote = True
        result = {
            "source": "code",
            "artifact_filename": code_output_filename(repo),
            "lengths": per_length,
            "stage_timings_s": timings,
            "materialize_stats": materialize_stats,
        }
        if source_quarantine_receipt is not None:
            result["source_quarantine_receipt"] = source_quarantine_receipt
        return result
    finally:
        if not promoted and not success_without_promote:
            discard_dedup_stage(dedup_db, stage_id, stage_db)


def process_one_commit_source(
    key: str,
    commit_inputs: Sequence[Path],
    target_lengths: Sequence[int],
    work_root: Path,
    repo_root: Path | None,
    repo_dir: Path | None,
    dedup_db: Path | None = None,
    dedup_near: bool = True,
    memory_limit_gb: float = 10.0,
    analysis_cache_entries: int = 128,
    *,
    project_id: str,
) -> dict:
    work = work_root / filesystem_repo_key(key)
    project_id = require_project_identity(
        project_id, source=f"process_one_commit_source({key})"
    )
    work.mkdir(parents=True, exist_ok=True)
    lengths_sorted = sorted(int(x) for x in target_lengths)
    materialize_budget = lossless_materialize_budget(lengths_sorted)
    stage_id = commit_stage_id(key) if dedup_db is not None else None
    stage_db = commit_stage_db(work, key) if dedup_db is not None else None
    promoted = False
    try:
        timings: dict[str, float] = {}
        started = time.monotonic()
        enriched = stage_index_commits(
            key, commit_inputs, work, repo_root, repo_dir,
            dedup_db, dedup_near, stage_id,
            stage_db,
            memory_limit_gb=memory_limit_gb,
            analysis_cache_entries=analysis_cache_entries,
            project_id=project_id,
        )
        timings["process_commits_s"] = round(time.monotonic() - started, 6)
        if enriched is None:
            return {"source": "commits", "lengths": {}, "stage_timings_s": timings}
        started = time.monotonic()
        tok = stage_materialize(
            repo=key,
            enriched=enriched,
            work=work,
            memory_limit_gb=memory_limit_gb,
            project_id=project_id,
            max_tokens=materialize_budget,
            fixed_shape_max_tokens=lengths_sorted[-1],
        )
        materialize_stats = read_materialize_stats(tok)
        timings["materialize_s"] = round(time.monotonic() - started, 6)

        started = time.monotonic()
        _bucket_for, route_by_fit = _route_by_fit_impl()
        route_dir = work / "routed"
        routed = route_by_fit(tok, lengths_sorted, route_dir, repo=key)
        if not routed:
            raise RepoFailure(key, "route_by_fit", f"no docs routed for {key}")
        timings["route_by_fit_s"] = round(time.monotonic() - started, 6)

        started = time.monotonic()
        packed_by_length: dict[int, Path] = {}
        for L, route_parquet in sorted(routed.items()):
            packed_by_length[L] = stage_pack(key, route_parquet, L, work)
        per_length = publish_outputs(key, packed_by_length, lengths_sorted)
        timings["pack_s"] = round(time.monotonic() - started, 6)
        timings.update(promote_dedup_stage(dedup_db, stage_id, stage_db))
        promoted = True
        return {
            "source": "commits",
            "lengths": per_length,
            "stage_timings_s": timings,
            "materialize_stats": materialize_stats,
        }
    finally:
        if not promoted:
            discard_dedup_stage(dedup_db, stage_id, stage_db)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_target_lengths = ",".join(str(length) for length in DEFAULT_TARGET_LENGTHS)
    p.add_argument(
        "--target-lengths",
        default=default_target_lengths,
        help=f"Comma-separated packed lengths (default: {default_target_lengths}).",
    )
    p.add_argument("--max-repos", type=int, default=None,
                   help="Process at most N repos this run (after resume filtering).")
    p.add_argument("--token-budget", type=int, default=None,
                   help="Stop after the cumulative valid-token count (summed over "
                        "the smallest target length) reaches this many tokens.")
    p.add_argument("--resume", action="store_true",
                   help="Skip repos already marked done in _done.json (default on; "
                        "flag kept for explicitness).")
    p.add_argument("--no-resume", action="store_true",
                   help="Reprocess even repos already marked done.")
    p.add_argument("--commit-source", action="append", default=[],
                   help="key=path[,path2,...] commit JSONL source re-indexed via "
                        "process_commits.py. Repeatable.")
    p.add_argument("--commit-repo-root", default=None,
                   help="Single repo root passed to process_commits --repo-root.")
    p.add_argument("--commit-repo-dir", default=None,
                   help="Parent dir of repos passed to process_commits --repo-dir.")
    p.add_argument("--keep-temp", action="store_true",
                   help="Do not delete per-repo temp dirs (debugging).")
    p.add_argument("--work-dir", default=None,
                   help="Staging root (default: a fresh mkdtemp under the system "
                        "temp dir).")
    p.add_argument("--dedup-db", default=None,
                   help="Path to the SHARED global dedup SQLite store. Function-level "
                        "tokenized-hash dedup (code) and commit-doc tokenized-hash "
                        "dedup (commits) both write to this ONE db, so dedup is "
                        "global + resumable across repos AND across the code+commit "
                        "streams. Pass the SAME path to streaming_reindex_commits.py.")
    p.add_argument("--no-near-dedup", action="store_true",
                   help="Disable MinHash-LSH near dedup (exact-only).")
    p.add_argument("--global-symbol-index", default=None,
                   help="Path to the GLOBAL cross-repo base-lib symbol SQLite store "
                        "(built by scripts/crossrepo/build_global_symbol_index.py). "
                        "When set, the CODE stage threads it into index_project so "
                        "unresolved base-lib callees are pulled in as bounded "
                        "depth-1 deps tagged crosslib:<repo>. DEFAULT off.")
    p.add_argument(
        "--repo-list",
        default=str(DEFAULT_REPO_LIST),
        help="repo_list.json mapping bare extraction names to canonical project "
        f"identities. Default {DEFAULT_REPO_LIST}.",
    )
    p.add_argument(
        "--source-quarantine-manifest",
        default=str(DEFAULT_SOURCE_QUARANTINE_MANIFEST),
        help="Exact path+size+SHA manifest for verified non-C++ files stored "
             "under C/C++ suffixes. Passed to every code indexing subprocess; "
             f"default {DEFAULT_SOURCE_QUARANTINE_MANIFEST}.",
    )
    p.add_argument("--memory-limit-gb", type=float, default=10.0,
                   help="Per-stage fail-loud RSS limit passed to index/materialize/"
                        "commit processors (default 10.0).")
    p.add_argument("--analysis-cache-entries", type=int, default=128,
                   help="Bounded per-process LRU entries passed to "
                        "process_commits.py for commit-source runs. Default 128; "
                        "use 0 to disable.")
    p.add_argument("--parse-workers", type=int, default=2,
                   help="Parse workers passed to index_project for the code stage "
                        "(default 2; keep this low when multiple repos run).")
    p.add_argument("--code-index-timeout-s", type=int, default=0,
                   help="Optional fail-loud timeout for each index_project code "
                        "stage. 0 disables the timeout (default).")
    p.add_argument("--code-index-stall-timeout-s", type=int, default=0,
                   help="Optional fail-loud stall watchdog for index_project: "
                        "kill when the log file has no size/mtime progress for "
                        "this many seconds. 0 disables the watchdog (default).")
    p.add_argument("--source-cache-dir", default=None,
                   help="Optional repo-level source cache for code-only runs. "
                        "Complete cached repos are reused before opening the "
                        "source tarball; uncached repos are materialized into "
                        "the cache and marked with a completion sentinel.")
    p.add_argument("--source-cache-only", action="store_true",
                   help="Only process repos already complete in --source-cache-dir; "
                        "do not open/decompress the source tarball.")
    p.add_argument("--source-cache-populate-only", action="store_true",
                   help="Populate --source-cache-dir from the source tarball and "
                        "exit without indexing/tokenizing. Use the resulting cache "
                        "with --source-cache-only for hot code runs.")
    p.add_argument("--source-dir-root", action="append", default=[],
                   help="Already-extracted repo root to process directly, without "
                        "opening the source tarball. May be passed multiple times; "
                        "children named <repo>/_src or <repo> are treated as repo dirs.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    target_lengths = [int(x) for x in args.target_lengths.split(",") if x.strip()]
    if not target_lengths:
        raise SystemExit("--target-lengths produced no lengths")
    smallest = min(target_lengths)

    for path in (VENV_PYTHON, TOKENIZER_PATH, INDEX_PROJECT, PROCESS_COMMITS,
                 MATERIALIZER, PACKER):
        if not Path(path).exists():
            raise SystemExit(f"required path missing: {path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for tl in target_lengths:
        (OUTPUT_ROOT / str(tl)).mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(MANIFEST_PATH)
    resume = not args.no_resume
    source_cache_dir = Path(args.source_cache_dir) if args.source_cache_dir else None
    source_dir_roots = [Path(p) for p in args.source_dir_root]
    source_quarantine_manifest = (
        Path(args.source_quarantine_manifest)
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
    if args.source_cache_populate_only and args.commit_source:
        raise SystemExit("--source-cache-populate-only is code-only; remove --commit-source")
    if source_dir_roots and (source_cache_dir is not None or args.source_cache_only):
        raise SystemExit("--source-dir-root cannot be combined with source cache flags")

    # Shared global dedup db (cross-repo + cross-stream). FAIL LOUD up front:
    # open it once here so a bad path / missing datasketch crashes before any
    # work (RULE #1). Each per-repo subprocess reopens against the same WAL db.
    dedup_db = Path(args.dedup_db) if args.dedup_db else None
    dedup_near = not args.no_near_dedup
    if dedup_db is not None:
        sys.path.insert(0, str(ROOT / "tools" / "clang_indexer"))
        from dedup_store import DedupStore  # noqa: E402
        dedup_db.parent.mkdir(parents=True, exist_ok=True)
        # Path/schema validation only; avoid rebuilding persisted MinHash/LSH in
        # the driver parent before any repo work starts.
        DedupStore(str(dedup_db), near=False, commit_every=1000).close()
        print(f"Dedup: SHARED global store at {dedup_db} "
              f"(exact{'+near' if dedup_near else ''}, tokenized hash)",
              file=sys.stderr)

    # Optional cross-repo base-lib symbol index. FAIL LOUD if given but missing.
    global_symbol_index = Path(args.global_symbol_index) if args.global_symbol_index else None
    if global_symbol_index is not None:
        if not global_symbol_index.exists():
            raise SystemExit(f"--global-symbol-index not found: {global_symbol_index}")
        print(f"Cross-lib: GLOBAL base-lib symbol index at {global_symbol_index} "
              f"threaded into CODE stage (bounded depth-1 pulls).", file=sys.stderr)
    repo_list = Path(args.repo_list) if args.repo_list else None
    project_id_map = (
        load_project_identity_map(repo_list) if repo_list is not None else {}
    )

    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        own_work_root = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix="streaming_reindex_"))
        own_work_root = True

    processed = 0
    cumulative_valid = 0
    run_report: dict[str, dict] = {}

    # ----- commit sources first (independent of tarball extraction) -----
    commit_sources: list[tuple[str, str, list[Path]]] = []
    for spec in args.commit_source:
        if "=" not in spec:
            raise SystemExit(f"--commit-source must be key=path,...; got: {spec}")
        key, paths = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--commit-source key must not be empty: {spec}")
        files = [Path(p) for p in paths.split(",") if p.strip()]
        for f in files:
            if not f.exists():
                raise SystemExit(f"commit source file missing: {f}")
        try:
            project_id = (
                require_project_identity(key, source="streaming commit source")
                if "/" in key
                else project_id_map[key]
            )
        except KeyError as exc:
            raise SymbolIdentityError(
                f"{repo_list}: no canonical project identity for "
                f"bare commit source {key!r}"
            ) from exc
        commit_sources.append((key, project_id, files))

    for key, project_id, files in commit_sources:
        manifest_key = f"commits:{key}"
        if resume and manifest.is_done(manifest_key):
            print(f"SKIP (done) {manifest_key}", file=sys.stderr)
            continue
        if args.max_repos is not None and processed >= args.max_repos:
            break
        manifest.mark_started(manifest_key)
        try:
            info = process_one_commit_source(
                key, files, target_lengths, work_root,
                Path(args.commit_repo_root) if args.commit_repo_root else None,
                Path(args.commit_repo_dir) if args.commit_repo_dir else None,
                dedup_db, dedup_near, args.memory_limit_gb,
                args.analysis_cache_entries,
                project_id=project_id,
            )
            manifest.mark_done(manifest_key, info)
            run_report[manifest_key] = info
            processed += 1
            # KeyError-safe: with route-by-fit a source may have NO doc in the
            # smallest bucket. Mirror streaming_reindex_commits.py:466.
            cumulative_valid += info["lengths"].get(str(smallest), {}).get("valid_tokens", 0)
            if not args.keep_temp:
                shutil.rmtree(
                    work_root / filesystem_repo_key(key),
                    ignore_errors=True,
                )
        except RepoFailure as exc:
            print(f"FAIL {manifest_key}: {exc}", file=sys.stderr)
            manifest.mark_failed(manifest_key, exc.stage, exc.detail)
        if args.token_budget is not None and cumulative_valid >= args.token_budget:
            print(f"Token budget {args.token_budget} reached.", file=sys.stderr)

    # ----- code repos from the tarball: TRUE single streaming pass -----
    budget_reached = (
        args.token_budget is not None and cumulative_valid >= args.token_budget
    )
    cap_reached = args.max_repos is not None and processed >= args.max_repos

    if not budget_reached and not cap_reached:
        def should_process(repo: str) -> bool:
            return is_code_worktree_repo(repo) and not (resume and manifest.is_done(repo))

        if args.source_cache_populate_only:
            report = populate_source_cache(
                work_root,
                should_process,
                source_cache_dir,
                max_repos=args.max_repos,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            if own_work_root and not args.keep_temp:
                shutil.rmtree(work_root, ignore_errors=True)
            return 0
        gen = (
            stream_repo_dirs(source_dir_roots, should_process)
            if source_dir_roots
            else stream_repo_subtrees(
                work_root,
                should_process,
                source_cache_dir=source_cache_dir,
                source_cache_only=args.source_cache_only,
            )
        )
        try:
            for repo, repo_dir in gen:
                manifest.mark_started(repo)
                try:
                    try:
                        project_id = (
                            require_project_identity(repo, source="streaming repo")
                            if "/" in repo
                            else project_id_map[repo]
                        )
                    except KeyError as exc:
                        raise SymbolIdentityError(
                            f"{repo_list}: no canonical project identity for "
                            f"bare repo {repo!r}"
                        ) from exc
                    info = process_one_repo(repo, repo_dir, target_lengths, work_root,
                                            dedup_db, dedup_near,
                                            global_symbol_index,
                                            args.memory_limit_gb,
                                            args.parse_workers,
                                            args.code_index_timeout_s,
                                            args.code_index_stall_timeout_s,
                                            project_id=project_id,
                                            source_quarantine_manifest=(
                                                source_quarantine_manifest
                                            ))
                    manifest.mark_done(repo, info)
                    run_report[repo] = info
                    processed += 1
                    # KeyError-safe smallest-bucket stats (route-by-fit may skip it).
                    added = info["lengths"].get(str(smallest), {}).get("valid_tokens", 0)
                    cumulative_valid += added
                    print(
                        f"DONE {repo}: +{added} tok @ {smallest} "
                        f"(cum {cumulative_valid}, repos {processed})",
                        file=sys.stderr, flush=True,
                    )
                except RepoFailure as exc:
                    print(f"FAIL {repo}: {exc}", file=sys.stderr)
                    manifest.mark_failed(repo, exc.stage, exc.detail)
                finally:
                    if not args.keep_temp:
                        shutil.rmtree(
                            work_root / filesystem_repo_key(repo),
                            ignore_errors=True,
                        )
                stop = False
                if args.max_repos is not None and processed >= args.max_repos:
                    stop = True
                if args.token_budget is not None and cumulative_valid >= args.token_budget:
                    print(f"Token budget {args.token_budget} reached.", file=sys.stderr)
                    stop = True
                if stop:
                    break
        finally:
            gen.close()

    if own_work_root and not args.keep_temp:
        shutil.rmtree(work_root, ignore_errors=True)

    # ----- final report -----
    totals = {str(tl): {"rows": 0, "valid_tokens": 0, "pad_tokens": 0, "capacity_tokens": 0}
              for tl in target_lengths}
    for info in run_report.values():
        for tl_s, st in info["lengths"].items():
            agg = totals[tl_s]
            agg["rows"] += st["rows"]
            agg["valid_tokens"] += st["valid_tokens"]
            agg["pad_tokens"] += st["pad_tokens"]
            agg["capacity_tokens"] += st["capacity_tokens"]
    summary = {
        "processed_this_run": processed,
        "total_done": len(manifest.done),
        "total_failed": len(manifest.failed),
        "per_length_totals": totals,
        "materialize_split_totals": summarize_materialize_stats(
            list(run_report.values())
        ),
        "manifest": str(MANIFEST_PATH),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not manifest.failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
