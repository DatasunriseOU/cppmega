#!/usr/bin/env python3
"""Rescue failed CI run archives through GitHub's per-job log endpoint.

The normal CI fetcher downloads one ZIP for a complete workflow attempt.  Very
large archives can repeatedly terminate early, including when the backing blob
ignores a byte-range resume.  This worker deliberately changes the unit of
recovery to an individual job:

* only an exact, failed fetch-state attempt with durable jobs evidence is used;
* every job is resolved to a complete streamed log or an API-proven 404/410;
* resolved jobs are durable and independently replay-validated;
* a deterministic synthetic ZIP is published to the normal rescue spool only
  after coverage reaches 100%; and
* the exact unchanged failed row is auditably requeued in one SQLite
  transaction.

Authorization is sent only to ``api.github.com``.  Signed redirect URLs are
validated and fetched without GitHub credentials.  Redirect URLs, response
bodies, and tokens are never persisted in records or receipts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import sys
import threading
import time
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Sequence, cast
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci_stream_fetch import (  # noqa: E402
    ArchiveError as FetchArchiveError,
    _fsync_directory,
    _safe_zip_infos,
)
from scripts.ci_stream_inventory import (  # noqa: E402
    GITHUB_API_VERSION,
    InventoryError,
    TokenPool,
    load_token_pool,
)


SCHEMA_VERSION = "cppmega_ci_job_log_rescue_v1"
JOB_RECORD_SCHEMA = "cppmega_ci_job_log_rescue_job_v1"
RESOLVED_JOBS_SCHEMA = "cppmega_ci_job_log_rescue_resolved_jobs_v1"
RECEIPT_SCHEMA = "cppmega_ci_job_log_rescue_receipt_v1"
PROGRESS_SCHEMA = "cppmega_ci_job_log_rescue_progress_v1"
BINDING_SCHEMA = "cppmega_ci_job_log_rescue_binding_v1"

DEFAULT_TIMEOUT = 90.0
DEFAULT_JOB_ATTEMPTS = 8
DEFAULT_WORKERS = 4
DEFAULT_MAX_JOB_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_ZIP_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_POLL_SECONDS = 30.0
_STREAM_CHUNK_BYTES = 1024 * 1024
_ERROR_BODY_PREFIX_BYTES = 64 * 1024
_ELIGIBLE_ERROR_CLASSES = {"ArchiveError", "IncompleteRead"}
_SIGNED_ARCHIVE_403_FRAGMENT = "signed archive URL returned HTTP 403"
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9_.-]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_SIGNED_HOST_SUFFIXES = (
    ".blob.core.windows.net",
    ".actions.githubusercontent.com",
    ".githubusercontent.com",
)
_SAFE_SIGNED_EXACT_HOSTS = {
    "results-receiver.actions.githubusercontent.com",
}
_SIGNATURE_QUERY_KEYS = {
    "sig",
    "signature",
    "token",
    "jwt",
    "x-amz-signature",
}
_MANIFEST_FIELDS = (
    "repo",
    "run_id",
    "attempt",
    "created_at",
    "status",
    "bytes",
    "sha256",
    "finished_at",
)


class RescueError(RuntimeError):
    """Base class for a fail-closed rescue failure."""


class StateBindingError(RescueError):
    """The fetch-state evidence is absent, corrupt, or changed."""


class UnsafeRedirectError(RescueError):
    """GitHub returned a redirect that is unsafe to follow."""


class TransportExhausted(RescueError):
    """A job log did not reach a complete EOF within its retry budget."""


class ByteLimitError(RescueError):
    """A configured per-job, aggregate, or ZIP byte bound was exceeded."""


@dataclass(frozen=True)
class JobEvidence:
    ordinal: int
    job_id: int
    name: str
    member_name: str
    endpoint: str


@dataclass(frozen=True)
class SourceAttempt:
    state_path: Path
    repo: str
    canonical_repo: str
    run_id: int
    attempt: int
    created_at: str
    status: str
    tries: int
    error_class: str | None
    archive_source: str | None
    archive_sha256: str | None
    archive_size: int | None
    row_sha256: str
    run_metadata_sha256: str
    run_metadata_raw_size: int
    jobs_sha256: str
    jobs_raw_size: int
    jobs_ledger_sha256: str
    jobs_ledger_ids: tuple[int, ...]
    jobs: tuple[JobEvidence, ...]

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.repo, self.run_id, self.attempt

    @property
    def spool_base_name(self) -> str:
        return f"{self.repo.replace('/', '__')}--{self.run_id}--attempt-{self.attempt}"


@dataclass(frozen=True)
class JobOutcome:
    job: JobEvidence
    record: dict[str, object] | None
    error_class: str | None = None
    error_message: str | None = None

    @property
    def resolved(self) -> bool:
        return self.record is not None


class ByteBudget:
    """Thread-safe aggregate byte budget for durable full log files."""

    def __init__(self, limit: int, *, initial: int = 0):
        if limit < 0 or initial < 0 or initial > limit:
            raise ValueError("invalid byte budget")
        self.limit = limit
        self._used = initial
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def reserve(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("cannot reserve negative bytes")
        with self._lock:
            if self._used + amount > self.limit:
                raise ByteLimitError(
                    f"resolved log bytes would exceed aggregate limit {self.limit}"
                )
            self._used += amount

    def release(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("cannot release negative bytes")
        with self._lock:
            if amount > self._used:
                raise AssertionError("byte budget release exceeds usage")
            self._used -= amount


class _BoundedSeekableWriter:
    """Seekable file facade that never grows beyond an exact byte limit."""

    def __init__(self, handle: Any, limit: int):
        self._handle = handle
        self._limit = limit

    def write(self, payload: bytes) -> int:
        start = int(self._handle.tell())
        end = start + len(payload)
        if end > self._limit:
            raise ByteLimitError(f"synthetic ZIP would exceed byte limit {self._limit}")
        written = int(self._handle.write(payload))
        return written

    def tell(self) -> int:
        return int(self._handle.tell())

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return int(self._handle.seek(offset, whence))

    def flush(self) -> None:
        self._handle.flush()

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _default_opener(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    try:
        return _NO_REDIRECT_OPENER.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return exc


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_STREAM_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_error(value: object, secrets: Iterable[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    # Signed URLs never need to be useful in an operator-facing error.
    text = re.sub(r"https://[^ \t\r\n]+", "<redacted-url>", text)
    return text[:2000]


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("atomic write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(value) + b"\n")


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise RescueError(f"unsafe directory: {path}")
    os.chmod(path, 0o700)


def _decode_canonical_blob(
    row: Mapping[str, object],
    *,
    blob_column: str,
    size_column: str,
    digest_column: str,
    label: str,
) -> tuple[bytes, object]:
    raw_blob = row[blob_column]
    if raw_blob is None:
        raise StateBindingError(f"failed attempt has no {label} evidence")
    if not isinstance(raw_blob, (bytes, bytearray, memoryview)):
        raise StateBindingError(f"{label} evidence is not a SQLite BLOB")
    try:
        raw = zlib.decompress(bytes(raw_blob))
    except (TypeError, zlib.error) as exc:
        raise StateBindingError(f"{label} evidence is not valid zlib") from exc
    expected_size = row[size_column]
    expected_digest = row[digest_column]
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_digest, str)
        or _SHA256_RE.fullmatch(expected_digest) is None
    ):
        raise StateBindingError(f"{label} evidence metadata is invalid")
    if len(raw) != expected_size or _sha256_bytes(raw) != expected_digest:
        raise StateBindingError(f"{label} evidence digest/size mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StateBindingError(f"{label} evidence is not JSON") from exc
    if raw != _canonical_json_bytes(value):
        raise StateBindingError(f"{label} evidence is not canonical JSON")
    return raw, value


def _row_sha256(row: Mapping[str, object]) -> str:
    """Hash a SQLite row with unambiguous type and column framing."""

    digest = hashlib.sha256()
    for name in sorted(row):
        value = row[name]
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        if value is None:
            payload = b""
            tag = b"n"
        elif isinstance(value, bool):
            payload = b"1" if value else b"0"
            tag = b"t"
        elif isinstance(value, int):
            payload = str(value).encode("ascii")
            tag = b"i"
        elif isinstance(value, float):
            payload = value.hex().encode("ascii")
            tag = b"f"
        elif isinstance(value, str):
            payload = value.encode("utf-8")
            tag = b"s"
        elif isinstance(value, (bytes, bytearray, memoryview)):
            payload = bytes(value)
            tag = b"b"
        else:
            raise StateBindingError(
                f"attempt row column {name!r} has unsupported SQLite type"
            )
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _response_status(response: Any) -> int:
    raw = getattr(response, "status", None)
    if raw is None:
        raw = getattr(response, "code", None)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RescueError("HTTP response has no integer status")
    return raw


def _response_headers(response: Any) -> dict[str, str]:
    raw = getattr(response, "headers", {})
    try:
        items = raw.items()
    except AttributeError as exc:
        raise RescueError("HTTP response headers are not a mapping") from exc
    return {str(key).casefold(): str(value) for key, value in items}


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _read_error_prefix(response: Any) -> tuple[int, str, bool]:
    digest = hashlib.sha256()
    captured = 0
    truncated = False
    while captured <= _ERROR_BODY_PREFIX_BYTES:
        block = response.read(
            min(_STREAM_CHUNK_BYTES, _ERROR_BODY_PREFIX_BYTES + 1 - captured)
        )
        if not block:
            break
        captured += len(block)
        if captured > _ERROR_BODY_PREFIX_BYTES:
            digest.update(block[: len(block) - 1])
            captured -= 1
            truncated = True
            break
        digest.update(block)
    return captured, digest.hexdigest(), truncated


def _validate_signed_redirect(url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeRedirectError("signed redirect URL is malformed") from exc
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if (
        parsed.scheme.casefold() != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.fragment
    ):
        raise UnsafeRedirectError("signed redirect is not safe HTTPS")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise UnsafeRedirectError("signed redirect cannot use an IP literal")
    provider_host = (
        hostname in _SAFE_SIGNED_EXACT_HOSTS
        or any(hostname.endswith(suffix) for suffix in _SAFE_SIGNED_HOST_SUFFIXES)
        or (
            hostname.endswith(".amazonaws.com")
            and hostname.startswith("github-production-")
        )
    )
    if not provider_host:
        raise UnsafeRedirectError("signed redirect host is not GitHub storage")
    signed_parameters = [
        (key.casefold(), value)
        for key, value in urllib.parse.parse_qsl(
            parsed.query,
            keep_blank_values=True,
        )
        if key.casefold() in _SIGNATURE_QUERY_KEYS
    ]
    if not any(value for _key, value in signed_parameters):
        raise UnsafeRedirectError("signed redirect has no signature credential")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
    )


class FetchStateEvidence:
    """Read and transactionally mutate only the explicitly named fetch state."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser().resolve()
        if self.path.is_symlink() or not self.path.is_file():
            raise StateBindingError(
                f"fetch-state SQLite does not exist safely: {self.path}"
            )
        state_stat = self.path.stat(follow_symlinks=False)
        self._file_identity = (state_stat.st_dev, state_stat.st_ino)
        self.connection = sqlite3.connect(
            f"{self.path.as_uri()}?mode=rw",
            uri=True,
            timeout=60.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        try:
            self._validate_schema()
        except BaseException:
            self.connection.close()
            raise

    def _assert_file_identity(self) -> None:
        try:
            current = self.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise StateBindingError("fetch-state SQLite path disappeared") from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != self._file_identity
        ):
            raise StateBindingError("fetch-state SQLite path identity changed")

    def _validate_schema(self) -> None:
        self._assert_file_identity()
        tables = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"settings", "attempts", "request_ledger"}
        if not required.issubset(tables):
            raise StateBindingError("fetch-state SQLite lacks required durable tables")
        settings = dict(self.connection.execute("SELECT key,value FROM settings"))
        schema = settings.get("schema")
        if not isinstance(schema, str) or not schema.startswith(
            "cppmega_ci_stream_fetch_v"
        ):
            raise StateBindingError("SQLite is not a ci_stream_fetch state")
        attempt_columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(attempts)")
        }
        required_attempt_columns = {
            "repo",
            "run_id",
            "attempt",
            "created_at",
            "run_metadata_sha256",
            "run_metadata_raw_size",
            "run_metadata_zlib",
            "run_metadata_source",
            "run_metadata_source_attempt",
            "run_metadata_exact",
            "status",
            "tries",
            "archive_source",
            "archive_sha256",
            "archive_size",
            "jobs_sha256",
            "jobs_raw_size",
            "jobs_zlib",
            "error_class",
            "error_message",
            "updated_at",
        }
        if not required_attempt_columns.issubset(attempt_columns):
            raise StateBindingError("fetch-state attempts schema is incompatible")

    def close(self) -> None:
        self.connection.close()

    def scan(
        self,
        *,
        target: tuple[str, int, int] | None,
    ) -> list[tuple[str, int, int]]:
        with self._lock:
            self._assert_file_identity()
            if target is not None:
                row = self.connection.execute(
                    """
                    SELECT repo,run_id,attempt FROM attempts
                    WHERE lower(repo)=lower(?) AND run_id=? AND attempt=?
                    """,
                    target,
                ).fetchone()
                if row is None:
                    raise StateBindingError("explicit attempt does not exist")
                return [(str(row["repo"]), int(row["run_id"]), int(row["attempt"]))]
            placeholders = ",".join("?" for _ in _ELIGIBLE_ERROR_CLASSES)
            rows = self.connection.execute(
                f"""
                SELECT repo,run_id,attempt FROM attempts
                WHERE status='failed'
                  AND (
                    error_class IN ({placeholders})
                    OR error_message LIKE '%IncompleteRead%'
                    OR (
                      error_class='APIError'
                      AND error_message LIKE ?
                    )
                  )
                ORDER BY created_at,repo,run_id,attempt
                """,
                (
                    *sorted(_ELIGIBLE_ERROR_CLASSES),
                    f"%{_SIGNED_ARCHIVE_403_FRAGMENT}%",
                ),
            ).fetchall()
        return [
            (str(row["repo"]), int(row["run_id"]), int(row["attempt"])) for row in rows
        ]

    def _selected_jobs_ledger(
        self,
        *,
        repo: str,
        canonical_repo: str,
        run_id: int,
        attempt: int,
        job_count: int,
    ) -> tuple[tuple[int, ...], str]:
        endpoint = (
            f"/repos/{canonical_repo}/actions/runs/{run_id}/attempts/{attempt}/jobs"
        )
        rows = self.connection.execute(
            """
            SELECT id,requested_at,page_no,http_status,outcome
            FROM request_ledger
            WHERE repo=? AND run_id=? AND attempt=? AND endpoint=?
              AND http_status=200 AND outcome='success'
            ORDER BY id
            """,
            (repo, run_id, attempt, endpoint),
        ).fetchall()
        expected_pages = max(1, math.ceil(job_count / 100))
        selected: list[sqlite3.Row] = []
        for page in range(1, expected_pages + 1):
            page_rows = [row for row in rows if int(row["page_no"] or 0) == page]
            if not page_rows:
                raise StateBindingError(
                    f"jobs evidence lacks successful ledger page {page}"
                )
            selected.append(page_rows[-1])
        evidence = [
            {
                "id": int(row["id"]),
                "http_status": int(row["http_status"]),
                "outcome": str(row["outcome"]),
                "page_no": int(row["page_no"]),
                "requested_at": str(row["requested_at"]),
            }
            for row in selected
        ]
        return (
            tuple(int(row["id"]) for row in selected),
            _sha256_bytes(_canonical_json_bytes(evidence)),
        )

    def load_attempt(
        self,
        identity: tuple[str, int, int],
        *,
        explicit: bool,
    ) -> SourceAttempt:
        repo, run_id, attempt = identity
        with self._lock:
            self._assert_file_identity()
            row = self.connection.execute(
                """
                SELECT * FROM attempts
                WHERE repo=? AND run_id=? AND attempt=?
                """,
                (repo, run_id, attempt),
            ).fetchone()
            if row is None:
                raise StateBindingError("attempt disappeared before rescue")
            row_dict = dict(row)
            if str(row["status"]) != "failed":
                raise StateBindingError(
                    "attempt is not still failed; refusing a state rewrite"
                )
            error_class = (
                None if row["error_class"] is None else str(row["error_class"])
            )
            error_message = (
                "" if row["error_message"] is None else str(row["error_message"])
            )
            if (
                not explicit
                and error_class not in _ELIGIBLE_ERROR_CLASSES
                and "IncompleteRead" not in error_message
                and not (
                    error_class == "APIError"
                    and _SIGNED_ARCHIVE_403_FRAGMENT in error_message
                )
            ):
                raise StateBindingError(
                    "failed attempt is not an eligible archive transport failure"
                )
            if int(row["run_metadata_exact"]) != 1:
                raise StateBindingError(
                    "job rescue requires exact workflow-attempt metadata"
                )
            if int(row["run_metadata_source_attempt"]) != attempt:
                raise StateBindingError(
                    "run metadata is not bound to the target attempt"
                )
            metadata_raw, metadata_value = _decode_canonical_blob(
                row_dict,
                blob_column="run_metadata_zlib",
                size_column="run_metadata_raw_size",
                digest_column="run_metadata_sha256",
                label="run metadata",
            )
            if not isinstance(metadata_value, dict):
                raise StateBindingError("run metadata is not an object")
            if (
                metadata_value.get("id") != run_id
                or metadata_value.get("run_attempt") != attempt
                or metadata_value.get("status") != "completed"
            ):
                raise StateBindingError(
                    "run metadata does not prove this completed exact attempt"
                )
            metadata_created_at = metadata_value.get("created_at")
            state_created_at = row["created_at"]
            if (
                not isinstance(metadata_created_at, str)
                or metadata_created_at != state_created_at
                or not metadata_created_at
                or any(
                    character in metadata_created_at for character in ("\t", "\r", "\n")
                )
            ):
                raise StateBindingError(
                    "run metadata created_at does not match the state row"
                )
            repository = metadata_value.get("repository")
            if not isinstance(repository, dict):
                raise StateBindingError(
                    "run metadata lacks canonical repository identity"
                )
            canonical_repo = repository.get("full_name")
            repository_id = repository.get("id")
            if (
                not isinstance(canonical_repo, str)
                or _REPOSITORY_RE.fullmatch(canonical_repo) is None
                or _REPOSITORY_RE.fullmatch(repo) is None
                or (
                    repository_id is not None
                    and (
                        isinstance(repository_id, bool)
                        or not isinstance(repository_id, int)
                        or repository_id <= 0
                    )
                )
            ):
                raise StateBindingError("run metadata repository identity is invalid")
            jobs_raw, jobs_value = _decode_canonical_blob(
                row_dict,
                blob_column="jobs_zlib",
                size_column="jobs_raw_size",
                digest_column="jobs_sha256",
                label="jobs",
            )
            if not isinstance(jobs_value, list):
                raise StateBindingError("jobs evidence is not an array")
            jobs: list[JobEvidence] = []
            seen_ids: set[int] = set()
            for ordinal, value in enumerate(jobs_value):
                if not isinstance(value, dict):
                    raise StateBindingError("jobs evidence contains a non-object")
                job_id = value.get("id")
                name = value.get("name")
                if (
                    isinstance(job_id, bool)
                    or not isinstance(job_id, int)
                    or job_id <= 0
                    or job_id in seen_ids
                    or not isinstance(name, str)
                    or not name
                    or value.get("status") != "completed"
                ):
                    raise StateBindingError(
                        f"jobs evidence item {ordinal} is not exact/completed"
                    )
                if (
                    value.get("run_id", run_id) != run_id
                    or value.get("run_attempt", attempt) != attempt
                ):
                    raise StateBindingError(
                        f"job {job_id} belongs to another run attempt"
                    )
                seen_ids.add(job_id)
                jobs.append(
                    JobEvidence(
                        ordinal=ordinal,
                        job_id=job_id,
                        name=name,
                        member_name=f"{ordinal}_{job_id}.txt",
                        endpoint=(
                            f"/repos/{canonical_repo}/actions/jobs/{job_id}/logs"
                        ),
                    )
                )
            ledger_ids, ledger_sha = self._selected_jobs_ledger(
                repo=repo,
                canonical_repo=canonical_repo,
                run_id=run_id,
                attempt=attempt,
                job_count=len(jobs),
            )
            archive_source = (
                None if row["archive_source"] is None else str(row["archive_source"])
            )
            archive_sha256 = (
                None if row["archive_sha256"] is None else str(row["archive_sha256"])
            )
            archive_size = (
                None if row["archive_size"] is None else int(row["archive_size"])
            )
            archive_values = (
                archive_source,
                archive_sha256,
                archive_size,
            )
            if not (
                all(value is None for value in archive_values)
                or all(value is not None for value in archive_values)
            ) or (
                archive_source is not None
                and (
                    not archive_source
                    or any(
                        character in archive_source for character in ("\t", "\r", "\n")
                    )
                    or _SHA256_RE.fullmatch(str(archive_sha256)) is None
                    or archive_size is None
                    or archive_size < 0
                )
            ):
                raise StateBindingError("failed raw archive evidence is inconsistent")
        return SourceAttempt(
            state_path=self.path,
            repo=repo,
            canonical_repo=canonical_repo,
            run_id=run_id,
            attempt=attempt,
            created_at=str(row["created_at"]),
            status=str(row["status"]),
            tries=int(row["tries"]),
            error_class=error_class,
            archive_source=archive_source,
            archive_sha256=archive_sha256,
            archive_size=archive_size,
            row_sha256=_row_sha256(row_dict),
            run_metadata_sha256=_sha256_bytes(metadata_raw),
            run_metadata_raw_size=len(metadata_raw),
            jobs_sha256=_sha256_bytes(jobs_raw),
            jobs_raw_size=len(jobs_raw),
            jobs_ledger_sha256=ledger_sha,
            jobs_ledger_ids=ledger_ids,
            jobs=tuple(jobs),
        )

    def commit_rescue(
        self,
        source: SourceAttempt,
        *,
        receipt_sha256: str,
        publish: Callable[[], None],
    ) -> None:
        audit_message = (
            f"receipt_sha256={receipt_sha256} source_row_sha256={source.row_sha256}"
        )
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self._assert_file_identity()
                current = self.connection.execute(
                    """
                    SELECT * FROM attempts
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    source.identity,
                ).fetchone()
                if current is None:
                    raise StateBindingError("attempt disappeared before rescue requeue")
                if (
                    str(current["status"]) != "failed"
                    or _row_sha256(dict(current)) != source.row_sha256
                ):
                    raise StateBindingError(
                        "attempt row changed during rescue; nothing was requeued"
                    )
                ledger_ids, ledger_sha = self._selected_jobs_ledger(
                    repo=source.repo,
                    canonical_repo=source.canonical_repo,
                    run_id=source.run_id,
                    attempt=source.attempt,
                    job_count=len(source.jobs),
                )
                if (
                    ledger_ids != source.jobs_ledger_ids
                    or ledger_sha != source.jobs_ledger_sha256
                ):
                    raise StateBindingError(
                        "jobs request evidence changed during rescue"
                    )
                # Keep the SQLite write reservation until the ZIP commit
                # marker and audit sidecars are visible. A fetcher cannot
                # observe retry before its complete rescue input exists.
                publish()
                cursor = self.connection.execute(
                    """
                    UPDATE attempts SET status='retry',tries=0,updated_at=?
                    WHERE repo=? AND run_id=? AND attempt=? AND status='failed'
                    """,
                    (_utc_now(), *source.identity),
                )
                if cursor.rowcount != 1:
                    raise StateBindingError(
                        "failed attempt was not updated exactly once"
                    )
                self.connection.execute(
                    """
                    INSERT INTO request_ledger(
                      requested_at,repo,run_id,attempt,endpoint,page_no,
                      request_attempt,http_status,outcome,latency_ms,
                      error_class,error_message
                    ) VALUES (?,?,?,?,?,NULL,1,NULL,?,0,?,?)
                    """,
                    (
                        _utc_now(),
                        source.repo,
                        source.run_id,
                        source.attempt,
                        "operator/job_rescue",
                        "operator/job_rescue",
                        "JobRescueReceipt",
                        audit_message,
                    ),
                )
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    def completed_rescue_audit(
        self,
        identity: tuple[str, int, int],
    ) -> tuple[str, str] | None:
        """Return the bound receipt/source digests for an earlier requeue."""

        with self._lock:
            self._assert_file_identity()
            row = self.connection.execute(
                """
                SELECT status FROM attempts
                WHERE repo=? AND run_id=? AND attempt=?
                """,
                identity,
            ).fetchone()
            if row is None or str(row["status"]) == "failed":
                return None
            audit = self.connection.execute(
                """
                SELECT error_message FROM request_ledger
                WHERE repo=? AND run_id=? AND attempt=?
                  AND endpoint='operator/job_rescue'
                  AND outcome='operator/job_rescue'
                  AND error_class='JobRescueReceipt'
                ORDER BY id DESC LIMIT 1
                """,
                identity,
            ).fetchone()
        if audit is None or not isinstance(audit["error_message"], str):
            return None
        match = re.fullmatch(
            r"receipt_sha256=([0-9a-f]{64}) "
            r"source_row_sha256=([0-9a-f]{64})",
            str(audit["error_message"]),
        )
        if match is None:
            return None
        return match.group(1), match.group(2)


class JobLogClient:
    """Bounded per-job client with token rotation and no credential forwarding."""

    def __init__(
        self,
        tokens: Sequence[str],
        *,
        opener: Callable[..., Any] = _default_opener,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_JOB_ATTEMPTS,
        max_job_bytes: int = DEFAULT_MAX_JOB_BYTES,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_job_bytes < 0:
            raise ValueError("max_job_bytes must be non-negative")
        self.pool = TokenPool(tokens, sleeper=sleeper)
        self.opener = opener
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.max_job_bytes = max_job_bytes
        self.sleeper = sleeper
        self.api_base = "https://api.github.com"

    @property
    def secrets(self) -> tuple[str, ...]:
        return self.pool.secrets

    def _open(self, request: urllib.request.Request) -> Any:
        return self.opener(request, timeout=self.timeout)

    @staticmethod
    def _rate_limited(status: int, headers: Mapping[str, str]) -> bool:
        return status == 429 or (
            status == 403
            and (
                headers.get("x-ratelimit-remaining") == "0" or "retry-after" in headers
            )
        )

    def _stream_response(
        self,
        response: Any,
        destination: Path,
        *,
        budget: ByteBudget,
    ) -> tuple[int, str]:
        headers = _response_headers(response)
        encoding = headers.get("content-encoding", "identity").casefold()
        if encoding not in {"", "identity"}:
            raise RescueError(
                f"job log returned unsupported Content-Encoding {encoding!r}"
            )
        declared: int | None = None
        if "content-length" in headers:
            try:
                declared = int(headers["content-length"])
            except ValueError as exc:
                raise RescueError("job log Content-Length is not an integer") from exc
            if declared < 0:
                raise RescueError("job log Content-Length is negative")
            if declared > self.max_job_bytes:
                raise ByteLimitError(
                    f"job log Content-Length {declared} exceeds {self.max_job_bytes}"
                )
        transfer_encoding = headers.get("transfer-encoding", "").casefold()
        if declared is not None and transfer_encoding:
            raise RescueError(
                "job log has conflicting Content-Length and Transfer-Encoding"
            )
        if declared is None and transfer_encoding != "chunked":
            raise RescueError("job log response lacks complete length/chunk framing")
        if destination.exists() or destination.is_symlink():
            raise RescueError("job log temporary destination already exists")
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        total = 0
        digest = hashlib.sha256()
        try:
            while True:
                try:
                    block = response.read(_STREAM_CHUNK_BYTES)
                except http.client.IncompleteRead as exc:
                    raise TransportExhausted(
                        "job log transport ended before EOF (IncompleteRead)"
                    ) from exc
                if not block:
                    break
                if total + len(block) > self.max_job_bytes:
                    raise ByteLimitError(
                        f"job log exceeded byte limit {self.max_job_bytes}"
                    )
                budget.reserve(len(block))
                try:
                    view = memoryview(block)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError("job log write made no progress")
                        view = view[written:]
                except BaseException:
                    budget.release(len(block))
                    raise
                total += len(block)
                digest.update(block)
            if declared is not None and total != declared:
                raise TransportExhausted(f"job log length mismatch {total}!={declared}")
            os.fsync(descriptor)
        except BaseException:
            budget.release(total)
            raise
        finally:
            os.close(descriptor)
        return total, digest.hexdigest()

    def _record_terminal(
        self,
        source: SourceAttempt,
        job: JobEvidence,
        *,
        status: int,
        response: Any,
        request_attempts: int,
    ) -> dict[str, object]:
        captured, body_sha, truncated = _read_error_prefix(response)
        return {
            "schema": JOB_RECORD_SCHEMA,
            "source_row_sha256": source.row_sha256,
            "jobs_sha256": source.jobs_sha256,
            "repo": source.repo,
            "run_id": source.run_id,
            "attempt": source.attempt,
            "ordinal": job.ordinal,
            "job_id": job.job_id,
            "job_name": job.name,
            "endpoint": job.endpoint,
            "member_name": None,
            "outcome": f"terminal_{status}",
            "api_http_status": status,
            "signed_http_status": None,
            "request_attempts": request_attempts,
            "log": None,
            "terminal": {
                "http_status": status,
                "body_prefix_bytes": captured,
                "body_prefix_sha256": body_sha,
                "body_truncated": truncated,
            },
        }

    def fetch(
        self,
        source: SourceAttempt,
        job: JobEvidence,
        *,
        log_path: Path,
        budget: ByteBudget,
    ) -> JobOutcome:
        part_path = log_path.with_suffix(".log.part")
        if log_path.exists() or log_path.is_symlink():
            if log_path.is_symlink() or not log_path.is_file():
                return JobOutcome(
                    job,
                    None,
                    "RescueError",
                    "unsafe unbound job log",
                )
            orphan = log_path.with_suffix(".log.unbound")
            if orphan.exists() or orphan.is_symlink():
                return JobOutcome(
                    job,
                    None,
                    "RescueError",
                    "conflicting unbound job log",
                )
            os.replace(log_path, orphan)
            _fsync_directory(log_path.parent)
        if part_path.exists() or part_path.is_symlink():
            if part_path.is_symlink() or not part_path.is_file():
                return JobOutcome(
                    job,
                    None,
                    "RescueError",
                    "unsafe prior job partial",
                )
            part_path.unlink()
        last_error: BaseException | None = None
        for request_attempt in range(1, self.max_attempts + 1):
            token_index, token = self.pool.acquire()
            api_request = urllib.request.Request(
                f"{self.api_base}{job.endpoint}",
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "cppmega-ci-job-log-rescue/1",
                    "X-GitHub-Api-Version": GITHUB_API_VERSION,
                },
                method="GET",
            )
            api_response: Any | None = None
            try:
                api_response = self._open(api_request)
                api_status = _response_status(api_response)
                api_headers = _response_headers(api_response)
                self.pool.observe(token_index, api_headers)
                if self._rate_limited(api_status, api_headers):
                    self.pool.rate_limited(
                        token_index,
                        api_headers,
                        secondary=api_status == 403,
                    )
                    _read_error_prefix(api_response)
                    continue
                if api_status in {404, 410}:
                    record = self._record_terminal(
                        source,
                        job,
                        status=api_status,
                        response=api_response,
                        request_attempts=request_attempt,
                    )
                    return JobOutcome(job, record)
                if api_status >= 500:
                    _read_error_prefix(api_response)
                    last_error = RescueError(
                        f"GitHub job-log API returned HTTP {api_status}"
                    )
                    self.sleeper(min(2 ** (request_attempt - 1), 30))
                    continue
                if api_status == 200:
                    size, digest = self._stream_response(
                        api_response,
                        part_path,
                        budget=budget,
                    )
                    try:
                        os.replace(part_path, log_path)
                    except BaseException:
                        budget.release(size)
                        raise
                    _fsync_directory(log_path.parent)
                    return JobOutcome(
                        job,
                        {
                            "schema": JOB_RECORD_SCHEMA,
                            "source_row_sha256": source.row_sha256,
                            "jobs_sha256": source.jobs_sha256,
                            "repo": source.repo,
                            "run_id": source.run_id,
                            "attempt": source.attempt,
                            "ordinal": job.ordinal,
                            "job_id": job.job_id,
                            "job_name": job.name,
                            "endpoint": job.endpoint,
                            "member_name": job.member_name,
                            "outcome": "log",
                            "api_http_status": 200,
                            "signed_http_status": None,
                            "request_attempts": request_attempt,
                            "log": {
                                "path": (f"logs/{log_path.name}"),
                                "bytes": size,
                                "sha256": digest,
                            },
                            "terminal": None,
                        },
                    )
                if api_status != 302:
                    _read_error_prefix(api_response)
                    raise RescueError(f"GitHub job-log API returned HTTP {api_status}")
                location = api_headers.get("location")
                if not location:
                    raise UnsafeRedirectError("GitHub job-log redirect has no Location")
                signed_url = _validate_signed_redirect(location)
            except (ByteLimitError, UnsafeRedirectError) as exc:
                return JobOutcome(
                    job,
                    None,
                    type(exc).__name__,
                    _safe_error(exc, self.secrets),
                )
            except TransportExhausted as exc:
                last_error = exc
                if part_path.exists():
                    part_path.unlink()
                self.sleeper(min(2 ** (request_attempt - 1), 30))
                continue
            except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
                last_error = exc
                if part_path.exists():
                    part_path.unlink()
                self.sleeper(min(2 ** (request_attempt - 1), 30))
                continue
            except RescueError as exc:
                last_error = exc
                break
            finally:
                if api_response is not None:
                    _close_response(api_response)

            signed_request = urllib.request.Request(
                signed_url,
                headers={"User-Agent": "cppmega-ci-job-log-rescue/1"},
                method="GET",
            )
            signed_response: Any | None = None
            try:
                signed_response = self._open(signed_request)
                signed_status = _response_status(signed_response)
                if signed_status == 200:
                    size, digest = self._stream_response(
                        signed_response,
                        part_path,
                        budget=budget,
                    )
                    try:
                        os.replace(part_path, log_path)
                    except BaseException:
                        budget.release(size)
                        raise
                    _fsync_directory(log_path.parent)
                    return JobOutcome(
                        job,
                        {
                            "schema": JOB_RECORD_SCHEMA,
                            "source_row_sha256": source.row_sha256,
                            "jobs_sha256": source.jobs_sha256,
                            "repo": source.repo,
                            "run_id": source.run_id,
                            "attempt": source.attempt,
                            "ordinal": job.ordinal,
                            "job_id": job.job_id,
                            "job_name": job.name,
                            "endpoint": job.endpoint,
                            "member_name": job.member_name,
                            "outcome": "log",
                            "api_http_status": 302,
                            "signed_http_status": 200,
                            "request_attempts": request_attempt,
                            "log": {
                                "path": f"logs/{log_path.name}",
                                "bytes": size,
                                "sha256": digest,
                            },
                            "terminal": None,
                        },
                    )
                _read_error_prefix(signed_response)
                if 300 <= signed_status < 400:
                    raise UnsafeRedirectError(
                        "signed job-log response attempted another redirect"
                    )
                last_error = RescueError(
                    f"signed job-log download returned HTTP {signed_status}"
                )
            except (ByteLimitError, UnsafeRedirectError) as exc:
                return JobOutcome(
                    job,
                    None,
                    type(exc).__name__,
                    _safe_error(exc, self.secrets),
                )
            except (
                OSError,
                urllib.error.URLError,
                http.client.HTTPException,
                TransportExhausted,
            ) as exc:
                last_error = exc
            except RescueError as exc:
                last_error = exc
                break
            finally:
                if signed_response is not None:
                    _close_response(signed_response)
                if part_path.exists():
                    part_path.unlink()
            self.sleeper(min(2 ** (request_attempt - 1), 30))
        error = last_error or TransportExhausted(
            "job-log retries exhausted without a complete result"
        )
        return JobOutcome(
            job,
            None,
            type(error).__name__,
            _safe_error(error, self.secrets),
        )


class JobLogRescueWorker:
    def __init__(
        self,
        *,
        state_path: str | os.PathLike[str],
        work_dir: str | os.PathLike[str],
        rescue_spool: str | os.PathLike[str],
        tokens: Sequence[str],
        workers: int = DEFAULT_WORKERS,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_JOB_ATTEMPTS,
        max_job_bytes: int = DEFAULT_MAX_JOB_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
        opener: Callable[..., Any] = _default_opener,
        sleeper: Callable[[float], None] = time.sleep,
        before_publish: Callable[[SourceAttempt], None] | None = None,
    ):
        if workers <= 0:
            raise ValueError("workers must be positive")
        if max_total_bytes < 0 or max_zip_bytes < 0:
            raise ValueError("byte limits must be non-negative")
        self.state = FetchStateEvidence(state_path)
        try:
            self.work_dir = Path(work_dir).expanduser().resolve()
            self.rescue_spool = Path(rescue_spool).expanduser().resolve()
            if (
                self.work_dir == self.rescue_spool
                or self.rescue_spool in self.work_dir.parents
                or self.work_dir in self.rescue_spool.parents
            ):
                raise ValueError("work directory and rescue spool must be disjoint")
            _ensure_private_directory(self.work_dir)
            _ensure_private_directory(self.rescue_spool)
            self.workers = workers
            self.max_job_bytes = max_job_bytes
            self.max_total_bytes = max_total_bytes
            self.max_zip_bytes = max_zip_bytes
            self.client = JobLogClient(
                tokens,
                opener=opener,
                timeout=timeout,
                max_attempts=max_attempts,
                max_job_bytes=max_job_bytes,
                sleeper=sleeper,
            )
            self.sleeper = sleeper
            self.before_publish = before_publish
        except BaseException:
            self.state.close()
            raise

    def close(self) -> None:
        self.state.close()

    def _attempt_work_dir(self, source: SourceAttempt) -> Path:
        path = self.work_dir / source.spool_base_name / source.row_sha256
        _ensure_private_directory(path)
        _ensure_private_directory(path / "logs")
        _ensure_private_directory(path / "records")
        return path

    @staticmethod
    def _binding(source: SourceAttempt) -> dict[str, object]:
        return {
            "schema": BINDING_SCHEMA,
            "state_path": str(source.state_path),
            "source": {
                "repo": source.repo,
                "canonical_repo": source.canonical_repo,
                "run_id": source.run_id,
                "attempt": source.attempt,
                "created_at": source.created_at,
                "status": source.status,
                "tries": source.tries,
                "error_class": source.error_class,
                "failed_raw_archive": {
                    "source": source.archive_source,
                    "sha256": source.archive_sha256,
                    "bytes": source.archive_size,
                    "preservation": "source fetcher artifact is not modified",
                },
                "row_sha256": source.row_sha256,
                "run_metadata_sha256": source.run_metadata_sha256,
                "run_metadata_raw_size": source.run_metadata_raw_size,
                "jobs_sha256": source.jobs_sha256,
                "jobs_raw_size": source.jobs_raw_size,
                "jobs_ledger_sha256": source.jobs_ledger_sha256,
                "jobs_ledger_ids": list(source.jobs_ledger_ids),
                "job_count": len(source.jobs),
            },
        }

    def _ensure_binding(self, source: SourceAttempt, root: Path) -> None:
        path = root / "binding.json"
        expected = _canonical_json_bytes(self._binding(source)) + b"\n"
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise StateBindingError("rescue binding path is unsafe")
            if path.read_bytes() != expected:
                raise StateBindingError("rescue work binding changed")
            return
        _atomic_write_bytes(path, expected)

    @staticmethod
    def _record_path(root: Path, job: JobEvidence) -> Path:
        return root / "records" / f"{job.ordinal:06d}--{job.job_id}.json"

    @staticmethod
    def _log_path(root: Path, job: JobEvidence) -> Path:
        return root / "logs" / f"{job.ordinal:06d}--{job.job_id}.log"

    @staticmethod
    def _record_log_size(record: Mapping[str, object]) -> int:
        value = record.get("log")
        if not isinstance(value, Mapping):
            raise StateBindingError("resolved job record has no log metadata")
        size = value.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise StateBindingError("resolved job log size is invalid")
        return size

    def _validate_record(
        self,
        source: SourceAttempt,
        job: JobEvidence,
        root: Path,
        record: object,
    ) -> dict[str, object]:
        if not isinstance(record, dict):
            raise StateBindingError("job rescue record is not an object")
        expected_keys = {
            "schema",
            "source_row_sha256",
            "jobs_sha256",
            "repo",
            "run_id",
            "attempt",
            "ordinal",
            "job_id",
            "job_name",
            "endpoint",
            "member_name",
            "outcome",
            "api_http_status",
            "signed_http_status",
            "request_attempts",
            "log",
            "terminal",
        }
        if set(record) != expected_keys:
            raise StateBindingError("job rescue record fields are invalid")
        required_identity = {
            "schema": JOB_RECORD_SCHEMA,
            "source_row_sha256": source.row_sha256,
            "jobs_sha256": source.jobs_sha256,
            "repo": source.repo,
            "run_id": source.run_id,
            "attempt": source.attempt,
            "ordinal": job.ordinal,
            "job_id": job.job_id,
            "job_name": job.name,
            "endpoint": job.endpoint,
        }
        for key, expected in required_identity.items():
            if record.get(key) != expected:
                raise StateBindingError(
                    f"job {job.job_id} rescue record binding changed"
                )
        outcome = record.get("outcome")
        log_value = record.get("log")
        terminal = record.get("terminal")
        request_attempts = record.get("request_attempts")
        if (
            isinstance(request_attempts, bool)
            or not isinstance(request_attempts, int)
            or request_attempts <= 0
        ):
            raise StateBindingError("job rescue request count is invalid")
        if outcome == "log":
            if (
                record.get("member_name") != job.member_name
                or not isinstance(log_value, dict)
                or set(log_value) != {"path", "bytes", "sha256"}
                or terminal is not None
                or (
                    (
                        record.get("api_http_status"),
                        record.get("signed_http_status"),
                    )
                    not in {(200, None), (302, 200)}
                )
            ):
                raise StateBindingError("resolved log record is malformed")
            relative = log_value.get("path")
            size = log_value.get("bytes")
            digest = log_value.get("sha256")
            expected_relative = f"logs/{self._log_path(root, job).name}"
            if (
                not isinstance(relative, str)
                or relative != expected_relative
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or size > self.max_job_bytes
                or not isinstance(digest, str)
                or _SHA256_RE.fullmatch(digest) is None
            ):
                raise StateBindingError("resolved log metadata is malformed")
            log_path = root / PurePosixPath(relative)
            if (
                log_path.is_symlink()
                or not log_path.is_file()
                or log_path.stat().st_size != size
                or _sha256_file(log_path) != digest
            ):
                raise StateBindingError(f"resolved job log {job.job_id} changed")
        elif outcome in {"terminal_404", "terminal_410"}:
            expected_status = int(str(outcome).rsplit("_", 1)[-1])
            if (
                record.get("member_name") is not None
                or log_value is not None
                or not isinstance(terminal, dict)
                or set(terminal)
                != {
                    "http_status",
                    "body_prefix_bytes",
                    "body_prefix_sha256",
                    "body_truncated",
                }
                or terminal.get("http_status") != expected_status
                or record.get("api_http_status") != expected_status
                or record.get("signed_http_status") is not None
                or isinstance(terminal.get("body_prefix_bytes"), bool)
                or not isinstance(terminal.get("body_prefix_bytes"), int)
                or int(terminal["body_prefix_bytes"]) < 0
                or int(terminal["body_prefix_bytes"]) > _ERROR_BODY_PREFIX_BYTES
                or not isinstance(terminal.get("body_prefix_sha256"), str)
                or _SHA256_RE.fullmatch(str(terminal["body_prefix_sha256"])) is None
                or not isinstance(terminal.get("body_truncated"), bool)
            ):
                raise StateBindingError("terminal job rescue record is malformed")
        else:
            raise StateBindingError("job rescue outcome is invalid")
        return dict(record)

    def _load_records(
        self,
        source: SourceAttempt,
        root: Path,
    ) -> dict[int, dict[str, object]]:
        records: dict[int, dict[str, object]] = {}
        for job in source.jobs:
            record_path = self._record_path(root, job)
            if not record_path.exists():
                continue
            if record_path.is_symlink() or not record_path.is_file():
                raise StateBindingError("job record path is unsafe")
            raw = record_path.read_bytes()
            try:
                value = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise StateBindingError("job record is not JSON") from exc
            if raw != _canonical_json_bytes(value) + b"\n":
                raise StateBindingError("job record is not canonical JSON")
            records[job.job_id] = self._validate_record(
                source,
                job,
                root,
                value,
            )
        return records

    def _store_record(
        self,
        source: SourceAttempt,
        root: Path,
        outcome: JobOutcome,
    ) -> dict[str, object]:
        assert outcome.record is not None
        record = self._validate_record(
            source,
            outcome.job,
            root,
            outcome.record,
        )
        path = self._record_path(root, outcome.job)
        expected = _canonical_json_bytes(record) + b"\n"
        if path.exists():
            if path.read_bytes() != expected:
                raise StateBindingError("job record replay changed")
        else:
            _atomic_write_bytes(path, expected)
        return record

    @staticmethod
    def _resolved_jsonl(
        source: SourceAttempt,
        records: Mapping[int, Mapping[str, object]],
    ) -> bytes:
        lines = []
        for job in source.jobs:
            record = records[job.job_id]
            lines.append(
                _canonical_json_bytes(
                    {
                        "schema": RESOLVED_JOBS_SCHEMA,
                        "source_row_sha256": source.row_sha256,
                        "jobs_sha256": source.jobs_sha256,
                        "ordinal": job.ordinal,
                        "job_id": job.job_id,
                        "job_name": job.name,
                        "member_name": record["member_name"],
                        "outcome": record["outcome"],
                        "api_http_status": record["api_http_status"],
                        "signed_http_status": record["signed_http_status"],
                        "log": record["log"],
                        "terminal": record["terminal"],
                    }
                )
            )
        return b"".join(line + b"\n" for line in lines)

    def _build_zip(
        self,
        source: SourceAttempt,
        root: Path,
        records: Mapping[int, Mapping[str, object]],
    ) -> tuple[Path, int, str, int, int]:
        destination = root / "synthetic.zip"
        temporary = destination.with_name(".synthetic.zip.building")
        if temporary.is_symlink():
            raise StateBindingError("synthetic ZIP temporary path is unsafe")
        if temporary.exists():
            if not temporary.is_file():
                raise StateBindingError("synthetic ZIP temporary path is not a file")
            temporary.unlink()
        log_jobs = [
            job for job in source.jobs if records[job.job_id]["outcome"] == "log"
        ]
        with temporary.open("xb") as output:
            os.fchmod(output.fileno(), 0o600)
            bounded_output = _BoundedSeekableWriter(
                output,
                self.max_zip_bytes,
            )
            with zipfile.ZipFile(
                cast(BinaryIO, bounded_output),
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                strict_timestamps=True,
            ) as archive:
                for job in log_jobs:
                    info = zipfile.ZipInfo(
                        job.member_name,
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    info.extra = b""
                    info.comment = b""
                    with self._log_path(root, job).open("rb") as source_log:
                        with archive.open(info, "w", force_zip64=True) as member:
                            while block := source_log.read(_STREAM_CHUNK_BYTES):
                                member.write(block)
            output.flush()
            os.fsync(output.fileno())
        size = temporary.stat().st_size
        if size > self.max_zip_bytes:
            raise ByteLimitError(
                f"synthetic ZIP size {size} exceeds {self.max_zip_bytes}"
            )
        try:
            _safe_zip_infos(
                temporary,
                max_members=len(source.jobs),
                max_member_bytes=self.max_job_bytes,
                max_uncompressed_bytes=self.max_total_bytes,
            )
        except FetchArchiveError as exc:
            raise StateBindingError(
                f"synthetic ZIP safety validation failed: {exc}"
            ) from exc
        digest = _sha256_file(temporary)
        if destination.exists():
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.stat().st_size != size
                or _sha256_file(destination) != digest
            ):
                raise StateBindingError("deterministic ZIP replay changed")
            temporary.unlink()
        else:
            os.replace(temporary, destination)
            _fsync_directory(root)
        return (
            destination,
            size,
            digest,
            len(log_jobs),
            sum(self._record_log_size(records[job.job_id]) for job in log_jobs),
        )

    @staticmethod
    def _receipt(
        source: SourceAttempt,
        *,
        completed_at: str,
        resolved_path: Path,
        zip_path: Path,
        records: Mapping[int, Mapping[str, object]],
        zip_member_count: int,
        uncompressed_log_bytes: int,
    ) -> dict[str, object]:
        counts = {"log": 0, "terminal_404": 0, "terminal_410": 0}
        for record in records.values():
            counts[str(record["outcome"])] += 1
        return {
            "schema": RECEIPT_SCHEMA,
            "completed_at": completed_at,
            "source_state": {
                "path": str(source.state_path),
                "repo": source.repo,
                "canonical_repo": source.canonical_repo,
                "run_id": source.run_id,
                "attempt": source.attempt,
                "created_at": source.created_at,
                "status": source.status,
                "tries": source.tries,
                "error_class": source.error_class,
                "failed_raw_archive": {
                    "source": source.archive_source,
                    "sha256": source.archive_sha256,
                    "bytes": source.archive_size,
                    "preservation": "source fetcher artifact is not modified",
                },
                "attempt_row_sha256": source.row_sha256,
                "run_metadata_sha256": source.run_metadata_sha256,
                "run_metadata_raw_size": source.run_metadata_raw_size,
                "jobs_sha256": source.jobs_sha256,
                "jobs_raw_size": source.jobs_raw_size,
                "jobs_ledger_sha256": source.jobs_ledger_sha256,
                "jobs_ledger_ids": list(source.jobs_ledger_ids),
            },
            "coverage": {
                "expected_jobs": len(source.jobs),
                "resolved_jobs": len(records),
                "unresolved_jobs": 0,
                "full_logs": counts["log"],
                "terminal_404": counts["terminal_404"],
                "terminal_410": counts["terminal_410"],
                "zip_members": zip_member_count,
                "uncompressed_log_bytes": uncompressed_log_bytes,
            },
            "artifacts": {
                "resolved_jobs": {
                    "name": "resolved_jobs.jsonl",
                    "bytes": resolved_path.stat().st_size,
                    "sha256": _sha256_file(resolved_path),
                },
                "synthetic_zip": {
                    "name": "synthetic.zip",
                    "bytes": zip_path.stat().st_size,
                    "sha256": _sha256_file(zip_path),
                },
            },
        }

    @staticmethod
    def _atomic_publish_file(source: Path, destination: Path) -> None:
        if destination.exists():
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.stat().st_size != source.stat().st_size
                or _sha256_file(destination) != _sha256_file(source)
            ):
                raise StateBindingError(
                    f"conflicting rescue spool artifact: {destination.name}"
                )
            return
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        try:
            try:
                os.link(source, temporary)
            except OSError as exc:
                if exc.errno not in {
                    errno.EXDEV,
                    errno.EPERM,
                    errno.EACCES,
                    errno.ENOTSUP,
                }:
                    raise
                with source.open("rb") as reader:
                    descriptor = os.open(
                        temporary,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                    try:
                        with os.fdopen(descriptor, "wb", closefd=False) as writer:
                            shutil.copyfileobj(
                                reader, writer, length=_STREAM_CHUNK_BYTES
                            )
                            writer.flush()
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _update_manifest(
        self,
        source: SourceAttempt,
        *,
        zip_size: int,
        zip_sha256: str,
        completed_at: str,
    ) -> None:
        path = self.rescue_spool / "manifest.tsv"
        header = "\t".join(_MANIFEST_FIELDS)
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise StateBindingError("rescue manifest path is unsafe")
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines or lines[0] != header:
                raise StateBindingError("rescue manifest header is incompatible")
        else:
            lines = [header]
        record = {
            "repo": source.repo,
            "run_id": str(source.run_id),
            "attempt": str(source.attempt),
            "created_at": source.created_at,
            "status": "zip",
            "bytes": str(zip_size),
            "sha256": zip_sha256,
            "finished_at": completed_at,
        }
        rendered = "\t".join(record[field] for field in _MANIFEST_FIELDS)
        if rendered not in lines[1:]:
            lines.append(rendered)
        _atomic_write_bytes(
            path,
            ("\n".join(lines) + "\n").encode("utf-8"),
        )

    def _publish(
        self,
        source: SourceAttempt,
        *,
        receipt_path: Path,
        resolved_path: Path,
        zip_path: Path,
        completed_at: str,
    ) -> None:
        base = source.spool_base_name
        # The ZIP is the fetcher's visibility/commit marker, so publish both
        # auditable sidecars and the manifest row before publishing it last.
        self._atomic_publish_file(
            resolved_path,
            self.rescue_spool / f"{base}.resolved_jobs.jsonl",
        )
        self._atomic_publish_file(
            receipt_path,
            self.rescue_spool / f"{base}.receipt.json",
        )
        self._update_manifest(
            source,
            zip_size=zip_path.stat().st_size,
            zip_sha256=_sha256_file(zip_path),
            completed_at=completed_at,
        )
        self._atomic_publish_file(
            zip_path,
            self.rescue_spool / f"{base}.zip",
        )

    def _write_progress(
        self,
        root: Path,
        source: SourceAttempt,
        *,
        resolved: int,
        unresolved: Sequence[JobOutcome],
    ) -> dict[str, object]:
        value = {
            "schema": PROGRESS_SCHEMA,
            "updated_at": _utc_now(),
            "source_row_sha256": source.row_sha256,
            "repo": source.repo,
            "run_id": source.run_id,
            "attempt": source.attempt,
            "expected_jobs": len(source.jobs),
            "resolved_jobs": resolved,
            "unresolved_jobs": len(unresolved),
            "unresolved": [
                {
                    "ordinal": item.job.ordinal,
                    "job_id": item.job.job_id,
                    "error_class": item.error_class,
                    "error_message": _safe_error(
                        item.error_message or "unresolved",
                        self.client.secrets,
                    ),
                }
                for item in unresolved
            ],
        }
        _atomic_write_json(root / "progress.json", value)
        return value

    def rescue_attempt(
        self,
        identity: tuple[str, int, int],
        *,
        explicit: bool,
    ) -> dict[str, object]:
        source = self.state.load_attempt(identity, explicit=explicit)
        root = self._attempt_work_dir(source)
        self._ensure_binding(source, root)
        records = self._load_records(source, root)
        initial_bytes = sum(
            self._record_log_size(record)
            for record in records.values()
            if record["outcome"] == "log"
        )
        if initial_bytes > self.max_total_bytes:
            raise ByteLimitError(
                f"durable resolved logs use {initial_bytes} bytes, exceeding "
                f"aggregate limit {self.max_total_bytes}"
            )
        budget = ByteBudget(self.max_total_bytes, initial=initial_bytes)
        missing = [job for job in source.jobs if job.job_id not in records]
        outcomes: list[JobOutcome] = []
        with ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="ci-job-rescue",
        ) as executor:
            futures: dict[Future[JobOutcome], JobEvidence] = {
                executor.submit(
                    self.client.fetch,
                    source,
                    job,
                    log_path=self._log_path(root, job),
                    budget=budget,
                ): job
                for job in missing
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    outcome = future.result()
                except BaseException as exc:
                    outcome = JobOutcome(
                        job,
                        None,
                        type(exc).__name__,
                        _safe_error(exc, self.client.secrets),
                    )
                outcomes.append(outcome)
                if outcome.resolved:
                    records[job.job_id] = self._store_record(
                        source,
                        root,
                        outcome,
                    )
        unresolved = sorted(
            (item for item in outcomes if not item.resolved),
            key=lambda item: item.job.ordinal,
        )
        if unresolved or len(records) != len(source.jobs):
            known_unresolved = {item.job.job_id for item in unresolved}
            for job in source.jobs:
                if job.job_id not in records and job.job_id not in known_unresolved:
                    unresolved.append(
                        JobOutcome(
                            job,
                            None,
                            "UnresolvedJob",
                            "job has no durable terminal result",
                        )
                    )
            progress = self._write_progress(
                root,
                source,
                resolved=len(records),
                unresolved=unresolved,
            )
            return {"status": "unresolved", **progress}
        resolved_bytes = self._resolved_jsonl(source, records)
        resolved_path = root / "resolved_jobs.jsonl"
        _atomic_write_bytes(resolved_path, resolved_bytes)
        (
            zip_path,
            _zip_size,
            _zip_sha,
            zip_member_count,
            uncompressed_log_bytes,
        ) = self._build_zip(source, root, records)
        completed_at = _utc_now()
        receipt_path = root / "completion_receipt.json"
        if receipt_path.exists():
            try:
                receipt = json.loads(receipt_path.read_bytes())
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise StateBindingError("completion receipt is not JSON") from exc
            if not isinstance(receipt, dict):
                raise StateBindingError("completion receipt is not an object")
            completed_at_value = receipt.get("completed_at")
            if not isinstance(completed_at_value, str):
                raise StateBindingError("completion receipt lacks completed_at")
            completed_at = completed_at_value
        receipt = self._receipt(
            source,
            completed_at=completed_at,
            resolved_path=resolved_path,
            zip_path=zip_path,
            records=records,
            zip_member_count=zip_member_count,
            uncompressed_log_bytes=uncompressed_log_bytes,
        )
        encoded_receipt = _canonical_json_bytes(receipt) + b"\n"
        if receipt_path.exists():
            if receipt_path.read_bytes() != encoded_receipt:
                raise StateBindingError("completion receipt replay changed")
        else:
            _atomic_write_bytes(receipt_path, encoded_receipt)
        if self.before_publish is not None:
            self.before_publish(source)
        receipt_sha = _sha256_file(receipt_path)
        self.state.commit_rescue(
            source,
            receipt_sha256=receipt_sha,
            publish=lambda: self._publish(
                source,
                receipt_path=receipt_path,
                resolved_path=resolved_path,
                zip_path=zip_path,
                completed_at=completed_at,
            ),
        )
        progress = self._write_progress(
            root,
            source,
            resolved=len(records),
            unresolved=[],
        )
        return {
            "status": "complete",
            "receipt_sha256": receipt_sha,
            "receipt": receipt,
            **progress,
        }

    def _completed_result(
        self,
        identity: tuple[str, int, int],
    ) -> dict[str, object] | None:
        audit = self.state.completed_rescue_audit(identity)
        if audit is None:
            return None
        receipt_sha, source_row_sha = audit
        repo, run_id, attempt = identity
        base = f"{repo.replace('/', '__')}--{run_id}--attempt-{attempt}"
        receipt_path = self.rescue_spool / f"{base}.receipt.json"
        zip_candidates = (
            self.rescue_spool / f"{base}.zip",
            self.rescue_spool / "consumed" / f"{base}.zip",
        )
        zip_path = next(
            (
                path
                for path in zip_candidates
                if path.is_file() and not path.is_symlink()
            ),
            zip_candidates[0],
        )
        if (
            receipt_path.is_symlink()
            or not receipt_path.is_file()
            or _sha256_file(receipt_path) != receipt_sha
            or zip_path.is_symlink()
            or not zip_path.is_file()
        ):
            raise StateBindingError(
                "prior job-rescue ledger has no matching spool artifacts"
            )
        receipt_raw = receipt_path.read_bytes()
        try:
            receipt = json.loads(receipt_raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StateBindingError("prior job-rescue receipt is not JSON") from exc
        if (
            not isinstance(receipt, dict)
            or receipt_raw != _canonical_json_bytes(receipt) + b"\n"
        ):
            raise StateBindingError("prior job-rescue receipt encoding is invalid")
        source_value = receipt.get("source_state")
        artifacts = receipt.get("artifacts")
        if (
            receipt.get("schema") != RECEIPT_SCHEMA
            or not isinstance(source_value, dict)
            or source_value.get("attempt_row_sha256") != source_row_sha
            or source_value.get("repo") != repo
            or source_value.get("run_id") != run_id
            or source_value.get("attempt") != attempt
            or not isinstance(artifacts, dict)
            or not isinstance(artifacts.get("synthetic_zip"), dict)
            or not isinstance(artifacts.get("resolved_jobs"), dict)
        ):
            raise StateBindingError("prior job-rescue receipt binding changed")
        zip_value = artifacts["synthetic_zip"]
        if zip_value.get("bytes") != zip_path.stat().st_size or zip_value.get(
            "sha256"
        ) != _sha256_file(zip_path):
            raise StateBindingError("prior synthetic ZIP changed")
        resolved_path = self.rescue_spool / f"{base}.resolved_jobs.jsonl"
        resolved_value = artifacts["resolved_jobs"]
        if (
            resolved_path.is_symlink()
            or not resolved_path.is_file()
            or resolved_value.get("bytes") != resolved_path.stat().st_size
            or resolved_value.get("sha256") != _sha256_file(resolved_path)
        ):
            raise StateBindingError("prior resolved-jobs evidence changed")
        return {
            "status": "complete",
            "idempotent_replay": True,
            "receipt_sha256": receipt_sha,
            "receipt": receipt,
        }

    def run_once(
        self,
        *,
        target: tuple[str, int, int] | None = None,
    ) -> dict[str, object]:
        identities = self.state.scan(target=target)
        results: list[dict[str, object]] = []
        failures = 0
        for identity in identities:
            try:
                result = self._completed_result(identity)
                if result is None:
                    result = self.rescue_attempt(
                        identity,
                        explicit=target is not None,
                    )
            except RescueError as exc:
                result = {
                    "status": "error",
                    "repo": identity[0],
                    "run_id": identity[1],
                    "attempt": identity[2],
                    "error_class": type(exc).__name__,
                    "error_message": _safe_error(exc, self.client.secrets),
                }
            if result["status"] != "complete":
                failures += 1
            results.append(result)
        return {
            "schema": PROGRESS_SCHEMA,
            "scanned_attempts": len(identities),
            "failed_attempts": failures,
            "results": results,
        }

    def run(
        self,
        *,
        target: tuple[str, int, int] | None = None,
        continuous: bool = False,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> dict[str, object]:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        while True:
            result = self.run_once(target=target)
            if not continuous:
                return result
            if target is not None and result["failed_attempts"] == 0:
                return result
            self.sleeper(poll_seconds)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Rescue failed ci_stream_fetch archives via per-job log downloads")
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--rescue-spool", required=True)
    parser.add_argument("--tokens")
    parser.add_argument("--repo")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--request-attempts",
        type=int,
        default=DEFAULT_JOB_ATTEMPTS,
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--max-job-bytes",
        type=int,
        default=DEFAULT_MAX_JOB_BYTES,
    )
    parser.add_argument(
        "--max-total-bytes",
        type=int,
        default=DEFAULT_MAX_TOTAL_BYTES,
    )
    parser.add_argument(
        "--max-zip-bytes",
        type=int,
        default=DEFAULT_MAX_ZIP_BYTES,
    )
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    target_values = (args.repo, args.run_id, args.attempt)
    if any(value is not None for value in target_values) and not all(
        value is not None for value in target_values
    ):
        print(
            "[ci-job-log-rescue] ERROR: --repo, --run-id, and --attempt "
            "must be provided together",
            file=sys.stderr,
        )
        return 2
    if args.run_id is not None and args.run_id <= 0:
        print(
            "[ci-job-log-rescue] ERROR: --run-id must be positive",
            file=sys.stderr,
        )
        return 2
    if args.attempt is not None and args.attempt <= 0:
        print(
            "[ci-job-log-rescue] ERROR: --attempt must be positive",
            file=sys.stderr,
        )
        return 2
    target = (
        None
        if args.repo is None
        else (str(args.repo), int(args.run_id), int(args.attempt))
    )
    worker: JobLogRescueWorker | None = None
    try:
        tokens = load_token_pool(args.tokens)
        worker = JobLogRescueWorker(
            state_path=args.state,
            work_dir=args.work_dir,
            rescue_spool=args.rescue_spool,
            tokens=tokens,
            workers=args.workers,
            timeout=args.timeout,
            max_attempts=args.request_attempts,
            max_job_bytes=args.max_job_bytes,
            max_total_bytes=args.max_total_bytes,
            max_zip_bytes=args.max_zip_bytes,
        )
        result = worker.run(
            target=target,
            continuous=args.continuous,
            poll_seconds=args.poll_seconds,
        )
    except (
        OSError,
        sqlite3.Error,
        InventoryError,
        RescueError,
        ValueError,
    ) as exc:
        print(
            f"[ci-job-log-rescue] ERROR: {_safe_error(exc)}",
            file=sys.stderr,
        )
        return 1
    finally:
        if worker is not None:
            worker.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    failed_attempts = result.get("failed_attempts")
    if isinstance(failed_attempts, bool) or not isinstance(failed_attempts, int):
        print(
            "[ci-job-log-rescue] ERROR: invalid worker result",
            file=sys.stderr,
        )
        return 1
    return 1 if failed_attempts else 0


if __name__ == "__main__":
    raise SystemExit(main())
