#!/usr/bin/env python3
"""Stream GitHub Actions logs into the exact-deduplicated CI content store.

The inventory stage and this fetch stage deliberately use separate SQLite
databases.  The inventory may continue adding immutable run identities while
this process consumes the oldest visible runs.  A content-store commit happens
before an attempt is marked complete, so replay after a crash is idempotent.

Only canonical, secret-redacted payload chunks enter the content store.  Raw
ZIP archives are bounded temporary inputs.  A separately created rescue spool
can be imported through the same validation/parser/tokenizer path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci_content_store import (  # noqa: E402
    CIContentStore,
    PRODUCTION_TARGET_UNIQUE_TOKENS,
    hash_token_sequence,
)
from scripts.ci_log_sidecars import canonicalize_ci_log  # noqa: E402
from scripts.ci_stream_inventory import (  # noqa: E402
    GITHUB_API_VERSION,
    HTTPResponse,
    TokenPool,
    load_token_pool,
)
from cppmega.data.tokenizer_contract import (  # noqa: E402
    TOKENIZER_CONTRACT_SHA256,
)
from cppmega.tokenizer.cpp_tokenizer import (  # noqa: E402
    TokenizerContractError,
    load_cppmega_tokenizer,
)


SCHEMA_VERSION = "cppmega_ci_stream_fetch_v3"
PROGRESS_SCHEMA = "cppmega_ci_stream_fetch_progress_v3"
RECEIPT_SCHEMA = "cppmega_ci_stream_fetch_receipt_v3"
DEFAULT_TOKENIZER = (
    "../cppmega.mlx/outputs/megatron_ready/"
    "case5_v4_20260714_093120_mini9/tokenizer/tokenizer.json"
)
DEFAULT_TARGET = PRODUCTION_TARGET_UNIQUE_TOKENS
DEFAULT_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_MEMBER_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_MEMBERS = 20_000
DEFAULT_MAX_CHUNK_CHARS = 128_000
DEFAULT_DISCOVERY_ROWS = 20_000
DEFAULT_API_ATTEMPTS = 12
DEFAULT_ARCHIVE_TRANSFER_ATTEMPTS = 16
DEFAULT_TIMEOUT = 90.0

_RUN_ATTEMPT_STATES = {
    "pending",
    "processing",
    "retry",
    "done",
    "empty",
    "terminal_404",
    "terminal_410",
    "failed",
}
_TERMINAL_STATES = {
    "done",
    "empty",
    "terminal_404",
    "terminal_410",
    "failed",
}
_RUN_METADATA_SOURCES = {
    "inventory-run-list",
    "github-workflow-run-attempt-api",
}
_MAIN_MEMBER_RE = re.compile(r"^(?P<ordinal>\d+)_(?P<name>.+)\.txt$")
_SECRET_QUERY_KEYS = {
    "sig",
    "signature",
    "token",
    "se",
    "sp",
    "sv",
    "srt",
    "spr",
}


class FetchError(RuntimeError):
    """Base fail-closed fetch error."""


class BindingError(FetchError):
    """Durable state does not match the current producer contract."""


class APIError(FetchError):
    """A GitHub API operation exhausted its safe retry policy."""


class MalformedResponseError(APIError):
    """A response cannot prove the expected endpoint contract."""


class ArchiveError(FetchError):
    """A workflow log archive violates a safety or conservation rule."""


class TerminalHTTP(FetchError):
    """An immutable endpoint result proves that no archive can be fetched."""

    def __init__(self, status: int, body: bytes, endpoint: str):
        super().__init__(f"GitHub HTTP {status} for {endpoint}")
        self.status = status
        self.body = body
        self.endpoint = endpoint


@dataclass(frozen=True)
class Attempt:
    repo: str
    run_id: int
    attempt: int
    created_at: str
    run_metadata: dict[str, Any]
    run_metadata_sha256: str
    run_metadata_source: str
    run_metadata_source_attempt: int
    run_metadata_exact: bool
    inventory_seed_attempt: int
    inventory_seed_metadata_sha256: str

    @property
    def run_attempt_key(self) -> str:
        return f"{self.run_id}:{self.attempt}"


@dataclass(frozen=True)
class ArchiveSource:
    path: Path
    source: str
    raw_sha256: str
    raw_size: int
    recoverable: bool


@dataclass(frozen=True)
class PreparedArchive:
    repository: str
    run_id: int
    attempt: int
    source: str
    inline_body: bytes | None
    signed_url: str | None


@dataclass(frozen=True)
class RequestResult:
    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class RepositoryIdentity:
    requested: str
    canonical: str
    repository_id: int | None
    source: str
    source_repository_id: int | None


_REPOSITORY_FULL_NAME_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/"
    r"[A-Za-z0-9_.-]+"
)


def _repository_object_identity(
    value: object,
    *,
    field: str,
) -> tuple[str, int | None] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise MalformedResponseError(
            f"run metadata {field} is not an object"
        )
    full_name = value.get("full_name")
    if (
        not isinstance(full_name, str)
        or _REPOSITORY_FULL_NAME_RE.fullmatch(full_name) is None
    ):
        raise MalformedResponseError(
            f"run metadata {field}.full_name is invalid"
        )
    raw_id = value.get("id")
    if raw_id is None:
        repository_id = None
    elif (
        isinstance(raw_id, bool)
        or not isinstance(raw_id, int)
        or raw_id <= 0
    ):
        raise MalformedResponseError(
            f"run metadata {field}.id is invalid"
        )
    else:
        repository_id = raw_id
    return full_name, repository_id


def _repository_identity(attempt: Attempt) -> RepositoryIdentity:
    canonical = _repository_object_identity(
        attempt.run_metadata.get("repository"),
        field="repository",
    )
    source = _repository_object_identity(
        attempt.run_metadata.get("head_repository"),
        field="head_repository",
    )
    canonical_name, repository_id = (
        (attempt.repo, None) if canonical is None else canonical
    )
    source_name, source_repository_id = (
        (canonical_name, repository_id) if source is None else source
    )
    return RepositoryIdentity(
        requested=attempt.repo,
        canonical=canonical_name,
        repository_id=repository_id,
        source=source_name,
        source_repository_id=source_repository_id,
    )


def _validate_run_metadata_identity(
    value: Mapping[str, object],
    *,
    run_id: int,
    attempt: int,
) -> None:
    metadata_run_id = value.get("id")
    metadata_attempt = value.get("run_attempt")
    if (
        isinstance(metadata_run_id, bool)
        or not isinstance(metadata_run_id, int)
        or metadata_run_id != run_id
    ):
        raise MalformedResponseError(
            f"run metadata id {metadata_run_id!r} does not match {run_id}"
        )
    if (
        isinstance(metadata_attempt, bool)
        or not isinstance(metadata_attempt, int)
        or metadata_attempt != attempt
    ):
        raise MalformedResponseError(
            "run metadata attempt "
            f"{metadata_attempt!r} does not match {attempt}"
        )
    created_at = value.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise MalformedResponseError("run metadata created_at is invalid")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _script_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _parser_sha256() -> str:
    import scripts.ci_log_sidecars as parser_module

    return _sha256_file(Path(parser_module.__file__).resolve())


def _content_store_sha256() -> str:
    import scripts.ci_content_store as store_module

    return _sha256_file(Path(store_module.__file__).resolve())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | os.PathLike[str], value: object) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_error(value: object, secrets: Iterable[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text[:4000]


def _safe_url_for_ledger(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if not parsed.query:
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, "", "")
        )
    keys = []
    for key, _ in urllib.parse.parse_qsl(
        parsed.query, keep_blank_values=True
    ):
        keys.append("<redacted>" if key.casefold() in _SECRET_QUERY_KEYS else key)
    query = "&".join(f"{key}=<redacted>" for key in keys)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, query, "")
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _default_no_redirect_requester(
    method: str,
    url: str,
    headers: Mapping[str, str],
    timeout: float,
) -> HTTPResponse:
    request = urllib.request.Request(
        url, headers=dict(headers), method=method
    )
    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
            return HTTPResponse(
                status=int(response.status),
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HTTPResponse(
            status=int(exc.code),
            headers=dict(exc.headers.items()) if exc.headers is not None else {},
            body=exc.read(),
        )


def _default_archive_downloader(
    url: str,
    destination: Path,
    *,
    timeout: float,
    max_bytes: int,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    max_transfer_attempts: int = DEFAULT_ARCHIVE_TRANSFER_ATTEMPTS,
) -> tuple[int, str]:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ArchiveError("GitHub returned an unsafe signed archive URL")
    if (
        isinstance(max_transfer_attempts, bool)
        or not isinstance(max_transfer_attempts, int)
        or max_transfer_attempts < 1
    ):
        raise ValueError("max_transfer_attempts must be a positive integer")
    if destination.exists() or destination.is_symlink():
        raise ArchiveError("archive download destination already exists")

    open_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        destination_fd = os.open(destination, open_flags, 0o600)
    except FileExistsError as exc:
        raise ArchiveError(
            "archive download destination already exists"
        ) from exc
    except OSError as exc:
        raise ArchiveError(
            "archive download destination could not be created safely"
        ) from exc

    created_stat = os.fstat(destination_fd)
    created_identity = (created_stat.st_dev, created_stat.st_ino)

    def verify_destination_identity() -> os.stat_result:
        current = os.fstat(destination_fd)
        try:
            visible = destination.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArchiveError(
                "archive download destination identity changed"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or (current.st_dev, current.st_ino) != created_identity
            or (visible.st_dev, visible.st_ino) != created_identity
        ):
            raise ArchiveError(
                "archive download destination identity changed"
            )
        return current

    def strong_etag(headers: Mapping[str, str]) -> str | None:
        value = headers.get("ETag")
        if value is None:
            return None
        value = value.strip()
        if (
            len(value) < 2
            or not value.startswith('"')
            or not value.endswith('"')
            or value[:2].casefold() == "w/"
        ):
            return None
        return value

    def write_all(payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:
                raise OSError("archive destination write made no progress")
            view = view[written:]

    def digest_created_file() -> str:
        verify_destination_identity()
        os.lseek(destination_fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while block := os.read(destination_fd, 1024 * 1024):
            digest.update(block)
        verify_destination_identity()
        return digest.hexdigest()

    expected_total: int | None = None
    validator: str | None = None
    last_transport_error: BaseException | None = None
    try:
        for transfer_attempt in range(1, max_transfer_attempts + 1):
            current = verify_destination_identity()
            offset = current.st_size
            resume_offset = offset if offset and validator is not None else 0

            # Deliberately no Authorization header: the signed URL is the
            # credential. A byte-range append is allowed only with a strong
            # ETag for this exact signed representation.
            headers = {"User-Agent": "cppmega-ci-stream-fetch/1"}
            if resume_offset:
                headers["Range"] = f"bytes={resume_offset}-"
                headers["If-Range"] = validator
            request = urllib.request.Request(
                url,
                headers=headers,
                method="GET",
            )
            try:
                with urlopen(request, timeout=timeout) as response:
                    status_code = int(response.status)
                    response_validator = strong_etag(response.headers)
                    response_end_exclusive: int | None = None
                    append = resume_offset > 0 and status_code == 206
                    if append:
                        content_range = response.headers.get("Content-Range")
                        match = (
                            None
                            if content_range is None
                            else re.fullmatch(
                                r"bytes ([0-9]+)-([0-9]+)/([0-9]+)",
                                content_range.strip(),
                            )
                        )
                        if match is None:
                            raise MalformedResponseError(
                                "signed archive resume lacks a valid "
                                "Content-Range"
                            )
                        start, end, total = (
                            int(value) for value in match.groups()
                        )
                        if (
                            start != resume_offset
                            or end < start
                            or end >= total
                            or total > max_bytes
                            or (
                                expected_total is not None
                                and total != expected_total
                            )
                            or response_validator != validator
                        ):
                            raise MalformedResponseError(
                                "signed archive resumed byte range is "
                                "inconsistent"
                            )
                        content_length = response.headers.get(
                            "Content-Length"
                        )
                        if content_length is not None:
                            try:
                                remaining = int(content_length)
                            except ValueError as exc:
                                raise MalformedResponseError(
                                    "signed archive Content-Length is not an "
                                    "integer"
                                ) from exc
                            if remaining != end - start + 1:
                                raise MalformedResponseError(
                                    "signed archive resumed Content-Length is "
                                    "inconsistent"
                                )
                        expected_total = total
                        response_end_exclusive = end + 1
                        os.lseek(destination_fd, resume_offset, os.SEEK_SET)
                    elif status_code == 206:
                        raise MalformedResponseError(
                            "signed archive returned a byte range without a "
                            "strong resume validator"
                        )
                    elif status_code == 200:
                        # If-Range permits a complete 200 when the
                        # representation changed. It is safe only as a full
                        # restart of the stable private file descriptor.
                        content_length = response.headers.get(
                            "Content-Length"
                        )
                        if content_length is None:
                            expected_total = None
                        else:
                            try:
                                expected_total = int(content_length)
                            except ValueError as exc:
                                raise MalformedResponseError(
                                    "signed archive Content-Length is not an "
                                    "integer"
                                ) from exc
                            if (
                                expected_total < 0
                                or expected_total > max_bytes
                            ):
                                raise ArchiveError(
                                    f"archive Content-Length {expected_total} "
                                    f"exceeds limit {max_bytes}"
                                )
                        validator = response_validator
                        response_end_exclusive = expected_total
                        os.ftruncate(destination_fd, 0)
                        os.lseek(destination_fd, 0, os.SEEK_SET)
                    else:
                        raise APIError(
                            f"signed archive URL returned HTTP {status_code}"
                        )

                    verify_destination_identity()
                    while True:
                        try:
                            block = response.read(1024 * 1024)
                        except (
                            urllib.error.URLError,
                            http.client.HTTPException,
                            TimeoutError,
                            ConnectionError,
                        ) as exc:
                            partial = getattr(exc, "partial", b"")
                            if isinstance(partial, bytes) and partial:
                                position = os.lseek(
                                    destination_fd, 0, os.SEEK_CUR
                                )
                                if (
                                    response_end_exclusive is not None
                                    and position + len(partial)
                                    > response_end_exclusive
                                ):
                                    raise MalformedResponseError(
                                        "signed archive response exceeded its "
                                        "declared byte range"
                                    ) from exc
                                if position + len(partial) > max_bytes:
                                    raise ArchiveError(
                                        f"archive exceeded byte limit "
                                        f"{max_bytes}"
                                    ) from exc
                                write_all(partial)
                            os.fsync(destination_fd)
                            raise
                        if not block:
                            break
                        position = os.lseek(
                            destination_fd, 0, os.SEEK_CUR
                        )
                        if (
                            response_end_exclusive is not None
                            and position + len(block)
                            > response_end_exclusive
                        ):
                            raise MalformedResponseError(
                                "signed archive response exceeded its "
                                "declared byte range"
                            )
                        if position + len(block) > max_bytes:
                            raise ArchiveError(
                                f"archive exceeded byte limit {max_bytes}"
                            )
                        write_all(block)
                    os.fsync(destination_fd)
                    position = os.lseek(
                        destination_fd, 0, os.SEEK_CUR
                    )
                    if (
                        response_end_exclusive is not None
                        and position != response_end_exclusive
                    ):
                        raise http.client.IncompleteRead(
                            b"",
                            max(0, response_end_exclusive - position),
                        )
                    completed = verify_destination_identity()
                    if (
                        expected_total is not None
                        and completed.st_size != expected_total
                    ):
                        raise http.client.IncompleteRead(
                            b"",
                            max(0, expected_total - completed.st_size),
                        )
                    if completed.st_size == 0:
                        raise ArchiveError(
                            "signed archive response was empty"
                        )
                    return completed.st_size, digest_created_file()
            except urllib.error.HTTPError as exc:
                raise APIError(
                    f"signed archive URL returned HTTP {exc.code}"
                ) from exc
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                ConnectionError,
            ) as exc:
                last_transport_error = exc
                current = verify_destination_identity()
                if current.st_size > max_bytes:
                    raise ArchiveError(
                        f"archive exceeded byte limit {max_bytes}"
                    ) from exc
                if transfer_attempt == max_transfer_attempts:
                    break
                continue

        assert last_transport_error is not None
        raise ArchiveError(
            "signed archive transport retries exhausted before EOF: "
            f"{type(last_transport_error).__name__}"
        ) from last_transport_error
    finally:
        os.close(destination_fd)


def _stream_signed_archive_response(
    response: Any,
    destination: Path,
    *,
    max_bytes: int,
) -> tuple[int, str]:
    """Durably stream one complete signed-URL response into a bounded file."""

    status_code = int(response.status)
    if status_code != 200:
        raise APIError(
            f"signed archive URL returned HTTP {status_code}"
        )
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise MalformedResponseError(
                "signed archive Content-Length is not an integer"
            ) from exc
        if declared < 0 or declared > max_bytes:
            raise ArchiveError(
                f"archive Content-Length {declared} exceeds limit "
                f"{max_bytes}"
            )

    digest = hashlib.sha256()
    total = 0
    with destination.open("xb") as output:
        while True:
            try:
                block = response.read(1024 * 1024)
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                ConnectionError,
            ) as exc:
                raise ArchiveError(
                    "signed archive transport failed before EOF: "
                    f"{type(exc).__name__}"
                ) from exc
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ArchiveError(
                    f"archive exceeded byte limit {max_bytes}"
                )
            output.write(block)
            digest.update(block)
        output.flush()
        os.fsync(output.fileno())
    if total == 0:
        raise ArchiveError("signed archive response was empty")
    return total, digest.hexdigest()


class ExactTokenizer:
    """Frozen training-tokenizer adapter with an auditable fingerprint."""

    def __init__(self, tokenizer_json: str | os.PathLike[str]):
        path = Path(tokenizer_json).expanduser().resolve()
        try:
            from tokenizers import __version__ as tokenizers_version
        except ImportError as exc:
            raise FetchError(
                "the existing project environment lacks the tokenizers package"
            ) from exc
        if not path.is_file() or path.is_symlink():
            raise FetchError(f"tokenizer.json is missing or unsafe: {path}")
        raw = path.read_bytes()
        self.path = path
        self.artifact_sha256 = _sha256_bytes(raw)
        try:
            self._tokenizer = load_cppmega_tokenizer(path)
        except TokenizerContractError as exc:
            raise FetchError(
                f"tokenizer.json does not satisfy the frozen cppmega "
                f"training contract: {path}: {exc}"
            ) from exc
        import cppmega.data.prompt_graph as prompt_graph_module
        import cppmega.tokenizer.cpp_tokenizer as tokenizer_module

        self.contract = {
            "schema": "cppmega_exact_ci_training_tokenizer_v2",
            "artifact_sha256": self.artifact_sha256,
            "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
            "library": "tokenizers",
            "library_version": str(tokenizers_version),
            "training_adapter": (
                "cppmega.tokenizer.cpp_tokenizer."
                "CppMegaTokenizer.encode_batch"
            ),
            "training_adapter_module_sha256": _sha256_file(
                Path(tokenizer_module.__file__).resolve()
            ),
            "whitespace_normalizer": (
                "cppmega.data.prompt_graph."
                "normalize_cpp_whitespace_with_offsets"
            ),
            "whitespace_normalizer_module_sha256": _sha256_file(
                Path(prompt_graph_module.__file__).resolve()
            ),
            "prepend_token": None,
            "append_token": None,
            "payload_only": True,
        }
        self.fingerprint = _sha256_bytes(
            _canonical_json_bytes(self.contract)
        )

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        if not texts:
            return []
        try:
            token_ids = self._tokenizer.encode_batch(list(texts))
        except (TypeError, ValueError) as exc:
            raise FetchError(
                f"cppmega training tokenizer rejected a CI payload: {exc}"
            ) from exc
        if len(token_ids) != len(texts):
            raise FetchError("tokenizer changed the batch cardinality")
        return token_ids


_PROCESS_TOKENIZERS: dict[str, ExactTokenizer] = {}


def _section_for_parsed_chunk(
    parsed: Mapping[str, object], chunk: Mapping[str, object]
) -> Mapping[str, object] | None:
    ordinal = chunk.get("section_ordinal")
    sections = parsed.get("sections")
    if (
        isinstance(ordinal, int)
        and not isinstance(ordinal, bool)
        and isinstance(sections, list)
        and 0 <= ordinal < len(sections)
        and isinstance(sections[ordinal], dict)
    ):
        return sections[ordinal]
    return None


def _materialize_parsed_member(
    raw: bytes,
    metadata: Mapping[str, object],
    *,
    max_chunk_chars: int,
    parser: Callable[..., Mapping[str, object]],
    tokenizer: ExactTokenizer,
) -> dict[str, object]:
    parsed = parser(raw, metadata, max_chunk_chars=max_chunk_chars)
    if not isinstance(parsed, Mapping):
        raise FetchError("CI parser returned a non-mapping result")
    canonical_text = parsed.get("canonical_text")
    dedup_text = parsed.get("dedup_text")
    chunks = parsed.get("chunks")
    sidecar = parsed.get("sidecar")
    if (
        not isinstance(canonical_text, str)
        or not isinstance(dedup_text, str)
        or not isinstance(chunks, list)
        or not isinstance(sidecar, dict)
        or any(not isinstance(item, dict) for item in chunks)
    ):
        raise FetchError("CI parser returned an invalid result contract")

    retained_chunks: list[dict[str, object]] = []
    chunk_texts: list[str] = []
    for raw_chunk in chunks:
        text = raw_chunk.get("text")
        if not isinstance(text, str):
            raise FetchError("parser chunk text is missing")
        if not text:
            continue
        retained_chunks.append(dict(raw_chunk))
        chunk_texts.append(text)
    token_batches = tokenizer.encode_batch(chunk_texts)
    materialized_chunks: list[dict[str, object]] = []
    for chunk, text, token_ids in zip(
        retained_chunks, chunk_texts, token_batches, strict=True
    ):
        ordinal = chunk.get("ordinal")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
        ):
            raise FetchError("parser chunk ordinal is invalid")
        section = _section_for_parsed_chunk(parsed, chunk)
        compact_chunk = {
            key: value
            for key, value in chunk.items()
            if key not in {"text", "canonical_text", "dedup_text"}
        }
        compact_section = None
        if section is not None:
            compact_section = {
                key: value
                for key, value in section.items()
                if key not in {"text", "dedup_text"}
            }
        materialized_chunks.append(
            {
                "ordinal": ordinal,
                "text": text,
                "token_count": len(token_ids),
                "token_sequence_sha256": hash_token_sequence(token_ids),
                "chunk": compact_chunk,
                "section": compact_section,
            }
        )
    return {
        "canonical_sha256": _sha256_bytes(canonical_text.encode("utf-8")),
        "dedup_sha256": _sha256_bytes(dedup_text.encode("utf-8")),
        "sidecar": sidecar,
        "chunks": materialized_chunks,
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }


def _process_parse_member(
    raw: bytes,
    metadata: Mapping[str, object],
    max_chunk_chars: int,
    tokenizer_path: str,
) -> dict[str, object]:
    tokenizer = _PROCESS_TOKENIZERS.get(tokenizer_path)
    if tokenizer is None:
        tokenizer = ExactTokenizer(tokenizer_path)
        _PROCESS_TOKENIZERS[tokenizer_path] = tokenizer
    return _materialize_parsed_member(
        raw,
        metadata,
        max_chunk_chars=max_chunk_chars,
        parser=canonicalize_ci_log,
        tokenizer=tokenizer,
    )


_BINDING_KEYS = (
    "fetcher_script_sha256",
    "parser_script_sha256",
    "content_store_script_sha256",
)
_BINDING_UPGRADES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS binding_upgrades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    binding_key TEXT NOT NULL CHECK (
      binding_key IN (
        'fetcher_script_sha256',
        'parser_script_sha256',
        'content_store_script_sha256'
      )
    ),
    from_sha256 TEXT NOT NULL CHECK (length(from_sha256) = 64),
    to_sha256 TEXT NOT NULL CHECK (length(to_sha256) = 64),
    reason TEXT NOT NULL,
    upgraded_at TEXT NOT NULL,
    UNIQUE(binding_key,from_sha256,to_sha256)
)
"""

_STATE_SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    repo TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    run_metadata_sha256 TEXT NOT NULL,
    run_metadata_raw_size INTEGER NOT NULL,
    run_metadata_zlib BLOB NOT NULL,
    run_metadata_source TEXT NOT NULL CHECK (
      run_metadata_source IN (
        'inventory-run-list',
        'github-workflow-run-attempt-api'
      )
    ),
    run_metadata_source_attempt INTEGER NOT NULL CHECK (
      run_metadata_source_attempt >= 1
    ),
    run_metadata_exact INTEGER NOT NULL CHECK (
      run_metadata_exact IN (0,1)
    ),
    inventory_seed_attempt INTEGER NOT NULL CHECK (
      inventory_seed_attempt >= 1
    ),
    inventory_seed_metadata_sha256 TEXT NOT NULL CHECK (
      length(inventory_seed_metadata_sha256) = 64
    ),
    status TEXT NOT NULL CHECK (
      status IN (
        'pending','processing','retry','done','empty',
        'terminal_404','terminal_410','failed'
      )
    ),
    tries INTEGER NOT NULL DEFAULT 0,
    archive_source TEXT,
    archive_sha256 TEXT,
    archive_size INTEGER,
    jobs_sha256 TEXT,
    jobs_raw_size INTEGER,
    jobs_zlib BLOB,
    member_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    occurrence_tokens INTEGER NOT NULL DEFAULT 0,
    terminal_http_status INTEGER,
    terminal_body_sha256 TEXT,
    error_class TEXT,
    error_message TEXT,
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(repo,run_id,attempt)
);
CREATE INDEX IF NOT EXISTS idx_attempts_work
ON attempts(status,created_at,repo,run_id,attempt);
CREATE TABLE IF NOT EXISTS members (
    repo TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    archive_member TEXT NOT NULL,
    job_key TEXT NOT NULL,
    raw_sha256 TEXT NOT NULL,
    raw_size INTEGER NOT NULL,
    canonical_sha256 TEXT NOT NULL,
    dedup_sha256 TEXT NOT NULL,
    sidecar_sha256 TEXT NOT NULL,
    sidecar_raw_size INTEGER NOT NULL,
    sidecar_zlib BLOB NOT NULL,
    chunk_count INTEGER NOT NULL,
    occurrence_tokens INTEGER NOT NULL,
    PRIMARY KEY(repo,run_id,attempt,archive_member),
    FOREIGN KEY(repo,run_id,attempt)
      REFERENCES attempts(repo,run_id,attempt)
);
CREATE TABLE IF NOT EXISTS request_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    requested_at TEXT NOT NULL,
    repo TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    page_no INTEGER,
    request_attempt INTEGER NOT NULL,
    http_status INTEGER,
    outcome TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    error_class TEXT,
    error_message TEXT
);
""" + _BINDING_UPGRADES_TABLE_SQL + ";\n"


def _validate_binding_upgrade_authorization(
    *,
    binding_key: str,
    source_sha256: str | None,
    reason: str | None,
    resume: bool,
) -> None:
    label = binding_key.removesuffix("_sha256").replace("_", " ")
    if source_sha256 is not None:
        if not resume:
            raise ValueError(f"{label} binding upgrade requires resume=True")
        if (
            not isinstance(source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
        ):
            raise ValueError(
                f"{label} binding upgrade source must be a lowercase SHA-256"
            )
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or reason != reason.strip()
            or len(reason) > 200
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in reason
            )
        ):
            raise ValueError(
                f"{label} binding upgrade reason must be 1-200 printable "
                "characters without surrounding whitespace"
            )
    elif reason is not None:
        raise ValueError(
            f"{label} binding upgrade reason requires an authorized source SHA-256"
        )


def _ensure_binding_upgrades_table_schema(
    connection: sqlite3.Connection,
) -> None:
    """Atomically widen either known legacy binding-upgrade ledger."""

    row = connection.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name='binding_upgrades'
        """
    ).fetchone()
    if row is None or not isinstance(row[0], str):
        raise BindingError("fetch-state binding-upgrade table is missing")
    table_sql = str(row[0])
    columns = tuple(
        str(item[1])
        for item in connection.execute(
            "PRAGMA table_info(binding_upgrades)"
        )
    )
    expected_columns = (
        "id",
        "binding_key",
        "from_sha256",
        "to_sha256",
        "reason",
        "upgraded_at",
    )
    if columns != expected_columns:
        raise BindingError("fetch-state binding-upgrade table is unsupported")
    if "'content_store_script_sha256'" in table_sql:
        return
    has_parser_binding = "'parser_script_sha256'" in table_sql
    required_legacy_fragments = (
        "length(from_sha256) = 64",
        "length(to_sha256) = 64",
        "UNIQUE(binding_key,from_sha256,to_sha256)",
    )
    compact_sql = " ".join(table_sql.split())
    if (
        any(fragment not in compact_sql for fragment in required_legacy_fragments)
        or "'fetcher_script_sha256'" not in compact_sql
        or (
            has_parser_binding
            and "binding_key IN" not in compact_sql
        )
        or (
            not has_parser_binding
            and "binding_key = 'fetcher_script_sha256'" not in compact_sql
        )
    ):
        raise BindingError(
            "fetch-state binding-upgrade table is not the known legacy schema"
        )
    allowed_legacy_keys = {"fetcher_script_sha256"}
    if has_parser_binding:
        allowed_legacy_keys.add("parser_script_sha256")
    stored_keys = {
        str(item[0])
        for item in connection.execute(
            "SELECT DISTINCT binding_key FROM binding_upgrades"
        )
    }
    if not stored_keys.issubset(allowed_legacy_keys):
        raise BindingError(
            "fetch-state binding-upgrade table contains unsupported keys"
        )
    unique_indexes = [
        str(index[1])
        for index in connection.execute(
            "PRAGMA index_list(binding_upgrades)"
        )
        if int(index[2]) == 1
    ]
    if len(unique_indexes) != 1:
        raise BindingError(
            "fetch-state binding-upgrade uniqueness contract is unsupported"
        )
    unique_columns = tuple(
        str(item[2])
        for item in connection.execute(
            f"PRAGMA index_info({unique_indexes[0]!r})"
        )
    )
    if unique_columns != (
        "binding_key",
        "from_sha256",
        "to_sha256",
    ):
        raise BindingError(
            "fetch-state binding-upgrade uniqueness contract is unsupported"
        )

    connection.execute(
        "ALTER TABLE binding_upgrades RENAME TO binding_upgrades_legacy"
    )
    connection.execute(_BINDING_UPGRADES_TABLE_SQL)
    connection.execute(
        """
        INSERT INTO binding_upgrades(
          id,binding_key,from_sha256,to_sha256,reason,upgraded_at
        )
        SELECT id,binding_key,from_sha256,to_sha256,reason,upgraded_at
        FROM binding_upgrades_legacy
        ORDER BY id
        """
    )
    connection.execute("DROP TABLE binding_upgrades_legacy")


class FetchState:
    """Durable attempt, request, and compact full-sidecar ledger."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        inventory_path: str | os.PathLike[str],
        content_store_path: str | os.PathLike[str],
        tokenizer: ExactTokenizer,
        resume: bool,
        content_store_creator_script_sha256: str | None = None,
        allow_fetcher_script_upgrade_from_sha256: str | None = None,
        fetcher_script_upgrade_reason: str | None = None,
        allow_parser_script_upgrade_from_sha256: str | None = None,
        parser_script_upgrade_reason: str | None = None,
        allow_content_store_script_upgrade_from_sha256: str | None = None,
        content_store_script_upgrade_reason: str | None = None,
    ):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.inventory_path = Path(inventory_path).expanduser().resolve()
        self.content_store_path = (
            Path(content_store_path).expanduser().resolve()
        )
        content_store_binding = (
            _content_store_sha256()
            if content_store_creator_script_sha256 is None
            else content_store_creator_script_sha256
        )
        if (
            not isinstance(content_store_binding, str)
            or re.fullmatch(r"[0-9a-f]{64}", content_store_binding) is None
        ):
            raise ValueError(
                "content-store creator script binding must be a lowercase "
                "SHA-256"
            )
        self._discovery_cursor: tuple[str, str, int, int] | None = None
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=60.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=60000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(_STATE_SCHEMA)
        expected = {
            "schema": SCHEMA_VERSION,
            "inventory_path": str(self.inventory_path),
            "content_store_path": str(self.content_store_path),
            "tokenizer_contract": _canonical_json(tokenizer.contract),
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "fetcher_script_sha256": _script_sha256(),
            "parser_script_sha256": _parser_sha256(),
            "content_store_script_sha256": content_store_binding,
            "chunk_semantics": (
                "parser-dedup-text-cppmega-training-tokenizer-"
                "payload-only-no-framing-v2"
            ),
        }
        upgrade_authorizations = {
            "fetcher_script_sha256": (
                allow_fetcher_script_upgrade_from_sha256,
                fetcher_script_upgrade_reason,
            ),
            "parser_script_sha256": (
                allow_parser_script_upgrade_from_sha256,
                parser_script_upgrade_reason,
            ),
            "content_store_script_sha256": (
                allow_content_store_script_upgrade_from_sha256,
                content_store_script_upgrade_reason,
            ),
        }
        for binding_key, (source_sha256, reason) in (
            upgrade_authorizations.items()
        ):
            _validate_binding_upgrade_authorization(
                binding_key=binding_key,
                source_sha256=source_sha256,
                reason=reason,
                resume=resume,
            )
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = dict(
                    self._connection.execute(
                        "SELECT key,value FROM settings"
                    ).fetchall()
                )
                if current:
                    if not resume:
                        raise BindingError(
                            f"fetch state exists at {self.path}; pass --resume"
                        )
                    script_upgrades: dict[
                        str,
                        tuple[str, str, str],
                    ] = {}
                    for binding_key in _BINDING_KEYS:
                        current_value = current.get(binding_key)
                        expected_value = expected[binding_key]
                        source_sha256, reason = upgrade_authorizations[
                            binding_key
                        ]
                        if (
                            current_value != expected_value
                            and current_value == source_sha256
                        ):
                            assert current_value is not None
                            assert reason is not None
                            script_upgrades[binding_key] = (
                                current_value,
                                expected_value,
                                reason,
                            )
                    mismatches = {
                        key: (current.get(key), value)
                        for key, value in expected.items()
                        if current.get(key) != value
                        and key not in script_upgrades
                    }
                    if mismatches:
                        rendered = ", ".join(
                            f"{key}={old!r}->{new!r}"
                            for key, (old, new) in sorted(mismatches.items())
                        )
                        raise BindingError(
                            f"fetch-state binding mismatch: {rendered}"
                        )
                    for binding_key in _BINDING_KEYS:
                        source_sha256, reason = upgrade_authorizations[
                            binding_key
                        ]
                        if (
                            source_sha256 is None
                            or binding_key in script_upgrades
                        ):
                            continue
                        replay = self._connection.execute(
                            """
                            SELECT from_sha256,to_sha256,reason
                            FROM binding_upgrades
                            WHERE binding_key=?
                            ORDER BY id DESC
                            LIMIT 1
                            """,
                            (binding_key,),
                        ).fetchone()
                        if replay is None or (
                            str(replay["from_sha256"]),
                            str(replay["to_sha256"]),
                            str(replay["reason"]),
                        ) != (
                            source_sha256,
                            expected[binding_key],
                            reason,
                        ):
                            raise BindingError(
                                f"{binding_key} upgrade authorization does not "
                                "replay the latest audited transition"
                            )
                    _ensure_binding_upgrades_table_schema(self._connection)
                    upgraded_at = _utc_now()
                    for binding_key in _BINDING_KEYS:
                        upgrade = script_upgrades.get(binding_key)
                        if upgrade is None:
                            continue
                        previous_script_sha256, next_script_sha256, reason = (
                            upgrade
                        )
                        self._connection.execute(
                            """
                            INSERT INTO binding_upgrades(
                              binding_key,from_sha256,to_sha256,
                              reason,upgraded_at
                            ) VALUES (?,?,?,?,?)
                            """,
                            (
                                binding_key,
                                previous_script_sha256,
                                next_script_sha256,
                                reason,
                                upgraded_at,
                            ),
                        )
                        self._connection.execute(
                            """
                            UPDATE settings SET value=?
                            WHERE key=?
                            """,
                            (next_script_sha256, binding_key),
                        )
                else:
                    self._connection.executemany(
                        "INSERT INTO settings(key,value) VALUES (?,?)",
                        sorted(expected.items()),
                    )
                    self._connection.execute(
                        "INSERT INTO settings(key,value) VALUES ('created_at',?)",
                        (_utc_now(),),
                    )
                if resume:
                    self._connection.execute(
                        """
                        UPDATE attempts SET status='retry',
                            error_class='InterruptedAttempt',
                            error_message='processing interrupted before closure',
                            updated_at=?
                        WHERE status='processing'
                        """,
                        (_utc_now(),),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                self._connection.close()
                raise

    def close(self) -> None:
        self._connection.close()

    def _inventory_connection(self) -> sqlite3.Connection:
        if not self.inventory_path.is_file():
            raise FetchError(
                f"inventory SQLite does not exist: {self.inventory_path}"
            )
        connection = sqlite3.connect(
            f"file:{self.inventory_path}?mode=ro",
            uri=True,
            timeout=60.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA query_only=ON")
        return connection

    def discover(self, *, row_limit: int = DEFAULT_DISCOVERY_ROWS) -> int:
        if row_limit <= 0:
            raise ValueError("row_limit must be positive")
        inventory = self._inventory_connection()
        try:
            if self._discovery_cursor is None:
                rows = inventory.execute(
                    """
                    SELECT repo_key,run_id,run_attempt,created_at,
                           metadata_blob,metadata_sha256
                    FROM runs
                    ORDER BY created_at,repo_key,run_id,run_attempt
                    LIMIT ?
                    """,
                    (row_limit,),
                ).fetchall()
            else:
                created_at, repo_key, run_id, run_attempt = (
                    self._discovery_cursor
                )
                rows = inventory.execute(
                    """
                    SELECT repo_key,run_id,run_attempt,created_at,
                           metadata_blob,metadata_sha256
                    FROM runs
                    WHERE (created_at,repo_key,run_id,run_attempt)
                          > (?,?,?,?)
                    ORDER BY created_at,repo_key,run_id,run_attempt
                    LIMIT ?
                    """,
                    (
                        created_at,
                        repo_key,
                        run_id,
                        run_attempt,
                        row_limit,
                    ),
                ).fetchall()
        finally:
            inventory.close()
        if rows:
            final_row = rows[-1]
            self._discovery_cursor = (
                str(final_row["created_at"]),
                str(final_row["repo_key"]),
                int(final_row["run_id"]),
                int(final_row["run_attempt"]),
            )
        else:
            # A completed sweep restarts so runs inserted late into an older
            # repository/window cannot be missed permanently.
            self._discovery_cursor = None
        now = _utc_now()
        inserted = 0
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    blob = bytes(row["metadata_blob"])
                    try:
                        metadata_bytes = zlib.decompress(blob)
                        metadata = json.loads(metadata_bytes)
                    except (zlib.error, UnicodeError, json.JSONDecodeError) as exc:
                        raise FetchError(
                            f"corrupt inventory metadata for "
                            f"{row['repo_key']}#{row['run_id']}"
                        ) from exc
                    if not isinstance(metadata, dict):
                        raise FetchError("inventory run metadata is not an object")
                    metadata_sha = _sha256_bytes(metadata_bytes)
                    if metadata_sha != str(row["metadata_sha256"]):
                        raise FetchError("inventory run metadata digest mismatch")
                    raw_attempt = int(row["run_attempt"])
                    if raw_attempt < 1:
                        raise FetchError(
                            "inventory run attempt must be positive"
                        )
                    run_id = int(row["run_id"])
                    _validate_run_metadata_identity(
                        metadata,
                        run_id=run_id,
                        attempt=raw_attempt,
                    )
                    for attempt in range(1, raw_attempt + 1):
                        exact = int(attempt == raw_attempt)
                        cursor = self._connection.execute(
                            """
                            INSERT INTO attempts(
                              repo,run_id,attempt,created_at,
                              run_metadata_sha256,run_metadata_raw_size,
                              run_metadata_zlib,run_metadata_source,
                              run_metadata_source_attempt,run_metadata_exact,
                              inventory_seed_attempt,
                              inventory_seed_metadata_sha256,
                              status,discovered_at,updated_at
                            ) VALUES (
                              ?,?,?,?,?,?,?,
                              'inventory-run-list',?,?,?,?,
                              'pending',?,?
                            )
                            ON CONFLICT(repo,run_id,attempt) DO UPDATE SET
                              inventory_seed_attempt=
                                excluded.inventory_seed_attempt,
                              inventory_seed_metadata_sha256=
                                excluded.inventory_seed_metadata_sha256,
                              created_at=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.created_at
                                ELSE excluded.created_at
                              END,
                              run_metadata_sha256=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.run_metadata_sha256
                                ELSE excluded.run_metadata_sha256
                              END,
                              run_metadata_raw_size=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.run_metadata_raw_size
                                ELSE excluded.run_metadata_raw_size
                              END,
                              run_metadata_zlib=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.run_metadata_zlib
                                ELSE excluded.run_metadata_zlib
                              END,
                              run_metadata_source=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.run_metadata_source
                                ELSE excluded.run_metadata_source
                              END,
                              run_metadata_source_attempt=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.run_metadata_source_attempt
                                ELSE excluded.run_metadata_source_attempt
                              END,
                              run_metadata_exact=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN 1
                                ELSE excluded.run_metadata_exact
                              END,
                              updated_at=excluded.updated_at
                            WHERE attempts.status IN ('pending','retry')
                            """,
                            (
                                str(row["repo_key"]),
                                run_id,
                                attempt,
                                str(row["created_at"]),
                                metadata_sha,
                                len(metadata_bytes),
                                sqlite3.Binary(zlib.compress(metadata_bytes, 6)),
                                raw_attempt,
                                exact,
                                raw_attempt,
                                metadata_sha,
                                now,
                                now,
                            ),
                        )
                        inserted += int(cursor.rowcount > 0)
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return inserted

    @staticmethod
    def _decode_attempt(row: sqlite3.Row) -> Attempt:
        blob = bytes(row["run_metadata_zlib"])
        try:
            raw = zlib.decompress(blob)
            value = json.loads(raw)
        except (zlib.error, UnicodeError, json.JSONDecodeError) as exc:
            raise FetchError("fetch-state run metadata is corrupt") from exc
        if not isinstance(value, dict):
            raise FetchError("fetch-state run metadata is not an object")
        if len(raw) != int(row["run_metadata_raw_size"]):
            raise FetchError("fetch-state run metadata size mismatch")
        digest = _sha256_bytes(raw)
        if digest != str(row["run_metadata_sha256"]):
            raise FetchError("fetch-state run metadata digest mismatch")
        run_id = int(row["run_id"])
        attempt = int(row["attempt"])
        source = str(row["run_metadata_source"])
        source_attempt = int(row["run_metadata_source_attempt"])
        exact_raw = int(row["run_metadata_exact"])
        seed_attempt = int(row["inventory_seed_attempt"])
        seed_sha = str(row["inventory_seed_metadata_sha256"])
        if source not in _RUN_METADATA_SOURCES:
            raise FetchError("fetch-state run metadata source is invalid")
        if exact_raw not in {0, 1}:
            raise FetchError("fetch-state run metadata exactness is invalid")
        exact = bool(exact_raw)
        if seed_attempt < attempt:
            raise FetchError("inventory seed attempt precedes target attempt")
        if re.fullmatch(r"[0-9a-f]{64}", seed_sha) is None:
            raise FetchError("inventory seed metadata digest is invalid")
        if source == "inventory-run-list" and source_attempt != seed_attempt:
            raise FetchError("inventory metadata source attempt is inconsistent")
        if source == "github-workflow-run-attempt-api" and not exact:
            raise FetchError("attempt API metadata must be exact")
        if exact != (source_attempt == attempt):
            raise FetchError("fetch-state run metadata exactness is inconsistent")
        _validate_run_metadata_identity(
            value,
            run_id=run_id,
            attempt=source_attempt,
        )
        return Attempt(
            repo=str(row["repo"]),
            run_id=run_id,
            attempt=attempt,
            created_at=str(row["created_at"]),
            run_metadata=value,
            run_metadata_sha256=digest,
            run_metadata_source=source,
            run_metadata_source_attempt=source_attempt,
            run_metadata_exact=exact,
            inventory_seed_attempt=seed_attempt,
            inventory_seed_metadata_sha256=seed_sha,
        )

    def next_attempt(self, *, retry_only: bool = False) -> Attempt | None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    (
                        """
                    SELECT * FROM attempts
                    WHERE status='retry'
                    ORDER BY created_at,repo,run_id,attempt
                    LIMIT 1
                        """
                        if retry_only
                        else """
                    SELECT * FROM attempts
                    WHERE status IN ('pending','retry')
                    ORDER BY created_at,repo,run_id,attempt
                    LIMIT 1
                        """
                    )
                ).fetchone()
                if row is None:
                    self._connection.execute("COMMIT")
                    return None
                self._connection.execute(
                    """
                    UPDATE attempts SET status='processing',tries=tries+1,
                      error_class=NULL,error_message=NULL,updated_at=?
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (
                        _utc_now(),
                        str(row["repo"]),
                        int(row["run_id"]),
                        int(row["attempt"]),
                    ),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return self._decode_attempt(row)

    def bind_exact_run_metadata(
        self,
        attempt: Attempt,
        metadata: Mapping[str, object],
    ) -> Attempt:
        if attempt.run_metadata_exact:
            raise BindingError("run metadata is already exact")
        exact = dict(metadata)
        _validate_run_metadata_identity(
            exact,
            run_id=attempt.run_id,
            attempt=attempt.attempt,
        )
        seed_repository = _repository_identity(attempt)
        exact_repository = _repository_object_identity(
            exact.get("repository"),
            field="repository",
        )
        if exact_repository is None:
            raise MalformedResponseError(
                "attempt API metadata has no repository identity"
            )
        exact_name, exact_id = exact_repository
        if exact_name.casefold() != seed_repository.canonical.casefold():
            raise MalformedResponseError(
                "attempt API repository does not match inventory metadata"
            )
        if (
            exact_id is not None
            and seed_repository.repository_id is not None
            and exact_id != seed_repository.repository_id
        ):
            raise MalformedResponseError(
                "attempt API repository id does not match inventory metadata"
            )
        raw = _canonical_json_bytes(exact)
        digest = _sha256_bytes(raw)
        now = _utc_now()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT status,run_metadata_sha256,run_metadata_exact,
                           (SELECT COUNT(*) FROM members
                            WHERE repo=attempts.repo
                              AND run_id=attempts.run_id
                              AND attempt=attempts.attempt) AS member_count
                    FROM attempts
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (attempt.repo, attempt.run_id, attempt.attempt),
                ).fetchone()
                if row is None:
                    raise BindingError("attempt disappeared before metadata bind")
                if str(row["status"]) != "processing":
                    raise BindingError(
                        "attempt metadata can bind only while processing"
                    )
                if int(row["run_metadata_exact"]) != 0:
                    raise BindingError("attempt metadata became exact concurrently")
                if str(row["run_metadata_sha256"]) != attempt.run_metadata_sha256:
                    raise BindingError("attempt seed metadata changed concurrently")
                if int(row["member_count"]) != 0:
                    raise BindingError(
                        "attempt metadata cannot change after member commits"
                    )
                self._connection.execute(
                    """
                    UPDATE attempts SET
                      created_at=?,
                      run_metadata_sha256=?,
                      run_metadata_raw_size=?,
                      run_metadata_zlib=?,
                      run_metadata_source=
                        'github-workflow-run-attempt-api',
                      run_metadata_source_attempt=?,
                      run_metadata_exact=1,
                      updated_at=?
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (
                        str(exact["created_at"]),
                        digest,
                        len(raw),
                        sqlite3.Binary(zlib.compress(raw, 6)),
                        attempt.attempt,
                        now,
                        attempt.repo,
                        attempt.run_id,
                        attempt.attempt,
                    ),
                )
                updated = self._connection.execute(
                    """
                    SELECT * FROM attempts
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (attempt.repo, attempt.run_id, attempt.attempt),
                ).fetchone()
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        assert updated is not None
        return self._decode_attempt(updated)

    def record_request(
        self,
        attempt: Attempt,
        *,
        endpoint: str,
        page_no: int | None,
        request_attempt: int,
        http_status: int | None,
        outcome: str,
        latency_ms: int,
        error: BaseException | str | None = None,
        secrets: Iterable[str] = (),
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO request_ledger(
                  requested_at,repo,run_id,attempt,endpoint,page_no,
                  request_attempt,http_status,outcome,latency_ms,
                  error_class,error_message
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _utc_now(),
                    attempt.repo,
                    attempt.run_id,
                    attempt.attempt,
                    endpoint,
                    page_no,
                    request_attempt,
                    http_status,
                    outcome,
                    latency_ms,
                    None if error is None else type(error).__name__,
                    None if error is None else _safe_error(error, secrets),
                ),
            )

    def store_member(
        self,
        attempt: Attempt,
        *,
        archive_member: str,
        job_key: str,
        raw_sha256: str,
        raw_size: int,
        canonical_sha256: str,
        dedup_sha256: str,
        sidecar: Mapping[str, object],
        chunk_count: int,
        occurrence_tokens: int,
    ) -> None:
        if not attempt.run_metadata_exact:
            raise BindingError(
                "cannot store a member without exact attempt metadata"
            )
        sidecar_bytes = _canonical_json_bytes(sidecar)
        sidecar_sha = _sha256_bytes(sidecar_bytes)
        compressed = zlib.compress(sidecar_bytes, 6)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                previous = self._connection.execute(
                    """
                    SELECT raw_sha256,canonical_sha256,dedup_sha256,
                           sidecar_sha256,chunk_count,occurrence_tokens
                    FROM members
                    WHERE repo=? AND run_id=? AND attempt=?
                      AND archive_member=?
                    """,
                    (
                        attempt.repo,
                        attempt.run_id,
                        attempt.attempt,
                        archive_member,
                    ),
                ).fetchone()
                identity = (
                    raw_sha256,
                    canonical_sha256,
                    dedup_sha256,
                    sidecar_sha,
                    chunk_count,
                    occurrence_tokens,
                )
                if previous is not None:
                    old = (
                        str(previous["raw_sha256"]),
                        str(previous["canonical_sha256"]),
                        str(previous["dedup_sha256"]),
                        str(previous["sidecar_sha256"]),
                        int(previous["chunk_count"]),
                        int(previous["occurrence_tokens"]),
                    )
                    if old != identity:
                        raise BindingError(
                            f"member replay changed: {archive_member}"
                        )
                else:
                    self._connection.execute(
                        """
                        INSERT INTO members(
                          repo,run_id,attempt,archive_member,job_key,
                          raw_sha256,raw_size,canonical_sha256,dedup_sha256,
                          sidecar_sha256,sidecar_raw_size,sidecar_zlib,
                          chunk_count,occurrence_tokens
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            attempt.repo,
                            attempt.run_id,
                            attempt.attempt,
                            archive_member,
                            job_key,
                            raw_sha256,
                            raw_size,
                            canonical_sha256,
                            dedup_sha256,
                            sidecar_sha,
                            len(sidecar_bytes),
                            sqlite3.Binary(compressed),
                            chunk_count,
                            occurrence_tokens,
                        ),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def replayed_member(
        self,
        attempt: Attempt,
        *,
        archive_member: str,
        job_key: str,
        raw_sha256: str,
        raw_size: int,
    ) -> tuple[int, int] | None:
        """Return durable member totals only after exact replay validation."""

        with self._lock:
            previous = self._connection.execute(
                """
                SELECT job_key,raw_sha256,raw_size,
                       chunk_count,occurrence_tokens
                FROM members
                WHERE repo=? AND run_id=? AND attempt=?
                  AND archive_member=?
                """,
                (
                    attempt.repo,
                    attempt.run_id,
                    attempt.attempt,
                    archive_member,
                ),
            ).fetchone()
        if previous is None:
            return None
        expected = (job_key, raw_sha256, raw_size)
        actual = (
            str(previous["job_key"]),
            str(previous["raw_sha256"]),
            int(previous["raw_size"]),
        )
        if actual != expected:
            raise BindingError(
                f"committed member replay changed: {archive_member}"
            )
        return (
            int(previous["chunk_count"]),
            int(previous["occurrence_tokens"]),
        )

    def fail_terminal_probe_with_durable_members(
        self,
        attempt: Attempt,
        *,
        error: TerminalHTTP,
        secrets: Iterable[str] = (),
    ) -> bool:
        """Refuse a terminal classification when this attempt already owns CAS.

        A retry can observe HTTP 404/410 after an earlier process parsed only
        part of a complete archive.  Marking that attempt terminal would hide
        its durable members from the normal terminal summary while leaving
        their CAS occurrences behind.  Keep the row explicitly failed so
        receipt finalization remains blocked until an audited archive recovery
        completes the attempt.
        """

        with self._lock, self._connection:
            durable_members = int(
                self._connection.execute(
                    """
                    SELECT COUNT(*) FROM members
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (attempt.repo, attempt.run_id, attempt.attempt),
                ).fetchone()[0]
            )
            if durable_members == 0:
                return False
            self._connection.execute(
                """
                UPDATE attempts SET
                  status='failed',
                  member_count=0,chunk_count=0,occurrence_tokens=0,
                  terminal_http_status=?,terminal_body_sha256=?,
                  error_class=?,error_message=?,updated_at=?
                WHERE repo=? AND run_id=? AND attempt=?
                """,
                (
                    error.status,
                    _sha256_bytes(error.body),
                    type(error).__name__,
                    _safe_error(error, secrets),
                    _utc_now(),
                    attempt.repo,
                    attempt.run_id,
                    attempt.attempt,
                ),
            )
        return True

    def finish_attempt(
        self,
        attempt: Attempt,
        *,
        status: str,
        archive_source: str | None = None,
        archive_sha256: str | None = None,
        archive_size: int | None = None,
        jobs: Sequence[Mapping[str, object]] | None = None,
        member_count: int = 0,
        chunk_count: int = 0,
        occurrence_tokens: int = 0,
        terminal_http_status: int | None = None,
        terminal_body_sha256: str | None = None,
        error: BaseException | str | None = None,
        retry: bool = False,
        secrets: Iterable[str] = (),
    ) -> None:
        if status not in _RUN_ATTEMPT_STATES:
            raise ValueError(f"invalid attempt status {status!r}")
        if retry and status != "retry":
            raise ValueError("retry flag requires retry status")
        if status in {"done", "empty"} and not attempt.run_metadata_exact:
            raise BindingError(
                f"cannot mark {status} without exact attempt metadata"
            )
        jobs_bytes = (
            None if jobs is None else _canonical_json_bytes(list(jobs))
        )
        with self._lock, self._connection:
            if status in {"done", "empty"}:
                durable = self._connection.execute(
                    """
                    SELECT COUNT(*) AS member_count,
                           COALESCE(SUM(chunk_count),0) AS chunk_count,
                           COALESCE(SUM(occurrence_tokens),0)
                             AS occurrence_tokens
                    FROM members
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (attempt.repo, attempt.run_id, attempt.attempt),
                ).fetchone()
                if durable is None:
                    raise BindingError(
                        "completed attempt durable-member accounting is missing"
                    )
                durable_counts = (
                    int(durable["member_count"]),
                    int(durable["chunk_count"]),
                    int(durable["occurrence_tokens"]),
                )
                reported_counts = (
                    int(member_count),
                    int(chunk_count),
                    int(occurrence_tokens),
                )
                if any(
                    actual < reported
                    for actual, reported in zip(
                        durable_counts,
                        reported_counts,
                        strict=True,
                    )
                ):
                    raise BindingError(
                        "completed attempt counters exceed its durable members"
                    )
                member_count, chunk_count, occurrence_tokens = durable_counts
            self._connection.execute(
                """
                UPDATE attempts SET
                  status=?,archive_source=?,archive_sha256=?,archive_size=?,
                  jobs_sha256=?,jobs_raw_size=?,jobs_zlib=?,
                  member_count=?,chunk_count=?,occurrence_tokens=?,
                  terminal_http_status=?,terminal_body_sha256=?,
                  error_class=?,error_message=?,updated_at=?
                WHERE repo=? AND run_id=? AND attempt=?
                """,
                (
                    status,
                    archive_source,
                    archive_sha256,
                    archive_size,
                    None if jobs_bytes is None else _sha256_bytes(jobs_bytes),
                    None if jobs_bytes is None else len(jobs_bytes),
                    None
                    if jobs_bytes is None
                    else sqlite3.Binary(zlib.compress(jobs_bytes, 6)),
                    member_count,
                    chunk_count,
                    occurrence_tokens,
                    terminal_http_status,
                    terminal_body_sha256,
                    None if error is None else type(error).__name__,
                    None if error is None else _safe_error(error, secrets),
                    _utc_now(),
                    attempt.repo,
                    attempt.run_id,
                    attempt.attempt,
                ),
            )

    def summary(self) -> dict[str, object]:
        with self._lock:
            status_counts = {
                str(row["status"]): int(row["n"])
                for row in self._connection.execute(
                    "SELECT status,COUNT(*) AS n FROM attempts GROUP BY status"
                )
            }
            totals = self._connection.execute(
                """
                SELECT COUNT(*) AS attempts,
                       COALESCE(SUM(member_count),0) AS members,
                       COALESCE(SUM(chunk_count),0) AS chunks,
                       COALESCE(SUM(occurrence_tokens),0) AS occurrence_tokens
                FROM attempts
                WHERE status IN (
                  'done','empty','terminal_404','terminal_410'
                )
                """
            ).fetchone()
            requests = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM request_ledger"
                ).fetchone()[0]
            )
            metadata_rows = self._connection.execute(
                """
                SELECT run_metadata_source,run_metadata_exact,status,
                       COUNT(*) AS n
                FROM attempts
                GROUP BY run_metadata_source,run_metadata_exact,status
                """
            ).fetchall()
            exact_metadata = sum(
                int(row["n"])
                for row in metadata_rows
                if int(row["run_metadata_exact"]) == 1
            )
            unresolved_by_status: dict[str, int] = {}
            exact_by_source: dict[str, int] = {}
            for row in metadata_rows:
                count = int(row["n"])
                if int(row["run_metadata_exact"]) == 1:
                    source = str(row["run_metadata_source"])
                    exact_by_source[source] = (
                        exact_by_source.get(source, 0) + count
                    )
                else:
                    status = str(row["status"])
                    unresolved_by_status[status] = (
                        unresolved_by_status.get(status, 0) + count
                    )
            content_without_exact_metadata = sum(
                count
                for status, count in unresolved_by_status.items()
                if status in {"done", "empty"}
            )
            if content_without_exact_metadata:
                raise BindingError(
                    "completed content attempt lacks exact run metadata"
                )
            sidecar_digest = hashlib.sha256()
            for row in self._connection.execute(
                """
                SELECT repo,run_id,attempt,archive_member,sidecar_sha256
                FROM members
                ORDER BY repo,run_id,attempt,archive_member
                """
            ):
                sidecar_digest.update(
                    (
                        f"{row['repo']}\t{row['run_id']}\t{row['attempt']}\t"
                        f"{row['archive_member']}\t{row['sidecar_sha256']}\n"
                    ).encode("utf-8")
                )
            binding_upgrades = [
                {
                    "binding_key": str(row["binding_key"]),
                    "from_sha256": str(row["from_sha256"]),
                    "to_sha256": str(row["to_sha256"]),
                    "reason": str(row["reason"]),
                    "upgraded_at": str(row["upgraded_at"]),
                }
                for row in self._connection.execute(
                    """
                    SELECT binding_key,from_sha256,to_sha256,
                           reason,upgraded_at
                    FROM binding_upgrades
                    ORDER BY id
                    """
                )
            ]
            return {
                "attempt_statuses": status_counts,
                "attempts_terminal": int(totals["attempts"]),
                "members": int(totals["members"]),
                "chunks": int(totals["chunks"]),
                "occurrence_tokens": int(totals["occurrence_tokens"]),
                "requests": requests,
                "sidecar_set_sha256": sidecar_digest.hexdigest(),
                "run_metadata": {
                    "exact_attempts": exact_metadata,
                    "unresolved_attempts": sum(
                        unresolved_by_status.values()
                    ),
                    "exact_by_source": dict(sorted(exact_by_source.items())),
                    "unresolved_by_status": dict(
                        sorted(unresolved_by_status.items())
                    ),
                    "content_attempts_without_exact_metadata": 0,
                },
                "binding_upgrades": binding_upgrades,
            }


class GitHubAttemptClient:
    """Jobs and attempt-log client that never forwards API auth to blob URLs."""

    def __init__(
        self,
        tokens: Sequence[str],
        state: FetchState,
        *,
        requester: Callable[
            [str, str, Mapping[str, str], float], HTTPResponse
        ] = _default_no_redirect_requester,
        archive_downloader: Callable[..., tuple[int, str]] = (
            _default_archive_downloader
        ),
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_API_ATTEMPTS,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.pool = TokenPool(tokens, sleeper=sleeper)
        self.state = state
        self.requester = requester
        self.archive_downloader = archive_downloader
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.max_archive_bytes = max_archive_bytes
        self.sleeper = sleeper
        self.api_base = "https://api.github.com"

    @property
    def secrets(self) -> tuple[str, ...]:
        return self.pool.secrets

    @staticmethod
    def _body_message(body: bytes) -> str:
        try:
            value = json.loads(body)
        except (UnicodeError, json.JSONDecodeError):
            return body.decode("utf-8", errors="replace")[:1000]
        if isinstance(value, dict):
            return str(value.get("message") or value)[:1000]
        return str(value)[:1000]

    def _request(
        self,
        attempt: Attempt,
        endpoint: str,
        *,
        query: Mapping[str, object] | None = None,
        page_no: int | None = None,
        accepted: set[int],
    ) -> RequestResult:
        url = f"{self.api_base}{endpoint}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        for request_attempt in range(1, self.max_attempts + 1):
            token_index, token = self.pool.acquire()
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "cppmega-ci-stream-fetch/1",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            }
            started = time.monotonic()
            try:
                response = self.requester(
                    "GET", url, headers, self.timeout
                )
            except Exception as exc:
                elapsed = int((time.monotonic() - started) * 1000)
                self.state.record_request(
                    attempt,
                    endpoint=endpoint,
                    page_no=page_no,
                    request_attempt=request_attempt,
                    http_status=None,
                    outcome="transport_retry",
                    latency_ms=elapsed,
                    error=exc,
                    secrets=self.secrets,
                )
                if request_attempt == self.max_attempts:
                    raise APIError(
                        f"transport retries exhausted for {endpoint}"
                    ) from exc
                self.sleeper(min(2 ** (request_attempt - 1), 30))
                continue
            elapsed = int((time.monotonic() - started) * 1000)
            self.pool.observe(token_index, response.headers)
            lowered = {
                str(key).casefold(): str(value)
                for key, value in response.headers.items()
            }
            message = self._body_message(response.body)
            rate_limited = response.status == 429 or (
                response.status == 403
                and (
                    lowered.get("x-ratelimit-remaining") == "0"
                    or "rate limit" in message.casefold()
                    or "abuse" in message.casefold()
                )
            )
            if rate_limited:
                self.pool.rate_limited(
                    token_index,
                    response.headers,
                    secondary="secondary" in message.casefold(),
                )
                self.state.record_request(
                    attempt,
                    endpoint=endpoint,
                    page_no=page_no,
                    request_attempt=request_attempt,
                    http_status=response.status,
                    outcome="rate_limit_retry",
                    latency_ms=elapsed,
                    error=message,
                    secrets=self.secrets,
                )
                if request_attempt == self.max_attempts:
                    raise APIError(
                        f"rate-limit retries exhausted for {endpoint}"
                    )
                continue
            if response.status >= 500:
                self.state.record_request(
                    attempt,
                    endpoint=endpoint,
                    page_no=page_no,
                    request_attempt=request_attempt,
                    http_status=response.status,
                    outcome="server_retry",
                    latency_ms=elapsed,
                    error=message,
                    secrets=self.secrets,
                )
                if request_attempt == self.max_attempts:
                    raise APIError(
                        f"server retries exhausted for {endpoint}"
                    )
                self.sleeper(min(2 ** (request_attempt - 1), 30))
                continue
            if response.status not in accepted:
                self.state.record_request(
                    attempt,
                    endpoint=endpoint,
                    page_no=page_no,
                    request_attempt=request_attempt,
                    http_status=response.status,
                    outcome="permanent_error",
                    latency_ms=elapsed,
                    error=message,
                    secrets=self.secrets,
                )
                raise APIError(
                    f"GitHub HTTP {response.status} for {endpoint}: "
                    f"{_safe_error(message, self.secrets)}"
                )
            self.state.record_request(
                attempt,
                endpoint=endpoint,
                page_no=page_no,
                request_attempt=request_attempt,
                http_status=response.status,
                outcome="success",
                latency_ms=elapsed,
                secrets=self.secrets,
            )
            return RequestResult(
                status=response.status,
                headers=response.headers,
                body=response.body,
            )
        raise AssertionError("unreachable request loop")

    def fetch_run_metadata(self, attempt: Attempt) -> dict[str, Any]:
        if attempt.run_metadata_exact:
            raise BindingError("exact run metadata does not need refetching")
        repository = _repository_identity(attempt).canonical
        endpoint = (
            f"/repos/{repository}/actions/runs/{attempt.run_id}/"
            f"attempts/{attempt.attempt}"
        )
        result = self._request(
            attempt,
            endpoint,
            accepted={200, 404, 410},
        )
        if result.status in {404, 410}:
            raise TerminalHTTP(result.status, result.body, endpoint)
        try:
            value = json.loads(result.body)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise MalformedResponseError(
                "workflow run attempt metadata is not JSON"
            ) from exc
        if not isinstance(value, dict):
            raise MalformedResponseError(
                "workflow run attempt metadata is not an object"
            )
        _validate_run_metadata_identity(
            value,
            run_id=attempt.run_id,
            attempt=attempt.attempt,
        )
        return dict(value)

    def fetch_jobs(self, attempt: Attempt) -> list[dict[str, Any]]:
        repository = _repository_identity(attempt).canonical
        endpoint = (
            f"/repos/{repository}/actions/runs/{attempt.run_id}/"
            f"attempts/{attempt.attempt}/jobs"
        )
        jobs: list[dict[str, Any]] = []
        total: int | None = None
        page = 1
        while True:
            result = self._request(
                attempt,
                endpoint,
                query={"filter": "all", "per_page": 100, "page": page},
                page_no=page,
                accepted={200, 404, 410},
            )
            if result.status in {404, 410}:
                raise TerminalHTTP(result.status, result.body, endpoint)
            try:
                payload = json.loads(result.body)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise MalformedResponseError(
                    f"jobs page {page} is not JSON"
                ) from exc
            if (
                not isinstance(payload, dict)
                or isinstance(payload.get("total_count"), bool)
                or not isinstance(payload.get("total_count"), int)
                or int(payload["total_count"]) < 0
                or not isinstance(payload.get("jobs"), list)
                or any(not isinstance(item, dict) for item in payload["jobs"])
            ):
                raise MalformedResponseError(
                    f"jobs page {page} has an invalid schema"
                )
            page_total = int(payload["total_count"])
            if total is None:
                total = page_total
            elif total != page_total:
                raise MalformedResponseError(
                    f"jobs total_count changed {total}->{page_total}"
                )
            page_jobs = [dict(item) for item in payload["jobs"]]
            expected_pages = max(1, math.ceil(page_total / 100))
            expected_items = (
                100
                if page < expected_pages
                else page_total - 100 * (expected_pages - 1)
            )
            if len(page_jobs) != expected_items:
                raise MalformedResponseError(
                    f"jobs page {page} has {len(page_jobs)} items, "
                    f"expected {expected_items}"
                )
            jobs.extend(page_jobs)
            if page >= expected_pages:
                break
            page += 1
        assert total is not None
        ids = []
        for job in jobs:
            job_id = job.get("id")
            if isinstance(job_id, bool) or not isinstance(job_id, int):
                raise MalformedResponseError("job id is not an integer")
            ids.append(job_id)
        if len(jobs) != total or len(set(ids)) != total:
            raise MalformedResponseError(
                "jobs enumeration is incomplete or contains duplicates"
            )
        return jobs

    def prepare_archive(self, attempt: Attempt) -> PreparedArchive:
        repository = _repository_identity(attempt).canonical
        endpoint = (
            f"/repos/{repository}/actions/runs/{attempt.run_id}/"
            f"attempts/{attempt.attempt}/logs"
        )
        result = self._request(
            attempt,
            endpoint,
            accepted={200, 302, 404, 410},
        )
        if result.status in {404, 410}:
            raise TerminalHTTP(result.status, result.body, endpoint)
        if result.status == 200:
            if len(result.body) > self.max_archive_bytes:
                raise ArchiveError("inline archive exceeds byte limit")
            return PreparedArchive(
                repository=repository,
                run_id=attempt.run_id,
                attempt=attempt.attempt,
                source="github-inline",
                inline_body=result.body,
                signed_url=None,
            )
        location = None
        for key, value in result.headers.items():
            if str(key).casefold() == "location":
                location = str(value)
                break
        if not location:
            raise MalformedResponseError(
                "attempt-log redirect lacks Location"
            )
        return PreparedArchive(
            repository=repository,
            run_id=attempt.run_id,
            attempt=attempt.attempt,
            source="github-signed-url",
            inline_body=None,
            signed_url=location,
        )

    def fetch_archive(
        self,
        attempt: Attempt,
        destination: Path,
        *,
        prepared: PreparedArchive | None = None,
    ) -> ArchiveSource:
        archive = (
            self.prepare_archive(attempt)
            if prepared is None
            else prepared
        )
        if (
            archive.repository != _repository_identity(attempt).canonical
            or archive.run_id != attempt.run_id
            or archive.attempt != attempt.attempt
        ):
            raise BindingError(
                "prepared archive does not match the requested attempt"
            )
        if archive.source == "github-inline":
            if archive.inline_body is None or archive.signed_url is not None:
                raise BindingError("inline archive preparation is invalid")
            if len(archive.inline_body) > self.max_archive_bytes:
                raise ArchiveError("inline archive exceeds byte limit")
            with destination.open("xb") as output:
                output.write(archive.inline_body)
                output.flush()
                os.fsync(output.fileno())
            return ArchiveSource(
                path=destination,
                source="github-inline",
                raw_sha256=_sha256_bytes(archive.inline_body),
                raw_size=len(archive.inline_body),
                recoverable=False,
            )
        if (
            archive.source != "github-signed-url"
            or archive.inline_body is not None
            or archive.signed_url is None
        ):
            raise BindingError(
                "signed archive preparation is invalid"
            )
        size, digest = self.archive_downloader(
            archive.signed_url,
            destination,
            timeout=self.timeout,
            max_bytes=self.max_archive_bytes,
        )
        return ArchiveSource(
            path=destination,
            source="github-signed-url",
            raw_sha256=digest,
            raw_size=size,
            recoverable=False,
        )


def _normalized_job_name(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _member_job_hint(name: str) -> tuple[int | None, str]:
    posix = PurePosixPath(name)
    if len(posix.parts) >= 2 and posix.name.casefold() == "system.txt":
        return None, posix.parts[-2]
    match = _MAIN_MEMBER_RE.fullmatch(posix.name)
    if match:
        return int(match.group("ordinal")), match.group("name")
    return None, posix.stem


def _job_for_member(
    name: str, jobs: Sequence[Mapping[str, object]]
) -> dict[str, object] | None:
    ordinal, hint = _member_job_hint(name)
    normalized_hint = _normalized_job_name(hint)
    exact = [
        dict(job)
        for job in jobs
        if _normalized_job_name(str(job.get("name") or "")) == normalized_hint
    ]
    if len(exact) == 1:
        return exact[0]
    if ordinal is not None and 0 <= ordinal < len(jobs):
        return dict(jobs[ordinal])
    return None


def _safe_zip_infos(
    archive: Path,
    *,
    max_members: int,
    max_member_bytes: int,
    max_uncompressed_bytes: int,
) -> list[zipfile.ZipInfo]:
    if archive.is_symlink() or not archive.is_file():
        raise ArchiveError(f"archive path is unsafe: {archive}")
    try:
        handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveError(f"invalid ZIP archive: {exc}") from exc
    with handle:
        infos = handle.infolist()
        if len(infos) > max_members:
            raise ArchiveError(
                f"ZIP member count {len(infos)} exceeds {max_members}"
            )
        names: set[str] = set()
        total = 0
        safe: list[zipfile.ZipInfo] = []
        for info in infos:
            name = info.filename
            if "\x00" in name or "\\" in name:
                raise ArchiveError(f"unsafe ZIP member name: {name!r}")
            pure = PurePosixPath(name)
            if pure.is_absolute() or any(part == ".." for part in pure.parts):
                raise ArchiveError(f"unsafe ZIP traversal member: {name!r}")
            if name in names:
                raise ArchiveError(f"duplicate ZIP member: {name!r}")
            names.add(name)
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                raise ArchiveError(f"ZIP symlink member is forbidden: {name!r}")
            if info.flag_bits & 0x1:
                raise ArchiveError(f"encrypted ZIP member is forbidden: {name!r}")
            if info.file_size < 0 or info.compress_size < 0:
                raise ArchiveError("ZIP member has negative size")
            if info.file_size > max_member_bytes:
                raise ArchiveError(
                    f"ZIP member {name!r} exceeds {max_member_bytes} bytes"
                )
            total += info.file_size
            if total > max_uncompressed_bytes:
                raise ArchiveError(
                    "ZIP uncompressed total exceeds configured limit"
                )
            if not info.is_dir():
                safe.append(info)
        return safe


def _read_zip_member(
    archive: Path, info: zipfile.ZipInfo, *, max_member_bytes: int
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        with zipfile.ZipFile(archive) as handle, handle.open(info) as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > max_member_bytes or total > info.file_size:
                    raise ArchiveError(
                        f"ZIP member changed size while reading: {info.filename}"
                    )
                chunks.append(block)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ArchiveError(
            f"cannot read ZIP member {info.filename!r}: {exc}"
        ) from exc
    if total != info.file_size:
        raise ArchiveError(
            f"ZIP member {info.filename!r} is truncated "
            f"({total}!={info.file_size})"
        )
    return b"".join(chunks)


def _load_rescue_manifest(root: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    path = root / "manifest.tsv"
    if not path.is_file():
        return {}
    records: dict[tuple[str, int, int], dict[str, str]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    fields = lines[0].split("\t")
    expected = [
        "repo",
        "run_id",
        "attempt",
        "created_at",
        "status",
        "bytes",
        "sha256",
        "finished_at",
    ]
    if fields != expected:
        raise ArchiveError("rescue manifest header is invalid")
    for line_no, line in enumerate(lines[1:], start=2):
        values = line.split("\t")
        if len(values) != len(fields):
            raise ArchiveError(
                f"rescue manifest line {line_no} has invalid field count"
            )
        record = dict(zip(fields, values))
        try:
            key = (
                record["repo"].casefold(),
                int(record["run_id"]),
                int(record["attempt"]),
            )
        except ValueError as exc:
            raise ArchiveError(
                f"rescue manifest line {line_no} has invalid identity"
            ) from exc
        previous = records.get(key)
        # The one-off rescue may have retried a failed record.  Prefer the
        # latest valid ZIP/terminal proof, otherwise retain the latest row.
        if previous is None or record["status"] in {"zip", "http410"}:
            records[key] = record
    return records


class RescueSpool:
    def __init__(self, root: str | os.PathLike[str] | None):
        self.root = (
            None if root is None else Path(root).expanduser().resolve()
        )
        self.manifest = (
            {} if self.root is None else _load_rescue_manifest(self.root)
        )

    @staticmethod
    def _base_name(attempt: Attempt) -> str:
        return (
            f"{attempt.repo.replace('/', '__')}--{attempt.run_id}"
            f"--attempt-{attempt.attempt}"
        )

    def locate(
        self, attempt: Attempt
    ) -> ArchiveSource | TerminalHTTP | None:
        if self.root is None:
            return None
        key = (attempt.repo.casefold(), attempt.run_id, attempt.attempt)
        manifest = self.manifest.get(key)
        candidates: list[tuple[str, Path]] = []
        for directory in (self.root, self.root / "consumed"):
            base = directory / self._base_name(attempt)
            candidates.extend(
                [
                    ("zip", base.with_suffix(".zip")),
                    ("http410", base.with_suffix(".http410.json")),
                    ("invalid", base.with_suffix(".invalid")),
                ]
            )
        for kind, path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            size = path.stat().st_size
            digest = _sha256_file(path)
            if (
                manifest is not None
                and manifest.get("status") in {"zip", "http410"}
            ):
                if (
                    int(manifest["bytes"]) != size
                    or manifest["sha256"] != digest
                ):
                    raise ArchiveError(
                        f"rescue artifact digest mismatch: {path.name}"
                    )
            if kind == "http410":
                return TerminalHTTP(410, path.read_bytes(), "rescue-spool")
            if kind == "invalid":
                raw = path.read_bytes()
                try:
                    payload = json.loads(raw)
                except (UnicodeError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, dict) and payload.get("status") in {
                    404,
                    410,
                }:
                    return TerminalHTTP(
                        int(payload["status"]), raw, "rescue-spool"
                    )
                if not zipfile.is_zipfile(path):
                    continue
            return ArchiveSource(
                path=path,
                source="rescue-spool",
                raw_sha256=digest,
                raw_size=size,
                recoverable=True,
            )
        return None

    def mark_consumed(self, source: ArchiveSource) -> None:
        if not source.recoverable or self.root is None:
            return
        consumed = self.root / "consumed"
        consumed.mkdir(exist_ok=True)
        if source.path.parent == consumed:
            return
        destination = consumed / source.path.name
        if destination.exists():
            if (
                destination.stat().st_size != source.raw_size
                or _sha256_file(destination) != source.raw_sha256
            ):
                raise ArchiveError(
                    f"conflicting consumed rescue archive: {destination.name}"
                )
            if source.path.exists():
                source.path.unlink()
            return
        os.replace(source.path, destination)
        _fsync_directory(consumed)
        _fsync_directory(self.root)


class CIStreamFetcher:
    def __init__(
        self,
        *,
        inventory_path: str | os.PathLike[str],
        state_path: str | os.PathLike[str],
        content_store_path: str | os.PathLike[str],
        tokenizer_path: str | os.PathLike[str],
        tokens: Sequence[str],
        progress_path: str | os.PathLike[str],
        receipt_path: str | os.PathLike[str],
        rescue_path: str | os.PathLike[str] | None = None,
        work_path: str | os.PathLike[str] | None = None,
        resume: bool = False,
        allow_fetcher_script_upgrade_from_sha256: str | None = None,
        fetcher_script_upgrade_reason: str | None = None,
        allow_parser_script_upgrade_from_sha256: str | None = None,
        parser_script_upgrade_reason: str | None = None,
        allow_content_store_script_upgrade_from_sha256: str | None = None,
        content_store_script_upgrade_reason: str | None = None,
        target_unique_tokens: int = DEFAULT_TARGET,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
        max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
        max_members: int = DEFAULT_MAX_MEMBERS,
        parser_workers: int = 0,
        parser: Callable[..., Mapping[str, object]] = canonicalize_ci_log,
        requester: Callable[
            [str, str, Mapping[str, str], float], HTTPResponse
        ] = _default_no_redirect_requester,
        archive_downloader: Callable[..., tuple[int, str]] = (
            _default_archive_downloader
        ),
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if target_unique_tokens <= 0:
            raise ValueError("target_unique_tokens must be positive")
        if (
            isinstance(parser_workers, bool)
            or not isinstance(parser_workers, int)
            or parser_workers < 0
        ):
            raise ValueError("parser_workers must be a non-negative integer")
        if parser_workers and parser is not canonicalize_ci_log:
            raise ValueError(
                "parser_workers requires the canonical production parser"
            )
        self.inventory_path = Path(inventory_path).expanduser().resolve()
        self.progress_path = Path(progress_path).expanduser().resolve()
        self.receipt_path = Path(receipt_path).expanduser().resolve()
        self.work_path = (
            Path(work_path).expanduser().resolve()
            if work_path is not None
            else Path(state_path).expanduser().resolve().with_suffix(".work")
        )
        self.work_path.mkdir(parents=True, exist_ok=True)
        (self.work_path / "tmp").mkdir(exist_ok=True)
        (self.work_path / "failed").mkdir(exist_ok=True)
        self.tokenizer = ExactTokenizer(tokenizer_path)
        self.store = CIContentStore(content_store_path)
        try:
            self.state = FetchState(
                state_path,
                inventory_path=self.inventory_path,
                content_store_path=content_store_path,
                tokenizer=self.tokenizer,
                resume=resume,
                # The store receipt reports the immutable producer binding,
                # not the hash of whichever read-only verifier opened it.
                content_store_creator_script_sha256=self.store.script_sha256,
                allow_fetcher_script_upgrade_from_sha256=(
                    allow_fetcher_script_upgrade_from_sha256
                ),
                fetcher_script_upgrade_reason=fetcher_script_upgrade_reason,
                allow_parser_script_upgrade_from_sha256=(
                    allow_parser_script_upgrade_from_sha256
                ),
                parser_script_upgrade_reason=parser_script_upgrade_reason,
                allow_content_store_script_upgrade_from_sha256=(
                    allow_content_store_script_upgrade_from_sha256
                ),
                content_store_script_upgrade_reason=(
                    content_store_script_upgrade_reason
                ),
            )
        except BaseException:
            self.store.close()
            raise
        self.client = GitHubAttemptClient(
            tokens,
            self.state,
            requester=requester,
            archive_downloader=archive_downloader,
            max_archive_bytes=max_archive_bytes,
            sleeper=sleeper,
        )
        self.rescue = RescueSpool(rescue_path)
        self.target_unique_tokens = target_unique_tokens
        self.max_chunk_chars = max_chunk_chars
        self.max_archive_bytes = max_archive_bytes
        self.max_member_bytes = max_member_bytes
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_members = max_members
        self.parser = parser
        self.parser_workers = parser_workers
        self._parser_executor = (
            ProcessPoolExecutor(
                max_workers=parser_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            if parser_workers
            else None
        )
        self.sleeper = sleeper

    def close(self) -> None:
        if self._parser_executor is not None:
            self._parser_executor.shutdown(wait=True, cancel_futures=False)
            self._parser_executor = None
        self.store.close()
        self.state.close()

    def _temp_archive_path(self, attempt: Attempt) -> Path:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=(
                f"{attempt.repo.replace('/', '__')}--{attempt.run_id}"
                f"--{attempt.attempt}--"
            ),
            suffix=".zip.partial",
            dir=self.work_path / "tmp",
        )
        os.close(descriptor)
        path = Path(raw_path)
        # Downloaders require exclusive creation so remove only this empty,
        # freshly allocated path inside the validated temp directory.
        path.unlink()
        return path

    def _process_member(
        self,
        attempt: Attempt,
        *,
        archive: ArchiveSource,
        info: zipfile.ZipInfo,
        jobs: Sequence[Mapping[str, object]],
    ) -> tuple[int, int]:
        if not attempt.run_metadata_exact:
            raise BindingError(
                "cannot parse a member without exact attempt metadata"
            )
        raw = _read_zip_member(
            archive.path, info, max_member_bytes=self.max_member_bytes
        )
        raw_sha = _sha256_bytes(raw)
        job = _job_for_member(info.filename, jobs)
        job_id = None if job is None else job.get("id")
        job_name = None if job is None else job.get("name")
        job_key = (
            f"{job_id if isinstance(job_id, int) else 'unresolved'}:"
            f"{info.filename}"
        )
        replayed = self.state.replayed_member(
            attempt,
            archive_member=info.filename,
            job_key=job_key,
            raw_sha256=raw_sha,
            raw_size=len(raw),
        )
        if replayed is not None:
            return replayed
        repository_identity = _repository_identity(attempt)
        metadata: dict[str, object] = dict(attempt.run_metadata)
        metadata.update(
            {
                "repository": repository_identity.canonical,
                "repository_requested": repository_identity.requested,
                "repository_id": repository_identity.repository_id,
                "source_repository": repository_identity.source,
                "source_repository_id": (
                    repository_identity.source_repository_id
                ),
                "run_id": attempt.run_id,
                "run_attempt": attempt.attempt,
                "job": job,
                "job_id": job_id,
                "job_name": job_name,
                "archive_member": info.filename,
                "archive_member_raw_sha256": raw_sha,
            }
        )
        if self._parser_executor is None:
            materialized = _materialize_parsed_member(
                raw,
                metadata,
                max_chunk_chars=self.max_chunk_chars,
                parser=self.parser,
                tokenizer=self.tokenizer,
            )
        else:
            materialized = self._parser_executor.submit(
                _process_parse_member,
                raw,
                metadata,
                self.max_chunk_chars,
                str(self.tokenizer.path),
            ).result()
        sidecar = materialized.get("sidecar")
        chunks = materialized.get("chunks")
        if (
            materialized.get("tokenizer_fingerprint")
            != self.tokenizer.fingerprint
            or not isinstance(chunks, list)
            or not isinstance(sidecar, dict)
            or any(not isinstance(item, dict) for item in chunks)
        ):
            raise FetchError("materialized parser result is invalid")
        records: list[dict[str, object]] = []
        occurrence_tokens = 0
        for materialized_chunk in chunks:
            ordinal = materialized_chunk.get("ordinal")
            text = materialized_chunk.get("text")
            token_count = materialized_chunk.get("token_count")
            sequence_sha = materialized_chunk.get(
                "token_sequence_sha256"
            )
            chunk = materialized_chunk.get("chunk")
            compact_section = materialized_chunk.get("section")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or not isinstance(text, str)
                or not text
                or isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count < 0
                or not isinstance(sequence_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", sequence_sha) is None
                or not isinstance(chunk, dict)
                or (
                    compact_section is not None
                    and not isinstance(compact_section, dict)
                )
            ):
                raise FetchError("materialized parser chunk is invalid")
            section_id = (
                str(chunk.get("section_id") or f"section:{ordinal}")
            )
            step_key = (
                f"{section_id}:"
                f"{chunk.get('step_ordinal') if chunk.get('step_ordinal') is not None else 'none'}"
            )
            provenance: dict[str, object] = {
                "schema": "cppmega_ci_chunk_occurrence_v3",
                "repository": repository_identity.canonical,
                "repository_requested": repository_identity.requested,
                "repository_id": repository_identity.repository_id,
                "source_repository": repository_identity.source,
                "source_repository_id": (
                    repository_identity.source_repository_id
                ),
                "repository_scope_key": attempt.repo,
                "run_id": attempt.run_id,
                "run_attempt": attempt.attempt,
                "run_metadata_evidence": {
                    "exact_attempt_match": attempt.run_metadata_exact,
                    "source": attempt.run_metadata_source,
                    "source_attempt": attempt.run_metadata_source_attempt,
                    "sha256": attempt.run_metadata_sha256,
                    "inventory_seed_attempt": (
                        attempt.inventory_seed_attempt
                    ),
                    "inventory_seed_metadata_sha256": (
                        attempt.inventory_seed_metadata_sha256
                    ),
                },
                "workflow": {
                    "id": attempt.run_metadata.get("workflow_id"),
                    "name": attempt.run_metadata.get("name"),
                    "path": attempt.run_metadata.get("path"),
                    "event": attempt.run_metadata.get("event"),
                    "run_number": attempt.run_metadata.get("run_number"),
                    "status": attempt.run_metadata.get("status"),
                    "conclusion": attempt.run_metadata.get("conclusion"),
                    "created_at": attempt.run_metadata.get("created_at"),
                    "updated_at": attempt.run_metadata.get("updated_at"),
                    "started_at": attempt.run_metadata.get(
                        "run_started_at"
                    ),
                    "display_title": attempt.run_metadata.get(
                        "display_title"
                    ),
                    "head_branch": attempt.run_metadata.get("head_branch"),
                    "head_sha": attempt.run_metadata.get("head_sha"),
                    "head_commit": attempt.run_metadata.get("head_commit"),
                    "actor": attempt.run_metadata.get("actor"),
                    "triggering_actor": attempt.run_metadata.get(
                        "triggering_actor"
                    ),
                },
                "job": job,
                "archive": {
                    "member": info.filename,
                    "member_raw_sha256": raw_sha,
                },
                "parser_sidecar_sha256": sidecar.get("sidecar_sha256"),
                "chunk": chunk,
                "section": compact_section,
            }
            records.append(
                {
                    "content": text,
                    "provenance": provenance,
                    "occurrence_key": {
                        "repo": attempt.repo,
                        "run_attempt": attempt.run_attempt_key,
                        "job": job_key,
                        "step": step_key,
                        "chunk_ordinal": ordinal,
                    },
                    "token_count": token_count,
                    "tokenizer_fingerprint": self.tokenizer.fingerprint,
                    "token_sequence_sha256": sequence_sha,
                }
            )
            occurrence_tokens += token_count
        if records:
            self.store.add_chunks(records)
        canonical_sha = materialized.get("canonical_sha256")
        dedup_sha = materialized.get("dedup_sha256")
        if (
            not isinstance(canonical_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", canonical_sha) is None
            or not isinstance(dedup_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", dedup_sha) is None
        ):
            raise FetchError("materialized member digests are invalid")
        self.state.store_member(
            attempt,
            archive_member=info.filename,
            job_key=job_key,
            raw_sha256=raw_sha,
            raw_size=len(raw),
            canonical_sha256=canonical_sha,
            dedup_sha256=dedup_sha,
            sidecar=sidecar,
            chunk_count=len(records),
            occurrence_tokens=occurrence_tokens,
        )
        return len(records), occurrence_tokens

    def process_attempt(self, attempt: Attempt) -> None:
        jobs: list[dict[str, Any]] | None = None
        archive: ArchiveSource | None = None
        temporary: Path | None = None
        try:
            rescued = self.rescue.locate(attempt)
            if isinstance(rescued, TerminalHTTP):
                raise rescued
            if not attempt.run_metadata_exact:
                exact_metadata = self.client.fetch_run_metadata(attempt)
                attempt = self.state.bind_exact_run_metadata(
                    attempt,
                    exact_metadata,
                )
            if isinstance(rescued, ArchiveSource):
                archive = rescued
                jobs = self.client.fetch_jobs(attempt)
            else:
                temporary = self._temp_archive_path(attempt)
                prepared = self.client.prepare_archive(attempt)
                jobs = self.client.fetch_jobs(attempt)
                archive = self.client.fetch_archive(
                    attempt,
                    temporary,
                    prepared=prepared,
                )
            if archive.raw_size > self.max_archive_bytes:
                raise ArchiveError("archive exceeds configured byte limit")
            if archive.path.stat().st_size != archive.raw_size:
                raise ArchiveError("archive size changed before processing")
            if _sha256_file(archive.path) != archive.raw_sha256:
                raise ArchiveError("archive digest changed before processing")
            infos = _safe_zip_infos(
                archive.path,
                max_members=self.max_members,
                max_member_bytes=self.max_member_bytes,
                max_uncompressed_bytes=self.max_uncompressed_bytes,
            )
            chunk_count = 0
            occurrence_tokens = 0
            for info in infos:
                member_chunks, member_tokens = self._process_member(
                    attempt, archive=archive, info=info, jobs=jobs
                )
                chunk_count += member_chunks
                occurrence_tokens += member_tokens
            status = "done" if chunk_count else "empty"
            self.state.finish_attempt(
                attempt,
                status=status,
                archive_source=archive.source,
                archive_sha256=archive.raw_sha256,
                archive_size=archive.raw_size,
                jobs=jobs,
                member_count=len(infos),
                chunk_count=chunk_count,
                occurrence_tokens=occurrence_tokens,
                secrets=self.client.secrets,
            )
            self.rescue.mark_consumed(archive)
        except TerminalHTTP as exc:
            if not self.state.fail_terminal_probe_with_durable_members(
                attempt,
                error=exc,
                secrets=self.client.secrets,
            ):
                status = (
                    "terminal_410" if exc.status == 410 else "terminal_404"
                )
                self.state.finish_attempt(
                    attempt,
                    status=status,
                    jobs=jobs,
                    terminal_http_status=exc.status,
                    terminal_body_sha256=_sha256_bytes(exc.body),
                    error=exc,
                    secrets=self.client.secrets,
                )
        except (APIError, ArchiveError, FetchError, OSError, zipfile.BadZipFile) as exc:
            with self.state._lock:
                tries_row = self.state._connection.execute(
                    """
                    SELECT tries FROM attempts
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (attempt.repo, attempt.run_id, attempt.attempt),
                ).fetchone()
            tries = 1 if tries_row is None else int(tries_row[0])
            retry = tries < 4
            self.state.finish_attempt(
                attempt,
                status="retry" if retry else "failed",
                archive_source=None if archive is None else archive.source,
                archive_sha256=None if archive is None else archive.raw_sha256,
                archive_size=None if archive is None else archive.raw_size,
                jobs=jobs,
                error=exc,
                retry=retry,
                secrets=self.client.secrets,
            )
        finally:
            if temporary is not None and temporary.exists():
                if archive is not None:
                    failed = self.work_path / "failed" / temporary.name
                    if failed.exists():
                        failed = failed.with_name(
                            f"{failed.name}.{int(time.time())}"
                        )
                    # Preserve a failed raw archive for diagnosis.  Successful
                    # attempts were already durably committed and can discard
                    # their bounded network temporary.
                    with self.state._lock:
                        row = self.state._connection.execute(
                            """
                            SELECT status FROM attempts
                            WHERE repo=? AND run_id=? AND attempt=?
                            """,
                            (attempt.repo, attempt.run_id, attempt.attempt),
                        ).fetchone()
                    terminal = None if row is None else str(row[0])
                    if terminal in {"done", "empty"}:
                        temporary.unlink()
                    else:
                        os.replace(temporary, failed)
                        _fsync_directory(failed.parent)
                else:
                    temporary.unlink()

    def progress(self) -> dict[str, object]:
        store_status = self.store.status()
        inventory_progress = None
        candidate = self.inventory_path.with_suffix(".progress.json")
        if candidate.is_file():
            try:
                inventory_progress = json.loads(
                    candidate.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                inventory_progress = None
        return {
            "schema": PROGRESS_SCHEMA,
            "generated_at": _utc_now(),
            "inventory": (
                {"path": str(self.inventory_path)}
                if inventory_progress is None
                else inventory_progress
            ),
            "fetch": self.state.summary(),
            "content_store": store_status,
            "token_accounting": {
                "semantics": (
                    "exact unique token-id sequences over canonical "
                    "dedup payloads after cppmega training whitespace "
                    "normalization; excludes framing and padding"
                ),
                "tokenizer_contract": self.tokenizer.contract,
                "tokenizer_fingerprint": self.tokenizer.fingerprint,
            },
            "target_exact_unique_payload_tokens": self.target_unique_tokens,
        }

    def write_progress(self) -> dict[str, object]:
        value = self.progress()
        atomic_write_json(self.progress_path, value)
        return value

    def threshold_met(self) -> bool:
        counters = self.store.status()["counters"]
        assert isinstance(counters, dict)
        value = counters.get("exact_unique_payload_tokens")
        return value is not None and int(value) >= self.target_unique_tokens

    def run(
        self,
        *,
        continuous: bool,
        max_runs: int | None = None,
        poll_seconds: float = 5.0,
        workers: int = 1,
    ) -> dict[str, object]:
        if workers <= 0:
            raise ValueError("workers must be positive")
        processed = 0
        submitted = 0
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ci-stream-fetch"
        ) as executor:
            while True:
                self.state.discover()
                futures: dict[Future[None], Attempt] = {}
                work_exhausted = False
                while True:
                    threshold_met = self.threshold_met()
                    while (
                        not work_exhausted
                        and len(futures) < workers
                        and (max_runs is None or submitted < max_runs)
                    ):
                        attempt = self.state.next_attempt(
                            retry_only=threshold_met
                        )
                        if attempt is None:
                            work_exhausted = True
                            break
                        future = executor.submit(
                            self.process_attempt, attempt
                        )
                        futures[future] = attempt
                        submitted += 1
                    if not futures:
                        break
                    completed, _pending = wait(
                        futures,
                        timeout=max(0.1, poll_seconds),
                        return_when=FIRST_COMPLETED,
                    )
                    if not completed:
                        self.write_progress()
                        continue
                    for future in completed:
                        future.result()
                        futures.pop(future)
                        processed += 1
                        self.write_progress()
                if self.threshold_met():
                    return self.write_progress()
                if max_runs is not None and submitted >= max_runs:
                    return self.write_progress()
                if not continuous:
                    return self.write_progress()
                self.write_progress()
                self.sleeper(max(0.1, poll_seconds))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream GitHub Actions attempt logs into the cppmega CI CAS"
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--content-store", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokens")
    parser.add_argument("--progress", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument(
        "--store-receipt",
        help=(
            "separate frozen content-store receipt; defaults beside --receipt"
        ),
    )
    parser.add_argument("--rescue-dir")
    parser.add_argument("--work-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-fetcher-script-upgrade-from-sha256",
        help=(
            "explicitly authorize one resume migration from this exact "
            "previous fetcher script SHA-256"
        ),
    )
    parser.add_argument(
        "--fetcher-script-upgrade-reason",
        help=(
            "required printable audit reason for an explicitly authorized "
            "fetcher script migration"
        ),
    )
    parser.add_argument(
        "--allow-parser-script-upgrade-from-sha256",
        help=(
            "explicitly authorize one resume migration from this exact "
            "previous CI sidecar parser SHA-256"
        ),
    )
    parser.add_argument(
        "--parser-script-upgrade-reason",
        help=(
            "required printable audit reason for an explicitly authorized "
            "CI sidecar parser migration"
        ),
    )
    parser.add_argument(
        "--allow-content-store-script-upgrade-from-sha256",
        help=(
            "explicitly authorize one resume migration from this exact "
            "previous content-store script SHA-256"
        ),
    )
    parser.add_argument(
        "--content-store-script-upgrade-reason",
        help=(
            "required printable audit reason for an explicitly authorized "
            "content-store script migration"
        ),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--parser-workers",
        type=int,
        default=8,
        help=(
            "spawned CPU workers for canonicalization/tokenization; "
            "use 0 only for deterministic inline diagnostics"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--target-exact-unique-payload-tokens",
        type=int,
        default=DEFAULT_TARGET,
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
    )
    parser.add_argument(
        "--max-archive-bytes",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_BYTES,
    )
    parser.add_argument(
        "--max-member-bytes",
        type=int,
        default=DEFAULT_MAX_MEMBER_BYTES,
    )
    parser.add_argument(
        "--max-uncompressed-bytes",
        type=int,
        default=DEFAULT_MAX_UNCOMPRESSED_BYTES,
    )
    parser.add_argument("--max-members", type=int, default=DEFAULT_MAX_MEMBERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_runs is not None and args.max_runs <= 0:
        raise SystemExit("--max-runs must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.parser_workers < 0:
        raise SystemExit("--parser-workers must be non-negative")
    tokens = load_token_pool(args.tokens)
    fetcher: CIStreamFetcher | None = None
    threshold_met = False
    try:
        fetcher = CIStreamFetcher(
            inventory_path=args.inventory,
            state_path=args.state,
            content_store_path=args.content_store,
            tokenizer_path=args.tokenizer,
            tokens=tokens,
            progress_path=args.progress,
            receipt_path=args.receipt,
            rescue_path=args.rescue_dir,
            work_path=args.work_dir,
            resume=args.resume,
            allow_fetcher_script_upgrade_from_sha256=(
                args.allow_fetcher_script_upgrade_from_sha256
            ),
            fetcher_script_upgrade_reason=(
                args.fetcher_script_upgrade_reason
            ),
            allow_parser_script_upgrade_from_sha256=(
                args.allow_parser_script_upgrade_from_sha256
            ),
            parser_script_upgrade_reason=(
                args.parser_script_upgrade_reason
            ),
            allow_content_store_script_upgrade_from_sha256=(
                args.allow_content_store_script_upgrade_from_sha256
            ),
            content_store_script_upgrade_reason=(
                args.content_store_script_upgrade_reason
            ),
            target_unique_tokens=args.target_exact_unique_payload_tokens,
            max_chunk_chars=args.max_chunk_chars,
            max_archive_bytes=args.max_archive_bytes,
            max_member_bytes=args.max_member_bytes,
            max_uncompressed_bytes=args.max_uncompressed_bytes,
            max_members=args.max_members,
            parser_workers=args.parser_workers,
        )
        result = fetcher.run(
            continuous=not args.once,
            max_runs=args.max_runs,
            poll_seconds=args.poll_seconds,
            workers=args.workers,
        )
        threshold_met = fetcher.threshold_met()
    except (FetchError, sqlite3.Error, OSError, ValueError) as exc:
        print(f"[ci-stream-fetch] ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if fetcher is not None:
            fetcher.close()
    if threshold_met:
        try:
            from scripts.ci_stream_receipts import finalize_fetch_receipts

            result = finalize_fetch_receipts(
                state_path=args.state,
                content_store_path=args.content_store,
                tokenizer_path=args.tokenizer,
                target_unique_tokens=args.target_exact_unique_payload_tokens,
                fetch_receipt_path=args.receipt,
                store_receipt_path=args.store_receipt,
                original_state_path=args.state,
                original_content_store_path=args.content_store,
                original_inventory_path=args.inventory,
            )
        except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
            print(f"[ci-stream-fetch] ERROR: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
