#!/usr/bin/env python3
"""Execute receipt-bound source jobs on transient local SSD.

Network sources are fetched with ``git clone --mirror``.  The worker records the
complete refs snapshot, HEAD, checkout tree, and all-object inventory before it
indexes the pinned commit.  Non-network sources must be immutable,
generation-pinned GCS tar.zst objects.  Workers deliberately do not receive the
tokenizer or a dedup database on the indexer command line: their output is a
canonical pre-global-dedup enriched stream, never a training-ready shard.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    gcs_join,
    load_json_object,
    require_exact_fields,
    require_git_object,
    require_int,
    require_sha256,
    run_checked,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.source_manifest import (  # noqa: E402
    PRE_GLOBAL_SCHEMA,
    load_source_manifest,
    repositories_for_worker,
    validate_source_manifest,
)
from scripts.distributed_data_prep.source_quarantine_projection import (  # noqa: E402
    PINNED_TREE_PROJECTION_MODE,
    build_pinned_tree_quarantine_projection,
)

SOURCE_WORKER_RECEIPT_SCHEMA = "cppmega.distributed_source_worker_receipt_v2"
ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA = (
    "cppmega.distributed_source_assignment_completion_receipt_v1"
)
CANONICAL_DOCUMENT_ORDER = "canonical_enriched_json_v1"
_SORT_FIELDS = (
    "repo",
    "filepath",
    "doc_type",
    "header_fragment_kind",
    "commit_hash",
    "file_local_commit_index",
)
_TRANSPORT_MAX_RETRIES = 4
_TRANSPORT_RETRY_BASE_SECONDS = 1.0
_TRANSPORT_RETRY_MAX_SECONDS = 16.0
_TRANSIENT_CURL_RETURN_CODES = frozenset({5, 6, 7, 18, 28, 35, 47, 52, 55, 56, 92})
_GIT_HTTP_STATUS_RE = re.compile(
    r"(?:\bHTTP(?:/[0-9.]+)?(?:\s+error)?\s*|\breturned error:\s*)([1-5][0-9]{2})",
    re.IGNORECASE,
)
_GIT_CURL_TRANSPORT_RE = re.compile(
    r"\bcurl\s+(5|6|7|18|28|35|47|52|55|56|92)\b", re.IGNORECASE
)
_TRANSIENT_GIT_MARKERS = (
    "connection reset",
    "connection timed out",
    "could not resolve host",
    "early eof",
    "empty reply from server",
    "internal server error",
    "network is unreachable",
    "operation timed out",
    "remote end hung up",
    "server closed the connection",
    "temporary failure",
    "tls connection was non-properly terminated",
    "unexpected disconnect",
)
_TRANSIENT_GCLOUD_MARKERS = _TRANSIENT_GIT_MARKERS + (
    "deadline exceeded",
    "resource exhausted",
    "service unavailable",
    "unavailable",
)
_NONRETRYABLE_GIT_MARKERS = (
    "authentication failed",
    "could not read username",
    "destination path",
    "not found",
    "permission denied",
    "repository not found",
)
QUARANTINE_PROJECTION_MODE_OFF = "off"
_QUARANTINE_PROJECTION_MODES = frozenset(
    {QUARANTINE_PROJECTION_MODE_OFF, PINNED_TREE_PROJECTION_MODE}
)
GIT_FSCK_RECEIPT_SCHEMA = "cppmega.git_fsck_receipt_v1"
_GIT_FSCK_COMMAND = ("git", "fsck", "--full", "--strict")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_KEYDB_ZERO_PADDED_FILEMODE = {
    "remote_url": "https://github.com/Snapchat/KeyDB.git",
    "expected_commit": "603ebb27fb82a27fb98b0feb6749b0f7661a1c4b",
    "checkout_tree": "f5269110f16e1833586e15dd59dde6255c8cc787",
    "historical_commit": "b435f64510a032528c42fc1cfc4eca15a4474a1b",
    "object_id": "1f9ef1b6556b375d56767fd78bf06c7d90e9abea",
    "object_type": "tree",
    "object_size_bytes": 532,
    "object_payload_sha256": (
        "3032eb4682653aa4b6b0b3a603de8181342139cce177f0842681f2f0f3537ffc"
    ),
    "message_id": "zeroPaddedFilemode",
    "diagnostic": (
        "error in tree 1f9ef1b6556b375d56767fd78bf06c7d90e9abea: "
        "zeroPaddedFilemode: contains zero-padded file modes"
    ),
    "returncode": 4,
}
SOURCE_QUARANTINE_RECEIPT_SCHEMA_V1 = "cppmega.source_quarantine_receipt_v1"
SOURCE_QUARANTINE_RECEIPT_SCHEMA_V2 = "cppmega.source_quarantine_receipt_v2"
SOURCE_TREE_ENTRY_EXCLUSIONS_SCHEMA = "cppmega.source_tree_entry_exclusions_v1"
_SOURCE_TREE_ENTRY_EXCLUSIONS_POLICY = (
    "skip_only_dangling_symlink_blobs_targeting_unmaterialized_gitlinks"
)


class TransientTransportError(RuntimeError):
    """A bounded transport retry budget was exhausted and may resume later."""


class _GcsRequestError(RuntimeError):
    """One GCS REST request failed with an observed status and response body."""

    def __init__(
        self,
        *,
        operation: str,
        uri: str,
        attempt: int,
        returncode: int,
        status: int | None,
        detail: str,
        transient: bool,
    ) -> None:
        self.status = status
        self.transient = transient
        status_text = "unknown" if status is None else str(status)
        super().__init__(
            f"GCS {operation} failed for {uri} on attempt {attempt}: "
            f"curl_exit={returncode} http={status_text}: {detail[-4000:]}"
        )


def _retry_delay_seconds(
    retry_index: int,
    *,
    base_seconds: float = _TRANSPORT_RETRY_BASE_SECONDS,
    maximum_seconds: float = _TRANSPORT_RETRY_MAX_SECONDS,
) -> float:
    """Return a deterministic, bounded exponential transport retry delay."""

    if isinstance(retry_index, bool) or retry_index < 0:
        raise ValueError("retry_index must be a non-negative integer")
    if base_seconds <= 0 or maximum_seconds < base_seconds:
        raise ValueError("invalid transport retry delay bounds")
    return min(base_seconds * (2**retry_index), maximum_seconds)


def _validate_retry_settings(
    *, max_retries: int, retry_base_seconds: float, retry_max_seconds: float
) -> None:
    if isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    _retry_delay_seconds(
        0,
        base_seconds=retry_base_seconds,
        maximum_seconds=retry_max_seconds,
    )


def _is_transient_http_status(status: int | None) -> bool:
    return status in {408, 429} or (status is not None and 500 <= status <= 599)


def _parse_curl_status(value: str) -> int | None:
    raw = value.strip()
    if not raw:
        return None
    if re.fullmatch(r"[0-9]{3}", raw) is None:
        raise ContractError(f"curl returned an invalid HTTP status: {raw!r}")
    return int(raw)


def _is_transient_curl_failure(*, returncode: int, status: int | None) -> bool:
    if status in {401, 403, 404, 409, 412}:
        return False
    if _is_transient_http_status(status):
        return True
    return returncode in _TRANSIENT_CURL_RETURN_CODES


def _is_transient_git_failure(
    *, returncode: int, stdout: str | None, stderr: str | None
) -> bool:
    """Classify only known retryable Git transport failures as transient."""

    if returncode == 0:
        return False
    detail = f"{stdout or ''}\n{stderr or ''}".lower()
    if any(marker in detail for marker in _NONRETRYABLE_GIT_MARKERS):
        return False
    statuses = [int(value) for value in _GIT_HTTP_STATUS_RE.findall(detail)]
    if any(status in {401, 403, 404, 409, 412} for status in statuses):
        return False
    if any(_is_transient_http_status(status) for status in statuses):
        return True
    if _GIT_CURL_TRANSPORT_RE.search(detail):
        return True
    return any(marker in detail for marker in _TRANSIENT_GIT_MARKERS)


def _is_transient_gcloud_failure(
    *, returncode: int, stdout: str | None, stderr: str | None
) -> bool:
    if returncode == 0:
        return False
    detail = f"{stdout or ''}\n{stderr or ''}".lower()
    if any(marker in detail for marker in _NONRETRYABLE_GIT_MARKERS):
        return False
    statuses = [int(value) for value in _GIT_HTTP_STATUS_RE.findall(detail)]
    if any(status in {401, 403, 404, 409, 412} for status in statuses):
        return False
    if any(_is_transient_http_status(status) for status in statuses):
        return True
    return any(marker in detail for marker in _TRANSIENT_GCLOUD_MARKERS)


def _run_git_network_command(
    command: Sequence[str | os.PathLike[str]],
    *,
    operation: str,
    before_attempt: Callable[[int], None] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    max_retries: int = _TRANSPORT_MAX_RETRIES,
    retry_base_seconds: float = _TRANSPORT_RETRY_BASE_SECONDS,
    retry_max_seconds: float = _TRANSPORT_RETRY_MAX_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run a Git clone/fetch command with bounded retry for transport failures."""

    _validate_retry_settings(
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )
    argv = [str(item) for item in command]
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    for attempt in range(max_retries + 1):
        if before_attempt is not None:
            before_attempt(attempt)
        completed = runner(
            argv,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return completed
        transient = _is_transient_git_failure(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if not transient or attempt == max_retries:
            exception = TransientTransportError if transient else RuntimeError
            raise exception(
                f"Git {operation} failed after {attempt + 1} attempt(s): {argv!r}\n"
                f"stdout:\n{(completed.stdout or '')[-8000:]}\n"
                f"stderr:\n{(completed.stderr or '')[-8000:]}"
            )
        delay = _retry_delay_seconds(
            attempt,
            base_seconds=retry_base_seconds,
            maximum_seconds=retry_max_seconds,
        )
        print(
            f"Git {operation} transient failure; retrying in {delay:.1f}s "
            f"({attempt + 1}/{max_retries})",
            file=sys.stderr,
            flush=True,
        )
        sleeper(delay)


def _remove_partial_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _tail_response(path: Path, fallback: str) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 4000))
            payload = stream.read()
    except OSError:
        return fallback[-4000:]
    text = payload.decode("utf-8", errors="replace").strip()
    return text[-4000:] if text else fallback[-4000:]


def _requested_generation(value: str | None, *, where: str) -> str | None:
    if value is None:
        return None
    generation = str(value)
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError(f"{where} must be a positive GCS generation")
    return generation


class ObjectStore(Protocol):
    def publish_if_absent(self, source: Path, uri: str) -> Mapping[str, object]: ...

    def download(
        self, uri: str, destination: Path, *, generation: str | None = None
    ) -> Mapping[str, object]: ...

    def describe_if_present(
        self, uri: str, *, generation: str | None = None
    ) -> Mapping[str, object] | None: ...


class GcloudObjectStore:
    """Generation-aware GCS transport without adding a Python dependency."""

    def __init__(
        self,
        executable: str = "gcloud",
        *,
        curl_executable: str = "curl",
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
        max_retries: int = _TRANSPORT_MAX_RETRIES,
        retry_base_seconds: float = _TRANSPORT_RETRY_BASE_SECONDS,
        retry_max_seconds: float = _TRANSPORT_RETRY_MAX_SECONDS,
    ) -> None:
        self.executable = executable
        self.curl_executable = curl_executable
        self.runner = runner
        self.sleeper = sleeper
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        _validate_retry_settings(
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )

    def _access_token(self) -> str:
        command = [self.executable, "auth", "print-access-token"]
        for attempt in range(self.max_retries + 1):
            completed = self.runner(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                token = (completed.stdout or "").strip()
                if not token or any(character.isspace() for character in token):
                    raise ContractError("gcloud returned an invalid access token")
                return token
            transient = _is_transient_gcloud_failure(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            detail = (
                f"stdout:\n{(completed.stdout or '')[-4000:]}\n"
                f"stderr:\n{(completed.stderr or '')[-4000:]}"
            )
            if not transient:
                raise RuntimeError(
                    f"gcloud access-token command failed after {attempt + 1} attempt(s): "
                    f"{detail}"
                )
            if attempt >= self.max_retries:
                raise TransientTransportError(
                    "gcloud access-token command exhausted its transport retry budget: "
                    f"{detail}"
                )
            delay = _retry_delay_seconds(
                attempt,
                base_seconds=self.retry_base_seconds,
                maximum_seconds=self.retry_max_seconds,
            )
            print(
                "gcloud access-token transient failure; "
                f"retrying in {delay:.1f}s ({attempt + 1}/{self.max_retries})",
                file=sys.stderr,
                flush=True,
            )
            self.sleeper(delay)
        raise AssertionError("unreachable")

    def _curl_once(
        self,
        *,
        token: str,
        endpoint: str,
        response: Path,
        config: str,
        method: str = "GET",
        upload: Path | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], int | None]:
        command = [
            self.curl_executable,
            "--config",
            "-",
            "--silent",
            "--show-error",
            "--location",
            "--output",
            str(response),
            "--write-out",
            "%{http_code}",
        ]
        if method != "GET":
            command.extend(["--request", method])
        if upload is not None:
            command.extend(["--upload-file", str(upload)])
        command.append(endpoint)
        completed = self.runner(
            command,
            input=config.replace("__TOKEN__", token),
            capture_output=True,
            text=True,
            check=False,
        )
        status = _parse_curl_status(completed.stdout or "")
        return completed, status

    def _request_with_retry(
        self,
        *,
        operation: str,
        uri: str,
        endpoint: str,
        response: Path,
        config: str,
        accepted_statuses: frozenset[int],
    ) -> int:
        for attempt in range(self.max_retries + 1):
            response.unlink(missing_ok=True)
            token = self._access_token()
            completed, status = self._curl_once(
                token=token,
                endpoint=endpoint,
                response=response,
                config=config,
            )
            if completed.returncode == 0 and status in accepted_statuses:
                return int(status)
            transient = _is_transient_curl_failure(
                returncode=completed.returncode,
                status=status,
            )
            detail = _tail_response(response, completed.stderr or "")
            if not transient:
                raise _GcsRequestError(
                    operation=operation,
                    uri=uri,
                    attempt=attempt + 1,
                    returncode=completed.returncode,
                    status=status,
                    detail=detail,
                    transient=False,
                )
            if attempt >= self.max_retries:
                raise TransientTransportError(
                    f"GCS {operation} exhausted {attempt + 1} transport attempt(s) for "
                    f"{uri}: curl_exit={completed.returncode} http={status}: {detail}"
                )
            delay = _retry_delay_seconds(
                attempt,
                base_seconds=self.retry_base_seconds,
                maximum_seconds=self.retry_max_seconds,
            )
            print(
                f"GCS {operation} transient failure; retrying in {delay:.1f}s "
                f"({attempt + 1}/{self.max_retries})",
                file=sys.stderr,
                flush=True,
            )
            self.sleeper(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _endpoint(uri: str, *, action: str, generation: str | None = None) -> str:
        bucket, object_name = uri[len("gs://") :].split("/", 1)
        prefix = f"{action}/" if action else ""
        endpoint = (
            f"https://storage.googleapis.com/{prefix}storage/v1/b/"
            f"{urllib.parse.quote(bucket, safe='')}/o/"
            f"{urllib.parse.quote(object_name, safe='')}"
        )
        query: dict[str, str] = {}
        if generation is not None:
            query["generation"] = generation
        if action == "download":
            query["alt"] = "media"
        if query:
            endpoint += "?" + urllib.parse.urlencode(query)
        return endpoint

    @staticmethod
    def _metadata_from_response(
        response: Path, *, uri: str, generation: str | None
    ) -> dict[str, object]:
        try:
            raw = json.loads(response.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"GCS returned invalid object metadata for {uri}") from exc
        if not isinstance(raw, dict):
            raise ContractError(f"GCS returned non-object metadata for {uri}")
        expected_name = uri[len("gs://") :].split("/", 1)[1]
        if raw.get("name") != expected_name:
            raise ContractError(f"GCS object name drifted for {uri}")
        resolved_generation = str(raw.get("generation", ""))
        if not resolved_generation.isdecimal() or int(resolved_generation) < 1:
            raise ContractError(f"GCS object has no valid generation: {uri}")
        if generation is not None and resolved_generation != generation:
            raise ContractError(f"GCS generation selector drifted for {uri}")
        try:
            size_int = int(raw["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"GCS object has no valid size: {uri}") from exc
        return {
            "uri": uri,
            "generation": resolved_generation,
            "size_bytes": size_int,
            "crc32c": raw.get("crc32c"),
            "md5_hash": raw.get("md5Hash"),
        }

    def describe(
        self, uri: str, *, generation: str | None = None
    ) -> dict[str, object]:
        validated = validate_gcs_uri(uri, where="GCS object")
        requested = _requested_generation(generation, where="GCS generation")
        result = self.describe_if_present(validated, generation=requested)
        if result is None:
            raise ContractError(f"GCS object is missing: {validated}")
        return dict(result)

    def describe_if_present(
        self, uri: str, *, generation: str | None = None
    ) -> dict[str, object] | None:
        """Return exact metadata for one known object without listing a bucket.

        Slot completion receipts have deterministic names.  The worker service
        account is deliberately unable to list objects, so resume has to make a
        single object GET and distinguish an absent receipt from an operational
        error.  A 404 is the only non-error negative result.
        """

        validated = validate_gcs_uri(uri, where="GCS object")
        requested = _requested_generation(generation, where="GCS generation")
        endpoint = self._endpoint(validated, action="", generation=requested)
        with tempfile.TemporaryDirectory(prefix="cppmega-gcs-describe-") as raw_tmp:
            response = Path(raw_tmp) / "response.json"
            status = self._request_with_retry(
                operation="object metadata lookup",
                uri=validated,
                endpoint=endpoint,
                response=response,
                config='header = "Authorization: Bearer __TOKEN__"\n',
                accepted_statuses=frozenset({200, 404}),
            )
            if status == 404:
                return None
            return self._metadata_from_response(
                response, uri=validated, generation=requested
            )

    def publish_if_absent(self, source: Path, uri: str) -> Mapping[str, object]:
        validated = validate_gcs_uri(uri, where="GCS publication URI")
        if not source.is_file():
            raise FileNotFoundError(source)
        bucket, object_name = uri[len("gs://") :].split("/", 1)
        endpoint = (
            "https://storage.googleapis.com/upload/storage/v1/b/"
            f"{urllib.parse.quote(bucket, safe='')}/o?"
            + urllib.parse.urlencode(
                {
                    "uploadType": "media",
                    "name": object_name,
                    "ifGenerationMatch": "0",
                }
            )
        )
        curl_config = (
            'header = "Authorization: Bearer __TOKEN__"\n'
            'header = "Content-Type: application/octet-stream"\n'
        )
        source_sha256 = sha256_file(source)
        with tempfile.TemporaryDirectory(prefix="cppmega-gcs-publish-") as raw_tmp:
            response = Path(raw_tmp) / "response.json"
            for attempt in range(self.max_retries + 1):
                response.unlink(missing_ok=True)
                token = self._access_token()
                completed, status = self._curl_once(
                    token=token,
                    endpoint=endpoint,
                    response=response,
                    config=curl_config,
                    method="POST",
                    upload=source,
                )
                if completed.returncode == 0 and status == 200:
                    metadata = self.describe(validated)
                    if int(metadata["size_bytes"]) != source.stat().st_size:
                        raise ContractError(
                            f"published GCS object size mismatch: {validated}"
                        )
                    return metadata

                transient = _is_transient_curl_failure(
                    returncode=completed.returncode,
                    status=status,
                )
                detail = _tail_response(response, completed.stderr or "")
                # A failed POST may have committed before the connection broke.
                # Reconcile every transient response and HTTP 412 by hashing the
                # immutable object before retrying or accepting it.
                if transient or status == 412:
                    with tempfile.TemporaryDirectory(
                        prefix="cppmega-gcs-publish-verify-"
                    ) as verify_tmp:
                        existing = self.describe_if_present(validated)
                        if existing is not None:
                            existing_path = Path(verify_tmp) / "existing"
                            self.download(
                                validated,
                                existing_path,
                                generation=str(existing["generation"]),
                            )
                            if sha256_file(existing_path) != source_sha256:
                                raise ContractError(
                                    "immutable GCS object already exists with different "
                                    f"bytes: {validated}"
                                )
                            return existing
                if not transient:
                    if status == 412:
                        raise ContractError(
                            "GCS immutable publication returned HTTP 412 but the "
                            f"existing object was not readable: {validated}"
                        )
                    raise _GcsRequestError(
                        operation="immutable publication",
                        uri=validated,
                        attempt=attempt + 1,
                        returncode=completed.returncode,
                        status=status,
                        detail=detail,
                        transient=False,
                    )
                if attempt >= self.max_retries:
                    raise TransientTransportError(
                        "GCS immutable publication exhausted "
                        f"{attempt + 1} transport attempt(s) for {validated}: {detail}"
                    )
                delay = _retry_delay_seconds(
                    attempt,
                    base_seconds=self.retry_base_seconds,
                    maximum_seconds=self.retry_max_seconds,
                )
                print(
                    "GCS immutable publication transient failure; "
                    f"retrying in {delay:.1f}s ({attempt + 1}/{self.max_retries})",
                    file=sys.stderr,
                    flush=True,
                )
                self.sleeper(delay)
        raise AssertionError("unreachable")

    def download(
        self, uri: str, destination: Path, *, generation: str | None = None
    ) -> Mapping[str, object]:
        validated = validate_gcs_uri(uri, where="GCS download URI")
        requested = _requested_generation(generation, where="GCS generation")
        endpoint = self._endpoint(validated, action="download", generation=requested)
        curl_config = 'header = "Authorization: Bearer __TOKEN__"\n'
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = destination.with_name(
            f".{destination.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
        )
        try:
            with tempfile.TemporaryDirectory(prefix="cppmega-gcs-download-") as raw_tmp:
                response = Path(raw_tmp) / "response.body"
                self._request_with_retry(
                    operation="exact download",
                    uri=validated,
                    endpoint=endpoint,
                    response=response,
                    config=curl_config,
                    accepted_statuses=frozenset({200}),
                )
                metadata = self.describe(validated, generation=requested)
                if response.stat().st_size != int(metadata["size_bytes"]):
                    raise ContractError(
                        f"downloaded GCS object size mismatch: {validated}"
                    )
                shutil.copyfile(response, stage)
            os.replace(stage, destination)
            return metadata
        finally:
            stage.unlink(missing_ok=True)


class LocalObjectStore:
    """Filesystem object store used by bounded smoke tests."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path(self, uri: str) -> Path:
        validate_gcs_uri(uri, where="local object URI")
        relative = uri[len("gs://") :]
        return self.root / relative

    def publish_if_absent(self, source: Path, uri: str) -> Mapping[str, object]:
        destination = self._path(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise ContractError(f"local immutable object collision: {uri}")
        else:
            stage = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            shutil.copyfile(source, stage)
            os.replace(stage, destination)
        return {
            "uri": uri,
            "generation": "1",
            "size_bytes": destination.stat().st_size,
            "crc32c": None,
            "md5_hash": None,
        }

    def download(
        self, uri: str, destination: Path, *, generation: str | None = None
    ) -> Mapping[str, object]:
        if generation not in {None, "1"}:
            raise ContractError(f"unknown local object generation for {uri}")
        source = self._path(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return {
            "uri": uri,
            "generation": "1",
            "size_bytes": destination.stat().st_size,
            "crc32c": None,
            "md5_hash": None,
        }

    def describe_if_present(
        self, uri: str, *, generation: str | None = None
    ) -> Mapping[str, object] | None:
        if generation not in {None, "1"}:
            raise ContractError(f"unknown local object generation for {uri}")
        source = self._path(uri)
        if not source.is_file():
            return None
        return {
            "uri": uri,
            "generation": "1",
            "size_bytes": source.stat().st_size,
            "crc32c": None,
            "md5_hash": None,
        }


def _git(git_dir: Path, *args: str) -> str:
    return run_checked(["git", f"--git-dir={git_dir}", *args]).stdout.strip()


def _matching_git_fsck_exception(
    source: Mapping[str, object],
    checkout_tree: str,
    *,
    known_exception: Mapping[str, object] = _KEYDB_ZERO_PADDED_FILEMODE,
) -> Mapping[str, object] | None:
    if (
        source.get("remote_url") == known_exception["remote_url"]
        and source.get("expected_commit") == known_exception["expected_commit"]
        and checkout_tree == known_exception["checkout_tree"]
    ):
        return known_exception
    return None


def _expected_git_fsck_exception_receipt(
    known_exception: Mapping[str, object],
) -> dict[str, object]:
    diagnostic = str(known_exception["diagnostic"])
    return {
        "schema": GIT_FSCK_RECEIPT_SCHEMA,
        "status": "accepted_known_historical_diagnostic",
        "command": list(_GIT_FSCK_COMMAND),
        "returncode": int(known_exception["returncode"]),
        "stdout_sha256": _EMPTY_SHA256,
        "stderr_sha256": hashlib.sha256(
            (diagnostic + "\n").encode("utf-8")
        ).hexdigest(),
        "source_binding": {
            "remote_url": str(known_exception["remote_url"]),
            "expected_commit": str(known_exception["expected_commit"]),
            "checkout_tree": str(known_exception["checkout_tree"]),
        },
        "diagnostics": [
            {
                "message_id": str(known_exception["message_id"]),
                "diagnostic": diagnostic,
                "object_id": str(known_exception["object_id"]),
                "object_type": str(known_exception["object_type"]),
                "object_size_bytes": int(known_exception["object_size_bytes"]),
                "object_payload_sha256": str(
                    known_exception["object_payload_sha256"]
                ),
                "historical_commit": str(known_exception["historical_commit"]),
            }
        ],
    }


def _accept_known_git_fsck_diagnostic(
    source: Mapping[str, object],
    checkout_tree: str,
    mirror: Path,
    fsck: subprocess.CompletedProcess[str],
    *,
    known_exception: Mapping[str, object] = _KEYDB_ZERO_PADDED_FILEMODE,
) -> dict[str, object]:
    """Accept one byte-verified historical diagnostic for one pinned source.

    The unmodified strict fsck command must emit exactly the known line and no
    other output.  We deliberately do not use ``fsck.skipList``: that setting
    would suppress every diagnostic attached to the listed object rather than
    proving that only the independently verified diagnostic was observed.
    """

    policy = _matching_git_fsck_exception(
        source,
        checkout_tree,
        known_exception=known_exception,
    )
    if policy is None:
        raise ContractError(f"full mirror failed git fsck: {fsck.stderr[-8000:]}")
    expected_receipt = _expected_git_fsck_exception_receipt(policy)
    expected_stderr = str(policy["diagnostic"]) + "\n"
    if (
        fsck.returncode != policy["returncode"]
        or fsck.stdout != ""
        or fsck.stderr != expected_stderr
    ):
        raise ContractError(
            "full mirror git fsck did not match the exact known historical "
            f"diagnostic: stdout={fsck.stdout[-4000:]!r} "
            f"stderr={fsck.stderr[-8000:]!r}"
        )

    object_id = str(policy["object_id"])
    object_type = str(policy["object_type"])
    if _git(mirror, "cat-file", "-t", object_id) != object_type:
        raise ContractError("known git fsck object type drifted")
    object_payload = subprocess.run(
        ["git", f"--git-dir={mirror}", "cat-file", object_type, object_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if object_payload.returncode != 0 or object_payload.stderr:
        raise ContractError(
            "cannot read the exact known git fsck object: "
            + object_payload.stderr[-4000:].decode(errors="replace")
        )
    if (
        len(object_payload.stdout) != policy["object_size_bytes"]
        or hashlib.sha256(object_payload.stdout).hexdigest()
        != policy["object_payload_sha256"]
    ):
        raise ContractError("known git fsck object payload drifted")

    historical_commit = str(policy["historical_commit"])
    if (
        _git(mirror, "rev-parse", f"{historical_commit}^{{commit}}")
        != historical_commit
        or _git(mirror, "rev-parse", f"{historical_commit}^{{tree}}") != object_id
    ):
        raise ContractError("known git fsck historical commit binding drifted")
    ancestry = subprocess.run(
        [
            "git",
            f"--git-dir={mirror}",
            "merge-base",
            "--is-ancestor",
            historical_commit,
            str(policy["expected_commit"]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestry.returncode != 0 or ancestry.stdout or ancestry.stderr:
        raise ContractError(
            "known git fsck object is not bound to the pinned commit ancestry"
        )
    return expected_receipt


def validate_git_fsck_snapshot(
    source: Mapping[str, object], source_snapshot: Mapping[str, object]
) -> None:
    """Validate normal fsck success or the one exact historical exception."""

    checkout_tree = str(source_snapshot.get("tree", ""))
    policy = _matching_git_fsck_exception(source, checkout_tree)
    fsck = source_snapshot.get("fsck")
    if policy is None:
        if fsck != "ok":
            raise ContractError("Git source snapshot has unsupported fsck evidence")
        return
    if not isinstance(fsck, Mapping):
        raise ContractError("Git source snapshot omitted known fsck diagnostic evidence")
    expected = _expected_git_fsck_exception_receipt(policy)
    if canonical_json_bytes(fsck) != canonical_json_bytes(expected):
        raise ContractError("Git source snapshot known fsck evidence drifted")


def _sorted_file_digest(source: Path, destination: Path) -> tuple[str, int, int, dict[str, int]]:
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    completed = subprocess.run(
        ["sort", "-o", str(destination), str(source)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"object inventory sort failed: {completed.stderr[-4000:]}")
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    types: dict[str, int] = {}
    with destination.open("rb") as stream:
        for line in stream:
            digest.update(line)
            fields = line.rstrip(b"\n").split(b" ")
            if len(fields) != 3:
                raise ContractError("Git object inventory line is malformed")
            object_type = fields[1].decode("ascii")
            try:
                object_size = int(fields[2])
            except ValueError as exc:
                raise ContractError("Git object inventory size is malformed") from exc
            count += 1
            total_bytes += object_size
            types[object_type] = types.get(object_type, 0) + 1
    return digest.hexdigest(), count, total_bytes, types


def acquire_git_mirror(
    source: Mapping[str, object],
    scratch: Path,
    *,
    git_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    max_retries: int = _TRANSPORT_MAX_RETRIES,
) -> tuple[Path, dict[str, object]]:
    """Clone a full mirror and materialize exactly the manifest-pinned commit."""

    remote = str(source["remote_url"])
    expected_commit = require_git_object(
        source["expected_commit"], where="git source expected_commit"
    )
    mirror = scratch / "mirror.git"
    checkout = scratch / "checkout"
    def reset_partial_clone(_attempt: int) -> None:
        _remove_partial_path(mirror)
        _remove_partial_path(checkout)

    _run_git_network_command(
        ["git", "clone", "--mirror", "--no-hardlinks", remote, mirror],
        operation="mirror clone",
        before_attempt=reset_partial_clone,
        runner=git_runner,
        sleeper=sleeper,
        max_retries=max_retries,
    )
    if _git(mirror, "rev-parse", "--is-bare-repository") != "true":
        raise ContractError("git clone --mirror did not create a bare repository")
    resolved_commit = _git(mirror, "rev-parse", f"{expected_commit}^{{commit}}")
    if resolved_commit != expected_commit:
        raise ContractError(
            f"mirror resolved a different commit: {resolved_commit} != {expected_commit}"
        )
    tree = _git(mirror, "rev-parse", f"{expected_commit}^{{tree}}")
    expected_tree = source.get("expected_tree")
    if expected_tree is not None and tree != expected_tree:
        raise ContractError(f"pinned Git tree drifted: {tree} != {expected_tree}")

    fsck = subprocess.run(
        ["git", f"--git-dir={mirror}", "fsck", "--full", "--strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    if fsck.returncode == 0:
        fsck_receipt: str | dict[str, object] = "ok"
    else:
        fsck_receipt = _accept_known_git_fsck_diagnostic(
            source,
            tree,
            mirror,
            fsck,
        )

    refs_lines = sorted(
        line
        for line in _git(
            mirror,
            "for-each-ref",
            "--format=%(refname)%00%(objectname)%00%(objecttype)",
        ).splitlines()
        if line
    )
    refs_payload = ("\n".join(refs_lines) + ("\n" if refs_lines else "")).encode(
        "utf-8"
    )
    try:
        head_ref = _git(mirror, "symbolic-ref", "-q", "HEAD")
    except RuntimeError:
        head_ref = None
    try:
        head_commit = _git(mirror, "rev-parse", "HEAD^{commit}")
    except RuntimeError:
        head_commit = None

    unordered = scratch / "objects.unsorted"
    ordered = scratch / "objects.sorted"
    with unordered.open("wb") as stream:
        completed = subprocess.run(
            [
                "git",
                f"--git-dir={mirror}",
                "cat-file",
                "--batch-all-objects",
                "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            ],
            stdout=stream,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git object inventory failed: {completed.stderr[-8000:].decode(errors='replace')}"
        )
    inventory_sha, object_count, object_bytes, object_types = _sorted_file_digest(
        unordered, ordered
    )

    run_checked(
        [
            "git",
            f"--git-dir={mirror}",
            "worktree",
            "add",
            "--detach",
            checkout,
            expected_commit,
        ]
    )
    checked_out = run_checked(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"]
    ).stdout.strip()
    if checked_out != expected_commit:
        raise ContractError("materialized worktree commit drifted")
    status = run_checked(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"]
    ).stdout
    if status:
        raise ContractError("fresh materialized worktree is dirty")
    gitlinks = run_checked(
        ["git", "-C", str(checkout), "ls-files", "--stage"]
    ).stdout.splitlines()
    gitlink_count = sum(1 for line in gitlinks if line.startswith("160000 "))
    return checkout, {
        "kind": "git_mirror",
        "remote_url": remote,
        "expected_commit": expected_commit,
        "resolved_commit": resolved_commit,
        "tree": tree,
        "head_ref": head_ref,
        "head_commit": head_commit,
        "refs": {
            "count": len(refs_lines),
            "sha256": hashlib.sha256(refs_payload).hexdigest(),
        },
        "objects": {
            "count": object_count,
            "logical_bytes": object_bytes,
            "types": dict(sorted(object_types.items())),
            "inventory_sha256": inventory_sha,
        },
        "gitlink_count": gitlink_count,
        "fsck": fsck_receipt,
    }


def _safe_archive_relative(name: str, strip_components: int) -> Path | None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError(f"immutable source archive contains unsafe path: {name!r}")
    if len(pure.parts) <= strip_components:
        return None
    return Path(*pure.parts[strip_components:])


def extract_immutable_tar_zst(
    archive: Path, destination: Path, *, strip_components: int
) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=False)
    zstd = subprocess.Popen(
        ["zstd", "-dc", "--", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert zstd.stdout is not None
    files = 0
    bytes_written = 0
    try:
        with tarfile.open(fileobj=zstd.stdout, mode="r|") as tar:
            for member in tar:
                relative = _safe_archive_relative(member.name, strip_components)
                if relative is None:
                    continue
                target = destination / relative
                try:
                    target.resolve().relative_to(destination.resolve())
                except ValueError as exc:
                    raise ContractError(
                        f"immutable source archive escaped extraction root: {member.name}"
                    ) from exc
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ContractError(
                        "immutable source archive contains a link/device/special entry: "
                        f"{member.name}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ContractError(f"cannot read tar member: {member.name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                if target.stat().st_size != member.size:
                    raise ContractError(f"tar member size mismatch: {member.name}")
                files += 1
                bytes_written += member.size
    finally:
        zstd.stdout.close()
    stderr = zstd.stderr.read() if zstd.stderr is not None else b""
    return_code = zstd.wait()
    if return_code != 0:
        raise RuntimeError(f"zstd extraction failed: {stderr[-8000:].decode(errors='replace')}")
    if files == 0:
        raise ContractError("immutable source archive contains no regular files")
    return {"file_count": files, "extracted_bytes": bytes_written}


def acquire_immutable_gcs_tar(
    source: Mapping[str, object], scratch: Path, store: ObjectStore
) -> tuple[Path, dict[str, object]]:
    uri = validate_gcs_uri(source["uri"], where="immutable source URI")
    generation = str(source["generation"])
    archive = scratch / "source.tar.zst"
    metadata = dict(store.download(uri, archive, generation=generation))
    if str(metadata.get("generation")) != generation:
        raise ContractError("immutable source object generation drifted")
    digest = sha256_file(archive)
    if digest != source["sha256"]:
        raise ContractError("immutable source object SHA-256 drifted")
    checkout = scratch / "checkout"
    extraction = extract_immutable_tar_zst(
        archive,
        checkout,
        strip_components=int(source["strip_components"]),
    )
    return checkout, {
        "kind": "immutable_gcs_tar",
        "object": {
            **metadata,
            "sha256": digest,
        },
        "archive_format": "tar.zst",
        "strip_components": int(source["strip_components"]),
        **extraction,
    }


def _canonical_sort_key(document: Mapping[str, object], payload_sha256: str) -> str:
    values = [str(document.get(field, "")) for field in _SORT_FIELDS]
    return json.dumps(values, ensure_ascii=True, separators=(",", ":")) + payload_sha256


def canonicalize_enriched_jsonl(
    source: Path,
    destination: Path,
    *,
    project_id: str,
    chunk_rows: int = 10_000,
) -> dict[str, object]:
    """Canonicalize JSON keys and externally sort documents with bounded RAM."""

    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunk_root = Path(tempfile.mkdtemp(prefix="candidate-sort-", dir=destination.parent))
    chunks: list[Path] = []
    rows: list[tuple[str, bytes]] = []
    documents = 0
    source_bytes = 0

    def flush() -> None:
        if not rows:
            return
        rows.sort(key=lambda item: (item[0], item[1]))
        path = chunk_root / f"chunk-{len(chunks):08d}.txt"
        with path.open("wb") as stream:
            for key, payload in rows:
                stream.write(key.encode("ascii"))
                stream.write(b"\t")
                stream.write(payload)
                stream.write(b"\n")
        chunks.append(path)
        rows.clear()

    try:
        with source.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                source_bytes += len(raw)
                if not raw.endswith(b"\n"):
                    raise ContractError(f"indexer JSONL line {line_number} is truncated")
                try:
                    document = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ContractError(
                        f"indexer JSONL line {line_number} is invalid"
                    ) from exc
                if not isinstance(document, dict):
                    raise ContractError(f"indexer JSONL line {line_number} is not an object")
                text = document.get("text")
                if not isinstance(text, str) or not text:
                    raise ContractError(
                        f"indexer JSONL line {line_number} has no non-empty text"
                    )
                row_repo = document.get("repo")
                if row_repo != project_id:
                    raise ContractError(
                        f"indexer JSONL line {line_number} repo drifted: {row_repo!r}"
                    )
                payload = canonical_json_bytes(document)
                payload_sha = hashlib.sha256(payload).hexdigest()
                rows.append((_canonical_sort_key(document, payload_sha), payload))
                documents += 1
                if len(rows) >= chunk_rows:
                    flush()
        flush()
        if documents == 0:
            raise ContractError("indexer emitted no pre-global-dedup documents")
        digest = hashlib.sha256()
        output_bytes = 0
        handles = [path.open("rb") for path in chunks]
        try:
            with destination.open("wb") as output:
                for encoded in heapq.merge(*handles):
                    _key, separator, payload = encoded.partition(b"\t")
                    if not separator or not payload.endswith(b"\n"):
                        raise ContractError("canonical sort spool is corrupt")
                    output.write(payload)
                    digest.update(payload)
                    output_bytes += len(payload)
        finally:
            for handle in handles:
                handle.close()
        return {
            "schema": PRE_GLOBAL_SCHEMA,
            "document_order": CANONICAL_DOCUMENT_ORDER,
            "documents": documents,
            "indexer_bytes": source_bytes,
            "canonical_bytes": output_bytes,
            "canonical_stream_sha256": digest.hexdigest(),
        }
    finally:
        shutil.rmtree(chunk_root, ignore_errors=True)


def compress_zstd(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        completed = subprocess.run(
            ["zstd", "-19", "-T1", "--no-progress", "-c", "--", str(source)],
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"zstd compression failed: {completed.stderr[-8000:].decode(errors='replace')}"
        )
    run_checked(["zstd", "-t", "--", str(destination)])
    version = run_checked(["zstd", "--version"]).stdout.strip()
    return {
        "compression": "zstd",
        "level": 19,
        "threads": 1,
        "zstd_version": version,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _verify_pipeline_files(
    manifest: Mapping[str, object],
    *,
    repo_root: Path,
    indexer: Path,
    tokenizer: Path,
    quarantine_manifest: Path,
) -> None:
    pipeline = manifest["pipeline"]
    assert isinstance(pipeline, Mapping)
    for path, field in (
        (indexer, "indexer_sha256"),
        (tokenizer, "tokenizer_sha256"),
        (quarantine_manifest, "quarantine_manifest_sha256"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"pipeline input is not a regular file: {path}")
        if sha256_file(path) != pipeline[field]:
            raise ContractError(f"pipeline input hash drifted: {path}")
    revision = run_checked(["git", "-C", str(repo_root), "rev-parse", "HEAD"]).stdout.strip()
    if revision != manifest["code_revision"]:
        raise ContractError("worker checkout does not match manifest code_revision")
    if run_checked(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"]
    ).stdout:
        raise ContractError("worker code checkout has tracked changes")


def _run_indexer(
    *,
    python: Path,
    indexer: Path,
    source_root: Path,
    project_id: str,
    raw_output: Path,
    quarantine_manifest: Path,
    quarantine_receipt: Path,
    parse_workers: int,
    memory_limit_gb: float,
    max_tokens: int,
) -> dict[str, object]:
    # No --tokenizer-path and no --dedup-db: both per-repo and global claims are
    # intentionally disabled.  The central reducer owns the first primary-copy
    # decision in canonical manifest/document order.
    command = [
        str(python),
        str(indexer),
        "--project-dir",
        str(source_root),
        "--project-id",
        project_id,
        "--output",
        str(raw_output),
        "--enriched",
        "--max-tokens",
        str(max_tokens),
        "--exclude-dirs",
        "__pycache__,node_modules,build,.git",
        "--memory-limit-gb",
        str(memory_limit_gb),
        "--parse-workers",
        str(parse_workers),
        "--source-quarantine-manifest",
        str(quarantine_manifest),
        "--source-quarantine-receipt",
        str(quarantine_receipt),
    ]
    run_checked(command, capture_output=False)
    if not raw_output.is_file() or raw_output.stat().st_size == 0:
        raise ContractError("indexer did not produce a non-empty enriched JSONL")
    if not quarantine_receipt.is_file():
        raise ContractError("indexer did not publish a quarantine receipt")
    return {
        # Keep the receipt retry-stable: the actual argv contains random local
        # scratch paths, while every semantic switch is captured below and the
        # manifest separately binds the exact indexer bytes.
        "mode": "single_project_pre_global_enriched_v1",
        "project_id": project_id,
        "enriched": True,
        "max_tokens": max_tokens,
        "parse_workers": parse_workers,
        "memory_limit_gb": memory_limit_gb,
        "excluded_directories": ["__pycache__", "node_modules", "build", ".git"],
        "dedup_applied": False,
        "tokenizer_passed_to_indexer": False,
        "raw_output_sha256": sha256_file(raw_output),
        "quarantine_receipt_sha256": sha256_file(quarantine_receipt),
    }


def _validate_source_tree_entry_exclusions(
    value: object,
    *,
    source_snapshot: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError("source tree entry exclusions receipt is missing")
    receipt = dict(value)
    require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "policy",
            "git_tree",
            "excluded_count",
            "records_sha256",
            "records",
        },
        where="source tree entry exclusions receipt",
    )
    if (
        receipt["schema"] != SOURCE_TREE_ENTRY_EXCLUSIONS_SCHEMA
        or receipt["status"] != "complete"
        or receipt["policy"] != _SOURCE_TREE_ENTRY_EXCLUSIONS_POLICY
    ):
        raise ContractError("source tree entry exclusions policy drifted")
    excluded_count = require_int(
        receipt["excluded_count"], where="source tree excluded count"
    )
    records = receipt["records"]
    if not isinstance(records, list) or len(records) != excluded_count:
        raise ContractError("source tree entry exclusion count does not close")
    if canonical_sha256(records) != require_sha256(
        receipt["records_sha256"],
        where="source tree entry exclusion records sha256",
    ):
        raise ContractError("source tree entry exclusion records digest drifted")
    git_tree = receipt["git_tree"]
    if excluded_count:
        git_tree = require_git_object(git_tree, where="source tree exclusion Git tree")
    elif git_tree is not None:
        raise ContractError("empty source tree exclusions must not claim a Git tree")
    if excluded_count and (
        source_snapshot.get("kind") != "git_mirror"
        or source_snapshot.get("tree") != git_tree
    ):
        raise ContractError(
            "source tree entry exclusions are not bound to the worker checkout tree"
        )

    expected_record_fields = {
        "relative_path",
        "reason",
        "git_tree",
        "entry_mode",
        "entry_object_id",
        "entry_object_type",
        "entry_object_size_bytes",
        "entry_object_sha256",
        "symlink_target",
        "target_relative_path",
        "target_gitlink_path",
        "target_gitlink_mode",
        "target_gitlink_commit",
    }
    previous_path: str | None = None
    for index, record_value in enumerate(records):
        if not isinstance(record_value, Mapping):
            raise ContractError(f"source tree exclusion record {index} is malformed")
        record = dict(record_value)
        require_exact_fields(
            record,
            expected_record_fields,
            where=f"source tree exclusion record {index}",
        )
        relative_path = str(record["relative_path"])
        target_relative_path = str(record["target_relative_path"])
        gitlink_path = str(record["target_gitlink_path"])
        for label, candidate in (
            ("relative_path", relative_path),
            ("target_relative_path", target_relative_path),
            ("target_gitlink_path", gitlink_path),
        ):
            parsed = PurePosixPath(candidate)
            if (
                not candidate
                or parsed.is_absolute()
                or ".." in parsed.parts
                or parsed.as_posix() != candidate
            ):
                raise ContractError(
                    f"source tree exclusion {label} is not canonical: {candidate!r}"
                )
        if previous_path is not None and relative_path <= previous_path:
            raise ContractError("source tree exclusion records are not canonical")
        previous_path = relative_path
        if not target_relative_path.startswith(gitlink_path + "/"):
            raise ContractError("source tree exclusion target escaped its gitlink")
        if (
            record["reason"]
            != "dangling_symlink_target_below_unmaterialized_gitlink"
            or record["git_tree"] != git_tree
            or record["entry_mode"] != "120000"
            or record["entry_object_type"] != "blob"
            or record["target_gitlink_mode"] != "160000"
        ):
            raise ContractError("source tree exclusion record semantics drifted")
        require_git_object(
            record["entry_object_id"], where="source tree symlink object id"
        )
        require_git_object(
            record["target_gitlink_commit"], where="source tree gitlink commit"
        )
        symlink_target = record["symlink_target"]
        if (
            not isinstance(symlink_target, str)
            or not symlink_target
            or "\0" in symlink_target
            or PurePosixPath(symlink_target).is_absolute()
        ):
            raise ContractError("source tree symlink target is invalid")
        expected_target = posixpath.normpath(
            posixpath.join(PurePosixPath(relative_path).parent.as_posix(), symlink_target)
        )
        if expected_target != target_relative_path:
            raise ContractError("source tree symlink target path drifted")
        symlink_payload = os.fsencode(symlink_target)
        if (
            require_int(
                record["entry_object_size_bytes"],
                where="source tree symlink object size",
                minimum=1,
            )
            != len(symlink_payload)
            or require_sha256(
                record["entry_object_sha256"],
                where="source tree symlink object sha256",
            )
            != hashlib.sha256(symlink_payload).hexdigest()
        ):
            raise ContractError("source tree symlink object payload drifted")
    return receipt


def validate_quarantine_receipt_file(
    path: Path,
    *,
    project_id: str,
    manifest_sha256: str,
    source_snapshot: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate the physical source-quarantine sidecar used by release audits."""

    raw, receipt = load_json_object(path, where="source quarantine receipt")
    base_fields = {
        "schema",
        "project_id",
        "manifest_path",
        "manifest_sha256",
        "manifest_entry_count",
        "project_manifest_entry_count",
        "candidate_count_before_quarantine",
        "candidate_count_after_quarantine",
        "quarantined_count",
        "entries",
        "external_reference_omissions",
        "parse_recovery",
    }
    schema = receipt.get("schema")
    if schema == SOURCE_QUARANTINE_RECEIPT_SCHEMA_V1:
        require_exact_fields(
            receipt,
            base_fields,
            where="source quarantine v1 receipt",
        )
    elif schema == SOURCE_QUARANTINE_RECEIPT_SCHEMA_V2:
        require_exact_fields(
            receipt,
            base_fields | {"source_tree_entry_exclusions"},
            where="source quarantine v2 receipt",
        )
        if not isinstance(source_snapshot, Mapping):
            raise ContractError("source quarantine v2 receipt omitted source snapshot")
        receipt["source_tree_entry_exclusions"] = (
            _validate_source_tree_entry_exclusions(
                receipt["source_tree_entry_exclusions"],
                source_snapshot=source_snapshot,
            )
        )
    else:
        raise ContractError("source quarantine receipt schema drifted")
    if (
        receipt["project_id"] != project_id
        or receipt["manifest_sha256"] != manifest_sha256
    ):
        raise ContractError("source quarantine receipt binding drifted")
    before = require_int(
        receipt["candidate_count_before_quarantine"],
        where="quarantine candidates before",
    )
    after = require_int(
        receipt["candidate_count_after_quarantine"],
        where="quarantine candidates after",
    )
    quarantined = require_int(
        receipt["quarantined_count"], where="quarantined count"
    )
    if after + quarantined != before:
        raise ContractError("source quarantine candidate counts do not close")
    entries = receipt["entries"]
    if not isinstance(entries, list) or len(entries) != quarantined:
        raise ContractError("source quarantine entry count does not close")
    require_int(receipt["manifest_entry_count"], where="quarantine manifest entries")
    require_int(
        receipt["project_manifest_entry_count"],
        where="project quarantine manifest entries",
    )
    for field, schema in (
        ("external_reference_omissions", "cppmega.external_reference_omissions_v1"),
        ("parse_recovery", "cppmega.source_parse_recovery_v1"),
    ):
        value = receipt[field]
        if (
            not isinstance(value, Mapping)
            or value.get("schema") != schema
            or not isinstance(value.get("status"), str)
        ):
            raise ContractError(f"source quarantine {field} receipt drifted")
    receipt["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    receipt["receipt_size_bytes"] = len(raw)
    return receipt


def validate_worker_receipt(
    receipt: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    job: Mapping[str, object],
) -> dict[str, object]:
    value = dict(receipt)
    expected_fields = {
        "schema",
        "status",
        "manifest_sha256",
        "manifest_file_sha256",
        "assignment",
        "source_snapshot",
        "candidate",
        "artifact",
        "quarantine_artifact",
        "indexer",
        "training_ready",
    }
    has_projection = "quarantine_projection_artifact" in value
    if has_projection:
        expected_fields.add("quarantine_projection_artifact")
    require_exact_fields(value, expected_fields, where="source worker receipt")
    if value["schema"] != SOURCE_WORKER_RECEIPT_SCHEMA or value["status"] != "complete":
        raise ContractError("source worker receipt schema/status is unsupported")
    if value["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ContractError("source worker receipt manifest binding drifted")
    require_sha256(value["manifest_file_sha256"], where="manifest_file_sha256")
    if value["training_ready"] is not False:
        raise ContractError("worker candidate must never claim training readiness")
    assignment = value["assignment"]
    if not isinstance(assignment, Mapping) or dict(assignment) != {
        key: job[key]
        for key in ("ordinal", "repo", "project_id", "worker", "assignment_sha256")
    }:
        raise ContractError("source worker assignment binding drifted")
    source = job.get("source")
    source_snapshot = value["source_snapshot"]
    if not isinstance(source, Mapping) or not isinstance(source_snapshot, Mapping):
        raise ContractError("source worker source snapshot is malformed")
    if source.get("kind") == "git_mirror":
        validate_git_fsck_snapshot(source, source_snapshot)
    candidate = value["candidate"]
    if not isinstance(candidate, Mapping):
        raise ContractError("source worker candidate receipt is missing")
    if (
        candidate.get("schema") != PRE_GLOBAL_SCHEMA
        or candidate.get("document_order") != CANONICAL_DOCUMENT_ORDER
        or candidate.get("dedup_applied") is not False
    ):
        raise ContractError("source worker candidate is not pre-global-dedup")
    require_int(candidate.get("documents"), where="candidate.documents", minimum=1)
    require_sha256(
        candidate.get("canonical_stream_sha256"),
        where="candidate.canonical_stream_sha256",
    )
    artifact = value["artifact"]
    if not isinstance(artifact, Mapping):
        raise ContractError("source worker artifact receipt is missing")
    require_exact_fields(
        artifact,
        {
            "uri",
            "generation",
            "size_bytes",
            "crc32c",
            "md5_hash",
            "sha256",
            "compression",
        },
        where="source worker artifact",
    )
    artifact_uri = validate_gcs_uri(artifact.get("uri"), where="worker artifact URI")
    generation = str(artifact.get("generation", ""))
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError("source worker artifact generation is invalid")
    artifact_sha256 = require_sha256(
        artifact.get("sha256"), where="worker artifact sha256"
    )
    require_int(artifact.get("size_bytes"), where="worker artifact size", minimum=1)
    expected_uri = gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-candidates",
        str(manifest["manifest_sha256"]),
        f"{int(job['ordinal']):05d}-{job['repo']}",
        f"{artifact_sha256}.jsonl.zst",
    )
    if artifact_uri != expected_uri:
        raise ContractError("source worker artifact URI escaped its manifest namespace")
    compression = artifact.get("compression")
    if (
        not isinstance(compression, Mapping)
        or compression.get("compression") != "zstd"
        or compression.get("sha256") != artifact.get("sha256")
        or compression.get("size_bytes") != artifact.get("size_bytes")
    ):
        raise ContractError("source worker artifact compression binding drifted")
    quarantine_artifact = value["quarantine_artifact"]
    if not isinstance(quarantine_artifact, Mapping):
        raise ContractError("source worker quarantine artifact receipt is missing")
    require_exact_fields(
        quarantine_artifact,
        {"uri", "generation", "size_bytes", "crc32c", "md5_hash", "sha256"},
        where="source worker quarantine artifact",
    )
    quarantine_sha256 = require_sha256(
        quarantine_artifact.get("sha256"), where="quarantine artifact sha256"
    )
    quarantine_uri = validate_gcs_uri(
        quarantine_artifact.get("uri"), where="quarantine artifact URI"
    )
    expected_quarantine_uri = gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-quarantine-receipts",
        str(manifest["manifest_sha256"]),
        f"{int(job['ordinal']):05d}-{job['repo']}",
        f"{quarantine_sha256}.quarantine.json",
    )
    if quarantine_uri != expected_quarantine_uri:
        raise ContractError("quarantine artifact URI escaped its manifest namespace")
    quarantine_generation = str(quarantine_artifact.get("generation", ""))
    if not quarantine_generation.isdecimal() or int(quarantine_generation) < 1:
        raise ContractError("quarantine artifact generation is invalid")
    require_int(
        quarantine_artifact.get("size_bytes"),
        where="quarantine artifact size",
        minimum=1,
    )
    indexer = value["indexer"]
    if not isinstance(indexer, Mapping):
        raise ContractError("source worker indexer receipt is missing")
    if (
        indexer.get("mode") != "single_project_pre_global_enriched_v1"
        or indexer.get("project_id") != job["project_id"]
        or indexer.get("enriched") is not True
        or indexer.get("dedup_applied") is not False
        or indexer.get("tokenizer_passed_to_indexer") is not False
    ):
        raise ContractError("source worker indexer execution contract drifted")
    require_sha256(indexer.get("raw_output_sha256"), where="indexer raw output sha256")
    require_sha256(
        indexer.get("quarantine_receipt_sha256"),
        where="indexer quarantine receipt sha256",
    )
    if indexer.get("quarantine_receipt_sha256") != quarantine_sha256:
        raise ContractError("indexer quarantine receipt publication drifted")
    if has_projection:
        projection_artifact = value["quarantine_projection_artifact"]
        if not isinstance(projection_artifact, Mapping):
            raise ContractError(
                "source worker quarantine projection artifact receipt is missing"
            )
        require_exact_fields(
            projection_artifact,
            {"uri", "generation", "size_bytes", "crc32c", "md5_hash", "sha256"},
            where="source worker quarantine projection artifact",
        )
        projection_sha256 = require_sha256(
            projection_artifact.get("sha256"),
            where="quarantine projection artifact sha256",
        )
        projection_uri = validate_gcs_uri(
            projection_artifact.get("uri"),
            where="quarantine projection artifact URI",
        )
        expected_projection_uri = gcs_join(
            str(manifest["gcs_output_prefix"]),
            "source-quarantine-projections",
            str(manifest["manifest_sha256"]),
            f"{int(job['ordinal']):05d}-{job['repo']}",
            f"{projection_sha256}.projection.json",
        )
        if projection_uri != expected_projection_uri:
            raise ContractError(
                "quarantine projection artifact URI escaped its manifest namespace"
            )
        projection_generation = str(projection_artifact.get("generation", ""))
        if not projection_generation.isdecimal() or int(projection_generation) < 1:
            raise ContractError("quarantine projection artifact generation is invalid")
        require_int(
            projection_artifact.get("size_bytes"),
            where="quarantine projection artifact size",
            minimum=1,
        )
    return value


def assignment_completion_uri(
    manifest: Mapping[str, object], job: Mapping[str, object]
) -> str:
    """Return the deterministic resume pointer for one manifest assignment."""

    assignment_sha256 = require_sha256(
        job.get("assignment_sha256"), where="assignment_sha256"
    )
    return gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-assignment-completions",
        str(manifest["manifest_sha256"]),
        f"{assignment_sha256}.complete.json",
    )


def _source_receipt_uri(
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    receipt: Mapping[str, object],
) -> str:
    artifact = receipt.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ContractError("source worker receipt has no artifact")
    compression = artifact.get("compression")
    if not isinstance(compression, Mapping):
        raise ContractError("source worker receipt has no compression metadata")
    compressed_sha256 = require_sha256(
        compression.get("sha256"), where="source receipt compressed sha256"
    )
    return gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-receipts",
        str(manifest["manifest_sha256"]),
        f"{int(job['ordinal']):05d}-{job['repo']}",
        f"{compressed_sha256}.receipt.json",
    )


def validate_assignment_completion_receipt(
    receipt: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
) -> dict[str, object]:
    """Validate the immutable pointer that makes one job safely resumable."""

    value = dict(receipt)
    require_exact_fields(
        value,
        {
            "schema",
            "status",
            "manifest_sha256",
            "manifest_file_sha256",
            "assignment",
            "source_receipt",
            "training_ready",
        },
        where="source assignment completion receipt",
    )
    if (
        value["schema"] != ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA
        or value["status"] != "complete"
        or value["manifest_sha256"] != manifest["manifest_sha256"]
        or value["manifest_file_sha256"] != manifest_file_sha256
        or value["training_ready"] is not False
    ):
        raise ContractError("source assignment completion receipt binding drifted")
    expected_assignment = {
        key: job[key]
        for key in ("ordinal", "repo", "project_id", "worker", "assignment_sha256")
    }
    assignment = value["assignment"]
    if not isinstance(assignment, Mapping) or dict(assignment) != expected_assignment:
        raise ContractError("source assignment completion assignment drifted")
    source_receipt = value["source_receipt"]
    if not isinstance(source_receipt, Mapping):
        raise ContractError("source assignment completion source receipt is missing")
    source = dict(source_receipt)
    require_exact_fields(
        source,
        {"uri", "generation", "size_bytes", "sha256"},
        where="source assignment completion source receipt",
    )
    validate_gcs_uri(source["uri"], where="source assignment completion receipt URI")
    generation = str(source["generation"])
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError("source assignment completion receipt generation is invalid")
    require_int(
        source["size_bytes"],
        where="source assignment completion receipt size",
        minimum=1,
    )
    require_sha256(source["sha256"], where="source assignment completion receipt sha256")
    return value


def _load_completed_assignment(
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object] | None:
    """Return a read-back source receipt for a confirmed assignment, if any."""

    pointer_uri = assignment_completion_uri(manifest, job)
    pointer_metadata = object_store.describe_if_present(pointer_uri)
    if pointer_metadata is None:
        return None
    pointer_generation = str(pointer_metadata.get("generation", ""))
    if not pointer_generation.isdecimal() or int(pointer_generation) < 1:
        raise ContractError(f"assignment completion pointer has invalid generation: {pointer_uri}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="assignment-resume-", dir=scratch_root) as raw_tmp:
        temporary = Path(raw_tmp)
        pointer_path = temporary / "pointer.json"
        downloaded_pointer = object_store.download(
            pointer_uri, pointer_path, generation=pointer_generation
        )
        if (
            str(downloaded_pointer.get("uri")) != pointer_uri
            or str(downloaded_pointer.get("generation")) != pointer_generation
            or pointer_path.stat().st_size != int(pointer_metadata.get("size_bytes", -1))
        ):
            raise ContractError("assignment completion pointer readback drifted")
        _pointer_raw, pointer = load_json_object(
            pointer_path, where="source assignment completion receipt"
        )
        validated_pointer = validate_assignment_completion_receipt(
            pointer,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            job=job,
        )
        source = validated_pointer["source_receipt"]
        assert isinstance(source, Mapping)
        source_uri = validate_gcs_uri(source["uri"], where="source receipt URI")
        source_generation = str(source["generation"])
        source_metadata = object_store.describe_if_present(
            source_uri, generation=source_generation
        )
        if source_metadata is None:
            raise ContractError("assignment pointer references a missing source receipt")
        source_path = temporary / "source-receipt.json"
        downloaded_source = object_store.download(
            source_uri, source_path, generation=source_generation
        )
        if (
            str(downloaded_source.get("uri")) != source_uri
            or str(downloaded_source.get("generation")) != source_generation
            or source_path.stat().st_size != int(source["size_bytes"])
            or sha256_file(source_path) != source["sha256"]
        ):
            raise ContractError("assignment completion source receipt readback drifted")
        _source_raw, source_receipt = load_json_object(
            source_path, where="source worker receipt"
        )
        validate_worker_receipt(source_receipt, manifest=manifest, job=job)
        if _source_receipt_uri(manifest, job, source_receipt) != source_uri:
            raise ContractError("assignment pointer source receipt URI drifted")
        return source_receipt


def _publish_assignment_completion(
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
    source_receipt: Mapping[str, object],
    source_receipt_path: Path,
    object_store: ObjectStore,
    scratch_root: Path,
) -> None:
    """Publish and read back a deterministic pointer after the source receipt."""

    receipt_uri = _source_receipt_uri(manifest, job, source_receipt)
    metadata = object_store.describe_if_present(receipt_uri)
    if metadata is None:
        raise ContractError("source receipt disappeared before assignment completion")
    generation = str(metadata.get("generation", ""))
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError("source receipt has invalid generation")
    source_sha256 = sha256_file(source_receipt_path)
    with tempfile.TemporaryDirectory(prefix="assignment-publish-", dir=scratch_root) as raw_tmp:
        temporary = Path(raw_tmp)
        verified_source = temporary / "source-receipt.json"
        downloaded = object_store.download(
            receipt_uri, verified_source, generation=generation
        )
        if (
            str(downloaded.get("uri")) != receipt_uri
            or str(downloaded.get("generation")) != generation
            or verified_source.stat().st_size != source_receipt_path.stat().st_size
            or sha256_file(verified_source) != source_sha256
        ):
            raise ContractError("published source receipt readback drifted")
        pointer: dict[str, object] = {
            "schema": ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": manifest_file_sha256,
            "assignment": {
                key: job[key]
                for key in (
                    "ordinal",
                    "repo",
                    "project_id",
                    "worker",
                    "assignment_sha256",
                )
            },
            "source_receipt": {
                "uri": receipt_uri,
                "generation": generation,
                "size_bytes": source_receipt_path.stat().st_size,
                "sha256": source_sha256,
            },
            "training_ready": False,
        }
        validate_assignment_completion_receipt(
            pointer,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            job=job,
        )
        pointer_path = temporary / "assignment-completion.json"
        atomic_write_json(pointer_path, pointer)
        pointer_uri = assignment_completion_uri(manifest, job)
        published = object_store.publish_if_absent(pointer_path, pointer_uri)
        pointer_generation = str(published.get("generation", ""))
        if str(published.get("uri")) != pointer_uri or not pointer_generation.isdecimal():
            raise ContractError("assignment completion pointer publication drifted")
        verified_pointer = temporary / "assignment-completion.verify.json"
        pointer_download = object_store.download(
            pointer_uri, verified_pointer, generation=pointer_generation
        )
        if (
            str(pointer_download.get("generation")) != pointer_generation
            or verified_pointer.stat().st_size != pointer_path.stat().st_size
            or sha256_file(verified_pointer) != sha256_file(pointer_path)
        ):
            raise ContractError("assignment completion pointer readback drifted")


def run_source_worker(
    manifest: Mapping[str, object],
    *,
    manifest_file_sha256: str,
    worker: str,
    scratch_root: Path,
    receipt_root: Path,
    repo_root: Path,
    python: Path,
    indexer: Path,
    tokenizer: Path,
    quarantine_manifest: Path,
    object_store: ObjectStore,
    parse_workers: int = 4,
    memory_limit_gb: float = 14.0,
    max_tokens: int | None = None,
    assignment_sha256: str | None = None,
    quarantine_projection_mode: str = QUARANTINE_PROJECTION_MODE_OFF,
) -> tuple[dict[str, object], ...]:
    """Run one selected assignment or every assignment owned by a worker."""

    plan = validate_source_manifest(manifest)
    require_sha256(manifest_file_sha256, where="manifest_file_sha256")
    pipeline = plan["pipeline"]
    assert isinstance(pipeline, Mapping)
    manifest_max_tokens = require_int(
        pipeline.get("index_max_tokens"),
        where="pipeline.index_max_tokens",
        minimum=1,
    )
    if max_tokens is None:
        max_tokens = manifest_max_tokens
    if max_tokens != manifest_max_tokens:
        raise ContractError("worker max_tokens drifted from the source manifest")
    if parse_workers < 1 or memory_limit_gb <= 0:
        raise ValueError("worker resource/index limits must be positive")
    if quarantine_projection_mode not in _QUARANTINE_PROJECTION_MODES:
        raise ValueError(
            "unsupported quarantine projection mode: "
            f"{quarantine_projection_mode!r}"
        )
    jobs = repositories_for_worker(plan, worker)
    if assignment_sha256 is not None:
        selected_sha256 = require_sha256(
            assignment_sha256, where="selected assignment_sha256"
        )
        jobs = tuple(job for job in jobs if job["assignment_sha256"] == selected_sha256)
        if len(jobs) != 1:
            raise ContractError(
                "selected assignment is not owned by the requested manifest worker"
            )
    _verify_pipeline_files(
        plan,
        repo_root=repo_root,
        indexer=indexer,
        tokenizer=tokenizer,
        quarantine_manifest=quarantine_manifest,
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, object]] = []
    for job in jobs:
        resumed = _load_completed_assignment(
            manifest=plan,
            manifest_file_sha256=manifest_file_sha256,
            job=job,
            object_store=object_store,
            scratch_root=scratch_root,
        )
        if resumed is not None:
            local_receipt = receipt_root / f"{int(job['ordinal']):05d}-{job['repo']}.json"
            atomic_write_json(local_receipt, resumed)
            receipts.append(resumed)
            continue
        with tempfile.TemporaryDirectory(
            prefix=f"source-{job['ordinal']:05d}-{job['repo']}-", dir=scratch_root
        ) as raw_scratch:
            scratch = Path(raw_scratch)
            source = job["source"]
            assert isinstance(source, Mapping)
            if source["kind"] == "git_mirror":
                checkout, source_snapshot = acquire_git_mirror(source, scratch)
            elif source["kind"] == "immutable_gcs_tar":
                checkout, source_snapshot = acquire_immutable_gcs_tar(
                    source, scratch, object_store
                )
            else:  # validate_source_manifest already rejects this.
                raise ContractError(f"unsupported source kind: {source['kind']}")

            effective_quarantine_manifest = quarantine_manifest
            projection_receipt_path: Path | None = None
            if quarantine_projection_mode == PINNED_TREE_PROJECTION_MODE:
                effective_quarantine_manifest = scratch / "projected-quarantine.json"
                projection_receipt_path = scratch / "quarantine-projection.json"
                build_pinned_tree_quarantine_projection(
                    base_manifest_path=quarantine_manifest,
                    source_root=checkout,
                    project_id=str(job["project_id"]),
                    source_snapshot=source_snapshot,
                    projected_manifest_path=effective_quarantine_manifest,
                    receipt_path=projection_receipt_path,
                )

            raw_output = scratch / "pre-global.enriched.jsonl"
            quarantine_receipt = scratch / "source-quarantine-receipt.json"
            indexer_receipt = _run_indexer(
                python=python,
                indexer=indexer,
                source_root=checkout,
                project_id=str(job["project_id"]),
                raw_output=raw_output,
                quarantine_manifest=effective_quarantine_manifest,
                quarantine_receipt=quarantine_receipt,
                parse_workers=parse_workers,
                memory_limit_gb=memory_limit_gb,
                max_tokens=max_tokens,
            )
            validated_quarantine = validate_quarantine_receipt_file(
                quarantine_receipt,
                project_id=str(job["project_id"]),
                manifest_sha256=sha256_file(effective_quarantine_manifest),
                source_snapshot=source_snapshot,
            )
            if (
                validated_quarantine["receipt_sha256"]
                != indexer_receipt["quarantine_receipt_sha256"]
            ):
                raise ContractError("indexer quarantine receipt digest drifted")
            projection_artifact: dict[str, object] | None = None
            if projection_receipt_path is not None:
                projection_sha256 = sha256_file(projection_receipt_path)
                projection_uri = gcs_join(
                    str(plan["gcs_output_prefix"]),
                    "source-quarantine-projections",
                    str(plan["manifest_sha256"]),
                    f"{int(job['ordinal']):05d}-{job['repo']}",
                    f"{projection_sha256}.projection.json",
                )
                projection_artifact = dict(
                    object_store.publish_if_absent(
                        projection_receipt_path,
                        projection_uri,
                    )
                )
                projection_generation = str(projection_artifact.get("generation", ""))
                projection_verified = scratch / "published-projection.verify"
                projection_metadata = object_store.download(
                    projection_uri,
                    projection_verified,
                    generation=projection_generation,
                )
                if (
                    str(projection_metadata.get("generation")) != projection_generation
                    or projection_verified.stat().st_size
                    != projection_receipt_path.stat().st_size
                    or sha256_file(projection_verified) != projection_sha256
                ):
                    raise ContractError(
                        "published quarantine projection content verification failed"
                    )
                projection_verified.unlink()
                projection_artifact["sha256"] = projection_sha256
            canonical_output = scratch / "canonical.enriched.jsonl"
            candidate = canonicalize_enriched_jsonl(
                raw_output,
                canonical_output,
                project_id=str(job["project_id"]),
            )
            candidate["dedup_applied"] = False
            compressed = scratch / "canonical.enriched.jsonl.zst"
            compression = compress_zstd(canonical_output, compressed)
            artifact_uri = gcs_join(
                str(plan["gcs_output_prefix"]),
                "source-candidates",
                str(plan["manifest_sha256"]),
                f"{int(job['ordinal']):05d}-{job['repo']}",
                f"{compression['sha256']}.jsonl.zst",
            )
            published = dict(object_store.publish_if_absent(compressed, artifact_uri))
            if (
                int(published.get("size_bytes", -1)) != compressed.stat().st_size
                or str(published.get("uri")) != artifact_uri
            ):
                raise ContractError("published candidate object metadata drifted")
            published_generation = str(published.get("generation", ""))
            verified_download = scratch / "published-candidate.verify"
            verified_metadata = object_store.download(
                artifact_uri,
                verified_download,
                generation=published_generation,
            )
            if (
                str(verified_metadata.get("generation")) != published_generation
                or verified_download.stat().st_size != compressed.stat().st_size
                or sha256_file(verified_download) != compression["sha256"]
            ):
                raise ContractError("published candidate content verification failed")
            verified_download.unlink()
            artifact = {
                **published,
                "sha256": compression["sha256"],
                "compression": compression,
            }
            quarantine_sha256 = str(indexer_receipt["quarantine_receipt_sha256"])
            quarantine_uri = gcs_join(
                str(plan["gcs_output_prefix"]),
                "source-quarantine-receipts",
                str(plan["manifest_sha256"]),
                f"{int(job['ordinal']):05d}-{job['repo']}",
                f"{quarantine_sha256}.quarantine.json",
            )
            quarantine_artifact = dict(
                object_store.publish_if_absent(quarantine_receipt, quarantine_uri)
            )
            quarantine_generation = str(quarantine_artifact.get("generation", ""))
            quarantine_verified = scratch / "published-quarantine.verify"
            quarantine_metadata = object_store.download(
                quarantine_uri,
                quarantine_verified,
                generation=quarantine_generation,
            )
            if (
                str(quarantine_metadata.get("generation"))
                != quarantine_generation
                or quarantine_verified.stat().st_size
                != quarantine_receipt.stat().st_size
                or sha256_file(quarantine_verified) != quarantine_sha256
            ):
                raise ContractError(
                    "published quarantine receipt content verification failed"
                )
            quarantine_verified.unlink()
            quarantine_artifact["sha256"] = quarantine_sha256
            receipt: dict[str, object] = {
                "schema": SOURCE_WORKER_RECEIPT_SCHEMA,
                "status": "complete",
                "manifest_sha256": plan["manifest_sha256"],
                "manifest_file_sha256": manifest_file_sha256,
                "assignment": {
                    key: job[key]
                    for key in (
                        "ordinal",
                        "repo",
                        "project_id",
                        "worker",
                        "assignment_sha256",
                    )
                },
                "source_snapshot": source_snapshot,
                "candidate": candidate,
                "artifact": artifact,
                "quarantine_artifact": quarantine_artifact,
                "indexer": indexer_receipt,
                "training_ready": False,
            }
            if projection_artifact is not None:
                receipt["quarantine_projection_artifact"] = projection_artifact
            validate_worker_receipt(receipt, manifest=plan, job=job)
            local_receipt = receipt_root / f"{int(job['ordinal']):05d}-{job['repo']}.json"
            atomic_write_json(local_receipt, receipt)
            receipt_uri = _source_receipt_uri(plan, job, receipt)
            # Publication order is intentional: an uploaded candidate without a
            # receipt is garbage-collectable; a receipt can never point to missing
            # data because it is uploaded last.
            object_store.publish_if_absent(local_receipt, receipt_uri)
            _publish_assignment_completion(
                manifest=plan,
                manifest_file_sha256=manifest_file_sha256,
                job=job,
                source_receipt=receipt,
                source_receipt_path=local_receipt,
                object_store=object_store,
                scratch_root=scratch,
            )
            receipts.append(receipt)
    return tuple(receipts)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--receipt-root", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--indexer", type=Path, default=Path("tools/clang_indexer/index_project.py")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("cppmega/tokenizer/tokenizer.json")
    )
    parser.add_argument(
        "--quarantine-manifest",
        type=Path,
        default=Path("configs/source_quarantine_manifest.json"),
    )
    parser.add_argument("--parse-workers", type=int, default=4)
    parser.add_argument("--memory-limit-gb", type=float, default=14.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--assignment-sha256")
    parser.add_argument(
        "--quarantine-projection-mode",
        choices=sorted(_QUARANTINE_PROJECTION_MODES),
        default=QUARANTINE_PROJECTION_MODE_OFF,
    )
    args = parser.parse_args(argv)
    try:
        manifest, raw_sha256 = load_source_manifest(args.manifest)
        run_source_worker(
            manifest,
            manifest_file_sha256=raw_sha256,
            worker=args.worker,
            scratch_root=args.scratch_root,
            receipt_root=args.receipt_root,
            repo_root=args.repo_root.resolve(),
            python=args.python.resolve(),
            indexer=(args.repo_root / args.indexer).resolve()
            if not args.indexer.is_absolute()
            else args.indexer.resolve(),
            tokenizer=(args.repo_root / args.tokenizer).resolve()
            if not args.tokenizer.is_absolute()
            else args.tokenizer.resolve(),
            quarantine_manifest=(args.repo_root / args.quarantine_manifest).resolve()
            if not args.quarantine_manifest.is_absolute()
            else args.quarantine_manifest.resolve(),
            object_store=GcloudObjectStore(),
            parse_workers=args.parse_workers,
            memory_limit_gb=args.memory_limit_gb,
            max_tokens=args.max_tokens,
            assignment_sha256=args.assignment_sha256,
            quarantine_projection_mode=args.quarantine_projection_mode,
        )
    except TransientTransportError as exc:
        parser.exit(75, f"distributed source worker transient transport failure: {exc}\n")
    except (ContractError, RuntimeError, OSError, ValueError) as exc:
        parser.exit(2, f"distributed source worker failed: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA",
    "CANONICAL_DOCUMENT_ORDER",
    "GcloudObjectStore",
    "PINNED_TREE_PROJECTION_MODE",
    "QUARANTINE_PROJECTION_MODE_OFF",
    "LocalObjectStore",
    "ObjectStore",
    "SOURCE_WORKER_RECEIPT_SCHEMA",
    "TransientTransportError",
    "acquire_git_mirror",
    "assignment_completion_uri",
    "canonicalize_enriched_jsonl",
    "compress_zstd",
    "extract_immutable_tar_zst",
    "run_source_worker",
    "validate_assignment_completion_receipt",
    "validate_worker_receipt",
    "validate_quarantine_receipt_file",
]
