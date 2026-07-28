from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
from pathlib import Path
from functools import lru_cache
import re
import sqlite3
import subprocess
import sys
import threading
from typing import Any, Mapping
import urllib.error
import zipfile
import zlib

import pytest

from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
from scripts import ci_job_log_rescue as job_rescue
from scripts import ci_stream_fetch as ci
from scripts import recover_ci_preserved_archives as preserved_recovery
from scripts.ci_stream_receipts import (
    ReceiptFinalizationError,
    exhaustive_coverage_proof,
)
from scripts.ci_zlib_evidence import (
    ZlibEvidenceError,
    strict_bounded_zlib_decode,
)


_FROZEN_TOKENIZER = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "tokenizer_v2"
    / "tokenizer.json"
)


@lru_cache(maxsize=None)
def _compressed_repetition(
    raw_size: int,
    value: bytes = b"x",
) -> tuple[bytes, str]:
    compressor = zlib.compressobj(9)
    digest = hashlib.sha256()
    parts: list[bytes] = []
    chunk = value * (1024 * 1024)
    remaining = raw_size
    while remaining:
        current = chunk[: min(len(chunk), remaining)]
        digest.update(current)
        parts.append(compressor.compress(current))
        remaining -= len(current)
    parts.append(compressor.flush())
    return b"".join(parts), digest.hexdigest()


def test_strict_bounded_zlib_evidence_contract_matrix() -> None:
    raw_limit = 1024
    raw = b"x" * raw_limit
    digest = hashlib.sha256(raw).hexdigest()
    compressed = zlib.compress(raw, 9)
    kwargs = {
        "expected_raw_size": raw_limit,
        "expected_sha256": digest,
        "max_raw_size": raw_limit,
        "max_compressed_size": len(compressed),
        "where": "matrix evidence",
    }
    assert strict_bounded_zlib_decode(compressed, **kwargs) == raw

    over_limit = raw + b"x"
    with pytest.raises(ZlibEvidenceError, match="semantic bound"):
        strict_bounded_zlib_decode(
            zlib.compress(over_limit, 9),
            expected_raw_size=len(over_limit),
            expected_sha256=hashlib.sha256(over_limit).hexdigest(),
            max_raw_size=raw_limit,
            max_compressed_size=len(zlib.compress(over_limit, 9)),
            where="raw limit+1",
        )
    with pytest.raises(ZlibEvidenceError, match="compressed bytes"):
        strict_bounded_zlib_decode(
            compressed,
            **{**kwargs, "max_compressed_size": len(compressed) - 1},
        )
    with pytest.raises(ZlibEvidenceError, match="trailing"):
        strict_bounded_zlib_decode(
            compressed + zlib.compress(b""),
            **{**kwargs, "max_compressed_size": len(compressed) + 16},
        )
    with pytest.raises(ZlibEvidenceError, match="truncated"):
        strict_bounded_zlib_decode(
            compressed[:-1],
            **kwargs,
        )
    with pytest.raises(ZlibEvidenceError, match="declared bound"):
        strict_bounded_zlib_decode(
            compressed,
            **{**kwargs, "expected_raw_size": raw_limit - 1},
        )
    with pytest.raises(ZlibEvidenceError, match="SHA-256"):
        strict_bounded_zlib_decode(
            compressed,
            **{**kwargs, "expected_sha256": "0" * 64},
        )


def _run_metadata(run_id: int, *, attempt: int = 1) -> dict[str, Any]:
    return {
        "id": run_id,
        "run_attempt": attempt,
        "created_at": f"2026-04-27T16:{run_id % 60:02d}:00Z",
        "updated_at": f"2026-04-27T16:{run_id % 60:02d}:01Z",
        "run_started_at": f"2026-04-27T16:{run_id % 60:02d}:00Z",
        "status": "completed",
        "conclusion": "success",
        "workflow_id": 77,
        "path": ".github/workflows/ci.yml",
        "run_number": run_id,
        "display_title": f"CI run {run_id}",
        "name": "CI",
        "event": "push",
        "head_branch": "main",
        "head_sha": f"{run_id:040x}"[-40:],
        "head_commit": {
            "id": f"{run_id:040x}"[-40:],
            "message": f"commit {run_id}",
            "author": {"name": "builder"},
            "committer": {"name": "builder"},
        },
        "actor": {"login": "builder"},
        "repository": {"full_name": "owner/repo", "id": 1},
        "head_repository": {"full_name": "owner/repo", "id": 1},
    }


def _inventory(path: Path, count: int) -> Path:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE runs(
              repo_key TEXT NOT NULL,
              run_id INTEGER NOT NULL,
              run_attempt INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              metadata_blob BLOB NOT NULL,
              metadata_sha256 TEXT NOT NULL
            )
            """
        )
        for run_id in range(1, count + 1):
            value = _run_metadata(run_id)
            raw = json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
            connection.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?)",
                (
                    "owner/repo",
                    run_id,
                    1,
                    value["created_at"],
                    zlib.compress(raw, 6),
                    hashlib.sha256(raw).hexdigest(),
                ),
            )
    return path


def _replace_inventory_run(path: Path, metadata: Mapping[str, object]) -> str:
    raw = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(raw).hexdigest()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE runs SET
              run_attempt=?,
              created_at=?,
              metadata_blob=?,
              metadata_sha256=?
            WHERE run_id=?
            """,
            (
                metadata["run_attempt"],
                metadata["created_at"],
                zlib.compress(raw, 6),
                digest,
                metadata["id"],
            ),
        )
    return digest


def _tokenizer(path: Path) -> Path:
    del path
    return _FROZEN_TOKENIZER


def _attempt_set_sha256(keys: list[tuple[str, int, int]]) -> str:
    digest = hashlib.sha256()
    for repo, run_id, attempt in keys:
        digest.update(f"{repo}\t{run_id}\t{attempt}\n".encode())
    return digest.hexdigest()


def _exhaustive_binding(
    *,
    run_count: int,
    attempt_keys: list[tuple[str, int, int]],
) -> ci.ExhaustiveInventoryBinding:
    return ci.ExhaustiveInventoryBinding(
        receipt_path=Path("/verified/inventory-receipt.json"),
        receipt_sha256="a" * 64,
        database_sha256="b" * 64,
        db_logical_sha256="c" * 64,
        expected_run_count=run_count,
        expected_attempt_count=len(attempt_keys),
        expected_attempt_set_sha256=_attempt_set_sha256(attempt_keys),
    )


def _coverage_inventory_receipt(
    *,
    run_count: int,
    attempt_keys: list[tuple[str, int, int]],
) -> dict[str, object]:
    return {
        "run_count": run_count,
        "expected_attempt_count": len(attempt_keys),
        "expected_attempt_set_sha256": _attempt_set_sha256(
            attempt_keys
        ),
        "per_repo_ledger": [
            {
                "repo": "owner/repo",
                "canonical": "owner/repo",
                "ordinal": 0,
                "run_count": run_count,
                "expected_attempt_count": len(attempt_keys),
                "expected_attempt_set_sha256": _attempt_set_sha256(
                    attempt_keys
                ),
            }
        ],
    }


def test_exact_tokenizer_matches_training_wrapper_and_rejects_other_artifacts(
    tmp_path: Path,
) -> None:
    payloads = [
        "hello  build\tworld\r\nnext\n",
        '[command]cmake  -S . -B build && ninja -C build\n',
        'printf("literal   whitespace\\n"); // comment  text\n',
    ]
    exact = ci.ExactTokenizer(_FROZEN_TOKENIZER)
    training = load_cppmega_tokenizer(_FROZEN_TOKENIZER)
    expected = training.encode_batch(payloads)

    assert exact.encode_batch(payloads) == expected
    assert exact.contract["schema"] == (
        "cppmega_exact_ci_training_tokenizer_v3"
    )
    assert "cppmega." not in json.dumps(exact.contract, sort_keys=True)
    semantic_hashes = exact.contract["semantic_function_sha256"]
    assert isinstance(semantic_hashes, dict)
    assert set(semantic_hashes) == {
        "tokenizer_init",
        "tokenizer_encode",
        "tokenizer_encode_batch",
        "tokenizer_loader",
        "whitespace_normalizer",
    }
    assert exact.contract["tokenizer_contract_sha256"]
    assert all(
        ci.hash_token_sequence(actual) == ci.hash_token_sequence(wanted)
        for actual, wanted in zip(exact.encode_batch(payloads), expected)
    )

    def semantically_changed_normalizer(value: str) -> tuple[str, list[int]]:
        return value, list(range(len(value)))

    assert ci._semantic_callable_sha256(
        semantically_changed_normalizer
    ) != semantic_hashes["whitespace_normalizer"]

    invalid = tmp_path / "not-cppmega-tokenizer.json"
    invalid.write_text('{"version":"1.0"}')
    with pytest.raises(ci.FetchError, match="frozen cppmega training contract"):
        ci.ExactTokenizer(invalid)


def test_fetch_cli_default_tokenizer_is_tracked_and_cwd_independent() -> None:
    args = ci._build_parser().parse_args(
        [
            "--inventory",
            "inventory.sqlite3",
            "--state",
            "fetch.sqlite3",
            "--content-store",
            "content-store",
            "--progress",
            "progress.json",
            "--receipt",
            "receipt.json",
        ]
    )
    default_path = Path(args.tokenizer)
    assert default_path.is_absolute()
    assert default_path == _FROZEN_TOKENIZER
    exact = ci.ExactTokenizer(default_path)
    assert exact.artifact_sha256 == hashlib.sha256(
        _FROZEN_TOKENIZER.read_bytes()
    ).hexdigest()


def _zip_bytes(
    members: Mapping[str, bytes] | None = None,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in (members or {"0_build.txt": b"hello build world\n"}).items():
            archive.writestr(name, value)
    return output.getvalue()


class _IncompleteArchiveResponse:
    status = 200
    headers: dict[str, str] = {}

    def __init__(self) -> None:
        self._reads = 0

    def read(self, _size: int) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return b"partial archive bytes"
        raise http.client.IncompleteRead(b"truncated", 100)


def test_incomplete_chunked_archive_response_is_retryable(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "archive.zip.partial"
    with pytest.raises(
        ci.ArchiveError,
        match="transport failed before EOF: IncompleteRead",
    ):
        ci._stream_signed_archive_response(
            _IncompleteArchiveResponse(),
            destination,
            max_bytes=1024,
        )
    assert destination.read_bytes() == b"partial archive bytes"


class _SignedArchiveResponse:
    def __init__(
        self,
        *,
        status: int,
        headers: Mapping[str, str],
        reads: list[bytes | BaseException],
    ) -> None:
        self.status = status
        self.headers = dict(headers)
        self._reads = iter(reads)

    def __enter__(self) -> "_SignedArchiveResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        try:
            value = next(self._reads)
        except StopIteration:
            return b""
        if isinstance(value, BaseException):
            raise value
        return value


def test_signed_archive_download_resumes_exact_range_after_incomplete_read(
    tmp_path: Path,
) -> None:
    payload = b"abcdefgh"
    responses = iter(
        [
            _SignedArchiveResponse(
                status=200,
                headers={"Content-Length": "8", "ETag": '"stable"'},
                reads=[
                    b"abc",
                    http.client.IncompleteRead(b"de", 3),
                ],
            ),
            _SignedArchiveResponse(
                status=206,
                headers={
                    "Content-Length": "3",
                    "Content-Range": "bytes 5-7/8",
                    "ETag": '"stable"',
                },
                reads=[b"fgh", b""],
            ),
        ]
    )
    requests: list[Any] = []

    def urlopen(request: Any, *, timeout: float) -> _SignedArchiveResponse:
        assert timeout == 4.0
        requests.append(request)
        return next(responses)

    destination = tmp_path / "archive.zip.partial"
    size, digest = ci._default_archive_downloader(
        "https://signed.example/archive.zip",
        destination,
        timeout=4.0,
        max_bytes=1024,
        urlopen=urlopen,
        max_transfer_attempts=2,
    )

    assert destination.read_bytes() == payload
    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert requests[0].get_header("Range") is None
    assert requests[1].get_header("Range") == "bytes=5-"
    assert requests[1].get_header("If-range") == '"stable"'


def test_signed_archive_resume_rejects_inconsistent_content_range(
    tmp_path: Path,
) -> None:
    responses = iter(
        [
            _SignedArchiveResponse(
                status=200,
                headers={"Content-Length": "8", "ETag": '"stable"'},
                reads=[b"abc", http.client.IncompleteRead(b"", 5)],
            ),
            _SignedArchiveResponse(
                status=206,
                headers={
                    "Content-Length": "5",
                    "Content-Range": "bytes 2-6/8",
                    "ETag": '"stable"',
                },
                reads=[b"defgh"],
            ),
        ]
    )

    def urlopen(request: Any, *, timeout: float) -> _SignedArchiveResponse:
        del request, timeout
        return next(responses)

    with pytest.raises(
        ci.MalformedResponseError,
        match="resumed byte range is inconsistent",
    ):
        ci._default_archive_downloader(
            "https://signed.example/archive.zip",
            tmp_path / "archive.zip.partial",
            timeout=4.0,
            max_bytes=1024,
            urlopen=urlopen,
            max_transfer_attempts=2,
        )


@pytest.mark.parametrize("etag", [None, 'W/"weak"'])
def test_signed_archive_download_without_strong_validator_restarts_full(
    tmp_path: Path,
    etag: str | None,
) -> None:
    payload = b"abcdefgh"
    first_headers = {"Content-Length": "8"}
    if etag is not None:
        first_headers["ETag"] = etag
    responses = iter(
        [
            _SignedArchiveResponse(
                status=200,
                headers=first_headers,
                reads=[b"abc", http.client.IncompleteRead(b"de", 3)],
            ),
            _SignedArchiveResponse(
                status=200,
                headers={"Content-Length": "8"},
                reads=[payload, b""],
            ),
        ]
    )
    requests: list[Any] = []

    def urlopen(request: Any, *, timeout: float) -> _SignedArchiveResponse:
        del timeout
        requests.append(request)
        return next(responses)

    destination = tmp_path / "archive.zip.partial"
    size, digest = ci._default_archive_downloader(
        "https://signed.example/archive.zip",
        destination,
        timeout=4.0,
        max_bytes=1024,
        urlopen=urlopen,
        max_transfer_attempts=2,
    )

    assert destination.read_bytes() == payload
    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert requests[1].get_header("Range") is None
    assert requests[1].get_header("If-range") is None


def test_signed_archive_range_mismatch_restarts_from_complete_200(
    tmp_path: Path,
) -> None:
    replacement = b"replacement"
    responses = iter(
        [
            _SignedArchiveResponse(
                status=200,
                headers={"Content-Length": "8", "ETag": '"old"'},
                reads=[b"abc", http.client.IncompleteRead(b"", 5)],
            ),
            _SignedArchiveResponse(
                status=200,
                headers={
                    "Content-Length": str(len(replacement)),
                    "ETag": '"new"',
                },
                reads=[replacement, b""],
            ),
        ]
    )
    requests: list[Any] = []

    def urlopen(request: Any, *, timeout: float) -> _SignedArchiveResponse:
        del timeout
        requests.append(request)
        return next(responses)

    destination = tmp_path / "archive.zip.partial"
    size, digest = ci._default_archive_downloader(
        "https://signed.example/archive.zip",
        destination,
        timeout=4.0,
        max_bytes=1024,
        urlopen=urlopen,
        max_transfer_attempts=2,
    )

    assert requests[1].get_header("Range") == "bytes=3-"
    assert requests[1].get_header("If-range") == '"old"'
    assert destination.read_bytes() == replacement
    assert size == len(replacement)
    assert digest == hashlib.sha256(replacement).hexdigest()


def test_signed_archive_resume_requires_same_strong_validator(
    tmp_path: Path,
) -> None:
    responses = iter(
        [
            _SignedArchiveResponse(
                status=200,
                headers={"Content-Length": "8", "ETag": '"stable"'},
                reads=[b"abc", http.client.IncompleteRead(b"", 5)],
            ),
            _SignedArchiveResponse(
                status=206,
                headers={
                    "Content-Length": "5",
                    "Content-Range": "bytes 3-7/8",
                },
                reads=[b"defgh", b""],
            ),
        ]
    )

    def urlopen(request: Any, *, timeout: float) -> _SignedArchiveResponse:
        del request, timeout
        return next(responses)

    with pytest.raises(
        ci.MalformedResponseError,
        match="resumed byte range is inconsistent",
    ):
        ci._default_archive_downloader(
            "https://signed.example/archive.zip",
            tmp_path / "archive.zip.partial",
            timeout=4.0,
            max_bytes=1024,
            urlopen=urlopen,
            max_transfer_attempts=2,
        )


def test_signed_archive_download_accepts_multiple_valid_ranges(
    tmp_path: Path,
) -> None:
    payload = b"abcdefghij"
    responses = iter(
        [
            _SignedArchiveResponse(
                status=200,
                headers={"Content-Length": "10", "ETag": '"stable"'},
                reads=[b"abc", http.client.IncompleteRead(b"", 7)],
            ),
            _SignedArchiveResponse(
                status=206,
                headers={
                    "Content-Length": "3",
                    "Content-Range": "bytes 3-5/10",
                    "ETag": '"stable"',
                },
                reads=[b"def", b""],
            ),
            _SignedArchiveResponse(
                status=206,
                headers={
                    "Content-Length": "4",
                    "Content-Range": "bytes 6-9/10",
                    "ETag": '"stable"',
                },
                reads=[b"ghij", b""],
            ),
        ]
    )
    requests: list[Any] = []

    def urlopen(request: Any, *, timeout: float) -> _SignedArchiveResponse:
        del timeout
        requests.append(request)
        return next(responses)

    destination = tmp_path / "archive.zip.partial"
    size, digest = ci._default_archive_downloader(
        "https://signed.example/archive.zip",
        destination,
        timeout=4.0,
        max_bytes=1024,
        urlopen=urlopen,
        max_transfer_attempts=3,
    )

    assert [request.get_header("Range") for request in requests] == [
        None,
        "bytes=3-",
        "bytes=6-",
    ]
    assert destination.read_bytes() == payload
    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()


def test_signed_archive_resume_rejects_body_beyond_declared_range(
    tmp_path: Path,
) -> None:
    responses = iter(
        [
            _SignedArchiveResponse(
                status=200,
                headers={"Content-Length": "8", "ETag": '"stable"'},
                reads=[b"abcde", http.client.IncompleteRead(b"", 3)],
            ),
            _SignedArchiveResponse(
                status=206,
                headers={
                    "Content-Range": "bytes 5-6/8",
                    "ETag": '"stable"',
                },
                reads=[b"XYZ", b""],
            ),
        ]
    )

    def urlopen(request: Any, *, timeout: float) -> _SignedArchiveResponse:
        del request, timeout
        return next(responses)

    destination = tmp_path / "archive.zip.partial"
    with pytest.raises(
        ci.MalformedResponseError,
        match="exceeded its declared byte range",
    ):
        ci._default_archive_downloader(
            "https://signed.example/archive.zip",
            destination,
            timeout=4.0,
            max_bytes=1024,
            urlopen=urlopen,
            max_transfer_attempts=2,
        )

    assert destination.read_bytes() == b"abcde"


def test_signed_archive_download_keeps_stable_destination_fd(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "archive.zip.partial"
    victim = tmp_path / "victim"
    victim.write_bytes(b"do not overwrite")

    def urlopen(request: Any, *, timeout: float) -> _SignedArchiveResponse:
        del request, timeout
        destination.unlink()
        destination.symlink_to(victim)
        raise urllib.error.URLError("transport failed")

    with pytest.raises(
        ci.ArchiveError,
        match="destination identity changed",
    ):
        ci._default_archive_downloader(
            "https://signed.example/archive.zip",
            destination,
            timeout=4.0,
            max_bytes=1024,
            urlopen=urlopen,
            max_transfer_attempts=2,
        )

    assert victim.read_bytes() == b"do not overwrite"


def test_fetch_state_script_binding_upgrade_is_explicit_and_audited(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = ci.ExactTokenizer(_tokenizer(tmp_path / "tokenizer.json"))
    state_path = tmp_path / "state.sqlite"
    store_path = tmp_path / "store"
    state = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=store_path,
        tokenizer=tokenizer,
        resume=False,
    )
    state.close()

    previous_sha256 = "a" * 64
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """
            UPDATE settings SET value=?
            WHERE key='fetcher_script_sha256'
            """,
            (previous_sha256,),
        )

    with pytest.raises(ci.BindingError, match="fetcher_script_sha256"):
        ci.FetchState(
            state_path,
            inventory_path=inventory,
            content_store_path=store_path,
            tokenizer=tokenizer,
            resume=True,
        )
    with pytest.raises(ValueError, match="upgrade reason"):
        ci.FetchState(
            state_path,
            inventory_path=inventory,
            content_store_path=store_path,
            tokenizer=tokenizer,
            resume=True,
            allow_fetcher_script_upgrade_from_sha256="b" * 64,
        )
    with pytest.raises(ci.BindingError, match="fetcher_script_sha256"):
        ci.FetchState(
            state_path,
            inventory_path=inventory,
            content_store_path=store_path,
            tokenizer=tokenizer,
            resume=True,
            allow_fetcher_script_upgrade_from_sha256="b" * 64,
            fetcher_script_upgrade_reason="test wrong source binding",
        )

    reason = "skip terminal jobs requests and replay committed members"
    upgraded = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=store_path,
        tokenizer=tokenizer,
        resume=True,
        allow_fetcher_script_upgrade_from_sha256=previous_sha256,
        fetcher_script_upgrade_reason=reason,
    )
    try:
        assert upgraded._connection.execute(
            """
            SELECT value FROM settings
            WHERE key='fetcher_script_sha256'
            """
        ).fetchone()[0] == ci._script_sha256()
        assert upgraded.summary()["binding_upgrades"] == [
            {
                "binding_key": "fetcher_script_sha256",
                "from_sha256": previous_sha256,
                "to_sha256": ci._script_sha256(),
                "reason": reason,
                "upgraded_at": upgraded._connection.execute(
                    "SELECT upgraded_at FROM binding_upgrades"
                ).fetchone()[0],
            }
        ]
    finally:
        upgraded.close()


def test_operator_producer_lineage_is_explicit_ordered_and_fail_closed() -> None:
    origin = {
        "script_sha256": "a" * 64,
        "semantic_contract_sha256": (
            ci.JOB_RESCUE_LEGACY_SEMANTIC_CONTRACT_SHA256
        ),
    }
    current = {
        "script_sha256": "b" * 64,
        "semantic_contract_sha256": (
            ci.JOB_RESCUE_SEMANTIC_CONTRACT_SHA256
        ),
    }
    base = ci._producer_lineage(origin)
    with pytest.raises(ValueError, match="reason"):
        ci._authorize_producer_lineage_upgrade(
            base,
            current_binding=current,
            allow_from_sha256=origin["script_sha256"],
            reason=None,
            authorized_at="2026-07-28T10:00:00Z",
        )

    lineage = ci._authorize_producer_lineage_upgrade(
        base,
        current_binding=current,
        allow_from_sha256=origin["script_sha256"],
        reason="explicit operator-authorized semantic migration",
        authorized_at="2026-07-28T10:00:00Z",
    )
    assert (
        ci._validate_producer_lineage(
            lineage,
            artifact_binding=origin,
            current_binding=current,
        )
        == origin
    )
    assert lineage["origin"] == origin
    assert lineage["current"] == current
    assert lineage["upgrades"] == [
        {
            "from": origin,
            "to": current,
            "reason": "explicit operator-authorized semantic migration",
            "authorized_at": "2026-07-28T10:00:00Z",
        }
    ]

    forged = json.loads(json.dumps(lineage))
    forged["upgrades"][0]["from"]["script_sha256"] = "c" * 64
    with pytest.raises(ci.BindingError, match="producer lineage"):
        ci._validate_producer_lineage(
            forged,
            artifact_binding=origin,
            current_binding=current,
        )


def test_fetch_state_parser_upgrade_migrates_legacy_ledger_and_replays_exactly(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = ci.ExactTokenizer(_tokenizer(tmp_path / "tokenizer.json"))
    state_path = tmp_path / "state.sqlite"
    store_path = tmp_path / "store"
    state = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=store_path,
        tokenizer=tokenizer,
        resume=False,
    )
    state.close()

    previous_sha256 = "b" * 64
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """
            UPDATE settings SET value=?
            WHERE key='parser_script_sha256'
            """,
            (previous_sha256,),
        )
        connection.execute("DROP TABLE binding_upgrades")
        connection.executescript(
            """
            CREATE TABLE binding_upgrades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                binding_key TEXT NOT NULL CHECK (
                  binding_key = 'fetcher_script_sha256'
                ),
                from_sha256 TEXT NOT NULL CHECK (length(from_sha256) = 64),
                to_sha256 TEXT NOT NULL CHECK (length(to_sha256) = 64),
                reason TEXT NOT NULL,
                upgraded_at TEXT NOT NULL,
                UNIQUE(binding_key,from_sha256,to_sha256)
            );
            """
        )
        connection.commit()

    with pytest.raises(ci.BindingError, match="parser_script_sha256"):
        ci.FetchState(
            state_path,
            inventory_path=inventory,
            content_store_path=store_path,
            tokenizer=tokenizer,
            resume=True,
        )
    with sqlite3.connect(state_path) as connection:
        legacy_sql = str(
            connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type='table' AND name='binding_upgrades'
                """
            ).fetchone()[0]
        )
    assert "'parser_script_sha256'" not in legacy_sql

    with pytest.raises(ValueError, match="parser script binding upgrade reason"):
        ci.FetchState(
            state_path,
            inventory_path=inventory,
            content_store_path=store_path,
            tokenizer=tokenizer,
            resume=True,
            allow_parser_script_upgrade_from_sha256=previous_sha256,
        )
    with pytest.raises(ci.BindingError, match="parser_script_sha256"):
        ci.FetchState(
            state_path,
            inventory_path=inventory,
            content_store_path=store_path,
            tokenizer=tokenizer,
            resume=True,
            allow_parser_script_upgrade_from_sha256="c" * 64,
            parser_script_upgrade_reason="wrong parser source",
        )

    reason = "replace quadratic scans with output-equivalent linear bucketing"
    upgraded = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=store_path,
        tokenizer=tokenizer,
        resume=True,
        allow_parser_script_upgrade_from_sha256=previous_sha256,
        parser_script_upgrade_reason=reason,
    )
    try:
        rows = upgraded.summary()["binding_upgrades"]
        assert rows == [
            {
                "binding_key": "parser_script_sha256",
                "from_sha256": previous_sha256,
                "to_sha256": ci._parser_sha256(),
                "reason": reason,
                "upgraded_at": rows[0]["upgraded_at"],
            }
        ]
        widened_sql = str(
            upgraded._connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type='table' AND name='binding_upgrades'
                """
            ).fetchone()[0]
        )
        assert "'parser_script_sha256'" in widened_sql
        assert "'content_store_script_sha256'" in widened_sql
    finally:
        upgraded.close()

    replayed = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=store_path,
        tokenizer=tokenizer,
        resume=True,
        allow_parser_script_upgrade_from_sha256=previous_sha256,
        parser_script_upgrade_reason=reason,
    )
    try:
        assert len(replayed.summary()["binding_upgrades"]) == 1
    finally:
        replayed.close()
    with pytest.raises(
        ci.BindingError,
        match="does not replay the latest audited transition",
    ):
        ci.FetchState(
            state_path,
            inventory_path=inventory,
            content_store_path=store_path,
            tokenizer=tokenizer,
            resume=True,
            allow_parser_script_upgrade_from_sha256=previous_sha256,
            parser_script_upgrade_reason="different parser migration reason",
        )


def test_fetch_state_content_store_upgrade_migrates_two_key_ledger(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = ci.ExactTokenizer(_tokenizer(tmp_path / "tokenizer.json"))
    state_path = tmp_path / "state.sqlite"
    store_path = tmp_path / "store"
    state = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=store_path,
        tokenizer=tokenizer,
        resume=False,
    )
    state.close()

    previous_sha256 = "c" * 64
    existing_from = "d" * 64
    existing_to = "e" * 64
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """
            UPDATE settings SET value=?
            WHERE key='content_store_script_sha256'
            """,
            (previous_sha256,),
        )
        connection.execute("DROP TABLE binding_upgrades")
        connection.executescript(
            """
            CREATE TABLE binding_upgrades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                binding_key TEXT NOT NULL CHECK (
                  binding_key IN (
                    'fetcher_script_sha256',
                    'parser_script_sha256'
                  )
                ),
                from_sha256 TEXT NOT NULL CHECK (length(from_sha256) = 64),
                to_sha256 TEXT NOT NULL CHECK (length(to_sha256) = 64),
                reason TEXT NOT NULL,
                upgraded_at TEXT NOT NULL,
                UNIQUE(binding_key,from_sha256,to_sha256)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO binding_upgrades(
              binding_key,from_sha256,to_sha256,reason,upgraded_at
            ) VALUES ('parser_script_sha256',?,?,?,?)
            """,
            (
                existing_from,
                existing_to,
                "existing parser migration",
                "2026-07-27T00:00:00Z",
            ),
        )
        connection.commit()

    with pytest.raises(ci.BindingError, match="content_store_script_sha256"):
        ci.FetchState(
            state_path,
            inventory_path=inventory,
            content_store_path=store_path,
            tokenizer=tokenizer,
            resume=True,
        )
    with pytest.raises(
        ValueError,
        match="content store script binding upgrade reason",
    ):
        ci.FetchState(
            state_path,
            inventory_path=inventory,
            content_store_path=store_path,
            tokenizer=tokenizer,
            resume=True,
            allow_content_store_script_upgrade_from_sha256=previous_sha256,
        )

    reason = "replace quadratic orphan scan with equivalent set difference"
    upgraded = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=store_path,
        tokenizer=tokenizer,
        resume=True,
        allow_content_store_script_upgrade_from_sha256=previous_sha256,
        content_store_script_upgrade_reason=reason,
    )
    try:
        rows = upgraded.summary()["binding_upgrades"]
        assert rows == [
            {
                "binding_key": "parser_script_sha256",
                "from_sha256": existing_from,
                "to_sha256": existing_to,
                "reason": "existing parser migration",
                "upgraded_at": "2026-07-27T00:00:00Z",
            },
            {
                "binding_key": "content_store_script_sha256",
                "from_sha256": previous_sha256,
                "to_sha256": ci._content_store_sha256(),
                "reason": reason,
                "upgraded_at": rows[1]["upgraded_at"],
            },
        ]
        widened_sql = str(
            upgraded._connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type='table' AND name='binding_upgrades'
                """
            ).fetchone()[0]
        )
        assert "'content_store_script_sha256'" in widened_sql
    finally:
        upgraded.close()

    replayed = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=store_path,
        tokenizer=tokenizer,
        resume=True,
        allow_content_store_script_upgrade_from_sha256=previous_sha256,
        content_store_script_upgrade_reason=reason,
    )
    try:
        assert len(replayed.summary()["binding_upgrades"]) == 2
    finally:
        replayed.close()


def test_fetcher_binds_state_to_immutable_content_store_creator(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer_path = _tokenizer(tmp_path / "tokenizer.json")
    tokenizer = ci.ExactTokenizer(tokenizer_path)
    state_path = tmp_path / "state.sqlite"
    store_path = tmp_path / "store"

    store = ci.CIContentStore(store_path)
    store_db = store.db_path
    store.close()
    creator_sha256 = "c" * 64
    with sqlite3.connect(store_db) as connection:
        connection.execute(
            """
            UPDATE settings SET value=?
            WHERE key='creator_script_sha256'
            """,
            (creator_sha256,),
        )

    # Reproduce the bad durable state produced by the old resume path: it
    # recorded the current verifier hash instead of the immutable CAS creator.
    state = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=store_path,
        tokenizer=tokenizer,
        resume=False,
    )
    state.close()
    runtime_sha256 = ci._content_store_sha256()
    assert runtime_sha256 != creator_sha256

    reason = "restore immutable content-store creator binding"
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer_path,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        resume=True,
        allow_content_store_script_upgrade_from_sha256=runtime_sha256,
        content_store_script_upgrade_reason=reason,
    )
    try:
        assert fetcher.store.script_sha256 == creator_sha256
        assert fetcher.state._connection.execute(
            """
            SELECT value FROM settings
            WHERE key='content_store_script_sha256'
            """
        ).fetchone()[0] == creator_sha256
        assert fetcher.state.summary()["binding_upgrades"][-1] == {
            "binding_key": "content_store_script_sha256",
            "from_sha256": runtime_sha256,
            "to_sha256": creator_sha256,
            "reason": reason,
            "upgraded_at": fetcher.state._connection.execute(
                """
                SELECT upgraded_at FROM binding_upgrades
                ORDER BY id DESC LIMIT 1
                """
            ).fetchone()[0],
        }
    finally:
        fetcher.close()


def _projection_fixture_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_fetch_state_v3_projection_fixture(
    path: Path,
    *,
    transitional: bool,
    content_store_path: Path,
) -> bytes:
    from scripts.ci_fetch_state_migration import LEGACY_FETCH_STATE_SCHEMA

    schema = ci._STATE_SCHEMA
    if not transitional:
        schema = schema.replace("    archive_zlib BLOB,\n", "")
    empty_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(empty_zip_buffer, "w"):
        pass
    empty_zip = empty_zip_buffer.getvalue()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(schema)
        settings = {
            "schema": LEGACY_FETCH_STATE_SCHEMA,
            "inventory_path": str(path.parent / "inventory.sqlite"),
            "content_store_path": str(content_store_path.resolve()),
            "tokenizer_contract": "{}",
            "tokenizer_fingerprint": "f" * 64,
            "fetcher_script_sha256": "1" * 64,
            "parser_script_sha256": "2" * 64,
            "content_store_script_sha256": "3" * 64,
            "chunk_semantics": (
                "parser-dedup-text-cppmega-training-tokenizer-"
                "payload-only-no-framing-v2"
            ),
            "created_at": "2026-07-28T10:00:00Z",
        }
        connection.executemany(
            "INSERT INTO settings(key,value) VALUES (?,?)",
            sorted(settings.items()),
        )
        columns = tuple(
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(attempts)")
        )
        statuses = (
            "pending",
            "processing",
            "retry",
            "failed",
            "terminal_404",
            "terminal_410",
            "done",
            "empty",
            "empty",
        )
        for run_id, status in enumerate(statuses, 1):
            metadata_raw = _projection_fixture_json(
                {
                    "created_at": "2026-07-28T09:00:00Z",
                    "id": run_id,
                    "run_attempt": 1,
                }
            )
            metadata_sha256 = hashlib.sha256(metadata_raw).hexdigest()
            jobs_raw = _projection_fixture_json(
                [{"id": run_id, "name": f"job-{run_id}"}]
            )
            completed = status in {"done", "empty"}
            member_count = int(run_id in {7, 8})
            chunk_count = int(run_id == 7)
            occurrence_tokens = 5 if run_id == 7 else 0
            values: dict[str, object] = {
                "repo": "owner/repo",
                "run_id": run_id,
                "attempt": 1,
                "created_at": "2026-07-28T09:00:00Z",
                "run_metadata_sha256": metadata_sha256,
                "run_metadata_raw_size": len(metadata_raw),
                "run_metadata_zlib": sqlite3.Binary(
                    zlib.compress(metadata_raw, 6)
                ),
                "run_metadata_source": "inventory-run-list",
                "run_metadata_source_attempt": 1,
                "run_metadata_exact": 1,
                "inventory_seed_attempt": 1,
                "inventory_seed_metadata_sha256": metadata_sha256,
                "status": status,
                "tries": 4,
                "archive_source": (
                    "github-signed-url" if completed else None
                ),
                "archive_sha256": (
                    hashlib.sha256(empty_zip).hexdigest()
                    if completed
                    else None
                ),
                "archive_size": len(empty_zip) if completed else None,
                "jobs_sha256": (
                    hashlib.sha256(jobs_raw).hexdigest()
                    if completed
                    else None
                ),
                "jobs_raw_size": len(jobs_raw) if completed else None,
                "jobs_zlib": (
                    sqlite3.Binary(zlib.compress(jobs_raw, 6))
                    if completed
                    else None
                ),
                "member_count": member_count,
                "chunk_count": chunk_count,
                "occurrence_tokens": occurrence_tokens,
                "terminal_http_status": (
                    404 if status == "terminal_404"
                    else 410 if status == "terminal_410"
                    else None
                ),
                "terminal_body_sha256": (
                    "4" * 64
                    if status in {"terminal_404", "terminal_410"}
                    else None
                ),
                "error_class": (
                    "TerminalHTTP"
                    if status in {"terminal_404", "terminal_410"}
                    else None
                ),
                "error_message": (
                    f"terminal {status}"
                    if status in {"terminal_404", "terminal_410"}
                    else None
                ),
                "discovered_at": "2026-07-28T10:00:00Z",
                "updated_at": "2026-07-28T10:01:00Z",
            }
            if transitional:
                values["archive_zlib"] = (
                    sqlite3.Binary(zlib.compress(empty_zip, 6))
                    if run_id == 9
                    else None
                )
            connection.execute(
                f"""
                INSERT INTO attempts({",".join(columns)})
                VALUES ({",".join("?" for _column in columns)})
                """,
                tuple(values[column] for column in columns),
            )
            if member_count:
                sidecar_raw = _projection_fixture_json(
                    {"schema": "projection-fixture-sidecar-v1"}
                )
                connection.execute(
                    """
                    INSERT INTO members(
                      repo,run_id,attempt,archive_member,job_key,
                      raw_sha256,raw_size,canonical_sha256,dedup_sha256,
                      sidecar_sha256,sidecar_raw_size,sidecar_zlib,
                      chunk_count,occurrence_tokens
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        "owner/repo",
                        run_id,
                        1,
                        f"{run_id}_job.txt",
                        f"{run_id}:{run_id}_job.txt",
                        "5" * 64,
                        1,
                        "6" * 64,
                        "7" * 64,
                        hashlib.sha256(sidecar_raw).hexdigest(),
                        len(sidecar_raw),
                        sqlite3.Binary(zlib.compress(sidecar_raw, 6)),
                        chunk_count,
                        occurrence_tokens,
                    ),
                )
        connection.execute(
            """
            INSERT INTO request_ledger(
              id,requested_at,repo,run_id,attempt,endpoint,page_no,
              request_attempt,http_status,outcome,latency_ms,
              error_class,error_message
            ) VALUES (7,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "2026-07-28T10:00:01Z",
                "owner/repo",
                1,
                1,
                "/fixture",
                1,
                1,
                200,
                "success",
                1,
                None,
                None,
            ),
        )
        connection.execute(
            """
            INSERT INTO request_ledger(
              id,requested_at,repo,run_id,attempt,endpoint,page_no,
              request_attempt,http_status,outcome,latency_ms
            ) VALUES (
              11,'2026-07-28T10:00:02Z','owner/repo',1,1,
              '/deleted',1,1,200,'success',1
            )
            """
        )
        connection.execute("DELETE FROM request_ledger WHERE id=11")
        connection.execute(
            """
            INSERT INTO binding_upgrades(
              id,binding_key,from_sha256,to_sha256,reason,upgraded_at
            ) VALUES (13,'fetcher_script_sha256',?,?,?,?)
            """,
            (
                "8" * 64,
                "9" * 64,
                "projection fixture",
                "2026-07-28T10:00:03Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO binding_upgrades(
              id,binding_key,from_sha256,to_sha256,reason,upgraded_at
            ) VALUES (17,'parser_script_sha256',?,?,?,?)
            """,
            (
                "a" * 64,
                "b" * 64,
                "deleted projection fixture",
                "2026-07-28T10:00:04Z",
            ),
        )
        connection.execute("DELETE FROM binding_upgrades WHERE id=17")
        connection.commit()
    finally:
        connection.close()
    return empty_zip


def test_fetch_state_contract_classifier_dispatches_only_exact_pairs() -> None:
    from scripts.ci_fetch_state_migration import (
        CURRENT_FETCH_STATE_SCHEMA,
        CURRENT_V4_LAYOUT,
        CURRENT_V4_SQLITE_SCHEMA_SHA256,
        LEGACY_FETCH_STATE_SCHEMA,
        LEGACY_V3_LAYOUT,
        LEGACY_V3_SQLITE_SCHEMA_SHA256,
        TRANSITIONAL_V3_LAYOUT,
        FetchStateMigrationError,
        classify_fetch_state_contract,
    )

    assert classify_fetch_state_contract(
        settings_schema=LEGACY_FETCH_STATE_SCHEMA,
        sqlite_schema_sha256=LEGACY_V3_SQLITE_SCHEMA_SHA256,
    ) == LEGACY_V3_LAYOUT
    assert classify_fetch_state_contract(
        settings_schema=LEGACY_FETCH_STATE_SCHEMA,
        sqlite_schema_sha256=CURRENT_V4_SQLITE_SCHEMA_SHA256,
    ) == TRANSITIONAL_V3_LAYOUT
    assert classify_fetch_state_contract(
        settings_schema=CURRENT_FETCH_STATE_SCHEMA,
        sqlite_schema_sha256=CURRENT_V4_SQLITE_SCHEMA_SHA256,
    ) == CURRENT_V4_LAYOUT
    with pytest.raises(
        FetchStateMigrationError,
        match="settings/schema pair is unsupported",
    ):
        classify_fetch_state_contract(
            settings_schema=CURRENT_FETCH_STATE_SCHEMA,
            sqlite_schema_sha256=LEGACY_V3_SQLITE_SCHEMA_SHA256,
        )


@pytest.mark.parametrize(
    ("transitional", "expected_zero_empty", "expected_requeues"),
    [
        (False, "retry", 1),
        (True, "empty", 0),
    ],
)
def test_fetch_state_v3_projection_status_matrix_is_immutable_and_exact(
    tmp_path: Path,
    transitional: bool,
    expected_zero_empty: str,
    expected_requeues: int,
) -> None:
    from scripts.ci_content_store import _sqlite_schema_sha256
    from scripts.ci_fetch_state_migration import (
        CURRENT_V4_SQLITE_SCHEMA_SHA256,
        LEGACY_REQUEUE_REASON,
        _row_sha256,
        project_fetch_state_v3_to_v4,
    )

    source = tmp_path / "source-v3.sqlite"
    destination = tmp_path / "destination-v4.sqlite"
    store_path = tmp_path / "store"
    empty_zip = _write_fetch_state_v3_projection_fixture(
        source,
        transitional=transitional,
        content_store_path=store_path,
    )
    source_before = (
        source.stat().st_size,
        source.stat().st_mtime_ns,
        source.stat().st_ino,
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    result = project_fetch_state_v3_to_v4(source, destination)

    source_after = (
        source.stat().st_size,
        source.stat().st_mtime_ns,
        source.stat().st_ino,
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    assert source_after == source_before
    assert not Path(f"{source}-wal").exists()
    assert not Path(f"{source}-journal").exists()
    assert result.attempts == 9
    assert result.requeued_attempts == expected_requeues
    assert len(result.ledger_records) == expected_requeues
    with sqlite3.connect(source) as legacy, sqlite3.connect(
        destination
    ) as projected:
        legacy.row_factory = sqlite3.Row
        projected.row_factory = sqlite3.Row
        assert _sqlite_schema_sha256(projected) == (
            CURRENT_V4_SQLITE_SCHEMA_SHA256
        )
        assert projected.execute(
            "SELECT value FROM settings WHERE key='schema'"
        ).fetchone()[0] == ci.SCHEMA_VERSION
        statuses = {
            int(row["run_id"]): str(row["status"])
            for row in projected.execute(
                "SELECT run_id,status FROM attempts ORDER BY run_id"
            )
        }
        assert statuses == {
            1: "pending",
            2: "processing",
            3: "retry",
            4: "failed",
            5: "terminal_404",
            6: "terminal_410",
            7: "done",
            8: "empty",
            9: expected_zero_empty,
        }
        source_row = dict(
            legacy.execute(
                "SELECT * FROM attempts WHERE run_id=9"
            ).fetchone()
        )
        projected_row = dict(
            projected.execute(
                "SELECT * FROM attempts WHERE run_id=9"
            ).fetchone()
        )
        for field, value in source_row.items():
            if field in {
                "status",
                "tries",
                "terminal_http_status",
                "terminal_body_sha256",
                "error_class",
                "error_message",
            } and not transitional:
                continue
            assert projected_row[field] == value
        if transitional:
            assert zlib.decompress(projected_row["archive_zlib"]) == empty_zip
        else:
            assert projected_row["archive_zlib"] is None
            assert projected_row["tries"] == 0
        for table, order_by in (
            ("members", "repo,run_id,attempt,archive_member"),
            ("request_ledger", "id"),
            ("binding_upgrades", "id"),
        ):
            assert legacy.execute(
                f"SELECT * FROM {table} ORDER BY {order_by}"
            ).fetchall() == projected.execute(
                f"SELECT * FROM {table} ORDER BY {order_by}"
            ).fetchall()
        sequences = dict(
            projected.execute(
                """
                SELECT name,seq FROM sqlite_sequence
                WHERE name IN ('request_ledger','binding_upgrades')
                """
            ).fetchall()
        )
        assert sequences == {
            "request_ledger": 11,
            "binding_upgrades": 17,
        }
    if not transitional:
        ledger = result.ledger_records[0]
        assert ledger["key"] == {
            "repo": "owner/repo",
            "run_id": 9,
            "attempt": 1,
        }
        assert ledger["archive_identity"] == {
            "source": "github-signed-url",
            "sha256": hashlib.sha256(empty_zip).hexdigest(),
            "bytes": len(empty_zip),
        }
        assert ledger["reason"] == LEGACY_REQUEUE_REASON
        assert ledger["action"] == "requeue"
        assert ledger["legacy_row_sha256"] == _row_sha256(source_row)
        assert ledger["projected_row_sha256"] == _row_sha256(
            projected_row
        )


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("wrong-pair", "settings/schema pair is unsupported"),
        ("corrupt-archive-zlib", "exact valid empty ZIP"),
        ("corrupt-jobs-zlib", "exact bounded zlib evidence"),
        ("inconsistent-counts", "member accounting is inconsistent"),
        ("forged-terminal", "inconsistent status"),
        ("unexpected-sequence", "unexpected table"),
        ("sqlite-stat1", "unexpected internal SQLite schema artifacts"),
        ("nonempty-wal", "not frozen"),
        ("nonempty-shm", "not frozen"),
    ],
)
def test_fetch_state_v3_projection_tampering_fails_without_mutation(
    tmp_path: Path,
    tamper: str,
    error: str,
) -> None:
    from scripts.ci_fetch_state_migration import (
        FetchStateMigrationError,
        project_fetch_state_v3_to_v4,
    )

    source = tmp_path / "tampered-v3.sqlite"
    destination = tmp_path / "must-not-exist.sqlite"
    _write_fetch_state_v3_projection_fixture(
        source,
        transitional=tamper == "corrupt-archive-zlib",
        content_store_path=tmp_path / "store",
    )
    if tamper == "wrong-pair":
        with sqlite3.connect(source) as connection:
            connection.execute(
                "UPDATE settings SET value=? WHERE key='schema'",
                (ci.SCHEMA_VERSION,),
            )
    elif tamper == "corrupt-archive-zlib":
        with sqlite3.connect(source) as connection:
            connection.execute(
                """
                UPDATE attempts SET archive_zlib=?
                WHERE status='empty' AND member_count=0
                """,
                (sqlite3.Binary(b"not-zlib"),),
            )
    elif tamper == "corrupt-jobs-zlib":
        with sqlite3.connect(source) as connection:
            connection.execute(
                "UPDATE attempts SET jobs_zlib=? WHERE run_id=9",
                (sqlite3.Binary(b"not-zlib"),),
            )
    elif tamper == "inconsistent-counts":
        with sqlite3.connect(source) as connection:
            connection.execute(
                "UPDATE attempts SET member_count=0 WHERE run_id=8"
            )
    elif tamper == "forged-terminal":
        with sqlite3.connect(source) as connection:
            connection.execute(
                """
                UPDATE attempts
                SET terminal_http_status=410,
                    terminal_body_sha256=NULL,
                    error_class=NULL,
                    error_message=NULL
                WHERE status='terminal_404'
                """
            )
    elif tamper == "unexpected-sequence":
        with sqlite3.connect(source) as connection:
            connection.execute(
                "INSERT INTO sqlite_sequence(name,seq) VALUES ('rogue',99)"
            )
    elif tamper == "sqlite-stat1":
        with sqlite3.connect(source) as connection:
            connection.execute("ANALYZE")
    elif tamper == "nonempty-wal":
        Path(f"{source}-wal").write_bytes(b"pending-wal")
    elif tamper == "nonempty-shm":
        Path(f"{source}-shm").write_bytes(b"pending-shm")
    source_before = (
        source.stat().st_size,
        source.stat().st_mtime_ns,
        source.stat().st_ino,
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    with pytest.raises(FetchStateMigrationError, match=error):
        project_fetch_state_v3_to_v4(source, destination)

    assert not destination.exists()
    assert (
        source.stat().st_size,
        source.stat().st_mtime_ns,
        source.stat().st_ino,
        hashlib.sha256(source.read_bytes()).hexdigest(),
    ) == source_before
    assert not Path(f"{source}-journal").exists()


def test_fetch_state_v3_inspection_reads_guarded_inode_and_detects_path_swap(
    tmp_path: Path,
) -> None:
    from scripts.ci_fetch_state_migration import (
        FetchStateMigrationError,
        _assert_same_snapshot,
        _open_inspection,
    )

    source = tmp_path / "source.sqlite"
    decoy = tmp_path / "decoy.sqlite"
    moved_source = tmp_path / "moved-source.sqlite"
    original_store = tmp_path / "original-store"
    decoy_store = tmp_path / "decoy-store"
    _write_fetch_state_v3_projection_fixture(
        source,
        transitional=False,
        content_store_path=original_store,
    )
    _write_fetch_state_v3_projection_fixture(
        decoy,
        transitional=False,
        content_store_path=decoy_store,
    )
    opened = _open_inspection(
        source,
        allow_current=False,
        where="path-swap fixture",
    )
    try:
        os.replace(source, moved_source)
        os.replace(decoy, source)
        guarded_store = opened.connection.execute(
            """
            SELECT value FROM settings
            WHERE key='content_store_path'
            """
        ).fetchone()[0]
        assert guarded_store == str(original_store.resolve())
        assert guarded_store != str(decoy_store.resolve())
        with pytest.raises(
            FetchStateMigrationError,
            match="stable regular non-symlink|changed during immutable inspection",
        ):
            _assert_same_snapshot(
                opened.inspection.snapshot,
                opened.guard_descriptor,
                where="path-swap fixture",
            )
    finally:
        opened.close()


def test_fetch_state_projection_builds_only_through_guarded_descriptor(
    tmp_path: Path,
) -> None:
    from scripts.ci_fetch_state_migration import (
        CURRENT_V4_SQLITE_SCHEMA_SHA256,
        _build_destination,
        _open_inspection,
    )
    from scripts.ci_content_store import _sqlite_schema_sha256

    source = tmp_path / "source.sqlite"
    temporary = tmp_path / "projection-temporary.sqlite"
    moved_temporary = tmp_path / "guarded-projection.sqlite"
    _write_fetch_state_v3_projection_fixture(
        source,
        transitional=False,
        content_store_path=tmp_path / "store",
    )
    opened = _open_inspection(
        source,
        allow_current=False,
        where="guarded-destination fixture",
    )
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_RDWR,
        0o600,
    )
    try:
        os.replace(temporary, moved_temporary)
        temporary.write_bytes(b"decoy must remain untouched")
        attempts, ledger = _build_destination(
            opened.connection,
            source_layout=opened.inspection.layout,
            temporary_descriptor=descriptor,
        )
        assert attempts == 9
        assert len(ledger) == 1
        assert temporary.read_bytes() == b"decoy must remain untouched"
    finally:
        os.close(descriptor)
        opened.close()

    with sqlite3.connect(moved_temporary) as projected:
        projected.row_factory = sqlite3.Row
        assert _sqlite_schema_sha256(projected) == (
            CURRENT_V4_SQLITE_SCHEMA_SHA256
        )
        assert projected.execute(
            "SELECT value FROM settings WHERE key='schema'"
        ).fetchone()[0] == ci.SCHEMA_VERSION


def test_fetch_state_projection_inspects_source_before_destination_creation(
    tmp_path: Path,
) -> None:
    from scripts.ci_fetch_state_migration import (
        FetchStateMigrationError,
        project_fetch_state_v3_to_v4,
    )

    source = tmp_path / "invalid-source.sqlite"
    destination_parent = tmp_path / "must-not-be-created"
    _write_fetch_state_v3_projection_fixture(
        source,
        transitional=False,
        content_store_path=tmp_path / "store",
    )
    with sqlite3.connect(source) as connection:
        connection.execute(
            "UPDATE settings SET value=? WHERE key='schema'",
            (ci.SCHEMA_VERSION,),
        )

    with pytest.raises(
        FetchStateMigrationError,
        match="settings/schema pair is unsupported",
    ):
        project_fetch_state_v3_to_v4(
            source,
            destination_parent / "projected.sqlite",
        )

    assert not destination_parent.exists()


def test_fetch_state_projection_requires_private_existing_destination_parent(
    tmp_path: Path,
) -> None:
    from scripts.ci_fetch_state_migration import (
        FetchStateMigrationError,
        project_fetch_state_v3_to_v4,
    )

    source = tmp_path / "valid-source.sqlite"
    public_parent = tmp_path / "public-parent"
    public_parent.mkdir(mode=0o755)
    public_parent.chmod(0o755)
    _write_fetch_state_v3_projection_fixture(
        source,
        transitional=False,
        content_store_path=tmp_path / "store",
    )

    with pytest.raises(
        FetchStateMigrationError,
        match="without group/world permissions",
    ):
        project_fetch_state_v3_to_v4(
            source,
            public_parent / "projected.sqlite",
        )

    assert not (public_parent / "projected.sqlite").exists()


def test_frozen_fetch_state_rejects_raw_v3_and_accepts_projection(
    tmp_path: Path,
) -> None:
    from scripts.ci_fetch_state_migration import project_fetch_state_v3_to_v4
    from scripts.export_ci_content_store_case5 import (
        ExportError,
        FrozenFetchState,
    )

    class FixtureTokenizer:
        contract: dict[str, object] = {}
        fingerprint = "f" * 64

    class FixtureStore:
        def __init__(self, root: Path):
            self.root = root.resolve()
            self.receipt = {"script_sha256": "3" * 64}

    source = tmp_path / "raw-v3.sqlite"
    destination = tmp_path / "projected-v4.sqlite"
    store_path = tmp_path / "store"
    _write_fetch_state_v3_projection_fixture(
        source,
        transitional=False,
        content_store_path=store_path,
    )
    tokenizer = FixtureTokenizer()
    store = FixtureStore(store_path)

    with pytest.raises(ExportError, match="frozen v4"):
        with FrozenFetchState(
            source,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
        ):
            pass

    project_fetch_state_v3_to_v4(source, destination)
    with FrozenFetchState(
        destination,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
    ) as frozen:
        assert frozen.settings["schema"] == ci.SCHEMA_VERSION
        assert frozen.summary["attempt_statuses"]["retry"] == 2


def test_fetcher_lease_precedes_work_and_content_store_mutation(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "fetch.sqlite"
    work_path = tmp_path / "fresh-work"
    store_path = tmp_path / "existing-store"
    store_path.mkdir()
    sentinel = store_path / "sentinel"
    sentinel.write_bytes(b"must not change")
    store_before = (
        store_path.stat().st_mtime_ns,
        sentinel.stat().st_mtime_ns,
        hashlib.sha256(sentinel.read_bytes()).hexdigest(),
        tuple(sorted(path.name for path in store_path.iterdir())),
    )
    descriptor = ci._acquire_fetch_state_process_lease(
        state_path,
        owner="test-finalizer",
    )
    try:
        with pytest.raises(ci.BindingError, match="live process lease"):
            ci.CIStreamFetcher(
                inventory_path=tmp_path / "inventory.sqlite",
                state_path=state_path,
                content_store_path=store_path,
                tokenizer_path=tmp_path / "tokenizer.json",
                tokens=["secret"],
                progress_path=tmp_path / "progress.json",
                receipt_path=tmp_path / "receipt.json",
                work_path=work_path,
            )
    finally:
        ci._release_fetch_state_process_lease(descriptor)

    assert not work_path.exists()
    assert (
        store_path.stat().st_mtime_ns,
        sentinel.stat().st_mtime_ns,
        hashlib.sha256(sentinel.read_bytes()).hexdigest(),
        tuple(sorted(path.name for path in store_path.iterdir())),
    ) == store_before


def test_fetch_state_lease_rejects_raw_state_symlink_without_touching_targets(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "fetch.sqlite"
    state_target = tmp_path / "state-target.sqlite"
    state_target.write_bytes(b"state target")
    lease_target = tmp_path / "lease-target.txt"
    lease_target.write_bytes(b"lease target")
    state_path.symlink_to(state_target)
    state_path.with_name(f"{state_path.name}.lease").symlink_to(lease_target)
    before = (
        state_target.read_bytes(),
        state_target.stat().st_mtime_ns,
        lease_target.read_bytes(),
        lease_target.stat().st_mtime_ns,
    )

    with pytest.raises(ci.BindingError, match="fetch-state path is unsafe"):
        ci._acquire_fetch_state_process_lease(
            state_path,
            owner="symlink-test",
        )

    assert (
        state_target.read_bytes(),
        state_target.stat().st_mtime_ns,
        lease_target.read_bytes(),
        lease_target.stat().st_mtime_ns,
    ) == before


def test_fetch_state_inode_lease_blocks_renamed_hardlink_alias(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "fetch.sqlite"
    alias_path = tmp_path / "same-state.sqlite"
    state_path.write_bytes(b"guarded state inode")
    descriptor = ci._acquire_fetch_state_process_lease(
        state_path,
        owner="original-path",
    )
    try:
        os.link(state_path, alias_path)
        state_path.unlink()
        with pytest.raises(ci.BindingError, match="live process lease"):
            ci._acquire_fetch_state_process_lease(
                alias_path,
                owner="hardlink-alias",
            )
    finally:
        ci._release_fetch_state_process_lease(descriptor)


def test_fetch_state_early_constructor_failure_releases_adopted_lease(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "fetch.sqlite"
    descriptor = ci._acquire_fetch_state_process_lease(
        state_path,
        owner="constructor-caller",
    )
    with pytest.raises(
        ValueError,
        match="content-store creator script binding",
    ):
        ci.FetchState(
            state_path,
            inventory_path=tmp_path / "inventory.sqlite",
            content_store_path=tmp_path / "store",
            tokenizer=ci.ExactTokenizer(
                _tokenizer(tmp_path / "tokenizer.json")
            ),
            resume=False,
            content_store_creator_script_sha256="invalid",
            _adopted_lease_descriptor=descriptor,
        )

    reacquired = ci._acquire_fetch_state_process_lease(
        state_path,
        owner="cleanup-proof",
    )
    ci._release_fetch_state_process_lease(reacquired)


def test_fetcher_constructor_failure_releases_adopted_state_lease(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state_path = tmp_path / "fetch.sqlite"
    with pytest.raises(ValueError, match="token pool must not be empty"):
        ci.CIStreamFetcher(
            inventory_path=inventory,
            state_path=state_path,
            content_store_path=tmp_path / "store",
            tokenizer_path=_tokenizer(tmp_path / "tokenizer.json"),
            tokens=[],
            progress_path=tmp_path / "progress.json",
            receipt_path=tmp_path / "receipt.json",
            work_path=tmp_path / "work",
            target_unique_tokens=1,
        )

    descriptor = ci._acquire_fetch_state_process_lease(
        state_path,
        owner="cleanup-proof",
    )
    ci._release_fetch_state_process_lease(descriptor)


def _fake_parser(
    raw: bytes,
    metadata: Mapping[str, object],
    *,
    max_chunk_chars: int,
) -> dict[str, object]:
    assert max_chunk_chars > 0
    text = raw.decode()
    digest = hashlib.sha256(text.encode()).hexdigest()
    sidecar = {
        "schema": "fake-sidecar-v1",
        "sidecar_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    return {
        "canonical_text": text,
        "dedup_text": text,
        "sections": [
            {
                "ordinal": 0,
                "section_id": "section:000000",
                "title": "build",
            }
        ],
        "chunks": [
            {
                "ordinal": 0,
                "section_id": "section:000000",
                "section_ordinal": 0,
                "step_ordinal": 0,
                "char_start": 0,
                "char_end": len(text),
                "sha256": digest,
                "text": text,
            }
        ],
        "sidecar": sidecar,
    }


def _zero_training_parser(
    raw: bytes,
    metadata: Mapping[str, object],
    *,
    max_chunk_chars: int,
) -> dict[str, object]:
    assert max_chunk_chars > 0
    assert raw == b""
    sidecar = {
        "schema": "fake-empty-sidecar-v1",
        "sidecar_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    return {
        "canonical_text": "",
        "dedup_text": "",
        "sections": [],
        "chunks": [],
        "sidecar": sidecar,
    }


def test_discovery_keyset_sweeps_beyond_first_batch_and_wraps(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 25)
    tokenizer = ci.ExactTokenizer(_tokenizer(tmp_path / "tokenizer.json"))
    state = ci.FetchState(
        tmp_path / "state.sqlite",
        inventory_path=inventory,
        content_store_path=tmp_path / "store",
        tokenizer=tokenizer,
        resume=False,
    )
    try:
        state.discover(row_limit=10)
        state.discover(row_limit=10)
        state.discover(row_limit=10)
        assert state._connection.execute(
            "SELECT COUNT(*) FROM attempts"
        ).fetchone()[0] == 25
        assert state._discovery_cursor is not None
        state.discover(row_limit=10)
        assert state._discovery_cursor is None
        state.discover(row_limit=10)
        assert state._connection.execute(
            "SELECT COUNT(*) FROM attempts"
        ).fetchone()[0] == 25
    finally:
        state.close()


def test_exhaustive_discovery_cursor_survives_processes_past_20k(
    tmp_path: Path,
) -> None:
    run_count = 20_001
    inventory = _inventory(tmp_path / "inventory.sqlite", run_count)
    state = tmp_path / "fetch.sqlite"
    store = tmp_path / "store"
    expected_sha256 = _attempt_set_sha256(
        [("owner/repo", run_id, 1) for run_id in range(1, run_count + 1)]
    )
    program = """
import json
from pathlib import Path
import sys
from scripts.ci_stream_fetch import (
    ExactTokenizer,
    ExhaustiveInventoryBinding,
    FetchState,
)
inventory, state, store, tokenizer, run_count, expected_sha = sys.argv[1:]
state_path = Path(state)
fetch = FetchState(
    state_path,
    inventory_path=inventory,
    content_store_path=store,
    tokenizer=ExactTokenizer(tokenizer),
    resume=state_path.exists(),
)
binding = ExhaustiveInventoryBinding(
    receipt_path=Path("/verified/inventory-receipt.json"),
    receipt_sha256="a" * 64,
    database_sha256="b" * 64,
    db_logical_sha256="c" * 64,
    expected_run_count=int(run_count),
    expected_attempt_count=int(run_count),
    expected_attempt_set_sha256=expected_sha,
)
inserted = fetch.discover(
    row_limit=20_000,
    exhaustive_inventory=binding,
)
summary = fetch.exhaustive_discovery_summary()
fetch.close()
print(json.dumps({"inserted": inserted, "summary": summary}, sort_keys=True))
"""

    def run_once() -> dict[str, Any]:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                str(inventory),
                str(state),
                str(store),
                str(_FROZEN_TOKENIZER),
                str(run_count),
                expected_sha256,
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    first = run_once()
    second = run_once()
    third = run_once()

    assert first["inserted"] == 20_000
    assert first["summary"]["discovery_eof"] is False
    assert second["inserted"] == 1
    assert second["summary"]["discovery_eof"] is True
    assert second["summary"]["rows_seen"] == run_count
    assert third["inserted"] == 0
    assert third["summary"] == second["summary"]
    with sqlite3.connect(state) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM attempts"
        ).fetchone()[0] == run_count
        assert "discovery_sweeps" not in {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table'"
            )
        }


def test_exhaustive_discovery_expands_exact_rerun_keyset_and_requeues_failed(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    _replace_inventory_run(inventory, _run_metadata(1, attempt=3))
    keys = [("owner/repo", 1, attempt) for attempt in (1, 2, 3)]
    state = ci.FetchState(
        tmp_path / "fetch.sqlite",
        inventory_path=inventory,
        content_store_path=tmp_path / "store",
        tokenizer=ci.ExactTokenizer(_FROZEN_TOKENIZER),
        resume=False,
    )
    try:
        assert state.discover(
            row_limit=10,
            exhaustive_inventory=_exhaustive_binding(
                run_count=1,
                attempt_keys=keys,
            ),
        ) == 3
        summary = state.exhaustive_discovery_summary()
        assert summary is not None and summary["discovery_eof"] is True
        with state._connection:
            state._connection.execute(
                """
                UPDATE attempts SET status='failed'
                WHERE repo='owner/repo' AND run_id=1 AND attempt=2
                """
            )
        assert state.requeue_failed() == 1
        assert [
            tuple(row)
            for row in state._connection.execute(
                """
                SELECT repo,run_id,attempt
                FROM attempts ORDER BY repo,run_id,attempt
                """
            )
        ] == keys
        assert state._connection.execute(
            """
            SELECT status FROM attempts
            WHERE repo='owner/repo' AND run_id=1 AND attempt=2
            """
        ).fetchone()[0] == "retry"
    finally:
        state.close()


@pytest.mark.parametrize(
    "blocked_status",
    ["pending", "processing", "retry", "failed"],
)
def test_exhaustive_v4_proof_blocks_every_incomplete_status(
    tmp_path: Path,
    blocked_status: str,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    keys = [("owner/repo", 1, 1)]
    state_path = tmp_path / "fetch.sqlite"
    state = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=tmp_path / "store",
        tokenizer=ci.ExactTokenizer(_FROZEN_TOKENIZER),
        resume=False,
    )
    try:
        state.discover(
            row_limit=10,
            exhaustive_inventory=_exhaustive_binding(
                run_count=1,
                attempt_keys=keys,
            ),
        )
        with state._connection:
            state._connection.execute(
                "UPDATE attempts SET status=?",
                (blocked_status,),
            )
    finally:
        state.close()
    inventory_connection = sqlite3.connect(inventory)
    fetch_connection = sqlite3.connect(state_path)
    inventory_connection.row_factory = sqlite3.Row
    fetch_connection.row_factory = sqlite3.Row
    try:
        with pytest.raises(
            ReceiptFinalizationError,
            match="requires only done, empty",
        ):
            exhaustive_coverage_proof(
                inventory_connection,
                fetch_connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=keys,
                ),
                require_discovery_eof=True,
                discovery_sweep=ci.load_exhaustive_discovery_sidecar(
                    ci.exhaustive_discovery_sidecar_path(state_path)
                ),
            )
    finally:
        fetch_connection.close()
        inventory_connection.close()


@pytest.mark.parametrize("defect", ["missing", "extra"])
def test_exhaustive_v4_proof_blocks_missing_and_extra_attempts(
    tmp_path: Path,
    defect: str,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    keys = [("owner/repo", 1, 1)]
    state_path = tmp_path / "fetch.sqlite"
    state = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=tmp_path / "store",
        tokenizer=ci.ExactTokenizer(_FROZEN_TOKENIZER),
        resume=False,
    )
    try:
        state.discover(
            row_limit=10,
            exhaustive_inventory=_exhaustive_binding(
                run_count=1,
                attempt_keys=keys,
            ),
        )
        with state._connection:
            state._connection.execute("UPDATE attempts SET status='empty'")
            if defect == "missing":
                state._connection.execute("DELETE FROM attempts")
            else:
                columns = [
                    str(row[1])
                    for row in state._connection.execute(
                        "PRAGMA table_info(attempts)"
                    )
                ]
                row = dict(
                    state._connection.execute(
                        "SELECT * FROM attempts"
                    ).fetchone()
                )
                row["run_id"] = 999
                state._connection.execute(
                    f"""
                    INSERT INTO attempts({",".join(columns)})
                    VALUES ({",".join("?" for _ in columns)})
                    """,
                    tuple(row[column] for column in columns),
                )
    finally:
        state.close()
    inventory_connection = sqlite3.connect(inventory)
    fetch_connection = sqlite3.connect(state_path)
    inventory_connection.row_factory = sqlite3.Row
    fetch_connection.row_factory = sqlite3.Row
    try:
        with pytest.raises(
            ReceiptFinalizationError,
            match="not exactly equal",
        ):
            exhaustive_coverage_proof(
                inventory_connection,
                fetch_connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=keys,
                ),
                require_discovery_eof=True,
                discovery_sweep=ci.load_exhaustive_discovery_sidecar(
                    ci.exhaustive_discovery_sidecar_path(state_path)
                ),
            )
    finally:
        fetch_connection.close()
        inventory_connection.close()


def test_rerun_attempt_binds_exact_attempt_metadata_before_cas_write(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    seed = _run_metadata(1, attempt=2)
    seed["display_title"] = "second attempt"
    seed["conclusion"] = "success"
    seed_sha = _replace_inventory_run(inventory, seed)
    first = _run_metadata(1, attempt=1)
    first["display_title"] = "first attempt"
    first["conclusion"] = "failure"
    first_raw = json.dumps(
        first,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    first_sha = hashlib.sha256(first_raw).hexdigest()
    github = FakeGitHub(
        _zip_bytes(),
        attempt_metadata={1: first},
    )
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_tokenizer(tmp_path / "tokenizer.json"),
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        progress = fetcher.run(continuous=False, max_runs=1)
        assert progress["fetch"]["attempt_statuses"] == {
            "done": 1,
            "pending": 1,
        }
        rows = fetcher.state._connection.execute(
            """
            SELECT attempt,run_metadata_sha256,run_metadata_source,
                   run_metadata_source_attempt,run_metadata_exact,
                   inventory_seed_attempt,inventory_seed_metadata_sha256
            FROM attempts ORDER BY attempt
            """
        ).fetchall()
        assert tuple(rows[0]) == (
            1,
            first_sha,
            "github-workflow-run-attempt-api",
            1,
            1,
            2,
            seed_sha,
        )
        assert tuple(rows[1]) == (
            2,
            seed_sha,
            "inventory-run-list",
            2,
            1,
            2,
            seed_sha,
        )
        assert github.api_urls[0].endswith(
            "/repos/owner/repo/actions/runs/1/attempts/1"
        )
        occurrence = next(fetcher.store.iter_occurrences())
        provenance = occurrence["provenance"]
        assert provenance["schema"] == "cppmega_ci_chunk_occurrence_v3"
        assert provenance["workflow"]["display_title"] == "first attempt"
        assert provenance["workflow"]["conclusion"] == "failure"
        assert provenance["run_metadata_evidence"] == {
            "exact_attempt_match": True,
            "source": "github-workflow-run-attempt-api",
            "source_attempt": 1,
            "sha256": first_sha,
            "inventory_seed_attempt": 2,
            "inventory_seed_metadata_sha256": seed_sha,
        }
        assert progress["fetch"]["run_metadata"] == {
            "exact_attempts": 2,
            "unresolved_attempts": 0,
            "exact_by_source": {
                "github-workflow-run-attempt-api": 1,
                "inventory-run-list": 1,
            },
            "unresolved_by_status": {},
            "content_attempts_without_exact_metadata": 0,
        }
    finally:
        fetcher.close()


def test_rerun_metadata_identity_mismatch_fails_before_cas_write(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    seed = _run_metadata(1, attempt=2)
    _replace_inventory_run(inventory, seed)
    github = FakeGitHub(
        _zip_bytes(),
        attempt_metadata={1: _run_metadata(1, attempt=2)},
    )
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_tokenizer(tmp_path / "tokenizer.json"),
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        progress = fetcher.run(continuous=False, max_runs=1)
        assert progress["fetch"]["attempt_statuses"] == {
            "pending": 1,
            "retry": 1,
        }
        assert fetcher.store.status()["counters"]["occurrence_count"] == 0
        row = fetcher.state._connection.execute(
            """
            SELECT run_metadata_exact,error_class,error_message
            FROM attempts WHERE attempt=1
            """
        ).fetchone()
        assert int(row["run_metadata_exact"]) == 0
        assert row["error_class"] == "MalformedResponseError"
        assert "does not match 1" in row["error_message"]
    finally:
        fetcher.close()


def test_zip_validation_rejects_traversal_symlink_and_duplicate(
    tmp_path: Path,
) -> None:
    traversal = tmp_path / "traversal.zip"
    traversal.write_bytes(_zip_bytes({"../secret.txt": b"x"}))
    with pytest.raises(ci.ArchiveError, match="traversal"):
        ci._safe_zip_infos(
            traversal,
            max_members=10,
            max_member_bytes=100,
            max_uncompressed_bytes=100,
        )

    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as archive:
        info = zipfile.ZipInfo("link.txt")
        info.external_attr = (0o120777 << 16)
        archive.writestr(info, "target")
    with pytest.raises(ci.ArchiveError, match="symlink"):
        ci._safe_zip_infos(
            symlink,
            max_members=10,
            max_member_bytes=100,
            max_uncompressed_bytes=100,
        )

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("same.txt", "one")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("same.txt", "two")
    with pytest.raises(ci.ArchiveError, match="duplicate"):
        ci._safe_zip_infos(
            duplicate,
            max_members=10,
            max_member_bytes=100,
            max_uncompressed_bytes=100,
        )


class FakeGitHub:
    def __init__(
        self,
        archive: bytes,
        *,
        attempt_metadata: Mapping[int, Mapping[str, object]] | None = None,
        log_status: int = 302,
    ):
        self.archive = archive
        self.log_status = log_status
        self.attempt_metadata = {
            int(attempt): dict(value)
            for attempt, value in (attempt_metadata or {}).items()
        }
        self.api_headers: list[dict[str, str]] = []
        self.api_urls: list[str] = []
        self.signed_url: str | None = None

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> ci.HTTPResponse:
        assert method == "GET"
        assert timeout > 0
        self.api_headers.append(dict(headers))
        self.api_urls.append(url)
        attempt_suffix = url.rsplit("/attempts/", 1)[-1]
        if attempt_suffix.isdigit():
            attempt = int(attempt_suffix)
            if attempt not in self.attempt_metadata:
                raise AssertionError(url)
            return ci.HTTPResponse(
                200,
                {},
                json.dumps(self.attempt_metadata[attempt]).encode(),
            )
        if url.endswith("/jobs?filter=all&per_page=100&page=1"):
            body = {
                "total_count": 1,
                "jobs": [
                    {
                        "id": 99,
                        "name": "build",
                        "status": "completed",
                        "conclusion": "success",
                        "runner_name": "GitHub Actions 1",
                        "labels": ["ubuntu-24.04"],
                        "steps": [{"name": "compile", "conclusion": "success"}],
                    }
                ],
            }
            return ci.HTTPResponse(200, {}, json.dumps(body).encode())
        if url.endswith("/logs"):
            if self.log_status in {404, 410}:
                return ci.HTTPResponse(
                    self.log_status,
                    {},
                    json.dumps(
                        {
                            "message": (
                                "Not Found"
                                if self.log_status == 404
                                else "Gone"
                            )
                        }
                    ).encode(),
                )
            assert self.log_status == 302
            return ci.HTTPResponse(
                302,
                {
                    "Location": (
                        "https://results.example.test/archive?"
                        "sig=super-secret&se=tomorrow"
                    )
                },
                b"",
            )
        raise AssertionError(url)

    def download(
        self,
        url: str,
        destination: Path,
        *,
        timeout: float,
        max_bytes: int,
    ) -> tuple[int, str]:
        assert timeout > 0
        assert len(self.archive) < max_bytes
        self.signed_url = url
        destination.write_bytes(self.archive)
        return len(self.archive), hashlib.sha256(self.archive).hexdigest()


def test_terminal_log_probe_does_not_spend_a_jobs_request(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    github = FakeGitHub(_zip_bytes(), log_status=410)
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_tokenizer(tmp_path / "tokenizer.json"),
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        assert any(url.endswith("/logs") for url in github.api_urls)
        assert not any("/jobs?" in url for url in github.api_urls)
        row = fetcher.state._connection.execute(
            "SELECT status,terminal_http_status FROM attempts"
        ).fetchone()
        assert tuple(row) == ("terminal_410", 410)
        assert github.signed_url is None
    finally:
        fetcher.close()


def test_lone_masked_404_retries_and_never_terminalizes(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    github = FakeGitHub(_zip_bytes(), log_status=404)
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_tokenizer(tmp_path / "tokenizer.json"),
        tokens=["only-token"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        row = fetcher.state._connection.execute(
            "SELECT status,terminal_http_status FROM attempts"
        ).fetchone()
        assert tuple(row) == ("retry", None)
        candidates = fetcher.state._connection.execute(
            """
            SELECT COUNT(*),COUNT(DISTINCT error_message)
            FROM request_ledger
            WHERE outcome='terminal_candidate' AND http_status=404
            """
        ).fetchone()
        assert tuple(candidates) == (ci.DEFAULT_API_ATTEMPTS, 1)
        assert not any("/jobs?" in url for url in github.api_urls)
    finally:
        fetcher.close()


def test_distinct_token_404_plus_jobs_access_is_receiptable(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state_path = tmp_path / "fetch.sqlite"
    github = FakeGitHub(_zip_bytes(), log_status=404)
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=tmp_path / "store",
        tokenizer_path=_tokenizer(tmp_path / "tokenizer.json"),
        tokens=["token-a", "token-b"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        row = fetcher.state._connection.execute(
            """
            SELECT status,terminal_http_status,jobs_sha256,jobs_zlib
            FROM attempts
            """
        ).fetchone()
        assert row["status"] == "terminal_404"
        assert row["terminal_http_status"] == 404
        assert row["jobs_sha256"] is not None
        assert row["jobs_zlib"] is not None
        evidence = [
            json.loads(str(item[0]))
            for item in fetcher.state._connection.execute(
                """
                SELECT error_message FROM request_ledger
                WHERE outcome='terminal_candidate' AND http_status=404
                ORDER BY id
                """
            )
        ]
        assert len({item["token_sha256"] for item in evidence}) == 2
        inventory_connection = sqlite3.connect(inventory)
        inventory_connection.row_factory = sqlite3.Row
        try:
            proof = exhaustive_coverage_proof(
                inventory_connection,
                fetcher.state._connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=[("owner/repo", 1, 1)],
                ),
                require_discovery_eof=False,
            )
        finally:
            inventory_connection.close()
        assert proof["terminal_statuses"] == {"terminal_404": 1}
    finally:
        fetcher.close()


def test_empty_attempt_requires_archive_jobs_and_zero_member_proof(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state_path = tmp_path / "fetch.sqlite"
    state = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=tmp_path / "store",
        tokenizer=ci.ExactTokenizer(_FROZEN_TOKENIZER),
        resume=False,
    )
    try:
        state.discover()
        with state._connection:
            state._connection.execute("UPDATE attempts SET status='empty'")
    finally:
        state.close()
    inventory_connection = sqlite3.connect(inventory)
    fetch_connection = sqlite3.connect(state_path)
    inventory_connection.row_factory = sqlite3.Row
    fetch_connection.row_factory = sqlite3.Row
    try:
        with pytest.raises(
            ReceiptFinalizationError,
            match="empty attempt archive_sha256",
        ):
            exhaustive_coverage_proof(
                inventory_connection,
                fetch_connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=[("owner/repo", 1, 1)],
                ),
                require_discovery_eof=False,
            )
    finally:
        fetch_connection.close()
        inventory_connection.close()


def test_verified_empty_zip_attempt_satisfies_empty_proof(
    tmp_path: Path,
) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w"):
        pass
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    github = FakeGitHub(archive_buffer.getvalue())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_tokenizer(tmp_path / "tokenizer.json"),
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        inventory_connection = sqlite3.connect(inventory)
        inventory_connection.row_factory = sqlite3.Row
        try:
            proof = exhaustive_coverage_proof(
                inventory_connection,
                fetcher.state._connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=[("owner/repo", 1, 1)],
                ),
                require_discovery_eof=False,
            )
        finally:
            inventory_connection.close()
        assert proof["terminal_statuses"] == {"empty": 1}
    finally:
        fetcher.close()


def test_nonempty_archive_with_zero_training_chunks_is_terminal_and_resumable(
    tmp_path: Path,
) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("0_99.txt", b"")
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    github = FakeGitHub(archive_buffer.getvalue())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_zero_training_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        row = fetcher.state._connection.execute(
            """
            SELECT status,tries,archive_zlib,
                   member_count,chunk_count,occurrence_tokens
            FROM attempts
            """
        ).fetchone()
        assert tuple(row) == ("empty", 1, None, 1, 0, 0)
        member = fetcher.state._connection.execute(
            """
            SELECT archive_member,raw_size,chunk_count,occurrence_tokens,
                   raw_sha256,canonical_sha256,dedup_sha256
            FROM members
            """
        ).fetchone()
        assert tuple(member[:4]) == ("0_99.txt", 0, 0, 0)
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in member[4:])
        inventory_connection = sqlite3.connect(inventory)
        inventory_connection.row_factory = sqlite3.Row
        try:
            proof = exhaustive_coverage_proof(
                inventory_connection,
                fetcher.state._connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=[("owner/repo", 1, 1)],
                ),
                require_discovery_eof=False,
            )
        finally:
            inventory_connection.close()
        assert proof["terminal_statuses"] == {"empty": 1}
        assert fetcher.state.summary()["members"] == 1
    finally:
        fetcher.close()

    resumed = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_zero_training_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        resume=True,
        sleeper=lambda _: None,
    )
    try:
        resumed.run(continuous=False, max_runs=1)
        assert tuple(
            resumed.state._connection.execute(
                """
                SELECT status,tries,member_count,chunk_count,
                       occurrence_tokens
                FROM attempts
                """
            ).fetchone()
        ) == ("empty", 1, 1, 0, 0)
    finally:
        resumed.close()


def test_preserved_archive_recovery_replays_into_fetch_and_final_receipt(
    tmp_path: Path,
) -> None:
    from scripts.ci_stream_receipts import finalize_fetch_receipts

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("0_99.txt", b"first member\n")
        archive.writestr("1_test.txt", b"second member\n")
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    work_path = tmp_path / "work"
    rescue_path = tmp_path / "rescue"
    rescue_path.mkdir()
    github = FakeGitHub(archive_buffer.getvalue())

    def fail_after_first_member(
        raw: bytes,
        metadata: Mapping[str, object],
        *,
        max_chunk_chars: int,
    ) -> dict[str, object]:
        if metadata["archive_member"] == "1_test.txt":
            raise ci.FetchError("deliberate interrupted parse")
        return _fake_parser(
            raw,
            metadata,
            max_chunk_chars=max_chunk_chars,
        )

    first = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "unused-live-receipt.json",
        work_path=work_path,
        parser=fail_after_first_member,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        first.run(continuous=False, max_runs=4)
        attempt = first.state._connection.execute(
            """
            SELECT status,member_count,chunk_count,occurrence_tokens
            FROM attempts
            """
        ).fetchone()
        assert tuple(attempt) == ("failed", 0, 0, 0)
        durable = first.state._connection.execute(
            """
            SELECT COUNT(*),SUM(chunk_count),SUM(occurrence_tokens)
            FROM members
            """
        ).fetchone()
        assert tuple(durable) == (1, 1, durable[2])
        assert int(durable[2]) > 0
    finally:
        first.close()

    plans = preserved_recovery.build_plans(
        state_path=state_path,
        work_dir=work_path,
        rescue_spool=rescue_path,
        target=("owner/repo", 1, 1),
    )
    assert len(plans) == 1
    recovery_result = preserved_recovery.apply_plan(plans[0])
    assert recovery_result["status"] == "requeued"

    resumed = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "unused-live-receipt.json",
        rescue_path=rescue_path,
        work_path=work_path,
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        resume=True,
        sleeper=lambda _: None,
    )
    try:
        resumed.run(continuous=False, max_runs=1)
        row = resumed.state._connection.execute(
            """
            SELECT status,archive_source,member_count,
                   chunk_count,occurrence_tokens
            FROM attempts
            """
        ).fetchone()
        assert row["status"] == "done"
        assert row["archive_source"] == "preserved-local-archive"
        assert tuple(row[2:4]) == (2, 2)
        assert int(row["occurrence_tokens"]) > 0
        assert resumed.state._connection.execute(
            """
            SELECT COUNT(*) FROM request_ledger
            WHERE endpoint='operator/preserved_archive_recovery'
              AND outcome='preserved_archive_consumed'
              AND error_class='PreservedArchiveProvenance'
            """
        ).fetchone()[0] == 1
        audit_row = resumed.state._connection.execute(
            """
            SELECT id,error_message FROM request_ledger
            WHERE endpoint='operator/preserved_archive_recovery'
              AND outcome='operator/preserved_archive_recovery'
              AND error_class='PreservedArchiveRecoveryReceipt'
            """
        ).fetchone()
        consumed_row = resumed.state._connection.execute(
            """
            SELECT id,error_message FROM request_ledger
            WHERE endpoint='operator/preserved_archive_recovery'
              AND outcome='preserved_archive_consumed'
              AND error_class='PreservedArchiveProvenance'
            """
        ).fetchone()
        assert audit_row is not None and consumed_row is not None
        original_audit = str(audit_row["error_message"])
        original_consumed = str(consumed_row["error_message"])
        forged_audit = json.loads(original_audit)
        forged_provenance = json.loads(original_consumed)
        forged_receipt = forged_provenance["recovery_receipt"]["receipt"]
        witness = forged_receipt["proof"]["durable_member_witness"]
        witness["members"][0]["job_key"] = "forged:durable-member"
        witness_tuples = [
            [
                item["archive_member"],
                item["job_key"],
                item["raw_sha256"],
                item["raw_size"],
                item["chunk_count"],
                item["occurrence_tokens"],
            ]
            for item in witness["members"]
        ]
        witness["set_sha256"] = ci._sha256_bytes(
            ci._canonical_json_bytes(witness_tuples)
        )
        forged_receipt["recovery_id"] = ci._sha256_bytes(
            ci._canonical_json_bytes(forged_receipt["proof"])
        )
        forged_receipt_raw = (
            ci._canonical_json_bytes(forged_receipt) + b"\n"
        )
        forged_receipt_name = (
            "owner__repo--1--attempt-1.preserved-recovery-"
            f"{forged_receipt['recovery_id'][:16]}.json"
        )
        forged_provenance["recovery_receipt"].update(
            {
                "name": forged_receipt_name,
                "bytes": len(forged_receipt_raw),
                "sha256": ci._sha256_bytes(forged_receipt_raw),
            }
        )
        forged_audit.update(
            {
                "recovery_id": forged_receipt["recovery_id"],
                "witness_set_sha256": witness["set_sha256"],
                "receipt": {
                    "name": forged_receipt_name,
                    "bytes": len(forged_receipt_raw),
                    "sha256": ci._sha256_bytes(forged_receipt_raw),
                },
            }
        )
        with resumed.state._connection:
            resumed.state._connection.execute(
                "UPDATE request_ledger SET error_message=? WHERE id=?",
                (
                    ci._canonical_json_bytes(forged_audit).decode("utf-8"),
                    audit_row["id"],
                ),
            )
            resumed.state._connection.execute(
                "UPDATE request_ledger SET error_message=? WHERE id=?",
                (
                    ci._canonical_json_bytes(
                        forged_provenance
                    ).decode("utf-8"),
                    consumed_row["id"],
                ),
            )
        forged_inventory_connection = sqlite3.connect(inventory)
        forged_inventory_connection.row_factory = sqlite3.Row
        try:
            with pytest.raises(
                ReceiptFinalizationError,
                match="preserved member witness changed",
            ):
                exhaustive_coverage_proof(
                    forged_inventory_connection,
                    resumed.state._connection,
                    inventory_receipt=_coverage_inventory_receipt(
                        run_count=1,
                        attempt_keys=[("owner/repo", 1, 1)],
                    ),
                    require_discovery_eof=False,
                )
        finally:
            forged_inventory_connection.close()
        with resumed.state._connection:
            resumed.state._connection.execute(
                "UPDATE request_ledger SET error_message=? WHERE id=?",
                (original_audit, audit_row["id"]),
            )
            resumed.state._connection.execute(
                "UPDATE request_ledger SET error_message=? WHERE id=?",
                (original_consumed, consumed_row["id"]),
            )
        inventory_connection = sqlite3.connect(inventory)
        inventory_connection.row_factory = sqlite3.Row
        try:
            proof = exhaustive_coverage_proof(
                inventory_connection,
                resumed.state._connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=[("owner/repo", 1, 1)],
                ),
                require_discovery_eof=False,
            )
        finally:
            inventory_connection.close()
        assert proof["terminal_statuses"] == {"done": 1}
    finally:
        resumed.close()

    final_receipt = finalize_fetch_receipts(
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=_FROZEN_TOKENIZER,
        target_unique_tokens=1,
        fetch_receipt_path=tmp_path / "fetch-receipt.json",
        store_receipt_path=tmp_path / "store-receipt.json",
    )
    assert final_receipt["fetch_state"]["attempt_statuses"] == {"done": 1}

    state_connection = sqlite3.connect(state_path)
    state_connection.row_factory = sqlite3.Row
    inventory_connection = sqlite3.connect(inventory)
    inventory_connection.row_factory = sqlite3.Row
    try:
        with state_connection:
            state_connection.execute(
                """
                DELETE FROM request_ledger
                WHERE endpoint='operator/preserved_archive_recovery'
                """
            )
        with pytest.raises(
            ReceiptFinalizationError,
            match="lacks preserved-recovery ledger evidence",
        ):
            exhaustive_coverage_proof(
                inventory_connection,
                state_connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=[("owner/repo", 1, 1)],
                ),
                require_discovery_eof=False,
            )
    finally:
        inventory_connection.close()
        state_connection.close()


def test_failed_attempt_preserves_exactly_one_downloaded_archive(
    tmp_path: Path,
) -> None:
    archive_bytes = _zip_bytes()
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    work_path = tmp_path / "work"
    github = FakeGitHub(archive_bytes)

    def fail_parser(
        raw: bytes,
        metadata: Mapping[str, object],
        *,
        max_chunk_chars: int,
    ) -> dict[str, object]:
        del raw, metadata, max_chunk_chars
        raise ci.FetchError("deliberate parse failure")

    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        work_path=work_path,
        parser=fail_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        assert fetcher.state._connection.execute(
            "SELECT status FROM attempts"
        ).fetchone()[0] == "retry"
        assert not list((work_path / "tmp").iterdir())
        preserved = list((work_path / "failed").iterdir())
        assert len(preserved) == 1
        assert preserved[0].read_bytes() == archive_bytes
    finally:
        fetcher.close()


def test_job_rescue_consumption_close_time_replays_current_members(
    tmp_path: Path,
) -> None:
    from scripts.ci_stream_receipts import finalize_fetch_receipts

    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    rescue_path = tmp_path / "rescue"
    work_path = tmp_path / "work"
    github = FakeGitHub(b"unused")

    def fail_archive_download(
        url: str,
        destination: Path,
        *,
        timeout: float,
        max_bytes: int,
    ) -> tuple[int, str]:
        del url, destination, timeout, max_bytes
        raise ci.ArchiveError(
            "signed archive transport retries exhausted: IncompleteRead"
        )

    first = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "unused-live-receipt.json",
        work_path=work_path,
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=fail_archive_download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        first.run(continuous=False, max_runs=4)
        failed = first.state._connection.execute(
            """
            SELECT status,error_class,jobs_sha256,jobs_zlib
            FROM attempts
            """
        ).fetchone()
        assert failed["status"] == "failed"
        assert failed["error_class"] == "ArchiveError"
        assert failed["jobs_sha256"] is not None
        assert failed["jobs_zlib"] is not None
        assert first.state._connection.execute(
            """
            SELECT COUNT(*) FROM request_ledger
            WHERE endpoint LIKE '%/logs'
              AND http_status=302 AND outcome='success'
            """
        ).fetchone()[0] == 4
    finally:
        first.close()

    rescued_log = b"rescued build log\n"

    class CompleteJobLogResponse:
        def __init__(self, raw: bytes):
            self.status = 200
            self.headers = {"Content-Length": str(len(raw))}
            self._body = io.BytesIO(raw)

        def read(self, amount: int = -1) -> bytes:
            return self._body.read(amount)

        def close(self) -> None:
            self._body.close()

    def open_job_log(request: Any, *, timeout: float) -> CompleteJobLogResponse:
        assert timeout > 0
        assert request.full_url.endswith("/actions/jobs/99/logs")
        assert request.get_header("Authorization") == "Bearer api-secret"
        return CompleteJobLogResponse(rescued_log)

    worker = job_rescue.JobLogRescueWorker(
        state_path=state_path,
        work_dir=tmp_path / "job-rescue-work",
        rescue_spool=rescue_path,
        tokens=["api-secret"],
        workers=1,
        max_attempts=1,
        max_job_bytes=1024,
        max_total_bytes=4096,
        max_zip_bytes=4096,
        opener=open_job_log,
        sleeper=lambda _: None,
    )
    try:
        rescue_result = worker.run_once(target=("owner/repo", 1, 1))
    finally:
        worker.close()
    assert rescue_result["failed_attempts"] == 0
    assert rescue_result["results"][0]["status"] == "complete"

    resumed = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "unused-live-receipt.json",
        rescue_path=rescue_path,
        work_path=work_path,
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        resume=True,
        sleeper=lambda _: None,
    )
    try:
        resumed.run(continuous=False, max_runs=1)
        attempt = resumed.state._connection.execute(
            """
            SELECT status,archive_source,member_count,chunk_count,
                   occurrence_tokens
            FROM attempts
            """
        ).fetchone()
        assert attempt["status"] == "done"
        assert attempt["archive_source"] == "rescue-spool"
        assert int(attempt["member_count"]) == 1
        assert int(attempt["chunk_count"]) == 1
        assert int(attempt["occurrence_tokens"]) > 0
        member = resumed.state._connection.execute(
            """
            SELECT archive_member,raw_size,raw_sha256
            FROM members
            """
        ).fetchone()
        assert tuple(member) == (
            "0_99.txt",
            len(rescued_log),
            hashlib.sha256(rescued_log).hexdigest(),
        )
        audit_row = resumed.state._connection.execute(
            """
            SELECT id,error_message FROM request_ledger
            WHERE endpoint='operator/job_rescue'
              AND outcome='operator/job_rescue'
              AND error_class='JobRescueReceipt'
            """
        ).fetchone()
        consumed_row = resumed.state._connection.execute(
            """
            SELECT id,error_message FROM request_ledger
            WHERE endpoint='operator/job_rescue'
              AND outcome='rescue_archive_consumed'
              AND error_class='RescueArchiveProvenance'
            """
        ).fetchone()
        assert audit_row is not None and consumed_row is not None

        inventory_connection = sqlite3.connect(inventory)
        inventory_connection.row_factory = sqlite3.Row
        try:
            proof = exhaustive_coverage_proof(
                inventory_connection,
                resumed.state._connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=[("owner/repo", 1, 1)],
                ),
                require_discovery_eof=False,
            )
        finally:
            inventory_connection.close()
        assert proof["terminal_statuses"] == {"done": 1}

        original_audit = str(audit_row["error_message"])
        original_consumed = str(consumed_row["error_message"])
        forged_audit = json.loads(original_audit)
        forged_provenance = json.loads(original_consumed)
        forged_record = forged_provenance["resolved_jobs"]["records"][0]
        forged_record["log"]["sha256"] = "f" * 64
        forged_resolved_raw = b"".join(
            ci._canonical_json_bytes(record) + b"\n"
            for record in forged_provenance["resolved_jobs"]["records"]
        )
        forged_resolved_sha256 = hashlib.sha256(
            forged_resolved_raw
        ).hexdigest()
        forged_provenance["resolved_jobs"].update(
            {
                "bytes": len(forged_resolved_raw),
                "sha256": forged_resolved_sha256,
            }
        )
        forged_receipt = forged_provenance["job_rescue_receipt"]["receipt"]
        forged_receipt["artifacts"]["resolved_jobs"].update(
            {
                "bytes": len(forged_resolved_raw),
                "sha256": forged_resolved_sha256,
            }
        )
        forged_receipt_raw = (
            ci._canonical_json_bytes(forged_receipt) + b"\n"
        )
        forged_receipt_sha256 = hashlib.sha256(
            forged_receipt_raw
        ).hexdigest()
        forged_provenance["job_rescue_receipt"].update(
            {
                "bytes": len(forged_receipt_raw),
                "sha256": forged_receipt_sha256,
            }
        )
        forged_audit.update(
            {
                "receipt_sha256": forged_receipt_sha256,
                "source_state_sha256": hashlib.sha256(
                    ci._canonical_json_bytes(
                        forged_receipt["source_state"]
                    )
                ).hexdigest(),
                "jobs_ledger_sha256": forged_receipt["source_state"][
                    "jobs_ledger_sha256"
                ],
            }
        )
        with resumed.state._connection:
            resumed.state._connection.execute(
                "UPDATE request_ledger SET error_message=? WHERE id=?",
                (
                    ci._canonical_json_bytes(forged_audit).decode("utf-8"),
                    audit_row["id"],
                ),
            )
            resumed.state._connection.execute(
                "UPDATE request_ledger SET error_message=? WHERE id=?",
                (
                    ci._canonical_json_bytes(
                        forged_provenance
                    ).decode("utf-8"),
                    consumed_row["id"],
                ),
            )
        forged_inventory_connection = sqlite3.connect(inventory)
        forged_inventory_connection.row_factory = sqlite3.Row
        try:
            with pytest.raises(
                ReceiptFinalizationError,
                match="rescue archive provenance differs",
            ):
                exhaustive_coverage_proof(
                    forged_inventory_connection,
                    resumed.state._connection,
                    inventory_receipt=_coverage_inventory_receipt(
                        run_count=1,
                        attempt_keys=[("owner/repo", 1, 1)],
                    ),
                    require_discovery_eof=False,
                )
        finally:
            forged_inventory_connection.close()

        with resumed.state._connection:
            resumed.state._connection.execute(
                "UPDATE request_ledger SET error_message=? WHERE id=?",
                (original_audit, audit_row["id"]),
            )
            resumed.state._connection.execute(
                "UPDATE request_ledger SET error_message=? WHERE id=?",
                (original_consumed, consumed_row["id"]),
            )
    finally:
        resumed.close()

    final_receipt = finalize_fetch_receipts(
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=_FROZEN_TOKENIZER,
        target_unique_tokens=1,
        fetch_receipt_path=tmp_path / "fetch-receipt.json",
        store_receipt_path=tmp_path / "store-receipt.json",
    )
    assert final_receipt["fetch_state"]["attempt_statuses"] == {"done": 1}


@pytest.mark.parametrize(
    "evidence_field",
    ("run_metadata", "jobs", "archive"),
)
def test_empty_receipt_rejects_bounded_zlib_bomb_evidence(
    tmp_path: Path,
    evidence_field: str,
) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w"):
        pass
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    github = FakeGitHub(archive_buffer.getvalue())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        if evidence_field == "run_metadata":
            raw_size = ci.MAX_RUN_METADATA_BYTES + 1
            blob, digest = _compressed_repetition(raw_size)
            assignment = (
                "run_metadata_raw_size=?,run_metadata_sha256=?,"
                "run_metadata_zlib=?"
            )
        elif evidence_field == "jobs":
            raw_size = ci.MAX_JOBS_EVIDENCE_BYTES + 1
            blob, digest = _compressed_repetition(raw_size)
            assignment = "jobs_raw_size=?,jobs_sha256=?,jobs_zlib=?"
        else:
            raw_size = ci.MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES + 1
            blob, digest = _compressed_repetition(raw_size)
            assignment = "archive_size=?,archive_sha256=?,archive_zlib=?"
        with fetcher.state._connection:
            fetcher.state._connection.execute(
                f"UPDATE attempts SET {assignment}",
                (raw_size, digest, sqlite3.Binary(blob)),
            )
        inventory_connection = sqlite3.connect(inventory)
        inventory_connection.row_factory = sqlite3.Row
        try:
            with pytest.raises(ReceiptFinalizationError):
                exhaustive_coverage_proof(
                    inventory_connection,
                    fetcher.state._connection,
                    inventory_receipt=_coverage_inventory_receipt(
                        run_count=1,
                        attempt_keys=[("owner/repo", 1, 1)],
                    ),
                    require_discovery_eof=False,
                )
        finally:
            inventory_connection.close()
    finally:
        fetcher.close()


def test_fetch_state_rejects_run_metadata_zlib_bomb_before_decode(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state = ci.FetchState(
        tmp_path / "fetch.sqlite",
        inventory_path=inventory,
        content_store_path=tmp_path / "store",
        tokenizer=ci.ExactTokenizer(_FROZEN_TOKENIZER),
        resume=False,
    )
    try:
        state.discover()
        largest_materialized_blob = 0

        def guarded_row_factory(
            cursor: sqlite3.Cursor,
            values: tuple[object, ...],
        ) -> sqlite3.Row:
            nonlocal largest_materialized_blob
            largest_materialized_blob = max(
                (
                    len(value)
                    for value in values
                    if isinstance(value, bytes)
                ),
                default=largest_materialized_blob,
            )
            if largest_materialized_blob > ci.MAX_RUN_METADATA_COMPRESSED_BYTES:
                raise AssertionError("oversized evidence BLOB was materialized")
            return sqlite3.Row(cursor, values)

        state._connection.row_factory = guarded_row_factory
        with sqlite3.connect(state.path) as attacker:
            attacker.execute(
                """
                UPDATE attempts SET
                  run_metadata_zlib=zeroblob(?)
                """,
                (ci.MAX_RUN_METADATA_COMPRESSED_BYTES + 1,),
            )
        with pytest.raises(
            ci.BindingError,
            match="attempt evidence exceeds its versioned SQLite byte bounds",
        ):
            state.next_attempt()
        assert largest_materialized_blob <= ci.MAX_RUN_METADATA_COMPRESSED_BYTES
    finally:
        state.close()


def test_inventory_page_rechecks_blob_bounds_in_its_read_snapshot(
    tmp_path: Path,
) -> None:
    inventory_path = _inventory(tmp_path / "inventory.sqlite", 1)
    state = ci.FetchState(
        tmp_path / "fetch.sqlite",
        inventory_path=inventory_path,
        content_store_path=tmp_path / "store",
        tokenizer=ci.ExactTokenizer(_FROZEN_TOKENIZER),
        resume=False,
    )
    inventory = state._inventory_connection()
    try:
        largest_materialized_blob = 0

        def guarded_row_factory(
            cursor: sqlite3.Cursor,
            values: tuple[object, ...],
        ) -> sqlite3.Row:
            nonlocal largest_materialized_blob
            largest_materialized_blob = max(
                (
                    len(value)
                    for value in values
                    if isinstance(value, bytes)
                ),
                default=largest_materialized_blob,
            )
            if largest_materialized_blob > ci.MAX_RUN_METADATA_COMPRESSED_BYTES:
                raise AssertionError("oversized inventory BLOB was materialized")
            return sqlite3.Row(cursor, values)

        inventory.row_factory = guarded_row_factory
        with sqlite3.connect(inventory_path) as attacker:
            attacker.execute(
                """
                UPDATE runs SET metadata_blob=zeroblob(?)
                WHERE repo_key='owner/repo' AND run_id=1
                """,
                (ci.MAX_RUN_METADATA_COMPRESSED_BYTES + 1,),
            )
        with pytest.raises(
            ci.FetchError,
            match="inventory page metadata exceeds its versioned SQLite byte bound",
        ):
            state._fetch_inventory_page(
                inventory,
                row_limit=1,
                cursor=None,
            )
        assert largest_materialized_blob <= ci.MAX_RUN_METADATA_COMPRESSED_BYTES
    finally:
        inventory.close()
        state.close()


def test_genuine_done_attempt_satisfies_positive_evidence_contract(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    github = FakeGitHub(_zip_bytes())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        inventory_connection = sqlite3.connect(inventory)
        inventory_connection.row_factory = sqlite3.Row
        try:
            proof = exhaustive_coverage_proof(
                inventory_connection,
                fetcher.state._connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=[("owner/repo", 1, 1)],
                ),
                require_discovery_eof=False,
            )
        finally:
            inventory_connection.close()
        assert proof["terminal_statuses"] == {"done": 1}
    finally:
        fetcher.close()


def test_pending_attempt_forged_to_done_without_evidence_is_rejected(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state_path = tmp_path / "fetch.sqlite"
    state = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=tmp_path / "store",
        tokenizer=ci.ExactTokenizer(_FROZEN_TOKENIZER),
        resume=False,
    )
    try:
        state.discover()
        with state._connection:
            state._connection.execute("UPDATE attempts SET status='done'")
    finally:
        state.close()
    inventory_connection = sqlite3.connect(inventory)
    fetch_connection = sqlite3.connect(state_path)
    inventory_connection.row_factory = sqlite3.Row
    fetch_connection.row_factory = sqlite3.Row
    try:
        with pytest.raises(
            ReceiptFinalizationError,
            match="done attempt archive_sha256",
        ):
            exhaustive_coverage_proof(
                inventory_connection,
                fetch_connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=[("owner/repo", 1, 1)],
                ),
                require_discovery_eof=False,
            )
    finally:
        fetch_connection.close()
        inventory_connection.close()


def test_empty_attempt_rejects_forged_non_zip_archive_evidence(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state = ci.FetchState(
        tmp_path / "fetch.sqlite",
        inventory_path=inventory,
        content_store_path=tmp_path / "store",
        tokenizer=ci.ExactTokenizer(_FROZEN_TOKENIZER),
        resume=False,
    )
    try:
        state.discover()
        attempt = state.next_attempt()
        assert attempt is not None
        for endpoint, page_no in (
            (
                "/repos/owner/repo/actions/runs/1/attempts/1/logs",
                None,
            ),
            (
                "/repos/owner/repo/actions/runs/1/attempts/1/jobs",
                1,
            ),
        ):
            state.record_request(
                attempt,
                endpoint=endpoint,
                page_no=page_no,
                request_attempt=1,
                http_status=200,
                outcome="success",
                latency_ms=1,
            )
        forged = b"x" * 32
        with pytest.raises(ci.ArchiveError, match="invalid empty ZIP"):
            state.finish_attempt(
                attempt,
                status="empty",
                archive_source="forged-inline",
                archive_sha256=hashlib.sha256(forged).hexdigest(),
                archive_size=len(forged),
                archive_bytes=forged,
                jobs=[],
            )
    finally:
        state.close()


def test_oversized_empty_zip_is_rejected_before_evidence_read(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "oversized-empty.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for ordinal in range(12_000):
            archive.writestr(
                f"directory-{ordinal:05d}-{'x' * 64}/",
                b"",
            )
    assert (
        archive_path.stat().st_size
        > ci.MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES
    )
    with pytest.raises(
        ci.ArchiveError,
        match="exceeds its bounded raw-byte contract",
    ):
        ci._read_empty_archive_evidence(
            archive_path,
            expected_size=archive_path.stat().st_size,
        )


def test_empty_attempt_proof_requires_successful_logs_request_ledger(
    tmp_path: Path,
) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w"):
        pass
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    github = FakeGitHub(archive_buffer.getvalue())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_tokenizer(tmp_path / "tokenizer.json"),
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        with fetcher.state._connection:
            fetcher.state._connection.execute(
                "DELETE FROM request_ledger WHERE endpoint LIKE '%/logs'"
            )
        inventory_connection = sqlite3.connect(inventory)
        inventory_connection.row_factory = sqlite3.Row
        try:
            with pytest.raises(
                ReceiptFinalizationError,
                match=(
                    "lacks successful logs or authorized rescue evidence"
                ),
            ):
                exhaustive_coverage_proof(
                    inventory_connection,
                    fetcher.state._connection,
                    inventory_receipt=_coverage_inventory_receipt(
                        run_count=1,
                        attempt_keys=[("owner/repo", 1, 1)],
                    ),
                    require_discovery_eof=False,
                )
        finally:
            inventory_connection.close()
    finally:
        fetcher.close()


@pytest.mark.parametrize(
    ("log_status", "expected_status"),
    ((302, "empty"), (410, "terminal_410")),
)
def test_exhaustive_endpoint_proof_uses_renamed_canonical_repository(
    tmp_path: Path,
    log_status: int,
    expected_status: str,
) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w"):
        pass
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    metadata = _run_metadata(1)
    metadata["repository"] = {
        "full_name": "renamed/repository",
        "id": 1,
    }
    metadata["head_repository"] = {
        "full_name": "renamed/repository",
        "id": 1,
    }
    _replace_inventory_run(inventory, metadata)
    github = FakeGitHub(
        archive_buffer.getvalue(),
        log_status=log_status,
    )
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        inventory_connection = sqlite3.connect(inventory)
        inventory_connection.row_factory = sqlite3.Row
        try:
            proof = exhaustive_coverage_proof(
                inventory_connection,
                fetcher.state._connection,
                inventory_receipt=_coverage_inventory_receipt(
                    run_count=1,
                    attempt_keys=[("owner/repo", 1, 1)],
                ),
                require_discovery_eof=False,
            )
        finally:
            inventory_connection.close()
        assert proof["terminal_statuses"] == {expected_status: 1}
        assert any(
            "/repos/renamed/repository/actions/" in url
            for url in github.api_urls
        )
    finally:
        fetcher.close()


def test_empty_endpoint_proof_rejects_wrong_canonical_repository(
    tmp_path: Path,
) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w"):
        pass
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    metadata = _run_metadata(1)
    metadata["repository"] = {
        "full_name": "renamed/repository",
        "id": 1,
    }
    metadata["head_repository"] = {
        "full_name": "renamed/repository",
        "id": 1,
    }
    _replace_inventory_run(inventory, metadata)
    github = FakeGitHub(archive_buffer.getvalue())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_FROZEN_TOKENIZER,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        with fetcher.state._connection:
            fetcher.state._connection.execute(
                """
                UPDATE request_ledger
                SET endpoint=REPLACE(
                  endpoint,'renamed/repository','wrong/repository'
                )
                WHERE endpoint LIKE '%/logs'
                """
            )
        inventory_connection = sqlite3.connect(inventory)
        inventory_connection.row_factory = sqlite3.Row
        try:
            with pytest.raises(
                ReceiptFinalizationError,
                match=(
                    "lacks successful logs or authorized rescue evidence"
                ),
            ):
                exhaustive_coverage_proof(
                    inventory_connection,
                    fetcher.state._connection,
                    inventory_receipt=_coverage_inventory_receipt(
                        run_count=1,
                        attempt_keys=[("owner/repo", 1, 1)],
                    ),
                    require_discovery_eof=False,
                )
        finally:
            inventory_connection.close()
    finally:
        fetcher.close()


def test_fetch_state_process_lease_prevents_late_competing_owner(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state_path = tmp_path / "fetch.sqlite"
    tokenizer = ci.ExactTokenizer(_FROZEN_TOKENIZER)
    owner = ci.FetchState(
        state_path,
        inventory_path=inventory,
        content_store_path=tmp_path / "store",
        tokenizer=tokenizer,
        resume=False,
    )
    try:
        owner.discover()
        claimed = owner.next_attempt()
        assert claimed is not None
        with pytest.raises(ci.BindingError, match="live process lease"):
            ci.FetchState(
                state_path,
                inventory_path=inventory,
                content_store_path=tmp_path / "store",
                tokenizer=tokenizer,
                resume=True,
            )
        owner.finish_attempt(
            claimed,
            status="retry",
            error="lease test completed",
            retry=True,
        )
        assert owner._connection.execute(
            "SELECT status FROM attempts"
        ).fetchone()[0] == "retry"
    finally:
        owner.close()


def test_fetch_state_lease_refuses_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    state_path = tmp_path / "fetch.sqlite"
    victim = tmp_path / "victim.txt"
    victim.write_text("must remain intact", encoding="utf-8")
    state_path.with_name(f"{state_path.name}.lease").symlink_to(victim)

    with pytest.raises(ci.BindingError, match="lease path is unsafe"):
        ci.FetchState(
            state_path,
            inventory_path=inventory,
            content_store_path=tmp_path / "store",
            tokenizer=ci.ExactTokenizer(_FROZEN_TOKENIZER),
            resume=False,
        )

    assert victim.read_text(encoding="utf-8") == "must remain intact"
    assert not state_path.exists()


@pytest.mark.parametrize(
    ("diagnostic_partial", "once", "max_runs", "expected"),
    [
        (False, False, None, 1),
        (False, True, None, 1),
        (False, False, 1, 1),
        (True, False, None, 1),
        (True, True, None, 0),
        (True, False, 1, 0),
    ],
)
def test_incomplete_exhaustive_exit_is_nonzero_except_explicit_diagnostic_bound(
    diagnostic_partial: bool,
    once: bool,
    max_runs: int | None,
    expected: int,
) -> None:
    assert ci._incomplete_exhaustive_exit_code(
        diagnostic_partial=diagnostic_partial,
        once=once,
        max_runs=max_runs,
    ) == expected


def test_terminal_probe_cannot_hide_durable_member_occurrences(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    first = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=FakeGitHub(_zip_bytes()).request,
        archive_downloader=FakeGitHub(_zip_bytes()).download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        first.run(continuous=False, max_runs=1)
        assert first.state._connection.execute(
            "SELECT COUNT(*) FROM members"
        ).fetchone()[0] == 1
    finally:
        first.close()

    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """
            UPDATE attempts SET
              status='retry',
              error_class='SyntheticInterruptedAttempt',
              error_message='test terminal probe after durable member'
            """
        )

    terminal_github = FakeGitHub(_zip_bytes(), log_status=410)
    resumed = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=terminal_github.request,
        archive_downloader=terminal_github.download,
        target_unique_tokens=1_000_000,
        resume=True,
        sleeper=lambda _: None,
    )
    try:
        resumed.run(continuous=False, max_runs=1)
        row = resumed.state._connection.execute(
            """
            SELECT status,terminal_http_status,member_count,
                   chunk_count,occurrence_tokens,error_class
            FROM attempts
            """
        ).fetchone()
        assert tuple(row) == (
            "failed",
            410,
            0,
            0,
            0,
            "TerminalHTTP",
        )
        assert resumed.state._connection.execute(
            "SELECT COUNT(*) FROM members"
        ).fetchone()[0] == 1
        assert resumed.store.status()["counters"]["occurrence_count"] == 1
        assert not any("/jobs?" in url for url in terminal_github.api_urls)
    finally:
        resumed.close()


def test_progress_heartbeat_is_written_while_parser_is_still_running(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    github = FakeGitHub(_zip_bytes())
    parser_started = threading.Event()
    release_parser = threading.Event()
    progress_path = tmp_path / "progress.json"
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def slow_parser(
        raw: bytes,
        metadata: Mapping[str, object],
        *,
        max_chunk_chars: int,
    ) -> dict[str, object]:
        parser_started.set()
        if not release_parser.wait(5):
            raise AssertionError("test parser was not released")
        return _fake_parser(
            raw,
            metadata,
            max_chunk_chars=max_chunk_chars,
        )

    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_tokenizer(tmp_path / "tokenizer.json"),
        tokens=["api-secret"],
        progress_path=progress_path,
        receipt_path=tmp_path / "receipt.json",
        parser=slow_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )

    def run_fetcher() -> None:
        try:
            results.append(
                fetcher.run(
                    continuous=False,
                    max_runs=1,
                    poll_seconds=0.01,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    runner = threading.Thread(target=run_fetcher)
    runner.start()
    try:
        assert parser_started.wait(2)
        for _ in range(100):
            if progress_path.is_file():
                break
            release_parser.wait(0.01)
        assert progress_path.is_file()
        heartbeat = json.loads(progress_path.read_text())
        assert heartbeat["fetch"]["attempt_statuses"] == {"processing": 1}
        assert heartbeat["content_store"]["counters"][
            "exact_unique_payload_tokens"
        ] is None
    finally:
        release_parser.set()
        runner.join(5)
        fetcher.close()
    assert not runner.is_alive()
    assert not errors
    assert len(results) == 1


def test_full_attempt_streams_through_parser_tokenizer_and_cas_idempotently(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    github = FakeGitHub(_zip_bytes())
    paths = {
        "state": tmp_path / "fetch.sqlite",
        "store": tmp_path / "store",
        "progress": tmp_path / "progress.json",
        "receipt": tmp_path / "receipt.json",
    }
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=paths["state"],
        content_store_path=paths["store"],
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=paths["progress"],
        receipt_path=paths["receipt"],
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        progress = fetcher.run(continuous=False, max_runs=1)
        counters = progress["content_store"]["counters"]
        assert counters["occurrence_count"] == 1
        assert counters["unique_content_count"] == 1
        expected_tokens = len(
            load_cppmega_tokenizer(tokenizer).encode_batch(
                ["hello build world\n"]
            )[0]
        )
        assert counters["exact_unique_payload_tokens"] == expected_tokens
        assert progress["fetch"]["attempt_statuses"] == {"done": 1}
        assert progress["fetch"]["members"] == 1
        assert progress["fetch"]["chunks"] == 1
        assert github.signed_url is not None
        assert "super-secret" in github.signed_url
        assert all(
            headers["Authorization"] == "Bearer api-secret"
            for headers in github.api_headers
        )
        assert "api-secret" not in paths["progress"].read_text()
        assert not list((paths["state"].with_suffix(".work") / "tmp").iterdir())
    finally:
        fetcher.close()

    # Exact resume has no replay conflict and does not add an occurrence.
    resumed = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=paths["state"],
        content_store_path=paths["store"],
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=paths["progress"],
        receipt_path=paths["receipt"],
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        resume=True,
        sleeper=lambda _: None,
    )
    try:
        resumed.run(continuous=False)
        assert resumed.store.status()["counters"]["occurrence_count"] == 1
    finally:
        resumed.close()


def test_threshold_receipt_is_finalized_only_after_writers_close(
    tmp_path: Path,
) -> None:
    from scripts.ci_stream_receipts import finalize_fetch_receipts
    from scripts.export_ci_content_store_case5 import FrozenFetchState, FrozenStore

    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    github = FakeGitHub(_zip_bytes())
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    progress_path = tmp_path / "progress.json"
    fetch_receipt_path = tmp_path / "fetch-receipt.json"
    store_receipt_path = tmp_path / "store-receipt.json"
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=progress_path,
        receipt_path=fetch_receipt_path,
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1,
        sleeper=lambda _: None,
    )
    try:
        progress = fetcher.run(continuous=False, max_runs=1)
        assert progress["content_store"]["counters"][
            "exact_unique_payload_tokens"
        ] >= 1
        assert not fetch_receipt_path.exists()
        with pytest.raises(
            RuntimeError,
            match="live process lease|could not be frozen|not frozen",
        ):
            finalize_fetch_receipts(
                state_path=state_path,
                content_store_path=store_path,
                tokenizer_path=tokenizer,
                target_unique_tokens=1,
                fetch_receipt_path=fetch_receipt_path,
                store_receipt_path=store_receipt_path,
            )
    finally:
        fetcher.close()

    with pytest.raises(ValueError, match="paths must differ"):
        finalize_fetch_receipts(
            state_path=state_path,
            content_store_path=store_path,
            tokenizer_path=tokenizer,
            target_unique_tokens=1,
            fetch_receipt_path=fetch_receipt_path,
            store_receipt_path=fetch_receipt_path,
        )
    assert not fetch_receipt_path.exists()

    symlink_receipt = tmp_path / "fetch-receipt-link.json"
    symlink_receipt.symlink_to(fetch_receipt_path)
    with pytest.raises(RuntimeError, match="cannot be a symlink"):
        finalize_fetch_receipts(
            state_path=state_path,
            content_store_path=store_path,
            tokenizer_path=tokenizer,
            target_unique_tokens=1,
            fetch_receipt_path=symlink_receipt,
            store_receipt_path=store_receipt_path,
        )
    symlink_receipt.unlink()

    receipt = finalize_fetch_receipts(
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        target_unique_tokens=1,
        fetch_receipt_path=fetch_receipt_path,
        store_receipt_path=store_receipt_path,
    )
    with sqlite3.connect(state_path) as frozen_connection:
        assert frozen_connection.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0] == "delete"
    assert receipt["fetch_state"] == receipt["frozen_fetch_state"]["summary"]
    assert receipt["content_store_receipt"]["status"] == "complete"

    exact_tokenizer = ci.ExactTokenizer(tokenizer)
    with FrozenStore(store_path, store_receipt_path) as frozen_store:
        with FrozenFetchState(
            state_path,
            tokenizer=exact_tokenizer,
            store=frozen_store,
        ) as frozen_state:
            expected_binding = frozen_state.receipt_binding()
            expected_binding["artifact"]["path"] = str(state_path.resolve())
            assert receipt["frozen_fetch_state"] == expected_binding
            assert receipt["fetch_state"] == frozen_state.summary
            frozen_state.require_unchanged()
        frozen_store.require_unchanged()


def test_cli_finalizes_merge_compatible_receipts_after_resume(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    github = FakeGitHub(_zip_bytes())
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    progress_path = tmp_path / "progress.json"
    fetch_receipt_path = tmp_path / "fetch-receipt.json"
    store_receipt_path = tmp_path / "store-receipt.json"
    tokens_path = tmp_path / "tokens.txt"
    tokens_path.write_text("api-secret\n", encoding="utf-8")
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=progress_path,
        receipt_path=fetch_receipt_path,
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
    finally:
        fetcher.close()

    result = ci.main(
        [
            "--inventory",
            str(inventory),
            "--state",
            str(state_path),
            "--content-store",
            str(store_path),
            "--tokenizer",
            str(tokenizer),
            "--tokens",
            str(tokens_path),
            "--progress",
            str(progress_path),
            "--receipt",
            str(fetch_receipt_path),
            "--store-receipt",
            str(store_receipt_path),
            "--resume",
            "--once",
            "--workers",
            "1",
            "--parser-workers",
            "0",
            "--target-exact-unique-payload-tokens",
            "1",
        ]
    )
    assert result == 0
    receipt = json.loads(fetch_receipt_path.read_text(encoding="utf-8"))
    store_receipt = json.loads(store_receipt_path.read_text(encoding="utf-8"))
    assert receipt["content_store_receipt"] == store_receipt
    assert receipt["fetch_state"] == receipt["frozen_fetch_state"]["summary"]


def test_receipt_cli_bootstraps_repo_from_foreign_cwd(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    script = Path(ci.__file__).with_name("ci_stream_receipts.py")

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Finalize frozen store/fetch receipts" in completed.stdout


def test_receipt_refuses_cas_bound_to_retry_attempt(tmp_path: Path) -> None:
    from scripts.ci_stream_receipts import finalize_fetch_receipts

    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    fetch_receipt_path = tmp_path / "fetch-receipt.json"
    store_receipt_path = tmp_path / "store-receipt.json"
    github = FakeGitHub(_zip_bytes())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=fetch_receipt_path,
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
    finally:
        fetcher.close()

    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """
            UPDATE attempts SET
              status='retry',
              error_class='SyntheticInterruptedAttempt',
              error_message='test hidden CAS'
            """
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE").fetchone()

    with pytest.raises(RuntimeError, match="non-done attempt"):
        finalize_fetch_receipts(
            state_path=state_path,
            content_store_path=store_path,
            tokenizer_path=tokenizer,
            target_unique_tokens=1,
            fetch_receipt_path=fetch_receipt_path,
            store_receipt_path=store_receipt_path,
        )
    assert not fetch_receipt_path.exists()
    assert not store_receipt_path.exists()


def test_receipt_preflights_per_attempt_accounting_before_publication(
    tmp_path: Path,
) -> None:
    from scripts.ci_stream_receipts import finalize_fetch_receipts

    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    fetch_receipt_path = tmp_path / "fetch-receipt.json"
    store_receipt_path = tmp_path / "store-receipt.json"
    github = FakeGitHub(_zip_bytes())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=fetch_receipt_path,
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
    finally:
        fetcher.close()

    connection = sqlite3.connect(state_path)
    try:
        connection.execute(
            "UPDATE attempts SET member_count=0,chunk_count=0,"
            "occurrence_tokens=0"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        RuntimeError,
        match=(
            r"per-attempt member accounting is inconsistent: "
            r"owner/repo/1/1"
        ),
    ):
        finalize_fetch_receipts(
            state_path=state_path,
            content_store_path=store_path,
            tokenizer_path=tokenizer,
            target_unique_tokens=1,
            fetch_receipt_path=fetch_receipt_path,
            store_receipt_path=store_receipt_path,
        )
    assert not fetch_receipt_path.exists()
    assert not store_receipt_path.exists()


def test_receipt_joins_content_token_counts_to_member_conservation(
    tmp_path: Path,
) -> None:
    from scripts.ci_stream_receipts import finalize_fetch_receipts

    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    github = FakeGitHub(_zip_bytes())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "fetch-receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
    finally:
        fetcher.close()
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            "UPDATE members SET occurrence_tokens=occurrence_tokens+1"
        )
        connection.execute(
            "UPDATE attempts SET occurrence_tokens=occurrence_tokens+1"
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE").fetchone()

    with pytest.raises(
        ReceiptFinalizationError,
        match="member token conservation differs",
    ):
        finalize_fetch_receipts(
            state_path=state_path,
            content_store_path=store_path,
            tokenizer_path=tokenizer,
            target_unique_tokens=1,
            fetch_receipt_path=tmp_path / "fetch-receipt.json",
            store_receipt_path=tmp_path / "store-receipt.json",
        )


def test_receipt_rejects_consistently_forged_token_counters(
    tmp_path: Path,
) -> None:
    from scripts.ci_stream_receipts import finalize_fetch_receipts

    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    fetch_receipt_path = tmp_path / "fetch-receipt.json"
    store_receipt_path = tmp_path / "store-receipt.json"
    github = FakeGitHub(_zip_bytes())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=fetch_receipt_path,
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
    finally:
        fetcher.close()

    with sqlite3.connect(store_path / "index.sqlite3") as connection:
        actual_token_count = int(
            connection.execute(
                "SELECT token_count FROM contents"
            ).fetchone()[0]
        )
        forged_token_count = actual_token_count + 21
        connection.execute(
            "UPDATE contents SET token_count=?",
            (forged_token_count,),
        )
        connection.execute(
            "UPDATE token_sequences SET token_count=?",
            (forged_token_count,),
        )
        connection.execute(
            "UPDATE stats SET exact_unique_payload_tokens=?",
            (forged_token_count,),
        )
    connection.close()
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            "UPDATE members SET occurrence_tokens=?",
            (forged_token_count,),
        )
        connection.execute(
            "UPDATE attempts SET occurrence_tokens=?",
            (forged_token_count,),
        )
    connection.close()

    with pytest.raises(
        ReceiptFinalizationError,
        match="token metadata differs from exact retokenization",
    ):
        finalize_fetch_receipts(
            state_path=state_path,
            content_store_path=store_path,
            tokenizer_path=tokenizer,
            target_unique_tokens=1,
            fetch_receipt_path=fetch_receipt_path,
            store_receipt_path=store_receipt_path,
        )
    assert not fetch_receipt_path.exists()
    assert not store_receipt_path.exists()


def test_receipt_reconstructs_content_and_token_sequence_dedup(
    tmp_path: Path,
) -> None:
    from scripts.ci_stream_receipts import finalize_fetch_receipts

    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer_path = _tokenizer(tmp_path / "tokenizer.json")
    first = "hello build world\n"
    whitespace_variant = "hello  build world\n"
    exact = ci.ExactTokenizer(tokenizer_path)
    token_sequences = exact.encode_batch([first, whitespace_variant])
    assert token_sequences[0] == token_sequences[1]
    token_count = len(token_sequences[0])
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    fetch_receipt_path = tmp_path / "fetch-receipt.json"
    store_receipt_path = tmp_path / "store-receipt.json"
    github = FakeGitHub(
        _zip_bytes(
            {
                "0_build.txt": first.encode(),
                "1_build.txt": whitespace_variant.encode(),
                "2_build.txt": first.encode(),
            }
        )
    )
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer_path,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=fetch_receipt_path,
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=token_count,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
    finally:
        fetcher.close()

    receipt = finalize_fetch_receipts(
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer_path,
        target_unique_tokens=token_count,
        fetch_receipt_path=fetch_receipt_path,
        store_receipt_path=store_receipt_path,
    )
    counters = receipt["content_store_receipt"]["counters"]
    assert counters["unique_content_count"] == 2
    assert counters["tokenized_unique_content_count"] == 2
    assert counters["unique_token_sequence_count"] == 1
    assert counters["exact_unique_payload_tokens"] == token_count
    assert counters["occurrence_count"] == 3
    assert receipt["fetch_state"]["occurrence_tokens"] == 3 * token_count


def test_retry_validates_and_skips_a_committed_member(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    github = FakeGitHub(_zip_bytes())
    state_path = tmp_path / "fetch.sqlite"
    store_path = tmp_path / "store"
    parser_calls: list[str] = []

    def counting_parser(
        raw: bytes,
        metadata: Mapping[str, object],
        *,
        max_chunk_chars: int,
    ) -> dict[str, object]:
        parser_calls.append(str(metadata["archive_member"]))
        return _fake_parser(
            raw,
            metadata,
            max_chunk_chars=max_chunk_chars,
        )

    first = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=counting_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        first.run(continuous=False, max_runs=1)
        assert parser_calls == ["0_build.txt"]
        attempt_row = first.state._connection.execute(
            "SELECT * FROM attempts"
        ).fetchone()
        assert attempt_row is not None
        synthetic_attempt = first.state._decode_attempt(attempt_row)
        first.state.store_member(
            synthetic_attempt,
            archive_member="stale-from-earlier-snapshot.txt",
            job_key="synthetic:stale-from-earlier-snapshot.txt",
            raw_sha256="1" * 64,
            raw_size=17,
            canonical_sha256="2" * 64,
            dedup_sha256="3" * 64,
            sidecar={"schema": "synthetic-retry-sidecar-v1"},
            chunk_count=2,
            occurrence_tokens=17,
        )
    finally:
        first.close()

    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """
            UPDATE attempts SET
              status='retry',
              error_class='SyntheticInterruptedAttempt',
              error_message='test replay'
            """
        )

    resumed = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=state_path,
        content_store_path=store_path,
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=counting_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        resume=True,
        sleeper=lambda _: None,
    )
    try:
        resumed.run(continuous=False, max_runs=1)
        assert parser_calls == ["0_build.txt"]
        assert resumed.store.status()["counters"]["occurrence_count"] == 1
        row = resumed.state._connection.execute(
            """
            SELECT status,tries,member_count,chunk_count,occurrence_tokens
            FROM attempts
            """
        ).fetchone()
        actual = resumed.state._connection.execute(
            """
            SELECT COUNT(*),COALESCE(SUM(chunk_count),0),
                   COALESCE(SUM(occurrence_tokens),0)
            FROM members
            """
        ).fetchone()
        assert tuple(row[:2]) == ("done", 2)
        assert tuple(row[2:]) == tuple(actual)
    finally:
        resumed.close()


def test_renamed_repository_uses_canonical_api_route_and_preserves_alias(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    with sqlite3.connect(inventory) as connection:
        row = connection.execute(
            "SELECT metadata_blob FROM runs WHERE run_id=1"
        ).fetchone()
        metadata = json.loads(zlib.decompress(row[0]))
        metadata["repository"] = {
            "full_name": "new-owner/repo",
            "id": 123,
        }
        metadata["head_repository"] = {
            "full_name": "contributor/repo",
            "id": 456,
        }
        raw = json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        connection.execute(
            """
            UPDATE runs
            SET metadata_blob=?, metadata_sha256=?
            WHERE run_id=1
            """,
            (zlib.compress(raw, 6), hashlib.sha256(raw).hexdigest()),
        )

    github = FakeGitHub(_zip_bytes())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=_tokenizer(tmp_path / "tokenizer.json"),
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        assert github.api_urls
        assert all("/repos/new-owner/repo/" in url for url in github.api_urls)
        occurrence = next(fetcher.store.iter_occurrences())
        provenance = occurrence["provenance"]
        assert provenance["schema"] == "cppmega_ci_chunk_occurrence_v3"
        assert provenance["repository"] == "new-owner/repo"
        assert provenance["repository_requested"] == "owner/repo"
        assert provenance["repository_id"] == 123
        assert provenance["source_repository"] == "contributor/repo"
        assert provenance["source_repository_id"] == 456
        assert provenance["repository_scope_key"] == "owner/repo"
        assert provenance["run_metadata_evidence"] == {
            "exact_attempt_match": True,
            "source": "inventory-run-list",
            "source_attempt": 1,
            "sha256": provenance["run_metadata_evidence"]["sha256"],
            "inventory_seed_attempt": 1,
            "inventory_seed_metadata_sha256": (
                provenance["run_metadata_evidence"]["sha256"]
            ),
        }
        assert provenance["workflow"]["path"] == (
            ".github/workflows/ci.yml"
        )
        assert provenance["workflow"]["run_number"] == 1
        assert provenance["workflow"]["status"] == "completed"
        assert provenance["workflow"]["conclusion"] == "success"
        assert provenance["workflow"]["created_at"] == (
            "2026-04-27T16:01:00Z"
        )
        assert provenance["workflow"]["head_commit"]["id"] == (
            f"{1:040x}"
        )
    finally:
        fetcher.close()


def test_rescue_terminal_410_is_diagnostic_and_cannot_terminalize(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    rescue = tmp_path / "rescue"
    rescue.mkdir()
    body = b'{"message":"Gone","status":410}'
    artifact = rescue / "owner__repo--1--attempt-1.http410.json"
    artifact.write_bytes(body)
    manifest = (
        "repo\trun_id\tattempt\tcreated_at\tstatus\tbytes\tsha256\tfinished_at\n"
        f"owner/repo\t1\t1\t2026-04-27T16:01:00Z\thttp410\t{len(body)}\t"
        f"{hashlib.sha256(body).hexdigest()}\t2026-07-26T16:00:00Z\n"
    )
    (rescue / "manifest.tsv").write_text(manifest)

    github = FakeGitHub(_zip_bytes())
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        rescue_path=rescue,
        parser=_fake_parser,
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        sleeper=lambda _: None,
    )
    try:
        fetcher.run(continuous=False, max_runs=1)
        row = fetcher.state._connection.execute(
            """
            SELECT status,terminal_http_status,terminal_body_sha256
            FROM attempts
            """
        ).fetchone()
        assert tuple(row) == ("retry", None, None)
        # Rescue markers are retained evidence, but they are not a GitHub
        # endpoint/access receipt and therefore cannot close production work.
        assert github.api_headers == []
        assert github.signed_url is None
    finally:
        fetcher.close()


def test_rolling_scheduler_refills_a_slot_before_a_slow_attempt_finishes() -> None:
    release_slow = threading.Event()
    third_started = threading.Event()
    attempts = iter([1, 2, 3])

    class State:
        def discover(self) -> None:
            return None

        def next_attempt(self, *, retry_only: bool = False) -> int | None:
            assert not retry_only
            return next(attempts, None)

    class Store:
        @staticmethod
        def status() -> dict[str, object]:
            return {"counters": {"exact_unique_payload_tokens": 0}}

    fetcher = object.__new__(ci.CIStreamFetcher)
    fetcher.state = State()
    fetcher.store = Store()
    fetcher.target_unique_tokens = 1_000_000
    fetcher.sleeper = lambda _: None
    fetcher.write_progress = lambda: {"status": "ok"}

    def process(attempt: int) -> None:
        if attempt == 1:
            assert release_slow.wait(2), (
                "scheduler waited for the whole fixed wave"
            )
        elif attempt == 3:
            third_started.set()
            release_slow.set()

    fetcher.process_attempt = process
    result = fetcher.run(
        continuous=False,
        max_runs=3,
        workers=2,
    )

    assert result == {"status": "ok"}
    assert third_started.is_set()


def test_scheduler_drains_retry_work_after_token_target_is_met() -> None:
    class State:
        def __init__(self) -> None:
            self.calls: list[bool] = []
            self.pending = ["pending"]
            self.retries = ["retry"]

        def discover(self) -> None:
            return None

        def next_attempt(self, *, retry_only: bool = False) -> str | None:
            self.calls.append(retry_only)
            queue = self.retries if retry_only else self.pending
            return queue.pop(0) if queue else None

    class Store:
        def __init__(self) -> None:
            self.tokens = 0

        def status(self) -> dict[str, object]:
            return {
                "counters": {
                    "exact_unique_payload_tokens": self.tokens,
                }
            }

    fetcher = object.__new__(ci.CIStreamFetcher)
    fetcher.state = State()
    fetcher.store = Store()
    fetcher.target_unique_tokens = 1
    fetcher.sleeper = lambda _: None
    fetcher.write_progress = lambda: {"status": "ok"}
    processed: list[str] = []

    def process(attempt: str) -> None:
        processed.append(attempt)
        fetcher.store.tokens = 1

    fetcher.process_attempt = process
    result = fetcher.run(continuous=False, workers=1)

    assert result == {"status": "ok"}
    assert processed == ["pending", "retry"]
    assert fetcher.state.calls == [False, True, True]


def test_exhaustive_scheduler_ignores_met_target_and_drains_pending() -> None:
    class State:
        def __init__(self) -> None:
            self.pending = ["first", "second"]
            self.retry_only_calls: list[bool] = []
            self.discovered = False

        def discover(
            self,
            *,
            exhaustive_inventory: object,
        ) -> int:
            assert exhaustive_inventory == "verified-inventory"
            self.discovered = True
            return 2

        def next_attempt(self, *, retry_only: bool = False) -> str | None:
            self.retry_only_calls.append(retry_only)
            return self.pending.pop(0) if self.pending else None

        def exhaustive_discovery_summary(self) -> dict[str, object]:
            return {"discovery_eof": True}

    class Store:
        @staticmethod
        def status() -> dict[str, object]:
            return {
                "counters": {
                    "exact_unique_payload_tokens": 1_000_000,
                }
            }

    fetcher = object.__new__(ci.CIStreamFetcher)
    fetcher.state = State()
    fetcher.store = Store()
    fetcher.target_unique_tokens = 1
    fetcher.completion_mode = (
        ci.COMPLETION_MODE_INVENTORY_EXHAUSTIVE
    )
    fetcher.exhaustive_inventory = "verified-inventory"
    fetcher.sleeper = lambda _: None
    fetcher.write_progress = lambda: {"status": "ok"}
    processed: list[str] = []
    fetcher.process_attempt = processed.append

    result = fetcher.run(continuous=True, workers=1)

    assert result == {"status": "ok"}
    assert processed == ["first", "second"]
    assert fetcher.state.discovered is True
    assert fetcher.state.retry_only_calls == [False, False, False]


def test_exhaustive_completion_gate_refuses_failed_attempts() -> None:
    class State:
        statuses = {"done": 1, "failed": 1}

        @staticmethod
        def exhaustive_discovery_summary() -> dict[str, object]:
            return {"discovery_eof": True}

        def summary(self) -> dict[str, object]:
            return {"attempt_statuses": self.statuses}

    class Binding:
        expected_attempt_count = 2

    fetcher = object.__new__(ci.CIStreamFetcher)
    fetcher.state = State()
    fetcher.exhaustive_inventory = Binding()

    assert fetcher.exhaustive_completion_ready() is False
    fetcher.state.statuses = {"done": 1, "terminal_410": 1}
    assert fetcher.exhaustive_completion_ready() is True


def test_spawned_parser_worker_emits_full_training_sidecars(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path / "inventory.sqlite", 1)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    github = FakeGitHub(
        _zip_bytes(
            {
                "0_build.txt": (
                    b"[command]ninja -C build app src/a.cpp -o build/app\n"
                )
            }
        )
    )
    fetcher = ci.CIStreamFetcher(
        inventory_path=inventory,
        state_path=tmp_path / "fetch.sqlite",
        content_store_path=tmp_path / "store",
        tokenizer_path=tokenizer,
        tokens=["api-secret"],
        progress_path=tmp_path / "progress.json",
        receipt_path=tmp_path / "receipt.json",
        requester=github.request,
        archive_downloader=github.download,
        target_unique_tokens=1_000_000,
        parser_workers=2,
        sleeper=lambda _: None,
    )
    try:
        progress = fetcher.run(
            continuous=False,
            max_runs=1,
            workers=1,
        )
        assert progress["fetch"]["attempt_statuses"] == {"done": 1}
        occurrence = next(fetcher.store.iter_occurrences())
        training = occurrence["provenance"]["chunk"]["training_sidecars"]
        assert training["schema"] == "cppmega_ci_chunk_training_sidecars_v2"
        assert training["build_actions"]
        assert training["edges"]
    finally:
        fetcher.close()
