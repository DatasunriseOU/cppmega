from __future__ import annotations

import hashlib
import http.client
import io
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Mapping
import zipfile
import zlib

import pytest

from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
from scripts import ci_stream_fetch as ci


_FROZEN_TOKENIZER = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "tokenizer_v2"
    / "tokenizer.json"
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
        "cppmega_exact_ci_training_tokenizer_v2"
    )
    assert exact.contract["whitespace_normalizer"].endswith(
        "normalize_cpp_whitespace_with_offsets"
    )
    assert exact.contract["tokenizer_contract_sha256"]
    assert all(
        ci.hash_token_sequence(actual) == ci.hash_token_sequence(wanted)
        for actual, wanted in zip(exact.encode_batch(payloads), expected)
    )

    invalid = tmp_path / "not-cppmega-tokenizer.json"
    invalid.write_text('{"version":"1.0"}')
    with pytest.raises(ci.FetchError, match="frozen cppmega training contract"):
        ci.ExactTokenizer(invalid)


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
            "SELECT status,tries FROM attempts"
        ).fetchone()
        assert tuple(row) == ("done", 2)
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


def test_rescue_terminal_410_is_proven_and_never_downloaded(tmp_path: Path) -> None:
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
        assert tuple(row) == (
            "terminal_410",
            410,
            hashlib.sha256(body).hexdigest(),
        )
        # A durable rescue proof avoids spending a GitHub API request on a
        # log archive that is already known to be irretrievably expired.
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

        def next_attempt(self) -> int | None:
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
