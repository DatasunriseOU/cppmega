from __future__ import annotations

import hashlib
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
        "status": "completed",
        "conclusion": "success",
        "workflow_id": 77,
        "name": "CI",
        "event": "push",
        "head_branch": "main",
        "head_sha": f"{run_id:040x}"[-40:],
        "actor": {"login": "builder"},
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
    def __init__(self, archive: bytes):
        self.archive = archive
        self.api_headers: list[dict[str, str]] = []
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
        assert training["schema"] == "cppmega_ci_chunk_training_sidecars_v1"
        assert training["build_actions"]
        assert training["edges"]
    finally:
        fetcher.close()
