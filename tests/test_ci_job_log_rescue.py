from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Mapping
import urllib.error
import urllib.parse
import zipfile
import zlib

import pytest

from scripts import ci_job_log_rescue as rescue
from scripts import ci_stream_fetch as fetch


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _job(job_id: int, name: str, *, run_id: int = 17) -> dict[str, object]:
    return {
        "id": job_id,
        "name": name,
        "run_id": run_id,
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
        "runner_name": "GitHub Actions 1",
        "labels": ["ubuntu-24.04"],
        "steps": [{"name": "compile", "conclusion": "success"}],
    }


def _state(
    path: Path,
    jobs: list[Mapping[str, object]],
    *,
    run_id: int = 17,
    canonical_repo: str = "owner/repo",
    error_class: str = "ArchiveError",
    error_message: str = (
        "signed archive transport retries exhausted before EOF: IncompleteRead"
    ),
) -> Path:
    metadata = {
        "id": run_id,
        "run_attempt": 1,
        "created_at": "2026-07-25T10:00:00Z",
        "updated_at": "2026-07-25T10:30:00Z",
        "status": "completed",
        "conclusion": "failure",
        "head_sha": "a" * 40,
        "repository": {"full_name": canonical_repo, "id": 99},
        "head_repository": {"full_name": "owner/repo", "id": 99},
    }
    metadata_raw = _canonical(metadata)
    jobs_raw = _canonical(jobs)
    with sqlite3.connect(path) as connection:
        connection.executescript(fetch._STATE_SCHEMA)
        connection.execute(
            "INSERT INTO settings(key,value) VALUES ('schema',?)",
            (fetch.SCHEMA_VERSION,),
        )
        connection.execute(
            """
            INSERT INTO attempts(
              repo,run_id,attempt,created_at,
              run_metadata_sha256,run_metadata_raw_size,run_metadata_zlib,
              run_metadata_source,run_metadata_source_attempt,
              run_metadata_exact,inventory_seed_attempt,
              inventory_seed_metadata_sha256,status,tries,
              archive_source,archive_sha256,archive_size,
              jobs_sha256,jobs_raw_size,jobs_zlib,
              error_class,error_message,discovered_at,updated_at
            ) VALUES (
              'owner/repo',?,1,'2026-07-25T10:00:00Z',
              ?,?,?,'github-workflow-run-attempt-api',1,1,1,?,
              'failed',4,'github-signed-url',?,1234,?,?,?,
              ?,?,'2026-07-25T10:31:00Z','2026-07-25T10:35:00Z'
            )
            """,
            (
                run_id,
                hashlib.sha256(metadata_raw).hexdigest(),
                len(metadata_raw),
                sqlite3.Binary(zlib.compress(metadata_raw, 6)),
                hashlib.sha256(metadata_raw).hexdigest(),
                "b" * 64,
                hashlib.sha256(jobs_raw).hexdigest(),
                len(jobs_raw),
                sqlite3.Binary(zlib.compress(jobs_raw, 6)),
                error_class,
                error_message,
            ),
        )
        endpoint = f"/repos/{canonical_repo}/actions/runs/{run_id}/attempts/1/jobs"
        for page in range(1, max(1, (len(jobs) + 99) // 100) + 1):
            connection.execute(
                """
                INSERT INTO request_ledger(
                  requested_at,repo,run_id,attempt,endpoint,page_no,
                  request_attempt,http_status,outcome,latency_ms
                ) VALUES (
                  '2026-07-25T10:31:00Z','owner/repo',?,1,?,?,
                  1,200,'success',7
                )
                """,
                (run_id, endpoint, page),
            )
    return path


class _Response:
    def __init__(
        self,
        status: int,
        body: bytes = b"",
        headers: Mapping[str, str] | None = None,
    ):
        self.status = status
        self.headers = dict(headers or {})
        self._body = io.BytesIO(body)
        self.closed = False

    def read(self, amount: int = -1) -> bytes:
        return self._body.read(amount)

    def close(self) -> None:
        self.closed = True


class _IncompleteResponse(_Response):
    def __init__(self, prefix: bytes):
        super().__init__(200, prefix, {"Content-Length": str(len(prefix) + 9)})
        self._read_count = 0

    def read(self, amount: int = -1) -> bytes:
        self._read_count += 1
        if self._read_count == 1:
            return super().read(amount)
        raise urllib.error.URLError("connection ended")


class _ScriptedOpener:
    """Explicit HTTP seam; no process-global monkeypatching."""

    def __init__(
        self,
        api_actions: Mapping[int, list[object]],
        *,
        signed_bodies: Mapping[str, bytes | tuple[int, bytes]] | None = None,
    ):
        self._api_actions = {
            job_id: list(actions) for job_id, actions in api_actions.items()
        }
        self._signed_bodies = dict(signed_bodies or {})
        self.api_calls: dict[int, int] = defaultdict(int)
        self.signed_calls: list[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def inline(body: bytes) -> tuple[str, bytes]:
        return ("inline", body)

    @staticmethod
    def terminal(status: int, body: bytes = b"gone") -> tuple[str, int, bytes]:
        return ("terminal", status, body)

    @staticmethod
    def redirect(url: str) -> tuple[str, str]:
        return ("redirect", url)

    def __call__(
        self,
        request: Any,
        *,
        timeout: float,
    ) -> _Response:
        assert timeout > 0
        url = request.full_url
        parsed = urllib.parse.urlsplit(url)
        if parsed.hostname == "api.github.com":
            assert request.get_header("Authorization") == "Bearer api-secret"
            match = re.search(r"/actions/jobs/([0-9]+)/logs$", parsed.path)
            assert match is not None
            job_id = int(match.group(1))
            with self._lock:
                call = self.api_calls[job_id]
                self.api_calls[job_id] += 1
                actions = self._api_actions[job_id]
                action = actions[min(call, len(actions) - 1)]
            if isinstance(action, BaseException):
                raise action
            kind = action[0]
            if kind == "inline":
                body = action[1]
                return _Response(
                    200,
                    body,
                    {"Content-Length": str(len(body))},
                )
            if kind == "terminal":
                return _Response(action[1], action[2])
            if kind == "redirect":
                return _Response(302, headers={"Location": action[1]})
            if kind == "incomplete":
                return _IncompleteResponse(action[1])
            raise AssertionError(action)
        assert request.get_header("Authorization") is None
        self.signed_calls.append(url)
        if url not in self._signed_bodies:
            raise AssertionError(f"unexpected signed URL {url}")
        signed_result = self._signed_bodies[url]
        if isinstance(signed_result, tuple):
            status, body = signed_result
            return _Response(status, body)
        body = signed_result
        return _Response(
            200,
            body,
            {"Content-Length": str(len(body))},
        )


def _worker(
    tmp_path: Path,
    state: Path,
    opener: _ScriptedOpener,
    **kwargs: object,
) -> rescue.JobLogRescueWorker:
    return rescue.JobLogRescueWorker(
        state_path=state,
        work_dir=tmp_path / "rescue-work",
        rescue_spool=tmp_path / "rescue-spool",
        tokens=["api-secret"],
        workers=int(kwargs.pop("workers", 2)),
        max_attempts=int(kwargs.pop("max_attempts", 2)),
        max_job_bytes=int(kwargs.pop("max_job_bytes", 1024)),
        max_total_bytes=int(kwargs.pop("max_total_bytes", 4096)),
        max_zip_bytes=int(kwargs.pop("max_zip_bytes", 4096)),
        opener=opener,
        sleeper=lambda _seconds: None,
        **kwargs,
    )


def _attempt_row(state: Path) -> sqlite3.Row:
    connection = sqlite3.connect(state)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT * FROM attempts WHERE repo='owner/repo'"
        ).fetchone()
        assert row is not None
        return row
    finally:
        connection.close()


def _artifact_bytes(root: Path) -> bytes:
    payload = bytearray()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        payload.extend(path.read_bytes())
    return bytes(payload)


def _guard_oversized_run_metadata(
    evidence: rescue.FetchStateEvidence,
) -> list[int]:
    largest_materialized_blob = [0]

    def guarded_row_factory(
        cursor: sqlite3.Cursor,
        values: tuple[object, ...],
    ) -> sqlite3.Row:
        largest_materialized_blob[0] = max(
            (
                len(value)
                for value in values
                if isinstance(value, bytes)
            ),
            default=largest_materialized_blob[0],
        )
        if (
            largest_materialized_blob[0]
            > rescue.MAX_RUN_METADATA_COMPRESSED_BYTES
        ):
            raise AssertionError("oversized fetch-state BLOB was materialized")
        return sqlite3.Row(cursor, values)

    evidence.connection.row_factory = guarded_row_factory
    return largest_materialized_blob


def test_load_attempt_rechecks_bounds_after_second_connection_mutation(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "fetch.sqlite", [_job(101, "build")])
    evidence = rescue.FetchStateEvidence(state)
    try:
        largest = _guard_oversized_run_metadata(evidence)
        with sqlite3.connect(state) as attacker:
            attacker.execute(
                """
                UPDATE attempts SET run_metadata_zlib=zeroblob(?)
                WHERE repo='owner/repo' AND run_id=17 AND attempt=1
                """,
                (rescue.MAX_RUN_METADATA_COMPRESSED_BYTES + 1,),
            )
        with pytest.raises(
            rescue.StateBindingError,
            match="attempt evidence exceeds its versioned SQLite byte bounds",
        ):
            evidence.load_attempt(("owner/repo", 17, 1), explicit=True)
        assert largest[0] <= rescue.MAX_RUN_METADATA_COMPRESSED_BYTES
    finally:
        evidence.close()


def test_commit_rescue_preflights_mutated_row_before_publish(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "fetch.sqlite", [_job(101, "build")])
    evidence = rescue.FetchStateEvidence(state)
    try:
        source = evidence.load_attempt(("owner/repo", 17, 1), explicit=True)
        largest = _guard_oversized_run_metadata(evidence)
        with sqlite3.connect(state) as attacker:
            attacker.execute(
                """
                UPDATE attempts SET run_metadata_zlib=zeroblob(?)
                WHERE repo='owner/repo' AND run_id=17 AND attempt=1
                """,
                (rescue.MAX_RUN_METADATA_COMPRESSED_BYTES + 1,),
            )
        publish_called = False

        def publish() -> None:
            nonlocal publish_called
            publish_called = True

        with pytest.raises(
            rescue.StateBindingError,
            match="attempt evidence exceeds its versioned SQLite byte bounds",
        ):
            evidence.commit_rescue(
                source,
                receipt_sha256="a" * 64,
                synthetic_zip_sha256="b" * 64,
                synthetic_zip_bytes=1,
                publish=publish,
            )
        assert publish_called is False
        assert largest[0] <= rescue.MAX_RUN_METADATA_COMPRESSED_BYTES
    finally:
        evidence.close()


def test_fetch_state_evidence_rejects_raw_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    target = _state(
        tmp_path / "target.sqlite",
        [_job(101, "build")],
    )
    target_before = target.read_bytes()
    alias = tmp_path / "alias.sqlite"
    alias.symlink_to(target)

    with pytest.raises(
        rescue.StateBindingError,
        match="symlink|does not exist safely",
    ):
        rescue.FetchStateEvidence(alias)

    assert alias.is_symlink()
    assert target.read_bytes() == target_before


def test_fetch_state_evidence_refuses_live_finalization_lease(
    tmp_path: Path,
) -> None:
    from scripts import ci_stream_receipts as receipts

    state = _state(
        tmp_path / "fetch.sqlite",
        [_job(101, "build")],
    )
    state_before = state.read_bytes()
    state_stat_before = state.stat()
    descriptor = receipts._acquire_fetch_state_finalization_lease(state)
    try:
        with pytest.raises(
            rescue.StateBindingError,
            match="live process lease",
        ):
            rescue.FetchStateEvidence(state)
    finally:
        receipts._release_fetch_state_finalization_lease(descriptor)

    state_stat_after = state.stat()
    assert state.read_bytes() == state_before
    assert state_stat_after.st_size == state_stat_before.st_size
    assert state_stat_after.st_mtime_ns == state_stat_before.st_mtime_ns


def test_fetch_state_evidence_holds_lease_until_close(
    tmp_path: Path,
) -> None:
    from scripts import ci_stream_receipts as receipts

    state = _state(
        tmp_path / "fetch.sqlite",
        [_job(101, "build")],
    )
    evidence = rescue.FetchStateEvidence(state)
    try:
        with pytest.raises(
            receipts.ReceiptFinalizationError,
            match="live process lease",
        ):
            receipts._acquire_fetch_state_finalization_lease(state)
    finally:
        evidence.close()

    descriptor = receipts._acquire_fetch_state_finalization_lease(state)
    receipts._release_fetch_state_finalization_lease(descriptor)


def test_fetch_state_path_swap_fails_before_publish_and_preserves_target(
    tmp_path: Path,
) -> None:
    state = _state(
        tmp_path / "fetch.sqlite",
        [_job(101, "build")],
    )
    target = _state(
        tmp_path / "target.sqlite",
        [_job(202, "other")],
        run_id=18,
    )
    target_before = target.read_bytes()
    evidence = rescue.FetchStateEvidence(state)
    original = tmp_path / "fetch.original.sqlite"
    try:
        source = evidence.load_attempt(
            ("owner/repo", 17, 1),
            explicit=True,
        )
        state.replace(original)
        state.symlink_to(target)
        publish_called = False

        def publish() -> None:
            nonlocal publish_called
            publish_called = True

        with pytest.raises(
            rescue.StateBindingError,
            match="identity changed|symlink|path is unsafe",
        ):
            evidence.commit_rescue(
                source,
                receipt_sha256="a" * 64,
                synthetic_zip_sha256="b" * 64,
                synthetic_zip_bytes=1,
                publish=publish,
            )
        assert publish_called is False
        assert target.read_bytes() == target_before
    finally:
        evidence.close()
        if state.is_symlink():
            state.unlink()
        if original.exists():
            original.replace(state)


def test_full_200_log_and_proven_404_complete_and_requeue(
    tmp_path: Path,
) -> None:
    jobs = [_job(101, "build"), _job(102, "skipped")]
    state = _state(tmp_path / "fetch.sqlite", jobs)
    opener = _ScriptedOpener(
        {
            101: [
                ("incomplete", b"partial"),
                _ScriptedOpener.inline(b"complete build log\n"),
            ],
            102: [
                _ScriptedOpener.terminal(
                    404,
                    b"api-secret and signed response details",
                )
            ],
        }
    )
    worker = _worker(tmp_path, state, opener)
    try:
        result = worker.run_once(target=("OWNER/REPO", 17, 1))
    finally:
        worker.close()

    assert result["failed_attempts"] == 0
    assert result["results"][0]["status"] == "complete"
    assert opener.api_calls[101] == 2
    row = _attempt_row(state)
    assert (row["status"], row["tries"]) == ("retry", 0)
    with sqlite3.connect(state) as connection:
        audit = connection.execute(
            """
            SELECT endpoint,outcome,error_class,error_message
            FROM request_ledger ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
    assert audit is not None
    assert audit[:3] == (
        "operator/job_rescue",
        "operator/job_rescue",
        "JobRescueReceipt",
    )
    audit_evidence = json.loads(audit[3])
    assert audit[3] == _canonical(audit_evidence).decode("utf-8")
    assert set(audit_evidence) == {
        "schema",
        "producer_lineage",
        "receipt_sha256",
        "source_row_sha256",
        "source_state_sha256",
        "synthetic_zip",
        "jobs_ledger_sha256",
    }
    assert audit_evidence["schema"] == rescue.LEDGER_EVIDENCE_SCHEMA
    assert audit_evidence["producer_lineage"] == rescue._producer_lineage(
        rescue._producer_binding()
    )
    for field in (
        "receipt_sha256",
        "source_row_sha256",
        "source_state_sha256",
        "jobs_ledger_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", audit_evidence[field])

    spool = tmp_path / "rescue-spool"
    archive_path = spool / "owner__repo--17--attempt-1.zip"
    with sqlite3.connect(state) as connection:
        metadata_row = connection.execute(
            """
            SELECT run_metadata_sha256,run_metadata_zlib
            FROM attempts
            WHERE repo='owner/repo' AND run_id=17 AND attempt=1
            """
        ).fetchone()
    assert metadata_row is not None
    run_metadata = json.loads(zlib.decompress(metadata_row[1]))
    located = fetch.RescueSpool(spool).locate(
        fetch.Attempt(
            repo="owner/repo",
            run_id=17,
            attempt=1,
            created_at="2026-07-25T10:00:00Z",
            run_metadata=run_metadata,
            run_metadata_sha256=metadata_row[0],
            run_metadata_source="github-workflow-run-attempt-api",
            run_metadata_source_attempt=1,
            run_metadata_exact=True,
            inventory_seed_attempt=1,
            inventory_seed_metadata_sha256="a" * 64,
        ),
        audit=audit_evidence,
    )
    assert isinstance(located, fetch.ArchiveSource)
    assert located.path == archive_path
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        assert [info.filename for info in infos] == ["0_101.txt"]
        assert infos[0].date_time == (1980, 1, 1, 0, 0, 0)
        assert infos[0].compress_type == zipfile.ZIP_DEFLATED
        assert archive.read(infos[0]) == b"complete build log\n"

    resolved_path = spool / "owner__repo--17--attempt-1.resolved_jobs.jsonl"
    records = [
        json.loads(line)
        for line in resolved_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["outcome"] for record in records] == [
        "log",
        "terminal_404",
    ]
    assert resolved_path.read_bytes() == b"".join(
        _canonical(record) + b"\n" for record in records
    )
    receipt = json.loads(
        (spool / "owner__repo--17--attempt-1.receipt.json").read_bytes()
    )
    assert receipt["coverage"] == {
        "expected_jobs": 2,
        "full_logs": 1,
        "resolved_jobs": 2,
        "terminal_404": 1,
        "terminal_410": 0,
        "uncompressed_log_bytes": 19,
        "unresolved_jobs": 0,
        "zip_members": 1,
    }
    durable_members = {
        "0_101.txt": (
            hashlib.sha256(b"complete build log\n").hexdigest(),
            len(b"complete build log\n"),
        )
    }
    validation_kwargs = {
        "repo": "owner/repo",
        "canonical_repo": "owner/repo",
        "run_id": 17,
        "attempt": 1,
        "created_at": "2026-07-25T10:00:00Z",
        "run_metadata_sha256": str(row["run_metadata_sha256"]),
        "run_metadata_raw_size": int(row["run_metadata_raw_size"]),
        "archive_sha256": located.raw_sha256,
        "archive_size": located.raw_size,
        "jobs_sha256": str(row["jobs_sha256"]),
        "jobs_raw_size": int(row["jobs_raw_size"]),
        "job_count": len(jobs),
        "member_count": 1,
        "member_uncompressed_bytes": len(b"complete build log\n"),
        "jobs": jobs,
    }
    fetch._validate_rescue_archive_provenance(
        located.provenance,
        **validation_kwargs,
        durable_members=durable_members,
    )

    forged_provenance = json.loads(_canonical(located.provenance))
    forged_record = forged_provenance["resolved_jobs"]["records"][0]
    forged_record["log"]["sha256"] = "f" * 64
    with pytest.raises(
        fetch.BindingError,
        match="artifact differs from embedded records",
    ):
        fetch._validate_rescue_archive_provenance(
            forged_provenance,
            **validation_kwargs,
        )
    forged_resolved_raw = b"".join(
        _canonical(record) + b"\n"
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
    forged_receipt_raw = _canonical(forged_receipt) + b"\n"
    forged_provenance["job_rescue_receipt"].update(
        {
            "bytes": len(forged_receipt_raw),
            "sha256": hashlib.sha256(forged_receipt_raw).hexdigest(),
        }
    )
    forged_audit = dict(audit_evidence)
    forged_audit.update(
        {
            "receipt_sha256": hashlib.sha256(
                forged_receipt_raw
            ).hexdigest(),
            "source_state_sha256": hashlib.sha256(
                _canonical(forged_receipt["source_state"])
            ).hexdigest(),
            "jobs_ledger_sha256": forged_receipt["source_state"][
                "jobs_ledger_sha256"
            ],
        }
    )
    forged_receipt_sha256, forged_source_row_sha256 = (
        fetch._validate_rescue_archive_provenance(
            forged_provenance,
            **validation_kwargs,
        )
    )
    fetch._validate_job_rescue_operator_audit(
        forged_audit,
        receipt_sha256=forged_receipt_sha256,
        source_row_sha256=forged_source_row_sha256,
        source_state=forged_receipt["source_state"],
        archive_sha256=located.raw_sha256,
        archive_size=located.raw_size,
    )
    with pytest.raises(
        fetch.BindingError,
        match="resolved log differs from current durable member",
    ):
        fetch._validate_rescue_archive_provenance(
            forged_provenance,
            **validation_kwargs,
            durable_members=durable_members,
        )
    assert b"api-secret" not in _artifact_bytes(tmp_path / "rescue-work")
    assert b"api-secret" not in _artifact_bytes(spool)


def test_transient_unresolved_is_nonzero_and_does_not_requeue(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "fetch.sqlite", [_job(101, "build")])
    opener = _ScriptedOpener(
        {101: [urllib.error.URLError("temporary network failure")]}
    )
    worker = _worker(tmp_path, state, opener)
    try:
        result = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()

    assert result["failed_attempts"] == 1
    attempt_result = result["results"][0]
    assert attempt_result["status"] == "unresolved"
    assert attempt_result["unresolved_jobs"] == 1
    row = _attempt_row(state)
    assert (row["status"], row["tries"]) == ("failed", 4)
    assert not (tmp_path / "rescue-spool" / "owner__repo--17--attempt-1.zip").exists()
    with sqlite3.connect(state) as connection:
        assert (
            connection.execute(
                """
            SELECT COUNT(*) FROM request_ledger
            WHERE outcome='operator/job_rescue'
            """
            ).fetchone()[0]
            == 0
        )
    assert opener.api_calls[101] == 2


def test_resume_is_deterministic_and_completed_replay_is_idempotent(
    tmp_path: Path,
) -> None:
    jobs = [_job(101, "build"), _job(102, "test")]
    state = _state(tmp_path / "fetch.sqlite", jobs)
    first = _ScriptedOpener(
        {
            101: [_ScriptedOpener.inline(b"build log\n")],
            102: [urllib.error.URLError("temporary")],
        }
    )
    worker = _worker(tmp_path, state, first)
    try:
        first_result = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()
    assert first_result["failed_attempts"] == 1
    assert first.api_calls == {101: 1, 102: 2}

    second = _ScriptedOpener(
        {
            101: [AssertionError("resolved job was refetched")],
            102: [_ScriptedOpener.inline(b"test log\n")],
        }
    )
    worker = _worker(tmp_path, state, second)
    try:
        second_result = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()
    assert second_result["failed_attempts"] == 0
    assert second.api_calls[101] == 0
    assert second.api_calls[102] == 1
    archive_path = tmp_path / "rescue-spool" / "owner__repo--17--attempt-1.zip"
    archive_bytes = archive_path.read_bytes()
    consumed = tmp_path / "rescue-spool" / "consumed"
    consumed.mkdir()
    consumed_archive = consumed / archive_path.name
    archive_path.replace(consumed_archive)

    replay = _ScriptedOpener(
        {
            101: [AssertionError("completed job was refetched")],
            102: [AssertionError("completed job was refetched")],
        }
    )
    worker = _worker(tmp_path, state, replay)
    try:
        replay_result = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()
    assert replay_result["failed_attempts"] == 0
    assert replay_result["results"][0]["idempotent_replay"] is True
    assert replay.api_calls == {}
    assert consumed_archive.read_bytes() == archive_bytes
    with sqlite3.connect(state) as connection:
        assert (
            connection.execute(
                """
            SELECT COUNT(*) FROM request_ledger
            WHERE outcome='operator/job_rescue'
            """
            ).fetchone()[0]
            == 1
        )

    other = tmp_path / "other"
    other.mkdir()
    other_state = _state(other / "fetch.sqlite", jobs)
    other_opener = _ScriptedOpener(
        {
            101: [_ScriptedOpener.inline(b"build log\n")],
            102: [_ScriptedOpener.inline(b"test log\n")],
        }
    )
    other_worker = _worker(other, other_state, other_opener)
    try:
        other_result = other_worker.run_once(target=("owner/repo", 17, 1))
    finally:
        other_worker.close()
    assert other_result["failed_attempts"] == 0
    assert (
        other / "rescue-spool" / "owner__repo--17--attempt-1.zip"
    ).read_bytes() == archive_bytes


def test_legacy_job_rescue_audit_requires_explicit_append_only_migration(
    tmp_path: Path,
) -> None:
    jobs = [_job(101, "build")]
    state = _state(tmp_path / "fetch.sqlite", jobs)
    opener = _ScriptedOpener(
        {101: [_ScriptedOpener.inline(b"legacy build log\n")]}
    )
    worker = _worker(tmp_path, state, opener)
    try:
        first = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()
    assert first["failed_attempts"] == 0

    spool = tmp_path / "rescue-spool"
    base = "owner__repo--17--attempt-1"
    receipt_path = spool / f"{base}.receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    original_binding = dict(receipt.pop("producer_binding"))
    legacy_receipt_raw = _canonical(receipt) + b"\n"
    receipt_path.write_bytes(legacy_receipt_raw)
    legacy_receipt_sha = hashlib.sha256(legacy_receipt_raw).hexdigest()
    source_row_sha = str(receipt["source_state"]["attempt_row_sha256"])
    legacy_audit = (
        f"receipt_sha256={legacy_receipt_sha} "
        f"source_row_sha256={source_row_sha}"
    )
    with sqlite3.connect(state) as connection:
        connection.execute(
            """
            UPDATE request_ledger SET error_message=?
            WHERE endpoint='operator/job_rescue'
              AND outcome='operator/job_rescue'
              AND error_class='JobRescueReceipt'
            """,
            (legacy_audit,),
        )

    denied = _worker(
        tmp_path,
        state,
        _ScriptedOpener(
            {101: [AssertionError("completed rescue must not refetch")]}
        ),
    )
    try:
        denied_result = denied.run_once(target=("owner/repo", 17, 1))
    finally:
        denied.close()
    assert denied_result["failed_attempts"] == 1
    assert "explicit producer upgrade authorization" in str(
        denied_result["results"][0]["error_message"]
    )

    prior_sha = "6ac4ac14" + "0" * 56
    reason = "authorize audited migration of the legacy rescue producer"
    migrated = _worker(
        tmp_path,
        state,
        _ScriptedOpener(
            {101: [AssertionError("completed rescue must not refetch")]}
        ),
        allow_producer_upgrade_from_sha256=prior_sha,
        producer_upgrade_reason=reason,
    )
    try:
        migrated_result = migrated.run_once(
            target=("owner/repo", 17, 1)
        )
    finally:
        migrated.close()
    assert migrated_result["failed_attempts"] == 0
    assert migrated_result["results"][0]["idempotent_replay"] is True
    assert receipt_path.read_bytes() == legacy_receipt_raw

    with sqlite3.connect(state) as connection:
        rows = connection.execute(
            """
            SELECT error_message FROM request_ledger
            WHERE endpoint='operator/job_rescue'
              AND outcome='operator/job_rescue'
              AND error_class='JobRescueReceipt'
            ORDER BY id
            """
        ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == legacy_audit
    migrated_audit = json.loads(rows[1][0])
    lineage = migrated_audit["producer_lineage"]
    assert lineage["origin"] == {
        "script_sha256": prior_sha,
        "semantic_contract_sha256": (
            fetch.JOB_RESCUE_LEGACY_SEMANTIC_CONTRACT_SHA256
        ),
    }
    assert lineage["current"] == rescue._producer_binding()
    assert lineage["upgrades"][-1]["reason"] == reason
    assert lineage["upgrades"][-1]["from"]["script_sha256"] == prior_sha
    assert (
        lineage["upgrades"][-1]["to"]["script_sha256"]
        == rescue._producer_binding()["script_sha256"]
    )
    assert original_binding != lineage["origin"]

    zip_path = spool / f"{base}.zip"
    fetch._validate_job_rescue_operator_audit(
        migrated_audit,
        receipt_sha256=legacy_receipt_sha,
        source_row_sha256=source_row_sha,
        source_state=receipt["source_state"],
        archive_sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        archive_size=zip_path.stat().st_size,
        receipt_producer_binding=None,
    )


def test_row_changed_race_fails_before_spool_publish(tmp_path: Path) -> None:
    state = _state(tmp_path / "fetch.sqlite", [_job(101, "build")])
    opener = _ScriptedOpener({101: [_ScriptedOpener.inline(b"complete log\n")]})

    def change_row(_source: rescue.SourceAttempt) -> None:
        with sqlite3.connect(state) as connection:
            connection.execute(
                """
                UPDATE attempts SET updated_at='2026-07-25T11:00:00Z'
                WHERE repo='owner/repo' AND run_id=17 AND attempt=1
                """
            )

    worker = _worker(
        tmp_path,
        state,
        opener,
        before_publish=change_row,
    )
    try:
        result = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()

    assert result["failed_attempts"] == 1
    assert result["results"][0]["error_class"] == "StateBindingError"
    assert _attempt_row(state)["status"] == "failed"
    assert not (tmp_path / "rescue-spool" / "owner__repo--17--attempt-1.zip").exists()


def test_unsafe_redirect_is_unresolved_and_never_followed(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "fetch.sqlite", [_job(101, "build")])
    opener = _ScriptedOpener(
        {101: [_ScriptedOpener.redirect("http://127.0.0.1/internal?sig=do-not-follow")]}
    )
    worker = _worker(tmp_path, state, opener)
    try:
        result = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()

    attempt_result = result["results"][0]
    assert result["failed_attempts"] == 1
    assert attempt_result["unresolved"][0]["error_class"] == ("UnsafeRedirectError")
    assert opener.signed_calls == []
    assert _attempt_row(state)["status"] == "failed"
    assert b"do-not-follow" not in _artifact_bytes(tmp_path / "rescue-work")


def test_job_log_byte_limit_is_fail_closed(tmp_path: Path) -> None:
    state = _state(tmp_path / "fetch.sqlite", [_job(101, "build")])
    opener = _ScriptedOpener({101: [_ScriptedOpener.inline(b"12345")]})
    worker = _worker(tmp_path, state, opener, max_job_bytes=4)
    try:
        result = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()

    assert result["failed_attempts"] == 1
    unresolved = result["results"][0]["unresolved"][0]
    assert unresolved["error_class"] == "ByteLimitError"
    assert _attempt_row(state)["status"] == "failed"
    assert not list((tmp_path / "rescue-spool").glob("*.zip"))


def test_safe_signed_redirect_streams_without_forwarding_authorization(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "fetch.sqlite", [_job(101, "build")])
    signed_url = (
        "https://productionresultssa0.blob.core.windows.net/"
        "actions-results/job.log?se=2026-07-26T10%3A00Z&sig=signed-secret"
    )
    opener = _ScriptedOpener(
        {101: [_ScriptedOpener.redirect(signed_url)]},
        signed_bodies={signed_url: b"signed complete log\n"},
    )
    worker = _worker(tmp_path, state, opener)
    try:
        result = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()

    assert result["failed_attempts"] == 0
    assert opener.signed_calls == [signed_url]
    assert b"signed-secret" not in _artifact_bytes(tmp_path / "rescue-work")
    assert b"signed-secret" not in _artifact_bytes(tmp_path / "rescue-spool")


def test_safe_signed_redirect_proven_404_is_terminal(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "fetch.sqlite", [_job(101, "expired")])
    signed_url = (
        "https://productionresultssa0.blob.core.windows.net/"
        "actions-results/job.log?se=2026-07-26T10%3A00Z&sig=signed-secret"
    )
    opener = _ScriptedOpener(
        {101: [_ScriptedOpener.redirect(signed_url)]},
        signed_bodies={signed_url: (404, b"api-secret expired")},
    )
    worker = _worker(tmp_path, state, opener)
    try:
        result = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()

    assert result["failed_attempts"] == 0
    record = result["results"][0]["receipt"]
    assert record["coverage"]["terminal_404"] == 1
    assert record["coverage"]["unresolved_jobs"] == 0
    resolved_path = (
        tmp_path / "rescue-spool" / "owner__repo--17--attempt-1.resolved_jobs.jsonl"
    )
    [resolved] = [
        json.loads(line)
        for line in resolved_path.read_text(encoding="utf-8").splitlines()
    ]
    assert resolved["api_http_status"] == 302
    assert resolved["signed_http_status"] == 404
    assert resolved["terminal"]["http_status"] == 404
    assert b"api-secret" not in _artifact_bytes(tmp_path / "rescue-work")
    assert b"signed-secret" not in _artifact_bytes(tmp_path / "rescue-spool")


def test_scan_narrowly_accepts_signed_archive_api_403(tmp_path: Path) -> None:
    state = _state(
        tmp_path / "fetch.sqlite",
        [_job(101, "build")],
        error_class="APIError",
        error_message="signed archive URL returned HTTP 403",
    )
    opener = _ScriptedOpener({101: [_ScriptedOpener.inline(b"rescued\n")]})
    worker = _worker(tmp_path, state, opener)
    try:
        result = worker.run_once()
    finally:
        worker.close()
    assert result["scanned_attempts"] == 1
    assert result["failed_attempts"] == 0

    generic_root = tmp_path / "generic"
    generic_root.mkdir()
    generic_state = _state(
        generic_root / "fetch.sqlite",
        [_job(101, "build")],
        error_class="APIError",
        error_message="GitHub HTTP 403 for /actions/jobs",
    )
    generic_opener = _ScriptedOpener(
        {101: [AssertionError("generic API 403 must not auto-scan")]}
    )
    generic_worker = _worker(generic_root, generic_state, generic_opener)
    try:
        generic_result = generic_worker.run_once()
    finally:
        generic_worker.close()
    assert generic_result["scanned_attempts"] == 0
    assert generic_opener.api_calls == {}


def test_repository_alias_uses_exact_metadata_canonical_api_route(
    tmp_path: Path,
) -> None:
    state = _state(
        tmp_path / "fetch.sqlite",
        [_job(101, "build")],
        canonical_repo="new-owner/repo",
    )
    opener = _ScriptedOpener(
        {101: [_ScriptedOpener.inline(b"renamed repository log\n")]}
    )
    worker = _worker(tmp_path, state, opener)
    try:
        result = worker.run_once(target=("owner/repo", 17, 1))
    finally:
        worker.close()

    assert result["failed_attempts"] == 0
    receipt = result["results"][0]["receipt"]
    assert receipt["source_state"]["repo"] == "owner/repo"
    assert receipt["source_state"]["canonical_repo"] == "new-owner/repo"
    assert (tmp_path / "rescue-spool" / "owner__repo--17--attempt-1.zip").is_file()
