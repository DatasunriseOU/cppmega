from __future__ import annotations

import fcntl
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.corpus_pipeline_watchdog import (
    WATCHDOG_SCHEMA,
    WatchdogError,
    load_config,
    matching_processes,
    run_watchdog,
)


def _ledger(path: Path, *, status: int | None, message: str = "") -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE request_ledger (
                id INTEGER PRIMARY KEY,
                requested_at TEXT NOT NULL,
                http_status INTEGER,
                error_class TEXT,
                error_message TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO request_ledger(id, requested_at, http_status, error_class, error_message)
            VALUES(1, '2026-08-04T00:00:00Z', ?, '', ?)
            """,
            (status, message),
        )
        connection.commit()
    finally:
        connection.close()


def _config(tmp_path: Path, *, terminal: Path | None = None) -> dict[str, object]:
    ledger = tmp_path / "fetch_state.sqlite3"
    _ledger(ledger, status=429, message="Too Many Requests")
    lane: dict[str, object] = {
        "name": "ci-old",
        "process_contains": ["scripts/ci_stream_fetch.py", "--state", str(ledger)],
        "transient_429": {"kind": "sqlite_request_ledger", "path": str(ledger)},
        "resume": {
            "argv": ["/usr/bin/true"],
            "cwd": str(tmp_path),
            "stdout": str(tmp_path / "resume.log"),
        },
        "resume_guard": {"unlocked_files": [str(tmp_path / "fetch_state.sqlite3.lease")]},
    }
    if terminal is not None:
        lane["terminal_receipt"] = {
            "path": str(terminal),
            "status_values": ["verified", "complete"],
        }
    (tmp_path / "fetch_state.sqlite3.lease").touch()
    return {
        "schema": WATCHDOG_SCHEMA,
        "state_path": str(tmp_path / "state.json"),
        "report_path": str(tmp_path / "report.json"),
        "lanes": [lane],
    }


def test_matching_processes_requires_every_fragment() -> None:
    lines = [
        "12 python scripts/ci_stream_fetch.py --state /tmp/a.sqlite3",
        "13 python scripts/ci_stream_fetch.py --state /tmp/b.sqlite3",
        "14 tmux new-session python scripts/ci_stream_fetch.py --state /tmp/a.sqlite3",
    ]
    assert matching_processes(
        lines,
        ["ci_stream_fetch.py", "/tmp/a.sqlite3"],
        ["tmux new-session"],
    ) == [lines[0]]


def test_running_lane_never_restarts_even_when_latest_ledger_is_429(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[tuple[object, object, object]] = []

    result = run_watchdog(
        config,
        process_runner=lambda: [
            f"12 python scripts/ci_stream_fetch.py --state {tmp_path / 'fetch_state.sqlite3'}"
        ],
        launcher=lambda argv, cwd, stdout: calls.append((argv, cwd, stdout)) or 91,
        now=lambda: 100,
    )

    assert result["lanes"] == [{"name": "ci-old", "matched_processes": 1, "state": "running"}]
    assert calls == []


def test_stopped_lane_resumes_once_for_new_durable_429(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[tuple[list[str], Path | None, Path | None]] = []

    launcher = lambda argv, cwd, stdout: calls.append((list(argv), cwd, stdout)) or 456
    first = run_watchdog(
        config,
        process_runner=lambda: [],
        launcher=launcher,
        now=lambda: 100,
    )
    second = run_watchdog(
        config,
        process_runner=lambda: [],
        launcher=launcher,
        now=lambda: 200,
    )

    assert first["lanes"][0]["state"] == "resumed_after_429"
    assert first["lanes"][0]["pid"] == 456
    assert second["lanes"][0]["state"] == "stopped_429_already_handled"
    assert calls == [(["/usr/bin/true"], tmp_path, tmp_path / "resume.log")]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert len(state["lanes"]["ci-old"]["handled_evidence"]) == 1


def test_terminal_receipt_suppresses_restart(tmp_path: Path) -> None:
    terminal = tmp_path / "completion.json"
    terminal.write_text(json.dumps({"status": "verified"}), encoding="utf-8")
    config = _config(tmp_path, terminal=terminal)

    result = run_watchdog(
        config,
        process_runner=lambda: [],
        launcher=lambda *_args: pytest.fail("must not resume a terminal lane"),
        now=lambda: 100,
    )

    assert result["lanes"] == [
        {"name": "ci-old", "matched_processes": 0, "state": "terminal", "terminal_status": "verified"}
    ]


def test_non_429_latest_request_never_restarts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    ledger = tmp_path / "fetch_state.sqlite3"
    connection = sqlite3.connect(ledger)
    try:
        connection.execute(
            """
            INSERT INTO request_ledger(id, requested_at, http_status, error_class, error_message)
            VALUES(2, '2026-08-04T00:01:00Z', 500, '', 'server error')
            """
        )
        connection.commit()
    finally:
        connection.close()

    result = run_watchdog(
        config,
        process_runner=lambda: [],
        launcher=lambda *_args: pytest.fail("must not resume non-429 failure"),
        now=lambda: 100,
    )

    assert result["lanes"][0]["state"] == "stopped_without_confirmed_429"


def test_config_requires_evidence_and_resume_as_a_pair(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lane = config["lanes"][0]
    assert isinstance(lane, dict)
    lane.pop("resume")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(WatchdogError, match="together"):
        load_config(path)


def test_held_lease_prevents_resume_even_for_new_429(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lease = tmp_path / "fetch_state.sqlite3.lease"
    with lease.open("r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_watchdog(
            config,
            process_runner=lambda: [],
            launcher=lambda *_args: pytest.fail("held lease must prevent launch"),
            now=lambda: 100,
        )

    assert result["lanes"][0]["state"] == "stopped_429_resume_guard_held"


def test_missing_lease_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "fetch_state.sqlite3.lease").unlink()

    result = run_watchdog(
        config,
        process_runner=lambda: [],
        launcher=lambda *_args: pytest.fail("missing lease must prevent launch"),
        now=lambda: 100,
    )

    assert result["lanes"][0]["state"] == "stopped_429_resume_guard_invalid"


def test_dry_run_does_not_consume_recovery_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_watchdog(
        config,
        process_runner=lambda: [],
        launcher=lambda *_args: pytest.fail("dry run must not launch"),
        now=lambda: 100,
        dry_run=True,
    )

    assert result["lanes"][0]["state"] == "would_resume_after_429"
    assert (tmp_path / "state.json").exists()
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["lanes"]["ci-old"]["handled_evidence"] == []
