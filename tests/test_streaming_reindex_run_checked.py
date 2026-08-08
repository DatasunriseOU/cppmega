from __future__ import annotations

import sys

import pytest


def test_run_checked_times_out_fail_loud(tmp_path) -> None:
    from scripts import streaming_reindex

    log_path = tmp_path / "sleep.log"
    with pytest.raises(streaming_reindex.RepoFailure) as caught:
        streaming_reindex.run_checked(
            "repo",
            "index_project",
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            log_path=log_path,
            timeout=1,
        )

    assert caught.value.repo == "repo"
    assert caught.value.stage == "index_project"
    assert "timed out after 1s" in caught.value.detail
    assert "started" in log_path.read_text(encoding="utf-8")


def test_run_checked_stall_watchdog_fail_loud(tmp_path) -> None:
    from scripts import streaming_reindex

    log_path = tmp_path / "stall.log"
    with pytest.raises(streaming_reindex.RepoFailure) as caught:
        streaming_reindex.run_checked(
            "repo",
            "index_project",
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            log_path=log_path,
            timeout=20,
            stall_timeout=1,
        )

    assert caught.value.repo == "repo"
    assert caught.value.stage == "index_project"
    assert "stalled after 1s without log progress" in caught.value.detail
    assert "started" in log_path.read_text(encoding="utf-8")


def test_run_checked_stall_watchdog_stops_cpu_spin_without_log_progress(
    tmp_path,
) -> None:
    from scripts import streaming_reindex

    log_path = tmp_path / "cpu-progress.log"
    with pytest.raises(streaming_reindex.RepoFailure) as caught:
        streaming_reindex.run_checked(
            "repo",
            "index_project",
            [
                sys.executable,
                "-c",
                (
                    "import time\n"
                    "print('started', flush=True)\n"
                    "deadline = time.time() + 30.0\n"
                    "x = 0\n"
                    "while time.time() < deadline:\n"
                    "    x += 1\n"
                ),
            ],
            log_path=log_path,
            timeout=10,
            stall_timeout=1,
        )

    assert "stalled after 1s without log progress" in caught.value.detail
    assert "started" in log_path.read_text(encoding="utf-8")
def test_run_checked_stall_watchdog_ignores_duplicate_heartbeats(tmp_path) -> None:
    from scripts import streaming_reindex

    log_path = tmp_path / "duplicate-heartbeat.log"
    heartbeat = (
        "Parse pool heartbeat: completed_batches=396 submitted_batches=400 "
        "pending_batches=4 running_batches=4"
    )
    with pytest.raises(streaming_reindex.RepoFailure) as caught:
        streaming_reindex.run_checked(
            "repo",
            "index_project",
            [
                sys.executable,
                "-c",
                (
                    "import time\n"
                    f"heartbeat = {heartbeat!r}\n"
                    "deadline = time.time() + 30.0\n"
                    "while time.time() < deadline:\n"
                    "    print(heartbeat, flush=True)\n"
                    "    time.sleep(0.1)\n"
                ),
            ],
            log_path=log_path,
            timeout=10,
            stall_timeout=1,
        )

    assert "stalled after 1s without log progress" in caught.value.detail
    assert log_path.read_text(encoding="utf-8").count(heartbeat) > 1


def test_run_checked_stall_watchdog_accepts_advancing_heartbeats(tmp_path) -> None:
    from scripts import streaming_reindex

    log_path = tmp_path / "advancing-heartbeat.log"
    streaming_reindex.run_checked(
        "repo",
        "index_project",
        [
            sys.executable,
            "-c",
            (
                "import time\n"
                "for completed in range(6):\n"
                "    print(\n"
                "        'Parse pool heartbeat: '\n"
                "        f'completed_batches={completed} submitted_batches=6 '\n"
                "        f'pending_batches={6 - completed} running_batches=1',\n"
                "        flush=True,\n"
                "    )\n"
                "    time.sleep(0.4)\n"
            ),
        ],
        log_path=log_path,
        timeout=10,
        stall_timeout=1,
    )

    output = log_path.read_text(encoding="utf-8")
    assert "completed_batches=0" in output
    assert "completed_batches=5" in output
