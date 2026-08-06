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
