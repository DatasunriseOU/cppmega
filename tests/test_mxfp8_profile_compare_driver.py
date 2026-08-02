"""P087 regression: profile configs run as isolated, fail-fast subprocesses."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DRIVER = _ROOT / "runs" / "mxfp8_profile_compare" / "run_compare.py"


def _run_driver(
    tmp_path: Path,
    *arguments: str,
    fail_on: str = "",
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    calls_log = tmp_path / "calls.log"
    runner = tmp_path / "fake_runner.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$$|${RUN_ID}|${LOG}|$*" >> "${CALLS_LOG}"\n'
        'if [[ -n "${FAIL_ON:-}" && "$*" == *"${FAIL_ON}"* ]]; then exit 7; fi\n'
    )
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    env["CALLS_LOG"] = str(calls_log)
    env["FAIL_ON"] = fail_on
    completed = subprocess.run(
        [
            sys.executable,
            str(_DRIVER),
            "--runner",
            str(runner),
            "--out-dir",
            str(tmp_path / "out"),
            *arguments,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (
        [line.split("|") for line in calls_log.read_text().splitlines()]
        if calls_log.exists()
        else []
    )
    return completed, calls


def test_full_matrix_preserves_five_configs_and_process_isolation(tmp_path: Path) -> None:
    completed, calls = _run_driver(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert len(calls) == 5
    assert len({call[0] for call in calls}) == 5
    assert [call[1].split("_20", 1)[0] for call in calls] == [
        "profile_bf16",
        "profile_mxfp8_gemm_ready",
        "profile_mxfp8_legacy",
        "profile_bf16_b16",
        "profile_mxfp8_gemm_ready_b16",
    ]
    assert len({call[2] for call in calls}) == 5


def test_failure_stops_before_the_next_config(tmp_path: Path) -> None:
    completed, calls = _run_driver(
        tmp_path,
        fail_on="--mxfp8-linear-kernel-contract legacy",
    )

    assert completed.returncode == 7
    assert len(calls) == 3
    assert calls[-1][1].startswith("profile_mxfp8_legacy_")
    assert "skipped remaining configs" in completed.stderr


def test_batch16_suite_runs_only_batch16_pair(tmp_path: Path) -> None:
    completed, calls = _run_driver(tmp_path, "--suite", "b16")

    assert completed.returncode == 0, completed.stderr
    assert len(calls) == 2
    assert calls[0][1].startswith("profile_bf16_b16_")
    assert calls[1][1].startswith("profile_mxfp8_gemm_ready_b16_")
    assert all("--micro-batch-size 16" in call[3] for call in calls)


def test_unknown_config_is_rejected_before_launch(tmp_path: Path) -> None:
    completed, calls = _run_driver(tmp_path, "--configs", "nope")

    assert completed.returncode == 2
    assert calls == []
