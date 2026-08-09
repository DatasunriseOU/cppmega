"""ProcessPool spawn requires a live getcwd(); CLEANUP can delete sibling workdirs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.clang_indexer import index_project as ip


def test_ensure_durable_process_cwd_keeps_existing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    got = ip.ensure_durable_process_cwd()
    assert got == tmp_path.resolve()
    assert Path.cwd() == tmp_path.resolve()


def test_ensure_durable_process_cwd_recovers_when_cwd_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vanished = tmp_path / "vanished"
    vanished.mkdir()
    monkeypatch.chdir(vanished)
    # Delete the directory out from under the process (macOS/Linux allow this).
    os.rmdir(vanished)
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    monkeypatch.setenv("CPPMEGA_PROCESS_CWD", str(durable_root))
    got = ip.ensure_durable_process_cwd()
    assert got == durable_root.resolve()
    assert Path.cwd() == durable_root.resolve()
    # getcwd must succeed after recovery (what ProcessPool spawn needs)
    assert os.getcwd() == str(durable_root.resolve())
