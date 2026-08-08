"""Regression: tokenizer resolution must not require a live process cwd."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.nanochat_data.token_budget import resolve_tokenizer_path


def test_resolve_tokenizer_path_uses_explicit_path_when_cwd_is_gone(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")

    def _dead_cwd() -> Path:
        raise FileNotFoundError("No such file or directory")

    with patch.object(Path, "cwd", side_effect=_dead_cwd):
        resolved = resolve_tokenizer_path(str(tokenizer))
    assert Path(resolved).resolve() == tokenizer.resolve()


def test_resolve_tokenizer_path_falls_back_to_repo_tokenizer_when_cwd_is_gone() -> None:
    repo_tokenizer = (
        Path(__file__).resolve().parents[1]
        / "cppmega"
        / "tokenizer"
        / "tokenizer.json"
    )
    if not repo_tokenizer.is_file():
        pytest.skip("repo tokenizer.json is not present in this checkout")

    def _dead_cwd() -> Path:
        raise OSError(2, "No such file or directory")

    with patch.object(Path, "cwd", side_effect=_dead_cwd):
        resolved = resolve_tokenizer_path(None)
    assert Path(resolved).resolve() == repo_tokenizer.resolve()
