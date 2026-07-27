from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_indexer():
    try:
        from tools.clang_indexer import index_project

        index_project._configure_libclang()
    except Exception as exc:  # pragma: no cover - environment without libclang
        pytest.skip(f"libclang unavailable: {exc}")
    return index_project


def test_parse_file_batch_fails_loud_with_file_and_cause(tmp_path: Path) -> None:
    index_project = _load_indexer()
    source = tmp_path / "broken.cpp"
    source.write_text("int main() { return 0; }\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as raised:
        index_project._parse_file_batch(
            (
                [str(source)],
                {},
                ["-x", "definitely-not-a-language"],
                str(tmp_path),
                "fixture/parse-failure",
            )
        )

    message = str(raised.value)
    assert str(source) in message
    assert "TranslationUnitLoadError" in message
    assert "libclang parse failed" in message


def test_sequential_project_parse_fails_loud_instead_of_publishing(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "broken.cpp"
    source.write_text("int main() { return 0; }\n", encoding="utf-8")
    (tmp_path / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(tmp_path),
                    "file": str(source),
                    "arguments": [
                        "clang++",
                        "-x",
                        "definitely-not-a-language",
                        str(source),
                    ],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as raised:
        index_project.process_project(
            str(tmp_path),
            enriched=True,
            project_id="fixture/parse-failure",
        )

    message = str(raised.value)
    assert str(source) in message
    assert "TranslationUnitLoadError" in message
    assert "libclang parse failed" in message
