from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path
from threading import Timer

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


def test_cp1252_source_round_trips_without_clang_token_utf8_decode(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "legacy.cpp"
    raw_type = b"struct /* compiler\x92s type */ LegacyType { int value; };\r\n"
    raw_function = (
        b"int /* compiler\x92s invalid candidate */ "
        b"legacy_answer() { return 42; }\r\n"
    )
    raw_source = raw_type + raw_function
    source.write_bytes(raw_source)

    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
            str(tmp_path),
            "fixture/cp1252-source",
        )
    )

    assert parsed_count == 1
    function = next(
        item for item in payload["functions"]
        if item["name"] == "legacy_answer"
    )
    type_definition = next(
        item for item in payload["typedefs"]
        if item["name"] == "LegacyType"
    )
    assert function["text"] == raw_function[:-2].decode("cp1252")
    assert "compiler’s invalid candidate" in function["text"]
    assert type_definition["text"] == raw_type[:-3].decode("cp1252")
    assert "compiler’s type" in type_definition["text"]
    for field in (
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "semantic_symbol_ids",
        "semantic_call_targets",
        "semantic_type_refs",
        "semantic_def_use",
    ):
        assert len(function[field]) == len(function["text"])


def test_source_with_nul_byte_is_rejected() -> None:
    index_project = _load_indexer()

    with pytest.raises(ValueError, match="source contains NUL byte"):
        index_project._decode_source_bytes(b"int value = 0;\0\n", "binary.cpp")


def test_mixed_legacy_source_uses_byte_exact_latin1_fallback() -> None:
    from tools.clang_indexer import index_project

    cp1252_source = b"// compiler\x92s type\n"
    cp1252_text, cp1252_encoding = index_project._decode_source_bytes(
        cp1252_source,
        "cp1252.h",
    )
    assert cp1252_encoding == "cp1252"
    assert cp1252_text.encode(cp1252_encoding) == cp1252_source

    mixed_source = b'// "\x8d\xc5\x8f\x89\x82\xcc\x8ds: \xb1\xb2\xb3"\n'
    mixed_text, mixed_encoding = index_project._decode_source_bytes(
        mixed_source,
        "mixed-shift-jis-font-table.h",
    )
    assert mixed_encoding == "latin-1"
    assert mixed_text.encode(mixed_encoding) == mixed_source


def test_parse_pool_emits_heartbeat_while_a_batch_is_still_running(
    capsys,
) -> None:
    from tools.clang_indexer import index_project

    class DelayedExecutor:
        def __init__(self) -> None:
            self.future = Future()
            self.timer = Timer(
                0.05,
                self.future.set_result,
                args=[("batch-result", 1)],
            )

        def submit(self, _fn, _batch):
            self.timer.start()
            return self.future

    executor = DelayedExecutor()
    results = list(
        index_project._iter_parse_batch_results(
            executor,
            ["slow-batch"],
            max_in_flight=1,
            heartbeat_interval_s=0.01,
        )
    )
    executor.timer.join()

    assert results == [("batch-result", 1)]
    assert "Parse pool heartbeat:" in capsys.readouterr().err
