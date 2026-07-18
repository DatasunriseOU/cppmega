from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data import memory_guard


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def get_vocab(self) -> dict[str, int]:
        return {}


def test_check_memory_limit_uses_current_rss_not_historical_peak() -> None:
    memory_guard.check_memory_limit(
        10.0,
        label="fixture-converter",
        rss_reader=lambda: 1 * 1024**3,
    )


def test_check_memory_limit_fails_when_current_rss_exceeds_limit() -> None:
    with pytest.raises(MemoryError, match="fixture-converter exceeded memory limit"):
        memory_guard.check_memory_limit(
            10.0,
            label="fixture-converter",
            rss_reader=lambda: 11 * 1024**3,
        )


def test_current_rss_continues_after_probe_failure() -> None:
    def denied() -> int | None:
        raise PermissionError("denied")

    assert memory_guard.current_rss_bytes(
        probes=(lambda: None, denied, lambda: 3 * 1024**3)
    ) == 3 * 1024**3


def test_current_rss_fails_closed_when_all_probes_are_unavailable() -> None:
    with pytest.raises(RuntimeError, match="current RSS is unavailable"):
        memory_guard.current_rss_bytes(probes=(lambda: None,))


def test_embedded_data_apis_accept_explicit_unbounded_fixture_budget(
    tmp_path: Path,
) -> None:
    """Fixture callers can explicitly disable the process-wide guard."""
    from scripts.nanochat_data import clang_enriched_to_parquet
    from tools.clang_indexer import index_project

    docs = index_project.process_project(
        str(tmp_path),
        enriched=True,
        parse_workers=1,
        project_id="tests/embedded-memory-guard",
        memory_limit_gb=0.0,
    )
    assert docs == []

    text = "int embedded() { return 1; }"
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.parquet"
    input_path.write_text(
        json.dumps(
            {
                "symbol_identity_schema_version": 3,
                "symbol_identities": [],
                "text": text,
                "structure_ids": [3] * len(text),
                "chunk_boundaries": [
                    {"start": 0, "end": len(text), "kind": 3, "dep_level": 0}
                ],
                "call_edges": [],
                "type_edges": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = clang_enriched_to_parquet.convert_local_jsonl_to_parquet(
        input_path,
        output_path,
        tokenizer=_CharTokenizer(),
        max_tokens=4096,
        overflow_policy="drop",
        memory_limit_gb=0.0,
    )
    assert summary == {"docs_in": 1, "docs_out": 1}
    assert output_path.is_file()
