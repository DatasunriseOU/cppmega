from __future__ import annotations

from pathlib import Path

import pytest

from cppmega.data.domain_ingestion import (
    extract_embedded_domain_blocks,
    parse_domain_document,
)
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    delimiter_token_ids,
)
from cppmega.data.nanochat_pipeline.tokenized_enriched import (
    materialize_tokenized_enriched_batch,
)
from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer


_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "cppmega"
    / "tokenizer"
    / "tokenizer.json"
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_cppmega_tokenizer(_TOKENIZER_PATH)


@pytest.mark.parametrize(
    ("filepath", "source", "content_marker"),
    (
        (
            "docs/conf.py",
            "def build_commands():\n"
            "    commands = f\"\"\"\n"
            "    export TAR=/bin/tar\n"
            "    cd {CURR_PATH.parent}\n"
            "    \"\"\"\n"
            "    return commands\n",
            "export",
        ),
        (
            "VisualC/examples/generate.py",
            "def generate(project_guid):\n"
            "    text = f\"\"\"\n"
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<ProjectGuid>{project_guid}</ProjectGuid>\n"
            "\"\"\"\n"
            "    return text\n",
            "<?xml",
        ),
        (
            "python-package/lightgbm/dask.py",
            "def bind_docs(_before_kwargs):\n"
            "    text = f\"\"\"\n"
            "        {_before_kwargs}client : Client or None\n"
            "    \"\"\"\n"
            "    return text\n",
            "client",
        ),
    ),
)
def test_python_fstring_edges_anchor_to_tokenizable_content(
    tokenizer,
    filepath: str,
    source: str,
    content_marker: str,
) -> None:
    enriched = parse_domain_document(filepath, source).to_enriched_document()
    fstring_start = source.index('f"""')
    content_start = source.index(content_marker)

    assert any(
        edge["kind"] == int(DomainEdgeKind.AST_PARENT)
        and edge["from_char"] == fstring_start
        and edge["to_char"] == content_start
        for edge in enriched["domain_edges"]
    )
    assert all(
        not source[int(edge[endpoint])].isspace()
        for edge in enriched["domain_edges"]
        for endpoint in ("from_char", "to_char")
    )

    tokenized = materialize_tokenized_enriched_batch(
        [enriched],
        tokenizer,
        num_threads=1,
    )[0]

    assert len(tokenized["token_domain_edges"]) == len(enriched["domain_edges"])


@pytest.mark.parametrize("line_ending", ("\r", "\r\n"))
def test_python_edges_with_universal_newlines_anchor_to_tokens(
    tokenizer,
    line_ending: str,
) -> None:
    source = line_ending.join(
        (
            "# exercise Python universal-newline coordinates",
            "print(repr('value'))",
            "",
        )
    )
    enriched = parse_domain_document(
        "tests/basics/string_cr_conversion.py",
        source,
    ).to_enriched_document()

    assert enriched["domain_edges"]
    assert all(
        not source[int(edge[endpoint])].isspace()
        for edge in enriched["domain_edges"]
        for endpoint in ("from_char", "to_char")
    )

    tokenized = materialize_tokenized_enriched_batch(
        [enriched],
        tokenizer,
        num_threads=1,
    )[0]

    assert len(tokenized["token_domain_edges"]) == len(enriched["domain_edges"])


@pytest.mark.parametrize("gap", ("\v", "\f"))
def test_domain_edges_anchor_to_next_token_across_tokenizer_gap(
    tokenizer,
    gap: str,
) -> None:
    source = f"left{gap}right"
    tokenized = materialize_tokenized_enriched_batch(
        [
            {
                "text": source,
                "repo": "owner/repo",
                "filepath": "src/control-gap.cc",
                "domain_edges": [
                    {
                        "from_char": 0,
                        "to_char": source.index(gap),
                        "kind": int(DomainEdgeKind.AST_PARENT),
                    }
                ],
            }
        ],
        tokenizer,
        num_threads=1,
    )[0]

    assert tokenized["token_domain_edges"] == [
        {"from": 1, "to": 2, "kind": int(DomainEdgeKind.AST_PARENT)}
    ]


def test_multiline_embedded_sql_span_uses_first_and_last_overlapping_tokens(
    tokenizer,
) -> None:
    source = (
        "void load(sqlite3* db) {\n"
        "  auto query = R\"SQL(\n"
        "    SELECT id FROM jobs;\n"
        "  )SQL\";\n"
        "}\n"
    )
    enriched = parse_domain_document("src/load.cpp", source).to_enriched_document()
    span = enriched["embedded_domain_spans"][0]

    assert source[int(span["start"])].isspace()
    assert source[int(span["end"]) - 1].isspace()

    tokenized = materialize_tokenized_enriched_batch(
        [enriched],
        tokenizer,
        num_threads=1,
    )[0]
    sql_start, sql_end = delimiter_token_ids(DomainKind.SQL)

    assert sql_start in tokenized["token_ids"]
    assert sql_end in tokenized["token_ids"]


def test_sptag_ini_raw_string_is_not_misclassified_as_sql() -> None:
    source = (
        "auto config = R\"(\n"
        "    Update=true\n"
        "    SteadyState=true\n"
        "    MergeThreshold=10\n"
        "    BufferLength=64\n"
        ")\";\n"
    )

    assert extract_embedded_domain_blocks(source, host_domain=DomainKind.CPP) == []

    sql_source = 'auto query = R\"(\n  UPDATE jobs SET ready = 1;\n)\";\n'
    sql_blocks = extract_embedded_domain_blocks(
        sql_source,
        host_domain=DomainKind.CPP,
    )
    assert len(sql_blocks) == 1
    assert sql_blocks[0].domain == DomainKind.SQL


def test_embedded_domain_span_without_tokenizable_content_still_fails_loudly(
    tokenizer,
) -> None:
    source = 'auto query = R\"SQL(\n)SQL\";\n'
    start = source.index("\n")
    end = source.index(")SQL")

    with pytest.raises(
        ValueError,
        match="embedded domain span .* could not be mapped to token spans",
    ):
        materialize_tokenized_enriched_batch(
            [
                {
                    "text": source,
                    "embedded_domain_spans": [
                        {
                            "start": start,
                            "end": end,
                            "domain_kind": int(DomainKind.SQL),
                        }
                    ],
                }
            ],
            tokenizer,
            num_threads=1,
        )
