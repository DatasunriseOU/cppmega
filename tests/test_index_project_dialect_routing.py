from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from cppmega.data.domain_ingestion import discover_project_domain_files
from cppmega.data.domain_schema import DomainKind, DomainRoleKind, ParseConfidence
from tools.clang_indexer import index_project as ip


def _write_files(root: Path, fixtures: dict[str, str]) -> None:
    for relative, text in fixtures.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_build_and_domain_discovery_covers_frozen_build_dialects(
    tmp_path: Path,
) -> None:
    fixtures = {
        "meson.build": "project('demo')\n",
        "BUILD.gn": 'executable("demo") { sources = [ "main.cc" ] }\n',
        "config/compiler.gni": "declare_args() { use_demo = true }\n",
        "SConstruct": 'Program("demo", ["main.cpp"])\n',
        "sub/SConscript": 'Library("demo", ["lib.cpp"])\n',
        "xmake.lua": 'target("demo")\n  set_kind("binary")\n',
    }
    _write_files(tmp_path, fixtures)

    build_files = {
        Path(path).relative_to(tmp_path).as_posix(): kind
        for path, kind in ip.find_build_files(str(tmp_path))
    }
    assert build_files == {
        "BUILD.gn": "gn",
        "SConstruct": "scons",
        "config/compiler.gni": "gn",
        "meson.build": "meson",
        "sub/SConscript": "scons",
        "xmake.lua": "xmake",
    }

    discovered = {
        item.path.relative_to(tmp_path).as_posix(): (item.domain, item.adapter)
        for item in discover_project_domain_files(tmp_path, include_cpp=False)
    }
    assert discovered == {
        "BUILD.gn": (DomainKind.GN, "gn-raw"),
        "SConstruct": (DomainKind.SCONS, "scons-raw"),
        "config/compiler.gni": (DomainKind.GN, "gn-raw"),
        "meson.build": (DomainKind.MESON, "meson"),
        "sub/SConscript": (DomainKind.SCONS, "scons-raw"),
        "xmake.lua": (DomainKind.XMAKE, "xmake-raw"),
    }


def test_process_project_preserves_domain_and_powershell_dialect(
    tmp_path: Path,
) -> None:
    fixtures = {
        "BUILD.gn": 'executable("demo") { sources = [ "main.cc" ] }\n',
        "SConstruct": 'Program("demo", ["main.cpp"])\n',
        "xmake.lua": 'target("demo")\n  set_kind("binary")\n',
        "scripts/build.ps1": (
            '$Root = "./src"\n'
            "Get-ChildItem -Path $Root | Where-Object { $_.Name }\n"
        ),
        "scripts/run.txt": "#!/usr/bin/env pwsh\nWrite-Output typed-only\n",
        "scripts/build.cmd": (
            "@echo off\n"
            "rem ignored: type secret.txt | upload.exe\n"
            ":: type secret.txt | upload.exe\n"
            "echo ^| literal\n"
        ),
        "schema.sql": "CREATE TABLE demo(id INTEGER PRIMARY KEY);\n",
    }
    _write_files(tmp_path, fixtures)

    documents = ip.process_project(
        str(tmp_path),
        enriched=True,
        project_id="fixture/dialect-routing",
    )
    by_path = {str(doc["filepath"]): doc for doc in documents}

    assert set(by_path) == set(fixtures)
    expected_build_domains = {
        "BUILD.gn": DomainKind.GN,
        "SConstruct": DomainKind.SCONS,
        "xmake.lua": DomainKind.XMAKE,
    }
    for filepath, domain in expected_build_domains.items():
        doc = by_path[filepath]
        assert doc["doc_type"] == "build"
        assert doc["domain_kind"] == int(domain)
        assert set(doc["domain_confidence_ids"]) == {int(ParseConfidence.RAW)}

    powershell = by_path["scripts/build.ps1"]
    assert powershell["doc_type"] == "shell"
    assert powershell["build_kind"] == "powershell"
    assert powershell["domain_kind"] == int(DomainKind.SH)
    assert powershell["language_info"]["primary_language"] == "powershell"
    assert powershell["language_info"]["primary_dialect"] == "powershell"
    assert powershell["language_info"]["primary_standard"] is None
    assert powershell["domain_parse_info"]["parser_adapter"] == "powershell"
    assert powershell["domain_parse_info"]["shell_dialect"] == "powershell"
    assert powershell["shell_edges"]

    typed_only_powershell = by_path["scripts/run.txt"]
    assert typed_only_powershell["doc_type"] == "shell"
    assert typed_only_powershell["build_kind"] == "powershell"
    assert typed_only_powershell["domain_kind"] == int(DomainKind.SH)
    assert typed_only_powershell["language_info"]["primary_dialect"] == "powershell"
    assert typed_only_powershell["domain_parse_info"]["parser_adapter"] == "powershell"

    cmd = by_path["scripts/build.cmd"]
    assert cmd["doc_type"] == "shell"
    assert cmd["build_kind"] == "cmd"
    assert cmd["domain_kind"] == int(DomainKind.SH)
    assert cmd["language_info"]["primary_dialect"] == "cmd"
    assert cmd["domain_parse_info"]["parser_adapter"] == "cmd"
    assert cmd["domain_parse_info"]["shell_dialect"] == "cmd"
    assert set(cmd["domain_confidence_ids"]) == {int(ParseConfidence.RAW)}
    assert cmd["domain_parse_info"]["raw_reason"] in {
        "cmd_native_parser_unavailable",
        "malformed_cmd_shell",
    }
    assert cmd["shell_edges"] == []

    sql = by_path["schema.sql"]
    assert sql["doc_type"] == "sql"
    assert sql["build_kind"] == "sql"
    assert sql["domain_kind"] == int(DomainKind.SQL)
    assert int(DomainRoleKind.TARGET) in sql["domain_role_ids"]
    assert sql["language_info"]["primary_standard"] is None
    assert sql["language_info"]["signals"] == ["sql_file:sql"]
    assert sql["language_info"]["detector_sources"] == ["sql_file"]

    for filepath, text in fixtures.items():
        doc = by_path[filepath]
        assert doc["text"] == text
        assert len(doc["domain_ids"]) == len(text)
        assert doc["filepath"] == filepath


def test_meson_tree_sitter_edges_use_character_offsets_after_utf8_prefix() -> None:
    from cppmega.data.build_parsers.base import ParsedDomainDocument
    from cppmega.data.build_parsers.meson import _attach_edges_to_doc
    from cppmega.data.domain_ingestion import parse_domain_document
    from cppmega.data.domain_schema import DomainEdgeKind

    text = "# Привет, мир\nexecutable('demo', 'src/main.cpp')\n"
    source = text.encode("utf-8")
    direct = ParsedDomainDocument.new(domain=DomainKind.MESON, text=text)
    _attach_edges_to_doc(
        direct,
        [
            (
                source.index(b"executable"),
                source.index(b"'src/main.cpp'"),
                int(DomainEdgeKind.BUILD_TARGET_SOURCE),
            )
        ],
        source,
    )
    assert [
        (direct.tokens[source_idx].text, direct.tokens[target_idx].text, kind)
        for source_idx, target_idx, kind in direct.edges
    ] == [
        (
            "executable",
            "'src/main.cpp'",
            int(DomainEdgeKind.BUILD_TARGET_SOURCE),
        )
    ]

    parsed = parse_domain_document("meson.build", text)
    assert parsed.domain == DomainKind.MESON
    assert parsed.metadata["parser_adapter"] == "meson"
    if parsed.metadata.get("tree_sitter_language") == "meson":
        edge_tokens = {
            (parsed.tokens[source_idx].text, parsed.tokens[target_idx].text, kind)
            for source_idx, target_idx, kind in parsed.edges
        }
        assert (
            "executable",
            "'src/main.cpp'",
            int(DomainEdgeKind.BUILD_TARGET_SOURCE),
        ) in edge_tokens


def test_emit_build_documents_chunks_build_shell_and_sql_losslessly(
    tmp_path: Path,
    capsys,
) -> None:
    chunk_bytes = 64

    def fixed_line(payload: str) -> str:
        padding = chunk_bytes - len(payload.encode("utf-8")) - 1
        assert padding >= 0
        return f"{payload}{' ' * padding}\n"

    def fixed_sql_statement(payload: str) -> str:
        padding = chunk_bytes - len(payload.encode("utf-8")) - 1
        assert padding >= 0
        return f"{payload}{' ' * padding};"

    powershell_chunk = fixed_line('Write-Output "повтор"')
    whitespace_chunk = fixed_line("")
    powershell_text = (
        powershell_chunk * 2
        + whitespace_chunk
        + powershell_chunk * 2
    )
    sql_chunk = fixed_sql_statement("INSERT INTO events VALUES ('повтор')")
    sql_text = sql_chunk * 4
    gn_chunk = fixed_line('source_set("повтор") { }')
    gn_text = gn_chunk * 4
    fixtures = {
        "BUILD.gn": gn_text,
        "scripts/large.ps1": powershell_text,
        "schema.sql": sql_text,
    }
    _write_files(tmp_path, fixtures)

    documents = ip.emit_build_documents(
        [
            (str(tmp_path / "BUILD.gn"), "gn"),
            (str(tmp_path / "scripts/large.ps1"), "powershell"),
            (str(tmp_path / "schema.sql"), "sql"),
        ],
        source_root=str(tmp_path),
        project_id="fixture/oversized-domains",
        default_build_info=None,
        max_chunk_bytes=chunk_bytes,
        tokenizer_path=str(
            Path(ip.__file__).resolve().parents[2]
            / "cppmega/tokenizer/tokenizer.json"
        ),
        dedup_db=str(tmp_path / "dedup.sqlite"),
        dedup_near=True,
    )
    by_path: defaultdict[str, list[dict]] = defaultdict(list)
    for document in documents:
        by_path[str(document["filepath"])].append(document)

    assert set(by_path) == set(fixtures)
    for filepath, source_text in fixtures.items():
        chunks = sorted(
            by_path[filepath],
            key=lambda doc: int(doc["source_span"]["chunk_index"]),
        )
        assert len(chunks) > 1
        assert "".join(str(doc["text"]) for doc in chunks) == source_text
        assert [
            int(doc["source_span"]["chunk_index"])
            for doc in chunks
        ] == list(range(len(chunks)))
        assert int(chunks[0]["source_span"]["byte_start"]) == 0
        assert int(chunks[-1]["source_span"]["byte_end"]) == len(
            source_text.encode("utf-8")
        )
        for previous, current in zip(chunks, chunks[1:]):
            assert previous["source_span"]["byte_end"] == current["source_span"]["byte_start"]
            assert previous["source_span"]["char_end"] == current["source_span"]["char_start"]
        for doc in chunks:
            assert len(str(doc["text"]).encode("utf-8")) <= chunk_bytes
            assert doc["source_span"] == doc["domain_parse_info"]["source_span"]
            assert doc["build_kind"]
        assert Counter(str(doc["text"]) for doc in chunks).most_common(1)[0][1] >= 2

    assert whitespace_chunk in {
        str(doc["text"]) for doc in by_path["scripts/large.ps1"]
    }
    assert not (tmp_path / "dedup.sqlite").exists()

    assert {
        (doc["doc_type"], doc["build_kind"], doc["domain_kind"])
        for doc in by_path["scripts/large.ps1"]
    } == {("shell", "powershell", int(DomainKind.SH))}
    assert {
        (doc["doc_type"], doc["build_kind"], doc["domain_kind"])
        for doc in by_path["schema.sql"]
    } == {("sql", "sql", int(DomainKind.SQL))}
    assert {
        (doc["doc_type"], doc["build_kind"], doc["domain_kind"])
        for doc in by_path["BUILD.gn"]
    } == {("build", "gn", int(DomainKind.GN))}

    expected_chars = sum(len(text) for text in fixtures.values())
    assert capsys.readouterr().err.splitlines()[-1] == (
        f"  Build docs: emitted={len(documents)} "
        f"source_chars_in={expected_chars} "
        f"source_chars_out={expected_chars} "
        "skipped_zero_length=0 "
        "source_chunk_dedup=disabled_for_lossless_spans"
    )


def test_build_kind_survives_tokenized_and_packed_materialization(
    tmp_path: Path,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("tokenizers")

    from cppmega.data.nanochat_pipeline.tokenized_enriched import (
        materialize_tokenized_enriched_batch,
    )
    from scripts.nanochat_data.clang_enriched_to_parquet import rows_to_table
    from scripts.nanochat_data.pack_enriched_rows import (
        pack_documents,
        read_tokenized_documents,
    )
    from scripts.nanochat_data.token_budget import load_tokenizer

    fixtures = [
        ("scripts/build.ps1", "$Root = './src'\nWrite-Output $root\n", "powershell"),
        ("scripts/build.cmd", "@echo off\ndir ./src\n", "cmd"),
        ("schema.sql", "CREATE TABLE demo(id INTEGER);\n", "sql"),
        ("BUILD.gn", 'source_set("demo") { sources = [ "a.cc" ] }\n', "gn"),
    ]
    documents = [
        ip.build_build_doc(
            filepath,
            text,
            build_kind,
            project_id="fixture/build-kind-e2e",
        )
        for filepath, text, build_kind in fixtures
    ]
    tokenizer = load_tokenizer(
        str(
            Path(ip.__file__).resolve().parents[2]
            / "cppmega/tokenizer/tokenizer.json"
        )
    )
    tokenized = materialize_tokenized_enriched_batch(
        documents,
        tokenizer,
        num_threads=1,
    )
    tokenized_path = tmp_path / "tokenized.parquet"
    tokenized_table = rows_to_table(documents, tokenized_rows=tokenized)
    pq.write_table(tokenized_table, tokenized_path)
    assert tokenized_table.column("build_kind").to_pylist() == [
        "powershell",
        "cmd",
        "sql",
        "gn",
    ]

    normalized = read_tokenized_documents(tokenized_path)
    packed_rows, overflow = pack_documents(
        normalized,
        target_length=sum(document.token_count for document in normalized) + 8,
    )

    assert overflow == []
    assert len(packed_rows) == 1
    assert packed_rows[0]["source_build_kinds"] == [
        "powershell",
        "cmd",
        "sql",
        "gn",
    ]
    assert all(packed_rows[0]["source_build_kinds"])
