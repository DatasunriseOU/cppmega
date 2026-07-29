from __future__ import annotations

import json
from pathlib import Path

import pytest

from cppmega.data.domain_schema import (
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
    validate_case5_contract_metadata,
)
from cppmega.data.tokenizer_contract import TOKENIZER_CONTRACT_SHA256_METADATA_KEY


def _role_tokens(parsed, role: DomainRoleKind) -> set[str]:
    return {
        token.text
        for token, role_id in zip(parsed.tokens, parsed.role_ids, strict=True)
        if int(role_id) == int(role)
    }


def test_root_package_contracts_match_canonical_ksh_python_assignments() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    canonical_domain = json.loads(
        (repo_root / "data" / "domain_schema_v1.json").read_text(encoding="utf-8")
    )
    package_domain = json.loads(
        (repo_root / "cppmega" / "data" / "domain_schema_v1.json").read_text(
            encoding="utf-8"
        )
    )
    canonical_tokenizer = json.loads(
        (
            repo_root
            / "data"
            / "tokenizer_v2"
            / "tokenizer_contract_v1.json"
        ).read_text(encoding="utf-8")
    )
    package_tokenizer = json.loads(
        (
            repo_root / "cppmega" / "tokenizer" / "tokenizer_contract_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert package_domain == canonical_domain
    assert package_tokenizer == canonical_tokenizer
    assert canonical_domain["domain_kinds"]["KSH"] == 24
    assert canonical_domain["domain_kinds"]["PYTHON"] == 31
    assignments = canonical_tokenizer["reserved_role_assignments"]
    assert {
        role: assignments[role]
        for role in ("KSH_START", "KSH_END", "PYTHON_START", "PYTHON_END")
    } == {
        "KSH_START": 245,
        "KSH_END": 246,
        "PYTHON_START": 247,
        "PYTHON_END": 248,
    }


@pytest.mark.parametrize(
    "tokenizer_hash",
    (
        "c3bb669015c48e2049e3b82ccb8c98c6eceae0644f7da0b5b8600c573d7087a5",
        "80e73699e26d2c19fe4477cf8194886e52c7a5e114023df27e55d6a69b62c198",
    ),
)
def test_complete_pre_ksh_case5_contract_triples_remain_readable(
    tokenizer_hash: str,
) -> None:
    validate_case5_contract_metadata(
        {
            b"cppmega.domain_schema_sha256": (
                b"9c3517b5a3fda01c4f55d55bc0d12dff4af3edb3db6321bda6c22489061b4fdd"
            ),
            TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode("utf-8"): (
                tokenizer_hash.encode("ascii")
            ),
            DOMAIN_DELIMITER_CONTRACT_METADATA_KEY.encode("utf-8"): (
                b"1f2e35d7917409fc03704d32c2d55d0fb3e29f1bd9e60acca775a392cf2f53e6"
            ),
        },
        where="legacy-case5.parquet",
    )


def test_ksh_parser_keeps_domain_and_shell_graph_edges() -> None:
    from cppmega.data.shell_parsers import parse_ksh

    parsed = parse_ksh("typeset name=value\nprint $name | sed 's/x/y/' > out.txt\n")

    assert parsed.domain == DomainKind.KSH
    assert parsed.metadata["parser_adapter"] == "ksh"
    assert "typeset" in _role_tokens(parsed, DomainRoleKind.KEYWORD)
    assert "print" in _role_tokens(parsed, DomainRoleKind.COMMAND)
    assert "sed" in _role_tokens(parsed, DomainRoleKind.COMMAND)
    assert "out.txt" in _role_tokens(parsed, DomainRoleKind.PATH)
    assert any(kind == int(DomainEdgeKind.SHELL_PIPE) for _src, _dst, kind in parsed.edges)
    assert any(
        kind == int(DomainEdgeKind.SHELL_REDIR_OUT)
        for _src, _dst, kind in parsed.edges
    )


def test_python_parser_uses_stdlib_ast_and_tokenize_for_typed_spans_and_edges() -> None:
    from cppmega.data.python_parsers import parse_python

    source = (
        '"""module docs"""\n'
        "# module comment\n"
        "def greet(name: str) -> str:\n"
        '    message = f"Hello {name}"\n'
        "    print(message)\n"
        "    return message\n"
    )
    parsed = parse_python(source)

    assert parsed.domain == DomainKind.PYTHON
    assert parsed.metadata["parser_adapter"] == "python-ast-tokenize"
    assert "def" in _role_tokens(parsed, DomainRoleKind.KEYWORD)
    assert "greet" in _role_tokens(parsed, DomainRoleKind.IDENTIFIER)
    assert "# module comment" in _role_tokens(parsed, DomainRoleKind.COMMENT)
    assert any(
        parsed.text[token.start : token.end] == '"""module docs"""'
        and role_id == int(DomainRoleKind.DOCSTRING)
        for token, role_id in zip(parsed.tokens, parsed.role_ids, strict=True)
    )
    assert set(parsed.confidence_ids) == {int(ParseConfidence.EXACT)}
    assert any(kind == int(DomainEdgeKind.AST_PARENT) for _src, _dst, kind in parsed.edges)
    assert any(kind == int(DomainEdgeKind.CALL) for _src, _dst, kind in parsed.edges)
    assert any(kind == int(DomainEdgeKind.DEF_USE) for _src, _dst, kind in parsed.edges)

    enriched = parsed.to_enriched_document()
    assert enriched["domain_kind"] == int(DomainKind.PYTHON)
    assert len(enriched["domain_ids"]) == len(source)
    assert enriched["domain_edges"]


def test_python_parser_marks_syntax_errors_raw_without_losing_domain_identity() -> None:
    from cppmega.data.python_parsers import parse_python

    parsed = parse_python("def broken(:\n    pass\n")

    assert parsed.domain == DomainKind.PYTHON
    assert set(parsed.confidence_ids) == {int(ParseConfidence.RAW)}
    assert parsed.metadata["unsupported_syntax"] == "malformed_python_syntax"


def test_domain_dispatch_and_discovery_cover_ksh_and_python(tmp_path: Path) -> None:
    from cppmega.data.domain_ingestion import (
        discover_project_domain_files,
        parse_domain_document,
        resolve_domain_parser,
    )

    ksh_text = "#!/bin/ksh\nprint ok\n"
    python_text = "def answer():\n    return 42\n"
    ksh_path = tmp_path / "script.ksh"
    python_path = tmp_path / "module.py"
    shebang_path = tmp_path / "run"
    ksh_path.write_text(ksh_text, encoding="utf-8")
    python_path.write_text(python_text, encoding="utf-8")
    shebang_path.write_text("#!/usr/bin/env python3\nprint(42)\n", encoding="utf-8")

    assert resolve_domain_parser(ksh_path, ksh_text).domain == DomainKind.KSH
    assert resolve_domain_parser(python_path, python_text).domain == DomainKind.PYTHON
    assert resolve_domain_parser(shebang_path, shebang_path.read_text()).domain == DomainKind.PYTHON
    assert parse_domain_document(python_path, python_text).domain == DomainKind.PYTHON

    discovered = {
        item.path.relative_to(tmp_path).as_posix(): item.domain
        for item in discover_project_domain_files(tmp_path)
    }
    assert discovered == {
        "module.py": DomainKind.PYTHON,
        "run": DomainKind.PYTHON,
        "script.ksh": DomainKind.KSH,
    }


def test_utf16le_sql_chunks_preserve_original_encoded_byte_spans(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import (
        DiscoveredDomainFile,
        discover_project_domain_files,
        iter_domain_file_chunks,
    )

    sql = (
        "-- generated SQL\r\n"
        + "INSERT INTO audit_log VALUES (N'café', N'Москва');\r\n" * 32
    )
    encoded = b"\xff\xfe" + sql.encode("utf-16-le")
    path = tmp_path / "legacy/schema.sql"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    assert discover_project_domain_files(tmp_path) == [
        DiscoveredDomainFile(
            path=path,
            domain=DomainKind.SQL,
            adapter="sql-lexical",
        )
    ]
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=128))
    assert len(chunks) > 2
    assert "".join(chunk.text for chunk in chunks) == sql
    assert {chunk.source_encoding for chunk in chunks} == {"utf-16-le"}
    assert all(chunk.byte_end - chunk.byte_start <= 128 for chunk in chunks)

    byte_cursor = 0
    char_cursor = 0
    for chunk in chunks:
        assert (chunk.byte_start, chunk.char_start) == (byte_cursor, char_cursor)
        raw = encoded[chunk.byte_start : chunk.byte_end]
        if chunk.index == 0:
            assert raw.startswith(b"\xff\xfe")
            raw = raw[2:]
        assert raw.decode("utf-16-le") == chunk.text
        assert chunk.source_span()["source_encoding"] == "utf-16-le"
        byte_cursor = chunk.byte_end
        char_cursor = chunk.char_end
    assert (byte_cursor, char_cursor) == (len(encoded), len(sql))


def test_utf16be_chunks_do_not_split_non_bmp_code_points(tmp_path: Path) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    sql = "SELECT N'😀';\n" * 12
    encoded = b"\xfe\xff" + sql.encode("utf-16-be")
    path = tmp_path / "legacy-be.sql"
    path.write_bytes(encoded)

    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=24))
    assert "".join(chunk.text for chunk in chunks) == sql
    assert {chunk.source_encoding for chunk in chunks} == {"utf-16-be"}
    for chunk in chunks:
        raw = encoded[chunk.byte_start : chunk.byte_end]
        if chunk.index == 0:
            raw = raw[2:]
        assert raw.decode("utf-16-be") == chunk.text


def test_windows_1252_sql_is_decoded_without_replacement(tmp_path: Path) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    sql = "-- Microsoft’s legacy export\r\nSELECT 'café';\r\n"
    encoded = sql.encode("cp1252")
    path = tmp_path / "legacy.sql"
    path.write_bytes(encoded)

    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=32))
    assert "".join(chunk.text for chunk in chunks) == sql
    assert {chunk.source_encoding for chunk in chunks} == {"windows-1252"}
    assert b"".join(chunk.text.encode("cp1252") for chunk in chunks) == encoded
    assert all(chunk.byte_end - chunk.byte_start <= 32 for chunk in chunks)


@pytest.mark.parametrize(
    ("filename", "codec", "source_encoding", "text"),
    [
        ("big5.sql", "big5", "big5", "-- 中文\nSELECT 1;\n"),
        ("euc_cn.sql", "euc_cn", "euc-cn", "-- 中文\nSELECT 1;\n"),
        ("euc_jp.sql", "euc_jp", "euc-jp", "-- 日本語\nSELECT 1;\n"),
        ("euc_kr.sql", "euc_kr", "euc-kr", "-- 한국어\nSELECT 1;\n"),
        ("gb18030.sql", "gb18030", "gb18030", "-- 中文编码\nSELECT 1;\n"),
        ("sjis.sql", "shift_jis", "shift-jis", "-- 日本語\nSELECT 1;\n"),
    ],
)
def test_filename_declared_legacy_sql_round_trips_in_bounded_chunks(
    tmp_path: Path,
    filename: str,
    codec: str,
    source_encoding: str,
    text: str,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    sql = text * 12
    encoded = sql.encode(codec)
    path = tmp_path / filename
    path.write_bytes(encoded)

    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=32))

    assert "".join(chunk.text for chunk in chunks) == sql
    assert {chunk.source_encoding for chunk in chunks} == {source_encoding}
    assert b"".join(chunk.text.encode(codec) for chunk in chunks) == encoded
    assert all(chunk.byte_end - chunk.byte_start <= 32 for chunk in chunks)


def test_filename_declared_euc_tw_sql_uses_strict_native_round_trip(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    encoded = b"-- \xc4\xe3\xc5\xc6\nSELECT 1;\n"
    path = tmp_path / "euc_tw.sql"
    path.write_bytes(encoded)

    chunks = list(iter_domain_file_chunks(path))

    assert [chunk.text for chunk in chunks] == ["-- 中文\nSELECT 1;\n"]
    assert chunks[0].source_encoding == "euc-tw"
    assert (chunks[0].byte_start, chunks[0].byte_end) == (0, len(encoded))


def test_single_trailing_nul_is_explicit_in_source_provenance(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    sql = "BEGIN TRANSACTION;\r\nCOMMIT TRANSACTION;\r\n"
    encoded = sql.encode("utf-8") + b"\0"
    path = tmp_path / "terminated.sql"
    path.write_bytes(encoded)

    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=24))
    assert "".join(chunk.text for chunk in chunks) == sql
    assert chunks[-1].byte_end == len(encoded)
    assert all(chunk.source_trailing_nul_bytes == 1 for chunk in chunks)
    assert all(
        chunk.source_span()["source_trailing_nul_bytes"] == 1
        for chunk in chunks
    )
    rebuilt = b"".join(chunk.text.encode("utf-8") for chunk in chunks) + b"\0"
    assert rebuilt == encoded


@pytest.mark.parametrize("name", ["module.py", "script.ksh", "Makefile"])
def test_domain_discovery_can_audit_and_skip_invalid_explicit_inputs(
    tmp_path: Path,
    name: str,
) -> None:
    from cppmega.data.domain_ingestion import discover_project_domain_files

    explicit = tmp_path / name
    explicit.write_bytes(b"explicit domain\x81")
    rejected: list[tuple[Path, str]] = []

    discovered = discover_project_domain_files(
        tmp_path,
        invalid_input_handler=lambda path, exc: rejected.append((path, str(exc))),
    )

    assert discovered == []
    assert len(rejected) == 1
    assert rejected[0][0] == explicit
    assert rejected[0][1].startswith("invalid UTF-8 or Windows-1252")


@pytest.mark.parametrize(
    ("name", "payload", "raises"),
    [
        ("compiler-output.txt", b"error: candidate\0binary", False),
        ("compiler-output.txt", b"error: candidate\x81", False),
        ("module.py", b"print('typed')\x81", True),
        ("script.ksh", b"print typed\x81", True),
    ],
)
def test_domain_discovery_only_rejects_invalid_explicit_inputs(
    tmp_path: Path,
    name: str,
    payload: bytes,
    raises: bool,
) -> None:
    from cppmega.data.domain_ingestion import discover_project_domain_files

    (tmp_path / name).write_bytes(payload)
    if raises:
        with pytest.raises(ValueError, match="invalid UTF-8 or Windows-1252"):
            discover_project_domain_files(tmp_path)
    else:
        assert discover_project_domain_files(tmp_path) == []


def test_indexer_classifies_ksh_and_python_documents(tmp_path: Path) -> None:
    from tools.clang_indexer import index_project

    ksh_path = tmp_path / "run.ksh"
    ksh_path.write_text("#!/bin/ksh\nprint ok\n", encoding="utf-8")

    assert index_project.find_shell_files(str(tmp_path)) == [
        (str(ksh_path), "ksh")
    ]

    doc = index_project.build_build_doc(
        "module.py",
        "def answer():\n    return 42\n",
        "python",
        project_id="fixture/domain-contract",
    )
    assert doc["domain_kind"] == int(DomainKind.PYTHON)
    assert doc["doc_type"] == "code"
    assert doc["domain_parse_info"]["parser"] == "python-ast-tokenize"
    assert doc["language_info"]["primary_language"] == "python"


def test_indexer_accepts_shell_file_with_trailing_nul_terminator(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import discover_project_domain_files
    from tools.clang_indexer import index_project

    script = tmp_path / "single.ksh"
    script.write_bytes(b"#!/bin/ksh\nprint ok\0")

    assert index_project.find_shell_files(str(tmp_path)) == [(str(script), "ksh")]
    discovered = discover_project_domain_files(tmp_path)
    assert [(item.path, item.domain) for item in discovered] == [
        (script, DomainKind.KSH)
    ]
