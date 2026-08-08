from __future__ import annotations

import base64
import hashlib
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


_FREEBSD_DIALOG_TESTDATA_8BIT_B64 = (
    "IyEvYmluL3NoCiMgJElkOiB0ZXN0ZGF0YS04Yml0LHYgMS4yIDIwMTEvMTAvMTYgMjM6MjY6MzIg"
    "dG9tIEV4cCAkCgojIFNlbGVjdCBvbmUgb2YgdGhlICJTQU1QTEU9IiBsaW5lcywgdG8gdGVzdCBo"
    "YW5kbGluZyBvZiBjaGFyYWN0ZXJzIHdoaWNoCiMgYXJlIG5vbnByaW50aW5nIGluIGEgUE9TSVgg"
    "bG9jYWxlOgoKY2FzZSAuJDEgaW4KCSMgQzEgY29udHJvbHMKLjgpCglTQU1QTEU9IoCBgoOEhYaH"
    "iImKi4yNjo8iCgk7OwouOSkKCVNBTVBMRT0ikJGSk5SVlpeYmZqbnJ2enyIKCTs7CgojIExhdGlu"
    "LTEKLlthQV0pCglTQU1QTEU9IqChoqOkpaanqKmqq6ytrq8iCgk7OwouW2JCXSkKCVNBTVBMRT0i"
    "sLGys7S1tre4ubq7vL2+vyIKCTs7Ci5bY0NdKQoJU0FNUExFPSLAwcLDxMXGx8jJysvMzc7PIgoJ"
    "OzsKLltkRF0pCglTQU1QTEU9ItDR0tPU1dbX2Nna29zd3t8iCgk7OwouW2VFXSkKCVNBTVBMRT0i"
    "4OHi4+Tl5ufo6err7O3u7yIKCTs7Ci5bZkZdKQoJU0FNUExFPSLw8fLz9PX29/j5+vv8/f7/IgoJ"
    "OzsKKikKCSMgQzAgY29udHJvbHMgKGV4Y2VwdCBhIGZldyB3aGljaCBhcmUgYWx3YXlzIHRyZWF0"
    "ZWQgc3BlY2lhbGx5IGJ5IGN1cnNlcyk6CglTQU1QTEU9IgECAwQFBgcLDA4PEBESExQVFhcYGRoi"
    "Cgk7Owplc2FjCgojIFRoaXMgc2NyaXB0IGlzIHNvdXJjZSdkIGZyb20gb3RoZXIgc2NyaXB0cywg"
    "YW5kIHVzZXMgdGhlIHBhcmFtZXRlciBsaXN0IGZyb20KIyB0aG9zZSBleHBsaWNpdGx5LiAgQnV0"
    "IHRoZXkgbWF5IHVzZSB0aGUgcGFyYW1ldGVyIGxpc3QgbGF0ZXIsIHRvIHNldCBvcHRpb25zCiMg"
    "c3BlY2lhbGx5IGZvciBkaWFsb2cuICBXb3JrIGFyb3VuZCB0aGUgY29uZmxpY3RpbmcgdXNlcyBi"
    "eSByZW1vdmluZyB0aGUKIyBwYXJhbWV0ZXIgd2hpY2ggd2UganVzdCB1c2VkIHRvIHNlbGVjdCBh"
    "IHNldCBvZiBkYXRhLgppZiB0ZXN0ICQjICE9IDAKdGhlbgoJc2hpZnQgMQpmaQo="
)

_GLIBC_TEST_GENCAT_SHIFT_JIS_B64 = (
    "IyEvYmluL3NoCiMgVGVzdCBlc2NhcGUgY2hhcmFjdGVyIGhhbmRsaW5nIGluIGdlbmNhdC4KIyBD"
    "b3B5cmlnaHQgKEMpIDIwMDAtMjAyNiBGcmVlIFNvZnR3YXJlIEZvdW5kYXRpb24sIEluYy4KIyBU"
    "aGlzIGZpbGUgaXMgcGFydCBvZiB0aGUgR05VIEMgTGlicmFyeS4KCiMgVGhlIEdOVSBDIExpYnJh"
    "cnkgaXMgZnJlZSBzb2Z0d2FyZTsgeW91IGNhbiByZWRpc3RyaWJ1dGUgaXQgYW5kL29yCiMgbW9k"
    "aWZ5IGl0IHVuZGVyIHRoZSB0ZXJtcyBvZiB0aGUgR05VIExlc3NlciBHZW5lcmFsIFB1YmxpYwoj"
    "IExpY2Vuc2UgYXMgcHVibGlzaGVkIGJ5IHRoZSBGcmVlIFNvZnR3YXJlIEZvdW5kYXRpb247IGVp"
    "dGhlcgojIHZlcnNpb24gMi4xIG9mIHRoZSBMaWNlbnNlLCBvciAoYXQgeW91ciBvcHRpb24pIGFu"
    "eSBsYXRlciB2ZXJzaW9uLgoKIyBUaGUgR05VIEMgTGlicmFyeSBpcyBkaXN0cmlidXRlZCBpbiB0"
    "aGUgaG9wZSB0aGF0IGl0IHdpbGwgYmUgdXNlZnVsLAojIGJ1dCBXSVRIT1VUIEFOWSBXQVJSQU5U"
    "WTsgd2l0aG91dCBldmVuIHRoZSBpbXBsaWVkIHdhcnJhbnR5IG9mCiMgTUVSQ0hBTlRBQklMSVRZ"
    "IG9yIEZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFLiAgU2VlIHRoZSBHTlUKIyBMZXNz"
    "ZXIgR2VuZXJhbCBQdWJsaWMgTGljZW5zZSBmb3IgbW9yZSBkZXRhaWxzLgoKIyBZb3Ugc2hvdWxk"
    "IGhhdmUgcmVjZWl2ZWQgYSBjb3B5IG9mIHRoZSBHTlUgTGVzc2VyIEdlbmVyYWwgUHVibGljCiMg"
    "TGljZW5zZSBhbG9uZyB3aXRoIHRoZSBHTlUgQyBMaWJyYXJ5OyBpZiBub3QsIHNlZQojIDxodHRw"
    "czovL3d3dy5nbnUub3JnL2xpY2Vuc2VzLz4uCgpzZXQgLWUKCmNvbW1vbl9vYmpwZng9JDEKdGVz"
    "dF9wcm9ncmFtX2NtZF9iZWZvcmVfZW52PSQyCnJ1bl9wcm9ncmFtX2Vudj0kMwp0ZXN0X3Byb2dy"
    "YW1fY21kX2FmdGVyX2Vudj0kNAoKIyBSdW4gdGhlIHRlc3QgcHJvZ3JhbS4KJHt0ZXN0X3Byb2dy"
    "YW1fY21kX2JlZm9yZV9lbnZ9IFwKICAke3J1bl9wcm9ncmFtX2Vudn0gXAogIE5MU1BBVEg9JHtj"
    "b21tb25fb2JqcGZ4fWNhdGdldHMvJU4uJWMuY2F0IExDX0FMTD1qYV9KUC5TSklTIFwKICAke3Rl"
    "c3RfcHJvZ3JhbV9jbWRfYWZ0ZXJfZW52fSBcCiAgICA+ICR7Y29tbW9uX29ianBmeH1jYXRnZXRz"
    "L3Rlc3QtZ2VuY2F0Lm91dAoKIyBDb21wYXJlIHdpdGggdGhlIGV4cGVjdGVkIHJlc3VsdC4KY21w"
    "IC0gJHtjb21tb25fb2JqcGZ4fWNhdGdldHMvdGVzdC1nZW5jYXQub3V0IDw8IkVPRiIKTENfTUVT"
    "U0FHRVMgPSBqYV9KUC5TSklTCnNhbXBsZTE6QUJDREVGOgpzYW1wbGUyOpP6lnuM6joKc2FtcGxl"
    "MzqXXJLolVw6CnNhbXBsZTQ6VEVTVAlUQUI6CnNhbXBsZTU6i0CUXAmPXI7tl946CmRvdWJsZSBz"
    "bGFzaFwKYW5vdGhlciBsaW5lCkVPRgpyZXM9JD8KCmNhdCA8PEVPRiB8CiNkZWZpbmUgQW5vdGhl"
    "clNldCAweDIJLyogKnN0YW5kYXJkIGlucHV0KjoxMyAqLwojZGVmaW5lIEFub3RoZXJGT08gMHgx"
    "CS8qICpzdGFuZGFyZCBpbnB1dCo6MTQgKi8KRU9GCmNtcCAke2NvbW1vbl9vYmpwZnh9Y2F0Z2V0"
    "cy90ZXN0LWdlbmNhdC5oIC0gfHwgcmVzPTEKCmV4aXQgJHJlcwo="
)


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


@pytest.mark.parametrize(
    ("codec", "bom"),
    [
        ("utf-32-le", b"\xff\xfe\0\0"),
        ("utf-32-be", b"\0\0\xfe\xff"),
    ],
)
def test_utf32_cmake_chunks_preserve_original_encoded_byte_spans(
    tmp_path: Path,
    codec: str,
    bom: bytes,
) -> None:
    from cppmega.data.domain_ingestion import (
        DiscoveredDomainFile,
        discover_project_domain_files,
        iter_domain_file_chunks,
    )

    text = "set(VALUE \"😀\")\n" * 12
    encoded = bom + text.encode(codec)
    path = tmp_path / "BOM-UTF-32.cmake"
    path.write_bytes(encoded)

    assert discover_project_domain_files(tmp_path) == [
        DiscoveredDomainFile(
            path=path,
            domain=DomainKind.CMAKE,
            adapter="cmake",
        )
    ]
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=40))
    assert "".join(chunk.text for chunk in chunks) == text
    assert {chunk.source_encoding for chunk in chunks} == {codec}
    for chunk in chunks:
        raw = encoded[chunk.byte_start : chunk.byte_end]
        if chunk.index == 0:
            raw = raw[len(bom) :]
        assert raw.decode(codec) == chunk.text


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


def test_rowbinary_sql_literal_preserves_every_source_byte(tmp_path: Path) -> None:
    from cppmega.data.domain_ingestion import (
        discover_project_domain_files,
        iter_domain_file_chunks,
    )

    encoded = (
        b"-- This query throw high-level exception instead of low-level "
        b'"too large size passed to allocator":\n\n'
        b"SELECT * FROM format(RowBinary,\n"
        b"'payload String',\n"
        b"'head\x00\x81\xff\x02tail'); -- { serverError TOO_LARGE_ARRAY_SIZE }\n"
    )
    path = tmp_path / "rowbinary.sql"
    path.write_bytes(encoded)

    discovered = discover_project_domain_files(tmp_path)
    assert [item.path for item in discovered] == [path]
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=32))

    assert {chunk.source_encoding for chunk in chunks} == {"iso-8859-1"}
    assert b"".join(
        chunk.text.encode("latin-1", errors="strict") for chunk in chunks
    ) == encoded
    assert sum(chunk.text.count("\0") for chunk in chunks) == 1
    assert all(chunk.byte_end - chunk.byte_start <= 32 for chunk in chunks)


def test_rowbinary_policy_rejects_structural_nul(tmp_path: Path) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    path = tmp_path / "rowbinary.sql"
    path.write_bytes(
        b"SELECT * FROM format(RowBinary, 'safe');\x00DROP TABLE audit;\n"
    )

    with pytest.raises(ValueError, match="binary domain input contains NUL byte"):
        list(iter_domain_file_chunks(path))


def test_rowbinary_marker_does_not_authorize_unrelated_binary_literal(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    path = tmp_path / "rowbinary.sql"
    path.write_bytes(
        b"-- format(RowBinary, 'not executable SQL')\n"
        b"SELECT 'unrelated\x00literal';\n"
    )

    with pytest.raises(ValueError, match="binary domain input contains NUL byte"):
        list(iter_domain_file_chunks(path))


def test_posix_invalid_byte_fixture_preserves_every_source_byte(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    encoded = (
        b"#!/bin/sh\n"
        b"# valid UTF-8 before the raw-byte cases\n"
        b'test_ps "E=\xc3\xb1" "LC_CTYPE=C"\n'
        b"# invalid 8-bit bytes\n"
        b'test_ps "E=x\xffx" ""\n'
        b'test_ps "E=x\x81x" ""\n'
    )
    path = tmp_path / "command.sh"
    path.write_bytes(encoded)

    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=40))

    assert {chunk.source_encoding for chunk in chunks} == {"iso-8859-1"}
    assert b"".join(
        chunk.text.encode("latin-1", errors="strict") for chunk in chunks
    ) == encoded
    assert all(chunk.byte_end - chunk.byte_start <= 40 for chunk in chunks)


def test_freebsd_dialog_8bit_fixture_preserves_exact_upstream_bytes(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import (
        discover_project_domain_files,
        iter_domain_file_chunks,
    )

    encoded = base64.b64decode(_FREEBSD_DIALOG_TESTDATA_8BIT_B64)
    assert len(encoded) == 959
    assert hashlib.sha256(encoded).hexdigest() == (
        "8da95be352cc07a792179bb103aa6f7a7a073b59ba007a28b94fd8b30afb37dc"
    )
    path = tmp_path / "contrib/dialog/samples/testdata-8bit"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    discovered = discover_project_domain_files(tmp_path)
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=96))

    assert [item.path for item in discovered] == [path]
    assert {chunk.source_encoding for chunk in chunks} == {"iso-8859-1"}
    assert b"".join(chunk.text.encode("latin-1") for chunk in chunks) == encoded
    assert all(chunk.byte_end - chunk.byte_start <= 96 for chunk in chunks)


def test_glibc_shift_jis_expected_output_preserves_exact_upstream_bytes(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import (
        discover_project_domain_files,
        iter_domain_file_chunks,
    )

    encoded = base64.b64decode(_GLIBC_TEST_GENCAT_SHIFT_JIS_B64)
    assert len(encoded) == 1577
    assert hashlib.sha256(encoded).hexdigest() == (
        "88a7a81dc5c99fe901b1fe8966bdee605aea949b2dc20cee26156db55d4cdc4d"
    )
    path = tmp_path / "catgets/test-gencat.sh"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    discovered = discover_project_domain_files(tmp_path)
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=128))

    assert [item.path for item in discovered] == [path]
    assert {chunk.source_encoding for chunk in chunks} == {
        "mixed-utf-8-shift-jis-byte-preserving"
    }
    assert b"".join(chunk.text.encode("latin-1") for chunk in chunks) == encoded
    assert all(chunk.byte_end - chunk.byte_start <= 128 for chunk in chunks)


def test_glibc_shift_jis_marker_does_not_authorize_bytes_outside_heredoc(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    encoded = base64.b64decode(_GLIBC_TEST_GENCAT_SHIFT_JIS_B64).replace(
        b"set -e\n",
        b"set -e\x81\n",
        1,
    )
    path = tmp_path / "catgets/test-gencat.sh"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    with pytest.raises(ValueError, match="invalid UTF-8 or Windows-1252"):
        list(iter_domain_file_chunks(path))


def test_invalid_shell_bytes_without_fixture_contract_still_fail(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    path = tmp_path / "command.sh"
    path.write_bytes(b'#!/bin/sh\nprintf "x\x81x"\n')

    with pytest.raises(ValueError, match="invalid UTF-8 or Windows-1252"):
        list(iter_domain_file_chunks(path))


def test_shell_byte_fixture_marker_does_not_authorize_arbitrary_invalid_bytes(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    path = tmp_path / "command.sh"
    path.write_bytes(
        b"#!/bin/sh\n"
        b"# invalid 8-bit bytes\n"
        b'printf "unrelated \x81 byte"\n'
    )

    with pytest.raises(ValueError, match="invalid UTF-8 or Windows-1252"):
        list(iter_domain_file_chunks(path))


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


def test_postgres_mule_internal_fixture_round_trips_byte_exactly(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import (
        decode_domain_prefix,
        discover_project_domain_files,
        iter_domain_file_chunks,
    )

    encoded = (
        b"-- MULE \x92 internal encoding fixture\n"
        b"SELECT '\x81' AS byte_value;\n"
    )
    path = tmp_path / "src/test/mb/sql/mule_internal.sql"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    discovered = discover_project_domain_files(tmp_path)
    decoded_prefix = decode_domain_prefix(encoded, path=path)
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=17))

    assert [item.path for item in discovered] == [path]
    assert decoded_prefix.encode("latin-1") == encoded
    assert {chunk.source_encoding for chunk in chunks} == {"mule-internal"}
    assert b"".join(chunk.text.encode("latin-1") for chunk in chunks) == encoded
    assert all(chunk.byte_end - chunk.byte_start <= 17 for chunk in chunks)


def test_postgres_mule_internal_contract_requires_signature(tmp_path: Path) -> None:
    from cppmega.data.domain_ingestion import (
        decode_domain_prefix,
        iter_domain_file_chunks,
    )

    encoded = b"-- unrelated \x92 fixture\nSELECT 1;\n"
    path = tmp_path / "src/test/mb/sql/mule_internal.sql"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    decoded_prefix = decode_domain_prefix(encoded, path=path)
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=17))

    assert decoded_prefix.encode("cp1252") == encoded
    assert {chunk.source_encoding for chunk in chunks} == {"windows-1252"}
    assert b"".join(chunk.text.encode("cp1252") for chunk in chunks) == encoded


@pytest.mark.parametrize(
    ("relative_path", "payload", "error"),
    [
        (
            "src/test/mb/sql/not_mule_internal.sql",
            b"-- near miss \x92\nSELECT '\x81';\n",
            "invalid UTF-8 or Windows-1252",
        ),
        (
            "src/test/mb/sql/mule_internal.sql",
            b"-- MULE \x92\nSELECT '\x81\x00';\n",
            "NUL byte",
        ),
    ],
)
def test_postgres_mule_internal_contract_fails_closed(
    tmp_path: Path,
    relative_path: str,
    payload: bytes,
    error: str,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=error):
        list(iter_domain_file_chunks(path))


def test_japanese_localized_batch_file_round_trips_shift_jis(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    text = "@echo off\r\nrem 日本語ヘルプ\r\n"
    encoded = text.encode("shift_jis")
    path = tmp_path / "Loc" / "VCMunge" / "Jpn" / "Res" / "MakeHelp.Bat"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    chunks = list(iter_domain_file_chunks(path))

    assert "".join(chunk.text for chunk in chunks) == text
    assert {chunk.source_encoding for chunk in chunks} == {"shift-jis"}
    assert b"".join(chunk.text.encode("shift_jis") for chunk in chunks) == encoded


def test_nt5_japanese_localized_cmd_round_trips_shift_jis(
    tmp_path: Path,
) -> None:
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    text = "@echo off\r\nrem 日本語セットアップ\r\n"
    encoded = text.encode("shift_jis")
    path = (
        tmp_path
        / "nt5src/Source/XPSP1/NT/termsrv/admtools/appcmpt/jpn/msie4usr.cmd"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=24))

    assert "".join(chunk.text for chunk in chunks) == text
    assert {chunk.source_encoding for chunk in chunks} == {"shift-jis"}
    assert b"".join(chunk.text.encode("shift_jis") for chunk in chunks) == encoded


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
