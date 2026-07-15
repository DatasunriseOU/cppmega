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
