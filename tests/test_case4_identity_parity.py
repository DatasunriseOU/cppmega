from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from cppmega import prompt_graph_index as prompt_graph_index_module
from cppmega import prompt_graph as prompt_graph_module
from cppmega.data import prompt_graph_index as data_prompt_graph_index_module
from cppmega.data import prompt_graph as data_prompt_graph_module
from tools.clang_indexer import index_project as indexer


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "tests" / "fixtures" / "case4_identity_payload_contract.json").read_text(
        encoding="utf-8"
    )
)


def _cursor(repo_root: Path, case: dict[str, Any]) -> SimpleNamespace:
    source = repo_root / str(case["source_path"])
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("\n" * 12, encoding="utf-8")

    parent = SimpleNamespace(
        kind=SimpleNamespace(name="NAMESPACE"),
        spelling="api",
        displayname="api",
        semantic_parent=None,
        lexical_parent=None,
    )
    type_info = SimpleNamespace(spelling=str(case["type_spelling"]))
    type_info.get_canonical = lambda: type_info
    result_info = SimpleNamespace(spelling=str(case["result_spelling"]))
    result_info.get_canonical = lambda: result_info
    arguments: list[SimpleNamespace] = []
    if case["argument_spelling"]:
        argument_type = SimpleNamespace(spelling=str(case["argument_spelling"]))
        argument_type.get_canonical = lambda: argument_type
        arguments.append(SimpleNamespace(type=argument_type))

    return SimpleNamespace(
        kind=SimpleNamespace(name="FUNCTION_DECL"),
        spelling="route" if case["name"] != "location_fallback" else "opaque",
        displayname=str(case["displayname"]),
        semantic_parent=parent,
        lexical_parent=parent,
        location=SimpleNamespace(
            file=SimpleNamespace(name=str(source)),
            line=int(case["line"]),
            column=int(case["column"]),
        ),
        linkage=SimpleNamespace(name="EXTERNAL"),
        storage_class=SimpleNamespace(name="NONE"),
        type=type_info,
        result_type=result_info,
        get_arguments=lambda: arguments,
        get_usr=lambda: str(case["usr"]),
        exception_specification_kind=(
            None
            if case["name"] == "location_fallback"
            else SimpleNamespace(name="NONE")
        ),
    )


def _root_payloads(
    repo_root: Path,
    graph_index_module: Any = prompt_graph_index_module,
) -> dict[str, list[dict[str, Any]]]:
    references: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for case in CONTRACT["cases"]:
        cursor = _cursor(repo_root, case)
        references.append(
            indexer.symbol_reference_for_cursor(
                cursor,
                project_dir=str(repo_root),
                project_id=CONTRACT["project_id"],
                fallback_file=str(case["source_path"]),
            )
        )
        identity = graph_index_module._identity_for_cursor(
            indexer,
            cursor,
            repo_root=repo_root,
            project_id=CONTRACT["project_id"],
            source_path=str(case["source_path"]),
        )
        assert identity is not None
        identities.append(dict(identity.__dict__))
    return {"references": references, "identities": identities}


def _expected_payloads() -> dict[str, list[dict[str, Any]]]:
    return {
        "references": [case["reference"] for case in CONTRACT["cases"]],
        "identities": [case["identity"] for case in CONTRACT["cases"]],
    }


def test_case4_reference_and_prompt_graph_payload_match_stable_fixture(
    tmp_path: Path,
) -> None:
    payloads = _root_payloads(tmp_path / "repo")

    assert list(payloads["references"][0]) == CONTRACT["reference_fields"]
    assert list(payloads["identities"][0]) == CONTRACT["identity_fields"]
    assert payloads == _expected_payloads()

    overloads = payloads["references"][:2]
    assert overloads[0]["qname"] == overloads[1]["qname"]
    assert overloads[0]["usr"] != overloads[1]["usr"]
    assert overloads[0]["canonical_signature"] != overloads[1]["canonical_signature"]
    assert overloads[0]["symbol_key"] != overloads[1]["symbol_key"]
    assert overloads[0]["symbol_id"] != overloads[1]["symbol_id"]


def test_case4_data_namespace_uses_the_same_identity_payload_contract(
    tmp_path: Path,
) -> None:
    payloads = _root_payloads(
        tmp_path / "repo",
        graph_index_module=data_prompt_graph_index_module,
    )
    assert payloads == _expected_payloads()


def test_real_prompt_graph_payload_preserves_case4_provenance(
    tmp_path: Path,
) -> None:
    from cppmega.prompt_graph import (
        PRODUCTION_IDENTITY_PROVENANCE_CONTRACT,
        PromptProjectIndex,
    )
    from cppmega.prompt_graph_index import ClangPromptProjectIndexProducer

    repository = ROOT / "tests" / "fixtures" / "case3_prompt_repo"
    project_id = "tests/case4-prompt-graph"
    index = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    ).build(repository, project_id=project_id).index

    assert (
        index.provenance["identity_provenance_contract"]
        == PRODUCTION_IDENTITY_PROVENANCE_CONTRACT
    )
    definitions = [
        symbol
        for symbol in index.symbols
        if symbol.kind in {"function", "type", "variable"}
    ]
    assert definitions
    assert all(symbol.identity_project == project_id for symbol in definitions)
    assert all(symbol.identity_file == symbol.source_path for symbol in definitions)
    assert all(symbol.identity_line > 0 for symbol in definitions)
    assert all(symbol.identity_column > 0 for symbol in definitions)
    assert all(symbol.identity_kind for symbol in definitions)

    overloads = [
        symbol
        for symbol in definitions
        if symbol.qname == "case3_repo::repository_helper"
    ]
    assert len(overloads) == 2
    assert len({symbol.usr for symbol in overloads}) == 2
    assert len({symbol.symbol_key for symbol in overloads}) == 2

    payload = index.to_dict()
    tampered = next(
        row
        for row in payload["symbols"]
        if row["kind"] == "function"
    )
    tampered["identity_column"] += 1
    resealed = PromptProjectIndex.from_dict(payload).with_integrity()
    with pytest.raises(ValueError, match="identity.*provenance"):
        resealed.validate_production_repository_index(
            expected_project_id=project_id,
            repository_root=repository,
            expected_indexer_root=ROOT,
        )

    for field, value, pattern in (
        ("usr", "c:@N@case3_repo@F@tampered#", "identity.*provenance|USR"),
        (
            "canonical_signature",
            "display=tampered()",
            "identity.*provenance|signature",
        ),
    ):
        payload = index.to_dict()
        row = next(
            row
            for row in payload["symbols"]
            if row["kind"] == "function"
        )
        row[field] = value
        invalid = PromptProjectIndex.from_dict(payload).with_integrity()
        with pytest.raises(ValueError, match=pattern):
            invalid.validate_production_repository_index(
                expected_project_id=project_id,
                repository_root=repository,
                expected_indexer_root=ROOT,
            )


@pytest.mark.parametrize(
    ("module",),
    [(prompt_graph_module,), (data_prompt_graph_module,)],
)
def test_prompt_graph_symbol_serializes_case4_identity_provenance(module: Any) -> None:
    row = CONTRACT["cases"][0]
    identity = row["identity"]
    symbol = module.PromptGraphSymbol(
        id=1,
        symbol_id=int(identity["symbol_id"]),
        identity="symbol:1",
        kind="function",
        document_id=1,
        source_path=str(row["source_path"]),
        start=4,
        end=9,
        semantic_identity=str(identity["semantic_identity"]),
        symbol_key=str(identity["symbol_key"]),
        usr=str(identity["usr"]),
        canonical_signature=str(identity["canonical_signature"]),
        qname=str(identity["qname"]),
        identity_project=str(identity["identity_project"]),
        identity_file=str(identity["identity_file"]),
        identity_line=int(identity["identity_line"]),
        identity_column=int(identity["identity_column"]),
        identity_kind=str(identity["symbol_kind"]),
        identity_provider=str(identity["identity_provider"]),
        identity_include_provenance=str(identity["identity_include_provenance"]),
    )

    payload = symbol.to_dict()
    assert {
        key: payload[key]
        for key in CONTRACT["identity_fields"][7:13]
    } == {
        "identity_project": "tests/case4-parity",
        "identity_file": "src/overloads.cpp",
        "identity_line": 7,
        "identity_column": 5,
        "identity_provider": "",
        "identity_include_provenance": "",
    }
    restored = module.PromptGraphSymbol.from_dict(payload)
    assert restored == symbol


def test_case4_identity_binds_repository_project_file_line_and_column(
    tmp_path: Path,
) -> None:
    case = CONTRACT["cases"][0]
    repo_root = tmp_path / "repo"
    cursor = _cursor(repo_root, case)
    valid = dict(case["reference"])

    class _Indexer:
        @staticmethod
        def symbol_reference_for_cursor(_cursor: Any, **_kwargs: Any) -> dict[str, Any]:
            return dict(valid)

    for field, value in (
        ("project", "other/project"),
        ("file", "other.cpp"),
        ("line", 8),
        ("column", 6),
    ):
        invalid = dict(valid)
        invalid[field] = value

        class _InvalidIndexer:
            @staticmethod
            def symbol_reference_for_cursor(
                _cursor: Any, *, _invalid: dict[str, Any] = invalid, **_kwargs: Any
            ) -> dict[str, Any]:
                return dict(_invalid)

        with pytest.raises(ValueError, match=field):
            prompt_graph_index_module._identity_for_cursor(
                _InvalidIndexer,
                cursor,
                repo_root=repo_root,
                project_id=CONTRACT["project_id"],
                source_path=str(case["source_path"]),
            )

    outside = _cursor(tmp_path / "outside", case)
    with pytest.raises(ValueError, match="outside repository"):
        prompt_graph_index_module._identity_for_cursor(
            _Indexer,
            outside,
            repo_root=repo_root,
            project_id=CONTRACT["project_id"],
            source_path=str(case["source_path"]),
        )


def test_case4_identity_requires_native_helper_and_semantic_identity() -> None:
    case = CONTRACT["cases"][0]
    cursor = SimpleNamespace(
        kind=SimpleNamespace(name="FUNCTION_DECL"),
        location=SimpleNamespace(
            file=SimpleNamespace(name="/tmp/repo/src/overloads.cpp"),
            line=7,
            column=5,
        ),
    )

    with pytest.raises(RuntimeError, match="native CASE 4"):
        prompt_graph_index_module._identity_for_cursor(
            SimpleNamespace(),
            cursor,
            repo_root=Path("/tmp/repo"),
            project_id=CONTRACT["project_id"],
            source_path=str(case["source_path"]),
        )

    class _QnameOnlyIndexer:
        @staticmethod
        def symbol_reference_for_cursor(_cursor: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "symbol_identity_schema_version": 3,
                "symbol_key": "fallback:schema=v3\x1fqname=api::route",
                "symbol_id": 1,
                "qname": "api::route",
                "usr": "",
                "canonical_signature": "",
                "symbol_kind": "FUNCTION_DECL",
                "project": CONTRACT["project_id"],
                "file": "src/overloads.cpp",
                "line": 7,
                "column": 5,
                "provider": "",
                "include_provenance": "",
            }

    with pytest.raises(ValueError, match="USR/signature"):
        prompt_graph_index_module._identity_for_cursor(
            _QnameOnlyIndexer,
            cursor,
            repo_root=Path("/tmp/repo"),
            project_id=CONTRACT["project_id"],
            source_path=str(case["source_path"]),
        )


_MLX_PROBE = r'''
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tools.clang_indexer import index_project as indexer
from cppmega_mlx.data import prompt_graph_index as graph_index

contract = json.loads(sys.stdin.read())

def cursor(repo_root, case):
    source = repo_root / case["source_path"]
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("\\n" * 12, encoding="utf-8")
    parent = SimpleNamespace(
        kind=SimpleNamespace(name="NAMESPACE"),
        spelling="api",
        displayname="api",
        semantic_parent=None,
        lexical_parent=None,
    )
    type_info = SimpleNamespace(spelling=case["type_spelling"])
    type_info.get_canonical = lambda: type_info
    result_info = SimpleNamespace(spelling=case["result_spelling"])
    result_info.get_canonical = lambda: result_info
    arguments = []
    if case["argument_spelling"]:
        argument_type = SimpleNamespace(spelling=case["argument_spelling"])
        argument_type.get_canonical = lambda: argument_type
        arguments.append(SimpleNamespace(type=argument_type))
    return SimpleNamespace(
        kind=SimpleNamespace(name="FUNCTION_DECL"),
        spelling="route" if case["name"] != "location_fallback" else "opaque",
        displayname=case["displayname"],
        semantic_parent=parent,
        lexical_parent=parent,
        location=SimpleNamespace(
            file=SimpleNamespace(name=str(source)),
            line=case["line"],
            column=case["column"],
        ),
        linkage=SimpleNamespace(name="EXTERNAL"),
        storage_class=SimpleNamespace(name="NONE"),
        type=type_info,
        result_type=result_info,
        get_arguments=lambda: arguments,
        get_usr=lambda: case["usr"],
        exception_specification_kind=(
            None
            if case["name"] == "location_fallback"
            else SimpleNamespace(name="NONE")
        ),
    )

with tempfile.TemporaryDirectory() as raw:
    repo = Path(raw)
    references = []
    identities = []
    for case in contract["cases"]:
        current = cursor(repo, case)
        references.append(indexer.symbol_reference_for_cursor(
            current,
            project_dir=str(repo),
            project_id=contract["project_id"],
            fallback_file=case["source_path"],
        ))
        identities.append(dict(graph_index._identity_for_cursor(
            indexer,
            current,
            repo_root=repo,
            project_id=contract["project_id"],
            source_path=case["source_path"],
        ).__dict__))
    print(json.dumps({"references": references, "identities": identities}, sort_keys=True))
'''


def test_root_payload_matches_current_mlx_payload_contract(tmp_path: Path) -> None:
    raw_root = os.environ.get("CPPMEGA_MLX_REFERENCE_ROOT")
    expected_commit = os.environ.get("CPPMEGA_MLX_REFERENCE_COMMIT")
    if not raw_root or not expected_commit:
        pytest.fail(
            "cross-repository parity requires explicit "
            "CPPMEGA_MLX_REFERENCE_ROOT and CPPMEGA_MLX_REFERENCE_COMMIT"
        )
    reference_root = Path(raw_root).expanduser().resolve()
    if not (reference_root / "tools" / "clang_indexer" / "index_project.py").is_file():
        pytest.fail(f"MLX reference checkout is unavailable: {reference_root}")
    actual_commit = subprocess.run(
        ("git", "-C", str(reference_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        pytest.fail(
            "MLX parity checkout commit mismatch: "
            f"expected={expected_commit} actual={actual_commit}"
        )

    result = subprocess.run(
        [sys.executable, "-c", _MLX_PROBE],
        cwd=reference_root,
        env={
            **os.environ,
            "PYTHONPATH": str(reference_root),
        },
        input=json.dumps(CONTRACT),
        text=True,
        capture_output=True,
        check=True,
    )
    mlx_payloads = json.loads(result.stdout)
    root_payloads = _root_payloads(tmp_path / "root-repo")
    assert root_payloads == mlx_payloads == _expected_payloads()
