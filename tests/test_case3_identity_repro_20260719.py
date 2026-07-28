from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from cppmega import prompt_graph_index as graph_index
from cppmega import prompt_graph_provenance as provenance
from cppmega.prompt_graph import (
    INDEX_SCHEMA,
    RELATION_NAMES,
    PromptGraphContext,
    PromptProjectIndex,
    repository_snapshot,
)
from cppmega.symbol_identity import (
    SymbolIdentityError,
    canonical_usr_identity,
    parse_usr_identity,
)
from tools.clang_indexer import index_project as indexer


ROOT = Path(__file__).resolve().parents[1]


def _type_info(spelling: str) -> SimpleNamespace:
    value = SimpleNamespace(spelling=spelling)
    value.get_canonical = lambda: value
    return value


def _cursor(
    path: Path,
    *,
    usr: str,
    displayname: str,
    parent_spellings: tuple[str, ...] = ("api",),
    line: int = 2,
    column: int = 5,
    linkage: str = "EXTERNAL",
    storage: str = "NONE",
    kind: str = "FUNCTION_DECL",
    signature_type: str = "int ()",
    result_type: str = "int",
) -> SimpleNamespace:
    parent: SimpleNamespace | None = None
    for spelling in parent_spellings:
        parent = SimpleNamespace(
            kind=SimpleNamespace(name="NAMESPACE"),
            spelling=spelling,
            semantic_parent=parent,
            lexical_parent=parent,
        )
    return SimpleNamespace(
        kind=SimpleNamespace(name=kind),
        spelling=displayname.split("(", 1)[0].split("::")[-1],
        displayname=displayname,
        semantic_parent=parent,
        lexical_parent=parent,
        location=SimpleNamespace(
            file=SimpleNamespace(name=str(path)),
            line=line,
            column=column,
        ),
        linkage=SimpleNamespace(name=linkage),
        storage_class=SimpleNamespace(name=storage),
        type=_type_info(signature_type),
        result_type=_type_info(result_type),
        get_arguments=lambda: [],
        get_usr=lambda: usr,
        exception_specification_kind=(
            SimpleNamespace(name="NONE")
            if displayname
            else None
        ),
    )


def test_two_overloads_with_one_qname_route_by_exact_reference(tmp_path: Path) -> None:
    source = tmp_path / "src.cpp"
    source.write_text("\n" * 20, encoding="utf-8")
    first = _cursor(
        source,
        usr="c:@N@api@F@route#I#",
        displayname="route(int)",
        line=2,
    )
    second = _cursor(
        source,
        usr="c:@N@api@F@route#d#",
        displayname="route(double)",
        line=8,
        signature_type="double (double)",
        result_type="double",
    )

    first_reference = indexer.symbol_reference_for_cursor(
        first,
        project_dir=str(tmp_path),
        project_id="tests/case3",
    )
    second_reference = indexer.symbol_reference_for_cursor(
        second,
        project_dir=str(tmp_path),
        project_id="tests/case3",
    )

    assert first_reference["qname"] == second_reference["qname"]
    assert first_reference["usr"] != second_reference["usr"]
    assert first_reference["canonical_signature"] != second_reference[
        "canonical_signature"
    ]
    assert first_reference["symbol_key"] != second_reference["symbol_key"]

    first_def = indexer.FunctionDef(
        "route",
        "api::route",
        "src.cpp",
        2,
        "int route(int value) { return value; }",
        [],
        symbol_key=str(first_reference["symbol_key"]),
        usr=str(first_reference["usr"]),
        canonical_signature=str(first_reference["canonical_signature"]),
    )
    second_def = indexer.FunctionDef(
        "route",
        "api::route",
        "src.cpp",
        8,
        "double route(double value) { return value; }",
        [],
        symbol_key=str(second_reference["symbol_key"]),
        usr=str(second_reference["usr"]),
        canonical_signature=str(second_reference["canonical_signature"]),
    )
    project = indexer.ProjectIndex()
    project.add_function(first_def)
    project.add_function(second_def)

    assert project.resolve_function_key("api::route") is None
    assert project.resolve_function_key(first_reference) == first_def.symbol_key
    assert project.resolve_function_key(second_reference) == second_def.symbol_key


def test_template_and_overload_with_one_qname_route_by_usr(tmp_path: Path) -> None:
    source = tmp_path / "templates.cpp"
    source.write_text("\n" * 20, encoding="utf-8")
    overload = _cursor(
        source,
        usr="c:@N@api@F@route#I#",
        displayname="route(int)",
        line=2,
    )
    template = _cursor(
        source,
        usr="c:@N@api@FT@route#T#",
        displayname="route(T)",
        kind="FUNCTION_TEMPLATE",
        signature_type="T (T)",
        result_type="T",
        line=8,
    )

    overload_reference = indexer.symbol_reference_for_cursor(
        overload,
        project_dir=str(tmp_path),
        project_id="tests/case3",
    )
    template_reference = indexer.symbol_reference_for_cursor(
        template,
        project_dir=str(tmp_path),
        project_id="tests/case3",
    )

    assert overload_reference["qname"] == template_reference["qname"]
    assert overload_reference["usr"] != template_reference["usr"]
    assert overload_reference["symbol_key"] != template_reference["symbol_key"]


def test_inline_namespace_fallbacks_do_not_share_a_qname_key(tmp_path: Path) -> None:
    source = tmp_path / "inline.cpp"
    source.write_text("\n" * 20, encoding="utf-8")
    libcxx = _cursor(
        source,
        usr="",
        displayname="route()",
        parent_spellings=("std", "__1"),
        line=2,
        column=5,
        signature_type="std::__1::route ()",
    )
    alternate = _cursor(
        source,
        usr="",
        displayname="route()",
        parent_spellings=("std", "__2"),
        line=2,
        column=5,
        signature_type="std::__2::route ()",
    )

    left_key = indexer.symbol_identity_for_cursor(
        libcxx,
        project_dir=str(tmp_path),
        project="tests/case3",
    )[0]
    right_key = indexer.symbol_identity_for_cursor(
        alternate,
        project_dir=str(tmp_path),
        project="tests/case3",
    )[0]

    assert indexer.get_qualified_name(libcxx) == indexer.get_qualified_name(
        alternate
    )
    assert indexer._cursor_canonical_signature(libcxx) != (
        indexer._cursor_canonical_signature(alternate)
    )
    assert left_key != right_key


def test_file_local_symbols_are_scoped_even_when_clang_reuses_a_usr(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "one.cpp"
    second_path = tmp_path / "two.cpp"
    first_path.write_text("\n" * 20, encoding="utf-8")
    second_path.write_text("\n" * 20, encoding="utf-8")
    first = _cursor(
        first_path,
        usr="c:@F@file_local#",
        displayname="file_local()",
        line=2,
        linkage="NO_LINKAGE",
        storage="NONE",
    )
    second = _cursor(
        second_path,
        usr="c:@F@file_local#",
        displayname="file_local()",
        line=2,
        linkage="INTERNAL",
        storage="STATIC",
    )

    first_key = indexer.symbol_identity_for_cursor(
        first,
        project_dir=str(tmp_path),
        project="tests/case3",
    )[0]
    second_key = indexer.symbol_identity_for_cursor(
        second,
        project_dir=str(tmp_path),
        project="tests/case3",
    )[0]

    assert first_key != second_key
    assert "file=one.cpp" in first_key
    assert "file=two.cpp" in second_key


def test_anonymous_namespace_uses_stable_kind_signature_and_file_scope(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "one.cpp"
    second_path = tmp_path / "two.cpp"
    first_path.write_text("namespace { namespace {} }\n", encoding="utf-8")
    second_path.write_text("namespace { namespace {} }\n", encoding="utf-8")
    usr = "c:@N@api@aN@aN"
    first = _cursor(
        first_path,
        usr=usr,
        displayname="",
        line=1,
        column=23,
        linkage="INTERNAL",
        storage="INVALID",
        kind="NAMESPACE",
        signature_type="",
        result_type="",
    )
    reopened = _cursor(
        first_path,
        usr=usr,
        displayname="",
        line=8,
        column=11,
        linkage="INTERNAL",
        storage="INVALID",
        kind="NAMESPACE",
        signature_type="",
        result_type="",
    )
    second = _cursor(
        second_path,
        usr=usr,
        displayname="",
        line=1,
        column=23,
        linkage="INTERNAL",
        storage="INVALID",
        kind="NAMESPACE",
        signature_type="",
        result_type="",
    )

    first_key, _, first_signature = indexer.symbol_identity_for_cursor(
        first,
        project_dir=str(tmp_path),
        project="tests/case3",
    )
    reopened_key = indexer.symbol_identity_for_cursor(
        reopened,
        project_dir=str(tmp_path),
        project="tests/case3",
    )[0]
    second_key = indexer.symbol_identity_for_cursor(
        second,
        project_dir=str(tmp_path),
        project="tests/case3",
    )[0]

    assert first_signature == "kind=NAMESPACE"
    assert first_key == reopened_key
    assert first_key != second_key
    assert "file=one.cpp" in first_key
    assert "sig=kind%3DNAMESPACE" in first_key


def test_scoped_usr_identity_accepts_repository_path_with_spaces() -> None:
    repository_path = (
        "third_party/libsdl2/Xcode-iOS/Template/"
        "SDL iOS Application/main.c"
    )

    identity = canonical_usr_identity(
        usr="c:main.c@221@F@randomInt@min",
        project="google/filament",
        file=repository_path,
        canonical_signature="display=min|type=int|exception=UNPARSED",
    )

    parsed = parse_usr_identity(identity)
    assert parsed.file == repository_path
    assert f"file={repository_path}" in identity


@pytest.mark.parametrize(
    "repository_path",
    (
        "/absolute/main.c",
        "C:/absolute/main.c",
        "../main.c",
        "src/../main.c",
        r"src\main.c",
        "src/\x1f/main.c",
        "src/\tmain.c",
        "src/\u00a0main.c",
        " src/main.c",
        "src/main.c ",
    ),
)
def test_scoped_usr_identity_still_rejects_unsafe_repository_paths(
    repository_path: str,
) -> None:
    with pytest.raises(
        SymbolIdentityError,
        match="canonical and repository-relative",
    ):
        canonical_usr_identity(
            usr="c:main.c@221@F@randomInt@min",
            project="google/filament",
            file=repository_path,
            canonical_signature="display=min|type=int|exception=UNPARSED",
        )


def test_signature_fallback_contains_project_file_and_location(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "one.cpp"
    second_path = tmp_path / "two.cpp"
    first_path.write_text("\n" * 20, encoding="utf-8")
    second_path.write_text("\n" * 20, encoding="utf-8")
    first = _cursor(
        first_path,
        usr="",
        displayname="route()",
        line=2,
        column=5,
    )
    second = _cursor(
        second_path,
        usr="",
        displayname="route()",
        line=2,
        column=5,
    )

    first_key = indexer.symbol_identity_for_cursor(
        first,
        project_dir=str(tmp_path),
        project="tests/case3",
    )[0]
    second_key = indexer.symbol_identity_for_cursor(
        second,
        project_dir=str(tmp_path),
        project="tests/case3",
    )[0]

    assert first_key != second_key
    other_project_key = indexer.symbol_identity_for_cursor(
        first,
        project_dir=str(tmp_path),
        project="other/case3",
    )[0]
    assert other_project_key != first_key
    assert "scope=project=tests/case3|file=one.cpp|line=2|column=5" in first_key
    assert "scope=project=tests/case3|file=two.cpp|line=2|column=5" in second_key


def test_indexer_module_name_changes_with_dependency_closure(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    entrypoint = checkout / "tools" / "clang_indexer" / "index_project.py"
    helper = checkout / "pkg" / "helper.py"
    entrypoint.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    entrypoint.write_text("from pkg import helper\n", encoding="utf-8")
    helper.write_text("VALUE = 1\n", encoding="utf-8")

    _first_manifest, first_hash = provenance.indexer_dependency_hash(
        entrypoint,
        checkout,
    )
    first_name = graph_index._indexer_module_name(entrypoint, first_hash)
    helper.write_text("VALUE = 2\n", encoding="utf-8")
    _second_manifest, second_hash = provenance.indexer_dependency_hash(
        entrypoint,
        checkout,
    )
    second_name = graph_index._indexer_module_name(entrypoint, second_hash)

    assert first_hash != second_hash
    assert first_name != second_name


def test_loaded_indexer_carries_the_exact_dependency_closure() -> None:
    module, path = graph_index._load_indexer(ROOT)
    manifest, dependency_hash = provenance.indexer_dependency_hash(path, ROOT)

    assert module.__cppmega_indexer_dependency_hash__ == dependency_hash
    assert module.__cppmega_indexer_dependency_manifest__ == manifest


def test_loader_rejects_a_dependency_loaded_from_another_checkout(
    tmp_path: Path,
) -> None:
    expected_root = tmp_path / "expected"
    foreign_root = tmp_path / "foreign"
    for root in (expected_root, foreign_root):
        package = root / "pkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("\n", encoding="utf-8")
        (package / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    script = textwrap.dedent(
        """
        import importlib
        from pathlib import Path
        from cppmega.prompt_graph_index import _validate_loaded_dependency_modules

        foreign_root = Path(__import__("sys").argv[1])
        expected_root = Path(__import__("sys").argv[2])
        __import__("sys").path.insert(0, str(foreign_root))
        importlib.import_module("pkg.helper")
        _validate_loaded_dependency_modules(
            expected_root,
            {"pkg/__init__.py": "0" * 64, "pkg/helper.py": "1" * 64},
        )
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(foreign_root),
            str(expected_root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "wrong checkout" in result.stderr


def _empty_production_index(tmp_path: Path) -> PromptProjectIndex:
    source_path = tmp_path / "src" / "empty.cpp"
    source_path.parent.mkdir(parents=True)
    source = "int main() { return 0; }\n"
    source_path.write_text(source, encoding="utf-8")
    repository_hash, repository_manifest = repository_snapshot(tmp_path)
    indexer_path = ROOT / "tools" / "clang_indexer" / "index_project.py"
    dependency_manifest, dependency_hash = provenance.indexer_dependency_hash(
        indexer_path,
        ROOT,
    )
    indexer_hash = sha256(indexer_path.read_bytes()).hexdigest()
    project_id = "tests/case3-empty"
    receipt = {
        "producer": "ClangPromptProjectIndexProducer",
        "producer_version": "3",
        "index_integrity_version": provenance.INDEX_INTEGRITY_VERSION,
        "schema": INDEX_SCHEMA,
        "identity_provenance_contract": (
            provenance.PRODUCTION_IDENTITY_PROVENANCE_CONTRACT
        ),
        "project_id": project_id,
        "cache_key": "0" * 64,
        "strict_diagnostics": True,
        "symbol_identity_schema_version": 3,
        "identity_adapters": sorted(provenance.TRUSTED_IDENTITY_ADAPTERS),
        "hashes": {
            "repository_sha256": repository_hash,
            "dependency_closure_sha256": "1" * 64,
            "compile_args_sha256": "2" * 64,
            "indexer_sha256": indexer_hash,
            provenance.INDEXER_DEPENDENCY_HASH_KEY: dependency_hash,
            "libclang_version_sha256": "3" * 64,
        },
        "repository_manifest": repository_manifest,
        "dependency_closure_policy": "all_indexed_repository_sources_v1",
        "dependency_manifest": {"src/empty.cpp": repository_manifest["src/empty.cpp"]},
        "indexer_path": str(indexer_path),
        "indexer_checkout_root": str(ROOT),
        "indexer_dependency_policy": provenance.INDEXER_DEPENDENCY_POLICY,
        provenance.INDEXER_DEPENDENCY_MANIFEST_KEY: dependency_manifest,
        "toolchain": {
            "libclang_version": "test-libclang",
            "libclang_path": "/tmp/test-libclang",
            "compile_args_by_file": {"src/empty.cpp": ["-std=c++20"]},
        },
        "external_references": [],
        "external_reference_count": 0,
        "document_count": 1,
        "symbol_count": 0,
        "chunk_count": 0,
        "edge_counts": {relation: 0 for relation in RELATION_NAMES},
        "diagnostics": {"src/empty.cpp": []},
    }
    return PromptProjectIndex.from_dict(
        {
            "schema": INDEX_SCHEMA,
            "project_id": project_id,
            "documents": [
                {"id": 1, "source_path": "src/empty.cpp", "source": source}
            ],
            "symbols": [],
            "chunks": [],
            "edges": [],
            "provenance": receipt,
        }
    ).with_integrity()


def test_production_validator_rejects_an_empty_or_synthetic_repository_graph(
    tmp_path: Path,
) -> None:
    index = _empty_production_index(tmp_path)

    with pytest.raises(ValueError, match="graph|definition|chunk|edge"):
        index.validate_production_repository_index(
            expected_project_id=index.project_id,
            repository_root=tmp_path,
            expected_indexer_root=ROOT,
        )


def test_repository_context_rejects_a_nonproduction_synthetic_index() -> None:
    source = "int first() { return 1; }\n"
    index = PromptProjectIndex.from_dict(
        {
            "schema": INDEX_SCHEMA,
            "project_id": "tests/synthetic",
            "documents": [{"id": 1, "source_path": "src/a.cpp", "source": source}],
            "symbols": [],
            "chunks": [],
            "edges": [],
            "provenance": {
                "producer": "synthetic_fixture",
                "symbol_identity_schema_version": 3,
            },
        }
    )

    with pytest.raises(ValueError, match="production repository index"):
        PromptGraphContext.from_repository_prompt(
            index,
            source,
            document_id=1,
            source_path="src/a.cpp",
            source_start=0,
        )
