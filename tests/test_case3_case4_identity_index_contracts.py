from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cppmega import prompt_graph_index as graph_index
from cppmega import prompt_graph_provenance as provenance
from cppmega.data import prompt_graph_index as data_graph_index
from cppmega.symbol_identity import SymbolIdentityError, compute_symbol_id
from tools.clang_indexer import index_project as indexer


ROOT = Path(__file__).resolve().parents[1]


def _mlx_root() -> Path:
    configured = os.environ.get("CPPMEGA_MLX_REFERENCE_ROOT")
    expected_commit = os.environ.get("CPPMEGA_MLX_REFERENCE_COMMIT")
    if not configured or not expected_commit:
        pytest.skip(
            "cross-repository identity checks require explicit "
            "CPPMEGA_MLX_REFERENCE_ROOT and CPPMEGA_MLX_REFERENCE_COMMIT"
        )
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        pytest.skip(f"MLX reference checkout is unavailable: {root}")
    actual_commit = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        pytest.fail(
            "MLX reference checkout commit mismatch: "
            f"expected={expected_commit} actual={actual_commit}"
        )
    return root


def _cursor(path: Path, *, usr: str, displayname: str = "move(int)") -> SimpleNamespace:
    type_info = SimpleNamespace(spelling="int (int)")
    type_info.get_canonical = lambda: type_info
    result_info = SimpleNamespace(spelling="int")
    result_info.get_canonical = lambda: result_info
    parent = SimpleNamespace(
        kind=SimpleNamespace(name="NAMESPACE"),
        spelling="std",
        semantic_parent=None,
    )
    return SimpleNamespace(
        kind=SimpleNamespace(name="FUNCTION_DECL"),
        spelling="move",
        displayname=displayname,
        semantic_parent=parent,
        lexical_parent=parent,
        location=SimpleNamespace(
            file=SimpleNamespace(name=str(path)),
            line=7,
            column=4,
        ),
        linkage=SimpleNamespace(name="EXTERNAL"),
        storage_class=SimpleNamespace(name="NONE"),
        type=type_info,
        result_type=result_info,
        get_arguments=lambda: [],
        get_usr=lambda: usr,
        exception_specification_kind=SimpleNamespace(name="NONE"),
    )


def test_checkout_bound_unsafe_usr_uses_stable_scoped_fallback(
    tmp_path: Path,
) -> None:
    references = []
    for checkout_name in ("checkout-a", "checkout-b"):
        checkout = tmp_path / checkout_name
        header = checkout / "include" / "EALoad.h"
        header.parent.mkdir(parents=True)
        header.write_text("struct { int value; };\n", encoding="utf-8")
        generated_name = f"(unnamed struct at {header}:9:9)"
        cursor = _cursor(
            header,
            usr=f"c:EALoad.h@S@189@F@{generated_name}#",
            displayname=f"{generated_name}()",
        )
        references.append(
            indexer.symbol_reference_for_cursor(
                cursor,
                project_dir=str(checkout),
                project_id="owner/repo",
            )
        )

    left, right = references
    assert left["usr"] == right["usr"] == ""
    assert left["symbol_key"] == right["symbol_key"]
    assert left["canonical_signature"] == right["canonical_signature"]
    assert str(tmp_path / "checkout-a") not in left["canonical_signature"]
    assert str(tmp_path / "checkout-b") not in right["canonical_signature"]
    assert "include/EALoad.h:9:9" in left["canonical_signature"]


def test_semantic_whitespace_in_conversion_operator_usr_is_preserved(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    source = checkout / "operator.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("struct Box { operator int() const; };\n", encoding="utf-8")
    usr = "c:@S@Box@F@operator int#1"

    reference = indexer.symbol_reference_for_cursor(
        _cursor(source, usr=usr, displayname="operator int()"),
        project_dir=str(checkout),
        project_id="owner/repo",
    )

    assert reference["usr"] == usr
    assert f"usr={usr}" in reference["symbol_key"]


def test_mlx_emits_the_shared_integrity_version() -> None:
    mlx_root = _mlx_root()
    probe = (
        "from cppmega_mlx.data import prompt_graph_index as m; "
        "from cppmega_mlx.data import prompt_graph_provenance as p; "
        "assert m.INDEX_INTEGRITY_VERSION == p.INDEX_INTEGRITY_VERSION == '1'"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(mlx_root)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=mlx_root,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_shared_receipt_rejects_integrity_version_drift() -> None:
    manifest, dependency_hash = provenance.indexer_dependency_hash(
        ROOT / "tools" / "clang_indexer" / "index_project.py",
        ROOT,
    )
    receipt = {
        "index_integrity_version": provenance.INDEX_INTEGRITY_VERSION,
        "indexer_dependency_policy": provenance.INDEXER_DEPENDENCY_POLICY,
        "indexer_path": str(ROOT / "tools" / "clang_indexer" / "index_project.py"),
        "indexer_checkout_root": str(ROOT),
        "indexer_dependency_manifest": manifest,
        "hashes": {provenance.INDEXER_DEPENDENCY_HASH_KEY: dependency_hash},
        "external_references": [],
        "external_reference_count": 0,
    }
    index = SimpleNamespace(provenance=receipt, project_id="owner/repo", documents=())
    provenance.validate_shared_provenance(index, expected_indexer_root=ROOT)
    receipt["index_integrity_version"] = "0"
    with pytest.raises(ValueError, match="integrity version"):
        provenance.validate_shared_provenance(index, expected_indexer_root=ROOT)


def test_imported_local_indexer_dependency_manifest_is_complete(tmp_path: Path) -> None:
    manifest_builder = getattr(provenance, "indexer_dependency_manifest", None)
    assert callable(manifest_builder)
    manifest = manifest_builder(
        ROOT / "tools" / "clang_indexer" / "index_project.py",
        ROOT,
    )
    assert "tools/clang_indexer/index_project.py" in manifest
    assert "cppmega/symbol_identity.py" in manifest
    assert "cppmega/data/source_identity.py" in manifest

    checkout = tmp_path / "checkout"
    entrypoint = checkout / "tools" / "clang_indexer" / "index_project.py"
    helper = checkout / "pkg" / "helper.py"
    transitive = checkout / "pkg" / "transitive.py"
    entrypoint.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    entrypoint.write_text("from pkg import helper\n", encoding="utf-8")
    helper.write_text("from pkg import transitive\n", encoding="utf-8")
    transitive.write_text("VALUE = 1\n", encoding="utf-8")
    first_manifest, first_hash = provenance.indexer_dependency_hash(
        entrypoint,
        checkout,
    )
    transitive.write_text("VALUE = 2\n", encoding="utf-8")
    second_manifest, second_hash = provenance.indexer_dependency_hash(
        entrypoint,
        checkout,
    )
    assert first_hash != second_hash
    assert first_manifest["tools/clang_indexer/index_project.py"] == second_manifest[
        "tools/clang_indexer/index_project.py"
    ]
    assert first_manifest["pkg/transitive.py"] != second_manifest["pkg/transitive.py"]
    hashes = {
        "repository_sha256": "0" * 64,
        "dependency_closure_sha256": "1" * 64,
        "compile_args_sha256": "2" * 64,
        "indexer_sha256": "3" * 64,
        "libclang_version_sha256": "4" * 64,
    }
    first_key = graph_index._prompt_graph_cache_key(
        project_id="owner/repo",
        strict_diagnostics=True,
        fingerprint_hashes={
            **hashes,
            provenance.INDEXER_DEPENDENCY_HASH_KEY: first_hash,
        },
        libclang_version="clang-test",
        libclang_path="/tmp/libclang",
    )
    second_key = graph_index._prompt_graph_cache_key(
        project_id="owner/repo",
        strict_diagnostics=True,
        fingerprint_hashes={
            **hashes,
            provenance.INDEXER_DEPENDENCY_HASH_KEY: second_hash,
        },
        libclang_version="clang-test",
        libclang_path="/tmp/libclang",
    )
    assert first_key != second_key


def test_indexer_dependency_manifest_includes_package_initializers(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    entrypoint = checkout / "tools" / "clang_indexer" / "index_project.py"
    package = checkout / "pkg"
    subpackage = package / "sub"
    module = subpackage / "module.py"
    entrypoint.parent.mkdir(parents=True)
    subpackage.mkdir(parents=True)
    entrypoint.write_text("import pkg.sub.module\n", encoding="utf-8")
    (package / "__init__.py").write_text("ROOT = 1\n", encoding="utf-8")
    (subpackage / "__init__.py").write_text("SUB = 1\n", encoding="utf-8")
    module.write_text("VALUE = 1\n", encoding="utf-8")

    manifest = provenance.indexer_dependency_manifest(entrypoint, checkout)

    assert "pkg/__init__.py" in manifest
    assert "pkg/sub/__init__.py" in manifest
    assert "pkg/sub/module.py" in manifest


def test_external_provider_reference_keeps_provider_identity(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    cursor = _cursor(
        tmp_path / "sdk" / "libcxx" / "include" / "vector",
        usr="c:@N@std@F@move#I#",
    )

    reference = indexer.symbol_reference_for_cursor(
        cursor,
        project_dir=str(project),
        project_id="owner/repo",
    )
    assert reference["project"] == "llvm/llvm-project"
    assert reference["file"] == "@provider/libc++/vector"
    identity = graph_index._identity_for_cursor(
        indexer,
        cursor,
        repo_root=project,
        project_id="owner/repo",
        source_path="src.cpp",
    )
    assert identity.identity_project == "llvm/llvm-project"
    assert identity.identity_provider == "libc++"
    assert identity.identity_include_provenance == "vector"
    data_identity = data_graph_index._identity_for_cursor(
        indexer,
        cursor,
        repo_root=project,
        project_id="owner/repo",
        source_path="src.cpp",
    )
    assert data_identity.__dict__ == identity.__dict__


def test_vendored_provider_named_directory_keeps_repository_identity(
    tmp_path: Path,
) -> None:
    project = tmp_path / "repo"
    vendored_header = (
        project
        / "VTK"
        / "Infovis"
        / "Boost"
        / "vtkVariantBoostSerialization.h"
    )
    vendored_header.parent.mkdir(parents=True)
    vendored_header.write_text("#pragma once\n", encoding="utf-8")
    cursor = _cursor(
        vendored_header,
        usr="c:@F@vtkVariantBoostSerialization",
    )

    reference = indexer.symbol_reference_for_cursor(
        cursor,
        project_dir=str(project),
        project_id="Kitware/ParaView",
    )

    assert reference["project"] == "Kitware/ParaView"
    assert reference["file"] == "VTK/Infovis/Boost/vtkVariantBoostSerialization.h"
    assert reference["provider"] == ""
    assert reference["include_provenance"] == ""
    assert indexer._normalize_symbol_reference(reference) == reference


def test_unknown_external_graph_reference_is_omitted_with_receipt(
    tmp_path: Path,
) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    cursor = _cursor(
        Path("/usr/local/include/openssl/kdf.h"),
        usr="c:@F@EVP_KDF_fetch",
    )
    omissions: indexer.ExternalReferenceOmissions = {}

    with pytest.raises(SymbolIdentityError, match="stable provider identity"):
        indexer.symbol_reference_for_cursor(
            cursor,
            project_dir=str(project),
            project_id="aws/s2n-tls",
            fallback_file="src/kdf.c",
        )

    assert (
        indexer._optional_symbol_reference_for_cursor(
            cursor,
            relation="call",
            omissions=omissions,
            project_dir=str(project),
            project_id="aws/s2n-tls",
            fallback_file="src/kdf.c",
        )
        is None
    )
    summary = indexer._external_reference_omission_summary(omissions)
    assert summary["schema"] == "cppmega.external_reference_omissions_v1"
    assert summary["status"] == "complete"
    assert summary["reason"] == "unknown_external_provider"
    assert summary["observation_count"] == 1
    assert summary["unique_reference_count"] == 1
    assert summary["location_count"] == 1
    location = summary["locations"][0]
    assert location["relation"] == "call"
    assert location["symbol_kind"] == "FUNCTION_DECL"
    assert location["observed_path"] == "/usr/local/include/openssl/kdf.h"
    assert location["observations"] == 1
    assert location["unique_qname_count"] == 1
    assert location["qname_examples"] == ["std::move"]
    assert location["qname_examples_truncated"] is False


def test_optional_external_reference_does_not_swallow_other_identity_errors(
    tmp_path: Path,
) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    cursor = _cursor(
        project / "route.hpp",
        usr="c:@N@api@F@route#I#",
    )

    with pytest.raises(SymbolIdentityError):
        indexer._optional_symbol_reference_for_cursor(
            cursor,
            relation="call",
            omissions={},
            project_dir=str(project),
            project_id="not-a-canonical-project",
            fallback_file="route.hpp",
        )


def test_external_reference_receipt_is_typed_and_fail_closed(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    reference = indexer.symbol_reference_for_cursor(
        _cursor(
            tmp_path / "sdk" / "libcxx" / "include" / "vector",
            usr="c:@N@std@F@move#I#",
        ),
        project_dir=str(project),
        project_id="owner/repo",
    )
    external = {
        **reference,
        "relation": "call",
        "document_id": 1,
        "source_path": "src.cpp",
        "start": 0,
        "end": 4,
    }
    receipt = {
        "external_references": [external],
        "external_reference_count": 1,
    }
    graph = SimpleNamespace(
        project_id="owner/repo",
        documents=(
            SimpleNamespace(id=1, source_path="src.cpp", source="move(value);"),
        ),
    )

    provenance.validate_external_references(
        receipt,
        project_id=graph.project_id,
        index=graph,
    )
    receipt["external_references"] = [
        {**external, "project": "gcc-mirror/gcc"}
    ]
    with pytest.raises(ValueError, match="provider identity"):
        provenance.validate_external_references(
            receipt,
            project_id=graph.project_id,
            index=graph,
        )


def test_same_usr_provider_and_signature_claims_never_alias(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    usr = "c:@N@std@F@move#I#"
    libcxx = _cursor(
        tmp_path / "sdk-a" / "libcxx" / "include" / "vector",
        usr=usr,
        displayname="move(int)",
    )
    libcxx_signature = _cursor(
        tmp_path / "sdk-a" / "libcxx" / "include" / "vector",
        usr=usr,
        displayname="move(long)",
    )
    libstdcxx = _cursor(
        tmp_path / "sdk-b" / "libstdc++-v3" / "include" / "bits" / "stl_vector.h",
        usr=usr,
        displayname="move(long)",
    )
    left_key = indexer.symbol_identity_for_cursor(
        libcxx,
        project_dir=str(project),
        project="owner/repo",
    )[0]
    right_key = indexer.symbol_identity_for_cursor(
        libstdcxx,
        project_dir=str(project),
        project="owner/repo",
    )[0]
    signature_key = indexer.symbol_identity_for_cursor(
        libcxx_signature,
        project_dir=str(project),
        project="owner/repo",
    )[0]
    assert len({left_key, signature_key, right_key}) == 3
    assert len(
        {
            compute_symbol_id(left_key),
            compute_symbol_id(signature_key),
            compute_symbol_id(right_key),
        }
    ) == 3
    direct_key = indexer.canonical_symbol_identity(
        qname="std::move",
        kind="FUNCTION_DECL",
        usr=usr,
        canonical_signature="display=move(int)|type=int (int)|result=int|exception=NONE",
        provider="libc++",
        include_provenance="vector",
    )
    assert direct_key == left_key


def test_external_reference_normalization_preserves_and_validates_provider_claims(
    tmp_path: Path,
) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    reference = indexer.symbol_reference_for_cursor(
        _cursor(
            tmp_path / "sdk" / "libcxx" / "include" / "vector",
            usr="c:@N@std@F@move#I#",
        ),
        project_dir=str(project),
        project_id="owner/repo",
    )

    normalized = indexer._normalize_symbol_reference(reference)
    assert normalized is not None
    assert normalized["project"] == "llvm/llvm-project"
    assert normalized["file"] == "@provider/libc++/vector"
    assert normalized["provider"] == "libc++"
    assert normalized["include_provenance"] == "vector"

    metadata = indexer._symbol_part_metadata(
        str(reference["symbol_key"]),
        qname=str(reference["qname"]),
        symbol_id=int(reference["symbol_id"]),
        canonical_signature=str(reference["canonical_signature"]),
        usr=str(reference["usr"]),
        kind=str(reference["symbol_kind"]),
        provider=str(reference["provider"]),
        include_provenance=str(reference["include_provenance"]),
    )
    assert metadata is not None
    assert metadata["provider"] == "libc++"
    assert metadata["include_provenance"] == "vector"

    with pytest.raises(SymbolIdentityError, match="provider provenance"):
        indexer._normalize_symbol_reference(
            {**reference, "project": "gcc-mirror/gcc"}
        )
    with pytest.raises(SymbolIdentityError, match="provider provenance"):
        indexer._normalize_symbol_reference(
            {**reference, "file": "vector"}
        )


@pytest.mark.parametrize("module", (graph_index, data_graph_index))
def test_root_rejects_foreign_indexer_checkout(
    module: object,
    tmp_path: Path,
) -> None:
    foreign_root = _mlx_root()
    with pytest.raises(ValueError, match="same checkout|Cross-checkout"):
        module._load_indexer(foreign_root)
    foreign_link = tmp_path / "foreign-link"
    foreign_link.symlink_to(foreign_root, target_is_directory=True)
    with pytest.raises(ValueError, match="same checkout|Cross-checkout"):
        module._load_indexer(foreign_link)
