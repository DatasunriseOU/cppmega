from __future__ import annotations

from importlib.machinery import ModuleSpec
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import cppmega.data.prompt_graph_index as data_prompt_graph_index
import cppmega.prompt_graph_index as prompt_graph_index
from cppmega.symbol_identity import SYMBOL_IDENTITY_SCHEMA_VERSION, compute_symbol_id


ROOT = Path(__file__).resolve().parents[1]
INDEXER_PATH = ROOT / "tools" / "clang_indexer" / "index_project.py"
TOKENIZER_PATH = ROOT / "cppmega" / "tokenizer" / "tokenizer.json"


@pytest.mark.parametrize(
    "loader_module",
    (prompt_graph_index, data_prompt_graph_index),
)
def test_loader_rejects_foreign_cached_module(
    monkeypatch: pytest.MonkeyPatch,
    loader_module: ModuleType,
) -> None:
    if hasattr(loader_module, "_indexer_module_name"):
        _manifest, dependency_hash = loader_module.indexer_dependency_hash(
            INDEXER_PATH,
            ROOT,
        )
        module_name = loader_module._indexer_module_name(
            INDEXER_PATH,
            dependency_hash,
        )
    else:
        module_name = (
            "_cppmega_prompt_graph_clang_indexer_"
            + loader_module._sha_file(INDEXER_PATH)[:12]
        )
    foreign_path = Path("/tmp/foreign/index_project.py")
    foreign = ModuleType(module_name)
    foreign.__file__ = str(foreign_path)
    foreign.__spec__ = ModuleSpec(
        module_name,
        loader=None,
        origin=str(foreign_path),
    )
    monkeypatch.setitem(sys.modules, module_name, foreign)

    with pytest.raises(ValueError, match="cached clang indexer provenance"):
        loader_module._load_indexer(ROOT)


def test_deferred_dedup_import_survives_loader_sys_path_restore(
    tmp_path: Path,
) -> None:
    module, _path = prompt_graph_index._load_indexer(ROOT)
    build_file = tmp_path / "build.ninja"
    build_file.write_text("\n", encoding="utf-8")

    documents = module.emit_build_documents(
        [(str(build_file), "ninja")],
        default_build_info=None,
        tokenizer_path=str(TOKENIZER_PATH),
        dedup_db=str(tmp_path / "dedup.sqlite"),
    )

    assert documents == []


def test_loader_restores_the_original_sys_path_object() -> None:
    script = f"""
import sys
from pathlib import Path
from cppmega.prompt_graph_index import _load_indexer

before = sys.path
_load_indexer(Path({str(ROOT)!r}))
assert sys.path is before
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "loader_module",
    (prompt_graph_index, data_prompt_graph_index),
)
def test_both_loaders_accept_explicit_location_identity_without_signature(
    loader_module,
    tmp_path: Path,
):
    repo_root = tmp_path / "repo"
    source = repo_root / "src" / "a.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("int run();\n", encoding="utf-8")
    key = (
        "repo_file_location:"
        f"schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}\x1f"
        "project=owner/repo\x1ffile=src/a.cpp\x1fline=1\x1fcolumn=5\x1f"
        "kind=FUNCTION_DECL\x1fqname=run"
    )

    class _Indexer:
        @staticmethod
        def symbol_reference_for_cursor(*_args, **_kwargs):
            return {
                "usr": "",
                "canonical_signature": "",
                "symbol_key": key,
                "symbol_id": compute_symbol_id(key),
                "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
                "qname": "run",
                "symbol_kind": "FUNCTION_DECL",
                "project": "owner/repo",
                "file": "src/a.cpp",
                "line": 1,
                "column": 5,
                "provider": "",
                "include_provenance": "",
            }

    identity = loader_module._identity_for_cursor(
        _Indexer(),
        SimpleNamespace(
            location=SimpleNamespace(
                file=SimpleNamespace(name=str(source)),
                line=1,
                column=5,
            )
        ),
        repo_root=repo_root,
        project_id="owner/repo",
        source_path="src/a.cpp",
    )
    assert identity is not None
    assert identity.symbol_key == key
    assert identity.identity_file == "src/a.cpp"
    assert identity.identity_column == 5
