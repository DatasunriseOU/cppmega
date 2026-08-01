from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

PORTABLE_MODULES = (
    "cppmega.data.nanochat_pipeline",
    "cppmega.data.nanochat_pipeline.corpus_stats",
    "cppmega.data.nanochat_pipeline.document_identity",
    "cppmega.data.nanochat_pipeline.language_info",
    "cppmega.data.nanochat_pipeline.packed_rows_schema",
    "cppmega.data.nanochat_pipeline.platform_vocab",
    "cppmega.data.nanochat_pipeline.tokenized_enriched",
    "cppmega.data.nanochat_pipeline.tokenized_enriched_schema",
    "cppmega.data.pr_primary_membership",
    "scripts.nanochat_data",
    "scripts.nanochat_data.atomic_publish",
    "scripts.nanochat_data.memory_guard",
    "scripts.nanochat_data.extract_git_history",
    "scripts.nanochat_data.token_budget",
    "scripts.nanochat_data.clang_enriched_to_parquet",
    "scripts.nanochat_data.materialize_tokenized_enriched_parquet",
    "scripts.nanochat_data.pack_enriched_rows",
    "scripts.streaming_reindex",
    "scripts.streaming_reindex_commits",
    "scripts.streaming_conveyor",
    "scripts.report_training_steps",
    "scripts.fix_packed_parquet_boundaries",
    "scripts.repair_packed_document_boundaries",
    "scripts.drop_invalid_packed_parquet_rows",
    "scripts.backfill_commit_pr_metadata",
    "scripts.crossrepo.build_global_symbol_index",
    "scripts.crossrepo.export_base16k_sampler",
)

PORTABLE_PATHS = tuple(
    REPO_ROOT / relative
    for relative in (
        "cppmega/data/nanochat_pipeline/__init__.py",
        "cppmega/data/nanochat_pipeline/corpus_stats.py",
        "cppmega/data/nanochat_pipeline/document_identity.py",
        "cppmega/data/nanochat_pipeline/language_info.py",
        "cppmega/data/nanochat_pipeline/packed_rows_schema.py",
        "cppmega/data/nanochat_pipeline/platform_vocab.py",
        "cppmega/data/nanochat_pipeline/tokenized_enriched.py",
        "cppmega/data/nanochat_pipeline/tokenized_enriched_schema.py",
        "cppmega/data/pr_primary_membership.py",
        "scripts/nanochat_data/__init__.py",
        "scripts/nanochat_data/atomic_publish.py",
        "scripts/nanochat_data/memory_guard.py",
        "scripts/nanochat_data/extract_git_history.py",
        "scripts/nanochat_data/token_budget.py",
        "scripts/nanochat_data/clang_enriched_to_parquet.py",
        "scripts/nanochat_data/materialize_tokenized_enriched_parquet.py",
        "scripts/nanochat_data/pack_enriched_rows.py",
        "scripts/streaming_reindex.py",
        "scripts/streaming_reindex_commits.py",
        "scripts/streaming_conveyor.py",
        "scripts/report_training_steps.py",
        "scripts/fix_packed_parquet_boundaries.py",
        "scripts/repair_packed_document_boundaries.py",
        "scripts/drop_invalid_packed_parquet_rows.py",
        "scripts/backfill_commit_pr_metadata.py",
        "scripts/crossrepo/build_global_symbol_index.py",
        "scripts/crossrepo/export_base16k_sampler.py",
    )
)

FORBIDDEN_IMPORT_ROOTS = frozenset({"cppmega_mlx", "cppmega_v4", "mlx", "mlx_lm"})


def test_portable_source_pipeline_imports_without_sibling_checkout() -> None:
    script = f"""
import importlib
import importlib.abc
from pathlib import Path
import sys

repo_root = Path({str(REPO_ROOT)!r}).resolve()
sibling_root = (repo_root.parent / "cppmega.mlx").resolve()

class BlockSiblingNamespaces(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {tuple(sorted(FORBIDDEN_IMPORT_ROOTS))!r}:
            raise ImportError(f"blocked non-portable import: {{fullname}}")
        return None

sys.meta_path.insert(0, BlockSiblingNamespaces())
sys.path = [
    entry
    for entry in sys.path
    if not entry or Path(entry).resolve() != sibling_root
]
sys.path.insert(0, str(repo_root))
assert sibling_root not in [Path(entry).resolve() for entry in sys.path if entry]

for module_name in {PORTABLE_MODULES!r}:
    module = importlib.import_module(module_name)
    assert module.__file__ is not None, module_name
    assert Path(module.__file__).resolve().is_relative_to(repo_root), module.__file__

from cppmega.data.nanochat_pipeline.document_identity import stable_doc_signature
from scripts import fix_packed_parquet_boundaries as fixer
from scripts import streaming_conveyor as conveyor
from scripts import streaming_reindex as reindex
from scripts import streaming_reindex_commits as commits
from scripts.crossrepo import build_global_symbol_index as global_index

assert stable_doc_signature({{"source_doc_id": 17}}) == "source_doc_id:17"
assert reindex.ROOT == repo_root
assert reindex.TOKENIZER_PATH == repo_root / "cppmega" / "tokenizer" / "tokenizer.json"
assert commits.ROOT == repo_root
assert conveyor.ROOT == repo_root
assert conveyor.sr is reindex
assert conveyor.src is commits
assert commits.sr is reindex
assert conveyor.RepoFailure is reindex.RepoFailure is commits.RepoFailure
assert conveyor.Manifest is reindex.Manifest is commits.Manifest
assert global_index.ROOT == repo_root
assert fixer.TOKENIZER_DIR == repo_root / "cppmega" / "tokenizer"
assert ":(top)cppmega" in conveyor.CODE_REVISION_PATHS
assert all("cppmega_mlx" not in path for path in conveyor.CODE_REVISION_PATHS)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_streaming_conveyor_direct_import_reuses_one_reindex_module() -> None:
    script = f"""
from pathlib import Path
import sys

repo_root = Path({str(REPO_ROOT)!r}).resolve()
sys.path.insert(0, str(repo_root / "scripts"))

import streaming_conveyor as conveyor
import streaming_reindex as reindex
import streaming_reindex_commits as commits

assert conveyor.sr is reindex
assert conveyor.src is commits
assert commits.sr is reindex
assert conveyor.RepoFailure is reindex.RepoFailure is commits.RepoFailure
assert conveyor.Manifest is reindex.Manifest is commits.Manifest
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_base16k_doc_carries_symbol_identity_contract() -> None:
    from cppmega.data.symbol_identity import (
        SYMBOL_IDENTITIES_COLUMN,
        SYMBOL_IDENTITY_SCHEMA_VERSION,
        compute_symbol_id,
    )
    from scripts.crossrepo.export_base16k_sampler import BaseSymbol, _base_doc

    symbol_key = (
        "repo_file_location:project=boostorg/boost;file=boost/base.hpp;"
        "line=10;column=0;kind=func;qname=boost::base_fn"
    )
    symbol_id = compute_symbol_id(symbol_key)
    doc = _base_doc(
        BaseSymbol(
            qname="boost::base_fn",
            base_lib="boost",
            base_repo="boostorg/boost",
            kind=2,
            sym_type="func",
            file="boost/base.hpp",
            line=10,
            token_est=8,
            body_len=19,
            text="int base_fn() { }",
            symbol_id=symbol_id,
            symbol_key=symbol_key,
        ),
        repeat_index=0,
        token_count=8,
    )

    assert doc["symbol_identity_schema_version"] == SYMBOL_IDENTITY_SCHEMA_VERSION
    assert doc[SYMBOL_IDENTITIES_COLUMN] == [{"symbol_id": symbol_id, "symbol_key": symbol_key}]
    assert doc["chunk_boundaries"][0]["symbol_id"] == symbol_id
    assert doc["symbol_ids"] == [symbol_id] * len(doc["text"])


def test_portable_source_pipeline_has_no_mlx_or_sibling_imports() -> None:
    missing = [str(path.relative_to(REPO_ROOT)) for path in PORTABLE_PATHS if not path.is_file()]
    assert missing == [], f"missing portable production files: {missing}"

    violations: list[str] = []
    for path in PORTABLE_PATHS:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPO_ROOT)
        for forbidden_text in (
            "cppmega_mlx",
            "cppmega_v4",
            "MLX_ROOT",
            "mlx.core",
            "/Volumes/external/sources/cppmega.mlx",
        ):
            if forbidden_text in text:
                violations.append(f"{relative}: contains {forbidden_text!r}")

        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
            for module_name in imported:
                if module_name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(
                        f"{relative}:{node.lineno}: forbidden import {module_name}"
                    )

    assert violations == []


def test_nanochat_entrypoints_support_direct_file_help() -> None:
    for relative in (
        "scripts/nanochat_data/materialize_tokenized_enriched_parquet.py",
        "scripts/nanochat_data/pack_enriched_rows.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / relative), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert completed.returncode == 0, f"{relative}: {completed.stderr}"
