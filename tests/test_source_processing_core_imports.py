from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PROCESSING_MODULES = (
    "cppmega.data",
    "cppmega.symbol_identity",
    "cppmega.data.build_context",
    "cppmega.data.language_info",
    "cppmega.data.source_identity",
    "cppmega.data.symbol_identity",
    "cppmega.data.tokenizer_contract",
    "cppmega.data.domain_schema",
    "cppmega.data.domain_packet",
    "cppmega.data.domain_ingestion",
    "cppmega.data.prompt_graph",
    "cppmega.data.prompt_graph_index",
    "cppmega.data.build_parsers",
    "cppmega.data.shell_parsers",
    "cppmega.data.diagnostic_parsers",
    "cppmega.tokenizer",
    "cppmega.tokenizer.cpp_tokenizer",
    "cppmega.tokenizer.fingerprint",
    "tools.clang_indexer.dedup_store",
    "tools.clang_indexer.index_project",
    "tools.clang_indexer.process_commits",
    "scripts.data.atomic_publish",
    "scripts.data.memory_guard",
)

OWNED_PRODUCTION_PATHS = (
    REPO_ROOT / "cppmega" / "symbol_identity.py",
    REPO_ROOT / "cppmega" / "data",
    REPO_ROOT / "cppmega" / "tokenizer",
    REPO_ROOT / "tools" / "clang_indexer",
    REPO_ROOT / "scripts" / "data" / "atomic_publish.py",
    REPO_ROOT / "scripts" / "data" / "memory_guard.py",
)


def _production_python_files() -> list[Path]:
    files: list[Path] = []
    for path in OWNED_PRODUCTION_PATHS:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


def test_source_processing_import_smoke() -> None:
    for module_name in SOURCE_PROCESSING_MODULES:
        module = importlib.import_module(module_name)
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(REPO_ROOT)


def test_source_processing_imports_do_not_require_cppmega_mlx() -> None:
    script = f"""
import importlib
import importlib.abc
import sys

class BlockCppMegaMlx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cppmega_mlx' or fullname.startswith('cppmega_mlx.'):
            raise ImportError(f'blocked sibling import: {{fullname}}')
        return None

sys.meta_path.insert(0, BlockCppMegaMlx())
sys.path.insert(0, {str(REPO_ROOT)!r})
for module_name in {SOURCE_PROCESSING_MODULES!r}:
    importlib.import_module(module_name)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_owned_production_imports_never_reference_cppmega_mlx() -> None:
    files = _production_python_files()
    assert files, "source-processing production files were not created"
    violations: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.ImportFrom):
                imported = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "cppmega_mlx" or alias.name.startswith(
                        "cppmega_mlx."
                    ):
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {alias.name}"
                        )
            if imported == "cppmega_mlx" or (
                imported is not None and imported.startswith("cppmega_mlx.")
            ):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {imported}"
                )
    assert violations == []
