from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = (
    REPO_ROOT / "cppmega" / "megatron" / "fp8_activations.py",
    REPO_ROOT / "cppmega" / "megatron" / "dsa_splitk_indexer_loss.py",
)
FORBIDDEN_IMPORT_ROOTS = frozenset({"nanochat", "cppmega_mlx"})


def _forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = (node.module,)
        else:
            continue
        for name in names:
            if name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                violations.append(f"{path.name}:{node.lineno}: {name}")
    return violations


def test_owned_runtime_modules_have_no_external_repo_imports() -> None:
    violations = [
        violation
        for path in RUNTIME_PATHS
        for violation in _forbidden_imports(path)
    ]

    assert violations == []


def test_owned_runtime_modules_import_without_external_namespaces() -> None:
    script = f"""
import importlib
import importlib.abc
import sys

attempted = []

class BlockExternalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {set(FORBIDDEN_IMPORT_ROOTS)!r}:
            attempted.append(fullname)
            raise ImportError(f'blocked external runtime import: {{fullname}}')
        return None

sys.meta_path.insert(0, BlockExternalRuntime())
sys.path.insert(0, {str(REPO_ROOT)!r})
for module_name in (
    'cppmega.megatron.fp8_activations',
    'cppmega.megatron.dsa_splitk_indexer_loss',
):
    importlib.import_module(module_name)
assert attempted == [], attempted
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_fp8_rejects_external_backend_names_explicitly() -> None:
    from cppmega.megatron.fp8_activations import (
        enable_fp8_activation_checkpointing,
    )

    with pytest.raises(ValueError, match="unsupported FP8 activation backend"):
        enable_fp8_activation_checkpointing(torch.nn.Linear(1, 1), backend="tilelang")


def test_dsa_splitk_rejects_non_cuda_inputs_explicitly() -> None:
    from cppmega.megatron.dsa_splitk_indexer_loss import (
        compute_dsa_indexer_loss_splitk,
    )

    query = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    key = torch.zeros_like(query)
    index_scores = torch.zeros((1, 1, 1), dtype=torch.float32)
    topk_indices = torch.zeros((1, 1, 1), dtype=torch.int64)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        compute_dsa_indexer_loss_splitk(
            index_scores,
            topk_indices,
            query,
            key,
            softmax_scale=1.0,
            loss_coeff=1.0,
            sparse_loss=False,
            pg_collection=None,
        )


def test_root_runtime_defaults_do_not_name_external_projects() -> None:
    fp8_source = RUNTIME_PATHS[0].read_text(encoding="utf-8")
    builder_path = (
        REPO_ROOT / "scripts" / "data" / "build_macro_routes_megatron_bundle.py"
    )
    builder_source = builder_path.read_text(encoding="utf-8")
    mirror_source = (
        REPO_ROOT / "scripts" / "data" / "mirror_mlx_parquet.py"
    ).read_text(encoding="utf-8")

    assert "from nanochat." not in fp8_source
    assert "NANOCHAT_FP8_" not in fp8_source
    assert "from cppmega_mlx" not in builder_source
    assert "import cppmega_mlx" not in builder_source
    assert "/Volumes/external/sources/cppmega.mlx" not in builder_source
    assert "/Volumes/external/sources/cppmega.mlx" not in mirror_source


def test_bundle_builder_defaults_are_root_local_or_explicit() -> None:
    from scripts.data.build_macro_routes_megatron_bundle import build_arg_parser

    args = build_arg_parser().parse_args([])

    assert args.code_root is None
    assert args.commit_root is None
    assert args.source_composition is None
    for attribute in ("output_dir", "audit_script", "repair_script", "tokenizer_dir"):
        value = getattr(args, attribute)
        assert isinstance(value, Path)
        assert value.resolve().is_relative_to(REPO_ROOT)
