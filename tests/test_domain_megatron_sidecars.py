from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cppmega.megatron import structure_dataset_patch as patch  # noqa: E402


def _load_converter_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "data_prep_parquet_to_megatron.py"
    )
    spec = importlib.util.spec_from_file_location(
        "data_prep_parquet_to_megatron",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_megatron_converter_defaults_include_domain_token_and_graph_sidecars():
    converter = _load_converter_module()
    token_names = [name for name, _ in converter.DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]
    graph_specs = {
        name: (kind, dtype)
        for name, kind, dtype in converter.DEFAULT_CPPMEGA_GRAPH_SIDECARS
    }

    assert "token_domain_ids" in token_names
    assert "token_role_ids" in token_names
    assert "token_confidence_ids" in token_names
    assert graph_specs["token_domain_edges"] == ("edge_triples", "int32")
    assert graph_specs["token_build_edges"] == ("edge_triples", "int32")
    assert graph_specs["token_shell_edges"] == ("edge_triples", "int32")
    assert graph_specs["token_diagnostic_edges"] == ("edge_triples", "int32")
    assert graph_specs["token_cross_domain_edges"] == ("edge_triples", "int32")


def test_megatron_structure_patch_requires_domain_route_columns():
    assert "domain_ids" in patch._TOKEN_BATCH_COLS
    assert patch._TOKEN_COL_ALIASES["domain_ids"][0] == "token_domain_ids"
    assert "confidence_ids" in patch._TOKEN_BATCH_COLS
    for column in (
        "token_domain_edges",
        "token_build_edges",
        "token_shell_edges",
        "token_diagnostic_edges",
        "token_cross_domain_edges",
    ):
        assert column in patch._GRAPH_ROUTE_COLS


def test_structure_patch_requires_only_consumed_token_sidecars(monkeypatch):
    monkeypatch.delenv("CPPMEGA_DOMAIN_EMBEDDING_ENABLED", raising=False)
    assert patch._required_token_batch_cols() == set(
        patch._REQUIRED_STRUCTURE_TOKEN_COLS
    )
    assert "source_doc_ids" not in patch._required_token_batch_cols()
    assert "platform_ids" not in patch._required_token_batch_cols()

    monkeypatch.setenv("CPPMEGA_DOMAIN_EMBEDDING_ENABLED", "1")
    assert set(patch._REQUIRED_DOMAIN_TOKEN_COLS) <= (
        patch._required_token_batch_cols()
    )


def test_structure_dataset_patch_remaps_domain_edge_triples_to_batch_tensors():
    graph_sidecars = {
        "token_call_edges": {
            "offsets": np.array([0, 0]),
            "data": np.empty((0, 2), dtype=np.int32),
        },
        "token_type_edges": {
            "offsets": np.array([0, 0]),
            "data": np.empty((0, 2), dtype=np.int32),
        },
        "token_domain_edges": {
            "offsets": np.array([0, 2]),
            "data": np.array([[1, 3, 20], [7, 8, 20]], dtype=np.int32),
        },
        "token_build_edges": {
            "offsets": np.array([0, 1]),
            "data": np.array([[2, 4, 21]], dtype=np.int32),
        },
        "token_shell_edges": {
            "offsets": np.array([0, 0]),
            "data": np.empty((0, 3), dtype=np.int32),
        },
        "token_diagnostic_edges": {
            "offsets": np.array([0, 1]),
            "data": np.array([[4, 1, 60]], dtype=np.int32),
        },
        "token_cross_domain_edges": {
            "offsets": np.array([0, 1]),
            "data": np.array([[4, 2, 62]], dtype=np.int32),
        },
        "token_chunk_starts": {
            "offsets": np.array([0, 0]),
            "data": np.empty((0,), dtype=np.uint32),
        },
        "token_chunk_ends": {
            "offsets": np.array([0, 0]),
            "data": np.empty((0,), dtype=np.uint32),
        },
        "token_chunk_kinds": {
            "offsets": np.array([0, 0]),
            "data": np.empty((0,), dtype=np.uint8),
        },
        "token_chunk_dep_levels": {
            "offsets": np.array([0, 0]),
            "data": np.empty((0,), dtype=np.uint16),
        },
    }
    spans = [
        {
            "real_doc": 0,
            "doc_start_token": 0,
            "source_start": 1,
            "source_end": 5,
            "target_start": 0,
        }
    ]

    out = patch._build_graph_route_tensors(
        graph_sidecars,
        spans,
        target_len=4,
        max_edges=2,
        max_chunks=1,
    )

    assert torch.equal(
        out["graph_domain_edges"], torch.tensor([[0, 2, 20], [-1, -1, -1]])
    )
    assert out["graph_domain_edge_counts"].item() == 1
    assert torch.equal(
        out["graph_build_edges"], torch.tensor([[1, 3, 21], [-1, -1, -1]])
    )
    assert out["graph_build_edge_counts"].item() == 1
    assert torch.equal(
        out["graph_diagnostic_edges"], torch.tensor([[3, 0, 60], [-1, -1, -1]])
    )
    assert torch.equal(
        out["graph_cross_domain_edges"], torch.tensor([[3, 1, 62], [-1, -1, -1]])
    )
