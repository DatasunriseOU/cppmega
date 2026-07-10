from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "subset_megatron_sidecar_prefix.py"
    )
    spec = importlib.util.spec_from_file_location(
        "subset_megatron_sidecar_prefix", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_graph_sidecar(
    base: Path,
    column: str,
    *,
    offsets: list[int],
    rows: np.ndarray,
    item_count: int,
) -> dict[str, object]:
    """Write an edge-triples graph sidecar (offsets + data) and return its manifest entry."""
    offsets_name = f"{base.name}_{column}_offsets.bin"
    data_name = f"{base.name}_{column}_data.bin"
    np.asarray(offsets, dtype=np.int64).tofile(base.parent / offsets_name)
    rows.astype(np.int32, copy=False).tofile(base.parent / data_name)
    return {
        "kind": "edge_triples",
        "offsets_path": offsets_name,
        "data_path": data_name,
        "offset_dtype": "int64",
        "dtype": "int32",
        "item_count": item_count,
        "shape_tail": [3],
    }


def test_correct_manifest_passes(tmp_path: Path):
    module = _load_module()
    src_base = tmp_path / "src"
    dst_base = tmp_path / "dst"
    rows = np.arange(5 * 3, dtype=np.int32).reshape(5, 3)  # 5 items, 3 cols
    entry = _write_graph_sidecar(
        src_base, "token_domain_edges", offsets=[0, 2, 5], rows=rows, item_count=5
    )
    copied = dict(entry)
    copied["offsets_path"] = f"{dst_base.name}_token_domain_edges_offsets.bin"
    copied["data_path"] = f"{dst_base.name}_token_domain_edges_data.bin"

    module._copy_graph_sidecar(
        src_base=src_base,
        dst_base=dst_base,
        src_manifest_entry=entry,
        dst_manifest_entry=copied,
        document_count=1,  # subset to first doc -> offsets [0, 2]
    )

    assert copied["item_count"] == 2
    out_data = np.fromfile(
        dst_base.parent / str(copied["data_path"]), dtype=np.int32
    ).reshape(-1, 3)
    assert np.array_equal(out_data, rows[:2])
    out_offsets = np.fromfile(dst_base.parent / str(copied["offsets_path"]), dtype=np.int64)
    assert out_offsets.tolist() == [0, 2]


def test_wrong_item_count_raises(tmp_path: Path):
    module = _load_module()
    src_base = tmp_path / "src"
    dst_base = tmp_path / "dst"
    rows = np.arange(5 * 3, dtype=np.int32).reshape(5, 3)  # file holds 5 items
    entry = _write_graph_sidecar(
        src_base, "token_domain_edges", offsets=[0, 2, 5], rows=rows, item_count=4  # wrong
    )
    copied = {
        "offsets_path": f"{dst_base.name}_token_domain_edges_offsets.bin",
        "data_path": f"{dst_base.name}_token_domain_edges_data.bin",
    }
    with pytest.raises(ValueError, match="size mismatch"):
        module._copy_graph_sidecar(
            src_base=src_base,
            dst_base=dst_base,
            src_manifest_entry=entry,
            dst_manifest_entry=copied,
            document_count=2,
        )


def test_non_monotonic_offsets_raises(tmp_path: Path):
    module = _load_module()
    src_base = tmp_path / "src"
    dst_base = tmp_path / "dst"
    rows = np.arange(5 * 3, dtype=np.int32).reshape(5, 3)
    entry = _write_graph_sidecar(
        src_base, "token_domain_edges", offsets=[0, 5, 2], rows=rows, item_count=5  # goes down
    )
    copied = {
        "offsets_path": f"{dst_base.name}_token_domain_edges_offsets.bin",
        "data_path": f"{dst_base.name}_token_domain_edges_data.bin",
    }
    with pytest.raises(ValueError, match="monotonic"):
        module._copy_graph_sidecar(
            src_base=src_base,
            dst_base=dst_base,
            src_manifest_entry=entry,
            dst_manifest_entry=copied,
            document_count=2,
        )


def test_final_offset_exceeds_declared_raises(tmp_path: Path):
    module = _load_module()
    src_base = tmp_path / "src"
    dst_base = tmp_path / "dst"
    rows = np.arange(5 * 3, dtype=np.int32).reshape(5, 3)  # file + declared item_count = 5
    # size matches (5 items) and offsets are monotonic, but final offset 8 > declared 5.
    entry = _write_graph_sidecar(
        src_base, "token_domain_edges", offsets=[0, 2, 8], rows=rows, item_count=5
    )
    copied = {
        "offsets_path": f"{dst_base.name}_token_domain_edges_offsets.bin",
        "data_path": f"{dst_base.name}_token_domain_edges_data.bin",
    }
    with pytest.raises(ValueError, match="exceeds declared"):
        module._copy_graph_sidecar(
            src_base=src_base,
            dst_base=dst_base,
            src_manifest_entry=entry,
            dst_manifest_entry=copied,
            document_count=2,
        )
