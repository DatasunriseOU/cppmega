from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


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


def test_megatron_dtype_codes_match_mmididx_enum() -> None:
    converter = _load_converter_module()

    assert converter._megatron_dtype_code(np.uint8) == 1
    assert converter._megatron_dtype_code(np.int32) == 4
    assert converter._megatron_dtype_code(np.int64) == 5
    assert converter._megatron_dtype_code(np.uint16) == 8


def test_numpy_uint32_index_dtype_fails_closed() -> None:
    converter = _load_converter_module()

    with pytest.raises(ValueError, match="unsupported Megatron MMIDIDX dtype uint32"):
        converter._megatron_dtype_code(np.uint32)


def test_legacy_uint32_cli_dtype_is_explicit_int32_alias(
    capsys: pytest.CaptureFixture[str],
) -> None:
    converter = _load_converter_module()

    dtype = converter._resolve_output_dtype("uint32")

    assert dtype is np.int32
    assert converter._megatron_dtype_code(dtype) == 4
    assert "no uint32 dtype code" in capsys.readouterr().err


def test_side_channel_length_mismatch_fails_closed() -> None:
    converter = _load_converter_module()

    with pytest.raises(ValueError, match="token_def_use.*length 2.*token_ids length 3"):
        converter._require_token_aligned_side_channel(
            "token_def_use",
            [1, 0],
            [10, 11, 12],
            shard_path="shard_00000.parquet",
            row_idx=7,
        )


def test_default_cppmega_side_channels_are_full_token_aligned_profile() -> None:
    converter = _load_converter_module()

    names = [name for name, _ in converter.DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]
    dtypes = dict(converter.DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS)

    assert names == [
        "loss_mask",
        "doc_ids",
        "token_domain_ids",
        "token_role_ids",
        "token_entity_ids",
        "token_scope_ids",
        "token_source_doc_ids",
        "token_confidence_ids",
        "token_structure_ids",
        "token_dep_levels",
        "token_ast_depth",
        "token_sibling_index",
        "token_ast_node_type",
        "token_symbol_ids",
        "token_call_targets",
        "token_type_refs",
        "token_def_use",
        "token_change_mask_pre",
        "token_change_mask_post",
        "token_platform_ids",
    ]
    assert dtypes["loss_mask"] == "uint8"
    assert dtypes["doc_ids"] == "uint32"
    assert dtypes["token_domain_ids"] == "uint16"
    assert dtypes["token_role_ids"] == "uint16"
    assert dtypes["token_entity_ids"] == "uint32"
    assert dtypes["token_confidence_ids"] == "uint8"
    assert dtypes["token_symbol_ids"] == "uint32"
    assert dtypes["token_def_use"] == "uint8"


def test_default_cppmega_graph_sidecars_are_document_aligned_route_profile() -> None:
    converter = _load_converter_module()

    assert converter.DEFAULT_CPPMEGA_GRAPH_SIDECARS == (
        ("token_call_edges", "edge_pairs", "int32"),
        ("token_type_edges", "edge_pairs", "int32"),
        ("token_domain_edges", "edge_triples", "int32"),
        ("token_build_edges", "edge_triples", "int32"),
        ("token_shell_edges", "edge_triples", "int32"),
        ("token_diagnostic_edges", "edge_triples", "int32"),
        ("token_cross_domain_edges", "edge_triples", "int32"),
        ("token_chunk_starts", "ragged_1d", "uint32"),
        ("token_chunk_ends", "ragged_1d", "uint32"),
        ("token_chunk_kinds", "ragged_1d", "uint16"),
        ("token_chunk_dep_levels", "ragged_1d", "uint16"),
    )


def test_normalize_edge_pairs_accepts_parquet_struct_dicts() -> None:
    converter = _load_converter_module()

    pairs = converter._normalize_edge_pairs(
        [{"from": 3, "to": 1}, {"from": 2, "to": 0}],
        column="token_call_edges",
        shard_path="shard.parquet",
        row_idx=9,
    )

    assert pairs.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(pairs, np.array([[3, 1], [2, 0]], dtype=np.int32))


def test_normalize_edge_triples_accepts_domain_route_dicts() -> None:
    converter = _load_converter_module()

    triples = converter._normalize_edge_triples(
        [{"from": 3, "to": 1, "kind": 20}, {"src": 2, "dst": 0, "kind": 60}],
        column="token_domain_edges",
        shard_path="shard.parquet",
        row_idx=9,
    )

    assert triples.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        triples,
        np.array([[3, 1, 20], [2, 0, 60]], dtype=np.int32),
    )


def test_graph_sidecar_writer_writes_offsets_data_and_manifest(tmp_path: Path) -> None:
    converter = _load_converter_module()
    prefix = tmp_path / "cppmega_train"
    writer = converter._GraphSidecarWriters(
        str(prefix),
        (
            ("token_call_edges", "edge_pairs", "int32"),
            ("token_domain_edges", "edge_triples", "int32"),
            ("token_chunk_starts", "ragged_1d", "uint32"),
        ),
    )

    writer.append(
        {
            "token_call_edges": [{"from": 1, "to": 0}, {"from": 2, "to": 1}],
            "token_domain_edges": [{"from": 0, "to": 3, "kind": 20}],
            "token_chunk_starts": [0, 8, 16],
        },
        shard_path="shard.parquet",
        row_idx=0,
    )
    writer.append(
        {"token_call_edges": [], "token_domain_edges": [], "token_chunk_starts": [0]},
        shard_path="shard.parquet",
        row_idx=1,
    )
    manifest = writer.close()

    assert manifest["token_call_edges"] == {
        "kind": "edge_pairs",
        "offsets_path": "cppmega_train_token_call_edges_offsets.bin",
        "data_path": "cppmega_train_token_call_edges_data.bin",
        "offset_dtype": "int64",
        "dtype": "int32",
        "item_count": 2,
        "shape_tail": [2],
    }
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "cppmega_train_token_call_edges_offsets.bin", dtype=np.int64),
        np.array([0, 2, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "cppmega_train_token_call_edges_data.bin", dtype=np.int32).reshape(-1, 2),
        np.array([[1, 0], [2, 1]], dtype=np.int32),
    )
    assert manifest["token_domain_edges"]["shape_tail"] == [3]
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "cppmega_train_token_domain_edges_offsets.bin", dtype=np.int64),
        np.array([0, 1, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "cppmega_train_token_domain_edges_data.bin", dtype=np.int32).reshape(-1, 3),
        np.array([[0, 3, 20]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "cppmega_train_token_chunk_starts_offsets.bin", dtype=np.int64),
        np.array([0, 3, 4], dtype=np.int64),
    )
    starts = np.fromfile(tmp_path / "cppmega_train_token_chunk_starts_data.bin", dtype=np.uint32)
    np.testing.assert_array_equal(starts, np.array([0, 8, 16, 0], dtype=np.uint32))
    json.dumps(manifest)


def test_find_parquet_shards_all_keeps_every_file(tmp_path: Path) -> None:
    converter = _load_converter_module()

    for name in ("a.parquet", "b.parquet", "val_shard.parquet"):
        (tmp_path / name).write_bytes(b"not a real parquet")

    shards = converter.find_parquet_shards(str(tmp_path), "all")

    assert [Path(shard).name for shard in shards] == [
        "a.parquet",
        "b.parquet",
        "val_shard.parquet",
    ]
