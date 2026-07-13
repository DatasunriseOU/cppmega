from __future__ import annotations

import importlib.util
import json
import sys
import types
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
    ]
    assert dtypes["loss_mask"] == "uint8"
    assert dtypes["doc_ids"] == "uint16"
    assert dtypes["token_domain_ids"] == "uint16"
    assert dtypes["token_role_ids"] == "uint16"
    assert dtypes["token_entity_ids"] == "uint32"
    assert dtypes["token_confidence_ids"] == "uint8"
    assert dtypes["token_symbol_ids"] == "uint32"
    assert dtypes["token_def_use"] == "uint8"


def test_source_platform_writer_preserves_multi_label_document_context(
    tmp_path: Path,
) -> None:
    converter = _load_converter_module()
    prefix = tmp_path / "cppmega_train"
    writer = converter._SourcePlatformSidecarWriter(str(prefix))

    writer.append(
        [[2, 62, 93, 109], [3, 64, 94, 111]],
        doc_ids=[1, 1, 1, 2, 2, 2, 2, 2],
        token_count=8,
        shard_path="shard.parquet",
        row_idx=0,
    )
    writer.append(
        [[2, 62, 94]],
        doc_ids=[1, 1, 1],
        token_count=3,
        shard_path="shard.parquet",
        row_idx=1,
    )
    manifest = writer.close()

    assert manifest["schema"] == "cppmega_source_platform_v1"
    assert manifest["source_document_count"] == 3
    assert manifest["platform_id_count"] == 11
    np.testing.assert_array_equal(
        np.fromfile(
            tmp_path / "cppmega_train_source_platform_sequence_doc_offsets.bin",
            dtype=np.int64,
        ),
        np.array([0, 2, 3], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.fromfile(
            tmp_path / "cppmega_train_source_platform_doc_id_offsets.bin",
            dtype=np.int64,
        ),
        np.array([0, 4, 8, 11], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.fromfile(
            tmp_path / "cppmega_train_source_platform_ids.bin",
            dtype=np.uint16,
        ),
        np.array([2, 62, 93, 109, 3, 64, 94, 111, 2, 62, 94], dtype=np.uint16),
    )


def test_source_platform_writer_rejects_nonlocal_document_ids(tmp_path: Path) -> None:
    converter = _load_converter_module()
    writer = converter._SourcePlatformSidecarWriter(str(tmp_path / "bad"))

    with pytest.raises(ValueError, match="row-local IDs 1..2"):
        writer.append(
            [[2], [3]],
            doc_ids=[7, 7, 8, 8],
            token_count=4,
            shard_path="shard.parquet",
            row_idx=0,
        )
    writer.abort_close()


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
        "coordinate_space": "chunk_index",
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
    assert manifest["token_domain_edges"]["coordinate_space"] == "token_index"
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


def test_graph_sidecar_writer_rejects_route_past_trimmed_row(tmp_path: Path) -> None:
    converter = _load_converter_module()
    writer = converter._GraphSidecarWriters(
        str(tmp_path / "bad"),
        (("token_domain_edges", "edge_triples", "int32"),),
    )

    with pytest.raises(ValueError, match="endpoint exceeds valid token count 3"):
        writer.append(
            {"token_domain_edges": [{"from": 0, "to": 3, "kind": 20}]},
            shard_path="shard.parquet",
            row_idx=0,
            token_count=3,
        )
    writer.abort_close()


def test_graph_sidecar_writer_rejects_chunk_edge_past_chunk_count(tmp_path: Path) -> None:
    converter = _load_converter_module()
    writer = converter._GraphSidecarWriters(
        str(tmp_path / "bad_chunk_edge"),
        (
            ("token_call_edges", "edge_pairs", "int32"),
            ("token_chunk_starts", "ragged_1d", "uint32"),
        ),
    )

    with pytest.raises(ValueError, match="endpoint exceeds chunk count 2"):
        writer.append(
            {
                "token_call_edges": [{"from": 0, "to": 2}],
                "token_chunk_starts": [0, 2],
            },
            shard_path="shard.parquet",
            row_idx=0,
            token_count=4,
        )
    writer.abort_close()


def test_graph_sidecar_writer_rejects_misaligned_chunk_arrays(tmp_path: Path) -> None:
    converter = _load_converter_module()
    writer = converter._GraphSidecarWriters(
        str(tmp_path / "bad_chunks"),
        (
            ("token_chunk_starts", "ragged_1d", "uint32"),
            ("token_chunk_ends", "ragged_1d", "uint32"),
            ("token_chunk_kinds", "ragged_1d", "uint16"),
        ),
    )

    with pytest.raises(ValueError, match=r"token_chunk_\* sidecars must have equal lengths"):
        writer.append(
            {
                "token_chunk_starts": [0, 2],
                "token_chunk_ends": [2],
                "token_chunk_kinds": [1, 1],
            },
            shard_path="shard.parquet",
            row_idx=0,
            token_count=4,
        )
    writer.abort_close()


def test_explicit_mmididx_writer_trims_padding_and_all_sidecars(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    converter = _load_converter_module()
    input_dir = tmp_path / "parquet"
    input_dir.mkdir()
    row = {
        "valid_token_count": [3],
        "input_ids": [[7, 8, 9, 0, 0]],
        "source_platform_ids": [[[2, 62, 93, 109]]],
    }
    for name, _dtype in converter.DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS:
        row[name] = [[1, 1, 0, 0, 0]]
    row["doc_ids"] = [[1, 1, 1, 1, 1]]
    for name, kind, _dtype in converter.DEFAULT_CPPMEGA_GRAPH_SIDECARS:
        if name == "token_domain_edges":
            row[name] = [[{"from": 0, "to": 2, "kind": 20}]]
        elif kind == "edge_pairs":
            row[name] = [[]]
        elif kind == "edge_triples":
            row[name] = [[]]
        elif name == "token_chunk_starts":
            row[name] = [[0]]
        elif name == "token_chunk_ends":
            row[name] = [[3]]
        else:
            row[name] = [[1]]
    pq.write_table(pa.table(row), input_dir / "repo.parquet")

    output_prefix = tmp_path / "cppmega_1024_train"
    converter.convert_parquet_to_megatron(
        input_dir=str(input_dir),
        output_prefix=str(output_prefix),
        split="all",
        token_column="auto",
        length_column="auto",
        writer_backend="mmididx",
    )

    np.testing.assert_array_equal(
        np.fromfile(output_prefix.with_suffix(".bin"), dtype=np.uint16),
        np.array([7, 8, 9], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "cppmega_1024_train_loss_mask.bin", dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
    )
    manifest = json.loads(output_prefix.with_suffix(".json").read_text())
    assert manifest["token_count"] == 3
    assert manifest["source_capacity_token_count"] == 5
    assert manifest["trained_token_count"] == 2
    assert manifest["document_count"] == 1
    assert manifest["token_column"] == "input_ids"
    assert manifest["length_column"] == "valid_token_count"
    assert manifest["writer_backend"] == "mmididx"
    assert set(manifest["side_channel_paths"]) == {
        name for name, _dtype in converter.DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS
    }
    assert set(manifest["graph_sidecar_paths"]) == {
        name for name, _kind, _dtype in converter.DEFAULT_CPPMEGA_GRAPH_SIDECARS
    }
    assert manifest["source_platform_sidecar"]["schema"] == (
        "cppmega_source_platform_v1"
    )


def test_mmididx_row_group_batching_preserves_document_order_and_offsets(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    converter = _load_converter_module()
    input_dir = tmp_path / "parquet"
    input_dir.mkdir()
    rows: dict[str, object] = {
        "valid_token_count": [2, 4, 1],
        "input_ids": [[10, 11, 0, 0], [20, 21, 22, 23], [30, 0, 0, 0]],
        "source_platform_ids": [[[2]], [[3], [4]], [[5]]],
    }
    for name, _dtype in converter.DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS:
        rows[name] = [
            [1, 2, 90, 90],
            [3, 4, 5, 6],
            [7, 90, 90, 90],
        ]
    rows["loss_mask"] = [[1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]]
    rows["doc_ids"] = [[1, 1, 0, 0], [1, 1, 2, 2], [1, 0, 0, 0]]
    for name, kind, _dtype in converter.DEFAULT_CPPMEGA_GRAPH_SIDECARS:
        if name == "token_call_edges":
            rows[name] = [[], [{"from": 0, "to": 1}], []]
        elif kind in {"edge_pairs", "edge_triples"}:
            rows[name] = [[], [], []]
        elif name == "token_chunk_starts":
            rows[name] = [[0], [0, 2], [0]]
        elif name == "token_chunk_ends":
            rows[name] = [[2], [2, 4], [1]]
        else:
            rows[name] = [[1], [1, 1], [1]]
    pq.write_table(
        pa.table(rows), input_dir / "repo.parquet", row_group_size=2
    )

    output_prefix = tmp_path / "cppmega_train"
    converter.convert_parquet_to_megatron(
        input_dir=str(input_dir),
        output_prefix=str(output_prefix),
        split="all",
        token_column="auto",
        length_column="auto",
        writer_backend="mmididx",
    )

    np.testing.assert_array_equal(
        np.fromfile(output_prefix.with_suffix(".bin"), dtype=np.uint16),
        np.array([10, 11, 20, 21, 22, 23, 30], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "cppmega_train_loss_mask.bin", dtype=np.uint8),
        np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.uint8),
    )
    with output_prefix.with_suffix(".idx").open("rb") as idx:
        idx.seek(34)
        np.testing.assert_array_equal(
            np.fromfile(idx, dtype=np.int32, count=3), np.array([2, 4, 1])
        )
        np.testing.assert_array_equal(
            np.fromfile(idx, dtype=np.int64, count=3), np.array([0, 4, 12])
        )
    np.testing.assert_array_equal(
        np.fromfile(
            tmp_path / "cppmega_train_token_call_edges_offsets.bin",
            dtype=np.int64,
        ),
        np.array([0, 0, 1, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.fromfile(
            tmp_path / "cppmega_train_token_call_edges_data.bin",
            dtype=np.int32,
        ),
        np.array([0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.fromfile(
            tmp_path / "cppmega_train_source_platform_sequence_doc_offsets.bin",
            dtype=np.int64,
        ),
        np.array([0, 1, 3, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.fromfile(
            tmp_path / "cppmega_train_source_platform_ids.bin", dtype=np.uint16
        ),
        np.array([2, 3, 4, 5], dtype=np.uint16),
    )


def test_auto_token_column_fails_on_ambiguous_schema(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    converter = _load_converter_module()
    pq.write_table(
        pa.table({"input_ids": [[1]], "token_ids": [[1]]}),
        tmp_path / "ambiguous.parquet",
    )

    with pytest.raises(ValueError, match="exactly one of input_ids/token_ids"):
        converter._convert_parquet_to_numpy(
            input_dir=str(tmp_path),
            output_prefix=str(tmp_path / "out"),
            split="all",
            token_column="auto",
            dtype_str="uint16",
            side_channels=[],
            side_channel_dtypes=[],
            graph_sidecars=None,
        )


def test_missing_megatron_import_fails_loud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    converter = _load_converter_module()

    def _forbidden_numpy_fallback(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "silent numpy fallback writer must not be used when the Megatron "
            "import fails"
        )

    monkeypatch.setattr(
        converter, "_convert_parquet_to_numpy", _forbidden_numpy_fallback
    )
    # Stub pyarrow so the function reaches the megatron import (pyarrow is
    # imported first and is not needed before the megatron import raises).
    pyarrow_stub = types.ModuleType("pyarrow")
    parquet_stub = types.ModuleType("pyarrow.parquet")
    pyarrow_stub.parquet = parquet_stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow_stub)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet_stub)
    # Simulate a missing/broken Megatron-Core install: None in sys.modules makes
    # `from megatron.core... import ...` raise ImportError at call time.
    monkeypatch.setitem(sys.modules, "megatron", None)

    with pytest.raises(RuntimeError, match="IndexedDatasetBuilder"):
        converter.convert_parquet_to_megatron(
            input_dir=str(tmp_path),
            output_prefix=str(tmp_path / "cppmega_train"),
            split="train",
        )


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
