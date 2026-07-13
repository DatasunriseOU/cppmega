from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from cppmega.megatron.objective_contract import (
    OBJECTIVE_CONTRACT_SCHEMA,
    OBJECTIVE_GRAPH_SIDECARS,
    OBJECTIVE_IDS,
    OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA,
    OBJECTIVE_TOKEN_SIDE_CHANNELS,
    validate_materialized_objective_contract,
    validate_objective_contract,
)


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


def _stamp_v3_identity_table(pa, table, converter):
    for column in ("token_symbol_ids", "token_call_targets", "token_type_refs"):
        index = table.schema.get_field_index(column)
        table = table.set_column(
            index,
            column,
            pa.array(table.column(column).to_pylist(), type=pa.list_(pa.uint64())),
        )
    table = table.append_column(
        "symbol_identities",
        pa.array(
            [[] for _ in range(table.num_rows)],
            type=pa.list_(
                pa.struct(
                    [
                        pa.field("symbol_id", pa.uint64()),
                        pa.field("symbol_key", pa.string()),
                    ]
                )
            ),
        ),
    )
    return table.replace_schema_metadata(
        {converter.SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode(): b"3"}
    )


def _objective_contract() -> dict[str, object]:
    tasks = ("causal_lm", "fim", "ast_fim", "ifim", "commit_diff", "pre_to_post")
    return {
        "schema": OBJECTIVE_CONTRACT_SCHEMA,
        "algorithm": "hamilton_eligibility_bipartite_v1",
        "seed": 17,
        "quota_window_samples": 6,
        "task_order": list(tasks),
        "objective_ids": {task: OBJECTIVE_IDS[task] for task in tasks},
        "configured_rates": {task: "1/6" for task in tasks},
        "planned_samples": {task: 1 for task in tasks},
        "realized": {
            task: {
                "samples": 1,
                "input_tokens": 3,
                "loss_tokens": 3 if task == "causal_lm" else 2,
            }
            for task in tasks
        },
        "totals": {"samples": 6, "input_tokens": 18, "loss_tokens": 13},
        "typed_sources": {
            "ifim_instruction": "ifim_instruction_token_ids",
            "commit_message": "commit_msg_token_ids",
            "diff": "diff_token_ids",
            "pre": "pre_token_ids",
            "post": "post_token_ids",
            "missing_fields": "ineligible",
            "rendered_text_parsing": False,
        },
        "graph_auxiliary": {
            "relations": ["call", "type"],
            "eligible_samples": 1,
            "positive_edges": 5,
            "global_weight": "1",
            "indexer_weight": "1/1000",
            "layer_weight": "1",
            "layer_reduction": "sum",
            "bce_weight": "1/10",
            "coverage_weight": "1/20",
            "topk": 8,
            "pos_weight": "1",
            "margin": "1",
            "included_in_total_loss": True,
            "runtime": "megatron_dsa_indexer_v1",
            "pair_mask": "causal_same_document_upstream_v1",
            "chunk_edge_expansion": "cartesian_token_spans_v1",
        },
        "materialization": {
            "format": "shifted_lm_document_v1",
            "token_column": "input_ids",
            "loss_mask_column": "loss_mask",
            "length_column": "valid_token_count",
            "objective_column": "objective_kind",
            "document_id_column": "doc_ids",
            "source_document_id_column": "token_source_doc_ids",
        },
    }


def _write_objective_artifact(input_dir: Path) -> Path:
    contract = _objective_contract()
    contract_path = input_dir / "objective_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    shards = sorted(input_dir.glob("*.parquet"))
    artifact = {
        "schema": OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA,
        "documents": contract["totals"]["samples"],  # type: ignore[index]
        "objective_contract": {
            "path": contract_path.name,
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "size_bytes": contract_path.stat().st_size,
            "file_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        },
        "parquet_shards": [
            {
                "path": shard.name,
                "size_bytes": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
            for shard in shards
        ],
        "converter": {
            "split": "all",
            "token_column": "input_ids",
            "length_column": "valid_token_count",
            "side_channels": [
                {"column": column, "dtype": dtype}
                for column, dtype in OBJECTIVE_TOKEN_SIDE_CHANNELS
            ],
            "graph_sidecars": [
                {"column": column, "kind": kind, "dtype": dtype}
                for column, kind, dtype in OBJECTIVE_GRAPH_SIDECARS
            ],
            "source_platform_sidecar": "require",
            "graph_relations": ["call", "type"],
            "graph_pair_mask": "causal_same_document_upstream_v1",
            "chunk_edge_expansion": "cartesian_token_spans_v1",
        },
    }
    artifact["artifact_set_sha256"] = hashlib.sha256(
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    artifact_path = input_dir / "objective_materialization.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path


def test_objective_conversion_rejects_materialized_column_drift() -> None:
    converter = _load_converter_module()
    contract = converter.validate_objective_contract(_objective_contract())

    with pytest.raises(ValueError, match="token column"):
        converter._validate_objective_conversion_config(
            contract,
            token_column="token_ids",
            length_column="valid_token_count",
            side_channels=["loss_mask"],
            graph_columns=["token_call_edges", "token_type_edges"],
        )


def test_objective_conversion_requires_document_id_sidecar() -> None:
    converter = _load_converter_module()
    contract = converter.validate_objective_contract(_objective_contract())

    with pytest.raises(ValueError, match="doc_ids"):
        converter._validate_objective_conversion_config(
            contract,
            token_column="input_ids",
            length_column="valid_token_count",
            side_channels=["loss_mask"],
            graph_columns=["token_call_edges", "token_type_edges"],
        )


def test_objective_conversion_rejects_bare_contract(tmp_path: Path) -> None:
    converter = _load_converter_module()
    contract_path = tmp_path / "objective_contract.json"
    contract_path.write_text(json.dumps(_objective_contract()), encoding="utf-8")

    with pytest.raises(ValueError, match="bare --objective-contract"):
        converter.convert_parquet_to_megatron(
            input_dir=str(tmp_path),
            output_prefix=str(tmp_path / "train"),
            objective_contract_path=str(contract_path),
            writer_backend="mmididx",
        )


def test_objective_contract_accepts_deterministic_zero_hamilton_quota() -> None:
    contract = _objective_contract()
    contract["quota_window_samples"] = 3
    contract["totals"] = {"samples": 3, "input_tokens": 9, "loss_tokens": 8}
    contract["planned_samples"] = {
        "causal_lm": 1,
        "fim": 1,
        "ast_fim": 1,
        "ifim": 0,
        "commit_diff": 0,
        "pre_to_post": 0,
    }
    contract["realized"] = {
        task: {
            "samples": samples,
            "input_tokens": samples * 3,
            "loss_tokens": samples * 3 - (1 if task == "fim" else 0),
        }
        for task, samples in contract["planned_samples"].items()
    }

    assert validate_objective_contract(contract).planned_samples["ifim"] == 0


def test_materialized_objective_contract_rejects_id_histogram_drift(tmp_path: Path) -> None:
    contract = validate_objective_contract(_objective_contract())
    sidecar = tmp_path / "objective_ids.bin"
    np.asarray([1, 1, 3, 4, 5, 6], dtype=np.uint8).tofile(sidecar)
    wrapper = {
        "schema": OBJECTIVE_CONTRACT_SCHEMA,
        "sha256": contract.sha256,
        "payload": contract.payload,
        "objective_id_sidecar": {
            "path": sidecar.name,
            "dtype": "uint8",
            "document_aligned": True,
        },
    }

    with pytest.raises(ValueError, match="histogram differs"):
        validate_materialized_objective_contract(
            wrapper, base_dir=str(tmp_path), document_count=6
        )


def test_materialized_objective_contract_rejects_unknown_id(tmp_path: Path) -> None:
    contract = validate_objective_contract(_objective_contract())
    sidecar = tmp_path / "objective_ids.bin"
    np.asarray([1, 2, 3, 4, 5, 255], dtype=np.uint8).tofile(sidecar)
    wrapper = {
        "schema": OBJECTIVE_CONTRACT_SCHEMA,
        "sha256": contract.sha256,
        "payload": contract.payload,
        "objective_id_sidecar": {
            "path": sidecar.name,
            "dtype": "uint8",
            "document_aligned": True,
        },
    }

    with pytest.raises(ValueError, match="unknown objective IDs"):
        validate_materialized_objective_contract(
            wrapper, base_dir=str(tmp_path), document_count=6
        )


def test_objective_conversion_requires_source_document_provenance() -> None:
    converter = _load_converter_module()
    contract = converter.validate_objective_contract(_objective_contract())

    with pytest.raises(ValueError, match="token_source_doc_ids"):
        converter._validate_objective_conversion_config(
            contract,
            token_column="input_ids",
            length_column="valid_token_count",
            side_channels=["loss_mask", "doc_ids"],
            graph_columns=["token_call_edges", "token_type_edges"],
        )


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
    ]
    assert dtypes["loss_mask"] == "uint8"
    assert dtypes["doc_ids"] == "uint16"
    assert dtypes["token_domain_ids"] == "uint16"
    assert dtypes["token_role_ids"] == "uint16"
    assert dtypes["token_entity_ids"] == "uint32"
    assert dtypes["token_source_doc_ids"] == "uint32"
    assert dtypes["token_confidence_ids"] == "uint8"
    assert dtypes["token_symbol_ids"] == "uint64"
    assert dtypes["token_call_targets"] == "uint64"
    assert dtypes["token_type_refs"] == "uint64"
    assert dtypes["token_def_use"] == "uint8"


def test_mmididx_preserves_symbol_id_above_signed_int64(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    converter = _load_converter_module()
    key_index = 0
    while True:
        symbol_key = f"usr:owner/repo:c:@F@wide_id_{key_index}#"
        symbol_id = converter._compute_symbol_id(symbol_key)
        if symbol_id > np.iinfo(np.int64).max:
            break
        key_index += 1

    input_dir = tmp_path / "parquet"
    input_dir.mkdir()
    schema = pa.schema(
        [
            pa.field("input_ids", pa.list_(pa.uint32())),
            pa.field("token_symbol_ids", pa.list_(pa.uint64())),
            pa.field("token_call_targets", pa.list_(pa.uint64())),
            pa.field("token_type_refs", pa.list_(pa.uint64())),
            pa.field("token_def_use", pa.list_(pa.uint8())),
            pa.field(
                "symbol_identities",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("symbol_id", pa.uint64()),
                            pa.field("symbol_key", pa.string()),
                        ]
                    )
                ),
            ),
        ],
        metadata={
            converter.SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii"): b"3"
        },
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "input_ids": [7, 8],
                    "token_symbol_ids": [symbol_id, symbol_id],
                    "token_call_targets": [0, symbol_id],
                    "token_type_refs": [symbol_id, 0],
                    "token_def_use": [0, 1],
                    "symbol_identities": [
                        {"symbol_id": symbol_id, "symbol_key": symbol_key}
                    ],
                }
            ],
            schema=schema,
        ),
        input_dir / "wide.parquet",
    )

    output_prefix = tmp_path / "wide_train"
    converter.convert_parquet_to_megatron(
        input_dir=str(input_dir),
        output_prefix=str(output_prefix),
        split="all",
        token_column="input_ids",
        side_channels=[
            "token_symbol_ids",
            "token_call_targets",
            "token_type_refs",
            "token_def_use",
        ],
        side_channel_dtypes=["uint64", "uint64", "uint64", "uint8"],
        graph_sidecars=None,
        source_platform_sidecar=False,
        writer_backend="mmididx",
    )

    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "wide_train_token_symbol_ids.bin", dtype=np.uint64),
        np.array([symbol_id, symbol_id], dtype=np.uint64),
    )
    manifest = json.loads(output_prefix.with_suffix(".json").read_text())
    assert manifest["side_channel_paths"]["token_symbol_ids"]["dtype"] == "uint64"
    assert manifest["symbol_identity_schema_version"] == 3


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
        [{"from": 3, "to": 1, "kind": 5}, {"src": 2, "dst": 0, "kind": 8}],
        column="token_domain_edges",
        shard_path="shard.parquet",
        row_idx=9,
    )

    assert triples.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        triples,
        np.array([[3, 1, 5], [2, 0, 8]], dtype=np.int32),
    )


def test_normalize_edge_triples_rejects_wrong_family_kind_26() -> None:
    converter = _load_converter_module()

    with pytest.raises(ValueError, match="not valid for token_domain_edges"):
        converter._normalize_edge_triples(
            [{"from": 0, "to": 1, "kind": 26}],
            column="token_domain_edges",
            shard_path="shard.parquet",
            row_idx=4,
        )


def test_normalize_edge_triples_rejects_unknown_domain_route_kind() -> None:
    converter = _load_converter_module()

    with pytest.raises(ValueError, match="unknown domain route edge kind 9999"):
        converter._normalize_edge_triples(
            [{"from": 0, "to": 1, "kind": 9999}],
            column="token_domain_edges",
            shard_path="shard.parquet",
            row_idx=4,
        )


def test_normalize_edge_triples_rejects_missing_fields_and_wrong_edge_family() -> None:
    converter = _load_converter_module()

    with pytest.raises(ValueError, match="missing src/dst/kind"):
        converter._normalize_edge_triples(
            [{"from": 0, "kind": 20}],
            column="token_build_edges",
            shard_path="shard.parquet",
            row_idx=4,
        )

    with pytest.raises(ValueError, match="edge kind 60 is not valid for token_build_edges"):
        converter._normalize_edge_triples(
            [{"from": 0, "to": 1, "kind": 60}],
            column="token_build_edges",
            shard_path="shard.parquet",
            row_idx=4,
        )


def test_domain_route_sidecars_validate_enums_delimiters_and_source_ids() -> None:
    converter = _load_converter_module()
    values = {
        "token_domain_ids": [1, 1, 1],
        "token_role_ids": [1, 2, 1],
        "token_entity_ids": [0, 7, 0],
        "token_scope_ids": [0, 0, 0],
        "token_source_doc_ids": [9, 9, 9],
        "token_confidence_ids": [4, 2, 4],
    }

    converter._validate_domain_route_sidecars(
        [191, 1000, 192],
        values,
        shard_path="shard.parquet",
        row_idx=3,
    )

    bad_domain = dict(values)
    bad_domain["token_domain_ids"] = [1, 999, 1]
    with pytest.raises(ValueError, match="unknown token_domain_ids value 999"):
        converter._validate_domain_route_sidecars(
            [191, 1000, 192],
            bad_domain,
            shard_path="shard.parquet",
            row_idx=3,
        )

    wrong_delimiter_domain = dict(values)
    wrong_delimiter_domain["token_domain_ids"] = [2, 1, 1]
    with pytest.raises(ValueError, match="delimiter token id 191 requires domain id 1"):
        converter._validate_domain_route_sidecars(
            [191, 1000, 192],
            wrong_delimiter_domain,
            shard_path="shard.parquet",
            row_idx=3,
        )

    missing_source = dict(values)
    missing_source.pop("token_source_doc_ids")
    with pytest.raises(ValueError, match="complete token-aligned domain route profile"):
        converter._validate_domain_route_sidecars(
            [191, 1000, 192],
            missing_source,
            shard_path="shard.parquet",
            row_idx=3,
        )


def test_domain_route_sidecars_validate_nested_sql_and_uint32_bounds() -> None:
    converter = _load_converter_module()
    nested = {
        "token_domain_ids": [1, 30, 30, 30, 1],
        "token_role_ids": [1, 1, 2, 1, 1],
        "token_entity_ids": [0, 0, 7, 0, 0],
        "token_scope_ids": [0, 0, 0, 0, 0],
        "token_source_doc_ids": [9, 9, 9, 9, 9],
        "token_confidence_ids": [4, 4, 2, 4, 4],
    }

    converter._validate_domain_route_sidecars(
        [191, 239, 1000, 240, 192],
        nested,
        shard_path="shard.parquet",
        row_idx=5,
    )

    oversized_entity = dict(nested)
    oversized_entity["token_entity_ids"] = [0, 0, 2**32, 0, 0]
    with pytest.raises(ValueError, match="token_entity_ids must fit uint32"):
        converter._validate_domain_route_sidecars(
            [191, 239, 1000, 240, 192],
            oversized_entity,
            shard_path="shard.parquet",
            row_idx=5,
        )

    oversized_source = dict(nested)
    oversized_source["token_source_doc_ids"] = [9, 9, 2**32, 9, 9]
    with pytest.raises(ValueError, match="token_source_doc_ids must fit uint32"):
        converter._validate_domain_route_sidecars(
            [191, 239, 1000, 240, 192],
            oversized_source,
            shard_path="shard.parquet",
            row_idx=5,
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
            "token_domain_edges": [{"from": 0, "to": 3, "kind": 5}],
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
        np.array([[0, 3, 5]], dtype=np.int32),
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
            {"token_domain_edges": [{"from": 0, "to": 3, "kind": 5}]},
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
    for name in ("token_symbol_ids", "token_call_targets", "token_type_refs"):
        row[name] = [[0, 0, 0, 0, 0]]
    row["token_domain_ids"] = [[0, 0, 0, 0, 0]]
    row["token_role_ids"] = [[0, 0, 0, 0, 0]]
    row["token_entity_ids"] = [[0, 0, 0, 0, 0]]
    row["token_scope_ids"] = [[0, 0, 0, 0, 0]]
    row["token_source_doc_ids"] = [[1, 1, 1, 0, 0]]
    row["token_confidence_ids"] = [[0, 0, 0, 0, 0]]
    row["doc_ids"] = [[1, 1, 1, 1, 1]]
    for name, kind, _dtype in converter.DEFAULT_CPPMEGA_GRAPH_SIDECARS:
        if name == "token_domain_edges":
            row[name] = [[{"from": 0, "to": 2, "kind": 5}]]
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
    table = _stamp_v3_identity_table(pa, pa.table(row), converter)
    pq.write_table(table, input_dir / "repo.parquet")

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
    np.testing.assert_array_equal(
        np.fromfile(
            tmp_path / "cppmega_1024_train_token_source_doc_ids.bin",
            dtype=np.uint32,
        ),
        np.array([1, 1, 1], dtype=np.uint32),
    )
    manifest = json.loads(output_prefix.with_suffix(".json").read_text())
    assert manifest["token_count"] == 3
    assert manifest["source_capacity_token_count"] == 5
    assert manifest["trained_token_count"] == 2
    assert manifest["document_count"] == 1
    assert manifest["token_column"] == "input_ids"
    assert manifest["length_column"] == "valid_token_count"
    assert manifest["writer_backend"] == "mmididx"
    assert manifest["symbol_identity_schema_version"] == 3
    assert set(manifest["side_channel_paths"]) == {
        name for name, _dtype in converter.DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS
    }
    assert set(manifest["graph_sidecar_paths"]) == {
        name for name, _kind, _dtype in converter.DEFAULT_CPPMEGA_GRAPH_SIDECARS
    }
    assert manifest["source_platform_sidecar"]["schema"] == (
        "cppmega_source_platform_v1"
    )


def test_mmididx_writer_binds_pre_materialized_objective_contract(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    converter = _load_converter_module()
    tasks = tuple(_objective_contract()["task_order"])
    input_dir = tmp_path / "parquet"
    input_dir.mkdir()
    rows: dict[str, object] = {
        "valid_token_count": [4] * len(tasks),
        "input_ids": [
            [10 + index, 20 + index, 30 + index, 40 + index]
            for index in range(len(tasks))
        ],
        "loss_mask": [
            [1, 1, 1, 0] if task == "causal_lm" else [0, 1, 1, 0]
            for task in tasks
        ],
        "objective_kind": list(tasks),
        "doc_ids": [[1, 1, 1, 1] for _task in tasks],
        "token_source_doc_ids": [[7, 7, 7, 7] for _task in tasks],
        "source_platform_ids": [[[2]] for _task in tasks],
        "token_call_edges": [
            [{"from": 0, "to": 0}, {"from": 1, "to": 0}]
            if task == "causal_lm"
            else []
            for task in tasks
        ],
        "token_type_edges": [[] for _task in tasks],
        "token_chunk_starts": [
            [0, 2] if task == "causal_lm" else [] for task in tasks
        ],
        "token_chunk_ends": [
            [2, 4] if task == "causal_lm" else [] for task in tasks
        ],
        "token_chunk_kinds": [
            [1, 1] if task == "causal_lm" else [] for task in tasks
        ],
        "token_chunk_dep_levels": [
            [0, 0] if task == "causal_lm" else [] for task in tasks
        ],
    }
    for column, _dtype in OBJECTIVE_TOKEN_SIDE_CHANNELS:
        rows.setdefault(column, [[0, 0, 0, 0] for _task in tasks])
    for column, _kind, _dtype in OBJECTIVE_GRAPH_SIDECARS:
        rows.setdefault(column, [[] for _task in tasks])
    table = _stamp_v3_identity_table(pa, pa.table(rows), converter)
    pq.write_table(table, input_dir / "objectives.parquet")
    artifact_path = _write_objective_artifact(input_dir)
    output_prefix = tmp_path / "objective_train"

    converter.convert_parquet_to_megatron(
        input_dir=str(input_dir),
        output_prefix=str(output_prefix),
        split="all",
        token_column="auto",
        length_column="auto",
        objective_artifact_path=str(artifact_path),
        writer_backend="mmididx",
    )

    manifest = json.loads(output_prefix.with_suffix(".json").read_text())
    assert manifest["objective_contract"]["payload"] == _objective_contract()
    assert manifest["objective_materialization"]["artifact_set_sha256"]
    assert manifest["symbol_identity_schema_version"] == 3
    np.testing.assert_array_equal(
        np.fromfile(
            tmp_path / "objective_train_objective_ids.bin", dtype=np.uint8
        ),
        np.array([OBJECTIVE_IDS[task] for task in tasks], dtype=np.uint8),
    )


def test_mmididx_conversion_rejects_invalid_domain_profile_before_success(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    converter = _load_converter_module()
    input_dir = tmp_path / "parquet"
    input_dir.mkdir()
    row: dict[str, object] = {
        "valid_token_count": [2],
        "input_ids": [[10, 11]],
    }
    for name, _dtype in converter.DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS:
        row[name] = [[0, 0]]
    row["doc_ids"] = [[1, 1]]
    row["token_domain_ids"] = [[999, 0]]
    row["token_source_doc_ids"] = [[1, 1]]
    for name, kind, _dtype in converter.DEFAULT_CPPMEGA_GRAPH_SIDECARS:
        if kind in {"edge_pairs", "edge_triples"}:
            row[name] = [[]]
        elif name == "token_chunk_starts":
            row[name] = [[0]]
        elif name == "token_chunk_ends":
            row[name] = [[2]]
        else:
            row[name] = [[1]]
    table = _stamp_v3_identity_table(pa, pa.table(row), converter)
    pq.write_table(table, input_dir / "bad.parquet")

    with pytest.raises(ValueError, match="unknown token_domain_ids value 999"):
        converter.convert_parquet_to_megatron(
            input_dir=str(input_dir),
            output_prefix=str(tmp_path / "bad_train"),
            split="all",
            token_column="auto",
            length_column="auto",
            writer_backend="mmididx",
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
    for name in ("token_symbol_ids", "token_call_targets", "token_type_refs"):
        rows[name] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for name in (
        "token_domain_ids",
        "token_role_ids",
        "token_entity_ids",
        "token_scope_ids",
        "token_confidence_ids",
    ):
        rows[name] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    rows["token_source_doc_ids"] = [
        [11, 11, 0, 0],
        [21, 21, 22, 22],
        [31, 0, 0, 0],
    ]
    rows["loss_mask"] = [[1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]]
    rows["doc_ids"] = [[1, 1, 0, 0], [1, 1, 2, 2], [1, 0, 0, 0]]
    for name, kind, _dtype in converter.DEFAULT_CPPMEGA_GRAPH_SIDECARS:
        if name == "token_call_edges":
            rows[name] = [[], [{"from": 0, "to": 0}], []]
        elif kind in {"edge_pairs", "edge_triples"}:
            rows[name] = [[], [], []]
        elif name == "token_chunk_starts":
            rows[name] = [[0], [0, 2], [0]]
        elif name == "token_chunk_ends":
            rows[name] = [[2], [2, 4], [1]]
        else:
            rows[name] = [[1], [1, 1], [1]]
    table = _stamp_v3_identity_table(pa, pa.table(rows), converter)
    pq.write_table(table, input_dir / "repo.parquet", row_group_size=2)

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
        np.array([0, 0], dtype=np.int32),
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


def test_semantic_parquet_requires_usr_identity_schema_metadata(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    converter = _load_converter_module()
    pq.write_table(
        pa.table(
            {
                "input_ids": [[1]],
                "token_symbol_ids": [[7]],
                "token_call_targets": [[0]],
                "token_type_refs": [[0]],
                "token_def_use": [[1]],
            }
        ),
        tmp_path / "stale.parquet",
    )

    with pytest.raises(RuntimeError, match="regenerate.*clang USR"):
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
