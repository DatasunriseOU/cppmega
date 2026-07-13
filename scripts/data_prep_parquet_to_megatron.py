#!/usr/bin/env python3
"""Convert packed parquet to Megatron indexed binary format.

Reads ``input_ids`` or legacy ``token_ids`` from parquet shards and writes
``.bin`` + ``.idx`` files that Megatron's GPTDataset / MMapIndexedDataset can
consume directly.  The current cppmega packed schema should be converted with
``--split all --token-column auto --length-column valid_token_count`` so no
bucket shard is silently repurposed as validation data and trailing padding is
not materialized in the mmap files.

This script must run on the H200 machine where megatron-core is installed.

Usage:
    python data_prep_parquet_to_megatron.py \
        --input-dir /home/dave/cppmega-root/data/parquet/clang_semantic_4k_v10 \
        --output-prefix /home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10 \
        --split train

    python data_prep_parquet_to_megatron.py \
        --input-dir /home/dave/cppmega-root/data/parquet/clang_commits_4k_v1 \
        --output-prefix /home/dave/cppmega-root/data/megatron/clang_commits_4k_v1 \
        --split train
"""

from __future__ import annotations

import argparse
from array import array
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SymbolIdentityRegistry,
    compute_symbol_id,
)


_OUTPUT_DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int32": np.int32,
    "int64": np.int64,
    "uint64": np.uint64,
    # Historical CLI compatibility. Megatron's MMIDIDX dtype enum has no
    # uint32 token dtype, so positive token IDs that need more than uint16 must
    # be stored as int32.
    "uint32": np.int32,
}

_SIDECAR_DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}

_MEGATRON_DTYPE_CODE_MAP = {
    np.uint8: 1,
    np.int32: 4,
    np.int64: 5,
    np.uint16: 8,
}

DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS: tuple[tuple[str, str], ...] = (
    ("loss_mask", "uint8"),
    ("doc_ids", "uint16"),
    ("token_domain_ids", "uint16"),
    ("token_role_ids", "uint16"),
    ("token_entity_ids", "uint32"),
    ("token_scope_ids", "uint32"),
    ("token_confidence_ids", "uint8"),
    ("token_structure_ids", "uint8"),
    ("token_dep_levels", "uint16"),
    ("token_ast_depth", "uint16"),
    ("token_sibling_index", "uint16"),
    ("token_ast_node_type", "uint16"),
    ("token_symbol_ids", "uint64"),
    ("token_call_targets", "uint64"),
    ("token_type_refs", "uint64"),
    ("token_def_use", "uint8"),
    ("token_change_mask_pre", "uint8"),
    ("token_change_mask_post", "uint8"),
)

# ``token_platform_ids`` is a legacy scalar mirror and cannot represent the
# multi-label platform context carried by each packed source document.  The
# current packed schema stores the lossless relation as
# ``source_platform_ids`` plus row-local ``doc_ids``.  The converter preserves
# that relation in a compact nested-CSR sidecar instead of expanding a
# (tokens, MAX_PLATFORM_IDS) tensor on disk.
SOURCE_PLATFORM_IDS_COLUMN = "source_platform_ids"
SOURCE_PLATFORM_SIDECAR_SCHEMA = "cppmega_source_platform_v1"
MAX_SOURCE_PLATFORM_IDS = 32
PLATFORM_VOCAB_SIZE = 113

DEFAULT_CPPMEGA_GRAPH_SIDECARS: tuple[tuple[str, str, str], ...] = (
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

REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION = SYMBOL_IDENTITY_SCHEMA_VERSION
_SYMBOL_ID_COLUMNS = (
    "token_symbol_ids",
    "token_call_targets",
    "token_type_refs",
)
_compute_symbol_id = compute_symbol_id


def _require_symbol_identity_schema_metadata(
    metadata: dict[bytes, bytes] | None,
    source: str,
) -> int:
    key = SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")
    raw_version = (metadata or {}).get(key)
    try:
        version = int(raw_version) if raw_version is not None else None
    except (TypeError, ValueError):
        version = None
    if version != REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION:
        raise RuntimeError(
            f"{source}: missing or incompatible {SYMBOL_IDENTITY_SCHEMA_METADATA_KEY} "
            f"metadata ({raw_version!r}); regenerate parquet with clang USR/signature "
            f"schema v{REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION} before conversion"
        )
    return version


def _require_symbol_identity_schema(shards: list[str]) -> int | None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    accepted_version: int | None = None
    semantic_presence: bool | None = None
    identity_columns = {
        *_SYMBOL_ID_COLUMNS,
        "token_def_use",
    }
    corpus_registry = SymbolIdentityRegistry()
    for shard in shards:
        parquet_file = pq.ParquetFile(shard)
        schema = parquet_file.schema_arrow
        present_identity_columns = identity_columns & set(schema.names)
        if present_identity_columns and present_identity_columns != identity_columns:
            missing = sorted(identity_columns - present_identity_columns)
            raise RuntimeError(
                f"{shard}: partial semantic-symbol columns; missing={missing}. "
                "Regenerate one complete clang USR/signature dataset"
            )
        has_identity_columns = bool(present_identity_columns)
        if semantic_presence is None:
            semantic_presence = has_identity_columns
        elif has_identity_columns != semantic_presence:
            raise RuntimeError(
                f"{shard}: mixed semantic-symbol column presence across parquet shards; "
                "regenerate one consistent clang USR/signature dataset"
            )
        if not has_identity_columns:
            continue
        if SYMBOL_IDENTITIES_COLUMN not in schema.names:
            raise RuntimeError(
                f"{shard}: semantic symbol columns require "
                f"{SYMBOL_IDENTITIES_COLUMN!r} collision claims; regenerate parquet "
                "with clang USR/signature identities"
            )
        identity_type = schema.field(SYMBOL_IDENTITIES_COLUMN).type
        if not (pa.types.is_list(identity_type) or pa.types.is_large_list(identity_type)):
            raise RuntimeError(
                f"{shard}: {SYMBOL_IDENTITIES_COLUMN} must be a list, got {identity_type}"
            )
        identity_value_type = identity_type.value_type
        if not pa.types.is_struct(identity_value_type):
            raise RuntimeError(
                f"{shard}: {SYMBOL_IDENTITIES_COLUMN} values must be structs"
            )
        try:
            symbol_id_type = identity_value_type.field("symbol_id").type
            symbol_key_type = identity_value_type.field("symbol_key").type
        except KeyError as error:
            raise RuntimeError(
                f"{shard}: {SYMBOL_IDENTITIES_COLUMN} requires symbol_id/symbol_key"
            ) from error
        if symbol_id_type != pa.uint64() or symbol_key_type != pa.string():
            raise RuntimeError(
                f"{shard}: invalid {SYMBOL_IDENTITIES_COLUMN} types: {identity_value_type}"
            )
        for column in _SYMBOL_ID_COLUMNS:
            column_type = schema.field(column).type
            if not (
                pa.types.is_list(column_type) or pa.types.is_large_list(column_type)
            ) or column_type.value_type != pa.uint64():
                raise RuntimeError(
                    f"{shard}: {column} must use list<uint64>, got {column_type}"
                )
        metadata = schema.metadata
        version = _require_symbol_identity_schema_metadata(metadata, shard)
        if accepted_version is None:
            accepted_version = version
        elif version != accepted_version:
            raise RuntimeError(
                f"{shard}: mixed symbol identity schema versions in one conversion; "
                "regenerate all parquet shards with clang USR/signature identities"
            )
        row_offset = 0
        columns = [*_SYMBOL_ID_COLUMNS, SYMBOL_IDENTITIES_COLUMN]
        for batch in parquet_file.iter_batches(columns=columns):
            rows = batch.to_pylist()
            for local_index, row in enumerate(rows):
                source = f"{shard}:row={row_offset + local_index}"
                row_registry = SymbolIdentityRegistry()
                corpus_registry.register_records(
                    row[SYMBOL_IDENTITIES_COLUMN], source=source
                )
                row_registry.register_records(
                    row[SYMBOL_IDENTITIES_COLUMN], source=source
                )
                used_ids = {
                    int(value)
                    for column in _SYMBOL_ID_COLUMNS
                    for value in row[column]
                    if int(value) != 0
                }
                row_registry.require_ids(used_ids, source=source)
            row_offset += batch.num_rows
    return accepted_version


def _add_symbol_identity_manifest(
    sidecar_data: dict[str, object],
    version: int,
) -> None:
    if version != REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"cannot write manifest for symbol identity schema v{version}; "
            f"expected v{REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION}"
        )
    sidecar_data["symbol_identity_schema_version"] = version


def _resolve_output_dtype(dtype_str: str) -> type[np.generic]:
    try:
        dtype = _OUTPUT_DTYPE_MAP[dtype_str]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype_str}") from exc
    if dtype_str == "uint32":
        print(
            "WARNING: Megatron MMIDIDX has no uint32 dtype code; "
            "writing token IDs as int32 instead",
            file=sys.stderr,
        )
    return dtype


def _resolve_sidecar_dtype(dtype_str: str) -> np.dtype:
    try:
        return np.dtype(_SIDECAR_DTYPE_MAP[dtype_str])
    except KeyError as exc:
        raise ValueError(f"unsupported sidecar dtype: {dtype_str}") from exc


def _require_symbol_sidecar_dtypes(
    side_channels: list[str] | None,
    side_channel_dtypes: list[str] | None,
) -> None:
    requested = dict(
        zip(side_channels or [], side_channel_dtypes or [], strict=True)
    )
    present = set(_SYMBOL_ID_COLUMNS) & set(requested)
    if present and present != set(_SYMBOL_ID_COLUMNS):
        missing = sorted(set(_SYMBOL_ID_COLUMNS) - present)
        raise ValueError(
            f"partial symbol identity sidecars are not allowed; missing={missing}"
        )
    invalid = {
        column: requested[column]
        for column in present
        if requested[column] != "uint64"
    }
    if invalid:
        raise ValueError(
            f"v{SYMBOL_IDENTITY_SCHEMA_VERSION} symbol sidecars must use uint64: "
            f"{invalid}"
        )


def _megatron_dtype_code(dtype: type[np.generic] | np.dtype) -> int:
    dtype_type = np.dtype(dtype).type
    if dtype_type not in _MEGATRON_DTYPE_CODE_MAP:
        supported = ", ".join(
            sorted(
                np.dtype(supported_dtype).name
                for supported_dtype in _MEGATRON_DTYPE_CODE_MAP
            )
        )
        raise ValueError(
            f"unsupported Megatron MMIDIDX dtype {np.dtype(dtype).name}; "
            f"supported dtypes: {supported}"
        )
    return _MEGATRON_DTYPE_CODE_MAP[dtype_type]


def find_parquet_shards(input_dir: str, split: str) -> list[str]:
    """Find all parquet shard files for a given split.

    Convention from nanochat:
    - train shards: shard_00000.parquet ... shard_NNNNN.parquet (all except last)
    - val shard: val_shard.parquet or last shard
    """
    input_path = Path(input_dir)
    all_parquets = sorted(input_path.glob("*.parquet"))
    if not all_parquets:
        raise FileNotFoundError(f"no parquet files in {input_dir}")

    val_shard = input_path / "val_shard.parquet"
    has_explicit_val = val_shard.exists()

    if split == "train":
        if has_explicit_val:
            return [str(p) for p in all_parquets if p.name != "val_shard.parquet"]
        # Last shard is val by convention
        return [str(p) for p in all_parquets[:-1]] if len(all_parquets) > 1 else [str(all_parquets[0])]
    elif split == "val":
        if has_explicit_val:
            return [str(val_shard)]
        return [str(all_parquets[-1])] if len(all_parquets) > 1 else [str(all_parquets[0])]
    elif split == "all":
        return [str(p) for p in all_parquets]
    else:
        raise ValueError(f"unknown split: {split}")


def _resolve_token_column(shards: list[str], requested: str) -> str:
    """Resolve ``auto`` against the first shard and fail on ambiguous schemas."""

    if requested != "auto":
        return requested
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(shards[0]).schema_arrow.names)
    present = [name for name in ("input_ids", "token_ids") if name in names]
    if len(present) != 1:
        raise ValueError(
            "--token-column auto requires exactly one of input_ids/token_ids; "
            f"found {present or 'neither'} in {shards[0]}"
        )
    return present[0]


def _resolve_length_column(shards: list[str], requested: str | None) -> str | None:
    """Resolve the optional packed-row length column."""

    if requested in (None, "", "none"):
        return None
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(shards[0]).schema_arrow.names)
    if requested == "auto":
        return "valid_token_count" if "valid_token_count" in names else None
    if requested not in names:
        raise ValueError(f"length column {requested!r} is absent from {shards[0]}")
    return requested


def _resolve_source_platform_sidecar(
    shards: list[str], requested: bool | None
) -> bool:
    """Resolve auto/required source-platform preservation against the schema."""

    import pyarrow.parquet as pq

    present = SOURCE_PLATFORM_IDS_COLUMN in set(
        pq.ParquetFile(shards[0]).schema_arrow.names
    )
    if requested is True and not present:
        raise ValueError(
            f"required {SOURCE_PLATFORM_IDS_COLUMN} is absent from {shards[0]}"
        )
    return present if requested is None else bool(requested)


def _row_token_length(
    raw_length: object,
    capacity: int,
    *,
    length_column: str | None,
    shard_path: str,
    row_idx: int,
) -> int:
    if length_column is None:
        return capacity
    if hasattr(raw_length, "as_py"):
        raw_length = raw_length.as_py()
    if raw_length is None:
        raise ValueError(
            f"length column {length_column} is null at {shard_path}#row{row_idx}"
        )
    length = int(raw_length)
    if length <= 0 or length > capacity:
        raise ValueError(
            f"length column {length_column}={length} is outside [1, {capacity}] "
            f"at {shard_path}#row{row_idx}"
        )
    return length


def _validate_token_ids(
    token_ids: list[int],
    *,
    dtype: type[np.generic],
    vocab_size: int,
    shard_path: str,
    row_idx: int,
) -> None:
    if not token_ids:
        raise ValueError(f"empty token row at {shard_path}#row{row_idx}")
    min_id = min(token_ids)
    max_id = max(token_ids)
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"token ids [{min_id}, {max_id}] are outside vocab [0, {vocab_size}) "
            f"at {shard_path}#row{row_idx}"
        )
    info = np.iinfo(dtype)
    if max_id > info.max:
        raise ValueError(
            f"token id {max_id} exceeds output dtype {np.dtype(dtype).name} max "
            f"{info.max} at {shard_path}#row{row_idx}"
        )


def _require_token_aligned_side_channel(
    column: str,
    side_val: list[int] | None,
    token_ids: list[int],
    *,
    shard_path: str,
    row_idx: int,
) -> list[int]:
    """Return a side-channel row only if it is exactly token-aligned."""

    if side_val is None:
        raise ValueError(
            f"side-channel {column} is null at {shard_path}#row{row_idx}; "
            f"token_ids length {len(token_ids)}"
        )
    if len(side_val) != len(token_ids):
        raise ValueError(
            f"side-channel {column} length {len(side_val)} != "
            f"token_ids length {len(token_ids)} at {shard_path}#row{row_idx}"
        )
    return side_val


def _fixed_width_list_matrix(
    column: object,
    *,
    column_name: str,
    expected_rows: int,
    expected_width: int | None,
    shard_path: str,
    row_group_idx: int,
) -> tuple[np.ndarray, int]:
    """Return a canonical Arrow list column as a dense row-major matrix.

    Packed cppmega parquet stores every token-aligned row at the bucket width.
    Converting one scalar at a time is prohibitively expensive for millions of
    rows, so this helper validates that contract once per row group and exposes
    the contiguous values buffer to NumPy.
    """

    combined = column.combine_chunks()  # type: ignore[union-attr]
    if len(combined) != expected_rows:
        raise ValueError(
            f"{column_name} row count {len(combined)} != {expected_rows} at "
            f"{shard_path}#row_group{row_group_idx}"
        )
    if combined.null_count:
        raise ValueError(
            f"{column_name} contains null rows at "
            f"{shard_path}#row_group{row_group_idx}"
        )
    if not hasattr(combined, "offsets") or not hasattr(combined, "values"):
        raise ValueError(
            f"{column_name} must be an Arrow list column at "
            f"{shard_path}#row_group{row_group_idx}"
        )

    offsets = np.asarray(combined.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    lengths = np.diff(offsets)
    width = int(lengths[0]) if len(lengths) else 0
    if width <= 0 or np.any(lengths != width):
        raise ValueError(
            f"{column_name} rows must have one positive packed width at "
            f"{shard_path}#row_group{row_group_idx}; widths={np.unique(lengths).tolist()}"
        )
    if expected_width is not None and width != expected_width:
        raise ValueError(
            f"side-channel {column_name} packed width {width} != token packed "
            f"width {expected_width} at {shard_path}#row_group{row_group_idx}"
        )

    start = int(offsets[0])
    item_count = int(offsets[-1] - offsets[0])
    values = combined.values.slice(start, item_count)
    if values.null_count:
        raise ValueError(
            f"{column_name} contains null list elements at "
            f"{shard_path}#row_group{row_group_idx}"
        )
    matrix = np.asarray(values.to_numpy(zero_copy_only=False)).reshape(
        expected_rows, width
    )
    return matrix, width


def _row_group_lengths(
    column: object | None,
    *,
    row_count: int,
    capacity: int,
    length_column: str | None,
    shard_path: str,
    row_group_idx: int,
) -> np.ndarray:
    if column is None:
        return np.full(row_count, capacity, dtype=np.int64)
    combined = column.combine_chunks()  # type: ignore[union-attr]
    if len(combined) != row_count or combined.null_count:
        raise ValueError(
            f"length column {length_column} is null or misaligned at "
            f"{shard_path}#row_group{row_group_idx}"
        )
    lengths = np.asarray(combined.to_numpy(zero_copy_only=False), dtype=np.int64)
    invalid = np.flatnonzero((lengths <= 0) | (lengths > capacity))
    if invalid.size:
        row_idx = int(invalid[0])
        raise ValueError(
            f"length column {length_column}={int(lengths[row_idx])} is outside "
            f"[1, {capacity}] at {shard_path}#row{row_idx}"
        )
    return lengths


def _unique_columns(*groups: list[str] | tuple[str, ...] | None) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for column in group or ():
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def _normalize_edge_pairs(
    value: object,
    *,
    column: str,
    shard_path: str,
    row_idx: int,
) -> np.ndarray:
    """Normalize a parquet edge-list cell to an ``(E, 2)`` int32 array."""

    if value is None:
        return np.zeros((0, 2), dtype=np.int32)
    if isinstance(value, np.ndarray) and value.dtype.fields:
        fields = value.dtype.fields
        if "from" not in fields or "to" not in fields:
            raise ValueError(
                f"{column} structured edge array must contain from/to fields at "
                f"{shard_path}#row{row_idx}"
            )
        pairs = np.stack([value["from"], value["to"]], axis=1)
    else:
        pairs_list: list[tuple[int, int]] = []
        for edge in value:  # type: ignore[union-attr]
            if hasattr(edge, "as_py"):
                edge = edge.as_py()
            if isinstance(edge, dict):
                src = edge.get("from")
                dst = edge.get("to")
            elif isinstance(edge, (list, tuple, np.ndarray)) and len(edge) >= 2:
                src = edge[0]
                dst = edge[1]
            else:
                raise ValueError(
                    f"{column} edge must be {{from,to}} or pair at "
                    f"{shard_path}#row{row_idx}; got {type(edge).__name__}"
                )
            if src is None or dst is None:
                raise ValueError(
                    f"{column} edge has missing endpoint at {shard_path}#row{row_idx}"
                )
            src_i = int(src)
            dst_i = int(dst)
            if src_i < 0 or dst_i < 0:
                raise ValueError(
                    f"{column} edge endpoints must be non-negative at "
                    f"{shard_path}#row{row_idx}: ({src_i}, {dst_i})"
                )
            pairs_list.append((src_i, dst_i))
        pairs = np.asarray(pairs_list, dtype=np.int64)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            f"{column} must normalize to (E, 2), got shape {tuple(pairs.shape)} "
            f"at {shard_path}#row{row_idx}"
        )
    if np.any(pairs < 0):
        raise ValueError(f"{column} contains negative edge endpoint at {shard_path}#row{row_idx}")
    if np.any(pairs > np.iinfo(np.int32).max):
        raise ValueError(f"{column} edge endpoint exceeds int32 at {shard_path}#row{row_idx}")
    return pairs.astype(np.int32, copy=False)


def _normalize_edge_triples(
    value: object,
    *,
    column: str,
    shard_path: str,
    row_idx: int,
) -> np.ndarray:
    """Normalize a parquet edge-list cell to an ``(E, 3)`` int32 array.

    Accepted element formats:
    - ``{"from": i, "to": j, "kind": k}``
    - ``{"src": i, "dst": j, "kind": k}``
    - ``[i, j, k]`` / ``(i, j, k)``
    """

    if value is None:
        return np.zeros((0, 3), dtype=np.int32)
    triples_list: list[tuple[int, int, int]] = []
    for edge in value:  # type: ignore[union-attr]
        if hasattr(edge, "as_py"):
            edge = edge.as_py()
        if isinstance(edge, dict):
            src = edge.get("from", edge.get("src"))
            dst = edge.get("to", edge.get("dst"))
            kind = edge.get("kind")
        elif isinstance(edge, (list, tuple, np.ndarray)) and len(edge) >= 3:
            src, dst, kind = edge[0], edge[1], edge[2]
        else:
            raise ValueError(
                f"{column} edge must be {{from,to,kind}}/{{src,dst,kind}} or "
                f"triple at {shard_path}#row{row_idx}; got {type(edge).__name__}"
            )
        if src is None or dst is None or kind is None:
            raise ValueError(
                f"{column} edge has missing src/dst/kind at {shard_path}#row{row_idx}"
            )
        src_i = int(src)
        dst_i = int(dst)
        kind_i = int(kind)
        if src_i < 0 or dst_i < 0 or kind_i < 0:
            raise ValueError(
                f"{column} edge triples must be non-negative at "
                f"{shard_path}#row{row_idx}: ({src_i}, {dst_i}, {kind_i})"
            )
        triples_list.append((src_i, dst_i, kind_i))
    triples = np.asarray(triples_list, dtype=np.int64)
    if triples.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(
            f"{column} must normalize to (E, 3), got shape {tuple(triples.shape)} "
            f"at {shard_path}#row{row_idx}"
        )
    if np.any(triples > np.iinfo(np.int32).max):
        raise ValueError(f"{column} edge value exceeds int32 at {shard_path}#row{row_idx}")
    return triples.astype(np.int32, copy=False)


def _normalize_ragged_int_vector(
    value: object,
    *,
    dtype: np.dtype,
    column: str,
    shard_path: str,
    row_idx: int,
) -> np.ndarray:
    """Normalize a ragged parquet scalar to a 1-D integer array."""

    if value is None:
        return np.zeros((0,), dtype=dtype)
    arr = np.asarray(value)
    if arr.size == 0:
        return np.zeros((0,), dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(
            f"{column} must be a 1-D ragged integer vector, got shape "
            f"{tuple(arr.shape)} at {shard_path}#row{row_idx}"
        )
    if np.any(arr < 0):
        raise ValueError(f"{column} contains negative value at {shard_path}#row{row_idx}")
    info = np.iinfo(dtype)
    if np.any(arr > info.max):
        raise ValueError(
            f"{column} value exceeds {dtype.name} max {info.max} at {shard_path}#row{row_idx}"
        )
    return arr.astype(dtype, copy=False)


class _GraphSidecarWriters:
    """Write document-aligned CSR-style graph sidecars next to .bin/.idx."""

    def __init__(
        self,
        output_prefix: str,
        specs: tuple[tuple[str, str, str], ...] = DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    ) -> None:
        self._output_prefix = output_prefix
        self._specs = specs
        self._data_files = {
            column: open(f"{output_prefix}_{column}_data.bin", "wb")
            for column, _, _ in specs
        }
        # Compact int64 buffers avoid hundreds of MiB of Python-int overhead on
        # multi-million-document buckets while retaining one sequential write
        # per sidecar at close.
        self._offsets = {column: array("q", [0]) for column, _, _ in specs}
        self._item_counts = {column: 0 for column, _, _ in specs}
        self._pending = {column: [] for column, _, _ in specs}
        self._closed = False

    @property
    def columns(self) -> list[str]:
        return [column for column, _, _ in self._specs]

    def append(
        self,
        values: dict[str, object],
        *,
        shard_path: str,
        row_idx: int,
        token_count: int | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("graph sidecar writers are already closed")
        ragged_arrays: dict[str, np.ndarray] = {}
        for column, kind, dtype_str in self._specs:
            if kind == "ragged_1d":
                ragged_arrays[column] = _normalize_ragged_int_vector(
                    values.get(column),
                    dtype=_resolve_sidecar_dtype(dtype_str),
                    column=column,
                    shard_path=shard_path,
                    row_idx=row_idx,
                )
        chunk_lengths = {
            column: len(arr)
            for column, arr in ragged_arrays.items()
            if column.startswith("token_chunk_")
        }
        if chunk_lengths and len(set(chunk_lengths.values())) != 1:
            raise ValueError(
                "token_chunk_* sidecars must have equal lengths at "
                f"{shard_path}#row{row_idx}: {chunk_lengths}"
            )
        starts = ragged_arrays.get("token_chunk_starts")
        ends = ragged_arrays.get("token_chunk_ends")
        if starts is not None and ends is not None and (
            np.any(starts >= ends)
            or (token_count is not None and np.any(ends > token_count))
        ):
            raise ValueError(
                "token chunk spans must satisfy 0 <= start < end <= "
                f"valid token count at {shard_path}#row{row_idx}"
            )
        chunk_count = len(starts) if starts is not None else None

        for column, kind, dtype_str in self._specs:
            dtype = _resolve_sidecar_dtype(dtype_str)
            if kind == "edge_pairs":
                arr = _normalize_edge_pairs(
                    values.get(column),
                    column=column,
                    shard_path=shard_path,
                    row_idx=row_idx,
                )
                if arr.size and chunk_count is None:
                    raise ValueError(
                        f"{column} requires token_chunk_starts at "
                        f"{shard_path}#row{row_idx}"
                    )
                if arr.size and np.any(arr >= chunk_count):
                    raise ValueError(
                        f"{column} endpoint exceeds chunk count {chunk_count} at "
                        f"{shard_path}#row{row_idx}"
                    )
                arr = arr.astype(dtype, copy=False)
                item_count = int(arr.shape[0])
            elif kind == "edge_triples":
                arr = _normalize_edge_triples(
                    values.get(column),
                    column=column,
                    shard_path=shard_path,
                    row_idx=row_idx,
                )
                if token_count is not None and arr.size and np.any(arr[:, :2] >= token_count):
                    raise ValueError(
                        f"{column} endpoint exceeds valid token count {token_count} at "
                        f"{shard_path}#row{row_idx}"
                    )
                arr = arr.astype(dtype, copy=False)
                item_count = int(arr.shape[0])
            elif kind == "ragged_1d":
                arr = ragged_arrays[column]
                invalid_chunk_offset = False
                if token_count is not None and arr.size:
                    if column == "token_chunk_starts":
                        invalid_chunk_offset = bool(np.any(arr >= token_count))
                    elif column == "token_chunk_ends":
                        # Chunk ends are exclusive; a final chunk ending exactly
                        # at valid_token_count is canonical and must be accepted.
                        invalid_chunk_offset = bool(np.any(arr > token_count))
                if invalid_chunk_offset:
                    raise ValueError(
                        f"{column} value exceeds valid token count {token_count} at "
                        f"{shard_path}#row{row_idx}"
                    )
                item_count = int(arr.shape[0])
            else:
                raise ValueError(f"unsupported graph sidecar kind {kind!r} for {column}")
            self._item_counts[column] += item_count
            self._offsets[column].append(self._item_counts[column])
            if item_count:
                self._pending[column].append(arr)

    def flush(self) -> None:
        """Write one contiguous block per graph column for the current batch."""

        if self._closed:
            raise RuntimeError("graph sidecar writers are already closed")
        for column, parts in self._pending.items():
            if parts:
                np.concatenate(parts, axis=0).tofile(self._data_files[column])
                parts.clear()

    def close(self) -> dict[str, dict[str, object]]:
        if self._closed:
            raise RuntimeError("graph sidecar writers are already closed")
        self.flush()
        for fh in self._data_files.values():
            fh.close()
        manifest: dict[str, dict[str, object]] = {}
        base = os.path.basename(self._output_prefix)
        for column, kind, dtype_str in self._specs:
            offsets_path = f"{self._output_prefix}_{column}_offsets.bin"
            np.asarray(self._offsets[column], dtype=np.int64).tofile(offsets_path)
            entry: dict[str, object] = {
                "kind": kind,
                "offsets_path": f"{base}_{column}_offsets.bin",
                "data_path": f"{base}_{column}_data.bin",
                "offset_dtype": "int64",
                "dtype": dtype_str,
                "item_count": self._item_counts[column],
            }
            if kind == "edge_pairs":
                entry["shape_tail"] = [2]
                entry["coordinate_space"] = "chunk_index"
            elif kind == "edge_triples":
                entry["shape_tail"] = [3]
                entry["coordinate_space"] = "token_index"
            elif column in {"token_chunk_starts", "token_chunk_ends"}:
                entry["coordinate_space"] = "token_index"
            else:
                entry["coordinate_space"] = "chunk_index"
            manifest[column] = entry
        self._closed = True
        return manifest

    def abort_close(self) -> None:
        if self._closed:
            return
        for fh in self._data_files.values():
            fh.close()
        self._closed = True


class _SourcePlatformSidecarWriter:
    """Write source-document platform bags as compact nested CSR arrays."""

    def __init__(self, output_prefix: str) -> None:
        self._output_prefix = output_prefix
        self._platform_path = f"{output_prefix}_source_platform_ids.bin"
        self._platform_file = open(self._platform_path, "wb")
        self._sequence_doc_offsets = array("q", [0])
        self._doc_platform_offsets = array("q", [0])
        self._source_document_count = 0
        self._platform_id_count = 0
        self._pending: list[np.ndarray] = []
        self._closed = False

    def append(
        self,
        value: object,
        *,
        doc_ids: list[int],
        token_count: int,
        shard_path: str,
        row_idx: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("source platform sidecar writer is already closed")
        if value is None:
            raise ValueError(
                f"{SOURCE_PLATFORM_IDS_COLUMN} is null at {shard_path}#row{row_idx}"
            )
        groups = value.as_py() if hasattr(value, "as_py") else value
        if not isinstance(groups, list) or not groups:
            raise ValueError(
                f"{SOURCE_PLATFORM_IDS_COLUMN} must contain one platform bag per "
                f"source document at {shard_path}#row{row_idx}"
            )
        if len(doc_ids) < token_count:
            raise ValueError(
                f"doc_ids length {len(doc_ids)} < valid token count {token_count} at "
                f"{shard_path}#row{row_idx}"
            )
        valid_doc_ids = [int(value) for value in doc_ids[:token_count]]
        if not valid_doc_ids:
            raise ValueError(f"empty valid doc_ids at {shard_path}#row{row_idx}")
        expected_doc_ids = set(range(1, len(groups) + 1))
        actual_doc_ids = set(valid_doc_ids)
        if actual_doc_ids != expected_doc_ids:
            raise ValueError(
                f"doc_ids must reference every source platform bag exactly by row-local "
                f"IDs 1..{len(groups)} at {shard_path}#row{row_idx}; "
                f"got {sorted(actual_doc_ids)}"
            )

        for source_doc_index, raw_ids in enumerate(groups):
            ids = raw_ids.as_py() if hasattr(raw_ids, "as_py") else raw_ids
            if ids is None:
                ids = []
            if not isinstance(ids, list):
                raise ValueError(
                    f"{SOURCE_PLATFORM_IDS_COLUMN}[{source_doc_index}] must be a list "
                    f"at {shard_path}#row{row_idx}"
                )
            normalized = sorted(set(int(value) for value in ids))
            if len(normalized) > MAX_SOURCE_PLATFORM_IDS:
                raise ValueError(
                    f"{SOURCE_PLATFORM_IDS_COLUMN}[{source_doc_index}] has "
                    f"{len(normalized)} IDs; max={MAX_SOURCE_PLATFORM_IDS} at "
                    f"{shard_path}#row{row_idx}"
                )
            if normalized and (
                normalized[0] <= 0 or normalized[-1] >= PLATFORM_VOCAB_SIZE
            ):
                raise ValueError(
                    f"{SOURCE_PLATFORM_IDS_COLUMN}[{source_doc_index}] IDs must be "
                    f"inside [1,{PLATFORM_VOCAB_SIZE}) at {shard_path}#row{row_idx}: "
                    f"{normalized}"
                )
            if normalized:
                self._pending.append(np.asarray(normalized, dtype=np.uint16))
            self._platform_id_count += len(normalized)
            self._doc_platform_offsets.append(self._platform_id_count)

        self._source_document_count += len(groups)
        self._sequence_doc_offsets.append(self._source_document_count)

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError("source platform sidecar writer is already closed")
        if self._pending:
            np.concatenate(self._pending).tofile(self._platform_file)
            self._pending.clear()

    def close(self) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("source platform sidecar writer is already closed")
        self.flush()
        self._platform_file.close()
        sequence_offsets_path = (
            f"{self._output_prefix}_source_platform_sequence_doc_offsets.bin"
        )
        platform_offsets_path = (
            f"{self._output_prefix}_source_platform_doc_id_offsets.bin"
        )
        np.asarray(self._sequence_doc_offsets, dtype=np.int64).tofile(
            sequence_offsets_path
        )
        np.asarray(self._doc_platform_offsets, dtype=np.int64).tofile(
            platform_offsets_path
        )
        base = os.path.basename(self._output_prefix)
        self._closed = True
        return {
            "schema": SOURCE_PLATFORM_SIDECAR_SCHEMA,
            "sequence_doc_offsets_path": (
                f"{base}_source_platform_sequence_doc_offsets.bin"
            ),
            "doc_platform_offsets_path": (
                f"{base}_source_platform_doc_id_offsets.bin"
            ),
            "platform_ids_path": f"{base}_source_platform_ids.bin",
            "offset_dtype": "int64",
            "dtype": "uint16",
            "source_document_count": self._source_document_count,
            "platform_id_count": self._platform_id_count,
            "max_platform_ids": MAX_SOURCE_PLATFORM_IDS,
            "document_id_sidecar": "doc_ids",
            "document_id_base": 1,
        }

    def abort_close(self) -> None:
        if self._closed:
            return
        self._platform_file.close()
        self._closed = True


def _graph_sidecar_values(
    graph_cols: dict[str, object],
    row_idx: int,
) -> dict[str, object]:
    return {column: graph_cols[column][row_idx].as_py() for column in graph_cols}  # type: ignore[index]


def _add_graph_manifest(
    sidecar_data: dict[str, object],
    graph_sidecar_paths: dict[str, dict[str, object]] | None,
) -> None:
    if not graph_sidecar_paths:
        return
    sidecar_data["graph_sidecar_schema"] = "cppmega_graph_routes_v2"
    sidecar_data["graph_sidecar_paths"] = graph_sidecar_paths


def _add_source_platform_manifest(
    sidecar_data: dict[str, object],
    source_platform_sidecar: dict[str, object] | None,
) -> None:
    if source_platform_sidecar is not None:
        sidecar_data["source_platform_sidecar"] = source_platform_sidecar


def _convert_parquet_to_numpy(
    input_dir: str,
    output_prefix: str,
    split: str,
    token_column: str,
    dtype_str: str,
    length_column: str | None = None,
    side_channels: list[str] | None = None,
    side_channel_dtypes: list[str] | None = None,
    graph_sidecars: tuple[tuple[str, str, str], ...] | None = DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    source_platform_sidecar: bool | None = None,
    vocab_size: int = 65536,
) -> None:
    """Fallback: write Megatron-compatible .bin + .idx using raw numpy.

    Format:
    - .bin: contiguous flat array of all token IDs
    - .idx: magic(9), version(1), dtype_code(1), num_sequences(8), num_documents(8),
            then sizes[num_docs] as int32, then pointers[num_docs] as int64
    """
    import pyarrow.parquet as pq
    import struct

    dtype = _resolve_output_dtype(dtype_str)
    dtype_code = _megatron_dtype_code(dtype)

    shards = find_parquet_shards(input_dir, split)
    token_column = _resolve_token_column(shards, token_column)
    length_column = _resolve_length_column(shards, length_column)
    symbol_identity_schema_version = _require_symbol_identity_schema(shards)
    _require_symbol_sidecar_dtypes(side_channels, side_channel_dtypes)
    write_source_platform = _resolve_source_platform_sidecar(
        shards, source_platform_sidecar
    )
    if write_source_platform and "doc_ids" not in (side_channels or []):
        raise ValueError(
            "source platform sidecar requires token-aligned doc_ids sidecar"
        )
    print(f"found {len(shards)} {split} shards")

    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    side_dtypes = {col: np.dtype(dt) for col, dt in zip(side_channels or [], side_channel_dtypes or [], strict=True)}
    t0 = time.time()

    graph_writers = _GraphSidecarWriters(output_prefix, graph_sidecars) if graph_sidecars else None
    source_platform_writer = (
        _SourcePlatformSidecarWriter(output_prefix)
        if write_source_platform
        else None
    )
    graph_columns = graph_writers.columns if graph_writers is not None else []
    columns_to_read = _unique_columns(
        [token_column],
        [length_column] if length_column else None,
        side_channels,
        graph_columns,
        [SOURCE_PLATFORM_IDS_COLUMN] if write_source_platform else None,
    )

    bin_path = output_prefix + ".bin"
    side_writers = {
        col: open(f"{output_prefix}_{col}.bin", "wb")
        for col in (side_channels or [])
    }
    sizes = array("i")
    pointers = array("q")
    total_tokens = 0
    source_capacity_tokens = 0
    trained_tokens = 0
    graph_sidecar_paths: dict[str, dict[str, object]] | None = None

    try:
        with open(bin_path, "wb") as bin_fh:
            for shard_idx, shard_path in enumerate(shards):
                pf = pq.ParquetFile(shard_path)
                for rg_idx in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(rg_idx, columns=columns_to_read)
                    token_col = table.column(token_column)
                    length_col = table.column(length_column) if length_column else None
                    side_cols = {col: table.column(col) for col in (side_channels or [])}
                    graph_cols = {col: table.column(col) for col in graph_columns}
                    source_platform_col = (
                        table.column(SOURCE_PLATFORM_IDS_COLUMN)
                        if write_source_platform
                        else None
                    )
                    row_count = len(token_col)
                    token_matrix, capacity = _fixed_width_list_matrix(
                        token_col,
                        column_name=token_column,
                        expected_rows=row_count,
                        expected_width=None,
                        shard_path=shard_path,
                        row_group_idx=rg_idx,
                    )
                    lengths = _row_group_lengths(
                        length_col,
                        row_count=row_count,
                        capacity=capacity,
                        length_column=length_column,
                        shard_path=shard_path,
                        row_group_idx=rg_idx,
                    )
                    valid_mask = np.arange(capacity)[None, :] < lengths[:, None]
                    flat_tokens = token_matrix[valid_mask]
                    if flat_tokens.size == 0:
                        raise ValueError(
                            f"empty token row group at {shard_path}#row_group{rg_idx}"
                        )
                    min_id = int(flat_tokens.min())
                    max_id = int(flat_tokens.max())
                    if min_id < 0 or max_id >= vocab_size:
                        raise ValueError(
                            f"token ids [{min_id}, {max_id}] are outside vocab "
                            f"[0, {vocab_size}) at {shard_path}#row_group{rg_idx}"
                        )
                    if max_id > np.iinfo(dtype).max:
                        raise ValueError(
                            f"token id {max_id} exceeds output dtype "
                            f"{np.dtype(dtype).name} max {np.iinfo(dtype).max} at "
                            f"{shard_path}#row_group{rg_idx}"
                        )

                    row_starts = total_tokens + np.concatenate(
                        (np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1]))
                    )
                    pointers.extend((row_starts * dtype().itemsize).tolist())
                    sizes.extend(lengths.astype(np.int32, copy=False).tolist())
                    flat_tokens.astype(dtype, copy=False).tofile(bin_fh)

                    side_matrices: dict[str, np.ndarray] = {}
                    for col in (side_channels or []):
                        side_matrix, _ = _fixed_width_list_matrix(
                            side_cols[col],
                            column_name=col,
                            expected_rows=row_count,
                            expected_width=capacity,
                            shard_path=shard_path,
                            row_group_idx=rg_idx,
                        )
                        side_matrices[col] = side_matrix
                        flat_side = side_matrix[valid_mask].astype(
                            side_dtypes[col], copy=False
                        )
                        flat_side.tofile(side_writers[col])
                        if col == "loss_mask":
                            trained_tokens += int(flat_side.sum(dtype=np.int64))

                    for row_idx, token_count in enumerate(lengths.tolist()):
                        if graph_writers is not None:
                            graph_writers.append(
                                _graph_sidecar_values(graph_cols, row_idx),
                                shard_path=shard_path,
                                row_idx=row_idx,
                                token_count=token_count,
                            )
                        if source_platform_writer is not None:
                            source_platform_writer.append(
                                source_platform_col[row_idx],
                                doc_ids=side_matrices["doc_ids"][row_idx].tolist(),
                                token_count=token_count,
                                shard_path=shard_path,
                                row_idx=row_idx,
                            )
                    if graph_writers is not None:
                        graph_writers.flush()
                    if source_platform_writer is not None:
                        source_platform_writer.flush()

                    group_tokens = int(lengths.sum(dtype=np.int64))
                    total_tokens += group_tokens
                    source_capacity_tokens += row_count * capacity
                if (shard_idx + 1) % 10 == 0 or shard_idx + 1 == len(shards):
                    print(
                        f"  read {shard_idx + 1}/{len(shards)} shards, "
                        f"{len(sizes):,} docs, {total_tokens:,} tokens"
                    )
        if graph_writers is not None:
            graph_sidecar_paths = graph_writers.close()
        source_platform_paths = (
            source_platform_writer.close()
            if source_platform_writer is not None
            else None
        )
    except Exception:
        if graph_writers is not None:
            graph_writers.abort_close()
        if source_platform_writer is not None:
            source_platform_writer.abort_close()
        raise
    finally:
        for writer in side_writers.values():
            writer.close()

    print(f"total: {len(sizes)} documents")

    sizes_arr = np.array(sizes, dtype=np.int32)
    pointers_arr = np.array(pointers, dtype=np.int64)

    # Write .idx (Megatron MMapIndexedDataset format)
    idx_path = output_prefix + ".idx"
    MAGIC = b"MMIDIDX\x00\x00"  # 9 bytes
    VERSION = 1
    with open(idx_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", VERSION))
        f.write(struct.pack("<B", dtype_code))
        f.write(struct.pack("<Q", len(sizes_arr)))  # num sequences
        f.write(struct.pack("<Q", len(sizes_arr) + 1))  # num documents (includes sentinel)
        sizes_arr.tofile(f)
        pointers_arr.tofile(f)
        # Document indices (each doc is one sequence)
        doc_idx = np.arange(len(sizes_arr) + 1, dtype=np.int64)
        doc_idx.tofile(f)

    # Write JSON sidecar
    json_path = output_prefix + ".json"
    sidecar_data = {
        "vocab_size": vocab_size,
        "tokenizer_contract": "megacpp",
        "dtype": dtype_str,
        "token_count": total_tokens,
        "source_capacity_token_count": source_capacity_tokens,
        "trained_token_count": trained_tokens if side_channels and "loss_mask" in side_channels else None,
        "document_count": len(sizes_arr),
        "token_column": token_column,
        "length_column": length_column,
        "writer_backend": "mmididx",
    }
    if side_channels:
        side_channel_paths = {}
        for col, dt_str in zip(side_channels, side_channel_dtypes or [], strict=True):
            side_channel_paths[col] = {
                "path": f"{os.path.basename(output_prefix)}_{col}.bin",
                "dtype": dt_str,
            }
        sidecar_data["side_channel_paths"] = side_channel_paths
    _add_graph_manifest(sidecar_data, graph_sidecar_paths)
    _add_source_platform_manifest(sidecar_data, source_platform_paths)
    if symbol_identity_schema_version is not None:
        _add_symbol_identity_manifest(sidecar_data, symbol_identity_schema_version)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(sidecar_data, jf, indent=4)

    elapsed = time.time() - t0
    bin_size = os.path.getsize(bin_path) / (1024**3)
    print(f"\n{split}: {len(sizes_arr)} docs, {total_tokens:,} tokens, {bin_size:.2f} GiB in {elapsed:.1f}s")
    print(f"output: {bin_path} + {idx_path}")
    return


def convert_parquet_to_megatron(
    input_dir: str,
    output_prefix: str,
    split: str = "train",
    token_column: str = "auto",
    dtype_str: str = "uint16",
    length_column: str | None = None,
    side_channels: list[str] | None = None,
    side_channel_dtypes: list[str] | None = None,
    graph_sidecars: tuple[tuple[str, str, str], ...] | None = DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    source_platform_sidecar: bool | None = None,
    vocab_size: int = 65536,
    writer_backend: str = "megatron",
) -> None:
    """Convert packed parquet to Megatron MMapIndexedDataset format.

    ``writer_backend='mmididx'`` is an explicit local writer for the same v1
    MMIDIDX layout.  It is never selected as an automatic fallback when the
    Megatron import is broken.
    """
    import pyarrow.parquet as pq
    import json

    if side_channels is None and side_channel_dtypes is None:
        side_channels = [name for name, _ in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]
        side_channel_dtypes = [dtype for _, dtype in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]

    if writer_backend == "mmididx":
        return _convert_parquet_to_numpy(
            input_dir=input_dir,
            output_prefix=output_prefix,
            split=split,
            token_column=token_column,
            dtype_str=dtype_str,
            length_column=length_column,
            side_channels=side_channels,
            side_channel_dtypes=side_channel_dtypes,
            graph_sidecars=graph_sidecars,
            source_platform_sidecar=source_platform_sidecar,
            vocab_size=vocab_size,
        )
    if writer_backend != "megatron":
        raise ValueError(f"unsupported writer backend: {writer_backend}")

    # Import Megatron's dataset builder. A missing/broken Megatron install must
    # FAIL LOUD: silently falling back to the raw-numpy writer would emit a
    # different on-disk layout that the training pipeline may not consume, so a
    # broken env would masquerade as a successful conversion.
    try:
        from megatron.core.datasets.indexed_dataset import (  # pyright: ignore[reportMissingImports]
            IndexedDatasetBuilder as MMapIndexedDatasetBuilder,
        )
    except Exception as e:
        raise RuntimeError(
            "convert_parquet_to_megatron: failed to import "
            "megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder "
            f"({type(e).__name__}: {e}). Install/repair Megatron-Core "
            "(pip install megatron-core) so the Megatron IndexedDataset writer "
            "is available; refusing to silently fall back to the raw-numpy writer "
            "which emits an incompatible on-disk format."
        ) from e

    shards = find_parquet_shards(input_dir, split)
    token_column = _resolve_token_column(shards, token_column)
    length_column = _resolve_length_column(shards, length_column)
    symbol_identity_schema_version = _require_symbol_identity_schema(shards)
    _require_symbol_sidecar_dtypes(side_channels, side_channel_dtypes)
    write_source_platform = _resolve_source_platform_sidecar(
        shards, source_platform_sidecar
    )
    if write_source_platform and "doc_ids" not in (side_channels or []):
        raise ValueError(
            "source platform sidecar requires token-aligned doc_ids sidecar"
        )
    print(f"found {len(shards)} {split} shards in {input_dir}")

    # Determine dtype
    dtype = _resolve_output_dtype(dtype_str)

    # Create output directory
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    builder = MMapIndexedDatasetBuilder(output_prefix + ".bin", dtype=dtype)

    # Open side channel writers
    side_writers = {}
    side_dtypes = {}
    if side_channels:
        for col, dt_str in zip(side_channels, side_channel_dtypes or [], strict=True):
            side_bin_path = f"{output_prefix}_{col}.bin"
            side_writers[col] = open(side_bin_path, "wb")
            side_dtypes[col] = np.dtype(dt_str)

    graph_writers = _GraphSidecarWriters(output_prefix, graph_sidecars) if graph_sidecars else None
    source_platform_writer = (
        _SourcePlatformSidecarWriter(output_prefix)
        if write_source_platform
        else None
    )
    graph_columns = graph_writers.columns if graph_writers is not None else []
    graph_sidecar_paths: dict[str, dict[str, object]] | None = None

    total_docs = 0
    total_tokens = 0
    source_capacity_tokens = 0
    trained_tokens = 0
    t0 = time.time()

    columns_to_read = _unique_columns(
        [token_column],
        [length_column] if length_column else None,
        side_channels,
        graph_columns,
        [SOURCE_PLATFORM_IDS_COLUMN] if write_source_platform else None,
    )

    try:
        for shard_idx, shard_path in enumerate(shards):
            pf = pq.ParquetFile(shard_path)
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=columns_to_read)
                token_col = table.column(token_column)
                length_col = table.column(length_column) if length_column else None
                side_cols = {col: table.column(col) for col in (side_channels or [])}
                graph_cols = {col: table.column(col) for col in graph_columns}
                source_platform_col = (
                    table.column(SOURCE_PLATFORM_IDS_COLUMN)
                    if write_source_platform
                    else None
                )
                for row_idx in range(len(token_col)):
                    raw_token_ids = token_col[row_idx].as_py()
                    if not raw_token_ids:
                        continue
                    token_count = _row_token_length(
                        length_col[row_idx] if length_col is not None else None,
                        len(raw_token_ids),
                        length_column=length_column,
                        shard_path=shard_path,
                        row_idx=row_idx,
                    )
                    token_ids = raw_token_ids[:token_count]
                    _validate_token_ids(
                        token_ids,
                        dtype=dtype,
                        vocab_size=vocab_size,
                        shard_path=shard_path,
                        row_idx=row_idx,
                    )
                    arr = np.array(token_ids, dtype=dtype)
                    builder.add_document(arr, [len(arr)])

                    # Write aligned side channel values
                    for col in (side_channels or []):
                        side_val = _require_token_aligned_side_channel(
                            col,
                            side_cols[col][row_idx].as_py(),
                            raw_token_ids,
                            shard_path=shard_path,
                            row_idx=row_idx,
                        )
                        trimmed_side = side_val[:token_count]
                        arr_side = np.array(trimmed_side, dtype=side_dtypes[col])
                        arr_side.tofile(side_writers[col])
                        if col == "loss_mask":
                            trained_tokens += sum(int(value) for value in trimmed_side)
                    if graph_writers is not None:
                        graph_writers.append(
                            _graph_sidecar_values(graph_cols, row_idx),
                            shard_path=shard_path,
                            row_idx=row_idx,
                            token_count=token_count,
                        )
                    if source_platform_writer is not None:
                        source_platform_writer.append(
                            source_platform_col[row_idx],
                            doc_ids=side_cols["doc_ids"][row_idx].as_py(),
                            token_count=token_count,
                            shard_path=shard_path,
                            row_idx=row_idx,
                        )

                    total_docs += 1
                    total_tokens += len(arr)
                    source_capacity_tokens += len(raw_token_ids)

            elapsed = time.time() - t0
            print(
                f"  shard {shard_idx + 1}/{len(shards)}: "
                f"{total_docs:,} docs, {total_tokens:,} tokens "
                f"({elapsed:.1f}s)"
            )

        builder.finalize(output_prefix + ".idx")
        if graph_writers is not None:
            graph_sidecar_paths = graph_writers.close()
        source_platform_paths = (
            source_platform_writer.close()
            if source_platform_writer is not None
            else None
        )
    except Exception:
        if graph_writers is not None:
            graph_writers.abort_close()
        if source_platform_writer is not None:
            source_platform_writer.abort_close()
        raise
    finally:
        for writer in side_writers.values():
            writer.close()

    # Write JSON sidecar
    json_path = output_prefix + ".json"
    sidecar_data = {
        "vocab_size": vocab_size,
        "tokenizer_contract": "megacpp",
        "dtype": dtype_str,
        "token_count": total_tokens,
        "source_capacity_token_count": source_capacity_tokens,
        "trained_token_count": trained_tokens if side_channels and "loss_mask" in side_channels else None,
        "document_count": total_docs,
        "token_column": token_column,
        "length_column": length_column,
        "writer_backend": "megatron",
    }
    if side_channels:
        side_channel_paths = {}
        for col, dt_str in zip(side_channels, side_channel_dtypes or [], strict=True):
            side_channel_paths[col] = {
                "path": f"{os.path.basename(output_prefix)}_{col}.bin",
                "dtype": dt_str,
            }
        sidecar_data["side_channel_paths"] = side_channel_paths
    _add_graph_manifest(sidecar_data, graph_sidecar_paths)
    _add_source_platform_manifest(sidecar_data, source_platform_paths)
    if symbol_identity_schema_version is not None:
        _add_symbol_identity_manifest(sidecar_data, symbol_identity_schema_version)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(sidecar_data, jf, indent=4)

    elapsed = time.time() - t0
    print(
        f"\n{split} conversion complete: "
        f"{total_docs:,} documents, {total_tokens:,} tokens "
        f"in {elapsed:.1f}s"
    )
    print(f"output: {output_prefix}.bin + {output_prefix}.idx")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert nanochat tokenized parquet to Megatron indexed binary"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing parquet shards",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix for .bin and .idx files",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="all",
        help="Which split to convert (default: all; safest for bucketed parquet)",
    )
    parser.add_argument(
        "--token-column",
        default="auto",
        help="Parquet token column, or auto for exactly one of input_ids/token_ids",
    )
    parser.add_argument(
        "--length-column",
        default="auto",
        help=(
            "Optional packed valid-length column. auto uses valid_token_count "
            "when present; use none to preserve padded row capacity."
        ),
    )
    parser.add_argument(
        "--writer-backend",
        choices=["megatron", "mmididx"],
        default="megatron",
        help=(
            "Indexed writer implementation. mmididx is an explicit compatible "
            "v1 writer and is never an automatic fallback."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["uint8", "uint16", "int32", "int64", "uint32"],
        default="uint16",
        help=(
            "Output dtype for token IDs. uint32 is accepted as a deprecated "
            "alias that writes int32, because Megatron MMIDIDX has no uint32 "
            "dtype code."
        ),
    )
    parser.add_argument(
        "--side-channels",
        default=None,
        help=(
            "Comma-separated side-channel columns to convert. Overrides the "
            "default cppmega-full token-aligned sidecar profile."
        ),
    )
    parser.add_argument(
        "--side-channel-dtypes",
        default=None,
        help=(
            "Comma-separated side-channel dtypes. Required when "
            "--side-channels is explicitly provided."
        ),
    )
    parser.add_argument(
        "--no-side-channels",
        action="store_true",
        help=(
            "Write only token .bin/.idx. This also disables graph sidecars. "
            "It is legacy flat-LM behavior and must be explicit; cppmega-full "
            "sidecars are the default."
        ),
    )
    parser.add_argument(
        "--no-graph-sidecars",
        action="store_true",
        help=(
            "Disable document-aligned graph route sidecars while keeping any "
            "token-aligned side channels. Default writes token_call_edges, "
            "token_type_edges, and token_chunk_* CSR sidecars."
        ),
    )
    parser.add_argument(
        "--source-platform-sidecar",
        choices=["auto", "require", "off"],
        default="auto",
        help=(
            "Preserve packed source_platform_ids as compact nested CSR. "
            "auto enables it when the column exists; require fails if absent."
        ),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=65536,
        help="Tokenizer vocab size to write to the JSON sidecar (default: 65536)",
    )
    args = parser.parse_args()

    if args.no_side_channels and args.side_channels:
        raise ValueError("--no-side-channels cannot be combined with --side-channels")
    if args.no_side_channels and args.side_channel_dtypes:
        raise ValueError("--no-side-channels cannot be combined with --side-channel-dtypes")

    if args.no_side_channels:
        side_channels_list = []
        side_channel_dtypes_list = []
    elif args.side_channels:
        side_channels_list = [x.strip() for x in args.side_channels.split(",") if x.strip()]
        side_channel_dtypes_list = (
            [x.strip() for x in args.side_channel_dtypes.split(",") if x.strip()]
            if args.side_channel_dtypes
            else None
        )
    else:
        side_channels_list = [name for name, _ in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]
        side_channel_dtypes_list = [dtype for _, dtype in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]

    if side_channels_list and not side_channel_dtypes_list:
        raise ValueError("--side-channel-dtypes must be specified when --side-channels is provided")
    if side_channels_list and side_channel_dtypes_list and len(side_channels_list) != len(side_channel_dtypes_list):
        raise ValueError("Number of --side-channels must match number of --side-channel-dtypes")
    graph_sidecars = (
        None
        if args.no_side_channels or args.no_graph_sidecars
        else DEFAULT_CPPMEGA_GRAPH_SIDECARS
    )
    source_platform_sidecar = {
        "auto": None,
        "require": True,
        "off": False,
    }[args.source_platform_sidecar]
    if args.no_side_channels:
        source_platform_sidecar = False

    convert_parquet_to_megatron(
        input_dir=args.input_dir,
        output_prefix=args.output_prefix,
        split=args.split,
        token_column=args.token_column,
        dtype_str=args.dtype,
        length_column=args.length_column,
        side_channels=side_channels_list,
        side_channel_dtypes=side_channel_dtypes_list,
        graph_sidecars=graph_sidecars,
        source_platform_sidecar=source_platform_sidecar,
        vocab_size=args.vocab_size,
        writer_backend=args.writer_backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
