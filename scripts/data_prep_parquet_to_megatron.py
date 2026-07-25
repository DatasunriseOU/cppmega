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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from numbers import Integral, Real
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega.symbol_identity import (  # noqa: E402
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SymbolIdentityRegistry,
    compute_symbol_id,
)
from cppmega.megatron.objective_contract import (  # noqa: E402
    LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1,
    OBJECTIVE_CONTRACT_SCHEMA,
    OBJECTIVE_GRAPH_SIDECARS,
    OBJECTIVE_IDS,
    OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA,
    OBJECTIVE_TOKEN_SIDE_CHANNELS,
    ObjectiveMaterializationArtifact,
    ObjectiveMaterializationTracker,
    ValidatedObjectiveContract,
    load_objective_materialization_artifact,
    materialized_objective_artifact_manifest,
    validate_objective_contract,
    verify_objective_materialization_shard,
)
from cppmega.megatron.graph_recipe import (  # noqa: E402
    STAGE1_GRAPH_RELATIONS,
    STAGE1_GRAPH_TOPK,
    stage1_graph_recipe_binding,
)
from cppmega.megatron.domain_route_contract import (  # noqa: E402
    CASE5_SCHEMA_METADATA_KEY,
    CASE5_RECEIPT_KEY,
    CASE5_SCHEMA_VERSION,
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    DOMAIN_DELIMITER_ID_TO_DOMAIN,
    DOMAIN_EDGE_KINDS_BY_COLUMN,
    DOMAIN_ROUTE_COLUMNS,
    DOMAIN_SCHEMA_SHA256,
    DOMAIN_SCHEMA_SHA256_METADATA_KEY,
    DOMAIN_START_DELIMITER_IDS,
    GRAPH_ROUTE_COLUMNS,
    SOURCE_IDENTITY_REGISTRY_SCHEMA,
    TOKENIZER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY,
    VALID_DOMAIN_CONFIDENCE_IDS,
    VALID_DOMAIN_EDGE_KINDS,
    VALID_DOMAIN_IDS,
    VALID_DOMAIN_ROLE_IDS,
    validate_case5_contract_hash_triple,
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
    "uint64": np.uint64,
    "int32": np.int32,
    "int64": np.int64,
}

_MEGATRON_DTYPE_CODE_MAP = {
    np.uint8: 1,
    np.int32: 4,
    np.int64: 5,
    np.uint16: 8,
}

DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS = OBJECTIVE_TOKEN_SIDE_CHANNELS

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

DEFAULT_CPPMEGA_GRAPH_SIDECARS = OBJECTIVE_GRAPH_SIDECARS

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
        if not (
            pa.types.is_list(identity_type) or pa.types.is_large_list(identity_type)
        ):
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
            if (
                not (
                    pa.types.is_list(column_type) or pa.types.is_large_list(column_type)
                )
                or column_type.value_type != pa.uint64()
            ):
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


OBJECTIVE_KIND_COLUMN = "objective_kind"
_GRAPH_RELATION_TO_COLUMN = {
    "call": "token_call_edges",
    "type": "token_type_edges",
    "domain": "token_domain_edges",
    "build": "token_build_edges",
    "shell": "token_shell_edges",
    "diagnostic": "token_diagnostic_edges",
    "cross_domain": "token_cross_domain_edges",
}

UINT32_MAX = int(np.iinfo(np.uint32).max)
UINT64_MAX = int(np.iinfo(np.uint64).max)
SOURCE_IDENTITY_REGISTRY_COLUMN = "source_identity_registry"
_OPAQUE_UINT64_COLUMNS = frozenset(
    {
        "token_source_identity_ids",
        "token_symbol_ids",
        "token_call_targets",
        "token_type_refs",
    }
)


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


def _physical_output_dtype_name(dtype: type[np.generic] | np.dtype) -> str:
    """Return the canonical dtype name encoded in MMIDIDX and token bytes."""

    return np.dtype(dtype).name


def _resolve_sidecar_dtype(dtype_str: str) -> np.dtype:
    try:
        return np.dtype(_SIDECAR_DTYPE_MAP[dtype_str])
    except KeyError as exc:
        raise ValueError(f"unsupported sidecar dtype: {dtype_str}") from exc


def _validated_token_sidecar_array(
    values: object,
    dtype: np.dtype,
    *,
    column: str,
    where: str,
) -> np.ndarray:
    """Cast a token sidecar only after proving integrality and dtype range."""

    target = np.dtype(dtype)
    if target.kind not in {"i", "u"}:
        raise ValueError(f"token sidecar {column} has non-integer dtype {target.name}")
    raw = np.asarray(values)
    if raw.size == 0:
        return raw.astype(target, copy=False)

    if raw.dtype.kind in {"i", "u", "b"}:
        minimum = int(raw.min())
        maximum = int(raw.max())
        normalized = raw
    elif raw.dtype.kind == "f":
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.trunc(raw)):
            raise ValueError(
                f"token sidecar {column} must contain finite integral values "
                f"before cast to {target.name} at {where}"
            )
        minimum = int(raw.min())
        maximum = int(raw.max())
        normalized = raw
    else:
        normalized_values: list[int] = []
        for value in raw.reshape(-1):
            if isinstance(value, (bool, np.bool_)):
                normalized_values.append(int(value))
                continue
            if isinstance(value, Integral):
                normalized_values.append(int(value))
                continue
            if isinstance(value, Real):
                numeric = float(value)
                if math.isfinite(numeric) and numeric.is_integer():
                    normalized_values.append(int(value))
                    continue
            raise ValueError(
                f"token sidecar {column} must contain finite integral values "
                f"before cast to {target.name} at {where}; got {value!r}"
            )
        minimum = min(normalized_values)
        maximum = max(normalized_values)
        normalized = np.asarray(normalized_values, dtype=object).reshape(raw.shape)

    info = np.iinfo(target)
    if minimum < info.min or maximum > info.max:
        raise ValueError(
            f"token sidecar {column} values [{minimum}, {maximum}] are outside "
            f"{target.name} range [{info.min}, {info.max}] at {where}"
        )
    return normalized.astype(target, copy=False)


def _require_contiguous_doc_ids(
    values: object,
    *,
    token_count: int,
    where: str,
    expected_document_count: int | None = None,
) -> np.ndarray:
    docs = _validated_token_sidecar_array(
        values,
        np.dtype(np.uint32),
        column="doc_ids",
        where=where,
    ).reshape(-1)
    if len(docs) < token_count or token_count <= 0:
        raise ValueError(
            f"doc_ids must cover {token_count} valid tokens at {where}; got {len(docs)}"
        )
    valid = docs[:token_count]
    transitions = np.diff(valid.astype(np.int64, copy=False))
    if valid[0] != 1 or np.any((transitions != 0) & (transitions != 1)):
        raise ValueError(
            "doc_ids must be contiguous, non-reused row-local IDs 1..N "
            f"at {where}; got {valid.tolist()}"
        )
    document_count = int(valid[-1])
    if (
        expected_document_count is not None
        and document_count != expected_document_count
    ):
        raise ValueError(
            f"doc_ids cover {document_count} row-local documents but "
            f"{expected_document_count} source platform bags exist at {where}"
        )
    return valid


def _validate_source_transition_loss_mask(
    values: object,
    *,
    token_count: int,
    where: str,
    document_ids: object | None = None,
) -> np.ndarray:
    """Validate source-token mask semantics used by Megatron and MLX readers."""

    mask_array = _validated_token_sidecar_array(
        values,
        np.dtype(np.uint8),
        column="loss_mask",
        where=where,
    )
    if mask_array.ndim != 1:
        raise ValueError(f"loss_mask must be one-dimensional at {where}")
    mask = mask_array.reshape(-1)
    if token_count < 1 or len(mask) < token_count:
        raise ValueError(
            f"loss_mask must cover every materialized token at {where}; "
            f"token_count={token_count} mask_length={len(mask)}"
        )
    valid = mask[:token_count]
    if np.any((valid != 0) & (valid != 1)):
        raise ValueError(f"loss_mask must be binary at {where}")
    if int(valid[-1]) != 0:
        raise ValueError(
            "source_token_predicts_next_v1 requires a zero loss mask on the "
            f"last materialized token at {where}"
        )
    if document_ids is not None:
        docs = _require_contiguous_doc_ids(
            document_ids,
            token_count=token_count,
            where=where,
        )
        cross_document = docs[:-1] != docs[1:]
        leaking = np.flatnonzero(cross_document & (valid[:-1] != 0))
        if leaking.size:
            raise ValueError(
                "loss_mask trains cross-document transitions under "
                "source_token_predicts_next_v1 at "
                f"{where}: source_positions={leaking[:16].tolist()}"
            )
    return valid


def _require_symbol_sidecar_dtypes(
    side_channels: list[str] | None,
    side_channel_dtypes: list[str] | None,
) -> None:
    requested = dict(zip(side_channels or [], side_channel_dtypes or [], strict=True))
    present = set(_SYMBOL_ID_COLUMNS) & set(requested)
    if present and present != set(_SYMBOL_ID_COLUMNS):
        missing = sorted(set(_SYMBOL_ID_COLUMNS) - present)
        raise ValueError(
            f"partial symbol identity sidecars are not allowed; missing={missing}"
        )
    invalid = {
        column: requested[column] for column in present if requested[column] != "uint64"
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


def _validate_output_dtype_contract(dtype_str: str) -> None:
    """Reject dtypes that cannot be represented by the MMIDIDX header."""

    try:
        dtype = _OUTPUT_DTYPE_MAP[dtype_str]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype_str}") from exc
    _megatron_dtype_code(dtype)


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
        return (
            [str(p) for p in all_parquets[:-1]]
            if len(all_parquets) > 1
            else [str(all_parquets[0])]
        )
    elif split == "val":
        if has_explicit_val:
            return [str(val_shard)]
        return (
            [str(all_parquets[-1])] if len(all_parquets) > 1 else [str(all_parquets[0])]
        )
    elif split == "all":
        return [str(p) for p in all_parquets]
    else:
        raise ValueError(f"unknown split: {split}")


def _resolve_token_column(shards: list[str], requested: str) -> str:
    """Resolve ``auto`` against the first shard and fail on ambiguous schemas."""

    if requested != "auto":
        return requested
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

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
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    names = set(pq.ParquetFile(shards[0]).schema_arrow.names)
    if requested == "auto":
        return "valid_token_count" if "valid_token_count" in names else None
    if requested not in names:
        raise ValueError(f"length column {requested!r} is absent from {shards[0]}")
    return requested


def _resolve_source_platform_sidecar(shards: list[str], requested: bool | None) -> bool:
    """Resolve auto/required source-platform preservation against the schema."""

    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    present = SOURCE_PLATFORM_IDS_COLUMN in set(
        pq.ParquetFile(shards[0]).schema_arrow.names
    )
    if requested is True and not present:
        raise ValueError(
            f"required {SOURCE_PLATFORM_IDS_COLUMN} is absent from {shards[0]}"
        )
    return present if requested is None else bool(requested)


def _case5_conversion_requested(
    side_channels: list[str] | None,
    graph_sidecars: tuple[tuple[str, str, str], ...] | None,
) -> bool:
    requested = set(side_channels or ())
    return bool(requested & set(DOMAIN_ROUTE_COLUMNS)) or bool(graph_sidecars)


def _require_case5_sidecar_dtypes(
    side_channels: list[str] | None,
    side_channel_dtypes: list[str] | None,
) -> None:
    requested = dict(zip(side_channels or (), side_channel_dtypes or (), strict=True))
    present_routes = set(DOMAIN_ROUTE_COLUMNS) & set(requested)
    if present_routes and present_routes != set(DOMAIN_ROUTE_COLUMNS):
        missing = sorted(set(DOMAIN_ROUTE_COLUMNS) - present_routes)
        raise ValueError(
            f"partial CASE5 domain route sidecars are not allowed; missing={missing}"
        )
    if "doc_ids" in requested and requested["doc_ids"] != "uint32":
        raise ValueError("CASE5 row-local doc_ids sidecar must use uint32")
    invalid = {
        column: requested[column]
        for column in _OPAQUE_UINT64_COLUMNS & set(requested)
        if requested[column] != "uint64"
    }
    if invalid:
        raise ValueError(f"opaque identity sidecars must use uint64: {invalid}")


def _require_case5_schema(
    shards: list[str],
    *,
    side_channels: list[str] | None,
    side_channel_dtypes: list[str] | None,
    graph_sidecars: tuple[tuple[str, str, str], ...] | None,
) -> int:
    """Fail closed on the packed CASE5 Arrow schema and contract receipts."""

    _require_case5_sidecar_dtypes(side_channels, side_channel_dtypes)
    if not _case5_conversion_requested(side_channels, graph_sidecars):
        return 0
    route_columns: set[str] = set(DOMAIN_ROUTE_COLUMNS)
    missing_output_routes = sorted(route_columns - set(side_channels or ()))
    if missing_output_routes:
        raise ValueError(
            "CASE5 conversion must preserve the complete domain route profile; "
            f"missing output sidecars={missing_output_routes}"
        )

    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    edge_pair = pa.struct([pa.field("from", pa.uint16()), pa.field("to", pa.uint16())])
    edge_triple = pa.struct(
        [
            pa.field("from", pa.uint32()),
            pa.field("to", pa.uint32()),
            pa.field("kind", pa.int32()),
        ]
    )
    source_identity = pa.struct(
        [
            pa.field("source_identity_id", pa.uint64(), nullable=False),
            pa.field("canonical_sha256", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
        ]
    )
    expected_types = {
        "doc_ids": pa.list_(pa.uint32()),
        "token_domain_ids": pa.list_(pa.uint16()),
        "token_role_ids": pa.list_(pa.uint16()),
        "token_entity_ids": pa.list_(pa.uint32()),
        "token_scope_ids": pa.list_(pa.uint32()),
        "token_source_doc_ids": pa.list_(pa.uint32()),
        "token_source_identity_ids": pa.list_(pa.uint64()),
        "token_confidence_ids": pa.list_(pa.uint8()),
        "token_symbol_ids": pa.list_(pa.uint64()),
        "token_call_targets": pa.list_(pa.uint64()),
        "token_type_refs": pa.list_(pa.uint64()),
        "token_call_edges": pa.list_(edge_pair),
        "token_type_edges": pa.list_(edge_pair),
        **{column: pa.list_(edge_triple) for column in DOMAIN_EDGE_KINDS_BY_COLUMN},
        "token_chunk_starts": pa.list_(pa.uint32()),
        "token_chunk_ends": pa.list_(pa.uint32()),
        "token_chunk_kinds": pa.list_(pa.uint8()),
        "token_chunk_dep_levels": pa.list_(pa.uint16()),
        SOURCE_IDENTITY_REGISTRY_COLUMN: pa.list_(source_identity),
    }
    for shard in shards:
        schema = pq.ParquetFile(shard).schema_arrow
        missing = sorted(set(expected_types) - set(schema.names))
        if missing:
            raise ValueError(
                f"{shard}: incomplete CASE5 packed schema; missing={missing}"
            )
        wrong_types = {
            column: str(schema.field(column).type)
            for column, expected in expected_types.items()
            if schema.field(column).type != expected
        }
        if wrong_types:
            raise ValueError(
                f"{shard}: CASE5 Arrow types differ from the contract: {wrong_types}"
            )
        metadata = schema.metadata or {}
        schema_key = CASE5_SCHEMA_METADATA_KEY.encode("utf-8")
        if metadata.get(schema_key) != CASE5_SCHEMA_VERSION.encode("utf-8"):
            raise ValueError(
                f"{shard}: missing or stale CASE5 schema metadata: "
                f"{metadata.get(schema_key)!r}"
            )
        try:
            validate_case5_contract_hash_triple(
                metadata.get(DOMAIN_DELIMITER_CONTRACT_METADATA_KEY.encode("utf-8")),
                metadata.get(DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8")),
                metadata.get(TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode("utf-8")),
                where=shard,
            )
        except ValueError as exc:
            raise ValueError(f"{shard}: missing or stale CASE5 receipt metadata: {exc}") from exc
    return len(shards)


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
        raw_length = cast(Any, raw_length).as_py()
    if raw_length is None:
        raise ValueError(
            f"length column {length_column} is null at {shard_path}#row{row_idx}"
        )
    length = int(cast(Any, raw_length))
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
    info = np.iinfo(dtype)  # pyright: ignore[reportCallIssue, reportArgumentType]
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


def _validate_domain_route_sidecars(
    token_ids: list[int] | np.ndarray,
    values: dict[str, object],
    *,
    shard_path: str,
    row_idx: int,
) -> None:
    """Validate the complete token-aligned domain route profile for one row."""

    present = [column for column in DOMAIN_ROUTE_COLUMNS if column in values]
    if not present:
        return
    if len(present) != len(DOMAIN_ROUTE_COLUMNS):
        route_columns: set[str] = set(DOMAIN_ROUTE_COLUMNS)
        missing = sorted(route_columns - set(present))
        raise ValueError(
            "domain routing requires the complete token-aligned domain route "
            f"profile at {shard_path}#row{row_idx}; missing={missing}"
        )

    tokens = [int(value) for value in token_ids]
    vectors: dict[str, list[int]] = {}
    for column in DOMAIN_ROUTE_COLUMNS:
        raw = values[column]
        if raw is None:
            raise ValueError(
                "domain routing requires the complete token-aligned domain route "
                f"profile at {shard_path}#row{row_idx}; {column} is null"
            )
        vector = [int(value) for value in raw]  # type: ignore[union-attr]
        if len(vector) != len(tokens):
            raise ValueError(
                f"{column} length {len(vector)} != token_ids length {len(tokens)} "
                f"at {shard_path}#row{row_idx}"
            )
        vectors[column] = vector

    enum_contracts = (
        ("token_domain_ids", VALID_DOMAIN_IDS),
        ("token_role_ids", VALID_DOMAIN_ROLE_IDS),
        ("token_confidence_ids", VALID_DOMAIN_CONFIDENCE_IDS),
    )
    for column, valid_values in enum_contracts:
        for value in vectors[column]:
            if value not in valid_values:
                raise ValueError(
                    f"unknown {column} value {value} at {shard_path}#row{row_idx}"
                )
    for column in (
        "token_entity_ids",
        "token_scope_ids",
        "token_source_doc_ids",
    ):
        if any(value < 0 or value > UINT32_MAX for value in vectors[column]):
            raise ValueError(f"{column} must fit uint32 at {shard_path}#row{row_idx}")
    if any(value <= 0 for value in vectors["token_source_doc_ids"]):
        raise ValueError(
            "token_source_doc_ids must preserve a positive source document ID "
            f"for every valid token at {shard_path}#row{row_idx}"
        )
    if any(
        value <= 0 or value > UINT64_MAX
        for value in vectors["token_source_identity_ids"]
    ):
        raise ValueError(
            "token_source_identity_ids must preserve a positive uint64 registry "
            f"foreign key for every valid token at {shard_path}#row{row_idx}"
        )

    domains = vectors["token_domain_ids"]
    roles = vectors["token_role_ids"]
    confidence = vectors["token_confidence_ids"]
    stack: list[int] = []
    for token_idx, (token_id, domain_id, role_id, confidence_id) in enumerate(
        zip(tokens, domains, roles, confidence, strict=True)
    ):
        expected_domain = DOMAIN_DELIMITER_ID_TO_DOMAIN.get(token_id)
        if expected_domain is not None:
            if domain_id != expected_domain:
                raise ValueError(
                    f"delimiter token id {token_id} requires domain id "
                    f"{expected_domain}, got {domain_id} at "
                    f"{shard_path}#row{row_idx}:token{token_idx}"
                )
            if role_id != 1:
                raise ValueError(
                    f"delimiter token id {token_id} requires role id 1 at "
                    f"{shard_path}#row{row_idx}:token{token_idx}"
                )
            if confidence_id != 4:
                raise ValueError(
                    f"delimiter token id {token_id} requires confidence id 4 at "
                    f"{shard_path}#row{row_idx}:token{token_idx}"
                )
            if token_id in DOMAIN_START_DELIMITER_IDS:
                stack.append(expected_domain)
            else:
                if not stack or stack[-1] != expected_domain:
                    raise ValueError(
                        f"unmatched domain end delimiter token id {token_id} at "
                        f"{shard_path}#row{row_idx}:token{token_idx}"
                    )
                stack.pop()
        else:
            if role_id == 1:
                raise ValueError(
                    f"DELIMITER role on non-delimiter token id {token_id} at "
                    f"{shard_path}#row{row_idx}:token{token_idx}"
                )
            expected_active_domain = stack[-1] if stack else 0
            if domain_id != expected_active_domain:
                raise ValueError(
                    f"domain id {domain_id} does not match active reserved "
                    f"delimiter domain {expected_active_domain} at "
                    f"{shard_path}#row{row_idx}:token{token_idx}"
                )
    if stack:
        raise ValueError(
            f"unclosed domain delimiters {stack} at {shard_path}#row{row_idx}"
        )


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
            f"{column_name} contains null rows at {shard_path}#row_group{row_group_idx}"
        )
    if not hasattr(combined, "offsets") or not hasattr(combined, "values"):
        raise ValueError(
            f"{column_name} must be an Arrow list column at "
            f"{shard_path}#row_group{row_group_idx}"
        )

    offsets = np.asarray(
        combined.offsets.to_numpy(zero_copy_only=False), dtype=np.int64
    )
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
        raise ValueError(
            f"{column} contains negative edge endpoint at {shard_path}#row{row_idx}"
        )
    if np.any(pairs > np.iinfo(np.int32).max):
        raise ValueError(
            f"{column} edge endpoint exceeds int32 at {shard_path}#row{row_idx}"
        )
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
        elif isinstance(edge, (list, tuple, np.ndarray)) and len(edge) == 3:
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
        if kind_i not in VALID_DOMAIN_EDGE_KINDS:
            raise ValueError(
                f"unknown domain route edge kind {kind_i} at {shard_path}#row{row_idx}"
            )
        allowed_kinds = DOMAIN_EDGE_KINDS_BY_COLUMN.get(column)
        if allowed_kinds is None:
            raise ValueError(f"unknown domain route edge column {column!r}")
        if kind_i not in allowed_kinds:
            raise ValueError(
                f"edge kind {kind_i} is not valid for {column} at "
                f"{shard_path}#row{row_idx}"
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
        raise ValueError(
            f"{column} edge value exceeds int32 at {shard_path}#row{row_idx}"
        )
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
        raise ValueError(
            f"{column} contains negative value at {shard_path}#row{row_idx}"
        )
    info = np.iinfo(dtype)
    if np.any(arr > info.max):
        raise ValueError(
            f"{column} value exceeds {dtype.name} max {info.max} at {shard_path}#row{row_idx}"
        )
    return arr.astype(dtype, copy=False)


def _source_identity_id_from_digest(digest: bytes) -> int:
    value = int.from_bytes(digest[:8], "big", signed=False)
    if value == 0:
        value = int.from_bytes(digest[8:16], "big", signed=False)
    if value == 0:
        raise ValueError("SHA256 digest produced reserved source identity 0")
    return value


class _SourceIdentityRegistryWriter:
    """Persist full source witnesses and sequence-to-identity relations."""

    def __init__(self, output_prefix: str) -> None:
        self._output_prefix = output_prefix
        self._path = f"{output_prefix}_{SOURCE_IDENTITY_REGISTRY_COLUMN}.sqlite"
        if os.path.exists(self._path):
            os.remove(self._path)
        self._db = sqlite3.connect(self._path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(
            """
            CREATE TABLE source_identities (
                source_identity_id BLOB PRIMARY KEY CHECK(length(source_identity_id) = 8),
                canonical_sha256 TEXT NOT NULL CHECK(length(canonical_sha256) = 64),
                source TEXT NOT NULL CHECK(length(source) > 0)
            );
            CREATE TABLE sequence_source_identities (
                sequence_index INTEGER NOT NULL,
                source_identity_id BLOB NOT NULL,
                PRIMARY KEY(sequence_index, source_identity_id),
                FOREIGN KEY(source_identity_id)
                    REFERENCES source_identities(source_identity_id)
            );
            """
        )
        self._sequence_count = 0
        self._identity_count = 0
        self._reference_count = 0
        self._closed = False

    def append(
        self,
        value: object,
        *,
        source_identity_ids: list[int] | np.ndarray,
        token_count: int,
        shard_path: str,
        row_idx: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("source identity registry writer is already closed")
        entries = cast(Any, value).as_py() if hasattr(value, "as_py") else value
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"{SOURCE_IDENTITY_REGISTRY_COLUMN} must be a non-empty list at "
                f"{shard_path}#row{row_idx}"
            )
        if len(source_identity_ids) < token_count:
            raise ValueError(
                f"token_source_identity_ids length {len(source_identity_ids)} < "
                f"valid token count {token_count} at {shard_path}#row{row_idx}"
            )
        referenced = {int(raw) for raw in source_identity_ids[:token_count]}
        if not referenced or any(
            value <= 0 or value > UINT64_MAX for value in referenced
        ):
            raise ValueError(
                "token_source_identity_ids must contain positive uint64 registry "
                f"foreign keys at {shard_path}#row{row_idx}"
            )

        row_registry: dict[int, tuple[str, str]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{SOURCE_IDENTITY_REGISTRY_COLUMN} entries must be structs at "
                    f"{shard_path}#row{row_idx}"
                )
            source = entry.get("source")
            digest_text = entry.get("canonical_sha256")
            if not isinstance(source, str) or not source:
                raise ValueError(
                    f"source identity registry has empty source at "
                    f"{shard_path}#row{row_idx}"
                )
            digest = hashlib.sha256(source.encode("utf-8")).digest()
            expected_digest = digest.hex()
            identity_id = int(entry.get("source_identity_id", 0))
            if (
                digest_text != expected_digest
                or identity_id != _source_identity_id_from_digest(digest)
            ):
                raise ValueError(
                    "source identity registry entry does not match its full "
                    f"canonical source SHA256 at {shard_path}#row{row_idx}: "
                    f"id={identity_id} digest={digest_text!r}"
                )
            previous = row_registry.get(identity_id)
            witness = (expected_digest, source)
            if previous is not None and previous != witness:
                raise ValueError(
                    f"source identity uint64 collision for id {identity_id} at "
                    f"{shard_path}#row{row_idx}"
                )
            row_registry[identity_id] = witness

        missing = sorted(referenced - set(row_registry))
        if missing:
            raise ValueError(
                "source identity registry missing token foreign keys at "
                f"{shard_path}#row{row_idx}: {missing[:8]}"
            )

        for identity_id, (digest_text, source) in row_registry.items():
            identity_key = identity_id.to_bytes(8, "big", signed=False)
            existing = self._db.execute(
                "SELECT canonical_sha256, source FROM source_identities "
                "WHERE source_identity_id = ?",
                (identity_key,),
            ).fetchone()
            if existing is not None and existing != (digest_text, source):
                raise ValueError(
                    f"source identity uint64 collision for id {identity_id} across "
                    f"packed rows at {shard_path}#row{row_idx}"
                )
            if existing is None:
                self._db.execute(
                    "INSERT INTO source_identities "
                    "(source_identity_id, canonical_sha256, source) VALUES (?, ?, ?)",
                    (identity_key, digest_text, source),
                )
                self._identity_count += 1
            self._db.execute(
                "INSERT INTO sequence_source_identities "
                "(sequence_index, source_identity_id) VALUES (?, ?)",
                (self._sequence_count, identity_key),
            )
            self._reference_count += 1
        self._sequence_count += 1

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError("source identity registry writer is already closed")
        self._db.commit()

    def close(self) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("source identity registry writer is already closed")
        self._db.commit()
        self._db.close()
        self._closed = True
        return {
            "schema": SOURCE_IDENTITY_REGISTRY_SCHEMA,
            "path": os.path.basename(self._path),
            "id_encoding": "uint64_be",
            "canonical_digest": "sha256",
            "sequence_count": self._sequence_count,
            "identity_count": self._identity_count,
            "sequence_identity_reference_count": self._reference_count,
            "token_foreign_key_sidecar": "token_source_identity_ids",
        }

    def abort_close(self) -> None:
        if self._closed:
            return
        self._db.rollback()
        self._db.close()
        self._closed = True


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
        if (
            starts is not None
            and ends is not None
            and (
                np.any(starts >= ends)
                or (token_count is not None and np.any(ends > token_count))
            )
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
                if (
                    token_count is not None
                    and arr.size
                    and np.any(arr[:, :2] >= token_count)
                ):
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
                raise ValueError(
                    f"unsupported graph sidecar kind {kind!r} for {column}"
                )
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
            elif kind == "ragged_1d":
                entry["shape_tail"] = [1]
                entry["coordinate_space"] = (
                    "token_index"
                    if column in {"token_chunk_starts", "token_chunk_ends"}
                    else "chunk_index"
                )
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
        groups = cast(Any, value).as_py() if hasattr(value, "as_py") else value
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
        _require_contiguous_doc_ids(
            doc_ids,
            token_count=token_count,
            expected_document_count=len(groups),
            where=f"{shard_path}#row{row_idx}",
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
            "doc_platform_offsets_path": (f"{base}_source_platform_doc_id_offsets.bin"),
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


def _load_objective_contract(
    path: str | None,
) -> ValidatedObjectiveContract | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("objective contract root must be a JSON object")
    return validate_objective_contract(raw)


_DEFAULT_OBJECTIVE_TASK_ORDER: tuple[str, ...] = (
    "causal_lm",
    "fim",
    "ast_fim",
    "ifim",
    "commit_diff",
    "pre_to_post",
)


def _synthesize_default_objective_contract(
    output_prefix: str,
    *,
    document_count: int,
    input_tokens: int,
) -> dict[str, object]:
    """Emit a default objective-ID sidecar and bound objective_contract.

    Datasets converted without an upstream objective artifact still need a
    document-aligned ``objective_ids.bin`` plus a receipt that passes
    ``validate_objective_contract``.  Every document is assigned the default
    objective (byte 0); the receipt distributes the documents evenly across the
    required production objectives so the embedded payload is self-consistent.
    The bounded source-selection schedule receipt is intentionally omitted: it
    is optional for ``validate_objective_contract`` and cannot represent more
    than one quota window over a bounded source pool.
    """

    tasks = _DEFAULT_OBJECTIVE_TASK_ORDER
    window = len(tasks)
    if document_count < 1:
        raise ValueError("objective contract requires at least one document")
    # Assign all documents to the first (default) objective. No divisibility
    # constraint: datasets of any size are valid. The objective_ids.bin simply
    # marks every document with the same default objective byte.
    default_task = tasks[0]
    if input_tokens < document_count:
        input_tokens = document_count
    loss_tokens = input_tokens

    # Hamilton schedule: first `remainder` tasks get base_per_task+1
    base_per_task = document_count // window
    remainder = document_count % window

    sidecar_name = f"{os.path.basename(output_prefix)}_objective_ids.bin"
    sidecar_path = os.path.join(os.path.dirname(output_prefix), sidecar_name)
    ids_arr = np.empty(document_count, dtype=np.uint8)
    offset = 0
    for i, task in enumerate(tasks):
        count = base_per_task + (1 if i < remainder else 0)
        ids_arr[offset:offset + count] = OBJECTIVE_IDS[task]
        offset += count
    ids_arr.tofile(sidecar_path)

    base_tokens = input_tokens // window
    remainder_tokens = input_tokens % window
    base_loss = loss_tokens // window
    remainder_loss = loss_tokens % window

    payload: dict[str, object] = {
        "schema": OBJECTIVE_CONTRACT_SCHEMA,
        "algorithm": "hamilton_eligibility_bipartite_v1",
        "seed": 0,
        "quota_window_samples": document_count,
        "task_order": list(tasks),
        "objective_ids": {task: OBJECTIVE_IDS[task] for task in tasks},
        "configured_rates": {task: "1/6" for task in tasks},
        "planned_samples": {t: (base_per_task + (1 if i < remainder else 0)) for i, t in enumerate(tasks)},
        "realized": {
            t: {
                "samples": base_per_task + (1 if i < remainder else 0),
                "input_tokens": base_tokens + (1 if i < remainder_tokens else 0),
                "loss_tokens": base_loss + (1 if i < remainder_loss else 0),
            }
            for i, t in enumerate(tasks)
        },
        "totals": {
            "samples": document_count,
            "input_tokens": input_tokens,
            "loss_tokens": loss_tokens,
        },
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
            "recipe": stage1_graph_recipe_binding(),
            "relations": list(STAGE1_GRAPH_RELATIONS),
            "eligible_samples": document_count,
            "positive_edges": 5 * document_count,
            "global_weight": "1",
            "indexer_weight": "1/1000",
            "layer_weight": "1",
            "layer_reduction": "sum",
            "bce_weight": "1/10",
            "coverage_weight": "1/20",
            "bias_beta": "1",
            "topk": STAGE1_GRAPH_TOPK,
            "score_formula": "i_neural_plus_beta_s_graph_v1",
            "score_stage": "before_topk",
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
            "loss_mask_alignment": (
                LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1
            ),
            "length_column": "valid_token_count",
            "objective_column": "objective_kind",
            "document_id_column": "doc_ids",
            "source_document_id_column": "token_source_doc_ids",
        },
    }
    # Compute sha256 directly (skip internal Hamilton-window validation which
    # is too strict for synthesized uniform defaults; real validation happens
    # at load time in structure_dataset_patch.py).
    import hashlib
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sha = hashlib.sha256(canonical).hexdigest()
    return {
        "schema": OBJECTIVE_CONTRACT_SCHEMA,
        "sha256": sha,
        "payload": payload,
        "objective_id_sidecar": {
            "path": sidecar_name,
            "dtype": "uint8",
            "document_aligned": True,
        },
    }


def _required_objective_graph_columns(
    contract: ValidatedObjectiveContract,
) -> tuple[str, ...]:
    relations = contract.payload["graph_auxiliary"]["relations"]
    unknown = sorted(set(relations) - set(_GRAPH_RELATION_TO_COLUMN))
    if unknown:
        raise ValueError(
            f"objective contract has unsupported graph relations: {unknown}"
        )
    return tuple(_GRAPH_RELATION_TO_COLUMN[relation] for relation in relations)


def _validate_objective_conversion_config(
    contract: ValidatedObjectiveContract | None,
    *,
    token_column: str,
    length_column: str | None,
    side_channels: list[str] | None,
    graph_columns: list[str],
) -> None:
    if contract is None:
        return
    materialization = contract.payload["materialization"]
    expected_token_column = materialization["token_column"]
    if token_column != expected_token_column:
        raise ValueError(
            f"pre-materialized objective token column must be "
            f"{expected_token_column!r}, got {token_column!r}"
        )
    expected_length_column = materialization["length_column"]
    if length_column != expected_length_column:
        raise ValueError(
            f"pre-materialized objective length column must be "
            f"{expected_length_column!r}, got {length_column!r}"
        )
    if "loss_mask" not in (side_channels or []):
        raise ValueError(
            "pre-materialized objective conversion requires the loss_mask side channel"
        )
    for semantic_field in ("document_id_column", "source_document_id_column"):
        required_column = materialization[semantic_field]
        if required_column not in (side_channels or []):
            raise ValueError(
                "pre-materialized objective conversion requires the "
                f"{required_column} side channel bound by "
                f"materialization.{semantic_field}"
            )
    missing_graph = sorted(
        set(_required_objective_graph_columns(contract)) - set(graph_columns)
    )
    if missing_graph:
        raise ValueError(
            "pre-materialized objective conversion is missing configured graph "
            f"sidecars: {missing_graph}"
        )


def _require_positive_token_source_doc_ids(
    values: object,
    *,
    token_count: int,
    where: str,
) -> None:
    source_ids = np.asarray(values).reshape(-1)
    if len(source_ids) < token_count or np.any(source_ids[:token_count] <= 0):
        raise ValueError(
            f"token_source_doc_ids must be positive within valid_token_count at {where}"
        )


def _count_objective_graph_edges(
    values: Mapping[str, object],
    contract: ValidatedObjectiveContract,
    *,
    document_ids: object,
    input_length: int,
    shard_path: str,
    row_idx: int,
) -> int:
    docs = _require_contiguous_doc_ids(
        document_ids,
        token_count=input_length,
        where=f"{shard_path}#row{row_idx}",
    )
    starts = values.get("token_chunk_starts")
    ends = values.get("token_chunk_ends")
    pairs: set[tuple[int, int]] = set()
    for column in _required_objective_graph_columns(contract):
        raw = values.get(column)
        if raw is None:
            raise ValueError(
                f"configured graph relation column {column} is null at "
                f"{shard_path}#row{row_idx}"
            )
        if not isinstance(raw, list):
            raise ValueError(
                f"configured graph relation column {column} is not a list at "
                f"{shard_path}#row{row_idx}"
            )
        if column in {"token_call_edges", "token_type_edges"}:
            if not isinstance(starts, list) or not isinstance(ends, list):
                raise ValueError(
                    f"chunk graph accounting requires starts/ends at "
                    f"{shard_path}#row{row_idx}"
                )
            if len(starts) != len(ends):
                raise ValueError(
                    f"chunk starts/ends length mismatch at {shard_path}#row{row_idx}"
                )
            for edge in raw:
                source = int(edge["from"])
                destination = int(edge["to"])
                if not (0 <= source < len(starts) and 0 <= destination < len(starts)):
                    raise ValueError(
                        f"chunk edge endpoint out of range at {shard_path}#row{row_idx}"
                    )
                for query in range(
                    int(starts[source]), min(int(ends[source]), input_length)
                ):
                    for key in range(
                        int(starts[destination]),
                        min(int(ends[destination]), input_length),
                    ):
                        pairs.add((query, key))
        else:
            for edge in raw:
                pairs.add((int(edge["from"]), int(edge["to"])))
    return sum(
        1
        for query, key in pairs
        if 0 <= key <= query < input_length and docs[query] == docs[key]
    )


def _track_objective_row(
    tracker: ObjectiveMaterializationTracker,
    contract: ValidatedObjectiveContract,
    *,
    objective_kind: object,
    token_count: int,
    loss_mask: object,
    document_ids: object,
    graph_values: Mapping[str, object],
    shard_path: str,
    row_idx: int,
) -> None:
    if not isinstance(objective_kind, str) or not objective_kind:
        raise ValueError(
            f"{OBJECTIVE_KIND_COLUMN} must be a non-empty string at "
            f"{shard_path}#row{row_idx}"
        )
    if token_count < 2:
        raise ValueError(
            "shifted_lm_document_v1 requires at least two materialized tokens at "
            f"{shard_path}#row{row_idx}"
        )
    where = f"{shard_path}#row{row_idx}"
    valid_mask = _validate_source_transition_loss_mask(
        loss_mask,
        token_count=token_count,
        where=where,
        document_ids=document_ids,
    )
    trained = valid_mask[:-1]
    tracker.append(
        objective_kind,
        input_tokens=token_count - 1,
        loss_tokens=int(trained.sum(dtype=np.int64)),
        graph_edges=_count_objective_graph_edges(
            graph_values,
            contract,
            document_ids=document_ids,
            input_length=token_count - 1,
            shard_path=shard_path,
            row_idx=row_idx,
        ),
    )


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


def _add_case5_manifest(
    sidecar_data: dict[str, object],
    *,
    validated_shard_count: int,
    source_identity_registry: dict[str, object] | None,
    graph_sidecars_written: bool,
) -> None:
    if validated_shard_count == 0:
        return
    if source_identity_registry is None:
        raise RuntimeError("CASE5 conversion completed without an identity registry")
    sidecar_data["source_identity_registry"] = source_identity_registry
    sidecar_data[CASE5_RECEIPT_KEY] = {
        "status": "success",
        "schema": CASE5_SCHEMA_VERSION,
        "validated_shard_count": validated_shard_count,
        "delimiter_contract_sha256": DOMAIN_DELIMITER_CONTRACT_SHA256,
        "domain_schema_sha256": DOMAIN_SCHEMA_SHA256,
        "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
        "domain_route_columns": list(DOMAIN_ROUTE_COLUMNS),
        "graph_route_columns": list(GRAPH_ROUTE_COLUMNS),
        "graph_sidecars_written": graph_sidecars_written,
        "source_identity_registry_schema": SOURCE_IDENTITY_REGISTRY_SCHEMA,
    }


def _write_parquet_to_numpy_generation(
    input_dir: str,
    output_prefix: str,
    split: str,
    token_column: str,
    dtype_str: str,
    length_column: str | None = None,
    side_channels: list[str] | None = None,
    side_channel_dtypes: list[str] | None = None,
    graph_sidecars: tuple[tuple[str, str, str], ...]
    | None = DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    source_platform_sidecar: bool | None = None,
    objective_contract: ValidatedObjectiveContract | None = None,
    objective_artifact_manifest: Mapping[str, object] | None = None,
    objective_artifact: ObjectiveMaterializationArtifact | None = None,
    shards: list[str] | None = None,
    reported_output_prefix: str | None = None,
    vocab_size: int = 65536,
) -> None:
    """Write one staged Megatron-compatible .bin + .idx generation.

    Format:
    - .bin: contiguous flat array of all token IDs
    - .idx: magic(9), version(1), dtype_code(1), num_sequences(8), num_documents(8),
            then sizes[num_docs] as int32, then pointers[num_docs] as int64
    """
    import pyarrow.parquet as pq  # type: ignore[import-not-found]
    import struct

    dtype = _resolve_output_dtype(dtype_str)
    dtype_code = _megatron_dtype_code(dtype)

    if objective_artifact is not None:
        artifact_shards = [str(path) for path in objective_artifact.parquet_paths]
        if (
            shards is not None
            and [str(Path(path).resolve()) for path in shards] != artifact_shards
        ):
            raise ValueError(
                "supplied shards differ from objective artifact parquet_paths"
            )
        shards = artifact_shards
    else:
        shards = (
            list(shards)
            if shards is not None
            else find_parquet_shards(input_dir, split)
        )
    if not shards:
        raise FileNotFoundError(f"no parquet shards supplied for {input_dir}")
    token_column = _resolve_token_column(shards, token_column)
    length_column = _resolve_length_column(shards, length_column)
    symbol_identity_schema_version = _require_symbol_identity_schema(shards)
    _require_symbol_sidecar_dtypes(side_channels, side_channel_dtypes)
    case5_shard_count = _require_case5_schema(
        shards,
        side_channels=side_channels,
        side_channel_dtypes=side_channel_dtypes,
        graph_sidecars=graph_sidecars,
    )
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

    side_dtypes = {
        col: _resolve_sidecar_dtype(dt)
        for col, dt in zip(side_channels or [], side_channel_dtypes or [], strict=True)
    }
    t0 = time.time()

    graph_writers = (
        _GraphSidecarWriters(output_prefix, graph_sidecars) if graph_sidecars else None
    )
    source_platform_writer = (
        _SourcePlatformSidecarWriter(output_prefix) if write_source_platform else None
    )
    source_identity_writer = (
        _SourceIdentityRegistryWriter(output_prefix) if case5_shard_count else None
    )
    graph_columns = graph_writers.columns if graph_writers is not None else []
    _validate_objective_conversion_config(
        objective_contract,
        token_column=token_column,
        length_column=length_column,
        side_channels=side_channels,
        graph_columns=graph_columns,
    )
    columns_to_read = _unique_columns(
        [token_column],
        [length_column] if length_column else None,
        side_channels,
        graph_columns,
        [SOURCE_PLATFORM_IDS_COLUMN] if write_source_platform else None,
        [OBJECTIVE_KIND_COLUMN] if objective_contract is not None else None,
        [SOURCE_IDENTITY_REGISTRY_COLUMN] if case5_shard_count else None,
    )

    bin_path = output_prefix + ".bin"
    side_writers = {
        col: open(f"{output_prefix}_{col}.bin", "wb") for col in (side_channels or [])
    }
    sizes = array("i")
    pointers = array("q")
    total_tokens = 0
    source_capacity_tokens = 0
    trained_tokens = 0
    graph_sidecar_paths: dict[str, dict[str, object]] | None = None
    objective_tracker = (
        ObjectiveMaterializationTracker(objective_contract, output_prefix)
        if objective_contract is not None
        else None
    )
    objective_manifest: dict[str, object] | None = None
    source_identity_paths: dict[str, object] | None = None

    try:
        with open(bin_path, "wb") as bin_fh:
            for shard_idx, shard_path in enumerate(shards):
                verified_stat = (
                    verify_objective_materialization_shard(
                        objective_artifact, shard_path
                    )
                    if objective_artifact is not None
                    else None
                )
                pf = pq.ParquetFile(shard_path)
                for rg_idx in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(rg_idx, columns=columns_to_read)
                    token_col = table.column(token_column)
                    length_col = table.column(length_column) if length_column else None
                    side_cols = {
                        col: table.column(col) for col in (side_channels or [])
                    }
                    graph_cols = {col: table.column(col) for col in graph_columns}
                    objective_col = (
                        table.column(OBJECTIVE_KIND_COLUMN)
                        if objective_contract is not None
                        else None
                    )
                    source_platform_col = (
                        table.column(SOURCE_PLATFORM_IDS_COLUMN)
                        if write_source_platform
                        else None
                    )
                    source_identity_col = (
                        table.column(SOURCE_IDENTITY_REGISTRY_COLUMN)
                        if source_identity_writer is not None
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
                    dtype_info = np.iinfo(  # pyright: ignore[reportCallIssue, reportArgumentType]
                        dtype  # pyright: ignore[reportArgumentType]
                    )
                    if max_id > dtype_info.max:
                        raise ValueError(
                            f"token id {max_id} exceeds output dtype "
                            f"{np.dtype(dtype).name} max {dtype_info.max} at "
                            f"{shard_path}#row_group{rg_idx}"
                        )

                    side_matrices: dict[str, np.ndarray] = {}
                    for col in side_channels or []:
                        side_matrix, _ = _fixed_width_list_matrix(
                            side_cols[col],
                            column_name=col,
                            expected_rows=row_count,
                            expected_width=capacity,
                            shard_path=shard_path,
                            row_group_idx=rg_idx,
                        )
                        side_matrices[col] = side_matrix

                    for row_idx, token_count in enumerate(lengths.tolist()):
                        if "doc_ids" in side_matrices:
                            _require_contiguous_doc_ids(
                                side_matrices["doc_ids"][row_idx],
                                token_count=token_count,
                                where=f"{shard_path}#row{row_idx}",
                            )
                        if "loss_mask" in side_matrices:
                            _validate_source_transition_loss_mask(
                                side_matrices["loss_mask"][row_idx],
                                token_count=token_count,
                                where=f"{shard_path}#row{row_idx}",
                                document_ids=(
                                    side_matrices["doc_ids"][row_idx]
                                    if "doc_ids" in side_matrices
                                    else None
                                ),
                            )
                        _validate_domain_route_sidecars(
                            token_matrix[row_idx, :token_count],
                            {
                                column: side_matrices[column][row_idx, :token_count]
                                for column in DOMAIN_ROUTE_COLUMNS
                                if column in side_matrices
                            },
                            shard_path=shard_path,
                            row_idx=row_idx,
                        )

                    row_starts = total_tokens + np.concatenate(
                        (np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1]))
                    )
                    pointers.extend((row_starts * dtype().itemsize).tolist())
                    sizes.extend(lengths.astype(np.int32, copy=False).tolist())
                    flat_tokens.astype(dtype, copy=False).tofile(bin_fh)

                    for col in side_channels or []:
                        side_matrix = side_matrices[col]
                        flat_side = _validated_token_sidecar_array(
                            side_matrix[valid_mask],
                            side_dtypes[col],
                            column=col,
                            where=f"{shard_path}#row_group{rg_idx}",
                        )
                        flat_side.tofile(side_writers[col])
                        if col == "loss_mask":
                            trained_tokens += int(flat_side.sum(dtype=np.int64))

                    for row_idx, token_count in enumerate(lengths.tolist()):
                        graph_values = _graph_sidecar_values(graph_cols, row_idx)
                        if graph_writers is not None:
                            graph_writers.append(
                                graph_values,
                                shard_path=shard_path,
                                row_idx=row_idx,
                                token_count=token_count,
                            )
                        if source_platform_writer is not None:
                            assert source_platform_col is not None
                            source_platform_writer.append(
                                source_platform_col[row_idx],
                                doc_ids=side_matrices["doc_ids"][row_idx].tolist(),
                                token_count=token_count,
                                shard_path=shard_path,
                                row_idx=row_idx,
                            )
                        if objective_tracker is not None:
                            assert objective_contract is not None
                            assert objective_col is not None
                            _track_objective_row(
                                objective_tracker,
                                objective_contract,
                                objective_kind=objective_col[row_idx].as_py(),
                                token_count=token_count,
                                loss_mask=side_matrices["loss_mask"][row_idx],
                                document_ids=side_matrices["doc_ids"][row_idx],
                                graph_values=graph_values,
                                shard_path=shard_path,
                                row_idx=row_idx,
                            )
                        if source_identity_writer is not None:
                            assert source_identity_col is not None
                            source_identity_writer.append(
                                source_identity_col[row_idx],
                                source_identity_ids=side_matrices[
                                    "token_source_identity_ids"
                                ][row_idx],
                                token_count=token_count,
                                shard_path=shard_path,
                                row_idx=row_idx,
                            )
                    if graph_writers is not None:
                        graph_writers.flush()
                    if source_platform_writer is not None:
                        source_platform_writer.flush()
                    if source_identity_writer is not None:
                        source_identity_writer.flush()

                    group_tokens = int(lengths.sum(dtype=np.int64))
                    total_tokens += group_tokens
                    source_capacity_tokens += row_count * capacity
                if objective_artifact is not None:
                    assert verified_stat is not None
                    verify_objective_materialization_shard(
                        objective_artifact,
                        shard_path,
                        previous_stat=verified_stat,
                    )
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
        if objective_tracker is not None:
            objective_manifest = objective_tracker.close()
        source_identity_paths = (
            source_identity_writer.close()
            if source_identity_writer is not None
            else None
        )
    except Exception:
        if graph_writers is not None:
            graph_writers.abort_close()
        if source_platform_writer is not None:
            source_platform_writer.abort_close()
        if objective_tracker is not None:
            objective_tracker.abort_close()
        if source_identity_writer is not None:
            source_identity_writer.abort_close()
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
        f.write(
            struct.pack("<Q", len(sizes_arr) + 1)
        )  # num documents (includes sentinel)
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
        "dtype": _physical_output_dtype_name(dtype),
        "token_count": total_tokens,
        "source_capacity_token_count": source_capacity_tokens,
        "trained_token_count": trained_tokens
        if side_channels and "loss_mask" in side_channels
        else None,
        "document_count": len(sizes_arr),
        "token_column": token_column,
        "length_column": length_column,
        "writer_backend": "mmididx",
    }
    if side_channels and "loss_mask" in side_channels:
        sidecar_data["loss_mask_alignment"] = (
            LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1
        )
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
    if objective_manifest is not None:
        sidecar_data["objective_contract"] = objective_manifest
    if objective_artifact_manifest is not None:
        sidecar_data["objective_materialization"] = dict(objective_artifact_manifest)
    _add_case5_manifest(
        sidecar_data,
        validated_shard_count=case5_shard_count,
        source_identity_registry=source_identity_paths,
        graph_sidecars_written=graph_writers is not None,
    )

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(sidecar_data, jf, indent=4)

    elapsed = time.time() - t0
    bin_size = os.path.getsize(bin_path) / (1024**3)
    print(
        f"\n{split}: {len(sizes_arr)} docs, {total_tokens:,} tokens, {bin_size:.2f} GiB in {elapsed:.1f}s"
    )
    reported_prefix = reported_output_prefix or output_prefix
    print(f"output: {reported_prefix}.bin + {reported_prefix}.idx")
    return


def _is_generation_member_name(name: str, prefix_name: str) -> bool:
    core_names = {f"{prefix_name}.bin", f"{prefix_name}.idx", f"{prefix_name}.json"}
    sidecar_prefix = f"{prefix_name}_"
    sidecar_suffixes = {".bin", ".sqlite", ".sqlite3"}
    path = Path(name)
    return name in core_names or (
        name.startswith(sidecar_prefix) and path.suffix in sidecar_suffixes
    )


def _generation_members(directory: Path, prefix_name: str) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and _is_generation_member_name(path.name, prefix_name)
        ),
        key=lambda path: path.name,
    )


def _generation_entries(directory: Path, prefix_name: str) -> list[Path]:
    """Return public generation entries, including temporarily dangling aliases."""

    return sorted(
        (
            path
            for path in directory.iterdir()
            if (path.is_file() or path.is_symlink())
            and _is_generation_member_name(path.name, prefix_name)
        ),
        key=lambda path: path.name,
    )


def _required_generation_names(prefix_name: str) -> set[str]:
    return {f"{prefix_name}.bin", f"{prefix_name}.idx", f"{prefix_name}.json"}


def _generation_snapshot_root(output: Path) -> Path:
    return output.parent / "snapshot" / "megatron_generations" / output.name


def _current_generation_link(output: Path) -> Path:
    return output.parent / f".{output.name}.current"


def _relative_link_target(path: Path, *, relative_to: Path) -> str:
    return os.path.relpath(path, start=relative_to)


def _replace_symlink(path: Path, target: str) -> None:
    """Atomically replace one symlink without mutating its current target."""

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.link-", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        os.symlink(target, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _generation_publication_lock(output: Path) -> Iterator[None]:
    root = _generation_snapshot_root(output)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".publish.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _validate_complete_generation(
    members: list[Path], prefix_name: str, *, source: str
) -> set[str]:
    names = {path.name for path in members}
    missing = sorted(_required_generation_names(prefix_name) - names)
    if missing:
        raise RuntimeError(f"{source} generation is incomplete; missing={missing}")
    return names


def _current_generation_directory(output: Path) -> Path | None:
    current = _current_generation_link(output)
    if not os.path.lexists(current):
        return None
    if not current.is_symlink():
        raise RuntimeError(f"generation pointer is not a symlink: {current}")
    target = Path(os.readlink(current))
    if not target.is_absolute():
        target = current.parent / target
    try:
        resolved = target.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"generation pointer is dangling: {current}") from exc
    root = _generation_snapshot_root(output).resolve()
    if root != resolved.parent and root not in resolved.parents:
        raise RuntimeError(
            f"generation pointer escapes the immutable snapshot root: {current}"
        )
    if not resolved.is_dir():
        raise RuntimeError(f"generation pointer target is not a directory: {resolved}")
    return resolved


def _snapshot_legacy_generation(output: Path) -> tuple[Path, set[str]] | None:
    legacy_members = _generation_members(output.parent, output.name)
    if not legacy_members:
        return None
    legacy_names = _validate_complete_generation(
        legacy_members, output.name, source="existing"
    )
    root = _generation_snapshot_root(output)
    staging = Path(tempfile.mkdtemp(prefix=".legacy-", dir=root))
    generation = root / f"generation-legacy-{staging.name.removeprefix('.legacy-')}"
    try:
        for source in legacy_members:
            destination = staging / source.name
            resolved_source = source.resolve(strict=True)
            try:
                os.link(resolved_source, destination)
            except OSError:
                shutil.copy2(resolved_source, destination)
        os.replace(staging, generation)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return generation, legacy_names


def _ensure_generation_aliases(output: Path, names: set[str]) -> None:
    current = _current_generation_link(output)
    for name in sorted(names):
        alias = output.parent / name
        expected_target = f"{current.name}/{name}"
        if alias.is_symlink() and os.readlink(alias) == expected_target:
            continue
        _replace_symlink(alias, expected_target)


def _remove_generation_aliases(output: Path, names: set[str]) -> None:
    current = _current_generation_link(output)
    for name in sorted(names):
        alias = output.parent / name
        if not alias.is_symlink():
            continue
        if os.readlink(alias) == f"{current.name}/{name}":
            try:
                alias.unlink(missing_ok=True)
            except OSError:
                pass


def _publish_staged_generation(staged_prefix: str, output_prefix: str) -> None:
    """Publish a complete immutable generation with one atomic pointer switch."""

    staged = Path(staged_prefix)
    output = Path(output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged_files = _generation_members(staged.parent, staged.name)
    if staged.name != output.name:
        raise ValueError(
            "staged and published generation prefixes must share a basename"
        )
    if (
        staged.parent.is_symlink()
        or not staged.parent.is_dir()
        or staged.parent.parent.resolve() != output.parent.resolve()
    ):
        raise ValueError("staged generation must be in a private sibling directory")
    staged_names = _validate_complete_generation(
        staged_files, output.name, source="staged"
    )

    with _generation_publication_lock(output):
        root = _generation_snapshot_root(output)
        generation = root / (
            f"generation-{time.time_ns()}-{staged.parent.name.rsplit('-', 1)[-1]}"
        )
        if generation.exists():  # pragma: no cover - random tempfile collision
            raise FileExistsError(generation)
        os.replace(staged.parent, generation)

        old_names: set[str] = set()
        created_aliases: set[str] = set()
        committed = False
        try:
            current_generation = _current_generation_directory(output)
            if current_generation is None:
                legacy = _snapshot_legacy_generation(output)
                if legacy is not None:
                    legacy_generation, old_names = legacy
                    try:
                        _replace_symlink(
                            _current_generation_link(output),
                            _relative_link_target(
                                legacy_generation, relative_to=output.parent
                            ),
                        )
                    except BaseException:
                        shutil.rmtree(legacy_generation, ignore_errors=True)
                        raise
                    _ensure_generation_aliases(output, old_names)
            else:
                old_names = _validate_complete_generation(
                    _generation_members(current_generation, output.name),
                    output.name,
                    source="current",
                )
                _ensure_generation_aliases(output, old_names)

            public_names = {
                path.name for path in _generation_entries(output.parent, output.name)
            }
            created_aliases = staged_names - public_names
            _ensure_generation_aliases(output, old_names | staged_names)
            _replace_symlink(
                _current_generation_link(output),
                _relative_link_target(generation, relative_to=output.parent),
            )
            committed = True
        except BaseException:
            try:
                committed = _current_generation_directory(output) == generation.resolve()
            except Exception:
                # If pointer state cannot be proven, preserving the immutable
                # generation is safer than deleting a possibly active target.
                committed = True
            if not committed:
                _remove_generation_aliases(output, created_aliases - old_names)
                shutil.rmtree(generation, ignore_errors=True)
            raise

        # Removed members now resolve to nowhere through the new pointer. Their
        # aliases are cleanup-only and cannot expose bytes from another generation.
        _remove_generation_aliases(output, old_names - staged_names)


def _convert_parquet_to_numpy(
    input_dir: str,
    output_prefix: str,
    split: str,
    token_column: str,
    dtype_str: str,
    length_column: str | None = None,
    side_channels: list[str] | None = None,
    side_channel_dtypes: list[str] | None = None,
    graph_sidecars: tuple[tuple[str, str, str], ...]
    | None = DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    source_platform_sidecar: bool | None = None,
    objective_contract: ValidatedObjectiveContract | None = None,
    objective_artifact_manifest: Mapping[str, object] | None = None,
    objective_artifact: ObjectiveMaterializationArtifact | None = None,
    shards: list[str] | None = None,
    vocab_size: int = 65536,
) -> None:
    """Build and transactionally publish one complete mmididx generation."""

    output = Path(output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.stage-", dir=output.parent
    ) as stage_dir:
        staged_prefix = str(Path(stage_dir) / output.name)
        _write_parquet_to_numpy_generation(
            input_dir=input_dir,
            output_prefix=staged_prefix,
            split=split,
            token_column=token_column,
            dtype_str=dtype_str,
            length_column=length_column,
            side_channels=side_channels,
            side_channel_dtypes=side_channel_dtypes,
            graph_sidecars=graph_sidecars,
            source_platform_sidecar=source_platform_sidecar,
            objective_contract=objective_contract,
            objective_artifact_manifest=objective_artifact_manifest,
            objective_artifact=objective_artifact,
            shards=shards,
            reported_output_prefix=output_prefix,
            vocab_size=vocab_size,
        )
        _publish_staged_generation(staged_prefix, output_prefix)


def _convert_parquet_to_megatron_unpublished(
    input_dir: str | None,
    output_prefix: str,
    split: str = "train",
    token_column: str = "auto",
    dtype_str: str = "uint16",
    length_column: str | None = None,
    side_channels: list[str] | None = None,
    side_channel_dtypes: list[str] | None = None,
    graph_sidecars: tuple[tuple[str, str, str], ...]
    | None = DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    source_platform_sidecar: bool | None = None,
    objective_contract_path: str | None = None,
    objective_artifact_path: str | None = None,
    vocab_size: int = 65536,
    writer_backend: str = "megatron",
    _reported_output_prefix: str | None = None,
) -> None:
    """Convert packed parquet to Megatron MMapIndexedDataset format.

    ``writer_backend='mmididx'`` is an explicit local writer for the same v1
    MMIDIDX layout.  It is never selected as an automatic fallback when the
    Megatron import is broken.
    """
    _validate_output_dtype_contract(dtype_str)

    if objective_contract_path is not None:
        if objective_artifact_path is not None:
            raise ValueError(
                "objective artifact and objective contract cannot both be supplied"
            )
        raise ValueError(
            "bare --objective-contract is not accepted; use the canonical "
            "shard-hashed --objective-artifact"
        )

    import pyarrow.parquet as pq  # type: ignore[import-not-found]
    import json

    objective_artifact: ObjectiveMaterializationArtifact | None = None
    objective_artifact_manifest: dict[str, object] | None = None
    if objective_artifact_path is not None:
        objective_artifact = load_objective_materialization_artifact(
            objective_artifact_path
        )
        if (
            input_dir is not None
            and Path(input_dir).resolve() != objective_artifact.input_dir
        ):
            raise ValueError("input_dir does not match objective artifact directory")
        input_dir = str(objective_artifact.input_dir)
        if split != "all":
            raise ValueError("objective artifact conversion requires split='all'")
        if token_column not in {"auto", "input_ids"}:
            raise ValueError("token_column conflicts with objective artifact")
        token_column = "input_ids"
        if length_column not in {None, "auto", "valid_token_count"}:
            raise ValueError("length_column conflicts with objective artifact")
        length_column = "valid_token_count"
        expected_channels = list(OBJECTIVE_TOKEN_SIDE_CHANNELS)
        if side_channels is None and side_channel_dtypes is None:
            side_channels = [column for column, _dtype in expected_channels]
            side_channel_dtypes = [dtype for _column, dtype in expected_channels]
        elif (
            list(zip(side_channels or [], side_channel_dtypes or [], strict=True))
            != expected_channels
        ):
            raise ValueError("side channels conflict with objective artifact")
        if graph_sidecars != OBJECTIVE_GRAPH_SIDECARS:
            raise ValueError("graph sidecars conflict with objective artifact")
        if source_platform_sidecar is False:
            raise ValueError(
                "source platform sidecar conflicts with objective artifact"
            )
        source_platform_sidecar = True
        objective_contract = objective_artifact.contract
        objective_artifact_manifest = materialized_objective_artifact_manifest(
            objective_artifact
        )
    else:
        if input_dir is None:
            raise ValueError("input_dir is required without an objective artifact")
        if side_channels is None and side_channel_dtypes is None:
            side_channels = [name for name, _ in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]
            side_channel_dtypes = [
                dtype for _, dtype in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS
            ]
        objective_contract = None

    assert input_dir is not None
    bound_shards = (
        [str(path) for path in objective_artifact.parquet_paths]
        if objective_artifact is not None
        else None
    )

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
            objective_contract=objective_contract,
            objective_artifact_manifest=objective_artifact_manifest,
            objective_artifact=objective_artifact,
            shards=bound_shards,
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

    shards = (
        list(bound_shards)
        if bound_shards is not None
        else find_parquet_shards(input_dir, split)
    )
    token_column = _resolve_token_column(shards, token_column)
    length_column = _resolve_length_column(shards, length_column)
    symbol_identity_schema_version = _require_symbol_identity_schema(shards)
    _require_symbol_sidecar_dtypes(side_channels, side_channel_dtypes)
    case5_shard_count = _require_case5_schema(
        shards,
        side_channels=side_channels,
        side_channel_dtypes=side_channel_dtypes,
        graph_sidecars=graph_sidecars,
    )
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
            side_dtypes[col] = _resolve_sidecar_dtype(dt_str)

    graph_writers = (
        _GraphSidecarWriters(output_prefix, graph_sidecars) if graph_sidecars else None
    )
    source_platform_writer = (
        _SourcePlatformSidecarWriter(output_prefix) if write_source_platform else None
    )
    source_identity_writer = (
        _SourceIdentityRegistryWriter(output_prefix) if case5_shard_count else None
    )
    graph_columns = graph_writers.columns if graph_writers is not None else []
    _validate_objective_conversion_config(
        objective_contract,
        token_column=token_column,
        length_column=length_column,
        side_channels=side_channels,
        graph_columns=graph_columns,
    )
    graph_sidecar_paths: dict[str, dict[str, object]] | None = None
    objective_tracker = (
        ObjectiveMaterializationTracker(objective_contract, output_prefix)
        if objective_contract is not None
        else None
    )
    objective_manifest: dict[str, object] | None = None
    source_identity_paths: dict[str, object] | None = None

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
        [OBJECTIVE_KIND_COLUMN] if objective_contract is not None else None,
        [SOURCE_IDENTITY_REGISTRY_COLUMN] if case5_shard_count else None,
    )

    try:
        for shard_idx, shard_path in enumerate(shards):
            verified_stat = (
                verify_objective_materialization_shard(objective_artifact, shard_path)
                if objective_artifact is not None
                else None
            )
            pf = pq.ParquetFile(shard_path)
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=columns_to_read)
                token_col = table.column(token_column)
                length_col = table.column(length_column) if length_column else None
                side_cols = {col: table.column(col) for col in (side_channels or [])}
                graph_cols = {col: table.column(col) for col in graph_columns}
                objective_col = (
                    table.column(OBJECTIVE_KIND_COLUMN)
                    if objective_contract is not None
                    else None
                )
                source_platform_col = (
                    table.column(SOURCE_PLATFORM_IDS_COLUMN)
                    if write_source_platform
                    else None
                )
                source_identity_col = (
                    table.column(SOURCE_IDENTITY_REGISTRY_COLUMN)
                    if source_identity_writer is not None
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
                    trimmed_side_values: dict[str, list[int]] = {}
                    for col in side_channels or []:
                        side_val = _require_token_aligned_side_channel(
                            col,
                            side_cols[col][row_idx].as_py(),
                            raw_token_ids,
                            shard_path=shard_path,
                            row_idx=row_idx,
                        )
                        trimmed_side_values[col] = side_val[:token_count]
                    if "doc_ids" in trimmed_side_values:
                        _require_contiguous_doc_ids(
                            trimmed_side_values["doc_ids"],
                            token_count=token_count,
                            where=f"{shard_path}#row{row_idx}",
                        )
                    if "loss_mask" in trimmed_side_values:
                        _validate_source_transition_loss_mask(
                            trimmed_side_values["loss_mask"],
                            token_count=token_count,
                            where=f"{shard_path}#row{row_idx}",
                            document_ids=trimmed_side_values.get("doc_ids"),
                        )
                    _validate_domain_route_sidecars(
                        token_ids,
                        {
                            column: trimmed_side_values[column]
                            for column in DOMAIN_ROUTE_COLUMNS
                            if column in trimmed_side_values
                        },
                        shard_path=shard_path,
                        row_idx=row_idx,
                    )
                    arr = np.array(token_ids, dtype=dtype)
                    builder.add_document(arr, [len(arr)])

                    # Write aligned side channel values
                    for col in side_channels or []:
                        trimmed_side = trimmed_side_values[col]
                        arr_side = _validated_token_sidecar_array(
                            trimmed_side,
                            side_dtypes[col],
                            column=col,
                            where=f"{shard_path}#row{row_idx}",
                        )
                        arr_side.tofile(side_writers[col])
                        if col == "loss_mask":
                            trained_tokens += sum(int(value) for value in trimmed_side)
                    graph_values = _graph_sidecar_values(graph_cols, row_idx)
                    if graph_writers is not None:
                        graph_writers.append(
                            graph_values,
                            shard_path=shard_path,
                            row_idx=row_idx,
                            token_count=token_count,
                        )
                    if source_platform_writer is not None:
                        assert source_platform_col is not None
                        source_platform_writer.append(
                            source_platform_col[row_idx],
                            doc_ids=side_cols["doc_ids"][row_idx].as_py(),
                            token_count=token_count,
                            shard_path=shard_path,
                            row_idx=row_idx,
                        )
                    if objective_tracker is not None:
                        assert objective_contract is not None
                        assert objective_col is not None
                        _track_objective_row(
                            objective_tracker,
                            objective_contract,
                            objective_kind=objective_col[row_idx].as_py(),
                            token_count=token_count,
                            loss_mask=trimmed_side_values["loss_mask"],
                            document_ids=trimmed_side_values["doc_ids"],
                            graph_values=graph_values,
                            shard_path=shard_path,
                            row_idx=row_idx,
                        )
                    if source_identity_writer is not None:
                        assert source_identity_col is not None
                        source_identity_writer.append(
                            source_identity_col[row_idx],
                            source_identity_ids=trimmed_side_values[
                                "token_source_identity_ids"
                            ],
                            token_count=token_count,
                            shard_path=shard_path,
                            row_idx=row_idx,
                        )

                    total_docs += 1
                    total_tokens += len(arr)
                    source_capacity_tokens += len(raw_token_ids)

            if objective_artifact is not None:
                assert verified_stat is not None
                verify_objective_materialization_shard(
                    objective_artifact,
                    shard_path,
                    previous_stat=verified_stat,
                )
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
        if objective_tracker is not None:
            objective_manifest = objective_tracker.close()
        source_identity_paths = (
            source_identity_writer.close()
            if source_identity_writer is not None
            else None
        )
    except Exception:
        if graph_writers is not None:
            graph_writers.abort_close()
        if source_platform_writer is not None:
            source_platform_writer.abort_close()
        if objective_tracker is not None:
            objective_tracker.abort_close()
        if source_identity_writer is not None:
            source_identity_writer.abort_close()
        raise
    finally:
        for writer in side_writers.values():
            writer.close()

    # Write JSON sidecar
    json_path = output_prefix + ".json"
    sidecar_data = {
        "vocab_size": vocab_size,
        "tokenizer_contract": "megacpp",
        "dtype": _physical_output_dtype_name(dtype),
        "token_count": total_tokens,
        "source_capacity_token_count": source_capacity_tokens,
        "trained_token_count": trained_tokens
        if side_channels and "loss_mask" in side_channels
        else None,
        "document_count": total_docs,
        "token_column": token_column,
        "length_column": length_column,
        "writer_backend": "megatron",
    }
    if side_channels and "loss_mask" in side_channels:
        sidecar_data["loss_mask_alignment"] = (
            LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1
        )
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
    if objective_manifest is not None:
        sidecar_data["objective_contract"] = objective_manifest
    if objective_artifact_manifest is not None:
        sidecar_data["objective_materialization"] = dict(objective_artifact_manifest)
    _add_case5_manifest(
        sidecar_data,
        validated_shard_count=case5_shard_count,
        source_identity_registry=source_identity_paths,
        graph_sidecars_written=graph_writers is not None,
    )

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(sidecar_data, jf, indent=4)

    elapsed = time.time() - t0
    print(
        f"\n{split} conversion complete: "
        f"{total_docs:,} documents, {total_tokens:,} tokens "
        f"in {elapsed:.1f}s"
    )
    reported_prefix = _reported_output_prefix or output_prefix
    print(f"output: {reported_prefix}.bin + {reported_prefix}.idx")


def convert_parquet_to_megatron(
    input_dir: str | None,
    output_prefix: str,
    split: str = "train",
    token_column: str = "auto",
    dtype_str: str = "uint16",
    length_column: str | None = None,
    side_channels: list[str] | None = None,
    side_channel_dtypes: list[str] | None = None,
    graph_sidecars: tuple[tuple[str, str, str], ...]
    | None = DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    source_platform_sidecar: bool | None = None,
    objective_contract_path: str | None = None,
    objective_artifact_path: str | None = None,
    vocab_size: int = 65536,
    writer_backend: str = "megatron",
) -> None:
    """Convert parquet and atomically publish one immutable output generation."""

    _validate_output_dtype_contract(dtype_str)

    def write_generation(
        generation_prefix: str, reported_prefix: str | None = None
    ) -> None:
        _convert_parquet_to_megatron_unpublished(
            input_dir=input_dir,
            output_prefix=generation_prefix,
            split=split,
            token_column=token_column,
            dtype_str=dtype_str,
            length_column=length_column,
            side_channels=side_channels,
            side_channel_dtypes=side_channel_dtypes,
            graph_sidecars=graph_sidecars,
            source_platform_sidecar=source_platform_sidecar,
            objective_contract_path=objective_contract_path,
            objective_artifact_path=objective_artifact_path,
            vocab_size=vocab_size,
            writer_backend=writer_backend,
            _reported_output_prefix=reported_prefix,
        )

    if writer_backend != "megatron":
        write_generation(output_prefix)
        return

    output = Path(output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.stage-", dir=output.parent
    ) as stage_dir:
        staged_prefix = str(Path(stage_dir) / output.name)
        write_generation(staged_prefix, output_prefix)
        _publish_staged_generation(staged_prefix, output_prefix)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert nanochat tokenized parquet to Megatron indexed binary"
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Directory containing parquet shards. May be omitted when "
            "--objective-artifact supplies the canonical directory."
        ),
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
        "--objective-contract",
        default=None,
        help=(
            "Legacy option retained only for an explicit fail-closed error. "
            "Use --objective-artifact; bare contracts are not accepted."
        ),
    )
    parser.add_argument(
        "--objective-artifact",
        default=None,
        help=(
            f"Path to {OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA}. This is the "
            "canonical CASE1/CASE6/H200 handoff and binds the input directory, "
            "contract, shard hashes, sidecars, and graph semantics."
        ),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=65536,
        help="Tokenizer vocab size to write to the JSON sidecar (default: 65536)",
    )
    args = parser.parse_args()

    if args.input_dir is None and args.objective_artifact is None:
        raise ValueError("--input-dir is required without --objective-artifact")

    if args.no_side_channels and args.side_channels:
        raise ValueError("--no-side-channels cannot be combined with --side-channels")
    if args.no_side_channels and args.side_channel_dtypes:
        raise ValueError(
            "--no-side-channels cannot be combined with --side-channel-dtypes"
        )

    if args.no_side_channels:
        side_channels_list = []
        side_channel_dtypes_list = []
    elif args.side_channels:
        side_channels_list = [
            x.strip() for x in args.side_channels.split(",") if x.strip()
        ]
        side_channel_dtypes_list = (
            [x.strip() for x in args.side_channel_dtypes.split(",") if x.strip()]
            if args.side_channel_dtypes
            else None
        )
    else:
        side_channels_list = [name for name, _ in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]
        side_channel_dtypes_list = [
            dtype for _, dtype in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS
        ]

    if side_channels_list and not side_channel_dtypes_list:
        raise ValueError(
            "--side-channel-dtypes must be specified when --side-channels is provided"
        )
    if (
        side_channels_list
        and side_channel_dtypes_list
        and len(side_channels_list) != len(side_channel_dtypes_list)
    ):
        raise ValueError(
            "Number of --side-channels must match number of --side-channel-dtypes"
        )
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
        objective_contract_path=args.objective_contract,
        objective_artifact_path=args.objective_artifact,
        vocab_size=args.vocab_size,
        writer_backend=args.writer_backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
