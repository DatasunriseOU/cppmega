#!/usr/bin/env python3
"""Route packed source rows into primary native and auxiliary Python shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from cppmega.data import symbol_identity as symbol_identity_schema
from cppmega.data.commit_scope import is_native_workflow_shell_path
from cppmega.data.nanochat_pipeline import packed_rows_schema as packed
from cppmega.data.nanochat_pipeline import tokenized_enriched_schema as enriched
from cppmega.data.source_conveyor_composition import (
    PACKED_SOURCE_INVENTORY_SCHEMA,
    SourceComposition,
    source_composition_receipt_sha256,
)
from scripts.nanochat_data import pack_enriched_rows as packer
from scripts.nanochat_data.atomic_publish import atomic_output_file

SCHEMA = "cppmega_packed_source_route_v2"
PYTHON_BUILD_KIND = "python"
EXCLUDED_ROUTE = "excluded_non_primary"
ROUTES = ("primary", "aux_python", EXCLUDED_ROUTE)
_SHELL_BUILD_KINDS = frozenset(
    {"bash", "sh", "zsh", "tcsh", "ksh", "powershell", "cmd"}
)
_PRIMARY_BUILD_KINDS = frozenset(
    {
        "autoconf",
        "automake",
        "bazel",
        "build_diagnostic",
        "cmake",
        "compiler_diagnostic",
        "compile_commands",
        "conan",
        "configure",
        "dockerfile",
        "gn",
        "linker_diagnostic",
        "make",
        "meson",
        "msvc",
        "ninja",
        "sanitizer_output",
        "scons",
        "vcpkg",
        "xmake",
    }
)
_ALLOWED_LEGACY_MISSING_COLUMNS = frozenset({packed.SOURCE_COMMIT_HASHES_COLUMN})
_ALIGNED_TO_CHRONOLOGY = {
    packed.SOURCE_REPO_STABLE_IDS_COLUMN: enriched.REPO_STABLE_ID_COLUMN,
    packed.SOURCE_FILEPATH_STABLE_IDS_COLUMN: enriched.FILEPATH_STABLE_ID_COLUMN,
    packed.SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN: (
        enriched.FILE_LOCAL_COMMIT_INDEX_COLUMN
    ),
    packed.SOURCE_COMMIT_HASHES_COLUMN: enriched.COMMIT_HASH_COLUMN,
    packed.SOURCE_PR_NUMBERS_COLUMN: enriched.PR_NUMBER_COLUMN,
    packed.SOURCE_HAS_PR_DISCUSSIONS_COLUMN: enriched.HAS_PR_DISCUSSION_COLUMN,
    packed.SOURCE_PR_DISCUSSION_CHARS_COLUMN: (enriched.PR_DISCUSSION_CHARS_COLUMN),
    packed.SOURCE_PR_DISCUSSION_LINES_COLUMN: (enriched.PR_DISCUSSION_LINES_COLUMN),
    packed.SOURCE_DOC_TYPES_COLUMN: enriched.DOC_TYPE_COLUMN,
    packed.SOURCE_HEADER_FRAGMENT_KINDS_COLUMN: (enriched.HEADER_FRAGMENT_KIND_COLUMN),
    packed.SOURCE_BUILD_KINDS_COLUMN: enriched.BUILD_KIND_COLUMN,
}
_DOC_ALIGNED_COLUMNS = (
    packed.SOURCE_DOC_IDS_COLUMN,
    packer.SOURCE_DOC_TOKEN_LENGTHS_COLUMN,
    packed.SOURCE_PLATFORM_IDS_COLUMN,
    *_ALIGNED_TO_CHRONOLOGY,
    *packed.PACKED_ROWS_OBJECTIVE_SOURCE_TO_TOKEN_COLUMN,
)
_CHUNK_EDGE_COLUMNS = (
    enriched.TOKEN_CALL_EDGES_COLUMN,
    enriched.TOKEN_TYPE_EDGES_COLUMN,
)
_TOKEN_EDGE_COLUMNS = (
    enriched.TOKEN_DOMAIN_EDGES_COLUMN,
    enriched.TOKEN_BUILD_EDGES_COLUMN,
    enriched.TOKEN_SHELL_EDGES_COLUMN,
    enriched.TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    enriched.TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation() -> dict[str, str]:
    return {
        "router_sha256": _sha256_file(Path(__file__).resolve()),
        "packer_sha256": _sha256_file(Path(packer.__file__).resolve()),
        "packed_schema_sha256": _sha256_file(Path(packed.__file__).resolve()),
        "enriched_schema_sha256": _sha256_file(Path(enriched.__file__).resolve()),
        "symbol_identity_schema_sha256": _sha256_file(
            Path(symbol_identity_schema.__file__).resolve()
        ),
    }


def _write_json_atomic(path: Path, value: object) -> None:
    with atomic_output_file(path) as staged:
        staged.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _aligned_values(
    row: dict[str, Any],
    column: str,
    *,
    num_docs: int,
) -> list[Any]:
    values = row.get(column)
    if not isinstance(values, list) or len(values) != num_docs:
        raise ValueError(
            f"{column} must contain one value per source document: "
            f"expected {num_docs}, got {values!r}"
        )
    return values


def _upgrade_legacy_row(row: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(row)
    num_docs = int(upgraded.get(packed.NUM_DOCS_COLUMN, -1))
    if num_docs <= 0:
        raise ValueError(f"invalid num_docs={num_docs}")
    if packed.SOURCE_COMMIT_HASHES_COLUMN not in upgraded:
        upgraded[packed.SOURCE_COMMIT_HASHES_COLUMN] = [
            upgraded.get(enriched.COMMIT_HASH_COLUMN)
        ] * num_docs
    for column in _DOC_ALIGNED_COLUMNS:
        _aligned_values(upgraded, column, num_docs=num_docs)
    lengths = [int(value) for value in upgraded[packer.SOURCE_DOC_TOKEN_LENGTHS_COLUMN]]
    valid = int(upgraded.get(packed.VALID_TOKEN_COUNT_COLUMN, -1))
    if any(length <= 0 for length in lengths) or sum(lengths) != valid:
        raise ValueError(
            f"source_doc_token_lengths={lengths!r} do not sum to "
            f"valid_token_count={valid}"
        )
    capacity = len(upgraded.get(packed.INPUT_IDS_COLUMN) or [])
    if not 0 < valid <= capacity:
        raise ValueError(f"invalid valid_token_count={valid} for capacity={capacity}")
    for column in (
        packed.INPUT_IDS_COLUMN,
        packed.TARGET_IDS_COLUMN,
        packed.LOSS_MASK_COLUMN,
        packed.DOC_IDS_COLUMN,
        *packed.PACKED_ROWS_TOKEN_METADATA_COLUMNS,
    ):
        values = upgraded.get(column)
        if not isinstance(values, list) or len(values) != capacity:
            raise ValueError(
                f"{column} must have packed capacity {capacity}, got "
                f"{len(values) if isinstance(values, list) else values!r}"
            )
    trained = int(upgraded.get(packer.TRAINED_TOKEN_COUNT_COLUMN, -1))
    if trained != sum(int(value) > 0 for value in upgraded[packed.LOSS_MASK_COLUMN]):
        raise ValueError(
            f"trained_token_count={trained} does not match positive loss_mask sum"
        )
    return upgraded


def _validate_input_schema(schema: pa.Schema, *, where: Path) -> None:
    expected = packer.PACKED_ROW_OUTPUT_SCHEMA
    actual_names = set(schema.names)
    expected_names = set(expected.names)
    missing = expected_names - actual_names
    extra = actual_names - expected_names
    if not missing <= _ALLOWED_LEGACY_MISSING_COLUMNS or extra:
        raise ValueError(
            f"{where}: incompatible packed schema; missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    for name in actual_names:
        if schema.field(name).type != expected.field(name).type:
            raise ValueError(
                f"{where}: {name} type {schema.field(name).type} != "
                f"{expected.field(name).type}"
            )
    metadata = schema.metadata or {}
    for key, value in (expected.metadata or {}).items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{where}: schema metadata {key!r}={metadata.get(key)!r} "
                f"!= required {value!r}"
            )


def _chunk_metadata_for_doc(
    row: dict[str, Any],
    *,
    token_start: int,
    token_end: int,
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[int],
    dict[int, int],
]:
    starts = [int(value) for value in row[enriched.TOKEN_CHUNK_STARTS_COLUMN]]
    ends = [int(value) for value in row[enriched.TOKEN_CHUNK_ENDS_COLUMN]]
    kinds = [int(value) for value in row[enriched.TOKEN_CHUNK_KINDS_COLUMN]]
    levels = [int(value) for value in row[enriched.TOKEN_CHUNK_DEP_LEVELS_COLUMN]]
    if len({len(starts), len(ends), len(kinds), len(levels)}) != 1:
        raise ValueError("packed chunk metadata columns are not aligned")

    selected: list[int] = []
    for chunk_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        overlaps = start < token_end and end > token_start
        contained = token_start <= start < end <= token_end
        if overlaps and not contained:
            raise ValueError(
                f"chunk {chunk_index} span ({start}, {end}) crosses a "
                "source-document boundary"
            )
        if contained:
            selected.append(chunk_index)
    remap = {old: new for new, old in enumerate(selected)}
    return (
        [starts[index] - token_start for index in selected],
        [ends[index] - token_start for index in selected],
        [kinds[index] for index in selected],
        [levels[index] for index in selected],
        remap,
    )


def _chunk_edges_for_doc(
    row: dict[str, Any],
    *,
    column: str,
    remap: dict[int, int],
) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for edge in row[column]:
        source = int(edge["from"])
        target = int(edge["to"])
        source_here = source in remap
        target_here = target in remap
        if source_here != target_here:
            raise ValueError(
                f"{column} edge ({source}, {target}) crosses a "
                "source-document boundary"
            )
        if source_here:
            result.append({"from": remap[source], "to": remap[target]})
    return result


def _token_edges_for_doc(
    row: dict[str, Any],
    *,
    column: str,
    token_start: int,
    token_end: int,
) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for edge in row[column]:
        source = int(edge["from"])
        target = int(edge["to"])
        source_here = token_start <= source < token_end
        target_here = token_start <= target < token_end
        if source_here != target_here:
            raise ValueError(
                f"{column} edge ({source}, {target}) crosses a "
                "source-document boundary"
            )
        if source_here:
            result.append(
                {
                    "from": source - token_start,
                    "to": target - token_start,
                    "kind": int(edge["kind"]),
                }
            )
    return result


def _changed_chunks_for_doc(
    row: dict[str, Any],
    *,
    token_start: int,
    token_end: int,
    remap: dict[int, int],
) -> tuple[list[int], list[tuple[int, int]]]:
    chunk_ids = [int(value) for value in row[enriched.CHANGED_CHUNK_IDS_COLUMN]]
    spans = row[enriched.CHANGED_CHUNK_SPANS_COLUMN]
    if len(chunk_ids) != len(spans):
        raise ValueError("changed chunk IDs and spans are not aligned")
    selected_ids: list[int] = []
    selected_spans: list[tuple[int, int]] = []
    for chunk_id, span in zip(chunk_ids, spans, strict=True):
        start = int(span["start"])
        end = int(span["end"])
        if chunk_id in remap:
            if not token_start <= start < end <= token_end:
                raise ValueError(
                    f"changed chunk {chunk_id} span ({start}, {end}) crosses "
                    "a source-document boundary"
                )
            selected_ids.append(remap[chunk_id])
            selected_spans.append((start - token_start, end - token_start))
        elif start < token_end and end > token_start:
            raise ValueError(
                f"changed span ({start}, {end}) overlaps a document without "
                f"its chunk {chunk_id}"
            )
    return selected_ids, selected_spans


def _extract_document(
    row: dict[str, Any],
    *,
    doc_index: int,
    token_start: int,
    token_end: int,
) -> packer.NormalizedDoc:
    num_docs = int(row[packed.NUM_DOCS_COLUMN])
    token_meta = {
        column: [int(value) for value in row[column][token_start:token_end]]
        for column in packed.PACKED_ROWS_TOKEN_METADATA_COLUMNS
    }
    (
        chunk_starts,
        chunk_ends,
        chunk_kinds,
        chunk_dep_levels,
        chunk_remap,
    ) = _chunk_metadata_for_doc(
        row,
        token_start=token_start,
        token_end=token_end,
    )
    changed_ids, changed_spans = _changed_chunks_for_doc(
        row,
        token_start=token_start,
        token_end=token_end,
        remap=chunk_remap,
    )

    chronology = {
        column: row.get(column) for column in packer.PACKED_ROW_PROVENANCE_COLUMNS
    }
    for source_column, chronology_column in _ALIGNED_TO_CHRONOLOGY.items():
        chronology[chronology_column] = _aligned_values(
            row, source_column, num_docs=num_docs
        )[doc_index]

    referenced_source_ids = {
        int(value)
        for value in token_meta[enriched.TOKEN_SOURCE_IDENTITY_IDS_COLUMN]
        if int(value) > 0
    }
    source_registry = {
        int(entry["source_identity_id"]): dict(entry)
        for entry in row[enriched.SOURCE_IDENTITY_REGISTRY_COLUMN]
    }
    missing_source_ids = referenced_source_ids - set(source_registry)
    if not referenced_source_ids or missing_source_ids:
        raise ValueError(
            "source identity registry does not cover routed document: "
            f"referenced={sorted(referenced_source_ids)} "
            f"missing={sorted(missing_source_ids)}"
        )

    referenced_symbol_ids = {
        int(value)
        for column in (
            enriched.TOKEN_SYMBOL_IDS_COLUMN,
            enriched.TOKEN_CALL_TARGETS_COLUMN,
            enriched.TOKEN_TYPE_REFS_COLUMN,
        )
        for value in token_meta[column]
        if int(value) > 0
    }
    symbol_registry = {
        int(entry["symbol_id"]): dict(entry)
        for entry in row[symbol_identity_schema.SYMBOL_IDENTITIES_COLUMN]
    }
    missing_symbol_ids = referenced_symbol_ids - set(symbol_registry)
    if missing_symbol_ids:
        raise ValueError(
            f"symbol registry is missing routed IDs {sorted(missing_symbol_ids)}"
        )

    source_doc_index = int(
        _aligned_values(row, packed.SOURCE_DOC_IDS_COLUMN, num_docs=num_docs)[doc_index]
    )
    stable_doc_values = token_meta[enriched.TOKEN_SOURCE_DOC_IDS_COLUMN]
    stable_doc_id = next(
        (int(value) for value in stable_doc_values if int(value) > 0),
        source_doc_index + 1,
    )
    objective_token_ids = {
        token_column: [
            int(value)
            for value in _aligned_values(row, source_column, num_docs=num_docs)[
                doc_index
            ]
        ]
        for source_column, token_column in (
            packed.PACKED_ROWS_OBJECTIVE_SOURCE_TO_TOKEN_COLUMN.items()
        )
    }
    chunk_edges = {
        column: _chunk_edges_for_doc(row, column=column, remap=chunk_remap)
        for column in _CHUNK_EDGE_COLUMNS
    }
    token_edges = {
        column: _token_edges_for_doc(
            row,
            column=column,
            token_start=token_start,
            token_end=token_end,
        )
        for column in _TOKEN_EDGE_COLUMNS
    }
    return packer.NormalizedDoc(
        source_doc_index=source_doc_index,
        stable_doc_id=stable_doc_id,
        stable_source_id=next(
            int(value)
            for value in token_meta[enriched.TOKEN_SOURCE_IDENTITY_IDS_COLUMN]
            if int(value) > 0
        ),
        source_identity_registry=tuple(
            source_registry[identity]
            for identity in source_registry
            if identity in referenced_source_ids
        ),
        token_ids=[
            int(value) for value in row[packed.INPUT_IDS_COLUMN][token_start:token_end]
        ],
        token_meta=token_meta,
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_kinds=chunk_kinds,
        chunk_dep_levels=chunk_dep_levels,
        call_edges=chunk_edges[enriched.TOKEN_CALL_EDGES_COLUMN],
        type_edges=chunk_edges[enriched.TOKEN_TYPE_EDGES_COLUMN],
        platform_ids=[
            int(value)
            for value in _aligned_values(
                row, packed.SOURCE_PLATFORM_IDS_COLUMN, num_docs=num_docs
            )[doc_index]
        ],
        changed_chunk_ids=changed_ids,
        changed_chunk_spans=changed_spans,
        chronology=chronology,
        symbol_identities=[
            symbol_registry[identity]
            for identity in symbol_registry
            if identity in referenced_symbol_ids
        ],
        objective_token_ids=objective_token_ids,
        domain_edges=token_edges[enriched.TOKEN_DOMAIN_EDGES_COLUMN],
        build_edges=token_edges[enriched.TOKEN_BUILD_EDGES_COLUMN],
        shell_edges=token_edges[enriched.TOKEN_SHELL_EDGES_COLUMN],
        diagnostic_edges=token_edges[enriched.TOKEN_DIAGNOSTIC_EDGES_COLUMN],
        cross_domain_edges=token_edges[enriched.TOKEN_CROSS_DOMAIN_EDGES_COLUMN],
    )


def _chain_documents(docs: list[packer.NormalizedDoc]) -> list[packer.NormalizedDoc]:
    chained: list[packer.NormalizedDoc] = []
    previous: int | None = None
    for doc in docs:
        chained.append(
            replace(doc, doc_dep_edges=(() if previous is None else (previous,)))
        )
        previous = doc.source_doc_index
    return chained


def _assert_routed_content(
    source: dict[str, Any],
    routed: dict[str, Any],
    *,
    doc_indices: list[int],
) -> None:
    lengths = [int(value) for value in source[packer.SOURCE_DOC_TOKEN_LENGTHS_COLUMN]]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    expected_tokens = [
        int(token)
        for doc_index in doc_indices
        for token in source[packed.INPUT_IDS_COLUMN][
            offsets[doc_index] : offsets[doc_index + 1]
        ]
    ]
    valid = int(routed[packed.VALID_TOKEN_COUNT_COLUMN])
    if routed[packed.INPUT_IDS_COLUMN][:valid] != expected_tokens:
        raise ValueError("routed token payload is not lossless")
    for column in _DOC_ALIGNED_COLUMNS:
        expected = [source[column][index] for index in doc_indices]
        if routed[column] != expected:
            raise ValueError(f"routed {column} is not lossless")
    for column in packed.PACKED_ROWS_TOKEN_METADATA_COLUMNS:
        expected = [
            int(value)
            for doc_index in doc_indices
            for value in source[column][offsets[doc_index] : offsets[doc_index + 1]]
        ]
        if routed[column][:valid] != expected:
            raise ValueError(f"routed {column} is not lossless")


def _source_path_for_doc(
    row: dict[str, Any],
    *,
    token_start: int,
    token_end: int,
) -> str:
    registry = {
        int(entry["source_identity_id"]): entry["source"]
        for entry in row[enriched.SOURCE_IDENTITY_REGISTRY_COLUMN]
    }
    paths: set[str] = set()
    for raw_identity in row[enriched.TOKEN_SOURCE_IDENTITY_IDS_COLUMN][
        token_start:token_end
    ]:
        identity = int(raw_identity)
        if identity <= 0:
            continue
        source = registry.get(identity)
        if not isinstance(source, str):
            raise ValueError(f"missing source identity registry entry {identity}")
        try:
            provenance = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ValueError("source identity provenance is not canonical JSON") from exc
        if not isinstance(provenance, dict):
            continue
        path = provenance.get("filepath") or provenance.get("source_path")
        if isinstance(path, str) and path:
            paths.add(path)
    if len(paths) > 1:
        raise ValueError(f"shell document has ambiguous source paths: {sorted(paths)}")
    return next(iter(paths), "")


def _source_doc_route(
    row: dict[str, Any],
    *,
    build_kind: object,
    doc_type: object,
    token_start: int,
    token_end: int,
) -> str:
    if build_kind is None:
        return "primary" if doc_type in {"code", "code_header"} else EXCLUDED_ROUTE
    kind = str(build_kind).casefold()
    if kind == PYTHON_BUILD_KIND:
        return "aux_python"
    if kind == "sql" or kind.startswith("sql:") or kind in _PRIMARY_BUILD_KINDS:
        return "primary"
    if kind in _SHELL_BUILD_KINDS and is_native_workflow_shell_path(
        _source_path_for_doc(
            row,
            token_start=token_start,
            token_end=token_end,
        )
    ):
        return "primary"
    return EXCLUDED_ROUTE


def route_packed_row(
    raw_row: dict[str, Any],
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    bool,
]:
    """Return primary, Python, excluded, and mixed-row flag for one packed row."""

    row = _upgrade_legacy_row(raw_row)
    kinds = [
        value
        for value in _aligned_values(
            row,
            packed.SOURCE_BUILD_KINDS_COLUMN,
            num_docs=int(row[packed.NUM_DOCS_COLUMN]),
        )
    ]
    doc_types = _aligned_values(
        row,
        packed.SOURCE_DOC_TYPES_COLUMN,
        num_docs=int(row[packed.NUM_DOCS_COLUMN]),
    )
    lengths = [int(value) for value in row[packer.SOURCE_DOC_TOKEN_LENGTHS_COLUMN]]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    routes = [
        _source_doc_route(
            row,
            build_kind=kind,
            doc_type=doc_types[index],
            token_start=offsets[index],
            token_end=offsets[index + 1],
        )
        for index, kind in enumerate(kinds)
    ]
    if len(set(routes)) == 1:
        only_route = routes[0]
        return (
            row if only_route == "primary" else None,
            row if only_route == "aux_python" else None,
            row if only_route == EXCLUDED_ROUTE else None,
            False,
        )

    docs = [
        _extract_document(
            row,
            doc_index=index,
            token_start=offsets[index],
            token_end=offsets[index + 1],
        )
        for index in range(len(lengths))
    ]
    capacity = len(row[packed.INPUT_IDS_COLUMN])
    pad_token_id = int(row[packed.TARGET_IDS_COLUMN][-1])
    routed: dict[str, dict[str, Any] | None] = {}
    for route in ROUTES:
        indices = [index for index, value in enumerate(routes) if value == route]
        if not indices:
            routed[route] = None
            continue
        routed_row = packer._materialize_packed_row(
            _chain_documents([docs[index] for index in indices]),
            target_length=capacity,
            pad_token_id=pad_token_id,
            pack_id=int(row[packed.PACK_ID_COLUMN]),
        )
        _assert_routed_content(row, routed_row, doc_indices=indices)
        routed[route] = routed_row
    return (
        routed["primary"],
        routed["aux_python"],
        routed[EXCLUDED_ROUTE],
        True,
    )


def _empty_counts() -> dict[str, int]:
    return {
        "rows": 0,
        "valid_tokens": 0,
        "trained_tokens": 0,
        "documents": 0,
        "capacity_tokens": 0,
    }


def _add_row(counts: dict[str, int], row: dict[str, Any]) -> None:
    counts["rows"] += 1
    counts["valid_tokens"] += int(row[packed.VALID_TOKEN_COUNT_COLUMN])
    counts["trained_tokens"] += int(row[packer.TRAINED_TOKEN_COUNT_COLUMN])
    counts["documents"] += int(row[packed.NUM_DOCS_COLUMN])
    counts["capacity_tokens"] += len(row[packed.INPUT_IDS_COLUMN])


def _assert_conservation(
    source: dict[str, int],
    primary: dict[str, int],
    auxiliary: dict[str, int],
    excluded: dict[str, int],
    *,
    where: Path,
) -> None:
    for key in ("valid_tokens", "trained_tokens", "documents"):
        routed = primary[key] + auxiliary[key] + excluded[key]
        if source[key] != routed:
            raise ValueError(
                f"{where}: {key} is not conserved: source={source[key]} "
                f"primary={primary[key]} aux_python={auxiliary[key]} "
                f"{EXCLUDED_ROUTE}={excluded[key]}"
            )


def _validate_output(path: Path, *, expected_rows: int) -> None:
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(packer.PACKED_ROW_OUTPUT_SCHEMA) or dict(
        parquet.schema_arrow.metadata or {}
    ) != dict(packer.PACKED_ROW_OUTPUT_SCHEMA.metadata or {}):
        raise ValueError(f"{path}: output schema is not canonical")
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} rows, got "
            f"{parquet.metadata.num_rows}"
        )
    codecs = {
        parquet.metadata.row_group(group).column(column).compression
        for group in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.row_group(group).num_columns)
    }
    if codecs and codecs != {"ZSTD"}:
        raise ValueError(f"{path}: output codecs must be ZSTD, got {sorted(codecs)}")


def _artifact(
    path: Path, counts: dict[str, int], *, output_root: Path
) -> dict[str, Any]:
    _validate_output(path, expected_rows=counts["rows"])
    return {
        "path": str(path.relative_to(output_root)),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        **counts,
    }


def _verify_completed_file(
    marker: Path,
    *,
    input_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "complete":
        raise ValueError(f"{marker}: invalid route marker")
    if receipt.get("implementation") != _implementation():
        raise ValueError(f"{marker}: implementation changed after routing")
    if receipt["input"]["sha256"] != _sha256_file(input_path):
        raise ValueError(f"{marker}: input changed after routing")
    for route in ROUTES:
        artifact = receipt["routes"][route]
        path = output_root / artifact["path"]
        if _sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"{marker}: routed artifact changed: {path}")
        _validate_output(path, expected_rows=int(artifact["rows"]))
    return receipt


def route_file(
    input_path_str: str,
    *,
    input_root_str: str,
    output_root_str: str,
    compression_level: int,
    resume: bool,
) -> dict[str, Any]:
    input_path = Path(input_path_str)
    input_root = Path(input_root_str)
    output_root = Path(output_root_str)
    relative = input_path.relative_to(input_root)
    route_paths = {
        route: output_root / route / relative
        for route in ROUTES
    }
    marker = output_root / "state" / relative.with_suffix(".route.json")
    if marker.exists():
        if not resume:
            raise FileExistsError(marker)
        return _verify_completed_file(
            marker, input_path=input_path, output_root=output_root
        )

    parquet = pq.ParquetFile(input_path)
    _validate_input_schema(parquet.schema_arrow, where=input_path)
    bucket = int(relative.parent.name)
    batch_size = max(1, 131_072 // bucket)
    counts = {
        "source": _empty_counts(),
        **{route: _empty_counts() for route in ROUTES},
    }
    mixed_rows = 0
    with ExitStack() as stack:
        writers = {
            route: pq.ParquetWriter(
                stack.enter_context(atomic_output_file(route_paths[route])),
                packer.PACKED_ROW_OUTPUT_SCHEMA,
                compression="zstd",
                compression_level=compression_level,
                use_dictionary=True,
            )
            for route in ROUTES
        }
        try:
            for batch in parquet.iter_batches(batch_size=batch_size):
                routed_rows: dict[str, list[dict[str, Any]]] = {
                    route: [] for route in ROUTES
                }
                for raw_row in batch.to_pylist():
                    _add_row(counts["source"], raw_row)
                    primary, auxiliary, excluded, mixed = route_packed_row(raw_row)
                    mixed_rows += int(mixed)
                    for route, routed_row in zip(
                        ROUTES,
                        (primary, auxiliary, excluded),
                        strict=True,
                    ):
                        if routed_row is not None:
                            _add_row(counts[route], routed_row)
                            routed_rows[route].append(routed_row)
                for route, rows in routed_rows.items():
                    if rows:
                        writers[route].write_table(
                            pa.Table.from_pylist(
                                rows, schema=packer.PACKED_ROW_OUTPUT_SCHEMA
                            ),
                            row_group_size=batch_size,
                        )
        finally:
            for writer in writers.values():
                writer.close()

    _assert_conservation(
        counts["source"],
        counts["primary"],
        counts["aux_python"],
        counts[EXCLUDED_ROUTE],
        where=input_path,
    )
    receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "created_at": _utc_now(),
        "input": {
            "path": str(relative),
            "sha256": _sha256_file(input_path),
            "size": input_path.stat().st_size,
            **counts["source"],
        },
        "routes": {
            route: _artifact(
                route_paths[route], counts[route], output_root=output_root
            )
            for route in ROUTES
        },
        "mixed_rows_split": mixed_rows,
        "implementation": _implementation(),
        "unresolved_count": 0,
    }
    _write_json_atomic(marker, receipt)
    return receipt


def _discover(input_root: Path, buckets: tuple[int, ...]) -> list[Path]:
    files: list[Path] = []
    for bucket in buckets:
        bucket_root = input_root / str(bucket)
        if not bucket_root.is_dir():
            raise FileNotFoundError(bucket_root)
        files.extend(sorted(bucket_root.glob("*.parquet")))
    if not files:
        raise RuntimeError(f"no packed parquet files found under {input_root}")
    return sorted(files, key=lambda path: path.relative_to(input_root).as_posix())


def _parse_buckets(value: str) -> tuple[int, ...]:
    buckets = tuple(int(part) for part in value.split(",") if part.strip())
    if not buckets or any(bucket <= 0 for bucket in buckets):
        raise argparse.ArgumentTypeError(
            "buckets must be a non-empty comma-separated list"
        )
    if buckets != tuple(sorted(set(buckets))):
        raise argparse.ArgumentTypeError(
            "buckets must be unique and strictly increasing"
        )
    return buckets


def _complete_input_receipt(
    path: Path,
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or value.get("status") != "complete"
        or value.get("unresolved_count") != 0
    ):
        raise ValueError(
            f"{path}: input receipt must have status=complete and unresolved_count=0"
        )
    inventory = value.get("source_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError(f"{path}: complete receipt has no source_inventory")
    inventory_sha256 = value.get("source_inventory_sha256")
    if inventory_sha256 != _canonical_sha256(inventory):
        raise ValueError(f"{path}: source_inventory_sha256 does not match inventory")

    normalized: list[dict[str, Any]] = []
    for record in inventory:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not record["path"]
            or Path(record["path"]).is_absolute()
            or ".." in Path(record["path"]).parts
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(
                character not in "0123456789abcdef" for character in record["sha256"]
            )
            or not isinstance(record.get("size"), int)
            or isinstance(record["size"], bool)
            or record["size"] <= 0
        ):
            raise ValueError(f"{path}: malformed source inventory record {record!r}")
        normalized.append(
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "size": record["size"],
            }
        )
    if normalized != sorted(normalized, key=lambda record: record["path"]):
        raise ValueError(f"{path}: source_inventory must be sorted by path")
    if len({record["path"] for record in normalized}) != len(normalized):
        raise ValueError(f"{path}: source_inventory contains duplicate paths")
    return hashlib.sha256(raw).hexdigest(), inventory_sha256, normalized, value


def load_primary_route_receipt(
    receipt_path: Path,
    *,
    input_root: Path,
    source_composition: SourceComposition,
    kind: str,
    buckets: tuple[int, ...],
) -> dict[str, Any]:
    """Validate one completed route and return its primary snapshot inputs."""

    if kind not in {"code", "commits"}:
        raise ValueError(f"unsupported source route kind: {kind!r}")
    receipt_path = receipt_path.expanduser()
    if receipt_path.is_symlink():
        raise ValueError(f"source route receipt must not be a symlink: {receipt_path}")
    receipt_path = receipt_path.resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("unresolved_count") != 0
        or receipt.get("implementation") != _implementation()
    ):
        raise ValueError(f"{receipt_path}: source route receipt is not complete")

    input_root = input_root.expanduser().resolve()
    if Path(str(receipt.get("input_root"))).expanduser().resolve() != input_root:
        raise ValueError(f"{receipt_path}: source route input root drifted")
    output_root = Path(str(receipt.get("output_root"))).expanduser().resolve()
    primary_root = output_root / "primary"
    if output_root.is_symlink() or not primary_root.is_dir():
        raise ValueError(f"{receipt_path}: source route output root is invalid")

    expected_paths: dict[str, int] = {}
    for bucket in buckets:
        files = source_composition.allowlist.get((kind, bucket))
        if files is None:
            raise ValueError(f"source composition lacks {kind}/{bucket} allowlist")
        for filename, rows in files.items():
            relative = f"{bucket}/{filename}"
            if relative in expected_paths:
                raise ValueError(f"duplicate source route input path: {relative}")
            expected_paths[relative] = int(rows)

    input_binding = receipt.get("input_receipt")
    if not isinstance(input_binding, dict):
        raise ValueError(f"{receipt_path}: source route input receipt is missing")
    bound_input_receipt = Path(str(input_binding.get("path"))).expanduser().resolve()
    (
        input_sha256,
        inventory_sha256,
        source_inventory,
        input_receipt,
    ) = _complete_input_receipt(bound_input_receipt)
    if input_binding != {
        "path": str(bound_input_receipt),
        "sha256": input_sha256,
        "source_inventory_sha256": inventory_sha256,
    }:
        raise ValueError(f"{receipt_path}: source route input receipt drifted")
    inventory_by_path = {record["path"]: record for record in source_inventory}
    if set(inventory_by_path) != set(expected_paths):
        raise ValueError(
            f"{receipt_path}: source route inventory differs from source composition"
        )
    plan_sha256 = _sha256_file(source_composition.plan_path)
    expected_composition = {
        "plan": {
            "path": str(source_composition.plan_path),
            "sha256": plan_sha256,
        },
        "receipt_sha256": source_composition_receipt_sha256(
            source_composition.receipt
        ),
    }
    expected_totals = {
        "files": len(source_inventory),
        "rows": sum(expected_paths.values()),
        "bytes": sum(int(record["size"]) for record in source_inventory),
    }
    if (
        plan_sha256 != source_composition.receipt.get("plan_sha256")
        or input_receipt.get("schema") != PACKED_SOURCE_INVENTORY_SCHEMA
        or input_receipt.get("kind") != kind
        or input_receipt.get("input_root") != str(input_root)
        or input_receipt.get("buckets") != list(buckets)
        or input_receipt.get("source_composition") != expected_composition
        or input_receipt.get("totals") != expected_totals
    ):
        raise ValueError(
            f"{receipt_path}: source inventory provenance binding drifted"
        )

    raw_files = receipt.get("files")
    if not isinstance(raw_files, list):
        raise ValueError(f"{receipt_path}: source route file receipts are missing")
    file_receipts: dict[str, dict[str, Any]] = {}
    counts = {
        "source": _empty_counts(),
        **{route: _empty_counts() for route in ROUTES},
    }
    primary_allowlist: dict[tuple[str, int], dict[str, int]] = {
        (kind, bucket): {} for bucket in buckets
    }
    primary_sha256: dict[tuple[str, int, str], str] = {}
    output_inventory: list[dict[str, str]] = []
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            raise ValueError(f"{receipt_path}: malformed source route file receipt")
        source = raw_file.get("input")
        routes = raw_file.get("routes")
        if (
            raw_file.get("schema") != SCHEMA
            or raw_file.get("status") != "complete"
            or raw_file.get("unresolved_count") != 0
            or raw_file.get("implementation") != receipt["implementation"]
            or not isinstance(source, dict)
            or not isinstance(routes, dict)
            or set(routes) != set(ROUTES)
        ):
            raise ValueError(f"{receipt_path}: malformed source route file receipt")
        relative = source.get("path")
        if not isinstance(relative, str) or relative in file_receipts:
            raise ValueError(f"{receipt_path}: invalid source route input path")
        expected_rows = expected_paths.get(relative)
        inventory = inventory_by_path.get(relative)
        if (
            expected_rows is None
            or not isinstance(inventory, dict)
            or source.get("rows") != expected_rows
            or source.get("sha256") != inventory["sha256"]
            or source.get("size") != inventory["size"]
        ):
            raise ValueError(
                f"{receipt_path}: source route input binding drifted: {relative}"
            )
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) != 2
            or not relative_path.parts[0].isdigit()
            or int(relative_path.parts[0]) not in buckets
            or relative_path.suffix != ".parquet"
        ):
            raise ValueError(f"{receipt_path}: unsafe source route path: {relative}")
        bucket = int(relative_path.parts[0])
        filename = relative_path.name
        marker = output_root / "state" / relative_path.with_suffix(".route.json")
        if marker.is_symlink() or not marker.is_file():
            raise ValueError(f"{receipt_path}: source route marker is missing: {marker}")
        if json.loads(marker.read_text(encoding="utf-8")) != raw_file:
            raise ValueError(f"{receipt_path}: source route marker drifted: {marker}")

        for key in _empty_counts():
            value = source.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{receipt_path}: invalid source {key} for {relative}"
                )
            counts["source"][key] += value
        for route in ROUTES:
            artifact = routes[route]
            if not isinstance(artifact, dict):
                raise ValueError(
                    f"{receipt_path}: malformed {route} artifact for {relative}"
                )
            expected_artifact_path = f"{route}/{relative}"
            artifact_path = output_root / expected_artifact_path
            digest = artifact.get("sha256")
            if (
                artifact.get("path") != expected_artifact_path
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or artifact_path.is_symlink()
                or not artifact_path.is_file()
                or artifact.get("size") != artifact_path.stat().st_size
            ):
                raise ValueError(
                    f"{receipt_path}: {route} artifact binding drifted: {relative}"
                )
            _validate_output(artifact_path, expected_rows=int(artifact.get("rows", -1)))
            for key in _empty_counts():
                value = artifact.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"{receipt_path}: invalid {route} {key} for {relative}"
                    )
                counts[route][key] += value
            output_inventory.append(
                {
                    "route": route,
                    "path": expected_artifact_path,
                    "sha256": digest,
                }
            )
        _assert_conservation(
            {key: int(source[key]) for key in _empty_counts()},
            {key: int(routes["primary"][key]) for key in _empty_counts()},
            {key: int(routes["aux_python"][key]) for key in _empty_counts()},
            {
                key: int(routes[EXCLUDED_ROUTE][key])
                for key in _empty_counts()
            },
            where=input_root / relative_path,
        )
        primary = routes["primary"]
        primary_allowlist[(kind, bucket)][filename] = int(primary["rows"])
        primary_sha256[(kind, bucket, filename)] = primary["sha256"]
        file_receipts[relative] = raw_file

    if set(file_receipts) != set(expected_paths):
        raise ValueError(f"{receipt_path}: source route file set is incomplete")
    if (
        receipt.get("totals") != counts
        or receipt.get("input_inventory_sha256")
        != _canonical_sha256(source_inventory)
        or receipt.get("output_inventory_sha256")
        != _canonical_sha256(output_inventory)
        or receipt.get("output_schema_sha256")
        != hashlib.sha256(
            packer.PACKED_ROW_OUTPUT_SCHEMA.serialize().to_pybytes()
        ).hexdigest()
    ):
        raise ValueError(f"{receipt_path}: source route aggregate binding drifted")
    return {
        "binding": {
            "schema": SCHEMA,
            "kind": kind,
            "receipt_sha256": _sha256_file(receipt_path),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "primary_root": str(primary_root),
            "input_inventory_sha256": receipt["input_inventory_sha256"],
            "output_inventory_sha256": receipt["output_inventory_sha256"],
            "totals": counts,
            "implementation": receipt["implementation"],
        },
        "receipt_path": receipt_path,
        "primary_root": primary_root,
        "allowlist": primary_allowlist,
        "sha256": primary_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--buckets",
        type=_parse_buckets,
        default=(1024, 2048, 4096, 8192, 16384),
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1))
    )
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = args.input_root.resolve()
    input_receipt = args.input_receipt.resolve()
    output_root = args.output_root.resolve()
    (
        receipt_sha256,
        receipt_inventory_sha256,
        expected_inventory,
        _input_receipt,
    ) = _complete_input_receipt(input_receipt)
    files = _discover(input_root, args.buckets)
    relative_files = [str(path.relative_to(input_root)) for path in files]
    if relative_files != [record["path"] for record in expected_inventory]:
        raise RuntimeError("input parquet paths differ from completion receipt")
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{output_root} is not empty; pass --resume for this exact route"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    receipts: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                route_file,
                str(path),
                input_root_str=str(input_root),
                output_root_str=str(output_root),
                compression_level=args.compression_level,
                resume=args.resume,
            )
            for path in files
        ]
        for future in as_completed(futures):
            receipts.append(future.result())
    receipts.sort(key=lambda value: value["input"]["path"])

    if relative_files != [
        str(path.relative_to(input_root))
        for path in _discover(input_root, args.buckets)
    ]:
        raise RuntimeError("input parquet inventory changed while routing")
    if _sha256_file(input_receipt) != receipt_sha256:
        raise RuntimeError("input completion receipt changed while routing")
    observed_inventory = [
        {
            "path": receipt["input"]["path"],
            "sha256": receipt["input"]["sha256"],
            "size": receipt["input"]["size"],
        }
        for receipt in receipts
    ]
    if observed_inventory != expected_inventory:
        raise RuntimeError("input parquet bytes differ from completion receipt")

    totals = {
        "source": {
            key: sum(int(receipt["input"][key]) for receipt in receipts)
            for key in _empty_counts()
        },
        **{
            route: {
                key: sum(int(receipt["routes"][route][key]) for receipt in receipts)
                for key in _empty_counts()
            }
            for route in ROUTES
        },
    }
    _assert_conservation(
        totals["source"],
        totals["primary"],
        totals["aux_python"],
        totals[EXCLUDED_ROUTE],
        where=input_root,
    )
    global_receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "created_at": _utc_now(),
        "input_root": str(input_root),
        "input_receipt": {
            "path": str(input_receipt),
            "sha256": receipt_sha256,
            "source_inventory_sha256": receipt_inventory_sha256,
        },
        "output_root": str(output_root),
        "routing": {
            "primary": (
                "C/C++/SQL/native build diagnostics plus native-workflow "
                "path-bound shell"
            ),
            "aux_python": "source_build_kinds == 'python'",
            EXCLUDED_ROUTE: "annotated but outside the primary training scope",
            "mixed_rows": sum(int(receipt["mixed_rows_split"]) for receipt in receipts),
            "legacy_schema_upgrade": {
                "added_column": packed.SOURCE_COMMIT_HASHES_COLUMN,
                "value": "shared commit_hash or null per source document",
            },
        },
        "totals": totals,
        "input_inventory_sha256": _canonical_sha256(observed_inventory),
        "output_inventory_sha256": _canonical_sha256(
            [
                {
                    "route": route,
                    "path": receipt["routes"][route]["path"],
                    "sha256": receipt["routes"][route]["sha256"],
                }
                for receipt in receipts
                for route in ROUTES
            ]
        ),
        "output_schema_sha256": hashlib.sha256(
            packer.PACKED_ROW_OUTPUT_SCHEMA.serialize().to_pybytes()
        ).hexdigest(),
        "implementation": _implementation(),
        "files": receipts,
        "unresolved_count": 0,
    }
    _write_json_atomic(output_root / "route.receipt.json", global_receipt)
    print(
        json.dumps(
            {
                "status": "complete",
                "totals": totals,
                "mixed_rows": global_receipt["routing"]["mixed_rows"],
                "receipt": str(output_root / "route.receipt.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
