#!/usr/bin/env python3
# ruff: noqa: E402
"""Add token-level enriched columns to an existing enriched parquet dataset.

Reads local parquet shards, preserves existing columns, and writes a sibling
dataset with token_ids plus token-level enriched metadata columns materialized
offline. This is intended for A/B dataloader benchmarking against the legacy
char-level parquet path without round-tripping through JSONL again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
from cppmega.data.nanochat_pipeline.tokenized_enriched import materialize_tokenized_enriched_batch
from cppmega.data.nanochat_pipeline.tokenized_enriched_schema import TOKENIZED_ENRICHED_COLUMNS
from cppmega.data.nanochat_pipeline.tokenized_enriched_schema import (
    TOKEN_CALL_TARGETS_COLUMN,
    TOKEN_SYMBOL_IDS_COLUMN,
    TOKEN_TYPE_REFS_COLUMN,
)
from cppmega.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SymbolIdentityError,
    SymbolIdentityRegistry,
)
from scripts.nanochat_data.atomic_publish import atomic_output_directory


_SYMBOL_ID_TOKEN_COLUMNS = frozenset(
    (TOKEN_SYMBOL_IDS_COLUMN, TOKEN_CALL_TARGETS_COLUMN, TOKEN_TYPE_REFS_COLUMN)
)


def _decode_json_like(value):
    while isinstance(value, str) and value:
        decoded = json.loads(value)
        if decoded == value:
            break
        value = decoded
    return value


def _list_parquet_files(input_dir: str) -> list[Path]:
    return sorted(
        Path(input_dir).glob("*.parquet"),
        key=lambda p: p.name,
    )


def _table_to_docs(table: pa.Table) -> list[dict]:
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    num_rows = table.num_rows
    docs = []
    for i in range(num_rows):
        row = {name: values[i] for name, values in columns.items()}
        for json_list_field in (
            "chunk_boundaries",
            "call_edges",
            "type_edges",
            "domain_edges",
            "build_edges",
            "shell_edges",
            "diagnostic_edges",
            "cross_domain_edges",
        ):
            value = row.get(json_list_field)
            if value is None:
                continue
            value = _decode_json_like(value)
            if isinstance(value, list):
                value = [_decode_json_like(item) for item in value]
            row[json_list_field] = value
        for json_obj_field in ("platform_info", "language_info", "build_info"):
            value = row.get(json_obj_field)
            if value is None:
                continue
            row[json_obj_field] = _decode_json_like(value)
        docs.append(row)
    return docs


def _merge_table_with_tokenized(
    table: pa.Table,
    tokenized_rows: list[dict],
) -> pa.Table:
    base_columns = {name: table.column(name) for name in table.column_names}
    for column_name in TOKENIZED_ENRICHED_COLUMNS:
        base_columns[column_name] = pa.array(
            [row.get(column_name, []) for row in tokenized_rows],
            type=(pa.list_(pa.uint64()) if column_name in _SYMBOL_ID_TOKEN_COLUMNS else None),
        )
    return pa.table(base_columns).replace_schema_metadata(table.schema.metadata)


def _validate_symbol_identity_table(
    table: pa.Table,
    *,
    source: Path,
    corpus_registry: SymbolIdentityRegistry,
) -> None:
    raw_version = (table.schema.metadata or {}).get(
        SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")
    )
    if raw_version != str(SYMBOL_IDENTITY_SCHEMA_VERSION).encode("ascii"):
        raise SymbolIdentityError(
            f"{source}: symbol identity schema must be v"
            f"{SYMBOL_IDENTITY_SCHEMA_VERSION}, got {raw_version!r}"
        )
    if SYMBOL_IDENTITIES_COLUMN not in table.column_names:
        raise SymbolIdentityError(f"{source}: missing {SYMBOL_IDENTITIES_COLUMN!r}")
    for column in ("symbol_ids", "call_targets", "type_refs"):
        if column not in table.column_names:
            continue
        field_type = table.schema.field(column).type
        if not pa.types.is_list(field_type) or field_type.value_type != pa.uint64():
            raise SymbolIdentityError(
                f"{source}: {column} must use uint64 symbol IDs, got "
                f"{field_type}"
            )
    identity_rows = table.column(SYMBOL_IDENTITIES_COLUMN).to_pylist()
    semantic_rows = {
        column: table.column(column).to_pylist()
        for column in ("symbol_ids", "call_targets", "type_refs")
        if column in table.column_names
    }
    for row_index, records in enumerate(identity_rows):
        row_registry = SymbolIdentityRegistry()
        normalized = row_registry.register_records(
            records,
            source=f"{source}:row={row_index}",
        )
        corpus_registry.register_records(
            normalized,
            source=f"{source}:row={row_index}",
        )
        used_ids = {
            int(value)
            for rows in semantic_rows.values()
            for value in rows[row_index]
            if int(value) != 0
        }
        row_registry.require_ids(used_ids, source=f"{source}:row={row_index}")


def _copy_completion_marker(input_dir: str, output_dir: Path) -> None:
    src = Path(input_dir) / "_COMPLETE"
    if src.exists():
        (output_dir / "_COMPLETE").write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Only process the first N parquet shards (0 = all).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="Optional tokenizer.json path used for token materialization.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=1024,
        help="Row group size for rewritten output shards.",
    )
    args = parser.parse_args()

    all_parquet_files = _list_parquet_files(args.input_dir)
    parquet_files = list(all_parquet_files)
    if args.max_files > 0:
        parquet_files = parquet_files[: args.max_files]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.input_dir}")

    tokenizer_path = args.tokenizer_path
    if not tokenizer_path:
        raise ValueError("--tokenizer-path is required for parquet materialization")
    tokenizer = load_cppmega_tokenizer(tokenizer_path)
    identity_registry = SymbolIdentityRegistry()

    with atomic_output_directory(args.output_dir) as staged_output:
        for idx, src_path in enumerate(parquet_files, 1):
            dst_path = staged_output / src_path.name
            print(f"[{idx}/{len(parquet_files)}] {src_path.name}")
            table = pq.read_table(src_path)
            _validate_symbol_identity_table(
                table,
                source=src_path,
                corpus_registry=identity_registry,
            )
            docs = _table_to_docs(table)
            tokenized_rows = materialize_tokenized_enriched_batch(docs, tokenizer)
            merged = _merge_table_with_tokenized(table, tokenized_rows)
            pq.write_table(
                merged,
                dst_path,
                row_group_size=args.row_group_size,
                compression="snappy",
            )

        if len(parquet_files) == len(all_parquet_files):
            _copy_completion_marker(args.input_dir, staged_output)
    print(f"Wrote tokenized parquet dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
