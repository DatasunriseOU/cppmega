#!/usr/bin/env python3
# ruff: noqa: E402
"""Convert Clang-enriched clang-indexer JSONL to hard-budgeted parquet.

Reads gs://nanochat-training-data-2026/v5_enriched/*.jsonl.gz, applies:
  1. Dead-platform filter (removes __SYMBIAN32__, _MSDOS, __VXWORKS__ blocks)
  2. Platform header prepend (x86_64-linux-gnu / g++ / c++17 default)
  3. tokenizer-aware hard-budget handling using chunk boundaries

Writes parquet shards to:
  gs://nanochat-training-data-2026/data/parquet/clang_enriched_<size>/

Usage:
    python scripts/data/clang_enriched_to_4k_parquet.py --size 4k [--dry-run] [--shard-size 10000]
    python scripts/data/clang_enriched_to_4k_parquet.py --input-file repo.jsonl --output-file repo.parquet --overflow-policy drop
    python scripts/data/clang_enriched_to_4k_parquet.py --help
"""

import argparse
from bisect import bisect_left
import gzip
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

from cppmega.data.domain_schema import (
    DOMAIN_EDGE_FIELD_FAMILIES,
    DOMAIN_SCHEMA_SHA256,
    DOMAIN_SCHEMA_SHA256_METADATA_KEY,
    canonicalize_domain_edge_fields,
    normalize_domain_edge_record,
    remap_embedded_domain_spans,
)
from cppmega.data.nanochat_pipeline.language_info import language_info_to_prefix
from cppmega.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SymbolIdentityError,
    SymbolIdentityRegistry,
    require_project_identity,
)
from cppmega.data.source_identity import (
    normalize_positive_source_ids,
    normalize_row_local_doc_ids,
    source_identity,
    validate_source_identity_registry,
)
from cppmega.data.tokenizer_contract import (
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY,
)
from cppmega.data.nanochat_pipeline.tokenized_enriched import (
    PLATFORM_IDS_COLUMN,
    TOKEN_AST_DEPTH_COLUMN,
    TOKEN_AST_NODE_TYPE_COLUMN,
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_CALL_TARGETS_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_CONFIDENCE_IDS_COLUMN,
    TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
    TOKEN_DEF_USE_COLUMN,
    TOKEN_DEP_LEVELS_COLUMN,
    TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    TOKEN_DOMAIN_EDGES_COLUMN,
    TOKEN_DOMAIN_IDS_COLUMN,
    TOKEN_BUILD_EDGES_COLUMN,
    TOKEN_ENTITY_IDS_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_ROLE_IDS_COLUMN,
    TOKEN_SCOPE_IDS_COLUMN,
    TOKEN_SIBLING_INDEX_COLUMN,
    TOKEN_SHELL_EDGES_COLUMN,
    TOKEN_SOURCE_DOC_IDS_COLUMN,
    TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
    TOKEN_STRUCTURE_IDS_COLUMN,
    TOKEN_SYMBOL_IDS_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
    TOKEN_TYPE_REFS_COLUMN,
    materialize_tokenized_enriched_batch,
)
from cppmega.data.nanochat_pipeline.tokenized_enriched_schema import (
    AUTHOR_TIMESTAMP_COLUMN,
    BUILD_KIND_COLUMN,
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
    COMMIT_MSG_TEXT_COLUMN,
    COMMIT_MSG_TOKEN_IDS_COLUMN,
    COMMIT_HASH_COLUMN,
    COMMIT_TIMESTAMP_COLUMN,
    DOC_TYPE_COLUMN,
    DIFF_TEXT_COLUMN,
    DIFF_TOKEN_IDS_COLUMN,
    EDIT_OP_PER_TOKEN_COLUMN,
    FILEPATH_COLUMN,
    FILEPATH_STABLE_ID_COLUMN,
    FILE_LOCAL_COMMIT_INDEX_COLUMN,
    HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN,
    HEADER_FRAGMENT_KIND_COLUMN,
    HAS_PR_DISCUSSION_COLUMN,
    HAS_RENAME_AMBIGUITY_COLUMN,
    HUNK_ID_PER_TOKEN_COLUMN,
    IFIM_INSTRUCTION_TEXT_COLUMN,
    IFIM_INSTRUCTION_TOKEN_IDS_COLUMN,
    IS_MERGE_COMMIT_COLUMN,
    PARENT_COUNT_COLUMN,
    PARENT_HASHES_COLUMN,
    POST_TEXT_COLUMN,
    POST_TOKEN_IDS_COLUMN,
    PRE_TEXT_COLUMN,
    PRE_TOKEN_IDS_COLUMN,
    PR_DISCUSSION_CHARS_COLUMN,
    PR_DISCUSSION_LINES_COLUMN,
    PR_NUMBER_COLUMN,
    REPO_COLUMN,
    REPO_STABLE_ID_COLUMN,
    TIMESTAMP_COLUMN,
    TOKEN_CHANGE_MASK_POST_COLUMN,
    TOKEN_CHANGE_MASK_PRE_COLUMN,
    SOURCE_IDENTITY_REGISTRY_COLUMN,
)

def _normalize_char_domain_edges(
    raw_edges: object,
    *,
    family: str,
) -> list[dict[str, int]]:
    return [
        {"from_char": src, "to_char": dst, "kind": kind}
        for src, dst, kind in (
            normalize_domain_edge_record(edge, family=family)
            for edge in (raw_edges or [])  # type: ignore[union-attr]
        )
    ]


from scripts.nanochat_data.token_budget import (
    chunk_enriched_document,
    count_tokens,
    load_tokenizer,
    size_label_to_tokens,
    tokenizer_fingerprint,
)
from scripts.nanochat_data.memory_guard import check_memory_limit, start_memory_guard
from scripts.nanochat_data.atomic_publish import atomic_output_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("clang_enriched_to_4k")

GCS_BUCKET = "nanochat-training-data-2026"
GCS_INPUT_PREFIX = "v5_enriched"
GCS_OUTPUT_PREFIX_TEMPLATE = "data/parquet/clang_enriched_{size}"
_TOKENIZED_ENRICHED_TOKENIZER = None
_MEMORY_LIMIT_GB = 10.0
_OVERFLOW_POLICIES = ("split", "drop")
_MATERIALIZE_SPLIT_STATS_SCHEMA = "cppmega.materialize_split_stats_v1"
_CROSS_PIECE_EDGE_FIELDS = (
    "call_edges",
    "type_edges",
    *DOMAIN_EDGE_FIELD_FAMILIES,
)
REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION = SYMBOL_IDENTITY_SCHEMA_VERSION
_CHAR_LEVEL_METADATA_FIELDS = (
    "ast_depth",
    "sibling_index",
    "ast_node_type",
    "symbol_ids",
    "call_targets",
    "type_refs",
    "def_use",
    # Per-char commit edit-signal arrays (from process_commits.py). These are
    # consumed by materialize_tokenized_enriched_batch via the char->token
    # mapping; they must be header-offset-aligned exactly like the others so
    # token positions line up after the platform/language header is prepended.
    "change_mask_pre",
    "change_mask_post",
    "hunk_id_per_char",
    "edit_op_per_char",
    "domain_ids",
    "domain_role_ids",
    "domain_entity_ids",
    "domain_scope_ids",
    "domain_source_doc_ids",
    "domain_source_identity_ids",
    "domain_confidence_ids",
)
_STATIC_DOC_TYPES = {"code", "code_header", "build"}


def stable_repo_id(repo_name: str) -> str:
    return hashlib.sha1(repo_name.encode("utf-8")).hexdigest()[:16]


def stable_filepath_id(repo_name: str, filepath: str) -> str:
    key = f"{repo_name}\0{filepath}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _has_value(value: object) -> bool:
    return value is not None and value != ""


def _is_static_code_record(record: dict) -> bool:
    has_temporal_provenance = any(
        _has_value(record.get(column))
        for column in (
            COMMIT_HASH_COLUMN,
            FILE_LOCAL_COMMIT_INDEX_COLUMN,
            AUTHOR_TIMESTAMP_COLUMN,
            COMMIT_TIMESTAMP_COLUMN,
            TIMESTAMP_COLUMN,
        )
    )
    if has_temporal_provenance:
        return False
    doc_type = record.get(DOC_TYPE_COLUMN)
    if doc_type in _STATIC_DOC_TYPES:
        return True
    if not _has_value(record.get(FILEPATH_COLUMN)):
        return False
    return True


def _validate_static_provenance(record: dict) -> None:
    if not _is_static_code_record(record):
        return
    if not _has_value(record.get(REPO_COLUMN)):
        raise ValueError(
            "missing static code repo provenance "
            f"for filepath={record.get(FILEPATH_COLUMN)!r}"
        )
    if not _has_value(record.get(FILEPATH_COLUMN)):
        raise ValueError(
            "missing static code filepath provenance "
            f"for repo={record.get(REPO_COLUMN)!r}"
        )


def _with_default_static_provenance(
    record: dict,
    *,
    default_repo: str | None,
) -> dict:
    repo = record.get(REPO_COLUMN)
    if not _has_value(repo) and default_repo:
        record = dict(record)
        record[REPO_COLUMN] = default_repo
        repo = default_repo
    if not _has_value(repo):
        _validate_static_provenance(record)
        return record
    repo = require_project_identity(
        repo, source=f"clang enriched row {record.get(FILEPATH_COLUMN)!r}"
    )
    if record.get(REPO_COLUMN) != repo:
        record = dict(record)
        record[REPO_COLUMN] = repo

    filepath = record.get(FILEPATH_COLUMN)
    if not _has_value(record.get(REPO_STABLE_ID_COLUMN)):
        record = dict(record)
        record[REPO_STABLE_ID_COLUMN] = stable_repo_id(str(repo))
    if _has_value(filepath) and not _has_value(record.get(FILEPATH_STABLE_ID_COLUMN)):
        record = dict(record)
        record[FILEPATH_STABLE_ID_COLUMN] = stable_filepath_id(
            str(repo),
            str(filepath),
        )
    _validate_static_provenance(record)
    return record

# ---------------------------------------------------------------------------
# Platform header (default — enriched docs are already processed, no repo_dir)
# ---------------------------------------------------------------------------

_DEFAULT_PLATFORM_INFO = {
    "os": ["linux"],
    "rtos": [],
    "gpu": [],
    "arch": ["x64"],
    "compiler": ["gcc"],
    "cpp_std": "c++17",
}

_DEFAULT_PLATFORM_HEADER_INFO = {
    "platform": "x86_64-linux-gnu",
    "compiler": "g++",
    "standard": "c++17",
    "arch": "x86_64",
    "mode": "user",
}


def build_platform_header(platform_info: dict) -> str:
    """Build a platform context comment header."""
    platform = platform_info.get("platform", "x86_64-linux-gnu")
    compiler = platform_info.get("compiler", "g++")
    standard = platform_info.get("standard", "c++17")
    arch = platform_info.get("arch", "x86_64")
    mode = platform_info.get("mode", "user")
    lines = [
        "// <BOS>",
        f"// platform: {platform}",
        f"// compiler: {compiler}",
        f"// standard: {standard}",
        f"// arch: {arch}",
        f"// mode: {mode}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dead-platform filter
# ---------------------------------------------------------------------------

_DEAD_PLATFORM_MARKERS = [
    "__SYMBIAN32__",
    "_MSDOS",
    "__VXWORKS__",
]


def filter_dead_platforms_with_mapping(text: str) -> tuple[str, list[int] | None]:
    """Remove dead-platform #ifdef blocks and return kept-char provenance.

    Returns ``(filtered_text, kept_indices)``.  ``kept_indices`` is ``None`` when
    no filtering was applied; otherwise each filtered char maps back to its
    original char offset.  Downstream sidecars use this to remap metadata instead
    of zeroing graph routes for the whole document.
    """
    if not any(marker in text for marker in _DEAD_PLATFORM_MARKERS):
        return text, None

    lines = text.splitlines(keepends=True)
    result_parts: list[str] = []
    kept_indices: list[int] = []
    skip_depth = 0
    offset = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        line_start = offset
        offset += len(line)
        stripped = line.strip()

        if skip_depth == 0:
            is_dead = False
            for marker in _DEAD_PLATFORM_MARKERS:
                if re.match(r'^\s*#\s*ifdef\s+' + re.escape(marker) + r'\b', line):
                    is_dead = True
                    break
                if re.match(r'^\s*#\s*if\s+defined\s*\(\s*' + re.escape(marker) + r'\s*\)', line):
                    is_dead = True
                    break

            if is_dead:
                skip_depth = 1
                i += 1
                continue

            result_parts.append(line)
            kept_indices.extend(range(line_start, line_start + len(line)))
        else:
            if re.match(r'^\s*#\s*(?:ifdef|ifndef|if)\b', stripped):
                skip_depth += 1
            elif re.match(r'^\s*#\s*endif\b', stripped):
                skip_depth -= 1
                if skip_depth == 0:
                    i += 1
                    continue

        i += 1

    filtered = "".join(result_parts)
    if len(kept_indices) == len(text):
        return filtered, None
    return filtered, kept_indices


def filter_dead_platforms(text: str) -> str:
    """Remove dead-platform #ifdef blocks from C++ source text."""
    return filter_dead_platforms_with_mapping(text)[0]


# ---------------------------------------------------------------------------
# Token-budgeted chunker
# ---------------------------------------------------------------------------

PREAMBLE_KIND = 1
# Build/compilation files are emitted with structure_ids ALL set to this kind
# (extends the 0-8 code-kind vocab from index_project.build_enriched_doc). A
# whole 'build' doc is a single BUILD_KIND span; it carries no call/type graph.
BUILD_KIND = 9


def chunk_document_exact(doc: dict, tokenizer, max_tokens: int) -> list:
    """Chunk an enriched document into exact-token-budgeted sub-documents."""
    def sort_key(cb):
        is_preamble = (
            cb.get("dep_level", 0) == 0 and cb.get("kind", 0) == PREAMBLE_KIND
        )
        return (0 if is_preamble else 1, cb.get("dep_level", 0), cb.get("start", 0))

    return chunk_enriched_document(
        doc,
        max_tokens,
        tokenizer,
        boundary_sort_key=lambda item: sort_key(item[1]),
    )


def maybe_keep_document_exact(doc: dict, tokenizer, max_tokens: int) -> list[dict]:
    """Emit the whole document only when it already fits the exact budget."""
    exact_total = count_tokens(doc.get("text", ""), tokenizer)
    if exact_total > max_tokens:
        return []
    out = dict(doc)
    out["actual_token_count"] = exact_total
    return [out]


# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------

_SCHEMA = pa.schema([
    pa.field("text", pa.string()),
    pa.field(SYMBOL_IDENTITIES_COLUMN, pa.list_(pa.struct([
        pa.field("symbol_id", pa.uint64()),
        pa.field("symbol_key", pa.string()),
    ]))),
    pa.field("source_text", pa.string()),
    pa.field(IFIM_INSTRUCTION_TEXT_COLUMN, pa.string()),
    pa.field(COMMIT_MSG_TEXT_COLUMN, pa.string()),
    pa.field(PRE_TEXT_COLUMN, pa.string()),
    pa.field(POST_TEXT_COLUMN, pa.string()),
    pa.field(DIFF_TEXT_COLUMN, pa.string()),
    pa.field("source_doc_id", pa.string()),
    pa.field(DOC_TYPE_COLUMN, pa.string()),
    pa.field(HEADER_FRAGMENT_KIND_COLUMN, pa.string()),
    pa.field(BUILD_KIND_COLUMN, pa.string()),
    pa.field("tokenizer_fingerprint", pa.string()),
    pa.field("actual_token_count", pa.int32()),
    pa.field("structure_ids", pa.list_(pa.int8())),
    pa.field("chunk_boundaries", pa.list_(pa.struct([
        pa.field("start", pa.int32()),
        pa.field("end", pa.int32()),
        pa.field("kind", pa.int8()),
        pa.field("dep_level", pa.int32()),
        pa.field("name", pa.string()),
        pa.field("symbol_id", pa.uint64()),
    ]))),
    pa.field("call_edges", pa.list_(pa.struct([
        pa.field("from", pa.int32()),
        pa.field("to", pa.int32()),
    ]))),
    pa.field("type_edges", pa.list_(pa.struct([
        pa.field("from", pa.int32()),
        pa.field("to", pa.int32()),
    ]))),
    pa.field("ast_depth", pa.list_(pa.uint16())),
    pa.field("sibling_index", pa.list_(pa.uint16())),
    pa.field("ast_node_type", pa.list_(pa.uint16())),
    pa.field("symbol_ids", pa.list_(pa.uint64())),
    pa.field("call_targets", pa.list_(pa.uint64())),
    pa.field("type_refs", pa.list_(pa.uint64())),
    pa.field("def_use", pa.list_(pa.uint8())),
    pa.field("domain_kind", pa.uint16()),
    pa.field("domain_ids", pa.list_(pa.uint16())),
    pa.field("domain_role_ids", pa.list_(pa.uint16())),
    pa.field("domain_entity_ids", pa.list_(pa.uint32())),
    pa.field("domain_scope_ids", pa.list_(pa.uint32())),
    pa.field("domain_source_doc_ids", pa.list_(pa.uint32())),
    pa.field("domain_source_identity_ids", pa.list_(pa.uint64())),
    pa.field("domain_confidence_ids", pa.list_(pa.uint8())),
    pa.field("domain_edges", pa.list_(pa.struct([
        pa.field("from_char", pa.int32()),
        pa.field("to_char", pa.int32()),
        pa.field("kind", pa.int32()),
    ]))),
    pa.field("build_edges", pa.list_(pa.struct([
        pa.field("from_char", pa.int32()),
        pa.field("to_char", pa.int32()),
        pa.field("kind", pa.int32()),
    ]))),
    pa.field("shell_edges", pa.list_(pa.struct([
        pa.field("from_char", pa.int32()),
        pa.field("to_char", pa.int32()),
        pa.field("kind", pa.int32()),
    ]))),
    pa.field("diagnostic_edges", pa.list_(pa.struct([
        pa.field("from_char", pa.int32()),
        pa.field("to_char", pa.int32()),
        pa.field("kind", pa.int32()),
    ]))),
    pa.field("cross_domain_edges", pa.list_(pa.struct([
        pa.field("from_char", pa.int32()),
        pa.field("to_char", pa.int32()),
        pa.field("kind", pa.int32()),
    ]))),
    # Per-char commit edit-signal arrays (from process_commits.py). Persisting
    # them here is what lets the standalone materializer
    # (materialize_tokenized_enriched_parquet.py) map them to populated
    # token-level edit columns; without these columns the two-stage path would
    # silently emit EMPTY token edit signals (a forbidden zero/absent fallback).
    pa.field("change_mask_pre", pa.list_(pa.uint8())),
    pa.field("change_mask_post", pa.list_(pa.uint8())),
    pa.field("hunk_id_per_char", pa.list_(pa.int32())),
    pa.field("edit_op_per_char", pa.list_(pa.uint8())),
    pa.field("platform_info", pa.string()),
    pa.field("language_info", pa.string()),
    pa.field("build_info", pa.string()),
    pa.field("constituent_provenance", pa.list_(pa.struct([
        pa.field("filepath", pa.string()),
        pa.field("language_info", pa.string()),
        pa.field("build_info", pa.string()),
    ]))),
    pa.field("constituent_provenance_json", pa.string()),
    pa.field(REPO_COLUMN, pa.string()),
    pa.field(FILEPATH_COLUMN, pa.string()),
    pa.field(COMMIT_HASH_COLUMN, pa.string()),
    pa.field(TIMESTAMP_COLUMN, pa.string()),
    pa.field(PR_NUMBER_COLUMN, pa.int64()),
    pa.field(HAS_PR_DISCUSSION_COLUMN, pa.bool_()),
    pa.field(PR_DISCUSSION_CHARS_COLUMN, pa.int32()),
    pa.field(PR_DISCUSSION_LINES_COLUMN, pa.int32()),
    pa.field(PARENT_HASHES_COLUMN, pa.list_(pa.string())),
    pa.field(PARENT_COUNT_COLUMN, pa.int32()),
    pa.field(IS_MERGE_COMMIT_COLUMN, pa.bool_()),
    pa.field(AUTHOR_TIMESTAMP_COLUMN, pa.string()),
    pa.field(COMMIT_TIMESTAMP_COLUMN, pa.string()),
    pa.field(REPO_STABLE_ID_COLUMN, pa.string()),
    pa.field(FILEPATH_STABLE_ID_COLUMN, pa.string()),
    pa.field(FILE_LOCAL_COMMIT_INDEX_COLUMN, pa.int32()),
    pa.field(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN, pa.bool_()),
    pa.field(HAS_RENAME_AMBIGUITY_COLUMN, pa.bool_()),
    pa.field(TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(IFIM_INSTRUCTION_TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(COMMIT_MSG_TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(PRE_TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(POST_TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(DIFF_TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(PLATFORM_IDS_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_STRUCTURE_IDS_COLUMN, pa.list_(pa.uint8())),
    pa.field(TOKEN_DEP_LEVELS_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_AST_DEPTH_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_SIBLING_INDEX_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_AST_NODE_TYPE_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_SYMBOL_IDS_COLUMN, pa.list_(pa.uint64())),
    pa.field(TOKEN_CALL_TARGETS_COLUMN, pa.list_(pa.uint64())),
    pa.field(TOKEN_TYPE_REFS_COLUMN, pa.list_(pa.uint64())),
    pa.field(TOKEN_DEF_USE_COLUMN, pa.list_(pa.uint8())),
    pa.field(TOKEN_DOMAIN_IDS_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_ROLE_IDS_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_ENTITY_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(TOKEN_SCOPE_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(TOKEN_SOURCE_DOC_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(TOKEN_SOURCE_IDENTITY_IDS_COLUMN, pa.list_(pa.uint64())),
    pa.field(TOKEN_CONFIDENCE_IDS_COLUMN, pa.list_(pa.uint8())),
    pa.field(TOKEN_CHANGE_MASK_PRE_COLUMN, pa.list_(pa.uint8())),
    pa.field(TOKEN_CHANGE_MASK_POST_COLUMN, pa.list_(pa.uint8())),
    pa.field(HUNK_ID_PER_TOKEN_COLUMN, pa.list_(pa.int32())),
    pa.field(EDIT_OP_PER_TOKEN_COLUMN, pa.list_(pa.uint8())),
    pa.field(TOKEN_CHUNK_STARTS_COLUMN, pa.list_(pa.uint32())),
    pa.field(TOKEN_CHUNK_ENDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(TOKEN_CHUNK_KINDS_COLUMN, pa.list_(pa.uint8())),
    pa.field(TOKEN_CHUNK_DEP_LEVELS_COLUMN, pa.list_(pa.uint16())),
    pa.field(CHANGED_CHUNK_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(
        CHANGED_CHUNK_SPANS_COLUMN,
        pa.list_(
            pa.struct(
                [
                    pa.field("start", pa.uint32()),
                    pa.field("end", pa.uint32()),
                ]
            )
        ),
    ),
    pa.field(TOKEN_CALL_EDGES_COLUMN, pa.list_(pa.struct([
        pa.field("from", pa.uint16()),
        pa.field("to", pa.uint16()),
    ]))),
    pa.field(TOKEN_TYPE_EDGES_COLUMN, pa.list_(pa.struct([
        pa.field("from", pa.uint16()),
        pa.field("to", pa.uint16()),
    ]))),
    pa.field(TOKEN_DOMAIN_EDGES_COLUMN, pa.list_(pa.struct([
        pa.field("from", pa.uint32()),
        pa.field("to", pa.uint32()),
        pa.field("kind", pa.int32()),
    ]))),
    pa.field(TOKEN_BUILD_EDGES_COLUMN, pa.list_(pa.struct([
        pa.field("from", pa.uint32()),
        pa.field("to", pa.uint32()),
        pa.field("kind", pa.int32()),
    ]))),
    pa.field(TOKEN_SHELL_EDGES_COLUMN, pa.list_(pa.struct([
        pa.field("from", pa.uint32()),
        pa.field("to", pa.uint32()),
        pa.field("kind", pa.int32()),
    ]))),
    pa.field(TOKEN_DIAGNOSTIC_EDGES_COLUMN, pa.list_(pa.struct([
        pa.field("from", pa.uint32()),
        pa.field("to", pa.uint32()),
        pa.field("kind", pa.int32()),
    ]))),
    pa.field(TOKEN_CROSS_DOMAIN_EDGES_COLUMN, pa.list_(pa.struct([
        pa.field("from", pa.uint32()),
        pa.field("to", pa.uint32()),
        pa.field("kind", pa.int32()),
    ]))),
    pa.field(SOURCE_IDENTITY_REGISTRY_COLUMN, pa.list_(pa.struct([
        pa.field("source_identity_id", pa.uint64(), nullable=False),
        pa.field("canonical_sha256", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]))),
])
_SCHEMA = _SCHEMA.with_metadata(
    {
        SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii"): str(
            REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION
        ).encode("ascii"),
        DOMAIN_DELIMITER_CONTRACT_METADATA_KEY.encode("utf-8"): (
            DOMAIN_DELIMITER_CONTRACT_SHA256.encode("ascii")
        ),
        DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8"): (
            DOMAIN_SCHEMA_SHA256.encode("ascii")
        ),
        TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode("utf-8"): (
            TOKENIZER_CONTRACT_SHA256.encode("ascii")
        ),
        b"cppmega.case5_schema": b"case5_domain_routes_v1",
    }
)


def rows_to_table(
    rows: list,
    *,
    tokenized_rows: list[dict] | None = None,
    identity_registry: SymbolIdentityRegistry | None = None,
) -> pa.Table:
    """Convert a list of doc dicts to a PyArrow table."""
    tokenized_rows = tokenized_rows or [{} for _ in rows]
    if len(tokenized_rows) != len(rows):
        raise ValueError(
            "tokenized_rows length must match rows: "
            f"{len(tokenized_rows)} != {len(rows)}"
        )
    corpus_registry = identity_registry or SymbolIdentityRegistry()
    texts = []
    symbol_identities_col = []
    source_texts = []
    ifim_instruction_texts = []
    commit_msg_texts = []
    pre_texts = []
    post_texts = []
    diff_texts = []
    source_doc_ids = []
    doc_types = []
    header_fragment_kinds = []
    build_kinds = []
    tokenizer_fingerprints = []
    token_counts = []
    structure_ids_col = []
    chunk_boundaries_col = []
    call_edges_col = []
    type_edges_col = []
    ast_depth_col = []
    sibling_index_col = []
    ast_node_type_col = []
    symbol_ids_col = []
    call_targets_col = []
    type_refs_col = []
    def_use_col = []
    domain_kind_col = []
    domain_ids_col = []
    domain_role_ids_col = []
    domain_entity_ids_col = []
    domain_scope_ids_col = []
    domain_source_doc_ids_col = []
    domain_source_identity_ids_col = []
    domain_confidence_ids_col = []
    domain_edges_col = []
    build_edges_col = []
    shell_edges_col = []
    diagnostic_edges_col = []
    cross_domain_edges_col = []
    change_mask_pre_col = []
    change_mask_post_col = []
    hunk_id_per_char_col = []
    edit_op_per_char_col = []
    platform_info_col = []
    language_info_col = []
    build_info_col = []
    constituent_provenance_col = []
    constituent_provenance_json_col = []
    repos = []
    filepaths = []
    commits = []
    timestamps = []
    pr_numbers = []
    has_pr_discussions = []
    pr_discussion_chars = []
    pr_discussion_lines = []
    parent_hashes = []
    parent_counts = []
    is_merge_commits = []
    author_timestamps = []
    commit_timestamps = []
    repo_stable_ids = []
    filepath_stable_ids = []
    file_local_commit_indices = []
    ambiguous_reconstruction = []
    rename_ambiguity = []
    token_ids_col = []
    ifim_instruction_token_ids_col = []
    commit_msg_token_ids_col = []
    pre_token_ids_col = []
    post_token_ids_col = []
    diff_token_ids_col = []
    platform_ids_col = []
    token_structure_ids_col = []
    token_dep_levels_col = []
    token_ast_depth_col = []
    token_sibling_index_col = []
    token_ast_node_type_col = []
    token_symbol_ids_col = []
    token_call_targets_col = []
    token_type_refs_col = []
    token_def_use_col = []
    token_domain_ids_col = []
    token_role_ids_col = []
    token_entity_ids_col = []
    token_scope_ids_col = []
    token_source_doc_ids_col = []
    token_source_identity_ids_col = []
    token_confidence_ids_col = []
    token_change_mask_pre_col = []
    token_change_mask_post_col = []
    hunk_id_per_token_col = []
    edit_op_per_token_col = []
    token_chunk_starts_col = []
    token_chunk_ends_col = []
    token_chunk_kinds_col = []
    token_chunk_dep_levels_col = []
    changed_chunk_ids_col = []
    changed_chunk_spans_col = []
    token_call_edges_col = []
    token_type_edges_col = []
    token_domain_edges_col = []
    token_build_edges_col = []
    token_shell_edges_col = []
    token_diagnostic_edges_col = []
    token_cross_domain_edges_col = []
    source_identity_registry_col = []

    for row_index, (row, tokenized) in enumerate(
        zip(rows, tokenized_rows, strict=True)
    ):
        canonical_edge_fields = canonicalize_domain_edge_fields(
            row,
            source_length=len(row.get("text", "")),
        )
        identity_version = row.get("symbol_identity_schema_version")
        if identity_version != REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION:
            raise SymbolIdentityError(
                "clang-enriched row has missing or stale symbol identity schema "
                f"version {identity_version!r}; regenerate it with clang USR/signature "
                f"schema v{REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION}"
            )
        if SYMBOL_IDENTITIES_COLUMN not in row:
            raise SymbolIdentityError(
                f"clang-enriched row {row_index} has no {SYMBOL_IDENTITIES_COLUMN!r} "
                "collision registry; regenerate it with symbol identity schema v3"
            )
        row_registry = SymbolIdentityRegistry()
        normalized_identities = row_registry.register_records(
            row.get(SYMBOL_IDENTITIES_COLUMN),
            source=f"clang-enriched row {row_index}",
        )
        corpus_registry.register_records(
            normalized_identities,
            source=f"clang-enriched row {row_index}",
        )
        used_symbol_ids = {
            int(value)
            for field in ("symbol_ids", "call_targets", "type_refs")
            for value in row.get(field, [])
            if int(value) != 0
        }
        used_symbol_ids.update(
            int(boundary["symbol_id"])
            for boundary in row.get("chunk_boundaries", [])
            if isinstance(boundary, dict) and boundary.get("symbol_id") not in (None, 0)
        )
        used_symbol_ids.update(
            int(value)
            for field in (
                TOKEN_SYMBOL_IDS_COLUMN,
                TOKEN_CALL_TARGETS_COLUMN,
                TOKEN_TYPE_REFS_COLUMN,
            )
            for value in tokenized.get(field, [])
            if int(value) != 0
        )
        row_registry.require_ids(
            used_symbol_ids,
            source=f"clang-enriched row {row_index}",
        )
        symbol_identities_col.append(row_registry.records(used_symbol_ids))
        row_text = row.get("text", "")
        texts.append(row_text)
        source_texts.append(row.get("source_text"))
        ifim_instruction_texts.append(row.get(IFIM_INSTRUCTION_TEXT_COLUMN))
        commit_msg_texts.append(row.get(COMMIT_MSG_TEXT_COLUMN))
        pre_texts.append(row.get(PRE_TEXT_COLUMN))
        post_texts.append(row.get(POST_TEXT_COLUMN))
        diff_texts.append(row.get(DIFF_TEXT_COLUMN))
        raw_source_doc_id = row.get("source_doc_id")
        source_doc_ids.append(
            None if raw_source_doc_id is None else str(raw_source_doc_id)
        )
        raw_doc_type = row.get(DOC_TYPE_COLUMN)
        doc_types.append(None if raw_doc_type is None else str(raw_doc_type))
        raw_header_fragment_kind = row.get(HEADER_FRAGMENT_KIND_COLUMN)
        header_fragment_kinds.append(
            None
            if raw_header_fragment_kind is None
            else str(raw_header_fragment_kind)
        )
        raw_build_kind = row.get(BUILD_KIND_COLUMN)
        build_kinds.append(
            None if raw_build_kind is None else str(raw_build_kind)
        )
        raw_tokenizer_fingerprint = row.get("tokenizer_fingerprint")
        tokenizer_fingerprints.append(
            None
            if raw_tokenizer_fingerprint is None
            else str(raw_tokenizer_fingerprint)
        )
        token_counts.append(int(row.get("actual_token_count", 0)))
        structure_ids_col.append(row.get("structure_ids", []))
        # Normalize chunk boundaries
        cbs = row.get("chunk_boundaries", [])
        chunk_boundaries_col.append([
            {
                "start": int(cb.get("start", 0)),
                "end": int(cb.get("end", 0)),
                "kind": int(cb.get("kind", 0)),
                "dep_level": int(cb.get("dep_level", 0)),
                "name": str(cb.get("name", "")),
                "symbol_id": (
                    None if cb.get("symbol_id") is None else int(cb["symbol_id"])
                ),
            }
            for cb in cbs
        ])
        call_edges_col.append([
            {"from": int(e.get("from", 0)), "to": int(e.get("to", 0))}
            for e in row.get("call_edges", [])
        ])
        type_edges_col.append([
            {"from": int(e.get("from", 0)), "to": int(e.get("to", 0))}
            for e in row.get("type_edges", [])
        ])
        ast_depth_col.append(row.get("ast_depth", []))
        sibling_index_col.append(row.get("sibling_index", []))
        ast_node_type_col.append(row.get("ast_node_type", []))
        symbol_ids_col.append(row.get("symbol_ids", []))
        call_targets_col.append(row.get("call_targets", []))
        type_refs_col.append(row.get("type_refs", []))
        def_use_col.append(row.get("def_use", []))
        domain_kind_col.append(row.get("domain_kind"))
        domain_ids_col.append(row.get("domain_ids", []))
        domain_role_ids_col.append(row.get("domain_role_ids", []))
        domain_entity_ids_col.append(row.get("domain_entity_ids", []))
        domain_scope_ids_col.append(row.get("domain_scope_ids", []))
        domain_source_doc_ids_col.append(row.get("domain_source_doc_ids", []))
        domain_source_identity_ids_col.append(
            row.get("domain_source_identity_ids", [])
        )
        domain_confidence_ids_col.append(row.get("domain_confidence_ids", []))
        domain_edges_col.append(
            _normalize_char_domain_edges(
                canonical_edge_fields["domain_edges"], family="domain"
            )
        )
        build_edges_col.append(
            _normalize_char_domain_edges(
                canonical_edge_fields["build_edges"], family="build"
            )
        )
        shell_edges_col.append(
            _normalize_char_domain_edges(
                canonical_edge_fields["shell_edges"], family="shell"
            )
        )
        diagnostic_edges_col.append(
            _normalize_char_domain_edges(
                canonical_edge_fields["diagnostic_edges"], family="diagnostic"
            )
        )
        cross_domain_edges_col.append(
            _normalize_char_domain_edges(
                canonical_edge_fields["cross_domain_edges"], family="cross_domain"
            )
        )
        change_mask_pre_col.append(row.get("change_mask_pre", []))
        change_mask_post_col.append(row.get("change_mask_post", []))
        hunk_id_per_char_col.append(row.get("hunk_id_per_char", []))
        edit_op_per_char_col.append(row.get("edit_op_per_char", []))
        pi = row.get("platform_info")
        platform_info_col.append(json.dumps(pi) if pi else None)
        li = row.get("language_info")
        language_info_col.append(json.dumps(li) if li else None)
        bi = row.get("build_info")
        build_info_col.append(json.dumps(bi) if bi else None)
        raw_constituents = row.get("constituent_provenance")
        if raw_constituents is None and row.get("constituent_provenance_json"):
            try:
                raw_constituents = json.loads(row["constituent_provenance_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                raw_constituents = None
        normalized_constituents = []
        if isinstance(raw_constituents, list):
            for item in raw_constituents:
                if not isinstance(item, dict):
                    continue
                normalized_constituents.append(
                    {
                        "filepath": item.get("filepath"),
                        "language_info": (
                            json.dumps(item["language_info"])
                            if item.get("language_info") is not None
                            else None
                        ),
                        "build_info": (
                            json.dumps(item["build_info"])
                            if item.get("build_info") is not None
                            else None
                        ),
                    }
                )
        constituent_provenance_col.append(normalized_constituents)
        constituent_provenance_json_col.append(row.get("constituent_provenance_json"))
        repos.append(row.get(REPO_COLUMN, ""))
        filepaths.append(row.get(FILEPATH_COLUMN, ""))
        commits.append(row.get(COMMIT_HASH_COLUMN, row.get("commit", "")))
        timestamps.append(row.get(TIMESTAMP_COLUMN, ""))
        _pr = row.get(PR_NUMBER_COLUMN)
        pr_numbers.append(int(_pr) if _pr not in (None, "", 0) else None)
        _discussion = str(row.get("pr_discussion") or "").strip()
        has_pr_discussions.append(bool(_discussion))
        pr_discussion_chars.append(len(_discussion))
        pr_discussion_lines.append(
            0 if not _discussion else _discussion.count("\n") + 1
        )
        parent_hashes.append(row.get(PARENT_HASHES_COLUMN, []))
        parent_counts.append(row.get(PARENT_COUNT_COLUMN))
        is_merge_commits.append(row.get(IS_MERGE_COMMIT_COLUMN))
        author_timestamps.append(row.get(AUTHOR_TIMESTAMP_COLUMN))
        commit_timestamps.append(row.get(COMMIT_TIMESTAMP_COLUMN))
        repo_stable_ids.append(row.get(REPO_STABLE_ID_COLUMN))
        filepath_stable_ids.append(row.get(FILEPATH_STABLE_ID_COLUMN))
        file_local_commit_indices.append(row.get(FILE_LOCAL_COMMIT_INDEX_COLUMN))
        ambiguous_reconstruction.append(
            row.get(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN, False)
        )
        rename_ambiguity.append(row.get(HAS_RENAME_AMBIGUITY_COLUMN, False))
        token_ids_col.append(tokenized.get(TOKEN_IDS_COLUMN, []))
        ifim_instruction_token_ids_col.append(
            tokenized.get(IFIM_INSTRUCTION_TOKEN_IDS_COLUMN, [])
        )
        commit_msg_token_ids_col.append(
            tokenized.get(COMMIT_MSG_TOKEN_IDS_COLUMN, [])
        )
        pre_token_ids_col.append(tokenized.get(PRE_TOKEN_IDS_COLUMN, []))
        post_token_ids_col.append(tokenized.get(POST_TOKEN_IDS_COLUMN, []))
        diff_token_ids_col.append(tokenized.get(DIFF_TOKEN_IDS_COLUMN, []))
        platform_ids_col.append(tokenized.get(PLATFORM_IDS_COLUMN, []))
        token_structure_ids_col.append(tokenized.get(TOKEN_STRUCTURE_IDS_COLUMN, []))
        token_dep_levels_col.append(tokenized.get(TOKEN_DEP_LEVELS_COLUMN, []))
        token_ast_depth_col.append(tokenized.get(TOKEN_AST_DEPTH_COLUMN, []))
        token_sibling_index_col.append(tokenized.get(TOKEN_SIBLING_INDEX_COLUMN, []))
        token_ast_node_type_col.append(tokenized.get(TOKEN_AST_NODE_TYPE_COLUMN, []))
        token_symbol_ids_col.append(tokenized.get(TOKEN_SYMBOL_IDS_COLUMN, []))
        token_call_targets_col.append(tokenized.get(TOKEN_CALL_TARGETS_COLUMN, []))
        token_type_refs_col.append(tokenized.get(TOKEN_TYPE_REFS_COLUMN, []))
        token_def_use_col.append(tokenized.get(TOKEN_DEF_USE_COLUMN, []))
        token_domain_ids_col.append(tokenized.get(TOKEN_DOMAIN_IDS_COLUMN, []))
        token_role_ids_col.append(tokenized.get(TOKEN_ROLE_IDS_COLUMN, []))
        token_entity_ids_col.append(tokenized.get(TOKEN_ENTITY_IDS_COLUMN, []))
        token_scope_ids_col.append(tokenized.get(TOKEN_SCOPE_IDS_COLUMN, []))
        token_source_doc_ids_col.append(tokenized.get(TOKEN_SOURCE_DOC_IDS_COLUMN, []))
        token_source_identity_ids_col.append(
            tokenized.get(TOKEN_SOURCE_IDENTITY_IDS_COLUMN, [])
        )
        token_confidence_ids_col.append(tokenized.get(TOKEN_CONFIDENCE_IDS_COLUMN, []))
        token_change_mask_pre_col.append(
            tokenized.get(TOKEN_CHANGE_MASK_PRE_COLUMN, row.get(TOKEN_CHANGE_MASK_PRE_COLUMN, []))
        )
        token_change_mask_post_col.append(
            tokenized.get(TOKEN_CHANGE_MASK_POST_COLUMN, row.get(TOKEN_CHANGE_MASK_POST_COLUMN, []))
        )
        hunk_id_per_token_col.append(
            tokenized.get(HUNK_ID_PER_TOKEN_COLUMN, row.get(HUNK_ID_PER_TOKEN_COLUMN, []))
        )
        edit_op_per_token_col.append(
            tokenized.get(EDIT_OP_PER_TOKEN_COLUMN, row.get(EDIT_OP_PER_TOKEN_COLUMN, []))
        )
        token_chunk_starts_col.append(tokenized.get(TOKEN_CHUNK_STARTS_COLUMN, []))
        token_chunk_ends_col.append(tokenized.get(TOKEN_CHUNK_ENDS_COLUMN, []))
        token_chunk_kinds_col.append(tokenized.get(TOKEN_CHUNK_KINDS_COLUMN, []))
        token_chunk_dep_levels_col.append(
            tokenized.get(TOKEN_CHUNK_DEP_LEVELS_COLUMN, [])
        )
        changed_chunk_ids_col.append(
            tokenized.get(CHANGED_CHUNK_IDS_COLUMN, row.get(CHANGED_CHUNK_IDS_COLUMN, []))
        )
        changed_chunk_spans_col.append(
            tokenized.get(CHANGED_CHUNK_SPANS_COLUMN, row.get(CHANGED_CHUNK_SPANS_COLUMN, []))
        )
        token_call_edges_col.append(tokenized.get(TOKEN_CALL_EDGES_COLUMN, []))
        token_type_edges_col.append(tokenized.get(TOKEN_TYPE_EDGES_COLUMN, []))
        token_domain_edges_col.append(tokenized.get(TOKEN_DOMAIN_EDGES_COLUMN, []))
        token_build_edges_col.append(tokenized.get(TOKEN_BUILD_EDGES_COLUMN, []))
        token_shell_edges_col.append(tokenized.get(TOKEN_SHELL_EDGES_COLUMN, []))
        token_diagnostic_edges_col.append(tokenized.get(TOKEN_DIAGNOSTIC_EDGES_COLUMN, []))
        token_cross_domain_edges_col.append(tokenized.get(TOKEN_CROSS_DOMAIN_EDGES_COLUMN, []))
        source_identity_registry_col.append(
            tokenized.get(
                SOURCE_IDENTITY_REGISTRY_COLUMN,
                row.get(SOURCE_IDENTITY_REGISTRY_COLUMN, []),
            )
        )

    return pa.table(
        {
            "text": pa.array(texts, type=_SCHEMA.field("text").type),
            SYMBOL_IDENTITIES_COLUMN: pa.array(
                symbol_identities_col,
                type=_SCHEMA.field(SYMBOL_IDENTITIES_COLUMN).type,
            ),
            "source_text": pa.array(
                source_texts, type=_SCHEMA.field("source_text").type
            ),
            IFIM_INSTRUCTION_TEXT_COLUMN: pa.array(
                ifim_instruction_texts,
                type=_SCHEMA.field(IFIM_INSTRUCTION_TEXT_COLUMN).type,
            ),
            COMMIT_MSG_TEXT_COLUMN: pa.array(
                commit_msg_texts, type=_SCHEMA.field(COMMIT_MSG_TEXT_COLUMN).type
            ),
            PRE_TEXT_COLUMN: pa.array(
                pre_texts, type=_SCHEMA.field(PRE_TEXT_COLUMN).type
            ),
            POST_TEXT_COLUMN: pa.array(
                post_texts, type=_SCHEMA.field(POST_TEXT_COLUMN).type
            ),
            DIFF_TEXT_COLUMN: pa.array(
                diff_texts, type=_SCHEMA.field(DIFF_TEXT_COLUMN).type
            ),
            "source_doc_id": pa.array(
                source_doc_ids, type=_SCHEMA.field("source_doc_id").type
            ),
            DOC_TYPE_COLUMN: pa.array(
                doc_types, type=_SCHEMA.field(DOC_TYPE_COLUMN).type
            ),
            HEADER_FRAGMENT_KIND_COLUMN: pa.array(
                header_fragment_kinds,
                type=_SCHEMA.field(HEADER_FRAGMENT_KIND_COLUMN).type,
            ),
            BUILD_KIND_COLUMN: pa.array(
                build_kinds,
                type=_SCHEMA.field(BUILD_KIND_COLUMN).type,
            ),
            "tokenizer_fingerprint": pa.array(
                tokenizer_fingerprints,
                type=_SCHEMA.field("tokenizer_fingerprint").type,
            ),
            "actual_token_count": pa.array(
                token_counts, type=_SCHEMA.field("actual_token_count").type
            ),
            "structure_ids": pa.array(
                structure_ids_col, type=_SCHEMA.field("structure_ids").type
            ),
            "chunk_boundaries": pa.array(
                chunk_boundaries_col, type=_SCHEMA.field("chunk_boundaries").type
            ),
            "call_edges": pa.array(
                call_edges_col, type=_SCHEMA.field("call_edges").type
            ),
            "type_edges": pa.array(
                type_edges_col, type=_SCHEMA.field("type_edges").type
            ),
            "ast_depth": pa.array(
                ast_depth_col, type=_SCHEMA.field("ast_depth").type
            ),
            "sibling_index": pa.array(
                sibling_index_col, type=_SCHEMA.field("sibling_index").type
            ),
            "ast_node_type": pa.array(
                ast_node_type_col, type=_SCHEMA.field("ast_node_type").type
            ),
            "symbol_ids": pa.array(
                symbol_ids_col, type=_SCHEMA.field("symbol_ids").type
            ),
            "call_targets": pa.array(
                call_targets_col, type=_SCHEMA.field("call_targets").type
            ),
            "type_refs": pa.array(
                type_refs_col, type=_SCHEMA.field("type_refs").type
            ),
            "def_use": pa.array(def_use_col, type=_SCHEMA.field("def_use").type),
            "domain_kind": pa.array(
                domain_kind_col, type=_SCHEMA.field("domain_kind").type
            ),
            "domain_ids": pa.array(
                domain_ids_col, type=_SCHEMA.field("domain_ids").type
            ),
            "domain_role_ids": pa.array(
                domain_role_ids_col, type=_SCHEMA.field("domain_role_ids").type
            ),
            "domain_entity_ids": pa.array(
                domain_entity_ids_col, type=_SCHEMA.field("domain_entity_ids").type
            ),
            "domain_scope_ids": pa.array(
                domain_scope_ids_col, type=_SCHEMA.field("domain_scope_ids").type
            ),
            "domain_source_doc_ids": pa.array(
                domain_source_doc_ids_col,
                type=_SCHEMA.field("domain_source_doc_ids").type,
            ),
            "domain_source_identity_ids": pa.array(
                domain_source_identity_ids_col,
                type=_SCHEMA.field("domain_source_identity_ids").type,
            ),
            "domain_confidence_ids": pa.array(
                domain_confidence_ids_col,
                type=_SCHEMA.field("domain_confidence_ids").type,
            ),
            "domain_edges": pa.array(
                domain_edges_col, type=_SCHEMA.field("domain_edges").type
            ),
            "build_edges": pa.array(
                build_edges_col, type=_SCHEMA.field("build_edges").type
            ),
            "shell_edges": pa.array(
                shell_edges_col, type=_SCHEMA.field("shell_edges").type
            ),
            "diagnostic_edges": pa.array(
                diagnostic_edges_col, type=_SCHEMA.field("diagnostic_edges").type
            ),
            "cross_domain_edges": pa.array(
                cross_domain_edges_col,
                type=_SCHEMA.field("cross_domain_edges").type,
            ),
            "change_mask_pre": pa.array(
                change_mask_pre_col, type=_SCHEMA.field("change_mask_pre").type
            ),
            "change_mask_post": pa.array(
                change_mask_post_col, type=_SCHEMA.field("change_mask_post").type
            ),
            "hunk_id_per_char": pa.array(
                hunk_id_per_char_col, type=_SCHEMA.field("hunk_id_per_char").type
            ),
            "edit_op_per_char": pa.array(
                edit_op_per_char_col, type=_SCHEMA.field("edit_op_per_char").type
            ),
            "platform_info": pa.array(
                platform_info_col, type=_SCHEMA.field("platform_info").type
            ),
            "language_info": pa.array(
                language_info_col, type=_SCHEMA.field("language_info").type
            ),
            "build_info": pa.array(
                build_info_col, type=_SCHEMA.field("build_info").type
            ),
            "constituent_provenance": pa.array(
                constituent_provenance_col,
                type=_SCHEMA.field("constituent_provenance").type,
            ),
            "constituent_provenance_json": pa.array(
                constituent_provenance_json_col,
                type=_SCHEMA.field("constituent_provenance_json").type,
            ),
            REPO_COLUMN: pa.array(repos, type=_SCHEMA.field(REPO_COLUMN).type),
            FILEPATH_COLUMN: pa.array(filepaths, type=_SCHEMA.field(FILEPATH_COLUMN).type),
            COMMIT_HASH_COLUMN: pa.array(commits, type=_SCHEMA.field(COMMIT_HASH_COLUMN).type),
            TIMESTAMP_COLUMN: pa.array(timestamps, type=_SCHEMA.field(TIMESTAMP_COLUMN).type),
            PR_NUMBER_COLUMN: pa.array(pr_numbers, type=_SCHEMA.field(PR_NUMBER_COLUMN).type),
            HAS_PR_DISCUSSION_COLUMN: pa.array(
                has_pr_discussions,
                type=_SCHEMA.field(HAS_PR_DISCUSSION_COLUMN).type,
            ),
            PR_DISCUSSION_CHARS_COLUMN: pa.array(
                pr_discussion_chars,
                type=_SCHEMA.field(PR_DISCUSSION_CHARS_COLUMN).type,
            ),
            PR_DISCUSSION_LINES_COLUMN: pa.array(
                pr_discussion_lines,
                type=_SCHEMA.field(PR_DISCUSSION_LINES_COLUMN).type,
            ),
            PARENT_HASHES_COLUMN: pa.array(parent_hashes, type=_SCHEMA.field(PARENT_HASHES_COLUMN).type),
            PARENT_COUNT_COLUMN: pa.array(parent_counts, type=_SCHEMA.field(PARENT_COUNT_COLUMN).type),
            IS_MERGE_COMMIT_COLUMN: pa.array(is_merge_commits, type=_SCHEMA.field(IS_MERGE_COMMIT_COLUMN).type),
            AUTHOR_TIMESTAMP_COLUMN: pa.array(author_timestamps, type=_SCHEMA.field(AUTHOR_TIMESTAMP_COLUMN).type),
            COMMIT_TIMESTAMP_COLUMN: pa.array(commit_timestamps, type=_SCHEMA.field(COMMIT_TIMESTAMP_COLUMN).type),
            REPO_STABLE_ID_COLUMN: pa.array(repo_stable_ids, type=_SCHEMA.field(REPO_STABLE_ID_COLUMN).type),
            FILEPATH_STABLE_ID_COLUMN: pa.array(filepath_stable_ids, type=_SCHEMA.field(FILEPATH_STABLE_ID_COLUMN).type),
            FILE_LOCAL_COMMIT_INDEX_COLUMN: pa.array(
                file_local_commit_indices,
                type=_SCHEMA.field(FILE_LOCAL_COMMIT_INDEX_COLUMN).type,
            ),
            HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN: pa.array(
                ambiguous_reconstruction,
                type=_SCHEMA.field(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN).type,
            ),
            HAS_RENAME_AMBIGUITY_COLUMN: pa.array(
                rename_ambiguity,
                type=_SCHEMA.field(HAS_RENAME_AMBIGUITY_COLUMN).type,
            ),
            TOKEN_IDS_COLUMN: pa.array(
                token_ids_col, type=_SCHEMA.field(TOKEN_IDS_COLUMN).type
            ),
            IFIM_INSTRUCTION_TOKEN_IDS_COLUMN: pa.array(
                ifim_instruction_token_ids_col,
                type=_SCHEMA.field(IFIM_INSTRUCTION_TOKEN_IDS_COLUMN).type,
            ),
            COMMIT_MSG_TOKEN_IDS_COLUMN: pa.array(
                commit_msg_token_ids_col,
                type=_SCHEMA.field(COMMIT_MSG_TOKEN_IDS_COLUMN).type,
            ),
            PRE_TOKEN_IDS_COLUMN: pa.array(
                pre_token_ids_col, type=_SCHEMA.field(PRE_TOKEN_IDS_COLUMN).type
            ),
            POST_TOKEN_IDS_COLUMN: pa.array(
                post_token_ids_col, type=_SCHEMA.field(POST_TOKEN_IDS_COLUMN).type
            ),
            DIFF_TOKEN_IDS_COLUMN: pa.array(
                diff_token_ids_col, type=_SCHEMA.field(DIFF_TOKEN_IDS_COLUMN).type
            ),
            PLATFORM_IDS_COLUMN: pa.array(
                platform_ids_col, type=_SCHEMA.field(PLATFORM_IDS_COLUMN).type
            ),
            TOKEN_STRUCTURE_IDS_COLUMN: pa.array(
                token_structure_ids_col,
                type=_SCHEMA.field(TOKEN_STRUCTURE_IDS_COLUMN).type,
            ),
            TOKEN_DEP_LEVELS_COLUMN: pa.array(
                token_dep_levels_col, type=_SCHEMA.field(TOKEN_DEP_LEVELS_COLUMN).type
            ),
            TOKEN_AST_DEPTH_COLUMN: pa.array(
                token_ast_depth_col, type=_SCHEMA.field(TOKEN_AST_DEPTH_COLUMN).type
            ),
            TOKEN_SIBLING_INDEX_COLUMN: pa.array(
                token_sibling_index_col,
                type=_SCHEMA.field(TOKEN_SIBLING_INDEX_COLUMN).type,
            ),
            TOKEN_AST_NODE_TYPE_COLUMN: pa.array(
                token_ast_node_type_col,
                type=_SCHEMA.field(TOKEN_AST_NODE_TYPE_COLUMN).type,
            ),
            TOKEN_SYMBOL_IDS_COLUMN: pa.array(
                token_symbol_ids_col,
                type=_SCHEMA.field(TOKEN_SYMBOL_IDS_COLUMN).type,
            ),
            TOKEN_CALL_TARGETS_COLUMN: pa.array(
                token_call_targets_col,
                type=_SCHEMA.field(TOKEN_CALL_TARGETS_COLUMN).type,
            ),
            TOKEN_TYPE_REFS_COLUMN: pa.array(
                token_type_refs_col,
                type=_SCHEMA.field(TOKEN_TYPE_REFS_COLUMN).type,
            ),
            TOKEN_DEF_USE_COLUMN: pa.array(
                token_def_use_col,
                type=_SCHEMA.field(TOKEN_DEF_USE_COLUMN).type,
            ),
            TOKEN_DOMAIN_IDS_COLUMN: pa.array(
                token_domain_ids_col,
                type=_SCHEMA.field(TOKEN_DOMAIN_IDS_COLUMN).type,
            ),
            TOKEN_ROLE_IDS_COLUMN: pa.array(
                token_role_ids_col,
                type=_SCHEMA.field(TOKEN_ROLE_IDS_COLUMN).type,
            ),
            TOKEN_ENTITY_IDS_COLUMN: pa.array(
                token_entity_ids_col,
                type=_SCHEMA.field(TOKEN_ENTITY_IDS_COLUMN).type,
            ),
            TOKEN_SCOPE_IDS_COLUMN: pa.array(
                token_scope_ids_col,
                type=_SCHEMA.field(TOKEN_SCOPE_IDS_COLUMN).type,
            ),
            TOKEN_SOURCE_DOC_IDS_COLUMN: pa.array(
                token_source_doc_ids_col,
                type=_SCHEMA.field(TOKEN_SOURCE_DOC_IDS_COLUMN).type,
            ),
            TOKEN_SOURCE_IDENTITY_IDS_COLUMN: pa.array(
                token_source_identity_ids_col,
                type=_SCHEMA.field(TOKEN_SOURCE_IDENTITY_IDS_COLUMN).type,
            ),
            TOKEN_CONFIDENCE_IDS_COLUMN: pa.array(
                token_confidence_ids_col,
                type=_SCHEMA.field(TOKEN_CONFIDENCE_IDS_COLUMN).type,
            ),
            TOKEN_CHANGE_MASK_PRE_COLUMN: pa.array(
                token_change_mask_pre_col,
                type=_SCHEMA.field(TOKEN_CHANGE_MASK_PRE_COLUMN).type,
            ),
            TOKEN_CHANGE_MASK_POST_COLUMN: pa.array(
                token_change_mask_post_col,
                type=_SCHEMA.field(TOKEN_CHANGE_MASK_POST_COLUMN).type,
            ),
            HUNK_ID_PER_TOKEN_COLUMN: pa.array(
                hunk_id_per_token_col,
                type=_SCHEMA.field(HUNK_ID_PER_TOKEN_COLUMN).type,
            ),
            EDIT_OP_PER_TOKEN_COLUMN: pa.array(
                edit_op_per_token_col,
                type=_SCHEMA.field(EDIT_OP_PER_TOKEN_COLUMN).type,
            ),
            TOKEN_CHUNK_STARTS_COLUMN: pa.array(
                token_chunk_starts_col,
                type=_SCHEMA.field(TOKEN_CHUNK_STARTS_COLUMN).type,
            ),
            TOKEN_CHUNK_ENDS_COLUMN: pa.array(
                token_chunk_ends_col, type=_SCHEMA.field(TOKEN_CHUNK_ENDS_COLUMN).type
            ),
            TOKEN_CHUNK_KINDS_COLUMN: pa.array(
                token_chunk_kinds_col, type=_SCHEMA.field(TOKEN_CHUNK_KINDS_COLUMN).type
            ),
            TOKEN_CHUNK_DEP_LEVELS_COLUMN: pa.array(
                token_chunk_dep_levels_col,
                type=_SCHEMA.field(TOKEN_CHUNK_DEP_LEVELS_COLUMN).type,
            ),
            CHANGED_CHUNK_IDS_COLUMN: pa.array(
                changed_chunk_ids_col,
                type=_SCHEMA.field(CHANGED_CHUNK_IDS_COLUMN).type,
            ),
            CHANGED_CHUNK_SPANS_COLUMN: pa.array(
                changed_chunk_spans_col,
                type=_SCHEMA.field(CHANGED_CHUNK_SPANS_COLUMN).type,
            ),
            TOKEN_CALL_EDGES_COLUMN: pa.array(
                token_call_edges_col, type=_SCHEMA.field(TOKEN_CALL_EDGES_COLUMN).type
            ),
            TOKEN_TYPE_EDGES_COLUMN: pa.array(
                token_type_edges_col, type=_SCHEMA.field(TOKEN_TYPE_EDGES_COLUMN).type
            ),
            TOKEN_DOMAIN_EDGES_COLUMN: pa.array(
                token_domain_edges_col,
                type=_SCHEMA.field(TOKEN_DOMAIN_EDGES_COLUMN).type,
            ),
            TOKEN_BUILD_EDGES_COLUMN: pa.array(
                token_build_edges_col,
                type=_SCHEMA.field(TOKEN_BUILD_EDGES_COLUMN).type,
            ),
            TOKEN_SHELL_EDGES_COLUMN: pa.array(
                token_shell_edges_col,
                type=_SCHEMA.field(TOKEN_SHELL_EDGES_COLUMN).type,
            ),
            TOKEN_DIAGNOSTIC_EDGES_COLUMN: pa.array(
                token_diagnostic_edges_col,
                type=_SCHEMA.field(TOKEN_DIAGNOSTIC_EDGES_COLUMN).type,
            ),
            TOKEN_CROSS_DOMAIN_EDGES_COLUMN: pa.array(
                token_cross_domain_edges_col,
                type=_SCHEMA.field(TOKEN_CROSS_DOMAIN_EDGES_COLUMN).type,
            ),
            SOURCE_IDENTITY_REGISTRY_COLUMN: pa.array(
                source_identity_registry_col,
                type=_SCHEMA.field(SOURCE_IDENTITY_REGISTRY_COLUMN).type,
            ),
        },
        schema=_SCHEMA,
    )


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def gcs_list_files(prefix: str) -> list:
    """List GCS files under a prefix using gcloud storage."""
    uri = f"gs://{GCS_BUCKET}/{prefix}/"
    result = subprocess.run(
        ["gcloud", "storage", "ls", uri],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gcloud storage ls failed for {uri}: {result.stderr}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gcs_object_exists(uri: str) -> bool:
    """Return whether one exact GCS object exists, failing on access errors."""

    result = subprocess.run(
        ["gcloud", "storage", "ls", uri],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        return True
    detail = f"{result.stdout}\n{result.stderr}".lower()
    if "matched no objects" in detail or "no urls matched" in detail:
        return False
    raise RuntimeError(f"gcloud storage ls failed for {uri}: {result.stderr}")


def gcs_download(gcs_uri: str, local_path: str):
    """Download a GCS file to local path."""
    subprocess.run(
        ["gcloud", "storage", "cp", gcs_uri, local_path],
        check=True, timeout=600,
    )


def gcs_upload(local_path: str, gcs_uri: str):
    """Upload a local file to GCS."""
    subprocess.run(
        ["gcloud", "storage", "cp", local_path, gcs_uri],
        check=True, timeout=600,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

_PLATFORM_HEADER = build_platform_header(_DEFAULT_PLATFORM_HEADER_INFO)
_HEADER_LEN = len(_PLATFORM_HEADER)


def _align_structure_ids(values: list, text_len: int) -> list[int]:
    """Return per-char structure IDs aligned to the emitted text length."""
    if not values:
        return [0] * text_len
    if len(values) != text_len:
        raise ValueError(
            f"structure_ids length {len(values)} does not match text length {text_len}"
        )
    return [int(v) for v in values]


def _align_optional_char_metadata(values: list, text_len: int) -> list[int]:
    """Return per-char metadata aligned to `text_len`, preserving absence."""
    if not values:
        return []
    if len(values) != text_len:
        raise ValueError(
            f"optional char metadata length {len(values)} does not match text length {text_len}"
        )
    return [int(v) for v in values]


def _remap_structure_ids(values: list, kept_indices: list[int], original_text_len: int) -> list[int]:
    aligned = _align_structure_ids(values, original_text_len)
    return [aligned[i] if 0 <= i < len(aligned) else 0 for i in kept_indices]


def _remap_optional_char_metadata(
    values: list,
    kept_indices: list[int],
    original_text_len: int,
) -> list[int]:
    aligned = _align_optional_char_metadata(values, original_text_len)
    if not aligned:
        return []
    return [aligned[i] if 0 <= i < len(aligned) else 0 for i in kept_indices]


def _remap_chunk_boundaries(
    chunk_boundaries: list,
    kept_indices: list[int],
) -> tuple[list[dict], dict[int, int]]:
    remapped: list[dict] = []
    old_to_new: dict[int, int] = {}
    if not kept_indices:
        return remapped, old_to_new

    for old_idx, cb in enumerate(chunk_boundaries or []):
        if not isinstance(cb, dict):
            continue
        start = max(0, int(cb.get("start", 0)))
        end = max(start, int(cb.get("end", start)))
        new_start = bisect_left(kept_indices, start)
        new_end = bisect_left(kept_indices, end)
        if new_start >= new_end:
            continue
        old_to_new[old_idx] = len(remapped)
        remapped.append(
            {
                "start": new_start,
                "end": new_end,
                "kind": cb.get("kind", 0),
                "dep_level": cb.get("dep_level", 0),
                "name": cb.get("name", ""),
                "symbol_id": cb.get("symbol_id"),
            }
        )
    return remapped, old_to_new


def _remap_chunk_edges(raw_edges: list, old_to_new_chunk: dict[int, int]) -> list[dict]:
    remapped: list[dict] = []
    for edge in raw_edges or []:
        if not isinstance(edge, dict):
            continue
        old_from = int(edge.get("from", -1))
        old_to = int(edge.get("to", -1))
        if old_from not in old_to_new_chunk or old_to not in old_to_new_chunk:
            continue
        remapped.append(
            {
                "from": old_to_new_chunk[old_from],
                "to": old_to_new_chunk[old_to],
            }
        )
    return remapped


def _map_exact_kept_char(kept_indices: list[int], original_offset: int) -> int | None:
    pos = bisect_left(kept_indices, original_offset)
    if pos < len(kept_indices) and kept_indices[pos] == original_offset:
        return pos
    return None


def _shift_char_edge_triples(
    raw_edges: list,
    header_len: int,
    *,
    family: str,
) -> list[dict]:
    shifted = []
    for src, dst, kind in (
        normalize_domain_edge_record(edge, family=family)
        for edge in raw_edges or []
    ):
        shifted.append(
            {
                "from_char": src + header_len,
                "to_char": dst + header_len,
                "kind": kind,
            }
        )
    return shifted


def _remap_char_edge_triples(
    raw_edges: list,
    kept_indices: list[int],
    header_len: int,
    *,
    family: str,
    source_length: int,
) -> list[dict]:
    remapped = []
    for src, dst, kind in (
        normalize_domain_edge_record(edge, family=family)
        for edge in raw_edges or []
    ):
        if src == dst:
            if src > source_length:
                raise ValueError(
                    f"character point {src} is outside source point bounds "
                    f"[0, {source_length}]"
                )
            mapped_from = mapped_to = bisect_left(kept_indices, src)
        else:
            mapped_from = _map_exact_kept_char(
                kept_indices,
                src,
            )
            mapped_to = _map_exact_kept_char(
                kept_indices,
                dst,
            )
        if mapped_from is None or mapped_to is None:
            continue
        remapped.append(
            {
                "from_char": mapped_from + header_len,
                "to_char": mapped_to + header_len,
                "kind": kind,
            }
        )
    return remapped


def process_record(record: dict, tokenizer, max_tokens: int) -> list:
    """Apply full processing pipeline to one enriched JSONL record."""
    return process_record_with_policy(
        record,
        tokenizer,
        max_tokens,
        overflow_policy="split",
    )


def process_record_with_policy(
    record: dict,
    tokenizer,
    max_tokens: int,
    *,
    overflow_policy: str = "split",
) -> list[dict]:
    """Apply full processing pipeline with explicit handling for oversized docs."""
    if overflow_policy not in _OVERFLOW_POLICIES:
        raise ValueError(
            f"Unsupported overflow_policy={overflow_policy!r}; "
            f"expected one of {_OVERFLOW_POLICIES}"
        )

    text = record.get("text", "")
    record = {
        **record,
        **canonicalize_domain_edge_fields(record, source_length=len(text)),
    }
    structure_ids = record.get("structure_ids", [])
    chunk_boundaries = record.get("chunk_boundaries", [])

    # 1. Dead-platform filter
    filtered_text, kept_indices = filter_dead_platforms_with_mapping(text)
    metadata_stale = kept_indices is not None

    # 2. Prepend metadata headers; adjust structure_ids + chunk offsets
    language_prefix = language_info_to_prefix(record.get("language_info"))
    header_prefix = language_prefix + _PLATFORM_HEADER
    header_len = len(header_prefix)
    full_text = header_prefix + filtered_text
    if metadata_stale:
        filtered_structure_ids = _remap_structure_ids(
            structure_ids,
            kept_indices,
            len(text),
        )
        filtered_chunk_boundaries, old_to_new_chunk = _remap_chunk_boundaries(
            chunk_boundaries,
            kept_indices,
        )
        filtered_call_edges = _remap_chunk_edges(
            record.get("call_edges", []),
            old_to_new_chunk,
        )
        filtered_type_edges = _remap_chunk_edges(
            record.get("type_edges", []),
            old_to_new_chunk,
        )
        filtered_domain_edge_fields = {
            name: _remap_char_edge_triples(
                record.get(name, []),
                kept_indices,
                header_len,
                family=family,
                source_length=len(text),
            )
            for name, family in DOMAIN_EDGE_FIELD_FAMILIES.items()
        }
        filtered_embedded_domain_spans = remap_embedded_domain_spans(
            record.get("embedded_domain_spans", []),
            source_length=len(text),
            kept_indices=kept_indices,
            prefix_length=header_len,
        )
    else:
        filtered_structure_ids = _align_structure_ids(structure_ids, len(filtered_text))
        filtered_chunk_boundaries = chunk_boundaries
        filtered_call_edges = record.get("call_edges", [])
        filtered_type_edges = record.get("type_edges", [])

        filtered_domain_edge_fields = {
            name: _shift_char_edge_triples(
                record.get(name, []), header_len, family=family
            )
            for name, family in DOMAIN_EDGE_FIELD_FAMILIES.items()
        }
        filtered_embedded_domain_spans = remap_embedded_domain_spans(
            record.get("embedded_domain_spans", []),
            source_length=len(text),
            kept_indices=None,
            prefix_length=header_len,
        )

    full_sids = [0] * header_len + filtered_structure_ids
    source_identity_registry = [
        dict(entry) for entry in record.get(SOURCE_IDENTITY_REGISTRY_COLUMN, [])
    ]
    if source_identity_registry:
        validated_registry = validate_source_identity_registry(source_identity_registry)
        source_identity_id = next(iter(validated_registry))
    else:
        identity = source_identity(record)
        source_identity_registry = [identity.as_dict()]
        source_identity_id = identity.source_identity_id
    char_metadata: dict[str, list[int]] = {}
    for key in _CHAR_LEVEL_METADATA_FIELDS:
        values = record.get(key, [])
        if metadata_stale:
            remapped = _remap_optional_char_metadata(values, kept_indices, len(text))
            if key == "domain_source_doc_ids":
                remapped = normalize_row_local_doc_ids(
                    remapped,
                    length=len(filtered_text),
                    fallback_doc_id=1,
                )
                char_metadata[key] = [remapped[0] if remapped else 1] * header_len + remapped
            elif key == "domain_source_identity_ids":
                if len(source_identity_registry) > 1 and (
                    not remapped or any(int(value) == 0 for value in remapped)
                ):
                    raise ValueError(
                        "multi-source document lost exact character source identities"
                    )
                remapped = normalize_positive_source_ids(
                    remapped,
                    length=len(filtered_text),
                    fallback_source_id=source_identity_id,
                )
                char_metadata[key] = [source_identity_id] * header_len + remapped
            else:
                char_metadata[key] = [0] * header_len + remapped if remapped else []
        else:
            aligned = _align_optional_char_metadata(values, len(filtered_text))
            if key == "domain_source_doc_ids":
                aligned = normalize_row_local_doc_ids(
                    aligned,
                    length=len(filtered_text),
                    fallback_doc_id=1,
                )
                char_metadata[key] = [aligned[0] if aligned else 1] * header_len + aligned
            elif key == "domain_source_identity_ids":
                if len(source_identity_registry) > 1 and (
                    not aligned or any(int(value) == 0 for value in aligned)
                ):
                    raise ValueError(
                        "multi-source document is missing exact character source identities"
                    )
                aligned = normalize_positive_source_ids(
                    aligned,
                    length=len(filtered_text),
                    fallback_source_id=source_identity_id,
                )
                char_metadata[key] = [source_identity_id] * header_len + aligned
            else:
                char_metadata[key] = [0] * header_len + aligned if aligned else []

    adjusted_chunks = [
        {
            "start": cb.get("start", 0) + header_len,
            "end": cb.get("end", 0) + header_len,
            "kind": cb.get("kind", 0),
            "dep_level": cb.get("dep_level", 0),
            "name": cb.get("name", ""),
            "symbol_id": cb.get("symbol_id"),
        }
        for cb in filtered_chunk_boundaries
    ]
    validate_source_identity_registry(
        source_identity_registry,
        referenced_ids=char_metadata.get("domain_source_identity_ids", []),
    )

    combined = {
        **{k: v for k, v in record.items()
           if k not in ("text", "structure_ids", "chunk_boundaries",
                        "call_edges", "type_edges", "actual_token_count",
                        "domain_edges", "build_edges", "shell_edges",
                        "diagnostic_edges", "cross_domain_edges",
                        "embedded_domain_spans",
                        *_CHAR_LEVEL_METADATA_FIELDS)},
        "text": full_text,
        # source_text mirrors the emitted (header-prefixed) document text so the
        # IFIM objective, which reads metadata['source_text'], aligns with the
        # tokenized text exactly.
        "source_text": full_text,
        "source_identity_id": source_identity_id,
        SOURCE_IDENTITY_REGISTRY_COLUMN: source_identity_registry,
        "structure_ids": full_sids,
        "chunk_boundaries": adjusted_chunks,
        "call_edges": filtered_call_edges,
        "type_edges": filtered_type_edges,
        "embedded_domain_spans": filtered_embedded_domain_spans,
        **filtered_domain_edge_fields,
        **char_metadata,
    }
    if not combined.get("platform_info"):
        combined["platform_info"] = dict(_DEFAULT_PLATFORM_INFO)

    if overflow_policy == "drop":
        docs = maybe_keep_document_exact(combined, tokenizer, max_tokens=max_tokens)
        if not docs:
            log.info(
                "drop overflow record repo=%s filepath=%s actual_token_count>%d",
                record.get("repo", ""),
                record.get("filepath", ""),
                max_tokens,
            )
        return docs

    return chunk_document_exact(combined, tokenizer, max_tokens=max_tokens)


def _open_jsonl(path: str | os.PathLike[str]):
    target = Path(path)
    if target.suffix == ".gz":
        return gzip.open(target, "rt", encoding="utf-8", errors="replace")
    return open(target, "r", encoding="utf-8", errors="replace")


def _convert_local_jsonl_to_parquet_path(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    tokenizer,
    max_tokens: int,
    overflow_policy: str = "split",
    dry_run: bool = False,
    materialize_tokenized_enriched: bool = False,
    local_batch_size: int = 512,
    memory_limit_gb: float = 10.0,
    default_repo: str | None = None,
) -> dict[str, object]:
    """Convert a local clang JSONL/JSONL.GZ file into one parquet file."""
    if overflow_policy not in _OVERFLOW_POLICIES:
        raise ValueError(
            f"Unsupported overflow_policy={overflow_policy!r}; "
            f"expected one of {_OVERFLOW_POLICIES}"
        )

    source = Path(input_path)
    target = Path(output_path)
    stats: dict[str, object] = {
        "schema": _MATERIALIZE_SPLIT_STATS_SCHEMA,
        "overflow_policy": overflow_policy,
        "max_tokens": int(max_tokens),
        "docs_in": 0,
        "docs_out": 0,
        "source_docs_emitted": 0,
        "split_input_docs": 0,
        "split_output_docs": 0,
        "dropped_input_docs": 0,
        "emitted_valid_tokens": 0,
        "split_emitted_tokens": 0,
        "emitted_chars": 0,
        "split_emitted_chars": 0,
        "max_emitted_tokens": 0,
        "max_emitted_chars": 0,
        "materialized_rows": 0,
        "materialized_output_tokens": 0,
        "max_materialized_tokens": 0,
        **{
            f"cross_piece_{edge_field}": 0
            for edge_field in _CROSS_PIECE_EDGE_FIELDS
        },
    }
    rows: list[dict] = []
    wrote_rows = False
    active_tokenizer_fingerprint = tokenizer_fingerprint(tokenizer)
    identity_registry = SymbolIdentityRegistry()

    def flush_rows() -> None:
        nonlocal rows, wrote_rows, writer
        if not rows:
            return
        if dry_run:
            rows = []
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tokenized_rows = (
            materialize_tokenized_enriched_batch(rows, tokenizer)
            if materialize_tokenized_enriched
            else None
        )
        if tokenized_rows is not None:
            materialized_lengths = [
                len(row.get(TOKEN_IDS_COLUMN, [])) for row in tokenized_rows
            ]
            stats["materialized_rows"] = int(stats["materialized_rows"]) + len(
                materialized_lengths
            )
            stats["materialized_output_tokens"] = int(
                stats["materialized_output_tokens"]
            ) + sum(materialized_lengths)
            stats["max_materialized_tokens"] = max(
                int(stats["max_materialized_tokens"]),
                max(materialized_lengths, default=0),
            )
        table = rows_to_table(
            rows,
            tokenized_rows=tokenized_rows,
            identity_registry=identity_registry,
        )
        if writer is None:
            writer = pq.ParquetWriter(target, _SCHEMA, compression="snappy")
        writer.write_table(table)
        wrote_rows = True
        rows = []
        check_memory_limit(memory_limit_gb, label="clang_enriched_to_parquet")

    writer = None
    try:
        with _open_jsonl(source) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = _with_default_static_provenance(
                    json.loads(line),
                    default_repo=default_repo,
                )
                stats["docs_in"] = int(stats["docs_in"]) + 1
                sub_docs = process_record_with_policy(
                    record,
                    tokenizer,
                    max_tokens,
                    overflow_policy=overflow_policy,
                )
                # V7-G02: prefer the helper's stable signature so the
                # same logical document gets the same id even if
                # shards are reordered later. Fall back to the legacy
                # name:index when the helper rejects the row.
                source_doc_id = record.get("source_doc_id")
                if not source_doc_id:
                    try:
                        from cppmega.data.nanochat_pipeline.document_identity import (
                            stable_doc_signature)
                        sig = stable_doc_signature(record)
                        source_doc_id = sig or f"{source.name}:{stats['docs_in']}"
                    except SymbolIdentityError:
                        raise
                    except Exception:
                        source_doc_id = f"{source.name}:{stats['docs_in']}"
                if not sub_docs:
                    stats["dropped_input_docs"] = int(
                        stats["dropped_input_docs"]
                    ) + 1
                    if overflow_policy == "split":
                        raise ValueError(
                            "lossless split emitted zero pieces for "
                            f"repo={record.get('repo')!r} "
                            f"filepath={record.get('filepath')!r}"
                        )
                else:
                    stats["source_docs_emitted"] = int(
                        stats["source_docs_emitted"]
                    ) + 1
                is_split = len(sub_docs) > 1
                if is_split:
                    stats["split_input_docs"] = int(stats["split_input_docs"]) + 1
                    stats["split_output_docs"] = int(
                        stats["split_output_docs"]
                    ) + len(sub_docs)
                for sub_doc in sub_docs:
                    sub_doc.setdefault("source_doc_id", source_doc_id)
                    sub_doc.setdefault(
                        "tokenizer_fingerprint",
                        active_tokenizer_fingerprint,
                    )
                    emitted_tokens = int(sub_doc.get("actual_token_count", 0))
                    emitted_chars = len(str(sub_doc.get("text", "")))
                    if emitted_tokens > max_tokens:
                        raise ValueError(
                            "split emitted an over-budget document: "
                            f"{emitted_tokens} > {max_tokens}"
                        )
                    stats["emitted_valid_tokens"] = int(
                        stats["emitted_valid_tokens"]
                    ) + emitted_tokens
                    stats["emitted_chars"] = int(stats["emitted_chars"]) + emitted_chars
                    stats["max_emitted_tokens"] = max(
                        int(stats["max_emitted_tokens"]), emitted_tokens
                    )
                    stats["max_emitted_chars"] = max(
                        int(stats["max_emitted_chars"]), emitted_chars
                    )
                    if is_split:
                        stats["split_emitted_tokens"] = int(
                            stats["split_emitted_tokens"]
                        ) + emitted_tokens
                        stats["split_emitted_chars"] = int(
                            stats["split_emitted_chars"]
                        ) + emitted_chars
                    split_audit = sub_doc.get("_lossless_split_audit", {})
                    cross_piece_edges = (
                        split_audit.get("cross_piece_edges", {})
                        if isinstance(split_audit, dict)
                        else {}
                    )
                    if not isinstance(cross_piece_edges, dict):
                        raise ValueError(
                            "_lossless_split_audit.cross_piece_edges must be an object"
                        )
                    for edge_field in _CROSS_PIECE_EDGE_FIELDS:
                        counter_name = f"cross_piece_{edge_field}"
                        edge_count = int(cross_piece_edges.get(edge_field, 0))
                        if edge_count < 0:
                            raise ValueError(
                                f"{counter_name} must be non-negative, got {edge_count}"
                            )
                        stats[counter_name] = int(stats[counter_name]) + edge_count
                rows.extend(sub_docs)
                stats["docs_out"] = int(stats["docs_out"]) + len(sub_docs)
                if len(rows) >= local_batch_size:
                    flush_rows()
                if int(stats["docs_in"]) % 1000 == 0:
                    check_memory_limit(memory_limit_gb, label="clang_enriched_to_parquet")
        flush_rows()
    finally:
        if writer is not None:
            writer.close()

    if dry_run:
        log.info(
            "[DRY RUN] local convert %s -> %s docs_in=%d docs_out=%d policy=%s",
            source,
            target,
            stats["docs_in"],
            stats["docs_out"],
            overflow_policy,
        )
        return stats

    if not wrote_rows:
        target.unlink(missing_ok=True)
        log.info("empty parquet")
        return stats
    log.info(
        "wrote local parquet %s docs_in=%d docs_out=%d policy=%s",
        target,
        stats["docs_in"],
        stats["docs_out"],
        overflow_policy,
    )
    return stats


def _write_materialize_stats(
    stats_file: str | os.PathLike[str],
    stats: dict[str, object],
) -> None:
    target = Path(stats_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        staged.write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_local_jsonl_to_parquet(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    tokenizer,
    max_tokens: int,
    overflow_policy: str = "split",
    dry_run: bool = False,
    materialize_tokenized_enriched: bool = False,
    local_batch_size: int = 512,
    memory_limit_gb: float = 10.0,
    default_repo: str | None = None,
    stats_file: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Convert locally, publishing the parquet only after full validation."""

    if default_repo is not None:
        default_repo = require_project_identity(
            default_repo, source="clang_enriched_to_parquet --default-repo"
        )
    kwargs = {
        "tokenizer": tokenizer,
        "max_tokens": max_tokens,
        "overflow_policy": overflow_policy,
        "dry_run": dry_run,
        "materialize_tokenized_enriched": materialize_tokenized_enriched,
        "local_batch_size": local_batch_size,
        "memory_limit_gb": memory_limit_gb,
        "default_repo": default_repo,
    }
    if dry_run:
        if stats_file is not None:
            raise ValueError("stats_file is not supported with dry_run")
        return _convert_local_jsonl_to_parquet_path(
            input_path, output_path, **kwargs
        )
    with atomic_output_file(output_path) as staged_output:
        summary = _convert_local_jsonl_to_parquet_path(
            input_path, staged_output, **kwargs
        )
    if stats_file is not None:
        published = Path(output_path)
        summary["output_file_size"] = published.stat().st_size
        summary["output_file_sha256"] = _sha256_file(published)
        _write_materialize_stats(stats_file, summary)
    return summary


def process_input_file(gcs_uri: str, tmpdir: str,
                       dry_run: bool, shard_size: int,
                       shard_counter: list, output_rows: list,
                       output_prefix_local: str,
                       output_prefix_gcs: str,
                       tokenizer,
                       max_tokens: int,
                       overflow_policy: str = "split",
                       identity_registry: SymbolIdentityRegistry | None = None) -> tuple:
    """Download, decompress, and process one .jsonl.gz file.

    Returns (docs_in, docs_out) counts.
    """
    fname = gcs_uri.split("/")[-1]
    local_gz = os.path.join(tmpdir, fname)

    log.info("Downloading %s ...", gcs_uri)
    if not dry_run:
        gcs_download(gcs_uri, local_gz)
    else:
        log.info("[DRY RUN] Would download %s", gcs_uri)
        return 0, 0

    docs_in = 0
    docs_out = 0
    active_tokenizer_fingerprint = tokenizer_fingerprint(tokenizer)

    with gzip.open(local_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = _with_default_static_provenance(
                    json.loads(line),
                    default_repo=None,
                )
            except json.JSONDecodeError as e:
                log.warning("JSON decode error in %s: %s", fname, e)
                continue

            docs_in += 1
            if overflow_policy == "split":
                sub_docs = process_record(record, tokenizer, max_tokens)
            else:
                sub_docs = process_record_with_policy(
                    record,
                    tokenizer,
                    max_tokens,
                    overflow_policy=overflow_policy,
                )
            # V7-G02: same fallback as the bucket path above — use the
            # stable doc signature so cross-shard duplicates collapse.
            source_doc_id = record.get("source_doc_id")
            if not source_doc_id:
                try:
                    from cppmega.data.nanochat_pipeline.document_identity import (
                        stable_doc_signature)
                    sig = stable_doc_signature(record)
                    source_doc_id = sig or f"{fname}:{docs_in}"
                except SymbolIdentityError:
                    raise
                except Exception:
                    source_doc_id = f"{fname}:{docs_in}"
            for sub_doc in sub_docs:
                sub_doc.setdefault("source_doc_id", source_doc_id)
                sub_doc.setdefault(
                    "tokenizer_fingerprint",
                    active_tokenizer_fingerprint,
                )
            output_rows.extend(sub_docs)
            docs_out += len(sub_docs)
            if docs_in % 1000 == 0:
                check_memory_limit(_MEMORY_LIMIT_GB, label="clang_enriched_to_parquet")

            # Flush shard when we hit shard_size
            while len(output_rows) >= shard_size:
                shard_rows = output_rows[:shard_size]
                output_rows[:] = output_rows[shard_size:]
                _flush_shard(
                    shard_rows,
                    shard_counter,
                    dry_run,
                    output_prefix_local,
                    output_prefix_gcs,
                    identity_registry=identity_registry,
                )

    os.unlink(local_gz)
    return docs_in, docs_out


def _flush_shard(
    rows: list,
    counter: list,
    dry_run: bool,
    output_prefix_local: str,
    output_prefix_gcs: str,
    *,
    identity_registry: SymbolIdentityRegistry | None = None,
):
    """Write rows to a parquet file and upload to GCS."""
    shard_idx = counter[0]
    counter[0] += 1
    fname = f"train_{shard_idx:05d}.parquet"
    local_path = os.path.join(output_prefix_local, fname)
    gcs_uri = f"gs://{GCS_BUCKET}/{output_prefix_gcs}/{fname}"

    log.info("Writing shard %d (%d rows) -> %s", shard_idx, len(rows), gcs_uri)
    if dry_run:
        log.info("[DRY RUN] Would write %d rows to %s", len(rows), gcs_uri)
        return

    table = rows_to_table(
        rows,
        tokenized_rows=(
            materialize_tokenized_enriched_batch(rows, _TOKENIZED_ENRICHED_TOKENIZER)
            if _TOKENIZED_ENRICHED_TOKENIZER is not None
            else None
        ),
        identity_registry=identity_registry,
    )
    check_memory_limit(_MEMORY_LIMIT_GB, label="clang_enriched_to_parquet")
    pq.write_table(table, local_path, compression="snappy")
    check_memory_limit(_MEMORY_LIMIT_GB, label="clang_enriched_to_parquet")
    # V7-G04: emit a sidecar corpus-stats JSON next to the parquet shard
    # so the UI DataInspector can render token coverage / doc-length
    # percentiles / vocab usage without rescanning the shard. The helper
    # is no-op safe when token_ids are absent (token_id_lists empty).
    try:
        from cppmega.data.nanochat_pipeline.corpus_stats import compute_corpus_stats
        import json as _json
        token_lists: list[list[int]] = []
        try:
            for col_token_ids in table.column(TOKEN_IDS_COLUMN).to_pylist():
                if col_token_ids:
                    token_lists.append(list(col_token_ids))
        except SymbolIdentityError:
            raise
        except Exception:
            token_lists = []
        if token_lists and _TOKENIZED_ENRICHED_TOKENIZER is not None:
            stats = compute_corpus_stats(
                token_lists,
                vocab_size=int(
                    _TOKENIZED_ENRICHED_TOKENIZER.get_vocab_size()),
            )
            stats_path = local_path + ".corpus_stats.json"
            with open(stats_path, "w") as _f:
                _json.dump(stats, _f)
            try:
                gcs_upload(stats_path, gcs_uri + ".corpus_stats.json")
            except SymbolIdentityError:
                raise
            except Exception:
                pass
            try:
                os.unlink(stats_path)
            except SymbolIdentityError:
                raise
            except Exception:
                pass
    except SymbolIdentityError:
        raise
    except Exception as _exc:
        log.warning("corpus_stats emit failed: %s", _exc)
    gcs_upload(local_path, gcs_uri)
    os.unlink(local_path)
    log.info("Uploaded shard %d (%d rows, %.1f MB)",
             shard_idx, len(rows),
             os.path.getsize(local_path) / 1e6 if os.path.exists(local_path) else 0)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Clang-enriched v5 JSONL.GZ files to hard-budgeted parquet shards.\n\n"
            "Reads from: gs://nanochat-training-data-2026/v5_enriched/*.jsonl.gz\n"
            "Writes to:  gs://nanochat-training-data-2026/data/parquet/clang_enriched_<size>/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--size",
        type=str,
        default="4k",
        help="Hard token budget label, e.g. 4k or 8k (default: 4k).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Download and process files but do not write or upload parquet.",
    )
    parser.add_argument(
        "--shard-size", type=int, default=10000,
        help="Number of chunked sub-documents per parquet shard (default: 10000).",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=512,
        help="Rows per local parquet row-group write in --input-file mode (default: 512).",
    )
    parser.add_argument(
        "--input-prefix", default=GCS_INPUT_PREFIX,
        help=f"GCS prefix to read JSONL.GZ from (default: {GCS_INPUT_PREFIX}).",
    )
    parser.add_argument(
        "--input-file",
        default="",
        help="Local JSONL or JSONL.GZ input for single-file conversion mode.",
    )
    parser.add_argument(
        "--output-prefix", default="",
        help="GCS prefix to write parquet shards to. "
        "Defaults to data/parquet/clang_enriched_<size>.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Local parquet output path for single-file conversion mode.",
    )
    parser.add_argument(
        "--stats-file",
        default="",
        help=(
            "Atomically write an exact local split/materialization receipt. "
            "Supported only with --input-file/--output-file."
        ),
    )
    parser.add_argument(
        "--max-files", type=int, default=0,
        help="Maximum number of input files to process (0 = all).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer.json used for exact token budgeting.",
    )
    parser.add_argument(
        "--materialize-tokenized-enriched",
        action="store_true",
        help="Also emit token_ids and token-level enriched metadata columns.",
    )
    parser.add_argument(
        "--overflow-policy",
        choices=_OVERFLOW_POLICIES,
        default="split",
        help="How to handle docs that exceed the exact budget after metadata prefixes "
            "are applied. 'split' preserves legacy behavior; 'drop' is strict no-crop.",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=10.0,
        help="Abort if this Python wrapper exceeds this max RSS in GiB (default: 10).",
    )
    parser.add_argument(
        "--default-repo",
        default="",
        help=(
            "Backfill repo/repo_stable_id/filepath_stable_id for local static-code "
            "JSONL rows that do not carry repo provenance."
        ),
    )
    args = parser.parse_args()
    start_memory_guard(args.memory_limit_gb, label="clang_enriched_to_parquet")

    size_label = args.size.lower()
    max_tokens = size_label_to_tokens(size_label)
    tokenizer = load_tokenizer(args.tokenizer_path)
    global _TOKENIZED_ENRICHED_TOKENIZER
    global _MEMORY_LIMIT_GB
    _TOKENIZED_ENRICHED_TOKENIZER = (
        tokenizer if args.materialize_tokenized_enriched else None
    )
    _MEMORY_LIMIT_GB = args.memory_limit_gb
    default_output_prefix = GCS_OUTPUT_PREFIX_TEMPLATE.format(size=size_label)
    local_mode = bool(args.input_file or args.output_file)
    if local_mode and not (args.input_file and args.output_file):
        parser.error("--input-file and --output-file must be provided together")
    if local_mode and args.max_files:
        parser.error("--max-files is only supported in GCS mode")
    if local_mode and args.input_prefix != GCS_INPUT_PREFIX:
        parser.error("--input-prefix is only supported in GCS mode")
    if local_mode and args.output_prefix:
        parser.error("--output-prefix is only supported in GCS mode")
    if args.stats_file and not local_mode:
        parser.error("--stats-file is only supported in local mode")
    if args.stats_file and args.dry_run:
        parser.error("--stats-file cannot be combined with --dry-run")
    if args.local_batch_size <= 0:
        parser.error("--local-batch-size must be positive")
    if not local_mode and not args.output_prefix:
        args.output_prefix = default_output_prefix

    if local_mode:
        summary = convert_local_jsonl_to_parquet(
            args.input_file,
            args.output_file,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overflow_policy=args.overflow_policy,
            dry_run=args.dry_run,
            materialize_tokenized_enriched=args.materialize_tokenized_enriched,
            local_batch_size=args.local_batch_size,
            memory_limit_gb=args.memory_limit_gb,
            default_repo=args.default_repo or None,
            stats_file=args.stats_file or None,
        )
        log.info(
            "Done. Local convert: %d docs in -> %d docs out.",
            summary["docs_in"],
            summary["docs_out"],
        )
        return

    log.info("Listing input files at gs://%s/%s/", GCS_BUCKET, args.input_prefix)
    input_files = gcs_list_files(args.input_prefix)
    all_gz_files = [f for f in input_files if f.endswith(".jsonl.gz")]
    gz_files = list(all_gz_files)

    if not gz_files:
        log.error("No .jsonl.gz files found at gs://%s/%s/",
                  GCS_BUCKET, args.input_prefix)
        sys.exit(1)

    if args.max_files > 0:
        gz_files = gz_files[:args.max_files]
    full_input = len(gz_files) == len(all_gz_files)

    log.info(
        "Found %d input files. Processing with hard_budget=%d tokens, shard_size=%d, dry_run=%s",
        len(gz_files),
        max_tokens,
        args.shard_size,
        args.dry_run,
    )

    total_in = 0
    total_out = 0
    shard_counter = [0]  # mutable counter shared across calls
    output_rows = []
    identity_registry = SymbolIdentityRegistry()
    staging_prefix = (
        f"{args.output_prefix}/.staging-{uuid.uuid4().hex}"
        if not args.dry_run
        else args.output_prefix
    )

    with tempfile.TemporaryDirectory(prefix=f"clang_{size_label}_") as tmpdir:
        # Local dir for shard staging
        output_prefix_local = os.path.join(tmpdir, "shards")
        os.makedirs(output_prefix_local, exist_ok=True)

        for gcs_uri in gz_files:
            try:
                docs_in, docs_out = process_input_file(
                    gcs_uri, tmpdir, args.dry_run,
                    args.shard_size, shard_counter, output_rows,
                    output_prefix_local,
                    staging_prefix,
                    tokenizer,
                    max_tokens,
                    args.overflow_policy,
                    identity_registry,
                )
                total_in += docs_in
                total_out += docs_out
                log.info("  %s: %d in -> %d out (cumulative: %d in, %d out, %d shards)",
                         gcs_uri.split("/")[-1], docs_in, docs_out,
                         total_in, total_out, shard_counter[0])
            except SymbolIdentityError:
                raise
            except Exception as e:
                raise RuntimeError(f"Failed to process {gcs_uri}: {e}") from e

        # Flush remaining rows
        if output_rows:
            _flush_shard(
                output_rows,
                shard_counter,
                args.dry_run,
                output_prefix_local,
                staging_prefix,
                identity_registry=identity_registry,
            )
            output_rows.clear()

    log.info("Done. Total: %d docs in -> %d chunks out, %d shards written.",
             total_in, total_out, shard_counter[0])

    if not args.dry_run:
        staged_root = f"gs://{GCS_BUCKET}/{staging_prefix}/"
        final_root = f"gs://{GCS_BUCKET}/{args.output_prefix}/"
        sentinel_uri = final_root + "_COMPLETE"
        try:
            if gcs_object_exists(sentinel_uri):
                subprocess.run(
                    ["gcloud", "storage", "rm", sentinel_uri],
                    check=True,
                    timeout=600,
                )
            staged_files = gcs_list_files(staging_prefix)
            if not staged_files:
                raise RuntimeError(f"no staged parquet outputs under {staged_root}")
            for staged_uri in staged_files:
                relative = staged_uri.removeprefix(staged_root)
                if not relative or relative == "_COMPLETE":
                    continue
                subprocess.run(
                    ["gcloud", "storage", "cp", staged_uri, final_root + relative],
                    check=True,
                    timeout=600,
                )
            if full_input:
                subprocess.run(
                    ["gcloud", "storage", "cp", "/dev/null", sentinel_uri],
                    check=True,
                    timeout=600,
                )
                log.info("Wrote sentinel: %s", sentinel_uri)
            else:
                log.info("Partial --max-files run: not publishing _COMPLETE")
        finally:
            subprocess.run(
                ["gcloud", "storage", "rm", "--recursive", staged_root],
                check=False,
                timeout=600,
            )


if __name__ == "__main__":
    main()
