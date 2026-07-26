#!/usr/bin/env python3
# ruff: noqa: E402
"""Pack tokenized enriched parquet documents into fixed-length training rows.

This is the cppmega-local port of nanochat's offline packed-row materializer.
The input rows must already contain tokenized enriched columns such as
``token_ids`` and token-aligned structure/AST/chunk metadata. This stage does
not parse C++; it only repacks whole enriched documents into dense LM rows.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import sys
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, SupportsInt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

from cppmega.data.nanochat_pipeline.document_identity import stable_doc_signature
from cppmega.data.domain_schema import (
    DOMAIN_SCHEMA_SHA256,
    DOMAIN_SCHEMA_SHA256_METADATA_KEY,
    DomainKind,
    ParseConfidence,
    TOKEN_DOMAIN_EDGE_COLUMN_FAMILIES,
    normalize_domain_edge_record,
    validate_case5_contract_metadata,
)
from cppmega.data.nanochat_pipeline.packed_rows_schema import (
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
    DOC_IDS_COLUMN,
    INPUT_IDS_COLUMN,
    LOSS_MASK_COLUMN,
    NUM_DOCS_COLUMN,
    PACKED_ROWS_CHUNK_METADATA_COLUMNS,
    PACKED_ROWS_DENSE_FALLBACK_FILL_VALUES,
    PACKED_ROWS_OBJECTIVE_SOURCE_TO_TOKEN_COLUMN,
    PACKED_ROWS_TOKEN_METADATA_COLUMNS,
    PACK_ID_COLUMN,
    ROW_PLATFORM_IDS_COLUMN,
    SOURCE_DOC_IDS_COLUMN,
    SOURCE_FILEPATH_STABLE_IDS_COLUMN,
    SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN,
    SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
    SOURCE_HEADER_FRAGMENT_KINDS_COLUMN,
    SOURCE_IFIM_INSTRUCTION_TOKEN_IDS_COLUMN,
    SOURCE_COMMIT_MSG_TOKEN_IDS_COLUMN,
    SOURCE_PRE_TOKEN_IDS_COLUMN,
    SOURCE_POST_TOKEN_IDS_COLUMN,
    SOURCE_DIFF_TOKEN_IDS_COLUMN,
    SOURCE_PR_DISCUSSION_CHARS_COLUMN,
    SOURCE_PR_DISCUSSION_LINES_COLUMN,
    SOURCE_PR_NUMBERS_COLUMN,
    SOURCE_REPO_STABLE_IDS_COLUMN,
    SOURCE_BUILD_KINDS_COLUMN,
    SOURCE_DOC_TYPES_COLUMN,
    TARGET_IDS_COLUMN,
    TOKEN_PLATFORM_IDS_COLUMN,
    VALID_TOKEN_COUNT_COLUMN,
)
from cppmega.data.nanochat_pipeline.tokenized_enriched_schema import (
    AUTHOR_TIMESTAMP_COLUMN,
    BUILD_KIND_COLUMN,
    COMMIT_HASH_COLUMN,
    COMMIT_TIMESTAMP_COLUMN,
    DOC_TYPE_COLUMN,
    EDIT_OP_PER_TOKEN_COLUMN,
    FILEPATH_COLUMN,
    FILEPATH_STABLE_ID_COLUMN,
    FILE_LOCAL_COMMIT_INDEX_COLUMN,
    HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN,
    HEADER_FRAGMENT_KIND_COLUMN,
    HAS_PR_DISCUSSION_COLUMN,
    HAS_RENAME_AMBIGUITY_COLUMN,
    HUNK_ID_PER_TOKEN_COLUMN,
    IS_MERGE_COMMIT_COLUMN,
    PARENT_COUNT_COLUMN,
    PARENT_HASHES_COLUMN,
    PLATFORM_IDS_COLUMN,
    PR_DISCUSSION_CHARS_COLUMN,
    PR_DISCUSSION_LINES_COLUMN,
    PR_NUMBER_COLUMN,
    RAW_COMMIT_CHRONOLOGY_COLUMNS,
    REPO_COLUMN,
    REPO_STABLE_ID_COLUMN,
    TIMESTAMP_COLUMN,
    TOKEN_AST_DEPTH_COLUMN,
    TOKEN_AST_NODE_TYPE_COLUMN,
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_CALL_TARGETS_COLUMN,
    TOKEN_CHANGE_MASK_POST_COLUMN,
    TOKEN_CHANGE_MASK_PRE_COLUMN,
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
    TOKEN_ROLE_IDS_COLUMN,
    TOKEN_SCOPE_IDS_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_SIBLING_INDEX_COLUMN,
    TOKEN_SHELL_EDGES_COLUMN,
    TOKEN_SOURCE_DOC_IDS_COLUMN,
    TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
    TOKEN_STRUCTURE_IDS_COLUMN,
    TOKEN_SYMBOL_IDS_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
    TOKEN_TYPE_REFS_COLUMN,
    SOURCE_IDENTITY_REGISTRY_COLUMN,
)
from cppmega.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SymbolIdentityError,
    SymbolIdentityRegistry,
)
from cppmega.data.source_identity import (
    source_identity,
    validate_source_identity_registry,
)
from cppmega.data.tokenizer_contract import (
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY,
)
from scripts.nanochat_data.atomic_publish import atomic_output_file

PackingStrategy = Literal["best_fit", "sequential"]

TRAINED_TOKEN_COUNT_COLUMN = "trained_token_count"
SLACK_TOKENS_COLUMN = "slack_tokens"
SOURCE_DOC_INDICES_COLUMN = SOURCE_DOC_IDS_COLUMN
SOURCE_DOC_TOKEN_LENGTHS_COLUMN = "source_doc_token_lengths"
SOURCE_PLATFORM_IDS_COLUMN = "source_platform_ids"
SOURCE_DOC_ID_COLUMN = "source_doc_id"
# Optional per-document input column carrying doc-level dependency edges
# (a list of source_doc_index values this document depends on). Used to enforce
# a cross-document partial order during in-bin topological packing.
DOC_DEP_EDGES_COLUMN = "doc_dep_edges"
PACKED_ROW_MACRO_ROUTES_METADATA_KEY = "cppmega.macro_routes_version"
PACKED_ROW_MACRO_ROUTES_VERSION = "full_macro_concept_routes_v1"
REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION = SYMBOL_IDENTITY_SCHEMA_VERSION

PACKED_TOKEN_METADATA_COLUMNS = PACKED_ROWS_TOKEN_METADATA_COLUMNS
_STATIC_DOC_TYPES = {"code", "code_header", "build", "shell"}
_DIAGNOSTIC_DOMAIN_IDS = frozenset(
    int(domain) for domain in DomainKind if int(domain) >= 40
)
DEFAULT_PACK_TOKEN_WINDOW = 1024 * 1024


def stable_repo_id(repo_name: str) -> str:
    return hashlib.sha1(repo_name.encode("utf-8")).hexdigest()[:16]


def stable_filepath_id(repo_name: str, filepath: str) -> str:
    key = f"{repo_name}\0{filepath}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _has_value(value: object) -> bool:
    return value is not None and value != ""


def _is_static_code_record(record: dict[str, Any]) -> bool:
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


def _validate_static_provenance(
    record: dict[str, Any],
    *,
    repo: object,
    filepath: object,
) -> None:
    if not _is_static_code_record(record):
        return
    if not _has_value(repo):
        raise ValueError(
            "static code document missing repo provenance "
            f"for filepath={filepath!r}"
        )
    if not _has_value(filepath):
        raise ValueError(
            "static code document missing filepath provenance "
            f"for repo={repo!r}"
        )
PACKED_CHUNK_METADATA_COLUMNS = PACKED_ROWS_CHUNK_METADATA_COLUMNS
PACKED_ROW_PROVENANCE_COLUMNS = tuple(
    column for column in RAW_COMMIT_CHRONOLOGY_COLUMNS if column != TIMESTAMP_COLUMN
) + (TIMESTAMP_COLUMN,)

CHANGED_CHUNK_SPAN_STRUCT = pa.struct(
    [
        pa.field("start", pa.uint32()),
        pa.field("end", pa.uint32()),
    ]
)

EDGE_STRUCT = pa.struct(
    [
        pa.field("from", pa.uint16()),
        pa.field("to", pa.uint16()),
    ]
)

EDGE_TRIPLE_STRUCT = pa.struct(
    [
        pa.field("from", pa.uint32()),
        pa.field("to", pa.uint32()),
        pa.field("kind", pa.int32()),
    ]
)

SOURCE_IDENTITY_STRUCT = pa.struct(
    [
        pa.field("source_identity_id", pa.uint64(), nullable=False),
        pa.field("canonical_sha256", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)

PACKED_ROW_SCHEMA = pa.schema(
    [
        pa.field(PACK_ID_COLUMN, pa.int64()),
        pa.field(VALID_TOKEN_COUNT_COLUMN, pa.int32()),
        pa.field(TRAINED_TOKEN_COUNT_COLUMN, pa.int32()),
        pa.field(NUM_DOCS_COLUMN, pa.int32()),
        pa.field(SLACK_TOKENS_COLUMN, pa.int32()),
        pa.field(INPUT_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TARGET_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(LOSS_MASK_COLUMN, pa.list_(pa.uint8())),
        pa.field(DOC_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(SOURCE_DOC_INDICES_COLUMN, pa.list_(pa.int64())),
        pa.field(SOURCE_DOC_TOKEN_LENGTHS_COLUMN, pa.list_(pa.int32())),
        pa.field(SOURCE_PLATFORM_IDS_COLUMN, pa.list_(pa.list_(pa.uint16()))),
        pa.field(SOURCE_REPO_STABLE_IDS_COLUMN, pa.list_(pa.string())),
        pa.field(SOURCE_FILEPATH_STABLE_IDS_COLUMN, pa.list_(pa.string())),
        pa.field(SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN, pa.list_(pa.int32())),
        pa.field(SOURCE_PR_NUMBERS_COLUMN, pa.list_(pa.int64())),
        pa.field(SOURCE_HAS_PR_DISCUSSIONS_COLUMN, pa.list_(pa.bool_())),
        pa.field(SOURCE_PR_DISCUSSION_CHARS_COLUMN, pa.list_(pa.int32())),
        pa.field(SOURCE_PR_DISCUSSION_LINES_COLUMN, pa.list_(pa.int32())),
        pa.field(SOURCE_DOC_TYPES_COLUMN, pa.list_(pa.string())),
        pa.field(SOURCE_HEADER_FRAGMENT_KINDS_COLUMN, pa.list_(pa.string())),
        pa.field(SOURCE_BUILD_KINDS_COLUMN, pa.list_(pa.string())),
        pa.field(
            SOURCE_IFIM_INSTRUCTION_TOKEN_IDS_COLUMN,
            pa.list_(pa.list_(pa.uint32())),
        ),
        pa.field(SOURCE_COMMIT_MSG_TOKEN_IDS_COLUMN, pa.list_(pa.list_(pa.uint32()))),
        pa.field(SOURCE_PRE_TOKEN_IDS_COLUMN, pa.list_(pa.list_(pa.uint32()))),
        pa.field(SOURCE_POST_TOKEN_IDS_COLUMN, pa.list_(pa.list_(pa.uint32()))),
        pa.field(SOURCE_DIFF_TOKEN_IDS_COLUMN, pa.list_(pa.list_(pa.uint32()))),
        pa.field(ROW_PLATFORM_IDS_COLUMN, pa.list_(pa.uint16())),
        pa.field(SYMBOL_IDENTITIES_COLUMN, pa.list_(pa.struct([
            pa.field("symbol_id", pa.uint64()),
            pa.field("symbol_key", pa.string()),
        ]))),
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
        pa.field(TOKEN_PLATFORM_IDS_COLUMN, pa.list_(pa.uint16())),
        pa.field(TOKEN_STRUCTURE_IDS_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_DEP_LEVELS_COLUMN, pa.list_(pa.uint16())),
        pa.field(TOKEN_AST_DEPTH_COLUMN, pa.list_(pa.int32())),
        pa.field(TOKEN_SIBLING_INDEX_COLUMN, pa.list_(pa.int32())),
        pa.field(TOKEN_AST_NODE_TYPE_COLUMN, pa.list_(pa.int32())),
        pa.field(TOKEN_DOMAIN_IDS_COLUMN, pa.list_(pa.uint16())),
        pa.field(TOKEN_ROLE_IDS_COLUMN, pa.list_(pa.uint16())),
        pa.field(TOKEN_ENTITY_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TOKEN_SCOPE_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TOKEN_SOURCE_DOC_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TOKEN_SOURCE_IDENTITY_IDS_COLUMN, pa.list_(pa.uint64())),
        pa.field(TOKEN_CONFIDENCE_IDS_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_SYMBOL_IDS_COLUMN, pa.list_(pa.uint64())),
        pa.field(TOKEN_CALL_TARGETS_COLUMN, pa.list_(pa.uint64())),
        pa.field(TOKEN_TYPE_REFS_COLUMN, pa.list_(pa.uint64())),
        pa.field(TOKEN_DEF_USE_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_CHANGE_MASK_PRE_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_CHANGE_MASK_POST_COLUMN, pa.list_(pa.uint8())),
        pa.field(HUNK_ID_PER_TOKEN_COLUMN, pa.list_(pa.int32())),
        pa.field(EDIT_OP_PER_TOKEN_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_CHUNK_STARTS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TOKEN_CHUNK_ENDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TOKEN_CHUNK_KINDS_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_CHUNK_DEP_LEVELS_COLUMN, pa.list_(pa.uint16())),
        pa.field(TOKEN_CALL_EDGES_COLUMN, pa.list_(EDGE_STRUCT)),
        pa.field(TOKEN_TYPE_EDGES_COLUMN, pa.list_(EDGE_STRUCT)),
        pa.field(TOKEN_DOMAIN_EDGES_COLUMN, pa.list_(EDGE_TRIPLE_STRUCT)),
        pa.field(TOKEN_BUILD_EDGES_COLUMN, pa.list_(EDGE_TRIPLE_STRUCT)),
        pa.field(TOKEN_SHELL_EDGES_COLUMN, pa.list_(EDGE_TRIPLE_STRUCT)),
        pa.field(TOKEN_DIAGNOSTIC_EDGES_COLUMN, pa.list_(EDGE_TRIPLE_STRUCT)),
        pa.field(TOKEN_CROSS_DOMAIN_EDGES_COLUMN, pa.list_(EDGE_TRIPLE_STRUCT)),
        pa.field(CHANGED_CHUNK_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(CHANGED_CHUNK_SPANS_COLUMN, pa.list_(CHANGED_CHUNK_SPAN_STRUCT)),
        pa.field(SOURCE_IDENTITY_REGISTRY_COLUMN, pa.list_(SOURCE_IDENTITY_STRUCT)),
    ]
)


def _packed_row_schema_with_metadata() -> pa.Schema:
    metadata = dict(PACKED_ROW_SCHEMA.metadata or {})
    metadata[PACKED_ROW_MACRO_ROUTES_METADATA_KEY.encode("utf-8")] = (
        PACKED_ROW_MACRO_ROUTES_VERSION.encode("utf-8")
    )
    metadata[SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")] = str(
        REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION
    ).encode("ascii")
    metadata[DOMAIN_DELIMITER_CONTRACT_METADATA_KEY.encode("utf-8")] = (
        DOMAIN_DELIMITER_CONTRACT_SHA256.encode("ascii")
    )
    metadata[DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8")] = (
        DOMAIN_SCHEMA_SHA256.encode("ascii")
    )
    metadata[TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode("utf-8")] = (
        TOKENIZER_CONTRACT_SHA256.encode("ascii")
    )
    metadata[b"cppmega.case5_schema"] = b"case5_domain_routes_v1"
    return PACKED_ROW_SCHEMA.with_metadata(metadata)


PACKED_ROW_OUTPUT_SCHEMA = _packed_row_schema_with_metadata()


@dataclass(frozen=True)
class NormalizedDoc:
    """A normalized per-document row ready for offline packing."""

    source_doc_index: int
    stable_doc_id: int
    stable_source_id: int
    source_identity_registry: tuple[dict[str, int | str], ...]
    token_ids: list[int]
    token_meta: dict[str, list[int]]
    chunk_starts: list[int]
    chunk_ends: list[int]
    chunk_kinds: list[int]
    chunk_dep_levels: list[int]
    call_edges: list[dict[str, int]]
    type_edges: list[dict[str, int]]
    platform_ids: list[int]
    changed_chunk_ids: list[int]
    changed_chunk_spans: list[tuple[int, int]]
    chronology: dict[str, Any]
    symbol_identities: list[dict[str, object]] = field(default_factory=list)
    objective_token_ids: dict[str, list[int]] = field(default_factory=dict)
    # Explicit doc-level dependency edges: source_doc_index values this document
    # depends on (a dependency must be packed BEFORE this document in the row).
    # Empty when only dep_level ordering applies.
    doc_dep_edges: tuple[int, ...] = ()
    domain_edges: list[dict[str, int]] = field(default_factory=list)
    build_edges: list[dict[str, int]] = field(default_factory=list)
    shell_edges: list[dict[str, int]] = field(default_factory=list)
    diagnostic_edges: list[dict[str, int]] = field(default_factory=list)
    cross_domain_edges: list[dict[str, int]] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def dep_level(self) -> int:
        """Topological level of this document.

        The dep level is the minimum topological level encoded in the
        document's per-chunk dep levels (``chunk_dep_levels``), falling back to
        the minimum per-token dep level when chunk levels are absent, and to 0
        when no dependency metadata exists. Using the minimum makes a document
        that *defines* low-level (dependency) symbols sort before documents that
        only *use* higher-level ones, which is the property a dependency-first
        packed order requires.
        """

        if self.chunk_dep_levels:
            return int(min(self.chunk_dep_levels))
        token_dep_levels = self.token_meta.get(TOKEN_DEP_LEVELS_COLUMN)
        if token_dep_levels:
            return int(min(token_dep_levels))
        return 0

    def to_overflow_record(self) -> dict[str, Any]:
        record = {
            "source_doc_index": int(self.source_doc_index),
            SOURCE_DOC_ID_COLUMN: int(self.stable_doc_id),
            "token_count": int(self.token_count),
            TOKEN_IDS_COLUMN: list(self.token_ids),
            PLATFORM_IDS_COLUMN: list(self.platform_ids),
            TOKEN_CHUNK_STARTS_COLUMN: list(self.chunk_starts),
            TOKEN_CHUNK_ENDS_COLUMN: list(self.chunk_ends),
            TOKEN_CHUNK_KINDS_COLUMN: list(self.chunk_kinds),
            TOKEN_CHUNK_DEP_LEVELS_COLUMN: list(self.chunk_dep_levels),
            TOKEN_CALL_EDGES_COLUMN: [dict(edge) for edge in self.call_edges],
            TOKEN_TYPE_EDGES_COLUMN: [dict(edge) for edge in self.type_edges],
            TOKEN_DOMAIN_EDGES_COLUMN: [dict(edge) for edge in self.domain_edges],
            TOKEN_BUILD_EDGES_COLUMN: [dict(edge) for edge in self.build_edges],
            TOKEN_SHELL_EDGES_COLUMN: [dict(edge) for edge in self.shell_edges],
            TOKEN_DIAGNOSTIC_EDGES_COLUMN: [
                dict(edge) for edge in self.diagnostic_edges
            ],
            TOKEN_CROSS_DOMAIN_EDGES_COLUMN: [
                dict(edge) for edge in self.cross_domain_edges
            ],
            CHANGED_CHUNK_IDS_COLUMN: list(self.changed_chunk_ids),
            CHANGED_CHUNK_SPANS_COLUMN: [
                {"start": int(start), "end": int(end)}
                for start, end in self.changed_chunk_spans
            ],
            SYMBOL_IDENTITIES_COLUMN: [dict(item) for item in self.symbol_identities],
            SOURCE_IDENTITY_REGISTRY_COLUMN: [
                dict(entry) for entry in self.source_identity_registry
            ],
        }
        record.update(self.chronology)
        record.update(
            {
                token_column: list(values)
                for token_column, values in self.objective_token_ids.items()
            }
        )
        for column, values in self.token_meta.items():
            record[column] = list(values)
        return record


@dataclass
class PackBin:
    """One fixed-length output row under construction."""

    docs: list[NormalizedDoc] = field(default_factory=list)
    used_tokens: int = 0

    def can_fit(self, token_count: int, target_length: int) -> bool:
        return self.used_tokens + token_count <= target_length

    def add(self, doc: NormalizedDoc) -> None:
        self.docs.append(doc)
        self.used_tokens += doc.token_count


def _doc_has_temporal_chronology(doc: NormalizedDoc) -> bool:
    """Return True only for commit/PR chronology, not static code provenance.

    Static code/header rows legitimately carry repo/filepath stable IDs.  Those
    identifiers must stay available for audit/sampling, but they are not a
    commit window and must not prevent unrelated static documents from sharing a
    packed row.
    """
    return any(
        doc.chronology.get(column) not in (None, "")
        for column in (
            COMMIT_HASH_COLUMN,
            TIMESTAMP_COLUMN,
            FILE_LOCAL_COMMIT_INDEX_COLUMN,
            AUTHOR_TIMESTAMP_COLUMN,
            COMMIT_TIMESTAMP_COLUMN,
            PR_NUMBER_COLUMN,
        )
    )


def _doc_commit_window_key(doc: NormalizedDoc) -> tuple[str, str, int] | None:
    repo_stable_id = doc.chronology.get(REPO_STABLE_ID_COLUMN)
    filepath_stable_id = doc.chronology.get(FILEPATH_STABLE_ID_COLUMN)
    file_local_commit_index = doc.chronology.get(FILE_LOCAL_COMMIT_INDEX_COLUMN)
    if repo_stable_id in (None, "") or filepath_stable_id in (None, ""):
        return None
    if file_local_commit_index is None:
        return None
    return (
        str(repo_stable_id),
        str(filepath_stable_id),
        int(file_local_commit_index),
    )


def _docs_form_strict_commit_window(docs: list[NormalizedDoc]) -> bool:
    if len(docs) <= 1:
        return True
    keys = [_doc_commit_window_key(doc) for doc in docs]
    if any(key is None for key in keys):
        return False
    repo_file_pairs = {(key[0], key[1]) for key in keys if key is not None}
    if len(repo_file_pairs) != 1:
        return False
    indices = sorted(key[2] for key in keys if key is not None)
    if len(set(indices)) != len(indices):
        return False
    expected = list(range(indices[0], indices[0] + len(indices)))
    return indices == expected


def _pack_bin_accepts_doc(bin_docs: list[NormalizedDoc], candidate: NormalizedDoc) -> bool:
    if not bin_docs:
        return True
    existing_has_temporal = any(_doc_has_temporal_chronology(doc) for doc in bin_docs)
    candidate_has_temporal = _doc_has_temporal_chronology(candidate)
    if not existing_has_temporal and not candidate_has_temporal:
        return True
    return _docs_form_strict_commit_window([*bin_docs, candidate])


def _topological_doc_order(docs: list[NormalizedDoc]) -> list[NormalizedDoc]:
    """Order documents dependency-first within a packed row.

    Ordering is a stable Kahn topological sort over the doc-level dependency DAG
    implied by ``doc_dep_edges`` (a dependency must precede its dependents),
    with the ready-set prioritized by ``dep_level`` ascending and
    ``source_doc_index`` ascending for determinism. When no explicit edges are
    present this degenerates to a stable sort by ``(dep_level, source_doc_index)``.

    Edges referencing documents outside this bin are ignored (the dependency is
    in a different row); only intra-bin partial order is enforced here.
    """

    in_bin = {doc.source_doc_index for doc in docs}
    by_index = {doc.source_doc_index: doc for doc in docs}

    # Build adjacency: dependency -> dependents, and indegree per doc.
    dependents: dict[int, list[int]] = {idx: [] for idx in in_bin}
    indegree: dict[int, int] = {idx: 0 for idx in in_bin}
    for doc in docs:
        for dep in doc.doc_dep_edges:
            dep = int(dep)
            if dep == doc.source_doc_index or dep not in in_bin:
                continue
            dependents[dep].append(doc.source_doc_index)
            indegree[doc.source_doc_index] += 1

    def _sort_key(idx: int) -> tuple[int, int]:
        doc = by_index[idx]
        return (doc.dep_level, doc.source_doc_index)

    ready = sorted((idx for idx in in_bin if indegree[idx] == 0), key=_sort_key)
    ordered: list[NormalizedDoc] = []
    while ready:
        idx = ready.pop(0)
        ordered.append(by_index[idx])
        for dependent in dependents[idx]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                # Insert keeping the ready set sorted by (dep_level, index).
                key = _sort_key(dependent)
                lo = 0
                hi = len(ready)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if _sort_key(ready[mid]) < key:
                        lo = mid + 1
                    else:
                        hi = mid
                ready.insert(lo, dependent)

    if len(ordered) != len(docs):
        cyclic = sorted(set(in_bin) - {doc.source_doc_index for doc in ordered})
        raise ValueError(
            "cyclic doc_dep_edges detected within pack bin; cannot topologically "
            f"order documents involving source_doc_index={cyclic}"
        )
    return ordered


def _order_docs_for_row(docs: list[NormalizedDoc]) -> list[NormalizedDoc]:
    if len(docs) > 1 and _docs_form_strict_commit_window(docs):
        return sorted(
            docs,
            key=lambda doc: (
                _coerce_optional_int(doc.chronology.get(FILE_LOCAL_COMMIT_INDEX_COLUMN))
                or 0,
                doc.source_doc_index,
            ),
        )
    return _topological_doc_order(docs)


def _shared_chronology_for_docs(docs: list[NormalizedDoc]) -> dict[str, Any]:
    shared: dict[str, Any] = {}
    for column in PACKED_ROW_PROVENANCE_COLUMNS:
        values = [doc.chronology.get(column) for doc in docs]
        first = values[0] if values else None
        shared[column] = first if all(value == first for value in values) else None
    return shared


def _merged_platform_ids_for_docs(docs: list[NormalizedDoc]) -> list[int]:
    # This row-level value is provenance, not the fixed-width model input.
    # Mixed packed rows may legitimately describe more platforms than one
    # source document.  Exact per-document bags remain in
    # ``source_platform_ids`` and are expanded through row-local ``doc_ids`` by
    # the data loader, so capping the union here only makes valid packs fail.
    return sorted(
        {
            int(platform_id)
            for doc in docs
            for platform_id in doc.platform_ids
            if int(platform_id) > 0
        }
    )


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray, int, bool)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, SupportsInt):
        return int(value)
    return None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(item) for item in value]


def _normalize_edge_list(value: Any) -> list[dict[str, int]]:
    edges: list[dict[str, int]] = []
    for item in value or []:
        if isinstance(item, dict):
            src = int(item.get("from", 0))
            dst = int(item.get("to", 0))
        else:
            src = int(item[0])
            dst = int(item[1])
        edges.append({"from": src, "to": dst})
    return edges


def _normalize_edge_triples(
    value: Any,
    *,
    family: str,
) -> list[dict[str, int]]:
    edges: list[dict[str, int]] = []
    for item in value or []:
        src, dst, kind = normalize_domain_edge_record(item, family=family)
        edges.append({"from": src, "to": dst, "kind": kind})
    return edges


def _validate_token_edge_triples(
    *,
    source_doc_index: int,
    token_count: int,
    column: str,
    edges: list[dict[str, int]],
) -> None:
    for edge in edges:
        src = int(edge["from"])
        dst = int(edge["to"])
        if not (0 <= src < token_count and 0 <= dst < token_count):
            raise ValueError(
                f"{column} edge out of range for source_doc_index={source_doc_index}: "
                f"got ({src}, {dst}) with token_count={token_count}"
            )


def _normalize_changed_chunk_spans(value: Any) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for item in value or []:
        if isinstance(item, dict):
            start = int(item.get("start", 0))
            end = int(item.get("end", 0))
        else:
            start = int(item[0])
            end = int(item[1])
        spans.append((start, end))
    return spans


def _validate_chunk_graph_references(
    *,
    source_doc_index: int,
    chunk_count: int,
    call_edges: list[dict[str, int]],
    type_edges: list[dict[str, int]],
    changed_chunk_ids: list[int],
) -> None:
    has_graph_metadata = bool(call_edges or type_edges or changed_chunk_ids)
    if has_graph_metadata and chunk_count == 0:
        raise ValueError(
            "graph/chunk metadata requires non-empty token_chunk_* layout "
            f"for source_doc_index={source_doc_index}"
        )

    for column, edges in (
        (TOKEN_CALL_EDGES_COLUMN, call_edges),
        (TOKEN_TYPE_EDGES_COLUMN, type_edges),
    ):
        for edge in edges:
            src = int(edge["from"])
            dst = int(edge["to"])
            if not (0 <= src < chunk_count and 0 <= dst < chunk_count):
                raise ValueError(
                    f"{column} edge out of range for source_doc_index={source_doc_index}: "
                    f"got ({src}, {dst}) with chunk_count={chunk_count}"
                )

    for chunk_id in changed_chunk_ids:
        if not (0 <= int(chunk_id) < chunk_count):
            raise ValueError(
                f"{CHANGED_CHUNK_IDS_COLUMN} out of range for "
                f"source_doc_index={source_doc_index}: got {chunk_id} "
                f"with chunk_count={chunk_count}"
            )


def _normalize_chronology(record: dict[str, Any]) -> dict[str, Any]:
    repo = record.get(REPO_COLUMN)
    filepath = record.get(FILEPATH_COLUMN)
    repo_stable_id = record.get(REPO_STABLE_ID_COLUMN)
    if not _has_value(repo_stable_id) and _has_value(repo):
        repo_stable_id = stable_repo_id(str(repo))
    filepath_stable_id = record.get(FILEPATH_STABLE_ID_COLUMN)
    if (
        not _has_value(filepath_stable_id)
        and _has_value(repo)
        and _has_value(filepath)
    ):
        filepath_stable_id = stable_filepath_id(str(repo), str(filepath))
    _validate_static_provenance(record, repo=repo, filepath=filepath)
    chronology: dict[str, Any] = {
        REPO_COLUMN: repo,
        FILEPATH_COLUMN: filepath,
        COMMIT_HASH_COLUMN: record.get(COMMIT_HASH_COLUMN),
        TIMESTAMP_COLUMN: record.get(TIMESTAMP_COLUMN),
        PR_NUMBER_COLUMN: _coerce_optional_int(record.get(PR_NUMBER_COLUMN)),
        HAS_PR_DISCUSSION_COLUMN: _coerce_bool(
            record.get(HAS_PR_DISCUSSION_COLUMN, False)
        ),
        PR_DISCUSSION_CHARS_COLUMN: int(
            record.get(PR_DISCUSSION_CHARS_COLUMN) or 0
        ),
        PR_DISCUSSION_LINES_COLUMN: int(
            record.get(PR_DISCUSSION_LINES_COLUMN) or 0
        ),
        PARENT_HASHES_COLUMN: [
            str(item) for item in (record.get(PARENT_HASHES_COLUMN) or [])
        ],
        AUTHOR_TIMESTAMP_COLUMN: record.get(AUTHOR_TIMESTAMP_COLUMN),
        COMMIT_TIMESTAMP_COLUMN: record.get(COMMIT_TIMESTAMP_COLUMN),
        REPO_STABLE_ID_COLUMN: repo_stable_id,
        FILEPATH_STABLE_ID_COLUMN: filepath_stable_id,
        DOC_TYPE_COLUMN: record.get(DOC_TYPE_COLUMN),
        HEADER_FRAGMENT_KIND_COLUMN: record.get(HEADER_FRAGMENT_KIND_COLUMN),
        BUILD_KIND_COLUMN: record.get(BUILD_KIND_COLUMN),
    }
    parent_count = record.get(PARENT_COUNT_COLUMN)
    if parent_count is None:
        parent_count = len(chronology[PARENT_HASHES_COLUMN])
    chronology[PARENT_COUNT_COLUMN] = int(parent_count)
    chronology[IS_MERGE_COMMIT_COLUMN] = bool(
        record.get(IS_MERGE_COMMIT_COLUMN, chronology[PARENT_COUNT_COLUMN] > 1)
    )
    file_local_index = record.get(FILE_LOCAL_COMMIT_INDEX_COLUMN)
    chronology[FILE_LOCAL_COMMIT_INDEX_COLUMN] = (
        int(file_local_index) if file_local_index is not None else None
    )
    chronology[HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN] = bool(
        record.get(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN, False)
    )
    chronology[HAS_RENAME_AMBIGUITY_COLUMN] = bool(
        record.get(HAS_RENAME_AMBIGUITY_COLUMN, False)
    )
    return chronology


def _normalize_token_meta(
    record: dict[str, Any],
    *,
    token_count: int,
    source_doc_index: int,
) -> dict[str, list[int]]:
    token_meta: dict[str, list[int]] = {}
    for column in PACKED_TOKEN_METADATA_COLUMNS:
        values = _as_int_list(record.get(column))
        if values and len(values) != token_count:
            raise ValueError(
                f"{column} length mismatch for source_doc_index={source_doc_index}: "
                f"expected {token_count}, got {len(values)}"
            )
        token_meta[column] = values
    return token_meta


def _normalize_chunk_meta(
    record: dict[str, Any],
    *,
    source_doc_index: int,
    token_count: int,
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[int],
    list[dict[str, int]],
    list[dict[str, int]],
    list[int],
    list[tuple[int, int]],
]:
    starts = _as_int_list(record.get(TOKEN_CHUNK_STARTS_COLUMN))
    ends = _as_int_list(record.get(TOKEN_CHUNK_ENDS_COLUMN))
    kinds = _as_int_list(record.get(TOKEN_CHUNK_KINDS_COLUMN))
    dep_levels = _as_int_list(record.get(TOKEN_CHUNK_DEP_LEVELS_COLUMN))

    nonempty = [values for values in (starts, ends, kinds, dep_levels) if values]
    if nonempty:
        expected = len(nonempty[0])
        for column_name, values in (
            (TOKEN_CHUNK_STARTS_COLUMN, starts),
            (TOKEN_CHUNK_ENDS_COLUMN, ends),
            (TOKEN_CHUNK_KINDS_COLUMN, kinds),
            (TOKEN_CHUNK_DEP_LEVELS_COLUMN, dep_levels),
        ):
            if len(values) != expected:
                raise ValueError(
                    f"{column_name} length mismatch for source_doc_index={source_doc_index}: "
                    f"expected {expected}, got {len(values)}"
                )
        for start, end in zip(starts, ends):
            if not (0 <= start < end <= token_count):
                raise ValueError(
                    f"chunk span out of range for source_doc_index={source_doc_index}: "
                    f"got ({start}, {end}) with token_count={token_count}"
                )

    call_edges = _normalize_edge_list(record.get(TOKEN_CALL_EDGES_COLUMN))
    type_edges = _normalize_edge_list(record.get(TOKEN_TYPE_EDGES_COLUMN))
    changed_chunk_ids = _as_int_list(record.get(CHANGED_CHUNK_IDS_COLUMN))
    changed_chunk_spans = _normalize_changed_chunk_spans(
        record.get(CHANGED_CHUNK_SPANS_COLUMN)
    )
    if len(changed_chunk_ids) != len(changed_chunk_spans):
        raise ValueError(
            f"{CHANGED_CHUNK_IDS_COLUMN}/{CHANGED_CHUNK_SPANS_COLUMN} length mismatch "
            f"for source_doc_index={source_doc_index}: expected equal lengths, got "
            f"{len(changed_chunk_ids)} and {len(changed_chunk_spans)}"
        )
    _validate_chunk_graph_references(
        source_doc_index=source_doc_index,
        chunk_count=len(starts),
        call_edges=call_edges,
        type_edges=type_edges,
        changed_chunk_ids=changed_chunk_ids,
    )
    for start, end in changed_chunk_spans:
        if not (0 <= start < end <= token_count):
            raise ValueError(
                f"{CHANGED_CHUNK_SPANS_COLUMN} out of range for source_doc_index={source_doc_index}: "
                f"got ({start}, {end}) with token_count={token_count}"
            )
    return (
        starts,
        ends,
        kinds,
        dep_levels,
        call_edges,
        type_edges,
        changed_chunk_ids,
        changed_chunk_spans,
    )


def _normalize_domain_graph_meta(
    record: dict[str, Any],
    *,
    source_doc_index: int,
    token_count: int,
) -> tuple[
    list[dict[str, int]],
    list[dict[str, int]],
    list[dict[str, int]],
    list[dict[str, int]],
    list[dict[str, int]],
]:
    columns = (
        TOKEN_DOMAIN_EDGES_COLUMN,
        TOKEN_BUILD_EDGES_COLUMN,
        TOKEN_SHELL_EDGES_COLUMN,
        TOKEN_DIAGNOSTIC_EDGES_COLUMN,
        TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
    )
    normalized = tuple(
        _normalize_edge_triples(
            record.get(column),
            family=TOKEN_DOMAIN_EDGE_COLUMN_FAMILIES[column],
        )
        for column in columns
    )
    for column, edges in zip(columns, normalized):
        _validate_token_edge_triples(
            source_doc_index=source_doc_index,
            token_count=token_count,
            column=column,
            edges=edges,
        )
    return (
        normalized[0],
        normalized[1],
        normalized[2],
        normalized[3],
        normalized[4],
    )


def _normalize_diagnostic_confidence(
    *,
    source_doc_index: int,
    token_meta: dict[str, list[int]],
    diagnostic_edges: list[dict[str, int]],
) -> None:
    """Make edge-free diagnostic documents explicitly uncertified.

    HEURISTIC/PARTIAL confidence is only meaningful when the parser emitted a
    structural diagnostic edge. At the packed-row publication boundary, an
    edge-free diagnostic must therefore carry RAW confidence on its diagnostic
    payload instead of looking structurally certified.
    """

    if diagnostic_edges:
        return
    domain_ids = token_meta[TOKEN_DOMAIN_IDS_COLUMN]
    diagnostic_positions = [
        index
        for index, domain_id in enumerate(domain_ids)
        if int(domain_id) in _DIAGNOSTIC_DOMAIN_IDS
    ]
    if not diagnostic_positions:
        return

    confidence_ids = token_meta[TOKEN_CONFIDENCE_IDS_COLUMN]
    if not confidence_ids:
        raise ValueError(
            "edge-free diagnostic document is missing token_confidence_ids for "
            f"source_doc_index={source_doc_index}"
        )
    if any(
        int(confidence_ids[index]) == int(ParseConfidence.RAW)
        for index in diagnostic_positions
    ):
        return

    uncertified = {
        int(ParseConfidence.HEURISTIC),
        int(ParseConfidence.PARTIAL),
    }
    changed = False
    for index in diagnostic_positions:
        if int(confidence_ids[index]) in uncertified:
            confidence_ids[index] = int(ParseConfidence.RAW)
            changed = True
    if not changed:
        raise ValueError(
            "edge-free diagnostic document must provide RAW or downgradeable "
            "HEURISTIC/PARTIAL confidence for "
            f"source_doc_index={source_doc_index}"
        )


def normalize_document_record(
    record: dict[str, Any],
    *,
    source_doc_index: int,
    stable_doc_id: int | None = None,
    identity_registry: SymbolIdentityRegistry | None = None,
) -> NormalizedDoc:
    """Validate and normalize one input parquet record."""

    token_ids = _as_int_list(record.get(TOKEN_IDS_COLUMN))
    if not token_ids:
        raise ValueError(
            f"Missing or empty {TOKEN_IDS_COLUMN} for source_doc_index={source_doc_index}"
        )

    token_meta = _normalize_token_meta(
        record,
        token_count=len(token_ids),
        source_doc_index=source_doc_index,
    )
    row_registry = SymbolIdentityRegistry()
    normalized_identities = row_registry.register_records(
        record.get(SYMBOL_IDENTITIES_COLUMN, []),
        source=f"source_doc_index={source_doc_index}",
    )
    used_symbol_ids = {
        int(value)
        for column in (
            TOKEN_SYMBOL_IDS_COLUMN,
            TOKEN_CALL_TARGETS_COLUMN,
            TOKEN_TYPE_REFS_COLUMN,
        )
        for value in token_meta[column]
        if int(value) != 0
    }
    row_registry.require_ids(
        used_symbol_ids,
        source=f"source_doc_index={source_doc_index}",
    )
    if identity_registry is not None:
        identity_registry.register_records(
            normalized_identities,
            source=f"source_doc_index={source_doc_index}",
        )

    raw_registry = [
        dict(entry) for entry in record.get(SOURCE_IDENTITY_REGISTRY_COLUMN, [])
    ]
    if raw_registry:
        registry = validate_source_identity_registry(raw_registry)
        stable_source_id = next(iter(registry))
    else:
        identity = source_identity(record)
        raw_registry = [identity.as_dict()]
        stable_source_id = identity.source_identity_id

    source_doc_values = token_meta[TOKEN_SOURCE_DOC_IDS_COLUMN]
    token_meta[TOKEN_SOURCE_DOC_IDS_COLUMN] = [
        int(value) if int(value) > 0 else 1
        for value in (source_doc_values or [1] * len(token_ids))
    ]
    source_identity_values = token_meta[TOKEN_SOURCE_IDENTITY_IDS_COLUMN]
    if len(raw_registry) > 1 and (
        not source_identity_values
        or any(int(value) == 0 for value in source_identity_values)
    ):
        raise ValueError(
            "multi-source document is missing exact token source identities"
        )
    token_meta[TOKEN_SOURCE_IDENTITY_IDS_COLUMN] = [
        int(value) if int(value) > 0 else stable_source_id
        for value in (source_identity_values or [stable_source_id] * len(token_ids))
    ]
    validate_source_identity_registry(
        raw_registry,
        referenced_ids=token_meta[TOKEN_SOURCE_IDENTITY_IDS_COLUMN],
    )
    for column in (
        TOKEN_SYMBOL_IDS_COLUMN,
        TOKEN_CALL_TARGETS_COLUMN,
        TOKEN_TYPE_REFS_COLUMN,
        TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
    ):
        if any(not 0 <= int(value) < (1 << 64) for value in token_meta[column]):
            raise ValueError(f"{column} must contain uint64 values")
    if any(
        not 0 <= int(value) < (1 << 32)
        for value in token_meta[TOKEN_SOURCE_DOC_IDS_COLUMN]
    ):
        raise ValueError(f"{TOKEN_SOURCE_DOC_IDS_COLUMN} must contain uint32 values")
    (
        chunk_starts,
        chunk_ends,
        chunk_kinds,
        chunk_dep_levels,
        call_edges,
        type_edges,
        changed_chunk_ids,
        changed_chunk_spans,
    ) = _normalize_chunk_meta(
        record,
        source_doc_index=source_doc_index,
        token_count=len(token_ids),
    )
    (
        domain_edges,
        build_edges,
        shell_edges,
        diagnostic_edges,
        cross_domain_edges,
    ) = _normalize_domain_graph_meta(
        record,
        source_doc_index=source_doc_index,
        token_count=len(token_ids),
    )
    _normalize_diagnostic_confidence(
        source_doc_index=source_doc_index,
        token_meta=token_meta,
        diagnostic_edges=diagnostic_edges,
    )

    return NormalizedDoc(
        source_doc_index=int(source_doc_index),
        stable_doc_id=int(source_doc_index + 1 if stable_doc_id is None else stable_doc_id),
        stable_source_id=stable_source_id,
        source_identity_registry=tuple(raw_registry),
        token_ids=token_ids,
        token_meta=token_meta,
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_kinds=chunk_kinds,
        chunk_dep_levels=chunk_dep_levels,
        call_edges=call_edges,
        type_edges=type_edges,
        domain_edges=domain_edges,
        build_edges=build_edges,
        shell_edges=shell_edges,
        diagnostic_edges=diagnostic_edges,
        cross_domain_edges=cross_domain_edges,
        platform_ids=_as_int_list(record.get(PLATFORM_IDS_COLUMN)),
        changed_chunk_ids=changed_chunk_ids,
        changed_chunk_spans=changed_chunk_spans,
        chronology=_normalize_chronology(record),
        symbol_identities=row_registry.records(used_symbol_ids),
        objective_token_ids={
            token_column: _as_int_list(record.get(token_column))
            for token_column in PACKED_ROWS_OBJECTIVE_SOURCE_TO_TOKEN_COLUMN.values()
        },
        doc_dep_edges=tuple(
            int(dep) for dep in (record.get(DOC_DEP_EDGES_COLUMN) or [])
        ),
    )


def _list_input_files(input_path: str | os.PathLike[str]) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            child for child in path.iterdir() if child.is_file() and child.suffix == ".parquet"
        )
        if not files:
            raise FileNotFoundError(f"No parquet files found under {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _require_symbol_identity_schema(input_path: str | os.PathLike[str]) -> None:
    key = SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")
    corpus_registry = SymbolIdentityRegistry()
    identity_type = pa.list_(
        pa.struct(
            [
                pa.field("symbol_id", pa.uint64()),
                pa.field("symbol_key", pa.string()),
            ]
        )
    )
    for path in _list_input_files(input_path):
        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        metadata = schema.metadata or {}
        validate_case5_contract_metadata(metadata, where=path)
        raw_version = metadata.get(key)
        try:
            version = int(raw_version) if raw_version is not None else None
        except (TypeError, ValueError):
            version = None
        if version != REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION:
            raise SymbolIdentityError(
                f"{path}: missing or stale symbol identity metadata {raw_version!r}; "
                "regenerate tokenized parquet with clang USR/signature identities "
                f"before packing (required v{REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION})"
            )
        if SYMBOL_IDENTITIES_COLUMN not in schema.names:
            raise SymbolIdentityError(
                f"{path}: missing {SYMBOL_IDENTITIES_COLUMN!r} collision registry"
            )
        if schema.field(SYMBOL_IDENTITIES_COLUMN).type != identity_type:
            raise SymbolIdentityError(
                f"{path}: {SYMBOL_IDENTITIES_COLUMN} must be {identity_type}, got "
                f"{schema.field(SYMBOL_IDENTITIES_COLUMN).type}"
            )
        for column in (
            TOKEN_SYMBOL_IDS_COLUMN,
            TOKEN_CALL_TARGETS_COLUMN,
            TOKEN_TYPE_REFS_COLUMN,
        ):
            if column not in schema.names:
                continue
            field_type = schema.field(column).type
            if not pa.types.is_list(field_type) or field_type.value_type != pa.uint64():
                raise SymbolIdentityError(
                    f"{path}: {column} must use uint64 symbol IDs, got "
                    f"{field_type}"
                )
        row_offset = 0
        for batch in parquet_file.iter_batches(columns=[SYMBOL_IDENTITIES_COLUMN]):
            for local_index, records in enumerate(
                batch.column(0).to_pylist()
            ):
                corpus_registry.register_records(
                    records,
                    source=f"{path}:row={row_offset + local_index}",
                )
            row_offset += batch.num_rows


def _has_stable_doc_signature(record: dict[str, Any]) -> bool:
    explicit = (
        SOURCE_DOC_ID_COLUMN,
        "source_document_id",
        "document_id",
        "doc_id",
    )
    return any(record.get(column) is not None for column in explicit)


def _stable_doc_id_for_record(
    record: dict[str, Any],
    *,
    source_doc_index: int,
    signature_to_id: dict[str, int],
) -> int:
    del source_doc_index
    signature = stable_doc_signature(record)
    doc_id = signature_to_id.get(signature)
    if doc_id is None:
        doc_id = int.from_bytes(
            hashlib.sha256(signature.encode("utf-8")).digest()[:4], "big"
        ) or 1
        collision = next(
            (
                existing_signature
                for existing_signature, existing_id in signature_to_id.items()
                if existing_id == doc_id and existing_signature != signature
            ),
            None,
        )
        if collision is not None:
            raise ValueError(
                "stable document identity hash collision: "
                f"id={doc_id} signatures={collision!r}, {signature!r}"
            )
        signature_to_id[signature] = doc_id
    return int(doc_id)


def read_tokenized_documents(
    input_path: str | os.PathLike[str],
    *,
    token_budget: int | None = None,
    start_source_doc_index: int = 0,
    signature_to_id: dict[str, int] | None = None,
) -> list[NormalizedDoc]:
    """Read tokenized per-document parquet rows from a file or directory.

    ``token_budget`` caps intake: reading stops (whole docs only, never split)
    once the running sum of ``token_count`` would meet or exceed the budget. This
    keeps the loader memory-bounded for very large shard sets while preserving
    the whole-function invariant. ``start_source_doc_index`` lets a caller mix
    several sources into one contiguous ``source_doc_index`` space; passing a
    shared ``signature_to_id`` keeps stable doc IDs unique across sources.
    """

    if signature_to_id is None:
        signature_to_id = {}
    docs: list[NormalizedDoc] = []
    accumulated_tokens = 0
    for doc in iter_tokenized_documents(
        input_path,
        start_source_doc_index=start_source_doc_index,
        signature_to_id=signature_to_id,
    ):
        docs.append(doc)
        accumulated_tokens += doc.token_count
        if token_budget is not None and accumulated_tokens >= token_budget:
            break
    return docs


def _selected_input_columns(available: set[str]) -> list[str]:
    return [
        column
        for column in (
            TOKEN_IDS_COLUMN,
            SYMBOL_IDENTITIES_COLUMN,
            SOURCE_IDENTITY_REGISTRY_COLUMN,
            SOURCE_DOC_ID_COLUMN,
            "source_document_id",
            "document_id",
            "doc_id",
            PLATFORM_IDS_COLUMN,
            *RAW_COMMIT_CHRONOLOGY_COLUMNS,
            HAS_PR_DISCUSSION_COLUMN,
            PR_DISCUSSION_CHARS_COLUMN,
            PR_DISCUSSION_LINES_COLUMN,
            DOC_TYPE_COLUMN,
            HEADER_FRAGMENT_KIND_COLUMN,
            BUILD_KIND_COLUMN,
            *PACKED_TOKEN_METADATA_COLUMNS,
            *PACKED_CHUNK_METADATA_COLUMNS,
            *PACKED_ROWS_OBJECTIVE_SOURCE_TO_TOKEN_COLUMN.values(),
        )
        if column in available
    ]


def iter_tokenized_documents(
    input_path: str | os.PathLike[str],
    *,
    start_source_doc_index: int = 0,
    signature_to_id: dict[str, int] | None = None,
    input_batch_size: int = 1024,
) -> Iterator[NormalizedDoc]:
    """Yield normalized input documents without materializing a whole shard."""

    source_doc_index = int(start_source_doc_index)
    if signature_to_id is None:
        signature_to_id = {}
    identity_registry = SymbolIdentityRegistry()
    for path in _list_input_files(input_path):
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        if TOKEN_IDS_COLUMN not in available:
            raise ValueError(f"{path} is missing required column {TOKEN_IDS_COLUMN}")

        selected_columns = _selected_input_columns(available)
        for batch in parquet_file.iter_batches(
            columns=selected_columns,
            batch_size=input_batch_size,
        ):
            for record in batch.to_pylist():
                doc = normalize_document_record(
                    record,
                    source_doc_index=source_doc_index,
                    stable_doc_id=_stable_doc_id_for_record(
                        record,
                        source_doc_index=source_doc_index,
                        signature_to_id=signature_to_id,
                    ),
                    identity_registry=identity_registry,
                )
                yield doc
                source_doc_index += 1


def _pack_docs_best_fit(
    docs: list[NormalizedDoc],
    *,
    target_length: int,
) -> tuple[list[PackBin], list[dict[str, Any]]]:
    """Best-fit-decreasing bin-packing that minimizes residual padding.

    Documents are placed largest-first; each document goes into the already-open
    bin whose remaining slack it shrinks the most (least leftover slack after
    placement) while still satisfying the chronology/commit-window acceptance
    rule. A new bin opens only when no open bin can take the document. This
    drives per-row padding toward zero far more aggressively than filling one
    bin at a time. Ordering *within* a bin is handled later by the dependency
    topological sort in ``_order_docs_for_row``; placement here only governs
    which documents share a row.
    """

    overflow: list[dict[str, Any]] = []
    fitting: list[NormalizedDoc] = []
    oversized: list[NormalizedDoc] = []
    for doc in docs:
        if doc.token_count > target_length:
            # Whole-function invariant: never split a doc. An over-long doc is
            # kept WHOLE in its own single-doc block (materialized to a row whose
            # width grows to the doc length) and is also recorded in overflow.
            oversized.append(doc)
            overflow.append(doc.to_overflow_record())
        else:
            fitting.append(doc)

    # Largest-first; ties broken by source_doc_index for determinism.
    ordered = sorted(fitting, key=lambda doc: (-doc.token_count, doc.source_doc_index))

    bins: list[PackBin] = []
    for doc in oversized:
        own_block = PackBin()
        own_block.add(doc)
        bins.append(own_block)
    if not any(_doc_has_temporal_chronology(doc) for doc in ordered):
        # Fast indexed best-fit for static code/header docs.  The generic path
        # below scans all open bins for every document so it becomes quadratic on
        # macro-heavy header shards with tens of thousands of tiny documents.
        bins_by_remaining: dict[int, deque[PackBin]] = {}
        remaining_keys: list[int] = []

        def _push_open_bin(pack: PackBin) -> None:
            remaining = target_length - pack.used_tokens
            if remaining <= 0:
                return
            queue = bins_by_remaining.get(remaining)
            if queue is None:
                bins_by_remaining[remaining] = deque([pack])
                bisect.insort(remaining_keys, remaining)
            else:
                queue.append(pack)

        for doc in ordered:
            key_idx = bisect.bisect_left(remaining_keys, doc.token_count)
            if key_idx == len(remaining_keys):
                best_bin = PackBin()
                bins.append(best_bin)
            else:
                remaining = remaining_keys[key_idx]
                queue = bins_by_remaining[remaining]
                best_bin = queue.popleft()
                if not queue:
                    del bins_by_remaining[remaining]
                    remaining_keys.pop(key_idx)
            best_bin.add(doc)
            _push_open_bin(best_bin)

        bins.sort(key=lambda pack: min(doc.source_doc_index for doc in pack.docs))
        return bins, overflow

    for doc in ordered:
        best_bin: PackBin | None = None
        best_slack = target_length + 1
        for pack in bins:
            if not pack.can_fit(doc.token_count, target_length):
                continue
            if not _pack_bin_accepts_doc(pack.docs, doc):
                continue
            slack_after = target_length - (pack.used_tokens + doc.token_count)
            if slack_after < best_slack:
                best_slack = slack_after
                best_bin = pack
        if best_bin is None:
            best_bin = PackBin()
            bins.append(best_bin)
        best_bin.add(doc)

    # Deterministic row order: by first (smallest) contained source_doc_index.
    bins.sort(key=lambda pack: min(doc.source_doc_index for doc in pack.docs))
    return bins, overflow


def _pack_docs_sequential(
    docs: list[NormalizedDoc],
    *,
    target_length: int,
) -> tuple[list[PackBin], list[dict[str, Any]]]:
    bins: list[PackBin] = []
    overflow: list[dict[str, Any]] = []
    current = PackBin()
    for doc in sorted(docs, key=lambda item: item.source_doc_index):
        if doc.token_count > target_length:
            # Whole-function invariant: keep an over-long doc WHOLE in its own
            # block (never split/truncate). Flush any in-progress bin first so the
            # oversized doc stays in input order, then also record it in overflow.
            if current.docs:
                bins.append(current)
                current = PackBin()
            own_block = PackBin()
            own_block.add(doc)
            bins.append(own_block)
            overflow.append(doc.to_overflow_record())
            continue
        if current.docs and not current.can_fit(doc.token_count, target_length):
            bins.append(current)
            current = PackBin()
        if current.docs and not _pack_bin_accepts_doc(current.docs, doc):
            bins.append(current)
            current = PackBin()
        current.add(doc)
        if current.used_tokens == target_length:
            bins.append(current)
            current = PackBin()
    if current.docs:
        bins.append(current)
    return bins, overflow


def _pad(values: list[int], target_length: int, *, pad_value: int = 0) -> list[int]:
    return list(values) + [pad_value] * max(target_length - len(values), 0)


def _target_ids_for_packed_tokens(
    concatenated_tokens: list[int],
    *,
    target_length: int,
    pad_token_id: int,
) -> list[int]:
    return _pad(concatenated_tokens[1:], target_length, pad_value=pad_token_id)


def _loss_mask_for_packed_docs(
    doc_ids: list[int],
    *,
    target_length: int,
) -> list[int]:
    keep: list[int] = []
    for pos in range(len(doc_ids)):
        if pos + 1 >= len(doc_ids):
            keep.append(0)
        else:
            keep.append(1 if doc_ids[pos] == doc_ids[pos + 1] else 0)
    return _pad(keep, target_length, pad_value=0)


def _materialize_packed_row(
    docs: list[NormalizedDoc],
    *,
    target_length: int,
    pad_token_id: int,
    pack_id: int,
) -> dict[str, Any]:
    ordered_docs = _order_docs_for_row(docs)
    concatenated_tokens: list[int] = []
    doc_ids: list[int] = []
    source_doc_indices: list[int] = []
    source_doc_token_lengths: list[int] = []
    source_platform_ids: list[list[int]] = []
    source_repo_stable_ids: list[str | None] = []
    source_filepath_stable_ids: list[str | None] = []
    source_file_local_commit_indices: list[int | None] = []
    source_pr_numbers: list[int | None] = []
    source_has_pr_discussions: list[bool] = []
    source_pr_discussion_chars: list[int] = []
    source_pr_discussion_lines: list[int] = []
    source_doc_types: list[str | None] = []
    source_header_fragment_kinds: list[str | None] = []
    source_build_kinds: list[str | None] = []
    objective_source_ids: dict[str, list[list[int]]] = {
        source_column: []
        for source_column in PACKED_ROWS_OBJECTIVE_SOURCE_TO_TOKEN_COLUMN
    }
    chronology = _shared_chronology_for_docs(ordered_docs)
    symbol_identity_registry = SymbolIdentityRegistry()

    token_meta_acc: dict[str, list[int]] = {
        column: [] for column in PACKED_TOKEN_METADATA_COLUMNS
    }
    chunk_starts: list[int] = []
    chunk_ends: list[int] = []
    chunk_kinds: list[int] = []
    chunk_dep_levels: list[int] = []
    call_edges: list[dict[str, int]] = []
    type_edges: list[dict[str, int]] = []
    domain_edges: list[dict[str, int]] = []
    build_edges: list[dict[str, int]] = []
    shell_edges: list[dict[str, int]] = []
    diagnostic_edges: list[dict[str, int]] = []
    cross_domain_edges: list[dict[str, int]] = []
    changed_chunk_ids: list[int] = []
    changed_chunk_spans: list[dict[str, int]] = []
    source_identity_registry: dict[int, dict[str, int | str]] = {}

    token_offset = 0
    chunk_offset = 0
    # doc_ids is a row-local attention/loss segmentation channel. Every input
    # document gets a distinct segment even when multiple fragments share one
    # stable source identity; token_source_doc_ids preserves that provenance.
    for row_doc_id, doc in enumerate(ordered_docs, start=1):
        symbol_identity_registry.register_records(
            doc.symbol_identities,
            source=f"packed row {pack_id}:source_doc_index={doc.source_doc_index}",
        )
        concatenated_tokens.extend(doc.token_ids)
        doc_ids.extend([row_doc_id] * doc.token_count)
        source_doc_indices.append(doc.source_doc_index)
        source_doc_token_lengths.append(doc.token_count)
        source_platform_ids.append(list(doc.platform_ids))
        source_repo_stable_ids.append(doc.chronology.get(REPO_STABLE_ID_COLUMN))
        source_filepath_stable_ids.append(doc.chronology.get(FILEPATH_STABLE_ID_COLUMN))
        source_file_local_commit_indices.append(
            doc.chronology.get(FILE_LOCAL_COMMIT_INDEX_COLUMN)
        )
        source_pr_numbers.append(doc.chronology.get(PR_NUMBER_COLUMN))
        source_has_pr_discussions.append(
            bool(doc.chronology.get(HAS_PR_DISCUSSION_COLUMN, False))
        )
        source_pr_discussion_chars.append(
            int(doc.chronology.get(PR_DISCUSSION_CHARS_COLUMN) or 0)
        )
        source_pr_discussion_lines.append(
            int(doc.chronology.get(PR_DISCUSSION_LINES_COLUMN) or 0)
        )
        source_doc_types.append(doc.chronology.get(DOC_TYPE_COLUMN))
        source_header_fragment_kinds.append(
            doc.chronology.get(HEADER_FRAGMENT_KIND_COLUMN)
        )
        source_build_kinds.append(doc.chronology.get(BUILD_KIND_COLUMN))
        for source_column, token_column in (
            PACKED_ROWS_OBJECTIVE_SOURCE_TO_TOKEN_COLUMN.items()
        ):
            objective_source_ids[source_column].append(
                list(doc.objective_token_ids.get(token_column, []))
            )
        for entry in doc.source_identity_registry:
            identity_id = int(entry["source_identity_id"])
            normalized_entry = dict(entry)
            previous = source_identity_registry.get(identity_id)
            if previous is not None and previous != normalized_entry:
                raise ValueError(f"source identity uint64 collision for id {identity_id}")
            source_identity_registry[identity_id] = normalized_entry

        for column in PACKED_TOKEN_METADATA_COLUMNS:
            values = doc.token_meta[column]
            if column == TOKEN_SOURCE_DOC_IDS_COLUMN:
                token_meta_acc[column].extend(
                    int(value) if int(value) > 0 else 1
                    for value in (values or [0] * doc.token_count)
                )
            elif column == TOKEN_SOURCE_IDENTITY_IDS_COLUMN:
                token_meta_acc[column].extend(
                    int(value) if int(value) > 0 else doc.stable_source_id
                    for value in (values or [0] * doc.token_count)
                )
            elif values:
                token_meta_acc[column].extend(values)
            else:
                fill = PACKED_ROWS_DENSE_FALLBACK_FILL_VALUES.get(column, 0)
                token_meta_acc[column].extend([int(fill)] * doc.token_count)

        # Edge-remap into block-coordinate space. Every per-doc structure that
        # references positions WITHIN the doc must be shifted by this doc's start
        # position in the packed block so it points at the correct global slot:
        #   * chunk_starts / chunk_ends / changed_chunk_spans -> TOKEN coords
        #     (shift by token_offset, this doc's first-token index in the block).
        #   * call_edges / type_edges endpoints and changed_chunk_ids -> CHUNK
        #     indices (shift by chunk_offset, the count of chunks already emitted
        #     by preceding docs in the block).
        # The shift is applied unconditionally per kind (not gated on a single
        # array being truthy), so edges/spans are never silently dropped when a
        # doc carries graph metadata with a degenerate-but-valid chunk layout,
        # and chunk_offset always advances by this doc's real chunk count.
        chunk_starts.extend(token_offset + value for value in doc.chunk_starts)
        chunk_ends.extend(token_offset + value for value in doc.chunk_ends)
        chunk_kinds.extend(doc.chunk_kinds)
        chunk_dep_levels.extend(doc.chunk_dep_levels)
        call_edges.extend(
            {
                "from": chunk_offset + int(edge["from"]),
                "to": chunk_offset + int(edge["to"]),
            }
            for edge in doc.call_edges
        )
        type_edges.extend(
            {
                "from": chunk_offset + int(edge["from"]),
                "to": chunk_offset + int(edge["to"]),
            }
            for edge in doc.type_edges
        )
        for dest, source in (
            (domain_edges, doc.domain_edges),
            (build_edges, doc.build_edges),
            (shell_edges, doc.shell_edges),
            (diagnostic_edges, doc.diagnostic_edges),
            (cross_domain_edges, doc.cross_domain_edges),
        ):
            dest.extend(
                {
                    "from": token_offset + int(edge["from"]),
                    "to": token_offset + int(edge["to"]),
                    "kind": int(edge.get("kind", 0)),
                }
                for edge in source
            )
        changed_chunk_ids.extend(chunk_offset + value for value in doc.changed_chunk_ids)
        changed_chunk_spans.extend(
            {
                "start": token_offset + int(start),
                "end": token_offset + int(end),
            }
            for start, end in doc.changed_chunk_spans
        )
        chunk_offset += len(doc.chunk_starts)

        token_offset += doc.token_count

    valid_token_count = len(concatenated_tokens)
    # Whole-function invariant: a single document longer than target_length is
    # NEVER split. Such a doc is packed alone into its own block, and that row's
    # fixed width must grow to hold the whole doc so no token is truncated. For
    # all normal (fitting) rows row_length == target_length, leaving behavior and
    # output shape unchanged.
    row_length = max(target_length, valid_token_count)
    trained_token_count = sum(
        _loss_mask_for_packed_docs(doc_ids, target_length=valid_token_count)
    )
    slack_tokens = row_length - valid_token_count
    pad_doc_id = len(ordered_docs)
    used_symbol_ids = {
        int(value)
        for column in (
            TOKEN_SYMBOL_IDS_COLUMN,
            TOKEN_CALL_TARGETS_COLUMN,
            TOKEN_TYPE_REFS_COLUMN,
        )
        for value in token_meta_acc[column]
        if int(value) != 0
    }
    symbol_identity_registry.require_ids(
        used_symbol_ids,
        source=f"packed row {pack_id}",
    )

    row: dict[str, Any] = {
        PACK_ID_COLUMN: int(pack_id),
        VALID_TOKEN_COUNT_COLUMN: int(valid_token_count),
        TRAINED_TOKEN_COUNT_COLUMN: int(trained_token_count),
        NUM_DOCS_COLUMN: int(len(ordered_docs)),
        SLACK_TOKENS_COLUMN: int(slack_tokens),
        INPUT_IDS_COLUMN: _pad(concatenated_tokens, row_length, pad_value=pad_token_id),
        TARGET_IDS_COLUMN: _target_ids_for_packed_tokens(
            concatenated_tokens,
            target_length=row_length,
            pad_token_id=pad_token_id,
        ),
        LOSS_MASK_COLUMN: _loss_mask_for_packed_docs(
            doc_ids,
            target_length=row_length,
        ),
        DOC_IDS_COLUMN: _pad(doc_ids, row_length, pad_value=pad_doc_id),
        SOURCE_DOC_INDICES_COLUMN: source_doc_indices,
        SOURCE_DOC_TOKEN_LENGTHS_COLUMN: source_doc_token_lengths,
        SOURCE_PLATFORM_IDS_COLUMN: source_platform_ids,
        SOURCE_REPO_STABLE_IDS_COLUMN: source_repo_stable_ids,
        SOURCE_FILEPATH_STABLE_IDS_COLUMN: source_filepath_stable_ids,
        SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN: source_file_local_commit_indices,
        SOURCE_PR_NUMBERS_COLUMN: source_pr_numbers,
        SOURCE_HAS_PR_DISCUSSIONS_COLUMN: source_has_pr_discussions,
        SOURCE_PR_DISCUSSION_CHARS_COLUMN: source_pr_discussion_chars,
        SOURCE_PR_DISCUSSION_LINES_COLUMN: source_pr_discussion_lines,
        SOURCE_DOC_TYPES_COLUMN: source_doc_types,
        SOURCE_HEADER_FRAGMENT_KINDS_COLUMN: source_header_fragment_kinds,
        SOURCE_BUILD_KINDS_COLUMN: source_build_kinds,
        **objective_source_ids,
        ROW_PLATFORM_IDS_COLUMN: _merged_platform_ids_for_docs(ordered_docs),
        SYMBOL_IDENTITIES_COLUMN: symbol_identity_registry.records(used_symbol_ids),
        HAS_PR_DISCUSSION_COLUMN: any(source_has_pr_discussions),
        PR_DISCUSSION_CHARS_COLUMN: sum(source_pr_discussion_chars),
        PR_DISCUSSION_LINES_COLUMN: sum(source_pr_discussion_lines),
        CHANGED_CHUNK_IDS_COLUMN: changed_chunk_ids,
        CHANGED_CHUNK_SPANS_COLUMN: changed_chunk_spans,
        TOKEN_CHUNK_STARTS_COLUMN: chunk_starts,
        TOKEN_CHUNK_ENDS_COLUMN: chunk_ends,
        TOKEN_CHUNK_KINDS_COLUMN: chunk_kinds,
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: chunk_dep_levels,
        TOKEN_CALL_EDGES_COLUMN: call_edges,
        TOKEN_TYPE_EDGES_COLUMN: type_edges,
        TOKEN_DOMAIN_EDGES_COLUMN: domain_edges,
        TOKEN_BUILD_EDGES_COLUMN: build_edges,
        TOKEN_SHELL_EDGES_COLUMN: shell_edges,
        TOKEN_DIAGNOSTIC_EDGES_COLUMN: diagnostic_edges,
        TOKEN_CROSS_DOMAIN_EDGES_COLUMN: cross_domain_edges,
        SOURCE_IDENTITY_REGISTRY_COLUMN: list(source_identity_registry.values()),
    }
    for column in PACKED_ROW_PROVENANCE_COLUMNS:
        value = chronology.get(column) if chronology else None
        if column == PARENT_HASHES_COLUMN:
            row[column] = [str(item) for item in (value or [])]
        elif column in (PARENT_COUNT_COLUMN, FILE_LOCAL_COMMIT_INDEX_COLUMN):
            row[column] = int(value) if value is not None else None
        elif column in (
            IS_MERGE_COMMIT_COLUMN,
            HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN,
            HAS_RENAME_AMBIGUITY_COLUMN,
        ):
            row[column] = bool(value) if value is not None else None
        else:
            row[column] = value
    for column in PACKED_TOKEN_METADATA_COLUMNS:
        pad_value = int(PACKED_ROWS_DENSE_FALLBACK_FILL_VALUES.get(column, 0))
        row[column] = _pad(token_meta_acc[column], row_length, pad_value=pad_value)
    validate_source_identity_registry(
        row[SOURCE_IDENTITY_REGISTRY_COLUMN],
        referenced_ids=row[TOKEN_SOURCE_IDENTITY_IDS_COLUMN][:valid_token_count],
    )
    return row


def pack_documents(
    docs: list[NormalizedDoc],
    *,
    target_length: int,
    pad_token_id: int = 0,
    strategy: PackingStrategy = "best_fit",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pack whole documents into fixed-length rows with no truncation."""

    if target_length <= 0:
        raise ValueError("target_length must be > 0")
    if strategy not in {"best_fit", "sequential"}:
        raise ValueError("strategy must be 'best_fit' or 'sequential'")

    if strategy == "sequential":
        bins, overflow = _pack_docs_sequential(docs, target_length=target_length)
    else:
        bins, overflow = _pack_docs_best_fit(docs, target_length=target_length)

    packed_rows = [
        _materialize_packed_row(
            pack.docs,
            target_length=target_length,
            pad_token_id=pad_token_id,
            pack_id=pack_id,
        )
        for pack_id, pack in enumerate(bins)
    ]
    return packed_rows, overflow


def rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Convert materialized packed rows into a fixed-schema Arrow table."""

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                PACK_ID_COLUMN: int(row.get(PACK_ID_COLUMN, 0)),
                VALID_TOKEN_COUNT_COLUMN: int(row.get(VALID_TOKEN_COUNT_COLUMN, 0)),
                TRAINED_TOKEN_COUNT_COLUMN: int(row.get(TRAINED_TOKEN_COUNT_COLUMN, 0)),
                NUM_DOCS_COLUMN: int(row.get(NUM_DOCS_COLUMN, 0)),
                SLACK_TOKENS_COLUMN: int(row.get(SLACK_TOKENS_COLUMN, 0)),
                INPUT_IDS_COLUMN: _as_int_list(row.get(INPUT_IDS_COLUMN)),
                TARGET_IDS_COLUMN: _as_int_list(row.get(TARGET_IDS_COLUMN)),
                LOSS_MASK_COLUMN: _as_int_list(row.get(LOSS_MASK_COLUMN)),
                DOC_IDS_COLUMN: _as_int_list(row.get(DOC_IDS_COLUMN)),
                SOURCE_DOC_INDICES_COLUMN: _as_int_list(row.get(SOURCE_DOC_INDICES_COLUMN)),
                SOURCE_DOC_TOKEN_LENGTHS_COLUMN: _as_int_list(
                    row.get(SOURCE_DOC_TOKEN_LENGTHS_COLUMN)
                ),
                SOURCE_PLATFORM_IDS_COLUMN: [
                    _as_int_list(item)
                    for item in row.get(SOURCE_PLATFORM_IDS_COLUMN, [])
                ],
                SOURCE_REPO_STABLE_IDS_COLUMN: [
                    None if item is None else str(item)
                    for item in row.get(SOURCE_REPO_STABLE_IDS_COLUMN, [])
                ],
                SOURCE_FILEPATH_STABLE_IDS_COLUMN: [
                    None if item is None else str(item)
                    for item in row.get(SOURCE_FILEPATH_STABLE_IDS_COLUMN, [])
                ],
                SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN: [
                    int(item) if item is not None else None
                    for item in row.get(SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN, [])
                ],
                SOURCE_PR_NUMBERS_COLUMN: [
                    int(item) if item is not None else None
                    for item in row.get(SOURCE_PR_NUMBERS_COLUMN, [])
                ],
                SOURCE_HAS_PR_DISCUSSIONS_COLUMN: [
                    bool(item)
                    for item in row.get(SOURCE_HAS_PR_DISCUSSIONS_COLUMN, [])
                ],
                SOURCE_PR_DISCUSSION_CHARS_COLUMN: [
                    int(item)
                    for item in row.get(SOURCE_PR_DISCUSSION_CHARS_COLUMN, [])
                ],
                SOURCE_PR_DISCUSSION_LINES_COLUMN: [
                    int(item)
                    for item in row.get(SOURCE_PR_DISCUSSION_LINES_COLUMN, [])
                ],
                SOURCE_DOC_TYPES_COLUMN: [
                    None if item is None else str(item)
                    for item in row.get(SOURCE_DOC_TYPES_COLUMN, [])
                ],
                SOURCE_HEADER_FRAGMENT_KINDS_COLUMN: [
                    None if item is None else str(item)
                    for item in row.get(SOURCE_HEADER_FRAGMENT_KINDS_COLUMN, [])
                ],
                SOURCE_BUILD_KINDS_COLUMN: [
                    None if item is None else str(item)
                    for item in row.get(SOURCE_BUILD_KINDS_COLUMN, [])
                ],
                **{
                    source_column: [
                        _as_int_list(item)
                        for item in row.get(source_column, [])
                    ]
                    for source_column in PACKED_ROWS_OBJECTIVE_SOURCE_TO_TOKEN_COLUMN
                },
                ROW_PLATFORM_IDS_COLUMN: _as_int_list(row.get(ROW_PLATFORM_IDS_COLUMN)),
                SYMBOL_IDENTITIES_COLUMN: [
                    {
                        "symbol_id": int(item["symbol_id"]),
                        "symbol_key": str(item["symbol_key"]),
                    }
                    for item in row.get(SYMBOL_IDENTITIES_COLUMN, [])
                ],
                REPO_COLUMN: row.get(REPO_COLUMN),
                FILEPATH_COLUMN: row.get(FILEPATH_COLUMN),
                COMMIT_HASH_COLUMN: row.get(COMMIT_HASH_COLUMN),
                TIMESTAMP_COLUMN: row.get(TIMESTAMP_COLUMN),
                PR_NUMBER_COLUMN: _coerce_optional_int(row.get(PR_NUMBER_COLUMN)),
                HAS_PR_DISCUSSION_COLUMN: (
                    bool(row.get(HAS_PR_DISCUSSION_COLUMN))
                    if row.get(HAS_PR_DISCUSSION_COLUMN) is not None
                    else None
                ),
                PR_DISCUSSION_CHARS_COLUMN: int(
                    row.get(PR_DISCUSSION_CHARS_COLUMN) or 0
                ),
                PR_DISCUSSION_LINES_COLUMN: int(
                    row.get(PR_DISCUSSION_LINES_COLUMN) or 0
                ),
                PARENT_HASHES_COLUMN: [
                    str(item) for item in row.get(PARENT_HASHES_COLUMN, [])
                ],
                PARENT_COUNT_COLUMN: _coerce_optional_int(row.get(PARENT_COUNT_COLUMN)),
                IS_MERGE_COMMIT_COLUMN: (
                    bool(row.get(IS_MERGE_COMMIT_COLUMN))
                    if row.get(IS_MERGE_COMMIT_COLUMN) is not None
                    else None
                ),
                AUTHOR_TIMESTAMP_COLUMN: row.get(AUTHOR_TIMESTAMP_COLUMN),
                COMMIT_TIMESTAMP_COLUMN: row.get(COMMIT_TIMESTAMP_COLUMN),
                REPO_STABLE_ID_COLUMN: row.get(REPO_STABLE_ID_COLUMN),
                FILEPATH_STABLE_ID_COLUMN: row.get(FILEPATH_STABLE_ID_COLUMN),
                FILE_LOCAL_COMMIT_INDEX_COLUMN: _coerce_optional_int(
                    row.get(FILE_LOCAL_COMMIT_INDEX_COLUMN)
                ),
                HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN: (
                    bool(row.get(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN))
                    if row.get(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN) is not None
                    else None
                ),
                HAS_RENAME_AMBIGUITY_COLUMN: (
                    bool(row.get(HAS_RENAME_AMBIGUITY_COLUMN))
                    if row.get(HAS_RENAME_AMBIGUITY_COLUMN) is not None
                    else None
                ),
                TOKEN_PLATFORM_IDS_COLUMN: _as_int_list(
                    row.get(TOKEN_PLATFORM_IDS_COLUMN)
                ),
                TOKEN_STRUCTURE_IDS_COLUMN: _as_int_list(
                    row.get(TOKEN_STRUCTURE_IDS_COLUMN)
                ),
                TOKEN_DEP_LEVELS_COLUMN: _as_int_list(row.get(TOKEN_DEP_LEVELS_COLUMN)),
                TOKEN_AST_DEPTH_COLUMN: _as_int_list(row.get(TOKEN_AST_DEPTH_COLUMN)),
                TOKEN_SIBLING_INDEX_COLUMN: _as_int_list(
                    row.get(TOKEN_SIBLING_INDEX_COLUMN)
                ),
                TOKEN_AST_NODE_TYPE_COLUMN: _as_int_list(
                    row.get(TOKEN_AST_NODE_TYPE_COLUMN)
                ),
                TOKEN_DOMAIN_IDS_COLUMN: _as_int_list(row.get(TOKEN_DOMAIN_IDS_COLUMN)),
                TOKEN_ROLE_IDS_COLUMN: _as_int_list(row.get(TOKEN_ROLE_IDS_COLUMN)),
                TOKEN_ENTITY_IDS_COLUMN: _as_int_list(row.get(TOKEN_ENTITY_IDS_COLUMN)),
                TOKEN_SCOPE_IDS_COLUMN: _as_int_list(row.get(TOKEN_SCOPE_IDS_COLUMN)),
                TOKEN_SOURCE_DOC_IDS_COLUMN: _as_int_list(
                    row.get(TOKEN_SOURCE_DOC_IDS_COLUMN)
                ),
                TOKEN_SOURCE_IDENTITY_IDS_COLUMN: _as_int_list(
                    row.get(TOKEN_SOURCE_IDENTITY_IDS_COLUMN)
                ),
                TOKEN_CONFIDENCE_IDS_COLUMN: _as_int_list(
                    row.get(TOKEN_CONFIDENCE_IDS_COLUMN)
                ),
                TOKEN_SYMBOL_IDS_COLUMN: _as_int_list(row.get(TOKEN_SYMBOL_IDS_COLUMN)),
                TOKEN_CALL_TARGETS_COLUMN: _as_int_list(
                    row.get(TOKEN_CALL_TARGETS_COLUMN)
                ),
                TOKEN_TYPE_REFS_COLUMN: _as_int_list(row.get(TOKEN_TYPE_REFS_COLUMN)),
                TOKEN_DEF_USE_COLUMN: _as_int_list(row.get(TOKEN_DEF_USE_COLUMN)),
                TOKEN_CHANGE_MASK_PRE_COLUMN: _as_int_list(
                    row.get(TOKEN_CHANGE_MASK_PRE_COLUMN)
                ),
                TOKEN_CHANGE_MASK_POST_COLUMN: _as_int_list(
                    row.get(TOKEN_CHANGE_MASK_POST_COLUMN)
                ),
                HUNK_ID_PER_TOKEN_COLUMN: _as_int_list(row.get(HUNK_ID_PER_TOKEN_COLUMN)),
                EDIT_OP_PER_TOKEN_COLUMN: _as_int_list(row.get(EDIT_OP_PER_TOKEN_COLUMN)),
                TOKEN_CHUNK_STARTS_COLUMN: _as_int_list(
                    row.get(TOKEN_CHUNK_STARTS_COLUMN)
                ),
                TOKEN_CHUNK_ENDS_COLUMN: _as_int_list(row.get(TOKEN_CHUNK_ENDS_COLUMN)),
                TOKEN_CHUNK_KINDS_COLUMN: _as_int_list(row.get(TOKEN_CHUNK_KINDS_COLUMN)),
                TOKEN_CHUNK_DEP_LEVELS_COLUMN: _as_int_list(
                    row.get(TOKEN_CHUNK_DEP_LEVELS_COLUMN)
                ),
                CHANGED_CHUNK_IDS_COLUMN: _as_int_list(row.get(CHANGED_CHUNK_IDS_COLUMN)),
                CHANGED_CHUNK_SPANS_COLUMN: [
                    {
                        "start": int(span["start"]),
                        "end": int(span["end"]),
                    }
                    for span in row.get(CHANGED_CHUNK_SPANS_COLUMN, [])
                ],
                TOKEN_CALL_EDGES_COLUMN: [
                    {
                        "from": int(edge["from"]),
                        "to": int(edge["to"]),
                    }
                    for edge in row.get(TOKEN_CALL_EDGES_COLUMN, [])
                ],
                TOKEN_TYPE_EDGES_COLUMN: [
                    {
                        "from": int(edge["from"]),
                        "to": int(edge["to"]),
                    }
                    for edge in row.get(TOKEN_TYPE_EDGES_COLUMN, [])
                ],
                TOKEN_DOMAIN_EDGES_COLUMN: [
                    {
                        "from": int(edge["from"]),
                        "to": int(edge["to"]),
                        "kind": int(edge["kind"]),
                    }
                    for edge in row.get(TOKEN_DOMAIN_EDGES_COLUMN, [])
                ],
                TOKEN_BUILD_EDGES_COLUMN: [
                    {
                        "from": int(edge["from"]),
                        "to": int(edge["to"]),
                        "kind": int(edge["kind"]),
                    }
                    for edge in row.get(TOKEN_BUILD_EDGES_COLUMN, [])
                ],
                TOKEN_SHELL_EDGES_COLUMN: [
                    {
                        "from": int(edge["from"]),
                        "to": int(edge["to"]),
                        "kind": int(edge["kind"]),
                    }
                    for edge in row.get(TOKEN_SHELL_EDGES_COLUMN, [])
                ],
                TOKEN_DIAGNOSTIC_EDGES_COLUMN: [
                    {
                        "from": int(edge["from"]),
                        "to": int(edge["to"]),
                        "kind": int(edge["kind"]),
                    }
                    for edge in row.get(TOKEN_DIAGNOSTIC_EDGES_COLUMN, [])
                ],
                TOKEN_CROSS_DOMAIN_EDGES_COLUMN: [
                    {
                        "from": int(edge["from"]),
                        "to": int(edge["to"]),
                        "kind": int(edge["kind"]),
                    }
                    for edge in row.get(TOKEN_CROSS_DOMAIN_EDGES_COLUMN, [])
                ],
                SOURCE_IDENTITY_REGISTRY_COLUMN: [
                    {
                        "source_identity_id": int(entry["source_identity_id"]),
                        "canonical_sha256": str(entry["canonical_sha256"]),
                        "source": str(entry["source"]),
                    }
                    for entry in row.get(SOURCE_IDENTITY_REGISTRY_COLUMN, [])
                ],
            }
        )
    return pa.Table.from_pylist(normalized_rows, schema=PACKED_ROW_OUTPUT_SCHEMA)


def _write_packed_rows(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
    *,
    row_group_size: int,
) -> None:
    for start in range(0, len(rows), row_group_size):
        writer.write_table(
            rows_to_table(rows[start : start + row_group_size]),
            row_group_size=row_group_size,
        )


def _empty_overflow_table() -> pa.Table:
    return pa.Table.from_pylist(
        [],
        schema=pa.schema(
            [
                pa.field("source_doc_index", pa.int64()),
                pa.field("token_count", pa.int32()),
                pa.field(TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
                pa.field(REPO_COLUMN, pa.string()),
                pa.field(FILEPATH_COLUMN, pa.string()),
                pa.field(COMMIT_HASH_COLUMN, pa.string()),
            ]
        ),
    )


def write_overflow_records(
    overflow_records: list[dict[str, Any]],
    output_path: str | os.PathLike[str],
) -> None:
    """Write overflow docs as JSONL or parquet."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for record in overflow_records:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")
        return

    if overflow_records:
        table = pa.Table.from_pylist(overflow_records)
    else:
        table = _empty_overflow_table()
    pq.write_table(table, path)


def _pack_parquet_dataset_to_paths(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    target_length: int,
    pad_token_id: int = 0,
    strategy: PackingStrategy = "best_fit",
    overflow_output: str | os.PathLike[str] | None = None,
    row_group_size: int = 128,
    pack_token_window: int = DEFAULT_PACK_TOKEN_WINDOW,
    input_batch_size: int = 1024,
) -> dict[str, int]:
    """Pack a parquet dataset end to end and write packed rows."""

    if pack_token_window <= 0:
        raise ValueError("pack_token_window must be > 0")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be > 0")
    if input_batch_size <= 0:
        raise ValueError("input_batch_size must be > 0")

    _require_symbol_identity_schema(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    input_docs = 0
    packed_rows_count = 0
    overflow: list[dict[str, Any]] = []
    window_docs: list[NormalizedDoc] = []
    window_tokens = 0
    signature_to_id: dict[str, int] = {}
    writer: pq.ParquetWriter | None = None

    def flush_window() -> None:
        nonlocal packed_rows_count, window_docs, window_tokens, writer
        if not window_docs:
            return
        packed_rows, window_overflow = pack_documents(
            window_docs,
            target_length=target_length,
            pad_token_id=pad_token_id,
            strategy=strategy,
        )
        for row in packed_rows:
            row[PACK_ID_COLUMN] = packed_rows_count + int(row.get(PACK_ID_COLUMN, 0))
        if packed_rows:
            if writer is None:
                writer = pq.ParquetWriter(output, PACKED_ROW_OUTPUT_SCHEMA)
            _write_packed_rows(
                writer,
                packed_rows,
                row_group_size=row_group_size,
            )
            packed_rows_count += len(packed_rows)
        overflow.extend(window_overflow)
        window_docs = []
        window_tokens = 0

    try:
        for doc in iter_tokenized_documents(
            input_path,
            signature_to_id=signature_to_id,
            input_batch_size=input_batch_size,
        ):
            input_docs += 1
            window_docs.append(doc)
            window_tokens += doc.token_count
            if window_tokens >= pack_token_window:
                flush_window()
        flush_window()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(rows_to_table([]), output, row_group_size=row_group_size)

    if overflow_output is not None:
        write_overflow_records(overflow, overflow_output)

    return {
        "input_docs": input_docs,
        "packed_rows": packed_rows_count,
        "overflow_docs": len(overflow),
    }


def pack_parquet_dataset(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    target_length: int,
    pad_token_id: int = 0,
    strategy: PackingStrategy = "best_fit",
    overflow_output: str | os.PathLike[str] | None = None,
    row_group_size: int = 128,
    pack_token_window: int = DEFAULT_PACK_TOKEN_WINDOW,
    input_batch_size: int = 1024,
) -> dict[str, int]:
    """Stage every artifact and publish the primary packed parquet last."""

    with ExitStack() as stack:
        staged_output = stack.enter_context(atomic_output_file(output_path))
        staged_overflow = None
        if overflow_output is not None:
            staged_overflow = stack.enter_context(
                atomic_output_file(overflow_output)
            )
        return _pack_parquet_dataset_to_paths(
            input_path,
            staged_output,
            target_length=target_length,
            pad_token_id=pad_token_id,
            strategy=strategy,
            overflow_output=staged_overflow,
            row_group_size=row_group_size,
            pack_token_window=pack_token_window,
            input_batch_size=input_batch_size,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline no-crop packer for tokenized enriched parquet rows."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input parquet file or directory of per-document tokenized parquet shards.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output parquet path for fixed-length packed rows.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        required=True,
        help="Fixed packed row length, typically 4096 or 8192.",
    )
    parser.add_argument(
        "--pad-token-id",
        type=int,
        default=0,
        help="Token ID used for right padding (default: 0).",
    )
    parser.add_argument(
        "--strategy",
        choices=("best_fit", "sequential"),
        default="best_fit",
        help="Packing strategy for whole-document rows (default: best_fit).",
    )
    parser.add_argument(
        "--overflow-output",
        default="",
        help="Optional JSONL or parquet path for oversized docs that did not fit.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=128,
        help="Output parquet row group size (default: 128).",
    )
    parser.add_argument(
        "--pack-token-window",
        type=int,
        default=DEFAULT_PACK_TOKEN_WINDOW,
        help=(
            "Maximum real tokens to hold while best-fit packing before writing "
            f"a bounded parquet window (default: {DEFAULT_PACK_TOKEN_WINDOW})."
        ),
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=1024,
        help="Input parquet row batch size for streaming normalization (default: 1024).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = pack_parquet_dataset(
        args.input,
        args.output,
        target_length=args.target_length,
        pad_token_id=args.pad_token_id,
        strategy=args.strategy,
        overflow_output=args.overflow_output or None,
        row_group_size=args.row_group_size,
        pack_token_window=args.pack_token_window,
        input_batch_size=args.input_batch_size,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DOC_IDS_COLUMN",
    "INPUT_IDS_COLUMN",
    "LOSS_MASK_COLUMN",
    "NUM_DOCS_COLUMN",
    "PACKED_ROW_MACRO_ROUTES_METADATA_KEY",
    "PACKED_ROW_MACRO_ROUTES_VERSION",
    "REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION",
    "SYMBOL_IDENTITY_SCHEMA_METADATA_KEY",
    "PACK_ID_COLUMN",
    "SOURCE_DOC_INDICES_COLUMN",
    "TARGET_IDS_COLUMN",
    "VALID_TOKEN_COUNT_COLUMN",
    "NormalizedDoc",
    "normalize_document_record",
    "iter_tokenized_documents",
    "pack_documents",
    "pack_parquet_dataset",
    "read_tokenized_documents",
    "rows_to_table",
    "write_overflow_records",
]
