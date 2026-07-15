"""Shared schema constants for token-level enriched parquet columns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import SupportsInt, TypeAlias, TypedDict, TypeGuard

import numpy as np


IntLike: TypeAlias = SupportsInt | str | bytes | bytearray
ScalarLike: TypeAlias = int | float | str | bool
ArrayLikeSequence: TypeAlias = list[object] | tuple[object, ...] | np.ndarray
TokenValueSequence: TypeAlias = Sequence[object] | np.ndarray


class SpanMapping(TypedDict):
    start: object
    end: object


TokenMetadataRow: TypeAlias = Mapping[str, object]
SpanLike: TypeAlias = SpanMapping | Sequence[object] | np.ndarray


def _is_sequence(val: object) -> TypeGuard[ArrayLikeSequence]:
    """Return True for list or numpy array (sequence-like, not str)."""
    return isinstance(val, (list, tuple, np.ndarray))


def _is_token_value_sequence(val: object) -> TypeGuard[TokenValueSequence]:
    """Return True for token-aligned sequence payloads, excluding strings."""
    return isinstance(val, (list, tuple, np.ndarray))


def _is_non_empty_token_sequence(val: object) -> TypeGuard[TokenValueSequence]:
    return _is_token_value_sequence(val) and len(val) > 0


def _as_int(value: object) -> int | None:
    try:
        if isinstance(value, (str, bytes, bytearray, int, np.integer, np.bool_)):
            return int(value)
        if isinstance(value, float):
            return int(value)
        return None
    except (TypeError, ValueError):
        return None


def _extract_span_bounds(span: object) -> tuple[int, int] | None:
    if isinstance(span, Mapping):
        start_i = _as_int(span.get("start"))
        end_i = _as_int(span.get("end"))
    elif _is_sequence(span) and len(span) == 2:
        start_i = _as_int(span[0])
        end_i = _as_int(span[1])
    else:
        return None
    if start_i is None or end_i is None:
        return None
    return start_i, end_i

TOKEN_IDS_COLUMN = "token_ids"
PLATFORM_IDS_COLUMN = "platform_ids"
TOKEN_STRUCTURE_IDS_COLUMN = "token_structure_ids"
TOKEN_DEP_LEVELS_COLUMN = "token_dep_levels"
TOKEN_AST_DEPTH_COLUMN = "token_ast_depth"
TOKEN_SIBLING_INDEX_COLUMN = "token_sibling_index"
TOKEN_AST_NODE_TYPE_COLUMN = "token_ast_node_type"
TOKEN_SYMBOL_IDS_COLUMN = "token_symbol_ids"
TOKEN_CALL_TARGETS_COLUMN = "token_call_targets"
TOKEN_TYPE_REFS_COLUMN = "token_type_refs"
TOKEN_DEF_USE_COLUMN = "token_def_use"
TOKEN_CHUNK_STARTS_COLUMN = "token_chunk_starts"
TOKEN_CHUNK_ENDS_COLUMN = "token_chunk_ends"
TOKEN_CHUNK_KINDS_COLUMN = "token_chunk_kinds"
TOKEN_CHUNK_DEP_LEVELS_COLUMN = "token_chunk_dep_levels"
TOKEN_CALL_EDGES_COLUMN = "token_call_edges"
TOKEN_TYPE_EDGES_COLUMN = "token_type_edges"
TOKEN_DOMAIN_IDS_COLUMN = "token_domain_ids"
TOKEN_ROLE_IDS_COLUMN = "token_role_ids"
TOKEN_ENTITY_IDS_COLUMN = "token_entity_ids"
TOKEN_SCOPE_IDS_COLUMN = "token_scope_ids"
TOKEN_SOURCE_DOC_IDS_COLUMN = "token_source_doc_ids"
TOKEN_SOURCE_IDENTITY_IDS_COLUMN = "token_source_identity_ids"
TOKEN_CONFIDENCE_IDS_COLUMN = "token_confidence_ids"
TOKEN_DOMAIN_EDGES_COLUMN = "token_domain_edges"
TOKEN_BUILD_EDGES_COLUMN = "token_build_edges"
TOKEN_SHELL_EDGES_COLUMN = "token_shell_edges"
TOKEN_DIAGNOSTIC_EDGES_COLUMN = "token_diagnostic_edges"
TOKEN_CROSS_DOMAIN_EDGES_COLUMN = "token_cross_domain_edges"
TOKEN_CHANGE_MASK_PRE_COLUMN = "token_change_mask_pre"
TOKEN_CHANGE_MASK_POST_COLUMN = "token_change_mask_post"
CHANGED_CHUNK_IDS_COLUMN = "changed_chunk_ids"
CHANGED_CHUNK_SPANS_COLUMN = "changed_chunk_spans"
HUNK_ID_PER_TOKEN_COLUMN = "hunk_id_per_token"
EDIT_OP_PER_TOKEN_COLUMN = "edit_op_per_token"
SOURCE_IDENTITY_REGISTRY_COLUMN = "source_identity_registry"

# Objective source sections. Text fields are emitted by the authoritative
# upstream extractor; token fields are materialized directly from those values.
IFIM_INSTRUCTION_TEXT_COLUMN = "ifim_instruction_text"
COMMIT_MSG_TEXT_COLUMN = "commit_msg_text"
PRE_TEXT_COLUMN = "pre_text"
POST_TEXT_COLUMN = "post_text"
DIFF_TEXT_COLUMN = "diff_text"
IFIM_INSTRUCTION_TOKEN_IDS_COLUMN = "ifim_instruction_token_ids"
COMMIT_MSG_TOKEN_IDS_COLUMN = "commit_msg_token_ids"
PRE_TOKEN_IDS_COLUMN = "pre_token_ids"
POST_TOKEN_IDS_COLUMN = "post_token_ids"
DIFF_TOKEN_IDS_COLUMN = "diff_token_ids"

DOC_TYPE_COLUMN = "doc_type"
HEADER_FRAGMENT_KIND_COLUMN = "header_fragment_kind"

REPO_COLUMN = "repo"
FILEPATH_COLUMN = "filepath"
COMMIT_HASH_COLUMN = "commit_hash"
TIMESTAMP_COLUMN = "timestamp"
PR_NUMBER_COLUMN = "pr_number"
HAS_PR_DISCUSSION_COLUMN = "has_pr_discussion"
PR_DISCUSSION_CHARS_COLUMN = "pr_discussion_chars"
PR_DISCUSSION_LINES_COLUMN = "pr_discussion_lines"
PARENT_HASHES_COLUMN = "parent_hashes"
PARENT_COUNT_COLUMN = "parent_count"
IS_MERGE_COMMIT_COLUMN = "is_merge_commit"
AUTHOR_TIMESTAMP_COLUMN = "author_timestamp"
COMMIT_TIMESTAMP_COLUMN = "commit_timestamp"
REPO_STABLE_ID_COLUMN = "repo_stable_id"
FILEPATH_STABLE_ID_COLUMN = "filepath_stable_id"
FILE_LOCAL_COMMIT_INDEX_COLUMN = "file_local_commit_index"
HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN = "has_ambiguous_reconstruction"
HAS_RENAME_AMBIGUITY_COLUMN = "has_rename_ambiguity"

TOKENIZED_ENRICHED_STRUCTURE_COLUMNS = (
    TOKEN_STRUCTURE_IDS_COLUMN,
    TOKEN_DEP_LEVELS_COLUMN,
)

TOKENIZED_ENRICHED_AST_COLUMNS = (
    TOKEN_AST_DEPTH_COLUMN,
    TOKEN_SIBLING_INDEX_COLUMN,
    TOKEN_AST_NODE_TYPE_COLUMN,
)

TOKENIZED_ENRICHED_SEMANTIC_COLUMNS = (
    TOKEN_SYMBOL_IDS_COLUMN,
    TOKEN_CALL_TARGETS_COLUMN,
    TOKEN_TYPE_REFS_COLUMN,
    TOKEN_DEF_USE_COLUMN,
)

TOKENIZED_ENRICHED_CHUNK_COLUMNS = (
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
)

TOKENIZED_ENRICHED_GRAPH_COLUMNS = (
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
)

TOKENIZED_ENRICHED_DOMAIN_TOKEN_COLUMNS = (
    TOKEN_DOMAIN_IDS_COLUMN,
    TOKEN_ROLE_IDS_COLUMN,
    TOKEN_ENTITY_IDS_COLUMN,
    TOKEN_SCOPE_IDS_COLUMN,
    TOKEN_SOURCE_DOC_IDS_COLUMN,
    TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
    TOKEN_CONFIDENCE_IDS_COLUMN,
)

TOKENIZED_ENRICHED_DOMAIN_GRAPH_COLUMNS = (
    TOKEN_DOMAIN_EDGES_COLUMN,
    TOKEN_BUILD_EDGES_COLUMN,
    TOKEN_SHELL_EDGES_COLUMN,
    TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
)

TOKENIZED_ENRICHED_TEMPORAL_TOKEN_COLUMNS = (
    TOKEN_CHANGE_MASK_PRE_COLUMN,
    TOKEN_CHANGE_MASK_POST_COLUMN,
    HUNK_ID_PER_TOKEN_COLUMN,
    EDIT_OP_PER_TOKEN_COLUMN,
)

TOKENIZED_ENRICHED_TEMPORAL_CHUNK_COLUMNS = (
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
)

TOKENIZED_ENRICHED_TEMPORAL_COLUMNS = (
    *TOKENIZED_ENRICHED_TEMPORAL_TOKEN_COLUMNS,
    *TOKENIZED_ENRICHED_TEMPORAL_CHUNK_COLUMNS,
)

TOKENIZED_ENRICHED_OBJECTIVE_COLUMNS = (
    IFIM_INSTRUCTION_TOKEN_IDS_COLUMN,
    COMMIT_MSG_TOKEN_IDS_COLUMN,
    PRE_TOKEN_IDS_COLUMN,
    POST_TOKEN_IDS_COLUMN,
    DIFF_TOKEN_IDS_COLUMN,
)

RAW_COMMIT_CHRONOLOGY_COLUMNS = (
    REPO_COLUMN,
    FILEPATH_COLUMN,
    COMMIT_HASH_COLUMN,
    TIMESTAMP_COLUMN,
    PR_NUMBER_COLUMN,
    PARENT_HASHES_COLUMN,
    PARENT_COUNT_COLUMN,
    IS_MERGE_COMMIT_COLUMN,
    AUTHOR_TIMESTAMP_COLUMN,
    COMMIT_TIMESTAMP_COLUMN,
    REPO_STABLE_ID_COLUMN,
    FILEPATH_STABLE_ID_COLUMN,
    FILE_LOCAL_COMMIT_INDEX_COLUMN,
    HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN,
    HAS_RENAME_AMBIGUITY_COLUMN,
)

TOKENIZED_ENRICHED_COLUMNS = (
    TOKEN_IDS_COLUMN,
    PLATFORM_IDS_COLUMN,
    *TOKENIZED_ENRICHED_STRUCTURE_COLUMNS,
    *TOKENIZED_ENRICHED_AST_COLUMNS,
    *TOKENIZED_ENRICHED_SEMANTIC_COLUMNS,
    *TOKENIZED_ENRICHED_CHUNK_COLUMNS,
    *TOKENIZED_ENRICHED_GRAPH_COLUMNS,
    *TOKENIZED_ENRICHED_DOMAIN_TOKEN_COLUMNS,
    *TOKENIZED_ENRICHED_DOMAIN_GRAPH_COLUMNS,
    *TOKENIZED_ENRICHED_TEMPORAL_COLUMNS,
    *TOKENIZED_ENRICHED_OBJECTIVE_COLUMNS,
    SOURCE_IDENTITY_REGISTRY_COLUMN,
)


def has_materialized_token_metadata(
    row: TokenMetadataRow,
    *,
    need_structure_meta: bool,
    need_ast_metadata: bool,
    need_chunk_relations: bool,
    need_semantic_metadata: bool = False,
    need_temporal_metadata: bool = False,
    token_count: int | None = None,
    require_token_ids: bool = True,
) -> bool:
    """Return True when row-local token metadata is present and shape-valid.

    Missing or empty token metadata is treated as "not materialized" so callers
    can fall back to legacy char-level derivation or omit optional tensors.
    Shape-invalid non-empty metadata is rejected by returning False here; packed
    row consumers may still choose to fail closed with a more specific error.
    """

    if token_count is None:
        token_ids = row.get(TOKEN_IDS_COLUMN)
        if not _is_non_empty_token_sequence(token_ids):
            return False
        token_count = len(token_ids)
    else:
        normalized_token_count = _as_int(token_count)
        if normalized_token_count is None:
            return False
        token_count = normalized_token_count
        if token_count <= 0:
            return False
        if require_token_ids:
            token_ids = row.get(TOKEN_IDS_COLUMN)
            if not _is_token_value_sequence(token_ids) or len(token_ids) != token_count:
                return False

    if need_structure_meta:
        for column in TOKENIZED_ENRICHED_STRUCTURE_COLUMNS:
            values = row.get(column)
            if not _is_token_value_sequence(values) or len(values) != token_count:
                return False

    if need_ast_metadata:
        for column in TOKENIZED_ENRICHED_AST_COLUMNS:
            values = row.get(column)
            if not _is_token_value_sequence(values) or len(values) != token_count:
                return False

    if need_semantic_metadata:
        for column in TOKENIZED_ENRICHED_SEMANTIC_COLUMNS:
            values = row.get(column)
            if not _is_token_value_sequence(values) or len(values) != token_count:
                return False

    if need_chunk_relations:
        chunk_columns = []
        for column in TOKENIZED_ENRICHED_CHUNK_COLUMNS:
            values = row.get(column)
            if not _is_token_value_sequence(values):
                return False
            chunk_columns.append(values)
        starts, ends, kinds, dep_levels = chunk_columns
        if not (len(starts) == len(ends) == len(kinds) == len(dep_levels)):
            return False
        if row.get("chunk_boundaries") and len(starts) == 0:
            return False
        prev_start: int | None = None
        prev_end: int | None = None
        for start, end in zip(starts, ends):
            start_i = _as_int(start)
            end_i = _as_int(end)
            if start_i is None or end_i is None:
                return False
            if not (0 <= start_i < end_i <= token_count):
                return False
            if prev_start is not None and start_i <= prev_start:
                return False
            if prev_end is not None and end_i <= prev_end:
                return False
            prev_start = start_i
            prev_end = end_i

    if need_temporal_metadata:
        for column in TOKENIZED_ENRICHED_TEMPORAL_TOKEN_COLUMNS:
            values = row.get(column)
            if not _is_token_value_sequence(values) or len(values) != token_count:
                return False

        changed_chunk_ids = row.get(CHANGED_CHUNK_IDS_COLUMN)
        changed_chunk_spans = row.get(CHANGED_CHUNK_SPANS_COLUMN)
        if not _is_token_value_sequence(changed_chunk_ids) or not _is_token_value_sequence(changed_chunk_spans):
            return False
        if len(changed_chunk_ids) != len(changed_chunk_spans):
            return False
        for span in changed_chunk_spans:
            bounds = _extract_span_bounds(span)
            if bounds is None:
                return False
            start_i, end_i = bounds
            if not (0 <= start_i < end_i <= token_count):
                return False

    return True


__all__ = [
    "TOKEN_IDS_COLUMN",
    "PLATFORM_IDS_COLUMN",
    "TOKEN_STRUCTURE_IDS_COLUMN",
    "TOKEN_DEP_LEVELS_COLUMN",
    "TOKEN_AST_DEPTH_COLUMN",
    "TOKEN_SIBLING_INDEX_COLUMN",
    "TOKEN_AST_NODE_TYPE_COLUMN",
    "TOKEN_SYMBOL_IDS_COLUMN",
    "TOKEN_CALL_TARGETS_COLUMN",
    "TOKEN_TYPE_REFS_COLUMN",
    "TOKEN_DEF_USE_COLUMN",
    "TOKEN_CHUNK_STARTS_COLUMN",
    "TOKEN_CHUNK_ENDS_COLUMN",
    "TOKEN_CHUNK_KINDS_COLUMN",
    "TOKEN_CHUNK_DEP_LEVELS_COLUMN",
    "TOKEN_CALL_EDGES_COLUMN",
    "TOKEN_TYPE_EDGES_COLUMN",
    "TOKEN_DOMAIN_IDS_COLUMN",
    "TOKEN_ROLE_IDS_COLUMN",
    "TOKEN_ENTITY_IDS_COLUMN",
    "TOKEN_SCOPE_IDS_COLUMN",
    "TOKEN_SOURCE_DOC_IDS_COLUMN",
    "TOKEN_SOURCE_IDENTITY_IDS_COLUMN",
    "TOKEN_CONFIDENCE_IDS_COLUMN",
    "TOKEN_DOMAIN_EDGES_COLUMN",
    "TOKEN_BUILD_EDGES_COLUMN",
    "TOKEN_SHELL_EDGES_COLUMN",
    "TOKEN_DIAGNOSTIC_EDGES_COLUMN",
    "TOKEN_CROSS_DOMAIN_EDGES_COLUMN",
    "TOKEN_CHANGE_MASK_PRE_COLUMN",
    "TOKEN_CHANGE_MASK_POST_COLUMN",
    "CHANGED_CHUNK_IDS_COLUMN",
    "CHANGED_CHUNK_SPANS_COLUMN",
    "HUNK_ID_PER_TOKEN_COLUMN",
    "EDIT_OP_PER_TOKEN_COLUMN",
    "IFIM_INSTRUCTION_TEXT_COLUMN",
    "COMMIT_MSG_TEXT_COLUMN",
    "PRE_TEXT_COLUMN",
    "POST_TEXT_COLUMN",
    "DIFF_TEXT_COLUMN",
    "IFIM_INSTRUCTION_TOKEN_IDS_COLUMN",
    "COMMIT_MSG_TOKEN_IDS_COLUMN",
    "PRE_TOKEN_IDS_COLUMN",
    "POST_TOKEN_IDS_COLUMN",
    "DIFF_TOKEN_IDS_COLUMN",
    "SOURCE_IDENTITY_REGISTRY_COLUMN",
    "DOC_TYPE_COLUMN",
    "HEADER_FRAGMENT_KIND_COLUMN",
    "REPO_COLUMN",
    "FILEPATH_COLUMN",
    "COMMIT_HASH_COLUMN",
    "TIMESTAMP_COLUMN",
    "PR_NUMBER_COLUMN",
    "HAS_PR_DISCUSSION_COLUMN",
    "PR_DISCUSSION_CHARS_COLUMN",
    "PR_DISCUSSION_LINES_COLUMN",
    "PARENT_HASHES_COLUMN",
    "PARENT_COUNT_COLUMN",
    "IS_MERGE_COMMIT_COLUMN",
    "AUTHOR_TIMESTAMP_COLUMN",
    "COMMIT_TIMESTAMP_COLUMN",
    "REPO_STABLE_ID_COLUMN",
    "FILEPATH_STABLE_ID_COLUMN",
    "FILE_LOCAL_COMMIT_INDEX_COLUMN",
    "HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN",
    "HAS_RENAME_AMBIGUITY_COLUMN",
    "TOKENIZED_ENRICHED_STRUCTURE_COLUMNS",
    "TOKENIZED_ENRICHED_AST_COLUMNS",
    "TOKENIZED_ENRICHED_SEMANTIC_COLUMNS",
    "TOKENIZED_ENRICHED_CHUNK_COLUMNS",
    "TOKENIZED_ENRICHED_GRAPH_COLUMNS",
    "TOKENIZED_ENRICHED_DOMAIN_TOKEN_COLUMNS",
    "TOKENIZED_ENRICHED_DOMAIN_GRAPH_COLUMNS",
    "TOKENIZED_ENRICHED_TEMPORAL_TOKEN_COLUMNS",
    "TOKENIZED_ENRICHED_TEMPORAL_CHUNK_COLUMNS",
    "TOKENIZED_ENRICHED_TEMPORAL_COLUMNS",
    "TOKENIZED_ENRICHED_OBJECTIVE_COLUMNS",
    "RAW_COMMIT_CHRONOLOGY_COLUMNS",
    "TOKENIZED_ENRICHED_COLUMNS",
    "TokenMetadataRow",
    "SpanMapping",
    "has_materialized_token_metadata",
]
