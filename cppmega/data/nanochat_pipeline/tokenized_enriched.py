"""Helpers for offline token-level enriched parquet materialization."""

from __future__ import annotations

import inspect
import json
from typing import Any, Sequence, cast

from cppmega.tokenizer.cpp_tokenizer import normalize_whitespace_with_offsets
from cppmega.data.domain_schema import (
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
    canonicalize_domain_edge_fields,
    delimiter_token_ids,
    domain_edge_family,
    normalize_domain_edge_record,
    normalize_embedded_domain_spans,
)
from cppmega.data.source_identity import (
    normalize_positive_source_ids,
    normalize_row_local_doc_ids,
    source_identity,
    validate_source_identity_registry,
)

from cppmega.data.nanochat_pipeline.tokenized_enriched_schema import (
    COMMIT_MSG_TEXT_COLUMN,
    COMMIT_MSG_TOKEN_IDS_COLUMN,
    DIFF_TEXT_COLUMN,
    DIFF_TOKEN_IDS_COLUMN,
    IFIM_INSTRUCTION_TEXT_COLUMN,
    IFIM_INSTRUCTION_TOKEN_IDS_COLUMN,
    PLATFORM_IDS_COLUMN,
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
    EDIT_OP_PER_TOKEN_COLUMN,
    HUNK_ID_PER_TOKEN_COLUMN,
    POST_TEXT_COLUMN,
    POST_TOKEN_IDS_COLUMN,
    PRE_TEXT_COLUMN,
    PRE_TOKEN_IDS_COLUMN,
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
    TOKEN_CHANGE_MASK_POST_COLUMN,
    TOKEN_CHANGE_MASK_PRE_COLUMN,
    SOURCE_IDENTITY_REGISTRY_COLUMN,
)

_KIND_STR_TO_INT = {
    "other": 0,
    "preamble": 1,
    "func_signature": 2,
    "func_body": 3,
    "class_decl": 4,
    "class_member": 5,
    "comment": 6,
    "typedef": 7,
    "namespace": 8,
}


def _kind_to_int(kind: Any) -> int:
    if isinstance(kind, int):
        return kind
    text = str(kind)
    if text.isdigit():
        return int(text)
    return _KIND_STR_TO_INT.get(text.lower(), 0)


def _normalize_graph_edge_pairs(raw_edges: Any) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for edge in raw_edges or []:
        if isinstance(edge, dict) and "from" in edge and "to" in edge:
            pairs.append((int(edge["from"]), int(edge["to"])))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            pairs.append((int(edge[0]), int(edge[1])))
    return pairs


def _normalize_graph_edge_triples(
    raw_edges: Any,
    *,
    family: str | None = None,
) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for edge in raw_edges or []:
        edge_family = family
        if family is None:
            if isinstance(edge, dict) and "kind" in edge:
                edge_family = domain_edge_family(int(edge["kind"]))
            elif hasattr(edge, "__len__") and len(edge) == 3:
                edge_family = domain_edge_family(int(edge[2]))
            else:
                raise ValueError("domain edge missing src/dst/kind")
        assert edge_family is not None
        triples.append(normalize_domain_edge_record(edge, family=edge_family))
    return triples


def _encode_batch_with_optional_char_spans(
    tokenizer,
    texts: list[str],
    *,
    prepend=None,
    append=None,
    num_threads: int = 8,
) -> tuple[list[list[int]], list[list[tuple[int, int]]] | None]:
    def _call_tokenizer_encode(payload):
        kwargs = {}
        try:
            params = inspect.signature(tokenizer.encode).parameters
        except (TypeError, ValueError):
            params = {}
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_var_kwargs or "prepend" in params:
            kwargs["prepend"] = prepend
        if accepts_var_kwargs or "append" in params:
            kwargs["append"] = append
        if accepts_var_kwargs or "num_threads" in params:
            kwargs["num_threads"] = num_threads
        return tokenizer.encode(payload, **kwargs)

    hf_tok = getattr(tokenizer, "_tokenizer", None) or getattr(
        tokenizer, "tokenizer", None
    )
    if hf_tok is None or not hasattr(hf_tok, "encode_batch"):
        return _call_tokenizer_encode(texts), None

    def _resolve_special_id(value: str | int | None) -> int | None:
        if value is None or isinstance(value, int):
            return value
        if hasattr(tokenizer, "encode_special"):
            return int(tokenizer.encode_special(value))
        raise ValueError(f"Tokenizer cannot resolve special token {value!r}")

    prepend_id = _resolve_special_id(prepend)
    append_id = _resolve_special_id(append)

    # CRITICAL: apply the SAME whitespace-sentinel normalization as
    # ``CppMegaTokenizer.encode`` so the stored ``input_ids`` contain
    # ``<NL>``(47)/``<SPACE>``(46) and newline/indent structure is preserved.
    # Normalization changes char offsets, so we keep a per-character map from
    # the normalized string back onto the ORIGINAL text and translate every
    # per-token span through it. The doc sidecars (structure_ids/ast_depth/
    # change_mask/...) are char-aligned to the ORIGINAL text, so the returned
    # spans MUST be in original-text coordinates to stay aligned (no off-by-N).
    normalized_texts: list[str] = []
    norm_maps: list[list[tuple[int, int]]] = []
    for text in texts:
        normalized, norm_to_orig = normalize_whitespace_with_offsets(text)
        normalized_texts.append(normalized)
        norm_maps.append(norm_to_orig)

    try:
        encodings = hf_tok.encode_batch(normalized_texts, add_special_tokens=False)
    except TypeError:
        encodings = hf_tok.encode_batch(normalized_texts)

    token_lists: list[list[int]] = []
    token_spans: list[list[tuple[int, int]]] = []
    for text, norm_to_orig, enc in zip(texts, norm_maps, encodings):
        ids = list(enc.ids)
        spans = [
            _normalized_span_to_original_span(norm_to_orig, int(start), int(end))
            for start, end in enc.offsets
        ]
        if prepend is not None:
            assert prepend_id is not None
            ids.insert(0, int(prepend_id))
            spans.insert(0, (0, 0))
        if append is not None:
            assert append_id is not None
            ids.append(int(append_id))
            text_len = len(text)
            spans.append((text_len, text_len))
        token_lists.append(ids)
        token_spans.append(spans)
    return token_lists, token_spans


def _normalized_span_to_original_span(
    normalized_to_original: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int]:
    """Translate a [start, end) span in normalized coords to original coords.

    ``normalized_to_original[k]`` is the original ``(start, end)`` span the k-th
    normalized character came from. A token's normalized span maps to the union
    of the original spans of the normalized characters it covers. Empty/zero-
    width tokens map to ``(0, 0)`` (treated as invalid by the char->token
    projection, matching the prior behavior for special-token positions).
    """
    if not normalized_to_original:
        return (0, 0)
    start = min(max(start, 0), len(normalized_to_original))
    end = min(max(end, start), len(normalized_to_original))
    if end <= start:
        return (0, 0)
    covered = normalized_to_original[start:end]
    return min(item[0] for item in covered), max(item[1] for item in covered)


def _extract_token_char_starts_and_valid(
    token_positions: list[int] | list[tuple[int, int]],
) -> tuple[list[int], list[bool]]:
    if not token_positions:
        return [], []
    first = token_positions[0]
    if isinstance(first, tuple) and len(first) == 2:
        spans = cast(list[tuple[int, int]], token_positions)
        span_starts = [int(start) for start, _ in spans]
        span_valid = [int(end) > int(start) for start, end in spans]
        return span_starts, span_valid

    lengths = cast(list[int], token_positions)
    starts: list[int] = []
    valid: list[bool] = []
    running = 0
    for length in lengths:
        n = max(int(length), 0)
        starts.append(running)
        valid.append(n > 0)
        running += n
    return starts, valid


def _chars_to_tokens_structure_ids(
    char_structure_ids: list[int],
    text: str,
    token_lengths: list[int] | list[tuple[int, int]],
) -> list[int]:
    del text
    if not char_structure_ids:
        return [0] * len(token_lengths)
    starts, valid = _extract_token_char_starts_and_valid(token_lengths)
    n_struct = len(char_structure_ids)
    out: list[int] = []
    for start, is_valid in zip(starts, valid):
        if is_valid and start < n_struct:
            out.append(int(char_structure_ids[start]))
        else:
            out.append(0)
    return out


def _inherit_zero_width_source_tokens(
    values: list[int],
    token_spans: list[tuple[int, int]],
    *,
    field: str,
) -> list[int]:
    """Assign synthetic zero-width tokens to the nearest exact source token.

    BOS/EOS-like tokenizer insertions have no character span. They are part of
    the surrounding document and may inherit its nearest source identity. A
    zero on a token that covers real source characters is data loss and remains
    a hard error.
    """

    if len(values) != len(token_spans):
        raise ValueError(
            f"{field} length {len(values)} != token span count {len(token_spans)}"
        )
    positive_indices = [index for index, value in enumerate(values) if int(value) > 0]
    if not positive_indices:
        return [int(value) for value in values]

    result = [int(value) for value in values]
    for index, value in enumerate(result):
        if value > 0:
            continue
        start, end = (int(part) for part in token_spans[index])
        if end > start:
            raise ValueError(
                f"{field} is missing exact source identity for nonempty token "
                f"span [{start}, {end}) at token {index}"
            )
        nearest = min(
            positive_indices,
            key=lambda candidate: (abs(candidate - index), candidate),
        )
        result[index] = result[nearest]
    return result


def _chunk_boundaries_to_token_offsets(
    chunk_boundaries: list[dict[str, Any]],
    text: str,
    token_lengths: list[int] | list[tuple[int, int]],
) -> list[dict[str, Any]]:
    del text
    if not chunk_boundaries or not token_lengths:
        return []
    if token_lengths and isinstance(token_lengths[0], tuple):
        spans = [
            (int(start), int(end))
            for start, end in cast(list[tuple[int, int]], token_lengths)
        ]
    else:
        starts, valid = _extract_token_char_starts_and_valid(token_lengths)
        lengths = cast(list[int], token_lengths)
        spans = [
            (start, start + max(int(length), 0)) if is_valid else (start, start)
            for start, length, is_valid in zip(starts, lengths, valid, strict=True)
        ]
    if not spans:
        return []

    result: list[dict[str, Any]] = []
    for chunk_index, cb in enumerate(chunk_boundaries):
        if "start" not in cb or "end" not in cb:
            raise ValueError(
                f"chunk_boundaries[{chunk_index}] requires exact clang start/end"
            )
        char_start = int(cb["start"])
        char_end = int(cb["end"])
        if not 0 <= char_start < char_end:
            raise ValueError(
                f"chunk_boundaries[{chunk_index}] invalid half-open clang span "
                f"[{char_start}, {char_end})"
            )
        overlapping = [
            token_index
            for token_index, (token_start, token_end) in enumerate(spans)
            if token_end > token_start
            and token_end > char_start
            and token_start < char_end
        ]
        if not overlapping:
            raise ValueError(
                f"chunk_boundaries[{chunk_index}] clang span "
                f"[{char_start}, {char_end}) overlaps no tokenizer span"
            )
        result.append(
            {
                "token_offset": overlapping[0],
                "token_end": overlapping[-1] + 1,
                "end_char": char_end,
                "kind": cb.get("kind", 0),
                "name": cb.get("name", ""),
                "dep_level": cb.get("dep_level", 0),
            }
        )
    return result


def _compute_token_dep_levels(
    tok_struct_ids: list[int],
    tok_chunks: list[dict[str, Any]],
    num_tokens: int,
) -> list[int]:
    del tok_struct_ids
    if not tok_chunks:
        return [0] * num_tokens

    sorted_chunks = sorted(tok_chunks, key=lambda c: int(c["token_offset"]))
    out = [0] * num_tokens
    chunk_idx = 0
    current_dep = 0
    for token_idx in range(num_tokens):
        while chunk_idx < len(sorted_chunks) and int(sorted_chunks[chunk_idx]["token_offset"]) <= token_idx:
            current_dep = int(sorted_chunks[chunk_idx].get("dep_level", 0))
            chunk_idx += 1
        out[token_idx] = current_dep
    return out


def _remap_token_edges(
    raw_edges: list[tuple[int, int]],
    index_map: dict[int, int],
) -> list[dict[str, int]]:
    remapped = []
    for src, dst in raw_edges:
        if src in index_map and dst in index_map:
            remapped.append({"from": int(index_map[src]), "to": int(index_map[dst])})
    return remapped


def _char_position_to_token_index(
    token_spans: list[tuple[int, int]],
    char_pos: int,
    *,
    source_length: int,
) -> int | None:
    char_pos = int(char_pos)
    if char_pos < 0 or char_pos >= int(source_length):
        raise ValueError(
            f"character endpoint {char_pos} is outside source text bounds "
            f"[0, {source_length})"
        )
    for idx, (start, end) in enumerate(token_spans):
        start_i = int(start)
        end_i = int(end)
        if end_i > start_i and start_i <= char_pos < end_i:
            return idx
    return None


def _char_point_to_token_index(
    token_spans: list[tuple[int, int]],
    char_pos: int,
    *,
    source_length: int,
) -> int | None:
    """Map a source insertion point to a deterministic nonempty token.

    Point anchors use the token on their right. If no right token exists,
    including at EOF, they use the final source token on their left. This
    represents a valid closed-range point without changing its coordinate.
    """

    char_pos = int(char_pos)
    source_length = int(source_length)
    if char_pos < 0 or char_pos > source_length:
        raise ValueError(
            f"character point {char_pos} is outside source point bounds "
            f"[0, {source_length}]"
        )

    nonempty: list[tuple[int, int, int]] = []
    for idx, (start, end) in enumerate(token_spans):
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            continue
        nonempty.append((idx, start_i, end_i))
        if start_i <= char_pos < end_i:
            return idx
    for idx, start_i, _end_i in nonempty:
        if start_i >= char_pos:
            return idx
    if nonempty:
        return nonempty[-1][0]
    return None


def _char_span_to_token_bounds(
    token_spans: list[tuple[int, int]],
    start_char: int,
    end_char: int,
    *,
    source_length: int,
) -> tuple[int, int] | None:
    start_char = int(start_char)
    end_char = int(end_char)
    source_length = int(source_length)
    if start_char < 0 or end_char <= start_char or end_char > source_length:
        raise ValueError(
            f"character span {start_char}:{end_char} is outside source span bounds "
            f"[0, {source_length}]"
        )

    first: int | None = None
    last: int | None = None
    for idx, (start, end) in enumerate(token_spans):
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i or end_i <= start_char or start_i >= end_char:
            continue
        if first is None:
            first = idx
        last = idx
    if first is None or last is None:
        return None
    return first, last + 1


def _remap_char_edge_triples_to_tokens(
    raw_edges: Any,
    token_spans: list[tuple[int, int]],
    *,
    family: str | None = None,
    source_length: int | None = None,
) -> list[dict[str, int]]:
    remapped: list[dict[str, int]] = []
    triples = _normalize_graph_edge_triples(
        raw_edges,
        family=family,
    )
    if triples and not token_spans:
        src_char, dst_char, _kind = triples[0]
        raise ValueError(
            f"domain edge {src_char}->{dst_char} could not be mapped to token spans"
        )
    if source_length is None:
        source_length = max((int(end) for _start, end in token_spans), default=0)
    for src_char, dst_char, kind in triples:
        if src_char == dst_char:
            src = dst = _char_point_to_token_index(
                token_spans,
                src_char,
                source_length=source_length,
            )
        else:
            src = _char_position_to_token_index(
                token_spans,
                src_char,
                source_length=source_length,
            )
            dst = _char_position_to_token_index(
                token_spans,
                dst_char,
                source_length=source_length,
            )
        if src is None or dst is None:
            raise ValueError(
                f"domain edge {src_char}->{dst_char} endpoint is not contained "
                "in a nonempty token span"
            )
        remapped.append({"from": int(src), "to": int(dst), "kind": int(kind)})
    return remapped


def _build_token_chunk_layout(
    doc: dict[str, Any],
    tok_chunks: list[dict[str, Any]],
    token_count: int,
) -> dict[str, list[int] | list[dict[str, int]]]:
    if not tok_chunks:
        return {
            TOKEN_CHUNK_STARTS_COLUMN: [],
            TOKEN_CHUNK_ENDS_COLUMN: [],
            TOKEN_CHUNK_KINDS_COLUMN: [],
            TOKEN_CHUNK_DEP_LEVELS_COLUMN: [],
            TOKEN_CALL_EDGES_COLUMN: [],
            TOKEN_TYPE_EDGES_COLUMN: [],
        }

    chunk_entries = list(enumerate(tok_chunks))
    chunk_entries.sort(key=lambda item: int(item[1].get("token_offset", 0)))
    index_map = {orig_idx: new_idx for new_idx, (orig_idx, _) in enumerate(chunk_entries)}

    starts = [int(chunk.get("token_offset", 0)) for _, chunk in chunk_entries]
    ends = [int(chunk["token_end"]) for _, chunk in chunk_entries]
    for chunk_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if not 0 <= start < end <= token_count:
            raise ValueError(
                f"token chunk {chunk_index} exact span [{start}, {end}) is "
                f"outside 0..{token_count}"
            )
    kinds = [_kind_to_int(chunk.get("kind", 0)) for _, chunk in chunk_entries]
    dep_levels = [int(chunk.get("dep_level", 0)) for _, chunk in chunk_entries]

    call_edges = _remap_token_edges(
        _normalize_graph_edge_pairs(doc.get("call_edges", [])),
        index_map,
    )
    type_edges = _remap_token_edges(
        _normalize_graph_edge_pairs(doc.get("type_edges", [])),
        index_map,
    )

    return {
        TOKEN_CHUNK_STARTS_COLUMN: starts,
        TOKEN_CHUNK_ENDS_COLUMN: ends,
        TOKEN_CHUNK_KINDS_COLUMN: kinds,
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: dep_levels,
        TOKEN_CALL_EDGES_COLUMN: call_edges,
        TOKEN_TYPE_EDGES_COLUMN: type_edges,
    }


def _tokenize_optional_char_field(
    doc: dict[str, Any],
    keys: tuple[str, ...],
    token_spans: list[tuple[int, int]],
) -> list[int]:
    for key in keys:
        values = doc.get(key, [])
        if values:
            return _chars_to_tokens_structure_ids(values, "", token_spans)
    return []


def _domain_kind_from_doc(doc: dict[str, Any]) -> DomainKind | None:
    raw = doc.get("domain_kind")
    if raw in (None, ""):
        return None
    try:
        if isinstance(raw, str) and not raw.isdigit():
            domain = DomainKind[raw.upper()]
        else:
            domain = DomainKind(int(raw))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unknown domain_kind {raw!r}") from exc
    if domain == DomainKind.UNKNOWN:
        raise ValueError("unknown domain_kind 0 disables delimiter insertion")
    return domain


def materialized_marker_token_overhead(doc: dict[str, Any]) -> int:
    """Return the exact BOS/domain-marker tokens added during materialization."""

    embedded_spans = normalize_embedded_domain_spans(
        doc.get("embedded_domain_spans", []),
        source_length=len(doc.get("text", "")),
    )
    top_level_pair = 2 if _domain_kind_from_doc(doc) is not None else 0
    return 1 + 2 * len(embedded_spans) + top_level_pair


def _shift_token_index_for_pair(value: int, *, start_at: int, end_at: int) -> int:
    value_i = int(value)
    return value_i + int(value_i >= start_at) + int(value_i >= end_at)


def _shift_exclusive_end_for_pair(
    value: int,
    *,
    start_at: int,
    end_at: int,
) -> int:
    value_i = int(value)
    return value_i + int(value_i > start_at) + int(value_i > end_at)


def _insert_domain_delimiter_pair(
    row: dict[str, Any],
    *,
    domain: DomainKind,
    start_at: int,
    end_at: int,
) -> None:
    domain = DomainKind(domain)
    if domain == DomainKind.UNKNOWN:
        raise ValueError("UNKNOWN domain cannot be inserted without delimiters")
    try:
        start_id, end_id = delimiter_token_ids(domain)
    except KeyError as exc:
        raise ValueError(f"missing delimiter token contract for domain {domain!r}") from exc

    token_ids = list(row[TOKEN_IDS_COLUMN])
    token_count_before = len(token_ids)
    start_at = max(0, min(int(start_at), token_count_before))
    end_at = max(start_at, min(int(end_at), token_count_before))
    row[TOKEN_IDS_COLUMN] = (
        token_ids[:start_at]
        + [int(start_id)]
        + token_ids[start_at:end_at]
        + [int(end_id)]
        + token_ids[end_at:]
    )

    domain_value = int(domain)
    delimiter_role = int(DomainRoleKind.DELIMITER)
    exact_confidence = int(ParseConfidence.EXACT)
    dense_defaults = {
        TOKEN_DOMAIN_IDS_COLUMN: (domain_value, domain_value),
        TOKEN_ROLE_IDS_COLUMN: (delimiter_role, delimiter_role),
        TOKEN_ENTITY_IDS_COLUMN: (0, 0),
        TOKEN_SCOPE_IDS_COLUMN: (0, 0),
        TOKEN_SOURCE_DOC_IDS_COLUMN: (0, 0),
        TOKEN_SOURCE_IDENTITY_IDS_COLUMN: (0, 0),
        TOKEN_CONFIDENCE_IDS_COLUMN: (exact_confidence, exact_confidence),
    }
    for column, marker_values in dense_defaults.items():
        values = list(row.get(column, []))
        if not values:
            values = [0] * token_count_before
        if len(values) != token_count_before:
            raise ValueError(
                f"{column} length {len(values)} does not match token count {token_count_before}"
            )
        start_marker, end_marker = marker_values
        interior = values[start_at:end_at]
        if column == TOKEN_DOMAIN_IDS_COLUMN:
            interior = [
                domain_value if int(value) == int(DomainKind.UNKNOWN) else int(value)
                for value in interior
            ]
        if column in {
            TOKEN_SOURCE_DOC_IDS_COLUMN,
            TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
        } and values:
            start_source_idx = min(start_at, token_count_before - 1)
            end_source_idx = max(0, min(end_at - 1, token_count_before - 1))
            start_marker = int(values[start_source_idx])
            end_marker = int(values[end_source_idx])
        row[column] = (
            values[:start_at]
            + [int(start_marker)]
            + interior
            + [int(end_marker)]
            + values[end_at:]
        )

    for column in (
        TOKEN_STRUCTURE_IDS_COLUMN,
        TOKEN_DEP_LEVELS_COLUMN,
        TOKEN_AST_DEPTH_COLUMN,
        TOKEN_SIBLING_INDEX_COLUMN,
        TOKEN_AST_NODE_TYPE_COLUMN,
        TOKEN_SYMBOL_IDS_COLUMN,
        TOKEN_CALL_TARGETS_COLUMN,
        TOKEN_TYPE_REFS_COLUMN,
        TOKEN_DEF_USE_COLUMN,
        TOKEN_CHANGE_MASK_PRE_COLUMN,
        TOKEN_CHANGE_MASK_POST_COLUMN,
        HUNK_ID_PER_TOKEN_COLUMN,
        EDIT_OP_PER_TOKEN_COLUMN,
    ):
        values = list(row.get(column, []))
        if not values:
            continue
        if len(values) != token_count_before:
            raise ValueError(
                f"{column} length {len(values)} does not match token count {token_count_before}"
            )
        pad_value = -1 if column == HUNK_ID_PER_TOKEN_COLUMN else 0
        row[column] = (
            values[:start_at]
            + [pad_value]
            + values[start_at:end_at]
            + [pad_value]
            + values[end_at:]
        )

    row[TOKEN_CHUNK_STARTS_COLUMN] = [
        _shift_token_index_for_pair(value, start_at=start_at, end_at=end_at)
        for value in row.get(TOKEN_CHUNK_STARTS_COLUMN, [])
    ]
    row[TOKEN_CHUNK_ENDS_COLUMN] = [
        _shift_exclusive_end_for_pair(value, start_at=start_at, end_at=end_at)
        for value in row.get(TOKEN_CHUNK_ENDS_COLUMN, [])
    ]

    for column in (
        TOKEN_DOMAIN_EDGES_COLUMN,
        TOKEN_BUILD_EDGES_COLUMN,
        TOKEN_SHELL_EDGES_COLUMN,
        TOKEN_DIAGNOSTIC_EDGES_COLUMN,
        TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
    ):
        shifted: list[dict[str, int]] = []
        for edge in row.get(column, []):
            shifted.append(
                {
                    "from": _shift_token_index_for_pair(
                        int(edge["from"]), start_at=start_at, end_at=end_at
                    ),
                    "to": _shift_token_index_for_pair(
                        int(edge["to"]), start_at=start_at, end_at=end_at
                    ),
                    "kind": int(edge["kind"]),
                }
            )
        row[column] = shifted

    if len(row[TOKEN_IDS_COLUMN]) != token_count_before + 2:
        raise AssertionError("domain delimiter insertion corrupted token length")


def _insert_domain_delimiters(
    row: dict[str, Any],
    *,
    domain: DomainKind,
    insert_at: int = 1,
) -> None:
    _insert_domain_delimiter_pair(
        row,
        domain=domain,
        start_at=insert_at,
        end_at=len(row[TOKEN_IDS_COLUMN]),
    )


def _insert_embedded_domain_delimiters(
    row: dict[str, Any],
    *,
    doc: dict[str, Any],
    token_spans: list[tuple[int, int]],
) -> None:
    resolved: list[tuple[int, int, DomainKind]] = []
    text_length = len(doc.get("text", ""))
    spans = normalize_embedded_domain_spans(
        doc.get("embedded_domain_spans", []),
        source_length=text_length,
    )
    for span in reversed(spans):
        start_char = int(span["start"])
        end_char = int(span["end"])
        domain = DomainKind(int(span["domain_kind"]))
        token_bounds = _char_span_to_token_bounds(
            token_spans,
            start_char,
            end_char,
            source_length=text_length,
        )
        if token_bounds is None:
            raise ValueError(
                f"embedded domain span {start_char}:{end_char} could not be mapped to token spans"
            )
        start_token, end_token = token_bounds
        resolved.append((start_token, end_token, domain))

    for start_token, end_token, domain in resolved:
        _insert_domain_delimiter_pair(
            row,
            domain=domain,
            start_at=start_token,
            end_at=end_token,
        )


def _changed_chunk_metadata(
    token_chunk_starts: list[int],
    token_chunk_ends: list[int],
    change_masks: tuple[list[int], ...],
) -> tuple[list[int], list[dict[str, int]]]:
    if not token_chunk_starts or not token_chunk_ends:
        return [], []
    changed_ids: list[int] = []
    changed_spans: list[dict[str, int]] = []
    for chunk_idx, (start, end) in enumerate(zip(token_chunk_starts, token_chunk_ends)):
        start_i = max(int(start), 0)
        end_i = max(int(end), start_i)
        changed = False
        for mask in change_masks:
            if not mask:
                continue
            bounded_end = min(end_i, len(mask))
            if start_i < bounded_end and any(int(value) != 0 for value in mask[start_i:bounded_end]):
                changed = True
                break
        if changed:
            changed_ids.append(chunk_idx)
            changed_spans.append({"start": start_i, "end": end_i})
    return changed_ids, changed_spans


def _platform_ids_from_doc(doc: dict[str, Any]) -> list[int]:
    platform_info = doc.get("platform_info")
    if not platform_info:
        return []
    if isinstance(platform_info, str):
        try:
            platform_info = json.loads(platform_info)
        except json.JSONDecodeError:
            return []
    if not isinstance(platform_info, dict):
        return []

    from cppmega.data.nanochat_pipeline.platform_vocab import platform_info_to_ids

    return platform_info_to_ids(platform_info)


def materialize_tokenized_enriched_batch(
    docs: list[dict[str, Any]],
    tokenizer,
    *,
    num_threads: int = 8,
) -> list[dict[str, Any]]:
    """Convert enriched char-level docs into token-level parquet-ready metadata.

    The returned token arrays align to `token_ids`, which already include the BOS
    token at position 0.
    """
    if not docs:
        return []

    texts = [str(doc.get("text", "")) for doc in docs]
    if hasattr(tokenizer, "get_bos_token_id"):
        bos_token = tokenizer.get_bos_token_id()
    else:
        bos_token = tokenizer.bos_token_id
    token_lists, token_spans_batch = _encode_batch_with_optional_char_spans(
        tokenizer,
        texts,
        prepend=bos_token,
        num_threads=num_threads,
    )
    if token_spans_batch is None:
        raise ValueError(
            "Tokenizer did not expose per-token character spans; "
            "cannot materialize token-level enriched metadata offline."
        )

    objective_token_batches: dict[str, list[list[int]]] = {}
    for text_column, token_column in (
        (IFIM_INSTRUCTION_TEXT_COLUMN, IFIM_INSTRUCTION_TOKEN_IDS_COLUMN),
        (COMMIT_MSG_TEXT_COLUMN, COMMIT_MSG_TOKEN_IDS_COLUMN),
        (PRE_TEXT_COLUMN, PRE_TOKEN_IDS_COLUMN),
        (POST_TEXT_COLUMN, POST_TOKEN_IDS_COLUMN),
        (DIFF_TEXT_COLUMN, DIFF_TOKEN_IDS_COLUMN),
    ):
        section_texts = [
            value if isinstance(value, str) and value.strip() else ""
            for doc in docs
            for value in (doc.get(text_column),)
        ]
        encoded, _ = _encode_batch_with_optional_char_spans(
            tokenizer,
            section_texts,
            num_threads=num_threads,
        )
        objective_token_batches[token_column] = [
            [int(token_id) for token_id in token_ids] if text else []
            for text, token_ids in zip(section_texts, encoded)
        ]

    out = []
    for doc_index, (doc, token_ids, token_spans) in enumerate(
        zip(docs, token_lists, token_spans_batch)
    ):
        token_ids = [int(tok) for tok in token_ids]
        canonical_edge_fields = canonicalize_domain_edge_fields(
            doc,
            source_length=len(doc.get("text", "")),
        )
        token_structure_ids = _chars_to_tokens_structure_ids(
            doc.get("structure_ids", []),
            "",
            token_spans,
        )
        tok_chunks = _chunk_boundaries_to_token_offsets(
            doc.get("chunk_boundaries", []),
            "",
            token_spans,
        )
        token_dep_levels = _compute_token_dep_levels(
            token_structure_ids,
            tok_chunks,
            len(token_ids),
        )

        token_change_mask_pre = _tokenize_optional_char_field(
            doc,
            ("change_mask_pre", TOKEN_CHANGE_MASK_PRE_COLUMN),
            token_spans,
        )
        token_change_mask_post = _tokenize_optional_char_field(
            doc,
            ("change_mask_post", "token_change_mask", TOKEN_CHANGE_MASK_POST_COLUMN),
            token_spans,
        )
        hunk_id_per_token = _tokenize_optional_char_field(
            doc,
            ("hunk_id_per_char", HUNK_ID_PER_TOKEN_COLUMN),
            token_spans,
        )
        edit_op_per_token = _tokenize_optional_char_field(
            doc,
            ("edit_op_per_char", "token_edit_op", EDIT_OP_PER_TOKEN_COLUMN),
            token_spans,
        )
        registry_entries = [
            dict(entry) for entry in doc.get(SOURCE_IDENTITY_REGISTRY_COLUMN, [])
        ]
        if not registry_entries:
            resolved_identity = source_identity(doc)
            registry_entries = [resolved_identity.as_dict()]
        fallback_identity_id = int(registry_entries[0]["source_identity_id"])
        raw_token_source_doc_ids = _chars_to_tokens_structure_ids(
            doc.get("domain_source_doc_ids", doc.get("source_doc_ids", [])),
            "",
            token_spans,
        )
        raw_token_source_doc_ids = _inherit_zero_width_source_tokens(
            raw_token_source_doc_ids,
            token_spans,
            field=TOKEN_SOURCE_DOC_IDS_COLUMN,
        )
        token_source_doc_ids = normalize_row_local_doc_ids(
            raw_token_source_doc_ids,
            length=len(token_ids),
            fallback_doc_id=1,
        )
        raw_token_source_identity_ids = _chars_to_tokens_structure_ids(
            doc.get("domain_source_identity_ids", []),
            "",
            token_spans,
        )
        raw_token_source_identity_ids = _inherit_zero_width_source_tokens(
            raw_token_source_identity_ids,
            token_spans,
            field=TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
        )
        if len(registry_entries) > 1 and (
            not raw_token_source_identity_ids
            or any(int(value) == 0 for value in raw_token_source_identity_ids)
        ):
            raise ValueError(
                "multi-source document is missing exact token source identities"
            )
        token_source_identity_ids = normalize_positive_source_ids(
            raw_token_source_identity_ids,
            length=len(token_ids),
            fallback_source_id=fallback_identity_id,
        )
        validate_source_identity_registry(
            registry_entries,
            referenced_ids=token_source_identity_ids,
        )

        row: dict[str, Any] = {
            TOKEN_IDS_COLUMN: token_ids,
            PLATFORM_IDS_COLUMN: _platform_ids_from_doc(doc),
            TOKEN_STRUCTURE_IDS_COLUMN: token_structure_ids,
            TOKEN_DEP_LEVELS_COLUMN: token_dep_levels,
            TOKEN_AST_DEPTH_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("ast_depth", []),
                "",
                token_spans,
            ),
            TOKEN_SIBLING_INDEX_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("sibling_index", []),
                "",
                token_spans,
            ),
            TOKEN_AST_NODE_TYPE_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("ast_node_type", []),
                "",
                token_spans,
            ),
            TOKEN_SYMBOL_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("symbol_ids", []),
                "",
                token_spans,
            ),
            TOKEN_CALL_TARGETS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("call_targets", []),
                "",
                token_spans,
            ),
            TOKEN_TYPE_REFS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("type_refs", []),
                "",
                token_spans,
            ),
            TOKEN_DEF_USE_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("def_use", []),
                "",
                token_spans,
            ),
            TOKEN_DOMAIN_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_ids", []),
                "",
                token_spans,
            ),
            TOKEN_ROLE_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_role_ids", doc.get("role_ids", [])),
                "",
                token_spans,
            ),
            TOKEN_ENTITY_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_entity_ids", doc.get("entity_ids", [])),
                "",
                token_spans,
            ),
            TOKEN_SCOPE_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_scope_ids", doc.get("scope_ids", [])),
                "",
                token_spans,
            ),
            TOKEN_SOURCE_DOC_IDS_COLUMN: token_source_doc_ids,
            TOKEN_SOURCE_IDENTITY_IDS_COLUMN: token_source_identity_ids,
            TOKEN_CONFIDENCE_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_confidence_ids", doc.get("confidence_ids", [])),
                "",
                token_spans,
            ),
            TOKEN_DOMAIN_EDGES_COLUMN: _remap_char_edge_triples_to_tokens(
                canonical_edge_fields["domain_edges"],
                token_spans,
                family="domain",
                source_length=len(doc.get("text", "")),
            ),
            TOKEN_BUILD_EDGES_COLUMN: _remap_char_edge_triples_to_tokens(
                canonical_edge_fields["build_edges"],
                token_spans,
                family="build",
                source_length=len(doc.get("text", "")),
            ),
            TOKEN_SHELL_EDGES_COLUMN: _remap_char_edge_triples_to_tokens(
                canonical_edge_fields["shell_edges"],
                token_spans,
                family="shell",
                source_length=len(doc.get("text", "")),
            ),
            TOKEN_DIAGNOSTIC_EDGES_COLUMN: _remap_char_edge_triples_to_tokens(
                canonical_edge_fields["diagnostic_edges"],
                token_spans,
                family="diagnostic",
                source_length=len(doc.get("text", "")),
            ),
            TOKEN_CROSS_DOMAIN_EDGES_COLUMN: _remap_char_edge_triples_to_tokens(
                canonical_edge_fields["cross_domain_edges"],
                token_spans,
                family="cross_domain",
                source_length=len(doc.get("text", "")),
            ),
            TOKEN_CHANGE_MASK_PRE_COLUMN: token_change_mask_pre,
            TOKEN_CHANGE_MASK_POST_COLUMN: token_change_mask_post,
            HUNK_ID_PER_TOKEN_COLUMN: hunk_id_per_token,
            EDIT_OP_PER_TOKEN_COLUMN: edit_op_per_token,
            **{
                column: token_batches[doc_index]
                for column, token_batches in objective_token_batches.items()
            },
            SOURCE_IDENTITY_REGISTRY_COLUMN: registry_entries,
        }
        row.update(cast(dict[str, list[int]], _build_token_chunk_layout(doc, tok_chunks, len(token_ids))))
        _insert_embedded_domain_delimiters(row, doc=doc, token_spans=token_spans)
        domain = _domain_kind_from_doc(doc)
        if domain is not None:
            _insert_domain_delimiters(row, domain=domain)
        raw_token_count = len(token_ids) - 1
        actual_overhead = len(row[TOKEN_IDS_COLUMN]) - raw_token_count
        expected_overhead = materialized_marker_token_overhead(doc)
        if actual_overhead != expected_overhead:
            raise AssertionError(
                "materialized marker overhead does not match the delimiter contract: "
                f"{actual_overhead} != {expected_overhead}"
            )
        changed_chunk_ids, changed_chunk_spans = _changed_chunk_metadata(
            cast(list[int], row[TOKEN_CHUNK_STARTS_COLUMN]),
            cast(list[int], row[TOKEN_CHUNK_ENDS_COLUMN]),
            (
                cast(list[int], row[TOKEN_CHANGE_MASK_PRE_COLUMN]),
                cast(list[int], row[TOKEN_CHANGE_MASK_POST_COLUMN]),
            ),
        )
        row[CHANGED_CHUNK_IDS_COLUMN] = changed_chunk_ids
        row[CHANGED_CHUNK_SPANS_COLUMN] = changed_chunk_spans
        out.append(row)

    return out
