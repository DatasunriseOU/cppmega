"""Tokenizer-aware token budget helpers for offline data conversion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from cppmega.data.domain_schema import (
    DOMAIN_EDGE_FIELD_FAMILIES,
    normalize_domain_edge_record,
    slice_embedded_domain_spans,
)
from cppmega.tokenizer.cpp_tokenizer import CppMegaTokenizer, load_cppmega_tokenizer
from cppmega.tokenizer.fingerprint import (
    tokenizer_fingerprint as _tokenizer_fingerprint,
)


class TokenCounter(Protocol):
    def encode(self, text: str) -> list[int]: ...


def resolve_tokenizer_path(tokenizer_path: str | None = None) -> str:
    """Resolve a usable tokenizer.json path for offline data scripts."""
    candidates: list[str] = []
    if tokenizer_path:
        candidates.append(tokenizer_path)
    env_path = os.environ.get("NANOCHAT_TOKENIZER_PATH")
    if env_path:
        candidates.append(env_path)

    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            str(repo_root / "cppmega" / "tokenizer" / "tokenizer.json"),
            str(repo_root / "tokenizer.json"),
            str(repo_root / "tokenizer" / "tokenizer.json"),
        ]
    )

    requested_base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if requested_base_dir:
        candidates.append(str(Path(requested_base_dir) / "tokenizer" / "tokenizer.json"))

    candidates.append(str(Path.cwd() / "tokenizer.json"))
    candidates.append(str(Path.cwd() / "tokenizer" / "tokenizer.json"))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find tokenizer.json. Set NANOCHAT_TOKENIZER_PATH or place "
        "tokenizer.json in the repo root / tokenizer/ directory."
    )


def load_tokenizer(tokenizer_path: str | None = None) -> CppMegaTokenizer:
    return load_cppmega_tokenizer(resolve_tokenizer_path(tokenizer_path))


def tokenizer_fingerprint(tokenizer_or_path: Any | None = None) -> str:
    if tokenizer_or_path is None:
        tokenizer_or_path = resolve_tokenizer_path(None)
    return _tokenizer_fingerprint(tokenizer_or_path)


def count_tokens(text: str, tokenizer: TokenCounter) -> int:
    return len(tokenizer.encode(text))


def size_label_to_tokens(size_label: str) -> int:
    """Parse labels like 4k, 8k, 16k into integer token budgets."""
    s = size_label.strip().lower()
    if s.endswith("k"):
        return int(s[:-1]) * 1024
    return int(s)


def _best_split_char_index(
    text: str,
    max_tokens: int,
    tokenizer: TokenCounter,
    *,
    search_window: int = 256,
) -> int:
    """Return the largest prefix char index whose exact token count fits."""
    if not text:
        return 0

    lo = 1
    hi = len(text)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if count_tokens(candidate, tokenizer) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best <= 0:
        raise ValueError(
            "cannot split document losslessly: no non-empty character prefix "
            f"fits the exact {max_tokens}-token budget"
        )

    window_start = max(1, best - search_window)
    preferred_breaks = ("\n\n", "\n", " ", "\t")
    for marker in preferred_breaks:
        idx = text.rfind(marker, window_start, best + 1)
        if idx > 0:
            preferred = idx + len(marker)
            # BPE token counts are not monotonic over character prefixes: a
            # longer prefix can merge tokens that remain separate in a shorter
            # one. ``best`` was measured exactly above, but this prettier,
            # shorter boundary was not necessarily one of the binary-search
            # probes. Never return it without an exact budget check.
            if count_tokens(text[:preferred], tokenizer) <= max_tokens:
                return preferred
    return best


_CHAR_LEVEL_METADATA_FIELDS = (
    "ast_depth",
    "sibling_index",
    "ast_node_type",
    "symbol_ids",
    "call_targets",
    "type_refs",
    "def_use",
    "change_mask_pre",
    "change_mask_post",
    "token_change_mask",
    "hunk_id_per_char",
    "edit_op_per_char",
    "token_edit_op",
    "domain_ids",
    "domain_role_ids",
    "role_ids",
    "domain_entity_ids",
    "entity_ids",
    "domain_scope_ids",
    "scope_ids",
    "domain_source_doc_ids",
    "source_doc_ids",
    "domain_source_identity_ids",
    "domain_confidence_ids",
    "confidence_ids",
)
_SPLIT_AUDIT_FIELD = "_lossless_split_audit"


def _validate_aligned_character_metadata(
    doc: dict[str, Any],
    *,
    text_length: int,
) -> None:
    """Reject present character sidecars that cannot be sliced exactly."""

    structure_ids = doc.get("structure_ids", [])
    if structure_ids and len(structure_ids) != text_length:
        raise ValueError(
            "structure_ids length "
            f"{len(structure_ids)} does not match text length {text_length}"
        )
    for field in _CHAR_LEVEL_METADATA_FIELDS:
        values = doc.get(field, [])
        if values and len(values) != text_length:
            raise ValueError(
                f"{field} sidecar length {len(values)} does not match "
                f"text length {text_length}"
            )

    source_text = doc.get("source_text")
    if source_text is not None:
        if not isinstance(source_text, str):
            raise ValueError("source_text must be a string when present")
        if len(source_text) != text_length:
            raise ValueError(
                "source_text length "
                f"{len(source_text)} does not match text length {text_length}"
            )


def _slice_doc_char_range(
    doc: dict[str, Any],
    start_char: int,
    end_char: int,
    *,
    selected_boundaries: list[tuple[int, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Slice an enriched document by char range and preserve compatible metadata."""
    text = doc.get("text", "")
    _validate_aligned_character_metadata(doc, text_length=len(text))
    sliced = {
        k: v
        for k, v in doc.items()
        if k
        not in {
            "text",
            "source_text",
            "structure_ids",
            "chunk_boundaries",
            "call_edges",
            "type_edges",
            "actual_token_count",
            "embedded_domain_spans",
            *_CHAR_LEVEL_METADATA_FIELDS,
        }
    }
    sliced["text"] = text[start_char:end_char]
    if "source_text" in doc:
        source_text = doc.get("source_text")
        sliced["source_text"] = (
            None if source_text is None else source_text[start_char:end_char]
        )
    sliced["embedded_domain_spans"] = slice_embedded_domain_spans(
        doc.get("embedded_domain_spans", []),
        source_length=len(text),
        start=start_char,
        end=end_char,
    )

    structure_ids = doc.get("structure_ids", [])
    if structure_ids:
        sliced["structure_ids"] = list(structure_ids[start_char:end_char])
    else:
        sliced["structure_ids"] = []

    for key in _CHAR_LEVEL_METADATA_FIELDS:
        values = doc.get(key, [])
        if values:
            sliced[key] = list(values[start_char:end_char])
        else:
            sliced[key] = []

    if selected_boundaries is None:
        raw_boundaries = list(enumerate(doc.get("chunk_boundaries", [])))
        selected_boundaries = [
            (orig_idx, cb)
            for orig_idx, cb in raw_boundaries
            if int(cb.get("start", 0)) < end_char and int(cb.get("end", 0)) > start_char
        ]

    selected_sorted = sorted(
        selected_boundaries,
        key=lambda item: int(item[1].get("start", 0)),
    )
    remap: dict[int, int] = {}
    adjusted_boundaries: list[dict[str, Any]] = []
    for orig_idx, cb in selected_sorted:
        cb_start = max(int(cb.get("start", 0)), start_char)
        cb_end = min(int(cb.get("end", cb_start)), end_char)
        if cb_end <= cb_start:
            continue
        remap[orig_idx] = len(adjusted_boundaries)
        adjusted_boundaries.append(
            {
                "start": cb_start - start_char,
                "end": cb_end - start_char,
                "kind": cb.get("kind", 0),
                "dep_level": cb.get("dep_level", 0),
                "name": cb.get("name", ""),
                "symbol_id": cb.get("symbol_id"),
            }
        )
    sliced["chunk_boundaries"] = adjusted_boundaries

    def _normalize_edges(raw_edges: Any) -> list[tuple[int, int]]:
        normalized: list[tuple[int, int]] = []
        for edge in raw_edges or []:
            if isinstance(edge, dict) and "from" in edge and "to" in edge:
                normalized.append((int(edge["from"]), int(edge["to"])))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                normalized.append((int(edge[0]), int(edge[1])))
        return normalized

    def _remap_edges(raw_edges: Any) -> list[dict[str, int]]:
        remapped: list[dict[str, int]] = []
        for src_idx, dst_idx in _normalize_edges(raw_edges):
            if src_idx in remap and dst_idx in remap:
                remapped.append({"from": remap[src_idx], "to": remap[dst_idx]})
        return remapped

    sliced["call_edges"] = _remap_edges(doc.get("call_edges", []))
    sliced["type_edges"] = _remap_edges(doc.get("type_edges", []))

    def _remap_char_edge_triples(
        raw_edges: Any,
        *,
        family: str,
    ) -> list[dict[str, int]]:
        remapped: list[dict[str, int]] = []
        for src, dst, kind in (
            normalize_domain_edge_record(edge, family=family)
            for edge in raw_edges or []
        ):
            if start_char <= src < end_char and start_char <= dst < end_char:
                remapped.append(
                    {
                        "from_char": src - start_char,
                        "to_char": dst - start_char,
                        "kind": kind,
                    }
                )
        return remapped

    for edge_field, family in DOMAIN_EDGE_FIELD_FAMILIES.items():
        sliced[edge_field] = _remap_char_edge_triples(
            doc.get(edge_field, []),
            family=family,
        )
    return sliced


def _split_doc_without_boundaries(
    doc: dict[str, Any],
    max_tokens: int,
    tokenizer: TokenCounter,
) -> list[dict[str, Any]]:
    text = doc.get("text", "")
    if not text:
        return [{**doc, "actual_token_count": 0}]

    remaining = text
    cursor = 0
    pieces: list[dict[str, Any]] = []
    while remaining:
        exact = count_tokens(remaining, tokenizer)
        if exact <= max_tokens:
            piece = _slice_doc_char_range(doc, cursor, len(text))
            piece["actual_token_count"] = exact
            pieces.append(piece)
            break

        cut = _best_split_char_index(remaining, max_tokens, tokenizer)
        abs_cut = cursor + cut
        piece = _slice_doc_char_range(doc, cursor, abs_cut)
        piece["actual_token_count"] = count_tokens(piece["text"], tokenizer)
        if piece["actual_token_count"] > max_tokens:
            raise ValueError(
                "lossless split produced an over-budget piece: "
                f"{piece['actual_token_count']} > {max_tokens}"
            )
        pieces.append(piece)
        cursor = abs_cut
        remaining = text[cursor:]
    return pieces


def _annotate_cross_piece_edge_audit(
    doc: dict[str, Any],
    pieces: list[dict[str, Any]],
) -> None:
    """Record graph edges that cannot exist inside any one split sequence."""

    if len(pieces) <= 1:
        return
    text_length = len(doc.get("text", ""))
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        end = cursor + len(piece.get("text", ""))
        ranges.append((cursor, end))
        cursor = end
    if cursor != text_length:
        raise ValueError(
            f"split pieces cover {cursor} chars but source has {text_length}"
        )

    def point_piece(offset: int) -> int:
        if offset == text_length and ranges:
            return len(ranges) - 1
        if offset < 0 or offset >= text_length:
            raise ValueError(
                f"edge character offset {offset} is outside [0, {text_length}]"
            )
        for piece_index, (start, end) in enumerate(ranges):
            if start <= offset < end:
                return piece_index
        raise AssertionError(f"no split piece owns character offset {offset}")

    cross_piece: dict[str, int] = {}
    for edge_field, family in DOMAIN_EDGE_FIELD_FAMILIES.items():
        count = 0
        for src, dst, _kind in (
            normalize_domain_edge_record(edge, family=family)
            for edge in doc.get(edge_field, []) or []
        ):
            if point_piece(src) != point_piece(dst):
                count += 1
        cross_piece[edge_field] = count

    boundaries = doc.get("chunk_boundaries", []) or []

    def boundary_pieces(boundary_index: int) -> set[int]:
        if boundary_index < 0 or boundary_index >= len(boundaries):
            raise ValueError(
                f"chunk edge references boundary {boundary_index}, "
                f"but only {len(boundaries)} boundaries exist"
            )
        boundary = boundaries[boundary_index]
        start = int(boundary.get("start", 0))
        end = int(boundary.get("end", start))
        if start < 0 or end < start or end > text_length:
            raise ValueError(
                f"invalid chunk boundary [{start}, {end}) for text length {text_length}"
            )
        if start == end:
            return {point_piece(start)}
        return {
            piece_index
            for piece_index, (piece_start, piece_end) in enumerate(ranges)
            if start < piece_end and end > piece_start
        }

    def chunk_cross_count(raw_edges: Any) -> int:
        count = 0
        for edge in raw_edges or []:
            if isinstance(edge, dict) and "from" in edge and "to" in edge:
                source_index = int(edge["from"])
                target_index = int(edge["to"])
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                source_index = int(edge[0])
                target_index = int(edge[1])
            else:
                raise ValueError(f"invalid chunk edge record: {edge!r}")
            if not (
                boundary_pieces(source_index) & boundary_pieces(target_index)
            ):
                count += 1
        return count

    cross_piece["call_edges"] = chunk_cross_count(doc.get("call_edges", []))
    cross_piece["type_edges"] = chunk_cross_count(doc.get("type_edges", []))
    pieces[0][_SPLIT_AUDIT_FIELD] = {
        "cross_piece_edges": cross_piece,
    }


def chunk_enriched_document(
    doc: dict[str, Any],
    max_tokens: int,
    tokenizer: TokenCounter,
    *,
    boundary_sort_key=None,
) -> list[dict[str, Any]]:
    """Split a document into contiguous, sidecar-aligned, exact-budget pieces.

    Output pieces cover ``text`` exactly once and remain in source order. Chunk
    boundaries are clipped/remapped into each piece; aligned character sidecars
    are sliced over the identical character ranges. This is intentionally a
    lossless text path: semantic boundaries may influence metadata, but they may
    never cause uncovered prefix/suffix characters or reordered source spans.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    text = doc.get("text", "")
    _validate_aligned_character_metadata(doc, text_length=len(text))
    exact_total = count_tokens(text, tokenizer)
    if exact_total <= max_tokens:
        out = dict(doc)
        out["actual_token_count"] = exact_total
        return [out]
    # `boundary_sort_key` remains in the public signature for compatibility.
    # Output order is deliberately source order; callers may not use dependency
    # sorting to omit or reorder characters.
    del boundary_sort_key
    pieces = _split_doc_without_boundaries(doc, max_tokens, tokenizer)
    _annotate_cross_piece_edge_audit(doc, pieces)
    return pieces
