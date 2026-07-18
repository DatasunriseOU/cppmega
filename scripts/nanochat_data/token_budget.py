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
        return 1

    window_start = max(1, best - search_window)
    preferred_breaks = ("\n\n", "\n", " ", "\t")
    for marker in preferred_breaks:
        idx = text.rfind(marker, window_start, best + 1)
        if idx > 0:
            return idx + len(marker)
    return best


_CHAR_LEVEL_METADATA_FIELDS = (
    "ast_depth",
    "sibling_index",
    "ast_node_type",
    "symbol_ids",
    "call_targets",
    "type_refs",
    "def_use",
    "domain_ids",
    "domain_role_ids",
    "domain_entity_ids",
    "domain_scope_ids",
    "domain_source_doc_ids",
    "domain_confidence_ids",
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
    sliced = {
        k: v
        for k, v in doc.items()
        if k
        not in {
            "text",
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
            piece = _slice_doc_char_range(doc, cursor, len(text), selected_boundaries=[])
            piece["actual_token_count"] = exact
            pieces.append(piece)
            break

        cut = _best_split_char_index(remaining, max_tokens, tokenizer)
        abs_cut = cursor + cut
        piece = _slice_doc_char_range(doc, cursor, abs_cut, selected_boundaries=[])
        piece["actual_token_count"] = count_tokens(piece["text"], tokenizer)
        pieces.append(piece)
        cursor = abs_cut
        remaining = text[cursor:]
    return pieces


def chunk_enriched_document(
    doc: dict[str, Any],
    max_tokens: int,
    tokenizer: TokenCounter,
    *,
    boundary_sort_key=None,
) -> list[dict[str, Any]]:
    """Split an enriched document so every emitted piece fits `max_tokens` exactly."""
    text = doc.get("text", "")
    exact_total = count_tokens(text, tokenizer)
    if exact_total <= max_tokens:
        out = dict(doc)
        out["actual_token_count"] = exact_total
        return [out]

    raw_boundaries = list(enumerate(doc.get("chunk_boundaries", []) or []))
    if not raw_boundaries:
        return _split_doc_without_boundaries(doc, max_tokens, tokenizer)

    key_fn = boundary_sort_key or (lambda item: int(item[1].get("start", 0)))
    ordered = sorted(raw_boundaries, key=key_fn)
    source_ordered = sorted(raw_boundaries, key=lambda item: int(item[1].get("start", 0)))
    source_rank = {orig_idx: rank for rank, (orig_idx, _) in enumerate(source_ordered)}

    chunks: list[dict[str, Any]] = []
    current: list[tuple[int, dict[str, Any]]] = []
    consumed_orig: set[int] = set()
    consumed_ranks: set[int] = set()

    def _expanded_source_span(
        entries: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]] | None:
        ranks = [
            source_rank[orig_idx]
            for orig_idx, _ in entries
            if orig_idx not in consumed_orig
        ]
        if not ranks:
            return []
        lo = min(ranks)
        hi = max(ranks)
        if any(rank in consumed_ranks for rank in range(lo, hi + 1)):
            return None
        return source_ordered[lo : hi + 1]

    def _entries_text(
        entries: list[tuple[int, dict[str, Any]]],
    ) -> tuple[int, int, str, list[tuple[int, dict[str, Any]]]] | None:
        by_source = _expanded_source_span(entries)
        if by_source is None:
            return None
        if not by_source:
            return (0, 0, "", [])
        start = int(by_source[0][1].get("start", 0))
        end = int(by_source[-1][1].get("end", start))
        return start, end, text[start:end], by_source

    def _flush(entries: list[tuple[int, dict[str, Any]]]) -> None:
        if not entries:
            return
        payload = _entries_text(entries)
        if payload is None:
            return
        start, end, chunk_text, by_source = payload
        if not by_source:
            return
        exact = count_tokens(chunk_text, tokenizer)
        if exact <= max_tokens:
            piece = _slice_doc_char_range(doc, start, end, selected_boundaries=by_source)
            piece["actual_token_count"] = exact
            chunks.append(piece)
        else:
            # Single oversized boundary/span: fall back to exact prefix splitting.
            temp_piece = _slice_doc_char_range(doc, start, end, selected_boundaries=by_source)
            chunks.extend(_split_doc_without_boundaries(temp_piece, max_tokens, tokenizer))

        for orig_idx, _ in by_source:
            consumed_orig.add(orig_idx)
            consumed_ranks.add(source_rank[orig_idx])

    for entry in ordered:
        if entry[0] in consumed_orig:
            continue
        candidate = current + [entry]
        payload = _entries_text(candidate)
        if payload is not None:
            _, _, candidate_text, _ = payload
        else:
            candidate_text = None
        if candidate_text is not None and count_tokens(candidate_text, tokenizer) <= max_tokens:
            current = candidate
            continue
        if current:
            _flush(current)
            if entry[0] not in consumed_orig:
                current = [entry]
            else:
                current = []
        else:
            _flush([entry])
            current = []

    if current:
        _flush(current)
    return chunks
