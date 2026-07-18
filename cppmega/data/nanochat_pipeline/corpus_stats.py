"""Portable corpus statistics emitted next to materialized shards."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def compute_corpus_stats(
    token_id_lists: Iterable[list[int]],
    *,
    vocab_size: int,
    topk: int = 50,
    hist_bin: int = 64,
) -> dict:
    """Aggregate token coverage, document lengths, and vocabulary usage."""

    seen: set[int] = set()
    doc_lengths: list[int] = []
    token_counts: Counter[int] = Counter()
    for ids in token_id_lists:
        if not ids:
            continue
        doc_lengths.append(len(ids))
        for token_id in ids:
            seen.add(token_id)
            token_counts[token_id] += 1

    def percentile(values: list[int], percent: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(round(percent / 100 * len(ordered)))))
        return int(ordered[index])

    if vocab_size <= 0:
        coverage_pct = 0.0
    else:
        coverage_pct = round(len(seen) / vocab_size * 100.0, 4)

    if doc_lengths:
        max_length = max(doc_lengths)
        bin_count = max(1, (max_length + hist_bin) // hist_bin)
        histogram = [0] * bin_count
        for length in doc_lengths:
            histogram[min(bin_count - 1, length // hist_bin)] += 1
        histogram_edges = [index * hist_bin for index in range(bin_count + 1)]
    else:
        histogram = []
        histogram_edges = []

    return {
        "n_docs": len(doc_lengths),
        "vocab_size": int(vocab_size),
        "token_coverage_pct": coverage_pct,
        "unique_tokens_seen": len(seen),
        "doc_length_p50": percentile(doc_lengths, 50),
        "doc_length_p90": percentile(doc_lengths, 90),
        "doc_length_p99": percentile(doc_lengths, 99),
        "doc_length_hist_edges": histogram_edges,
        "doc_length_hist_counts": histogram,
        "vocab_topk": [
            {"token_id": int(token_id), "count": int(count)}
            for token_id, count in token_counts.most_common(topk)
        ],
        "vocab_long_tail_count": sum(
            1 for count in token_counts.values() if count <= 1
        ),
    }


__all__ = ["compute_corpus_stats"]
