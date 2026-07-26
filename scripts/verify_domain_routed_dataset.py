#!/usr/bin/env python3
"""Verify domain-routed parquet slices before training/upload."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cppmega.data.domain_schema import DomainKind, ParseConfidence  # noqa: E402
from cppmega.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS  # noqa: E402


TOKEN_SIDECARS = (
    "token_domain_ids",
    "token_role_ids",
    "token_entity_ids",
    "token_scope_ids",
    "token_source_doc_ids",
    "token_confidence_ids",
)
EDGE_COLUMNS = (
    "token_domain_edges",
    "token_build_edges",
    "token_shell_edges",
    "token_diagnostic_edges",
    "token_cross_domain_edges",
    "token_call_edges",
    "token_type_edges",
)
DIAGNOSTIC_DOMAINS = {
    int(DomainKind.COMPILER_DIAGNOSTIC),
    int(DomainKind.BUILD_DIAGNOSTIC),
    int(DomainKind.COMPILER_ERROR),
    int(DomainKind.BUILD_ERROR),
    int(DomainKind.LINKER_ERROR),
    int(DomainKind.TEST_OUTPUT),
    int(DomainKind.TOOL_OUTPUT),
}
BUILD_DOMAINS = {
    int(DomainKind.CMAKE),
    int(DomainKind.MAKE),
    int(DomainKind.NINJA),
    int(DomainKind.BAZEL),
    int(DomainKind.AUTOCONF),
    int(DomainKind.AUTOMAKE),
    int(DomainKind.MESON),
    int(DomainKind.GN),
    int(DomainKind.SCONS),
    int(DomainKind.XMAKE),
}


def _list_lengths(column: Any) -> np.ndarray:
    return np.asarray(
        pc.fill_null(pc.list_value_length(column), 0)
        .combine_chunks()
        .to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )


def _flat(column: Any) -> np.ndarray:
    arr = pc.list_flatten(column).combine_chunks()
    if len(arr) == 0:
        return np.asarray([], dtype=np.int64)
    return np.asarray(arr.to_numpy(zero_copy_only=False))


def _discover(root: Path, kind: str, buckets: set[str] | None) -> list[tuple[str, Path, str]]:
    out: list[tuple[str, Path, str]] = []
    if not root.exists():
        return out
    for path in sorted(root.glob("*/*.parquet")):
        bucket = path.parent.name
        if buckets is None or bucket in buckets:
            out.append((kind, path, bucket))
    return out


def verify_file(kind: str, path: Path, bucket: str) -> dict[str, Any]:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    columns = ["input_ids"]
    columns += [name for name in TOKEN_SIDECARS if name in names]
    columns += [name for name in EDGE_COLUMNS if name in names]
    columns += [
        name
        for name in (
            "valid_token_count",
            "doc_ids",
            "build_kind",
            "source_build_kinds",
            "doc_type",
            "text",
        )
        if name in names
    ]
    table = pq.read_table(path, columns=columns)
    rows = table.num_rows
    input_lengths = _list_lengths(table.column("input_ids"))
    valid = (
        np.asarray(table.column("valid_token_count").combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
        if "valid_token_count" in names
        else input_lengths
    )
    result: dict[str, Any] = {
        "kind": kind,
        "bucket": bucket,
        "path": str(path),
        "rows": rows,
        "valid_tokens": int(valid.sum()),
        "sidecar_rows_present": {},
        "sidecar_slots_nonzero": {},
        "edge_count": {},
        "domain_token_counts": Counter(),
        "confidence_token_counts": Counter(),
        "build_kind_counts": Counter(),
        "errors": [],
    }

    flat_input = _flat(table.column("input_ids"))
    delimiter_ids = set(DOMAIN_DELIMITER_TOKEN_IDS.values())
    delimiter_rows = np.zeros(rows, dtype=bool)
    if len(flat_input):
        delimiter_rows = _rows_from_mask(input_lengths, np.isin(flat_input, list(delimiter_ids)))

    if np.any(delimiter_rows):
        for required in ("token_domain_ids", "token_role_ids", "token_confidence_ids"):
            if required not in names:
                result["errors"].append(f"{int(np.count_nonzero(delimiter_rows))} rows contain domain delimiters but missing {required}")

    domain_rows = np.zeros(rows, dtype=bool)
    diagnostic_rows = np.zeros(rows, dtype=bool)
    build_rows = np.zeros(rows, dtype=bool)
    if "token_domain_ids" in names:
        domain_flat = _flat(table.column("token_domain_ids")).astype(np.int64)
        result["domain_token_counts"].update(Counter(int(x) for x in domain_flat if int(x) != 0))
        domain_rows = _rows_from_mask(input_lengths, domain_flat != 0)
        diagnostic_rows = _rows_from_mask(input_lengths, np.isin(domain_flat, list(DIAGNOSTIC_DOMAINS)))
        build_rows = _rows_from_mask(input_lengths, np.isin(domain_flat, list(BUILD_DOMAINS)))
    if "token_confidence_ids" in names:
        conf_flat = _flat(table.column("token_confidence_ids")).astype(np.int64)
        result["confidence_token_counts"].update(Counter(int(x) for x in conf_flat if int(x) != 0))
        raw_rows = _rows_from_mask(input_lengths, conf_flat == int(ParseConfidence.RAW))
    else:
        raw_rows = np.zeros(rows, dtype=bool)

    for col in TOKEN_SIDECARS:
        if col not in names:
            continue
        lengths = _list_lengths(table.column(col))
        flat = _flat(table.column(col))
        result["sidecar_rows_present"][col] = int(np.count_nonzero(lengths))
        result["sidecar_slots_nonzero"][col] = int(np.count_nonzero(flat))

    diag_edge_rows = np.zeros(rows, dtype=bool)
    for col in EDGE_COLUMNS:
        if col not in names:
            result["edge_count"][col] = 0
            continue
        lengths = _list_lengths(table.column(col))
        result["edge_count"][col] = int(lengths.sum())
        if col == "token_diagnostic_edges":
            diag_edge_rows = lengths > 0

    bad_diag = diagnostic_rows & ~diag_edge_rows & ~raw_rows
    if np.any(bad_diag):
        result["errors"].append(
            f"{int(np.count_nonzero(bad_diag))} diagnostic/error rows have no token_diagnostic_edges and no RAW confidence"
        )

    if "source_build_kinds" in names:
        source_build_kinds = table.column("source_build_kinds").to_pylist()
        result["build_kind_counts"].update(
            Counter(
                str(kind)
                for row_kinds in source_build_kinds
                for kind in (row_kinds or [])
                if kind
            )
        )
        if "doc_ids" not in names:
            result["errors"].append(
                "source_build_kinds is present but doc_ids is missing"
            )
        elif "token_domain_ids" in names:
            flat_doc_ids = _flat(table.column("doc_ids")).astype(np.int64)
            domain_flat = _flat(table.column("token_domain_ids")).astype(np.int64)
            if len(flat_doc_ids) != len(domain_flat):
                result["errors"].append(
                    "doc_ids and token_domain_ids have different flattened lengths"
                )
            else:
                missing_build_kinds = 0
                offset = 0
                for length, row_kinds in zip(
                    input_lengths,
                    source_build_kinds,
                    strict=True,
                ):
                    stop = offset + int(length)
                    row_domain_ids = domain_flat[offset:stop]
                    row_doc_ids = flat_doc_ids[offset:stop]
                    build_doc_ids = set(
                        int(doc_id)
                        for doc_id in row_doc_ids[
                            np.isin(row_domain_ids, list(BUILD_DOMAINS))
                        ]
                        if int(doc_id) > 0
                    )
                    kinds = list(row_kinds or [])
                    missing_build_kinds += sum(
                        doc_id > len(kinds) or not kinds[doc_id - 1]
                        for doc_id in build_doc_ids
                    )
                    offset = stop
                if missing_build_kinds:
                    result["errors"].append(
                        f"{missing_build_kinds} packed build document(s) have "
                        "no aligned source_build_kinds value"
                    )
    elif "build_kind" in names:
        result["build_kind_counts"].update(
            Counter(
                str(value)
                for value in table.column("build_kind").to_pylist()
                if value
            )
        )
    elif np.any(build_rows):
        result["errors"].append(
            "build domain rows present but build_kind/source_build_kinds is missing"
        )

    if "text" in names and "doc_type" in names:
        for text, doc_type in zip(table.column("text").to_pylist(), table.column("doc_type").to_pylist(), strict=False):
            if not isinstance(text, str):
                continue
            if "<COMPILER_ERROR_START>" in text and "/**" in text and doc_type != "commit_discussion":
                result["errors"].append("compiler error marker appears inside C++ comment/docstring outside commit_discussion")
                break

    result["domain_rows"] = int(np.count_nonzero(domain_rows))
    result["diagnostic_rows"] = int(np.count_nonzero(diagnostic_rows))
    result["build_rows"] = int(np.count_nonzero(build_rows))
    result["domain_token_counts"] = dict(result["domain_token_counts"])
    result["confidence_token_counts"] = dict(result["confidence_token_counts"])
    result["build_kind_counts"] = dict(result["build_kind_counts"])
    return result


def _rows_from_mask(lengths: np.ndarray, flat_mask: np.ndarray) -> np.ndarray:
    rows = np.zeros(len(lengths), dtype=bool)
    if len(lengths) == 0 or len(flat_mask) == 0:
        return rows
    starts = np.zeros(len(lengths), dtype=np.int64)
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], out=starts[1:])
    nonempty = lengths > 0
    if np.any(nonempty):
        rows[nonempty] = np.add.reduceat(flat_mask.astype(np.int8), starts[nonempty]) > 0
    return rows


def rollup(file_results: list[dict[str, Any]], *, min_cpp_graph_coverage: float) -> dict[str, Any]:
    total_tokens = 0
    rows = 0
    edge_count: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    build_kind_counts: Counter[str] = Counter()
    errors: list[str] = []
    cpp_rows = 0
    cpp_graph_rows = 0
    by_kind_bucket: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "valid_tokens": 0})
    for item in file_results:
        total_tokens += int(item["valid_tokens"])
        rows += int(item["rows"])
        edge_count.update(item["edge_count"])
        domain_counts.update({str(k): int(v) for k, v in item["domain_token_counts"].items()})
        confidence_counts.update({str(k): int(v) for k, v in item["confidence_token_counts"].items()})
        build_kind_counts.update(item["build_kind_counts"])
        errors.extend(f"{item['path']}: {err}" for err in item["errors"])
        key = f"{item['kind']}/{item['bucket']}"
        by_kind_bucket[key]["rows"] += int(item["rows"])
        by_kind_bucket[key]["valid_tokens"] += int(item["valid_tokens"])
        cpp_token_count = int(item["domain_token_counts"].get(int(DomainKind.CPP), 0))
        if cpp_token_count:
            cpp_rows += int(item["domain_rows"])
            if int(item["edge_count"].get("token_call_edges", 0)) or int(item["edge_count"].get("token_type_edges", 0)):
                cpp_graph_rows += int(item["domain_rows"])

    cpp_graph_coverage = (cpp_graph_rows / cpp_rows) if cpp_rows else 1.0
    if cpp_graph_coverage < min_cpp_graph_coverage:
        errors.append(
            f"C++ graph-route coverage {cpp_graph_coverage:.4f} below threshold {min_cpp_graph_coverage:.4f}"
        )

    return {
        "files": len(file_results),
        "rows": rows,
        "valid_tokens": total_tokens,
        "by_kind_bucket": dict(sorted(by_kind_bucket.items())),
        "domain_token_counts": dict(sorted(domain_counts.items())),
        "confidence_token_counts": dict(sorted(confidence_counts.items())),
        "edge_count": dict(sorted(edge_count.items())),
        "build_kind_counts": dict(sorted(build_kind_counts.items())),
        "cpp_graph_coverage": cpp_graph_coverage,
        "errors": errors[:2000],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--commit-root", type=Path)
    parser.add_argument("--pr-root", type=Path)
    parser.add_argument("--buckets", nargs="*")
    parser.add_argument("--out", type=Path, default=Path("outputs/reports/domain_routed_dataset_verification.json"))
    parser.add_argument("--min-cpp-graph-coverage", type=float, default=0.0)
    args = parser.parse_args()

    buckets = set(args.buckets) if args.buckets else None
    work: list[tuple[str, Path, str]] = []
    if args.code_root:
        work.extend(_discover(args.code_root, "code", buckets))
    if args.commit_root:
        work.extend(_discover(args.commit_root, "commit", buckets))
    if args.pr_root:
        work.extend(_discover(args.pr_root, "pr", buckets))
    if not work:
        raise SystemExit("no parquet shards discovered")

    file_results = [verify_file(kind, path, bucket) for kind, path, bucket in work]
    report = rollup(file_results, min_cpp_graph_coverage=args.min_cpp_graph_coverage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("files", "rows", "valid_tokens", "cpp_graph_coverage")}, sort_keys=True))
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
