#!/usr/bin/env python3
"""Phase-0 audit of a megacpp tokenized parquet dataset.

Scans every shard of a dataset directory and emits a markdown + json report
covering:

  * rows, real tokens (sum of ``actual_token_count``), padded@seq tokens
  * token-length distribution (p50 / p95 / min / max / mean)
  * repo distribution and filepath/extension distribution
  * per-column population (POPULATED / SPARSE / EMPTY / ABSENT) for the key
    graph / semantic / provenance columns of the modern v12 schema
  * parse / compile-confidence distribution if such a column is present
  * a cheap near-duplicate rate (hash of the leading token-ID prefix)

The script is fail-closed (RULE #1, fail loud). It exits non-zero on:

  * empty ``repo`` for a static-code dataset kind
  * any token_id >= vocab_size
  * unexpectedly-empty ``type_edges`` (and ``token_type_edges``) for a graph
    dataset kind

Every raised error names WHERE (shard + row group + row) and WHAT failed.

Usage:
    .venv/bin/python audit_megacpp_4k.py \
        --dataset-dir /Users/dave/sources/parquet/clang_semantic_4k_v10 \
        --kind static_code \
        --vocab-size 65536 \
        --seq-len 4096 \
        --out-dir /tmp/audit_clang_semantic_4k_v10

    .venv/bin/python audit_megacpp_4k.py \
        --dataset-dir /Users/dave/sources/parquet/clang_commits_4k_v1 \
        --kind commits --graph
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq


# --- modern v12 schema column groups (read-only reference, mirrored here) -----
# Token-aligned side channels carried per row.
TOKEN_SIDE_CHANNEL_COLUMNS = (
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
    "token_def_use",
)
# Semantic / graph columns from clang_enriched_to_parquet v12 _SCHEMA.
SEMANTIC_COLUMNS = (
    "structure_ids",
    "symbol_ids",
    "call_targets",
    "type_refs",
    "def_use",
)
GRAPH_COLUMNS = (
    "call_edges",
    "type_edges",
    "token_call_edges",
    "token_type_edges",
    "chunk_boundaries",
)
PROVENANCE_COLUMNS = (
    "repo",
    "commit",
    "commit_hash",
    "filepath",
    "timestamp",
    "constituent_provenance",
    "provenance",
)
# Columns that may carry a per-row parse/compile confidence signal.
CONFIDENCE_COLUMNS = (
    "parse_confidence",
    "compile_confidence",
    "parse_status",
    "compile_status",
)

TOKEN_COLUMN = "token_ids"
ACTUAL_TOKEN_COUNT_COLUMN = "actual_token_count"

# Populated thresholds: fraction of non-empty rows.
SPARSE_THRESHOLD = 0.5

STATIC_CODE_KINDS = ("static_code",)
COMMIT_KINDS = ("commits",)


class AuditError(RuntimeError):
    """Raised when a fail-closed audit invariant is violated."""


def _fail(where: str, what: str) -> None:
    raise AuditError(f"WHERE={where} WHAT={what}")


def _is_empty_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def _list_len(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _ext_of(filepath: Any) -> str:
    if not isinstance(filepath, str) or not filepath:
        return "<none>"
    name = filepath.rsplit("/", 1)[-1]
    if "." not in name:
        return "<none>"
    return "." + name.rsplit(".", 1)[-1]


def find_shards(dataset_dir: Path) -> list[Path]:
    shards = sorted(dataset_dir.glob("*.parquet"))
    if not shards:
        _fail(str(dataset_dir), "no *.parquet shards found")
    return shards


def _present_columns(shards: list[Path]) -> set[str]:
    names: set[str] = set()
    for shard in shards:
        names.update(pq.ParquetFile(shard).schema_arrow.names)
    return names


def audit(
    *,
    dataset_dir: Path,
    kind: str,
    vocab_size: int,
    seq_len: int,
    graph: bool,
    dedup_prefix: int,
    batch_size: int,
) -> dict[str, Any]:
    shards = find_shards(dataset_dir)
    present = _present_columns(shards)
    if TOKEN_COLUMN not in present:
        _fail(str(dataset_dir), f"required token column {TOKEN_COLUMN!r} absent")

    all_tracked = (
        TOKEN_SIDE_CHANNEL_COLUMNS
        + SEMANTIC_COLUMNS
        + GRAPH_COLUMNS
        + PROVENANCE_COLUMNS
        + CONFIDENCE_COLUMNS
    )
    # Per-column non-empty / total counts (only for columns actually present).
    col_nonempty: dict[str, int] = {c: 0 for c in all_tracked if c in present}
    col_total = 0

    total_rows = 0
    real_tokens = 0
    token_lengths: list[int] = []
    repo_counter: Counter[str] = Counter()
    ext_counter: Counter[str] = Counter()
    confidence_counter: Counter[str] = Counter()
    dup_hashes: Counter[str] = Counter()

    has_repo = "repo" in present
    has_filepath = "filepath" in present
    confidence_present = [c for c in CONFIDENCE_COLUMNS if c in present]
    type_edge_present = [c for c in ("type_edges", "token_type_edges") if c in present]

    read_cols = sorted(
        {TOKEN_COLUMN}
        | ({ACTUAL_TOKEN_COUNT_COLUMN} if ACTUAL_TOKEN_COUNT_COLUMN in present else set())
        | {c for c in col_nonempty}
    )

    graph_type_edge_seen = False

    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
            cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
            n = batch.num_rows
            token_lists = cols[TOKEN_COLUMN]
            actual = cols.get(ACTUAL_TOKEN_COUNT_COLUMN)
            repos = cols.get("repo")
            filepaths = cols.get("filepath")

            for row in range(n):
                row_where = f"{shard.name}#row{total_rows + row}"
                tokens = token_lists[row]
                if tokens is None:
                    _fail(row_where, f"{TOKEN_COLUMN} is null")
                tlen = len(tokens)
                token_lengths.append(tlen)
                if tokens:
                    mx = max(tokens)
                    if mx >= vocab_size:
                        _fail(
                            row_where,
                            f"token_id {mx} >= vocab_size {vocab_size}",
                        )
                if actual is not None and actual[row] is not None:
                    real_tokens += int(actual[row])
                else:
                    real_tokens += tlen

                if has_repo:
                    rv = repos[row] if repos is not None else None
                    repo_counter[rv if isinstance(rv, str) and rv else "<empty>"] += 1
                if has_filepath:
                    fp = filepaths[row] if filepaths is not None else None
                    ext_counter[_ext_of(fp)] += 1

                for c in confidence_present:
                    cv = cols[c][row]
                    confidence_counter[str(cv)] += 1

                for c in col_nonempty:
                    if not _is_empty_scalar(cols[c][row]):
                        col_nonempty[c] += 1
                        if c in ("type_edges", "token_type_edges"):
                            graph_type_edge_seen = True

                if dedup_prefix > 0 and tokens:
                    prefix = tokens[:dedup_prefix]
                    h = hashlib.blake2b(
                        b",".join(str(int(t)).encode() for t in prefix),
                        digest_size=16,
                    ).hexdigest()
                    dup_hashes[h] += 1

            col_total += n
            total_rows += n

    if not token_lengths:
        _fail(str(dataset_dir), "dataset has zero rows")

    sorted_lengths = sorted(token_lengths)

    def _pct(p: float) -> int:
        idx = min(len(sorted_lengths) - 1, int(p * len(sorted_lengths)))
        return sorted_lengths[idx]

    padded_at_seq = total_rows * seq_len

    # Column population classification.
    column_population: dict[str, dict[str, Any]] = {}
    for c in all_tracked:
        if c not in present:
            column_population[c] = {"status": "ABSENT", "nonempty": 0, "total": col_total}
            continue
        nonempty = col_nonempty[c]
        frac = nonempty / col_total if col_total else 0.0
        if nonempty == 0:
            status = "EMPTY"
        elif frac < SPARSE_THRESHOLD:
            status = "SPARSE"
        else:
            status = "POPULATED"
        column_population[c] = {
            "status": status,
            "nonempty": nonempty,
            "total": col_total,
            "fraction": round(frac, 4),
        }

    near_dups = sum(count - 1 for count in dup_hashes.values() if count > 1)
    near_dup_rate = near_dups / total_rows if total_rows else 0.0

    report: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "kind": kind,
        "graph": graph,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "shards": len(shards),
        "rows": total_rows,
        "real_tokens": real_tokens,
        "padded_tokens_at_seq": padded_at_seq,
        "token_length": {
            "min": sorted_lengths[0],
            "p50": _pct(0.50),
            "p95": _pct(0.95),
            "max": sorted_lengths[-1],
            "mean": round(mean(sorted_lengths), 2),
        },
        "repo_distribution": dict(repo_counter.most_common(50)),
        "distinct_repos": len(repo_counter),
        "filepath_extension_distribution": dict(ext_counter.most_common(50)),
        "confidence_distribution": dict(confidence_counter.most_common(50))
        if confidence_present
        else None,
        "confidence_columns": confidence_present,
        "column_population": column_population,
        "near_dup_rate": round(near_dup_rate, 6),
        "near_dup_groups": sum(1 for c in dup_hashes.values() if c > 1),
        "dedup_prefix": dedup_prefix,
    }

    # --- fail-closed invariants -------------------------------------------
    if kind in STATIC_CODE_KINDS:
        if not has_repo:
            _fail(str(dataset_dir), "static_code dataset has no 'repo' column")
        nonempty_repos = total_rows - repo_counter.get("<empty>", 0)
        if nonempty_repos == 0:
            _fail(
                str(dataset_dir),
                "static_code dataset has EMPTY repo for ALL rows "
                f"({total_rows} rows, 0 with a non-empty repo)",
            )

    if graph:
        if not type_edge_present:
            _fail(
                str(dataset_dir),
                "graph dataset declared but no type_edges/token_type_edges column present",
            )
        if not graph_type_edge_seen:
            _fail(
                str(dataset_dir),
                "graph dataset has UNEXPECTEDLY-EMPTY type_edges for ALL rows "
                f"(columns checked: {', '.join(type_edge_present)})",
            )

    return report


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# megacpp 4k audit — {Path(report['dataset_dir']).name}")
    lines.append("")
    lines.append(f"- dataset_dir: `{report['dataset_dir']}`")
    lines.append(f"- kind: `{report['kind']}`  graph: `{report['graph']}`")
    lines.append(f"- vocab_size: {report['vocab_size']}  seq_len: {report['seq_len']}")
    lines.append(f"- shards: {report['shards']}  rows: {report['rows']:,}")
    lines.append(f"- real_tokens (sum actual_token_count): {report['real_tokens']:,}")
    lines.append(f"- padded_tokens@seq: {report['padded_tokens_at_seq']:,}")
    tl = report["token_length"]
    lines.append(
        f"- token_length: min={tl['min']} p50={tl['p50']} p95={tl['p95']} "
        f"max={tl['max']} mean={tl['mean']}"
    )
    lines.append(f"- distinct_repos: {report['distinct_repos']}")
    lines.append(f"- near_dup_rate: {report['near_dup_rate']} "
                 f"(prefix={report['dedup_prefix']}, groups={report['near_dup_groups']})")
    lines.append("")
    lines.append("## repo distribution (top 50)")
    for k, v in report["repo_distribution"].items():
        lines.append(f"- `{k}`: {v:,}")
    lines.append("")
    lines.append("## filepath extension distribution (top 50)")
    for k, v in report["filepath_extension_distribution"].items():
        lines.append(f"- `{k}`: {v:,}")
    lines.append("")
    if report["confidence_distribution"] is not None:
        lines.append("## parse/compile confidence distribution")
        lines.append(f"- columns: {', '.join(report['confidence_columns'])}")
        for k, v in report["confidence_distribution"].items():
            lines.append(f"- `{k}`: {v:,}")
        lines.append("")
    lines.append("## column population")
    lines.append("| column | status | nonempty | total | fraction |")
    lines.append("| --- | --- | --- | --- | --- |")
    for c, info in report["column_population"].items():
        lines.append(
            f"| `{c}` | {info['status']} | {info['nonempty']:,} | "
            f"{info['total']:,} | {info.get('fraction', 0.0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--kind",
        required=True,
        choices=["static_code", "commits", "other"],
        help="Dataset kind. 'static_code' requires non-empty repo for >=1 row.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=65536,
        help="Tokenizer vocab; any token_id >= this fails (default canonical 65536).",
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Treat as graph dataset; type_edges must be non-empty for >=1 row.",
    )
    parser.add_argument(
        "--dedup-prefix",
        type=int,
        default=64,
        help="Token prefix length for cheap near-dup hashing (0 disables).",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for audit_<name>.md / .json (default: print json to stdout).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 2

    try:
        report = audit(
            dataset_dir=dataset_dir,
            kind=args.kind,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            graph=args.graph,
            dedup_prefix=args.dedup_prefix,
            batch_size=args.batch_size,
        )
    except AuditError as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 1

    markdown = _to_markdown(report)
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        name = dataset_dir.name
        (out_dir / f"audit_{name}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (out_dir / f"audit_{name}.md").write_text(markdown, encoding="utf-8")
        print(f"wrote {out_dir / f'audit_{name}.json'}")
        print(f"wrote {out_dir / f'audit_{name}.md'}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
