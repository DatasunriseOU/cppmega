#!/usr/bin/env python3
"""Audit and repair packed parquet rows with collapsed generated section markers.

Older tokenizer whitespace normalization could treat apostrophes inside PR
docstrings (for example ``can't``) as C++ char literals.  Once that happened,
newlines later in the document were not emitted as ``<NL>`` sentinel tokens, so
packed commit rows could contain sequences such as::

    */// === PRE-COMMIT ===bool ...

instead of::

    */
    // === PRE-COMMIT ===
    bool ...

This fixer is intentionally narrow.  It only inserts ``<NL>`` around generated
cppmega commit section markers (PRE-COMMIT, POST-COMMIT, CONTEXT, DIFF), and
only when the row has enough padding slack.  Token-aligned sidecar columns are
shifted with the inserted token; chunk/span token offsets are adjusted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cppmega.tokenizer.cpp_tokenizer import (  # noqa: E402
    CppMegaTokenizer,
    load_cppmega_tokenizer,
)

TOKENIZER_DIR = REPO_ROOT / "cppmega" / "tokenizer"


@lru_cache(maxsize=1)
def get_tokenizer() -> CppMegaTokenizer:
    return load_cppmega_tokenizer(TOKENIZER_DIR)

# Reuse the producer's EXACT inter-document loss-mask rule so the boundary fixer
# can never drift from pack_enriched_rows: loss_mask[pos] == 1 iff
# doc_ids[pos] == doc_ids[pos + 1] within the valid prefix (0 at every
# inter-document boundary, at the final valid token, and across padding).
from scripts.nanochat_data.pack_enriched_rows import (  # noqa: E402
    _loss_mask_for_packed_docs,
)


PAD_ID = 0
BOS_ID = 2
NL_ID = 47
EOS_ID = 3

TOKEN_ALIGNED_COLUMNS = (
    "doc_ids",
    "platform_ids",
    "token_platform_ids",
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
    "token_symbol_ids",
    "token_call_targets",
    "token_type_refs",
    "token_def_use",
    "token_change_mask_pre",
    "token_change_mask_post",
    "hunk_id_per_token",
    "edit_op_per_token",
)

# token_chunk_starts (inclusive) and token_chunk_ends (exclusive) are shifted
# explicitly with their respective semantics in repair_row; see
# _shift_offset / _shift_offset_exclusive.

TOKEN_SPAN_COLUMNS = (
    "changed_chunk_spans",
)

# Anchor to the EXACT generated marker forms emitted by the producer
# (tools/clang_indexer/process_commits.py:1260/1283/1361/1382): PRE-COMMIT,
# CONTEXT and DIFF are fixed strings; POST-COMMIT carries a variable commit
# subject after the colon (``// === POST-COMMIT: {subject} ===``).  We compare
# whitespace-normalized text so that ordinary source comments that merely
# CONTAIN one of these words (for example ``// === DIFF ALGORITHM ===`` or
# ``// === CONTEXT SWITCH ===``) are NOT misclassified as generated markers and
# never have <NL> tokens injected into real training text.
_EXACT_MARKER_TEXTS = (
    "// === PRE-COMMIT ===",
    "// === CONTEXT ===",
    "// === DIFF ===",
)
_POST_COMMIT_MARKER_PREFIX = "// === POST-COMMIT:"


class TokenPieces:
    def __init__(self, tokenizer: Any) -> None:
        self._cache: dict[int, str] = {
            token_id: tokenizer.token_for_id(token_id) or ""
            for token_id in range(int(tokenizer.vocab_size))
        }

    def piece(self, token_id: int) -> str:
        token_id = int(token_id)
        return self._cache.get(token_id, "")

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.piece(int(token_id)) for token_id in ids)
        return text.replace("<SPACE>", " ").replace("<NL>", "\n")


def _is_space(piece: str) -> bool:
    return piece in {" ", "<SPACE>"}


def _skip_spaces(pieces: TokenPieces, ids: list[int], idx: int, end: int) -> int:
    while idx < end and _is_space(pieces.piece(ids[idx])):
        idx += 1
    return idx


def _generated_marker_end(
    pieces: TokenPieces,
    ids: list[int],
    start: int,
    valid: int,
) -> int | None:
    """Return inclusive end token index for a generated ``// === ... ===`` marker."""
    if start >= valid or pieces.piece(ids[start]) != "//":
        return None
    idx = _skip_spaces(pieces, ids, start + 1, valid)
    if idx + 1 >= valid or pieces.piece(ids[idx]) != "==" or pieces.piece(ids[idx + 1]) != "=":
        return None
    idx = _skip_spaces(pieces, ids, idx + 2, valid)

    scan_limit = min(valid - 1, idx + 160)
    close: int | None = None
    while idx <= scan_limit:
        maybe = _skip_spaces(pieces, ids, idx, valid)
        if maybe + 1 < valid and pieces.piece(ids[maybe]) == "==" and pieces.piece(ids[maybe + 1]) == "=":
            close = maybe + 1
            break
        idx += 1
    if close is None:
        return None

    # Require an EXACT generated marker form (whitespace-normalized).  Any other
    # ``// === ... ===`` comment is ordinary source text and is left untouched
    # (return None): we do NOT inject newlines into rows we cannot positively
    # identify as collapsed cppmega commit markers.
    marker_text = " ".join(pieces.decode(ids[start : close + 1]).split())
    if marker_text in _EXACT_MARKER_TEXTS or marker_text.startswith(_POST_COMMIT_MARKER_PREFIX):
        return close
    return None


def _needed_insert_positions(row: dict[str, Any], pieces: TokenPieces) -> list[int]:
    ids = [int(x) for x in row["input_ids"]]
    valid = int(row.get("valid_token_count") or _infer_valid_count(ids))
    positions: set[int] = set()
    idx = 0
    while idx < valid:
        end = _generated_marker_end(pieces, ids, idx, valid)
        if end is None:
            idx += 1
            continue

        if idx > 0 and ids[idx - 1] != NL_ID:
            positions.add(idx)
        after = end + 1
        if after < valid and ids[after] not in (PAD_ID, BOS_ID, EOS_ID, NL_ID):
            positions.add(after)
        idx = max(end + 1, idx + 1)
    return sorted(positions)


def _infer_valid_count(ids: list[int]) -> int:
    for idx, token_id in enumerate(ids):
        if int(token_id) == PAD_ID:
            return idx
    return len(ids)


def _insert_with_source(values: list[Any], positions: list[int], capacity: int) -> list[Any]:
    if not positions:
        return values
    position_set = set(positions)
    out: list[Any] = []
    for idx, value in enumerate(values):
        if idx in position_set:
            source_idx = idx - 1 if idx > 0 else idx
            out.append(values[source_idx])
        out.append(value)
    if len(values) in position_set:
        out.append(values[-1] if values else 0)
    return out[:capacity]


def _insert_token_aligned(values: list[Any], positions: list[int], valid: int, capacity: int) -> list[Any]:
    if not positions:
        return values
    position_set = set(positions)
    out: list[Any] = []
    prefix = values[:valid]
    tail = values[valid:]
    for idx, value in enumerate(prefix):
        if idx in position_set:
            source_idx = idx - 1 if idx > 0 else idx
            out.append(prefix[source_idx])
        out.append(value)
    if valid in position_set:
        out.append(prefix[-1] if prefix else 0)
    out.extend(tail)
    return out[:capacity]


def _insert_newline_tokens(ids: list[int], positions: list[int], valid: int, capacity: int) -> list[int]:
    if not positions:
        return ids
    position_set = set(positions)
    out: list[int] = []
    prefix = ids[:valid]
    tail = ids[valid:]
    for idx, token_id in enumerate(prefix):
        if idx in position_set:
            out.append(NL_ID)
        out.append(int(token_id))
    if valid in position_set:
        out.append(NL_ID)
    out.extend(int(x) for x in tail)
    return out[:capacity]


def _shift_offset(value: int, positions: list[int]) -> int:
    """Shift an INCLUSIVE coordinate (chunk start / token-aligned index).

    A token originally at index ``i`` moves to ``i + 1`` for every insertion at
    ``pos <= i`` (``out >= pos``), so it keeps pointing at the SAME token.
    """
    out = int(value)
    for pos in positions:
        if out >= pos:
            out += 1
    return out


def _shift_offset_exclusive(value: int, positions: list[int]) -> int:
    """Shift an EXCLUSIVE end coordinate (one-past-last; chunk end / span end).

    A chunk covering ``[start, end)`` has its last token at ``end - 1``.  An
    insertion at ``pos == end`` lands AFTER that last token (before the first
    excluded token), so the chunk must NOT grow to absorb the inserted <NL>:
    shift only when the insertion is strictly inside the covered range
    (``out > pos``).  Using ``>=`` here would over-shift by +1 and extend the
    prior chunk over the new token.
    """
    out = int(value)
    for pos in positions:
        if out > pos:
            out += 1
    return out


def _shift_span(span: Any, positions: list[int]) -> Any:
    if not isinstance(span, dict):
        return span
    out = dict(span)
    if "start" in out:
        out["start"] = _shift_offset(int(out["start"]), positions)
    if "end" in out:
        out["end"] = _shift_offset_exclusive(int(out["end"]), positions)
    return out


def _update_source_doc_lengths(row: dict[str, Any], positions: list[int], valid: int) -> None:
    lengths = row.get("source_doc_token_lengths")
    if not isinstance(lengths, list):
        return
    if not lengths:
        return
    if len(lengths) == 1:
        row["source_doc_token_lengths"] = [int(lengths[0]) + len(positions)]
        return
    normalized_lengths = [int(length) for length in lengths]
    increments = [0] * len(normalized_lengths)
    cumulative: list[int] = []
    running = 0
    for length in normalized_lengths:
        running += length
        cumulative.append(running)
    for pos in positions:
        source_idx = pos - 1 if pos > 0 else pos
        if 0 <= source_idx < valid:
            for doc_index, end in enumerate(cumulative):
                if source_idx < end:
                    increments[doc_index] += 1
                    break
    row["source_doc_token_lengths"] = [
        length + increments[index]
        for index, length in enumerate(normalized_lengths)
    ]


def repair_row(row: dict[str, Any], pieces: TokenPieces) -> tuple[dict[str, Any], dict[str, Any]]:
    positions = _needed_insert_positions(row, pieces)
    if not positions:
        return row, {"changed": False, "insertions": 0, "overflow": False}

    ids = [int(x) for x in row["input_ids"]]
    capacity = len(ids)
    old_valid = int(row.get("valid_token_count") or _infer_valid_count(ids))
    new_valid = old_valid + len(positions)
    if new_valid > capacity:
        return row, {
            "changed": False,
            "insertions": len(positions),
            "overflow": True,
        }

    repaired = dict(row)
    repaired_ids = _insert_newline_tokens(ids, positions, old_valid, capacity)
    repaired_ids = repaired_ids[:new_valid] + [PAD_ID] * (capacity - new_valid)
    repaired["input_ids"] = repaired_ids
    repaired["target_ids"] = repaired_ids[1:] + [PAD_ID]
    repaired["valid_token_count"] = new_valid
    repaired["slack_tokens"] = capacity - new_valid

    for col in TOKEN_ALIGNED_COLUMNS:
        values = repaired.get(col)
        if isinstance(values, list) and len(values) == capacity:
            repaired[col] = _insert_token_aligned(values, positions, old_valid, capacity)

    # Rebuild loss_mask / trained_token_count from the REPAIRED doc_ids (above),
    # NOT as all-ones.  ``doc_ids`` is a TOKEN_ALIGNED_COLUMN, so it has already
    # been shifted for the inserted <NL> tokens by the loop above.  An all-ones
    # mask would silently train the model to predict document B's first token
    # from document A's last token across unrelated packed commits and inflate
    # trained_token_count by num_docs-1 on every multi-doc row.  We instead apply
    # the producer's exact rule over the valid prefix so the two paths cannot
    # drift (pad region of doc_ids holds the constant pad_doc_id and must be
    # excluded, mirroring how the packer passes only the unpadded valid doc_ids).
    repaired_doc_ids = repaired.get("doc_ids")
    if not isinstance(repaired_doc_ids, list) or len(repaired_doc_ids) != capacity:
        raise ValueError(
            "repair_row: repaired doc_ids missing or wrong length "
            f"(expected list of length {capacity}, got "
            f"type={type(repaired_doc_ids).__name__} "
            f"len={len(repaired_doc_ids) if isinstance(repaired_doc_ids, list) else 'n/a'}); "
            "cannot recompute inter-document loss_mask"
        )
    valid_doc_ids = [int(x) for x in repaired_doc_ids[:new_valid]]
    loss_mask = _loss_mask_for_packed_docs(valid_doc_ids, target_length=capacity)
    repaired["loss_mask"] = loss_mask
    repaired["trained_token_count"] = sum(loss_mask)

    # token_chunk_starts are INCLUSIVE starts; token_chunk_ends are EXCLUSIVE
    # (one-past-last, mirroring the producer in
    # cppmega/data/nanochat_pipeline/tokenized_enriched.py).  They must use
    # different shift semantics at an insertion that falls on a chunk edge.
    starts = repaired.get("token_chunk_starts")
    if isinstance(starts, list):
        repaired["token_chunk_starts"] = [_shift_offset(int(v), positions) for v in starts]
    ends = repaired.get("token_chunk_ends")
    if isinstance(ends, list):
        repaired["token_chunk_ends"] = [_shift_offset_exclusive(int(v), positions) for v in ends]

    for col in TOKEN_SPAN_COLUMNS:
        values = repaired.get(col)
        if isinstance(values, list):
            repaired[col] = [_shift_span(span, positions) for span in values]

    _update_source_doc_lengths(repaired, positions, old_valid)
    return repaired, {
        "changed": True,
        "insertions": len(positions),
        "overflow": False,
    }


def _repair_file(
    path: Path,
    pieces: TokenPieces,
    *,
    dry_run: bool,
    backup_suffix: str,
    drop_unrepairable: bool,
    compression_level: int,
) -> dict[str, Any]:
    scan = _scan_file(path, pieces)
    if dry_run or (not scan["changed_rows"] and not scan["overflow_rows"]):
        scan["path"] = str(path)
        scan.setdefault("dropped_rows", 0)
        return scan

    table = pq.read_table(path)
    rows = table.to_pylist()
    changed_rows = 0
    insertions = 0
    overflow_rows = 0
    dropped_rows = 0
    repaired_rows: list[dict[str, Any]] = []
    for row in rows:
        repaired, info = repair_row(row, pieces)
        changed_rows += int(info["changed"])
        insertions += int(info["insertions"] if info["changed"] else 0)
        if info["overflow"]:
            if drop_unrepairable:
                dropped_rows += 1
                continue
            overflow_rows += 1
        repaired_rows.append(repaired)

    if (changed_rows or dropped_rows) and not dry_run:
        if backup_suffix:
            backup = path.with_name(path.name + backup_suffix)
            if not backup.exists():
                shutil.copy2(path, backup)
        repaired_table = pa.Table.from_pylist(repaired_rows, schema=table.schema)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=path.name,
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            pq.write_table(
                repaired_table,
                tmp_path,
                compression="zstd",
                compression_level=compression_level,
            )
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)

    return {
        "path": str(path),
        "rows": len(rows),
        "changed_rows": changed_rows,
        "insertions": insertions,
        "overflow_rows": overflow_rows,
        "dropped_rows": dropped_rows,
    }


def _scan_file(path: Path, pieces: TokenPieces) -> dict[str, Any]:
    table = pq.read_table(path, columns=["input_ids", "valid_token_count"])
    changed_rows = 0
    insertions = 0
    overflow_rows = 0
    rows = table.to_pylist()
    for row in rows:
        positions = _needed_insert_positions(row, pieces)
        if not positions:
            continue
        ids = [int(x) for x in row["input_ids"]]
        valid = int(row.get("valid_token_count") or _infer_valid_count(ids))
        if valid + len(positions) > len(ids):
            overflow_rows += 1
        else:
            changed_rows += 1
            insertions += len(positions)
    return {
        "path": str(path),
        "rows": len(rows),
        "changed_rows": changed_rows,
        "insertions": insertions,
        "overflow_rows": overflow_rows,
        "dropped_rows": 0,
    }


def _iter_parquet_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".parquet":
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.parquet")))
    return sorted(set(files))


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("outputs/reindexed_commits")],
        help="Parquet file or directory roots to audit/repair.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", type=Path, default=Path("outputs/parquet_boundary_fix_report.json"))
    ap.add_argument("--backup-suffix", default=".pre_boundary_fix")
    ap.add_argument("--fail-on-remaining", action="store_true")
    ap.add_argument(
        "--drop-unrepairable",
        action="store_true",
        help="Drop rows that need marker-boundary repair but have no padding slack.",
    )
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--compression-level", type=int, default=6)
    args = ap.parse_args(argv)

    tokenizer = get_tokenizer()
    pieces = TokenPieces(tokenizer)
    files = _iter_parquet_files(args.roots)
    if not files:
        raise SystemExit("no parquet files matched roots: " + ", ".join(str(x) for x in args.roots))

    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                _repair_file,
                path,
                pieces,
                dry_run=args.dry_run,
                backup_suffix=args.backup_suffix,
                drop_unrepairable=args.drop_unrepairable,
                compression_level=args.compression_level,
            )
            for path in files
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            reports.append(future.result())
            if idx % 100 == 0:
                print(f"processed {idx}/{len(files)} parquet files", flush=True)

    total = {
        "files": len(files),
        "rows": sum(int(x["rows"]) for x in reports),
        "changed_files": sum(1 for x in reports if int(x["changed_rows"]) > 0),
        "changed_rows": sum(int(x["changed_rows"]) for x in reports),
        "insertions": sum(int(x["insertions"]) for x in reports),
        "overflow_rows": sum(int(x["overflow_rows"]) for x in reports),
        "dropped_rows": sum(int(x.get("dropped_rows", 0)) for x in reports),
        "dry_run": bool(args.dry_run),
        "drop_unrepairable": bool(args.drop_unrepairable),
    }
    payload = {
        "total": total,
        "changed": [x for x in reports if int(x["changed_rows"]) or int(x["overflow_rows"])],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(total, indent=2), flush=True)
    remaining_changed = total["changed_rows"] if args.dry_run else 0
    remaining_overflow = total["overflow_rows"]
    if args.fail_on_remaining and (remaining_changed or remaining_overflow):
        raise SystemExit(
            "unrepaired marker-boundary rows: "
            f"changed_rows={remaining_changed} overflow_rows={remaining_overflow}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
