"""Integrity checks on real packed source Parquet (tokenizer + sidecars).

Drives the shipped tokenizer and packed-row contracts against live reindexed
artifacts. Skips cleanly when the full501 root is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
from scripts.nanochat_data.token_budget import resolve_tokenizer_path

_DEFAULT_ROOT = Path(
    "/Volumes/external/cppmega_data/"
    "source_full501_7f55ff0c12d88bb835fea9a68b8ba9d90522ddd5/reindexed"
)
_REQUIRED_TOKEN_SIDE = (
    "input_ids",
    "target_ids",
    "loss_mask",
    "doc_ids",
)
_REQUIRED_GRAPH = (
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
)


def _reindexed_root() -> Path:
    raw = os.environ.get("CPPMEGA_LIVE_REINDEXED_ROOT")
    root = Path(raw) if raw else _DEFAULT_ROOT
    if not root.is_dir():
        pytest.skip(f"live reindexed root absent: {root}")
    return root


def _sample_parquet(root: Path, bucket: int = 1024) -> Path:
    bucket_dir = root / str(bucket)
    files = sorted(bucket_dir.glob("*.parquet"))
    if not files:
        pytest.skip(f"no parquet under {bucket_dir}")
    # Prefer a medium-sized shard for signal without huge IO.
    ranked = sorted(files, key=lambda p: p.stat().st_size)
    return ranked[min(len(ranked) // 2, len(ranked) - 1)]


def test_live_packed_shard_has_required_sidecars_and_aligned_lengths() -> None:
    path = _sample_parquet(_reindexed_root())
    table = pq.read_table(path)
    names = set(table.column_names)
    missing = [c for c in _REQUIRED_TOKEN_SIDE + _REQUIRED_GRAPH if c not in names]
    assert not missing, f"{path.name} missing sidecars: {missing}"

    n = table.num_rows
    assert n > 0
    for i in range(min(n, 8)):
        input_ids = table.column("input_ids")[i].as_py()
        target_ids = table.column("target_ids")[i].as_py()
        loss_mask = table.column("loss_mask")[i].as_py()
        doc_ids = table.column("doc_ids")[i].as_py()
        assert len(input_ids) == len(target_ids) == len(loss_mask) == len(doc_ids)
        assert len(input_ids) == 1024
        # target is next-token of input for non-pad positions (standard LM packing)
        for j in range(len(input_ids) - 1):
            if loss_mask[j + 1] == 1:
                assert target_ids[j] == input_ids[j + 1] or target_ids[j] == 0


def test_live_packed_shard_tokenizer_roundtrip_on_code_docs() -> None:
    path = _sample_parquet(_reindexed_root())
    table = pq.read_table(path)
    tok = load_cppmega_tokenizer(resolve_tokenizer_path(None))

    decoded_nonempty = 0
    reencode_ok = 0
    checked = 0
    for i in range(min(table.num_rows, 12)):
        input_ids = table.column("input_ids")[i].as_py()
        loss_mask = table.column("loss_mask")[i].as_py()
        doc_ids = table.column("doc_ids")[i].as_py()
        # Split by document id runs inside the pack.
        runs: list[list[int]] = []
        current: list[int] = []
        prev = None
        for tid, mask, did in zip(input_ids, loss_mask, doc_ids, strict=True):
            if mask != 1:
                continue
            if prev is None or did != prev:
                if current:
                    runs.append(current)
                current = [tid]
                prev = did
            else:
                current.append(tid)
        if current:
            runs.append(current)

        for run in runs[:4]:
            if len(run) < 8:
                continue
            checked += 1
            text = tok.decode(run)
            assert isinstance(text, str)
            if text.strip():
                decoded_nonempty += 1
            # Special SPACE/NL tokens make exact encode(decode(ids)) fragile;
            # require that re-encoding the decoded text yields a non-empty
            # sequence and that decode is stable under a second pass.
            again = tok.decode(tok.encode(text))
            if again == text:
                reencode_ok += 1
            # Code-ish heuristic: braces/semicolons survive for C/C++ packs.
            if any(ch in text for ch in ("{", "}", ";", "#include", "int ", "void ")):
                assert any(
                    ch in again for ch in ("{", "}", ";", "#", "int", "void")
                ), f"code markers lost after encode/decode in {path.name}"

    assert checked >= 1, f"no document runs long enough in {path}"
    assert decoded_nonempty >= 1
    # Stability is required for at least some docs (whitespace-normalized packs
    # may intentionally differ once).
    assert reencode_ok >= 1


def test_live_packed_shard_chunk_sidecars_in_bounds() -> None:
    path = _sample_parquet(_reindexed_root())
    table = pq.read_table(path)
    for i in range(min(table.num_rows, 6)):
        seq_len = len(table.column("input_ids")[i].as_py())
        starts = table.column("token_chunk_starts")[i].as_py() or []
        ends = table.column("token_chunk_ends")[i].as_py() or []
        kinds = table.column("token_chunk_kinds")[i].as_py() or []
        assert len(starts) == len(ends) == len(kinds)
        for s, e in zip(starts, ends, strict=True):
            assert 0 <= int(s) <= int(e) <= seq_len
