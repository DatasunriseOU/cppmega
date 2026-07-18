#!/usr/bin/env python3
"""Export carefully repeated base-library functions into a 16k-only stream.

The normal code/commit streams use ``semantic_chunk:v1`` claims with
``max_count=1`` so exact function/class/type forms do not overlap across
1024/2048/4096/8192 buckets. This script is the explicit exception lane: it
samples foundational base-library function bodies from the global cross-repo
symbol index and emits them ONLY into a 16384-token output stream, while using
the SAME claim namespace with ``max_count=3``.

That means the global invariant is:

  * normal streams: a semantic chunk form can appear once;
  * base16k stream: selected base functions may top up the same form to at most
    three total appearances across the whole training period.

No fallback path: the symbol DB, dedup DB, tokenizer, materializer and packer
must all work or the script raises.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "tools" / "clang_indexer") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "clang_indexer"))

import streaming_reindex as sr  # noqa: E402
from streaming_reindex_commits import recompress_zstd_max  # noqa: E402
from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer  # noqa: E402
from dedup_store import DedupStore  # noqa: E402


DEFAULT_SYMBOL_INDEX = REPO_ROOT / "outputs" / "crossrepo" / "global_symbols.sqlite"
DEFAULT_DEDUP_DB = REPO_ROOT / "outputs" / "dedup_seen.sqlite"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "reindexed_base16k"
SEMANTIC_CHUNK_NAMESPACE = "semantic_chunk:v1"
TARGET_LENGTH = 16384
BASE_LIB_ORDER = {
    "boost": 0,
    "abseil": 1,
    "folly": 2,
    "openssl": 3,
    "boringssl": 4,
    "protobuf": 5,
    "eigen": 6,
    "fmt": 7,
    "glib": 8,
    "std": 9,
    "libc": 10,
}


@dataclass(frozen=True)
class BaseSymbol:
    qname: str
    base_lib: str
    base_repo: str
    kind: int
    sym_type: str
    file: str
    line: int
    token_est: int
    body_len: int
    text: str


def _connect_symbol_index(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"--symbol-index does not exist: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_candidate_symbols(
    symbol_index: Path,
    *,
    libs: tuple[str, ...],
    limit: int,
    offset: int,
    min_body_len: int,
    max_token_est: int,
) -> list[BaseSymbol]:
    conn = _connect_symbol_index(symbol_index)
    try:
        params: list[object] = [int(min_body_len), int(max_token_est)]
        where = [
            "sym_type='func'",
            "body_len >= ?",
            "token_est <= ?",
            "text IS NOT NULL",
            "length(text) > 0",
        ]
        if libs:
            where.append("base_lib IN (%s)" % ",".join("?" for _ in libs))
            params.extend(libs)
        sql = (
            "SELECT qname, base_lib, base_repo, kind, sym_type, file, line, "
            "token_est, body_len, text "
            "FROM symbols WHERE "
            + " AND ".join(where)
        )
        rows = list(conn.execute(sql, params))
    finally:
        conn.close()

    symbols = [
        BaseSymbol(
            qname=str(r["qname"]),
            base_lib=str(r["base_lib"]),
            base_repo=str(r["base_repo"]),
            kind=int(r["kind"]),
            sym_type=str(r["sym_type"]),
            file=str(r["file"]),
            line=int(r["line"]),
            token_est=int(r["token_est"]),
            body_len=int(r["body_len"]),
            text=str(r["text"]),
        )
        for r in rows
    ]
    symbols.sort(
        key=lambda s: (
            BASE_LIB_ORDER.get(s.base_lib, 100),
            -s.body_len,
            -s.token_est,
            s.qname,
            s.file,
            s.line,
        )
    )
    end = None if limit < 0 else offset + limit
    return symbols[offset:end]


def _base_doc(symbol: BaseSymbol, *, repeat_index: int, token_count: int) -> dict:
    text = symbol.text
    provenance = [{
        "kind": "base16k_repeat",
        "qname": symbol.qname,
        "base_lib": symbol.base_lib,
        "base_repo": symbol.base_repo,
        "file": symbol.file,
        "line": symbol.line,
        "repeat_index": repeat_index,
    }]
    return {
        "text": text,
        "source_text": text,
        "source_doc_id": (
            f"base16k:{symbol.base_lib}:{symbol.qname}:r{repeat_index}"
        ),
        "actual_token_count": token_count,
        "structure_ids": [int(symbol.kind)] * len(text),
        "chunk_boundaries": [{
            "start": 0,
            "end": len(text),
            "kind": int(symbol.kind),
            "dep_level": 0,
            "name": symbol.qname.split("::")[-1],
        }],
        "call_edges": [],
        "type_edges": [],
        "ast_depth": [],
        "sibling_index": [],
        "ast_node_type": [],
        "symbol_ids": [],
        "call_targets": [],
        "type_refs": [],
        "def_use": [],
        "repo": symbol.base_repo,
        "filepath": symbol.file,
        "doc_type": "base16k_repeat",
        "language": "cpp",
        "constituent_provenance_json": json.dumps(
            provenance,
            separators=(",", ":"),
        ),
    }


def build_repeat_docs(
    symbols: list[BaseSymbol],
    *,
    tokenizer_path: Path,
    dedup_db: Path,
    repeats: int,
    max_count: int,
    target_length: int = TARGET_LENGTH,
) -> tuple[list[dict], dict[str, int]]:
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if max_count < 1:
        raise ValueError("--max-count must be >= 1")
    tok = load_cppmega_tokenizer(str(tokenizer_path))
    store = DedupStore(str(dedup_db), near=False, commit_every=1)
    docs: list[dict] = []
    stats = {
        "candidate_symbols": len(symbols),
        "symbols_emitted": 0,
        "symbols_saturated": 0,
        "docs_emitted": 0,
        "claims_rejected": 0,
        "too_long": 0,
        "empty": 0,
    }
    try:
        for symbol in symbols:
            token_ids = tok.encode(symbol.text)
            if not token_ids:
                stats["empty"] += 1
                continue
            if len(token_ids) > target_length:
                stats["too_long"] += 1
                continue
            emitted_for_symbol = 0
            for repeat_index in range(int(repeats)):
                ok = store.claim_chunk_tokens(
                    token_ids,
                    namespace=SEMANTIC_CHUNK_NAMESPACE,
                    max_count=int(max_count),
                )
                if not ok:
                    stats["claims_rejected"] += 1
                    break
                docs.append(
                    _base_doc(
                        symbol,
                        repeat_index=repeat_index,
                        token_count=len(token_ids),
                    )
                )
                emitted_for_symbol += 1
            if emitted_for_symbol:
                stats["symbols_emitted"] += 1
                stats["docs_emitted"] += emitted_for_symbol
            else:
                stats["symbols_saturated"] += 1
    finally:
        store.close()
    return docs, stats


def _write_jsonl(docs: list[dict], path: Path) -> None:
    if not docs:
        raise RuntimeError("base16k sampler emitted zero docs")
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")


def _append_output(shard_name: str, packed: Path, output_root: Path) -> dict:
    out_dir = output_root / str(TARGET_LENGTH)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{shard_name}.parquet"
    dest.write_bytes(packed.read_bytes())
    recompress_zstd_max(dest)
    return sr._parquet_stats(dest, TARGET_LENGTH)


def export_base16k(args: argparse.Namespace) -> dict:
    symbol_index = Path(args.symbol_index)
    dedup_db = Path(args.dedup_db)
    tokenizer_path = Path(args.tokenizer_path)
    output_root = Path(args.output_root)
    libs = tuple(x.strip() for x in args.libs.split(",") if x.strip())
    shard_name = args.shard_name or f"base16k_{int(args.offset):08d}"

    symbols = load_candidate_symbols(
        symbol_index,
        libs=libs,
        limit=int(args.limit),
        offset=int(args.offset),
        min_body_len=int(args.min_body_len),
        max_token_est=int(args.max_token_est),
    )
    docs, stats = build_repeat_docs(
        symbols,
        tokenizer_path=tokenizer_path,
        dedup_db=dedup_db,
        repeats=int(args.repeats),
        max_count=int(args.max_count),
        target_length=TARGET_LENGTH,
    )

    with tempfile.TemporaryDirectory(prefix="base16k_export_") as tmp:
        work = Path(tmp)
        jsonl = work / f"{shard_name}.jsonl"
        _write_jsonl(docs, jsonl)
        tok = sr.stage_materialize(
            shard_name,
            jsonl,
            work,
            memory_limit_gb=float(args.memory_limit_gb),
        )
        packed = sr.stage_pack(shard_name, tok, TARGET_LENGTH, work)
        parquet_stats = _append_output(shard_name, packed, output_root)

    return {
        "source": "base16k",
        "symbol_index": str(symbol_index),
        "dedup_db": str(dedup_db),
        "output_root": str(output_root),
        "shard": shard_name,
        "target_length": TARGET_LENGTH,
        "libs": list(libs),
        "stats": stats,
        "lengths": {str(TARGET_LENGTH): parquet_stats},
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol-index", default=str(DEFAULT_SYMBOL_INDEX))
    p.add_argument("--dedup-db", default=str(DEFAULT_DEDUP_DB))
    p.add_argument(
        "--tokenizer-path",
        default=str(REPO_ROOT / "cppmega" / "tokenizer" / "tokenizer.json"),
    )
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument(
        "--libs",
        default="boost,abseil,folly,openssl,boringssl,protobuf,eigen,fmt,glib,std,libc",
        help="Comma-separated base_lib keys to sample from.",
    )
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-count", type=int, default=3)
    p.add_argument("--min-body-len", type=int, default=200)
    p.add_argument("--max-token-est", type=int, default=TARGET_LENGTH)
    p.add_argument("--memory-limit-gb", type=float, default=10.0)
    p.add_argument("--shard-name", default=None)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    print(json.dumps(export_base16k(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
