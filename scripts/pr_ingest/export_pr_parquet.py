#!/usr/bin/env python3
"""Export PR discussions from prs.sqlite into route-by-fit training parquet.

This is the standalone PR stream companion to the commit pipeline. Commit docs
already inject ``record['pr_discussion']`` at the head of PRE/POST/diff samples;
this script emits the same PR discussion content as its own document stream for
curriculum mixes that want explicit ``pr`` batches.

The output is intentionally compatible with the existing conveyor layout:

    outputs/reindexed_pr/{1024,2048,4096,...}/pr_discussions_<repo>_<offset>.parquet

RULE #1: every stage uses the existing materializer/packer path and raises on
missing input, empty output, or malformed store rows. There is no fallback path.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cppmega.data.symbol_identity import (  # noqa: E402
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
)
from scripts.pr_ingest.pr_store import connect, get_by_pr  # noqa: E402
from scripts.pr_ingest.render_discussion import render_discussion  # noqa: E402


DEFAULT_STORE = REPO_ROOT / "outputs" / "pr_ingest" / "prs.sqlite"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "reindexed_pr"
ZSTD_LEVELS = (1024, 2048, 4096, 8192, 16384)


def _load_pipeline():
    """Load the canonical root materializer or fail closed with its missing module."""
    try:
        import streaming_reindex as sr
        from streaming_reindex_commits import route_by_fit, recompress_zstd_max
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PR parquet export requires root scripts/streaming_reindex.py and "
            "scripts/streaming_reindex_commits.py; no sibling or degraded fallback "
            "is permitted"
        ) from exc
    return sr, route_by_fit, recompress_zstd_max


def _comment_safe(text: str) -> str:
    return text.replace("*/", "* /")


def _render_training_doc(rec: dict) -> str:
    # The standalone PR corpus must retain the complete assembled discussion.
    # render_discussion's defaults are appropriate for embedding PR context in
    # an atomic commit sample, but would silently clip bodies/items and cap
    # comments before this lossless splitter sees them.
    discussion = render_discussion(
        rec,
        max_comments=len(rec.get("comments") or []),
        max_reviews=len(rec.get("reviews") or []),
        max_body_chars=sys.maxsize,
        max_item_chars=sys.maxsize,
        max_total_chars=sys.maxsize,
    )
    if not discussion:
        return ""
    lines = [
        "/*",
        "@doc_type pr_discussion",
        f"@repo {rec['repo']}",
        f"@pr {rec['pr_number']}",
    ]
    sha = rec.get("merge_commit_sha")
    if sha:
        lines.append(f"@merge_commit_sha {sha}")
    lines.extend(["@discussion", _comment_safe(discussion), "*/", ""])
    return "\n".join(lines)


def _iter_pr_keys(
    conn: sqlite3.Connection,
    *,
    repo: str | None,
    offset: int,
    limit: int | None,
):
    sql = "SELECT repo, pr_number FROM prs"
    params: list[object] = []
    if repo:
        sql += " WHERE repo=?"
        params.append(repo)
    sql += " ORDER BY repo, pr_number LIMIT ? OFFSET ?"
    params.append(-1 if limit is None else int(limit))
    params.append(int(offset))
    yield from conn.execute(sql, params)


def _count_pr_keys(
    conn: sqlite3.Connection,
    *,
    repo: str | None,
    offset: int,
    limit: int | None,
) -> int:
    sql = "SELECT COUNT(*) AS n FROM (SELECT 1 FROM prs"
    params: list[object] = []
    if repo:
        sql += " WHERE repo=?"
        params.append(repo)
    sql += " ORDER BY repo, pr_number LIMIT ? OFFSET ?)"
    params.append(-1 if limit is None else int(limit))
    params.append(int(offset))
    return int(conn.execute(sql, params).fetchone()["n"])


def _write_pr_jsonl(
    conn: sqlite3.Connection,
    out_jsonl: Path,
    *,
    repo: str | None,
    offset: int,
    limit: int | None,
) -> int:
    n = 0
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in _iter_pr_keys(conn, repo=repo, offset=offset, limit=limit):
            rec = get_by_pr(conn, row["repo"], int(row["pr_number"]))
            if rec is None:
                raise RuntimeError(
                    f"PR key disappeared during export: {row['repo']}#{row['pr_number']}"
                )
            text = _render_training_doc(rec)
            if not text:
                continue
            payload = {
                "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
                SYMBOL_IDENTITIES_COLUMN: [],
                "text": text,
                "source_text": text,
                "repo": rec["repo"],
                "filepath": f"PR#{rec['pr_number']}",
                "doc_type": "pr_discussion",
                "language": "pr",
                "pr_number": int(rec["pr_number"]),
                "commit_hash": rec.get("merge_commit_sha") or "",
                "constituent_provenance_json": json.dumps(
                    [{
                        "kind": "pr_discussion",
                        "repo": rec["repo"],
                        "pr_number": int(rec["pr_number"]),
                        "merge_commit_sha": rec.get("merge_commit_sha") or "",
                    }],
                    separators=(",", ":"),
                ),
            }
            fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
            n += 1
    if n == 0:
        raise RuntimeError(
            f"PR export produced zero rendered docs (repo={repo!r}, offset={offset}, limit={limit})"
        )
    return n


def _append_output(
    shard_name: str,
    packed: Path,
    target_length: int,
    output_root: Path,
    *,
    pipeline,
) -> dict:
    sr, _route_by_fit, recompress_zstd_max = pipeline
    out_dir = output_root / str(target_length)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{shard_name}.parquet"
    dest.write_bytes(packed.read_bytes())
    recompress_zstd_max(dest)
    return sr._parquet_stats(dest, target_length)


def _repo_slug(repo: str | None) -> str:
    if not repo:
        return "all"
    return repo.replace("/", "__").replace(":", "_")


def _shard_name(repo: str | None, offset: int) -> str:
    return f"pr_discussions_{_repo_slug(repo)}_{int(offset):08d}"


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"done": {}, "failed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def export_pr_parquet(args: argparse.Namespace) -> dict:
    store = Path(args.store)
    if not store.exists():
        raise FileNotFoundError(f"--store does not exist: {store}")
    output_root = Path(args.output_root)
    lengths = tuple(sorted(int(x) for x in args.target_lengths.split(",") if x.strip()))
    if not lengths:
        raise ValueError("--target-lengths produced no lengths")

    pipeline = _load_pipeline()
    sr, route_by_fit, _recompress_zstd_max = pipeline
    materialize_budget = sr.lossless_materialize_budget(lengths)
    conn = connect(str(store), create=False)
    try:
        shard_name = _shard_name(args.repo, int(args.offset))
        with tempfile.TemporaryDirectory(prefix="pr_parquet_export_") as tmp:
            work = Path(tmp)
            jsonl = work / f"{shard_name}.jsonl"
            n_docs = _write_pr_jsonl(
                conn,
                jsonl,
                repo=args.repo,
                offset=int(args.offset),
                limit=args.limit,
            )
            tok = sr.stage_materialize(
                shard_name,
                jsonl,
                work,
                memory_limit_gb=float(args.memory_limit_gb),
                max_tokens=materialize_budget,
                fixed_shape_max_tokens=max(lengths),
            )
            materialize_stats = sr.read_materialize_stats(tok)
            routed = route_by_fit(tok, lengths, work / "routed")
            if not routed:
                raise RuntimeError(f"no PR docs routed for {jsonl}")
            per_length: dict[str, dict] = {}
            for length, route_parquet in sorted(routed.items()):
                packed = sr.stage_pack(shard_name, route_parquet, length, work)
                per_length[str(length)] = _append_output(
                    shard_name, packed, length, output_root, pipeline=pipeline,
                )
    finally:
        conn.close()
    return {
        "source": "pr",
        "store": str(store),
        "output_root": str(output_root),
        "shard": shard_name,
        "rendered_docs": n_docs,
        "lengths": per_length,
        "materialize_stats": materialize_stats,
    }


def export_pr_parquet_batches(args: argparse.Namespace) -> dict:
    store = Path(args.store)
    if not store.exists():
        raise FileNotFoundError(f"--store does not exist: {store}")
    manifest_path = Path(args.manifest) if args.manifest else Path(args.output_root) / "_done.json"
    manifest = _load_manifest(manifest_path)
    resume = not args.no_resume
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    sr, _route_by_fit, _recompress_zstd_max = _load_pipeline()

    conn = connect(str(store), create=False)
    try:
        offset = int(args.offset)
        max_shards = args.max_shards
        shards: list[dict] = []
        totals_by_length: dict[str, dict[str, int]] = {}
        while True:
            shard_name = _shard_name(args.repo, offset)
            done_key = f"{_repo_slug(args.repo)}:{offset}"
            n_keys = _count_pr_keys(
                conn,
                repo=args.repo,
                offset=offset,
                limit=batch_size,
            )
            if n_keys == 0:
                break
            if resume and done_key in manifest.get("done", {}):
                info = manifest["done"][done_key]
                shards.append({"shard": shard_name, "skipped": True, **info})
            else:
                shard_args = argparse.Namespace(**vars(args))
                shard_args.limit = batch_size
                shard_args.offset = offset
                try:
                    info = export_pr_parquet(shard_args)
                except Exception as exc:
                    manifest.setdefault("failed", {})[done_key] = {
                        "offset": offset,
                        "limit": batch_size,
                        "repo": args.repo,
                        "stage": "export_pr_parquet",
                        "detail": str(exc)[:2000],
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    _save_manifest(manifest_path, manifest)
                    raise
                manifest.setdefault("done", {})[done_key] = info
                manifest.get("failed", {}).pop(done_key, None)
                _save_manifest(manifest_path, manifest)
                shards.append(info)

            last = shards[-1]
            for length, st in last.get("lengths", {}).items():
                agg = totals_by_length.setdefault(
                    length,
                    {"rows": 0, "valid_tokens": 0, "pad_tokens": 0, "capacity_tokens": 0},
                )
                for key in ("rows", "valid_tokens", "pad_tokens", "capacity_tokens"):
                    agg[key] += int(st.get(key, 0))

            offset += batch_size
            if max_shards is not None and len(shards) >= int(max_shards):
                break
        return {
            "source": "pr",
            "store": str(store),
            "output_root": str(Path(args.output_root)),
            "manifest": str(manifest_path),
            "repo": args.repo,
            "start_offset": int(args.offset),
            "batch_size": batch_size,
            "next_offset": offset,
            "shards": shards,
            "n_shards": len(shards),
            "lengths": totals_by_length,
            "materialize_split_totals": sr.summarize_materialize_stats(shards),
        }
    finally:
        conn.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=str(DEFAULT_STORE))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--target-lengths", default=",".join(str(x) for x in ZSTD_LEVELS))
    p.add_argument("--repo", default=None, help="Optional owner/repo filter.")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=10_000)
    p.add_argument("--all", action="store_true",
                   help="Export every PR row from --offset in resumable batches.")
    p.add_argument("--batch-size", type=int, default=10_000,
                   help="Rows per shard when --all is set.")
    p.add_argument("--max-shards", type=int, default=None,
                   help="Optional cap on number of shards exported in this run.")
    p.add_argument("--manifest", default=None,
                   help="Resume manifest for --all batched export. Default: "
                        "<output-root>/_done.json.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore completed PR export shards in --manifest.")
    p.add_argument("--memory-limit-gb", type=float, default=10.0)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = export_pr_parquet_batches(args) if args.all else export_pr_parquet(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
