#!/usr/bin/env python
"""Full-scan corpus inventory over all 4 row-based parquet datasets.

Fail-fast: any unexpected schema / missing column / parse failure raises with
WHERE + WHAT. No silent fallbacks.
"""
import os
import re
import sys
import json
import glob
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq
import pyarrow.compute as pc

BASE = "/Users/dave/sources/parquet"

# kind: CODE => no repo recoverable (verified); COMMITS => parse Repository: from text
DATASETS = [
    ("clang_semantic_4k_v10", "CODE"),
    ("clang_commits_4k_v1", "COMMITS"),
    ("treesitter_compilable_4k_v9", "CODE"),
    ("enriched_commits_4k_v9", "COMMITS"),
]

REPO_RE = re.compile(r"Repository:\s*(\S+)")


def scan_shard(args):
    dataset, kind, path = args
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    if "actual_token_count" not in names:
        raise RuntimeError(f"WHERE={path}: WHAT=missing 'actual_token_count' column; have {names}")
    if "text" not in names:
        raise RuntimeError(f"WHERE={path}: WHAT=missing 'text' column; have {names}")

    rows = 0
    real_tokens = 0
    repo_counter = Counter()
    matched = 0  # rows where Repository: matched (COMMITS only)

    cols = ["actual_token_count"] + (["text"] if kind == "COMMITS" else [])
    for batch in pf.iter_batches(batch_size=20000, columns=cols):
        n = batch.num_rows
        rows += n
        atc = batch.column("actual_token_count")
        s = pc.sum(atc).as_py()
        if s is None:
            raise RuntimeError(f"WHERE={path}: WHAT=actual_token_count sum is None (all-null batch)")
        real_tokens += int(s)
        if kind == "COMMITS":
            for txt in batch.column("text").to_pylist():
                if txt is None:
                    raise RuntimeError(f"WHERE={path}: WHAT=null text value in COMMITS dataset")
                m = REPO_RE.search(txt)
                if m:
                    repo_counter[m.group(1)] += 1
                    matched += 1

    return {
        "dataset": dataset,
        "kind": kind,
        "path": path,
        "rows": rows,
        "real_tokens": real_tokens,
        "repos": dict(repo_counter),
        "matched": matched,
        "size_on_disk": os.path.getsize(path),
    }


def main():
    tasks = []
    per_ds_paths = {}
    for ds, kind in DATASETS:
        shards = sorted(glob.glob(os.path.join(BASE, ds, "*.parquet")))
        if not shards:
            raise RuntimeError(f"WHERE={BASE}/{ds}: WHAT=no parquet shards found")
        per_ds_paths[ds] = shards
        for p in shards:
            tasks.append((ds, kind, p))

    print(f"Total shards to scan: {len(tasks)}", file=sys.stderr, flush=True)

    results = {ds: {"kind": k} for ds, k in DATASETS}
    for ds, _ in DATASETS:
        results[ds].update(
            shards=len(per_ds_paths[ds]), rows=0, real_tokens=0,
            repos=Counter(), matched=0, size_on_disk=0,
        )

    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(scan_shard, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            ds = r["dataset"]
            agg = results[ds]
            agg["rows"] += r["rows"]
            agg["real_tokens"] += r["real_tokens"]
            agg["matched"] += r["matched"]
            agg["size_on_disk"] += r["size_on_disk"]
            agg["repos"].update(r["repos"])
            done += 1
            print(f"[{done}/{len(tasks)}] {os.path.basename(r['path'])} ({ds}) "
                  f"rows={r['rows']} tok={r['real_tokens']}", file=sys.stderr, flush=True)

    # Build JSON + markdown
    json_out = {"base": BASE, "datasets": []}
    md = []
    md.append("# C++ Mega Corpus Inventory\n")
    md.append(f"_Full scan of all shards under `{BASE}` (no sampling)._\n")
    md.append(f"_Generated: 2026-06-23. padded@4k = rows * 4096 (4k context budget)._\n")

    grand_rows = 0
    grand_tokens = 0

    for ds, kind in DATASETS:
        agg = results[ds]
        repos = agg["repos"]
        distinct = len(repos)
        if kind == "CODE":
            note = ("No repo recoverable: 'repo'/'commit' columns empty, "
                    "constituent_provenance empty, no 'Repository:' text header. distinct_repos=0.")
        else:
            note = (f"Repository: parsed from text in {agg['matched']}/{agg['rows']} rows "
                    f"({100.0*agg['matched']/agg['rows']:.2f}%).")
        padded = agg["rows"] * 4096
        grand_rows += agg["rows"]
        grand_tokens += agg["real_tokens"]

        sorted_repos = sorted(repos.items(), key=lambda kv: (-kv[1], kv[0]))

        json_out["datasets"].append({
            "name": ds, "kind": kind, "shards": agg["shards"], "rows": agg["rows"],
            "real_tokens": agg["real_tokens"], "padded_at_4k": padded,
            "size_on_disk_bytes": agg["size_on_disk"], "distinct_repos": distinct,
            "matched_rows": agg["matched"], "note": note,
            "repo_distribution": dict(sorted_repos),
        })

        gib = agg["size_on_disk"] / (1024**3)
        md.append(f"\n## {ds} ({kind})\n")
        md.append("| metric | value |")
        md.append("|---|---|")
        md.append(f"| kind | {kind} |")
        md.append(f"| shards | {agg['shards']} |")
        md.append(f"| rows | {agg['rows']:,} |")
        md.append(f"| real_tokens (sum actual_token_count) | {agg['real_tokens']:,} |")
        md.append(f"| padded@4k (rows*4096) | {padded:,} |")
        md.append(f"| size_on_disk | {agg['size_on_disk']:,} bytes ({gib:.2f} GiB) |")
        md.append(f"| distinct_repos | {distinct} |")
        md.append(f"| note | {note} |")
        md.append("")
        if distinct:
            md.append("### Full repo distribution (sorted by count desc, then name)\n")
            md.append("| repo | count |")
            md.append("|---|---|")
            for repo, cnt in sorted_repos:
                md.append(f"| {repo} | {cnt:,} |")
        else:
            md.append("_No repo distribution (see note)._\n")

    md.append("\n## Totals\n")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| total rows | {grand_rows:,} |")
    md.append(f"| total real_tokens | {grand_tokens:,} |")

    json_out["totals"] = {"rows": grand_rows, "real_tokens": grand_tokens}

    with open("/Volumes/external/sources/cppmega/corpus_inventory.md", "w") as f:
        f.write("\n".join(md) + "\n")
    with open("/Volumes/external/sources/cppmega/corpus_inventory.json", "w") as f:
        json.dump(json_out, f, indent=2)

    print("DONE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
