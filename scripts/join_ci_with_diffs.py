#!/usr/bin/env python3
"""Option B: Join commit diffs with CI build logs.

Creates paired training documents: "here's what changed" + "here's what CI said".
Uses extract_cache for diffs and fetched CI logs for build output.

Output format is compatible with clang_enriched_to_parquet.py, with:
- doc_type = "diagnostic"
- domain_kind set per CI output type (COMPILER_ERROR=42, TEST_OUTPUT=45, etc.)
- text contains: diff context + CI log (delimited for the tokenizer)

Usage:
    python scripts/join_ci_with_diffs.py \
        --ci-root /path/to/ci_diagnostics \
        --ci-logs /path/to/ci_logs_enriched.jsonl \
        --extract-cache /path/to/extract_cache_case5_shared \
        --output /path/to/ci_paired_enriched.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def classify_ci_output(text: str) -> int:
    """Classify CI log text into a DomainKind value."""
    text_lower = text.lower()
    if "sanitizer" in text_lower or "asan" in text_lower or "ubsan" in text_lower:
        return 48  # SANITIZER_OUTPUT
    if "undefined reference" in text_lower or "ld returned" in text_lower or "linker" in text_lower:
        return 44  # LINKER_ERROR
    if re.search(r"error:", text_lower) and re.search(r"\.(c|cpp|cc|h|hpp):\d+", text_lower):
        return 42  # COMPILER_ERROR
    if "cmake error" in text_lower or "make.*error" in text_lower or "ninja: build stopped" in text_lower:
        return 43  # BUILD_ERROR
    if re.search(r"test.*failed|ctest.*failed|FAILED", text):
        return 45  # TEST_OUTPUT
    if "warning:" in text_lower:
        return 40  # COMPILER_DIAGNOSTIC
    return 41  # BUILD_DIAGNOSTIC (default)


def load_ci_logs(path: str) -> dict[int, dict]:
    """Load fetched CI logs indexed by job_id."""
    logs = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            meta = obj.get("ci_metadata", {})
            job_id = meta.get("job_id")
            if job_id:
                logs[job_id] = obj
    return logs


def load_extract_cache_index(cache_root: str) -> dict[str, str]:
    """Build index: bare_repo_name -> JSONL path."""
    index = {}
    cache_path = Path(cache_root)
    for d in cache_path.iterdir():
        if d.is_dir():
            jsonl = d / f"{d.name}_commits.jsonl"
            if jsonl.exists():
                index[d.name] = str(jsonl)
    return index


def find_commit_diffs(jsonl_path: str, commit_sha: str, max_files: int = 10) -> list[dict]:
    """Find all file diffs for a given commit in extract cache JSONL."""
    diffs = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("commit_hash") == commit_sha:
                    diffs.append(obj)
                    if len(diffs) >= max_files:
                        break
            except json.JSONDecodeError:
                pass
    return diffs


def format_diff_context(diffs: list[dict], max_chars: int = 4000) -> str:
    """Format commit diffs into a readable context block."""
    parts = []
    total = 0
    for d in diffs:
        filepath = d.get("filepath", "?")
        diff_text = d.get("diff", "")
        subject = d.get("subject", "")
        if not diff_text:
            continue
        # Truncate very large diffs
        if len(diff_text) > 2000:
            diff_text = diff_text[:2000] + "\n... (truncated)"
        entry = f"--- {filepath}\n{diff_text}"
        if total + len(entry) > max_chars:
            parts.append("... (more files changed)")
            break
        parts.append(entry)
        total += len(entry)

    header = f"Commit: {diffs[0].get('commit_hash', '?')[:12]}" if diffs else "Commit: ?"
    if diffs and diffs[0].get("subject"):
        header += f" — {diffs[0]['subject']}"
    return f"{header}\n\n" + "\n\n".join(parts)


def format_paired_document(
    ci_record: dict,
    ci_log_text: str,
    diff_context: str,
    domain_kind: int,
) -> dict:
    """Create a paired diff+CI document."""
    repo = ci_record.get("repo", "unknown")
    diagnostics = ci_record.get("diagnostics", [])

    # Build the paired text: diff first, then CI output
    text_parts = [
        f"// Paired CI Report: {repo}",
        f"// Conclusion: {ci_record.get('conclusion', '?')}",
        f"// Platform: {ci_record.get('platform', '?')} | Compiler: {ci_record.get('compiler_info', '?')}",
        "",
        "// === CHANGES ===",
        diff_context,
        "",
        "// === CI OUTPUT ===",
        ci_log_text,
    ]

    # Add structured diagnostics if present
    if diagnostics:
        text_parts.append("")
        text_parts.append("// === DIAGNOSTICS SUMMARY ===")
        for d in diagnostics[:10]:
            sev = d.get("severity", "?")
            msg = d.get("message", "")
            file = d.get("file", "")
            line_no = d.get("line", "")
            text_parts.append(f"// [{sev}] {file}:{line_no} — {msg}")

    full_text = "\n".join(text_parts)

    return {
        "text": full_text,
        "doc_type": "diagnostic",
        "domain_kind": domain_kind,
        "repo": repo,
        "commit_hash": ci_record.get("commit_sha", ""),
        "filepath": diagnostics[0].get("file", "") if diagnostics else "",
        "source_doc_id": f"ci_paired:{ci_record.get('run_id', 0)}:{ci_record.get('job_id', 0)}",
        "ci_metadata": {
            "run_id": ci_record.get("run_id"),
            "job_id": ci_record.get("job_id"),
            "workflow": ci_record.get("workflow"),
            "job_name": ci_record.get("job_name"),
            "conclusion": ci_record.get("conclusion"),
            "platform": ci_record.get("platform"),
            "compiler_info": ci_record.get("compiler_info"),
            "build_command": ci_record.get("build_command"),
            "created_at": ci_record.get("created_at"),
            "diagnostics_count": len(diagnostics),
            "severities": list(set(d.get("severity") for d in diagnostics)),
            "has_diff": True,
            "diff_files": len(diff_context.split("--- ")) - 1,
        },
        "symbol_identities": [],
        "domain_sidecars": {},
    }


def main():
    parser = argparse.ArgumentParser(description="Join commit diffs with CI logs")
    parser.add_argument("--ci-root", required=True, help="Path to ci_diagnostics/ directory")
    parser.add_argument("--ci-logs", required=True, help="Path to fetched ci_logs_enriched.jsonl")
    parser.add_argument("--extract-cache", required=True, help="Path to extract_cache_case5_shared/")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-docs", type=int, default=0, help="Max documents (0=all)")
    parser.add_argument("--min-text-chars", type=int, default=300, help="Min text length")
    args = parser.parse_args()

    # Load CI logs
    print(f"Loading CI logs from {args.ci_logs}...")
    ci_logs = load_ci_logs(args.ci_logs)
    print(f"  {len(ci_logs)} CI logs loaded")

    # Load extract cache index
    print(f"Indexing extract cache at {args.extract_cache}...")
    cache_index = load_extract_cache_index(args.extract_cache)
    print(f"  {len(cache_index)} repos indexed")

    # Load CI diagnostic records
    ci_root = Path(args.ci_root)
    records = []
    for f in sorted(ci_root.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if obj.get("commit_sha") and obj.get("job_id"):
                        records.append(obj)
                except json.JSONDecodeError:
                    pass
    print(f"  {len(records)} CI records with commit_sha")

    # Match: CI record → diff from extract cache → CI log
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    no_log = 0
    no_diff = 0
    too_short = 0

    # Build repo name mapping (owner/repo → bare_name in cache)
    # The cache uses bare names like "mysql", "postgres", etc.
    # CI records use "owner/repo" like "mysql/mysql-server"
    # We need to match by the repo part after /
    repo_to_cache = {}
    for bare_name in cache_index:
        repo_to_cache[bare_name.lower()] = bare_name

    with open(output_path, "w") as out:
        for i, record in enumerate(records):
            if args.max_docs > 0 and written >= args.max_docs:
                break

            job_id = record["job_id"]
            commit_sha = record["commit_sha"]
            repo_full = record.get("repo", "")

            # Get CI log
            ci_log_obj = ci_logs.get(job_id)
            if not ci_log_obj:
                no_log += 1
                continue
            ci_log_text = ci_log_obj.get("text", "")
            # Strip the header we added in fetch_ci_logs (keep only the log portion)
            if "// === CI OUTPUT ===" in ci_log_text:
                ci_log_text = ci_log_text.split("// === CI OUTPUT ===", 1)[1].strip()
            elif "\n\n" in ci_log_text:
                # Skip the header block
                parts = ci_log_text.split("\n\n", 1)
                ci_log_text = parts[1] if len(parts) > 1 else ci_log_text

            # Find diff in extract cache
            # Try to match repo name
            repo_bare = repo_full.split("/")[-1].lower() if "/" in repo_full else repo_full.lower()
            cache_key = repo_to_cache.get(repo_bare)

            diffs = []
            if cache_key:
                diffs = find_commit_diffs(cache_index[cache_key], commit_sha)

            if not diffs:
                no_diff += 1
                # Still write the doc but without diff context
                diff_context = f"(diff not available for {commit_sha[:12]} in {repo_full})"
            else:
                diff_context = format_diff_context(diffs)

            # Classify the CI output
            domain_kind = classify_ci_output(ci_log_text)

            # Format paired document
            doc = format_paired_document(record, ci_log_text, diff_context, domain_kind)

            if len(doc["text"]) < args.min_text_chars:
                too_short += 1
                continue

            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(records)}] written={written} no_log={no_log} no_diff={no_diff}")

    print(f"\nDone: written={written} no_log={no_log} no_diff={no_diff} too_short={too_short}")
    print(f"Output: {output_path}")

    if written > 0:
        total_chars = 0
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line)
                total_chars += len(obj.get("text", ""))
        est_tokens = total_chars // 4
        steps = est_tokens // (192 * 1024)
        print(f"Estimated: ~{est_tokens:,} tokens = ~{steps} steps (bs=192 seq=1024)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
