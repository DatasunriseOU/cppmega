#!/usr/bin/env python3
"""Fetch real CI build logs from GitHub Actions and produce enriched JSONL
for tokenization as doc_type=diagnostic.

Reads ci_diagnostics/*.jsonl (structured metadata with run_id/job_id),
downloads actual job logs via `gh api`, filters to relevant build output
(compiler errors, warnings, linker errors, test failures), and writes
enriched documents compatible with clang_enriched_to_parquet.py.

Usage:
    python scripts/fetch_ci_logs.py \
        --ci-root /path/to/ci_diagnostics \
        --output /path/to/ci_enriched.jsonl \
        --max-jobs 100 \
        --max-log-lines 500

Rate limit: ~5000 req/hr on GitHub API. 1855 jobs fits in one batch.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Patterns that indicate relevant build output
_ERROR_PATTERNS = re.compile(
    r"(error:|warning:|fatal error|undefined reference|ld returned|"
    r"FAILED|BUILD FAILED|CMake Error|compilation terminated|"
    r"linker command failed|cannot find|No such file|"
    r"segfault|SIGSEGV|SIGABRT|Assertion.*failed|"
    r"Test.*FAILED|ctest.*Failed|make.*Error|"
    r"ninja: build stopped|cl : Command line error|"
    r"MSVC.*error|LINK : fatal)",
    re.IGNORECASE,
)

# Patterns to SKIP (boilerplate)
_SKIP_PATTERNS = re.compile(
    r"(Pulling fs layer|Waiting$|Already exists|"
    r"##\[group\]Runner Image|##\[group\]Operating System|"
    r"##\[group\]GITHUB_TOKEN|Download action repository|"
    r"Pulling from|Digest:|Status: Downloaded|"
    r"Successfully tagged|Creating network|"
    r"##\[command\]/usr/bin/docker (pull|create|start))",
    re.IGNORECASE,
)

# Timestamp prefix in GH Actions logs
_TS_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ")


def strip_timestamp(line: str) -> str:
    return _TS_PREFIX.sub("", line)


def is_relevant_line(line: str) -> bool:
    if _SKIP_PATTERNS.search(line):
        return False
    return True


def extract_relevant_portion(log_text: str, max_lines: int = 500) -> str:
    """Extract the relevant build portion from a full job log.

    Strategy: find the build/compile/test section, skip setup/teardown.
    Keep lines around errors/warnings with context.
    """
    lines = log_text.splitlines()
    total = len(lines)

    # Find where the actual build starts (after checkout/setup)
    build_start = 0
    for i, line in enumerate(lines):
        stripped = strip_timestamp(line)
        if any(
            marker in stripped
            for marker in [
                "##[group]Run ",
                "cmake",
                "make",
                "ninja",
                "msbuild",
                "gcc",
                "g++",
                "clang",
                "cl.exe",
                "cargo build",
                "configure",
                "meson",
                "bazel",
            ]
        ):
            build_start = max(0, i - 2)
            break

    # Collect relevant lines with context around errors
    relevant = []
    error_context = 3  # lines before/after an error

    # First pass: mark lines near errors
    error_indices = set()
    for i in range(build_start, total):
        stripped = strip_timestamp(lines[i])
        if _ERROR_PATTERNS.search(stripped):
            for j in range(max(build_start, i - error_context), min(total, i + error_context + 1)):
                error_indices.add(j)

    # Second pass: collect build output section
    in_build = False
    collected = 0
    for i in range(build_start, total):
        stripped = strip_timestamp(lines[i])

        # Detect build section boundaries
        if "##[group]Run " in stripped or "##[command]" in stripped:
            in_build = True
        if "##[endgroup]" in stripped and in_build:
            in_build = False
            if collected > 0:
                relevant.append("")  # separator

        # Always include lines near errors
        if i in error_indices:
            if is_relevant_line(stripped):
                relevant.append(stripped)
                collected += 1
        # Include build command output
        elif in_build and is_relevant_line(stripped):
            # Skip pure progress lines (docker layers, download progress)
            if stripped.strip() and not re.match(r"^[\s\d.]+%?\s*$", stripped):
                relevant.append(stripped)
                collected += 1

        if collected >= max_lines:
            break

    # If we got nothing useful, fall back to last N lines (usually has the error)
    if not relevant:
        tail_start = max(build_start, total - max_lines)
        for i in range(tail_start, total):
            stripped = strip_timestamp(lines[i])
            if is_relevant_line(stripped) and stripped.strip():
                relevant.append(stripped)

    return "\n".join(relevant[:max_lines])


def fetch_job_log(owner: str, repo: str, job_id: int) -> str | None:
    """Download job log via gh api."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{owner}/{repo}/actions/jobs/{job_id}/logs"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout
        # 410 = log expired (GitHub keeps logs ~90 days)
        if "410" in result.stderr or "expired" in result.stderr.lower():
            return None
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def format_ci_document(record: dict, log_text: str) -> dict:
    """Format a CI diagnostic record + log into an enriched document."""
    repo = record.get("repo", "unknown")
    diagnostics = record.get("diagnostics", [])

    # Build a structured header
    header_parts = [
        f"// CI Build Log: {repo}",
        f"// Job: {record.get('job_name', 'unknown')}",
        f"// Platform: {record.get('platform', 'unknown')}",
        f"// Compiler: {record.get('compiler_info', 'unknown')}",
        f"// Conclusion: {record.get('conclusion', 'unknown')}",
    ]
    if record.get("build_command"):
        header_parts.append(f"// Build: {record['build_command']}")
    if record.get("commit_sha"):
        header_parts.append(f"// Commit: {record['commit_sha'][:12]}")

    # Structured diagnostics summary
    if diagnostics:
        header_parts.append("// Diagnostics:")
        for d in diagnostics[:20]:  # cap at 20
            sev = d.get("severity", "?")
            msg = d.get("message", "")
            file = d.get("file", "")
            line_no = d.get("line", "")
            header_parts.append(f"//   [{sev}] {file}:{line_no} {msg}")

    header = "\n".join(header_parts)
    full_text = f"{header}\n\n{log_text}"

    return {
        "text": full_text,
        "doc_type": "diagnostic",
        "repo": repo,
        "commit_hash": record.get("commit_sha", ""),
        "filepath": diagnostics[0].get("file", "") if diagnostics else "",
        "source_doc_id": f"ci:{record.get('run_id', 0)}:{record.get('job_id', 0)}",
        "ci_metadata": {
            "run_id": record.get("run_id"),
            "job_id": record.get("job_id"),
            "workflow": record.get("workflow"),
            "job_name": record.get("job_name"),
            "conclusion": record.get("conclusion"),
            "platform": record.get("platform"),
            "compiler_info": record.get("compiler_info"),
            "build_command": record.get("build_command"),
            "created_at": record.get("created_at"),
            "diagnostics_count": len(diagnostics),
            "severities": list(set(d.get("severity") for d in diagnostics)),
        },
        "symbol_identities": [],
        "domain_kind": 0,
        "domain_sidecars": {},
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch CI logs and produce enriched JSONL")
    parser.add_argument("--ci-root", required=True, help="Path to ci_diagnostics/ directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-jobs", type=int, default=0, help="Max jobs to fetch (0=all)")
    parser.add_argument("--max-log-lines", type=int, default=500, help="Max relevant lines per log")
    parser.add_argument("--min-text-chars", type=int, default=200, help="Skip docs shorter than this")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (sec)")
    parser.add_argument("--dry-run", action="store_true", help="Count records without fetching")
    args = parser.parse_args()

    ci_root = Path(args.ci_root)
    jsonl_files = sorted(ci_root.glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: no .jsonl files in {ci_root}", file=sys.stderr)
        return 1

    # Load all records
    records = []
    for f in jsonl_files:
        with open(f) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if obj.get("job_id") and obj.get("repo"):
                        records.append(obj)
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(records)} CI records from {len(jsonl_files)} files")

    if args.dry_run:
        repos = set(r["repo"] for r in records)
        print(f"Unique repos: {len(repos)}")
        print(f"Unique job_ids: {len(set(r['job_id'] for r in records))}")
        return 0

    # Deduplicate by job_id (some records may repeat)
    seen_jobs = set()
    unique_records = []
    for r in records:
        jid = r["job_id"]
        if jid not in seen_jobs:
            seen_jobs.add(jid)
            unique_records.append(r)

    if args.max_jobs > 0:
        unique_records = unique_records[: args.max_jobs]

    print(f"Fetching logs for {len(unique_records)} unique jobs...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fetched = 0
    written = 0
    expired = 0
    too_short = 0
    failed = 0

    with open(output_path, "w") as out:
        for i, record in enumerate(unique_records):
            repo_full = record["repo"]
            if "/" not in repo_full:
                failed += 1
                continue
            owner, repo = repo_full.split("/", 1)
            job_id = record["job_id"]

            log_text = fetch_job_log(owner, repo, job_id)
            if log_text is None:
                expired += 1
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(unique_records)}] fetched={fetched} expired={expired} written={written}")
                continue

            fetched += 1
            relevant = extract_relevant_portion(log_text, max_lines=args.max_log_lines)

            if len(relevant) < args.min_text_chars:
                too_short += 1
                continue

            doc = format_ci_document(record, relevant)
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(unique_records)}] fetched={fetched} expired={expired} written={written}")

            time.sleep(args.delay)

    print(f"\nDone: fetched={fetched} expired={expired} too_short={too_short} failed={failed} written={written}")
    print(f"Output: {output_path}")

    if written > 0:
        # Quick token estimate
        total_chars = 0
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line)
                total_chars += len(obj.get("text", ""))
        est_tokens = total_chars // 4  # rough estimate
        steps = est_tokens // (192 * 1024)
        print(f"Estimated: ~{est_tokens:,} tokens = ~{steps} steps (bs=192 seq=1024)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
