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
from dataclasses import dataclass
import hashlib
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


@dataclass(frozen=True)
class JobLogFetch:
    status: str
    text: str = ""
    detail: str = ""


def _classify_job_log_process_result(
    owner: str,
    repo: str,
    job_id: int,
    result: subprocess.CompletedProcess[str],
) -> JobLogFetch:
    if result.returncode == 0:
        return JobLogFetch(status="fetched", text=result.stdout)
    stderr = (result.stderr or "").strip()
    if re.search(r"\bHTTP\s+410\b", stderr, re.IGNORECASE):
        return JobLogFetch(status="expired", detail=stderr)
    raise RuntimeError(
        f"{owner}/{repo} job {job_id}: GitHub log request failed "
        f"(exit={result.returncode}): {stderr[:500]}"
    )


def fetch_job_log(owner: str, repo: str, job_id: int) -> JobLogFetch:
    """Download one job log, distinguishing true expiry from fetch failure."""

    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{owner}/{repo}/actions/jobs/{job_id}/logs"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("GitHub CLI `gh` is required to fetch CI logs") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{owner}/{repo} job {job_id}: GitHub log request timed out"
        ) from exc
    return _classify_job_log_process_result(
        owner,
        repo,
        job_id,
        result,
    )


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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl_fsync(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_ci_source_records(
    ci_root: Path,
) -> tuple[list[dict], list[dict], int, list[dict]]:
    paths = sorted(ci_root.glob("*.jsonl"))
    if not paths:
        raise RuntimeError(f"no .jsonl files in {ci_root}")

    records: list[dict] = []
    non_job_records = 0
    inventory: list[dict] = []
    for path in paths:
        inventory.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"{path}:{line_number}: CI source row must be an object"
                    )
                if not record.get("job_id") or not record.get("repo"):
                    non_job_records += 1
                    continue
                if (
                    not isinstance(record["job_id"], int)
                    or isinstance(record["job_id"], bool)
                    or record["job_id"] < 1
                    or not isinstance(record["repo"], str)
                    or "/" not in record["repo"]
                ):
                    raise RuntimeError(
                        f"{path}:{line_number}: invalid repo/job identity"
                    )
                records.append(record)

    unique: dict[int, dict] = {}
    aliases: list[dict] = []
    for record in records:
        job_id = int(record["job_id"])
        previous = unique.get(job_id)
        if previous is None:
            unique[job_id] = record
            continue
        previous_without_repo = {**previous, "repo": ""}
        current_without_repo = {**record, "repo": ""}
        if previous_without_repo != current_without_repo:
            raise RuntimeError(
                f"job_id {job_id} maps to conflicting CI source records"
            )
        aliases.append(
            {
                "job_id": job_id,
                "canonical_repo": previous["repo"],
                "alias_repo": record["repo"],
            }
        )
    return list(unique.values()), inventory, non_job_records, aliases


def _load_existing_output(path: Path) -> dict[int, dict]:
    output: dict[int, dict] = {}
    if not path.exists():
        return output
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON output row: {exc}"
                ) from exc
            metadata = document.get("ci_metadata") if isinstance(document, dict) else None
            job_id = metadata.get("job_id") if isinstance(metadata, dict) else None
            if (
                not isinstance(job_id, int)
                or isinstance(job_id, bool)
                or job_id < 1
            ):
                raise RuntimeError(
                    f"{path}:{line_number}: output row lacks an integer CI job_id"
                )
            if job_id in output:
                raise RuntimeError(f"{path}: duplicate output job_id {job_id}")
            output[job_id] = document
    return output


def _load_fetch_state(path: Path) -> dict[int, dict]:
    states: dict[int, dict] = {}
    if not path.exists():
        return states
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                state = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid fetch state: {exc}"
                ) from exc
            job_id = state.get("job_id") if isinstance(state, dict) else None
            if (
                not isinstance(job_id, int)
                or isinstance(job_id, bool)
                or job_id < 1
                or state.get("status") not in {"fetched", "expired", "too_short"}
            ):
                raise RuntimeError(f"{path}:{line_number}: malformed fetch state")
            if job_id in states:
                raise RuntimeError(f"{path}: duplicate fetch state for job {job_id}")
            states[job_id] = state
    return states


def _completion_receipt(
    *,
    source_inventory: list[dict],
    records: list[dict],
    source_row_count: int,
    non_job_records: int,
    aliases: list[dict],
    output_path: Path,
    output: dict[int, dict],
    state_path: Path,
    states: dict[int, dict],
    errors: list[str],
    max_jobs: int,
) -> dict:
    expected_jobs = {int(record["job_id"]) for record in records}
    accounted_jobs = set(states)
    unresolved = sorted(expected_jobs - accounted_jobs)
    fetched = sorted(
        job_id for job_id, state in states.items() if state["status"] == "fetched"
    )
    expired = sorted(
        job_id for job_id, state in states.items() if state["status"] == "expired"
    )
    too_short = sorted(
        job_id for job_id, state in states.items() if state["status"] == "too_short"
    )
    status = (
        "complete"
        if not unresolved
        and not errors
        and not max_jobs
        and set(output) == set(fetched)
        else "incomplete"
    )
    return {
        "schema": "cppmega_ci_log_extraction_v1",
        "status": status,
        "source_inventory": source_inventory,
        "source_inventory_sha256": _canonical_sha256(source_inventory),
        "source_row_count": source_row_count,
        "non_job_source_row_count": non_job_records,
        "unique_job_count": len(expected_jobs),
        "duplicate_alias_count": len(aliases),
        "duplicate_aliases": aliases,
        "job_set_sha256": _canonical_sha256(sorted(expected_jobs)),
        "fetched_count": len(fetched),
        "expired_count": len(expired),
        "too_short_count": len(too_short),
        "unresolved_count": len(unresolved),
        "expired_jobs": [
            {
                "job_id": job_id,
                "repo": states[job_id]["repo"],
                "detail": states[job_id].get("detail", ""),
            }
            for job_id in expired
        ],
        "unresolved_jobs": unresolved,
        "errors": errors,
        "scope_limit": max_jobs or None,
        "output": {
            "path": str(output_path.resolve()),
            "row_count": len(output),
            "size": output_path.stat().st_size if output_path.exists() else 0,
            "sha256": _sha256_file(output_path) if output_path.exists() else None,
            "job_set_sha256": _canonical_sha256(sorted(output)),
        },
        "state": {
            "path": str(state_path.resolve()),
            "row_count": len(states),
            "size": state_path.stat().st_size if state_path.exists() else 0,
            "sha256": _sha256_file(state_path) if state_path.exists() else None,
        },
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and continue an existing output/state pair.",
    )
    parser.add_argument(
        "--state",
        help="Durable per-job outcome JSONL (default: OUTPUT.fetch-state.jsonl).",
    )
    parser.add_argument(
        "--completion-receipt",
        help="Atomic extraction receipt (default: OUTPUT.completion.json).",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    ci_root = Path(args.ci_root)
    records, source_inventory, non_job_records, aliases = _load_ci_source_records(
        ci_root
    )
    source_row_count = len(records) + len(aliases) + non_job_records
    print(
        f"Loaded {len(records)} unique CI jobs from {len(source_inventory)} files "
        f"({source_row_count} total rows, {non_job_records} non-job records, "
        f"{len(aliases)} duplicate aliases)"
    )

    if args.dry_run:
        repos = set(r["repo"] for r in records)
        print(f"Unique repos: {len(repos)}")
        print(f"Unique job_ids: {len(set(r['job_id'] for r in records))}")
        return 0

    unique_records = records
    if args.max_jobs > 0:
        unique_records = unique_records[: args.max_jobs]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state or f"{output_path}.fetch-state.jsonl")
    receipt_path = Path(
        args.completion_receipt or f"{output_path}.completion.json"
    )
    if (output_path.exists() or state_path.exists()) and not args.resume:
        raise RuntimeError(
            "output/state already exists; pass --resume to validate and continue "
            "instead of silently overwriting CI evidence"
        )
    output_path.touch(exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.touch(exist_ok=True)

    source_by_job = {int(record["job_id"]): record for record in records}
    output = _load_existing_output(output_path)
    states = _load_fetch_state(state_path)
    unexpected_output = sorted(set(output) - set(source_by_job))
    unexpected_states = sorted(set(states) - set(source_by_job))
    if unexpected_output or unexpected_states:
        raise RuntimeError(
            f"resume evidence contains jobs outside the source corpus: "
            f"output={unexpected_output[:10]} state={unexpected_states[:10]}"
        )

    for job_id, state in states.items():
        source_sha256 = _canonical_sha256(source_by_job[job_id])
        if state.get("source_sha256") != source_sha256:
            raise RuntimeError(f"source record drifted for resumed job {job_id}")
        if state["status"] == "fetched":
            document = output.get(job_id)
            if document is None:
                raise RuntimeError(
                    f"fetch state says job {job_id} was fetched but output is missing"
                )
            if state.get("document_sha256") != _canonical_sha256(document):
                raise RuntimeError(
                    f"output document drifted for resumed job {job_id}"
                )
        elif job_id in output:
            raise RuntimeError(
                f"job {job_id} has both {state['status']} state and an output row"
            )

    # An interrupted process can fsync the output row just before its matching
    # state row. Recover that one-way window by deriving and appending the state.
    for job_id in sorted(set(output) - set(states)):
        record = source_by_job[job_id]
        document = output[job_id]
        if document.get("repo") not in {
            record["repo"],
            *(
                item["alias_repo"]
                for item in aliases
                if item["job_id"] == job_id
            ),
        }:
            raise RuntimeError(f"output repo drifted for job {job_id}")
        state = {
            "job_id": job_id,
            "repo": document["repo"],
            "status": "fetched",
            "source_sha256": _canonical_sha256(record),
            "document_sha256": _canonical_sha256(document),
        }
        _append_jsonl_fsync(state_path, state)
        states[job_id] = state

    pending = [
        record
        for record in unique_records
        if int(record["job_id"]) not in states
    ]
    print(
        f"Fetching {len(pending)} pending of {len(unique_records)} selected jobs "
        f"(resumed={len(states)})..."
    )
    errors: list[str] = []
    for index, record in enumerate(pending, start=1):
        repo_full = record["repo"]
        owner, repo = repo_full.split("/", 1)
        job_id = int(record["job_id"])
        fetch: JobLogFetch | None = None
        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                fetch = fetch_job_log(owner, repo, job_id)
                break
            except RuntimeError as exc:
                if attempt >= max(1, args.max_retries):
                    errors.append(str(exc))
                    break
                time.sleep(min(8.0, float(2 ** (attempt - 1))))
        if fetch is None:
            break

        state = {
            "job_id": job_id,
            "repo": repo_full,
            "status": fetch.status,
            "source_sha256": _canonical_sha256(record),
        }
        if fetch.status == "fetched":
            relevant = extract_relevant_portion(
                fetch.text,
                max_lines=args.max_log_lines,
            )
            if len(relevant) < args.min_text_chars:
                state["status"] = "too_short"
                state["relevant_text_chars"] = len(relevant)
            else:
                document = format_ci_document(record, relevant)
                state["document_sha256"] = _canonical_sha256(document)
                _append_jsonl_fsync(output_path, document)
                output[job_id] = document
                time.sleep(args.delay)
        elif fetch.status == "expired":
            state["http_status"] = 410
            state["detail"] = fetch.detail
        else:
            raise RuntimeError(
                f"unsupported fetch outcome for job {job_id}: {fetch.status!r}"
            )
        _append_jsonl_fsync(state_path, state)
        states[job_id] = state

        if index % 50 == 0 or index == len(pending):
            counts = {
                status: sum(
                    1 for item in states.values() if item["status"] == status
                )
                for status in ("fetched", "expired", "too_short")
            }
            print(
                f"  [{index}/{len(pending)}] fetched={counts['fetched']} "
                f"expired={counts['expired']} too_short={counts['too_short']}"
            )

    receipt = _completion_receipt(
        source_inventory=source_inventory,
        records=records,
        source_row_count=source_row_count,
        non_job_records=non_job_records,
        aliases=aliases,
        output_path=output_path,
        output=output,
        state_path=state_path,
        states=states,
        errors=errors,
        max_jobs=args.max_jobs,
    )
    _write_json_atomic(receipt_path, receipt)

    print(
        f"\nDone: status={receipt['status']} fetched={receipt['fetched_count']} "
        f"expired={receipt['expired_count']} "
        f"too_short={receipt['too_short_count']} "
        f"unresolved={receipt['unresolved_count']}"
    )
    print(f"Output: {output_path}")
    print(f"Completion: {receipt_path}")

    if output:
        # Quick token estimate
        total_chars = 0
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line)
                total_chars += len(obj.get("text", ""))
        est_tokens = total_chars // 4  # rough estimate
        steps = est_tokens // (192 * 1024)
        print(f"Estimated: ~{est_tokens:,} tokens = ~{steps} steps (bs=192 seq=1024)")

    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
