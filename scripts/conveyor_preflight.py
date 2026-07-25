#!/usr/bin/env python3
"""Pre-flight validation for the streaming conveyor commits stream.

Scans the extract cache and predicts which repos/ranges will fail BEFORE
running the conveyor, so operators know in advance what will break.

Usage:
    python scripts/conveyor_preflight.py [--extract-cache-root PATH] [--conveyor-root PATH]

Exit codes:
    0 = all clear (or only known-fail repos)
    1 = unexpected failures predicted
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

MLX_ROOT = Path(__file__).resolve().parent.parent.parent / "cppmega.mlx"
DEFAULT_EXTRACT_CACHE = MLX_ROOT / "outputs" / "extract_cache_case5_shared"
DEFAULT_CONVEYOR_ROOT = MLX_ROOT / "outputs" / "conveyor_case5_v11_commits_20260719_180000"


def check_bad_units(repo_dir: Path) -> list[str]:
    """Check if extract checkpoint has bad_units that will trigger fail policy."""
    issues = []
    checkpoint_dirs = list(repo_dir.glob("*.extract-checkpoint"))
    for cp in checkpoint_dirs:
        bad_units_file = cp / "bad_units.jsonl"
        if not bad_units_file.exists():
            continue
        for subdir in cp.iterdir():
            if not subdir.is_dir():
                continue
            bu = subdir / "bad_units.jsonl"
            if bu.exists() and bu.stat().st_size > 0:
                count = sum(1 for _ in open(bu))
                issues.append(
                    f"bad_units={count} in {bu.relative_to(repo_dir)}"
                )
        if bad_units_file.exists() and bad_units_file.stat().st_size > 0:
            count = sum(1 for _ in open(bad_units_file))
            issues.append(f"bad_units={count} in {bad_units_file.name}")
    # Also check for failures manifest
    for failures in repo_dir.glob("*.failures.json"):
        try:
            data = json.loads(failures.read_text())
            if data:
                issues.append(f"failure manifest: {failures.name} ({len(data)} entries)")
        except (json.JSONDecodeError, OSError):
            pass
    return issues


def validate_jsonl(jsonl_path: Path, max_lines: int = 0) -> dict:
    """Validate JSON parsing of a commit JSONL file.

    Returns stats about parse health.
    """
    stats = {
        "total_lines": 0,
        "json_errors": 0,
        "empty_diff": 0,
        "missing_fields": 0,
        "valid": 0,
    }
    required_fields = {"commit_hash", "filepath"}
    with open(jsonl_path, "r", errors="replace") as f:
        for line in f:
            stats["total_lines"] += 1
            if max_lines and stats["total_lines"] > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue
            missing = required_fields - set(record.keys())
            if missing:
                stats["missing_fields"] += 1
                continue
            diff = record.get("diff") or ""
            if not diff.strip():
                stats["empty_diff"] += 1
                continue
            stats["valid"] += 1
    return stats


def check_conveyor_manifest(conveyor_root: Path) -> set[str]:
    """Return set of repo names whose commits stream is complete."""
    manifest_path = conveyor_root / "_done.json"
    if not manifest_path.exists():
        return set()
    try:
        data = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    done = data.get("done", {})
    return {
        key.removesuffix("::commits")
        for key, val in done.items()
        if key.endswith("::commits") and isinstance(val, dict) and val.get("complete")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Conveyor pre-flight validation")
    parser.add_argument(
        "--extract-cache-root",
        type=Path,
        default=DEFAULT_EXTRACT_CACHE,
    )
    parser.add_argument(
        "--conveyor-root",
        type=Path,
        default=DEFAULT_CONVEYOR_ROOT,
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Max lines to validate per JSONL (0=all)",
    )
    args = parser.parse_args()

    extract_cache = args.extract_cache_root
    conveyor_root = args.conveyor_root

    if not extract_cache.is_dir():
        print(f"ERROR: extract cache not found: {extract_cache}", file=sys.stderr)
        return 1

    done_repos = check_conveyor_manifest(conveyor_root)
    print(f"Conveyor manifest: {len(done_repos)} repos already done")
    print(f"Extract cache: {extract_cache}")
    print()

    will_fail = []
    will_warn = []
    healthy = []
    skipped_done = []

    repo_dirs = sorted(d for d in extract_cache.iterdir() if d.is_dir())
    for repo_dir in repo_dirs:
        repo_name = repo_dir.name
        commit_jsonls = list(repo_dir.glob("*_commits.jsonl"))
        failure_manifests = list(repo_dir.glob("*.failures.json"))

        if repo_name in done_repos:
            skipped_done.append(repo_name)
            continue

        issues = []

        # Check 0: extract-stage failure (has .failures.json, no JSONL produced)
        if not commit_jsonls and failure_manifests:
            for fm in failure_manifests:
                issues.append(f"extract failed: {fm.name}")
            will_fail.append((repo_name, issues))
            continue

        if not commit_jsonls:
            continue

        # Check 1: bad_units in extract checkpoint
        bu_issues = check_bad_units(repo_dir)
        if bu_issues:
            issues.extend(bu_issues)

        # Check 2: validate JSONL parsing
        for jsonl in commit_jsonls:
            stats = validate_jsonl(jsonl, max_lines=args.max_lines)
            if stats["json_errors"] > 0:
                issues.append(
                    f"{jsonl.name}: {stats['json_errors']} JSON parse errors "
                    f"/ {stats['total_lines']} lines"
                )
            if stats["missing_fields"] > 0:
                issues.append(
                    f"{jsonl.name}: {stats['missing_fields']} records missing "
                    f"required fields (commit_hash, filepath)"
                )
            if stats["total_lines"] > 0:
                empty_ratio = stats["empty_diff"] / stats["total_lines"]
                if empty_ratio > 0.5:
                    will_warn.append(
                        f"{repo_name}: {empty_ratio:.0%} empty diffs "
                        f"({stats['empty_diff']}/{stats['total_lines']})"
                    )

        if issues:
            will_fail.append((repo_name, issues))
        else:
            healthy.append(repo_name)

    # Report
    print("=" * 70)
    print("PRE-FLIGHT RESULTS")
    print("=" * 70)

    if skipped_done:
        print(f"\n  DONE (skipped): {len(skipped_done)} repos")
        for r in skipped_done:
            print(f"    [done] {r}")

    if healthy:
        print(f"\n  HEALTHY: {len(healthy)} repos will process normally")
        for r in healthy:
            print(f"    [ok] {r}")

    if will_warn:
        print(f"\n  WARNINGS: {len(will_warn)}")
        for w in will_warn:
            print(f"    [warn] {w}")

    if will_fail:
        print(f"\n  WILL FAIL: {len(will_fail)} repos (fail-loud policy)")
        for repo_name, issues in will_fail:
            print(f"    [FAIL] {repo_name}:")
            for issue in issues:
                print(f"           - {issue}")

    print()
    print("-" * 70)
    total_pending = len(healthy) + len(will_fail)
    print(
        f"Summary: {total_pending} pending | "
        f"{len(healthy)} healthy | "
        f"{len(will_fail)} will fail | "
        f"{len(skipped_done)} already done"
    )

    if will_fail:
        print(
            f"\nThe {len(will_fail)} failing repo(s) will trigger fail-loud "
            f"and be skipped by the conveyor. This is expected behavior."
        )

    print(
        "\nNOTE: process_record (clang indexing) parse errors cannot be "
        "predicted by this pre-flight.\nRepos with valid JSONL may still "
        "fail at runtime if clang cannot parse individual commit diffs.\n"
        "These failures surface as 'refusing partial commit range: N record "
        "parse error(s)' in the conveyor log."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
