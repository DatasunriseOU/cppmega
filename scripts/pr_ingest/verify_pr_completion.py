#!/usr/bin/env python3
"""Build a fail-closed completion receipt for the canonical PR corpus.

The training conveyor must not infer PR completeness from the mere existence of
``prs.sqlite``.  This verifier binds the exact repo list, GraphQL resume
manifest, SQLite snapshot, and (when needed) the per-PR gap-fill receipt.
Every expected GitHub repository must be terminal ``done`` and its exact stored
PR count must equal GraphQL ``totalCount``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Iterable

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.pr_ingest import pr_store
from scripts.pr_ingest.graphql_pr_stream import (
    GRAPHQL_MANIFEST_SCHEMA,
    GRAPHQL_QUERY_CONTRACT_SHA256,
    load_repo_list,
)


PR_COMPLETION_SCHEMA = "cppmega_pr_completion_v2"
PR_GAP_COMPLETION_SCHEMA = "cppmega_pr_gap_completion_v1"


class PRCompletionError(RuntimeError):
    """The current PR inputs do not prove a complete immutable snapshot."""


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise PRCompletionError(f"required PR input is missing: {path}")
    digest = hashlib.sha256()
    try:
        before = path.stat()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise PRCompletionError(f"cannot hash required PR input {path}: {exc}") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise PRCompletionError(f"required PR input changed while hashing: {path}")
    return digest.hexdigest()


def _require_unchanged_file(path: Path, expected_sha256: str, *, what: str) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PRCompletionError(
            f"{what} changed while building the completion receipt: {path}"
        )


def _require_checkpointed_store(path: Path) -> None:
    wal_path = Path(f"{path}-wal")
    try:
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
    except OSError as exc:
        raise PRCompletionError(f"cannot inspect PR store WAL {wal_path}: {exc}") from exc
    if wal_size:
        raise PRCompletionError(
            f"PR store has an uncheckpointed WAL while building receipt: {wal_path}"
        )


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_target_set_sha256(
    targets: Iterable[tuple[str, int]],
) -> str:
    normalized = sorted({(str(repo), int(number)) for repo, number in targets})
    return _canonical_json_sha256(
        [{"repo": repo, "pr_number": number} for repo, number in normalized]
    )


def _load_json_object(path: Path, *, what: str) -> dict:
    if not path.is_file():
        raise PRCompletionError(f"{what} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PRCompletionError(f"{what} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PRCompletionError(f"{what} must be a JSON object: {path}")
    return payload


def _load_targets(path: Path | None) -> tuple[tuple[str, int], ...]:
    if path is None:
        return ()
    if not path.is_file():
        raise PRCompletionError(f"truncated target list is missing: {path}")
    targets: set[tuple[str, int]] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PRCompletionError(
                f"{path}:{line_number}: invalid JSON target: {exc}"
            ) from exc
        if not isinstance(item, dict):
            raise PRCompletionError(
                f"{path}:{line_number}: target must be an object"
            )
        repo = item.get("repo")
        number = item.get("pr_number")
        if not isinstance(repo, str) or "/" not in repo:
            raise PRCompletionError(
                f"{path}:{line_number}: invalid repo: {repo!r}"
            )
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise PRCompletionError(
                f"{path}:{line_number}: invalid pr_number: {number!r}"
            )
        targets.add((repo, number))
    return tuple(sorted(targets))


def _verify_gap_completion(
    *,
    targets: tuple[tuple[str, int], ...],
    gap_completion_path: Path | None,
) -> tuple[dict | None, dict[tuple[str, int], str]]:
    if not targets:
        if gap_completion_path is not None:
            raise PRCompletionError(
                "gap completion receipt was supplied but truncated target set is empty"
            )
        return None, {}
    if gap_completion_path is None:
        raise PRCompletionError(
            "truncated PR targets exist; an exact gap completion receipt is required"
        )
    receipt = _load_json_object(
        gap_completion_path,
        what="PR gap completion receipt",
    )
    if receipt.get("schema") != PR_GAP_COMPLETION_SCHEMA:
        raise PRCompletionError(
            f"unsupported PR gap completion schema: {receipt.get('schema')!r}"
        )
    if receipt.get("status") != "verified":
        raise PRCompletionError(
            f"PR gap completion status is not verified: {receipt.get('status')!r}"
        )
    expected_hash = canonical_target_set_sha256(targets)
    if receipt.get("targets_sha256") != expected_hash:
        raise PRCompletionError("PR gap completion target-set hash mismatch")
    completed_raw = receipt.get("completed")
    if not isinstance(completed_raw, list):
        raise PRCompletionError("PR gap completion receipt lacks completed targets")
    completed: set[tuple[str, int]] = set()
    record_sha256s: dict[tuple[str, int], str] = {}
    for index, item in enumerate(completed_raw):
        if not isinstance(item, dict):
            raise PRCompletionError(
                f"PR gap completion completed[{index}] must be an object"
            )
        repo = item.get("repo")
        number = item.get("pr_number")
        if not isinstance(repo, str) or not isinstance(number, int):
            raise PRCompletionError(
                f"invalid PR gap completion target at completed[{index}]"
            )
        target = (repo, number)
        record_sha256 = item.get("record_sha256")
        if (
            not isinstance(record_sha256, str)
            or len(record_sha256) != 64
            or any(char not in "0123456789abcdef" for char in record_sha256)
        ):
            raise PRCompletionError(
                f"invalid record_sha256 at PR gap completed[{index}]"
            )
        if target in completed:
            raise PRCompletionError(
                f"duplicate PR gap completion target at completed[{index}]"
            )
        completed.add(target)
        record_sha256s[target] = record_sha256
    expected = set(targets)
    if completed != expected:
        missing = sorted(expected - completed)[:10]
        extra = sorted(completed - expected)[:10]
        raise PRCompletionError(
            f"PR gap completion set mismatch: missing={missing} extra={extra}"
        )
    if receipt.get("target_count") != len(expected):
        raise PRCompletionError("PR gap completion target_count mismatch")
    if receipt.get("completed_count") != len(expected):
        raise PRCompletionError("PR gap completion completed_count mismatch")
    if receipt.get("miss_count") != 0:
        raise PRCompletionError("PR gap completion contains unresolved misses")
    return (
        {
            "path": str(gap_completion_path.resolve()),
            "sha256": sha256_file(gap_completion_path),
            "targets_sha256": expected_hash,
            "target_count": len(expected),
        },
        record_sha256s,
    )


def _checkpoint_sqlite(path: Path) -> None:
    if not path.is_file():
        raise PRCompletionError(f"PR store is missing: {path}")
    try:
        conn = sqlite3.connect(str(path), timeout=60.0)
        try:
            result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result is None or int(result[0]) != 0:
                raise PRCompletionError(
                    f"PR store WAL checkpoint did not complete: {path}: {result}"
                )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise PRCompletionError(
            f"cannot checkpoint PR store before hashing: {path}: {exc}"
        ) from exc


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def verify_pr_completion(
    *,
    repo_list_path: Path,
    graphql_manifest_path: Path,
    store_path: Path,
    truncated_targets_path: Path | None = None,
    gap_completion_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    repo_list_sha256 = sha256_file(repo_list_path)
    graphql_manifest_sha256 = sha256_file(graphql_manifest_path)
    expected_repos = tuple(load_repo_list(str(repo_list_path)))
    manifest = _load_json_object(
        graphql_manifest_path,
        what="GraphQL PR stream manifest",
    )
    if manifest.get("schema") != GRAPHQL_MANIFEST_SCHEMA:
        raise PRCompletionError(
            "unsupported GraphQL PR stream manifest schema: "
            f"{manifest.get('schema')!r}"
        )
    if manifest.get("query_contract_sha256") != GRAPHQL_QUERY_CONTRACT_SHA256:
        raise PRCompletionError(
            "GraphQL PR stream manifest query contract does not match the "
            "complete nested-connection contract"
        )
    scan_id = manifest.get("scan_id")
    if (
        not isinstance(scan_id, str)
        or len(scan_id) != 64
        or any(char not in "0123456789abcdef" for char in scan_id)
    ):
        raise PRCompletionError(
            "GraphQL PR stream manifest lacks a valid exact scan_id"
        )
    manifest_repos = manifest.get("repos")
    if not isinstance(manifest_repos, dict):
        raise PRCompletionError("GraphQL PR stream manifest lacks a repos object")

    unexpected = sorted(set(manifest_repos) - set(expected_repos))
    if unexpected:
        raise PRCompletionError(
            f"GraphQL manifest contains repos outside the bound repo list: {unexpected[:10]}"
        )

    nonterminal: list[tuple[str, object]] = []
    declared_counts: dict[str, int] = {}
    declared_truncated = 0
    for repo in expected_repos:
        record = manifest_repos.get(repo)
        if not isinstance(record, dict) or record.get("status") != "done":
            status = record.get("status") if isinstance(record, dict) else None
            nonterminal.append((repo, status))
            continue
        if record.get("cursor") is not None:
            raise PRCompletionError(
                f"GraphQL-done repo still has a resume cursor: {repo}"
            )
        total_count = record.get("total_count")
        if (
            not isinstance(total_count, int)
            or isinstance(total_count, bool)
            or total_count < 0
        ):
            raise PRCompletionError(
                f"GraphQL-done repo lacks a valid total_count: {repo}: {total_count!r}"
            )
        truncated = record.get("truncated", 0)
        if (
            not isinstance(truncated, int)
            or isinstance(truncated, bool)
            or truncated < 0
        ):
            raise PRCompletionError(
                f"GraphQL repo has invalid truncated count: {repo}: {truncated!r}"
            )
        declared_counts[repo] = total_count
        declared_truncated += truncated
    if nonterminal:
        raise PRCompletionError(
            "expected repos are not terminal GraphQL-done: "
            + ", ".join(f"{repo}={status!r}" for repo, status in nonterminal[:20])
        )

    targets = _load_targets(truncated_targets_path)
    if declared_truncated != len(targets):
        raise PRCompletionError(
            "GraphQL truncated count does not match the unique target set: "
            f"manifest={declared_truncated} targets={len(targets)}"
        )
    gap_binding, gap_record_sha256s = _verify_gap_completion(
        targets=targets,
        gap_completion_path=gap_completion_path,
    )

    _checkpoint_sqlite(store_path)
    _require_checkpointed_store(store_path)
    store_sha256 = sha256_file(store_path)
    try:
        uri = store_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            stored_counts = {
                str(repo): int(count)
                for repo, count in conn.execute(
                    "SELECT repo, COUNT(*) FROM prs "
                    "WHERE scan_id=? GROUP BY repo",
                    (scan_id,),
                )
            }
            total_store_pr_count = int(
                conn.execute("SELECT COUNT(*) FROM prs").fetchone()[0]
            )
            for target, expected_record_sha256 in gap_record_sha256s.items():
                stored = pr_store.get_by_pr(conn, target[0], target[1])
                if stored is None:
                    raise PRCompletionError(
                        f"gap-completed PR is absent from the store: "
                        f"{target[0]}#{target[1]}"
                    )
                if stored.get("scan_id") != scan_id:
                    raise PRCompletionError(
                        f"gap-completed PR is outside the verified scan: "
                        f"{target[0]}#{target[1]}"
                    )
                actual_record_sha256 = pr_store.record_content_sha256(stored)
                if actual_record_sha256 != expected_record_sha256:
                    raise PRCompletionError(
                        f"gap-completed PR content hash mismatch: "
                        f"{target[0]}#{target[1]}"
                    )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise PRCompletionError(f"cannot inspect PR store: {store_path}: {exc}") from exc

    unexpected_store_repos = sorted(set(stored_counts) - set(expected_repos))
    if unexpected_store_repos:
        raise PRCompletionError(
            "PR store contains repos outside the bound repo list: "
            f"{unexpected_store_repos[:10]}"
        )
    count_mismatches = [
        (repo, declared_counts[repo], stored_counts.get(repo, 0))
        for repo in expected_repos
        if stored_counts.get(repo, 0) != declared_counts[repo]
    ]
    if count_mismatches:
        raise PRCompletionError(
            "stored PR count mismatch against GraphQL totalCount: "
            + ", ".join(
                f"{repo}:declared={declared}:stored={stored}"
                for repo, declared, stored in count_mismatches[:20]
            )
        )

    _require_checkpointed_store(store_path)
    _require_unchanged_file(
        store_path,
        store_sha256,
        what="PR store",
    )
    _require_unchanged_file(
        repo_list_path,
        repo_list_sha256,
        what="PR repo list",
    )
    _require_unchanged_file(
        graphql_manifest_path,
        graphql_manifest_sha256,
        what="GraphQL PR stream manifest",
    )
    if gap_binding is not None:
        _require_unchanged_file(
            Path(str(gap_binding["path"])),
            str(gap_binding["sha256"]),
            what="PR gap completion receipt",
        )

    receipt = {
        "schema": PR_COMPLETION_SCHEMA,
        "status": "verified",
        "repo_list": {
            "path": str(repo_list_path.resolve()),
            "sha256": repo_list_sha256,
        },
        "graphql_manifest": {
            "path": str(graphql_manifest_path.resolve()),
            "sha256": graphql_manifest_sha256,
        },
        "scan_id": scan_id,
        "pr_store": {
            "path": str(store_path.resolve()),
            "sha256": store_sha256,
            "size": store_path.stat().st_size,
        },
        "gap_completion": gap_binding,
        "expected_repo_count": len(expected_repos),
        "expected_repos_sha256": _canonical_json_sha256(list(expected_repos)),
        "declared_pr_count": sum(declared_counts.values()),
        "stored_pr_count": sum(stored_counts.values()),
        "unverified_store_pr_count": (
            total_store_pr_count - sum(stored_counts.values())
        ),
        "truncated_target_count": len(targets),
    }
    if output_path is not None:
        _atomic_write_json(output_path, receipt)
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-list", required=True, type=Path)
    parser.add_argument("--graphql-manifest", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--truncated-targets", type=Path)
    parser.add_argument("--gap-completion", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        receipt = verify_pr_completion(
            repo_list_path=args.repo_list,
            graphql_manifest_path=args.graphql_manifest,
            store_path=args.store,
            truncated_targets_path=args.truncated_targets,
            gap_completion_path=args.gap_completion,
            output_path=args.output,
        )
    except PRCompletionError as exc:
        raise SystemExit(f"PR_COMPLETION_FAILED: {exc}") from exc
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
