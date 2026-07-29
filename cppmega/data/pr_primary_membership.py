"""Fail-closed PR membership derived from primary commit parquet provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterator

import pyarrow.parquet as pq

from cppmega.data.nanochat_pipeline.packed_rows_schema import (
    NUM_DOCS_COLUMN,
    SOURCE_COMMIT_HASHES_COLUMN,
    SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
    SOURCE_PR_NUMBERS_COLUMN,
)
from cppmega.data.nanochat_pipeline.tokenized_enriched_schema import REPO_COLUMN
from cppmega.data.source_conveyor_composition import (
    SOURCE_COMPOSITION_SCHEMA,
    SourceComposition,
)


PRIMARY_PR_MEMBERSHIP_SCHEMA = "cppmega_primary_pr_membership_v1"
PRIMARY_PR_MEMBERSHIP_POLICY = (
    "exact_allowlisted_primary_commit_source_documents_v1"
)
PRIMARY_PR_MEMBERSHIP_TABLE = "_cppmega_primary_pr_membership"
_SOURCE_REFS_TABLE = "_cppmega_primary_pr_source_refs"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_COLUMNS = frozenset(
    {
        REPO_COLUMN,
        NUM_DOCS_COLUMN,
        SOURCE_PR_NUMBERS_COLUMN,
        SOURCE_COMMIT_HASHES_COLUMN,
        SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
    }
)


@dataclass(frozen=True)
class _CommitArtifact:
    bucket: int
    filename: str
    path: Path
    rows: int
    byte_size: int
    sha256: str


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_scan_id(scan_id: object) -> str:
    if not isinstance(scan_id, str) or _SHA256_RE.fullmatch(scan_id) is None:
        raise ValueError("PR scan_id must be a lowercase SHA-256")
    return scan_id


def _resolve_commit_artifacts(
    *,
    source_composition: SourceComposition,
    commit_root: Path,
    buckets: tuple[int, ...],
) -> tuple[list[_CommitArtifact], dict[str, object]]:
    if (
        source_composition.receipt.get("schema") != SOURCE_COMPOSITION_SCHEMA
        or source_composition.receipt.get("status") != "complete"
    ):
        raise ValueError("source composition is not complete")
    if not buckets or len(set(buckets)) != len(buckets):
        raise ValueError("primary PR membership buckets must be unique and non-empty")
    normalized_buckets = tuple(sorted(int(bucket) for bucket in buckets))
    if any(bucket <= 0 for bucket in normalized_buckets):
        raise ValueError("primary PR membership buckets must be positive")

    raw_root = commit_root.expanduser()
    if raw_root.is_symlink():
        raise ValueError(f"commit root must not be a symlink: {raw_root}")
    resolved_root = raw_root.resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(resolved_root)

    artifacts: list[_CommitArtifact] = []
    bucket_stats: dict[str, dict[str, int]] = {}
    for bucket in normalized_buckets:
        allowlist = source_composition.allowlist.get(("commits", bucket))
        if not isinstance(allowlist, dict) or not allowlist:
            raise ValueError(
                f"source composition has no commit allowlist for bucket {bucket}"
            )
        bucket_root = raw_root / str(bucket)
        if bucket_root.is_symlink():
            raise ValueError(
                f"commit bucket root must not be a symlink: {bucket_root}"
            )
        resolved_bucket_root = bucket_root.resolve()
        try:
            resolved_bucket_root.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"commit bucket escapes commit root: {bucket_root}"
            ) from exc
        if not resolved_bucket_root.is_dir():
            raise FileNotFoundError(resolved_bucket_root)

        bucket_rows = 0
        bucket_bytes = 0
        for filename, raw_expected_rows in sorted(allowlist.items()):
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).is_absolute()
                or ".." in Path(filename).parts
            ):
                raise ValueError(
                    f"invalid commit allowlist filename for bucket {bucket}: "
                    f"{filename!r}"
                )
            if (
                isinstance(raw_expected_rows, bool)
                or not isinstance(raw_expected_rows, int)
                or raw_expected_rows < 1
            ):
                raise ValueError(
                    f"invalid commit allowlist row count for "
                    f"{bucket}/{filename}: {raw_expected_rows!r}"
                )
            raw_path = bucket_root / filename
            if raw_path.is_symlink():
                raise ValueError(
                    f"commit artifact must not be a symlink: {raw_path}"
                )
            path = raw_path.resolve()
            try:
                path.relative_to(resolved_bucket_root)
            except ValueError as exc:
                raise ValueError(
                    f"commit artifact escapes bucket root: {raw_path}"
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(path)

            parquet = pq.ParquetFile(path)
            actual_rows = int(parquet.metadata.num_rows)
            if actual_rows != raw_expected_rows:
                raise ValueError(
                    f"{path}: source composition rows={raw_expected_rows}, "
                    f"parquet rows={actual_rows}"
                )
            missing = sorted(_REQUIRED_COLUMNS - set(parquet.schema_arrow.names))
            if missing:
                raise ValueError(
                    f"{path}: primary commit parquet missing provenance columns: "
                    + ", ".join(missing)
                )
            byte_size = path.stat().st_size
            artifact = _CommitArtifact(
                bucket=bucket,
                filename=filename,
                path=path,
                rows=actual_rows,
                byte_size=byte_size,
                sha256=_sha256_file(path),
            )
            artifacts.append(artifact)
            bucket_rows += actual_rows
            bucket_bytes += byte_size
        bucket_stats[str(bucket)] = {
            "files": len(allowlist),
            "rows": bucket_rows,
            "byte_size": bucket_bytes,
        }

    artifact_identities = [
        {
            "bucket": artifact.bucket,
            "filename": artifact.filename,
            "rows": artifact.rows,
            "byte_size": artifact.byte_size,
            "sha256": artifact.sha256,
        }
        for artifact in artifacts
    ]
    source_composition_sha256 = _canonical_sha256(source_composition.receipt)
    binding: dict[str, object] = {
        "schema": "cppmega_primary_commit_artifact_binding_v1",
        "source_composition_sha256": source_composition_sha256,
        "source_composition_plan_sha256": source_composition.receipt.get(
            "plan_sha256"
        ),
        "buckets": list(normalized_buckets),
        "files": len(artifacts),
        "rows": sum(artifact.rows for artifact in artifacts),
        "byte_size": sum(artifact.byte_size for artifact in artifacts),
        "artifact_set_sha256": _canonical_sha256(artifact_identities),
        "by_bucket": bucket_stats,
    }
    return artifacts, binding


def primary_commit_artifact_binding(
    *,
    source_composition: SourceComposition,
    commit_root: Path,
    buckets: tuple[int, ...],
) -> dict[str, object]:
    """Hash and validate the exact commit shards used to derive PR membership."""

    _artifacts, binding = _resolve_commit_artifacts(
        source_composition=source_composition,
        commit_root=commit_root,
        buckets=buckets,
    )
    return binding


def _create_membership_tables(conn: sqlite3.Connection) -> None:
    conn.execute(f"DROP TABLE IF EXISTS temp.{PRIMARY_PR_MEMBERSHIP_TABLE}")
    conn.execute(f"DROP TABLE IF EXISTS temp.{_SOURCE_REFS_TABLE}")
    conn.execute(
        f"""
        CREATE TEMP TABLE {PRIMARY_PR_MEMBERSHIP_TABLE} (
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL CHECK(pr_number > 0),
            PRIMARY KEY(repo, pr_number)
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        f"""
        CREATE TEMP TABLE {_SOURCE_REFS_TABLE} (
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            commit_hash TEXT NOT NULL,
            source_docs INTEGER NOT NULL CHECK(source_docs > 0),
            PRIMARY KEY(repo, pr_number, commit_hash)
        ) WITHOUT ROWID
        """
    )


def _insert_source_refs(
    conn: sqlite3.Connection,
    refs: dict[tuple[str, int, str], int],
) -> None:
    conn.executemany(
        f"""
        INSERT INTO {_SOURCE_REFS_TABLE}(
            repo, pr_number, commit_hash, source_docs
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT(repo, pr_number, commit_hash) DO UPDATE SET
            source_docs = source_docs + excluded.source_docs
        """,
        [
            (repo, pr_number, commit_hash, source_docs)
            for (repo, pr_number, commit_hash), source_docs in sorted(refs.items())
        ],
    )


def _scan_source_documents(
    conn: sqlite3.Connection,
    artifacts: list[_CommitArtifact],
) -> dict[str, int]:
    source_docs = 0
    source_docs_with_pr_number = 0
    source_docs_with_pr_discussion = 0
    ignored_unverified_pr_number_source_docs = 0
    source_docs_with_commit_sha = 0
    rows = 0
    for artifact in artifacts:
        parquet = pq.ParquetFile(artifact.path)
        rows_in_artifact = 0
        for batch in parquet.iter_batches(
            columns=[
                REPO_COLUMN,
                NUM_DOCS_COLUMN,
                SOURCE_PR_NUMBERS_COLUMN,
                SOURCE_COMMIT_HASHES_COLUMN,
                SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
            ],
            batch_size=1024,
        ):
            refs: dict[tuple[str, int, str], int] = {}
            for row in batch.to_pylist():
                rows += 1
                rows_in_artifact += 1
                repo = row.get(REPO_COLUMN)
                if not isinstance(repo, str) or not repo.strip():
                    raise ValueError(
                        f"{artifact.path}: packed commit row {rows_in_artifact - 1} "
                        "has no canonical repo"
                    )
                raw_num_docs = row.get(NUM_DOCS_COLUMN)
                if (
                    isinstance(raw_num_docs, bool)
                    or not isinstance(raw_num_docs, int)
                    or raw_num_docs < 1
                ):
                    raise ValueError(
                        f"{artifact.path}: packed commit row "
                        f"{rows_in_artifact - 1} has invalid num_docs"
                    )
                pr_numbers = row.get(SOURCE_PR_NUMBERS_COLUMN)
                commit_hashes = row.get(SOURCE_COMMIT_HASHES_COLUMN)
                has_pr_discussions = row.get(SOURCE_HAS_PR_DISCUSSIONS_COLUMN)
                if (
                    not isinstance(pr_numbers, list)
                    or not isinstance(commit_hashes, list)
                    or not isinstance(has_pr_discussions, list)
                    or len(pr_numbers) != raw_num_docs
                    or len(commit_hashes) != raw_num_docs
                    or len(has_pr_discussions) != raw_num_docs
                ):
                    raise ValueError(
                        f"{artifact.path}: per-document provenance shape differs "
                        f"from num_docs at row {rows_in_artifact - 1}: "
                        f"num_docs={raw_num_docs} "
                        f"pr_numbers={len(pr_numbers) if isinstance(pr_numbers, list) else None} "
                        f"commit_hashes={len(commit_hashes) if isinstance(commit_hashes, list) else None} "
                        f"has_pr_discussions={len(has_pr_discussions) if isinstance(has_pr_discussions, list) else None}"
                    )
                for doc_index, (
                    raw_pr_number,
                    raw_commit_hash,
                    raw_has_pr_discussion,
                ) in enumerate(
                    zip(
                        pr_numbers,
                        commit_hashes,
                        has_pr_discussions,
                        strict=True,
                    )
                ):
                    if (
                        not isinstance(raw_commit_hash, str)
                        or _COMMIT_RE.fullmatch(raw_commit_hash) is None
                    ):
                        raise ValueError(
                            f"{artifact.path}: invalid source commit hash at "
                            f"row {rows_in_artifact - 1}, doc {doc_index}"
                        )
                    if raw_pr_number is None:
                        pr_number = -1
                    elif (
                        isinstance(raw_pr_number, bool)
                        or not isinstance(raw_pr_number, int)
                        or raw_pr_number < 1
                    ):
                        raise ValueError(
                            f"{artifact.path}: invalid source PR number at "
                            f"row {rows_in_artifact - 1}, doc {doc_index}"
                        )
                    else:
                        source_docs_with_pr_number += 1
                        pr_number = raw_pr_number
                    if not isinstance(raw_has_pr_discussion, bool):
                        raise ValueError(
                            f"{artifact.path}: invalid source PR discussion flag at "
                            f"row {rows_in_artifact - 1}, doc {doc_index}"
                        )
                    if raw_has_pr_discussion:
                        if pr_number < 1:
                            raise ValueError(
                                f"{artifact.path}: PR discussion has no canonical "
                                f"PR number at row {rows_in_artifact - 1}, "
                                f"doc {doc_index}"
                            )
                        source_docs_with_pr_discussion += 1
                    else:
                        if pr_number > 0:
                            ignored_unverified_pr_number_source_docs += 1
                        pr_number = -1
                    source_docs += 1
                    source_docs_with_commit_sha += 1
                    key = (repo, pr_number, raw_commit_hash)
                    refs[key] = refs.get(key, 0) + 1
            if refs:
                _insert_source_refs(conn, refs)
        if rows_in_artifact != artifact.rows:
            raise RuntimeError(
                f"{artifact.path}: iterated rows={rows_in_artifact}, "
                f"metadata rows={artifact.rows}"
            )
    return {
        "rows": rows,
        "source_docs": source_docs,
        "source_docs_with_pr_number": source_docs_with_pr_number,
        "source_docs_with_pr_discussion": source_docs_with_pr_discussion,
        "ignored_unverified_pr_number_source_docs": (
            ignored_unverified_pr_number_source_docs
        ),
        "source_docs_with_commit_sha": source_docs_with_commit_sha,
    }


def _materialize_membership(
    conn: sqlite3.Connection,
    *,
    scan_id: str,
) -> dict[str, int]:
    unmatched_direct = conn.execute(
        f"""
        SELECT r.repo, r.pr_number
        FROM {_SOURCE_REFS_TABLE} AS r
        LEFT JOIN prs AS p
          ON p.repo = r.repo
         AND p.pr_number = r.pr_number
         AND p.scan_id = ?
        WHERE r.pr_number > 0 AND p.pr_number IS NULL
        LIMIT 1
        """,
        (scan_id,),
    ).fetchone()
    if unmatched_direct is not None:
        raise ValueError(
            "primary commit provenance references a PR outside the exact "
            f"verified scan: {unmatched_direct['repo']}#{unmatched_direct['pr_number']}"
        )

    conflict = conn.execute(
        f"""
        SELECT r.repo, r.pr_number AS direct_pr, p.pr_number AS sha_pr
        FROM {_SOURCE_REFS_TABLE} AS r
        JOIN pr_by_sha AS s
          ON s.repo = r.repo
         AND s.merge_commit_sha = r.commit_hash
        JOIN prs AS p
          ON p.repo = s.repo
         AND p.pr_number = s.pr_number
         AND p.merge_commit_sha = s.merge_commit_sha
         AND p.scan_id = ?
        WHERE r.pr_number > 0 AND r.pr_number != p.pr_number
        LIMIT 1
        """,
        (scan_id,),
    ).fetchone()
    if conflict is not None:
        raise ValueError(
            "primary commit PR number conflicts with merge SHA mapping: "
            f"{conflict['repo']} direct=#{conflict['direct_pr']} "
            f"sha=#{conflict['sha_pr']}"
        )

    conn.execute(
        f"""
        INSERT OR IGNORE INTO {PRIMARY_PR_MEMBERSHIP_TABLE}(repo, pr_number)
        SELECT DISTINCT repo, pr_number
        FROM {_SOURCE_REFS_TABLE}
        WHERE pr_number > 0
        """
    )
    conn.execute(
        f"""
        INSERT OR IGNORE INTO {PRIMARY_PR_MEMBERSHIP_TABLE}(repo, pr_number)
        SELECT DISTINCT p.repo, p.pr_number
        FROM {_SOURCE_REFS_TABLE} AS r
        JOIN pr_by_sha AS s
          ON s.repo = r.repo
         AND s.merge_commit_sha = r.commit_hash
        JOIN prs AS p
          ON p.repo = s.repo
         AND p.pr_number = s.pr_number
         AND p.merge_commit_sha = s.merge_commit_sha
         AND p.scan_id = ?
        WHERE r.pr_number = -1
        """,
        (scan_id,),
    )
    sha_only_matched = int(
        conn.execute(
            f"""
            SELECT COALESCE(SUM(r.source_docs), 0)
            FROM {_SOURCE_REFS_TABLE} AS r
            JOIN pr_by_sha AS s
              ON s.repo = r.repo
             AND s.merge_commit_sha = r.commit_hash
            JOIN prs AS p
              ON p.repo = s.repo
             AND p.pr_number = s.pr_number
             AND p.merge_commit_sha = s.merge_commit_sha
             AND p.scan_id = ?
            WHERE r.pr_number = -1
            """,
            (scan_id,),
        ).fetchone()[0]
    )
    unmatched_sha = int(
        conn.execute(
            f"""
            SELECT COALESCE(SUM(r.source_docs), 0)
            FROM {_SOURCE_REFS_TABLE} AS r
            LEFT JOIN pr_by_sha AS s
              ON s.repo = r.repo
             AND s.merge_commit_sha = r.commit_hash
            LEFT JOIN prs AS p
              ON p.repo = s.repo
             AND p.pr_number = s.pr_number
             AND p.merge_commit_sha = s.merge_commit_sha
             AND p.scan_id = ?
            WHERE r.pr_number = -1 AND p.pr_number IS NULL
            """,
            (scan_id,),
        ).fetchone()[0]
    )
    selected = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {PRIMARY_PR_MEMBERSHIP_TABLE}"
        ).fetchone()[0]
    )
    if selected < 1:
        raise RuntimeError(
            "primary commit provenance selected zero PR discussions"
        )
    return {
        "selected_pr_count": selected,
        "sha_only_matched_source_docs": sha_only_matched,
        "unmatched_commit_sha_source_docs": unmatched_sha,
    }


def _membership_sha256(conn: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for row in conn.execute(
        f"""
        SELECT repo, pr_number
        FROM {PRIMARY_PR_MEMBERSHIP_TABLE}
        ORDER BY repo, pr_number
        """
    ):
        encoded = f"{row['repo']}\0{int(row['pr_number'])}".encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_primary_pr_membership(
    conn: sqlite3.Connection,
    *,
    source_composition: SourceComposition,
    commit_root: Path,
    buckets: tuple[int, ...],
    scan_id: str,
) -> dict[str, object]:
    """Build a TEMP PR allowlist from exact per-document primary commit provenance."""

    scan_id = _require_scan_id(scan_id)
    artifacts, artifact_binding = _resolve_commit_artifacts(
        source_composition=source_composition,
        commit_root=commit_root,
        buckets=buckets,
    )
    _create_membership_tables(conn)
    scan_stats = _scan_source_documents(conn, artifacts)
    membership_stats = _materialize_membership(conn, scan_id=scan_id)
    receipt: dict[str, object] = {
        "schema": PRIMARY_PR_MEMBERSHIP_SCHEMA,
        "policy": PRIMARY_PR_MEMBERSHIP_POLICY,
        "scan_id": scan_id,
        "commit_artifacts": artifact_binding,
        **scan_stats,
        **membership_stats,
        "selected_membership_sha256": _membership_sha256(conn),
        "validation": {
            "source_composition_complete": True,
            "exact_allowlisted_commit_artifacts": True,
            "exact_source_doc_shapes": True,
            "exact_scan_membership": True,
            "direct_pr_sha_conflicts": 0,
        },
    }
    return receipt


def verify_primary_pr_membership_binding(
    membership: object,
    *,
    source_composition: SourceComposition,
    commit_root: Path,
    buckets: tuple[int, ...],
) -> None:
    """Revalidate that a membership receipt still points at identical commit bytes."""

    if (
        not isinstance(membership, dict)
        or membership.get("schema") != PRIMARY_PR_MEMBERSHIP_SCHEMA
        or membership.get("policy") != PRIMARY_PR_MEMBERSHIP_POLICY
    ):
        raise RuntimeError("primary PR membership receipt is unsupported")
    try:
        current = primary_commit_artifact_binding(
            source_composition=source_composition,
            commit_root=commit_root,
            buckets=buckets,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"primary PR membership commit artifact binding drifted: {exc}"
        ) from exc
    if membership.get("commit_artifacts") != current:
        raise RuntimeError("primary PR membership commit artifact binding drifted")


def iter_primary_pr_keys(
    conn: sqlite3.Connection,
    *,
    repo: str | None,
    scan_id: str,
    offset: int,
    limit: int | None,
) -> Iterator[sqlite3.Row]:
    """Iterate exact-scan PR keys admitted by the TEMP primary membership table."""

    scan_id = _require_scan_id(scan_id)
    sql = (
        f"SELECT p.repo, p.pr_number "
        f"FROM prs AS p "
        f"JOIN {PRIMARY_PR_MEMBERSHIP_TABLE} AS m "
        f"ON m.repo=p.repo AND m.pr_number=p.pr_number "
        "WHERE p.scan_id=?"
    )
    params: list[object] = [scan_id]
    if repo:
        sql += " AND p.repo=?"
        params.append(repo)
    sql += " ORDER BY p.repo, p.pr_number LIMIT ? OFFSET ?"
    params.append(-1 if limit is None else int(limit))
    params.append(int(offset))
    yield from conn.execute(sql, params)


def count_primary_pr_keys(
    conn: sqlite3.Connection,
    *,
    repo: str | None,
    scan_id: str,
    offset: int,
    limit: int | None,
) -> int:
    """Count exact-scan PR keys admitted by the TEMP primary membership table."""

    scan_id = _require_scan_id(scan_id)
    sql = (
        "SELECT COUNT(*) AS n FROM ("
        "SELECT 1 FROM prs AS p "
        f"JOIN {PRIMARY_PR_MEMBERSHIP_TABLE} AS m "
        "ON m.repo=p.repo AND m.pr_number=p.pr_number "
        "WHERE p.scan_id=?"
    )
    params: list[object] = [scan_id]
    if repo:
        sql += " AND p.repo=?"
        params.append(repo)
    sql += " ORDER BY p.repo, p.pr_number LIMIT ? OFFSET ?)"
    params.append(-1 if limit is None else int(limit))
    params.append(int(offset))
    return int(conn.execute(sql, params).fetchone()["n"])


__all__ = [
    "PRIMARY_PR_MEMBERSHIP_POLICY",
    "PRIMARY_PR_MEMBERSHIP_SCHEMA",
    "PRIMARY_PR_MEMBERSHIP_TABLE",
    "build_primary_pr_membership",
    "count_primary_pr_keys",
    "iter_primary_pr_keys",
    "primary_commit_artifact_binding",
    "verify_primary_pr_membership_binding",
]
