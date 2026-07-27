#!/usr/bin/env python3
"""Auditably requeue complete CI ZIPs preserved after interrupted parsing.

Dry-run is the default. Stop both the fetcher and rescue writer before
``--apply``. A candidate is accepted only when the entire bounded ZIP is
readable and every already-committed member matches by name, size, and SHA-256.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import stat
import sys
from typing import Mapping, Sequence
import zipfile


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci_job_log_rescue import (  # noqa: E402
    FetchStateEvidence,
    JobLogRescueWorker,
    StateBindingError,
    _atomic_write_bytes,
    _canonical_json_bytes,
    _row_sha256,
    _sha256_bytes,
    _sha256_file,
    _utc_now,
)
from scripts.ci_stream_fetch import (  # noqa: E402
    ArchiveError as FetchArchiveError,
    DEFAULT_MAX_ARCHIVE_BYTES,
    DEFAULT_MAX_MEMBER_BYTES,
    DEFAULT_MAX_MEMBERS,
    DEFAULT_MAX_UNCOMPRESSED_BYTES,
    _fsync_directory,
    _safe_zip_infos,
)


SCHEMA = "cppmega_ci_preserved_archive_recovery_v1"
_STREAM_BYTES = 1024 * 1024
_RECEIPT_MAX_BYTES = 4 * 1024 * 1024
_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
_MANIFEST_FIELDS = (
    "repo",
    "run_id",
    "attempt",
    "created_at",
    "status",
    "bytes",
    "sha256",
    "finished_at",
)
_ELIGIBLE = ("failed", "terminal_404", "terminal_410")


class RecoveryError(RuntimeError):
    """Preserved-archive evidence is absent, unsafe, changed, or ambiguous."""


@dataclass(frozen=True)
class ArchiveProof:
    path: Path
    sha256: str
    byte_size: int
    member_count: int
    uncompressed_bytes: int


@dataclass(frozen=True)
class RecoveryPlan:
    state_path: Path
    rescue_spool: Path
    row: dict[str, object]
    row_sha256: str
    witnesses: tuple[tuple[object, ...], ...]
    witness_set_sha256: str
    archive: ArchiveProof
    rejected_candidates: tuple[dict[str, str], ...]

    @property
    def identity(self) -> tuple[str, int, int]:
        return (
            str(self.row["repo"]),
            int(self.row["run_id"]),
            int(self.row["attempt"]),
        )

    @property
    def spool_base_name(self) -> str:
        repo, run_id, attempt = self.identity
        return f"{repo.replace('/', '__')}--{run_id}--attempt-{attempt}"

    @property
    def rescue_archive(self) -> Path:
        return self.rescue_spool / f"{self.spool_base_name}.zip"

    def proof(self) -> dict[str, object]:
        repo, run_id, attempt = self.identity
        return {
            "state": {
                "path": str(self.state_path),
                "attempt_row_sha256": self.row_sha256,
            },
            "attempt": {
                "repo": repo,
                "run_id": run_id,
                "attempt": attempt,
                "created_at": str(self.row["created_at"]),
                "prior_status": str(self.row["status"]),
                "tries": int(self.row["tries"]),
                "terminal_http_status": self.row["terminal_http_status"],
                "terminal_body_sha256": self.row["terminal_body_sha256"],
            },
            "durable_member_witness": {
                "count": len(self.witnesses),
                "chunk_count": sum(int(item[4]) for item in self.witnesses),
                "occurrence_tokens": sum(
                    int(item[5]) for item in self.witnesses
                ),
                "set_sha256": self.witness_set_sha256,
            },
            "source_archive": {
                "path": str(self.archive.path),
                "bytes": self.archive.byte_size,
                "sha256": self.archive.sha256,
                "zip_members": self.archive.member_count,
                "uncompressed_bytes": self.archive.uncompressed_bytes,
            },
            "rescue_archive": {
                "path": str(self.rescue_archive),
                "bytes": self.archive.byte_size,
                "sha256": self.archive.sha256,
            },
            "verification": {
                "complete_zip_crc_read": True,
                "all_durable_members_matched_name_size_sha256": True,
                "different_valid_archive_candidates_rejected": True,
            },
            "rejected_candidates": list(self.rejected_candidates),
            "recovery_script_sha256": _sha256_file(Path(__file__).resolve()),
        }


def _safe_directory(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise RecoveryError(f"unsafe symlink directory: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise RecoveryError(f"unsafe directory: {resolved}")
    return resolved


def _attempt_rows(
    connection: sqlite3.Connection,
    target: tuple[str, int, int] | None,
) -> tuple[dict[str, object], ...]:
    target_clause = (
        ""
        if target is None
        else "AND lower(repo)=lower(?) AND run_id=? AND attempt=?"
    )
    rows = connection.execute(
        f"""
        SELECT * FROM attempts
        WHERE status IN ('failed','terminal_404','terminal_410')
          {target_clause}
          AND EXISTS (
            SELECT 1 FROM members
            WHERE members.repo=attempts.repo
              AND members.run_id=attempts.run_id
              AND members.attempt=attempts.attempt
          )
        ORDER BY created_at,repo,run_id,attempt
        """,
        () if target is None else target,
    ).fetchall()
    if target is not None and not rows:
        raise RecoveryError(
            "explicit target is not eligible and backed by durable members"
        )
    return tuple(dict(row) for row in rows)


def _witnesses(
    connection: sqlite3.Connection,
    identity: tuple[str, int, int],
) -> tuple[tuple[object, ...], ...]:
    values = tuple(
        (
            str(row["archive_member"]),
            str(row["job_key"]),
            str(row["raw_sha256"]),
            int(row["raw_size"]),
            int(row["chunk_count"]),
            int(row["occurrence_tokens"]),
        )
        for row in connection.execute(
            """
            SELECT archive_member,job_key,raw_sha256,raw_size,
                   chunk_count,occurrence_tokens
            FROM members
            WHERE repo=? AND run_id=? AND attempt=?
            ORDER BY archive_member
            """,
            identity,
        )
    )
    if not values:
        raise RecoveryError(f"attempt has no member witnesses: {identity}")
    if any(
        not item[0]
        or re.fullmatch(r"[0-9a-f]{64}", str(item[2])) is None
        or any(int(item[index]) < 0 for index in (3, 4, 5))
        for item in values
    ):
        raise RecoveryError(f"attempt has an invalid member witness: {identity}")
    return values


def _witness_digest(values: Sequence[tuple[object, ...]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(values)))


def _candidate_paths(
    work_dir: Path,
    identity: tuple[str, int, int],
) -> tuple[Path, ...]:
    repo, run_id, attempt = identity
    prefix = f"{repo.replace('/', '__')}--{run_id}--{attempt}--"
    pattern = re.compile(
        rf"{re.escape(prefix)}[A-Za-z0-9_-]+"
        r"\.zip\.partial(?:\.\d+)?"
    )
    candidates: list[Path] = []
    for directory in (work_dir / "failed", work_dir / "tmp"):
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise RecoveryError(f"unsafe work directory: {directory}")
        candidates.extend(
            path
            for path in sorted(directory.iterdir())
            if pattern.fullmatch(path.name)
        )
    return tuple(candidates)


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise RecoveryError(f"candidate is not a regular file: {path}")
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_archive(
    path: Path,
    witnesses: Sequence[tuple[object, ...]],
    *,
    max_archive_bytes: int,
    max_member_bytes: int,
    max_uncompressed_bytes: int,
    max_members: int,
) -> ArchiveProof:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"unsafe candidate: {path}")
    before = _identity(path)
    if not 0 < before[2] <= max_archive_bytes:
        raise RecoveryError(f"candidate exceeds archive bound: {path}")
    digest = _sha256_file(path)
    try:
        infos = _safe_zip_infos(
            path,
            max_members=max_members,
            max_member_bytes=max_member_bytes,
            max_uncompressed_bytes=max_uncompressed_bytes,
        )
    except FetchArchiveError as exc:
        raise RecoveryError(f"unsafe candidate ZIP: {path}: {exc}") from exc
    expected = {str(item[0]): item for item in witnesses}
    matched: set[str] = set()
    uncompressed = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for info in infos:
                size = 0
                member_digest = hashlib.sha256()
                with archive.open(info) as source:
                    while block := source.read(_STREAM_BYTES):
                        size += len(block)
                        if size > info.file_size or size > max_member_bytes:
                            raise RecoveryError(
                                f"member exceeded bound: {info.filename}"
                            )
                        member_digest.update(block)
                if size != info.file_size:
                    raise RecoveryError(f"truncated member: {info.filename}")
                uncompressed += size
                witness = expected.get(info.filename)
                if witness is not None:
                    if (
                        size != int(witness[3])
                        or member_digest.hexdigest() != str(witness[2])
                    ):
                        raise RecoveryError(
                            f"member witness mismatch: {info.filename}"
                        )
                    matched.add(info.filename)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise RecoveryError(f"cannot read complete ZIP {path}: {exc}") from exc
    missing = sorted(set(expected) - matched)
    if missing:
        raise RecoveryError(
            "candidate lacks witnessed members: " + ", ".join(missing[:5])
        )
    if _identity(path) != before:
        raise RecoveryError(f"candidate changed during validation: {path}")
    return ArchiveProof(
        path=path.resolve(),
        sha256=digest,
        byte_size=before[2],
        member_count=len(infos),
        uncompressed_bytes=uncompressed,
    )


def _select_archive(
    row: Mapping[str, object],
    paths: Sequence[Path],
    witnesses: Sequence[tuple[object, ...]],
    **limits: int,
) -> tuple[ArchiveProof, tuple[dict[str, str], ...]]:
    if not paths:
        raise RecoveryError("no preserved candidate archive")
    bound = (
        row["archive_source"],
        row["archive_sha256"],
        row["archive_size"],
    )
    if not (
        all(value is None for value in bound)
        or all(value is not None for value in bound)
    ):
        raise RecoveryError("attempt archive binding is incomplete")
    valid: list[ArchiveProof] = []
    rejected: list[dict[str, str]] = []
    for path in paths:
        try:
            proof = _validate_archive(path, witnesses, **limits)
            if bound[1] is not None and (
                proof.sha256 != str(bound[1])
                or proof.byte_size != int(bound[2])
            ):
                raise RecoveryError("candidate differs from attempt binding")
            valid.append(proof)
        except RecoveryError as exc:
            rejected.append({"path": str(path.resolve()), "reason": str(exc)})
    if not valid:
        raise RecoveryError(
            "no candidate satisfies ZIP and durable-member evidence"
        )
    if len({(item.sha256, item.byte_size) for item in valid}) != 1:
        raise RecoveryError("multiple different valid archives are ambiguous")
    return sorted(valid, key=lambda item: str(item.path))[0], tuple(rejected)


def build_plans(
    *,
    state_path: Path,
    work_dir: Path,
    rescue_spool: Path,
    target: tuple[str, int, int] | None = None,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> tuple[RecoveryPlan, ...]:
    expanded_state = state_path.expanduser()
    if expanded_state.is_symlink():
        raise RecoveryError(f"unsafe state symlink: {expanded_state}")
    state_path = expanded_state.resolve()
    work_dir = _safe_directory(work_dir)
    rescue_spool = _safe_directory(rescue_spool)
    state = FetchStateEvidence(state_path)
    try:
        plans: list[RecoveryPlan] = []
        for row in _attempt_rows(state.connection, target):
            identity = (
                str(row["repo"]),
                int(row["run_id"]),
                int(row["attempt"]),
            )
            members = _witnesses(state.connection, identity)
            archive, rejected = _select_archive(
                row,
                _candidate_paths(work_dir, identity),
                members,
                max_archive_bytes=max_archive_bytes,
                max_member_bytes=max_member_bytes,
                max_uncompressed_bytes=max_uncompressed_bytes,
                max_members=max_members,
            )
            plans.append(
                RecoveryPlan(
                    state_path=state_path,
                    rescue_spool=rescue_spool,
                    row=row,
                    row_sha256=_row_sha256(row),
                    witnesses=members,
                    witness_set_sha256=_witness_digest(members),
                    archive=archive,
                    rejected_candidates=rejected,
                )
            )
        return tuple(plans)
    finally:
        state.close()


def _receipt(plan: RecoveryPlan) -> tuple[Path, dict[str, object], str]:
    proof = plan.proof()
    recovery_id = _sha256_bytes(_canonical_json_bytes(proof))
    path = (
        plan.rescue_spool
        / f"{plan.spool_base_name}.preserved-recovery-{recovery_id[:16]}.json"
    )
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RecoveryError(f"unsafe prior receipt: {path}")
        with path.open("rb") as handle:
            raw = handle.read(_RECEIPT_MAX_BYTES + 1)
        if len(raw) > _RECEIPT_MAX_BYTES:
            raise RecoveryError(f"oversized prior receipt: {path}")
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RecoveryError(f"invalid prior receipt: {path}") from exc
        if (
            not isinstance(value, dict)
            or value.get("schema") != SCHEMA
            or value.get("status") != "verified"
            or value.get("recovery_id") != recovery_id
            or value.get("proof") != proof
        ):
            raise RecoveryError(f"conflicting prior receipt: {path}")
    else:
        value = {
            "schema": SCHEMA,
            "status": "verified",
            "verified_at": _utc_now(),
            "recovery_id": recovery_id,
            "proof": proof,
        }
        _atomic_write_bytes(path, _canonical_json_bytes(value) + b"\n")
    return path, value, _sha256_file(path)


def _update_manifest(plan: RecoveryPlan, finished_at: str) -> None:
    path = plan.rescue_spool / "manifest.tsv"
    header = "\t".join(_MANIFEST_FIELDS)
    if path.exists():
        if path.is_symlink() or path.stat().st_size > _MANIFEST_MAX_BYTES:
            raise RecoveryError("unsafe or oversized rescue manifest")
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != header:
            raise RecoveryError("incompatible rescue manifest")
    else:
        lines = [header]
    repo, run_id, attempt = plan.identity
    values = (
        repo,
        str(run_id),
        str(attempt),
        str(plan.row["created_at"]),
        "zip",
        str(plan.archive.byte_size),
        plan.archive.sha256,
        finished_at,
    )
    rendered = "\t".join(values)
    prefix = f"{repo}\t{run_id}\t{attempt}\t"
    if any(line.startswith(prefix) and line != rendered for line in lines[1:]):
        raise RecoveryError("manifest already binds a different archive")
    if rendered not in lines[1:]:
        lines.append(rendered)
        _atomic_write_bytes(path, ("\n".join(lines) + "\n").encode())


def apply_plan(plan: RecoveryPlan) -> dict[str, object]:
    state = FetchStateEvidence(plan.state_path)
    try:
        state.connection.execute("BEGIN IMMEDIATE")
        state._assert_file_identity()
        current = state.connection.execute(
            "SELECT * FROM attempts WHERE repo=? AND run_id=? AND attempt=?",
            plan.identity,
        ).fetchone()
        if current is None or (
            str(current["status"]) not in _ELIGIBLE
            or _row_sha256(dict(current)) != plan.row_sha256
        ):
            raise RecoveryError("attempt changed; nothing was requeued")
        if _witness_digest(
            _witnesses(state.connection, plan.identity)
        ) != plan.witness_set_sha256:
            raise RecoveryError("member witnesses changed")
        if (
            plan.archive.path.stat().st_size != plan.archive.byte_size
            or _sha256_file(plan.archive.path) != plan.archive.sha256
        ):
            raise RecoveryError("archive changed before publication")

        receipt_path, receipt, receipt_sha = _receipt(plan)
        verified_at = str(receipt["verified_at"])
        _update_manifest(plan, verified_at)
        rescue_archive_existed = plan.rescue_archive.exists()
        JobLogRescueWorker._atomic_publish_file(
            plan.archive.path,
            plan.rescue_archive,
        )
        if (
            plan.rescue_archive.stat().st_size != plan.archive.byte_size
            or _sha256_file(plan.rescue_archive) != plan.archive.sha256
        ):
            if not rescue_archive_existed:
                plan.rescue_archive.unlink()
                _fsync_directory(plan.rescue_spool)
            raise RecoveryError("published rescue archive changed identity")
        audit = (
            f"recovery_id={receipt['recovery_id']} "
            f"receipt_sha256={receipt_sha} "
            f"source_row_sha256={plan.row_sha256}"
        )
        cursor = state.connection.execute(
            """
            UPDATE attempts SET
              status='retry',tries=0,
              archive_source='preserved-local-archive',
              archive_sha256=?,archive_size=?,
              terminal_http_status=NULL,terminal_body_sha256=NULL,
              error_class='PreservedArchiveRecovery',
              error_message=?,updated_at=?
            WHERE repo=? AND run_id=? AND attempt=? AND status=?
            """,
            (
                plan.archive.sha256,
                plan.archive.byte_size,
                audit,
                verified_at,
                *plan.identity,
                str(plan.row["status"]),
            ),
        )
        if cursor.rowcount != 1:
            raise RecoveryError("attempt was not requeued exactly once")
        state.connection.execute(
            """
            INSERT INTO request_ledger(
              requested_at,repo,run_id,attempt,endpoint,page_no,
              request_attempt,http_status,outcome,latency_ms,
              error_class,error_message
            ) VALUES (?,?,?,?,?,NULL,1,NULL,?,0,?,?)
            """,
            (
                verified_at,
                *plan.identity,
                "operator/preserved_archive_recovery",
                "operator/preserved_archive_recovery",
                "PreservedArchiveRecoveryReceipt",
                audit,
            ),
        )
        state.connection.execute("COMMIT")
        return {
            "identity": list(plan.identity),
            "status": "requeued",
            "archive_sha256": plan.archive.sha256,
            "archive_bytes": plan.archive.byte_size,
            "durable_members": len(plan.witnesses),
            "receipt": str(receipt_path),
            "receipt_sha256": receipt_sha,
            "rescue_archive": str(plan.rescue_archive),
        }
    except BaseException:
        if state.connection.in_transaction:
            state.connection.execute("ROLLBACK")
        raise
    finally:
        state.close()


def _target(value: str) -> tuple[str, int, int]:
    try:
        repo, run_id, attempt = value.rsplit(":", 2)
        result = repo, int(run_id), int(attempt)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target must be OWNER/REPO:RUN_ID:ATTEMPT"
        ) from exc
    if "/" not in repo or result[1] <= 0 or result[2] <= 0:
        raise argparse.ArgumentTypeError(
            "target must be OWNER/REPO:RUN_ID:ATTEMPT"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify preserved CI ZIPs and requeue exact attempts"
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--rescue-spool", required=True)
    parser.add_argument("--target", type=_target)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--max-archive-bytes", type=int, default=DEFAULT_MAX_ARCHIVE_BYTES
    )
    parser.add_argument(
        "--max-member-bytes", type=int, default=DEFAULT_MAX_MEMBER_BYTES
    )
    parser.add_argument(
        "--max-uncompressed-bytes",
        type=int,
        default=DEFAULT_MAX_UNCOMPRESSED_BYTES,
    )
    parser.add_argument("--max-members", type=int, default=DEFAULT_MAX_MEMBERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    limits = {
        "max_archive_bytes": args.max_archive_bytes,
        "max_member_bytes": args.max_member_bytes,
        "max_uncompressed_bytes": args.max_uncompressed_bytes,
        "max_members": args.max_members,
    }
    if any(value <= 0 for value in limits.values()):
        print("[preserved-recovery] ERROR: limits must be positive", file=sys.stderr)
        return 1
    try:
        plans = build_plans(
            state_path=Path(args.state),
            work_dir=Path(args.work_dir),
            rescue_spool=Path(args.rescue_spool),
            target=args.target,
            **limits,
        )
        result = {
            "schema": SCHEMA,
            "mode": "apply" if args.apply else "dry-run",
            "eligible_attempts": len(plans),
            "plans": [
                {
                    "identity": list(plan.identity),
                    "prior_status": str(plan.row["status"]),
                    "archive": str(plan.archive.path),
                    "archive_bytes": plan.archive.byte_size,
                    "archive_sha256": plan.archive.sha256,
                    "durable_members": len(plan.witnesses),
                    "durable_occurrence_tokens": sum(
                        int(item[5]) for item in plan.witnesses
                    ),
                    "rejected_candidates": list(plan.rejected_candidates),
                }
                for plan in plans
            ],
            "applied": (
                [apply_plan(plan) for plan in plans]
                if args.apply
                else []
            ),
        }
    except (
        OSError,
        RecoveryError,
        StateBindingError,
        sqlite3.Error,
        zipfile.BadZipFile,
    ) as exc:
        print(f"[preserved-recovery] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
