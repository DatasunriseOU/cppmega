#!/usr/bin/env python3
"""Archive, verify, and immutably publish a source-reducer smoke tree.

The source reducer intentionally publishes locally and atomically.  This
module is the transient-worker handoff: it accepts only a completed,
non-training-ready reducer tree, builds a deterministic ``tar.zst`` with a
complete member inventory, publishes the archive with create-only semantics,
downloads the exact generation again, and publishes a receipt last.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_sha256,
    gcs_join,
    load_json_object,
    require_exact_fields,
    require_sha256,
    run_checked,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.source_reducer import (  # noqa: E402
    SOURCE_REDUCER_RECEIPT_SCHEMA,
)
from scripts.distributed_data_prep.source_worker import (  # noqa: E402
    GcloudObjectStore,
    ObjectStore,
)

SOURCE_REDUCER_SMOKE_PUBLICATION_SCHEMA = (
    "cppmega.source_reducer_smoke_publication_v1"
)


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def inventory_tree(
    root: Path,
) -> tuple[list[dict[str, object]], dict[str, tuple[int, ...]]]:
    """Hash a symlink-free regular-file tree in canonical path order."""

    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"reducer root must be a regular directory: {root}")
    records: list[dict[str, object]] = []
    identities: dict[str, tuple[int, ...]] = {}
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ContractError(f"reducer tree contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError(f"reducer tree contains a special file: {relative}")
        before = _stat_identity(path)
        digest = sha256_file(path)
        after = _stat_identity(path)
        if before != after:
            raise ContractError(f"reducer artifact changed while hashing: {relative}")
        records.append(
            {"path": relative, "size_bytes": before[2], "sha256": digest}
        )
        identities[relative] = before
    if not records:
        raise ContractError("reducer tree is empty")
    return records, identities


def _validate_reducer_receipt(
    root: Path, records: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    by_path = {str(record["path"]): record for record in records}
    descriptor = by_path.get("reducer_receipt.json")
    if descriptor is None:
        raise ContractError("reducer tree has no reducer_receipt.json")
    raw, receipt = load_json_object(
        root / "reducer_receipt.json", where="source reducer receipt"
    )
    if (
        len(raw) != descriptor["size_bytes"]
        or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]
    ):
        raise ContractError("source reducer receipt inventory binding drifted")
    if (
        receipt.get("schema") != SOURCE_REDUCER_RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("training_ready") is not False
    ):
        raise ContractError("source reducer smoke receipt schema/status drifted")
    packing = receipt.get("packing")
    if not isinstance(packing, Mapping) or packing.get("executed") is not False:
        raise ContractError("source reducer smoke must not contain packed output")
    dedup = receipt.get("dedup")
    if not isinstance(dedup, Mapping):
        raise ContractError("source reducer receipt has no dedup descriptor")
    database_path = dedup.get("path")
    if not isinstance(database_path, str) or not database_path:
        raise ContractError("source reducer dedup path is invalid")
    database = by_path.get(database_path)
    if database is None:
        raise ContractError("source reducer dedup database is missing")
    if (
        database.get("size_bytes") != dedup.get("size_bytes")
        or database.get("sha256") != dedup.get("sha256")
    ):
        raise ContractError("source reducer dedup database binding drifted")
    if dedup.get("sidecars") != []:
        raise ContractError("source reducer receipt does not attest a sidecar-free database")
    unexpected_sidecars = [
        suffix
        for suffix in ("-shm", "-wal", "-journal")
        if f"{database_path}{suffix}" in by_path
    ]
    if unexpected_sidecars:
        raise ContractError(
            "source reducer tree contains unreceipted SQLite sidecars: "
            + ", ".join(unexpected_sidecars)
        )
    require_sha256(receipt.get("manifest_sha256"), where="logical manifest sha256")
    require_sha256(
        receipt.get("manifest_file_sha256"), where="manifest file sha256"
    )
    require_sha256(
        receipt.get("worker_receipts_sha256"), where="worker receipt set sha256"
    )
    return receipt


def _build_deterministic_archive(
    root: Path,
    records: Sequence[Mapping[str, object]],
    identities: Mapping[str, tuple[int, ...]],
    destination: Path,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tar = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tar", dir=destination.parent
    )
    os.close(descriptor)
    uncompressed = Path(raw_tar)
    try:
        with tarfile.open(
            uncompressed, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive:
            for record in records:
                relative = str(record["path"])
                path = root / relative
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = 0
                info.gid = 0
                info.uname = "root"
                info.gname = "root"
                info.mtime = 0
                info.mode = 0o640
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
                if _stat_identity(path) != identities[relative]:
                    raise ContractError(
                        f"reducer artifact changed while archiving: {relative}"
                    )
        with destination.open("xb") as output:
            completed = subprocess.run(
                ["zstd", "-19", "-T1", "--no-progress", "-c", "--", uncompressed],
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"zstd reducer archive failed: "
                f"{completed.stderr[-8000:].decode(errors='replace')}"
            )
    finally:
        uncompressed.unlink(missing_ok=True)
    run_checked(["zstd", "-t", "--", destination])
    return {
        "compression": "zstd",
        "level": 19,
        "threads": 1,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def publish_source_reducer_smoke(
    *,
    reducer_root: Path,
    scratch_root: Path,
    gcs_prefix: str,
    object_store: ObjectStore,
) -> tuple[dict[str, object], dict[str, object]]:
    """Publish a verified reducer smoke archive and then its immutable receipt."""

    prefix = validate_gcs_uri(gcs_prefix.rstrip("/"), where="publication prefix")
    records, identities = inventory_tree(reducer_root)
    reducer_receipt = _validate_reducer_receipt(reducer_root, records)
    artifact_set_sha256 = canonical_sha256(records)
    scratch_root.mkdir(parents=True, exist_ok=True)
    archive_path = scratch_root / f"{artifact_set_sha256}.tar.zst"
    archive_stage = scratch_root / (
        f".{artifact_set_sha256}.{os.getpid()}.tar.zst"
    )
    if archive_stage.exists():
        raise ContractError(f"publication stage already exists: {archive_stage}")
    archive = _build_deterministic_archive(
        reducer_root, records, identities, archive_stage
    )
    if archive_path.exists():
        if (
            archive_path.is_symlink()
            or not archive_path.is_file()
            or archive_path.stat().st_size != archive["size_bytes"]
            or sha256_file(archive_path) != archive["sha256"]
        ):
            raise ContractError(
                f"publication scratch archive collision: {archive_path}"
            )
        archive_stage.unlink()
    else:
        os.replace(archive_stage, archive_path)
    archive_uri = gcs_join(
        prefix,
        "reducer-artifacts",
        artifact_set_sha256,
        f"{archive['sha256']}.tar.zst",
    )
    published_archive = dict(object_store.publish_if_absent(archive_path, archive_uri))
    archive_generation = str(published_archive.get("generation", ""))
    with tempfile.TemporaryDirectory(
        prefix="source-reducer-publish-verify-", dir=scratch_root
    ) as raw_tmp:
        verified = Path(raw_tmp) / "archive.tar.zst"
        metadata = object_store.download(
            archive_uri, verified, generation=archive_generation
        )
        if (
            str(metadata.get("generation")) != archive_generation
            or verified.stat().st_size != archive["size_bytes"]
            or sha256_file(verified) != archive["sha256"]
        ):
            raise ContractError(
                "published reducer archive exact-generation check failed"
            )
        run_checked(["zstd", "-t", "--", verified])

    receipt: dict[str, object] = {
        "schema": SOURCE_REDUCER_SMOKE_PUBLICATION_SCHEMA,
        "status": "verified",
        "training_ready": False,
        "manifest_sha256": reducer_receipt["manifest_sha256"],
        "manifest_file_sha256": reducer_receipt["manifest_file_sha256"],
        "worker_receipts_sha256": reducer_receipt["worker_receipts_sha256"],
        "reducer_receipt_sha256": next(
            record["sha256"]
            for record in records
            if record["path"] == "reducer_receipt.json"
        ),
        "artifact_set_sha256": artifact_set_sha256,
        "members": records,
        "archive": {
            **published_archive,
            **archive,
            "uri": archive_uri,
        },
        "blocking_gates": reducer_receipt.get("blocking_gates"),
    }
    receipt_path = scratch_root / f"{artifact_set_sha256}.receipt.json"
    atomic_write_json(receipt_path, receipt)
    receipt_sha256 = sha256_file(receipt_path)
    receipt_uri = gcs_join(
        prefix,
        "reducer-receipts",
        artifact_set_sha256,
        f"{receipt_sha256}.receipt.json",
    )
    published_receipt = dict(
        object_store.publish_if_absent(receipt_path, receipt_uri)
    )
    with tempfile.TemporaryDirectory(
        prefix="source-reducer-receipt-verify-", dir=scratch_root
    ) as raw_tmp:
        verified_receipt = Path(raw_tmp) / "receipt.json"
        metadata = object_store.download(
            receipt_uri,
            verified_receipt,
            generation=str(published_receipt.get("generation", "")),
        )
        if (
            str(metadata.get("generation"))
            != str(published_receipt.get("generation", ""))
            or sha256_file(verified_receipt) != receipt_sha256
        ):
            raise ContractError(
                "published reducer receipt exact-generation check failed"
            )
    publication = {
        **published_receipt,
        "uri": receipt_uri,
        "sha256": receipt_sha256,
    }
    return receipt, publication


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reducer-root", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--gcs-prefix", required=True)
    args = parser.parse_args(argv)
    try:
        _receipt, publication = publish_source_reducer_smoke(
            reducer_root=args.reducer_root,
            scratch_root=args.scratch_root,
            gcs_prefix=args.gcs_prefix,
            object_store=GcloudObjectStore(),
        )
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"source reducer smoke publication failed: {exc}\n")
    print(publication["uri"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "SOURCE_REDUCER_SMOKE_PUBLICATION_SCHEMA",
    "inventory_tree",
    "publish_source_reducer_smoke",
]
