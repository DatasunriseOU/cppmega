from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.distributed_data_prep._common import ContractError, atomic_write_json
from scripts.distributed_data_prep.publish_reducer_smoke import (
    SOURCE_REDUCER_SMOKE_PUBLICATION_SCHEMA,
    publish_source_reducer_smoke,
)
from scripts.distributed_data_prep.source_reducer import (
    SOURCE_REDUCER_RECEIPT_SCHEMA,
)
from scripts.distributed_data_prep.source_worker import LocalObjectStore


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reducer_tree(tmp_path: Path) -> Path:
    root = tmp_path / "reduced"
    accepted = root / "accepted"
    accepted.mkdir(parents=True)
    (accepted / "00000-project.jsonl.gz").write_bytes(b"gzip-fixture")
    database = root / "global_dedup.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE exact(hash BLOB PRIMARY KEY)")
    connection.commit()
    connection.close()
    database_bytes = database.read_bytes()
    receipt = {
        "schema": SOURCE_REDUCER_RECEIPT_SCHEMA,
        "status": "complete",
        "manifest_sha256": _sha("manifest"),
        "manifest_file_sha256": _sha("manifest-file"),
        "worker_receipts_sha256": _sha("worker-receipts"),
        "packing": {"executed": False},
        "dedup": {
            "path": database.name,
            "size_bytes": len(database_bytes),
            "sha256": hashlib.sha256(database_bytes).hexdigest(),
            "sidecars": [],
        },
        "training_ready": False,
        "blocking_gates": ["packed_sidecar_validation", "megatron_sealing"],
    }
    atomic_write_json(root / "reducer_receipt.json", receipt)
    return root


def test_reducer_smoke_publication_is_exact_and_idempotent(tmp_path: Path) -> None:
    root = _reducer_tree(tmp_path)
    store = LocalObjectStore(tmp_path / "objects")

    receipt, publication = publish_source_reducer_smoke(
        reducer_root=root,
        scratch_root=tmp_path / "scratch-a",
        gcs_prefix="gs://cppmega-run/source-pilot",
        object_store=store,
    )
    second, second_publication = publish_source_reducer_smoke(
        reducer_root=root,
        scratch_root=tmp_path / "scratch-a",
        gcs_prefix="gs://cppmega-run/source-pilot",
        object_store=store,
    )

    assert second == receipt
    assert second_publication == publication
    assert receipt["schema"] == SOURCE_REDUCER_SMOKE_PUBLICATION_SCHEMA
    assert receipt["status"] == "verified"
    assert receipt["training_ready"] is False
    assert [record["path"] for record in receipt["members"]] == [
        "accepted/00000-project.jsonl.gz",
        "global_dedup.sqlite",
        "reducer_receipt.json",
    ]
    assert publication["generation"] == "1"
    assert publication["uri"].endswith(f"/{publication['sha256']}.receipt.json")


def test_reducer_smoke_publication_rejects_packed_or_symlinked_tree(
    tmp_path: Path,
) -> None:
    root = _reducer_tree(tmp_path)
    receipt_path = root / "reducer_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["packing"]["executed"] = True
    atomic_write_json(receipt_path, receipt)
    with pytest.raises(ContractError, match="must not contain packed"):
        publish_source_reducer_smoke(
            reducer_root=root,
            scratch_root=tmp_path / "scratch-packed",
            gcs_prefix="gs://cppmega-run/source-pilot",
            object_store=LocalObjectStore(tmp_path / "objects-packed"),
        )

    receipt["packing"]["executed"] = False
    atomic_write_json(receipt_path, receipt)
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ContractError, match="symlink"):
        publish_source_reducer_smoke(
            reducer_root=root,
            scratch_root=tmp_path / "scratch-symlink",
            gcs_prefix="gs://cppmega-run/source-pilot",
            object_store=LocalObjectStore(tmp_path / "objects-symlink"),
        )


def test_reducer_smoke_publication_rejects_unreceipted_sqlite_sidecars(
    tmp_path: Path,
) -> None:
    root = _reducer_tree(tmp_path)
    (root / "global_dedup.sqlite-wal").write_bytes(b"")

    with pytest.raises(ContractError, match="unreceipted SQLite sidecars"):
        publish_source_reducer_smoke(
            reducer_root=root,
            scratch_root=tmp_path / "scratch-sidecar",
            gcs_prefix="gs://cppmega-run/source-pilot",
            object_store=LocalObjectStore(tmp_path / "objects-sidecar"),
        )
