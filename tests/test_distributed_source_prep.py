from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import zlib
from pathlib import Path

import pytest

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    gcs_join,
    sha256_file,
)
from scripts.distributed_data_prep.source_manifest import (
    LOSSLESS_INDEX_MAX_TOKENS,
    PRE_GLOBAL_SCHEMA,
    build_source_manifest,
    repositories_for_worker,
    validate_source_manifest,
)
from scripts.distributed_data_prep.source_reducer import (
    load_worker_receipts,
    reduce_source_candidates,
)
from scripts.distributed_data_prep.source_quarantine_projection import (
    build_pinned_tree_quarantine_projection,
)
from scripts.distributed_data_prep.source_worker import (
    CANONICAL_DOCUMENT_ORDER,
    LocalObjectStore,
    _accept_known_git_fsck_diagnostic,
    _expected_git_fsck_exception_receipt,
    _KEYDB_ZERO_PADDED_FILEMODE,
    acquire_git_mirror,
    canonicalize_enriched_jsonl,
    compress_zstd,
    validate_git_fsck_snapshot,
)

_SHA = "a" * 64
_COMMIT = "b" * 40


def _repositories() -> list[dict[str, object]]:
    return [
        {
            "repo": "zeta",
            "project_id": "owner/zeta",
            "source": {
                "kind": "git_mirror",
                "remote_url": "https://github.com/owner/zeta.git",
                "expected_commit": _COMMIT,
                "expected_tree": None,
            },
        },
        {
            "repo": "private-source",
            "project_id": "corpus.local/private-source",
            "source": {
                "kind": "immutable_gcs_tar",
                "uri": "gs://source-snapshots/private-source.tar.zst",
                "generation": "42",
                "sha256": "c" * 64,
                "archive_format": "tar.zst",
                "strip_components": 1,
            },
        },
        {
            "repo": "alpha",
            "project_id": "owner/alpha",
            "source": {
                "kind": "git_mirror",
                "remote_url": "https://gitlab.com/owner/alpha.git",
                "expected_commit": "d" * 40,
                "expected_tree": "e" * 40,
            },
        },
    ]


def _manifest(repositories: list[dict[str, object]] | None = None):
    return build_source_manifest(
        repositories or _repositories(),
        worker_count=2,
        gcs_output_prefix="gs://cppmega-run/source-v1",
        code_revision="f" * 40,
        indexer_sha256=_SHA,
        tokenizer_sha256="1" * 64,
        quarantine_manifest_sha256="2" * 64,
    )


def test_source_manifest_is_deterministic_and_balanced() -> None:
    forward = _manifest(_repositories())
    reverse = _manifest(list(reversed(_repositories())))

    assert forward == reverse
    assert [job["project_id"] for job in forward["repositories"]] == [
        "corpus.local/private-source",
        "owner/alpha",
        "owner/zeta",
    ]
    assert [job["worker"] for job in forward["repositories"]] == [
        "worker-0000",
        "worker-0001",
        "worker-0000",
    ]
    assert len(repositories_for_worker(forward, "worker-0000")) == 2
    assert validate_source_manifest(forward) == forward


def test_source_manifest_rejects_fake_corpus_remote_and_tampering() -> None:
    repository = _repositories()[0]
    repository["source"] = {
        "kind": "git_mirror",
        "remote_url": "https://corpus.local/private.git",
        "expected_commit": _COMMIT,
        "expected_tree": None,
    }
    with pytest.raises(ContractError, match="immutable_gcs_tar"):
        _manifest([repository])

    manifest = _manifest()
    manifest["repositories"][0]["worker"] = "worker-0001"
    with pytest.raises(ContractError, match="digest|assignment"):
        validate_source_manifest(manifest)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_full_mirror_acquisition_pins_refs_tree_and_objects(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.name", "Source Test")
    _git(source, "config", "user.email", "source@example.test")
    (source / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-q", "-m", "fixture")
    commit = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")

    bare = tmp_path / "source.git"
    subprocess.run(
        ["git", "clone", "--bare", str(source), str(bare)],
        check=True,
        capture_output=True,
        text=True,
    )
    checkout, receipt = acquire_git_mirror(
        {
            "kind": "git_mirror",
            "remote_url": bare.as_uri(),
            "expected_commit": commit,
            "expected_tree": tree,
        },
        tmp_path / "scratch",
    )

    assert (checkout / "main.cpp").is_file()
    assert receipt["resolved_commit"] == commit
    assert receipt["tree"] == tree
    assert receipt["refs"]["count"] >= 1
    assert receipt["objects"]["count"] >= 3
    assert len(receipt["objects"]["inventory_sha256"]) == 64
    assert receipt["fsck"] == "ok"


def _write_loose_git_object(git_dir: Path, object_type: str, payload: bytes) -> str:
    serialized = f"{object_type} {len(payload)}\0".encode("ascii") + payload
    object_id = hashlib.sha1(serialized).hexdigest()
    destination = git_dir / "objects" / object_id[:2] / object_id[2:]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(zlib.compress(serialized))
    return object_id


def test_git_fsck_exception_accepts_only_exact_pinned_object_diagnostic(
    tmp_path: Path,
) -> None:
    mirror = tmp_path / "fixture.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", str(mirror)],
        check=True,
        capture_output=True,
        text=True,
    )
    blob_id = _write_loose_git_object(mirror, "blob", b"fixture\n")
    tree_payload = b"0100644 fixture.cpp\0" + bytes.fromhex(blob_id)
    tree_id = _write_loose_git_object(mirror, "tree", tree_payload)
    commit_payload = (
        f"tree {tree_id}\n"
        "author Fixture <fixture@example.test> 0 +0000\n"
        "committer Fixture <fixture@example.test> 0 +0000\n"
        "\nfixture\n"
    ).encode("ascii")
    commit_id = _write_loose_git_object(mirror, "commit", commit_payload)
    subprocess.run(
        ["git", f"--git-dir={mirror}", "update-ref", "refs/heads/main", commit_id],
        check=True,
        capture_output=True,
        text=True,
    )
    diagnostic = (
        f"error in tree {tree_id}: zeroPaddedFilemode: "
        "contains zero-padded file modes"
    )
    policy = {
        "remote_url": "https://example.test/exact.git",
        "expected_commit": commit_id,
        "checkout_tree": tree_id,
        "historical_commit": commit_id,
        "object_id": tree_id,
        "object_type": "tree",
        "object_size_bytes": len(tree_payload),
        "object_payload_sha256": hashlib.sha256(tree_payload).hexdigest(),
        "message_id": "zeroPaddedFilemode",
        "diagnostic": diagnostic,
        "returncode": 0,
    }
    source = {
        "kind": "git_mirror",
        "remote_url": policy["remote_url"],
        "expected_commit": commit_id,
        "expected_tree": tree_id,
    }
    fsck = subprocess.run(
        ["git", f"--git-dir={mirror}", "fsck", "--full", "--strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fsck.returncode != 0
    policy["returncode"] = fsck.returncode

    receipt = _accept_known_git_fsck_diagnostic(
        source,
        tree_id,
        mirror,
        fsck,
        known_exception=policy,
    )

    assert receipt == _expected_git_fsck_exception_receipt(policy)
    assert receipt["status"] == "accepted_known_historical_diagnostic"
    assert receipt["diagnostics"][0]["object_payload_sha256"] == hashlib.sha256(
        tree_payload
    ).hexdigest()

    extra_diagnostic = subprocess.CompletedProcess(
        fsck.args,
        fsck.returncode,
        fsck.stdout,
        fsck.stderr + "error in commit deadbeef: another failure\n",
    )
    with pytest.raises(ContractError, match="did not match the exact"):
        _accept_known_git_fsck_diagnostic(
            source,
            tree_id,
            mirror,
            extra_diagnostic,
            known_exception=policy,
        )

    wrong_payload_policy = dict(policy)
    wrong_payload_policy["object_payload_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="object payload drifted"):
        _accept_known_git_fsck_diagnostic(
            source,
            tree_id,
            mirror,
            fsck,
            known_exception=wrong_payload_policy,
        )


def test_git_fsck_exception_receipt_is_pinned_and_tamper_evident() -> None:
    policy = _KEYDB_ZERO_PADDED_FILEMODE
    source = {
        "kind": "git_mirror",
        "remote_url": policy["remote_url"],
        "expected_commit": policy["expected_commit"],
        "expected_tree": None,
    }
    snapshot = {
        "kind": "git_mirror",
        "remote_url": policy["remote_url"],
        "expected_commit": policy["expected_commit"],
        "resolved_commit": policy["expected_commit"],
        "tree": policy["checkout_tree"],
        "fsck": _expected_git_fsck_exception_receipt(policy),
    }

    validate_git_fsck_snapshot(source, snapshot)

    snapshot["tree"] = "0" * 40
    with pytest.raises(ContractError, match="checkout tree drifted"):
        validate_git_fsck_snapshot(source, snapshot)

    snapshot["tree"] = policy["checkout_tree"]
    snapshot["fsck"] = "ok"
    with pytest.raises(ContractError, match="omitted known fsck diagnostic"):
        validate_git_fsck_snapshot(source, snapshot)

    snapshot["fsck"] = _expected_git_fsck_exception_receipt(policy)
    snapshot["fsck"]["diagnostics"][0]["diagnostic"] += " tampered"
    with pytest.raises(ContractError, match="evidence drifted"):
        validate_git_fsck_snapshot(source, snapshot)


def test_candidate_canonicalization_is_independent_of_emission_order(
    tmp_path: Path,
) -> None:
    documents = [
        {"text": "z", "repo": "owner/project", "filepath": "z.cpp", "doc_type": "code"},
        {"doc_type": "code", "filepath": "a.cpp", "repo": "owner/project", "text": "a"},
        {"text": "a", "repo": "owner/project", "filepath": "a.cpp", "doc_type": "code"},
    ]
    outputs = []
    receipts = []
    for index, rows in enumerate((documents, list(reversed(documents)))):
        source = tmp_path / f"raw-{index}.jsonl"
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        output = tmp_path / f"canonical-{index}.jsonl"
        receipts.append(
            canonicalize_enriched_jsonl(
                source, output, project_id="owner/project", chunk_rows=1
            )
        )
        outputs.append(output.read_bytes())

    assert outputs[0] == outputs[1]
    assert receipts[0]["documents"] == 3
    assert receipts[0]["canonical_stream_sha256"] == receipts[1][
        "canonical_stream_sha256"
    ]
    assert outputs[0].count(b"\n") == 3


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class _SqliteExactDedup:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        assert self.connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        self.connection.execute("PRAGMA wal_autocheckpoint=0")
        for statement in (
            "CREATE TABLE exact (hash BLOB PRIMARY KEY)",
            "CREATE TABLE lsh (band_id INTEGER, band_hash BLOB, doc_id INTEGER)",
            "CREATE TABLE minhash (doc_id INTEGER PRIMARY KEY, sig BLOB)",
            "CREATE TABLE dedup_meta (key TEXT PRIMARY KEY, val INTEGER)",
            "CREATE TABLE chunk_claims (namespace TEXT, hash BLOB, claim_count INTEGER)",
            "CREATE TABLE dedup_stages (stage_id TEXT, created_at REAL, next_doc_id INTEGER)",
            "CREATE TABLE exact_stage (stage_id TEXT, hash BLOB)",
            "CREATE TABLE minhash_stage (stage_id TEXT, stage_doc_id INTEGER, sig BLOB)",
            "CREATE TABLE lsh_stage (stage_id TEXT, band_id INTEGER, band_hash BLOB, stage_doc_id INTEGER)",
            "CREATE TABLE chunk_claims_stage (stage_id TEXT, namespace TEXT, hash BLOB, claim_count INTEGER)",
        ):
            self.connection.execute(statement)
        self.connection.execute(
            "INSERT INTO dedup_meta(key,val) VALUES ('next_doc_id',0)"
        )
        self.connection.commit()

    def seen_exact_tokens(self, token_ids: list[int]) -> bool:
        digest = hashlib.sha1(bytes(token_ids)).digest()
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO exact(hash) VALUES (?)", (digest,)
        )
        return cursor.rowcount == 0

    def seen_near_tokens(self, token_ids: list[int]) -> bool:
        return False

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _reducer_fixture(tmp_path: Path, *, with_projection: bool = False):
    source_sha = "3" * 64
    base_quarantine_sha256 = "6" * 64
    source_root: Path | None = None
    base_quarantine: Path | None = None
    if with_projection:
        payload = (
            '<?xml version="1.0" encoding="utf-16"?>\r\n'
            "<license><name>fixture</name></license>\r\n"
        ).encode("utf-16")
        source_root = tmp_path / "projection-checkout"
        present = source_root / "sdk/present.cc"
        present.parent.mkdir(parents=True)
        present.write_bytes(payload)
        base_quarantine = tmp_path / "base-quarantine.json"
        entries = []
        for relative_path in ("sdk/present.cc", "sdk/removed.cc"):
            entries.append(
                {
                    "project_id": "owner/project",
                    "relative_path": relative_path,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "classification": "mislabeled_non_cpp",
                    "detected_format": "xml_utf16le",
                    "reason": "exact projection fixture",
                }
            )
        atomic_write_json(
            base_quarantine,
            {
                "schema": "cppmega.source_quarantine_manifest_v2",
                "entries": entries,
                "collections": [],
            },
        )
        base_quarantine_sha256 = sha256_file(base_quarantine)
        source = {
            "kind": "git_mirror",
            "remote_url": "https://github.com/owner/project.git",
            "expected_commit": "a" * 40,
            "expected_tree": "b" * 40,
        }
    else:
        source = {
            "kind": "immutable_gcs_tar",
            "uri": "gs://snapshots/project.tar.zst",
            "generation": "7",
            "sha256": source_sha,
            "archive_format": "tar.zst",
            "strip_components": 1,
        }
    repositories = [{"repo": "project", "project_id": "owner/project", "source": source}]
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}\n", encoding="utf-8")
    manifest = build_source_manifest(
        repositories,
        worker_count=1,
        gcs_output_prefix="gs://cppmega-run/reducer-test",
        code_revision="4" * 40,
        indexer_sha256="5" * 64,
        tokenizer_sha256=sha256_file(tokenizer),
        quarantine_manifest_sha256=base_quarantine_sha256,
    )
    manifest_file_sha = "7" * 64
    raw = tmp_path / "candidate.jsonl"
    rows = [
        {"doc_type": "code", "filepath": "a.cpp", "repo": "owner/project", "text": "same"},
        {"doc_type": "code", "filepath": "b.cpp", "repo": "owner/project", "text": "same"},
        {"doc_type": "code", "filepath": "c.cpp", "repo": "owner/project", "text": "unique"},
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    canonical = tmp_path / "canonical.jsonl"
    candidate = canonicalize_enriched_jsonl(
        raw, canonical, project_id="owner/project", chunk_rows=2
    )
    candidate["dedup_applied"] = False
    compressed = tmp_path / "candidate.jsonl.zst"
    compression = compress_zstd(canonical, compressed)
    store = LocalObjectStore(tmp_path / "objects")
    artifact_uri = gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-candidates",
        str(manifest["manifest_sha256"]),
        "00000-project",
        f"{compression['sha256']}.jsonl.zst",
    )
    published = dict(store.publish_if_absent(compressed, artifact_uri))
    if with_projection:
        assert source_root is not None
        assert base_quarantine is not None
        source_snapshot = {
            "kind": "git_mirror",
            "remote_url": source["remote_url"],
            "expected_commit": source["expected_commit"],
            "resolved_commit": source["expected_commit"],
            "tree": source["expected_tree"],
            "head_ref": "refs/heads/main",
            "head_commit": source["expected_commit"],
            "refs": {"count": 1, "sha256": "8" * 64},
            "objects": {
                "count": 3,
                "logical_bytes": 1,
                "types": {"blob": 1, "commit": 1, "tree": 1},
                "inventory_sha256": "9" * 64,
            },
            "gitlink_count": 0,
            "fsck": "ok",
        }
        projected_manifest = tmp_path / "projected-quarantine.json"
        projection_path = tmp_path / "source-quarantine-projection.json"
        projection = build_pinned_tree_quarantine_projection(
            base_manifest_path=base_quarantine,
            source_root=source_root,
            project_id="owner/project",
            source_snapshot=source_snapshot,
            projected_manifest_path=projected_manifest,
            receipt_path=projection_path,
        )
        effective_quarantine_sha256 = str(projection["projected_manifest"]["sha256"])
    else:
        source_snapshot = {
            "kind": "immutable_gcs_tar",
            "object": {
                "uri": repositories[0]["source"]["uri"],
                "generation": repositories[0]["source"]["generation"],
                "sha256": source_sha,
                "size_bytes": 1,
            },
            "archive_format": "tar.zst",
            "strip_components": 1,
            "file_count": 1,
            "extracted_bytes": 1,
        }
        projection_path = None
        projection = None
        effective_quarantine_sha256 = base_quarantine_sha256
    quarantine = {
        "schema": "cppmega.source_quarantine_receipt_v1",
        "project_id": "owner/project",
        "manifest_path": "/fixture/source_quarantine_manifest.json",
        "manifest_sha256": effective_quarantine_sha256,
        "manifest_entry_count": 0,
        "project_manifest_entry_count": 0,
        "candidate_count_before_quarantine": 3,
        "candidate_count_after_quarantine": 3,
        "quarantined_count": 0,
        "entries": [],
        "external_reference_omissions": {
            "schema": "cppmega.external_reference_omissions_v1",
            "status": "complete",
        },
        "parse_recovery": {
            "schema": "cppmega.source_parse_recovery_v1",
            "status": "complete",
        },
    }
    quarantine_path = tmp_path / "source-quarantine.json"
    atomic_write_json(quarantine_path, quarantine)
    quarantine_sha256 = sha256_file(quarantine_path)
    quarantine_uri = gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-quarantine-receipts",
        str(manifest["manifest_sha256"]),
        "00000-project",
        f"{quarantine_sha256}.quarantine.json",
    )
    published_quarantine = dict(
        store.publish_if_absent(quarantine_path, quarantine_uri)
    )
    receipt = {
        "schema": "cppmega.distributed_source_worker_receipt_v2",
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha,
        "assignment": {
            key: manifest["repositories"][0][key]
            for key in (
                "ordinal",
                "repo",
                "project_id",
                "worker",
                "assignment_sha256",
            )
        },
        "source_snapshot": source_snapshot,
        "candidate": candidate,
        "artifact": {
            **published,
            "sha256": compression["sha256"],
            "compression": compression,
        },
        "quarantine_artifact": {
            **published_quarantine,
            "sha256": quarantine_sha256,
        },
        "indexer": {
            "mode": "single_project_pre_global_enriched_v1",
            "project_id": "owner/project",
            "enriched": True,
            "max_tokens": LOSSLESS_INDEX_MAX_TOKENS,
            "parse_workers": 4,
            "memory_limit_gb": 14.0,
            "excluded_directories": ["__pycache__", "node_modules", "build", ".git"],
            "dedup_applied": False,
            "tokenizer_passed_to_indexer": False,
            "raw_output_sha256": "8" * 64,
            "quarantine_receipt_sha256": quarantine_sha256,
        },
        "training_ready": False,
    }
    if with_projection:
        assert projection_path is not None
        assert projection is not None
        projection_sha256 = sha256_file(projection_path)
        projection_uri = gcs_join(
            str(manifest["gcs_output_prefix"]),
            "source-quarantine-projections",
            str(manifest["manifest_sha256"]),
            "00000-project",
            f"{projection_sha256}.projection.json",
        )
        published_projection = dict(
            store.publish_if_absent(projection_path, projection_uri)
        )
        receipt["quarantine_projection_artifact"] = {
            **published_projection,
            "sha256": projection_sha256,
        }
    receipt_path = tmp_path / "worker-receipt.json"
    atomic_write_json(receipt_path, receipt)
    return manifest, manifest_file_sha, tokenizer, store, receipt_path


def test_reducer_requires_exact_receipt_coverage_and_dedups_before_pack(
    tmp_path: Path,
) -> None:
    manifest, manifest_file_sha, tokenizer, store, receipt_path = _reducer_fixture(
        tmp_path
    )

    with pytest.raises(ContractError, match="coverage"):
        load_worker_receipts(manifest, [])

    receipt = reduce_source_candidates(
        manifest,
        [receipt_path],
        manifest_file_sha256=manifest_file_sha,
        output_root=tmp_path / "reduced",
        scratch_root=tmp_path / "scratch",
        tokenizer_path=tokenizer,
        object_store=store,
        dedup_factory=_SqliteExactDedup,
        tokenizer_factory=lambda _path: _Tokenizer(),
        pack=False,
    )

    assert receipt["totals"] == {
        "candidate_documents": 3,
        "accepted_documents": 2,
        "dropped_exact": 1,
        "dropped_near": 0,
    }
    assert receipt["packing"]["executed"] is False
    assert receipt["training_ready"] is False
    assert receipt["dedup"]["checkpoint"] == {
        "mode": "TRUNCATE",
        "busy": 0,
        "log_frames": 0,
        "checkpointed_frames": 0,
        "wal_size_bytes": 0,
    }
    assert receipt["dedup"]["sidecars"] == []
    assert (tmp_path / "reduced" / "global_dedup.sqlite").is_file()
    assert not (tmp_path / "reduced" / "global_dedup.sqlite-wal").exists()
    assert not (tmp_path / "reduced" / "global_dedup.sqlite-shm").exists()
    accepted = next((tmp_path / "reduced" / "accepted").glob("*.jsonl.gz"))
    assert accepted.is_file()


def test_reducer_readbacks_and_binds_pinned_tree_quarantine_projection(
    tmp_path: Path,
) -> None:
    manifest, manifest_file_sha, tokenizer, store, receipt_path = _reducer_fixture(
        tmp_path,
        with_projection=True,
    )

    receipt = reduce_source_candidates(
        manifest,
        [receipt_path],
        manifest_file_sha256=manifest_file_sha,
        output_root=tmp_path / "reduced",
        scratch_root=tmp_path / "scratch",
        tokenizer_path=tokenizer,
        object_store=store,
        dedup_factory=_SqliteExactDedup,
        tokenizer_factory=lambda _path: _Tokenizer(),
        pack=False,
    )

    binding = receipt["worker_receipts"][0]
    assert binding["quarantine_projection_sha256"]
    assert binding["quarantine_projection_generation"] == "1"
    assert len(binding["quarantine_projection_summary_sha256"]) == 64


def test_worker_receipt_tampering_is_rejected(tmp_path: Path) -> None:
    manifest, _manifest_file_sha, _tokenizer, _store, receipt_path = _reducer_fixture(
        tmp_path
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["assignment"]["repo"] = "other"
    atomic_write_json(receipt_path, receipt)
    with pytest.raises(ContractError, match="assignment"):
        load_worker_receipts(manifest, [receipt_path])


def test_worker_receipt_cannot_escape_manifest_artifact_namespace(
    tmp_path: Path,
) -> None:
    manifest, _manifest_file_sha, _tokenizer, _store, receipt_path = _reducer_fixture(
        tmp_path
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact"]["uri"] = "gs://other-bucket/unbound.jsonl.zst"
    atomic_write_json(receipt_path, receipt)
    with pytest.raises(ContractError, match="manifest namespace"):
        load_worker_receipts(manifest, [receipt_path])
