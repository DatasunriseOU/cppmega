from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.distributed_data_prep._common import sha256_file
from scripts.distributed_data_prep.source_manifest import build_source_manifest
from scripts.distributed_data_prep.source_quarantine_projection import (
    SourceQuarantineProjectionError,
    build_pinned_tree_quarantine_projection,
    validate_pinned_tree_quarantine_projection,
)
from scripts.distributed_data_prep import source_worker as source_worker_module
from scripts.distributed_data_prep.source_worker import (
    LocalObjectStore,
    PINNED_TREE_PROJECTION_MODE,
    run_source_worker,
)
from tools.clang_indexer.source_quarantine import SourceQuarantineError


PROJECT_ID = "acme/project"


def _snapshot() -> dict[str, object]:
    return {
        "kind": "git_mirror",
        "remote_url": "https://github.com/acme/project.git",
        "expected_commit": "a" * 40,
        "resolved_commit": "a" * 40,
        "tree": "b" * 40,
    }


def _xml_bytes() -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-16"?>\r\n'
        "<license><name>not C++</name></license>\r\n"
    ).encode("utf-16")


def _der(tag: int, payload: bytes) -> bytes:
    if len(payload) < 0x80:
        length = bytes([len(payload)])
    else:
        encoded = len(payload).to_bytes((len(payload).bit_length() + 7) // 8, "big")
        length = bytes([0x80 | len(encoded)]) + encoded
    return bytes([tag]) + length + payload


def _certificate_pair_bytes() -> bytes:
    certificate = _der(
        0x30,
        _der(0x30, b"\x02\x01\x01")
        + _der(0x30, b"\x06\x03\x2a\x03\x04")
        + _der(0x03, b"\x00\x01"),
    )
    return _der(0x30, _der(0xA1, certificate))


def _write_manifest(
    path: Path,
    *,
    entries: list[dict[str, object]],
    collections: list[dict[str, object]] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "cppmega.source_quarantine_manifest_v2",
                "entries": entries,
                "collections": collections or [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _xml_entry(relative_path: str, payload: bytes) -> dict[str, object]:
    return {
        "project_id": PROJECT_ID,
        "relative_path": relative_path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "classification": "mislabeled_non_cpp",
        "detected_format": "xml_utf16le",
        "reason": "exact non-C++ fixture",
    }


def test_projection_omits_only_absent_rules_and_binds_pinned_tree(tmp_path: Path) -> None:
    source_root = tmp_path / "checkout"
    source_root.mkdir()
    payload = _xml_bytes()
    present = source_root / "sdk/present.cc"
    present.parent.mkdir()
    present.write_bytes(payload)
    manifest = tmp_path / "base.json"
    _write_manifest(
        manifest,
        entries=[
            _xml_entry("sdk/present.cc", payload),
            _xml_entry("sdk/removed.cc", payload),
            {
                **_xml_entry("sdk/other.cc", payload),
                "project_id": "other/project",
            },
        ],
    )
    projected = tmp_path / "projected.json"
    receipt_path = tmp_path / "projection.json"

    receipt = build_pinned_tree_quarantine_projection(
        base_manifest_path=manifest,
        source_root=source_root,
        project_id=PROJECT_ID,
        source_snapshot=_snapshot(),
        projected_manifest_path=projected,
        receipt_path=receipt_path,
    )

    derived = json.loads(projected.read_text(encoding="utf-8"))
    assert [entry["relative_path"] for entry in derived["entries"]] == [
        "sdk/present.cc"
    ]
    assert receipt["selection"] == {
        "included_entry_paths": ["sdk/present.cc"],
        "omitted_entry_paths": ["sdk/removed.cc"],
        "included_collections": [],
        "omitted_collections": [],
    }
    assert receipt["projected_manifest"]["sha256"] == hashlib.sha256(
        projected.read_bytes()
    ).hexdigest()
    assert validate_pinned_tree_quarantine_projection(
        json.loads(receipt_path.read_text(encoding="utf-8")),
        project_id=PROJECT_ID,
        base_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_snapshot=_snapshot(),
        projected_manifest_path=projected,
    ) == receipt


def test_projection_retains_and_verifies_exact_collection(tmp_path: Path) -> None:
    source_root = tmp_path / "checkout"
    payloads = {
        "vectors/certs/one.cp": _certificate_pair_bytes(),
        "vectors/certs/two.cp": _certificate_pair_bytes(),
    }
    rows = [
        [path, len(payload), hashlib.sha256(payload).hexdigest()]
        for path, payload in sorted(payloads.items())
    ]
    for relative_path, payload in payloads.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = tmp_path / "base.json"
    _write_manifest(
        manifest,
        entries=[],
        collections=[
            {
                "project_id": PROJECT_ID,
                "relative_path_prefix": "vectors/certs/",
                "relative_path_suffix": ".cp",
                "expected_file_count": len(payloads),
                "content_set_sha256": hashlib.sha256(
                    json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode(
                        "ascii"
                    )
                ).hexdigest(),
                "classification": "mislabeled_non_cpp",
                "detected_format": "asn1_der_x509_certificate_pair",
                "reason": "exact certificate pair collection",
            }
        ],
    )

    receipt = build_pinned_tree_quarantine_projection(
        base_manifest_path=manifest,
        source_root=source_root,
        project_id=PROJECT_ID,
        source_snapshot=_snapshot(),
        projected_manifest_path=tmp_path / "projected.json",
        receipt_path=tmp_path / "projection.json",
    )

    assert receipt["selection"]["included_collections"] == [
        {
            "relative_path_prefix": "vectors/certs/",
            "relative_path_suffix": ".cp",
        }
    ]
    assert receipt["selection"]["omitted_collections"] == []


def test_projection_rejects_present_rule_with_changed_bytes(tmp_path: Path) -> None:
    source_root = tmp_path / "checkout"
    path = source_root / "sdk/present.cc"
    path.parent.mkdir(parents=True)
    path.write_bytes(_xml_bytes() + b"changed")
    expected = _xml_bytes()
    manifest = tmp_path / "base.json"
    _write_manifest(manifest, entries=[_xml_entry("sdk/present.cc", expected)])

    with pytest.raises(SourceQuarantineError, match="size mismatch"):
        build_pinned_tree_quarantine_projection(
            base_manifest_path=manifest,
            source_root=source_root,
            project_id=PROJECT_ID,
            source_snapshot=_snapshot(),
            projected_manifest_path=tmp_path / "projected.json",
            receipt_path=tmp_path / "projection.json",
        )


def test_projection_validation_rejects_source_or_logical_digest_drift(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "checkout"
    source_root.mkdir()
    payload = _xml_bytes()
    path = source_root / "sdk/present.cc"
    path.parent.mkdir()
    path.write_bytes(payload)
    manifest = tmp_path / "base.json"
    _write_manifest(manifest, entries=[_xml_entry("sdk/present.cc", payload)])
    projected = tmp_path / "projected.json"
    receipt = build_pinned_tree_quarantine_projection(
        base_manifest_path=manifest,
        source_root=source_root,
        project_id=PROJECT_ID,
        source_snapshot=_snapshot(),
        projected_manifest_path=projected,
        receipt_path=tmp_path / "projection.json",
    )

    changed_snapshot = _snapshot()
    changed_snapshot["tree"] = "c" * 40
    with pytest.raises(SourceQuarantineProjectionError, match="source binding"):
        validate_pinned_tree_quarantine_projection(
            receipt,
            project_id=PROJECT_ID,
            base_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            source_snapshot=changed_snapshot,
            projected_manifest_path=projected,
        )

    tampered = dict(receipt)
    tampered["projection_sha256"] = "0" * 64
    with pytest.raises(SourceQuarantineProjectionError, match="logical digest"):
        validate_pinned_tree_quarantine_projection(
            tampered,
            project_id=PROJECT_ID,
            base_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            source_snapshot=_snapshot(),
            projected_manifest_path=projected,
        )


def test_worker_publishes_projected_quarantine_and_resumes_from_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise projection creation, immutable publish/readback, and resume."""

    repo_root = tmp_path / "pipeline"
    indexer = repo_root / "tools/clang_indexer/index_project.py"
    tokenizer = repo_root / "cppmega/tokenizer/tokenizer.json"
    quarantine = repo_root / "configs/source_quarantine_manifest.json"
    for path, contents in (
        (indexer, "# fake indexer; the test replaces its invocation\n"),
        (tokenizer, "{}\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    source_root = tmp_path / "checkout"
    payload = _xml_bytes()
    present = source_root / "sdk/present.cc"
    present.parent.mkdir(parents=True)
    present.write_bytes(payload)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        quarantine,
        entries=[
            _xml_entry("sdk/present.cc", payload),
            _xml_entry("sdk/removed.cc", payload),
        ],
    )
    subprocess.run(["git", "init", "-q", str(repo_root)], check=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.name", "Projection Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.email", "test@example.test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo_root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    manifest = build_source_manifest(
        [
            {
                "repo": "project",
                "project_id": PROJECT_ID,
                "source": {
                    "kind": "git_mirror",
                    "remote_url": "https://github.com/acme/project.git",
                    "expected_commit": "a" * 40,
                    "expected_tree": None,
                },
            }
        ],
        worker_count=1,
        gcs_output_prefix="gs://projection-test/run",
        code_revision=revision,
        indexer_sha256=sha256_file(indexer),
        tokenizer_sha256=sha256_file(tokenizer),
        quarantine_manifest_sha256=sha256_file(quarantine),
    )
    job = manifest["repositories"][0]
    assert isinstance(job, dict)
    source_snapshot = {
        **_snapshot(),
        "head_ref": "refs/heads/main",
        "head_commit": "a" * 40,
        "refs": {"count": 1, "sha256": "c" * 64},
        "objects": {
            "count": 3,
            "logical_bytes": 1,
            "types": {"blob": 1, "commit": 1, "tree": 1},
            "inventory_sha256": "d" * 64,
        },
        "gitlink_count": 0,
        "fsck": "ok",
    }
    monkeypatch.setattr(
        source_worker_module,
        "acquire_git_mirror",
        lambda _source, _scratch: (source_root, source_snapshot),
    )
    invocations: list[Path] = []

    def fake_run_indexer(
        *,
        python: Path,
        indexer: Path,
        source_root: Path,
        project_id: str,
        raw_output: Path,
        quarantine_manifest: Path,
        quarantine_receipt: Path,
        parse_workers: int,
        memory_limit_gb: float,
        max_tokens: int,
    ) -> dict[str, object]:
        del python, indexer, source_root
        invocations.append(quarantine_manifest)
        derived = json.loads(quarantine_manifest.read_text(encoding="utf-8"))
        assert [entry["relative_path"] for entry in derived["entries"]] == [
            "sdk/present.cc"
        ]
        raw_output.write_text(
            json.dumps(
                {
                    "repo": project_id,
                    "filepath": "source.cpp",
                    "doc_type": "code",
                    "text": "int projected_source;\n",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        quarantine_receipt.write_text(
            json.dumps(
                {
                    "schema": "cppmega.source_quarantine_receipt_v1",
                    "project_id": project_id,
                    "manifest_path": str(quarantine_manifest),
                    "manifest_sha256": sha256_file(quarantine_manifest),
                    "manifest_entry_count": 1,
                    "project_manifest_entry_count": 1,
                    "candidate_count_before_quarantine": 1,
                    "candidate_count_after_quarantine": 1,
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
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "mode": "single_project_pre_global_enriched_v1",
            "project_id": project_id,
            "enriched": True,
            "max_tokens": max_tokens,
            "parse_workers": parse_workers,
            "memory_limit_gb": memory_limit_gb,
            "excluded_directories": ["__pycache__", "node_modules", "build", ".git"],
            "dedup_applied": False,
            "tokenizer_passed_to_indexer": False,
            "raw_output_sha256": sha256_file(raw_output),
            "quarantine_receipt_sha256": sha256_file(quarantine_receipt),
        }

    monkeypatch.setattr(source_worker_module, "_run_indexer", fake_run_indexer)
    store = LocalObjectStore(tmp_path / "objects")
    kwargs = {
        "manifest_file_sha256": "e" * 64,
        "worker": str(job["worker"]),
        "scratch_root": tmp_path / "scratch",
        "receipt_root": tmp_path / "receipts",
        "repo_root": repo_root,
        "python": Path(sys.executable),
        "indexer": indexer,
        "tokenizer": tokenizer,
        "quarantine_manifest": quarantine,
        "object_store": store,
        "quarantine_projection_mode": PINNED_TREE_PROJECTION_MODE,
    }
    receipts = run_source_worker(manifest, **kwargs)
    assert len(receipts) == 1
    receipt = receipts[0]
    artifact = receipt["quarantine_projection_artifact"]
    assert isinstance(artifact, dict)
    downloaded = tmp_path / "projection.readback.json"
    store.download(str(artifact["uri"]), downloaded, generation=str(artifact["generation"]))
    projection = json.loads(downloaded.read_text(encoding="utf-8"))
    assert projection["selection"]["omitted_entry_paths"] == ["sdk/removed.cc"]
    assert validate_pinned_tree_quarantine_projection(
        projection,
        project_id=PROJECT_ID,
        base_manifest_sha256=sha256_file(quarantine),
        source_snapshot=source_snapshot,
    ) == projection
    assert len(invocations) == 1

    resumed = run_source_worker(manifest, **kwargs)
    assert resumed == receipts
    assert len(invocations) == 1
