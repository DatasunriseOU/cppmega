from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.clang_indexer import index_project as ip
from tools.clang_indexer.source_quarantine import (
    MANIFEST_SCHEMA,
    ProjectSourceQuarantine,
    RECEIPT_SCHEMA,
    SourceQuarantineError,
)


PROJECT_ID = "fixture/source-quarantine"
RELATIVE_XML = "sdk/license.cc"
RELATIVE_CRASH_FIXTURE = "tools/clang/test/Parser/crash-report.c"
RELATIVE_CERTIFICATE_PAIR = "vectors/certpairs/reverseCertificatePair.cp"


def _xml_bytes() -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-16"?>\r\n'
        "<license><name>not C++</name></license>\r\n"
    ).encode("utf-16")


def _clang_crash_fixture_bytes() -> bytes:
    return (
        b"// RUN: not --crash %clang_cc1 %s 2>&1 | FileCheck %s\n"
        b"// REQUIRES: crash-recovery\n"
        b"\n"
        b"// FIXME: CHECKs might be incompatible to win32.\n"
        b"// Stack traces also require back traces.\n"
        b"// REQUIRES: shell, backtrace\n"
        b"\n"
        b"#prag\\\n"
        b"ma clang __debug crash\n"
        b"\n"
        b"// CHECK: prag\\\n"
        b"// CHECK-NEXT: ma\n"
        b"\n"
    )


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
    payload: bytes,
    *,
    sha256: str | None = None,
    classification: str = "mislabeled_non_cpp",
    detected_format: str = "xml_utf16le",
    relative_path: str = RELATIVE_XML,
    reason: str = "fixture XML stored under a .cc suffix",
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "entries": [
                    {
                        "project_id": PROJECT_ID,
                        "relative_path": relative_path,
                        "size_bytes": len(payload),
                        "sha256": sha256 or hashlib.sha256(payload).hexdigest(),
                        "classification": classification,
                        "detected_format": detected_format,
                        "reason": reason,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_cpp_discovery_preserves_large_and_nonproduction_source_trees(
    tmp_path: Path,
) -> None:
    fixtures = {
        "src/main.cpp": b"int main() { return 0; }\n",
        "tests/test_main.cpp": b"void test_main() {}\n",
        "third_party/vendor.hpp": b"#pragma once\n",
        "examples/demo.cc": b"void demo() {}\n",
        "docs/snippet.cxx": b"void documented() {}\n",
        "fuzzing/fuzz.cpp": b"void fuzz() {}\n",
        "src/large.cpp": b"//" + b"x" * 500_001,
    }
    for relative_path, payload in fixtures.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    discovered = {
        Path(path).relative_to(tmp_path).as_posix()
        for path in ip.find_cpp_files(str(tmp_path))
    }
    assert discovered == set(fixtures)

    explicitly_filtered = {
        Path(path).relative_to(tmp_path).as_posix()
        for path in ip.find_cpp_files(
            str(tmp_path),
            extra_exclude_dirs={"third_party"},
        )
    }
    assert explicitly_filtered == set(fixtures) - {"third_party/vendor.hpp"}


def test_exact_quarantine_filters_verified_non_cpp_and_builds_receipt(
    tmp_path: Path,
) -> None:
    payload = _xml_bytes()
    candidate = tmp_path / RELATIVE_XML
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    code = tmp_path / "src/main.cpp"
    code.parent.mkdir(parents=True)
    code.write_text("int main() { return 0; }\n", encoding="utf-8")
    manifest = tmp_path / "quarantine.json"
    _write_manifest(manifest, payload)

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    candidates = ip.find_cpp_files(str(tmp_path))
    kept, receipt = policy.filter_candidates(tmp_path, candidates)

    assert kept == [str(code)]
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["candidate_count_before_quarantine"] == 2
    assert receipt["candidate_count_after_quarantine"] == 1
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"] == [
        {
            "project_id": PROJECT_ID,
            "relative_path": RELATIVE_XML,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "classification": "mislabeled_non_cpp",
            "detected_format": "xml_utf16le",
            "reason": "fixture XML stored under a .cc suffix",
        }
    ]


def test_quarantine_hash_mismatch_fails_without_filtering(
    tmp_path: Path,
) -> None:
    payload = _xml_bytes()
    candidate = tmp_path / RELATIVE_XML
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(manifest, payload, sha256="0" * 64)

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="SHA-256 mismatch"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_exact_quarantine_filters_deliberate_clang_crash_fixture(
    tmp_path: Path,
) -> None:
    payload = _clang_crash_fixture_bytes()
    candidate = tmp_path / RELATIVE_CRASH_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_crash_fixture",
        detected_format="clang_debug_crash_pragma",
        relative_path=RELATIVE_CRASH_FIXTURE,
        reason="fixture deliberately crashes Clang",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["classification"] == (
        "deliberate_compiler_crash_fixture"
    )
    assert receipt["entries"][0]["detected_format"] == (
        "clang_debug_crash_pragma"
    )


def test_exact_quarantine_filters_der_x509_certificate_pair(
    tmp_path: Path,
) -> None:
    payload = _certificate_pair_bytes()
    candidate = tmp_path / RELATIVE_CERTIFICATE_PAIR
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="asn1_der_x509_certificate_pair",
        relative_path=RELATIVE_CERTIFICATE_PAIR,
        reason="DER certificate-pair fixture stored under a .cp suffix",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["detected_format"] == (
        "asn1_der_x509_certificate_pair"
    )


def test_certificate_pair_quarantine_rejects_non_certificate_der(
    tmp_path: Path,
) -> None:
    payload = _der(0x30, _der(0xA1, _der(0x30, _der(0x02, b"\x01"))))
    candidate = tmp_path / RELATIVE_CERTIFICATE_PAIR
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="asn1_der_x509_certificate_pair",
        relative_path=RELATIVE_CERTIFICATE_PAIR,
        reason="forged certificate pair",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="field layout"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_checked_in_clang_crash_manifest_matches_reference_fixture() -> None:
    payload = _clang_crash_fixture_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "google/filament"
    )

    assert len(payload) == 271
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "deliberate_compiler_crash_fixture"
    assert entry["detected_format"] == "clang_debug_crash_pragma"


def test_checked_in_xemu_certificate_pair_manifest_matches_archive_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "xemu-project/xemu"
    )

    assert entry["size_bytes"] == 955
    assert entry["sha256"] == (
        "8734808c3859f30101cb1934bf7d71d153430dd85ae357d41c1641fc7a8addfe"
    )
    assert entry["classification"] == "mislabeled_non_cpp"
    assert entry["detected_format"] == "asn1_der_x509_certificate_pair"


def test_clang_crash_quarantine_requires_independent_fixture_signature(
    tmp_path: Path,
) -> None:
    payload = b"int main(void) { return 0; }\n"
    candidate = tmp_path / RELATIVE_CRASH_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_crash_fixture",
        detected_format="clang_debug_crash_pragma",
        relative_path=RELATIVE_CRASH_FIXTURE,
        reason="forged crash fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(
        SourceQuarantineError,
        match="crash-test contract is incomplete",
    ):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_quarantine_entry_must_be_discovered_and_cannot_hide_parse_errors(
    tmp_path: Path,
) -> None:
    payload = _xml_bytes()
    manifest = tmp_path / "quarantine.json"
    _write_manifest(manifest, payload)
    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="were not discovered"):
        policy.filter_candidates(tmp_path, [])

    _write_manifest(
        manifest,
        payload,
        classification="parse_error",
        detected_format="xml_utf16le",
    )
    with pytest.raises(SourceQuarantineError, match="unsupported quarantine"):
        ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)


def test_process_project_writes_atomic_bound_receipt(
    tmp_path: Path,
) -> None:
    payload = _xml_bytes()
    candidate = tmp_path / RELATIVE_XML
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    receipt_path = tmp_path / "receipts/source.json"
    _write_manifest(manifest, payload)

    documents = ip.process_project(
        str(tmp_path),
        enriched=True,
        project_id=PROJECT_ID,
        source_quarantine_manifest=str(manifest),
        source_quarantine_receipt=str(receipt_path),
    )

    assert documents == []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["project_id"] == PROJECT_ID
    assert receipt["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert receipt["quarantined_count"] == 1
    omission_receipt = receipt["external_reference_omissions"]
    assert omission_receipt["schema"] == "cppmega.external_reference_omissions_v1"
    assert omission_receipt["status"] == "complete"
    assert omission_receipt["reason"] == "unknown_external_provider"
    assert omission_receipt["observation_count"] == 0
    assert omission_receipt["unique_reference_count"] == 0
    assert omission_receipt["location_count"] == 0
    assert omission_receipt["locations"] == []
