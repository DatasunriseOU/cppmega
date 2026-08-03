from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from tools.clang_indexer import index_project as ip
from tools.clang_indexer.source_quarantine import (
    MANIFEST_SCHEMA,
    RECEIPT_SCHEMA,
    ProjectSourceQuarantine,
    SourceQuarantineError,
)

PROJECT_ID = "fixture/source-quarantine"
RELATIVE_XML = "sdk/license.cc"
RELATIVE_CRASH_FIXTURE = "tools/clang/test/Parser/crash-report.c"
RELATIVE_INDEX_CRASH_FIXTURE = "tools/clang/test/Index/crash-recovery.c"
RELATIVE_PARSER_CRASH_FIXTURE = (
    "external/bsd/llvm/dist/clang/test/Driver/crash report spaces.c"
)
RELATIVE_CERTIFICATE_PAIR = "vectors/certpairs/reverseCertificatePair.cp"
RELATIVE_GENERATED_BLOB = "ports_module/example_build/module_code.c"
RELATIVE_EXECUTABLE_ARCHIVE = "bin/self-executing-tool"


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


def _clang_index_crash_fixture_bytes() -> bytes:
    return (
        b"// RUN: not c-index-test -test-load-source all %s 2> %t.err\n"
        b"// RUN: FileCheck < %t.err -check-prefix=CHECK-LOAD-SOURCE-CRASH %s\n"
        b"// CHECK-LOAD-SOURCE-CRASH: Unable to load translation unit\n"
        b"// RUN: env LIBCLANG_DISABLE_CRASH_RECOVERY=1 not --crash "
        b"c-index-test -test-load-source all %s\n"
        b"//\n"
        b"// REQUIRES: crash-recovery\n"
        b"\n"
        b"#pragma clang __debug crash\n"
    )


def _clang_parser_crash_fixture_bytes() -> bytes:
    return (
        b'// RUN: rm -rf "%t"\n'
        b'// RUN: mkdir "%t"\n'
        b'// RUN: not env TMPDIR="%t" TEMP="%t" TMP="%t" '
        b'RC_DEBUG_OPTIONS=1 %clang -fsyntax-only "%s" 2>&1 | FileCheck "%s"\n'
        b'// RUN: cat "%t/crash report spaces"-*.c | '
        b'FileCheck --check-prefix=CHECKSRC "%s"\n'
        b'// RUN: cat "%t/crash report spaces"-*.sh | '
        b'FileCheck --check-prefix=CHECKSH "%s"\n'
        b"// REQUIRES: crash-recovery\n"
        b"\n"
        b"// because of the glob (*.c, *.sh)\n"
        b"// REQUIRES: shell\n"
        b"\n"
        b"#pragma clang __debug parser_crash\n"
        b"// CHECK: Preprocessed source(s) and associated run script(s) are located at:\n"
        b"// CHECK-NEXT: note: diagnostic msg: {{.*}}.c\n"
        b"FOO\n"
        b"// CHECKSRC: FOO\n"
        b'// CHECKSH: "-cc1"\n'
        b'// CHECKSH: "-main-file-name" "crash report spaces.c"\n'
        b'// CHECKSH: "crash report spaces-{{[^ ]*}}.c"\n'
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


def _mixed_utf8_utf16le_c_array_bytes(*, byte_count: int = 1024) -> bytes:
    prefix = (
        "/* Copyright (c) 2026 Eclipse ThreadX contributors */\n"
        "/* SPDX-License-Identifier: MIT */\n\n"
    ).encode()
    byte_literals = ", ".join(
        f"0x{value % 256:02X}" for value in range(byte_count)
    )
    generated = (
        "/* \n\n"
        "   Input ELF file: sample_threadx_module.axf\n\n"
        "   Output C Array file: module_code.c\n\n"
        "*/\n\n"
        "__align(4096) unsigned char  module_code[] = {\n"
        "/* Address  Contents */\n"
        f"/* 0x00000000 */ {byte_literals}}};\n"
    ).encode("utf-16le")
    return prefix + generated


def _self_executing_zip_bytes() -> bytes:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr("payload.txt", "exact fixture payload\n")
    return (
        b"#!/bin/sh\n"
        b'exec java -jar "$0" "$@"\n'
        b"exit 1\n"
        + archive_buffer.getvalue()
    )


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


@pytest.mark.parametrize(
    ("relative_path", "payload"),
    [
        (RELATIVE_CRASH_FIXTURE, _clang_crash_fixture_bytes()),
        (RELATIVE_INDEX_CRASH_FIXTURE, _clang_index_crash_fixture_bytes()),
    ],
)
def test_exact_quarantine_filters_deliberate_clang_crash_fixture(
    tmp_path: Path,
    relative_path: str,
    payload: bytes,
) -> None:
    candidate = tmp_path / relative_path
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_crash_fixture",
        detected_format="clang_debug_crash_pragma",
        relative_path=relative_path,
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


def test_exact_quarantine_filters_deliberate_clang_parser_crash_fixture(
    tmp_path: Path,
) -> None:
    payload = _clang_parser_crash_fixture_bytes()
    candidate = tmp_path / RELATIVE_PARSER_CRASH_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_crash_fixture",
        detected_format="clang_debug_parser_crash_pragma",
        relative_path=RELATIVE_PARSER_CRASH_FIXTURE,
        reason="fixture deliberately crashes the Clang parser",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["detected_format"] == (
        "clang_debug_parser_crash_pragma"
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


def test_exact_quarantine_filters_mixed_utf16_generated_binary_blob(
    tmp_path: Path,
) -> None:
    payload = _mixed_utf8_utf16le_c_array_bytes()
    candidate = tmp_path / RELATIVE_GENERATED_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_binary_blob",
        detected_format="mixed_utf8_utf16le_c_array",
        relative_path=RELATIVE_GENERATED_BLOB,
        reason="generated binary blob fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["classification"] == "generated_binary_blob"


def test_exact_quarantine_filters_self_executing_zip(
    tmp_path: Path,
) -> None:
    payload = _self_executing_zip_bytes()
    candidate = tmp_path / RELATIVE_EXECUTABLE_ARCHIVE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_executable_archive",
        detected_format="posix_shell_appended_zip",
        relative_path=RELATIVE_EXECUTABLE_ARCHIVE,
        reason="self-executing archive fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["detected_format"] == (
        "posix_shell_appended_zip"
    )


def test_executable_archive_quarantine_rejects_invalid_zip(
    tmp_path: Path,
) -> None:
    payload = b'#!/bin/sh\nexec java -jar "$0" "$@"\nPK\x03\x04not-a-zip\n'
    candidate = tmp_path / RELATIVE_EXECUTABLE_ARCHIVE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_executable_archive",
        detected_format="posix_shell_appended_zip",
        relative_path=RELATIVE_EXECUTABLE_ARCHIVE,
        reason="forged self-executing archive",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="appended ZIP is invalid"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_generated_binary_blob_quarantine_rejects_small_c_array(
    tmp_path: Path,
) -> None:
    payload = _mixed_utf8_utf16le_c_array_bytes(byte_count=16)
    candidate = tmp_path / RELATIVE_GENERATED_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_binary_blob",
        detected_format="mixed_utf8_utf16le_c_array",
        relative_path=RELATIVE_GENERATED_BLOB,
        reason="forged generated binary blob",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="contract is incomplete"):
        policy.filter_candidates(tmp_path, [str(candidate)])


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
    entries = [
        item
        for item in manifest["entries"]
        if item["project_id"]
        in {"google/filament", "microsoft/DirectXShaderCompiler"}
        and item["relative_path"].endswith(RELATIVE_CRASH_FIXTURE)
    ]

    assert len(payload) == 271
    assert {entry["project_id"] for entry in entries} == {
        "google/filament",
        "microsoft/DirectXShaderCompiler",
    }
    for entry in entries:
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["classification"] == "deliberate_compiler_crash_fixture"
        assert entry["detected_format"] == "clang_debug_crash_pragma"


def test_checked_in_filament_index_crash_manifest_matches_reference_fixture() -> None:
    payload = _clang_index_crash_fixture_bytes()
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
        and item["relative_path"].endswith(RELATIVE_INDEX_CRASH_FIXTURE)
    )

    assert len(payload) == 344
    assert hashlib.sha256(payload).hexdigest() == (
        "1dae510e0b173890f77aa3ef905b892614b3b5c7a98add3df7b58a555ccef727"
    )
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "deliberate_compiler_crash_fixture"
    assert entry["detected_format"] == "clang_debug_crash_pragma"


def test_checked_in_minix_parser_crash_manifest_matches_archive_member() -> None:
    payload = _clang_parser_crash_fixture_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "Stichting-MINIX-Research-Foundation/minix"
    )

    assert len(payload) == 700
    assert hashlib.sha256(payload).hexdigest() == (
        "e970a5aab931388aa671bf2589426acfcd2ebdcf60c70342cfd500086d697131"
    )
    assert entry["relative_path"] == RELATIVE_PARSER_CRASH_FIXTURE
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "deliberate_compiler_crash_fixture"
    assert entry["detected_format"] == "clang_debug_parser_crash_pragma"


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


def test_checked_in_threadx_generated_blob_manifest_matches_upstream_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "eclipse-threadx/threadx"
    )

    assert entry["size_bytes"] == 61551
    assert entry["sha256"] == (
        "2d49edeeb4233af4972ac4f9cec96b171d92ffad0738eaf3b4dcd536a05e9294"
    )
    assert entry["classification"] == "generated_binary_blob"
    assert entry["detected_format"] == "mixed_utf8_utf16le_c_array"


@pytest.mark.parametrize(
    ("project_id", "relative_path", "size_bytes", "sha256"),
    [
        (
            "python/cpython",
            "Lib/test/archivetestdata/exe_with_z64",
            978,
            "b1a8382acacce4022b02daa25b293ddfc1dc6ce6a3ddb8b3d95b517592c5a428",
        ),
        (
            "questdb/questdb",
            "core/src/main/bin/linux-x86-64/jfrconv",
            125326,
            "1806e97395e39bd37386eb3ef4ad0a2e83fe9a70b6a05e6779d542684342def4",
        ),
    ],
)
def test_checked_in_executable_archive_manifest_matches_archive_receipt(
    project_id: str,
    relative_path: str,
    size_bytes: int,
    sha256: str,
) -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item for item in manifest["entries"] if item["project_id"] == project_id
    )

    assert entry["relative_path"] == relative_path
    assert entry["size_bytes"] == size_bytes
    assert entry["sha256"] == sha256
    assert entry["classification"] == "generated_executable_archive"
    assert entry["detected_format"] == "posix_shell_appended_zip"


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
    with pytest.raises(
        SourceQuarantineError,
        match="were not discovered as source candidates",
    ):
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


def test_process_project_quarantines_non_cpp_executable_archive(
    tmp_path: Path,
) -> None:
    payload = _self_executing_zip_bytes()
    candidate = tmp_path / RELATIVE_EXECUTABLE_ARCHIVE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    receipt_path = tmp_path / "receipts/source.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_executable_archive",
        detected_format="posix_shell_appended_zip",
        relative_path=RELATIVE_EXECUTABLE_ARCHIVE,
        reason="self-executing archive fixture",
    )

    documents = ip.process_project(
        str(tmp_path),
        enriched=True,
        project_id=PROJECT_ID,
        source_quarantine_manifest=str(manifest),
        source_quarantine_receipt=str(receipt_path),
    )

    assert documents == []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["relative_path"] == RELATIVE_EXECUTABLE_ARCHIVE
