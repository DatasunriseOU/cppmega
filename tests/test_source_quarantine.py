from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from tools.clang_indexer import index_project as ip
from tools.clang_indexer.source_quarantine import (
    LEGACY_MANIFEST_SCHEMA,
    MANIFEST_SCHEMA,
    RECEIPT_SCHEMA,
    ProjectSourceQuarantine,
    SourceQuarantineError,
)

PROJECT_ID = "fixture/source-quarantine"
RELATIVE_XML = "sdk/license.cc"
RELATIVE_CRASH_FIXTURE = "tools/clang/test/Parser/crash-report.c"
RELATIVE_INDEX_CRASH_FIXTURE = "tools/clang/test/Index/crash-recovery.c"
RELATIVE_INDEX_REMAP_CRASH_FIXTURE = (
    "tools/clang/test/Index/Inputs/crash-recovery-code-complete-remap.c"
)
RELATIVE_PARSER_CRASH_FIXTURE = (
    "external/bsd/llvm/dist/clang/test/Driver/crash report spaces.c"
)
RELATIVE_CERTIFICATE_PAIR = "vectors/certpairs/reverseCertificatePair.cp"
CERTIFICATE_PAIR_PREFIX = "vectors/certpairs/"
RELATIVE_GENERATED_BLOB = "ports_module/example_build/module_code.c"
RELATIVE_EXECUTABLE_ARCHIVE = "bin/self-executing-tool"
RELATIVE_CLICKHOUSE_BINARY_SQL = "tests/queries/0_stateless/binary_fixture.sql"
RELATIVE_GCC_PR119001 = "gcc/testsuite/gcc.dg/pr119001-1.c"


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


def _clang_index_remap_crash_fixture_bytes() -> bytes:
    return (
        b"// RUN: echo env CINDEXTEST_EDITING=1 \\\n"
        b"// RUN:   not c-index-test -test-load-source-reparse 1 local \\\n"
        b'// RUN:   -remap-file="%s,%S/Inputs/crash-recovery-code-complete-remap.c" \\\n'
        b"// RUN:   %s 2> %t.err\n"
        b"// RUN: FileCheck < %t.err -check-prefix=CHECK-CODE-COMPLETE-CRASH %s\n"
        b"// CHECK-CODE-COMPLETE-CRASH: Unable to reparse translation unit\n"
        b"\n"
        b"#warning parsing original file\n"
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


def _gcc_pr119001_fixture_bytes() -> bytes:
    return (
        b"/* PR c/119001 */\n"
        b"/* { dg-do run } */\n"
        b"/* { dg-options \"\" } */\n\n"
        b"union U { char a[]; int i; };\n"
        b"union U u = { \"12345\" };\n"
        b"union U v = { .a = \"6789\" };\n"
        b"union U w = { { 1, 2, 3, 4, 5, 6 } };\n"
        b"union U x = { .a = { 7, 8, 9 } };\n"
        b"union V { int i; char a[]; };\n"
        b"union V y = { .a = \"abcdefghijk\" };\n"
        b"union V z = { .a = { 10, 11, 12, 13, 14, 15, 16, 17 } };\n\n"
        b"int\nmain ()\n{\n"
        b"  for (int i = 0; i < 6; ++i)\n"
        b"    if (u.a[i] != \"12345\"[i])\n      __builtin_abort ();\n"
        b"  for (int i = 0; i < 5; ++i)\n"
        b"    if (v.a[i] != \"6789\"[i])\n      __builtin_abort ();\n"
        b"  for (int i = 0; i < 6; ++i)\n"
        b"    if (w.a[i] != i + 1)\n      __builtin_abort ();\n"
        b"  for (int i = 0; i < 3; ++i)\n"
        b"    if (x.a[i] != i + 7)\n      __builtin_abort ();\n"
        b"  for (int i = 0; i < 12; ++i)\n"
        b"    if (y.a[i] != \"abcdefghijk\"[i])\n      __builtin_abort ();\n"
        b"  for (int i = 0; i < 8; ++i)\n"
        b"    if (z.a[i] != i + 10)\n      __builtin_abort ();\n"
        b"}\n"
    )


def _der(tag: int, payload: bytes) -> bytes:
    if len(payload) < 0x80:
        length = bytes([len(payload)])
    else:
        encoded = len(payload).to_bytes((len(payload).bit_length() + 7) // 8, "big")
        length = bytes([0x80 | len(encoded)]) + encoded
    return bytes([tag]) + length + payload


def _certificate_pair_bytes(*, wrapper_tag: int = 0xA1) -> bytes:
    certificate = _der(
        0x30,
        _der(0x30, b"\x02\x01\x01")
        + _der(0x30, b"\x06\x03\x2a\x03\x04")
        + _der(0x03, b"\x00\x01"),
    )
    return _der(0x30, _der(wrapper_tag, certificate))


def _mixed_utf8_utf16le_c_array_bytes(*, byte_count: int = 1024) -> bytes:
    prefix = (
        b"/* Copyright (c) 2026 Eclipse ThreadX contributors */\n"
        b"/* SPDX-License-Identifier: MIT */\n\n"
    )
    byte_literals = ", ".join(f"0x{value % 256:02X}" for value in range(byte_count))
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
    return b'#!/bin/sh\nexec java -jar "$0" "$@"\nexit 1\n' + archive_buffer.getvalue()


def _clickhouse_binary_sql_bytes(
    *,
    input_format: str = "Native",
    server_error: str = "TOO_LARGE_ARRAY_SIZE",
) -> bytes:
    return (
        b"-- It correctly throws a high-level exception:\n"
        b"SELECT * FROM format("
        + input_format.encode("ascii")
        + b", 'value UInt64',\n$$\x00\xffbinary-protocol-fixture$$); -- { "
        b"serverError " + server_error.encode("ascii") + b" }\n"
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
                "collections": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_collection_manifest(
    path: Path,
    payloads: dict[str, bytes],
    *,
    expected_file_count: int | None = None,
    content_set_sha256: str | None = None,
) -> None:
    rows = [
        [relative_path, len(payload), hashlib.sha256(payload).hexdigest()]
        for relative_path, payload in sorted(payloads.items())
    ]
    digest = hashlib.sha256(
        json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "entries": [],
                "collections": [
                    {
                        "project_id": PROJECT_ID,
                        "relative_path_prefix": CERTIFICATE_PAIR_PREFIX,
                        "relative_path_suffix": ".cp",
                        "expected_file_count": (
                            expected_file_count
                            if expected_file_count is not None
                            else len(payloads)
                        ),
                        "content_set_sha256": content_set_sha256 or digest,
                        "classification": "mislabeled_non_cpp",
                        "detected_format": "asn1_der_x509_certificate_pair",
                        "reason": "DER certificate-pair fixtures stored under .cp",
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


def test_legacy_point_manifest_remains_supported(tmp_path: Path) -> None:
    payload = _xml_bytes()
    candidate = tmp_path / RELATIVE_XML
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(manifest, payload)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["schema"] = LEGACY_MANIFEST_SCHEMA
    del raw["collections"]
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1


@pytest.mark.parametrize(
    ("relative_path", "payload"),
    [
        (RELATIVE_CRASH_FIXTURE, _clang_crash_fixture_bytes()),
        (RELATIVE_INDEX_CRASH_FIXTURE, _clang_index_crash_fixture_bytes()),
        (
            RELATIVE_INDEX_REMAP_CRASH_FIXTURE,
            _clang_index_remap_crash_fixture_bytes(),
        ),
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
    assert receipt["entries"][0]["detected_format"] == ("clang_debug_crash_pragma")


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


def test_exact_quarantine_filters_gcc_pr119001_regression_fixture(
    tmp_path: Path,
) -> None:
    payload = _gcc_pr119001_fixture_bytes()
    candidate = tmp_path / RELATIVE_GCC_PR119001
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="compiler_regression_fixture",
        detected_format="gcc_c_flexible_array_union_initializer_regression",
        relative_path=RELATIVE_GCC_PR119001,
        reason="GCC PR119001 crashes the pinned libclang parser",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["classification"] == (
        "compiler_regression_fixture"
    )
    assert receipt["entries"][0]["detected_format"] == (
        "gcc_c_flexible_array_union_initializer_regression"
    )


def test_gcc_pr119001_quarantine_rejects_unrelated_flexible_array_source(
    tmp_path: Path,
) -> None:
    payload = _gcc_pr119001_fixture_bytes().replace(
        b"/* PR c/119001 */",
        b"/* unrelated  */",
    )
    candidate = tmp_path / RELATIVE_GCC_PR119001
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="compiler_regression_fixture",
        detected_format="gcc_c_flexible_array_union_initializer_regression",
        relative_path=RELATIVE_GCC_PR119001,
        reason="forged GCC regression fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="contract is incomplete"):
        policy.filter_candidates(tmp_path, [str(candidate)])


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


def test_exact_collection_quarantine_filters_complete_der_set(
    tmp_path: Path,
) -> None:
    payloads = {
        f"{CERTIFICATE_PAIR_PREFIX}forward.cp": _certificate_pair_bytes(),
        f"{CERTIFICATE_PAIR_PREFIX}reverse.cp": _certificate_pair_bytes(
            wrapper_tag=0xA0
        ),
    }
    candidates = []
    for relative_path, payload in payloads.items():
        candidate = tmp_path / relative_path
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(payload)
        candidates.append(str(candidate))
    manifest = tmp_path / "quarantine.json"
    _write_collection_manifest(manifest, payloads)

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, candidates)

    assert kept == []
    assert receipt["project_manifest_entry_count"] == 1
    assert receipt["quarantined_count"] == 2
    assert [entry["relative_path"] for entry in receipt["entries"]] == sorted(payloads)


def test_collection_quarantine_rejects_incomplete_or_drifted_set(
    tmp_path: Path,
) -> None:
    relative_path = f"{CERTIFICATE_PAIR_PREFIX}forward.cp"
    payload = _certificate_pair_bytes()
    candidate = tmp_path / relative_path
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_collection_manifest(
        manifest,
        {relative_path: payload},
        expected_file_count=2,
    )
    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="count mismatch"):
        policy.filter_candidates(tmp_path, [str(candidate)])

    _write_collection_manifest(
        manifest,
        {relative_path: payload},
        content_set_sha256="0" * 64,
    )
    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="content-set SHA-256 mismatch"):
        policy.filter_candidates(tmp_path, [str(candidate)])


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
    assert receipt["entries"][0]["detected_format"] == ("posix_shell_appended_zip")


@pytest.mark.parametrize(
    ("input_format", "server_error"),
    [
        ("Native", "TOO_LARGE_ARRAY_SIZE"),
        ("BSONEachRow", "INCORRECT_DATA"),
        (
            "BSONEachRow",
            "INCORRECT_DATA, UNKNOWN_TYPE, CANNOT_READ_ALL_DATA",
        ),
    ],
)
def test_exact_quarantine_filters_clickhouse_binary_sql_fixture(
    tmp_path: Path,
    input_format: str,
    server_error: str,
) -> None:
    payload = _clickhouse_binary_sql_bytes(
        input_format=input_format,
        server_error=server_error,
    )
    candidate = tmp_path / RELATIVE_CLICKHOUSE_BINARY_SQL
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="binary_protocol_test_fixture",
        detected_format="clickhouse_dollar_quoted_binary_sql",
        relative_path=RELATIVE_CLICKHOUSE_BINARY_SQL,
        reason="ClickHouse binary protocol exception fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["classification"] == ("binary_protocol_test_fixture")
    assert receipt["entries"][0]["detected_format"] == (
        "clickhouse_dollar_quoted_binary_sql"
    )


def test_clickhouse_binary_sql_quarantine_rejects_plain_text_fixture(
    tmp_path: Path,
) -> None:
    payload = (
        b"SELECT * FROM format(Native, 'value UInt64',\n"
        b"$$plain text$$); -- { serverError TOO_LARGE_ARRAY_SIZE }\n"
    )
    candidate = tmp_path / RELATIVE_CLICKHOUSE_BINARY_SQL
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="binary_protocol_test_fixture",
        detected_format="clickhouse_dollar_quoted_binary_sql",
        relative_path=RELATIVE_CLICKHOUSE_BINARY_SQL,
        reason="forged ClickHouse binary protocol fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="contract is incomplete"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_clickhouse_binary_sql_quarantine_rejects_mismatched_error_list(
    tmp_path: Path,
) -> None:
    payload = _clickhouse_binary_sql_bytes(
        input_format="Native",
        server_error="TOO_LARGE_ARRAY_SIZE, CANNOT_READ_ALL_DATA",
    )
    candidate = tmp_path / RELATIVE_CLICKHOUSE_BINARY_SQL
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="binary_protocol_test_fixture",
        detected_format="clickhouse_dollar_quoted_binary_sql",
        relative_path=RELATIVE_CLICKHOUSE_BINARY_SQL,
        reason="forged ClickHouse error-list fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="expected server error disagree"):
        policy.filter_candidates(tmp_path, [str(candidate)])


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
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        item
        for item in manifest["entries"]
        if item["project_id"] in {"google/filament", "microsoft/DirectXShaderCompiler"}
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
    fixtures = {
        RELATIVE_INDEX_CRASH_FIXTURE: _clang_index_crash_fixture_bytes(),
        RELATIVE_INDEX_REMAP_CRASH_FIXTURE: _clang_index_remap_crash_fixture_bytes(),
    }
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = {
        relative_path: next(
            item
            for item in manifest["entries"]
            if item["project_id"] == "google/filament"
            and item["relative_path"].endswith(relative_path)
        )
        for relative_path in fixtures
    }

    assert len(fixtures[RELATIVE_INDEX_CRASH_FIXTURE]) == 344
    assert (
        hashlib.sha256(fixtures[RELATIVE_INDEX_CRASH_FIXTURE]).hexdigest()
        == "1dae510e0b173890f77aa3ef905b892614b3b5c7a98add3df7b58a555ccef727"
    )
    assert len(fixtures[RELATIVE_INDEX_REMAP_CRASH_FIXTURE]) == 398
    assert (
        hashlib.sha256(fixtures[RELATIVE_INDEX_REMAP_CRASH_FIXTURE]).hexdigest()
        == "4170335b0ad9450e204fcf9625e6d7f506f84308b10857b7b57eb37973b66590"
    )
    assert set(entries) == set(fixtures)
    for relative_path, payload in fixtures.items():
        entry = entries[relative_path]
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["classification"] == "deliberate_compiler_crash_fixture"
        assert entry["detected_format"] == "clang_debug_crash_pragma"


def test_checked_in_minix_parser_crash_manifest_matches_archive_member() -> None:
    payload = _clang_parser_crash_fixture_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
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


def test_checked_in_gcc_pr119001_manifest_matches_pinned_fixture() -> None:
    payload = _gcc_pr119001_fixture_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "gcc-mirror/gcc"
    )

    assert len(payload) == 867
    assert hashlib.sha256(payload).hexdigest() == (
        "a01d63621d40ce04f9d95341d6a3931d38da7168eace62765970d8f4f382c178"
    )
    assert entry["relative_path"] == RELATIVE_GCC_PR119001
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "compiler_regression_fixture"
    assert entry["detected_format"] == (
        "gcc_c_flexible_array_union_initializer_regression"
    )


def test_checked_in_xemu_certificate_pair_collection_matches_archive_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    collection = next(
        item
        for item in manifest["collections"]
        if item["project_id"] == "xemu-project/xemu"
    )

    assert collection["relative_path_prefix"] == (
        "roms/edk2/CryptoPkg/Library/OpensslLib/openssl/pyca-cryptography/"
        "vectors/cryptography_vectors/x509/PKITS_data/certpairs/"
    )
    assert collection["relative_path_suffix"] == ".cp"
    assert collection["expected_file_count"] == 348
    assert collection["content_set_sha256"] == (
        "4d92e2254cef41f0a84525e6e30a1d6fcda5237d0b878522ed911cfa973c6ef7"
    )
    assert collection["classification"] == "mislabeled_non_cpp"
    assert collection["detected_format"] == "asn1_der_x509_certificate_pair"


def test_checked_in_threadx_generated_blob_manifest_matches_upstream_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
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


def test_checked_in_clickhouse_binary_sql_manifest_matches_diagnosis_receipts() -> None:
    expected = {
        "tests/queries/0_stateless/02683_native_too_large_size.sql": (
            5742,
            "0dd70078b534e164c86d82c27c926573745c186970189914d886aab6aa0259ea",
        ),
        "tests/queries/0_stateless/02684_bson.sql": (
            9071,
            "52a2ea38b7ee657f6ad8b7c4ce4ca9f652ec032dc5af0415852c6d515b858137",
        ),
        "tests/queries/0_stateless/02685_bson2.sql": (
            21329,
            "e6be67fb4b042a5b0845243d3790db2230483cfeb2fd072491415aa34a106507",
        ),
        "tests/queries/0_stateless/02686_bson3.sql": (
            21327,
            "7b1b4bd8e2f641baf789ddd854f89c5c1583c5240999f494360f27fc9473e90d",
        ),
        "tests/queries/0_stateless/02687_native_fuzz.sql": (
            630,
            "e96d08c5033a3409725549d8c0909dc40b4f94a687fdd3b1e390b810db691e2c",
        ),
    }
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = {
        item["relative_path"]: item
        for item in manifest["entries"]
        if item["project_id"] == "ClickHouse/ClickHouse"
    }

    assert set(entries) == set(expected)
    for relative_path, (size_bytes, sha256) in expected.items():
        entry = entries[relative_path]
        assert entry["size_bytes"] == size_bytes
        assert entry["sha256"] == sha256
        assert entry["classification"] == "binary_protocol_test_fixture"
        assert entry["detected_format"] == "clickhouse_dollar_quoted_binary_sql"


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
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
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
    assert (
        receipt["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
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


def test_process_project_quarantines_clickhouse_binary_sql_before_domain_discovery(
    tmp_path: Path,
) -> None:
    payload = _clickhouse_binary_sql_bytes()
    candidate = tmp_path / RELATIVE_CLICKHOUSE_BINARY_SQL
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    receipt_path = tmp_path / "receipts/source.json"
    _write_manifest(
        manifest,
        payload,
        classification="binary_protocol_test_fixture",
        detected_format="clickhouse_dollar_quoted_binary_sql",
        relative_path=RELATIVE_CLICKHOUSE_BINARY_SQL,
        reason="ClickHouse binary protocol exception fixture",
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
    assert receipt["entries"][0]["relative_path"] == RELATIVE_CLICKHOUSE_BINARY_SQL
