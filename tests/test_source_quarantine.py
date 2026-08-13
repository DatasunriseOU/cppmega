from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.distributed_data_prep._common import ContractError
from scripts.distributed_data_prep.source_worker import (
    _validate_source_tree_entry_exclusions,
    validate_quarantine_receipt_file,
)
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
RELATIVE_NUL_DIAGNOSTIC_FIXTURE = "clang/test/Misc/diag-null-bytes-in-line.cpp"
RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE = "clang/test/Lexer/newline-nul.c"
RELATIVE_NUL_IN_LITERAL_FIXTURE = "clang/test/Lexer/null-character-in-literal.c"
RELATIVE_CERTIFICATE_PAIR = "vectors/certpairs/reverseCertificatePair.cp"
CERTIFICATE_PAIR_PREFIX = "vectors/certpairs/"
RELATIVE_GENERATED_BLOB = "ports_module/example_build/module_code.c"
RELATIVE_EXECUTABLE_ARCHIVE = "bin/self-executing-tool"
RELATIVE_CLICKHOUSE_BINARY_SQL = "tests/queries/0_stateless/binary_fixture.sql"
RELATIVE_GCC_PR119001 = "gcc/testsuite/gcc.dg/pr119001-1.c"
RELATIVE_PLUMHALL_D412 = (
    "xbox_leak_may_2020/xbox trunk/xbox/private/test/crttests/test/"
    "conformance/c_plumhall/D412.c"
)
RELATIVE_NUL_FF_BLOB = "unknown_version_2/Source/drivers/spb/spbcx/sys/driver.h"
RELATIVE_TRUNCATED_UTF32BE_BOM = "Tests/RunCMake/Syntax/Broken-BOM-UTF-32-BE.cmake"
RELATIVE_TRUNCATED_UTF32LE_BOM = "Tests/RunCMake/Syntax/Broken-BOM-UTF-32-LE.cmake"
RELATIVE_CMAKE_NULL_AFTER_BACKSLASH = (
    "Tests/RunCMake/Syntax/NullAfterBackslash.cmake"
)
RELATIVE_CMAKE_NULL_TERMINATED_ARGUMENT = (
    "Tests/RunCMake/Syntax/NullTerminatedArgument.cmake"
)
RELATIVE_BIG5_SHELL_HEREDOC = (
    "external/gpl2/gettext/dist/gettext-tools/tests/msgconv-1"
)
RELATIVE_AUTOTOOLS_BIG5_SHELL_HEREDOC = "gettext-tools/tests/msgconv-1"
RELATIVE_GIT_SHORTLOG_INVALID_UTF8 = "t/t4201-shortlog.sh"


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


def _clang_embedded_nul_diagnostic_bytes() -> bytes:
    return (
        b"// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | "
        b"FileCheck -strict-whitespace %s\n"
        b"\n"
        b"int x[sizeof\0int];\n"
        b"// CHECK: warning: null character ignored\n"
        b"// CHECK-NEXT: int x[sizeof<U+0000>int];\n"
        b"// CHECK-NEXT:             ^\n"
        b"\n"
        b"// CHECK: error: expected parentheses around type name in "
        b"sizeof expression\n"
        b"// CHECK-NEXT: int x[sizeof<U+0000>int];\n"
        b"// CHECK-NEXT:             ^\n"
        b"// CHECK-NEXT:             (          )\n"
    )


def _clang_newline_nul_diagnostic_bytes() -> bytes:
    return (
        b"// RUN: %clang_cc1 -E %s -verify\n"
        b"\n"
        b"// We used to crash if a line continuation was followed by a nul byte "
        b"within a\n"
        b"// preprocessing directive.\n"
        b"# if 1 \\\n"
        b"\0#if something_else // expected-warning {{null character ignored}} "
        b"expected-error {{not a valid binary operator}}\n"
        b"#error error\n"
        b"#endif\n"
        b"#endif // expected-error {{#endif without #if}}\n"
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


def _utf16le_generated_c_array_bytes(*, byte_count: int = 1024) -> bytes:
    byte_literals = ", ".join(f"0x{value % 256:02X}" for value in range(byte_count))
    generated = (
        "/* \r\n"
        "   Input ELF file: sample_threadx_module.axf\r\n"
        "   Output C Array file: module_code.c\r\n"
        "*/\r\n\r\n"
        "__align(4096) unsigned char  module_code[] = {\r\n"
        "/* Address  Contents */\r\n"
        f"/* 0x00000000 */ {byte_literals}}};\r\n"
    ).encode("utf-16le")
    return b"\xff\xfe" + generated


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


def _truncated_utf32be_bom_bytes() -> bytes:
    return b"\x00\x00\xfe"


def _truncated_utf32le_bom_bytes() -> bytes:
    return b"\xff\xfe\x00"


def _cmake_null_after_backslash_bytes() -> bytes:
    return b"A(" + (b"A" * 52) + b"\\\0\n(" + (b"A" * 54) + b"\n"


def _cmake_null_terminated_argument_bytes() -> bytes:
    return (
        b"LIST(APPEND foo TEST\x000000000000000000000000000 )\n"
        b"CMAKE_HOST_SYSTEM_INFORMATION(RESULT bar QUERY HOSTNAME)\n"
    )


def _clang_null_character_in_literal_bytes() -> bytes:
    # Exact tip fixture used by residual eee7 (intel/llvm + llvm-project).
    return (
        Path(__file__).parent
        / "fixtures/source_quarantine/null-character-in-literal.c"
    ).read_bytes()


def _big5_shell_heredoc_bytes() -> bytes:
    big5_translation = bytes.fromhex(
        "a6b9a55cafe0bbddad6eabeaa66eabfca977a8e2add3bfe9a44ac0c9"
    )
    utf8_translation = (
        "\u6b64\u529f\u80fd\u9700\u8981\u6070\u597d\u6307\u5b9a"
        "\u5169\u500b\u8f38\u5165\u6a94"
    ).encode("utf-8")
    po_body = (
        b"# Chinese translation for GNU gettext messages.\n"
        b"#\n"
        b'msgid ""\n'
        b'msgstr ""\n'
        b'"MIME-Version: 1.0\\n"\n'
        b'"Content-Type: text/plain; charset=big5\\n"\n'
        b'"Content-Transfer-Encoding: 8bit\\n"\n\n'
        b"#: src/msgcmp.c:155 src/msgmerge.c:273\n"
        b'msgid "exactly 2 input files required"\n'
        b'msgstr "'
        + big5_translation
        + b'"\n'
    )
    ok_body = (
        b"# Chinese translation for GNU gettext messages.\n"
        b"#\n"
        b'msgid ""\n'
        b'msgstr ""\n'
        b'"MIME-Version: 1.0\\n"\n'
        b'"Content-Type: text/plain; charset=UTF-8\\n"\n'
        b'"Content-Transfer-Encoding: 8bit\\n"\n\n'
        b"#: src/msgcmp.c:155 src/msgmerge.c:273\n"
        b'msgid "exactly 2 input files required"\n'
        b'msgstr "'
        + utf8_translation
        + b'"\n'
    )
    return (
        b"#! /bin/sh\n\n"
        b"# Test conversion from BIG5 to UTF-8.\n\n"
        b'tmpfiles=""\n'
        b"trap 'rm -fr $tmpfiles' 1 2 3 15\n\n"
        b'tmpfiles="$tmpfiles mco-test1.po"\n'
        b"cat <<\\EOF > mco-test1.po\n"
        + po_body
        + b"EOF\n\n"
        + b'tmpfiles="$tmpfiles mco-test1.out"\n'
        + b": ${MSGCONV=msgconv}\n"
        + b"${MSGCONV} --to-code=UTF-8 -o mco-test1.out mco-test1.po\n"
        + b"test $? = 0 || { rm -fr $tmpfiles; exit 1; }\n\n"
        + b'tmpfiles="$tmpfiles mco-test1.ok"\n'
        + b"cat <<\\EOF > mco-test1.ok\n"
        + ok_body
        + b"EOF\n"
        + b"\n: ${DIFF=diff}\n"
        + b"# Redirect stdout, so as not to fill the user's screen with "
        + b"non-ASCII bytes.\n"
        + b"${DIFF} mco-test1.ok mco-test1.out >/dev/null\n"
        + b"result=$?\n\n"
        + b"rm -fr $tmpfiles\n\n"
        + b"exit $result\n"
    )


def _autotools_big5_shell_heredoc_bytes() -> bytes:
    legacy = _big5_shell_heredoc_bytes()
    po_marker = b"cat <<\\EOF > mco-test1.po\n"
    ok_marker = b"cat <<\\EOF > mco-test1.ok\n"
    po_start = legacy.index(po_marker) + len(po_marker)
    po_end = legacy.index(b"\nEOF\n", po_start)
    ok_start = legacy.index(ok_marker) + len(ok_marker)
    ok_end = legacy.index(b"\nEOF\n", ok_start)
    po_body = legacy[po_start:po_end]
    ok_body = legacy[ok_start:ok_end]
    return (
        b"#! /bin/sh\n"
        b'. "${srcdir=.}/init.sh"; path_prepend_ . ../src\n\n'
        b"# Test conversion from BIG5 to UTF-8.\n\n"
        b"cat <<\\EOF > mco-test1.po\n"
        + po_body
        + b"\nEOF\n\n"
        + b": ${MSGCONV=msgconv}\n"
        + b"${MSGCONV} --to-code=UTF-8 -o mco-test1.out mco-test1.po || Exit 1\n\n"
        + b"cat <<\\EOF > mco-test1.ok\n"
        + ok_body
        + b"\nEOF\n\n"
        + b": ${DIFF=diff}\n"
        + b"# Redirect stdout, so as not to fill the user's screen with "
        + b"non-ASCII bytes.\n"
        + b"${DIFF} mco-test1.ok mco-test1.out >/dev/null\n"
        + b"result=$?\n\n"
        + b"exit $result\n"
    )


def _git_shortlog_invalid_utf8_bytes() -> bytes:
    valid = b"\xf0\x9d\x84\x9e"
    malformed = b"\xf8\x9d\x84\x9e"
    return (
        b"#!/bin/sh\n"
        b"test_description='git shortlog\n'\n"
        b"# when replacing all is by treble clefs.\n"
        b'tr 1234 "\\360\\235\\204\\236"\n'
        + valid * 8
        + b"\n# now fsck up the utf8\n"
        + b"git config i18n.commitencoding non-utf-8\n"
        + b'tr 1234 "\\370\\235\\204\\236"\n'
        + b"# NOTE: do not quote this heredoc, Dash 0.5.13 has a bug with heredocs\n"
        + malformed * 8
        + b"\n"
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


def _git_fixture(root: Path, *args: str, stdin: str | None = None) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _create_dangling_gitlink_header_fixture(root: Path) -> Path:
    _git_fixture(root, "init", "-q")
    _git_fixture(root, "config", "user.name", "Fixture")
    _git_fixture(root, "config", "user.email", "fixture@example.invalid")
    empty_tree = _git_fixture(root, "mktree", stdin="")
    gitlink_commit = _git_fixture(
        root,
        "commit-tree",
        empty_tree,
        stdin="submodule fixture\n",
    )
    header = root / "include/onednn/dnnl_debug.h"
    header.parent.mkdir(parents=True)
    (root / "third_party/onednn").mkdir(parents=True)
    header.symlink_to("../../third_party/onednn/include/dnnl_debug.h")
    _git_fixture(root, "add", "include/onednn/dnnl_debug.h")
    _git_fixture(
        root,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{gitlink_commit},third_party/onednn",
    )
    _git_fixture(root, "commit", "-qm", "fixture")
    return header


def test_cpp_discovery_receipts_exact_dangling_gitlink_header(
    tmp_path: Path,
) -> None:
    header = _create_dangling_gitlink_header_fixture(tmp_path)
    collector = ip.GitTreeSourceEntryExclusions(tmp_path)

    assert ip.find_cpp_files(
        str(tmp_path),
        ineligible_entry_handler=collector.record,
    ) == []

    receipt = collector.receipt()
    assert receipt["schema"] == "cppmega.source_tree_entry_exclusions_v1"
    assert receipt["excluded_count"] == 1
    assert receipt["git_tree"] == _git_fixture(tmp_path, "rev-parse", "HEAD^{tree}")
    assert receipt["records"] == [
        {
            "relative_path": "include/onednn/dnnl_debug.h",
            "reason": "dangling_symlink_target_below_unmaterialized_gitlink",
            "git_tree": receipt["git_tree"],
            "entry_mode": "120000",
            "entry_object_id": _git_fixture(
                tmp_path,
                "rev-parse",
                "HEAD:include/onednn/dnnl_debug.h",
            ),
            "entry_object_type": "blob",
            "entry_object_size_bytes": len(header.readlink().as_posix().encode()),
            "entry_object_sha256": hashlib.sha256(
                header.readlink().as_posix().encode()
            ).hexdigest(),
            "symlink_target": "../../third_party/onednn/include/dnnl_debug.h",
            "target_relative_path": "third_party/onednn/include/dnnl_debug.h",
            "target_gitlink_path": "third_party/onednn",
            "target_gitlink_mode": "160000",
            "target_gitlink_commit": _git_fixture(
                tmp_path,
                "rev-parse",
                "HEAD:third_party/onednn",
            ),
        }
    ]
    assert receipt["records_sha256"] == hashlib.sha256(
        json.dumps(
            receipt["records"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    validated = _validate_source_tree_entry_exclusions(
        receipt,
        source_snapshot={"kind": "git_mirror", "tree": receipt["git_tree"]},
    )
    assert validated == receipt

    tampered = json.loads(json.dumps(receipt))
    tampered["records"][0]["symlink_target"] = "../../third_party/onednn/include/changed.h"
    tampered["records_sha256"] = hashlib.sha256(
        json.dumps(
            tampered["records"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ContractError, match="target path drifted"):
        _validate_source_tree_entry_exclusions(
            tampered,
            source_snapshot={"kind": "git_mirror", "tree": receipt["git_tree"]},
        )
    with pytest.raises(ContractError, match="not bound to the worker checkout tree"):
        _validate_source_tree_entry_exclusions(
            receipt,
            source_snapshot={"kind": "git_mirror", "tree": "0" * 40},
        )


def test_cpp_discovery_rejects_mutated_dangling_symlink(
    tmp_path: Path,
) -> None:
    header = _create_dangling_gitlink_header_fixture(tmp_path)
    header.unlink()
    header.symlink_to("../../third_party/onednn/include/changed.h")
    collector = ip.GitTreeSourceEntryExclusions(tmp_path)

    with pytest.raises(OSError, match="non-regular entry proof failed"):
        ip.find_cpp_files(
            str(tmp_path),
            ineligible_entry_handler=collector.record,
        )


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


def test_exact_quarantine_filters_nul_ff_binary_blob(tmp_path: Path) -> None:
    payload = b"\0\xff" * 386 + b"\0"
    candidate = tmp_path / RELATIVE_NUL_FF_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="nul_ff_binary_blob",
        relative_path=RELATIVE_NUL_FF_BLOB,
        reason="binary 0x00/0xff payload stored under a header suffix",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["detected_format"] == "nul_ff_binary_blob"


def test_exact_quarantine_filters_truncated_utf32be_bom(tmp_path: Path) -> None:
    payload = _truncated_utf32be_bom_bytes()
    candidate = tmp_path / RELATIVE_TRUNCATED_UTF32BE_BOM
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="truncated_utf32be_bom",
        relative_path=RELATIVE_TRUNCATED_UTF32BE_BOM,
        reason="truncated UTF-32BE BOM fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["detected_format"] == "truncated_utf32be_bom"


def test_truncated_utf32be_bom_quarantine_rejects_other_payload(
    tmp_path: Path,
) -> None:
    payload = b"\x00\x00\xff"
    candidate = tmp_path / RELATIVE_TRUNCATED_UTF32BE_BOM
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="truncated_utf32be_bom",
        relative_path=RELATIVE_TRUNCATED_UTF32BE_BOM,
        reason="forged truncated UTF-32BE BOM fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="exactly the three-byte"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_exact_quarantine_filters_truncated_utf32le_bom(tmp_path: Path) -> None:
    payload = _truncated_utf32le_bom_bytes()
    candidate = tmp_path / RELATIVE_TRUNCATED_UTF32LE_BOM
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="truncated_utf32le_bom",
        relative_path=RELATIVE_TRUNCATED_UTF32LE_BOM,
        reason="truncated UTF-32LE BOM fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["detected_format"] == "truncated_utf32le_bom"


def test_truncated_utf32le_bom_quarantine_rejects_other_payload(
    tmp_path: Path,
) -> None:
    payload = b"\xff\xfe\x01"
    candidate = tmp_path / RELATIVE_TRUNCATED_UTF32LE_BOM
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="truncated_utf32le_bom",
        relative_path=RELATIVE_TRUNCATED_UTF32LE_BOM,
        reason="forged truncated UTF-32LE BOM fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="exactly the three-byte"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_exact_quarantine_filters_cmake_null_after_backslash_fixture(
    tmp_path: Path,
) -> None:
    payload = _cmake_null_after_backslash_bytes()
    candidate = tmp_path / RELATIVE_CMAKE_NULL_AFTER_BACKSLASH
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_parser_regression_fixture",
        detected_format="cmake_escaped_newline_nul_syntax_fixture",
        relative_path=RELATIVE_CMAKE_NULL_AFTER_BACKSLASH,
        reason="CMake escaped-newline/NUL syntax fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["detected_format"] == (
        "cmake_escaped_newline_nul_syntax_fixture"
    )


def test_cmake_null_after_backslash_quarantine_rejects_other_nul_payload(
    tmp_path: Path,
) -> None:
    payload = b"message(\\\0\nSTATUS forged)\n"
    candidate = tmp_path / RELATIVE_CMAKE_NULL_AFTER_BACKSLASH
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_parser_regression_fixture",
        detected_format="cmake_escaped_newline_nul_syntax_fixture",
        relative_path=RELATIVE_CMAKE_NULL_AFTER_BACKSLASH,
        reason="forged CMake escaped-newline/NUL syntax fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="exact two-line"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_exact_quarantine_filters_big5_shell_heredoc(tmp_path: Path) -> None:
    payload = _big5_shell_heredoc_bytes()
    candidate = tmp_path / RELATIVE_BIG5_SHELL_HEREDOC
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="big5_shell_heredoc",
        relative_path=RELATIVE_BIG5_SHELL_HEREDOC,
        reason="BIG5 shell heredoc fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["detected_format"] == "big5_shell_heredoc"


def test_big5_shell_heredoc_quarantine_rejects_changed_message(
    tmp_path: Path,
) -> None:
    payload = _big5_shell_heredoc_bytes().replace(b"--to-code=UTF-8", b"--to-code=BIG5")
    candidate = tmp_path / RELATIVE_BIG5_SHELL_HEREDOC
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="big5_shell_heredoc",
        relative_path=RELATIVE_BIG5_SHELL_HEREDOC,
        reason="forged BIG5 shell heredoc fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="conversion and cleanup"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_exact_quarantine_filters_autotools_big5_shell_heredoc(
    tmp_path: Path,
) -> None:
    payload = _autotools_big5_shell_heredoc_bytes()
    candidate = tmp_path / RELATIVE_AUTOTOOLS_BIG5_SHELL_HEREDOC
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="big5_shell_heredoc",
        relative_path=RELATIVE_AUTOTOOLS_BIG5_SHELL_HEREDOC,
        reason="modern BIG5 shell heredoc fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["detected_format"] == "big5_shell_heredoc"


def test_autotools_big5_shell_heredoc_rejects_changed_wrapper(
    tmp_path: Path,
) -> None:
    payload = _autotools_big5_shell_heredoc_bytes().replace(
        b"|| Exit 1", b"|| exit 1"
    )
    candidate = tmp_path / RELATIVE_AUTOTOOLS_BIG5_SHELL_HEREDOC
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="big5_shell_heredoc",
        relative_path=RELATIVE_AUTOTOOLS_BIG5_SHELL_HEREDOC,
        reason="forged modern BIG5 shell heredoc fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="conversion and cleanup"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_exact_quarantine_filters_git_shortlog_invalid_utf8(
    tmp_path: Path,
) -> None:
    payload = _git_shortlog_invalid_utf8_bytes()
    candidate = tmp_path / RELATIVE_GIT_SHORTLOG_INVALID_UTF8
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_encoding_regression_fixture",
        detected_format="git_shortlog_invalid_utf8_shell",
        relative_path=RELATIVE_GIT_SHORTLOG_INVALID_UTF8,
        reason="deliberate malformed UTF-8 shell fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["detected_format"] == (
        "git_shortlog_invalid_utf8_shell"
    )


def test_git_shortlog_invalid_utf8_rejects_changed_payload(
    tmp_path: Path,
) -> None:
    payload = _git_shortlog_invalid_utf8_bytes().replace(
        b"\xf8\x9d\x84\x9e", b"\xf0\x9d\x84\x9e", 1
    )
    candidate = tmp_path / RELATIVE_GIT_SHORTLOG_INVALID_UTF8
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_encoding_regression_fixture",
        detected_format="git_shortlog_invalid_utf8_shell",
        relative_path=RELATIVE_GIT_SHORTLOG_INVALID_UTF8,
        reason="forged malformed UTF-8 shell fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="contract is incomplete"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_nul_ff_binary_blob_verification_streams_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"\0" * (1024 * 1024) + b"\xff" * (1024 * 1024)
    candidate = tmp_path / RELATIVE_NUL_FF_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="nul_ff_binary_blob",
        relative_path=RELATIVE_NUL_FF_BLOB,
        reason="binary 0x00/0xff payload stored under a header suffix",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    original_read_bytes = Path.read_bytes

    def reject_candidate_read_bytes(path: Path) -> bytes:
        if path == candidate:
            raise AssertionError("nul_ff_binary_blob verification must stream input")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_candidate_read_bytes)

    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1


@pytest.mark.parametrize(
    "payload",
    [b"", b"\0" * 4, b"\xff" * 4, b"\0\xff\x01"],
)
def test_nul_ff_binary_blob_requires_both_values_and_no_others(
    tmp_path: Path,
    payload: bytes,
) -> None:
    candidate = tmp_path / RELATIVE_NUL_FF_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="nul_ff_binary_blob",
        relative_path=RELATIVE_NUL_FF_BLOB,
        reason="binary 0x00/0xff payload stored under a header suffix",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="only 0x00 and 0xff"):
        policy.filter_candidates(tmp_path, [str(candidate)])


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


def test_exact_quarantine_filters_clang_embedded_nul_diagnostic(
    tmp_path: Path,
) -> None:
    payload = _clang_embedded_nul_diagnostic_bytes()
    candidate = tmp_path / RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="clang_embedded_nul_diagnostic",
        relative_path=RELATIVE_NUL_DIAGNOSTIC_FIXTURE,
        reason="fixture intentionally embeds a NUL for Clang diagnostics",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["classification"] == (
        "deliberate_compiler_diagnostic_fixture"
    )
    assert receipt["entries"][0]["detected_format"] == (
        "clang_embedded_nul_diagnostic"
    )


def test_exact_quarantine_filters_clang_newline_nul_diagnostic(
    tmp_path: Path,
) -> None:
    payload = _clang_newline_nul_diagnostic_bytes()
    candidate = tmp_path / RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="clang_escaped_newline_nul_preprocessor_diagnostic",
        relative_path=RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE,
        reason="fixture intentionally embeds a NUL after an escaped newline",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["classification"] == (
        "deliberate_compiler_diagnostic_fixture"
    )
    assert receipt["entries"][0]["detected_format"] == (
        "clang_escaped_newline_nul_preprocessor_diagnostic"
    )


def test_clang_newline_nul_quarantine_requires_preprocessor_signature(
    tmp_path: Path,
) -> None:
    payload = _clang_newline_nul_diagnostic_bytes().replace(
        b"expected-error {{not a valid binary operator}}",
        b"expected-error {{unrelated diagnostic}}",
    )
    candidate = tmp_path / RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="clang_escaped_newline_nul_preprocessor_diagnostic",
        relative_path=RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE,
        reason="forged newline/NUL diagnostic fixture",
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


def test_exact_quarantine_filters_utf16le_generated_binary_blob(
    tmp_path: Path,
) -> None:
    payload = _utf16le_generated_c_array_bytes()
    candidate = tmp_path / RELATIVE_GENERATED_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_binary_blob",
        detected_format="utf16le_generated_c_array",
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
    payload = _utf16le_generated_c_array_bytes(byte_count=16)
    candidate = tmp_path / RELATIVE_GENERATED_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_binary_blob",
        detected_format="utf16le_generated_c_array",
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


def test_checked_in_intel_nul_diagnostic_manifest_matches_reference_fixture() -> None:
    payload = _clang_embedded_nul_diagnostic_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "intel/llvm"
        and item["relative_path"] == RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    )

    assert len(payload) == 398
    assert hashlib.sha256(payload).hexdigest() == (
        "acba383a8c05e95c15d885e06c467d58edf39f5c7f84a0376f86cdb20d40be3a"
    )
    assert entry["relative_path"] == RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "deliberate_compiler_diagnostic_fixture"
    assert entry["detected_format"] == "clang_embedded_nul_diagnostic"


def test_checked_in_intel_newline_nul_manifest_matches_reference_fixture() -> None:
    payload = _clang_newline_nul_diagnostic_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "intel/llvm"
        and item["relative_path"] == RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE
    )

    assert len(payload) == 332
    assert hashlib.sha256(payload).hexdigest() == (
        "11548a466e2e5eb0a686cf422f7872ea94f616fe43f4bdb9311a0f1764391474"
    )
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "deliberate_compiler_diagnostic_fixture"
    assert entry["detected_format"] == (
        "clang_escaped_newline_nul_preprocessor_diagnostic"
    )


def test_checked_in_cmake_syntax_manifest_matches_archive_evidence() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        item
        for item in manifest["entries"]
        if item["project_id"] == "Kitware/CMake"
    ]

    expected = {
        RELATIVE_TRUNCATED_UTF32BE_BOM: (
            _truncated_utf32be_bom_bytes(),
            "truncated_utf32be_bom",
            "mislabeled_non_cpp",
        ),
        RELATIVE_TRUNCATED_UTF32LE_BOM: (
            _truncated_utf32le_bom_bytes(),
            "truncated_utf32le_bom",
            "mislabeled_non_cpp",
        ),
        RELATIVE_CMAKE_NULL_AFTER_BACKSLASH: (
            _cmake_null_after_backslash_bytes(),
            "cmake_escaped_newline_nul_syntax_fixture",
            "deliberate_parser_regression_fixture",
        ),
        RELATIVE_CMAKE_NULL_TERMINATED_ARGUMENT: (
            _cmake_null_terminated_argument_bytes(),
            "cmake_null_terminated_argument_fixture",
            "deliberate_parser_regression_fixture",
        ),
    }
    assert {entry["relative_path"] for entry in entries} == set(expected)
    for entry in entries:
        payload, detected_format, classification = expected[entry["relative_path"]]
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["classification"] == classification
        assert entry["detected_format"] == detected_format


def test_checked_in_windows_nul_ff_manifest_matches_pinned_archive() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        item
        for item in manifest["entries"]
        if item["project_id"] == "corpus.local/windows_10_shared_source_kit"
        and item["detected_format"] == "nul_ff_binary_blob"
    ]

    expected = {
        "windows_10_shared_source_kit/unknown_version_2/Source/drivers/spb/"
        "spbcx/sys/driver.h": (
            773,
            "38f8873ec81398a0a0e025690b3e2baa55c5c1221441d68ca0144b0b557fcb4b",
        ),
        "windows_10_shared_source_kit/unknown_version_2/Source/drivers/wdm/usb/"
        "usb3/usbxhci/sys/driver/driver.h": (
            1552,
            "1a165209c95259239d74a5db250ca275ff9356b8a40e6ec740d0e910e9eaabed",
        ),
    }
    assert {
        entry["relative_path"]: (entry["size_bytes"], entry["sha256"])
        for entry in entries
    } == expected
    assert {
        (entry["classification"], entry["detected_format"]) for entry in entries
    } == {("mislabeled_non_cpp", "nul_ff_binary_blob")}

    usb_payload = b"\0\0\xff\xff" * 388
    assert len(usb_payload) == 1552
    assert hashlib.sha256(usb_payload).hexdigest() == expected[
        "windows_10_shared_source_kit/unknown_version_2/Source/drivers/wdm/usb/"
        "usb3/usbxhci/sys/driver/driver.h"
    ][1]


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


def test_checked_in_threadx_generated_blob_manifest_matches_frozen_receipt() -> None:
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

    # Rebound after GCP residual eee7 size mismatch (60766 -> 61551 on tip commit).
    assert entry["size_bytes"] == 61551
    assert entry["sha256"] == (
        "2d49edeeb4233af4972ac4f9cec96b171d92ffad0738eaf3b4dcd536a05e9294"
    )
    assert entry["classification"] == "generated_binary_blob"
    assert entry["detected_format"] == "utf16le_generated_c_array"


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


def test_checked_in_netbsd_big5_shell_manifest_matches_archive_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "NetBSD/src"
    )

    payload = _big5_shell_heredoc_bytes()
    assert entry["relative_path"] == RELATIVE_BIG5_SHELL_HEREDOC
    assert entry["size_bytes"] == len(payload) == 1155
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "mislabeled_non_cpp"
    assert entry["detected_format"] == "big5_shell_heredoc"


def test_checked_in_autotools_big5_shell_manifest_matches_pinned_commit() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "autotools-mirror/gettext"
    )

    payload = _autotools_big5_shell_heredoc_bytes()
    assert entry["relative_path"] == RELATIVE_AUTOTOOLS_BIG5_SHELL_HEREDOC
    assert entry["size_bytes"] == len(payload) == 1001
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "mislabeled_non_cpp"
    assert entry["detected_format"] == "big5_shell_heredoc"


def test_checked_in_git_shortlog_manifest_matches_pinned_commit_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "git/git"
    )

    assert entry["relative_path"] == RELATIVE_GIT_SHORTLOG_INVALID_UTF8
    assert entry["size_bytes"] == 11480
    assert entry["sha256"] == (
        "017c37e96a1bf295be7436f35ff9da9b42f7baa40220e353e3414dfca6af11dd"
    )
    assert entry["classification"] == "deliberate_encoding_regression_fixture"
    assert entry["detected_format"] == "git_shortlog_invalid_utf8_shell"


@pytest.mark.parametrize(
    ("project_id", "relative_path", "size_bytes", "sha256"),
    [
        (
            "python/cpython",
            "Lib/test/archivetestdata/exe_with_zip",
            990,
            "2f27f5c9108936a693fd496565e5c5050b5c62cfbb61d1d5da9d97c89533d637",
        ),
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
        item
        for item in manifest["entries"]
        if item["project_id"] == project_id
        and item["relative_path"] == relative_path
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


def test_clang_embedded_nul_quarantine_requires_diagnostic_signature(
    tmp_path: Path,
) -> None:
    payload = b"int x[sizeof\0int];\n"
    candidate = tmp_path / RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="clang_embedded_nul_diagnostic",
        relative_path=RELATIVE_NUL_DIAGNOSTIC_FIXTURE,
        reason="forged diagnostic fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(
        SourceQuarantineError,
        match="embedded-NUL diagnostic contract is incomplete",
    ):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_clang_embedded_nul_quarantine_requires_both_caret_markers(
    tmp_path: Path,
) -> None:
    payload = _clang_embedded_nul_diagnostic_bytes().replace(
        b"// CHECK-NEXT:             ^\n",
        b"// CHECK-NEXT:             x\n",
        1,
    )
    candidate = tmp_path / RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="clang_embedded_nul_diagnostic",
        relative_path=RELATIVE_NUL_DIAGNOSTIC_FIXTURE,
        reason="forged diagnostic fixture missing one caret marker",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(
        SourceQuarantineError,
        match="embedded-NUL diagnostic contract is incomplete",
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
    assert receipt["schema"] == "cppmega.source_quarantine_receipt_v2"
    assert receipt["project_id"] == PROJECT_ID
    assert (
        receipt["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert receipt["quarantined_count"] == 1
    assert receipt["source_tree_entry_exclusions"]["excluded_count"] == 0
    omission_receipt = receipt["external_reference_omissions"]
    assert omission_receipt["schema"] == "cppmega.external_reference_omissions_v1"
    assert omission_receipt["status"] == "complete"
    assert omission_receipt["reason"] == "unknown_external_provider"
    assert omission_receipt["observation_count"] == 0
    assert omission_receipt["unique_reference_count"] == 0
    assert omission_receipt["location_count"] == 0
    assert omission_receipt["locations"] == []

    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    validated_v2 = validate_quarantine_receipt_file(
        receipt_path,
        project_id=PROJECT_ID,
        manifest_sha256=manifest_sha256,
        source_snapshot={"kind": "git_mirror", "tree": "0" * 40},
    )
    assert validated_v2["schema"] == "cppmega.source_quarantine_receipt_v2"

    legacy_receipt = dict(receipt)
    legacy_receipt["schema"] = "cppmega.source_quarantine_receipt_v1"
    legacy_receipt.pop("source_tree_entry_exclusions")
    legacy_path = tmp_path / "receipts/source-v1.json"
    legacy_path.write_text(
        json.dumps(legacy_receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    validated_v1 = validate_quarantine_receipt_file(
        legacy_path,
        project_id=PROJECT_ID,
        manifest_sha256=manifest_sha256,
        source_snapshot={"kind": "git_mirror", "tree": "0" * 40},
    )
    assert validated_v1["schema"] == "cppmega.source_quarantine_receipt_v1"
    assert "source_tree_entry_exclusions" not in validated_v1

    malformed_v2 = dict(legacy_receipt)
    malformed_v2["schema"] = "cppmega.source_quarantine_receipt_v2"
    malformed_v2_path = tmp_path / "receipts/source-v2-missing-exclusions.json"
    malformed_v2_path.write_text(
        json.dumps(malformed_v2, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="fields drifted"):
        validate_quarantine_receipt_file(
            malformed_v2_path,
            project_id=PROJECT_ID,
            manifest_sha256=manifest_sha256,
            source_snapshot={"kind": "git_mirror", "tree": "0" * 40},
        )


def test_process_project_quarantines_clang_embedded_nul_diagnostic(
    tmp_path: Path,
) -> None:
    payload = _clang_embedded_nul_diagnostic_bytes()
    candidate = tmp_path / RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    newline_payload = _clang_newline_nul_diagnostic_bytes()
    newline_candidate = tmp_path / RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE
    newline_candidate.parent.mkdir(parents=True)
    newline_candidate.write_bytes(newline_payload)
    literal_payload = _clang_null_character_in_literal_bytes()
    literal_candidate = tmp_path / RELATIVE_NUL_IN_LITERAL_FIXTURE
    literal_candidate.parent.mkdir(parents=True, exist_ok=True)
    literal_candidate.write_bytes(literal_payload)
    manifest = Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
    receipt_path = tmp_path / "receipts/source.json"

    documents = ip.process_project(
        str(tmp_path),
        enriched=True,
        project_id="intel/llvm",
        source_quarantine_manifest=str(manifest),
        source_quarantine_receipt=str(receipt_path),
    )

    assert documents == []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["project_id"] == "intel/llvm"
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert receipt["quarantined_count"] == 3
    assert {entry["relative_path"] for entry in receipt["entries"]} == {
        RELATIVE_NUL_DIAGNOSTIC_FIXTURE,
        RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE,
        RELATIVE_NUL_IN_LITERAL_FIXTURE,
    }


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


def test_process_project_quarantines_cmake_nul_before_all_domain_routes(
    tmp_path: Path,
) -> None:
    payload = _cmake_null_after_backslash_bytes()
    candidate = tmp_path / RELATIVE_CMAKE_NULL_AFTER_BACKSLASH
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    receipt_path = tmp_path / "receipts/source.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_parser_regression_fixture",
        detected_format="cmake_escaped_newline_nul_syntax_fixture",
        relative_path=RELATIVE_CMAKE_NULL_AFTER_BACKSLASH,
        reason="CMake escaped-newline/NUL syntax fixture",
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
    assert receipt["entries"][0]["relative_path"] == (
        RELATIVE_CMAKE_NULL_AFTER_BACKSLASH
    )


def test_utf16le_source_text_format_accepts_bom_comment_header(tmp_path):
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )
    import hashlib
    # UTF-16LE BOM + "// hello\n"
    payload = b"\xff\xfe" + "// hello\n".encode("utf-16le")
    path = tmp_path / "resource.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="example/repo",
        relative_path="resource.h",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="mislabeled_non_cpp",
        detected_format="utf16le_source_text",
        reason="test fixture",
    )
    _verify_detected_format(path, entry)


def test_utf16le_source_text_rejects_missing_bom(tmp_path):
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )
    import hashlib
    import pytest
    payload = b"// no bom\n"
    path = tmp_path / "resource.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="example/repo",
        relative_path="resource.h",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="mislabeled_non_cpp",
        detected_format="utf16le_source_text",
        reason="test fixture",
    )
    with pytest.raises(SourceQuarantineError, match="UTF-16LE BOM"):
        _verify_detected_format(path, entry)

def _plumhall_d412_fixture_bytes() -> bytes:
    return (
        b"/* The Plum Hall Validation Suite for C\n"
        b" * Unpublished copyright (c) 1986-1990, Chiron Systems Inc and Plum Hall Inc.\n"
        b" */\n"
        b"#define LIB_TEST 1\n"
        b'#include "defs.h"\n'
        b"#if !ANSI\n"
        b"#define SKIP412 1 /* This file is almost irrelevant for non-ANSI */\n"
        b"#endif\n"
        b"#ifndef SKIP412\n"
        b"/*\n"
        b" * 4.12 - Date and time\n"
        b" */\n"
        b"#include <time.h>\n"
        b"#include <limits.h>\n"
        b"#include <string.h>\n"
        b"struct tm tm1, tm2, *ptm;\n"
        b"static time_t time_t1, time_t2, time_t3;\n"
        b"static void d4_12_1();\n"
        b"static void d4_12_2();\n"
        b"static void d4_12_3();\n"
        b"void d4_12()\n"
        b"{\n"
        b'Filename = "d412.c";\n'
        b"d4_12_1();\n"
        b"d4_12_2();\n"
        b"d4_12_3();\n"
        b"}\n"
        b"#endif\n"
    )


def test_exact_quarantine_filters_plumhall_d412_libclang_hang_fixture(
    tmp_path: Path,
) -> None:
    payload = _plumhall_d412_fixture_bytes()
    candidate = tmp_path / RELATIVE_PLUMHALL_D412
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="compiler_regression_fixture",
        detected_format="plumhall_c_date_time_libclang_hang",
        relative_path=RELATIVE_PLUMHALL_D412,
        reason="Plum Hall D412 hangs the pinned libclang parser",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["classification"] == "compiler_regression_fixture"
    assert receipt["entries"][0]["detected_format"] == (
        "plumhall_c_date_time_libclang_hang"
    )


def test_plumhall_d412_contract_rejects_incomplete_fixture(tmp_path: Path) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "D412.c"
    path.write_text("int main(void) { return 0; }\n", encoding="ascii")
    entry = SourceQuarantineEntry(
        project_id=PROJECT_ID,
        relative_path="D412.c",
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="plumhall_c_date_time_libclang_hang",
        reason="negative test",
    )
    with pytest.raises(SourceQuarantineError, match="Plum Hall 4.12"):
        _verify_detected_format(path, entry)


def test_invalid_utf8_and_windows1252_domain_blob_accepts_old_pcre1_testoutput1(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "erlang_old_pcre1_testoutput1"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "testoutput1"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="erlang/otp",
        relative_path="lib/stdlib/test/re_SUITE_data/old_pcre1/testoutput1",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="deliberate_encoding_regression_fixture",
        detected_format="invalid_utf8_and_windows1252_domain_blob",
        reason="old_pcre1 dual-encoding regression fixture",
    )
    _verify_detected_format(path, entry)


def test_apple_security_ssdl_session_libclang_hang_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "SSDLSession.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "SSDLSession.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path="OSX/libsecurity_apple_cspdl/lib/SSDLSession.h",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_ssdl_session_libclang_hang",
        reason="Security SSDLSession.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_ssdl_session_contract_rejects_unrelated_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "SSDLSession.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path="OSX/libsecurity_apple_cspdl/lib/SSDLSession.h",
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_ssdl_session_libclang_hang",
        reason="negative test",
    )
    with pytest.raises(SourceQuarantineError, match="Apple Security libclang-timeout header contract"):
        _verify_detected_format(path, entry)


def test_apple_security_cssmcontext_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "cssmcontext.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "cssmcontext.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path="OSX/libsecurity_cssm/lib/cssmcontext.h",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security cssmcontext.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_plan9_astar_a100p_binary_blob_accepts_fixture(tmp_path: Path) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "a100p.cp"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "a100p.cp"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="plan9foundation/plan9",
        relative_path="sys/lib/astar/a100p.cp",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="mislabeled_non_cpp",
        detected_format="binary_blob_with_embedded_nul",
        reason="Plan 9 astar a100p.cp code page",
    )
    _verify_detected_format(path, entry)


def test_plan9_astar_a100p_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "a100p.cp"
    path.write_bytes(b"not-a-code-page\n")
    entry = SourceQuarantineEntry(
        project_id="plan9foundation/plan9",
        relative_path="sys/lib/astar/a100p.cp",
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="mislabeled_non_cpp",
        detected_format="binary_blob_with_embedded_nul",
        reason="negative test",
    )
    with pytest.raises(SourceQuarantineError, match="binary_blob_with_embedded_nul"):
        _verify_detected_format(path, entry)


def test_apple_security_sdcontext_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "SDContext.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "SDContext.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path="OSX/libsecurity_sd_cspdl/lib/SDContext.h",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security SDContext.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_sdcontext_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "SDContext.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path="OSX/libsecurity_sd_cspdl/lib/SDContext.h",
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


RELATIVE_GCC_ENCODING_ISSUES_BYTES = (
    "gcc/testsuite/gcc.dg/encoding-issues-bytes.c"
)


def test_gcc_encoding_issues_bytes_accepts_fixture(tmp_path: Path) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "encoding-issues-bytes.c"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "encoding-issues-bytes.c"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="gcc-mirror/gcc",
        relative_path=RELATIVE_GCC_ENCODING_ISSUES_BYTES,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="gcc_embedded_nul_diagnostic",
        reason="GCC encoding-issues-bytes.c diagnostic fixture",
    )
    _verify_detected_format(path, entry)


def test_gcc_encoding_issues_bytes_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "encoding-issues-bytes.c"
    path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="gcc-mirror/gcc",
        relative_path=RELATIVE_GCC_ENCODING_ISSUES_BYTES,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="gcc_embedded_nul_diagnostic",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="gcc_embedded_nul_diagnostic"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_gcc_encoding_issues_bytes_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "encoding-issues-bytes.c"
    )
    payload = fixture.read_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["relative_path"] == RELATIVE_GCC_ENCODING_ISSUES_BYTES
    )
    assert len(payload) == 595
    assert hashlib.sha256(payload).hexdigest() == (
        "c1cd6c749b597a7547c374a348f1ea2af12a22af94ccdb93c7204713b142dcc3"
    )
    assert payload.count(b"\x00") == 1
    assert payload.count(b"\x80") == 1
    assert entry["project_id"] == "gcc-mirror/gcc"
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "deliberate_compiler_diagnostic_fixture"
    assert entry["detected_format"] == "gcc_embedded_nul_diagnostic"


RELATIVE_GCC_ENCODING_ISSUES_UNICODE = (
    "gcc/testsuite/gcc.dg/encoding-issues-unicode.c"
)


def test_gcc_encoding_issues_unicode_accepts_fixture(tmp_path: Path) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "encoding-issues-unicode.c"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "encoding-issues-unicode.c"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="gcc-mirror/gcc",
        relative_path=RELATIVE_GCC_ENCODING_ISSUES_UNICODE,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="gcc_embedded_nul_diagnostic",
        reason="GCC encoding-issues-unicode.c diagnostic fixture",
    )
    _verify_detected_format(path, entry)


def test_gcc_encoding_issues_unicode_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "encoding-issues-unicode.c"
    path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="gcc-mirror/gcc",
        relative_path=RELATIVE_GCC_ENCODING_ISSUES_UNICODE,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="gcc_embedded_nul_diagnostic",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="gcc_embedded_nul_diagnostic"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_gcc_encoding_issues_unicode_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "encoding-issues-unicode.c"
    )
    payload = fixture.read_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["relative_path"] == RELATIVE_GCC_ENCODING_ISSUES_UNICODE
    )
    assert len(payload) == 613
    assert hashlib.sha256(payload).hexdigest() == (
        "263f7289a5e9fb2eb40e53d0a4e7ab6f4691cc173de5a9f981396195598bc084"
    )
    assert payload.count(b"\x00") == 1
    assert payload.count(b"\x80") == 1
    assert payload.count(b"\x01") == 1
    assert entry["project_id"] == "gcc-mirror/gcc"
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "deliberate_compiler_diagnostic_fixture"
    assert entry["detected_format"] == "gcc_embedded_nul_diagnostic"


RELATIVE_KCDATABASE = "securityd/src/kcdatabase.h"


def test_apple_security_kcdatabase_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "kcdatabase.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "kcdatabase.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_KCDATABASE,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security kcdatabase.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_kcdatabase_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "kcdatabase.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_KCDATABASE,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


RELATIVE_CHILD_H = "securityd/src/child.h"


def test_apple_security_child_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "child.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "child.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_CHILD_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security child.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_child_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "child.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_CHILD_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


RELATIVE_SSKEY_H = "OSX/libsecurity_apple_cspdl/lib/SSKey.h"


def test_apple_security_sskey_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "SSKey.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "SSKey.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SSKEY_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security SSKey.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_sskey_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "SSKey.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SSKEY_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


RELATIVE_KCKEY_H = "securityd/src/kckey.h"


def test_apple_security_kckey_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "kckey.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "kckey.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_KCKEY_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security kckey.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_kckey_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "kckey.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_KCKEY_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


RELATIVE_SSDATABASE_H = "OSX/libsecurity_apple_cspdl/lib/SSDatabase.h"


def test_apple_security_ssdatabase_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "SSDatabase.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "SSDatabase.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SSDATABASE_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security SSDatabase.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_ssdatabase_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "SSDatabase.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SSDATABASE_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


RELATIVE_SSCONTEXT_H = "OSX/libsecurity_apple_cspdl/lib/SSContext.h"


def test_apple_security_sscontext_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "SSContext.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "SSContext.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SSCONTEXT_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security SSContext.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_sscontext_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "SSContext.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SSCONTEXT_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


RELATIVE_DYLDCACHE_H = "OSX/include/security_utilities/dyldcache.h"
DYLDCACHE_SIZE = 5792
DYLDCACHE_SHA256 = (
    "d3fdddc450250300a28662c106c873c95980bfa55c6c33c32ea933a8ae86dc1c"
)


def test_apple_security_dyldcache_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "dyldcache.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "dyldcache.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_DYLDCACHE_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security dyldcache.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_dyldcache_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "dyldcache.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_DYLDCACHE_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_dyldcache_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "dyldcache.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == DYLDCACHE_SIZE
    assert hashlib.sha256(payload).hexdigest() == DYLDCACHE_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_DYLDCACHE_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == DYLDCACHE_SIZE
    assert entry["sha256"] == DYLDCACHE_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_DYLDCACHE_LIB_H = "OSX/libsecurity_utilities/lib/dyldcache.h"


def test_apple_security_dyldcache_lib_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "dyldcache.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "dyldcache.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_DYLDCACHE_LIB_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security lib/dyldcache.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_dyldcache_lib_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "dyldcache.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_DYLDCACHE_LIB_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_dyldcache_lib_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "dyldcache.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == DYLDCACHE_SIZE
    assert hashlib.sha256(payload).hexdigest() == DYLDCACHE_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    include_entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_DYLDCACHE_H
    ]
    lib_entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_DYLDCACHE_LIB_H
    ]
    assert len(include_entries) == 1
    assert len(lib_entries) == 1
    entry = lib_entries[0]
    assert entry["size_bytes"] == DYLDCACHE_SIZE
    assert entry["sha256"] == DYLDCACHE_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    assert include_entries[0]["sha256"] == entry["sha256"]


RELATIVE_AUTHHOST_H = "securityd/src/authhost.h"
AUTHHOST_SIZE = 1347
AUTHHOST_SHA256 = (
    "202d27b393db9e82b0cbf71c195c7a61f3dd287fc51e90a8205fd9f0e74d0f24"
)


def test_apple_security_authhost_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "authhost.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "authhost.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_AUTHHOST_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security authhost.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_authhost_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "authhost.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_AUTHHOST_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_authhost_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "authhost.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == AUTHHOST_SIZE
    assert hashlib.sha256(payload).hexdigest() == AUTHHOST_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_AUTHHOST_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == AUTHHOST_SIZE
    assert entry["sha256"] == AUTHHOST_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_CONNECTION_H = "securityd/src/connection.h"
CONNECTION_SIZE = 3852
CONNECTION_SHA256 = (
    "ec59489c9c405870d7d47a0ff994e056933e439a3849e282504908a05bacfd4d"
)


def test_apple_security_connection_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "connection.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "connection.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_CONNECTION_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security connection.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_connection_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "connection.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_CONNECTION_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_connection_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "connection.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == CONNECTION_SIZE
    assert hashlib.sha256(payload).hexdigest() == CONNECTION_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_CONNECTION_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == CONNECTION_SIZE
    assert entry["sha256"] == CONNECTION_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_KEY_H = "securityd/src/key.h"
KEY_H_SIZE = 3169
KEY_H_SHA256 = (
    "d0e1076edcfd9821003cf03d4dd8cbbcb5c0340ae02d9527eb5491f698c960f8"
)


def test_apple_security_key_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "key.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "key.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_KEY_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security key.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_key_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "key.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_KEY_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_key_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "key.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == KEY_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == KEY_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_KEY_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == KEY_H_SIZE
    assert entry["sha256"] == KEY_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_STRUCTURE_H = "securityd/src/structure.h"
STRUCTURE_H_SIZE = 7630
STRUCTURE_H_SHA256 = (
    "e4a1e0ee784a3ee50a5a919c14f7aa443396008cae2ec6e9f48a9dcef7d57151"
)


def test_apple_security_structure_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "structure.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "structure.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_STRUCTURE_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security structure.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_structure_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "structure.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_STRUCTURE_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_structure_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "structure.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == STRUCTURE_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == STRUCTURE_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_STRUCTURE_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == STRUCTURE_H_SIZE
    assert entry["sha256"] == STRUCTURE_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_SLCREP_H = "OSX/include/security_codesigning/slcrep.h"
SLCREP_H_SIZE = 2766
SLCREP_H_SHA256 = (
    "cf2b7726b5045af854870a010061854498c53484e169be226ce0bbe5b1d6ba33"
)


def test_apple_security_slcrep_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "slcrep.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "slcrep.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SLCREP_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security slcrep.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_slcrep_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "slcrep.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SLCREP_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_slcrep_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "slcrep.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == SLCREP_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == SLCREP_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_SLCREP_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == SLCREP_H_SIZE
    assert entry["sha256"] == SLCREP_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_SLCREP_LIB_H = "OSX/libsecurity_codesigning/lib/slcrep.h"


def test_apple_security_slcrep_lib_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "slcrep.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "slcrep.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SLCREP_LIB_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security lib/slcrep.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_slcrep_lib_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "slcrep.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SLCREP_LIB_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_slcrep_lib_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "slcrep.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == SLCREP_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == SLCREP_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    include_entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_SLCREP_H
    ]
    lib_entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_SLCREP_LIB_H
    ]
    assert len(include_entries) == 1
    assert len(lib_entries) == 1
    entry = lib_entries[0]
    assert entry["size_bytes"] == SLCREP_H_SIZE
    assert entry["sha256"] == SLCREP_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    assert include_entries[0]["sha256"] == entry["sha256"]
    assert include_entries[0]["relative_path"] != entry["relative_path"]


RELATIVE_TOKEND_H = "securityd/src/tokend.h"
TOKEND_H_SIZE = 4299
TOKEND_H_SHA256 = (
    "eebdefb340ff183a9406b5d8785007406f3e1cfe81982631e6034e0f5b78ebab"
)


def test_apple_security_tokend_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "tokend.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "tokend.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKEND_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security tokend.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_tokend_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "tokend.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKEND_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_tokend_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "tokend.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == TOKEND_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == TOKEND_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_TOKEND_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == TOKEND_H_SIZE
    assert entry["sha256"] == TOKEND_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_PROCESS_H = "securityd/src/process.h"
PROCESS_H_SIZE = 3878
PROCESS_H_SHA256 = (
    "e135e6ee4c9f9d3328eaec4b80379c505094bb37eae4b5f9c0ae0ba14a9d53d4"
)


def test_apple_security_process_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "process.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "process.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_PROCESS_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security process.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_process_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "process.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_PROCESS_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_process_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "process.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == PROCESS_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == PROCESS_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_PROCESS_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == PROCESS_H_SIZE
    assert entry["sha256"] == PROCESS_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_LOCALDATABASE_H = "securityd/src/localdatabase.h"
LOCALDATABASE_H_SIZE = 4036
LOCALDATABASE_H_SHA256 = (
    "039005732bbbc7ef7e88c52f40f80926e2b13088bde16138a15188ffe68e430c"
)


def test_apple_security_localdatabase_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "localdatabase.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "localdatabase.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_LOCALDATABASE_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security localdatabase.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_localdatabase_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "localdatabase.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_LOCALDATABASE_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_localdatabase_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "localdatabase.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == LOCALDATABASE_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == LOCALDATABASE_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_LOCALDATABASE_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == LOCALDATABASE_H_SIZE
    assert entry["sha256"] == LOCALDATABASE_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_BLOCKCRYPTOR_H = "OSX/libsecurity_apple_csp/lib/BlockCryptor.h"
BLOCKCRYPTOR_H_SIZE = 6828
BLOCKCRYPTOR_H_SHA256 = (
    "e7083bad4d583fbec21a29a0bae6f2264d95ce041df3cc9ae4b5a2a0f78f516f"
)
BLOCKCRYPTOR_FORMAT = (
    "apple_security_blockcryptor_macroman_nbsp_libclang_timeout"
)


def test_apple_security_blockcryptor_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "BlockCryptor.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "BlockCryptor.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_BLOCKCRYPTOR_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=BLOCKCRYPTOR_FORMAT,
        reason="Security BlockCryptor.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_blockcryptor_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "BlockCryptor.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_BLOCKCRYPTOR_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=BLOCKCRYPTOR_FORMAT,
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="MacRoman NBSP identity is missing"
    ):
        _verify_detected_format(path, entry)


def test_apple_security_blockcryptor_rejects_utf8_nbsp_stripped_copy(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "BlockCryptor.h"
    )
    payload = fixture.read_bytes().replace(b"\xca", b" ")
    path = tmp_path / "BlockCryptor.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_BLOCKCRYPTOR_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=BLOCKCRYPTOR_FORMAT,
        reason="negative test stripped 0xCA",
    )
    with pytest.raises(
        SourceQuarantineError, match="MacRoman NBSP identity is missing"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_blockcryptor_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "BlockCryptor.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == BLOCKCRYPTOR_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == BLOCKCRYPTOR_H_SHA256
    assert payload.count(b"\xca") == 1
    assert payload[4318] == 0xCA
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_BLOCKCRYPTOR_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == BLOCKCRYPTOR_H_SIZE
    assert entry["sha256"] == BLOCKCRYPTOR_H_SHA256
    assert entry["detected_format"] == BLOCKCRYPTOR_FORMAT
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_TOKENDATABASE_H = "securityd/src/tokendatabase.h"
TOKENDATABASE_H_SIZE = 8749
TOKENDATABASE_H_SHA256 = (
    "b2935733193851992d322316940963cdbae3818126666d70ec7eb9674f012c20"
)


def test_apple_security_tokendatabase_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "tokendatabase.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "tokendatabase.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKENDATABASE_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security tokendatabase.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_tokendatabase_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "tokendatabase.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKENDATABASE_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_tokendatabase_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "tokendatabase.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == TOKENDATABASE_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == TOKENDATABASE_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_TOKENDATABASE_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == TOKENDATABASE_H_SIZE
    assert entry["sha256"] == TOKENDATABASE_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    tokend = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == "securityd/src/tokend.h"
    ]
    assert len(tokend) == 1
    assert tokend[0]["sha256"] != entry["sha256"]


RELATIVE_TOKENKEY_H = "securityd/src/tokenkey.h"
TOKENKEY_H_SIZE = 1957
TOKENKEY_H_SHA256 = (
    "2b40f19d3ed954ef8e3141397be3dbd1b96882e24ee920f702682a73caffae16"
)


def test_apple_security_tokenkey_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "tokenkey.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "tokenkey.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKENKEY_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security tokenkey.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_tokenkey_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "tokenkey.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKENKEY_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_tokenkey_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "tokenkey.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == TOKENKEY_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == TOKENKEY_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_TOKENKEY_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == TOKENKEY_H_SIZE
    assert entry["sha256"] == TOKENKEY_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e.get("relative_path")
        for e in manifest["entries"]
        if e.get("relative_path")
        in {
            "securityd/src/key.h",
            "securityd/src/kckey.h",
            "OSX/libsecurity_apple_cspdl/lib/SSKey.h",
        }
    ]
    assert len(siblings) == 3
    assert entry["sha256"] not in {
        e["sha256"]
        for e in manifest["entries"]
        if e.get("relative_path")
        in {
            "securityd/src/key.h",
            "securityd/src/kckey.h",
            "OSX/libsecurity_apple_cspdl/lib/SSKey.h",
        }
    }


RELATIVE_INPUTBUFFER_HPP = "OSX/libsecurity_codesigning/antlr2/antlr/InputBuffer.hpp"
INPUTBUFFER_HPP_SIZE = 3509
INPUTBUFFER_HPP_SHA256 = (
    "334401734869ccc3021d825449448511f7c12a8cfa4bd8d20258e3d440b3455c"
)
INPUTBUFFER_FORMAT = "antlr_inputbuffer_hpp_libclang_timeout"


def test_antlr_inputbuffer_hpp_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "InputBuffer.hpp"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "InputBuffer.hpp"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_INPUTBUFFER_HPP,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=INPUTBUFFER_FORMAT,
        reason="Security ANTLR InputBuffer.hpp libclang hang",
    )
    _verify_detected_format(path, entry)


def test_antlr_inputbuffer_hpp_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "InputBuffer.hpp"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_INPUTBUFFER_HPP,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=INPUTBUFFER_FORMAT,
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="ANTLR InputBuffer.hpp contract"
    ):
        _verify_detected_format(path, entry)


def test_antlr_inputbuffer_hpp_rejects_apple_security_header_format(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "InputBuffer.hpp"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "InputBuffer.hpp"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_INPUTBUFFER_HPP,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="wrong format for ANTLR hpp",
    )
    with pytest.raises(
        SourceQuarantineError,
        match="Apple Security libclang-timeout header contract",
    ):
        _verify_detected_format(path, entry)


def test_checked_in_antlr_inputbuffer_hpp_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "InputBuffer.hpp"
    )
    payload = fixture.read_bytes()
    assert len(payload) == INPUTBUFFER_HPP_SIZE
    assert hashlib.sha256(payload).hexdigest() == INPUTBUFFER_HPP_SHA256
    assert b"Apple Inc." not in payload
    assert b"Apple Computer, Inc." not in payload
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_INPUTBUFFER_HPP
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == INPUTBUFFER_HPP_SIZE
    assert entry["sha256"] == INPUTBUFFER_HPP_SHA256
    assert entry["detected_format"] == INPUTBUFFER_FORMAT
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_BASEAST_HPP = "OSX/libsecurity_codesigning/antlr2/antlr/BaseAST.hpp"
BASEAST_HPP_SIZE = 4658
BASEAST_HPP_SHA256 = (
    "76e10e87e01f5d41502a925f37a31c03eaaee47c3f5e4b7c90c2764df1605d79"
)
BASEAST_FORMAT = "antlr_baseast_hpp_libclang_timeout"


def test_antlr_baseast_hpp_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "BaseAST.hpp"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "BaseAST.hpp"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_BASEAST_HPP,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=BASEAST_FORMAT,
        reason="Security ANTLR BaseAST.hpp libclang hang",
    )
    _verify_detected_format(path, entry)


def test_antlr_baseast_hpp_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "BaseAST.hpp"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_BASEAST_HPP,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=BASEAST_FORMAT,
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="ANTLR BaseAST.hpp contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_antlr_baseast_hpp_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "BaseAST.hpp"
    )
    payload = fixture.read_bytes()
    assert len(payload) == BASEAST_HPP_SIZE
    assert hashlib.sha256(payload).hexdigest() == BASEAST_HPP_SHA256
    assert b"Apple Inc." not in payload
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_BASEAST_HPP
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == BASEAST_HPP_SIZE
    assert entry["sha256"] == BASEAST_HPP_SHA256
    assert entry["detected_format"] == BASEAST_FORMAT
    assert entry["project_id"] == "apple-oss-distributions/Security"
    ib = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_INPUTBUFFER_HPP
    ]
    assert len(ib) == 1
    assert ib[0]["sha256"] != entry["sha256"]


RELATIVE_CHARSCANNER_HPP = "OSX/libsecurity_codesigning/antlr2/antlr/CharScanner.hpp"
CHARSCANNER_HPP_SIZE = 13780
CHARSCANNER_HPP_SHA256 = (
    "ce437b7a257d90fd3c33bcc4b9ede24f1a72c104e9fc7a7ad7f2bd8c9622de9d"
)
CHARSCANNER_FORMAT = "antlr_charscanner_hpp_libclang_timeout"


def test_antlr_charscanner_hpp_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "CharScanner.hpp"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "CharScanner.hpp"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_CHARSCANNER_HPP,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=CHARSCANNER_FORMAT,
        reason="Security ANTLR CharScanner.hpp libclang hang",
    )
    _verify_detected_format(path, entry)


def test_antlr_charscanner_hpp_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "CharScanner.hpp"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_CHARSCANNER_HPP,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=CHARSCANNER_FORMAT,
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="ANTLR CharScanner.hpp contract"
    ):
        _verify_detected_format(path, entry)


def test_antlr_charscanner_hpp_rejects_apple_security_header_format(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "CharScanner.hpp"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "CharScanner.hpp"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_CHARSCANNER_HPP,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="wrong format for ANTLR hpp",
    )
    with pytest.raises(
        SourceQuarantineError,
        match="Apple Security libclang-timeout header contract",
    ):
        _verify_detected_format(path, entry)


def test_checked_in_antlr_charscanner_hpp_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "CharScanner.hpp"
    )
    payload = fixture.read_bytes()
    assert len(payload) == CHARSCANNER_HPP_SIZE
    assert hashlib.sha256(payload).hexdigest() == CHARSCANNER_HPP_SHA256
    assert b"Apple Inc." not in payload
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_CHARSCANNER_HPP
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == CHARSCANNER_HPP_SIZE
    assert entry["sha256"] == CHARSCANNER_HPP_SHA256
    assert entry["detected_format"] == CHARSCANNER_FORMAT
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e
        for e in manifest["entries"]
        if e.get("relative_path")
        in {RELATIVE_INPUTBUFFER_HPP, RELATIVE_BASEAST_HPP}
    ]
    assert len(siblings) == 2
    assert entry["sha256"] not in {e["sha256"] for e in siblings}


RELATIVE_TOKENBUFFER_HPP = "OSX/libsecurity_codesigning/antlr2/antlr/TokenBuffer.hpp"
TOKENBUFFER_HPP_SIZE = 2895
TOKENBUFFER_HPP_SHA256 = (
    "a697e25fe54e75383d125bf091ec1285da67e69557305e80d8bd6d09ee7bb8d5"
)
TOKENBUFFER_FORMAT = "antlr_tokenbuffer_hpp_libclang_timeout"


def test_antlr_tokenbuffer_hpp_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "TokenBuffer.hpp"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "TokenBuffer.hpp"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKENBUFFER_HPP,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=TOKENBUFFER_FORMAT,
        reason="Security ANTLR TokenBuffer.hpp libclang hang",
    )
    _verify_detected_format(path, entry)


def test_antlr_tokenbuffer_hpp_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "TokenBuffer.hpp"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKENBUFFER_HPP,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=TOKENBUFFER_FORMAT,
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="ANTLR TokenBuffer.hpp contract"
    ):
        _verify_detected_format(path, entry)


def test_antlr_tokenbuffer_hpp_rejects_apple_security_header_format(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "TokenBuffer.hpp"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "TokenBuffer.hpp"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKENBUFFER_HPP,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="wrong format for ANTLR hpp",
    )
    with pytest.raises(
        SourceQuarantineError,
        match="Apple Security libclang-timeout header contract",
    ):
        _verify_detected_format(path, entry)


def test_checked_in_antlr_tokenbuffer_hpp_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "TokenBuffer.hpp"
    )
    payload = fixture.read_bytes()
    assert len(payload) == TOKENBUFFER_HPP_SIZE
    assert hashlib.sha256(payload).hexdigest() == TOKENBUFFER_HPP_SHA256
    assert b"Apple Inc." not in payload
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_TOKENBUFFER_HPP
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == TOKENBUFFER_HPP_SIZE
    assert entry["sha256"] == TOKENBUFFER_HPP_SHA256
    assert entry["detected_format"] == TOKENBUFFER_FORMAT
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e
        for e in manifest["entries"]
        if e.get("relative_path")
        in {
            RELATIVE_INPUTBUFFER_HPP,
            RELATIVE_BASEAST_HPP,
            RELATIVE_CHARSCANNER_HPP,
        }
    ]
    assert len(siblings) == 3
    assert entry["sha256"] not in {e["sha256"] for e in siblings}


RELATIVE_LOCALKEY_H = "securityd/src/localkey.h"
LOCALKEY_H_SIZE = 4755
LOCALKEY_H_SHA256 = (
    "d54ea5e7a90b72f27118591ec98eedf613ae02d246dfdf32c12b3cbabae96c98"
)
LOCALKEY_SIBLING_PATHS = {
    "securityd/src/key.h",
    "securityd/src/kckey.h",
    "OSX/libsecurity_apple_cspdl/lib/SSKey.h",
    "securityd/src/tokenkey.h",
}


def test_apple_security_localkey_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "localkey.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "localkey.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_LOCALKEY_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security localkey.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_localkey_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "localkey.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_LOCALKEY_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_localkey_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "localkey.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == LOCALKEY_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == LOCALKEY_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_LOCALKEY_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == LOCALKEY_H_SIZE
    assert entry["sha256"] == LOCALKEY_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") in LOCALKEY_SIBLING_PATHS
    ]
    assert len(siblings) == 4
    assert entry["sha256"] not in {e["sha256"] for e in siblings}


RELATIVE_DESCONTEXT_H = "OSX/libsecurity_apple_csp/lib/desContext.h"
DESCONTEXT_H_SIZE = 3007
DESCONTEXT_H_SHA256 = (
    "81f0ef1905bc85a74565cb71cc64fb07c138d130955288dd66826967a06e9384"
)
DESCONTEXT_SIBLING_PATHS = {
    "OSX/libsecurity_apple_csp/lib/BlockCryptor.h",
    "OSX/libsecurity_apple_cspdl/lib/SSContext.h",
    "OSX/libsecurity_sd_cspdl/lib/SDContext.h",
}


def test_apple_security_descontext_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "desContext.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "desContext.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_DESCONTEXT_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security desContext.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_descontext_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "desContext.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_DESCONTEXT_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_descontext_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "desContext.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == DESCONTEXT_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == DESCONTEXT_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_DESCONTEXT_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == DESCONTEXT_H_SIZE
    assert entry["sha256"] == DESCONTEXT_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") in DESCONTEXT_SIBLING_PATHS
    ]
    assert len(siblings) == 3
    assert entry["sha256"] not in {e["sha256"] for e in siblings}


RELATIVE_TEMPDATABASE_H = "securityd/src/tempdatabase.h"
TEMPDATABASE_H_SIZE = 2350
TEMPDATABASE_H_SHA256 = (
    "dc3d63f5de38e2b76e8742408325e89ef175da7f5ed0135fb8ee86bc7085f4f6"
)
TEMPDATABASE_SIBLING_PATHS = {
    "securityd/src/localdatabase.h",
    "securityd/src/tokendatabase.h",
}


def test_apple_security_tempdatabase_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "tempdatabase.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "tempdatabase.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TEMPDATABASE_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security tempdatabase.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_tempdatabase_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "tempdatabase.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TEMPDATABASE_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_tempdatabase_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "tempdatabase.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == TEMPDATABASE_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == TEMPDATABASE_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_TEMPDATABASE_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == TEMPDATABASE_H_SIZE
    assert entry["sha256"] == TEMPDATABASE_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") in TEMPDATABASE_SIBLING_PATHS
    ]
    assert len(siblings) == 2
    assert entry["sha256"] not in {e["sha256"] for e in siblings}


RELATIVE_READER_H = "securityd/src/reader.h"
READER_H_SIZE = 2247
READER_H_SHA256 = (
    "bbb74ca697faa371f44778272931781d230f577ab97f6ccccd4cb648dfc085c7"
)


def test_apple_security_reader_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "reader.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "reader.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_READER_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security reader.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_reader_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "reader.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_READER_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_reader_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "reader.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == READER_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == READER_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_READER_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == READER_H_SIZE
    assert entry["sha256"] == READER_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"


RELATIVE_APPLECSPCONTEXT_H = "OSX/libsecurity_apple_csp/lib/AppleCSPContext.h"
APPLECSPCONTEXT_H_SIZE = 5431
APPLECSPCONTEXT_H_SHA256 = (
    "da24e598006d8b24efceedecce674e2c809e9574f019b72af50efaff4ac2150e"
)
APPLECSPCONTEXT_SIBLING_PATHS = {
    "OSX/libsecurity_apple_csp/lib/BlockCryptor.h",
    "OSX/libsecurity_apple_cspdl/lib/SSContext.h",
    "OSX/libsecurity_sd_cspdl/lib/SDContext.h",
    "OSX/libsecurity_apple_csp/lib/desContext.h",
}


def test_apple_security_applecspcontext_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "AppleCSPContext.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "AppleCSPContext.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_APPLECSPCONTEXT_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security AppleCSPContext.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_applecspcontext_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "AppleCSPContext.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_APPLECSPCONTEXT_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_applecspcontext_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "AppleCSPContext.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == APPLECSPCONTEXT_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == APPLECSPCONTEXT_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_APPLECSPCONTEXT_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == APPLECSPCONTEXT_H_SIZE
    assert entry["sha256"] == APPLECSPCONTEXT_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") in APPLECSPCONTEXT_SIBLING_PATHS
    ]
    assert len(siblings) == 4
    assert entry["sha256"] not in {e["sha256"] for e in siblings}


RELATIVE_DH_KEYS_H = "OSX/libsecurity_apple_csp/lib/DH_keys.h"
DH_KEYS_H_SIZE = 4280
DH_KEYS_H_SHA256 = (
    "fda77a2ee6b9821fb8067f830e298f3aa4c473bd5409353fab896bca3e389b96"
)
DH_KEYS_FORMAT = "apple_security_dh_keys_macroman_nbsp_libclang_timeout"
DH_KEYS_SIBLING_PATHS = {
    "OSX/libsecurity_apple_csp/lib/AppleCSPContext.h",
    "OSX/libsecurity_apple_csp/lib/BlockCryptor.h",
    "OSX/libsecurity_apple_csp/lib/desContext.h",
}


def test_apple_security_dh_keys_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "DH_keys.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "DH_keys.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_DH_KEYS_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=DH_KEYS_FORMAT,
        reason="Security DH_keys.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_dh_keys_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "DH_keys.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_DH_KEYS_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=DH_KEYS_FORMAT,
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="MacRoman NBSP identity is missing"
    ):
        _verify_detected_format(path, entry)


def test_apple_security_dh_keys_rejects_utf8_nbsp_stripped_copy(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "DH_keys.h"
    )
    payload = fixture.read_bytes().replace(b"\xca", b" ")
    path = tmp_path / "DH_keys.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_DH_KEYS_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format=DH_KEYS_FORMAT,
        reason="negative test stripped 0xCA",
    )
    with pytest.raises(
        SourceQuarantineError, match="MacRoman NBSP identity is missing"
    ):
        _verify_detected_format(path, entry)


def test_apple_security_dh_keys_utf8_header_format_raises(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "DH_keys.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "DH_keys.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_DH_KEYS_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="utf-8 format must reject non-UTF-8 DH_keys.h",
    )
    with pytest.raises(SourceQuarantineError, match="header is not UTF-8"):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_dh_keys_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "DH_keys.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == DH_KEYS_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == DH_KEYS_H_SHA256
    assert payload.count(b"\xca") == 1
    assert payload[2825] == 0xCA
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_DH_KEYS_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == DH_KEYS_H_SIZE
    assert entry["sha256"] == DH_KEYS_H_SHA256
    assert entry["detected_format"] == DH_KEYS_FORMAT
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") in DH_KEYS_SIBLING_PATHS
    ]
    assert len(siblings) == 3
    assert entry["sha256"] not in {e["sha256"] for e in siblings}


RELATIVE_PCSCMONITOR_H = "securityd/src/pcscmonitor.h"
PCSCMONITOR_H_SIZE = 2855
PCSCMONITOR_H_SHA256 = (
    "88e24d2a61409b167381714d1de998b764bb1dacbca5f1794003e6f07acd34b8"
)
PCSCMONITOR_SIBLING_PATHS = {
    "securityd/src/reader.h",
    "securityd/src/tokend.h",
    "securityd/src/process.h",
}


def test_apple_security_pcscmonitor_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "pcscmonitor.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "pcscmonitor.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_PCSCMONITOR_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security pcscmonitor.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_pcscmonitor_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "pcscmonitor.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_PCSCMONITOR_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_pcscmonitor_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "pcscmonitor.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == PCSCMONITOR_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == PCSCMONITOR_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_PCSCMONITOR_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == PCSCMONITOR_H_SIZE
    assert entry["sha256"] == PCSCMONITOR_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") in PCSCMONITOR_SIBLING_PATHS
    ]
    assert len(siblings) == 3
    assert entry["sha256"] not in {e["sha256"] for e in siblings}


RELATIVE_SERVER_H = "securityd/src/server.h"
SERVER_H_SIZE = 7044
SERVER_H_SHA256 = (
    "6ca9d0f5e77a6a46e0ca1c14d099288ada5a28d9920163331de6b6240286bbb4"
)
SERVER_SIBLING_PATHS = {
    "securityd/src/connection.h",
    "securityd/src/process.h",
    "securityd/src/structure.h",
    "securityd/src/pcscmonitor.h",
}


def test_apple_security_server_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "server.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "server.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SERVER_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security server.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_server_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "server.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_SERVER_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_server_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "server.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == SERVER_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == SERVER_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_SERVER_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == SERVER_H_SIZE
    assert entry["sha256"] == SERVER_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") in SERVER_SIBLING_PATHS
    ]
    assert len(siblings) == 4
    assert entry["sha256"] not in {e["sha256"] for e in siblings}


RELATIVE_TOKEN_H = "securityd/src/token.h"
TOKEN_H_SIZE = 3663
TOKEN_H_SHA256 = (
    "78d1f5cf0bd4c8546b178ac6d8bef2ea594dc854ea3556fa71a3f52b6a943445"
)
TOKEN_SIBLING_PATHS = {
    "securityd/src/tokend.h",
    "securityd/src/tokenkey.h",
    "securityd/src/tokendatabase.h",
    "securityd/src/reader.h",
}


def test_apple_security_token_libclang_timeout_accepts_header(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "token.h"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "token.h"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKEN_H,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="Security token.h libclang hang",
    )
    _verify_detected_format(path, entry)


def test_apple_security_token_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "token.h"
    path.write_text("#pragma once\nclass X {};\n", encoding="utf-8")
    entry = SourceQuarantineEntry(
        project_id="apple-oss-distributions/Security",
        relative_path=RELATIVE_TOKEN_H,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="apple_security_libclang_timeout_header",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="Apple Security libclang-timeout header contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_apple_security_token_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "token.h"
    )
    payload = fixture.read_bytes()
    assert len(payload) == TOKEN_H_SIZE
    assert hashlib.sha256(payload).hexdigest() == TOKEN_H_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_TOKEN_H
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == TOKEN_H_SIZE
    assert entry["sha256"] == TOKEN_H_SHA256
    assert entry["detected_format"] == "apple_security_libclang_timeout_header"
    assert entry["project_id"] == "apple-oss-distributions/Security"
    siblings = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") in TOKEN_SIBLING_PATHS
    ]
    assert len(siblings) == 4
    assert entry["sha256"] not in {e["sha256"] for e in siblings}


RELATIVE_GLIBC_BUG28 = "stdio-common/bug28.c"
GLIBC_BUG28_SIZE = 1216
GLIBC_BUG28_SHA256 = (
    "6cd92333ab2c85a993dd690ccd74cfb8a8014e8160b57feabefb89e2f7e7644b"
)


def test_glibc_stdio_bug28_libclang_timeout_accepts_fixture(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        _verify_detected_format,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "bug28.c"
    )
    payload = fixture.read_bytes()
    path = tmp_path / "bug28.c"
    path.write_bytes(payload)
    entry = SourceQuarantineEntry(
        project_id="sourceware.org/git%2Fglibc",
        relative_path=RELATIVE_GLIBC_BUG28,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="glibc_stdio_bug28_libclang_timeout",
        reason="glibc bug28.c libclang hang",
    )
    _verify_detected_format(path, entry)


def test_glibc_stdio_bug28_contract_rejects_unrelated_standin(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.source_quarantine import (
        SourceQuarantineEntry,
        SourceQuarantineError,
        _verify_detected_format,
    )

    path = tmp_path / "bug28.c"
    path.write_text("int main(void) { return 0; }\n", encoding="ascii")
    entry = SourceQuarantineEntry(
        project_id="sourceware.org/git%2Fglibc",
        relative_path=RELATIVE_GLIBC_BUG28,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        classification="compiler_regression_fixture",
        detected_format="glibc_stdio_bug28_libclang_timeout",
        reason="negative test",
    )
    with pytest.raises(
        SourceQuarantineError, match="glibc stdio-common/bug28.c contract"
    ):
        _verify_detected_format(path, entry)


def test_checked_in_glibc_bug28_manifest_matches_pinned_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "source_quarantine"
        / "bug28.c"
    )
    payload = fixture.read_bytes()
    assert len(payload) == GLIBC_BUG28_SIZE
    assert hashlib.sha256(payload).hexdigest() == GLIBC_BUG28_SHA256
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        e
        for e in manifest["entries"]
        if e.get("relative_path") == RELATIVE_GLIBC_BUG28
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["size_bytes"] == GLIBC_BUG28_SIZE
    assert entry["sha256"] == GLIBC_BUG28_SHA256
    assert entry["detected_format"] == "glibc_stdio_bug28_libclang_timeout"
    assert entry["project_id"] == "sourceware.org/git%2Fglibc"
