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
