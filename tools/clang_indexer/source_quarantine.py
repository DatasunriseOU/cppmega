"""Exact, receipt-bound quarantine for non-parser inputs and compiler fixtures.

The quarantine is deliberately narrow: it only accepts files whose relative
path, byte size, SHA-256 digest, and independently verifiable format match an
exact entry or an exact content-set collection. It is not a parse-error
suppression mechanism.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

LEGACY_MANIFEST_SCHEMA = "cppmega.source_quarantine_manifest_v1"
MANIFEST_SCHEMA = "cppmega.source_quarantine_manifest_v2"
RECEIPT_SCHEMA = "cppmega.source_quarantine_receipt_v1"
_ENTRY_KEYS = frozenset(
    {
        "project_id",
        "relative_path",
        "size_bytes",
        "sha256",
        "classification",
        "detected_format",
        "reason",
    }
)
_COLLECTION_KEYS = frozenset(
    {
        "project_id",
        "relative_path_prefix",
        "relative_path_suffix",
        "expected_file_count",
        "content_set_sha256",
        "classification",
        "detected_format",
        "reason",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FORMAT_RE = re.compile(r"[a-z0-9][a-z0-9_.+-]*")
_SUPPORTED_CLASSIFICATION_FORMATS = {
    (
        "deliberate_compiler_crash_fixture",
        "clang_debug_crash_pragma",
    ),
    (
        "deliberate_compiler_crash_fixture",
        "clang_debug_parser_crash_pragma",
    ),
    (
        "compiler_regression_fixture",
        "gcc_c_flexible_array_union_initializer_regression",
    ),
    (
        "compiler_regression_fixture",
        "plumhall_c_date_time_libclang_hang",
    ),
    (
        "compiler_regression_fixture",
        "apple_security_ssdl_session_libclang_hang",
    ),
    (
        "compiler_regression_fixture",
        "apple_security_libclang_timeout_header",
    ),
    (
        "compiler_regression_fixture",
        "apple_security_blockcryptor_macroman_nbsp_libclang_timeout",
    ),
    (
        "compiler_regression_fixture",
        "apple_security_dh_keys_macroman_nbsp_libclang_timeout",
    ),
    (
        "compiler_regression_fixture",
        "apple_security_rsa_dsa_keys_macroman_nbsp_libclang_timeout",
    ),
    (
        "compiler_regression_fixture",
        "antlr_inputbuffer_hpp_libclang_timeout",
    ),
    (
        "compiler_regression_fixture",
        "antlr_baseast_hpp_libclang_timeout",
    ),
    (
        "compiler_regression_fixture",
        "antlr_charscanner_hpp_libclang_timeout",
    ),
    (
        "compiler_regression_fixture",
        "antlr_tokenbuffer_hpp_libclang_timeout",
    ),
    (
        "compiler_regression_fixture",
        "glibc_stdio_bug28_libclang_timeout",
    ),
    (
        "binary_protocol_test_fixture",
        "clickhouse_dollar_quoted_binary_sql",
    ),
    (
        "deliberate_compiler_diagnostic_fixture",
        "clang_embedded_nul_diagnostic",
    ),
    (
        "deliberate_compiler_diagnostic_fixture",
        "gcc_embedded_nul_diagnostic",
    ),
    (
        "deliberate_compiler_diagnostic_fixture",
        "clang_escaped_newline_nul_preprocessor_diagnostic",
    ),
    (
        "deliberate_parser_regression_fixture",
        "cmake_escaped_newline_nul_syntax_fixture",
    ),
    (
        "deliberate_parser_regression_fixture",
        "cmake_null_terminated_argument_fixture",
    ),
    (
        "deliberate_compiler_diagnostic_fixture",
        "clang_embedded_nul_in_literal",
    ),
    ("generated_binary_blob", "utf16le_generated_c_array"),
    ("generated_executable_archive", "posix_shell_appended_zip"),
    ("mislabeled_non_cpp", "xml_utf16le"),
    ("mislabeled_non_cpp", "utf16le_source_text"),
    ("mislabeled_non_cpp", "nul_ff_binary_blob"),
    ("mislabeled_non_cpp", "binary_blob_with_embedded_nul"),
    ("mislabeled_non_cpp", "asn1_der_x509_certificate_pair"),
    ("mislabeled_non_cpp", "truncated_utf32be_bom"),
    ("mislabeled_non_cpp", "truncated_utf32le_bom"),
    ("mislabeled_non_cpp", "big5_shell_heredoc"),
    (
        "deliberate_encoding_regression_fixture",
        "git_shortlog_invalid_utf8_shell",
    ),
    (
        "deliberate_encoding_regression_fixture",
        "invalid_utf8_and_windows1252_domain_blob",
    ),
}


class SourceQuarantineError(ValueError):
    """A quarantine manifest or candidate failed exact validation."""


@dataclass(frozen=True)
class SourceQuarantineEntry:
    project_id: str
    relative_path: str
    size_bytes: int
    sha256: str
    classification: str
    detected_format: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "classification": self.classification,
            "detected_format": self.detected_format,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SourceQuarantineCollection:
    project_id: str
    relative_path_prefix: str
    relative_path_suffix: str
    expected_file_count: int
    content_set_sha256: str
    classification: str
    detected_format: str
    reason: str


def _require_string(
    value: object,
    *,
    field: str,
    index: int,
    container: str = "entries",
) -> str:
    if not isinstance(value, str) or not value:
        raise SourceQuarantineError(
            f"{container}[{index}].{field} must be a non-empty string"
        )
    return value


def _parse_entry(raw: object, *, index: int) -> SourceQuarantineEntry:
    if not isinstance(raw, dict):
        raise SourceQuarantineError(f"entries[{index}] must be an object")
    keys = frozenset(raw)
    if keys != _ENTRY_KEYS:
        missing = sorted(_ENTRY_KEYS - keys)
        unknown = sorted(keys - _ENTRY_KEYS)
        raise SourceQuarantineError(
            f"entries[{index}] has invalid fields: missing={missing} unknown={unknown}"
        )

    project_id = _require_string(
        raw["project_id"],
        field="project_id",
        index=index,
    )
    relative_path = _require_string(
        raw["relative_path"],
        field="relative_path",
        index=index,
    )
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or relative_path != pure_path.as_posix()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or "\\" in relative_path
    ):
        raise SourceQuarantineError(
            f"entries[{index}].relative_path is not a canonical safe POSIX path: "
            f"{relative_path!r}"
        )

    size_bytes = raw["size_bytes"]
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise SourceQuarantineError(
            f"entries[{index}].size_bytes must be a non-negative integer"
        )
    sha256 = _require_string(raw["sha256"], field="sha256", index=index)
    if _SHA256_RE.fullmatch(sha256) is None:
        raise SourceQuarantineError(
            f"entries[{index}].sha256 must be 64 lowercase hexadecimal characters"
        )
    classification = _require_string(
        raw["classification"],
        field="classification",
        index=index,
    )
    detected_format = _require_string(
        raw["detected_format"],
        field="detected_format",
        index=index,
    )
    if _FORMAT_RE.fullmatch(detected_format) is None:
        raise SourceQuarantineError(
            f"entries[{index}].detected_format is invalid: {detected_format!r}"
        )
    if (classification, detected_format) not in _SUPPORTED_CLASSIFICATION_FORMATS:
        raise SourceQuarantineError(
            f"entries[{index}] unsupported quarantine classification/format: "
            f"{classification}/{detected_format}"
        )
    reason = _require_string(raw["reason"], field="reason", index=index)
    return SourceQuarantineEntry(
        project_id=project_id,
        relative_path=relative_path,
        size_bytes=size_bytes,
        sha256=sha256,
        classification=classification,
        detected_format=detected_format,
        reason=reason,
    )


def _parse_collection(raw: object, *, index: int) -> SourceQuarantineCollection:
    if not isinstance(raw, dict):
        raise SourceQuarantineError(f"collections[{index}] must be an object")
    keys = frozenset(raw)
    if keys != _COLLECTION_KEYS:
        missing = sorted(_COLLECTION_KEYS - keys)
        unknown = sorted(keys - _COLLECTION_KEYS)
        raise SourceQuarantineError(
            f"collections[{index}] has invalid fields: "
            f"missing={missing} unknown={unknown}"
        )

    project_id = _require_string(
        raw["project_id"],
        field="project_id",
        index=index,
        container="collections",
    )
    prefix = _require_string(
        raw["relative_path_prefix"],
        field="relative_path_prefix",
        index=index,
        container="collections",
    )
    pure_prefix = PurePosixPath(prefix.removesuffix("/"))
    if (
        not prefix.endswith("/")
        or pure_prefix.is_absolute()
        or not pure_prefix.parts
        or prefix != f"{pure_prefix.as_posix()}/"
        or any(part in {"", ".", ".."} for part in pure_prefix.parts)
        or "\\" in prefix
    ):
        raise SourceQuarantineError(
            f"collections[{index}].relative_path_prefix is not a canonical "
            f"safe POSIX directory prefix: {prefix!r}"
        )
    suffix = _require_string(
        raw["relative_path_suffix"],
        field="relative_path_suffix",
        index=index,
        container="collections",
    )
    if "/" in suffix or "\\" in suffix:
        raise SourceQuarantineError(
            f"collections[{index}].relative_path_suffix must not contain a "
            "path separator"
        )
    expected_file_count = raw["expected_file_count"]
    if (
        isinstance(expected_file_count, bool)
        or not isinstance(expected_file_count, int)
        or expected_file_count <= 0
    ):
        raise SourceQuarantineError(
            f"collections[{index}].expected_file_count must be a positive integer"
        )
    content_set_sha256 = _require_string(
        raw["content_set_sha256"],
        field="content_set_sha256",
        index=index,
        container="collections",
    )
    if _SHA256_RE.fullmatch(content_set_sha256) is None:
        raise SourceQuarantineError(
            f"collections[{index}].content_set_sha256 must be 64 lowercase "
            "hexadecimal characters"
        )
    classification = _require_string(
        raw["classification"],
        field="classification",
        index=index,
        container="collections",
    )
    detected_format = _require_string(
        raw["detected_format"],
        field="detected_format",
        index=index,
        container="collections",
    )
    if _FORMAT_RE.fullmatch(detected_format) is None:
        raise SourceQuarantineError(
            f"collections[{index}].detected_format is invalid: {detected_format!r}"
        )
    if (classification, detected_format) not in _SUPPORTED_CLASSIFICATION_FORMATS:
        raise SourceQuarantineError(
            f"collections[{index}] unsupported quarantine "
            f"classification/format: {classification}/{detected_format}"
        )
    reason = _require_string(
        raw["reason"],
        field="reason",
        index=index,
        container="collections",
    )
    return SourceQuarantineCollection(
        project_id=project_id,
        relative_path_prefix=prefix,
        relative_path_suffix=suffix,
        expected_file_count=expected_file_count,
        content_set_sha256=content_set_sha256,
        classification=classification,
        detected_format=detected_format,
        reason=reason,
    )


def _content_set_sha256(entries: Iterable[SourceQuarantineEntry]) -> str:
    rows = [
        [entry.relative_path, entry.size_bytes, entry.sha256]
        for entry in sorted(entries, key=lambda item: item.relative_path)
    ]
    payload = json.dumps(
        rows,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _der_tlv_bounds(payload: bytes, offset: int) -> tuple[int, int, int]:
    if offset + 2 > len(payload):
        raise ValueError("truncated DER tag or length")
    tag = payload[offset]
    first_length = payload[offset + 1]
    if first_length < 0x80:
        content_start = offset + 2
        length = first_length
    else:
        length_bytes = first_length & 0x7F
        if (
            length_bytes == 0
            or length_bytes > 4
            or offset + 2 + length_bytes > len(payload)
        ):
            raise ValueError("invalid DER long-form length")
        encoded_length = payload[offset + 2 : offset + 2 + length_bytes]
        if encoded_length[0] == 0:
            raise ValueError("non-minimal DER length")
        length = int.from_bytes(encoded_length, "big")
        if length < 0x80:
            raise ValueError("non-minimal DER long-form length")
        content_start = offset + 2 + length_bytes
    content_end = content_start + length
    if content_end > len(payload):
        raise ValueError("DER value exceeds input")
    return tag, content_start, content_end


def _verify_detected_format(path: Path, entry: SourceQuarantineEntry) -> None:
    if entry.detected_format == "truncated_utf32be_bom":
        payload = path.read_bytes()
        if payload != b"\x00\x00\xfe":
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared truncated_utf32be_bom but "
                "the payload is not exactly the three-byte UTF-32BE BOM prefix"
            )
        return

    if entry.detected_format == "truncated_utf32le_bom":
        payload = path.read_bytes()
        if payload != b"\xff\xfe\x00":
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared truncated_utf32le_bom but "
                "the payload is not exactly the three-byte UTF-32LE BOM prefix"
            )
        return

    if entry.detected_format == "cmake_escaped_newline_nul_syntax_fixture":
        payload = path.read_bytes()
        expected = b"A(" + (b"A" * 52) + b"\\\0\n(" + (b"A" * 54) + b"\n"
        if payload != expected:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "cmake_escaped_newline_nul_syntax_fixture but the exact "
                "two-line escaped-newline/NUL parser-test shape is absent"
            )
        return

    if entry.detected_format == "cmake_null_terminated_argument_fixture":
        payload = path.read_bytes()
        expected = (
            b"LIST(APPEND foo TEST\x000000000000000000000000000 )\n"
            b"CMAKE_HOST_SYSTEM_INFORMATION(RESULT bar QUERY HOSTNAME)\n"
        )
        if payload != expected:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "cmake_null_terminated_argument_fixture but the exact "
                "LIST(APPEND)/NUL argument fixture shape is absent"
            )
        return

    if entry.detected_format == "clang_embedded_nul_in_literal":
        payload = path.read_bytes()
        if payload.count(b"\0") < 1:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clang_embedded_nul_in_literal "
                "but the fixture contains no NUL bytes"
            )
        try:
            # Decode allowing NULs by replacing for structural checks.
            decoded = payload.replace(b"\0", b"").decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clang_embedded_nul_in_literal "
                f"but non-ASCII payload outside NULs: {exc}"
            ) from exc
        required_snippets = (
            "RUN: %clang_cc1",
            "null character",
            "expected-warning",
        )
        if any(snippet not in decoded for snippet in required_snippets):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clang_embedded_nul_in_literal "
                "but the literal-NUL diagnostic contract is incomplete"
            )
        return

    if entry.detected_format == "binary_blob_with_embedded_nul":
        payload = path.read_bytes()
        if not payload or b"\0" not in payload:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared binary_blob_with_embedded_nul "
                "but the payload is empty or has no embedded NUL bytes"
            )
        # Reject pure text UTF-8 without high-bit bytes — this format is for
        # non-source binary tables (e.g. Plan 9 code pages).
        if b"\0" in payload and all(32 <= b < 127 or b in (9, 10, 13, 0) for b in payload):
            # still OK if mostly printable with NULs; require at least one
            # high-bit or control-class non-text byte besides NUL/tab/lf/cr.
            if not any(b >= 128 or b < 9 for b in payload if b != 0):
                raise SourceQuarantineError(
                    f"{entry.relative_path}: declared binary_blob_with_embedded_nul "
                    "but the payload looks like plain text with NULs"
                )
        return

    if entry.detected_format == "gcc_embedded_nul_diagnostic":
        payload = path.read_bytes()
        expected_name = Path(entry.relative_path).name
        gcc_nul_contracts: dict[str, tuple[str, tuple[bytes, ...]]] = {
            "encoding-issues-bytes.c": (
                "gcc/testsuite/gcc.dg/encoding-issues-bytes.c",
                (
                    b'dg-options "-fdiagnostics-show-caret -fdiagnostics-escape-format=bytes"',
                    b'dg-warning "null character\\\\(s\\\\) ignored"',
                    b"Stray UTF-8 trailing byte:",
                    b"stray '.200' in program",
                    b"unknown escape sequence",
                    b"dg-begin-multiline-output",
                ),
            ),
            "encoding-issues-unicode.c": (
                "gcc/testsuite/gcc.dg/encoding-issues-unicode.c",
                (
                    b'dg-options "-fdiagnostics-show-caret -fdiagnostics-escape-format=unicode"',
                    b'dg-warning "null character\\\\(s\\\\) ignored"',
                    b"Stray UTF-8 trailing byte:",
                    b"stray '.200' in program",
                    b"unknown escape sequence",
                    b"<U+0000>",
                    b"dg-begin-multiline-output",
                ),
            ),
        }
        contract = gcc_nul_contracts.get(expected_name)
        if (
            contract is None
            or path.suffix.casefold() != ".c"
            or entry.relative_path != contract[0]
            or payload.count(b"\x00") != 1
            or payload.count(b"\x80") != 1
            or payload.count(b"\x01") != 1
            or any(snippet not in payload for snippet in contract[1])
            or payload.count(b"dg-begin-multiline-output") != 3
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared gcc_embedded_nul_diagnostic "
                "but the GCC encoding-issues DejaGNU contract is "
                "incomplete or ambiguous"
            )
        return

    if entry.detected_format == "clang_embedded_nul_diagnostic":
        payload = path.read_bytes()
        try:
            decoded = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clang_embedded_nul_diagnostic "
                f"but the fixture is not ASCII: {exc}"
            ) from exc
        lines = decoded.splitlines()
        source_line = "int x[sizeof\0int];"
        rendered_source_line = "// CHECK-NEXT: int x[sizeof<U+0000>int];"
        run_line = (
            "// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | "
            "FileCheck -strict-whitespace %s"
        )
        warning_line = "// CHECK: warning: null character ignored"
        caret_line = "// CHECK-NEXT:             ^"
        error_line = (
            "// CHECK: error: expected parentheses around type name in "
            "sizeof expression"
        )
        required_lines = {
            run_line,
            source_line,
            warning_line,
            rendered_source_line,
            caret_line,
            error_line,
            "// CHECK-NEXT:             (          )",
        }
        if (
            payload.count(b"\0") != 1
            or not required_lines.issubset(lines)
            or lines.count(run_line) != 1
            or lines.count(warning_line) != 1
            or lines.count(source_line) != 1
            or lines.count(rendered_source_line) != 2
            or lines.count(caret_line) != 2
            or lines.count(error_line) != 1
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clang_embedded_nul_diagnostic "
                "but the embedded-NUL diagnostic contract is incomplete or ambiguous"
            )
        return

    if (
        entry.detected_format
        == "clang_escaped_newline_nul_preprocessor_diagnostic"
    ):
        payload = path.read_bytes()
        try:
            decoded = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "clang_escaped_newline_nul_preprocessor_diagnostic but the "
                f"fixture is not ASCII: {exc}"
            ) from exc
        lines = decoded.splitlines()
        run_line = "// RUN: %clang_cc1 -E %s -verify"
        continuation_line = "# if 1 \\"
        nul_directive_line = (
            "\0#if something_else // expected-warning {{null character ignored}} "
            "expected-error {{not a valid binary operator}}"
        )
        trailing_error_line = (
            "#endif // expected-error {{#endif without #if}}"
        )
        required_lines = {
            run_line,
            continuation_line,
            nul_directive_line,
            "#error error",
            "#endif",
            trailing_error_line,
        }
        if (
            payload.count(b"\0") != 1
            or not required_lines.issubset(lines)
            or any(lines.count(line) != 1 for line in required_lines)
            or not nul_directive_line.startswith("\0#if")
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "clang_escaped_newline_nul_preprocessor_diagnostic but the "
                "escaped-newline/NUL diagnostic contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "nul_ff_binary_blob":
        seen_values: set[int] = set()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                values = set(chunk)
                if not values <= {0x00, 0xFF}:
                    raise SourceQuarantineError(
                        f"{entry.relative_path}: declared nul_ff_binary_blob but the "
                        "payload is not a non-empty mixture of only 0x00 and 0xff bytes"
                    )
                seen_values.update(values)
        if seen_values != {0x00, 0xFF}:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared nul_ff_binary_blob but the "
                "payload is not a non-empty mixture of only 0x00 and 0xff bytes"
            )
        return

    if entry.detected_format == "xml_utf16le":
        with path.open("rb") as source:
            prefix = source.read(8192)
        if not prefix.startswith(b"\xff\xfe"):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared xml_utf16le but UTF-16LE BOM "
                "is absent"
            )
        if len(prefix) % 2:
            prefix = prefix[:-1]
        try:
            decoded = prefix.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared xml_utf16le but prefix is "
                f"invalid: {exc}"
            ) from exc
        if not decoded.lstrip("\ufeff \t\r\n").startswith(("<", "<?xml")):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared xml_utf16le but XML prefix is absent"
            )
        return

    if entry.detected_format == "utf16le_source_text":
        with path.open("rb") as source:
            prefix = source.read(8192)
        if not prefix.startswith(b"\xff\xfe"):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared utf16le_source_text but UTF-16LE BOM "
                "is absent"
            )
        if len(prefix) % 2:
            prefix = prefix[:-1]
        try:
            decoded = prefix.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared utf16le_source_text but prefix is "
                f"invalid: {exc}"
            ) from exc
        body = decoded.lstrip("\ufeff \t\r\n")
        starts_ok = body.startswith(
            (
                "//",
                "/*",
                "#",
                "using ",
                "namespace ",
                "struct ",
                "class ",
                "int ",
                "void ",
            )
        ) or any(ch.isalpha() for ch in body[:64])
        if not starts_ok:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared utf16le_source_text but decoded "
                "prefix is not C/C++-like text"
            )
        full = path.read_bytes()
        if b"\x00" not in full:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared utf16le_source_text but payload "
                "has no NUL bytes"
            )
        return

    if entry.detected_format == "clang_debug_crash_pragma":
        payload = path.read_bytes()
        try:
            decoded = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clang_debug_crash_pragma "
                f"but the fixture is not ASCII: {exc}"
            ) from exc
        lines = decoded.splitlines()
        compiler_contract = {
            "// RUN: not --crash %clang_cc1 %s 2>&1 | FileCheck %s",
            "// REQUIRES: crash-recovery",
            "// CHECK: prag\\",
            "// CHECK-NEXT: ma",
        }.issubset(lines) and decoded.count("#prag\\\nma clang __debug crash\n") == 1
        index_contract = {
            "// RUN: not c-index-test -test-load-source all %s 2> %t.err",
            "// RUN: FileCheck < %t.err -check-prefix=CHECK-LOAD-SOURCE-CRASH %s",
            "// CHECK-LOAD-SOURCE-CRASH: Unable to load translation unit",
            (
                "// RUN: env LIBCLANG_DISABLE_CRASH_RECOVERY=1 not --crash "
                "c-index-test -test-load-source all %s"
            ),
            "// REQUIRES: crash-recovery",
            "#pragma clang __debug crash",
        }.issubset(lines) and lines.count("#pragma clang __debug crash") == 1
        remap_contract = {
            "// RUN: echo env CINDEXTEST_EDITING=1 \\",
            "// RUN:   not c-index-test -test-load-source-reparse 1 local \\",
            (
                '// RUN:   -remap-file="%s,%S/Inputs/'
                'crash-recovery-code-complete-remap.c" \\'
            ),
            "// RUN:   %s 2> %t.err",
            ("// RUN: FileCheck < %t.err -check-prefix=CHECK-CODE-COMPLETE-CRASH %s"),
            ("// CHECK-CODE-COMPLETE-CRASH: Unable to reparse translation unit"),
            "#pragma clang __debug crash",
        }.issubset(lines) and lines.count("#pragma clang __debug crash") == 1
        if not (compiler_contract or index_contract or remap_contract):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clang_debug_crash_pragma "
                "but the Clang crash-test contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "clang_debug_parser_crash_pragma":
        payload = path.read_bytes()
        try:
            decoded = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clang_debug_parser_crash_pragma "
                f"but the fixture is not ASCII: {exc}"
            ) from exc
        lines = decoded.splitlines()
        required_lines = {
            "// REQUIRES: crash-recovery",
            "#pragma clang __debug parser_crash",
            "FOO",
            "// CHECKSRC: FOO",
            '// CHECKSH: "-cc1"',
        }
        crash_runs = [
            line
            for line in lines
            if line.startswith("// RUN: not ")
            and "%clang" in line
            and "-fsyntax-only" in line
            and "FileCheck" in line
        ]
        if (
            not required_lines.issubset(lines)
            or len(crash_runs) != 1
            or lines.count("#pragma clang __debug parser_crash") != 1
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "clang_debug_parser_crash_pragma but the Clang parser-crash "
                "test contract is incomplete or ambiguous"
            )
        return

    if (
        entry.detected_format
        == "gcc_c_flexible_array_union_initializer_regression"
    ):
        payload = path.read_bytes()
        try:
            decoded = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "gcc_c_flexible_array_union_initializer_regression but the "
                f"fixture is not ASCII: {exc}"
            ) from exc
        lines = decoded.splitlines()
        required_lines = {
            "/* PR c/119001 */",
            "/* { dg-do run } */",
            '/* { dg-options "" } */',
            "union U { char a[]; int i; };",
            "union U u = { \"12345\" };",
            "union U v = { .a = \"6789\" };",
            "union U w = { { 1, 2, 3, 4, 5, 6 } };",
            "union U x = { .a = { 7, 8, 9 } };",
            "union V { int i; char a[]; };",
            "union V y = { .a = \"abcdefghijk\" };",
            "union V z = { .a = { 10, 11, 12, 13, 14, 15, 16, 17 } };",
        }
        if (
            path.suffix.casefold() != ".c"
            or not required_lines.issubset(lines)
            or decoded.count("__builtin_abort ();") != 6
            or decoded.count("char a[];") != 2
            or decoded.count("int\nmain ()\n{") != 1
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "gcc_c_flexible_array_union_initializer_regression but the "
                "GCC PR119001 DejaGNU run-test contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "clickhouse_dollar_quoted_binary_sql":
        payload = path.read_bytes()
        if path.suffix.casefold() != ".sql" or payload.count(b"$$") != 2:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clickhouse_dollar_quoted_binary_sql "
                "but the SQL dollar-quoted payload boundary is absent or ambiguous"
            )
        payload_start = payload.index(b"$$")
        payload_end = payload.index(b"$$", payload_start + 2)
        sql_prefix = payload[:payload_start]
        binary_payload = payload[payload_start + 2 : payload_end]
        sql_suffix = payload[payload_end + 2 :]
        try:
            decoded_prefix = sql_prefix.decode("utf-8")
            decoded_suffix = sql_suffix.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clickhouse_dollar_quoted_binary_sql "
                f"but its SQL envelope is not UTF-8/ASCII: {exc}"
            ) from exc
        format_match = re.match(
            r"\A(?:[ \t]*(?:--[^\r\n]*)?\r?\n)*[ \t]*"
            r"SELECT\s+\*\s+FROM\s+format\s*\(\s*"
            r"((?:'(?:Native|BSONEachRow)')|(?:Native|BSONEachRow))\s*,",
            decoded_prefix,
            flags=re.IGNORECASE,
        )
        error_match = re.fullmatch(
            r"\s*\)\s*;\s*--\s*\{\s*serverError\s+"
            r"(?P<errors>"
            r"(?:TOO_LARGE_STRING_SIZE|TOO_LARGE_ARRAY_SIZE|INCORRECT_DATA|"
            r"UNKNOWN_TYPE|CANNOT_READ_ALL_DATA)"
            r"(?:\s*,\s*(?:TOO_LARGE_STRING_SIZE|TOO_LARGE_ARRAY_SIZE|"
            r"INCORRECT_DATA|UNKNOWN_TYPE|CANNOT_READ_ALL_DATA))*"
            r")"
            r"\s*\}\s*",
            decoded_suffix,
        )
        if (
            format_match is None
            or not sql_prefix.rstrip().endswith(b",")
            or not binary_payload
            or b"\0" not in binary_payload
            or error_match is None
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clickhouse_dollar_quoted_binary_sql "
                "but the binary format regression-test contract is incomplete"
            )
        input_format = format_match.group(1).strip("'").casefold()
        allowed_errors = {
            "native": {"TOO_LARGE_STRING_SIZE", "TOO_LARGE_ARRAY_SIZE"},
            "bsoneachrow": {
                "INCORRECT_DATA",
                "UNKNOWN_TYPE",
                "CANNOT_READ_ALL_DATA",
            },
        }
        observed_errors = [
            value.strip() for value in error_match.group("errors").split(",")
        ]
        valid_native_errors = (
            input_format == "native"
            and len(observed_errors) == 1
            and observed_errors[0] in allowed_errors[input_format]
        )
        valid_bson_errors = (
            input_format == "bsoneachrow"
            and observed_errors[0] == "INCORRECT_DATA"
            and len(observed_errors) == len(set(observed_errors))
            and set(observed_errors) <= allowed_errors[input_format]
        )
        if not (valid_native_errors or valid_bson_errors):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared clickhouse_dollar_quoted_binary_sql "
                "but its format and expected server error disagree"
            )
        return

    if entry.detected_format == "utf16le_generated_c_array":
        payload = path.read_bytes()
        generated: str | None = None
        if payload.startswith(b"\xff\xfe"):
            try:
                generated = payload[2:].decode("utf-16le")
            except UnicodeDecodeError as exc:
                raise SourceQuarantineError(
                    f"{entry.relative_path}: declared utf16le_generated_c_array "
                    f"but the encoded source is invalid: {exc}"
                ) from exc
        else:
            # ThreadX module_code.c shape: ASCII C-comment header, then a
            # UTF-16LE generated array body without a leading BOM.
            utf16_anchor = payload.find(b"I\x00n\x00p\x00u\x00t\x00 \x00E\x00L\x00F")
            if utf16_anchor < 0:
                utf16_anchor = payload.find(b"/\x00*\x00 \x00\n\x00\n\x00")
            if utf16_anchor < 0:
                raise SourceQuarantineError(
                    f"{entry.relative_path}: declared utf16le_generated_c_array "
                    "but neither a UTF-16LE BOM nor a UTF-16LE generated body "
                    "anchor is present"
                )
            # Prefer an even-length trailing slice so utf-16le decoding is exact.
            start = utf16_anchor
            if (len(payload) - start) % 2:
                if start > 0 and (len(payload) - (start - 1)) % 2 == 0:
                    start = start - 1
                else:
                    raise SourceQuarantineError(
                        f"{entry.relative_path}: declared utf16le_generated_c_array "
                        "but the UTF-16LE body is not even-length aligned"
                    )
            ascii_prefix = payload[:start]
            if not ascii_prefix.lstrip().startswith((b"/*", b"/****")):
                raise SourceQuarantineError(
                    f"{entry.relative_path}: declared utf16le_generated_c_array "
                    "but the ASCII prefix is not a C comment header"
                )
            try:
                generated = payload[start:].decode("utf-16le")
            except UnicodeDecodeError as exc:
                raise SourceQuarantineError(
                    f"{entry.relative_path}: declared utf16le_generated_c_array "
                    f"but the UTF-16LE body is invalid: {exc}"
                ) from exc
        assert generated is not None
        required_generated = (
            "Input ELF file:",
            "Output C Array file:",
            "__align(4096) unsigned char  module_code[] = {",
            "/* Address",
        )
        byte_literals = re.findall(
            r"(?<![0-9A-F])0x[0-9A-F]{2}(?![0-9A-F])",
            generated,
        )
        if (
            not all(marker in generated for marker in required_generated)
            or len(byte_literals) < 1024
            or "\x00" in generated
            or not generated.rstrip().endswith("};")
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared utf16le_generated_c_array "
                "but the generated binary-array contract is incomplete"
            )
        return

    if entry.detected_format == "posix_shell_appended_zip":
        payload = path.read_bytes()
        archive_start = payload.find(b"PK\x03\x04")
        if archive_start <= 0:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared posix_shell_appended_zip "
                "but an appended ZIP local header is absent"
            )
        try:
            shell_prefix = payload[:archive_start].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared posix_shell_appended_zip "
                f"but the shell prefix is not UTF-8: {exc}"
            ) from exc
        first_line = shell_prefix.splitlines()[0] if shell_prefix else ""
        if (
            first_line not in {"#!/bin/sh", "#!/bin/bash"}
            or '"$0"' not in shell_prefix
            or "\x00" in shell_prefix
            or not shell_prefix.endswith("\n")
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared posix_shell_appended_zip "
                "but the self-executing shell contract is incomplete"
            )
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                first_bad_member = archive.testzip()
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared posix_shell_appended_zip "
                f"but the appended ZIP is invalid: {exc}"
            ) from exc
        if not members or not any(not member.is_dir() for member in members):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared posix_shell_appended_zip "
                "but the appended ZIP has no file members"
            )
        if first_bad_member is not None:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared posix_shell_appended_zip "
                f"but ZIP CRC validation failed for {first_bad_member!r}"
            )
        return

    if entry.detected_format == "big5_shell_heredoc":
        payload = path.read_bytes()
        if b"\x00" in payload:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared big5_shell_heredoc but the "
                "shell fixture contains a NUL byte"
            )
        legacy_prefix = (
            b"#! /bin/sh\n\n"
            b"# Test conversion from BIG5 to UTF-8.\n\n"
            b'tmpfiles=""\n'
            b"trap 'rm -fr $tmpfiles' 1 2 3 15\n\n"
            b'tmpfiles="$tmpfiles mco-test1.po"\n'
        )
        modern_prefix = (
            b"#! /bin/sh\n"
            b'. "${srcdir=.}/init.sh"; path_prepend_ . ../src\n\n'
            b"# Test conversion from BIG5 to UTF-8.\n\n"
        )
        if payload.startswith(legacy_prefix):
            wrapper_variant = "legacy"
        elif payload.startswith(modern_prefix):
            wrapper_variant = "modern"
        else:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared big5_shell_heredoc but the "
                "canonical shell preamble is absent"
            )
        po_marker = b"cat <<\\EOF > mco-test1.po\n"
        ok_marker = b"cat <<\\EOF > mco-test1.ok\n"
        if payload.count(po_marker) != 1 or payload.count(ok_marker) != 1:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared big5_shell_heredoc but the "
                "two expected heredoc declarations are not unique"
            )
        po_start = payload.index(po_marker) + len(po_marker)
        po_end = payload.find(b"\nEOF\n", po_start)
        ok_decl = payload.index(ok_marker)
        ok_start = ok_decl + len(ok_marker)
        ok_end = payload.find(b"\nEOF\n", ok_start)
        if (
            po_end < 0
            or ok_end < 0
            or po_end >= ok_start
            or payload.count(b"\nEOF\n") != 2
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared big5_shell_heredoc but heredoc "
                "boundaries are invalid"
            )
        try:
            po_text = payload[po_start:po_end].decode("big5")
            ok_text = payload[ok_start:ok_end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared big5_shell_heredoc but an "
                f"heredoc payload has invalid encoding: {exc}"
            ) from exc
        common_prefix = (
            '# Chinese translation for GNU gettext messages.\n'
            '#\n'
            'msgid ""\n'
            'msgstr ""\n'
            '"MIME-Version: 1.0\\n"\n'
        )
        common_suffix = (
            '"Content-Transfer-Encoding: 8bit\\n"\n'
            '\n'
            '#: src/msgcmp.c:155 src/msgmerge.c:273\n'
            'msgid "exactly 2 input files required"\n'
        )
        po_contract = (
            common_prefix
            + '"Content-Type: text/plain; charset=big5\\n"\n'
            + common_suffix
        )
        ok_contract = (
            common_prefix
            + '"Content-Type: text/plain; charset=UTF-8\\n"\n'
            + common_suffix
        )
        if not po_text.startswith(po_contract) or not ok_text.startswith(
            ok_contract
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared big5_shell_heredoc but the "
                "gettext message contract is incomplete"
            )
        expected_translation = (
            'msgstr "\u6b64\u529f\u80fd\u9700\u8981\u6070\u597d'
            '\u6307\u5b9a\u5169\u500b\u8f38\u5165\u6a94"'
        )
        if (
            not po_text.endswith(expected_translation)
            or not ok_text.endswith(expected_translation)
            or not any(byte >= 0x80 for byte in payload[po_start:po_end])
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared big5_shell_heredoc but the "
                "BIG5/UTF-8 message bodies do not match"
            )
        legacy_middle = (
            b"\ntmpfiles=\"$tmpfiles mco-test1.out\"\n"
            b": ${MSGCONV=msgconv}\n"
            b"${MSGCONV} --to-code=UTF-8 -o mco-test1.out mco-test1.po\n"
            b"test $? = 0 || { rm -fr $tmpfiles; exit 1; }\n\n"
            b'tmpfiles="$tmpfiles mco-test1.ok"\n'
        )
        modern_middle = (
            b"\n: ${MSGCONV=msgconv}\n"
            b"${MSGCONV} --to-code=UTF-8 -o mco-test1.out mco-test1.po "
            b"|| Exit 1\n\n"
        )
        legacy_suffix = (
            b"\n: ${DIFF=diff}\n"
            b"# Redirect stdout, so as not to fill the user's screen with "
            b"non-ASCII bytes.\n"
            b"${DIFF} mco-test1.ok mco-test1.out >/dev/null\n"
            b"result=$?\n\n"
            b"rm -fr $tmpfiles\n\n"
            b"exit $result\n"
        )
        modern_suffix = (
            b"\n: ${DIFF=diff}\n"
            b"# Redirect stdout, so as not to fill the user's screen with "
            b"non-ASCII bytes.\n"
            b"${DIFF} mco-test1.ok mco-test1.out >/dev/null\n"
            b"result=$?\n\n"
            b"exit $result\n"
        )
        expected_middle = (
            legacy_middle if wrapper_variant == "legacy" else modern_middle
        )
        expected_suffix = (
            legacy_suffix if wrapper_variant == "legacy" else modern_suffix
        )
        if (
            payload[po_end + len(b"\nEOF\n") : ok_decl] != expected_middle
            or not payload.endswith(expected_suffix)
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared big5_shell_heredoc but the "
                "conversion and cleanup shell contract is incomplete"
            )
        return

    if entry.detected_format == "git_shortlog_invalid_utf8_shell":
        payload = path.read_bytes()
        malformed_treble_clef = b"\xf8\x9d\x84\x9e"
        valid_treble_clef = b"\xf0\x9d\x84\x9e"
        required_fragments = (
            b"#!/bin/sh\n",
            b"test_description='git shortlog\n'",
            b"# when replacing all is by treble clefs.",
            b'tr 1234 "\\360\\235\\204\\236"',
            b"# now fsck up the utf8",
            b"git config i18n.commitencoding non-utf-8",
            b'tr 1234 "\\370\\235\\204\\236"',
            b"# NOTE: do not quote this heredoc, Dash 0.5.13 has a bug with heredocs",
        )
        if (
            b"\x00" in payload
            or any(fragment not in payload for fragment in required_fragments)
            or payload.count(malformed_treble_clef) != 8
            or payload.count(valid_treble_clef) != 8
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared git_shortlog_invalid_utf8_shell "
                "but the deliberate malformed-UTF8 shortlog contract is incomplete"
            )
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.reason != "invalid start byte" or payload[exc.start] != 0xF8:
                raise SourceQuarantineError(
                    f"{entry.relative_path}: declared "
                    "git_shortlog_invalid_utf8_shell but its first UTF-8 failure "
                    "is not the deliberate 0xf8 leading byte"
                ) from exc
        else:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared git_shortlog_invalid_utf8_shell "
                "but the shell fixture is valid UTF-8"
            )
        return

    if entry.detected_format == "invalid_utf8_and_windows1252_domain_blob":
        # Erlang re_SUITE_data/testoutput* (and similar): domain text fixtures
        # that are neither valid UTF-8 nor valid Windows-1252, so the indexer
        # fail-closes before parse.  No NULs; mostly ASCII with sparse high bytes.
        payload = path.read_bytes()
        if not payload or b"\x00" in payload:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "invalid_utf8_and_windows1252_domain_blob but the payload is "
                "empty or contains NULs"
            )
        high = sum(1 for byte in payload if byte >= 0x80)
        if high < 1:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "invalid_utf8_and_windows1252_domain_blob but the payload has "
                "no high bytes"
            )
        if high / len(payload) > 0.05:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "invalid_utf8_and_windows1252_domain_blob but the payload is "
                "not mostly ASCII"
            )
        utf8_failed = False
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            utf8_failed = True
        cp1252_failed = False
        try:
            payload.decode("cp1252")
        except UnicodeDecodeError:
            cp1252_failed = True
        if not utf8_failed or not cp1252_failed:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared "
                "invalid_utf8_and_windows1252_domain_blob but the payload is "
                "valid as UTF-8 or Windows-1252"
            )
        return

    if entry.detected_format == "asn1_der_x509_certificate_pair":
        payload = path.read_bytes()
        try:
            outer_tag, outer_start, outer_end = _der_tlv_bounds(payload, 0)
            if outer_tag != 0x30 or outer_end != len(payload):
                raise ValueError("outer value is not one exact DER SEQUENCE")
            wrapper_tags: set[int] = set()
            offset = outer_start
            while offset < outer_end:
                wrapper_tag, wrapper_start, wrapper_end = _der_tlv_bounds(
                    payload,
                    offset,
                )
                if wrapper_tag not in {0xA0, 0xA1} or wrapper_tag in wrapper_tags:
                    raise ValueError("invalid or duplicate CertificatePair wrapper")
                wrapper_tags.add(wrapper_tag)
                cert_tag, cert_start, cert_end = _der_tlv_bounds(
                    payload,
                    wrapper_start,
                )
                if cert_tag != 0x30 or cert_end != wrapper_end:
                    raise ValueError(
                        "wrapper does not contain one certificate SEQUENCE"
                    )
                child_offset = cert_start
                for expected_tag in (0x30, 0x30, 0x03):
                    child_tag, _, child_offset = _der_tlv_bounds(
                        payload,
                        child_offset,
                    )
                    if child_tag != expected_tag:
                        raise ValueError("invalid X.509 certificate field layout")
                if child_offset != cert_end:
                    raise ValueError("certificate SEQUENCE has trailing fields")
                offset = wrapper_end
            if not wrapper_tags:
                raise ValueError("CertificatePair is empty")
        except ValueError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared asn1_der_x509_certificate_pair "
                f"but the DER structure is invalid: {exc}"
            ) from exc
        return


    if entry.detected_format == "plumhall_c_date_time_libclang_hang":
        payload = path.read_bytes()
        try:
            decoded = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared plumhall_c_date_time_libclang_hang "
                f"but the fixture is not ASCII: {exc}"
            ) from exc
        # Exact Plum Hall 4.12 date/time conformance fixture: pinned libclang
        # enters a non-terminating parse spin in Parser::isTypeSpecifierQualifier
        # (observed multi-hour hang on a 4KB file under corpus.local/xbox_leak_may_2020).
        required_substrings = (
            "Plum Hall Validation Suite for C",
            '#include <time.h>',
            'Filename = "d412.c"',
            "void d4_12()",
            "#define SKIP412",
            "4.12 - Date and time",
        )
        if (
            path.suffix.casefold() != ".c"
            or any(s not in decoded for s in required_substrings)
            or decoded.count('#include <time.h>') < 1
            or "struct tm" not in decoded
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared plumhall_c_date_time_libclang_hang "
                "but the Plum Hall 4.12 date/time fixture contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "glibc_stdio_bug28_libclang_timeout":
        payload = path.read_bytes()
        try:
            decoded = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared glibc_stdio_bug28_libclang_timeout "
                f"but the fixture is not ASCII: {exc}"
            ) from exc
        required_substrings = (
            "do_test (void)",
            'size_t instances = 16384;',
            '#define X0 "\\n%1$s\\n"',
            "#define X14 X12 X12 X12 X12",
            '#define TRAILER "%%%%%%%%%%%%%%%%%%%%%%%%%%"',
            '#include "../test-skeleton.c"',
        )
        if (
            path.name != "bug28.c"
            or Path(entry.relative_path).as_posix() != "stdio-common/bug28.c"
            or any(s not in decoded for s in required_substrings)
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared glibc_stdio_bug28_libclang_timeout "
                "but the glibc stdio-common/bug28.c contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "antlr_inputbuffer_hpp_libclang_timeout":
        payload = path.read_bytes()
        relative = Path(entry.relative_path).as_posix()
        if (
            path.name != "InputBuffer.hpp"
            or path.suffix.casefold() != ".hpp"
            or relative != "OSX/libsecurity_codesigning/antlr2/antlr/InputBuffer.hpp"
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "path is not OSX/libsecurity_codesigning/antlr2/antlr/InputBuffer.hpp"
            )
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                f"header is not UTF-8: {exc}"
            ) from exc
        required_substrings = (
            "#ifndef INC_InputBuffer_hpp__",
            "#define INC_InputBuffer_hpp__",
            "ANTLR Translator Generator",
            "Terence Parr",
            "#include <antlr/config.hpp>",
            "class ANTLR_API InputBuffer",
            "virtual int getChar()=0",
            "#endif //INC_InputBuffer_hpp__",
        )
        if any(s not in decoded for s in required_substrings):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "ANTLR InputBuffer.hpp contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "antlr_baseast_hpp_libclang_timeout":
        payload = path.read_bytes()
        relative = Path(entry.relative_path).as_posix()
        if (
            path.name != "BaseAST.hpp"
            or path.suffix.casefold() != ".hpp"
            or relative != "OSX/libsecurity_codesigning/antlr2/antlr/BaseAST.hpp"
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "path is not OSX/libsecurity_codesigning/antlr2/antlr/BaseAST.hpp"
            )
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                f"header is not UTF-8: {exc}"
            ) from exc
        required_substrings = (
            "#ifndef INC_BaseAST_hpp__",
            "#define INC_BaseAST_hpp__",
            "ANTLR Translator Generator",
            "Terence Parr",
            "#include <antlr/AST.hpp>",
            "class ANTLR_API BaseAST : public AST",
            "virtual RefAST clone( void ) const = 0",
            "#endif //INC_BaseAST_hpp__",
        )
        if any(s not in decoded for s in required_substrings):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "ANTLR BaseAST.hpp contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "antlr_charscanner_hpp_libclang_timeout":
        payload = path.read_bytes()
        relative = Path(entry.relative_path).as_posix()
        if (
            path.name != "CharScanner.hpp"
            or path.suffix.casefold() != ".hpp"
            or relative != "OSX/libsecurity_codesigning/antlr2/antlr/CharScanner.hpp"
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "path is not OSX/libsecurity_codesigning/antlr2/antlr/CharScanner.hpp"
            )
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                f"header is not UTF-8: {exc}"
            ) from exc
        required_substrings = (
            "#ifndef INC_CharScanner_hpp__",
            "#define INC_CharScanner_hpp__",
            "ANTLR Translator Generator",
            "Terence Parr",
            "#include <antlr/InputBuffer.hpp>",
            "class ANTLR_API CharScanner : public TokenStream",
            "virtual int LA(unsigned int i);",
            "#endif //INC_CharScanner_hpp__",
        )
        if any(s not in decoded for s in required_substrings):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "ANTLR CharScanner.hpp contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "antlr_tokenbuffer_hpp_libclang_timeout":
        payload = path.read_bytes()
        relative = Path(entry.relative_path).as_posix()
        if (
            path.name != "TokenBuffer.hpp"
            or path.suffix.casefold() != ".hpp"
            or relative != "OSX/libsecurity_codesigning/antlr2/antlr/TokenBuffer.hpp"
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "path is not OSX/libsecurity_codesigning/antlr2/antlr/TokenBuffer.hpp"
            )
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                f"header is not UTF-8: {exc}"
            ) from exc
        required_substrings = (
            "#ifndef INC_TokenBuffer_hpp__",
            "#define INC_TokenBuffer_hpp__",
            "ANTLR Translator Generator",
            "Terence Parr",
            "#include <antlr/TokenStream.hpp>",
            "class ANTLR_API TokenBuffer {",
            "virtual unsigned int entries() const;",
            "#endif //INC_TokenBuffer_hpp__",
        )
        if any(s not in decoded for s in required_substrings):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "ANTLR TokenBuffer.hpp contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "apple_security_blockcryptor_macroman_nbsp_libclang_timeout":
        payload = path.read_bytes()
        relative = Path(entry.relative_path).as_posix()
        if (
            path.name != "BlockCryptor.h"
            or relative != "OSX/libsecurity_apple_csp/lib/BlockCryptor.h"
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "path is not OSX/libsecurity_apple_csp/lib/BlockCryptor.h"
            )
        high_offsets = [i for i, byte in enumerate(payload) if byte >= 0x80]
        if high_offsets != [4318] or payload[4318] != 0xCA:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "MacRoman NBSP identity is missing (need exactly one 0xCA at offset 4318)"
            )
        decoded = payload.decode("latin-1")
        required_substrings = (
            "#ifndef\t_BLOCK_CRYPTOR_H_",
            "#define _BLOCK_CRYPTOR_H_",
            "BlockCryptor.h - common context for block-oriented encryption algorithms",
            '#include "AppleCSPContext.h"',
            "class BlockCryptor : public AppleCSPContext",
            "BCM_CBC",
            "Apple Inc.",
        )
        if any(s not in decoded for s in required_substrings):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "BlockCryptor.h contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "apple_security_dh_keys_macroman_nbsp_libclang_timeout":
        payload = path.read_bytes()
        relative = Path(entry.relative_path).as_posix()
        if (
            path.name != "DH_keys.h"
            or relative != "OSX/libsecurity_apple_csp/lib/DH_keys.h"
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "path is not OSX/libsecurity_apple_csp/lib/DH_keys.h"
            )
        high_offsets = [i for i, byte in enumerate(payload) if byte >= 0x80]
        if high_offsets != [2825] or payload[2825] != 0xCA:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "MacRoman NBSP identity is missing (need exactly one 0xCA at offset 2825)"
            )
        decoded = payload.decode("latin-1")
        required_substrings = (
            "#ifndef\t_DH_KEYS_H_",
            "#define _DH_KEYS_H_",
            "DH_keys.h - Diffie-Hellman key pair support",
            "#include <AppleCSPContext.h>",
            "class DHBinaryKey : public BinaryKey",
            "class DHKeyPairGenContext :",
            "class DHKeyInfoProvider",
            "Apple Inc.",
        )
        if any(s not in decoded for s in required_substrings):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "DH_keys.h contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format == "apple_security_rsa_dsa_keys_macroman_nbsp_libclang_timeout":
        payload = path.read_bytes()
        relative = Path(entry.relative_path).as_posix()
        if (
            path.name != "RSA_DSA_keys.h"
            or relative != "OSX/libsecurity_apple_csp/lib/RSA_DSA_keys.h"
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "path is not OSX/libsecurity_apple_csp/lib/RSA_DSA_keys.h"
            )
        high_offsets = [i for i, byte in enumerate(payload) if byte >= 0x80]
        if high_offsets != [5356] or payload[5356] != 0xCA:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "MacRoman NBSP identity is missing (need exactly one 0xCA at offset 5356)"
            )
        decoded = payload.decode("latin-1")
        required_substrings = (
            "#ifndef\t_RSA_DSA_KEYS_H_",
            "#define _RSA_DSA_KEYS_H_",
            "RSA_DSA_keys.h - key pair support for RSA/DSA",
            "#include <AppleCSPContext.h>",
            "class RSABinaryKey : public BinaryKey {",
            "class RSAKeyPairGenContext :",
            "class DSAKeyInfoProvider : public CSPKeyInfoProvider",
            "Apple Inc.",
        )
        if any(s not in decoded for s in required_substrings):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "RSA_DSA_keys.h contract is incomplete or ambiguous"
            )
        return

    if entry.detected_format in {
        "apple_security_ssdl_session_libclang_hang",
        "apple_security_libclang_timeout_header",
    }:
        # Apple Security headers that deterministically hit FAIL_CLOSED libclang
        # parse timeout (300s) then BrokenProcessPool on GCP even with MacOSX.sdk.
        # Per-basename content contracts keep the quarantine identity-checked.
        payload = path.read_bytes()
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                f"header is not UTF-8: {exc}"
            ) from exc
        expected_name = Path(entry.relative_path).name
        header_contracts: dict[str, tuple[str, ...]] = {
            "SSDLSession.h": (
                "#ifndef _H_SSDLSESSION",
                "#define _H_SSDLSESSION",
                "class SSDLSession : public DLPluginSession",
                "#include <security_cdsa_plugin/DLsession.h>",
                "SSCSPDLSession &mSSCSPDLSession",
                "SecurityServer::ClientSession &clientSession()",
            ),
            "cssmcontext.h": (
                "#ifndef _H_CSSMCONTEXT",
                "#define _H_CSSMCONTEXT",
                '#include "cssmint.h"',
                '#include "cspattachment.h"',
                "#include <security_cdsa_utilities/context.h>",
                "context - manage CSSM",
            ),
            "clNssUtils.h": (
                "#define _CL_NSS_UTILS_H_",
                '#include <security_asn1/SecNssCoder.h>',
                '#include <Security/cssm.h>',
                '#include "DecodedCert.h"',
                "class ArenaAllocator : public Security::Allocator",
                "clNssUtils.h - support for libnssasn1-based ASN1 encode/decode",
            ),
            "SDContext.h": (
                "#ifndef _H_SD_CONTEXT",
                "#define _H_SD_CONTEXT",
                "SDContext.h - Security Server contexts",
                "#include <security_cdsa_plugin/CSPsession.h>",
                "#include <securityd_client/ssclient.h>",
                "class SDCSPSession",
            ),
            "kcdatabase.h": (
                "#ifndef _H_KCDATABASE",
                "#define _H_KCDATABASE",
                "kcdatabase - software database container implementation",
                '#include "localdatabase.h"',
                "#include <securityd_client/ss_types.h>",
                "class KeychainDatabase",
            ),
            "child.h": (
                "#ifndef _CHILD_H_",
                "#define _CHILD_H_",
                "child - track a single child process and its belongings",
                "#include <security_utilities/mach++.h>",
                "#include <security_utilities/unixchild.h>",
                "class ServerChild : public UnixPlusPlus::Child",
                "Apple Computer, Inc.",
            ),
            "SSKey.h": (
                "#define _H_SSKEY_",
                "SSKey.h - CSP-wide SSKey base class",
                "#include <security_cdsa_plugin/CSPsession.h>",
                '#include "SSDatabase.h"',
                "#include <securityd_client/ssclient.h>",
                "class SSKey : public ReferencedKey",
            ),
            "kckey.h": (
                "#ifndef _H_KCKEY",
                "#define _H_KCKEY",
                "key - representation of SecurityServer key objects",
                '#include "localkey.h"',
                "#include <security_cdsa_client/keyclient.h>",
                "class KeychainKey : public LocalKey, public SecurityServerAcl",
                "Apple Computer, Inc.",
            ),
            "SSDatabase.h": (
                "#define _H_SSDATABASE_",
                "SSDatabase.h - Security Server database object",
                "#include <security_cdsa_client/dlclient.h>",
                "#include <securityd_client/ssclient.h>",
                "class SSDatabaseImpl : public CssmClient::DbImpl",
                "class SSDatabase : public CssmClient::Db",
            ),
            "SSContext.h": (
                "#ifndef _H_SS_CONTEXT",
                "#define _H_SS_CONTEXT",
                "SSContext.h - Security Server contexts",
                "#include <security_cdsa_plugin/CSPsession.h>",
                "#include <securityd_client/ssclient.h>",
                "class SSContext : public CSPFullPluginSession::CSPContext",
            ),
            "dyldcache.h": (
                "#ifndef _H_DYLDCACHE",
                "#define _H_DYLDCACHE",
                "dyldcache - access layer to the DYLD Shared Library Cache file",
                "#include <security_utilities/unix++.h>",
                '#include "dyld_cache_format.h"',
                "class DYLDCache : public UnixPlusPlus::AutoFileDesc",
            ),
            "authhost.h": (
                "#ifndef _H_AUTHHOST",
                "#define _H_AUTHHOST",
                '#include "structure.h"',
                '#include "child.h"',
                "class AuthHostInstance : public PerSession, public ServerChild",
                "bool inDarkWake()",
            ),
            "connection.h": (
                "#ifndef _H_CONNECTION",
                "#define _H_CONNECTION",
                "connection - manage connections to clients",
                '#include "process.h"',
                "class Connection : public PerConnection, public Listener::JitterBuffer",
                "void beginWork(audit_token_t &auditToken)",
            ),
            "key.h": (
                "#ifndef _H_KEY",
                "#define _H_KEY",
                "key - representation of securityd key objects",
                '#include "structure.h"',
                '#include "database.h"',
                "class Key : public Database::Subsidiary, public AclSource",
                "virtual const CssmData &canonicalDigest() = 0",
            ),
            "structure.h": (
                "#ifndef _H_STRUCTURE",
                "#define _H_STRUCTURE",
                "structure - structural framework for securityd objects",
                "#include <security_utilities/refcount.h>",
                "#include <security_utilities/mach++.h>",
                "class PerConnection;",
                'Repeat after me: "Everything that matters is a Node."',
            ),
            "slcrep.h": (
                "#ifndef _H_SLCREP",
                "#define _H_SLCREP",
                "slcrep - DiskRep representing the Mac OS Shared Library Cache",
                '#include "singlediskrep.h"',
                '#include "sigblob.h"',
                "#include <security_utilities/dyldcache.h>",
                "class DYLDCacheRep : public SingleDiskRep",
            ),
            "tokend.h": (
                "#ifndef _H_TOKEND",
                "#define _H_TOKEND",
                "tokend - internal tracker for a tokend smartcard driver process",
                '#include "structure.h"',
                '#include "child.h"',
                '#include "tokencache.h"',
                "class TokenDaemon",
                "virtual void relayFault(bool async) = 0",
            ),
            "process.h": (
                "#ifndef _H_PROCESS",
                "#define _H_PROCESS",
                "process - track a single client process and its belongings",
                '#include "structure.h"',
                '#include "session.h"',
                "class Process : public PerProcess",
                "void changeSession(Session::SessionId sessionId)",
            ),
            "localdatabase.h": (
                "#ifndef _H_LOCALDATABASE",
                "#define _H_LOCALDATABASE",
                "localdatabase - locally implemented database using internal CSP cryptography",
                '#include "database.h"',
                "class LocalDatabase : public Database",
                "virtual RefPointer<Key> makeKey(const CssmKey &newKey, uint32 moreAttributes,",
            ),
            "tokendatabase.h": (
                "#ifndef _H_TOKENDATABASE",
                "#define _H_TOKENDATABASE",
                "tokendatabase - software database container implementation.",
                '#include "database.h"',
                '#include "tokenacl.h"',
                "class TokenDatabase : public Database",
                "TokenDaemon &tokend();",
            ),
            "tokenkey.h": (
                "#ifndef _H_TOKENKEY",
                "#define _H_TOKENKEY",
                "tokenkey - remote reference key on an attached hardware token",
                '#include "key.h"',
                '#include "tokenacl.h"',
                "class TokenKey : public Key, public TokenAcl",
                "KeyHandle tokenHandle() const",
            ),
            "localkey.h": (
                "#ifndef _H_LOCALKEY",
                "#define _H_LOCALKEY",
                "localkey - Key objects that store a local CSSM key object",
                '#include "key.h"',
                "#include <security_cdsa_client/keyclient.h>",
                "class LocalKey : public Key",
                "virtual void getKey();",
            ),
            "desContext.h": (
                "#ifndef _DES_CONTEXT_H_",
                "#define _DES_CONTEXT_H_",
                "desContext.h - glue between BlockCrytpor and DES/3DES implementations",
                '#include "BlockCryptor.h"',
                "class DESContext : public BlockCryptor",
                "class DES3Context : public BlockCryptor",
                "virtual ~DESContext();",
            ),
            "tempdatabase.h": (
                "#ifndef _H_TEMPDATABASE",
                "#define _H_TEMPDATABASE",
                "tempdatabase - temporary (scratch) storage for keys",
                '#include "localdatabase.h"',
                "class TempDatabase : public LocalDatabase",
                "bool transient() const",
                "void getSecurePassphrase(const Context &context, string &passphrase)",
            ),
            "reader.h": (
                "#ifndef _H_READER",
                "#define _H_READER",
                "reader - token reader objects",
                '#include "structure.h"',
                '#include "token.h"',
                '#include "tokencache.h"',
                "class Reader : public PerGlobal",
                "void insertToken(TokenDaemon *tokend)",
            ),
            "AppleCSPContext.h": (
                "#ifndef _H_APPLE_CSP_CONTEXT",
                "#define _H_APPLE_CSP_CONTEXT",
                "AppleCSPContext.h - CSP-wide contexts",
                "#include <security_cdsa_plugin/CSPsession.h>",
                '#include "BinaryKey.h"',
                "class AppleCSPContext : public CSPFullPluginSession::CSPContext",
                "class YarrowContext : public AppleCSPContext",
                "static void symmetricKeyBits(",
            ),
            "pcscmonitor.h": (
                "#ifndef _H_PCSCMONITOR",
                "#define _H_PCSCMONITOR",
                "pcscmonitor - use PCSC to monitor smartcard reader/card state for securityd",
                '#include "server.h"',
                '#include "tokencache.h"',
                '#include "reader.h"',
                "class PCSCMonitor : private Listener, private MachServer::Timer",
                "void startSoftTokens();",
            ),
            "server.h": (
                "#ifndef _H_SERVER",
                "#define _H_SERVER",
                "server - securityd main server object",
                '#include "structure.h"',
                '#include "connection.h"',
                "class Server : public PerGlobal,",
                "static Server &active()",
                "void beginShutdown();",
            ),
            "token.h": (
                "#ifndef _H_TOKEN",
                "#define _H_TOKEN",
                "token - internal representation of a (single distinct) hardware token",
                '#include "structure.h"',
                '#include "tokend.h"',
                "class Token : public PerGlobal, public virtual TokenAcl, public FaultRelay",
                "TokenDaemon &tokend();",
                "void insert(::Reader &slot, RefPointer<TokenDaemon> tokend);",
            ),
            "gladmanContext.h": (
                "#ifndef _H_GLADMAN_CONTEXT",
                "#define _H_GLADMAN_CONTEXT",
                "gladmanContext.h - Gladman AES context class",
                '#include "AppleCSPContext.h"',
                '#include "BlockCryptor.h"',
                "class GAESContext : public BlockCryptor {",
                "void encryptBlock(",
                "CCCryptorRef",
            ),
            "rc5Context.h": (
                "#ifndef _RC5_CONTEXT_H_",
                "#define _RC5_CONTEXT_H_",
                "rc5Context.h - glue between BlockCrytpor and ssleay RC5 implementation",
                "#include <BlockCryptor.h>",
                "#include <openssl/rc5_legacy.h>",
                "class RC5Context : public BlockCryptor {",
                "RC5_32_KEY",
                "#endif //_RC2_CONTEXT_H_",
            ),
            "tokenaccess.h": (
                "#ifndef _H_TOKENACCESS",
                "#define _H_TOKENACCESS",
                "tokenaccess - access management to a TokenDatabase's Token's TokenDaemon's tokend",
                '#include "tokendatabase.h"',
                '#include "tokenkey.h"',
                '#include "server.h"',
                "class Access : public Token::Access {",
                "#endif //_H_TOKENACCESS",
            ),
            "rc4Context.h": (
                "#ifndef _RC4_CONTEXT_H_",
                "#define _RC4_CONTEXT_H_",
                "rc4Context.h - glue between BlockCrytpor and ssleay RC4 implementation",
                '#include "AppleCSPContext.h"',
                "#include <CommonCrypto/CommonCryptor.h>",
                "class RC4Context : public AppleCSPContext {",
                "CCCryptorRef    rc4Key;",
                "#endif //_RC4_CONTEXT_H_",
            ),
            "rc2Context.h": (
                "#ifndef _RC2_CONTEXT_H_",
                "#define _RC2_CONTEXT_H_",
                "rc2Context.h - glue between BlockCrytpor and ssleay RC2 implementation",
                "#include <BlockCryptor.h>",
                "#include <openssl/rc2_legacy.h>",
                "class RC2Context : public BlockCryptor {",
                "RC2_KEY",
                "#endif //_RC2_CONTEXT_H_",
            ),
            "notifications.h": (
                "#ifndef _H_NOTIFICATIONS",
                "#define _H_NOTIFICATIONS",
                "notifications - handling of securityd-gated notification messages",
                "#include <securityd_client/ssclient.h>",
                '#include "SharedMemoryCommon.h"',
                "class Listener: public RefCount {",
                "class SharedMemoryListener",
            ),
            "agentquery.h": (
                "#ifndef _H_AGENTQUERY",
                "#define _H_AGENTQUERY",
                "passphrases - canonical code to obtain passphrases",
                '#include "kcdatabase.h"',
                '#include "authhost.h"',
                '#include "server.h"',
                "class SecurityAgentXPCConnection",
                "class SecurityAgentXPCQuery : public SecurityAgentXPCConnection",
                "#endif //_H_AGENTQUERY",
            ),
            "MacContext.h": (
                "#ifndef	_MAC_CONTEXT_H_",
                "#define _MAC_CONTEXT_H_",
                "MacContext.h - AppleCSPContext for HMAC{SHA1,MD5}",
                "#include <AppleCSPContext.h>",
                "#include <CommonCrypto/CommonHMAC.h>",
                "class MacContext : public AppleCSPContext  {",
                "class MacLegacyContext : public AppleCSPContext  {",
                "#endif	/* _MAC_CONTEXT_H_ */",
            ),
            "session.h": (
                "#ifndef _H_SESSION",
                "#define _H_SESSION",
                "session - authentication session domains",
                '#include "structure.h"',
                '#include "acls.h"',
                '#include "authhost.h"',
                "class Session : public PerSession {",
                "class RootSession : public Session {",
                "#endif //_H_SESSION",
            ),
            "SignatureContext.h": (
                "#ifndef	_SIGNATURE_CONTEXT_H_",
                "#define _SIGNATURE_CONTEXT_H_",
                "SignatureContext.h - AppleCSPContext subclass for generic sign/verify",
                "#include <RawSigner.h>",
                "#include <security_cdsa_utilities/digestobject.h>",
                "#include <AppleCSPContext.h>",
                "class SignatureContext : public AppleCSPContext  {",
                "#endif	/* _SIGNATURE_CONTEXT_H_ */",
            ),
        }
        required_substrings = header_contracts.get(expected_name)
        apple_copyright = (
            "Apple Inc." in decoded or "Apple Computer, Inc." in decoded
        )
        if (
            path.suffix.casefold() != ".h"
            or required_substrings is None
            or any(s not in decoded for s in required_substrings)
            or not apple_copyright
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared {entry.detected_format} but the "
                "Apple Security libclang-timeout header contract is incomplete or "
                "ambiguous"
            )
        return

    raise SourceQuarantineError(
        f"unsupported detected format at runtime: {entry.detected_format}"
    )


@dataclass
class ProjectSourceQuarantine:
    manifest_path: Path
    manifest_sha256: str
    manifest_entry_count: int
    project_id: str
    entries_by_path: dict[str, SourceQuarantineEntry]
    collections: tuple[SourceQuarantineCollection, ...]

    @classmethod
    def load(
        cls,
        manifest_path: str | os.PathLike[str],
        *,
        project_id: str,
    ) -> ProjectSourceQuarantine:
        path = Path(manifest_path)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise SourceQuarantineError(
                f"cannot read source quarantine manifest {path}: {exc}"
            ) from exc
        try:
            raw = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceQuarantineError(
                f"invalid source quarantine manifest JSON {path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise SourceQuarantineError(f"{path}: manifest root must be an object")
        schema = raw.get("schema")
        expected_keys = (
            {"schema", "entries"}
            if schema == LEGACY_MANIFEST_SCHEMA
            else {"schema", "entries", "collections"}
        )
        if schema not in {LEGACY_MANIFEST_SCHEMA, MANIFEST_SCHEMA}:
            raise SourceQuarantineError(
                f"{path}: unsupported schema {schema!r}; expected one of "
                f"{[LEGACY_MANIFEST_SCHEMA, MANIFEST_SCHEMA]!r}"
            )
        if set(raw) != expected_keys:
            raise SourceQuarantineError(
                f"{path}: expected exactly {sorted(expected_keys)!r} fields"
            )
        raw_entries = raw["entries"]
        if not isinstance(raw_entries, list):
            raise SourceQuarantineError(f"{path}: entries must be a list")
        parsed = [
            _parse_entry(item, index=index) for index, item in enumerate(raw_entries)
        ]
        raw_collections = raw.get("collections", [])
        if not isinstance(raw_collections, list):
            raise SourceQuarantineError(f"{path}: collections must be a list")
        parsed_collections = [
            _parse_collection(item, index=index)
            for index, item in enumerate(raw_collections)
        ]
        identities = [(entry.project_id, entry.relative_path) for entry in parsed]
        if len(set(identities)) != len(identities):
            raise SourceQuarantineError(f"{path}: duplicate project/path entries")
        collection_identities = [
            (
                collection.project_id,
                collection.relative_path_prefix,
                collection.relative_path_suffix,
            )
            for collection in parsed_collections
        ]
        if len(set(collection_identities)) != len(collection_identities):
            raise SourceQuarantineError(f"{path}: duplicate collection entries")
        project_entries = {
            entry.relative_path: entry
            for entry in parsed
            if entry.project_id == project_id
        }
        project_collections = tuple(
            collection
            for collection in parsed_collections
            if collection.project_id == project_id
        )
        return cls(
            manifest_path=path,
            manifest_sha256=hashlib.sha256(payload).hexdigest(),
            manifest_entry_count=len(parsed) + len(parsed_collections),
            project_id=project_id,
            entries_by_path=project_entries,
            collections=project_collections,
        )

    def filter_candidates(
        self,
        project_root: str | os.PathLike[str],
        candidates: Iterable[str],
    ) -> tuple[list[str], dict[str, object]]:
        root = os.path.abspath(os.fspath(project_root))
        kept: list[str] = []
        consumed: dict[str, SourceQuarantineEntry] = {}
        collection_entries = {collection: [] for collection in self.collections}
        candidate_list = list(candidates)
        for candidate in candidate_list:
            absolute_candidate = os.path.abspath(candidate)
            relative = os.path.relpath(absolute_candidate, root)
            relative_posix = Path(relative).as_posix()
            if relative_posix == ".." or relative_posix.startswith("../"):
                raise SourceQuarantineError(
                    f"source candidate escapes project root: {candidate}"
                )
            entry = self.entries_by_path.get(relative_posix)
            matching_collections = [
                collection
                for collection in self.collections
                if relative_posix.startswith(collection.relative_path_prefix)
                and relative_posix.endswith(collection.relative_path_suffix)
            ]
            if len(matching_collections) > 1 or (
                entry is not None and matching_collections
            ):
                raise SourceQuarantineError(
                    f"{relative_posix}: candidate matches multiple quarantine rules"
                )
            collection = matching_collections[0] if matching_collections else None
            if entry is None and collection is None:
                kept.append(candidate)
                continue
            path = Path(absolute_candidate)
            try:
                observed_size = path.stat().st_size
            except OSError as exc:
                raise SourceQuarantineError(
                    f"cannot stat quarantined candidate {path}: {exc}"
                ) from exc
            if entry is not None and observed_size != entry.size_bytes:
                raise SourceQuarantineError(
                    f"{relative_posix}: quarantine size mismatch: "
                    f"observed={observed_size} expected={entry.size_bytes}"
                )
            observed_sha256 = _sha256_file(path)
            if entry is not None and observed_sha256 != entry.sha256:
                raise SourceQuarantineError(
                    f"{relative_posix}: quarantine SHA-256 mismatch: "
                    f"observed={observed_sha256} expected={entry.sha256}"
                )
            if entry is not None:
                observed_entry = entry
            else:
                assert collection is not None
                observed_entry = SourceQuarantineEntry(
                    project_id=self.project_id,
                    relative_path=relative_posix,
                    size_bytes=observed_size,
                    sha256=observed_sha256,
                    classification=collection.classification,
                    detected_format=collection.detected_format,
                    reason=collection.reason,
                )
            _verify_detected_format(path, observed_entry)
            consumed[relative_posix] = observed_entry
            if collection is not None:
                collection_entries[collection].append(observed_entry)

        missing = sorted(set(self.entries_by_path) - set(consumed))
        if missing:
            raise SourceQuarantineError(
                f"{self.project_id}: manifest entries were not discovered as source "
                f"candidates: {missing}"
            )
        for collection, observed_entries in collection_entries.items():
            observed_count = len(observed_entries)
            if observed_count != collection.expected_file_count:
                raise SourceQuarantineError(
                    f"{self.project_id}: quarantine collection "
                    f"{collection.relative_path_prefix!r} count mismatch: "
                    f"observed={observed_count} "
                    f"expected={collection.expected_file_count}"
                )
            observed_digest = _content_set_sha256(observed_entries)
            if observed_digest != collection.content_set_sha256:
                raise SourceQuarantineError(
                    f"{self.project_id}: quarantine collection "
                    f"{collection.relative_path_prefix!r} content-set SHA-256 "
                    f"mismatch: observed={observed_digest} "
                    f"expected={collection.content_set_sha256}"
                )
        quarantined_entries = [consumed[path].as_dict() for path in sorted(consumed)]
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "project_id": self.project_id,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "manifest_entry_count": self.manifest_entry_count,
            "project_manifest_entry_count": (
                len(self.entries_by_path) + len(self.collections)
            ),
            "candidate_count_before_quarantine": len(candidate_list),
            "candidate_count_after_quarantine": len(kept),
            "quarantined_count": len(quarantined_entries),
            "entries": quarantined_entries,
        }
        return kept, receipt
