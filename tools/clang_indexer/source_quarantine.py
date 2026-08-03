"""Exact, receipt-bound quarantine for non-parser inputs and crash fixtures.

The quarantine is deliberately narrow: it only accepts files whose relative
path, byte size, SHA-256 digest, and independently verifiable format all match a
versioned manifest entry. It is not a parse-error suppression mechanism.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

MANIFEST_SCHEMA = "cppmega.source_quarantine_manifest_v1"
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
    ("generated_binary_blob", "mixed_utf8_utf16le_c_array"),
    ("generated_executable_archive", "posix_shell_appended_zip"),
    ("mislabeled_non_cpp", "xml_utf16le"),
    ("mislabeled_non_cpp", "asn1_der_x509_certificate_pair"),
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


def _require_string(value: object, *, field: str, index: int) -> str:
    if not isinstance(value, str) or not value:
        raise SourceQuarantineError(
            f"entries[{index}].{field} must be a non-empty string"
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

    project_id = _require_string(raw["project_id"], field="project_id", index=index)
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
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
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
        if length_bytes == 0 or length_bytes > 4 or offset + 2 + length_bytes > len(payload):
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
                f"{entry.relative_path}: declared xml_utf16le but XML prefix "
                "is absent"
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
        compiler_contract = {
            "// RUN: not --crash %clang_cc1 %s 2>&1 | FileCheck %s",
            "// REQUIRES: crash-recovery",
            "// CHECK: prag\\",
            "// CHECK-NEXT: ma",
        }.issubset(decoded.splitlines()) and decoded.count(
            "#prag\\\nma clang __debug crash\n"
        ) == 1
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
        }.issubset(decoded.splitlines()) and decoded.splitlines().count(
            "#pragma clang __debug crash"
        ) == 1
        if not (compiler_contract or index_contract):
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

    if entry.detected_format == "mixed_utf8_utf16le_c_array":
        payload = path.read_bytes()
        marker = "/* \n\n   Input ELF file:".encode("utf-16le")
        boundary = payload.find(marker)
        if boundary <= 0:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared mixed_utf8_utf16le_c_array "
                "but the UTF-16LE generated-array header is absent"
            )
        try:
            prefix = payload[:boundary].decode("utf-8")
            generated = payload[boundary:].decode("utf-16le")
        except UnicodeDecodeError as exc:
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared mixed_utf8_utf16le_c_array "
                f"but an encoded section is invalid: {exc}"
            ) from exc
        required_prefix = (
            "Eclipse ThreadX contributors",
            "SPDX-License-Identifier: MIT",
        )
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
            not all(marker in prefix for marker in required_prefix)
            or not all(marker in generated for marker in required_generated)
            or len(byte_literals) < 1024
            or "\x00" in generated
            or not generated.rstrip().endswith("};")
        ):
            raise SourceQuarantineError(
                f"{entry.relative_path}: declared mixed_utf8_utf16le_c_array "
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
                    raise ValueError("wrapper does not contain one certificate SEQUENCE")
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

    @classmethod
    def load(
        cls,
        manifest_path: str | os.PathLike[str],
        *,
        project_id: str,
    ) -> "ProjectSourceQuarantine":
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
        if not isinstance(raw, dict) or set(raw) != {"schema", "entries"}:
            raise SourceQuarantineError(
                f"{path}: expected exactly schema and entries fields"
            )
        if raw["schema"] != MANIFEST_SCHEMA:
            raise SourceQuarantineError(
                f"{path}: unsupported schema {raw['schema']!r}; "
                f"expected {MANIFEST_SCHEMA!r}"
            )
        raw_entries = raw["entries"]
        if not isinstance(raw_entries, list):
            raise SourceQuarantineError(f"{path}: entries must be a list")
        parsed = [_parse_entry(item, index=index) for index, item in enumerate(raw_entries)]
        identities = [(entry.project_id, entry.relative_path) for entry in parsed]
        if len(set(identities)) != len(identities):
            raise SourceQuarantineError(f"{path}: duplicate project/path entries")
        project_entries = {
            entry.relative_path: entry
            for entry in parsed
            if entry.project_id == project_id
        }
        return cls(
            manifest_path=path,
            manifest_sha256=hashlib.sha256(payload).hexdigest(),
            manifest_entry_count=len(parsed),
            project_id=project_id,
            entries_by_path=project_entries,
        )

    def filter_candidates(
        self,
        project_root: str | os.PathLike[str],
        candidates: Iterable[str],
    ) -> tuple[list[str], dict[str, object]]:
        root = os.path.abspath(os.fspath(project_root))
        kept: list[str] = []
        consumed: dict[str, SourceQuarantineEntry] = {}
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
            if entry is None:
                kept.append(candidate)
                continue
            path = Path(absolute_candidate)
            try:
                observed_size = path.stat().st_size
            except OSError as exc:
                raise SourceQuarantineError(
                    f"cannot stat quarantined candidate {path}: {exc}"
                ) from exc
            if observed_size != entry.size_bytes:
                raise SourceQuarantineError(
                    f"{relative_posix}: quarantine size mismatch: "
                    f"observed={observed_size} expected={entry.size_bytes}"
                )
            observed_sha256 = _sha256_file(path)
            if observed_sha256 != entry.sha256:
                raise SourceQuarantineError(
                    f"{relative_posix}: quarantine SHA-256 mismatch: "
                    f"observed={observed_sha256} expected={entry.sha256}"
                )
            _verify_detected_format(path, entry)
            consumed[relative_posix] = entry

        missing = sorted(set(self.entries_by_path) - set(consumed))
        if missing:
            raise SourceQuarantineError(
                f"{self.project_id}: manifest entries were not discovered as source "
                f"candidates: {missing}"
            )
        quarantined_entries = [
            consumed[path].as_dict()
            for path in sorted(consumed)
        ]
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "project_id": self.project_id,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "manifest_entry_count": self.manifest_entry_count,
            "project_manifest_entry_count": len(self.entries_by_path),
            "candidate_count_before_quarantine": len(candidate_list),
            "candidate_count_after_quarantine": len(kept),
            "quarantined_count": len(quarantined_entries),
            "entries": quarantined_entries,
        }
        return kept, receipt
