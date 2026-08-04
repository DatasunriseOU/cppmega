"""Typed ingestion registry for build, shell, diagnostic, and embedded domains."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path
import os
import re
import subprocess
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence, cast

from cppmega.data.build_parsers import (
    parse_autoconf,
    parse_automake,
    parse_bazel,
    parse_cmake,
    parse_compile_commands,
    parse_configure,
    parse_dockerfile,
    parse_make,
    parse_meson,
    parse_ninja,
)
from cppmega.data.build_parsers.shell import parse_shell as parse_extended_shell
from cppmega.data.build_parsers.base import (
    ParsedDomainDocument,
    lex_text,
    strip_quotes,
)
from cppmega.data.diagnostic_parsers import (
    parse_build_error,
    parse_clang_diagnostic,
    parse_linker_error,
    parse_sanitizer_output,
    parse_test_output,
)
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)
from cppmega.data.python_parsers import parse_python
from cppmega.data.shell_parsers import (
    parse_bash,
    parse_ksh,
    parse_sh,
    parse_tcsh,
    parse_zsh,
)
from cppmega.data.source_identity import (
    MAX_ROW_LOCAL_DOC_ID,
    MAX_SOURCE_ID,
    SourceIdentity,
    source_identity,
)


Parser = Callable[[str], ParsedDomainDocument]


@dataclass(frozen=True)
class DomainParserAdapter:
    name: str
    domain: DomainKind
    parser: Parser


@dataclass(frozen=True)
class DiscoveredDomainFile:
    path: Path
    domain: DomainKind
    adapter: str


@dataclass(frozen=True)
class DomainFileChunk:
    path: Path
    domain: DomainKind
    adapter: str
    index: int
    text: str
    byte_start: int
    byte_end: int
    char_start: int
    char_end: int
    source_size_bytes: int
    chunk_limit_bytes: int
    split_reason: str
    source_encoding: str = "utf-8"
    source_trailing_nul_bytes: int = 0

    def source_span(self) -> dict[str, int | str]:
        """Return the exact original-file span represented by this chunk."""

        span: dict[str, int | str] = {
            "chunk_index": self.index,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "source_size_bytes": self.source_size_bytes,
            "chunk_limit_bytes": self.chunk_limit_bytes,
            "split_reason": self.split_reason,
        }
        if self.source_encoding != "utf-8":
            span["source_encoding"] = self.source_encoding
        if self.source_trailing_nul_bytes:
            span["source_trailing_nul_bytes"] = self.source_trailing_nul_bytes
        return span


@dataclass(frozen=True)
class _ValidatedDomainText:
    codec: str
    source_encoding: str
    bom: bytes
    signature_text: str
    trailing_nul_bytes: int
    allow_embedded_nul: bool = False


@dataclass(frozen=True)
class EmbeddedDomainBlock:
    start: int
    end: int
    domain: DomainKind
    parsed: ParsedDomainDocument
    cross_domain_edge: tuple[int, int, int]


_CPP_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cxx", ".c++", ".cp", ".h", ".hh", ".hpp",
    ".hxx", ".h++", ".inl", ".inc", ".ipp", ".tpp", ".txx", ".cu", ".cuh",
}
_CPP_KEYWORDS = {
    "alignas", "alignof", "asm", "auto", "bool", "break", "case", "catch",
    "char", "class", "concept", "const", "consteval", "constexpr", "constinit",
    "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype",
    "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
    "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
    "nullptr", "operator", "override", "private", "protected", "public", "register",
    "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "template", "this",
    "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
    "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while",
}
_SQL_KEYWORDS = {
    "ADD", "ALTER", "AS", "BEGIN", "BY", "COMMIT", "CREATE", "DELETE", "DROP",
    "EXEC", "EXISTS", "FROM", "GROUP", "HAVING", "INDEX", "INSERT", "INTO", "JOIN", "KEY",
    "LIMIT", "NOT", "NULL", "ON", "OR", "ORDER", "PRIMARY", "REFERENCES", "ROLLBACK",
    "PRAGMA", "SELECT", "SET", "SQL", "TABLE", "TRIGGER", "UNION", "UNIQUE", "UPDATE", "VALUES", "VIEW",
    "WHERE", "WITH",
}
_SQL_TARGET_PREDECESSORS = {"TABLE", "INTO", "UPDATE", "FROM", "JOIN", "VIEW", "INDEX"}

DOMAIN_INPUT_SIZE_LIMIT_BYTES = 500_000
DOMAIN_STREAM_READ_BYTES = 64 * 1024
DOMAIN_SIGNATURE_READ_BYTES = 4 * 1024
_ICONV_EUC_TW_CODEC = "iconv-euc-tw"
_FILENAME_DECLARED_CODECS = {
    "big5.sql": ("big5", "big5"),
    "euc_cn.sql": ("euc_cn", "euc-cn"),
    "euc_jp.sql": ("euc_jp", "euc-jp"),
    "euc_kr.sql": ("euc_kr", "euc-kr"),
    "euc_tw.sql": (_ICONV_EUC_TW_CODEC, "euc-tw"),
    "gb18030.sql": ("gb18030", "gb18030"),
    "sjis.sql": ("shift_jis", "shift-jis"),
}
_LARGE_DOMAIN_KINDS = frozenset(
    {
        DomainKind.CMAKE,
        DomainKind.MAKE,
        DomainKind.NINJA,
        DomainKind.BAZEL,
        DomainKind.AUTOCONF,
        DomainKind.AUTOMAKE,
        DomainKind.MESON,
        DomainKind.GN,
        DomainKind.SCONS,
        DomainKind.XMAKE,
        DomainKind.COMPILE_COMMANDS,
        DomainKind.CONFIGURE,
        DomainKind.BASH,
        DomainKind.ZSH,
        DomainKind.SH,
        DomainKind.TCSH,
        DomainKind.KSH,
        DomainKind.SQL,
        DomainKind.PYTHON,
        DomainKind.COMPILER_DIAGNOSTIC,
        DomainKind.BUILD_DIAGNOSTIC,
        DomainKind.COMPILER_ERROR,
        DomainKind.BUILD_ERROR,
        DomainKind.LINKER_ERROR,
        DomainKind.TEST_OUTPUT,
        DomainKind.LINKER_DIAGNOSTIC,
        DomainKind.SANITIZER_OUTPUT,
    }
)
_EXPLICIT_DOMAIN_NAMES = frozenset(
    {
        "CMakeLists.txt",
        "Makefile",
        "GNUmakefile",
        "makefile",
        "build.ninja",
        "BUILD",
        "BUILD.bazel",
        "WORKSPACE",
        "WORKSPACE.bazel",
        "MODULE.bazel",
        "configure",
        "configure.ac",
        "configure.in",
        "Makefile.am",
        "Makefile.in",
        "meson.build",
        "meson_options.txt",
        "BUILD.gn",
        "SConstruct",
        "SConscript",
        "xmake.lua",
        "compile_commands.json",
        "conanfile.txt",
        "conanfile.py",
        "vcpkg.json",
        "Dockerfile",
    }
)
_EXPLICIT_DOMAIN_SUFFIXES = frozenset(
    {
        ".cmake",
        ".mk",
        ".ninja",
        ".bzl",
        ".bash",
        ".sh",
        ".zsh",
        ".tcsh",
        ".csh",
        ".ksh",
        ".ps1",
        ".psm1",
        ".psd1",
        ".bat",
        ".cmd",
        ".sql",
        ".ddl",
        ".dml",
        ".psql",
        ".py",
        ".m4",
        ".gn",
        ".gni",
        ".vcxproj",
        ".sln",
    }
)
_TEXT_SIGNATURE_SUFFIXES = frozenset(
    {".diag", ".err", ".log", ".stderr", ".stdout", ".txt"}
)


@dataclass(frozen=True)
class _SqlLexState:
    mode: str = "normal"
    delimiter: bytes = b""


@dataclass(frozen=True)
class _JsonObjectLexState:
    object_depth: int = 0
    in_string: bool = False
    escaped: bool = False


_SQL_DOLLAR_QUOTE_RE = re.compile(rb"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
_SQL_ROWBINARY_LITERAL_RE = re.compile(
    rb"\bformat\s*\(\s*RowBinary\s*,\s*",
    re.IGNORECASE,
)


def _scan_domain_stream(
    stream: BinaryIO,
    *,
    path: Path,
    expected_size: int,
    codec: str,
    source_encoding: str,
    reject_raw_nul: bool,
    bom: bytes = b"",
    trailing_nul_bytes: int = 0,
) -> _ValidatedDomainText:
    stream.seek(0)
    if bom:
        actual_bom = stream.read(len(bom))
        if actual_bom != bom:
            raise ValueError(f"domain input encoding marker changed: {path}")
    decoder = codecs.getincrementaldecoder(codec)("strict")
    bytes_read = len(bom)
    signature_parts: list[str] = []
    signature_chars = 0
    content_end = expected_size - trailing_nul_bytes
    while bytes_read < content_end:
        block = stream.read(min(DOMAIN_STREAM_READ_BYTES, content_end - bytes_read))
        if not block:
            break
        bytes_read += len(block)
        if reject_raw_nul and b"\0" in block:
            raise ValueError(f"binary domain input contains NUL byte: {path}")
        decoded = decoder.decode(block, final=False)
        if "\0" in decoded:
            raise ValueError(f"decoded domain input contains NUL character: {path}")
        if signature_chars < DOMAIN_SIGNATURE_READ_BYTES:
            remaining = DOMAIN_SIGNATURE_READ_BYTES - signature_chars
            signature_parts.append(decoded[:remaining])
            signature_chars += len(decoded[:remaining])
    decoded = decoder.decode(b"", final=True)
    if "\0" in decoded:
        raise ValueError(f"decoded domain input contains NUL character: {path}")
    if signature_chars < DOMAIN_SIGNATURE_READ_BYTES:
        signature_parts.append(decoded[: DOMAIN_SIGNATURE_READ_BYTES - signature_chars])
    if trailing_nul_bytes:
        terminator = stream.read(trailing_nul_bytes)
        if terminator != b"\0" * trailing_nul_bytes:
            raise ValueError(f"domain input trailing NUL marker changed: {path}")
        bytes_read += trailing_nul_bytes
    if bytes_read != expected_size:
        raise OSError(
            f"domain input changed size while reading {path}: "
            f"expected {expected_size}, read {bytes_read}"
        )
    return _ValidatedDomainText(
        codec=codec,
        source_encoding=source_encoding,
        bom=bom,
        signature_text="".join(signature_parts),
        trailing_nul_bytes=trailing_nul_bytes,
    )


def _trailing_nul_bytes(
    stream: BinaryIO,
    *,
    expected_size: int,
    codec: str,
) -> int:
    width = 4 if codec.startswith("utf-32") else 2 if codec.startswith("utf-16") else 1
    if expected_size < width:
        return 0
    stream.seek(-width, os.SEEK_END)
    return width if stream.read(width) == b"\0" * width else 0


_MULE_INTERNAL_SIGNATURE = b"-- MULE \x92"


def _path_declared_codec(
    path: Path,
    *,
    source_prefix: bytes = b"",
) -> tuple[str, str] | None:
    if tuple(part.casefold() for part in path.parts[-5:]) == (
        "src",
        "test",
        "mb",
        "sql",
        "mule_internal.sql",
    ):
        if source_prefix.startswith(_MULE_INTERNAL_SIGNATURE):
            return "latin-1", "mule-internal"
        return None
    declared = _FILENAME_DECLARED_CODECS.get(path.name.casefold())
    if declared is not None:
        return declared
    if path.suffix.casefold() == ".bat" and any(
        part.casefold() == "jpn" for part in path.parts
    ):
        return "shift_jis", "shift-jis"
    return None


def _decode_euc_tw(payload: bytes, *, path: Path) -> str:
    def convert(source: str, target: str, raw: bytes) -> bytes:
        try:
            result = subprocess.run(
                ["iconv", "-f", source, "-t", target],
                input=raw,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise ValueError(
                f"cannot run iconv for path-declared EUC-TW input {path}: {exc}"
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()[:512]
            raise ValueError(
                f"invalid path-declared EUC-TW domain input {path}: {detail}"
            )
        return result.stdout

    utf8 = convert("EUC-TW", "UTF-8", payload)
    text = utf8.decode("utf-8", errors="strict")
    if convert("UTF-8", "EUC-TW", utf8) != payload:
        raise ValueError(
            f"path-declared EUC-TW input does not round-trip exactly: {path}"
        )
    return text


def _validate_euc_tw_stream(
    stream: BinaryIO,
    *,
    path: Path,
    expected_size: int,
    trailing_nul_bytes: int,
) -> _ValidatedDomainText:
    if expected_size > DOMAIN_INPUT_SIZE_LIMIT_BYTES:
        raise ValueError(
            "path-declared EUC-TW input exceeds the bounded native iconv "
            f"decoder limit of {DOMAIN_INPUT_SIZE_LIMIT_BYTES} bytes: {path}"
        )
    stream.seek(0)
    raw = stream.read(expected_size + 1)
    if len(raw) != expected_size:
        raise OSError(
            f"domain input changed size while reading {path}: "
            f"expected {expected_size}, read {len(raw)}"
        )
    payload_end = len(raw) - trailing_nul_bytes
    payload = raw[:payload_end]
    if b"\0" in payload:
        raise ValueError(f"binary domain input contains NUL byte: {path}")
    text = _decode_euc_tw(payload, path=path)
    if "\0" in text:
        raise ValueError(f"decoded domain input contains NUL character: {path}")
    return _ValidatedDomainText(
        codec=_ICONV_EUC_TW_CODEC,
        source_encoding="euc-tw",
        bom=b"",
        signature_text=text[:DOMAIN_SIGNATURE_READ_BYTES],
        trailing_nul_bytes=trailing_nul_bytes,
    )


def _sql_rowbinary_literal_is_byte_preserving(payload: bytes) -> bool:
    """Accept binary bytes only inside a complete RowBinary SQL literal."""

    def single_quote_end(quote_start: int) -> int | None:
        index = quote_start + 1
        while index < len(payload):
            byte = payload[index]
            following = payload[index + 1] if index + 1 < len(payload) else None
            if byte == 0x5C and following is not None:
                index += 2
                continue
            if byte == 0x27:
                if following == 0x27:
                    index += 2
                    continue
                return index
            index += 1
        return None

    def skip_space(index: int) -> int:
        while index < len(payload) and payload[index] in b" \t\r\n":
            index += 1
        return index

    literal_spans: list[tuple[int, int]] = []
    for match in _SQL_ROWBINARY_LITERAL_RE.finditer(payload):
        if len(literal_spans) >= 64:
            return False
        prefix = bytearray(payload[: match.start()])
        _statement_cut, _lexical_cut, state = _scan_sql_chunk_prefix(
            prefix,
            limit=len(prefix),
            initial_state=_SqlLexState(),
        )
        if state.mode != "normal":
            continue
        structure_start = skip_space(match.end())
        if structure_start >= len(payload) or payload[structure_start] != 0x27:
            continue
        structure_end = single_quote_end(structure_start)
        if structure_end is None:
            continue
        comma = skip_space(structure_end + 1)
        if comma >= len(payload) or payload[comma] != 0x2C:
            continue
        data_start = skip_space(comma + 1)
        if data_start >= len(payload) or payload[data_start] != 0x27:
            continue
        data_end = single_quote_end(data_start)
        if data_end is None:
            continue
        closing_paren = skip_space(data_end + 1)
        if closing_paren >= len(payload) or payload[closing_paren] != 0x29:
            continue
        literal_spans.append((data_start + 1, data_end))
    if not literal_spans:
        return False

    found_embedded_nul = False
    for index, byte in enumerate(payload):
        unsafe = (
            byte == 0
            or byte >= 0x80
            or (byte < 0x20 and byte not in b"\t\n\r")
        )
        if not unsafe:
            continue
        if not any(start <= index < end for start, end in literal_spans):
            return False
        found_embedded_nul = found_embedded_nul or byte == 0
    return found_embedded_nul


def _posix_shell_invalid_byte_test_is_byte_preserving(payload: bytes) -> bool:
    """Recognize a POSIX shell fixture that deliberately exercises raw bytes."""

    marker = b"# invalid 8-bit bytes\n"
    marker_end = payload.find(marker)
    if (
        not payload.startswith(b"#!/bin/sh\n")
        or marker_end < 0
        or b"\0" in payload
    ):
        return False
    marker_end += len(marker)
    try:
        payload[:marker_end].decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    found_invalid_fixture = False
    for line in payload[marker_end:].splitlines(keepends=True):
        try:
            line.decode("utf-8", errors="strict")
            continue
        except UnicodeDecodeError:
            pass
        prefix = b'test_ps "'
        if not line.startswith(prefix):
            return False
        quote_start = len(prefix) - 1
        index = quote_start + 1
        quote_end: int | None = None
        while index < len(line):
            if line[index] == 0x5C and index + 1 < len(line):
                index += 2
                continue
            if line[index] == 0x22:
                quote_end = index
                break
            index += 1
        if quote_end is None:
            return False
        try:
            line[: quote_start + 1].decode("utf-8", errors="strict")
            line[quote_end:].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False
        try:
            line[quote_start + 1 : quote_end].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            found_invalid_fixture = True
            continue
        return False
    return found_invalid_fixture


def _validate_byte_preserving_domain_stream(
    stream: BinaryIO,
    *,
    path: Path,
    expected_size: int,
    trailing_nul_bytes: int,
) -> _ValidatedDomainText | None:
    """Return a narrow one-byte codec contract for proven byte-oriented text."""

    if expected_size > DOMAIN_INPUT_SIZE_LIMIT_BYTES:
        return None
    stream.seek(0)
    raw = stream.read(expected_size + 1)
    if len(raw) != expected_size:
        raise OSError(
            f"domain input changed size while reading {path}: "
            f"expected {expected_size}, read {len(raw)}"
        )
    payload = raw[: expected_size - trailing_nul_bytes]
    suffix = path.suffix.casefold()
    allow_embedded_nul = (
        suffix == ".sql"
        and _sql_rowbinary_literal_is_byte_preserving(payload)
    )
    if allow_embedded_nul:
        accepted = True
    else:
        accepted = (
            suffix == ".sh"
            and _posix_shell_invalid_byte_test_is_byte_preserving(payload)
        )
    if not accepted:
        return None
    text = payload.decode("latin-1", errors="strict")
    if text.encode("latin-1", errors="strict") != payload:
        raise ValueError(f"byte-preserving domain input did not round-trip: {path}")
    return _ValidatedDomainText(
        codec="latin-1",
        source_encoding="iso-8859-1",
        bom=b"",
        signature_text=text[:DOMAIN_SIGNATURE_READ_BYTES],
        trailing_nul_bytes=trailing_nul_bytes,
        allow_embedded_nul=allow_embedded_nul,
    )


def _validate_domain_stream(
    stream: BinaryIO,
    *,
    path: Path,
    expected_size: int,
) -> _ValidatedDomainText:
    stream.seek(0)
    marker = stream.read(4)
    if marker.startswith(codecs.BOM_UTF32_LE):
        trailing_nul_bytes = _trailing_nul_bytes(
            stream,
            expected_size=expected_size,
            codec="utf-32-le",
        )
        try:
            return _scan_domain_stream(
                stream,
                path=path,
                expected_size=expected_size,
                codec="utf-32-le",
                source_encoding="utf-32-le",
                bom=codecs.BOM_UTF32_LE,
                reject_raw_nul=False,
                trailing_nul_bytes=trailing_nul_bytes,
            )
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-32LE domain input {path}: {exc}") from exc
    if marker.startswith(codecs.BOM_UTF32_BE):
        trailing_nul_bytes = _trailing_nul_bytes(
            stream,
            expected_size=expected_size,
            codec="utf-32-be",
        )
        try:
            return _scan_domain_stream(
                stream,
                path=path,
                expected_size=expected_size,
                codec="utf-32-be",
                source_encoding="utf-32-be",
                bom=codecs.BOM_UTF32_BE,
                reject_raw_nul=False,
                trailing_nul_bytes=trailing_nul_bytes,
            )
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-32BE domain input {path}: {exc}") from exc
    if marker.startswith(codecs.BOM_UTF16_LE):
        trailing_nul_bytes = _trailing_nul_bytes(
            stream,
            expected_size=expected_size,
            codec="utf-16-le",
        )
        try:
            return _scan_domain_stream(
                stream,
                path=path,
                expected_size=expected_size,
                codec="utf-16-le",
                source_encoding="utf-16-le",
                bom=codecs.BOM_UTF16_LE,
                reject_raw_nul=False,
                trailing_nul_bytes=trailing_nul_bytes,
            )
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-16LE domain input {path}: {exc}") from exc
    if marker.startswith(codecs.BOM_UTF16_BE):
        trailing_nul_bytes = _trailing_nul_bytes(
            stream,
            expected_size=expected_size,
            codec="utf-16-be",
        )
        try:
            return _scan_domain_stream(
                stream,
                path=path,
                expected_size=expected_size,
                codec="utf-16-be",
                source_encoding="utf-16-be",
                bom=codecs.BOM_UTF16_BE,
                reject_raw_nul=False,
                trailing_nul_bytes=trailing_nul_bytes,
            )
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-16BE domain input {path}: {exc}") from exc
    if marker.startswith(codecs.BOM_UTF8):
        trailing_nul_bytes = _trailing_nul_bytes(
            stream,
            expected_size=expected_size,
            codec="utf-8",
        )
        try:
            return _scan_domain_stream(
                stream,
                path=path,
                expected_size=expected_size,
                codec="utf-8",
                source_encoding="utf-8-sig",
                bom=codecs.BOM_UTF8,
                reject_raw_nul=True,
                trailing_nul_bytes=trailing_nul_bytes,
            )
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-8 domain input {path}: {exc}") from exc
    trailing_nul_bytes = _trailing_nul_bytes(
        stream,
        expected_size=expected_size,
        codec="utf-8",
    )
    try:
        return _scan_domain_stream(
            stream,
            path=path,
            expected_size=expected_size,
            codec="utf-8",
            source_encoding="utf-8",
            reject_raw_nul=True,
            trailing_nul_bytes=trailing_nul_bytes,
        )
    except UnicodeDecodeError as utf8_exc:
        stream.seek(0)
        declared_codec = _path_declared_codec(
            path,
            source_prefix=stream.read(len(_MULE_INTERNAL_SIGNATURE)),
        )
        stream.seek(0)
        if declared_codec is not None:
            codec, source_encoding = declared_codec
            if codec == _ICONV_EUC_TW_CODEC:
                return _validate_euc_tw_stream(
                    stream,
                    path=path,
                    expected_size=expected_size,
                    trailing_nul_bytes=trailing_nul_bytes,
                )
            try:
                return _scan_domain_stream(
                    stream,
                    path=path,
                    expected_size=expected_size,
                    codec=codec,
                    source_encoding=source_encoding,
                    reject_raw_nul=True,
                    trailing_nul_bytes=trailing_nul_bytes,
                )
            except UnicodeDecodeError as declared_exc:
                raise ValueError(
                    f"invalid UTF-8 or path-declared {source_encoding} "
                    f"domain input {path}: utf-8={utf8_exc}; "
                    f"{source_encoding}={declared_exc}"
                ) from declared_exc
        try:
            return _scan_domain_stream(
                stream,
                path=path,
                expected_size=expected_size,
                codec="cp1252",
                source_encoding="windows-1252",
                reject_raw_nul=True,
                trailing_nul_bytes=trailing_nul_bytes,
            )
        except UnicodeDecodeError as cp1252_exc:
            byte_preserving = _validate_byte_preserving_domain_stream(
                stream,
                path=path,
                expected_size=expected_size,
                trailing_nul_bytes=trailing_nul_bytes,
            )
            if byte_preserving is not None:
                return byte_preserving
            raise ValueError(
                f"invalid UTF-8 or Windows-1252 domain input {path}: "
                f"utf-8={utf8_exc}; windows-1252={cp1252_exc}"
            ) from cp1252_exc
    except ValueError:
        byte_preserving = _validate_byte_preserving_domain_stream(
            stream,
            path=path,
            expected_size=expected_size,
            trailing_nul_bytes=trailing_nul_bytes,
        )
        if byte_preserving is not None:
            return byte_preserving
        raise


def _validate_domain_path(
    path: Path,
    *,
    expected_size: int,
) -> _ValidatedDomainText:
    with path.open("rb") as stream:
        return _validate_domain_stream(
            stream,
            path=path,
            expected_size=expected_size,
        )


def _decode_domain_bytes(
    raw: bytes,
    *,
    path: Path,
    validated: _ValidatedDomainText,
) -> str:
    selected = validated
    if not raw.startswith(selected.bom):
        raise ValueError(f"domain input encoding marker changed: {path}")
    if selected.trailing_nul_bytes and not raw.endswith(
        b"\0" * selected.trailing_nul_bytes
    ):
        raise ValueError(f"domain input trailing NUL marker changed: {path}")
    payload_end = len(raw) - selected.trailing_nul_bytes
    payload = raw[len(selected.bom) : payload_end]
    if selected.codec not in {
        "utf-16-le",
        "utf-16-be",
        "utf-32-le",
        "utf-32-be",
    } and b"\0" in payload and not selected.allow_embedded_nul:
        raise ValueError(f"binary domain input contains NUL byte: {path}")
    if selected.codec == _ICONV_EUC_TW_CODEC:
        return _decode_euc_tw(payload, path=path)
    try:
        text = payload.decode(selected.codec, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"invalid {selected.source_encoding} domain input {path}: {exc}"
        ) from exc
    if "\0" in text and not selected.allow_embedded_nul:
        raise ValueError(f"decoded domain input contains NUL character: {path}")
    return text


def _read_validated_domain_text(path: Path, *, expected_size: int) -> str:
    if expected_size > DOMAIN_INPUT_SIZE_LIMIT_BYTES:
        raise ValueError(
            f"whole domain read exceeds {DOMAIN_INPUT_SIZE_LIMIT_BYTES}-byte cap: {path}"
        )
    with path.open("rb") as stream:
        raw = stream.read(DOMAIN_INPUT_SIZE_LIMIT_BYTES + 1)
    if len(raw) != expected_size:
        raise OSError(
            f"domain input changed size while reading {path}: "
            f"expected {expected_size}, read {len(raw)}"
        )
    validated = _validate_domain_path(path, expected_size=expected_size)
    return _decode_domain_bytes(raw, path=path, validated=validated)


def _is_domain_text_integrity_error(exc: ValueError) -> bool:
    message = str(exc)
    return (
        message.startswith("binary domain input contains NUL byte")
        or message.startswith("decoded domain input contains NUL character")
        or message.startswith("invalid UTF-8")
        or message.startswith("invalid UTF-16")
        or message.startswith("invalid UTF-32")
        or message.startswith("invalid windows-1252")
        or message.startswith("invalid path-declared")
        or message.startswith("path-declared")
    )


def _utf8_boundary_at_or_before(data: bytearray, limit: int) -> int:
    cut = min(int(limit), len(data))
    while cut > 0 and cut < len(data) and data[cut] & 0xC0 == 0x80:
        cut -= 1
    if cut == 0:
        raise UnicodeDecodeError(
            "utf-8",
            bytes(data[: min(len(data), 4)]),
            0,
            min(len(data), 1),
            "no complete UTF-8 code point within chunk limit",
        )
    return cut


def _scan_sql_chunk_prefix(
    data: bytearray,
    *,
    limit: int,
    initial_state: _SqlLexState,
) -> tuple[int, int, _SqlLexState]:
    """Find SQL statement and neutral lexical boundaries in ``data[:limit]``."""

    mode = initial_state.mode
    delimiter = initial_state.delimiter
    statement_cut = 0
    lexical_cut = 0
    index = 0
    while index < limit:
        byte = data[index]
        following = data[index + 1] if index + 1 < len(data) else None

        if mode == "normal":
            if byte == 0x2D and following == 0x2D:  # -- line comment
                mode = "line_comment"
                index += 2
                continue
            if byte == 0x2F and following == 0x2A:  # /* block comment */
                mode = "block_comment"
                index += 2
                continue
            if byte == 0x27:
                mode = "single_quote"
            elif byte == 0x22:
                mode = "double_quote"
            elif byte == 0x60:
                mode = "backtick_quote"
            elif byte == 0x5B:
                mode = "bracket_quote"
            elif byte == 0x24:
                match = _SQL_DOLLAR_QUOTE_RE.match(data, index, limit)
                if match is not None:
                    mode = "dollar_quote"
                    delimiter = bytes(match.group())
                    index = match.end()
                    continue
            elif byte == 0x3B:  # ;
                statement_cut = index + 1
                lexical_cut = index + 1
            elif byte in b" \t\r\n,":
                lexical_cut = index + 1
        elif mode in {"single_quote", "double_quote", "backtick_quote"}:
            quote = {
                "single_quote": 0x27,
                "double_quote": 0x22,
                "backtick_quote": 0x60,
            }[mode]
            if byte == 0x5C and following is not None:
                index += 2
                continue
            if byte == quote:
                if following == quote:
                    index += 2
                    continue
                mode = "normal"
                lexical_cut = index + 1
        elif mode == "bracket_quote":
            if byte == 0x5D:
                if following == 0x5D:
                    index += 2
                    continue
                mode = "normal"
                lexical_cut = index + 1
        elif mode == "line_comment":
            if byte in {0x0A, 0x0D}:
                mode = "normal"
                lexical_cut = index + 1
        elif mode == "block_comment":
            if byte == 0x2A and following == 0x2F:
                mode = "normal"
                if index + 2 <= limit:
                    lexical_cut = index + 2
                index += 2
                continue
        elif mode == "dollar_quote":
            if delimiter and data.startswith(delimiter, index, limit):
                index += len(delimiter)
                mode = "normal"
                delimiter = b""
                lexical_cut = index
                continue
        else:
            raise ValueError(f"unknown SQL chunk lexical state {mode!r}")
        index += 1

    return statement_cut, lexical_cut, _SqlLexState(mode, delimiter)


def _sql_chunk_cut(
    data: bytearray,
    *,
    max_chunk_bytes: int,
    initial_state: _SqlLexState,
) -> tuple[int, str, _SqlLexState]:
    limit = _utf8_boundary_at_or_before(data, max_chunk_bytes)
    statement_cut, lexical_cut, hard_state = _scan_sql_chunk_prefix(
        data,
        limit=limit,
        initial_state=initial_state,
    )
    minimum_preferred = max(1, max_chunk_bytes // 2)
    if statement_cut >= minimum_preferred:
        return statement_cut, "sql_statement", _SqlLexState()
    if lexical_cut >= minimum_preferred:
        return lexical_cut, "sql_lexical_boundary", _SqlLexState()
    return limit, "hard_limit", hard_state


def _line_chunk_cut(
    data: bytearray,
    *,
    max_chunk_bytes: int,
) -> tuple[int, str]:
    """Prefer a complete line without allowing tiny or oversized chunks."""

    limit = _utf8_boundary_at_or_before(data, max_chunk_bytes)
    minimum_preferred = max(1, max_chunk_bytes // 2)
    newline = data.rfind(b"\n", minimum_preferred - 1, limit)
    if newline >= 0:
        return newline + 1, "line_boundary"
    return limit, "hard_limit"


def _compile_commands_chunk_cut(
    data: bytearray,
    *,
    max_chunk_bytes: int,
    initial_state: _JsonObjectLexState,
) -> tuple[int, str, _JsonObjectLexState]:
    """Prefer a complete top-level compilation-database entry.

    A compilation database is commonly emitted as one minified JSON array, so
    line boundaries are not meaningful. Structural scanning only recognizes
    braces outside JSON strings and carries lexical state across a hard split.
    """

    limit = _utf8_boundary_at_or_before(data, max_chunk_bytes)
    object_depth = initial_state.object_depth
    in_string = initial_state.in_string
    escaped = initial_state.escaped
    last_complete_object = 0

    for index in range(limit):
        byte = data[index]
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # double quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte == 0x7B:  # {
            object_depth += 1
        elif byte == 0x7D:  # }
            if object_depth > 0:
                object_depth -= 1
                if object_depth == 0:
                    last_complete_object = index + 1

    if last_complete_object:
        return (
            last_complete_object,
            "compile_commands_entry",
            _JsonObjectLexState(),
        )
    return (
        limit,
        "hard_limit",
        _JsonObjectLexState(object_depth, in_string, escaped),
    )


def _decodable_prefix_at_or_before(
    data: bytearray,
    *,
    limit: int,
    codec: str,
) -> int:
    cut = min(int(limit), len(data))
    if codec == "utf-8":
        return _utf8_boundary_at_or_before(data, cut)
    if codec in {"cp1252", "latin-1"}:
        return cut
    if codec not in {
        "utf-16-le",
        "utf-16-be",
        "utf-32-le",
        "utf-32-be",
        "big5",
        "euc_cn",
        "euc_jp",
        "euc_kr",
        "gb18030",
        "shift_jis",
    }:
        raise ValueError(f"unsupported domain source codec {codec!r}")
    step = 4 if codec.startswith("utf-32") else 2 if codec.startswith("utf-16") else 1
    cut -= cut % step
    while cut > 0:
        try:
            bytes(data[:cut]).decode(codec, errors="strict")
            return cut
        except UnicodeDecodeError:
            cut -= step
    raise UnicodeDecodeError(
        codec,
        bytes(data[: min(len(data), 4)]),
        0,
        min(len(data), 2),
        "no complete code point within chunk limit",
    )


def decode_domain_prefix(
    raw: bytes,
    *,
    path: str | Path,
    allow_trailing_nul: bool = False,
) -> str:
    """Decode a bounded discovery prefix without consuming an incomplete tail."""

    path_obj = Path(path)
    if raw.startswith(codecs.BOM_UTF32_LE):
        codec = "utf-32-le"
        payload = bytearray(raw[len(codecs.BOM_UTF32_LE) :])
    elif raw.startswith(codecs.BOM_UTF32_BE):
        codec = "utf-32-be"
        payload = bytearray(raw[len(codecs.BOM_UTF32_BE) :])
    elif raw.startswith(codecs.BOM_UTF16_LE):
        codec = "utf-16-le"
        payload = bytearray(raw[len(codecs.BOM_UTF16_LE) :])
    elif raw.startswith(codecs.BOM_UTF16_BE):
        codec = "utf-16-be"
        payload = bytearray(raw[len(codecs.BOM_UTF16_BE) :])
    else:
        payload_bytes = raw[len(codecs.BOM_UTF8) :] if raw.startswith(
            codecs.BOM_UTF8
        ) else raw
        if allow_trailing_nul and payload_bytes.endswith(b"\0"):
            payload_bytes = payload_bytes[:-1]
        if b"\0" in payload_bytes:
            raise ValueError(f"binary domain input contains NUL byte: {path_obj}")
        try:
            return codecs.getincrementaldecoder("utf-8")("strict").decode(
                payload_bytes,
                final=False,
            )
        except UnicodeDecodeError as utf8_exc:
            declared_codec = _path_declared_codec(
                path_obj,
                source_prefix=payload_bytes,
            )
            if declared_codec is not None:
                codec, source_encoding = declared_codec
                if codec == _ICONV_EUC_TW_CODEC:
                    return _decode_euc_tw(payload_bytes, path=path_obj)
                try:
                    return codecs.getincrementaldecoder(codec)("strict").decode(
                        payload_bytes,
                        final=False,
                    )
                except UnicodeDecodeError as declared_exc:
                    raise ValueError(
                        f"invalid UTF-8 or path-declared {source_encoding} "
                        f"domain input {path_obj}: utf-8={utf8_exc}; "
                        f"{source_encoding}={declared_exc}"
                    ) from declared_exc
            try:
                return codecs.getincrementaldecoder("cp1252")("strict").decode(
                    payload_bytes,
                    final=False,
                )
            except UnicodeDecodeError as cp1252_exc:
                raise ValueError(
                    f"invalid UTF-8 or Windows-1252 domain input {path_obj}: "
                    f"utf-8={utf8_exc}; windows-1252={cp1252_exc}"
                ) from cp1252_exc
    trailing_width = 4 if codec in {"utf-32-le", "utf-32-be"} else 2
    if allow_trailing_nul and payload.endswith(b"\0" * trailing_width):
        del payload[-trailing_width:]
    if not payload:
        return ""
    cut = _decodable_prefix_at_or_before(
        payload,
        limit=len(payload),
        codec=codec,
    )
    text = bytes(payload[:cut]).decode(codec, errors="strict")
    if "\0" in text:
        raise ValueError(f"decoded domain input contains NUL character: {path_obj}")
    return text


def _encoded_chunk_cut(
    data: bytearray,
    *,
    max_source_bytes: int,
    codec: str,
    sql: bool,
    sql_state: _SqlLexState,
    compile_commands: bool,
    json_state: _JsonObjectLexState,
) -> tuple[int, str, _SqlLexState, _JsonObjectLexState]:
    source_limit = _decodable_prefix_at_or_before(
        data,
        limit=max_source_bytes,
        codec=codec,
    )
    candidate_text = bytes(data[:source_limit]).decode(codec, errors="strict")
    normalized = bytearray(candidate_text.encode("utf-8"))
    if sql:
        normalized_cut, split_reason, next_state = _sql_chunk_cut(
            normalized,
            max_chunk_bytes=len(normalized),
            initial_state=sql_state,
        )
        next_json_state = json_state
    elif compile_commands:
        normalized_cut, split_reason, next_json_state = (
            _compile_commands_chunk_cut(
                normalized,
                max_chunk_bytes=len(normalized),
                initial_state=json_state,
            )
        )
        next_state = sql_state
    else:
        normalized_cut, split_reason = _line_chunk_cut(
            normalized,
            max_chunk_bytes=len(normalized),
        )
        next_state = sql_state
        next_json_state = json_state
    selected_text = bytes(normalized[:normalized_cut]).decode(
        "utf-8",
        errors="strict",
    )
    source_cut = len(selected_text.encode(codec, errors="strict"))
    if source_cut <= 0 or source_cut > source_limit:
        raise ValueError(
            "domain chunk boundary could not be mapped to original encoding: "
            f"codec={codec} source_cut={source_cut} source_limit={source_limit}"
        )
    return source_cut, split_reason, next_state, next_json_state


def _is_explicit_domain_path(path: Path, *, include_cpp: bool) -> bool:
    if path.name in _EXPLICIT_DOMAIN_NAMES:
        return True
    suffix = path.suffix.lower()
    if suffix in _EXPLICIT_DOMAIN_SUFFIXES:
        return True
    return include_cpp and suffix in _CPP_EXTENSIONS


def _allows_domain_content_signatures(path: Path) -> bool:
    suffix = path.suffix.lower()
    return not suffix or suffix in _TEXT_SIGNATURE_SUFFIXES


def _large_domain_is_chunkable(path: Path, adapter: DomainParserAdapter) -> bool:
    if adapter.domain in _LARGE_DOMAIN_KINDS:
        return True
    return adapter.name == "raw-output" and _is_explicit_domain_path(
        path,
        include_cpp=False,
    )


def iter_domain_file_chunks(
    path: str | Path,
    *,
    max_chunk_bytes: int = DOMAIN_INPUT_SIZE_LIMIT_BYTES,
) -> Iterator[DomainFileChunk]:
    """Yield complete, bounded text chunks for one typed domain input.

    Inputs are validated in a bounded first pass before any chunk is yielded, so
    invalid encodings or binary data cannot leave a partially ingested document.
    UTF-8, BOM-marked UTF-16/32, path-declared legacy SQL encodings, and
    strict Windows-1252 are decoded without replacement. Syntax-confined
    RowBinary SQL literals and explicit POSIX invalid-byte test sections use a
    one-byte ISO-8859-1 round trip; structural NULs and unmarked invalid shell
    bytes still fail closed. SQL prefers statement boundaries; build, shell,
    and diagnostic text prefers line boundaries. Every emitted chunk has an
    explicit hard byte cap and exact byte/character provenance into the
    original encoded file.
    """

    if not 4 <= max_chunk_bytes <= DOMAIN_INPUT_SIZE_LIMIT_BYTES:
        raise ValueError(
            "max_chunk_bytes must be between 4 and "
            f"{DOMAIN_INPUT_SIZE_LIMIT_BYTES}, got {max_chunk_bytes}"
        )
    path_obj = Path(path)
    try:
        with path_obj.open("rb") as stream:
            source_size = os.fstat(stream.fileno()).st_size
            validated = _validate_domain_stream(
                stream,
                path=path_obj,
                expected_size=source_size,
            )
            stream.seek(len(validated.bom))

            if source_size <= max_chunk_bytes:
                stream.seek(0)
                raw = stream.read(max_chunk_bytes + 1)
                if len(raw) != source_size:
                    raise OSError(
                        f"domain input changed size while reading {path_obj}: "
                        f"expected {source_size}, read {len(raw)}"
                    )
                text = _decode_domain_bytes(
                    raw,
                    path=path_obj,
                    validated=validated,
                )
                adapter = resolve_domain_parser(path_obj, text)
                yield DomainFileChunk(
                    path=path_obj,
                    domain=adapter.domain,
                    adapter=adapter.name,
                    index=0,
                    text=text,
                    byte_start=0,
                    byte_end=source_size,
                    char_start=0,
                    char_end=len(text),
                    source_size_bytes=source_size,
                    chunk_limit_bytes=max_chunk_bytes,
                    split_reason="eof",
                    source_encoding=validated.source_encoding,
                    source_trailing_nul_bytes=validated.trailing_nul_bytes,
                )
                return

            path_adapter = resolve_domain_parser(
                path_obj,
                validated.signature_text,
            )
            if not _large_domain_is_chunkable(path_obj, path_adapter):
                raise ValueError(
                    f"domain input exceeds {max_chunk_bytes}-byte limit and "
                    f"{path_adapter.domain.name} has no large-input chunk policy: {path_obj}"
                )

            buffer = bytearray()
            byte_start = 0
            source_cursor = len(validated.bom)
            char_start = 0
            chunk_index = 0
            sql_state = _SqlLexState()
            json_state = _JsonObjectLexState()
            payload_remaining = (
                source_size - len(validated.bom) - validated.trailing_nul_bytes
            )
            while payload_remaining:
                block = stream.read(
                    min(DOMAIN_STREAM_READ_BYTES, payload_remaining)
                )
                if not block:
                    raise OSError(
                        f"domain input changed size while reading {path_obj}: "
                        f"{payload_remaining} payload bytes missing"
                    )
                payload_remaining -= len(block)
                buffer.extend(block)
                leading_bytes = len(validated.bom) if chunk_index == 0 else 0
                payload_limit = (
                    max_chunk_bytes
                    - leading_bytes
                    - validated.trailing_nul_bytes
                )
                if payload_limit <= 0:
                    raise ValueError(
                        "max_chunk_bytes cannot hold source encoding markers: "
                        f"{path_obj}"
                    )
                while len(buffer) > payload_limit:
                    cut, split_reason, sql_state, json_state = _encoded_chunk_cut(
                        buffer,
                        max_source_bytes=payload_limit,
                        codec=validated.codec,
                        sql=path_adapter.domain == DomainKind.SQL,
                        sql_state=sql_state,
                        compile_commands=(
                            path_adapter.domain == DomainKind.COMPILE_COMMANDS
                        ),
                        json_state=json_state,
                    )
                    raw = bytes(buffer[:cut])
                    del buffer[:cut]
                    text = raw.decode(validated.codec, errors="strict")
                    source_cursor += len(raw)
                    byte_end = source_cursor
                    char_end = char_start + len(text)
                    yield DomainFileChunk(
                        path=path_obj,
                        domain=path_adapter.domain,
                        adapter=path_adapter.name,
                        index=chunk_index,
                        text=text,
                        byte_start=byte_start,
                        byte_end=byte_end,
                        char_start=char_start,
                        char_end=char_end,
                        source_size_bytes=source_size,
                        chunk_limit_bytes=max_chunk_bytes,
                        split_reason=split_reason,
                        source_encoding=validated.source_encoding,
                        source_trailing_nul_bytes=validated.trailing_nul_bytes,
                    )
                    byte_start = byte_end
                    char_start = char_end
                    chunk_index += 1
                    payload_limit = (
                        max_chunk_bytes - validated.trailing_nul_bytes
                    )

            raw = bytes(buffer)
            text = raw.decode(validated.codec, errors="strict")
            if "\0" in text and not validated.allow_embedded_nul:
                raise ValueError(
                    f"decoded domain input contains NUL character: {path_obj}"
                )
            terminator = stream.read(validated.trailing_nul_bytes)
            if terminator != b"\0" * validated.trailing_nul_bytes:
                raise ValueError(
                    f"domain input trailing NUL marker changed: {path_obj}"
                )
            source_cursor += len(raw) + validated.trailing_nul_bytes
            byte_end = source_cursor
            if byte_end != source_size:
                raise OSError(
                    f"domain input changed size while reading {path_obj}: "
                    f"expected {source_size}, read {byte_end}"
                )
            yield DomainFileChunk(
                path=path_obj,
                domain=path_adapter.domain,
                adapter=path_adapter.name,
                index=chunk_index,
                text=text,
                byte_start=byte_start,
                byte_end=byte_end,
                char_start=char_start,
                char_end=char_start + len(text),
                source_size_bytes=source_size,
                chunk_limit_bytes=max_chunk_bytes,
                split_reason="eof",
                source_encoding=validated.source_encoding,
                source_trailing_nul_bytes=validated.trailing_nul_bytes,
            )
    except OSError as exc:
        raise OSError(f"failed to read domain input {path_obj}: {exc}") from exc


def _tokens_overlapping(
    parsed: ParsedDomainDocument,
    start: int,
    end: int,
) -> list[int]:
    return [
        idx
        for idx, token in enumerate(parsed.tokens)
        if token.start < end and token.end > start
    ]


def parse_cpp_lexical(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.CPP,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"parser_adapter": "cpp-lexical", "language": "c-cpp"},
    )
    for idx, token in enumerate(doc.tokens):
        value = strip_quotes(token.text)
        if value in _CPP_KEYWORDS:
            doc.set_role(idx, DomainRoleKind.KEYWORD)
        elif token.text.startswith(('"', "'")):
            doc.set_role(idx, DomainRoleKind.STRING)
        elif value.isidentifier():
            doc.set_role(idx, DomainRoleKind.IDENTIFIER)

    comment_spans: list[tuple[int, int, DomainRoleKind]] = []
    for pattern, flags, role in (
        (r"(?m)//(?!/)[^\n]*$", 0, DomainRoleKind.COMMENT),
        (r"/\*(?!\*).*?\*/", re.DOTALL, DomainRoleKind.COMMENT),
        (r"(?m)///[^\n]*$", 0, DomainRoleKind.DOCSTRING),
        (r"/\*\*.*?\*/", re.DOTALL, DomainRoleKind.DOCSTRING),
        (r"(?m)^[ \t]*#.*$", 0, DomainRoleKind.PREPROCESSOR),
    ):
        for match in re.finditer(pattern, text, flags):
            comment_spans.append((match.start(), match.end(), role))
    for start, end, role in comment_spans:
        for idx in _tokens_overlapping(doc, start, end):
            doc.set_role(idx, role)
    return doc


def parse_sql_lexical(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.SQL,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"parser_adapter": "sql-lexical"},
    )
    previous_keyword: str | None = None
    for idx, token in enumerate(doc.tokens):
        value = strip_quotes(token.text)
        upper = value.upper()
        if upper in _SQL_KEYWORDS:
            doc.set_role(idx, DomainRoleKind.KEYWORD)
            previous_keyword = upper
        elif token.text.startswith(('"', "'")):
            doc.set_role(idx, DomainRoleKind.STRING)
        elif value.isidentifier():
            if previous_keyword in _SQL_TARGET_PREDECESSORS:
                doc.set_role(idx, DomainRoleKind.TARGET, entity=idx + 1)
            else:
                doc.set_role(idx, DomainRoleKind.IDENTIFIER)
            previous_keyword = None
        elif value not in {"(", ")", ",", ";", "."}:
            previous_keyword = None
    for quote in ('"', "'"):
        if len(re.findall(rf"(?<!\\){re.escape(quote)}", text)) % 2:
            return doc.mark_raw("malformed_sql_literal")
    return doc


def _parse_raw_output(text: str) -> ParsedDomainDocument:
    return ParsedDomainDocument.new(
        domain=DomainKind.TOOL_OUTPUT,
        text=text,
        confidence=ParseConfidence.RAW,
        metadata={
            "parser_adapter": "raw-output",
            "unsupported_syntax": "unrecognized_domain_path",
            "raw_reason": "unrecognized_domain_path",
        },
    )


def _raw_typed_parser(
    domain: DomainKind,
    adapter: str,
) -> Parser:
    """Keep a known dialect distinct when no native graph parser exists yet."""

    def parse(text: str) -> ParsedDomainDocument:
        return ParsedDomainDocument.new(
            domain=domain,
            text=text,
            confidence=ParseConfidence.RAW,
            metadata={
                "parser_adapter": adapter,
                "unsupported_syntax": f"{adapter}_native_parser_unavailable",
                "raw_reason": f"{adapter}_native_parser_unavailable",
            },
        )

    return parse


def parse_powershell(text: str) -> ParsedDomainDocument:
    """Parse PowerShell with its own dialect parser on the shared shell domain."""

    return parse_extended_shell(text, "powershell")


def parse_batch(text: str) -> ParsedDomainDocument:
    """Parse Windows batch/cmd with its own parser on the shared shell domain."""

    return parse_extended_shell(text, "cmd")


_ADAPTERS = {
    "cpp": DomainParserAdapter("cpp-lexical", DomainKind.CPP, parse_cpp_lexical),
    "cmake": DomainParserAdapter("cmake", DomainKind.CMAKE, parse_cmake),
    "make": DomainParserAdapter("make", DomainKind.MAKE, parse_make),
    "ninja": DomainParserAdapter("ninja", DomainKind.NINJA, parse_ninja),
    "bazel": DomainParserAdapter("bazel-starlark", DomainKind.BAZEL, parse_bazel),
    "configure": DomainParserAdapter("configure-shell", DomainKind.CONFIGURE, parse_configure),
    "autoconf": DomainParserAdapter("autoconf", DomainKind.AUTOCONF, parse_autoconf),
    "automake": DomainParserAdapter("automake", DomainKind.AUTOMAKE, parse_automake),
    "meson": DomainParserAdapter("meson", DomainKind.MESON, parse_meson),
    "gn": DomainParserAdapter(
        "gn-raw",
        DomainKind.GN,
        _raw_typed_parser(DomainKind.GN, "gn-raw"),
    ),
    "scons": DomainParserAdapter(
        "scons-raw",
        DomainKind.SCONS,
        _raw_typed_parser(DomainKind.SCONS, "scons-raw"),
    ),
    "xmake": DomainParserAdapter(
        "xmake-raw",
        DomainKind.XMAKE,
        _raw_typed_parser(DomainKind.XMAKE, "xmake-raw"),
    ),
    "compile_commands": DomainParserAdapter(
        "compile-commands-json",
        DomainKind.COMPILE_COMMANDS,
        parse_compile_commands,
    ),
    "dockerfile": DomainParserAdapter(
        "dockerfile",
        DomainKind.CONFIGURE,
        parse_dockerfile,
    ),
    "bash": DomainParserAdapter("bash", DomainKind.BASH, parse_bash),
    "sh": DomainParserAdapter("posix-sh", DomainKind.SH, parse_sh),
    "zsh": DomainParserAdapter("zsh", DomainKind.ZSH, parse_zsh),
    "tcsh": DomainParserAdapter("tcsh", DomainKind.TCSH, parse_tcsh),
    "ksh": DomainParserAdapter("ksh", DomainKind.KSH, parse_ksh),
    "powershell": DomainParserAdapter(
        "powershell",
        DomainKind.SH,
        parse_powershell,
    ),
    "cmd": DomainParserAdapter(
        "cmd",
        DomainKind.SH,
        parse_batch,
    ),
    "compiler": DomainParserAdapter(
        "clang-diagnostic",
        DomainKind.COMPILER_DIAGNOSTIC,
        cast(Parser, parse_clang_diagnostic),
    ),
    "linker": DomainParserAdapter(
        "linker-diagnostic",
        DomainKind.LINKER_DIAGNOSTIC,
        cast(Parser, parse_linker_error),
    ),
    "build-diagnostic": DomainParserAdapter(
        "build-diagnostic",
        DomainKind.BUILD_DIAGNOSTIC,
        cast(Parser, parse_build_error),
    ),
    "test": DomainParserAdapter("test-output", DomainKind.TEST_OUTPUT, parse_test_output),
    "sanitizer": DomainParserAdapter(
        "sanitizer-output", DomainKind.SANITIZER_OUTPUT, parse_sanitizer_output
    ),
    "sql": DomainParserAdapter("sql-lexical", DomainKind.SQL, parse_sql_lexical),
    "python": DomainParserAdapter(
        "python-ast-tokenize",
        DomainKind.PYTHON,
        parse_python,
    ),
    "raw": DomainParserAdapter("raw-output", DomainKind.TOOL_OUTPUT, _parse_raw_output),
}


def _script_kind_from_shebang(text: str) -> str | None:
    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if not first_line.startswith("#!"):
        return None
    words = re.findall(r"[A-Za-z0-9_+.-]+", first_line.lower())
    if "powershell" in words or "pwsh" in words:
        return "powershell"
    for kind in ("tcsh", "csh", "zsh", "bash", "ksh", "sh"):
        if kind in words:
            return "tcsh" if kind == "csh" else kind
    if any(word.startswith("python") for word in words):
        return "python"
    return None


def resolve_domain_parser(path: str | Path, text: str = "") -> DomainParserAdapter:
    """Resolve one parser using exact path contracts, then content signatures."""

    path_obj = Path(path)
    name = path_obj.name
    lower_name = name.lower()
    suffix = path_obj.suffix.lower()

    if name == "CMakeLists.txt":
        return _ADAPTERS["cmake"]
    if name in {"Makefile", "GNUmakefile", "makefile"}:
        return _ADAPTERS["make"]
    if name == "build.ninja":
        return _ADAPTERS["ninja"]
    if name in {"BUILD", "BUILD.bazel", "WORKSPACE", "WORKSPACE.bazel", "MODULE.bazel"}:
        return _ADAPTERS["bazel"]
    if name == "configure":
        return _ADAPTERS["configure"]
    if name in {"configure.ac", "configure.in"} or suffix == ".m4":
        return _ADAPTERS["autoconf"]
    if name in {"Makefile.am", "Makefile.in"}:
        return _ADAPTERS["automake"]
    if name in {"meson.build", "meson_options.txt"}:
        return _ADAPTERS["meson"]
    if name == "BUILD.gn":
        return _ADAPTERS["gn"]
    if name in {"SConstruct", "SConscript"}:
        return _ADAPTERS["scons"]
    if name == "xmake.lua":
        return _ADAPTERS["xmake"]
    if name == "compile_commands.json":
        return _ADAPTERS["compile_commands"]
    if name == "Dockerfile":
        return _ADAPTERS["dockerfile"]

    shebang_kind = _script_kind_from_shebang(text)
    if shebang_kind is not None:
        return _ADAPTERS[shebang_kind]

    if suffix in _CPP_EXTENSIONS:
        return _ADAPTERS["cpp"]
    if suffix == ".cmake":
        return _ADAPTERS["cmake"]
    if suffix == ".mk":
        return _ADAPTERS["make"]
    if suffix == ".ninja":
        return _ADAPTERS["ninja"]
    if suffix == ".bzl":
        return _ADAPTERS["bazel"]
    if suffix in {".bash"}:
        return _ADAPTERS["bash"]
    if suffix == ".sh":
        return _ADAPTERS["sh"]
    if suffix == ".zsh":
        return _ADAPTERS["zsh"]
    if suffix in {".tcsh", ".csh"}:
        return _ADAPTERS["tcsh"]
    if suffix == ".ksh":
        return _ADAPTERS["ksh"]
    if suffix in {".ps1", ".psm1", ".psd1"}:
        return _ADAPTERS["powershell"]
    if suffix in {".bat", ".cmd"}:
        return _ADAPTERS["cmd"]
    if suffix in {".gn", ".gni"}:
        return _ADAPTERS["gn"]
    if suffix in {".sql", ".ddl", ".dml", ".psql"}:
        return _ADAPTERS["sql"]
    if suffix == ".py":
        return _ADAPTERS["python"]

    # Known but currently raw build formats are still path-typed inputs. Do not
    # reinterpret their contents as diagnostics before the indexer's build-kind
    # adapter assigns the final domain identity.
    if _is_explicit_domain_path(path_obj, include_cpp=False):
        return _ADAPTERS["raw"]

    # Content signatures are only meaningful for text-like logs or
    # extensionless files. Arbitrary assets must not become diagnostics because
    # their basename starts with "link", "test", or "build", or because binary
    # bytes happen to contain one of the textual markers below.
    if not _allows_domain_content_signatures(path_obj):
        return _ADAPTERS["raw"]

    lower = text.lower()
    if any(tool.lower() in lower for tool in (
        "AddressSanitizer", "LeakSanitizer", "MemorySanitizer",
        "ThreadSanitizer", "UndefinedBehaviorSanitizer",
    )):
        return _ADAPTERS["sanitizer"]
    if lower_name.startswith(("test", "pytest")) or re.search(r"(?m)^(FAILED|PASSED)\s+\S+::", text):
        return _ADAPTERS["test"]
    if lower_name.startswith(("link", "ld.")) or re.search(
        r"(?im)^(?:ld|lld|link(?:\.exe)?):|undefined reference|unresolved external symbol",
        text,
    ):
        return _ADAPTERS["linker"]
    if re.search(
        r"(?m)^[^:\n]+:\d+:(?:\d+:)?\s*(?:fatal error|error|warning|note):",
        text,
    ):
        return _ADAPTERS["compiler"]
    if lower_name.startswith("build") or any(
        marker in lower for marker in ("ninja:", "cmake error", "build stopped")
    ):
        return _ADAPTERS["build-diagnostic"]
    return _ADAPTERS["raw"]


_CPP_RAW_SQL_RE = re.compile(
    r'R"(?P<tag>[A-Za-z0-9_]{0,16})\((?P<body>.*?)\)(?P=tag)"',
    re.DOTALL,
)
_CPP_ORDINARY_STRING_RE = re.compile(
    r'(?<![A-Za-z0-9_])(?:u8|u|U|L)?"(?P<body>(?:\\.|[^"\\])*)"',
    re.DOTALL,
)
_CPP_LITERAL_SEPARATOR_RE = re.compile(
    r"(?:\s|//[^\n]*(?:\n|$)|/\*.*?\*/)*\Z",
    re.DOTALL,
)
_SQL_CALL_RE = re.compile(
    r"\b(?P<call>sqlite3_exec|sqlite3_prepare(?:_v2)?|PQexec|PQprepare|mysql_query)\s*\(",
)
_SQL_CALL_QUERY_ARGUMENT = {
    "sqlite3_exec": 1,
    "sqlite3_prepare": 1,
    "sqlite3_prepare_v2": 1,
    "PQexec": 1,
    "PQprepare": 2,
    "mysql_query": 1,
}
_EXEC_SQL_RE = re.compile(r"(?im)\bEXEC[ \t]+SQL\b[^;]*;")


def _looks_like_sql_statement(text: str) -> bool:
    values = [strip_quotes(token.text).upper() for token in lex_text(text)]
    if not values:
        return False
    head = values[0]
    if head == "CREATE":
        return any(
            value in {"DATABASE", "INDEX", "SCHEMA", "TABLE", "TRIGGER", "VIEW"}
            for value in values[1:6]
        )
    if head == "ALTER":
        return len(values) > 1 and values[1] in {
            "DATABASE",
            "INDEX",
            "SCHEMA",
            "TABLE",
            "VIEW",
        }
    if head == "INSERT":
        return "INTO" in values[1:4]
    if head == "UPDATE":
        return (
            len(values) > 3
            and values[1] not in {":", "="}
            and "SET" in values[2:]
        )
    if head == "DELETE":
        return len(values) > 1 and values[1] == "FROM"
    if head == "SELECT":
        return len(values) > 1
    if head == "DROP":
        return len(values) > 1 and values[1] in {
            "DATABASE",
            "INDEX",
            "SCHEMA",
            "TABLE",
            "TRIGGER",
            "VIEW",
        }
    if head == "WITH":
        return "AS" in values[1:] and "(" in values[1:]
    if head == "PRAGMA":
        return len(values) > 1 and values[1] not in {":", "="}
    if head == "BEGIN":
        return len(values) == 1 or values[1] in {
            ";",
            "DEFERRED",
            "EXCLUSIVE",
            "IMMEDIATE",
            "TRANSACTION",
            "WORK",
        }
    return False


def _call_argument_spans(text: str, open_paren: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    argument_start = open_paren + 1
    depth = 0
    quote: str | None = None
    escaped = False
    idx = argument_start
    while idx < len(text):
        char = text[idx]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\" and quote != "'":
                escaped = True
            elif char == quote:
                quote = None
        elif text.startswith('R"', idx):
            raw_literal = _CPP_RAW_SQL_RE.match(text, idx)
            if raw_literal is None:
                return []
            idx = raw_literal.end()
            continue
        elif text.startswith("//", idx):
            newline = text.find("\n", idx + 2)
            idx = len(text) if newline < 0 else newline
            continue
        elif text.startswith("/*", idx):
            comment_end = text.find("*/", idx + 2)
            if comment_end < 0:
                return []
            idx = comment_end + 2
            continue
        elif char in {'"', "'"}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            if char == ")" and depth == 0:
                spans.append((argument_start, idx))
                return spans
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            spans.append((argument_start, idx))
            argument_start = idx + 1
        idx += 1
    return []


def _sql_call_query_spans(text: str) -> list[tuple[int, int, int]]:
    query_spans: list[tuple[int, int, int]] = []
    for call_match in _SQL_CALL_RE.finditer(text):
        argument_spans = _call_argument_spans(text, call_match.end() - 1)
        query_index = _SQL_CALL_QUERY_ARGUMENT[call_match.group("call")]
        if query_index >= len(argument_spans):
            continue
        argument_start, argument_end = argument_spans[query_index]
        query_spans.append((argument_start, argument_end, call_match.start("call")))
    return query_spans


def _adjacent_cpp_string_body_groups(
    text: str,
    start: int,
    end: int,
) -> list[list[tuple[int, int, str]]]:
    matches = list(_CPP_ORDINARY_STRING_RE.finditer(text, start, end))
    groups: list[list[tuple[int, int, str]]] = []
    current: list[tuple[int, int, str]] = []
    previous_end = start
    for match in matches:
        separator = text[previous_end:match.start()]
        item = (*match.span("body"), match.group("body"))
        if current and _CPP_LITERAL_SEPARATOR_RE.fullmatch(separator) is None:
            groups.append(current)
            current = []
        current.append(item)
        previous_end = match.end()
    if current:
        groups.append(current)
    return groups


def extract_embedded_domain_blocks(
    text: str,
    *,
    host_domain: DomainKind,
    source_doc_id: int | None = None,
    source_identity_value: SourceIdentity | None = None,
) -> list[EmbeddedDomainBlock]:
    """Extract exact host-character spans for supported embedded languages."""

    if DomainKind(host_domain) != DomainKind.CPP:
        return []
    resolved_source_doc_id = 1 if source_doc_id in (None, 0) else int(source_doc_id)
    if not 0 < resolved_source_doc_id <= MAX_ROW_LOCAL_DOC_ID:
        raise ValueError(
            f"source_doc_id must be positive uint32, got {resolved_source_doc_id}"
        )
    resolved_identity = source_identity_value or source_identity({"text": text})
    if not 0 < resolved_identity.source_identity_id <= MAX_SOURCE_ID:
        raise ValueError("source identity must be positive uint64")
    candidates: list[tuple[int, int, int]] = []
    query_call_spans = _sql_call_query_spans(text)
    for match in _CPP_RAW_SQL_RE.finditer(text):
        body = match.group("body")
        tag = match.group("tag").upper()
        if not body.strip():
            continue
        if "SQL" not in tag and not _looks_like_sql_statement(body):
            continue
        start, end = match.span("body")
        owner = next(
            (
                call_start
                for argument_start, argument_end, call_start in query_call_spans
                if argument_start <= match.start() and match.end() <= argument_end
            ),
            None,
        )
        candidates.append((start, end, start if owner is None else owner))

    for argument_start, argument_end, call_anchor in query_call_spans:
        for group in _adjacent_cpp_string_body_groups(
            text,
            argument_start,
            argument_end,
        ):
            if not _looks_like_sql_statement(
                "".join(body for _, _, body in group)
            ):
                continue
            candidates.extend((start, end, call_anchor) for start, end, _ in group)

    for match in _EXEC_SQL_RE.finditer(text):
        candidates.append((match.start(), match.end(), match.start()))

    blocks: list[EmbeddedDomainBlock] = []
    previous_end = -1
    for start, end, call_anchor in sorted(set(candidates)):
        if start < previous_end:
            continue
        body = text[start:end]
        parsed = parse_sql_lexical(body)
        parsed.set_source_doc_id(resolved_source_doc_id)
        parsed.set_source_identity(resolved_identity)
        parsed.metadata["embedded_in"] = host_domain.name
        blocks.append(
            EmbeddedDomainBlock(
                start=start,
                end=end,
                domain=DomainKind.SQL,
                parsed=parsed,
                cross_domain_edge=(
                    call_anchor,
                    start,
                    int(DomainEdgeKind.EMBEDDED_DOMAIN),
                ),
            )
        )
        previous_end = end
    return blocks


def parse_domain_document(
    path: str | Path,
    text: str,
    *,
    source_doc_id: int | None = None,
    provenance: Mapping[str, object] | None = None,
    diagnostic_links: Sequence[Mapping[str, object]] | None = None,
) -> ParsedDomainDocument:
    """Parse one file and enforce token-aligned domain/provenance vectors."""

    adapter = resolve_domain_parser(path, text)
    resolved_source_doc_id = 1 if source_doc_id in (None, 0) else int(source_doc_id)
    if not 0 < resolved_source_doc_id <= MAX_ROW_LOCAL_DOC_ID:
        raise ValueError(
            f"source_doc_id must be positive uint32, got {resolved_source_doc_id}"
        )
    resolved_identity = source_identity(
        {
            **dict(provenance or {}),
            "source_path": str(path),
            "text": text,
        }
    )
    parsed = adapter.parser(text)
    if parsed.domain != adapter.domain:
        allowed_dynamic = {
            "clang-diagnostic": {DomainKind.COMPILER_DIAGNOSTIC, DomainKind.COMPILER_ERROR},
            "linker-diagnostic": {DomainKind.LINKER_DIAGNOSTIC, DomainKind.LINKER_ERROR},
            "build-diagnostic": {DomainKind.BUILD_DIAGNOSTIC, DomainKind.BUILD_ERROR},
        }
        if parsed.domain not in allowed_dynamic.get(adapter.name, set()):
            raise ValueError(
                f"parser adapter {adapter.name} returned {parsed.domain.name}, "
                f"expected {adapter.domain.name}"
            )
    parsed.metadata["parser_adapter"] = adapter.name
    parsed.metadata["source_path"] = str(path)
    parsed.set_source_doc_id(resolved_source_doc_id)
    parsed.set_source_identity(resolved_identity)
    if provenance is not None:
        parsed.metadata["provenance"] = dict(provenance)
    if diagnostic_links is not None:
        parsed.metadata["diagnostic_links"] = [dict(link) for link in diagnostic_links]
    if parsed.domain == DomainKind.CPP:
        parsed.embedded_blocks = extract_embedded_domain_blocks(
            text,
            host_domain=DomainKind.CPP,
            source_doc_id=resolved_source_doc_id,
            source_identity_value=resolved_identity,
        )
    parsed.validate()
    return parsed


def discover_project_domain_files(
    root: str | Path,
    *,
    extra_exclude_dirs: set[str] | None = None,
    include_cpp: bool = True,
    invalid_input_handler: Callable[[Path, ValueError], None] | None = None,
    excluded_paths: set[str] | None = None,
) -> list[DiscoveredDomainFile]:
    """Discover typed text inputs without opening unrelated binary assets."""

    root_path = Path(root)
    skip_dirs = {".git"} | (extra_exclude_dirs or set())
    excluded = {os.path.abspath(path) for path in (excluded_paths or set())}
    discovered: list[DiscoveredDomainFile] = []

    def candidate_paths() -> Iterator[Path]:
        for directory, dirs, filenames in os.walk(root_path):
            dirs[:] = sorted(name for name in dirs if name not in skip_dirs)
            for name in sorted(filenames):
                yield Path(directory) / name

    for path in candidate_paths():
        if os.path.abspath(path) in excluded:
            continue
        explicit_candidate = _is_explicit_domain_path(path, include_cpp=include_cpp)
        signature_candidate = _allows_domain_content_signatures(path)
        if not explicit_candidate and not signature_candidate:
            continue

        ambiguous_extensionless = not explicit_candidate and not path.suffix
        try:
            source_size = path.stat().st_size
            if ambiguous_extensionless:
                with path.open("rb") as stream:
                    prefix = stream.read(DOMAIN_SIGNATURE_READ_BYTES)
                try:
                    prefix_text = decode_domain_prefix(prefix, path=path)
                except ValueError:
                    continue
                prefix_adapter = resolve_domain_parser(path, prefix_text)
                implicit_typed = prefix_adapter.name != "raw-output"
                if source_size > DOMAIN_INPUT_SIZE_LIMIT_BYTES and not implicit_typed:
                    continue
            else:
                implicit_typed = False

            if source_size > DOMAIN_INPUT_SIZE_LIMIT_BYTES:
                validated = _validate_domain_path(
                    path,
                    expected_size=source_size,
                )
                adapter = resolve_domain_parser(
                    path,
                    validated.signature_text,
                )
                if adapter.name == "raw-output" and not explicit_candidate:
                    continue
                if not _large_domain_is_chunkable(path, adapter):
                    raise ValueError(
                        "domain input exceeds "
                        f"{DOMAIN_INPUT_SIZE_LIMIT_BYTES}-byte limit and "
                        f"{adapter.domain.name} has no large-input chunk policy: {path}"
                    )
            else:
                try:
                    text = _read_validated_domain_text(path, expected_size=source_size)
                except ValueError:
                    if ambiguous_extensionless and not implicit_typed:
                        continue
                    raise
                adapter = resolve_domain_parser(path, text)
        except ValueError as exc:
            # Signature-based candidates are opportunistic. Repository trees
            # commonly contain generated/binary ``.txt`` and log artifacts;
            # malformed bytes in those files must not abort discovery of the
            # rest of the project. Explicit domain paths and extensionless
            # files that resolved to a typed adapter remain fail-closed.
            if (
                signature_candidate
                and not explicit_candidate
                and not ambiguous_extensionless
                and _is_domain_text_integrity_error(exc)
            ):
                continue
            if invalid_input_handler is not None and _is_domain_text_integrity_error(exc):
                invalid_input_handler(path, exc)
                continue
            raise
        except OSError as exc:
            if explicit_candidate or not ambiguous_extensionless:
                raise OSError(f"failed to read domain input {path}: {exc}") from exc
            continue

        if adapter.name == "raw-output" or (
            not include_cpp and adapter.domain == DomainKind.CPP
        ):
            continue
        discovered.append(
            DiscoveredDomainFile(path=path, domain=adapter.domain, adapter=adapter.name)
        )
    return discovered


__all__ = [
    "DOMAIN_INPUT_SIZE_LIMIT_BYTES",
    "DiscoveredDomainFile",
    "DomainFileChunk",
    "DomainParserAdapter",
    "decode_domain_prefix",
    "EmbeddedDomainBlock",
    "discover_project_domain_files",
    "extract_embedded_domain_blocks",
    "iter_domain_file_chunks",
    "parse_cpp_lexical",
    "parse_batch",
    "parse_domain_document",
    "parse_powershell",
    "parse_python",
    "parse_sql_lexical",
    "resolve_domain_parser",
]
