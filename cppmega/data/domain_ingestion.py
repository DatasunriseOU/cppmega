"""Typed ingestion registry for build, shell, diagnostic, and embedded domains."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence, cast

from cppmega.data.build_parsers import (
    parse_autoconf,
    parse_automake,
    parse_bazel,
    parse_cmake,
    parse_configure,
    parse_make,
    parse_ninja,
)
from cppmega.data.build_parsers.base import ParsedDomainDocument, strip_quotes
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

    def source_span(self) -> dict[str, int | str]:
        """Return the exact original-file span represented by this chunk."""

        return {
            "chunk_index": self.index,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "source_size_bytes": self.source_size_bytes,
            "chunk_limit_bytes": self.chunk_limit_bytes,
            "split_reason": self.split_reason,
        }


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
        ".sql",
        ".py",
        ".m4",
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


_SQL_DOLLAR_QUOTE_RE = re.compile(rb"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")


def _validate_utf8_domain_stream(
    stream: BinaryIO,
    *,
    path: Path,
    expected_size: int,
) -> bytes:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    bytes_read = 0
    prefix = bytearray()
    try:
        while block := stream.read(DOMAIN_STREAM_READ_BYTES):
            bytes_read += len(block)
            if b"\0" in block:
                raise ValueError(f"binary domain input contains NUL byte: {path}")
            decoder.decode(block, final=False)
            if len(prefix) < DOMAIN_SIGNATURE_READ_BYTES:
                remaining = DOMAIN_SIGNATURE_READ_BYTES - len(prefix)
                prefix.extend(block[:remaining])
        decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 domain input {path}: {exc}") from exc
    if bytes_read != expected_size:
        raise OSError(
            f"domain input changed size while reading {path}: "
            f"expected {expected_size}, read {bytes_read}"
        )
    return bytes(prefix)


def _validate_utf8_domain_path(path: Path, *, expected_size: int) -> bytes:
    with path.open("rb") as stream:
        return _validate_utf8_domain_stream(
            stream,
            path=path,
            expected_size=expected_size,
        )


def _decode_domain_bytes(raw: bytes, *, path: Path) -> str:
    if b"\0" in raw:
        raise ValueError(f"binary domain input contains NUL byte: {path}")
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 domain input {path}: {exc}") from exc


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
    return _decode_domain_bytes(raw, path=path)


def _is_domain_text_integrity_error(exc: ValueError) -> bool:
    message = str(exc)
    return message.startswith("binary domain input contains NUL byte") or message.startswith(
        "invalid UTF-8 domain input"
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
    """Yield complete, bounded UTF-8 chunks for one typed domain input.

    Inputs are validated in a bounded first pass before any chunk is yielded, so
    invalid UTF-8 or binary data cannot leave a partially ingested document.
    SQL prefers statement boundaries; build, shell, and diagnostic text prefers
    line boundaries. Every emitted chunk has an explicit hard byte cap and exact
    byte/character provenance into the original file.
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
            signature_prefix = _validate_utf8_domain_stream(
                stream,
                path=path_obj,
                expected_size=source_size,
            )
            stream.seek(0)

            if source_size <= max_chunk_bytes:
                raw = stream.read(max_chunk_bytes + 1)
                if len(raw) != source_size:
                    raise OSError(
                        f"domain input changed size while reading {path_obj}: "
                        f"expected {source_size}, read {len(raw)}"
                    )
                text = _decode_domain_bytes(raw, path=path_obj)
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
                )
                return

            prefix_text = _decode_domain_bytes(signature_prefix, path=path_obj)
            path_adapter = resolve_domain_parser(path_obj, prefix_text)
            if not _large_domain_is_chunkable(path_obj, path_adapter):
                raise ValueError(
                    f"domain input exceeds {max_chunk_bytes}-byte limit and "
                    f"{path_adapter.domain.name} has no large-input chunk policy: {path_obj}"
                )

            buffer = bytearray()
            byte_start = 0
            char_start = 0
            chunk_index = 0
            sql_state = _SqlLexState()
            while block := stream.read(DOMAIN_STREAM_READ_BYTES):
                buffer.extend(block)
                while len(buffer) > max_chunk_bytes:
                    if path_adapter.domain == DomainKind.SQL:
                        cut, split_reason, sql_state = _sql_chunk_cut(
                            buffer,
                            max_chunk_bytes=max_chunk_bytes,
                            initial_state=sql_state,
                        )
                    else:
                        cut, split_reason = _line_chunk_cut(
                            buffer,
                            max_chunk_bytes=max_chunk_bytes,
                        )
                    raw = bytes(buffer[:cut])
                    del buffer[:cut]
                    text = _decode_domain_bytes(raw, path=path_obj)
                    byte_end = byte_start + len(raw)
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
                    )
                    byte_start = byte_end
                    char_start = char_end
                    chunk_index += 1

            raw = bytes(buffer)
            text = _decode_domain_bytes(raw, path=path_obj)
            byte_end = byte_start + len(raw)
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


_ADAPTERS = {
    "cpp": DomainParserAdapter("cpp-lexical", DomainKind.CPP, parse_cpp_lexical),
    "cmake": DomainParserAdapter("cmake", DomainKind.CMAKE, parse_cmake),
    "make": DomainParserAdapter("make", DomainKind.MAKE, parse_make),
    "ninja": DomainParserAdapter("ninja", DomainKind.NINJA, parse_ninja),
    "bazel": DomainParserAdapter("bazel-starlark", DomainKind.BAZEL, parse_bazel),
    "configure": DomainParserAdapter("configure-shell", DomainKind.CONFIGURE, parse_configure),
    "autoconf": DomainParserAdapter("autoconf", DomainKind.AUTOCONF, parse_autoconf),
    "automake": DomainParserAdapter("automake", DomainKind.AUTOMAKE, parse_automake),
    "bash": DomainParserAdapter("bash", DomainKind.BASH, parse_bash),
    "sh": DomainParserAdapter("posix-sh", DomainKind.SH, parse_sh),
    "zsh": DomainParserAdapter("zsh", DomainKind.ZSH, parse_zsh),
    "tcsh": DomainParserAdapter("tcsh", DomainKind.TCSH, parse_tcsh),
    "ksh": DomainParserAdapter("ksh", DomainKind.KSH, parse_ksh),
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
    if suffix == ".sql":
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
_SQL_PREFIX_RE = re.compile(
    r"^\s*(?:CREATE|ALTER|INSERT|UPDATE|DELETE|SELECT|DROP|WITH|PRAGMA|BEGIN)\b",
    re.IGNORECASE,
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
        if "SQL" not in tag and not _SQL_PREFIX_RE.match(body):
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
            if not _SQL_PREFIX_RE.match("".join(body for _, _, body in group)):
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
) -> list[DiscoveredDomainFile]:
    """Discover typed text inputs without opening unrelated binary assets."""

    root_path = Path(root)
    skip_dirs = {".git"} | (extra_exclude_dirs or set())
    discovered: list[DiscoveredDomainFile] = []

    def candidate_paths() -> Iterator[Path]:
        for directory, dirs, filenames in os.walk(root_path):
            dirs[:] = sorted(name for name in dirs if name not in skip_dirs)
            for name in sorted(filenames):
                yield Path(directory) / name

    for path in candidate_paths():
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
                    prefix_text = _decode_domain_bytes(prefix, path=path)
                except ValueError:
                    continue
                prefix_adapter = resolve_domain_parser(path, prefix_text)
                implicit_typed = prefix_adapter.name != "raw-output"
                if source_size > DOMAIN_INPUT_SIZE_LIMIT_BYTES and not implicit_typed:
                    continue
            else:
                implicit_typed = False

            if source_size > DOMAIN_INPUT_SIZE_LIMIT_BYTES:
                prefix = _validate_utf8_domain_path(path, expected_size=source_size)
                adapter = resolve_domain_parser(
                    path,
                    _decode_domain_bytes(prefix, path=path),
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
    "EmbeddedDomainBlock",
    "discover_project_domain_files",
    "extract_embedded_domain_blocks",
    "iter_domain_file_chunks",
    "parse_cpp_lexical",
    "parse_domain_document",
    "parse_python",
    "parse_sql_lexical",
    "resolve_domain_parser",
]
