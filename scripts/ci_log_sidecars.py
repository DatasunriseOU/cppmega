#!/usr/bin/env python3
"""Deterministic GitHub Actions log canonicalization and CI sidecar extraction.

The module deliberately has no cppmega or third-party imports.  Character
spans are offsets into ``canonical_text`` and the emitted role/domain/edge IDs
match the frozen cppmega domain-sidecar v1 integer contract.

``canonicalize_ci_log`` is the public entry point.  It accepts either the
original job-log bytes or already-decoded text and returns JSON-serializable
canonical text, deduplication text, section-aligned chunks, and a provenance
sidecar.  The canonicalizer never truncates payload lines.
"""

from __future__ import annotations

from bisect import bisect_right
import calendar
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any


CANONICALIZATION_SCHEMA = "github_actions_ci_log_canonical_v1"
DEDUPLICATION_SCHEMA = "github_actions_ci_log_dedup_v1"
SIDECAR_SCHEMA = "cppmega_ci_log_sidecar_v2"
TRAINING_SIDECAR_SCHEMA = "cppmega_ci_chunk_training_sidecars_v2"
BUILD_ACTION_NORMALIZATION_SCHEMA = "ci_build_action_shape_v1"
DEFAULT_MAX_CHUNK_CHARS = 128_000
MAX_AUDIT_SAMPLES = 8
MAX_ENTITY_GROUPS = 24
MAX_EDGE_GROUPS = 32
MAX_CLASSIFICATION_GROUPS = 16
MAX_OCCURRENCE_SAMPLES = 2
MAX_EVIDENCE_TEXT_CHARS = 256

# Local copies intentionally avoid coupling this ingestion module to the model
# or data packages.  The values match cppmega.data.domain_schema v1.
DOMAIN_IDS = {
    "UNKNOWN": 0,
    "CPP": 1,
    "CMAKE": 2,
    "MAKE": 3,
    "NINJA": 4,
    "BAZEL": 5,
    "AUTOCONF": 6,
    "AUTOMAKE": 7,
    "MESON": 8,
    "GN": 9,
    "SCONS": 10,
    "XMAKE": 11,
    "COMPILE_COMMANDS": 12,
    "CONFIGURE": 13,
    "BASH": 20,
    "ZSH": 21,
    "SH": 22,
    "TCSH": 23,
    "KSH": 24,
    "SQL": 30,
    "PYTHON": 31,
    "COMPILER_DIAGNOSTIC": 40,
    "BUILD_DIAGNOSTIC": 41,
    "COMPILER_ERROR": 42,
    "BUILD_ERROR": 43,
    "LINKER_ERROR": 44,
    "TEST_OUTPUT": 45,
    "TOOL_OUTPUT": 46,
    "LINKER_DIAGNOSTIC": 47,
    "SANITIZER_OUTPUT": 48,
}

ROLE_IDS = {
    "NONE": 0,
    "DELIMITER": 1,
    "KEYWORD": 2,
    "IDENTIFIER": 3,
    "TARGET": 4,
    "VARIABLE": 5,
    "COMMAND": 6,
    "PATH": 7,
    "OPTION": 8,
    "STRING": 9,
    "LABEL": 10,
    "RULE": 11,
    "ATTRIBUTE": 12,
    "SOURCE": 13,
    "LIBRARY": 14,
    "PREREQUISITE": 15,
    "OUTPUT": 16,
    "INPUT": 17,
    "ENVIRONMENT": 18,
    "REDIRECT": 19,
    "PIPE": 20,
    "COMMENT": 21,
    "DOCSTRING": 22,
    "PREPROCESSOR": 23,
    "SEVERITY": 30,
    "MESSAGE": 31,
    "FILE": 32,
    "LINE": 33,
    "COLUMN": 34,
    "SYMBOL": 35,
    "FIXIT": 36,
    "NOTE": 37,
    "EXIT_CODE": 38,
    "TEST_NAME": 39,
}

EDGE_IDS = {
    "BUILD_TARGET_DEP": 20,
    "BUILD_TARGET_SOURCE": 21,
    "BUILD_RULE_COMMAND": 22,
    "BUILD_ACTION_INPUT": 23,
    "BUILD_ACTION_OUTPUT": 24,
    "BUILD_VAR_DEF_USE": 25,
    "BUILD_COMMAND_TARGET": 26,
    "SHELL_PIPE": 40,
    "SHELL_REDIR_IN": 41,
    "SHELL_REDIR_OUT": 42,
    "SHELL_VAR_DEF_USE": 43,
    "SHELL_COMMAND_FILE": 44,
    "DIAG_PRIMARY_LOCATION": 60,
    "DIAG_NOTE": 61,
    "DIAG_FIXIT": 62,
    "DIAG_COMMAND": 63,
    "DIAG_BUILD_TARGET": 64,
    "LINK_UNDEFINED_SYMBOL": 70,
    "LINK_CANDIDATE_DEF": 71,
    "TEST_FAILURE_LOCATION": 80,
    "TOOL_ACTION_RESULT": 90,
    "EMBEDDED_DOMAIN": 100,
}

EDGE_FAMILIES = {
    name: (
        "build"
        if value < 40
        else "shell"
        if value < 60
        else "diagnostic"
        if value < 100
        else "cross_domain"
    )
    for name, value in EDGE_IDS.items()
}

_GITHUB_TIMESTAMP_RE = re.compile(
    r"^(?P<timestamp>"
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z"
    r")(?P<separator>[ \t])"
)
_MIDSTREAM_RECORD_BOM_RE = re.compile(
    "\ufeff(?="
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,9})?Z[ \t])"
)
_PHYSICAL_LINE_BOUNDARY_RE = re.compile(r"\r\n|\r|\n")
_ANSI_RE = re.compile(
    r"(?:"
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|"
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|"
    r"\x1b[@-_]"
    r")"
)

_UUID_PATTERN = (
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
)

_SECRET_RULES: tuple[tuple[str, re.Pattern[str], str | None], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END(?: [A-Z0-9]+)* PRIVATE KEY-----",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "github_fine_grained_pat",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,255}\b"),
        None,
    ),
    (
        "github_token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b"),
        None,
    ),
    (
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        None,
    ),
    (
        "bearer_credential",
        re.compile(
            r"(?i)\bbearer[ \t]+"
            r"(?P<secret>[A-Za-z0-9._~+/=-]{8,})"
        ),
        "secret",
    ),
    (
        "basic_credential",
        re.compile(
            r"(?i)\bbasic[ \t]+"
            r"(?P<secret>(?!\*{3}(?:\s|$))[A-Za-z0-9+/]{8,}={0,2})"
        ),
        "secret",
    ),
    (
        "credentialed_url",
        re.compile(
            r"(?i)\b[a-z][a-z0-9+.-]*://"
            r"(?P<secret>[^/@\s:]+:[^/@\s]+)@"
        ),
        "secret",
    ),
    (
        "named_credential",
        re.compile(
            r"(?ix)\b"
            r"(?:github_token|gh_token|aws_secret_access_key|aws_session_token|"
            r"password|passwd|api_key|access_token|client_secret)"
            r"\s*[:=]\s*['\"]?"
            r"(?P<secret>(?!\*{3}(?:['\"\s]|$))[^'\"\s]{6,})"
        ),
        "secret",
    ),
)

_DEDUP_RULES: tuple[
    tuple[str, re.Pattern[str], str, str], ...
] = (
    (
        "hosted_runner_numeric_instance",
        re.compile(
            r"(?im)(?:hosted runner:\s*GitHub Actions\s+)"
            r"(?P<value>\d{6,})"
        ),
        "value",
        "digits",
    ),
    (
        "worker_id_uuid",
        re.compile(
            rf"(?im)(?:Worker ID:\s*\{{?)(?P<value>{_UUID_PATTERN})(?=\}}?)"
        ),
        "value",
        "uuid",
    ),
    (
        "posix_temp_uuid",
        re.compile(rf"(?i)(?<=/_temp/)(?P<value>{_UUID_PATTERN})"),
        "value",
        "uuid",
    ),
    (
        "windows_temp_uuid",
        re.compile(rf"(?i)(?<=\\_temp\\)(?P<value>{_UUID_PATTERN})"),
        "value",
        "uuid",
    ),
    (
        "ephemeral_pid",
        re.compile(
            r"(?im)\b(?:pid|process id)(?:\s*[:=#]\s*|\s+)"
            r"(?P<value>\d{2,})\b"
        ),
        "value",
        "digits",
    ),
    (
        "github_debug_progress_duration",
        re.compile(
            r"(?im)^##\[debug\][^\n]*?"
            r"(?:elapsed|duration|took)\s*[:=]?\s*"
            r"(?P<value>\d+(?:\.\d+)?(?:ms|s))\b"
        ),
        "value",
        "numeric",
    ),
)

_BUILD_SYSTEM_PATTERNS: tuple[
    tuple[str, str, re.Pattern[str]], ...
] = (
    ("cmake", "CMAKE", re.compile(r"(?i)(?<![\w.-])cmake(?:\.exe)?(?![\w.-])")),
    ("make", "MAKE", re.compile(r"(?i)(?<![\w.-])(?:g?make)(?:\.exe)?(?![\w.-])")),
    ("ninja", "NINJA", re.compile(r"(?i)(?<![\w.-])ninja(?:\.exe)?(?![\w.-])")),
    (
        "bazel",
        "BAZEL",
        re.compile(r"(?i)(?<![\w.-])(?:bazel|bazelisk)(?:\.exe)?(?![\w.-])"),
    ),
    ("meson", "MESON", re.compile(r"(?i)(?<![\w.-])meson(?:\.py)?(?![\w.-])")),
    (
        "autotools",
        "AUTOCONF",
        re.compile(
            r"(?i)(?<![\w.-])"
            r"(?:autoconf|automake|autoreconf|autoheader|configure)"
            r"(?:\.sh)?(?![\w.-])"
        ),
    ),
    ("gn", "GN", re.compile(r"(?i)(?<![\w.-])gn(?:\.exe)?(?![\w.-])")),
    ("scons", "SCONS", re.compile(r"(?i)(?<![\w.-])scons(?:\.py)?(?![\w.-])")),
    ("xmake", "XMAKE", re.compile(r"(?i)(?<![\w.-])xmake(?:\.exe)?(?![\w.-])")),
    (
        "msbuild",
        "BUILD_DIAGNOSTIC",
        re.compile(
            r"(?i)(?<![\w.-])(?:msbuild|devenv)(?:\.exe)?(?![\w.-])"
        ),
    ),
)

_COMPILER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "clang-cl",
        re.compile(r"(?i)(?<![\w.-])clang-cl(?:\.exe)?(?![\w.-])"),
    ),
    (
        "clang++",
        re.compile(r"(?i)(?<![\w.-])clang\+\+(?:\.exe)?(?![\w.-])"),
    ),
    ("clang", re.compile(r"(?i)(?<![\w.-])clang(?:\.exe)?(?![\w.+-])")),
    ("g++", re.compile(r"(?i)(?<![\w.-])g\+\+(?:\.exe)?(?![\w.-])")),
    ("gcc", re.compile(r"(?i)(?<![\w.-])gcc(?:\.exe)?(?![\w.-])")),
    ("nvcc", re.compile(r"(?i)(?<![\w.-])nvcc(?:\.exe)?(?![\w.-])")),
    (
        "msvc",
        re.compile(
            r"(?i)(?<![\w.-])cl(?:\.exe)?(?![\w.-])"
            r"|Microsoft \(R\) C/C\+\+ Optimizing Compiler"
        ),
    ),
    (
        "intel",
        re.compile(r"(?i)(?<![\w.-])(?:icx|icpx|icc|icpc)(?:\.exe)?(?![\w.-])"),
    ),
)

_SHELL_EXECUTABLE_RE = re.compile(
    r"(?i)(?:^|[/\\])"
    r"(?P<shell>pwsh|powershell|cmd(?:\.exe)?|bash|zsh|tcsh|ksh|sh)"
    r"(?:\.exe)?(?:\s|$)"
)

_SQL_DIALECT_COMMAND_PATTERNS: tuple[
    tuple[str, re.Pattern[str]], ...
] = (
    (
        "clickhouse",
        re.compile(r"(?i)(?<![\w.-])clickhouse-client(?:\.exe)?(?![\w.-])"),
    ),
    ("duckdb", re.compile(r"(?i)(?<![\w.-])duckdb(?:\.exe)?(?![\w.-])")),
    (
        "mariadb",
        re.compile(r"(?i)(?<![\w.-])mariadb(?:\.exe)?(?![\w.-])"),
    ),
    ("mysql", re.compile(r"(?i)(?<![\w.-])mysql(?:\.exe)?(?![\w.-])")),
    ("oracle", re.compile(r"(?i)(?<![\w.-])sqlplus(?:\.exe)?(?![\w.-])")),
    (
        "postgresql",
        re.compile(r"(?i)(?<![\w.-])psql(?:\.exe)?(?![\w.-])"),
    ),
    (
        "sqlite",
        re.compile(r"(?i)(?<![\w.-])sqlite3(?:\.exe)?(?![\w.-])"),
    ),
    (
        "sql-server",
        re.compile(r"(?i)(?<![\w.-])sqlcmd(?:\.exe)?(?![\w.-])"),
    ),
)
_SQL_CLIENT_COMMAND_PREFIX_RE = re.compile(
    r"""
    \s*
    (?:
        (?:command|exec|nohup)\s+
        |
        env(?:\s+-[^\s]+)*\s+
        |
        sudo(?:\s+(?:-[ug]\s+[^\s]+|-[^\s]+))*\s+
        |
        [A-Za-z_][A-Za-z0-9_]*=[^\s]+\s+
    )*
    (?:[^\s;&|"'`]*[\\/])?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PATH_RE = re.compile(
    r"(?P<path>"
    r"[A-Za-z]:[\\/][^\s\"'<>|]+"
    r"|"
    r"(?:/|\./|\.\./)[^\s\"'<>|]+"
    r"|"
    r"(?<![A-Za-z0-9_.@+*?-])"
    r"(?:[A-Za-z0-9_.@+-]+[/\\])+[A-Za-z0-9_.@+*?-]+"
    r"|"
    r"(?:CMakeLists\.txt|Makefile|GNUmakefile|meson\.build|"
    r"BUILD(?:\.bazel)?|WORKSPACE(?:\.bazel)?|configure\.ac|"
    r"(?<![A-Za-z0-9_.@+-])"
    r"[A-Za-z0-9_.@+-]+\."
    r"(?:c|cc|cpp|cxx|h|hh|hpp|hxx|cu|cuh|ixx|cppm|mpp|"
    r"cmake|mk|ninja|bzl|bazel|gn|gni|py|pyi|"
    r"js|jsx|mjs|cjs|ts|tsx|mts|cts|"
    r"sh|bash|zsh|"
    r"ksh|ps1|bat|cmd|sql|o|obj|a|so|dylib|dll|exe|lib|pdb)"
    r")"
    r")"
)

_SOURCE_EXTENSIONS = {
    ".c": "C",
    ".h": "C/C++",
    ".cc": "C++",
    ".cpp": "C++",
    ".cxx": "C++",
    ".hh": "C++",
    ".hpp": "C++",
    ".hxx": "C++",
    ".cu": "CUDA",
    ".cuh": "CUDA",
    ".ixx": "module",
    ".cppm": "module",
    ".mpp": "module",
}
_AUX_SOURCE_EXTENSIONS = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".mts": "TypeScript",
    ".cts": "TypeScript",
}
_BUILD_EXTENSIONS = {
    ".cmake",
    ".mk",
    ".ninja",
    ".bzl",
    ".bazel",
    ".gn",
    ".gni",
}
_SHELL_EXTENSIONS = {".sh", ".bash", ".zsh", ".ksh", ".ps1", ".bat", ".cmd"}
_OUTPUT_EXTENSIONS = {
    ".o",
    ".obj",
    ".a",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".lib",
    ".pdb",
}

_GCC_DIAGNOSTIC_RE = re.compile(
    r"^(?P<file>(?:[A-Za-z]:)?[^:\n]+):"
    r"(?P<line>\d+):(?P<column>\d+):\s*"
    r"(?P<severity>fatal error|error|warning|note):\s*"
    r"(?P<message>.+)$",
    re.IGNORECASE,
)
_MSVC_DIAGNOSTIC_RE = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+)"
    r"(?:,(?P<column>\d+))?\):\s*"
    r"(?P<severity>fatal error|error|warning|note)\s+"
    r"(?P<code>[A-Z]+\d+):\s*(?P<message>.+)$",
    re.IGNORECASE,
)
_CMAKE_DIAGNOSTIC_RE = re.compile(
    r"^CMake\s+(?P<severity>Error|Warning)(?:\s+at\s+"
    r"(?P<file>[^:\n]+):(?P<line>\d+))?(?P<message>.*)$",
    re.IGNORECASE,
)

_LINK_ARCHIVE_TOOL_RE = re.compile(
    r"(?i)(?<![\w.+-])"
    r"(?P<tool>(?:ld(?:\.lld)?|lld-link|link|ar|llvm-ar|lib)(?:\.exe)?)"
    r"(?=\s|$)"
)

_PACKAGE_BUILD_ID_RE = re.compile(r"^[A-Za-z0-9.+-]+_\d+$")
_PACKAGE_SIZE_RE = re.compile(r"(?i)^\d+(?:\.\d+)?(?:[KMGT]i?B)$")
_PACKAGE_CHANNELS = frozenset(
    {
        "anaconda",
        "conda-forge",
        "defaults",
        "pkgs/main",
        "pkgs/r",
        "pypi",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _framed_digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sequence_digest(values: Sequence[Any]) -> str:
    return _framed_digest([stable_json_dumps(value) for value in values])


def _bounded_samples(
    values: Sequence[Any],
    *,
    limit: int = MAX_AUDIT_SAMPLES,
) -> list[dict[str, Any]]:
    """Return deterministic head/tail evidence with original sequence indexes."""

    if len(values) <= limit:
        indexes = list(range(len(values)))
    else:
        head = (limit + 1) // 2
        tail = limit - head
        indexes = list(range(head)) + list(range(len(values) - tail, len(values)))
    return [
        {"sequence_index": index, "value": _json_safe(values[index])}
        for index in indexes
    ]


def _bounded_sequence_receipt(
    values: Sequence[Any],
    *,
    limit: int = MAX_AUDIT_SAMPLES,
) -> dict[str, Any]:
    return {
        "count": len(values),
        "sample_limit": limit,
        "samples": _bounded_samples(values, limit=limit),
        "omitted_count": max(0, len(values) - limit),
        "ordered_sha256": _sequence_digest(values),
    }


def stable_json_dumps(value: Any) -> str:
    """Serialize a returned artifact deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe(item) for item in value]
    return str(value)


def _decode_raw_log(raw_log: bytes | bytearray | memoryview | str) -> tuple[str, dict]:
    if isinstance(raw_log, str):
        try:
            raw_bytes = raw_log.encode("utf-8")
            return raw_log, {
                "input_type": "text",
                "encoding": "utf-8",
                "status": "provided_text",
                "invalid_sequence_count": 0,
                "invalid_byte_spans": [],
                "replacement_char_count": 0,
                "raw_byte_count": len(raw_bytes),
                "raw_sha256": _sha256_bytes(raw_bytes),
            }
        except UnicodeEncodeError:
            raw_bytes = raw_log.encode("utf-8", errors="surrogatepass")
            replaced = raw_log.encode(
                "utf-8", errors="replace"
            ).decode("utf-8")
            surrogate_count = sum(
                0xD800 <= ord(character) <= 0xDFFF for character in raw_log
            )
            return replaced, {
                "input_type": "text",
                "encoding": "utf-8",
                "status": "text_surrogates_replaced",
                "invalid_sequence_count": surrogate_count,
                "invalid_byte_spans": [],
                "replacement_char_count": surrogate_count,
                "raw_byte_count": len(raw_bytes),
                "raw_sha256": _sha256_bytes(raw_bytes),
            }

    if not isinstance(raw_log, (bytes, bytearray, memoryview)):
        raise TypeError("raw_log must be bytes-like or str")
    raw_bytes = bytes(raw_log)
    try:
        decoded = raw_bytes.decode("utf-8")
        return decoded, {
            "input_type": "bytes",
            "encoding": "utf-8",
            "status": "valid",
            "invalid_sequence_count": 0,
            "invalid_byte_spans": [],
            "replacement_char_count": 0,
            "raw_byte_count": len(raw_bytes),
            "raw_sha256": _sha256_bytes(raw_bytes),
        }
    except UnicodeDecodeError:
        pieces: list[str] = []
        invalid_spans: list[dict[str, int]] = []
        offset = 0
        while offset < len(raw_bytes):
            try:
                pieces.append(raw_bytes[offset:].decode("utf-8"))
                offset = len(raw_bytes)
            except UnicodeDecodeError as exc:
                start = offset + exc.start
                end = offset + exc.end
                if start > offset:
                    pieces.append(raw_bytes[offset:start].decode("utf-8"))
                pieces.append("\ufffd")
                invalid_spans.append({"byte_start": start, "byte_end": end})
                offset = end
        return "".join(pieces), {
            "input_type": "bytes",
            "encoding": "utf-8",
            "status": "invalid_replaced",
            "invalid_sequence_count": len(invalid_spans),
            "invalid_byte_spans": invalid_spans,
            "replacement_char_count": len(invalid_spans),
            "raw_byte_count": len(raw_bytes),
            "raw_sha256": _sha256_bytes(raw_bytes),
        }


def _normalize_record_boundaries(text: str) -> tuple[str, list[dict], int]:
    """Split only the known GitHub archive BOM+timestamp concatenation."""

    entries: list[dict[str, Any]] = []
    removed_chars = 0
    start = 0
    output: list[str] = []

    if text.startswith("\ufeff"):
        entries.append(
            {
                "decoded_char_offset": 0,
                "kind": "leading_utf8_bom",
                "action": "removed",
            }
        )
        removed_chars += 1
        start = 1

    cursor = start
    for match in _MIDSTREAM_RECORD_BOM_RE.finditer(text, start):
        output.append(text[cursor : match.start()])
        previous = text[match.start() - 1] if match.start() else ""
        if previous in "\r\n":
            replacement = ""
            removed_chars += 1
            action = "removed_before_existing_line_boundary"
        else:
            replacement = "\n"
            action = "replaced_with_missing_line_boundary"
        output.append(replacement)
        entries.append(
            {
                "decoded_char_offset": match.start(),
                "kind": "midstream_bom_before_github_timestamp",
                "action": action,
            }
        )
        cursor = match.end()
    output.append(text[cursor:])
    normalized = "".join(output)

    recognized = {entry["decoded_char_offset"] for entry in entries}
    for index, character in enumerate(text):
        if character == "\ufeff" and index not in recognized:
            entries.append(
                {
                    "decoded_char_offset": index,
                    "kind": "unrecognized_bom",
                    "action": "preserved",
                }
            )
    entries.sort(key=lambda item: int(item["decoded_char_offset"]))
    return normalized, entries, removed_chars


def _physical_lines(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    cursor = 0
    for boundary in _PHYSICAL_LINE_BOUNDARY_RE.finditer(text):
        start, end = boundary.span()
        lines.append(
            {
                "source_start": cursor,
                "source_end": end,
                "content": text[cursor:start],
                "terminator": boundary.group(0),
            }
        )
        cursor = end
    if cursor < len(text):
        lines.append(
            {
                "source_start": cursor,
                "source_end": len(text),
                "content": text[cursor:],
                "terminator": "",
            }
        )
    return lines


def _timestamp_epoch_ns(match: re.Match[str]) -> int | None:
    try:
        timestamp = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None
    fraction = (match.group("fraction") or "").ljust(9, "0")
    return calendar.timegm(timestamp.utctimetuple()) * 1_000_000_000 + int(
        fraction or "0"
    )


def _ansi_kind(sequence: str) -> str:
    if sequence.startswith("\x1b]"):
        return "osc"
    if sequence.startswith("\x1b["):
        return "csi"
    return "escape"


def _strip_ansi(
    payload: str,
    *,
    line_index: int,
    canonical_line_start: int,
) -> tuple[str, list[dict], list[str]]:
    output: list[str] = []
    entries: list[dict[str, Any]] = []
    sequences: list[str] = []
    source_cursor = 0
    canonical_cursor = 0
    for match in _ANSI_RE.finditer(payload):
        plain = payload[source_cursor : match.start()]
        output.append(plain)
        canonical_cursor += len(plain)
        sequence = match.group(0)
        sequences.append(sequence)
        entries.append(
            {
                "line_index": line_index,
                "payload_char_start": match.start(),
                "payload_char_end": match.end(),
                "canonical_char_offset": canonical_line_start + canonical_cursor,
                "char_count": len(sequence),
                "kind": _ansi_kind(sequence),
            }
        )
        source_cursor = match.end()
    output.append(payload[source_cursor:])
    return "".join(output), entries, sequences


def _secret_candidates(text: str) -> list[tuple[int, int, int, str]]:
    candidates: list[tuple[int, int, int, str]] = []
    for priority, (kind, pattern, group_name) in enumerate(_SECRET_RULES):
        for match in pattern.finditer(text):
            start, end = (
                match.span(group_name) if group_name is not None else match.span()
            )
            if start == end or text[start:end] == "***":
                continue
            candidates.append((start, end, priority, kind))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    accepted: list[tuple[int, int, int, str]] = []
    occupied_end = -1
    for candidate in candidates:
        if candidate[0] < occupied_end:
            continue
        accepted.append(candidate)
        occupied_end = candidate[1]
    return accepted


def _structure_contains_secret(value: Any) -> bool:
    """Scan JSON-like string leaves without matching across field boundaries."""
    if isinstance(value, str):
        return bool(_secret_candidates(value))
    if isinstance(value, Mapping):
        return any(
            _structure_contains_secret(key)
            or _structure_contains_secret(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        return any(_structure_contains_secret(item) for item in value)
    return False


def _redact_secrets(text: str) -> tuple[str, list[dict[str, Any]]]:
    candidates = _secret_candidates(text)
    if not candidates:
        return text, []
    output: list[str] = []
    ledger: list[dict[str, Any]] = []
    cursor = 0
    line_starts = _line_starts(text)
    for start, end, _priority, kind in candidates:
        output.append(text[cursor:start])
        secret = text[start:end]
        replacement = "".join(
            character if character in "\r\n" else "*" for character in secret
        )
        output.append(replacement)
        ledger.append(
            {
                "kind": kind,
                "canonical_char_start": start,
                "canonical_char_end": end,
                "line_start": _line_for_offset(line_starts, start),
                "line_end": _line_for_offset(line_starts, max(start, end - 1)),
                "replacement": "*",
                "replacement_policy": "same_length_star_mask_preserving_newlines",
                "replacement_char_count": sum(
                    character not in "\r\n" for character in secret
                ),
            }
        )
        cursor = end
    output.append(text[cursor:])
    result = "".join(output)
    if len(result) != len(text):
        raise AssertionError("secret masking must preserve character offsets")
    return result, ledger


def _sanitize_metadata(
    value: Any,
    *,
    path: str,
    ledger: list[dict[str, Any]],
) -> Any:
    value = _json_safe(value)
    if isinstance(value, str):
        redacted, entries = _redact_secrets(value)
        for entry in entries:
            ledger.append(
                {
                    "path": path,
                    "kind": entry["kind"],
                    "replacement": "*",
                    "replacement_policy": entry["replacement_policy"],
                    "replacement_char_count": entry["replacement_char_count"],
                }
            )
        return redacted
    if isinstance(value, list):
        return [
            _sanitize_metadata(item, path=f"{path}[{index}]", ledger=ledger)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _sanitize_metadata(
                item, path=f"{path}.{key}" if path else key, ledger=ledger
            )
            for key, item in value.items()
        }
    return value


def _mask_dedup_value(value: str, mask_kind: str) -> str:
    if mask_kind == "digits":
        return "".join("0" if character.isdigit() else character for character in value)
    if mask_kind == "uuid":
        return "".join(
            "0" if character.lower() in "0123456789abcdef" else character
            for character in value
        )
    if mask_kind == "numeric":
        return "".join("0" if character.isdigit() else character for character in value)
    raise ValueError(f"unknown dedup mask {mask_kind}")


def _deduplicate_volatiles(text: str) -> tuple[str, list[dict[str, Any]]]:
    candidates: list[tuple[int, int, int, str, str]] = []
    for priority, (rule_id, pattern, group_name, mask_kind) in enumerate(
        _DEDUP_RULES
    ):
        for match in pattern.finditer(text):
            start, end = match.span(group_name)
            candidates.append((start, end, priority, rule_id, mask_kind))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

    accepted: list[tuple[int, int, int, str, str]] = []
    occupied_end = -1
    for candidate in candidates:
        if candidate[0] < occupied_end:
            continue
        accepted.append(candidate)
        occupied_end = candidate[1]

    output: list[str] = []
    ledger: list[dict[str, Any]] = []
    cursor = 0
    for start, end, _priority, rule_id, mask_kind in accepted:
        value = text[start:end]
        replacement = _mask_dedup_value(value, mask_kind)
        output.append(text[cursor:start])
        output.append(replacement)
        ledger.append(
            {
                "rule_id": rule_id,
                "canonical_char_start": start,
                "canonical_char_end": end,
                "original": value,
                "replacement": replacement,
                "length_preserving": True,
            }
        )
        cursor = end
    output.append(text[cursor:])
    result = "".join(output)
    if len(result) != len(text):
        raise AssertionError("dedup normalization must preserve character offsets")
    return result, ledger


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(match.end() for match in re.finditer("\n", text))
    return starts


def _line_for_offset(starts: Sequence[int], offset: int) -> int:
    return max(0, bisect_right(starts, offset) - 1)


def _canonical_lines(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    cursor = 0
    for index, physical in enumerate(_physical_lines(text)):
        terminator = physical["terminator"]
        content_end = int(physical["source_end"]) - len(terminator)
        lines.append(
            {
                "index": index,
                "start": cursor,
                "content_end": content_end,
                "end": int(physical["source_end"]),
                "content": physical["content"],
                "terminator": terminator,
            }
        )
        cursor = int(physical["source_end"])
    return lines


def _path_get(mapping: Mapping[str, Any], path: str) -> tuple[Any, str | None]:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None, None
        current = current[part]
    if current is None:
        return None, None
    return current, path


def _pick(
    metadata: Mapping[str, Any], paths: Sequence[str]
) -> tuple[Any, str | None]:
    for path in paths:
        value, source = _path_get(metadata, path)
        if source is not None:
            return value, source
    return None, None


def _confidence(
    score: float,
    *,
    source: str | None,
    level: str | None = None,
) -> dict[str, Any]:
    if level is None:
        if score >= 1.0:
            level = "exact"
        elif score >= 0.8:
            level = "high"
        elif score > 0:
            level = "heuristic"
        else:
            level = "absent"
    return {
        "score": float(score),
        "level": level,
        "source": source,
    }


def _extract_provenance(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_redactions: list[dict[str, Any]] = []
    aliases: dict[str, tuple[str, ...]] = {
        "repository": (
            "repository.full_name",
            "repository",
            "repo",
            "owner_repo",
        ),
        "repository_requested": (
            "repository_requested",
            "requested_repository",
        ),
        "repository_id": ("repository.id", "repository_id"),
        "source_repository": (
            "source_repository",
            "head_repository.full_name",
        ),
        "source_repository_id": (
            "source_repository_id",
            "head_repository.id",
        ),
        "workflow": (
            "workflow.name",
            "workflow_name",
            "workflow",
            "name",
        ),
        "workflow_id": ("workflow.id", "workflow_id"),
        "workflow_path": ("workflow.path", "workflow_path", "path"),
        "run_id": ("run.id", "run_id", "id"),
        "run_attempt": ("run.attempt", "run_attempt", "attempt"),
        "run_number": ("run.number", "run_number"),
        "run_status": ("run.status", "run_status", "status"),
        "run_conclusion": (
            "run.conclusion",
            "run_conclusion",
            "conclusion",
        ),
        "run_created_at": ("run.created_at", "run_created_at", "created_at"),
        "run_updated_at": ("run.updated_at", "run_updated_at", "updated_at"),
        "run_started_at": (
            "run.run_started_at",
            "run_started_at",
        ),
        "display_title": ("run.display_title", "display_title"),
        "event": ("event.name", "event_name", "event"),
        "head_sha": ("head_sha", "run.head_sha", "commit_sha", "sha"),
        "head_commit": ("run.head_commit", "head_commit"),
        "branch": ("head_branch", "run.head_branch", "branch", "ref"),
        "actor": ("actor.login", "actor", "run.actor.login"),
        "triggering_actor": (
            "triggering_actor.login",
            "triggering_actor",
            "run.triggering_actor.login",
        ),
        "job_id": ("job.id", "job_id"),
        "job_name": ("job.name", "job_name"),
        "job_status": ("job.status", "job_status", "status"),
        "job_conclusion": (
            "job.conclusion",
            "job_conclusion",
            "conclusion",
        ),
        "job_started_at": ("job.started_at", "job_started_at"),
        "job_completed_at": ("job.completed_at", "job_completed_at"),
        "runner_name": ("job.runner_name", "runner.name", "runner_name"),
        "runner_group_id": (
            "job.runner_group_id",
            "runner.group_id",
            "runner_group_id",
        ),
        "runner_group_name": (
            "job.runner_group_name",
            "runner.group_name",
            "runner_group_name",
        ),
        "runner_os": ("runner.os", "runner_os"),
        "runner_arch": ("runner.arch", "runner_arch"),
        "runner_image": ("runner.image", "runner_image", "image"),
        "container": ("job.container", "container"),
        "matrix": ("job.matrix", "matrix"),
        "runner_labels": (
            "job.labels",
            "runner.labels",
            "runner_labels",
            "labels",
        ),
        "steps": ("job.steps", "steps"),
    }

    values: dict[str, Any] = {}
    field_confidence: dict[str, dict[str, Any]] = {}
    for field, paths in aliases.items():
        value, source = _pick(metadata, paths)
        if source is None:
            values[field] = [] if field in {"runner_labels", "steps"} else None
            field_confidence[field] = _confidence(0.0, source=None)
            continue
        sanitized = _sanitize_metadata(
            value,
            path=f"metadata.{source}",
            ledger=metadata_redactions,
        )
        values[field] = sanitized
        field_confidence[field] = _confidence(
            1.0, source=f"metadata.{source}"
        )

    runner_labels = values["runner_labels"]
    if isinstance(runner_labels, str):
        values["runner_labels"] = [
            item.strip() for item in runner_labels.split(",") if item.strip()
        ]
    elif not isinstance(runner_labels, list):
        values["runner_labels"] = [runner_labels] if runner_labels else []

    raw_steps = values["steps"]
    steps: list[dict[str, Any]] = []
    if isinstance(raw_steps, list):
        for index, raw_step in enumerate(raw_steps):
            if isinstance(raw_step, Mapping):
                steps.append(
                    {
                        "ordinal": index,
                        "number": raw_step.get("number"),
                        "id": raw_step.get("id"),
                        "name": raw_step.get("name"),
                        "status": raw_step.get("status"),
                        "conclusion": raw_step.get("conclusion"),
                        "started_at": raw_step.get("started_at"),
                        "completed_at": raw_step.get("completed_at"),
                    }
                )
            else:
                steps.append(
                    {
                        "ordinal": index,
                        "number": None,
                        "id": None,
                        "name": str(raw_step),
                        "status": None,
                        "conclusion": None,
                        "started_at": None,
                        "completed_at": None,
                    }
                )
    values["steps"] = steps

    provenance = {
        "repository": values["repository"],
        "repository_requested": (
            values["repository_requested"] or values["repository"]
        ),
        "repository_id": values["repository_id"],
        "source_repository": (
            values["source_repository"] or values["repository"]
        ),
        "source_repository_id": (
            values["source_repository_id"] or values["repository_id"]
        ),
        "repository_alias_changed": (
            values["repository_requested"] is not None
            and values["repository_requested"] != values["repository"]
        ),
        "workflow": {
            "name": values["workflow"],
            "id": values["workflow_id"],
            "path": values["workflow_path"],
        },
        "run": {
            "id": values["run_id"],
            "attempt": values["run_attempt"],
            "number": values["run_number"],
            "status": values["run_status"],
            "conclusion": values["run_conclusion"],
            "created_at": values["run_created_at"],
            "updated_at": values["run_updated_at"],
            "started_at": values["run_started_at"],
            "display_title": values["display_title"],
            "event": values["event"],
            "head_sha": values["head_sha"],
            "head_commit": values["head_commit"],
            "branch": values["branch"],
        },
        "actors": {
            "actor": values["actor"],
            "triggering_actor": values["triggering_actor"],
        },
        "job": {
            "id": values["job_id"],
            "name": values["job_name"],
            "status": values["job_status"],
            "conclusion": values["job_conclusion"],
            "started_at": values["job_started_at"],
            "completed_at": values["job_completed_at"],
            "matrix": values["matrix"],
            "container": values["container"],
        },
        "runner": {
            "name": values["runner_name"],
            "group_id": values["runner_group_id"],
            "group_name": values["runner_group_name"],
            "os": values["runner_os"],
            "arch": values["runner_arch"],
            "image": values["runner_image"],
            "labels": values["runner_labels"],
        },
        "steps": steps,
        "field_confidence": field_confidence,
    }
    return provenance, metadata_redactions


def _normalize_step_name(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip()).casefold()
    for prefix in ("run ", "post ", "uses "):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _section_boundary(content: str) -> tuple[str, str] | None:
    match = re.match(
        r"^##\[group\](?P<prefix>Run|Post|Uses)\s+(?P<title>.+?)\s*$",
        content,
        re.IGNORECASE,
    )
    if match:
        prefix = match.group("prefix").lower()
        kind = "post_step" if prefix == "post" else "step"
        return kind, f"{match.group('prefix')} {match.group('title')}"
    if re.match(
        r"^(?:Complete job|Cleanup|Post job cleanup)\s*$",
        content,
        re.IGNORECASE,
    ):
        return "job_epilogue", content.strip()
    return None


def _build_sections(
    canonical_text: str,
    dedup_text: str,
    lines: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    *,
    system_log: bool,
) -> list[dict[str, Any]]:
    boundaries: list[tuple[int, str, str]] = []
    for line in lines:
        boundary = _section_boundary(str(line["content"]))
        if boundary is not None:
            boundaries.append((int(line["index"]), boundary[0], boundary[1]))

    starts: list[tuple[int, str, str]] = []
    if not boundaries or boundaries[0][0] > 0:
        starts.append(
            (
                0,
                "system" if system_log else "job_preamble",
                "System metadata" if system_log else "Job preamble",
            )
        )
    starts.extend(boundaries)
    if not starts:
        starts.append((0, "job_preamble", "Job preamble"))

    metadata_steps = list(provenance.get("steps") or [])
    used_steps: set[int] = set()
    next_step_index = 0
    sections: list[dict[str, Any]] = []
    observed_step_ordinal = 0

    for ordinal, (line_start, kind, title) in enumerate(starts):
        line_end = starts[ordinal + 1][0] if ordinal + 1 < len(starts) else len(lines)
        char_start = (
            int(lines[line_start]["start"])
            if line_start < len(lines)
            else len(canonical_text)
        )
        char_end = (
            int(lines[line_end - 1]["end"])
            if line_end > line_start and line_end <= len(lines)
            else char_start
        )
        step_ordinal: int | None = None
        metadata_step_index: int | None = None
        correlation = _confidence(0.0, source=None)
        metadata_step: dict[str, Any] | None = None
        if kind in {"step", "post_step"}:
            step_ordinal = observed_step_ordinal
            observed_step_ordinal += 1
            normalized_title = _normalize_step_name(title)
            for index, step in enumerate(metadata_steps):
                if index in used_steps:
                    continue
                if _normalize_step_name(step.get("name")) == normalized_title:
                    metadata_step_index = index
                    correlation = _confidence(
                        1.0,
                        source=f"metadata.steps[{index}].name",
                    )
                    break
            if metadata_step_index is None:
                while next_step_index in used_steps:
                    next_step_index += 1
                if next_step_index < len(metadata_steps):
                    metadata_step_index = next_step_index
                    correlation = _confidence(
                        0.55,
                        source=f"metadata.steps[{next_step_index}]",
                        level="sequential",
                    )
            if metadata_step_index is not None:
                used_steps.add(metadata_step_index)
                next_step_index = max(next_step_index, metadata_step_index + 1)
                metadata_step = dict(metadata_steps[metadata_step_index])

        canonical_slice = canonical_text[char_start:char_end]
        dedup_slice = dedup_text[char_start:char_end]
        sections.append(
            {
                "section_id": f"section:{ordinal:06d}",
                "job_ordinal": 0,
                "ordinal": ordinal,
                "kind": kind,
                "title": title,
                "step_ordinal": step_ordinal,
                "metadata_step_index": metadata_step_index,
                "metadata_step": metadata_step,
                "metadata_correlation_confidence": correlation,
                "line_start": line_start,
                "line_end": line_end,
                "char_start": char_start,
                "char_end": char_end,
                "text": canonical_slice,
                "dedup_text": dedup_slice,
                "canonical_sha256": _sha256_text(canonical_slice),
                "dedup_sha256": _sha256_text(dedup_slice),
            }
        )
    return sections


def _section_for_line(
    sections: Sequence[Mapping[str, Any]], line_index: int
) -> Mapping[str, Any]:
    for section in sections:
        if int(section["line_start"]) <= line_index < int(section["line_end"]):
            return section
    return sections[-1]


def _build_chunks(
    canonical_text: str,
    dedup_text: str,
    lines: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    *,
    max_chunk_chars: int,
) -> list[dict[str, Any]]:
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be positive")
    chunks: list[dict[str, Any]] = []
    for section in sections:
        section_line_start = int(section["line_start"])
        section_line_end = int(section["line_end"])
        if section_line_start == section_line_end:
            continue
        cursor = section_line_start
        while cursor < section_line_end:
            chunk_line_start = cursor
            chunk_char_start = int(lines[cursor]["start"])
            chunk_char_end = int(lines[cursor]["end"])
            cursor += 1
            while cursor < section_line_end:
                proposed_end = int(lines[cursor]["end"])
                if proposed_end - chunk_char_start > max_chunk_chars:
                    break
                chunk_char_end = proposed_end
                cursor += 1
            canonical_slice = canonical_text[chunk_char_start:chunk_char_end]
            dedup_slice = dedup_text[chunk_char_start:chunk_char_end]
            ordinal = len(chunks)
            chunks.append(
                {
                    "chunk_id": f"chunk:{ordinal:06d}",
                    "job_ordinal": 0,
                    "section_ordinal": int(section["ordinal"]),
                    "section_id": section["section_id"],
                    "step_ordinal": section["step_ordinal"],
                    "ordinal": ordinal,
                    "line_start": chunk_line_start,
                    "line_end": cursor,
                    "char_start": chunk_char_start,
                    "char_end": chunk_char_end,
                    "dedup_char_start": chunk_char_start,
                    "dedup_char_end": chunk_char_end,
                    "text": dedup_slice,
                    "canonical_text": canonical_slice,
                    "sha256": _sha256_text(dedup_slice),
                    "canonical_sha256": _sha256_text(canonical_slice),
                    "oversized_single_line": (
                        len(canonical_slice) > max_chunk_chars
                        and cursor == chunk_line_start + 1
                    ),
                }
            )
    return chunks


_ROLE_SPAN_PRIORITY = {
    "SEVERITY": 100,
    "SYMBOL": 98,
    "FILE": 96,
    "LINE": 95,
    "COLUMN": 95,
    "TEST_NAME": 94,
    "TARGET": 92,
    "OUTPUT": 90,
    "SOURCE": 89,
    "INPUT": 88,
    "LIBRARY": 87,
    "PATH": 85,
    "OPTION": 75,
    "COMMAND": 60,
    "MESSAGE": 50,
    "ATTRIBUTE": 40,
    "KEYWORD": 35,
    "NONE": 0,
}


def _select_span_entity(
    active: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
) -> Mapping[str, Any] | None:
    if not active:
        return None
    if field == "role":
        return max(
            active.values(),
            key=lambda entity: (
                _ROLE_SPAN_PRIORITY.get(str(entity["role"]), 20),
                float(entity["confidence"]["score"]),
                -(int(entity["end_char"]) - int(entity["start_char"])),
                str(entity["entity_id"]),
            ),
        )
    return max(
        active.values(),
        key=lambda entity: (
            int(entity["domain_id"]) != DOMAIN_IDS["UNKNOWN"],
            _ROLE_SPAN_PRIORITY.get(str(entity["role"]), 20),
            float(entity["confidence"]["score"]),
            -(int(entity["end_char"]) - int(entity["start_char"])),
            str(entity["entity_id"]),
        ),
    )


def _chunk_semantic_rle(
    chunk: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    chunk_start = int(chunk["char_start"])
    chunk_end = int(chunk["char_end"])
    length = chunk_end - chunk_start
    events: defaultdict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {"start": [], "end": []}
    )
    for entity in entities:
        start = max(chunk_start, int(entity["start_char"]))
        end = min(chunk_end, int(entity["end_char"]))
        if start >= end:
            continue
        events[start - chunk_start]["start"].append(entity)
        events[end - chunk_start]["end"].append(entity)

    active: dict[str, Mapping[str, Any]] = {}
    spans: list[dict[str, Any]] = []
    previous = 0
    id_field = "role_id" if field == "role" else "domain_id"
    default_id = ROLE_IDS["NONE"] if field == "role" else DOMAIN_IDS["UNKNOWN"]

    def emit(start: int, end: int) -> None:
        if start >= end:
            return
        selected = _select_span_entity(active, field=field)
        value_id = default_id if selected is None else int(selected[id_field])
        confidence = (
            0.0
            if selected is None
            else float(selected["confidence"]["score"])
        )
        if (
            spans
            and spans[-1][id_field] == value_id
            and spans[-1]["confidence"] == confidence
            and spans[-1]["end_char"] == start
        ):
            spans[-1]["end_char"] = end
        else:
            spans.append(
                {
                    "start_char": start,
                    "end_char": end,
                    id_field: value_id,
                    "confidence": confidence,
                }
            )

    for position in sorted(events):
        emit(previous, position)
        for entity in events[position]["end"]:
            active.pop(str(entity["entity_id"]), None)
        for entity in events[position]["start"]:
            active[str(entity["entity_id"])] = entity
        previous = position
    emit(previous, length)
    if length and (
        not spans
        or spans[0]["start_char"] != 0
        or spans[-1]["end_char"] != length
        or any(
            left["end_char"] != right["start_char"]
            for left, right in zip(spans, spans[1:])
        )
    ):
        raise AssertionError("chunk semantic RLE does not cover content exactly")
    return spans


def _attach_chunk_semantic_rle(
    chunks: Sequence[dict[str, Any]],
    entities: Sequence[Mapping[str, Any]],
) -> None:
    entities_by_chunk = _records_by_overlapping_chunk(chunks, entities)
    for chunk, chunk_entities in zip(
        chunks, entities_by_chunk, strict=True
    ):
        chunk["semantic_span_offset_basis"] = "chunk_local_canonical_chars"
        chunk["role_spans"] = _chunk_semantic_rle(
            chunk, chunk_entities, field="role"
        )
        chunk["domain_spans"] = _chunk_semantic_rle(
            chunk, chunk_entities, field="domain"
        )


class _EntityBuilder:
    def __init__(self, text: str) -> None:
        self.text = text
        self._entities: list[dict[str, Any]] = []
        self._entity_keys: dict[tuple[Any, ...], int] = {}
        self._edges: list[dict[str, Any]] = []
        self._edge_keys: set[tuple[int, int, str]] = set()

    def add(
        self,
        *,
        kind: str,
        role: str,
        domain: str,
        start: int,
        end: int,
        confidence: float,
        method: str,
        line_index: int,
        section_ordinal: int,
        step_ordinal: int | None,
        attributes: Mapping[str, Any] | None = None,
    ) -> int | None:
        if start < 0 or end <= start or end > len(self.text):
            return None
        key = (kind, role, domain, start, end)
        existing = self._entity_keys.get(key)
        if existing is not None:
            entity = self._entities[existing]
            if confidence > float(entity["confidence"]["score"]):
                entity["confidence"] = _confidence(confidence, source=method)
            if attributes:
                entity["attributes"].update(_json_safe(attributes))
            return existing
        entity_index = len(self._entities)
        self._entity_keys[key] = entity_index
        self._entities.append(
            {
                "_temp_id": entity_index,
                "kind": kind,
                "role": role,
                "role_id": ROLE_IDS.get(role, ROLE_IDS["NONE"]),
                "domain": domain,
                "domain_id": DOMAIN_IDS.get(domain, DOMAIN_IDS["UNKNOWN"]),
                "start_char": start,
                "end_char": end,
                "text": self.text[start:end],
                "line_index": line_index,
                "section_ordinal": section_ordinal,
                "step_ordinal": step_ordinal,
                "confidence": _confidence(confidence, source=method),
                "attributes": _json_safe(attributes or {}),
            }
        )
        return entity_index

    def edge(
        self,
        source: int | None,
        target: int | None,
        kind: str,
        *,
        confidence: float,
        method: str,
    ) -> None:
        if source is None or target is None or source == target:
            return
        if kind not in EDGE_IDS:
            raise ValueError(f"unknown edge kind {kind}")
        key = (source, target, kind)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self._edges.append(
            {
                "_source": source,
                "_target": target,
                "kind": kind,
                "kind_id": EDGE_IDS[kind],
                "family": EDGE_FAMILIES[kind],
                "confidence": _confidence(confidence, source=method),
            }
        )

    def finish(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
        ordered = sorted(
            self._entities,
            key=lambda item: (
                int(item["start_char"]),
                int(item["end_char"]),
                str(item["kind"]),
                int(item["_temp_id"]),
            ),
        )
        id_for_temp: dict[int, str] = {}
        entities: list[dict[str, Any]] = []
        for ordinal, entity in enumerate(ordered):
            entity_id = f"entity:{ordinal:06d}"
            id_for_temp[int(entity["_temp_id"])] = entity_id
            output = dict(entity)
            output.pop("_temp_id")
            output["entity_id"] = entity_id
            entities.append(output)

        by_temp = {int(item["_temp_id"]): item for item in self._entities}
        edges = []
        for edge in sorted(
            self._edges,
            key=lambda item: (
                int(by_temp[int(item["_source"])]["start_char"]),
                int(by_temp[int(item["_target"])]["start_char"]),
                int(item["kind_id"]),
            ),
        ):
            source_entity = by_temp[int(edge["_source"])]
            target_entity = by_temp[int(edge["_target"])]
            output = dict(edge)
            output.pop("_source")
            output.pop("_target")
            output.update(
                edge_id=f"edge:{len(edges):06d}",
                source=id_for_temp[int(edge["_source"])],
                target=id_for_temp[int(edge["_target"])],
                from_char=int(source_entity["start_char"]),
                to_char=int(target_entity["start_char"]),
            )
            edges.append(output)
        return entities, edges, id_for_temp


def _localized_training_record(
    record: Mapping[str, Any],
    *,
    chunk_start: int,
    chunk_end: int,
    omitted_text_fields: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    raw_start = record.get("start_char")
    raw_end = record.get("end_char")
    if (
        isinstance(raw_start, bool)
        or not isinstance(raw_start, int)
        or isinstance(raw_end, bool)
        or not isinstance(raw_end, int)
        or raw_end <= chunk_start
        or raw_start >= chunk_end
    ):
        return None
    output = {
        key: _json_safe(value)
        for key, value in record.items()
        if key not in omitted_text_fields
    }
    clipped_start = max(raw_start, chunk_start)
    clipped_end = min(raw_end, chunk_end)
    output["start_char"] = clipped_start - chunk_start
    output["end_char"] = clipped_end - chunk_start
    output["source_span_clipped"] = (
        clipped_start != raw_start or clipped_end != raw_end
    )
    return output


def _records_by_overlapping_chunk(
    chunks: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    """Bucket interval records without rescanning every record per chunk."""

    buckets: list[list[Mapping[str, Any]]] = [[] for _chunk in chunks]
    if not chunks:
        return buckets
    chunk_starts = [int(chunk["char_start"]) for chunk in chunks]
    chunk_ends = [int(chunk["char_end"]) for chunk in chunks]
    for record in records:
        raw_start = record.get("start_char")
        raw_end = record.get("end_char")
        if (
            isinstance(raw_start, bool)
            or not isinstance(raw_start, int)
            or isinstance(raw_end, bool)
            or not isinstance(raw_end, int)
            or raw_end <= raw_start
        ):
            continue
        first = bisect_right(chunk_ends, raw_start)
        last = bisect_right(chunk_starts, raw_end - 1)
        for index in range(first, last):
            if (
                raw_start < chunk_ends[index]
                and raw_end > chunk_starts[index]
            ):
                buckets[index].append(record)
    return buckets


def _edges_by_endpoint_chunk(
    chunks: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    """Bucket each edge only with chunks that contain one of its endpoints."""

    buckets: list[list[Mapping[str, Any]]] = [[] for _chunk in chunks]
    if not chunks:
        return buckets
    chunk_starts = [int(chunk["char_start"]) for chunk in chunks]
    chunk_ends = [int(chunk["char_end"]) for chunk in chunks]
    for edge in edges:
        indexes: list[int] = []
        for field in ("from_char", "to_char"):
            offset = int(edge[field])
            index = bisect_right(chunk_starts, offset) - 1
            if (
                0 <= index < len(chunks)
                and offset < chunk_ends[index]
                and index not in indexes
            ):
                indexes.append(index)
        for index in indexes:
            buckets[index].append(edge)
    return buckets


def _attach_chunk_training_sidecars(
    chunks: Sequence[dict[str, Any]],
    entities: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    commands: Sequence[Mapping[str, Any]],
    build_actions: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach exhaustive, token-projectable records to each payload chunk."""

    total_entities = 0
    total_edges = 0
    total_cross_chunk_edge_references = 0
    total_cross_chunk_edges = 0
    total_actions = 0
    total_tests = 0
    total_diagnostics = 0
    training_receipts: list[dict[str, Any]] = []

    entities_by_chunk = _records_by_overlapping_chunk(chunks, entities)
    edges_by_chunk = _edges_by_endpoint_chunk(chunks, edges)
    commands_by_chunk = _records_by_overlapping_chunk(chunks, commands)
    actions_by_chunk = _records_by_overlapping_chunk(
        chunks, build_actions
    )
    tests_by_chunk = _records_by_overlapping_chunk(chunks, tests)
    diagnostics_by_chunk = _records_by_overlapping_chunk(
        chunks, diagnostics
    )

    for chunk_index, chunk in enumerate(chunks):
        chunk_start = int(chunk["char_start"])
        chunk_end = int(chunk["char_end"])
        chunk_length = chunk_end - chunk_start
        localized_entities: list[dict[str, Any]] = []
        local_entity_ids: set[str] = set()
        for entity in entities_by_chunk[chunk_index]:
            localized = _localized_training_record(
                entity,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                omitted_text_fields=frozenset({"text"}),
            )
            if localized is None:
                continue
            entity_id = str(entity["entity_id"])
            local_entity_ids.add(entity_id)
            localized_entities.append(localized)

        localized_edges: list[dict[str, Any]] = []
        crossing_edges: list[dict[str, Any]] = []
        outbound_cross_chunk_edges: list[dict[str, Any]] = []
        for edge in edges_by_chunk[chunk_index]:
            from_char = int(edge["from_char"])
            to_char = int(edge["to_char"])
            from_local = chunk_start <= from_char < chunk_end
            to_local = chunk_start <= to_char < chunk_end
            if from_local and to_local:
                source = str(edge["source"])
                target = str(edge["target"])
                if source not in local_entity_ids or target not in local_entity_ids:
                    raise AssertionError(
                        "in-chunk training edge is missing an entity endpoint"
                    )
                localized_edges.append(
                    {
                        **{
                            key: _json_safe(value)
                            for key, value in edge.items()
                            if key not in {"from_char", "to_char"}
                        },
                        "from_char": from_char - chunk_start,
                        "to_char": to_char - chunk_start,
                    }
                )
            elif from_local or to_local:
                crossing_edges.append(
                    {
                        "edge_id": edge["edge_id"],
                        "kind_id": int(edge["kind_id"]),
                        "from_char": from_char,
                        "to_char": to_char,
                    }
                )
                if from_local:
                    outbound_cross_chunk_edges.append(
                        {
                            **{
                                key: _json_safe(value)
                                for key, value in edge.items()
                                if key not in {"from_char", "to_char"}
                            },
                            "from_char": from_char - chunk_start,
                            "to_member_char": to_char,
                            "target_coordinate_space": (
                                "canonical_member_chars_v1"
                            ),
                        }
                    )

        localized_commands = [
            localized
            for record in commands_by_chunk[chunk_index]
            if (
                localized := _localized_training_record(
                    record,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    omitted_text_fields=frozenset({"text"}),
                )
            )
            is not None
        ]
        localized_actions = [
            localized
            for record in actions_by_chunk[chunk_index]
            if (
                localized := _localized_training_record(
                    record,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    omitted_text_fields=frozenset({"command"}),
                )
            )
            is not None
        ]
        localized_tests = [
            localized
            for record in tests_by_chunk[chunk_index]
            if (
                localized := _localized_training_record(
                    record,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
            )
            is not None
        ]
        localized_diagnostics = [
            localized
            for record in diagnostics_by_chunk[chunk_index]
            if (
                localized := _localized_training_record(
                    record,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    omitted_text_fields=frozenset({"message"}),
                )
            )
            is not None
        ]

        training = {
            "schema": TRAINING_SIDECAR_SCHEMA,
            "coordinate_space": "chunk_local_dedup_chars_v1",
            "dedup_offsets_equal_canonical_offsets": True,
            "chunk_char_count": chunk_length,
            "entities": localized_entities,
            "edges": localized_edges,
            "commands": localized_commands,
            "build_actions": localized_actions,
            "tests": localized_tests,
            "diagnostics": localized_diagnostics,
            "cross_chunk_edges": outbound_cross_chunk_edges,
            "cross_chunk_edge_accounting": {
                "count": len(crossing_edges),
                "outbound_count": len(outbound_cross_chunk_edges),
                "sha256": _sequence_digest(crossing_edges),
            },
        }
        if any(
            not 0 <= int(record["start_char"]) < int(record["end_char"]) <= chunk_length
            for records in (
                localized_entities,
                localized_commands,
                localized_actions,
                localized_tests,
                localized_diagnostics,
            )
            for record in records
        ):
            raise AssertionError("training sidecar span is outside its chunk")
        chunk["training_sidecars"] = training
        total_entities += len(localized_entities)
        total_edges += len(localized_edges)
        total_cross_chunk_edge_references += len(crossing_edges)
        total_cross_chunk_edges += len(outbound_cross_chunk_edges)
        total_actions += len(localized_actions)
        total_tests += len(localized_tests)
        total_diagnostics += len(localized_diagnostics)
        training_receipts.append(
            {
                "chunk_id": chunk["chunk_id"],
                "sha256": _sha256_text(stable_json_dumps(training)),
            }
        )

    return {
        "schema": TRAINING_SIDECAR_SCHEMA,
        "chunk_count": len(chunks),
        "entity_span_count": total_entities,
        "in_chunk_edge_count": total_edges,
        "cross_chunk_edge_reference_count": total_cross_chunk_edge_references,
        "cross_chunk_edge_count": total_cross_chunk_edges,
        "build_action_count": total_actions,
        "test_record_count": total_tests,
        "diagnostic_record_count": total_diagnostics,
        "chunk_sidecar_set_sha256": _sequence_digest(training_receipts),
    }


def _line_context(
    line: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
) -> tuple[int, int | None]:
    section = _section_for_line(sections, int(line["index"]))
    return int(section["ordinal"]), section["step_ordinal"]


def _shell_name(value: str) -> str | None:
    lowered = value.lower()
    if lowered in {"pwsh", "powershell"}:
        return "powershell"
    if lowered.startswith("cmd"):
        return "cmd"
    if lowered in {"bash", "zsh", "tcsh", "ksh", "sh"}:
        return lowered
    return None


def _extract_shells_and_commands(
    text: str,
    lines: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    builder: _EntityBuilder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    shell_records: list[dict[str, Any]] = []
    section_shell: dict[int, str] = {}
    for line in lines:
        content = str(line["content"])
        shell_field = re.match(r"^\s*shell:\s*(?P<spec>.+?)\s*$", content)
        if not shell_field:
            continue
        executable = _SHELL_EXECUTABLE_RE.search(shell_field.group("spec") + " ")
        if not executable:
            continue
        shell = _shell_name(executable.group("shell"))
        if shell is None:
            continue
        section_ordinal, step_ordinal = _line_context(line, sections)
        start = (
            int(line["start"])
            + shell_field.start("spec")
            + executable.start("shell")
        )
        end = start + len(executable.group("shell"))
        entity_ref = builder.add(
            kind="shell_dialect",
            role="ATTRIBUTE",
            domain={
                "powershell": "TOOL_OUTPUT",
                "cmd": "TOOL_OUTPUT",
                "bash": "BASH",
                "zsh": "ZSH",
                "tcsh": "TCSH",
                "ksh": "KSH",
                "sh": "SH",
            }[shell],
            start=start,
            end=end,
            confidence=1.0,
            method="github_actions_shell_field",
            line_index=int(line["index"]),
            section_ordinal=section_ordinal,
            step_ordinal=step_ordinal,
        )
        section_shell[section_ordinal] = shell
        shell_records.append(
            {
                "name": shell,
                "entity_ref": entity_ref,
                "line_index": int(line["index"]),
                "section_ordinal": section_ordinal,
                "step_ordinal": step_ordinal,
                "confidence": _confidence(
                    1.0, source="github_actions_shell_field"
                ),
            }
        )

    commands: list[dict[str, Any]] = []
    command_pattern = re.compile(
        r"^(?:"
        r"##\[group\](?:Run|Post|Uses)\s+"
        r"|\[command\]"
        r"|\+\s+"
        r"|\$\s+"
        r")(?P<command>.+?)\s*$",
        re.IGNORECASE,
    )
    for line in lines:
        content = str(line["content"])
        match = command_pattern.match(content)
        if not match:
            continue
        section_ordinal, step_ordinal = _line_context(line, sections)
        start = int(line["start"]) + match.start("command")
        end = int(line["start"]) + match.end("command")
        entity_ref = builder.add(
            kind="command",
            role="COMMAND",
            domain="TOOL_OUTPUT",
            start=start,
            end=end,
            confidence=1.0,
            method="github_actions_command_marker",
            line_index=int(line["index"]),
            section_ordinal=section_ordinal,
            step_ordinal=step_ordinal,
            attributes={"shell_dialect": section_shell.get(section_ordinal)},
        )
        commands.append(
            {
                "text": text[start:end],
                "start_char": start,
                "end_char": end,
                "line_index": int(line["index"]),
                "section_ordinal": section_ordinal,
                "step_ordinal": step_ordinal,
                "shell_dialect": section_shell.get(section_ordinal),
                "entity_ref": entity_ref,
                "confidence": _confidence(
                    1.0, source="github_actions_command_marker"
                ),
            }
        )
    return shell_records, commands, section_shell


def _extract_sql_dialects(
    commands: Sequence[Mapping[str, Any]],
    builder: _EntityBuilder,
) -> list[dict[str, Any]]:
    """Classify SQL only when a retained command executes a known client."""

    occurrences: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for command in commands:
        text = str(command["text"])
        command_start = int(command["start_char"])
        for dialect, pattern in _SQL_DIALECT_COMMAND_PATTERNS:
            for match in pattern.finditer(text):
                if _SQL_CLIENT_COMMAND_PREFIX_RE.fullmatch(
                    text[: match.start()]
                ) is None:
                    continue
                start = command_start + match.start()
                end = command_start + match.end()
                entity_ref = builder.add(
                    kind="sql_dialect",
                    role="KEYWORD",
                    domain="SQL",
                    start=start,
                    end=end,
                    confidence=0.98,
                    method=f"sql_client_command_v1:{dialect}",
                    line_index=int(command["line_index"]),
                    section_ordinal=int(command["section_ordinal"]),
                    step_ordinal=command.get("step_ordinal"),
                )
                builder.edge(
                    command["entity_ref"],
                    entity_ref,
                    "EMBEDDED_DOMAIN",
                    confidence=0.98,
                    method=f"sql_client_command_v1:{dialect}",
                )
                occurrences[dialect].append(
                    {
                        "start_char": start,
                        "end_char": end,
                        "line_index": int(command["line_index"]),
                        "entity_ref": entity_ref,
                    }
                )
    return [
        {
            "name": dialect,
            "occurrence_count": len(occurrences[dialect]),
            "occurrences": occurrences[dialect],
            "confidence": _confidence(
                0.98, source=f"sql_client_command_v1:{dialect}"
            ),
        }
        for dialect in sorted(occurrences)
    ]


def _extract_build_systems_and_toolchains(
    lines: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    builder: _EntityBuilder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    build_occurrences: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    compiler_occurrences: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for line in lines:
        content = str(line["content"])
        section_ordinal, step_ordinal = _line_context(line, sections)
        for name, domain, pattern in _BUILD_SYSTEM_PATTERNS:
            for match in pattern.finditer(content):
                start = int(line["start"]) + match.start()
                end = int(line["start"]) + match.end()
                entity_ref = builder.add(
                    kind="build_system",
                    role="KEYWORD",
                    domain=domain,
                    start=start,
                    end=end,
                    confidence=0.95,
                    method=f"build_system_token_v1:{name}",
                    line_index=int(line["index"]),
                    section_ordinal=section_ordinal,
                    step_ordinal=step_ordinal,
                )
                build_occurrences[name].append(
                    {
                        "start_char": start,
                        "end_char": end,
                        "line_index": int(line["index"]),
                        "entity_ref": entity_ref,
                    }
                )
        for name, pattern in _COMPILER_PATTERNS:
            for match in pattern.finditer(content):
                start = int(line["start"]) + match.start()
                end = int(line["start"]) + match.end()
                version_match = re.search(
                    r"\b(?:version\s*)?(\d+(?:\.\d+){1,3})\b",
                    content[match.end() : match.end() + 100],
                    re.IGNORECASE,
                )
                version = version_match.group(1) if version_match else None
                entity_ref = builder.add(
                    kind="compiler",
                    role="KEYWORD",
                    domain="CPP" if name != "nvcc" else "CPP",
                    start=start,
                    end=end,
                    confidence=0.95,
                    method=f"compiler_token_v1:{name}",
                    line_index=int(line["index"]),
                    section_ordinal=section_ordinal,
                    step_ordinal=step_ordinal,
                    attributes={"version": version},
                )
                compiler_occurrences[name].append(
                    {
                        "start_char": start,
                        "end_char": end,
                        "line_index": int(line["index"]),
                        "entity_ref": entity_ref,
                        "version": version,
                    }
                )

    build_systems = [
        {
            "name": name,
            "occurrence_count": len(build_occurrences[name]),
            "occurrences": build_occurrences[name],
            "confidence": _confidence(
                0.95, source=f"build_system_token_v1:{name}"
            ),
        }
        for name in sorted(build_occurrences)
    ]
    toolchains = [
        {
            "name": name,
            "versions": sorted(
                {
                    occurrence["version"]
                    for occurrence in compiler_occurrences[name]
                    if occurrence["version"]
                }
            ),
            "occurrence_count": len(compiler_occurrences[name]),
            "occurrences": compiler_occurrences[name],
            "confidence": _confidence(
                0.95, source=f"compiler_token_v1:{name}"
            ),
        }
        for name in sorted(compiler_occurrences)
    ]
    return build_systems, toolchains


def _trim_path(value: str) -> str:
    return value.rstrip(",;:)]}")


def _path_language(path: str) -> str:
    lowered = path.lower()
    basename = re.split(r"[/\\]", lowered)[-1]
    dot = basename.rfind(".")
    extension = basename[dot:] if dot >= 0 else ""
    if extension in _SOURCE_EXTENSIONS:
        return _SOURCE_EXTENSIONS[extension]
    if extension in _AUX_SOURCE_EXTENSIONS:
        return _AUX_SOURCE_EXTENSIONS[extension]
    if (
        extension in _BUILD_EXTENSIONS
        or basename
        in {
            "cmakelists.txt",
            "makefile",
            "gnumakefile",
            "meson.build",
            "build",
            "workspace",
            "configure.ac",
        }
    ):
        return "build"
    if extension in _SHELL_EXTENSIONS:
        return "shell"
    if extension == ".sql":
        return "SQL"
    return "other"


def _path_category(
    path: str,
    line: str,
    path_start: int,
) -> tuple[str, str]:
    lowered = path.lower()
    basename = re.split(r"[/\\]", lowered)[-1]
    dot = basename.rfind(".")
    extension = basename[dot:] if dot >= 0 else ""
    if extension in _OUTPUT_EXTENSIONS:
        return "output", "OUTPUT"
    if extension in _SOURCE_EXTENSIONS:
        return "source", "SOURCE"
    if extension in _AUX_SOURCE_EXTENSIONS:
        return "source", "SOURCE"
    token_end = path_start
    while token_end and line[token_end - 1].isspace():
        token_end -= 1
    for token, category, role in (
        ("--input", "input", "INPUT"),
        ("/out:", "output", "OUTPUT"),
        ("-o", "output", "OUTPUT"),
        ("-i", "input", "INPUT"),
        (">", "output", "OUTPUT"),
        ("<", "input", "INPUT"),
    ):
        token_start = token_end - len(token)
        if (
            token_start >= 0
            and line[token_start:token_end].lower() == token
            and (token_start == 0 or line[token_start - 1].isspace())
        ):
            return category, role
    if (
        extension in _BUILD_EXTENSIONS
        or basename
        in {
            "cmakelists.txt",
            "makefile",
            "gnumakefile",
            "meson.build",
            "build",
            "workspace",
            "configure.ac",
        }
    ):
        return "input", "INPUT"
    return "path", "PATH"


def _extract_paths(
    lines: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    builder: _EntityBuilder,
    commands: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths: list[dict[str, Any]] = []
    languages: defaultdict[str, list[dict[str, int]]] = defaultdict(list)
    commands_by_line = {
        int(command["line_index"]): command for command in commands
    }
    seen: set[tuple[int, int]] = set()

    for line in lines:
        content = str(line["content"])
        section_ordinal, step_ordinal = _line_context(line, sections)
        for match in _PATH_RE.finditer(content):
            raw_path = match.group("path")
            path = _trim_path(raw_path)
            if not path:
                continue
            start = int(line["start"]) + match.start("path")
            if start >= 5 and builder.text[start - 5 : start].lower().endswith(
                "http:"
            ):
                continue
            end = start + len(path)
            if (start, end) in seen:
                continue
            seen.add((start, end))
            category, role = _path_category(
                path,
                content,
                match.start("path"),
            )
            language = _path_language(path)
            domain = {
                "C": "CPP",
                "C/C++": "CPP",
                "C++": "CPP",
                "CUDA": "CPP",
                "module": "CPP",
                "build": "CMAKE",
                "shell": "BASH",
                "Python": "PYTHON",
                "JavaScript": "TOOL_OUTPUT",
                "TypeScript": "TOOL_OUTPUT",
                "SQL": "SQL",
                "other": "TOOL_OUTPUT",
            }[language]
            entity_ref = builder.add(
                kind="path",
                role=role,
                domain=domain,
                start=start,
                end=end,
                confidence=0.9,
                method="path_pattern_v1",
                line_index=int(line["index"]),
                section_ordinal=section_ordinal,
                step_ordinal=step_ordinal,
                attributes={"category": category, "likely_language": language},
            )
            paths.append(
                {
                    "value": path,
                    "category": category,
                    "likely_language": language,
                    "start_char": start,
                    "end_char": end,
                    "line_index": int(line["index"]),
                    "section_ordinal": section_ordinal,
                    "step_ordinal": step_ordinal,
                    "entity_ref": entity_ref,
                    "confidence": _confidence(0.9, source="path_pattern_v1"),
                }
            )
            languages[language].append({"start_char": start, "end_char": end})
            command = commands_by_line.get(int(line["index"]))
            if command is not None:
                relation = (
                    "BUILD_ACTION_OUTPUT"
                    if category == "output"
                    else "BUILD_ACTION_INPUT"
                )
                builder.edge(
                    command["entity_ref"],
                    entity_ref,
                    relation,
                    confidence=0.85,
                    method="same_command_line_path_role",
                )
                if command.get("shell_dialect"):
                    builder.edge(
                        command["entity_ref"],
                        entity_ref,
                        "SHELL_COMMAND_FILE",
                        confidence=0.8,
                        method="same_shell_command_line_path",
                    )

    language_records = [
        {
            "name": name,
            "evidence_count": len(languages[name]),
            "spans": languages[name],
            "confidence": _confidence(0.9, source="path_extension_v1"),
        }
        for name in sorted(languages)
    ]
    return paths, language_records


def _target_matches(command: str) -> list[re.Match[str]]:
    patterns = (
        re.compile(r"(?i)(?:--target(?:=|\s+))(?P<target>[^\s]+)"),
        re.compile(
            r"(?i)\bbazel(?:isk)?(?:\.exe)?\s+"
            r"(?:build|test|run)\s+(?P<target>(?://|:)[^\s]+)"
        ),
        re.compile(
            r"(?i)\bmsbuild(?:\.exe)?.*?"
            r"/t(?:arget)?:(?P<target>[^\s]+)"
        ),
        re.compile(
            r"(?i)\bxmake(?:\.exe)?\s+(?:build\s+)?"
            r"(?P<target>[A-Za-z0-9_.:+/-]+)"
        ),
        re.compile(
            r"(?i)\bninja(?:\.exe)?"
            r"(?:\s+-C\s+\S+|\s+-[^\s]+)*\s+"
            r"(?P<target>(?!-)[A-Za-z0-9_.:+/\\-]+)"
        ),
        re.compile(
            r"(?i)\b(?:g?make)(?:\.exe)?"
            r"(?:\s+-C\s+\S+|\s+-[^\s]+|\s+[A-Za-z_][A-Za-z0-9_]*=\S+)*"
            r"\s+(?P<target>(?!-)[A-Za-z0-9_.:+/\\-]+)"
        ),
        re.compile(
            r"(?i)\bmeson(?:\.py)?\s+compile"
            r"(?:\s+-C\s+\S+|\s+-[^\s]+)*\s+"
            r"(?P<target>(?!-)[A-Za-z0-9_.:+/\\-]+)"
        ),
    )
    matches: list[re.Match[str]] = []
    for pattern in patterns:
        matches.extend(pattern.finditer(command))
    return sorted(matches, key=lambda match: match.start("target"))


def _extract_targets(
    commands: Sequence[Mapping[str, Any]],
    paths: Sequence[Mapping[str, Any]],
    builder: _EntityBuilder,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    paths_by_line: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for path in paths:
        paths_by_line[int(path["line_index"])].append(path)

    for command in commands:
        command_text = str(command["text"])
        for match in _target_matches(command_text):
            target = match.group("target").rstrip(",;")
            start = int(command["start_char"]) + match.start("target")
            end = start + len(target)
            entity_ref = builder.add(
                kind="build_target",
                role="TARGET",
                domain="BUILD_DIAGNOSTIC",
                start=start,
                end=end,
                confidence=0.95,
                method="explicit_build_target_option_v1",
                line_index=int(command["line_index"]),
                section_ordinal=int(command["section_ordinal"]),
                step_ordinal=command["step_ordinal"],
            )
            record = {
                "name": target,
                "start_char": start,
                "end_char": end,
                "line_index": int(command["line_index"]),
                "section_ordinal": int(command["section_ordinal"]),
                "step_ordinal": command["step_ordinal"],
                "entity_ref": entity_ref,
                "confidence": _confidence(
                    0.95, source="explicit_build_target_option_v1"
                ),
            }
            targets.append(record)
            builder.edge(
                command["entity_ref"],
                entity_ref,
                "BUILD_COMMAND_TARGET",
                confidence=0.95,
                method="explicit_build_target_option_v1",
            )
            for path in paths_by_line[int(command["line_index"])]:
                if path["category"] == "source":
                    builder.edge(
                        entity_ref,
                        path["entity_ref"],
                        "BUILD_TARGET_SOURCE",
                        confidence=0.75,
                        method="target_and_source_on_same_command",
                    )
    return targets


def _chunk_for_char(
    chunks: Sequence[Mapping[str, Any]], offset: int
) -> str | None:
    for chunk in chunks:
        if int(chunk["char_start"]) <= offset < int(chunk["char_end"]):
            return str(chunk["chunk_id"])
    return None


def _tool_basename(value: str) -> str:
    basename = re.split(r"[/\\]", value)[-1].lower()
    return basename[:-4] if basename.endswith(".exe") else basename


def _compiler_action_match(content: str) -> tuple[str, re.Match[str]] | None:
    matches: list[tuple[int, str, re.Match[str]]] = []
    for name, pattern in _COMPILER_PATTERNS:
        for match in pattern.finditer(content):
            matches.append((match.start(), name, match))
    if not matches:
        return None
    _start, name, match = min(matches, key=lambda item: item[0])
    return name, match


def _is_package_version(value: str) -> bool:
    """Recognize dotted package versions in linear time."""

    if not value:
        return False
    offset = 1 if value[0] in {"v", "V"} else 0
    numeric_components = 0
    length = len(value)
    while offset < length:
        start = offset
        while offset < length and "0" <= value[offset] <= "9":
            offset += 1
        if offset == start:
            return False
        numeric_components += 1
        if (
            offset + 1 < length
            and value[offset] == "."
            and "0" <= value[offset + 1] <= "9"
        ):
            offset += 1
            continue
        break
    if numeric_components < 2:
        return False
    while offset < length:
        if value[offset] in "-+._~":
            offset += 1
            if offset == length:
                return False
        start = offset
        while offset < length and (
            ("0" <= value[offset] <= "9")
            or ("A" <= value[offset] <= "Z")
            or ("a" <= value[offset] <= "z")
        ):
            offset += 1
        if offset == start:
            return False
    return True


def _is_package_listing_row(
    content: str, tool_match: re.Match[str]
) -> bool:
    """Reject package inventory rows that merely begin with a build tool."""

    suffix = content[tool_match.end() :]
    if suffix.startswith("/") and re.search(
        r"(?i)\[(?:installed|upgradable)\]", suffix
    ):
        return True
    tokens = content[tool_match.start() :].strip().split()
    if len(tokens) < 2:
        return False
    version = tokens[1].rstrip(",;")
    if not _is_package_version(version):
        return False
    if re.match(r"^\s{2,}", suffix):
        return True
    remaining = [token.rstrip(",;") for token in tokens[2:]]
    return bool(
        any(token.lower() in _PACKAGE_CHANNELS for token in remaining)
        or any(_PACKAGE_BUILD_ID_RE.fullmatch(token) for token in remaining)
        or any(_PACKAGE_SIZE_RE.fullmatch(token) for token in remaining)
    )


def _build_action_paths(
    line_paths: Sequence[Mapping[str, Any]],
    *,
    action_start: int,
    command: str,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    source_inputs: list[str] = []
    outputs: list[str] = []
    replacements: list[dict[str, Any]] = []
    for path in line_paths:
        start = int(path["start_char"])
        if start < action_start:
            continue
        value = str(path["value"])
        local_start = start - action_start
        local_end = int(path["end_char"]) - action_start
        language = str(path.get("likely_language") or "")
        category = str(path.get("category") or "")
        if language in {"C", "C/C++", "C++", "CUDA", "module"}:
            source_inputs.append(value)
            replacement = "<SOURCE>"
        elif category == "output":
            outputs.append(value)
            replacement = "<OUTPUT>"
        else:
            continue
        replacements.append(
            {
                "start": local_start,
                "end": local_end,
                "replacement": replacement,
            }
        )

    explicit_output_patterns = (
        re.compile(r"(?:^|\s)-o\s+(?P<value>\S+)"),
        re.compile(r"(?i)(?:^|\s)/(?:Fo|Fe|OUT:)(?P<value>\S+)"),
    )
    for pattern in explicit_output_patterns:
        for match in pattern.finditer(command):
            value = match.group("value").rstrip(",;")
            if value not in outputs:
                outputs.append(value)
            start, end = match.span("value")
            replacements.append(
                {"start": start, "end": end, "replacement": "<OUTPUT>"}
            )
    return (
        list(dict.fromkeys(source_inputs)),
        list(dict.fromkeys(outputs)),
        replacements,
    )


def _normalized_action_shape(
    command: str,
    replacements: Sequence[Mapping[str, Any]],
) -> str:
    output = command
    unique_replacements = {
        (int(replacement["start"]), int(replacement["end"])): str(
            replacement["replacement"]
        )
        for replacement in replacements
    }
    for replacement in sorted(
        (
            {"start": start, "end": end, "replacement": value}
            for (start, end), value in unique_replacements.items()
        ),
        key=lambda item: int(item["start"]),
        reverse=True,
    ):
        start = int(replacement["start"])
        end = int(replacement["end"])
        if 0 <= start < end <= len(output):
            output = (
                output[:start]
                + str(replacement["replacement"])
                + output[end:]
            )
    return re.sub(r"[ \t]+", " ", output).strip()


def _action_flags(command: str) -> tuple[list[str], str]:
    flags: list[str] = []
    for token in re.findall(r"(?:^|\s)(\S+)", command):
        if token.startswith("-"):
            flags.append(token)
        elif re.match(
            r"(?i)^/(?:c|D|I|Fo|Fe|EH|std:|O|W|MD|MT|OUT:|LIBPATH:)",
            token,
        ):
            flags.append(token)
    return flags[:24], _sequence_digest(flags)


def _repo_source_binding(
    path: str,
    provenance: Mapping[str, Any],
    *,
    cwd: str | None,
) -> dict[str, Any] | None:
    run = provenance.get("run") or {}
    head_sha = run.get("head_sha") if isinstance(run, Mapping) else None
    event = run.get("event") if isinstance(run, Mapping) else None
    canonical_repository = provenance.get("repository")
    source_repository = provenance.get("source_repository")
    repository = (
        canonical_repository
        if event in {"pull_request", "pull_request_target"}
        else source_repository or canonical_repository
    )
    if not isinstance(repository, str) or not repository:
        return None
    if not isinstance(head_sha, str) or not head_sha:
        return None
    repo_name = repository.rsplit("/", 1)[-1]
    normalized = path.replace("\\", "/")
    normalized_cwd = cwd.replace("\\", "/") if isinstance(cwd, str) else None
    marker = f"/{repo_name}/"
    marker_index = normalized.rfind(marker)
    if marker_index >= 0:
        source_path = normalized[marker_index + len(marker) :]
        score = 0.8
        method = "workspace_repo_basename_suffix_v1"
    elif not normalized.startswith("/"):
        source_path = _normalized_repo_relative_path(normalized, normalized_cwd)
        if source_path is None:
            return None
        score = 0.95
        method = "relative_source_path_v1"
    else:
        return None
    return {
        "repository": repository,
        "head_sha": head_sha,
        "source_path": source_path,
        "confidence": _confidence(score, source=method),
    }


def _normalized_repo_relative_path(
    source_path: str,
    cwd: str | None,
) -> str | None:
    cwd_suffix: list[str] = []
    if cwd:
        cwd_components = [part for part in cwd.split("/") if part]
        folded = [part.casefold() for part in cwd_components]
        if (
            len(cwd_components) >= 5
            and folded[:3] in (
                ["home", "runner", "work"],
                ["users", "runner", "work"],
            )
        ):
            cwd_suffix = cwd_components[5:]
        elif len(cwd_components) >= 3 and folded[0] == "__w":
            cwd_suffix = cwd_components[3:]
        elif (
            len(cwd_components) >= 4
            and re.fullmatch(r"[a-z]:", folded[0])
            and folded[1] == "a"
        ):
            cwd_suffix = cwd_components[4:]

    components = list(cwd_suffix)
    for component in source_path.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            if not components:
                return None
            components.pop()
            continue
        if "\0" in component:
            return None
        components.append(component)
    return "/".join(components) or None


def _extract_build_actions(
    lines: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    paths: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    builder: _EntityBuilder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract bounded compile/link/archive/build actions from exact log lines."""

    paths_by_line: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for path in paths:
        paths_by_line[int(path["line_index"])].append(path)
    current_cwd: str | None = None
    raw_actions: list[dict[str, Any]] = []

    for line in lines:
        content = str(line["content"])
        cwd_match = re.search(
            r"(?:Working directory is ['\"](?P<quoted>[^'\"]+)['\"]"
            r"|Running command in (?P<running>[^:]+):)",
            content,
            re.IGNORECASE,
        )
        if cwd_match:
            current_cwd = (cwd_match.group("quoted") or cwd_match.group("running")).strip()
        inline_cwd_match = re.match(
            r"^\s*cd\s+(?P<cwd>.+?)\s+&&\s+", content
        )
        inline_cwd = inline_cwd_match.group("cwd").strip("'\"") if inline_cwd_match else None

        compiler_match = _compiler_action_match(content)
        linker_match = _LINK_ARCHIVE_TOOL_RE.search(content)
        tool_match: re.Match[str] | None = None
        tool: str | None = None
        kind: str | None = None
        action_method: str | None = None
        if compiler_match is not None:
            compiler_name, candidate = compiler_match
            candidate_command = content[candidate.start() :]
            has_compile_signal = bool(
                re.search(r"(?:^|\s)(?:-c|/c)(?:\s|$)", candidate_command, re.I)
            )
            has_source = any(
                str(path.get("likely_language"))
                in {"C", "C/C++", "C++", "CUDA", "module"}
                and int(path["start_char"])
                >= int(line["start"]) + candidate.start()
                for path in paths_by_line[int(line["index"])]
            )
            has_output = bool(
                re.search(
                    r"(?:^|\s)(?:-o\s+\S+|/(?:Fo|Fe|OUT:)\S+)",
                    candidate_command,
                    re.I,
                )
            )
            if has_source and (has_compile_signal or has_output):
                tool_match = candidate
                tool = compiler_name
                kind = "compile" if has_compile_signal else "link"
                action_method = "compiler_command_v1"
        if (
            tool_match is None
            and linker_match is not None
            and re.search(r"\.(?:o|obj|a|lib)(?:\s|$)", content, re.I)
        ):
            tool_match = linker_match
            tool = _tool_basename(linker_match.group("tool"))
            kind = "archive" if tool in {"ar", "llvm-ar", "lib"} else "link"
            action_method = "linker_archive_command_v1"
        if tool_match is None:
            build_candidates: list[
                tuple[int, str, re.Match[str]]
            ] = []
            for build_name, _domain, pattern in _BUILD_SYSTEM_PATTERNS:
                for candidate in pattern.finditer(content):
                    build_candidates.append(
                        (candidate.start(), build_name, candidate)
                    )
            if build_candidates:
                _candidate_start, build_name, candidate = min(
                    build_candidates, key=lambda item: item[0]
                )
                prefix = content[: candidate.start()].strip()
                command_prefix = bool(
                    not prefix
                    or re.fullmatch(
                        r"(?:##\[group\](?:Run|Post|Uses)|\[command\]|\+|\$)",
                        prefix,
                        re.IGNORECASE,
                    )
                    or re.match(r"^cd\s+.+?\s+&&$", prefix)
                )
                if command_prefix and not _is_package_listing_row(
                    content, candidate
                ):
                    tool_match = candidate
                    tool = build_name
                    kind = (
                        "configure"
                        if build_name in {"cmake", "meson", "autotools"}
                        and not re.search(
                            r"\b(?:--build|compile|make)\b",
                            content[candidate.start() :],
                            re.IGNORECASE,
                        )
                        else "build"
                    )
                    action_method = "build_system_command_v1"

        if (
            tool_match is None
            or tool is None
            or kind is None
            or action_method is None
        ):
            continue
        local_start = tool_match.start()
        action_start = int(line["start"]) + local_start
        command = content[local_start:].strip()
        command_end = action_start + len(command)
        source_inputs, outputs, replacements = _build_action_paths(
            paths_by_line[int(line["index"])],
            action_start=action_start,
            command=command,
        )
        shape = _normalized_action_shape(command, replacements)
        explicit_targets = [
            match.group("target").rstrip(",;")
            for match in _target_matches(command)
        ]
        flags, flags_sha256 = _action_flags(command)
        section_ordinal, step_ordinal = _line_context(line, sections)
        bindings = [
            binding
            for source in source_inputs
            if (
                binding := _repo_source_binding(
                    source,
                    provenance,
                    cwd=inline_cwd or current_cwd,
                )
            )
            is not None
        ]
        action_domain = {
            "cmake": "CMAKE",
            "make": "MAKE",
            "ninja": "NINJA",
            "bazel": "BAZEL",
            "meson": "MESON",
            "autotools": "CONFIGURE",
            "gn": "GN",
            "scons": "SCONS",
            "xmake": "XMAKE",
            "msbuild": "BUILD_DIAGNOSTIC",
        }.get(tool, "CPP" if kind == "compile" else "BUILD_DIAGNOSTIC")
        action_entity_ref = builder.add(
            kind="build_action",
            role="COMMAND",
            domain=action_domain,
            start=action_start,
            end=command_end,
            confidence=0.98,
            method=action_method,
            line_index=int(line["index"]),
            section_ordinal=section_ordinal,
            step_ordinal=step_ordinal,
            attributes={
                "tool": tool,
                "action_kind": kind,
                "command_sha256": _sha256_text(command),
            },
        )
        for path in paths_by_line[int(line["index"])]:
            if int(path["start_char"]) < action_start:
                continue
            category = str(path.get("category") or "")
            if category != "output":
                builder.edge(
                    action_entity_ref,
                    path.get("entity_ref"),
                    "BUILD_ACTION_INPUT",
                    confidence=0.95,
                    method="build_action_path_role_v1",
                )
            elif category == "output":
                builder.edge(
                    action_entity_ref,
                    path.get("entity_ref"),
                    "BUILD_ACTION_OUTPUT",
                    confidence=0.95,
                    method="build_action_path_role_v1",
                )
        for target_match in _target_matches(command):
            target_value = target_match.group("target").rstrip(",;")
            target_start = action_start + target_match.start("target")
            target_ref = builder.add(
                kind="build_target",
                role="TARGET",
                domain="BUILD_DIAGNOSTIC",
                start=target_start,
                end=target_start + len(target_value),
                confidence=0.95,
                method="build_action_explicit_target_v1",
                line_index=int(line["index"]),
                section_ordinal=section_ordinal,
                step_ordinal=step_ordinal,
            )
            builder.edge(
                action_entity_ref,
                target_ref,
                "BUILD_COMMAND_TARGET",
                confidence=0.95,
                method="build_action_explicit_target_v1",
            )
        raw_actions.append(
            {
                "normalization_schema": BUILD_ACTION_NORMALIZATION_SCHEMA,
                "tool": tool,
                "kind": kind,
                "command": command,
                "command_char_count": len(command),
                "command_sha256": _sha256_text(command),
                "action_shape": shape,
                "action_shape_sha256": _sha256_text(shape),
                "flags": flags,
                "all_flags_sha256": flags_sha256,
                "source_inputs": source_inputs,
                "source_input_count": len(source_inputs),
                "all_source_inputs_sha256": _sequence_digest(source_inputs),
                "outputs": outputs,
                "output_count": len(outputs),
                "all_outputs_sha256": _sequence_digest(outputs),
                "target": (
                    explicit_targets[0]
                    if explicit_targets
                    else outputs[0]
                    if outputs
                    else None
                ),
                "cwd": inline_cwd or current_cwd,
                "repository_source_bindings": bindings,
                "repository_source_binding_count": len(bindings),
                "action_entity_ref": action_entity_ref,
                "start_char": action_start,
                "end_char": command_end,
                "line_index": int(line["index"]),
                "section_ordinal": section_ordinal,
                "step_ordinal": step_ordinal,
                "chunk_id": _chunk_for_char(chunks, action_start),
                "confidence": _confidence(
                    0.98, source=action_method
                ),
            }
        )

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for action in raw_actions:
        key = (
            action["tool"],
            action["kind"],
            action["action_shape_sha256"],
            action["section_ordinal"],
            action["chunk_id"],
        )
        groups.setdefault(key, []).append(action)
    aggregated: list[dict[str, Any]] = []
    for _key, actions in groups.items():
        representative = dict(actions[0])
        representative["source_inputs"] = representative["source_inputs"][:8]
        representative["outputs"] = representative["outputs"][:8]
        representative["repository_source_bindings"] = representative[
            "repository_source_bindings"
        ][:8]
        representative["occurrence_count"] = len(actions)
        evidence = []
        for sample in _bounded_samples(
            actions, limit=MAX_OCCURRENCE_SAMPLES
        ):
            action = sample["value"]
            command = str(action["command"])
            evidence.append(
                {
                    "sequence_index": sample["sequence_index"],
                    "start_char": action["start_char"],
                    "end_char": action["end_char"],
                    "line_index": action["line_index"],
                    "command": command[:MAX_EVIDENCE_TEXT_CHARS],
                    "command_char_count": len(command),
                    "command_sha256": action["command_sha256"],
                    "source_inputs": action["source_inputs"],
                    "outputs": action["outputs"],
                }
            )
        representative["representative_actions"] = evidence
        representative["all_actions_sha256"] = _sequence_digest(actions)
        representative["omitted_action_count"] = max(
            0, len(actions) - MAX_OCCURRENCE_SAMPLES
        )
        _clip_evidence_field(representative, "command")
        _clip_evidence_field(representative, "action_shape")
        aggregated.append(representative)
    return aggregated, raw_actions


def _duration_ms(value: str, unit: str) -> float:
    amount = float(value)
    return amount if unit.lower().startswith("ms") else amount * 1000.0


def _extract_tests(
    lines: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    builder: _EntityBuilder,
    commands: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    framework_commands: dict[str, int | None] = {}
    for command in commands:
        lowered = str(command["text"]).lower()
        for framework in ("pytest", "ctest"):
            if re.search(rf"(?<![\w.-]){framework}(?![\w.-])", lowered):
                framework_commands[framework] = command["entity_ref"]

    pytest_case = re.compile(
        r"^(?P<name>\S+::\S+)\s+"
        r"(?P<result>PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)"
        r"(?:\s+\[[^\]]+\])?",
        re.IGNORECASE,
    )
    gtest_case = re.compile(
        r"^\[\s*(?P<result>RUN|OK|FAILED|SKIPPED)\s*\]\s+"
        r"(?P<name>[A-Za-z0-9_./-]+\.[A-Za-z0-9_./-]+)"
        r"(?:\s+\((?P<duration>\d+(?:\.\d+)?)\s*ms\))?",
        re.IGNORECASE,
    )
    ctest_case = re.compile(
        r"^\s*(?P<ordinal>\d+)/(?P<total>\d+)\s+"
        r"Test\s+#\d+:\s+(?P<name>.+?)\s+\.+\s+"
        r"(?P<result>Passed|\*\*\*Failed|Failed|Not Run|Timeout)"
        r"(?:\s+(?P<duration>\d+(?:\.\d+)?)\s+sec)?\s*$",
        re.IGNORECASE,
    )

    for line in lines:
        content = str(line["content"])
        section_ordinal, step_ordinal = _line_context(line, sections)
        match: re.Match[str] | None
        framework: str
        duration_ms: float | None = None
        suite: str | None = None
        command_ref: int | None = None

        match = pytest_case.match(content)
        if match:
            framework = "pytest"
            name = match.group("name")
            result = match.group("result").lower()
            suite = name.split("::", 1)[0]
            command_ref = framework_commands.get(framework)
        else:
            match = gtest_case.match(content)
            if match:
                framework = "gtest"
                name = match.group("name")
                raw_result = match.group("result").lower()
                result = "running" if raw_result == "run" else raw_result
                suite = name.rsplit(".", 1)[0]
                if match.group("duration"):
                    duration_ms = float(match.group("duration"))
                command_ref = framework_commands.get(framework)
            else:
                match = ctest_case.match(content)
                if not match:
                    continue
                framework = "ctest"
                name = match.group("name").strip()
                raw_result = match.group("result").lower().replace("*", "")
                result = (
                    "failed"
                    if "failed" in raw_result
                    else "skipped"
                    if raw_result in {"not run", "timeout"}
                    else "passed"
                )
                if match.group("duration"):
                    duration_ms = float(match.group("duration")) * 1000.0
                command_ref = framework_commands.get(framework)

        start = int(line["start"]) + match.start("name")
        end = int(line["start"]) + match.end("name")
        entity_ref = builder.add(
            kind="test_case",
            role="TEST_NAME",
            domain="TEST_OUTPUT",
            start=start,
            end=end,
            confidence=1.0,
            method=f"{framework}_case_output_v1",
            line_index=int(line["index"]),
            section_ordinal=section_ordinal,
            step_ordinal=step_ordinal,
            attributes={"result": result, "suite": suite},
        )
        test_records.append(
            {
                "framework": framework,
                "command_entity_ref": command_ref,
                "suite": suite,
                "case": name,
                "result": result,
                "count": 1,
                "duration_ms": duration_ms,
                "start_char": start,
                "end_char": end,
                "line_index": int(line["index"]),
                "section_ordinal": section_ordinal,
                "step_ordinal": step_ordinal,
                "entity_ref": entity_ref,
                "confidence": _confidence(
                    1.0, source=f"{framework}_case_output_v1"
                ),
            }
        )

    pytest_summary_event = re.compile(
        r"\b(?:"
        r"(?P<count>\d+)\s+(?P<result>passed|failed|skipped)"
        r"|in\s+(?P<duration>\d+(?:\.\d+)?)s"
        r")\b",
        re.IGNORECASE,
    )
    ctest_summary = re.compile(
        r"(?P<percent>\d+)% tests passed,\s*"
        r"(?P<failed>\d+) tests failed out of (?P<total>\d+)",
        re.IGNORECASE,
    )
    gtest_summary = re.compile(
        r"^\[=+\]\s+(?P<total>\d+) tests? from "
        r"(?P<suites>\d+) test suites? ran\.\s*"
        r"\((?P<duration>\d+(?:\.\d+)?)\s*ms total\)",
        re.IGNORECASE,
    )
    for line in lines:
        content = str(line["content"])
        match = ctest_summary.search(content)
        if match:
            total = int(match.group("total"))
            failed = int(match.group("failed"))
            summaries.append(
                {
                    "framework": "ctest",
                    "passed": total - failed,
                    "failed": failed,
                    "skipped": None,
                    "total": total,
                    "duration_ms": None,
                    "line_index": int(line["index"]),
                    "confidence": _confidence(
                        1.0, source="ctest_summary_v1"
                    ),
                }
            )
            continue
        match = gtest_summary.search(content)
        if match:
            summaries.append(
                {
                    "framework": "gtest",
                    "passed": None,
                    "failed": None,
                    "skipped": None,
                    "total": int(match.group("total")),
                    "suite_count": int(match.group("suites")),
                    "duration_ms": float(match.group("duration")),
                    "line_index": int(line["index"]),
                    "confidence": _confidence(
                        1.0, source="gtest_summary_v1"
                    ),
                }
            )
            continue
        pytest_counts: dict[str, int] = {}
        for event in pytest_summary_event.finditer(content):
            result = event.group("result")
            if result is not None:
                pytest_counts[result.lower()] = int(event.group("count"))
                continue
            if "passed" in pytest_counts or "failed" in pytest_counts:
                passed = pytest_counts.get("passed", 0)
                failed = pytest_counts.get("failed", 0)
                skipped = pytest_counts.get("skipped", 0)
                summaries.append(
                    {
                        "framework": "pytest",
                        "passed": passed,
                        "failed": failed,
                        "skipped": skipped,
                        "total": passed + failed + skipped,
                        "duration_ms": float(event.group("duration")) * 1000.0,
                        "line_index": int(line["index"]),
                        "confidence": _confidence(
                            1.0, source="pytest_summary_v1"
                        ),
                    }
                )
                break
            pytest_counts.clear()
    return test_records, summaries


def _nearest_command(
    commands: Sequence[Mapping[str, Any]],
    *,
    section_ordinal: int,
    line_index: int,
) -> Mapping[str, Any] | None:
    candidates = [
        command
        for command in commands
        if int(command["section_ordinal"]) == section_ordinal
        and int(command["line_index"]) <= line_index
    ]
    return candidates[-1] if candidates else None


def _extract_diagnostics(
    lines: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    builder: _EntityBuilder,
    commands: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    targets_by_section: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for target in targets:
        targets_by_section[int(target["section_ordinal"])].append(target)

    for line in lines:
        content = str(line["content"])
        section_ordinal, step_ordinal = _line_context(line, sections)
        match = (
            _MSVC_DIAGNOSTIC_RE.match(content)
            or _GCC_DIAGNOSTIC_RE.match(content)
            or _CMAKE_DIAGNOSTIC_RE.match(content)
        )
        category: str | None = None
        tool: str | None = None
        severity: str | None = None
        code: str | None = None
        message = content
        file_value: str | None = None
        source_line: int | None = None
        source_column: int | None = None
        file_span: tuple[int, int] | None = None

        if match:
            severity = (match.groupdict().get("severity") or "error").lower()
            code = match.groupdict().get("code")
            message = (match.groupdict().get("message") or content).strip()
            file_value = match.groupdict().get("file")
            if file_value:
                file_span = (
                    int(line["start"]) + match.start("file"),
                    int(line["start"]) + match.end("file"),
                )
            if match.groupdict().get("line"):
                source_line = int(match.group("line"))
            if match.groupdict().get("column"):
                source_column = int(match.group("column"))
            if _MSVC_DIAGNOSTIC_RE.match(content):
                category, tool = "compiler", "msvc"
            elif _CMAKE_DIAGNOSTIC_RE.match(content):
                category, tool = "build", "cmake"
            else:
                category, tool = "compiler", "gcc_or_clang"
        elif re.search(
            r"AddressSanitizer|ThreadSanitizer|LeakSanitizer|"
            r"MemorySanitizer|UndefinedBehaviorSanitizer|"
            r"\bruntime error:",
            content,
            re.IGNORECASE,
        ):
            category, tool, severity = "sanitizer", "sanitizer", "error"
        elif re.search(
            r"undefined reference|unresolved external|LNK(?:2001|2019)|"
            r"\bld(?:\.exe)?: (?:fatal )?error|collect2: error",
            content,
            re.IGNORECASE,
        ):
            category, tool, severity = "linker", "linker", "error"
        elif content.startswith("##[error]"):
            category, tool, severity = "workflow", "github_actions", "error"
            message = content[len("##[error]") :].strip()
        elif re.search(
            r"ninja: build stopped|make(?:\[\d+\])?: \*\*\*|"
            r"BUILD FAILED|CMake Error",
            content,
            re.IGNORECASE,
        ):
            category, tool, severity = "build", "build_system", "error"
        if category is None:
            continue

        line_start = int(line["start"])
        line_end = int(line["content_end"])
        domain = {
            "compiler": (
                "COMPILER_ERROR" if severity in {"error", "fatal error"} else "COMPILER_DIAGNOSTIC"
            ),
            "build": "BUILD_ERROR",
            "linker": "LINKER_ERROR",
            "sanitizer": "SANITIZER_OUTPUT",
            "workflow": "BUILD_ERROR",
        }[category]
        diagnostic_ref = builder.add(
            kind="diagnostic",
            role="MESSAGE",
            domain=domain,
            start=line_start,
            end=line_end,
            confidence=0.98 if match else 0.9,
            method=f"{category}_diagnostic_v1",
            line_index=int(line["index"]),
            section_ordinal=section_ordinal,
            step_ordinal=step_ordinal,
            attributes={
                "category": category,
                "tool": tool,
                "severity": severity,
                "code": code,
            },
        )
        severity_ref: int | None = None
        if severity:
            severity_match = re.search(
                rf"\b{re.escape(severity)}\b", content, re.IGNORECASE
            )
            if severity_match:
                severity_ref = builder.add(
                    kind="severity",
                    role="SEVERITY",
                    domain=domain,
                    start=line_start + severity_match.start(),
                    end=line_start + severity_match.end(),
                    confidence=1.0,
                    method="diagnostic_severity_token",
                    line_index=int(line["index"]),
                    section_ordinal=section_ordinal,
                    step_ordinal=step_ordinal,
                )
        file_ref: int | None = None
        if file_span and file_value:
            file_ref = builder.add(
                kind="path",
                role="FILE",
                domain=domain,
                start=file_span[0],
                end=file_span[1],
                confidence=1.0,
                method="diagnostic_location",
                line_index=int(line["index"]),
                section_ordinal=section_ordinal,
                step_ordinal=step_ordinal,
                attributes={
                    "category": "source",
                    "likely_language": _path_language(file_value),
                },
            )
            builder.edge(
                diagnostic_ref,
                file_ref,
                "DIAG_PRIMARY_LOCATION",
                confidence=1.0,
                method="diagnostic_location",
            )

        symbol: str | None = None
        symbol_ref: int | None = None
        symbol_match = re.search(
            r"undefined reference to\s+[`'\"](?P<symbol>.+?)[`'\"]",
            content,
            re.IGNORECASE,
        )
        if symbol_match:
            symbol = symbol_match.group("symbol")
            symbol_ref = builder.add(
                kind="symbol",
                role="SYMBOL",
                domain="LINKER_ERROR",
                start=line_start + symbol_match.start("symbol"),
                end=line_start + symbol_match.end("symbol"),
                confidence=1.0,
                method="linker_undefined_symbol",
                line_index=int(line["index"]),
                section_ordinal=section_ordinal,
                step_ordinal=step_ordinal,
            )
            builder.edge(
                diagnostic_ref,
                symbol_ref,
                "LINK_UNDEFINED_SYMBOL",
                confidence=1.0,
                method="linker_undefined_symbol",
            )

        command = _nearest_command(
            commands,
            section_ordinal=section_ordinal,
            line_index=int(line["index"]),
        )
        if command is not None:
            builder.edge(
                diagnostic_ref,
                command["entity_ref"],
                "DIAG_COMMAND",
                confidence=0.8,
                method="nearest_preceding_command_in_step",
            )
        for target in targets_by_section[section_ordinal]:
            builder.edge(
                diagnostic_ref,
                target["entity_ref"],
                "DIAG_BUILD_TARGET",
                confidence=0.65,
                method="target_in_same_step",
            )

        diagnostics.append(
            {
                "category": category,
                "tool": tool,
                "severity": severity,
                "code": code,
                "message": message,
                "file": file_value,
                "source_line": source_line,
                "source_column": source_column,
                "symbol": symbol,
                "start_char": line_start,
                "end_char": line_end,
                "line_index": int(line["index"]),
                "section_ordinal": section_ordinal,
                "step_ordinal": step_ordinal,
                "entity_ref": diagnostic_ref,
                "file_entity_ref": file_ref,
                "severity_entity_ref": severity_ref,
                "symbol_entity_ref": symbol_ref,
                "confidence": _confidence(
                    0.98 if match else 0.9,
                    source=f"{category}_diagnostic_v1",
                ),
            }
        )
    return diagnostics


def _extract_system_metadata(
    lines: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    group: str | None = None
    group_values: defaultdict[str, list[tuple[str, int, int]]] = defaultdict(list)

    def observe(
        field: str,
        value: Any,
        line: Mapping[str, Any],
        *,
        score: float = 1.0,
        method: str = "github_actions_system_line",
    ) -> None:
        observations.append(
            {
                "field": field,
                "value": value,
                "start_char": int(line["start"]),
                "end_char": int(line["content_end"]),
                "line_index": int(line["index"]),
                "confidence": _confidence(score, source=method),
            }
        )

    for line in lines:
        content = str(line["content"]).strip()
        group_match = re.match(r"^##\[group\](.+)$", content)
        if group_match:
            group = group_match.group(1).strip().lower()
            continue
        if content == "##[endgroup]":
            group = None
            continue
        if not content:
            continue
        if group in {"operating system", "runner image", "runner image provisioner"}:
            group_values[group].append(
                (content, int(line["start"]), int(line["content_end"]))
            )

        direct_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
            (
                "requested_labels",
                re.compile(r"^Requested labels:\s*(?P<value>.+)$", re.I),
            ),
            (
                "workflow_definition",
                re.compile(r"^Job defined at:\s*(?P<value>.+)$", re.I),
            ),
            (
                "runner_instance",
                re.compile(
                    r"hosted runner:\s*(?P<value>GitHub Actions \d+)$", re.I
                ),
            ),
            (
                "runner_version",
                re.compile(
                    r"^Current runner version:\s*['\"]?(?P<value>[^'\"]+)['\"]?$",
                    re.I,
                ),
            ),
            (
                "architecture",
                re.compile(
                    r"^(?:Architecture|Runner architecture):\s*(?P<value>.+)$",
                    re.I,
                ),
            ),
            (
                "container",
                re.compile(
                    r"^(?:Container|Job container):\s*(?P<value>.+)$", re.I
                ),
            ),
        )
        for field, pattern in direct_patterns:
            match = pattern.search(content)
            if not match:
                continue
            value: Any = match.group("value").strip()
            if field == "requested_labels":
                value = [part.strip() for part in value.split(",") if part.strip()]
            observe(field, value, line)

    operating_system = group_values.get("operating system", [])
    plain_os_values = [
        item for item in operating_system if not item[0].startswith("##[")
    ]
    if plain_os_values:
        name = plain_os_values[0][0]
        observations.append(
            {
                "field": "os_name",
                "value": name,
                "start_char": plain_os_values[0][1],
                "end_char": plain_os_values[0][2],
                "line_index": next(
                    (
                        int(line["index"])
                        for line in lines
                        if int(line["start"]) == plain_os_values[0][1]
                    ),
                    None,
                ),
                "confidence": _confidence(
                    0.95, source="operating_system_group_position_v1"
                ),
            }
        )
        if len(plain_os_values) > 1:
            observations.append(
                {
                    "field": "os_version",
                    "value": plain_os_values[1][0],
                    "start_char": plain_os_values[1][1],
                    "end_char": plain_os_values[1][2],
                    "line_index": next(
                        (
                            int(line["index"])
                            for line in lines
                            if int(line["start"]) == plain_os_values[1][1]
                        ),
                        None,
                    ),
                    "confidence": _confidence(
                        0.9, source="operating_system_group_position_v1"
                    ),
                }
            )

    for group_name, field_prefix in (
        ("runner image", "runner_image"),
        ("runner image provisioner", "runner_image_provisioner"),
    ):
        for content, start, end in group_values.get(group_name, []):
            field_match = re.match(
                r"^(?P<name>Image|Version|Image OS):\s*(?P<value>.+)$",
                content,
                re.I,
            )
            if not field_match:
                continue
            suffix = field_match.group("name").lower().replace(" ", "_")
            field = (
                field_prefix
                if suffix == "image" and field_prefix == "runner_image"
                else f"{field_prefix}_{suffix}"
            )
            observations.append(
                {
                    "field": field,
                    "value": field_match.group("value").strip(),
                    "start_char": start,
                    "end_char": end,
                    "line_index": next(
                        (
                            int(line["index"])
                            for line in lines
                            if int(line["start"]) == start
                        ),
                        None,
                    ),
                    "confidence": _confidence(
                        1.0, source=f"github_actions_{group_name.replace(' ', '_')}_field"
                    ),
                }
            )

    latest: dict[str, Any] = {}
    for observation in observations:
        latest[str(observation["field"])] = observation
    return observations, latest


def _platform_field(
    *,
    metadata_value: Any,
    metadata_confidence: Mapping[str, Any],
    observed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if metadata_value is not None:
        candidates.append(
            {
                "value": metadata_value,
                "confidence": dict(metadata_confidence),
            }
        )
    if observed is not None:
        candidates.append(
            {
                "value": observed["value"],
                "confidence": observed["confidence"],
                "span": {
                    "start_char": observed["start_char"],
                    "end_char": observed["end_char"],
                },
            }
        )
    if not candidates:
        return {
            "value": None,
            "confidence": _confidence(0.0, source=None),
            "candidates": [],
        }
    selected = max(
        candidates,
        key=lambda candidate: float(candidate["confidence"]["score"]),
    )
    return {
        "value": selected["value"],
        "confidence": selected["confidence"],
        "candidates": candidates,
    }


def _build_platform(
    provenance: Mapping[str, Any],
    observed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    field_confidence = provenance["field_confidence"]
    runner = provenance["runner"]
    job = provenance["job"]
    return {
        "os": _platform_field(
            metadata_value=runner.get("os"),
            metadata_confidence=field_confidence["runner_os"],
            observed=observed.get("os_name"),
        ),
        "os_version": _platform_field(
            metadata_value=None,
            metadata_confidence=_confidence(0.0, source=None),
            observed=observed.get("os_version"),
        ),
        "architecture": _platform_field(
            metadata_value=runner.get("arch"),
            metadata_confidence=field_confidence["runner_arch"],
            observed=observed.get("architecture"),
        ),
        "runner_name": _platform_field(
            metadata_value=runner.get("name"),
            metadata_confidence=field_confidence["runner_name"],
            observed=observed.get("runner_instance"),
        ),
        "runner_version": _platform_field(
            metadata_value=None,
            metadata_confidence=_confidence(0.0, source=None),
            observed=observed.get("runner_version"),
        ),
        "runner_image": _platform_field(
            metadata_value=runner.get("image"),
            metadata_confidence=field_confidence["runner_image"],
            observed=observed.get("runner_image"),
        ),
        "runner_image_version": _platform_field(
            metadata_value=None,
            metadata_confidence=_confidence(0.0, source=None),
            observed=observed.get("runner_image_version"),
        ),
        "container": _platform_field(
            metadata_value=job.get("container"),
            metadata_confidence=field_confidence["container"],
            observed=observed.get("container"),
        ),
        "runner_labels": _platform_field(
            metadata_value=runner.get("labels") or None,
            metadata_confidence=field_confidence["runner_labels"],
            observed=observed.get("requested_labels"),
        ),
        "matrix": {
            "value": job.get("matrix"),
            "confidence": field_confidence["matrix"],
        },
    }


def _replace_entity_refs(value: Any, id_for_temp: Mapping[int, str]) -> Any:
    if isinstance(value, list):
        return [_replace_entity_refs(item, id_for_temp) for item in value]
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if key == "entity_ref" or key.endswith("_entity_ref"):
                output[key.removesuffix("_ref") + "_id"] = (
                    id_for_temp[item] if isinstance(item, int) else None
                )
            else:
                output[key] = _replace_entity_refs(item, id_for_temp)
        return output
    return value


def _remap_entity_ids(value: Any, old_to_new: Mapping[str, str]) -> Any:
    if isinstance(value, list):
        return [_remap_entity_ids(item, old_to_new) for item in value]
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("entity_id") and isinstance(item, str):
                output[key] = old_to_new.get(item)
            else:
                output[key] = _remap_entity_ids(item, old_to_new)
        return output
    return value


def _clip_evidence_field(
    record: dict[str, Any],
    field: str,
    *,
    limit: int = MAX_EVIDENCE_TEXT_CHARS,
) -> None:
    value = record.get(field)
    if not isinstance(value, str) or len(value) <= limit:
        return
    record[field] = value[:limit]
    record[f"{field}_char_count"] = len(value)
    record[f"{field}_sha256"] = _sha256_text(value)
    record[f"{field}_truncated"] = True


def _compact_entities_and_edges(
    entities: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
    dict[str, Any],
]:
    """Group repeated semantic entities and retain bounded, digested spans."""

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    old_to_group: dict[str, tuple[Any, ...]] = {}
    for entity in entities:
        key = (
            entity.get("kind"),
            entity.get("role"),
            entity.get("domain"),
            entity.get("text"),
            entity.get("section_ordinal"),
            entity.get("step_ordinal"),
        )
        old_to_group[str(entity["entity_id"])] = key
        group = groups.setdefault(
            key,
            {
                "representative": dict(entity),
                "occurrences": [],
                "attributes": [],
            },
        )
        group["occurrences"].append(
            {
                "start_char": int(entity["start_char"]),
                "end_char": int(entity["end_char"]),
                "line_index": int(entity["line_index"]),
            }
        )
        group["attributes"].append(entity.get("attributes") or {})

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            int(item[1]["representative"]["start_char"]),
            int(item[1]["representative"]["end_char"]),
            str(item[0]),
        ),
    )
    all_group_receipts = [
        {
            "identity": list(key),
            "occurrence_count": len(group["occurrences"]),
            "occurrence_sha256": _sequence_digest(group["occurrences"]),
            "attribute_sha256": _sequence_digest(group["attributes"]),
        }
        for key, group in ordered_groups
    ]
    entity_priority = {
        "diagnostic": 0,
        "symbol": 0,
        "severity": 0,
        "test_case": 1,
        "build_target": 1,
        "command": 2,
        "build_system": 3,
        "compiler": 3,
        "shell_dialect": 3,
        "path": 4,
    }
    # Reserve both endpoints of the first occurrence of every edge kind.  This
    # keeps each observed relation usable after global entity compaction while
    # the full occurrence set remains represented by the accounting digest.
    representative_endpoint_groups: set[tuple[Any, ...]] = set()
    represented_edge_kinds: set[str] = set()
    for edge in edges:
        edge_kind = str(edge.get("kind"))
        if edge_kind in represented_edge_kinds:
            continue
        represented_edge_kinds.add(edge_kind)
        for endpoint in ("source", "target"):
            group_key = old_to_group.get(str(edge.get(endpoint)))
            if group_key is not None:
                representative_endpoint_groups.add(group_key)
    selected_group_keys = {
        key
        for key, _group in sorted(
            ordered_groups,
            key=lambda item: (
                (
                    -1
                    if item[0] in representative_endpoint_groups
                    else entity_priority.get(str(item[0][0]), 3)
                ),
                int(item[1]["representative"]["start_char"]),
                str(item[0]),
            ),
        )[:MAX_ENTITY_GROUPS]
    }
    retained_groups = [
        (key, group)
        for key, group in ordered_groups
        if key in selected_group_keys
    ]
    compact_entities: list[dict[str, Any]] = []
    compact_id_for_group: dict[tuple[Any, ...], str] = {}
    for ordinal, (key, group) in enumerate(retained_groups):
        compact_id = f"entity:{ordinal:06d}"
        compact_id_for_group[key] = compact_id
        representative = dict(group["representative"])
        representative["entity_id"] = compact_id
        occurrences = group["occurrences"]
        representative["occurrence_count"] = len(occurrences)
        representative["occurrence_spans"] = [
            sample["value"]
            for sample in _bounded_samples(
                occurrences, limit=MAX_OCCURRENCE_SAMPLES
            )
        ]
        representative["occurrence_span_sha256"] = _sequence_digest(occurrences)
        representative["occurrence_span_omitted_count"] = max(
            0, len(occurrences) - MAX_OCCURRENCE_SAMPLES
        )
        representative["attribute_variants_sha256"] = _sequence_digest(
            group["attributes"]
        )
        _clip_evidence_field(representative, "text")
        compact_entities.append(representative)

    old_to_new = {
        old_id: compact_id_for_group[group_key]
        for old_id, group_key in old_to_group.items()
        if group_key in compact_id_for_group
    }

    edge_groups: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    omitted_endpoint_edge_count = 0
    for edge in edges:
        source = old_to_new.get(str(edge["source"]))
        target = old_to_new.get(str(edge["target"]))
        if source is None or target is None:
            omitted_endpoint_edge_count += 1
            continue
        key = (
            source,
            target,
            str(edge["kind"]),
            int(edge["kind_id"]),
            str(edge["family"]),
        )
        group = edge_groups.setdefault(
            key, {"representative": dict(edge), "occurrences": []}
        )
        group["occurrences"].append(
            {
                "from_char": int(edge["from_char"]),
                "to_char": int(edge["to_char"]),
            }
        )
    ordered_edges = sorted(
        edge_groups.items(),
        key=lambda item: (
            int(item[1]["representative"]["from_char"]),
            int(item[1]["representative"]["to_char"]),
            int(item[0][3]),
        ),
    )
    all_edge_receipts = [
        {
            "identity": list(key),
            "occurrence_count": len(group["occurrences"]),
            "occurrence_sha256": _sequence_digest(group["occurrences"]),
        }
        for key, group in ordered_edges
    ]
    compact_edges: list[dict[str, Any]] = []
    for ordinal, (key, group) in enumerate(ordered_edges[:MAX_EDGE_GROUPS]):
        representative = dict(group["representative"])
        representative.update(
            edge_id=f"edge:{ordinal:06d}",
            source=key[0],
            target=key[1],
            occurrence_count=len(group["occurrences"]),
            occurrence_spans=[
                sample["value"]
                for sample in _bounded_samples(
                    group["occurrences"], limit=MAX_OCCURRENCE_SAMPLES
                )
            ],
            occurrence_span_sha256=_sequence_digest(group["occurrences"]),
            occurrence_span_omitted_count=max(
                0, len(group["occurrences"]) - MAX_OCCURRENCE_SAMPLES
            ),
        )
        compact_edges.append(representative)

    edge_kind_occurrences: defaultdict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for edge in edges:
        edge_kind_occurrences[str(edge["kind"])].append(
            {
                "from_char": int(edge["from_char"]),
                "to_char": int(edge["to_char"]),
            }
        )
    edge_kind_summary = []
    for kind in sorted(
        edge_kind_occurrences, key=lambda value: EDGE_IDS.get(value, 0)
    ):
        occurrences = edge_kind_occurrences[kind]
        edge_kind_summary.append(
            {
                "kind": kind,
                "kind_id": EDGE_IDS[kind],
                "family": EDGE_FAMILIES[kind],
                "occurrence_count": len(occurrences),
                "occurrence_samples": [
                    sample["value"]
                    for sample in _bounded_samples(
                        occurrences, limit=MAX_OCCURRENCE_SAMPLES
                    )
                ],
                "all_occurrences_sha256": _sequence_digest(occurrences),
            }
        )

    accounting = {
        "entity_occurrence_count": len(entities),
        "entity_group_count": len(ordered_groups),
        "entity_group_limit": MAX_ENTITY_GROUPS,
        "retained_entity_group_count": len(compact_entities),
        "omitted_entity_group_count": max(
            0, len(ordered_groups) - MAX_ENTITY_GROUPS
        ),
        "all_entity_groups_sha256": _sequence_digest(all_group_receipts),
        "edge_occurrence_count": len(edges),
        "edge_group_count_with_retained_endpoints": len(ordered_edges),
        "edge_group_limit": MAX_EDGE_GROUPS,
        "retained_edge_group_count": len(compact_edges),
        "omitted_edge_group_count": max(
            0, len(ordered_edges) - MAX_EDGE_GROUPS
        ),
        "omitted_endpoint_edge_count": omitted_endpoint_edge_count,
        "all_retained_endpoint_edge_groups_sha256": _sequence_digest(
            all_edge_receipts
        ),
        "edge_kind_summary": edge_kind_summary,
    }
    return compact_entities, compact_edges, old_to_new, accounting


def _compact_record_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str],
    limit: int = MAX_CLASSIFICATION_GROUPS,
    clip_fields: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        identity = {
            field: record.get(field)
            for field in (*identity_fields, "section_ordinal", "step_ordinal")
        }
        key = stable_json_dumps(identity)
        if key not in groups:
            groups[key] = {"representative": dict(record), "occurrences": []}
            order.append(key)
        occurrence = {
            field: record.get(field)
            for field in (
                "start_char",
                "end_char",
                "line_index",
                "entity_id",
            )
            if field in record
        }
        groups[key]["occurrences"].append(occurrence)

    receipts = [
        {
            "identity_sha256": _sha256_text(key),
            "occurrence_count": len(groups[key]["occurrences"]),
            "occurrence_sha256": _sequence_digest(groups[key]["occurrences"]),
        }
        for key in order
    ]
    compact: list[dict[str, Any]] = []
    for key in order[:limit]:
        group = groups[key]
        record = dict(group["representative"])
        occurrences = group["occurrences"]
        record["occurrence_count"] = len(occurrences)
        if occurrences and occurrences[0]:
            record["evidence_spans"] = [
                sample["value"]
                for sample in _bounded_samples(
                    occurrences, limit=MAX_OCCURRENCE_SAMPLES
                )
            ]
            record["evidence_span_sha256"] = _sequence_digest(occurrences)
            record["evidence_span_omitted_count"] = max(
                0, len(occurrences) - MAX_OCCURRENCE_SAMPLES
            )
        for field in clip_fields:
            _clip_evidence_field(record, field)
        compact.append(record)
    return compact, {
        "occurrence_count": len(records),
        "group_count": len(order),
        "group_limit": limit,
        "retained_group_count": len(compact),
        "omitted_group_count": max(0, len(order) - limit),
        "all_groups_sha256": _sequence_digest(receipts),
    }


def _bound_occurrence_field(
    record: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    output = dict(record)
    occurrences = list(output.get(field) or [])
    output[field] = [
        sample["value"]
        for sample in _bounded_samples(
            occurrences, limit=MAX_OCCURRENCE_SAMPLES
        )
    ]
    output[f"{field}_count"] = len(occurrences)
    output[f"{field}_sha256"] = _sequence_digest(occurrences)
    output[f"{field}_omitted_count"] = max(
        0, len(occurrences) - MAX_OCCURRENCE_SAMPLES
    )
    return output


def _compact_classifications(
    classifications: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    compact = dict(classifications)
    accounting: dict[str, Any] = {}
    specs: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "shell_dialects": (("name",), ()),
        "commands": (("text", "shell_dialect"), ("text",)),
        "build_targets": (("name",), ("name",)),
        "paths": (
            ("value", "category", "likely_language"),
            ("value",),
        ),
        "tests": (
            ("framework", "suite", "case", "result"),
            ("case",),
        ),
        "test_summaries": (
            ("framework", "passed", "failed", "skipped", "total"),
            (),
        ),
        "diagnostics": (
            (
                "category",
                "tool",
                "severity",
                "code",
                "message",
                "file",
                "source_line",
                "source_column",
            ),
            ("message", "file"),
        ),
        "system_metadata": (("field", "value"), ()),
    }
    for field, (identity_fields, clip_fields) in specs.items():
        compact[field], accounting[field] = _compact_record_groups(
            list(classifications.get(field) or []),
            identity_fields=identity_fields,
            clip_fields=clip_fields,
        )

    sql_dialects = [
        _bound_occurrence_field(record, field="occurrences")
        for record in classifications.get("sql_dialects") or []
    ]
    compact["sql_dialects"] = sql_dialects[:MAX_CLASSIFICATION_GROUPS]
    accounting["sql_dialects"] = {
        "group_count": len(sql_dialects),
        "retained_group_count": min(
            len(sql_dialects), MAX_CLASSIFICATION_GROUPS
        ),
        "omitted_group_count": max(
            0, len(sql_dialects) - MAX_CLASSIFICATION_GROUPS
        ),
        "all_groups_sha256": _sequence_digest(sql_dialects),
    }

    build_systems = [
        _bound_occurrence_field(record, field="occurrences")
        for record in classifications.get("build_systems") or []
    ]
    compact["build_systems"] = build_systems[:MAX_CLASSIFICATION_GROUPS]
    accounting["build_systems"] = {
        "group_count": len(build_systems),
        "retained_group_count": min(
            len(build_systems), MAX_CLASSIFICATION_GROUPS
        ),
        "omitted_group_count": max(
            0, len(build_systems) - MAX_CLASSIFICATION_GROUPS
        ),
        "all_groups_sha256": _sequence_digest(build_systems),
    }
    toolchains = [
        _bound_occurrence_field(record, field="occurrences")
        for record in classifications.get("toolchains") or []
    ]
    compact["toolchains"] = toolchains[:MAX_CLASSIFICATION_GROUPS]
    accounting["toolchains"] = {
        "group_count": len(toolchains),
        "retained_group_count": min(
            len(toolchains), MAX_CLASSIFICATION_GROUPS
        ),
        "omitted_group_count": max(
            0, len(toolchains) - MAX_CLASSIFICATION_GROUPS
        ),
        "all_groups_sha256": _sequence_digest(toolchains),
    }
    languages = [
        _bound_occurrence_field(record, field="spans")
        for record in classifications.get("languages") or []
    ]
    compact["languages"] = languages[:MAX_CLASSIFICATION_GROUPS]
    accounting["languages"] = {
        "group_count": len(languages),
        "retained_group_count": min(len(languages), MAX_CLASSIFICATION_GROUPS),
        "omitted_group_count": max(
            0, len(languages) - MAX_CLASSIFICATION_GROUPS
        ),
        "all_groups_sha256": _sequence_digest(languages),
    }
    build_actions = list(classifications.get("build_actions") or [])
    compact["build_actions"] = build_actions[:MAX_CLASSIFICATION_GROUPS]
    accounting["build_actions"] = {
        "action_occurrence_count": sum(
            int(action.get("occurrence_count", 1))
            for action in build_actions
        ),
        "group_count": len(build_actions),
        "retained_group_count": min(
            len(build_actions), MAX_CLASSIFICATION_GROUPS
        ),
        "omitted_group_count": max(
            0, len(build_actions) - MAX_CLASSIFICATION_GROUPS
        ),
        "all_groups_sha256": _sequence_digest(build_actions),
        "normalization_schema": BUILD_ACTION_NORMALIZATION_SCHEMA,
    }
    return compact, accounting


def _compact_section_index(
    sections: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for section in sections:
        entry = {
            key: value
            for key, value in section.items()
            if key not in {"text", "dedup_text", "metadata_step"}
        }
        _clip_evidence_field(entry, "title")
        index.append(entry)
    return index


def _canonicalize_payload(
    decoded_text: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    boundary_text, boundary_entries, boundary_removed_chars = (
        _normalize_record_boundaries(decoded_text)
    )
    physical = _physical_lines(boundary_text)
    canonical_parts: list[str] = []
    timestamps: list[dict[str, Any]] = []
    ansi_entries: list[dict[str, Any]] = []
    ansi_sequences: list[str] = []
    previous_epoch_ns: int | None = None
    timestamp_removed_chars = 0
    ansi_removed_chars = 0
    newline_removed_chars = 0
    newline_counts = Counter()
    canonical_cursor = 0

    for line_index, line in enumerate(physical):
        content = str(line["content"])
        timestamp_match = _GITHUB_TIMESTAMP_RE.match(content)
        if timestamp_match:
            prefix = timestamp_match.group(0)
            payload = content[timestamp_match.end() :]
            epoch_ns = _timestamp_epoch_ns(timestamp_match)
            delta_ns = (
                None
                if epoch_ns is None or previous_epoch_ns is None
                else epoch_ns - previous_epoch_ns
            )
            timestamps.append(
                {
                    "line_index": line_index,
                    "timestamp": timestamp_match.group("timestamp"),
                    "separator": timestamp_match.group("separator"),
                    "prefix": prefix,
                    "prefix_char_count": len(prefix),
                    "epoch_ns": epoch_ns,
                    "delta_ns": delta_ns,
                    "canonical_char_offset": canonical_cursor,
                }
            )
            if epoch_ns is not None:
                previous_epoch_ns = epoch_ns
            timestamp_removed_chars += len(prefix)
        else:
            payload = content

        stripped, entries, sequences = _strip_ansi(
            payload,
            line_index=line_index,
            canonical_line_start=canonical_cursor,
        )
        ansi_entries.extend(entries)
        ansi_sequences.extend(sequences)
        ansi_removed_chars += sum(len(sequence) for sequence in sequences)
        terminator = str(line["terminator"])
        if terminator:
            newline_counts[
                "crlf" if terminator == "\r\n" else "cr" if terminator == "\r" else "lf"
            ] += 1
            normalized_terminator = "\n"
            newline_removed_chars += len(terminator) - 1
        else:
            newline_counts["none"] += 1
            normalized_terminator = ""
        canonical_line = stripped + normalized_terminator
        canonical_parts.append(canonical_line)
        canonical_cursor += len(canonical_line)

    pre_redaction = "".join(canonical_parts)
    canonical_text, secret_redactions = _redact_secrets(pre_redaction)
    scrubbed_ansi = [_redact_secrets(sequence)[0] for sequence in ansi_sequences]
    timestamp_deltas = [
        item["delta_ns"] for item in timestamps if item["delta_ns"] is not None
    ]
    timestamp_sequence_receipt = _bounded_sequence_receipt(timestamps)
    ansi_occurrence_receipt = _bounded_sequence_receipt(ansi_entries)
    secret_receipt = _bounded_sequence_receipt(secret_redactions)
    secret_kind_counts = Counter(
        str(item["kind"]) for item in secret_redactions
    )
    receipt = {
        "canonicalization_schema": CANONICALIZATION_SCHEMA,
        "operations": [
            "decode_utf8_with_accounted_replacement",
            "split_midstream_bom_before_github_timestamp",
            "strip_github_line_timestamp_prefix",
            "strip_and_digest_ansi_escape_sequences",
            "normalize_crlf_and_cr_to_lf",
            "same_length_secret_redaction",
        ],
        "decoded_char_count": len(decoded_text),
        "record_boundary_anomalies": boundary_entries,
        "record_boundary_removed_char_count": boundary_removed_chars,
        "physical_line_count": len(physical),
        "timestamp_prefixes": {
            "count": len(timestamps),
            "removed_char_count": timestamp_removed_chars,
            "ordered_prefix_sha256": _framed_digest(
                [str(item["prefix"]) for item in timestamps]
            ),
            "ordered_timestamp_and_delta_sha256": timestamp_sequence_receipt[
                "ordered_sha256"
            ],
            "first": timestamps[0] if timestamps else None,
            "last": timestamps[-1] if timestamps else None,
            "delta_ns": {
                "count": len(timestamp_deltas),
                "minimum": min(timestamp_deltas) if timestamp_deltas else None,
                "maximum": max(timestamp_deltas) if timestamp_deltas else None,
                "negative_count": sum(value < 0 for value in timestamp_deltas),
            },
            "sequence_samples": timestamp_sequence_receipt["samples"],
            "sequence_sample_limit": timestamp_sequence_receipt["sample_limit"],
            "sequence_omitted_count": timestamp_sequence_receipt["omitted_count"],
        },
        "ansi": {
            "sequence_count": len(ansi_entries),
            "removed_char_count": ansi_removed_chars,
            "ordered_redacted_sequence_sha256": _framed_digest(scrubbed_ansi),
            "digest_scope": (
                "length-framed ordered ANSI sequences after secret masking"
            ),
            "ordered_occurrence_sha256": ansi_occurrence_receipt[
                "ordered_sha256"
            ],
            "occurrence_samples": ansi_occurrence_receipt["samples"],
            "occurrence_sample_limit": ansi_occurrence_receipt["sample_limit"],
            "occurrence_omitted_count": ansi_occurrence_receipt["omitted_count"],
        },
        "newlines": {
            "source_counts": {
                "lf": newline_counts["lf"],
                "crlf": newline_counts["crlf"],
                "cr": newline_counts["cr"],
                "unterminated": newline_counts["none"],
            },
            "removed_char_count": newline_removed_chars,
            "canonical_newline_count": canonical_text.count("\n"),
            "final_newline": canonical_text.endswith("\n"),
        },
        "secrets": {
            "preexisting_github_masks_preserved": pre_redaction.count("***"),
            "redaction_count": len(secret_redactions),
            "redacted_char_count": sum(
                int(item["replacement_char_count"])
                for item in secret_redactions
            ),
            "kind_counts": dict(sorted(secret_kind_counts.items())),
            "ordered_redaction_sha256": secret_receipt["ordered_sha256"],
            "redaction_samples": secret_receipt["samples"],
            "redaction_sample_limit": secret_receipt["sample_limit"],
            "redaction_omitted_count": secret_receipt["omitted_count"],
            "raw_secret_values_or_hashes_retained": False,
        },
        "canonical_char_count": len(canonical_text),
        "canonical_sha256": _sha256_text(canonical_text),
    }
    receipt["accounting"] = {
        "equation": (
            "canonical_char_count = decoded_char_count"
            " - record_boundary_removed_char_count"
            " - timestamp_removed_char_count"
            " - ansi_removed_char_count"
            " - newline_removed_char_count"
        ),
        "expected_canonical_char_count": (
            len(decoded_text)
            - boundary_removed_chars
            - timestamp_removed_chars
            - ansi_removed_chars
            - newline_removed_chars
        ),
        "actual_canonical_char_count": len(canonical_text),
        "character_count_conserved": (
            len(decoded_text)
            - boundary_removed_chars
            - timestamp_removed_chars
            - ansi_removed_chars
            - newline_removed_chars
            == len(canonical_text)
        ),
        "secret_redaction_length_preserving": True,
    }
    return canonical_text, secret_redactions, receipt, boundary_entries


def canonicalize_ci_log(
    raw_log: bytes | bytearray | memoryview | str,
    metadata: Mapping[str, Any] | None = None,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> dict[str, Any]:
    """Canonicalize one GitHub Actions job log and extract its sidecar.

    ``chunks[*].char_start`` / ``char_end`` partition ``canonical_text``.
    Chunk ``text`` and ``sha256`` use the length-aligned ``dedup_text`` for
    training deduplication; ``canonical_text`` and ``canonical_sha256`` retain
    the exact redacted canonical slice for conservation checks.
    """

    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if not isinstance(max_chunk_chars, int) or isinstance(max_chunk_chars, bool):
        raise TypeError("max_chunk_chars must be an integer")
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be positive")

    decoded_text, raw_receipt = _decode_raw_log(raw_log)
    canonical_text, _secret_redactions, canonicalization, _boundaries = (
        _canonicalize_payload(decoded_text)
    )
    dedup_text, dedup_ledger = _deduplicate_volatiles(canonical_text)
    lines = _canonical_lines(canonical_text)
    provenance, metadata_redactions = _extract_provenance(metadata)
    system_observations, observed_system = _extract_system_metadata(lines)
    system_log = bool(
        observed_system.get("requested_labels")
        or observed_system.get("workflow_definition")
    ) and not any(_section_boundary(str(line["content"])) for line in lines)
    sections = _build_sections(
        canonical_text,
        dedup_text,
        lines,
        provenance,
        system_log=system_log,
    )
    chunks = _build_chunks(
        canonical_text,
        dedup_text,
        lines,
        sections,
        max_chunk_chars=max_chunk_chars,
    )

    builder = _EntityBuilder(canonical_text)
    shell_dialects, commands, _section_shell = _extract_shells_and_commands(
        canonical_text, lines, sections, builder
    )
    sql_dialects = _extract_sql_dialects(commands, builder)
    build_systems, toolchains = _extract_build_systems_and_toolchains(
        lines, sections, builder
    )
    paths, languages = _extract_paths(
        lines, sections, builder, commands
    )
    targets = _extract_targets(commands, paths, builder)
    build_actions, raw_build_actions = _extract_build_actions(
        lines,
        sections,
        chunks,
        paths,
        provenance,
        builder,
    )
    tests, test_summaries = _extract_tests(
        lines, sections, builder, commands
    )
    diagnostics = _extract_diagnostics(
        lines, sections, builder, commands, targets
    )
    raw_entities, raw_edges, id_for_temp = builder.finish()
    _attach_chunk_semantic_rle(chunks, raw_entities)
    training_accounting = _attach_chunk_training_sidecars(
        chunks,
        raw_entities,
        raw_edges,
        commands=_replace_entity_refs(commands, id_for_temp),
        build_actions=_replace_entity_refs(raw_build_actions, id_for_temp),
        tests=_replace_entity_refs(tests, id_for_temp),
        diagnostics=_replace_entity_refs(diagnostics, id_for_temp),
    )
    for chunk in chunks:
        training_sidecars = chunk["training_sidecars"]
        if _structure_contains_secret(training_sidecars):
            raise ValueError(
                "chunk training sidecars retained a secret-like value"
            )

    classifications: dict[str, Any] = {
        "shell_dialects": shell_dialects,
        "sql_dialects": sql_dialects,
        "commands": commands,
        "build_systems": build_systems,
        "toolchains": toolchains,
        "platform": _build_platform(provenance, observed_system),
        "build_targets": targets,
        "build_actions": build_actions,
        "paths": paths,
        "languages": languages,
        "tests": tests,
        "test_summaries": test_summaries,
        "diagnostics": diagnostics,
        "system_metadata": system_observations,
    }
    classifications = _replace_entity_refs(classifications, id_for_temp)
    entities, edges, old_to_compact_entity, graph_accounting = (
        _compact_entities_and_edges(raw_entities, raw_edges)
    )
    classifications = _remap_entity_ids(
        classifications, old_to_compact_entity
    )
    classifications, classification_accounting = _compact_classifications(
        classifications
    )

    canonical_chunk_join = "".join(
        str(chunk["canonical_text"]) for chunk in chunks
    )
    dedup_chunk_join = "".join(str(chunk["text"]) for chunk in chunks)
    section_join = "".join(str(section["text"]) for section in sections)
    section_dedup_join = "".join(
        str(section["dedup_text"]) for section in sections
    )
    semantic_rle_covers_exactly = all(
        all(
            (
                (not spans and int(chunk["char_end"]) == int(chunk["char_start"]))
                or (
                    bool(spans)
                    and int(spans[0]["start_char"]) == 0
                    and int(spans[-1]["end_char"])
                    == int(chunk["char_end"]) - int(chunk["char_start"])
                    and all(
                        int(left["end_char"]) == int(right["start_char"])
                        for left, right in zip(spans, spans[1:])
                    )
                )
            )
            for spans in (chunk["role_spans"], chunk["domain_spans"])
        )
        for chunk in chunks
    )
    conservation = {
        "canonical_char_count": len(canonical_text),
        "dedup_char_count": len(dedup_text),
        "canonical_line_count": len(lines),
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "section_canonical_chars_cover_exactly_once": (
            section_join == canonical_text
        ),
        "section_dedup_chars_cover_exactly_once": (
            section_dedup_join == dedup_text
        ),
        "chunk_canonical_chars_cover_exactly_once": (
            canonical_chunk_join == canonical_text
        ),
        "chunk_dedup_chars_cover_exactly_once": dedup_chunk_join == dedup_text,
        "chunk_spans_contiguous": all(
            int(chunk["char_start"])
            == (
                0 if index == 0 else int(chunks[index - 1]["char_end"])
            )
            for index, chunk in enumerate(chunks)
        )
        and (not chunks or int(chunks[-1]["char_end"]) == len(canonical_text)),
        "chunk_semantic_rle_covers_every_char_once": semantic_rle_covers_exactly,
        "canonical_sha256": _sha256_text(canonical_text),
        "dedup_sha256": _sha256_text(dedup_text),
    }
    if not all(
        conservation[key]
        for key in (
            "section_canonical_chars_cover_exactly_once",
            "section_dedup_chars_cover_exactly_once",
            "chunk_canonical_chars_cover_exactly_once",
            "chunk_dedup_chars_cover_exactly_once",
            "chunk_spans_contiguous",
            "chunk_semantic_rle_covers_every_char_once",
        )
    ):
        raise AssertionError("internal CI log section/chunk conservation failure")

    sidecar: dict[str, Any] = {
        "schema": SIDECAR_SCHEMA,
        "canonicalization_schema": CANONICALIZATION_SCHEMA,
        "deduplication_schema": DEDUPLICATION_SCHEMA,
        "raw": raw_receipt,
        "canonicalization": canonicalization,
        "deduplication": {
            "schema": DEDUPLICATION_SCHEMA,
            "rules": [rule[0] for rule in _DEDUP_RULES],
            "substitution_count": len(dedup_ledger),
            "substitutions": dedup_ledger,
            "length_preserving": True,
            "sha256": _sha256_text(dedup_text),
        },
        "security": {
            "canonical_secret_redactions": canonicalization["secrets"],
            "metadata_secret_redaction_count": len(metadata_redactions),
            "metadata_secret_redaction_samples": _bounded_sequence_receipt(
                metadata_redactions
            )["samples"],
            "metadata_secret_redaction_omitted_count": max(
                0, len(metadata_redactions) - MAX_AUDIT_SAMPLES
            ),
            "metadata_secret_redactions_sha256": _sequence_digest(
                metadata_redactions
            ),
            "retains_only_whole_raw_log_hash": True,
        },
        "provenance": provenance,
        "classifications": classifications,
        "entities": entities,
        "edges": edges,
        "evidence_accounting": {
            "graph": graph_accounting,
            "classifications": classification_accounting,
            "training_sidecars": training_accounting,
        },
        "section_index": _compact_section_index(sections),
        "chunk_index": [
            {
                key: value
                for key, value in chunk.items()
                if key
                not in {
                    "text",
                    "canonical_text",
                    "role_spans",
                    "domain_spans",
                    "training_sidecars",
                }
            }
            for chunk in chunks
        ],
        "conservation": conservation,
    }
    sidecar["sidecar_sha256"] = _sha256_text(stable_json_dumps(sidecar))

    # Defense in depth: selected metadata is sanitized before it reaches the
    # sidecar.  This invariant catches a future field addition that forgets to
    # follow the same rule.
    if _structure_contains_secret(sidecar):
        raise ValueError("sidecar serialization retained a secret-like value")

    return {
        "canonical_text": canonical_text,
        "dedup_text": dedup_text,
        "sections": sections,
        "chunks": chunks,
        "sidecar": sidecar,
    }


def extract_ci_log_sidecar(
    raw_log: bytes | bytearray | memoryview | str,
    metadata: Mapping[str, Any] | None = None,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> dict[str, Any]:
    """Return only the rich sidecar for callers storing content separately."""

    return canonicalize_ci_log(
        raw_log,
        metadata,
        max_chunk_chars=max_chunk_chars,
    )["sidecar"]


# Descriptive aliases for ingestion callers; all execute the same pure parser.
canonicalize_job_log = canonicalize_ci_log
extract_ci_log_sidecars = canonicalize_ci_log


__all__ = [
    "CANONICALIZATION_SCHEMA",
    "BUILD_ACTION_NORMALIZATION_SCHEMA",
    "DEDUPLICATION_SCHEMA",
    "DEFAULT_MAX_CHUNK_CHARS",
    "DOMAIN_IDS",
    "EDGE_IDS",
    "ROLE_IDS",
    "SIDECAR_SCHEMA",
    "TRAINING_SIDECAR_SCHEMA",
    "canonicalize_ci_log",
    "canonicalize_job_log",
    "extract_ci_log_sidecar",
    "extract_ci_log_sidecars",
    "stable_json_dumps",
]
