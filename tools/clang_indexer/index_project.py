#!/usr/bin/env python3
# ruff: noqa: E402
"""
Clang-based cross-file dependency indexer for C++ training data preparation.

Uses libclang to parse C++ translation units with full semantic analysis,
building a cross-file call graph and generating bottom-up training documents.

Architecture:
  1. Walk project directory, find all .cpp/.cc/.cxx/.c files
  2. Parse each with libclang (optionally using compile_commands.json)
  3. Extract functions, classes, and cross-file call references
  4. Build global call graph
  5. Topological sort: HAL/system → drivers → subsystems → API
  6. Generate 16K-token training documents with bottom-up dependency ordering

Usage:
  # With compile_commands.json (best quality):
  python index_project.py --project-dir /path/to/project --project-id owner/repo --output chunks.jsonl

  # Without build system (fallback mode):
  python index_project.py --project-dir /path/to/project --project-id owner/repo --output chunks.jsonl --no-compile-db

  # Process multiple projects in parallel:
  python index_project.py --projects-list projects.txt --output chunks.jsonl --workers 48
"""

import argparse
import codecs
import ctypes.util
import glob
import gzip
import hashlib
import importlib
import json
import os
import posixpath
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import warnings
from array import array

# Increase recursion limit for deeply nested ASTs (gcc-mirror, llvm-project, boost)
sys.setrecursionlimit(50000)
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    TextIO,
    TypeAlias,
    cast,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega.data.build_context import (
    BuildContextEvidenceError,
    detect_build_context,
    find_compile_commands_file,
    load_compile_commands_file,
    normalize_macos_sdk_path_argument,
    validate_macos_sdk_path,
)
from cppmega.data.language_info import detect_language_info
from cppmega.data.source_identity import source_identity, source_identity_for_path
from cppmega.symbol_identity import (
    EXTERNAL_PROVIDER_PROJECTS,
    SYMBOL_ID_MAX,  # noqa: F401 - re-exported for global index consumers
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SymbolIdentityError,
    SymbolIdentityRegistry,
    canonical_external_provider_file,
    canonical_external_usr_identity,
    canonical_usr_identity,
    compute_symbol_id,
    external_provider_project,
    is_repo_file_location_identity,
    parse_external_provider_file,
    parse_repo_file_location_identity,
    parse_usr_identity,
    require_project_identity,
)
from scripts.data.atomic_publish import atomic_output_file
from scripts.data.memory_guard import check_memory_limit, start_memory_guard

if __package__:
    from .source_quarantine import ProjectSourceQuarantine
else:
    _INDEXER_MODULE_ROOT = Path(__file__).resolve().parent
    if str(_INDEXER_MODULE_ROOT) not in sys.path:
        sys.path.insert(0, str(_INDEXER_MODULE_ROOT))
    from source_quarantine import ProjectSourceQuarantine

if TYPE_CHECKING:
    import clang.cindex as clang_cindex  # pyright: ignore[reportMissingImports]
    from clang.cindex import (
        Config as ClangConfig,  # pyright: ignore[reportMissingImports]
    )
    from clang.cindex import (  # pyright: ignore[reportMissingImports]
        Cursor,
        CursorKind,
        Index,
        TranslationUnit,
    )
else:
    clang_cindex = None
    ClangConfig = object
    Cursor = object
    CursorKind = object
    Index = object
    TranslationUnit = object


class _MissingCursorKind:
    def __getattr__(self, name: str) -> str:
        return f"<missing-clang-cursorkind:{name}>"


def _open_jsonl_output(
    path: Path,
    *,
    append: bool,
    compressed: bool,
) -> TextIO:
    mode = "at" if append else "wt"
    if compressed:
        return gzip.open(
            path,
            mode,
            compresslevel=1,
            encoding="utf-8",
            newline="\n",
        )
    return path.open(mode, encoding="utf-8", newline="\n")


class _MissingIndex:
    @staticmethod
    def create():
        raise ImportError(
            "libclang Python bindings not found. Install with: pip install libclang"
        )


class _ClangCIndexModule(Protocol):
    __file__: str


class _ClangConfig(Protocol):
    @staticmethod
    def set_library_file(filename: str) -> None: ...


class _IndexFactory(Protocol):
    @staticmethod
    def create() -> "Index": ...


class PlatformDetectFn(Protocol):
    def __call__(self, text: str) -> dict[str, object] | None: ...


DetectLanguageInfoFn = Callable[
    [str, str | None, dict[str, object] | None, list[str] | None, dict | None],
    dict[str, object] | None,
]


clang_cindex_module: _ClangCIndexModule | None
clang_config_cls: _ClangConfig
index_cls: _IndexFactory
_CLANG_IMPORT_ERROR: ImportError | None = None

try:
    clang_cindex_runtime = importlib.import_module("clang.cindex")
    clang_cindex_module = cast(_ClangCIndexModule, clang_cindex_runtime)
    clang_config_cls = cast(_ClangConfig, getattr(clang_cindex_runtime, "Config"))
    index_cls = cast(_IndexFactory, getattr(clang_cindex_runtime, "Index"))
    if not TYPE_CHECKING:
        clang_cindex = clang_cindex_runtime  # type: ignore[assignment]
        ClangConfig = getattr(clang_cindex_runtime, "Config")  # type: ignore[assignment]
        Cursor = getattr(clang_cindex_runtime, "Cursor")  # type: ignore[assignment]
        CursorKind = getattr(clang_cindex_runtime, "CursorKind")  # type: ignore[assignment]
        Index = getattr(clang_cindex_runtime, "Index")  # type: ignore[assignment]
        TranslationUnit = getattr(clang_cindex_runtime, "TranslationUnit")  # type: ignore[assignment]
except ImportError as _clang_import_err:
    _CLANG_IMPORT_ERROR = _clang_import_err
    clang_cindex_module = None
    clang_config_cls = cast(_ClangConfig, object)
    index_cls = cast(_IndexFactory, _MissingIndex)
    if not TYPE_CHECKING:
        CursorKind = _MissingCursorKind()  # type: ignore[assignment]
        Index = _MissingIndex  # type: ignore[assignment]


_LIBCLANG_CONFIGURED = False
_COMMON_LIBCLANG_CANDIDATES = [
    "/usr/lib/llvm-21/lib/libclang-21.so.1",
    "/usr/lib/llvm-21/lib/libclang.so.1",
    "/usr/lib/llvm-20/lib/libclang-20.so.1",
    "/usr/lib/llvm-20/lib/libclang.so.1",
    "/usr/lib/llvm-19/lib/libclang-19.so.1",
    "/usr/lib/llvm-19/lib/libclang.so.1",
    "/usr/lib/llvm-18/lib/libclang-18.so.1",
    "/usr/lib/llvm-18/lib/libclang.so.1",
    "/lib/x86_64-linux-gnu/libclang-18.so.18",
    "/lib/x86_64-linux-gnu/libclang-18.so",
    "/lib/x86_64-linux-gnu/libclang.so.1",
    "/lib/x86_64-linux-gnu/libclang.so",
    "/usr/lib/libclang.so.1",
    "/usr/lib/libclang.so",
]


def _iter_bundled_libclang_candidates() -> Iterator[str]:
    if clang_cindex_module is None:
        return
    module_path = getattr(clang_cindex_module, "__file__", "")
    if not module_path:
        return
    native_dir = Path(module_path).resolve().parent / "native"
    if not native_dir.is_dir():
        return
    for pattern in ("libclang.so*", "libclang.dylib*", "libclang.dll*"):
        for candidate in sorted(native_dir.glob(pattern)):
            if candidate.is_file():
                yield str(candidate)


def _iter_libclang_candidates(explicit_path: str | None = None) -> Iterator[str]:
    seen: set[str] = set()

    def add(value: str | None) -> str | None:
        if not value:
            return None
        candidate = value.strip()
        if not candidate or candidate in seen:
            return None
        seen.add(candidate)
        return candidate

    candidate = add(explicit_path)
    if candidate:
        yield candidate
    for env_name in (
        "NANOCHAT_LIBCLANG_PATH",
        "LIBCLANG_PATH",
        "LIBCLANG_FILE",
        "CLANG_LIBRARY_FILE",
    ):
        candidate = add(os.environ.get(env_name))
        if candidate:
            yield candidate
    for candidate in _iter_bundled_libclang_candidates():
        normalized = add(candidate)
        if normalized:
            yield normalized
    for candidate in _COMMON_LIBCLANG_CANDIDATES:
        if os.path.exists(candidate):
            normalized = add(candidate)
            if normalized:
                yield normalized
    detected = add(ctypes.util.find_library("clang"))
    if detected:
        yield detected
    for pattern in (
        "/usr/lib*/**/libclang*.so*",
        "/usr/local/lib*/**/libclang*.so*",
        "/lib*/**/libclang*.so*",
    ):
        for candidate in glob.iglob(pattern, recursive=True):
            if os.path.isfile(candidate):
                normalized = add(candidate)
                if normalized:
                    yield normalized


def _configure_libclang(libclang_path: str | None = None) -> str | None:
    global _LIBCLANG_CONFIGURED
    if _LIBCLANG_CONFIGURED:
        return os.environ.get("NANOCHAT_LIBCLANG_PATH") or libclang_path
    if _CLANG_IMPORT_ERROR is not None:
        raise ImportError(
            "libclang Python bindings not found. Install with: pip install libclang"
        ) from _CLANG_IMPORT_ERROR

    last_error: Exception | None = None
    for candidate in _iter_libclang_candidates(libclang_path):
        try:
            clang_config_cls.set_library_file(candidate)
            index_cls.create()
        except SymbolIdentityError:
            raise
        except Exception as exc:  # pragma: no cover - runtime-specific
            last_error = exc
            continue
        os.environ["NANOCHAT_LIBCLANG_PATH"] = candidate
        _LIBCLANG_CONFIGURED = True
        return candidate

    try:
        index_cls.create()
    except SymbolIdentityError:
        raise
    except Exception as exc:
        last_error = exc
    else:
        _LIBCLANG_CONFIGURED = True
        return None

    message = (
        "Unable to load libclang. "
        "Set --libclang-path or NANOCHAT_LIBCLANG_PATH to an absolute libclang.so path."
    )
    if last_error is not None:
        raise RuntimeError(f"{message} Last error: {last_error}") from last_error
    raise RuntimeError(message)

# A doc PART. Historically a 5-tuple (text, kind, dep_level, name, qname_or_none).
# Cross-repo base-lib pulls append a 6th element: dep_source — a provenance
# marker like 'crosslib:<repo>' (None / absent for normal local parts). Macro
# chunks may append a 7th element with scanner provenance so domain route edges
# point to the precise conditional/redefinition/include-order macro nodes rather
# than being re-derived from assembled text. Symbol-bearing chunks may append an
# 8th element with canonical identity metadata and a 9th source-path field.
# Builders accept all historical forms.
PartInfo: TypeAlias = tuple[str, int, int, str, str | None] | tuple[
    str, int, int, str, str | None, str | None
] | tuple[
    str, int, int, str, str | None, str | None, dict[str, object] | None
] | tuple[
    str,
    int,
    int,
    str,
    str | None,
    str | None,
    dict[str, object] | None,
    str,
] | tuple[
    str,
    int,
    int,
    str,
    str | None,
    str | None,
    dict[str, object] | None,
    dict[str, object] | None,
] | tuple[
    str,
    int,
    int,
    str,
    str | None,
    str | None,
    dict[str, object] | None,
    dict[str, object] | None,
    str,
]
SymbolReference: TypeAlias = dict[str, object]
ExternalReferenceOmissionKey: TypeAlias = tuple[str, str, str, str]
ExternalReferenceOmissions: TypeAlias = dict[ExternalReferenceOmissionKey, int]


def _part_dep_source(part: tuple) -> str | None:
    """dep_source provenance of a doc part (None for normal local parts)."""
    value = part[5] if len(part) >= 6 else None
    return value if isinstance(value, str) else None


def _part_macro_provenance(part: tuple) -> dict[str, object] | None:
    """Scanner-derived macro provenance of a doc part, when present."""
    value = part[6] if len(part) >= 7 else None
    return value if isinstance(value, dict) else None


def _symbol_part_metadata(
    symbol_key: str | None,
    *,
    qname: str | None = None,
    symbol_id: int | None = None,
    canonical_signature: str | None = None,
    usr: str | None = None,
    kind: str | None = None,
    provider: str | None = None,
    include_provenance: str | None = None,
) -> dict[str, object] | None:
    if not symbol_key:
        return None
    return {
        "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
        "symbol_key": symbol_key,
        "symbol_id": int(symbol_id if symbol_id is not None else _compute_symbol_id(symbol_key)),
        "qname": qname or "",
        "canonical_signature": canonical_signature or "",
        "usr": usr or "",
        "kind": kind or "",
        "provider": provider or "",
        "include_provenance": include_provenance or "",
    }


def _function_part(
    func: "FunctionDef",
    *,
    kind: int = 2,
    dep_source: str | None = None,
) -> PartInfo:
    return (
        func.text,
        kind,
        func.dep_level,
        func.name,
        func.qualified_name,
        dep_source,
        None,
        _symbol_part_metadata(
            func.symbol_key,
            qname=func.qualified_name,
            symbol_id=func.symbol_id,
            canonical_signature=func.canonical_signature,
            usr=func.usr,
            kind=func.symbol_kind,
        ),
        func.file,
    )


def _typedef_part(td: "TypeDef") -> PartInfo:
    return (
        td.text,
        td.kind,
        0,
        td.name,
        td.qualified_name,
        None,
        None,
        _symbol_part_metadata(
            td.symbol_key,
            qname=td.qualified_name,
            symbol_id=td.symbol_id,
            canonical_signature=td.canonical_signature,
            usr=td.usr,
            kind=td.symbol_kind,
        ),
        td.file,
    )


def _macro_part(macro: "MacroDef") -> PartInfo:
    return (
        macro.text,
        MACRO_KIND,
        0,
        macro.name,
        macro.name,
        None,
        _macro_part_metadata(macro),
        _symbol_part_metadata(
            macro.symbol_key,
            qname=macro.name,
            symbol_id=macro.symbol_id,
            canonical_signature="(" + ",".join(macro.params) + ")",
            kind="MACRO_DEFINITION",
        ),
        macro.file,
    )


def _global_symbol_part(record: dict[str, object], dep_level: int) -> PartInfo:
    qname = str(record.get("qname") or "")
    return (
        str(record.get("text") or ""),
        2,
        dep_level,
        qname.split("::")[-1],
        qname,
        f"crosslib:{record.get('base_repo')}",
        None,
        _symbol_part_metadata(
            str(record.get("symbol_key") or ""),
            qname=qname,
            canonical_signature=str(record.get("canonical_signature") or ""),
            usr=str(record.get("usr") or ""),
            kind=str(record.get("symbol_kind") or ""),
            provider=str(record.get("provider") or ""),
            include_provenance=str(record.get("include_provenance") or ""),
        ),
        str(
            record.get("file")
            or f"crosslib://{record.get('base_repo')}/{qname}"
        ),
    )


def _part_symbol_metadata(part: tuple) -> dict[str, object] | None:
    value = part[7] if len(part) >= 8 else None
    return value if isinstance(value, dict) else None


def _part_symbol_key(part: tuple) -> str | None:
    metadata = _part_symbol_metadata(part)
    if metadata is None:
        return None
    value = metadata.get("symbol_key")
    return value if isinstance(value, str) and value else None


def _document_symbol_identities(
    parts_info: Sequence[tuple],
    index: "ProjectIndex",
    *symbol_id_sequences: Iterable[int],
    source: str,
) -> list[dict[str, object]]:
    """Return complete, collision-checked ID/key claims used by one document."""

    registry = SymbolIdentityRegistry()
    part_symbol_ids: set[int] = set()
    for part_index, part in enumerate(parts_info):
        metadata = _part_symbol_metadata(part)
        if metadata is None:
            continue
        symbol_key = metadata.get("symbol_key")
        symbol_id = metadata.get("symbol_id")
        if not isinstance(symbol_key, str) or symbol_id is None:
            raise SymbolIdentityError(
                f"{source}: part {part_index} has incomplete canonical symbol metadata"
            )
        registry.register(
            symbol_key,
            symbol_id=int(symbol_id),
            source=f"{source}:part[{part_index}]",
        )
        part_symbol_ids.add(int(symbol_id))

    used_ids = part_symbol_ids | {
        int(symbol_id)
        for sequence in symbol_id_sequences
        for symbol_id in sequence
        if int(symbol_id) != 0
    }
    for symbol_id in sorted(used_ids):
        if symbol_id in registry.keys_by_id:
            continue
        symbol_key = index.symbol_id_keys.get(symbol_id)
        if symbol_key is None:
            raise SymbolIdentityError(
                f"{source}: semantic symbol ID {symbol_id} has no canonical identity key"
            )
        registry.register(
            symbol_key,
            symbol_id=symbol_id,
            source=f"{source}:ProjectIndex",
        )
    registry.require_ids(used_ids, source=source)
    return registry.records(used_ids)


def _semantic_identity_records_for_arrays(
    semantic_metadata: dict[str, object],
    *semantic_arrays: Iterable[int],
    source: str,
) -> list[dict[str, object]]:
    """Select the canonical claims needed by sliced semantic arrays."""

    registry = SymbolIdentityRegistry()
    registry.register_records(
        semantic_metadata.get(SYMBOL_IDENTITIES_COLUMN, []),
        source=source,
    )
    used_ids = {
        int(symbol_id)
        for values in semantic_arrays
        for symbol_id in values
        if int(symbol_id) != 0
    }
    registry.require_ids(used_ids, source=source)
    return registry.records(used_ids)


def _part_source_path(
    part: tuple,
    index: "ProjectIndex",
    fallback: str,
) -> str:
    explicit = (
        part[8]
        if len(part) >= 9
        else part[7]
        if len(part) >= 8 and isinstance(part[7], str)
        else None
    )
    if isinstance(explicit, str) and explicit:
        return explicit
    metadata = _part_macro_provenance(part)
    if metadata is not None and isinstance(metadata.get("file"), str):
        return cast(str, metadata["file"])
    symbol_key = _part_symbol_key(part)
    qname = part[4] if len(part) >= 5 else None
    if symbol_key or isinstance(qname, str):
        function_key = index.resolve_function_key(symbol_key or qname)
        function = index.functions.get(function_key) if function_key else None
        if function is not None:
            return function.file
        type_key = index.resolve_type_key(symbol_key or qname)
        typedef = index.typedefs.get(type_key) if type_key else None
        if typedef is not None:
            return typedef.file
    return fallback


def _stable_repo_id(repo_name: str) -> str:
    return hashlib.sha1(repo_name.encode("utf-8")).hexdigest()[:16]


def _stable_filepath_id(repo_name: str, filepath: str) -> str:
    return hashlib.sha1(f"{repo_name}\0{filepath}".encode("utf-8")).hexdigest()[:16]


def _canonical_project_path(filepath: str) -> str:
    if "://" in filepath:
        return filepath
    normalized = os.path.normpath(filepath).replace(os.sep, "/")
    return normalized.removeprefix("./")


def _project_path_provenance(project_id: str, filepath: str) -> dict[str, str]:
    stable_project_id = require_project_identity(
        project_id,
        source=f"source provenance for {filepath}",
    )
    canonical_filepath = _canonical_project_path(filepath)
    return {
        "repo": stable_project_id,
        "filepath": canonical_filepath,
        "repo_stable_id": _stable_repo_id(stable_project_id),
        "filepath_stable_id": _stable_filepath_id(
            stable_project_id,
            canonical_filepath,
        ),
    }


def _source_identity_for_project_path(
    filepath: str,
    *,
    project_id: str | None,
):
    if project_id is None:
        return source_identity_for_path(filepath)
    return source_identity(_project_path_provenance(project_id, filepath))


# C++ source file extensions
CPP_EXTENSIONS = {'.cpp', '.cc', '.cxx', '.c', '.c++', '.cp'}
HEADER_EXTENSIONS = {
    '.h',
    '.hpp',
    '.hxx',
    '.hh',
    '.h++',
    '.inl',
    '.inc',
    '.ipp',
    '.tpp',
    '.txx',
}
# Files we feed to libclang for indexing. Headers are included so that struct/
# class/enum/typedef DEFINITIONS (which live in headers) enter the index as
# type-def chunks, enabling function->type-def ``type_edges``. Headers are parsed
# as standalone TUs; libclang resolves the definitions they contain.
INDEX_EXTENSIONS = CPP_EXTENSIONS | HEADER_EXTENSIONS


def _is_header_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in HEADER_EXTENSIONS

# --------------------------------------------------------------------------- #
# Build / compilation file discovery (ADDITIVE; C/C++ path unchanged).
#
# Build files are ingested as their OWN 'build' doc type (distinct from code):
# full tokenized text + a 'build' structure kind + a build-system language tag +
# a platform sidecar where derivable. They carry NO call/type/symbol graph
# (build files are not C++), exactly like commit docs on the code channels.
#
# BUILD_NAME_KINDS maps an EXACT basename -> build-system tag (the language tag).
# BUILD_EXT_KINDS maps a lowercase extension -> build-system tag. A file is a
# build file if its basename hits BUILD_NAME_KINDS OR its extension hits
# BUILD_EXT_KINDS (basename wins on conflict).
# --------------------------------------------------------------------------- #
BUILD_NAME_KINDS: dict[str, str] = {
    # CMake
    "CMakeLists.txt": "cmake",
    # Make
    "Makefile": "make",
    "GNUmakefile": "make",
    "makefile": "make",
    # Autotools
    "configure": "configure",
    "configure.ac": "autoconf",
    "configure.in": "autoconf",
    "Makefile.am": "automake",
    "Makefile.in": "automake",
    # Bazel
    "BUILD": "bazel",
    "BUILD.bazel": "bazel",
    "WORKSPACE": "bazel",
    "WORKSPACE.bazel": "bazel",
    "MODULE.bazel": "bazel",
    # GN
    "BUILD.gn": "gn",
    # SCons
    "SConstruct": "scons",
    "SConscript": "scons",
    # xmake
    "xmake.lua": "xmake",
    # Meson
    "meson.build": "meson",
    "meson_options.txt": "meson",
    # Ninja
    "build.ninja": "ninja",
    # compile_commands
    "compile_commands.json": "compile_commands",
    # Conan / vcpkg
    "conanfile.txt": "conan",
    "conanfile.py": "conan",
    "vcpkg.json": "vcpkg",
    # Docker (build env)
    "Dockerfile": "dockerfile",
}
BUILD_EXT_KINDS: dict[str, str] = {
    # Assembly is source code emitted losslessly on the frozen broad code route.
    ".s": "assembly",
    ".asm": "assembly",
    ".cmake": "cmake",
    ".mk": "make",
    ".m4": "autoconf",
    ".bzl": "bazel",
    ".gn": "gn",
    ".gni": "gn",
    ".ninja": "ninja",
    ".vcxproj": "msvc",
    ".sln": "msvc",
}
# Hard cap for each emitted build/domain document. Legitimate larger text files
# are streamed into chunks at this size instead of aborting repository ingestion.
BUILD_FILE_SIZE_CAP = 500_000
SHELL_EXT_KINDS: dict[str, str] = {
    ".bash": "bash",
    ".ksh": "ksh",
    ".sh": "sh",
    ".zsh": "zsh",
    ".csh": "tcsh",
    ".tcsh": "tcsh",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".bat": "cmd",
    ".cmd": "cmd",
}

# Source structure ids.  The first 0-8 values are the historical code kinds used
# by build_enriched_doc; build/domain docs extended that with BUILD_KIND=9.  Header
# fragments are C++ code, but need their own small ids so the model can distinguish
# standalone header context and preprocessor macro bodies from ordinary function
# chunks.
BUILD_KIND = 9
HEADER_FRAGMENT_KIND = 10
MACRO_KIND = 11


def classify_build_file(fname: str) -> str | None:
    """Return the build-system tag for ``fname`` or None if it is not a build file.

    Exact-basename match (CMakeLists.txt, Makefile, BUILD, ...) takes priority
    over extension match (.cmake, .mk, .bzl, ...). conanfile.* is matched by the
    explicit names above; no silent wildcard guessing (RULE #1: ONE clear path).
    """
    if fname in BUILD_NAME_KINDS:
        return BUILD_NAME_KINDS[fname]
    ext = os.path.splitext(fname)[1].lower()
    if ext in BUILD_EXT_KINDS:
        return BUILD_EXT_KINDS[ext]
    return None


# System/stdlib function prefixes (skip for dependency tracking). UNCHANGED from
# the original: callees under these prefixes are NOT recorded as normal callees,
# so the local dep-resolution / call-edge path behaves EXACTLY as before whether
# or not cross-repo linking is enabled. (OFF behavior is byte-identical.)
SYSTEM_PREFIXES = (
    'std::', 'boost::', '__builtin', '__', 'operator', 'printf', 'fprintf',
    'sprintf', 'snprintf', 'scanf', 'malloc', 'calloc', 'realloc', 'free',
    'memcpy', 'memmove', 'memset', 'memcmp', 'strlen', 'strcpy', 'strcat',
    'strcmp', 'fopen', 'fclose', 'fread', 'fwrite', 'exit', 'abort',
    'assert', 'pthread_', 'EXPECT_', 'ASSERT_', 'TEST',
)

# Base-library NAMESPACE prefixes that ARE cross-linkable (present in the global
# symbol store built by scripts/crossrepo/build_global_symbol_index.py). A callee
# under one of these is filtered OUT of the normal `callees` list (above) but is
# captured SEPARATELY in FunctionDef.baselib_callees so the cross-repo linker can
# resolve it WITHOUT changing the normal callee/call-edge behavior. We do NOT
# include libc-style bare names here (memcpy/strlen/...) because those are not in
# the A1 selection and would only add noise; only the namespaced base libs.
CROSSLINKABLE_NS_PREFIXES = ('std::', 'boost::')
_STD_INLINE_NAMESPACE_SEGMENTS = ('__1', '__2', '__3', '__cxx11', '__ndk1')
_STD_INLINE_NAMESPACE_RE = re.compile(
    r"(?<![A-Za-z0-9_])std::(?:"
    + "|".join(map(re.escape, _STD_INLINE_NAMESPACE_SEGMENTS))
    + r")::"
)


def _normalize_inline_namespace_text(value: str) -> str:
    previous = None
    while value != previous:
        previous = value
        value = _STD_INLINE_NAMESPACE_RE.sub("std::", value)
    return value


def normalize_inline_namespace_qname(qname: str) -> str:
    """Normalize standard-library inline namespaces in qualified names.

    libc++/libstdc++ commonly surface symbols as ``std::__1::...`` or
    ``std::__cxx11::...`` through libclang while the global base-lib index stores
    the canonical public spelling.  Keep normalization narrow to ``std::`` so a
    project namespace named ``__1`` is not rewritten.
    """
    if not qname.startswith('std::'):
        return qname
    return _normalize_inline_namespace_text(qname)


def _normalize_signature_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _project_identity_path_prefixes(project_dir: str | None) -> tuple[str, ...]:
    """Return native and cross-platform spellings of the checkout prefix."""

    if not project_dir:
        return ()
    roots = {
        os.path.abspath(str(project_dir)),
        os.path.realpath(str(project_dir)),
    }
    prefixes: set[str] = set()
    for root in roots:
        stripped = root.rstrip("/\\")
        for spelling in {
            stripped,
            stripped.replace("\\", "/"),
            stripped.replace("/", "\\"),
        }:
            prefixes.add(spelling.rstrip("/\\") + "/")
            prefixes.add(spelling.rstrip("/\\") + "\\")
    return tuple(sorted(prefixes, key=len, reverse=True))


def _canonicalize_clang_identity_text(
    value: str | None,
    project_dir: str | None,
) -> str:
    """Remove checkout-specific prefixes from clang-generated identity text."""

    text = str(value or "")
    for prefix in _project_identity_path_prefixes(project_dir):
        text = text.replace(prefix, "")
    return _normalize_signature_text(text)


_PROVIDER_PATH_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "libc++",
        ("/libc++/include/", "/libcxx/include/", "/include/c++/v1/"),
    ),
    (
        "libstdc++",
        ("/libstdc++-v3/include/", "/libstdc++-v3/libsupc++/", "/include/c++/"),
    ),
    ("msvc-stl", ("/stl/inc/", "/microsoft visual studio/")),
    ("boost", ("/boost/",)),
)
_PROVIDER_PROJECT_IDENTITIES = {
    **EXTERNAL_PROVIDER_PROJECTS,
}


def symbol_provider_provenance(file_path: str | None) -> tuple[str, str]:
    """Return authoritative implementation provider and provider-relative include."""

    if not file_path:
        return "", ""
    raw_path = str(file_path).replace("\\", "/").strip("/")
    if raw_path.startswith("@provider/"):
        try:
            return parse_external_provider_file(
                raw_path,
                source="symbol provider provenance",
            )
        except SymbolIdentityError:
            return "", ""
    normalized = "/" + raw_path
    lowered = normalized.lower()
    for provider, markers in _PROVIDER_PATH_MARKERS:
        for marker in markers:
            offset = lowered.rfind(marker) if provider == "boost" else lowered.find(marker)
            if offset < 0:
                continue
            include_path = normalized[offset + len(marker) :]
            if provider == "libstdc++" and marker == "/include/c++/":
                parts = include_path.split("/", 1)
                if len(parts) == 2 and re.fullmatch(r"[0-9.]+", parts[0]):
                    include_path = parts[1]
            elif provider == "msvc-stl" and marker == "/microsoft visual studio/":
                include_marker = "/include/"
                include_offset = include_path.lower().rfind(include_marker)
                if include_offset < 0:
                    return provider, ""
                include_path = include_path[include_offset + len(include_marker) :]
            elif provider == "boost":
                include_path = "boost/" + include_path
            try:
                canonical = canonical_external_provider_file(
                    provider,
                    include_path.strip("/"),
                    source="symbol provider provenance",
                )
                return parse_external_provider_file(
                    canonical,
                    source="symbol provider provenance",
                )
            except SymbolIdentityError:
                return provider, ""
    return "", ""


def normalize_qualified_name(qname: str) -> str:
    """Return the stable qualified-name spelling used by fallback identity."""

    normalized = _normalize_signature_text(qname)
    normalized = re.sub(r"\s*::\s*", "::", normalized)
    return normalize_inline_namespace_qname(normalized)


def _cursor_kind(cursor: Cursor):
    """Return a binding-known CursorKind, or None for newer opaque kinds.

    Some libclang releases emit cursor IDs that their bundled Python cindex
    table does not register.  Accessing ``cursor.kind`` then raises ValueError;
    those cursors are non-semantic attributes for our purposes and must not
    abort the translation unit.
    """
    try:
        return cursor.kind
    except ValueError:
        return None


def _cursor_kind_name(cursor: Cursor) -> str:
    kind = _cursor_kind(cursor)
    if kind is None:
        raw_kind = getattr(cursor, "_kind_id", None)
        return f"UNKNOWN_CURSOR_{raw_kind}" if raw_kind is not None else "UNKNOWN_CURSOR"
    name = getattr(kind, "name", None)
    if isinstance(name, str):
        return name
    text = str(kind or "")
    return text.rsplit(".", 1)[-1]


def _cursor_usr(cursor: Cursor, *, project_dir: str | None = None) -> str:
    get_usr = getattr(cursor, "get_usr", None)
    if not callable(get_usr):
        return ""
    try:
        usr = str(get_usr() or "")
    except SymbolIdentityError:
        raise
    except Exception as exc:
        raise SymbolIdentityError("clang USR extraction failed") from exc
    # Clang can surface empty/placeholder USRs for unexposed/local constructs.
    # Anonymous declarations can also carry a checkout-absolute path and spaces
    # inside the USR. Such a value is not portable and is invalid under the
    # canonical USR contract, so route it through the scoped fallback identity.
    if (
        not usr
        or usr.startswith("<")
        or "invalid" in usr.lower()
        or usr != usr.strip()
        or "\x1f" in usr
        # Whitespace is legal semantic payload in Clang USRs for conversion
        # operators (for example ``operator int``).  Reject only control
        # characters here; checkout-bound anonymous USRs are rejected by the
        # explicit project-prefix check below.
        or any(ord(char) < 32 or ord(char) == 127 for char in usr)
        or any(
            prefix in usr
            for prefix in _project_identity_path_prefixes(project_dir)
        )
    ):
        return ""
    return usr


def _cursor_canonical_signature(cursor: Cursor) -> str:
    """Return a stable, clang-derived signature string for identity fallback."""
    try:
        return _cursor_canonical_signature_unchecked(cursor)
    except SymbolIdentityError:
        raise
    except Exception as exc:
        raise SymbolIdentityError(
            "clang canonical signature extraction failed"
        ) from exc


def _cursor_exception_specification_name(cursor: Cursor) -> str:
    """Return a stable exception-spec name across libclang/binding skew."""
    try:
        exception_kind = getattr(cursor, "exception_specification_kind", None)
    except ValueError as exc:
        raw_id: int | None = None
        if clang_cindex_module is not None and isinstance(cursor, Cursor):
            conf = getattr(clang_cindex_module, "conf", None)
            lib = getattr(conf, "lib", None)
            getter = getattr(lib, "clang_getCursorExceptionSpecificationType", None)
            if callable(getter):
                raw_id = int(getter(cursor))
        if raw_id is None:
            match = re.search(r"(-?\d+)\D*$", str(exc))
            if match is not None:
                raw_id = int(match.group(1))
        return f"UNKNOWN_{raw_id}" if raw_id is not None else "UNKNOWN"

    name = getattr(exception_kind, "name", "") or str(exception_kind or "")
    return name.rsplit(".", 1)[-1]


def _cursor_canonical_signature_unchecked(cursor: Cursor) -> str:
    pieces: list[str] = []
    display = _normalize_signature_text(getattr(cursor, "displayname", "") or "")
    if display:
        pieces.append(f"display={display}")
    cursor_type = getattr(cursor, "type", None)
    canonical_type = cursor_type
    get_canonical = getattr(cursor_type, "get_canonical", None)
    if callable(get_canonical):
        canonical_type = get_canonical()
    type_spelling = _normalize_signature_text(
        getattr(canonical_type, "spelling", "")
        or getattr(cursor_type, "spelling", "")
        or ""
    )
    if type_spelling:
        pieces.append(f"type={type_spelling}")
    result_type = getattr(cursor, "result_type", None)
    canonical_result = result_type
    get_canonical_result = getattr(result_type, "get_canonical", None)
    if callable(get_canonical_result):
        canonical_result = get_canonical_result()
    result_spelling = _normalize_signature_text(
        getattr(canonical_result, "spelling", "")
        or getattr(result_type, "spelling", "")
        or ""
    )
    if result_spelling:
        pieces.append(f"result={result_spelling}")
    arg_types: list[str] = []
    get_arguments = getattr(cursor, "get_arguments", None)
    if callable(get_arguments):
        for arg in get_arguments():
            arg_type = getattr(arg, "type", None)
            canonical_arg = arg_type
            get_canonical_arg = getattr(arg_type, "get_canonical", None)
            if callable(get_canonical_arg):
                canonical_arg = get_canonical_arg()
            arg_types.append(
                _normalize_signature_text(
                    getattr(canonical_arg, "spelling", "")
                    or getattr(arg_type, "spelling", "")
                    or ""
                )
            )
    if arg_types:
        pieces.append("args=(" + ",".join(arg_types) + ")")
    exception_name = _cursor_exception_specification_name(cursor)
    if pieces and exception_name:
        pieces.append(f"exception={exception_name.rsplit('.', 1)[-1]}")
    return "|".join(pieces)


def _identity_scope_key(
    *,
    project: str | None = None,
    file: str | None = None,
    line: int | None = None,
    column: int | None = None,
    force_file_scope: bool = False,
) -> str:
    parts: list[str] = []
    if project:
        parts.append(f"project={project}")
    if file and force_file_scope:
        parts.append(f"file={file}")
    if line is not None and line > 0 and force_file_scope:
        parts.append(f"line={int(line)}")
    if column is not None and column > 0 and force_file_scope:
        parts.append(f"column={int(column)}")
    return "|".join(parts)


def canonical_symbol_identity(
    *,
    qname: str,
    kind: str,
    usr: str | None = None,
    canonical_signature: str | None = None,
    project: str | None = None,
    repo: str | None = None,
    file: str | None = None,
    line: int | None = None,
    column: int | None = None,
    provider: str | None = None,
    include_provenance: str | None = None,
    force_file_scope: bool = False,
    repo_file_location_fallback: bool = False,
) -> str:
    """Canonical symbol identity.

    Stable clang USR wins when present and is namespaced by the owning project so
    independent repositories cannot alias. A canonical signature keeps the
    existing fallback contract. Clang cursor callers can require a typed
    repo/file/location fallback when neither is usable, preventing qname-only
    collisions without claiming cross-repo or cross-revision authority.
    """
    normalized_kind = _normalize_signature_text(kind or "symbol")
    normalized_signature = _normalize_signature_text(canonical_signature)
    normalized_usr = _normalize_signature_text(usr)
    normalized_qname = normalize_qualified_name(qname)
    raw_project = project or repo
    owning_project = (
        require_project_identity(raw_project, source="canonical_symbol_identity")
        if raw_project
        else ""
    )
    if provider or include_provenance:
        return canonical_external_usr_identity(
            usr=normalized_usr,
            canonical_signature=normalized_signature,
            provider=provider,
            include_provenance=include_provenance,
            project=owning_project or None,
            source="canonical_symbol_identity external provider",
        )
    if normalized_usr:
        if not owning_project:
            raise SymbolIdentityError(
                "clang USR identity requires a canonical project"
            )
        scoped_file = (
            _normalize_repo_relative_identity_file(file)
            if force_file_scope
            else None
        )
        return canonical_usr_identity(
            usr=normalized_usr,
            project=owning_project,
            file=scoped_file,
            canonical_signature=normalized_signature,
            source="canonical_symbol_identity",
        )
    if repo_file_location_fallback and not normalized_signature:
        normalized_file = _normalize_repo_relative_identity_file(file)
        try:
            normalized_line = int(line or 0)
            normalized_column = int(column or 0)
        except (TypeError, ValueError) as exc:
            raise SymbolIdentityError(
                "repo_file_location identity requires an integer declaration location"
            ) from exc
        if not owning_project:
            raise SymbolIdentityError(
                "repo_file_location identity requires a canonical project"
            )
        if not normalized_file or normalized_line <= 0 or normalized_column <= 0:
            raise SymbolIdentityError(
                "repo_file_location identity requires a repo-relative file, line, "
                "and column"
            )
        payload = [f"schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}"]
        if owning_project:
            payload.append(f"project={owning_project}")
        payload.extend(
            [
                f"file={normalized_file}",
                f"line={normalized_line}",
            ]
        )
        if normalized_column > 0:
            payload.append(f"column={normalized_column}")
        payload.extend(
            [
                f"kind={normalized_kind}",
                f"qname={normalized_qname}",
            ]
        )
        identity_key = "repo_file_location:" + "\x1f".join(payload)
        parse_repo_file_location_identity(
            identity_key, source="canonical_symbol_identity"
        )
        return identity_key
    scope = _identity_scope_key(
        project=owning_project,
        file=file,
        line=line,
        column=column,
        force_file_scope=force_file_scope,
    )
    payload = [
        f"schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}",
        f"qname={normalized_qname}",
        f"kind={normalized_kind}",
        f"sig={normalized_signature}",
    ]
    if scope:
        payload.append(f"scope={scope}")
    return "fallback:" + "\x1f".join(payload)


_WINDOWS_ABSOLUTE_IDENTITY_PATH_RE = re.compile(r"^[A-Za-z]:/")


def _normalize_identity_path(value: str | None) -> str:
    if not value:
        return ""
    normalized = posixpath.normpath(str(value).replace("\\", "/"))
    if normalized == ".":
        return ""
    if _WINDOWS_ABSOLUTE_IDENTITY_PATH_RE.match(normalized):
        normalized = normalized.casefold()
    return normalized.removeprefix("./")


def _is_absolute_identity_path(value: str) -> bool:
    return posixpath.isabs(value) or bool(
        _WINDOWS_ABSOLUTE_IDENTITY_PATH_RE.match(value)
    )


class _UnknownExternalProviderError(SymbolIdentityError):
    """An external declaration cannot be bound to a trusted provider."""

    def __init__(self, observed_path: str):
        self.observed_path = observed_path
        super().__init__(observed_path)

    def __str__(self) -> str:
        return (
            "external declaration requires a stable provider identity, "
            f"got path={self.observed_path!r}"
        )


def _record_external_reference_omission(
    omissions: ExternalReferenceOmissions | None,
    *,
    relation: str,
    cursor: Cursor,
    error: _UnknownExternalProviderError,
) -> None:
    if omissions is None:
        return
    key = (
        relation,
        normalize_qualified_name(get_qualified_name(cursor)),
        _cursor_kind_name(cursor),
        error.observed_path,
    )
    omissions[key] = omissions.get(key, 0) + 1


def _normalize_repo_relative_identity_file(file: str | None) -> str:
    normalized = _normalize_identity_path(file)
    if normalized and _is_absolute_identity_path(normalized):
        raise SymbolIdentityError(
            "repo_file_location identity cannot contain an absolute path"
        )
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise SymbolIdentityError(
            "repo_file_location identity requires a canonical repo-relative file"
        )
    return normalized


def validate_repo_file_location_identity_claim(
    symbol_key: str,
    *,
    project: str,
    file: str,
    line: int,
    column: int,
    kind: str,
    qname: str,
    source: str,
) -> None:
    """Require an explicit location key to agree with its serialized record."""

    identity = parse_repo_file_location_identity(symbol_key, source=source)
    expected = {
        "project": require_project_identity(project, source=f"{source}:project"),
        "file": _normalize_repo_relative_identity_file(file),
        "line": int(line),
        "column": int(column),
        "kind": _normalize_signature_text(kind),
        "qname": normalize_qualified_name(qname),
    }
    actual = {
        "project": identity.project,
        "file": identity.file,
        "line": identity.line,
        "column": identity.column,
        "kind": identity.kind,
        "qname": identity.qname,
    }
    if actual != expected:
        raise SymbolIdentityError(
            f"{source}: repo_file_location key does not match serialized fields: "
            f"expected={expected!r} actual={actual!r}"
        )


def _cursor_repo_file_location_identity(
    cursor: Cursor,
    *,
    project_dir: str | None,
    project: str | None,
    fallback_file: str | None,
) -> tuple[str, str]:
    """Return a stable file/project pair for explicit location identity."""

    location = getattr(cursor, "location", None)
    location_file = getattr(location, "file", None)
    location_name = getattr(location_file, "name", None)
    candidate = _normalize_identity_path(
        str(location_name) if location_name else fallback_file
    )
    if candidate.startswith("@provider/"):
        provider, include_provenance = parse_external_provider_file(
            candidate,
            source="cursor provider identity",
        )
        return (
            canonical_external_provider_file(
                provider,
                include_provenance,
                source="cursor provider identity",
            ),
            external_provider_project(provider, source="cursor provider identity"),
        )
    project_path = _normalize_identity_path(project_dir)
    if not _is_absolute_identity_path(candidate):
        return _normalize_repo_relative_identity_file(candidate), str(project or "")

    relative_file = ""
    if project_path and _is_absolute_identity_path(project_path):
        candidate_is_windows = bool(
            _WINDOWS_ABSOLUTE_IDENTITY_PATH_RE.match(candidate)
        )
        project_is_windows = bool(
            _WINDOWS_ABSOLUTE_IDENTITY_PATH_RE.match(project_path)
        )
        same_root_kind = candidate_is_windows == project_is_windows
        same_drive = (
            not candidate_is_windows or candidate[:2] == project_path[:2]
        )
        if same_root_kind and same_drive:
            relative_file = posixpath.relpath(candidate, project_path)
            if relative_file == ".." or relative_file.startswith("../"):
                relative_file = ""
    if relative_file:
        return _normalize_repo_relative_identity_file(relative_file), str(project or "")

    provider, include_provenance = symbol_provider_provenance(candidate)
    if provider and include_provenance:
        return (
            canonical_external_provider_file(
                provider,
                include_provenance,
                source="cursor provider identity",
            ),
            external_provider_project(provider, source="cursor provider identity"),
        )
    raise _UnknownExternalProviderError(candidate)


def _cursor_identity_location(
    cursor: Cursor,
    *,
    project_dir: str | None,
    fallback_file: str | None,
) -> tuple[str, bool]:
    """Return identity file plus whether clang locates it inside project_dir."""
    rel_file = fallback_file or ""
    location = getattr(cursor, "location", None)
    location_file = getattr(location, "file", None)
    location_name = getattr(location_file, "name", None)
    if not location_name:
        return rel_file, True
    if not project_dir:
        return str(location_name), True
    try:
        absolute_file = os.path.abspath(str(location_name))
        absolute_project = os.path.abspath(project_dir)
        is_project_local = os.path.commonpath(
            (absolute_file, absolute_project)
        ) == absolute_project
        rel_file = os.path.relpath(absolute_file, absolute_project)
    except (OSError, ValueError):
        return fallback_file or str(location_name), False
    return rel_file, is_project_local


def symbol_identity_for_cursor(
    cursor: Cursor,
    *,
    project_dir: str | None = None,
    project: str | None = None,
    repo: str | None = None,
    fallback_file: str | None = None,
    force_file_scope: bool = False,
) -> tuple[str, str, str]:
    """Return ``(identity_key, usr, canonical_signature)`` for a clang cursor."""
    qname = _canonicalize_clang_identity_text(
        get_qualified_name(cursor),
        project_dir,
    )
    usr = _cursor_usr(cursor, project_dir=project_dir)
    signature = _canonicalize_clang_identity_text(
        _cursor_canonical_signature(cursor),
        project_dir,
    )
    kind_name = _cursor_kind_name(cursor)
    # Anonymous namespaces have a stable clang USR but intentionally expose no
    # display name or type spelling.  Their clang kind is the only remaining
    # canonical signature claim; file + USR still distinguish the actual
    # namespace, including nested anonymous namespaces.
    if usr and not signature and kind_name == "NAMESPACE":
        signature = "kind=NAMESPACE"
    file_scope = force_file_scope
    uses_repo_file_location = not usr and not signature
    rel_file, identity_project = _cursor_repo_file_location_identity(
        cursor,
        project_dir=project_dir,
        project=project or repo,
        fallback_file=fallback_file,
    )
    requested_project = project or repo
    is_external = rel_file.startswith("@provider/") or (
        bool(requested_project) and identity_project != requested_project
    )
    provider, include_provenance = symbol_provider_provenance(rel_file)
    loc = getattr(cursor, "location", None)
    linkage = getattr(cursor, "linkage", None)
    linkage_name = getattr(linkage, "name", "") or str(linkage or "")
    storage = getattr(cursor, "storage_class", None)
    storage_name = getattr(storage, "name", "") or str(storage or "")
    semantic_parent = getattr(cursor, "semantic_parent", None)
    parent_kind = _cursor_kind_name(semantic_parent) if semantic_parent is not None else ""
    is_local = parent_kind in {
        "FUNCTION_DECL",
        "FUNCTION_TEMPLATE",
        "CXX_METHOD",
        "CONSTRUCTOR",
        "DESTRUCTOR",
        "CONVERSION_FUNCTION",
        "LAMBDA_EXPR",
    }
    if (
        any(
            marker in linkage_name
            for marker in ("INTERNAL", "NO_LINKAGE", "UNIQUE_EXTERNAL")
        )
        or storage_name == "STATIC"
        or is_local
    ):
        file_scope = True
    if not is_external and not usr and signature:
        if not identity_project:
            raise SymbolIdentityError(
                "signature fallback requires a canonical project"
            )
        line = getattr(loc, "line", None)
        column = getattr(loc, "column", None)
        if (
            isinstance(line, bool)
            or not isinstance(line, int)
            or line <= 0
            or isinstance(column, bool)
            or not isinstance(column, int)
            or column <= 0
        ):
            raise SymbolIdentityError(
                "signature fallback requires a positive line and column"
            )
        rel_file = _normalize_repo_relative_identity_file(rel_file)
        if not rel_file:
            raise SymbolIdentityError(
                "signature fallback requires a repository-relative file"
            )
        file_scope = True
    if is_external and usr:
        identity_key = canonical_external_usr_identity(
            usr=usr,
            canonical_signature=signature,
            provider=provider,
            include_provenance=include_provenance,
            project=identity_project,
            source="symbol_identity_for_cursor",
        )
    else:
        try:
            identity_key = canonical_symbol_identity(
                qname=qname,
                kind=kind_name,
                usr=usr,
                canonical_signature=signature,
                project=identity_project,
                file=rel_file,
                line=getattr(loc, "line", None),
                column=getattr(loc, "column", None),
                force_file_scope=file_scope or is_external or not usr,
                repo_file_location_fallback=uses_repo_file_location,
            )
        except SymbolIdentityError as exc:
            raise SymbolIdentityError(
                "symbol_identity_for_cursor: "
                f"kind={kind_name!r} qname={qname!r} "
                f"usr={usr!r} signature={signature!r} file={rel_file!r} "
                f"line={getattr(loc, 'line', None)!r} "
                f"column={getattr(loc, 'column', None)!r} "
                f"linkage={linkage_name!r} storage={storage_name!r} "
                f"parent_kind={parent_kind!r}: {exc}"
            ) from exc
    return identity_key, usr, signature


def symbol_reference_for_cursor(
    cursor: Cursor,
    *,
    project_dir: str | None = None,
    project_id: str | None = None,
    fallback_file: str | None = None,
    force_file_scope: bool = False,
) -> SymbolReference:
    key, usr, signature = symbol_identity_for_cursor(
        cursor,
        project_dir=project_dir,
        project=project_id,
        fallback_file=fallback_file,
        force_file_scope=force_file_scope,
    )
    location = getattr(cursor, "location", None)
    if is_repo_file_location_identity(key):
        location_identity = parse_repo_file_location_identity(
            key, source="symbol_reference_for_cursor"
        )
        relative_file = location_identity.file
        reference_project = location_identity.project
        reference_line = location_identity.line
        reference_column = location_identity.column
    else:
        relative_file, reference_project = _cursor_repo_file_location_identity(
            cursor,
            project_dir=project_dir,
            project=project_id,
            fallback_file=fallback_file,
        )
        reference_line = int(getattr(location, "line", 0) or 0)
        reference_column = int(getattr(location, "column", 0) or 0)
    provider, include_provenance = (
        symbol_provider_provenance(relative_file)
        if relative_file.startswith("@provider/")
        else ("", "")
    )
    if reference_project != project_id:
        if not provider or not include_provenance:
            raise SymbolIdentityError(
                "symbol_reference_for_cursor cannot authorize an unknown external provider"
            )
        expected_file = canonical_external_provider_file(
            provider,
            include_provenance,
            source="symbol_reference_for_cursor",
        )
        if relative_file != expected_file:
            raise SymbolIdentityError(
                "symbol_reference_for_cursor provider file is not canonical"
            )
    return {
        "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
        "symbol_key": key,
        "symbol_id": _compute_symbol_id(key),
        "qname": _canonicalize_clang_identity_text(
            get_qualified_name(cursor),
            project_dir,
        ),
        "usr": usr,
        "canonical_signature": signature,
        "symbol_kind": _cursor_kind_name(cursor),
        "project": reference_project,
        "file": relative_file,
        "line": reference_line,
        "column": reference_column,
        "provider": provider,
        "include_provenance": include_provenance,
    }


def _optional_symbol_reference_for_cursor(
    cursor: Cursor,
    *,
    relation: str,
    omissions: ExternalReferenceOmissions | None,
    project_dir: str | None,
    project_id: str | None,
    fallback_file: str | None,
) -> SymbolReference | None:
    """Return only references with authoritative local/provider provenance."""

    try:
        return symbol_reference_for_cursor(
            cursor,
            project_dir=project_dir,
            project_id=project_id,
            fallback_file=fallback_file,
        )
    except _UnknownExternalProviderError as exc:
        _record_external_reference_omission(
            omissions,
            relation=relation,
            cursor=cursor,
            error=exc,
        )
        return None


def _normalize_symbol_reference(value: object) -> SymbolReference | None:
    if not isinstance(value, dict):
        return None
    key = value.get("symbol_key")
    qname = value.get("qname")
    if not isinstance(key, str) or not key:
        return None
    if not isinstance(qname, str):
        qname = ""
    identity_version = int(
        value.get("symbol_identity_schema_version") or SYMBOL_IDENTITY_SCHEMA_VERSION
    )
    if identity_version != SYMBOL_IDENTITY_SCHEMA_VERSION:
        raise SymbolIdentityError(
            "symbol reference uses incompatible identity schema: "
            f"got v{identity_version}, expected v{SYMBOL_IDENTITY_SCHEMA_VERSION}"
        )
    expected_symbol_id = _compute_symbol_id(key)
    claimed_symbol_id = value.get("symbol_id")
    if (
        claimed_symbol_id is not None
        and (
            isinstance(claimed_symbol_id, bool)
            or int(claimed_symbol_id) != expected_symbol_id
        )
    ):
        raise SymbolIdentityError(
            "symbol reference ID does not match canonical key: "
            f"claimed={claimed_symbol_id} expected={expected_symbol_id} key={key!r}"
        )
    raw_line = value.get("line")
    raw_column = value.get("column")
    if (
        isinstance(raw_line, bool)
        or not isinstance(raw_line, int)
        or raw_line <= 0
        or isinstance(raw_column, bool)
        or not isinstance(raw_column, int)
        or raw_column <= 0
    ):
        raise SymbolIdentityError(
            "normalized symbol reference requires positive line and column"
        )
    normalized = {
        "symbol_identity_schema_version": identity_version,
        "symbol_key": key,
        "symbol_id": expected_symbol_id,
        "qname": normalize_qualified_name(qname),
        "usr": str(value.get("usr") or ""),
        "canonical_signature": _normalize_signature_text(
            str(value.get("canonical_signature") or "")
        ),
        "symbol_kind": str(value.get("symbol_kind") or value.get("kind") or ""),
        "project": str(value.get("project") or ""),
        "file": str(value.get("file") or ""),
        "line": int(raw_line),
        "column": int(raw_column),
        "provider": str(value.get("provider") or ""),
        "include_provenance": str(value.get("include_provenance") or ""),
    }
    provider = str(normalized["provider"])
    include_provenance = str(normalized["include_provenance"])
    reference_project = str(normalized["project"])
    reference_file = str(normalized["file"])
    has_external_claim = bool(
        provider or include_provenance or reference_file.startswith("@provider/")
    )
    if has_external_claim:
        expected_project = external_provider_project(
            provider,
            source="normalized external symbol reference",
        )
        expected_file = canonical_external_provider_file(
            provider,
            include_provenance,
            source="normalized external symbol reference",
        )
        if reference_project != expected_project or reference_file != expected_file:
            raise SymbolIdentityError(
                "normalized external symbol reference provider provenance is inconsistent"
            )
        if normalized["usr"]:
            expected_key = canonical_external_usr_identity(
                usr=normalized["usr"],
                canonical_signature=normalized["canonical_signature"],
                provider=provider,
                include_provenance=include_provenance,
                project=reference_project,
                source="normalized external symbol reference",
            )
            if key != expected_key:
                raise SymbolIdentityError(
                    "normalized external symbol reference key is inconsistent"
                )
    else:
        if not reference_project:
            raise SymbolIdentityError(
                "normalized local symbol reference project provenance is missing"
            )
        if normalized["usr"]:
            parsed_usr = parse_usr_identity(
                key,
                source="normalized local symbol reference",
            )
            if parsed_usr.project != reference_project or parsed_usr.usr != normalized[
                "usr"
            ]:
                raise SymbolIdentityError(
                    "normalized local symbol reference USR/project identity is inconsistent"
                )
            if parsed_usr.file and (
                parsed_usr.file != reference_file
                or parsed_usr.canonical_signature
                != normalized["canonical_signature"]
            ):
                raise SymbolIdentityError(
                    "normalized scoped local symbol reference provenance is inconsistent"
                )
        elif normalized["canonical_signature"]:
            if not key.startswith("fallback:"):
                raise SymbolIdentityError(
                    "normalized local symbol reference has an unsupported fallback key"
                )
            fields: dict[str, str] = {}
            for part in key.removeprefix("fallback:").split("\x1f"):
                field, separator, field_value = part.partition("=")
                if separator:
                    fields[field] = field_value
            scope: dict[str, str] = {}
            for part in fields.get("scope", "").split("|"):
                field, separator, field_value = part.partition("=")
                if separator:
                    scope[field] = field_value
            if (
                fields.get("schema") != f"v{SYMBOL_IDENTITY_SCHEMA_VERSION}"
                or fields.get("qname") != normalized["qname"]
                or fields.get("kind") != normalized["symbol_kind"]
                or fields.get("sig") != normalized["canonical_signature"]
                or scope.get("project") != reference_project
                or scope.get("file") != reference_file
                or scope.get("line") != str(normalized["line"])
                or scope.get("column") != str(normalized["column"])
            ):
                raise SymbolIdentityError(
                    "normalized local symbol reference fallback provenance is inconsistent"
                )
        elif is_repo_file_location_identity(key):
            location_identity = parse_repo_file_location_identity(
                key,
                source="normalized local symbol reference",
            )
            if (
                location_identity.project != reference_project
                or location_identity.file != reference_file
                or location_identity.line != normalized["line"]
                or location_identity.column != normalized["column"]
                or location_identity.kind != normalized["symbol_kind"]
                or location_identity.qname != normalized["qname"]
            ):
                raise SymbolIdentityError(
                    "normalized local symbol reference location provenance is inconsistent"
                )
        else:
            raise SymbolIdentityError(
                "normalized local symbol reference lacks a trusted identity"
            )
    return normalized


def _compute_symbol_id(symbol_key: str) -> int:
    """Deterministic unsigned 64-bit ID from a canonical symbol identity."""

    return compute_symbol_id(symbol_key)


class FunctionDef:
    """A function definition with its source location and call references."""
    __slots__ = ['name', 'qualified_name', 'file', 'line', 'column', 'end_line', 'text',
                 'callees', 'referenced_types', 'dep_level', 'is_definition',
                 'ast_depth', 'sibling_index', 'ast_node_type',
                 'baselib_callees', 'symbol_key', 'symbol_id', 'usr',
                 'canonical_signature', 'symbol_kind', 'callee_keys',
                 'referenced_type_keys', 'callee_refs', 'baselib_callee_refs',
                 'referenced_type_refs', 'semantic_symbol_ids',
                 'semantic_call_targets', 'semantic_type_refs',
                 'semantic_def_use', 'semantic_symbol_identities']

    def __init__(self, name: str, qualified_name: str, file: str, line: int,
                 text: str, callees: list, is_definition: bool = True,
                 column: int = 0, end_line: int = 0,
                 ast_depth: list[int] | None = None,
                 sibling_index: list[int] | None = None,
                 ast_node_type: list[int] | None = None,
                 referenced_types: list | None = None,
                 baselib_callees: list | None = None,
                 symbol_key: str | None = None,
                 usr: str | None = None,
                 canonical_signature: str | None = None,
                 symbol_kind: str = "function",
                 callee_keys: list | None = None,
                 referenced_type_keys: list | None = None,
                 callee_refs: list[SymbolReference] | None = None,
                 baselib_callee_refs: list[SymbolReference] | None = None,
                 referenced_type_refs: list[SymbolReference] | None = None,
                 semantic_symbol_ids: list[int] | None = None,
                 semantic_call_targets: list[int] | None = None,
                 semantic_type_refs: list[int] | None = None,
                 semantic_def_use: list[int] | None = None,
                 semantic_symbol_identities: list[dict[str, object]] | None = None):
        self.name = name
        self.qualified_name = normalize_qualified_name(qualified_name)
        self.file = file
        self.line = line
        self.column = int(column or 0)
        self.end_line = end_line or (line + text.count('\n'))
        self.text = text
        self.callees = [normalize_qualified_name(value) for value in callees]
        # qualified names of record/enum/typedef types referenced by this
        # function (params, return, locals, member access) -- captured during the
        # SAME libclang parse as callees so it round-trips IPC and feeds the
        # offline type_refs/type_edges builders. Mirrors `callees`.
        self.referenced_types = [
            normalize_qualified_name(value) for value in (referenced_types or [])
        ]
        # Cross-linkable base-lib callees (std::/boost::) dropped from `callees`
        # by the normal system-prefix filter, kept SEPARATELY for the optional
        # cross-repo linker. Empty/ignored unless --global-symbol-index is given.
        self.baselib_callees = [
            normalize_qualified_name(value) for value in (baselib_callees or [])
        ]
        self.usr = str(usr or "")
        self.canonical_signature = _normalize_signature_text(canonical_signature)
        self.symbol_kind = str(symbol_kind or "function")
        self.symbol_key = symbol_key or canonical_symbol_identity(
            qname=qualified_name,
            kind=self.symbol_kind,
            canonical_signature=self.canonical_signature,
            file=file,
            line=line,
            column=self.column,
            force_file_scope=True,
        )
        self.symbol_id = _compute_symbol_id(self.symbol_key)
        # Identity-key mirrors of display qnames. Parsed clang paths populate
        # these with canonical identities; legacy/manual tests may leave them
        # empty and ProjectIndex will resolve unique qnames as a compatibility
        # fallback.
        self.callee_keys = list(callee_keys or [])
        self.referenced_type_keys = list(referenced_type_keys or [])
        self.callee_refs = [
            normalized
            for value in (callee_refs or [])
            if (normalized := _normalize_symbol_reference(value)) is not None
        ]
        self.baselib_callee_refs = [
            normalized
            for value in (baselib_callee_refs or [])
            if (normalized := _normalize_symbol_reference(value)) is not None
        ]
        self.referenced_type_refs = [
            normalized
            for value in (referenced_type_refs or [])
            if (normalized := _normalize_symbol_reference(value)) is not None
        ]
        self.dep_level = 0
        self.is_definition = is_definition
        # Store per-character AST sidecars compactly in the repo-wide index.
        # Large projects can have 80k+ function bodies; Python list[int] would
        # dominate RSS before document emission even starts. Values are already
        # clamped to uint16 by extract_clang_ast_metadata().
        self.ast_depth = array('H', ast_depth or [])
        self.sibling_index = array('H', sibling_index or [])
        self.ast_node_type = array('H', ast_node_type or [])
        self.semantic_symbol_ids = array('Q', semantic_symbol_ids or [])
        self.semantic_call_targets = array('Q', semantic_call_targets or [])
        self.semantic_type_refs = array('Q', semantic_type_refs or [])
        self.semantic_def_use = array('B', semantic_def_use or [])
        identity_registry = SymbolIdentityRegistry()
        self.semantic_symbol_identities = identity_registry.register_records(
            semantic_symbol_identities or [],
            source=f"FunctionDef({self.qualified_name})",
        )

    def to_dict(self) -> dict:
        """Serialize for multiprocessing IPC."""
        return {
            'name': self.name, 'qualified_name': self.qualified_name,
            'file': self.file, 'line': self.line, 'column': self.column,
            'text': self.text,
            'callees': self.callees, 'is_definition': self.is_definition,
            'end_line': self.end_line,
            'ast_depth': self.ast_depth,
            'sibling_index': self.sibling_index,
            'ast_node_type': self.ast_node_type,
            'referenced_types': self.referenced_types,
            'baselib_callees': self.baselib_callees,
            'symbol_key': self.symbol_key,
            'usr': self.usr,
            'canonical_signature': self.canonical_signature,
            'symbol_kind': self.symbol_kind,
            'callee_keys': self.callee_keys,
            'referenced_type_keys': self.referenced_type_keys,
            'callee_refs': self.callee_refs,
            'baselib_callee_refs': self.baselib_callee_refs,
            'referenced_type_refs': self.referenced_type_refs,
            'semantic_symbol_ids': self.semantic_symbol_ids,
            'semantic_call_targets': self.semantic_call_targets,
            'semantic_type_refs': self.semantic_type_refs,
            'semantic_def_use': self.semantic_def_use,
            'semantic_symbol_identities': self.semantic_symbol_identities,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FunctionDef':
        return cls(**d)


class TypeDef:
    """A type DEFINITION (struct/class/enum/union/typedef/using).

    Captured during the same libclang parse as functions so the offline doc
    builder can emit a type-def CHUNK whose qname == the type's qualified name.
    That chunk is the *target* of a function->type ``type_edge`` (the source is
    the function chunk's ``referenced_types``). Definitions usually live in
    headers, which is why headers are now part of the index file set.
    """
    __slots__ = ['name', 'qualified_name', 'file', 'line', 'column', 'end_line', 'text',
                 'kind', 'symbol_key', 'symbol_id', 'usr',
                 'canonical_signature', 'symbol_kind']

    def __init__(self, name: str, qualified_name: str, file: str, line: int,
                 text: str, kind: int, column: int = 0, end_line: int = 0,
                 symbol_key: str | None = None, usr: str | None = None,
                 canonical_signature: str | None = None,
                 symbol_kind: str = "type"):
        self.name = name
        self.qualified_name = normalize_qualified_name(qualified_name)
        self.file = file
        self.line = line
        self.column = int(column or 0)
        self.end_line = end_line or (line + text.count('\n'))
        self.text = text
        # node-bucket kind for parts_info (4=class/struct, 7=typedef/using)
        self.kind = kind
        self.usr = str(usr or "")
        self.canonical_signature = _normalize_signature_text(canonical_signature)
        self.symbol_kind = str(symbol_kind or "type")
        self.symbol_key = symbol_key or canonical_symbol_identity(
            qname=qualified_name,
            kind=self.symbol_kind,
            canonical_signature=self.canonical_signature,
            file=file,
            line=line,
            column=self.column,
            force_file_scope=True,
        )
        self.symbol_id = _compute_symbol_id(self.symbol_key)

    def to_dict(self) -> dict:
        return {
            'name': self.name, 'qualified_name': self.qualified_name,
            'file': self.file, 'line': self.line, 'column': self.column,
            'text': self.text,
            'kind': self.kind, 'end_line': self.end_line,
            'symbol_key': self.symbol_key,
            'usr': self.usr,
            'canonical_signature': self.canonical_signature,
            'symbol_kind': self.symbol_kind,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TypeDef':
        return cls(**d)


class MacroDef:
    """A trainable preprocessor macro definition captured from project headers."""

    __slots__ = [
        'name',
        'file',
        'line',
        'text',
        'params',
        'visible_in_file',
        'visible_line',
        'sequence',
        'condition_names',
        'condition_lines',
        'undef_text',
        'previous',
        'project_id',
        'symbol_key',
        'symbol_id',
    ]

    def __init__(
        self,
        name: str,
        file: str,
        line: int,
        text: str,
        params: list[str] | None = None,
        *,
        project_id: str,
        visible_in_file: str | None = None,
        visible_line: int | None = None,
        sequence: int = 0,
        condition_names: list[str] | None = None,
        condition_lines: list[str] | None = None,
        undef_text: str = "",
        previous: "MacroDef | None" = None,
    ):
        self.name = name
        self.file = file
        self.line = line
        self.text = text
        self.params = list(params or [])
        self.project_id = require_project_identity(
            project_id,
            source=f"MacroDef({name})",
        )
        self.visible_in_file = visible_in_file or file
        self.visible_line = visible_line if visible_line is not None else line
        self.sequence = int(sequence)
        self.condition_names = list(condition_names or [])
        self.condition_lines = list(condition_lines or [])
        self.undef_text = undef_text
        self.previous = previous
        self.symbol_key = canonical_symbol_identity(
            qname=name,
            kind="macro",
            canonical_signature="(" + ",".join(self.params) + ")",
            project=self.project_id,
            file=self.file,
            line=self.line,
            force_file_scope=True,
        )
        self.symbol_id = _compute_symbol_id(self.symbol_key)


class _MacroVisibilityRecord:
    """Compact root-specific view of one canonical macro definition."""

    __slots__ = ["definition", "visible_line", "sequence", "previous"]

    def __init__(
        self,
        definition: MacroDef,
        *,
        visible_line: int,
        sequence: int,
        previous: "_MacroVisibilityRecord | None",
    ):
        self.definition = definition
        self.visible_line = int(visible_line)
        self.sequence = int(sequence)
        self.previous = previous


# Preserve the old registry's fail-closed memory envelope in bytes rather than
# silently turning its 250k MacroDef cap into an unbounded number of root views.
# This is the measured shallow allocation of one first retained MacroDef slot:
# object + three owned lists + occurrence tuple/set + name-history list +
# macro/name dictionaries. String/symbol payloads are deliberately omitted, so
# the process-wide RSS guard below remains the authoritative total-memory cap.
_LEGACY_MACRO_OCCURRENCE_KEY_FIXTURE = (None,) * 5
LEGACY_MACRO_RETAINED_SHALLOW_BYTES = (
    sys.getsizeof(MacroDef.__new__(MacroDef))
    + 3 * sys.getsizeof([])
    + sys.getsizeof(_LEGACY_MACRO_OCCURRENCE_KEY_FIXTURE)
    + sys.getsizeof({_LEGACY_MACRO_OCCURRENCE_KEY_FIXTURE})
    + sys.getsizeof([None])
    + 2 * sys.getsizeof({"macro": None})
)


def _macro_definition_key(macro: MacroDef) -> tuple[object, ...]:
    """Identity of source definition content, excluding root visibility."""

    return (
        macro.name,
        macro.file,
        macro.line,
        macro.text,
        tuple(macro.params),
        tuple(macro.condition_names),
        tuple(macro.condition_lines),
        macro.undef_text,
        macro.project_id,
    )


class ProjectIndex:
    """Cross-file function index for a single project."""

    def __init__(self):
        # canonical symbol identity -> FunctionDef (definitions only).
        # qname remains display/search metadata; qname indexes below are only
        # compatibility/lookup aids and can be ambiguous.
        self.functions: dict[str, FunctionDef] = {}
        self.function_qnames: dict[str, list[str]] = defaultdict(list)
        # file -> list of function symbol identities defined there
        self.file_functions: dict[str, list[str]] = defaultdict(list)
        # file -> preamble text (includes, typedefs, forward decls)
        self.file_preambles: dict[str, str] = {}
        # callee symbol identity -> list of caller symbol identities
        self.callers: dict[str, list[str]] = defaultdict(list)
        # symbol identity -> TypeDef (struct/class/enum/union/typedef/using).
        # Separate from `functions` so the call graph is untouched; consulted
        # when building docs to pull a referenced type's DEFINITION in as a
        # type-edge target chunk.
        self.typedefs: dict[str, TypeDef] = {}
        self.typedef_qnames: dict[str, list[str]] = defaultdict(list)
        # macro name -> latest MacroDef plus full definition history. Header macro
        # definitions are not clang AST nodes, so they live in a small side registry
        # used for standalone macro docs and same-document invocation routes.
        self.macros: dict[str, MacroDef] = {}
        self.macros_by_name: dict[str, list[MacroDef]] = defaultdict(list)
        self.macro_definitions: list[MacroDef] = []
        self._macro_occurrence_keys: set[tuple[str, str, str, int, int]] = set()
        self._macro_definition_by_key: dict[
            tuple[object, ...],
            MacroDef,
        ] = {}
        self.macro_visibility_by_file: dict[
            str,
            dict[str, list[_MacroVisibilityRecord]],
        ] = {}
        self.macro_visibility_record_count = 0
        self.macro_visibility_bytes = 0
        self.symbol_id_registry = SymbolIdentityRegistry()
        self.symbol_id_keys = self.symbol_id_registry.keys_by_id

    def _register_symbol_key(self, symbol_key: str) -> None:
        if not symbol_key:
            return
        self.symbol_id_registry.register(symbol_key, source="ProjectIndex")

    def add_typedef(self, td: TypeDef):
        """Register a type definition. First definition with text wins; never
        overwrite a real definition with a shorter/forward one."""
        key = td.symbol_key
        if not key:
            return
        self._register_symbol_key(key)
        existing = self.typedefs.get(key)
        if existing is not None and len(existing.text) >= len(td.text):
            return
        self.typedefs[key] = td
        qname = normalize_inline_namespace_qname(td.qualified_name)
        if qname and key not in self.typedef_qnames[qname]:
            self.typedef_qnames[qname].append(key)

    def add_macro(self, macro: MacroDef):
        if not macro.name:
            return
        self._register_symbol_key(macro.symbol_key)
        occurrence_key = (
            macro.name,
            macro.visible_in_file,
            macro.file,
            macro.line,
            macro.sequence,
        )
        if occurrence_key in self._macro_occurrence_keys:
            return
        self._macro_occurrence_keys.add(occurrence_key)
        existing = self.macros.get(macro.name)
        if (
            existing is not None
            and existing.visible_in_file == macro.visible_in_file
            and existing.sequence > macro.sequence
        ):
            self.macros_by_name[macro.name].append(macro)
            self.macro_definitions.append(macro)
            return
        self.macros_by_name[macro.name].append(macro)
        self.macro_definitions.append(macro)
        self.macros[macro.name] = macro

    def new_canonical_macro_definition_count(
        self,
        macros: Sequence[MacroDef],
    ) -> int:
        keys = {_macro_definition_key(macro) for macro in macros}
        return sum(key not in self._macro_definition_by_key for key in keys)

    def add_macro_environment(
        self,
        target_file: str,
        macros: Sequence[MacroDef],
    ) -> tuple[int, int, int]:
        """Compact one root's retained macros into canonical defs + views."""

        if target_file in self.macro_visibility_by_file:
            raise ValueError(
                f"macro visibility environment already exists for {target_file}"
            )
        environment: dict[str, list[_MacroVisibilityRecord]] = defaultdict(list)
        records_by_occurrence: dict[int, _MacroVisibilityRecord] = {}
        canonical_before = len(self.macro_definitions)
        top_level_bytes_before = sys.getsizeof(self.macro_visibility_by_file)
        visibility_count = 0
        for macro in sorted(macros, key=lambda item: item.sequence):
            definition_key = _macro_definition_key(macro)
            canonical = self._macro_definition_by_key.get(definition_key)
            if canonical is None:
                canonical = MacroDef(
                    name=macro.name,
                    file=macro.file,
                    line=macro.line,
                    text=macro.text,
                    params=macro.params,
                    project_id=macro.project_id,
                    visible_in_file=macro.file,
                    visible_line=macro.line,
                    sequence=len(self.macro_definitions),
                    condition_names=macro.condition_names,
                    condition_lines=macro.condition_lines,
                    undef_text=macro.undef_text,
                )
                self._macro_definition_by_key[definition_key] = canonical
                self.add_macro(canonical)
            previous = None
            if macro.previous is not None:
                previous = records_by_occurrence.get(id(macro.previous))
                if previous is None:
                    raise ValueError(
                        "retained macro environment omitted a previous "
                        f"definition: root={target_file} macro={macro.name}"
                    )
            record = _MacroVisibilityRecord(
                canonical,
                visible_line=macro.visible_line,
                sequence=macro.sequence,
                previous=previous,
            )
            records_by_occurrence[id(macro)] = record
            environment[macro.name].append(record)
            visibility_count += 1

        stored_environment = dict(environment)
        environment_bytes = (
            sys.getsizeof(stored_environment)
            + sys.getsizeof(target_file)
            + sum(
                sys.getsizeof(name)
                + sys.getsizeof(records)
                + sum(sys.getsizeof(record) for record in records)
                for name, records in stored_environment.items()
            )
        )
        self.macro_visibility_by_file[target_file] = stored_environment
        environment_bytes += max(
            0,
            sys.getsizeof(self.macro_visibility_by_file)
            - top_level_bytes_before,
        )
        self.macro_visibility_record_count += visibility_count
        self.macro_visibility_bytes += environment_bytes
        return (
            len(self.macro_definitions) - canonical_before,
            visibility_count,
            environment_bytes,
        )

    def add_function(self, func: FunctionDef):
        """Add a function definition to the index."""
        key = func.symbol_key
        self._register_symbol_key(key)
        self.symbol_id_registry.register_records(
            func.semantic_symbol_identities,
            source=f"ProjectIndex:{func.file}:{func.line}:{func.qualified_name}",
        )
        for ref in (
            list(getattr(func, "callee_refs", []))
            + list(getattr(func, "baselib_callee_refs", []))
            + list(getattr(func, "referenced_type_refs", []))
        ):
            ref_key = ref.get("symbol_key")
            if isinstance(ref_key, str):
                self._register_symbol_key(ref_key)
        if key in self.functions and self.functions[key].is_definition:
            return  # don't overwrite definitions with declarations
        self.functions[key] = func
        qname = normalize_inline_namespace_qname(func.qualified_name)
        if qname and key not in self.function_qnames[qname]:
            self.function_qnames[qname].append(key)
        if func.is_definition:
            self.file_functions[func.file].append(key)

    def resolve_function_key(
        self, ref: str | SymbolReference | None
    ) -> str | None:
        """Resolve an identity/reference or an unambiguous display qname."""
        if not ref:
            return None
        if isinstance(ref, dict):
            normalized = _normalize_symbol_reference(ref)
            if normalized is None:
                return None
            symbol_key = str(normalized["symbol_key"])
            if symbol_key in self.functions:
                return symbol_key
            if (
                normalized.get("provider")
                or normalized.get("include_provenance")
                or str(normalized.get("file") or "").startswith("@provider/")
            ):
                return None
            candidates = list(
                self.function_qnames.get(
                    normalize_inline_namespace_qname(str(normalized["qname"])), []
                )
            )
            usr = str(normalized.get("usr") or "")
            if usr:
                candidates = [key for key in candidates if self.functions[key].usr == usr]
            signature = str(normalized.get("canonical_signature") or "")
            if signature:
                candidates = [
                    key
                    for key in candidates
                    if self.functions[key].canonical_signature == signature
                ]
            symbol_kind = str(normalized.get("symbol_kind") or "")
            if symbol_kind:
                candidates = [
                    key for key in candidates if self.functions[key].symbol_kind == symbol_kind
                ]
            return candidates[0] if len(candidates) == 1 else None
        if ref in self.functions:
            return ref
        keys = self.function_qnames.get(normalize_inline_namespace_qname(ref), [])
        if len(keys) == 1:
            return keys[0]
        return None

    def resolve_type_key(self, ref: str | SymbolReference | None) -> str | None:
        if not ref:
            return None
        if isinstance(ref, dict):
            normalized = _normalize_symbol_reference(ref)
            if normalized is None:
                return None
            symbol_key = str(normalized["symbol_key"])
            if symbol_key in self.typedefs:
                return symbol_key
            if (
                normalized.get("provider")
                or normalized.get("include_provenance")
                or str(normalized.get("file") or "").startswith("@provider/")
            ):
                return None
            candidates = list(
                self.typedef_qnames.get(
                    normalize_inline_namespace_qname(str(normalized["qname"])), []
                )
            )
            usr = str(normalized.get("usr") or "")
            if usr:
                candidates = [key for key in candidates if self.typedefs[key].usr == usr]
            signature = str(normalized.get("canonical_signature") or "")
            if signature:
                candidates = [
                    key
                    for key in candidates
                    if self.typedefs[key].canonical_signature == signature
                ]
            symbol_kind = str(normalized.get("symbol_kind") or "")
            if symbol_kind:
                candidates = [
                    key for key in candidates if self.typedefs[key].symbol_kind == symbol_kind
                ]
            return candidates[0] if len(candidates) == 1 else None
        if ref in self.typedefs:
            return ref
        keys = self.typedef_qnames.get(normalize_inline_namespace_qname(ref), [])
        if len(keys) == 1:
            return keys[0]
        return None

    def _function_callee_keys(self, func: FunctionDef) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        refs: list[str | SymbolReference] = list(getattr(func, "callee_refs", []) or [])
        refs.extend(list(getattr(func, "callee_keys", []) or []))
        refs.extend(func.callees)
        for ref in refs:
            key = self.resolve_function_key(ref)
            if key is None or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        return keys

    def _function_referenced_type_keys(self, func: FunctionDef) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        refs: list[str | SymbolReference] = list(
            getattr(func, "referenced_type_refs", []) or []
        )
        refs.extend(list(getattr(func, "referenced_type_keys", []) or []))
        refs.extend(list(getattr(func, "referenced_types", []) or []))
        for ref in refs:
            key = self.resolve_type_key(ref)
            if key is None or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        return keys

    def build_reverse_edges(self):
        """Build caller -> callee reverse edges for dep level computation."""
        self.callers.clear()
        for symbol_key, func in self.functions.items():
            for callee in self._function_callee_keys(func):
                if callee in self.functions:
                    self.callers[callee].append(symbol_key)

    def compute_dep_levels(self):
        """Compute dependency levels via BFS from leaves."""
        # Find leaves: functions with no callees in the index
        in_degree = {}
        for symbol_key, func in self.functions.items():
            local_callees = [
                c for c in self._function_callee_keys(func)
                if c in self.functions and c != symbol_key
            ]
            in_degree[symbol_key] = len(local_callees)

        queue = deque()
        for symbol_key, deg in in_degree.items():
            if deg == 0:
                self.functions[symbol_key].dep_level = 0
                queue.append(symbol_key)

        self.build_reverse_edges()

        while queue:
            symbol_key = queue.popleft()
            level = self.functions[symbol_key].dep_level
            for caller_name in self.callers.get(symbol_key, []):
                new_level = level + 1
                if new_level > self.functions[caller_name].dep_level:
                    self.functions[caller_name].dep_level = new_level
                in_degree[caller_name] -= 1
                if in_degree[caller_name] == 0:
                    queue.append(caller_name)

        # Handle cycles
        max_level = max((f.dep_level for f in self.functions.values()), default=0)
        for symbol_key, deg in in_degree.items():
            if deg > 0:
                self.functions[symbol_key].dep_level = max_level + 1

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            'total_functions': len(self.functions),
            'total_files': len(self.file_functions),
            'definitions': sum(1 for f in self.functions.values() if f.is_definition),
            'max_dep_level': max((f.dep_level for f in self.functions.values()), default=0),
        }


def is_system_function(name: str | None) -> bool:
    """Check if a function name looks like a system/stdlib function."""
    if not isinstance(name, str) or not name:
        return False
    return any(name.startswith(p) for p in SYSTEM_PREFIXES)


def is_crosslinkable_baselib(name: str | None) -> bool:
    """True if ``name`` is a cross-linkable base-lib symbol (std::/boost::).

    Such a callee is filtered out of the normal ``callees`` list (it matches
    SYSTEM_PREFIXES) but is captured separately so the cross-repo linker can
    resolve it against the global symbol index. This does NOT affect normal
    callee/call-edge behavior.
    """
    if not isinstance(name, str) or not name:
        return False
    return any(name.startswith(p) for p in CROSSLINKABLE_NS_PREFIXES)


_DROP_CLANG_ARG_PREFIXES = (
    "-fsanitize=",
    "-fno-sanitize=",
    "-fsanitize-recover=",
    "-fno-sanitize-recover=",
    "-fsanitize-trap=",
    "-fsanitize-ignorelist=",
    "-fsanitize-blacklist=",
    "-fsanitize-coverage=",
)

_DROP_CLANG_ARGS = {
    "-fsanitize-address-use-after-scope",
    "-fno-sanitize-address-use-after-scope",
    "-shared-libasan",
    "-static-libasan",
    "-shared-libsan",
    "-static-libsan",
}


def _is_runtime_only_sanitizer_flag(text: str) -> bool:
    if text in _DROP_CLANG_ARGS:
        return True
    if text.startswith(("-shared-lib", "-static-lib")) and "san" in text:
        return True
    return False


def _sanitize_compile_args_for_clang(args: list[str] | None) -> list[str]:
    """Drop nullish and runtime-only compile args that break libclang parsing."""
    if not args:
        return []
    sanitized: list[str] = []
    for arg in args:
        if arg is None:
            continue
        text = str(arg).strip()
        if not text:
            continue
        if _is_runtime_only_sanitizer_flag(text):
            continue
        if any(text.startswith(prefix) for prefix in _DROP_CLANG_ARG_PREFIXES):
            continue
        sanitized.append(text)
    return sanitized


_WIDE_SOURCE_BOM_CODECS = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)
_WIDE_SOURCE_CODECS = frozenset(
    encoding for _bom, encoding in _WIDE_SOURCE_BOM_CODECS
)


def _wide_source_codec(raw: bytes) -> str | None:
    for bom, encoding in _WIDE_SOURCE_BOM_CODECS:
        if raw.startswith(bom):
            return encoding
    return None


def _decode_source_bytes(raw: bytes, filename: str) -> tuple[str, str]:
    """Decode source with a byte-exact fallback for mixed legacy text."""

    encoding = _wide_source_codec(raw)
    if encoding is not None:
        # Use the explicit endian codec so the decoded U+FEFF round-trips to
        # the original BOM. Malformed BOM-declared input must fail closed
        # instead of falling through to a single-byte codec.
        text = raw.decode(encoding, errors="strict")
    else:
        try:
            text, encoding = raw.decode("utf-8", errors="strict"), "utf-8"
        except UnicodeDecodeError:
            try:
                text, encoding = raw.decode("cp1252", errors="strict"), "cp1252"
            except UnicodeDecodeError:
                # Historical source trees can mix Shift-JIS comments with raw
                # single-byte font tables in one translation unit. No semantic
                # codec covers that mixture; ISO-8859-1 preserves every byte and
                # keeps libclang byte offsets exact.
                text, encoding = raw.decode("latin-1", errors="strict"), "latin-1"
    if "\0" in text and "\0" in _mask_non_code(text):
        raise ValueError(f"source contains NUL byte: {filename}")
    return text, encoding


def _read_source_file(filename: str) -> tuple[str, bytes, str]:
    with open(filename, "rb") as source_file:
        source_bytes = source_file.read()
    source, source_encoding = _decode_source_bytes(source_bytes, filename)
    if source.encode(source_encoding, errors="strict") != source_bytes:
        raise ValueError(f"source decoding did not round-trip exactly: {filename}")
    if source_encoding in _WIDE_SOURCE_CODECS:
        # libclang rejects UTF-16/32 source files even when they carry a BOM.
        # Parse an equivalent UTF-8 unsaved buffer and make every offset mapper
        # and text extractor use those same parser bytes.
        source_bytes = source.encode("utf-8", errors="strict")
        source_encoding = "utf-8"
    return source, source_bytes, source_encoding


def get_function_text(cursor: Cursor, tu: TranslationUnit) -> str:
    """Extract the source text for a cursor's extent."""
    extent = cursor.extent
    start = extent.start
    end = extent.end

    try:
        filename = start.file.name if start.file else None
        if not filename or not os.path.exists(filename):
            return ""

        _content, content_bytes, source_encoding = _read_source_file(filename)
        start_offset = int(start.offset)
        end_offset = int(end.offset)
        if 0 <= start_offset < end_offset <= len(content_bytes):
            return content_bytes[start_offset:end_offset].decode(
                source_encoding,
                errors="strict",
            )
    except SymbolIdentityError:
        raise
    except OSError:
        return ""
    return ""


def _byte_to_char_mapper(
    source: str,
    source_encoding: str = "utf-8",
) -> Callable[[int], int]:
    source_bytes = source.encode(source_encoding, errors="strict")
    byte_len = len(source_bytes)
    if byte_len == len(source):
        return lambda byte_offset: max(0, min(byte_offset, len(source)))

    byte_to_char = [0] * (byte_len + 1)
    byte_offset = 0
    for char_idx, ch in enumerate(source):
        ch_len = len(ch.encode(source_encoding, errors="strict"))
        for inner in range(ch_len):
            if byte_offset + inner < byte_len:
                byte_to_char[byte_offset + inner] = char_idx
        byte_offset += ch_len
    byte_to_char[byte_len] = len(source)

    def _map(byte_offset: int) -> int:
        if byte_offset <= 0:
            return 0
        if byte_offset >= byte_len:
            return len(source)
        return byte_to_char[byte_offset]

    return _map


def _cursor_extent_offsets(
    cursor: Cursor,
    filename: str,
    byte_to_char: Callable[[int], int],
) -> tuple[int, int] | None:
    extent = cursor.extent
    start = extent.start
    end = extent.end
    start_file = start.file.name if start.file else None
    end_file = end.file.name if end.file else None
    normalized_filename = os.path.normcase(os.path.normpath(filename))
    if (
        start_file is None
        or end_file is None
        or os.path.normcase(os.path.normpath(start_file)) != normalized_filename
        or os.path.normcase(os.path.normpath(end_file)) != normalized_filename
    ):
        return None
    char_start = byte_to_char(int(start.offset))
    char_end = byte_to_char(int(end.offset))
    if char_start >= char_end:
        return None
    return char_start, char_end


_CLANG_CURSOR_BUCKETS = {
    # 1-9: declarations
    "FUNCTION_DECL": 1,
    "CXX_METHOD": 1,
    "CONSTRUCTOR": 1,
    "DESTRUCTOR": 1,
    "CONVERSION_FUNCTION": 1,
    "FUNCTION_TEMPLATE": 1,
    "CLASS_DECL": 2,
    "CLASS_TEMPLATE": 2,
    "CLASS_TEMPLATE_PARTIAL_SPECIALIZATION": 2,
    "STRUCT_DECL": 3,
    "UNION_DECL": 3,
    "ENUM_DECL": 4,
    "NAMESPACE": 5,
    "VAR_DECL": 6,
    "PARM_DECL": 6,
    "FIELD_DECL": 7,
    "TYPEDEF_DECL": 9,
    "TYPE_ALIAS_DECL": 9,
    "USING_DECLARATION": 9,
    "USING_DIRECTIVE": 9,
    # 10-19: statements
    "COMPOUND_STMT": 10,
    "IF_STMT": 11,
    "FOR_STMT": 12,
    "WHILE_STMT": 13,
    "DO_STMT": 13,
    "RETURN_STMT": 14,
    "SWITCH_STMT": 15,
    "CASE_STMT": 15,
    "DEFAULT_STMT": 15,
    "TRY_STMT": 17,
    "CXX_CATCH_STMT": 17,
    "CXX_THROW_EXPR": 17,
    "BREAK_STMT": 18,
    "CONTINUE_STMT": 18,
    "GOTO_STMT": 18,
    "LABEL_STMT": 19,
    # 20-29: expressions
    "CALL_EXPR": 20,
    "BINARY_OPERATOR": 21,
    "COMPOUND_ASSIGNMENT_OPERATOR": 23,
    "UNARY_OPERATOR": 22,
    "CONDITIONAL_OPERATOR": 24,
    "MEMBER_REF_EXPR": 25,
    "ARRAY_SUBSCRIPT_EXPR": 26,
    "CSTYLE_CAST_EXPR": 27,
    "CXX_STATIC_CAST_EXPR": 27,
    "CXX_DYNAMIC_CAST_EXPR": 27,
    "CXX_REINTERPRET_CAST_EXPR": 27,
    "CXX_CONST_CAST_EXPR": 27,
    "CXX_NEW_EXPR": 28,
    "CXX_DELETE_EXPR": 28,
    "LAMBDA_EXPR": 29,
    "PAREN_EXPR": 29,
    # 30-39: type and reference nodes
    "TYPE_REF": 30,
    "TEMPLATE_REF": 32,
    "NAMESPACE_REF": 33,
    "DECL_REF_EXPR": 83,
    "OVERLOADED_DECL_REF": 83,
    "MEMBER_REF": 83,
    # 40-49: literals
    "INTEGER_LITERAL": 40,
    "FLOATING_LITERAL": 40,
    "IMAGINARY_LITERAL": 40,
    "STRING_LITERAL": 41,
    "CHARACTER_LITERAL": 42,
    "CXX_BOOL_LITERAL_EXPR": 43,
    "CXX_NULL_PTR_LITERAL_EXPR": 44,
    # 80-89: misc/preprocessor
    "INCLUSION_DIRECTIVE": 81,
    "MACRO_DEFINITION": 81,
    "MACRO_INSTANTIATION": 81,
    "TRANSLATION_UNIT": 82,
}


def _bucket_clang_cursor_kind(kind) -> int:
    name = getattr(kind, "name", str(kind))
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    return _CLANG_CURSOR_BUCKETS.get(name, 255)


def extract_clang_ast_metadata(
    source: str,
    tu: TranslationUnit,
    filename: str,
    source_encoding: str = "utf-8",
) -> tuple[list[int], list[int], list[int]]:
    """Extract per-character AST metadata from clang cursor extents."""
    text_len = len(source)
    ast_depth = [0] * text_len
    sibling_index = [0] * text_len
    ast_node_type = [0] * text_len
    if text_len == 0 or tu is None:
        return ast_depth, sibling_index, ast_node_type

    byte_to_char = _byte_to_char_mapper(source, source_encoding)
    stack: list[tuple[Cursor, int, int]] = [(tu.cursor, 0, 0)]
    while stack:
        cursor, depth, sib_idx = stack.pop()
        offsets = _cursor_extent_offsets(cursor, filename, byte_to_char)
        if offsets is not None:
            start, end = offsets
            end = min(end, text_len)
            bucket = _bucket_clang_cursor_kind(_cursor_kind(cursor))
            clamped_depth = min(depth, 65535)
            clamped_sib = min(sib_idx, 65535)
            for char_idx in range(start, end):
                ast_depth[char_idx] = clamped_depth
                sibling_index[char_idx] = clamped_sib
                ast_node_type[char_idx] = bucket

        children = list(cursor.get_children())
        child_depth = depth + 1
        for child_sib in range(len(children) - 1, -1, -1):
            stack.append((children[child_sib], child_depth, child_sib))

    return ast_depth, sibling_index, ast_node_type


def _slice_metadata(values: list[int], start: int, end: int) -> list[int]:
    if not values:
        return []
    return list(values[start:end])


def _cursor_text_and_metadata(
    cursor: Cursor,
    source: str,
    filename: str,
    byte_to_char: Callable[[int], int],
    ast_depth: list[int],
    sibling_index: list[int],
    ast_node_type: list[int],
) -> tuple[str, list[int], list[int], list[int], tuple[int, int] | None]:
    offsets = _cursor_extent_offsets(cursor, filename, byte_to_char)
    if offsets is None:
        return "", [], [], [], None
    start, end = offsets
    if start < 0 or end > len(source) or start >= end:
        return "", [], [], [], None
    return (
        source[start:end],
        _slice_metadata(ast_depth, start, end),
        _slice_metadata(sibling_index, start, end),
        _slice_metadata(ast_node_type, start, end),
        offsets,
    )


def extract_callees(cursor: Cursor) -> tuple[list[str], list[str]]:
    """Extract function call references from a cursor's children.

    Returns ``(callees, baselib_callees)``:

    * ``callees`` — normal resolvable callees (UNCHANGED filter: system/stdlib
      prefixes dropped). Feeds local dep-resolution + call-edges exactly as
      before, so OFF behavior is byte-identical.
    * ``baselib_callees`` — callees that were dropped by the normal filter BUT
      are cross-linkable base-lib namespaces (std::/boost::). Captured here in the
      SAME walk so the cross-repo linker can resolve them against the global
      symbol index WITHOUT touching the normal callee path. Ignored entirely when
      cross-repo linking is disabled.
    """
    callees: set[str] = set()
    baselib_callees: set[str] = set()

    def walk(node: Cursor):
        if _cursor_kind(node) == CursorKind.CALL_EXPR:
            ref = node.referenced
            if ref and ref.spelling:
                # Get fully qualified name
                qname = get_qualified_name(ref)
                if qname:
                    if not is_system_function(qname):
                        callees.add(qname)
                    elif is_crosslinkable_baselib(qname):
                        baselib_callees.add(qname)
        for child in node.get_children():
            walk(child)

    walk(cursor)
    return list(callees), list(baselib_callees)


def extract_callee_symbol_keys(
    cursor: Cursor,
    *,
    project_dir: str | None = None,
    repo: str | None = None,
    fallback_file: str | None = None,
) -> list[str]:
    """Extract canonical identity keys for non-system call targets."""
    return [
        str(ref["symbol_key"])
        for ref in extract_callee_references(
            cursor,
            project_dir=project_dir,
            project_id=repo,
            fallback_file=fallback_file,
        )[0]
    ]


def extract_callee_references(
    cursor: Cursor,
    *,
    project_dir: str | None = None,
    project_id: str | None = None,
    fallback_file: str | None = None,
    external_reference_omissions: ExternalReferenceOmissions | None = None,
) -> tuple[list[SymbolReference], list[SymbolReference]]:
    """Return resolved local and base-library call references from one AST walk."""
    local: dict[str, SymbolReference] = {}
    baselib: dict[str, SymbolReference] = {}

    def walk(node: Cursor):
        if _cursor_kind(node) == CursorKind.CALL_EXPR:
            ref = node.referenced
            if ref and ref.spelling:
                qname = get_qualified_name(ref)
                if qname:
                    reference = _optional_symbol_reference_for_cursor(
                        ref,
                        relation="call",
                        omissions=external_reference_omissions,
                        project_dir=project_dir,
                        project_id=project_id,
                        fallback_file=fallback_file,
                    )
                    if reference is not None:
                        key = str(reference["symbol_key"])
                        if not is_system_function(qname):
                            local[key] = reference
                        elif is_crosslinkable_baselib(qname):
                            baselib[key] = reference
        for child in node.get_children():
            walk(child)

    walk(cursor)
    return (
        [local[key] for key in sorted(local)],
        [baselib[key] for key in sorted(baselib)],
    )


_REFERENCED_TYPE_DECL_KINDS = frozenset({
    CursorKind.STRUCT_DECL,
    CursorKind.CLASS_DECL,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
    CursorKind.ENUM_DECL,
    CursorKind.UNION_DECL,
    CursorKind.TYPEDEF_DECL,
    CursorKind.TYPE_ALIAS_DECL,
    CursorKind.TYPE_ALIAS_TEMPLATE_DECL,
})
_REFERENCED_TYPE_CURSOR_KINDS = frozenset({
    CursorKind.TYPE_REF,
    CursorKind.TEMPLATE_REF,
})


def extract_referenced_types(cursor: Cursor) -> list[str]:
    """Extract qualified names of record/enum/typedef types referenced by a
    function (params, return, locals, member access, casts, template args).

    Mirrors :func:`extract_callees` but for TYPE relationships — captured during
    the SAME libclang parse pass and persisted on ``FunctionDef.referenced_types``
    so the offline doc-build path can emit ``type_refs``/``type_edges``.
    TYPE_REF and TEMPLATE_REF cursors are clang's explicit type-usage markers;
    we resolve each to its declaring record/enum/typedef qname and drop system
    types.
    """
    types: set[str] = set()

    def walk(node: Cursor):
        if _cursor_kind(node) in _REFERENCED_TYPE_CURSOR_KINDS:
            ref = node.referenced
            if (
                ref is not None
                and ref.spelling
                and _cursor_kind(ref) in _REFERENCED_TYPE_DECL_KINDS
            ):
                qname = get_qualified_name(ref)
                if qname and not is_system_function(qname):
                    types.add(qname)
        for child in node.get_children():
            walk(child)

    walk(cursor)
    return list(types)


def extract_referenced_type_keys(
    cursor: Cursor,
    *,
    project_dir: str | None = None,
    repo: str | None = None,
    fallback_file: str | None = None,
) -> list[str]:
    """Extract canonical identity keys of referenced record/enum/typedef types."""
    return [
        str(ref["symbol_key"])
        for ref in extract_referenced_type_references(
            cursor,
            project_dir=project_dir,
            project_id=repo,
            fallback_file=fallback_file,
        )
    ]


def extract_referenced_type_references(
    cursor: Cursor,
    *,
    project_dir: str | None = None,
    project_id: str | None = None,
    fallback_file: str | None = None,
    external_reference_omissions: ExternalReferenceOmissions | None = None,
) -> list[SymbolReference]:
    """Return canonical references for non-system record/enum/typedef uses."""
    refs: dict[str, SymbolReference] = {}

    def walk(node: Cursor):
        if _cursor_kind(node) in _REFERENCED_TYPE_CURSOR_KINDS:
            ref = node.referenced
            if (
                ref is not None
                and ref.spelling
                and _cursor_kind(ref) in _REFERENCED_TYPE_DECL_KINDS
            ):
                qname = get_qualified_name(ref)
                if qname and not is_system_function(qname):
                    reference = _optional_symbol_reference_for_cursor(
                        ref,
                        relation="type",
                        omissions=external_reference_omissions,
                        project_dir=project_dir,
                        project_id=project_id,
                        fallback_file=fallback_file,
                    )
                    if reference is not None:
                        refs[str(reference["symbol_key"])] = reference
        for child in node.get_children():
            walk(child)

    walk(cursor)
    return [refs[key] for key in sorted(refs)]


def get_qualified_name(cursor: Cursor) -> str:
    """Get the fully qualified name of a cursor (namespace::class::func)."""
    parts = []
    c = cursor
    while c and _cursor_kind(c) != CursorKind.TRANSLATION_UNIT:
        if c.spelling:
            parts.append(c.spelling)
        c = c.semantic_parent
    parts.reverse()
    return normalize_inline_namespace_qname('::'.join(parts))


def extract_preamble(tu: TranslationUnit, filename: str) -> str:
    """Extract #include directives and forward declarations from a file."""
    preamble_parts = []
    for cursor in tu.cursor.get_children():
        if cursor.location.file and cursor.location.file.name != filename:
            continue
        if _cursor_kind(cursor) in (
            CursorKind.INCLUSION_DIRECTIVE,
            CursorKind.USING_DIRECTIVE,
            CursorKind.USING_DECLARATION,
            CursorKind.TYPEDEF_DECL,
            CursorKind.TYPE_ALIAS_DECL,
            CursorKind.NAMESPACE_ALIAS,
        ):
            text = get_function_text(cursor, tu)
            if text:
                preamble_parts.append(text)
    return '\n'.join(preamble_parts)


FUNCTION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.CONVERSION_FUNCTION,
}

CONTAINER_KINDS = {
    CursorKind.NAMESPACE,
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
}


def _cursor_kind_or_none(name: str):
    return getattr(CursorKind, name, None)

# Cursor kinds whose DEFINITIONS we register as TypeDef chunks (type-edge
# targets). class/struct/union map to node-bucket kind 4; typedef/using map to
# kind 7 (see _CLANG_PART_NODE_BUCKETS). Enums get kind 4 as well (record-like).
_TYPE_DEF_KIND_BUCKET = {
    CursorKind.STRUCT_DECL: 4,
    CursorKind.CLASS_DECL: 4,
    CursorKind.CLASS_TEMPLATE: 4,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION: 4,
    CursorKind.UNION_DECL: 4,
    CursorKind.ENUM_DECL: 4,
    CursorKind.TYPEDEF_DECL: 7,
    CursorKind.TYPE_ALIAS_DECL: 7,
    CursorKind.TYPE_ALIAS_TEMPLATE_DECL: 7,
}
_CONCEPT_DECL_KIND = _cursor_kind_or_none("CONCEPT_DECL")
_UNEXPOSED_DECL_KIND = _cursor_kind_or_none("UNEXPOSED_DECL")


def _header_decl_kind(cursor, text: str, *, in_primary_file: bool) -> int | None:
    """Return a structure kind for trainable non-type header declarations."""
    if not in_primary_file:
        return None
    kind = _cursor_kind(cursor)
    if kind == _CONCEPT_DECL_KIND:
        return HEADER_FRAGMENT_KIND
    if kind not in {CursorKind.VAR_DECL, _UNEXPOSED_DECL_KIND}:
        return None
    stripped = text.lstrip()
    padded = f" {stripped} "
    if stripped.startswith("template ") and (
        " inline " in padded or " constexpr " in padded
    ):
        return HEADER_FRAGMENT_KIND
    if stripped.startswith("inline constexpr ") or stripped.startswith("constexpr inline "):
        return HEADER_FRAGMENT_KIND
    return None


def parse_translation_unit(
    filepath: str,
    index: Index,
    compile_args: list[str],
    project_dir: str,
    allow_system_types: bool = False,
    *,
    project_id: str,
    external_reference_omissions: ExternalReferenceOmissions | None = None,
    allow_include_recovery: bool = False,
    parse_recovery_records: list[dict[str, object]] | None = None,
) -> tuple[list[FunctionDef], list[TypeDef]]:
    """Parse a single C/C++ file (source OR header) and extract function
    definitions (with callees + referenced types) AND type definitions
    (struct/class/enum/union/typedef/using).

    Type definitions are returned alongside functions so the index can register
    them as type-edge target chunks. Headers are parsed as standalone TUs.

    ``allow_system_types`` (default False): the type-def branch normally DROPS
    names that look like system/stdlib symbols (``is_system_function`` — e.g.
    anything under ``std::``/``boost::``) so the NORMAL repo indexer never indexes
    the standard library's own types. The cross-repo GLOBAL symbol-index builder
    (scripts/crossrepo/build_global_symbol_index.py), when indexing the C++
    standard library itself (libc++/libstdc++), sets this True so ``std::vector``,
    ``std::map``, ``std::basic_string``, ... ARE captured as type definitions.
    Type definitions are always attributed to their ACTUAL project-local defining
    file (not the primary TU path) so a class/template reached through a .cc
    include still emits as a standalone header fragment.  When True we apply the
    same actual-file attribution to std/base-lib symbols too.
    """
    functions: list[FunctionDef] = []
    typedefs: list[TypeDef] = []
    stable_project_id = require_project_identity(
        project_id,
        source=f"parse_translation_unit({filepath})",
    )
    try:
        source, _source_bytes, source_encoding = _read_source_file(filepath)
    except OSError:
        return functions, typedefs

    tu = _load_translation_unit_with_include_recovery(
        filepath,
        index,
        compile_args,
        project_dir,
        allow_include_recovery=allow_include_recovery,
        parse_recovery_records=parse_recovery_records,
    )

    rel_path = os.path.relpath(filepath, project_dir)
    ast_depth, sibling_index, ast_node_type = extract_clang_ast_metadata(
        source,
        tu,
        filepath,
        source_encoding,
    )
    semantic_meta = extract_semantic_metadata(
        source,
        tu,
        filepath,
        project_dir=project_dir,
        project_id=stable_project_id,
        fallback_file=rel_path,
        source_encoding=source_encoding,
        external_reference_omissions=external_reference_omissions,
    )
    byte_to_char = _byte_to_char_mapper(source, source_encoding)
    project_dir_abs = os.path.abspath(project_dir)

    def _in_project(cursor) -> bool:
        """True if the cursor's location file lives under project_dir.

        Type DEFINITIONS usually live in project HEADERS that get #included into
        the .cpp TU, so we cannot restrict to ``== filepath`` for them or we'd
        capture nothing. We DO exclude system/third-party headers (outside
        project_dir) to avoid indexing std:: types.
        """
        loc = cursor.location.file
        if loc is None:
            return False
        return os.path.abspath(loc.name).startswith(project_dir_abs + os.sep)

    def _cursor_rel_file(cursor) -> str:
        """The cursor's ACTUAL defining file, relative to project_dir.

        Type definitions reached through includes must keep their header
        provenance (correct standalone header docs + stable dedup key). Falls
        back to the primary ``rel_path``.
        """
        loc = cursor.location.file
        if loc is None:
            return rel_path
        try:
            return os.path.relpath(loc.name, project_dir)
        except SymbolIdentityError:
            raise
        except Exception:
            return rel_path

    def visit(cursor):
        """Recursively visit cursors, descending into namespaces and classes.

        Functions are captured only from the primary file (``== filepath``) to
        avoid duplicating inline defs across every TU that includes the header.
        Type DEFINITIONS are captured from any project-local file (incl. headers
        pulled in via #include); duplicates across TUs are deduped by canonical
        symbol identity in ``ProjectIndex.add_typedef``.
        """
        loc = cursor.location.file
        if loc is None:
            return
        in_primary_file = loc.name == filepath
        in_project = in_primary_file or _in_project(cursor)
        if not in_project:
            return  # skip system/third-party headers entirely

        kind = _cursor_kind(cursor)
        if in_primary_file and kind in FUNCTION_KINDS and cursor.is_definition():
            text, func_ast_depth, func_sibling_index, func_ast_node_type, offsets = (
                _cursor_text_and_metadata(
                    cursor,
                    source,
                    filepath,
                    byte_to_char,
                    ast_depth,
                    sibling_index,
                    ast_node_type,
                )
            )
            if text and len(text) >= 20:
                callee_refs, baselib_callee_refs = extract_callee_references(
                    cursor,
                    project_dir=project_dir,
                    project_id=stable_project_id,
                    fallback_file=rel_path,
                    external_reference_omissions=external_reference_omissions,
                )
                referenced_type_refs = extract_referenced_type_references(
                    cursor,
                    project_dir=project_dir,
                    project_id=stable_project_id,
                    fallback_file=rel_path,
                    external_reference_omissions=external_reference_omissions,
                )
                callees = [str(ref["qname"]) for ref in callee_refs]
                baselib_callees = [str(ref["qname"]) for ref in baselib_callee_refs]
                callee_keys = [str(ref["symbol_key"]) for ref in callee_refs]
                referenced_types = [str(ref["qname"]) for ref in referenced_type_refs]
                referenced_type_keys = [
                    str(ref["symbol_key"]) for ref in referenced_type_refs
                ]
                qname = _canonicalize_clang_identity_text(
                    get_qualified_name(cursor),
                    project_dir,
                )
                symbol_key, usr, canonical_signature = symbol_identity_for_cursor(
                    cursor,
                    project_dir=project_dir,
                    project=stable_project_id,
                    fallback_file=rel_path,
                )
                sem_symbol_ids: list[int] = []
                sem_call_targets: list[int] = []
                sem_type_refs: list[int] = []
                sem_def_use: list[int] = []
                if offsets is not None:
                    s0, s1 = offsets
                    sem_symbol_ids = list(semantic_meta["symbol_ids"][s0:s1])
                    sem_call_targets = list(semantic_meta["call_targets"][s0:s1])
                    sem_type_refs = list(semantic_meta["type_refs"][s0:s1])
                    sem_def_use = list(semantic_meta["def_use"][s0:s1])
                functions.append(FunctionDef(
                    name=cursor.spelling,
                    qualified_name=qname,
                    file=(_cursor_rel_file(cursor) if allow_system_types else rel_path),
                    line=cursor.location.line,
                    column=getattr(cursor.location, "column", 0),
                    text=text,
                    callees=callees,
                    is_definition=True,
                    end_line=cursor.extent.end.line,
                    ast_depth=func_ast_depth,
                    sibling_index=func_sibling_index,
                    ast_node_type=func_ast_node_type,
                    referenced_types=referenced_types,
                    baselib_callees=baselib_callees,
                    symbol_key=symbol_key,
                    usr=usr,
                    canonical_signature=canonical_signature,
                    symbol_kind=_cursor_kind_name(cursor),
                    callee_keys=callee_keys,
                    referenced_type_keys=referenced_type_keys,
                    callee_refs=callee_refs,
                    baselib_callee_refs=baselib_callee_refs,
                    referenced_type_refs=referenced_type_refs,
                    semantic_symbol_ids=sem_symbol_ids,
                    semantic_call_targets=sem_call_targets,
                    semantic_type_refs=sem_type_refs,
                    semantic_def_use=sem_def_use,
                    semantic_symbol_identities=_semantic_identity_records_for_arrays(
                        semantic_meta,
                        sem_symbol_ids,
                        sem_call_targets,
                        sem_type_refs,
                        source=f"{rel_path}:{cursor.location.line}:{qname}",
                    ),
                ))
            return

        # Register type DEFINITIONS (struct/class/enum/union/typedef/using) as
        # type-edge target chunks. We require an actual definition so a forward
        # declaration never shadows the real body.
        type_bucket = _TYPE_DEF_KIND_BUCKET.get(kind)
        if type_bucket is not None and cursor.is_definition():
            qname = _canonicalize_clang_identity_text(
                get_qualified_name(cursor),
                project_dir,
            )
            # Normal path drops std::/boost:: ("system") type names; the std
            # cross-link builder sets allow_system_types so std:: class templates
            # (std::vector/std::map/std::basic_string/...) ARE captured.
            if qname and (allow_system_types or not is_system_function(qname)):
                td_text = get_function_text(cursor, tu)
                if td_text and len(td_text) >= 8:
                    symbol_key, usr, canonical_signature = symbol_identity_for_cursor(
                        cursor,
                        project_dir=project_dir,
                        project=stable_project_id,
                        fallback_file=_cursor_rel_file(cursor),
                    )
                    typedefs.append(TypeDef(
                        name=cursor.spelling,
                        qualified_name=qname,
                        file=_cursor_rel_file(cursor),
                        line=cursor.location.line,
                        column=getattr(cursor.location, "column", 0),
                        text=td_text,
                        kind=type_bucket,
                        end_line=cursor.extent.end.line,
                        symbol_key=symbol_key,
                        usr=usr,
                        canonical_signature=canonical_signature,
                        symbol_kind=_cursor_kind_name(cursor),
                    ))
            # Records (class/struct) can contain nested types/methods — recurse.

        header_decl_text = ""
        header_decl_kind = None
        if kind in {CursorKind.VAR_DECL, _UNEXPOSED_DECL_KIND, _CONCEPT_DECL_KIND}:
            header_decl_text = get_function_text(cursor, tu)
            if header_decl_text:
                header_decl_kind = _header_decl_kind(
                    cursor,
                    header_decl_text,
                    in_primary_file=in_primary_file,
                )
        if header_decl_kind is not None and header_decl_text and len(header_decl_text) >= 8:
            qname = _canonicalize_clang_identity_text(
                get_qualified_name(cursor),
                project_dir,
            )
            if qname and (allow_system_types or not is_system_function(qname)):
                symbol_key, usr, canonical_signature = symbol_identity_for_cursor(
                    cursor,
                    project_dir=project_dir,
                    project=stable_project_id,
                    fallback_file=_cursor_rel_file(cursor),
                )
                typedefs.append(TypeDef(
                    name=cursor.spelling,
                    qualified_name=qname,
                    file=_cursor_rel_file(cursor),
                    line=cursor.location.line,
                    column=getattr(cursor.location, "column", 0),
                    text=header_decl_text,
                    kind=header_decl_kind,
                    end_line=cursor.extent.end.line,
                    symbol_key=symbol_key,
                    usr=usr,
                    canonical_signature=canonical_signature,
                    symbol_kind=_cursor_kind_name(cursor),
                ))

        if kind in CONTAINER_KINDS:
            # Recurse into namespaces, classes, structs for nested defs.
            for child in cursor.get_children():
                visit(child)

    for cursor in tu.cursor.get_children():
        visit(cursor)

    return functions, typedefs


# Corpus callers may explicitly exclude extraction/build artifacts, but source
# categories such as tests, examples, vendored libraries, fuzzers, and docs are
# real C/C++ and must never disappear because of a hidden indexer policy.
_DEFAULT_SKIP_DIRS = frozenset({'.git'})

C_EXTENSIONS = {'.c'}

SOURCE_TREE_ENTRY_EXCLUSIONS_SCHEMA = "cppmega.source_tree_entry_exclusions_v1"
_SOURCE_TREE_ENTRY_EXCLUSIONS_POLICY = (
    "skip_only_dangling_symlink_blobs_targeting_unmaterialized_gitlinks"
)


def _canonical_receipt_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class GitTreeSourceEntryExclusions:
    """Prove the only safe-to-skip non-regular Git checkout entries."""

    def __init__(self, project_dir: str | os.PathLike[str]) -> None:
        self.project_dir = os.path.abspath(os.fspath(project_dir))
        self._git_tree: str | None = None
        self._index_entries: dict[str, tuple[str, str, str]] | None = None
        self._records: dict[str, dict[str, object]] = {}

    def _git(self, *args: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", self.project_dir, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr[-4000:].decode(errors="replace")
            raise OSError(
                "cannot prove non-regular source entry against pinned Git "
                f"checkout: git {' '.join(args)}: {detail}"
            )
        return completed.stdout

    def _load_git_index(self) -> None:
        if self._index_entries is not None:
            return
        raw_tree = self._git("rev-parse", "HEAD^{tree}").strip()
        if re.fullmatch(rb"[0-9a-f]{40}(?:[0-9a-f]{24})?", raw_tree) is None:
            raise OSError("pinned Git checkout returned an invalid tree id")
        self._git_tree = raw_tree.decode("ascii")
        entries: dict[str, tuple[str, str, str]] = {}
        for encoded in self._git("ls-files", "--stage", "-z").split(b"\0"):
            if not encoded:
                continue
            metadata, separator, raw_path = encoded.partition(b"\t")
            fields = metadata.split(b" ")
            if not separator or len(fields) != 3:
                raise OSError("pinned Git index returned a malformed stage entry")
            mode, object_id, stage = fields
            try:
                relative_path = os.fsdecode(raw_path)
                mode_text = mode.decode("ascii")
                object_text = object_id.decode("ascii")
                stage_text = stage.decode("ascii")
            except UnicodeError as exc:
                raise OSError("pinned Git index entry metadata is invalid") from exc
            if relative_path in entries:
                raise OSError(
                    f"pinned Git index contains duplicate stages for {relative_path}"
                )
            entries[relative_path] = (mode_text, object_text, stage_text)
        self._index_entries = entries

    def _clean_status(self, relative_path: str, gitlink_path: str) -> None:
        status = self._git(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=no",
            "--",
            relative_path,
            gitlink_path,
        )
        if status:
            raise OSError(
                "source entry or target gitlink changed during immutable inspection"
            )

    def record(self, filepath: str, _stat_error: OSError) -> None:
        absolute_path = os.path.abspath(filepath)
        try:
            if os.path.commonpath((self.project_dir, absolute_path)) != self.project_dir:
                raise OSError("source candidate escapes project root")
        except ValueError as exc:
            raise OSError("source candidate has an invalid project-relative path") from exc
        relative_path = Path(
            os.path.relpath(absolute_path, self.project_dir)
        ).as_posix()

        try:
            before = os.lstat(absolute_path)
        except OSError as exc:
            raise OSError(
                "source candidate vanished instead of remaining a provable "
                f"non-regular tree entry: {relative_path}"
            ) from exc
        if not stat.S_ISLNK(before.st_mode) or os.path.exists(absolute_path):
            raise OSError(
                "unreadable source candidate is not a dangling symlink tree entry: "
                f"{relative_path}"
            )
        symlink_target = os.readlink(absolute_path)
        symlink_target_bytes = os.fsencode(symlink_target)
        if b"\0" in symlink_target_bytes:
            raise OSError("Git symlink target unexpectedly contains NUL")

        self._load_git_index()
        assert self._index_entries is not None
        assert self._git_tree is not None
        index_entry = self._index_entries.get(relative_path)
        if index_entry is None:
            raise OSError(
                f"dangling source symlink is not tracked by Git: {relative_path}"
            )
        entry_mode, entry_object_id, entry_stage = index_entry
        if entry_mode != "120000" or entry_stage != "0":
            raise OSError(
                "unreadable source candidate is not an exact stage-0 mode-120000 "
                f"Git entry: {relative_path}"
            )
        object_type = self._git("cat-file", "-t", entry_object_id).strip()
        if object_type != b"blob":
            raise OSError(f"Git symlink object is not a blob: {relative_path}")
        object_payload = self._git("cat-file", "blob", entry_object_id)
        if object_payload != symlink_target_bytes:
            raise OSError(
                f"Git symlink blob and checkout target differ: {relative_path}"
            )

        if os.path.isabs(symlink_target):
            raise OSError(f"absolute dangling source symlink is unsupported: {relative_path}")
        target_relative_path = posixpath.normpath(
            posixpath.join(posixpath.dirname(relative_path), symlink_target)
        )
        if target_relative_path == ".." or target_relative_path.startswith("../"):
            raise OSError(f"dangling source symlink escapes checkout: {relative_path}")
        if os.path.lexists(os.path.join(self.project_dir, target_relative_path)):
            raise OSError(
                f"dangling source target unexpectedly materialized: {relative_path}"
            )

        target_parts = target_relative_path.split("/")
        gitlink_path = ""
        gitlink_entry: tuple[str, str, str] | None = None
        for length in range(len(target_parts) - 1, 0, -1):
            candidate = "/".join(target_parts[:length])
            candidate_entry = self._index_entries.get(candidate)
            if candidate_entry is not None and candidate_entry[0] == "160000":
                gitlink_path = candidate
                gitlink_entry = candidate_entry
                break
        if gitlink_entry is None or gitlink_entry[2] != "0":
            raise OSError(
                "dangling source symlink is not beneath an exact stage-0 Git "
                f"gitlink: {relative_path} -> {target_relative_path}"
            )

        self._clean_status(relative_path, gitlink_path)
        try:
            after = os.lstat(absolute_path)
            after_target = os.readlink(absolute_path)
        except OSError as exc:
            raise OSError(
                f"source symlink changed during immutable inspection: {relative_path}"
            ) from exc
        stable_fields = ("st_mode", "st_ino", "st_dev", "st_size", "st_mtime_ns")
        if (
            any(getattr(before, field) != getattr(after, field) for field in stable_fields)
            or after_target != symlink_target
            or os.path.exists(absolute_path)
        ):
            raise OSError(
                f"source symlink eligibility changed during inspection: {relative_path}"
            )
        self._clean_status(relative_path, gitlink_path)

        record = {
            "relative_path": relative_path,
            "reason": "dangling_symlink_target_below_unmaterialized_gitlink",
            "git_tree": self._git_tree,
            "entry_mode": entry_mode,
            "entry_object_id": entry_object_id,
            "entry_object_type": "blob",
            "entry_object_size_bytes": len(object_payload),
            "entry_object_sha256": hashlib.sha256(object_payload).hexdigest(),
            "symlink_target": symlink_target,
            "target_relative_path": target_relative_path,
            "target_gitlink_path": gitlink_path,
            "target_gitlink_mode": gitlink_entry[0],
            "target_gitlink_commit": gitlink_entry[1],
        }
        previous = self._records.setdefault(relative_path, record)
        if previous != record:
            raise OSError(f"source entry exclusion receipt drifted: {relative_path}")

    def receipt(self) -> dict[str, object]:
        records = [self._records[key] for key in sorted(self._records)]
        return {
            "schema": SOURCE_TREE_ENTRY_EXCLUSIONS_SCHEMA,
            "status": "complete",
            "policy": _SOURCE_TREE_ENTRY_EXCLUSIONS_POLICY,
            "git_tree": self._git_tree if records else None,
            "excluded_count": len(records),
            "records_sha256": _canonical_receipt_sha256(records),
            "records": records,
        }


def find_cpp_files(
    project_dir: str,
    extra_exclude_dirs: set[str] | None = None,
    *,
    ineligible_entry_handler: Callable[[str, OSError], None] | None = None,
) -> list[str]:
    """Find all C/C++ source files in a directory."""
    skip_dirs = _DEFAULT_SKIP_DIRS | (extra_exclude_dirs or set())
    files = []
    for root, dirs, filenames in os.walk(project_dir):
        # Prune recursion in-place — os.walk won't descend into skipped dirs
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in INDEX_EXTENSIONS:
                filepath = os.path.join(root, fname)
                try:
                    os.path.getsize(filepath)
                except OSError as exc:
                    if ineligible_entry_handler is not None:
                        try:
                            ineligible_entry_handler(filepath, exc)
                        except OSError as proof_error:
                            raise OSError(
                                f"failed to stat C/C++ input {filepath}: {exc}; "
                                f"non-regular entry proof failed: {proof_error}"
                            ) from proof_error
                        continue
                    raise OSError(
                        f"failed to stat C/C++ input {filepath}: {exc}"
                    ) from exc
                files.append(filepath)
    return files


def find_build_files(
    project_dir: str,
    extra_exclude_dirs: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Find build/domain text files. Sibling of ``find_cpp_files``.

    Returns a list of ``(abs_path, build_kind)`` tuples so the build-system tag
    (cmake/make/bazel/ninja/meson/...) is known at discovery time. Compilation
    databases are also consumed by ``load_compile_commands`` but remain useful
    training evidence for the exact compiler invocation, so they are emitted on
    their dedicated frozen domain as well. Build files legitimately live under
    dirs that code discovery prunes (e.g.
    ``third_party/`` for vendored CMake), so only ``.git`` and the caller's extra
    excludes are pruned here.
    """
    skip_dirs = {'.git'} | (extra_exclude_dirs or set())
    files: list[tuple[str, str]] = []
    for root, dirs, filenames in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in filenames:
            build_kind = classify_build_file(fname)
            if build_kind is None:
                continue
            filepath = os.path.join(root, fname)
            try:
                _ = os.path.getsize(filepath)
            except OSError as exc:
                raise OSError(f"failed to stat build/domain input {filepath}: {exc}") from exc
            files.append((filepath, build_kind))
    return files


def _classify_shell_file(filepath: str, fname: str) -> str | None:
    from cppmega.data.domain_ingestion import decode_domain_prefix

    if classify_build_file(fname) is not None:
        return None
    suffix = os.path.splitext(fname)[1].lower()
    extension_kind = SHELL_EXT_KINDS.get(suffix)
    if extension_kind is None and suffix:
        return None
    try:
        with Path(filepath).open("rb") as fh:
            first_line_bytes = fh.readline(1024)
            prefix_reaches_eof = fh.tell() == os.fstat(fh.fileno()).st_size
    except OSError as exc:
        if extension_kind is not None:
            raise OSError(f"failed to read shell/domain input {filepath}: {exc}") from exc
        return None
    try:
        first_line = decode_domain_prefix(
            first_line_bytes,
            path=filepath,
            allow_trailing_nul=prefix_reaches_eof,
        ).lower()
    except ValueError:
        if extension_kind is not None:
            raise
        return None
    if first_line.startswith("#!"):
        words = set(re.findall(r"[a-z0-9_+.-]+", first_line))
        for shell in (
            "powershell",
            "pwsh",
            "tcsh",
            "csh",
            "zsh",
            "bash",
            "ksh",
            "sh",
        ):
            if shell in words:
                if shell in {"powershell", "pwsh"}:
                    return "powershell"
                return "tcsh" if shell == "csh" else shell
    return extension_kind


def find_shell_files(
    project_dir: str,
    extra_exclude_dirs: set[str] | None = None,
    invalid_input_handler: Callable[[Path, ValueError], None] | None = None,
    excluded_paths: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Find shell sources, including extensionless files with a shell shebang."""

    skip_dirs = {'.git'} | (extra_exclude_dirs or set())
    excluded = {os.path.abspath(path) for path in (excluded_paths or set())}
    files: list[tuple[str, str]] = []
    for root, dirs, filenames in os.walk(project_dir):
        dirs[:] = [directory for directory in dirs if directory not in skip_dirs]
        for fname in filenames:
            filepath = os.path.join(root, fname)
            if os.path.abspath(filepath) in excluded:
                continue
            try:
                shell_kind = _classify_shell_file(filepath, fname)
            except ValueError as exc:
                if invalid_input_handler is not None:
                    invalid_input_handler(Path(filepath), exc)
                    continue
                raise
            if shell_kind is None:
                continue
            try:
                _ = os.path.getsize(filepath)
            except OSError as exc:
                raise OSError(f"failed to stat shell/domain input {filepath}: {exc}") from exc
            files.append((filepath, shell_kind))
    return files


# ---------------------------------------------------------------------------
# Shell edge extraction (lexical, lightweight)
# ---------------------------------------------------------------------------

_SHELL_COMMAND_TOKEN_RE = re.compile(
    r"[A-Za-z_./][A-Za-z0-9_./:+-]*"
)

_SHELL_ASSIGNMENT_PREFIX_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*=[^\s]*[ \t]*"
)
_SHELL_CONTROL_WORDS = {
    "begin",
    "case",
    "do",
    "elif",
    "else",
    "end",
    "esac",
    "if",
    "in",
    "process",
    "then",
    "until",
    "while",
}

_SHELL_SOURCE_RE = re.compile(
    r"(?m)^[ \t]*(?:\.|source)[ \t]+([^\s;#]+)"
)

_SHELL_VAR_DEF_RE = re.compile(
    r"(?m)^[ \t]*(?:export[ \t]+|declare[ \t]+(?:-[a-zA-Z]+[ \t]+)*|local[ \t]+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)="
)

_SHELL_VAR_USE_RE = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)

_SHELL_COMMAND_FILE_RE = re.compile(
    r"(?m)^[ \t]*([A-Za-z_][A-Za-z0-9_./-]*)"  # command
    r"(?:[ \t]+-{1,2}[A-Za-z0-9_-]*(?:=[^\s]*)?)*"  # optional flags
    r"[ \t]+([^\s;|&<>#]+\.[A-Za-z0-9]+)"  # file argument with extension
)

_PS_CMDLET_RE = re.compile(
    r"(?m)^[ \t]*([A-Z][a-z]+-[A-Z][A-Za-z]+)"  # Verb-Noun cmdlet
    r"((?:[ \t]+-[A-Za-z]+[ \t]+[^\s;#|]+)*)"   # -Param Value pairs
)

_PS_PARAM_RE = re.compile(
    r"-([A-Za-z]+)[ \t]+([^\s;#|]+)"
)

_TCSH_SETENV_RE = re.compile(
    r"(?m)^[ \t]*setenv[ \t]+([A-Za-z_][A-Za-z0-9_]*)"
)

_TCSH_VAR_USE_RE = re.compile(
    r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?"
)

_PS_VAR_DEF_RE = re.compile(
    r"(?m)(?:^|[;{}])[ \t]*"
    r"(?P<variable>\$\{?(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\}?)"
    r"[ \t]*(?:=|\+=|-=|\*=|/=)"
)
_PS_TYPED_VAR_DEF_RE = re.compile(
    r"(?m)(?:^|[;{}])[ \t]*"
    r"(?:\[[^\]\r\n]+\][ \t]*)+"
    r"(?P<variable>\$\{?(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\}?)"
    r"[ \t]*(?:=|\+=|-=|\*=|/=)"
)
_PS_PARAM_START_RE = re.compile(r"(?i)\bparam[ \t\r\n]*\(")
_PS_PARAM_VAR_RE = re.compile(
    r"(?:^|,)[ \t\r\n]*"
    r"(?:\[[^\]\r\n]+\][ \t\r\n]*)*"
    r"(?P<variable>\$\{?(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\}?)"
)

_PS_VAR_USE_RE = re.compile(
    r"\$\{?([A-Za-z_][A-Za-z0-9_:]*)\}?"
)

_SHELL_REDIR_OP_RE = re.compile(
    r"(?P<op>(?:[0-9]+|&)?>{1,2}|<)"
)
_SHELL_HEREDOC_RE = re.compile(
    r"<<(?P<strip>-?)[ \t]*(?P<quote>['\"]?)"
    r"(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
)


def _powershell_variable_definitions(
    structural_text: str,
) -> Iterator[tuple[str, int]]:
    """Yield typed, untyped, and param-block PowerShell definitions."""

    seen: set[tuple[str, int]] = set()
    for pattern in (_PS_VAR_DEF_RE, _PS_TYPED_VAR_DEF_RE):
        for match in pattern.finditer(structural_text):
            item = (match.group("name").casefold(), match.start("variable"))
            if item not in seen:
                seen.add(item)
                yield item
    for param_start in _PS_PARAM_START_RE.finditer(structural_text):
        body_start = param_start.end()
        depth = 1
        cursor = body_start
        while cursor < len(structural_text) and depth:
            if structural_text[cursor] == "(":
                depth += 1
            elif structural_text[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth:
            continue
        body = structural_text[body_start : cursor - 1]
        for match in _PS_PARAM_VAR_RE.finditer(body):
            item = (
                match.group("name").casefold(),
                body_start + match.start("variable"),
            )
            if item not in seen:
                seen.add(item)
                yield item


def _shell_heredoc_regions(
    text: str,
) -> list[tuple[int, int, int, int, int, bool]]:
    """Return declaration/body/closing spans for POSIX-style heredocs."""

    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    def declarations(line: str, line_start: int) -> list[tuple[int, int, str, bool, bool]]:
        code = [True] * len(line)
        quote: str | None = None
        escaped = False
        comment = False
        for index, char in enumerate(line):
            if comment:
                code[index] = False
                continue
            if escaped:
                code[index] = False
                escaped = False
                continue
            if char == "\\" and quote != "'":
                code[index] = False
                escaped = True
                continue
            if quote is not None:
                code[index] = False
                if char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                code[index] = False
                quote = char
                continue
            if char == "#" and (
                index == 0
                or line[index - 1].isspace()
                or line[index - 1] in ";|&(){}"
            ):
                code[index] = False
                comment = True

        found: list[tuple[int, int, str, bool, bool]] = []
        for match in _SHELL_HEREDOC_RE.finditer(line):
            if match.start() >= len(code) or not code[match.start()]:
                continue
            if match.start() > 0 and line[match.start() - 1] == "<":
                continue
            prefix = line[: match.start()]
            if prefix.rfind("((") > prefix.rfind("))"):
                continue
            found.append(
                (
                    line_start + match.start(),
                    line_start + match.end(),
                    match.group("delimiter"),
                    match.group("strip") == "-",
                    not bool(match.group("quote")),
                )
            )
        return found

    regions: list[tuple[int, int, int, int, int, bool]] = []
    line_index = 0
    while line_index < len(lines):
        pending = declarations(lines[line_index], offsets[line_index])
        if not pending:
            line_index += 1
            continue
        scan_index = line_index + 1
        for declaration_start, declaration_end, delimiter, strip_tabs, interpolate in pending:
            body_start = offsets[scan_index] if scan_index < len(lines) else len(text)
            close_start = len(text)
            close_end = len(text)
            while scan_index < len(lines):
                candidate = lines[scan_index].rstrip("\r\n")
                comparison = candidate.lstrip("\t") if strip_tabs else candidate
                if comparison == delimiter:
                    close_start = offsets[scan_index]
                    close_end = close_start + len(lines[scan_index])
                    scan_index += 1
                    break
                scan_index += 1
            regions.append(
                (
                    declaration_start,
                    declaration_end,
                    body_start,
                    close_start,
                    close_end,
                    interpolate,
                )
            )
        line_index = max(line_index + 1, scan_index)
    return regions


def _unquoted_pipe_offsets(
    text: str,
    *,
    is_powershell: bool,
) -> list[int]:
    """Return single-pipe offsets outside quotes/comments (never ``||``)."""

    masked = _mask_shell_quotes_and_comment(
        text,
        is_powershell=is_powershell,
    )
    offsets: list[int] = []
    for index, char in enumerate(masked):
        if (
            char == "|"
            and (index == 0 or masked[index - 1] != "|")
            and (index + 1 == len(masked) or masked[index + 1] != "|")
        ):
            offsets.append(index)
    return offsets


def _powershell_comment_boundary(text: str, index: int) -> bool:
    return (
        index == 0
        or text[index - 1].isspace()
        or text[index - 1] in ";|&(){}[],=+-*/%!?:<>"
    )


def _mask_shell_quotes_and_comment(
    text: str,
    *,
    is_powershell: bool,
) -> str:
    """Blank shell strings/comments while preserving newlines and offsets."""

    masked = list(text)
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    here_string_end: str | None = None
    escape_char = "`" if is_powershell else "\\"
    index = 0
    while index < len(text):
        char = text[index]
        if block_comment:
            if text.startswith("#>", index):
                masked[index : index + 2] = [" ", " "]
                block_comment = False
                index += 2
                continue
            if char not in "\r\n":
                masked[index] = " "
            index += 1
            continue
        if here_string_end is not None:
            at_line_start = index == 0 or text[index - 1] in "\r\n"
            if at_line_start:
                line_end = len(text)
                for newline in ("\n", "\r"):
                    candidate = text.find(newline, index)
                    if candidate >= 0:
                        line_end = min(line_end, candidate)
                content = text[index:line_end]
                stripped = content.strip(" \t")
                if stripped == here_string_end:
                    for position in range(index, line_end):
                        masked[position] = " "
                    here_string_end = None
                    index = line_end
                    continue
            if char not in "\r\n":
                masked[index] = " "
            index += 1
            continue
        if line_comment:
            if char in "\r\n":
                line_comment = False
            else:
                masked[index] = " "
            index += 1
            continue
        if escaped:
            masked[index] = " "
            escaped = False
            index += 1
            continue
        if char == escape_char and quote != "'":
            masked[index] = " "
            escaped = True
            index += 1
            continue
        if quote is not None:
            if char not in "\r\n":
                masked[index] = " "
            if char == quote:
                if (
                    is_powershell
                    and quote == "'"
                    and index + 1 < len(text)
                    and text[index + 1] == "'"
                ):
                    masked[index + 1] = " "
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if (
            is_powershell
            and text.startswith("<#", index)
            and _powershell_comment_boundary(text, index)
        ):
            masked[index : index + 2] = [" ", " "]
            block_comment = True
            index += 2
            continue
        if is_powershell and text.startswith(("@'", '@"'), index):
            line_end = len(text)
            for newline in ("\n", "\r"):
                candidate = text.find(newline, index)
                if candidate >= 0:
                    line_end = min(line_end, candidate)
            if not text[index + 2 : line_end].strip(" \t"):
                here_string_end = "'@" if text[index + 1] == "'" else '"@'
                for position in range(index, line_end):
                    masked[position] = " "
                index = line_end
                continue
        if char in {"'", '"'}:
            masked[index] = " "
            quote = char
            index += 1
            continue
        if (
            char == "#"
            and not (
                is_powershell
                and index > 0
                and text[index - 1] == "<"
            )
            and (
            (is_powershell and _powershell_comment_boundary(text, index))
            or (not is_powershell and index == 0)
            or text[index - 1].isspace()
            or text[index - 1] in ";|&(){}"
            )
        ):
            masked[index] = " "
            line_comment = True
        index += 1
    if not is_powershell:
        for (
            declaration_start,
            declaration_end,
            body_start,
            _close_start,
            close_end,
            _interpolate,
        ) in _shell_heredoc_regions(text):
            for position in range(declaration_start, declaration_end):
                masked[position] = " "
            for position in range(body_start, close_end):
                if text[position] not in "\r\n":
                    masked[position] = " "
    return "".join(masked)


def _mask_powershell_non_interpolating_regions(text: str) -> str:
    """Keep live PowerShell variable syntax and blank non-interpolating regions."""

    masked = list(text)
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    here_string_end: str | None = None
    here_string_interpolates = False
    index = 0
    while index < len(text):
        char = text[index]
        if block_comment:
            if text.startswith("#>", index):
                masked[index : index + 2] = [" ", " "]
                block_comment = False
                index += 2
                continue
            if char not in "\r\n":
                masked[index] = " "
            index += 1
            continue
        if here_string_end is not None:
            at_line_start = index == 0 or text[index - 1] in "\r\n"
            if at_line_start:
                line_end = len(text)
                for newline in ("\n", "\r"):
                    candidate = text.find(newline, index)
                    if candidate >= 0:
                        line_end = min(line_end, candidate)
                if text[index:line_end].strip(" \t") == here_string_end:
                    for position in range(index, line_end):
                        masked[position] = " "
                    here_string_end = None
                    here_string_interpolates = False
                    index = line_end
                    continue
            if here_string_interpolates and char == "`":
                masked[index] = " "
                if index + 1 < len(text):
                    masked[index + 1] = " "
                    index += 2
                    continue
            if not here_string_interpolates and char not in "\r\n":
                masked[index] = " "
            index += 1
            continue
        if line_comment:
            if char in "\r\n":
                line_comment = False
            else:
                masked[index] = " "
            index += 1
            continue
        if escaped:
            masked[index] = " "
            escaped = False
            index += 1
            continue
        if char == "`" and quote != "'":
            masked[index] = " "
            escaped = True
            index += 1
            continue
        if quote == "'":
            masked[index] = " "
            if char == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    masked[index + 1] = " "
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if quote == '"':
            if char == '"':
                quote = None
            index += 1
            continue
        if text.startswith("<#", index) and _powershell_comment_boundary(
            text,
            index,
        ):
            masked[index : index + 2] = [" ", " "]
            block_comment = True
            index += 2
            continue
        if text.startswith(("@'", '@"'), index):
            line_end = len(text)
            for newline in ("\n", "\r"):
                candidate = text.find(newline, index)
                if candidate >= 0:
                    line_end = min(line_end, candidate)
            if not text[index + 2 : line_end].strip(" \t"):
                here_string_end = "'@" if text[index + 1] == "'" else '"@'
                here_string_interpolates = text[index + 1] == '"'
                for position in range(index, line_end):
                    masked[position] = " "
                index = line_end
                continue
        if char == "'":
            masked[index] = " "
            quote = "'"
        elif char == '"':
            quote = '"'
        elif (
            char == "#"
            and not (index > 0 and text[index - 1] == "<")
            and _powershell_comment_boundary(text, index)
        ):
            masked[index] = " "
            line_comment = True
        index += 1
    return "".join(masked)


def _mask_posix_variable_regions(text: str) -> str:
    """Keep live `$` expansions while blanking comments and single quotes."""

    masked = list(text)
    quote: str | None = None
    escaped = False
    line_comment = False
    for index, char in enumerate(text):
        if line_comment:
            if char in "\r\n":
                line_comment = False
            else:
                masked[index] = " "
            continue
        if escaped:
            masked[index] = " "
            escaped = False
            continue
        if char == "\\" and quote != "'":
            masked[index] = " "
            escaped = True
            continue
        if quote == "'":
            masked[index] = " "
            if char == "'":
                quote = None
            continue
        if quote == '"':
            if char == '"':
                quote = None
            continue
        if char == "'":
            masked[index] = " "
            quote = "'"
        elif char == '"':
            quote = '"'
        elif char == "#" and (
            index == 0
            or text[index - 1].isspace()
            or text[index - 1] in ";|&(){}"
        ):
            masked[index] = " "
            line_comment = True
    for (
        declaration_start,
        declaration_end,
        body_start,
        close_start,
        close_end,
        interpolate,
    ) in _shell_heredoc_regions(text):
        for position in range(declaration_start, declaration_end):
            masked[position] = " "
        for position in range(body_start, close_end):
            if text[position] not in "\r\n":
                masked[position] = " "
        if interpolate:
            body = text[body_start:close_start]
            for match in _SHELL_VAR_USE_RE.finditer(body):
                absolute_start = body_start + match.start()
                backslashes = 0
                cursor = absolute_start - 1
                while cursor >= body_start and text[cursor] == "\\":
                    backslashes += 1
                    cursor -= 1
                if backslashes % 2:
                    continue
                for position in range(
                    absolute_start,
                    body_start + match.end(),
                ):
                    masked[position] = text[position]
    return "".join(masked)


def _shell_segment_command_start(
    line: str,
    start: int,
    end: int,
    *,
    is_powershell: bool,
    prefer_innermost_block: bool = False,
) -> int | None:
    """Locate the command/expression head in one pipeline segment."""

    cursor = start
    while cursor < end and line[cursor].isspace():
        cursor += 1
    if cursor >= end:
        return None

    block_start = -1
    if prefer_innermost_block:
        for position in range(cursor, end):
            if line[position] != "{":
                continue
            if position > 0 and line[position - 1] in {"$", "@"}:
                continue
            if not is_powershell:
                previous_is_boundary = (
                    position == 0
                    or line[position - 1].isspace()
                    or line[position - 1] in ";|&()"
                )
                following_is_boundary = (
                    position + 1 >= len(line)
                    or line[position + 1].isspace()
                )
                if not (previous_is_boundary and following_is_boundary):
                    continue
            block_start = position
    if block_start >= 0:
        cursor = block_start + 1
        while cursor < end and line[cursor].isspace():
            cursor += 1
    elif not is_powershell:
        prefix = line[cursor:end]
        if re.search(r"\bcase\b", prefix) and re.search(r"\bin\b", prefix):
            case_label_end = line.rfind(")", cursor, end)
            if case_label_end >= 0:
                cursor = case_label_end + 1
                while cursor < end and line[cursor].isspace():
                    cursor += 1

    if is_powershell:
        assignment = re.match(
            r"\$\{?[A-Za-z_][A-Za-z0-9_:]*\}?[ \t]*=[ \t]*",
            line[cursor:end],
        )
        if assignment is not None:
            cursor += assignment.end()
            while cursor < end and line[cursor].isspace():
                cursor += 1
        if cursor < end and line[cursor] == "$":
            return cursor
        if cursor < end and line[cursor] == "&":
            cursor += 1
    else:
        while assignment := _SHELL_ASSIGNMENT_PREFIX_RE.match(line, cursor, end):
            cursor = assignment.end()

    match = _SHELL_COMMAND_TOKEN_RE.search(line, cursor, end)
    while match is not None:
        if (
            (match.start() == 0 or line[match.start() - 1] not in {"$", "-"})
            and match.group(0).casefold() not in _SHELL_CONTROL_WORDS
        ):
            return match.start()
        match = _SHELL_COMMAND_TOKEN_RE.search(line, match.end(), end)
    return None


def _shell_redirect_edges(
    text: str,
    *,
    is_powershell: bool,
) -> Iterator[tuple[int, int, bool]]:
    """Yield command, target, and input/output direction for redirects."""

    masked = _mask_shell_quotes_and_comment(
        text,
        is_powershell=is_powershell,
    )
    separator_ends = [
        match.end()
        for match in re.finditer(r"\r\n|\r|\n|;|&&|\|\|", masked)
    ]
    separator_ends.extend(
        offset + 1
        for offset in _unquoted_pipe_offsets(
            text,
            is_powershell=is_powershell,
        )
    )
    separator_ends.sort()
    for match in _SHELL_REDIR_OP_RE.finditer(masked):
        target_start = match.end()
        while target_start < len(text) and text[target_start] in " \t":
            target_start += 1
        if target_start >= len(text) or text[target_start] in "\r\n;|&<>#":
            continue
        if text[target_start] in {"'", '"'}:
            quote = text[target_start]
            target_end = target_start + 1
            escaped = False
            while target_end < len(text):
                char = text[target_end]
                if escaped:
                    escaped = False
                elif char == ("\\" if not is_powershell else "`"):
                    escaped = True
                elif char == quote:
                    break
                elif char in "\r\n":
                    target_end = target_start
                    break
                target_end += 1
            if target_end <= target_start or target_end >= len(text):
                continue
        elif not masked[target_start : target_start + 1].strip():
            continue
        segment_start = max(
            (
                separator_end
                for separator_end in separator_ends
                if separator_end <= match.start()
            ),
            default=0,
        )
        command_start = _shell_segment_command_start(
            masked,
            segment_start,
            match.start(),
            is_powershell=is_powershell,
            prefer_innermost_block=True,
        )
        if command_start is None:
            continue
        yield (
            command_start,
            target_start,
            match.group("op") == "<",
        )


def _shell_pipeline_command_pairs(
    text: str,
    *,
    is_powershell: bool,
) -> Iterator[tuple[int, int]]:
    """Yield command-head pairs for each lexical pipeline edge."""

    masked = _mask_shell_quotes_and_comment(
        text,
        is_powershell=is_powershell,
    )
    pipe_offsets = _unquoted_pipe_offsets(
        text,
        is_powershell=is_powershell,
    )
    control_spans = [
        match.span()
        for match in re.finditer(r"\r\n|\r|\n|;|&&|\|\|", masked)
    ]
    for pipe_index, pipe_offset in enumerate(pipe_offsets):
        left_start = max(
            (
                end
                for start, end in control_spans
                if end <= pipe_offset
            ),
            default=0,
        )
        if pipe_index > 0:
            left_start = max(left_start, pipe_offsets[pipe_index - 1] + 1)

        right_cursor = pipe_offset + 1
        while right_cursor < len(masked) and masked[right_cursor].isspace():
            right_cursor += 1
        if right_cursor >= len(masked):
            continue
        right_end = min(
            (
                start
                for start, _end in control_spans
                if start >= right_cursor
            ),
            default=len(masked),
        )
        if pipe_index + 1 < len(pipe_offsets):
            right_end = min(right_end, pipe_offsets[pipe_index + 1])

        left = _shell_segment_command_start(
            masked,
            left_start,
            pipe_offset,
            is_powershell=is_powershell,
            prefer_innermost_block=True,
        )
        right = _shell_segment_command_start(
            masked,
            right_cursor,
            right_end,
            is_powershell=is_powershell,
        )
        if left is not None and right is not None:
            yield left, right


def _shell_edges_from_script(text: str, shell_kind: str) -> list[dict[str, int]]:
    """Extract graph edges from shell script text using lightweight regex parsing.

    Returns a list of char-level edge triples:
        {"from_char": int, "to_char": int, "kind": int}

    Edge kinds (from DomainEdgeKind):
        SHELL_PIPE = 40          cmd1 | cmd2
        SHELL_REDIR_IN = 41      cmd < file
        SHELL_REDIR_OUT = 42     cmd > file
        SHELL_VAR_DEF_USE = 43   VAR=... → $VAR usage
        SHELL_COMMAND_FILE = 44  command → file argument / source dep
    """
    from cppmega.data.domain_schema import DomainEdgeKind

    edges: list[dict[str, int]] = []
    text_len = len(text)
    seen: set[tuple[int, int, int]] = set()

    def _add_edge(from_char: int, to_char: int, kind: int) -> None:
        if from_char < 0 or to_char < 0 or from_char >= text_len or to_char >= text_len:
            return
        if from_char == to_char:
            return
        key = (from_char, to_char, kind)
        if key not in seen:
            seen.add(key)
            edges.append({"from_char": from_char, "to_char": to_char, "kind": kind})

    is_powershell = shell_kind in {"powershell", "pwsh", "ps1"}
    is_tcsh = shell_kind in {"tcsh", "csh"}
    structural_text = _mask_shell_quotes_and_comment(
        text,
        is_powershell=is_powershell,
    )

    # --- Pipe edges: cmd1 | cmd2 → cmd1 feeds cmd2 ---
    for cmd1_start, cmd2_start in _shell_pipeline_command_pairs(
        text,
        is_powershell=is_powershell,
    ):
        _add_edge(cmd1_start, cmd2_start, int(DomainEdgeKind.SHELL_PIPE))

    # --- Source/dot edges: . ./script.sh or source ./script.sh ---
    for m in _SHELL_SOURCE_RE.finditer(structural_text):
        from_char = m.start(0) + len(m.group(0)) - len(m.group(0).lstrip())
        target_start = m.start(1)
        _add_edge(from_char, target_start, int(DomainEdgeKind.SHELL_COMMAND_FILE))

    # --- Env variable def→use edges ---
    if is_tcsh:
        # tcsh uses setenv VAR value
        var_defs: defaultdict[str, list[int]] = defaultdict(list)
        for m in _TCSH_SETENV_RE.finditer(structural_text):
            var_defs[m.group(1)].append(m.start(1))
        variable_text = _mask_posix_variable_regions(text)
        for m in _TCSH_VAR_USE_RE.finditer(variable_text):
            var_name = m.group(1)
            use_pos = m.start(0)
            def_pos = next(
                (
                    position
                    for position in reversed(var_defs.get(var_name, []))
                    if position < use_pos
                ),
                None,
            )
            if def_pos is not None:
                _add_edge(def_pos, use_pos, int(DomainEdgeKind.SHELL_VAR_DEF_USE))
    elif is_powershell:
        var_defs_ps: defaultdict[str, list[int]] = defaultdict(list)
        powershell_definition_text = _mask_shell_quotes_and_comment(
            text,
            is_powershell=True,
        )
        for name, position in _powershell_variable_definitions(
            powershell_definition_text
        ):
            var_defs_ps[name].append(position)
        ps_definition_positions = {
            position
            for positions in var_defs_ps.values()
            for position in positions
        }
        powershell_variable_text = _mask_powershell_non_interpolating_regions(text)
        for m in _PS_VAR_USE_RE.finditer(powershell_variable_text):
            var_name = m.group(1).casefold()
            use_pos = m.start(0)
            if use_pos in ps_definition_positions:
                continue
            def_pos = next(
                (
                    position
                    for position in reversed(var_defs_ps.get(var_name, []))
                    if position < use_pos
                ),
                None,
            )
            if def_pos is not None:
                _add_edge(def_pos, use_pos, int(DomainEdgeKind.SHELL_VAR_DEF_USE))
    else:
        # bash/sh/zsh/ksh: VAR=value definitions
        var_defs_posix: defaultdict[str, list[int]] = defaultdict(list)
        for m in _SHELL_VAR_DEF_RE.finditer(structural_text):
            var_defs_posix[m.group(1)].append(m.start(1))
        variable_text = _mask_posix_variable_regions(text)
        for m in _SHELL_VAR_USE_RE.finditer(variable_text):
            var_name = m.group(1) or m.group(2)
            use_pos = m.start(0)
            def_pos = next(
                (
                    position
                    for position in reversed(var_defs_posix.get(var_name, []))
                    if position < use_pos
                ),
                None,
            )
            if def_pos is not None:
                _add_edge(def_pos, use_pos, int(DomainEdgeKind.SHELL_VAR_DEF_USE))

    # --- Redirect edges: command < input / command > output ---
    for command_start, target_start, is_input in _shell_redirect_edges(
        text,
        is_powershell=is_powershell,
    ):
        edge_kind = (
            DomainEdgeKind.SHELL_REDIR_IN
            if is_input
            else DomainEdgeKind.SHELL_REDIR_OUT
        )
        _add_edge(command_start, target_start, int(edge_kind))

    # --- Command→file argument edges ---
    if not is_powershell:
        for m in _SHELL_COMMAND_FILE_RE.finditer(structural_text):
            cmd_start = m.start(1)
            file_start = m.start(2)
            _add_edge(cmd_start, file_start, int(DomainEdgeKind.SHELL_COMMAND_FILE))

    # --- PowerShell: cmdlet→parameter edges ---
    if is_powershell:
        for m in _PS_CMDLET_RE.finditer(structural_text):
            cmdlet_start = m.start(1)
            params_str = m.group(2)
            if not params_str:
                continue
            params_offset = m.start(2)
            for pm in _PS_PARAM_RE.finditer(params_str):
                param_value_start = params_offset + pm.start(2)
                _add_edge(
                    cmdlet_start,
                    param_value_start,
                    int(DomainEdgeKind.SHELL_COMMAND_FILE),
                )

    return edges


def load_compile_commands(project_dir: str) -> Optional[dict]:
    """Load compile_commands.json if available."""
    cc_path = find_compile_commands_file(project_dir)
    if cc_path is None:
        return None
    compile_index = load_compile_commands_file(cc_path)
    if compile_index is None:
        return None
    file_args = {}
    for entry in compile_index.entries:
        file_args[entry.filepath] = {
            "compile_args": list(entry.compile_args),
            "build_info": dict(entry.build_info),
        }
    return file_args
    return None


def _discover_include_dirs(project_dir: str) -> list[str]:
    """Recursively find directories named 'include' for -I flags."""
    include_dirs = set()
    for candidate in ['include', 'src', 'lib', 'source', '.']:
        d = os.path.join(project_dir, candidate)
        if os.path.isdir(d):
            include_dirs.add(d)
    # Walk up to 3 levels deep for nested include/ dirs
    for root, dirs, _ in os.walk(project_dir):
        depth = root[len(project_dir):].count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if d not in {'.git', 'build', 'third_party', 'external', 'test', 'tests'}]
        if os.path.basename(root) == 'include':
            include_dirs.add(root)
    return sorted(include_dirs)


_RECOVERY_INCLUDE_DIR_CACHE: dict[str, tuple[str, ...]] = {}
_RECOVERY_INCLUDE_DIR_RE = re.compile(
    r"(?:include|inc(?:\.[A-Za-z0-9_-]+)?)",
    re.IGNORECASE,
)
_MISSING_INCLUDE_DIAGNOSTIC_MARKERS = (
    "file not found",
    "cannot open include file",
    "no such file or directory",
)
_RECOVERY_SOURCE_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s+["<]([^">]+)[">]',
)
_RECOVERY_SOURCE_SCAN_MAX_BYTES = 4 * 1024 * 1024
_RECOVERY_MAX_ROUNDS = 3


def _discover_recovery_include_dirs(project_dir: str) -> tuple[str, ...]:
    """Find nested include roots for a receipt-recorded legacy parse retry.

    This is deliberately separate from the normal inferred build context.  It
    is consulted only after libclang cannot load a translation unit or reports
    a missing include and only when no compile-command entry exists.
    """

    project_root = os.path.realpath(os.path.abspath(project_dir))
    cached = _RECOVERY_INCLUDE_DIR_CACHE.get(project_root)
    if cached is not None:
        return cached

    discovered: list[str] = []
    for root, dirs, _files in os.walk(project_root):
        dirs[:] = sorted(
            directory
            for directory in dirs
            if directory not in {".git", ".hg", ".svn"}
            and not os.path.islink(os.path.join(root, directory))
        )
        basename = os.path.basename(root)
        is_vex_public_headers = (
            basename.lower() == "pub"
            and os.path.basename(os.path.dirname(root)).lower() == "vex"
        )
        if (
            _RECOVERY_INCLUDE_DIR_RE.fullmatch(basename)
            or is_vex_public_headers
        ):
            discovered.append(os.path.abspath(root))
    result = tuple(sorted(set(discovered)))
    _RECOVERY_INCLUDE_DIR_CACHE[project_root] = result
    return result


def _compile_arg_include_dirs(
    compile_args: Sequence[str],
    *,
    project_dir: str,
) -> set[str]:
    """Return normalized include directories already present in clang args."""

    existing: set[str] = set()
    expect_path = False
    for value in compile_args:
        text = str(value)
        if expect_path:
            candidate = text
            expect_path = False
        elif text in {"-I", "-isystem", "-iquote"}:
            expect_path = True
            continue
        elif text.startswith("-I") and text != "-I":
            candidate = text[2:]
        elif text.startswith("-isystem") and text != "-isystem":
            candidate = text.removeprefix("-isystem")
        elif text.startswith("-iquote") and text != "-iquote":
            candidate = text.removeprefix("-iquote")
        else:
            continue
        if not os.path.isabs(candidate):
            candidate = os.path.join(project_dir, candidate)
        existing.add(os.path.realpath(os.path.abspath(candidate)))
    return existing


def _normalize_recovery_include_name(value: object) -> str | None:
    text = str(value or "").strip().replace("\\", "/")
    if (
        not text
        or len(text) > 4096
        or text.startswith("/")
        or re.match(r"^[A-Za-z]:/", text)
        or "\x00" in text
        or any(ord(char) < 32 or ord(char) == 127 for char in text)
    ):
        return None
    parts = tuple(part for part in text.split("/") if part not in {"", "."})
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def _diagnostic_missing_include_names(
    diagnostics: Sequence[str],
) -> tuple[str, ...]:
    names: set[str] = set()
    for spelling in diagnostics:
        match = re.search(r"""['"]([^'"]+)['"]""", str(spelling))
        if match is None:
            continue
        normalized = _normalize_recovery_include_name(match.group(1))
        if normalized is not None:
            names.add(normalized)
    return tuple(sorted(names))


def _forced_include_names(compile_args: Sequence[str]) -> tuple[str, ...]:
    names: set[str] = set()
    expect_path = False
    for value in compile_args:
        text = str(value)
        candidate: str | None = None
        if expect_path:
            candidate = text
            expect_path = False
        elif text == "-include":
            expect_path = True
            continue
        elif text.startswith("-include") and text != "-include":
            candidate = text.removeprefix("-include")
        elif text.lower().startswith("/fi") and len(text) > 3:
            candidate = text[3:]
        if candidate is None:
            continue
        normalized = _normalize_recovery_include_name(candidate)
        if normalized is not None:
            names.add(normalized)
    return tuple(sorted(names))


def _source_include_names(filepath: str) -> tuple[str, ...]:
    names: set[str] = set()
    try:
        with open(filepath, "rb") as stream:
            raw = stream.read(_RECOVERY_SOURCE_SCAN_MAX_BYTES + 1)
    except OSError:
        return ()
    raw = raw[:_RECOVERY_SOURCE_SCAN_MAX_BYTES]
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    text = raw.decode(encoding, errors="ignore")
    for line in text.splitlines():
        match = _RECOVERY_SOURCE_INCLUDE_RE.match(line)
        if match is None:
            continue
        normalized = _normalize_recovery_include_name(match.group(1))
        if normalized is not None:
            names.add(normalized)
    return tuple(sorted(names))


def _seed_recovery_include_names(
    *,
    filepath: str,
    compile_args: Sequence[str],
    missing_diagnostics: Sequence[str],
) -> tuple[str, ...]:
    diagnostic_names = _diagnostic_missing_include_names(missing_diagnostics)
    if diagnostic_names:
        return diagnostic_names
    return tuple(
        sorted(
            {
                *_forced_include_names(compile_args),
                *_source_include_names(filepath),
            }
        )
    )


def _include_recovery_args(
    compile_args: Sequence[str],
    *,
    filepath: str,
    project_dir: str,
    include_names: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """Add only roots that contain requested headers, choosing the closest."""

    project_root = os.path.realpath(os.path.abspath(project_dir))
    source_dir = os.path.realpath(os.path.abspath(os.path.dirname(filepath)))
    candidates = {source_dir, *_discover_recovery_include_dirs(project_root)}
    existing = _compile_arg_include_dirs(
        compile_args,
        project_dir=project_root,
    )

    def _relative_parts(path: str) -> tuple[str, ...]:
        try:
            relative = os.path.relpath(path, project_root)
        except ValueError:
            return ()
        if relative == ".." or relative.startswith(f"..{os.sep}"):
            return ()
        return Path(relative).parts

    source_parts = _relative_parts(source_dir)

    def _rank(path: str) -> tuple[int, int, str]:
        parts = _relative_parts(path)
        shared = 0
        for left, right in zip(source_parts, parts):
            if left != right:
                break
            shared += 1
        return (-shared, len(parts), Path(*parts).as_posix())

    ranked_candidates = sorted(candidates, key=_rank)

    def _contains_project_local_header(candidate: str, name: str) -> bool:
        header = os.path.join(candidate, *name.split("/"))
        if not os.path.isfile(header):
            return False
        try:
            return os.path.commonpath(
                (project_root, os.path.realpath(header))
            ) == project_root
        except ValueError:
            return False

    selected: set[str] = set()
    unresolved: list[str] = []
    for include_name in sorted(set(map(str, include_names))):
        normalized = _normalize_recovery_include_name(include_name)
        if normalized is None:
            continue
        matches = [
            candidate
            for candidate in ranked_candidates
            if _contains_project_local_header(candidate, normalized)
        ]
        if not matches:
            unresolved.append(normalized)
            continue
        selected.add(matches[0])

    added_absolute = [
        candidate
        for candidate in ranked_candidates
        if candidate in selected and candidate not in existing
    ]
    added_relative = [
        Path(*_relative_parts(candidate)).as_posix()
        for candidate in added_absolute
    ]
    return (
        [*map(str, compile_args), *(f"-I{path}" for path in added_absolute)],
        added_relative,
        unresolved,
    )


def _missing_include_diagnostics(tu: TranslationUnit) -> list[str]:
    diagnostics: list[str] = []
    for diagnostic in getattr(tu, "diagnostics", ()):
        spelling = str(getattr(diagnostic, "spelling", "") or "")
        lowered = spelling.lower()
        if any(marker in lowered for marker in _MISSING_INCLUDE_DIAGNOSTIC_MARKERS):
            diagnostics.append(spelling)
    return diagnostics


def _translation_unit_load_error(exc: BaseException) -> bool:
    return type(exc).__name__ == "TranslationUnitLoadError"


def _has_translation_unit_load_error(exc: BaseException) -> bool:
    """Return true only when a chained failure came from libclang TU loading."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _translation_unit_load_error(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _load_translation_unit(
    filepath: str,
    index: Index,
    compile_args: Sequence[str],
) -> TranslationUnit:
    with open(filepath, "rb") as source_file:
        prefix = source_file.read(4)
    unsaved_files = None
    if _wide_source_codec(prefix) is not None:
        source, _parser_bytes, parser_encoding = _read_source_file(filepath)
        if parser_encoding != "utf-8":
            raise ValueError(
                f"wide source was not normalized for libclang: {filepath}"
            )
        unsaved_files = [(filepath, source)]
    return index.parse(
        filepath,
        args=list(compile_args),
        unsaved_files=unsaved_files,
        # Corpus indexing parses each translation unit once. PCH preamble
        # caches are a native-memory win only for repeated reparses; here they
        # make many parallel clang workers retain large buffers.
        options=TranslationUnit.PARSE_INCOMPLETE,
    )


def _load_translation_unit_with_include_recovery(
    filepath: str,
    index: Index,
    compile_args: Sequence[str],
    project_dir: str,
    *,
    allow_include_recovery: bool,
    parse_recovery_records: list[dict[str, object]] | None,
) -> TranslationUnit:
    """Load one TU, retrying missing legacy include context transparently."""

    initial_tu: TranslationUnit | None = None
    initial_error: Exception | None = None
    try:
        initial_tu = _load_translation_unit(filepath, index, compile_args)
    except SymbolIdentityError:
        raise
    except Exception as exc:
        if not _translation_unit_load_error(exc):
            raise RuntimeError(
                f"libclang parse failed for {filepath}: {exc}"
            ) from exc
        initial_error = exc

    initial_missing = (
        _missing_include_diagnostics(initial_tu)
        if initial_tu is not None
        else []
    )
    if initial_error is None and not initial_missing:
        assert initial_tu is not None
        return initial_tu
    if not allow_include_recovery:
        if initial_error is not None:
            raise RuntimeError(
                f"libclang parse failed for {filepath}: {initial_error}"
            ) from initial_error
        assert initial_tu is not None
        return initial_tu

    include_names = _seed_recovery_include_names(
        filepath=filepath,
        compile_args=compile_args,
        missing_diagnostics=initial_missing,
    )
    if not include_names:
        if initial_error is not None:
            raise RuntimeError(
                f"libclang parse failed for {filepath}: {initial_error}"
            ) from initial_error
        assert initial_tu is not None
        return initial_tu

    relative_path = Path(os.path.relpath(filepath, project_dir)).as_posix()
    active_args = list(map(str, compile_args))
    added_include_dirs: list[str] = []
    unresolved_include_names: set[str] = set()
    recovered_tu: TranslationUnit | None = None
    recovered_missing: list[str] = []
    retry_error: Exception | None = None
    retry_round_count = 0
    requested_include_names: set[str] = set(include_names)
    for _round in range(_RECOVERY_MAX_ROUNDS):
        recovery_args, round_added, round_unresolved = _include_recovery_args(
            active_args,
            filepath=filepath,
            project_dir=project_dir,
            include_names=include_names,
        )
        unresolved_include_names.update(round_unresolved)
        if not round_added:
            break
        retry_round_count += 1
        active_args = recovery_args
        added_include_dirs.extend(round_added)
        try:
            candidate_tu = _load_translation_unit(
                filepath,
                index,
                active_args,
            )
        except Exception as recovery_error:
            retry_error = recovery_error
            break
        candidate_missing = _missing_include_diagnostics(candidate_tu)
        recovered_tu = candidate_tu
        recovered_missing = candidate_missing
        include_names = _diagnostic_missing_include_names(candidate_missing)
        requested_include_names.update(include_names)
        if not include_names:
            break

    if not added_include_dirs:
        if initial_error is not None:
            raise RuntimeError(
                f"libclang parse failed for {filepath}: {initial_error}"
            ) from initial_error
        assert initial_tu is not None
        return initial_tu

    encoded_include_dirs = json.dumps(
        added_include_dirs,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded_include_names = json.dumps(
        sorted(requested_include_names),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    record: dict[str, object] = {
        "relative_path": relative_path,
        "trigger": (
            "translation_unit_load_error"
            if initial_error is not None
            else "missing_include_diagnostic"
        ),
        "added_include_dir_count": len(added_include_dirs),
        "added_include_dirs_sha256": hashlib.sha256(
            encoded_include_dirs
        ).hexdigest(),
        "added_include_dir_examples": added_include_dirs[:8],
        "added_include_dir_examples_truncated": len(added_include_dirs) > 8,
        "requested_include_name_count": len(requested_include_names),
        "requested_include_names_sha256": hashlib.sha256(
            encoded_include_names
        ).hexdigest(),
        "requested_include_name_examples": sorted(requested_include_names)[:8],
        "requested_include_name_examples_truncated": (
            len(requested_include_names) > 8
        ),
        "unresolved_include_name_count": len(unresolved_include_names),
        "retry_round_count": retry_round_count,
        "initial_missing_include_count": len(initial_missing),
    }
    if recovered_tu is None:
        assert retry_error is not None
        record.update(
            {
                "status": "unresolved",
                "retry_error_type": type(retry_error).__name__,
            }
        )
        if parse_recovery_records is not None:
            parse_recovery_records.append(record)
        if initial_error is not None:
            raise RuntimeError(
                f"libclang parse failed for {filepath} after nested include "
                f"recovery: {retry_error}"
            ) from retry_error
        assert initial_tu is not None
        return initial_tu

    recovered = initial_error is not None or (
        len(recovered_missing) < len(initial_missing)
    )
    record.update(
        {
            "status": "recovered" if recovered else "unresolved",
            "retry_missing_include_count": len(recovered_missing),
        }
    )
    if parse_recovery_records is not None:
        parse_recovery_records.append(record)
    if recovered:
        print(
            "  Parse include recovery: "
            f"{relative_path} added_dirs={len(added_include_dirs)} "
            f"missing_includes={len(initial_missing)}->{len(recovered_missing)}",
            file=sys.stderr,
            flush=True,
        )
        return recovered_tu
    assert initial_tu is not None
    return initial_tu


_UNRESOLVED_AUTOCONF_ARG_MARKERS = (
    "@<:@",
    "@:>@",
    "@S|@",
    "@%:@",
    "@{:@",
    "@:}@",
    "<arch>",
    "<os>",
    "<variant>",
)
_UNRESOLVED_M4_SUBSTITUTION_RE = re.compile(
    r"@[A-Za-z_][A-Za-z0-9_]*@",
    re.ASCII,
)
_MACRO_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z", re.ASCII)
_TARGET_VALUE_RE = re.compile(r"[A-Za-z0-9_.+:-]+\Z", re.ASCII)
_STANDARD_FLAG_PREFIXES = ("-std=", "--std=", "-cl-std=")
_CLANG_STANDARD_CONTEXTS: dict[str, tuple[str, str]] = {
    "c89": ("c", "c89"),
    "c90": ("c", "c90"),
    "iso9899:1990": ("c", "c90"),
    "iso9899:199409": ("c", "c95"),
    "gnu89": ("c", "c89"),
    "gnu90": ("c", "c90"),
    "c99": ("c", "c99"),
    "c9x": ("c", "c99"),
    "iso9899:1999": ("c", "c99"),
    "iso9899:199x": ("c", "c99"),
    "gnu99": ("c", "c99"),
    "gnu9x": ("c", "c99"),
    "c11": ("c", "c11"),
    "c1x": ("c", "c11"),
    "iso9899:2011": ("c", "c11"),
    "iso9899:201x": ("c", "c11"),
    "gnu11": ("c", "c11"),
    "gnu1x": ("c", "c11"),
    "c17": ("c", "c17"),
    "c18": ("c", "c17"),
    "iso9899:2017": ("c", "c17"),
    "iso9899:2018": ("c", "c17"),
    "gnu17": ("c", "c17"),
    "gnu18": ("c", "c17"),
    "c23": ("c", "c23"),
    "c2x": ("c", "c23"),
    "gnu23": ("c", "c23"),
    "gnu2x": ("c", "c23"),
    "c2y": ("c", "c23"),
    "gnu2y": ("c", "c23"),
    "c++98": ("c++", "c++98"),
    "c++03": ("c++", "c++98"),
    "gnu++98": ("c++", "c++98"),
    "gnu++03": ("c++", "c++98"),
    "c++11": ("c++", "c++11"),
    "c++0x": ("c++", "c++11"),
    "gnu++11": ("c++", "c++11"),
    "gnu++0x": ("c++", "c++11"),
    "c++14": ("c++", "c++14"),
    "c++1y": ("c++", "c++14"),
    "gnu++14": ("c++", "c++14"),
    "gnu++1y": ("c++", "c++14"),
    "c++17": ("c++", "c++17"),
    "c++1z": ("c++", "c++17"),
    "gnu++17": ("c++", "c++17"),
    "gnu++1z": ("c++", "c++17"),
    "c++20": ("c++", "c++20"),
    "c++2a": ("c++", "c++20"),
    "gnu++20": ("c++", "c++20"),
    "gnu++2a": ("c++", "c++20"),
    "c++23": ("c++", "c++23"),
    "c++2b": ("c++", "c++23"),
    "gnu++23": ("c++", "c++23"),
    "gnu++2b": ("c++", "c++23"),
    "c++26": ("c++", "c++26"),
    "c++2c": ("c++", "c++26"),
    "gnu++26": ("c++", "c++26"),
    "gnu++2c": ("c++", "c++26"),
    "cl1.0": ("opencl", "cl1.0"),
    "cl1.1": ("opencl", "cl1.1"),
    "cl1.2": ("opencl", "cl1.2"),
    "cl2.0": ("opencl", "cl2.0"),
    "cl3.0": ("opencl", "cl3.0"),
    "clc++": ("opencl", "clc++1.0"),
    "clc++1.0": ("opencl", "clc++1.0"),
    "clc++2021": ("opencl", "clc++2021"),
}
_CLANG_LANGUAGE_FAMILIES = {
    "c": "c",
    "c-header": "c",
    "cpp-output": "c",
    "objective-c": "c",
    "objective-c-header": "c",
    "objective-c-cpp-output": "c",
    "c++": "c++",
    "c++-header": "c++",
    "c++-cpp-output": "c++",
    "c++-module": "c++",
    "c++-system-header": "c++",
    "objective-c++": "c++",
    "objective-c++-header": "c++",
    "objective-c++-cpp-output": "c++",
    "cuda": "c++",
    "hip": "c++",
    "cl": "opencl",
    "opencl": "opencl",
    "opencl-c": "opencl",
    "clcpp": "opencl",
    "openclcpp": "opencl",
}
_CLANG_LANGUAGE_VALUES = frozenset(
    {
        *_CLANG_LANGUAGE_FAMILIES,
        "assembler",
        "assembler-with-cpp",
        "ir",
        "ast",
        "pcm",
        "precompiled-header",
    }
)
_COMPILE_ARG_VALUE_KINDS = {
    "-D": "define",
    "-U": "undef",
    "-x": "language",
    "-I": "path",
    "-isystem": "path",
    "-iquote": "path",
    "-idirafter": "path",
    "-F": "path",
    "-iframework": "path",
    "-include": "path",
    "-imacros": "path",
    "-isysroot": "path",
    "--sysroot": "path",
    "-resource-dir": "path",
    "-target": "target",
    "--target": "target",
}
_NONEMPTY_EQUALS_ARG_PREFIXES = (
    "--target=",
    "-march=",
    "-mcpu=",
    "--sysroot=",
    "-isysroot=",
    "-resource-dir=",
)
_JOINED_PATH_ARG_PREFIXES = (
    "-isystem",
    "-iquote",
    "-idirafter",
    "-iframework",
    "-include",
    "-imacros",
    "-I",
    "-F",
)


def _compile_arg_has_garbage(value: object) -> bool:
    return (
        not isinstance(value, str)
        or not value
        or ";" in value
        or "$" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or any(marker in value for marker in _UNRESOLVED_AUTOCONF_ARG_MARKERS)
        or _UNRESOLVED_M4_SUBSTITUTION_RE.search(value) is not None
    )


def _standard_flag_context(
    arg: str,
) -> tuple[str, str, str] | None:
    for prefix in _STANDARD_FLAG_PREFIXES:
        if arg.startswith(prefix):
            raw_value = arg.removeprefix(prefix).lower()
            context = _CLANG_STANDARD_CONTEXTS.get(raw_value)
            if context is None:
                return None
            family, canonical_standard = context
            return raw_value, family, canonical_standard
    return None


def _standard_flag_contexts(
    args: Sequence[str],
) -> list[tuple[str, str, str]]:
    return [
        context
        for arg in args
        if any(arg.startswith(prefix) for prefix in _STANDARD_FLAG_PREFIXES)
        if (context := _standard_flag_context(arg)) is not None
    ]


def _is_sane_compile_args(args: list[str]) -> bool:
    """Check if build_context returned usable args (not shell script garbage)."""
    if not args:
        return False

    machine_width_flags: set[str] = set()
    target_values: dict[str, set[str]] = {
        "target": set(),
        "march": set(),
        "mcpu": set(),
    }
    language_values: set[str] = set()
    standard_values: set[str] = set()
    standard_families: set[str] = set()
    has_opencl_standard_flag = False
    index = 0
    while index < len(args):
        arg = args[index]
        if _compile_arg_has_garbage(arg):
            return False

        value_kind = _COMPILE_ARG_VALUE_KINDS.get(arg)
        if value_kind is not None:
            if index + 1 >= len(args):
                return False
            operand = args[index + 1]
            if _compile_arg_has_garbage(operand) or operand.startswith("-"):
                return False
            if value_kind == "define":
                macro_name = operand.split("=", 1)[0].split("(", 1)[0]
                if _MACRO_NAME_RE.fullmatch(macro_name) is None:
                    return False
            elif value_kind == "undef":
                if _MACRO_NAME_RE.fullmatch(operand) is None:
                    return False
            elif value_kind == "language":
                language = operand.lower()
                if language not in _CLANG_LANGUAGE_VALUES:
                    return False
                language_values.add(language)
            elif value_kind == "target":
                if _TARGET_VALUE_RE.fullmatch(operand) is None:
                    return False
                target_values["target"].add(operand)
            index += 2
            continue

        if arg in {"-std", "--std", "-cl-std"}:
            return False
        if any(arg.startswith(prefix) for prefix in _STANDARD_FLAG_PREFIXES):
            context = _standard_flag_context(arg)
            if context is None:
                return False
            if arg.startswith("-cl-std="):
                has_opencl_standard_flag = True
            raw_standard, standard_family, _canonical_standard = context
            standard_values.add(raw_standard)
            standard_families.add(standard_family)
        elif arg.startswith("-x"):
            language = arg.removeprefix("-x").lower()
            if language not in _CLANG_LANGUAGE_VALUES:
                return False
            language_values.add(language)
        elif arg.startswith("-D"):
            definition = arg.removeprefix("-D")
            macro_name = definition.split("=", 1)[0].split("(", 1)[0]
            if _MACRO_NAME_RE.fullmatch(macro_name) is None:
                return False
        elif arg.startswith("-U"):
            if _MACRO_NAME_RE.fullmatch(arg.removeprefix("-U")) is None:
                return False

        for prefix in _NONEMPTY_EQUALS_ARG_PREFIXES:
            if arg.startswith(prefix):
                value = arg.removeprefix(prefix)
                if not value:
                    return False
                if prefix in {"--target=", "-march=", "-mcpu="} and (
                    _TARGET_VALUE_RE.fullmatch(value) is None
                ):
                    return False
                target_key = {
                    "--target=": "target",
                    "-march=": "march",
                    "-mcpu=": "mcpu",
                }.get(prefix)
                if target_key is not None:
                    target_values[target_key].add(value)

        for prefix in _JOINED_PATH_ARG_PREFIXES:
            if arg.startswith(prefix) and arg != prefix:
                if not arg.removeprefix(prefix):
                    return False
                break

        if arg in {"-m32", "-m64"}:
            machine_width_flags.add(arg)

        if not arg.startswith("-"):
            return False
        if ' ' in arg and not arg.startswith('-D'):
            return False
        index += 1

    if {"-m32", "-m64"}.issubset(machine_width_flags):
        return False
    if any(len(values) > 1 for values in target_values.values()):
        return False
    if len(language_values) > 1 or len(standard_values) > 1:
        return False
    if len(standard_families) > 1:
        return False
    if language_values and standard_families:
        language_family = _CLANG_LANGUAGE_FAMILIES.get(
            next(iter(language_values))
        )
        if language_family != next(iter(standard_families)):
            return False
    if has_opencl_standard_flag:
        if len(language_values) != 1:
            return False
        language_family = _CLANG_LANGUAGE_FAMILIES.get(
            next(iter(language_values))
        )
        if language_family != "opencl":
            return False
    return True


_SIMPLE_FALLBACK_ARGS = ["-fsyntax-only", "-Wno-everything"]
_DEFAULT_BUILD_INFO_KEYS = frozenset(
    {"build_system", "source", "compiler", "standard"}
)
_UNUSABLE_DETECTED_ARGS_STATUS = "fallback_unusable_detected_args"
DEFAULT_PARSE_BATCH_FILES = 25
PARSE_SUBMIT_WINDOW_PER_WORKER = 2
PARSE_HEARTBEAT_FILES = 25
PARSE_HEARTBEAT_SECONDS = 30.0


def compute_parse_batch_size(
    num_files: int,
    effective_workers: int,
    *,
    max_batch_files: int = DEFAULT_PARSE_BATCH_FILES,
) -> int:
    """Bound parse IPC payload size independently of worker count.

    Each parse future returns serialized FunctionDef/TypeDef payloads to the
    parent process. For very large repos, ``len(files) / workers`` creates huge
    IPC payloads and parent RSS spikes; keep each future's payload bounded while
    preserving a minimum batch size for scheduler overhead.
    """
    if num_files <= 0:
        return 0
    if effective_workers <= 0:
        raise ValueError(f"effective_workers must be positive, got {effective_workers}")
    if max_batch_files <= 0:
        raise ValueError(f"max_batch_files must be positive, got {max_batch_files}")
    nominal = max(25, (num_files + effective_workers - 1) // effective_workers)
    return min(max_batch_files, nominal)


def _resolve_default_compile_context(
    project_dir: str,
    platform_info: dict[str, object],
    detected_args: list[str],
) -> tuple[list[str], dict[str, object]]:
    """Resolve one atomic default parser context and its emitted provenance."""
    detected_args_usable = _is_sane_compile_args(detected_args)
    if detected_args_usable:
        result = list(detected_args)
    else:
        # build_context returned garbage (e.g., shell fragments from configure)
        result = list(_SIMPLE_FALLBACK_ARGS)

    if not any(arg.startswith('-I') for arg in result):
        result.append(f'-I{project_dir}')
        for d in _discover_include_dirs(project_dir):
            result.append(f'-I{d}')

    build_info = {
        key: value
        for key, value in platform_info.items()
        if key in _DEFAULT_BUILD_INFO_KEYS and value is not None
    }
    if not detected_args_usable:
        # Compiler/standard were inferred from the rejected aggregate and do not
        # describe the simple parser fallback that actually loaded the source.
        build_info.pop("compiler", None)
        build_info.pop("standard", None)
        build_info["compile_args_status"] = _UNUSABLE_DETECTED_ARGS_STATUS
    else:
        standard_contexts = _standard_flag_contexts(result)
        if standard_contexts:
            build_info["standard"] = standard_contexts[-1][2]
    return result, build_info


def _harmonize_build_info_with_compile_args(
    build_info: dict | None,
    compile_args: Sequence[str],
) -> dict | None:
    """Align emitted dialect provenance with the args that parsed this file."""
    if not build_info:
        return build_info
    standard_contexts = _standard_flag_contexts(compile_args)
    if not standard_contexts:
        return build_info
    harmonized = dict(build_info)
    harmonized["standard"] = standard_contexts[-1][2]
    return harmonized


def get_default_compile_args(
    project_dir: str,
    *,
    macos_sdk_path: str | None = None,
) -> list[str]:
    """Generate default compile args for projects without compile_commands.json."""
    platform_info, args, _compile_index = detect_build_context(
        project_dir,
        macos_sdk_path=macos_sdk_path,
    )
    result, _build_info = _resolve_default_compile_context(
        project_dir,
        platform_info,
        args,
    )
    return result


def _is_legacy_cpp_c_path(filepath: str) -> bool:
    """Return whether an upstream path explicitly declares a .c file as C++."""

    parts = tuple(part.casefold() for part in Path(filepath).parts)
    return any(
        parts[index : index + 2] == ("bld", "plusplus")
        for index in range(len(parts) - 1)
    )


def _adapt_args_for_file(args: list[str], filepath: str) -> list[str]:
    """Adapt compile args to the file's extension or explicit path dialect."""
    raw_ext = os.path.splitext(filepath)[1]
    ext = raw_ext.lower()
    if ext in HEADER_EXTENSIONS:
        explicit_language: str | None = None
        for arg_index, arg in enumerate(args):
            if arg == '-x' and arg_index + 1 < len(args):
                explicit_language = args[arg_index + 1].lower()
            elif arg.startswith('-x') and arg != '-x':
                explicit_language = arg[2:].lower()
        is_c_header = explicit_language in {'c', 'c-header'}
        if explicit_language is None:
            is_c_header = any(
                arg.startswith('-std=c') and not arg.startswith('-std=c++')
                for arg in args
            )

        # Change only the language form. The compile database or detected project
        # context owns the dialect flag, including C++20/C++23/C++26.
        adapted = []
        skip_next = False
        standard_indexes = [
            index for index, arg in enumerate(args) if arg.startswith('-std=')
        ]
        last_standard_index = standard_indexes[-1] if standard_indexes else None
        for arg_index, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg == '-x':
                skip_next = True
                continue
            if arg.startswith('-x') and arg != '-x':
                continue
            if arg.startswith('-std=') and arg_index != last_standard_index:
                continue
            adapted.append(arg)
        header_language = 'c-header' if is_c_header else 'c++-header'
        return ['-x', header_language] + adapted
    mixed_case_cpp_source = (
        raw_ext == '.C'
        or (
            ext in (CPP_EXTENSIONS - C_EXTENSIONS)
            and raw_ext != ext
        )
    )
    if mixed_case_cpp_source:
        has_explicit_language = any(
            arg == '-x' or (arg.startswith('-x') and arg != '-x')
            for arg in args
        )
        if has_explicit_language:
            return args
        # Libclang does not reliably infer mixed-case suffixes such as .Cpp.
        # Preserve all caller-owned flags and add only the missing language.
        return ['-x', 'c++'] + args
    if ext in C_EXTENSIONS:
        adapted = []
        skip_next = False
        force_cpp = _is_legacy_cpp_c_path(filepath)
        has_c_standard = False
        has_cpp_standard = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg == '-x':
                # Skip -x <lang> pair
                skip_next = True
                continue
            if arg.startswith('-x') and arg != '-x':
                continue

            standard_context = (
                _standard_flag_context(arg)
                if any(
                    arg.startswith(prefix)
                    for prefix in _STANDARD_FLAG_PREFIXES
                )
                else None
            )
            if standard_context is not None:
                if standard_context[1] == 'c' and not force_cpp:
                    adapted.append(arg)
                    has_c_standard = True
                elif standard_context[1] == 'c++' and force_cpp:
                    adapted.append(arg)
                    has_cpp_standard = True
                # A standard for the other language cannot describe this TU.
                continue
            adapted.append(arg)
        if force_cpp:
            # Open Watcom keeps C++ regression inputs under bld/plusplus with
            # .c suffixes. Parsing those as C can make libclang loop forever.
            prefix = ['-x', 'c++']
            if not has_cpp_standard:
                prefix.append('-std=c++17')
            return prefix + adapted
        # Ensure C mode is set, and use the established parser fallback only
        # when the build context did not supply a valid C dialect.
        prefix = ['-x', 'c']
        if not has_c_standard:
            prefix.append('-std=c11')
        return prefix + adapted
    return args


def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 bytes per token for code)."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Cross-repo base-lib symbol linking (OPTIONAL; default off).
#
# When --global-symbol-index <path> is given, an unresolved callee (one NOT in
# the local repo index) is looked up in the GLOBAL base-lib symbol store built by
# scripts/crossrepo/build_global_symbol_index.py. If it is a selected base-lib
# symbol, its DEFINITION is pulled in as the DEEPEST dependency chunk of the
# document and tagged dep_source='crosslib:<repo>' so the model knows it is a
# base-lib impl and which lib it came from. Behavior is UNCHANGED when the flag
# is absent (global_symbols is None everywhere).
#
# HARD BOUNDS (RULE #1: bounded, no silent explosion):
#   * cross-lib depth = 1 (pulled defs are leaves; we never follow their callees)
#   * a cap on the number of cross-lib deps per document
#   * a cross-lib token budget per document
#   * trivial/tiny inline bodies are excluded at BUILD time (body-len floor) so
#     they never enter the store.
# --------------------------------------------------------------------------- #
CROSSLINK_MAX_DEPS_PER_DOC = 12
CROSSLINK_TOKEN_BUDGET_PER_DOC = 6144
GLOBAL_SYMBOL_DB_SCHEMA_VERSION = 5


class AmbiguousGlobalSymbolError(SymbolIdentityError):
    """Raised when authoritative global-symbol fields do not select one provider."""


class CrossLinkBudget:
    """Per-document bound on cross-lib pulls (count + token budget)."""

    __slots__ = [
        "max_deps",
        "token_budget",
        "used_deps",
        "used_tokens",
        "ambiguous_lookups",
        "ambiguity_examples",
    ]

    def __init__(self, max_deps: int = CROSSLINK_MAX_DEPS_PER_DOC,
                 token_budget: int = CROSSLINK_TOKEN_BUDGET_PER_DOC):
        self.max_deps = max_deps
        self.token_budget = token_budget
        self.used_deps = 0
        self.used_tokens = 0
        self.ambiguous_lookups = 0
        self.ambiguity_examples: list[str] = []

    def has_room(self) -> bool:
        return self.used_deps < self.max_deps and self.used_tokens < self.token_budget

    def can_afford(self, tok: int) -> bool:
        return (self.used_deps < self.max_deps
                and self.used_tokens + tok <= self.token_budget)

    def spend(self, tok: int) -> None:
        self.used_deps += 1
        self.used_tokens += tok

    def note_ambiguity(self, qname: str) -> None:
        self.ambiguous_lookups += 1
        if qname and qname not in self.ambiguity_examples and len(self.ambiguity_examples) < 8:
            self.ambiguity_examples.append(qname)


class GlobalSymbolReader:
    """Read-only accessor over the global base-lib symbol SQLite store.

    A miss returns None (the callee is not a selected base-lib symbol) — that is
    the normal case for most callees and is NOT a fallback: the ONE clear path is
    "consult the index; pull if present, otherwise leave unresolved as before".
    """

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"--global-symbol-index not found: {path} (build it with "
                f"scripts/crossrepo/build_global_symbol_index.py)"
            )
        self.path = path
        uri = f"file:{path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(symbols)").fetchall()
        }
        if "symbol_uid" not in cols:
            raise SymbolIdentityError(
                f"--global-symbol-index uses old qname-only schema: {path}. "
                "Rebuild it with scripts/crossrepo/build_global_symbol_index.py "
                "so overloaded symbols and multi-implementation std symbols are "
                "not collapsed."
            )
        required = {
            "symbol_key",
            "symbol_id",
            "usr",
            "canonical_signature",
            "symbol_kind",
            "identity_schema_version",
            "column",
            "provider",
            "include_provenance",
        }
        missing = sorted(required - cols)
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if missing or version != GLOBAL_SYMBOL_DB_SCHEMA_VERSION:
            raise SymbolIdentityError(
                f"--global-symbol-index schema is incompatible: {path}; "
                f"user_version={version}, missing_columns={missing}. Migrate only "
                "when provider provenance is reconstructable, or rebuild it."
            )
        incompatible_rows = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM symbols "
                "WHERE identity_schema_version!=? OR symbol_key='' OR symbol_id='' "
                "OR symbol_kind='' OR (usr='' AND canonical_signature='' "
                "AND symbol_key NOT LIKE 'repo_file_location:%') "
                "OR canonical_signature LIKE 'legacy-kind=%text-sha1=%' "
                "OR provider='' OR include_provenance=''",
                (SYMBOL_IDENTITY_SCHEMA_VERSION,),
            ).fetchone()[0]
        )
        if incompatible_rows:
            raise SymbolIdentityError(
                f"--global-symbol-index has incompatible symbol identity rows: {path}; "
                f"count={incompatible_rows}, expected_version="
                f"{SYMBOL_IDENTITY_SCHEMA_VERSION}. Migrate or rebuild the index."
            )
        for row in self._conn.execute(
            "SELECT symbol_uid, symbol_key, qname, base_repo, file, line, column, "
            "symbol_kind "
            "FROM symbols WHERE usr='' AND canonical_signature=''"
        ):
            (
                symbol_uid,
                symbol_key,
                qname,
                project_id,
                file,
                line,
                column,
                symbol_kind,
            ) = row
            validate_repo_file_location_identity_claim(
                str(symbol_key),
                project=str(project_id),
                file=str(file),
                line=int(line),
                column=int(column),
                kind=str(symbol_kind),
                qname=str(qname),
                source=f"global symbol index {path}:{symbol_uid}",
            )
        for (project_id,) in self._conn.execute(
            "SELECT DISTINCT base_repo FROM symbols"
        ):
            require_project_identity(
                project_id, source=f"global symbol index {path}"
            )
        missing_std_provenance = self._conn.execute(
            "SELECT symbol_uid, file FROM symbols WHERE base_lib='std' "
            "AND (provider NOT IN ('libc++', 'libstdc++', 'msvc-stl') "
            "OR include_provenance='') LIMIT 1"
        ).fetchone()
        if missing_std_provenance is not None:
            raise SymbolIdentityError(
                f"{path}: std symbol lacks authoritative provider/include provenance: "
                f"uid={missing_std_provenance[0]} file={missing_std_provenance[1]!r}"
            )
        registry_table = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='symbol_identities'"
        ).fetchone()
        if registry_table is None:
            raise SymbolIdentityError(
                f"--global-symbol-index has no corpus collision registry: {path}. "
                "Migrate or rebuild the index."
            )
        unregistered_rows = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM symbols AS s "
                "LEFT JOIN symbol_identities AS i "
                "ON i.symbol_id=s.symbol_id AND i.symbol_key=s.symbol_key "
                "WHERE i.symbol_id IS NULL"
            ).fetchone()[0]
        )
        if unregistered_rows:
            raise SymbolIdentityError(
                f"--global-symbol-index has unregistered symbol IDs: {path}; "
                f"count={unregistered_rows}. Migrate or rebuild the index."
            )
        self._cache: dict[str, tuple[dict[str, object], ...]] = {}

    def lookup_candidates(self, qname: str) -> list[dict[str, object]]:
        """Return every function candidate for a display/search qname."""
        key = normalize_inline_namespace_qname(qname)
        if key in self._cache:
            return [dict(record) for record in self._cache[key]]
        rows = self._conn.execute(
            "SELECT symbol_uid, symbol_key, symbol_id, qname, base_lib, base_repo, text, "
            "token_est, kind, file, line, column, end_line, usr, canonical_signature, "
            "symbol_kind, identity_schema_version, provider, include_provenance "
            "FROM symbols WHERE qname=? AND sym_type='func' "
            "ORDER BY base_repo, file, line, column, symbol_uid",
            (key,),
        ).fetchall()
        records = tuple(
            {
                "symbol_uid": row[0],
                "symbol_key": row[1],
                "symbol_id": int(str(row[2]), 16),
                "qname": row[3],
                "base_lib": row[4],
                "base_repo": row[5],
                "text": row[6],
                "token_est": row[7],
                "kind": row[8],
                "file": row[9],
                "line": row[10],
                "column": row[11],
                "end_line": row[12],
                "usr": row[13],
                "canonical_signature": row[14],
                "symbol_kind": row[15],
                "identity_schema_version": row[16],
                "provider": row[17],
                "include_provenance": row[18],
            }
            for row in rows
        )
        self._cache[key] = records
        return [dict(record) for record in records]

    def lookup(
        self,
        qname: str,
        *,
        symbol_key: str | None = None,
        usr: str | None = None,
        canonical_signature: str | None = None,
        symbol_kind: str | None = None,
        project: str | None = None,
        file: str | None = None,
        provider: str | None = None,
        include_provenance: str | None = None,
    ) -> dict[str, object] | None:
        """Resolve one exact candidate and fail closed on ambiguity."""
        all_candidates = self.lookup_candidates(qname)
        candidates = list(all_candidates)
        identity_filter_used = False
        if symbol_key:
            identity_filter_used = True
            candidates = [row for row in candidates if row["symbol_key"] == symbol_key]
        if usr:
            identity_filter_used = True
            candidates = [row for row in candidates if row["usr"] == usr]
        if canonical_signature:
            identity_filter_used = True
            signature = _normalize_signature_text(canonical_signature)
            candidates = [
                row
                for row in candidates
                if _normalize_signature_text(str(row["canonical_signature"])) == signature
            ]
        if symbol_kind:
            identity_filter_used = True
            candidates = [row for row in candidates if row["symbol_kind"] == symbol_kind]
        if not identity_filter_used:
            candidates = list(all_candidates)
        if project:
            candidates = [row for row in candidates if row["base_repo"] == project]
        if file:
            candidates = [row for row in candidates if row["file"] == file]
        if provider:
            candidates = [row for row in candidates if row["provider"] == provider]
        if include_provenance:
            candidates = [
                row
                for row in candidates
                if row["include_provenance"] == include_provenance
            ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            preview = [
                f"{row['base_repo']}:{row['file']}:{row['line']}"
                for row in candidates[:8]
            ]
            raise AmbiguousGlobalSymbolError(
                "ambiguous global symbol lookup: "
                f"qname={qname!r} usr={usr or ''!r} provider={provider or ''!r} "
                f"include={include_provenance or ''!r} candidates={preview}"
            )
        return None

    def close(self) -> None:
        self._conn.close()


def collect_transitive_deps(
    root_symbol: str,
    index: ProjectIndex,
    max_depth: int = 5,
    *,
    global_symbols: "GlobalSymbolReader | None" = None,
    crosslink_visited: dict[str, dict[str, object]] | None = None,
    crosslink_budget: "CrossLinkBudget | None" = None,
) -> list[str]:
    """BFS to collect transitive dependencies of a function.

    When ``global_symbols`` is supplied (the optional cross-repo base-lib symbol
    index), an UNRESOLVED callee — one that is NOT in the local repo index — is
    looked up in the global index. If it is a selected base-lib symbol it is
    recorded as a CROSS-LIB pull in ``crosslink_visited`` (symbol UIDs) under HARD
    BOUNDS enforced by ``crosslink_budget`` (cross-lib depth = 1, a per-doc cap
    on the number of cross-lib deps, and a per-doc cross-lib token budget). We do
    NOT recurse into a pulled base-lib function's own transitive closure inside
    the base lib (depth-1 only), and we SKIP trivial/tiny inline bodies — that is
    bounded by ``crosslink_budget`` + the builder's body-len floor.
    """
    crosslink_active = (
        global_symbols is not None
        and crosslink_visited is not None
        and crosslink_budget is not None
    )

    def _try_crosslink(reference: str | SymbolReference) -> None:
        # Depth-1 only: a pulled base-lib def is a LEAF; we never enqueue it, so
        # its base-lib-internal callees are not followed. Bounds (count + token
        # budget) are enforced by crosslink_budget. ONE clear path; a miss (not a
        # selected base-lib symbol) is simply a no-op, never a fallback.
        if not crosslink_budget.has_room():
            return
        try:
            if isinstance(reference, dict):
                qname = str(reference.get("qname") or "")
                reference_project = (
                    None
                    if is_crosslinkable_baselib(qname)
                    else str(reference.get("project") or "") or None
                )
                hit = global_symbols.lookup(
                    qname,
                    symbol_key=(
                        str(reference.get("symbol_key") or "") or None
                        if reference_project
                        else None
                    ),
                    usr=str(reference.get("usr") or "") or None,
                    canonical_signature=(
                        str(reference.get("canonical_signature") or "") or None
                    ),
                    symbol_kind=str(reference.get("symbol_kind") or "") or None,
                    project=reference_project,
                    file=(str(reference.get("file") or "") or None)
                    if reference_project
                    else None,
                    provider=str(reference.get("provider") or "") or None,
                    include_provenance=(
                        str(reference.get("include_provenance") or "") or None
                    ),
                )
            else:
                qname = reference
                hit = global_symbols.lookup(qname)
        except AmbiguousGlobalSymbolError:
            # Cross-repo enrichment is optional context. Never guess a provider:
            # leave the route unresolved and emit an aggregate audit receipt.
            crosslink_budget.note_ambiguity(qname)
            return
        if hit is None:
            return
        target_key = hit.get("symbol_key")
        if not isinstance(target_key, str) or not target_key:
            return
        index._register_symbol_key(target_key)
        uid = str(hit["symbol_uid"])
        if uid in crosslink_visited:
            return
        if not crosslink_budget.can_afford(hit["token_est"]):
            return
        crosslink_visited[uid] = hit
        crosslink_budget.spend(hit["token_est"])

    resolved_root = index.resolve_function_key(root_symbol)
    if resolved_root is None:
        return []
    visited = {resolved_root}
    queue = deque([(resolved_root, 0)])
    deps = []

    while queue:
        symbol_key, depth = queue.popleft()
        if depth >= max_depth:
            continue
        func = index.functions.get(symbol_key)
        if not func:
            continue
        local_refs: list[str | SymbolReference] = list(
            getattr(func, "callee_refs", []) or []
        )
        if not local_refs:
            local_refs = list(getattr(func, "callee_keys", []) or []) + list(func.callees)
        for reference in local_refs:
            callee = index.resolve_function_key(reference)
            if callee is not None and callee in visited:
                continue
            if callee is not None:
                visited.add(callee)
                deps.append(callee)
                queue.append((callee, depth + 1))
            elif crosslink_active:
                # An unresolved callee (not local) MIGHT be a base-lib symbol
                # (e.g. a vendored/forward-declared libc-style name). Try it.
                _try_crosslink(reference)
        # Base-lib-namespaced callees (std::/boost::) were filtered out of
        # `callees` by the normal system-prefix filter and live here. They are
        # the PRIMARY cross-link source. Only consulted when cross-linking is on.
        if crosslink_active:
            base_refs: list[str | SymbolReference] = list(
                getattr(func, "baselib_callee_refs", []) or []
            )
            if not base_refs:
                base_refs = list(func.baselib_callees)
            for reference in base_refs:
                _try_crosslink(reference)

    return deps


_CLANG_PART_NODE_BUCKETS = {
    0: 0,   # other
    1: 81,  # preamble / preprocessor context
    2: 35,  # function signature/declarator
    3: 1,   # function body/definition
    4: 2,   # class or struct declaration
    5: 7,   # class member
    6: 80,  # comment/docstring
    7: 9,   # typedef/using
    8: 5,   # namespace
    HEADER_FRAGMENT_KIND: 2,  # standalone header fragment
    MACRO_KIND: 81,  # preprocessor macro definition
}


def extract_ast_metadata_from_parts(
    full_text: str,
    parts_info: list[PartInfo],
    index: ProjectIndex | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Emit clang AST metadata for assembled documents.

    The clang indexer builds documents from clang-extracted parts rather than
    reparsing the assembled training text. For function parts this copies the
    exact clang cursor metadata captured while parsing the original translation
    unit. Non-code scaffolding such as section comments and separators stays
    zero because it is not part of the clang AST.
    """
    text_len = len(full_text)
    ast_depth = [0] * text_len
    sibling_index = [0] * text_len
    ast_node_type = [0] * text_len
    offset = 0

    for part_index, part in enumerate(parts_info):
        part_text, kind, _dep_level, _name, display_qname = (
            part[0], part[1], part[2], part[3], part[4]
        )
        part_len = len(part_text)
        if part_len <= 0:
            continue
        end = min(offset + part_len, text_len)
        if offset >= text_len or offset >= end:
            break
        func = None
        if index is not None:
            key = _part_symbol_key(part) or (
                display_qname if isinstance(display_qname, str) else None
            )
            resolved = index.resolve_function_key(key)
            func = index.functions.get(resolved) if resolved is not None else None
        if (
            func is not None
            and len(func.ast_depth) == part_len
            and len(func.sibling_index) == part_len
            and len(func.ast_node_type) == part_len
        ):
            ast_depth[offset:end] = func.ast_depth[: end - offset]
            sibling_index[offset:end] = func.sibling_index[: end - offset]
            ast_node_type[offset:end] = func.ast_node_type[: end - offset]
        else:
            fallback_node_type = _CLANG_PART_NODE_BUCKETS.get(int(kind), 0)
            if fallback_node_type:
                span_len = end - offset
                ast_node_type[offset:end] = [fallback_node_type] * span_len
                sibling_index[offset:end] = [min(part_index, 65535)] * span_len
        offset += part_len
        if part_index < len(parts_info) - 1:
            offset += 2

    return ast_depth, sibling_index, ast_node_type


# -------------------------------------------------------------------------
# Semantic metadata extraction (symbol_ids, call_targets, type_refs, def_use)
# -------------------------------------------------------------------------

# Def-use constants
DEF_USE_NONE = 0
DEF_USE_DEF = 1
DEF_USE_USE = 2

# CursorKind sets for classification
_DEFINITION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.CONVERSION_FUNCTION,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.VAR_DECL,
    CursorKind.FIELD_DECL,
    CursorKind.PARM_DECL,
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.UNION_DECL,
    CursorKind.ENUM_DECL,
    CursorKind.ENUM_CONSTANT_DECL,
    CursorKind.TYPEDEF_DECL,
    CursorKind.TYPE_ALIAS_DECL,
    CursorKind.NAMESPACE,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
}

_CALL_KINDS = {
    CursorKind.CALL_EXPR,
}

_TYPE_REF_KINDS = {
    CursorKind.TYPE_REF,
    CursorKind.TEMPLATE_REF,
}

_REFERENCE_KINDS = {
    CursorKind.DECL_REF_EXPR,
    CursorKind.MEMBER_REF_EXPR,
    CursorKind.MEMBER_REF,
    CursorKind.TYPE_REF,
    CursorKind.TEMPLATE_REF,
    CursorKind.NAMESPACE_REF,
    CursorKind.OVERLOADED_DECL_REF,
    CursorKind.VARIABLE_REF,
}


def extract_semantic_metadata(
    source: str,
    tu: 'TranslationUnit',
    filename: str,
    *,
    project_dir: str | None = None,
    project_id: str,
    fallback_file: str | None = None,
    source_encoding: str = "utf-8",
    external_reference_omissions: ExternalReferenceOmissions | None = None,
) -> dict:
    """Extract per-character semantic metadata from a translation unit.

    Walks the clang AST and produces four char-level arrays:

    - symbol_ids:   per-char deterministic hash of the symbol's qualified name
                    (0 = no symbol)
    - call_targets: per-char symbol ID on the explicit callee token
                    (0 = not a call)
    - type_refs:    per-char symbol ID on TYPE_REF/TEMPLATE_REF name tokens
                    (0 = not a type reference)
    - def_use:      per-char def/use marker (0=none, 1=def, 2=use)

    Returns:
        dict with keys 'symbol_ids', 'call_targets', 'type_refs', 'def_use',
        each a list[int] of length len(source).
    """
    stable_project_id = require_project_identity(
        project_id,
        source=f"extract_semantic_metadata({filename})",
    )
    text_len = len(source)
    symbol_ids = [0] * text_len
    call_targets = [0] * text_len
    type_refs = [0] * text_len
    def_use = [0] * text_len
    identity_registry = SymbolIdentityRegistry()

    if text_len == 0 or tu is None:
        return {
            "symbol_ids": symbol_ids,
            "call_targets": call_targets,
            "type_refs": type_refs,
            "def_use": def_use,
            SYMBOL_IDENTITIES_COLUMN: [],
        }

    source_bytes = source.encode(source_encoding, errors="strict")
    byte_to_char = _byte_to_char_mapper(source, source_encoding)

    def _register_symbol_key(symbol_key: str, cursor: Cursor) -> int:
        return identity_registry.register(
            symbol_key,
            source=(
                f"{filename}:{int(getattr(cursor.location, 'line', 0) or 0)}:"
                f"{_cursor_kind_name(cursor)}"
            ),
        )

    def _char_span(byte_start: int, byte_end: int) -> tuple[int, int] | None:
        char_start = byte_to_char(byte_start)
        char_end = byte_to_char(byte_end)
        if char_start >= text_len or char_start >= char_end:
            return None
        return char_start, min(char_end, text_len)

    def _cursor_name_spans(cursor: Cursor, spelling: str) -> list[tuple[int, int]]:
        """Return only the source token(s) that spell a cursor's name."""
        if not spelling:
            return []
        location = getattr(cursor, "location", None)
        raw_location_offset = getattr(location, "offset", None)
        location_offset = (
            int(raw_location_offset) if raw_location_offset is not None else -1
        )
        spelling_bytes = spelling.encode("utf-8")
        if (
            location_offset >= 0
            and source_bytes[location_offset:location_offset + len(spelling_bytes)]
            == spelling_bytes
        ):
            span = _char_span(
                location_offset,
                location_offset + len(spelling_bytes),
            )
            return [span] if span is not None else []

        token_spellings = {spelling}
        if spelling.startswith("operator"):
            operator_token = spelling.removeprefix("operator").strip()
            if operator_token:
                token_spellings.add(operator_token)
        cursor_offsets = _cursor_extent_offsets(
            cursor,
            filename,
            byte_to_char,
        )
        if cursor_offsets is not None:
            cursor_start, cursor_end = cursor_offsets
            if not any(
                value
                and source.find(value, cursor_start, cursor_end) >= 0
                for value in token_spellings
            ):
                return []
        token_spelling_bytes = {
            value.encode(source_encoding, errors="strict")
            for value in token_spellings
        }
        matches: list[tuple[int, int]] = []
        for token in cursor.get_tokens():
            byte_start = int(token.extent.start.offset)
            byte_end = int(token.extent.end.offset)
            if (
                byte_start < 0
                or byte_start >= byte_end
                or byte_end > len(source_bytes)
                or source_bytes[byte_start:byte_end] not in token_spelling_bytes
            ):
                continue
            span = _char_span(byte_start, byte_end)
            if span is not None:
                matches.append(span)
        return matches[:1]

    def _symbol_key_for_reference(
        ref: Cursor,
        *,
        relation: str,
    ) -> str | None:
        try:
            return symbol_identity_for_cursor(
                ref,
                project_dir=project_dir,
                project=stable_project_id,
                fallback_file=fallback_file,
            )[0]
        except _UnknownExternalProviderError as exc:
            _record_external_reference_omission(
                external_reference_omissions,
                relation=relation,
                cursor=ref,
                error=exc,
            )
            return None

    def _call_target_spans(
        call_cursor: Cursor,
        target: Cursor,
        target_key: str,
    ) -> list[tuple[int, int]]:
        target_spellings = {target.spelling}
        if target.spelling.startswith("operator"):
            operator_token = target.spelling.removeprefix("operator").strip()
            if operator_token:
                target_spellings.add(operator_token)
        call_offsets = _cursor_extent_offsets(
            call_cursor,
            filename,
            byte_to_char,
        )
        if call_offsets is not None:
            call_start, call_end = call_offsets
            if not any(
                spelling
                and source.find(spelling, call_start, call_end) >= 0
                for spelling in target_spellings
            ):
                # Macro expansions can expose enormous synthetic call ASTs
                # whose callee token does not exist in the original source
                # extent. A recursive search cannot produce a valid explicit
                # source span in that case.
                return []

        def find_reference(node: Cursor) -> list[tuple[int, int]]:
            for child in node.get_children():
                child_kind = _cursor_kind(child)
                if child_kind in _REFERENCE_KINDS:
                    child_target = child.referenced
                    if (
                        child_target is not None
                        and child_target.spelling
                        and _symbol_key_for_reference(
                            child_target,
                            relation="semantic_call",
                        )
                        == target_key
                    ):
                        spans = _cursor_name_spans(child, child_target.spelling)
                        if spans:
                            return spans
                if child_kind not in _CALL_KINDS:
                    spans = find_reference(child)
                    if spans:
                        return spans
            return []

        return find_reference(call_cursor) or _cursor_name_spans(
            call_cursor,
            target.spelling,
        )

    def _annotate(
        values: list[int],
        spans: Iterable[tuple[int, int]],
        value: int,
    ) -> None:
        for span_start, span_end in spans:
            values[span_start:span_end] = [value] * (span_end - span_start)

    def _visit(cursor):
        """Walk the AST and annotate char ranges."""
        loc = cursor.location
        if not loc or not loc.file:
            return
        if loc.file.name != filename:
            return

        kind = _cursor_kind(cursor)

        # Symbol identification: get qualified name for definitions and refs
        qname = ""
        if kind in _DEFINITION_KINDS and cursor.is_definition():
            qname = get_qualified_name(cursor)
            if qname:
                symbol_key, _usr, _sig = symbol_identity_for_cursor(
                    cursor,
                    project_dir=project_dir,
                    project=stable_project_id,
                    fallback_file=fallback_file,
                )
                sym_id = _register_symbol_key(symbol_key, cursor)
                spans = _cursor_name_spans(cursor, cursor.spelling)
                _annotate(symbol_ids, spans, sym_id)
                _annotate(def_use, spans, DEF_USE_DEF)

        elif kind in _REFERENCE_KINDS:
            ref = cursor.referenced
            if ref and ref.spelling:
                qname = get_qualified_name(ref)
                if qname:
                    symbol_key = _symbol_key_for_reference(
                        ref,
                        relation="semantic_symbol",
                    )
                    if symbol_key is not None:
                        sym_id = _register_symbol_key(symbol_key, ref)
                        spans = _cursor_name_spans(cursor, ref.spelling)
                        _annotate(symbol_ids, spans, sym_id)
                        _annotate(def_use, spans, DEF_USE_USE)

        # Call target annotation
        if kind in _CALL_KINDS:
            ref = cursor.referenced
            if ref and ref.spelling:
                target_qname = get_qualified_name(ref)
                if target_qname:
                    symbol_key = _symbol_key_for_reference(
                        ref,
                        relation="semantic_call",
                    )
                    if symbol_key is not None:
                        target_id = _register_symbol_key(symbol_key, ref)
                        _annotate(
                            call_targets,
                            _call_target_spans(cursor, ref, symbol_key),
                            target_id,
                        )

        # Type reference annotation
        if kind in _TYPE_REF_KINDS:
            ref = cursor.referenced
            if ref and ref.spelling:
                ref_qname = get_qualified_name(ref)
                if ref_qname:
                    symbol_key = _symbol_key_for_reference(
                        ref,
                        relation="semantic_type",
                    )
                    if symbol_key is not None:
                        ref_id = _register_symbol_key(symbol_key, ref)
                        _annotate(
                            type_refs,
                            _cursor_name_spans(cursor, ref.spelling),
                            ref_id,
                        )

        # Recurse into children
        for child in cursor.get_children():
            _visit(child)

    for child in tu.cursor.get_children():
        _visit(child)

    return {
        "symbol_ids": symbol_ids,
        "call_targets": call_targets,
        "type_refs": type_refs,
        "def_use": def_use,
        SYMBOL_IDENTITIES_COLUMN: identity_registry.records(),
    }


def extract_semantic_metadata_from_parts(
    full_text: str,
    parts_info: list[PartInfo],
    index: ProjectIndex,
    part_functions: dict[int, FunctionDef] | None = None,
    part_semantic_arrays: dict[int, dict[str, object]] | None = None,
) -> dict:
    """Produce semantic metadata from pre-extracted parts without running libclang.

    This is the offline path used when the enriched doc is built from
    already-parsed FunctionDef objects. It assigns symbol_ids and def_use
    based on canonical part metadata and the call graph in the ProjectIndex.
    A display qname may resolve an already-indexed unique symbol but never
    synthesizes identity on its own.

    Returns:
        dict with keys 'symbol_ids', 'call_targets', 'type_refs', 'def_use',
        each a list[int] of length len(full_text).
    """
    text_len = len(full_text)
    symbol_ids = [0] * text_len
    call_targets = [0] * text_len
    type_refs = [0] * text_len
    def_use_arr = [0] * text_len

    def _named_token_spans(text: str, name: str) -> list[tuple[int, int]]:
        simple_name = name.split("::")[-1]
        if not simple_name:
            return []
        return [
            (match.start(), match.end())
            for match in _iter_code_identifiers(text)
            if match.group(0) == simple_name
        ]

    def _annotate_part_spans(
        values: list[int],
        spans: Iterable[tuple[int, int]],
        value: int,
        *,
        part_offset: int,
    ) -> None:
        for start, end in spans:
            values[part_offset + start:part_offset + end] = [value] * (end - start)

    offset = 0
    for i, part in enumerate(parts_info):
        part_text = part[0]
        kind = part[1]
        part_len = len(part_text)
        if offset + part_len > text_len:
            break

        qname = part[4] if len(part) > 4 else None
        symbol_key = _part_symbol_key(part) or (
            index.resolve_function_key(qname) if isinstance(qname, str) else None
        ) or (
            index.resolve_type_key(qname) if isinstance(qname, str) else None
        )

        exact_semantics = (part_semantic_arrays or {}).get(i)
        exact_applied = False
        if exact_semantics is not None:
            exact_values = (
                exact_semantics.get("symbol_ids"),
                exact_semantics.get("call_targets"),
                exact_semantics.get("type_refs"),
                exact_semantics.get("def_use"),
            )
            if all(value is not None and len(value) == part_len for value in exact_values):
                symbol_ids[offset:offset + part_len] = list(exact_values[0])
                call_targets[offset:offset + part_len] = list(exact_values[1])
                type_refs[offset:offset + part_len] = list(exact_values[2])
                def_use_arr[offset:offset + part_len] = list(exact_values[3])
                exact_applied = True

        if qname:
            func_def = (part_functions or {}).get(i)
            if func_def is None:
                func_key = index.resolve_function_key(symbol_key) or index.resolve_function_key(qname)
                func_def = index.functions.get(func_key) if func_key is not None else None
            if not exact_applied and (
                func_def is not None
                and len(func_def.semantic_symbol_ids) == part_len
                and len(func_def.semantic_call_targets) == part_len
                and len(func_def.semantic_type_refs) == part_len
                and len(func_def.semantic_def_use) == part_len
            ):
                symbol_ids[offset:offset + part_len] = func_def.semantic_symbol_ids[:]
                call_targets[offset:offset + part_len] = func_def.semantic_call_targets[:]
                type_refs[offset:offset + part_len] = func_def.semantic_type_refs[:]
                def_use_arr[offset:offset + part_len] = func_def.semantic_def_use[:]
            elif not exact_applied and symbol_key:
                sym_id = _compute_symbol_id(symbol_key)
                definition_spans = _named_token_spans(part_text, str(part[3]))
                if func_def is not None and definition_spans:
                    open_paren = part_text.find("(")
                    before_parameters = [
                        span for span in definition_spans if span[0] < open_paren
                    ]
                    definition_spans = [
                        (before_parameters or definition_spans)[-1]
                    ]
                elif definition_spans:
                    definition_spans = definition_spans[:1]
                _annotate_part_spans(
                    symbol_ids,
                    definition_spans,
                    sym_id,
                    part_offset=offset,
                )
                _annotate_part_spans(
                    def_use_arr,
                    definition_spans,
                    DEF_USE_DEF,
                    part_offset=offset,
                )

            if (
                not exact_applied
                and func_def
                and not any(call_targets[offset:offset + part_len])
            ):
                def _mark(arr, name, value):
                    if not name or not value:
                        return
                    _annotate_part_spans(
                        arr,
                        _named_token_spans(part_text, name),
                        value,
                        part_offset=offset,
                    )
                for callee_qname in func_def.callees:
                    callee_key = index.resolve_function_key(callee_qname)
                    if callee_key is None:
                        continue
                    _mark(
                        call_targets,
                        callee_qname.split('::')[-1],
                        _compute_symbol_id(callee_key),
                    )
                for type_qname in getattr(func_def, 'referenced_types', []):
                    type_key = index.resolve_type_key(type_qname)
                    if type_key is None:
                        continue
                    _mark(
                        type_refs,
                        type_qname.split('::')[-1],
                        _compute_symbol_id(type_key),
                    )

        elif kind == 1:  # PREAMBLE
            # Preamble: mark as use sites (references to included entities)
            for ci in range(offset, offset + part_len):
                def_use_arr[ci] = DEF_USE_USE

        offset += part_len
        if i < len(parts_info) - 1:
            offset += 2  # "\n\n" separator

    return {
        "symbol_ids": symbol_ids,
        "call_targets": call_targets,
        "type_refs": type_refs,
        "def_use": def_use_arr,
    }


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_RAW_LITERAL_PREFIXES = ('u8R"', 'uR"', 'UR"', 'LR"', 'R"')
_QUOTED_LITERAL_PREFIXES = (
    'u8"', "u8'", 'u"', "u'", 'U"', "U'", 'L"', "L'", '"', "'",
)
_IDENTIFIER_CHARS = frozenset(
    "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def _mask_non_code(text: str) -> str:
    """Blank comments and literals while preserving offsets and newlines."""
    masked = list(text)
    text_len = len(text)

    def blank(start: int, end: int) -> None:
        for index in range(start, min(end, text_len)):
            if text[index] not in "\r\n":
                masked[index] = " "

    def prefixed_at(index: int, prefixes: Sequence[str]) -> str | None:
        for prefix in prefixes:
            if not text.startswith(prefix, index):
                continue
            if (
                prefix[0].isalpha()
                and index > 0
                and text[index - 1] in _IDENTIFIER_CHARS
            ):
                continue
            return prefix
        return None

    def include_literal_suffix(end: int) -> int:
        while end < text_len and text[end] in _IDENTIFIER_CHARS:
            end += 1
        return end

    index = 0
    while index < text_len:
        if text.startswith("//", index):
            end = index + 2
            while True:
                newline = text.find("\n", end)
                if newline < 0:
                    end = text_len
                    break
                slash_index = newline - 1
                if slash_index >= index and text[slash_index] == "\r":
                    slash_index -= 1
                continued = slash_index >= index and text[slash_index] == "\\"
                end = newline + 1
                if not continued:
                    break
            blank(index, end)
            index = end
            continue

        if text.startswith("/*", index):
            close = text.find("*/", index + 2)
            end = text_len if close < 0 else close + 2
            blank(index, end)
            index = end
            continue

        raw_prefix = prefixed_at(index, _RAW_LITERAL_PREFIXES)
        if raw_prefix is not None:
            delimiter_start = index + len(raw_prefix)
            open_paren = text.find("(", delimiter_start, delimiter_start + 17)
            if open_paren >= 0:
                delimiter = text[delimiter_start:open_paren]
                if not any(char in " ()\\\t\v\f\r\n" for char in delimiter):
                    marker = ")" + delimiter + '"'
                    close = text.find(marker, open_paren + 1)
                    end = text_len if close < 0 else close + len(marker)
                    end = include_literal_suffix(end)
                    blank(index, end)
                    index = end
                    continue

        quoted_prefix = prefixed_at(index, _QUOTED_LITERAL_PREFIXES)
        if quoted_prefix is not None:
            quote = quoted_prefix[-1]
            end = index + len(quoted_prefix)
            while end < text_len:
                char = text[end]
                if char == "\\":
                    if end + 1 < text_len and text[end + 1] == "\r":
                        end += 3 if end + 2 < text_len and text[end + 2] == "\n" else 2
                    else:
                        end += 2
                    continue
                if char == quote:
                    end += 1
                    break
                if char in "\r\n":
                    break
                end += 1
            end = include_literal_suffix(end)
            blank(index, end)
            index = end
            continue

        index += 1

    return "".join(masked)


def _split_source_and_masked_lines(text: str) -> tuple[list[str], list[str]]:
    """Split source and its mask at the exact same physical boundaries."""

    masked = _mask_non_code(text)
    if len(masked) != len(text):
        raise RuntimeError(
            "non-code mask changed source length: "
            f"source={len(text)} masked={len(masked)}"
        )
    source_lines = text.splitlines(keepends=True)
    code_lines: list[str] = []
    offset = 0
    for line in source_lines:
        end = offset + len(line)
        code_lines.append(masked[offset:end])
        offset = end
    if offset != len(text):
        raise RuntimeError(
            "source line split did not cover complete input: "
            f"covered={offset} source={len(text)}"
        )
    return source_lines, code_lines


def _iter_code_identifiers(text: str) -> Iterator[re.Match[str]]:
    return _IDENTIFIER_RE.finditer(_mask_non_code(text))


def _apply_char_span(values: list[int], start: int, end: int, value: int) -> None:
    for idx in range(max(0, start), min(len(values), end)):
        values[idx] = value


def _stable_entity_id(name: str) -> int:
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _parse_macro_signature(
    macro_text: str,
) -> tuple[str, tuple[int, int], dict[str, tuple[int, int]], int] | None:
    match = _DEFINE_RE.match(macro_text)
    if match is None:
        return None
    name = match.group(1)
    name_span = (match.start(1), match.end(1))
    params: dict[str, tuple[int, int]] = {}
    body_start = match.end()
    if body_start < len(macro_text) and macro_text[body_start] == "(":
        close = macro_text.find(")", body_start + 1)
        newline = macro_text.find("\n", body_start + 1)
        if close != -1 and (newline == -1 or close < newline):
            param_text = macro_text[body_start + 1:close]
            base = body_start + 1
            search_from = 0
            for raw_param in param_text.split(","):
                param = raw_param.strip()
                if not param or param == "...":
                    search_from += len(raw_param) + 1
                    continue
                if param.endswith("..."):
                    param = param[:-3].strip()
                param_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*$", param)
                if param_match is None:
                    search_from += len(raw_param) + 1
                    continue
                param_name = param_match.group(0)
                local_start = param_text.find(param_name, search_from)
                if local_start == -1:
                    local_start = param_text.find(param_name)
                if local_start != -1:
                    params[param_name] = (
                        base + local_start,
                        base + local_start + len(param_name),
                    )
                search_from += len(raw_param) + 1
            body_start = close + 1
    return name, name_span, params, body_start


def _macro_body_dependency_names(
    macro_text: str,
    *,
    macro_name: str,
    params: Sequence[str] | None = None,
) -> list[str]:
    """Macro identifiers referenced by the replacement list.

    Scanner ``MacroDef.text`` may include condition and undef lines before the
    real ``#define``.  Parse the logical define block inside that text, then
    inspect only the replacement list so parameter names and the macro's own
    name do not become expansion dependencies.
    """

    blocks = extract_macro_blocks(macro_text)
    if blocks:
        matching = [
            block_text for _start, _end, name, block_text in blocks
            if name == macro_name
        ]
        source = matching[-1] if matching else blocks[-1][3]
    else:
        source = macro_text

    parsed = _parse_macro_signature(source)
    if parsed is None:
        return []
    name, _name_span, parsed_params, body_start = parsed
    ignored = {name, macro_name, *(params or ()), *parsed_params}
    out: list[str] = []
    seen: set[str] = set()
    body = source[body_start:]
    for match in _iter_code_identifiers(body):
        dep_name = match.group(0)
        if dep_name in ignored or dep_name.lower() in _CONDITION_SKIP_IDENTIFIERS:
            continue
        if dep_name in seen:
            continue
        seen.add(dep_name)
        out.append(dep_name)
    return out


def _macro_part_metadata(macro: MacroDef) -> dict[str, object]:
    previous_sequence = macro.previous.sequence if macro.previous is not None else None
    return {
        "name": macro.name,
        "file": macro.file,
        "line": macro.line,
        "project_id": macro.project_id,
        "visible_in_file": macro.visible_in_file,
        "visible_line": macro.visible_line,
        "sequence": macro.sequence,
        "previous_sequence": previous_sequence,
        "condition_names": list(macro.condition_names),
        "body_macro_names": _macro_body_dependency_names(
            macro.text,
            macro_name=macro.name,
            params=macro.params,
        ),
    }


def _macro_route_part(
    part_text: str,
    *,
    offset: int,
    metadata: dict[str, object] | None,
) -> dict[str, object] | None:
    if not metadata:
        return None
    target_name = metadata.get("name")
    if not isinstance(target_name, str) or not target_name:
        return None
    for start, end, name, macro_text in extract_macro_blocks(part_text):
        if name != target_name:
            continue
        parsed = _parse_macro_signature(macro_text)
        if parsed is None:
            return None
        _name, name_span, _params, _body_start = parsed
        condition_names = [
            str(value) for value in metadata.get("condition_names", [])
            if isinstance(value, str) and value
        ]
        condition_name_set = set(condition_names)
        condition_spans: list[dict[str, object]] = []
        if condition_name_set:
            prefix = part_text[:start]
            for match in _iter_code_identifiers(prefix):
                if match.group(0) not in condition_name_set:
                    continue
                condition_spans.append(
                    {
                        "name": match.group(0),
                        "start": offset + match.start(),
                        "end": offset + match.end(),
                    }
                )
        condition_directive_spans: list[dict[str, object]] = []
        prefix = part_text[:start]
        for line_match in re.finditer(
            r"(?m)^\s*#\s*(if|ifdef|ifndef|elif|else)\b",
            prefix,
        ):
            condition_directive_spans.append(
                {
                    "name": line_match.group(1),
                    "start": offset + line_match.start(1),
                    "end": offset + line_match.end(1),
                    "role": "keyword",
                }
            )
        suffix = part_text[end:]
        for line_match in re.finditer(r"(?m)^\s*#\s*(endif)\b", suffix):
            condition_directive_spans.append(
                {
                    "name": line_match.group(1),
                    "start": offset + end + line_match.start(1),
                    "end": offset + end + line_match.end(1),
                    "role": "keyword",
                }
            )
        route_part = dict(metadata)
        route_part.update(
            {
                "name_start": offset + start + name_span[0],
                "name_end": offset + start + name_span[1],
                "macro_start": offset + start,
                "macro_end": offset + end,
                "condition_spans": condition_spans,
                "condition_directive_spans": condition_directive_spans,
            }
        )
        return route_part
    return None


def _source_line_for_text_offset(text: str, *, start_line: int, char_offset: int) -> int:
    return int(start_line) + text.count("\n", 0, max(0, char_offset))


def _macro_invocation_route_parts(
    part_text: str,
    *,
    offset: int,
    index: ProjectIndex,
    target_file: str,
    start_line: int,
) -> list[dict[str, object]]:
    """Precise use-site -> macro-definition routes for one source chunk.

    The assembled training document puts all pulled macro definitions before the
    source chunks.  Pure lexical remapping inside the assembled text would route
    every use to the last pulled definition, which is wrong for dependency
    functions that live before a later ``#undef/#define`` window.  Use the
    original source line for each identifier so each invocation points at the
    macro definition visible at that source location.
    """

    macro_ranges = [
        (start, end) for start, end, _name, _macro_text in extract_macro_blocks(part_text)
    ]
    routes: list[dict[str, object]] = []
    for match in _iter_code_identifiers(part_text):
        if any(start <= match.start() < end for start, end in macro_ranges):
            continue
        source_line = _source_line_for_text_offset(
            part_text,
            start_line=start_line,
            char_offset=match.start(),
        )
        macro = _select_visible_macro(
            index,
            match.group(0),
            target_file=target_file,
            max_line=source_line,
        )
        if macro is None:
            continue
        routes.append(
            {
                "name": macro.name,
                "start": offset + match.start(),
                "end": offset + match.end(),
                "target_sequence": macro.sequence,
            }
        )
    return routes


def _cpp_domain_sidecars(
    text: str,
    index: ProjectIndex | None = None,
    *,
    source_doc_id: int | None = None,
    source_path: str = "assembled.cpp",
    project_id: str | None = None,
    fragment_sources: list[
        tuple[int, int, str, int] | tuple[int, int, str, int, str | None]
    ] | None = None,
    macro_parts: list[dict[str, object]] | None = None,
    macro_invocations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Char-aligned domain sidecars for C/C++ code documents.

    The token materializer uses ``domain_kind`` to insert the existing
    <CPP_CODE_START>/<CPP_CODE_END> reserved tokens.  The dense char arrays keep
    the C++ domain distinct from build/shell/diagnostic docs without pretending
    that we have a separate build-style parser for C++ tokens here; clang graph
    routes remain in the code-specific sidecars.
    """
    from cppmega.data.domain_schema import (
        DomainEdgeKind,
        DomainKind,
        DomainRoleKind,
        ParseConfidence,
    )

    text_len = len(text)
    resolved_source_doc_id = int(source_doc_id or 1)
    domain = int(DomainKind.CPP)
    role_ids = [0] * text_len
    entity_ids = [0] * text_len
    domain_edges: list[dict[str, int]] = []
    macro_definitions_by_name: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    macro_occurrences: list[dict[str, int | str]] = []
    macro_ranges: list[tuple[int, int]] = []
    use_precise_macro_graph = bool(macro_parts)

    def _condition_identifier_spans(define_start: int) -> list[tuple[str, int, int]]:
        block_start = text.rfind("\n\n", 0, define_start)
        block_start = 0 if block_start == -1 else block_start + 2
        prefix = text[block_start:define_start]
        spans: list[tuple[str, int, int]] = []
        for line_match in re.finditer(r"(?m)^\s*#\s*(if|ifdef|ifndef|elif)\b.*$", prefix):
            line = line_match.group(0)
            abs_line_start = block_start + line_match.start()
            for ident in _iter_code_identifiers(line):
                name = ident.group(0)
                if name.lower() in _CONDITION_SKIP_IDENTIFIERS or name in {
                    "if",
                    "ifdef",
                    "ifndef",
                    "elif",
                }:
                    continue
                spans.append((name, abs_line_start + ident.start(), abs_line_start + ident.end()))
        return spans

    for start, end, _name, macro_text in extract_macro_blocks(text):
        parsed = _parse_macro_signature(macro_text)
        if parsed is None:
            continue
        name, name_span, params, body_start = parsed
        entity = _stable_entity_id(f"macro:{name}")
        macro_ranges.append((start, end))
        occurrence = {
            "name": name,
            "name_start": start + name_span[0],
            "name_end": start + name_span[1],
        }
        macro_definitions_by_name[name].append(occurrence)
        macro_occurrences.append(occurrence)
        _apply_char_span(
            role_ids,
            start + name_span[0],
            start + name_span[1],
            int(DomainRoleKind.IDENTIFIER),
        )
        _apply_char_span(entity_ids, start + name_span[0], start + name_span[1], entity)
        for param_name, (param_start, param_end) in params.items():
            param_entity = _stable_entity_id(f"macro:{name}:param:{param_name}")
            _apply_char_span(
                role_ids,
                start + param_start,
                start + param_end,
                int(DomainRoleKind.VARIABLE),
            )
            _apply_char_span(entity_ids, start + param_start, start + param_end, param_entity)
            body = macro_text[body_start:]
            for use in _iter_code_identifiers(body):
                if use.group(0) != param_name:
                    continue
                use_start = start + body_start + use.start()
                use_end = start + body_start + use.end()
                _apply_char_span(role_ids, use_start, use_end, int(DomainRoleKind.VARIABLE))
                _apply_char_span(entity_ids, use_start, use_end, param_entity)
                domain_edges.append(
                    {
                        "from_char": use_start,
                        "to_char": start + param_start,
                        "kind": int(DomainEdgeKind.MACRO_PARAM_USE),
                    }
                )
        if not use_precise_macro_graph:
            for cond_name, cond_start, cond_end in _condition_identifier_spans(start):
                cond_entity = _stable_entity_id(f"macro_condition:{cond_name}")
                _apply_char_span(role_ids, cond_start, cond_end, int(DomainRoleKind.VARIABLE))
                _apply_char_span(entity_ids, cond_start, cond_end, cond_entity)
                domain_edges.append(
                    {
                        "from_char": start + name_span[0],
                        "to_char": cond_start,
                        "kind": int(DomainEdgeKind.MACRO_CONDITION),
                    }
                )

    if use_precise_macro_graph:
        precise_parts = [
            part for part in (macro_parts or [])
            if isinstance(part.get("name"), str)
            and isinstance(part.get("sequence"), int)
            and isinstance(part.get("name_start"), int)
            and isinstance(part.get("name_end"), int)
        ]
        precise_parts.sort(key=lambda item: int(item["sequence"]))
        by_sequence = {int(item["sequence"]): item for item in precise_parts}
        by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in precise_parts:
            by_name[str(item["name"])].append(item)
        previous_by_sequence: dict[int, dict[str, object]] = {}
        for previous, current in zip(precise_parts, precise_parts[1:]):
            previous_by_sequence[int(current["sequence"])] = previous

        def _iter_condition_spans(item: dict[str, object]) -> Iterator[dict[str, object]]:
            for key in ("condition_spans", "condition_directive_spans"):
                for span in item.get(key, []):
                    if isinstance(span, dict):
                        yield span

        def _latest_prior_part(name: str, before_sequence: int) -> dict[str, object] | None:
            candidates = [
                candidate for candidate in by_name.get(name, [])
                if int(candidate["sequence"]) < before_sequence
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda candidate: int(candidate["sequence"]))

        def _expansion_context_targets(
            item: dict[str, object],
        ) -> list[tuple[int, int]]:
            current_sequence = int(item["sequence"])
            targets: list[tuple[int, int]] = []
            for cond_span in _iter_condition_spans(item):
                cond_start = cond_span.get("start")
                if isinstance(cond_start, int):
                    targets.append((cond_start, int(DomainEdgeKind.MACRO_EXPANSION_CONDITION)))
            for cond_name in item.get("condition_names", []):
                target = _latest_prior_part(str(cond_name), current_sequence)
                if target is not None:
                    targets.append((
                        int(target["name_start"]),
                        int(DomainEdgeKind.MACRO_EXPANSION_CONDITION),
                    ))
            for body_name in item.get("body_macro_names", []):
                target = _latest_prior_part(str(body_name), current_sequence)
                if target is not None:
                    targets.append((
                        int(target["name_start"]),
                        int(DomainEdgeKind.MACRO_EXPANSION_INCLUDE_ORDER),
                    ))
            previous_sequence = item.get("previous_sequence")
            seen_previous: set[int] = set()
            while isinstance(previous_sequence, int) and previous_sequence not in seen_previous:
                seen_previous.add(previous_sequence)
                previous = by_sequence.get(previous_sequence)
                if previous is None:
                    break
                targets.append((
                    int(previous["name_start"]),
                    int(DomainEdgeKind.MACRO_EXPANSION_REDEFINITION),
                ))
                previous_sequence = previous.get("previous_sequence")
            include_previous = previous_by_sequence.get(current_sequence)
            if include_previous is not None:
                targets.append((
                    int(include_previous["name_start"]),
                    int(DomainEdgeKind.MACRO_EXPANSION_INCLUDE_ORDER),
                ))
            return targets

        def _emit_macro_invocation(
            *,
            use_start: int,
            use_end: int,
            selected: dict[str, object],
        ) -> None:
            name = str(selected["name"])
            entity = _stable_entity_id(f"macro:{name}")
            _apply_char_span(role_ids, use_start, use_end, int(DomainRoleKind.IDENTIFIER))
            _apply_char_span(entity_ids, use_start, use_end, entity)
            domain_edges.append(
                {
                    "from_char": use_start,
                    "to_char": int(selected["name_start"]),
                    "kind": int(DomainEdgeKind.MACRO_INVOCATION),
                }
            )
            for target_start, edge_kind in _expansion_context_targets(selected):
                domain_edges.append(
                    {
                        "from_char": use_start,
                        "to_char": target_start,
                        "kind": edge_kind,
                    }
                )

        for item in precise_parts:
            name = str(item["name"])
            name_start = int(item["name_start"])
            name_end = int(item["name_end"])
            current_sequence = int(item["sequence"])
            for cond_span in _iter_condition_spans(item):
                cond_name = str(cond_span.get("name", ""))
                cond_start = cond_span.get("start")
                cond_end = cond_span.get("end")
                if not isinstance(cond_start, int) or not isinstance(cond_end, int):
                    continue
                entity_prefix = (
                    "macro_branch"
                    if cond_span.get("role") == "keyword"
                    else "macro_condition"
                )
                role = (
                    int(DomainRoleKind.KEYWORD)
                    if cond_span.get("role") == "keyword"
                    else int(DomainRoleKind.VARIABLE)
                )
                cond_entity = _stable_entity_id(f"{entity_prefix}:{cond_name}")
                _apply_char_span(role_ids, cond_start, cond_end, role)
                _apply_char_span(entity_ids, cond_start, cond_end, cond_entity)
                domain_edges.append(
                    {
                        "from_char": name_start,
                        "to_char": cond_start,
                        "kind": int(DomainEdgeKind.MACRO_CONDITION),
                    }
                )

            for cond_name in item.get("condition_names", []):
                cond_defs = [
                    candidate for candidate in by_name.get(str(cond_name), [])
                    if int(candidate["sequence"]) < current_sequence
                ]
                if not cond_defs:
                    continue
                target = max(cond_defs, key=lambda candidate: int(candidate["sequence"]))
                domain_edges.append(
                    {
                        "from_char": name_start,
                        "to_char": int(target["name_start"]),
                        "kind": int(DomainEdgeKind.MACRO_CONDITION),
                    }
                )

            previous_sequence = item.get("previous_sequence")
            if isinstance(previous_sequence, int) and previous_sequence in by_sequence:
                previous = by_sequence[previous_sequence]
                domain_edges.append(
                    {
                        "from_char": name_start,
                        "to_char": int(previous["name_start"]),
                        "kind": int(DomainEdgeKind.MACRO_REDEFINITION),
                    }
                )

            _apply_char_span(role_ids, name_start, name_end, int(DomainRoleKind.IDENTIFIER))
            _apply_char_span(entity_ids, name_start, name_end, _stable_entity_id(f"macro:{name}"))

        for previous, current in zip(precise_parts, precise_parts[1:]):
            domain_edges.append(
                {
                    "from_char": int(current["name_start"]),
                    "to_char": int(previous["name_start"]),
                    "kind": int(DomainEdgeKind.MACRO_INCLUDE_ORDER),
                }
            )

        precise_ranges = [
            (int(item["macro_start"]), int(item["macro_end"]))
            for item in precise_parts
            if isinstance(item.get("macro_start"), int)
            and isinstance(item.get("macro_end"), int)
        ]

        routed_invocation_spans: set[tuple[int, int]] = set()
        for route in macro_invocations or []:
            use_start = route.get("start")
            use_end = route.get("end")
            target_sequence = route.get("target_sequence")
            if (
                not isinstance(use_start, int)
                or not isinstance(use_end, int)
                or not isinstance(target_sequence, int)
            ):
                continue
            selected = by_sequence.get(target_sequence)
            if selected is None:
                continue
            _emit_macro_invocation(
                use_start=use_start,
                use_end=use_end,
                selected=selected,
            )
            routed_invocation_spans.add((use_start, use_end))

        for match in _iter_code_identifiers(text):
            if (match.start(), match.end()) in routed_invocation_spans:
                continue
            definitions = by_name.get(match.group(0), [])
            if not definitions:
                continue
            if any(start <= match.start() < end for start, end in precise_ranges):
                continue
            selected = max(definitions, key=lambda item: int(item["sequence"]))
            _emit_macro_invocation(
                use_start=match.start(),
                use_end=match.end(),
                selected=selected,
            )
    else:
        macro_occurrences.sort(key=lambda item: int(item["name_start"]))
        for previous, current in zip(macro_occurrences, macro_occurrences[1:]):
            domain_edges.append(
                {
                    "from_char": int(current["name_start"]),
                    "to_char": int(previous["name_start"]),
                    "kind": int(DomainEdgeKind.MACRO_INCLUDE_ORDER),
                }
            )
        for definitions in macro_definitions_by_name.values():
            definitions.sort(key=lambda item: int(item["name_start"]))
            for previous, current in zip(definitions, definitions[1:]):
                domain_edges.append(
                    {
                        "from_char": int(current["name_start"]),
                        "to_char": int(previous["name_start"]),
                        "kind": int(DomainEdgeKind.MACRO_REDEFINITION),
                    }
                )

    if not use_precise_macro_graph and index is not None and macro_definitions_by_name:
        for match in _iter_code_identifiers(text):
            definitions = macro_definitions_by_name.get(match.group(0), [])
            if not definitions:
                continue
            if any(start <= match.start() < end for start, end in macro_ranges):
                continue
            selected: dict[str, int | str] | None = None
            for definition in definitions:
                if int(definition["name_start"]) < match.start():
                    selected = definition
                else:
                    break
            if selected is None:
                continue
            entity = _stable_entity_id(f"macro:{match.group(0)}")
            _apply_char_span(role_ids, match.start(), match.end(), int(DomainRoleKind.IDENTIFIER))
            _apply_char_span(entity_ids, match.start(), match.end(), entity)
            domain_edges.append(
                {
                    "from_char": match.start(),
                    "to_char": int(selected["name_start"]),
                    "kind": int(DomainEdgeKind.MACRO_INVOCATION),
                }
            )

    deduped_domain_edges: list[dict[str, int]] = []
    seen_edges: set[tuple[int, int, int]] = set()
    for edge in domain_edges:
        key = (int(edge["from_char"]), int(edge["to_char"]), int(edge["kind"]))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        deduped_domain_edges.append(edge)

    from cppmega.data.domain_ingestion import parse_domain_document

    lexical = parse_domain_document(
        source_path,
        text,
        source_doc_id=resolved_source_doc_id,
    ).to_enriched_document()
    if fragment_sources:
        source_doc_ids = [0] * text_len
        source_identity_ids = [0] * text_len
        registry: dict[int, dict[str, int | str]] = {}
        for fragment in fragment_sources:
            start, end, fragment_path, row_local_id = fragment[:4]
            fragment_project_id = fragment[4] if len(fragment) >= 5 else project_id
            identity = _source_identity_for_project_path(
                fragment_path,
                project_id=fragment_project_id,
            )
            entry = identity.as_dict()
            previous = registry.get(identity.source_identity_id)
            if previous is not None and previous != entry:
                raise ValueError(
                    f"source identity uint64 collision for id {identity.source_identity_id}"
                )
            registry[identity.source_identity_id] = entry
            for char_idx in range(max(0, start), min(end, text_len)):
                source_doc_ids[char_idx] = int(row_local_id)
                source_identity_ids[char_idx] = identity.source_identity_id
        if any(value == 0 for value in source_doc_ids + source_identity_ids):
            raise ValueError("assembled source provenance does not cover every character")
        lexical["domain_source_doc_ids"] = source_doc_ids
        lexical["domain_source_identity_ids"] = source_identity_ids
        lexical["source_identity_registry"] = list(registry.values())
    lexical_domains = cast(list[int], lexical["domain_ids"])
    lexical_roles = cast(list[int], lexical["domain_role_ids"])
    lexical_entities = cast(list[int], lexical["domain_entity_ids"])
    lexical_confidence = cast(list[int], lexical["domain_confidence_ids"])
    for edge in cast(list[dict[str, int]], lexical["domain_edges"]):
        key = (int(edge["from_char"]), int(edge["to_char"]), int(edge["kind"]))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        deduped_domain_edges.append(edge)
    for char_idx in range(text_len):
        is_embedded = lexical_domains[char_idx] != domain
        if is_embedded or role_ids[char_idx] == int(DomainRoleKind.NONE):
            role_ids[char_idx] = lexical_roles[char_idx]
        if is_embedded or entity_ids[char_idx] == 0:
            entity_ids[char_idx] = lexical_entities[char_idx]

    cross_domain_edges = cast(list[dict[str, int]], lexical["cross_domain_edges"])
    embedded_domain_spans = cast(list[dict[str, int]], lexical["embedded_domain_spans"])
    confidence_ids = [int(ParseConfidence.EXACT)] * text_len
    for char_idx, lexical_domain in enumerate(lexical_domains):
        if lexical_domain != domain:
            confidence_ids[char_idx] = lexical_confidence[char_idx]

    return {
        "domain_kind": domain,
        "domain_ids": lexical_domains,
        "domain_role_ids": role_ids,
        "domain_entity_ids": entity_ids,
        "domain_scope_ids": [0] * text_len,
        "domain_source_doc_ids": lexical["domain_source_doc_ids"],
        "domain_source_identity_ids": lexical["domain_source_identity_ids"],
        "source_identity_registry": lexical["source_identity_registry"],
        "domain_confidence_ids": confidence_ids,
        "domain_edges": deduped_domain_edges,
        "build_edges": [],
        "shell_edges": [],
        "diagnostic_edges": [],
        "cross_domain_edges": cross_domain_edges,
        "embedded_domain_spans": embedded_domain_spans,
        "domain_parse_info": lexical["domain_parse_info"],
    }


def build_enriched_doc(
    parts_info: list[PartInfo],
    index: ProjectIndex,
    filepath: str | None = None,
    compile_args: list[str] | None = None,
    build_info: dict | None = None,
    *,
    project_id: str | None = None,
) -> dict:
    """Build enriched document from parts with metadata.

    Args:
        parts_info: list of (text, kind, dep_level, name, qname_or_none) tuples
            kind: 0=other, 1=preamble, 2=func_signature, 3=func_body,
                  4=class_decl, 5=class_member, 6=comment, 7=typedef, 8=namespace
        index: ProjectIndex for computing edges

    Returns:
        dict with text, structure_ids, chunk_boundaries, call_edges, type_edges,
        ast_depth, sibling_index, ast_node_type
    """
    stable_project_id = (
        require_project_identity(project_id, source="build_enriched_doc")
        if project_id is not None
        else None
    )
    texts = [p[0] for p in parts_info]
    full_text = '\n\n'.join(texts)
    text_len = len(full_text)

    structure_ids = [0] * text_len
    chunk_boundaries = []
    offset = 0

    # Map chunk_idx -> canonical symbol identity for edge computation.
    chunk_symbols = {}       # local function chunks (call-edge SOURCES; carry callees)
    chunk_target_symbols = {} # local funcs + cross-lib base-lib pull targets
    chunk_callees = {}
    chunk_all_symbols = {}   # ANY named chunk incl. type defs (type-edge targets)
    chunk_types = {}         # referenced_types per function chunk (type-edge sources)
    chunk_ranges: dict[int, tuple[int, int]] = {}
    call_target_id_rewrites: dict[int, dict[int, int]] = defaultdict(dict)
    macro_route_parts: list[dict[str, object]] = []
    macro_invocation_routes: list[dict[str, object]] = []
    crosslink_targets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate_part in parts_info:
        source = _part_dep_source(candidate_part)
        metadata = _part_symbol_metadata(candidate_part)
        candidate_qname = candidate_part[4] if len(candidate_part) > 4 else None
        if (
            source is not None
            and source.startswith("crosslib:")
            and metadata is not None
            and isinstance(candidate_qname, str)
        ):
            crosslink_targets[
                normalize_inline_namespace_qname(candidate_qname)
            ].append(metadata)

    def _resolve_pulled_crosslink(reference: str | SymbolReference) -> str | None:
        if isinstance(reference, dict):
            qname = normalize_inline_namespace_qname(
                str(reference.get("qname") or "")
            )
            candidates = list(crosslink_targets.get(qname, []))
            if not candidates:
                return None
            identity_filter_used = False
            usr = str(reference.get("usr") or "")
            reference_key = str(reference.get("symbol_key") or "")
            if usr:
                identity_filter_used = True
                candidates = [target for target in candidates if target.get("usr") == usr]
            signature = _normalize_signature_text(
                str(reference.get("canonical_signature") or "")
            )
            if signature:
                identity_filter_used = True
                candidates = [
                    target
                    for target in candidates
                    if _normalize_signature_text(
                        str(target.get("canonical_signature") or "")
                    ) == signature
                ]
            symbol_kind = str(reference.get("symbol_kind") or "")
            if symbol_kind:
                identity_filter_used = True
                candidates = [
                    target
                    for target in candidates
                    if str(target.get("kind") or "") == symbol_kind
                ]
            if reference_key:
                identity_filter_used = True
                candidates = [
                    target
                    for target in candidates
                    if target.get("symbol_key") == reference_key
                ]
            provider = str(reference.get("provider") or "")
            if provider:
                identity_filter_used = True
                candidates = [
                    target
                    for target in candidates
                    if str(target.get("provider") or "") == provider
                ]
            include_provenance = str(reference.get("include_provenance") or "")
            if include_provenance:
                identity_filter_used = True
                candidates = [
                    target
                    for target in candidates
                    if str(target.get("include_provenance") or "")
                    == include_provenance
                ]
            if not identity_filter_used:
                return None
        else:
            qname = normalize_inline_namespace_qname(reference)
            candidates = [
                target
                for target in crosslink_targets.get(qname, [])
                if str(target.get("canonical_signature") or "").startswith(
                    "legacy-kind="
                )
            ]
        if len(candidates) > 1:
            raise SymbolIdentityError(
                f"ambiguous pulled crosslink for {qname!r}: "
                f"{[target.get('symbol_key') for target in candidates[:8]]}"
            )
        if len(candidates) != 1:
            return None
        key = candidates[0].get("symbol_key")
        return key if isinstance(key, str) and key else None

    fragment_sources: list[tuple[int, int, str, int, str | None]] = []

    for i, part in enumerate(parts_info):
        part_text, kind, dep_level, name, qname = part[0], part[1], part[2], part[3], part[4]
        dep_source = _part_dep_source(part)
        macro_metadata = _part_macro_provenance(part)
        part_len = len(part_text)
        if offset + part_len > text_len:
            break
        chunk_ranges[i] = (offset, offset + part_len)
        fragment_end = offset + part_len + (2 if i < len(parts_info) - 1 else 0)
        fragment_sources.append(
            (
                offset,
                fragment_end,
                _part_source_path(part, index, filepath or "assembled.cpp"),
                i + 1,
                (
                    require_project_identity(
                        dep_source.removeprefix("crosslib:"),
                        source=f"crosslib source for {name}",
                    )
                    if dep_source is not None and dep_source.startswith("crosslib:")
                    else stable_project_id
                ),
            )
        )

        # Fill structure_ids
        for j in range(offset, offset + part_len):
            structure_ids[j] = kind

        boundary = {
            'start': offset,
            'end': offset + part_len,
            'kind': kind,
            'dep_level': dep_level,
            'name': name,
        }
        symbol_metadata = _part_symbol_metadata(part)
        if symbol_metadata is not None:
            boundary['symbol_id'] = int(symbol_metadata['symbol_id'])
        # Tag cross-repo base-lib pulls with their provenance so the model knows
        # this chunk is a base-lib impl and which lib it came from.
        if dep_source is not None:
            boundary['dep_source'] = dep_source
        chunk_boundaries.append(boundary)

        part_symbol_key = _part_symbol_key(part)
        # Track identities for edge computation. qname only resolves when
        # unambiguous; ambiguous qnames intentionally do not create graph edges.
        if qname:
            resolved_type_key = part_symbol_key or index.resolve_type_key(qname)
            resolved_func_key = part_symbol_key or index.resolve_function_key(qname)
            if resolved_type_key:
                chunk_all_symbols[i] = resolved_type_key
            # Call-edge TARGET candidates = local function chunks AND cross-lib
            # base-lib pulls (a pulled def is a real callee target even though it
            # is not in the LOCAL index). This lets root->base-lib call_edges
            # resolve to the pulled chunk.
            if (resolved_func_key and resolved_func_key in index.functions) or (
                dep_source is not None and dep_source.startswith("crosslib:")
            ):
                target_key = part_symbol_key or resolved_func_key
                if target_key:
                    chunk_target_symbols[i] = target_key
        if qname:
            func_key = part_symbol_key or index.resolve_function_key(qname)
        else:
            func_key = None
        if func_key and func_key in index.functions:
            func = index.functions[func_key]
            chunk_symbols[i] = func_key
            callee_refs = index._function_callee_keys(func)
            for ref in getattr(func, "baselib_callee_refs", []):
                ref_key = _resolve_pulled_crosslink(ref)
                if isinstance(ref_key, str) and ref_key and ref_key not in callee_refs:
                    callee_refs.append(ref_key)
                    source_key = ref.get("symbol_key")
                    if isinstance(source_key, str) and source_key != ref_key:
                        call_target_id_rewrites[i][
                            _compute_symbol_id(source_key)
                        ] = _compute_symbol_id(ref_key)
            for ref in list(func.callees) + list(getattr(func, "baselib_callees", [])):
                edge_key = _resolve_pulled_crosslink(ref)
                if edge_key is not None and edge_key not in callee_refs:
                    callee_refs.append(edge_key)
            chunk_callees[i] = callee_refs
            chunk_types[i] = index._function_referenced_type_keys(func)
            macro_invocation_routes.extend(
                _macro_invocation_route_parts(
                    part_text,
                    offset=offset,
                    index=index,
                    target_file=func.file,
                    start_line=func.line,
                )
            )
        macro_route_part = _macro_route_part(
            part_text,
            offset=offset,
            metadata=macro_metadata,
        )
        if macro_route_part is not None:
            macro_route_parts.append(macro_route_part)

        offset += part_len
        if i < len(parts_info) - 1:
            offset += 2  # "\n\n"

    # Compute call_edges. Sources are local function chunks (they carry callees);
    # targets include local function chunks AND cross-lib base-lib pulls.
    call_edges = []
    for ci, caller_symbol in chunk_symbols.items():
        callees = chunk_callees.get(ci, [])
        for callee_symbol in callees:
            for cj, target_symbol in chunk_target_symbols.items():
                if ci != cj and target_symbol == callee_symbol:
                    call_edges.append({'from': ci, 'to': cj})

    # Compute type_edges: a function chunk referencing type T -> the chunk that
    # defines T (mirror of the call_edges loop, over referenced_types).
    type_edges: list[dict[str, object]] = []
    for ci, ref_types in chunk_types.items():
        for type_symbol in ref_types:
            for cj, symbol in chunk_all_symbols.items():
                if ci != cj and symbol == type_symbol:
                    type_edges.append({'from': ci, 'to': cj})

    ast_depth, sibling_index, ast_node_type = extract_ast_metadata_from_parts(
        full_text,
        parts_info,
        index,
    )

    # v4 enrichment: per-file platform detection
    import sys as _sys
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _v5_dir = os.path.join(_parent, 'v5_gke_orchestrator')
    if _v5_dir not in _sys.path:
        _sys.path.insert(0, _v5_dir)
    platform_info: dict[str, object] | None
    try:
        _platform_detect = importlib.import_module("platform_detect")
        _detect_plat = cast(PlatformDetectFn, getattr(_platform_detect, "detect_platforms"))
        platform_info = _detect_plat(full_text)
    except ImportError:
        platform_info = None

    # Extract semantic metadata (symbol IDs, call targets, type refs, def-use)
    semantic_meta = extract_semantic_metadata_from_parts(full_text, parts_info, index)
    for part_index, rewrites in call_target_id_rewrites.items():
        start, end = chunk_ranges[part_index]
        for char_index in range(start, end):
            call_id = semantic_meta['call_targets'][char_index]
            if call_id in rewrites:
                semantic_meta['call_targets'][char_index] = rewrites[call_id]
            symbol_id = semantic_meta['symbol_ids'][char_index]
            if symbol_id in rewrites:
                semantic_meta['symbol_ids'][char_index] = rewrites[symbol_id]

    symbol_identities = _document_symbol_identities(
        parts_info,
        index,
        semantic_meta['symbol_ids'],
        semantic_meta['call_targets'],
        semantic_meta['type_refs'],
        source=filepath or "clang enriched document",
    )

    result = {
        'text': full_text,
        'doc_type': 'code',
        'symbol_identity_schema_version': SYMBOL_IDENTITY_SCHEMA_VERSION,
        SYMBOL_IDENTITIES_COLUMN: symbol_identities,
        'filepath': filepath or '',
        'structure_ids': structure_ids,
        'chunk_boundaries': chunk_boundaries,
        'call_edges': call_edges,
        'type_edges': type_edges,
        'ast_depth': ast_depth,
        'sibling_index': sibling_index,
        'ast_node_type': ast_node_type,
        'symbol_ids': semantic_meta['symbol_ids'],
        'call_targets': semantic_meta['call_targets'],
        'type_refs': semantic_meta['type_refs'],
        'def_use': semantic_meta['def_use'],
        **_cpp_domain_sidecars(
            full_text,
            index,
            source_doc_id=1,
            source_path=filepath or "assembled.cpp",
            project_id=stable_project_id,
            fragment_sources=fragment_sources,
            macro_parts=macro_route_parts,
            macro_invocations=macro_invocation_routes,
        ),
    }
    if stable_project_id is not None:
        result.update(
            _project_path_provenance(
                stable_project_id,
                filepath or "assembled.cpp",
            )
        )
    language_info = detect_language_info(
        full_text,
        filepath,
        platform_info,
        compile_args=compile_args,
        build_info=build_info,
    )

    if platform_info:
        result['platform_info'] = platform_info
    if build_info:
        result['build_info'] = build_info
    if language_info:
        result['language_info'] = language_info
    return result


# --------------------------------------------------------------------------- #
# Build-file doc emission (ADDITIVE).
#
# A build-file chunk is emitted as a 'build' enriched doc with the SAME dict
# shape as build_enriched_doc, but: exact span text; structure_ids ALL set to
# BUILD_KIND=9
# (extends the 0-8 code-kind vocab); EMPTY call/type/symbol graph (correct --
# build files are not C++, like commit docs on the code channels); a doc_type of
# 'build'; and a language_info whose primary_language is the build-system tag
# (cmake/make/bazel/ninja/meson/...). platform_info / build_info carry the
# derived A-platform signal where available. NO libclang involved.
def _build_domain_sidecars(
    text: str,
    build_kind: str,
    *,
    filepath: str | None = None,
    source_doc_id: int,
    project_id: str | None = None,
) -> dict[str, object]:
    """Parse build-system text into char-aligned domain sidecars.

    The Clang path emits C++ call/type graph metadata.  Build systems need a
    different graph: target->source, target->dependency, rule->command, and
    shell-like command/file relations.  This helper keeps those routes as
    char-origin edge triples so the token materializer can map them to LLM token
    positions after tokenizer normalization.
    """

    from cppmega.data.build_parsers.base import ParsedDomainDocument
    from cppmega.data.domain_ingestion import (
        parse_domain_document,
        resolve_domain_parser,
    )
    from cppmega.data.domain_schema import (
        DomainKind,
        ParseConfidence,
        domain_edge_family,
    )

    kind = build_kind.lower().replace("-", "_")
    parser_path_by_kind = {
        "cmake": "CMakeLists.txt",
        "cmakelists": "CMakeLists.txt",
        "make": "Makefile",
        "gmake": "Makefile",
        "makefile": "Makefile",
        "automake": "Makefile.am",
        "makefile_am": "Makefile.am",
        "autoconf": "configure.ac",
        "configure_ac": "configure.ac",
        "configure_in": "configure.in",
        "configure": "configure",
        "ninja": "build.ninja",
        "bazel": "BUILD.bazel",
        "build_bazel": "BUILD.bazel",
        "workspace_bazel": "WORKSPACE.bazel",
        "meson": "meson.build",
        "gn": "BUILD.gn",
        "scons": "SConstruct",
        "xmake": "xmake.lua",
        "dockerfile": "Dockerfile",
        "bash": "script.bash",
        "sh": "script.sh",
        "zsh": "script.zsh",
        "tcsh": "script.tcsh",
        "ksh": "script.ksh",
        "powershell": "script.ps1",
        "pwsh": "script.ps1",
        "ps1": "script.ps1",
        "batch": "script.cmd",
        "cmd": "script.cmd",
        "sql": "schema.sql",
        "python": "module.py",
        "assembly": "source.s",
    }
    parser_path = parser_path_by_kind.get(kind)
    resolved_path = filepath or parser_path
    resolved_adapter = (
        resolve_domain_parser(resolved_path, text) if resolved_path is not None else None
    )
    raw_shared_configure_kinds = {
        "conan",
        "vcpkg",
        "msvc",
    }
    if parser_path is not None:
        parsed = parse_domain_document(
            parser_path,
            text,
            source_doc_id=source_doc_id,
        )
        if resolved_path is not None and resolved_path != parser_path:
            parsed.metadata["source_path"] = resolved_path
            parsed.set_source_identity(
                source_identity_for_path(resolved_path, text=text)
            )
    elif kind in raw_shared_configure_kinds:
        parsed = ParsedDomainDocument.new(
            domain=DomainKind.CONFIGURE,
            text=text,
            confidence=ParseConfidence.RAW,
            metadata={
                "parser_adapter": f"{kind}-raw",
                "build_dialect": build_kind,
                "shared_domain": "configure",
                "unsupported_syntax": f"{kind}_native_parser_unavailable",
                "raw_reason": f"{kind}_native_parser_unavailable",
            },
        )
        parsed.set_source_doc_id(source_doc_id)
        parsed.set_source_identity(
            source_identity_for_path(
                resolved_path or parser_path or build_kind,
                text=text,
            )
        )
    elif resolved_adapter is not None and resolved_adapter.name != "raw-output":
        assert resolved_path is not None
        parsed = parse_domain_document(
            resolved_path,
            text,
            source_doc_id=source_doc_id,
        )
    else:
        raw_domain_by_kind = {
            "meson": DomainKind.MESON,
            "gn": DomainKind.GN,
            "scons": DomainKind.SCONS,
            "xmake": DomainKind.XMAKE,
            "compile_commands": DomainKind.COMPILE_COMMANDS,
        }
        domain = raw_domain_by_kind.get(kind, DomainKind.CONFIGURE)
        parsed = ParsedDomainDocument.new(
            domain=domain,
            text=text,
            confidence=ParseConfidence.RAW,
            metadata={
                "parser_adapter": "raw-build",
                "build_kind": build_kind,
                "shared_domain": (
                    "configure" if domain == DomainKind.CONFIGURE else None
                ),
                "unsupported_syntax": f"unsupported_build_domain:{build_kind}",
                "raw_reason": f"unsupported_build_domain:{build_kind}",
            },
        )
        parsed.set_source_doc_id(source_doc_id)
        parsed.set_source_identity(
            source_identity_for_path(resolved_path or parser_path or build_kind, text=text)
        )

    enriched = parsed.to_enriched_document()
    enriched["domain_edges"] = [
        edge
        for edge in cast(list[dict[str, int]], enriched["domain_edges"])
        if domain_edge_family(int(edge["kind"])) == "domain"
    ]
    if project_id is not None and resolved_path is not None:
        identity = _source_identity_for_project_path(
            resolved_path,
            project_id=project_id,
        )
        enriched["domain_source_identity_ids"] = [
            identity.source_identity_id
        ] * len(text)
        enriched["source_identity_registry"] = [identity.as_dict()]
    enriched["domain_parse_info"].update(
        {
            "parser": parsed.metadata.get("parser_adapter", "raw-build"),
            "build_kind": build_kind,
        }
    )
    enriched.pop("text", None)
    return enriched


def build_build_doc(
    filepath: str,
    text: str,
    build_kind: str,
    *,
    source_root: str | None = None,
    project_id: str | None = None,
    platform_info: dict | None = None,
    build_info: dict | None = None,
    source_span: dict[str, int | str] | None = None,
) -> dict:
    """Build a single 'build' enriched doc from a build/compilation file.

    Callers pass already-read text. Zero-length inputs are skipped by the
    emitter; non-empty whitespace remains source content and is represented.
    """
    emitted_filepath = filepath
    if source_root is not None:
        absolute_root = os.path.abspath(source_root)
        absolute_file = os.path.abspath(filepath)
        try:
            if os.path.commonpath((absolute_root, absolute_file)) == absolute_root:
                emitted_filepath = os.path.relpath(absolute_file, absolute_root)
        except ValueError:
            pass
    emitted_filepath = _canonical_project_path(emitted_filepath)
    stable_project_id = (
        require_project_identity(project_id, source="build_build_doc")
        if project_id is not None
        else None
    )
    text_len = len(text)
    source_doc_id = 1
    domain_sidecars = _build_domain_sidecars(
        text,
        build_kind,
        filepath=emitted_filepath,
        source_doc_id=source_doc_id,
        project_id=stable_project_id,
    )
    if source_span is not None:
        cast(dict[str, object], domain_sidecars["domain_parse_info"])[
            "source_span"
        ] = dict(source_span)
    structure_ids = [BUILD_KIND] * text_len
    chunk_boundaries = [{
        'start': 0,
        'end': text_len,
        'kind': BUILD_KIND,
        'dep_level': 0,
        'name': os.path.basename(emitted_filepath),
    }]

    # Per-file platform detection from the build text itself (additive to the
    # repo-level build_info threaded in by process_project).
    detected_platform: dict[str, object] | None = platform_info
    if detected_platform is None:
        import sys as _sys
        _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _v5_dir = os.path.join(_parent, 'v5_gke_orchestrator')
        if _v5_dir not in _sys.path:
            _sys.path.insert(0, _v5_dir)
        try:
            _platform_detect = importlib.import_module("platform_detect")
            _detect_plat = cast(PlatformDetectFn, getattr(_platform_detect, "detect_platforms"))
            detected_platform = _detect_plat(text)
        except ImportError:
            detected_platform = None

    # Language tag: the build-system family IS the primary_language so the model
    # knows which build system this is. detect_language_info has no build-system
    # output key, so we set primary_language directly (additive, rendered for
    # free by language_info_to_prefix as "// language: primary=<build_kind> ...").
    from cppmega.data.domain_schema import DomainKind

    domain = DomainKind(int(cast(int, domain_sidecars["domain_kind"])))
    is_shell_doc = domain in {
        DomainKind.BASH,
        DomainKind.SH,
        DomainKind.ZSH,
        DomainKind.TCSH,
        DomainKind.KSH,
    }
    if is_shell_doc:
        shell_kind_for_edges = build_kind.lower().replace("-", "_")
        # CMD/batch shares the frozen SH domain ID, but its escaping and
        # comments are not POSIX-compatible. Keep native parser edges when a
        # grammar exists; never manufacture POSIX edges for a RAW CMD document.
        extracted_shell_edges = (
            []
            if shell_kind_for_edges in {"cmd", "batch"}
            else _shell_edges_from_script(text, shell_kind_for_edges)
        )
        if extracted_shell_edges:
            shell_edges = cast(list[dict[str, int]], domain_sidecars["shell_edges"])
            seen_shell_edges = {
                (
                    int(edge["from_char"]),
                    int(edge["to_char"]),
                    int(edge["kind"]),
                )
                for edge in shell_edges
            }
            for edge in extracted_shell_edges:
                triple = (
                    int(edge["from_char"]),
                    int(edge["to_char"]),
                    int(edge["kind"]),
                )
                if triple not in seen_shell_edges:
                    shell_edges.append(edge)
                    seen_shell_edges.add(triple)
    is_code_doc = domain in {DomainKind.CPP, DomainKind.PYTHON}
    is_assembly_doc = build_kind == "assembly"
    is_sql_doc = domain == DomainKind.SQL
    is_diagnostic_doc = int(domain) >= int(DomainKind.COMPILER_DIAGNOSTIC)
    doc_type = (
        "shell"
        if is_shell_doc
        else "code"
        if is_code_doc
        else "diagnostic"
        if is_diagnostic_doc
        else "sql"
        if is_sql_doc
        else "build"
    )
    detected_sql_language = (
        detect_language_info(
            text,
            emitted_filepath,
            detected_platform,
            build_info=build_info,
        )
        if is_sql_doc
        else None
    )
    sql_dialect = (
        str(detected_sql_language["primary_dialect"])
        if detected_sql_language
        and detected_sql_language.get("primary_dialect")
        else None
    )
    resolved_build_kind = (
        f"sql:{sql_dialect}" if is_sql_doc and sql_dialect else build_kind
    )
    if sql_dialect:
        cast(dict[str, object], domain_sidecars["domain_parse_info"])[
            "sql_dialect"
        ] = sql_dialect
    language_info = {
        "primary_language": build_kind,
        "primary_standard": (
            None
            if is_shell_doc or is_sql_doc or is_assembly_doc
            else (build_info or {}).get("standard")
        ),
        "primary_dialect": (
            build_kind if is_shell_doc else sql_dialect if is_sql_doc else None
        ),
        "embedded_languages": [],
        "signals": sorted(set([
            f"shell_file:{build_kind}"
            if is_shell_doc
            else f"code_file:{build_kind}"
            if is_code_doc
            else f"sql_file:{build_kind}"
            if is_sql_doc
            else f"build_file:{build_kind}"
        ] + (
            list(detected_sql_language.get("signals", []))
            if detected_sql_language
            else []
        ))),
        "detector_sources": sorted(set([
            "shell_file"
            if is_shell_doc
            else "code_file"
            if is_code_doc
            else "sql_file"
            if is_sql_doc
            else "build_file"
        ] + (
            list(detected_sql_language.get("detector_sources", []))
            if detected_sql_language
            else []
        ))),
        "confidence": (
            detected_sql_language.get("confidence", "high")
            if detected_sql_language
            else "high"
        ),
    }

    result: dict[str, object] = {
        'text': text,
        'doc_type': doc_type,
        'symbol_identity_schema_version': SYMBOL_IDENTITY_SCHEMA_VERSION,
        SYMBOL_IDENTITIES_COLUMN: [],
        'source_identity_id': int(
            cast(
                int,
                cast(
                    list[dict[str, object]],
                    domain_sidecars["source_identity_registry"],
                )[0]["source_identity_id"],
            )
        ),
        'build_kind': resolved_build_kind,
        'filepath': emitted_filepath,
        'structure_ids': structure_ids,
        'chunk_boundaries': chunk_boundaries,
        # Build files carry NO call/type/symbol graph (not C++) -- empty/0, like
        # commit docs on the code channels. This is correct, not a fallback.
        'call_edges': [],
        'type_edges': [],
        'ast_depth': [],
        'sibling_index': [],
        'ast_node_type': [],
        'symbol_ids': [],
        'call_targets': [],
        'type_refs': [],
        'def_use': [],
        'language_info': language_info,
        **domain_sidecars,
    }
    if stable_project_id is not None:
        result.update(
            _project_path_provenance(stable_project_id, emitted_filepath)
        )
    if detected_platform:
        result['platform_info'] = detected_platform
    if build_info:
        result['build_info'] = build_info
    if source_span is not None:
        result['source_span'] = dict(source_span)
    return result


_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)(?:\b|(?=\())")
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+["<]([^">]+)[">]')
_UNDEF_RE = re.compile(r"^\s*#\s*undef\s+([A-Za-z_][A-Za-z0-9_]*)")
_IF_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif)\b(.*)")
_ELSE_RE = re.compile(r"^\s*#\s*else\b")
_ENDIF_RE = re.compile(r"^\s*#\s*endif\b")
DEFAULT_MACRO_INCLUDE_DEPTH = int(os.environ.get("CPPMEGA_MACRO_INCLUDE_DEPTH", "0"))
DEFAULT_MACRO_INCLUDE_FILES_PER_ROOT = int(
    os.environ.get("CPPMEGA_MACRO_INCLUDE_FILES_PER_ROOT", "0")
)
DEFAULT_MACRO_DIRECTIVE_CACHE_ENTRIES = int(
    os.environ.get("CPPMEGA_MACRO_DIRECTIVE_CACHE_ENTRIES", "4096")
)
DEFAULT_MACRO_RESOLVE_CACHE_ENTRIES = int(
    os.environ.get("CPPMEGA_MACRO_RESOLVE_CACHE_ENTRIES", "65536")
)
DEFAULT_MAX_MACRO_CANDIDATES_PER_ROOT = int(
    os.environ.get("CPPMEGA_MAX_MACRO_CANDIDATES_PER_ROOT", "200000")
)
DEFAULT_MAX_RETAINED_MACROS = int(
    os.environ.get("CPPMEGA_MAX_RETAINED_MACROS", "250000")
)
_CONDITION_SKIP_IDENTIFIERS = {
    "defined",
    "and",
    "or",
    "not",
    "true",
    "false",
}
_INCLUDE_GUARD_SUFFIXES = (
    "_H",
    "_H_",
    "_HH",
    "_HH_",
    "_HPP",
    "_HPP_",
    "_HXX",
    "_HXX_",
    "_H_INCLUDED",
    "_HH_INCLUDED",
    "_HPP_INCLUDED",
    "_INCLUDED",
    "_INCLUDED_",
    "_INCLUDE_GUARD",
    "_INCLUDE_GUARD_",
    "_GUARD",
    "_GUARD_",
)


def _is_probable_include_guard_define(macro_text: str, name: str) -> bool:
    lines = macro_text.splitlines()
    if len(lines) != 1:
        return False
    match = _DEFINE_RE.match(lines[0])
    if match is None:
        return False
    rest = lines[0][match.end():].strip()
    if rest not in {"", "1"}:
        return False
    upper = name.upper()
    return upper.endswith(_INCLUDE_GUARD_SUFFIXES)


def extract_macro_blocks(text: str) -> list[tuple[int, int, str, str]]:
    """Return trainable ``#define`` logical lines, preserving continuations."""
    lines, code_lines = _split_source_and_masked_lines(text)
    blocks: list[tuple[int, int, str, str]] = []
    offset = 0
    line_offsets: list[int] = []
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    i = 0
    while i < len(lines):
        line = lines[i]
        match = _DEFINE_RE.match(code_lines[i])
        if match is None:
            i += 1
            continue
        name = match.group(1)
        start = line_offsets[i]
        end_line = i
        while lines[end_line].rstrip("\r\n").endswith("\\") and end_line + 1 < len(lines):
            end_line += 1
        end = line_offsets[end_line] + len(lines[end_line])
        macro_text = text[start:end]
        if (
            len(macro_text.strip()) >= 8
            and not _is_probable_include_guard_define(macro_text, name)
        ):
            blocks.append((start, end, name, macro_text))
        i = end_line + 1
    return blocks


def _condition_macro_names(line: str) -> list[str]:
    match = _IF_RE.match(line)
    if match is None:
        return []
    directive = match.group(1)
    rest = match.group(2)
    if directive in {"ifdef", "ifndef"}:
        ident = next(_iter_code_identifiers(rest), None)
        return [ident.group(0)] if ident else []
    names: list[str] = []
    for ident in _iter_code_identifiers(rest):
        name = ident.group(0)
        if name.lower() in _CONDITION_SKIP_IDENTIFIERS:
            continue
        names.append(name)
    return names


def _import_dedup_store_symbols():
    try:
        from dedup_store import DedupStore, _sha1_tokens  # type: ignore
    except ModuleNotFoundError:
        from tools.clang_indexer.dedup_store import DedupStore, _sha1_tokens
    return DedupStore, _sha1_tokens


def _resolve_local_include(
    include_name: str,
    *,
    current_abs: str,
    project_dir: str | None,
    include_dirs: list[str] | None = None,
    known_local_files: set[str] | None = None,
) -> str | None:
    candidates = [os.path.join(os.path.dirname(current_abs), include_name)]
    for include_dir in include_dirs or []:
        candidates.append(os.path.join(include_dir, include_name))
    if project_dir is not None:
        candidates.append(os.path.join(project_dir, include_name))
        candidates.append(os.path.join(project_dir, "include", include_name))
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if known_local_files is not None:
            if norm in known_local_files:
                return norm
            continue
        if os.path.isfile(norm):
            return norm
    return None


def _build_macro_local_file_index(project_dir: str | None) -> set[str] | None:
    """Index project-local files once for macro include resolution.

    Macro scanning runs per source root. Calling os.path.isfile for every
    include candidate across every root explodes on large projects with many
    include dirs (ITK hit a post-parse stat loop). The macro scanner only needs
    project-local headers/sources; system headers are intentionally not scanned
    into training docs.
    """
    if project_dir is None:
        return None
    root = os.path.normpath(os.path.abspath(project_dir))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"macro include index project_dir is not a directory: {root}")
    indexed: set[str] = set()
    skip_dirs = {
        ".git",
        ".svn",
        "node_modules",
        "build",
        "_build",
        "cmake-build-debug",
    }
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in skip_dirs]
        for filename in filenames:
            indexed.add(os.path.normpath(os.path.join(dirpath, filename)))
    return indexed


def _include_dirs_from_args(
    args: list[str] | None,
    *,
    project_dir: str | None,
) -> list[str]:
    include_dirs: list[str] = []
    if not args:
        return include_dirs
    i = 0
    while i < len(args):
        arg = args[i]
        value: str | None = None
        if arg in {"-I", "-iquote", "-isystem", "-idirafter"} and i + 1 < len(args):
            value = args[i + 1]
            i += 2
        elif arg.startswith("-I") and len(arg) > 2:
            value = arg[2:]
            i += 1
        elif arg.startswith("-iquote") and len(arg) > len("-iquote"):
            value = arg[len("-iquote"):]
            i += 1
        elif arg.startswith("-isystem") and len(arg) > len("-isystem"):
            value = arg[len("-isystem"):]
            i += 1
        else:
            i += 1
        if not value:
            continue
        if project_dir is not None and not os.path.isabs(value):
            value = os.path.join(project_dir, value)
        norm = os.path.normpath(value)
        if norm not in include_dirs:
            include_dirs.append(norm)
    return include_dirs


def _collect_macro_include_dirs(
    *,
    project_dir: str | None,
    compile_db: dict | None,
    default_args: list[str] | None,
) -> list[str]:
    dirs: list[str] = []

    def add_many(values: list[str]) -> None:
        for value in values:
            if value not in dirs:
                dirs.append(value)

    add_many(_include_dirs_from_args(default_args, project_dir=project_dir))
    if compile_db:
        for entry in compile_db.values():
            if not isinstance(entry, dict):
                continue
            add_many(
                _include_dirs_from_args(
                    entry.get("compile_args", []),
                    project_dir=project_dir,
                )
            )
    return dirs


def register_header_macros(
    index: ProjectIndex,
    header_files: list[str],
    *,
    project_dir: str | None,
    project_id: str,
    include_dirs: list[str] | None = None,
    max_include_depth: int | None = None,
    max_include_files_per_root: int | None = None,
    memory_limit_gb: float | None = None,
    macro_usage_texts_by_file: dict[
        str,
        Sequence[str | tuple[str, int | None]],
    ] | None = None,
    directive_cache_entries: int | None = None,
    resolve_cache_entries: int | None = None,
    max_macro_candidates_per_root: int | None = None,
    max_retained_macros: int | None = None,
    max_macro_visibility_bytes: int | None = None,
) -> dict[str, int]:
    stable_project_id = require_project_identity(
        project_id,
        source="register_header_macros",
    )
    sequence = [0]
    directive_cache: dict[str, list[dict[str, object]]] = {}
    resolve_cache: dict[tuple[str, str], str | None] = {}
    stats: defaultdict[str, int] = defaultdict(int)
    local_file_index = _build_macro_local_file_index(project_dir)
    if local_file_index is not None:
        stats["local_file_index_entries"] = len(local_file_index)
    max_include_depth = (
        DEFAULT_MACRO_INCLUDE_DEPTH
        if max_include_depth is None
        else int(max_include_depth)
    )
    max_include_files_per_root = (
        DEFAULT_MACRO_INCLUDE_FILES_PER_ROOT
        if max_include_files_per_root is None
        else int(max_include_files_per_root)
    )
    directive_cache_entries = (
        DEFAULT_MACRO_DIRECTIVE_CACHE_ENTRIES
        if directive_cache_entries is None
        else int(directive_cache_entries)
    )
    resolve_cache_entries = (
        DEFAULT_MACRO_RESOLVE_CACHE_ENTRIES
        if resolve_cache_entries is None
        else int(resolve_cache_entries)
    )
    max_macro_candidates_per_root = (
        DEFAULT_MAX_MACRO_CANDIDATES_PER_ROOT
        if max_macro_candidates_per_root is None
        else int(max_macro_candidates_per_root)
    )
    max_retained_macros = (
        DEFAULT_MAX_RETAINED_MACROS
        if max_retained_macros is None
        else int(max_retained_macros)
    )
    if max_macro_visibility_bytes is None:
        configured_visibility_bytes = os.environ.get(
            "CPPMEGA_MAX_MACRO_VISIBILITY_BYTES"
        )
        max_macro_visibility_bytes = (
            int(configured_visibility_bytes)
            if configured_visibility_bytes is not None
            else (
                max_retained_macros
                * LEGACY_MACRO_RETAINED_SHALLOW_BYTES
            )
        )
    else:
        max_macro_visibility_bytes = int(max_macro_visibility_bytes)
    if max_include_depth < 0:
        raise ValueError(f"max_include_depth must be >= 0, got {max_include_depth}")
    if max_include_files_per_root < 0:
        raise ValueError(
            f"max_include_files_per_root must be >= 0, got {max_include_files_per_root}"
        )
    if directive_cache_entries < 0:
        raise ValueError(
            f"directive_cache_entries must be >= 0, got {directive_cache_entries}"
        )
    if resolve_cache_entries < 0:
        raise ValueError(
            f"resolve_cache_entries must be >= 0, got {resolve_cache_entries}"
        )
    if max_macro_candidates_per_root <= 0:
        raise ValueError(
            "max_macro_candidates_per_root must be > 0, got "
            f"{max_macro_candidates_per_root}"
        )
    if max_retained_macros <= 0:
        raise ValueError(
            f"max_retained_macros must be > 0, got {max_retained_macros}"
        )
    if max_macro_visibility_bytes <= 0:
        raise ValueError(
            "max_macro_visibility_bytes must be > 0, got "
            f"{max_macro_visibility_bytes}"
        )
    stats["macro_visibility_byte_limit"] = max_macro_visibility_bytes

    def _check_memory() -> None:
        if memory_limit_gb is not None and memory_limit_gb > 0:
            check_memory_limit(memory_limit_gb, label="index_project macro scan")

    def _rel(path: str) -> str:
        return (
            os.path.relpath(path, project_dir)
            if project_dir is not None and os.path.isabs(path)
            else path
        )

    derived_usage_texts: defaultdict[
        str,
        list[str | tuple[str, int | None]],
    ] = defaultdict(list)
    for function in index.functions.values():
        if function.text:
            derived_usage_texts[os.path.normpath(function.file)].append(
                (function.text, function.line)
            )
    for type_def in index.typedefs.values():
        if type_def.file and type_def.text:
            derived_usage_texts[os.path.normpath(type_def.file)].append(
                (type_def.text, type_def.line)
            )
    for file, preamble in index.file_preambles.items():
        if preamble:
            derived_usage_texts[os.path.normpath(file)].append((preamble, 1))

    explicit_usage_texts = {
        os.path.normpath(file): list(texts)
        for file, texts in (macro_usage_texts_by_file or {}).items()
    }

    def _usage_texts_for_root(
        root_abs: str,
        root_rel: str,
    ) -> list[str | tuple[str, int | None]]:
        normalized_rel = os.path.normpath(root_rel)
        if normalized_rel in explicit_usage_texts:
            return explicit_usage_texts[normalized_rel]
        derived = derived_usage_texts.get(normalized_rel)
        if derived:
            return list(derived)
        try:
            root_text, _root_bytes, _root_encoding = _read_source_file(root_abs)
            return [(root_text, 1)]
        except OSError as exc:
            raise RuntimeError(
                f"failed to read C/C++ macro root for usage scan: {root_abs}"
            ) from exc

    def _directive_events(norm_abs: str) -> list[dict[str, object]]:
        cached = directive_cache.pop(norm_abs, None)
        if cached is not None:
            directive_cache[norm_abs] = cached
            stats["directive_cache_hits"] += 1
            return cached
        try:
            file_text, _file_bytes, _file_encoding = _read_source_file(norm_abs)
        except OSError as exc:
            raise RuntimeError(f"failed to read C/C++ file for macro scan: {norm_abs}") from exc

        lines, code_lines = _split_source_and_masked_lines(file_text)
        stats["directive_file_reads"] += 1
        events: list[dict[str, object]] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            code_line = code_lines[i]
            line_no = i + 1

            include_match = _INCLUDE_RE.match(line)
            if include_match is not None and re.match(
                r"^\s*#\s*include\b",
                code_line,
            ):
                events.append(
                    {
                        "kind": "include",
                        "line_no": line_no,
                        "include_name": include_match.group(1),
                    }
                )
                i += 1
                continue

            if _ENDIF_RE.match(code_line):
                events.append({"kind": "endif"})
                i += 1
                continue
            if _ELSE_RE.match(code_line):
                events.append({"kind": "else", "line": line, "code_line": code_line})
                i += 1
                continue
            if _IF_RE.match(code_line) is not None:
                events.append({"kind": "if", "line": line, "code_line": code_line})
                i += 1
                continue

            undef_match = _UNDEF_RE.match(code_line)
            if undef_match is not None:
                events.append(
                    {
                        "kind": "undef",
                        "line": line,
                        "name": undef_match.group(1),
                    }
                )
                i += 1
                continue

            define_match = _DEFINE_RE.match(code_line)
            if define_match is None:
                i += 1
                continue
            name = define_match.group(1)
            start_line = i
            end_line = i
            while lines[end_line].rstrip("\r\n").endswith("\\") and end_line + 1 < len(lines):
                end_line += 1
            macro_text = "".join(lines[start_line:end_line + 1])
            if len(macro_text.strip()) >= 8:
                parsed = _parse_macro_signature(macro_text)
                params = list(parsed[2]) if parsed is not None else []
                events.append(
                    {
                        "kind": "define",
                        "line_no": line_no,
                        "name": name,
                        "macro_text": macro_text,
                        "params": params,
                        "include_guard": _is_probable_include_guard_define(
                            macro_text,
                            name,
                        ),
                    }
                )
            i = end_line + 1
        if directive_cache_entries:
            directive_cache[norm_abs] = events
            while len(directive_cache) > directive_cache_entries:
                directive_cache.pop(next(iter(directive_cache)))
                stats["directive_cache_evictions"] += 1
            stats["directive_cache_peak_entries"] = max(
                stats["directive_cache_peak_entries"],
                len(directive_cache),
            )
        return events

    def _resolve_include_cached(include_name: str, *, current_abs: str) -> str | None:
        key = (include_name, os.path.dirname(current_abs))
        if key in resolve_cache:
            resolved = resolve_cache.pop(key)
            resolve_cache[key] = resolved
            stats["include_resolve_cache_hits"] += 1
            return resolved
        resolved = _resolve_local_include(
            include_name,
            current_abs=current_abs,
            project_dir=project_dir,
            include_dirs=include_dirs,
            known_local_files=local_file_index,
        )
        if resolve_cache_entries:
            resolve_cache[key] = resolved
            while len(resolve_cache) > resolve_cache_entries:
                resolve_cache.pop(next(iter(resolve_cache)))
                stats["resolve_cache_evictions"] += 1
            stats["resolve_cache_peak_entries"] = max(
                stats["resolve_cache_peak_entries"],
                len(resolve_cache),
            )
        return resolved

    def _scan_root(root_abs: str, root_rel: str) -> None:
        stats["roots"] += 1
        if stats["roots"] == 1 or stats["roots"] % 250 == 0:
            _check_memory()
            print(
                "  Macro scan heartbeat: "
                f"roots={stats['roots']} scanned_files={stats['scanned_files']} "
                f"discovered={stats['discovered_macro_occurrences']} "
                f"retained_views={stats['retained_macro_occurrences']} "
                f"canonical={stats['canonical_macro_definitions']} "
                f"visibility_bytes={index.macro_visibility_bytes} "
                f"resolve_cache_hits={stats['include_resolve_cache_hits']}",
                file=sys.stderr,
                flush=True,
            )
        root_index = ProjectIndex()
        root_macro_objects = 0
        root_discovered_macros = 0
        visited_files: set[str] = set()
        active_defs: dict[str, MacroDef] = {}
        active_macro_texts: dict[str, str] = {}
        last_defs: dict[str, MacroDef] = {}
        pending_undef: dict[str, str] = {}
        condition_stack: list[dict[str, object]] = []
        usage_names = {
            match.group(0)
            for item in _usage_texts_for_root(root_abs, root_rel)
            for match in _iter_code_identifiers(
                str(item[0]) if isinstance(item, tuple) else str(item)
            )
        }
        relevant_names = set(usage_names)

        def _count_macro_object() -> None:
            nonlocal root_macro_objects
            root_macro_objects += 1
            stats["peak_root_macro_candidates"] = max(
                stats["peak_root_macro_candidates"],
                root_macro_objects,
            )
            if root_macro_objects > max_macro_candidates_per_root:
                raise MemoryError(
                    "macro scan exceeded the per-root candidate bound: "
                    f"root={root_rel} candidates={root_macro_objects} "
                    f"limit={max_macro_candidates_per_root}. "
                    "Raise CPPMEGA_MAX_MACRO_CANDIDATES_PER_ROOT only after "
                    "profiling this include graph."
                )

        def _conditions_active() -> bool:
            return all(bool(item["active"]) for item in condition_stack)

        def _condition_lines(item: dict[str, object]) -> list[str]:
            lines = item.get("lines")
            if isinstance(lines, list):
                return [str(line) for line in lines]
            return [str(item.get("line", ""))]

        def _macro_value(name: str) -> int:
            macro_text = active_macro_texts.get(name)
            if macro_text is None:
                return 0
            parsed = _parse_macro_signature(macro_text)
            if parsed is None:
                return 1
            body = macro_text[parsed[3]:].strip()
            if body in {"", "true", "TRUE"}:
                return 1
            if body in {"false", "FALSE"}:
                return 0
            body = body.strip("() \t")
            try:
                return int(body, 0)
            except ValueError:
                return 1

        def _macro_truthy(name: str) -> bool:
            return _macro_value(name) != 0

        def _eval_expr(rest: str) -> bool:
            def repl_defined(match: re.Match[str]) -> str:
                name = match.group(1) or match.group(2)
                return "1" if name in active_macro_texts else "0"

            expr = re.sub(
                r"\bdefined\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
                r"|\bdefined\s+([A-Za-z_][A-Za-z0-9_]*)",
                repl_defined,
                rest,
            )

            def repl_ident(match: re.Match[str]) -> str:
                name = match.group(0)
                low = name.lower()
                if low == "true":
                    return "1"
                if low == "false":
                    return "0"
                if low in _CONDITION_SKIP_IDENTIFIERS:
                    return name
                return str(_macro_value(name))

            expr = _IDENTIFIER_RE.sub(repl_ident, expr)
            expr = expr.replace("&&", " and ").replace("||", " or ")
            expr = re.sub(r"(?<![=!<>])!(?!=)", " not ", expr)
            expr = expr.strip()
            if not expr:
                return True
            safe_expr = re.sub(r"\b(?:and|or|not)\b", "", expr)
            if not re.fullmatch(r"[0-9A-Fa-fxXbBoO_() \t.+*/%<>=!&|^~ -]+", safe_expr):
                return any(_macro_truthy(name) for name in _condition_macro_names("#if " + rest))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", SyntaxWarning)
                    return bool(eval(expr, {"__builtins__": {}}, {}))
            except (Exception, SyntaxWarning):
                return any(_macro_truthy(name) for name in _condition_macro_names("#if " + rest))

        def _evaluate_condition(line: str) -> tuple[bool, list[str]]:
            match = _IF_RE.match(line)
            if match is None:
                return True, []
            directive = match.group(1)
            rest = match.group(2)
            names = _condition_macro_names(line)
            if directive == "ifdef":
                return (names[0] in active_macro_texts if names else False), names
            if directive == "ifndef":
                return (names[0] not in active_macro_texts if names else True), names
            stripped = rest.strip()
            if stripped in {"0", "(0)"}:
                return False, names
            if stripped in {"1", "(1)"}:
                return True, names
            return _eval_expr(rest), names

        def _expand_relevant_names() -> None:
            """Resolve replacement/condition dependencies without MacroDef objects."""

            while True:
                before = len(relevant_names)
                discovery_visited: set[str] = set()
                discovery_conditions: list[list[str]] = []

                def walk(
                    abs_path: str,
                    *,
                    include_stack: set[str],
                    depth: int,
                ) -> None:
                    norm_abs = os.path.normpath(abs_path)
                    if norm_abs in include_stack or norm_abs in discovery_visited:
                        return
                    if (
                        max_include_files_per_root
                        and len(discovery_visited) >= max_include_files_per_root
                    ):
                        return
                    discovery_visited.add(norm_abs)
                    include_stack.add(norm_abs)
                    try:
                        for event in _directive_events(norm_abs):
                            kind = str(event["kind"])
                            if kind == "include":
                                if max_include_depth and depth >= max_include_depth:
                                    continue
                                included = _resolve_include_cached(
                                    str(event["include_name"]),
                                    current_abs=norm_abs,
                                )
                                if included is not None:
                                    walk(
                                        included,
                                        include_stack=include_stack,
                                        depth=depth + 1,
                                    )
                                continue
                            if kind == "endif":
                                if discovery_conditions:
                                    discovery_conditions.pop()
                                continue
                            if kind == "else":
                                continue
                            if kind == "if":
                                code_line = str(event["code_line"])
                                match = _IF_RE.match(code_line)
                                names = _condition_macro_names(code_line)
                                if (
                                    match is not None
                                    and match.group(1) == "elif"
                                    and discovery_conditions
                                ):
                                    discovery_conditions[-1] = names
                                else:
                                    discovery_conditions.append(names)
                                continue
                            if kind != "define" or bool(event["include_guard"]):
                                continue
                            name = str(event["name"])
                            if name not in relevant_names:
                                continue
                            relevant_names.update(
                                _macro_body_dependency_names(
                                    str(event["macro_text"]),
                                    macro_name=name,
                                    params=cast(list[str], event["params"]),
                                )
                            )
                            for condition_names in discovery_conditions:
                                relevant_names.update(condition_names)
                    finally:
                        include_stack.remove(norm_abs)

                walk(root_abs, include_stack=set(), depth=0)
                if len(relevant_names) == before:
                    break

            stats["peak_root_relevant_macro_names"] = max(
                stats["peak_root_relevant_macro_names"],
                len(relevant_names),
            )

        _expand_relevant_names()

        def _scan_file(
            abs_path: str,
            rel_path: str,
            *,
            root_visible_line: int | None,
            include_stack: set[str],
            depth: int,
        ) -> None:
            nonlocal root_discovered_macros
            norm_abs = os.path.normpath(abs_path)
            if norm_abs in include_stack:
                stats["skipped_include_cycle"] += 1
                return
            if norm_abs in visited_files:
                stats["skipped_include_revisit"] += 1
                return
            if max_include_files_per_root and len(visited_files) >= max_include_files_per_root:
                stats["skipped_include_file_cap"] += 1
                return
            visited_files.add(norm_abs)
            stats["scanned_files"] += 1
            if stats["scanned_files"] % 500 == 0:
                _check_memory()
            include_stack.add(norm_abs)
            try:
                for event in _directive_events(norm_abs):
                    kind = str(event["kind"])
                    if kind == "include":
                        stats["include_directives"] += 1
                        if max_include_depth and depth >= max_include_depth:
                            stats["skipped_include_depth"] += 1
                            continue
                        included = _resolve_include_cached(
                            str(event["include_name"]),
                            current_abs=norm_abs,
                        )
                        if included is not None:
                            _scan_file(
                                included,
                                _rel(included),
                                root_visible_line=root_visible_line
                                or int(event["line_no"]),
                                include_stack=include_stack,
                                depth=depth + 1,
                            )
                        else:
                            stats["skipped_unresolved_include"] += 1
                        continue

                    if kind == "endif":
                        if condition_stack:
                            condition_stack.pop()
                        continue
                    if kind == "else":
                        line = str(event["line"])
                        if condition_stack:
                            previous = condition_stack.pop()
                            parent_active = _conditions_active()
                            active = parent_active and not bool(previous["branch_taken"])
                            branch_lines = [*_condition_lines(previous), line]
                            condition_stack.append(
                                {
                                    "line": line,
                                    "lines": branch_lines,
                                    "names": previous["names"],
                                    "active": active,
                                    "branch_taken": bool(previous["branch_taken"]) or active,
                                }
                            )
                        continue
                    if kind == "if":
                        line = str(event["line"])
                        code_line = str(event["code_line"])
                        if_match = _IF_RE.match(code_line)
                        if if_match and if_match.group(1) == "elif" and condition_stack:
                            previous = condition_stack.pop()
                            parent_active = _conditions_active()
                            expr_active, names = _evaluate_condition(code_line)
                            active = (
                                parent_active
                                and not bool(previous["branch_taken"])
                                and expr_active
                            )
                            branch_lines = [*_condition_lines(previous), line]
                            condition_stack.append(
                                {
                                    "line": line,
                                    "lines": branch_lines,
                                    "names": names,
                                    "active": active,
                                    "branch_taken": bool(previous["branch_taken"]) or active,
                                }
                            )
                        else:
                            parent_active = _conditions_active()
                            expr_active, names = _evaluate_condition(code_line)
                            active = parent_active and expr_active
                            condition_stack.append(
                                {
                                    "line": line,
                                    "lines": [line],
                                    "names": names,
                                    "active": active,
                                    "branch_taken": active,
                                }
                            )
                        continue

                    if not _conditions_active():
                        continue

                    if kind == "undef":
                        name = str(event["name"])
                        line = str(event["line"])
                        if name in relevant_names:
                            pending_undef[name] = line
                        active_defs.pop(name, None)
                        active_macro_texts.pop(name, None)
                        continue

                    if kind != "define":
                        continue
                    name = str(event["name"])
                    line_no = int(event["line_no"])
                    macro_text = str(event["macro_text"])
                    params = cast(list[str], event["params"])
                    active_macro_texts[name] = macro_text
                    if not bool(event["include_guard"]):
                        root_discovered_macros += 1
                        stats["discovered_macro_occurrences"] += 1
                        if name not in relevant_names:
                            continue
                        condition_lines = [
                            line
                            for item in condition_stack
                            for line in _condition_lines(item)
                            if line
                        ]
                        condition_names: list[str] = []
                        for item in condition_stack:
                            condition_names.extend(cast(list[str], item["names"]))
                        undef_text = pending_undef.pop(name, "")
                        text_parts = [*condition_lines]
                        if undef_text:
                            text_parts.append(undef_text)
                        text_parts.append(macro_text)
                        text_parts.extend("#endif\n" for _item in condition_stack)
                        macro = MacroDef(
                            name=name,
                            file=rel_path,
                            line=line_no,
                            text="".join(text_parts),
                            params=params,
                            project_id=stable_project_id,
                            visible_in_file=root_rel,
                            visible_line=root_visible_line or line_no,
                            sequence=sequence[0],
                            condition_names=condition_names,
                            condition_lines=condition_lines,
                            undef_text=undef_text,
                            previous=last_defs.get(name),
                        )
                        _count_macro_object()
                        sequence[0] += 1
                        active_defs[name] = macro
                        last_defs[name] = macro
                        root_index.add_macro(macro)
            finally:
                include_stack.remove(norm_abs)

        _scan_file(
            root_abs,
            root_rel,
            root_visible_line=None,
            include_stack=set(),
            depth=0,
        )

        directly_used = [
            macro
            for name in sorted(usage_names)
            if (
                macro := _select_visible_macro(
                    root_index,
                    name,
                    target_file=root_rel,
                    max_line=None,
                )
            )
            is not None
        ]
        retained = _macro_dependency_closure(
            root_index,
            directly_used,
            target_file=root_rel,
            max_line=None,
        )
        stats["peak_root_retained_macros"] = max(
            stats["peak_root_retained_macros"],
            len(retained),
        )
        root_candidate_ids = {id(macro) for macro in root_index.macro_definitions}
        retained_candidate_ids = {
            id(macro) for macro in retained if id(macro) in root_candidate_ids
        }
        stats["pruned_macro_occurrences"] += max(
            0,
            root_discovered_macros - len(retained_candidate_ids),
        )
        incoming_canonical = index.new_canonical_macro_definition_count(
            retained
        )
        if (
            len(index.macro_definitions) + incoming_canonical
            > max_retained_macros
        ):
            raise MemoryError(
                "macro scan exceeded the canonical definition registry bound: "
                f"root={root_rel} canonical={len(index.macro_definitions)} "
                f"incoming_canonical={incoming_canonical} "
                f"visibility_records={index.macro_visibility_record_count} "
                f"limit={max_retained_macros}. "
                "Raise CPPMEGA_MAX_RETAINED_MACROS only after inspecting macro "
                "retention telemetry."
            )
        registered, visibility_count, visibility_bytes = index.add_macro_environment(
            root_rel,
            retained,
        )
        stats["registered_macros"] += visibility_count
        stats["registered_canonical_macros"] += registered
        stats["retained_macro_occurrences"] += visibility_count
        stats["compacted_macro_occurrences"] += (
            visibility_count - registered
        )
        stats["macro_visibility_records"] = (
            index.macro_visibility_record_count
        )
        stats["canonical_macro_definitions"] = len(
            index.macro_definitions
        )
        stats["macro_visibility_bytes"] = index.macro_visibility_bytes
        stats["last_root_macro_visibility_bytes"] = visibility_bytes
        if index.macro_visibility_bytes > max_macro_visibility_bytes:
            raise MemoryError(
                "macro scan exceeded the root visibility byte bound: "
                f"root={root_rel} "
                f"visibility_records={index.macro_visibility_record_count} "
                f"visibility_bytes={index.macro_visibility_bytes} "
                f"limit={max_macro_visibility_bytes}. "
                "The default preserves the prior MacroDef registry's measured "
                "minimum shallow-memory envelope; raise "
                "CPPMEGA_MAX_MACRO_VISIBILITY_BYTES only with retained-view "
                "telemetry."
            )
        _check_memory()

    seen_roots: set[str] = set()
    for path in header_files:
        rel = _rel(path)
        ext = os.path.splitext(rel)[1].lower()
        if ext not in INDEX_EXTENSIONS:
            continue
        abs_path = path
        if project_dir is not None and not os.path.isabs(abs_path):
            abs_path = os.path.join(project_dir, abs_path)
        normalized_abs = os.path.normpath(abs_path)
        if normalized_abs in seen_roots:
            stats["skipped_duplicate_roots"] += 1
            continue
        seen_roots.add(normalized_abs)
        _scan_root(normalized_abs, rel)

    return dict(stats)


def select_macro_scan_files(
    cpp_files: list[str],
    index: ProjectIndex,
    header_files: list[str],
    *,
    project_dir: str | None,
) -> list[str]:
    """Return macro roots that can contribute to emitted training documents.

    Full macro routing needs source-file visibility, but scanning every file in
    an archive dump also records millions of duplicate visible definitions for
    files that never emit docs. Keep files with indexed functions/types; their
    include traversal still reaches macro/type headers that are actually visible
    to trainable code. Skip no-signal sources and no-signal headers as roots.
    """
    known_by_rel: dict[str, str] = {}
    for path in cpp_files:
        rel = (
            os.path.relpath(path, project_dir)
            if project_dir is not None and os.path.isabs(path)
            else path
        )
        known_by_rel[os.path.normpath(rel)] = path

    selected: dict[str, str] = {}

    def add_path(path_or_rel: str) -> None:
        rel = (
            os.path.relpath(path_or_rel, project_dir)
            if project_dir is not None and os.path.isabs(path_or_rel)
            else path_or_rel
        )
        rel = os.path.normpath(rel)
        path = known_by_rel.get(rel)
        if path is None:
            path = (
                os.path.join(project_dir, rel)
                if project_dir is not None and not os.path.isabs(rel)
                else rel
            )
        selected[rel] = path

    for rel in index.file_functions:
        add_path(rel)
    for td in index.typedefs.values():
        if td.file:
            add_path(td.file)

    return [selected[rel] for rel in sorted(selected)]


def _select_visible_macro(
    index: ProjectIndex,
    name: str,
    *,
    target_file: str | None,
    max_line: int | None,
    before_sequence: int | None = None,
) -> MacroDef | None:
    if (
        target_file is not None
        and target_file in index.macro_visibility_by_file
    ):
        records = index.macro_visibility_by_file[target_file].get(name, [])
        scoped_records = [
            record
            for record in records
            if (max_line is None or record.visible_line <= max_line)
            and (
                before_sequence is None
                or record.sequence < before_sequence
            )
        ]
        if not scoped_records:
            return None
        selected = max(
            scoped_records,
            key=lambda record: (record.visible_line, record.sequence),
        )
        materialized: dict[int, MacroDef] = {}

        def materialize(record: _MacroVisibilityRecord) -> MacroDef:
            cached = materialized.get(id(record))
            if cached is not None:
                return cached
            definition = record.definition
            view = MacroDef(
                name=definition.name,
                file=definition.file,
                line=definition.line,
                text=definition.text,
                params=definition.params,
                project_id=definition.project_id,
                visible_in_file=target_file,
                visible_line=record.visible_line,
                sequence=record.sequence,
                condition_names=definition.condition_names,
                condition_lines=definition.condition_lines,
                undef_text=definition.undef_text,
                previous=(
                    materialize(record.previous)
                    if record.previous is not None
                    else None
                ),
            )
            materialized[id(record)] = view
            return view

        return materialize(selected)

    candidates = list(index.macros_by_name.get(name, []))
    if not candidates:
        return None
    if target_file is not None:
        scoped = [
            macro for macro in candidates
            if macro.visible_in_file == target_file
            and (max_line is None or macro.visible_line <= max_line)
            and (before_sequence is None or macro.sequence < before_sequence)
        ]
        if not scoped:
            return None
        candidates = scoped
    elif before_sequence is not None:
        scoped = [macro for macro in candidates if macro.sequence < before_sequence]
        if scoped:
            candidates = scoped
    return max(candidates, key=lambda macro: (macro.visible_line, macro.sequence))


def _macro_dependency_closure(
    index: ProjectIndex,
    selected: list[MacroDef],
    *,
    target_file: str | None,
    max_line: int | None,
) -> list[MacroDef]:
    out: dict[tuple[str, str, int, int], MacroDef] = {}
    visiting: set[tuple[str, str, int, int]] = set()

    def add(macro: MacroDef | None) -> None:
        if macro is None:
            return
        key = (macro.visible_in_file, macro.file, macro.line, macro.sequence)
        if key in out or key in visiting:
            return
        visiting.add(key)
        if macro.previous is not None:
            add(macro.previous)
        for cond_name in macro.condition_names:
            add(
                _select_visible_macro(
                    index,
                    cond_name,
                    target_file=target_file,
                    max_line=max_line,
                    before_sequence=macro.sequence,
                )
            )
        for body_name in _macro_body_dependency_names(
            macro.text,
            macro_name=macro.name,
            params=macro.params,
        ):
            add(
                _select_visible_macro(
                    index,
                    body_name,
                    target_file=target_file,
                    max_line=max_line,
                    before_sequence=macro.sequence,
                )
            )
        out[key] = macro
        visiting.remove(key)

    for macro in selected:
        add(macro)
    return sorted(out.values(), key=lambda macro: macro.sequence)


def _used_macro_defs(
    index: ProjectIndex,
    texts: list[str] | list[tuple[str, int | None]],
    *,
    target_file: str | None = None,
    max_line: int | None = None,
) -> list[MacroDef]:
    used: dict[tuple[str, str, int, int], MacroDef] = {}
    if not index.macros_by_name:
        return []

    def _iter_texts() -> Iterator[tuple[str, int | None]]:
        for item in texts:
            if isinstance(item, tuple):
                text, start_line = item
                yield str(text), int(start_line) if start_line is not None else None
            else:
                yield str(item), max_line

    for text, start_line in _iter_texts():
        for match in _iter_code_identifiers(text):
            use_line = (
                _source_line_for_text_offset(
                    text,
                    start_line=start_line,
                    char_offset=match.start(),
                )
                if start_line is not None
                else max_line
            )
            macro = _select_visible_macro(
                index,
                match.group(0),
                target_file=target_file,
                max_line=use_line,
            )
            if macro is not None:
                key = (
                    macro.visible_in_file,
                    macro.file,
                    macro.line,
                    macro.sequence,
                )
                used[key] = macro
    return _macro_dependency_closure(
        index,
        list(used.values()),
        target_file=target_file,
        max_line=max_line,
    )


def _compile_context_for_rel_file(
    rel_file: str,
    *,
    project_dir: str | None,
    compile_db: dict | None,
    default_args: list[str] | None,
    default_build_info: dict | None,
) -> tuple[list[str], dict | None]:
    compile_args = list(default_args or [])
    build_info = dict(default_build_info) if default_build_info else None
    context_path = rel_file
    if project_dir is not None:
        context_path = os.path.normpath(os.path.join(project_dir, rel_file))
        file_build = compile_db.get(context_path) if compile_db else None
        if file_build:
            compile_args = file_build.get("compile_args", compile_args)
            build_info = file_build.get("build_info") or build_info
    compile_args = _adapt_args_for_file(
        _sanitize_compile_args_for_clang(compile_args),
        context_path,
    )
    build_info = _harmonize_build_info_with_compile_args(
        build_info,
        compile_args,
    )
    return compile_args, build_info


def build_header_fragment_doc(
    *,
    index: ProjectIndex,
    rel_file: str,
    text: str,
    kind: int,
    name: str,
    qname: str | None,
    fragment_kind: str,
    compile_args: list[str] | None,
    build_info: dict | None,
    project_id: str,
    part: PartInfo | None = None,
) -> dict:
    doc = build_enriched_doc(
        [part or (text, kind, 0, name, qname)],
        index,
        filepath=rel_file,
        compile_args=compile_args,
        build_info=build_info,
        project_id=project_id,
    )
    doc["doc_type"] = "code_header"
    doc["header_fragment_kind"] = fragment_kind
    return doc


# --------------------------------------------------------------------------- #
# Function-level tokenized-hash dedup + semantic chunk claims.
#
# Per the user-specified ordering, dedup happens at the FUNCTION level, on the
# function AFTER OUR TOKENIZER (the whitespace sentinels canonicalize format),
# and BEFORE the dependency grouping. The dedup decides which functions get
# emitted as their OWN ROOT document. A separate semantic chunk-claim ledger then
# decides whether a function/class/type body may appear ANYWHERE in training
# output. That second ledger is what prevents the same helper from being packed
# in 1024 and 4096 buckets at the same time.
# --------------------------------------------------------------------------- #
_CPPMEGA_TOKENIZER = None


def _load_cppmega_tokenizer(tokenizer_path: str):
    """Load (and memoize) OUR CppMegaTokenizer. FAIL LOUD on failure (RULE #1)."""
    global _CPPMEGA_TOKENIZER
    if _CPPMEGA_TOKENIZER is not None:
        return _CPPMEGA_TOKENIZER
    from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
    _CPPMEGA_TOKENIZER = load_cppmega_tokenizer(tokenizer_path)
    return _CPPMEGA_TOKENIZER


class TrainingChunkClaims:
    """Claim semantic function/class/type chunks before assembling training docs.

    The granularity is the caller-provided semantic chunk: function body/text or
    class/typedef/enum text. The tokenized hash is only the storage key, chosen so
    claim identity matches what the model will actually see after our tokenizer.
    """

    def __init__(
        self,
        *,
        tokenizer_path: str,
        dedup_db: str | None,
        dedup_stage_id: str | None = None,
        dedup_stage_db: str | None = None,
        namespace: str = "semantic_chunk:v1",
    ):
        self.tok = _load_cppmega_tokenizer(tokenizer_path)
        self.namespace = namespace
        self.store = None
        self.local_counts: dict[bytes, int] = {}
        if dedup_db:
            DedupStore, _sha1_tokens = _import_dedup_store_symbols()
            self.store = DedupStore(
                dedup_db,
                near=False,
                commit_every=1,
                stage_id=dedup_stage_id,
                stage_db_path=dedup_stage_db,
            )

    def claim_text(self, text: str, *, max_count: int = 1) -> bool:
        if not text:
            return False
        token_ids = self.tok.encode(text)
        if not token_ids:
            return False
        if self.store is not None:
            return self.store.claim_chunk_tokens(
                token_ids,
                namespace=self.namespace,
                max_count=max_count,
            )
        _DedupStore, _sha1_tokens = _import_dedup_store_symbols()
        h = _sha1_tokens(token_ids)
        count = self.local_counts.get(h, 0)
        if count >= max_count:
            return False
        self.local_counts[h] = count + 1
        return True

    def close(self) -> None:
        if self.store is not None:
            self.store.close()


def emit_header_documents(
    *,
    index: ProjectIndex,
    header_files: list[str],
    project_dir: str | None,
    project_id: str,
    compile_db: dict | None,
    default_args: list[str] | None,
    default_build_info: dict | None,
    max_tokens: int,
    enriched: bool,
    chunk_claims: TrainingChunkClaims | None,
    emit_doc: Callable[[str | dict[str, object]], None],
) -> dict[str, int]:
    """Emit standalone header fragments for templates/types and macros.

    Function-root documents cannot cover header-only libraries by themselves:
    class templates, alias templates, and macro APIs often have no standalone
    function root.  This pass emits those fragments as C++ header docs while
    sharing the same semantic chunk-claim ledger as function/type chunks.
    """
    stats: defaultdict[str, int] = defaultdict(int)
    if not header_files:
        return stats

    header_rel_files: set[str] = set()
    header_abs_by_rel: dict[str, str] = {}
    for path in header_files:
        rel = (
            os.path.relpath(path, project_dir)
            if project_dir is not None and os.path.isabs(path)
            else path
        )
        ext = os.path.splitext(rel)[1].lower()
        if ext not in HEADER_EXTENSIONS:
            continue
        header_rel_files.add(rel)
        abs_path = path
        if project_dir is not None and not os.path.isabs(abs_path):
            abs_path = os.path.join(project_dir, abs_path)
        header_abs_by_rel[rel] = abs_path

    typedefs_by_file: dict[str, list[TypeDef]] = defaultdict(list)
    for td in index.typedefs.values():
        if td.file in header_rel_files and td.text:
            typedefs_by_file[td.file].append(td)

    for rel in sorted(header_rel_files):
        compile_args, build_info = _compile_context_for_rel_file(
            rel,
            project_dir=project_dir,
            compile_db=compile_db,
            default_args=default_args,
            default_build_info=default_build_info,
        )

        for td in sorted(typedefs_by_file.get(rel, []), key=lambda item: (item.line, item.qualified_name)):
            if estimate_tokens(td.text) > max_tokens * 2:
                stats["skipped_header_type_oversize"] += 1
                continue
            if chunk_claims is not None and not chunk_claims.claim_text(td.text, max_count=2):
                stats["skipped_header_type"] += 1
                continue
            stats["header_type"] += 1
            doc = build_header_fragment_doc(
                index=index,
                rel_file=rel,
                text=td.text,
                kind=td.kind,
                name=td.name,
                qname=td.qualified_name,
                fragment_kind="header_decl" if td.kind == HEADER_FRAGMENT_KIND else "type",
                compile_args=compile_args,
                build_info=build_info,
                project_id=project_id,
                part=_typedef_part(td),
            )
            emit_doc(doc if enriched else td.text)

        abs_path = header_abs_by_rel.get(rel)
        if not abs_path:
            continue
        try:
            header_text, _header_bytes, _header_encoding = _read_source_file(
                abs_path
            )
        except OSError as exc:
            raise RuntimeError(f"failed to read header for macro extraction: {abs_path}") from exc

        for start, _end, name, macro_text in extract_macro_blocks(header_text):
            if estimate_tokens(macro_text) > max_tokens * 2:
                stats["skipped_header_macro_oversize"] += 1
                continue
            if chunk_claims is not None and not chunk_claims.claim_text(macro_text, max_count=2):
                stats["skipped_header_macro"] += 1
                continue
            stats["header_macro"] += 1
            line = header_text.count("\n", 0, start) + 1
            macro = next(
                (
                    candidate
                    for candidate in index.macros_by_name.get(name, [])
                    if candidate.file == rel and candidate.line == line
                ),
                MacroDef(
                    name,
                    rel,
                    line,
                    macro_text,
                    project_id=project_id,
                ),
            )
            doc = build_header_fragment_doc(
                index=index,
                rel_file=rel,
                text=macro_text,
                kind=MACRO_KIND,
                name=name,
                qname=None,
                fragment_kind="macro",
                compile_args=compile_args,
                build_info=build_info,
                project_id=project_id,
                part=_macro_part(macro),
            )
            emit_doc(doc if enriched else macro_text)

    return stats


def dedup_root_functions(
    index: ProjectIndex,
    *,
    tokenizer_path: str,
    dedup_db: str | None,
    dedup_stage_id: str | None = None,
    dedup_stage_db: str | None = None,
    near: bool = True,
) -> tuple[set[str], dict[str, int]]:
    """Decide which root functions are duplicates (drop their standalone doc).

    For each function definition we compute ``token_ids = tokenizer.encode(text)``
    and test against:
      * exact: ``sha1(token_ids)`` (identical-after-tokenizer), and
      * near: MinHash-LSH @0.7 over token-id 5-gram shingles.

    A function whose hash was already seen is marked as a DROPPED ROOT: the
    grouping loop will not emit a standalone document for it. Functions stay in
    ``index.functions`` untouched so ``collect_transitive_deps`` can still
    resolve graph edges. Whether the resolved dependency text may be embedded is
    controlled later by ``TrainingChunkClaims``.

    When ``dedup_db`` is given the dedup is GLOBAL + resumable + cross-stream via
    the shared SQLite ``DedupStore`` (fail-loud if it cannot open / datasketch is
    missing). When absent, a per-repo in-RAM exact set is used (no near).

    Returns (dropped_root_symbol_keys, {dropped_exact, dropped_near, kept_roots}).
    """
    tok = _load_cppmega_tokenizer(tokenizer_path)

    store = None
    seen_local: set[bytes] = set()
    if dedup_db:
        # FAIL LOUD: open failure / missing datasketch raises inside DedupStore.
        DedupStore, _sha1_tokens = _import_dedup_store_symbols()
        store = DedupStore(
            dedup_db,
            near=near,
            commit_every=2000,
            stage_id=dedup_stage_id,
            stage_db_path=dedup_stage_db,
        )

    dropped_roots: set[str] = set()
    dropped_exact = 0
    dropped_near = 0
    kept_roots = 0

    # Deterministic order: by (file, line) so the canonical kept copy is stable
    # across runs (important for resumability and reproducible packs).
    items = sorted(
        index.functions.items(),
        key=lambda kv: (kv[1].file or "", kv[1].line or 0, kv[0]),
    )
    for symbol_key, func in items:
        if not (func.is_definition and func.text):
            continue
        token_ids = tok.encode(func.text)
        if store is not None:
            if store.seen_exact_tokens(token_ids):
                dropped_exact += 1
                dropped_roots.add(symbol_key)
                continue
            if near and store.seen_near_tokens(token_ids):
                dropped_near += 1
                dropped_roots.add(symbol_key)
                continue
        else:
            _DedupStore, _sha1_tokens = _import_dedup_store_symbols()
            h = _sha1_tokens(token_ids)
            if h in seen_local:
                dropped_exact += 1
                dropped_roots.add(symbol_key)
                continue
            seen_local.add(h)
        kept_roots += 1

    if store is not None:
        store.commit()
        store.close()

    return dropped_roots, {
        "kept_roots": kept_roots,
        "dropped_exact": dropped_exact,
        "dropped_near": dropped_near,
    }


def emit_build_documents(
    build_files: list[tuple[str, str]],
    *,
    source_root: str | None = None,
    project_id: str | None = None,
    default_build_info: dict | None,
    compile_index: object | None = None,
    max_chunk_bytes: int = BUILD_FILE_SIZE_CAP,
    tokenizer_path: str | None = None,
    dedup_db: str | None = None,
    dedup_stage_id: str | None = None,
    dedup_stage_db: str | None = None,
    dedup_near: bool = True,
    emit_doc: Callable[[dict[str, object]], None] | None = None,
    skip_invalid_inputs: bool = False,
) -> list[dict]:
    """Emit every bounded domain source chunk in source order.

    Inputs within the domain cap remain one whole-file document. Oversized SQL,
    build, shell, and diagnostic text use deterministic typed chunks; each
    chunk carries its exact source byte/character span. Chunk-level content
    dedup is intentionally disabled: repeated source chunks are still distinct
    source spans, and dropping one would create an unrecoverable hole. The
    tokenizer/dedup parameters remain accepted for caller compatibility; code
    function dedup continues in ``dedup_root_functions``.

    FAIL LOUD (RULE #1): an unreadable discovered file or a non-empty build file
    with invalid text RAISES. Structural NULs and malformed supported-encoding
    explicit domain inputs also raise before any chunk is emitted. UTF-8,
    BOM-marked UTF-16/32, filename-declared legacy SQL encodings, strict
    Windows-1252, and narrowly verified byte-oriented fixture contracts are
    decoded without replacement. Only zero-length inputs are skipped;
    whitespace is source content and remains losslessly represented.
    """
    docs: list[dict] = []
    if not build_files:
        return docs

    emitted = 0

    def record_doc(doc: dict[str, object]) -> None:
        nonlocal emitted
        if emit_doc is None:
            docs.append(doc)
        else:
            emit_doc(doc)
        emitted += 1

    skipped_zero_length = 0
    source_chars_in = 0
    source_chars_out = 0
    from cppmega.data.domain_ingestion import iter_domain_file_chunks

    for filepath, build_kind in sorted(build_files):
        try:
            source_chunks = iter_domain_file_chunks(
                filepath,
                max_chunk_bytes=max_chunk_bytes,
            )
            for source_chunk in source_chunks:
                text = source_chunk.text
                source_chars_in += len(text)
                if not text:
                    skipped_zero_length += 1
                    continue

                per_file_build_info = dict(default_build_info) if default_build_info else {}
                if build_kind not in {
                    "bash",
                    "sh",
                    "zsh",
                    "tcsh",
                    "ksh",
                    "powershell",
                    "pwsh",
                    "ps1",
                    "batch",
                    "cmd",
                    "python",
                    "assembly",
                    "sql",
                    "compiler_diagnostic",
                    "linker_diagnostic",
                    "build_diagnostic",
                    "sanitizer_output",
                    "test_output",
                    "tool_output",
                }:
                    per_file_build_info["build_system"] = build_kind

                record_doc(
                    build_build_doc(
                        filepath,
                        text,
                        build_kind,
                        source_root=source_root,
                        project_id=project_id,
                        build_info=per_file_build_info or None,
                        source_span=source_chunk.source_span(),
                    )
                )
                source_chars_out += len(text)
        except ValueError as exc:
            if skip_invalid_inputs and str(exc).startswith(
                (
                    "binary domain input contains NUL byte",
                    "decoded domain input contains NUL character",
                    "invalid UTF-8 domain input",
                    "invalid UTF-8 or Windows-1252 domain input",
                    "invalid UTF-16LE domain input",
                    "invalid UTF-16BE domain input",
                    "invalid UTF-32LE domain input",
                    "invalid UTF-32BE domain input",
                    "binary shell/domain input contains NUL byte",
                    "invalid UTF-8 shell/domain input",
                )
            ):
                print(f"  SKIP invalid typed domain input {filepath}: {exc}", file=sys.stderr)
                continue
            raise

    print(
        f"  Build docs: emitted={emitted} "
        f"source_chars_in={source_chars_in} "
        f"source_chars_out={source_chars_out} "
        f"skipped_zero_length={skipped_zero_length} "
        "source_chunk_dedup=disabled_for_lossless_spans",
        file=sys.stderr,
    )
    return docs


def build_training_documents(
    index: ProjectIndex,
    max_tokens: int = 16384,
    max_dep_depth: int = 5,
    enriched: bool = False,
    *,
    project_dir: str | None = None,
    project_id: str | None = None,
    compile_db: dict | None = None,
    default_args: list[str] | None = None,
    default_build_info: dict | None = None,
    tokenizer_path: str | None = None,
    dedup_db: str | None = None,
    dedup_stage_id: str | None = None,
    dedup_stage_db: str | None = None,
    dedup_near: bool = True,
    global_symbols: "GlobalSymbolReader | None" = None,
    header_files: list[str] | None = None,
    macro_scan_files: list[str] | None = None,
    emit_doc: Callable[[str | dict[str, object]], None] | None = None,
    memory_limit_gb: float | None = None,
) -> list:
    """Build training documents with bottom-up dependency ordering.

    Returns list[str] when enriched=False, list[dict] when enriched=True.  When
    ``emit_doc`` is supplied, documents are streamed to the callback as soon as
    they are built and the returned list stays empty.  Large repos carry
    per-token/per-char sidecar arrays, so holding all emitted docs until the end
    of the repo creates an avoidable RSS peak.

    When ``tokenizer_path`` is given, FUNCTION-LEVEL tokenized-hash root dedup
    runs before grouping, and semantic chunk claims run during grouping. Root
    dedup prevents duplicate standalone roots; chunk claims prevent the same
    function/class/type text from appearing in multiple 1k/2k/4k/8k outputs.

    When ``global_symbols`` is supplied (the optional --global-symbol-index), an
    unresolved callee that is a selected base-lib symbol has its DEFINITION pulled
    in as a DEEPEST dependency chunk tagged dep_source='crosslib:<repo>', under
    HARD per-doc bounds (CrossLinkBudget). Behavior is unchanged when None.
    """
    crosslink_total = 0
    crosslink_ambiguous_total = 0
    crosslink_ambiguity_examples: set[str] = set()
    documents: list[str | dict[str, object]] = []

    def _record_doc(doc: str | dict[str, object]) -> None:
        if emit_doc is not None:
            emit_doc(doc)
        else:
            documents.append(doc)

    index.compute_dep_levels()
    stable_project_id = (
        require_project_identity(
            project_id,
            source="build_training_documents",
        )
        if project_id is not None
        else None
    )
    if header_files or macro_scan_files:
        if stable_project_id is None:
            raise SymbolIdentityError(
                "build_training_documents macro scan requires project_id"
            )
        macro_stats = register_header_macros(
            index,
            macro_scan_files or header_files or [],
            project_dir=project_dir,
            project_id=stable_project_id,
            include_dirs=_collect_macro_include_dirs(
                project_dir=project_dir,
                compile_db=compile_db,
                default_args=default_args,
            ),
            memory_limit_gb=memory_limit_gb,
        )
        print(
            "  Macro scan: "
            f"roots={macro_stats.get('roots', 0)} "
            f"files={macro_stats.get('scanned_files', 0)} "
            f"file_reads={macro_stats.get('directive_file_reads', 0)} "
            f"directive_cache_hits={macro_stats.get('directive_cache_hits', 0)} "
            f"include_directives={macro_stats.get('include_directives', 0)} "
            f"resolve_cache_hits={macro_stats.get('include_resolve_cache_hits', 0)} "
            f"discovered={macro_stats.get('discovered_macro_occurrences', 0)} "
            f"materialized_peak={macro_stats.get('peak_root_macro_candidates', 0)} "
            f"retained_views={macro_stats.get('retained_macro_occurrences', 0)} "
            f"canonical={macro_stats.get('canonical_macro_definitions', 0)} "
            f"visibility_bytes={macro_stats.get('macro_visibility_bytes', 0)} "
            f"visibility_byte_limit={macro_stats.get('macro_visibility_byte_limit', 0)} "
            f"pruned={macro_stats.get('pruned_macro_occurrences', 0)} "
            f"directive_cache_peak={macro_stats.get('directive_cache_peak_entries', 0)} "
            f"resolve_cache_peak={macro_stats.get('resolve_cache_peak_entries', 0)} "
            f"local_file_index_entries={macro_stats.get('local_file_index_entries', 0)} "
            f"skipped_depth={macro_stats.get('skipped_include_depth', 0)} "
            f"skipped_file_cap={macro_stats.get('skipped_include_file_cap', 0)} "
            f"skipped_unresolved={macro_stats.get('skipped_unresolved_include', 0)}",
            file=sys.stderr,
        )
        capped_skips = {
            "skipped_include_depth": macro_stats.get("skipped_include_depth", 0),
            "skipped_include_file_cap": macro_stats.get("skipped_include_file_cap", 0),
        }
        if any(int(value) for value in capped_skips.values()):
            raise RuntimeError(
                "macro scan truncated project-local include traversal; "
                f"{capped_skips}. Increase/disable CPPMEGA_MACRO_INCLUDE_DEPTH "
                "and CPPMEGA_MACRO_INCLUDE_FILES_PER_ROOT before emitting docs."
            )

    # Function-level dedup BEFORE grouping (corrected design). Requires OUR
    # tokenizer; when no tokenizer_path is supplied we skip it (legacy callers /
    # unit tests) and rely on the caller's higher-level dedup.
    dropped_roots: set[str] = set()
    if tokenizer_path:
        dropped_roots, stats = dedup_root_functions(
            index,
            tokenizer_path=tokenizer_path,
            dedup_db=dedup_db,
            dedup_stage_id=dedup_stage_id,
            dedup_stage_db=dedup_stage_db,
            near=dedup_near,
        )
        print(
            f"  Function-level dedup: kept_roots={stats['kept_roots']} "
            f"dropped_exact={stats['dropped_exact']} "
            f"dropped_near={stats['dropped_near']}",
            file=sys.stderr,
        )

    chunk_claims: TrainingChunkClaims | None = None
    chunk_stats: defaultdict[str, int] = defaultdict(int)
    if tokenizer_path:
        chunk_claims = TrainingChunkClaims(
            tokenizer_path=tokenizer_path,
            dedup_db=dedup_db,
            dedup_stage_id=dedup_stage_id,
            dedup_stage_db=dedup_stage_db,
        )

    try:
        items = sorted(
            index.functions.items(),
            key=lambda kv: (
                -(kv[1].dep_level or 0),
                kv[1].file or "",
                kv[1].line or 0,
                kv[0],
            ),
        )
        for symbol_key, func in items:
            if not func.is_definition:
                continue
            # Skip emitting a STANDALONE root doc for a deduped function. It remains
            # resolvable as a graph node, but chunk-claiming below controls whether
            # its text may appear in training output again.
            if symbol_key in dropped_roots:
                continue

            # Collect transitive deps. When the global symbol index is present, also
            # collect cross-lib base-lib pulls (bounded; depth-1; tagged provenance).
            # When absent, crosslink_budget is None so collect_transitive_deps takes
            # the exact original path (behavior unchanged).
            crosslink_visited: dict[str, dict[str, object]] = {}
            crosslink_budget = (
                CrossLinkBudget() if global_symbols is not None else None
            )
            dep_keys = collect_transitive_deps(
                symbol_key, index, max_dep_depth,
                global_symbols=global_symbols,
                crosslink_visited=crosslink_visited,
                crosslink_budget=crosslink_budget,
            )
            if crosslink_budget is not None:
                crosslink_ambiguous_total += crosslink_budget.ambiguous_lookups
                for example in crosslink_budget.ambiguity_examples:
                    if len(crosslink_ambiguity_examples) < 8:
                        crosslink_ambiguity_examples.add(example)

            # Sort by dep_level (leaves/most foundational first). Claims are applied
            # after the candidate document passes the tiny-doc filter so a small
            # helper can still be used as dependency context for its first caller.
            dep_funcs: list[FunctionDef] = []
            for dep_key in dep_keys:
                df = index.functions.get(dep_key)
                if not (df and df.is_definition and df.text):
                    continue
                dep_funcs.append(df)
            dep_funcs.sort(key=lambda f: f.dep_level)
            preamble = index.file_preambles.get(func.file, '')
            macro_scan_inputs: list[tuple[str, int | None]] = []
            if preamble:
                macro_scan_inputs.append((preamble, 1))
            macro_scan_inputs.extend((df.text, df.line) for df in dep_funcs)
            macro_scan_inputs.append((func.text, func.line))
            macro_deps = _used_macro_defs(
                index,
                macro_scan_inputs,
                target_file=func.file,
                max_line=func.line,
            )

            # Resolve cross-lib pulls to (qname, record) DEEPEST deps. They are the
            # most foundational (base-lib impls) so they precede local deps in text.
            crosslink_deps: list[dict[str, object]] = [
                crosslink_visited[uid]
                for uid in sorted(crosslink_visited)
                if crosslink_visited[uid].get("text")
            ]

            def _assemble(dep_funcs_list: list[FunctionDef]) -> str:
                parts: list[str] = []
                if preamble:
                    parts.append(preamble)
                for macro in macro_deps:
                    parts.append(macro.text)
                # Cross-lib base-lib defs are the deepest foundation -> emitted first.
                for record in crosslink_deps:
                    parts.append(str(record["text"]))
                for df in dep_funcs_list:
                    parts.append(df.text)
                parts.append(func.text)
                return '\n\n'.join(parts)

            doc = _assemble(dep_funcs)
            tokens = estimate_tokens(doc)

            # Token budget management. Trim LOCAL deps from highest dep_level first;
            # cross-lib pulls are already bounded by CrossLinkBudget, but if the doc
            # is still too big after exhausting local deps, drop cross-lib pulls too.
            if tokens > max_tokens * 2 and (dep_funcs or macro_deps or crosslink_deps):
                while tokens > max_tokens * 2 and dep_funcs:
                    dep_funcs.pop()  # remove highest-level local dep
                    doc = _assemble(dep_funcs)
                    tokens = estimate_tokens(doc)
                while tokens > max_tokens * 2 and macro_deps:
                    macro_deps.pop()
                    doc = _assemble(dep_funcs)
                    tokens = estimate_tokens(doc)
                while tokens > max_tokens * 2 and crosslink_deps:
                    crosslink_deps.pop()
                    doc = _assemble(dep_funcs)
                    tokens = estimate_tokens(doc)

            if tokens < 20:
                continue

            if chunk_claims is not None:
                if not chunk_claims.claim_text(func.text, max_count=1):
                    chunk_stats["skipped_root"] += 1
                    continue
                chunk_stats["claimed_root"] += 1

                claimed_dep_funcs: list[FunctionDef] = []
                for df in dep_funcs:
                    if not chunk_claims.claim_text(df.text, max_count=1):
                        chunk_stats["skipped_dep"] += 1
                        continue
                    chunk_stats["claimed_dep"] += 1
                    claimed_dep_funcs.append(df)
                dep_funcs = claimed_dep_funcs

                claimed_macro_deps: list[MacroDef] = []
                for macro in macro_deps:
                    if not chunk_claims.claim_text(macro.text, max_count=1):
                        chunk_stats["skipped_macro_dep"] += 1
                        continue
                    chunk_stats["claimed_macro_dep"] += 1
                    claimed_macro_deps.append(macro)
                macro_deps = claimed_macro_deps

                claimed_crosslink_deps: list[dict[str, object]] = []
                for rec in crosslink_deps:
                    if not chunk_claims.claim_text(rec["text"], max_count=1):
                        chunk_stats["skipped_crosslib"] += 1
                        continue
                    chunk_stats["claimed_crosslib"] += 1
                    claimed_crosslink_deps.append(rec)
                crosslink_deps = claimed_crosslink_deps
                doc = _assemble(dep_funcs)
                tokens = estimate_tokens(doc)
            crosslink_total += len(crosslink_deps)

            # NOTE: doc-level md5 dedup removed (corrected design). Dedup/root claim
            # now happens at semantic function/class/type chunk granularity via
            # dedup_root_functions + TrainingChunkClaims.

            if enriched:
                # Build parts_info for enriched output. Thread repo-level build
                # context through fallback lanes so build-file-derived args keep
                # their authoritative provenance in language_info/build_info.
                parts_info: list[PartInfo] = []
                if preamble:
                    parts_info.append((preamble, 1, 0, '', None))  # kind=1 PREAMBLE
                for macro in macro_deps:
                    parts_info.append(_macro_part(macro))
                # Cross-lib base-lib pulls: DEEPEST deps. dep_level is set above the
                # max local dep_level so they sort as the most foundational chunks in
                # the SAME dep_levels/topo order, and they carry dep_source provenance
                # 'crosslib:<repo>' (6-tuple) so the model knows this is a base-lib
                # impl and which lib it came from. kind=2 (FUNC) like any function.
                _max_local_level = max(
                    (df.dep_level for df in dep_funcs), default=0
                )
                for rec in crosslink_deps:
                    parts_info.append(_global_symbol_part(rec, _max_local_level + 1))
                for df in dep_funcs:
                    parts_info.append(_function_part(df))
                parts_info.append(_function_part(func))

                # Pull in DEFINITIONS of types referenced by the root func + its
                # dep funcs as type-edge target chunks. build_enriched_doc matches
                # canonical referenced-type identities against type-definition
                # chunks to emit type_edges; the type def must therefore be present.
                # Definitions live in
                # headers (now indexed), registered in index.typedefs.
                func_symbols_in_doc = {func.symbol_key}
                func_symbols_in_doc.update(df.symbol_key for df in dep_funcs)
                referenced: list[str] = index._function_referenced_type_keys(func)
                for df in dep_funcs:
                    referenced.extend(index._function_referenced_type_keys(df))
                added_type_keys: set[str] = set()
                for type_key in referenced:
                    if type_key in added_type_keys or type_key in func_symbols_in_doc:
                        continue
                    td = index.typedefs.get(type_key)
                    if td is None or not td.text:
                        continue
                    if chunk_claims is not None:
                        if not chunk_claims.claim_text(td.text, max_count=1):
                            chunk_stats["skipped_type"] += 1
                            continue
                        chunk_stats["claimed_type"] += 1
                    added_type_keys.add(type_key)
                    # kind from TypeDef (4=class/struct/enum/union, 7=typedef/using)
                    parts_info.append(_typedef_part(td))

                compile_args = list(default_args or [])
                build_info = dict(default_build_info) if default_build_info else None
                context_path = func.file
                if project_dir is not None:
                    abs_func_path = os.path.normpath(os.path.join(project_dir, func.file))
                    context_path = abs_func_path
                    file_build = compile_db.get(abs_func_path) if compile_db else None
                    if file_build:
                        compile_args = file_build.get("compile_args", compile_args)
                        build_info = file_build.get("build_info") or build_info
                compile_args = _adapt_args_for_file(
                    _sanitize_compile_args_for_clang(compile_args),
                    context_path,
                )
                build_info = _harmonize_build_info_with_compile_args(
                    build_info,
                    compile_args,
                )
                out_doc = build_enriched_doc(
                    parts_info,
                    index,
                    filepath=func.file,
                    compile_args=compile_args,
                    build_info=build_info,
                    project_id=stable_project_id,
                )
                if _is_header_path(func.file) and func.text.lstrip().startswith("template"):
                    out_doc["doc_type"] = "code_header"
                    out_doc["header_fragment_kind"] = "function_template"
                _record_doc(out_doc)
            else:
                _record_doc(doc)

        if header_files:
            if stable_project_id is None:
                raise SymbolIdentityError(
                    "header emission requires a canonical owner/repo identity"
                )
            header_stats = emit_header_documents(
                index=index,
                header_files=header_files,
                project_dir=project_dir,
                project_id=stable_project_id,
                compile_db=compile_db,
                default_args=default_args,
                default_build_info=default_build_info,
                max_tokens=max_tokens,
                enriched=enriched,
                chunk_claims=chunk_claims,
                emit_doc=_record_doc,
            )
            for key, value in header_stats.items():
                chunk_stats[key] += value
    finally:
        if chunk_claims is not None:
            chunk_claims.close()

    if chunk_claims is not None:
        print(
            "  Semantic chunk claims: "
            f"root={chunk_stats['claimed_root']} "
            f"dep={chunk_stats['claimed_dep']} "
            f"macro_dep={chunk_stats['claimed_macro_dep']} "
            f"crosslib={chunk_stats['claimed_crosslib']} "
            f"type={chunk_stats['claimed_type']} "
            f"header_type={chunk_stats['header_type']} "
            f"header_macro={chunk_stats['header_macro']} "
            f"skipped_root={chunk_stats['skipped_root']} "
            f"skipped_dep={chunk_stats['skipped_dep']} "
            f"skipped_macro_dep={chunk_stats['skipped_macro_dep']} "
            f"skipped_crosslib={chunk_stats['skipped_crosslib']} "
            f"skipped_type={chunk_stats['skipped_type']} "
            f"skipped_header_type={chunk_stats['skipped_header_type']} "
            f"skipped_header_macro={chunk_stats['skipped_header_macro']} "
            f"skipped_header_type_oversize={chunk_stats['skipped_header_type_oversize']} "
            f"skipped_header_macro_oversize={chunk_stats['skipped_header_macro_oversize']} "
            "(function/class/type granularity, max_count=1)",
            file=sys.stderr,
        )

    if global_symbols is not None:
        print(
            f"  Cross-lib base-lib pulls: {crosslink_total} (depth-1, "
            f"cap {CROSSLINK_MAX_DEPS_PER_DOC}/doc, "
            f"{CROSSLINK_TOKEN_BUDGET_PER_DOC} tok/doc, tagged crosslib:<repo>); "
            f"ambiguous_unresolved={crosslink_ambiguous_total} "
            f"examples={sorted(crosslink_ambiguity_examples)}",
            file=sys.stderr,
        )
    return documents


def _resolve_file_args(filepath, compile_db, default_args):
    """Get per-file compile args, adapting C vs C++ flags."""
    if compile_db and filepath in compile_db:
        build_entry = compile_db[filepath]
        args = _sanitize_compile_args_for_clang(build_entry.get("compile_args", default_args))
    else:
        args = _sanitize_compile_args_for_clang(default_args)
    return _adapt_args_for_file(args, filepath)


CPP_LEXICAL_FALLBACK_CHUNK_BYTES = 500_000


def _cpp_lexical_fallback_compile_args_sha256(
    compile_args: Sequence[str],
) -> str:
    encoded_args = json.dumps(
        list(map(str, compile_args)),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded_args).hexdigest()


def _project_local_cpp_lexical_fallback_path(
    project_dir: str,
    relative_path: str,
) -> str:
    """Resolve one fallback source and reject root/symlink escapes."""

    canonical_path = _canonical_project_path(relative_path)
    project_root = os.path.realpath(project_dir)
    absolute_path = os.path.realpath(os.path.join(project_root, canonical_path))
    try:
        is_project_local = (
            not os.path.isabs(canonical_path)
            and "://" not in canonical_path
            and os.path.commonpath((project_root, absolute_path)) == project_root
        )
    except ValueError:
        is_project_local = False
    if not is_project_local:
        raise RuntimeError(
            "C/C++ lexical fallback path escapes the project root: "
            f"{relative_path}"
        )
    return absolute_path


def _record_cpp_lexical_fallback(
    filepath: str,
    compile_args: Sequence[str],
    project_dir: str,
    parse_error: BaseException,
    parse_recovery_records: list[dict[str, object]],
) -> str | None:
    """Authorize a lossless lexical fallback for one native libclang load error.

    Deliberately narrow: a bad/ambiguous compiler context must still fail loud,
    as must every exception other than TranslationUnitLoadError.  The returned
    path is project-relative and the receipt binds the exact source bytes that
    the later lexical emitter must reproduce.
    """

    normalized_args = list(map(str, compile_args))
    if (
        not _has_translation_unit_load_error(parse_error)
        or not _is_sane_compile_args(normalized_args)
    ):
        return None

    relative_path = _canonical_project_path(
        Path(os.path.relpath(filepath, project_dir)).as_posix()
    )
    absolute_path = _project_local_cpp_lexical_fallback_path(
        project_dir,
        relative_path,
    )
    source, source_bytes, source_encoding = _read_source_file(absolute_path)
    fallback_fields: dict[str, object] = {
        "status": "lexical_fallback",
        "fallback_mode": "lossless_cpp_lexical_v1",
        "fallback_reason": "translation_unit_load_error",
        "compile_args_status": "sane",
        "compile_arg_count": len(normalized_args),
        "compile_args_sha256": _cpp_lexical_fallback_compile_args_sha256(
            normalized_args
        ),
        "source_size_bytes": len(source_bytes),
        "source_char_count": len(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_encoding": source_encoding,
    }
    existing = next(
        (
            record
            for record in reversed(parse_recovery_records)
            if record.get("relative_path") == relative_path
            and record.get("status") == "unresolved"
        ),
        None,
    )
    if existing is None:
        parse_recovery_records.append(
            {
                "relative_path": relative_path,
                "trigger": "translation_unit_load_error",
                **fallback_fields,
            }
        )
    else:
        existing.update(fallback_fields)
    print(
        "  Parse lexical fallback: "
        f"{relative_path} bytes={len(source_bytes)} "
        f"sha256={fallback_fields['source_sha256']}",
        file=sys.stderr,
        flush=True,
    )
    return relative_path


def _iter_cpp_lexical_fallback_chunks(
    filepath: str,
    *,
    max_chunk_bytes: int = CPP_LEXICAL_FALLBACK_CHUNK_BYTES,
) -> Iterator[tuple[str, dict[str, int | str]]]:
    """Yield byte-exact decoded C/C++ chunks, preferring line boundaries."""

    if max_chunk_bytes < 4:
        raise ValueError(
            f"max_chunk_bytes must be at least 4, got {max_chunk_bytes}"
        )
    _source, source_bytes, source_encoding = _read_source_file(filepath)
    if not source_bytes:
        return

    byte_start = 0
    char_start = 0
    chunk_index = 0
    while byte_start < len(source_bytes):
        byte_end = min(byte_start + max_chunk_bytes, len(source_bytes))
        split_reason = "eof" if byte_end == len(source_bytes) else "byte_limit"
        if byte_end < len(source_bytes):
            newline = source_bytes.rfind(b"\n", byte_start, byte_end)
            if newline >= byte_start:
                byte_end = newline + 1
                split_reason = "line_boundary"

        while byte_end > byte_start:
            try:
                text = source_bytes[byte_start:byte_end].decode(
                    source_encoding,
                    errors="strict",
                )
            except UnicodeDecodeError:
                byte_end -= 1
                split_reason = "character_boundary"
                continue
            break
        else:
            raise ValueError(
                "cannot find a decodable C/C++ lexical fallback boundary: "
                f"{filepath} byte_start={byte_start} encoding={source_encoding}"
            )

        if text.encode(source_encoding, errors="strict") != source_bytes[
            byte_start:byte_end
        ]:
            raise ValueError(
                f"C/C++ lexical fallback chunk did not round-trip: {filepath}"
            )
        char_end = char_start + len(text)
        span: dict[str, int | str] = {
            "chunk_index": chunk_index,
            "byte_start": byte_start,
            "byte_end": byte_end,
            "char_start": char_start,
            "char_end": char_end,
            "source_size_bytes": len(source_bytes),
            "chunk_limit_bytes": max_chunk_bytes,
            "split_reason": split_reason,
            "source_encoding": source_encoding,
        }
        yield text, span
        byte_start = byte_end
        char_start = char_end
        chunk_index += 1


def emit_cpp_lexical_fallback_documents(
    relative_paths: Iterable[str],
    *,
    index: ProjectIndex,
    project_dir: str,
    project_id: str,
    compile_db: dict | None,
    default_args: list[str] | None,
    default_build_info: dict | None,
    parse_recovery_records: Sequence[dict[str, object]],
    enriched: bool,
    emit_doc: Callable[[str | dict[str, object]], None] | None = None,
) -> list[str | dict[str, object]]:
    """Emit every byte of each authorized lexical fallback source exactly once."""

    documents: list[str | dict[str, object]] = []
    stable_project_id = require_project_identity(
        project_id,
        source="emit_cpp_lexical_fallback_documents",
    )
    from cppmega.data.domain_schema import ParseConfidence

    for relative_path in sorted(set(map(str, relative_paths))):
        canonical_path = _canonical_project_path(relative_path)
        absolute_path = _project_local_cpp_lexical_fallback_path(
            project_dir,
            canonical_path,
        )
        source, source_bytes, source_encoding = _read_source_file(absolute_path)
        record = next(
            (
                candidate
                for candidate in parse_recovery_records
                if candidate.get("relative_path") == canonical_path
                and candidate.get("status") == "lexical_fallback"
            ),
            None,
        )
        if record is None:
            raise RuntimeError(
                "C/C++ lexical fallback emission lacks an authorization record: "
                f"{canonical_path}"
            )
        expected_binding = (
            int(record.get("source_size_bytes") or -1),
            int(record.get("source_char_count") or -1),
            str(record.get("source_sha256") or ""),
            str(record.get("source_encoding") or ""),
        )
        actual_binding = (
            len(source_bytes),
            len(source),
            hashlib.sha256(source_bytes).hexdigest(),
            source_encoding,
        )
        if actual_binding != expected_binding:
            raise RuntimeError(
                "C/C++ lexical fallback source changed after parse failure: "
                f"{canonical_path} expected={expected_binding} actual={actual_binding}"
            )

        compile_args, build_info = _compile_context_for_rel_file(
            canonical_path,
            project_dir=project_dir,
            compile_db=compile_db,
            default_args=default_args,
            default_build_info=default_build_info,
        )
        expected_compile_args_binding = str(
            record.get("compile_args_sha256") or ""
        )
        actual_compile_args_binding = (
            _cpp_lexical_fallback_compile_args_sha256(compile_args)
        )
        if (
            record.get("compile_args_status") != "sane"
            or actual_compile_args_binding != expected_compile_args_binding
        ):
            raise RuntimeError(
                "C/C++ lexical fallback compile args changed after parse failure: "
                f"{canonical_path} expected={expected_compile_args_binding} "
                f"actual={actual_compile_args_binding}"
            )
        emitted_bytes = bytearray()
        chunk_count = 0
        for text, source_span in _iter_cpp_lexical_fallback_chunks(absolute_path):
            emitted_bytes.extend(text.encode(source_encoding, errors="strict"))
            if enriched:
                chunk_name = os.path.basename(canonical_path)
                if source_span["chunk_index"]:
                    chunk_name += f"#{source_span['chunk_index']}"
                part: PartInfo = (
                    text,
                    0,
                    0,
                    chunk_name,
                    None,
                    None,
                    None,
                    None,
                    canonical_path,
                )
                document = build_enriched_doc(
                    [part],
                    index,
                    filepath=canonical_path,
                    compile_args=compile_args,
                    build_info=build_info,
                    project_id=stable_project_id,
                )
                document["source_span"] = dict(source_span)
                document["cpp_parse_fallback"] = {
                    "schema": "cppmega.cpp_parse_fallback_v1",
                    "mode": "lossless_cpp_lexical_v1",
                    "reason": "translation_unit_load_error",
                    "compile_args_status": "sane",
                    "source_sha256": actual_binding[2],
                    "source_encoding": source_encoding,
                    "source_span": dict(source_span),
                }
                document["domain_confidence_ids"] = [
                    int(ParseConfidence.HEURISTIC)
                ] * len(text)
                parse_info = cast(
                    dict[str, object],
                    document["domain_parse_info"],
                )
                parse_info.update(
                    {
                        "parser": "cpp-lexical",
                        "fallback_reason": "translation_unit_load_error",
                        "source_span": dict(source_span),
                    }
                )
                output: str | dict[str, object] = document
            else:
                output = text
            if emit_doc is None:
                documents.append(output)
            else:
                emit_doc(output)
            chunk_count += 1

        if bytes(emitted_bytes) != source_bytes:
            raise RuntimeError(
                f"C/C++ lexical fallback emission was not lossless: {canonical_path}"
            )
        if source and not chunk_count:
            raise RuntimeError(
                f"C/C++ lexical fallback emitted no chunks: {canonical_path}"
            )
        print(
            "  Emitted lossless C/C++ lexical fallback: "
            f"{canonical_path} chunks={chunk_count} bytes={len(source_bytes)}",
            file=sys.stderr,
            flush=True,
        )
    return documents


def _parse_file_batch(args_tuple):
    """Worker function for parallel parsing. Each worker creates its own Index."""
    filepaths, compile_db, default_args, project_dir, project_id = args_tuple
    sys.setrecursionlimit(50000)  # Set in each worker process too
    _configure_libclang()
    clang_index = Index.create()
    func_results: list[dict] = []
    type_results: list[dict] = []
    external_reference_omissions: ExternalReferenceOmissions = {}
    parse_recovery_records: list[dict[str, object]] = []
    lexical_fallback_files: list[str] = []
    last_heartbeat = time.monotonic()
    for idx, filepath in enumerate(filepaths, start=1):
        args = _resolve_file_args(filepath, compile_db, default_args)
        try:
            functions, typedefs = parse_translation_unit(
                filepath,
                clang_index,
                args,
                project_dir,
                project_id=project_id,
                external_reference_omissions=external_reference_omissions,
                allow_include_recovery=not (
                    compile_db and filepath in compile_db
                ),
                parse_recovery_records=parse_recovery_records,
            )
            func_results.extend(f.to_dict() for f in functions)
            type_results.extend(t.to_dict() for t in typedefs)
        except SymbolIdentityError:
            raise
        except Exception as exc:
            fallback_path = _record_cpp_lexical_fallback(
                filepath,
                args,
                project_dir,
                exc,
                parse_recovery_records,
            )
            if fallback_path is None:
                cause = exc.__cause__ or exc
                context = f"; {exc}" if cause is not exc else ""
                raise RuntimeError(
                    f"C/C++ parse failed for {filepath}: "
                    f"{type(cause).__name__}: {cause}{context}"
                ) from exc
            lexical_fallback_files.append(fallback_path)
        now = time.monotonic()
        if (
            idx == len(filepaths)
            or idx % PARSE_HEARTBEAT_FILES == 0
            or now - last_heartbeat >= PARSE_HEARTBEAT_SECONDS
        ):
            print(
                f"  Parse worker heartbeat: {idx}/{len(filepaths)} files",
                file=sys.stderr,
                flush=True,
            )
            last_heartbeat = now
    return {
        "functions": func_results,
        "typedefs": type_results,
        "external_reference_omissions": external_reference_omissions,
        "parse_recovery_records": parse_recovery_records,
        "lexical_fallback_files": lexical_fallback_files,
    }, len(filepaths)


def _iter_parse_batch_results(
    executor,
    batches: Iterable,
    *,
    max_in_flight: int,
    heartbeat_interval_s: float = PARSE_HEARTBEAT_SECONDS,
):
    """Yield batch results while keeping the parent process observably alive."""

    if max_in_flight <= 0:
        raise ValueError(f"max_in_flight must be positive, got {max_in_flight}")
    if heartbeat_interval_s <= 0:
        raise ValueError(
            "heartbeat_interval_s must be positive, got "
            f"{heartbeat_interval_s}"
        )
    batch_iter = iter(batches)
    pending = set()
    submitted_batches = 0
    completed_batches = 0

    def submit_one() -> bool:
        nonlocal submitted_batches
        try:
            batch = next(batch_iter)
        except StopIteration:
            return False
        pending.add(executor.submit(_parse_file_batch, batch))
        submitted_batches += 1
        return True

    try:
        while len(pending) < max_in_flight and submit_one():
            pass
        while pending:
            done, _ = wait(
                tuple(pending),
                timeout=heartbeat_interval_s,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                running_batches = sum(future.running() for future in pending)
                print(
                    "  Parse pool heartbeat: "
                    f"completed_batches={completed_batches} "
                    f"submitted_batches={submitted_batches} "
                    f"pending_batches={len(pending)} "
                    f"running_batches={running_batches}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            for future in done:
                pending.remove(future)
                completed_batches += 1
                yield future.result()
                while len(pending) < max_in_flight and submit_one():
                    pass
    finally:
        for future in pending:
            future.cancel()


def _write_source_quarantine_receipt(
    path: str | os.PathLike[str],
    receipt: dict[str, object],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_file(destination) as staged:
        staged.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _external_reference_omission_summary(
    omissions: ExternalReferenceOmissions,
) -> dict[str, object]:
    reference_records = [
        {
            "relation": relation,
            "qname": qname,
            "symbol_kind": symbol_kind,
            "observed_path": observed_path,
            "observations": observations,
        }
        for (relation, qname, symbol_kind, observed_path), observations in sorted(
            omissions.items()
        )
    ]
    encoded_reference_records = json.dumps(
        reference_records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    locations: dict[tuple[str, str, str], dict[str, object]] = {}
    for record in reference_records:
        location_key = (
            str(record["relation"]),
            str(record["symbol_kind"]),
            str(record["observed_path"]),
        )
        location = locations.setdefault(
            location_key,
            {"observations": 0, "qnames": []},
        )
        location["observations"] = (
            int(location["observations"]) + int(record["observations"])
        )
        qnames = location["qnames"]
        assert isinstance(qnames, list)
        qnames.append(str(record["qname"]))
    location_records: list[dict[str, object]] = []
    for (relation, symbol_kind, observed_path), location in sorted(locations.items()):
        qnames = sorted(set(str(value) for value in location["qnames"]))
        encoded_qnames = json.dumps(
            qnames,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        location_records.append(
            {
                "relation": relation,
                "symbol_kind": symbol_kind,
                "observed_path": observed_path,
                "observations": int(location["observations"]),
                "unique_qname_count": len(qnames),
                "qnames_sha256": hashlib.sha256(encoded_qnames).hexdigest(),
                "qname_examples": qnames[:8],
                "qname_examples_truncated": len(qnames) > 8,
            }
        )
    return {
        "schema": "cppmega.external_reference_omissions_v1",
        "status": "complete",
        "reason": "unknown_external_provider",
        "policy": "omit_graph_reference_keep_source_and_fail_on_other_identity_errors",
        "observation_count": sum(omissions.values()),
        "unique_reference_count": len(reference_records),
        "reference_records_sha256": hashlib.sha256(
            encoded_reference_records
        ).hexdigest(),
        "location_count": len(location_records),
        "locations": location_records,
    }


def _parse_recovery_summary(
    records: Iterable[dict[str, object]],
) -> dict[str, object]:
    normalized = sorted(
        (dict(record) for record in records),
        key=lambda record: (
            str(record.get("relative_path") or ""),
            str(record.get("trigger") or ""),
            str(record.get("status") or ""),
        ),
    )
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    semantic_recovered_count = sum(
        record.get("status") == "recovered" for record in normalized
    )
    lexical_fallback_count = sum(
        record.get("status") == "lexical_fallback" for record in normalized
    )
    recovered_count = semantic_recovered_count + lexical_fallback_count
    unresolved_count = sum(
        record.get("status") == "unresolved" for record in normalized
    )
    return {
        "schema": "cppmega.source_parse_recovery_v1",
        "status": (
            "complete_with_unresolved"
            if unresolved_count
            else "complete"
        ),
        "policy": (
            "retry_missing_includes_with_header_matched_project_local_dirs_"
            "only_without_compile_command_then_lossless_cpp_lexical_fallback_"
            "only_for_translation_unit_load_error_with_sane_compile_args"
        ),
        "attempted_file_count": len(normalized),
        "recovered_file_count": recovered_count,
        "semantic_recovered_file_count": semantic_recovered_count,
        "lexical_fallback_file_count": lexical_fallback_count,
        "lexical_fallback_source_bytes": sum(
            int(record.get("source_size_bytes") or 0)
            for record in normalized
            if record.get("status") == "lexical_fallback"
        ),
        "unresolved_file_count": unresolved_count,
        "records_sha256": hashlib.sha256(encoded).hexdigest(),
        "records": normalized,
    }


def process_project(
    project_dir: str,
    max_tokens: int = 16384,
    max_dep_depth: int = 5,
    parse_workers: int = 1,
    enriched: bool = False,
    extra_exclude_dirs: set[str] | None = None,
    memory_limit_gb: float = 10.0,
    tokenizer_path: str | None = None,
    dedup_db: str | None = None,
    dedup_stage_id: str | None = None,
    dedup_stage_db: str | None = None,
    dedup_near: bool = True,
    global_symbol_index: str | None = None,
    emit_doc: Callable[[str | dict[str, object]], None] | None = None,
    *,
    project_id: str,
    skip_invalid_domain_inputs: bool = False,
    source_quarantine_manifest: str | None = None,
    source_quarantine_receipt: str | None = None,
    macos_sdk_path: str | None = None,
) -> list:
    """Process a single project: parse all files, build index, generate docs.

    ``global_symbol_index`` (optional path) enables bounded cross-repo base-lib
    symbol linking; each worker process opens its OWN read-only connection to the
    store. None -> behavior unchanged. When ``emit_doc`` is supplied, generated
    docs are streamed to it and not accumulated in the returned list.
    """
    project_dir = os.path.abspath(project_dir)
    stable_project_id = require_project_identity(
        project_id,
        source=f"process_project({project_dir})",
    )

    # Open the cross-repo base-lib symbol index (read-only) for THIS process.
    global_symbols: GlobalSymbolReader | None = None
    if global_symbol_index:
        global_symbols = GlobalSymbolReader(global_symbol_index)
        print(f"  Cross-lib: using global symbol index {global_symbol_index}",
              file=sys.stderr)

    print(f"\n--- Processing project: {stable_project_id} ---", file=sys.stderr)

    if (source_quarantine_manifest is None) != (
        source_quarantine_receipt is None
    ):
        raise ValueError(
            "source_quarantine_manifest and source_quarantine_receipt must be "
            "provided together"
        )
    source_quarantine = (
        ProjectSourceQuarantine.load(
            source_quarantine_manifest,
            project_id=stable_project_id,
        )
        if source_quarantine_manifest is not None
        else None
    )
    quarantine_receipt: dict[str, object] | None = None
    quarantined_paths: set[str] = set()
    external_reference_omissions: ExternalReferenceOmissions = {}
    parse_recovery_records: list[dict[str, object]] = []
    lexical_fallback_files: set[str] = set()
    source_tree_entry_exclusions = GitTreeSourceEntryExclusions(project_dir)

    invalid_domain_paths: set[str] = set()

    def _handle_invalid_domain_input(path: Path, exc: ValueError) -> None:
        invalid_domain_paths.add(os.path.abspath(path))
        print(f"  SKIP invalid typed domain input {path}: {exc}", file=sys.stderr)

    invalid_handler = _handle_invalid_domain_input if skip_invalid_domain_inputs else None

    # Find source files
    cpp_files = find_cpp_files(
        project_dir,
        extra_exclude_dirs=extra_exclude_dirs,
        ineligible_entry_handler=(
            source_tree_entry_exclusions.record
            if source_quarantine is not None
            else None
        ),
    )
    if source_quarantine is not None:
        quarantine_candidates = list(cpp_files)
        candidate_identities = {
            os.path.abspath(candidate) for candidate in quarantine_candidates
        }
        for relative_path in source_quarantine.entries_by_path:
            candidate = os.path.abspath(os.path.join(project_dir, relative_path))
            if candidate not in candidate_identities:
                quarantine_candidates.append(candidate)
                candidate_identities.add(candidate)
        kept_candidates, quarantine_receipt = source_quarantine.filter_candidates(
            project_dir,
            quarantine_candidates,
        )
        quarantine_receipt["source_tree_entry_exclusions"] = (
            source_tree_entry_exclusions.receipt()
        )
        quarantine_receipt["schema"] = "cppmega.source_quarantine_receipt_v2"
        kept_candidate_identities = {
            os.path.abspath(candidate) for candidate in kept_candidates
        }
        quarantined_paths = {
            os.path.abspath(os.path.join(project_dir, relative_path))
            for relative_path in source_quarantine.entries_by_path
        }
        cpp_files = [
            candidate
            for candidate in cpp_files
            if os.path.abspath(candidate) in kept_candidate_identities
        ]
        assert source_quarantine_receipt is not None
        _write_source_quarantine_receipt(
            source_quarantine_receipt,
            quarantine_receipt,
        )
        print(
            "  Source quarantine: "
            f"verified={quarantine_receipt['quarantined_count']} "
            f"manifest_sha256={quarantine_receipt['manifest_sha256']}",
            file=sys.stderr,
        )
    print(f"  Found {len(cpp_files)} C/C++ source files", file=sys.stderr)
    if cpp_files:
        _configure_libclang()

    # Discover build/compilation files (ADDITIVE; emitted as 'build' docs).
    build_files = find_build_files(project_dir, extra_exclude_dirs=extra_exclude_dirs)
    print(f"  Found {len(build_files)} build/compilation files", file=sys.stderr)
    shell_files = find_shell_files(
        project_dir,
        extra_exclude_dirs=extra_exclude_dirs,
        invalid_input_handler=invalid_handler,
        excluded_paths=quarantined_paths,
    )
    print(f"  Found {len(shell_files)} project shell files", file=sys.stderr)
    from cppmega.data.domain_ingestion import discover_project_domain_files

    typed_domain_files = discover_project_domain_files(
        project_dir,
        extra_exclude_dirs=extra_exclude_dirs,
        include_cpp=False,
        invalid_input_handler=invalid_handler,
        excluded_paths=quarantined_paths,
    )
    domain_files_by_path = {
        os.path.abspath(filepath): (os.path.abspath(filepath), build_kind)
        for filepath, build_kind in (*build_files, *shell_files)
    }
    for discovered in typed_domain_files:
        filepath = os.path.abspath(discovered.path)
        if filepath in domain_files_by_path and classify_build_file(
            os.path.basename(filepath)
        ) is not None:
            # Exact build-file identity wins over a generic extension adapter
            # (notably conanfile.py -> conan rather than plain Python).
            continue
        # PowerShell deliberately shares the frozen SH domain ID, so its
        # adapter (not DomainKind.name) owns the dialect used downstream.
        discovered_kind = {
            "powershell": "powershell",
            "cmd": "cmd",
        }.get(discovered.adapter, discovered.domain.name.lower())
        domain_files_by_path[filepath] = (
            filepath,
            discovered_kind,
        )
    domain_files = sorted(domain_files_by_path.values())
    if invalid_domain_paths:
        domain_files = [
            item for item in domain_files if item[0] not in invalid_domain_paths
        ]
    print(
        f"  Found {len(domain_files)} total typed domain files after dedup",
        file=sys.stderr,
    )

    # Load or derive build context. Done unconditionally so build-only repos
    # (no C/C++ at all) still get A-platform enrichment for their build docs.
    compile_db = load_compile_commands(project_dir)
    _platform_info, _raw_args, _compile_index = detect_build_context(
        project_dir,
        macos_sdk_path=macos_sdk_path,
    )
    default_args, default_build_info = _resolve_default_compile_context(
        project_dir,
        _platform_info,
        _raw_args,
    )
    default_args = _sanitize_compile_args_for_clang(default_args)

    # Parse all files and build index
    index_obj = ProjectIndex()

    # Use parallel parsing for large projects
    effective_workers = min(parse_workers, max(1, len(cpp_files) // 100))
    if not cpp_files:
        # Build-only repo (no C/C++): skip libclang parsing entirely. Build docs
        # are still emitted below so we do NOT silently drop the repo.
        print("  No C/C++ sources -- emitting build docs only", file=sys.stderr)
    elif effective_workers > 1 and len(cpp_files) > 200:
        print(f"  Using {effective_workers} parse workers", file=sys.stderr)
        chunk_size = compute_parse_batch_size(len(cpp_files), effective_workers)
        print(
            f"  Parse batch size: {chunk_size} files "
            f"(max {DEFAULT_PARSE_BATCH_FILES}; bounded IPC payload)",
            file=sys.stderr,
        )
        submit_window = max(
            1,
            effective_workers * PARSE_SUBMIT_WINDOW_PER_WORKER,
        )
        print(
            f"  Parse submit window: {submit_window} batches "
            f"({PARSE_SUBMIT_WINDOW_PER_WORKER} per worker)",
            file=sys.stderr,
        )
        batches = []
        for i in range(0, len(cpp_files), chunk_size):
            batch = cpp_files[i:i + chunk_size]
            batches.append(
                (batch, compile_db, default_args, project_dir, stable_project_id)
            )

        total_parsed = 0
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            for payload, parsed_count in _iter_parse_batch_results(
                executor,
                batches,
                max_in_flight=submit_window,
            ):
                for d in payload["functions"]:
                    index_obj.add_function(FunctionDef.from_dict(d))
                for td in payload["typedefs"]:
                    index_obj.add_typedef(TypeDef.from_dict(td))
                for key, count in payload["external_reference_omissions"].items():
                    external_reference_omissions[key] = (
                        external_reference_omissions.get(key, 0) + int(count)
                    )
                parse_recovery_records.extend(payload["parse_recovery_records"])
                lexical_fallback_files.update(payload["lexical_fallback_files"])
                total_parsed += parsed_count
                check_memory_limit(memory_limit_gb, label="index_project")
                print(
                    f"  Parsed {total_parsed}/{len(cpp_files)} files, "
                    f"{len(index_obj.functions)} functions",
                    file=sys.stderr,
                )

        print(
            f"  Parsed {total_parsed} files, "
            f"{len(index_obj.functions)} functions indexed",
            file=sys.stderr,
        )
    else:
        # Sequential for small projects
        clang_index = Index.create()
        parsed = 0
        last_heartbeat = time.monotonic()
        for filepath in cpp_files:
            args = _resolve_file_args(filepath, compile_db, default_args)
            try:
                functions, typedefs = parse_translation_unit(
                    filepath,
                    clang_index,
                    args,
                    project_dir,
                    project_id=stable_project_id,
                    external_reference_omissions=external_reference_omissions,
                    allow_include_recovery=not (
                        compile_db and filepath in compile_db
                    ),
                    parse_recovery_records=parse_recovery_records,
                )
                for func in functions:
                    index_obj.add_function(func)
                for td in typedefs:
                    index_obj.add_typedef(td)
            except SymbolIdentityError:
                raise
            except Exception as exc:
                fallback_path = _record_cpp_lexical_fallback(
                    filepath,
                    args,
                    project_dir,
                    exc,
                    parse_recovery_records,
                )
                if fallback_path is None:
                    cause = exc.__cause__ or exc
                    context = f"; {exc}" if cause is not exc else ""
                    raise RuntimeError(
                        f"C/C++ parse failed for {filepath}: "
                        f"{type(cause).__name__}: {cause}{context}"
                    ) from exc
                lexical_fallback_files.add(fallback_path)
            parsed += 1
            processed = parsed
            now = time.monotonic()
            if (
                processed == len(cpp_files)
                or (processed > 0 and processed % PARSE_HEARTBEAT_FILES == 0)
                or now - last_heartbeat >= PARSE_HEARTBEAT_SECONDS
            ):
                check_memory_limit(memory_limit_gb, label="index_project")
                print(
                    f"  Parsed {processed}/{len(cpp_files)} files, "
                    f"{len(index_obj.functions)} functions",
                    file=sys.stderr,
                    flush=True,
                )
                last_heartbeat = now
        print(f"  Parsed {parsed} files, "
              f"{len(index_obj.functions)} functions indexed", file=sys.stderr)

    external_reference_omission_summary = _external_reference_omission_summary(
        external_reference_omissions
    )
    parse_recovery_summary = _parse_recovery_summary(parse_recovery_records)
    print(
        "  Unknown external graph references omitted: "
        f"observations={external_reference_omission_summary['observation_count']} "
        f"unique={external_reference_omission_summary['unique_reference_count']} "
        "reference_records_sha256="
        f"{external_reference_omission_summary['reference_records_sha256']}",
        file=sys.stderr,
    )
    print(
        "  Source parse recovery: "
        f"attempted={parse_recovery_summary['attempted_file_count']} "
        f"recovered={parse_recovery_summary['recovered_file_count']} "
        f"unresolved={parse_recovery_summary['unresolved_file_count']} "
        f"records_sha256={parse_recovery_summary['records_sha256']}",
        file=sys.stderr,
    )
    if quarantine_receipt is not None:
        quarantine_receipt["external_reference_omissions"] = (
            external_reference_omission_summary
        )
        quarantine_receipt["parse_recovery"] = parse_recovery_summary
        assert source_quarantine_receipt is not None
        _write_source_quarantine_receipt(
            source_quarantine_receipt,
            quarantine_receipt,
        )

    # Build training documents (C/C++ code path -- unchanged).
    documents: list = []
    emitted_docs = 0

    def _emit_counted(doc: str | dict[str, object]) -> None:
        nonlocal emitted_docs
        if emit_doc is None:
            raise RuntimeError("_emit_counted called without emit_doc")
        emit_doc(doc)
        emitted_docs += 1

    if cpp_files:
        header_files = [
            path for path in cpp_files
            if os.path.splitext(path)[1].lower() in HEADER_EXTENSIONS
        ]
        macro_scan_files = select_macro_scan_files(
            cpp_files,
            index_obj,
            header_files,
            project_dir=project_dir,
        )
        print(
            "  Macro scan roots selected: "
            f"{len(macro_scan_files)}/{len(cpp_files)} C/C++ files "
            "(function/type/header roots)",
            file=sys.stderr,
        )
        documents = build_training_documents(
            index_obj,
            max_tokens,
            max_dep_depth,
            enriched=enriched,
            project_dir=project_dir,
            project_id=stable_project_id,
            compile_db=compile_db,
            default_args=default_args,
            default_build_info=default_build_info,
            tokenizer_path=tokenizer_path,
            dedup_db=dedup_db,
            dedup_stage_id=dedup_stage_id,
            dedup_stage_db=dedup_stage_db,
            dedup_near=dedup_near,
            global_symbols=global_symbols,
            header_files=header_files,
            macro_scan_files=macro_scan_files,
            emit_doc=_emit_counted if emit_doc is not None else None,
            memory_limit_gb=memory_limit_gb,
        )
        fallback_documents = emit_cpp_lexical_fallback_documents(
            lexical_fallback_files,
            index=index_obj,
            project_dir=project_dir,
            project_id=stable_project_id,
            compile_db=compile_db,
            default_args=default_args,
            default_build_info=default_build_info,
            parse_recovery_records=parse_recovery_records,
            enriched=enriched,
            emit_doc=_emit_counted if emit_doc is not None else None,
        )
        if emit_doc is None:
            documents.extend(fallback_documents)
    check_memory_limit(memory_limit_gb, label="index_project")
    code_doc_count = emitted_docs if emit_doc is not None else len(documents)
    print(f"  Generated {code_doc_count} code training documents", file=sys.stderr)

    # Emit build/compilation files as their own 'build' docs (ADDITIVE). Only in
    # enriched mode: build docs are dict-shaped (structure_ids/language_info/
    # platform_ids); plain-text mode would lose the build-system tag + sidecars.
    if enriched and domain_files:
        build_docs = emit_build_documents(
            domain_files,
            source_root=project_dir,
            project_id=stable_project_id,
            default_build_info=default_build_info,
            compile_index=_compile_index,
            tokenizer_path=tokenizer_path,
            dedup_db=dedup_db,
            dedup_stage_id=dedup_stage_id,
            dedup_stage_db=dedup_stage_db,
            dedup_near=dedup_near,
            emit_doc=_emit_counted if emit_doc is not None else None,
            skip_invalid_inputs=skip_invalid_domain_inputs,
        )
        if emit_doc is None:
            documents.extend(build_docs)
        check_memory_limit(memory_limit_gb, label="index_project")
    elif domain_files and not enriched:
        print(
            "  WARN: build/shell files found but --enriched not set; domain docs "
            "are enriched-only and were NOT emitted",
            file=sys.stderr,
        )

    total_doc_count = emitted_docs if emit_doc is not None else len(documents)
    print(f"  Generated {total_doc_count} total training documents", file=sys.stderr)

    stats = index_obj.stats()
    print(f"  Index stats: {stats}", file=sys.stderr)

    return documents


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Clang-based cross-file C++ dependency indexer')
    parser.add_argument('--project-dir', type=str,
                        help='Single project directory to process')
    parser.add_argument('--project-id', type=str,
                        help='Canonical owner/repo identity for --project-dir')
    parser.add_argument('--projects-list', type=str,
                        help='Tab-separated owner/repo and project directory per line')
    parser.add_argument('--projects-dir', type=str,
                        help='Directory containing multiple project subdirectories')
    parser.add_argument('--project-map', type=str,
                        help='JSON object mapping absolute project directories to owner/repo; '
                             'required with --projects-dir')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file path')
    parser.add_argument('--max-tokens', type=int, default=16384,
                        help='Max tokens per training document (default: 16384)')
    parser.add_argument('--max-dep-depth', type=int, default=5,
                        help='Max dependency resolution depth (default: 5)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers for multi-project mode')
    parser.add_argument('--parse-workers', type=int, default=8,
                        help='Number of parallel parse workers within each project (default: 8)')
    parser.add_argument('--libclang-path', type=str, default=None,
                        help='Path to libclang.so (auto-detected if not set)')
    parser.add_argument(
        '--macos-sdk',
        type=normalize_macos_sdk_path_argument,
        default=None,
        help='Explicit macOS SDK root used only for projects whose bounded '
             '.xcconfig evidence requires a macOS build context. No ambient '
             'SDK lookup is performed.',
    )
    parser.add_argument('--append', action='store_true',
                        help='Append to output file instead of overwriting')
    parser.add_argument('--enriched', action='store_true',
                        help='Emit enriched output with structure_ids, chunk_boundaries, '
                             'call_edges, type_edges alongside text')
    parser.add_argument('--exclude-dirs', type=str, default=None,
                        help='Comma-separated extra directory names to exclude from file discovery')
    parser.add_argument(
        '--skip-invalid-domain-inputs',
        action='store_true',
        help='Corpus mode: log and skip individual typed domain files containing '
             'NUL or malformed supported encodings instead of failing the entire '
             'project. The default remains fail-loud.',
    )
    parser.add_argument(
        '--source-quarantine-manifest',
        type=str,
        default=None,
        help='Versioned exact path+size+SHA manifest for proven non-C++ files '
             'stored under C/C++ suffixes. Requires --source-quarantine-receipt.',
    )
    parser.add_argument(
        '--source-quarantine-receipt',
        type=str,
        default=None,
        help='Atomic verification receipt for --source-quarantine-manifest. '
             'Single-project mode only.',
    )
    parser.add_argument('--memory-limit-gb', type=float, default=10.0,
                        help='Abort if this Python wrapper exceeds this max RSS in GiB (default: 10)')
    parser.add_argument('--dedup-db', type=str, default=None,
                        help='Path to the shared global dedup SQLite store. When set, '
                             'FUNCTION-LEVEL exact+near duplicates (on the tokenized '
                             'hash, BEFORE grouping) are dropped GLOBALLY across all '
                             'repos/streams (no in-RAM fallback). When absent, a '
                             'per-repo in-RAM exact set is used.')
    parser.add_argument('--dedup-stage-id', type=str, default=None,
                        help='Optional transactional dedup stage id. When set, '
                             'dedup claims are written only to staging tables; the '
                             'parent conveyor must promote after successful '
                             'materialize/pack/append or discard on failure.')
    parser.add_argument('--dedup-stage-db', type=str, default=None,
                        help='Optional local SQLite stage DB under rwork. When set '
                             'with --dedup-stage-id, this process writes staging '
                             'claims there while reading --dedup-db read-only; the '
                             'parent promotes the local stage DB after append '
                             'success.')
    parser.add_argument('--tokenizer-path', type=str, default=None,
                        help='Path to OUR tokenizer.json. REQUIRED to enable the '
                             'function-level tokenized-hash dedup (corrected design). '
                             'Without it, no function-level dedup runs.')
    parser.add_argument('--no-near-dedup', action='store_true',
                        help='Disable MinHash-LSH near dedup (exact-only).')
    parser.add_argument('--global-symbol-index', type=str, default=None,
                        help='Path to the GLOBAL cross-repo base-lib symbol '
                             'SQLite store (built by '
                             'scripts/crossrepo/build_global_symbol_index.py). '
                             'When set, an unresolved callee that is a selected '
                             'base-lib symbol has its definition PULLED in as a '
                             'deepest dependency chunk tagged crosslib:<repo>, '
                             'under hard per-doc bounds. DEFAULT off -> behavior '
                             'unchanged when absent.')

    args = parser.parse_args()
    if args.macos_sdk is not None:
        try:
            validate_macos_sdk_path(args.macos_sdk)
        except BuildContextEvidenceError as exc:
            parser.error(str(exc))
    start_memory_guard(args.memory_limit_gb, label="index_project")

    try:
        _configure_libclang(args.libclang_path)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Collect project directories together with their canonical corpus identity.
    project_specs: list[tuple[str, str]] = []
    if args.project_dir:
        if not args.project_id:
            parser.error("--project-dir requires --project-id owner/repo")
        project_specs.append(
            (
                os.path.abspath(args.project_dir),
                require_project_identity(args.project_id, source="--project-id"),
            )
        )
    elif args.projects_list:
        with open(args.projects_list) as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.rstrip("\n")
                if not line.strip():
                    continue
                fields = line.split("\t")
                if len(fields) != 2:
                    parser.error(
                        f"{args.projects_list}:{line_number}: expected "
                        "owner/repo<TAB>/project/path"
                    )
                project_id, project_dir = fields
                project_specs.append(
                    (
                        os.path.abspath(project_dir),
                        require_project_identity(
                            project_id,
                            source=f"{args.projects_list}:{line_number}",
                        ),
                    )
                )
    elif args.projects_dir:
        if not args.project_map:
            parser.error("--projects-dir requires --project-map")
        with open(args.project_map, encoding="utf-8") as project_map_file:
            raw_project_map = json.load(project_map_file)
        if not isinstance(raw_project_map, dict):
            parser.error("--project-map must contain a JSON object")
        project_map = {
            os.path.abspath(str(path)): require_project_identity(
                project_id,
                source=f"--project-map[{path!r}]",
            )
            for path, project_id in raw_project_map.items()
        }
        for entry in sorted(os.listdir(args.projects_dir)):
            full = os.path.abspath(os.path.join(args.projects_dir, entry))
            if os.path.isdir(full):
                project_id = project_map.get(full)
                if project_id is None:
                    parser.error(f"--project-map has no identity for {full}")
                project_specs.append((full, project_id))
    else:
        parser.error("Provide --project-dir, --projects-list, or --projects-dir")

    if (args.source_quarantine_manifest is None) != (
        args.source_quarantine_receipt is None
    ):
        parser.error(
            "--source-quarantine-manifest and --source-quarantine-receipt "
            "must be provided together"
        )
    if args.source_quarantine_manifest is not None and len(project_specs) != 1:
        parser.error(
            "source quarantine receipt binding currently requires exactly one project"
        )

    print(f"Processing {len(project_specs)} project(s)", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Max tokens: {args.max_tokens}", file=sys.stderr)
    print(f"Memory limit: {args.memory_limit_gb} GiB", file=sys.stderr)

    extra_exclude = set(args.exclude_dirs.split(',')) if args.exclude_dirs else None

    total_docs = 0
    # CORRECTED design: dedup is FUNCTION-LEVEL, on the tokenized hash, BEFORE
    # grouping -- it happens inside process_project -> build_training_documents ->
    # dedup_root_functions, backed by the shared global SQLite DedupStore at
    # --dedup-db (cross-repo + cross-stream, fail-loud). There is NO second-pass
    # doc-level dedup here. The per-process DedupStore opens against --dedup-db so
    # the multi-project ProcessPoolExecutor mode dedups globally via SQLite WAL.
    dedup_near = not args.no_near_dedup
    if args.dedup_db:
        if not args.tokenizer_path:
            print(
                "ERROR: --dedup-db requires --tokenizer-path (function-level "
                "dedup hashes the tokenized function ids).",
                file=sys.stderr,
            )
            return 1
        if args.dedup_stage_id and len(project_specs) != 1:
            print(
                "ERROR: --dedup-stage-id is only supported for one project per "
                "index_project process; parent pipeline must own promotion.",
                file=sys.stderr,
            )
            return 1
        if args.dedup_stage_db and not args.dedup_stage_id:
            print(
                "ERROR: --dedup-stage-db requires --dedup-stage-id.",
                file=sys.stderr,
            )
            return 1
        # FAIL LOUD up front: open once here so a bad db / missing datasketch
        # crashes before any heavy parsing (RULE #1). Closed immediately; each
        # process_project reopens against the same WAL db.
        DedupStore, _sha1_tokens = _import_dedup_store_symbols()
        if args.dedup_stage_id:
            if args.dedup_stage_db:
                # Local-stage subprocess path: global db is read-only even during
                # validation; stale cleanup touches only the local rwork stage db.
                DedupStore.discard_stage(
                    args.dedup_db,
                    args.dedup_stage_id,
                    stage_db_path=args.dedup_stage_db,
                )
                DedupStore(
                    args.dedup_db,
                    near=False,
                    commit_every=1000,
                    stage_id=args.dedup_stage_id,
                    stage_db_path=args.dedup_stage_db,
                ).close()
            else:
                # Backward-compatible same-DB staging path.
                DedupStore(args.dedup_db, near=False, commit_every=1000).close()
                DedupStore.discard_stage(args.dedup_db, args.dedup_stage_id)
        else:
            # Path/schema validation only. Avoid rebuilding persisted MinHash/LSH
            # in the parent before project workers start; process_project opens
            # the real store with near=dedup_near when requested.
            DedupStore(args.dedup_db, near=False, commit_every=1000).close()
        print(
            f"Dedup: {'STAGED' if args.dedup_stage_id else 'GLOBAL'} "
            f"function-level store at {args.dedup_db} "
            f"(exact{'+near' if dedup_near else ''}, tokenized hash"
            f"{', stage_id=' + args.dedup_stage_id if args.dedup_stage_id else ''}"
            f"{', stage_db=' + args.dedup_stage_db if args.dedup_stage_db else ''})",
            file=sys.stderr,
        )
    elif args.tokenizer_path:
        print("Dedup: per-repo in-RAM function-level exact set (no --dedup-db)",
              file=sys.stderr)
    else:
        print("Dedup: DISABLED (no --tokenizer-path given)", file=sys.stderr)

    # FAIL LOUD up front if --global-symbol-index is given but missing (RULE #1).
    global_symbol_index = args.global_symbol_index
    if global_symbol_index:
        if not os.path.exists(global_symbol_index):
            print(f"ERROR: --global-symbol-index not found: {global_symbol_index}",
                  file=sys.stderr)
            return 1
        # Open+close once now so a corrupt store crashes before heavy parsing.
        GlobalSymbolReader(global_symbol_index).close()
        print(f"Cross-lib: GLOBAL base-lib symbol index at {global_symbol_index} "
              f"(bounded depth-1 pulls, tagged crosslib:<repo>)", file=sys.stderr)
    else:
        print("Cross-lib: DISABLED (no --global-symbol-index)", file=sys.stderr)

    append_mode = getattr(args, 'append', False)
    enriched = args.enriched
    final_output = Path(args.output)
    with atomic_output_file(final_output) as staged_output:
        if append_mode and final_output.exists():
            shutil.copyfile(final_output, staged_output)
        with _open_jsonl_output(
            staged_output,
            append=append_mode,
            compressed=final_output.suffix == ".gz",
        ) as out:
            def _write_doc(doc: str | dict[str, object]) -> None:
                nonlocal total_docs
                if enriched:
                    json.dump(doc, out)
                else:
                    json.dump({'text': doc}, out)
                out.write('\n')
                total_docs += 1

            if args.workers > 1 and len(project_specs) > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(
                            process_project, pd, args.max_tokens, args.max_dep_depth,
                            args.parse_workers, enriched, extra_exclude,
                            args.memory_limit_gb, args.tokenizer_path, args.dedup_db,
                            args.dedup_stage_id, args.dedup_stage_db, dedup_near,
                            global_symbol_index,
                            project_id=project_id,
                            skip_invalid_domain_inputs=args.skip_invalid_domain_inputs,
                            source_quarantine_manifest=args.source_quarantine_manifest,
                            source_quarantine_receipt=args.source_quarantine_receipt,
                            macos_sdk_path=args.macos_sdk,
                        ): (pd, project_id)
                        for pd, project_id in project_specs
                    }
                    for future in as_completed(futures):
                        pd, project_id = futures[future]
                        try:
                            docs = future.result()
                            for doc in docs:
                                _write_doc(doc)
                        except SymbolIdentityError:
                            raise
                        except Exception as exc:
                            raise RuntimeError(
                                f"failed processing {project_id} ({pd}): {exc}"
                            ) from exc
            else:
                for pd, project_id in project_specs:
                    try:
                        docs = process_project(
                            pd, args.max_tokens, args.max_dep_depth,
                            args.parse_workers, enriched, extra_exclude,
                            args.memory_limit_gb, args.tokenizer_path,
                            args.dedup_db, args.dedup_stage_id,
                            args.dedup_stage_db, dedup_near,
                            global_symbol_index, emit_doc=_write_doc,
                            project_id=project_id,
                            skip_invalid_domain_inputs=args.skip_invalid_domain_inputs,
                            source_quarantine_manifest=args.source_quarantine_manifest,
                            source_quarantine_receipt=args.source_quarantine_receipt,
                            macos_sdk_path=args.macos_sdk,
                        )
                        for doc in docs:
                            _write_doc(doc)
                        out.flush()
                    except SymbolIdentityError:
                        raise
                    except Exception as exc:
                        raise RuntimeError(
                            f"failed processing {project_id} ({pd}): {exc}"
                        ) from exc

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Total documents: {total_docs}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
