#!/usr/bin/env python3
"""Build the GLOBAL cross-repo base-library symbol index (BuildIndex phase).

Goal
----
We selectively cross-repo link a small set of *base libraries* that are used in
many places across the corpus. When a function in repo X calls e.g. a
``boost::``/``absl::``/``std::`` symbol whose definition is NOT in repo X's own
index, the offline doc builder (``index_project.collect_transitive_deps``) can
consult THIS global symbol index and PULL the called base-lib function definition
in as a deepest-level dependency chunk (tagged ``dep_source='crosslib:<repo>'``).

This script BUILDS that index. It does NOT modify the corpus or the running
conveyor. It:

  1. Stream-extracts ONLY the selected base-lib subtrees from the corpus tarball
     (``zstd -dc --long=31 | tar`` with a member filter) into a bounded temp
     working dir, one base-lib at a time (resumable).
  2. Indexes each base-lib's PUBLIC / exported function + type DEFINITIONS via
     libclang (reusing ``index_project``'s proven parse helpers).
  3. Stores ``qualified_name -> {base_repo, file, text, kind, signature,
     is_public, token_estimate, ...}`` in a single SQLite store at
     ``outputs/crossrepo/global_symbols.sqlite``.

Selection (the careful cut — see outputs/crossrepo/base_lib_crosslink_candidates.md)
-----------------------------------------------------------------------------------
A1 (high-value, tractable; FULL public-API index):
    boost, abseil-cpp, folly, openssl, boringssl, protobuf, eigen, fmt, glib
A2 (huge / template-heavy; PUBLIC symbols only, report cost):
    std  via  STL, stl (microsoft/STL) + gcc-mirror (libstdc++) + llvm-project (libc++)
    libc via  glibc + musl

RULE #1 (fail loud, bounded)
----------------------------
ONE clear path per base-lib. A parse error on a single file is counted and the
file is skipped (parse failures in 3rd-party headers are EXPECTED and not a bug
in OUR pipeline), but a missing tarball, a libclang load failure, a bad subtree
name, or a SQLite open failure RAISES immediately. No silent degraded index.

Resumability
------------
``symbol_index_libs(lib TEXT PRIMARY KEY, done INT, n_funcs INT, n_types INT,
elapsed_s REAL, ...)`` records each completed base-lib. ``--resume`` (default)
skips libs already marked done. Per-lib extraction goes to a temp dir that is
deleted after the lib is indexed (bounded disk: ~1 base-lib of source at a time).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Locate the indexer module so we reuse its libclang parse helpers verbatim.
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
INDEXER_DIR = ROOT / "tools" / "clang_indexer"
if str(INDEXER_DIR) not in sys.path:
    sys.path.insert(0, str(INDEXER_DIR))

# These imports REQUIRE libclang to be importable; a failure RAISES (RULE #1).
import index_project as ip  # noqa: E402

DEFAULT_TARBALL = Path("/Users/dave/sources/parquet/data-cpp_all/data-cpp_all.tar.zst")
DEFAULT_OUTPUT = ROOT / "outputs" / "crossrepo" / "global_symbols.sqlite"

# --------------------------------------------------------------------------- #
# Selected base libraries -> their tarball subtree name(s) under cpp_all/<name>/.
# Subtree names are AUTHORITATIVE from the corpus census provider lists
# (outputs/crossrepo/base_lib_usage_ranked.json) — verified present in tarball.
#
# Each entry: lib-key -> dict(
#   subtrees=[<cpp_all subtree dir names>],
#   tier='A1'|'A2',
#   public_only=bool,           # A2 libs: only header-declared/exported symbols
#   namespace_prefixes=[...],   # symbol-namespace cleanliness (prefixed cross-link)
#   include_path_markers=[...], # (optional) RESTRICT extraction to members whose
#                               #   path contains one of these — for the huge
#                               #   gcc/llvm monorepos this pins the cut to the
#                               #   C++ stdlib header trees and OFF gcc's compiler
#                               #   / libiberty internals (the C2 root cause).
#   index_extensionless_headers=bool,  # (optional) also index extensionless std
#                               #   headers (<vector>, <type_traits>, ...).
# )
# A1 = tractable full public-API index. A2 = huge, public symbols only.
# --------------------------------------------------------------------------- #
# ``lang`` controls how headers (.h/.inc) are parsed: 'c++' (default for the C++
# libs — header-only template libs like boost/eigen/fmt MUST parse as C++ or the
# templated public API yields zero symbols) or 'c' (glib/openssl/libc are C).
_NON_PROVIDER_PATH_SEGMENTS = ("test", "tests", "testing", "benchmark", "benchmarks")
BASE_LIBS: dict[str, dict] = {
    # ---- A1 (high-value, tractable) ----
    "boost":      {"subtrees": ["boost"],        "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["boost::"],
                   # The normal project index intentionally treats boost:: as a
                   # system namespace and drops its type definitions.  This is
                   # the provider index, so keep those definitions exactly as we
                   # do for std::; the generic zero-type guard below then makes a
                   # broken Boost type index fail loud instead of being marked done.
                   "allow_system_types": True},
    "abseil":     {"subtrees": ["abseil-cpp"],   "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["absl::"],
                   # Unit tests and benchmarks are useful training code but are
                   # not providers for the cross-library public API graph. Some
                   # benchmark TUs instantiate the full flags stack and can take
                   # minutes / GiBs in libclang, so exclude them explicitly here
                   # rather than weakening the per-file timeout.
                   "index_exclude_suffixes": [
                       "_test.cc", "_test.cpp", "_test.cxx",
                       "_benchmark.cc", "_benchmark.cpp", "_benchmark.cxx",
                   ],
                   "index_exclude_path_segments": _NON_PROVIDER_PATH_SEGMENTS},
    "folly":      {"subtrees": ["folly"],        "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["folly::"],
                   "index_exclude_path_segments": _NON_PROVIDER_PATH_SEGMENTS},
    "openssl":    {"subtrees": ["openssl"],      "tier": "A1", "public_only": False, "lang": "c",   "namespace_prefixes": [],
                   "index_exclude_path_segments": _NON_PROVIDER_PATH_SEGMENTS},
    "boringssl":  {"subtrees": ["boringssl"],    "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": [],
                   "index_exclude_path_segments": _NON_PROVIDER_PATH_SEGMENTS},
    "protobuf":   {"subtrees": ["protobuf"],     "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["google::protobuf::"],
                   "index_exclude_path_segments": _NON_PROVIDER_PATH_SEGMENTS},
    "eigen":      {"subtrees": ["eigen"],        "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["Eigen::"],
                   "index_exclude_path_segments": _NON_PROVIDER_PATH_SEGMENTS},
    "fmt":        {"subtrees": ["fmt"],          "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["fmt::"],
                   "index_exclude_path_segments": _NON_PROVIDER_PATH_SEGMENTS},
    "glib":       {"subtrees": ["glib"],         "tier": "A1", "public_only": False, "lang": "c",   "namespace_prefixes": [],
                   "index_exclude_path_segments": _NON_PROVIDER_PATH_SEGMENTS},
    # ---- A2 (huge / template-heavy) — PUBLIC symbols only ----
    # std: gcc-mirror and llvm-project are FULL compiler monorepos (the entire GCC
    # / LLVM source trees). Without include_path_markers the per-lib file cap is
    # spent on gcc's compiler + libiberty internals and ZERO real C++ standard
    # library headers get indexed (the C2 defect). Pin the cut to the libstdc++ /
    # libc++ / microsoft-STL public header trees only, and index their
    # EXTENSIONLESS public headers (<vector>, <type_traits>, ...).
    # The std umbrella headers DO NOT parse with the bare A1 args — they need a
    # real C++ stdlib compile env (clang builtins + a C sysroot + >= c++20) AND
    # the parser must be told to KEEP std:: type names (allow_system_types), or
    # libclang yields only a few free functions and ZERO class templates (the old
    # "std=202 funcs/0 types" defect). See _cxx_stdlib_sysroot_args / the std
    # experiment in build notes: with this env libc++ <vector> yields std::vector
    # + std::vector::push_back (and every container's public members).
    #
    # PRIMARY target = libc++ (llvm-project/libcxx/include): self-contained
    # (ships its own __config), needs ONLY clang builtins + a C lib -> it parses
    # and supplies the std:: surface.
    # libstdc++ (gcc-mirror/libstdc++-v3/include): its <bits/c++config.h> is
    # GENERATED by configure and is NOT in the source tree, so most libstdc++
    # umbrella headers fail to parse standalone here -> they are BEST-EFFORT
    # (per-file parse errors, counted in the report, never silently dropped). We
    # deliberately do NOT synthesize a c++config.h: libc++ already provides the
    # full std:: surface and dedup (INSERT OR IGNORE) merges any overlap.
    # MS-STL (microsoft/STL/stl/inc): needs MSVC intrinsics -> BEST-EFFORT via
    # -fms-compatibility (msvc_path_marker); headers needing genuine MSVC
    # builtins fail per-file (counted), not silently skipped.
    "std":        {"subtrees": ["STL", "stl", "gcc-mirror", "llvm-project"],
                   "tier": "A2", "public_only": True, "lang": "c++",
                   "namespace_prefixes": ["std::"],
                   "include_path_markers": [
                       "/libstdc++-v3/include/",    # gcc-mirror: libstdc++ public headers
                       "/libstdc++-v3/libsupc++/",  # gcc-mirror: <typeinfo>/<exception>/<new>
                       "/libcxx/include/",          # llvm-project: libc++ public headers
                       "/stl/inc/",                 # microsoft/STL public headers
                   ],
                   "index_extensionless_headers": True,
                   # ---- per-lib C++ standard-library compile environment ----
                   # cxx_std is chosen at BUILD TIME by an explicit probe (see
                   # _probe_highest_cxx_std): we try the candidates below highest
                   # first and pick the highest one libclang ACTUALLY accepts on a
                   # real libc++ <ranges>/<format> parse. A SUPERSET std captures
                   # the c++23/26 standard surface (ranges/concepts/format/expected/
                   # flat_map). The static "cxx_std" is the documented FLOOR used
                   # only if no probe candidates are configured; the probe (which
                   # includes c++23 as its lowest rung) RAISES rather than silently
                   # dropping below c++23 (RULE #1).
                   "cxx_std": "-std=c++23",          # floor; probe bumps to c++26/2c when accepted
                   "cxx_std_probe_candidates": ("-std=c++26", "-std=c++2c", "-std=c++23"),
                   "needs_cxx_stdlib_env": True,      # inject clang builtins + C sysroot
                   "allow_system_types": True,        # KEEP std:: TYPE definitions
                   "stdlib_include_markers": [        # per-FILE -I root (no cross-tree mix)
                       "/libcxx/include/",
                       "/libstdc++-v3/include/",
                       "/libstdc++-v3/libsupc++/",
                       "/stl/inc/",
                   ],
                   "msvc_path_marker": "/stl/inc/"},  # MS-STL best-effort MSVC compat
    # libc: C libraries have NO namespace; keep by-name public symbols (no prefix
    # filter). glibc/musl public surface lives in normal .h headers.
    "libc":       {"subtrees": ["glibc", "musl"],
                   "tier": "A2", "public_only": True, "lang": "c", "namespace_prefixes": [],
                   "index_public_headers_only": True,
                   "index_exclude_path_segments": _NON_PROVIDER_PATH_SEGMENTS},
}

# Corpus subtree names are extraction details, not project identities. Keep the
# authoritative owner/repository identity explicit so every worker emits the
# same canonical symbol key regardless of its temporary extraction directory.
PROJECT_ID_BY_SUBTREE = {
    "boost": "boostorg/boost",
    "abseil-cpp": "abseil/abseil-cpp",
    "folly": "facebook/folly",
    "openssl": "openssl/openssl",
    "boringssl": "google/boringssl",
    "protobuf": "protocolbuffers/protobuf",
    "eigen": "libeigen/eigen",
    "fmt": "fmtlib/fmt",
    "glib": "GNOME/glib",
    "STL": "microsoft/STL",
    "stl": "microsoft/STL",
    "gcc-mirror": "gcc-mirror/gcc",
    "llvm-project": "llvm/llvm-project",
    "glibc": "bminor/glibc",
    "musl": "ifduyue/musl",
}


def project_identity_for_subtree(subtree: str) -> str:
    try:
        project_id = PROJECT_ID_BY_SUBTREE[subtree]
    except KeyError as exc:
        raise ip.SymbolIdentityError(
            f"no canonical owner/repo identity configured for subtree {subtree!r}"
        ) from exc
    return ip.require_project_identity(
        project_id, source=f"global symbol subtree {subtree}"
    )

A1_LIBS = [k for k, v in BASE_LIBS.items() if v["tier"] == "A1"]
A2_LIBS = [k for k, v in BASE_LIBS.items() if v["tier"] == "A2"]

# Per-lib hard bounds (BuildIndex cost containment; RULE #1: bounded, no explosion)
MAX_FILES_PER_LIB_DEFAULT = 60_000       # cap files indexed per base-lib
MAX_BYTES_PER_FILE = 500_000             # mirror index_project's per-file cap
PARSE_TIMEOUT_S = 60                      # per-file libclang timeout
PROGRESS_EVERY = 2000
SYMBOL_BATCH_SIZE = 5000

# A2 std/libc: index ONLY symbols declared in headers (public surface). For std
# the public headers are the libstdc++ <bits/...>/<...> includes + libc++'s; for
# libc the installed headers. We treat any file under an "include"-ish path or a
# header extension as public-API-bearing. Bodies of huge template impls in .tcc
# files are still indexed when public_only and they declare exported symbols.


# --------------------------------------------------------------------------- #
# Public-symbol classification.
# --------------------------------------------------------------------------- #
_HEADER_EXTS = {".h", ".hpp", ".hxx", ".hh", ".h++", ".inl", ".inc", ".ipp", ".tcc"}

# Tokens that mark a symbol as INTERNAL/private — never part of the public API.
# Conservative: a leading underscore-uppercase or a "detail"/"internal"/"impl"
# namespace segment in the qualified name.
_INTERNAL_QNAME_SEG = re.compile(
    r"(^|::)(detail|internal|impl|_internal|__detail|__internal)(::|$)", re.IGNORECASE
)
_RESERVED_LEADING = re.compile(r"(^|::)(_[A-Z]|__)")  # __foo / _Foo reserved names
def normalize_inline_namespace_qname(qname: str) -> str:
    """Use the indexer's exact, std-only inline-namespace normalization."""
    return ip.normalize_inline_namespace_qname(qname)


def _is_public_header_path(rel_file: str) -> bool:
    ext = os.path.splitext(rel_file)[1].lower()
    if ext in _HEADER_EXTS:
        return True
    parts = [part for part in rel_file.replace("\\", "/").split("/") if part]
    return "include" in parts


def _filter_non_provider_sources(
    paths: Sequence[str], spec: Mapping[str, object]
) -> tuple[list[str], int]:
    suffixes = tuple(
        str(value).lower() for value in spec.get("index_exclude_suffixes", ())
    )
    excluded_segments = {
        str(value).lower()
        for value in spec.get("index_exclude_path_segments", ())
    }

    def keep(path: str) -> bool:
        normalized = path.replace("\\", "/")
        if suffixes and normalized.lower().endswith(suffixes):
            return False
        parts = {part.lower() for part in normalized.split("/") if part}
        if excluded_segments.intersection(parts):
            return False
        if spec.get("index_public_headers_only") and not _is_public_header_path(
            normalized
        ):
            return False
        return True

    kept = [path for path in paths if keep(path)]
    return kept, len(paths) - len(kept)


def is_public_symbol(
    qname: str,
    rel_file: str,
    public_only: bool,
    namespace_prefixes: Sequence[str] = (),
) -> bool:
    """Decide whether a parsed symbol belongs in the cross-link index.

    For A1 libs (public_only=False) we accept any non-internal qualified name
    that belongs to the lib's configured namespace prefixes, when provided.
    For A2 libs (public_only=True) we additionally REQUIRE the defining file to
    be a header (public API surface) and reject reserved/internal names — this
    bounds the std/libc index to the public symbol surface and keeps cost down.
    """
    qname = normalize_inline_namespace_qname(qname)
    if not qname:
        return False
    if namespace_prefixes and not any(qname.startswith(prefix) for prefix in namespace_prefixes):
        return False
    if _INTERNAL_QNAME_SEG.search(qname):
        return False
    if public_only:
        if not _is_public_header_path(rel_file):
            return False
        # Reserved-name implementation symbols (__copy, _M_..., _S_...) are not
        # the public API the model should learn; drop them for the A2 surface.
        if _RESERVED_LEADING.search(qname):
            return False
    return True


# --------------------------------------------------------------------------- #
# Trivial / tiny inline std body filter (skip — too small to be worth a chunk).
# --------------------------------------------------------------------------- #
def normalized_body_len(text: str) -> int:
    s = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    s = re.sub(r"//[^\n]*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return len(s)


# --------------------------------------------------------------------------- #
# SQLite global symbol store.
# --------------------------------------------------------------------------- #
GLOBAL_SYMBOL_DB_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class GlobalSymbolRecord:
    qname: str
    base_lib: str
    base_repo: str
    kind: int
    sym_type: str
    file: str
    line: int
    end_line: int
    is_public: int
    token_est: int
    body_len: int
    text: str
    symbol_key: str
    symbol_id: int | None = None
    usr: str = ""
    canonical_signature: str = ""
    symbol_kind: str = ""
    provider: str = ""
    include_provenance: str = ""
    identity_schema_version: int = ip.SYMBOL_IDENTITY_SCHEMA_VERSION


SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol_uid  TEXT NOT NULL PRIMARY KEY,
    qname        TEXT NOT NULL,
    base_lib     TEXT NOT NULL,   -- logical lib key (boost/std/libc/...)
    base_repo    TEXT NOT NULL,   -- the cpp_all subtree the def came from
    kind         INTEGER NOT NULL,-- 2=function, 4=record/enum, 7=typedef/using
    sym_type     TEXT NOT NULL,   -- 'func' | 'type'
    file         TEXT NOT NULL,
    line         INTEGER NOT NULL,
    end_line     INTEGER NOT NULL,
    is_public    INTEGER NOT NULL,
    token_est    INTEGER NOT NULL,
    body_len     INTEGER NOT NULL,
    text         TEXT NOT NULL,
    symbol_key   TEXT NOT NULL,
    symbol_id    TEXT NOT NULL,
    usr          TEXT NOT NULL,
    canonical_signature TEXT NOT NULL,
    symbol_kind  TEXT NOT NULL,
    identity_schema_version INTEGER NOT NULL,
    provider      TEXT NOT NULL,
    include_provenance TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_symbols_qname ON symbols(qname);
CREATE INDEX IF NOT EXISTS idx_symbols_qname_type ON symbols(qname, sym_type);
CREATE INDEX IF NOT EXISTS idx_symbols_lib   ON symbols(base_lib);
CREATE INDEX IF NOT EXISTS idx_symbols_symbol_key ON symbols(symbol_key);
CREATE INDEX IF NOT EXISTS idx_symbols_usr ON symbols(usr);
CREATE INDEX IF NOT EXISTS idx_symbols_qname_signature
    ON symbols(qname, canonical_signature, symbol_kind);
CREATE INDEX IF NOT EXISTS idx_symbols_provenance
    ON symbols(qname, provider, include_provenance);

CREATE TABLE IF NOT EXISTS symbol_identities (
    symbol_id TEXT NOT NULL PRIMARY KEY,
    symbol_key TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS symbol_index_libs (
    lib       TEXT PRIMARY KEY,
    tier      TEXT NOT NULL,
    done      INTEGER NOT NULL DEFAULT 0,
    n_files   INTEGER NOT NULL DEFAULT 0,
    n_funcs   INTEGER NOT NULL DEFAULT 0,
    n_types   INTEGER NOT NULL DEFAULT 0,
    n_errors  INTEGER NOT NULL DEFAULT 0,
    elapsed_s REAL    NOT NULL DEFAULT 0,
    subtrees  TEXT    NOT NULL DEFAULT '',
    finished_utc TEXT NOT NULL DEFAULT ''
);
"""


class GlobalSymbolStore:
    """Resumable SQLite store for cross-repo base-lib symbol definitions.

    The read path returns all display-qname candidates and resolves one only by
    canonical identity fields or proven uniqueness. The write path is used only
    by this builder.
    """

    def __init__(self, path: str, read_only: bool = False):
        self.path = path
        if read_only:
            uri = f"file:{path}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path, timeout=60.0)
        try:
            if read_only:
                self._require_current_schema()
            else:
                self.conn.execute("PRAGMA journal_mode=WAL;")
                self.conn.execute("PRAGMA synchronous=NORMAL;")
                self._initialize_or_migrate_schema()
                self._require_current_schema()
                self.conn.commit()
        except BaseException:
            self.conn.close()
            raise

    def _initialize_or_migrate_schema(self) -> None:
        version = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
        if version > GLOBAL_SYMBOL_DB_SCHEMA_VERSION:
            raise ip.SymbolIdentityError(
                f"{self.path}: newer global symbol schema cannot be downgraded: "
                f"user_version={version}, supported={GLOBAL_SYMBOL_DB_SCHEMA_VERSION}"
            )
        table_exists = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='symbols'"
        ).fetchone()
        if table_exists is None:
            self.conn.executescript(
                "BEGIN IMMEDIATE;\n"
                + SCHEMA
                + f"\nPRAGMA user_version={GLOBAL_SYMBOL_DB_SCHEMA_VERSION};\nCOMMIT;"
            )
            return

        cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(symbols)").fetchall()
        }
        identity_columns = {
            "symbol_uid",
            "qname",
            "base_lib",
            "base_repo",
            "file",
            "symbol_key",
            "symbol_id",
            "usr",
            "canonical_signature",
            "symbol_kind",
            "identity_schema_version",
        }
        missing_identity = sorted(identity_columns - cols)
        if missing_identity:
            raise ip.SymbolIdentityError(
                f"{self.path}: legacy global symbol schema cannot be migrated "
                "without canonical identity fields; "
                f"missing_columns={missing_identity}. Rebuild the index."
            )

        has_provider = "provider" in cols
        has_include = "include_provenance" in cols
        select_tail = (
            ", provider" if has_provider else ", '' AS provider"
        ) + (
            ", include_provenance"
            if has_include
            else ", '' AS include_provenance"
        )
        rows = self.conn.execute(
            "SELECT symbol_uid, qname, base_lib, base_repo, file, line, symbol_key, "
            "symbol_id, usr, canonical_signature, symbol_kind, "
            "identity_schema_version" + select_tail + " FROM symbols"
        ).fetchall()
        provenance_updates: list[tuple[str, str, str]] = []
        for row in rows:
            (
                symbol_uid,
                qname,
                base_lib,
                base_repo,
                file,
                line,
                symbol_key,
                symbol_id_hex,
                usr,
                canonical_signature,
                symbol_kind,
                identity_version,
                stored_provider,
                stored_include,
            ) = row
            project_id = ip.require_project_identity(
                base_repo, source=f"{self.path}:{symbol_uid}"
            )
            if int(identity_version) != ip.SYMBOL_IDENTITY_SCHEMA_VERSION:
                raise ip.SymbolIdentityError(
                    f"{self.path}:{symbol_uid}: identity schema v{identity_version} "
                    f"cannot be promoted to v{ip.SYMBOL_IDENTITY_SCHEMA_VERSION}"
                )
            has_location_identity = bool(symbol_key) and ip.is_repo_file_location_identity(
                str(symbol_key)
            )
            if (
                not symbol_key
                or not symbol_kind
                or (not usr and not canonical_signature and not has_location_identity)
            ):
                raise ip.SymbolIdentityError(
                    f"{self.path}:{symbol_uid}: canonical identity cannot be "
                    "reconstructed; require symbol_key, symbol_kind, and USR, "
                    "signature, or explicit repository location"
                )
            if str(canonical_signature).startswith("legacy-kind="):
                raise ip.SymbolIdentityError(
                    f"{self.path}:{symbol_uid}: synthetic legacy signature cannot "
                    "establish canonical clang identity; rebuild the index"
                )
            if has_location_identity:
                ip.validate_repo_file_location_identity_claim(
                    str(symbol_key),
                    project=project_id,
                    file=str(file),
                    line=int(line),
                    kind=str(symbol_kind),
                    qname=str(qname),
                    source=f"{self.path}:{symbol_uid}",
                )
            else:
                identity_kwargs = {
                    "qname": str(qname),
                    "kind": str(symbol_kind),
                    "usr": str(usr or ""),
                    "canonical_signature": str(canonical_signature or ""),
                    "project": project_id,
                    "file": str(file),
                    "line": int(line),
                }
                expected_keys = {
                    ip.canonical_symbol_identity(**identity_kwargs),
                    ip.canonical_symbol_identity(
                        **identity_kwargs, force_file_scope=True
                    ),
                }
                if str(symbol_key) not in expected_keys:
                    raise ip.SymbolIdentityError(
                        f"{self.path}:{symbol_uid}: stored canonical key does not match "
                        "its identity fields; rebuild the index"
                    )
            expected_hex = self._symbol_id_hex(ip._compute_symbol_id(str(symbol_key)))
            if str(symbol_id_hex) != expected_hex:
                raise ip.SymbolIdentityError(
                    f"{self.path}:{symbol_uid}: stored symbol_id {symbol_id_hex!r} "
                    f"does not match canonical key ({expected_hex})"
                )
            detected_provider, detected_include = ip.symbol_provider_provenance(str(file))
            provider = str(stored_provider or detected_provider or project_id)
            include_provenance = str(stored_include or detected_include or file)
            if base_lib == "std" and (not detected_provider or not detected_include):
                raise ip.SymbolIdentityError(
                    f"{self.path}:{symbol_uid}: std provider/include provenance "
                    f"cannot be reconstructed from {file!r}; rebuild the index"
                )
            if stored_provider and detected_provider and stored_provider != detected_provider:
                raise ip.SymbolIdentityError(
                    f"{self.path}:{symbol_uid}: provider provenance mismatch: "
                    f"stored={stored_provider!r} detected={detected_provider!r}"
                )
            if stored_include and detected_include and stored_include != detected_include:
                raise ip.SymbolIdentityError(
                    f"{self.path}:{symbol_uid}: include provenance mismatch: "
                    f"stored={stored_include!r} detected={detected_include!r}"
                )
            provenance_updates.append((provider, include_provenance, str(symbol_uid)))

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            if not has_provider:
                self.conn.execute(
                    "ALTER TABLE symbols ADD COLUMN provider TEXT NOT NULL DEFAULT ''"
                )
            if not has_include:
                self.conn.execute(
                    "ALTER TABLE symbols ADD COLUMN include_provenance "
                    "TEXT NOT NULL DEFAULT ''"
                )
            self.conn.executemany(
                "UPDATE symbols SET provider=?, include_provenance=? WHERE symbol_uid=?",
                provenance_updates,
            )
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS symbol_identities ("
                "symbol_id TEXT NOT NULL PRIMARY KEY, "
                "symbol_key TEXT NOT NULL UNIQUE)"
            )
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS symbol_index_libs ("
                "lib TEXT PRIMARY KEY, tier TEXT NOT NULL, "
                "done INTEGER NOT NULL DEFAULT 0, n_files INTEGER NOT NULL DEFAULT 0, "
                "n_funcs INTEGER NOT NULL DEFAULT 0, n_types INTEGER NOT NULL DEFAULT 0, "
                "n_errors INTEGER NOT NULL DEFAULT 0, elapsed_s REAL NOT NULL DEFAULT 0, "
                "subtrees TEXT NOT NULL DEFAULT '', finished_utc TEXT NOT NULL DEFAULT '')"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_symbol_key ON symbols(symbol_key)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_usr ON symbols(usr)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_qname_signature "
                "ON symbols(qname, canonical_signature, symbol_kind)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_provenance "
                "ON symbols(qname, provider, include_provenance)"
            )
            self._rebuild_symbol_identity_registry()
            self.conn.execute(f"PRAGMA user_version={GLOBAL_SYMBOL_DB_SCHEMA_VERSION}")
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def _require_current_schema(self) -> None:
        cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(symbols)").fetchall()
        }
        if "symbol_uid" not in cols:
            raise ip.SymbolIdentityError(
                f"{self.path}: global symbol index uses old qname-only schema; "
                "delete/rebuild it with scripts/crossrepo/build_global_symbol_index.py"
            )
        required = {
            "symbol_key",
            "symbol_id",
            "usr",
            "canonical_signature",
            "symbol_kind",
            "identity_schema_version",
            "provider",
            "include_provenance",
        }
        missing = sorted(required - cols)
        version = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
        if missing or version != GLOBAL_SYMBOL_DB_SCHEMA_VERSION:
            raise ip.SymbolIdentityError(
                f"{self.path}: incompatible global symbol schema: "
                f"user_version={version}, missing_columns={missing}"
            )
        incompatible_rows = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM symbols "
                "WHERE identity_schema_version!=? OR symbol_key='' OR symbol_id='' "
                "OR symbol_kind='' OR (usr='' AND canonical_signature='' "
                "AND symbol_key NOT LIKE 'repo_file_location:%') "
                "OR canonical_signature LIKE 'legacy-kind=%text-sha1=%' "
                "OR provider='' OR include_provenance=''",
                (ip.SYMBOL_IDENTITY_SCHEMA_VERSION,),
            ).fetchone()[0]
        )
        if incompatible_rows:
            raise ip.SymbolIdentityError(
                f"{self.path}: incompatible symbol identity rows: "
                f"count={incompatible_rows}, expected_version="
                f"{ip.SYMBOL_IDENTITY_SCHEMA_VERSION}"
            )
        for row in self.conn.execute(
            "SELECT symbol_uid, symbol_key, qname, base_repo, file, line, symbol_kind "
            "FROM symbols WHERE usr='' AND canonical_signature=''"
        ):
            symbol_uid, symbol_key, qname, project_id, file, line, symbol_kind = row
            ip.validate_repo_file_location_identity_claim(
                str(symbol_key),
                project=str(project_id),
                file=str(file),
                line=int(line),
                kind=str(symbol_kind),
                qname=str(qname),
                source=f"{self.path}:{symbol_uid}",
            )
        for (project_id,) in self.conn.execute(
            "SELECT DISTINCT base_repo FROM symbols"
        ):
            ip.require_project_identity(
                project_id, source=f"global symbol index {self.path}"
            )
        missing_std_provenance = self.conn.execute(
            "SELECT symbol_uid, file FROM symbols WHERE base_lib='std' "
            "AND (provider NOT IN ('libc++', 'libstdc++', 'msvc-stl') "
            "OR include_provenance='') LIMIT 1"
        ).fetchone()
        if missing_std_provenance is not None:
            raise ip.SymbolIdentityError(
                f"{self.path}: std symbol lacks authoritative provider/include "
                f"provenance: uid={missing_std_provenance[0]} "
                f"file={missing_std_provenance[1]!r}"
            )
        registry_table = self.conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='symbol_identities'"
        ).fetchone()
        if registry_table is None:
            raise ip.SymbolIdentityError(
                f"{self.path}: global symbol schema has no corpus collision registry"
            )
        unregistered_rows = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM symbols AS s "
                "LEFT JOIN symbol_identities AS i "
                "ON i.symbol_id=s.symbol_id AND i.symbol_key=s.symbol_key "
                "WHERE i.symbol_id IS NULL"
            ).fetchone()[0]
        )
        if unregistered_rows:
            raise ip.SymbolIdentityError(
                f"{self.path}: global symbol index has unregistered symbol IDs: "
                f"count={unregistered_rows}"
            )

    @staticmethod
    def _symbol_id_hex(symbol_id: int) -> str:
        value = int(symbol_id)
        if not 0 < value <= ip.SYMBOL_ID_MAX:
            raise ip.SymbolIdentityError(
                f"symbol_id is outside unsigned 64-bit range: {value}"
            )
        return f"{value:016x}"

    def _rebuild_symbol_identity_registry(self) -> None:
        self.conn.execute("DELETE FROM symbol_identities")
        rows = self.conn.execute(
            "SELECT symbol_uid, symbol_key, symbol_id FROM symbols "
            "ORDER BY symbol_uid"
        )
        for symbol_uid, symbol_key, symbol_id_hex in rows:
            expected_id = ip._compute_symbol_id(str(symbol_key))
            expected_hex = self._symbol_id_hex(expected_id)
            if str(symbol_id_hex) != expected_hex:
                raise ip.SymbolIdentityError(
                    f"{self.path}: symbol {symbol_uid} has ID {symbol_id_hex!r}, "
                    f"expected {expected_hex} for {symbol_key!r}"
                )
            existing = self.conn.execute(
                "SELECT symbol_key FROM symbol_identities WHERE symbol_id=?",
                (expected_hex,),
            ).fetchone()
            if existing is not None and str(existing[0]) != str(symbol_key):
                raise ip.SymbolIdentityError(
                    "canonical symbol ID collision while rebuilding global index: "
                    f"id={expected_id} first={existing[0]!r} second={symbol_key!r}"
                )
            self.conn.execute(
                "INSERT OR IGNORE INTO symbol_identities(symbol_id, symbol_key) "
                "VALUES (?, ?)",
                (expected_hex, str(symbol_key)),
            )

    def _validate_symbol_identity_claims(
        self, records: Sequence[GlobalSymbolRecord]
    ) -> list[int]:
        staged_by_id: dict[str, str] = {}
        staged_by_key: dict[str, str] = {}
        symbol_ids: list[int] = []
        for record in records:
            project_id = ip.require_project_identity(
                record.base_repo,
                source=f"global symbol {record.base_repo}:{record.file}:{record.line}",
            )
            if not record.symbol_key or not record.symbol_kind:
                raise ip.SymbolIdentityError(
                    f"{project_id}:{record.file}:{record.line}: missing canonical identity"
                )
            has_location_identity = ip.is_repo_file_location_identity(
                record.symbol_key
            )
            if (
                not record.usr
                and not record.canonical_signature
                and not has_location_identity
            ):
                raise ip.SymbolIdentityError(
                    f"{project_id}:{record.file}:{record.line}: identity requires USR "
                    "canonical signature, or explicit repository location"
                )
            if has_location_identity:
                ip.validate_repo_file_location_identity_claim(
                    record.symbol_key,
                    project=project_id,
                    file=record.file,
                    line=record.line,
                    kind=record.symbol_kind,
                    qname=record.qname,
                    source=f"{project_id}:{record.file}:{record.line}",
                )
            else:
                identity_kwargs = {
                    "qname": record.qname,
                    "kind": record.symbol_kind,
                    "usr": record.usr,
                    "canonical_signature": record.canonical_signature,
                    "project": project_id,
                    "file": record.file,
                    "line": record.line,
                }
                expected_keys = {
                    ip.canonical_symbol_identity(**identity_kwargs),
                    ip.canonical_symbol_identity(
                        **identity_kwargs, force_file_scope=True
                    ),
                }
                if record.symbol_key not in expected_keys:
                    raise ip.SymbolIdentityError(
                        f"{project_id}:{record.file}:{record.line}: canonical key does "
                        "not match the record identity fields"
                    )
            if not record.provider or not record.include_provenance:
                raise ip.SymbolIdentityError(
                    f"{project_id}:{record.file}:{record.line}: provider and include "
                    "provenance are required"
                )
            claimed_id = (
                ip._compute_symbol_id(record.symbol_key)
                if record.symbol_id is None
                else int(record.symbol_id)
            )
            claimed_hex = self._symbol_id_hex(claimed_id)
            source = f"{record.base_repo}:{record.file}:{record.line}"

            existing_key = staged_by_id.get(claimed_hex)
            if existing_key is None:
                row = self.conn.execute(
                    "SELECT symbol_key FROM symbol_identities WHERE symbol_id=?",
                    (claimed_hex,),
                ).fetchone()
                existing_key = None if row is None else str(row[0])
            if existing_key is not None and existing_key != record.symbol_key:
                raise ip.SymbolIdentityError(
                    "canonical symbol ID collision in global index: "
                    f"id={claimed_id} first={existing_key!r} "
                    f"second={record.symbol_key!r} ({source})"
                )

            existing_id = staged_by_key.get(record.symbol_key)
            if existing_id is None:
                row = self.conn.execute(
                    "SELECT symbol_id FROM symbol_identities WHERE symbol_key=?",
                    (record.symbol_key,),
                ).fetchone()
                existing_id = None if row is None else str(row[0])
            if existing_id is not None and existing_id != claimed_hex:
                raise ip.SymbolIdentityError(
                    f"{source}: canonical key {record.symbol_key!r} is already "
                    f"registered as ID {int(existing_id, 16)}, not {claimed_id}"
                )

            expected_id = ip._compute_symbol_id(record.symbol_key)
            if claimed_id != expected_id:
                raise ip.SymbolIdentityError(
                    f"{source}: symbol_id {claimed_id} does not match v"
                    f"{ip.SYMBOL_IDENTITY_SCHEMA_VERSION} ID {expected_id} for "
                    f"{record.symbol_key!r}"
                )
            if record.identity_schema_version != ip.SYMBOL_IDENTITY_SCHEMA_VERSION:
                raise ip.SymbolIdentityError(
                    f"{source}: symbol identity schema v{record.identity_schema_version} "
                    f"is incompatible with v{ip.SYMBOL_IDENTITY_SCHEMA_VERSION}"
                )
            staged_by_id[claimed_hex] = record.symbol_key
            staged_by_key[record.symbol_key] = claimed_hex
            symbol_ids.append(claimed_id)
        return symbol_ids

    @staticmethod
    def _symbol_uid(record: GlobalSymbolRecord) -> str:
        payload = "\x1f".join(
            str(part)
            for part in (
                record.base_lib,
                record.base_repo,
                record.sym_type,
                record.symbol_key,
                record.file,
                record.line,
                record.end_line,
                hashlib.sha1(record.text.encode("utf-8")).hexdigest(),
            )
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    # ---- write path (builder) ----
    def _delete_orphaned_identities(self) -> None:
        self.conn.execute(
            "DELETE FROM symbol_identities WHERE NOT EXISTS ("
            "SELECT 1 FROM symbols "
            "WHERE symbols.symbol_id=symbol_identities.symbol_id "
            "AND symbols.symbol_key=symbol_identities.symbol_key)"
        )

    def invalidate_lib(
        self,
        lib: str,
        tier: str,
        subtrees: Sequence[str],
    ) -> None:
        """Durably remove a stale generation before attempting a rebuild."""
        if self.conn.in_transaction:
            raise RuntimeError(
                f"[{lib}] cannot invalidate a library inside an open transaction"
            )
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("DELETE FROM symbols WHERE base_lib=?", (lib,))
            self._delete_orphaned_identities()
            self.conn.execute(
                "INSERT OR REPLACE INTO symbol_index_libs "
                "(lib, tier, done, n_files, n_funcs, n_types, n_errors, elapsed_s, "
                " subtrees, finished_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (lib, tier, 0, 0, 0, 0, 0, 0.0, ",".join(subtrees), ""),
            )
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    @contextmanager
    def rebuild_lib(
        self,
        lib: str,
        tier: str,
        subtrees: Sequence[str],
    ) -> Iterator[None]:
        """Publish one library as one transaction, with durable invalidation.

        The old completed generation is invalidated before parsing starts. A
        crash or parse failure therefore cannot leave stale ``done=1`` state,
        and every symbol inserted during the attempted rebuild rolls back.
        """
        if self.conn.in_transaction:
            raise RuntimeError(
                f"[{lib}] cannot start atomic rebuild inside an open transaction"
            )
        self.invalidate_lib(lib, tier, subtrees)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            row = self.conn.execute(
                "SELECT done, n_errors FROM symbol_index_libs WHERE lib=?", (lib,)
            ).fetchone()
            if row is None or int(row[0]) != 1 or int(row[1]) != 0:
                raise RuntimeError(
                    f"[{lib}] atomic rebuild ended without a clean done record"
                )
            self._delete_orphaned_identities()
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def lib_done(self, lib: str) -> bool:
        row = self.conn.execute(
            "SELECT done FROM symbol_index_libs WHERE lib=?", (lib,)
        ).fetchone()
        return bool(row and row[0])

    def insert_symbols(self, rows: list[GlobalSymbolRecord]) -> None:
        if any(not isinstance(row, GlobalSymbolRecord) for row in rows):
            raise ip.SymbolIdentityError(
                "legacy tuple symbol rows cannot reconstruct canonical identity; "
                "GlobalSymbolRecord is required"
            )
        records = list(rows)
        symbol_ids = self._validate_symbol_identity_claims(records)
        self.conn.executemany(
            "INSERT OR IGNORE INTO symbol_identities(symbol_id, symbol_key) VALUES (?, ?)",
            [
                (self._symbol_id_hex(symbol_id), record.symbol_key)
                for record, symbol_id in zip(records, symbol_ids, strict=True)
            ],
        )
        keyed_rows = [
            (
                self._symbol_uid(record),
                record.qname,
                record.base_lib,
                record.base_repo,
                record.kind,
                record.sym_type,
                record.file,
                record.line,
                record.end_line,
                record.is_public,
                record.token_est,
                record.body_len,
                record.text,
                record.symbol_key,
                self._symbol_id_hex(symbol_id),
                record.usr,
                record.canonical_signature,
                record.symbol_kind,
                record.identity_schema_version,
                record.provider,
                record.include_provenance,
            )
            for record, symbol_id in zip(records, symbol_ids, strict=True)
        ]
        self.conn.executemany(
            "INSERT OR IGNORE INTO symbols "
            "(symbol_uid, qname, base_lib, base_repo, kind, sym_type, file, line, end_line, "
            " is_public, token_est, body_len, text, symbol_key, symbol_id, usr, canonical_signature, "
            " symbol_kind, identity_schema_version, provider, include_provenance) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            keyed_rows,
        )

    def mark_lib_done(self, lib: str, tier: str, *, n_files: int, n_funcs: int,
                      n_types: int, n_errors: int, elapsed_s: float,
                      subtrees: list[str]) -> None:
        if not self.conn.in_transaction:
            raise RuntimeError(
                f"[{lib}] done state must be written inside an atomic rebuild"
            )
        if n_errors:
            raise RuntimeError(
                f"[{lib}] refusing to mark done after {n_errors} parse errors"
            )
        self.conn.execute(
            "INSERT OR REPLACE INTO symbol_index_libs "
            "(lib, tier, done, n_files, n_funcs, n_types, n_errors, elapsed_s, "
            " subtrees, finished_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (lib, tier, 1, n_files, n_funcs, n_types, n_errors, elapsed_s,
             ",".join(subtrees), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ---- read path (consulted by index_project) ----
    def lookup_candidates(self, qname: str) -> list[dict[str, object]]:
        qname = normalize_inline_namespace_qname(qname)
        rows = self.conn.execute(
            "SELECT symbol_uid, symbol_key, symbol_id, qname, base_lib, base_repo, kind, "
            "sym_type, file, line, end_line, is_public, token_est, body_len, text, "
            "usr, canonical_signature, symbol_kind, identity_schema_version, "
            "provider, include_provenance "
            "FROM symbols WHERE qname=? AND sym_type='func' "
            "ORDER BY base_repo, file, line, symbol_uid",
            (qname,),
        ).fetchall()
        fields = (
            "symbol_uid", "symbol_key", "symbol_id", "qname", "base_lib", "base_repo", "kind",
            "sym_type", "file", "line", "end_line", "is_public", "token_est",
            "body_len", "text", "usr", "canonical_signature", "symbol_kind",
            "identity_schema_version", "provider", "include_provenance",
        )
        records = [dict(zip(fields, row, strict=True)) for row in rows]
        for record in records:
            record["symbol_id"] = int(str(record["symbol_id"]), 16)
        return records

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
        candidates = self.lookup_candidates(qname)
        if usr:
            candidates = [row for row in candidates if row["usr"] == usr]
        else:
            fallback_used = False
            if canonical_signature:
                fallback_used = True
                signature = ip._normalize_signature_text(canonical_signature)
                candidates = [
                    row
                    for row in candidates
                    if ip._normalize_signature_text(
                        str(row["canonical_signature"])
                    ) == signature
                ]
            if symbol_kind:
                fallback_used = True
                candidates = [
                    row for row in candidates if row["symbol_kind"] == symbol_kind
                ]
            if not fallback_used and symbol_key:
                candidates = [row for row in candidates if row["symbol_key"] == symbol_key]
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
            raise ip.SymbolIdentityError(
                "ambiguous global symbol lookup: "
                f"qname={qname!r} usr={usr or ''!r} provider={provider or ''!r} "
                f"include={include_provenance or ''!r} candidates={preview}"
            )
        return None

    def counts(self) -> dict:
        cur = self.conn.execute("SELECT base_lib, sym_type, COUNT(*) FROM symbols "
                                "GROUP BY base_lib, sym_type")
        per: dict[str, dict[str, int]] = {}
        for lib, st, n in cur.fetchall():
            per.setdefault(lib, {"func": 0, "type": 0})[st] = n
        return per

    def lib_stats(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT lib, tier, done, n_files, n_funcs, n_types, n_errors, "
            "elapsed_s, subtrees FROM symbol_index_libs ORDER BY tier, lib")
        out = []
        for r in cur.fetchall():
            out.append({"lib": r[0], "tier": r[1], "done": r[2], "n_files": r[3],
                        "n_funcs": r[4], "n_types": r[5], "n_errors": r[6],
                        "elapsed_s": r[7], "subtrees": r[8]})
        return out


# --------------------------------------------------------------------------- #
# Streaming extraction of selected subtrees from the tarball.
# --------------------------------------------------------------------------- #
# Extensionless files that are documentation / build / metadata, NOT headers.
_NON_HEADER_BASENAMES = frozenset({
    "readme", "license", "licence", "copying", "copyright", "authors", "news",
    "todo", "changelog", "changes", "install", "notice", "version", "credits",
    "thanks", "bugs", "faq", "makefile", "gnumakefile", "dockerfile", "owners",
    "contributing", "manifest",
})
# Path components that hold the EXTENSIONLESS C++ stdlib public headers:
#   libstdc++/libc++/microsoft-STL use ``include``/``inc``; libstdc++ also keeps
#   <typeinfo>/<exception>/<new> under ``libsupc++``.
_HEADER_DIR_COMPONENTS = frozenset({"include", "inc", "libsupc++"})


def _is_extensionless_header(name: str) -> bool:
    """True if ``name`` is an EXTENSIONLESS C++ standard-library header.

    The libstdc++/libc++ public surface (<vector>, <type_traits>, <memory>,
    <typeinfo>, ...) has NO file extension, so the normal extension filter skips
    it entirely. We only treat an extensionless file as a header when it lives
    under a stdlib header dir component AND its basename is not a well-known
    doc/build file — this avoids grabbing extensionless license/readme/makefile
    noise.
    """
    base = os.path.basename(name)
    if not base or base.startswith("."):
        return False
    if os.path.splitext(base)[1] != "":
        return False  # has a real extension
    if base.lower() in _NON_HEADER_BASENAMES:
        return False
    parts = set(Path(name.replace("\\", "/")).parts)
    return bool(parts & _HEADER_DIR_COMPONENTS)


def _member_is_wanted(name: str, index_exts, *, path_markers, extensionless: bool) -> bool:
    """Decide whether a tarball member (by path) should be extracted for a lib.

    ``path_markers`` (when non-empty) RESTRICTS extraction to members whose path
    contains one of the markers. This is the C2 fix for the huge gcc/llvm
    monorepos: ``std`` pulls ONLY the C++ standard-library header trees
    (``/libstdc++-v3/include/``, ``/libcxx/include/``, ``/stl/inc/``) and NEVER
    gcc's compiler / libiberty internals, so the per-lib file cap is spent on real
    ``std::`` headers instead of compiler-internal noise.

    ``extensionless`` additionally admits extensionless std headers (<vector>…).
    """
    if path_markers and not any(m in name for m in path_markers):
        return False
    ext = os.path.splitext(name)[1].lower()
    if ext in index_exts:
        return True
    if extensionless and ext == "" and _is_extensionless_header(name):
        return True
    return False


def _find_extensionless_headers(dest: Path) -> list[str]:
    """Discover EXTENSIONLESS public std headers under ``dest`` (<vector>, …).

    ``ip.find_cpp_files`` filters by INDEX_EXTENSIONS and therefore skips the
    extensionless libstdc++/libc++ public headers entirely. We augment discovery
    with them (bounded by the same per-file size cap) so they are parsed.
    """
    out: list[str] = []
    for root, _dirs, files in os.walk(dest):
        for fn in files:
            full = os.path.join(root, fn)
            if not _is_extensionless_header(full):
                continue
            try:
                if os.path.getsize(full) > MAX_BYTES_PER_FILE:
                    continue
            except OSError:
                continue
            out.append(full)
    return out


def extract_subtrees(tarball: Path, subtrees: list[str], dest: Path,
                     *, max_files: int, member_cap: int | None = None,
                     path_markers: list[str] | None = None,
                     extensionless: bool = False) -> int:
    """Stream the tarball and extract members under cpp_all/<subtree>/ that are
    C/C++ source/header files, up to ``max_files`` per call.

    Returns the number of files extracted. Streaming (mode 'r|') so we never
    materialize the 235 GiB tarball; we filter by member name on the fly.
    RULE #1: a missing/zero subtree match RAISES at the caller (we just report 0).
    """
    if not tarball.exists():
        raise FileNotFoundError(f"tarball not found: {tarball}")
    wanted_prefixes = tuple(f"cpp_all/{s}/" for s in subtrees)
    dest.mkdir(parents=True, exist_ok=True)
    index_exts = ip.INDEX_EXTENSIONS

    proc = subprocess.Popen(
        ["zstd", "-dc", "--long=31", str(tarball)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    extracted = 0
    members_seen = 0
    try:
        tf = tarfile.open(fileobj=proc.stdout, mode="r|")
        for member in tf:
            members_seen += 1
            if member_cap is not None and members_seen > member_cap:
                break
            if not member.isfile():
                continue
            name = member.name
            if not name.startswith(wanted_prefixes):
                continue
            if not _member_is_wanted(name, index_exts, path_markers=path_markers,
                                     extensionless=extensionless):
                continue
            if member.size > MAX_BYTES_PER_FILE:
                continue
            # Sanitize path: strip leading cpp_all/ to keep dest shallow.
            rel = name[len("cpp_all/"):]
            out_path = dest / rel
            if ".." in Path(rel).parts:
                continue  # never escape dest
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            with open(out_path, "wb") as w:
                w.write(fobj.read())
            extracted += 1
            if extracted >= max_files:
                break
            if extracted % PROGRESS_EVERY == 0:
                print(f"    extracted {extracted} files "
                      f"({members_seen} members streamed)...", file=sys.stderr,
                      flush=True)
    finally:
        try:
            proc.stdout.close()
        except ip.SymbolIdentityError:
            raise
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except ip.SymbolIdentityError:
            raise
        except Exception:
            proc.kill()
    return extracted


def extract_many_subtrees(tarball: Path, lib_specs: dict[str, dict], dest_root: Path,
                          *, max_files: int, member_cap: int | None = None) -> dict[str, int]:
    """Extract the subtrees of MULTIPLE libs in ONE streaming tarball pass.

    Routes each matching member to ``dest_root/<lib>/<rel>``. This is the path
    used for a full multi-lib build so the 235 GiB tarball is streamed exactly
    ONCE instead of once per lib. Returns {lib: n_files_extracted}.
    """
    if not tarball.exists():
        raise FileNotFoundError(f"tarball not found: {tarball}")
    index_exts = ip.INDEX_EXTENSIONS
    # subtree-name -> lib (a subtree maps to exactly one lib here)
    prefix_to_lib: dict[str, str] = {}
    for lib, spec in lib_specs.items():
        for sub in spec["subtrees"]:
            prefix_to_lib[f"cpp_all/{sub}/"] = lib
    prefixes = tuple(prefix_to_lib.keys())
    counts: dict[str, int] = {lib: 0 for lib in lib_specs}
    dest_root.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        ["zstd", "-dc", "--long=31", str(tarball)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    members_seen = 0
    total = 0
    try:
        tf = tarfile.open(fileobj=proc.stdout, mode="r|")
        for member in tf:
            members_seen += 1
            if member_cap is not None and members_seen > member_cap:
                break
            if not member.isfile():
                continue
            name = member.name
            matched_prefix = None
            for p in prefixes:
                if name.startswith(p):
                    matched_prefix = p
                    break
            if matched_prefix is None:
                continue
            lib = prefix_to_lib[matched_prefix]
            if counts[lib] >= max_files:
                continue
            lspec = lib_specs[lib]
            if not _member_is_wanted(
                    name, index_exts,
                    path_markers=lspec.get("include_path_markers"),
                    extensionless=bool(lspec.get("index_extensionless_headers"))):
                continue
            if member.size > MAX_BYTES_PER_FILE:
                continue
            rel = name[len("cpp_all/"):]
            if ".." in Path(rel).parts:
                continue
            out_path = dest_root / lib / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            with open(out_path, "wb") as w:
                w.write(fobj.read())
            counts[lib] += 1
            total += 1
            if total % PROGRESS_EVERY == 0:
                done = {k: v for k, v in counts.items() if v}
                print(f"    extracted {total} files ({members_seen} members "
                      f"streamed): {done}", file=sys.stderr, flush=True)
    finally:
        try:
            proc.stdout.close()
        except ip.SymbolIdentityError:
            raise
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except ip.SymbolIdentityError:
            raise
        except Exception:
            proc.kill()
    return counts


def _tarball_fingerprint(tarball: Path) -> dict[str, object]:
    st = tarball.stat()
    return {
        "path": str(tarball.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def _extract_cache_signature(
    tarball: Path,
    spec: dict,
    *,
    max_files: int,
    member_cap: int | None,
) -> dict[str, object]:
    return {
        "version": 1,
        "tarball": _tarball_fingerprint(tarball),
        "subtrees": list(spec["subtrees"]),
        "include_path_markers": list(spec.get("include_path_markers") or []),
        "index_extensionless_headers": bool(spec.get("index_extensionless_headers")),
        "max_files": max_files,
        "member_cap": member_cap,
        "max_bytes_per_file": MAX_BYTES_PER_FILE,
    }


def _extract_cache_manifest_path(cache_root: Path, lib: str) -> Path:
    return cache_root / lib / ".gsi_extract_complete.json"


def _read_extract_cache_manifest(cache_root: Path, lib: str) -> dict | None:
    path = _extract_cache_manifest_path(cache_root, lib)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"corrupt extraction cache manifest: {path}: {exc}") from exc


def _write_extract_cache_manifest(
    cache_root: Path,
    lib: str,
    signature: dict[str, object],
    *,
    count: int,
) -> None:
    path = _extract_cache_manifest_path(cache_root, lib)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(signature)
    payload.update({
        "lib": lib,
        "count": count,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _extract_cache_hit(
    tarball: Path,
    spec: dict,
    cache_root: Path,
    lib: str,
    *,
    max_files: int,
    member_cap: int | None,
) -> int | None:
    manifest = _read_extract_cache_manifest(cache_root, lib)
    if manifest is None:
        return None
    signature = _extract_cache_signature(
        tarball, spec, max_files=max_files, member_cap=member_cap
    )
    for key, value in signature.items():
        if manifest.get(key) != value:
            return None
    count = int(manifest.get("count", 0))
    if count <= 0:
        return None
    if not (cache_root / lib).is_dir():
        return None
    return count


def prepare_extraction_cache(
    tarball: Path,
    lib_specs: dict[str, dict],
    cache_root: Path,
    *,
    max_files: int,
    member_cap: int | None = None,
) -> tuple[dict[str, Path], dict[str, int]]:
    """Return extracted source dirs, populating a reusable extraction cache.

    The expensive operation is the 235 GiB ``zstd | tar`` stream. This cache
    makes that step one-time per (tarball fingerprint, lib spec, max_files,
    member_cap): later rebuilds re-index the cached source dirs without reading
    the tarball again. The cache stores SOURCE FILES only; symbol extraction still
    uses the current parser/indexer code, so C++20/23/26 parser fixes take effect
    on every rebuild without re-extraction.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    dirs: dict[str, Path] = {}
    missing: dict[str, dict] = {}

    for lib, spec in lib_specs.items():
        hit = _extract_cache_hit(
            tarball, spec, cache_root, lib,
            max_files=max_files, member_cap=member_cap,
        )
        dirs[lib] = cache_root / lib
        if hit is None:
            missing[lib] = spec
        else:
            counts[lib] = hit
            print(f"  [{lib}] extraction cache hit: {hit} files",
                  file=sys.stderr, flush=True)

    if missing:
        for lib in missing:
            shutil.rmtree(cache_root / lib, ignore_errors=True)
        print(f"  extraction cache miss for {list(missing)} -> {cache_root}",
              file=sys.stderr, flush=True)
        extracted = extract_many_subtrees(
            tarball, missing, cache_root,
            max_files=max_files, member_cap=member_cap,
        )
        for lib, count in extracted.items():
            counts[lib] = count
            if count > 0:
                _write_extract_cache_manifest(
                    cache_root, lib,
                    _extract_cache_signature(
                        tarball, missing[lib], max_files=max_files,
                        member_cap=member_cap,
                    ),
                    count=count,
                )

    return dirs, counts


def populate_extraction_cache_from_source_cache(
    tarball: Path,
    lib_specs: dict[str, dict],
    source_cache_root: Path,
    cache_root: Path,
    *,
    max_files: int,
    member_cap: int | None = None,
) -> dict[str, int]:
    """Hardlink missing per-lib caches from a complete conveyor source cache."""
    if member_cap is not None:
        raise ValueError(
            "member_cap is archive-order-specific and cannot be used with "
            "source-cache extraction"
        )
    if not source_cache_root.is_dir():
        raise FileNotFoundError(f"source cache root not found: {source_cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    populated: dict[str, int] = {}

    for lib, spec in lib_specs.items():
        hit = _extract_cache_hit(
            tarball, spec, cache_root, lib,
            max_files=max_files, member_cap=None,
        )
        if hit is not None:
            populated[lib] = hit
            continue

        stage = cache_root / f".{lib}.building-{os.getpid()}"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        count = 0
        seen_source_dirs: set[tuple[int, int]] = set()
        try:
            for subtree in spec["subtrees"]:
                source_dir = source_cache_root / subtree
                sentinel = source_dir / ".cppmega_source_cache_complete.json"
                if not source_dir.is_dir() or not sentinel.is_file():
                    raise RuntimeError(
                        f"[{lib}] incomplete source-cache subtree: {source_dir}"
                    )
                source_stat = source_dir.stat()
                source_key = (int(source_stat.st_dev), int(source_stat.st_ino))
                if source_key in seen_source_dirs:
                    print(
                        f"  [{lib}] source-cache alias skipped: {subtree}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                seen_source_dirs.add(source_key)
                completion = json.loads(sentinel.read_text(encoding="utf-8"))
                cached_repo = completion.get("repo")
                if (
                    not isinstance(cached_repo, str)
                    or cached_repo.casefold() != subtree.casefold()
                ):
                    raise RuntimeError(
                        f"[{lib}] source-cache sentinel repo mismatch at {sentinel}"
                    )
                cached_source = completion.get("source")
                if not isinstance(cached_source, str) or not cached_source:
                    raise RuntimeError(
                        f"[{lib}] source-cache sentinel has no source at {sentinel}"
                    )
                try:
                    same_tarball = Path(cached_source).samefile(tarball)
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        f"[{lib}] source-cache tarball is unavailable: "
                        f"{cached_source}"
                    ) from exc
                if not same_tarball:
                    raise RuntimeError(
                        f"[{lib}] source cache came from {cached_source}, not "
                        f"{tarball}"
                    )

                for current_root, dirnames, filenames in os.walk(source_dir):
                    dirnames.sort()
                    filenames.sort()
                    for filename in filenames:
                        if count >= max_files:
                            break
                        source = Path(current_root) / filename
                        if source == sentinel or not source.is_file():
                            continue
                        relative = source.relative_to(source_dir)
                        member_name = f"cpp_all/{subtree}/{relative.as_posix()}"
                        if not _member_is_wanted(
                                member_name, ip.INDEX_EXTENSIONS,
                                path_markers=spec.get("include_path_markers"),
                                extensionless=bool(spec.get("index_extensionless_headers"))):
                            continue
                        if source.stat().st_size > MAX_BYTES_PER_FILE:
                            continue
                        target = stage / subtree / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.link(source, target)
                        count += 1
                    if count >= max_files:
                        break
                if count >= max_files:
                    break

            if count <= 0:
                raise RuntimeError(
                    f"[{lib}] source cache produced zero indexable files"
                )
            manifest = _extract_cache_signature(
                tarball, spec, max_files=max_files, member_cap=None
            )
            manifest.update({
                "lib": lib,
                "count": count,
                "finished_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "materialized_from": str(source_cache_root.resolve()),
                "publication": "hardlink",
            })
            (stage / ".gsi_extract_complete.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            final = cache_root / lib
            shutil.rmtree(final, ignore_errors=True)
            os.replace(stage, final)
            populated[lib] = count
            print(f"  [{lib}] source-cache extraction: {count} hardlinked files",
                  file=sys.stderr, flush=True)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return populated


# --------------------------------------------------------------------------- #
# Per-lib C++ standard-library compile environment (the std cross-link fix).
#
# The std umbrella headers (<vector>, <string>, <map>, ...) of libstdc++/libc++
# DO NOT parse standalone with the bare ['-x','c++','-std=cXX'] + self-include
# args that work for the A1 libs: they pull in compiler BUILTIN headers
# (<stddef.h>, <stdint.h>, <stdarg.h>, __stddef_*.h) that live in clang's
# RESOURCE-DIR, and C library headers (<cstddef>->stddef.h, <cstdint>->stdint.h)
# that live in a C sysroot. Without those, libclang only ever sees a handful of
# free functions and ZERO class templates (the "std=202 funcs/0 types" defect).
#
# We therefore build, ONCE per build (in the parent), the extra include args:
#   -isystem <clang-resource-dir>/include   (compiler builtins)
#   -isystem <sysroot>/usr/include + -isysroot <sysroot>  (C library headers)
# and pass them to every std parse worker. PRIMARY std target is libc++ (it ships
# its own __config and needs only builtins + a C lib). See BASE_LIBS["std"].
# --------------------------------------------------------------------------- #
def _find_clang_resource_include() -> str | None:
    """Locate a clang RESOURCE-DIR ``include`` that holds the builtin headers
    (stddef.h/stdint.h/stdarg.h). Tries ``clang -print-resource-dir`` for any
    clang on PATH / known toolchains, then well-known Homebrew/Xcode locations.

    The bundled libclang (pip ``libclang``) ships NO resource headers, so we must
    borrow them from an installed toolchain. A minor clang-version skew between
    the libclang parser and the resource headers is tolerated by the modular
    __stddef_*/__stdarg_* split (clang >= 18).
    """
    import glob as _glob
    cands: list[str] = []
    for clang_bin in ("clang", "/opt/homebrew/opt/llvm/bin/clang",
                      "/usr/bin/clang", "cc"):
        try:
            rd = subprocess.check_output(
                [clang_bin, "-print-resource-dir"],
                text=True, stderr=subprocess.DEVNULL).strip()
            if rd:
                cands.append(rd)
        except ip.SymbolIdentityError:
            raise
        except Exception:
            continue
    # Newest-first so a current toolchain (closest to modern libc++) wins.
    cands += sorted(_glob.glob("/opt/homebrew/Cellar/llvm/*/lib/clang/*"),
                    reverse=True)
    cands += sorted(_glob.glob(
        "/Applications/Xcode.app/Contents/Developer/Toolchains/"
        "XcodeDefault.xctoolchain/usr/lib/clang/*"), reverse=True)
    cands += sorted(_glob.glob("/usr/lib/clang/*"), reverse=True)
    for rd in cands:
        inc = os.path.join(rd, "include")
        if os.path.isfile(os.path.join(inc, "stddef.h")):
            return inc
    return None


def _find_c_sysroot() -> str | None:
    """Locate a C library sysroot for <cstddef>/<cstdint>/... resolution.

    macOS: the SDK from ``xcrun --show-sdk-path`` (headers under <sdk>/usr/include).
    Linux: ``/`` (headers under /usr/include). Returns the sysroot ROOT (the
    caller appends usr/include).
    """
    try:
        sdk = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True,
            stderr=subprocess.DEVNULL).strip()
        if sdk and os.path.isdir(os.path.join(sdk, "usr", "include")):
            return sdk
    except ip.SymbolIdentityError:
        raise
    except Exception:
        pass
    for sdk in ("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk", "/"):
        if os.path.isdir(os.path.join(sdk, "usr", "include")):
            return sdk
    return None


def _cxx_stdlib_sysroot_args() -> list[str]:
    """Extra libclang args giving the C++ stdlib umbrella headers their compiler
    builtins + a C library. RAISES if no clang resource-dir is found (RULE #1:
    std cannot be indexed without builtins — fail loud, never silently degrade)."""
    res_inc = _find_clang_resource_include()
    if not res_inc:
        raise RuntimeError(
            "std cross-link build requires clang builtin headers (stddef.h/"
            "stdint.h) from a clang RESOURCE-DIR; none found via "
            "`clang -print-resource-dir` or known Homebrew/Xcode paths. "
            "Install LLVM or Xcode command-line tools. RULE #1: fail loud."
        )
    args = ["-isystem", res_inc]
    sysroot = _find_c_sysroot()
    if sysroot:
        args += ["-isystem", os.path.join(sysroot, "usr", "include")]
        if sysroot != "/":
            args += ["-isysroot", sysroot]
    return args


def _stdlib_include_root_for(filepath: str, markers: Sequence[str] | None) -> str | None:
    """For a std header, the SINGLE include root of the stdlib tree it lives in.

    The std lib bundles THREE separate C++ standard libraries (libc++, libstdc++,
    MS-STL). Their umbrella header names collide (<vector> exists in all three),
    so a libc++ header must be compiled with ONLY libc++'s include root on -I or
    it could resolve a libstdc++/MS-STL sibling (cross-tree contamination). Given
    e.g. ``.../llvm-project/libcxx/include/__vector/vector.h`` and marker
    ``/libcxx/include/`` this returns ``.../llvm-project/libcxx/include``.
    """
    if not markers:
        return None
    p = filepath.replace("\\", "/")
    for m in markers:
        idx = p.find(m)
        if idx != -1:
            return p[:idx] + m.rstrip("/")
    return None


def _find_libcxx_include_root(cpp_files: Sequence[str]) -> str | None:
    """The libc++ public-header include root (``.../libcxx/include``) from the
    already-discovered std source files. Used as the ``-I`` root for the cxx-std
    probe so ``#include <ranges>`` / ``#include <format>`` resolve to the REAL
    extracted libc++ umbrella headers (not a system libc++)."""
    for fp in cpp_files:
        root = _stdlib_include_root_for(fp, ("/libcxx/include/",))
        if root:
            return root
    return None


def _probe_highest_cxx_std(
    candidates: Sequence[str],
    include_root: str | None,
    sysroot_args: Sequence[str],
    *,
    lib: str,
) -> str:
    """Pick the HIGHEST ``-std=c++NN`` flag libclang ACTUALLY accepts, verified by
    a real probe parse of libc++ ``<ranges>`` + ``<format>``.

    Candidates are tried highest-first. A candidate is REJECTED only when libclang
    emits a *driver-level* std diagnostic for it — ``invalid value '<v>' in
    '-std=<v>'`` or ``unknown argument '<v>'`` — i.e. the toolchain does not know
    that standard; we then step DOWN to the next candidate. (A missing-header or
    template error is NOT a std-flag rejection — stepping down would not fix it —
    so it never triggers a step-down.) The first accepted candidate wins and is
    logged with its probe evidence. If NO candidate is accepted we RAISE (RULE #1:
    this is an EXPLICIT probe, never a silent except — and we never drop below the
    configured candidate set)."""
    if not candidates:
        raise RuntimeError(f"[{lib}] cxx_std probe requested with no candidates")
    ip._configure_libclang()
    probe_index = ip.Index.create()
    parse_opts = int(getattr(ip.TranslationUnit, "PARSE_INCOMPLETE", 0) or 0)
    rejections: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"gsi_stdprobe_{lib}_") as td:
        probe_src = os.path.join(td, "cxx_std_probe.cpp")
        with open(probe_src, "w") as fh:
            fh.write("#include <ranges>\n#include <format>\nint main() { return 0; }\n")
        for cand in candidates:
            value = cand.split("=", 1)[1] if "=" in cand else cand
            compile_args = ["-x", "c++", cand]
            if include_root:
                compile_args += ["-I", include_root]
            compile_args += list(sysroot_args)
            # An UNKNOWN -std makes libclang's driver fail so hard it produces no
            # TU and raises TranslationUnitLoadError. That raised parse IS the
            # rejection signal for this candidate: record it and step DOWN. This
            # is the probe's explicit detection mechanism ("if the std flag errors,
            # step down"), NOT a silent fallback — if NO candidate parses we RAISE
            # below with every collected reason (RULE #1: fail loud).
            try:
                tu = probe_index.parse(probe_src, args=compile_args,
                                       options=parse_opts)
            except ip.SymbolIdentityError:
                raise
            except Exception as exc:  # noqa: BLE001 - per-candidate probe rejection
                rejections.append(f"{cand} (parse raised: {type(exc).__name__})")
                print(f"  [{lib}] cxx_std probe: {cand} REJECTED by libclang "
                      f"({type(exc).__name__}) -> stepping down",
                      file=sys.stderr, flush=True)
                continue
            diags = list(tu.diagnostics)
            std_rejected = any(
                value in d.spelling
                and ("invalid value" in d.spelling or "unknown argument" in d.spelling)
                for d in diags
            )
            if std_rejected:
                rejections.append(f"{cand} (driver diagnostic)")
                print(f"  [{lib}] cxx_std probe: {cand} REJECTED by libclang "
                      f"(driver diagnostic) -> stepping down",
                      file=sys.stderr, flush=True)
                continue
            n_fatal = sum(1 for d in diags if d.severity >= 4)
            headers_parsed = bool(include_root) and n_fatal == 0
            print(f"  [{lib}] cxx_std probe: chose {cand} "
                  f"(libclang accepts; diagnostics={len(diags)} fatal={n_fatal} "
                  f"libcxx_ranges_format="
                  f"{'parsed' if headers_parsed else 'flag-accepted'})",
                  file=sys.stderr, flush=True)
            return cand
    raise RuntimeError(
        f"[{lib}] no C++ standard from {tuple(candidates)} was accepted by "
        f"libclang on a libc++ <ranges>/<format> probe parse (tried: "
        f"{rejections}). RULE #1: refusing to index std with an unknown -std flag."
    )


# --------------------------------------------------------------------------- #
# Per-file libclang parse worker (isolated subprocess so a segfault is local).
# --------------------------------------------------------------------------- #
def _parse_file_worker(args_tuple):
    (
        filepath,
        project_dir,
        project_id,
        std_arg,
        lang,
        include_dirs,
        lib_env,
    ) = args_tuple
    project_id = ip.require_project_identity(
        project_id, source=f"global symbol worker {filepath}"
    )
    lib_env = lib_env or {}
    sys.setrecursionlimit(50000)
    ip._configure_libclang()
    clang_index = ip.Index.create()
    ext = os.path.splitext(filepath)[1].lower()
    p = filepath.replace("\\", "/")
    # C++ std level is per-lib: the std umbrella headers need >= c++20; other
    # libs use the run-wide --std-arg (default c++17).
    cxx_std = lib_env.get("cxx_std") or std_arg
    # Language is decided PER LIB (lang), not per extension: header-only C++
    # template libs (boost/eigen/fmt) MUST parse as C++ or their public API
    # yields zero symbols. A genuine .c file in a C++ lib is still parsed as C.
    if ext == ".c" or (lang == "c" and ext in {".h", ".inc", ".inl"}):
        compile_args = ["-x", "c", "-std=c11"]
    else:
        compile_args = ["-x", "c++", cxx_std]
    # Include roots. For std (stdlib_include_markers set) pin a SINGLE per-file
    # include root (the tree the header belongs to) to avoid libc++/libstdc++/
    # MS-STL cross-tree contamination. All other libs use the discovered
    # self-include dirs so a header resolves its own siblings (cuts parse errors
    # + time for header-heavy libs). PARSE_INCOMPLETE keeps going past unresolved
    # system includes.
    markers = lib_env.get("stdlib_include_markers")
    std_root = _stdlib_include_root_for(p, markers) if markers else None
    if std_root:
        compile_args += ["-I", std_root]
    else:
        for inc in include_dirs:
            compile_args += ["-I", inc]
    # MS-STL needs MSVC compatibility/intrinsics; enable best-effort for its
    # headers. Files that still need genuine MSVC intrinsics fail PER FILE (a
    # counted parse error in the report) — NOT a silent wholesale drop (RULE #1).
    msvc_marker = lib_env.get("msvc_path_marker")
    if msvc_marker and msvc_marker in p:
        compile_args += ["-fms-compatibility", "-fms-extensions",
                         "-fdelayed-template-parsing"]
    # clang builtin headers + C sysroot (required by std umbrella headers).
    compile_args += lib_env.get("sysroot_args") or []
    funcs, types = ip.parse_translation_unit(
        filepath, clang_index, compile_args, project_dir,
        allow_system_types=bool(lib_env.get("allow_system_types")),
        project_id=project_id,
    )
    return (
        [f.to_dict() for f in funcs],
        [t.to_dict() for t in types],
    )


class ParseWorkerTimeout(RuntimeError):
    """A per-file parse worker exceeded its wall-clock deadline."""


def _abort_process_pool(executor: ProcessPoolExecutor) -> None:
    """Stop running workers before waiting for executor shutdown."""
    process_map = getattr(executor, "_processes", None)
    processes = tuple(process_map.values()) if process_map else ()
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (OSError, ValueError):
            continue
    for process in processes:
        try:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        except (OSError, ValueError):
            continue
    executor.shutdown(wait=True, cancel_futures=True)


def _iter_parse_results(
    tasks: Iterable[tuple[str, str, tuple]],
    *,
    workers: int,
    timeout_s: float,
    worker_fn: Callable[[tuple], tuple[list[dict], list[dict]]] = _parse_file_worker,
) -> Iterator[tuple[str, str, tuple[list[dict], list[dict]]]]:
    """Run a bounded set of parse workers with a deadline per submitted file."""
    if timeout_s <= 0:
        raise ValueError(f"parse worker timeout must be positive, got {timeout_s}")

    executor = ProcessPoolExecutor(max_workers=max(1, workers))
    task_iter = iter(tasks)
    pending: dict[Future, tuple[str, str, float]] = {}
    exhausted = False
    aborted = False

    def submit_one() -> None:
        nonlocal exhausted
        if exhausted:
            return
        try:
            filepath, project_id, worker_args = next(task_iter)
        except StopIteration:
            exhausted = True
            return
        future = executor.submit(worker_fn, worker_args)
        pending[future] = (filepath, project_id, time.monotonic())

    try:
        for _ in range(max(1, workers)):
            submit_one()
        while pending:
            now = time.monotonic()
            future, (filepath, _project_id, started) = min(
                pending.items(), key=lambda item: item[1][2]
            )
            remaining = started + timeout_s - now
            if remaining <= 0 and not future.done():
                raise ParseWorkerTimeout(
                    f"parse worker timed out after {timeout_s:g}s for {filepath}"
                )
            done, _ = wait(
                pending,
                timeout=max(0.0, remaining),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                raise ParseWorkerTimeout(
                    f"parse worker timed out after {timeout_s:g}s for {filepath}"
                )
            for completed in sorted(done, key=lambda item: pending[item][0]):
                completed_filepath, completed_project, _started = pending.pop(completed)
                try:
                    result = completed.result()
                except ip.SymbolIdentityError:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"parse worker failed for {completed_filepath}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                submit_one()
                yield completed_filepath, completed_project, result
    except BaseException:
        aborted = True
        _abort_process_pool(executor)
        raise
    finally:
        if not aborted:
            executor.shutdown(wait=True, cancel_futures=True)


@dataclass
class LibResult:
    lib: str
    tier: str
    subtrees: list[str]
    n_files: int = 0
    n_funcs: int = 0
    n_types: int = 0
    n_errors: int = 0
    elapsed_s: float = 0.0
    func_qnames: set = field(default_factory=set)


def _discover_include_dirs(dest: Path, subtrees: list[str]) -> list[str]:
    """Common self-include roots so a base-lib header resolves its siblings.

    Adds the extraction root, each subtree root, and any ``include``/``src``
    subdir found one or two levels down. Bounded (no deep walk).
    """
    inc: list[str] = [str(dest)]
    for sub in subtrees:
        base = dest / sub
        if base.is_dir():
            inc.append(str(base))
            for cand in ("include", "src", "src/include"):
                p = base / cand
                if p.is_dir():
                    inc.append(str(p))
            # one level down: <sub>/<pkg>/include (e.g. abseil-cpp/absl/...)
            try:
                for child in base.iterdir():
                    if child.is_dir() and (child / "include").is_dir():
                        inc.append(str(child / "include"))
            except OSError:
                pass
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for d in inc:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _index_lib_from_dir_uncommitted(
    lib: str,
    spec: dict,
    dest: Path,
    store: GlobalSymbolStore,
    *,
    workers: int,
    std_arg: str,
    parse_timeout_s: float = PARSE_TIMEOUT_S,
) -> LibResult:
    """Parse an extracted base-lib inside the caller's SQLite transaction."""
    t0 = time.time()
    subtrees = spec["subtrees"]
    tier = spec["tier"]
    public_only = spec["public_only"]
    namespace_prefixes = tuple(spec.get("namespace_prefixes") or ())
    lang = spec.get("lang", "c++")
    res = LibResult(lib=lib, tier=tier, subtrees=subtrees)

    # Per-lib compile environment (only std sets these). Computed ONCE in the
    # parent and shipped to every worker so the C++ stdlib umbrella headers
    # resolve their builtin/C-library includes; allow_system_types lets the
    # parser keep std:: TYPE definitions (std::vector/std::map/...). For non-std
    # libs every field is absent -> the worker behaves exactly as before.
    sysroot_args: list[str] = []
    if spec.get("needs_cxx_stdlib_env"):
        sysroot_args = _cxx_stdlib_sysroot_args()  # RAISES if no builtins found

    cpp_files = ip.find_cpp_files(str(dest))
    if spec.get("index_extensionless_headers"):
        # ip.find_cpp_files skips extensionless files; the libstdc++/libc++ public
        # headers (<vector>, <type_traits>, ...) are extensionless, so add them or
        # the std public surface is never parsed (a C2 root cause).
        seen = set(cpp_files)
        for fp in _find_extensionless_headers(dest):
            if fp not in seen:
                cpp_files.append(fp)
                seen.add(fp)
    cpp_files, excluded = _filter_non_provider_sources(cpp_files, spec)
    if excluded:
        print(
            f"  [{lib}] configured non-provider source exclusions: {excluded} files "
            f"(suffixes={tuple(spec.get('index_exclude_suffixes', ()))}, "
            f"path_segments={tuple(spec.get('index_exclude_path_segments', ()))}, "
            f"public_headers_only={bool(spec.get('index_public_headers_only'))})",
            file=sys.stderr,
            flush=True,
        )
    res.n_files = len(cpp_files)
    if res.n_files == 0:
        raise RuntimeError(
            f"[{lib}] 0 parseable files in {dest} — extraction produced nothing "
            f"(subtree name(s) {subtrees} wrong/absent?). RULE #1: fail loud."
        )
    project_dir = str(dest)
    include_dirs = _discover_include_dirs(dest, subtrees)

    # ---- choose the C++ standard for this lib ----
    # When the spec configures cxx_std_probe_candidates (only std does), pick the
    # HIGHEST -std the installed libclang actually accepts on a real libc++
    # <ranges>/<format> probe parse — a SUPERSET std so the c++23/26 standard
    # surface (ranges/concepts/format/expected/flat_map) is captured. The static
    # spec["cxx_std"] is only the documented FLOOR; the probe (explicit, logged,
    # RAISES on total failure) replaces it. Non-std libs keep their static cxx_std.
    probe_candidates = spec.get("cxx_std_probe_candidates")
    if probe_candidates:
        libcxx_root = _find_libcxx_include_root(cpp_files)
        chosen_cxx_std = _probe_highest_cxx_std(
            probe_candidates, libcxx_root, sysroot_args, lib=lib,
        )
    else:
        chosen_cxx_std = spec.get("cxx_std")

    lib_env = {
        "cxx_std": chosen_cxx_std,
        "sysroot_args": sysroot_args,
        "stdlib_include_markers": spec.get("stdlib_include_markers"),
        "msvc_path_marker": spec.get("msvc_path_marker"),
        "allow_system_types": bool(spec.get("allow_system_types")),
    }

    print(f"  [{lib}] lang={lang} cxx_std={lib_env['cxx_std'] or std_arg} "
          f"include_dirs={len(include_dirs)} "
          f"stdlib_env={'on' if sysroot_args else 'off'} "
          f"allow_system_types={lib_env['allow_system_types']} "
          f"files_to_parse={res.n_files} ({workers} workers)",
          file=sys.stderr, flush=True)
    if sysroot_args:
        print(f"  [{lib}] stdlib sysroot_args: {' '.join(sysroot_args)}",
              file=sys.stderr, flush=True)

    batch: list[GlobalSymbolRecord] = []
    parsed = 0

    def parse_tasks() -> Iterator[tuple[str, str, tuple]]:
        for fp in cpp_files:
            relative = os.path.relpath(fp, project_dir)
            subtree = relative.split(os.sep, 1)[0]
            if subtree not in subtrees:
                raise ip.SymbolIdentityError(
                    f"[{lib}] parsed file is outside configured subtrees: {relative!r}"
                )
            project_id = project_identity_for_subtree(subtree)
            yield (
                fp,
                project_id,
                (
                    fp,
                    project_dir,
                    project_id,
                    std_arg,
                    lang,
                    include_dirs,
                    lib_env,
                ),
            )

    parse_results = _iter_parse_results(
        parse_tasks(), workers=workers, timeout_s=parse_timeout_s
    )
    try:
        for _fp, base_repo, (func_dicts, type_dicts) in parse_results:
            for d in func_dicts:
                qn = normalize_inline_namespace_qname(d["qualified_name"])
                frel = d["file"]
                if not is_public_symbol(qn, frel, public_only, namespace_prefixes):
                    continue
                body = d["text"]
                blen = normalized_body_len(body)
                # Skip trivial/tiny inline std bodies (not worth a chunk).
                if public_only and blen < 24:
                    continue
                if blen < 12:
                    continue
                tok = ip.estimate_tokens(body)
                provider, include_provenance = ip.symbol_provider_provenance(frel)
                if lib == "std" and (not provider or not include_provenance):
                    raise ip.SymbolIdentityError(
                        f"[{lib}] cannot determine provider/include provenance for "
                        f"{base_repo}:{frel}"
                    )
                batch.append(GlobalSymbolRecord(
                    qname=qn,
                    base_lib=lib,
                    base_repo=base_repo,
                    kind=2,
                    sym_type="func",
                    file=frel,
                    line=d["line"],
                    end_line=d.get("end_line", 0),
                    is_public=1,
                    token_est=tok,
                    body_len=blen,
                    text=body,
                    symbol_key=d["symbol_key"],
                    usr=d.get("usr", ""),
                    canonical_signature=d.get("canonical_signature", ""),
                    symbol_kind=d.get("symbol_kind", "FUNCTION_DECL"),
                    provider=provider or base_repo,
                    include_provenance=include_provenance or frel,
                ))
                res.n_funcs += 1
                res.func_qnames.add(qn)
            for d in type_dicts:
                qn = normalize_inline_namespace_qname(d["qualified_name"])
                frel = d["file"]
                if not is_public_symbol(qn, frel, public_only, namespace_prefixes):
                    continue
                body = d["text"]
                blen = normalized_body_len(body)
                if blen < 8:
                    continue
                tok = ip.estimate_tokens(body)
                provider, include_provenance = ip.symbol_provider_provenance(frel)
                if lib == "std" and (not provider or not include_provenance):
                    raise ip.SymbolIdentityError(
                        f"[{lib}] cannot determine provider/include provenance for "
                        f"{base_repo}:{frel}"
                    )
                batch.append(GlobalSymbolRecord(
                    qname=qn,
                    base_lib=lib,
                    base_repo=base_repo,
                    kind=d["kind"],
                    sym_type="type",
                    file=frel,
                    line=d["line"],
                    end_line=d.get("end_line", 0),
                    is_public=1,
                    token_est=tok,
                    body_len=blen,
                    text=body,
                    symbol_key=d["symbol_key"],
                    usr=d.get("usr", ""),
                    canonical_signature=d.get("canonical_signature", ""),
                    symbol_kind=d.get("symbol_kind", "TYPE_DECL"),
                    provider=provider or base_repo,
                    include_provenance=include_provenance or frel,
                ))
                res.n_types += 1
            parsed += 1
            if len(batch) >= SYMBOL_BATCH_SIZE:
                store.insert_symbols(batch)
                batch = []
            if parsed % PROGRESS_EVERY == 0:
                print(f"    [{lib}] parsed {parsed}/{res.n_files} files, "
                      f"{res.n_funcs} funcs / {res.n_types} types, "
                      f"{res.n_errors} errors", file=sys.stderr, flush=True)
    finally:
        parse_results.close()
    if batch:
        store.insert_symbols(batch)

    # Fail-loud guard for the std-type path (CLAUDE.md RULE #1): when a lib is
    # configured to capture system (std::) TYPES, parsing files but extracting
    # ZERO class types means the C++ stdlib compile-env is broken (missing
    # builtins / wrong include root / unparsable umbrellas) — exactly the old
    # "std=0 types" defect. Do NOT silently mark such a lib done; RAISE so the
    # broken env is surfaced and fixed.
    if spec.get("allow_system_types") and res.n_files > 0 and res.n_types == 0:
        raise RuntimeError(
            f"[{lib}] parsed {res.n_files} files but captured 0 class TYPES with "
            f"allow_system_types=on ({res.n_funcs} funcs, {res.n_errors} errors). "
            f"The C++ stdlib compile-env is broken (clang builtins / include root "
            f"/ umbrella parse). RULE #1: refusing to record a broken std index."
        )

    res.elapsed_s = time.time() - t0
    return res


def index_lib_from_dir(
    lib: str,
    spec: dict,
    dest: Path,
    store: GlobalSymbolStore,
    *,
    workers: int,
    std_arg: str,
    parse_timeout_s: float = PARSE_TIMEOUT_S,
) -> LibResult:
    """Atomically replace one base-lib generation and mark it complete."""
    with store.rebuild_lib(lib, spec["tier"], spec["subtrees"]):
        result = _index_lib_from_dir_uncommitted(
            lib,
            spec,
            dest,
            store,
            workers=workers,
            std_arg=std_arg,
            parse_timeout_s=parse_timeout_s,
        )
        store.mark_lib_done(
            lib,
            spec["tier"],
            n_files=result.n_files,
            n_funcs=result.n_funcs,
            n_types=result.n_types,
            n_errors=result.n_errors,
            elapsed_s=result.elapsed_s,
            subtrees=result.subtrees,
        )
    return result


def index_one_lib(lib: str, spec: dict, tarball: Path, store: GlobalSymbolStore,
                  *, workers: int, max_files: int, std_arg: str,
                  member_cap: int | None,
                  parse_timeout_s: float = PARSE_TIMEOUT_S) -> LibResult:
    """Extract (own tarball pass) + index ONE base-lib. Used for single-lib runs."""
    with tempfile.TemporaryDirectory(prefix=f"gsi_{lib}_") as td:
        dest = Path(td)
        print(f"  [{lib}] extracting subtrees {spec['subtrees']} "
              f"(cap {max_files} files)...", file=sys.stderr, flush=True)
        n_extracted = extract_subtrees(
            tarball, spec["subtrees"], dest, max_files=max_files,
            member_cap=member_cap,
            path_markers=spec.get("include_path_markers"),
            extensionless=bool(spec.get("index_extensionless_headers")),
        )
        if n_extracted == 0:
            raise RuntimeError(
                f"[{lib}] extracted 0 files from subtrees {spec['subtrees']} — "
                f"subtree name(s) wrong or absent from tarball (RULE #1)."
            )
        print(f"  [{lib}] extracted {n_extracted} files; indexing...",
              file=sys.stderr, flush=True)
        return index_lib_from_dir(lib, spec, dest, store,
                                  workers=workers, std_arg=std_arg,
                                  parse_timeout_s=parse_timeout_s)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tarball", default=str(DEFAULT_TARBALL),
                   help="Path to the corpus tarball (zstd).")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Path to the global symbol SQLite store.")
    p.add_argument("--libs", default="A1",
                   help="Comma list of lib keys to index, or 'A1'/'A2'/'all'. "
                        "(default A1 — the tractable high-value set).")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                   help="Parallel parse workers.")
    p.add_argument(
        "--parse-timeout-seconds",
        type=float,
        default=PARSE_TIMEOUT_S,
        help=(
            "Hard wall-clock deadline for one libclang source parse "
            f"(default {PARSE_TIMEOUT_S}s). Increase explicitly for unusually "
            "heavy template translation units; timeouts remain fail-closed."
        ),
    )
    p.add_argument("--max-files-per-lib", type=int, default=MAX_FILES_PER_LIB_DEFAULT,
                   help=f"Cap files indexed per base-lib (default {MAX_FILES_PER_LIB_DEFAULT}).")
    p.add_argument("--std-arg", default="-std=c++17",
                   help="C++ std flag for parsing (default -std=c++17).")
    p.add_argument("--member-cap", type=int, default=None,
                   help="Stop streaming the tarball after N members (debug/bounded).")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-index libs even if already marked done.")
    p.add_argument("--per-lib-pass", action="store_true",
                   help="Stream the tarball ONCE PER LIB instead of one shared "
                        "pass (slower; for targeted single-lib debugging).")
    p.add_argument("--keep-extract", action="store_true",
                   help="Keep the extracted source dirs after indexing (debug).")
    p.add_argument("--work-dir", default=None,
                   help="Staging root for extracted source (default: a fresh "
                        "mkdtemp deleted after the run).")
    p.add_argument("--extract-cache-dir", default=None,
                   help="Persistent source extraction cache. When set, the "
                        "tarball is streamed only for libs missing from the "
                        "cache; parser/indexer code still re-runs every build.")
    p.add_argument("--source-cache-dir", default=None,
                   help="Complete streaming-conveyor code source cache. Requires "
                        "--extract-cache-dir and hardlinks missing parser cache "
                        "inputs instead of re-reading the tarball.")
    p.add_argument("--libclang-path", default=None)
    p.add_argument("--report-only", action="store_true",
                   help="Print store counts/cost and exit (no indexing).")
    return p.parse_args(argv)


def resolve_libs(spec: str) -> list[str]:
    spec = spec.strip()
    if spec.lower() == "all":
        return list(BASE_LIBS.keys())
    if spec.upper() == "A1":
        return list(A1_LIBS)
    if spec.upper() == "A2":
        return list(A2_LIBS)
    libs = [x.strip() for x in spec.split(",") if x.strip()]
    for lk in libs:
        if lk not in BASE_LIBS:
            raise SystemExit(f"unknown lib key {lk!r}; known: {list(BASE_LIBS)}")
    return libs


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ip._configure_libclang(args.libclang_path)
    if args.extract_cache_dir and args.work_dir:
        raise SystemExit("--extract-cache-dir and --work-dir are mutually exclusive")
    if args.source_cache_dir and not args.extract_cache_dir:
        raise SystemExit("--source-cache-dir requires --extract-cache-dir")
    if args.source_cache_dir and args.member_cap is not None:
        raise SystemExit("--source-cache-dir cannot be combined with --member-cap")
    if args.parse_timeout_seconds <= 0:
        raise SystemExit("--parse-timeout-seconds must be positive")

    tarball = Path(args.tarball)
    store = GlobalSymbolStore(args.output, read_only=args.report_only)

    if args.report_only:
        _print_report(store)
        store.close()
        return 0

    libs = resolve_libs(args.libs)
    print(f"Building GLOBAL symbol index at {args.output}", file=sys.stderr)
    print(f"  tarball : {tarball}", file=sys.stderr)
    print(f"  libs    : {libs}", file=sys.stderr)
    print(f"  workers : {args.workers}  max_files/lib: {args.max_files_per_lib}",
          file=sys.stderr)

    resume = not args.no_resume
    todo = [lib for lib in libs if not (resume and store.lib_done(lib))]
    for lib in libs:
        if lib not in todo:
            print(f"  SKIP (done) {lib}", file=sys.stderr)
    if not todo:
        print("All requested libs already done.", file=sys.stderr)
        _print_report(store)
        store.close()
        return 0

    # Invalidate every selected generation before extraction starts. A tar/cache
    # failure must not leave an older generation marked complete for this run.
    for lib in todo:
        spec = BASE_LIBS[lib]
        store.invalidate_lib(lib, spec["tier"], spec["subtrees"])

    results: list[LibResult] = []

    def _finish(lib: str, _spec: dict, res: LibResult) -> None:
        results.append(res)
        print(f"  DONE {lib}: files={res.n_files} funcs={res.n_funcs} "
              f"types={res.n_types} errors={res.n_errors} "
              f"elapsed={res.elapsed_s:.1f}s", file=sys.stderr, flush=True)

    # Persistent cache: extract source once, re-index many times. This is the
    # path for full A1/A2 rebuilds while std/C++20/23/26 parser work evolves.
    if args.extract_cache_dir:
        extract_root = Path(args.extract_cache_dir)
        todo_specs = {lib: BASE_LIBS[lib] for lib in todo}
        print(f"\n=== extraction cache for {todo} -> {extract_root} ===",
              file=sys.stderr, flush=True)
        t_ext = time.time()
        if args.source_cache_dir:
            populate_extraction_cache_from_source_cache(
                tarball, todo_specs, Path(args.source_cache_dir), extract_root,
                max_files=args.max_files_per_lib,
            )
        dirs, counts = prepare_extraction_cache(
            tarball, todo_specs, extract_root,
            max_files=args.max_files_per_lib, member_cap=args.member_cap,
        )
        print(f"  extraction cache ready in {time.time()-t_ext:.1f}s: {counts}",
              file=sys.stderr, flush=True)
        for lib in todo:
            spec = BASE_LIBS[lib]
            if counts.get(lib, 0) == 0:
                raise RuntimeError(
                    f"[{lib}] extracted 0 files from {spec['subtrees']} — "
                    f"subtree name(s) wrong/absent (RULE #1: fail loud)."
                )
            print(f"\n=== indexing base-lib: {lib} (tier {spec['tier']}) ===",
                  file=sys.stderr, flush=True)
            res = index_lib_from_dir(
                lib, spec, dirs[lib], store,
                workers=args.workers, std_arg=args.std_arg,
                parse_timeout_s=args.parse_timeout_seconds,
            )
            _finish(lib, spec, res)
    # ONE tarball pass extracts ALL todo libs to a shared staging root (the
    # 235 GiB tarball is streamed ONCE, not once per lib), unless --per-lib-pass.
    elif len(todo) > 1 and not args.per_lib_pass:
        extract_root = Path(args.work_dir) if args.work_dir else Path(
            tempfile.mkdtemp(prefix="gsi_extract_"))
        extract_root.mkdir(parents=True, exist_ok=True)
        try:
            todo_specs = {lib: BASE_LIBS[lib] for lib in todo}
            print(f"\n=== single-pass extraction of {todo} -> {extract_root} ===",
                  file=sys.stderr, flush=True)
            t_ext = time.time()
            counts = extract_many_subtrees(
                tarball, todo_specs, extract_root,
                max_files=args.max_files_per_lib, member_cap=args.member_cap,
            )
            print(f"  extraction done in {time.time()-t_ext:.1f}s: {counts}",
                  file=sys.stderr, flush=True)
            for lib in todo:
                spec = BASE_LIBS[lib]
                if counts.get(lib, 0) == 0:
                    raise RuntimeError(
                        f"[{lib}] extracted 0 files from {spec['subtrees']} — "
                        f"subtree name(s) wrong/absent (RULE #1: fail loud)."
                    )
                print(f"\n=== indexing base-lib: {lib} (tier {spec['tier']}) ===",
                      file=sys.stderr, flush=True)
                res = index_lib_from_dir(
                    lib, spec, extract_root / lib, store,
                    workers=args.workers, std_arg=args.std_arg,
                    parse_timeout_s=args.parse_timeout_seconds,
                )
                _finish(lib, spec, res)
                if not args.keep_extract:
                    import shutil
                    shutil.rmtree(extract_root / lib, ignore_errors=True)
        finally:
            if not args.keep_extract and not args.work_dir:
                import shutil
                shutil.rmtree(extract_root, ignore_errors=True)
    else:
        for lib in todo:
            spec = BASE_LIBS[lib]
            print(f"\n=== indexing base-lib: {lib} (tier {spec['tier']}) ===",
                  file=sys.stderr, flush=True)
            res = index_one_lib(
                lib, spec, tarball, store,
                workers=args.workers, max_files=args.max_files_per_lib,
                std_arg=args.std_arg, member_cap=args.member_cap,
                parse_timeout_s=args.parse_timeout_seconds,
            )
            _finish(lib, spec, res)

    print(f"\n{'='*64}", file=sys.stderr)
    _print_report(store)
    store.close()
    return 0


def _print_report(store: GlobalSymbolStore) -> None:
    counts = store.counts()
    lib_stats = store.lib_stats()
    report = {
        "store_path": store.path,
        "per_lib_symbol_counts": counts,
        "lib_index_stats": lib_stats,
        "totals": {
            "funcs": sum(c.get("func", 0) for c in counts.values()),
            "types": sum(c.get("type", 0) for c in counts.values()),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
