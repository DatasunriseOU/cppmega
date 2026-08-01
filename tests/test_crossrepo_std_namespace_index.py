"""C2 regression — the GLOBAL std cross-link index must contain REAL ``std::``
symbols, not gcc compiler / libiberty internals.

Background (outputs/review/deep_review_24h.md §3, finding C2)
------------------------------------------------------------
The live ``base_lib='std'`` index was 100% gcc libiberty / compiler-internal
noise (0 rows with a ``std::`` qname) because of THREE root causes:

  1. ``namespace_prefixes`` was declared per base-lib but never enforced, so
     non-``std::`` gcc symbols were admitted under ``base_lib='std'``.
  2. The inline ABI namespaces libstdc++/libc++ use (``std::__1::`` /
     ``std::__cxx11::``) were not normalized, so even correctly extracted std
     symbols stored/looked-up under the wrong key.
  3. SUBTREE/FILE SELECTION: ``std`` pulls the FULL gcc-mirror / llvm-project
     monorepos. With no path restriction the per-lib file cap is spent on gcc's
     compiler + libiberty tree and ZERO real C++ standard-library headers get
     indexed — and the libstdc++/libc++ public headers are EXTENSIONLESS
     (<vector>, <type_traits>, ...) so the extension filter skipped them too.

These tests use REAL libclang on a tiny synthetic std-like tree (no mocks, no
tarball) and assert that real ``std::`` qnames are captured (including from an
EXTENSIONLESS libc++ header, with the ``__1`` inline namespace normalized away)
while gcc-internal qnames — even when they live in a header under an
``include/`` dir — are rejected.

RULE #1: no mocks of the real store / parser — a real SQLite store and real
libclang parse path are exercised end to end.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

MLX_ROOT = Path(__file__).resolve().parents[1]
for _p in (MLX_ROOT, MLX_ROOT / "tools" / "clang_indexer"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_builder():
    from scripts.crossrepo import build_global_symbol_index as b  # type: ignore
    return b


def _require_libclang():
    try:
        import index_project as ip  # type: ignore  # noqa: F401
        import clang.cindex  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment without libclang
        pytest.skip(f"libclang unavailable: {exc}")


# --------------------------------------------------------------------------- #
# Root cause #3 — SUBTREE / FILE selection (pure, fast).
# --------------------------------------------------------------------------- #
def test_member_selection_pins_std_to_stdlib_headers():
    """``_member_is_wanted`` (driven by BASE_LIBS['std']'s include_path_markers +
    index_extensionless_headers) admits ONLY the C++ standard-library header trees
    — including EXTENSIONLESS public headers — and rejects gcc compiler /
    libiberty internals even when they live in a ``.h`` under an ``include/`` dir.
    """
    b = _load_builder()
    import index_project as ip  # type: ignore

    std = b.BASE_LIBS["std"]
    markers = std["include_path_markers"]
    extless = std["index_extensionless_headers"]
    assert markers, "std must declare include_path_markers (C2 root cause #3)"
    assert extless is True, "std must index extensionless headers (<vector>)"

    def wanted(name: str) -> bool:
        return b._member_is_wanted(
            name, ip.INDEX_EXTENSIONS, path_markers=markers, extensionless=extless)

    # Real C++ standard-library public headers — MUST be admitted.
    assert wanted("cpp_all/gcc-mirror/libstdc++-v3/include/bits/stl_vector.h")
    assert wanted("cpp_all/gcc-mirror/libstdc++-v3/include/vector")       # extensionless
    assert wanted("cpp_all/gcc-mirror/libstdc++-v3/libsupc++/typeinfo")   # extensionless
    assert wanted("cpp_all/llvm-project/libcxx/include/deque")            # extensionless
    assert wanted("cpp_all/STL/stl/inc/memory")                          # extensionless

    # gcc / llvm COMPILER + libiberty internals — MUST be rejected. The libiberty
    # header is the killer case: it IS a ``.h`` under an ``include/`` dir, so only
    # the path-marker restriction (not the extension/header check) excludes it.
    assert not wanted("cpp_all/gcc-mirror/libiberty/include/demangle.h")
    assert not wanted("cpp_all/gcc-mirror/libiberty/cp-demangle.c")
    assert not wanted("cpp_all/gcc-mirror/gcc/cp/parser.cc")
    assert not wanted("cpp_all/llvm-project/clang/lib/Sema/SemaDecl.cpp")
    assert not wanted("cpp_all/llvm-project/llvm/lib/IR/Function.cpp")

    # Extensionless doc/build files under include/ are NOT headers.
    assert not wanted("cpp_all/gcc-mirror/libstdc++-v3/include/bits/README")
    assert not wanted("cpp_all/llvm-project/libcxx/include/LICENSE")

    # libc (no namespace_prefixes, no extensionless) keeps by-name .c/.h symbols
    # but does NOT slurp extensionless files.
    libc = b.BASE_LIBS["libc"]
    assert b._member_is_wanted(
        "cpp_all/glibc/stdlib/malloc.c", ip.INDEX_EXTENSIONS,
        path_markers=libc.get("include_path_markers"),
        extensionless=bool(libc.get("index_extensionless_headers")))
    assert not b._member_is_wanted(
        "cpp_all/glibc/include/some_extensionless_blob", ip.INDEX_EXTENSIONS,
        path_markers=libc.get("include_path_markers"),
        extensionless=bool(libc.get("index_extensionless_headers")))


def test_global_store_reopens_external_and_location_identities(tmp_path):
    b = _load_builder()
    external_file = b.ip.canonical_external_provider_file(
        "boost",
        "boost/move.hpp",
        source="test global store",
    )
    external_key = b.ip.canonical_symbol_identity(
        qname="boost::move",
        kind="FUNCTION_DECL",
        usr="c:@N@boost@F@move#",
        canonical_signature="display=move()",
        project="boostorg/boost",
        provider="boost",
        include_provenance="boost/move.hpp",
    )
    location_key = b.ip.canonical_symbol_identity(
        qname="local::anonymous",
        kind="STRUCT_DECL",
        project="owner/repo",
        file="src/local.cpp",
        line=31,
        column=9,
        repo_file_location_fallback=True,
    )
    db = str(tmp_path / "global.sqlite")
    store = b.GlobalSymbolStore(db)
    store.insert_symbols(
        [
            b.GlobalSymbolRecord(
                qname="boost::move",
                base_lib="boost",
                base_repo="boostorg/boost",
                kind=2,
                sym_type="func",
                file=external_file,
                line=7,
                column=4,
                end_line=9,
                is_public=1,
                token_est=8,
                body_len=32,
                text="void boost::move() { int value = 1; }",
                symbol_key=external_key,
                usr="c:@N@boost@F@move#",
                canonical_signature="display=move()",
                symbol_kind="FUNCTION_DECL",
                provider="boost",
                include_provenance="boost/move.hpp",
            ),
            b.GlobalSymbolRecord(
                qname="local::anonymous",
                base_lib="local",
                base_repo="owner/repo",
                kind=4,
                sym_type="type",
                file="src/local.cpp",
                line=31,
                column=9,
                end_line=33,
                is_public=1,
                token_est=8,
                body_len=32,
                text="struct anonymous { int value; };",
                symbol_key=location_key,
                symbol_kind="STRUCT_DECL",
                provider="owner/repo",
                include_provenance="src/local.cpp",
            ),
        ]
    )
    store.conn.commit()
    store.close()

    b.GlobalSymbolStore(db).close()
    reader = b.ip.GlobalSymbolReader(db)
    try:
        assert reader.lookup_candidates("boost::move")[0]["column"] == 4
    finally:
        reader.close()

    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE symbols SET column=10 WHERE symbol_key=?",
        (location_key,),
    )
    conn.commit()
    conn.close()
    with pytest.raises(b.ip.SymbolIdentityError, match="serialized fields"):
        b.GlobalSymbolStore(db)


# --------------------------------------------------------------------------- #
# End-to-end — REAL libclang parse of a synthetic std-like tree.
# --------------------------------------------------------------------------- #
_STL_VECTOR_H = """\
namespace std {
  template<class T>
  class vector {
   public:
    void push_back(const T& value) {
      int local_counter = 0;
      local_counter = local_counter + 1;
      local_counter = local_counter + 1;
      (void)value;
      (void)local_counter;
    }
  };
}
"""

# libc++ public header: EXTENSIONLESS, and wraps symbols in the ``__1`` inline
# ABI namespace -> stored key must normalize to std::deque::clear.
_LIBCXX_DEQUE = """\
namespace std {
  inline namespace __1 {
    template<class T>
    class deque {
     public:
      void clear() {
        int reset_state = 0;
        reset_state = reset_state + 1;
        reset_state = reset_state + 2;
        reset_state = reset_state + 3;
      }
    };
  }
}
"""

# gcc libiberty internal: a CLEAN, non-reserved global symbol living in a real
# header under an include/ dir. It passes the header + reserved-name checks; ONLY
# the namespace_prefixes=['std::'] enforcement keeps it out of base_lib='std'.
_LIBIBERTY_DEMANGLE_H = """\
struct demangle_component { int type; const char* name; };

int cplus_demangle_fill_name(struct demangle_component* p,
                             const char* s, int n) {
  int total = 0;
  total = total + n;
  total = total + (int)(s != 0);
  (void)p;
  return total;
}
"""


def _write(root: Path, rel: str, text: str) -> None:
    fp = root / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(text)


def test_index_std_tree_captures_std_and_rejects_gcc_internal(tmp_path):
    _require_libclang()
    b = _load_builder()

    dest = tmp_path / "std_src"
    _write(dest, "gcc-mirror/libstdc++-v3/include/bits/stl_vector.h", _STL_VECTOR_H)
    _write(dest, "llvm-project/libcxx/include/deque", _LIBCXX_DEQUE)  # extensionless
    _write(dest, "gcc-mirror/libiberty/include/demangle.h", _LIBIBERTY_DEMANGLE_H)

    db = str(tmp_path / "gsi.sqlite")
    store = b.GlobalSymbolStore(db)
    res = b.index_lib_from_dir(
        "std", b.BASE_LIBS["std"], dest, store, workers=1, std_arg="-std=c++17")
    store.close()

    # The extensionless libc++ header MUST have been discovered (root cause #3):
    # vector.h + demangle.h (2 .h) + the extensionless deque header.
    assert res.n_files >= 3, f"extensionless std header not discovered: {res.n_files}"

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT qname, sym_type FROM symbols WHERE base_lib='std'").fetchall()
    conn.close()
    qnames = {q for q, _ in rows}
    assert qnames, "no std symbols indexed at all"

    # (1) Real std:: symbols captured — including from the EXTENSIONLESS header.
    assert "std::vector::push_back" in qnames, qnames
    assert "std::deque::clear" in qnames, qnames

    # (2) Inline ABI namespace normalized away — never store std::__1::*.
    assert not any("__1" in q for q in qnames), sorted(qnames)
    assert not any("__cxx11" in q for q in qnames), sorted(qnames)

    # (3) namespace_prefixes enforced — gcc-internal qnames rejected even though
    # cplus_demangle_fill_name sits in a header under an include/ dir.
    assert "cplus_demangle_fill_name" not in qnames, sorted(qnames)
    assert "demangle_component" not in qnames, sorted(qnames)
    assert all(q.startswith("std::") for q in qnames), sorted(qnames)
