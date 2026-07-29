from __future__ import annotations

from scripts.nanochat_data.extract_git_history import should_skip_path


def test_commit_scope_keeps_cpp_tests_and_benchmarks() -> None:
    assert not should_skip_path("tests/parser/sql_parser_test.cpp")
    assert not should_skip_path("testing/runtime/allocator_test.cc")
    assert not should_skip_path("unittest/compiler/diagnostic_test.c")
    assert not should_skip_path("benchmarks/query_engine/scan_benchmark.cpp")


def test_commit_scope_still_excludes_vendored_and_generated_sources() -> None:
    assert should_skip_path("third_party/sqlite/sqlite3.c")
    assert should_skip_path("vendor/library/source.cpp")
    assert should_skip_path("generated/protocol.pb.cc")
