from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from cppmega.data.commit_scope import classify_primary_commit_path
from scripts.nanochat_data.extract_git_history import (
    get_commit_cpp_files,
    should_skip_path,
)


def test_commit_scope_keeps_cpp_tests_and_benchmarks() -> None:
    assert not should_skip_path("tests/parser/sql_parser_test.cpp")
    assert not should_skip_path("testing/runtime/allocator_test.cc")
    assert not should_skip_path("unittest/compiler/diagnostic_test.c")
    assert not should_skip_path("benchmarks/query_engine/scan_benchmark.cpp")


def test_commit_scope_still_excludes_vendored_and_generated_sources() -> None:
    assert should_skip_path("third_party/sqlite/sqlite3.c")
    assert should_skip_path("vendor/library/source.cpp")
    assert should_skip_path("generated/protocol.pb.cc")


def test_commit_scope_routes_only_primary_native_domains() -> None:
    expected = {
        "src/main.cpp": "cpp",
        "include/templates.tcc": "cpp",
        "kernels/attention.cu": "cpp",
        "schema/migrate.sql": "sql",
        "CMakeLists.txt": "cmake",
        "Makefile": "make",
        "build.ninja": "ninja",
        "BUILD.bazel": "bazel",
        "meson.build": "meson",
        "BUILD.gn": "gn",
        "SConstruct": "scons",
        "xmake.lua": "xmake",
        "compile_commands.json": "compile_commands",
        "Dockerfile": "dockerfile",
        "scripts/build.sh": "sh",
        "scripts/test.ps1": "powershell",
        "scripts/run.cmd": "cmd",
        "conanfile.py": "conan",
        "vcpkg.json": "vcpkg",
    }
    assert {
        path: classify_primary_commit_path(path) for path in expected
    } == expected
    assert classify_primary_commit_path("tools/helper.py") is None
    assert classify_primary_commit_path("web/build.js") is None
    assert classify_primary_commit_path("README.md") is None


def test_extractor_membership_uses_primary_native_scope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Scope Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "scope@example.invalid"],
        check=True,
    )
    files = {
        "schema.sql": "CREATE TABLE builds (id INTEGER PRIMARY KEY);\n",
        "CMakeLists.txt": "cmake_minimum_required(VERSION 3.25)\nproject(scope)\n",
        "one.cpp": "int one() { return 1; }\n",
        "two.c": "int two(void) { return 2; }\n",
        "three.h": "int three(void);\n",
        "four.cu": "__global__ void four() {}\n",
        "helper.py": "print('not primary')\n",
        "build.js": "console.log('not primary');\n",
    }
    for name, text in files.items():
        (repo / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)
    for name, text in files.items():
        (repo / name).write_text(text + "# changed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "modify"], check=True)
    commit_hash = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert set(get_commit_cpp_files(str(repo), commit_hash) or []) == {
        "CMakeLists.txt",
        "four.cu",
        "one.cpp",
        "schema.sql",
        "three.h",
        "two.c",
    }


@pytest.mark.native_toolchain
def test_sql_commit_uses_domain_sidecars_without_libclang(tmp_path: Path) -> None:
    from tools.clang_indexer.process_commits import process_record

    old_content = (
        "CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL);\n"
        "CREATE INDEX builds_status_idx ON builds(status);\n"
    )
    new_content = (
        "CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL, "
        "platform TEXT NOT NULL);\n"
        "CREATE INDEX builds_status_idx ON builds(status);\n"
    )
    diff = (
        "diff --git a/schema.sql b/schema.sql\n"
        "--- a/schema.sql\n"
        "+++ b/schema.sql\n"
        "@@ -1,2 +1,2 @@\n"
        "-CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL);\n"
        "+CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL, "
        "platform TEXT NOT NULL);\n"
        " CREATE INDEX builds_status_idx ON builds(status);\n"
    )
    documents = process_record(
        {
            "repo": "tests/commit-domain",
            "filepath": "schema.sql",
            "commit_hash": "a" * 40,
            "subject": "Track the build platform",
            "body": "Keep CI receipts attributable to their platform.",
            "old_content": old_content,
            "new_content": new_content,
            "diff": diff,
        },
        None,
        str(tmp_path),
        4096,
        200_000,
        "both",
        5,
    )

    assert len(documents) == 1
    document = documents[0]
    assert document["text"] == new_content
    assert document["doc_type"] == "sql"
    assert document["domain_parse_info"]["parser"] == "sql-lexical"
    assert len(document["domain_ids"]) == len(new_content)
    assert document["pre_text"] == old_content
    assert document["post_text"] == new_content
    assert document["diff_text"] == diff
    assert document["commit_msg_text"].startswith("Track the build platform")
    assert any(document["change_mask_post"])


@pytest.mark.native_toolchain
def test_short_domain_commit_uses_the_shared_fifty_char_gate(tmp_path: Path) -> None:
    from tools.clang_indexer.process_commits import process_record

    old_content = "SELECT id, status FROM builds WHERE status = 'queued';\n"
    new_content = "SELECT id, status FROM builds WHERE status = 'passed';\n"
    diff = (
        "diff --git a/query.sql b/query.sql\n"
        "--- a/query.sql\n"
        "+++ b/query.sql\n"
        "@@ -1 +1 @@\n"
        f"-{old_content}"
        f"+{new_content}"
    )

    documents = process_record(
        {
            "repo": "tests/commit-domain",
            "filepath": "query.sql",
            "commit_hash": "b" * 40,
            "old_content": old_content,
            "new_content": new_content,
            "diff": diff,
        },
        None,
        str(tmp_path),
        4096,
        200_000,
        "both",
        5,
    )

    assert 50 <= len(new_content) < 100
    assert len(documents) == 1
    assert documents[0]["text"] == new_content


@pytest.mark.native_toolchain
def test_domain_commit_dedup_keeps_distinct_incremental_changes(
    tmp_path: Path,
) -> None:
    from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
    from tools.clang_indexer.dedup_store import DedupStore
    from tools.clang_indexer.process_commits import process_jsonl_file

    states = [
        (
            "CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL);\n"
            "CREATE INDEX builds_status_idx ON builds(status);\n"
        ),
        (
            "CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL, "
            "platform TEXT NOT NULL);\n"
            "CREATE INDEX builds_status_idx ON builds(status);\n"
        ),
        (
            "CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL, "
            "platform TEXT NOT NULL, runner TEXT NOT NULL);\n"
            "CREATE INDEX builds_status_idx ON builds(status);\n"
        ),
    ]
    records = []
    for index, (old_content, new_content) in enumerate(zip(states, states[1:])):
        diff = (
            "diff --git a/schema.sql b/schema.sql\n"
            "--- a/schema.sql\n"
            "+++ b/schema.sql\n"
            "@@ -1,2 +1,2 @@\n"
            f"-{old_content.splitlines()[0]}\n"
            f"+{new_content.splitlines()[0]}\n"
            " CREATE INDEX builds_status_idx ON builds(status);\n"
        )
        records.append(
            {
                "repo": "tests/commit-domain",
                "filepath": "schema.sql",
                "commit_hash": f"{index + 1:040x}",
                "subject": f"Change schema path {index}",
                "old_content": old_content,
                "new_content": new_content,
                "diff": diff,
            }
        )
    input_path = tmp_path / "domain-commits.jsonl"
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output = io.StringIO()
    dedup_store = DedupStore(str(tmp_path / "dedup.sqlite"), near=True)

    try:
        stats = process_jsonl_file(
            str(input_path),
            output,
            None,
            str(tmp_path),
            4096,
            200_000,
            "both",
            5,
            1.0,
            dedup_store=dedup_store,
            dedup_tokenizer=load_cppmega_tokenizer("cppmega/tokenizer"),
            analysis_cache_entries=0,
        )
    finally:
        dedup_store.close()

    assert stats["documents_written"] == 2
    assert stats["records_skipped"] == 0
    assert len(output.getvalue().splitlines()) == 2
