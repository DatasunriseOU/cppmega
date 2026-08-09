from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from cppmega.data.commit_scope import classify_primary_commit_path
from scripts.nanochat_data.extract_git_history import (
    get_commit_diffs,
    get_commit_file_changes,
    get_commit_list,
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
    assert classify_primary_commit_path("scripts/backup.sh") is None
    assert classify_primary_commit_path("tools/release.ps1") is None


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


def test_extractor_keeps_added_deleted_renamed_large_and_shebang_changes(
    tmp_path: Path,
) -> None:
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
    (repo / "deleted.cpp").write_text(
        "int removed_value() { return 123456; }\n" * 3,
        encoding="utf-8",
    )
    (repo / "old_name.cpp").write_text(
        "int renamed_value() { return 123456; }\n" * 3,
        encoding="utf-8",
    )
    (repo / "large.cpp").write_text(
        "".join(
            f"int before_{index:04d}() {{ return {index}; }}\n"
            for index in range(2_500)
        ),
        encoding="utf-8",
    )
    (repo / "run_checks").write_text(
        "#!/bin/sh\nset -eu\ncmake -S . -B build\ncmake --build build\nctest --test-dir build\n",
        encoding="utf-8",
    )
    (repo / "README").write_text(
        "This extensionless prose file is deliberately outside the primary corpus.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)
    initial_hash = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    (repo / "deleted.cpp").unlink()
    subprocess.run(
        ["git", "-C", str(repo), "mv", "old_name.cpp", "renamed.cpp"],
        check=True,
    )
    (repo / "renamed.cpp").write_text(
        "int renamed_value() { return 123456; }\n" * 2
        + "int renamed_value() { return 654321; }\n",
        encoding="utf-8",
    )
    (repo / "large.cpp").write_text(
        "".join(
            f"int after_{index:04d}() {{ return {index + 1}; }}\n"
            for index in range(2_500)
        ),
        encoding="utf-8",
    )
    (repo / "run_checks").write_text(
        "#!/bin/sh\nset -eux\ncmake -S . -B build\ncmake --build build --parallel\n"
        "ctest --test-dir build --output-on-failure\n",
        encoding="utf-8",
    )
    (repo / "schema.sql").write_text(
        "CREATE TABLE build_receipts (id INTEGER PRIMARY KEY, status TEXT NOT NULL);\n",
        encoding="utf-8",
    )
    (repo / "README").write_text(
        "This changed extensionless prose file remains outside the primary corpus.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "exercise all file statuses"],
        check=True,
    )
    commit_hash = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    changes = get_commit_file_changes(str(repo), commit_hash) or []
    assert {change["status"] for change in changes} >= {"A", "D", "M", "R"}
    diffs = get_commit_diffs(str(repo), commit_hash) or []
    by_path = {item["filepath"]: item for item in diffs}
    assert set(by_path) == {
        "deleted.cpp",
        "large.cpp",
        "renamed.cpp",
        "run_checks",
        "schema.sql",
    }
    assert by_path["schema.sql"]["change_status"] == "A"
    assert by_path["schema.sql"]["old_content"] == ""
    assert by_path["deleted.cpp"]["change_status"] == "D"
    assert by_path["deleted.cpp"]["new_content"] == ""
    assert by_path["renamed.cpp"]["change_status"] == "R"
    assert by_path["renamed.cpp"]["old_filepath"] == "old_name.cpp"
    assert by_path["run_checks"]["new_content"].startswith("#!/bin/sh")
    assert len(by_path["large.cpp"]["diff"]) > 50_000
    assert commit_hash in get_commit_list(str(repo))
    root_diffs = get_commit_diffs(str(repo), initial_hash) or []
    assert root_diffs
    assert all(item["change_status"] == "A" for item in root_diffs)
    assert all(item["old_content"] == "" for item in root_diffs)
    assert "README" not in {item["filepath"] for item in root_diffs}


def test_extractor_ignores_gitlink_changes_without_reading_submodule_blob(
    tmp_path: Path,
) -> None:
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
    (repo / "main.cpp").write_text(
        "int before_value() { return 123456; }\n" * 3,
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "add", "main.cpp"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)

    (repo / "main.cpp").write_text(
        "int after_value() { return 654321; }\n" * 3,
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "add", "main.cpp"], check=True)
    submodule_cacheinfo = (
        "160000,"
        + "a" * 40
        + ",tests/cbmc/aws-templates-for-cbmc-proofs"
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "update-index",
            "--add",
            "--cacheinfo",
            submodule_cacheinfo,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "add submodule"],
        check=True,
    )
    commit_hash = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    changes = get_commit_file_changes(str(repo), commit_hash) or []
    assert "tests/cbmc/aws-templates-for-cbmc-proofs" not in {
        change["new_filepath"] or change["old_filepath"] for change in changes
    }
    assert {change["new_filepath"] for change in changes} == {"main.cpp"}

    (repo / "main.cpp").write_text(
        "int final_value() { return 111222; }\n" * 3,
        encoding="utf-8",
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "update-index",
            "--force-remove",
            "tests/cbmc/aws-templates-for-cbmc-proofs",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "main.cpp"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "remove submodule"],
        check=True,
    )
    delete_commit_hash = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    delete_changes = get_commit_file_changes(str(repo), delete_commit_hash) or []
    assert "tests/cbmc/aws-templates-for-cbmc-proofs" not in {
        change["new_filepath"] or change["old_filepath"]
        for change in delete_changes
    }
    assert {change["new_filepath"] for change in delete_changes} == {"main.cpp"}


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
def test_added_and_deleted_sql_commits_keep_directional_sidecars(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer.process_commits import process_record

    sql = (
        "CREATE TABLE build_receipts (id INTEGER PRIMARY KEY, status TEXT NOT NULL);\n"
        "CREATE INDEX build_receipts_status_idx ON build_receipts(status);\n"
    )
    added = process_record(
        {
            "repo": "tests/commit-domain",
            "filepath": "schema.sql",
            "old_filepath": "",
            "change_status": "A",
            "commit_hash": "c" * 40,
            "old_content": "",
            "new_content": sql,
            "diff": (
                "diff --git a/schema.sql b/schema.sql\n"
                "new file mode 100644\n"
                "--- /dev/null\n"
                "+++ b/schema.sql\n"
                "@@ -0,0 +1,2 @@\n"
                f"+{sql.splitlines()[0]}\n"
                f"+{sql.splitlines()[1]}\n"
            ),
        },
        None,
        str(tmp_path),
        4096,
        200_000,
        "both",
        5,
    )
    deleted = process_record(
        {
            "repo": "tests/commit-domain",
            "filepath": "schema.sql",
            "old_filepath": "schema.sql",
            "change_status": "D",
            "commit_hash": "d" * 40,
            "old_content": sql,
            "new_content": "",
            "diff": (
                "diff --git a/schema.sql b/schema.sql\n"
                "deleted file mode 100644\n"
                "--- a/schema.sql\n"
                "+++ /dev/null\n"
                "@@ -1,2 +0,0 @@\n"
                f"-{sql.splitlines()[0]}\n"
                f"-{sql.splitlines()[1]}\n"
            ),
        },
        None,
        str(tmp_path),
        4096,
        200_000,
        "both",
        5,
    )

    assert len(added) == 1
    assert added[0]["text"] == sql
    assert any(added[0]["change_mask_post"])
    assert not any(added[0]["change_mask_pre"])
    assert len(deleted) == 1
    assert deleted[0]["text"] == sql
    assert any(deleted[0]["change_mask_pre"])
    assert not any(deleted[0]["change_mask_post"])


@pytest.mark.native_toolchain
def test_added_and_deleted_cpp_commits_keep_directional_sidecars(
    tmp_path: Path,
) -> None:
    import tools.clang_indexer.process_commits as commit_processor

    commit_processor._configure_libclang(None)
    clang_index = commit_processor.Index.create()
    source = (
        "int build_receipt_status(int build_id) {\n"
        "    return build_id > 0 ? 1 : 0;\n"
        "}\n"
    )
    added = commit_processor.process_record(
        {
            "repo": "tests/commit-cpp",
            "filepath": "receipt.cpp",
            "change_status": "A",
            "commit_hash": "e" * 40,
            "old_content": "",
            "new_content": source,
            "diff": (
                "diff --git a/receipt.cpp b/receipt.cpp\n"
                "new file mode 100644\n"
                "--- /dev/null\n"
                "+++ b/receipt.cpp\n"
                "@@ -0,0 +1,3 @@\n"
                f"+{source.splitlines()[0]}\n"
                f"+{source.splitlines()[1]}\n"
                f"+{source.splitlines()[2]}\n"
            ),
        },
        clang_index,
        str(tmp_path),
        4096,
        200_000,
        "chain",
        5,
    )
    deleted = commit_processor.process_record(
        {
            "repo": "tests/commit-cpp",
            "filepath": "receipt.cpp",
            "old_filepath": "receipt.cpp",
            "change_status": "D",
            "commit_hash": "f" * 40,
            "old_content": source,
            "new_content": "",
            "diff": (
                "diff --git a/receipt.cpp b/receipt.cpp\n"
                "deleted file mode 100644\n"
                "--- a/receipt.cpp\n"
                "+++ /dev/null\n"
                "@@ -1,3 +0,0 @@\n"
                f"-{source.splitlines()[0]}\n"
                f"-{source.splitlines()[1]}\n"
                f"-{source.splitlines()[2]}\n"
            ),
        },
        clang_index,
        str(tmp_path),
        4096,
        200_000,
        "chain",
        5,
    )

    assert len(added) == 1
    assert any(added[0]["change_mask_post"])
    assert not any(added[0]["change_mask_pre"])
    assert len(deleted) == 1
    assert any(deleted[0]["change_mask_pre"])
    assert not any(deleted[0]["change_mask_post"])


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
def test_domain_commit_dedup_keeps_distinct_changes_with_same_post_state(
    tmp_path: Path,
) -> None:
    from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
    from tools.clang_indexer.dedup_store import DedupStore
    from tools.clang_indexer.process_commits import process_jsonl_file

    old_states = [
        (
            "CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL);\n"
            "CREATE INDEX builds_status_idx ON builds(status);\n"
        ),
        (
            "CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL, "
            "runner TEXT);\n"
            "CREATE INDEX builds_status_idx ON builds(status);\n"
        ),
    ]
    new_content = (
        "CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL, "
        "platform TEXT NOT NULL);\n"
        "CREATE INDEX builds_status_idx ON builds(status);\n"
    )
    records = []
    for index, old_content in enumerate(old_states):
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
