from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("standard", ("c++20", "c++23", "c++26"))
def test_compile_database_preserves_current_cpp_standards(
    tmp_path: Path,
    standard: str,
) -> None:
    from cppmega.data.build_context import detect_build_context

    source = tmp_path / "main.cpp"
    source.write_text("int main() { return 0; }\n", encoding="utf-8")
    (tmp_path / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(tmp_path),
                    "file": str(source),
                    "arguments": [
                        "clang++",
                        f"-std={standard}",
                        "-c",
                        str(source),
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    platform, default_args, compile_index = detect_build_context(str(tmp_path))

    assert compile_index is not None
    assert platform["standard"] == standard
    standard_args = [arg for arg in default_args if arg.startswith("-std=")]
    assert standard_args
    assert set(standard_args) == {f"-std={standard}"}


def test_domain_parsers_keep_build_shell_and_diagnostics_distinct() -> None:
    import numpy as np

    from cppmega.data.build_parsers import parse_cmake
    from cppmega.data.diagnostic_parsers import parse_clang_diagnostic
    from cppmega.data.domain_schema import DomainEdgeKind, DomainKind
    from cppmega.data.shell_parsers import parse_zsh

    cmake = parse_cmake("add_executable(app main.cpp)\n")
    shell = parse_zsh("cat input.txt | grep value > output.txt\n")
    diagnostic = parse_clang_diagnostic(
        "src/main.cpp:12:7: error: no matching function for call to 'foo'\n"
    )

    assert cmake.domain == DomainKind.CMAKE
    assert shell.domain == DomainKind.ZSH
    assert diagnostic.domain == DomainKind.COMPILER_ERROR
    assert any(
        kind == int(DomainEdgeKind.BUILD_TARGET_SOURCE)
        for _source, _target, kind in cmake.edges
    )
    assert any(
        kind == int(DomainEdgeKind.SHELL_PIPE)
        for _source, _target, kind in shell.edges
    )
    assert any(
        kind == int(DomainEdgeKind.DIAG_PRIMARY_LOCATION)
        for _source, _target, kind in diagnostic.edges
    )
    packet = cmake.to_packet()
    assert packet.token_axis_len == len(cmake.tokens)
    assert int(np.asarray(packet.domain_ids)[0]) == int(DomainKind.CMAKE)


def test_macro_scanner_ignores_comments_and_literals() -> None:
    from tools.clang_indexer.index_project import extract_macro_blocks

    source = r'''/* #define COMMENTED_OUT 1 */
const char *literal = "#define QUOTED 1";
#define ACTIVE(value) ((value) + 1)
'''

    blocks = extract_macro_blocks(source)

    assert [name for _start, _end, name, _text in blocks] == ["ACTIVE"]


def test_source_and_symbol_identity_registries_preserve_collision_witnesses() -> None:
    from cppmega.data.source_identity import (
        source_identity,
        validate_source_identity_registry,
    )
    from cppmega.data.symbol_identity import (
        SymbolIdentityError,
        SymbolIdentityRegistry,
        compute_symbol_id,
    )

    source = source_identity({"repo": "owner/project", "filepath": "src/a.cpp"})
    assert source.source_identity_id > 0
    assert validate_source_identity_registry(
        [source.as_dict()], referenced_ids=[source.source_identity_id]
    )[source.source_identity_id] == source

    registry = SymbolIdentityRegistry()
    key = "usr:schema=v3\x1fproject=owner/project\x1fusr=c:@F@run#"
    symbol_id = registry.register(key, source="core-test")
    assert symbol_id == compute_symbol_id(key)
    assert registry.records() == [{"symbol_id": symbol_id, "symbol_key": key}]
    with pytest.raises(SymbolIdentityError, match="canonical key"):
        registry.register(key, symbol_id=symbol_id + 1, source="bad-claim")


def test_data_symbol_identity_is_the_canonical_module_compatibility_surface() -> None:
    import cppmega.data.symbol_identity as compatibility
    import cppmega.symbol_identity as canonical

    shared_constants = (
        "SYMBOL_IDENTITY_SCHEMA_VERSION",
        "SYMBOL_ID_MAX",
    )
    shared_objects = (
        "SymbolIdentityError",
        "SymbolIdentityRegistry",
        "ResolvedProjectIdentity",
        "RepoFileLocationIdentity",
        "compute_symbol_id",
        "resolve_remote_project_identity",
    )
    for name in shared_constants:
        assert getattr(compatibility, name) == getattr(canonical, name)
    for name in shared_objects:
        assert getattr(compatibility, name) is getattr(canonical, name)

    registry = compatibility.SymbolIdentityRegistry()
    assert isinstance(registry, canonical.SymbolIdentityRegistry)
    key = "usr:schema=v3\x1fproject=owner/project\x1fusr=c:@F@shared#"
    symbol_id = registry.register(key, source="compatibility-test")
    assert registry.records() == [
        {"symbol_id": canonical.compute_symbol_id(key), "symbol_key": key}
    ]
    assert symbol_id == canonical.compute_symbol_id(key)


def test_commit_docstring_preserves_pr_provenance_and_discussion() -> None:
    from tools.clang_indexer.process_commits import build_docstring

    rendered = build_docstring(
        {
            "subject": "Fix template lookup (#1467)",
            "repo": "owner/project",
            "filepath": "include/widget.hpp",
            "commit_hash": "abc123",
            "pr_title": "Fix template lookup",
            "pr_discussion": "PR #1467: Fix template lookup\nReviewed and approved.",
            "body": (
                "Preserve dependent-name lookup.\n"
                "Reviewed-by: Ada\n"
                "Signed-off-by: Example <example@example.com>"
            ),
        }
    )

    assert "@repo owner/project" in rendered
    assert "@sha abc123" in rendered
    assert "@pr 1467 Fix template lookup" in rendered
    assert "@discussion" in rendered
    assert rendered.index("@discussion") < rendered.index("@details")
    assert "Reviewed-by: Ada" in rendered
    assert "Signed-off-by" not in rendered


def test_commit_pr_lookup_supports_root_store_readonly(tmp_path: Path) -> None:
    from scripts.pr_ingest.pr_store import PRStore
    from tools.clang_indexer.process_commits import PRDiscussionLookup

    store_path = tmp_path / "pull_requests.sqlite"
    with PRStore(str(store_path)) as store:
        store.upsert_pr(
            "owner/project",
            1467,
            title="Fix template lookup",
            body="Preserve the dependent-name lookup rules.",
            state="merged",
            author="ada",
            created_at="2026-07-15T00:00:00Z",
            merged_at="2026-07-15T01:00:00Z",
            merge_commit_sha="abc123",
            comments=[{"author": "reviewer", "body": "Approved."}],
            reviews=[],
            raw=None,
            fetched_at="2026-07-15T01:05:00Z",
        )
        store.commit()

    lookup = PRDiscussionLookup(str(store_path), None)
    record = {"repo": "owner/project", "commit_hash": "abc123"}
    try:
        assert lookup.attach(record) is True
        assert record["pr_number"] == 1467
        assert record["pr_title"] == "Fix template lookup"
        assert "Preserve the dependent-name lookup rules." in record["pr_discussion"]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            lookup._conn.execute("CREATE TABLE forbidden(value INTEGER)")
    finally:
        lookup.close()


def test_commit_pr_lookup_is_bound_to_exact_scan_membership(
    tmp_path: Path,
) -> None:
    from scripts.pr_ingest import pr_store
    from tools.clang_indexer.process_commits import PRDiscussionLookup

    current_scan = "a" * 64
    stale_scan = "b" * 64
    store_path = tmp_path / "pull_requests.sqlite"
    conn = pr_store.connect(str(store_path), create=True)
    try:
        pr_store.upsert_record(
            conn,
            {
                "repo": "owner/project",
                "pr_number": 1,
                "merge_commit_sha": "old-sha",
                "pr_title": "Old membership",
                "pr_body": "Must not be attached.",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
            scan_id=stale_scan,
        )
        pr_store.upsert_record(
            conn,
            {
                "repo": "owner/project",
                "pr_number": 1,
                "merge_commit_sha": "current-sha",
                "pr_title": "Current membership",
                "pr_body": "Verified discussion.",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
            scan_id=current_scan,
        )
        pr_store.upsert_record(
            conn,
            {
                "repo": "owner/project",
                "pr_number": 2,
                "merge_commit_sha": "stale-sha",
                "pr_title": "Stale row",
                "pr_body": "Must not be attached.",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
            scan_id=stale_scan,
        )
    finally:
        conn.close()

    lookup = PRDiscussionLookup(
        str(store_path),
        None,
        scan_id=current_scan,
    )
    try:
        current = {
            "repo": "owner/project",
            "commit_hash": "current-sha",
        }
        assert lookup.attach(current) is True
        assert current["pr_number"] == 1
        assert "Verified discussion." in current["pr_discussion"]

        assert lookup.attach(
            {"repo": "owner/project", "commit_hash": "old-sha"}
        ) is False
        assert lookup.attach(
            {"repo": "owner/project", "commit_hash": "stale-sha"}
        ) is False
        assert lookup.attach(
            {"repo": "owner/project", "pr_number": 2}
        ) is False
    finally:
        lookup.close()

    with pytest.raises(ValueError, match="invalid PR scan_id"):
        PRDiscussionLookup(str(store_path), None, scan_id="not-a-scan")


def test_atomic_publish_replaces_only_after_success(tmp_path: Path) -> None:
    from scripts.data.atomic_publish import atomic_output_file

    output = tmp_path / "result.jsonl"
    output.write_text("old\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stop"):
        with atomic_output_file(output) as stage:
            stage.write_text("partial\n", encoding="utf-8")
            raise RuntimeError("stop")
    assert output.read_text(encoding="utf-8") == "old\n"

    with atomic_output_file(output) as stage:
        stage.write_text("complete\n", encoding="utf-8")
    assert output.read_text(encoding="utf-8") == "complete\n"


def test_memory_guard_disabled_limit_is_a_noop() -> None:
    from scripts.data.memory_guard import check_memory_limit, max_rss_bytes

    assert max_rss_bytes() > 0
    check_memory_limit(0, label="source-processing-test")


def test_line_comment_newline_survives_tokenizer_roundtrip() -> None:
    pytest.importorskip("tokenizers")
    from cppmega.tokenizer import load_cppmega_tokenizer

    tokenizer = load_cppmega_tokenizer(REPO_ROOT / "cppmega" / "tokenizer")
    source = "// explanation\nint value = 7;\n"

    token_ids = tokenizer.encode(source)

    assert tokenizer.decode(token_ids) == source
    tokens = [tokenizer.token_for_id(token_id) for token_id in token_ids]
    assert "//" in tokens
    assert tokens.count("<NL>") == 2
    assert tokens[tokens.index("//") + 1 :].index("<NL>") >= 1
