#!/usr/bin/env python3
# ruff: noqa: E402
"""Process git commit diffs into enriched training documents using libclang.

Uses libclang for syntax tree metadata, qualified names, resolved call
references, and file-local dependency ordering.

Pipeline:
  1. Read JSONL records: {old_content, new_content, diff, subject, body, filepath, repo}
  2. Parse old/new content with libclang (via temp files)
  3. Extract functions/classes, build call graph, compute dep levels
  4. Parse diff hunks → find changed functions → BFS transitive deps → build chains
  5. Format PRE-COMMIT/POST-COMMIT documents with enriched metadata
  6. Tokenize, enforce max-tokens budget, output enriched JSONL

Usage:
    python3 tools/clang_indexer/process_commits.py \
        --inputs raw_commits.jsonl \
        --output enriched_commits.jsonl \
        --max-tokens 4096 --format both

    # Then convert to parquet:
    python3 scripts/data/clang_enriched_to_4k_parquet.py \
        --input-file enriched_commits.jsonl \
        --output-file enriched_commits.parquet \
        --overflow-policy drop
"""

import argparse
from array import array
import gc
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence, cast

# Increase recursion limit for deeply nested ASTs
sys.setrecursionlimit(50000)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse infrastructure from index_project.py
from tools.clang_indexer.index_project import (
    _configure_libclang,
    _adapt_args_for_file,
    _collect_macro_include_dirs,
    _compute_symbol_id,
    _document_symbol_identities,
    _function_part,
    _iter_code_identifiers,
    _macro_invocation_route_parts,
    _macro_part,
    _macro_route_part,
    _normalize_signature_text,
    _part_macro_provenance,
    _part_symbol_key,
    _part_symbol_metadata,
    _sanitize_compile_args_for_clang,
    _semantic_identity_records_for_arrays,
    _symbol_part_metadata,
    _used_macro_defs,
    canonical_symbol_identity,
    build_build_doc,
    FunctionDef,
    MacroDef,
    PartInfo,
    ProjectIndex,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    TypeDef,
    _cpp_domain_sidecars,
    extract_callee_references,
    extract_referenced_type_references,
    get_qualified_name,
    FUNCTION_KINDS,
    CONTAINER_KINDS,
    _byte_to_char_mapper,
    _cursor_text_and_metadata,
    extract_clang_ast_metadata,
    extract_semantic_metadata,
    extract_semantic_metadata_from_parts,
    normalize_qualified_name,
    register_header_macros,
    symbol_identity_for_cursor,
)
from cppmega.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SymbolIdentityError,
    require_project_identity,
)
from cppmega.data.build_context import detect_build_context
from cppmega.data.commit_scope import classify_primary_commit_path
from scripts.data.memory_guard import check_memory_limit, start_memory_guard
from scripts.data.atomic_publish import atomic_output_file
from scripts.pr_ingest import pr_store as _pr_store_mod


def _clip_pr_text(text: object, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) > max_chars:
        return value[:max_chars].rstrip() + " ...[truncated]"
    return value


def _json_list(value: object, *, field: str) -> list[dict[str, object]]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid PR store {field}: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"PR store {field} must be a list of objects")
    return [dict(item) for item in value]


def _normalize_pr_record(record: object) -> dict[str, object]:
    try:
        result = dict(record)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PR store returned an unsupported record: {record!r}") from exc
    if "pr_title" not in result:
        result["pr_title"] = result.get("title", "")
    if "pr_body" not in result:
        result["pr_body"] = result.get("body", "")
    if "comments" not in result:
        result["comments"] = _json_list(
            result.get("comments_json"), field="comments_json"
        )
    if "reviews" not in result:
        result["reviews"] = _json_list(
            result.get("reviews_json"), field="reviews_json"
        )
    result.setdefault("linked_issues", [])
    return result


def _render_discussion(
    record: object,
    *,
    max_comments: int = 30,
    max_reviews: int = 20,
    max_body_chars: int = 4000,
    max_item_chars: int = 1500,
    max_total_chars: int = 24000,
) -> str:
    rec = _normalize_pr_record(record)
    if "pr_number" not in rec:
        raise ValueError(f"PR store record missing pr_number: {rec!r}")

    title = str(rec.get("pr_title") or "").strip()
    number = rec["pr_number"]
    header = f"PR #{number}" + (f": {title}" if title else "")
    lines = [header]

    body = _clip_pr_text(rec.get("pr_body"), max_body_chars)
    if body:
        lines.extend(("", body))

    comments = _json_list(rec.get("comments"), field="comments")
    if comments:
        lines.extend(("", f"--- Discussion ({len(comments)} comments) ---"))
        for comment in comments[:max_comments]:
            user = str(
                comment.get("user")
                or comment.get("author")
                or comment.get("login")
                or "?"
            ).strip()
            path = comment.get("path")
            tag = f"@{user}" + (f" ({path})" if path else "")
            lines.append(
                f"{tag}: {_clip_pr_text(comment.get('body'), max_item_chars)}"
            )
        if len(comments) > max_comments:
            lines.append(f"...[+{len(comments) - max_comments} more comments]")

    reviews = _json_list(rec.get("reviews"), field="reviews")
    review_items = [
        review
        for review in reviews
        if str(review.get("body") or "").strip() or review.get("state")
    ]
    if review_items:
        lines.extend(("", f"--- Reviews ({len(review_items)}) ---"))
        for review in review_items[:max_reviews]:
            user = str(
                review.get("user")
                or review.get("author")
                or review.get("login")
                or "?"
            ).strip()
            state = str(review.get("state") or "").strip()
            head = f"@{user}" + (f" [{state}]" if state else "")
            review_body = _clip_pr_text(review.get("body"), max_item_chars)
            lines.append(f"{head}: {review_body}" if review_body else head)
        if len(review_items) > max_reviews:
            lines.append(f"...[+{len(review_items) - max_reviews} more reviews]")

    linked_issues = _json_list(rec.get("linked_issues"), field="linked_issues")
    if linked_issues:
        lines.extend(("", f"--- Linked issues ({len(linked_issues)}) ---"))
        for issue in linked_issues:
            title = str(issue.get("title") or "").strip()
            lines.append(f"#{issue.get('number')} {title}".rstrip())
            issue_body = _clip_pr_text(issue.get("body"), max_item_chars)
            if issue_body:
                lines.append(issue_body)

    text = "\n".join(lines).strip()
    if not text or text == header:
        return ""
    return _clip_pr_text(text, max_total_chars)


class PartialParseError(RuntimeError):
    """Raised when a commit range would publish after ordinary parse failures."""


class PRDiscussionLookup:
    """Live (repo, pr_number|merge_commit_sha) -> pr_discussion lookup glue.

    Opens the Tier-2 PR store (read-only) once. Verified scans use one explicit
    fixed owner/repo key; unverified direct/dev callers may still provide the
    legacy bare-name map. For each commit record it queries the store FIRST by
    (owner_repo, pr_number) then by (owner_repo, commit_hash), renders the
    assembled PR record via render_discussion, and writes the result into
    ``record['pr_discussion']`` IN PLACE.

    RULE #1 (fail-loud): a missing store / repo_list / malformed JSON RAISES at
    construction time. A per-record MISS is the NORMAL Tier-1 (git-only) path and
    does NOT fail — it simply leaves record['pr_discussion'] absent.
    """

    def __init__(
        self,
        store_path: str,
        repo_list_path: str | None,
        *,
        scan_id: str | None = None,
        owner_repo: str | None = None,
    ) -> None:
        if not os.path.exists(store_path):
            raise FileNotFoundError(f"--pr-store does not exist: {store_path}")
        if scan_id is not None and re.fullmatch(r"[0-9a-f]{64}", scan_id) is None:
            raise ValueError(f"invalid PR scan_id: {scan_id!r}")
        self._fixed_owner_repo = (
            require_project_identity(
                owner_repo,
                source="process_commits --pr-repo",
            )
            if owner_repo is not None
            else None
        )
        if scan_id is not None and self._fixed_owner_repo is None:
            raise ValueError("PR scan_id requires an explicit PR owner/repo key")
        self.scan_id = scan_id
        self._name_to_owner_repo: dict[str, str] = {}
        if repo_list_path:
            if not os.path.exists(repo_list_path):
                raise FileNotFoundError(
                    f"--repo-list does not exist: {repo_list_path}")
            with open(repo_list_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(
                    f"--repo-list {repo_list_path} must contain a JSON object"
                )
            entries = data.get("repos", [])
            if not isinstance(entries, list):
                raise ValueError(
                    f"--repo-list {repo_list_path} field 'repos' must be a list"
                )
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"--repo-list {repo_list_path} field 'repos[{index}]' "
                        "must be an object"
                    )
                name = entry.get("bare_name") or entry.get("name")
                owner_repo = entry.get("owner_repo")
                if name and owner_repo:
                    self._name_to_owner_repo[name] = owner_repo
        # Validate every control input before acquiring the read-only handle.
        if all(
            callable(getattr(_pr_store_mod, name, None))
            for name in ("connect", "get_by_pr", "get_by_sha")
        ):
            self._legacy_store = False
            self._conn = _pr_store_mod.connect(
                store_path,
                create=False,
                readonly=True,
            )
        else:
            self._legacy_store = True
            uri = Path(store_path).resolve().as_uri() + "?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, timeout=60.0)
            self._conn.row_factory = sqlite3.Row
        self.hits = 0
        self.misses = 0

    def _store_key(self, record: dict) -> str | None:
        """Resolve the (store-keyed) owner/repo for a record.

        Verified production scans always use the fixed ``--pr-repo`` key. The
        record-derived/list-derived path remains only for unverified direct/dev
        invocations.
        """
        if self._fixed_owner_repo is not None:
            return self._fixed_owner_repo
        repo = (record.get("repo") or "").strip()
        if not repo:
            return None
        if repo.startswith("corpus.local/"):
            return None
        if "/" in repo:
            return repo
        return self._name_to_owner_repo.get(repo)

    def _get_by_pr(self, owner_repo: str, pr_number: int) -> object | None:
        if not self._legacy_store:
            return _pr_store_mod.get_by_pr(
                self._conn,
                owner_repo,
                pr_number,
                scan_id=self.scan_id,
            )
        if self.scan_id is None:
            return self._conn.execute(
                "SELECT * FROM prs WHERE repo=? AND pr_number=?",
                (owner_repo, pr_number),
            ).fetchone()
        return self._conn.execute(
            "SELECT * FROM prs WHERE repo=? AND pr_number=? AND scan_id=?",
            (owner_repo, pr_number, self.scan_id),
        ).fetchone()

    def _get_by_sha(self, owner_repo: str, sha: str) -> object | None:
        if not self._legacy_store:
            return _pr_store_mod.get_by_sha(
                self._conn,
                owner_repo,
                sha,
                scan_id=self.scan_id,
            )
        query = """
            SELECT p.* FROM prs AS p
            JOIN pr_by_sha AS s
              ON s.repo=p.repo
             AND s.pr_number=p.pr_number
             AND s.merge_commit_sha=p.merge_commit_sha
            WHERE s.repo=? AND s.merge_commit_sha=?
        """
        values: tuple[object, ...] = (owner_repo, sha)
        if self.scan_id is not None:
            query += " AND p.scan_id=?"
            values = (*values, self.scan_id)
        return self._conn.execute(query, values).fetchone()

    def attach(self, record: dict) -> bool:
        """Look up the PR for this commit and set record['pr_discussion'] on hit.

        Lookup order: (owner_repo, pr_number) THEN (owner_repo, commit_hash).
        Returns True when a non-empty discussion was attached, else False (MISS,
        the normal Tier-1 git-only path — never fails).
        """
        owner_repo = self._store_key(record)
        if not owner_repo:
            self.misses += 1
            return False
        rec = None
        pr_number = record.get("pr_number")
        if pr_number is not None:
            rec = self._get_by_pr(owner_repo, int(pr_number))
        if rec is None:
            sha = (record.get("commit_hash") or "").strip()
            if sha:
                rec = self._get_by_sha(owner_repo, sha)
        if rec is None:
            self.misses += 1
            return False
        normalized = _normalize_pr_record(rec)
        if (
            self.scan_id is not None
            and normalized.get("scan_id") != self.scan_id
        ):
            raise ValueError(
                f"PR store returned a row outside scan_id={self.scan_id}"
            )
        discussion = _render_discussion(normalized)
        if not discussion:
            self.misses += 1
            return False
        canonical_pr_number = normalized.get("pr_number")
        if canonical_pr_number is None:
            raise ValueError(
                f"PR store hit for {owner_repo} has no canonical pr_number"
            )
        record["pr_number"] = int(canonical_pr_number)
        record["pr_title"] = str(normalized.get("pr_title") or "")
        record["pr_discussion"] = discussion
        self.hits += 1
        return True

    def close(self) -> None:
        self._conn.close()


if TYPE_CHECKING:
    from clang.cindex import CursorKind, Index, TranslationUnit  # pyright: ignore[reportMissingImports]
else:
    CursorKind = object
    Index = object
    TranslationUnit = object


class _ClangRuntimeSurface(Protocol):
    Index: type[object]
    TranslationUnit: type[object]
    CursorKind: object


clang_runtime: _ClangRuntimeSurface | None
try:
    clang_runtime = cast(_ClangRuntimeSurface, importlib.import_module("clang.cindex"))
except ImportError:
    clang_runtime = None
else:
    if not TYPE_CHECKING:
        Index = cast(type[object], clang_runtime.Index)  # type: ignore[assignment]
        TranslationUnit = cast(type[object], clang_runtime.TranslationUnit)  # type: ignore[assignment]
        CursorKind = clang_runtime.CursorKind  # type: ignore[assignment]

detect_language_info: Callable[..., dict[str, object] | None] | None
try:
    from cppmega.data.language_info import detect_language_info
except ImportError:
    detect_language_info = None

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClassDef:
    """A class/struct definition with source span."""
    name: str
    qualified_name: str
    text: str
    start_line: int
    end_line: int
    file: str = ""
    symbol_key: str = ""
    symbol_id: int = field(init=False)
    usr: str = ""
    canonical_signature: str = ""
    symbol_kind: str = "CLASS_DECL"
    member_symbol_keys: list[str] = field(default_factory=list)
    ast_depth: list[int] = field(default_factory=list)
    sibling_index: list[int] = field(default_factory=list)
    ast_node_type: list[int] = field(default_factory=list)
    semantic_symbol_ids: list[int] = field(default_factory=list)
    semantic_call_targets: list[int] = field(default_factory=list)
    semantic_type_refs: list[int] = field(default_factory=list)
    semantic_def_use: list[int] = field(default_factory=list)
    semantic_symbol_identities: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.qualified_name = normalize_qualified_name(self.qualified_name)
        self.canonical_signature = _normalize_signature_text(self.canonical_signature)
        if not self.symbol_key:
            self.symbol_key = canonical_symbol_identity(
                qname=self.qualified_name,
                kind=self.symbol_kind,
                canonical_signature=self.canonical_signature,
                file=self.file,
                line=self.start_line,
                force_file_scope=True,
            )
        self.symbol_id = _compute_symbol_id(self.symbol_key)
        self.ast_depth = array('H', self.ast_depth)
        self.sibling_index = array('H', self.sibling_index)
        self.ast_node_type = array('H', self.ast_node_type)
        self.semantic_symbol_ids = array('Q', self.semantic_symbol_ids)
        self.semantic_call_targets = array('Q', self.semantic_call_targets)
        self.semantic_type_refs = array('Q', self.semantic_type_refs)
        self.semantic_def_use = array('B', self.semantic_def_use)


def _class_part(cls: ClassDef) -> PartInfo:
    return (
        cls.text,
        4,
        0,
        cls.name,
        cls.qualified_name,
        None,
        None,
        _symbol_part_metadata(
            cls.symbol_key,
            qname=cls.qualified_name,
            symbol_id=cls.symbol_id,
            canonical_signature=cls.canonical_signature,
            usr=cls.usr,
            kind=cls.symbol_kind,
        ),
    )


@dataclass
class FileAnalysis:
    """Single-file analysis result from libclang."""
    preamble: str
    functions: list[FunctionDef] = field(default_factory=list)
    classes: list[ClassDef] = field(default_factory=list)
    preamble_ast_depth: list[int] = field(default_factory=list)
    preamble_sibling_index: list[int] = field(default_factory=list)
    preamble_ast_node_type: list[int] = field(default_factory=list)
    compile_args: list[str] = field(default_factory=list)
    build_info: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.preamble_ast_depth = array('H', self.preamble_ast_depth)
        self.preamble_sibling_index = array('H', self.preamble_sibling_index)
        self.preamble_ast_node_type = array('H', self.preamble_ast_node_type)

    def build_local_index(self) -> ProjectIndex:
        """Build a ProjectIndex from this file's functions for dep computation."""
        idx = ProjectIndex()
        for func in self.functions:
            idx.add_function(func)
        idx.compute_dep_levels()
        return idx


def _clone_function_def(func: FunctionDef) -> FunctionDef:
    return FunctionDef(
        func.name,
        func.qualified_name,
        func.file,
        func.line,
        func.text,
        list(func.callees),
        is_definition=func.is_definition,
        end_line=func.end_line,
        ast_depth=list(func.ast_depth),
        sibling_index=list(func.sibling_index),
        ast_node_type=list(func.ast_node_type),
        referenced_types=list(func.referenced_types),
        baselib_callees=list(func.baselib_callees),
        symbol_key=func.symbol_key,
        usr=func.usr,
        canonical_signature=func.canonical_signature,
        symbol_kind=func.symbol_kind,
        callee_keys=list(func.callee_keys),
        referenced_type_keys=list(func.referenced_type_keys),
        callee_refs=list(func.callee_refs),
        baselib_callee_refs=list(func.baselib_callee_refs),
        referenced_type_refs=list(func.referenced_type_refs),
        semantic_symbol_ids=list(func.semantic_symbol_ids),
        semantic_call_targets=list(func.semantic_call_targets),
        semantic_type_refs=list(func.semantic_type_refs),
        semantic_def_use=list(func.semantic_def_use),
        semantic_symbol_identities=list(func.semantic_symbol_identities),
    )


def _clone_class_def(cls: ClassDef) -> ClassDef:
    return ClassDef(
        name=cls.name,
        qualified_name=cls.qualified_name,
        text=cls.text,
        start_line=cls.start_line,
        end_line=cls.end_line,
        file=cls.file,
        symbol_key=cls.symbol_key,
        usr=cls.usr,
        canonical_signature=cls.canonical_signature,
        symbol_kind=cls.symbol_kind,
        member_symbol_keys=list(cls.member_symbol_keys),
        ast_depth=list(cls.ast_depth),
        sibling_index=list(cls.sibling_index),
        ast_node_type=list(cls.ast_node_type),
        semantic_symbol_ids=list(cls.semantic_symbol_ids),
        semantic_call_targets=list(cls.semantic_call_targets),
        semantic_type_refs=list(cls.semantic_type_refs),
        semantic_def_use=list(cls.semantic_def_use),
        semantic_symbol_identities=list(cls.semantic_symbol_identities),
    )


def _clone_file_analysis(analysis: FileAnalysis) -> FileAnalysis:
    return FileAnalysis(
        preamble=analysis.preamble,
        functions=[_clone_function_def(func) for func in analysis.functions],
        classes=[_clone_class_def(cls) for cls in analysis.classes],
        preamble_ast_depth=list(analysis.preamble_ast_depth),
        preamble_sibling_index=list(analysis.preamble_sibling_index),
        preamble_ast_node_type=list(analysis.preamble_ast_node_type),
        compile_args=list(analysis.compile_args),
        build_info=dict(analysis.build_info),
    )


class AnalysisCache:
    """Small LRU for expensive libclang file analyses inside one range worker."""

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max(0, int(max_entries))
        self._items: OrderedDict[tuple[str, str, str, tuple[str, ...], str, str], FileAnalysis] = (
            OrderedDict()
        )
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def _key(
        content: str,
        filepath: str,
        compile_args: list[str] | None,
        repo_root: str | None,
        build_info: dict[str, object] | None,
        project_id: str | None,
    ) -> tuple[str, str, str, tuple[str, ...], str, str]:
        digest = hashlib.sha1(
            content.encode("utf-8", errors="surrogatepass")
        ).hexdigest()
        build_key = json.dumps(build_info or {}, sort_keys=True, default=str)
        root_key = os.path.abspath(repo_root) if repo_root else ""
        return (project_id or "", filepath, root_key, tuple(compile_args or ()), build_key, digest)

    def get_or_analyze(
        self,
        content: str,
        filepath: str,
        clang_index: Index,
        tmpdir: str,
        *,
        compile_args: list[str] | None,
        repo_root: str | None,
        build_info: dict[str, object] | None,
        project_id: str | None,
        analyzer: Callable[..., FileAnalysis],
    ) -> FileAnalysis:
        if self.max_entries <= 0:
            self.misses += 1
            return analyzer(
                content,
                filepath,
                clang_index,
                tmpdir,
                compile_args=compile_args,
                repo_root=repo_root,
                build_info=build_info,
                project_id=project_id,
            )

        key = self._key(
            content, filepath, compile_args, repo_root, build_info, project_id
        )
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
            self.hits += 1
            return _clone_file_analysis(cached)

        self.misses += 1
        analysis = analyzer(
            content,
            filepath,
            clang_index,
            tmpdir,
            compile_args=compile_args,
            repo_root=repo_root,
            build_info=build_info,
            project_id=project_id,
        )
        self._items[key] = _clone_file_analysis(analysis)
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
            self.evictions += 1
        return analysis


@dataclass
class HunkRange:
    """Parsed unified diff hunk range."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

EDIT_OP_UNCHANGED = 0
EDIT_OP_INSERTED = 1
EDIT_OP_MODIFIED = 2
EDIT_OP_CONTEXT = 3


def parse_hunk_ranges(diff: str) -> list[HunkRange]:
    """Parse unified diff @@ lines into hunk ranges."""
    return [
        HunkRange(
            old_start=int(m.group(1)),
            old_count=int(m.group(2) or '1'),
            new_start=int(m.group(3)),
            new_count=int(m.group(4) or '1'),
        )
        for m in _HUNK_RE.finditer(diff)
    ]


def changed_lines(hunks: list[HunkRange], use_old: bool) -> set[int]:
    """Get set of line numbers touched by hunks (1-based)."""
    lines: set[int] = set()
    for h in hunks:
        start = h.old_start if use_old else h.new_start
        count = max(h.old_count if use_old else h.new_count, 1)
        lines.update(range(start, start + count))
    return lines


def compute_new_line_edit_ops(diff: str) -> tuple[dict[int, int], dict[int, int]]:
    """Map 1-based new-file lines to (edit-op id, 0-based hunk index).

    Returns ``(ops, line_hunk)`` where ``ops[new_line] -> EDIT_OP_*`` and
    ``line_hunk[new_line] -> hunk_index`` (0 for the first ``@@`` hunk, 1 for the
    second, ...). The hunk index lets the per-token ``hunk_id`` carry which edit
    region a token belongs to, not just a binary changed flag.
    """
    ops: dict[int, int] = {}
    line_hunk: dict[int, int] = {}
    current_new_line: int | None = None
    pending_removes = 0
    hunk_idx = -1
    for line in diff.splitlines():
        match = _HUNK_RE.match(line)
        if match:
            current_new_line = int(match.group(3))
            pending_removes = 0
            hunk_idx += 1
            continue
        if current_new_line is None:
            continue
        if line.startswith('-') and not line.startswith('---'):
            pending_removes += 1
            continue
        if line.startswith('+') and not line.startswith('+++'):
            if pending_removes > 0:
                ops[current_new_line] = EDIT_OP_MODIFIED
                pending_removes -= 1
            else:
                ops[current_new_line] = EDIT_OP_INSERTED
            line_hunk[current_new_line] = hunk_idx
            current_new_line += 1
            continue
        pending_removes = 0
        current_new_line += 1
    return ops, line_hunk


def compute_old_line_hunk(hunks: list[HunkRange]) -> dict[int, int]:
    """Map 1-based OLD-file lines to the 0-based hunk index that touches them."""
    line_hunk: dict[int, int] = {}
    for hunk_idx, h in enumerate(hunks):
        for line_no in range(h.old_start, h.old_start + max(h.old_count, 1)):
            line_hunk.setdefault(line_no, hunk_idx)
    return line_hunk


def _line_ranges_by_number(text: str) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    offset = 0
    for line_no, line in enumerate(text.splitlines(keepends=True), start=1):
        end = offset + len(line)
        ranges[line_no] = (offset, end)
        offset = end
    if text and (not text.endswith(('\n', '\r'))):
        ranges.setdefault(len(ranges) + 1, (offset, len(text)))
    return ranges


def _line_ranges_for_changed_functions(
    analysis: FileAnalysis,
    changed_line_numbers: set[int],
) -> list[tuple[int, int]]:
    if not analysis.functions and not analysis.classes:
        return []
    ranges: list[tuple[int, int]] = []
    for func in analysis.functions:
        if any(line in changed_line_numbers for line in range(func.line, func.end_line + 1)):
            ranges.append((func.line, func.end_line))
    for cls in analysis.classes:
        if any(line in changed_line_numbers for line in range(cls.start_line, cls.end_line + 1)):
            ranges.append((cls.start_line, cls.end_line))
    return ranges


def _build_commit_temporal_metadata(
    full_text: str,
    part_texts: list[str],
    section_kinds: list[str],
    *,
    record: dict,
    old_analysis: FileAnalysis,
    new_analysis: Optional[FileAnalysis],
) -> dict[str, list[int]]:
    text_len = len(full_text)
    change_mask_pre = [0] * text_len
    change_mask_post = [0] * text_len
    hunk_id_per_char = [-1] * text_len  # -1 = unchanged/context (no hunk)
    edit_op_per_char = [EDIT_OP_CONTEXT] * text_len

    diff = str(record.get('diff', '') or '')
    hunks = parse_hunk_ranges(diff)
    if not hunks:
        return {
            'change_mask_pre': [],
            'change_mask_post': [],
            'hunk_id_per_char': [],
            'edit_op_per_char': [],
        }

    old_changed_lines = changed_lines(hunks, use_old=True)
    new_line_ops, new_line_hunk = compute_new_line_edit_ops(diff)
    old_line_hunk = compute_old_line_hunk(hunks)
    old_changed_ranges = _line_ranges_for_changed_functions(old_analysis, old_changed_lines)
    if not old_changed_ranges and old_changed_lines:
        old_changed_ranges = [(min(old_changed_lines), max(old_changed_lines))]
    old_content = str(record.get('old_content', '') or '')
    new_content = str(record.get('new_content', '') or '')
    old_line_ranges = _line_ranges_by_number(str(record.get('old_content', '') or ''))
    new_line_ranges = _line_ranges_by_number(str(record.get('new_content', '') or ''))
    # Build text->op and the CONSISTENT text->hunk map in one pass so any text
    # marked changed also carries its hunk index (a changed line always has a
    # hunk via compute_new_line_edit_ops).
    new_line_ops_by_text: dict[str, int] = {}
    new_line_hunk_by_text: dict[str, int] = {}
    for line_no, (start, end) in new_line_ranges.items():
        stripped = new_content[start:end].strip()
        if not stripped:
            continue
        op = new_line_ops.get(line_no, EDIT_OP_UNCHANGED)
        previous = new_line_ops_by_text.get(stripped)
        if previous is None or (previous == EDIT_OP_UNCHANGED and op != EDIT_OP_UNCHANGED):
            new_line_ops_by_text[stripped] = op
            new_line_hunk_by_text[stripped] = new_line_hunk.get(line_no, -1)

    def old_part_hunk(part: str) -> int:
        """Hunk index of the first hunk overlapping a changed OLD part, else -1."""
        if not part.strip() or not old_changed_ranges:
            return -1
        for start_line, end_line in old_changed_ranges:
            start_end = old_line_ranges.get(start_line)
            end_end = old_line_ranges.get(end_line)
            if not start_end or not end_end:
                continue
            changed_text = old_content[start_end[0]:end_end[1]].strip()
            if changed_text and (part.strip() in changed_text or changed_text in part.strip()):
                for ln in range(start_line, end_line + 1):
                    if ln in old_line_hunk:
                        return old_line_hunk[ln]
                return -1
        return -1

    def new_line_op(part_line: str) -> int:
        stripped = part_line.strip()
        if not stripped:
            return EDIT_OP_CONTEXT
        return new_line_ops_by_text.get(stripped, EDIT_OP_UNCHANGED)

    def new_line_hunk_for(part_line: str) -> int:
        stripped = part_line.strip()
        if not stripped:
            return -1
        return new_line_hunk_by_text.get(stripped, -1)

    offset = 0
    for idx, part in enumerate(part_texts):
        source_kind = section_kinds[idx] if idx < len(section_kinds) else 'c'
        part_len = len(part)
        if part_len <= 0:
            continue
        part_end = min(offset + part_len, text_len)
        if source_kind == 'n':
            pos = offset
            for line in part.splitlines(keepends=True):
                line_len = len(line)
                op = new_line_op(line)
                changed = int(op in (EDIT_OP_INSERTED, EDIT_OP_MODIFIED))
                h_idx = new_line_hunk_for(line) if changed else -1
                # RULE #1: a changed new line that maps to no hunk is a real
                # inconsistency (op and hunk maps are built together), not a
                # degraded value -> fail loud rather than emit a bogus id.
                if changed and h_idx < 0:
                    raise ValueError(
                        f"process_commits: changed new line maps to no hunk in "
                        f"{record.get('commit_hash')}:{record.get('filepath')}")
                end = min(pos + line_len, text_len)
                for char_idx in range(pos, end):
                    change_mask_post[char_idx] = changed
                    hunk_id_per_char[char_idx] = h_idx
                    edit_op_per_char[char_idx] = op
                pos = end
        elif source_kind == 'o':
            h_idx = old_part_hunk(part)
            changed = int(h_idx >= 0)
            for char_idx in range(offset, part_end):
                change_mask_pre[char_idx] = changed
                hunk_id_per_char[char_idx] = h_idx
                edit_op_per_char[char_idx] = EDIT_OP_CONTEXT
        else:
            for char_idx in range(offset, part_end):
                edit_op_per_char[char_idx] = EDIT_OP_CONTEXT
        offset += part_len
        if idx < len(part_texts) - 1:
            offset += 2

    if not any(change_mask_pre) and not any(change_mask_post):
        return {
            'change_mask_pre': [],
            'change_mask_post': [],
            'hunk_id_per_char': [],
            'edit_op_per_char': [],
        }
    return {
        'change_mask_pre': change_mask_pre,
        'change_mask_post': change_mask_post,
        'hunk_id_per_char': hunk_id_per_char,
        'edit_op_per_char': edit_op_per_char,
    }


# ---------------------------------------------------------------------------
# Clang single-file analysis
# ---------------------------------------------------------------------------

_C_EXTENSIONS = {'.c'}
_SIMPLE_FALLBACK_ARGS_CPP = ['-fsyntax-only', '-Wno-everything', '-std=c++17']
_SIMPLE_FALLBACK_ARGS_C = ['-x', 'c', '-fsyntax-only', '-Wno-everything', '-std=c11']


def _same_clang_path(left: str | None, right: str) -> bool:
    if not left:
        return False
    return os.path.normcase(os.path.normpath(left)) == os.path.normcase(os.path.normpath(right))


def _add_unique_include(args: list[str], include_dir: str) -> None:
    if not include_dir:
        return
    include_arg = f"-I{os.path.normpath(include_dir)}"
    if include_arg not in args:
        args.append(include_arg)


def _analysis_compile_args(
    filepath: str,
    compile_args: list[str] | None,
    repo_root: str | None,
) -> list[str]:
    ext = Path(filepath).suffix.lower() or '.cpp'
    if compile_args:
        args = _sanitize_compile_args_for_clang(list(compile_args))
    elif ext in _C_EXTENSIONS:
        args = list(_SIMPLE_FALLBACK_ARGS_C)
    else:
        args = list(_SIMPLE_FALLBACK_ARGS_CPP)

    if repo_root:
        _add_unique_include(args, repo_root)
        _add_unique_include(args, os.path.dirname(os.path.join(repo_root, filepath)))
    return _adapt_args_for_file(args, filepath)


def _extract_preamble_from_source(
    tu: TranslationUnit,
    filename: str,
    source: str,
    byte_to_char: Callable[[int], int],
    ast_depth: list[int],
    sibling_index: list[int],
    ast_node_type: list[int],
) -> tuple[str, list[int], list[int], list[int]]:
    texts: list[str] = []
    depth_parts: list[int] = []
    sibling_parts: list[int] = []
    node_type_parts: list[int] = []
    for cursor in tu.cursor.get_children():
        cursor_file = cursor.location.file.name if cursor.location.file else None
        if not _same_clang_path(cursor_file, filename):
            continue
        if cursor.kind in (
            CursorKind.INCLUSION_DIRECTIVE,
            CursorKind.USING_DIRECTIVE,
            CursorKind.USING_DECLARATION,
            CursorKind.TYPEDEF_DECL,
            CursorKind.TYPE_ALIAS_DECL,
            CursorKind.NAMESPACE_ALIAS,
        ):
            text, depth, sibling, node_type, _offsets = _cursor_text_and_metadata(
                cursor,
                source,
                filename,
                byte_to_char,
                ast_depth,
                sibling_index,
                ast_node_type,
            )
            if not text:
                continue
            if texts:
                depth_parts.append(0)
                sibling_parts.append(0)
                node_type_parts.append(0)
            texts.append(text)
            depth_parts.extend(depth)
            sibling_parts.extend(sibling)
            node_type_parts.extend(node_type)
    return "\n".join(texts), depth_parts, sibling_parts, node_type_parts


class BuildContextResolver:
    """Resolve per-record compile context without reloading build files per row."""

    def __init__(self, repo_root: str | None = None, repo_dir: str | None = None) -> None:
        self.repo_root = os.path.abspath(repo_root) if repo_root else None
        self.repo_dir = os.path.abspath(repo_dir) if repo_dir else None
        self._cache: dict[str, tuple[dict, list[str], object | None]] = {}
        self._macro_cache: OrderedDict[
            tuple[str, str, str, tuple[str, ...], str], ProjectIndex
        ] = OrderedDict()
        self._macro_cache_max_entries = int(os.environ.get("CPPMEGA_COMMIT_MACRO_CACHE_ENTRIES", "16"))

    def _record_repo_root(self, record: dict) -> str | None:
        explicit = record.get("repo_path")
        if isinstance(explicit, str) and explicit:
            return os.path.abspath(explicit)
        if self.repo_root:
            return self.repo_root
        repo_name = record.get("repo")
        if self.repo_dir and isinstance(repo_name, str) and repo_name:
            candidate = os.path.join(self.repo_dir, repo_name)
            if os.path.isdir(candidate):
                return os.path.abspath(candidate)
        return None

    def _load(self, repo_root: str) -> tuple[dict, list[str], object | None]:
        cached = self._cache.get(repo_root)
        if cached is not None:
            return cached
        context = detect_build_context(repo_root)
        self._cache[repo_root] = context
        return context

    def resolve(self, record: dict) -> tuple[str | None, list[str] | None, dict[str, object]]:
        compile_args = record.get("compile_args")
        build_info = record.get("build_info")
        args = [str(arg) for arg in compile_args] if isinstance(compile_args, list) else None
        info = dict(build_info) if isinstance(build_info, dict) else {}

        repo_root = self._record_repo_root(record)
        if repo_root is None or not os.path.isdir(repo_root):
            return None, args, info

        platform_info, default_args, compile_index = self._load(repo_root)
        if not info:
            info = {
                key: value
                for key, value in platform_info.items()
                if key in {"build_system", "source", "compiler", "standard"} and value is not None
            }
        filepath = str(record.get("filepath") or "")
        if compile_index is not None and filepath:
            lookup_path = filepath if os.path.isabs(filepath) else os.path.join(repo_root, filepath)
            lookup = getattr(compile_index, "lookup", None)
            if callable(lookup):
                file_args, file_info = lookup(lookup_path)
                if file_args:
                    args = list(file_args)
                if file_info:
                    info = {**info, **dict(file_info)}
        if args is None:
            args = list(default_args)
        return repo_root, args, info

    def macro_index_for(
        self,
        *,
        repo_root: str | None,
        filepath: str,
        compile_args: list[str] | None,
        project_id: str,
        usage_texts: Sequence[str] | None = None,
    ) -> ProjectIndex | None:
        """Return an include-aware macro index for this commit file.

        Commit records carry old/new buffers, but include resolution still comes
        from the checked-out repository and compile context.  Cache by
        repo_root/file/args so a range worker does not rescan the same include
        tree for every commit touching the same file.
        """

        if not repo_root or not filepath:
            return None
        root = os.path.abspath(repo_root)
        if not os.path.isdir(root):
            return None
        root_file = filepath if os.path.isabs(filepath) else os.path.join(root, filepath)
        root_file = os.path.normpath(root_file)
        if not os.path.exists(root_file):
            return None
        stable_project_id = require_project_identity(
            project_id,
            source=f"macro_index_for({filepath})",
        )
        usage_names = sorted({
            match.group(0)
            for text in (usage_texts or ())
            for match in _iter_code_identifiers(text)
        })
        usage_fingerprint = hashlib.sha1(
            "\0".join(usage_names).encode("utf-8")
        ).hexdigest()
        relative_root_file = os.path.relpath(root_file, root)
        key = (
            stable_project_id,
            root,
            relative_root_file,
            tuple(compile_args or ()),
            usage_fingerprint,
        )
        cached = self._macro_cache.get(key)
        if cached is not None:
            self._macro_cache.move_to_end(key)
            return cached

        include_dirs = _collect_macro_include_dirs(
            project_dir=root,
            compile_db=None,
            default_args=compile_args or [],
        )
        index = ProjectIndex()
        register_header_macros(
            index,
            [root_file],
            project_dir=root,
            project_id=stable_project_id,
            include_dirs=include_dirs,
            macro_usage_texts_by_file=(
                {relative_root_file: list(usage_texts)}
                if usage_texts is not None
                else None
            ),
        )
        self._macro_cache[key] = index
        self._macro_cache.move_to_end(key)
        while len(self._macro_cache) > max(0, self._macro_cache_max_entries):
            self._macro_cache.popitem(last=False)
        return index


def analyze_file_clang(
    content: str,
    filepath: str,
    clang_index: Index,
    tmpdir: str,
    *,
    compile_args: list[str] | None = None,
    repo_root: str | None = None,
    build_info: dict[str, object] | None = None,
    project_id: str,
) -> FileAnalysis:
    """Parse a single file's content with libclang and extract functions/classes.

    Prefer the original repository path plus libclang unsaved_files so quoted
    includes and compile_commands flags describe the real translation unit.
    Fall back to a temp source only when no repository root is available.
    """
    stable_project_id = require_project_identity(
        project_id,
        source=f"analyze_file_clang({filepath})",
    )
    if not content or len(content) < 20:
        return FileAnalysis(preamble='')

    ext = Path(filepath).suffix.lower() or '.cpp'
    os.makedirs(tmpdir, exist_ok=True)
    if repo_root:
        source_path = filepath if os.path.isabs(filepath) else os.path.join(repo_root, filepath)
        source_path = os.path.normpath(source_path)
        unsaved_files = [(source_path, content)]
    else:
        source_path = os.path.join(tmpdir, f"source{ext}")
        unsaved_files = None
        with open(source_path, 'w', errors='replace') as f:
            f.write(content)

    args = _analysis_compile_args(source_path, compile_args, repo_root)

    try:
        tu = clang_index.parse(
            source_path,
            args=args,
            unsaved_files=unsaved_files,
            # Each commit record parses throwaway old/new buffers once. A
            # precompiled preamble cache is useful for repeated reparses of the
            # same TU, but here it pins native clang memory across hundreds of
            # unrelated records and was the main per-process RSS amplifier.
            options=TranslationUnit.PARSE_INCOMPLETE,
        )
    except SymbolIdentityError:
        raise
    except Exception as exc:
        raise RuntimeError(f"libclang parse failed for {filepath}: {exc}") from exc

    ast_depth, sibling_index, ast_node_type = extract_clang_ast_metadata(
        content,
        tu,
        source_path,
    )
    identity_project_dir = repo_root or tmpdir
    semantic_metadata = extract_semantic_metadata(
        content,
        tu,
        source_path,
        project_dir=identity_project_dir,
        project_id=stable_project_id,
        fallback_file=filepath,
    )
    byte_to_char = _byte_to_char_mapper(content)

    # Extract preamble
    (
        preamble,
        preamble_ast_depth,
        preamble_sibling_index,
        preamble_ast_node_type,
    ) = _extract_preamble_from_source(
        tu,
        source_path,
        content,
        byte_to_char,
        ast_depth,
        sibling_index,
        ast_node_type,
    )

    # Extract functions and classes
    functions: list[FunctionDef] = []
    classes: list[ClassDef] = []

    def visit(cursor):
        if not cursor.location.file:
            return
        if not _same_clang_path(cursor.location.file.name, source_path):
            return

        if cursor.kind in FUNCTION_KINDS and cursor.is_definition():
            text, func_ast_depth, func_sibling_index, func_ast_node_type, offsets = (
                _cursor_text_and_metadata(
                    cursor,
                    content,
                    source_path,
                    byte_to_char,
                    ast_depth,
                    sibling_index,
                    ast_node_type,
                )
            )
            if text and len(text) >= 20 and offsets is not None:
                callee_refs, baselib_callee_refs = extract_callee_references(
                    cursor,
                    project_dir=identity_project_dir,
                    project_id=stable_project_id,
                    fallback_file=filepath,
                )
                referenced_type_refs = extract_referenced_type_references(
                    cursor,
                    project_dir=identity_project_dir,
                    project_id=stable_project_id,
                    fallback_file=filepath,
                )
                qname = get_qualified_name(cursor)
                symbol_key, usr, canonical_signature = symbol_identity_for_cursor(
                    cursor,
                    project_dir=identity_project_dir,
                    project=stable_project_id,
                    fallback_file=filepath,
                )
                start_line = cursor.extent.start.line
                end_line = cursor.extent.end.line
                start_offset, end_offset = offsets
                functions.append(FunctionDef(
                    name=cursor.spelling,
                    qualified_name=qname,
                    file=filepath,
                    line=start_line,
                    text=text,
                    callees=[str(ref["qname"]) for ref in callee_refs],
                    is_definition=True,
                    end_line=end_line,
                    ast_depth=func_ast_depth,
                    sibling_index=func_sibling_index,
                    ast_node_type=func_ast_node_type,
                    referenced_types=[str(ref["qname"]) for ref in referenced_type_refs],
                    baselib_callees=[str(ref["qname"]) for ref in baselib_callee_refs],
                    symbol_key=symbol_key,
                    usr=usr,
                    canonical_signature=canonical_signature,
                    symbol_kind=getattr(cursor.kind, "name", str(cursor.kind)),
                    callee_keys=[str(ref["symbol_key"]) for ref in callee_refs],
                    referenced_type_keys=[
                        str(ref["symbol_key"]) for ref in referenced_type_refs
                    ],
                    callee_refs=callee_refs,
                    baselib_callee_refs=baselib_callee_refs,
                    referenced_type_refs=referenced_type_refs,
                    semantic_symbol_ids=semantic_metadata["symbol_ids"][start_offset:end_offset],
                    semantic_call_targets=semantic_metadata["call_targets"][start_offset:end_offset],
                    semantic_type_refs=semantic_metadata["type_refs"][start_offset:end_offset],
                    semantic_def_use=semantic_metadata["def_use"][start_offset:end_offset],
                    semantic_symbol_identities=_semantic_identity_records_for_arrays(
                        semantic_metadata,
                        semantic_metadata["symbol_ids"][start_offset:end_offset],
                        semantic_metadata["call_targets"][start_offset:end_offset],
                        semantic_metadata["type_refs"][start_offset:end_offset],
                        source=f"{filepath}:{start_line}:{qname}",
                    ),
                ))

        elif cursor.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL,
                              CursorKind.CLASS_TEMPLATE,
                              CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
            if cursor.is_definition():
                text, cls_ast_depth, cls_sibling_index, cls_ast_node_type, offsets = (
                    _cursor_text_and_metadata(
                        cursor,
                        content,
                        source_path,
                        byte_to_char,
                        ast_depth,
                        sibling_index,
                        ast_node_type,
                    )
                )
                if text and len(text) >= 20 and offsets is not None:
                    qname = get_qualified_name(cursor)
                    symbol_key, usr, canonical_signature = symbol_identity_for_cursor(
                        cursor,
                        project_dir=identity_project_dir,
                        project=stable_project_id,
                        fallback_file=filepath,
                    )
                    member_symbol_keys = []
                    for child in cursor.get_children():
                        if child.kind not in FUNCTION_KINDS:
                            continue
                        member_key, _member_usr, _member_signature = symbol_identity_for_cursor(
                            child,
                            project_dir=identity_project_dir,
                            project=stable_project_id,
                            fallback_file=filepath,
                        )
                        member_symbol_keys.append(member_key)
                    start_offset, end_offset = offsets
                    classes.append(ClassDef(
                        name=cursor.spelling,
                        qualified_name=qname,
                        text=text,
                        start_line=cursor.extent.start.line,
                        end_line=cursor.extent.end.line,
                        file=filepath,
                        symbol_key=symbol_key,
                        usr=usr,
                        canonical_signature=canonical_signature,
                        symbol_kind=getattr(cursor.kind, "name", str(cursor.kind)),
                        member_symbol_keys=member_symbol_keys,
                        ast_depth=cls_ast_depth,
                        sibling_index=cls_sibling_index,
                        ast_node_type=cls_ast_node_type,
                        semantic_symbol_ids=semantic_metadata["symbol_ids"][start_offset:end_offset],
                        semantic_call_targets=semantic_metadata["call_targets"][start_offset:end_offset],
                        semantic_type_refs=semantic_metadata["type_refs"][start_offset:end_offset],
                        semantic_def_use=semantic_metadata["def_use"][start_offset:end_offset],
                        semantic_symbol_identities=_semantic_identity_records_for_arrays(
                            semantic_metadata,
                            semantic_metadata["symbol_ids"][start_offset:end_offset],
                            semantic_metadata["call_targets"][start_offset:end_offset],
                            semantic_metadata["type_refs"][start_offset:end_offset],
                            source=(
                                f"{filepath}:{cursor.extent.start.line}:{qname}"
                            ),
                        ),
                    ))
                # Recurse into class children to extract inline methods,
                # constructors, and destructors defined inside the class body.
                for child in cursor.get_children():
                    visit(child)

        elif cursor.kind in CONTAINER_KINDS:
            for child in cursor.get_children():
                visit(child)

    for cursor in tu.cursor.get_children():
        visit(cursor)

    return FileAnalysis(
        preamble=preamble,
        functions=functions,
        classes=classes,
        preamble_ast_depth=preamble_ast_depth,
        preamble_sibling_index=preamble_sibling_index,
        preamble_ast_node_type=preamble_ast_node_type,
        compile_args=args,
        build_info=dict(build_info or {}),
    )


# ---------------------------------------------------------------------------
# Changed function/class detection
# ---------------------------------------------------------------------------

def find_changed_functions(
    analysis: FileAnalysis,
    changed_line_set: set[int],
) -> list[int]:
    """Return indices of functions whose line spans overlap changed lines."""
    result = []
    for i, func in enumerate(analysis.functions):
        end_line = func.end_line
        func_lines = set(range(func.line, end_line + 1))
        if func_lines & changed_line_set:
            result.append(i)
    return result


def find_changed_classes(
    analysis: FileAnalysis,
    changed_line_set: set[int],
) -> list[int]:
    """Return indices of classes whose line spans overlap changed lines."""
    result = []
    for i, cls in enumerate(analysis.classes):
        cls_lines = set(range(cls.start_line, cls.end_line + 1))
        if cls_lines & changed_line_set:
            result.append(i)
    return result


# ---------------------------------------------------------------------------
# Symbol-level change tracking
# ---------------------------------------------------------------------------

@dataclass
class SymbolEntry:
    """A function or class keyed by its canonical clang identity."""
    id: int
    symbol_key: str
    name: str
    qualified_name: str
    kind: str  # 'function' or 'class'
    start_line: int
    end_line: int


def build_symbol_table(analysis: FileAnalysis) -> list[SymbolEntry]:
    """Build a flat symbol table using deterministic canonical symbol IDs."""
    symbols: list[SymbolEntry] = []
    for func in analysis.functions:
        symbols.append(SymbolEntry(
            id=func.symbol_id,
            symbol_key=func.symbol_key,
            name=func.name,
            qualified_name=func.qualified_name,
            kind='function',
            start_line=func.line,
            end_line=func.end_line,
        ))
    for cls in analysis.classes:
        symbols.append(SymbolEntry(
            id=cls.symbol_id,
            symbol_key=cls.symbol_key,
            name=cls.name,
            qualified_name=cls.qualified_name,
            kind='class',
            start_line=cls.start_line,
            end_line=cls.end_line,
        ))
    return symbols


def compute_changed_symbol_ids(
    analysis: FileAnalysis,
    changed_line_set: set[int],
) -> list[int]:
    """Return stable IDs of symbols whose line spans overlap changed lines.

    IDs are derived from canonical clang USR/signature identities.
    """
    ids: list[int] = []
    for func in analysis.functions:
        func_lines = set(range(func.line, func.end_line + 1))
        if func_lines & changed_line_set:
            ids.append(func.symbol_id)
    for cls in analysis.classes:
        cls_lines = set(range(cls.start_line, cls.end_line + 1))
        if cls_lines & changed_line_set:
            ids.append(cls.symbol_id)
    return ids


def compute_ripple_candidates(
    analysis: FileAnalysis,
    changed_symbol_ids: list[int],
    max_depth: int = 3,
) -> list[dict]:
    """For each changed symbol, find callers and type dependents that might
    need updating.

    Returns a list of dicts:
        {
            'changed_symbol_id': int,
            'changed_symbol_name': str,
            'candidates': [
                {
                    'symbol_id': int,
                    'symbol_name': str,
                    'relation': str,  # 'caller' | 'type_dependent' | 'co_member'
                    'depth': int,     # BFS depth from the changed symbol
                }
            ]
        }

    Ripple relations:
    - 'caller': a function that calls the changed function (reverse edge)
    - 'type_dependent': a function that uses a changed class as parameter/return type
    - 'co_member': another method in the same class as a changed method
    """
    if not changed_symbol_ids:
        return []

    symbols = build_symbol_table(analysis)
    id_to_sym = {s.id: s for s in symbols}
    functions_by_id = {func.symbol_id: func for func in analysis.functions}
    classes_by_id = {cls.symbol_id: cls for cls in analysis.classes}
    local_index = analysis.build_local_index()

    # Build reverse call graph by canonical callee identity.
    callee_to_callers: dict[str, list[int]] = defaultdict(list)
    for func in analysis.functions:
        for callee_key in local_index._function_callee_keys(func):
            callee_to_callers[callee_key].append(func.symbol_id)

    # Class membership is captured from clang semantic children during parsing.
    class_members: dict[str, list[int]] = defaultdict(list)
    known_function_keys = {func.symbol_key: func.symbol_id for func in analysis.functions}
    for cls in analysis.classes:
        for member_key in cls.member_symbol_keys:
            member_id = known_function_keys.get(member_key)
            if member_id is not None:
                class_members[cls.symbol_key].append(member_id)

    class_keys = {cls.symbol_key: cls.symbol_id for cls in analysis.classes}
    func_uses_class: dict[int, list[int]] = defaultdict(list)
    for func in analysis.functions:
        for type_key in func.referenced_type_keys:
            class_id = class_keys.get(type_key)
            if class_id is not None:
                func_uses_class[func.symbol_id].append(class_id)

    result = []
    for changed_id in changed_symbol_ids:
        sym = id_to_sym.get(changed_id)
        if sym is None:
            continue

        candidates: list[dict] = []
        seen: set[int] = {changed_id}

        if sym.kind == 'function':
            # BFS through reverse call graph to find callers
            queue = deque([(sym.symbol_key, 1)])
            visited_keys: set[str] = {sym.symbol_key}
            while queue:
                symbol_key, depth = queue.popleft()
                if depth > max_depth:
                    continue
                for caller_id in callee_to_callers.get(symbol_key, []):
                    if caller_id not in seen:
                        seen.add(caller_id)
                        caller_func = functions_by_id[caller_id]
                        candidates.append({
                            'symbol_id': caller_func.symbol_id,
                            'symbol_name': caller_func.qualified_name,
                            'relation': 'caller',
                            'depth': depth,
                        })
                        if depth < max_depth and caller_func.symbol_key not in visited_keys:
                            visited_keys.add(caller_func.symbol_key)
                            queue.append((caller_func.symbol_key, depth + 1))

            # co_member: other methods in the same class
            for _class_key, member_ids in class_members.items():
                if changed_id in member_ids:
                    for mid in member_ids:
                        if mid not in seen:
                            seen.add(mid)
                            candidates.append({
                                'symbol_id': mid,
                                'symbol_name': functions_by_id[mid].qualified_name,
                                'relation': 'co_member',
                                'depth': 1,
                            })

        elif sym.kind == 'class':
            cls = classes_by_id[changed_id]
            # type_dependent: functions with an exact clang TYPE_REF to this class.
            for func in analysis.functions:
                if func.symbol_id not in seen and changed_id in func_uses_class[func.symbol_id]:
                    seen.add(func.symbol_id)
                    candidates.append({
                        'symbol_id': func.symbol_id,
                        'symbol_name': func.qualified_name,
                        'relation': 'type_dependent',
                        'depth': 1,
                    })
            # co_member: methods of this class
            for mid in class_members.get(cls.symbol_key, []):
                if mid not in seen:
                    seen.add(mid)
                    candidates.append({
                        'symbol_id': mid,
                        'symbol_name': functions_by_id[mid].qualified_name,
                        'relation': 'co_member',
                        'depth': 1,
                    })

        if candidates:
            result.append({
                'changed_symbol_id': changed_id,
                'changed_symbol_name': sym.qualified_name,
                'candidates': candidates,
            })

    return result


# ---------------------------------------------------------------------------
# Transitive dependency chain extraction
# ---------------------------------------------------------------------------

def extract_function_chain(
    analysis: FileAnalysis,
    changed_indices: list[int],
    max_depth: int = 5,
) -> list[FunctionDef]:
    """Extract changed functions + transitive deps, sorted by dep_level (leaves first)."""
    if not changed_indices:
        return []

    # Build local index for dep computation
    idx = analysis.build_local_index()

    # Collect all transitive deps for each changed function
    all_symbol_keys: set[str] = set()
    for i in changed_indices:
        func = analysis.functions[i]
        all_symbol_keys.add(func.symbol_key)
        # BFS through local call graph
        visited = {func.symbol_key}
        queue = deque([(func.symbol_key, 0)])
        while queue:
            symbol_key, depth = queue.popleft()
            if depth >= max_depth:
                continue
            f = idx.functions.get(symbol_key)
            if not f:
                continue
            for callee in idx._function_callee_keys(f):
                if callee not in visited and callee in idx.functions:
                    visited.add(callee)
                    all_symbol_keys.add(callee)
                    queue.append((callee, depth + 1))

    # Collect FunctionDefs, sorted by dep_level (leaves first)
    result = []
    for func in analysis.functions:
        if func.symbol_key in all_symbol_keys:
            result.append(func)
    result.sort(key=lambda f: f.dep_level)
    return result


def _macro_dependency_parts_for_commit_targets(
    macro_index: ProjectIndex | None,
    targets: list[tuple[str, str | None, int | None]],
    *,
    claim_part: Callable[[str], bool] | None = None,
    chunk_claim_stats: dict[str, int] | None = None,
) -> list[PartInfo]:
    """Build commit macro dependency parts from source-visible macro defs.

    Static code docs already prepend macro dependencies before a root chunk.
    Commit docs need the same shape, otherwise PRE/POST function chains contain
    macro invocations whose definitions are only implicit in headers.  Targets
    are ``(text, source_file, max_line)`` tuples so redefinition windows route
    through the macro definition visible at the original source location.
    """

    if macro_index is None or not macro_index.macros_by_name or not targets:
        return []
    selected: dict[tuple[str, str, int, int], MacroDef] = {}
    for text, target_file, max_line in targets:
        if not text:
            continue
        for macro in _used_macro_defs(
            macro_index,
            [text],
            target_file=target_file,
            max_line=max_line,
        ):
            key = (macro.visible_in_file, macro.file, macro.line, macro.sequence)
            selected[key] = macro

    parts: list[PartInfo] = []
    for macro in sorted(selected.values(), key=lambda item: item.sequence):
        if claim_part is not None and not claim_part(macro.text):
            if chunk_claim_stats is not None:
                chunk_claim_stats["commit_macro_chunks_skipped"] = (
                    chunk_claim_stats.get("commit_macro_chunks_skipped", 0) + 1
                )
            continue
        if chunk_claim_stats is not None:
            chunk_claim_stats["commit_macro_chunks_claimed"] = (
                chunk_claim_stats.get("commit_macro_chunks_claimed", 0) + 1
            )
        parts.append(_macro_part(macro))
    return parts


# ---------------------------------------------------------------------------
# Document formatting
# ---------------------------------------------------------------------------

# Trailing PR-number marker in commit subjects, e.g. "Fix foo (#1234)".
_SUBJECT_PR_RE = re.compile(r'\(#(\d+)\)\s*$')
# Body trailers referencing a PR/issue, e.g. "Closes #42", "fixes #7".
_BODY_PR_RE = re.compile(
    r'(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)[ :]+#(\d+)',
    re.IGNORECASE,
)
# Git trailer lines we KEEP in the NL docstring (provenance / review signal).
_KEPT_TRAILER_RE = re.compile(
    r'^(Closes|Fixes|Resolves|Reviewed-by|Reviewed-on|Change-Id|Bug|'
    r'Differential Revision)\s*:',
    re.IGNORECASE,
)
# Git trailer lines we STRIP from the NL docstring (noise / boilerplate).
_STRIPPED_TRAILER_RE = re.compile(
    r'^(Signed-off-by|Co-authored-by)\s*:',
    re.IGNORECASE,
)


def parse_pr_number(record: dict) -> Optional[int]:
    """Parse a PR/issue number from the subject trailer or body trailers.

    Precedence: explicit ``pr_number`` already on the record (e.g. attached by
    merge-commit mining) > subject ``(#N)`` trailer > body ``Closes/Fixes #N``.
    Returns None when no number can be parsed.
    """
    existing = record.get('pr_number')
    if existing not in (None, '', 0):
        return int(existing)

    subject = record.get('subject', '') or ''
    m = _SUBJECT_PR_RE.search(subject)
    if m:
        return int(m.group(1))

    body = record.get('body', '') or ''
    m = _BODY_PR_RE.search(body)
    if m:
        return int(m.group(1))

    return None


def _filter_body_trailers(body: str) -> list[str]:
    """Return body lines with Signed-off-by/Co-authored-by trailers stripped.

    Kept trailers (Closes/Fixes/Resolves/Reviewed-by/Reviewed-on/Change-Id/
    Bug/Differential Revision) and ordinary prose are preserved verbatim.
    """
    out: list[str] = []
    for line in body.splitlines():
        if _STRIPPED_TRAILER_RE.match(line.strip()):
            continue
        out.append(line)
    return out


def build_docstring(record: dict) -> str:
    """Build C++ docstring comment from commit metadata.

    Emits provenance lines (@pr / @repo / @sha) so the SHA, repo, and parsed PR
    number survive into the natural-language docstring that the model trains on,
    and so downstream joins (e.g. GitHub-Archive PR text) have a key. Strips
    Signed-off-by / Co-authored-by trailers from the body but keeps review /
    issue-reference trailers, while still capturing the FULL body as @details.
    """
    parts = ['/**']
    subject = record.get('subject', '').strip()
    if subject:
        parts.append(f' * @brief {subject}')
        parts.append(' *')

    repo = record.get('repo', '')
    filepath = record.get('filepath', '')
    commit_hash = record.get('commit_hash', '')
    pr_number = parse_pr_number(record)
    pr_title = (record.get('pr_title') or '').strip()
    source_branch = (record.get('source_branch') or '').strip()

    # Provenance block: repo / sha / pr keep the join keys in the NL text.
    if repo:
        parts.append(f' * @repo {repo}')
    if filepath:
        parts.append(f' * File: {filepath}')
    if commit_hash:
        parts.append(f' * @sha {commit_hash}')
    if pr_number is not None:
        if pr_title:
            parts.append(f' * @pr {pr_number} {pr_title}')
        else:
            parts.append(f' * @pr {pr_number}')
    if source_branch:
        parts.append(f' * @branch {source_branch}')

    # PR discussion (Tier-2 GH Archive + GraphQL join, attached by the extractor
    # as record['pr_discussion']). Emitted at the HEAD of the block — right after
    # provenance and BEFORE note_text / body / (and thus before PRE/POST/diff in
    # the document) — so the real PR thread (title, body, comments, reviews,
    # linked issues) leads the commit doc. A bigger discussion grows the doc's
    # token count, which routes it to a larger length bucket via route-by-fit.
    pr_discussion = (record.get('pr_discussion') or '').strip()
    if pr_discussion:
        parts.append(' *')
        parts.append(' * @discussion')
        for line in pr_discussion.splitlines():
            parts.append(f' * {line.rstrip()}')

    # Gerrit / code-review note text (best-effort, attached by the extractor).
    note_text = (record.get('note_text') or '').strip()
    if note_text:
        parts.append(' *')
        for line in note_text.splitlines()[:12]:
            parts.append(f' * {line.rstrip()}')

    # Full commit/PR body as @details: the rationale, design discussion, and any
    # spec text the author put in the message (squash-merged PRs carry the whole
    # PR description here). Capture it ALL (200-line safety cap vs pathological
    # bodies; the per-doc token limit downstream filters genuine giants). The
    # Signed-off-by / Co-authored-by trailers are stripped first (noise); review
    # / issue-reference trailers and ordinary prose are kept verbatim.
    body = record.get('body', '').strip()
    if body:
        parts.append(' *')
        parts.append(' * @details')
        for line in _filter_body_trailers(body)[:200]:
            parts.append(f' * {line.rstrip()}')

    parts.append(' */')
    return '\n'.join(parts)


def format_chain_document(
    record: dict,
    old_analysis: FileAnalysis,
    new_analysis: FileAnalysis,
    hunks: list[HunkRange],
    max_dep_depth: int = 5,
    *,
    macro_index: ProjectIndex | None = None,
    dedup_store=None,
    dedup_tokenizer=None,
    chunk_claim_stats: dict[str, int] | None = None,
) -> Optional[dict]:
    """Format A: PRE-COMMIT chain → POST-COMMIT chain with enriched metadata."""
    old_lines = changed_lines(hunks, use_old=True)
    new_lines = changed_lines(hunks, use_old=False)

    old_changed_funcs = find_changed_functions(old_analysis, old_lines)
    new_changed_funcs = find_changed_functions(new_analysis, new_lines)
    old_changed_classes = find_changed_classes(old_analysis, old_lines)
    new_changed_classes = find_changed_classes(new_analysis, new_lines)

    if (not old_changed_funcs and not new_changed_funcs
            and not old_changed_classes and not new_changed_classes):
        return None

    old_chain = extract_function_chain(old_analysis, old_changed_funcs, max_dep_depth)
    new_chain = extract_function_chain(new_analysis, new_changed_funcs, max_dep_depth)

    # Build parts_info: (text, kind, dep_level, name, qname)
    parts_info: list[PartInfo] = []
    section_kinds: list[str] = []
    semantic_parts = 0
    macro_targets: list[tuple[str, str | None, int | None]] = []

    def _claim_part(text: str) -> bool:
        ok = _claim_semantic_chunk(
            text,
            dedup_store=dedup_store,
            dedup_tokenizer=dedup_tokenizer,
            max_count=1,
        )
        if chunk_claim_stats is not None:
            key = 'commit_chunks_claimed' if ok else 'commit_chunks_skipped'
            chunk_claim_stats[key] = chunk_claim_stats.get(key, 0) + 1
        return ok

    # Docstring (kind=6: comment)
    docstring = build_docstring(record)
    parts_info.append((docstring, 6, 0, '', None))
    section_kinds.append('c')

    # PRE-COMMIT section
    if old_chain or old_changed_classes:
        parts_info.append(('// === PRE-COMMIT ===', 0, 0, '', None))
        section_kinds.append('c')
        short_preamble = '\n'.join(old_analysis.preamble.splitlines()[:20])
        if short_preamble:
            parts_info.append((short_preamble, 1, 0, '', None))
            section_kinds.append('o')
        for ci in old_changed_classes:
            cls = old_analysis.classes[ci]
            if not _claim_part(cls.text):
                continue
            parts_info.append(_class_part(cls))
            section_kinds.append('o')
            macro_targets.append((cls.text, record.get('filepath'), cls.end_line))
            semantic_parts += 1
        for func in old_chain:
            if not _claim_part(func.text):
                continue
            parts_info.append(_function_part(func, kind=3))
            section_kinds.append('o')
            macro_targets.append((func.text, func.file, func.end_line))
            semantic_parts += 1

    # POST-COMMIT section
    subject = record.get('subject', '').strip()
    if new_chain or new_changed_classes:
        parts_info.append((f'// === POST-COMMIT: {subject} ===', 0, 0, '', None))
        section_kinds.append('c')
        short_preamble = '\n'.join(new_analysis.preamble.splitlines()[:20])
        if short_preamble:
            parts_info.append((short_preamble, 1, 0, '', None))
            section_kinds.append('n')
        for ci in new_changed_classes:
            cls = new_analysis.classes[ci]
            if not _claim_part(cls.text):
                continue
            parts_info.append(_class_part(cls))
            section_kinds.append('n')
            macro_targets.append((cls.text, record.get('filepath'), cls.end_line))
            semantic_parts += 1
        for func in new_chain:
            if not _claim_part(func.text):
                continue
            parts_info.append(_function_part(func, kind=3))
            section_kinds.append('n')
            macro_targets.append((func.text, func.file, func.end_line))
            semantic_parts += 1

    if len(parts_info) <= 1 or semantic_parts == 0:
        return None

    macro_parts = _macro_dependency_parts_for_commit_targets(
        macro_index,
        macro_targets,
        claim_part=_claim_part,
        chunk_claim_stats=chunk_claim_stats,
    )
    if macro_parts:
        parts_info[1:1] = macro_parts
        section_kinds[1:1] = ['c'] * len(macro_parts)

    # Compute changed_symbol_ids and ripple_candidates from the NEW analysis
    # (post-commit state is what the model learns to predict).  Fall back to
    # old analysis when new is empty.
    primary = new_analysis if new_analysis.functions or new_analysis.classes else old_analysis
    csids = compute_changed_symbol_ids(primary, new_lines if (new_analysis.functions or new_analysis.classes) else old_lines)
    ripple = compute_ripple_candidates(primary, csids)

    return _build_enriched_from_parts(
        parts_info, old_analysis, new_analysis, record,
        changed_symbol_ids=csids, ripple_candidates=ripple,
        section_kinds=section_kinds,
    )


def format_diff_document(
    record: dict,
    old_analysis: FileAnalysis,
    hunks: list[HunkRange],
    max_dep_depth: int = 5,
    *,
    macro_index: ProjectIndex | None = None,
    dedup_store=None,
    dedup_tokenizer=None,
    chunk_claim_stats: dict[str, int] | None = None,
) -> Optional[dict]:
    """Format B: Context code from old version + raw unified diff."""
    old_lines = changed_lines(hunks, use_old=True)
    old_changed_funcs = find_changed_functions(old_analysis, old_lines)
    old_changed_classes = find_changed_classes(old_analysis, old_lines)

    if not old_changed_funcs and not old_changed_classes:
        return None

    old_chain = extract_function_chain(old_analysis, old_changed_funcs, max_dep_depth)

    parts_info: list[PartInfo] = []
    section_kinds: list[str] = []
    macro_targets: list[tuple[str, str | None, int | None]] = []

    def _claim_part(text: str) -> bool:
        ok = _claim_semantic_chunk(
            text,
            dedup_store=dedup_store,
            dedup_tokenizer=dedup_tokenizer,
            max_count=1,
        )
        if chunk_claim_stats is not None:
            key = 'commit_chunks_claimed' if ok else 'commit_chunks_skipped'
            chunk_claim_stats[key] = chunk_claim_stats.get(key, 0) + 1
        return ok

    # Docstring
    docstring = build_docstring(record)
    parts_info.append((docstring, 6, 0, '', None))
    section_kinds.append('c')

    # Context: old code chain
    parts_info.append(('// === CONTEXT ===', 0, 0, '', None))
    section_kinds.append('c')
    short_preamble = '\n'.join(old_analysis.preamble.splitlines()[:20])
    if short_preamble:
        parts_info.append((short_preamble, 1, 0, '', None))
        section_kinds.append('o')
    for ci in old_changed_classes:
        cls = old_analysis.classes[ci]
        if not _claim_part(cls.text):
            continue
        parts_info.append(_class_part(cls))
        section_kinds.append('o')
        macro_targets.append((cls.text, record.get('filepath'), cls.end_line))
    for func in old_chain:
        if not _claim_part(func.text):
            continue
        parts_info.append(_function_part(func, kind=3))
        section_kinds.append('o')
        macro_targets.append((func.text, func.file, func.end_line))

    # Raw diff
    diff_text = record.get('diff', '')
    if diff_text:
        parts_info.append(('// === DIFF ===', 0, 0, '', None))
        section_kinds.append('c')
        parts_info.append((diff_text, 0, 0, '', None))
        section_kinds.append('c')

    if len(parts_info) <= 2:
        return None

    macro_parts = _macro_dependency_parts_for_commit_targets(
        macro_index,
        macro_targets,
        claim_part=_claim_part,
        chunk_claim_stats=chunk_claim_stats,
    )
    if macro_parts:
        parts_info[1:1] = macro_parts
        section_kinds[1:1] = ['c'] * len(macro_parts)

    # Compute changed_symbol_ids and ripple_candidates from old analysis
    csids = compute_changed_symbol_ids(old_analysis, old_lines)
    ripple = compute_ripple_candidates(old_analysis, csids)

    return _build_enriched_from_parts(
        parts_info, old_analysis, None, record,
        changed_symbol_ids=csids, ripple_candidates=ripple,
        section_kinds=section_kinds,
    )


def _analysis_ast_maps(
    analysis: FileAnalysis | None,
) -> dict[str, tuple[list[int], list[int], list[int]]]:
    if analysis is None:
        return {}
    result: dict[str, tuple[list[int], list[int], list[int]]] = {}
    for func in analysis.functions:
        result[func.symbol_key] = (
            func.ast_depth,
            func.sibling_index,
            func.ast_node_type,
        )
    for cls in analysis.classes:
        result[cls.symbol_key] = (
            cls.ast_depth,
            cls.sibling_index,
            cls.ast_node_type,
        )
    return result


def _copy_clang_ast_part(
    ast_depth: list[int],
    sibling_index: list[int],
    ast_node_type: list[int],
    *,
    offset: int,
    part_len: int,
    source: tuple[list[int], list[int], list[int]] | None,
) -> None:
    if source is None:
        return
    src_depth, src_sibling, src_node_type = source
    if (
        len(src_depth) != part_len
        or len(src_sibling) != part_len
        or len(src_node_type) != part_len
    ):
        return
    end = offset + part_len
    ast_depth[offset:end] = src_depth
    sibling_index[offset:end] = src_sibling
    ast_node_type[offset:end] = src_node_type


def _attach_commit_provenance(result: dict, record: dict) -> None:
    result['repo'] = record.get('repo', '')
    result['filepath'] = record.get('filepath', '')
    result['commit_hash'] = record.get('commit_hash', record.get('commit', ''))
    result['timestamp'] = record.get('timestamp', '')
    result['pr_number'] = parse_pr_number(record)
    result['pr_title'] = record.get('pr_title', '')
    result['pr_discussion'] = record.get('pr_discussion', '')
    result['source_branch'] = record.get('source_branch', '')
    result['parent_hashes'] = list(record.get('parent_hashes', []) or [])
    result['parent_count'] = record.get('parent_count')
    result['is_merge_commit'] = record.get('is_merge_commit')
    result['author_timestamp'] = record.get('author_timestamp')
    result['commit_timestamp'] = record.get('commit_timestamp')
    result['repo_stable_id'] = record.get('repo_stable_id')
    result['filepath_stable_id'] = record.get('filepath_stable_id')
    result['file_local_commit_index'] = record.get('file_local_commit_index')
    result['has_ambiguous_reconstruction'] = record.get(
        'has_ambiguous_reconstruction', False
    )
    result['has_rename_ambiguity'] = record.get('has_rename_ambiguity', False)
    result['change_status'] = record.get('change_status', '')
    result['old_filepath'] = record.get('old_filepath', '')


def _build_enriched_from_parts(
    parts_info: list[PartInfo],
    old_analysis: FileAnalysis,
    new_analysis: Optional[FileAnalysis],
    record: dict,
    changed_symbol_ids: Optional[list[int]] = None,
    ripple_candidates: Optional[list[dict]] = None,
    section_kinds: Optional[list[str]] = None,
) -> dict:
    """Build enriched document dict from parts_info list.

    Produces the same schema as build_enriched_doc() in index_project.py:
    text, structure_ids, chunk_boundaries, call_edges, type_edges,
    ast_depth, sibling_index, ast_node_type, symbol_ids, call_targets,
    type_refs, def_use.

    Additionally emits:
    - changed_symbol_ids: list of int — IDs of symbols modified in this commit
    - ripple_candidates: list of dicts — symbols that may need updating due to changes
    """
    texts = [p[0] for p in parts_info]
    full_text = '\n\n'.join(texts)
    text_len = len(full_text)

    structure_ids = [0] * text_len
    ast_depth = [0] * text_len
    sibling_index = [0] * text_len
    ast_node_type = [0] * text_len
    chunk_boundaries = []
    offset = 0

    # Edge routing uses (commit section, canonical symbol key), never qname.
    chunk_symbols: dict[int, tuple[str, str]] = {}
    chunk_callees: dict[int, list[str]] = {}
    chunk_all_symbols: dict[int, tuple[str, str]] = {}
    chunk_types: dict[int, list[str]] = {}

    old_funcs = {func.symbol_key: func for func in old_analysis.functions}
    new_funcs = {
        func.symbol_key: func for func in (new_analysis.functions if new_analysis else [])
    }
    old_classes = {cls.symbol_key: cls for cls in old_analysis.classes}
    new_classes = {
        cls.symbol_key: cls for cls in (new_analysis.classes if new_analysis else [])
    }
    semantic_index = ProjectIndex()
    for func in old_funcs.values():
        semantic_index.add_function(func)
    for func in new_funcs.values():
        semantic_index.add_function(func)
    for cls in list(old_classes.values()) + list(new_classes.values()):
        semantic_index.add_typedef(TypeDef(
            name=cls.name,
            qualified_name=cls.qualified_name,
            file=cls.file,
            line=cls.start_line,
            end_line=cls.end_line,
            text=cls.text,
            kind=4,
            symbol_key=cls.symbol_key,
            usr=cls.usr,
            canonical_signature=cls.canonical_signature,
            symbol_kind=cls.symbol_kind,
        ))
        semantic_index.symbol_id_registry.register_records(
            cls.semantic_symbol_identities,
            source=f"commit class:{cls.file}:{cls.start_line}:{cls.qualified_name}",
        )
    for part in parts_info:
        metadata = _part_macro_provenance(part)
        if metadata is None:
            continue
        name = metadata.get("name")
        sequence = metadata.get("sequence")
        if not isinstance(name, str) or not isinstance(sequence, int):
            continue
        file = str(metadata.get("file") or "")
        line = int(metadata.get("line") or 1)
        semantic_index.add_macro(
            MacroDef(
                name=name,
                file=file,
                line=line,
                text=part[0],
                project_id=require_project_identity(
                    metadata.get("project_id"),
                    source=f"commit macro metadata {file}:{line}:{name}",
                ),
                visible_in_file=str(metadata.get("visible_in_file") or file),
                visible_line=int(metadata.get("visible_line") or line),
                sequence=sequence,
                condition_names=[
                    str(value)
                    for value in metadata.get("condition_names", [])
                    if isinstance(value, str)
                ],
            )
        )
    old_ast = _analysis_ast_maps(old_analysis)
    new_ast = _analysis_ast_maps(new_analysis)
    part_functions: dict[int, FunctionDef] = {}
    part_semantic_arrays: dict[int, dict[str, object]] = {}
    macro_route_parts: list[dict[str, object]] = []
    macro_invocation_routes: list[dict[str, object]] = []

    def _unique_qname_value(values, qname: str):
        normalized_qname = normalize_qualified_name(qname)
        matches = [
            value
            for value in values
            if normalize_qualified_name(value.qualified_name) == normalized_qname
        ]
        return matches[0] if len(matches) == 1 else None

    for i, part in enumerate(parts_info):
        part_text, kind, dep_level, name, qname = part[0], part[1], part[2], part[3], part[4]
        macro_metadata = _part_macro_provenance(part)
        part_len = len(part_text)
        if offset + part_len > text_len:
            break

        for j in range(offset, offset + part_len):
            structure_ids[j] = kind
        source_kind = section_kinds[i] if section_kinds and i < len(section_kinds) else 'c'
        symbol_key = _part_symbol_key(part)
        if source_kind == 'n':
            source_funcs = new_funcs
            source_classes = new_classes
        elif source_kind == 'o':
            source_funcs = old_funcs
            source_classes = old_classes
        else:
            source_funcs = {**old_funcs, **new_funcs}
            source_classes = {**old_classes, **new_classes}
        func = source_funcs.get(symbol_key) if symbol_key else None
        cls = source_classes.get(symbol_key) if symbol_key else None
        if symbol_key is None and isinstance(qname, str):
            func = _unique_qname_value(source_funcs.values(), qname)
            cls = _unique_qname_value(source_classes.values(), qname)
            selected = func or cls
            symbol_key = selected.symbol_key if selected is not None else None

        part_ast: tuple[list[int], list[int], list[int]] | None = None
        if symbol_key:
            if source_kind == 'n':
                part_ast = new_ast.get(symbol_key)
            else:
                part_ast = old_ast.get(symbol_key)
        elif kind == 1:
            analysis = new_analysis if source_kind == 'n' and new_analysis else old_analysis
            if analysis and len(analysis.preamble_ast_depth) >= part_len:
                part_ast = (
                    analysis.preamble_ast_depth[:part_len],
                    analysis.preamble_sibling_index[:part_len],
                    analysis.preamble_ast_node_type[:part_len],
                )
        _copy_clang_ast_part(
            ast_depth,
            sibling_index,
            ast_node_type,
            offset=offset,
            part_len=part_len,
            source=part_ast,
        )

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
        chunk_boundaries.append(boundary)

        if symbol_key and (func is not None or cls is not None):
            chunk_all_symbols[i] = (source_kind, symbol_key)
        if func is not None and symbol_key:
            chunk_symbols[i] = (source_kind, symbol_key)
            callee_keys: list[str] = []
            for reference in list(func.callee_refs) + list(func.baselib_callee_refs):
                key = reference.get('symbol_key')
                if isinstance(key, str) and key and key not in callee_keys:
                    callee_keys.append(key)
            for key in list(func.callee_keys):
                if key and key not in callee_keys:
                    callee_keys.append(key)
            if not callee_keys:
                for display_name in func.callees:
                    key = semantic_index.resolve_function_key(display_name)
                    if key is not None and key not in callee_keys:
                        callee_keys.append(key)
            chunk_callees[i] = callee_keys
            type_keys = list(func.referenced_type_keys)
            for reference in func.referenced_type_refs:
                key = reference.get('symbol_key')
                if isinstance(key, str) and key and key not in type_keys:
                    type_keys.append(key)
            chunk_types[i] = type_keys
            part_functions[i] = func
            part_semantic_arrays[i] = {
                'symbol_ids': func.semantic_symbol_ids,
                'call_targets': func.semantic_call_targets,
                'type_refs': func.semantic_type_refs,
                'def_use': func.semantic_def_use,
            }
            macro_invocation_routes.extend(
                _macro_invocation_route_parts(
                    part_text,
                    offset=offset,
                    index=semantic_index,
                    target_file=func.file,
                    start_line=func.line,
                )
            )
        elif cls is not None:
            part_semantic_arrays[i] = {
                'symbol_ids': cls.semantic_symbol_ids,
                'call_targets': cls.semantic_call_targets,
                'type_refs': cls.semantic_type_refs,
                'def_use': cls.semantic_def_use,
            }
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

    # Compute call_edges between chunks
    call_edges = []
    for ci, (source_kind, _caller_symbol) in chunk_symbols.items():
        callees = chunk_callees.get(ci, [])
        for callee_symbol in callees:
            for cj, target_symbol in chunk_symbols.items():
                if ci != cj and target_symbol == (source_kind, callee_symbol):
                    call_edges.append({'from': ci, 'to': cj})

    # Compute type_edges: function chunk referencing type T -> chunk defining T.
    type_edges: list[dict[str, object]] = []
    for ci, ref_types in chunk_types.items():
        source_kind = chunk_symbols[ci][0]
        for type_symbol in ref_types:
            for cj, target_symbol in chunk_all_symbols.items():
                if ci != cj and target_symbol == (source_kind, type_symbol):
                    type_edges.append({'from': ci, 'to': cj})

    semantic_meta = extract_semantic_metadata_from_parts(
        full_text,
        parts_info,
        semantic_index,
        part_functions=part_functions,
        part_semantic_arrays=part_semantic_arrays,
    )
    ripple_symbol_ids = [
        int(symbol_id)
        for ripple in (ripple_candidates or [])
        for symbol_id in (
            [ripple.get('changed_symbol_id')]
            + [candidate.get('symbol_id') for candidate in ripple.get('candidates', [])]
        )
        if symbol_id is not None
    ]
    symbol_identities = _document_symbol_identities(
        parts_info,
        semantic_index,
        semantic_meta['symbol_ids'],
        semantic_meta['call_targets'],
        semantic_meta['type_refs'],
        changed_symbol_ids or [],
        ripple_symbol_ids,
        source=str(record.get('filepath') or 'clang commit document'),
    )

    result: dict = {
        'text': full_text,
        'symbol_identity_schema_version': SYMBOL_IDENTITY_SCHEMA_VERSION,
        SYMBOL_IDENTITIES_COLUMN: symbol_identities,
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
        'changed_symbol_ids': changed_symbol_ids or [],
        'ripple_candidates': ripple_candidates or [],
    }
    typed_message_parts = [
        value.strip()
        for value in (record.get('subject'), record.get('body'))
        if isinstance(value, str) and value.strip()
    ]
    typed_commit_message = '\n\n'.join(typed_message_parts)
    typed_section_kinds = section_kinds or ['c'] * len(texts)
    result.update(
        {
            # These fields come from typed upstream values. Downstream objective
            # code must never recover them by parsing rendered doc wrappers.
            'ifim_instruction_text': typed_commit_message,
            'commit_msg_text': typed_commit_message,
            'pre_text': '\n\n'.join(
                text
                for text, section_kind in zip(texts, typed_section_kinds)
                if section_kind == 'o'
            ),
            'post_text': '\n\n'.join(
                text
                for text, section_kind in zip(texts, typed_section_kinds)
                if section_kind == 'n'
            ),
            'diff_text': (
                record.get('diff', '')
                if isinstance(record.get('diff', ''), str)
                else ''
            ),
        }
    )
    # Commit docs are also C++ world-code documents.  When macro dependency
    # parts are present, route use-sites by original source line instead of
    # assembled lexical order, matching the static code path.
    result.update(
        _cpp_domain_sidecars(
            full_text,
            semantic_index,
            source_doc_id=1,
            source_path=str(record.get("filepath") or "commit.cpp"),
            macro_parts=macro_route_parts,
            macro_invocations=macro_invocation_routes,
        )
    )
    temporal_meta = _build_commit_temporal_metadata(
        full_text,
        texts,
        typed_section_kinds,
        record=record,
        old_analysis=old_analysis,
        new_analysis=new_analysis,
    )
    result.update(temporal_meta)

    # Provenance round-trip: thread commit_hash / timestamp / repo / filepath /
    # pr_number (+ raw-chronology fields) from the source record into the doc
    # dict so clang_enriched_to_parquet.py can populate the parquet columns.
    # Without this the SHA + timestamp columns end up 0% populated and the later
    # GitHub-Archive PR-text join has no key.
    _attach_commit_provenance(result, record)

    # Platform info detection mirrors the source-indexer path.
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _v5_dir = os.path.join(_parent, 'v5_gke_orchestrator')
    if _v5_dir not in sys.path:
        sys.path.insert(0, _v5_dir)
    platform_info: dict[str, object] | None
    try:
        _platform_detect = importlib.import_module("platform_detect")
        _detect_plat = cast(
            Callable[[str], dict[str, object] | None],
            getattr(_platform_detect, "detect_platforms"),
        )
        platform_info = _detect_plat(full_text)
    except ImportError:
        platform_info = None
    if platform_info:
        result['platform_info'] = platform_info
    compile_args = record.get("compile_args")
    build_info = record.get("build_info")
    if isinstance(build_info, dict) and build_info:
        result["build_info"] = build_info

    # Language info detection
    filepath = record.get('filepath', '')
    if detect_language_info is not None:
        try:
            lang_info = detect_language_info(
                full_text,
                filepath,
                platform_info,
                compile_args=compile_args if isinstance(compile_args, list) else None,
                build_info=build_info if isinstance(build_info, dict) else None,
            )
        except TypeError:
            lang_info = detect_language_info(full_text, filepath)
        if lang_info:
            result['language_info'] = lang_info

    return result


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_tokenizer = None


def _load_tokenizer(path: Optional[str]):
    global _tokenizer
    if _tokenizer is not None:
        return
    if path and os.path.exists(path):
        try:
            from tokenizers import Tokenizer  # type: ignore[reportMissingImports]
            _tokenizer = Tokenizer.from_file(path)
            return
        except SymbolIdentityError:
            raise
        except Exception:
            pass
    # Fallback: estimate ~4 bytes per token
    _tokenizer = None


def count_tokens(text: str) -> int:
    if _tokenizer is not None:
        return len(_tokenizer.encode(text).ids)
    return max(1, len(text) // 4)


def _stable_repo_id(repo_name: str) -> str:
    return hashlib.sha1(repo_name.encode("utf-8")).hexdigest()[:16]


def _stable_filepath_id(repo_name: str, filepath: str) -> str:
    return hashlib.sha1(
        f"{repo_name}\0{filepath}".encode("utf-8")
    ).hexdigest()[:16]


def normalize_record_project_identity(
    record: dict,
    *,
    project_id: str,
) -> bool:
    """Repair legacy staged-checkout identity before enrichment and PR lookup."""
    canonical_project = require_project_identity(
        project_id,
        source="process_commits --project-id",
    )
    raw_repo = record.get("repo")
    if isinstance(raw_repo, str) and "/" in raw_repo:
        parsed_repo = require_project_identity(
            raw_repo,
            source=f"commit record {record.get('commit_hash') or '<unknown>'}:repo",
        )
        if parsed_repo != canonical_project:
            raise SymbolIdentityError(
                f"commit record {record.get('commit_hash') or '<unknown>'}: "
                f"record project identity {parsed_repo!r} conflicts with "
                f"--project-id {canonical_project!r}"
            )

    filepath = record.get("filepath")
    if not isinstance(filepath, str) or not filepath:
        raise SymbolIdentityError(
            f"commit record {record.get('commit_hash') or '<unknown>'}: "
            f"cannot recompute filepath identity without a non-empty filepath, "
            f"got {filepath!r}"
        )
    expected_repo_id = _stable_repo_id(canonical_project)
    expected_filepath_id = _stable_filepath_id(canonical_project, filepath)
    changed = (
        record.get("repo") != canonical_project
        or record.get("repo_stable_id") != expected_repo_id
        or record.get("filepath_stable_id") != expected_filepath_id
    )
    record["repo"] = canonical_project
    record["repo_stable_id"] = expected_repo_id
    record["filepath_stable_id"] = expected_filepath_id
    return changed


# OUR CppMegaTokenizer (with <SPACE>/<NL> whitespace canonicalization) used ONLY
# for the dedup hash, so the commit-doc hash is sha1(canonical token_ids). This
# is distinct from the plain `tokenizers.Tokenizer` above used for token COUNTS.
_dedup_tokenizer = None


def _load_dedup_tokenizer(path):
    """Load OUR CppMegaTokenizer for dedup hashing. FAIL LOUD (RULE #1)."""
    global _dedup_tokenizer
    if _dedup_tokenizer is not None:
        return _dedup_tokenizer
    if not path:
        raise ValueError("--dedup-db requires --tokenizer-path for the dedup hash")
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
    _dedup_tokenizer = load_cppmega_tokenizer(path)
    return _dedup_tokenizer


SEMANTIC_CHUNK_CLAIM_NAMESPACE = "semantic_chunk:v1"


def _claim_semantic_chunk(
    text: str,
    *,
    dedup_store,
    dedup_tokenizer,
    max_count: int = 1,
) -> bool:
    """Claim one function/class semantic part by its tokenized body.

    Commit docs are still commit examples, but their function/class parts share
    the same claim namespace as static code docs. This keeps the model from
    seeing the same exact function/class form again in 1k/2k/4k/8k streams.
    """
    if dedup_store is None or dedup_tokenizer is None:
        return True
    if not text:
        return False
    token_ids = dedup_tokenizer.encode(text)
    if not token_ids:
        return False
    return dedup_store.claim_chunk_tokens(
        token_ids,
        namespace=SEMANTIC_CHUNK_CLAIM_NAMESPACE,
        max_count=max_count,
    )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _process_domain_record(
    record: dict,
    *,
    filepath: str,
    build_kind: str,
    old_content: str,
    new_content: str,
    max_tokens: int,
    project_id: str,
    build_info: dict[str, object],
) -> list[dict]:
    document_text = new_content or old_content
    tokens = count_tokens(document_text)
    if tokens > max_tokens:
        return []

    document = build_build_doc(
        filepath,
        document_text,
        build_kind,
        project_id=project_id,
        build_info=build_info,
    )
    typed_message = '\n\n'.join(
        value.strip()
        for value in (record.get('subject'), record.get('body'))
        if isinstance(value, str) and value.strip()
    )
    document.update(
        {
            'ifim_instruction_text': typed_message,
            'commit_msg_text': typed_message,
            'pre_text': old_content,
            'post_text': new_content,
            'diff_text': (
                record.get('diff', '')
                if isinstance(record.get('diff', ''), str)
                else ''
            ),
            'changed_symbol_ids': [],
            'ripple_candidates': [],
        }
    )
    empty_analysis = FileAnalysis(preamble='')
    document.update(
        _build_commit_temporal_metadata(
            document_text,
            [document_text],
            ['n' if new_content else 'o'],
            record=record,
            old_analysis=empty_analysis,
            new_analysis=empty_analysis,
        )
    )
    _attach_commit_provenance(document, record)
    document['actual_token_count'] = tokens
    return [document]


def process_record(
    record: dict,
    clang_index: Index | None,
    tmpdir: str,
    max_tokens: int,
    max_file_bytes: int,
    doc_format: str,
    max_dep_depth: int,
    build_context: BuildContextResolver | None = None,
    dedup_store=None,
    dedup_tokenizer=None,
    chunk_claim_stats: dict[str, int] | None = None,
    analysis_cache: AnalysisCache | None = None,
    analyzer: Callable[..., FileAnalysis] | None = None,
) -> list[dict]:
    """Process a single commit record into enriched documents."""
    old_content = record.get('old_content', '')
    new_content = record.get('new_content', '')
    diff = record.get('diff', '')

    if len(old_content) > max_file_bytes or len(new_content) > max_file_bytes:
        return []
    if max(len(old_content), len(new_content)) < 50:
        return []

    hunks = parse_hunk_ranges(diff)
    if not hunks:
        return []

    filepath = record.get('filepath', 'source.cpp')
    primary_kind = classify_primary_commit_path(
        filepath,
        new_content or old_content,
    )
    if primary_kind is None:
        return []
    project_id = require_project_identity(
        record.get("repo"),
        source=f"commit record {record.get('commit_hash') or '<unknown>'}",
    )
    repo_root: str | None = None
    compile_args: list[str] | None = None
    build_info: dict[str, object] = {}
    if build_context is not None:
        repo_root, compile_args, build_info = build_context.resolve(record)
        if compile_args:
            record.setdefault("compile_args", compile_args)
        if build_info:
            record.setdefault("build_info", build_info)

    if primary_kind != 'cpp':
        return _process_domain_record(
            record,
            filepath=filepath,
            build_kind=primary_kind,
            old_content=old_content,
            new_content=new_content,
            max_tokens=max_tokens,
            project_id=project_id,
            build_info=build_info,
        )
    if clang_index is None:
        raise ValueError("C/C++ commit processing requires a libclang index")

    # Parse old and new with clang (use separate temp subdirs to avoid conflicts)
    old_dir = os.path.join(tmpdir, 'old')
    new_dir = os.path.join(tmpdir, 'new')
    os.makedirs(old_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)

    active_analyzer = analyze_file_clang if analyzer is None else analyzer
    if analysis_cache is not None:
        old_analysis = analysis_cache.get_or_analyze(
            old_content,
            filepath,
            clang_index,
            old_dir,
            compile_args=compile_args,
            repo_root=repo_root,
            build_info=build_info,
            project_id=project_id,
            analyzer=active_analyzer,
        )
        new_analysis = analysis_cache.get_or_analyze(
            new_content,
            filepath,
            clang_index,
            new_dir,
            compile_args=compile_args,
            repo_root=repo_root,
            build_info=build_info,
            project_id=project_id,
            analyzer=active_analyzer,
        )
    else:
        old_analysis = active_analyzer(
            old_content,
            filepath,
            clang_index,
            old_dir,
            compile_args=compile_args,
            repo_root=repo_root,
            build_info=build_info,
            project_id=project_id,
        )
        new_analysis = active_analyzer(
            new_content,
            filepath,
            clang_index,
            new_dir,
            compile_args=compile_args,
            repo_root=repo_root,
            build_info=build_info,
            project_id=project_id,
        )

    macro_index = (
        build_context.macro_index_for(
            repo_root=repo_root,
            filepath=filepath,
            compile_args=compile_args,
            project_id=project_id,
            usage_texts=(old_content, new_content),
        )
        if build_context is not None
        else None
    )

    documents: list[dict[str, object]] = []

    if doc_format in ('chain', 'both'):
        doc = format_chain_document(
            record,
            old_analysis,
            new_analysis,
            hunks,
            max_dep_depth,
            macro_index=macro_index,
            dedup_store=dedup_store,
            dedup_tokenizer=dedup_tokenizer,
            chunk_claim_stats=chunk_claim_stats,
        )
        if doc:
            tokens = count_tokens(doc['text'])
            if tokens <= max_tokens and len(doc['text']) >= 100:
                doc['actual_token_count'] = tokens
                documents.append(doc)

    if doc_format in ('diff', 'both'):
        doc = format_diff_document(
            record,
            old_analysis,
            hunks,
            max_dep_depth,
            macro_index=macro_index,
            dedup_store=dedup_store,
            dedup_tokenizer=dedup_tokenizer,
            chunk_claim_stats=chunk_claim_stats,
        )
        if doc:
            tokens = count_tokens(doc['text'])
            if tokens <= max_tokens and len(doc['text']) >= 100:
                doc['actual_token_count'] = tokens
                documents.append(doc)

    return documents


def _uses_typed_change_unit_dedup(document: dict) -> bool:
    return bool(
        document.get('doc_type') in {'build', 'shell', 'sql'}
        and document.get('commit_hash')
        and document.get('diff_text')
    )


def _commit_document_dedup_text(document: dict) -> str:
    if _uses_typed_change_unit_dedup(document):
        return (
            f"COMMIT MESSAGE\n{document.get('commit_msg_text') or ''}\n\n"
            f"UNIFIED DIFF\n{document['diff_text']}"
        )
    return str(document.get('text', ''))


def process_jsonl_file(
    input_path: str,
    output_file,
    clang_index: Index | None,
    tmpdir: str,
    max_tokens: int,
    max_file_bytes: int,
    doc_format: str,
    max_dep_depth: int,
    memory_limit_gb: float,
    build_context: BuildContextResolver | None = None,
    dedup_store=None,
    dedup_tokenizer=None,
    dedup_near: bool = True,
    pr_lookup: "PRDiscussionLookup | None" = None,
    analysis_cache_entries: int = 128,
    project_id: str | None = None,
) -> dict:
    """Process a JSONL input file, writing enriched docs to output.

    A commit is an ATOMIC change-unit. C/C++ rows dedup their rendered document;
    domain rows exact-dedup the typed message plus unified diff and intentionally
    skip lossy near-dedup so similar consecutive post-state files do not erase
    distinct changes. When ``dedup_store`` is given the dedup is GLOBAL +
    resumable + cross-stream (shared SQLite with the code stream). When absent,
    a per-file in-RAM md5 set is used.
    """
    stats = {
        'records_read': 0,
        'documents_written': 0,
        'records_skipped': 0,
        'records_empty': 0,
        'parse_errors': 0,
        'commit_chunks_claimed': 0,
        'commit_chunks_skipped': 0,
        'analysis_cache_hits': 0,
        'analysis_cache_misses': 0,
        'analysis_cache_evictions': 0,
        'legacy_identity_overrides': 0,
    }
    canonical_project_id = (
        require_project_identity(project_id, source="process_commits --project-id")
        if project_id is not None
        else None
    )
    seen_hashes: set[str] = set()
    analysis_cache = (
        AnalysisCache(max_entries=analysis_cache_entries)
        if analysis_cache_entries > 0
        else None
    )

    with open(input_path, 'r', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats['parse_errors'] += 1
                continue

            stats['records_read'] += 1

            if canonical_project_id is not None:
                if normalize_record_project_identity(
                    record,
                    project_id=canonical_project_id,
                ):
                    stats['legacy_identity_overrides'] += 1

            # Tier-2 PR-store lookup glue: set record['pr_discussion'] from the
            # live store (by pr_number then merge/commit SHA) BEFORE the doc is
            # built, so build_docstring emits the @discussion block at the HEAD.
            # A MISS leaves the record Tier-1 (git-only) — never fails (RULE #1:
            # the lookup is best-effort enrichment; a missing PR is normal).
            if pr_lookup is not None:
                pr_lookup.attach(record)

            try:
                docs = process_record(
                    record, clang_index, tmpdir,
                    max_tokens, max_file_bytes, doc_format, max_dep_depth,
                    build_context=build_context,
                    dedup_store=dedup_store,
                    dedup_tokenizer=dedup_tokenizer,
                    chunk_claim_stats=stats,
                    analysis_cache=analysis_cache,
                )
            except SymbolIdentityError:
                raise
            except Exception as e:
                stats['parse_errors'] += 1
                if stats['parse_errors'] <= 10:
                    print(f"  WARN: Record {line_num}: {e}", file=sys.stderr)
                continue

            if not docs:
                stats['records_empty'] += 1
                continue

            for doc in docs:
                dedup_text = _commit_document_dedup_text(doc)
                if dedup_store is not None:
                    # CANONICAL: tokenized-hash dedup of the atomic change unit.
                    token_ids = dedup_tokenizer.encode(dedup_text)
                    if dedup_store.seen_exact_tokens(token_ids):
                        stats['records_skipped'] += 1
                        continue
                    if (
                        dedup_near
                        and not _uses_typed_change_unit_dedup(doc)
                        and dedup_store.seen_near_tokens(token_ids)
                    ):
                        stats['records_skipped'] += 1
                        continue
                else:
                    doc_hash = hashlib.md5(dedup_text.encode()).hexdigest()
                    if doc_hash in seen_hashes:
                        stats['records_skipped'] += 1
                        continue
                    seen_hashes.add(doc_hash)
                output_file.write(json.dumps(doc, ensure_ascii=False) + '\n')
                stats['documents_written'] += 1

            if stats['records_read'] % 100 == 0:
                output_file.flush()
                gc.collect()
                check_memory_limit(memory_limit_gb, label="process_commits")
                print(
                    f"  [{input_path}] {stats['records_read']} records, "
                    f"{stats['documents_written']} docs written",
                    file=sys.stderr,
                )

    if analysis_cache is not None:
        stats['analysis_cache_hits'] = analysis_cache.hits
        stats['analysis_cache_misses'] = analysis_cache.misses
        stats['analysis_cache_evictions'] = analysis_cache.evictions
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Process git commit diffs into enriched training documents using libclang',
    )
    parser.add_argument(
        '--inputs', required=True, nargs='+',
        help='Input JSONL files with commit records (old_content, new_content, diff, ...)',
    )
    parser.add_argument(
        '--output', required=True,
        help='Output enriched JSONL file',
    )
    parser.add_argument(
        '--max-tokens', type=int, default=4096,
        help='Maximum tokens per document (default: 4096)',
    )
    parser.add_argument(
        '--max-file-bytes', type=int, default=500000,
        help='Skip files larger than this (default: 500000)',
    )
    parser.add_argument(
        '--max-dep-depth', type=int, default=5,
        help='Maximum BFS depth for transitive deps (default: 5)',
    )
    parser.add_argument(
        '--format', choices=['chain', 'diff', 'both'], default='both',
        dest='doc_format',
        help='Document format: chain (PRE/POST), diff (context+patch), or both',
    )
    parser.add_argument(
        '--tokenizer-path', default=None,
        help='Path to tokenizer.json for exact token counting (fallback: estimate)',
    )
    parser.add_argument(
        '--libclang-path', default=None,
        help='Explicit path to libclang.so',
    )
    parser.add_argument(
        '--repo-root', default=None,
        help='Single source repo root to use for compile_commands/include resolution',
    )
    parser.add_argument(
        '--repo-dir', default=None,
        help='Parent directory containing repos named by each record["repo"]',
    )
    parser.add_argument(
        '--project-id', default=None,
        help=(
            'Explicit canonical owner/repo identity for every input record. '
            'Required by the conveyor so staged _src paths cannot leak into '
            'PR lookup or output provenance.'
        ),
    )
    parser.add_argument(
        '--memory-limit-gb', type=float, default=10.0,
        help='Abort if this Python wrapper exceeds this max RSS in GiB (default: 10).',
    )
    parser.add_argument(
        '--analysis-cache-entries', type=int, default=128,
        help='Bounded per-range LRU for expensive libclang file analyses. '
             'Default 128; use 0 to disable.',
    )
    parser.add_argument(
        '--allow-parse-errors',
        action='store_true',
        help=(
            'Explicitly allow malformed/failed records to be skipped. Default '
            'fails the whole range so staged dedup and parquet publication roll back.'
        ),
    )
    parser.add_argument(
        '--dedup-db', default=None,
        help='Path to the SHARED global dedup SQLite store. When set, whole commit '
             'DOCS are deduped by their tokenized hash (exact+near) GLOBALLY across '
             'repos AND function/class parts are claimed in the shared semantic '
             'chunk namespace across code+commit streams (fail-loud, no fallback). '
             'Requires --tokenizer-path. When absent, a per-file in-RAM md5 set.',
    )
    parser.add_argument(
        '--dedup-stage-id', default=None,
        help='Optional transactional dedup stage id. When set, commit-doc/chunk '
             'claims are written only to staging tables; the parent conveyor '
             'promotes after successful materialize/pack/append or discards on '
             'failure.',
    )
    parser.add_argument(
        '--dedup-stage-db', default=None,
        help='Optional local SQLite stage DB under rwork. When set with '
             '--dedup-stage-id, this process writes staging claims there while '
             'reading --dedup-db read-only; the parent promotes after append '
             'success.',
    )
    parser.add_argument(
        '--no-near-dedup', action='store_true',
        help='Disable MinHash-LSH near dedup for commit docs (exact-only).',
    )
    parser.add_argument(
        '--pr-store', default=None,
        help='Path to the Tier-2 PR-discussion SQLite store. When set, each '
             'commit record is looked up by (owner_repo, pr_number) then '
             '(owner_repo, commit_hash); on a hit render_discussion populates '
             "record['pr_discussion'] (emitted at the HEAD of the commit doc). "
             'A miss leaves the record Tier-1 (git-only) — never fails.',
    )
    parser.add_argument(
        '--pr-scan-id', default=None,
        help='Exact 64-hex PR scan identity. When set, --pr-store lookups are '
             'restricted to rows proven by that scan; stale archival rows are '
             'treated as normal misses.',
    )
    parser.add_argument(
        '--pr-repo', default=None,
        help='Explicit fixed owner/repo key for every PR-store lookup in this '
             'process. Required with --pr-scan-id; record repo values cannot '
             'override it.',
    )
    parser.add_argument(
        '--repo-list', default=None,
        help='Path to outputs/pr_ingest/repo_list.json (bare-name -> owner/repo '
             'map) used to resolve the PR-store key for records whose repo is a '
             'bare directory name. Optional; records whose repo already contains '
             "'/' are used as-is.",
    )
    args = parser.parse_args()
    if args.pr_scan_id is not None:
        if re.fullmatch(r"[0-9a-f]{64}", args.pr_scan_id) is None:
            print(f"ERROR: invalid --pr-scan-id: {args.pr_scan_id!r}", file=sys.stderr)
            return 1
        if not args.pr_store or not args.pr_repo:
            print(
                "ERROR: --pr-scan-id requires --pr-store and --pr-repo",
                file=sys.stderr,
            )
            return 1
    if args.pr_repo is not None:
        if not args.pr_store:
            print("ERROR: --pr-repo requires --pr-store", file=sys.stderr)
            return 1
        try:
            args.pr_repo = require_project_identity(
                args.pr_repo,
                source="process_commits --pr-repo",
            )
        except SymbolIdentityError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    missing_inputs = [path for path in args.inputs if not os.path.exists(path)]
    if missing_inputs:
        print(
            "ERROR: missing input file(s): " + ", ".join(missing_inputs),
            file=sys.stderr,
        )
        return 1
    if args.project_id is not None:
        try:
            args.project_id = require_project_identity(
                args.project_id,
                source="process_commits --project-id",
            )
        except SymbolIdentityError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    start_memory_guard(args.memory_limit_gb, label="process_commits")

    print("Clang commit processor starting", file=sys.stderr)
    print(f"  inputs: {args.inputs}", file=sys.stderr)
    print(f"  output: {args.output}", file=sys.stderr)
    print(f"  max_tokens: {args.max_tokens}", file=sys.stderr)
    print(f"  format: {args.doc_format}", file=sys.stderr)
    print(f"  memory_limit_gb: {args.memory_limit_gb}", file=sys.stderr)
    print(f"  analysis_cache_entries: {args.analysis_cache_entries}", file=sys.stderr)
    if args.repo_root:
        print(f"  repo_root: {args.repo_root}", file=sys.stderr)
    if args.repo_dir:
        print(f"  repo_dir: {args.repo_dir}", file=sys.stderr)
    if args.project_id:
        print(f"  project_id: {args.project_id}", file=sys.stderr)

    # Configure libclang
    try:
        libclang_path = _configure_libclang(args.libclang_path)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"  libclang: {libclang_path or 'system default'}", file=sys.stderr)

    # Load tokenizer
    _load_tokenizer(args.tokenizer_path)
    if _tokenizer:
        print(f"  tokenizer: {args.tokenizer_path}", file=sys.stderr)
    else:
        print("  tokenizer: estimate (~4 bytes/token)", file=sys.stderr)

    # Commit-doc tokenized-hash dedup store (shared, global, cross-stream).
    # FAIL LOUD (RULE #1): a bad db / missing datasketch / missing tokenizer
    # raises here before any processing -- no silent dup pass.
    dedup_store = None
    dedup_tokenizer = None
    dedup_near = not args.no_near_dedup
    if args.dedup_db:
        if not args.tokenizer_path:
            print("ERROR: --dedup-db requires --tokenizer-path", file=sys.stderr)
            return 1
        if args.dedup_stage_db and not args.dedup_stage_id:
            print("ERROR: --dedup-stage-db requires --dedup-stage-id", file=sys.stderr)
            return 1
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dedup_store import DedupStore
        if args.dedup_stage_id:
            DedupStore.discard_stage(
                args.dedup_db,
                args.dedup_stage_id,
                stage_db_path=args.dedup_stage_db,
            )
        dedup_store = DedupStore(
            args.dedup_db,
            near=dedup_near,
            commit_every=1000,
            stage_id=args.dedup_stage_id,
            stage_db_path=args.dedup_stage_db,
        )
        dedup_tokenizer = _load_dedup_tokenizer(args.tokenizer_path)
        print(
            f"  dedup: {'STAGED' if args.dedup_stage_id else 'GLOBAL'} "
            f"commit-doc store at {args.dedup_db} "
            f"(exact{'+near' if dedup_near else ''}, tokenized hash"
            f"{', stage_id=' + args.dedup_stage_id if args.dedup_stage_id else ''}"
            f"{', stage_db=' + args.dedup_stage_db if args.dedup_stage_db else ''})",
            file=sys.stderr,
        )
    else:
        print("  dedup: per-file in-RAM md5 set (no --dedup-db)", file=sys.stderr)

    # Tier-2 PR-discussion lookup glue (fail-loud on a bad store/repo-list path).
    pr_lookup: PRDiscussionLookup | None = None
    if args.pr_store:
        pr_lookup = PRDiscussionLookup(
            args.pr_store,
            args.repo_list,
            scan_id=args.pr_scan_id,
            owner_repo=args.pr_repo,
        )
        print(f"  pr_store: live lookup at {args.pr_store} "
              f"(pr_repo={args.pr_repo}, repo_list={args.repo_list}, "
              f"scan_id={args.pr_scan_id})",
              file=sys.stderr)

    clang_index = Index.create()
    build_context = BuildContextResolver(repo_root=args.repo_root, repo_dir=args.repo_dir)

    t0 = time.time()
    total_stats = {
        'records_read': 0,
        'documents_written': 0,
        'records_skipped': 0,
        'records_empty': 0,
        'parse_errors': 0,
        'commit_chunks_claimed': 0,
        'commit_chunks_skipped': 0,
        'analysis_cache_hits': 0,
        'analysis_cache_misses': 0,
        'analysis_cache_evictions': 0,
        'legacy_identity_overrides': 0,
    }

    try:
        with atomic_output_file(args.output) as staged_output:
            with tempfile.TemporaryDirectory(prefix='clang_commits_') as tmpdir:
                # Use /dev/shm if available for faster temp file I/O
                shm_tmpdir = None
                if os.path.isdir('/dev/shm'):
                    shm_tmpdir = tempfile.mkdtemp(
                        prefix='clang_commits_', dir='/dev/shm'
                    )
                    actual_tmpdir = shm_tmpdir
                else:
                    actual_tmpdir = tmpdir

                try:
                    with staged_output.open('w') as out_f:
                        for input_path in args.inputs:
                            print(f"\n  Processing {input_path}...", file=sys.stderr)
                            stats = process_jsonl_file(
                                input_path, out_f, clang_index, actual_tmpdir,
                                args.max_tokens, args.max_file_bytes,
                                args.doc_format, args.max_dep_depth,
                                args.memory_limit_gb,
                                build_context=build_context,
                                dedup_store=dedup_store,
                                dedup_tokenizer=dedup_tokenizer,
                                dedup_near=dedup_near,
                                pr_lookup=pr_lookup,
                                analysis_cache_entries=args.analysis_cache_entries,
                                project_id=args.project_id,
                            )

                            for k in total_stats:
                                total_stats[k] += stats[k]

                            print(f"  Done: {stats}", file=sys.stderr)
                finally:
                    if shm_tmpdir:
                        import shutil
                        shutil.rmtree(shm_tmpdir, ignore_errors=True)

            if total_stats['parse_errors'] and not args.allow_parse_errors:
                raise PartialParseError(
                    "refusing partial commit range: "
                    f"{total_stats['parse_errors']} record parse error(s); "
                    "use --allow-parse-errors only for an explicitly lossy run"
                )
    except PartialParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if dedup_store is not None:
            dedup_store.close()
        if pr_lookup is not None:
            print(f"  pr_store: {pr_lookup.hits} discussions attached, "
                  f"{pr_lookup.misses} misses (Tier-1 git-only)", file=sys.stderr)
            pr_lookup.close()

    elapsed = time.time() - t0
    print(f"\nTotal: {total_stats}", file=sys.stderr)
    if args.project_id:
        print(
            "  identity normalization: "
            f"project_id={args.project_id} "
            f"legacy_identity_overrides={total_stats['legacy_identity_overrides']}",
            file=sys.stderr,
        )
    print(f"Time: {elapsed:.1f}s", file=sys.stderr)
    if total_stats['records_read'] > 0:
        rate = total_stats['records_read'] / elapsed
        print(f"Rate: {rate:.1f} records/sec", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
