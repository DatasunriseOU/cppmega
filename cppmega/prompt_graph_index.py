"""Production clang adapter for repository prompt-graph indexes.

The adapter reuses the repository clang indexer's discovery, compile-command,
libclang, and cursor-offset seams. Semantic identity delegates to CASE 4's
``symbol_reference_for_cursor`` when present. Until that lands in the shared
indexer, the compatibility seam transports clang USR plus canonical signature
directly and never derives identity from a qualified name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any

from .prompt_graph import INDEX_SCHEMA, PromptProjectIndex, repository_snapshot


PRODUCER_VERSION = "1"
SYMBOL_IDENTITY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PromptProjectIndexBuildResult:
    index: PromptProjectIndex
    path: Path
    cache_hit: bool
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class _CursorIdentity:
    semantic_identity: str
    symbol_key: str
    usr: str
    canonical_signature: str
    qname: str
    symbol_kind: str
    adapter: str


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha_json(value: Any) -> str:
    return sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _normalize_signature(value: object) -> str:
    return " ".join(str(value or "").split())


def _cursor_usr(cursor: Any) -> str:
    getter = getattr(cursor, "get_usr", None)
    if not callable(getter):
        return ""
    try:
        value = str(getter() or "")
    except Exception:
        return ""
    if not value or value.startswith("<") or "invalid" in value.lower():
        return ""
    return value


def _cursor_signature(cursor: Any) -> str:
    pieces: list[str] = []
    display = _normalize_signature(getattr(cursor, "displayname", ""))
    if display:
        pieces.append(f"display={display}")
    cursor_type = getattr(cursor, "type", None)
    type_spelling = _normalize_signature(getattr(cursor_type, "spelling", ""))
    if type_spelling:
        pieces.append(f"type={type_spelling}")
    result_type = getattr(cursor, "result_type", None)
    result_spelling = _normalize_signature(
        getattr(result_type, "spelling", "")
    )
    if result_spelling:
        pieces.append(f"result={result_spelling}")
    argument_types: list[str] = []
    getter = getattr(cursor, "get_arguments", None)
    if callable(getter):
        try:
            argument_types = [
                _normalize_signature(getattr(argument.type, "spelling", ""))
                for argument in getter()
            ]
        except Exception:
            argument_types = []
    if argument_types:
        pieces.append("args=(" + ",".join(argument_types) + ")")
    return "|".join(pieces)


def _cursor_kind_name(cursor: Any) -> str:
    kind = getattr(cursor, "kind", None)
    name = getattr(kind, "name", None)
    return str(name or str(kind).rsplit(".", 1)[-1] or "symbol")


def _identity_for_cursor(
    indexer: ModuleType,
    cursor: Any,
    *,
    repo_root: Path,
    project_id: str,
    source_path: str,
) -> _CursorIdentity | None:
    helper = getattr(indexer, "symbol_reference_for_cursor", None)
    if callable(helper):
        reference = helper(
            cursor,
            project_dir=str(repo_root),
            project_id=project_id,
            fallback_file=source_path,
        )
        usr = str(reference.get("usr") or "")
        signature = _normalize_signature(
            reference.get("canonical_signature")
        )
        symbol_key = str(reference.get("symbol_key") or "")
        if not usr or not signature or not symbol_key:
            return None
        version = int(
            reference.get("symbol_identity_schema_version") or 0
        )
        if version != SYMBOL_IDENTITY_SCHEMA_VERSION:
            raise ValueError(
                "CASE 4 symbol identity schema mismatch: "
                f"expected={SYMBOL_IDENTITY_SCHEMA_VERSION} actual={version}"
            )
        return _CursorIdentity(
            semantic_identity=symbol_key,
            symbol_key=symbol_key,
            usr=usr,
            canonical_signature=signature,
            qname=str(reference.get("qname") or ""),
            symbol_kind=str(reference.get("symbol_kind") or "symbol"),
            adapter="case4_symbol_reference_for_cursor_v2",
        )

    usr = _cursor_usr(cursor)
    signature = _cursor_signature(cursor)
    if not usr or not signature:
        return None
    symbol_key = (
        f"usr:schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}\x1f"
        f"project={project_id}\x1fusr={usr}"
    )
    return _CursorIdentity(
        semantic_identity=symbol_key,
        symbol_key=symbol_key,
        usr=usr,
        canonical_signature=signature,
        qname=str(indexer.get_qualified_name(cursor) or ""),
        symbol_kind=_cursor_kind_name(cursor),
        adapter="raw_clang_usr_signature_v2_adapter",
    )


def _default_indexer_root() -> Path:
    configured = os.environ.get("CPPMEGA_CLANG_INDEXER_ROOT")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    module_path = Path(__file__).resolve()
    for ancestor in module_path.parents:
        candidates.append(ancestor)
        candidates.extend(
            (
                ancestor / "cppmega.mlx",
                ancestor / "cppmega_mlx_case3_prompt",
            )
        )
    for candidate in candidates:
        path = candidate.resolve() / "tools" / "clang_indexer" / "index_project.py"
        if path.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "clang indexer not found; set CPPMEGA_CLANG_INDEXER_ROOT to a checkout "
        "containing tools/clang_indexer/index_project.py"
    )


def _load_indexer(indexer_root: Path) -> tuple[ModuleType, Path]:
    path = indexer_root / "tools" / "clang_indexer" / "index_project.py"
    if not path.is_file():
        raise FileNotFoundError(f"clang indexer module not found: {path}")
    module_name = "_cppmega_prompt_graph_clang_indexer_" + _sha_file(path)[:12]
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing, path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load clang indexer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module, path


def _cursor_offsets(
    indexer: ModuleType,
    cursor: Any,
    *,
    filename: str,
    byte_to_char: Any,
) -> tuple[int, int] | None:
    return indexer._cursor_extent_offsets(cursor, filename, byte_to_char)


def _name_span(
    cursor: Any,
    source: str,
    extent: tuple[int, int],
    *,
    fallback_name: str = "",
) -> tuple[int, int]:
    start, end = extent
    spelling = str(getattr(cursor, "spelling", "") or fallback_name)
    if spelling:
        relative = source.find(spelling, start, end)
        if relative >= 0:
            return relative, relative + len(spelling)
    return start, end


def _is_definition(cursor: Any) -> bool:
    method = getattr(cursor, "is_definition", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception:
        return False


def _select_chunks(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_end_by_document: dict[int, int] = {}
    for candidate in sorted(
        candidates,
        key=lambda row: (
            row["document_id"],
            row["start"],
            -(row["end"] - row["start"]),
            row["identity"],
        ),
    ):
        last_end = last_end_by_document.get(candidate["document_id"], -1)
        if candidate["start"] < last_end:
            continue
        selected.append(candidate)
        last_end_by_document[candidate["document_id"]] = candidate["end"]
    for chunk_id, row in enumerate(selected):
        row["id"] = chunk_id
    return selected


def _owning_chunk(
    symbol: Mapping[str, Any], chunks: list[dict[str, Any]]
) -> dict[str, Any] | None:
    containing = [
        chunk
        for chunk in chunks
        if chunk["document_id"] == symbol["document_id"]
        and chunk["start"] <= symbol["start"]
        and symbol["end"] <= chunk["end"]
    ]
    if not containing:
        return None
    return min(
        containing,
        key=lambda row: (row["end"] - row["start"], row["id"]),
    )


class ClangPromptProjectIndexProducer:
    """Build/cache a typed repository index at the clang cursor boundary."""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        indexer_root: str | Path | None = None,
        libclang_path: str | Path | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.indexer_root = (
            _default_indexer_root()
            if indexer_root is None
            else Path(indexer_root).resolve()
        )
        self.libclang_path = (
            None if libclang_path is None else str(Path(libclang_path).resolve())
        )

    def build(
        self,
        repo_root: str | Path,
        *,
        project_id: str,
    ) -> PromptProjectIndexBuildResult:
        root = Path(repo_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"prompt graph repository not found: {root}")
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("prompt graph project_id must be non-empty")
        project_id = project_id.strip()
        indexer, indexer_path = _load_indexer(self.indexer_root)
        repository_sha256, repository_manifest = repository_snapshot(root)
        indexer_sha256 = _sha_file(indexer_path)
        cache_key = _sha_json(
            {
                "schema": INDEX_SCHEMA,
                "producer": "ClangPromptProjectIndexProducer",
                "producer_version": PRODUCER_VERSION,
                "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
                "project_id": project_id,
                "repository_sha256": repository_sha256,
                "indexer_sha256": indexer_sha256,
            }
        )
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            return self._load_cached(
                path,
                root=root,
                cache_key=cache_key,
                repository_sha256=repository_sha256,
                indexer_sha256=indexer_sha256,
            )

        index = self._build_uncached(
            indexer,
            root=root,
            project_id=project_id,
            cache_key=cache_key,
            repository_sha256=repository_sha256,
            repository_manifest=repository_manifest,
            indexer_sha256=indexer_sha256,
            indexer_path=indexer_path,
        )
        self._write_cached(path, index)
        return PromptProjectIndexBuildResult(
            index=index,
            path=path,
            cache_hit=False,
            receipt=index.provenance,
        )

    def _build_uncached(
        self,
        indexer: ModuleType,
        *,
        root: Path,
        project_id: str,
        cache_key: str,
        repository_sha256: str,
        repository_manifest: Mapping[str, str],
        indexer_sha256: str,
        indexer_path: Path,
    ) -> PromptProjectIndex:
        indexer._configure_libclang(
            self.libclang_path or os.environ.get("NANOCHAT_LIBCLANG_PATH")
        )
        clang_index = indexer.Index.create()
        files = sorted(
            (Path(path).resolve() for path in indexer.find_cpp_files(str(root))),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        if not files:
            raise ValueError(f"clang prompt graph producer found no sources in {root}")
        compile_db = indexer.load_compile_commands(str(root))
        default_args = indexer.get_default_compile_args(str(root))

        documents: list[dict[str, Any]] = []
        document_by_path: dict[str, dict[str, Any]] = {}
        for document_id, path in enumerate(files, start=1):
            relative = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8", errors="replace")
            document = {
                "id": document_id,
                "source_path": relative,
                "source": source,
            }
            documents.append(document)
            document_by_path[relative] = document

        raw_symbols: list[dict[str, Any]] = []
        chunk_candidates: list[dict[str, Any]] = []
        diagnostics: dict[str, list[str]] = {}
        identity_adapters: set[str] = set()
        function_kinds = set(indexer.FUNCTION_KINDS)
        type_kinds = set(indexer._TYPE_DEF_KIND_BUCKET)
        variable_kinds = {
            kind
            for name in ("VAR_DECL", "FIELD_DECL", "PARM_DECL")
            if (kind := getattr(indexer.CursorKind, name, None)) is not None
        }
        call_kinds = set(indexer._CALL_KINDS)
        type_ref_kinds = set(indexer._TYPE_REF_KINDS)
        reference_kinds = set(indexer._REFERENCE_KINDS)

        for path in files:
            relative = path.relative_to(root).as_posix()
            document = document_by_path[relative]
            source = str(document["source"])
            compile_args = indexer._resolve_file_args(
                str(path), compile_db, default_args
            )
            try:
                translation_unit = clang_index.parse(
                    str(path),
                    args=compile_args,
                    options=indexer.TranslationUnit.PARSE_INCOMPLETE,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"clang prompt graph parse failed for {path}: {exc}"
                ) from exc
            diagnostics[relative] = [
                str(diagnostic) for diagnostic in translation_unit.diagnostics
            ]
            byte_to_char = indexer._byte_to_char_mapper(source)
            stack: list[tuple[Any, int]] = [(translation_unit.cursor, 0)]
            while stack:
                cursor, depth = stack.pop()
                children = list(cursor.get_children())
                stack.extend((child, depth + 1) for child in reversed(children))
                extent = _cursor_offsets(
                    indexer,
                    cursor,
                    filename=str(path),
                    byte_to_char=byte_to_char,
                )
                if extent is None:
                    continue
                kind = cursor.kind
                definition_kind: str | None = None
                chunk_kind: int | None = None
                if kind in function_kinds and _is_definition(cursor):
                    definition_kind = "function"
                    chunk_kind = 1
                elif kind in type_kinds and _is_definition(cursor):
                    definition_kind = "type"
                    chunk_kind = int(indexer._TYPE_DEF_KIND_BUCKET[kind])
                elif kind in variable_kinds:
                    definition_kind = "variable"

                if definition_kind is not None:
                    identity = _identity_for_cursor(
                        indexer,
                        cursor,
                        repo_root=root,
                        project_id=project_id,
                        source_path=relative,
                    )
                    if identity is not None:
                        identity_adapters.add(identity.adapter)
                        symbol_start, symbol_end = _name_span(
                            cursor, source, extent
                        )
                        raw_symbols.append(
                            {
                                "kind": definition_kind,
                                "document_id": document["id"],
                                "source_path": relative,
                                "start": symbol_start,
                                "end": symbol_end,
                                "semantic_identity": identity.semantic_identity,
                                "symbol_key": identity.symbol_key,
                                "usr": identity.usr,
                                "canonical_signature": identity.canonical_signature,
                                "qname": identity.qname,
                                "target_semantic_identity": None,
                                "relation": None,
                            }
                        )
                        if chunk_kind is not None:
                            chunk_candidates.append(
                                {
                                    "identity": (
                                        f"chunk:{document['id']}:{extent[0]}:"
                                        f"{extent[1]}:{identity.semantic_identity}"
                                    ),
                                    "document_id": document["id"],
                                    "source_path": relative,
                                    "start": extent[0],
                                    "end": extent[1],
                                    "kind": chunk_kind,
                                    "dep_level": max(0, depth - 1),
                                }
                            )

                relation: str | None = None
                if kind in call_kinds:
                    relation = "call"
                elif kind in type_ref_kinds:
                    relation = "type"
                elif kind in reference_kinds:
                    relation = "def_use"
                if relation is None:
                    continue
                referenced = getattr(cursor, "referenced", None)
                if referenced is None:
                    continue
                target = _identity_for_cursor(
                    indexer,
                    referenced,
                    repo_root=root,
                    project_id=project_id,
                    source_path=relative,
                )
                if target is None:
                    continue
                identity_adapters.add(target.adapter)
                start, end = _name_span(
                    cursor,
                    source,
                    extent,
                    fallback_name=str(getattr(referenced, "spelling", "") or ""),
                )
                raw_symbols.append(
                    {
                        "kind": f"{relation}site" if relation != "def_use" else "use",
                        "document_id": document["id"],
                        "source_path": relative,
                        "start": start,
                        "end": end,
                        "semantic_identity": target.semantic_identity,
                        "symbol_key": target.symbol_key,
                        "usr": target.usr,
                        "canonical_signature": target.canonical_signature,
                        "qname": target.qname,
                        "target_semantic_identity": target.semantic_identity,
                        "relation": relation,
                    }
                )

        chunks = _select_chunks(chunk_candidates)
        ordered_symbols = sorted(
            raw_symbols,
            key=lambda row: (
                row["document_id"],
                row["start"],
                row["end"],
                row["kind"],
                row["semantic_identity"],
            ),
        )
        definitions_by_semantic: dict[str, list[dict[str, Any]]] = {}
        symbols: list[dict[str, Any]] = []
        for symbol_id, row in enumerate(ordered_symbols, start=1):
            owner = _owning_chunk(row, chunks)
            local_identity = (
                f"symbol:{row['document_id']}:{row['start']}:{row['end']}:"
                f"{row['kind']}:{sha256(row['semantic_identity'].encode()).hexdigest()[:16]}"
            )
            symbol = {
                "id": symbol_id,
                "identity": local_identity,
                "semantic_identity": row["semantic_identity"],
                "symbol_key": row["symbol_key"],
                "usr": row["usr"],
                "canonical_signature": row["canonical_signature"],
                "qname": row["qname"],
                "kind": row["kind"],
                "document_id": row["document_id"],
                "source_path": row["source_path"],
                "start": row["start"],
                "end": row["end"],
                "chunk_identity": "" if owner is None else owner["identity"],
                "target_semantic_identity": row["target_semantic_identity"],
                "relation": row["relation"],
            }
            symbols.append(symbol)
            if row["relation"] is None:
                definitions_by_semantic.setdefault(
                    row["semantic_identity"], []
                ).append(symbol)

        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str, int | None]] = set()
        domain_kinds = {"call": 2, "type": 3, "def_use": 4}
        for source in symbols:
            relation = source.pop("relation")
            target_semantic = source.pop("target_semantic_identity")
            if relation is None or target_semantic is None:
                continue
            targets = definitions_by_semantic.get(target_semantic, [])
            if not targets:
                continue
            target = min(
                targets,
                key=lambda row: (
                    row["source_path"],
                    row["start"],
                    row["identity"],
                ),
            )
            if not source["chunk_identity"] or not target["chunk_identity"]:
                continue
            relation_key = (
                relation,
                source["identity"],
                target["identity"],
                None,
            )
            if relation_key not in seen_edges:
                edges.append(
                    {
                        "relation": relation,
                        "source": source["identity"],
                        "target": target["identity"],
                    }
                )
                seen_edges.add(relation_key)
            domain_key = (
                "domain",
                source["identity"],
                target["identity"],
                domain_kinds[relation],
            )
            if domain_key not in seen_edges:
                edges.append(
                    {
                        "relation": "domain",
                        "source": source["identity"],
                        "target": target["identity"],
                        "kind": domain_kinds[relation],
                    }
                )
                seen_edges.add(domain_key)

        for symbol in symbols:
            symbol.pop("relation", None)
            symbol.pop("target_semantic_identity", None)
        edge_counts = {
            relation: sum(1 for edge in edges if edge["relation"] == relation)
            for relation in (
                "call",
                "type",
                "def_use",
                "domain",
                "build",
                "shell",
                "diagnostic",
                "cross_domain",
            )
        }
        if not symbols or not chunks or sum(edge_counts.values()) == 0:
            raise ValueError(
                "clang prompt graph producer emitted unavailable graph data: "
                f"symbols={len(symbols)} chunks={len(chunks)} edges={sum(edge_counts.values())}"
            )
        definitions = [
            symbol
            for symbol in symbols
            if symbol["kind"] in {"function", "type", "variable"}
        ]
        if any(
            not symbol["usr"] or not symbol["canonical_signature"]
            for symbol in definitions
        ):
            raise ValueError(
                "clang prompt graph producer rejects qname-only definitions; "
                "CASE 4 USR and canonical signature are required"
            )

        provenance = {
            "producer": "ClangPromptProjectIndexProducer",
            "producer_version": PRODUCER_VERSION,
            "schema": INDEX_SCHEMA,
            "cache_key": cache_key,
            "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
            "identity_adapters": sorted(identity_adapters),
            "hashes": {
                "repository_sha256": repository_sha256,
                "indexer_sha256": indexer_sha256,
            },
            "repository_manifest": dict(sorted(repository_manifest.items())),
            "indexer_path": str(indexer_path),
            "document_count": len(documents),
            "symbol_count": len(symbols),
            "chunk_count": len(chunks),
            "edge_counts": edge_counts,
            "diagnostics": diagnostics,
        }
        return PromptProjectIndex.from_dict(
            {
                "schema": INDEX_SCHEMA,
                "project_id": project_id,
                "documents": documents,
                "symbols": symbols,
                "chunks": chunks,
                "edges": edges,
                "provenance": provenance,
            }
        )

    @staticmethod
    def _write_cached(path: Path, index: PromptProjectIndex) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(_stable_json(index.to_dict()) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _load_cached(
        path: Path,
        *,
        root: Path,
        cache_key: str,
        repository_sha256: str,
        indexer_sha256: str,
    ) -> PromptProjectIndexBuildResult:
        try:
            index = PromptProjectIndex.from_json_path(path)
            receipt = index.provenance
            if receipt.get("producer") != "ClangPromptProjectIndexProducer":
                raise ValueError("producer mismatch")
            if receipt.get("cache_key") != cache_key:
                raise ValueError("cache key mismatch")
            if int(receipt.get("symbol_identity_schema_version") or 0) != 2:
                raise ValueError("CASE 4 symbol identity schema mismatch")
            hashes = dict(receipt.get("hashes") or {})
            if hashes.get("repository_sha256") != repository_sha256:
                raise ValueError("repository hash mismatch")
            if hashes.get("indexer_sha256") != indexer_sha256:
                raise ValueError("indexer hash mismatch")
            index.verify_repository(root)
        except Exception as exc:
            raise ValueError(
                f"cached prompt project index is invalid: {path}: {exc}"
            ) from exc
        return PromptProjectIndexBuildResult(
            index=index,
            path=path,
            cache_hit=True,
            receipt=index.provenance,
        )


__all__ = [
    "ClangPromptProjectIndexProducer",
    "PRODUCER_VERSION",
    "PromptProjectIndexBuildResult",
    "SYMBOL_IDENTITY_SCHEMA_VERSION",
]
