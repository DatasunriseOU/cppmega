"""Production clang adapter for repository prompt-graph indexes.

The adapter reuses the repository clang indexer's discovery, compile-command,
libclang, and cursor-offset seams. Semantic identity delegates to CASE 4's v3
``symbol_reference_for_cursor`` contract and preserves its canonical uint64 ID.
The compatibility seam transports clang USR plus canonical signature directly
and never derives identity from a qualified name.
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

from .prompt_graph import (
    INDEX_SCHEMA,
    PromptProjectIndex,
    repository_snapshot,
    require_prompt_graph_project_id,
)
from cppmega.prompt_graph_provenance import (
    INDEX_INTEGRITY_VERSION,
    INDEXER_DEPENDENCY_HASH_KEY,
    INDEXER_DEPENDENCY_MANIFEST_KEY,
    INDEXER_DEPENDENCY_POLICY,
    PRODUCTION_IDENTITY_PROVENANCE_CONTRACT,
    indexer_dependency_hash,
)
from .symbol_identity import (
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    canonical_external_provider_file,
    canonical_external_usr_identity,
    compute_symbol_id,
    external_provider_project,
    is_repo_file_location_identity,
)


PRODUCER_VERSION = "3"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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
    symbol_id: int
    usr: str
    canonical_signature: str
    qname: str
    symbol_kind: str
    identity_project: str
    identity_file: str
    identity_line: int
    identity_column: int
    identity_provider: str
    identity_include_provenance: str
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


def _prompt_graph_cache_key(
    *,
    project_id: str,
    strict_diagnostics: bool,
    fingerprint_hashes: Mapping[str, str],
    libclang_version: str,
    libclang_path: str | None,
) -> str:
    return _sha_json(
        {
            "schema": INDEX_SCHEMA,
            "producer": "ClangPromptProjectIndexProducer",
            "producer_version": PRODUCER_VERSION,
            "index_integrity_version": INDEX_INTEGRITY_VERSION,
            "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
            "identity_provenance_contract": PRODUCTION_IDENTITY_PROVENANCE_CONTRACT,
            "project_id": project_id,
            "strict_diagnostics": strict_diagnostics,
            "indexer_dependency_policy": INDEXER_DEPENDENCY_POLICY,
            "hashes": dict(fingerprint_hashes),
            "libclang_version": libclang_version,
            "libclang_path": libclang_path,
        }
    )


def _normalize_signature(value: object) -> str:
    return " ".join(str(value or "").split())


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
) -> _CursorIdentity:
    helper = getattr(indexer, "symbol_reference_for_cursor", None)
    if not callable(helper):
        raise RuntimeError(
            "native CASE 4 v3 symbol_reference_for_cursor is required; "
            "raw clang identity fallback is forbidden"
        )
    reference = helper(
        cursor,
        project_dir=str(repo_root),
        project_id=project_id,
        fallback_file=source_path,
    )
    if not isinstance(reference, Mapping):
        raise TypeError(
            "CASE 4 v3 symbol_reference_for_cursor must return a mapping"
        )
    provenance_fields = (
        "project",
        "file",
        "line",
        "column",
        "provider",
        "include_provenance",
    )
    if any(field not in reference for field in provenance_fields):
        raise ValueError(
            "CASE 4 v3 symbol reference provenance is incomplete; "
            "project, file, line, and column are required"
        )
    reference_project = reference.get("project")
    if not isinstance(reference_project, str):
        raise ValueError("CASE 4 v3 symbol reference provenance project is invalid")
    reference_provider = reference.get("provider")
    reference_include = reference.get("include_provenance")
    if not isinstance(reference_provider, str):
        raise ValueError(
            "CASE 4 v3 symbol reference provenance provider is invalid"
        )
    if not isinstance(reference_include, str):
        raise ValueError(
            "CASE 4 v3 symbol reference include provenance is invalid"
        )
    reference_file = reference.get("file")
    if not isinstance(reference_file, str) or not reference_file:
        raise ValueError("CASE 4 v3 symbol reference provenance file is invalid")
    external_provider = reference_project != project_id
    if external_provider:
        try:
            expected_project = external_provider_project(
                reference_provider,
                source="CASE 4 v3 external provider reference",
            )
            expected_file = canonical_external_provider_file(
                reference_provider,
                reference_include,
                source="CASE 4 v3 external provider reference",
            )
        except Exception as exc:
            raise ValueError(
                "CASE 4 v3 symbol reference provenance project does not match "
                "the repository project or a trusted external provider"
            ) from exc
        if reference_project != expected_project or reference_file != expected_file:
            raise ValueError(
                "CASE 4 v3 external provider reference provenance is inconsistent"
            )
    line = reference.get("line")
    if isinstance(line, bool) or not isinstance(line, int) or line <= 0:
        raise ValueError("CASE 4 v3 symbol reference provenance line is invalid")
    column = reference.get("column")
    if isinstance(column, bool) or not isinstance(column, int) or column <= 0:
        raise ValueError(
            "CASE 4 v3 symbol reference provenance column is invalid"
        )
    location = getattr(cursor, "location", None)
    location_file = getattr(location, "file", None)
    location_name = getattr(location_file, "name", None)
    if not isinstance(location_name, (str, os.PathLike)) or not str(location_name):
        raise ValueError("CASE 4 v3 cursor file is missing")
    location_text = str(location_name)
    provider_resolver = getattr(indexer, "symbol_provider_provenance", None)
    if external_provider and location_text.startswith("@provider/"):
        if not callable(provider_resolver):
            raise ValueError("CASE 4 v3 external provider path resolver is missing")
        actual_provider, actual_include = provider_resolver(location_text)
    else:
        try:
            resolved_root = repo_root.resolve()
            resolved_file = Path(location_name).resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError("CASE 4 v3 cursor file could not be resolved") from exc
        try:
            relative_file = resolved_file.relative_to(resolved_root).as_posix()
        except ValueError:
            if not external_provider:
                raise ValueError("CASE 4 v3 cursor file is outside repository root")
            if not callable(provider_resolver):
                raise ValueError("CASE 4 v3 external provider path resolver is missing")
            actual_provider, actual_include = provider_resolver(location_text)
        else:
            if external_provider:
                raise ValueError(
                    "CASE 4 v3 external provider cursor resolves inside the repository"
                )
            if reference_file != relative_file:
                raise ValueError(
                    "CASE 4 v3 symbol reference provenance file does not match "
                    "the cursor's repository file"
                )
    if external_provider and (
        actual_provider != reference_provider
        or actual_include != reference_include
    ):
        raise ValueError(
            "CASE 4 v3 external provider cursor provenance does not match its path"
        )

    cursor_line = getattr(location, "line", None)
    if (
        isinstance(cursor_line, bool)
        or not isinstance(cursor_line, int)
        or cursor_line <= 0
    ):
        raise ValueError("CASE 4 v3 cursor line is invalid")
    if line != cursor_line:
        raise ValueError(
            "CASE 4 v3 symbol reference provenance line does not match cursor"
        )
    cursor_column = getattr(location, "column", None)
    if (
        isinstance(cursor_column, bool)
        or not isinstance(cursor_column, int)
        or cursor_column <= 0
    ):
        raise ValueError("CASE 4 v3 cursor column is invalid")
    if column != cursor_column:
        raise ValueError(
            "CASE 4 v3 symbol reference provenance column does not match cursor"
        )

    usr = str(reference.get("usr") or "")
    signature = _normalize_signature(reference.get("canonical_signature"))
    symbol_key = str(reference.get("symbol_key") or "")
    if not symbol_key:
        raise ValueError("CASE 4 v3 symbol reference requires a symbol_key")
    if not signature and not is_repo_file_location_identity(symbol_key):
        raise ValueError(
            "CASE 4 v3 symbol reference requires USR/signature or explicit "
            "repository-location identity"
        )
    if external_provider and usr:
        try:
            expected_symbol_key = canonical_external_usr_identity(
                usr=usr,
                canonical_signature=signature,
                provider=reference_provider,
                include_provenance=reference_include,
                project=reference_project,
                source="CASE 4 v3 external provider reference",
            )
        except Exception as exc:
            raise ValueError(
                "CASE 4 v3 external provider symbol key is invalid"
            ) from exc
        if symbol_key != expected_symbol_key:
            raise ValueError(
                "CASE 4 v3 external provider symbol key is inconsistent"
            )
    version = int(reference.get("symbol_identity_schema_version") or 0)
    if version != SYMBOL_IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            "CASE 4 symbol identity schema mismatch: "
            f"expected={SYMBOL_IDENTITY_SCHEMA_VERSION} actual={version}"
        )
    claimed_symbol_id = reference.get("symbol_id")
    if isinstance(claimed_symbol_id, bool) or not isinstance(
        claimed_symbol_id, int
    ):
        raise ValueError(
            "CASE 4 v3 symbol reference requires an integer symbol_id"
        )
    expected_symbol_id = compute_symbol_id(symbol_key)
    if claimed_symbol_id != expected_symbol_id:
        raise ValueError(
            "CASE 4 symbol ID does not match canonical key: "
            f"claimed={claimed_symbol_id} expected={expected_symbol_id}"
        )
    return _CursorIdentity(
        semantic_identity=symbol_key,
        symbol_key=symbol_key,
        symbol_id=claimed_symbol_id,
        usr=usr,
        canonical_signature=signature,
        qname=str(reference.get("qname") or ""),
        symbol_kind=str(reference.get("symbol_kind") or "symbol"),
        identity_project=reference_project,
        identity_file=reference_file,
        identity_line=line,
        identity_column=column,
        identity_provider=str(reference["provider"]),
        identity_include_provenance=str(reference["include_provenance"]),
        adapter="case4_symbol_reference_for_cursor_v3",
    )


def _load_indexer(indexer_root: Path) -> tuple[ModuleType, Path]:
    indexer_root = indexer_root.expanduser().resolve()
    if indexer_root != _REPOSITORY_ROOT:
        raise ValueError(
            "cppmega prompt-graph indexer must come from the same checkout as "
            "the imported cppmega package; "
            f"package_root={_REPOSITORY_ROOT}, indexer_root={indexer_root}. "
            "Cross-checkout cppmega/cppmega.mlx indexer mixing is unsupported."
        )
    path = indexer_root / "tools" / "clang_indexer" / "index_project.py"
    if not path.is_file():
        raise FileNotFoundError(f"clang indexer module not found: {path}")
    module_name = "_cppmega_prompt_graph_clang_indexer_" + _sha_file(path)[:12]
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        existing_spec = getattr(existing, "__spec__", None)
        existing_origin = getattr(existing_spec, "origin", None)
        origins = (existing_file, existing_origin)
        if any(
            not isinstance(origin, (str, os.PathLike))
            or Path(origin).expanduser().resolve(strict=False) != path
            for origin in origins
        ):
            raise ValueError(
                "cached clang indexer provenance does not match the requested "
                f"module: requested={path}, file={existing_file!r}, "
                f"spec_origin={existing_origin!r}"
            )
        return existing, path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load clang indexer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    indexer_root_text = str(indexer_root)
    original_sys_path = sys.path
    original_entries = list(original_sys_path)
    sys.path = [entry for entry in sys.path if entry != indexer_root_text]
    sys.path.insert(0, indexer_root_text)
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        original_sys_path[:] = original_entries
        sys.path = original_sys_path
    return module, path


def _libclang_identity(
    indexer: ModuleType,
    configured_path: str | None,
) -> tuple[str, str | None]:
    runtime = getattr(indexer, "clang_cindex_runtime", None)
    conf = getattr(runtime, "conf", None)
    cx_string = getattr(runtime, "_CXString", None)
    if conf is None or cx_string is None:
        raise RuntimeError(
            "clang prompt graph producer cannot inspect the configured "
            "libclang runtime version"
        )
    function = conf.lib.clang_getClangVersion
    function.restype = cx_string
    raw_version = function()
    version = cx_string.from_result(raw_version)
    if isinstance(version, bytes):
        version = version.decode("utf-8", errors="replace")
    version = str(version).strip()
    if not version:
        raise RuntimeError("configured libclang returned an empty version")

    config = getattr(runtime, "Config", None)
    library_file = getattr(config, "library_file", None)
    library_path = getattr(config, "library_path", None)
    resolved = configured_path or library_file or library_path
    return version, None if resolved is None else str(resolved)


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
    filename: str,
    byte_to_char: Any,
    fallback_name: str = "",
) -> tuple[int, int]:
    start, end = extent
    spellings: list[str] = []
    for raw in (getattr(cursor, "spelling", ""), fallback_name):
        spelling = str(raw or "")
        if spelling and spelling not in spellings:
            spellings.append(spelling)
        unqualified = spelling.rsplit("::", 1)[-1]
        if unqualified and unqualified not in spellings:
            spellings.append(unqualified)

    location = getattr(cursor, "location", None)
    location_file = getattr(getattr(location, "file", None), "name", None)
    location_offset: int | None = None
    if location_file is not None and os.path.normcase(
        os.path.normpath(str(location_file))
    ) == os.path.normcase(os.path.normpath(filename)):
        location_offset = byte_to_char(int(location.offset))

    if location_offset is not None:
        for spelling in spellings:
            candidate_end = location_offset + len(spelling)
            if (
                start <= location_offset < candidate_end <= end
                and source[location_offset:candidate_end] == spelling
            ):
                return location_offset, candidate_end

    token_candidates: list[tuple[int, int]] = []
    location_token_candidates: list[tuple[int, int]] = []
    getter = getattr(cursor, "get_tokens", None)
    if callable(getter):
        for token in getter():
            token_spelling = str(getattr(token, "spelling", "") or "")
            token_extent = getattr(token, "extent", None)
            token_start = getattr(token_extent, "start", None)
            token_end = getattr(token_extent, "end", None)
            start_file = getattr(getattr(token_start, "file", None), "name", None)
            end_file = getattr(getattr(token_end, "file", None), "name", None)
            if start_file is None or end_file is None:
                continue
            normalized = os.path.normcase(os.path.normpath(filename))
            if (
                os.path.normcase(os.path.normpath(str(start_file))) != normalized
                or os.path.normcase(os.path.normpath(str(end_file))) != normalized
            ):
                continue
            token_span = (
                byte_to_char(int(token_start.offset)),
                byte_to_char(int(token_end.offset)),
            )
            if (
                start <= token_span[0] < token_span[1] <= end
                and source[token_span[0] : token_span[1]] == token_spelling
            ):
                if (
                    location_offset is not None
                    and token_span[0] <= location_offset < token_span[1]
                ):
                    location_token_candidates.append(token_span)
                if token_spelling in spellings:
                    token_candidates.append(token_span)
    if location_token_candidates:
        return min(location_token_candidates)
    if token_candidates:
        if location_offset is not None:
            return min(
                token_candidates,
                key=lambda span: (
                    0 if span[0] <= location_offset < span[1] else 1,
                    abs(span[0] - location_offset),
                    span,
                ),
            )
        if len(token_candidates) == 1:
            return token_candidates[0]
    raise ValueError(
        "clang prompt graph could not select an exact identifier span: "
        f"kind={_cursor_kind_name(cursor)!r} spellings={spellings!r} "
        f"extent=[{start},{end}) location={location_offset}"
    )


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
        strict_diagnostics: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        if indexer_root is None:
            raise ValueError(
                "ClangPromptProjectIndexProducer requires explicit indexer_root "
                "pointing to tools/clang_indexer/index_project.py"
            )
        self.indexer_root = Path(indexer_root).resolve()
        self.libclang_path = (
            None if libclang_path is None else str(Path(libclang_path).resolve())
        )
        self.strict_diagnostics = bool(strict_diagnostics)

    def build(
        self,
        repo_root: str | Path,
        *,
        project_id: str,
    ) -> PromptProjectIndexBuildResult:
        root = Path(repo_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"prompt graph repository not found: {root}")
        project_id = require_prompt_graph_project_id(
            project_id, where="prompt graph project_id"
        )
        indexer, indexer_path = _load_indexer(self.indexer_root)
        configured_libclang = indexer._configure_libclang(
            self.libclang_path or os.environ.get("NANOCHAT_LIBCLANG_PATH")
        )
        libclang_version, resolved_libclang = _libclang_identity(
            indexer,
            configured_libclang,
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
        compile_args_by_file = {
            path.relative_to(root).as_posix(): list(
                indexer._resolve_file_args(str(path), compile_db, default_args)
            )
            for path in files
        }
        repository_sha256, repository_manifest = repository_snapshot(root)
        dependency_manifest = {
            path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in files
        }
        indexer_sha256 = _sha_file(indexer_path)
        indexer_dependency_manifest, indexer_dependency_sha256 = (
            indexer_dependency_hash(indexer_path, self.indexer_root)
        )
        fingerprint_hashes = {
            "repository_sha256": repository_sha256,
            "dependency_closure_sha256": _sha_json(dependency_manifest),
            "compile_args_sha256": _sha_json(compile_args_by_file),
            "indexer_sha256": indexer_sha256,
            INDEXER_DEPENDENCY_HASH_KEY: indexer_dependency_sha256,
            "libclang_version_sha256": sha256(
                libclang_version.encode("utf-8")
            ).hexdigest(),
        }
        cache_key = _prompt_graph_cache_key(
            project_id=project_id,
            strict_diagnostics=self.strict_diagnostics,
            fingerprint_hashes=fingerprint_hashes,
            libclang_version=libclang_version,
            libclang_path=resolved_libclang,
        )
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            return self._load_cached(
                path,
                root=root,
                cache_key=cache_key,
                expected_hashes=fingerprint_hashes,
                project_id=project_id,
                expected_indexer_root=self.indexer_root,
                libclang_version=libclang_version,
                resolved_libclang=resolved_libclang,
            )

        index = self._build_uncached(
            indexer,
            clang_index=clang_index,
            files=files,
            compile_args_by_file=compile_args_by_file,
            root=root,
            project_id=project_id,
            cache_key=cache_key,
            fingerprint_hashes=fingerprint_hashes,
            repository_manifest=repository_manifest,
            dependency_manifest=dependency_manifest,
            indexer_dependency_manifest=indexer_dependency_manifest,
            indexer_path=indexer_path,
            libclang_version=libclang_version,
            resolved_libclang=resolved_libclang,
        )
        index = index.with_integrity()
        index.validate_production_repository_index(
            expected_project_id=project_id,
            repository_root=root,
            expected_indexer_root=self.indexer_root,
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
        clang_index: Any,
        files: list[Path],
        compile_args_by_file: Mapping[str, list[str]],
        root: Path,
        project_id: str,
        cache_key: str,
        fingerprint_hashes: Mapping[str, str],
        repository_manifest: Mapping[str, str],
        dependency_manifest: Mapping[str, str],
        indexer_dependency_manifest: Mapping[str, str],
        indexer_path: Path,
        libclang_version: str,
        resolved_libclang: str | None,
    ) -> PromptProjectIndex:
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
        external_references: dict[tuple[Any, ...], dict[str, Any]] = {}
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
            compile_args = list(compile_args_by_file[relative])
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
            diagnostic_rows = [
                {
                    "severity": int(getattr(diagnostic, "severity", 3)),
                    "text": str(diagnostic),
                }
                for diagnostic in translation_unit.diagnostics
            ]
            diagnostics[relative] = [row["text"] for row in diagnostic_rows]
            parse_errors = [
                row for row in diagnostic_rows if int(row["severity"]) >= 3
            ]
            if self.strict_diagnostics and parse_errors:
                rendered = "; ".join(row["text"] for row in parse_errors[:5])
                raise RuntimeError(
                    f"strict clang prompt graph parse rejected {path}: {rendered}"
                )
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
                            cursor,
                            source,
                            extent,
                            filename=str(path),
                            byte_to_char=byte_to_char,
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
                                "symbol_id": identity.symbol_id,
                                "usr": identity.usr,
                                "canonical_signature": identity.canonical_signature,
                                "qname": identity.qname,
                                "identity_project": identity.identity_project,
                                "identity_file": identity.identity_file,
                                "identity_line": identity.identity_line,
                                "identity_column": identity.identity_column,
                                "identity_kind": identity.symbol_kind,
                                "identity_provider": identity.identity_provider,
                                "identity_include_provenance": (
                                    identity.identity_include_provenance
                                ),
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
                    filename=str(path),
                    byte_to_char=byte_to_char,
                    fallback_name=str(getattr(referenced, "spelling", "") or ""),
                )
                if target.identity_project != project_id:
                    external_reference = {
                        "relation": relation,
                        "document_id": int(document["id"]),
                        "source_path": relative,
                        "start": start,
                        "end": end,
                        "symbol_key": target.symbol_key,
                        "symbol_id": target.symbol_id,
                        "usr": target.usr,
                        "canonical_signature": target.canonical_signature,
                        "qname": target.qname,
                        "symbol_kind": target.symbol_kind,
                        "project": target.identity_project,
                        "file": target.identity_file,
                        "line": target.identity_line,
                        "column": target.identity_column,
                        "provider": target.identity_provider,
                        "include_provenance": target.identity_include_provenance,
                    }
                    external_key = (
                        relation,
                        int(document["id"]),
                        relative,
                        start,
                        end,
                        target.symbol_key,
                    )
                    external_references.setdefault(external_key, external_reference)
                    continue
                raw_symbols.append(
                    {
                        "kind": f"{relation}site" if relation != "def_use" else "use",
                        "document_id": document["id"],
                        "source_path": relative,
                        "start": start,
                        "end": end,
                        "semantic_identity": target.semantic_identity,
                        "symbol_key": target.symbol_key,
                        "symbol_id": target.symbol_id,
                        "usr": target.usr,
                        "canonical_signature": target.canonical_signature,
                        "qname": target.qname,
                        "identity_project": target.identity_project,
                        "identity_file": target.identity_file,
                        "identity_line": target.identity_line,
                        "identity_column": target.identity_column,
                        "identity_kind": target.symbol_kind,
                        "identity_provider": target.identity_provider,
                        "identity_include_provenance": (
                            target.identity_include_provenance
                        ),
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
        definition_provenance: dict[
            str, tuple[str, str, int, int, str, str, str]
        ] = {}
        for row in ordered_symbols:
            if row["relation"] is not None:
                continue
            contract = (
                row["identity_project"],
                row["identity_file"],
                row["identity_line"],
                row["identity_column"],
                row["identity_kind"],
                row["identity_provider"],
                row["identity_include_provenance"],
            )
            previous = definition_provenance.setdefault(
                row["semantic_identity"], contract
            )
            if previous != contract:
                raise ValueError(
                    "clang prompt graph producer found conflicting definition "
                    "provenance for semantic identity "
                    f"{row['semantic_identity']!r}"
                )
        provenance_fields = (
            "identity_project",
            "identity_file",
            "identity_line",
            "identity_column",
            "identity_kind",
            "identity_provider",
            "identity_include_provenance",
        )
        for row in ordered_symbols:
            if row["relation"] is None:
                continue
            contract = definition_provenance.get(row["semantic_identity"])
            if contract is None:
                continue
            for field, value in zip(provenance_fields, contract, strict=True):
                row[field] = value
        definitions_by_semantic: dict[str, list[dict[str, Any]]] = {}
        symbols: list[dict[str, Any]] = []
        for node_id, row in enumerate(ordered_symbols, start=1):
            owner = _owning_chunk(row, chunks)
            local_identity = (
                f"symbol:{row['document_id']}:{row['start']}:{row['end']}:"
                f"{row['kind']}:{sha256(row['semantic_identity'].encode()).hexdigest()[:16]}"
            )
            symbol = {
                "id": node_id,
                "symbol_id": row["symbol_id"],
                "identity": local_identity,
                "semantic_identity": row["semantic_identity"],
                "symbol_key": row["symbol_key"],
                "usr": row["usr"],
                "canonical_signature": row["canonical_signature"],
                "qname": row["qname"],
                "identity_project": row["identity_project"],
                "identity_file": row["identity_file"],
                "identity_line": row["identity_line"],
                "identity_column": row["identity_column"],
                "identity_kind": row["identity_kind"],
                "identity_provider": row["identity_provider"],
                "identity_include_provenance": row[
                    "identity_include_provenance"
                ],
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
        if not symbols or not chunks or (
            sum(edge_counts.values()) == 0 and not external_references
        ):
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
            not symbol["symbol_key"]
            or (
                not symbol["canonical_signature"]
                and not is_repo_file_location_identity(symbol["symbol_key"])
            )
            for symbol in definitions
        ):
            raise ValueError(
                "clang prompt graph producer rejects qname-only definitions; "
                "CASE 4 v3 symbol key plus a canonical signature or explicit "
                "repo_file_location identity are required"
            )

        provenance = {
            "producer": "ClangPromptProjectIndexProducer",
            "producer_version": PRODUCER_VERSION,
            "index_integrity_version": INDEX_INTEGRITY_VERSION,
            "schema": INDEX_SCHEMA,
            "project_id": project_id,
            "cache_key": cache_key,
            "strict_diagnostics": self.strict_diagnostics,
            "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
            "identity_provenance_contract": (
                PRODUCTION_IDENTITY_PROVENANCE_CONTRACT
            ),
            "identity_adapters": sorted(identity_adapters),
            "hashes": dict(fingerprint_hashes),
            "toolchain": {
                "libclang_version": libclang_version,
                "libclang_path": resolved_libclang,
                "compile_args_by_file": {
                    path: list(args)
                    for path, args in sorted(compile_args_by_file.items())
                },
            },
            "repository_manifest": dict(sorted(repository_manifest.items())),
            "dependency_closure_policy": "all_indexed_repository_sources_v1",
            "dependency_manifest": dict(sorted(dependency_manifest.items())),
            "indexer_path": str(indexer_path),
            "indexer_checkout_root": str(self.indexer_root),
            "indexer_dependency_policy": INDEXER_DEPENDENCY_POLICY,
            INDEXER_DEPENDENCY_MANIFEST_KEY: dict(
                sorted(indexer_dependency_manifest.items())
            ),
            "external_references": sorted(
                external_references.values(),
                key=lambda row: (
                    row["document_id"],
                    row["start"],
                    row["end"],
                    row["relation"],
                    row["symbol_key"],
                ),
            ),
            "external_reference_count": len(external_references),
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
        expected_hashes: Mapping[str, str],
        project_id: str,
        expected_indexer_root: Path,
        libclang_version: str,
        resolved_libclang: str | None,
    ) -> PromptProjectIndexBuildResult:
        try:
            index = PromptProjectIndex.from_json_path(path)
            index.validate_production_repository_index(
                expected_project_id=project_id,
                repository_root=root,
                expected_indexer_root=expected_indexer_root,
            )
            receipt = index.provenance
            if receipt.get("producer") != "ClangPromptProjectIndexProducer":
                raise ValueError("producer mismatch")
            if receipt.get("cache_key") != cache_key:
                raise ValueError("cache key mismatch")
            if receipt.get("index_integrity_version") != INDEX_INTEGRITY_VERSION:
                raise ValueError("index integrity version mismatch")
            if (
                int(receipt.get("symbol_identity_schema_version") or 0)
                != SYMBOL_IDENTITY_SCHEMA_VERSION
            ):
                raise ValueError("CASE 4 symbol identity schema mismatch")
            hashes = dict(receipt.get("hashes") or {})
            for name, expected in expected_hashes.items():
                if hashes.get(name) != expected:
                    raise ValueError(f"{name} mismatch")
            toolchain = dict(receipt.get("toolchain") or {})
            if toolchain.get("libclang_version") != libclang_version:
                raise ValueError("libclang version mismatch")
            if toolchain.get("libclang_path") != resolved_libclang:
                raise ValueError("libclang path mismatch")
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
    "PRODUCTION_IDENTITY_PROVENANCE_CONTRACT",
    "PromptProjectIndexBuildResult",
    "SYMBOL_IDENTITY_SCHEMA_VERSION",
]
