from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any
import zlib

import pyarrow.parquet as pq
import pytest

from cppmega.data.domain_schema import (
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
    delimiter_token_ids,
)
from cppmega.megatron.domain_route_contract import (
    DOMAIN_DELIMITER_ID_TO_DOMAIN,
    DOMAIN_END_DELIMITER_IDS,
    DOMAIN_START_DELIMITER_IDS,
)
from scripts.ci_content_store import (
    CIContentStore,
    _FRAME_HEADER,
    _FRAME_MAGIC,
    _PACK_MAGIC,
    _hash_records,
    hash_token_sequence,
)
from scripts.canonical_parquet_ledger import (
    CanonicalParquetLedgerError,
    CanonicalParquetLedgerWriter,
    iter_canonical_parquet_ledger,
)
from scripts.ci_log_sidecars import SIDECAR_SCHEMA as PARSER_SIDECAR_SCHEMA
from scripts.ci_source_binding_projection import (
    LEGACY_PARSER_SHA256,
    MAX_SOURCE_BINDING_PROJECTION_RECORD_BYTES,
    REVIEWED_PRIMARY_EQUIVALENT_PARSER_SHA256,
    REVIEWED_PRIMARY_EQUIVALENT_PARSER_UPGRADE_REASON,
    SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
    SOURCE_BINDING_PROJECTION_SCHEMA,
    target_parser_script_sha256,
)
from scripts.ci_stream_fetch import (
    SCHEMA_VERSION as FETCH_STATE_SCHEMA,
    _STATE_SCHEMA as FETCH_STATE_SQL_SCHEMA,
    ExactTokenizer,
)
from scripts.data.build_macro_routes_megatron_bundle import (
    _load_ci_manifest_allowlist,
)
from scripts.export_ci_content_store_case5 import (
    BUCKETS,
    EXPORT_SCHEMA,
    OCCURRENCE_SCHEMA,
    REPRESENTATIVE_LEDGER_SCHEMA,
    REPRESENTATIVE_METADATA_SCHEMA,
    TRAINING_SIDECAR_SCHEMA,
    ExportError,
    FrozenStore,
    OccurrenceRecord,
    _bounded_utf8_sha256,
    _decode_provenance,
    _fragment_ranges,
    _merge_bound_store_artifacts,
    _project_content,
    _publish_directory_no_replace,
    _sanitize_head_commit,
    _sequence_digest,
    _smallest_bucket,
    _source_binding_projection_writer,
    _split_for_sequence,
    _validate_occurrence_v3,
    export_store,
)
from scripts.ci_zlib_evidence import (
    MAX_CONTENT_FRAME_RAW_BYTES,
    MAX_JOBS_EVIDENCE_BYTES,
    MAX_STATE_JSON_EVIDENCE_BYTES,
)


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = ROOT / "cppmega" / "tokenizer" / "tokenizer.json"


@lru_cache(maxsize=None)
def _compressed_repetition(raw_size: int) -> tuple[bytes, str]:
    compressor = zlib.compressobj(9)
    digest = hashlib.sha256()
    parts: list[bytes] = []
    chunk = b"x" * (1024 * 1024)
    remaining = raw_size
    while remaining:
        current = chunk[: min(len(chunk), remaining)]
        digest.update(current)
        parts.append(compressor.compress(current))
        remaining -= len(current)
    parts.append(compressor.flush())
    return b"".join(parts), digest.hexdigest()


@pytest.fixture(scope="module")
def exact_tokenizer() -> ExactTokenizer:
    return ExactTokenizer(TOKENIZER_JSON)


def _span(
    start: int,
    end: int,
    *,
    domain: DomainKind = DomainKind.UNKNOWN,
    role: int = 0,
    confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        "start_char": start,
        "end_char": end,
        "domain_id": int(domain),
        "role_id": role,
        "confidence": confidence,
    }


def _entity(
    entity_id: str,
    start: int,
    end: int,
    *,
    domain: DomainKind,
    role: int,
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "kind": "fixture",
        "role": "fixture",
        "role_id": role,
        "domain": domain.name,
        "domain_id": int(domain),
        "start_char": start,
        "end_char": end,
        "line_index": 0,
        "section_ordinal": 0,
        "step_ordinal": 0,
        "confidence": {
            "score": 1.0,
            "level": "exact",
            "source": "fixture",
        },
        "attributes": {},
    }


def _provenance(
    text: str,
    *,
    ordinal: int = 0,
    domains: list[dict[str, Any]] | None = None,
    roles: list[dict[str, Any]] | None = None,
    entities: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
    cross_chunk_edges: list[dict[str, Any]] | None = None,
    archive_member: str | None = None,
) -> dict[str, Any]:
    char_count = len(text)
    domains = domains or [
        {
            "start_char": 0,
            "end_char": char_count,
            "domain_id": int(DomainKind.UNKNOWN),
            "confidence": 0.0,
        }
    ]
    roles = roles or [
        {
            "start_char": 0,
            "end_char": char_count,
            "role_id": int(DomainRoleKind.NONE),
            "confidence": 0.0,
        }
    ]
    if entities is None:
        entities = [
            _entity(
                "fixture-primary-scope",
                0,
                char_count,
                domain=DomainKind.CPP,
                role=int(DomainRoleKind.PATH),
            )
        ]
    edges = edges or []
    cross_chunk_edges = cross_chunk_edges or []
    crossing_edge_records = [
        {
            "edge_id": edge["edge_id"],
            "kind_id": edge["kind_id"],
            "from_char": edge["from_char"],
            "to_char": edge["to_member_char"],
        }
        for edge in cross_chunk_edges
    ]
    return {
        "schema": OCCURRENCE_SCHEMA,
        "repository": "owner/repo",
        "repository_requested": "owner/repo",
        "repository_id": 1,
        "source_repository": "owner/repo",
        "source_repository_id": 1,
        "repository_scope_key": "owner/repo",
        "run_id": 100,
        "run_attempt": 1,
        "run_metadata_evidence": {
            "exact_attempt_match": True,
            "source": "inventory-run-list",
            "source_attempt": 1,
            "sha256": "1" * 64,
            "inventory_seed_attempt": 1,
            "inventory_seed_metadata_sha256": "1" * 64,
        },
        "workflow": {
            "id": 77,
            "name": "CI",
            "path": ".github/workflows/ci.yml",
            "event": "push",
            "run_number": 42,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2026-07-26T10:00:00Z",
            "updated_at": "2026-07-26T10:01:00Z",
            "started_at": "2026-07-26T10:00:01Z",
            "display_title": "CI",
            "head_branch": "main",
            "head_sha": "a" * 40,
            "head_commit": {
                "id": "a" * 40,
                "message": "fixture",
                "author": {"name": "Builder"},
                "committer": {"name": "Builder"},
            },
            "actor": {"login": "builder", "id": 9},
            "triggering_actor": {"login": "builder", "id": 9},
        },
        "job": {},
        "archive": {
            "member": archive_member or "job.txt",
            "member_raw_sha256": "2" * 64,
        },
        "parser_sidecar_sha256": "3" * 64,
        "chunk": {
            "ordinal": ordinal,
            "chunk_id": f"chunk:{ordinal:06d}",
            "section_ordinal": 0,
            "section_id": "section:0",
            "step_ordinal": None,
            "char_start": 0,
            "char_end": char_count,
            "dedup_char_start": 0,
            "dedup_char_end": char_count,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "canonical_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "semantic_span_offset_basis": "chunk_local_canonical_chars",
            "role_spans": roles,
            "domain_spans": domains,
            "training_sidecars": {
                "schema": TRAINING_SIDECAR_SCHEMA,
                "coordinate_space": "chunk_local_dedup_chars_v1",
                "dedup_offsets_equal_canonical_offsets": True,
                "chunk_char_count": char_count,
                "entities": entities,
                "edges": edges,
                "commands": [],
                "build_actions": [],
                "tests": [],
                "diagnostics": [],
                "cross_chunk_edges": cross_chunk_edges,
                "cross_chunk_edge_accounting": {
                    "count": len(cross_chunk_edges),
                    "outbound_count": len(cross_chunk_edges),
                    "sha256": _sequence_digest(crossing_edge_records),
                },
            },
        },
        "section": {
            "section_id": "section:0",
            "ordinal": 0,
            "step_ordinal": None,
            "char_start": 0,
            "char_end": char_count,
            "canonical_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "dedup_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
    }


def _occurrence(
    provenance: dict[str, Any],
    *,
    content_sha256: str | None = None,
    ordinal: int = 0,
) -> OccurrenceRecord:
    if content_sha256 is None:
        content_sha256 = str(provenance["chunk"]["sha256"])
    job = provenance["job"]
    job_id = job.get("id") if isinstance(job.get("id"), int) else "unresolved"
    archive_member = provenance["archive"]["member"]
    chunk = provenance["chunk"]
    step_ordinal = chunk["step_ordinal"]
    return OccurrenceRecord(
        key=(
            str(provenance["repository_scope_key"]),
            f"{provenance['run_id']}:{provenance['run_attempt']}",
            f"{job_id}:{archive_member}",
            f"{chunk['section_id']}:"
            f"{step_ordinal if step_ordinal is not None else 'none'}",
            ordinal,
        ),
        content_sha256=content_sha256,
        provenance_sha256=hashlib.sha256(
            json.dumps(
                provenance,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        provenance=provenance,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_parquet_ledger(
    path: Path,
    *,
    domain: str | None = None,
    schema: str | None = None,
) -> list[dict[str, Any]]:
    return [
        record
        for record, _encoded in iter_canonical_parquet_ledger(
            path,
            expected_domain=domain,
            expected_record_schema=schema,
        )
    ]


def _run_metadata(provenance: dict[str, Any]) -> dict[str, Any]:
    workflow = provenance["workflow"]
    return {
        "id": provenance["run_id"],
        "run_attempt": provenance["run_attempt"],
        "workflow_id": workflow["id"],
        "name": workflow["name"],
        "path": workflow["path"],
        "event": workflow["event"],
        "run_number": workflow["run_number"],
        "status": workflow["status"],
        "conclusion": workflow["conclusion"],
        "created_at": workflow["created_at"],
        "updated_at": workflow["updated_at"],
        "run_started_at": workflow["started_at"],
        "display_title": workflow["display_title"],
        "head_branch": workflow["head_branch"],
        "head_sha": workflow["head_sha"],
        "head_commit": workflow["head_commit"],
        "actor": workflow["actor"],
        "triggering_actor": workflow["triggering_actor"],
    }


def _fixture_parser_sidecar(
    member_records: list[tuple[str, dict[str, Any], int]],
) -> tuple[dict[str, Any], str, str, int]:
    texts = [text for text, _provenance, _token_count in member_records]
    canonical_sha256 = hashlib.sha256("".join(texts).encode()).hexdigest()
    dedup_sha256 = canonical_sha256
    first_provenance = member_records[0][1]
    archive_member = first_provenance["archive"]["member"]
    is_opaque_zip = str(archive_member).casefold().endswith(".zip")
    raw_size = (
        10 if is_opaque_zip else max(1, sum(len(text.encode()) for text in texts))
    )
    invalid_count = 4 if is_opaque_zip else 0

    languages: set[str] = set()
    build_systems: set[str] = set()
    toolchains: set[str] = set()
    parser_tests: list[dict[str, Any]] = []
    for _text, provenance, _token_count in member_records:
        training = provenance["chunk"]["training_sidecars"]
        for entity in training["entities"]:
            attributes = entity.get("attributes", {})
            language = attributes.get("language") or attributes.get("likely_language")
            if isinstance(language, str):
                languages.add(language)
        for action in training["build_actions"]:
            tool = action.get("tool")
            if isinstance(tool, str):
                toolchains.add(tool)
                normalized = tool.casefold().removesuffix(".exe")
                if normalized in {"cmake", "ninja", "make", "msbuild"}:
                    build_systems.add(
                        "msbuild" if normalized == "msbuild" else normalized
                    )
        parser_tests.extend(training["tests"])

    job = first_provenance["job"]
    labels = [str(value) for value in job.get("labels", [])]
    platform: dict[str, Any] = {}
    if any(label.casefold().startswith("ubuntu-") for label in labels):
        platform["os"] = {"value": "Linux"}
    elif any(label.casefold().startswith("windows-") for label in labels):
        platform["os"] = {"value": "Windows"}
    if any(label.casefold() in {"x64", "amd64"} for label in labels):
        platform["architecture"] = {"value": "X64"}
    elif any(label.casefold() in {"arm", "arm64", "aarch64"} for label in labels):
        platform["architecture"] = {"value": "ARM64"}

    chunk_index = []
    for _text, provenance, _token_count in sorted(
        member_records,
        key=lambda item: int(item[1]["chunk"]["ordinal"]),
    ):
        chunk_index.append(
            {
                key: value
                for key, value in provenance["chunk"].items()
                if key not in {"role_spans", "domain_spans", "training_sidecars"}
            }
        )
    sidecar: dict[str, Any] = {
        "schema": PARSER_SIDECAR_SCHEMA,
        "raw": {
            "input_type": "bytes",
            "encoding": "utf-8",
            "status": "invalid_replaced" if is_opaque_zip else "valid",
            "invalid_sequence_count": invalid_count,
            "invalid_byte_spans": [],
            "replacement_char_count": invalid_count,
            "raw_byte_count": raw_size,
            "raw_sha256": first_provenance["archive"]["member_raw_sha256"],
        },
        "canonicalization": {"canonical_sha256": canonical_sha256},
        "deduplication": {"sha256": dedup_sha256},
        "provenance": {},
        "classifications": {
            "languages": [{"name": value} for value in sorted(languages)],
            "shell_dialects": [],
            "sql_dialects": [],
            "build_systems": [{"name": value} for value in sorted(build_systems)],
            "toolchains": [{"name": value} for value in sorted(toolchains)],
            "platform": platform,
            "tests": parser_tests,
        },
        "chunk_index": chunk_index,
        "section_index": [],
        "conservation": {
            "canonical_sha256": canonical_sha256,
            "dedup_sha256": dedup_sha256,
            "chunk_count": len(chunk_index),
        },
    }
    sidecar["sidecar_sha256"] = hashlib.sha256(_canonical_bytes(sidecar)).hexdigest()
    return sidecar, canonical_sha256, dedup_sha256, raw_size


def _write_fetch_state(
    path: Path,
    *,
    store_root: Path,
    store_receipt: dict[str, Any],
    exact_tokenizer: ExactTokenizer,
    records: list[tuple[str, dict[str, Any], int]],
    parser_script_sha256: str,
) -> None:
    attempts: dict[tuple[str, int, int], list[tuple[str, dict[str, Any], int]]] = {}
    members: dict[tuple[str, int, int, str], list[tuple[str, dict[str, Any], int]]] = {}
    for record in records:
        _text, provenance, _token_count = record
        attempt_key = (
            provenance["repository_scope_key"],
            provenance["run_id"],
            provenance["run_attempt"],
        )
        member_key = (*attempt_key, provenance["archive"]["member"])
        attempts.setdefault(attempt_key, []).append(record)
        members.setdefault(member_key, []).append(record)

    connection = sqlite3.connect(path)
    try:
        connection.executescript(FETCH_STATE_SQL_SCHEMA)
        settings = {
            "schema": FETCH_STATE_SCHEMA,
            "inventory_path": str(path.parent / "inventory.sqlite3"),
            "content_store_path": str(store_root.resolve()),
            "tokenizer_contract": _canonical_bytes(exact_tokenizer.contract).decode(),
            "tokenizer_fingerprint": exact_tokenizer.fingerprint,
            "fetcher_script_sha256": "4" * 64,
            "parser_script_sha256": parser_script_sha256,
            "content_store_script_sha256": store_receipt["script_sha256"],
            "chunk_semantics": (
                "parser-dedup-text-cppmega-training-tokenizer-"
                "payload-only-no-framing-v2"
            ),
            "created_at": "2026-07-26T10:00:00Z",
        }
        connection.executemany(
            "INSERT INTO settings(key,value) VALUES (?,?)",
            sorted(settings.items()),
        )
        for attempt_key, attempt_records in sorted(attempts.items()):
            provenance = attempt_records[0][1]
            metadata_raw = _canonical_bytes(_run_metadata(provenance))
            metadata_sha256 = hashlib.sha256(metadata_raw).hexdigest()
            evidence = provenance["run_metadata_evidence"]
            assert evidence["sha256"] == metadata_sha256
            jobs_by_id: dict[int, dict[str, Any]] = {}
            for _text, item_provenance, _token_count in attempt_records:
                job = item_provenance["job"]
                if job is None:
                    continue
                assert isinstance(job, dict)
                job_id = job["id"]
                assert isinstance(job_id, int) and not isinstance(job_id, bool)
                previous = jobs_by_id.setdefault(job_id, job)
                assert previous == job
            jobs_raw = _canonical_bytes(list(jobs_by_id.values()))
            attempt_member_count = sum(
                member_key[:3] == attempt_key for member_key in members
            )
            connection.execute(
                """
                INSERT INTO attempts(
                  repo,run_id,attempt,created_at,
                  run_metadata_sha256,run_metadata_raw_size,
                  run_metadata_zlib,run_metadata_source,
                  run_metadata_source_attempt,run_metadata_exact,
                  inventory_seed_attempt,inventory_seed_metadata_sha256,
                  status,tries,jobs_sha256,jobs_raw_size,jobs_zlib,
                  member_count,chunk_count,occurrence_tokens,
                  discovered_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    *attempt_key,
                    provenance["workflow"]["created_at"],
                    metadata_sha256,
                    len(metadata_raw),
                    sqlite3.Binary(zlib.compress(metadata_raw)),
                    evidence["source"],
                    evidence["source_attempt"],
                    1,
                    evidence["inventory_seed_attempt"],
                    evidence["inventory_seed_metadata_sha256"],
                    "done",
                    1,
                    hashlib.sha256(jobs_raw).hexdigest(),
                    len(jobs_raw),
                    sqlite3.Binary(zlib.compress(jobs_raw)),
                    attempt_member_count,
                    len(attempt_records),
                    sum(item[2] for item in attempt_records),
                    "2026-07-26T10:00:00Z",
                    "2026-07-26T10:01:00Z",
                ),
            )
        for member_key, member_records in sorted(members.items()):
            provenance = member_records[0][1]
            sidecar, canonical_sha256, dedup_sha256, raw_size = _fixture_parser_sidecar(
                member_records
            )
            sidecar_raw = _canonical_bytes(sidecar)
            job = provenance["job"]
            job_id = job.get("id") if isinstance(job.get("id"), int) else "unresolved"
            connection.execute(
                """
                INSERT INTO members(
                  repo,run_id,attempt,archive_member,job_key,
                  raw_sha256,raw_size,canonical_sha256,dedup_sha256,
                  sidecar_sha256,sidecar_raw_size,sidecar_zlib,
                  chunk_count,occurrence_tokens
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    *member_key,
                    f"{job_id}:{member_key[3]}",
                    provenance["archive"]["member_raw_sha256"],
                    raw_size,
                    canonical_sha256,
                    dedup_sha256,
                    hashlib.sha256(sidecar_raw).hexdigest(),
                    len(sidecar_raw),
                    sqlite3.Binary(zlib.compress(sidecar_raw)),
                    len(member_records),
                    sum(item[2] for item in member_records),
                ),
            )
        connection.commit()
    finally:
        connection.close()


def _replace_attempt_jobs(
    connection: sqlite3.Connection,
    jobs: list[dict[str, Any]],
    *,
    run_id: int = 100,
) -> None:
    raw = _canonical_bytes(jobs)
    connection.execute(
        """
        UPDATE attempts
        SET jobs_sha256=?,jobs_raw_size=?,jobs_zlib=?
        WHERE run_id=?
        """,
        (
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            sqlite3.Binary(zlib.compress(raw)),
            run_id,
        ),
    )


def _clone_attempt(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    status: str,
    member_count: int,
    chunk_count: int,
    occurrence_tokens: int,
) -> None:
    connection.row_factory = sqlite3.Row
    attempt = dict(
        connection.execute("SELECT * FROM attempts ORDER BY run_id LIMIT 1").fetchone()
    )
    attempt.update(
        {
            "run_id": run_id,
            "status": status,
            "member_count": member_count,
            "chunk_count": chunk_count,
            "occurrence_tokens": occurrence_tokens,
        }
    )
    metadata = json.loads(zlib.decompress(bytes(attempt["run_metadata_zlib"])))
    metadata["id"] = run_id
    metadata_raw = _canonical_bytes(metadata)
    metadata_sha256 = hashlib.sha256(metadata_raw).hexdigest()
    attempt.update(
        {
            "run_metadata_sha256": metadata_sha256,
            "run_metadata_raw_size": len(metadata_raw),
            "run_metadata_zlib": sqlite3.Binary(zlib.compress(metadata_raw)),
        }
    )
    if attempt["run_metadata_source"] == "inventory-run-list":
        attempt["inventory_seed_metadata_sha256"] = metadata_sha256
    columns = tuple(attempt)
    connection.execute(
        f"""
        INSERT INTO attempts({",".join(columns)})
        VALUES ({",".join("?" for _column in columns)})
        """,
        tuple(attempt[column] for column in columns),
    )


def _build_store(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
    records: list[tuple[str, dict[str, Any]]],
    *,
    wrong_token_sequence: bool = False,
    target_unique_tokens: int = 0,
    parser_script_sha256: str | None = None,
) -> tuple[Path, Path, Path]:
    for _text, provenance in records:
        if provenance.get("job") == {}:
            archive_member = str(provenance["archive"]["member"])
            provenance["job"] = {
                "id": int(
                    hashlib.sha256(archive_member.encode("utf-8")).hexdigest()[:15],
                    16,
                )
                + 1,
                "name": Path(archive_member).stem,
            }
        metadata_sha256 = hashlib.sha256(
            _canonical_bytes(_run_metadata(provenance))
        ).hexdigest()
        provenance["run_metadata_evidence"]["sha256"] = metadata_sha256
        if provenance["run_metadata_evidence"]["source"] == "inventory-run-list":
            provenance["run_metadata_evidence"]["inventory_seed_metadata_sha256"] = (
                metadata_sha256
            )

    prepared_records: list[tuple[str, dict[str, Any], int]] = []
    member_groups: dict[
        tuple[str, int, int, str], list[tuple[str, dict[str, Any], int]]
    ] = {}
    for text, provenance in records:
        token_count = len(exact_tokenizer.encode_batch([text])[0])
        prepared = (text, provenance, token_count)
        prepared_records.append(prepared)
        key = (
            provenance["repository_scope_key"],
            provenance["run_id"],
            provenance["run_attempt"],
            provenance["archive"]["member"],
        )
        member_groups.setdefault(key, []).append(prepared)
    for member_records in member_groups.values():
        sidecar, _canonical, _dedup, _raw_size = _fixture_parser_sidecar(member_records)
        for _text, provenance, _token_count in member_records:
            provenance["parser_sidecar_sha256"] = sidecar["sidecar_sha256"]

    store_root = tmp_path / "store"
    with CIContentStore(store_root) as store:
        for text, provenance in records:
            token_ids = exact_tokenizer.encode_batch([text])[0]
            sequence_sha = hash_token_sequence(token_ids)
            if wrong_token_sequence:
                sequence_sha = "0" * 64
            job = provenance["job"]
            job_id = job.get("id") if isinstance(job.get("id"), int) else "unresolved"
            chunk = provenance["chunk"]
            step_ordinal = chunk["step_ordinal"]
            store.add_chunk(
                text,
                provenance,
                {
                    "repo": provenance["repository_scope_key"],
                    "run_attempt": (
                        f"{provenance['run_id']}:{provenance['run_attempt']}"
                    ),
                    "job": f"{job_id}:{provenance['archive']['member']}",
                    "step": (
                        f"{chunk['section_id']}:"
                        f"{step_ordinal if step_ordinal is not None else 'none'}"
                    ),
                    "chunk_ordinal": chunk["ordinal"],
                },
                token_count=len(token_ids),
                tokenizer_fingerprint=exact_tokenizer.fingerprint,
                token_sequence_sha256=sequence_sha,
            )
        receipt = store.completion_receipt(target_unique_tokens=target_unique_tokens)
    receipt_path = tmp_path / "store-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fetch_state = tmp_path / "fetch-state.sqlite3"
    _write_fetch_state(
        fetch_state,
        store_root=store_root,
        store_receipt=receipt,
        exact_tokenizer=exact_tokenizer,
        records=prepared_records,
        parser_script_sha256=(
            target_parser_script_sha256()
            if parser_script_sha256 is None
            else parser_script_sha256
        ),
    )
    return store_root, receipt_path, fetch_state


def test_merge_bound_store_fast_path_is_exact_and_reports_progress(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = "compile source.cpp"
    store_root, receipt_path, _fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
    )
    def progress() -> set[tuple[str, str]]:
        captured = capsys.readouterr()
        assert captured.out == ""
        return {
            (event["phase"], event["status"])
            for line in captured.err.splitlines()
            if (event := json.loads(line))
        }

    capsys.readouterr()
    with FrozenStore(store_root, receipt_path) as store:
        snapshots = store._initial_snapshot
    assert progress().issuperset(
        {
            ("store-sqlite-integrity-check", "started"),
            ("store-sqlite-integrity-check", "complete"),
            ("store-verification", "complete:full-exporter-verification"),
        }
    )
    artifacts = {
        f"{store_root.name}/{item.relative_path}": {
            "byte_size": item.size,
            "sha256": item.sha256,
        }
        for item in snapshots
    }
    store_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bound = _merge_bound_store_artifacts(
        artifacts=artifacts,
        store_root=store_root,
        store_receipt=store_receipt,
    )
    assert bound is not None
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    malformed = {path: dict(artifact) for path, artifact in bound.items()}
    malformed["index.sqlite3"]["byte_size"] = "invalid"
    with pytest.raises(ExportError, match="index.sqlite3.*byte_size"):
        FrozenStore(
            store_root,
            receipt_path,
            merge_bound_artifacts=malformed,
            merge_bound_receipt_sha256=receipt_sha256,
        )

    with FrozenStore(
        store_root,
        receipt_path,
        merge_bound_artifacts=bound,
        merge_bound_receipt_sha256=receipt_sha256,
    ):
        pass
    events = progress()
    assert (
        "store-verification",
        "complete:merge-receipt-artifact-fast-path",
    ) in events
    assert not any(phase == "store-sqlite-integrity-check" for phase, _ in events)

    tampered = {path: dict(artifact) for path, artifact in bound.items()}
    tampered["index.sqlite3"]["sha256"] = "f" * 64
    with pytest.raises(
        ExportError,
        match="differs from merge receipt artifacts",
    ), FrozenStore(
        store_root,
        receipt_path,
        merge_bound_artifacts=tampered,
        merge_bound_receipt_sha256=receipt_sha256,
    ):
        pass
    del artifacts[f"{store_root.name}/index.sqlite3"]
    assert (
        _merge_bound_store_artifacts(
            artifacts=artifacts,
            store_root=store_root,
            store_receipt=store_receipt,
        )
        is None
    )


def _assert_balanced_case5_row(row: dict[str, Any], *, bucket: int) -> None:
    valid = int(row["valid_token_count"])
    assert len(row["input_ids"]) == bucket
    assert len(row["target_ids"]) == bucket
    assert len(row["loss_mask"]) == bucket
    assert len(row["token_domain_ids"]) == bucket
    assert len(row["token_role_ids"]) == bucket
    assert len(row["token_confidence_ids"]) == bucket
    assert row["input_ids"][valid:] == [0] * (bucket - valid)
    assert row["target_ids"] == row["input_ids"][1:] + [0]
    assert row["token_domain_ids"][valid:] == [0] * (bucket - valid)
    assert row["token_role_ids"][valid:] == [0] * (bucket - valid)
    assert row["token_confidence_ids"][valid:] == [0] * (bucket - valid)
    assert int(row["trained_token_count"]) == sum(row["loss_mask"])

    stack: list[int] = []
    for token_id, domain_id, role_id, confidence_id in zip(
        row["input_ids"][:valid],
        row["token_domain_ids"][:valid],
        row["token_role_ids"][:valid],
        row["token_confidence_ids"][:valid],
        strict=True,
    ):
        token_id = int(token_id)
        domain_id = int(domain_id)
        if token_id in DOMAIN_START_DELIMITER_IDS:
            assert role_id == int(DomainRoleKind.DELIMITER)
            assert confidence_id == int(ParseConfidence.EXACT)
            assert domain_id == DOMAIN_DELIMITER_ID_TO_DOMAIN[token_id]
            stack.append(domain_id)
        elif token_id in DOMAIN_END_DELIMITER_IDS:
            assert role_id == int(DomainRoleKind.DELIMITER)
            assert confidence_id == int(ParseConfidence.EXACT)
            assert stack.pop() == domain_id
        else:
            assert int(role_id) != int(DomainRoleKind.DELIMITER)
            assert domain_id == (stack[-1] if stack else int(DomainKind.UNKNOWN))
    assert stack == []


def test_canonical_parquet_ledger_is_bounded_zstd_and_logically_hashed(
    tmp_path: Path,
) -> None:
    records = [
        {"schema": "fixture_v1", "value": 1},
        {"schema": "fixture_v1", "value": "two"},
    ]
    path = tmp_path / "ledger.parquet"
    writer = CanonicalParquetLedgerWriter(
        path,
        domain="fixture-domain-v1",
        max_record_bytes=128,
        row_group_rows=1,
    )
    for record in records:
        writer.append(record)
    writer.close()

    assert writer.logical_sha256 == _hash_records(
        "fixture-domain-v1",
        iter(records),
    )
    assert [
        value
        for value, _encoded in iter_canonical_parquet_ledger(
            path,
            expected_domain="fixture-domain-v1",
            expected_record_schema="fixture_v1",
            max_record_bytes=128,
        )
    ] == records
    metadata = pq.ParquetFile(path).metadata
    assert metadata.num_rows == 2
    assert {
        str(
            metadata.row_group(row_group).column(column).compression
        )
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    } == {"ZSTD"}

    oversized = CanonicalParquetLedgerWriter(
        tmp_path / "oversized.parquet",
        max_record_bytes=8,
    )
    with pytest.raises(
        CanonicalParquetLedgerError,
        match="record exceeds",
    ):
        oversized.append({"too": "large"})
    oversized.close()


def test_tiny_end_to_end_selects_stable_token_sequence_representative(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    texts = ["alpha beta", "alpha      beta"]
    token_rows = exact_tokenizer.encode_batch(texts)
    assert token_rows[0] == token_rows[1]
    records = [
        (text, _provenance(text, archive_member=f"0_job-{index}.txt"))
        for index, text in enumerate(texts)
    ]
    records[0][1]["workflow"]["actor"] = {"login": "linux-builder", "id": 1}
    records[0][1]["job"] = {
        "id": 1001,
        "name": "linux",
        "labels": ["ubuntu-24.04", "x64"],
    }
    records[1][1]["run_id"] = 101
    records[1][1]["workflow"]["actor"] = {"login": "windows-builder", "id": 2}
    records[1][1]["job"] = {
        "id": 1002,
        "name": "windows",
        "labels": ["windows-2025", "x64"],
    }
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path, exact_tokenizer, records
    )

    output = tmp_path / "case5"
    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
        require_current_parser_only=True,
    )

    assert receipt["schema"] == EXPORT_SCHEMA
    assert receipt["status"] == "complete"
    assert receipt["parser_generation_policy"] == {
        "mode": "current-singleton-required",
        "expected_current_parser_script_sha256": (
            target_parser_script_sha256()
        ),
        "observed_parser_lineage": [target_parser_script_sha256()],
        "current_singleton": True,
    }
    assert receipt["representatives"]["count"] == 1
    assert receipt["counts"]["payload_tokens"] == len(token_rows[0])
    assert receipt["case5_contract"]["overflow_rows"] == 0
    assert receipt["case5_contract"]["parquet_compression"] == {
        "codec": "zstd",
        "level": 9,
    }
    assert receipt["validation"]["all_case5_parquet_zstd"] is True
    assert (
        receipt["case5_contract"]["parquet_layout"]
        == "bucket-first-split-in-filename-v1"
    )
    ledger = _read_parquet_ledger(
        output / receipt["representatives"]["ledger_artifact"],
        domain="cppmega-ci-case5-representative-ledger-v1",
        schema=REPRESENTATIVE_LEDGER_SCHEMA,
    )
    expected_content = min(
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
    )
    assert ledger[0]["representative_content_sha256"] == expected_content
    assert ledger[0]["candidate_content_count"] == 2
    assert set(ledger[0]["representative_occurrence_key"]) == {
        "repo",
        "run_attempt",
        "job",
        "step",
        "chunk_ordinal",
    }
    assert receipt["representatives"]["ledger_sha256"] == _hash_records(
        "cppmega-ci-case5-representative-ledger-v1",
        iter(ledger),
    )
    ledger_bytes = (output / receipt["representatives"]["ledger_artifact"]).read_bytes()
    assert (
        receipt["representatives"]["ledger_artifact_sha256"]
        == hashlib.sha256(ledger_bytes).hexdigest()
    )
    assert (output / "export_receipt.json").is_file()
    occurrence_receipt = receipt["occurrence_metadata"]
    assert occurrence_receipt["count"] == 2
    occurrence_records = [
        record
        for record, _encoded in iter_canonical_parquet_ledger(
            output / occurrence_receipt["artifact"],
            expected_domain=occurrence_receipt["logical_domain"],
            expected_record_schema=occurrence_receipt["schema"],
        )
    ]
    assert {
        (
            record["workflow"]["actor"]["login"],
            tuple(record["runner_evidence"]["os_label_evidence"]),
        )
        for record in occurrence_records
    } == {
        ("linux-builder", ("ubuntu-24.04",)),
        ("windows-builder", ("windows-2025",)),
    }

    parquet_artifacts = [
        artifact
        for artifact in receipt["artifacts"]
        if artifact["kind"] == "case5_parquet"
    ]
    assert len(parquet_artifacts) == 1
    artifact = parquet_artifacts[0]
    assert Path(str(artifact["path"])).parts == (
        str(artifact["bucket"]),
        (
            f"ci-case5-{artifact['split']}-{artifact['bucket']}-"
            "000000.parquet"
        ),
    )
    row = pq.read_table(output / artifact["path"]).to_pylist()[0]
    _assert_balanced_case5_row(row, bucket=int(artifact["bucket"]))
    allowed, normalized = _load_ci_manifest_allowlist(
        output / "export_receipt.json",
        output,
        (int(artifact["bucket"]),),
        cppmega_mlx_commit="unused-for-content-store-export",
        cppmega_mlx_tree_sha256="unused-for-content-store-export",
    )
    assert allowed == {
        ("ci", int(artifact["bucket"])): {
            Path(str(artifact["path"])).name: int(artifact["rows"])
        }
    }
    assert normalized["schema"] == EXPORT_SCHEMA
    assert len(normalized["provenance_artifacts"]) == 8


def test_nested_zip_binary_garbage_is_conserved_but_not_trained(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    eligible_text = "compile source.cpp with clang"
    opaque_text = "PK\u0003\u0004\ufffd\ufffd\ufffd diagnostic archive bytes"
    eligible = _provenance(
        eligible_text,
        archive_member="job-log.txt",
    )
    opaque = _provenance(
        opaque_text,
        archive_member="runner-diagnostic-logs/runner.zip",
    )
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(eligible_text, eligible), (opaque_text, opaque)],
    )
    output = tmp_path / "opaque-filtered"

    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )

    eligible_tokens = len(exact_tokenizer.encode_batch([eligible_text])[0])
    opaque_tokens = len(exact_tokenizer.encode_batch([opaque_text])[0])
    eligibility = receipt["eligibility"]
    assert eligibility["cas"]["exact_unique_payload_tokens"] == (
        eligible_tokens + opaque_tokens
    )
    assert eligibility["eligible"]["exact_unique_payload_tokens"] == eligible_tokens
    assert eligibility["excluded_only"]["exact_unique_payload_tokens"] == opaque_tokens
    assert eligibility["conservation"] == {
        "exact_unique_payload_tokens": True,
        "unique_token_sequences": True,
    }
    assert receipt["representatives"]["count"] == 1
    excluded_path = (
        output / eligibility["excluded_opaque_occurrences"]["ledger"]
    )
    excluded = _read_parquet_ledger(
        excluded_path,
        domain="cppmega-ci-case5-excluded-opaque-artifact-ledger-v1",
        schema="cppmega_ci_case5_excluded_opaque_artifact_v1",
    )
    assert len(excluded) == 1
    assert excluded[0]["archive_member"].endswith("/runner.zip")
    assert excluded[0]["exact_token_count"] == opaque_tokens
    assert excluded[0]["decode_evidence"]["invalid_ratio_ppm_floor"] == 400_000
    representatives = _read_parquet_ledger(
        output / receipt["representatives"]["ledger_artifact"],
        domain="cppmega-ci-case5-representative-ledger-v1",
        schema=REPRESENTATIVE_LEDGER_SCHEMA,
    )
    assert (
        representatives[0]["representative_content_sha256"]
        == hashlib.sha256(eligible_text.encode()).hexdigest()
    )
    occurrence_receipt = receipt["occurrence_metadata"]
    occurrence_records = [
        record
        for record, _encoded in iter_canonical_parquet_ledger(
            output / occurrence_receipt["artifact"],
            expected_domain=occurrence_receipt["logical_domain"],
            expected_record_schema=occurrence_receipt["schema"],
        )
    ]
    assert {
        record["case5_eligibility"]["status"]
        for record in occurrence_records
    } == {"eligible_primary", "excluded_opaque"}


def test_opaque_tokens_cannot_satisfy_the_eligible_target(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    eligible_text = "eligible compiler output"
    opaque_text = "PK\u0003\u0004\ufffd\ufffd\ufffd opaque nested zip"
    total_tokens = sum(
        len(row) for row in exact_tokenizer.encode_batch([eligible_text, opaque_text])
    )
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [
            (
                eligible_text,
                _provenance(eligible_text, archive_member="job-log.txt"),
            ),
            (
                opaque_text,
                _provenance(
                    opaque_text,
                    archive_member="runner-diagnostic-logs/runner.zip",
                ),
            ),
        ],
        target_unique_tokens=total_tokens,
    )
    output = tmp_path / "below-eligible-target"

    with pytest.raises(ExportError, match="eligible exact unique payload tokens"):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=output,
        )
    assert not output.exists()


def test_primary_scope_excludes_unrelated_and_routes_python_auxiliary(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    primary_text = "clang++ -c src/main.cpp -o build/main.o"
    python_text = "tests/test_api.py::test_health PASSED"
    irrelevant_text = "Downloading unrelated package metadata"
    primary = _provenance(primary_text, archive_member="build.txt")
    python = _provenance(
        python_text,
        archive_member="python-tests.txt",
        entities=[
            _entity(
                "python-path",
                0,
                len(python_text),
                domain=DomainKind.PYTHON,
                role=int(DomainRoleKind.PATH),
            )
        ],
    )
    irrelevant = _provenance(
        irrelevant_text,
        archive_member="setup.txt",
        entities=[],
    )
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [
            (primary_text, primary),
            (python_text, python),
            (irrelevant_text, irrelevant),
        ],
    )

    output = tmp_path / "scope-routed"
    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )

    eligibility = receipt["eligibility"]
    assert eligibility["eligible"]["unique_token_sequences"] == 1
    assert eligibility["excluded_training_scope_occurrences"][
        "occurrences"
    ] == 2
    excluded = _read_parquet_ledger(
        output
        / eligibility["excluded_training_scope_occurrences"]["ledger"],
        domain="cppmega-ci-case5-excluded-training-scope-ledger-v1",
        schema="cppmega_ci_case5_excluded_training_scope_v1",
    )
    assert {
        tuple(record["effective_routes"])
        for record in excluded
    } == {(), ("aux_python",)}
    assert eligibility["routing_accounting"]["occurrence_counts"] == {
        "aux_python": 1,
        "excluded_irrelevant": 1,
        "primary_cpp_sql_build_test": 1,
    }


def test_primary_scope_propagates_only_inside_the_exact_step(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    command_text = "cmake --build build --target all"
    output_text = "[42/100] Building native target"
    command = _provenance(
        command_text,
        ordinal=0,
        archive_member="native-build.txt",
    )
    output_chunk = _provenance(
        output_text,
        ordinal=1,
        archive_member="native-build.txt",
        entities=[],
    )
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(command_text, command), (output_text, output_chunk)],
    )

    output = tmp_path / "step-propagated"
    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )

    assert receipt["eligibility"]["eligible"]["unique_token_sequences"] == 2
    occurrences = [
        record
        for record, _encoded in iter_canonical_parquet_ledger(
            output / receipt["occurrence_metadata"]["artifact"],
            expected_domain=receipt["occurrence_metadata"]["logical_domain"],
            expected_record_schema=receipt["occurrence_metadata"]["schema"],
        )
    ]
    propagated = next(
        record
        for record in occurrences
        if record["content_sha256"]
        == hashlib.sha256(output_text.encode()).hexdigest()
    )
    scope = propagated["case5_eligibility"]["training_scope"]
    assert scope["local_primary"] is False
    assert scope["effective_primary"] is True
    assert "propagated:exact_step_primary_evidence" in scope["reasons"]


def test_primary_scope_does_not_cross_step_boundary(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    command_text = "cmake --build build --target all"
    unrelated_text = "Downloading unrelated package metadata"
    command = _provenance(
        command_text,
        ordinal=0,
        archive_member="same-job.txt",
    )
    unrelated = _provenance(
        unrelated_text,
        ordinal=1,
        archive_member="same-job.txt",
        entities=[],
    )
    unrelated["chunk"]["section_id"] = "section:1"
    unrelated["chunk"]["section_ordinal"] = 1
    unrelated["chunk"]["step_ordinal"] = 1
    unrelated["section"]["section_id"] = "section:1"
    unrelated["section"]["ordinal"] = 1
    unrelated["section"]["step_ordinal"] = 1
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(command_text, command), (unrelated_text, unrelated)],
    )

    output = tmp_path / "step-isolated"
    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )

    eligibility = receipt["eligibility"]
    assert eligibility["eligible"]["unique_token_sequences"] == 1
    assert eligibility["excluded_training_scope_occurrences"][
        "occurrences"
    ] == 1
    occurrences = [
        record
        for record, _encoded in iter_canonical_parquet_ledger(
            output / receipt["occurrence_metadata"]["artifact"],
            expected_domain=receipt["occurrence_metadata"]["logical_domain"],
            expected_record_schema=receipt["occurrence_metadata"]["schema"],
        )
    ]
    excluded = next(
        record
        for record in occurrences
        if record["content_sha256"]
        == hashlib.sha256(unrelated_text.encode()).hexdigest()
    )
    scope = excluded["case5_eligibility"]["training_scope"]
    assert scope["effective_routes"] == []
    assert scope["effective_primary"] is False


def test_explicit_eligible_target_uses_receipt_bound_cas_reserve(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    eligible_text = "eligible compiler output with exact provenance"
    opaque_text = "PK\u0003\u0004\ufffd\ufffd\ufffd opaque nested zip reserve"
    eligible_tokens, opaque_tokens = (
        len(row)
        for row in exact_tokenizer.encode_batch([eligible_text, opaque_text])
    )
    acquisition_target = eligible_tokens + opaque_tokens
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [
            (
                eligible_text,
                _provenance(eligible_text, archive_member="job-log.txt"),
            ),
            (
                opaque_text,
                _provenance(
                    opaque_text,
                    archive_member="runner-diagnostic-logs/runner.zip",
                ),
            ),
        ],
        target_unique_tokens=acquisition_target,
    )
    output = tmp_path / "receipt-bound-cas-reserve"

    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
        required_eligible_exact_unique_payload_tokens=eligible_tokens,
    )

    eligibility = receipt["eligibility"]
    assert eligibility["target_exact_unique_payload_tokens"] == eligible_tokens
    assert eligibility["target_source"] == "explicit_export_requirement"
    assert (
        eligibility["cas_acquisition_target_exact_unique_payload_tokens"]
        == acquisition_target
    )
    assert (
        eligibility["cas_reserve_exact_unique_payload_tokens"]
        == opaque_tokens
    )
    assert eligibility["eligible"]["exact_unique_payload_tokens"] == eligible_tokens
    assert eligibility["target_met"] is True

    with pytest.raises(
        ExportError,
        match="exceed the receipt-bound CAS acquisition target",
    ):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=tmp_path / "invalid-target",
            required_eligible_exact_unique_payload_tokens=(
                acquisition_target + 1
            ),
        )


def test_representative_metadata_is_explicit_sanitized_and_receipt_bound(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "compile test diagnostic"
    entity = _entity(
        "entity:000000",
        0,
        len("compile"),
        domain=DomainKind.CPP,
        role=int(DomainRoleKind.SOURCE),
    )
    entity["kind"] = "path"
    entity["attributes"] = {"likely_language": "C++", "raw_url": "https://ignored"}
    provenance = _provenance(text, entities=[entity])
    provenance["workflow"] = {
        "id": 77,
        "name": "CI",
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "run_number": 42,
        "status": "completed",
        "conclusion": "failure",
        "created_at": "2026-07-26T10:00:00Z",
        "updated_at": "2026-07-26T10:01:00Z",
        "started_at": "2026-07-26T10:00:01Z",
        "display_title": "CI metadata",
        "head_branch": "main",
        "head_sha": "a" * 40,
        "actor": {
            "login": "builder",
            "id": 9,
            "type": "User",
            "email": "public@example.test",
            "avatar_url": "https://example.test/avatar",
        },
        "triggering_actor": {
            "login": "rerunner",
            "id": 10,
            "url": "https://api.example.test/user",
        },
        "head_commit": {
            "id": "a" * 40,
            "message": "compile metadata",
            "author": {"name": "Builder", "email": "public@example.test"},
            "committer": {"name": "Builder", "email": "public@example.test"},
            "url": {"huge": ["https://api.example.test"] * 100},
        },
    }
    provenance["job"] = {
        "id": 99,
        "name": "build (ubuntu-24.04, x64)",
        "status": "completed",
        "conclusion": "failure",
        "runner_id": 1001,
        "runner_name": "GitHub Actions 1001",
        "runner_group_id": 1,
        "runner_group_name": "GitHub Actions",
        "labels": ["ubuntu-24.04", "x64"],
        "steps": [
            {
                "number": 1,
                "name": "Compile",
                "status": "completed",
                "conclusion": "failure",
                "url": "https://api.example.test/step",
            }
        ],
        "html_url": {"huge": ["https://api.example.test"] * 100},
    }
    provenance["archive"]["member"] = "0_build__ubuntu-24.04__x64_.txt"
    provenance["section"]["kind"] = "step"
    provenance["section"]["title"] = "Compile"
    training = provenance["chunk"]["training_sidecars"]
    training["build_actions"] = [
        {
            "normalization_schema": "fixture-v1",
            "tool": "clang++",
            "kind": "compile",
            "cwd": "/work/repo",
            "source_inputs": ["src/main.C"],
            "source_input_count": 1,
            "outputs": ["build/main.o"],
            "output_count": 1,
            "flags": ["-O2", "-c"],
            "target": "build/main.o",
            "command": "clang++ -O2 -c src/main.cpp",
            "command_sha256": "5" * 64,
            "action_shape_sha256": "6" * 64,
            "repository_source_bindings": [
                {
                    "repository": "owner/repo",
                    "head_sha": "a" * 40,
                    "source_path": "src/main.C",
                    "confidence": {
                        "score": 0.95,
                        "level": "high",
                        "source": "relative_source_path_v1",
                    },
                }
            ],
            "repository_source_binding_count": 1,
            "start_char": 0,
            "end_char": len("compile"),
            "line_index": 0,
            "section_ordinal": 0,
            "step_ordinal": None,
            "confidence": {
                "score": 0.98,
                "level": "high",
                "source": "fixture",
            },
        }
    ]
    training["tests"] = [
        {
            "framework": "pytest",
            "suite": "unit",
            "case": "test_export",
            "result": "failed",
            "count": 1,
            "duration_ms": 12.5,
            "start_char": len("compile "),
            "end_char": len("compile test"),
            "line_index": 0,
            "section_ordinal": 0,
            "step_ordinal": None,
            "confidence": {
                "score": 1.0,
                "level": "exact",
                "source": "fixture",
            },
        }
    ]
    training["diagnostics"] = [
        {
            "category": "compiler",
            "tool": "clang",
            "severity": "error",
            "code": "E1",
            "file": "src/main.C",
            "source_line": 3,
            "source_column": 2,
            "message": "raw diagnostic text must not be copied",
            "start_char": len("compile test "),
            "end_char": len(text),
            "line_index": 0,
            "section_ordinal": 0,
            "step_ordinal": None,
            "confidence": {
                "score": 1.0,
                "level": "exact",
                "source": "fixture",
            },
        }
    ]
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, provenance)],
    )
    output = tmp_path / "metadata-case5"

    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )

    metadata_receipt = receipt["representative_metadata"]
    assert metadata_receipt["schema"] == REPRESENTATIVE_METADATA_SCHEMA
    assert metadata_receipt["count"] == 1
    metadata = _read_parquet_ledger(
        output / metadata_receipt["artifact"],
        schema=metadata_receipt["schema"],
    )
    assert len(metadata) == 1
    metadata = metadata[0]
    assert metadata["run"]["run_id"] == 100
    assert metadata["run"]["run_attempt"] == 1
    assert metadata["run"]["metadata_evidence"]["exact_attempt_match"] is True
    assert metadata["workflow"]["actor"]["login"] == "builder"
    assert metadata["workflow"]["triggering_actor"]["login"] == "rerunner"
    assert metadata["job"]["id"] == 99
    assert metadata["job"]["steps"][0]["name"] == "Compile"
    assert metadata["step"]["title"] == "Compile"
    assert metadata["runner_evidence"]["runner_name"] == "GitHub Actions 1001"
    assert metadata["runner_evidence"]["os_label_evidence"] == ["ubuntu-24.04"]
    assert metadata["runner_evidence"]["architecture_label_evidence"] == ["x64"]
    sidecars = metadata["training_sidecars"]
    assert sidecars["language_evidence"][0]["language"] == "C++"
    assert sidecars["domain_spans"] == provenance["chunk"]["domain_spans"]
    assert sidecars["role_spans"] == provenance["chunk"]["role_spans"]
    assert sidecars["build_actions"][0]["source_inputs"] == ["src/main.C"]
    assert sidecars["build_actions"][0]["flags"] == ["-O2", "-c"]
    assert sidecars["tests"][0]["framework"] == "pytest"
    assert sidecars["diagnostics"][0]["code"] == "E1"

    def all_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(
                *(all_keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(all_keys(item) for item in value))
        return set()

    assert not {
        "email",
        "url",
        "html_url",
        "avatar_url",
        "logs_url",
        "command",
        "message",
    }.intersection(all_keys(metadata))
    classifications = metadata["derived_classifications"]
    assert classifications["scope_contract"] == {
        "workflow_job_runner": "exact-occurrence-api-metadata",
        "parser_classifications": "archive-member",
        "training_sidecars": "chunk-local",
        "derived_values": (
            "typed union; evidence_source identifies member versus chunk scope"
        ),
    }
    assert classifications["language"]["status"] == "resolved"
    assert classifications["shell_dialect"]["status"] == "unresolved"
    assert classifications["sql_dialect"]["status"] == "unresolved"
    assert {
        (item["extension"], item["language"])
        for item in classifications["language"]["source_extension_evidence"]
    } == {(".C", "C++")}
    assert classifications["source_extension"]["values"] == [".C"]
    assert classifications["system"]["values"] == ["linux"]
    assert classifications["platform"]["value"]["architecture_labels"] == ["x64"]
    assert classifications["platform"]["completeness"] == "complete"
    assert classifications["runner"]["value"]["runner_name"] == ("GitHub Actions 1001")
    assert classifications["build_system"]["status"] == "unresolved"
    assert classifications["test"]["value"]["framework"]["values"] == ["pytest"]
    assert classifications["tool"]["values"] == ["clang", "clang++"]
    assert classifications["action_kind"]["values"] == ["compile"]
    occurrence_receipt = receipt["occurrence_metadata"]
    assert occurrence_receipt["count"] == 1
    occurrence_path = output / occurrence_receipt["artifact"]
    occurrence_records = list(
        iter_canonical_parquet_ledger(
            occurrence_path,
            expected_domain=occurrence_receipt["logical_domain"],
            expected_record_schema=occurrence_receipt["schema"],
        )
    )
    assert len(occurrence_records) == 1
    occurrence_metadata = occurrence_records[0][0]
    assert occurrence_metadata["scope"] == (
        "one-record-per-frozen-cas-occurrence"
    )
    assert occurrence_metadata["case5_eligibility"]["status"] == (
        "eligible_primary"
    )
    assert occurrence_metadata["case5_eligibility"]["reason"] is None
    assert occurrence_metadata["case5_eligibility"]["primary_eligible"] is True
    assert occurrence_metadata["case5_eligibility"]["training_scope"][
        "effective_routes"
    ] == ["primary_cpp_sql_build_test"]
    parquet_metadata = pq.ParquetFile(occurrence_path).metadata
    assert {
        str(
            parquet_metadata.row_group(row_group)
            .column(column)
            .compression
        )
        for row_group in range(parquet_metadata.num_row_groups)
        for column in range(parquet_metadata.num_columns)
    } == {"ZSTD"}


def test_metadata_ledgers_accept_bounded_records_over_one_mib(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "compile source.cpp"
    provenance = _provenance(text)
    provenance["archive"]["member"] = "large bounded job metadata.txt"
    provenance["job"] = {
        "id": 99,
        "name": "large bounded job metadata",
        "steps": [
            {"number": index + 1, "name": "x" * 256}
            for index in range(4_096)
        ],
    }
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, provenance)],
    )
    output = tmp_path / "large-bounded-metadata"

    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )

    occurrence = _read_parquet_ledger(
        output / receipt["occurrence_metadata"]["artifact"],
        domain=receipt["occurrence_metadata"]["logical_domain"],
        schema=receipt["occurrence_metadata"]["schema"],
    )[0]
    representative = _read_parquet_ledger(
        output / receipt["representative_metadata"]["artifact"],
        schema=receipt["representative_metadata"]["schema"],
    )[0]
    assert len(_canonical_bytes(occurrence)) > 1024 * 1024
    assert len(_canonical_bytes(representative)) > 1024 * 1024


def test_head_commit_message_fingerprint_accepts_more_than_16k_chars() -> None:
    message = "large commit message\n" + "\u03bb" * 16_384

    sanitized = _sanitize_head_commit(
        {
            "id": "a" * 40,
            "message": message,
            "author": {"name": "Builder"},
            "committer": {"name": "Builder"},
        }
    )

    assert sanitized is not None
    assert sanitized["message_char_count"] == len(message)
    assert sanitized["message_sha256"] == hashlib.sha256(
        message.encode("utf-8")
    ).hexdigest()
    assert "message" not in sanitized


def test_bounded_utf8_fingerprint_enforces_encoded_byte_limit() -> None:
    with pytest.raises(ExportError, match="state JSON evidence limit"):
        _bounded_utf8_sha256(
            "\u03bb" * 16_384,
            where="workflow.head_commit.message",
            max_bytes=16_384,
        )


def test_legacy_source_bindings_require_authorization_and_export_as_overlay(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "clang++ -c src/main.cpp"
    provenance = _provenance(text)
    provenance["repository"] = "owner/base"
    provenance["repository_requested"] = "owner/base"
    provenance["repository_scope_key"] = "owner/base"
    provenance["source_repository"] = "fork/base"
    provenance["source_repository_id"] = 2
    provenance["workflow"]["event"] = "pull_request"
    provenance["archive"]["member"] = "0_build.txt"
    provenance["job"] = {
        "id": 99,
        "name": "build",
        "status": "completed",
        "conclusion": "success",
        "labels": ["ubuntu-24.04", "x64"],
    }
    action = {
        "normalization_schema": "cppmega_ci_build_action_normalization_v1",
        "tool": "clang++",
        "kind": "compile",
        "cwd": "/home/runner/work/base/base/build",
        "source_inputs": ["src/main.cpp"],
        "source_input_count": 1,
        "outputs": [],
        "output_count": 0,
        "flags": ["-c"],
        "repository_source_bindings": [
            {
                "repository": "fork/base",
                "head_sha": "a" * 40,
                "source_path": "src/main.cpp",
                "confidence": {
                    "score": 0.95,
                    "level": "high",
                    "source": "relative_source_path_v1",
                },
            }
        ],
        "repository_source_binding_count": 1,
        "command_sha256": "5" * 64,
        "action_shape_sha256": "6" * 64,
        "start_char": 0,
        "end_char": len(text),
        "line_index": 0,
        "section_ordinal": 0,
        "step_ordinal": None,
        "confidence": {
            "score": 0.98,
            "level": "high",
            "source": "fixture",
        },
    }
    provenance["chunk"]["training_sidecars"]["build_actions"] = [action]
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, provenance)],
        parser_script_sha256=LEGACY_PARSER_SHA256,
    )
    store_index = store_root / "index.sqlite3"
    input_hashes = {
        "receipt": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "store_index": hashlib.sha256(store_index.read_bytes()).hexdigest(),
        "fetch_state": hashlib.sha256(fetch_state.read_bytes()).hexdigest(),
    }

    refused_output = tmp_path / "legacy-refused"
    with pytest.raises(ExportError, match="requires exact explicit authorization"):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=refused_output,
        )
    assert not refused_output.exists()

    output = tmp_path / "legacy-projected"
    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
        source_binding_projection_from_parser_sha256=LEGACY_PARSER_SHA256,
    )
    projection = receipt["source_binding_projection"]
    assert projection["schema"] == SOURCE_BINDING_PROJECTION_SCHEMA
    assert projection["mode"] == "legacy_projection"
    assert projection["input_parser_script_sha256"] == LEGACY_PARSER_SHA256
    assert projection["coverage"] == {
        "order": "occurrence-key-then-action-index-then-source-input-index",
        "occurrence_count": 1,
        "action_count": 1,
        "source_input_count": 1,
        "old_binding_count": 1,
        "projected_binding_count": 1,
    }
    assert projection["change_counts"] == {"modified": 1}
    ledger = _read_parquet_ledger(
        output / projection["ledger_artifact"],
        domain=SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
        schema=SOURCE_BINDING_PROJECTION_SCHEMA,
    )[0]
    assert ledger["old_binding"]["repository"] == "fork/base"
    assert ledger["projected_binding"]["repository"] == "owner/base"
    assert ledger["projected_binding"]["source_path"] == "build/src/main.cpp"

    metadata_path = output / receipt["representative_metadata"]["artifact"]
    metadata = _read_parquet_ledger(
        metadata_path,
        schema=receipt["representative_metadata"]["schema"],
    )[0]
    projected_action = metadata["training_sidecars"]["build_actions"][0]
    assert projected_action["repository_source_bindings"][0]["repository"] == (
        "owner/base"
    )
    assert projected_action["repository_source_bindings"][0]["source_path"] == (
        "build/src/main.cpp"
    )
    assert metadata["source_binding_projection"]["mode"] == "legacy_projection"
    assert {
        "receipt": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "store_index": hashlib.sha256(store_index.read_bytes()).hexdigest(),
        "fetch_state": hashlib.sha256(fetch_state.read_bytes()).hexdigest(),
    } == input_hashes


def test_mixed_parser_generation_store_routes_legacy_and_current_actions(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    def mixed_provenance(
        text: str,
        *,
        run_id: int,
        archive_member: str,
        binding_repository: str,
        binding_source_path: str,
    ) -> dict[str, Any]:
        provenance = _provenance(text)
        provenance["repository"] = "owner/base"
        provenance["repository_requested"] = "owner/base"
        provenance["repository_scope_key"] = "owner/base"
        provenance["source_repository"] = "fork/base"
        provenance["source_repository_id"] = 2
        provenance["run_id"] = run_id
        provenance["workflow"]["id"] = run_id
        provenance["workflow"]["event"] = "pull_request"
        provenance["archive"]["member"] = archive_member
        provenance["job"] = {
            "id": run_id + 1000,
            "name": f"build-{run_id}",
            "status": "completed",
            "conclusion": "success",
            "labels": ["ubuntu-24.04", "x64"],
        }
        provenance["chunk"]["training_sidecars"]["build_actions"] = [
            {
                "normalization_schema": (
                    "cppmega_ci_build_action_normalization_v1"
                ),
                "tool": "clang++",
                "kind": "compile",
                "cwd": "/home/runner/work/base/base/build",
                "source_inputs": ["src/main.cpp"],
                "source_input_count": 1,
                "outputs": [],
                "output_count": 0,
                "flags": ["-c"],
                "repository_source_bindings": [
                    {
                        "repository": binding_repository,
                        "head_sha": "a" * 40,
                        "source_path": binding_source_path,
                        "confidence": {
                            "score": 0.95,
                            "level": "high",
                            "source": "relative_source_path_v1",
                        },
                    }
                ],
                "repository_source_binding_count": 1,
                "command_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "action_shape_sha256": hashlib.sha256(
                    f"shape:{text}".encode()
                ).hexdigest(),
                "start_char": 0,
                "end_char": len(text),
                "line_index": 0,
                "section_ordinal": 0,
                "step_ordinal": None,
                "confidence": {
                    "score": 0.98,
                    "level": "high",
                    "source": "fixture",
                },
            }
        ]
        return provenance

    legacy_text = "clang++ -c src/legacy.cpp"
    current_text = "clang++ -c src/current.cpp"
    legacy = mixed_provenance(
        legacy_text,
        run_id=101,
        archive_member="0_build-101.txt",
        binding_repository="fork/base",
        binding_source_path="src/main.cpp",
    )
    current = mixed_provenance(
        current_text,
        run_id=102,
        archive_member="0_build-102.txt",
        binding_repository="owner/base",
        binding_source_path="build/src/main.cpp",
    )
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(legacy_text, legacy), (current_text, current)],
    )
    with sqlite3.connect(fetch_state) as connection:
        connection.execute(
            """
            INSERT INTO binding_upgrades(
              binding_key,from_sha256,to_sha256,reason,upgraded_at
            ) VALUES (?,?,?,?,?)
            """,
            (
                "parser_script_sha256",
                LEGACY_PARSER_SHA256,
                target_parser_script_sha256(),
                "source binding semantics changed without rewriting frozen CAS",
                "2026-07-26T23:00:00Z",
            ),
        )
        connection.commit()

    refused = tmp_path / "mixed-refused"
    with pytest.raises(
        ExportError,
        match="requires exact explicit authorization",
    ):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=refused,
        )
    assert not refused.exists()

    output = tmp_path / "mixed-projected"
    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
        source_binding_projection_from_parser_sha256=LEGACY_PARSER_SHA256,
    )
    projection = receipt["source_binding_projection"]
    assert projection["mode"] == "mixed_lineage_projection"
    assert projection["parser_lineage"] == [
        LEGACY_PARSER_SHA256,
        target_parser_script_sha256(),
    ]
    assert projection["selection_policy"] == (
        "stored-binding-semantics-current-first-v1"
    )
    assert projection["selection_counts"] == {
        "current_audit": 1,
        "legacy_projection": 1,
    }
    records = _read_parquet_ledger(
        output / projection["ledger_artifact"],
        domain=SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
        schema=SOURCE_BINDING_PROJECTION_SCHEMA,
    )
    assert {record["mode"] for record in records} == {
        "current_audit",
        "legacy_projection",
    }
    assert {
        (record["mode"], record["change_kind"])
        for record in records
    } == {
        ("current_audit", "unchanged"),
        ("legacy_projection", "modified"),
    }
    metadata = _read_parquet_ledger(
        output / receipt["representative_metadata"]["artifact"],
        schema=receipt["representative_metadata"]["schema"],
    )
    assert {
        item["training_sidecars"]["build_actions"][0][
            "source_binding_projection"
        ]["mode"]
        for item in metadata
    } == {"current_audit", "legacy_projection"}


def test_parser_binding_rollback_preserves_generation_evidence(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "clang++ -c src/current.cpp"
    provenance = _provenance(text)
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, provenance)],
    )
    current = target_parser_script_sha256()
    rolled_back_from = "7" * 64
    with sqlite3.connect(fetch_state) as connection:
        connection.executemany(
            """
            INSERT INTO binding_upgrades(
              binding_key,from_sha256,to_sha256,reason,upgraded_at
            ) VALUES (?,?,?,?,?)
            """,
            (
                (
                    "parser_script_sha256",
                    current,
                    rolled_back_from,
                    "fixture upgrade",
                    "2026-07-26T22:00:00Z",
                ),
                (
                    "parser_script_sha256",
                    rolled_back_from,
                    current,
                    "fixture rollback",
                    "2026-07-26T23:00:00Z",
                ),
            ),
        )
        connection.commit()

    output = tmp_path / "rollback-export"
    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )

    projection = receipt["source_binding_projection"]
    assert projection["mode"] == "mixed_lineage_projection"
    assert projection["parser_lineage"] == [rolled_back_from, current]

    refused = tmp_path / "rollback-current-only-refused"
    with pytest.raises(
        ExportError,
        match="current-parser-only export requires exactly one parser generation",
    ):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=refused,
            require_current_parser_only=True,
        )
    assert not refused.exists()


def test_reviewed_primary_equivalent_parser_transition_is_receipt_bound(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "clang++ -c src/current.cpp"
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
    )
    upgrade = {
        "binding_key": "parser_script_sha256",
        "from_sha256": REVIEWED_PRIMARY_EQUIVALENT_PARSER_SHA256,
        "to_sha256": target_parser_script_sha256(),
        "reason": REVIEWED_PRIMARY_EQUIVALENT_PARSER_UPGRADE_REASON,
        "upgraded_at": "2026-07-30T13:38:03Z",
    }
    with sqlite3.connect(fetch_state) as connection:
        connection.execute(
            """
            INSERT INTO binding_upgrades(
              binding_key,from_sha256,to_sha256,reason,upgraded_at
            ) VALUES (:binding_key,:from_sha256,:to_sha256,:reason,:upgraded_at)
            """,
            upgrade,
        )
        connection.commit()

    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=tmp_path / "reviewed-primary-equivalent",
    )

    assert receipt["parser_generation_policy"] == {
        "mode": "reviewed-primary-equivalent-transition",
        "expected_current_parser_script_sha256": (
            target_parser_script_sha256()
        ),
        "observed_parser_lineage": [
            REVIEWED_PRIMARY_EQUIVALENT_PARSER_SHA256,
            target_parser_script_sha256(),
        ],
        "current_singleton": False,
    }
    assert receipt["input_fetch_state"]["summary"]["binding_upgrades"] == [
        upgrade
    ]
    assert receipt["source_binding_projection"]["mode"] == (
        "mixed_lineage_projection"
    )
    assert receipt["source_binding_projection"]["selection_counts"] == {}


def test_parser_binding_history_disconnected_from_current_fails_closed(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "clang++ -c src/current.cpp"
    provenance = _provenance(text)
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, provenance)],
    )
    with sqlite3.connect(fetch_state) as connection:
        connection.execute(
            """
            INSERT INTO binding_upgrades(
              binding_key,from_sha256,to_sha256,reason,upgraded_at
            ) VALUES (?,?,?,?,?)
            """,
            (
                "parser_script_sha256",
                "6" * 64,
                "7" * 64,
                "disconnected fixture",
                "2026-07-26T23:00:00Z",
            ),
        )
        connection.commit()

    output = tmp_path / "disconnected-export"
    with pytest.raises(ExportError, match="cannot return to the current parser"):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=output,
        )
    assert not output.exists()


def test_projection_writer_rejects_rows_the_consumer_cannot_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source_binding_projection.parquet"
    writer = _source_binding_projection_writer(path)
    try:
        with pytest.raises(
            CanonicalParquetLedgerError,
            match="record exceeds",
        ):
            writer.append(
                {
                    "source_input": (
                        "x" * MAX_SOURCE_BINDING_PROJECTION_RECORD_BYTES
                    )
                }
            )
    finally:
        writer.close()

    assert writer.count == 0
    assert pq.ParquetFile(path).metadata.num_rows == 0


def test_split_contract_declares_and_uses_the_exact_hash_projection() -> None:
    assert _split_for_sequence(f"{9799:016x}" + "0" * 48) == "train"
    assert _split_for_sequence(f"{9800:016x}" + "0" * 48) == "validation"
    assert _split_for_sequence(f"{9900:016x}" + "0" * 48) == "test"


def test_atomic_publish_never_replaces_an_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "new").write_text("new", encoding="utf-8")
    (destination / "existing").write_text("existing", encoding="utf-8")

    with pytest.raises(ExportError, match="output appeared"):
        _publish_directory_no_replace(source, destination)

    assert (source / "new").read_text(encoding="utf-8") == "new"
    assert (destination / "existing").read_text(encoding="utf-8") == "existing"


def test_frozen_store_rejects_index_replaced_after_immutable_open(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "immutable inode binding"
    store_root, receipt_path, _fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
    )
    frozen = FrozenStore(store_root, receipt_path)
    replacement = tmp_path / "replacement-index.sqlite3"
    shutil.copy2(store_root / "index.sqlite3", replacement)
    replacement.replace(store_root / "index.sqlite3")

    with pytest.raises(ExportError, match="immutable connection"):
        frozen.__enter__()


def test_fetch_state_sidecar_rejects_trailing_compressed_bytes(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "strict sidecar decompression"
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
    )
    with sqlite3.connect(fetch_state) as connection:
        compressed = bytes(
            connection.execute("SELECT sidecar_zlib FROM members").fetchone()[0]
        )
        connection.execute(
            "UPDATE members SET sidecar_zlib=?",
            (sqlite3.Binary(compressed + b"trailing"),),
        )
    output = tmp_path / "invalid-sidecar"

    with pytest.raises(ExportError, match="zlib stream is not exact"):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=output,
        )
    assert not output.exists()


def test_exporter_rejects_jobs_zlib_bomb_before_json_decode(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "bounded jobs evidence"
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
    )
    raw_size = MAX_JOBS_EVIDENCE_BYTES + 1
    compressed, digest = _compressed_repetition(raw_size)
    with sqlite3.connect(fetch_state) as connection:
        connection.execute(
            """
            UPDATE attempts SET
              jobs_raw_size=?,jobs_sha256=?,jobs_zlib=?
            """,
            (raw_size, digest, sqlite3.Binary(compressed)),
        )
    with pytest.raises(ExportError, match="not exact and bounded"):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=tmp_path / "jobs-bomb",
        )


def test_exporter_rejects_provenance_zlib_bomb_before_json_decode() -> None:
    raw_size = MAX_STATE_JSON_EVIDENCE_BYTES + 1
    compressed, digest = _compressed_repetition(raw_size)
    with pytest.raises(ExportError, match="not exact and bounded"):
        _decode_provenance(
            {
                "provenance_raw_size": raw_size,
                "provenance_sha256": digest,
                "provenance_zlib": compressed,
            }
        )


def test_exporter_rejects_member_sidecar_zlib_bomb(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "bounded member sidecar evidence"
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
    )
    raw_size = MAX_STATE_JSON_EVIDENCE_BYTES + 1
    compressed, digest = _compressed_repetition(raw_size)
    with sqlite3.connect(fetch_state) as connection:
        connection.execute(
            """
            UPDATE members SET
              sidecar_raw_size=?,sidecar_sha256=?,sidecar_zlib=?
            """,
            (raw_size, digest, sqlite3.Binary(compressed)),
        )
    with pytest.raises(ExportError, match="not exact and bounded"):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=tmp_path / "sidecar-bomb",
        )


def test_exporter_content_frame_cap_precedes_payload_pread(
    tmp_path: Path,
) -> None:
    raw_size = MAX_CONTENT_FRAME_RAW_BYTES + 1
    compressed, digest = _compressed_repetition(raw_size)
    pack = tmp_path / "pack-00000001.cicp"
    pack.write_bytes(
        _PACK_MAGIC
        + _FRAME_HEADER.pack(
            _FRAME_MAGIC,
            bytes.fromhex(digest),
            raw_size,
            len(compressed),
        )
        + compressed
    )
    frozen = object.__new__(FrozenStore)
    frozen.root = tmp_path
    frozen.pack_paths = {1: pack}
    frozen._pack_fds = OrderedDict()
    row = {
        "pack_id": 1,
        "offset": len(_PACK_MAGIC),
        "frame_size": _FRAME_HEADER.size + len(compressed),
        "compressed_size": len(compressed),
        "raw_size": raw_size,
        "sha256": digest,
    }
    try:
        with pytest.raises(ExportError, match="byte bound"):
            frozen.read_content_row(row)
    finally:
        for descriptor in frozen._pack_fds.values():
            os.close(descriptor)


def test_fetch_state_done_attempt_requires_exact_jobs_payload(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "missing jobs payload"
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
    )
    with sqlite3.connect(fetch_state) as connection:
        connection.execute(
            """
            UPDATE attempts
            SET jobs_sha256=NULL,jobs_raw_size=NULL,jobs_zlib=NULL
            """
        )
    output = tmp_path / "missing-jobs"

    with pytest.raises(ExportError, match="exact jobs evidence is missing"):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=output,
        )
    assert not output.exists()


def test_fetch_state_rejects_job_payload_contradicting_occurrence(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "contradictory jobs payload"
    provenance = _provenance(text)
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, provenance)],
    )
    with sqlite3.connect(fetch_state) as connection:
        raw = zlib.decompress(
            bytes(connection.execute("SELECT jobs_zlib FROM attempts").fetchone()[0])
        )
        jobs = json.loads(raw)
        jobs[0]["runner_name"] = "contradictory runner"
        _replace_attempt_jobs(connection, jobs)
    output = tmp_path / "contradictory-jobs"

    with pytest.raises(
        ExportError,
        match="occurrence job/member/sidecar binding differs",
    ):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=output,
        )
    assert not output.exists()


def test_fetch_state_rejects_duplicate_job_ids(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "duplicate jobs"
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
    )
    with sqlite3.connect(fetch_state) as connection:
        raw = zlib.decompress(
            bytes(connection.execute("SELECT jobs_zlib FROM attempts").fetchone()[0])
        )
        jobs = json.loads(raw)
        jobs.append({**jobs[0], "name": "duplicate"})
        _replace_attempt_jobs(connection, jobs)
    output = tmp_path / "duplicate-jobs"

    with pytest.raises(ExportError, match="jobs contain duplicate id"):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=output,
        )
    assert not output.exists()


def test_fetch_state_replays_job_ordinal_fallback(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "ordinal job fallback"
    provenance = _provenance(text, archive_member="0_unmatched-hint.txt")
    provenance["job"] = {
        "id": 99,
        "name": "actual API job name",
        "runner_name": "GitHub Actions 99",
    }
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, provenance)],
    )
    output = tmp_path / "ordinal-job-fallback"

    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )

    assert receipt["status"] == "complete"
    assert receipt["provenance_evidence"][
        "jobs_payload_sha256_recomputed_from_fetch_state"
    ]


def test_fetch_state_rejects_malformed_metadata_on_zero_member_empty_attempt(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "done attempt beside malformed empty attempt"
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
    )
    malformed = b"[]"
    malformed_sha256 = hashlib.sha256(malformed).hexdigest()
    with sqlite3.connect(fetch_state) as connection:
        _clone_attempt(
            connection,
            run_id=101,
            status="empty",
            member_count=0,
            chunk_count=0,
            occurrence_tokens=0,
        )
        connection.execute(
            """
            UPDATE attempts
            SET run_metadata_sha256=?,
                run_metadata_raw_size=?,
                run_metadata_zlib=?,
                inventory_seed_metadata_sha256=?
            WHERE run_id=101
            """,
            (
                malformed_sha256,
                len(malformed),
                sqlite3.Binary(zlib.compress(malformed)),
                malformed_sha256,
            ),
        )
    output = tmp_path / "malformed-empty-attempt"

    with pytest.raises(ExportError, match="not one canonical JSON object"):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=output,
        )
    assert not output.exists()


def test_fetch_state_accepts_empty_attempt_with_zero_chunk_member(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "done attempt"
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
    )
    with sqlite3.connect(fetch_state) as connection:
        connection.row_factory = sqlite3.Row
        _clone_attempt(
            connection,
            run_id=101,
            status="empty",
            member_count=1,
            chunk_count=0,
            occurrence_tokens=0,
        )

        member = dict(connection.execute("SELECT * FROM members").fetchone())
        sidecar = json.loads(zlib.decompress(bytes(member["sidecar_zlib"])))
        sidecar["chunk_index"] = []
        sidecar["conservation"]["chunk_count"] = 0
        sidecar_without_hash = dict(sidecar)
        del sidecar_without_hash["sidecar_sha256"]
        sidecar["sidecar_sha256"] = hashlib.sha256(
            _canonical_bytes(sidecar_without_hash)
        ).hexdigest()
        sidecar_raw = _canonical_bytes(sidecar)
        member.update(
            {
                "run_id": 101,
                "sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
                "sidecar_raw_size": len(sidecar_raw),
                "sidecar_zlib": sqlite3.Binary(zlib.compress(sidecar_raw)),
                "chunk_count": 0,
                "occurrence_tokens": 0,
            }
        )
        member_columns = tuple(member)
        connection.execute(
            f"""
            INSERT INTO members({",".join(member_columns)})
            VALUES ({",".join("?" for _column in member_columns)})
            """,
            tuple(member[column] for column in member_columns),
        )
    output = tmp_path / "empty-zero-chunk-member"

    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )

    assert receipt["status"] == "complete"
    assert receipt["input_fetch_state"]["summary"]["attempt_statuses"] == {
        "done": 1,
        "empty": 1,
    }


@pytest.mark.parametrize(
    "counter",
    ["member_count", "chunk_count", "occurrence_tokens"],
)
def test_fetch_state_rejects_compensating_per_attempt_accounting(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
    counter: str,
) -> None:
    first = _provenance("first attempt", archive_member="first.txt")
    second = _provenance("second attempt", archive_member="second.txt")
    second["run_id"] = 101
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [("first attempt", first), ("second attempt", second)],
    )
    with sqlite3.connect(fetch_state) as connection:
        rows = connection.execute(
            f"SELECT run_id,{counter} FROM attempts ORDER BY run_id"
        ).fetchall()
        assert len(rows) == 2
        assert int(rows[0][1]) > 0
        assert int(rows[1][1]) > 0
        connection.execute(
            f"UPDATE attempts SET {counter}=? WHERE run_id=?",
            (int(rows[0][1]) + int(rows[1][1]), int(rows[0][0])),
        )
        connection.execute(
            f"UPDATE attempts SET {counter}=0 WHERE run_id=?",
            (int(rows[1][0]),),
        )
    output = tmp_path / f"compensating-{counter}"

    with pytest.raises(
        ExportError,
        match="per-attempt member accounting is inconsistent",
    ):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=output,
        )
    assert not output.exists()


def test_reencoding_mismatch_fails_without_publishing(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "token metadata must be independently verified"
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, _provenance(text))],
        wrong_token_sequence=True,
    )
    output = tmp_path / "refused"

    with pytest.raises(ExportError, match="exact token metadata mismatch"):
        export_store(
            store_root=store_root,
            store_receipt=receipt_path,
            fetch_state=fetch_state,
            tokenizer_json=TOKENIZER_JSON,
            output=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "failure",
    ["missing", "bad-rle", "bad-edge", "unresolved-cross-edge"],
)
def test_missing_or_malformed_v2_training_sidecars_fail_closed(
    failure: str,
) -> None:
    text = "ab"
    entities = [
        _entity(
            "entity:000000",
            0,
            1,
            domain=DomainKind.BASH,
            role=int(DomainRoleKind.COMMAND),
        )
    ]
    provenance = _provenance(text, entities=entities)
    training = provenance["chunk"]["training_sidecars"]
    if failure == "missing":
        del provenance["chunk"]["training_sidecars"]
        expected = "training_sidecars"
    elif failure == "bad-rle":
        provenance["chunk"]["domain_spans"][0]["start_char"] = 1
        expected = "contiguous RLE"
    elif failure == "bad-edge":
        training["edges"] = [
            {
                "edge_id": "edge:000000",
                "source": "entity:000000",
                "target": "entity:missing",
                "from_char": 0,
                "to_char": 1,
                "kind": "BUILD_ACTION_INPUT",
                "kind_id": 23,
                "family": "build",
            }
        ]
        expected = "non-local entity endpoint"
    else:
        training["cross_chunk_edge_accounting"]["count"] = 1
        expected = "omits non-outbound"

    with pytest.raises(ExportError, match=expected):
        _validate_occurrence_v3(_occurrence(provenance), content_text=text)


def test_v3_occurrence_requires_exact_attempt_evidence() -> None:
    text = "exact attempt"
    provenance = _provenance(text)
    provenance["run_metadata_evidence"]["source_attempt"] = 2

    with pytest.raises(ExportError, match="attempt evidence"):
        _validate_occurrence_v3(_occurrence(provenance), content_text=text)


def test_attempt_api_evidence_requires_a_newer_inventory_seed() -> None:
    text = "attempt api"
    provenance = _provenance(text)
    evidence = provenance["run_metadata_evidence"]
    evidence["source"] = "github-workflow-run-attempt-api"
    evidence["inventory_seed_metadata_sha256"] = "9" * 64

    with pytest.raises(ExportError, match="newer inventory seed"):
        _validate_occurrence_v3(_occurrence(provenance), content_text=text)


def test_v3_occurrence_rejects_contradictory_workflow_projection() -> None:
    text = "exact workflow"
    provenance = _provenance(text)
    provenance["workflow"]["head_commit"]["id"] = "b" * 40

    with pytest.raises(ExportError, match="head_commit.id"):
        _validate_occurrence_v3(_occurrence(provenance), content_text=text)


def test_cross_chunk_accounting_uses_canonical_member_coordinates() -> None:
    text = "ab"
    provenance = _provenance(
        text,
        entities=[
            _entity(
                "entity:000000",
                0,
                1,
                domain=DomainKind.BASH,
                role=int(DomainRoleKind.SOURCE),
            )
        ],
        cross_chunk_edges=[
            {
                "edge_id": "edge:cross",
                "source": "entity:000000",
                "target": "entity:external",
                "from_char": 0,
                "to_member_char": 100,
                "target_coordinate_space": "canonical_member_chars_v1",
                "kind": "BUILD_ACTION_INPUT",
                "kind_id": 23,
                "family": "build",
            }
        ],
    )
    chunk = provenance["chunk"]
    chunk.update(
        char_start=50,
        char_end=52,
        dedup_char_start=50,
        dedup_char_end=52,
    )
    provenance["section"]["char_end"] = 52
    accounting = chunk["training_sidecars"]["cross_chunk_edge_accounting"]
    accounting["sha256"] = _sequence_digest(
        [
            {
                "edge_id": "edge:cross",
                "kind_id": 23,
                "from_char": 50,
                "to_char": 100,
            }
        ]
    )

    _validate_occurrence_v3(_occurrence(provenance), content_text=text)

    accounting["sha256"] = _sequence_digest(
        [
            {
                "edge_id": "edge:cross",
                "kind_id": 23,
                "from_char": 0,
                "to_char": 100,
            }
        ]
    )
    with pytest.raises(ExportError, match="accounting digest"):
        _validate_occurrence_v3(_occurrence(provenance), content_text=text)


def test_unicode_and_collapsed_whitespace_use_original_offsets(
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "α  \n\nβ\tγ"
    beta = text.index("β")
    gamma = text.index("γ")
    entities = [
        _entity(
            "entity:000000",
            0,
            1,
            domain=DomainKind.BASH,
            role=int(DomainRoleKind.COMMAND),
        ),
        _entity(
            "entity:000001",
            beta,
            beta + 1,
            domain=DomainKind.BASH,
            role=int(DomainRoleKind.TARGET),
        ),
    ]
    edge = {
        "edge_id": "edge:000000",
        "source": "entity:000000",
        "target": "entity:000001",
        "from_char": 0,
        "to_char": beta,
        "kind": "BUILD_COMMAND_TARGET",
        "kind_id": 26,
        "family": "build",
    }
    provenance = _provenance(
        text,
        domains=[
            {
                "start_char": 0,
                "end_char": gamma,
                "domain_id": int(DomainKind.BASH),
                "confidence": 1.0,
            },
            {
                "start_char": gamma,
                "end_char": len(text),
                "domain_id": int(DomainKind.TEST_OUTPUT),
                "confidence": 0.8,
            },
        ],
        entities=entities,
        edges=[edge],
    )
    occurrence = _occurrence(provenance)
    chunk = _validate_occurrence_v3(occurrence, content_text=text)
    projected = _project_content(
        tokenizer=exact_tokenizer,
        text=text,
        chunk=chunk,
    )

    expected_ids, expected_spans = exact_tokenizer._tokenizer.encode_with_offsets(text)
    assert projected.token_ids == expected_ids
    assert projected.token_spans == expected_spans
    assert len(projected.token_domain_ids) == len(expected_ids)
    assert projected.edges[0]["from"] != projected.edges[0]["to"]
    target_token = projected.edges[0]["to"]
    start, end = projected.token_spans[target_token]
    assert start <= beta < end
    assert any(start <= beta < end for start, end in projected.token_spans)


def test_over_16k_mixed_domain_fragments_are_balanced_and_conserved(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    text = "alpha  \n\n" * 6500
    split_char = len(text) // 2
    domains = [
        {
            "start_char": 0,
            "end_char": split_char,
            "domain_id": int(DomainKind.BASH),
            "confidence": 1.0,
        },
        {
            "start_char": split_char,
            "end_char": len(text),
            "domain_id": int(DomainKind.TEST_OUTPUT),
            "confidence": 0.8,
        },
    ]
    final_alpha = text.rfind("alpha")
    entities = [
        _entity(
            "entity:000000",
            0,
            len("alpha"),
            domain=DomainKind.BASH,
            role=int(DomainRoleKind.COMMAND),
        ),
        _entity(
            "entity:000001",
            final_alpha,
            final_alpha + len("alpha"),
            domain=DomainKind.TEST_OUTPUT,
            role=int(DomainRoleKind.TARGET),
        ),
        _entity(
            "entity:000002",
            final_alpha,
            final_alpha + len("alpha"),
            domain=DomainKind.CPP,
            role=int(DomainRoleKind.PATH),
        ),
    ]
    provenance = _provenance(
        text,
        domains=domains,
        entities=entities,
        edges=[
            {
                "edge_id": "edge:000000",
                "source": "entity:000000",
                "target": "entity:000001",
                "from_char": 0,
                "to_char": final_alpha,
                "kind": "BUILD_COMMAND_TARGET",
                "kind_id": 26,
                "family": "build",
            },
            {
                "edge_id": "edge:000001",
                "source": "entity:000000",
                "target": "entity:000000",
                "from_char": 0,
                "to_char": 0,
                "kind": "BUILD_ACTION_INPUT",
                "kind_id": 23,
                "family": "build",
            },
        ],
        cross_chunk_edges=[
            {
                "edge_id": "edge:cross",
                "source": "entity:000000",
                "target": "entity:external",
                "from_char": 0,
                "to_member_char": len(text) + 10,
                "target_coordinate_space": "canonical_member_chars_v1",
                "kind": "BUILD_ACTION_INPUT",
                "kind_id": 23,
                "family": "build",
            }
        ],
    )
    payload_ids = exact_tokenizer.encode_batch([text])[0]
    assert len(payload_ids) > BUCKETS[-1]
    store_root, receipt_path, fetch_state = _build_store(
        tmp_path,
        exact_tokenizer,
        [(text, provenance)],
    )
    output = tmp_path / "large-case5"

    receipt = export_store(
        store_root=store_root,
        store_receipt=receipt_path,
        fetch_state=fetch_state,
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )

    fragments = _read_parquet_ledger(
        output / receipt["fragment_ledger"]["artifact"],
    )
    assert len(fragments) > 1
    assert sum(item["payload_tokens"] for item in fragments) == len(payload_ids)
    assert [(item["payload_start"], item["payload_end"]) for item in fragments] == [
        (start, end)
        for start, end in zip(
            [0, *(item["payload_end"] for item in fragments[:-1])],
            [item["payload_end"] for item in fragments],
            strict=True,
        )
    ]
    assert fragments[-1]["payload_end"] == len(payload_ids)
    assert len({item["split"] for item in fragments}) == 1
    assert all(
        item["bucket"] == _smallest_bucket(item["valid_tokens"]) for item in fragments
    )
    assert receipt["case5_contract"]["overflow_rows"] == 0
    assert receipt["validation"]["payload_conserved"] is True
    assert receipt["counts"]["capacity_tokens"] == (
        receipt["counts"]["valid_tokens"] + receipt["counts"]["padding_tokens"]
    )
    assert receipt["graph_accounting"]["input_in_chunk_edges"] == 2
    assert receipt["graph_accounting"]["emitted_edges"] == 1
    assert receipt["graph_accounting"]["cross_fragment_edges_dropped"] == 1
    assert (
        receipt["graph_accounting"]["cross_chunk_reference_count_source_reported"] == 1
    )
    assert (
        receipt["graph_accounting"]["cross_chunk_outbound_reference_count_validated"]
        == 1
    )
    assert (
        receipt["graph_accounting"][
            "cross_chunk_non_outbound_reference_count_unresolved"
        ]
        == 0
    )
    assert receipt["graph_accounting"]["cross_chunk_outbound_edges_dropped"] == 1

    rows_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for artifact in receipt["artifacts"]:
        if artifact["kind"] != "case5_parquet":
            continue
        table = pq.read_table(output / artifact["path"])
        for row in table.to_pylist():
            _assert_balanced_case5_row(row, bucket=int(artifact["bucket"]))
            rows_by_key[
                (
                    str(artifact["split"]),
                    int(artifact["bucket"]),
                    int(row["pack_id"]),
                )
            ] = row

    reconstructed_payload: list[int] = []
    for fragment in fragments:
        row = rows_by_key[
            (
                str(fragment["split"]),
                int(fragment["bucket"]),
                int(fragment["row_index_within_split_bucket"]),
            )
        ]
        valid = int(row["valid_token_count"])
        reconstructed_payload.extend(
            int(token_id)
            for token_id, role_id in zip(
                row["input_ids"][1:valid],
                row["token_role_ids"][1:valid],
                strict=True,
            )
            if int(role_id) != int(DomainRoleKind.DELIMITER)
        )
    assert reconstructed_payload == payload_ids
    assert sum(len(row["token_build_edges"]) for row in rows_by_key.values()) == 1

    projected = _project_content(
        tokenizer=exact_tokenizer,
        text=text,
        chunk=provenance["chunk"],
    )
    assert _fragment_ranges(projected.token_domain_ids) == [
        (item["payload_start"], item["payload_end"]) for item in fragments
    ]
    bash_start, bash_end = delimiter_token_ids(DomainKind.BASH)
    test_start, test_end = delimiter_token_ids(DomainKind.TEST_OUTPUT)
    delimiter_ids = {bash_start, bash_end, test_start, test_end}
    for artifact in receipt["artifacts"]:
        if artifact["kind"] != "case5_parquet":
            continue
        for row in pq.read_table(output / artifact["path"]).to_pylist():
            valid = int(row["valid_token_count"])
            assert not delimiter_ids.intersection(row["input_ids"][valid:])
