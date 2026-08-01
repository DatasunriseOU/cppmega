#!/usr/bin/env python3
"""Publish one current, versioned training-data inventory and changelog.

The report deliberately keeps four states separate:

* packed Parquet that is structurally readable but not yet sealed;
* sealed Megatron ``.bin/.idx`` bundles;
* staged source stores (PR SQLite and CI CAS);
* legacy or validation-only artifacts that must not enter production totals.

It never adds overlapping snapshots or store-local CI dedup counters.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq


STATUS_SCHEMA = "cppmega_training_data_status_v1"
CONFIG_SCHEMA = "cppmega_training_data_status_config_v1"
CHANGELOG_SCHEMA = "cppmega_training_data_status_change_v1"
HEARTBEAT_SCHEMA = "cppmega_training_data_status_heartbeat_v1"
DEFAULT_STALE_MINUTES = 30.0
TARGET_LENGTHS = (1024, 2048, 4096, 8192, 16384)
REQUIRED_COUNTER_COLUMNS = (
    "valid_token_count",
    "trained_token_count",
    "num_docs",
)
SOURCE_CLASSIFICATION_COLUMNS = (
    "source_doc_types",
    "source_build_kinds",
    "source_doc_token_lengths",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_source_completion_summary(path: Path) -> dict[str, Any]:
    """Read only the bounded fields needed from the large conveyor receipt."""
    query = """
    {
      code_revision: .code_revision,
      source_repo_list: .source_repo_list,
      done: (
        .done
        | with_entries(
            .value = {
              source: .value.source,
              lengths: .value.lengths
            }
          )
      ),
      failed_count: (.failed | length)
    }
    """
    try:
        result = subprocess.run(
            ["jq", "-c", query, str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "jq is required to read the large source completion receipt "
            "without unbounded Python memory"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{path}: jq receipt projection failed: {exc.stderr}") from exc
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: projected completion receipt is not an object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_keys(
    value: Mapping[str, object],
    required: Iterable[str],
    *,
    where: str,
) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise ValueError(f"{where}: missing required fields: {missing}")


def _batch_stats(rows: int, length: int, batch_size: int) -> dict[str, int]:
    full_batches, remainder_rows = divmod(rows, batch_size)
    return {
        "batch_size": batch_size,
        "tensor_tokens_per_full_batch": batch_size * length,
        "full_batches": full_batches,
        "remainder_rows": remainder_rows,
        "full_batch_tensor_capacity_tokens": full_batches * batch_size * length,
    }


def _parquet_paths(root: Path) -> list[tuple[int, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Parquet root does not exist: {root}")
    result: list[tuple[int, Path]] = []
    for length_dir in sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    ):
        length = int(length_dir.name)
        if length not in TARGET_LENGTHS:
            continue
        result.extend((length, path) for path in sorted(length_dir.glob("*.parquet")))
    return result


def _path_inventory(
    root: Path,
    paths: Iterable[tuple[int, Path]],
) -> list[dict[str, object]]:
    inventory = []
    for length, path in paths:
        stat = path.stat()
        inventory.append(
            {
                "bucket": length,
                "path": str(path.relative_to(root)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return inventory


def _schema_descriptor(schema: object) -> dict[str, object]:
    arrow_schema = schema
    metadata = getattr(arrow_schema, "metadata", None) or {}
    return {
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in arrow_schema
        ],
        "metadata_hex": {
            key.hex(): value.hex() for key, value in sorted(metadata.items())
        },
    }


def _decoded_schema_metadata(schema: object) -> dict[str, str]:
    metadata = getattr(schema, "metadata", None) or {}
    return {
        key.decode("utf-8", errors="strict"): value.decode(
            "utf-8", errors="strict"
        )
        for key, value in sorted(metadata.items())
    }


def _empty_token_counts() -> dict[str, int]:
    return {"documents": 0, "valid_tokens": 0, "trained_tokens": 0}


def _add_token_counts(
    target: dict[str, int],
    *,
    documents: int,
    valid_tokens: int,
    trained_tokens: int,
) -> None:
    target["documents"] += documents
    target["valid_tokens"] += valid_tokens
    target["trained_tokens"] += trained_tokens


def _classify_source_document(
    doc_type: object,
    build_kind: object,
) -> tuple[str, str]:
    doc = "" if doc_type is None else str(doc_type).casefold()
    kind = "" if build_kind is None else str(build_kind).casefold()
    if kind == "python":
        return "python_aux", kind
    if doc == "code_header":
        return "c_cpp_headers", "code_header"
    if doc == "code" and not kind:
        return "c_cpp_source", "code"
    if doc == "build":
        return "build", kind or "unknown"
    if doc == "shell":
        return ("bash" if kind == "bash" else "shell_other"), kind or "unknown"
    if doc == "diagnostic":
        return "diagnostic", kind or "unknown"
    if doc == "sql":
        return "sql", kind or "sql"
    return "excluded_other", f"{doc or 'null'}:{kind or 'null'}"


def _scan_parquet_file(
    *,
    root: Path,
    length: int,
    path: Path,
    classify_documents: bool,
) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    names = set(schema.names)
    missing = sorted(set(REQUIRED_COUNTER_COLUMNS) - names)
    if missing:
        raise ValueError(f"{path}: missing counter columns {missing}")

    columns = list(REQUIRED_COUNTER_COLUMNS)
    can_classify = classify_documents and set(SOURCE_CLASSIFICATION_COLUMNS) <= names
    if can_classify:
        columns.extend(SOURCE_CLASSIFICATION_COLUMNS)
    rows = 0
    valid_tokens = 0
    trained_tokens = 0
    documents = 0
    categories: dict[str, dict[str, int]] = defaultdict(_empty_token_counts)
    subkinds: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(_empty_token_counts)
    )
    classified_documents = 0
    classified_valid_tokens = 0
    classified_trained_tokens = 0
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        rows += batch.num_rows

        def column(name: str) -> object:
            return batch.column(batch.schema.get_field_index(name))

        valid_tokens += int(
            column("valid_token_count").to_numpy(zero_copy_only=False).sum()
        )
        trained_tokens += int(
            column("trained_token_count").to_numpy(zero_copy_only=False).sum()
        )
        documents += int(column("num_docs").to_numpy(zero_copy_only=False).sum())
        if not can_classify:
            continue
        doc_types = column("source_doc_types").to_pylist()
        build_kinds = column("source_build_kinds").to_pylist()
        token_lengths = column("source_doc_token_lengths").to_pylist()
        for row_types, row_kinds, row_lengths in zip(
            doc_types, build_kinds, token_lengths, strict=True
        ):
            if len(row_types) != len(row_kinds) or len(row_types) != len(row_lengths):
                raise ValueError(f"{path}: source-document sidecars are not aligned")
            for doc_type, build_kind, raw_length in zip(
                row_types, row_kinds, row_lengths, strict=True
            ):
                token_length = int(raw_length)
                if token_length <= 0:
                    raise ValueError(
                        f"{path}: source_doc_token_lengths contains {token_length}"
                    )
                trained_length = token_length - 1
                category, subkind = _classify_source_document(doc_type, build_kind)
                _add_token_counts(
                    categories[category],
                    documents=1,
                    valid_tokens=token_length,
                    trained_tokens=trained_length,
                )
                _add_token_counts(
                    subkinds[category][subkind],
                    documents=1,
                    valid_tokens=token_length,
                    trained_tokens=trained_length,
                )
                classified_documents += 1
                classified_valid_tokens += token_length
                classified_trained_tokens += trained_length

    codecs = set()
    column_chunks = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            codecs.add(row_group.column(column_index).compression)
            column_chunks += 1

    stat = path.stat()
    descriptor = _schema_descriptor(schema)
    return {
        "bucket": length,
        "path": str(path.relative_to(root)),
        "bytes": stat.st_size,
        "rows": rows,
        "documents": documents,
        "capacity_tokens": rows * length,
        "valid_tokens": valid_tokens,
        "trained_tokens": trained_tokens,
        "schema_sha256": _sha256(descriptor),
        "schema_fields": schema.names,
        "schema_metadata": _decoded_schema_metadata(schema),
        "codecs": sorted(codecs),
        "column_chunks": column_chunks,
        "classification_available": can_classify,
        "classified_documents": classified_documents,
        "classified_valid_tokens": classified_valid_tokens,
        "classified_trained_tokens": classified_trained_tokens,
        "categories": dict(categories),
        "subkinds": {
            category: dict(values) for category, values in subkinds.items()
        },
    }


def _merge_nested_counts(
    target: dict[str, dict[str, int]],
    source: Mapping[str, Mapping[str, int]],
) -> None:
    for key, counts in source.items():
        destination = target.setdefault(key, _empty_token_counts())
        _add_token_counts(
            destination,
            documents=int(counts["documents"]),
            valid_tokens=int(counts["valid_tokens"]),
            trained_tokens=int(counts["trained_tokens"]),
        )


def _merge_subkind_counts(
    target: dict[str, dict[str, dict[str, int]]],
    source: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> None:
    for category, values in source.items():
        destination = target.setdefault(category, {})
        _merge_nested_counts(destination, values)


def _scan_parquet_snapshot_once(
    root: Path,
    *,
    batch_size: int,
    jobs: int,
    classify_documents: bool,
    allow_empty: bool,
) -> dict[str, object]:
    paths = _parquet_paths(root)
    if not paths and not allow_empty:
        raise ValueError(f"{root}: no target-length Parquet files found")
    before = _path_inventory(root, paths)
    results = []
    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as executor:
        futures = [
            executor.submit(
                _scan_parquet_file,
                root=root,
                length=length,
                path=path,
                classify_documents=classify_documents,
            )
            for length, path in paths
        ]
        for future in as_completed(futures):
            results.append(future.result())
    after_paths = _parquet_paths(root)
    after = _path_inventory(root, after_paths)
    if before != after:
        raise RuntimeError(f"{root}: Parquet inventory changed during scan")

    buckets: dict[int, dict[str, object]] = {}
    categories: dict[str, dict[str, int]] = {}
    categories_by_bucket: dict[int, dict[str, dict[str, int]]] = {}
    subkinds: dict[str, dict[str, dict[str, int]]] = {}
    schema_counts: Counter[str] = Counter()
    schema_fields: dict[str, list[str]] = {}
    schema_metadata: dict[str, dict[str, str]] = {}
    codecs: Counter[str] = Counter()
    column_chunks = 0

    for result in results:
        length = int(result["bucket"])
        bucket = buckets.setdefault(
            length,
            {
                "files": 0,
                "bytes": 0,
                "rows": 0,
                "documents": 0,
                "capacity_tokens": 0,
                "valid_tokens": 0,
                "trained_tokens": 0,
            },
        )
        for name in (
            "files",
            "bytes",
            "rows",
            "documents",
            "capacity_tokens",
            "valid_tokens",
            "trained_tokens",
        ):
            bucket[name] = int(bucket[name]) + (
                1 if name == "files" else int(result[name])
            )
        schema_hash = str(result["schema_sha256"])
        schema_counts[schema_hash] += 1
        schema_fields.setdefault(schema_hash, list(result["schema_fields"]))
        schema_metadata.setdefault(schema_hash, dict(result["schema_metadata"]))
        for codec in result["codecs"]:
            codecs[str(codec)] += int(result["column_chunks"])
        column_chunks += int(result["column_chunks"])
        _merge_nested_counts(categories, result["categories"])
        bucket_categories = categories_by_bucket.setdefault(length, {})
        _merge_nested_counts(bucket_categories, result["categories"])
        _merge_subkind_counts(subkinds, result["subkinds"])

    totals = {
        name: sum(int(bucket[name]) for bucket in buckets.values())
        for name in (
            "files",
            "bytes",
            "rows",
            "documents",
            "capacity_tokens",
            "valid_tokens",
            "trained_tokens",
        )
    }
    totals["pad_tokens"] = totals["capacity_tokens"] - totals["valid_tokens"]
    for length, bucket in buckets.items():
        bucket["pad_tokens"] = int(bucket["capacity_tokens"]) - int(
            bucket["valid_tokens"]
        )
        bucket["pad_fraction"] = (
            int(bucket["pad_tokens"]) / int(bucket["capacity_tokens"])
            if int(bucket["capacity_tokens"])
            else 0.0
        )
        bucket["batch"] = _batch_stats(int(bucket["rows"]), length, batch_size)

    classified = {
        name: sum(int(counts[name]) for counts in categories.values())
        for name in ("documents", "valid_tokens", "trained_tokens")
    }
    classification_conserved = (
        not classify_documents
        or (
            classified["documents"] == totals["documents"]
            and classified["valid_tokens"] == totals["valid_tokens"]
            and classified["trained_tokens"] == totals["trained_tokens"]
        )
    )
    return {
        "root": str(root),
        "snapshot_completed_at": _utc_now(),
        "inventory_sha256": _sha256(before),
        "files": totals["files"],
        "bytes": totals["bytes"],
        "column_chunks": column_chunks,
        "rows": totals["rows"],
        "documents": totals["documents"],
        "capacity_tokens": totals["capacity_tokens"],
        "valid_tokens": totals["valid_tokens"],
        "trained_tokens": totals["trained_tokens"],
        "pad_tokens": totals["pad_tokens"],
        "buckets": {str(key): value for key, value in sorted(buckets.items())},
        "schema": {
            "uniform": len(schema_counts) == 1,
            "counts": dict(sorted(schema_counts.items())),
            "fields_by_sha256": schema_fields,
            "metadata_by_sha256": schema_metadata,
        },
        "compression": {
            "all_zstd": set(codecs) == {"ZSTD"},
            "column_chunks_by_codec": dict(sorted(codecs.items())),
        },
        "classification": {
            "requested": classify_documents,
            "conserved": classification_conserved,
            "totals": classified,
            "by_category": dict(sorted(categories.items())),
            "by_bucket": {
                str(key): dict(sorted(value.items()))
                for key, value in sorted(categories_by_bucket.items())
            },
            "subkinds": {
                category: dict(sorted(values.items()))
                for category, values in sorted(subkinds.items())
            },
        },
    }


def scan_parquet_snapshot(
    root: Path,
    *,
    batch_size: int,
    jobs: int,
    classify_documents: bool,
    snapshot_retries: int = 2,
    allow_empty: bool = False,
) -> dict[str, object]:
    """Scan one stable physical Parquet inventory or fail loudly."""
    last_error: RuntimeError | None = None
    for _ in range(snapshot_retries + 1):
        try:
            return _scan_parquet_snapshot_once(
                root,
                batch_size=batch_size,
                jobs=jobs,
                classify_documents=classify_documents,
                allow_empty=allow_empty,
            )
        except RuntimeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _receipt_totals(done: Mapping[str, object]) -> dict[str, int]:
    result = {"rows": 0, "capacity_tokens": 0, "valid_tokens": 0}
    for entry in done.values():
        if not isinstance(entry, Mapping):
            raise ValueError("source completion done entry is not an object")
        lengths = entry.get("lengths")
        if not isinstance(lengths, Mapping):
            raise ValueError("source completion done entry has no lengths object")
        for bucket in lengths.values():
            if not isinstance(bucket, Mapping):
                raise ValueError("source completion bucket is not an object")
            for name in result:
                result[name] += int(bucket[name])
    return result


def _casefold_collisions(done: Mapping[str, object]) -> list[dict[str, object]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key in done:
        groups[key.casefold()].append(key)
    result = []
    for folded, keys in sorted(groups.items()):
        if len(keys) < 2:
            continue
        entries = []
        for key in sorted(keys):
            value = done[key]
            assert isinstance(value, Mapping)
            entries.append(
                {
                    "key": key,
                    "source": value.get("source"),
                    "totals": _receipt_totals({key: value}),
                }
            )
        result.append({"casefold_key": folded, "entries": entries})
    return result


def _logical_primary(
    categories: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    primary_categories = (
        "c_cpp_source",
        "c_cpp_headers",
        "build",
        "bash",
        "shell_other",
        "diagnostic",
        "sql",
    )
    return {
        name: sum(
            int(categories.get(category, {}).get(name, 0))
            for category in primary_categories
        )
        for name in ("documents", "valid_tokens", "trained_tokens")
    }


def collect_live_source(
    config: Mapping[str, object],
    *,
    batch_size: int,
    jobs: int,
) -> dict[str, object]:
    _require_keys(
        config,
        ("root", "completion_receipt", "launch_receipt"),
        where="source config",
    )
    root = Path(str(config["root"]))
    completion_path = Path(str(config["completion_receipt"]))
    launch_path = Path(str(config["launch_receipt"]))
    completion = _read_source_completion_summary(completion_path)
    launch = _read_json(launch_path)
    _require_keys(
        completion,
        ("code_revision", "done", "failed_count", "source_repo_list"),
        where=str(completion_path),
    )
    _require_keys(
        launch,
        ("expected_repository_count", "inputs", "outputs", "schema"),
        where=str(launch_path),
    )
    done = completion["done"]
    failed_count = int(completion["failed_count"])
    source_repo_list = completion["source_repo_list"]
    if not isinstance(done, Mapping):
        raise ValueError(f"{completion_path}: done must be an object")
    if not isinstance(source_repo_list, Mapping):
        raise ValueError(f"{completion_path}: source_repo_list must be an object")
    if launch["schema"] != "cppmega.canonical_source_launch_v1":
        raise ValueError(f"{launch_path}: unsupported source launch schema")
    launch_inputs = launch["inputs"]
    launch_outputs = launch["outputs"]
    if not isinstance(launch_inputs, Mapping) or not isinstance(
        launch_outputs, Mapping
    ):
        raise ValueError(f"{launch_path}: inputs and outputs must be objects")
    _require_keys(
        launch_inputs,
        ("archive_inventory_receipt", "repo_list"),
        where=f"{launch_path}: inputs",
    )
    _require_keys(
        launch_outputs,
        ("conveyor_manifest", "run_root"),
        where=f"{launch_path}: outputs",
    )
    archive_inventory = launch_inputs["archive_inventory_receipt"]
    launch_repo_list = launch_inputs["repo_list"]
    if not isinstance(archive_inventory, Mapping) or not isinstance(
        launch_repo_list, Mapping
    ):
        raise ValueError(f"{launch_path}: source input receipts must be objects")
    _require_keys(
        archive_inventory,
        ("repository_count",),
        where=f"{launch_path}: archive inventory receipt",
    )
    _require_keys(
        launch_repo_list,
        ("sha256",),
        where=f"{launch_path}: repo-list receipt",
    )
    _require_keys(
        source_repo_list,
        ("mapping_count", "sha256"),
        where=f"{completion_path}: source_repo_list",
    )
    expected_repository_count = int(launch["expected_repository_count"])
    if expected_repository_count <= 0:
        raise ValueError(f"{launch_path}: expected_repository_count must be positive")
    if int(archive_inventory["repository_count"]) != expected_repository_count:
        raise ValueError(f"{launch_path}: archive repository count does not match launch")
    if (
        Path(str(launch_outputs["conveyor_manifest"])).resolve()
        != completion_path.resolve()
    ):
        raise ValueError(f"{launch_path}: conveyor manifest does not match source config")
    if Path(str(launch_outputs["run_root"])).resolve() != root.resolve():
        raise ValueError(f"{launch_path}: run root does not match source config")
    if launch_repo_list["sha256"] != source_repo_list["sha256"]:
        raise ValueError(f"{launch_path}: repo-list binding does not match completion")

    parquet = scan_parquet_snapshot(
        root / "reindexed",
        batch_size=batch_size,
        jobs=jobs,
        classify_documents=True,
        allow_empty=True,
    )
    receipt_totals = _receipt_totals(done)
    physical_totals = {
        name: int(parquet[name])
        for name in ("rows", "capacity_tokens", "valid_tokens")
    }
    receipt_minus_physical = {
        name: receipt_totals[name] - physical_totals[name]
        for name in receipt_totals
    }
    collisions = _casefold_collisions(done)
    mapping_count = int(source_repo_list["mapping_count"])
    terminal_count = len(done) + failed_count
    if terminal_count > expected_repository_count:
        raise ValueError(
            f"{completion_path}: {terminal_count} terminal source units exceed "
            f"archive scope {expected_repository_count}"
        )
    categories = parquet["classification"]["by_category"]
    assert isinstance(categories, Mapping)

    blockers = []
    if any(receipt_minus_physical.values()):
        blockers.append("completion receipt totals differ from physical Parquet")
    if collisions and any(receipt_minus_physical.values()):
        blockers.append("case-folded source keys collide")
    if terminal_count < expected_repository_count:
        blockers.append("source conveyor is still incomplete")
    if failed_count:
        blockers.append("source conveyor has failed units")
    if not parquet["files"]:
        blockers.append("source conveyor has not produced Parquet yet")
    elif not parquet["schema"]["uniform"]:
        blockers.append("Parquet schemas are not uniform")
    if parquet["files"] and not parquet["compression"]["all_zstd"]:
        blockers.append("not every Parquet column chunk uses ZSTD")
    if not parquet["classification"]["conserved"]:
        blockers.append("source-document classification is not token-conserving")
    if int(categories.get("python_aux", {}).get("valid_tokens", 0)):
        blockers.append("Python auxiliary documents are still mixed into main rows")

    return {
        "state": "packed_unsealed",
        "training_readable": bool(parquet["files"]) and not any(
            blocker
            for blocker in blockers
            if blocker
            in {
                "Parquet schemas are not uniform",
                "not every Parquet column chunk uses ZSTD",
                "source-document classification is not token-conserving",
            }
        ),
        "release_ready": not blockers,
        "blockers": blockers,
        "root": str(root),
        "completion_receipt": str(completion_path),
        "version": {
            "code_revision": completion["code_revision"],
            "source_repo_list": source_repo_list,
            "source_launch_receipt": {
                "path": str(launch_path),
                "sha256": hashlib.sha256(launch_path.read_bytes()).hexdigest(),
                "expected_repository_count": expected_repository_count,
            },
            "parquet_schema": parquet["schema"],
        },
        "conveyor": {
            "mapping_count": mapping_count,
            "expected_repository_count": expected_repository_count,
            "done": len(done),
            "failed": failed_count,
            "not_terminal": expected_repository_count - terminal_count,
        },
        "receipt_totals": receipt_totals,
        "physical_totals": physical_totals,
        "receipt_minus_physical": receipt_minus_physical,
        "casefold_collisions": collisions,
        "strict_primary_logical_tokens": _logical_primary(categories),
        "strict_primary_is_separate_parquet": False,
        "parquet": parquet,
    }


def _compact_audit_counts(value: Mapping[str, object]) -> dict[str, int]:
    return {
        name: int(value[name])
        for name in (
            "files",
            "rows",
            "capacity_tokens",
            "valid_tokens",
            "trained_tokens",
            "pad_tokens",
        )
    }


def collect_megatron_bundle(
    manifest_path: Path,
    *,
    batch_size: int,
    role: str,
) -> dict[str, object]:
    manifest = _read_json(manifest_path)
    _require_keys(
        manifest,
        (
            "schema",
            "bundle_id",
            "created_at",
            "bucket_results",
            "source_snapshot",
            "audit",
            "artifacts",
            "artifact_bytes",
            "known_limitations",
        ),
        where=str(manifest_path),
    )
    audit_path = manifest_path.parent / str(manifest["audit"]["receipt"])
    audit = _read_json(audit_path)
    total = audit["total"]
    if not isinstance(total, Mapping):
        raise ValueError(f"{audit_path}: total must be an object")
    if int(manifest["audit"]["valid_tokens"]) != int(total["valid_tokens"]):
        raise ValueError(f"{manifest_path}: audit valid-token binding differs")
    if int(manifest["audit"]["trained_tokens"]) != int(total["trained_tokens"]):
        raise ValueError(f"{manifest_path}: audit trained-token binding differs")

    buckets = {}
    output_totals = {
        "files": 0,
        "rows": 0,
        "capacity_tokens": 0,
        "valid_tokens": 0,
        "trained_tokens": 0,
        "pad_tokens": 0,
    }
    dense_sidecars: set[str] = set()
    graph_sidecars: set[str] = set()
    source_platform_schemas: set[str] = set()
    for result in manifest["bucket_results"]:
        length = int(result["bucket"])
        value = result["manifest"]
        audit_bucket = audit["by_bucket"][str(length)]
        if role == "production":
            for manifest_name, audit_name in (
                ("document_count", "rows"),
                ("source_capacity_token_count", "capacity_tokens"),
                ("token_count", "valid_tokens"),
                ("trained_token_count", "trained_tokens"),
            ):
                if int(value[manifest_name]) != int(audit_bucket[audit_name]):
                    raise ValueError(
                        f"{manifest_path}: bucket {length} {manifest_name} differs "
                        "from Parquet audit"
                    )
        rows = int(value["document_count"])
        capacity_tokens = int(value["source_capacity_token_count"])
        valid_tokens = int(value["token_count"])
        trained_tokens = int(value["trained_token_count"])
        bucket_counts = {
            "files": int(audit_bucket["files"]),
            "rows": rows,
            "capacity_tokens": capacity_tokens,
            "valid_tokens": valid_tokens,
            "trained_tokens": trained_tokens,
            "pad_tokens": capacity_tokens - valid_tokens,
        }
        buckets[str(length)] = {
            **bucket_counts,
            "batch": _batch_stats(rows, length, batch_size),
        }
        for name in output_totals:
            output_totals[name] += bucket_counts[name]
        dense_sidecars.update(value["side_channel_paths"])
        graph_sidecars.update(value["graph_sidecar_paths"])
        source_platform_schemas.add(value["source_platform_sidecar"]["schema"])

    source_manifest_path = (
        manifest_path.parent / str(manifest["source_snapshot"]["manifest"])
    )
    source_manifest = _read_json(source_manifest_path)
    parquet_roots = sorted(
        {
            str(Path(str(entry["source"])).parents[1])
            for entry in source_manifest["files"]
        }
    )
    archive_receipt_path = manifest_path.parent / "archive_publish_receipt.json"
    archive_receipt = (
        _read_json(archive_receipt_path) if archive_receipt_path.is_file() else None
    )
    archive_uploaded = bool(
        archive_receipt
        and archive_receipt.get("archive", {}).get("status") == "uploaded_verified"
        and archive_receipt.get("archive_validation", {}).get("status") == "verified"
    )
    audit_clean = int(total["bad_files"]) == 0 and int(total["bad_rows"]) == 0
    release_ready = role == "production" and audit_clean and archive_uploaded
    return {
        "state": "sealed_megatron" if role == "production" else "validation_only",
        "role": role,
        "training_readable": audit_clean,
        "release_ready": release_ready,
        "blockers": (
            []
            if release_ready
            else (
                ["validation bundle is not a production corpus"]
                if role != "production"
                else ["audit or verified archive transport is incomplete"]
            )
        ),
        "manifest": str(manifest_path),
        "audit": str(audit_path),
        "archive_receipt": (
            str(archive_receipt_path) if archive_receipt_path.is_file() else None
        ),
        "version": {
            "bundle_id": manifest["bundle_id"],
            "created_at": manifest["created_at"],
            "schema": manifest["schema"],
            "tokenizer_contract": manifest.get("tokenizer_contract"),
            "implementation_sha256": manifest.get("implementation_sha256"),
            "artifact_set_sha256": manifest.get("artifact_set_sha256"),
            "git": manifest.get("git"),
        },
        "totals": output_totals,
        "source_audit_totals": _compact_audit_counts(total),
        "buckets": buckets,
        "by_kind": {
            key: _compact_audit_counts(value)
            for key, value in sorted(audit["by_kind"].items())
        },
        "source_parquet": {
            "roots": parquet_roots,
            "file_count": int(source_manifest["file_count"]),
            "by_kind_bucket": source_manifest["by_kind_bucket"],
        },
        "megatron_artifacts": {
            "writer_backend": manifest.get("writer_backend"),
            "artifact_count": int(manifest["artifact_count"]),
            "artifact_bytes": int(manifest["artifact_bytes"]),
            "local_snapshot_retained": bool(
                manifest["source_snapshot"]["local_snapshot_retained"]
            ),
            "archive_uploaded_verified": archive_uploaded,
        },
        "sidecars": {
            "dense": sorted(dense_sidecars),
            "ragged_graph": sorted(graph_sidecars),
            "source_platform_schemas": sorted(source_platform_schemas),
        },
        "known_limitations": manifest["known_limitations"],
    }


def collect_pr_status(config: Mapping[str, object]) -> dict[str, object]:
    _require_keys(
        config,
        (
            "completion_receipt",
            "gap_completion_receipt",
            "export_launch_receipt",
            "export_cancellation_receipt",
            "quarantine_receipt",
        ),
        where="PR config",
    )
    completion_path = Path(str(config["completion_receipt"]))
    gap_path = Path(str(config["gap_completion_receipt"]))
    launch_path = Path(str(config["export_launch_receipt"]))
    cancellation_path = Path(str(config["export_cancellation_receipt"]))
    quarantine_path = Path(str(config["quarantine_receipt"]))
    completion = _read_json(completion_path)
    gap = _read_json(gap_path)
    launch = _read_json(launch_path)
    cancellation = _read_json(cancellation_path)
    quarantine = _read_json(quarantine_path)
    if completion.get("status") != "verified":
        raise ValueError(f"{completion_path}: PR completion is not verified")
    if gap.get("status") != "verified":
        raise ValueError(f"{gap_path}: PR gap completion is not verified")
    output_root = Path(str(launch["output_root"]))
    eligible_parquets = (
        sorted(output_root.rglob("*.parquet")) if output_root.is_dir() else []
    )
    if eligible_parquets:
        raise ValueError(
            f"{output_root}: eligible PR Parquet exists without a configured "
            "completion receipt; update the status config before counting it"
        )
    return {
        "state": "verified_store_not_materialized",
        "training_readable": False,
        "release_ready": False,
        "blockers": ["primary PR export was cancelled before materialization"],
        "version": {
            "scan_id": completion["scan_id"],
            "schema": completion["schema"],
            "store_sha256": completion["pr_store"]["sha256"],
            "repo_list_sha256": completion["repo_list"]["sha256"],
        },
        "records": {
            "repos": int(completion["expected_repo_count"]),
            "declared_prs": int(completion["declared_pr_count"]),
            "stored_prs": int(completion["stored_pr_count"]),
            "unverified_store_prs": int(completion["unverified_store_pr_count"]),
            "completed_graphql_gaps": int(gap["completed_count"]),
            "unresolved_graphql_gaps": int(gap["unresolved_count"]),
        },
        "store": completion["pr_store"],
        "eligible_parquet": {
            "root": str(output_root),
            "files": 0,
            "rows": 0,
            "valid_tokens": 0,
            "trained_tokens": 0,
        },
        "quarantine": {
            "path": quarantine["archive"]["path"],
            "prs": int(quarantine["quarantined_pr_count"]),
            "reason": quarantine["reason"],
            "eligible": False,
        },
        "export": {
            "launch_receipt": str(launch_path),
            "cancellation_receipt": str(cancellation_path),
            "status": cancellation["status"],
            "reason": cancellation["reason"],
        },
        "record_sidecars": [
            "repo",
            "pr_number",
            "merge_commit_sha",
            "pr_title",
            "pr_body",
            "comments",
            "reviews",
            "linked_issues",
        ],
    }


def collect_ci_status(config: Mapping[str, object]) -> dict[str, object]:
    _require_keys(
        config,
        ("progress_receipts", "legacy_parquet_root"),
        where="CI config",
    )
    progress_paths = [Path(str(value)) for value in config["progress_receipts"]]
    stores = []
    store_unique_upper_bound = 0
    occurrence_tokens = 0
    for path in progress_paths:
        progress = _read_json(path)
        _require_keys(
            progress,
            (
                "schema",
                "generated_at",
                "inventory",
                "fetch",
                "content_store",
                "token_accounting",
            ),
            where=str(path),
        )
        counters = progress["content_store"]["counters"]
        unique_tokens = counters["exact_unique_payload_tokens"]
        if unique_tokens is None:
            raise ValueError(f"{path}: CI unique-token counter is not exact")
        unique_tokens = int(unique_tokens)
        store_unique_upper_bound += unique_tokens
        occurrence_tokens += int(progress["fetch"]["occurrence_tokens"])
        stores.append(
            {
                "progress": str(path),
                "generated_at": progress["generated_at"],
                "interval": progress["inventory"]["interval"],
                "inventory": {
                    "repos_closed": int(progress["inventory"]["repos_closed"]),
                    "repos_total": int(progress["inventory"]["repos_total"]),
                    "runs": int(progress["inventory"]["runs"]),
                    "expected_runs": int(
                        progress["fetch"]["exhaustive_discovery"][
                            "expected_run_count"
                        ]
                    ),
                    "expected_attempts": int(
                        progress["fetch"]["exhaustive_discovery"][
                            "expected_attempt_count"
                        ]
                    ),
                    "attempt_rows_seen": int(
                        progress["fetch"]["exhaustive_discovery"]["rows_seen"]
                    ),
                    "discovery_eof": bool(
                        progress["fetch"]["exhaustive_discovery"]["discovery_eof"]
                    ),
                },
                "attempt_statuses": progress["fetch"]["attempt_statuses"],
                "exact_unique_payload_tokens": unique_tokens,
                "occurrence_tokens": int(progress["fetch"]["occurrence_tokens"]),
                "members": int(progress["fetch"]["members"]),
                "chunks": int(progress["fetch"]["chunks"]),
                "sidecar_set_sha256": progress["fetch"]["sidecar_set_sha256"],
                "parser_binding_upgrades": progress["fetch"].get(
                    "binding_upgrades", []
                ),
                "tokenizer": progress["token_accounting"]["tokenizer_contract"],
                "content_store": {
                    "schema": progress["content_store"]["schema"],
                    "compression": progress["content_store"]["policy"][
                        "compression"
                    ],
                    "pack_count": int(progress["content_store"]["pack_count"]),
                    "committed_pack_bytes": int(
                        progress["content_store"]["committed_pack_bytes"]
                    ),
                    "quarantined_orphans": int(
                        progress["content_store"]["quarantined_orphan_count"]
                    ),
                },
            }
        )

    legacy_root = Path(str(config["legacy_parquet_root"]))
    legacy = scan_parquet_snapshot(
        legacy_root,
        batch_size=int(config.get("batch_size", 192)),
        jobs=int(config.get("jobs", 8)),
        classify_documents=True,
    )
    legacy_manifest = _read_json(legacy_root / "manifest.json")
    return {
        "state": "cas_staged_not_exported",
        "training_readable": False,
        "release_ready": False,
        "blockers": [
            "inventory fetch is not exhaustive",
            "cross-store canonical union/global dedup has not run",
            "primary-scope five-bucket Parquet export has not run",
        ],
        "stores": stores,
        "token_accounting": {
            "store_local_unique_upper_bound": store_unique_upper_bound,
            "global_unique_payload_tokens": None,
            "global_unique_reason": (
                "store-local exact counters may overlap across intervals"
            ),
            "occurrence_payload_tokens": occurrence_tokens,
            "ready_valid_tokens": 0,
            "ready_trained_tokens": 0,
        },
        "sidecars": {
            "training_schema": "cppmega_ci_chunk_training_sidecars_v2",
            "families": [
                "repository/run/workflow/job/step/actor/runner provenance",
                "language/platform/toolchain/build-system classification",
                "commands/build actions/targets",
                "tests and test summaries",
                "compiler/linker/sanitizer/build diagnostics",
                "entities and graph edges",
                "section/chunk boundaries and conservation",
            ],
        },
        "legacy_sample": {
            "state": "legacy_ineligible_sample",
            "eligible": False,
            "reason": (
                "limited 1,855-job extraction; not the current exhaustive CI store"
            ),
            "manifest": str(legacy_root / "manifest.json"),
            "source_completion": legacy_manifest["source_completion"],
            "domain_kind_counts": legacy_manifest["domain_kind_counts"],
            "parquet": legacy,
        },
    }


def _live_upstream_inputs(config: Mapping[str, object]) -> dict[str, Path]:
    """Inputs fed by still-running fetchers.

    Terminal artifacts (sealed bundles, the cancelled PR export, the legacy
    CI sample) are pinned inputs and intentionally excluded: they are not
    expected to change, so their age carries no signal.
    """
    inputs = {
        "source_completion_receipt": Path(
            str(config["source"]["completion_receipt"])
        ),
    }
    for path in config["ci"]["progress_receipts"]:
        inputs[f"ci_progress_receipt:{path}"] = Path(str(path))
    return inputs


def collect_freshness(
    config: Mapping[str, object],
    *,
    stale_minutes: float,
    now: float | None = None,
) -> dict[str, object]:
    """Report the age of each live upstream input and which ones are stale."""
    checked = time.time() if now is None else now
    stale_after_seconds = stale_minutes * 60.0
    upstreams: dict[str, dict[str, object]] = {}
    stale: list[str] = []
    for name, path in sorted(_live_upstream_inputs(config).items()):
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            upstreams[name] = {"path": str(path), "missing": True, "stale": True}
            stale.append(name)
            continue
        age_seconds = checked - mtime
        is_stale = age_seconds > stale_after_seconds
        upstreams[name] = {
            "path": str(path),
            "missing": False,
            "mtime": datetime.fromtimestamp(mtime, timezone.utc).isoformat(),
            "age_seconds": int(age_seconds),
            "stale": is_stale,
        }
        if is_stale:
            stale.append(name)
    return {
        "stale_minutes": stale_minutes,
        "checked_at": datetime.fromtimestamp(checked, timezone.utc).isoformat(),
        "stale": stale,
        "upstreams": upstreams,
    }


def check_heartbeat(
    path: Path,
    *,
    stale_after_seconds: float,
    now: float | None = None,
) -> dict[str, object]:
    """Read a watcher heartbeat and report whether the watcher looks dead."""
    checked = time.time() if now is None else now
    heartbeat = _read_json(path)
    _require_keys(heartbeat, ("schema", "pid", "recorded_at"), where=str(path))
    if heartbeat["schema"] != HEARTBEAT_SCHEMA:
        raise ValueError(
            f"{path}: unsupported heartbeat schema {heartbeat['schema']!r}"
        )
    recorded = datetime.fromisoformat(str(heartbeat["recorded_at"])).timestamp()
    age_seconds = checked - recorded
    return {
        "path": str(path),
        "pid": int(heartbeat["pid"]),
        "recorded_at": heartbeat["recorded_at"],
        "age_seconds": int(age_seconds),
        "stale": age_seconds > stale_after_seconds,
    }


def _validate_config(config: Mapping[str, object]) -> None:
    expected = {
        "schema",
        "batch_size",
        "output_dir",
        "source",
        "sealed_megatron",
        "validation_bundle",
        "pr",
        "ci",
    }
    if set(config) != expected:
        raise ValueError(
            "training-data status config fields differ: "
            f"expected={sorted(expected)} actual={sorted(config)}"
        )
    if config["schema"] != CONFIG_SCHEMA:
        raise ValueError(
            f"unsupported training-data status config schema {config['schema']!r}"
        )
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive")


def _without_volatile(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_volatile(item)
            for key, item in value.items()
            if key
            not in {
                "generated_at",
                "snapshot_completed_at",
                "status_sha256",
                "freshness",
            }
        }
    if isinstance(value, list):
        return [_without_volatile(item) for item in value]
    return value


def build_status(
    config: Mapping[str, object],
    *,
    jobs: int,
    stale_minutes: float,
) -> dict[str, object]:
    _validate_config(config)
    batch_size = int(config["batch_size"])
    source = collect_live_source(
        config["source"],
        batch_size=batch_size,
        jobs=jobs,
    )
    sealed = collect_megatron_bundle(
        Path(str(config["sealed_megatron"]["manifest"])),
        batch_size=batch_size,
        role="production",
    )
    validation = collect_megatron_bundle(
        Path(str(config["validation_bundle"]["manifest"])),
        batch_size=batch_size,
        role="validation",
    )
    pr = collect_pr_status(config["pr"])
    ci_config = dict(config["ci"])
    ci_config["batch_size"] = batch_size
    ci_config["jobs"] = jobs
    ci = collect_ci_status(ci_config)
    status: dict[str, object] = {
        "schema": STATUS_SCHEMA,
        "generated_at": _utc_now(),
        "batch_size": batch_size,
        "counting_policy": {
            "valid_tokens": "non-padding tokens in packed rows",
            "trained_tokens": "tokens with loss_mask == 1",
            "batch_count": (
                "floor(rows / batch_size) independently inside each length bucket"
            ),
            "overlap": (
                "datasets are never summed unless an explicit canonical union "
                "receipt proves disjointness"
            ),
            "ci_unique": (
                "store-local unique counters are an upper bound until cross-store "
                "canonical union/global dedup"
            ),
        },
        "datasets": {
            "live_source": source,
            "sealed_megatron": sealed,
            "validation_bundle": validation,
            "pr_mr": pr,
            "ci": ci,
        },
    }
    status["freshness"] = collect_freshness(config, stale_minutes=stale_minutes)
    status["status_sha256"] = _sha256(_without_volatile(status))
    return status


def _int(value: object) -> str:
    return f"{int(value):,}"


def _bucket_markdown(
    buckets: Mapping[str, Mapping[str, object]],
) -> list[str]:
    lines = [
        "| size | files | rows | valid | trained | pad | full batches | remainder |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in sorted(buckets.items(), key=lambda item: int(item[0])):
        batch = value["batch"]
        lines.append(
            "| "
            + " | ".join(
                (
                    key,
                    _int(value["files"]),
                    _int(value["rows"]),
                    _int(value["valid_tokens"]),
                    _int(value["trained_tokens"]),
                    _int(value["pad_tokens"]),
                    _int(batch["full_batches"]),
                    _int(batch["remainder_rows"]),
                )
            )
            + " |"
        )
    return lines


def render_markdown(status: Mapping[str, object]) -> str:
    datasets = status["datasets"]
    source = datasets["live_source"]
    sealed = datasets["sealed_megatron"]
    pr = datasets["pr_mr"]
    ci = datasets["ci"]
    lines = [
        "# Current Training-Data Status",
        "",
        f"Generated: `{status['generated_at']}`  ",
        f"Status SHA-256: `{status['status_sha256']}`  ",
        f"Batch size: `{status['batch_size']}`",
        "",
        "This file never adds overlapping snapshots or staged stores.",
        "",
        "## Summary",
        "",
        "| dataset | state | ready valid | ready trained | release ready |",
        "|---|---|---:|---:|---|",
        (
            f"| live source | {source['state']} | "
            f"{_int(source['parquet']['valid_tokens'])} | "
            f"{_int(source['parquet']['trained_tokens'])} | "
            f"{str(source['release_ready']).lower()} |"
        ),
        (
            f"| sealed Megatron | {sealed['state']} | "
            f"{_int(sealed['totals']['valid_tokens'])} | "
            f"{_int(sealed['totals']['trained_tokens'])} | "
            f"{str(sealed['release_ready']).lower()} |"
        ),
        (
            f"| PR/MR | {pr['state']} | 0 | 0 | "
            f"{str(pr['release_ready']).lower()} |"
        ),
        (
            f"| CI | {ci['state']} | 0 | 0 | "
            f"{str(ci['release_ready']).lower()} |"
        ),
        "",
        "The live and sealed source snapshots overlap and therefore have no sum.",
        "",
        "## Live source Parquet",
        "",
        f"Root: `{source['parquet']['root']}`",
        "",
        *_bucket_markdown(source["parquet"]["buckets"]),
        "",
        "Logical source-document tokens inside those rows:",
        "",
        "| category | documents | valid | trained |",
        "|---|---:|---:|---:|",
    ]
    for category, counts in source["parquet"]["classification"][
        "by_category"
    ].items():
        lines.append(
            f"| {category} | {_int(counts['documents'])} | "
            f"{_int(counts['valid_tokens'])} | {_int(counts['trained_tokens'])} |"
        )
    lines.extend(
        [
            "",
            "Blockers:",
            "",
            *[f"- {value}" for value in source["blockers"]],
            "",
            "## Sealed Megatron bundle",
            "",
            f"Manifest: `{sealed['manifest']}`",
            "",
            *_bucket_markdown(sealed["buckets"]),
            "",
            "## PR/MR",
            "",
            (
                f"Verified records: {_int(pr['records']['stored_prs'])}; "
                "eligible packed Parquet tokens: 0."
            ),
            "",
            "## CI",
            "",
            (
                "Store-local exact unique upper bound: "
                f"{_int(ci['token_accounting']['store_local_unique_upper_bound'])}; "
                "eligible packed Parquet tokens: 0."
            ),
            "",
            "Legacy 1,855-job sample:",
            "",
            *_bucket_markdown(ci["legacy_sample"]["parquet"]["buckets"]),
            "",
        ]
    )
    return "\n".join(lines)


def _status_summary(status: Mapping[str, object]) -> dict[str, object]:
    datasets = status["datasets"]
    source = datasets["live_source"]
    sealed = datasets["sealed_megatron"]
    validation = datasets["validation_bundle"]
    pr = datasets["pr_mr"]
    ci = datasets["ci"]
    legacy = ci["legacy_sample"]["parquet"]
    return {
        "live_source": {
            "state": source["state"],
            "files": source["parquet"]["files"],
            "rows": source["parquet"]["rows"],
            "valid_tokens": source["parquet"]["valid_tokens"],
            "trained_tokens": source["parquet"]["trained_tokens"],
            "release_ready": source["release_ready"],
            "schema_sha256s": sorted(source["parquet"]["schema"]["counts"]),
            "tokenizer_contract_sha256": next(
                iter(source["parquet"]["schema"]["metadata_by_sha256"].values()),
                {},
            ).get("cppmega.tokenizer_contract_sha256"),
            "source_repo_list_sha256": source["version"]["source_repo_list"][
                "sha256"
            ],
        },
        "sealed_megatron": {
            "state": sealed["state"],
            "bundle_id": sealed["version"]["bundle_id"],
            "artifact_set_sha256": sealed["version"]["artifact_set_sha256"],
            "rows": sealed["totals"]["rows"],
            "valid_tokens": sealed["totals"]["valid_tokens"],
            "trained_tokens": sealed["totals"]["trained_tokens"],
            "release_ready": sealed["release_ready"],
            "dense_sidecars": sealed["sidecars"]["dense"],
            "ragged_graph_sidecars": sealed["sidecars"]["ragged_graph"],
        },
        "validation_bundle": {
            "bundle_id": validation["version"]["bundle_id"],
            "valid_tokens": validation["totals"]["valid_tokens"],
            "trained_tokens": validation["totals"]["trained_tokens"],
        },
        "pr_mr": {
            "state": pr["state"],
            "scan_id": pr["version"]["scan_id"],
            "store_sha256": pr["version"]["store_sha256"],
            "stored_records": pr["records"]["stored_prs"],
            "ready_tokens": 0,
        },
        "ci": {
            "state": ci["state"],
            "store_local_unique_upper_bound": ci["token_accounting"][
                "store_local_unique_upper_bound"
            ],
            "ready_tokens": 0,
            "legacy_sample_valid_tokens": legacy["valid_tokens"],
            "stores": [
                {
                    "interval": store["interval"],
                    "sidecar_set_sha256": store["sidecar_set_sha256"],
                    "tokenizer_contract_sha256": store["tokenizer"][
                        "tokenizer_contract_sha256"
                    ],
                }
                for store in ci["stores"]
            ],
        },
    }


def _numeric_delta(current: object, previous: object) -> object:
    if (
        isinstance(current, int)
        and not isinstance(current, bool)
        and isinstance(previous, int)
        and not isinstance(previous, bool)
    ):
        return current - previous
    if isinstance(current, dict) and isinstance(previous, dict):
        result = {}
        for key in sorted(set(current) & set(previous)):
            delta = _numeric_delta(current[key], previous[key])
            if delta not in (None, 0, {}):
                result[key] = delta
        return result
    return None


def publish_status(status: Mapping[str, object], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_json = output_dir / "current.json"
    current_markdown = output_dir / "current.md"
    changelog = output_dir / "changelog.jsonl"
    heartbeat = output_dir / "heartbeat.json"
    lock_path = output_dir / "publish.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous = _read_json(current_json) if current_json.is_file() else None

        _atomic_write(current_json, _canonical_bytes(status))
        _atomic_write(
            current_markdown, (render_markdown(status) + "\n").encode("utf-8")
        )

        previous_sha = previous.get("status_sha256") if previous else None
        if previous_sha != status["status_sha256"]:
            summary = _status_summary(status)
            previous_summary = _status_summary(previous) if previous else {}
            entry = {
                "schema": CHANGELOG_SCHEMA,
                "recorded_at": status["generated_at"],
                "status_sha256": status["status_sha256"],
                "previous_status_sha256": previous_sha,
                "summary": summary,
                "numeric_delta": _numeric_delta(summary, previous_summary),
            }
            with changelog.open("ab") as handle:
                handle.write(_canonical_bytes(entry))
                handle.flush()
                os.fsync(handle.fileno())
        _atomic_write(
            heartbeat,
            _canonical_bytes(
                {
                    "schema": HEARTBEAT_SCHEMA,
                    "pid": os.getpid(),
                    "recorded_at": status["generated_at"],
                    "status_sha256": status["status_sha256"],
                }
            ),
        )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        "current_json": current_json,
        "current_markdown": current_markdown,
        "changelog": changelog,
        "heartbeat": heartbeat,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_data_status.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="Refresh forever at this interval; zero publishes once.",
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=DEFAULT_STALE_MINUTES,
        help=(
            "Warn (stderr + freshness marker in current.json) when a live "
            "upstream input is older than this many minutes."
        ),
    )
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.watch_seconds < 0:
        parser.error("--watch-seconds must be non-negative")
    if args.stale_minutes <= 0:
        parser.error("--stale-minutes must be positive")

    config = _read_json(args.config)
    _validate_config(config)
    output_dir = args.output_dir or Path(str(config["output_dir"]))
    while True:
        status = build_status(
            config, jobs=args.jobs, stale_minutes=args.stale_minutes
        )
        paths = publish_status(status, output_dir)
        stale_upstreams = status["freshness"]["stale"]
        if stale_upstreams:
            print(
                json.dumps(
                    {
                        "warning": "stale_upstreams",
                        "generated_at": status["generated_at"],
                        "stale_minutes": args.stale_minutes,
                        "stale": stale_upstreams,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        print(
            json.dumps(
                {
                    "generated_at": status["generated_at"],
                    "status_sha256": status["status_sha256"],
                    "paths": {key: str(value) for key, value in paths.items()},
                    "summary": _status_summary(status),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        if args.watch_seconds == 0:
            return 0
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
