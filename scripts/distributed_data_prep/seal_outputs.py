#!/usr/bin/env python3
"""Verify distributed outputs and emit a plan-only immutable cloud handoff.

This module never materializes, uploads, or repairs data.  It accepts exactly
one completed manifest for each training-data kind, verifies every referenced
local byte, validates materialized-bucket audit receipts, and requires an
explicit verified-zero receipt for sparse buckets.  The resulting seal receipt
uses the same artifact-set identity as the GCS and Nebius destination
descriptors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_sha256,
    load_json_object,
    require_exact_fields,
    require_int,
    require_nonempty,
    require_sha256,
    sha256_file,
    validate_gcs_uri,
)

OUTPUT_MANIFEST_SCHEMA = "cppmega.distributed_output_manifest_v1"
BUCKET_AUDIT_SCHEMA = "cppmega.distributed_bucket_audit_v1"
ZERO_RECEIPT_SCHEMA = "cppmega.distributed_verified_zero_v1"
SEAL_RECEIPT_SCHEMA = "cppmega.distributed_output_seal_receipt_v1"
HANDOFF_PLAN_SCHEMA = "cppmega.distributed_output_handoff_plan_v1"

TARGET_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
DATA_KINDS = ("source", "github_pr", "gitlab_mr", "ci")

_KIND_SET = frozenset(DATA_KINDS)
_MATERIALIZED = "materialized"
_VERIFIED_ZERO = "verified_zero"
_BUCKET_STATES = frozenset({_MATERIALIZED, _VERIFIED_ZERO})
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_PREFIX_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+=,@-]*$")
_ALLOWED_ARTIFACT_ROLES = frozenset(
    {
        "parquet",
        "megatron_bin",
        "megatron_idx",
        "megatron_manifest",
        "megatron_sidecar",
    }
)
_ROLE_CONTRACTS = {
    "parquet": ("parquet", "zstd", ".parquet"),
    "megatron_bin": ("megatron_mmididx_data", "none", ".bin"),
    "megatron_idx": ("megatron_mmididx_index", "none", ".idx"),
    "megatron_manifest": ("megatron_prefix_manifest", "none", ".json"),
    "megatron_sidecar": ("megatron_sidecar", "none", None),
}
_MEGATRON_DTYPES = {
    "uint8": (1, 1),
    "int32": (4, 4),
    "int64": (5, 8),
    "uint16": (8, 2),
}


def _without_digest(value: Mapping[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(dict(value))
    result.pop(field, None)
    return result


def output_manifest_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "manifest_sha256"))


def seal_receipt_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "receipt_sha256"))


def handoff_plan_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "plan_sha256"))


def artifact_set_sha256(records: Iterable[Mapping[str, object]]) -> str:
    """Use the canonical cppmega bundle artifact-set identity."""

    canonical = [
        {
            "path": str(record["path"]),
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }
        for record in sorted(records, key=lambda item: str(item["path"]))
    ]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _artifact_contracts_sha256(records: Iterable[Mapping[str, object]]) -> str:
    return canonical_sha256(
        [
            {
                "path": str(record["path"]),
                "role": str(record["role"]),
                "format": str(record["format"]),
                "compression": str(record["compression"]),
                "contract_sha256": str(record["contract_sha256"]),
            }
            for record in sorted(records, key=lambda item: str(item["path"]))
        ]
    )


def _require_relative_path(value: object, *, where: str) -> str:
    raw = require_nonempty(value, where=where)
    if "\\" in raw:
        raise ContractError(f"{where} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or str(path) != raw or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ContractError(f"{where} must be a canonical safe relative path")
    return raw


def _stable_file_record(
    root: Path,
    relative: str,
    *,
    expected_size: object,
    expected_sha256: object,
    where: str,
) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"artifact root must be a regular directory: {root}")
    relative_path = Path(relative)
    current = root
    for part in relative_path.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ContractError(f"{where} traverses a symlink: {current}")
    path = root / relative_path
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{where} must reference a regular file: {path}")
    size = require_int(expected_size, where=f"{where}.size", minimum=1)
    digest = require_sha256(expected_sha256, where=f"{where}.sha256")
    before = path.stat()
    if before.st_size != size:
        raise ContractError(f"{where} size differs from the local artifact")
    actual = sha256_file(path)
    after = path.stat()
    def identity(stat: Any) -> tuple[int, int, int, int, int]:
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    if identity(before) != identity(after):
        raise ContractError(f"{where} changed while hashing")
    if actual != digest:
        raise ContractError(f"{where} SHA-256 differs from the local artifact")
    return {"path": relative, "size": size, "sha256": digest}


def _validate_counts(value: object, *, where: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    require_exact_fields(
        value,
        {
            "document_count",
            "row_count",
            "valid_tokens",
            "trained_tokens",
            "payload_artifact_count",
        },
        where=where,
    )
    result = {
        key: require_int(value[key], where=f"{where}.{key}", minimum=1)
        for key in (
            "document_count",
            "row_count",
            "valid_tokens",
            "trained_tokens",
            "payload_artifact_count",
        )
    }
    if result["trained_tokens"] > result["valid_tokens"]:
        raise ContractError(f"{where}.trained_tokens exceeds valid_tokens")
    return result


def _validate_bucket_capacity(
    counts: Mapping[str, int], *, sequence_length: int, where: str
) -> None:
    capacity = counts["row_count"] * sequence_length
    if counts["valid_tokens"] > capacity:
        raise ContractError(f"{where}.valid_tokens exceeds packed row capacity")


def _validate_artifact_descriptor(
    value: object,
    *,
    artifact_root: Path,
    kind: str,
    sequence_length: int,
    index: int,
) -> dict[str, object]:
    where = f"{kind}/{sequence_length}.artifacts[{index}]"
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    require_exact_fields(
        value,
        {
            "path",
            "role",
            "format",
            "compression",
            "size",
            "sha256",
            "contract_sha256",
        },
        where=where,
    )
    relative = _require_relative_path(value["path"], where=f"{where}.path")
    expected_prefix = (kind, str(sequence_length))
    if PurePosixPath(relative).parts[:2] != expected_prefix:
        raise ContractError(
            f"{where}.path must start with {kind}/{sequence_length}/"
        )
    role = require_nonempty(value["role"], where=f"{where}.role")
    if role not in _ALLOWED_ARTIFACT_ROLES:
        raise ContractError(f"{where}.role is unsupported: {role!r}")
    expected_format, expected_compression, expected_suffix = _ROLE_CONTRACTS[role]
    if value["format"] != expected_format:
        raise ContractError(f"{where}.format does not match role {role}")
    if value["compression"] != expected_compression:
        raise ContractError(f"{where}.compression does not match role {role}")
    if expected_suffix is not None and not relative.endswith(expected_suffix):
        raise ContractError(f"{where}.path does not match role {role}")
    record = _stable_file_record(
        artifact_root,
        relative,
        expected_size=value["size"],
        expected_sha256=value["sha256"],
        where=where,
    )
    return {
        **record,
        "role": role,
        "format": expected_format,
        "compression": expected_compression,
        "contract_sha256": require_sha256(
            value["contract_sha256"], where=f"{where}.contract_sha256"
        ),
    }


def _validate_parquet_files(
    records: Sequence[Mapping[str, object]], *, artifact_root: Path, where: str
) -> None:
    parquet_records = [record for record in records if record["role"] == "parquet"]
    if not parquet_records:
        raise ContractError(f"{where} has no ZSTD Parquet artifact")
    for record in parquet_records:
        path = artifact_root / str(record["path"])
        if int(record["size"]) < 13:
            raise ContractError(f"{where} Parquet artifact is too small")
        with path.open("rb") as stream:
            header = stream.read(4)
            stream.seek(-8, 2)
            footer_length = struct.unpack("<I", stream.read(4))[0]
            trailer = stream.read(4)
        if (
            header != b"PAR1"
            or trailer != b"PAR1"
            or footer_length < 1
            or footer_length > int(record["size"]) - 12
        ):
            raise ContractError(f"{where} Parquet footer contract is invalid")


def _validate_megatron_prefixes(
    records: Sequence[Mapping[str, object]],
    *,
    artifact_root: Path,
    sequence_length: int,
    counts: Mapping[str, int],
    where: str,
) -> None:
    roles = ("megatron_bin", "megatron_idx", "megatron_manifest")
    role_records = {
        role: {
            str(PurePosixPath(str(record["path"])).with_suffix("")): record
            for record in records
            if record["role"] == role
        }
        for role in roles
    }
    if not role_records["megatron_bin"]:
        raise ContractError(f"{where} has no Megatron prefix")
    if len({frozenset(records) for records in role_records.values()}) != 1:
        raise ContractError(f"{where} Megatron .bin/.idx/.json prefixes disagree")
    sidecar_records = {
        str(record["path"]): record
        for record in records
        if record["role"] == "megatron_sidecar"
    }
    if not sidecar_records:
        raise ContractError(f"{where} has no Megatron sidecar")

    aggregate = {"document_count": 0, "token_count": 0, "trained_token_count": 0}
    claimed_sidecars: set[str] = set()
    for prefix in sorted(role_records["megatron_bin"]):
        data = role_records["megatron_bin"][prefix]
        index = role_records["megatron_idx"][prefix]
        manifest_record = role_records["megatron_manifest"][prefix]
        manifest_path = artifact_root / str(manifest_record["path"])
        raw, prefix_manifest = load_json_object(
            manifest_path,
            where=f"{where} Megatron prefix manifest {prefix!r}",
        )
        if (
            len(raw) != manifest_record["size"]
            or hashlib.sha256(raw).hexdigest() != manifest_record["sha256"]
        ):
            raise ContractError(
                f"{where} Megatron prefix manifest changed during verification"
            )
        if (
            prefix_manifest.get("tokenizer_contract") != "megacpp"
            or prefix_manifest.get("vocab_size") != 65536
            or prefix_manifest.get("loss_mask_alignment")
            != "source_token_predicts_next_v1"
            or prefix_manifest.get("graph_sidecar_schema")
            != "cppmega_graph_routes_v2"
        ):
            raise ContractError(f"{where} Megatron prefix contract is unsupported")
        token_count = require_int(
            prefix_manifest.get("token_count"),
            where=f"{where} Megatron prefix token_count",
            minimum=1,
        )
        document_count = require_int(
            prefix_manifest.get("document_count"),
            where=f"{where} Megatron prefix document_count",
            minimum=1,
        )
        trained_token_count = require_int(
            prefix_manifest.get("trained_token_count"),
            where=f"{where} Megatron prefix trained_token_count",
            minimum=1,
        )
        if trained_token_count > token_count:
            raise ContractError(f"{where} Megatron trained tokens exceed valid tokens")
        dtype = prefix_manifest.get("dtype")
        if dtype not in _MEGATRON_DTYPES:
            raise ContractError(f"{where} Megatron prefix dtype is unsupported")
        dtype_code, dtype_bytes = _MEGATRON_DTYPES[str(dtype)]
        if int(data["size"]) != token_count * dtype_bytes:
            raise ContractError(
                f"{where} Megatron .bin size does not match token_count"
            )
        index_path = artifact_root / str(index["path"])
        with index_path.open("rb") as stream:
            index_raw = stream.read()
        if (
            len(index_raw) != index["size"]
            or hashlib.sha256(index_raw).hexdigest() != index["sha256"]
        ):
            raise ContractError(f"{where} Megatron .idx changed during verification")
        header_size = 34
        if len(index_raw) < header_size or index_raw[:9] != b"MMIDIDX\x00\x00":
            raise ContractError(f"{where} Megatron .idx has an invalid header")
        version, observed_dtype, sequences, documents = struct.unpack_from(
            "<QBQQ", index_raw, 9
        )
        expected_size = header_size + sequences * 4 + sequences * 8 + documents * 8
        if (
            version != 1
            or observed_dtype != dtype_code
            or len(index_raw) != expected_size
        ):
            raise ContractError(f"{where} Megatron .idx contract is invalid")
        sizes_offset = header_size
        sizes = [
            item[0]
            for item in struct.iter_unpack(
                "<i", index_raw[sizes_offset : sizes_offset + sequences * 4]
            )
        ]
        if (
            sequences != document_count
            or documents != document_count + 1
            or any(length <= 0 or length > sequence_length for length in sizes)
            or sum(sizes) != token_count
        ):
            raise ContractError(f"{where} Megatron .idx counts do not close")
        pointers_offset = sizes_offset + sequences * 4
        pointers = [
            item[0]
            for item in struct.iter_unpack(
                "<q", index_raw[pointers_offset : pointers_offset + sequences * 8]
            )
        ]
        expected_pointers: list[int] = []
        token_offset = 0
        for length in sizes:
            expected_pointers.append(token_offset * dtype_bytes)
            token_offset += length
        document_offset = pointers_offset + sequences * 8
        document_indices = [
            item[0]
            for item in struct.iter_unpack("<q", index_raw[document_offset:])
        ]
        if pointers != expected_pointers or document_indices != list(
            range(document_count + 1)
        ):
            raise ContractError(f"{where} Megatron .idx offsets are invalid")

        referenced_paths: list[object] = []
        for field in (
            "side_channel_paths",
            "graph_sidecar_paths",
            "source_platform_sidecar",
        ):
            value = prefix_manifest.get(field)
            if not isinstance(value, Mapping) or not value:
                raise ContractError(
                    f"{where} Megatron prefix {field} must be a non-empty object"
                )
        for spec in prefix_manifest["side_channel_paths"].values():
            if not isinstance(spec, Mapping) or "path" not in spec:
                raise ContractError(f"{where} Megatron token sidecar spec is invalid")
            referenced_paths.append(spec["path"])
        for spec in prefix_manifest["graph_sidecar_paths"].values():
            if not isinstance(spec, Mapping):
                raise ContractError(f"{where} Megatron graph sidecar spec is invalid")
            for field in ("offsets_path", "data_path"):
                if field not in spec:
                    raise ContractError(
                        f"{where} Megatron graph sidecar spec is incomplete"
                    )
                referenced_paths.append(spec[field])
        platform = prefix_manifest["source_platform_sidecar"]
        assert isinstance(platform, Mapping)
        if platform.get("schema") != "cppmega_source_platform_v1":
            raise ContractError(
                f"{where} Megatron source platform schema is unsupported"
            )
        for field in (
            "sequence_doc_offsets_path",
            "doc_platform_offsets_path",
            "platform_ids_path",
        ):
            if field not in platform:
                raise ContractError(
                    f"{where} Megatron source platform sidecar is incomplete"
                )
            referenced_paths.append(platform[field])
        registry = prefix_manifest.get("source_identity_registry")
        if not isinstance(registry, Mapping) or "path" not in registry:
            raise ContractError(
                f"{where} Megatron source identity registry is missing"
            )
        referenced_paths.append(registry["path"])
        objective = prefix_manifest.get("objective_contract")
        if not isinstance(objective, Mapping):
            raise ContractError(f"{where} Megatron objective contract is missing")
        objective_sidecar = objective.get("objective_id_sidecar")
        if (
            not isinstance(objective_sidecar, Mapping)
            or "path" not in objective_sidecar
        ):
            raise ContractError(
                f"{where} Megatron objective sidecar is missing"
            )
        referenced_paths.append(objective_sidecar["path"])
        manifest_parent = PurePosixPath(str(manifest_record["path"])).parent
        for raw_path in referenced_paths:
            relative_path = _require_relative_path(
                raw_path, where=f"{where} Megatron sidecar path"
            )
            full_path = str(manifest_parent / relative_path)
            if full_path not in sidecar_records or full_path in claimed_sidecars:
                raise ContractError(
                    f"{where} Megatron sidecar is missing or claimed more than once"
                )
            claimed_sidecars.add(full_path)
        aggregate["document_count"] += document_count
        aggregate["token_count"] += token_count
        aggregate["trained_token_count"] += trained_token_count

    expected_counts = {
        "document_count": counts["row_count"],
        "token_count": counts["valid_tokens"],
        "trained_token_count": counts["trained_tokens"],
    }
    if aggregate != expected_counts:
        raise ContractError(f"{where} Megatron prefix counts do not close")
    if claimed_sidecars != set(sidecar_records):
        raise ContractError(f"{where} has unclaimed Megatron sidecars")


def _validate_receipt_descriptor(
    value: object,
    *,
    artifact_root: Path,
    kind: str,
    sequence_length: int,
    where: str,
) -> tuple[dict[str, object], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    require_exact_fields(value, {"path", "size", "sha256"}, where=where)
    relative = _require_relative_path(value["path"], where=f"{where}.path")
    if PurePosixPath(relative).parts[:2] != (kind, str(sequence_length)):
        raise ContractError(
            f"{where}.path must start with {kind}/{sequence_length}/"
        )
    record = _stable_file_record(
        artifact_root,
        relative,
        expected_size=value["size"],
        expected_sha256=value["sha256"],
        where=where,
    )
    raw, payload = load_json_object(
        artifact_root / relative,
        where=where,
    )
    if (
        len(raw) != record["size"]
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise ContractError(f"{where} changed between hashing and JSON validation")
    return record, payload


def _validate_audit(
    value: object,
    *,
    artifact_root: Path,
    kind: str,
    sequence_length: int,
    bindings: Mapping[str, object],
    counts: Mapping[str, int],
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    where = f"{kind}/{sequence_length}.audit"
    record, audit = _validate_receipt_descriptor(
        value,
        artifact_root=artifact_root,
        kind=kind,
        sequence_length=sequence_length,
        where=where,
    )
    require_exact_fields(
        audit,
        {
            "schema",
            "status",
            "kind",
            "sequence_length",
            "source_receipt_sha256",
            "producer_sha256",
            "tokenizer_sha256",
            "dataset_schema_sha256",
            "payload_artifact_set_sha256",
            "artifact_contracts_sha256",
            "payload_artifact_count",
            "counts",
            "bad_files",
            "bad_rows",
            "hashes_verified",
            "schema_verified",
            "parquet_verified",
            "megatron_verified",
            "packing_verified",
            "token_conservation_verified",
        },
        where=where,
    )
    expected = {
        "schema": BUCKET_AUDIT_SCHEMA,
        "status": "verified",
        "kind": kind,
        "sequence_length": sequence_length,
        "source_receipt_sha256": bindings["source_receipt_sha256"],
        "producer_sha256": bindings["producer_sha256"],
        "tokenizer_sha256": bindings["tokenizer_sha256"],
        "dataset_schema_sha256": bindings["dataset_schema_sha256"],
        "payload_artifact_set_sha256": artifact_set_sha256(artifacts),
        "artifact_contracts_sha256": _artifact_contracts_sha256(artifacts),
        "payload_artifact_count": len(artifacts),
        "counts": dict(counts),
        "bad_files": 0,
        "bad_rows": 0,
        "hashes_verified": True,
        "schema_verified": True,
        "parquet_verified": True,
        "megatron_verified": True,
        "packing_verified": True,
        "token_conservation_verified": True,
    }
    if audit != expected:
        raise ContractError(f"{where} is not the exact green bucket audit")
    return {**record, "role": "audit_receipt", "schema": BUCKET_AUDIT_SCHEMA}


def _validate_zero_receipt(
    value: object,
    *,
    artifact_root: Path,
    kind: str,
    sequence_length: int,
    bindings: Mapping[str, object],
) -> dict[str, object]:
    where = f"{kind}/{sequence_length}.zero_receipt"
    record, receipt = _validate_receipt_descriptor(
        value,
        artifact_root=artifact_root,
        kind=kind,
        sequence_length=sequence_length,
        where=where,
    )
    require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "kind",
            "sequence_length",
            "reason",
            "source_receipt_sha256",
            "producer_sha256",
            "tokenizer_sha256",
            "dataset_schema_sha256",
            "eligibility_query_sha256",
            "document_count",
            "row_count",
            "valid_tokens",
            "trained_tokens",
        },
        where=where,
    )
    if (
        receipt["schema"] != ZERO_RECEIPT_SCHEMA
        or receipt["status"] != _VERIFIED_ZERO
        or receipt["kind"] != kind
        or receipt["sequence_length"] != sequence_length
        or receipt["source_receipt_sha256"] != bindings["source_receipt_sha256"]
        or receipt["producer_sha256"] != bindings["producer_sha256"]
        or receipt["tokenizer_sha256"] != bindings["tokenizer_sha256"]
        or receipt["dataset_schema_sha256"] != bindings["dataset_schema_sha256"]
    ):
        raise ContractError(f"{where} binding drifted")
    require_nonempty(receipt["reason"], where=f"{where}.reason")
    require_sha256(
        receipt["eligibility_query_sha256"],
        where=f"{where}.eligibility_query_sha256",
    )
    for field in ("document_count", "row_count", "valid_tokens", "trained_tokens"):
        if require_int(receipt[field], where=f"{where}.{field}") != 0:
            raise ContractError(f"{where}.{field} must be zero")
    return {**record, "role": "zero_receipt", "schema": ZERO_RECEIPT_SCHEMA}


def _validate_bindings(value: object, *, where: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    fields = {
        "source_receipt_sha256",
        "producer_sha256",
        "tokenizer_sha256",
        "dataset_schema_sha256",
    }
    require_exact_fields(value, fields, where=where)
    return {
        field: require_sha256(value[field], where=f"{where}.{field}")
        for field in sorted(fields)
    }


def validate_output_manifest(
    value: Mapping[str, object], *, artifact_root: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = copy.deepcopy(dict(value))
    require_exact_fields(
        manifest,
        {
            "schema",
            "status",
            "kind",
            "sequence_lengths",
            "bindings",
            "buckets",
            "manifest_sha256",
        },
        where="output manifest",
    )
    if manifest["schema"] != OUTPUT_MANIFEST_SCHEMA or manifest["status"] != "complete":
        raise ContractError("output manifest schema/status is unsupported")
    kind = require_nonempty(manifest["kind"], where="output manifest kind")
    if kind not in _KIND_SET:
        raise ContractError(f"unsupported output manifest kind: {kind!r}")
    if manifest["sequence_lengths"] != list(TARGET_LENGTHS):
        raise ContractError("output manifest sequence ladder drifted")
    declared_digest = require_sha256(
        manifest["manifest_sha256"], where="output manifest manifest_sha256"
    )
    if output_manifest_sha256(manifest) != declared_digest:
        raise ContractError("output manifest logical digest is invalid")
    bindings = _validate_bindings(manifest["bindings"], where=f"{kind}.bindings")

    buckets = manifest["buckets"]
    if not isinstance(buckets, list) or len(buckets) != len(TARGET_LENGTHS):
        raise ContractError(f"{kind}.buckets must cover the complete sequence ladder")
    normalized_buckets: list[dict[str, object]] = []
    handoff_records: list[dict[str, object]] = []
    for expected_length, raw_bucket in zip(TARGET_LENGTHS, buckets, strict=True):
        where = f"{kind}/{expected_length}"
        if not isinstance(raw_bucket, Mapping):
            raise ContractError(f"{where} must be an object")
        state = raw_bucket.get("status")
        if state not in _BUCKET_STATES:
            raise ContractError(f"{where}.status is unsupported: {state!r}")
        if raw_bucket.get("sequence_length") != expected_length:
            raise ContractError(f"{where}.sequence_length drifted")
        if state == _VERIFIED_ZERO:
            require_exact_fields(
                raw_bucket,
                {"sequence_length", "status", "zero_receipt"},
                where=where,
            )
            zero = _validate_zero_receipt(
                raw_bucket["zero_receipt"],
                artifact_root=artifact_root,
                kind=kind,
                sequence_length=expected_length,
                bindings=bindings,
            )
            handoff_records.append(zero)
            normalized_buckets.append(
                {
                    "sequence_length": expected_length,
                    "status": _VERIFIED_ZERO,
                    "zero_receipt": {
                        key: zero[key] for key in ("path", "size", "sha256")
                    },
                }
            )
            continue

        require_exact_fields(
            raw_bucket,
            {"sequence_length", "status", "counts", "artifacts", "audit"},
            where=where,
        )
        counts = _validate_counts(raw_bucket["counts"], where=f"{where}.counts")
        _validate_bucket_capacity(
            counts,
            sequence_length=expected_length,
            where=f"{where}.counts",
        )
        artifacts_value = raw_bucket["artifacts"]
        if not isinstance(artifacts_value, list) or not artifacts_value:
            raise ContractError(f"{where}.artifacts must be a non-empty list")
        artifacts = [
            _validate_artifact_descriptor(
                raw,
                artifact_root=artifact_root,
                kind=kind,
                sequence_length=expected_length,
                index=index,
            )
            for index, raw in enumerate(artifacts_value)
        ]
        paths = [str(record["path"]) for record in artifacts]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ContractError(f"{where}.artifacts must be unique and path-sorted")
        _validate_parquet_files(
            artifacts,
            artifact_root=artifact_root,
            where=where,
        )
        _validate_megatron_prefixes(
            artifacts,
            artifact_root=artifact_root,
            sequence_length=expected_length,
            counts=counts,
            where=where,
        )
        if counts["payload_artifact_count"] != len(artifacts):
            raise ContractError(f"{where}.counts.payload_artifact_count drifted")
        audit = _validate_audit(
            raw_bucket["audit"],
            artifact_root=artifact_root,
            kind=kind,
            sequence_length=expected_length,
            bindings=bindings,
            counts=counts,
            artifacts=artifacts,
        )
        handoff_records.extend(artifacts)
        handoff_records.append(audit)
        normalized_buckets.append(
            {
                "sequence_length": expected_length,
                "status": _MATERIALIZED,
                "counts": counts,
                "payload_artifact_set_sha256": artifact_set_sha256(artifacts),
                "artifacts": artifacts,
                "audit": {key: audit[key] for key in ("path", "size", "sha256")},
            }
        )
    manifest["bindings"] = bindings
    manifest["buckets"] = normalized_buckets
    return manifest, handoff_records


def load_output_manifest(
    path: Path, *, artifact_root: Path
) -> tuple[dict[str, object], str, list[dict[str, object]]]:
    raw, payload = load_json_object(path, where="distributed output manifest")
    manifest, records = validate_output_manifest(payload, artifact_root=artifact_root)
    initial_raw_sha256 = hashlib.sha256(raw).hexdigest()
    if hashlib.sha256(path.read_bytes()).hexdigest() != initial_raw_sha256:
        raise ContractError(f"output manifest changed during verification: {path}")
    return manifest, initial_raw_sha256, records


def _safe_nebius_endpoint(value: object) -> str:
    endpoint = require_nonempty(value, where="Nebius endpoint URL")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ContractError("Nebius endpoint must be a credential-free HTTPS origin")
    return endpoint.rstrip("/")


def _safe_object_prefix(value: object, *, where: str) -> str:
    prefix = require_nonempty(value, where=where).strip("/")
    parts = prefix.split("/")
    if not parts or any(_PREFIX_PART_RE.fullmatch(part) is None for part in parts):
        raise ContractError(f"{where} contains an unsafe object component")
    return prefix


def _destination_descriptors(
    *,
    artifact_digest: str,
    gcs_prefix: str,
    nebius_endpoint_url: str,
    nebius_bucket: str,
    nebius_prefix: str,
) -> list[dict[str, object]]:
    gcs = validate_gcs_uri(gcs_prefix.rstrip("/"), where="GCS handoff prefix")
    bucket = require_nonempty(nebius_bucket, where="Nebius bucket")
    if _BUCKET_RE.fullmatch(bucket) is None:
        raise ContractError("Nebius bucket name is not canonical")
    prefix = _safe_object_prefix(nebius_prefix, where="Nebius object prefix")
    endpoint = _safe_nebius_endpoint(nebius_endpoint_url)
    suffix = f"artifact-sets/{artifact_digest}"
    return [
        {
            "provider": "gcs",
            "protocol": "gcs-if-generation-match-zero-v1",
            "artifact_set_sha256": artifact_digest,
            "root_uri": f"{gcs}/{suffix}",
            "immutable_create_precondition": {"if_generation_match": 0},
            "commit_object": f"{gcs}/{suffix}/seal_receipt.json",
        },
        {
            "provider": "nebius_s3",
            "protocol": "conditional-complete-v1",
            "artifact_set_sha256": artifact_digest,
            "endpoint_url": endpoint,
            "bucket": bucket,
            "key_prefix": f"{prefix}/{suffix}",
            "root_uri": f"s3://{bucket}/{prefix}/{suffix}",
            "immutable_create_precondition": {"if_none_match": "*"},
            "commit_object": f"{prefix}/{suffix}/seal_receipt.json",
        },
    ]


def seal_outputs(
    manifest_paths: Mapping[str, Path],
    *,
    artifact_root: Path,
    gcs_prefix: str,
    nebius_endpoint_url: str,
    nebius_bucket: str,
    nebius_prefix: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify all output manifests and return a seal receipt and handoff plan."""

    if set(manifest_paths) != _KIND_SET:
        raise ContractError(
            "output manifests must contain exactly " + ", ".join(DATA_KINDS)
        )
    unresolved_root = Path(artifact_root)
    if unresolved_root.is_symlink():
        raise ContractError("artifact root must not be a symlink")
    root = unresolved_root.resolve()
    manifests: list[dict[str, object]] = []
    input_records: list[dict[str, object]] = []
    all_artifacts: list[dict[str, object]] = []
    common_contract: tuple[str, str] | None = None
    for kind in DATA_KINDS:
        unresolved_path = Path(manifest_paths[kind])
        if unresolved_path.is_symlink():
            raise ContractError(
                f"output manifest must not be a symlink: {unresolved_path}"
            )
        path = unresolved_path.resolve()
        manifest, raw_sha256, records = load_output_manifest(
            path, artifact_root=root
        )
        if manifest["kind"] != kind:
            raise ContractError(
                f"manifest path for {kind} contains {manifest['kind']!r}"
            )
        bindings = manifest["bindings"]
        assert isinstance(bindings, Mapping)
        observed_contract = (
            str(bindings["tokenizer_sha256"]),
            str(bindings["dataset_schema_sha256"]),
        )
        if common_contract is None:
            common_contract = observed_contract
        elif observed_contract != common_contract:
            raise ContractError(
                "output manifests disagree on tokenizer/schema contract"
            )
        manifests.append(manifest)
        input_records.append(
            {
                "kind": kind,
                "path": str(path),
                "raw_sha256": raw_sha256,
                "logical_sha256": manifest["manifest_sha256"],
                "source_receipt_sha256": bindings["source_receipt_sha256"],
                "producer_sha256": bindings["producer_sha256"],
            }
        )
        all_artifacts.extend(records)

    paths = [str(record["path"]) for record in all_artifacts]
    if len(paths) != len(set(paths)):
        raise ContractError("two output manifests reference the same artifact path")
    all_artifacts.sort(key=lambda record: str(record["path"]))
    artifact_digest = artifact_set_sha256(all_artifacts)

    coverage: list[dict[str, object]] = []
    blocking_reasons: list[str] = []
    for index, sequence_length in enumerate(TARGET_LENGTHS):
        materialized: list[str] = []
        for manifest in manifests:
            buckets = manifest["buckets"]
            assert isinstance(buckets, list)
            bucket = buckets[index]
            assert isinstance(bucket, Mapping)
            if bucket["status"] == _MATERIALIZED:
                materialized.append(str(manifest["kind"]))
        zero = [kind for kind in DATA_KINDS if kind not in materialized]
        coverage.append(
            {
                "sequence_length": sequence_length,
                "materialized_kinds": materialized,
                "verified_zero_kinds": zero,
            }
        )
        if not materialized:
            blocking_reasons.append(
                f"sequence length {sequence_length} has no materialized training data"
            )
    training_ready = not blocking_reasons
    assert common_contract is not None
    receipt: dict[str, object] = {
        "schema": SEAL_RECEIPT_SCHEMA,
        "status": "verified",
        "training_ready": training_ready,
        "sequence_lengths": list(TARGET_LENGTHS),
        "data_kinds": list(DATA_KINDS),
        "tokenizer_sha256": common_contract[0],
        "dataset_schema_sha256": common_contract[1],
        "input_manifests": input_records,
        "coverage": coverage,
        "artifact_count": len(all_artifacts),
        "artifact_bytes": sum(int(record["size"]) for record in all_artifacts),
        "artifacts": [
            {key: record[key] for key in ("path", "size", "sha256")}
            for record in all_artifacts
        ],
        "artifact_set_sha256": artifact_digest,
        "blocking_reasons": blocking_reasons,
    }
    receipt["receipt_sha256"] = seal_receipt_sha256(receipt)
    destinations = _destination_descriptors(
        artifact_digest=artifact_digest,
        gcs_prefix=gcs_prefix,
        nebius_endpoint_url=nebius_endpoint_url,
        nebius_bucket=nebius_bucket,
        nebius_prefix=nebius_prefix,
    )
    plan: dict[str, object] = {
        "schema": HANDOFF_PLAN_SCHEMA,
        "status": "ready" if training_ready else "blocked",
        "training_ready": training_ready,
        "publication_authorized": training_ready,
        "upload_performed": False,
        "execution": "plan_only_no_upload",
        "seal_receipt_sha256": receipt["receipt_sha256"],
        "artifact_set_sha256": artifact_digest,
        "artifact_count": receipt["artifact_count"],
        "artifact_bytes": receipt["artifact_bytes"],
        "sequence_lengths": list(TARGET_LENGTHS),
        "data_kinds": list(DATA_KINDS),
        "blocking_reasons": blocking_reasons,
        "destinations": destinations,
    }
    plan["plan_sha256"] = handoff_plan_sha256(plan)
    return receipt, plan


def _parse_manifest(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--manifest must be KIND=PATH")
    kind, raw_path = value.split("=", 1)
    if kind not in _KIND_SET or not raw_path:
        raise argparse.ArgumentTypeError("--manifest uses an unsupported kind/path")
    return kind, Path(raw_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", action="append", type=_parse_manifest, required=True
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--gcs-prefix", required=True)
    parser.add_argument("--nebius-endpoint-url", required=True)
    parser.add_argument("--nebius-bucket", required=True)
    parser.add_argument("--nebius-prefix", required=True)
    parser.add_argument("--seal-receipt", type=Path, required=True)
    parser.add_argument("--handoff-plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest_paths: dict[str, Path] = {}
    for kind, path in args.manifest:
        if kind in manifest_paths:
            raise ContractError(f"duplicate --manifest kind: {kind}")
        manifest_paths[kind] = path
    receipt, plan = seal_outputs(
        manifest_paths,
        artifact_root=args.artifact_root,
        gcs_prefix=args.gcs_prefix,
        nebius_endpoint_url=args.nebius_endpoint_url,
        nebius_bucket=args.nebius_bucket,
        nebius_prefix=args.nebius_prefix,
    )
    atomic_write_json(args.seal_receipt, receipt)
    atomic_write_json(args.handoff_plan, plan)
    return 0 if plan["training_ready"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
