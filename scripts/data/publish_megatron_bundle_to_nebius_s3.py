#!/usr/bin/env python3
"""Publish an immutable, hashed Megatron bundle to Nebius Object Storage.

Every artifact is uploaded under ``<prefix>/bundles/<bundle_id>/`` with its
SHA-256 in object metadata and verified by HEAD.  The bundle manifest is
uploaded only after all artifacts verify; ``latest.json`` is the final small
commit pointer.  With ``--archive``, the exact same logical bundle is published
as one validated tar.zst object under ``<prefix>/transports/<bundle_id>/`` and
``latest_transport.json`` is committed last.  Consumers must ignore any bundle
lacking its manifest or transport descriptor.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
from typing import BinaryIO, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cppmega.megatron.objective_contract import (  # noqa: E402
    LEGACY_OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA,
    OBJECTIVE_GRAPH_SIDECARS,
    OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA,
    OBJECTIVE_TOKEN_SIDE_CHANNELS,
    validate_materialized_objective_contract,
    validate_objective_contract,
)
from cppmega.receipt_binding import validate_implementation_binding  # noqa: E402
from cppmega.megatron.graph_recipe import (  # noqa: E402
    stage1_graph_recipe_binding,
)
from cppmega.megatron.domain_route_contract import (  # noqa: E402
    CASE5_RECEIPT_KEY,
    CASE5_SCHEMA_VERSION,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    DOMAIN_ROUTE_COLUMNS,
    DOMAIN_SCHEMA_SHA256,
    GRAPH_ROUTE_COLUMNS,
    GRAPH_ROUTE_COORDINATE_SPACES,
    SOURCE_IDENTITY_REGISTRY_SCHEMA,
    TOKENIZER_CONTRACT_SHA256,
)
from cppmega.megatron.h200_preflight import (  # noqa: E402
    GRAPH_CHUNK_KIND_COUNT,
    GraphChunkKind,
)


DEFAULT_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_BUCKET = "cppmega-sidecar-20260627"
DEFAULT_PREFIX = "cppmega-megatron/macro-routes"
S3_SINGLE_PUT_MAX_BYTES = 5 * 1024**3
S3_MIN_MULTIPART_PART_BYTES = 5 * 1024**2
S3_MAX_MULTIPART_PART_BYTES = 5 * 1024**3
S3_MAX_MULTIPART_PARTS = 10_000
S3_MAX_OBJECT_BYTES = 5 * 1024**4
MULTIPART_DEFAULT_PART_BYTES = 512 * 1024**2
MULTIPART_PART_ALIGNMENT_BYTES = 1024**2
MULTIPART_PUBLICATION_PROTOCOL = "conditional-complete-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_BUNDLE_TOKENIZER_CONTRACT = "megacpp-vocab-65536"
EXPECTED_PREFIX_TOKENIZER_CONTRACT = "megacpp"
EXPECTED_LOSS_MASK_ALIGNMENT = "source_token_predicts_next_v1"
EXPECTED_VOCAB_SIZE = 65536
EXPECTED_GRAPH_SIDECAR_SCHEMA = "cppmega_graph_routes_v2"
OBJECTIVE_SAMPLING_MODE_V1 = "deterministic_epoch_shuffle_v1"
OBJECTIVE_SAMPLING_MODE_V2 = "deterministic_shard_row_group_record_batch_shuffle_v2"
OBJECTIVE_SAMPLING_BASE_KEYS = frozenset(
    {
        "mode",
        "seed",
        "requested_samples",
        "full_passes",
        "tail_rows",
        "min_row_reuse",
        "max_row_reuse",
    }
)
OBJECTIVE_SAMPLING_V2_KEYS = OBJECTIVE_SAMPLING_BASE_KEYS | {
    "record_batch_rows",
    "ordering",
    "cursor_semantics",
    "producer",
    "final_cursor",
}
OBJECTIVE_SAMPLING_V2_ORDERING = {
    "permutation": "sha256_sort_key_v1",
    "epochs": "ascending",
    "shards": "seeded_permutation_per_epoch",
    "row_groups": "seeded_permutation_per_shard_epoch",
    "record_batches": "physical_order_within_row_group",
    "rows": "seeded_permutation_within_record_batch",
}
OBJECTIVE_SAMPLING_V2_CURSOR_KEYS = frozenset(
    {
        "epoch",
        "shard_position",
        "shard_index",
        "row_group_position",
        "row_group_index",
        "record_batch_index",
        "row_shuffle_position",
        "row_index_in_record_batch",
        "source_index",
    }
)
OBJECTIVE_SAMPLING_V2_PRODUCER_NAME = "pyarrow.parquet.ParquetFile.iter_batches"
OBJECTIVE_SAMPLING_V2_PRODUCER_VERSION = 1
OBJECTIVE_SAMPLING_V2_PRODUCER_KEYS = frozenset(
    {"name", "version", "row_group_rows"}
)
CANONICAL_TOKENIZER_CONTRACT_PATH = (
    REPO_ROOT / "data/tokenizer_v2/tokenizer_contract_v1.json"
)
CANONICAL_DOMAIN_SCHEMA_PATH = REPO_ROOT / "data/domain_schema_v1.json"
CANONICAL_TOKENIZER_CONTRACT_BYTES = CANONICAL_TOKENIZER_CONTRACT_PATH.read_bytes()
CANONICAL_TOKENIZER_CONTRACT_SHA256 = hashlib.sha256(
    CANONICAL_TOKENIZER_CONTRACT_BYTES
).hexdigest()
CANONICAL_TOKENIZER_CONTRACT = json.loads(CANONICAL_TOKENIZER_CONTRACT_BYTES)
CANONICAL_DOMAIN_SCHEMA = json.loads(
    CANONICAL_DOMAIN_SCHEMA_PATH.read_text(encoding="utf-8")
)
EXPECTED_TOKENIZER_CORE_TOKENS = (
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<FIM_PREFIX>",
    "<FIM_MIDDLE>",
    "<FIM_SUFFIX>",
    "<CODE_START>",
    "<CODE_END>",
    "<THINK_START>",
    "<THINK_END>",
    "<QUERY_TOOL>",
    "<INDEX>",
    "<DEBUG_CONTEXT>",
    "<FILE_SEP>",
    "<DIFF_START>",
    "<DIFF_END>",
    "<COMMENT_START>",
    "<COMMENT_END>",
    "<TOOL_RESULT>",
    "<THINK_CODE>",
    "<THINK_ERROR>",
    "<THINK_FIX>",
    "<THINK_VERIFY>",
    "<THINK_PLAN>",
    "<THINK_TRACE>",
    "<SCRIPT_START>",
    "<SCRIPT_END>",
    "<SCRIPT_RESULT>",
    "<COMPILE_START>",
    "<COMPILE_END>",
    "<COMPILE_OK>",
    "<COMPILE_ERROR>",
    "<TEST_START>",
    "<TEST_END>",
    "<TEST_PASS>",
    "<TEST_FAIL>",
    "<AST_NODE>",
    "<SYMBOL_REF>",
    "<TYPE_INFO>",
    "<SCOPE_ENTER>",
    "<SCOPE_EXIT>",
    "<INCLUDE_CONTEXT>",
    "<TEMPLATE_INST>",
    "<OVERLOAD_SET>",
    "<FIM_INSTRUCTION>",
    "<SPACE>",
    "<NL>",
)
TOKEN_INDEX_DTYPE_CODES = {
    "uint8": 1,
    "int32": 4,
    "int64": 5,
    "uint16": 8,
}
DTYPE_SIZES = {
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "int32": 4,
    "int64": 8,
}
TOKEN_SIDECAR_DTYPES = dict(OBJECTIVE_TOKEN_SIDE_CHANNELS)
GRAPH_SIDECAR_SPECS = {
    name: (
        kind,
        dtype,
        [2] if kind == "edge_pairs" else [3] if kind == "edge_triples" else [1],
    )
    for name, kind, dtype in OBJECTIVE_GRAPH_SIDECARS
}
REQUIRED_TOKEN_SIDECARS = set(TOKEN_SIDECAR_DTYPES)
REQUIRED_GRAPH_SIDECARS = set(GRAPH_SIDECAR_SPECS)
if REQUIRED_GRAPH_SIDECARS != set(GRAPH_ROUTE_COORDINATE_SPACES):
    raise RuntimeError("graph sidecar coordinate contract is incomplete")
NONZERO_GRAPH_SIDECARS = {
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
}
ROUTE_GRAPH_SIDECARS = tuple(
    name for name in GRAPH_SIDECAR_SPECS if name.endswith("_edges")
)
_GRAPH_FAMILY_BY_COLUMN = {
    "token_domain_edges": "domain",
    "token_build_edges": "build",
    "token_shell_edges": "shell",
    "token_diagnostic_edges": "diagnostic",
    "token_cross_domain_edges": "cross_domain",
}
_ALLOWED_EDGE_KINDS = {
    f"token_{family}_edges": frozenset(int(value) for value in values)
    for family, values in CANONICAL_DOMAIN_SCHEMA["edge_families"].items()
}
_RESERVED_ROLE_IDS = CANONICAL_TOKENIZER_CONTRACT["reserved_role_assignments"]
_DELIMITER_BY_TOKEN_ID: dict[int, tuple[int, bool, int]] = {}
for _delimiter in CANONICAL_DOMAIN_SCHEMA["delimiter_roles"].values():
    _start_id = int(_RESERVED_ROLE_IDS[_delimiter["start"]])
    _end_id = int(_RESERVED_ROLE_IDS[_delimiter["end"]])
    _domain_id = int(_delimiter["domain_id"])
    _DELIMITER_BY_TOKEN_ID[_start_id] = (_domain_id, True, _end_id)
    _DELIMITER_BY_TOKEN_ID[_end_id] = (_domain_id, False, _start_id)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _canonical_artifact_records(records: Iterable[dict]) -> list[dict[str, object]]:
    return [
        {
            "path": str(record["path"]),
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }
        for record in sorted(records, key=lambda item: str(item["path"]))
    ]


def _artifact_set_sha256(records: Iterable[dict]) -> str:
    payload = json.dumps(
        _canonical_artifact_records(records), separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _s3_env() -> dict[str, str]:
    return _resolve_s3_env(os.environ)


def _resolve_s3_env(source: Mapping[str, str]) -> dict[str, str]:
    """Resolve one complete credential family without mixing providers."""

    env = dict(source)
    nebius_access_name = "NEBIUS_S3_ACCESS_KEY_ID"
    nebius_secret_name = "NEBIUS_S3_SECRET_ACCESS_KEY"
    if nebius_access_name in env or nebius_secret_name in env:
        access = env.get(nebius_access_name)
        secret = env.get(nebius_secret_name)
        if not access or not secret:
            raise SystemExit(
                "a complete Nebius S3 credential pair is required when either "
                "NEBIUS_S3 credential is set"
            )
        env["AWS_ACCESS_KEY_ID"] = access
        env["AWS_SECRET_ACCESS_KEY"] = secret
        env.pop("AWS_SESSION_TOKEN", None)
        env.pop("AWS_SECURITY_TOKEN", None)
        return env

    access = env.get("AWS_ACCESS_KEY_ID")
    secret = env.get("AWS_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise SystemExit("a complete AWS S3 credential pair is required")
    return env


def _validate_artifact_relative_path(relative: object) -> str:
    if not isinstance(relative, str):
        raise ValueError("bundle artifact path must be a string")
    posix = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or posix.is_absolute()
        or any(part in ("", ".", "..") for part in posix.parts)
        or posix.as_posix() != relative
    ):
        raise ValueError(f"unsafe artifact path in bundle manifest: {relative!r}")
    return relative


def _safe_artifact_path(bundle: Path, relative: str) -> Path:
    _validate_artifact_relative_path(relative)
    posix = PurePosixPath(relative)
    path = (bundle / Path(*posix.parts)).resolve()
    root = bundle.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact path escapes bundle: {relative!r}")
    return path


def _require_manifest_tokenizer_contract(manifest: dict) -> None:
    if manifest.get("tokenizer_contract") != EXPECTED_BUNDLE_TOKENIZER_CONTRACT:
        raise ValueError(
            "bundle tokenizer_contract must be "
            f"{EXPECTED_BUNDLE_TOKENIZER_CONTRACT!r}, got "
            f"{manifest.get('tokenizer_contract')!r}"
        )
    if int(manifest.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"bundle vocab_size must be {EXPECTED_VOCAB_SIZE}, "
            f"got {manifest.get('vocab_size')!r}"
        )


def _read_int64_offsets(path: Path, expected_count: int) -> list[int]:
    raw = path.read_bytes()
    if len(raw) != expected_count * 8:
        raise ValueError(
            f"{path}: offsets byte size {len(raw)} != {expected_count * 8}"
        )
    return list(struct.unpack(f"<{expected_count}q", raw))


def _read_mmididx(path: Path, *, expected_dtype: str) -> dict[str, object]:
    with path.open("rb") as stream:
        if stream.read(9) != b"MMIDIDX\x00\x00":
            raise ValueError(f"{path}: invalid MMIDIDX header")
        version_raw = stream.read(8)
        dtype_raw = stream.read(1)
        sequences_raw = stream.read(8)
        documents_raw = stream.read(8)
        if tuple(map(len, (version_raw, dtype_raw, sequences_raw, documents_raw))) != (
            8,
            1,
            8,
            8,
        ):
            raise ValueError(f"{path}: truncated MMIDIDX header")
        version = struct.unpack("<Q", version_raw)[0]
        dtype_code = struct.unpack("<B", dtype_raw)[0]
        sequences = struct.unpack("<Q", sequences_raw)[0]
        documents = struct.unpack("<Q", documents_raw)[0]
        expected_size = 34 + sequences * 4 + sequences * 8 + documents * 8
        if path.stat().st_size != expected_size:
            raise ValueError(
                f"{path}: MMIDIDX size {path.stat().st_size} != {expected_size}"
            )
        sizes_raw = stream.read(sequences * 4)
        pointers_raw = stream.read(sequences * 8)
        document_indices_raw = stream.read(documents * 8)
    if version != 1:
        raise ValueError(f"{path}: unsupported MMIDIDX version {version}, expected 1")
    expected_dtype_code = TOKEN_INDEX_DTYPE_CODES.get(expected_dtype)
    if expected_dtype_code is None:
        raise ValueError(f"{path}: unsupported MMIDIDX token dtype {expected_dtype!r}")
    if dtype_code != expected_dtype_code:
        raise ValueError(
            f"{path}: MMIDIDX dtype code {dtype_code} != {expected_dtype_code} "
            f"for {expected_dtype}"
        )
    lengths = [value[0] for value in struct.iter_unpack("<i", sizes_raw)]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"{path}: MMIDIDX sequence lengths must be positive")
    pointers = [value[0] for value in struct.iter_unpack("<q", pointers_raw)]
    expected_pointers: list[int] = []
    token_offset = 0
    for length in lengths:
        expected_pointers.append(token_offset * DTYPE_SIZES[expected_dtype])
        token_offset += length
    if pointers != expected_pointers:
        raise ValueError(f"{path}: MMIDIDX sequence pointers do not match token dtype")
    document_indices = [
        value[0] for value in struct.iter_unpack("<q", document_indices_raw)
    ]
    if document_indices != list(range(sequences + 1)):
        raise ValueError(
            f"{path}: MMIDIDX document indices must bind one packed sequence per document"
        )
    return {
        "version": version,
        "dtype_code": dtype_code,
        "sequences": sequences,
        "documents": documents,
        "tokens": token_offset,
        "lengths": lengths,
    }


def _contains_nonzero_byte(path: Path, chunk_size: int = 8 * 1024 * 1024) -> bool:
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            if any(chunk):
                return True
    return False


def _safe_prefix_file(prefix_dir: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or posix.is_absolute()
        or any(part in ("", ".", "..") for part in posix.parts)
        or posix.as_posix() != relative
    ):
        raise ValueError(f"unsafe sidecar path in prefix manifest: {relative!r}")
    root = prefix_dir.resolve()
    candidate = root
    for part in posix.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(
                f"sidecar path must be a regular file, not a symlink: {relative!r}"
            )
    path = candidate.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"sidecar path escapes prefix directory: {relative!r}")
    return path


def _numpy_dtype(name: str) -> np.dtype:
    return np.dtype(
        {
            "uint8": "<u1",
            "uint16": "<u2",
            "uint32": "<u4",
            "uint64": "<u8",
            "int32": "<i4",
            "int64": "<i8",
        }[name]
    )


def _read_graph_array(
    path: Path, *, dtype: str, item_count: int, width: int
) -> np.ndarray:
    if item_count == 0:
        return np.empty((0, width), dtype=_numpy_dtype(dtype))
    return np.memmap(
        path,
        mode="r",
        dtype=_numpy_dtype(dtype),
        shape=(item_count, width),
    )


def _validate_token_semantics(
    *,
    token_path: Path,
    token_dtype: str,
    lengths: list[int],
    sidecar_files: dict[str, Path],
) -> dict[str, int]:
    token_count = sum(lengths)
    tokens = np.memmap(
        token_path,
        mode="r",
        dtype=_numpy_dtype(token_dtype),
        shape=(token_count,),
    )
    domain_ids = np.memmap(
        sidecar_files["token_domain_ids"],
        mode="r",
        dtype="<u2",
        shape=(token_count,),
    )
    role_ids = np.memmap(
        sidecar_files["token_role_ids"],
        mode="r",
        dtype="<u2",
        shape=(token_count,),
    )
    confidence_ids = np.memmap(
        sidecar_files["token_confidence_ids"],
        mode="r",
        dtype="<u1",
        shape=(token_count,),
    )
    source_ids = np.memmap(
        sidecar_files["token_source_doc_ids"],
        mode="r",
        dtype="<u4",
        shape=(token_count,),
    )
    source_identity_ids = np.memmap(
        sidecar_files["token_source_identity_ids"],
        mode="r",
        dtype="<u8",
        shape=(token_count,),
    )
    loss_mask = np.memmap(
        sidecar_files["loss_mask"],
        mode="r",
        dtype="<u1",
        shape=(token_count,),
    )
    document_ids = np.memmap(
        sidecar_files["doc_ids"],
        mode="r",
        dtype="<u4",
        shape=(token_count,),
    )
    valid_domains = frozenset(
        int(value) for value in CANONICAL_DOMAIN_SCHEMA["domain_kinds"].values()
    )
    valid_roles = frozenset(
        int(value) for value in CANONICAL_DOMAIN_SCHEMA["role_kinds"].values()
    )
    valid_confidences = frozenset(
        int(value) for value in CANONICAL_DOMAIN_SCHEMA["confidence_kinds"].values()
    )
    delimiter_ids = np.asarray(sorted(_DELIMITER_BY_TOKEN_ID), dtype=tokens.dtype)
    delimiter_count = 0
    balanced_pairs = 0
    trained_token_count = 0
    start = 0
    for sequence_index, length in enumerate(lengths):
        end = start + length
        sequence_loss_mask = loss_mask[start:end]
        sequence_document_ids = document_ids[start:end]
        document_transitions = np.diff(
            sequence_document_ids.astype(np.int64, copy=False)
        )
        if (
            sequence_document_ids.size == 0
            or int(sequence_document_ids[0]) != 1
            or np.any((document_transitions != 0) & (document_transitions != 1))
        ):
            raise ValueError(
                "doc_ids must be positive contiguous, non-reused sequence-local "
                f"IDs 1..N in sequence {sequence_index}"
            )
        if np.any((sequence_loss_mask != 0) & (sequence_loss_mask != 1)):
            raise ValueError(f"loss_mask must be binary in sequence {sequence_index}")
        if int(sequence_loss_mask[-1]) != 0:
            raise ValueError(
                "source_token_predicts_next_v1 requires the final loss mask "
                f"to be zero in sequence {sequence_index}"
            )
        leaking = np.flatnonzero(
            (sequence_document_ids[:-1] != sequence_document_ids[1:])
            & (sequence_loss_mask[:-1] != 0)
        )
        if leaking.size:
            raise ValueError(
                "loss_mask trains cross-document transitions in sequence "
                f"{sequence_index}: source_positions={leaking[:16].tolist()}"
            )
        trained_token_count += int(sequence_loss_mask.sum(dtype=np.int64))
        sequence_sources = source_ids[start:end]
        if np.any(sequence_sources == 0):
            raise ValueError(
                "token_source_doc_ids must be positive for every valid token: "
                f"sequence={sequence_index}"
            )
        if np.any(source_identity_ids[start:end] == 0):
            raise ValueError(
                "token_source_identity_ids must be positive for every valid token: "
                f"sequence={sequence_index}"
            )
        sequence_domains = domain_ids[start:end]
        sequence_roles = role_ids[start:end]
        sequence_confidences = confidence_ids[start:end]
        if any(
            int(value) not in valid_domains for value in np.unique(sequence_domains)
        ):
            raise ValueError(f"unknown token_domain_ids in sequence {sequence_index}")
        if any(int(value) not in valid_roles for value in np.unique(sequence_roles)):
            raise ValueError(f"unknown token_role_ids in sequence {sequence_index}")
        if any(
            int(value) not in valid_confidences
            for value in np.unique(sequence_confidences)
        ):
            raise ValueError(
                f"unknown token_confidence_ids in sequence {sequence_index}"
            )
        sequence_tokens = tokens[start:end]
        marker_mask = np.isin(sequence_tokens, delimiter_ids)
        if np.any((sequence_roles == 1) & ~marker_mask):
            raise ValueError(
                f"DELIMITER role on non-delimiter token in sequence {sequence_index}"
            )
        stack: list[tuple[int, int]] = []
        for local_position in range(length):
            token_id = int(sequence_tokens[local_position])
            if token_id not in _DELIMITER_BY_TOKEN_ID:
                active_domain = stack[-1][1] if stack else 0
                if int(sequence_domains[local_position]) != active_domain:
                    raise ValueError(
                        "token domain does not match active delimiter scope in "
                        f"sequence {sequence_index} at token {local_position}: "
                        f"domain={int(sequence_domains[local_position])} "
                        f"active={active_domain}"
                    )
                continue
            expected_domain, is_start, counterpart = _DELIMITER_BY_TOKEN_ID[token_id]
            if int(sequence_domains[local_position]) != expected_domain:
                raise ValueError(
                    f"delimiter token ID {token_id} has wrong domain in "
                    f"sequence {sequence_index}"
                )
            if int(sequence_roles[local_position]) != 1:
                raise ValueError(
                    f"delimiter token ID {token_id} must have DELIMITER role"
                )
            if int(sequence_confidences[local_position]) != 4:
                raise ValueError(
                    f"delimiter token ID {token_id} must have EXACT confidence"
                )
            delimiter_count += 1
            if is_start:
                stack.append((counterpart, expected_domain))
            elif not stack or stack[-1][0] != token_id:
                raise ValueError(
                    f"crossing or unmatched delimiter token ID {token_id} in "
                    f"sequence {sequence_index}"
                )
            else:
                stack.pop()
                balanced_pairs += 1
        if stack:
            raise ValueError(
                f"unclosed domain delimiter pairs in sequence {sequence_index}"
            )
        start = end
    return {
        "token_count": token_count,
        "minimum_source_doc_id": int(source_ids.min()),
        "minimum_source_identity_id": int(source_identity_ids.min()),
        "delimiter_count": delimiter_count,
        "balanced_delimiter_pairs": balanced_pairs,
        "trained_token_count": trained_token_count,
    }


def _validate_graph_provenance(
    *,
    lengths: list[int],
    document_id_path: Path,
    source_document_offsets: list[int],
    graph_files: dict[str, tuple[list[int], Path, dict]],
) -> dict[str, object]:
    sequence_starts = np.cumsum([0, *lengths[:-1]], dtype=np.int64)
    document_ids = np.memmap(
        document_id_path,
        mode="r",
        dtype="<u4",
        shape=(sum(lengths),),
    )
    chunk_names = (
        "token_chunk_starts",
        "token_chunk_ends",
        "token_chunk_kinds",
        "token_chunk_dep_levels",
    )
    chunk_offsets = graph_files["token_chunk_starts"][0]
    for name in chunk_names[1:]:
        if graph_files[name][0] != chunk_offsets:
            raise ValueError("graph chunk CSR offsets disagree by sequence")
    chunk_arrays = {
        name: _read_graph_array(
            graph_files[name][1],
            dtype=str(graph_files[name][2]["dtype"]),
            item_count=int(graph_files[name][2]["item_count"]),
            width=1,
        ).reshape(-1)
        for name in chunk_names
    }
    graph_arrays = {
        name: _read_graph_array(
            graph_files[name][1],
            dtype=str(graph_files[name][2]["dtype"]),
            item_count=int(graph_files[name][2]["item_count"]),
            width=int(graph_files[name][2]["shape_tail"][0]),
        )
        for name in ROUTE_GRAPH_SIDECARS
    }
    route_counts = {name: 0 for name in ROUTE_GRAPH_SIDECARS}
    for sequence_index, length in enumerate(lengths):
        sequence_start = int(sequence_starts[sequence_index])
        sequence_docs = document_ids[sequence_start : sequence_start + length]
        transitions = np.diff(sequence_docs.astype(np.int64, copy=False))
        if (
            sequence_docs.size == 0
            or int(sequence_docs[0]) != 1
            or np.any((transitions != 0) & (transitions != 1))
        ):
            raise ValueError(
                "doc_ids must be positive contiguous, non-reused sequence-local "
                f"IDs 1..N in sequence {sequence_index}"
            )
        segment_count = int(sequence_docs[-1])
        expected_segment_count = int(
            source_document_offsets[sequence_index + 1]
            - source_document_offsets[sequence_index]
        )
        if segment_count != expected_segment_count:
            raise ValueError(
                f"doc_ids cover {segment_count} attention segments but source "
                f"platform sequence CSR declares {expected_segment_count} in "
                f"sequence {sequence_index}"
            )
        chunk_begin, chunk_end = (
            int(chunk_offsets[sequence_index]),
            int(chunk_offsets[sequence_index + 1]),
        )
        starts = chunk_arrays["token_chunk_starts"][chunk_begin:chunk_end]
        ends = chunk_arrays["token_chunk_ends"][chunk_begin:chunk_end]
        kinds = chunk_arrays["token_chunk_kinds"][chunk_begin:chunk_end]
        levels = chunk_arrays["token_chunk_dep_levels"][chunk_begin:chunk_end]
        if (
            np.any(starts < 0)
            or np.any(ends <= starts)
            or np.any(ends > length)
            or np.any(levels < 0)
        ):
            raise ValueError(
                f"invalid graph chunk spans in sequence {sequence_index}"
            )
        if np.any(kinds < int(GraphChunkKind.OTHER)) or np.any(
            kinds >= GRAPH_CHUNK_KIND_COUNT
        ):
            raise ValueError(
                "graph chunk kind is outside the canonical range "
                f"[0,{GRAPH_CHUNK_KIND_COUNT}) in sequence {sequence_index}"
            )
        if len(starts) > 1 and np.any(starts[1:] < ends[:-1]):
            raise ValueError(
                "graph chunks must be ordered and nonoverlapping in "
                f"sequence {sequence_index}"
            )
        chunk_docs: list[int] = []
        for local_start, local_end in zip(starts, ends, strict=True):
            docs = document_ids[
                sequence_start + int(local_start) : sequence_start + int(local_end)
            ]
            if docs.size == 0 or np.any(docs != docs[0]):
                raise ValueError(
                    "graph chunk crosses an attention-document boundary in "
                    f"sequence {sequence_index}"
                )
            chunk_docs.append(int(docs[0]))

        for name in ROUTE_GRAPH_SIDECARS:
            offsets, _path, _spec = graph_files[name]
            begin, end = int(offsets[sequence_index]), int(offsets[sequence_index + 1])
            edges = graph_arrays[name][begin:end]
            route_counts[name] += len(edges)
            if name in {"token_call_edges", "token_type_edges"}:
                for source, target in edges:
                    source_i, target_i = int(source), int(target)
                    if (
                        source_i < 0
                        or target_i < 0
                        or source_i >= len(chunk_docs)
                        or target_i >= len(chunk_docs)
                    ):
                        raise ValueError(
                            f"{name} endpoint exceeds chunk count in "
                            f"sequence {sequence_index}"
                        )
                    if chunk_docs[source_i] != chunk_docs[target_i]:
                        raise ValueError(
                            f"{name} endpoint document provenance mismatch in "
                            f"sequence {sequence_index}"
                        )
            else:
                allowed_kinds = _ALLOWED_EDGE_KINDS[name]
                for source, target, kind in edges:
                    source_i, target_i, kind_i = (
                        int(source),
                        int(target),
                        int(kind),
                    )
                    if (
                        source_i < 0
                        or target_i < 0
                        or source_i >= length
                        or target_i >= length
                    ):
                        raise ValueError(
                            f"{name} endpoint exceeds token count in "
                            f"sequence {sequence_index}"
                        )
                    if kind_i not in allowed_kinds:
                        raise ValueError(f"edge kind {kind_i} is not valid for {name}")
                    if (
                        document_ids[sequence_start + source_i]
                        != document_ids[sequence_start + target_i]
                    ):
                        raise ValueError(
                            f"{name} crosses an attention-document boundary in "
                            f"sequence {sequence_index}"
                        )
    active = {name: count for name, count in route_counts.items() if count > 0}
    if not active:
        raise ValueError("graph bundle has no supported nonempty route edge")
    return {"route_edge_count": sum(active.values()), "route_edge_counts": active}


def _source_identity_id(source: str) -> int:
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    if value == 0:
        value = int.from_bytes(digest[8:16], "big", signed=False)
    if value == 0:
        raise ValueError("source identity SHA256 maps to reserved ID 0")
    return value


def _decode_source_identity_key(raw: object) -> int:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError("source identity registry key must be an 8-byte BLOB")
    value = bytes(raw)
    if len(value) != 8:
        raise ValueError("source identity registry key must be an 8-byte BLOB")
    return int.from_bytes(value, "big", signed=False)


def _validate_case5_source_registry(
    *,
    prefix: Path,
    manifest: dict,
    lengths: list[int],
    source_identity_sidecar: Path,
) -> Path:
    receipt = manifest.get(CASE5_RECEIPT_KEY)
    if not isinstance(receipt, dict) or receipt.get("status") != "success":
        raise ValueError(f"{prefix}: successful {CASE5_RECEIPT_KEY} is required")
    expected_receipt = {
        "schema": CASE5_SCHEMA_VERSION,
        "delimiter_contract_sha256": DOMAIN_DELIMITER_CONTRACT_SHA256,
        "domain_schema_sha256": DOMAIN_SCHEMA_SHA256,
        "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
        "domain_route_columns": list(DOMAIN_ROUTE_COLUMNS),
        "graph_route_columns": list(GRAPH_ROUTE_COLUMNS),
        "graph_sidecars_written": True,
        "source_identity_registry_schema": SOURCE_IDENTITY_REGISTRY_SCHEMA,
    }
    drift = {
        key: receipt.get(key)
        for key, expected in expected_receipt.items()
        if receipt.get(key) != expected
    }
    if drift:
        raise ValueError(f"{prefix}: stale CASE5 ingestion receipt fields: {drift}")

    registry = manifest.get("source_identity_registry")
    if not isinstance(registry, dict):
        raise ValueError(f"{prefix}: source_identity_registry receipt is required")
    expected_registry = {
        "schema": SOURCE_IDENTITY_REGISTRY_SCHEMA,
        "id_encoding": "uint64_be",
        "canonical_digest": "sha256",
        "sequence_count": len(lengths),
        "token_foreign_key_sidecar": "token_source_identity_ids",
    }
    registry_drift = {
        key: registry.get(key)
        for key, expected in expected_registry.items()
        if registry.get(key) != expected
    }
    if registry_drift:
        raise ValueError(
            f"{prefix}: invalid source identity registry receipt: {registry_drift}"
        )
    registry_path = _safe_prefix_file(prefix.parent, str(registry.get("path", "")))
    if registry_path.is_symlink() or not registry_path.is_file():
        raise FileNotFoundError(registry_path)

    token_count = sum(lengths)
    token_ids = np.memmap(
        source_identity_sidecar,
        mode="r",
        dtype="<u8",
        shape=(token_count,),
    )
    connection = sqlite3.connect(
        f"{registry_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        connection.execute("PRAGMA query_only = ON")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise ValueError(f"{prefix}: source identity registry integrity failed")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        required_tables = {"source_identities", "sequence_source_identities"}
        if not required_tables.issubset(tables):
            raise ValueError(
                f"{prefix}: source identity registry tables missing: "
                f"{sorted(required_tables - tables)}"
            )
        foreign_key_errors = list(connection.execute("PRAGMA foreign_key_check"))
        if foreign_key_errors:
            raise ValueError(
                f"{prefix}: source identity registry foreign-key check failed"
            )
        missing_witnesses = int(
            connection.execute(
                "SELECT COUNT(*) FROM sequence_source_identities AS refs "
                "LEFT JOIN source_identities AS witnesses "
                "ON witnesses.source_identity_id = refs.source_identity_id "
                "WHERE witnesses.source_identity_id IS NULL"
            ).fetchone()[0]
        )
        if missing_witnesses:
            raise ValueError(
                f"{prefix}: source identity registry has {missing_witnesses} "
                "references without canonical witnesses"
            )

        identity_count = int(
            connection.execute("SELECT COUNT(*) FROM source_identities").fetchone()[0]
        )
        reference_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM sequence_source_identities"
            ).fetchone()[0]
        )
        if identity_count != int(registry.get("identity_count", -1)):
            raise ValueError(f"{prefix}: source identity registry count mismatch")
        if reference_count != int(
            registry.get("sequence_identity_reference_count", -1)
        ):
            raise ValueError(f"{prefix}: source identity reference count mismatch")

        witness_count = 0
        for raw_id, digest_text, source in connection.execute(
            "SELECT source_identity_id, canonical_sha256, source FROM source_identities"
        ):
            identity_id = _decode_source_identity_key(raw_id)
            if not isinstance(source, str) or not source:
                raise ValueError(f"{prefix}: empty canonical source identity witness")
            actual_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if digest_text != actual_digest or identity_id != _source_identity_id(
                source
            ):
                raise ValueError(
                    f"{prefix}: invalid source identity witness {identity_id}"
                )
            witness_count += 1
        if witness_count != identity_count:
            raise ValueError(f"{prefix}: source identity witness count mismatch")

        reference_rows = iter(
            connection.execute(
                "SELECT sequence_index, source_identity_id "
                "FROM sequence_source_identities "
                "ORDER BY sequence_index, source_identity_id"
            )
        )
        pending = next(reference_rows, None)
        start = 0
        for sequence_index, length in enumerate(lengths):
            end = start + length
            expected_ids = {int(value) for value in np.unique(token_ids[start:end])}
            actual_ids: set[int] = set()
            while pending is not None and int(pending[0]) == sequence_index:
                actual_ids.add(_decode_source_identity_key(pending[1]))
                pending = next(reference_rows, None)
            if pending is not None and int(pending[0]) < sequence_index:
                raise ValueError(f"{prefix}: unordered source identity references")
            if actual_ids != expected_ids:
                raise ValueError(
                    f"{prefix}: source identity registry/token mismatch in "
                    f"sequence {sequence_index}"
                )
            start = end
        if pending is not None:
            raise ValueError(f"{prefix}: source identity reference exceeds sequences")
    finally:
        connection.close()
    return registry_path


def _validate_prefix_manifest_contract(prefix: Path) -> tuple[dict, set[Path]]:
    prefix = prefix.resolve()
    manifest_path = prefix.with_suffix(".json")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("tokenizer_contract") != EXPECTED_PREFIX_TOKENIZER_CONTRACT:
        raise ValueError(
            f"{manifest_path}: tokenizer_contract={data.get('tokenizer_contract')!r}, "
            f"expected {EXPECTED_PREFIX_TOKENIZER_CONTRACT!r}"
        )
    if int(data.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"{manifest_path}: vocab_size={data.get('vocab_size')!r}, "
            f"expected {EXPECTED_VOCAB_SIZE}"
        )
    token_count = int(data.get("token_count", -1))
    document_count = int(data.get("document_count", -1))
    if token_count <= 0 or document_count <= 0:
        raise ValueError(
            f"{manifest_path}: token_count/document_count must be positive"
        )
    token_path = prefix.with_suffix(".bin")
    index_path = prefix.with_suffix(".idx")
    if (
        token_path.is_symlink()
        or index_path.is_symlink()
        or not token_path.is_file()
        or not index_path.is_file()
    ):
        raise FileNotFoundError(f"{prefix}: missing .bin/.idx pair")
    token_dtype = str(data.get("dtype", ""))
    if token_dtype not in DTYPE_SIZES:
        raise ValueError(f"{manifest_path}: unsupported token dtype {token_dtype!r}")
    if token_path.stat().st_size != token_count * DTYPE_SIZES[token_dtype]:
        raise ValueError(
            f"{manifest_path}: token binary size does not match token_count"
        )
    index = _read_mmididx(index_path, expected_dtype=token_dtype)
    if index["tokens"] != token_count or index["sequences"] != document_count:
        raise ValueError(f"{manifest_path}: MMIDIDX token/document counts disagree")
    if index["documents"] != document_count + 1:
        raise ValueError(f"{manifest_path}: MMIDIDX document sentinel is missing")
    lengths = [int(value) for value in index["lengths"]]

    referenced = {manifest_path, token_path, index_path}

    side_paths = data.get("side_channel_paths")
    if not isinstance(side_paths, dict):
        raise ValueError(f"{manifest_path}: side_channel_paths must be an object")
    if data.get("loss_mask_alignment") != EXPECTED_LOSS_MASK_ALIGNMENT:
        raise ValueError(
            f"{manifest_path}: loss_mask_alignment="
            f"{data.get('loss_mask_alignment')!r}, expected "
            f"{EXPECTED_LOSS_MASK_ALIGNMENT!r}"
        )
    missing_sidecars = sorted(REQUIRED_TOKEN_SIDECARS - set(side_paths))
    if missing_sidecars:
        raise ValueError(f"{manifest_path}: missing token sidecars {missing_sidecars}")
    sidecar_files: dict[str, Path] = {}
    for name in REQUIRED_TOKEN_SIDECARS:
        spec = side_paths[name]
        if not isinstance(spec, dict):
            raise ValueError(f"{manifest_path}: token sidecar {name} must be an object")
        dtype = str(spec.get("dtype", ""))
        expected_dtype = TOKEN_SIDECAR_DTYPES[name]
        if dtype != expected_dtype:
            raise ValueError(
                f"{manifest_path}: token sidecar {name} dtype {dtype!r} "
                f"!= {expected_dtype!r}"
            )
        side_path = _safe_prefix_file(prefix.parent, str(spec.get("path", "")))
        if not side_path.is_file():
            raise FileNotFoundError(side_path)
        expected_bytes = token_count * DTYPE_SIZES[dtype]
        if side_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"{manifest_path}: token sidecar {name} size "
                f"{side_path.stat().st_size} != {expected_bytes}"
            )
        if name == "token_structure_ids" and not _contains_nonzero_byte(side_path):
            raise ValueError(
                f"{manifest_path}: token_structure_ids must contain nonzero values"
            )
        referenced.add(side_path)
        sidecar_files[name] = side_path
    if data.get("symbol_identity_schema_version") != 3:
        raise ValueError(
            f"{manifest_path}: symbol_identity_schema_version=3 is required"
        )

    graph_paths = data.get("graph_sidecar_paths")
    if data.get("graph_sidecar_schema") != EXPECTED_GRAPH_SIDECAR_SCHEMA:
        raise ValueError(
            f"{manifest_path}: graph_sidecar_schema={data.get('graph_sidecar_schema')!r}, "
            f"expected {EXPECTED_GRAPH_SIDECAR_SCHEMA!r}"
        )
    if not isinstance(graph_paths, dict):
        raise ValueError(f"{manifest_path}: graph_sidecar_paths must be an object")
    graph_keys = set(graph_paths)
    if graph_keys != REQUIRED_GRAPH_SIDECARS:
        raise ValueError(
            f"{manifest_path}: graph sidecar key set must be exact; "
            f"missing={sorted(REQUIRED_GRAPH_SIDECARS - graph_keys)} "
            f"unexpected={sorted(graph_keys - REQUIRED_GRAPH_SIDECARS)}"
        )
    graph_item_counts: dict[str, int] = {}
    graph_files: dict[str, tuple[list[int], Path, dict]] = {}
    for name in REQUIRED_GRAPH_SIDECARS:
        spec = graph_paths[name]
        if not isinstance(spec, dict):
            raise ValueError(f"{manifest_path}: graph sidecar {name} must be an object")
        expected_kind, expected_dtype, expected_shape_tail = GRAPH_SIDECAR_SPECS[name]
        if spec.get("kind") != expected_kind:
            raise ValueError(f"{manifest_path}: graph sidecar {name} has bad kind")
        expected_coordinate_space = GRAPH_ROUTE_COORDINATE_SPACES[name]
        if spec.get("coordinate_space") != expected_coordinate_space:
            raise ValueError(
                f"{manifest_path}: graph sidecar {name} coordinate_space "
                f"{spec.get('coordinate_space')!r} != {expected_coordinate_space!r}"
            )
        dtype = str(spec.get("dtype", ""))
        if dtype != expected_dtype:
            raise ValueError(
                f"{manifest_path}: graph sidecar {name} has bad dtype {dtype!r}"
            )
        if spec.get("offset_dtype") != "int64":
            raise ValueError(
                f"{manifest_path}: graph sidecar {name} offsets must be int64"
            )
        if spec.get("shape_tail") != expected_shape_tail:
            raise ValueError(f"{manifest_path}: graph sidecar {name} bad shape_tail")
        tail = expected_shape_tail[0]
        item_count = int(spec.get("item_count", -1))
        if item_count < 0:
            raise ValueError(
                f"{manifest_path}: graph sidecar {name} has bad item_count"
            )
        graph_item_counts[name] = item_count
        if name in NONZERO_GRAPH_SIDECARS and item_count <= 0:
            raise ValueError(
                f"{manifest_path}: graph sidecar {name} must be nonzero for H200 ingress"
            )
        offsets_path = _safe_prefix_file(
            prefix.parent, str(spec.get("offsets_path", ""))
        )
        data_path = _safe_prefix_file(prefix.parent, str(spec.get("data_path", "")))
        if not offsets_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path}: graph sidecar {name} files missing"
            )
        offsets = _read_int64_offsets(offsets_path, document_count + 1)
        if offsets[0] != 0 or offsets[-1] != item_count:
            raise ValueError(
                f"{manifest_path}: graph sidecar {name} CSR offsets do not "
                f"span item_count={item_count}"
            )
        if any(left > right for left, right in zip(offsets, offsets[1:])):
            raise ValueError(
                f"{manifest_path}: graph sidecar {name} CSR offsets decrease"
            )
        expected_data_bytes = item_count * tail * DTYPE_SIZES[dtype]
        if data_path.stat().st_size != expected_data_bytes:
            raise ValueError(
                f"{manifest_path}: graph sidecar {name} data size "
                f"{data_path.stat().st_size} != {expected_data_bytes}"
            )
        referenced.update((offsets_path, data_path))
        graph_files[name] = (offsets, data_path, spec)
    chunk_item_counts = {
        graph_item_counts[name]
        for name in (
            "token_chunk_starts",
            "token_chunk_ends",
            "token_chunk_kinds",
            "token_chunk_dep_levels",
        )
    }
    if len(chunk_item_counts) != 1:
        raise ValueError(f"{manifest_path}: graph chunk CSR item counts disagree")
    token_semantics = _validate_token_semantics(
        token_path=token_path,
        token_dtype=token_dtype,
        lengths=lengths,
        sidecar_files=sidecar_files,
    )
    if int(data.get("trained_token_count", -1)) != token_semantics[
        "trained_token_count"
    ]:
        raise ValueError(
            f"{manifest_path}: trained_token_count does not match loss_mask"
        )
    source_identity_registry_path = _validate_case5_source_registry(
        prefix=prefix,
        manifest=data,
        lengths=lengths,
        source_identity_sidecar=sidecar_files["token_source_identity_ids"],
    )
    referenced.add(source_identity_registry_path)

    source_platform = data.get("source_platform_sidecar")
    if (
        not isinstance(source_platform, dict)
        or source_platform.get("schema") != "cppmega_source_platform_v1"
    ):
        raise ValueError(
            f"{manifest_path}: compact source platform sidecar missing or invalid"
        )
    source_document_count = int(source_platform.get("source_document_count", -1))
    platform_id_count = int(source_platform.get("platform_id_count", -1))
    if source_document_count <= 0 or platform_id_count <= 0:
        raise ValueError(f"{manifest_path}: invalid source platform counts")
    sequence_offsets_path = _safe_prefix_file(
        prefix.parent, str(source_platform.get("sequence_doc_offsets_path", ""))
    )
    document_offsets_path = _safe_prefix_file(
        prefix.parent, str(source_platform.get("doc_platform_offsets_path", ""))
    )
    platform_ids_path = _safe_prefix_file(
        prefix.parent, str(source_platform.get("platform_ids_path", ""))
    )
    if not all(
        path.is_file()
        for path in (sequence_offsets_path, document_offsets_path, platform_ids_path)
    ):
        raise FileNotFoundError(f"{manifest_path}: source platform files missing")
    sequence_offsets = _read_int64_offsets(sequence_offsets_path, document_count + 1)
    document_offsets = _read_int64_offsets(
        document_offsets_path, source_document_count + 1
    )
    for label, offsets, final in (
        ("sequence", sequence_offsets, source_document_count),
        ("document", document_offsets, platform_id_count),
    ):
        if offsets[0] != 0 or offsets[-1] != final:
            raise ValueError(
                f"{manifest_path}: source platform {label} CSR bounds mismatch"
            )
        if any(left > right for left, right in zip(offsets, offsets[1:])):
            raise ValueError(
                f"{manifest_path}: source platform {label} CSR offsets decrease"
            )
    if platform_ids_path.stat().st_size != platform_id_count * 2:
        raise ValueError(f"{manifest_path}: source platform IDs size mismatch")
    referenced.update((sequence_offsets_path, document_offsets_path, platform_ids_path))
    _validate_graph_provenance(
        lengths=lengths,
        document_id_path=sidecar_files["doc_ids"],
        source_document_offsets=sequence_offsets,
        graph_files=graph_files,
    )
    objective = data.get("objective_contract")
    if objective is None:
        raise ValueError(f"{manifest_path}: objective_contract is required")
    validated_objective = validate_materialized_objective_contract(
        objective,
        base_dir=str(prefix.parent),
        document_count=document_count,
    )
    if int(validated_objective.payload["totals"]["samples"]) != document_count:
        raise ValueError(
            f"{manifest_path}: objective samples do not match document_count"
        )
    objective_sidecar = objective["objective_id_sidecar"]
    objective_path = _safe_prefix_file(prefix.parent, str(objective_sidecar["path"]))
    referenced.add(objective_path)
    return data, referenced


def _validate_tokenizer_directory(tokenizer_root: Path) -> set[Path]:
    if tokenizer_root.is_symlink():
        raise ValueError("tokenizer directory must not be a symlink")
    tokenizer_root = tokenizer_root.resolve()
    required = {
        tokenizer_root / "tokenizer.json",
        tokenizer_root / "tokenizer_config.json",
        tokenizer_root / "special_tokens_map.json",
        tokenizer_root / "tokenizer_contract_v1.json",
    }
    entries = set(tokenizer_root.iterdir()) if tokenizer_root.is_dir() else set()
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("tokenizer directory may contain only regular files")
    if not required.issubset(entries):
        raise ValueError("tokenizer directory is missing required regular artifacts")
    contract_path = tokenizer_root / "tokenizer_contract_v1.json"
    if _sha256(contract_path) != CANONICAL_TOKENIZER_CONTRACT_SHA256:
        raise ValueError(
            "tokenizer_contract_v1.json does not match the frozen checked-out contract"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract != CANONICAL_TOKENIZER_CONTRACT:
        raise ValueError("tokenizer contract JSON payload drifted")
    tokenizer = json.loads(
        (tokenizer_root / "tokenizer.json").read_text(encoding="utf-8")
    )
    vocab = (tokenizer.get("model") or {}).get("vocab")
    if not isinstance(vocab, dict) or len(vocab) != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            "tokenizer vocab size mismatch: "
            f"expected {EXPECTED_VOCAB_SIZE}, got "
            f"{len(vocab) if isinstance(vocab, dict) else None}"
        )
    if (
        tokenizer.get("version") != "1.0"
        or (tokenizer.get("model") or {}).get("type") != "BPE"
    ):
        raise ValueError("tokenizer JSON schema must be version 1.0 BPE")
    vocab_ids = list(vocab.values())
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in vocab_ids
    ):
        raise ValueError("tokenizer vocab IDs must be integers")
    if set(vocab_ids) != set(range(EXPECTED_VOCAB_SIZE)):
        raise ValueError("tokenizer vocab IDs must be a complete unique 0..65535 range")

    added_tokens = tokenizer.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise ValueError("tokenizer added_tokens must be a list")
    added_by_id: dict[int, str] = {}
    for entry in added_tokens:
        if not isinstance(entry, dict):
            raise ValueError("tokenizer added token entries must be objects")
        token_id = entry.get("id")
        content = entry.get("content")
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not isinstance(content, str)
            or token_id in added_by_id
        ):
            raise ValueError(
                "tokenizer added token IDs/content must be unique and typed"
            )
        added_by_id[token_id] = content
        if vocab.get(content) != token_id:
            raise ValueError(
                f"tokenizer added token {content!r}={token_id} disagrees with vocab"
            )
    expected_tokens = {
        **{index: token for index, token in enumerate(EXPECTED_TOKENIZER_CORE_TOKENS)},
        **{
            int(token_id): f"<RESERVED_{int(token_id)}>"
            for role, token_id in contract["reserved_role_assignments"].items()
            if not role.startswith("_")
        },
    }
    for token_id, token in expected_tokens.items():
        if vocab.get(token) != token_id or added_by_id.get(token_id) != token:
            raise ValueError(
                f"tokenizer canonical token {token!r} must remain at ID {token_id}"
            )

    expected_specials = {
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
    }
    for filename in ("tokenizer_config.json", "special_tokens_map.json"):
        config = json.loads((tokenizer_root / filename).read_text(encoding="utf-8"))
        if any(config.get(key) != value for key, value in expected_specials.items()):
            raise ValueError(f"{filename}: canonical special-token mapping drifted")
    return entries


def _validate_tokenizer_descriptor(
    bundle: Path, manifest: dict, artifact_by_path: dict[str, dict]
) -> set[Path]:
    descriptor = manifest.get("tokenizer")
    if not isinstance(descriptor, dict):
        raise ValueError("bundle tokenizer descriptor is missing")
    if descriptor.get("contract") != EXPECTED_BUNDLE_TOKENIZER_CONTRACT:
        raise ValueError("bundle tokenizer descriptor contract mismatch")
    if int(descriptor.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE:
        raise ValueError("bundle tokenizer descriptor vocab size mismatch")
    tokenizer_root = _safe_artifact_path(bundle, str(descriptor.get("path", "")))
    records = descriptor.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("bundle tokenizer descriptor has no files")
    canonical = _canonical_artifact_records(records)
    if descriptor.get("artifact_set_sha256") != _artifact_set_sha256(canonical):
        raise ValueError("bundle tokenizer artifact-set SHA-256 mismatch")
    referenced: set[Path] = set()
    for record in canonical:
        relative = str(record["path"])
        if artifact_by_path.get(relative) != record:
            raise ValueError(
                f"tokenizer artifact is not bound by bundle manifest: {relative}"
            )
        path = _safe_artifact_path(bundle, relative)
        if tokenizer_root not in path.parents:
            raise ValueError(f"tokenizer artifact escapes tokenizer root: {relative}")
        referenced.add(path)
    required = _validate_tokenizer_directory(tokenizer_root)
    if not required.issubset(referenced):
        raise ValueError("bundle tokenizer descriptor is missing required artifacts")
    return referenced


def _validate_data_contract_descriptors(
    bundle: Path, manifest: dict, artifact_by_path: dict[str, dict]
) -> set[Path]:
    descriptors = manifest.get("data_contracts")
    if not isinstance(descriptors, dict) or set(descriptors) != {
        "domain_schema",
        "tokenizer_contract",
    }:
        raise ValueError("bundle data_contracts descriptor is missing or incomplete")
    expected = {
        "domain_schema": (
            CANONICAL_DOMAIN_SCHEMA_PATH,
            hashlib.sha256(CANONICAL_DOMAIN_SCHEMA_PATH.read_bytes()).hexdigest(),
        ),
        "tokenizer_contract": (
            CANONICAL_TOKENIZER_CONTRACT_PATH,
            CANONICAL_TOKENIZER_CONTRACT_SHA256,
        ),
    }
    referenced: set[Path] = set()
    for name, (_canonical_path, canonical_sha256) in expected.items():
        descriptor = descriptors[name]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "size",
            "sha256",
        }:
            raise ValueError(f"bundle data contract {name} descriptor is invalid")
        relative = str(descriptor["path"])
        record = artifact_by_path.get(relative)
        path = _safe_artifact_path(bundle, relative)
        if (
            record is None
            or record["size"] != descriptor["size"]
            or record["sha256"] != descriptor["sha256"]
            or descriptor["sha256"] != canonical_sha256
            or not path.is_file()
        ):
            raise ValueError(
                f"bundle data contract {name} is not canonically hash-bound"
            )
        referenced.add(path)
    return referenced


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _objective_sampling_permutation(
    size: int, *, seed: int, components: tuple[object, ...]
) -> list[int]:
    def sort_key(index: int) -> tuple[bytes, int]:
        encoded = json.dumps(
            [seed, *components, index],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).digest(), index

    return sorted(range(size), key=sort_key)


def _validate_objective_sampling_v2_producer(
    producer: object,
    *,
    bucket: int,
    total_rows: int,
    file_count: int,
    source_rows: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], ...]:
    if (
        not isinstance(producer, dict)
        or set(producer) != OBJECTIVE_SAMPLING_V2_PRODUCER_KEYS
    ):
        raise ValueError(
            f"bucket {bucket} objective source replay producer metadata is invalid"
        )
    if (
        producer["name"] != OBJECTIVE_SAMPLING_V2_PRODUCER_NAME
        or producer["version"] != OBJECTIVE_SAMPLING_V2_PRODUCER_VERSION
        or not _is_plain_int(producer["version"])
    ):
        raise ValueError(
            f"bucket {bucket} objective source replay producer is unsupported"
        )
    raw_layout = producer["row_group_rows"]
    if not isinstance(raw_layout, list) or len(raw_layout) != file_count:
        raise ValueError(
            f"bucket {bucket} objective source producer row-group layout is invalid"
        )
    layout: list[tuple[int, ...]] = []
    for shard_index, raw_groups in enumerate(raw_layout):
        if (
            not isinstance(raw_groups, list)
            or not raw_groups
            or any(not _is_plain_int(rows) or rows < 1 for rows in raw_groups)
        ):
            raise ValueError(
                f"bucket {bucket} objective source producer row-group layout "
                f"is invalid for shard {shard_index}"
            )
        groups = tuple(int(rows) for rows in raw_groups)
        if source_rows is not None and sum(groups) != source_rows[shard_index]:
            raise ValueError(
                f"bucket {bucket} objective source producer row-group layout "
                f"drifted for shard {shard_index}"
            )
        layout.append(groups)
    if sum(sum(groups) for groups in layout) != total_rows:
        raise ValueError(
            f"bucket {bucket} objective source producer row-group totals drifted"
        )
    return tuple(layout)


def _objective_sampling_v2_final_cursor(
    *,
    seed: int,
    requested_samples: int,
    record_batch_rows: int,
    row_group_rows: tuple[tuple[int, ...], ...],
) -> dict[str, int]:
    total_rows = sum(sum(groups) for groups in row_group_rows)
    source_index = requested_samples - 1
    epoch, remaining = divmod(source_index, total_rows)
    shard_order = _objective_sampling_permutation(
        len(row_group_rows),
        seed=seed,
        components=("shards", epoch),
    )
    for shard_position, shard_index in enumerate(shard_order):
        shard_rows = sum(row_group_rows[shard_index])
        if remaining < shard_rows:
            break
        remaining -= shard_rows
    else:  # pragma: no cover - validated positive layout makes this unreachable
        raise AssertionError("objective source replay exceeded shard layout")

    shard_row_groups = row_group_rows[shard_index]
    row_group_order = _objective_sampling_permutation(
        len(shard_row_groups),
        seed=seed,
        components=("row_groups", epoch, shard_index),
    )
    for row_group_position, row_group_index in enumerate(row_group_order):
        row_group_size = shard_row_groups[row_group_index]
        if remaining < row_group_size:
            break
        remaining -= row_group_size
    else:  # pragma: no cover - validated positive layout makes this unreachable
        raise AssertionError("objective source replay exceeded row-group layout")

    record_batch_index, row_shuffle_position = divmod(
        remaining, record_batch_rows
    )
    record_batch_start = record_batch_index * record_batch_rows
    record_batch_size = min(
        record_batch_rows,
        row_group_size - record_batch_start,
    )
    row_order = _objective_sampling_permutation(
        record_batch_size,
        seed=seed,
        components=(
            "rows",
            epoch,
            shard_index,
            row_group_index,
            record_batch_index,
        ),
    )
    return {
        "epoch": epoch,
        "shard_position": shard_position,
        "shard_index": shard_index,
        "row_group_position": row_group_position,
        "row_group_index": row_group_index,
        "record_batch_index": record_batch_index,
        "row_shuffle_position": row_shuffle_position,
        "row_index_in_record_batch": row_order[row_shuffle_position],
        "source_index": source_index,
    }


def _validate_objective_source_sampling(
    sampling: object,
    *,
    bucket: int,
    total_rows: int,
    file_count: int,
    source_rows: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if not isinstance(sampling, dict):
        raise ValueError(f"bucket {bucket} objective source sampling is invalid")

    mode = sampling.get("mode")
    if mode == OBJECTIVE_SAMPLING_MODE_V1:
        expected_keys = OBJECTIVE_SAMPLING_BASE_KEYS
    elif mode == OBJECTIVE_SAMPLING_MODE_V2:
        expected_keys = OBJECTIVE_SAMPLING_V2_KEYS
    else:
        raise ValueError(f"bucket {bucket} objective source sampling is unsupported")
    if set(sampling) != expected_keys:
        raise ValueError(f"bucket {bucket} objective source sampling fields drifted")

    integer_fields = (
        "seed",
        "requested_samples",
        "full_passes",
        "tail_rows",
        "min_row_reuse",
        "max_row_reuse",
    )
    if any(not _is_plain_int(sampling[field]) for field in integer_fields):
        raise ValueError(
            f"bucket {bucket} objective source sampling values are invalid"
        )

    requested = sampling["requested_samples"]
    if requested < 1:
        raise ValueError(f"bucket {bucket} objective source sampling drifted")
    full_passes, tail_rows = divmod(requested, total_rows)
    if (
        sampling["full_passes"] != full_passes
        or sampling["tail_rows"] != tail_rows
        or sampling["min_row_reuse"] != full_passes
        or sampling["max_row_reuse"] != full_passes + int(tail_rows > 0)
    ):
        raise ValueError(f"bucket {bucket} objective source sampling drifted")
    if mode == OBJECTIVE_SAMPLING_MODE_V1:
        return sampling

    record_batch_rows = sampling["record_batch_rows"]
    if not _is_plain_int(record_batch_rows) or record_batch_rows < 1:
        raise ValueError(
            f"bucket {bucket} objective source record_batch_rows is invalid"
        )
    if sampling["ordering"] != OBJECTIVE_SAMPLING_V2_ORDERING:
        raise ValueError(
            f"bucket {bucket} objective source deterministic ordering drifted"
        )
    if sampling["cursor_semantics"] != "last_yielded_row_v1":
        raise ValueError(f"bucket {bucket} objective source cursor semantics drifted")
    if source_rows is not None and (
        len(source_rows) != file_count or sum(source_rows) != total_rows
    ):
        raise ValueError(f"bucket {bucket} objective source row counts drifted")
    row_group_rows = _validate_objective_sampling_v2_producer(
        sampling["producer"],
        bucket=bucket,
        total_rows=total_rows,
        file_count=file_count,
        source_rows=source_rows,
    )

    cursor = sampling["final_cursor"]
    if not isinstance(cursor, dict) or set(cursor) != OBJECTIVE_SAMPLING_V2_CURSOR_KEYS:
        raise ValueError(f"bucket {bucket} objective source final_cursor is invalid")
    if any(
        not _is_plain_int(cursor[field]) or cursor[field] < 0
        for field in OBJECTIVE_SAMPLING_V2_CURSOR_KEYS
    ):
        raise ValueError(
            f"bucket {bucket} objective source final_cursor values are invalid"
        )

    expected_cursor = _objective_sampling_v2_final_cursor(
        seed=sampling["seed"],
        requested_samples=requested,
        record_batch_rows=record_batch_rows,
        row_group_rows=row_group_rows,
    )
    if cursor != expected_cursor:
        drift = {
            field: {"actual": cursor[field], "expected": expected_cursor[field]}
            for field in OBJECTIVE_SAMPLING_V2_CURSOR_KEYS
            if cursor[field] != expected_cursor[field]
        }
        raise ValueError(
            f"bucket {bucket} objective source final_cursor replay drifted: {drift}"
        )
    return sampling


def _objective_source_snapshot_summary(
    source_snapshot: object, *, bucket: int
) -> dict[str, object]:
    if not isinstance(source_snapshot, dict):
        raise ValueError(f"bucket {bucket} objective source_snapshot is missing")
    if source_snapshot.get("schema") == "cppmega_objective_source_snapshot_v2":
        expected_keys = {
            "schema",
            "sequence_length",
            "algorithm",
            "pool_order",
            "source_pool_manifest",
            "ci_export_receipt",
            "pools",
        }
        if set(source_snapshot) != expected_keys:
            raise ValueError(f"bucket {bucket} objective source_snapshot is invalid")
        if (
            not _is_plain_int(source_snapshot["sequence_length"])
            or source_snapshot["sequence_length"] != bucket
            or source_snapshot["algorithm"] != "alternate_primary_seed_v1"
            or source_snapshot["pool_order"] != ["primary_ci", "objective_seed"]
        ):
            raise ValueError(
                f"bucket {bucket} objective source_snapshot pool schedule drifted"
            )
        descriptors: dict[str, dict[str, object]] = {}
        for field, expected_path in (
            ("source_pool_manifest", "objective_source_pool_manifest.json"),
            ("ci_export_receipt", "ci_export_receipt.json"),
        ):
            descriptor = source_snapshot[field]
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"path", "size_bytes", "sha256"}
                or descriptor.get("path") != expected_path
                or not _is_plain_int(descriptor.get("size_bytes"))
                or int(descriptor["size_bytes"]) < 1
                or not isinstance(descriptor.get("sha256"), str)
                or not SHA256_RE.fullmatch(descriptor["sha256"])
            ):
                raise ValueError(
                    f"bucket {bucket} objective source_snapshot {field} is invalid"
                )
            descriptors[field] = dict(descriptor)
        pools = source_snapshot["pools"]
        if not isinstance(pools, dict) or set(pools) != {
            "primary_ci",
            "objective_seed",
        }:
            raise ValueError(
                f"bucket {bucket} objective source_snapshot pools are invalid"
            )
        if any(
            not isinstance(pools[name], dict)
            or pools[name].get("schema")
            != "cppmega_objective_source_snapshot_v1"
            for name in ("primary_ci", "objective_seed")
        ):
            raise ValueError(
                f"bucket {bucket} objective source_snapshot pools are invalid"
            )
        pool_summaries = {
            name: _objective_source_snapshot_summary(pools[name], bucket=bucket)
            for name in ("primary_ci", "objective_seed")
        }
        return {
            "schema": source_snapshot["schema"],
            "sequence_length": bucket,
            "algorithm": source_snapshot["algorithm"],
            "pool_order": list(source_snapshot["pool_order"]),
            **descriptors,
            "pools": pool_summaries,
        }
    expected_keys = {
        "schema",
        "sequence_length",
        "file_count",
        "row_count",
        "files",
        "sampling",
        "artifact_set_sha256",
    }
    if set(source_snapshot) != expected_keys:
        raise ValueError(f"bucket {bucket} objective source_snapshot is invalid")
    sequence_length = source_snapshot["sequence_length"]
    if (
        source_snapshot["schema"] != "cppmega_objective_source_snapshot_v1"
        or not _is_plain_int(sequence_length)
        or sequence_length != bucket
    ):
        raise ValueError(f"bucket {bucket} objective source_snapshot schema drifted")
    files = source_snapshot["files"]
    if not isinstance(files, list) or not files:
        raise ValueError(f"bucket {bucket} objective source files are missing")
    records: list[dict[str, object]] = []
    source_paths: list[str] = []
    source_rows: list[int] = []
    total_rows = 0
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
            "rows",
        }:
            raise ValueError(f"bucket {bucket} objective source file is invalid")
        path = record["path"]
        size = record["size_bytes"]
        rows = record["rows"]
        digest = record["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or not _is_plain_int(size)
            or size < 1
            or not _is_plain_int(rows)
            or rows < 1
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
        ):
            raise ValueError(
                f"bucket {bucket} objective source file values are invalid"
            )
        records.append({"path": path, "size": size, "sha256": digest})
        source_paths.append(path)
        source_rows.append(rows)
        total_rows += rows
    if (
        source_paths != sorted(source_paths, key=PurePosixPath)
        or len(source_paths) != len(set(source_paths))
    ):
        raise ValueError(f"bucket {bucket} objective source file ordering drifted")
    file_count = source_snapshot["file_count"]
    row_count = source_snapshot["row_count"]
    if (
        not _is_plain_int(file_count)
        or not _is_plain_int(row_count)
        or file_count != len(files)
        or row_count != total_rows
    ):
        raise ValueError(f"bucket {bucket} objective source counts drifted")
    digest = _artifact_set_sha256(records)
    if source_snapshot["artifact_set_sha256"] != digest:
        raise ValueError(f"bucket {bucket} objective source digest drifted")
    sampling = _validate_objective_source_sampling(
        source_snapshot["sampling"],
        bucket=bucket,
        total_rows=total_rows,
        file_count=len(files),
        source_rows=tuple(source_rows),
    )
    return {
        "schema": source_snapshot["schema"],
        "artifact_set_sha256": digest,
        "file_count": len(files),
        "row_count": total_rows,
        "sampling": sampling,
    }


def _validate_objective_source_summary(summary: object, *, bucket: int) -> None:
    if (
        isinstance(summary, dict)
        and summary.get("schema") == "cppmega_objective_source_snapshot_v2"
    ):
        expected_keys = {
            "schema",
            "sequence_length",
            "algorithm",
            "pool_order",
            "source_pool_manifest",
            "ci_export_receipt",
            "pools",
        }
        if (
            set(summary) != expected_keys
            or not _is_plain_int(summary.get("sequence_length"))
            or summary.get("sequence_length") != bucket
            or summary.get("algorithm") != "alternate_primary_seed_v1"
            or summary.get("pool_order") != ["primary_ci", "objective_seed"]
        ):
            raise ValueError(
                f"bundle objective source_snapshot descriptor is invalid for {bucket}"
            )
        for field, expected_path in (
            ("source_pool_manifest", "objective_source_pool_manifest.json"),
            ("ci_export_receipt", "ci_export_receipt.json"),
        ):
            descriptor = summary.get(field)
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"path", "size_bytes", "sha256"}
                or descriptor.get("path") != expected_path
                or not _is_plain_int(descriptor.get("size_bytes"))
                or int(descriptor["size_bytes"]) < 1
                or not isinstance(descriptor.get("sha256"), str)
                or not SHA256_RE.fullmatch(descriptor["sha256"])
            ):
                raise ValueError(
                    f"bundle objective source_snapshot descriptor drifted for {bucket}"
                )
        pools = summary.get("pools")
        if not isinstance(pools, dict) or set(pools) != {
            "primary_ci",
            "objective_seed",
        }:
            raise ValueError(
                f"bundle objective source_snapshot descriptor drifted for {bucket}"
            )
        if any(
            not isinstance(pools[name], dict)
            or pools[name].get("schema")
            != "cppmega_objective_source_snapshot_v1"
            for name in ("primary_ci", "objective_seed")
        ):
            raise ValueError(
                f"bundle objective source_snapshot descriptor drifted for {bucket}"
            )
        for name in ("primary_ci", "objective_seed"):
            _validate_objective_source_summary(pools[name], bucket=bucket)
        return
    if not isinstance(summary, dict) or set(summary) != {
        "schema",
        "artifact_set_sha256",
        "file_count",
        "row_count",
        "sampling",
    }:
        raise ValueError(
            f"bundle objective source_snapshot descriptor is invalid for {bucket}"
        )
    artifact_set_sha256 = summary["artifact_set_sha256"]
    file_count = summary["file_count"]
    row_count = summary["row_count"]
    if (
        summary["schema"] != "cppmega_objective_source_snapshot_v1"
        or not isinstance(artifact_set_sha256, str)
        or not SHA256_RE.fullmatch(artifact_set_sha256)
        or not _is_plain_int(file_count)
        or file_count < 1
        or not _is_plain_int(row_count)
        or row_count < 1
    ):
        raise ValueError(
            f"bundle objective source_snapshot descriptor drifted for {bucket}"
        )
    _validate_objective_source_sampling(
        summary["sampling"],
        bucket=bucket,
        total_rows=row_count,
        file_count=file_count,
    )


def _validate_embedded_prefix_manifest_contract(
    prefix_manifest: object,
    *,
    prefix_name: str,
    objective_descriptor: dict[str, object],
    artifact_by_path: dict[str, dict],
) -> None:
    if not isinstance(prefix_manifest, dict):
        raise ValueError(f"embedded prefix manifest {prefix_name} must be an object")
    if prefix_manifest.get("tokenizer_contract") != EXPECTED_PREFIX_TOKENIZER_CONTRACT:
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: tokenizer contract is invalid"
        )
    if int(prefix_manifest.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: vocab_size is invalid"
        )
    document_count = int(prefix_manifest.get("document_count", -1))
    token_count = int(prefix_manifest.get("token_count", -1))
    if document_count <= 0 or token_count <= 0:
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: token/document counts must be positive"
        )

    side_channels = prefix_manifest.get("side_channel_paths")
    if not isinstance(side_channels, dict):
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: side_channel_paths must be an object"
        )
    missing_token = sorted(REQUIRED_TOKEN_SIDECARS - set(side_channels))
    if missing_token:
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: missing token sidecars {missing_token}"
        )
    for name in REQUIRED_TOKEN_SIDECARS:
        spec = side_channels[name]
        if not isinstance(spec, dict):
            raise ValueError(
                f"embedded prefix manifest {prefix_name}: token sidecar {name} is invalid"
            )
        _validate_artifact_relative_path(spec.get("path"))
        if spec.get("dtype") != TOKEN_SIDECAR_DTYPES[name]:
            raise ValueError(
                f"embedded prefix manifest {prefix_name}: token sidecar {name} dtype is invalid"
            )

    if prefix_manifest.get("graph_sidecar_schema") != EXPECTED_GRAPH_SIDECAR_SCHEMA:
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: graph_sidecar_schema is invalid"
        )
    graph_paths = prefix_manifest.get("graph_sidecar_paths")
    if not isinstance(graph_paths, dict):
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: graph_sidecar_paths must be an object"
        )
    missing_graph = sorted(REQUIRED_GRAPH_SIDECARS - set(graph_paths))
    if missing_graph:
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: missing graph sidecars {missing_graph}"
        )
    chunk_counts: set[int] = set()
    for name, (expected_kind, expected_dtype, expected_shape) in GRAPH_SIDECAR_SPECS.items():
        spec = graph_paths[name]
        if not isinstance(spec, dict):
            raise ValueError(
                f"embedded prefix manifest {prefix_name}: graph sidecar {name} is invalid"
            )
        if (
            spec.get("kind") != expected_kind
            or spec.get("dtype") != expected_dtype
            or spec.get("offset_dtype") != "int64"
            or spec.get("shape_tail") != expected_shape
            or spec.get("coordinate_space") != GRAPH_ROUTE_COORDINATE_SPACES[name]
        ):
            raise ValueError(
                f"embedded prefix manifest {prefix_name}: graph sidecar {name} shape contract is invalid"
            )
        item_count = spec.get("item_count")
        if not isinstance(item_count, int) or isinstance(item_count, bool) or item_count < 0:
            raise ValueError(
                f"embedded prefix manifest {prefix_name}: graph sidecar {name} item_count is invalid"
            )
        if name in NONZERO_GRAPH_SIDECARS and item_count == 0:
            raise ValueError(
                f"embedded prefix manifest {prefix_name}: graph sidecar {name} must be nonzero"
            )
        if name.startswith("token_chunk_"):
            chunk_counts.add(item_count)
        _validate_artifact_relative_path(spec.get("offsets_path"))
        _validate_artifact_relative_path(spec.get("data_path"))
    if len(chunk_counts) != 1:
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: graph chunk counts disagree"
        )

    source_platform = prefix_manifest.get("source_platform_sidecar")
    if (
        not isinstance(source_platform, dict)
        or source_platform.get("schema") != "cppmega_source_platform_v1"
        or int(source_platform.get("source_document_count", -1)) <= 0
        or int(source_platform.get("platform_id_count", -1)) <= 0
    ):
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: source platform contract is invalid"
        )
    for field in (
        "sequence_doc_offsets_path",
        "doc_platform_offsets_path",
        "platform_ids_path",
    ):
        _validate_artifact_relative_path(source_platform.get(field))

    objective = prefix_manifest.get("objective_contract")
    validated_objective = validate_materialized_objective_contract(objective)
    if int(validated_objective.payload["totals"]["samples"]) != document_count:
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: objective samples do not match document_count"
        )
    if objective.get("sha256") != objective_descriptor["contract_sha256"]:
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: objective contract is not bundle-bound"
        )
    objective_sidecar = objective["objective_id_sidecar"]
    _validate_artifact_relative_path(objective_sidecar["path"])

    artifact_path = objective_descriptor["artifact_path"]
    artifact_record = artifact_by_path.get(str(artifact_path))
    if artifact_record is None or artifact_record.get("sha256") != objective_descriptor[
        "artifact_file_sha256"
    ]:
        raise ValueError(
            f"embedded prefix manifest {prefix_name}: objective artifact is not artifact-bound"
        )


def _validate_logical_manifest_contract(manifest: object) -> None:
    """Reject unsupported bundle contracts without touching artifact payloads."""

    if not isinstance(manifest, dict):
        raise ValueError("bundle logical manifest must be an object")
    if manifest.get("schema") not in {
        "cppmega_megatron_bundle_v3",
        "cppmega_megatron_bundle_v4",
    }:
        raise ValueError(f"unsupported bundle schema: {manifest.get('schema')!r}")
    _require_manifest_tokenizer_contract(manifest)
    if manifest.get("training_contract") != "objective_materialized":
        raise ValueError("bundle training_contract must be 'objective_materialized'")
    if manifest.get("known_limitations") != []:
        raise ValueError("complete bundle known_limitations must be empty")
    validate_implementation_binding(
        manifest.get("implementation"),
        where="bundle implementation",
        required_components=("cppmega", "cppmega_mlx", "clang_indexer"),
    )
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("bundle logical manifest artifacts must be a non-empty list")
    artifact_by_path: dict[str, dict] = {}
    for record in raw_artifacts:
        if not isinstance(record, dict):
            raise ValueError("bundle logical manifest artifact records must be objects")
        relative = _validate_artifact_relative_path(record.get("path"))
        if relative == "manifest.json":
            raise ValueError("bundle artifacts must not include manifest.json")
        if relative in artifact_by_path:
            raise ValueError(f"duplicate artifact path: {relative}")
        size = record.get("size")
        digest = record.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"artifact {relative} has invalid size")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"artifact {relative} is missing a valid sha256")
        artifact_by_path[relative] = record
    if len(raw_artifacts) != int(manifest.get("artifact_count", -1)):
        raise ValueError("bundle artifact_count does not match artifact list")
    if sum(int(record["size"]) for record in raw_artifacts) != int(
        manifest.get("artifact_bytes", -1)
    ):
        raise ValueError("bundle artifact_bytes does not match artifact list")
    artifact_set_sha256 = _artifact_set_sha256(raw_artifacts)
    if manifest.get("artifact_set_sha256") != artifact_set_sha256:
        raise ValueError("bundle artifact_set_sha256 does not match artifact list")
    bundle_id = manifest.get("bundle_id")
    if (
        not isinstance(bundle_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", bundle_id)
        or not bundle_id.endswith(artifact_set_sha256[:16])
    ):
        raise ValueError("bundle_id is not safely bound to artifact_set_sha256")
    source_snapshot = manifest.get("source_snapshot")
    if not isinstance(source_snapshot, dict):
        raise ValueError("bundle source_snapshot descriptor is missing")
    source_composition = source_snapshot.get("source_composition")
    if (
        not isinstance(source_composition, dict)
        or set(source_composition)
        != {
            "schema",
            "receipt",
            "plan",
            "dedup_receipt",
            "dedup_verifier",
            "runs",
        }
        or source_composition.get("schema")
        != "cppmega_source_conveyor_composition_v1"
    ):
        raise ValueError("bundle source composition descriptor is invalid")

    bound_paths: set[str] = set()

    def validate_source_artifact(
        binding: object,
        *,
        where: str,
        with_size: bool,
    ) -> str:
        if not isinstance(binding, dict):
            raise ValueError(f"{where} binding must be an object")
        expected_fields = {"path", "sha256"}
        if with_size:
            expected_fields.add("size_bytes")
        if set(binding) != expected_fields:
            raise ValueError(f"{where} binding fields drifted")
        relative = _validate_artifact_relative_path(binding.get("path"))
        if not relative.startswith("provenance/source_composition/"):
            raise ValueError(f"{where} escapes source composition provenance")
        if relative in bound_paths:
            raise ValueError(f"duplicate source composition artifact: {relative}")
        bound_paths.add(relative)
        digest = binding.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"{where} SHA-256 is invalid")
        record = artifact_by_path.get(relative)
        if record is None or record.get("sha256") != digest:
            raise ValueError(f"{where} is not bundle-artifact-bound")
        if with_size:
            size = binding.get("size_bytes")
            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or record.get("size") != size
            ):
                raise ValueError(f"{where} size is not bundle-artifact-bound")
        return relative

    for name in ("receipt", "plan", "dedup_receipt", "dedup_verifier"):
        validate_source_artifact(
            source_composition[name],
            where=f"source composition {name}",
            with_size=False,
        )
    raw_runs = source_composition.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("bundle source composition has no runs")
    run_ids: set[str] = set()
    required_run_artifacts = {
        "launch",
        "exit",
        "manifest",
        "archive_sha256_receipt",
        "archive_inventory",
        "repo_list",
        "source_quarantine_manifest",
        "tokenizer",
    }
    for raw_run in raw_runs:
        if not isinstance(raw_run, dict) or set(raw_run) != {"run_id", "artifacts"}:
            raise ValueError("bundle source composition run is malformed")
        run_id = raw_run.get("run_id")
        if (
            not isinstance(run_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id)
            or run_id in run_ids
        ):
            raise ValueError("bundle source composition run_id is invalid")
        run_ids.add(run_id)
        run_artifacts = raw_run.get("artifacts")
        if (
            not isinstance(run_artifacts, dict)
            or set(run_artifacts) != required_run_artifacts
        ):
            raise ValueError(f"bundle source composition run {run_id} artifacts drifted")
        for name, binding in run_artifacts.items():
            validate_source_artifact(
                binding,
                where=f"source composition run {run_id}/{name}",
                with_size=True,
            )
    source_routes = source_snapshot.get("source_routes")
    if manifest.get("schema") == "cppmega_megatron_bundle_v4":
        if (
            not isinstance(source_routes, dict)
            or set(source_routes) != {"schema", "policy", "routes"}
            or source_routes.get("schema")
            != "cppmega_packed_source_primary_routes_v1"
            or source_routes.get("policy")
            != "primary-only-code-and-commit-snapshot"
            or not isinstance(source_routes.get("routes"), dict)
            or set(source_routes["routes"]) != {"code", "commits"}
        ):
            raise ValueError("bundle source route descriptor is invalid")
        route_fields = {
            "path",
            "sha256",
            "route_schema",
            "input_inventory_sha256",
            "output_inventory_sha256",
            "primary",
        }
        for kind, binding in source_routes["routes"].items():
            if (
                not isinstance(binding, dict)
                or set(binding) != route_fields
                or binding.get("route_schema") != "cppmega_packed_source_route_v2"
            ):
                raise ValueError(f"bundle {kind} source route binding drifted")
            relative = _validate_artifact_relative_path(binding.get("path"))
            if not relative.startswith("provenance/source_routes/"):
                raise ValueError(f"bundle {kind} source route escapes provenance")
            digest = binding.get("sha256")
            if (
                not isinstance(digest, str)
                or not SHA256_RE.fullmatch(digest)
                or artifact_by_path.get(relative, {}).get("sha256") != digest
            ):
                raise ValueError(
                    f"bundle {kind} source route is not artifact-bound"
                )
            for field in ("input_inventory_sha256", "output_inventory_sha256"):
                if not SHA256_RE.fullmatch(str(binding.get(field, ""))):
                    raise ValueError(f"bundle {kind} source route {field} is invalid")
            primary = binding.get("primary")
            if (
                not isinstance(primary, dict)
                or set(primary)
                != {
                    "rows",
                    "valid_tokens",
                    "trained_tokens",
                    "documents",
                    "capacity_tokens",
                }
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in primary.values()
                )
                or any(
                    primary[name] < 1
                    for name in (
                        "rows",
                        "valid_tokens",
                        "trained_tokens",
                        "documents",
                    )
                )
            ):
                raise ValueError(f"bundle {kind} source route totals are invalid")
    objective_materialization = manifest.get("objective_materialization")
    if (
        not isinstance(objective_materialization, dict)
        or objective_materialization.get("schema")
        != "cppmega_bucketed_objective_materializations_v1"
        or not isinstance(objective_materialization.get("buckets"), dict)
        or not objective_materialization["buckets"]
    ):
        raise ValueError("bundle objective_materialization descriptor is invalid")
    required_objective_fields = {
        "artifact_path",
        "artifact_schema",
        "artifact_set_sha256",
        "artifact_file_sha256",
        "contract_path",
        "contract_schema",
        "contract_sha256",
        "contract_file_sha256",
        "source_snapshot",
    }
    for bucket, descriptor in objective_materialization["buckets"].items():
        if not isinstance(bucket, str) or not bucket.isdecimal():
            raise ValueError("objective materialization bucket keys must be integers")
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != required_objective_fields
        ):
            raise ValueError(
                f"bundle objective materialization descriptor is invalid for {bucket}"
            )
        if descriptor["artifact_schema"] == (
            LEGACY_OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA
        ):
            raise ValueError(
                f"legacy objective artifact schema for {bucket}; migration required: "
                "regenerate the objective artifact and bundle"
            )
        if descriptor["artifact_schema"] != OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported objective artifact schema for {bucket}")
        if descriptor["contract_schema"] != "cppmega_pre_materialized_objectives_v1":
            raise ValueError(f"unsupported objective contract schema for {bucket}")
        artifact_path = _validate_artifact_relative_path(descriptor["artifact_path"])
        contract_path = _validate_artifact_relative_path(descriptor["contract_path"])
        for field in (
            "artifact_set_sha256",
            "artifact_file_sha256",
            "contract_sha256",
            "contract_file_sha256",
        ):
            if not SHA256_RE.fullmatch(str(descriptor.get(field, ""))):
                raise ValueError(
                    f"objective materialization {bucket}.{field} is invalid"
                )
        if artifact_by_path.get(artifact_path, {}).get("sha256") != descriptor[
            "artifact_file_sha256"
        ]:
            raise ValueError(
                f"objective materialization {bucket} artifact is not artifact-bound"
            )
        if artifact_by_path.get(contract_path, {}).get("sha256") != descriptor[
            "contract_file_sha256"
        ]:
            raise ValueError(
                f"objective materialization {bucket} contract is not artifact-bound"
            )
        _validate_objective_source_summary(
            descriptor["source_snapshot"], bucket=int(bucket)
        )

    tokenizer = manifest.get("tokenizer")
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("contract") != EXPECTED_BUNDLE_TOKENIZER_CONTRACT
        or int(tokenizer.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE
        or not isinstance(tokenizer.get("files"), list)
        or not tokenizer["files"]
    ):
        raise ValueError("bundle tokenizer descriptor is missing")
    tokenizer_path = _validate_artifact_relative_path(tokenizer.get("path"))
    tokenizer_records: list[dict[str, object]] = []
    for record in tokenizer["files"]:
        if not isinstance(record, dict):
            raise ValueError("bundle tokenizer descriptor file must be an object")
        relative = _validate_artifact_relative_path(record.get("path"))
        if relative != tokenizer_path and not relative.startswith(tokenizer_path + "/"):
            raise ValueError("bundle tokenizer descriptor file escapes tokenizer path")
        size = record.get("size")
        digest = record.get("sha256")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
        ):
            raise ValueError("bundle tokenizer descriptor file identity is invalid")
        canonical_record = {"path": relative, "size": size, "sha256": digest}
        if artifact_by_path.get(relative) != canonical_record:
            raise ValueError(
                f"bundle tokenizer descriptor is not artifact-bound: {relative}"
            )
        tokenizer_records.append(canonical_record)
    if tokenizer.get("artifact_set_sha256") != _artifact_set_sha256(tokenizer_records):
        raise ValueError("bundle tokenizer artifact_set_sha256 is invalid")

    data_contracts = manifest.get("data_contracts")
    if not isinstance(data_contracts, dict) or not data_contracts:
        raise ValueError("bundle data_contracts descriptor is missing")
    for name, descriptor in data_contracts.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError("bundle data contract descriptors must be objects")
        relative = _validate_artifact_relative_path(descriptor.get("path"))
        size = descriptor.get("size")
        digest = descriptor.get("sha256")
        canonical_record = {"path": relative, "size": size, "sha256": digest}
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
            or artifact_by_path.get(relative) != canonical_record
        ):
            raise ValueError(f"bundle data contract {name} is not artifact-bound")

    buckets = manifest.get("buckets")
    if (
        not isinstance(buckets, list)
        or not buckets
        or any(not isinstance(bucket, int) or isinstance(bucket, bool) for bucket in buckets)
        or len(buckets) != len(set(buckets))
    ):
        raise ValueError("bundle buckets must be a non-empty unique integer list")
    bucket_results = manifest.get("bucket_results")
    if not isinstance(bucket_results, list) or not bucket_results:
        raise ValueError("bundle bucket_results must be a non-empty list")
    result_buckets: list[int] = []
    for result in bucket_results:
        if not isinstance(result, dict):
            raise ValueError("bundle bucket_results entries must be objects")
        bucket = result.get("bucket")
        if not isinstance(bucket, int) or isinstance(bucket, bool):
            raise ValueError("bundle bucket result has invalid bucket")
        result_buckets.append(bucket)
        prefix_name = _validate_artifact_relative_path(result.get("prefix"))
        descriptor = objective_materialization["buckets"].get(str(bucket))
        if not isinstance(descriptor, dict):
            raise ValueError(f"bundle bucket result has no objective descriptor for {bucket}")
        _validate_embedded_prefix_manifest_contract(
            result.get("manifest"),
            prefix_name=prefix_name,
            objective_descriptor=descriptor,
            artifact_by_path=artifact_by_path,
        )
    if result_buckets != buckets:
        raise ValueError(
            f"bundle bucket_results do not match buckets: {result_buckets} != {buckets}"
        )


def _load_bundle_manifest(bundle: Path) -> tuple[dict, list[dict]]:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_logical_manifest_contract(manifest)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("bundle manifest has no artifacts")
    if len(artifacts) != int(manifest.get("artifact_count", -1)):
        raise ValueError("bundle artifact_count does not match artifact list")
    seen: set[str] = set()
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("bundle artifact records must be objects")
        relative_raw = record.get("path")
        if not isinstance(relative_raw, str):
            raise ValueError("bundle artifact path must be a string")
        relative = relative_raw
        if relative == "manifest.json":
            raise ValueError("bundle artifacts must not include manifest.json")
        if relative in seen:
            raise ValueError(f"duplicate artifact path: {relative}")
        seen.add(relative)
        size = record.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"artifact {relative} has invalid size")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"artifact {relative} is missing a valid sha256")
        _safe_artifact_path(bundle, relative)
    if sum(int(record["size"]) for record in artifacts) != int(
        manifest["artifact_bytes"]
    ):
        raise ValueError("bundle artifact_bytes does not match artifact list")
    canonical = _canonical_artifact_records(artifacts)
    artifact_set_sha256 = _artifact_set_sha256(canonical)
    if manifest.get("artifact_set_sha256") != artifact_set_sha256:
        raise ValueError("bundle artifact_set_sha256 does not match artifact list")
    if not str(manifest.get("bundle_id", "")).endswith(artifact_set_sha256[:16]):
        raise ValueError("bundle_id is not bound to artifact_set_sha256")
    return manifest, artifacts


def _validate_source_composition_payloads(
    bundle: Path,
    manifest: dict,
) -> None:
    descriptor = manifest["source_snapshot"]["source_composition"]

    def load_json(binding: dict, *, where: str) -> tuple[bytes, dict]:
        path = _safe_artifact_path(bundle, str(binding["path"]))
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError(f"{where} payload SHA-256 drifted")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{where} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be a JSON object")
        return raw, value

    receipt_raw, receipt = load_json(
        descriptor["receipt"], where="source composition receipt"
    )
    del receipt_raw
    expected_receipt_fields = {
        "schema",
        "status",
        "plan_sha256",
        "buckets",
        "archive",
        "dedup",
        "runs",
        "source_producers",
        "source_producer_set_sha256",
        "coverage",
    }
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("schema") != descriptor["schema"]
        or receipt.get("status") != "complete"
        or receipt.get("buckets") != manifest.get("buckets")
    ):
        raise ValueError("source composition receipt contract drifted")
    if receipt.get("plan_sha256") != descriptor["plan"]["sha256"]:
        raise ValueError("source composition plan binding drifted")

    dedup_raw, dedup_receipt = load_json(
        descriptor["dedup_receipt"], where="global dedup receipt"
    )
    portable_dedup = dict(dedup_receipt)
    portable_database = portable_dedup.get("database")
    if not isinstance(portable_database, dict):
        raise ValueError("global dedup receipt database binding is malformed")
    portable_database = dict(portable_database)
    portable_database.pop("path", None)
    portable_dedup["database"] = portable_database
    portable_dedup["receipt_sha256"] = hashlib.sha256(dedup_raw).hexdigest()
    if receipt.get("dedup") != portable_dedup:
        raise ValueError("source composition portable dedup binding drifted")
    if (
        dedup_receipt.get("schema") != "cppmega_global_dedup_store_receipt_v1"
        or dedup_receipt.get("status") != "verified"
        or dedup_receipt.get("integrity_check") != "ok"
        or dedup_receipt.get("checkpoint")
        != {
            "mode": "TRUNCATE",
            "busy": 0,
            "log_frames": 0,
            "checkpointed_frames": 0,
            "wal_size_bytes": 0,
        }
    ):
        raise ValueError("global dedup receipt is not a completed snapshot")
    tables = dedup_receipt.get("tables")
    if not isinstance(tables, dict) or any(
        not isinstance(tables.get(name), dict)
        or tables[name].get("rows") != 0
        for name in (
            "dedup_stages",
            "exact_stage",
            "minhash_stage",
            "lsh_stage",
            "chunk_claims_stage",
        )
    ):
        raise ValueError("global dedup receipt contains unpromoted stage rows")
    if any(
        not isinstance(tables.get(name), dict)
        or not isinstance(tables[name].get("rows"), int)
        or isinstance(tables[name].get("rows"), bool)
        or tables[name]["rows"] < 1
        for name in ("exact", "lsh", "minhash", "dedup_meta", "chunk_claims")
    ):
        raise ValueError("global dedup receipt has empty production tables")
    policy = dedup_receipt.get("policy")
    near = policy.get("near") if isinstance(policy, dict) else None
    if near != {
        "enabled": True,
        "threshold": 0.7,
        "num_perm": 256,
        "shingle_k": 5,
    }:
        raise ValueError("global dedup receipt does not prove production near dedup")
    verifier = dedup_receipt.get("verifier")
    if (
        not isinstance(verifier, dict)
        or verifier.get("script_sha256") != descriptor["dedup_verifier"]["sha256"]
    ):
        raise ValueError("global dedup verifier artifact binding drifted")

    coverage = receipt.get("coverage")
    archive = receipt.get("archive")
    if not isinstance(coverage, dict) or not isinstance(archive, dict):
        raise ValueError("source composition coverage is missing")
    expected_repositories = coverage.get("expected_repositories")
    if (
        not isinstance(expected_repositories, int)
        or isinstance(expected_repositories, bool)
        or expected_repositories < 1
        or coverage.get("code_success_repositories") != expected_repositories
        or coverage.get("commit_success_repositories") != expected_repositories
        or coverage.get("unresolved_failed_units") != 0
        or archive.get("repository_count") != expected_repositories
    ):
        raise ValueError("source composition does not prove full repository coverage")
    allowlist_counts = coverage.get("allowlist_counts")
    expected_allowlist_keys = {
        f"{kind}/{bucket}"
        for kind in ("code", "commits")
        for bucket in manifest["buckets"]
    }
    if (
        not isinstance(allowlist_counts, dict)
        or set(allowlist_counts) != expected_allowlist_keys
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in allowlist_counts.values()
        )
    ):
        raise ValueError("source composition allowlist coverage drifted")

    source_producers = receipt.get("source_producers")
    if not isinstance(source_producers, list) or not source_producers:
        raise ValueError("source composition producer set is missing")
    producer_set_sha256 = hashlib.sha256(
        json.dumps(
            source_producers,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    if receipt.get("source_producer_set_sha256") != producer_set_sha256:
        raise ValueError("source composition producer set digest drifted")

    receipt_runs = receipt.get("runs")
    descriptor_runs = descriptor.get("runs")
    if (
        not isinstance(receipt_runs, list)
        or not isinstance(descriptor_runs, list)
        or len(receipt_runs) != len(descriptor_runs)
    ):
        raise ValueError("source composition run set drifted")
    descriptors_by_id = {run["run_id"]: run for run in descriptor_runs}
    input_file_keys = {
        "archive_sha256_receipt": "archive_sha256_receipt",
        "archive_inventory": "archive_inventory_receipt",
        "repo_list": "repo_list",
        "source_quarantine_manifest": "source_quarantine_manifest",
        "tokenizer": "tokenizer",
    }
    for run in receipt_runs:
        if not isinstance(run, dict) or run.get("run_id") not in descriptors_by_id:
            raise ValueError("source composition receipt run is not staged")
        staged = descriptors_by_id[run["run_id"]]["artifacts"]
        launch = run.get("launch")
        exit_receipt = run.get("exit")
        run_manifest = run.get("manifest")
        if not all(
            isinstance(value, dict)
            for value in (launch, exit_receipt, run_manifest)
        ):
            raise ValueError("source composition run receipt bindings are malformed")
        expected_hashes = {
            "launch": launch.get("sha256"),
            "exit": exit_receipt.get("sha256"),
            "manifest": run_manifest.get("sha256"),
        }
        inputs = run.get("input_artifacts")
        if not isinstance(inputs, dict):
            raise ValueError("source composition run input binding is missing")
        expected_hashes.update(
            {
                file_key: inputs.get(binding_key)
                for file_key, binding_key in input_file_keys.items()
            }
        )
        if any(
            staged[name]["sha256"] != expected_sha256
            for name, expected_sha256 in expected_hashes.items()
        ):
            raise ValueError(
                f"source composition run artifact binding drifted: {run['run_id']}"
            )


def _validate_source_route_payloads(bundle: Path, manifest: dict) -> None:
    source_snapshot = manifest["source_snapshot"]
    descriptor = source_snapshot["source_routes"]
    source_manifest_path = _safe_artifact_path(
        bundle, str(source_snapshot["manifest"])
    )
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source snapshot manifest is not valid JSON") from error
    if not isinstance(source_manifest, dict):
        raise ValueError("source snapshot manifest must be an object")
    route_bindings = source_manifest.get("source_routes")
    source_files = source_manifest.get("files")
    if (
        not isinstance(route_bindings, dict)
        or set(route_bindings) != {"code", "commits"}
        or not isinstance(source_files, list)
    ):
        raise ValueError("source snapshot lacks its primary route bindings")

    count_keys = (
        "rows",
        "valid_tokens",
        "trained_tokens",
        "documents",
        "capacity_tokens",
    )
    for kind in ("code", "commits"):
        staged = descriptor["routes"][kind]
        receipt_path = _safe_artifact_path(bundle, str(staged["path"]))
        raw = receipt_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != staged["sha256"]:
            raise ValueError(f"{kind} source route receipt SHA-256 drifted")
        try:
            receipt = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{kind} source route receipt is not valid JSON") from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != staged["route_schema"]
            or receipt.get("status") != "complete"
            or receipt.get("unresolved_count") != 0
            or receipt.get("input_inventory_sha256")
            != staged["input_inventory_sha256"]
            or receipt.get("output_inventory_sha256")
            != staged["output_inventory_sha256"]
            or not isinstance(receipt.get("implementation"), dict)
            or any(
                not SHA256_RE.fullmatch(str(value))
                for value in receipt["implementation"].values()
            )
        ):
            raise ValueError(f"{kind} source route receipt contract drifted")
        binding = route_bindings[kind]
        if (
            not isinstance(binding, dict)
            or binding.get("schema") != staged["route_schema"]
            or binding.get("kind") != kind
            or binding.get("receipt_sha256") != staged["sha256"]
            or binding.get("input_inventory_sha256")
            != staged["input_inventory_sha256"]
            or binding.get("output_inventory_sha256")
            != staged["output_inventory_sha256"]
            or binding.get("totals") != receipt.get("totals")
            or binding.get("implementation") != receipt.get("implementation")
        ):
            raise ValueError(f"{kind} source snapshot route binding drifted")

        files = receipt.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"{kind} source route receipt has no files")
        totals = {
            "source": {key: 0 for key in count_keys},
            "primary": {key: 0 for key in count_keys},
            "aux_python": {key: 0 for key in count_keys},
            "excluded_non_primary": {key: 0 for key in count_keys},
        }
        primary_files: dict[str, dict[str, object]] = {}
        output_inventory: list[dict[str, str]] = []
        output_root = Path(str(receipt.get("output_root")))
        for file_receipt in files:
            if not isinstance(file_receipt, dict):
                raise ValueError(f"{kind} source route file receipt is malformed")
            source = file_receipt.get("input")
            routes = file_receipt.get("routes")
            if (
                not isinstance(source, dict)
                or not isinstance(routes, dict)
                or set(routes)
                != {"primary", "aux_python", "excluded_non_primary"}
            ):
                raise ValueError(f"{kind} source route file binding drifted")
            relative = source.get("path")
            if not isinstance(relative, str):
                raise ValueError(f"{kind} source route path is invalid")
            for route_name, counts in (("source", source), *routes.items()):
                if not isinstance(counts, dict):
                    raise ValueError(f"{kind} source route counts are malformed")
                for key in count_keys:
                    value = counts.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ValueError(
                            f"{kind} source route {route_name}.{key} is invalid"
                        )
                    totals[route_name][key] += value
            for key in ("valid_tokens", "trained_tokens", "documents"):
                if source[key] != sum(routes[name][key] for name in routes):
                    raise ValueError(
                        f"{kind} source route does not conserve {key}: {relative}"
                    )
            for route_name in (
                "primary",
                "aux_python",
                "excluded_non_primary",
            ):
                artifact = routes[route_name]
                path = artifact.get("path")
                digest = artifact.get("sha256")
                if (
                    path != f"{route_name}/{relative}"
                    or not SHA256_RE.fullmatch(str(digest))
                ):
                    raise ValueError(
                        f"{kind} source route artifact binding drifted: {relative}"
                    )
                output_inventory.append(
                    {
                        "route": route_name,
                        "path": str(path),
                        "sha256": str(digest),
                    }
                )
            primary = routes["primary"]
            primary_source = str((output_root / str(primary["path"])).resolve())
            if primary_source in primary_files:
                raise ValueError(f"{kind} source route contains a duplicate artifact")
            primary_files[primary_source] = {
                "size": primary["size"],
                "sha256": primary["sha256"],
                "rows": primary["rows"],
            }
        if (
            receipt.get("totals") != totals
            or staged["primary"] != totals["primary"]
            or receipt.get("output_inventory_sha256")
            != hashlib.sha256(
                json.dumps(
                    output_inventory,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
        ):
            raise ValueError(f"{kind} source route aggregate binding drifted")

        snapshot_files: dict[str, dict] = {}
        for record in source_files:
            if not isinstance(record, dict) or record.get("kind") != kind:
                continue
            source = record.get("source")
            if not isinstance(source, str) or source in snapshot_files:
                raise ValueError(f"{kind} source snapshot file binding is invalid")
            snapshot_files[source] = record
        if set(snapshot_files) != set(primary_files):
            raise ValueError(f"{kind} source snapshot is not exactly the primary route")
        for source, expected in primary_files.items():
            actual = snapshot_files[source]
            if any(actual.get(field) != value for field, value in expected.items()):
                raise ValueError(
                    f"{kind} source snapshot differs from routed artifact: {source}"
                )


def _validate_bundle(bundle: Path, hash_jobs: int) -> tuple[dict, list[dict]]:
    manifest, artifacts = _load_bundle_manifest(bundle)
    artifact_by_path = {
        str(record["path"]): {
            "path": str(record["path"]),
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }
        for record in artifacts
    }
    _validate_tokenizer_descriptor(bundle, manifest, artifact_by_path)
    _validate_data_contract_descriptors(bundle, manifest, artifact_by_path)
    _validate_source_composition_payloads(bundle, manifest)
    if manifest.get("schema") == "cppmega_megatron_bundle_v4":
        _validate_source_route_payloads(bundle, manifest)
    buckets = manifest.get("buckets")
    if (
        not isinstance(buckets, list)
        or not buckets
        or any(
            not isinstance(bucket, int) or isinstance(bucket, bool)
            for bucket in buckets
        )
        or len(buckets) != len(set(buckets))
    ):
        raise ValueError("bundle buckets must be a non-empty unique integer list")
    objective_descriptors = manifest["objective_materialization"]["buckets"]
    if set(objective_descriptors) != {str(bucket) for bucket in buckets}:
        raise ValueError(
            "bundle objective materialization buckets do not match bundle buckets"
        )
    for bucket in buckets:
        descriptor = objective_descriptors[str(bucket)]
        contract_relative = str(descriptor["contract_path"])
        contract_path = _safe_artifact_path(bundle, contract_relative)
        contract_record = artifact_by_path.get(contract_relative)
        if (
            contract_record is None
            or contract_record["sha256"] != descriptor["contract_file_sha256"]
            or not contract_path.is_file()
        ):
            raise ValueError(f"bucket {bucket} objective contract is not hash-bound")
        raw_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        validated_contract = validate_objective_contract(raw_contract)
        if validated_contract.sha256 != descriptor["contract_sha256"]:
            raise ValueError(
                f"bucket {bucket} objective contract payload digest mismatch"
            )
        source_summary = _objective_source_snapshot_summary(
            validated_contract.payload.get("source_snapshot"),
            bucket=bucket,
        )
        if source_summary != descriptor["source_snapshot"]:
            raise ValueError(
                f"bucket {bucket} objective source_snapshot summary mismatch"
            )

        artifact_relative = str(descriptor["artifact_path"])
        artifact_path = _safe_artifact_path(bundle, artifact_relative)
        artifact_record = artifact_by_path.get(artifact_relative)
        if (
            artifact_record is None
            or artifact_record["sha256"] != descriptor["artifact_file_sha256"]
            or not artifact_path.is_file()
        ):
            raise ValueError(f"bucket {bucket} objective artifact is not hash-bound")
        raw_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact_digest = raw_artifact.get("artifact_set_sha256")
        artifact_payload = dict(raw_artifact)
        artifact_payload.pop("artifact_set_sha256", None)
        actual_artifact_digest = hashlib.sha256(
            json.dumps(
                artifact_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        if (
            raw_artifact.get("schema")
            == LEGACY_OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA
        ):
            raise ValueError(
                f"bucket {bucket} objective artifact is legacy; migration required: "
                "regenerate the objective artifact and bundle"
            )
        expected_artifact_fields = {
            "schema",
            "graph_recipe",
            "documents",
            "objective_contract",
            "parquet_shards",
            "converter",
            "artifact_set_sha256",
        }
        if set(raw_artifact) != expected_artifact_fields:
            raise ValueError(
                f"bucket {bucket} objective artifact fields drifted; "
                "regenerate the objective artifact and bundle"
            )
        if (
            raw_artifact.get("schema") != descriptor["artifact_schema"]
            or artifact_digest != descriptor["artifact_set_sha256"]
            or artifact_digest != actual_artifact_digest
        ):
            raise ValueError(
                f"bucket {bucket} objective artifact payload digest mismatch"
            )
        if raw_artifact.get("graph_recipe") != stage1_graph_recipe_binding():
            raise ValueError(
                f"bucket {bucket} objective artifact graph recipe mismatch; "
                "regenerate the objective artifact and bundle"
            )
    bucket_results = manifest.get("bucket_results")
    if not isinstance(bucket_results, list) or not bucket_results:
        raise ValueError("bundle bucket_results must be a non-empty list")
    result_buckets: list[int] = []
    referenced_paths: set[Path] = set()
    for result in bucket_results:
        if not isinstance(result, dict):
            raise ValueError("bundle bucket_results entries must be objects")
        bucket = result.get("bucket")
        if not isinstance(bucket, int) or isinstance(bucket, bool):
            raise ValueError("bundle bucket result has invalid bucket")
        result_buckets.append(bucket)
        prefix = _safe_artifact_path(bundle, str(result.get("prefix", "")))
        prefix_manifest, referenced = _validate_prefix_manifest_contract(prefix)
        descriptor = objective_descriptors[str(bucket)]
        if (
            prefix_manifest["objective_contract"]["sha256"]
            != descriptor["contract_sha256"]
        ):
            raise ValueError(
                f"{prefix}: objective contract does not match bundle descriptor"
            )
        materialization = prefix_manifest.get("objective_materialization")
        if (
            not isinstance(materialization, dict)
            or materialization.get("artifact_set_sha256")
            != descriptor["artifact_set_sha256"]
            or materialization.get("artifact_file_sha256")
            != descriptor["artifact_file_sha256"]
        ):
            raise ValueError(
                f"{prefix}: objective artifact does not match bundle descriptor"
            )
        if result.get("manifest") != prefix_manifest:
            raise ValueError(
                f"{prefix}: embedded prefix manifest does not match artifact"
            )
        referenced_paths.update(referenced)
    if result_buckets != buckets:
        raise ValueError(
            f"bundle bucket_results do not match buckets: {result_buckets} != {buckets}"
        )
    for path in referenced_paths:
        relative = path.relative_to(bundle.resolve()).as_posix()
        if relative not in artifact_by_path:
            raise ValueError(
                f"referenced bundle artifact has no hash record: {relative}"
            )

    def validate(record: dict) -> dict:
        relative = str(record["path"])
        path = _safe_artifact_path(bundle, relative)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(record["size"]):
            raise ValueError(
                f"artifact size mismatch for {relative}: {size} != {record['size']}"
            )
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"artifact sha256 mismatch for {relative}")
        return {**record, "local_path": str(path)}

    with ThreadPoolExecutor(max_workers=max(1, hash_jobs)) as pool:
        validated = list(pool.map(validate, artifacts))
    return manifest, validated


def _validate_archive_member_names(
    member_names: list[str], expected_names: set[str]
) -> None:
    if len(member_names) != len(set(member_names)):
        raise ValueError("archive contains duplicate member names")
    actual = set(member_names)
    missing = sorted(expected_names - actual)
    extra = sorted(actual - expected_names)
    if missing or extra:
        raise ValueError(
            f"archive member set mismatch: missing={missing[:5]} extra={extra[:5]}"
        )


def _validate_archive(
    *, bundle: Path, archive: Path, manifest: dict
) -> tuple[int, str]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    local_manifest = (bundle / "manifest.json").read_bytes()
    expected = {
        str(record["path"]): (int(record["size"]), str(record["sha256"]))
        for record in manifest["artifacts"]
    }
    expected["manifest.json"] = (
        len(local_manifest),
        hashlib.sha256(local_manifest).hexdigest(),
    )
    seen: list[str] = []
    decoder = subprocess.Popen(
        ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert decoder.stdout is not None
    try:
        with tarfile.open(fileobj=decoder.stdout, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    raise ValueError(
                        f"archive contains non-file member: {member.name!r}"
                    )
                seen.append(member.name)
                expected_record = expected.get(member.name)
                if expected_record is None:
                    raise ValueError(
                        f"archive contains unexpected member: {member.name!r}"
                    )
                expected_size, expected_sha256 = expected_record
                if member.size != expected_size:
                    raise ValueError(
                        f"archive member size mismatch for {member.name}: "
                        f"{member.size} != {expected_size}"
                    )
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read archive member: {member.name!r}")
                digest = hashlib.sha256()
                with extracted:
                    while chunk := extracted.read(8 * 1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise ValueError(
                        f"archive member SHA-256 mismatch for {member.name}"
                    )
    except BaseException:
        decoder.kill()
        decoder.wait()
        raise
    finally:
        decoder.stdout.close()
    decoder_stderr = decoder.communicate()[1]
    if decoder.returncode != 0:
        raise RuntimeError(
            "zstd archive validation failed: "
            f"{decoder_stderr.decode(errors='replace').strip()}"
        )
    _validate_archive_member_names(seen, set(expected))
    return archive.stat().st_size, _sha256(archive)


def _head(
    *,
    endpoint: str,
    bucket: str,
    key: str,
    env: dict[str, str],
) -> dict | None:
    cmd = [
        "aws",
        "s3api",
        "head-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--endpoint-url",
        endpoint,
        "--output",
        "json",
        "--checksum-mode",
        "ENABLED",
    ]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        error = result.stderr.lower()
        if any(marker in error for marker in ("404", "not found", "nosuchkey")):
            return None
        raise RuntimeError(
            f"remote HEAD failed ({result.returncode}) for s3://{bucket}/{key}: "
            f"{result.stderr.strip()}"
        )
    return json.loads(result.stdout) if result.stdout.strip() else {}


def _head_matches(head: dict | None, *, size: int, sha256: str) -> bool:
    if (
        not head
        or SHA256_RE.fullmatch(sha256) is None
        or int(head.get("ContentLength", -1)) != size
    ):
        return False
    metadata = {
        str(key).lower(): value for key, value in (head.get("Metadata") or {}).items()
    }
    if metadata.get("sha256") != sha256:
        return False
    if head.get("ChecksumType") != "FULL_OBJECT":
        return False
    expected_checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
    return head.get("ChecksumSHA256") == expected_checksum


def _verified_head_receipt(head: dict, *, sha256: str) -> dict[str, object]:
    return {
        "checksum_algorithm": "SHA256",
        "checksum_type": head["ChecksumType"],
        "checksum_sha256": head["ChecksumSHA256"],
        "metadata_sha256": sha256,
        "etag": head.get("ETag"),
    }


def _multipart_layout(size: int) -> tuple[int, int]:
    if size <= 0:
        raise ValueError(f"multipart object size must be positive, got {size}")
    if size > S3_MAX_OBJECT_BYTES:
        raise RuntimeError(
            f"object size {size} exceeds the supported S3 limit {S3_MAX_OBJECT_BYTES}"
        )
    minimum_for_part_limit = (size + S3_MAX_MULTIPART_PARTS - 1) // (
        S3_MAX_MULTIPART_PARTS
    )
    aligned_minimum = (
        (
            minimum_for_part_limit
            + MULTIPART_PART_ALIGNMENT_BYTES
            - 1
        )
        // MULTIPART_PART_ALIGNMENT_BYTES
        * MULTIPART_PART_ALIGNMENT_BYTES
    )
    part_size = max(
        S3_MIN_MULTIPART_PART_BYTES,
        MULTIPART_DEFAULT_PART_BYTES,
        aligned_minimum,
    )
    part_count = (size + part_size - 1) // part_size
    if part_size > S3_MAX_MULTIPART_PART_BYTES or part_count > S3_MAX_MULTIPART_PARTS:
        raise RuntimeError(
            f"no safe S3 multipart layout for size={size}: "
            f"part_size={part_size} part_count={part_count}"
        )
    return part_size, part_count


def _multipart_metadata(
    *, sha256: str, part_size: int, part_count: int
) -> dict[str, str]:
    return {
        "sha256": sha256,
        "publication-protocol": MULTIPART_PUBLICATION_PROTOCOL,
        "multipart-part-size": str(part_size),
        "multipart-part-count": str(part_count),
    }


def _multipart_layout_metadata_matches(
    head: dict | None,
    *,
    size: int,
    sha256: str,
    part_size: int,
    part_count: int,
) -> bool:
    """Check whether a HEAD is a verification candidate, not whether bytes match."""
    if (
        not head
        or SHA256_RE.fullmatch(sha256) is None
        or int(head.get("ContentLength", -1)) != size
    ):
        return False
    metadata = {
        str(key).lower(): str(value)
        for key, value in (head.get("Metadata") or {}).items()
    }
    return metadata == _multipart_metadata(
        sha256=sha256, part_size=part_size, part_count=part_count
    )


def _read_exact_part(
    source: BinaryIO, length: int, destination: BinaryIO | None = None
) -> str:
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        chunk = source.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError(
                f"multipart source ended with {remaining} bytes still expected"
            )
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
        remaining -= len(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _multipart_part_checksums(
    path: Path, *, size: int, part_size: int, part_count: int
) -> list[str]:
    checksums: list[str] = []
    with path.open("rb") as source:
        for part_number in range(1, part_count + 1):
            part_offset = (part_number - 1) * part_size
            part_length = min(part_size, size - part_offset)
            checksums.append(_read_exact_part(source, part_length))
        if source.read(1):
            raise RuntimeError(f"multipart source contains bytes beyond size={size}: {path}")
    return checksums


def _multipart_composite_sha256(part_checksums: list[str]) -> str:
    if not part_checksums:
        raise ValueError("multipart checksum requires at least one part")
    digest = hashlib.sha256()
    for checksum in part_checksums:
        try:
            raw = base64.b64decode(checksum, validate=True)
        except ValueError as error:
            raise ValueError(f"invalid multipart part SHA-256: {checksum!r}") from error
        if len(raw) != hashlib.sha256().digest_size:
            raise ValueError(f"invalid multipart part SHA-256 length: {checksum!r}")
        digest.update(raw)
    encoded = base64.b64encode(digest.digest()).decode("ascii")
    return f"{encoded}-{len(part_checksums)}"


def _multipart_head_matches(
    head: dict | None,
    *,
    size: int,
    sha256: str,
    part_size: int,
    part_count: int,
    checksum_sha256: str,
    etag: str | None = None,
) -> bool:
    if not _multipart_layout_metadata_matches(
        head,
        size=size,
        sha256=sha256,
        part_size=part_size,
        part_count=part_count,
    ):
        return False
    assert head is not None
    if (
        head.get("ChecksumType") != "COMPOSITE"
        or head.get("ChecksumSHA256") != checksum_sha256
    ):
        return False
    remote_etag = head.get("ETag")
    if not isinstance(remote_etag, str) or not remote_etag:
        return False
    return etag is None or remote_etag == etag


def _multipart_receipt(
    *,
    head: dict | None,
    key: str,
    size: int,
    sha256: str,
    part_size: int,
    part_count: int,
    checksum_sha256: str,
    status: str,
    etag: str | None = None,
) -> dict[str, object]:
    if not _multipart_head_matches(
        head,
        size=size,
        sha256=sha256,
        part_size=part_size,
        part_count=part_count,
        checksum_sha256=checksum_sha256,
        etag=etag,
    ):
        raise RuntimeError(f"remote multipart verification failed for key {key!r}")
    assert head is not None
    verified_metadata = _multipart_metadata(
        sha256=sha256, part_size=part_size, part_count=part_count
    )
    return {
        "key": key,
        "size": size,
        "sha256": sha256,
        "status": status,
        "upload_mode": "multipart",
        "part_size": part_size,
        "part_count": part_count,
        "checksum_type": "COMPOSITE",
        "checksum_sha256": checksum_sha256,
        "etag": head["ETag"],
        "verification": {
            "content_length": int(head["ContentLength"]),
            "metadata": verified_metadata,
            "checksum_type": head["ChecksumType"],
            "checksum_sha256": head["ChecksumSHA256"],
        },
    }


def _parse_s3_json(result: object, *, operation: str, uri: str) -> dict:
    returncode = int(getattr(result, "returncode", -1))
    stderr = str(getattr(result, "stderr", "")).strip()
    if returncode != 0:
        raise RuntimeError(f"{operation} failed for {uri}: {stderr}")
    stdout = str(getattr(result, "stdout", ""))
    try:
        payload = json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{operation} returned invalid JSON for {uri}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{operation} returned non-object JSON for {uri}")
    return payload


def _abort_multipart_upload(
    *, endpoint: str, bucket: str, key: str, upload_id: str, env: dict[str, str]
) -> str | None:
    command = [
        "aws",
        "s3api",
        "abort-multipart-upload",
        "--bucket",
        bucket,
        "--key",
        key,
        "--upload-id",
        upload_id,
        "--endpoint-url",
        endpoint,
    ]
    try:
        result = subprocess.run(
            command, env=env, text=True, capture_output=True, check=False
        )
    except Exception as error:  # pragma: no cover - subprocess launch failures are rare
        return str(error)
    if result.returncode == 0:
        return None
    error = result.stderr.strip()
    lowered = error.lower()
    if "nosuchupload" in lowered:
        return None
    return error or f"abort-multipart-upload exited {result.returncode}"


@contextmanager
def _stable_upload_snapshot(local: Path, *, size: int, sha256: str):
    if not local.is_file() or local.stat().st_size != size or _sha256(local) != sha256:
        raise RuntimeError(f"local upload source drifted before snapshot: {local}")
    snapshot = local.with_name(
        f".{local.name}.upload-{os.getpid()}-{threading.get_ident()}"
    )
    snapshot.unlink(missing_ok=True)
    clone_command = (
        ["cp", "-c", str(local), str(snapshot)]
        if sys.platform == "darwin"
        else ["cp", "--reflink=auto", "--", str(local), str(snapshot)]
    )
    cloned = subprocess.run(clone_command, capture_output=True, check=False)
    if cloned.returncode != 0:
        shutil.copyfile(local, snapshot)
    try:
        snapshot.chmod(0o400)
        if snapshot.stat().st_size != size or _sha256(snapshot) != sha256:
            raise RuntimeError(f"stable upload snapshot does not match {local}")
        yield snapshot
        if snapshot.stat().st_size != size or _sha256(snapshot) != sha256:
            raise RuntimeError(f"stable upload snapshot changed during upload: {local}")
    finally:
        snapshot.chmod(0o600)
        snapshot.unlink(missing_ok=True)


def _upload_multipart_file(
    *,
    local: Path,
    endpoint: str,
    bucket: str,
    key: str,
    size: int,
    sha256: str,
    env: dict[str, str],
    initial_head: dict | None,
    allow_overwrite: bool,
) -> dict[str, object]:
    """Publish through a checksum-bound conditional completion.

    Multipart creation is intentionally not the linearization point: incomplete
    uploads are invisible.  The destination changes only through a conditional
    CompleteMultipartUpload.  Endpoints that reject that condition or the
    checksum contract fail closed; there is no unconditional fallback.
    """
    uri = f"s3://{bucket}/{key}"
    part_size, part_count = _multipart_layout(size)
    existing_verification_candidate = _multipart_layout_metadata_matches(
        initial_head,
        size=size,
        sha256=sha256,
        part_size=part_size,
        part_count=part_count,
    )
    if (
        initial_head is not None
        and not existing_verification_candidate
        and not allow_overwrite
    ):
        metadata = {
            str(name).lower(): value
            for name, value in (initial_head.get("Metadata") or {}).items()
        }
        raise RuntimeError(
            f"immutable remote object mismatch for {uri}: "
            f"size={initial_head.get('ContentLength')} sha256={metadata.get('sha256')}; "
            f"local size={size} sha256={sha256}"
        )

    upload_id: str | None = None
    abort_attempted = False
    abort_failure: str | None = None

    def abort_once() -> str | None:
        nonlocal abort_attempted, abort_failure
        if upload_id is None or abort_attempted:
            return abort_failure
        abort_attempted = True
        abort_failure = _abort_multipart_upload(
            endpoint=endpoint,
            bucket=bucket,
            key=key,
            upload_id=upload_id,
            env=env,
        )
        return abort_failure

    try:
        with _stable_upload_snapshot(local, size=size, sha256=sha256) as snapshot:
            expected_part_checksums: list[str] | None = None
            if existing_verification_candidate:
                expected_part_checksums = _multipart_part_checksums(
                    snapshot,
                    size=size,
                    part_size=part_size,
                    part_count=part_count,
                )
                existing_checksum = _multipart_composite_sha256(
                    expected_part_checksums
                )
                current_head = _head(
                    endpoint=endpoint, bucket=bucket, key=key, env=env
                )
                if _multipart_head_matches(
                    current_head,
                    size=size,
                    sha256=sha256,
                    part_size=part_size,
                    part_count=part_count,
                    checksum_sha256=existing_checksum,
                ):
                    return _multipart_receipt(
                        head=current_head,
                        key=key,
                        size=size,
                        sha256=sha256,
                        part_size=part_size,
                        part_count=part_count,
                        checksum_sha256=existing_checksum,
                        status="already_verified",
                    )
                if not allow_overwrite:
                    raise RuntimeError(
                        "immutable remote multipart object cannot be verified exactly "
                        f"for {uri}; refusing replacement"
                    )
                initial_head = current_head

            if initial_head is None:
                complete_condition = ["--if-none-match", "*"]
            else:
                etag = initial_head.get("ETag")
                if not allow_overwrite or not isinstance(etag, str) or not etag:
                    raise RuntimeError(
                        f"remote object cannot be conditionally replaced: {uri}"
                    )
                complete_condition = ["--if-match", etag]

            metadata = _multipart_metadata(
                sha256=sha256, part_size=part_size, part_count=part_count
            )
            create_command = [
                "aws",
                "s3api",
                "create-multipart-upload",
                "--bucket",
                bucket,
                "--key",
                key,
                "--endpoint-url",
                endpoint,
                "--metadata",
                json.dumps(metadata, separators=(",", ":"), sort_keys=True),
                "--checksum-algorithm",
                "SHA256",
                "--checksum-type",
                "COMPOSITE",
                "--output",
                "json",
            ]
            create_result = subprocess.run(
                create_command,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            create_payload = _parse_s3_json(
                create_result, operation="create-multipart-upload", uri=uri
            )
            candidate_upload_id = create_payload.get("UploadId")
            if not isinstance(candidate_upload_id, str) or not candidate_upload_id:
                raise RuntimeError(
                    f"create-multipart-upload returned no UploadId for {uri}"
                )
            upload_id = candidate_upload_id

            uploaded_parts: list[dict[str, object]] = []
            uploaded_checksums: list[str] = []
            with snapshot.open("rb") as source:
                for part_number in range(1, part_count + 1):
                    part_offset = (part_number - 1) * part_size
                    part_length = min(part_size, size - part_offset)
                    part_path: Path | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            prefix=".cppmega-multipart-part-",
                            suffix=f"-{part_number:05d}",
                            dir=snapshot.parent,
                            delete=False,
                        ) as part_file:
                            part_path = Path(part_file.name)
                            part_checksum = _read_exact_part(
                                source, part_length, part_file
                            )
                        if (
                            expected_part_checksums is not None
                            and part_checksum
                            != expected_part_checksums[part_number - 1]
                        ):
                            raise RuntimeError(
                                f"stable multipart snapshot changed at part {part_number}"
                            )
                        upload_command = [
                            "aws",
                            "s3api",
                            "upload-part",
                            "--bucket",
                            bucket,
                            "--key",
                            key,
                            "--upload-id",
                            upload_id,
                            "--part-number",
                            str(part_number),
                            "--body",
                            str(part_path),
                            "--content-length",
                            str(part_length),
                            "--checksum-algorithm",
                            "SHA256",
                            "--checksum-sha256",
                            part_checksum,
                            "--endpoint-url",
                            endpoint,
                            "--output",
                            "json",
                        ]
                        upload_result = subprocess.run(
                            upload_command,
                            env=env,
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                        upload_payload = _parse_s3_json(
                            upload_result,
                            operation=f"upload-part {part_number}",
                            uri=uri,
                        )
                        etag = upload_payload.get("ETag")
                        remote_checksum = upload_payload.get("ChecksumSHA256")
                        if not isinstance(etag, str) or not etag:
                            raise RuntimeError(
                                f"upload-part {part_number} returned no ETag for {uri}"
                            )
                        if remote_checksum != part_checksum:
                            raise RuntimeError(
                                f"upload-part {part_number} checksum mismatch for {uri}: "
                                f"{remote_checksum!r} != {part_checksum!r}"
                            )
                        uploaded_parts.append(
                            {
                                "PartNumber": part_number,
                                "ETag": etag,
                                "ChecksumSHA256": part_checksum,
                            }
                        )
                        uploaded_checksums.append(part_checksum)
                    finally:
                        if part_path is not None:
                            part_path.unlink(missing_ok=True)
                if source.read(1):
                    raise RuntimeError(
                        f"multipart source contains bytes beyond size={size}: {snapshot}"
                    )

            composite_checksum = _multipart_composite_sha256(uploaded_checksums)
            checksum_header = composite_checksum.rsplit("-", 1)[0]
            manifest_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    prefix=".cppmega-multipart-complete-",
                    suffix=".json",
                    dir=snapshot.parent,
                    encoding="utf-8",
                    delete=False,
                ) as manifest_file:
                    manifest_path = Path(manifest_file.name)
                    json.dump(
                        {"Parts": uploaded_parts},
                        manifest_file,
                        separators=(",", ":"),
                    )
                complete_command = [
                    "aws",
                    "s3api",
                    "complete-multipart-upload",
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--upload-id",
                    upload_id,
                    "--multipart-upload",
                    f"file://{manifest_path}",
                    "--checksum-sha256",
                    checksum_header,
                    "--checksum-type",
                    "COMPOSITE",
                    "--mpu-object-size",
                    str(size),
                    "--endpoint-url",
                    endpoint,
                    "--output",
                    "json",
                    *complete_condition,
                ]
                complete_result = subprocess.run(
                    complete_command,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            finally:
                if manifest_path is not None:
                    manifest_path.unlink(missing_ok=True)

            if complete_result.returncode != 0:
                completion_error = complete_result.stderr.strip()
                cleanup_error = abort_once()
                if cleanup_error is not None:
                    raise RuntimeError(
                        f"conditional multipart completion failed for {uri}: "
                        f"{completion_error}; abort also failed: {cleanup_error}"
                    )
                concurrent_head = _head(
                    endpoint=endpoint, bucket=bucket, key=key, env=env
                )
                if _multipart_head_matches(
                    concurrent_head,
                    size=size,
                    sha256=sha256,
                    part_size=part_size,
                    part_count=part_count,
                    checksum_sha256=composite_checksum,
                ):
                    receipt = _multipart_receipt(
                        head=concurrent_head,
                        key=key,
                        size=size,
                        sha256=sha256,
                        part_size=part_size,
                        part_count=part_count,
                        checksum_sha256=composite_checksum,
                        status="already_verified",
                    )
                    receipt["race_resolution"] = "matching_concurrent_publisher"
                    return receipt
                raise RuntimeError(
                    f"conditional multipart completion failed for {uri}: "
                    f"{completion_error}; destination is absent or does not match"
                )

            complete_payload = _parse_s3_json(
                complete_result, operation="complete-multipart-upload", uri=uri
            )
            response_checksum = complete_payload.get("ChecksumSHA256")
            if response_checksum not in (None, composite_checksum):
                raise RuntimeError(
                    f"complete-multipart-upload checksum mismatch for {uri}: "
                    f"{response_checksum!r} != {composite_checksum!r}"
                )
            response_checksum_type = complete_payload.get("ChecksumType")
            if response_checksum_type not in (None, "COMPOSITE"):
                raise RuntimeError(
                    f"complete-multipart-upload checksum type mismatch for {uri}: "
                    f"{response_checksum_type!r}"
                )
            response_etag = complete_payload.get("ETag")
            expected_etag = response_etag if isinstance(response_etag, str) else None
            final_head = _head(endpoint=endpoint, bucket=bucket, key=key, env=env)
            receipt = _multipart_receipt(
                head=final_head,
                key=key,
                size=size,
                sha256=sha256,
                part_size=part_size,
                part_count=part_count,
                checksum_sha256=composite_checksum,
                status="uploaded_verified",
                etag=expected_etag,
            )
        return receipt
    except BaseException as error:
        cleanup_error = abort_once()
        if cleanup_error is not None:
            raise RuntimeError(
                f"{error}; multipart abort failed for {uri}: {cleanup_error}"
            ) from error
        raise


def _upload_file(
    *,
    local: Path,
    endpoint: str,
    bucket: str,
    key: str,
    size: int,
    sha256: str,
    env: dict[str, str],
    dry_run: bool,
    allow_overwrite: bool = False,
) -> dict[str, object]:
    head: dict | None = None
    if not dry_run:
        head = _head(endpoint=endpoint, bucket=bucket, key=key, env=env)
    if dry_run:
        return {"key": key, "size": size, "sha256": sha256, "status": "dry_run"}
    if size > S3_SINGLE_PUT_MAX_BYTES:
        return _upload_multipart_file(
            local=local,
            endpoint=endpoint,
            bucket=bucket,
            key=key,
            size=size,
            sha256=sha256,
            env=env,
            initial_head=head,
            allow_overwrite=allow_overwrite,
        )
    if _head_matches(head, size=size, sha256=sha256):
        assert head is not None
        return {
            "key": key,
            "size": size,
            "sha256": sha256,
            "status": "already_verified",
            "verification": _verified_head_receipt(head, sha256=sha256),
        }
    if head is not None and not allow_overwrite:
        remote_metadata = {
            str(key).lower(): value
            for key, value in (head.get("Metadata") or {}).items()
        }
        raise RuntimeError(
            f"immutable remote object mismatch for s3://{bucket}/{key}: "
            f"size={head.get('ContentLength')} sha256={remote_metadata.get('sha256')}; "
            f"local size={size} sha256={sha256}"
        )
    expected_checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
    with _stable_upload_snapshot(local, size=size, sha256=sha256) as snapshot:
        command = [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str(snapshot),
            "--endpoint-url",
            endpoint,
            "--metadata",
            f"sha256={sha256}",
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            expected_checksum,
            "--output",
            "json",
        ]
        if head is None:
            command.extend(["--if-none-match", "*"])
        else:
            etag = head.get("ETag")
            if not allow_overwrite or not isinstance(etag, str) or not etag:
                raise RuntimeError(
                    f"remote object cannot be conditionally replaced: s3://{bucket}/{key}"
                )
            command.extend(["--if-match", etag])
        result = subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"conditional S3 upload failed for s3://{bucket}/{key}: "
                f"{result.stderr.strip()}"
            )
    head = _head(endpoint=endpoint, bucket=bucket, key=key, env=env)
    if not _head_matches(head, size=size, sha256=sha256):
        raise RuntimeError(f"remote verification failed for s3://{bucket}/{key}")
    assert head is not None
    return {
        "key": key,
        "size": size,
        "sha256": sha256,
        "status": "uploaded_verified",
        "verification": _verified_head_receipt(head, sha256=sha256),
    }


def _publish_json(
    *,
    payload: dict,
    key: str,
    endpoint: str,
    bucket: str,
    env: dict[str, str],
    dry_run: bool,
    allow_overwrite: bool = False,
) -> dict[str, object]:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    if dry_run:
        return {"key": key, "size": len(data), "sha256": digest, "status": "dry_run"}
    with tempfile.NamedTemporaryFile(prefix="cppmega-s3-", suffix=".json") as fh:
        fh.write(data)
        fh.flush()
        return _upload_file(
            local=Path(fh.name),
            endpoint=endpoint,
            bucket=bucket,
            key=key,
            size=len(data),
            sha256=digest,
            env=env,
            dry_run=False,
            allow_overwrite=allow_overwrite,
        )


def _open_resumable_receipt(
    path: Path, *, schema: str, binding: dict[str, object]
) -> dict:
    if path.exists():
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("schema") != schema or any(
            receipt.get(key) != value for key, value in binding.items()
        ):
            raise ValueError(f"publish receipt binding mismatch: {path}")
        return receipt
    receipt = {
        "schema": schema,
        **binding,
        "status": "in_progress",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(path, receipt)
    return receipt


def _update_receipt(path: Path, receipt: dict, **updates: object) -> None:
    receipt.update(updates)
    receipt["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(path, receipt)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--hash-jobs", type=int, default=4)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--archive",
        type=Path,
        help="publish this exact manifest-bound tar.zst instead of loose objects",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bundle = args.bundle.resolve()
    _load_env_file(args.env_file)
    env = os.environ.copy() if args.dry_run else _s3_env()
    manifest, artifacts = _validate_bundle(bundle, args.hash_jobs)
    bundle_id = str(manifest["bundle_id"])

    if args.archive is not None:
        archive = args.archive.resolve()
        archive_size, archive_sha256 = _validate_archive(
            bundle=bundle, archive=archive, manifest=manifest
        )
        receipt_path = bundle / (
            "archive_publish_dry_run_receipt.json"
            if args.dry_run
            else "archive_publish_receipt.json"
        )
        receipt_payload = _open_resumable_receipt(
            receipt_path,
            schema="cppmega_megatron_archive_publish_receipt_v1",
            binding={
                "endpoint_url": args.endpoint_url,
                "bucket": args.bucket,
                "prefix": args.prefix.strip("/"),
                "bundle_id": bundle_id,
                "artifact_set_sha256": manifest["artifact_set_sha256"],
                "dry_run": args.dry_run,
            },
        )
        archive_validation = {
            "status": "verified",
            "member_count": len(artifacts) + 1,
            "artifact_set_sha256": manifest["artifact_set_sha256"],
            "logical_manifest_sha256": _sha256(bundle / "manifest.json"),
        }
        _update_receipt(
            receipt_path,
            receipt_payload,
            status="in_progress",
            archive_validation=archive_validation,
        )
        transport_base = f"{args.prefix.strip('/')}/transports/{bundle_id}"
        archive_key = f"{transport_base}/bundle-{archive_sha256}.tar.zst"
        archive_record = _upload_file(
            local=archive,
            endpoint=args.endpoint_url,
            bucket=args.bucket,
            key=archive_key,
            size=archive_size,
            sha256=archive_sha256,
            env=env,
            dry_run=args.dry_run,
        )
        _update_receipt(receipt_path, receipt_payload, archive=archive_record)
        logical_manifest_path = bundle / "manifest.json"
        logical_manifest_sha256 = _sha256(logical_manifest_path)
        logical_manifest_key = f"{transport_base}/logical_manifest.json"
        logical_manifest_record = _upload_file(
            local=logical_manifest_path,
            endpoint=args.endpoint_url,
            bucket=args.bucket,
            key=logical_manifest_key,
            size=logical_manifest_path.stat().st_size,
            sha256=logical_manifest_sha256,
            env=env,
            dry_run=args.dry_run,
        )
        _update_receipt(
            receipt_path,
            receipt_payload,
            logical_manifest=logical_manifest_record,
        )
        transport = {
            "schema": "cppmega_megatron_bundle_transport_v1",
            "bundle_id": bundle_id,
            "logical_manifest_sha256": logical_manifest_sha256,
            "artifact_set_sha256": manifest["artifact_set_sha256"],
            "artifact_count": manifest["artifact_count"],
            "artifact_bytes": manifest["artifact_bytes"],
            "logical_manifest": {
                "uri": f"s3://{args.bucket}/{logical_manifest_key}",
                "size": logical_manifest_path.stat().st_size,
                "sha256": logical_manifest_sha256,
            },
            "archive": {
                "uri": f"s3://{args.bucket}/{archive_key}",
                "size": archive_size,
                "sha256": archive_sha256,
                "format": "tar.zst",
            },
        }
        transport_record = _publish_json(
            payload=transport,
            key=f"{transport_base}/transport.json",
            endpoint=args.endpoint_url,
            bucket=args.bucket,
            env=env,
            dry_run=args.dry_run,
        )
        _update_receipt(receipt_path, receipt_payload, transport=transport_record)
        latest_transport = {
            "schema": "cppmega_megatron_latest_transport_v1",
            "bundle_id": bundle_id,
            "transport": f"s3://{args.bucket}/{transport_base}/transport.json",
            "transport_sha256": transport_record["sha256"],
            "archive": transport["archive"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        latest_record = _publish_json(
            payload=latest_transport,
            key=f"{args.prefix.strip('/')}/latest_transport.json",
            endpoint=args.endpoint_url,
            bucket=args.bucket,
            env=env,
            dry_run=args.dry_run,
            allow_overwrite=True,
        )
        _update_receipt(
            receipt_path,
            receipt_payload,
            status="complete",
            archive=archive_record,
            logical_manifest=logical_manifest_record,
            transport=transport_record,
            latest_transport=latest_record,
        )
        print(
            json.dumps(
                {"receipt": str(receipt_path), "latest_transport": latest_transport},
                indent=2,
            )
        )
        return 0

    base_key = f"{args.prefix.strip('/')}/bundles/{bundle_id}"

    receipt_path = bundle / (
        "publish_dry_run_receipt.json" if args.dry_run else "publish_receipt.json"
    )
    receipt_payload = _open_resumable_receipt(
        receipt_path,
        schema="cppmega_megatron_publish_receipt_v1",
        binding={
            "endpoint_url": args.endpoint_url,
            "bucket": args.bucket,
            "base_key": base_key,
            "bundle_id": bundle_id,
            "artifact_set_sha256": manifest["artifact_set_sha256"],
            "dry_run": args.dry_run,
        },
    )
    expected_receipts = {
        f"{base_key}/{record['path']}": {
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }
        for record in artifacts
    }
    prior_artifacts = receipt_payload.get("artifacts", [])
    if not isinstance(prior_artifacts, list):
        raise ValueError("publish receipt artifacts must be a list")
    receipts_by_key: dict[str, dict] = {}
    for item in prior_artifacts:
        if not isinstance(item, dict):
            raise ValueError("publish receipt artifact entries must be objects")
        key = str(item.get("key", ""))
        expected = expected_receipts.get(key)
        if (
            expected is None
            or item.get("size") != expected["size"]
            or item.get("sha256") != expected["sha256"]
            or item.get("status")
            not in {"dry_run", "already_verified", "uploaded_verified"}
        ):
            raise ValueError(f"publish receipt artifact mismatch: {key!r}")
        receipts_by_key[key] = item
    _update_receipt(
        receipt_path,
        receipt_payload,
        status="in_progress",
        artifacts=sorted(receipts_by_key.values(), key=lambda item: str(item["key"])),
    )
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(
                _upload_file,
                local=Path(record["local_path"]),
                endpoint=args.endpoint_url,
                bucket=args.bucket,
                key=f"{base_key}/{record['path']}",
                size=int(record["size"]),
                sha256=str(record["sha256"]),
                env=env,
                dry_run=args.dry_run,
            )
            for record in artifacts
        ]
        for future in as_completed(futures):
            receipt = future.result()
            receipts_by_key[str(receipt["key"])] = receipt
            _update_receipt(
                receipt_path,
                receipt_payload,
                artifacts=sorted(
                    receipts_by_key.values(), key=lambda item: str(item["key"])
                ),
            )
            print(json.dumps(receipt, sort_keys=True), flush=True)

    manifest_record = _publish_json(
        payload=manifest,
        key=f"{base_key}/manifest.json",
        endpoint=args.endpoint_url,
        bucket=args.bucket,
        env=env,
        dry_run=args.dry_run,
    )
    latest = {
        "schema": "cppmega_megatron_latest_v1",
        "bundle_id": bundle_id,
        "manifest": f"s3://{args.bucket}/{base_key}/manifest.json",
        "manifest_sha256": manifest_record["sha256"],
        "artifact_count": manifest["artifact_count"],
        "artifact_bytes": manifest["artifact_bytes"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    latest_record = _publish_json(
        payload=latest,
        key=f"{args.prefix.strip('/')}/latest.json",
        endpoint=args.endpoint_url,
        bucket=args.bucket,
        env=env,
        dry_run=args.dry_run,
        allow_overwrite=True,
    )
    _update_receipt(
        receipt_path,
        receipt_payload,
        status="complete",
        artifacts=sorted(receipts_by_key.values(), key=lambda item: str(item["key"])),
        manifest=manifest_record,
        latest=latest_record,
    )
    print(json.dumps({"receipt": str(receipt_path), "latest": latest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
