"""Canonical token-aligned domain-route contract derived from frozen data files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_SCHEMA_PATH = _REPO_ROOT / "data/domain_schema_v1.json"
TOKENIZER_CONTRACT_PATH = _REPO_ROOT / "data/tokenizer_v2/tokenizer_contract_v1.json"


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to load frozen contract {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"frozen contract {path} must contain a JSON object")
    return payload


DOMAIN_SCHEMA = _load_json(DOMAIN_SCHEMA_PATH)
TOKENIZER_CONTRACT = _load_json(TOKENIZER_CONTRACT_PATH)
DOMAIN_SCHEMA_SHA256 = hashlib.sha256(DOMAIN_SCHEMA_PATH.read_bytes()).hexdigest()
TOKENIZER_CONTRACT_SHA256 = hashlib.sha256(
    TOKENIZER_CONTRACT_PATH.read_bytes()
).hexdigest()
if DOMAIN_SCHEMA.get("schema") != "cppmega_domain_sidecars_v1":
    raise RuntimeError(f"unsupported frozen domain schema: {DOMAIN_SCHEMA_PATH}")


DOMAIN_ROUTE_COLUMNS = (
    "token_domain_ids",
    "token_role_ids",
    "token_entity_ids",
    "token_scope_ids",
    "token_source_doc_ids",
    "token_source_identity_ids",
    "token_confidence_ids",
)

VALID_DOMAIN_IDS = frozenset(
    int(value) for value in DOMAIN_SCHEMA["domain_kinds"].values()
)
VALID_DOMAIN_ROLE_IDS = frozenset(
    int(value) for value in DOMAIN_SCHEMA["role_kinds"].values()
)
VALID_DOMAIN_CONFIDENCE_IDS = frozenset(
    int(value) for value in DOMAIN_SCHEMA["confidence_kinds"].values()
)
_EDGE_FAMILY_TO_COLUMN = {
    "domain": "token_domain_edges",
    "build": "token_build_edges",
    "shell": "token_shell_edges",
    "diagnostic": "token_diagnostic_edges",
    "cross_domain": "token_cross_domain_edges",
}
DOMAIN_EDGE_KINDS_BY_COLUMN = {
    _EDGE_FAMILY_TO_COLUMN[family]: frozenset(int(kind) for kind in kinds)
    for family, kinds in DOMAIN_SCHEMA["edge_families"].items()
}
if set(DOMAIN_EDGE_KINDS_BY_COLUMN) != set(_EDGE_FAMILY_TO_COLUMN.values()):
    raise RuntimeError("frozen domain schema does not define every graph edge family")
VALID_DOMAIN_EDGE_KINDS = frozenset().union(*DOMAIN_EDGE_KINDS_BY_COLUMN.values())

GRAPH_ROUTE_COLUMNS = (
    "token_call_edges",
    "token_type_edges",
    *DOMAIN_EDGE_KINDS_BY_COLUMN,
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
)
GRAPH_ROUTE_COORDINATE_SPACES = {
    "token_call_edges": "chunk_index",
    "token_type_edges": "chunk_index",
    **{column: "token_index" for column in DOMAIN_EDGE_KINDS_BY_COLUMN},
    "token_chunk_starts": "token_index",
    "token_chunk_ends": "token_index",
    "token_chunk_kinds": "chunk_index",
    "token_chunk_dep_levels": "chunk_index",
}

# Reserved delimiter IDs are frozen by the tokenizer artifact while their
# domain mapping is frozen by the domain schema. Neither mapping is duplicated
# in Python.
_ASSIGNMENTS = TOKENIZER_CONTRACT.get("reserved_role_assignments")
_DELIMITER_ROLES = DOMAIN_SCHEMA.get("delimiter_roles")
if not isinstance(_ASSIGNMENTS, dict) or not isinstance(_DELIMITER_ROLES, dict):
    raise RuntimeError("frozen tokenizer/domain contracts are missing delimiter roles")

DOMAIN_DELIMITER_ID_TO_DOMAIN: dict[int, int] = {}
_start_ids: set[int] = set()
_end_ids: set[int] = set()
_delimiter_token_ids: dict[str, int] = {}
for domain_name, raw_spec in _DELIMITER_ROLES.items():
    if not isinstance(raw_spec, dict):
        raise RuntimeError(f"invalid delimiter spec for {domain_name!r}")
    domain_id = int(raw_spec["domain_id"])
    if domain_id not in VALID_DOMAIN_IDS:
        raise RuntimeError(
            f"delimiter {domain_name!r} has unknown domain id {domain_id}"
        )
    for direction, target in (("start", _start_ids), ("end", _end_ids)):
        role = raw_spec[direction]
        if role not in _ASSIGNMENTS:
            raise RuntimeError(f"tokenizer contract is missing delimiter role {role!r}")
        token_id = int(_ASSIGNMENTS[role])
        if token_id in DOMAIN_DELIMITER_ID_TO_DOMAIN:
            raise RuntimeError(f"duplicate domain delimiter token id {token_id}")
        DOMAIN_DELIMITER_ID_TO_DOMAIN[token_id] = domain_id
        _delimiter_token_ids[str(role)] = token_id
        target.add(token_id)

DOMAIN_START_DELIMITER_IDS = frozenset(_start_ids)
DOMAIN_END_DELIMITER_IDS = frozenset(_end_ids)
DOMAIN_DELIMITER_TOKEN_IDS = dict(sorted(_delimiter_token_ids.items()))
DOMAIN_DELIMITER_CONTRACT_METADATA_KEY = "cppmega.domain_delimiter_contract_sha256"
DOMAIN_SCHEMA_SHA256_METADATA_KEY = "cppmega.domain_schema_sha256"
TOKENIZER_CONTRACT_SHA256_METADATA_KEY = "cppmega.tokenizer_contract_sha256"
DOMAIN_DELIMITER_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        DOMAIN_DELIMITER_TOKEN_IDS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
LEGACY_CASE5_CONTRACT_HASH_TRIPLES = frozenset(
    {
        (
            "1f2e35d7917409fc03704d32c2d55d0fb3e29f1bd9e60acca775a392cf2f53e6",
            "9c3517b5a3fda01c4f55d55bc0d12dff4af3edb3db6321bda6c22489061b4fdd",
            "c3bb669015c48e2049e3b82ccb8c98c6eceae0644f7da0b5b8600c573d7087a5",
        ),
        (
            "1f2e35d7917409fc03704d32c2d55d0fb3e29f1bd9e60acca775a392cf2f53e6",
            "9c3517b5a3fda01c4f55d55bc0d12dff4af3edb3db6321bda6c22489061b4fdd",
            "80e73699e26d2c19fe4477cf8194886e52c7a5e114023df27e55d6a69b62c198",
        ),
    }
)


def is_accepted_case5_contract_hash_triple(
    delimiter_contract_sha256: object,
    domain_schema_sha256: object,
    tokenizer_contract_sha256: object,
) -> bool:
    """Accept the current v1 contract or one complete known v1 predecessor."""

    def hash_text(value: object) -> str:
        if isinstance(value, bytes):
            try:
                return value.decode("ascii")
            except UnicodeDecodeError:
                return ""
        return str(value)

    triple = (
        hash_text(delimiter_contract_sha256),
        hash_text(domain_schema_sha256),
        hash_text(tokenizer_contract_sha256),
    )
    current = (
        DOMAIN_DELIMITER_CONTRACT_SHA256,
        DOMAIN_SCHEMA_SHA256,
        TOKENIZER_CONTRACT_SHA256,
    )
    return triple == current or triple in LEGACY_CASE5_CONTRACT_HASH_TRIPLES


def validate_case5_contract_hash_triple(
    delimiter_contract_sha256: object,
    domain_schema_sha256: object,
    tokenizer_contract_sha256: object,
    *,
    where: str,
) -> None:
    if not is_accepted_case5_contract_hash_triple(
        delimiter_contract_sha256,
        domain_schema_sha256,
        tokenizer_contract_sha256,
    ):
        raise ValueError(
            f"{where}: unknown or mixed CASE5 v1 contract hashes: "
            f"delimiter={delimiter_contract_sha256!r}, "
            f"domain={domain_schema_sha256!r}, "
            f"tokenizer={tokenizer_contract_sha256!r}"
        )


CASE5_SCHEMA_METADATA_KEY = "cppmega.case5_schema"
CASE5_SCHEMA_VERSION = "case5_domain_routes_v1"
CASE5_RECEIPT_KEY = "case5_domain_ingestion_receipt"
SOURCE_IDENTITY_REGISTRY_SCHEMA = "cppmega_source_identity_registry_v1"


__all__ = [
    "DOMAIN_DELIMITER_ID_TO_DOMAIN",
    "DOMAIN_DELIMITER_CONTRACT_METADATA_KEY",
    "DOMAIN_DELIMITER_CONTRACT_SHA256",
    "DOMAIN_DELIMITER_TOKEN_IDS",
    "DOMAIN_EDGE_KINDS_BY_COLUMN",
    "DOMAIN_END_DELIMITER_IDS",
    "DOMAIN_ROUTE_COLUMNS",
    "DOMAIN_SCHEMA_SHA256",
    "DOMAIN_SCHEMA_SHA256_METADATA_KEY",
    "DOMAIN_START_DELIMITER_IDS",
    "GRAPH_ROUTE_COLUMNS",
    "GRAPH_ROUTE_COORDINATE_SPACES",
    "LEGACY_CASE5_CONTRACT_HASH_TRIPLES",
    "VALID_DOMAIN_CONFIDENCE_IDS",
    "VALID_DOMAIN_EDGE_KINDS",
    "VALID_DOMAIN_IDS",
    "VALID_DOMAIN_ROLE_IDS",
    "CASE5_SCHEMA_METADATA_KEY",
    "CASE5_RECEIPT_KEY",
    "CASE5_SCHEMA_VERSION",
    "SOURCE_IDENTITY_REGISTRY_SCHEMA",
    "TOKENIZER_CONTRACT_SHA256",
    "TOKENIZER_CONTRACT_SHA256_METADATA_KEY",
    "is_accepted_case5_contract_hash_triple",
    "validate_case5_contract_hash_triple",
]
