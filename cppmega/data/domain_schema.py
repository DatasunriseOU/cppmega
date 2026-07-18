"""Versioned domain sidecar schema for cppmega world-code documents.

The tokenizer artifact is frozen.  Domain boundaries therefore use logical
roles mapped to existing ``<RESERVED_N>`` ids in ``tokenizer_contract.py``.
This module owns the integer enums used by parquet sidecars, parser outputs,
MLX route tests, and the Megatron sidecar converter.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from typing import Any

from cppmega.data.tokenizer_contract import (
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    DOMAIN_DELIMITER_TOKEN_IDS,
    TOKENIZER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY,
)


class DomainKind(IntEnum):
    UNKNOWN = 0
    CPP = 1
    CMAKE = 2
    MAKE = 3
    NINJA = 4
    BAZEL = 5
    AUTOCONF = 6
    AUTOMAKE = 7
    MESON = 8
    GN = 9
    SCONS = 10
    XMAKE = 11
    COMPILE_COMMANDS = 12
    CONFIGURE = 13
    BASH = 20
    ZSH = 21
    SH = 22
    TCSH = 23
    KSH = 24
    SQL = 30
    PYTHON = 31
    COMPILER_DIAGNOSTIC = 40
    BUILD_DIAGNOSTIC = 41
    COMPILER_ERROR = 42
    BUILD_ERROR = 43
    LINKER_ERROR = 44
    TEST_OUTPUT = 45
    TOOL_OUTPUT = 46
    LINKER_DIAGNOSTIC = 47
    SANITIZER_OUTPUT = 48


class DomainRoleKind(IntEnum):
    NONE = 0
    DELIMITER = 1
    KEYWORD = 2
    IDENTIFIER = 3
    TARGET = 4
    VARIABLE = 5
    COMMAND = 6
    PATH = 7
    OPTION = 8
    STRING = 9
    LABEL = 10
    RULE = 11
    ATTRIBUTE = 12
    SOURCE = 13
    LIBRARY = 14
    PREREQUISITE = 15
    OUTPUT = 16
    INPUT = 17
    ENVIRONMENT = 18
    REDIRECT = 19
    PIPE = 20
    COMMENT = 21
    DOCSTRING = 22
    PREPROCESSOR = 23
    SEVERITY = 30
    MESSAGE = 31
    FILE = 32
    LINE = 33
    COLUMN = 34
    SYMBOL = 35
    FIXIT = 36
    NOTE = 37
    EXIT_CODE = 38
    TEST_NAME = 39


class DomainEdgeKind(IntEnum):
    UNKNOWN = 0
    AST_PARENT = 1
    CALL = 2
    TYPE = 3
    DEF_USE = 4
    INCLUDE = 5
    MACRO_PARAM_USE = 6
    MACRO_INVOCATION = 7
    MACRO_CONDITION = 8
    MACRO_REDEFINITION = 9
    MACRO_INCLUDE_ORDER = 10
    MACRO_EXPANSION_CONDITION = 11
    MACRO_EXPANSION_REDEFINITION = 12
    MACRO_EXPANSION_INCLUDE_ORDER = 13
    BUILD_TARGET_DEP = 20
    BUILD_TARGET_SOURCE = 21
    BUILD_RULE_COMMAND = 22
    BUILD_ACTION_INPUT = 23
    BUILD_ACTION_OUTPUT = 24
    BUILD_VAR_DEF_USE = 25
    BUILD_COMMAND_TARGET = 26
    SHELL_PIPE = 40
    SHELL_REDIR_IN = 41
    SHELL_REDIR_OUT = 42
    SHELL_VAR_DEF_USE = 43
    SHELL_COMMAND_FILE = 44
    DIAG_PRIMARY_LOCATION = 60
    DIAG_NOTE = 61
    DIAG_FIXIT = 62
    DIAG_COMMAND = 63
    DIAG_BUILD_TARGET = 64
    LINK_UNDEFINED_SYMBOL = 70
    LINK_CANDIDATE_DEF = 71
    TEST_FAILURE_LOCATION = 80
    TOOL_ACTION_RESULT = 90
    EMBEDDED_DOMAIN = 100


class ParseConfidence(IntEnum):
    ABSENT = 0
    RAW = 1
    HEURISTIC = 2
    PARTIAL = 3
    EXACT = 4


DOMAIN_SCHEMA_PATH = Path(__file__).with_name("domain_schema_v1.json")
try:
    _DOMAIN_SCHEMA_BYTES = DOMAIN_SCHEMA_PATH.read_bytes()
    DOMAIN_SCHEMA = json.loads(_DOMAIN_SCHEMA_BYTES.decode("utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise RuntimeError(
        f"cannot load frozen domain schema {DOMAIN_SCHEMA_PATH}: {exc}"
    ) from exc
if not isinstance(DOMAIN_SCHEMA, dict):
    raise RuntimeError(
        f"frozen domain schema {DOMAIN_SCHEMA_PATH} must be an object"
    )
DOMAIN_SCHEMA_SHA256 = hashlib.sha256(_DOMAIN_SCHEMA_BYTES).hexdigest()
DOMAIN_SCHEMA_SHA256_METADATA_KEY = "cppmega.domain_schema_sha256"
PREVIOUS_CASE5_V1_CONTRACT_HASH_TRIPLES = frozenset(
    {
        (
            "9c3517b5a3fda01c4f55d55bc0d12dff4af3edb3db6321bda6c22489061b4fdd",
            "c3bb669015c48e2049e3b82ccb8c98c6eceae0644f7da0b5b8600c573d7087a5",
            "1f2e35d7917409fc03704d32c2d55d0fb3e29f1bd9e60acca775a392cf2f53e6",
        ),
        (
            "9c3517b5a3fda01c4f55d55bc0d12dff4af3edb3db6321bda6c22489061b4fdd",
            "80e73699e26d2c19fe4477cf8194886e52c7a5e114023df27e55d6a69b62c198",
            "1f2e35d7917409fc03704d32c2d55d0fb3e29f1bd9e60acca775a392cf2f53e6",
        ),
    }
)
if DOMAIN_SCHEMA.get("schema") != "cppmega_domain_sidecars_v1":
    raise RuntimeError(f"unsupported frozen domain schema: {DOMAIN_SCHEMA_PATH}")


def validate_case5_contract_metadata(
    metadata: Mapping[bytes, bytes] | None,
    *,
    where: str | Path,
) -> None:
    """Require one exact domain/tokenizer/delimiter CASE5 hash triple."""

    actual_metadata = metadata or {}
    actual_triple = (
        actual_metadata.get(DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8")),
        actual_metadata.get(TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode("utf-8")),
        actual_metadata.get(DOMAIN_DELIMITER_CONTRACT_METADATA_KEY.encode("utf-8")),
    )
    accepted_triples = {
        (
            DOMAIN_SCHEMA_SHA256.encode("ascii"),
            TOKENIZER_CONTRACT_SHA256.encode("ascii"),
            DOMAIN_DELIMITER_CONTRACT_SHA256.encode("ascii"),
        ),
        *(
            (
                domain_hash.encode("ascii"),
                tokenizer_hash.encode("ascii"),
                delimiter_hash.encode("ascii"),
            )
            for domain_hash, tokenizer_hash, delimiter_hash in (
                PREVIOUS_CASE5_V1_CONTRACT_HASH_TRIPLES
            )
        ),
    }
    if actual_triple not in accepted_triples:
        raise ValueError(
            f"{where}: missing or stale frozen CASE5 contract hashes: "
            f"{DOMAIN_SCHEMA_SHA256_METADATA_KEY}={actual_triple[0]!r}; "
            f"{TOKENIZER_CONTRACT_SHA256_METADATA_KEY}={actual_triple[1]!r}; "
            f"{DOMAIN_DELIMITER_CONTRACT_METADATA_KEY}={actual_triple[2]!r}"
        )


def _enum_contract(enum_type: type[IntEnum], field: str) -> dict[str, int]:
    expected = DOMAIN_SCHEMA.get(field)
    actual = {item.name: int(item) for item in enum_type}
    if expected != actual:
        raise RuntimeError(
            f"{DOMAIN_SCHEMA_PATH}: {field} drift: expected={expected}, actual={actual}"
        )
    return actual


_enum_contract(DomainKind, "domain_kinds")
_enum_contract(DomainRoleKind, "role_kinds")
_enum_contract(DomainEdgeKind, "edge_kinds")
_enum_contract(ParseConfidence, "confidence_kinds")

DOMAIN_EDGE_FAMILIES: dict[str, frozenset[DomainEdgeKind]] = {
    family: frozenset(DomainEdgeKind(int(kind)) for kind in kinds)
    for family, kinds in DOMAIN_SCHEMA["edge_families"].items()
}
VALID_DOMAIN_EDGE_KINDS = frozenset(
    kind for kind in DomainEdgeKind if kind != DomainEdgeKind.UNKNOWN
)
DOMAIN_EDGE_FIELD_FAMILIES: dict[str, str] = {
    "domain_edges": "domain",
    "build_edges": "build",
    "shell_edges": "shell",
    "diagnostic_edges": "diagnostic",
    "cross_domain_edges": "cross_domain",
}
DOMAIN_EDGE_FAMILY_FIELDS: dict[str, str] = {
    family: field for field, family in DOMAIN_EDGE_FIELD_FAMILIES.items()
}
TOKEN_DOMAIN_EDGE_COLUMN_FAMILIES: dict[str, str] = {
    f"token_{field}": family
    for field, family in DOMAIN_EDGE_FIELD_FAMILIES.items()
}


def domain_edge_family(kind: DomainEdgeKind | int) -> str:
    """Return the one graph channel that owns ``kind``.

    Unknown and sentinel kinds are rejected. They must never be serialized as a
    plausible edge on a different graph channel.
    """

    try:
        edge_kind = DomainEdgeKind(int(kind))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown domain edge kind {kind!r}") from exc
    if edge_kind == DomainEdgeKind.UNKNOWN:
        raise ValueError("unknown domain edge kind 0")
    for family, kinds in DOMAIN_EDGE_FAMILIES.items():
        if edge_kind in kinds:
            return family
    raise ValueError(f"domain edge kind {int(edge_kind)} has no graph family")


def validate_domain_edge_kind(
    kind: DomainEdgeKind | int,
    *,
    family: str | None = None,
) -> DomainEdgeKind:
    """Validate an edge kind and, when supplied, its graph-channel family."""

    actual_family = domain_edge_family(kind)
    if family == "aggregate":
        return DomainEdgeKind(int(kind))
    if family is not None and family not in DOMAIN_EDGE_FAMILIES:
        raise ValueError(f"unknown domain edge family {family!r}")
    if family is not None and actual_family != family:
        raise ValueError(
            f"domain edge kind {int(kind)} belongs to {actual_family}, not {family}"
        )
    return DomainEdgeKind(int(kind))


def normalize_domain_edge_record(
    edge: Any,
    *,
    family: str,
) -> tuple[int, int, int]:
    """Normalize one serialized edge without inventing endpoints or a kind."""

    if hasattr(edge, "as_py"):
        edge = edge.as_py()
    if isinstance(edge, Mapping):
        endpoint_pairs = (
            ("from_char", "to_char"),
            ("from", "to"),
            ("src", "dst"),
        )
        selected = next(
            (
                (src_key, dst_key)
                for src_key, dst_key in endpoint_pairs
                if src_key in edge and dst_key in edge
            ),
            None,
        )
        if selected is None or "kind" not in edge:
            raise ValueError("domain edge missing src/dst/kind")
        src = edge[selected[0]]
        dst = edge[selected[1]]
        kind = edge["kind"]
    elif (
        (
            isinstance(edge, Sequence)
            or (hasattr(edge, "__len__") and hasattr(edge, "__getitem__"))
        )
        and not isinstance(edge, (str, bytes, bytearray))
        and len(edge) == 3
    ):
        src, dst, kind = edge
    else:
        raise ValueError("domain edge must be a mapping or length-3 sequence")
    src_i = int(src)
    dst_i = int(dst)
    if src_i < 0 or dst_i < 0:
        raise ValueError(f"domain edge endpoints must be non-negative: {src_i}->{dst_i}")
    edge_kind = validate_domain_edge_kind(int(kind), family=family)
    return src_i, dst_i, int(edge_kind)


def canonicalize_domain_edge_fields(
    record: Mapping[str, Any],
    *,
    source_length: int | None = None,
) -> dict[str, list[dict[str, int]]]:
    """Return family-pure character-edge fields from one enriched record.

    Older producers mirrored every typed edge into ``domain_edges`` as an
    aggregate while also writing specialized fields. The persisted graph
    contract has one channel per edge family, so route that legacy aggregate
    by its validated kind and coalesce only exact mirrored occurrences while
    preserving repeated-edge multiplicity. A typed edge placed in the wrong
    specialized field remains a hard error.
    """

    if source_length is not None:
        source_length = int(source_length)
        if source_length < 0:
            raise ValueError("domain edge source_length must be non-negative")

    canonical = {field: [] for field in DOMAIN_EDGE_FIELD_FAMILIES}
    aggregate_counts: dict[str, Counter[tuple[int, int, int]]] = {
        field: Counter() for field in DOMAIN_EDGE_FIELD_FAMILIES
    }
    specialized_counts: dict[str, Counter[tuple[int, int, int]]] = {
        field: Counter() for field in DOMAIN_EDGE_FIELD_FAMILIES
    }

    for source_field, expected_family in DOMAIN_EDGE_FIELD_FAMILIES.items():
        validation_family = (
            "aggregate" if source_field == "domain_edges" else expected_family
        )
        for edge_index, edge in enumerate(record.get(source_field, []) or []):
            try:
                src, dst, kind = normalize_domain_edge_record(
                    edge,
                    family=validation_family,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source_field}[{edge_index}]: {exc}") from exc
            if source_length is not None:
                if src == dst:
                    if src > source_length:
                        raise ValueError(
                            f"{source_field}[{edge_index}]: character point {src} "
                            "is outside source point bounds "
                            f"[0, {source_length}]"
                        )
                elif src >= source_length or dst >= source_length:
                    raise ValueError(
                        f"{source_field}[{edge_index}]: character edge endpoint "
                        f"{src}->{dst} is outside source text bounds "
                        f"[0, {source_length})"
                    )
            target_family = domain_edge_family(kind)
            target_field = DOMAIN_EDGE_FAMILY_FIELDS[target_family]
            triple = (src, dst, kind)
            if source_field == "domain_edges":
                aggregate_counts[target_field][triple] += 1
            else:
                specialized_counts[target_field][triple] += 1
                if (
                    specialized_counts[target_field][triple]
                    <= aggregate_counts[target_field][triple]
                ):
                    continue
            canonical[target_field].append(
                {"from_char": src, "to_char": dst, "kind": kind}
            )
    return canonical


def normalize_embedded_domain_spans(
    raw_spans: Any,
    *,
    source_length: int,
) -> list[dict[str, Any]]:
    """Validate sorted, non-overlapping ``[start, end)`` embedded spans."""

    source_length = int(source_length)
    if source_length < 0:
        raise ValueError("embedded domain source_length must be non-negative")
    normalized: list[dict[str, Any]] = []
    for raw in raw_spans or []:
        if hasattr(raw, "as_py"):
            raw = raw.as_py()
        if not isinstance(raw, Mapping):
            raise ValueError("embedded domain span must be a mapping")
        if not {"start", "end", "domain_kind"} <= set(raw):
            raise ValueError("embedded domain span requires start/end/domain_kind")
        start = int(raw["start"])
        end = int(raw["end"])
        if start < 0 or end <= start or end > source_length:
            raise ValueError(
                f"invalid embedded domain span {start}:{end} for source length "
                f"{source_length}"
            )
        try:
            domain = DomainKind(int(raw["domain_kind"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unknown embedded domain_kind {raw['domain_kind']!r}"
            ) from exc
        if domain == DomainKind.UNKNOWN:
            raise ValueError("embedded UNKNOWN domain disables delimiter insertion")
        span = dict(raw)
        span.update(start=start, end=end, domain_kind=int(domain))
        normalized.append(span)

    normalized.sort(key=lambda span: (int(span["start"]), int(span["end"])))
    previous_end = 0
    for span in normalized:
        if int(span["start"]) < previous_end:
            raise ValueError("overlapping embedded domain spans are unsupported")
        previous_end = int(span["end"])
    return normalized


def remap_embedded_domain_spans(
    raw_spans: Any,
    *,
    source_length: int,
    kept_indices: Sequence[int] | None,
    prefix_length: int = 0,
) -> list[dict[str, Any]]:
    """Remap spans through exact character filtering and prefix insertion."""

    spans = normalize_embedded_domain_spans(
        raw_spans,
        source_length=source_length,
    )
    prefix_length = int(prefix_length)
    if prefix_length < 0:
        raise ValueError("embedded domain prefix_length must be non-negative")
    if kept_indices is None:
        return [
            {
                **span,
                "start": int(span["start"]) + prefix_length,
                "end": int(span["end"]) + prefix_length,
            }
            for span in spans
        ]

    kept = [int(value) for value in kept_indices]
    if any(value < 0 or value >= source_length for value in kept):
        raise ValueError("kept character mapping is outside source text bounds")
    if any(left >= right for left, right in zip(kept, kept[1:])):
        raise ValueError("kept character mapping must be strictly increasing")

    remapped: list[dict[str, Any]] = []
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        left = bisect_left(kept, start)
        right = bisect_left(kept, end)
        covered = kept[left:right]
        if not covered:
            continue
        if (
            len(covered) != end - start
            or covered[0] != start
            or covered[-1] != end - 1
        ):
            raise ValueError(
                f"cannot exactly remap embedded domain span {start}:{end} "
                "through filtered text"
            )
        remapped.append(
            {
                **span,
                "start": left + prefix_length,
                "end": right + prefix_length,
            }
        )
    return remapped


def slice_embedded_domain_spans(
    raw_spans: Any,
    *,
    source_length: int,
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    """Clip embedded spans to one exact source slice and make them slice-local."""

    start = int(start)
    end = int(end)
    if start < 0 or end < start or end > int(source_length):
        raise ValueError(
            f"invalid embedded domain slice {start}:{end} for source length "
            f"{source_length}"
        )
    spans = normalize_embedded_domain_spans(
        raw_spans,
        source_length=source_length,
    )
    sliced: list[dict[str, Any]] = []
    for span in spans:
        clipped_start = max(int(span["start"]), start)
        clipped_end = min(int(span["end"]), end)
        if clipped_start >= clipped_end:
            continue
        sliced.append(
            {
                **span,
                "start": clipped_start - start,
                "end": clipped_end - start,
            }
        )
    return sliced


DOMAIN_DELIMITER_ROLES: dict[DomainKind, tuple[str, str]] = {
    DomainKind(int(spec["domain_id"])): (str(spec["start"]), str(spec["end"]))
    for spec in DOMAIN_SCHEMA["delimiter_roles"].values()
}


def delimiter_token_ids(domain: DomainKind) -> tuple[int, int]:
    """Return the reserved-token ids for ``domain``'s opener/closer."""

    try:
        start_role, end_role = DOMAIN_DELIMITER_ROLES[DomainKind(domain)]
    except KeyError as exc:
        raise KeyError(f"domain {domain!r} has no delimiter role pair") from exc
    return (
        DOMAIN_DELIMITER_TOKEN_IDS[start_role],
        DOMAIN_DELIMITER_TOKEN_IDS[end_role],
    )


def validate_domain_delimiter_contract(
    delimiter_roles: Mapping[DomainKind, tuple[str, str]] | None = None,
) -> None:
    """Fail loud if any logical domain delimiter is missing or malformed."""

    roles = DOMAIN_DELIMITER_ROLES if delimiter_roles is None else delimiter_roles
    expected_domains = set(DomainKind) - {DomainKind.UNKNOWN}
    actual_domains = set(roles)
    if actual_domains != expected_domains:
        raise ValueError(
            "domain delimiter map is incomplete: "
            f"missing={sorted(domain.name for domain in expected_domains - actual_domains)} "
            f"extra={sorted(domain.name for domain in actual_domains - expected_domains)}"
        )
    missing: list[str] = []
    seen_ids: dict[int, str] = {}
    for domain, (start_role, end_role) in roles.items():
        role_base = "CPP_CODE" if domain == DomainKind.CPP else domain.name
        expected_pair = (f"{role_base}_START", f"{role_base}_END")
        if (start_role, end_role) != expected_pair:
            raise ValueError(
                f"{domain.name}: delimiter roles must be {expected_pair}, got "
                f"{(start_role, end_role)}"
            )
        if start_role not in DOMAIN_DELIMITER_TOKEN_IDS:
            missing.append(start_role)
        if end_role not in DOMAIN_DELIMITER_TOKEN_IDS:
            missing.append(end_role)
        if (
            start_role in DOMAIN_DELIMITER_TOKEN_IDS
            and end_role in DOMAIN_DELIMITER_TOKEN_IDS
        ):
            start_id = DOMAIN_DELIMITER_TOKEN_IDS[start_role]
            end_id = DOMAIN_DELIMITER_TOKEN_IDS[end_role]
            if start_id == end_id:
                raise ValueError(f"{domain.name}: start/end delimiter ids collide")
            for role, token_id in ((start_role, start_id), (end_role, end_id)):
                existing = seen_ids.setdefault(token_id, role)
                if existing != role:
                    raise ValueError(
                        f"domain delimiter id {token_id} maps to both "
                        f"{existing} and {role}"
                    )
    if missing:
        raise ValueError(
            f"missing domain delimiter token roles: {sorted(set(missing))}"
        )
    expected_roles = {role for role_pair in roles.values() for role in role_pair}
    contract_roles = set(DOMAIN_DELIMITER_TOKEN_IDS)
    if contract_roles != expected_roles:
        raise ValueError(
            "domain delimiter parser map differs from tokenizer contract: "
            f"unmapped={sorted(contract_roles - expected_roles)} "
            f"undefined={sorted(expected_roles - contract_roles)}"
        )


validate_domain_delimiter_contract()


__all__ = [
    "DOMAIN_SCHEMA",
    "DOMAIN_SCHEMA_PATH",
    "DOMAIN_SCHEMA_SHA256",
    "DOMAIN_SCHEMA_SHA256_METADATA_KEY",
    "DOMAIN_DELIMITER_ROLES",
    "DOMAIN_EDGE_FAMILIES",
    "DOMAIN_EDGE_FAMILY_FIELDS",
    "DOMAIN_EDGE_FIELD_FAMILIES",
    "TOKEN_DOMAIN_EDGE_COLUMN_FAMILIES",
    "VALID_DOMAIN_EDGE_KINDS",
    "DomainEdgeKind",
    "DomainKind",
    "DomainRoleKind",
    "ParseConfidence",
    "PREVIOUS_CASE5_V1_CONTRACT_HASH_TRIPLES",
    "canonicalize_domain_edge_fields",
    "domain_edge_family",
    "delimiter_token_ids",
    "normalize_domain_edge_record",
    "normalize_embedded_domain_spans",
    "remap_embedded_domain_spans",
    "slice_embedded_domain_spans",
    "validate_case5_contract_metadata",
    "validate_domain_delimiter_contract",
    "validate_domain_edge_kind",
]
