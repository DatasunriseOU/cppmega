#!/usr/bin/env python3
"""Fail-closed canonical selection for ``source-pilot-20260803-002``.

Selection never depends on GCS listing order or timestamps.  The checked-in
adjudication binds exact receipt bytes, receipt generation, archive identity,
and archive generation.  The earlier publication containing SQLite sidecars
is explicitly superseded.  This remains a non-training-ready smoke artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_sha256,
    gcs_join,
    load_json_object,
    require_exact_fields,
    require_int,
    require_sha256,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.publish_reducer_smoke import (  # noqa: E402
    SOURCE_REDUCER_SMOKE_PUBLICATION_SCHEMA,
)

SOURCE_PILOT_ADJUDICATION_SCHEMA = "cppmega.source_pilot_canonical_adjudication_v1"
SOURCE_PILOT_SELECTION_SCHEMA = "cppmega.source_pilot_canonical_selection_v1"
DEFAULT_ADJUDICATION_PATH = (
    _REPO_ROOT / "configs/source_pilot_20260803_002_adjudication.json"
)
AUTHORITATIVE_ARTIFACT_SET_SHA256 = (
    "581c6b4685a8975c6414f0e17a909d0f84577b8b91defa4feeebe4f5810862ff"
)
AUTHORITATIVE_ARCHIVE_GENERATION = "1785783190042979"
AUTHORITATIVE_RECEIPT_GENERATION = "1785783193773475"
SUPERSEDED_ARTIFACT_SET_SHA256 = (
    "c88ace50fbde8a9aba775ae8b6b31e7526c7312a37f79aad6386b18b063f2209"
)
EXPECTED_ADJUDICATION_SHA256 = (
    "f49a8df27f7a7464b730e24697d67d19f2d6816479ab065d567d73bafeec70d9"
)
EXPECTED_RUN = {
    "run_id": "source-pilot-20260803-002",
    "gcs_prefix": (
        "gs://natural-bison-491019-t9-cppmega-corpus/" "runs/source-pilot-20260803-002"
    ),
    "manifest_sha256": (
        "83541fc49aedab353d1580695494618f8b7d344a69ef9424fa561586eda29b13"
    ),
    "manifest_file_sha256": (
        "8b928428520138ab90700640af3f2811bc988e3de883442eed938f28eb054ae2"
    ),
    "worker_receipts_sha256": (
        "4ecdae82d069e643e8482324706b56d7dba8360055f147ce63a23dcabf87241d"
    ),
}
EXPECTED_BLOCKING_GATES = [
    "semantic_function_and_chunk_dedup_parity",
    "packed_sidecar_validation",
    "megatron_sealing",
]


class SupersededPublicationError(ContractError):
    """The input exactly matches the explicitly rejected publication."""


def _mapping(
    value: object, expected_fields: set[str], *, where: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    require_exact_fields(value, expected_fields, where=where)
    return value


def _generation(value: object, *, where: str) -> str:
    result = str(value)
    if not result.isdecimal() or int(result) < 1:
        raise ContractError(f"{where} must be a positive decimal generation")
    return result


def _publication_binding(value: object, *, authoritative: bool) -> dict[str, object]:
    role_fields = (
        {"qualification"} if authoritative else {"reason", "disqualifying_members"}
    )
    binding = _mapping(
        value,
        {"artifact_set_sha256", "archive", "publication_receipt", *role_fields},
        where=("authoritative" if authoritative else "superseded") + " publication",
    )
    artifact_set = require_sha256(
        binding["artifact_set_sha256"], where="publication artifact set"
    )
    archive = _mapping(
        binding["archive"],
        {"uri", "generation", "sha256", "size_bytes"},
        where="publication archive binding",
    )
    receipt = _mapping(
        binding["publication_receipt"],
        {"uri", "generation", "sha256", "size_bytes"},
        where="publication receipt binding",
    )
    archive_sha = require_sha256(archive["sha256"], where="archive sha256")
    receipt_sha = require_sha256(receipt["sha256"], where="receipt sha256")
    for item, label in ((archive, "archive"), (receipt, "receipt")):
        _generation(item["generation"], where=f"{label} generation")
        require_int(item["size_bytes"], where=f"{label} size", minimum=1)
    prefix = str(EXPECTED_RUN["gcs_prefix"])
    expected_archive_uri = gcs_join(
        prefix, "reducer-artifacts", artifact_set, f"{archive_sha}.tar.zst"
    )
    expected_receipt_uri = gcs_join(
        prefix, "reducer-receipts", artifact_set, f"{receipt_sha}.receipt.json"
    )
    if (
        validate_gcs_uri(archive["uri"], where="archive URI") != expected_archive_uri
        or validate_gcs_uri(receipt["uri"], where="receipt URI") != expected_receipt_uri
    ):
        raise ContractError("publication binding escaped the source-pilot namespace")
    result = {
        "artifact_set_sha256": artifact_set,
        "archive": dict(archive),
        "publication_receipt": dict(receipt),
    }
    if authoritative:
        if binding["qualification"] != "sidecar_free_closed_sqlite_snapshot":
            raise ContractError("authoritative qualification drifted")
        result["qualification"] = binding["qualification"]
    else:
        if binding["reason"] != "unreceipted_sqlite_sidecars" or binding[
            "disqualifying_members"
        ] != ["global_dedup.sqlite-shm", "global_dedup.sqlite-wal"]:
            raise ContractError("superseded sidecar adjudication drifted")
        result["reason"] = binding["reason"]
        result["disqualifying_members"] = list(binding["disqualifying_members"])
    return result


def load_source_pilot_adjudication(path: Path) -> dict[str, object]:
    _raw, receipt = load_json_object(path, where="source-pilot adjudication receipt")
    require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "run",
            "authoritative",
            "superseded",
            "training_ready",
            "blocking_gates",
            "adjudication_sha256",
        },
        where="source-pilot adjudication receipt",
    )
    digest = require_sha256(receipt["adjudication_sha256"], where="adjudication sha256")
    payload = dict(receipt)
    payload.pop("adjudication_sha256")
    if canonical_sha256(payload) != digest or digest != EXPECTED_ADJUDICATION_SHA256:
        raise ContractError("source-pilot adjudication identity drifted")
    if (
        receipt["schema"] != SOURCE_PILOT_ADJUDICATION_SCHEMA
        or receipt["status"] != "adjudicated"
        or receipt["training_ready"] is not False
        or receipt["run"] != EXPECTED_RUN
        or receipt["blocking_gates"] != EXPECTED_BLOCKING_GATES
    ):
        raise ContractError("source-pilot adjudication contract drifted")
    authoritative = _publication_binding(receipt["authoritative"], authoritative=True)
    superseded_values = receipt["superseded"]
    if not isinstance(superseded_values, list) or len(superseded_values) != 1:
        raise ContractError("adjudication must name exactly one superseded publication")
    superseded = _publication_binding(superseded_values[0], authoritative=False)
    if (
        authoritative["artifact_set_sha256"] != AUTHORITATIVE_ARTIFACT_SET_SHA256
        or authoritative["archive"]["generation"] != AUTHORITATIVE_ARCHIVE_GENERATION
        or authoritative["publication_receipt"]["generation"]
        != AUTHORITATIVE_RECEIPT_GENERATION
        or superseded["artifact_set_sha256"] != SUPERSEDED_ARTIFACT_SET_SHA256
    ):
        raise ContractError("source-pilot canonical/superseded identity drifted")
    return {
        **receipt,
        "run": dict(EXPECTED_RUN),
        "authoritative": authoritative,
        "superseded": [superseded],
    }


def _validate_publication(
    receipt: Mapping[str, object],
    *,
    adjudication: Mapping[str, object],
    binding: Mapping[str, object],
    authoritative: bool,
) -> None:
    require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "training_ready",
            "manifest_sha256",
            "manifest_file_sha256",
            "worker_receipts_sha256",
            "reducer_receipt_sha256",
            "artifact_set_sha256",
            "members",
            "archive",
            "blocking_gates",
        },
        where="source-pilot publication receipt",
    )
    run = adjudication["run"]
    if not isinstance(run, Mapping):
        raise ContractError("adjudication run binding is malformed")
    if (
        receipt["schema"] != SOURCE_REDUCER_SMOKE_PUBLICATION_SCHEMA
        or receipt["status"] != "verified"
        or receipt["training_ready"] is not False
        or receipt["blocking_gates"] != adjudication["blocking_gates"]
        or any(
            receipt[field] != run[field]
            for field in (
                "manifest_sha256",
                "manifest_file_sha256",
                "worker_receipts_sha256",
            )
        )
    ):
        raise ContractError("source-pilot publication contract drifted")
    artifact_set = require_sha256(
        receipt["artifact_set_sha256"], where="publication artifact set"
    )
    if artifact_set != binding["artifact_set_sha256"]:
        raise ContractError("publication artifact set differs from adjudication")
    members = receipt["members"]
    if not isinstance(members, list) or not members:
        raise ContractError("publication member inventory is empty")
    paths: list[str] = []
    normalized: list[dict[str, object]] = []
    for index, member_value in enumerate(members):
        member = _mapping(
            member_value,
            {"path", "sha256", "size_bytes"},
            where=f"publication member {index}",
        )
        path = member["path"]
        pure = PurePosixPath(path) if isinstance(path, str) else PurePosixPath("/")
        if (
            not isinstance(path, str)
            or not path
            or pure.is_absolute()
            or pure.as_posix() != path
            or any(part in {"", ".", ".."} for part in pure.parts)
            or "\\" in path
        ):
            raise ContractError(f"publication member {index} path is unsafe")
        require_sha256(member["sha256"], where=f"publication member {index} sha256")
        require_int(member["size_bytes"], where=f"publication member {index} size")
        paths.append(path)
        normalized.append(dict(member))
    if (
        paths != sorted(set(paths))
        or canonical_sha256(normalized) != artifact_set
        or "global_dedup.sqlite" not in paths
        or "reducer_receipt.json" not in paths
    ):
        raise ContractError("publication member inventory does not close")
    reducer_member = normalized[paths.index("reducer_receipt.json")]
    if reducer_member["sha256"] != receipt["reducer_receipt_sha256"]:
        raise ContractError("reducer receipt is not bound into the member inventory")

    archive = _mapping(
        receipt["archive"],
        {
            "uri",
            "generation",
            "size_bytes",
            "crc32c",
            "md5_hash",
            "compression",
            "level",
            "threads",
            "sha256",
        },
        where="publication archive",
    )
    if (
        {key: archive[key] for key in ("uri", "generation", "sha256", "size_bytes")}
        != binding["archive"]
        or archive["compression"] != "zstd"
        or archive["level"] != 19
        or archive["threads"] != 1
    ):
        raise ContractError("publication archive differs from adjudication")
    sidecars = sorted(
        path
        for path in paths
        if path.startswith("global_dedup.sqlite")
        and path.endswith(("-journal", "-shm", "-wal"))
    )
    expected_sidecars = [] if authoritative else binding["disqualifying_members"]
    if sidecars != expected_sidecars:
        raise ContractError("publication SQLite sidecar evidence drifted")


def _selection(adjudication: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": SOURCE_PILOT_SELECTION_SCHEMA,
        "status": "selected",
        "adjudication_sha256": adjudication["adjudication_sha256"],
        "run": adjudication["run"],
        "selected": adjudication["authoritative"],
        "training_ready": False,
        "blocking_gates": adjudication["blocking_gates"],
    }
    result["selection_sha256"] = canonical_sha256(result)
    return result


def validate_source_pilot_selection(
    value: Mapping[str, object], *, adjudication: Mapping[str, object]
) -> dict[str, object]:
    require_exact_fields(
        value,
        {
            "schema",
            "status",
            "adjudication_sha256",
            "run",
            "selected",
            "training_ready",
            "blocking_gates",
            "selection_sha256",
        },
        where="source-pilot selection receipt",
    )
    digest = require_sha256(value["selection_sha256"], where="selection sha256")
    payload = dict(value)
    payload.pop("selection_sha256")
    if canonical_sha256(payload) != digest or dict(value) != _selection(adjudication):
        raise ContractError("source-pilot selection differs from adjudication")
    return dict(value)


def load_source_pilot_selection(
    path: Path, *, adjudication_path: Path = DEFAULT_ADJUDICATION_PATH
) -> dict[str, object]:
    _raw, selection = load_json_object(path, where="source-pilot selection receipt")
    return validate_source_pilot_selection(
        selection, adjudication=load_source_pilot_adjudication(adjudication_path)
    )


def select_source_pilot_publication(
    *,
    adjudication_path: Path,
    publication_receipt_path: Path,
    publication_receipt_uri: str,
    publication_receipt_generation: str,
) -> dict[str, object]:
    adjudication = load_source_pilot_adjudication(adjudication_path)
    raw, publication = load_json_object(
        publication_receipt_path, where="source-pilot publication receipt"
    )
    observation = {
        "uri": validate_gcs_uri(publication_receipt_uri, where="observed receipt URI"),
        "generation": _generation(
            publication_receipt_generation, where="observed receipt generation"
        ),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    authoritative = adjudication["authoritative"]
    superseded_values = adjudication["superseded"]
    if not isinstance(authoritative, Mapping) or not isinstance(
        superseded_values, list
    ):
        raise ContractError("validated adjudication has malformed bindings")
    superseded = superseded_values[0]
    if not isinstance(superseded, Mapping):
        raise ContractError("validated superseded binding is malformed")
    if observation == authoritative["publication_receipt"]:
        binding, is_authoritative = authoritative, True
    elif observation == superseded["publication_receipt"]:
        binding, is_authoritative = superseded, False
    else:
        raise ContractError(
            "source-pilot publication is neither the exact authoritative receipt "
            "nor an exactly adjudicated superseded receipt"
        )
    _validate_publication(
        publication,
        adjudication=adjudication,
        binding=binding,
        authoritative=is_authoritative,
    )
    if not is_authoritative:
        raise SupersededPublicationError(
            f"{binding['artifact_set_sha256']} is explicitly superseded: "
            f"{binding['reason']} ({', '.join(binding['disqualifying_members'])})"
        )
    return _selection(adjudication)


def write_selection_receipt(path: Path, selection: Mapping[str, object]) -> None:
    validated = validate_source_pilot_selection(
        selection,
        adjudication=load_source_pilot_adjudication(DEFAULT_ADJUDICATION_PATH),
    )
    if path.is_symlink():
        raise ContractError(f"selection output must not be a symlink: {path}")
    if path.exists():
        _raw, existing = load_json_object(path, where="source-pilot selection receipt")
        if existing != validated:
            raise ContractError(f"selection output already exists with drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{path.name}.", dir=path.parent
    ) as raw_tmp:
        stage = Path(raw_tmp) / "selection.json"
        atomic_write_json(stage, validated)
        try:
            os.link(stage, path)
        except FileExistsError:
            _raw, existing = load_json_object(
                path, where="source-pilot selection receipt"
            )
            if existing != validated:
                raise ContractError(
                    f"selection output raced with different bytes: {path}"
                )
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION_PATH)
    parser.add_argument("--publication-receipt", required=True, type=Path)
    parser.add_argument("--publication-receipt-uri", required=True)
    parser.add_argument("--publication-receipt-generation", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        selection = select_source_pilot_publication(
            adjudication_path=args.adjudication,
            publication_receipt_path=args.publication_receipt,
            publication_receipt_uri=args.publication_receipt_uri,
            publication_receipt_generation=args.publication_receipt_generation,
        )
        write_selection_receipt(args.output, selection)
    except SupersededPublicationError as exc:
        parser.exit(3, f"source-pilot publication superseded: {exc}\n")
    except (ContractError, OSError, ValueError) as exc:
        parser.exit(2, f"source-pilot adjudication failed: {exc}\n")
    print(selection["selection_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "AUTHORITATIVE_ARCHIVE_GENERATION",
    "AUTHORITATIVE_ARTIFACT_SET_SHA256",
    "AUTHORITATIVE_RECEIPT_GENERATION",
    "DEFAULT_ADJUDICATION_PATH",
    "EXPECTED_ADJUDICATION_SHA256",
    "SOURCE_PILOT_ADJUDICATION_SCHEMA",
    "SOURCE_PILOT_SELECTION_SCHEMA",
    "SUPERSEDED_ARTIFACT_SET_SHA256",
    "SupersededPublicationError",
    "load_source_pilot_adjudication",
    "load_source_pilot_selection",
    "select_source_pilot_publication",
    "validate_source_pilot_selection",
    "write_selection_receipt",
]
