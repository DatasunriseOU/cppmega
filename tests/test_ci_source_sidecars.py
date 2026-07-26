from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from scripts.ci_content_store import CIContentStore, hash_token_sequence
from scripts.ci_log_sidecars import canonicalize_ci_log
from scripts.ci_source_sidecars import (
    _FRAME_HEADER,
    CASE5_EXPORT_SCHEMA,
    CHECKOUT_PROVENANCE_UNRESOLVABLE,
    CONTENT_SEMANTICS,
    FETCH_RECEIPT_SCHEMA,
    GENERATED_OR_MUTATED_UNRESOLVABLE,
    INVENTORY_SCHEMA,
    PATH_ABSENT,
    RECEIPT_SCHEMA,
    REPRESENTATIVE_LEDGER_SCHEMA,
    RESOLVED,
    UNSUPPORTED_OBJECT,
    ExtractionError,
    LocalGitResolver,
    SourceSidecarStore,
    SourceStoreError,
    _checkout_binding,
    _content_store_sqlite_logical_sha256,
    _hash_records,
    _inventory_logical_sha256,
    _sha256_file,
    extract_binding_inventory,
    main,
    materialize_inventory,
    normalize_source_candidates,
    normalize_source_path,
    verify_binding_inventory,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> bytes:
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _run(
    *args: str,
    cwd: Path | None = None,
    input_bytes: bytes | None = None,
) -> bytes:
    result = subprocess.run(
        list(args),
        cwd=cwd,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{' '.join(args)} failed:\n{result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def _git_fixture(
    tmp_path: Path,
    *,
    object_format: str = "sha1",
) -> tuple[Path, str, str]:
    suffix = "" if object_format == "sha1" else f"-{object_format}"
    work = tmp_path / f"work{suffix}"
    mirror = tmp_path / f"repo{suffix}.git"
    work.mkdir()
    init = ["git", "init", "-q"]
    if object_format != "sha1":
        init.append(f"--object-format={object_format}")
    init.append(str(work))
    _run(*init)
    _run("git", "config", "user.name", "CI Source Test", cwd=work)
    _run("git", "config", "user.email", "ci-source@example.test", cwd=work)

    (work / "src" / "nested").mkdir(parents=True)
    payload = b"int main() { return 0; }\n"
    (work / "src" / "nested" / "main.cpp").write_bytes(payload)
    (work / "src" / "copy.cpp").write_bytes(payload)
    (work / "src" / "large.cpp").write_bytes(b"x" * 256)
    (work / "assets").mkdir()
    (work / "assets" / "bytes.bin").write_bytes(b"\x00\xff\x10binary\r\n")
    (work / "model.lfs").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"a" * 64 + b"\nsize 123456\n"
    )
    os.symlink("src/nested/main.cpp", work / "main-link")
    _run("git", "add", ".", cwd=work)
    _run("git", "commit", "-q", "-m", "base", cwd=work)
    base = _run("git", "rev-parse", "HEAD", cwd=work).decode().strip()
    _run(
        "git",
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{base},vendor/component",
        cwd=work,
    )
    _run("git", "commit", "-q", "-m", "gitlink", cwd=work)
    head = _run("git", "rev-parse", "HEAD", cwd=work).decode().strip()
    _run("git", "clone", "-q", "--bare", str(work), str(mirror))
    return mirror, head, base


def _binding(
    head: str,
    source_path: str,
    *,
    repository: str = "owner/repo",
    status: str = RESOLVED,
) -> dict[str, object]:
    return {
        "schema": INVENTORY_SCHEMA,
        "record_type": "binding",
        "repository": repository,
        "head_sha": head,
        "source_path": source_path,
        "normalization_status": status,
        "normalized_candidates": [source_path] if status == RESOLVED else [],
    }


def _reference(
    binding: dict[str, object],
    *,
    ordinal: int,
) -> dict[str, object]:
    digit = f"{(ordinal % 10):x}"
    occurrence_key = {
        "repo": "owner/repo",
        "run_attempt": "1:1",
        "job": f"linux-{ordinal}",
        "step": "compile:0",
        "chunk_ordinal": ordinal,
    }
    normalization = {
        "schema": "cppmega_ci_source_path_normalization_v2",
        "status": binding["normalization_status"],
        "candidates": binding["normalized_candidates"],
        "source_input": binding["source_path"],
        "cwd": "/home/runner/work/workspace/checkout",
        "reason": None,
    }
    return {
        "schema": INVENTORY_SCHEMA,
        "record_type": "reference",
        "repository": binding["repository"],
        "head_sha": binding["head_sha"],
        "source_path": binding["source_path"],
        "token_sequence_sha256": digit * 64,
        "representative_occurrence_key": occurrence_key,
        "representative_content_sha256": "a" * 64,
        "representative_provenance_sha256": "b" * 64,
        "representative_selection_record_sha256": "c" * 64,
        "action_index": 0,
        "action_entity_id": "entity:compile",
        "action_shape_sha256": "d" * 64,
        "command_sha256": "e" * 64,
        "source_input_index": 0,
        "source_input": binding["source_path"],
        "cwd": "/home/runner/work/workspace/checkout",
        "normalization": normalization,
        "checkout_evidence": {
            "workflow_event": "push",
            "reason": None,
        },
    }


def _frozen_fetch_state_binding(root: Path) -> dict[str, object]:
    sidecar_set_sha256 = "9" * 64
    return {
        "schema": "cppmega_ci_stream_fetch_v3",
        "artifact": {
            "path": str(root / "fetch-state.sqlite3"),
            "byte_size": 16_384,
            "mtime_ns": 1_774_444_500_000_000_000,
            "inode": 424_242,
            "sha256": "a" * 64,
        },
        "sqlite_schema_sha256": "b" * 64,
        "sqlite_logical_sha256": "c" * 64,
        "settings": {
            "schema": "cppmega_ci_stream_fetch_v3",
            "inventory_path": str(root / "inventory.sqlite3"),
            "content_store_path": str(root / "ci-store"),
            "tokenizer_contract": '{"schema":"cppmega-tokenizer-v1"}',
            "tokenizer_fingerprint": "tokenizer-test-v1",
            "fetcher_script_sha256": "d" * 64,
            "parser_script_sha256": "e" * 64,
            "content_store_script_sha256": "f" * 64,
            "chunk_semantics": (
                "parser-dedup-text-cppmega-training-tokenizer-payload-only-"
                "no-framing-v2"
            ),
            "created_at": "2026-07-26T04:35:00Z",
        },
        "summary": {
            "attempt_statuses": {"done": 1},
            "attempts_terminal": 1,
            "members": 1,
            "chunks": 2,
            "occurrence_tokens": 4,
            "requests": 3,
            "sidecar_set_sha256": sidecar_set_sha256,
        },
        "sidecar_set_sha256": sidecar_set_sha256,
    }


def _write_inventory(
    path: Path,
    bindings: list[dict[str, object]],
) -> Path:
    ordered_bindings = sorted(
        bindings,
        key=lambda value: (
            str(value["repository"]),
            str(value["head_sha"]),
            str(value["source_path"]),
        ),
    )
    references = [
        _reference(binding, ordinal=index + 1)
        for index, binding in enumerate(ordered_bindings)
    ]
    references.sort(
        key=lambda value: (
            str(value["repository"]),
            str(value["head_sha"]),
            str(value["source_path"]),
            hashlib.sha256(_canonical(value)).hexdigest(),
        )
    )
    frozen_fetch_state = _frozen_fetch_state_binding(path.parent)
    header: dict[str, object] = {
        "schema": INVENTORY_SCHEMA,
        "record_type": "header",
        "occurrence_schema": "cppmega_ci_chunk_occurrence_v3",
        "training_sidecar_schema": "cppmega_ci_chunk_training_sidecars_v2",
        "normalization_schema": "cppmega_ci_source_path_normalization_v2",
        "content_semantics": CONTENT_SEMANTICS,
        "occurrence_set_sha256": "1" * 64,
        "upstream_fetch_receipt_sha256": "2" * 64,
        "frozen_fetch_state": frozen_fetch_state,
        "frozen_fetch_state_sha256": hashlib.sha256(
            _canonical(frozen_fetch_state)
        ).hexdigest(),
        "content_store_receipt_sha256": "3" * 64,
        "content_store_sqlite_schema_sha256": "4" * 64,
        "content_store_sqlite_logical_sha256": "5" * 64,
        "case5_export_receipt_sha256": "6" * 64,
        "representative_ledger_schema": REPRESENTATIVE_LEDGER_SCHEMA,
        "representative_count": len(references),
        "representative_ledger_sha256": "7" * 64,
        "representative_ledger_artifact_sha256": "8" * 64,
        "binding_count": len(ordered_bindings),
        "reference_count": len(references),
        "binding_records_sha256": _hash_records(
            "cppmega-ci-source-binding-records-v2",
            ordered_bindings,
        ),
        "reference_records_sha256": _hash_records(
            "cppmega-ci-source-reference-records-v2",
            references,
        ),
    }
    header["inventory_logical_sha256"] = _inventory_logical_sha256(header)
    with path.open("wb") as handle:
        for record in [header, *ordered_bindings, *references]:
            handle.write(_canonical(record) + b"\n")
    verify_binding_inventory(path)
    return path


def _new_source_store(
    root: Path,
    inventory_path: Path,
    *,
    max_pack_bytes: int = 1024,
) -> SourceSidecarStore:
    return SourceSidecarStore(
        root,
        inventory=verify_binding_inventory(inventory_path),
        max_pack_bytes=max_pack_bytes,
    )


def _valid_provenance(
    head: str,
    *,
    event: str = "push",
    repository: str = "owner/repo",
    source_repository: str = "owner/repo",
    cwd: str | None = "/home/runner/work/workspace/checkout/build",
    binding_repository: str | None = None,
    binding_method: str = "relative_source_path_v1",
    binding_count_delta: int = 0,
    additional_binding_repository: str | None = None,
    source_input: str = "../src/nested/main.cpp",
    bound_source_path: str = "src/nested/main.cpp",
) -> dict[str, object]:
    bindings = [
        {
            "repository": binding_repository or repository,
            "head_sha": head,
            "source_path": bound_source_path,
            "confidence": {
                "score": 0.95,
                "level": "high",
                "source": binding_method,
            },
        }
    ]
    source_inputs = [source_input]
    if additional_binding_repository is not None:
        source_inputs.append("../src/other.cpp")
        bindings.append(
            {
                "repository": additional_binding_repository,
                "head_sha": head,
                "source_path": "src/other.cpp",
                "confidence": {
                    "score": 0.95,
                    "level": "high",
                    "source": "relative_source_path_v1",
                },
            }
        )
    for index in range(binding_count_delta):
        bindings.append(
            {
                "repository": f"other/repo{index}",
                "head_sha": head,
                "source_path": bound_source_path,
            }
        )
    action = {
        "action_entity_id": "entity:compile",
        "action_shape_sha256": "4" * 64,
        "command_sha256": "5" * 64,
        "cwd": cwd,
        "source_inputs": source_inputs,
        "repository_source_bindings": bindings,
        "repository_source_binding_count": len(bindings),
    }
    return {
        "schema": "cppmega_ci_chunk_occurrence_v3",
        "repository": repository,
        "source_repository": source_repository,
        "workflow": {
            "event": event,
            "head_sha": head,
        },
        "run_metadata_evidence": {"exact_attempt_match": True},
        "job": {"labels": ["ubuntu-24.04", "x64"]},
        "chunk": {
            "training_sidecars": {
                "schema": "cppmega_ci_chunk_training_sidecars_v2",
                "build_actions": [action],
            }
        },
    }


def _producer_checkout_provenance(
    *,
    head: str,
    event: str,
    repository: str,
    source_repository: str,
) -> tuple[dict[str, object], dict[str, object]]:
    raw = (
        "2026-07-26T04:35:00Z "
        "Working directory is '/home/runner/work/workspace/checkout/build'\n"
        "2026-07-26T04:35:01Z "
        "[command]g++ -c ../src/nested/main.cpp -o main.o\n"
    ).encode()
    produced = canonicalize_ci_log(
        raw,
        {
            "repository": {"full_name": repository},
            "source_repository": source_repository,
            "event_name": event,
            "head_sha": head,
            "job": {"labels": ["ubuntu-24.04", "x64"]},
        },
    )
    actions = [
        action
        for chunk in produced["chunks"]
        for action in chunk["training_sidecars"]["build_actions"]
    ]
    assert len(actions) == 1
    producer_provenance = produced["sidecar"]["provenance"]
    return (
        {
            "repository": producer_provenance["repository"],
            "source_repository": producer_provenance["source_repository"],
            "workflow": {
                "event": producer_provenance["run"]["event"],
                "head_sha": producer_provenance["run"]["head_sha"],
            },
            "job": {"labels": producer_provenance["runner"]["labels"]},
        },
        actions[0],
    )


def _frozen_case5_fixture(
    tmp_path: Path,
    head: str,
) -> dict[str, object]:
    root = tmp_path / "ci-store"
    content = "compile output\n"
    content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    token_sequence_sha = hash_token_sequence([1, 2])
    selected_key = {
        "repo": "owner/repo",
        "run_attempt": "1:1",
        "job": "a-linux",
        "step": "compile:0",
        "chunk_ordinal": 0,
    }
    nonrepresentative_key = {
        "repo": "owner/repo",
        "run_attempt": "1:1",
        "job": "z-linux",
        "step": "compile:0",
        "chunk_ordinal": 0,
    }
    selected_provenance = _valid_provenance(head)
    # This deliberately stale non-representative proves that only the selected
    # occurrence is interpreted as source provenance after receipt verification.
    nonrepresentative_provenance = {
        "schema": "stale_nonrepresentative_schema",
        "note": "must not be source-scanned",
    }
    with CIContentStore(root, max_pack_bytes=1024) as store:
        store.add_chunk(
            content,
            selected_provenance,
            selected_key,
            token_count=2,
            tokenizer_fingerprint="tokenizer-test-v1",
            token_sequence_sha256=token_sequence_sha,
        )
        store.add_chunk(
            content,
            nonrepresentative_provenance,
            nonrepresentative_key,
            token_count=2,
            tokenizer_fingerprint="tokenizer-test-v1",
            token_sequence_sha256=token_sequence_sha,
        )
        content_receipt = store.completion_receipt(target_unique_tokens=2)

    content_receipt_path = tmp_path / "content-receipt.json"
    content_receipt_raw = _write_json(content_receipt_path, content_receipt)
    frozen_fetch_state = _frozen_fetch_state_binding(tmp_path)
    fetch_receipt = {
        "schema": FETCH_RECEIPT_SCHEMA,
        "frozen_fetch_state": frozen_fetch_state,
        "content_store_receipt": content_receipt,
    }
    fetch_path = tmp_path / "fetch-receipt.json"
    _write_json(fetch_path, fetch_receipt)

    representative = {
        "schema": REPRESENTATIVE_LEDGER_SCHEMA,
        "token_sequence_sha256": token_sequence_sha,
        "token_count": 2,
        "candidate_content_count": 1,
        "candidate_occurrence_count": 2,
        "candidate_content_sha256_sequence_sha256": _hash_records(
            "cppmega-ci-candidate-content-sha256-sequence-v1",
            [content_sha],
        ),
        "representative_content_sha256": content_sha,
        "representative_occurrence_key": selected_key,
        "representative_provenance_sha256": hashlib.sha256(
            _canonical(selected_provenance)
        ).hexdigest(),
    }
    ledger_path = tmp_path / "representative_ledger.jsonl"
    ledger_raw = _canonical(representative) + b"\n"
    ledger_path.write_bytes(ledger_raw)
    ledger_logical = _hash_records(
        "cppmega-ci-case5-representative-ledger-v1",
        [representative],
    )
    ledger_artifact = hashlib.sha256(ledger_raw).hexdigest()
    export_receipt = {
        "schema": CASE5_EXPORT_SCHEMA,
        "status": "complete",
        "input_fetch_state": json.loads(_canonical(frozen_fetch_state)),
        "input_store": {
            "receipt_sha256": hashlib.sha256(content_receipt_raw).hexdigest(),
            "sqlite_schema_sha256": content_receipt["sqlite_schema_sha256"],
            "sqlite_logical_sha256": content_receipt["sqlite_logical_sha256"],
            "logical_content_set_sha256": content_receipt[
                "logical_content_set_sha256"
            ],
            "logical_token_sequence_set_sha256": content_receipt[
                "logical_token_sequence_set_sha256"
            ],
            "occurrence_set_sha256": content_receipt["occurrence_set_sha256"],
            "pack_hashes": content_receipt["pack_hashes"],
        },
        "representatives": {
            "schema": REPRESENTATIVE_LEDGER_SCHEMA,
            "selection": (
                "one-per-eligible-token-sequence; "
                "content-sha256-then-eligible-occurrence-key"
            ),
            "count": 1,
            "ledger_artifact": ledger_path.name,
            "ledger_sha256": ledger_logical,
            "ledger_artifact_sha256": ledger_artifact,
        },
        "artifacts": [
            {
                "path": ledger_path.name,
                "kind": "representative_ledger",
                "rows": 1,
                "byte_size": len(ledger_raw),
                "sha256": ledger_artifact,
            }
        ],
    }
    export_path = tmp_path / "case5-export-receipt.json"
    _write_json(export_path, export_receipt)
    return {
        "root": root,
        "fetch_path": fetch_path,
        "content_receipt_path": content_receipt_path,
        "export_path": export_path,
        "ledger_path": ledger_path,
        "selected_key": selected_key,
        "nonrepresentative_key": nonrepresentative_key,
        "representative": representative,
        "content_receipt": content_receipt,
        "export_receipt": export_receipt,
        "frozen_fetch_state": frozen_fetch_state,
    }


def _sync_case5_receipts(fixture: dict[str, object]) -> None:
    content_receipt = fixture["content_receipt"]
    export_receipt = fixture["export_receipt"]
    assert isinstance(content_receipt, dict)
    assert isinstance(export_receipt, dict)
    content_raw = _write_json(
        fixture["content_receipt_path"],  # type: ignore[arg-type]
        content_receipt,
    )
    _write_json(
        fixture["fetch_path"],  # type: ignore[arg-type]
        {
            "schema": FETCH_RECEIPT_SCHEMA,
            "frozen_fetch_state": fixture["frozen_fetch_state"],
            "content_store_receipt": content_receipt,
        },
    )
    input_store = export_receipt["input_store"]
    assert isinstance(input_store, dict)
    input_store.update(
        {
            "receipt_sha256": hashlib.sha256(content_raw).hexdigest(),
            "sqlite_schema_sha256": content_receipt["sqlite_schema_sha256"],
            "sqlite_logical_sha256": content_receipt["sqlite_logical_sha256"],
            "logical_content_set_sha256": content_receipt[
                "logical_content_set_sha256"
            ],
            "logical_token_sequence_set_sha256": content_receipt[
                "logical_token_sequence_set_sha256"
            ],
            "occurrence_set_sha256": content_receipt["occurrence_set_sha256"],
            "pack_hashes": content_receipt["pack_hashes"],
        }
    )
    _write_json(fixture["export_path"], export_receipt)  # type: ignore[arg-type]


def _sync_representative_ledger(
    fixture: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    ledger_path = fixture["ledger_path"]
    export_receipt = fixture["export_receipt"]
    assert isinstance(ledger_path, Path)
    assert isinstance(export_receipt, dict)
    raw = b"".join(_canonical(record) + b"\n" for record in records)
    ledger_path.write_bytes(raw)
    artifact_sha = hashlib.sha256(raw).hexdigest()
    logical_sha = _hash_records(
        "cppmega-ci-case5-representative-ledger-v1",
        records,
    )
    representatives = export_receipt["representatives"]
    artifact = export_receipt["artifacts"][0]
    assert isinstance(representatives, dict)
    assert isinstance(artifact, dict)
    representatives.update(
        {
            "count": len(records),
            "ledger_sha256": logical_sha,
            "ledger_artifact_sha256": artifact_sha,
        }
    )
    artifact.update(
        {
            "rows": len(records),
            "byte_size": len(raw),
            "sha256": artifact_sha,
        }
    )
    _write_json(fixture["export_path"], export_receipt)  # type: ignore[arg-type]


def _extract_fixture(
    fixture: dict[str, object],
    output: Path,
) -> dict[str, object]:
    return extract_binding_inventory(
        fixture["root"],  # type: ignore[arg-type]
        fixture["fetch_path"],  # type: ignore[arg-type]
        content_store_receipt_path=fixture["content_receipt_path"],  # type: ignore[arg-type]
        case5_export_receipt_path=fixture["export_path"],  # type: ignore[arg-type]
        representative_ledger_path=fixture["ledger_path"],  # type: ignore[arg-type]
        output_path=output,
    )


def _inventory_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_representative_only_inventory_build_and_receipt_hash_chain(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    fixture = _frozen_case5_fixture(tmp_path, head)
    inventory_path = tmp_path / "source-inventory.jsonl"

    extraction = _extract_fixture(fixture, inventory_path)
    verified = verify_binding_inventory(inventory_path)
    records = _inventory_records(inventory_path)
    header = records[0]
    binding = next(record for record in records if record["record_type"] == "binding")
    reference = next(
        record for record in records if record["record_type"] == "reference"
    )

    assert extraction["status"] == "complete"
    assert extraction["representative_count"] == 1
    assert verified.artifact_sha256 == hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest()
    assert header["representative_ledger_sha256"] == fixture[
        "export_receipt"
    ]["representatives"]["ledger_sha256"]  # type: ignore[index]
    assert header["representative_ledger_artifact_sha256"] == hashlib.sha256(
        fixture["ledger_path"].read_bytes()  # type: ignore[union-attr]
    ).hexdigest()
    assert header["frozen_fetch_state"] == fixture["frozen_fetch_state"]
    assert header["frozen_fetch_state_sha256"] == hashlib.sha256(
        _canonical(fixture["frozen_fetch_state"])
    ).hexdigest()
    assert binding["source_path"] == "src/nested/main.cpp"
    assert binding["normalization_status"] == RESOLVED
    assert reference["representative_occurrence_key"] == fixture["selected_key"]
    assert reference["representative_occurrence_key"] != fixture[
        "nonrepresentative_key"
    ]

    receipt = materialize_inventory(
        inventory_path,
        {"owner/repo": mirror},
        tmp_path / "source-store",
        max_pack_bytes=1024,
    )
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["status"] == "complete"
    assert receipt["input_inventory_artifact_sha256"] == verified.artifact_sha256
    assert receipt["representative_ledger_sha256"] == header[
        "representative_ledger_sha256"
    ]
    assert receipt["representative_ledger_artifact_sha256"] == header[
        "representative_ledger_artifact_sha256"
    ]
    assert receipt["frozen_fetch_state_sha256"] == header[
        "frozen_fetch_state_sha256"
    ]
    assert receipt["build_action_reference_count"] == 1
    assert receipt["missing_reference_count"] == 0

    ledger_path = tmp_path / "source-reference-ledger.jsonl"
    with SourceSidecarStore(tmp_path / "source-store") as store:
        summary = store.reference_ledger()
        written = store.write_reference_ledger(ledger_path)
    assert "entries" not in summary
    assert written["reference_count"] == 1
    ledger_records = _inventory_records(ledger_path)
    assert ledger_records[0]["record_type"] == "header"
    assert ledger_records[1]["representative_reference"][
        "token_sequence_sha256"
    ] == fixture["representative"]["token_sequence_sha256"]  # type: ignore[index]
    assert "body" not in ledger_path.read_text(encoding="utf-8")
    assert "content_bytes" not in ledger_path.read_text(encoding="utf-8")


def test_representative_ledger_rejects_nonmember_duplicate_and_count_drift(
    tmp_path: Path,
) -> None:
    _mirror, head, _base = _git_fixture(tmp_path)
    fixture = _frozen_case5_fixture(tmp_path, head)
    representative = dict(fixture["representative"])  # type: ignore[arg-type]
    wrong_key = dict(representative["representative_occurrence_key"])
    wrong_key["job"] = "missing-job"
    representative["representative_occurrence_key"] = wrong_key
    _sync_representative_ledger(fixture, [representative])
    with pytest.raises(ExtractionError, match="exact member"):
        _extract_fixture(fixture, tmp_path / "wrong-member.jsonl")

    fixture = _frozen_case5_fixture(tmp_path / "duplicate", head)
    representative = dict(fixture["representative"])  # type: ignore[arg-type]
    _sync_representative_ledger(fixture, [representative, representative])
    with pytest.raises(ExtractionError, match="sorted and unique"):
        _extract_fixture(fixture, tmp_path / "duplicate.jsonl")

    fixture = _frozen_case5_fixture(tmp_path / "count", head)
    export_receipt = fixture["export_receipt"]
    assert isinstance(export_receipt, dict)
    export_receipt["representatives"]["count"] = 2  # type: ignore[index]
    _write_json(fixture["export_path"], export_receipt)  # type: ignore[arg-type]
    with pytest.raises(ExtractionError, match="artifact metadata differs"):
        _extract_fixture(fixture, tmp_path / "count-drift.jsonl")

    fixture = _frozen_case5_fixture(tmp_path / "selection", head)
    export_receipt = fixture["export_receipt"]
    assert isinstance(export_receipt, dict)
    export_receipt["representatives"]["selection"] = (  # type: ignore[index]
        "one-per-token-sequence; content-sha256-then-occurrence-key"
    )
    _write_json(fixture["export_path"], export_receipt)  # type: ignore[arg-type]
    with pytest.raises(ExtractionError, match="missing or stale"):
        _extract_fixture(fixture, tmp_path / "selection-drift.jsonl")

    fixture = _frozen_case5_fixture(tmp_path / "zero-eligible", head)
    _sync_representative_ledger(fixture, [])
    empty_result = _extract_fixture(
        fixture,
        tmp_path / "zero-eligible-inventory.jsonl",
    )
    assert empty_result["representative_count"] == 0
    assert empty_result["binding_count"] == 0
    assert empty_result["reference_count"] == 0


def test_inventory_requires_exact_frozen_fetch_state_lineage(
    tmp_path: Path,
) -> None:
    _mirror, head, _base = _git_fixture(tmp_path)
    missing = _frozen_case5_fixture(tmp_path / "missing", head)
    fetch_receipt = json.loads(
        missing["fetch_path"].read_text(encoding="utf-8")  # type: ignore[union-attr]
    )
    del fetch_receipt["frozen_fetch_state"]
    _write_json(missing["fetch_path"], fetch_receipt)  # type: ignore[arg-type]
    with pytest.raises(ExtractionError, match="lineage binding is missing"):
        _extract_fixture(missing, tmp_path / "missing-inventory.jsonl")

    mismatch = _frozen_case5_fixture(tmp_path / "mismatch", head)
    export_receipt = json.loads(
        mismatch["export_path"].read_text(encoding="utf-8")  # type: ignore[union-attr]
    )
    export_receipt["input_fetch_state"]["summary"]["requests"] += 1
    _write_json(mismatch["export_path"], export_receipt)  # type: ignore[arg-type]
    with pytest.raises(ExtractionError, match="bindings differ"):
        _extract_fixture(mismatch, tmp_path / "mismatch-inventory.jsonl")


def test_path_normalization_is_platform_aware_and_has_no_basename_heuristic() -> None:
    assert (
        normalize_source_path(
            "../src/main.cpp",
            "/home/runner/work/workspace/checkout/build",
            platform="posix",
        )
        == "src/main.cpp"
    )
    assert (
        normalize_source_path(
            r"..\src\main.cpp",
            r"D:\a\workspace\checkout\build",
            platform="windows",
        )
        == "src/main.cpp"
    )
    assert (
        normalize_source_path(
            r"d:\A\WORKSPACE\CHECKOUT\src\main.cpp",
            r"D:\a\workspace\checkout\build",
            platform="windows",
        )
        == "src/main.cpp"
    )
    posix_backslash = normalize_source_candidates(
        r"src\main.cpp",
        "/home/runner/work/workspace/checkout",
        platform="posix",
    )
    assert posix_backslash.status == RESOLVED
    assert posix_backslash.candidates == (r"src\main.cpp",)

    basename_trick = normalize_source_candidates(
        "../src/main.cpp",
        "/opt/cache/repo/repo/build",
        repository="owner/repo",
        platform="posix",
    )
    assert basename_trick.status == CHECKOUT_PROVENANCE_UNRESOLVABLE
    assert basename_trick.candidates == ()

    drive_relative = normalize_source_candidates(
        r"C:src\main.cpp",
        r"D:\a\workspace\checkout",
        platform="windows",
    )
    assert drive_relative.status == GENERATED_OR_MUTATED_UNRESOLVABLE
    assert drive_relative.reason == "windows_drive_relative_path"


def test_real_log_binding_normalizes_relative_path_against_action_cwd() -> None:
    head = "a" * 40
    provenance, action = _producer_checkout_provenance(
        head=head,
        event="push",
        repository="owner/repo",
        source_repository="owner/repo",
    )

    repository, checkout_head, normalization, evidence = _checkout_binding(
        provenance,
        action,
        source_index=0,
        source_input=action["source_inputs"][0],
        cwd=action["cwd"],
    )

    assert action["repository_source_bindings"][0]["source_path"] == (
        "src/nested/main.cpp"
    )
    assert repository == "owner/repo"
    assert checkout_head == head
    assert normalization.status == RESOLVED
    assert normalization.candidates == ("src/nested/main.cpp",)
    assert evidence["reason"] is None


def test_real_log_fork_pr_binding_uses_canonical_merge_checkout_tuple() -> None:
    head = "b" * 40
    provenance, action = _producer_checkout_provenance(
        head=head,
        event="pull_request",
        repository="owner/base",
        source_repository="fork/head",
    )

    repository, checkout_head, normalization, evidence = _checkout_binding(
        provenance,
        action,
        source_index=0,
        source_input=action["source_inputs"][0],
        cwd=action["cwd"],
    )

    assert action["repository_source_bindings"][0]["repository"] == "owner/base"
    assert repository == "owner/base"
    assert checkout_head == head
    assert normalization.status == RESOLVED
    assert evidence["checkout_kind"] == "pull_request_merge"
    assert evidence["canonical_repository"] == "owner/base"
    assert evidence["head_repository"] == "fork/head"
    assert evidence["reason"] is None


def test_real_log_unproven_fork_checkout_is_a_typed_gap() -> None:
    head = "c" * 40
    provenance, action = _producer_checkout_provenance(
        head=head,
        event="workflow_dispatch",
        repository="owner/base",
        source_repository="fork/head",
    )

    _repository, _head, normalization, evidence = _checkout_binding(
        provenance,
        action,
        source_index=0,
        source_input=action["source_inputs"][0],
        cwd=action["cwd"],
    )

    assert normalization.status == CHECKOUT_PROVENANCE_UNRESOLVABLE
    assert evidence["checkout_kind"] == "unproven_head_or_fork_checkout"
    assert evidence["reason"] == "workflow_event_cannot_prove_checkout_tuple"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "event": "pull_request",
                "repository": "owner/base",
                "source_repository": "fork/head",
                "binding_repository": "owner/base",
            },
            RESOLVED,
        ),
        (
            {
                "event": "pull_request",
                "repository": "owner/base",
                "source_repository": "fork/head",
                "binding_repository": "fork/head",
            },
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
        ),
        (
            {"binding_count_delta": 1},
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
        ),
        (
            {"additional_binding_repository": "other/checkout"},
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
        ),
        (
            {"binding_method": "workspace_repo_basename_suffix_v1"},
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
        ),
        (
            {"cwd": "/home/runner/work/workspace/checkout/custom/build"},
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
        ),
        (
            {"cwd": None},
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
        ),
        (
            {
                "source_input": r"src\nested\main.cpp",
                "bound_source_path": "src/nested/main.cpp",
            },
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
        ),
    ],
)
def test_checkout_tuple_and_path_fail_closed(
    kwargs: dict[str, object],
    expected: str,
) -> None:
    head = "a" * 40
    provenance = _valid_provenance(head, **kwargs)
    action = provenance["chunk"]["training_sidecars"]["build_actions"][0]  # type: ignore[index]
    source_input = action["source_inputs"][0]
    _repository, _head, normalization, evidence = _checkout_binding(
        provenance,
        action,
        source_index=0,
        source_input=source_input,
        cwd=action["cwd"],
    )
    assert normalization.status == expected
    if expected != RESOLVED:
        assert evidence["reason"]
        resolution = LocalGitResolver({}).resolve(
            {
                "schema": INVENTORY_SCHEMA,
                "record_type": "binding",
                "repository": _repository,
                "head_sha": _head,
                "source_path": f"!unresolved/{'f' * 64}",
                "normalization_status": normalization.status,
                "normalized_candidates": [],
            }
        )
        assert resolution.status == expected


@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
def test_exact_git_resolution_supports_sha1_and_sha256(
    tmp_path: Path,
    object_format: str,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path, object_format=object_format)
    binding = _binding(head, "src/nested/main.cpp")
    result = LocalGitResolver({"owner/repo": mirror}).resolve(binding)
    payload = b"int main() { return 0; }\n"
    constructor = hashlib.sha1 if object_format == "sha1" else hashlib.sha256
    expected_oid = constructor(
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()

    assert result.status == RESOLVED
    assert result.object_format == object_format
    assert result.commit_oid == head
    assert result.blob_oid == expected_oid
    assert result.content == payload
    assert result.evidence["runner_filesystem_equivalence_claimed"] is False


def test_binary_symlink_submodule_and_bounded_git_object_are_fail_closed(
    tmp_path: Path,
) -> None:
    mirror, head, base = _git_fixture(tmp_path)
    resolver = LocalGitResolver({"owner/repo": mirror})
    binary = resolver.resolve(_binding(head, "assets/bytes.bin"))
    symlink = resolver.resolve(_binding(head, "main-link"))
    submodule = resolver.resolve(_binding(head, "vendor/component"))
    nested_submodule = resolver.resolve(
        _binding(head, "vendor/component/source.cpp")
    )
    oversized = LocalGitResolver(
        {"owner/repo": mirror},
        max_git_object_bytes=64,
    ).resolve(_binding(head, "src/large.cpp"))

    assert binary.status == RESOLVED
    assert binary.content_kind == "binary"
    assert symlink.status == RESOLVED
    assert symlink.object_type == "symlink"
    assert symlink.content == b"src/nested/main.cpp"
    assert submodule.status == UNSUPPORTED_OBJECT
    assert submodule.object_oid == base
    assert nested_submodule.status == UNSUPPORTED_OBJECT
    assert oversized.status == UNSUPPORTED_OBJECT
    assert oversized.evidence["reason"] == "git_object_exceeds_bounded_read_policy"


def test_store_deduplicates_batches_and_rejects_arbitrary_partial_membership(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    first = _binding(head, "src/nested/main.cpp")
    second = _binding(head, "src/copy.cpp")
    inventory = _write_inventory(tmp_path / "inventory.jsonl", [first, second])
    resolver = LocalGitResolver({"owner/repo": mirror})
    resolutions = [resolver.resolve(first), resolver.resolve(second)]
    assert resolutions[0].content_sha256 == resolutions[1].content_sha256

    root = tmp_path / "store"
    with _new_source_store(root, inventory) as store:
        assert store.add_resolutions(iter(resolutions), batch_size=2) == 2
        verification = store.verify()
        assert verification["binding_count"] == 2
        assert verification["blob_count"] == 1
        assert verification["reference_count"] == 2
        receipt = store.receipt()
        assert receipt["status"] == "complete"

    wrong = _binding(head, "assets/bytes.bin")
    wrong_resolution = resolver.resolve(wrong)
    one_inventory = _write_inventory(
        tmp_path / "one-inventory.jsonl",
        [first],
    )
    with _new_source_store(
        tmp_path / "one-store", one_inventory
    ) as store, pytest.raises(SourceStoreError, match="not a member"):
        store.add_resolution(wrong_resolution)


def test_store_reference_closure_is_required_for_complete_status(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    binding = _binding(head, "src/nested/main.cpp")
    inventory = _write_inventory(tmp_path / "inventory.jsonl", [binding])
    root = tmp_path / "store"

    with _new_source_store(root, inventory) as store:
        store.add_resolution(
            LocalGitResolver({"owner/repo": mirror}).resolve(binding)
        )
        assert store.receipt()["status"] == "complete"
        store._connection.execute("DELETE FROM inventory_references")
        store._connection.commit()

        verification = store.verify()
        receipt = store.receipt()
        assert verification["binding_count"] == 1
        assert verification["missing_binding_count"] == 0
        assert verification["reference_count"] == 0
        assert verification["missing_reference_count"] == 1
        assert verification["status"] == "incomplete"
        assert receipt["status"] == "incomplete"
        assert receipt["missing_reference_count"] == 1

    with sqlite3.connect(root / "index.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM inventory_references"
        ).fetchone()[0] == 0


def test_frame_size_is_validated_before_hostile_uint64_read(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    binding = _binding(head, "src/nested/main.cpp")
    inventory = _write_inventory(tmp_path / "inventory.jsonl", [binding])
    resolution = LocalGitResolver({"owner/repo": mirror}).resolve(binding)
    root = tmp_path / "store"
    with _new_source_store(root, inventory) as store:
        store.add_resolution(resolution)
        row = store._connection.execute(
            "SELECT content_sha256, pack_id, offset FROM blobs"
        ).fetchone()
        assert row is not None
        pack = root / store._connection.execute(
            "SELECT filename FROM packs WHERE pack_id = ?",
            (int(row["pack_id"]),),
        ).fetchone()[0]
        offset = int(row["offset"])
        digest = str(row["content_sha256"])
    with pack.open("r+b") as handle:
        handle.seek(offset)
        header = handle.read(_FRAME_HEADER.size)
        magic, raw_digest, _size = _FRAME_HEADER.unpack(header)
        handle.seek(offset)
        handle.write(_FRAME_HEADER.pack(magic, raw_digest, 2**64 - 1))
        handle.flush()
        os.fsync(handle.fileno())

    with SourceSidecarStore(root) as reopened, pytest.raises(
        SourceStoreError, match="header is invalid"
    ):
        reopened.read_blob(digest)


def test_pack_policy_rejects_first_blob_larger_than_pack(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    binding = _binding(head, "src/nested/main.cpp")
    inventory = _write_inventory(tmp_path / "inventory.jsonl", [binding])
    resolution = LocalGitResolver({"owner/repo": mirror}).resolve(binding)
    with _new_source_store(
        tmp_path / "store",
        inventory,
        max_pack_bytes=len(b"CISSPK1\n") + _FRAME_HEADER.size,
    ) as store, pytest.raises(SourceStoreError, match="pack size policy"):
        store.add_resolution(resolution)


@pytest.mark.parametrize("setting", ["creator_script_sha256", "resolver_sha256"])
def test_resume_requires_current_script_and_resolver(
    tmp_path: Path,
    setting: str,
) -> None:
    _mirror, head, _base = _git_fixture(tmp_path)
    binding = _binding(head, "src/nested/main.cpp")
    inventory = _write_inventory(tmp_path / "inventory.jsonl", [binding])
    root = tmp_path / "store"
    with _new_source_store(root, inventory):
        pass
    with sqlite3.connect(root / "index.sqlite3") as connection:
        connection.execute(
            "UPDATE settings SET value = ? WHERE key = ?",
            ("f" * 64, setting),
        )
        connection.commit()
    with pytest.raises(SourceStoreError, match="script differs|resolver differs"):
        SourceSidecarStore(root)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("sqlite_schema_sha256", "schema SHA-256 differs"),
        ("sqlite_logical_sha256", "logical SHA-256 differs"),
    ],
)
def test_full_content_store_receipt_binds_sqlite_hashes(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    _mirror, head, _base = _git_fixture(tmp_path)
    fixture = _frozen_case5_fixture(tmp_path, head)
    receipt = fixture["content_receipt"]
    assert isinstance(receipt, dict)
    receipt[field] = "f" * 64
    _sync_case5_receipts(fixture)
    with pytest.raises(ExtractionError, match=message):
        _extract_fixture(fixture, tmp_path / "inventory.jsonl")


def test_frozen_receipts_policy_and_recovery_artifacts_are_exact(
    tmp_path: Path,
) -> None:
    _mirror, head, _base = _git_fixture(tmp_path)
    fixture = _frozen_case5_fixture(tmp_path / "symlink", head)
    receipt_path = fixture["content_receipt_path"]
    assert isinstance(receipt_path, Path)
    target = receipt_path.with_name("actual-content-receipt.json")
    receipt_path.rename(target)
    receipt_path.symlink_to(target.name)
    with pytest.raises(ExtractionError, match="non-symlink"):
        _extract_fixture(fixture, tmp_path / "symlink-inventory.jsonl")

    fixture = _frozen_case5_fixture(tmp_path / "policy", head)
    receipt = fixture["content_receipt"]
    assert isinstance(receipt, dict)
    policy = dict(receipt["policy"])
    policy["compression"] = {"algorithm": "tampered"}
    receipt["policy"] = policy
    receipt["policy_sha256"] = hashlib.sha256(_canonical(policy)).hexdigest()
    _sync_case5_receipts(fixture)
    with pytest.raises(ExtractionError, match="policy digest differs"):
        _extract_fixture(fixture, tmp_path / "policy-inventory.jsonl")

    fixture = _frozen_case5_fixture(tmp_path / "recovery", head)
    root = fixture["root"]
    assert isinstance(root, Path)
    quarantine = root / "orphaned"
    quarantine.mkdir()
    (quarantine / "unmanifested.bin").write_bytes(b"orphan")
    with pytest.raises(ExtractionError, match="unmanifested artifacts"):
        _extract_fixture(fixture, tmp_path / "recovery-inventory.jsonl")

    fixture = _frozen_case5_fixture(tmp_path / "pack-frame", head)
    root = fixture["root"]
    receipt = fixture["content_receipt"]
    assert isinstance(root, Path)
    assert isinstance(receipt, dict)
    pack_record = receipt["pack_hashes"][0]
    pack_path = root / pack_record["filename"]
    pack_bytes = bytearray(pack_path.read_bytes())
    pack_bytes[-1] ^= 1
    pack_path.write_bytes(pack_bytes)
    pack_record["sha256"] = hashlib.sha256(pack_bytes).hexdigest()
    _sync_case5_receipts(fixture)
    with pytest.raises(ExtractionError, match="frame encoding|frame verification"):
        _extract_fixture(fixture, tmp_path / "pack-frame-inventory.jsonl")

    fixture = _frozen_case5_fixture(tmp_path / "counter", head)
    root = fixture["root"]
    receipt = fixture["content_receipt"]
    assert isinstance(root, Path)
    assert isinstance(receipt, dict)
    with sqlite3.connect(root / "index.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            "UPDATE stats SET occurrence_count = occurrence_count + 1"
        )
        connection.commit()
        receipt["sqlite_logical_sha256"] = (
            _content_store_sqlite_logical_sha256(connection)
        )
    receipt["counters"]["occurrence_count"] += 1
    _sync_case5_receipts(fixture)
    with pytest.raises(ExtractionError, match="counter mismatch"):
        _extract_fixture(fixture, tmp_path / "counter-inventory.jsonl")


def test_selected_provenance_rejects_zlib_trailing_garbage(
    tmp_path: Path,
) -> None:
    _mirror, head, _base = _git_fixture(tmp_path)
    fixture = _frozen_case5_fixture(tmp_path, head)
    root = fixture["root"]
    selected = fixture["selected_key"]
    assert isinstance(root, Path)
    assert isinstance(selected, dict)
    with sqlite3.connect(root / "index.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT provenance_zlib FROM occurrences
            WHERE repo = ? AND run_attempt = ? AND job = ?
              AND step = ? AND chunk_ordinal = ?
            """,
            (
                selected["repo"],
                selected["run_attempt"],
                selected["job"],
                selected["step"],
                selected["chunk_ordinal"],
            ),
        ).fetchone()
        assert row is not None
        connection.execute(
            """
            UPDATE occurrences SET provenance_zlib = ?
            WHERE repo = ? AND run_attempt = ? AND job = ?
              AND step = ? AND chunk_ordinal = ?
            """,
            (
                bytes(row["provenance_zlib"]) + b"trailing-garbage",
                selected["repo"],
                selected["run_attempt"],
                selected["job"],
                selected["step"],
                selected["chunk_ordinal"],
            ),
        )
        connection.commit()
        logical = _content_store_sqlite_logical_sha256(connection)
    receipt = fixture["content_receipt"]
    assert isinstance(receipt, dict)
    receipt["sqlite_logical_sha256"] = logical
    _sync_case5_receipts(fixture)

    with pytest.raises(ExtractionError, match="non-canonical zlib"):
        _extract_fixture(fixture, tmp_path / "inventory.jsonl")


def test_inventory_stream_verifier_rejects_orphan_reference(
    tmp_path: Path,
) -> None:
    _mirror, head, _base = _git_fixture(tmp_path)
    binding = _binding(head, "src/nested/main.cpp")
    inventory = _write_inventory(tmp_path / "inventory.jsonl", [binding])
    records = _inventory_records(inventory)
    reference = dict(records[-1])
    reference["source_path"] = "src/not-a-binding.cpp"
    references = [reference]
    header = records[0]
    header["reference_records_sha256"] = _hash_records(
        "cppmega-ci-source-reference-records-v2",
        references,
    )
    header["inventory_logical_sha256"] = _inventory_logical_sha256(header)
    with inventory.open("wb") as handle:
        for record in [header, records[1], *references]:
            handle.write(_canonical(record) + b"\n")
    with pytest.raises(ExtractionError, match="not a member"):
        verify_binding_inventory(inventory)


def test_cli_incomplete_is_nonzero_and_existing_receipts_need_authorization(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    missing = _binding(head, "src/missing.cpp")
    inventory = _write_inventory(tmp_path / "inventory.jsonl", [missing])
    mirrors = tmp_path / "mirrors.json"
    _write_json(mirrors, {"owner/repo": str(mirror)})
    receipt = tmp_path / "receipt.json"
    store = tmp_path / "store"
    args = [
        "build",
        "--inventory",
        str(inventory),
        "--mirrors",
        str(mirrors),
        "--store",
        str(store),
        "--receipt",
        str(receipt),
        "--max-pack-bytes",
        "1024",
    ]
    assert main(args) == 3
    capsys.readouterr()
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["status"] == "incomplete"
    assert value["gap_status_counts"] == {PATH_ABSENT: 1}

    assert main(args) == 2
    assert "refusing to overwrite existing artifact" in capsys.readouterr().err
    existing_sha = _sha256_file(receipt)
    assert main([*args, "--expected-receipt-sha256", existing_sha]) == 3
    capsys.readouterr()


def test_cli_verify_returns_incomplete_for_missing_bindings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mirror, head, _base = _git_fixture(tmp_path)
    inventory = _write_inventory(
        tmp_path / "inventory.jsonl",
        [_binding(head, "src/nested/main.cpp")],
    )
    store = tmp_path / "store"
    with _new_source_store(store, inventory):
        pass

    assert main(["verify", "--store", str(store)]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "incomplete"
    assert result["missing_binding_count"] == 1


def test_complete_cli_receipt_is_also_protected_from_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    binding = _binding(head, "src/nested/main.cpp")
    inventory = _write_inventory(tmp_path / "inventory.jsonl", [binding])
    mirrors = tmp_path / "mirrors.json"
    _write_json(mirrors, {"owner/repo": str(mirror)})
    receipt = tmp_path / "receipt.json"
    args = [
        "build",
        "--inventory",
        str(inventory),
        "--mirrors",
        str(mirrors),
        "--store",
        str(tmp_path / "store"),
        "--receipt",
        str(receipt),
    ]
    assert main(args) == 0
    capsys.readouterr()
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "complete"
    assert main(args) == 2
    assert "refusing to overwrite existing artifact" in capsys.readouterr().err
