from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from scripts.distributed_data_prep._common import ContractError
from scripts.distributed_data_prep.seal_outputs import (
    BUCKET_AUDIT_SCHEMA,
    DATA_KINDS,
    OUTPUT_MANIFEST_SCHEMA,
    TARGET_LENGTHS,
    ZERO_RECEIPT_SCHEMA,
    _artifact_contracts_sha256,
    artifact_set_sha256,
    output_manifest_sha256,
    seal_outputs,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_descriptor(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    raw = path.read_bytes()
    return {
        "path": relative,
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_payload(
    root: Path,
    relative: str,
    payload: bytes,
    *,
    role: str,
    format_name: str,
    compression: str,
) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        **_file_descriptor(root, relative),
        "role": role,
        "format": format_name,
        "compression": compression,
        "contract_sha256": _sha(f"contract:{role}"),
    }


def _mmididx_payload(*, token_count: int, document_count: int) -> bytes:
    assert token_count >= document_count > 0
    base, remainder = divmod(token_count, document_count)
    sizes = [base + (1 if index < remainder else 0) for index in range(document_count)]
    pointers: list[int] = []
    token_offset = 0
    for size in sizes:
        pointers.append(2 * token_offset)
        token_offset += size
    documents = list(range(document_count + 1))
    return b"".join(
        (
            b"MMIDIDX\x00\x00",
            struct.pack("<Q", 1),
            struct.pack("<B", 8),
            struct.pack("<Q", document_count),
            struct.pack("<Q", document_count + 1),
            struct.pack(f"<{document_count}i", *sizes),
            struct.pack(f"<{document_count}q", *pointers),
            struct.pack(f"<{document_count + 1}q", *documents),
        )
    )


@dataclass(frozen=True)
class Corpus:
    artifact_root: Path
    manifests: dict[str, Path]


def _build_corpus(
    tmp_path: Path,
    *,
    materialized: dict[str, set[int]] | None = None,
) -> Corpus:
    root = tmp_path / "artifacts"
    manifests_dir = tmp_path / "manifests"
    root.mkdir()
    if materialized is None:
        materialized = {
            "source": set(TARGET_LENGTHS),
            "github_pr": {1024},
            "gitlab_mr": {2048},
            "ci": {4096},
        }
    tokenizer = _sha("tokenizer")
    dataset_schema = _sha("dataset-schema")
    paths: dict[str, Path] = {}
    for kind in DATA_KINDS:
        bindings = {
            "source_receipt_sha256": _sha(f"source-receipt:{kind}"),
            "producer_sha256": _sha(f"producer:{kind}"),
            "tokenizer_sha256": tokenizer,
            "dataset_schema_sha256": dataset_schema,
        }
        buckets: list[dict[str, object]] = []
        for length in TARGET_LENGTHS:
            prefix = f"{kind}/{length}"
            if length not in materialized[kind]:
                relative = f"{prefix}/verified-zero.json"
                zero = {
                    "schema": ZERO_RECEIPT_SCHEMA,
                    "status": "verified_zero",
                    "kind": kind,
                    "sequence_length": length,
                    "reason": "eligibility query returned no rows",
                    "source_receipt_sha256": bindings["source_receipt_sha256"],
                    "producer_sha256": bindings["producer_sha256"],
                    "tokenizer_sha256": tokenizer,
                    "dataset_schema_sha256": dataset_schema,
                    "eligibility_query_sha256": _sha(f"query:{kind}:{length}"),
                    "document_count": 0,
                    "row_count": 0,
                    "valid_tokens": 0,
                    "trained_tokens": 0,
                }
                _write_json(root / relative, zero)
                buckets.append(
                    {
                        "sequence_length": length,
                        "status": "verified_zero",
                        "zero_receipt": _file_descriptor(root, relative),
                    }
                )
                continue

            parquet = _write_payload(
                root,
                f"{prefix}/data.parquet",
                (
                    b"PAR1"
                    + f"payload:{kind}:{length}".encode()
                    + b"\x00"
                    + struct.pack("<I", 1)
                    + b"PAR1"
                ),
                role="parquet",
                format_name="parquet",
                compression="zstd",
            )
            data = _write_payload(
                root,
                f"{prefix}/shard.bin",
                b"\x00\x00" * length,
                role="megatron_bin",
                format_name="megatron_mmididx_data",
                compression="none",
            )
            index = _write_payload(
                root,
                f"{prefix}/shard.idx",
                _mmididx_payload(token_count=length, document_count=1),
                role="megatron_idx",
                format_name="megatron_mmididx_index",
                compression="none",
            )
            sidecar_names = (
                "shard_loss_mask.bin",
                "shard_graph_offsets.bin",
                "shard_graph_data.bin",
                "shard_sequence_doc_offsets.bin",
                "shard_doc_platform_offsets.bin",
                "shard_platform_ids.bin",
                "shard_source_identity_registry.sqlite",
                "shard_objective_ids.bin",
            )
            sidecars = [
                _write_payload(
                    root,
                    f"{prefix}/{name}",
                    f"sidecar:{kind}:{length}:{name}".encode(),
                    role="megatron_sidecar",
                    format_name="megatron_sidecar",
                    compression="none",
                )
                for name in sidecar_names
            ]
            prefix_manifest = {
                "tokenizer_contract": "megacpp",
                "vocab_size": 65536,
                "dtype": "uint16",
                "token_count": length,
                "document_count": 1,
                "trained_token_count": length,
                "loss_mask_alignment": "source_token_predicts_next_v1",
                "graph_sidecar_schema": "cppmega_graph_routes_v2",
                "side_channel_paths": {
                    "loss_mask": {"path": "shard_loss_mask.bin", "dtype": "uint8"}
                },
                "graph_sidecar_paths": {
                    "token_chunks": {
                        "offsets_path": "shard_graph_offsets.bin",
                        "data_path": "shard_graph_data.bin",
                    }
                },
                "source_platform_sidecar": {
                    "schema": "cppmega_source_platform_v1",
                    "sequence_doc_offsets_path": "shard_sequence_doc_offsets.bin",
                    "doc_platform_offsets_path": "shard_doc_platform_offsets.bin",
                    "platform_ids_path": "shard_platform_ids.bin",
                },
                "source_identity_registry": {
                    "path": "shard_source_identity_registry.sqlite"
                },
                "objective_contract": {
                    "objective_id_sidecar": {"path": "shard_objective_ids.bin"}
                },
            }
            manifest_relative = f"{prefix}/shard.json"
            _write_json(root / manifest_relative, prefix_manifest)
            megatron_manifest = {
                **_file_descriptor(root, manifest_relative),
                "role": "megatron_manifest",
                "format": "megatron_prefix_manifest",
                "compression": "none",
                "contract_sha256": _sha("contract:megatron_manifest"),
            }
            artifacts = [parquet, data, index, megatron_manifest, *sidecars]
            artifacts.sort(key=lambda item: str(item["path"]))
            assert [item["path"] for item in artifacts] == sorted(
                str(item["path"]) for item in artifacts
            )
            counts = {
                "document_count": 1,
                "row_count": 1,
                "valid_tokens": length,
                "trained_tokens": length,
                "payload_artifact_count": len(artifacts),
            }
            audit = {
                "schema": BUCKET_AUDIT_SCHEMA,
                "status": "verified",
                "kind": kind,
                "sequence_length": length,
                "source_receipt_sha256": bindings["source_receipt_sha256"],
                "producer_sha256": bindings["producer_sha256"],
                "tokenizer_sha256": tokenizer,
                "dataset_schema_sha256": dataset_schema,
                "payload_artifact_set_sha256": artifact_set_sha256(artifacts),
                "artifact_contracts_sha256": _artifact_contracts_sha256(artifacts),
                "payload_artifact_count": len(artifacts),
                "counts": counts,
                "bad_files": 0,
                "bad_rows": 0,
                "hashes_verified": True,
                "schema_verified": True,
                "parquet_verified": True,
                "megatron_verified": True,
                "packing_verified": True,
                "token_conservation_verified": True,
            }
            audit_relative = f"{prefix}/audit.json"
            _write_json(root / audit_relative, audit)
            buckets.append(
                {
                    "sequence_length": length,
                    "status": "materialized",
                    "counts": counts,
                    "artifacts": artifacts,
                    "audit": _file_descriptor(root, audit_relative),
                }
            )
        manifest: dict[str, object] = {
            "schema": OUTPUT_MANIFEST_SCHEMA,
            "status": "complete",
            "kind": kind,
            "sequence_lengths": list(TARGET_LENGTHS),
            "bindings": bindings,
            "buckets": buckets,
        }
        manifest["manifest_sha256"] = output_manifest_sha256(manifest)
        manifest_path = manifests_dir / f"{kind}.json"
        _write_json(manifest_path, manifest)
        paths[kind] = manifest_path
    return Corpus(artifact_root=root, manifests=paths)


def _reseal_manifest(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    manifest = _read_json(path)
    mutate(manifest)
    manifest["manifest_sha256"] = output_manifest_sha256(manifest)
    _write_json(path, manifest)


def _invoke(
    corpus: Corpus,
    *,
    manifests: dict[str, Path] | None = None,
    gcs_prefix: str = "gs://cppmega-handoff/distributed",
    nebius_endpoint_url: str = "https://storage.eu-north1.nebius.cloud",
) -> tuple[dict[str, object], dict[str, object]]:
    return seal_outputs(
        manifests or corpus.manifests,
        artifact_root=corpus.artifact_root,
        gcs_prefix=gcs_prefix,
        nebius_endpoint_url=nebius_endpoint_url,
        nebius_bucket="cppmega-training",
        nebius_prefix="verified/distributed",
    )


def test_complete_sparse_corpus_is_sealed_plan_only_and_deterministically(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)

    receipt, plan = _invoke(corpus)
    reversed_paths = dict(reversed(list(corpus.manifests.items())))
    second_receipt, second_plan = _invoke(corpus, manifests=reversed_paths)

    assert (second_receipt, second_plan) == (receipt, plan)
    assert receipt["status"] == "verified"
    assert receipt["training_ready"] is True
    assert receipt["blocking_reasons"] == []
    assert plan["status"] == "ready"
    assert plan["execution"] == "plan_only_no_upload"
    assert plan["upload_performed"] is False
    assert plan["publication_authorized"] is True
    artifact_set = receipt["artifact_set_sha256"]
    assert artifact_set_sha256(receipt["artifacts"]) == artifact_set
    assert plan["artifact_set_sha256"] == artifact_set
    assert {item["artifact_set_sha256"] for item in plan["destinations"]} == {
        artifact_set
    }
    gcs, nebius = plan["destinations"]
    assert gcs["immutable_create_precondition"] == {"if_generation_match": 0}
    assert nebius["immutable_create_precondition"] == {"if_none_match": "*"}
    assert len(receipt["coverage"]) == len(TARGET_LENGTHS)


def test_complete_four_kind_seven_length_matrix_is_covered(tmp_path: Path) -> None:
    corpus = _build_corpus(
        tmp_path,
        materialized={kind: set(TARGET_LENGTHS) for kind in DATA_KINDS},
    )

    receipt, plan = _invoke(corpus)

    assert receipt["training_ready"] is True
    assert plan["training_ready"] is True
    assert [item["sequence_length"] for item in receipt["coverage"]] == list(
        TARGET_LENGTHS
    )
    for item in receipt["coverage"]:
        assert item["materialized_kinds"] == list(DATA_KINDS)
        assert item["verified_zero_kinds"] == []
    artifact_paths = {item["path"] for item in receipt["artifacts"]}
    for kind in DATA_KINDS:
        for length in (32768, 65536):
            prefix = f"{kind}/{length}"
            assert f"{prefix}/data.parquet" in artifact_paths
            assert f"{prefix}/shard.bin" in artifact_paths
            assert f"{prefix}/shard.idx" in artifact_paths
            assert f"{prefix}/shard.json" in artifact_paths
            assert f"{prefix}/shard_loss_mask.bin" in artifact_paths


def test_fully_zero_sequence_is_verified_but_not_training_ready(tmp_path: Path) -> None:
    materialized = {
        kind: (set(TARGET_LENGTHS[:-1]) if kind == "source" else set())
        for kind in DATA_KINDS
    }
    corpus = _build_corpus(tmp_path, materialized=materialized)

    receipt, plan = _invoke(corpus)

    assert receipt["status"] == "verified"
    assert receipt["training_ready"] is False
    assert receipt["blocking_reasons"] == [
        "sequence length 65536 has no materialized training data"
    ]
    assert plan["status"] == "blocked"
    assert plan["publication_authorized"] is False
    assert plan["upload_performed"] is False


def test_missing_kind_manifest_fails_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    paths = dict(corpus.manifests)
    paths.pop("gitlab_mr")

    with pytest.raises(ContractError, match="exactly"):
        _invoke(corpus, manifests=paths)


def test_missing_bucket_fails_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)

    def mutate(manifest: dict[str, object]) -> None:
        manifest["buckets"].pop()  # type: ignore[union-attr]

    _reseal_manifest(corpus.manifests["source"], mutate)
    with pytest.raises(ContractError, match="complete sequence ladder"):
        _invoke(corpus)


def test_zero_bucket_requires_exact_verified_zero_receipt(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)

    def mutate(manifest: dict[str, object]) -> None:
        bucket = manifest["buckets"][-1]  # type: ignore[index]
        bucket.pop("zero_receipt")

    _reseal_manifest(corpus.manifests["github_pr"], mutate)
    with pytest.raises(ContractError, match="fields drifted"):
        _invoke(corpus)


def test_nonzero_verified_zero_receipt_fails_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    manifest_path = corpus.manifests["github_pr"]
    manifest = _read_json(manifest_path)
    bucket = manifest["buckets"][-1]  # type: ignore[index]
    descriptor = bucket["zero_receipt"]
    receipt_path = corpus.artifact_root / descriptor["path"]
    zero = _read_json(receipt_path)
    zero["row_count"] = 1
    _write_json(receipt_path, zero)
    bucket["zero_receipt"] = _file_descriptor(
        corpus.artifact_root, str(descriptor["path"])
    )
    manifest["manifest_sha256"] = output_manifest_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(ContractError, match="row_count must be zero"):
        _invoke(corpus)


def test_payload_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    payload = corpus.artifact_root / "source/1024/data.parquet"
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(ContractError, match="size differs"):
        _invoke(corpus)


def test_output_manifest_schema_drift_fails_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)

    def mutate(manifest: dict[str, object]) -> None:
        manifest["schema"] = "cppmega.distributed_output_manifest_v999"

    _reseal_manifest(corpus.manifests["source"], mutate)
    with pytest.raises(ContractError, match="schema/status is unsupported"):
        _invoke(corpus)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "not_parquet", "format does not match"),
        ("compression", "none", "compression does not match"),
        ("contract_sha256", "bad", "lowercase SHA-256"),
    ],
)
def test_payload_contract_mismatch_fails_closed(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    corpus = _build_corpus(tmp_path)

    def mutate(manifest: dict[str, object]) -> None:
        bucket = manifest["buckets"][0]  # type: ignore[index]
        bucket["artifacts"][0][field] = value

    _reseal_manifest(corpus.manifests["source"], mutate)
    with pytest.raises(ContractError, match=message):
        _invoke(corpus)


def test_non_green_audit_fails_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    manifest_path = corpus.manifests["source"]
    manifest = _read_json(manifest_path)
    bucket = manifest["buckets"][0]  # type: ignore[index]
    descriptor = bucket["audit"]
    audit_path = corpus.artifact_root / descriptor["path"]
    audit = _read_json(audit_path)
    audit["hashes_verified"] = False
    _write_json(audit_path, audit)
    bucket["audit"] = _file_descriptor(
        corpus.artifact_root, str(descriptor["path"])
    )
    manifest["manifest_sha256"] = output_manifest_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(ContractError, match="exact green bucket audit"):
        _invoke(corpus)


def test_materialized_audit_requires_source_and_producer_lineage(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    manifest_path = corpus.manifests["source"]
    manifest = _read_json(manifest_path)
    bucket = manifest["buckets"][-1]  # type: ignore[index]
    descriptor = bucket["audit"]
    audit_path = corpus.artifact_root / descriptor["path"]
    audit = _read_json(audit_path)
    audit.pop("source_receipt_sha256")
    _write_json(audit_path, audit)
    bucket["audit"] = _file_descriptor(
        corpus.artifact_root, str(descriptor["path"])
    )
    manifest["manifest_sha256"] = output_manifest_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(ContractError, match="fields drifted"):
        _invoke(corpus)


def test_64k_valid_tokens_cannot_exceed_packed_capacity(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)

    def mutate(manifest: dict[str, object]) -> None:
        bucket = manifest["buckets"][-1]  # type: ignore[index]
        bucket["counts"]["valid_tokens"] = 65537

    _reseal_manifest(corpus.manifests["source"], mutate)
    with pytest.raises(ContractError, match="packed row capacity"):
        _invoke(corpus)


def test_64k_parquet_requires_physical_footer_contract(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    manifest_path = corpus.manifests["source"]
    manifest = _read_json(manifest_path)
    bucket = manifest["buckets"][-1]  # type: ignore[index]
    parquet_record = next(
        item for item in bucket["artifacts"] if item["role"] == "parquet"
    )
    parquet_path = corpus.artifact_root / parquet_record["path"]
    raw = parquet_path.read_bytes()
    parquet_path.write_bytes(raw[:-4] + b"FAIL")
    parquet_record.update(
        _file_descriptor(corpus.artifact_root, str(parquet_record["path"]))
    )
    manifest["manifest_sha256"] = output_manifest_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(ContractError, match="Parquet footer contract"):
        _invoke(corpus)


def test_source_document_count_may_exceed_packed_row_count(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    manifest_path = corpus.manifests["source"]
    manifest = _read_json(manifest_path)
    bucket = manifest["buckets"][0]  # type: ignore[index]
    bucket["counts"]["document_count"] = 3
    descriptor = bucket["audit"]
    audit_path = corpus.artifact_root / descriptor["path"]
    audit = _read_json(audit_path)
    audit["counts"] = bucket["counts"]
    _write_json(audit_path, audit)
    bucket["audit"] = _file_descriptor(
        corpus.artifact_root, str(descriptor["path"])
    )
    manifest["manifest_sha256"] = output_manifest_sha256(manifest)
    _write_json(manifest_path, manifest)

    receipt, plan = _invoke(corpus)
    assert receipt["training_ready"] is True
    assert plan["training_ready"] is True


@pytest.mark.parametrize("length", [32768, 65536])
@pytest.mark.parametrize(
    "role",
    ["megatron_bin", "megatron_idx", "megatron_manifest", "megatron_sidecar"],
)
def test_each_required_megatron_prefix_file_is_mandatory(
    tmp_path: Path, role: str, length: int
) -> None:
    corpus = _build_corpus(tmp_path)

    def mutate(manifest: dict[str, object]) -> None:
        index = TARGET_LENGTHS.index(length)
        bucket = manifest["buckets"][index]  # type: ignore[index]
        bucket["artifacts"] = [
            item for item in bucket["artifacts"] if item["role"] != role
        ]

    _reseal_manifest(corpus.manifests["source"], mutate)
    with pytest.raises(ContractError, match="Megatron"):
        _invoke(corpus)


@pytest.mark.parametrize("kind", DATA_KINDS)
@pytest.mark.parametrize("length", [32768, 65536])
@pytest.mark.parametrize(
    "flag", ["parquet_verified", "megatron_verified", "packing_verified"]
)
def test_long_context_prefix_requires_exact_green_packing_receipt(
    tmp_path: Path, kind: str, length: int, flag: str
) -> None:
    corpus = _build_corpus(
        tmp_path,
        materialized={
            item: ({length} if item == kind else set()) for item in DATA_KINDS
        },
    )
    manifest_path = corpus.manifests[kind]
    manifest = _read_json(manifest_path)
    bucket = manifest["buckets"][TARGET_LENGTHS.index(length)]  # type: ignore[index]
    descriptor = bucket["audit"]
    audit_path = corpus.artifact_root / descriptor["path"]
    audit = _read_json(audit_path)
    audit[flag] = False
    _write_json(audit_path, audit)
    bucket["audit"] = _file_descriptor(
        corpus.artifact_root, str(descriptor["path"])
    )
    manifest["manifest_sha256"] = output_manifest_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(ContractError, match="exact green bucket audit"):
        _invoke(corpus)


@pytest.mark.parametrize("kind", DATA_KINDS)
@pytest.mark.parametrize("length", [32768, 65536])
def test_long_context_mmididx_must_match_bin_and_counts(
    tmp_path: Path, kind: str, length: int
) -> None:
    corpus = _build_corpus(
        tmp_path,
        materialized={
            item: ({length} if item == kind else set()) for item in DATA_KINDS
        },
    )
    manifest_path = corpus.manifests[kind]
    manifest = _read_json(manifest_path)
    bucket = manifest["buckets"][TARGET_LENGTHS.index(length)]  # type: ignore[index]
    index_record = next(
        item for item in bucket["artifacts"] if item["role"] == "megatron_idx"
    )
    index_path = corpus.artifact_root / index_record["path"]
    raw = bytearray(index_path.read_bytes())
    struct.pack_into("<i", raw, 34, length - 1)
    index_path.write_bytes(raw)
    index_record.update(
        _file_descriptor(corpus.artifact_root, str(index_record["path"]))
    )
    manifest["manifest_sha256"] = output_manifest_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(ContractError, match="idx counts do not close"):
        _invoke(corpus)


def test_duplicate_artifact_path_fails_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)

    def mutate(manifest: dict[str, object]) -> None:
        bucket = manifest["buckets"][0]  # type: ignore[index]
        artifacts = bucket["artifacts"]
        artifacts.insert(1, dict(artifacts[0]))

    _reseal_manifest(corpus.manifests["source"], mutate)
    with pytest.raises(ContractError, match="unique and path-sorted"):
        _invoke(corpus)


def test_cross_kind_tokenizer_drift_fails_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    manifest_path = corpus.manifests["github_pr"]
    manifest = _read_json(manifest_path)
    replacement = _sha("different-tokenizer")
    manifest["bindings"]["tokenizer_sha256"] = replacement  # type: ignore[index]
    for bucket in manifest["buckets"]:  # type: ignore[union-attr]
        if bucket["status"] == "materialized":
            audit_descriptor = bucket["audit"]
            audit_relative = str(audit_descriptor["path"])
            audit_path = corpus.artifact_root / audit_relative
            audit = _read_json(audit_path)
            audit["tokenizer_sha256"] = replacement
            _write_json(audit_path, audit)
            bucket["audit"] = _file_descriptor(
                corpus.artifact_root, audit_relative
            )
            continue
        descriptor = bucket["zero_receipt"]
        relative = str(descriptor["path"])
        receipt_path = corpus.artifact_root / relative
        binding_receipt = _read_json(receipt_path)
        binding_receipt["tokenizer_sha256"] = replacement
        _write_json(receipt_path, binding_receipt)
        bucket["zero_receipt"] = _file_descriptor(corpus.artifact_root, relative)
    manifest["manifest_sha256"] = output_manifest_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(ContractError, match="tokenizer/schema"):
        _invoke(corpus)


def test_symlink_artifact_fails_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    artifact = corpus.artifact_root / "source/1024/data.parquet"
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(outside)

    with pytest.raises(ContractError, match="regular file"):
        _invoke(corpus)


def test_symlink_artifact_root_fails_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    linked_root = tmp_path / "linked-artifacts"
    linked_root.symlink_to(corpus.artifact_root, target_is_directory=True)
    linked_corpus = Corpus(artifact_root=linked_root, manifests=corpus.manifests)

    with pytest.raises(ContractError, match="artifact root must not be a symlink"):
        _invoke(linked_corpus)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gcs_prefix": "gs://cppmega-handoff/../escape"}, "unsafe object path"),
        (
            {"nebius_endpoint_url": "https://secret@example.invalid"},
            "credential-free HTTPS origin",
        ),
    ],
)
def test_unsafe_destination_fails_closed(
    tmp_path: Path, kwargs: dict[str, str], message: str
) -> None:
    corpus = _build_corpus(tmp_path)

    with pytest.raises(ContractError, match=message):
        _invoke(corpus, **kwargs)
