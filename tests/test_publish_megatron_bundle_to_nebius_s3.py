from __future__ import annotations

import hashlib
import base64
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

import scripts.data.publish_megatron_bundle_to_nebius_s3 as publisher
from scripts.data.publish_megatron_bundle_to_nebius_s3 import (
    _head,
    _head_matches,
    _stable_upload_snapshot,
    _upload_file,
    _validate_archive,
    _validate_archive_member_names,
    _validate_bundle,
    _validate_tokenizer_directory,
    main,
)


def _bundle(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    artifact = prefix.with_suffix(".bin")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return artifact, digest


def _archive(tmp_path, artifact):
    raw_archive = tmp_path / "bundle.tar"
    archive = tmp_path / "bundle.tar.zst"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    with tarfile.open(raw_archive, "w") as tar:
        for record in manifest["artifacts"]:
            tar.add(tmp_path / record["path"], arcname=record["path"])
        tar.add(tmp_path / "manifest.json", arcname="manifest.json")
    subprocess.run(
        ["zstd", "-q", "-1", str(raw_archive), "-o", str(archive)], check=True
    )
    return archive


def _write_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _objective_payload():
    tasks = ("causal_lm", "fim", "ast_fim", "ifim", "commit_diff", "pre_to_post")
    return {
        "schema": "cppmega_pre_materialized_objectives_v1",
        "algorithm": "hamilton_eligibility_bipartite_v1",
        "seed": 17,
        "quota_window_samples": 6,
        "task_order": list(tasks),
        "configured_rates": {task: "1/6" for task in tasks},
        "planned_samples": {task: 1 for task in tasks},
        "realized": {
            task: {
                "samples": 1,
                "input_tokens": 3,
                "loss_tokens": 3 if task == "causal_lm" else 2,
            }
            for task in tasks
        },
        "totals": {"samples": 6, "input_tokens": 18, "loss_tokens": 13},
        "typed_sources": {
            "ifim_instruction": "ifim_instruction_token_ids",
            "commit_message": "commit_msg_token_ids",
            "diff": "diff_token_ids",
            "pre": "pre_token_ids",
            "post": "post_token_ids",
            "missing_fields": "ineligible",
            "rendered_text_parsing": False,
        },
        "graph_auxiliary": {
            "relations": ["domain"],
            "eligible_samples": 1,
            "positive_edges": 1,
            "global_weight": "1",
            "bce_weight": "1/10",
            "coverage_weight": "1/20",
            "topk": 8,
            "pos_weight": "1",
            "margin": "1",
            "included_in_total_loss": True,
            "runtime": "megatron_dsa_indexer_v1",
            "pair_mask": "causal_same_document_upstream_v1",
            "chunk_edge_expansion": "cartesian_token_spans_v1",
        },
        "materialization": {
            "format": "shifted_lm_document_v1",
            "token_column": "input_ids",
            "loss_mask_column": "loss_mask",
            "length_column": "valid_token_count",
            "objective_column": "objective_kind",
            "document_id_column": "doc_ids",
            "source_document_id_column": "token_source_doc_ids",
        },
    }


def _prefix_bundle(tmp_path):
    prefix = tmp_path / "data" / "seq_1024" / "cppmega_train"
    tokens_per_document = 4
    document_count = 6
    token_count = tokens_per_document * document_count
    token_values = [
        value
        for document in range(document_count)
        for value in (10 + document, 20 + document, 30 + document, 40 + document)
    ]
    _write_bytes(
        prefix.with_suffix(".bin"),
        struct.pack(f"<{token_count}H", *token_values),
    )
    _write_bytes(
        prefix.with_suffix(".idx"),
        b"MMIDIDX\x00\x00"
        + struct.pack("<QBQQ", 1, 8, document_count, document_count + 1)
        + struct.pack(f"<{document_count}i", *([tokens_per_document] * document_count))
        + struct.pack(
            f"<{document_count}q",
            *(index * tokens_per_document * 2 for index in range(document_count)),
        )
        + struct.pack(
            f"<{document_count + 1}q", *range(document_count + 1)
        ),
    )

    side_channel_paths = {}
    required_token_sidecars = set(publisher.REQUIRED_TOKEN_SIDECARS) | {
        "token_source_doc_ids"
    }
    for name in sorted(required_token_sidecars):
        dtype = "uint8" if name in {
            "loss_mask",
            "token_confidence_ids",
            "token_structure_ids",
            "token_def_use",
            "token_change_mask_pre",
            "token_change_mask_post",
        } else "uint16"
        if name in {
            "token_entity_ids",
            "token_scope_ids",
            "token_symbol_ids",
            "token_call_targets",
            "token_type_refs",
            "token_source_doc_ids",
        }:
            dtype = "uint32"
        rel = f"{prefix.name}_{name}.bin"
        payload = bytearray(token_count * publisher.DTYPE_SIZES[dtype])
        if name == "token_structure_ids":
            payload[0] = 1
        if name == "loss_mask":
            payload[:] = bytes(
                value
                for document in range(document_count)
                for value in (
                    (1, 1, 1, 0)
                    if document == 0
                    else (0, 1, 1, 0)
                )
            )
        if name == "doc_ids":
            payload[:] = struct.pack(
                f"<{token_count}H",
                *(
                    value
                    for _document in range(document_count)
                    for value in (1, 1, 1, 1)
                ),
            )
        if name == "token_source_doc_ids":
            payload[:] = struct.pack(
                f"<{token_count}I",
                *(
                    document + 1
                    for document in range(document_count)
                    for _token in range(tokens_per_document)
                ),
            )
        _write_bytes(prefix.parent / rel, payload)
        side_channel_paths[name] = {"path": rel, "dtype": dtype}

    graph_sidecar_paths = {}
    for name in sorted(publisher.REQUIRED_GRAPH_SIDECARS):
        if name in {"token_call_edges", "token_type_edges"}:
            kind = "edge_pairs"
            dtype = "int32"
            shape_tail = [2]
            item_count = 0
        elif name.endswith("_edges"):
            kind = "edge_triples"
            dtype = "int32"
            shape_tail = [3]
            item_count = 1 if name == "token_domain_edges" else 0
        else:
            kind = "ragged_1d"
            dtype = "uint32" if name in {"token_chunk_starts", "token_chunk_ends"} else "uint16"
            shape_tail = [1]
            item_count = 1
        offsets_rel = f"{prefix.name}_{name}_offsets.bin"
        data_rel = f"{prefix.name}_{name}_data.bin"
        _write_bytes(
            prefix.parent / offsets_rel,
            struct.pack(
                f"<{document_count + 1}q",
                0,
                *([item_count] * document_count),
            ),
        )
        payload = b"\x00" * (
            item_count * shape_tail[0] * publisher.DTYPE_SIZES[dtype]
        )
        if name == "token_domain_edges":
            payload = struct.pack("<3i", 1, 0, 5)
        elif name == "token_chunk_ends":
            payload = struct.pack("<I", tokens_per_document)
        elif name == "token_chunk_kinds":
            payload = struct.pack("<H", 1)
        _write_bytes(prefix.parent / data_rel, payload)
        graph_sidecar_paths[name] = {
            "kind": kind,
            "offsets_path": offsets_rel,
            "data_path": data_rel,
            "offset_dtype": "int64",
            "dtype": dtype,
            "item_count": item_count,
            "shape_tail": shape_tail,
        }

    source_platform = {
        "schema": "cppmega_source_platform_v1",
        "sequence_doc_offsets_path": f"{prefix.name}_source_platform_sequence_doc_offsets.bin",
        "doc_platform_offsets_path": f"{prefix.name}_source_platform_doc_id_offsets.bin",
        "platform_ids_path": f"{prefix.name}_source_platform_ids.bin",
        "source_document_count": document_count,
        "platform_id_count": document_count,
    }
    _write_bytes(
        prefix.parent / source_platform["sequence_doc_offsets_path"],
        struct.pack(f"<{document_count + 1}q", *range(document_count + 1)),
    )
    _write_bytes(
        prefix.parent / source_platform["doc_platform_offsets_path"],
        struct.pack(f"<{document_count + 1}q", *range(document_count + 1)),
    )
    _write_bytes(
        prefix.parent / source_platform["platform_ids_path"],
        struct.pack(f"<{document_count}H", *range(1, document_count + 1)),
    )
    objective_payload = _objective_payload()
    objective_sha256 = hashlib.sha256(
        json.dumps(
            objective_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    objective_ids_rel = f"{prefix.name}_objective_ids.bin"
    _write_bytes(
        prefix.parent / objective_ids_rel,
        bytes(range(1, document_count + 1)),
    )
    objective_contract = {
        "schema": "cppmega_pre_materialized_objectives_v1",
        "sha256": objective_sha256,
        "payload": objective_payload,
        "objective_id_sidecar": {
            "path": objective_ids_rel,
            "dtype": "uint8",
            "document_aligned": True,
        },
    }
    prefix_manifest = {
        "vocab_size": 65536,
        "tokenizer_contract": "megacpp",
        "dtype": "uint16",
        "token_count": token_count,
        "document_count": document_count,
        "graph_sidecar_schema": "cppmega_graph_routes_v2",
        "side_channel_paths": side_channel_paths,
        "graph_sidecar_paths": graph_sidecar_paths,
        "source_platform_sidecar": source_platform,
        "objective_contract": objective_contract,
    }
    prefix.with_suffix(".json").write_text(json.dumps(prefix_manifest), encoding="utf-8")

    tokenizer_dir = tmp_path / "tokenizer"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "data/tokenizer_v2",
        tokenizer_dir,
    )
    provenance_dir = tmp_path / "provenance"
    provenance_dir.mkdir()
    objective_source = provenance_dir / "objective_contract.json"
    objective_source.write_text(json.dumps(objective_payload), encoding="utf-8")

    paths = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    records = [
        {
            "path": str(path.relative_to(tmp_path)),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    artifact_set_sha256 = hashlib.sha256(
        json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    tokenizer_records = [
        record for record in records if str(record["path"]).startswith("tokenizer/")
    ]
    tokenizer_set_sha256 = hashlib.sha256(
        json.dumps(tokenizer_records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema": "cppmega_megatron_bundle_v1",
        "bundle_id": f"test-bundle-{artifact_set_sha256[:16]}",
        "tokenizer_contract": "megacpp-vocab-65536",
        "vocab_size": 65536,
        "training_contract": "objective_materialized",
        "objective_materialization": {
            "path": "provenance/objective_contract.json",
            "schema": "cppmega_pre_materialized_objectives_v1",
            "sha256": objective_sha256,
            "file_sha256": hashlib.sha256(objective_source.read_bytes()).hexdigest(),
        },
        "buckets": [1024],
        "tokenizer": {
            "path": "tokenizer",
            "contract": "megacpp-vocab-65536",
            "vocab_size": 65536,
            "files": tokenizer_records,
            "artifact_set_sha256": tokenizer_set_sha256,
        },
        "bucket_results": [{"bucket": 1024, "prefix": str(prefix.relative_to(tmp_path)), "manifest": prefix_manifest}],
        "artifact_count": len(records),
        "artifact_bytes": sum(int(record["size"]) for record in records),
        "artifact_set_sha256": artifact_set_sha256,
        "artifacts": records,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return prefix


def _rehash_bundle_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in manifest["artifacts"]:
        path = tmp_path / record["path"]
        records.append(
            {
                "path": record["path"],
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    records.sort(key=lambda record: record["path"])
    artifact_set_sha256 = hashlib.sha256(
        json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest["artifacts"] = records
    manifest["artifact_count"] = len(records)
    manifest["artifact_bytes"] = sum(record["size"] for record in records)
    manifest["artifact_set_sha256"] = artifact_set_sha256
    manifest["bundle_id"] = f"test-bundle-{artifact_set_sha256[:16]}"
    for result in manifest["bucket_results"]:
        prefix_manifest_path = tmp_path / (str(result["prefix"]) + ".json")
        result["manifest"] = json.loads(
            prefix_manifest_path.read_text(encoding="utf-8")
        )
    tokenizer_records = [
        record for record in records if record["path"].startswith("tokenizer/")
    ]
    manifest["tokenizer"]["files"] = tokenizer_records
    manifest["tokenizer"]["artifact_set_sha256"] = hashlib.sha256(
        json.dumps(tokenizer_records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_validate_bundle_rehashes_every_manifest_artifact(tmp_path):
    artifact, digest = _bundle(tmp_path)

    manifest, records = _validate_bundle(tmp_path, hash_jobs=2)

    assert manifest["artifact_bytes"] == sum(record["size"] for record in records)
    artifact_record = next(record for record in records if record["local_path"] == str(artifact))
    assert artifact_record["sha256"] == digest


def test_validate_bundle_rejects_manifest_path_escape(tmp_path):
    artifact, digest = _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "../sample.bin"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe artifact path"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_artifact_count_mismatch(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_count"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_missing_per_file_sha256(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0].pop("sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="valid sha256"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_wrong_tokenizer_contract(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer_contract"] = "wrong-tokenizer"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer_contract"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_tokenizer_rejects_case5_semantic_reserved_id_drift(tmp_path):
    tokenizer_root = tmp_path / "tokenizer"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "data/tokenizer_v2",
        tokenizer_root,
    )
    tokenizer_path = tokenizer_root / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    del vocab["<RESERVED_237>"]
    vocab["<DRIFTED_237>"] = 237
    entry = next(item for item in tokenizer["added_tokens"] if item["id"] == 237)
    entry["content"] = "<DRIFTED_237>"
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")

    with pytest.raises(ValueError, match="reserved.*237|ID 237"):
        _validate_tokenizer_directory(tokenizer_root)


def test_validate_bundle_rejects_zero_route_edges(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest_path = prefix.with_suffix(".json")
    prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
    spec = prefix_manifest["graph_sidecar_paths"]["token_domain_edges"]
    spec["item_count"] = 0
    (prefix.parent / spec["offsets_path"]).write_bytes(
        struct.pack("<7q", *([0] * 7))
    )
    (prefix.parent / spec["data_path"]).write_bytes(b"")
    prefix_manifest_path.write_text(json.dumps(prefix_manifest), encoding="utf-8")
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="nonempty route edge"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_prefix_rejects_graph_routes_without_case1_objective_contract(
    tmp_path,
):
    prefix = _prefix_bundle(tmp_path)
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("objective_contract")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="objective_contract"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_bundle_requires_positive_uint32_source_doc_ids(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    spec = prefix_manifest["side_channel_paths"]["token_source_doc_ids"]
    (prefix.parent / spec["path"]).write_bytes(struct.pack("<24I", *([0] * 24)))
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="token_source_doc_ids.*positive"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_edge_endpoint_document_provenance_mismatch(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    spec = prefix_manifest["side_channel_paths"]["token_source_doc_ids"]
    source_ids = [
        document + 1
        for document in range(6)
        for _token in range(4)
    ]
    source_ids[1] = 2
    (prefix.parent / spec["path"]).write_bytes(
        struct.pack("<24I", *source_ids)
    )
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="provenance|source document"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_wrong_domain_edge_family_kind_26(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    spec = prefix_manifest["graph_sidecar_paths"]["token_domain_edges"]
    (prefix.parent / spec["data_path"]).write_bytes(struct.pack("<3i", 1, 0, 26))
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="kind 26.*token_domain_edges"):
        _validate_bundle(tmp_path, hash_jobs=1)


@pytest.mark.parametrize(
    ("tokens", "domains", "roles", "confidences", "message"),
    [
        (
            [191, 195, 196, 192],
            [1, 1, 2, 1],
            [1, 1, 1, 1],
            [4, 4, 4, 4],
            "wrong domain",
        ),
        (
            [192, 191, 10, 11],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [4, 4, 0, 0],
            "unmatched",
        ),
        (
            [191, 195, 192, 196],
            [1, 2, 1, 2],
            [1, 1, 1, 1],
            [4, 4, 4, 4],
            "crossing",
        ),
        (
            [191, 195, 196, 10],
            [1, 2, 2, 0],
            [1, 1, 1, 0],
            [4, 4, 4, 0],
            "unclosed",
        ),
    ],
)
def test_validate_bundle_rejects_invalid_domain_delimiter_mapping_and_balance(
    tmp_path, tokens, domains, roles, confidences, message
):
    prefix = _prefix_bundle(tmp_path)
    token_values = list(
        struct.unpack("<24H", prefix.with_suffix(".bin").read_bytes())
    )
    token_values[:4] = tokens
    prefix.with_suffix(".bin").write_bytes(struct.pack("<24H", *token_values))
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    for sidecar, values, dtype in (
        ("token_domain_ids", domains, "H"),
        ("token_role_ids", roles, "H"),
        ("token_confidence_ids", confidences, "B"),
    ):
        spec = manifest["side_channel_paths"][sidecar]
        path = prefix.parent / spec["path"]
        width = publisher.DTYPE_SIZES[spec["dtype"]]
        payload = bytearray(path.read_bytes())
        payload[: 4 * width] = struct.pack(f"<4{dtype}", *values)
        path.write_bytes(payload)
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match=message):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_missing_staged_tokenizer_contract(tmp_path):
    _prefix_bundle(tmp_path)
    contract_path = tmp_path / "tokenizer/tokenizer_contract_v1.json"
    contract_path.unlink()
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        record
        for record in manifest["artifacts"]
        if record["path"] != "tokenizer/tokenizer_contract_v1.json"
    ]
    manifest["tokenizer"]["files"] = [
        record
        for record in manifest["tokenizer"]["files"]
        if record["path"] != "tokenizer/tokenizer_contract_v1.json"
    ]
    manifest["tokenizer"]["artifact_set_sha256"] = publisher._artifact_set_sha256(
        manifest["tokenizer"]["files"]
    )
    manifest["artifact_count"] = len(manifest["artifacts"])
    manifest["artifact_bytes"] = sum(
        record["size"] for record in manifest["artifacts"]
    )
    manifest["artifact_set_sha256"] = publisher._artifact_set_sha256(
        manifest["artifacts"]
    )
    manifest["bundle_id"] = (
        f"test-bundle-{manifest['artifact_set_sha256'][:16]}"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required regular artifacts"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_requires_hashed_tokenizer_artifacts(tmp_path):
    _prefix_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("tokenizer")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer descriptor"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_tokenizer_vocab_drift_even_when_rehashed(tmp_path):
    _prefix_bundle(tmp_path)
    tokenizer_path = tmp_path / "tokenizer/tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"].pop(next(iter(tokenizer["model"]["vocab"])))
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="tokenizer vocab size"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_tokenizer_rejects_same_size_special_token_id_drift(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "data/tokenizer_v2",
        tokenizer_dir,
    )
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"]["<BOS>"] = 3
    tokenizer["model"]["vocab"]["<EOS>"] = 2
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")

    with pytest.raises(ValueError, match="token.*<BOS>.*(disagrees|must remain)"):
        publisher._validate_tokenizer_directory(tokenizer_dir)


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (9, struct.pack("<Q", 2), "MMIDIDX version"),
        (17, struct.pack("<B", 4), "MMIDIDX dtype"),
        (66, struct.pack("<q", 2), "sequence pointers"),
        (114, struct.pack("<q", 2), "document indices"),
    ],
)
def test_validate_prefix_rejects_mmididx_contract_drift(
    tmp_path, offset, replacement, message
):
    prefix = _prefix_bundle(tmp_path)
    index_path = prefix.with_suffix(".idx")
    payload = bytearray(index_path.read_bytes())
    payload[offset : offset + len(replacement)] = replacement
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_prefix_rejects_in_tree_sidecar_symlink(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    sidecar = prefix.parent / manifest["side_channel_paths"]["loss_mask"]["path"]
    sidecar.unlink()
    sidecar.symlink_to(prefix.with_suffix(".bin"))

    with pytest.raises(ValueError, match="regular file.*symlink"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_bundle_rejects_embedded_prefix_manifest_drift(tmp_path):
    _prefix_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bucket_results"][0]["manifest"]["vocab_size"] = 42
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="embedded prefix manifest"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_wrong_prefix_graph_schema(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["graph_sidecar_schema"] = "flat_tokens_only"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="graph_sidecar_schema"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_bad_graph_csr_offsets(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    bad_offsets = prefix.parent / f"{prefix.name}_token_chunk_starts_offsets.bin"
    bad_offsets.write_bytes(struct.pack("<7q", 1, 1, 1, 1, 1, 1, 1))

    with pytest.raises(ValueError, match="CSR offsets"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_prefix_rejects_misaligned_graph_chunk_csr_counts(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["graph_sidecar_paths"]["token_chunk_ends"]
    spec["item_count"] = 2
    (prefix.parent / spec["offsets_path"]).write_bytes(
        struct.pack("<7q", 0, 2, 2, 2, 2, 2, 2)
    )
    (prefix.parent / spec["data_path"]).write_bytes(struct.pack("<2I", 1, 2))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="chunk CSR item counts"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_head_contract_requires_size_and_sha_metadata():
    assert _head_matches(
        {"ContentLength": 8, "Metadata": {"sha256": "abc"}},
        size=8,
        sha256="abc",
    )
    assert not _head_matches(
        {"ContentLength": 7, "Metadata": {"sha256": "abc"}},
        size=8,
        sha256="abc",
    )
    assert not _head_matches(
        {"ContentLength": 8, "Metadata": {}}, size=8, sha256="abc"
    )


def test_head_contract_accepts_nebius_metadata_key_casing():
    assert _head_matches(
        {"ContentLength": 8, "Metadata": {"Sha256": "abc"}},
        size=8,
        sha256="abc",
    )


def test_head_contract_accepts_server_verified_multipart_composite_checksum():
    assert _head_matches(
        {
            "ContentLength": 8,
            "Metadata": {"sha256": "a" * 64},
            "ChecksumSHA256": "YWJjZA==-2",
            "ChecksumType": "COMPOSITE",
        },
        size=8,
        sha256="a" * 64,
    )


def test_head_distinguishes_missing_object_from_transport_failure(monkeypatch):
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=254, stdout="", stderr="An error occurred (404) when calling HeadObject"
        ),
    )
    assert _head(endpoint="https://s3.invalid", bucket="b", key="missing", env={}) is None

    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=255, stdout="", stderr="Could not connect to endpoint"
        ),
    )
    with pytest.raises(RuntimeError, match="remote HEAD failed"):
        _head(endpoint="https://s3.invalid", bucket="b", key="unknown", env={})


def test_dry_run_upload_never_calls_aws(tmp_path):
    artifact, digest = _bundle(tmp_path)

    receipt = _upload_file(
        local=artifact,
        endpoint="https://example.invalid",
        bucket="bucket",
        key="prefix/sample.bin",
        size=artifact.stat().st_size,
        sha256=digest,
        env={},
        dry_run=True,
    )

    assert receipt["status"] == "dry_run"
    assert receipt["sha256"] == digest


def test_immutable_bundle_object_rejects_existing_remote_mismatch(
    tmp_path, monkeypatch
):
    artifact, digest = _bundle(tmp_path)
    monkeypatch.setattr(
        publisher,
        "_head",
        lambda **_kwargs: {
            "ContentLength": artifact.stat().st_size,
            "Metadata": {"sha256": "different"},
        },
    )

    def forbidden_upload(*_args, **_kwargs):
        raise AssertionError("immutable mismatch must fail before aws s3 cp")

    monkeypatch.setattr(publisher.subprocess, "run", forbidden_upload)
    with pytest.raises(RuntimeError, match="immutable remote object mismatch"):
        _upload_file(
            local=artifact,
            endpoint="https://example.invalid",
            bucket="bucket",
            key="bundles/test-bundle/data/sample.bin",
            size=artifact.stat().st_size,
            sha256=digest,
            env={},
            dry_run=False,
        )


def test_small_upload_is_checksum_bound_and_create_only(tmp_path, monkeypatch):
    artifact, digest = _bundle(tmp_path)
    expected_checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    heads = iter(
        [
            None,
            {
                "ContentLength": artifact.stat().st_size,
                "Metadata": {"sha256": digest},
                "ChecksumSHA256": expected_checksum,
                "ETag": '"etag"',
            },
        ]
    )
    monkeypatch.setattr(publisher, "_head", lambda **_kwargs: next(heads))

    @contextmanager
    def stable_snapshot(local, **_kwargs):
        yield local

    monkeypatch.setattr(publisher, "_stable_upload_snapshot", stable_snapshot)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    receipt = _upload_file(
        local=artifact,
        endpoint="https://example.invalid",
        bucket="bucket",
        key="bundles/test-bundle/data/sample.bin",
        size=artifact.stat().st_size,
        sha256=digest,
        env={},
        dry_run=False,
    )

    command = commands[0]
    assert command[:3] == ["aws", "s3api", "put-object"]
    assert command[command.index("--checksum-sha256") + 1] == expected_checksum
    assert command[command.index("--if-none-match") + 1] == "*"
    assert receipt["status"] == "uploaded_verified"


def test_latest_pointer_update_uses_remote_etag_compare_and_swap(
    tmp_path, monkeypatch
):
    artifact, digest = _bundle(tmp_path)
    expected_checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    heads = iter(
        [
            {
                "ContentLength": artifact.stat().st_size,
                "Metadata": {"sha256": "0" * 64},
                "ETag": '"old-etag"',
            },
            {
                "ContentLength": artifact.stat().st_size,
                "Metadata": {"sha256": digest},
                "ChecksumSHA256": expected_checksum,
                "ETag": '"new-etag"',
            },
        ]
    )
    monkeypatch.setattr(publisher, "_head", lambda **_kwargs: next(heads))

    @contextmanager
    def stable_snapshot(local, **_kwargs):
        yield local

    monkeypatch.setattr(publisher, "_stable_upload_snapshot", stable_snapshot)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    _upload_file(
        local=artifact,
        endpoint="https://example.invalid",
        bucket="bucket",
        key="latest.json",
        size=artifact.stat().st_size,
        sha256=digest,
        env={},
        dry_run=False,
        allow_overwrite=True,
    )

    command = commands[0]
    assert command[command.index("--if-match") + 1] == '"old-etag"'


def test_upload_snapshot_isolated_from_source_mutation(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable")
    digest = hashlib.sha256(b"stable").hexdigest()

    with _stable_upload_snapshot(source, size=6, sha256=digest) as snapshot:
        source.write_bytes(b"changed")
        assert snapshot.read_bytes() == b"stable"


def test_archive_member_set_must_be_exact_and_unique():
    _validate_archive_member_names(
        ["data/sample.bin", "manifest.json"],
        {"data/sample.bin", "manifest.json"},
    )
    with pytest.raises(ValueError, match="duplicate"):
        _validate_archive_member_names(
            ["manifest.json", "manifest.json"], {"manifest.json"}
        )
    with pytest.raises(ValueError, match="member set mismatch"):
        _validate_archive_member_names(
            ["manifest.json", "unexpected.bin"],
            {"manifest.json", "data/sample.bin"},
        )


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_validate_archive_binds_exact_members_and_manifest(tmp_path):
    artifact, _digest = _bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    archive = _archive(tmp_path, artifact)

    size, digest = _validate_archive(
        bundle=tmp_path, archive=archive, manifest=manifest
    )

    assert size == archive.stat().st_size
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_validate_archive_rejects_payload_that_disagrees_with_manifest(tmp_path):
    artifact, _digest = _bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    payload = bytearray(artifact.read_bytes())
    payload[0] ^= 1
    artifact.write_bytes(payload)
    archive = _archive(tmp_path, artifact)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate_archive(bundle=tmp_path, archive=archive, manifest=manifest)


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_archive_publish_runs_full_prefix_contract_validation(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest_path = prefix.with_suffix(".json")
    prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
    prefix_manifest["graph_sidecar_schema"] = "stale_graph_schema"
    prefix_manifest_path.write_text(json.dumps(prefix_manifest), encoding="utf-8")
    manifest = _rehash_bundle_manifest(tmp_path)
    manifest["bucket_results"][0]["manifest"] = prefix_manifest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive = _archive(tmp_path, prefix.with_suffix(".bin"))

    with pytest.raises(ValueError, match="graph_sidecar_schema"):
        main(["--bundle", str(tmp_path), "--archive", str(archive), "--dry-run"])


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_archive_transport_dry_run_writes_commit_order_receipt(tmp_path):
    artifact, _digest = _bundle(tmp_path)
    archive = _archive(tmp_path, artifact)

    assert main(["--bundle", str(tmp_path), "--archive", str(archive), "--dry-run"]) == 0

    receipt = json.loads(
        (tmp_path / "archive_publish_receipt.json").read_text(encoding="utf-8")
    )
    bundle_id = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )["bundle_id"]
    assert receipt["archive"]["key"].startswith(
        f"cppmega-megatron/macro-routes/transports/{bundle_id}/bundle-"
    )
    assert receipt["archive"]["key"].endswith(".tar.zst")
    assert receipt["logical_manifest"]["key"].endswith(
        f"/{bundle_id}/logical_manifest.json"
    )
    assert receipt["transport"]["key"].endswith(f"/{bundle_id}/transport.json")
    assert receipt["latest_transport"]["key"].endswith("/latest_transport.json")
    assert receipt["archive"]["status"] == "dry_run"
    assert receipt["status"] == "complete"
    assert receipt["archive_validation"] == {
        "status": "verified",
        "member_count": len(
            json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[
                "artifacts"
            ]
        )
        + 1,
        "artifact_set_sha256": json.loads(
            (tmp_path / "manifest.json").read_text(encoding="utf-8")
        )["artifact_set_sha256"],
        "logical_manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.json").read_bytes()
        ).hexdigest(),
    }


def test_loose_publish_receipt_is_incremental_and_bundle_bound(tmp_path):
    _prefix_bundle(tmp_path)

    assert main(["--bundle", str(tmp_path), "--dry-run", "--jobs", "2"]) == 0

    receipt_path = tmp_path / "publish_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["bundle_id"] == manifest["bundle_id"]
    assert receipt["artifact_set_sha256"] == manifest["artifact_set_sha256"]
    assert len(receipt["artifacts"]) == manifest["artifact_count"]

    receipt["bundle_id"] = "different-bundle"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="publish receipt binding"):
        main(["--bundle", str(tmp_path), "--dry-run"])


def test_loose_publish_rejects_stale_artifact_receipt_entry(tmp_path):
    _prefix_bundle(tmp_path)
    assert main(["--bundle", str(tmp_path), "--dry-run"]) == 0
    receipt_path = tmp_path / "publish_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"][0]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="publish receipt artifact mismatch"):
        main(["--bundle", str(tmp_path), "--dry-run"])
