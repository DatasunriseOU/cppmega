import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
import tarfile

import pytest

from cppmega.megatron.graph_recipe import (
    STAGE1_GRAPH_RELATIONS,
    STAGE1_GRAPH_TOPK,
    stage1_graph_recipe_binding,
)
import scripts.data.publish_megatron_bundle_to_nebius_s3 as publisher
import scripts.nebius_h200_megatron_cpp_world_sweep as sweep_module

from scripts.nebius_h200_megatron_cpp_world_sweep import (
    DEFAULT_DOCKER_IMAGE,
    OVERLAY_PATHS,
    _assert_prefix_contract,
    _docker_auth_from_config,
    first_public_ip,
    instance_delete_allowed,
    main,
    make_checkpoint_tar,
    make_bundle_tar,
    make_ghcr_auth_tar,
    make_overlay_tar,
    make_multi_sidecar_tar,
    remote_run_script,
    validate_docker_image_digest,
    validate_nebius_resource_id,
)
from scripts.h200_megatron_preflight import (
    derive_graph_capacity_receipt,
    main as h200_preflight_main,
)


_DTYPE_SIZES = publisher.DTYPE_SIZES
_TEST_GRAPH_CAPACITY = (1, 1)
NONZERO_GRAPH_SIDECARS = publisher.NONZERO_GRAPH_SIDECARS
REQUIRED_GRAPH_SIDECARS = publisher.REQUIRED_GRAPH_SIDECARS
REQUIRED_TOKEN_SIDECARS = publisher.REQUIRED_TOKEN_SIDECARS


def _source_identity(source):
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    identity_id = int.from_bytes(digest[:8], "big", signed=False)
    if identity_id == 0:
        identity_id = int.from_bytes(digest[8:16], "big", signed=False)
    return identity_id, digest.hex()


def _write_valid_sidecar_prefix(
    prefix,
    *,
    edge_capacity=1,
    chunk_capacity=1,
    sequence_length=1024,
):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tokens_per_document = 4
    document_count = 6
    token_count = tokens_per_document * document_count
    source_identities = [
        (
            *_source_identity(f'{{"source_path":"src/doc-{document}.cc"}}'),
            f'{{"source_path":"src/doc-{document}.cc"}}',
        )
        for document in range(document_count)
    ]
    prefix.with_suffix(".bin").write_bytes(
        struct.pack(
            f"<{token_count}H",
            *(
                value
                for document in range(document_count)
                for value in (10 + document, 20 + document, 30 + document, 40 + document)
            ),
        )
    )
    prefix.with_suffix(".idx").write_bytes(
        b"MMIDIDX\x00\x00"
        + struct.pack("<QBQQ", 1, 8, document_count, document_count + 1)
        + struct.pack(f"<{document_count}i", *([tokens_per_document] * document_count))
        + struct.pack(
            f"<{document_count}q",
            *(index * tokens_per_document * 2 for index in range(document_count)),
        )
        + struct.pack(f"<{document_count + 1}q", *range(document_count + 1))
    )

    side_channel_paths = {}
    required_token_sidecars = set(REQUIRED_TOKEN_SIDECARS)
    for name in sorted(required_token_sidecars):
        dtype = publisher.TOKEN_SIDECAR_DTYPES[name]
        relative = f"{prefix.name}_{name}.bin"
        payload = bytearray(token_count * _DTYPE_SIZES[dtype])
        if name == "token_structure_ids":
            payload[0] = 1
        if name == "loss_mask":
            payload[:] = bytes(
                value
                for document in range(document_count)
                for value in ((1, 1, 1, 0) if document == 0 else (0, 1, 1, 0))
            )
        if name == "doc_ids":
            payload[:] = struct.pack(
                f"<{token_count}I",
                *(1 for _document in range(document_count) for _ in range(4)),
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
        if name == "token_source_identity_ids":
            payload[:] = struct.pack(
                f"<{token_count}Q",
                *(
                    source_identities[document][0]
                    for document in range(document_count)
                    for _token in range(tokens_per_document)
                ),
            )
        (prefix.parent / relative).write_bytes(payload)
        side_channel_paths[name] = {"path": relative, "dtype": dtype}

    graph_sidecar_paths = {}
    for name in sorted(REQUIRED_GRAPH_SIDECARS):
        kind, dtype, shape_tail = publisher.GRAPH_SIDECAR_SPECS[name]
        if name in {"token_chunk_starts", "token_chunk_ends", "token_chunk_kinds", "token_chunk_dep_levels"}:
            item_count = chunk_capacity
        elif name == "token_domain_edges":
            item_count = edge_capacity
        else:
            item_count = 1 if name in NONZERO_GRAPH_SIDECARS else 0
        offsets_relative = f"{prefix.name}_{name}_offsets.bin"
        data_relative = f"{prefix.name}_{name}_data.bin"
        (prefix.parent / offsets_relative).write_bytes(
            struct.pack(
                f"<{document_count + 1}q",
                0,
                *([item_count] * document_count),
            )
        )
        payload = bytearray(item_count * shape_tail[0] * _DTYPE_SIZES[dtype])
        if name == "token_domain_edges":
            payload[:] = struct.pack(f"<{item_count * 3}i", *((0, 1, 5) * item_count))
        elif name == "token_chunk_starts":
            payload[:] = struct.pack(f"<{item_count}I", *range(item_count))
        elif name == "token_chunk_ends":
            payload[:] = struct.pack(f"<{item_count}I", *range(1, item_count + 1))
        elif name == "token_chunk_kinds":
            payload[:] = struct.pack(f"<{item_count}B", *([1] * item_count))
        (prefix.parent / data_relative).write_bytes(payload)
        graph_sidecar_paths[name] = {
            "kind": kind,
            "coordinate_space": publisher.GRAPH_ROUTE_COORDINATE_SPACES[name],
            "offsets_path": offsets_relative,
            "data_path": data_relative,
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
    (prefix.parent / source_platform["sequence_doc_offsets_path"]).write_bytes(
        struct.pack(f"<{document_count + 1}q", *range(document_count + 1))
    )
    (prefix.parent / source_platform["doc_platform_offsets_path"]).write_bytes(
        struct.pack(f"<{document_count + 1}q", *range(document_count + 1))
    )
    (prefix.parent / source_platform["platform_ids_path"]).write_bytes(
        struct.pack(f"<{document_count}H", *range(1, document_count + 1))
    )
    registry_path = prefix.parent / f"{prefix.name}_source_identity_registry.sqlite"
    registry = sqlite3.connect(registry_path)
    try:
        registry.executescript(
            """
            CREATE TABLE source_identities (
                source_identity_id BLOB PRIMARY KEY,
                canonical_sha256 TEXT NOT NULL,
                source TEXT NOT NULL
            );
            CREATE TABLE sequence_source_identities (
                sequence_index INTEGER NOT NULL,
                source_identity_id BLOB NOT NULL,
                PRIMARY KEY(sequence_index, source_identity_id)
            );
            """
        )
        for sequence_index, (identity_id, digest, source) in enumerate(
            source_identities
        ):
            key = identity_id.to_bytes(8, "big", signed=False)
            registry.execute(
                "INSERT INTO source_identities VALUES (?, ?, ?)",
                (key, digest, source),
            )
            registry.execute(
                "INSERT INTO sequence_source_identities VALUES (?, ?)",
                (sequence_index, key),
            )
        registry.commit()
    finally:
        registry.close()
    tasks = ("causal_lm", "fim", "ast_fim", "ifim", "commit_diff", "pre_to_post")
    source_files = [
        {
            "path": "/frozen/code/1024/code.parquet",
            "size_bytes": 17,
            "sha256": "2" * 64,
            "rows": document_count,
        }
    ]
    source_digest = publisher._artifact_set_sha256(
        [
            {
                "path": record["path"],
                "size": record["size_bytes"],
                "sha256": record["sha256"],
            }
            for record in source_files
        ]
    )
    objective_payload = {
        "schema": "cppmega_pre_materialized_objectives_v1",
        "algorithm": "hamilton_eligibility_bipartite_v1",
        "seed": 17,
        "quota_window_samples": 6,
        "task_order": list(tasks),
        "objective_ids": {task: index + 1 for index, task in enumerate(tasks)},
        "configured_rates": {task: "1/6" for task in tasks},
        "planned_samples": {task: 1 for task in tasks},
        "realized": {
            task: {
                "samples": 1,
                "input_tokens": 4,
                "loss_tokens": 3 if task == "causal_lm" else 2,
            }
            for task in tasks
        },
        "totals": {"samples": 6, "input_tokens": 24, "loss_tokens": 13},
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
            "recipe": stage1_graph_recipe_binding(),
            "relations": list(STAGE1_GRAPH_RELATIONS),
            "eligible_samples": 1,
            "positive_edges": 1,
            "global_weight": "1",
            "indexer_weight": "1/1000",
            "layer_weight": "1",
            "layer_reduction": "sum",
            "bce_weight": "1/10",
            "coverage_weight": "1/20",
            "bias_beta": "1",
            "topk": STAGE1_GRAPH_TOPK,
            "score_formula": "i_neural_plus_beta_s_graph_v1",
            "score_stage": "before_topk",
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
            "loss_mask_alignment": "source_token_predicts_next_v1",
            "length_column": "valid_token_count",
            "objective_column": "objective_kind",
            "document_id_column": "doc_ids",
            "source_document_id_column": "token_source_doc_ids",
        },
        "source_snapshot": {
            "schema": "cppmega_objective_source_snapshot_v1",
            "sequence_length": 1024,
            "file_count": 1,
            "row_count": document_count,
            "files": source_files,
            "sampling": {
                "mode": "deterministic_epoch_shuffle_v1",
                "seed": 17,
                "requested_samples": document_count,
                "full_passes": 1,
                "tail_rows": 0,
                "min_row_reuse": 1,
                "max_row_reuse": 1,
            },
            "artifact_set_sha256": source_digest,
        },
    }
    objective_sha256 = hashlib.sha256(
        json.dumps(
            objective_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    objective_ids = f"{prefix.name}_objective_ids.bin"
    (prefix.parent / objective_ids).write_bytes(bytes(range(1, document_count + 1)))
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "vocab_size": 65536,
                "tokenizer_contract": "megacpp",
                "dtype": "uint16",
                "token_count": token_count,
                "trained_token_count": 13,
                "source_capacity_token_count": document_count * sequence_length,
                "document_count": document_count,
                "graph_sidecar_schema": "cppmega_graph_routes_v2",
                "loss_mask_alignment": "source_token_predicts_next_v1",
                "side_channel_paths": side_channel_paths,
                "graph_sidecar_paths": graph_sidecar_paths,
                "source_platform_sidecar": source_platform,
                "symbol_identity_schema_version": 3,
                "source_identity_registry": {
                    "schema": publisher.SOURCE_IDENTITY_REGISTRY_SCHEMA,
                    "path": registry_path.name,
                    "id_encoding": "uint64_be",
                    "canonical_digest": "sha256",
                    "sequence_count": document_count,
                    "identity_count": document_count,
                    "sequence_identity_reference_count": document_count,
                    "token_foreign_key_sidecar": "token_source_identity_ids",
                },
                publisher.CASE5_RECEIPT_KEY: {
                    "status": "success",
                    "schema": publisher.CASE5_SCHEMA_VERSION,
                    "validated_shard_count": 1,
                    "delimiter_contract_sha256": (
                        publisher.DOMAIN_DELIMITER_CONTRACT_SHA256
                    ),
                    "domain_schema_sha256": publisher.DOMAIN_SCHEMA_SHA256,
                    "tokenizer_contract_sha256": publisher.TOKENIZER_CONTRACT_SHA256,
                    "domain_route_columns": list(publisher.DOMAIN_ROUTE_COLUMNS),
                    "graph_route_columns": list(publisher.GRAPH_ROUTE_COLUMNS),
                    "graph_sidecars_written": True,
                    "source_identity_registry_schema": (
                        publisher.SOURCE_IDENTITY_REGISTRY_SCHEMA
                    ),
                },
                "objective_contract": {
                    "schema": "cppmega_pre_materialized_objectives_v1",
                    "sha256": objective_sha256,
                    "payload": objective_payload,
                    "objective_id_sidecar": {
                        "path": objective_ids,
                        "dtype": "uint8",
                        "document_aligned": True,
                    },
                },
            }
        )
    )
    return json.loads(prefix.with_suffix(".json").read_text())


def _write_tokenizer_dir(path):
    shutil.copytree(Path(__file__).resolve().parents[1] / "data/tokenizer_v2", path)


def _write_test_bundle(root, prefix, tokenizer):
    prefix_manifest = json.loads(prefix.with_suffix(".json").read_text())
    objective = prefix_manifest["objective_contract"]
    provenance = root / "provenance"
    provenance.mkdir()
    objective_path = provenance / "objective_contract.json"
    objective_path.write_text(json.dumps(objective["payload"]), encoding="utf-8")
    objective_file_sha256 = hashlib.sha256(objective_path.read_bytes()).hexdigest()
    artifact_payload = {
        "schema": "cppmega_objective_materialization_artifact_v2",
        "graph_recipe": stage1_graph_recipe_binding(),
        "documents": prefix_manifest["document_count"],
        "objective_contract": {
            "path": objective_path.name,
            "sha256": objective["sha256"],
            "size_bytes": objective_path.stat().st_size,
            "file_sha256": objective_file_sha256,
        },
        "parquet_shards": [
            {"path": "objectives.parquet", "size_bytes": 1, "sha256": "1" * 64}
        ],
        "converter": {
            "split": "all",
            "token_column": "input_ids",
            "length_column": "valid_token_count",
            "side_channels": [],
            "graph_sidecars": [],
            "source_platform_sidecar": "require",
            "loss_mask_alignment": "source_token_predicts_next_v1",
            "graph_relations": list(STAGE1_GRAPH_RELATIONS),
            "graph_pair_mask": "causal_same_document_upstream_v1",
            "chunk_edge_expansion": "cartesian_token_spans_v1",
        },
    }
    objective_artifact_sha256 = hashlib.sha256(
        json.dumps(
            artifact_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    artifact_payload["artifact_set_sha256"] = objective_artifact_sha256
    artifact_path = provenance / "objective_artifact.json"
    artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
    artifact_file_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    prefix_manifest["objective_materialization"] = {
        **artifact_payload,
        "artifact_file_sha256": artifact_file_sha256,
    }
    prefix.with_suffix(".json").write_text(json.dumps(prefix_manifest), encoding="utf-8")

    contracts = root / "contracts"
    contracts.mkdir()
    domain_contract = contracts / "domain_schema_v1.json"
    tokenizer_contract = contracts / "tokenizer_contract_v1.json"
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "data/domain_schema_v1.json",
        domain_contract,
    )
    shutil.copy2(
        Path(__file__).resolve().parents[1]
        / "data/tokenizer_v2/tokenizer_contract_v1.json",
        tokenizer_contract,
    )
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    artifact_set_sha256 = publisher._artifact_set_sha256(records)
    tokenizer_prefix = tokenizer.relative_to(root).as_posix() + "/"
    tokenizer_records = [
        record for record in records if record["path"].startswith(tokenizer_prefix)
    ]
    tokenizer_sha256 = publisher._artifact_set_sha256(tokenizer_records)
    manifest = {
        "schema": "cppmega_megatron_bundle_v1",
        "bundle_id": f"test-bundle-{artifact_set_sha256[:16]}",
        "tokenizer_contract": "megacpp-vocab-65536",
        "vocab_size": 65536,
        "training_contract": "objective_materialized",
        "objective_materialization": {
            "schema": "cppmega_bucketed_objective_materializations_v1",
            "buckets": {
                "1024": {
                    "artifact_path": artifact_path.relative_to(root).as_posix(),
                    "artifact_schema": "cppmega_objective_materialization_artifact_v2",
                    "artifact_set_sha256": objective_artifact_sha256,
                    "artifact_file_sha256": artifact_file_sha256,
                    "contract_path": objective_path.relative_to(root).as_posix(),
                    "contract_schema": "cppmega_pre_materialized_objectives_v1",
                    "contract_sha256": objective["sha256"],
                    "contract_file_sha256": objective_file_sha256,
                    "source_snapshot": {
                        key: objective["payload"]["source_snapshot"][key]
                        for key in (
                            "schema",
                            "artifact_set_sha256",
                            "file_count",
                            "row_count",
                            "sampling",
                        )
                    },
                }
            },
        },
        "buckets": [1024],
        "tokenizer": {
            "path": tokenizer.relative_to(root).as_posix(),
            "contract": "megacpp-vocab-65536",
            "vocab_size": 65536,
            "files": tokenizer_records,
            "artifact_set_sha256": tokenizer_sha256,
        },
        "data_contracts": {
            "domain_schema": {
                "path": domain_contract.relative_to(root).as_posix(),
                "size": domain_contract.stat().st_size,
                "sha256": hashlib.sha256(domain_contract.read_bytes()).hexdigest(),
            },
            "tokenizer_contract": {
                "path": tokenizer_contract.relative_to(root).as_posix(),
                "size": tokenizer_contract.stat().st_size,
                "sha256": hashlib.sha256(tokenizer_contract.read_bytes()).hexdigest(),
            },
        },
        "bucket_results": [
            {
                "bucket": 1024,
                "prefix": prefix.relative_to(root).as_posix(),
                "manifest": prefix_manifest,
            }
        ],
        "artifact_count": len(records),
        "artifact_bytes": sum(record["size"] for record in records),
        "artifact_set_sha256": artifact_set_sha256,
        "artifacts": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_first_public_ip_accepts_nebius_cidr_status_address():
    obj = {
        "status": {
            "network_interfaces": [
                {
                    "ip_address": {"address": "10.0.0.17/32"},
                    "public_ip_address": {"address": "89.169.108.186/32"},
                }
            ]
        }
    }

    assert first_public_ip(obj) == "89.169.108.186"


def test_first_public_ip_ignores_private_only_addresses():
    obj = {"status": {"network_interfaces": [{"ip_address": {"address": "10.0.0.17/32"}}]}}

    assert first_public_ip(obj) is None


def test_instance_deletion_requires_complete_artifact_retrieval():
    assert instance_delete_allowed(
        keep_instance=False,
        retrieval_succeeded=True,
    )
    assert not instance_delete_allowed(
        keep_instance=False,
        retrieval_succeeded=False,
    )
    assert not instance_delete_allowed(
        keep_instance=True,
        retrieval_succeeded=True,
    )


@pytest.mark.parametrize(
    ("save_checkpoint", "scp_returncodes"),
    [(False, [1]), (True, [0, 1])],
)
def test_failed_results_or_checkpoint_scp_preserves_instance(
    tmp_path,
    monkeypatch,
    save_checkpoint,
    scp_returncodes,
):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    prefix = bundle_root / "sample_train"
    tokenizer = bundle_root / "tok"
    _write_valid_sidecar_prefix(prefix)
    _write_tokenizer_dir(tokenizer)
    manifest = _write_test_bundle(bundle_root, prefix, tokenizer)
    pubkey = tmp_path / "id_ed25519.pub"
    pubkey.write_text("ssh-ed25519 TESTKEY codex\n", encoding="utf-8")
    commands = []
    scp_codes = iter(scp_returncodes)

    monkeypatch.setattr(sweep_module, "ROOT", tmp_path)
    monkeypatch.setattr(sweep_module, "make_overlay_tar", lambda _path: None)
    monkeypatch.setattr(
        sweep_module,
        "make_bundle_tar",
        lambda *_args, **_kwargs: (
            [manifest["bucket_results"][0]["prefix"]],
            manifest["tokenizer"]["path"],
        ),
    )
    monkeypatch.setattr(sweep_module, "create_instance", lambda *_args: "instance-1")
    monkeypatch.setattr(sweep_module, "wait_for_ip", lambda *_args: "203.0.113.10")
    monkeypatch.setattr(sweep_module, "wait_for_ssh", lambda *_args: None)
    monkeypatch.setattr(sweep_module, "stream_tar_to_remote", lambda *_args: None)
    monkeypatch.setattr(sweep_module, "ssh", lambda *_args, **_kwargs: None)

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        code = next(scp_codes) if cmd[0] == "scp" else 0
        return subprocess.CompletedProcess(cmd, code)

    monkeypatch.setattr(sweep_module, "run", fake_run)
    argv = [
        "--bundle-root",
        str(bundle_root),
        "--sidecar-prefix",
        str(prefix),
        "--batch-sizes",
        "1",
        "--train-iters",
        "1",
        "--ssh-pubkey",
        str(pubkey),
        "--ssh-key",
        str(tmp_path / "id_ed25519"),
        "--ssh-host-key",
        (
            "ssh-ed25519 "
            "AAAAC3NzaC1lZDI1NTE5AAAAIJRwravCVfVsFZfdgfvC/OlW0K7vrJ7pBjl5p86YKSSs"
        ),
        "--ssh-host-key-fingerprint",
        "SHA256:xGOQHYDUpAKZPiHLlYNYp01FiayrndE1tGC9wBoA+xw",
        "--instance-name",
        "retrieval-gate-test",
        "--no-ghcr-auth",
    ]
    if save_checkpoint:
        argv.append("--save-checkpoint")

    with pytest.raises(RuntimeError, match="artifact retrieval failed"):
        main(argv)

    assert not any(cmd[:4] == ["nebius", "compute", "instance", "delete"] for cmd in commands)


def test_remote_script_installs_docker_ce_not_conflicting_ubuntu_docker_io():
    script = remote_run_script(
        [256], 1, DEFAULT_DOCKER_IMAGE, graph_capacity=_TEST_GRAPH_CAPACITY
    )

    assert "docker-ce docker-ce-cli containerd.io" in script
    assert "apt-get install -y docker.io" not in script


def test_remote_script_logs_into_ghcr_from_token_file_without_literal_secret():
    script = remote_run_script(
        [256], 1, DEFAULT_DOCKER_IMAGE, graph_capacity=_TEST_GRAPH_CAPACITY
    )

    assert "/data/cppmega_auth/ghcr_token" in script
    assert "docker login ghcr.io" in script
    assert "--password-stdin" in script
    assert "rm -f /data/cppmega_auth/ghcr_token" in script
    assert "SECRET" not in script


def test_remote_script_runs_upstream_megatron_pretrain_from_real_tree():
    script = remote_run_script(
        [256], 1, DEFAULT_DOCKER_IMAGE, graph_capacity=_TEST_GRAPH_CAPACITY
    )

    assert "_inner = '/opt/megatron-lm/pretrain_mamba.py'" in script
    assert "pretrain_mamba_inner.py" not in script


def test_remote_script_does_not_shadow_upstream_model_provider():
    script = remote_run_script(
        [256], 1, DEFAULT_DOCKER_IMAGE, graph_capacity=_TEST_GRAPH_CAPACITY
    )

    assert "mamba_builders.py" in script
    assert "hybrid_builders.py" in script
    assert "cppmega_mamba_builder as hybrid_builder" in script
    assert "model_provider.py" not in script


def test_overlay_packages_complete_cppmega_and_contract_closure(tmp_path):
    output = tmp_path / "overlay.tgz"

    make_overlay_tar(output)

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "cppmega/__init__.py" in names
    assert "cppmega/recipes/nam56r_launch.py" in names
    assert "cppmega/recipes/nam56r_megatron.py" in names
    assert "cppmega/megatron/domain_route_contract.py" in names
    assert "cppmega/megatron/graph_objective_loss.py" in names
    assert "cppmega/megatron/structure_dataset_patch.py" in names
    assert "cppmega/receipt_binding.py" in names
    assert "scripts/h200_megatron_preflight.py" in names
    assert "scripts/data/publish_megatron_bundle_to_nebius_s3.py" in names
    assert "data/domain_schema_v1.json" in names
    assert "data/tokenizer_v2/tokenizer_contract_v1.json" in names
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert OVERLAY_PATHS[0] == "cppmega"

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    subprocess.run(
        ["tar", "-xzf", str(output), "-C", str(extracted)],
        check=True,
    )
    import_result = subprocess.run(
        [sys.executable, "-c", "import scripts.h200_megatron_preflight"],
        cwd=extracted,
        env={**os.environ, "PYTHONPATH": str(extracted)},
        capture_output=True,
        text=True,
    )
    assert import_result.returncode == 0, import_result.stderr


def test_docker_image_must_be_an_immutable_digest():
    assert validate_docker_image_digest(DEFAULT_DOCKER_IMAGE) == DEFAULT_DOCKER_IMAGE
    with pytest.raises(ValueError, match="mutable tags are rejected"):
        validate_docker_image_digest("ghcr.io/datasunriseou/cppmega:latest")
    with pytest.raises(ValueError, match="immutable image digest"):
        remote_run_script(
            [1],
            1,
            "ghcr.io/datasunriseou/cppmega:latest",
            graph_capacity=_TEST_GRAPH_CAPACITY,
        )


def test_nebius_resource_ids_are_strictly_bound():
    assert validate_nebius_resource_id(
        "computeimage-e00hbfk8kmf3w3prch", name="--image-id"
    ) == "computeimage-e00hbfk8kmf3w3prch"
    with pytest.raises(ValueError, match="resource identifier"):
        validate_nebius_resource_id("../escape", name="--image-id")
    with pytest.raises(ValueError, match="resource identifier"):
        validate_nebius_resource_id("ab", name="--image-id")


def test_remote_script_enables_graph_routes_and_uses_selected_data_prefix():
    script = remote_run_script(
        [256],
        1,
        DEFAULT_DOCKER_IMAGE,
        data_prefix_name="cppmega_1024_current_mix_graph_train",
        graph_capacity=_TEST_GRAPH_CAPACITY,
    )

    assert 'export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-1}"' in script
    assert 'export CPPMEGA_GRAPH_ROUTES_ENABLED="${CPPMEGA_GRAPH_ROUTES_ENABLED:-1}"' in script
    assert "export CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS=0" in script
    assert 'export CPPMEGA_DSA_PATCH_ENABLED="1"' in script
    assert "--enable-dsa-patch" in script
    assert "if os.environ.get('CPPMEGA_DSA_PATCH_ENABLED', '0') == '1'" in script
    assert "apply_dsa_indexer_fused_patch()" in script
    assert "export CPPMEGA_DSA_GRAPH_AUX_ENABLED=1" in script
    assert "export CPPMEGA_DSA_GRAPH_AUX_WEIGHT=1" in script
    assert "export CPPMEGA_DSA_INDEXER_LOSS_COEFF=0.001" in script
    assert "export CPPMEGA_DSA_SKIP_INDEXER_LOSS=0" in script
    assert "--experimental-attention-variant dsa" in script
    assert "--multi-latent-attention" in script
    assert "--dsa-indexer-loss-coeff 0.001" in script
    assert "cppmega.megatron.nam56r_full_spec" in script
    assert "build_cppmega_nam56r_full_stack_spec" in script
    assert "CPPMEGA_H200_DSA_GRAPH_RECEIPTS=1" in script
    assert "expected_dsa_coefficient=0.001" in script
    assert "GQA_ARGS" not in script
    assert "apply_graph_route_attention_bias_patch()" in script
    assert "1024:cppmega_1024_current_mix_graph_train" in script
    assert 'DATA_PREFIX="$CPPMEGA_BUNDLE_ROOT/${DATA_PREFIX_NAME}"' in script
    assert 'DATA_ARGS=(--data-path 1.0 \\"\\$DATA_PREFIX\\")' in script
    assert 'ATTN_ARGS=(--attention-backend \\"\\$CPPMEGA_ATTN_BACKEND\\")' in script
    assert "CPPMEGA_USE_FLASH_ATTN" in script
    assert "NVTE_DEBUG_LEVEL" in script
    assert "--attention-backend flash" not in script
    assert "--fp8-recipe off" in script
    assert "FP8_ARGS=()" in script
    assert "failed to CUDA calloc" in script
    assert "--eval-iters 1" in script
    assert "--eval-iters 0" not in script
    assert "--eval-interval 1" in script
    assert "--no-check-for-nan-in-loss-and-grad" not in script
    assert "write_graph_capacity_receipt" in script
    assert "write_training_loss_receipt" in script
    assert "--cross-entropy-fusion-impl te" in script
    assert "--cross-entropy-fusion-impl linear" not in script

    with pytest.raises(ValueError, match="requires the fused DSA patch"):
        remote_run_script(
            [256],
            1,
            DEFAULT_DOCKER_IMAGE,
            graph_capacity=_TEST_GRAPH_CAPACITY,
            enable_dsa_patch=False,
        )


def test_remote_script_runs_fail_closed_h200_preflight_before_sweep():
    script = remote_run_script(
        [64],
        3,
        DEFAULT_DOCKER_IMAGE,
        data_prefix_name="cppmega_1024_current_mix_graph_train",
        graph_capacity=_TEST_GRAPH_CAPACITY,
    )

    preflight = "python /opt/cppmega/scripts/h200_megatron_preflight.py"
    assert preflight in script
    assert '--bundle-root "$CPPMEGA_BUNDLE_ROOT"' in script
    assert "--data-prefix \"$DATA_PREFIX\"" in script
    assert '--tokenizer-model "$CPPMEGA_TOKENIZER_MODEL"' in script
    assert "--run-id nebius-h200-sweep" in script
    assert "--output /data/cppmega_h200_results/h200_preflight.json" in script
    assert "CPPMEGA_H200_PREFLIGHT_STATUS=PASS" in script
    assert script.index(preflight) < script.index('for SPEC in "${TEST_SPECS[@]}"')


def test_remote_script_can_sweep_multiple_seq_lengths_with_separate_prefixes():
    script = remote_run_script(
        [64, 128],
        100,
        DEFAULT_DOCKER_IMAGE,
        seq_data_prefixes=[
            (1024, "cppmega_h200_100step_seq1024_graph_train", 11, 7),
            (2048, "cppmega_h200_100step_seq2048_graph_train", 13, 9),
            (4096, "cppmega_h200_100step_seq4096_graph_train", 17, 12),
        ],
    )

    assert "1024:cppmega_h200_100step_seq1024_graph_train" in script
    assert "2048:cppmega_h200_100step_seq2048_graph_train" in script
    assert "4096:cppmega_h200_100step_seq4096_graph_train" in script
    assert "1024:cppmega_h200_100step_seq1024_graph_train:11:7" in script
    assert "4096:cppmega_h200_100step_seq4096_graph_train:17:12" in script
    assert "--seq-length ${SEQ}" in script
    assert "--max-position-embeddings ${SEQ}" in script
    assert "seq_${SEQ}_bs_${BS}.log" in script
    assert "CPPMEGA_BATCH_RESULT seq=${SEQ} batch=${BS}" in script


def test_remote_script_can_enable_tensorwise_fp8_flags():
    script = remote_run_script(
        [64],
        100,
        DEFAULT_DOCKER_IMAGE,
        graph_capacity=_TEST_GRAPH_CAPACITY,
        fp8_recipe="tensorwise",
    )

    assert "--fp8-recipe tensorwise" in script
    assert "--fp8-format \\\"\\$CPPMEGA_FP8_FORMAT\\\"" in script
    assert "--fp8-amax-history-len 16" in script
    assert "--fp8-amax-compute-algo max" in script
    assert "apply_te_checkpoint_kwarg_patch()" in script
    assert "RECOMPUTE_ARGS=(--recompute-granularity selective --recompute-modules mlp)" in script
    assert '\\\"\\${RECOMPUTE_ARGS[@]}\\\"' in script
    assert "RECOMPUTE_ARGS=()" not in script
    assert "FAIL_TE_CLEANUP_SIGSEGV" in script
    assert "OK_TE_CLEANUP_SIGSEGV" not in script
    assert "transformer_engine::rtc::Kernel::~Kernel" in script
    assert "status=OOM" in script


def test_remote_wrapper_destroys_process_group_before_te_cleanup():
    script = remote_run_script(
        [64], 1, DEFAULT_DOCKER_IMAGE, graph_capacity=_TEST_GRAPH_CAPACITY
    )

    assert "def _cppmega_distributed_shutdown()" in script
    assert "dist.destroy_process_group()" in script
    assert "torch.cuda.synchronize()" in script
    assert "CPPMEGA_DISTRIBUTED_SHUTDOWN_ERROR" in script


def test_remote_script_can_disable_te_nvrtc_for_cleanup_crash_probe():
    script = remote_run_script(
        [64],
        1,
        DEFAULT_DOCKER_IMAGE,
        graph_capacity=_TEST_GRAPH_CAPACITY,
        disable_nvrtc=True,
    )

    assert 'export NVTE_DISABLE_NVRTC="1"' in script


def test_remote_script_can_save_model_only_checkpoint():
    script = remote_run_script(
        [192],
        5000,
        DEFAULT_DOCKER_IMAGE,
        graph_capacity=_TEST_GRAPH_CAPACITY,
        fp8_recipe="tensorwise",
        save_checkpoint=True,
        save_interval=1000,
        save_model_only=True,
    )

    assert "CHECKPOINT_ROOT=/data/cppmega_h200_checkpoints/seq_${SEQ}_bs_${BS}" in script
    assert "CHECKPOINT_ARGS+=(--save \\$CHECKPOINT_ROOT --save-interval 1000)" in script
    assert "CHECKPOINT_ARGS+=(--no-save-optim --no-save-rng)" in script
    assert '--save-interval 50000000' not in script


def test_remote_script_with_checkpoint_is_bash_parseable(tmp_path):
    script = remote_run_script(
        [192],
        5000,
        DEFAULT_DOCKER_IMAGE,
        graph_capacity=_TEST_GRAPH_CAPACITY,
        save_checkpoint=True,
        save_interval=1000,
    )
    script_path = tmp_path / "run_cppmega_h200_sweep.sh"
    script_path.write_text(script)

    assert script.startswith("#!/usr/bin/env bash\n")
    assert "\nINNER\n\nsudo docker run" in script
    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_remote_script_disables_checkpointing_by_default():
    script = remote_run_script(
        [64], 100, DEFAULT_DOCKER_IMAGE, graph_capacity=_TEST_GRAPH_CAPACITY
    )

    assert "--save-interval" not in script
    assert "--no-save-optim" not in script


def test_remote_script_can_load_model_only_checkpoint():
    script = remote_run_script(
        [192],
        1,
        DEFAULT_DOCKER_IMAGE,
        graph_capacity=_TEST_GRAPH_CAPACITY,
        load_checkpoint_remote="/data/cppmega_load_checkpoint",
    )

    assert "CHECKPOINT_ARGS+=(--load /data/cppmega_load_checkpoint)" in script
    assert "CHECKPOINT_ARGS+=(--no-load-optim --no-load-rng)" in script
    assert "--no-save-optim" not in script


def test_remote_script_can_load_full_checkpoint_state():
    script = remote_run_script(
        [192],
        1,
        DEFAULT_DOCKER_IMAGE,
        graph_capacity=_TEST_GRAPH_CAPACITY,
        load_checkpoint_remote="/data/cppmega_load_checkpoint",
        load_model_only=False,
    )

    assert "CHECKPOINT_ARGS+=(--load /data/cppmega_load_checkpoint)" in script
    assert "--no-load-optim" not in script
    assert "--no-load-rng" not in script


def test_make_checkpoint_tar_requires_megatron_checkpoint_root(tmp_path):
    checkpoint = tmp_path / "not_checkpoint"
    checkpoint.mkdir()

    try:
        make_checkpoint_tar(checkpoint, tmp_path / "ckpt.tgz")
    except FileNotFoundError as exc:
        assert "latest_checkpointed_iteration.txt" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_make_checkpoint_tar_archives_checkpoint_contents(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("1000\n")
    iter_dir = checkpoint / "iter_0001000"
    iter_dir.mkdir()
    (iter_dir / "metadata.json").write_text("{}\n")
    out = tmp_path / "ckpt.tgz"

    make_checkpoint_tar(checkpoint, out)

    import tarfile

    with tarfile.open(out, "r:gz") as tf:
        names = set(tf.getnames())
    assert "./latest_checkpointed_iteration.txt" in names
    assert "./iter_0001000/metadata.json" in names


def test_fp8_tensorwise_dry_run_disables_nvrtc_by_default(tmp_path, capsys):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    prefix = bundle_root / "sample_train"
    tokenizer = bundle_root / "tok"
    _write_valid_sidecar_prefix(prefix)
    _write_tokenizer_dir(tokenizer)
    _write_test_bundle(bundle_root, prefix, tokenizer)
    pubkey = tmp_path / "id_ed25519.pub"
    pubkey.write_text("ssh-ed25519 TESTKEY codex\n")
    plan_script = tmp_path / "leader-plan.sh"
    rc = main(
        [
            "--dry-run",
            "--plan-script",
            str(plan_script),
            "--bundle-root",
            str(bundle_root),
            "--sidecar-prefix",
            str(prefix),
            "--fp8-recipe",
            "tensorwise",
            "--batch-sizes",
            "64",
            "--train-iters",
            "1",
            "--ssh-pubkey",
            str(pubkey),
        ]
    )

    assert rc == 0
    assert 'export NVTE_DISABLE_NVRTC="1"' in capsys.readouterr().out
    assert plan_script.stat().st_mode & 0o777 == 0o700
    plan = plan_script.read_text(encoding="utf-8")
    assert plan.startswith("#!/usr/bin/env bash\n")
    assert f"sudo docker pull {DEFAULT_DOCKER_IMAGE}" in plan


def test_fp8_tensorwise_can_keep_nvrtc_enabled_for_perf_probe(tmp_path, capsys):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    prefix = bundle_root / "sample_train"
    tokenizer = bundle_root / "tok"
    _write_valid_sidecar_prefix(prefix)
    _write_tokenizer_dir(tokenizer)
    _write_test_bundle(bundle_root, prefix, tokenizer)
    pubkey = tmp_path / "id_ed25519.pub"
    pubkey.write_text("ssh-ed25519 TESTKEY codex\n")
    rc = main(
        [
            "--dry-run",
            "--bundle-root",
            str(bundle_root),
            "--sidecar-prefix",
            str(prefix),
            "--fp8-recipe",
            "tensorwise",
            "--enable-nvrtc",
            "--batch-sizes",
            "64",
            "--train-iters",
            "1",
            "--ssh-pubkey",
            str(pubkey),
        ]
    )

    assert rc == 0
    assert 'export NVTE_DISABLE_NVRTC="0"' in capsys.readouterr().out


def test_make_ghcr_auth_tar_uses_token_file(tmp_path):
    token = tmp_path / "token.txt"
    token.write_text("SECRET_TOKEN")
    out = tmp_path / "auth.tgz"

    class Args:
        no_ghcr_auth = False
        ghcr_user = "datasunrise"
        ghcr_token_file = token

    assert make_ghcr_auth_tar(Args, out) is True

    import tarfile

    with tarfile.open(out, "r:gz") as tf:
        names = sorted(tf.getnames())
        assert names == ["cppmega_auth/ghcr_token", "cppmega_auth/ghcr_user"]
        assert tf.extractfile("cppmega_auth/ghcr_user").read().decode() == "datasunrise"
        assert tf.extractfile("cppmega_auth/ghcr_token").read().decode() == "SECRET_TOKEN"


def test_make_sidecar_tar_includes_graph_sidecars(tmp_path):
    prefix = tmp_path / "sample_train"
    tokenizer = tmp_path / "tok"
    _write_tokenizer_dir(tokenizer)
    _write_valid_sidecar_prefix(prefix)
    out = tmp_path / "sidecar.tgz"

    from scripts.nebius_h200_megatron_cpp_world_sweep import make_sidecar_tar

    make_sidecar_tar(prefix, tokenizer, out)

    import tarfile

    with tarfile.open(out, "r:gz") as tf:
        names = set(tf.getnames())
    assert "cppmega_sidecar/sample_train_token_structure_ids.bin" in names
    assert "cppmega_sidecar/sample_train_token_call_edges_offsets.bin" in names
    assert "cppmega_sidecar/sample_train_token_call_edges_data.bin" in names
    assert "cppmega_sidecar/sample_train_source_platform_ids.bin" in names
    assert (
        "cppmega_sidecar/sample_train_source_platform_sequence_doc_offsets.bin"
        in names
    )
    assert "cpp_tokenizer_hf/tokenizer.json" in names


def test_make_multi_sidecar_tar_includes_each_prefix_once(tmp_path):
    tokenizer = tmp_path / "tok"
    _write_tokenizer_dir(tokenizer)

    prefixes = [tmp_path / "seq1024_train", tmp_path / "seq2048_train"]
    for prefix in prefixes:
        _write_valid_sidecar_prefix(prefix)

    out = tmp_path / "sidecars.tgz"
    make_multi_sidecar_tar(prefixes, tokenizer, out)

    import tarfile

    with tarfile.open(out, "r:gz") as tf:
        names = set(tf.getnames())
    assert "cppmega_sidecar/seq1024_train.bin" in names
    assert "cppmega_sidecar/seq2048_train.bin" in names
    assert "cppmega_sidecar/seq1024_train_token_structure_ids.bin" in names
    assert "cppmega_sidecar/seq2048_train_token_structure_ids.bin" in names
    assert "cpp_tokenizer_hf/tokenizer.json" in names


def test_make_bundle_tar_preserves_manifest_bound_objective_and_layout(tmp_path):
    prefix = tmp_path / "sample_train"
    tokenizer = tmp_path / "tok"
    _write_valid_sidecar_prefix(prefix)
    _write_tokenizer_dir(tokenizer)
    manifest = _write_test_bundle(tmp_path, prefix, tokenizer)
    output = tmp_path / "bundle.tgz"

    prefixes, tokenizer_relative = make_bundle_tar(tmp_path, [prefix], output)

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    objective_path = manifest["bucket_results"][0]["manifest"]["objective_contract"][
        "objective_id_sidecar"
    ]["path"]
    assert prefixes == [manifest["bucket_results"][0]["prefix"]]
    assert tokenizer_relative == manifest["tokenizer"]["path"]
    assert "cppmega_bundle/manifest.json" in names
    assert f"cppmega_bundle/{objective_path}" in names
    assert f"cppmega_bundle/{prefix.name}.json" in names


def test_sidecar_preflight_rejects_missing_graph_sidecars(tmp_path):
    prefix = tmp_path / "sample_train"
    manifest = _write_valid_sidecar_prefix(prefix)
    manifest["graph_sidecar_paths"].pop("token_chunk_ends")
    prefix.with_suffix(".json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="graph sidecar key set.*missing"):
        _assert_prefix_contract(prefix)


def test_sidecar_preflight_rejects_zero_structure_values(tmp_path):
    prefix = tmp_path / "sample_train"
    manifest = _write_valid_sidecar_prefix(prefix)
    structure = prefix.parent / manifest["side_channel_paths"]["token_structure_ids"]["path"]
    structure.write_bytes(b"\x00" * structure.stat().st_size)

    with pytest.raises(ValueError, match="token_structure_ids.*nonzero"):
        _assert_prefix_contract(prefix)


def test_graph_capacities_are_derived_from_actual_csr_offsets(tmp_path):
    prefix = tmp_path / "sample_train"
    _write_valid_sidecar_prefix(
        prefix,
        edge_capacity=300,
        chunk_capacity=4,
    )

    receipt = derive_graph_capacity_receipt(prefix, sequence_length=1024)

    assert receipt["graph_max_edges"] == 300
    assert receipt["graph_max_chunks"] == 4
    assert receipt["sidecars"]["token_domain_edges"][
        "max_items_per_document"
    ] == 300
    assert receipt["sidecars"]["token_chunk_starts"][
        "max_items_per_document"
    ] == 4
    assert len(receipt["sidecars"]["token_domain_edges"]["offsets_sha256"]) == 64


def test_graph_capacity_derivation_rejects_manifest_sequence_drift(tmp_path):
    prefix = tmp_path / "sample_train"
    _write_valid_sidecar_prefix(prefix, sequence_length=1024)

    with pytest.raises(RuntimeError, match="fixed-row source capacity"):
        derive_graph_capacity_receipt(prefix, sequence_length=2048)


def test_sidecar_packaging_rejects_tokenizer_vocab_drift(tmp_path):
    prefix = tmp_path / "sample_train"
    _write_valid_sidecar_prefix(prefix)
    tokenizer = tmp_path / "tok"
    _write_tokenizer_dir(tokenizer)
    tokenizer_path = tokenizer / "tokenizer.json"
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    payload["model"]["vocab"].pop(next(iter(payload["model"]["vocab"])))
    tokenizer_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer vocab size"):
        make_multi_sidecar_tar([prefix], tokenizer, tmp_path / "sidecars.tgz")


def test_h200_preflight_real_local_dry_run_writes_bound_commands(tmp_path):
    prefix = tmp_path / "sample_train"
    tokenizer = tmp_path / "tok"
    _write_valid_sidecar_prefix(prefix)
    _write_tokenizer_dir(tokenizer)
    bundle = _write_test_bundle(tmp_path, prefix, tokenizer)
    output = tmp_path / "h200-preflight.json"
    checkpoint = tmp_path / "checkpoint"

    assert (
        h200_preflight_main(
            [
                "--bundle-root",
                str(tmp_path),
                "--data-prefix",
                str(prefix),
                "--tokenizer-model",
                str(tokenizer),
                "--run-id",
                "local-dry-run",
                "--sequence-length",
                "1024",
                "--checkpoint-root",
                str(checkpoint),
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema"] == "cppmega_h200_megatron_preflight_v1"
    assert receipt["status"] == "dry_run"
    assert receipt["bundle"]["bundle_id"] == bundle["bundle_id"]
    assert receipt["binding"]["run_id"] == "local-dry-run"
    assert receipt["data"]["manifest"]["graph_sidecar_schema"] == "cppmega_graph_routes_v2"
    assert receipt["data"]["graph_capacity"]["schema"] == "cppmega_graph_capacity_v1"
    assert receipt["data"]["graph_capacity"]["graph_max_edges"] == 1
    assert receipt["data"]["graph_capacity"]["graph_max_chunks"] == 1
    assert Path(receipt["data"]["graph_capacity_receipt"]).is_file()
    assert receipt["checkpoint"]["full_optimizer_and_rng_state"] is True
    assert receipt["commands"]["save"][receipt["commands"]["save"].index("--train-iters") + 1] == "1"
    assert receipt["commands"]["restore"][receipt["commands"]["restore"].index("--train-iters") + 1] == "2"
    assert "--load" in receipt["commands"]["restore"]
    assert "--no-check-for-nan-in-loss-and-grad" not in receipt["commands"]["save"]
    assert receipt["commands"]["save"][
        receipt["commands"]["save"].index("--eval-interval") + 1
    ] == "1"
    assert receipt["config"]["enable_dsa_patch"] is True
    save_command = receipt["commands"]["save"]
    assert save_command[save_command.index("--experimental-attention-variant") + 1] == "dsa"
    assert "--multi-latent-attention" in save_command
    assert save_command.count("--dsa-indexer-loss-coeff") == 1
    assert save_command[save_command.index("--dsa-indexer-loss-coeff") + 1] == "0.001"
    assert "--group-query-attention" not in save_command
    assert not checkpoint.exists()


def test_docker_auth_returns_none_when_config_absent(tmp_path, monkeypatch):
    # Genuine absence: no ~/.docker/config.json at all -> None (a normal "no creds").
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _docker_auth_from_config("ghcr.io") is None


def test_docker_auth_returns_none_when_config_has_no_creds(tmp_path, monkeypatch):
    # Config exists and parses but configures nothing for the host -> None.
    monkeypatch.setenv("HOME", str(tmp_path))
    docker_dir = tmp_path / ".docker"
    docker_dir.mkdir()
    (docker_dir / "config.json").write_text('{"auths": {}}')
    assert _docker_auth_from_config("ghcr.io") is None


def test_docker_auth_raises_when_config_present_but_corrupt(tmp_path, monkeypatch):
    # Configured-but-broken: config.json exists but is unparseable -> raise, do not
    # silently return None (which the caller would treat as "no creds configured").
    monkeypatch.setenv("HOME", str(tmp_path))
    docker_dir = tmp_path / ".docker"
    docker_dir.mkdir()
    (docker_dir / "config.json").write_text("{ this is not valid json ")
    with pytest.raises(RuntimeError, match="configured but broken"):
        _docker_auth_from_config("ghcr.io")
