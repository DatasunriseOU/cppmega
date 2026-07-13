import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from scripts.nebius_h200_megatron_cpp_world_sweep import (
    NONZERO_GRAPH_SIDECARS,
    OVERLAY_PATHS,
    REQUIRED_GRAPH_SIDECARS,
    REQUIRED_TOKEN_SIDECARS,
    _assert_prefix_contract,
    _docker_auth_from_config,
    first_public_ip,
    main,
    make_checkpoint_tar,
    make_ghcr_auth_tar,
    make_multi_sidecar_tar,
    remote_run_script,
)
from scripts.h200_megatron_preflight import main as h200_preflight_main


_DTYPE_SIZES = {"uint8": 1, "uint16": 2, "uint32": 4, "int32": 4}


def _write_valid_sidecar_prefix(prefix):
    tokens_per_document = 3
    document_count = 6
    token_count = tokens_per_document * document_count
    prefix.with_suffix(".bin").write_bytes(
        struct.pack(f"<{token_count}H", *range(1, token_count + 1))
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
    required_token_sidecars = set(REQUIRED_TOKEN_SIDECARS) | {
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
        relative = f"{prefix.name}_{name}.bin"
        payload = bytearray(token_count * _DTYPE_SIZES[dtype])
        if name == "token_structure_ids":
            payload[0] = 1
        if name == "loss_mask":
            payload[:] = b"\x01" * token_count
        if name == "token_source_doc_ids":
            payload[:] = struct.pack(
                f"<{token_count}I",
                *(
                    document + 1
                    for document in range(document_count)
                    for _token in range(tokens_per_document)
                ),
            )
        (prefix.parent / relative).write_bytes(payload)
        side_channel_paths[name] = {"path": relative, "dtype": dtype}

    graph_sidecar_paths = {}
    for name in sorted(REQUIRED_GRAPH_SIDECARS):
        if name in {"token_call_edges", "token_type_edges"}:
            kind, dtype, shape_tail = "edge_pairs", "int32", [2]
        elif name.endswith("_edges"):
            kind, dtype, shape_tail = "edge_triples", "int32", [3]
        else:
            kind = "ragged_1d"
            dtype = "uint32" if name in {"token_chunk_starts", "token_chunk_ends"} else "uint16"
            shape_tail = [1]
        item_count = (
            1
            if name in NONZERO_GRAPH_SIDECARS or name == "token_domain_edges"
            else 0
        )
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
            payload[:] = struct.pack("<3i", 0, 1, 5)
        elif name == "token_chunk_ends":
            payload[:] = struct.pack("<I", tokens_per_document)
        elif name == "token_chunk_kinds":
            payload[:] = struct.pack("<H", 1)
        (prefix.parent / data_relative).write_bytes(payload)
        graph_sidecar_paths[name] = {
            "kind": kind,
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
    tasks = ("causal_lm", "fim", "ast_fim", "ifim", "commit_diff", "pre_to_post")
    objective_payload = {
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
                "document_count": document_count,
                "graph_sidecar_schema": "cppmega_graph_routes_v2",
                "side_channel_paths": side_channel_paths,
                "graph_sidecar_paths": graph_sidecar_paths,
                "source_platform_sidecar": source_platform,
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
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    artifact_set_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tokenizer_prefix = tokenizer.relative_to(root).as_posix() + "/"
    tokenizer_records = [
        record for record in records if record["path"].startswith(tokenizer_prefix)
    ]
    tokenizer_sha256 = hashlib.sha256(
        json.dumps(tokenizer_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema": "cppmega_megatron_bundle_v1",
        "bundle_id": f"test-bundle-{artifact_set_sha256[:16]}",
        "tokenizer_contract": "megacpp-vocab-65536",
        "vocab_size": 65536,
        "training_contract": "objective_materialized",
        "objective_materialization": {
            "path": objective_path.relative_to(root).as_posix(),
            "schema": "cppmega_pre_materialized_objectives_v1",
            "sha256": objective["sha256"],
            "file_sha256": hashlib.sha256(objective_path.read_bytes()).hexdigest(),
        },
        "buckets": [1024],
        "tokenizer": {
            "path": tokenizer.relative_to(root).as_posix(),
            "contract": "megacpp-vocab-65536",
            "vocab_size": 65536,
            "files": tokenizer_records,
            "artifact_set_sha256": tokenizer_sha256,
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


def test_remote_script_installs_docker_ce_not_conflicting_ubuntu_docker_io():
    script = remote_run_script([256], 1, "ghcr.io/datasunriseou/cppmega:latest")

    assert "docker-ce docker-ce-cli containerd.io" in script
    assert "apt-get install -y docker.io" not in script


def test_remote_script_logs_into_ghcr_from_token_file_without_literal_secret():
    script = remote_run_script([256], 1, "ghcr.io/datasunriseou/cppmega:latest")

    assert "/data/cppmega_auth/ghcr_token" in script
    assert "docker login ghcr.io" in script
    assert "--password-stdin" in script
    assert "rm -f /data/cppmega_auth/ghcr_token" in script
    assert "SECRET" not in script


def test_remote_script_runs_upstream_megatron_pretrain_from_real_tree():
    script = remote_run_script([256], 1, "ghcr.io/datasunriseou/cppmega:latest")

    assert "_inner = '/opt/megatron-lm/pretrain_mamba.py'" in script
    assert "pretrain_mamba_inner.py" not in script


def test_remote_script_does_not_shadow_upstream_model_provider():
    script = remote_run_script([256], 1, "ghcr.io/datasunriseou/cppmega:latest")

    assert "mamba_builders.py" in script
    assert "hybrid_builders.py" in script
    assert "cppmega_mamba_builder as hybrid_builder" in script
    assert "model_provider.py" not in script


def test_overlay_includes_batch_and_dataset_sidecar_contract():
    assert "cppmega/megatron/custom_mamba_model.py" in OVERLAY_PATHS
    assert "cppmega/megatron/mamba_builder.py" in OVERLAY_PATHS
    assert "cppmega/megatron/te_checkpoint_kwarg_patch.py" in OVERLAY_PATHS
    assert "cppmega/megatron/dsa_indexer_fused_patch.py" in OVERLAY_PATHS
    assert "cppmega/megatron/graph_route_attention_bias_patch.py" in OVERLAY_PATHS
    assert "cppmega/megatron/structure_dataset_patch.py" in OVERLAY_PATHS
    assert "cppmega/megatron/structure_batch.py" in OVERLAY_PATHS
    assert "cppmega/megatron/h200_preflight.py" in OVERLAY_PATHS
    assert "scripts/h200_megatron_preflight.py" in OVERLAY_PATHS


def test_remote_script_enables_graph_routes_and_uses_selected_data_prefix():
    script = remote_run_script(
        [256],
        1,
        "ghcr.io/datasunriseou/cppmega:latest",
        data_prefix_name="cppmega_1024_current_mix_graph_train",
    )

    assert 'export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-1}"' in script
    assert 'export CPPMEGA_GRAPH_ROUTES_ENABLED="${CPPMEGA_GRAPH_ROUTES_ENABLED:-1}"' in script
    assert 'export CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS="${CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS:-1}"' in script
    assert "apply_dsa_indexer_fused_patch()" in script
    assert "apply_graph_route_attention_bias_patch()" in script
    assert "1024:cppmega_1024_current_mix_graph_train" in script
    assert 'DATA_PREFIX="/data/cppmega_sidecar/${DATA_PREFIX_NAME}"' in script
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
    assert "--cross-entropy-fusion-impl te" in script
    assert "--cross-entropy-fusion-impl linear" not in script


def test_remote_script_runs_fail_closed_h200_preflight_before_sweep():
    script = remote_run_script(
        [64],
        3,
        "ghcr.io/datasunriseou/cppmega:latest",
        data_prefix_name="cppmega_1024_current_mix_graph_train",
    )

    preflight = "python /opt/cppmega/scripts/h200_megatron_preflight.py"
    assert preflight in script
    assert "--data-prefix \"$DATA_PREFIX\"" in script
    assert "--tokenizer-model /data/cpp_tokenizer_hf" in script
    assert "--output /data/cppmega_h200_results/h200_preflight.json" in script
    assert "CPPMEGA_H200_PREFLIGHT_STATUS=PASS" in script
    assert script.index(preflight) < script.index('for SPEC in "${TEST_SPECS[@]}"')


def test_remote_script_can_sweep_multiple_seq_lengths_with_separate_prefixes():
    script = remote_run_script(
        [64, 128],
        100,
        "ghcr.io/datasunriseou/cppmega:latest",
        seq_data_prefixes=[
            (1024, "cppmega_h200_100step_seq1024_graph_train"),
            (2048, "cppmega_h200_100step_seq2048_graph_train"),
            (4096, "cppmega_h200_100step_seq4096_graph_train"),
        ],
    )

    assert "1024:cppmega_h200_100step_seq1024_graph_train" in script
    assert "2048:cppmega_h200_100step_seq2048_graph_train" in script
    assert "4096:cppmega_h200_100step_seq4096_graph_train" in script
    assert "--seq-length ${SEQ}" in script
    assert "--max-position-embeddings ${SEQ}" in script
    assert "seq_${SEQ}_bs_${BS}.log" in script
    assert "CPPMEGA_BATCH_RESULT seq=${SEQ} batch=${BS}" in script


def test_remote_script_can_enable_tensorwise_fp8_flags():
    script = remote_run_script(
        [64],
        100,
        "ghcr.io/datasunriseou/cppmega:latest",
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
    script = remote_run_script([64], 1, "ghcr.io/datasunriseou/cppmega:latest")

    assert "def _cppmega_distributed_shutdown()" in script
    assert "dist.destroy_process_group()" in script
    assert "torch.cuda.synchronize()" in script
    assert "CPPMEGA_DISTRIBUTED_SHUTDOWN_ERROR" in script


def test_remote_script_can_disable_te_nvrtc_for_cleanup_crash_probe():
    script = remote_run_script(
        [64],
        1,
        "ghcr.io/datasunriseou/cppmega:latest",
        disable_nvrtc=True,
    )

    assert 'export NVTE_DISABLE_NVRTC="1"' in script


def test_remote_script_can_save_model_only_checkpoint():
    script = remote_run_script(
        [192],
        5000,
        "ghcr.io/datasunriseou/cppmega:latest",
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
        "ghcr.io/datasunriseou/cppmega:latest",
        save_checkpoint=True,
        save_interval=1000,
    )
    script_path = tmp_path / "run_cppmega_h200_sweep.sh"
    script_path.write_text(script)

    assert script.startswith("#!/usr/bin/env bash\n")
    assert "\nINNER\n\nsudo docker run" in script
    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_remote_script_disables_checkpointing_by_default():
    script = remote_run_script([64], 100, "ghcr.io/datasunriseou/cppmega:latest")

    assert "--save-interval 50000000" in script
    assert "--no-save-optim" not in script


def test_remote_script_can_load_model_only_checkpoint():
    script = remote_run_script(
        [192],
        1,
        "ghcr.io/datasunriseou/cppmega:latest",
        load_checkpoint_remote="/data/cppmega_load_checkpoint",
    )

    assert "CHECKPOINT_ARGS+=(--load /data/cppmega_load_checkpoint)" in script
    assert "CHECKPOINT_ARGS+=(--no-load-optim --no-load-rng)" in script
    assert "--no-save-optim" not in script


def test_remote_script_can_load_full_checkpoint_state():
    script = remote_run_script(
        [192],
        1,
        "ghcr.io/datasunriseou/cppmega:latest",
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
    pubkey = tmp_path / "id_ed25519.pub"
    pubkey.write_text("ssh-ed25519 TESTKEY codex\n")
    rc = main(
        [
            "--dry-run",
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


def test_fp8_tensorwise_can_keep_nvrtc_enabled_for_perf_probe(tmp_path, capsys):
    pubkey = tmp_path / "id_ed25519.pub"
    pubkey.write_text("ssh-ed25519 TESTKEY codex\n")
    rc = main(
        [
            "--dry-run",
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


def test_sidecar_preflight_rejects_missing_graph_sidecars(tmp_path):
    prefix = tmp_path / "sample_train"
    manifest = _write_valid_sidecar_prefix(prefix)
    manifest["graph_sidecar_paths"].pop("token_chunk_ends")
    prefix.with_suffix(".json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing graph sidecars"):
        _assert_prefix_contract(prefix)


def test_sidecar_preflight_rejects_zero_structure_values(tmp_path):
    prefix = tmp_path / "sample_train"
    manifest = _write_valid_sidecar_prefix(prefix)
    structure = prefix.parent / manifest["side_channel_paths"]["token_structure_ids"]["path"]
    structure.write_bytes(b"\x00" * structure.stat().st_size)

    with pytest.raises(ValueError, match="token_structure_ids.*nonzero"):
        _assert_prefix_contract(prefix)


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
    assert receipt["checkpoint"]["full_optimizer_and_rng_state"] is True
    assert receipt["commands"]["save"][receipt["commands"]["save"].index("--train-iters") + 1] == "1"
    assert receipt["commands"]["restore"][receipt["commands"]["restore"].index("--train-iters") + 1] == "2"
    assert "--load" in receipt["commands"]["restore"]
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
