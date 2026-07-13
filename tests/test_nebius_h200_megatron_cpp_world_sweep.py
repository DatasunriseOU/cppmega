import subprocess

import pytest

from scripts.nebius_h200_megatron_cpp_world_sweep import (
    OVERLAY_PATHS,
    _docker_auth_from_config,
    first_public_ip,
    main,
    make_checkpoint_tar,
    make_ghcr_auth_tar,
    make_multi_sidecar_tar,
    remote_run_script,
)


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
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")

    (prefix.with_suffix(".bin")).write_bytes(b"tokens")
    (prefix.with_suffix(".idx")).write_bytes(b"index")
    (tmp_path / "sample_train_token_structure_ids.bin").write_bytes(b"s")
    (tmp_path / "sample_train_token_call_edges_offsets.bin").write_bytes(b"o")
    (tmp_path / "sample_train_token_call_edges_data.bin").write_bytes(b"d")
    for suffix in (
        "source_platform_sequence_doc_offsets.bin",
        "source_platform_doc_id_offsets.bin",
        "source_platform_ids.bin",
    ):
        (tmp_path / f"sample_train_{suffix}").write_bytes(b"p")
    prefix.with_suffix(".json").write_text(
        __import__("json").dumps(
            {
                "side_channel_paths": {
                    "token_structure_ids": {
                        "path": "sample_train_token_structure_ids.bin",
                        "dtype": "uint8",
                    }
                },
                "graph_sidecar_paths": {
                    "token_call_edges": {
                        "offsets_path": "sample_train_token_call_edges_offsets.bin",
                        "data_path": "sample_train_token_call_edges_data.bin",
                    }
                },
                "source_platform_sidecar": {
                    "sequence_doc_offsets_path": "sample_train_source_platform_sequence_doc_offsets.bin",
                    "doc_platform_offsets_path": "sample_train_source_platform_doc_id_offsets.bin",
                    "platform_ids_path": "sample_train_source_platform_ids.bin",
                },
            }
        )
    )
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
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")

    prefixes = [tmp_path / "seq1024_train", tmp_path / "seq2048_train"]
    for prefix in prefixes:
        prefix.with_suffix(".bin").write_bytes(b"tokens")
        prefix.with_suffix(".idx").write_bytes(b"index")
        (tmp_path / f"{prefix.name}_token_structure_ids.bin").write_bytes(b"s")
        prefix.with_suffix(".json").write_text(
            __import__("json").dumps(
                {
                    "side_channel_paths": {
                        "token_structure_ids": {
                            "path": f"{prefix.name}_token_structure_ids.bin",
                            "dtype": "uint8",
                        }
                    },
                    "graph_sidecar_paths": {},
                }
            )
        )

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
