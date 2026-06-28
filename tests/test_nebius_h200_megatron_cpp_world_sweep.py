from scripts.nebius_h200_megatron_cpp_world_sweep import (
    OVERLAY_PATHS,
    first_public_ip,
    make_ghcr_auth_tar,
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
    assert "model_provider.py" not in script


def test_overlay_includes_batch_and_dataset_sidecar_contract():
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
    assert "--data-path 1.0 /data/cppmega_sidecar/cppmega_1024_current_mix_graph_train" in script


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
    assert "cpp_tokenizer_hf/tokenizer.json" in names
