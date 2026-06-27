from scripts.nebius_h200_megatron_cpp_world_sweep import (
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
