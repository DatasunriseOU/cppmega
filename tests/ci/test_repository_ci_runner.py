from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import repository_runner as ci


REPO_ROOT = Path(__file__).resolve().parents[2]
HOSTS_CONFIG = REPO_ROOT / "configs" / "ci" / "hosts.json"
LANES_CONFIG = REPO_ROOT / "configs" / "ci" / "lanes.json"
CPPMEGA_MLX_LANES_CONFIG = REPO_ROOT / "configs" / "ci" / "cppmega_mlx_lanes.json"


def test_default_receipt_base_is_durable_and_repository_owned() -> None:
    assert ci.DEFAULT_RECEIPT_BASE == (
        REPO_ROOT / "outputs" / "ci_diagnostics" / "repository-ci"
    )


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        for child in value.values():
            keys.update(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_all_keys(child))
        return keys
    return set()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-q")
    (path / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(
        path,
        "-c",
        "user.name=Repository CI Test",
        "-c",
        "user.email=ci-test@example.invalid",
        "commit",
        "-q",
        "-m",
        "initial",
    )


def _minimal_lane_config(
    path: Path,
    *,
    command: list[str] | None = None,
    lane_id: str = "local-test",
    test_profile: str | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "lanes": [
            {
                "id": lane_id,
                "test_profile": test_profile,
                "system": platform.system().lower(),
                "machines": [platform.machine().lower()],
                "requires_cuda": False,
                "required_modules": [],
                "timeout_seconds": 20,
                "commands": [
                    {
                        "name": "test-command",
                        "timeout_seconds": 10,
                        "argv": command or ["{python}", "-c", "print('lane passed')"],
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _minimal_host_config(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "repository": "test/local",
        "hosts": [
            {
                "id": "local-test-host",
                "address": "local",
                "transport": "local",
                "system": platform.system().lower(),
                "machines": [platform.machine().lower()],
                "python": sys.executable,
                "lanes": ["local-test"],
                "dispatch_enabled": True,
                "required": True,
                "capabilities": ["unit-test"],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_inventory_maps_actual_hosts_and_supported_lanes() -> None:
    lanes = ci.load_lanes(LANES_CONFIG)
    _, rows = ci.load_hosts(HOSTS_CONFIG, lanes=lanes)
    hosts = {host["id"]: host for host in rows}

    assert {host["address"] for host in rows} >= {
        "10.0.0.11",
        "10.0.0.12",
        "10.0.0.16",
    }
    assert hosts["local-macos"]["lanes"] == ["macos-contracts"]
    assert hosts["legion-10-0-0-16"]["lanes"] == [
        "linux-contracts",
        "linux-cuda",
    ]
    assert hosts["legion-10-0-0-16"]["dispatch_enabled"] is True
    assert hosts["windows-10-0-0-11"]["lanes"] == []
    assert hosts["quarantined-10-0-0-12"]["trust"] == "quarantined"
    assert hosts["quarantined-10-0-0-12"]["dispatch_enabled"] is False


def test_local_macos_uses_owned_source_environment() -> None:
    lanes = ci.load_lanes(LANES_CONFIG)
    _, rows = ci.load_hosts(HOSTS_CONFIG, lanes=lanes)
    host = next(row for row in rows if row["id"] == "local-macos")

    assert host["python"] == "/Volumes/external/sources/.venvs/cppmega.source/bin/python"
    assert "/cppmega.mlx/.venv/" not in host["python"]
    assert "/nanochat/.venv/" not in host["python"]


def test_macos_lane_covers_cross_repo_case_contracts_and_preserves_peer_paths() -> None:
    lanes = ci.load_lanes(LANES_CONFIG)
    lane = lanes["macos-contracts"]
    argv = {
        argument
        for command in lane["commands"]
        for argument in command["argv"]
    }

    assert {
        "tests/test_audit_sidecar_parquet.py",
        "tests/test_case4_identity_parity.py",
        "tests/test_case5_v7_parser_token_span_regressions.py",
        "tests/test_case6_nebius_contracts.py",
        "tests/test_graph_objective_loss.py",
        "tests/test_graph_recipe.py",
        "tests/test_megatron_objective_contract.py",
        "tests/test_prompt_graph_index_loader_contract.py",
        "tests/test_prompt_graph_schema_v2.py",
        "tests/test_recipe_artifact_parity.py",
    }.issubset(argv)
    assert {
        "CPPMEGA_MLX_REFERENCE_ROOT",
        "CPPMEGA_RECIPE_PARITY_PEER_ROOT",
        "CPPMEGA_RECIPE_PARITY_PYTHON",
    }.issubset(ci._PASSTHROUGH_ENV)


def test_repository_runner_module_entrypoint_executes_cli() -> None:
    runner = REPO_ROOT / "scripts" / "ci" / "repository_runner.py"
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "list",
            "--host",
            "local-macos",
            "--lane",
            "macos-contracts",
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["hosts"][0]["id"] == "local-macos"
    assert set(payload["lanes"]) == {"macos-contracts"}


def test_configs_contain_no_credentials_or_mutable_install_steps() -> None:
    host_payload = json.loads(HOSTS_CONFIG.read_text(encoding="utf-8"))
    lane_payload = json.loads(LANES_CONFIG.read_text(encoding="utf-8"))
    forbidden = {
        "authorization",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
        "token",
    }

    assert not (_all_keys(host_payload) | _all_keys(lane_payload)).intersection(
        forbidden
    )
    serialized = json.dumps([host_payload, lane_payload]).lower()
    assert "pip install" not in serialized
    assert "conda install" not in serialized
    assert "apt-get install" not in serialized
    assert "brew install" not in serialized
    assert "github_pat_" not in serialized
    assert "ghp_" not in serialized

    ci.load_lanes(LANES_CONFIG)


def test_lane_parser_rejects_dependency_install_and_shell_commands(
    tmp_path: Path,
) -> None:
    install_path = _minimal_lane_config(
        tmp_path / "install.json",
        command=["{python}", "-m", "pip", "install", "pytest"],
    )
    with pytest.raises(ci.RepositoryCIError, match="may not mutate dependencies"):
        ci.load_lanes(install_path)

    shell_path = _minimal_lane_config(
        tmp_path / "shell.json",
        command=["bash", "-lc", "pytest -q"],
    )
    with pytest.raises(ci.RepositoryCIError, match="may not use a shell"):
        ci.load_lanes(shell_path)


def test_ssh_command_pins_identity_and_disables_interactive_or_forwarded_access(
    tmp_path: Path,
) -> None:
    lanes = ci.load_lanes(LANES_CONFIG)
    _, rows = ci.load_hosts(HOSTS_CONFIG, lanes=lanes)
    host = next(row for row in rows if row["id"] == "legion-10-0-0-16")
    known_hosts = ci._write_known_hosts(host, tmp_path)
    identity_file = tmp_path / "runner-key"
    identity_file.write_text("test-only\n", encoding="ascii")

    command = ci._ssh_base(
        host,
        known_hosts=known_hosts,
        connect_timeout=7,
        identity_file=identity_file,
    )
    rendered = " ".join(command)

    assert "BatchMode=yes" in rendered
    assert "PasswordAuthentication=no" in rendered
    assert "KbdInteractiveAuthentication=no" in rendered
    assert "PreferredAuthentications=publickey" in rendered
    assert "StrictHostKeyChecking=yes" in rendered
    assert f"UserKnownHostsFile={known_hosts}" in rendered
    assert "GlobalKnownHostsFile=/dev/null" in rendered
    assert "ForwardAgent=no" in rendered
    assert "ClearAllForwardings=yes" in rendered
    assert "IdentitiesOnly=yes" in rendered
    assert str(identity_file) in command
    assert "ConnectTimeout=7" in rendered
    assert command[-1] == "davidgor@10.0.0.16"
    assert known_hosts.read_text(encoding="ascii") == ci._known_hosts_line(host)


def test_trusted_fingerprints_and_quarantined_host_denies_auth() -> None:
    lanes = ci.load_lanes(LANES_CONFIG)
    _, rows = ci.load_hosts(HOSTS_CONFIG, lanes=lanes)
    trusted = [host for host in rows if host.get("trust") == "trusted"]
    for host in trusted:
        assert (
            ci._host_key_fingerprint(host["host_key"]["public_key"])
            == (host["host_key"]["fingerprint_sha256"])
        )
        assert (
            ci._host_identity_decision(
                host,
                {"status": "observed", "key": host["host_key"]["public_key"]},
            )["may_authenticate"]
            is True
        )

    quarantined = next(host for host in rows if host["id"] == "quarantined-10-0-0-12")
    decision = ci._host_identity_decision(
        quarantined,
        {
            "status": "observed",
            "fingerprint_sha256": "SHA256:observed-only",
            "key": "not-trusted",
        },
    )

    assert decision == {
        "may_authenticate": False,
        "status": "quarantined",
        "detail": "untrusted_host_identity",
        "identity_verified": False,
    }


@pytest.mark.parametrize(
    ("scan", "detail"),
    [
        ({"status": "keyscan_failed", "key": None}, "keyscan_failed"),
        ({"status": "observed", "key": "unexpected-key"}, "host_key_mismatch"),
    ],
)
def test_trusted_host_identity_decision_fails_closed(
    scan: dict[str, Any], detail: str
) -> None:
    lanes = ci.load_lanes(LANES_CONFIG)
    _, rows = ci.load_hosts(HOSTS_CONFIG, lanes=lanes)
    host = next(row for row in rows if row["id"] == "legion-10-0-0-16")

    decision = ci._host_identity_decision(host, scan)

    assert decision["may_authenticate"] is False
    assert decision["identity_verified"] is False
    assert decision["detail"] == detail


def test_step_timeout_kills_process_group(tmp_path: Path) -> None:
    started = time.monotonic()
    result = ci.run_step(
        name="bounded-sleep",
        command=(sys.executable, "-c", "import time; time.sleep(30)"),
        cwd=tmp_path,
        log_path=tmp_path / "bounded-sleep.log",
        timeout_seconds=0.1,
    )

    assert result["status"] == "timed_out"
    assert result["exit_code"] == 124
    assert time.monotonic() - started < 5


def test_sanitized_environment_disables_user_site_and_unsafe_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYTHONNOUSERSITE", "0")
    monkeypatch.setenv("PYTHONSAFEPATH", "0")
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted")

    environment = ci._sanitized_environment(tmp_path)

    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"
    assert environment["PYTHONPATH"] == str(tmp_path)


def test_lane_subprocess_binds_project_imports_to_reviewed_checkout(
    tmp_path: Path,
) -> None:
    foreign_root = tmp_path / "foreign-editable-checkout"
    foreign_package = foreign_root / "cppmega"
    foreign_package.mkdir(parents=True)
    (foreign_package / "__init__.py").write_text(
        "__file_marker__ = 'foreign'\n", encoding="utf-8"
    )
    lanes_path = _minimal_lane_config(
        tmp_path / "lanes.json",
        command=[
            "{python}",
            "-c",
            "from pathlib import Path; import cppmega; "
            "print(Path(cppmega.__file__).resolve())",
        ],
    )
    receipt_dir = tmp_path / "lane-receipt"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(foreign_root)
    runner = REPO_ROOT / "scripts" / "ci" / "run_repository_ci.py"

    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "lane",
            "--lanes-config",
            str(lanes_path),
            "--lane",
            "local-test",
            "--repo-root",
            str(REPO_ROOT),
            "--receipt-dir",
            str(receipt_dir),
            "--python",
            sys.executable,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    log = (receipt_dir / "test-command.log").read_text(encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    resolved = Path(log.strip()).resolve()
    assert resolved.is_relative_to(REPO_ROOT)
    assert not resolved.is_relative_to(foreign_root)


def test_run_step_preserves_explicit_test_profile(tmp_path: Path) -> None:
    driver = """
import sys
from pathlib import Path

from scripts.ci.repository_runner import run_step

result = run_step(
    name="profile",
    command=(
        sys.executable,
        "-c",
        "import os; print(os.environ.get('CPPMEGA_TEST_PROFILE'))",
    ),
    cwd=Path.cwd(),
    log_path=Path("profile.log"),
    timeout_seconds=5,
)
raise SystemExit(result["exit_code"])
"""
    environment = os.environ.copy()
    environment["CPPMEGA_TEST_PROFILE"] = "portable-data"
    environment["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "profile.log").read_text(encoding="utf-8").strip() == (
        "portable-data"
    )


def test_linux_contract_lane_sets_portable_test_profile(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    lanes_path = _minimal_lane_config(
        tmp_path / "lanes.json",
        lane_id="linux-contracts",
        test_profile="portable-data",
        command=[
            "{python}",
            "-c",
            "import os; print(os.environ.get('CPPMEGA_TEST_PROFILE'))",
        ],
    )
    receipt_dir = tmp_path / "receipt"
    environment = os.environ.copy()
    environment.pop("CPPMEGA_TEST_PROFILE", None)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    runner = REPO_ROOT / "scripts" / "ci" / "repository_runner.py"
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "lane",
            "--lanes-config",
            str(lanes_path),
            "--lane",
            "linux-contracts",
            "--repo-root",
            str(repo),
            "--receipt-dir",
            str(receipt_dir),
            "--python",
            sys.executable,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (receipt_dir / "test-command.log").read_text(
        encoding="utf-8"
    ).strip() == "portable-data"


def test_portable_profile_allows_repository_runner_regressions() -> None:
    import conftest

    assert "tests/ci/test_repository_ci_runner.py" in conftest._PORTABLE_TEST_ALLOWLIST


def test_step_logs_redact_environment_tokens_and_private_keys(
    tmp_path: Path,
) -> None:
    secret = "unit-test-secret-value-12345"
    code = (
        "print('token="
        + secret
        + "'); print('Bearer github_pat_abcdefghijklmnopqrstuvwxyz'); "
        "print('-----BEGIN PRIVATE KEY-----'); print('private-material'); "
        "print('-----END PRIVATE KEY-----')"
    )
    result = ci.run_step(
        name="redaction",
        command=(sys.executable, "-c", code),
        cwd=tmp_path,
        log_path=tmp_path / "redaction.log",
        timeout_seconds=5,
    )
    log = (tmp_path / "redaction.log").read_text(encoding="utf-8")

    assert result["status"] == "passed"
    assert secret not in log
    assert "github_pat_" not in log
    assert "private-material" not in log
    assert "<redacted>" in log
    assert "[REDACTED PRIVATE KEY]" in log


def test_provenance_detects_tracked_worktree_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    before = ci.capture_provenance(repo)
    (repo / "tracked.txt").write_text("after\n", encoding="utf-8")
    after = ci.capture_provenance(repo)

    assert before["dirty"] is False
    assert after["dirty"] is True
    assert ci.provenance_unchanged(before, after) is False


def test_list_json_is_machine_readable_and_omits_full_public_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = ci.main(["list", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema_version"] == ci.SCHEMA_VERSION
    assert {host["address"] for host in payload["hosts"]} >= {
        "10.0.0.11",
        "10.0.0.12",
        "10.0.0.16",
    }
    assert "public_key" not in json.dumps(payload)
    assert "host_key_fingerprint_sha256" in json.dumps(payload)


def test_local_dry_run_writes_before_after_provenance_receipt(tmp_path: Path) -> None:
    lanes_path = _minimal_lane_config(tmp_path / "lanes.json")
    hosts_path = _minimal_host_config(tmp_path / "hosts.json")
    receipt_base = tmp_path / "receipts"

    exit_code = ci.main(
        [
            "run",
            "--dry-run",
            "--hosts-config",
            str(hosts_path),
            "--lanes-config",
            str(lanes_path),
            "--repo-root",
            str(REPO_ROOT),
            "--receipt-dir",
            str(receipt_base),
            "--run-id",
            "unit-dry-run",
        ]
    )
    receipt = json.loads(
        (receipt_base / "unit-dry-run" / "orchestration.json").read_text(
            encoding="utf-8"
        )
    )

    assert exit_code == 0
    assert receipt["status"] == "dry_run_passed"
    assert receipt["dry_run"] is True
    assert receipt["source_provenance"]["before_preflight"]
    assert receipt["source_provenance"]["after_execution"]
    assert receipt["source_provenance"]["unchanged"] is True
    assert receipt["preflights"][0]["lane_status"]["local-test"]["available"]


def test_lane_receipt_captures_provenance_before_and_after_tests(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    lanes_path = _minimal_lane_config(tmp_path / "lanes.json")
    receipt_dir = tmp_path / "lane-receipt"
    expected_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    expected_commit = _git(repo, "rev-parse", "HEAD")

    exit_code = ci.main(
        [
            "lane",
            "--lanes-config",
            str(lanes_path),
            "--lane",
            "local-test",
            "--repo-root",
            str(repo),
            "--receipt-dir",
            str(receipt_dir),
            "--python",
            sys.executable,
            "--expected-source-commit",
            expected_commit,
            "--expected-source-tree",
            expected_tree,
            "--archive-sha256",
            "a" * 64,
        ]
    )
    receipt = json.loads((receipt_dir / "receipt.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert receipt["status"] == "passed"
    assert receipt["source"]["requested_commit"] == expected_commit
    assert receipt["source"]["requested_tree"] == expected_tree
    assert receipt["provenance"]["before_tests"]
    assert receipt["provenance"]["after_tests"]
    assert receipt["provenance"]["unchanged"] is True


def test_lane_rejects_a_mismatched_requested_source_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    lanes_path = _minimal_lane_config(tmp_path / "lanes.json")
    receipt_dir = tmp_path / "lane-receipt"
    expected_tree = _git(repo, "rev-parse", "HEAD^{tree}")

    exit_code = ci.main(
        [
            "lane",
            "--lanes-config",
            str(lanes_path),
            "--lane",
            "local-test",
            "--repo-root",
            str(repo),
            "--receipt-dir",
            str(receipt_dir),
            "--python",
            sys.executable,
            "--expected-source-commit",
            "0" * 40,
            "--expected-source-tree",
            expected_tree,
        ]
    )
    receipt = json.loads((receipt_dir / "receipt.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert receipt["status"] == "failed"
    assert "does not match the requested source commit" in receipt["error"]


def test_lane_fails_when_a_test_mutates_the_tracked_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    command = [
        "{python}",
        "-c",
        "from pathlib import Path; Path('tracked.txt').write_text('changed\\n')",
    ]
    lanes_path = _minimal_lane_config(tmp_path / "lanes.json", command=command)
    receipt_dir = tmp_path / "lane-receipt"

    exit_code = ci.main(
        [
            "lane",
            "--lanes-config",
            str(lanes_path),
            "--lane",
            "local-test",
            "--repo-root",
            str(repo),
            "--receipt-dir",
            str(receipt_dir),
            "--python",
            sys.executable,
        ]
    )
    receipt = json.loads((receipt_dir / "receipt.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert receipt["status"] == "failed_provenance_changed"
    assert receipt["provenance"]["unchanged"] is False


def test_contract_lane_manifests_include_runner_regressions_and_cuda_probes() -> None:
    lanes = ci.load_lanes(LANES_CONFIG)
    for lane_id in ("macos-contracts", "linux-contracts"):
        argv = [
            part for command in lanes[lane_id]["commands"] for part in command["argv"]
        ]
        assert "tests/ci/test_repository_ci_runner.py" in argv
        assert "tests/test_workflow_runner_policy.py" in argv

    cuda_argv = [
        part for command in lanes["linux-cuda"]["commands"] for part in command["argv"]
    ]
    assert "tests/test_m2rnn_pararnn_tiled_cuda.py" in cuda_argv
    assert "tests/test_noconv_f2_gpu.py" in cuda_argv
    assert lanes["linux-cuda"]["requires_cuda"] is True

    mlx_lanes = ci.load_lanes(CPPMEGA_MLX_LANES_CONFIG)
    mlx_argv = [
        part
        for command in mlx_lanes["macos-cppmega-mlx-contracts"]["commands"]
        for part in command["argv"]
    ]
    assert "tests/test_inference_generation.py" in mlx_argv
    assert "tests/test_train_eval_graph_routes.py" in mlx_argv
    assert "tests/test_self_hosted_ci.py" in mlx_argv
    assert "tests/test_workflow_runner_policy.py" in mlx_argv


def _run_lane_cli(lanes_path: Path, repo: Path, receipt_dir: Path, lane: str) -> int:
    return ci.main(
        [
            "lane",
            "--lanes-config",
            str(lanes_path),
            "--lane",
            lane,
            "--repo-root",
            str(repo),
            "--receipt-dir",
            str(receipt_dir),
            "--python",
            sys.executable,
        ]
    )


def test_lane_orchestrator_writes_failure_receipt_for_unknown_lane(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    lanes_path = _minimal_lane_config(tmp_path / "lanes.json")
    receipt_dir = tmp_path / "lane-receipt"

    exit_code = _run_lane_cli(lanes_path, repo, receipt_dir, "does-not-exist")
    receipt = json.loads((receipt_dir / "receipt.json").read_text(encoding="utf-8"))
    log = (receipt_dir / "orchestrator-failure.log").read_text(encoding="utf-8")

    assert exit_code == 2
    assert receipt["schema_version"] == ci.SCHEMA_VERSION
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == exit_code
    assert receipt["failure_stage"] == "orchestrator"
    assert receipt["lane"] == "does-not-exist"
    assert receipt["started_at"]
    assert receipt["completed_at"]
    assert "does-not-exist" in receipt["error"]
    assert "Traceback" in log
    assert "KeyError" in log


def test_lane_orchestrator_writes_failure_receipt_for_missing_lane_config(
    tmp_path: Path,
) -> None:
    receipt_dir = tmp_path / "lane-receipt"

    exit_code = _run_lane_cli(
        tmp_path / "missing.json", tmp_path, receipt_dir, "local-test"
    )
    receipt = json.loads((receipt_dir / "receipt.json").read_text(encoding="utf-8"))
    log = (receipt_dir / "orchestrator-failure.log").read_text(encoding="utf-8")

    assert exit_code == 2
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == exit_code
    assert receipt["failure_stage"] == "orchestrator"
    assert "cannot read lane config" in receipt["error"]
    assert "Traceback" in log


def test_early_failure_receipt_does_not_clobber_an_existing_lane_receipt(
    tmp_path: Path,
) -> None:
    receipt_dir = tmp_path / "lane-receipt"
    receipt_dir.mkdir()
    existing = {"schema_version": ci.SCHEMA_VERSION, "kind": "lane", "status": "passed"}
    (receipt_dir / "receipt.json").write_text(json.dumps(existing), encoding="utf-8")
    args = argparse.Namespace(
        command="lane",
        receipt_dir=str(receipt_dir),
        run_id="unit-early-failure",
        lane="local-test",
    )

    ci._write_early_failure_receipt(args, RuntimeError("post-receipt crash"))
    receipt = json.loads((receipt_dir / "receipt.json").read_text(encoding="utf-8"))

    assert receipt == existing
    assert "post-receipt crash" in (
        receipt_dir / "orchestrator-failure.log"
    ).read_text(encoding="utf-8")


def test_early_failure_receipt_redacts_secrets(tmp_path: Path) -> None:
    secret = "unit-test-orchestrator-secret-12345"
    receipt_dir = tmp_path / "lane-receipt"
    args = argparse.Namespace(
        command="lane",
        receipt_dir=str(receipt_dir),
        run_id=None,
        lane="local-test",
    )

    ci._write_early_failure_receipt(args, RuntimeError(f"token={secret} boom"))
    receipt = json.loads((receipt_dir / "receipt.json").read_text(encoding="utf-8"))

    assert secret not in receipt["error"]
    assert "<redacted>" in receipt["error"]
