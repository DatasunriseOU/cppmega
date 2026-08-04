from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.distributed_data_prep._common import ContractError, atomic_write_json
from scripts.gcp_isolated_worker_rollout import (
    ROLL_OUT_RECEIPT_SCHEMA,
    _main,
    _plan_environment,
    _validate_backend_metadata,
    backend_hcl,
    build_rollout_spec,
    validate_isolated_plan,
    write_backend_config,
)


def _spec():
    return build_rollout_spec(
        bucket="natural-bison-491019-t9-cppmega-corpus",
        run_id="source-prod-20260804-005",
        name_prefix="cppmega-corpus",
        worker_count=2,
        compact_placement=True,
    )


def _resource(address: str, resource_type: str, name: str, *, spec):
    values: dict[str, object] = {"name": name}
    if resource_type == "google_compute_instance":
        values["labels"] = {"run-id": spec.run_id}
        values["metadata"] = {"cppmega-run-root": spec.run_root}
    return {
        "address": address,
        "mode": "managed",
        "type": resource_type,
        "name": (
            "worker" if resource_type != "google_compute_resource_policy" else "compact"
        ),
        "values": values,
    }


def _plan(*, spec=None):
    spec = spec or _spec()
    resources = []
    for worker_name in spec.worker_names:
        resources.extend(
            [
                _resource(
                    f'google_compute_address.worker["{worker_name}"]',
                    "google_compute_address",
                    f"{worker_name}-{spec.run_id}",
                    spec=spec,
                ),
                _resource(
                    f'google_compute_instance.worker["{worker_name}"]',
                    "google_compute_instance",
                    f"{worker_name}-{spec.run_id}",
                    spec=spec,
                ),
            ]
        )
    if spec.compact_placement:
        resources.append(
            _resource(
                "google_compute_resource_policy.compact[0]",
                "google_compute_resource_policy",
                f"{spec.name_prefix}-{spec.run_id}-compact",
                spec=spec,
            )
        )
    return {
        "planned_values": {"root_module": {"resources": resources}},
        "prior_state": None,
        "resource_changes": [
            {
                "address": resource["address"],
                "mode": "managed",
                "change": {"actions": ["create"]},
            }
            for resource in resources
        ],
    }


def test_backend_is_run_scoped_and_refuses_binding_drift(tmp_path: Path) -> None:
    spec = _spec()
    path = tmp_path / "source-prod-005.backend.hcl"
    assert write_backend_config(path, spec=spec) is True
    assert path.read_text(encoding="ascii") == backend_hcl(spec)
    assert "terraform/source-runs/source-prod-20260804-005" in path.read_text()
    assert write_backend_config(path, spec=spec) is False

    other = build_rollout_spec(
        bucket=spec.bucket,
        run_id="source-prod-20260804-006",
        name_prefix=spec.name_prefix,
        worker_count=spec.worker_count,
        compact_placement=True,
    )
    with pytest.raises(ContractError, match="different run binding"):
        write_backend_config(path, spec=other)


def test_backend_metadata_must_bind_the_same_run(tmp_path: Path) -> None:
    spec = _spec()
    metadata_path = tmp_path / "terraform.tfstate"
    atomic_write_json(
        metadata_path,
        {
            "backend": {
                "type": "gcs",
                "config": {"bucket": spec.bucket, "prefix": spec.backend_prefix},
            }
        },
    )
    _validate_backend_metadata(tmp_path, spec=spec)

    atomic_write_json(
        metadata_path,
        {
            "backend": {
                "type": "gcs",
                "config": {"bucket": spec.bucket, "prefix": "terraform/workers"},
            }
        },
    )
    with pytest.raises(ContractError, match="not bound"):
        _validate_backend_metadata(tmp_path, spec=spec)


def test_gcloud_token_mode_overrides_ambient_credential_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/underprivileged.json")
    monkeypatch.setenv("GOOGLE_CREDENTIALS", "underprivileged")
    monkeypatch.setenv("GOOGLE_BACKEND_CREDENTIALS", "underprivileged")
    monkeypatch.setenv("GOOGLE_IMPERSONATE_SERVICE_ACCOUNT", "wrong@example.invalid")
    monkeypatch.setenv("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", "/tmp/wrong.json")
    monkeypatch.setattr(
        "scripts.gcp_isolated_worker_rollout._gcloud_access_token",
        lambda: "ya29.test-token",
    )

    environment, auth_mode = _plan_environment(use_gcloud_access_token=True)

    assert auth_mode == "gcloud-active-token"
    assert environment["GOOGLE_OAUTH_ACCESS_TOKEN"] == "ya29.test-token"
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in environment
    assert "GOOGLE_CREDENTIALS" not in environment
    assert "GOOGLE_BACKEND_CREDENTIALS" not in environment
    assert "GOOGLE_IMPERSONATE_SERVICE_ACCOUNT" not in environment
    assert "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE" not in environment


def test_plan_accepts_exact_target_creates_only() -> None:
    spec = _spec()
    receipt = validate_isolated_plan(_plan(spec=spec), spec=spec)
    assert receipt["schema"] == ROLL_OUT_RECEIPT_SCHEMA
    assert receipt["status"] == "validated_no_destroy_not_applied"
    assert receipt["backend"] == {
        "bucket": spec.bucket,
        "prefix": "terraform/source-runs/source-prod-20260804-005",
    }
    assert len(receipt["expected_managed_resources"]) == 5


def test_plan_rejects_delete_from_another_run_state() -> None:
    spec = _spec()
    plan = _plan(spec=spec)
    old = copy.deepcopy(plan["planned_values"]["root_module"]["resources"][0])
    old["values"]["name"] = "cppmega-corpus-00-source-prod-20260804-004"
    plan["prior_state"] = {"values": {"root_module": {"resources": [old]}}}
    plan["resource_changes"].append(
        {
            "address": old["address"],
            "mode": "managed",
            "change": {"actions": ["delete"]},
        }
    )
    with pytest.raises(ContractError, match="not bound to source-prod-20260804-005"):
        validate_isolated_plan(plan, spec=spec)


@pytest.mark.parametrize("actions", (["delete"], ["delete", "create"], ["update"]))
def test_plan_rejects_delete_replace_or_mutation(actions: list[str]) -> None:
    spec = _spec()
    plan = _plan(spec=spec)
    plan["resource_changes"][0]["change"]["actions"] = actions
    with pytest.raises(ContractError, match="delete/replace|non-immutable"):
        validate_isolated_plan(plan, spec=spec)


def test_plan_rejects_foreign_planned_resource_even_without_delete() -> None:
    spec = _spec()
    plan = _plan(spec=spec)
    foreign = _resource(
        'google_compute_address.worker["cppmega-corpus-99"]',
        "google_compute_address",
        "cppmega-corpus-99-source-prod-20260804-005",
        spec=spec,
    )
    plan["planned_values"]["root_module"]["resources"].append(foreign)
    with pytest.raises(ContractError, match="outside isolated run"):
        validate_isolated_plan(plan, spec=spec)


def test_plan_rejects_wrong_instance_run_root() -> None:
    spec = _spec()
    plan = _plan(spec=spec)
    instance = next(
        resource
        for resource in plan["planned_values"]["root_module"]["resources"]
        if resource["type"] == "google_compute_instance"
    )
    instance["values"]["metadata"][
        "cppmega-run-root"
    ] = "gs://natural-bison-491019-t9-cppmega-corpus/runs/source-prod-20260804-004"
    with pytest.raises(ContractError, match="cppmega-run-root"):
        validate_isolated_plan(plan, spec=spec)


def test_cli_reports_contract_error_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _main(
            [
                "--terraform-dir",
                ".",
                "--var-file",
                "missing.tfvars",
                "--output-root",
                ".",
                "--bucket",
                "natural-bison-491019-t9-cppmega-corpus",
                "--run-id",
                "source-prod-20260804-005",
                "--worker-count",
                "16",
            ]
        )
    assert exc_info.value.code == 2
    assert "Isolated GCP worker rollout failed:" in capsys.readouterr().err
