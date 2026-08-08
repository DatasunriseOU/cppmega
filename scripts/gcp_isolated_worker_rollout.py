#!/usr/bin/env python3
"""Prepare a destroy-free, run-isolated Terraform plan for GCP corpus workers.

The workers module is intentionally reusable, but an active source run must
never share its Terraform state with another source run.  This helper pins the
state backend to ``terraform/source-runs/<run_id>``, uses a separate Terraform
data directory, and rejects a plan that could mutate another run.

It deliberately has no apply subcommand.  A human can apply only the exact
binary plan after reviewing this helper's immutable local receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    load_json_object,
    require_int,
    sha256_file,
)

ROLL_OUT_RECEIPT_SCHEMA = "cppmega.gcp_isolated_worker_plan_v1"
STATE_PREFIX_ROOTS = frozenset({"terraform/source-runs", "terraform/lane-runs"})
_RUN_ID_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,26}[a-z0-9])?$")
_NAME_PREFIX_RE = re.compile(r"^[a-z]([-a-z0-9]{0,21}[a-z0-9])?$")
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")


@dataclass(frozen=True)
class RolloutSpec:
    """The immutable topology expected from one isolated workers plan."""

    bucket: str
    run_id: str
    name_prefix: str
    worker_count: int
    compact_placement: bool
    state_prefix_root: str

    @property
    def backend_prefix(self) -> str:
        return f"{self.state_prefix_root}/{self.run_id}"

    @property
    def run_root(self) -> str:
        return f"gs://{self.bucket}/runs/{self.run_id}"

    @property
    def worker_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.name_prefix}-{index:02d}" for index in range(self.worker_count)
        )


def build_rollout_spec(
    *,
    bucket: object,
    run_id: object,
    name_prefix: object,
    worker_count: object,
    compact_placement: object,
    state_prefix_root: object = "terraform/source-runs",
) -> RolloutSpec:
    """Validate the narrow inputs that bind a plan to one physical run."""

    if not isinstance(bucket, str) or _BUCKET_RE.fullmatch(bucket) is None:
        raise ContractError("bucket must be a canonical GCS bucket name")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ContractError("run_id must be a 1-28 character lowercase run identifier")
    if (
        not isinstance(name_prefix, str)
        or _NAME_PREFIX_RE.fullmatch(name_prefix) is None
    ):
        raise ContractError("name_prefix must match the workers Terraform contract")
    count = require_int(worker_count, where="worker_count", minimum=1)
    if count > 32:
        raise ContractError("worker_count must not exceed the workers Terraform limit")
    if not isinstance(compact_placement, bool):
        raise ContractError("compact_placement must be boolean")
    if compact_placement and count > 22:
        raise ContractError("compact placement supports at most 22 workers")
    if state_prefix_root not in STATE_PREFIX_ROOTS:
        raise ContractError("state_prefix_root must select source-runs or lane-runs")
    return RolloutSpec(
        bucket=bucket,
        run_id=run_id,
        name_prefix=name_prefix,
        worker_count=count,
        compact_placement=compact_placement,
        state_prefix_root=str(state_prefix_root),
    )


def backend_hcl(spec: RolloutSpec) -> str:
    """Return the exact backend configuration for this source run only."""

    return f'bucket = "{spec.bucket}"\nprefix = "{spec.backend_prefix}"\n'


def write_backend_config(path: Path, *, spec: RolloutSpec) -> bool:
    """Create a backend file once, refusing to replace a different binding."""

    expected = backend_hcl(spec)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"backend config must be a regular file: {path}")
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            raise ContractError(
                "backend config already exists with a different run binding: " f"{path}"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_stage = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    stage = Path(raw_stage)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        stage.unlink(missing_ok=True)
    return True


def _as_mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    return value


def _as_list(value: object, *, where: str) -> list[object]:
    if not isinstance(value, list):
        raise ContractError(f"{where} must be a list")
    return value


def _optional_list(value: object, *, where: str) -> list[object]:
    if value is None:
        return []
    return _as_list(value, where=where)


def _flatten_module_resources(
    module: Mapping[str, object], *, where: str
) -> Iterable[Mapping[str, object]]:
    for index, resource in enumerate(
        _optional_list(module.get("resources", []), where=f"{where}.resources")
    ):
        yield _as_mapping(resource, where=f"{where}.resources[{index}]")
    for index, child in enumerate(
        _optional_list(module.get("child_modules", []), where=f"{where}.child_modules")
    ):
        child_module = _as_mapping(child, where=f"{where}.child_modules[{index}]")
        yield from _flatten_module_resources(
            child_module, where=f"{where}.child_modules[{index}]"
        )


def _resource_address(resource: Mapping[str, object], *, where: str) -> str:
    address = resource.get("address")
    if not isinstance(address, str) or not address:
        raise ContractError(f"{where}.address must be non-empty")
    return address


def _expected_resources(spec: RolloutSpec) -> dict[str, tuple[str, str]]:
    expected: dict[str, tuple[str, str]] = {}
    for worker_name in spec.worker_names:
        expected[f'google_compute_address.worker["{worker_name}"]'] = (
            "google_compute_address",
            f"{worker_name}-{spec.run_id}",
        )
        expected[f'google_compute_instance.worker["{worker_name}"]'] = (
            "google_compute_instance",
            f"{worker_name}-{spec.run_id}",
        )
    if spec.compact_placement:
        expected["google_compute_resource_policy.compact[0]"] = (
            "google_compute_resource_policy",
            f"{spec.name_prefix}-{spec.run_id}-compact",
        )
    return expected


def _validate_target_resource(
    resource: Mapping[str, object],
    *,
    spec: RolloutSpec,
    expected: Mapping[str, tuple[str, str]],
    where: str,
) -> str:
    """Validate one managed planned/prior-state resource against this run."""

    address = _resource_address(resource, where=where)
    if resource.get("mode") != "managed":
        raise ContractError(f"{where} must be a managed resource")
    if address not in expected:
        raise ContractError(f"{where} is outside isolated run {spec.run_id}: {address}")
    expected_type, expected_name = expected[address]
    if resource.get("type") != expected_type:
        raise ContractError(f"{where}.type drifted for {address}")
    values = _as_mapping(resource.get("values"), where=f"{where}.values")
    if values.get("name") != expected_name:
        raise ContractError(f"{where}.values.name is not bound to {spec.run_id}")
    if expected_type == "google_compute_instance":
        labels = _as_mapping(values.get("labels"), where=f"{where}.values.labels")
        if labels.get("run-id") != spec.run_id:
            raise ContractError(
                f"{where}.values.labels.run-id is not bound to {spec.run_id}"
            )
        metadata = _as_mapping(values.get("metadata"), where=f"{where}.values.metadata")
        if metadata.get("cppmega-run-root") != spec.run_root:
            raise ContractError(
                f"{where}.values.metadata.cppmega-run-root is not bound to {spec.run_root}"
            )
    return address


def _plan_module(plan: Mapping[str, object], *, key: str) -> Mapping[str, object]:
    raw_section = plan.get(key)
    if key == "prior_state" and raw_section is None:
        return {"resources": [], "child_modules": []}
    section = _as_mapping(raw_section, where=key)
    if key == "prior_state":
        raw_values = section.get("values")
        if raw_values is None:
            return {"resources": [], "child_modules": []}
        section = _as_mapping(raw_values, where="prior_state.values")
    raw_root = section.get("root_module")
    if raw_root is None and key == "prior_state":
        return {"resources": [], "child_modules": []}
    return _as_mapping(raw_root, where=f"{key}.root_module")


def validate_isolated_plan(
    plan: Mapping[str, object], *, spec: RolloutSpec
) -> dict[str, object]:
    """Fail closed unless the plan creates/no-ops only this run's workers.

    A state accidentally pointed at `.004` produces either foreign prior-state
    resources or delete actions when planning `.005`; both are rejected before
    anyone can apply the generated binary plan.
    """

    expected = _expected_resources(spec)
    planned = _plan_module(plan, key="planned_values")
    planned_resources = [
        resource
        for resource in _flatten_module_resources(
            planned, where="planned_values.root_module"
        )
        if resource.get("mode") == "managed"
    ]
    planned_addresses = {
        _validate_target_resource(
            resource,
            spec=spec,
            expected=expected,
            where=f"planned managed resource[{index}]",
        )
        for index, resource in enumerate(planned_resources)
    }
    if len(planned_addresses) != len(planned_resources):
        raise ContractError("planned managed resources contain duplicate addresses")
    if planned_addresses != set(expected):
        raise ContractError(
            "planned managed resources do not exactly match the isolated run: "
            f"missing={sorted(set(expected) - planned_addresses)} "
            f"extra={sorted(planned_addresses - set(expected))}"
        )

    prior = _plan_module(plan, key="prior_state")
    for index, resource in enumerate(
        _flatten_module_resources(prior, where="prior_state.values.root_module")
    ):
        if resource.get("mode") == "managed":
            _validate_target_resource(
                resource,
                spec=spec,
                expected=expected,
                where=f"prior managed resource[{index}]",
            )

    changes = _optional_list(plan.get("resource_changes", []), where="resource_changes")
    actions_by_address: dict[str, list[str]] = {}
    for index, raw_change in enumerate(changes):
        change = _as_mapping(raw_change, where=f"resource_changes[{index}]")
        if change.get("mode") != "managed":
            continue
        address = _resource_address(change, where=f"resource_changes[{index}]")
        if address not in expected:
            raise ContractError(
                f"resource_changes[{index}] is outside isolated run {spec.run_id}: {address}"
            )
        delta = _as_mapping(
            change.get("change"), where=f"resource_changes[{index}].change"
        )
        actions = _as_list(
            delta.get("actions"), where=f"resource_changes[{index}].change.actions"
        )
        if not all(isinstance(action, str) for action in actions):
            raise ContractError(
                f"resource_changes[{index}].change.actions must contain strings"
            )
        normalized_actions = [str(action) for action in actions]
        if "delete" in normalized_actions:
            raise ContractError(
                f"isolated plan contains a delete/replace action for {address}"
            )
        if normalized_actions not in (["create"], ["no-op"]):
            raise ContractError(
                f"isolated plan contains a non-immutable action for {address}: "
                f"{normalized_actions}"
            )
        if address in actions_by_address:
            raise ContractError(f"duplicate plan change for {address}")
        actions_by_address[address] = normalized_actions

    return {
        "schema": ROLL_OUT_RECEIPT_SCHEMA,
        "status": "validated_no_destroy_not_applied",
        "run_id": spec.run_id,
        "backend": {"bucket": spec.bucket, "prefix": spec.backend_prefix},
        "expected_managed_resources": sorted(expected),
        "resource_actions": actions_by_address,
    }


def _validate_backend_metadata(tf_data_dir: Path, *, spec: RolloutSpec) -> None:
    """Verify init actually selected the generated remote backend, not `.004`."""

    _, state = load_json_object(
        tf_data_dir / "terraform.tfstate", where="Terraform backend metadata"
    )
    backend = _as_mapping(
        state.get("backend"), where="Terraform backend metadata.backend"
    )
    if backend.get("type") != "gcs":
        raise ContractError("Terraform backend is not GCS")
    config = _as_mapping(
        backend.get("config"), where="Terraform backend metadata.backend.config"
    )
    if (
        config.get("bucket") != spec.bucket
        or config.get("prefix") != spec.backend_prefix
    ):
        raise ContractError(
            "Terraform backend metadata is not bound to the requested isolated run"
        )


def _regular_file(path: Path, *, where: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ContractError(f"{where} must be a regular file: {path}")
    return resolved


def _empty_target(path: Path, *, where: str) -> Path:
    if path.exists() or path.is_symlink():
        raise ContractError(f"{where} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_checked(
    command: Sequence[str], *, env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command!r}\n"
            f"stdout:\n{(completed.stdout or '')[-8000:]}\n"
            f"stderr:\n{(completed.stderr or '')[-8000:]}"
        )
    return completed


_GOOGLE_CREDENTIAL_ENV_VARS = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CREDENTIALS",
    "GOOGLE_BACKEND_CREDENTIALS",
    "GOOGLE_IMPERSONATE_SERVICE_ACCOUNT",
    "GOOGLE_BACKEND_IMPERSONATE_SERVICE_ACCOUNT",
    "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
)


def _gcloud_access_token() -> str:
    """Read one short-lived token from the active gcloud account in memory."""

    environment = dict(os.environ)
    environment.pop("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", None)
    completed = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "gcloud auth print-access-token failed:\n"
            f"{(completed.stderr or '')[-4000:]}"
        )
    token = completed.stdout.strip()
    if not token or any(character.isspace() for character in token):
        raise ContractError("gcloud returned an empty or malformed OAuth token")
    return token


def _plan_environment(*, use_gcloud_access_token: bool) -> tuple[dict[str, str], str]:
    environment = dict(os.environ)
    if not use_gcloud_access_token:
        return environment, "ambient"
    token = _gcloud_access_token()
    for variable in _GOOGLE_CREDENTIAL_ENV_VARS:
        environment.pop(variable, None)
    environment["GOOGLE_OAUTH_ACCESS_TOKEN"] = token
    return environment, "gcloud-active-token"


def run_guarded_plan(
    *,
    terraform_dir: Path,
    var_file: Path,
    output_root: Path,
    spec: RolloutSpec,
    use_gcloud_access_token: bool = False,
) -> dict[str, object]:
    """Initialize an isolated backend, create a binary plan, and attest it."""

    terraform_dir = terraform_dir.resolve()
    if not terraform_dir.is_dir():
        raise ContractError(f"terraform_dir must be a directory: {terraform_dir}")
    var_file = _regular_file(var_file, where="var_file")
    output_root = output_root.resolve()
    rollout_root = output_root / "isolated-terraform"
    backend_path = rollout_root / f"{spec.run_id}.backend.hcl"
    tf_data_dir = rollout_root / f"terraform-data-{spec.run_id}"
    plan_path = rollout_root / f"{spec.run_id}.isolated.tfplan"
    plan_json_path = rollout_root / f"{spec.run_id}.isolated.tfplan.json"
    receipt_path = rollout_root / f"{spec.run_id}.isolated-plan-receipt.json"

    write_backend_config(backend_path, spec=spec)
    _empty_target(plan_path, where="isolated Terraform plan")
    _empty_target(plan_json_path, where="isolated Terraform plan JSON")
    _empty_target(receipt_path, where="isolated Terraform plan receipt")
    tf_data_dir.mkdir(parents=True, exist_ok=True)

    environment, auth_mode = _plan_environment(
        use_gcloud_access_token=use_gcloud_access_token
    )
    environment["TF_DATA_DIR"] = str(tf_data_dir)
    init = _run_checked(
        [
            "terraform",
            f"-chdir={terraform_dir}",
            "init",
            "-reconfigure",
            "-input=false",
            f"-backend-config={backend_path}",
        ],
        env=environment,
    )
    _validate_backend_metadata(tf_data_dir, spec=spec)
    try:
        _run_checked(
            [
                "terraform",
                f"-chdir={terraform_dir}",
                "plan",
                "-input=false",
                "-lock-timeout=5m",
                f"-var-file={var_file}",
                f"-out={plan_path}",
            ],
            env=environment,
        )
        shown = _run_checked(
            ["terraform", f"-chdir={terraform_dir}", "show", "-json", str(plan_path)],
            env=environment,
        )
        try:
            plan = json.loads(shown.stdout)
        except json.JSONDecodeError as exc:
            raise ContractError("terraform show did not return JSON") from exc
        if not isinstance(plan, Mapping):
            raise ContractError("terraform show JSON must be an object")
        receipt = validate_isolated_plan(plan, spec=spec)
        plan_json_path.write_text(shown.stdout, encoding="utf-8", newline="\n")
        receipt.update(
            {
                "terraform_init_stdout_sha256": hashlib.sha256(
                    init.stdout.encode("utf-8")
                ).hexdigest(),
                "plan_path": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
                "plan_json_path": str(plan_json_path),
                "plan_json_sha256": sha256_file(plan_json_path),
                "backend_config_path": str(backend_path),
                "backend_config_sha256": sha256_file(backend_path),
                "tf_data_dir": str(tf_data_dir),
                "auth_mode": auth_mode,
            }
        )
        atomic_write_json(receipt_path, receipt)
    except BaseException:
        plan_path.unlink(missing_ok=True)
        plan_json_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        raise
    return {**receipt, "receipt_path": str(receipt_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terraform-dir", required=True, type=Path)
    parser.add_argument("--var-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name-prefix", default="cppmega-corpus")
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument(
        "--state-prefix-root",
        choices=sorted(STATE_PREFIX_ROOTS),
        default="terraform/source-runs",
    )
    parser.add_argument(
        "--use-gcloud-access-token",
        action="store_true",
        help=(
            "use a short-lived token from the active gcloud account for both "
            "the GCS backend and provider; the token is not written to the receipt"
        ),
    )
    parser.add_argument(
        "--compact-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="expect the compact placement policy (default: enabled)",
    )
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        spec = build_rollout_spec(
            bucket=args.bucket,
            run_id=args.run_id,
            name_prefix=args.name_prefix,
            worker_count=args.worker_count,
            compact_placement=args.compact_placement,
            state_prefix_root=args.state_prefix_root,
        )
        receipt = run_guarded_plan(
            terraform_dir=args.terraform_dir,
            var_file=args.var_file,
            output_root=args.output_root,
            spec=spec,
            use_gcloud_access_token=args.use_gcloud_access_token,
        )
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"Isolated GCP worker rollout failed: {exc}\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
