from __future__ import annotations

import copy
import hashlib
import plistlib
from pathlib import Path

import pytest

from scripts import source_conveyor_supervisor as supervisor
from tests.test_source_conveyor_supervisor import _input_fixture


def _sdk(
    root: Path,
    *,
    marker_name: str = "SDKSettings.json",
    marker_payload: bytes = b'{"CanonicalName":"macosx"}\n',
) -> Path:
    sdk = root / "MacOSX.sdk"
    sdk.mkdir(parents=True)
    (sdk / marker_name).write_bytes(marker_payload)
    return sdk


def _validated_fixture(
    tmp_path: Path,
) -> tuple[Path, object, Path, dict[str, object]]:
    repo, argv = _input_fixture(tmp_path)
    sdk = _sdk(tmp_path / "xcode")
    argv[argv.index("--macos-sdk") + 1] = str(sdk)
    args = supervisor.parse_args(argv)
    inputs = supervisor.validate_inputs(args, repo_root=repo)
    return repo, args, sdk, inputs


def test_macos_sdk_is_optional_for_ordinary_production_input(tmp_path: Path) -> None:
    _repo, argv = _input_fixture(tmp_path)
    option = argv.index("--macos-sdk")
    del argv[option : option + 2]

    args = supervisor.parse_args(argv)
    assert args.macos_sdk is None


def test_macos_sdk_binding_prefers_json_and_is_forwarded(
    tmp_path: Path,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    sdk = _sdk(tmp_path / "xcode")
    (sdk / "SDKSettings.plist").write_bytes(b"plist fallback")
    argv[argv.index("--macos-sdk") + 1] = str(sdk)
    args = supervisor.parse_args(argv)

    inputs = supervisor.validate_inputs(args, repo_root=repo)
    command = supervisor.build_command(args, inputs)
    run_binding = supervisor.build_run_binding(args, inputs)
    launch = supervisor.build_launch_receipt(
        args,
        inputs=inputs,
        command=command,
        run_binding=run_binding,
        attempt=1,
    )

    marker = sdk / "SDKSettings.json"
    expected = {
        "resolved_path": str(sdk.resolve()),
        "settings": {
            "name": "SDKSettings.json",
            "resolved_path": str(marker.resolve()),
            "size_bytes": marker.stat().st_size,
            "sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
        },
    }
    assert inputs["macos_sdk"] == expected
    assert command[command.index("--macos-sdk") + 1] == str(sdk.resolve())
    assert run_binding["macos_sdk"] == expected
    assert launch["inputs"]["macos_sdk"] == expected


def test_macos_sdk_binding_uses_plist_fallback(tmp_path: Path) -> None:
    sdk = _sdk(
        tmp_path,
        marker_name="SDKSettings.plist",
        marker_payload=plistlib.dumps({"CanonicalName": "macosx99.1"}),
    )

    binding = supervisor._macos_sdk_binding(str(sdk))

    assert binding["settings"]["name"] == "SDKSettings.plist"


def test_macos_sdk_binding_rejects_unbound_or_indirect_roots(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.sdk"
    with pytest.raises(RuntimeError, match="cannot be resolved"):
        supervisor._macos_sdk_binding(str(missing))

    markerless = tmp_path / "markerless.sdk"
    markerless.mkdir()
    with pytest.raises(RuntimeError, match="no valid SDKSettings"):
        supervisor._macos_sdk_binding(str(markerless))

    sdk = _sdk(tmp_path / "xcode")
    linked_sdk = tmp_path / "linked.sdk"
    linked_sdk.symlink_to(sdk, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink components"):
        supervisor._macos_sdk_binding(str(linked_sdk))


def test_macos_sdk_binding_rejects_symlinked_or_malformed_markers(
    tmp_path: Path,
) -> None:
    sdk = tmp_path / "MacOSX.sdk"
    sdk.mkdir()
    marker_target = tmp_path / "SDKSettings.json"
    marker_target.write_text("{}\n", encoding="utf-8")
    (sdk / "SDKSettings.json").symlink_to(marker_target)
    with pytest.raises(RuntimeError, match="regular in-SDK file"):
        supervisor._macos_sdk_binding(str(sdk))

    (sdk / "SDKSettings.json").unlink()
    (sdk / "SDKSettings.json").mkdir()
    (sdk / "SDKSettings.plist").write_text("plist\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a regular file"):
        supervisor._macos_sdk_binding(str(sdk))


def test_macos_sdk_drift_is_rejected_by_resume_revalidation(
    tmp_path: Path,
) -> None:
    repo, args, sdk, inputs = _validated_fixture(tmp_path)
    launch = {
        "code_revision": args.expected_code_revision,
        "inputs": inputs,
    }

    live, revalidation_args = supervisor.revalidate_recorded_inputs(
        launch,
        run_root=Path(args.run_root),
        repo_root=repo,
    )
    assert live["macos_sdk"] == inputs["macos_sdk"]
    assert revalidation_args.macos_sdk == str(sdk.resolve())

    (sdk / "SDKSettings.json").write_text(
        '{"CanonicalName":"macosx","Version":"drift"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="source launch inputs drifted"):
        supervisor.revalidate_recorded_inputs(
            launch,
            run_root=Path(args.run_root),
            repo_root=repo,
        )


def test_targeted_repair_requires_identical_macos_sdk_binding(
    tmp_path: Path,
) -> None:
    _repo, args, _sdk_path, inputs = _validated_fixture(tmp_path)
    repair_base = {
        "inputs": copy.deepcopy(inputs),
        "target_lengths": tuple(args.target_lengths),
        "failed_repositories": ("project",),
    }
    args.only_repo = ["project"]
    supervisor.validate_repair_request(args, inputs, repair_base)

    alternate_sdk = _sdk(tmp_path / "alternate-xcode")
    alternate_inputs = copy.deepcopy(inputs)
    alternate_inputs["macos_sdk"] = supervisor._macos_sdk_binding(
        str(alternate_sdk)
    )
    with pytest.raises(RuntimeError, match="repair macOS SDK differs"):
        supervisor.validate_repair_request(args, alternate_inputs, repair_base)


def test_targeted_repair_can_bind_sdk_for_legacy_base_without_sdk(
    tmp_path: Path,
) -> None:
    _repo, args, _sdk_path, inputs = _validated_fixture(tmp_path)
    repair_base = {
        "inputs": copy.deepcopy(inputs),
        "target_lengths": tuple(args.target_lengths),
        "failed_repositories": ("project",),
    }
    repair_base["inputs"].pop("macos_sdk")
    args.only_repo = ["project"]

    # A historical full run predates the explicit SDK binding.  The repair may
    # add one, while a base that already bound an SDK remains immutable above.
    supervisor.validate_repair_request(args, inputs, repair_base)


def test_build_command_omits_unused_macos_sdk(tmp_path: Path) -> None:
    repo, argv = _input_fixture(tmp_path)
    option = argv.index("--macos-sdk")
    del argv[option : option + 2]
    args = supervisor.parse_args(argv)
    inputs = supervisor.validate_inputs(args, repo_root=repo)

    command = supervisor.build_command(args, inputs)

    assert "--macos-sdk" not in command
    assert inputs["macos_sdk"] is None
