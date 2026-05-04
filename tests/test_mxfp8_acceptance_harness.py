from pathlib import Path

from tools.profiling.mxfp8_acceptance_harness import (
    AcceptanceRunSpec,
    build_command,
    build_default_matrix,
    render_plan,
)


def test_acceptance_harness_builds_typed_bf16_and_mxfp8_commands(tmp_path):
    stamp = "20260501_000000"
    matrix = build_default_matrix(steps=20, log_dir=tmp_path, run_id_prefix="unit")

    bf16 = build_command(matrix.bf16, stamp)
    mxfp8 = build_command(matrix.mxfp8, stamp)

    assert bf16.label == "bf16_train_20step"
    assert mxfp8.label == "mxfp8_train_20step"
    assert bf16.log == tmp_path / "unit_bf16_train_20step_20260501_000000.log"
    assert mxfp8.log == tmp_path / "unit_mxfp8_train_20step_20260501_000000.log"

    assert bf16.profile_env["CPPMEGA_FP8_RECIPE"] == "off"
    assert bf16.profile_env["CPPMEGA_PARAM_STORAGE"] == "bf16"
    assert bf16.profile_env["CPPMEGA_TRAIN_ITERS"] == "20"
    assert bf16.profile_env["CPPMEGA_MEMORY_DEBUG"] == "1"

    assert mxfp8.profile_env["CPPMEGA_FP8_RECIPE"] == "mxfp8"
    assert mxfp8.profile_env["CPPMEGA_FP8_FORMAT"] == "e4m3"
    assert mxfp8.profile_env["CPPMEGA_PARAM_STORAGE"] == "mxfp8"
    assert mxfp8.profile_env["CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK"] == "0"
    assert mxfp8.profile_env["CPPMEGA_TE_MXFP8_DGRAD_BF16"] == "0"
    assert mxfp8.profile_env["CPPMEGA_TE_MXFP8_WGRAD_BF16"] == "0"
    assert mxfp8.profile_env["CPPMEGA_TE_MXFP8_DENSE_SAVED_OPERANDS"] == "1"
    assert mxfp8.profile_env["CPPMEGA_TE_MXFP8_GROUPED_GEMM_READY_BACKWARD"] == "1"

    assert "flock /tmp/cppmega_gpu_profile.lock" in mxfp8.flocked_command
    assert "scripts/local_gb10_quarter_train.sh" in mxfp8.flocked_command
    assert "export CPPMEGA_FP8_RECIPE=mxfp8" in mxfp8.command


def test_acceptance_harness_profiler_modes_are_separate_commands(tmp_path):
    matrix = build_default_matrix(steps=20, log_dir=tmp_path, run_id_prefix="unit")
    commands = [build_command(spec, "stamp") for spec in matrix.all_specs()]
    labels = [command.label for command in commands]

    assert labels == [
        "bf16_train_20step",
        "mxfp8_train_20step",
        "bf16_torch_20step",
        "mxfp8_torch_20step",
        "bf16_nsys_20step",
        "mxfp8_nsys_20step",
        "bf16_ncu_20step",
        "mxfp8_ncu_20step",
    ]
    assert commands[2].profile_env["CPPMEGA_TORCH_PROFILE"] == "1"
    assert commands[4].profile_env["CPPMEGA_NSYS_PROFILE"] == "1"
    assert commands[6].profile_env["CPPMEGA_CUDA_PROFILE"] == "1"
    assert commands[6].profile_env["CPPMEGA_CUDA_PROFILE_STEP_START"] == "3"
    assert commands[6].profile_env["CPPMEGA_CUDA_PROFILE_STEP_END"] == "4"


def test_acceptance_harness_rejects_unknown_extra_profile_args(tmp_path):
    spec = AcceptanceRunSpec(
        lane="mxfp8",
        log_dir=tmp_path,
        extra_profile_args=("--env-only-switch",),
    )

    try:
        build_command(spec, "stamp")
    except ValueError as exc:
        assert "unsupported extra profile arg" in str(exc)
    else:  # pragma: no cover - assertion failure path.
        raise AssertionError("unknown extra args must fail closed")


def test_acceptance_harness_renders_shell_plan(tmp_path):
    command = build_command(
        AcceptanceRunSpec(lane="bf16", steps=2, log_dir=tmp_path),
        "stamp",
    )

    shell = render_plan([command], "shell")

    assert shell.startswith("flock /tmp/cppmega_gpu_profile.lock")
    assert str(Path("scripts/local_gb10_quarter_train.sh")) in shell
