"""Typed BF16/MXFP8 acceptance and profiler harness for local GB10 runs.

This module owns the reproducible command plan for the 20-step BF16 vs MXFP8
lane. It intentionally builds commands from ``cppmega.recipes.run_profiles``
dataclasses and CLI parameters, then uses ``compare_bf16_mxfp8`` for parsing.
The only environment variables emitted here are the rendered profile exports
and run metadata such as ``RUN_ID``/log paths.

GPU/profiler runs must be serialized with ``flock /tmp/cppmega_gpu_profile.lock``.
The default command is ``plan`` so this script is safe to run during concurrent
TE builds or agent work.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cppmega.recipes.run_profiles import (  # noqa: E402
    RunProfile,
    get_run_profile,
    profile_shell_assignments,
)
from tools.profiling.compare_bf16_mxfp8 import (  # noqa: E402
    LogInput,
    build_comparison_report,
    render_json,
    render_table,
)


DEFAULT_LOCK = Path("/tmp/cppmega_gpu_profile.lock")
DEFAULT_LAUNCHER = Path("scripts/local_gb10_quarter_train.sh")


AcceptanceLane = Literal["bf16", "mxfp8"]
ProfilerMode = Literal["none", "torch", "nsys", "ncu"]


@dataclass(frozen=True)
class AcceptanceRunSpec:
    """One typed local GB10 run command.

    ``extra_profile_args`` are forwarded to ``run_profiles`` semantics through
    dataclass fields before shell rendering when possible; keep this for new
    flags that may exist on main while this harness is reviewed from a worktree.
    """

    lane: AcceptanceLane = "bf16"
    steps: int = 20
    profile_name: str = "local_gb10_quarter"
    profiler: ProfilerMode = "none"
    hot_step_start: int = 3
    cuda_profile_step_start: int = 3
    cuda_profile_step_end: int = 4
    nsys_capture_mode: str = "full"
    nsys_trace: str = "cuda,nvtx,osrt"
    log_dir: Path = Path("/home/dave/logs")
    launcher: Path = DEFAULT_LAUNCHER
    extra_profile_args: tuple[str, ...] = ()
    run_id_prefix: str = "wave44a"

    @property
    def label(self) -> str:
        suffix = self.profiler if self.profiler != "none" else "train"
        return f"{self.lane}_{suffix}_{self.steps}step"

    def run_id(self, stamp: str) -> str:
        return f"{self.run_id_prefix}_{self.label}_{stamp}"

    def log_path(self, stamp: str) -> Path:
        return self.log_dir / f"{self.run_id(stamp)}.log"


@dataclass(frozen=True)
class AcceptanceMatrix:
    """BF16/MXFP8 run matrix for correctness, memory, and profiler artifacts."""

    bf16: AcceptanceRunSpec
    mxfp8: AcceptanceRunSpec
    profilers: tuple[ProfilerMode, ...] = ("torch", "nsys", "ncu")

    def all_specs(self) -> tuple[AcceptanceRunSpec, ...]:
        specs: list[AcceptanceRunSpec] = [self.bf16, self.mxfp8]
        for mode in self.profilers:
            specs.append(replace(self.bf16, profiler=mode))
            specs.append(replace(self.mxfp8, profiler=mode))
        return tuple(specs)


@dataclass(frozen=True)
class PlannedCommand:
    label: str
    log: Path
    command: str
    flocked_command: str
    profile_env: dict[str, str] = field(default_factory=dict)


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _profile_for_spec(spec: AcceptanceRunSpec) -> RunProfile:
    profile = get_run_profile(spec.profile_name)
    profile.training.train_iters = spec.steps
    profile.profiling.memory_debug = True
    profile.profiling.mem_profile = True
    profile.profiling.mem_profile_steps = "1,2,3,10,20" if spec.steps >= 20 else "1,2"

    if spec.lane == "bf16":
        profile.precision.fp8_recipe = "off"
        profile.precision.fp8_format = "hybrid"
        profile.optimizer.param_storage = "bf16"
    else:
        profile.precision.fp8_recipe = "mxfp8"
        profile.precision.fp8_format = "e4m3"
        profile.optimizer.param_storage = "mxfp8"
        profile.precision.mxfp8_bwd_allow_bf16_fallback = False
        profile.precision.mxfp8_dgrad_bf16 = False
        profile.precision.mxfp8_wgrad_bf16 = False
        profile.precision.mxfp8_dense_saved_operands = True
        profile.precision.mxfp8_grouped_gemm_ready_backward = True

    if spec.profiler == "torch":
        profile.profiling.torch_profile = True
    elif spec.profiler == "nsys":
        profile.profiling.nsys_profile = True
        profile.profiling.nsys_capture_mode = spec.nsys_capture_mode
        profile.profiling.nsys_trace = spec.nsys_trace
    elif spec.profiler == "ncu":
        profile.profiling.cuda_profile = True
        profile.profiling.cuda_profile_step_start = spec.cuda_profile_step_start
        profile.profiling.cuda_profile_step_end = spec.cuda_profile_step_end

    if spec.extra_profile_args:
        profile = _apply_supported_extra_args(profile, spec.extra_profile_args)
    return profile


def _apply_supported_extra_args(profile: RunProfile, args: tuple[str, ...]) -> RunProfile:
    """Apply forward-compatible profile args without shell-only switches.

    The main branch may have additional typed options while this worktree is
    isolated. We support the stable flags used by acceptance runs and fail on
    unknown options so a command plan cannot silently depend on an env-only knob.
    """

    index = 0
    while index < len(args):
        item = args[index]
        next_value = args[index + 1] if index + 1 < len(args) else None
        if item == "--mxfp8-bwd-backend" and next_value is not None:
            profile.precision.mxfp8_bwd_backend = next_value  # type: ignore[assignment]
            index += 2
        elif item == "--mxfp8-cutlass-scale-backend" and next_value is not None:
            profile.precision.mxfp8_cutlass_scale_backend = next_value  # type: ignore[assignment]
            index += 2
        elif item == "--mxfp8-flashinfer-runner" and next_value is not None:
            profile.precision.mxfp8_flashinfer_runner = next_value  # type: ignore[assignment]
            index += 2
        elif item == "--mxfp8-flashinfer-tactic" and next_value is not None:
            profile.precision.mxfp8_flashinfer_tactic = int(next_value)
            index += 2
        elif item == "--mxfp8-dense-saved-operands":
            profile.precision.mxfp8_dense_saved_operands = True
            index += 1
        elif item == "--no-mxfp8-dense-saved-operands":
            profile.precision.mxfp8_dense_saved_operands = False
            index += 1
        elif item == "--mxfp8-grouped-gemm-ready-backward":
            profile.precision.mxfp8_grouped_gemm_ready_backward = True
            index += 1
        elif item == "--no-mxfp8-grouped-gemm-ready-backward":
            profile.precision.mxfp8_grouped_gemm_ready_backward = False
            index += 1
        else:
            raise ValueError(f"unsupported extra profile arg for typed harness: {item}")
    return profile


def build_command(spec: AcceptanceRunSpec, stamp: str) -> PlannedCommand:
    profile = _profile_for_spec(spec)
    env = profile_shell_assignments(profile)
    run_id = spec.run_id(stamp)
    log = spec.log_path(stamp)
    spec.log_dir.mkdir(parents=True, exist_ok=True)

    exports = [f"export {key}={shlex.quote(value)}" for key, value in env.items()]
    exports.extend(
        [
            f"export RUN_ID={_quote(run_id)}",
            f"export CPPMEGA_LOG_DIR={_quote(spec.log_dir)}",
        ]
    )
    body = " && ".join(
        [
            "set -euo pipefail",
            *exports,
            f"{_quote(spec.launcher)} 2>&1 | tee {_quote(log)}",
        ]
    )
    command = f"bash -lc {_quote(body)}"
    flocked = f"flock {_quote(DEFAULT_LOCK)} {command}"
    return PlannedCommand(
        label=spec.label,
        log=log,
        command=command,
        flocked_command=flocked,
        profile_env=env,
    )


def build_default_matrix(steps: int, log_dir: Path, run_id_prefix: str) -> AcceptanceMatrix:
    base = AcceptanceRunSpec(steps=steps, log_dir=log_dir, run_id_prefix=run_id_prefix)
    return AcceptanceMatrix(
        bf16=replace(base, lane="bf16"),
        mxfp8=replace(base, lane="mxfp8"),
    )


def render_plan(commands: Sequence[PlannedCommand], fmt: Literal["table", "json", "shell"]) -> str:
    if fmt == "json":
        return json.dumps([asdict(command) for command in commands], indent=2, default=str)
    if fmt == "shell":
        return "\n\n".join(command.flocked_command for command in commands)

    lines = ["label | log | command", "--- | --- | ---"]
    for command in commands:
        lines.append(
            f"{command.label} | {command.log} | `{command.flocked_command}`"
        )
    return "\n".join(lines)


def run_commands(commands: Sequence[PlannedCommand]) -> int:
    for command in commands:
        print(f"[mxfp8_acceptance] running {command.label}: {command.log}", flush=True)
        result = subprocess.run(command.flocked_command, shell=True, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_plan_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--steps", type=int, default=20)
        p.add_argument("--log-dir", type=Path, default=Path("/home/dave/logs"))
        p.add_argument("--run-id-prefix", default="wave44a")
        p.add_argument("--stamp", default=None)
        p.add_argument("--format", choices=("table", "json", "shell"), default="table")

    plan = sub.add_parser("plan", help="print flocked BF16/MXFP8 commands")
    add_plan_args(plan)

    run = sub.add_parser("run", help="execute the planned commands under flock")
    add_plan_args(run)
    run.add_argument(
        "--only",
        choices=("train", "profilers", "all"),
        default="train",
        help="train runs only by default; profilers are separate CUPTI subscribers",
    )

    compare = sub.add_parser("compare", help="parse two completed train logs")
    compare.add_argument("--bf16-log", type=Path, required=True)
    compare.add_argument("--mxfp8-log", type=Path, required=True)
    compare.add_argument("--hot-step-start", type=int, default=3)
    compare.add_argument("--hot-step-end", type=int, default=None)
    compare.add_argument("--format", choices=("table", "json"), default="table")
    return parser


def _planned_from_args(args: argparse.Namespace) -> list[PlannedCommand]:
    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    matrix = build_default_matrix(
        steps=args.steps,
        log_dir=args.log_dir.expanduser(),
        run_id_prefix=args.run_id_prefix,
    )
    return [build_command(spec, stamp) for spec in matrix.all_specs()]


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "compare":
        report = build_comparison_report(
            bf16=LogInput(label="bf16", log=args.bf16_log.expanduser()),
            mxfp8=LogInput(label="mxfp8", log=args.mxfp8_log.expanduser()),
            hot_step_start=args.hot_step_start,
            hot_step_end=args.hot_step_end,
        )
        print(render_json(report) if args.format == "json" else render_table(report))
        return 0

    commands = _planned_from_args(args)
    if args.command == "plan":
        print(render_plan(commands, args.format))
        return 0

    if args.only == "train":
        commands = [command for command in commands if "_train_" in command.label]
    elif args.only == "profilers":
        commands = [command for command in commands if "_train_" not in command.label]
    return run_commands(commands)


if __name__ == "__main__":
    raise SystemExit(main())
