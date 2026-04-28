"""Compare cppmega speed runs from train logs and optional nsys CSV exports."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cppmega.recipes.run_profiles import (
    _add_common_profile_overrides,
    apply_cli_overrides,
    get_run_profile,
)
from tools.probes.speed_harness import (
    build_run_inputs,
    build_speed_comparison_report,
    render_speed_json,
    render_speed_table,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run log as LABEL=PATH. Pass multiple times to compare variants.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline LABEL. Defaults to the first --run label.",
    )
    parser.add_argument(
        "--run-profile",
        default="local_gb10_quarter",
        help="Typed cppmega RunProfile used for expected token count and validation.",
    )
    parser.add_argument("--hot-step-start", type=int, default=3)
    parser.add_argument("--hot-step-end", type=int, default=None)
    parser.add_argument(
        "--nsys-kernel-csv",
        action="append",
        default=[],
        help="Optional nsys cuda_gpu_kern_sum CSV as LABEL=PATH.",
    )
    parser.add_argument("--nsys-top-n", type=int, default=8)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    _add_common_profile_overrides(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.hot_step_start < 1:
        raise SystemExit("--hot-step-start must be >= 1")
    if args.hot_step_end is not None and args.hot_step_end < args.hot_step_start:
        raise SystemExit("--hot-step-end must be >= --hot-step-start")
    if args.nsys_top_n < 1:
        raise SystemExit("--nsys-top-n must be >= 1")

    try:
        profile = apply_cli_overrides(get_run_profile(args.run_profile), args)
        runs = build_run_inputs(args.run, nsys_kernel_csv_values=args.nsys_kernel_csv)
        report = build_speed_comparison_report(
            runs,
            profile=profile,
            baseline_label=args.baseline,
            hot_step_start=args.hot_step_start,
            hot_step_end=args.hot_step_end,
            nsys_top_n=args.nsys_top_n,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.format == "json":
        print(render_speed_json(report))
    else:
        print(render_speed_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
