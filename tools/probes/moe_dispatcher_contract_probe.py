"""Probe cppmega's rendered MoE dispatcher contract.

This is intentionally a launch-contract probe, not an end-to-end training run.
It verifies that the typed profile emits one coherent Megatron dispatcher choice
and reports whether the local Python environment has the fused backend needed by
the Flex dispatcher.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import shlex
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from cppmega.recipes.run_profiles import get_run_profile, profile_shell_assignments


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _flag_value(args: list[str], flag: str) -> str | None:
    try:
        idx = args.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", default="local_gb10_quarter")
    parser.add_argument("--expect-dispatcher", choices=("alltoall", "allgather", "flex"))
    parser.add_argument("--strict", action="store_true", help="fail if flex backend is missing")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    env = profile_shell_assignments(get_run_profile(args.profile))
    native_args = shlex.split(env["NATIVE_ARGS"])

    dispatcher_flags = [
        native_args[i + 1]
        for i, token in enumerate(native_args[:-1])
        if token == "--moe-token-dispatcher-type"
    ]
    if len(dispatcher_flags) != 1:
        raise SystemExit(
            f"expected exactly one --moe-token-dispatcher-type, got {dispatcher_flags}"
        )

    dispatcher = dispatcher_flags[0]
    if dispatcher != env["CPPMEGA_MOE_TOKEN_DISPATCHER_TYPE"]:
        raise SystemExit(
            "dispatcher mismatch: "
            f"NATIVE_ARGS={dispatcher} env={env['CPPMEGA_MOE_TOKEN_DISPATCHER_TYPE']}"
        )
    if args.expect_dispatcher is not None and dispatcher != args.expect_dispatcher:
        raise SystemExit(f"expected dispatcher {args.expect_dispatcher}, got {dispatcher}")

    backend = _flag_value(native_args, "--moe-flex-dispatcher-backend")
    deep_ep_available = _has_module("deep_ep")

    print(f"profile={args.profile}")
    print(f"dispatcher={dispatcher}")
    print(f"flex_backend={backend or '<not-emitted>'}")
    print(f"expert_model_parallel_size={env['CPPMEGA_EP_SIZE']}")
    print(f"router_dtype={_flag_value(native_args, '--moe-router-dtype') or '<default>'}")
    print(f"permute_fusion={'--moe-permute-fusion' in native_args}")
    print(f"router_fusion={'--moe-router-fusion' in native_args}")
    print(f"deep_ep_available={deep_ep_available}")

    if args.strict and dispatcher == "flex" and backend == "deepep" and not deep_ep_available:
        raise SystemExit("strict flex/deepep probe failed: deep_ep is not importable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
