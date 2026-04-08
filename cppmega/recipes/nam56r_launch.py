"""Small launcher helpers that turn grounded cppmega plans into argv fragments."""

from __future__ import annotations

import argparse

from cppmega.recipes.megatron_args import MegatronArgsBundle, build_megatron_args_bundle
from cppmega.recipes.nam56r_megatron import MegatronHybridPlan, build_nam56r_feature_plan


def build_nam56r_megatron_native_args(
    *,
    plan: MegatronHybridPlan,
    enable_mla: bool = True,
    enable_mtp: bool = False,
    enable_fim: bool = False,
    enable_moe: bool = False,
    enable_dsa: bool = False,
) -> MegatronArgsBundle:
    """Return the current native Megatron feature fragment for NAM-style lanes.

    This intentionally emits only grounded built-in Megatron flags. Custom
    nanochat-only features stay in `custom_notes` until a narrow runtime seam is
    explicitly implemented.
    """

    return build_megatron_args_bundle(
        plan=plan,
        use_mla=enable_mla,
        use_mtp=enable_mtp,
        use_fim=enable_fim,
        use_moe=enable_moe,
        use_dsa=enable_dsa,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit grounded cppmega Megatron-native launcher args")
    parser.add_argument("--pattern", default="AEMEAEMEAEMR")
    parser.add_argument("--depth", type=int, default=52)
    parser.add_argument("--mtp-depths", type=int, default=0)
    parser.add_argument("--enable-mla", action="store_true")
    parser.add_argument("--enable-mtp", action="store_true")
    parser.add_argument("--enable-fim", action="store_true")
    parser.add_argument("--enable-moe", action="store_true")
    parser.add_argument("--enable-dsa", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    plan = build_nam56r_feature_plan(
        pattern=args.pattern,
        depth=args.depth,
        mtp_depths=args.mtp_depths,
    )
    bundle = build_nam56r_megatron_native_args(
        plan=plan,
        enable_mla=args.enable_mla,
        enable_mtp=args.enable_mtp,
        enable_fim=args.enable_fim,
        enable_moe=args.enable_moe,
        enable_dsa=args.enable_dsa,
    )
    print(bundle.to_shell_fragment())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
