"""Small launcher helpers that turn grounded cppmega plans into argv fragments."""

from __future__ import annotations

import argparse

from cppmega.recipes.megatron_args import MegatronArgsBundle, build_megatron_args_bundle
from cppmega.recipes.nam56r_megatron import (
    MegatronHybridPlan,
    build_nam56r_feature_plan,
    parse_nem_pattern,
)


def build_nam56r_megatron_native_args(
    *,
    plan: MegatronHybridPlan,
    enable_mla: bool = True,
    enable_mtp: bool = False,
    mtp_mode: str = "gpt",
    mtp_num_predictors: int = 1,
    enable_fim: bool = False,
    enable_moe: bool = False,
    moe_expert_model_parallel_size: int = 1,
    moe_token_dispatcher_type: str = "alltoall",
    moe_flex_dispatcher_backend: str = "deepep",
    moe_router_dtype: str | None = "fp32",
    enable_dsa: bool = False,
    dsa_indexer_dtype: str = "bf16",
    dsa_indexer_topk: int = 256,
    dsa_indexer_loss_coeff: float = 0.001,
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
        mtp_mode=mtp_mode,
        mtp_num_predictors=mtp_num_predictors,
        use_fim=enable_fim,
        use_moe=enable_moe,
        moe_expert_model_parallel_size=moe_expert_model_parallel_size,
        moe_token_dispatcher_type=moe_token_dispatcher_type,
        moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
        moe_router_dtype=moe_router_dtype,
        use_dsa=enable_dsa,
        dsa_indexer_dtype=dsa_indexer_dtype,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
    )


def build_nam56r_lite_main_pattern(
    *,
    pattern: str,
    depth: int,
    mtp_depths: int = 0,
    use_dsa_symbol: bool = False,
) -> str:
    """Build Megatron hybrid_layer_pattern from NAM56R symbols.

    When *use_dsa_symbol* is True (Megatron has PR #3553), ALL attention
    layers (``A`` in the nanochat pattern) emit ``D`` instead of ``*``.
    PR #3553 blocks mixing ``*`` and ``D`` in the same pattern, so we must
    use one symbol for all attention layers.  The cppmega
    ``CppMegaSelectiveAttentionLayer`` routes DSA vs MLA internally based on
    ``CPPMEGA_DSA_A_LAYER_RANKS``, so both DSA and MLA layers can share the
    same Megatron symbol.
    """
    attn_symbol = "D" if use_dsa_symbol else "*"

    mapped: list[str] = []
    for symbol in parse_nem_pattern(pattern, depth):
        if symbol == "A":
            mapped.append(attn_symbol)
        elif symbol == "E":
            mapped.append("E")
        elif symbol in {"M", "R"}:
            mapped.append("M")
        elif symbol in {"D", "G"}:
            mapped.append("G")
        else:
            raise ValueError(f"unsupported symbol for NAM56R-lite pattern: {symbol!r}")

    result = "".join(mapped)
    if mtp_depths > 0:
        result = result + "/" + "/".join(f"{attn_symbol}-" for _ in range(mtp_depths))
    return result


def get_custom_layer_indices(
    *,
    pattern: str,
    depth: int,
    custom_symbols: tuple[str, ...] = ("R",),
) -> tuple[int, ...]:
    symbols = frozenset(custom_symbols)
    return tuple(
        index + 1
        for index, symbol in enumerate(parse_nem_pattern(pattern, depth))
        if symbol in symbols
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit grounded cppmega Megatron-native launcher args")
    parser.add_argument("--pattern", default="AEMEAEMEAEMR")
    parser.add_argument("--depth", type=int, default=52)
    parser.add_argument("--mtp-depths", type=int, default=0)
    parser.add_argument(
        "--mtp-num-predictors",
        type=int,
        default=None,
        help="MTP predictor count for --mtp-num-layers; defaults to --mtp-depths when set",
    )
    parser.add_argument("--enable-mla", action="store_true")
    parser.add_argument("--enable-mtp", action="store_true")
    parser.add_argument("--mtp-mode", choices=("gpt", "hybrid"), default="gpt")
    parser.add_argument("--enable-fim", action="store_true")
    parser.add_argument("--enable-moe", action="store_true")
    parser.add_argument("--moe-expert-model-parallel-size", type=int, default=1)
    parser.add_argument(
        "--moe-token-dispatcher-type",
        choices=("flex", "alltoall", "allgather"),
        default="alltoall",
    )
    parser.add_argument(
        "--moe-flex-dispatcher-backend",
        choices=("deepep", "hybridep"),
        default="deepep",
    )
    parser.add_argument(
        "--moe-router-dtype",
        choices=("fp32", "none"),
        default="fp32",
        help="Router probability dtype flag; 'none' omits --moe-router-dtype",
    )
    parser.add_argument("--enable-dsa", action="store_true")
    parser.add_argument(
        "--dsa-indexer-dtype",
        choices=("bf16",),
        default="bf16",
        help="DSA indexer q@k^T compute dtype (bf16 only; FP8 indexer path was removed)",
    )
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
        mtp_mode=args.mtp_mode,
        mtp_num_predictors=(
            args.mtp_num_predictors
            if args.mtp_num_predictors is not None
            else max(args.mtp_depths, 1)
        ),
        enable_fim=args.enable_fim,
        enable_moe=args.enable_moe,
        moe_expert_model_parallel_size=args.moe_expert_model_parallel_size,
        moe_token_dispatcher_type=args.moe_token_dispatcher_type,
        moe_flex_dispatcher_backend=args.moe_flex_dispatcher_backend,
        moe_router_dtype=None if args.moe_router_dtype == "none" else args.moe_router_dtype,
        enable_dsa=args.enable_dsa,
        dsa_indexer_dtype=args.dsa_indexer_dtype,
    )
    print(bundle.to_shell_fragment())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
