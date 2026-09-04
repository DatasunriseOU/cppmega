"""Frozen contracts for the dense mini 15B H200 matrix.

This module is a ledger, not a launcher. Launchers must keep
``h200_cpp_world_mini`` dense GQA unless DSA is explicitly opted in.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FAMILY = "cppmega_dense_mini_h1280_d24"
PROFILE = "h200_cpp_world_mini"
H200_RUNTIME = "megatron_h200"

# Receipt from outputs/nebius/cppmega-h200-megatron-1782697038/seq_1024_bs_192.log
H200_TOTAL_PARAMS = 625_218_594
H200_BASE_DENSE_LM = 496_825_600
H200_NGRAM_PARAMS = 128_309_472
H200_STRUCTURE_CORE_PARAMS = 83_522

# MLX train_eval_stage1.log prints "714.68M"; not a second H200 architecture.
MLX_HISTORICAL_PARAMS_LABEL = "714.68M"

TRAINED_TOKEN_STOP = 15_000_000_000
ORIGIN_QUOTAS = {
    "source": 0.55,
    "commit_pr_mr": 0.25,
    "ci_diagnostic_trajectory": 0.20,
}
CONTEXT_QUOTAS = {
    1024: 0.24,
    2048: 0.20,
    4096: 0.16,
    8192: 0.14,
    16384: 0.12,
    32768: 0.08,
    65536: 0.06,
}
DP8_GBS_MBS = {
    1024: (192, 24),
    2048: (96, 12),
    4096: (48, 6),
    8192: (24, 3),
    16384: (8, 1),
}
H200_GBS_MBS = {
    1024: (192, 192),
    2048: (96, 96),
    4096: (48, 48),
    8192: (24, 24),
    16384: (12, 12),
    32768: (6, 6),
    65536: (3, 3),
}
ABLATION_SEEDS = (17, 23, 47)
FINAL_SEED = 101
VARIANTS = ("D0", "D1", "D2", "L2", "L4")
EVAL_TASKS = (
    "body_completion_c",
    "body_completion_cpp",
    "docstring_signature",
    "fim",
    "ifim",
    "commit_repair",
    "repo_repair",
    "build_repair",
    "ci_diagnostic_localization",
    "tool_action_prediction",
)
TRAINING_MATRIX_KEYS = (
    "this_corpus_megatron_bundle",
    "case5_adapter_manifest",
    "code_stream",
    "commit_stream",
    "pr_or_mr_stream",
    "ci_stream",
)


def h200_parameter_ledger() -> dict[str, Any]:
    parts = {
        "base_dense_lm": H200_BASE_DENSE_LM,
        "ngram_hash": H200_NGRAM_PARAMS,
        "structure_core": H200_STRUCTURE_CORE_PARAMS,
    }
    total = sum(parts.values())
    if total != H200_TOTAL_PARAMS:
        raise ValueError(f"H200 ledger {total} != receipt {H200_TOTAL_PARAMS}")
    return {
        "family": FAMILY,
        "profile": PROFILE,
        "runtime": H200_RUNTIME,
        "total": H200_TOTAL_PARAMS,
        "parts": parts,
        "mlx_historical_label": MLX_HISTORICAL_PARAMS_LABEL,
    }


def dp8_micro_batch(seq: int, nproc_per_node: int) -> tuple[int, int]:
    if seq not in DP8_GBS_MBS:
        raise ValueError(f"unsupported dense-mini context {seq}")
    gbs, mbs_at_8 = DP8_GBS_MBS[seq]
    if nproc_per_node == 8:
        return gbs, mbs_at_8
    if nproc_per_node <= 0 or gbs % nproc_per_node != 0:
        raise ValueError(f"GBS {gbs} is not divisible by nproc={nproc_per_node}")
    return gbs, gbs // nproc_per_node


def h200_micro_batch(seq: int) -> tuple[int, int]:
    if seq not in H200_GBS_MBS:
        raise ValueError(f"unsupported 1xH200 context {seq}")
    return H200_GBS_MBS[seq]


def training_matrix_ready(status: Mapping[str, Any]) -> bool:
    return all(status.get(key) is True for key in TRAINING_MATRIX_KEYS)
