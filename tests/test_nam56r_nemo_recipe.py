"""Tests for NeMo 3 Nano-style NAM56R recipe."""

from __future__ import annotations

import pytest

from cppmega.recipes.nam56r_nemo_recipe import (
    NAM56RNeMoRecipe,
    build_nemo_hybrid_pattern,
    nam56r_author_dp_pretrain,
    nam56r_mamba3_te_pretrain,
    nam56r_nemo_native_max_throughput,
    nam56r_nemo_native_pretrain,
    nam56r_smoke_test,
)


class TestHybridPattern:
    def test_moe_pattern_52_layers(self):
        pattern = build_nemo_hybrid_pattern(pattern="AEMEAEMEAEMR", depth=52, use_moe=True)
        assert len(pattern) == 52
        assert pattern.count("*") == 13  # A-layers
        assert pattern.count("E") == 22  # E-layers (MoE)
        assert pattern.count("M") == 17  # M + R layers

    def test_no_moe_pattern_52_layers(self):
        pattern = build_nemo_hybrid_pattern(pattern="AEMEAEMEAEMR", depth=52, use_moe=False)
        assert len(pattern) == 52
        assert pattern.count("*") == 13
        assert pattern.count("-") == 22  # MLP-only
        assert pattern.count("M") == 17

    def test_mtp_suffix(self):
        pattern = build_nemo_hybrid_pattern(
            pattern="AEMEAEMEAEMR", depth=52, mtp_depths=1, use_moe=True
        )
        assert "/" in pattern
        assert pattern.endswith("/*-")
        main = pattern.split("/")[0]
        assert len(main) == 52

    def test_layer_order_preserved(self):
        pattern = build_nemo_hybrid_pattern(pattern="AEMEAEMEAEMR", depth=12, use_moe=True)
        assert pattern == "*EME*EME*EMM"

    def test_layer_order_no_moe(self):
        pattern = build_nemo_hybrid_pattern(pattern="AEMEAEMEAEMR", depth=12, use_moe=False)
        assert pattern == "*-M-*-M-*-MM"


class TestRecipeArgs:
    def test_nemo_native_uses_tp1_dp8(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        tp_idx = args.index("--tensor-model-parallel-size")
        assert args[tp_idx + 1] == "1"
        assert "--sequence-parallel" not in args

    def test_author_dp_uses_tp1(self):
        recipe = nam56r_author_dp_pretrain()
        args = recipe.to_args()
        tp_idx = args.index("--tensor-model-parallel-size")
        assert args[tp_idx + 1] == "1"
        assert "--sequence-parallel" not in args

    def test_distributed_optimizer_always_enabled(self):
        for recipe in [nam56r_nemo_native_pretrain(), nam56r_author_dp_pretrain()]:
            args = recipe.to_args()
            assert "--use-distributed-optimizer" in args
            assert "--overlap-grad-reduce" in args
            assert "--overlap-param-gather" in args

    def test_gradient_accum_fusion_disabled_without_apex(self):
        """Without APEX, gradient accumulation fusion must be disabled."""
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        assert "--no-gradient-accumulation-fusion" in args

    def test_nemo_throughput_features(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        assert "--cross-entropy-loss-fusion" in args
        assert "--first-last-layers-bf16" in args
        assert "--attention-backend" in args
        idx = args.index("--attention-backend")
        assert args[idx + 1] == "auto"

    def test_selective_recompute_without_cuda_graphs(self):
        """Selective recompute is enabled when CUDA graphs are off."""
        recipe = NAM56RNeMoRecipe(use_selective_recompute=True, use_cuda_graphs=False)
        args = recipe.to_args()
        assert "--recompute-granularity" in args
        idx = args.index("--recompute-granularity")
        assert args[idx + 1] == "selective"

    def test_recompute_disabled_with_cuda_graphs(self):
        """CUDA graphed attention conflicts with core_attn recompute."""
        recipe = nam56r_nemo_native_pretrain()
        assert recipe.use_cuda_graphs is True
        args = recipe.to_args()
        assert "--recompute-granularity" not in args

    def test_moe_args_present(self):
        recipe = nam56r_author_dp_pretrain()
        args = recipe.to_args()
        assert "--num-experts" in args
        assert "--moe-router-topk" in args
        assert "--moe-grouped-gemm" in args
        assert "--moe-router-score-function" in args
        idx = args.index("--moe-router-score-function")
        assert args[idx + 1] == "sigmoid"

    def test_no_mla_in_author_dp_mode(self):
        """author_dp uses upstream TE attention (not MLA) for throughput."""
        recipe = nam56r_author_dp_pretrain()
        args = recipe.to_args()
        assert "--multi-latent-attention" not in args

    def test_no_mla_in_nemo_native_mode(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        assert "--multi-latent-attention" not in args

    def test_spec_in_author_mode(self):
        recipe = nam56r_author_dp_pretrain()
        args = recipe.to_args()
        assert "--spec" in args
        spec_idx = args.index("--spec")
        assert args[spec_idx + 1] == "cppmega.megatron.nam56r_te_spec"

    def test_spec_in_native_mode(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        assert "--spec" in args
        spec_idx = args.index("--spec")
        assert args[spec_idx + 1] == "megatron.core.models.mamba.mamba_layer_specs"

    def test_mock_data_flag(self):
        recipe = nam56r_smoke_test()
        args = recipe.to_args()
        assert "--mock-data" in args

    def test_eval_iters_set(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        assert "--eval-iters" in args

    def test_bf16_precision(self):
        recipe = NAM56RNeMoRecipe(precision="bf16")
        args = recipe.to_args()
        assert "--bf16" in args

    def test_is_hybrid_model_flag(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        assert "--is-hybrid-model" in args

    def test_mamba_dims_present(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        assert "--mamba-num-heads" in args
        assert "--mamba-state-dim" in args
        assert "--mamba-head-dim" in args


class TestCudaGraphs:
    def test_nemo_native_has_cuda_graphs(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        assert "--cuda-graph-impl" in args
        idx = args.index("--cuda-graph-impl")
        assert args[idx + 1] == "transformer_engine"
        assert "--cuda-graph-scope" in args

    def test_mamba3_te_has_cuda_graphs(self):
        recipe = nam56r_mamba3_te_pretrain()
        args = recipe.to_args()
        assert "--cuda-graph-impl" in args

    def test_max_throughput_has_cuda_graphs_and_fp8(self):
        recipe = nam56r_nemo_native_max_throughput()
        args = recipe.to_args()
        assert "--cuda-graph-impl" in args
        assert "--fp8-format" in args
        idx = args.index("--fp8-recipe")
        assert args[idx + 1] == "tensorwise"
        # FP8-aligned nheads
        nheads_idx = args.index("--mamba-num-heads")
        assert args[nheads_idx + 1] == "64"

    def test_max_throughput_no_recompute(self):
        recipe = nam56r_nemo_native_max_throughput()
        args = recipe.to_args()
        assert "--recompute-granularity" not in args
        assert recipe.micro_batch_size == 5
        assert recipe.global_batch_size == 320  # 8x grad accum

    def test_max_throughput_full_moe_graph(self):
        recipe = nam56r_nemo_native_max_throughput()
        args = recipe.to_args()
        scope_idx = args.index("--cuda-graph-scope")
        scopes = []
        for a in args[scope_idx + 1:]:
            if a.startswith("--"):
                break
            scopes.append(a)
        assert "moe" in scopes  # full MoE graph, not partial
        assert "moe_router" not in scopes
        assert "--moe-expert-capacity-factor" in args
        assert "--moe-pad-expert-input-to-capacity" in args

    def test_cuda_graph_scope_values(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        scope_idx = args.index("--cuda-graph-scope")
        scopes = []
        for a in args[scope_idx + 1:]:
            if a.startswith("--"):
                break
            scopes.append(a)
        assert "attn" in scopes
        assert "mamba" in scopes
        assert "moe_router" in scopes
        assert "moe_preprocess" in scopes

    def test_smoke_test_no_cuda_graphs(self):
        recipe = nam56r_smoke_test()
        args = recipe.to_args()
        assert "--cuda-graph-impl" not in args
        assert "--optimizer-cuda-graph" not in args


class TestFP8:
    def test_fp8_uses_tensorwise(self):
        recipe = NAM56RNeMoRecipe(precision="fp8")
        args = recipe.to_args()
        assert "--fp8-format" in args
        assert "--fp8-recipe" in args
        idx = args.index("--fp8-recipe")
        assert args[idx + 1] == "tensorwise"


class TestEnvDict:
    def test_base_env(self):
        recipe = NAM56RNeMoRecipe()
        env = recipe.to_env_dict()
        assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
        assert env["CPPMEGA_NEM_PATTERN"] == "AEMEAEMEAEMR"
        assert env["CPPMEGA_LAYER_DEPTH"] == "52"

    def test_nemo_perf_env_vars(self):
        recipe = NAM56RNeMoRecipe()
        env = recipe.to_env_dict()
        assert env["NVTE_FWD_LAYERNORM_SM_MARGIN"] == "16"
        assert env["NVTE_BWD_LAYERNORM_SM_MARGIN"] == "16"
        assert env["NVTE_NORM_FWD_USE_CUDNN"] == "1"
        assert env["NCCL_AVOID_RECORD_STREAMS"] == "1"
        assert env["NCCL_GRAPH_REGISTER"] == "0"

    def test_ngram_hash_env(self):
        recipe = NAM56RNeMoRecipe(ngram_hash_enabled=True)
        env = recipe.to_env_dict()
        assert env["CPPMEGA_NGRAM_HASH_ENABLED"] == "1"

    def test_structure_env(self):
        recipe = NAM56RNeMoRecipe(structure_enabled=True)
        env = recipe.to_env_dict()
        assert env["CPPMEGA_STRUCTURE_ENABLED"] == "1"


class TestSmokeRecipe:
    def test_small_dims(self):
        recipe = nam56r_smoke_test()
        assert recipe.hidden_size == 256
        assert recipe.train_iters == 2

    def test_no_moe(self):
        recipe = nam56r_smoke_test()
        args = recipe.to_args()
        assert "--num-experts" not in args


class TestShellFragment:
    def test_produces_string(self):
        recipe = nam56r_smoke_test()
        frag = recipe.to_shell_fragment()
        assert isinstance(frag, str)
        assert "--hidden-size" in frag
