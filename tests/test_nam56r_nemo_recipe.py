"""Tests for NeMo 3 Nano-style NAM56R recipe."""

from __future__ import annotations

import pytest

from cppmega.recipes.nam56r_nemo_recipe import (
    NAM56RNeMoRecipe,
    build_nemo_hybrid_pattern,
    nam56r_author_dp_pretrain,
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
    def test_nemo_native_uses_tp2(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        tp_idx = args.index("--tensor-model-parallel-size")
        assert args[tp_idx + 1] == "2"
        assert "--sequence-parallel" in args

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

    def test_gradient_accum_fusion_disabled_without_apex(self):
        """Without APEX, gradient accumulation fusion must be disabled."""
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        assert "--no-gradient-accumulation-fusion" in args
        assert "--no-masked-softmax-fusion" not in args

    def test_moe_args_present(self):
        recipe = nam56r_author_dp_pretrain()
        args = recipe.to_args()
        assert "--num-experts" in args
        assert "--moe-router-topk" in args
        assert "--moe-grouped-gemm" in args
        assert "--moe-router-score-function" in args
        idx = args.index("--moe-router-score-function")
        assert args[idx + 1] == "sigmoid"

    def test_mla_args_in_author_mode(self):
        recipe = nam56r_author_dp_pretrain()
        args = recipe.to_args()
        assert "--multi-latent-attention" in args
        assert "--q-lora-rank" in args

    def test_no_mla_in_native_mode(self):
        recipe = nam56r_nemo_native_pretrain()
        args = recipe.to_args()
        assert "--multi-latent-attention" not in args

    def test_spec_in_author_mode(self):
        recipe = nam56r_author_dp_pretrain()
        args = recipe.to_args()
        assert "--spec" in args
        spec_idx = args.index("--spec")
        assert args[spec_idx + 1] == "cppmega.megatron.nam56r_full_spec"

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


class TestEnvDict:
    def test_base_env(self):
        recipe = NAM56RNeMoRecipe()
        env = recipe.to_env_dict()
        assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
        assert env["CPPMEGA_NEM_PATTERN"] == "AEMEAEMEAEMR"
        assert env["CPPMEGA_LAYER_DEPTH"] == "52"

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
