from cppmega.recipes.run_profiles import (
    RunProfile,
    get_run_profile,
    main,
    profile_shell_assignments,
    render_shell,
    set_local_gb10_quarter_profile,
)


def test_local_gb10_profile_drives_mtp_pattern_and_num_layers():
    profile = get_run_profile("local_gb10_quarter")

    assert profile.model.mtp_depths == 2
    assert profile.hybrid_layer_pattern().endswith("/*-/*-")
    assert "--mtp-num-layers 2" in profile.native_args_fragment()


def test_h200_profile_renders_pipe_chunks_and_remote_native_overrides():
    profile = get_run_profile("h200_dsa_9_4_m")
    profile.model.moe_expert_model_parallel_size = 4
    profile.model.moe_token_dispatcher_type = "alltoall"
    profile.model.moe_router_dtype = None

    env = profile_shell_assignments(profile)

    assert env["HYBRID_LAYER_PATTERN"].count("|") == 3
    assert env["HYBRID_LAYER_PATTERN"].endswith("/*-/*-")
    assert env["CPPMEGA_MOE_TOKEN_DISPATCHER_TYPE"] == "alltoall"
    assert "--mtp-num-layers 2" in env["NATIVE_ARGS"]
    assert "--expert-model-parallel-size 4" in env["NATIVE_ARGS"]
    assert "--moe-token-dispatcher-type alltoall" in env["NATIVE_ARGS"]
    assert "--moe-router-dtype" not in env["NATIVE_ARGS"]


def test_local_gb10_profile_owns_liger_ack_and_optimizer_defaults():
    profile = get_run_profile("local_gb10_quarter")
    profile.precision.fp8_recipe = "mxfp8"
    env = profile_shell_assignments(profile)

    assert env["CPPMEGA_MTP_CE_KERNEL"] == "liger"
    assert env["CPPMEGA_I_UNDERSTAND_MTP_LIGER_CE_IS_DEPRECATED"] == "1"
    assert env["CPPMEGA_MOE_TOKEN_DISPATCHER_TYPE"] == "alltoall"
    assert env["CPPMEGA_OPTIMIZER"] == "muon"
    assert env["CPPMEGA_PARAM_STORAGE"] == "mxfp8"
    assert env["CPPMEGA_FP8_FORMAT"] == "e4m3"
    assert env["CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE"] == "int8"
    assert env["CPPMEGA_TE_MXFP8_BWD_BACKEND"] == "flashinfer_cutlass"
    assert env["CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND"] == "te"
    assert env["HYBRID_LAYER_PATTERN"].endswith("/*-/*-")
    assert "--mtp-num-layers 2" in env["NATIVE_ARGS"]


def test_profile_setter_allows_non_muon_optimizer_rendering():
    profile = RunProfile(name="test", description="test profile")
    set_local_gb10_quarter_profile(profile)
    profile.optimizer.optimizer = "adam"

    env = profile_shell_assignments(profile)

    assert env["CPPMEGA_OPTIMIZER"] == "adam"


def test_profile_can_render_pure_bf16_precision_lane():
    profile = get_run_profile("local_gb10_quarter")
    profile.precision.fp8_recipe = "off"

    env = profile_shell_assignments(profile)

    assert env["CPPMEGA_FP8_RECIPE"] == "off"
    assert env["CPPMEGA_PARAM_STORAGE"] == "bf16"
    assert "CPPMEGA_TE_MXFP8_BWD_BACKEND" not in env


def test_render_shell_quotes_profile_values():
    profile = get_run_profile("local_gb10_quarter")
    rendered = render_shell(profile)

    assert "export CPPMEGA_RUN_PROFILE=local_gb10_quarter" in rendered
    assert "export HYBRID_LAYER_PATTERN=" in rendered
    assert "export NATIVE_ARGS=" in rendered


def test_run_profile_cli_overrides_are_parameters_not_env(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_profiles",
            "shell",
            "local_gb10_quarter",
            "--train-iters",
            "1",
            "--mtp-depths",
            "1",
            "--optimizer",
            "adam",
            "--param-storage",
            "bf16",
            "--fp8-recipe",
            "mxfp8",
            "--fp8-format",
            "e4m3",
            "--mxfp8-bwd-backend",
            "cutlass_native",
            "--mxfp8-transpose-emit-backend",
            "off",
            "--fp8-param-gather",
            "--no-reuse-grad-buf-for-mxfp8-param-ag",
            "--no-mxfp8-transpose-emit-swizzled",
            "--no-mxfp8-transpose-emit-strict",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "export CPPMEGA_TRAIN_ITERS=1" in out
    assert "export MTP_DEPTHS=1" in out
    assert "export CPPMEGA_OPTIMIZER=adam" in out
    assert "export CPPMEGA_PARAM_STORAGE=bf16" in out
    assert "export CPPMEGA_FP8_FORMAT=e4m3" in out
    assert "export CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native" in out
    assert "export CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND=off" in out
    assert "export CPPMEGA_FP8_PARAM_GATHER=1" in out
    assert "export CPPMEGA_REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG=0" in out
    assert "export CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED=0" in out
    assert "export CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_STRICT=0" in out
    assert "--mtp-num-layers 1" in out


def test_local_gb10_quarter_mxfp8_defaults_to_flashinfer_cutlass():
    """The local_gb10_quarter profile defaults to TE payload + FlashInfer/CUTLASS."""
    profile = get_run_profile("local_gb10_quarter")
    assert profile.precision.mxfp8_bwd_backend == "flashinfer_cutlass"


def test_mxfp8_transpose_emit_defaults_to_te_for_tn_adapter():
    """The default MXFP8 backward path uses TE transpose emission."""
    profile = get_run_profile("local_gb10_quarter")
    profile.precision.fp8_recipe = "mxfp8"
    env = profile_shell_assignments(profile)
    assert env["CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND"] == "te"


def test_tensorwise_fp8_keeps_hybrid_format():
    profile = get_run_profile("local_gb10_quarter")
    profile.precision.fp8_recipe = "tensorwise"
    env = profile_shell_assignments(profile)

    assert env["CPPMEGA_FP8_FORMAT"] == "hybrid"


def test_mxfp8_rejects_hybrid_fp8_format(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_profiles",
            "shell",
            "local_gb10_quarter",
            "--fp8-recipe",
            "mxfp8",
            "--fp8-format",
            "hybrid",
        ],
    )

    try:
        main()
    except ValueError as exc:
        assert "requires fp8_format=e4m3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mxfp8 + hybrid fp8 format must fail")
