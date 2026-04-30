from pathlib import Path

import pytest

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
    assert env["CPPMEGA_MOE_FLEX_DISPATCHER_BACKEND"] == "deepep"
    assert "--mtp-num-layers 2" in env["NATIVE_ARGS"]
    assert "--expert-model-parallel-size 4" in env["NATIVE_ARGS"]
    assert "--moe-token-dispatcher-type alltoall" in env["NATIVE_ARGS"]
    assert "--moe-router-dtype" not in env["NATIVE_ARGS"]


def test_profile_can_render_explicit_flex_backend_for_ep_lanes():
    profile = get_run_profile("h200_dsa_9_4_m")
    profile.model.moe_expert_model_parallel_size = 4
    profile.model.moe_token_dispatcher_type = "flex"
    profile.model.moe_flex_dispatcher_backend = "hybridep"

    env = profile_shell_assignments(profile)

    assert env["CPPMEGA_MOE_TOKEN_DISPATCHER_TYPE"] == "flex"
    assert env["CPPMEGA_MOE_FLEX_DISPATCHER_BACKEND"] == "hybridep"
    assert "--moe-token-dispatcher-type flex" in env["NATIVE_ARGS"]
    assert "--moe-flex-dispatcher-backend hybridep" in env["NATIVE_ARGS"]
    assert "--moe-router-dtype fp32" in env["NATIVE_ARGS"]


def test_profile_rejects_flex_without_expert_parallelism():
    profile = get_run_profile("local_gb10_quarter")
    profile.model.moe_token_dispatcher_type = "flex"

    with pytest.raises(ValueError, match="requires moe_expert_model_parallel_size > 1"):
        profile_shell_assignments(profile)


def test_profile_rejects_flex_without_fp32_router_probs():
    profile = get_run_profile("h200_dsa_9_4_m")
    profile.model.moe_expert_model_parallel_size = 4
    profile.model.moe_token_dispatcher_type = "flex"
    profile.model.moe_router_dtype = None

    with pytest.raises(ValueError, match="requires moe_router_dtype=fp32"):
        profile_shell_assignments(profile)


def test_local_gb10_profile_owns_cce_mtp_and_optimizer_defaults():
    profile = get_run_profile("local_gb10_quarter")
    profile.precision.fp8_recipe = "mxfp8"
    env = profile_shell_assignments(profile)

    assert env["CPPMEGA_MTP_CE_KERNEL"] == "cce"
    assert "CPPMEGA_I_UNDERSTAND_MTP_LIGER_CE_IS_DEPRECATED" not in env
    assert env["CPPMEGA_MOE_TOKEN_DISPATCHER_TYPE"] == "alltoall"
    assert env["CPPMEGA_USE_FLASH_ATTN"] == "1"
    assert env["CPPMEGA_ATTN_BACKEND"] == "flash"
    assert env["CPPMEGA_EXTRA_PYTHONPATH"].split(":")[:2] == [
        "/home/dave/flash-attention-fa4",
        "/home/dave/TransformerEngine",
    ]
    assert env["CPPMEGA_FLASH_ATTN_SOURCE_ROOT"] == "/home/dave/flash-attention-fa4"
    assert env["CPPMEGA_TRANSFORMER_ENGINE_SOURCE_ROOT"] == "/home/dave/TransformerEngine"
    assert env["CPPMEGA_NOCONV_MAMBA_CHUNK_SIZE"] == "256"
    assert env["CPPMEGA_CCE_FUSE_MAIN_MTP_CE"] == "1"
    assert env["CPPMEGA_CCE_FILTER_EPS"] == "high"
    assert env["CPPMEGA_DSA_FP8_ATTENTION"] == "0"
    assert "--moe-token-dispatcher-type alltoall" in env["NATIVE_ARGS"]
    assert env["CPPMEGA_OPTIMIZER"] == "muon"
    assert env["CPPMEGA_PARAM_STORAGE"] == "mxfp8"
    assert env["CPPMEGA_FP8_FORMAT"] == "e4m3"
    assert env["CPPMEGA_MUON_NUM_NS_STEPS"] == "3"
    assert env["CPPMEGA_MUON_NS_CARRIER"] == "bf16"
    assert env["CPPMEGA_MUON_DTYPE_AUDIT"] == "0"
    assert env["CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE"] == "int8"
    assert env["CPPMEGA_TE_MXFP8_BWD_BACKEND"] == "te_tn_adapter"
    assert env["CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND"] == "te"
    assert env["CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND"] == "compact"
    assert env["CPPMEGA_TE_MXFP8_COMPACT_COLUMNWISE_BACKWARD"] == "0"
    assert env["CPPMEGA_TE_MXFP8_DENSE_SAVED_OPERANDS"] == "1"
    assert env["CPPMEGA_FLASHINFER_MXFP8_RUNNER"] == "mm_mxfp8"
    assert env["CPPMEGA_FLASHINFER_MXFP8_TACTIC"] == "0"
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
            "--muon-num-ns-steps",
            "5",
            "--fp8-recipe",
            "mxfp8",
            "--fp8-format",
            "e4m3",
            "--attention-backend",
            "fused",
            "--mxfp8-bwd-backend",
            "cutlass_native",
            "--mxfp8-transpose-emit-backend",
            "off",
            "--mxfp8-compact-columnwise-backward",
            "--no-mxfp8-dense-saved-operands",
            "--no-mxfp8-grouped-gemm-ready-backward",
            "--fp8-param-gather",
            "--no-reuse-grad-buf-for-mxfp8-param-ag",
            "--no-mxfp8-transpose-emit-swizzled",
            "--no-mxfp8-transpose-emit-strict",
            "--nsys-capture-mode",
            "delay",
            "--nsys-delay",
            "15",
            "--nsys-duration",
            "5",
            "--nsys-trace",
            "cuda,nvtx,osrt",
            "--noconv-mamba-chunk-size",
            "128",
            "--cce-fuse-main-mtp-ce",
            "--cce-filter-eps",
            "high",
            "--mxfp8-flashinfer-runner",
            "direct_tactic",
            "--mxfp8-flashinfer-tactic",
            "2",
            "--muon-ns-carrier",
            "mxfp8_probe",
            "--muon-dtype-audit",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "export CPPMEGA_TRAIN_ITERS=1" in out
    assert "export MTP_DEPTHS=1" in out
    assert "export CPPMEGA_OPTIMIZER=adam" in out
    assert "export CPPMEGA_PARAM_STORAGE=bf16" in out
    assert "export CPPMEGA_MUON_NUM_NS_STEPS=5" in out
    assert "export CPPMEGA_MUON_NS_CARRIER=mxfp8_probe" in out
    assert "export CPPMEGA_MUON_DTYPE_AUDIT=1" in out
    assert "export CPPMEGA_FP8_FORMAT=e4m3" in out
    assert "export CPPMEGA_ATTN_BACKEND=fused" in out
    assert "export CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native" in out
    assert "export CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND=off" in out
    assert "export CPPMEGA_TE_MXFP8_COMPACT_COLUMNWISE_BACKWARD=1" in out
    assert "export CPPMEGA_TE_MXFP8_DENSE_SAVED_OPERANDS=0" in out
    assert "export CPPMEGA_TE_MXFP8_GROUPED_GEMM_READY_BACKWARD=0" in out
    assert "export CPPMEGA_FP8_PARAM_GATHER=1" in out
    assert "export CPPMEGA_REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG=0" in out
    assert "export CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED=0" in out
    assert "export CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_STRICT=0" in out
    assert "export CPPMEGA_NSYS_CAPTURE_MODE=delay" in out
    assert "export CPPMEGA_NSYS_TRACE=cuda,nvtx,osrt" in out
    assert "export CPPMEGA_NSYS_DELAY=15" in out
    assert "export CPPMEGA_NSYS_DURATION=5" in out
    assert "export CPPMEGA_NOCONV_MAMBA_CHUNK_SIZE=128" in out
    assert "export CPPMEGA_CCE_FUSE_MAIN_MTP_CE=1" in out
    assert "export CPPMEGA_CCE_FILTER_EPS=high" in out
    assert "export CPPMEGA_FLASHINFER_MXFP8_RUNNER=direct_tactic" in out
    assert "export CPPMEGA_FLASHINFER_MXFP8_TACTIC=2" in out
    assert "--mtp-num-layers 1" in out


def test_compact_columnwise_cli_selects_cutlass_native_when_backend_implicit(
    capsys,
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_profiles",
            "shell",
            "local_gb10_quarter",
            "--fp8-recipe",
            "mxfp8",
            "--mxfp8-compact-columnwise-backward",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "export CPPMEGA_TE_MXFP8_COMPACT_COLUMNWISE_BACKWARD=1" in out
    assert "export CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native" in out


def test_compact_columnwise_rejects_non_cutlass_backend(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_profiles",
            "shell",
            "local_gb10_quarter",
            "--fp8-recipe",
            "mxfp8",
            "--mxfp8-bwd-backend",
            "flashinfer_cutlass",
            "--mxfp8-compact-columnwise-backward",
        ],
    )

    with pytest.raises(ValueError, match="requires --mxfp8-bwd-backend cutlass_native"):
        main()


def test_swizzled_cutlass_scale_cli_selects_cutlass_native(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_profiles",
            "shell",
            "local_gb10_quarter",
            "--fp8-recipe",
            "mxfp8",
            "--mxfp8-cutlass-scale-backend",
            "swizzled",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "export CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native" in out
    assert "export CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND=swizzled" in out
    assert "export CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND=te" in out
    assert "export CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED=1" in out
    assert "export CPPMEGA_TE_MXFP8_COMPACT_COLUMNWISE_BACKWARD=0" in out


def test_swizzled_cutlass_scale_rejects_non_cutlass_backend(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_profiles",
            "shell",
            "local_gb10_quarter",
            "--fp8-recipe",
            "mxfp8",
            "--mxfp8-bwd-backend",
            "flashinfer_cutlass",
            "--mxfp8-cutlass-scale-backend",
            "swizzled",
        ],
    )

    with pytest.raises(ValueError, match="requires --mxfp8-bwd-backend cutlass_native"):
        main()


def test_nsys_profile_defaults_to_full_capture_not_cuda_profiler_api():
    profile = get_run_profile("local_gb10_quarter")
    profile.profiling.nsys_profile = True

    env = profile_shell_assignments(profile)

    assert env["CPPMEGA_NSYS_CAPTURE_MODE"] == "full"
    assert env["CPPMEGA_NSYS_TRACE"] == "cuda-sw,nvtx,osrt"


def test_nsys_delay_can_capture_until_normal_process_exit():
    profile = get_run_profile("local_gb10_quarter")
    profile.profiling.nsys_profile = True
    profile.profiling.nsys_capture_mode = "delay"
    profile.profiling.nsys_delay = 115
    profile.profiling.nsys_duration = 0

    env = profile_shell_assignments(profile)
    script = Path("scripts/local_gb10_quarter_train.sh").read_text()

    assert env["CPPMEGA_NSYS_CAPTURE_MODE"] == "delay"
    assert env["CPPMEGA_NSYS_DELAY"] == "115"
    assert env["CPPMEGA_NSYS_DURATION"] == "0"
    assert 'cmd+=("--delay=${CPPMEGA_NSYS_DELAY}")' in script
    assert 'if [[ "${CPPMEGA_NSYS_DURATION}" -gt 0 ]]' in script


def test_local_launcher_lets_native_args_own_moe_dispatcher_flag():
    script = Path("scripts/local_gb10_quarter_train.sh").read_text()

    assert '${NATIVE_ARGS} \\' in script
    assert '--attention-backend "${CPPMEGA_ATTN_BACKEND}"' in script
    assert 'FLASH_ATTN_ARGS+=(--use-flash-attn)' in script
    assert "--attention-backend flash" not in script
    assert '--moe-token-dispatcher-type "${CPPMEGA_MOE_TOKEN_DISPATCHER_TYPE}"' not in script
    assert "apply_moe_dispatcher_identity_sort_patch()" in script
    assert "CPPMEGA_MUON_NUM_NS_STEPS:-3" in script
    assert "CPPMEGA_MUON_NS_CARRIER:-bf16" in script
    assert "CPPMEGA_MUON_DTYPE_AUDIT:-0" in script
    assert "CPPMEGA_CCE_FILTER_EPS:-none" in script
    assert "install_muon_dtype_audit" in script
    assert "CPPMEGA_EXTRA_PYTHONPATH" in script
    assert "flash_attn import:" in script


def test_fp8_shim_bridges_noconv_mamba_chunk_config():
    shim = Path("scripts/cppmega_fp8_shim.py").read_text()

    assert "CPPMEGA_NOCONV_MAMBA_CHUNK_SIZE" in shim
    assert "cppmega_noconv_mamba_chunk_size" in shim


def test_remote_gb10_launcher_preserves_muon_ns_override():
    script = Path("scripts/remote_train_gb10_nam56r_single.sh").read_text()

    assert "CPPMEGA_MUON_NUM_NS_STEPS:-3" in script
    assert '--muon-num-ns-steps "${CPPMEGA_MUON_NUM_NS_STEPS}"' in script
    assert "CPPMEGA_ATTN_BACKEND='${CPPMEGA_ATTN_BACKEND:-auto}'" in script
    assert '--attention-backend "${CPPMEGA_ATTN_BACKEND}"' in script
    assert "--attention-backend flash" not in script


def test_local_gb10_quarter_mxfp8_defaults_to_measured_te_tn_backend():
    """The local_gb10_quarter profile defaults to the fastest measured GB10 MXFP8 backend."""
    profile = get_run_profile("local_gb10_quarter")
    assert profile.precision.mxfp8_bwd_backend == "te_tn_adapter"
    assert profile.precision.mxfp8_flashinfer_runner == "mm_mxfp8"
    assert profile.precision.mxfp8_dense_saved_operands is True
    assert profile.precision.mxfp8_grouped_direct_backward is False


def test_mxfp8_transpose_emit_defaults_to_te_for_tn_adapter():
    """The default MXFP8 backward path uses TE transpose emission."""
    profile = get_run_profile("local_gb10_quarter")
    profile.precision.fp8_recipe = "mxfp8"
    env = profile_shell_assignments(profile)
    assert env["CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND"] == "te"
    assert env["CPPMEGA_TE_MXFP8_DENSE_SAVED_OPERANDS"] == "1"
    assert env["CPPMEGA_TE_MXFP8_GROUPED_DIRECT_BACKWARD"] == "0"
    assert env["CPPMEGA_TE_MXFP8_GROUPED_GEMM_READY_BACKWARD"] == "1"


def test_mxfp8_docs_pin_zero_copy_acceptance_counters():
    """The status docs should name the real MXFP8 counters, not vague wrappers."""
    architecture_doc = Path("docs/status/cppmega_architecture_status.md").read_text()
    token_flow_doc = Path("docs/status/cppmega_run_profiles_and_token_flow.md").read_text()
    docs = f"{architecture_doc}\n{token_flow_doc}"

    for needle in (
        "/home/dave/logs/gb10_mxfp8_zero_sidecars_20260428_171130.log",
        "/home/dave/logs/gb10_mxfp8_grouped_direct_smoke9_20260428_183814.log",
        "mxfp8_flashinfer_dgrad=204",
        "mxfp8_flashinfer_wgrad=204",
        "mxfp8_grouped_direct_dgrad=10",
        "mxfp8_grouped_direct_wgrad=10",
        "mxfp8_grouped_direct_miss_dgrad=0",
        "mxfp8_grouped_direct_miss_wgrad=0",
        "mxfp8_grouped_transpose_copy_fallback_dgrad=0",
        "mxfp8_grouped_transpose_copy_fallback_wgrad=0",
        "mxfp8_tn_adapter_saved_transpose_operand=408",
        "mxfp8_tn_adapter_copy_transpose=3084",
        "mxfp8_tn_adapter_missing_sidecar_copy=3084",
        "mxfp8_norm_quantize_sidecar_bridge=100",
        "_cppmega_mxfp8_colwise_as_rowwise_transpose",
        "--mxfp8-compact-columnwise-backward",
        "mxfp8_cutlass_native_dgrad>0",
        "mxfp8_cutlass_native_wgrad>0",
        "mxfp8_grouped_direct_dgrad>0",
        "mxfp8_grouped_direct_wgrad>0",
        "mxfp8_grouped_direct_miss_dgrad=0",
        "mxfp8_grouped_direct_miss_wgrad=0",
        "mxfp8_tn_adapter_te_emit=0",
        "mxfp8_tn_adapter_te_emit_deferred=0",
        "mxfp8_tn_adapter_saved_transpose_operand=0",
        "mxfp8_tn_adapter_copy_transpose=0",
        "mxfp8_tn_adapter_missing_sidecar_copy=0",
        "mxfp8_norm_quantize_sidecar_bridge=0",
        "mxfp8_tn_sidecar_attr_attached=0",
        "mxfp8_tn_sidecar_registry_peak=0",
        "mxfp8_tn_sidecar_registry_peak_bytes=0",
        "bf16_fallback_dgrad=0",
        "bf16_fallback_wgrad=0",
        "native_passthrough_dgrad=0",
        "native_passthrough_wgrad=0",
        "fallback_reasons={}",
    ):
        assert needle in docs


def test_tensorwise_fp8_keeps_hybrid_format():
    profile = get_run_profile("local_gb10_quarter")
    profile.precision.fp8_recipe = "tensorwise"
    env = profile_shell_assignments(profile)

    assert env["CPPMEGA_FP8_FORMAT"] == "hybrid"


def test_noconv_mamba_chunk_size_rejects_non_power_of_two():
    profile = get_run_profile("local_gb10_quarter")
    profile.runtime.noconv_mamba_chunk_size = 192

    with pytest.raises(ValueError, match="positive power of two"):
        profile_shell_assignments(profile)


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
