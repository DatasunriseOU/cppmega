"""Canonical cppmega shim: install Megatron/mamba3 compatibility patches.

This module is the single source of truth for the runtime monkey-patches that
cppmega training launches need to interoperate with upstream megatron-core +
mamba-ssm + transformer-engine at the version pins currently deployed on our
bench machines (torch 2.12 cu132, megatron-core 0.18rc0, TE 2.13, mamba_ssm
2.3.1).

Usage from a launch script::

    # Copy this file into the remote working directory and import it before
    # pretrain_mamba.py runs:
    cp scripts/cppmega_fp8_shim.py "${REMOTE_WORKDIR}/cppmega_fp8_shim.py"
    PYTHONPATH="${REMOTE_WORKDIR}:${PYTHONPATH}" python -c "import cppmega_fp8_shim"

Installed patches (each is best-effort and logs a clear message on failure):

  (1) `megatron.core.inference.contexts.static_context.deprecate_inference_params`
      compatibility alias — needed by `cppmega.megatron.mamba3_te_mixer` which
      imports this symbol from `static_context`, but in megatron-core 0.18rc0 the
      function lives in `megatron.core.utils`. Without this alias, any import of
      the mamba3_te_mixer module fails.

  (2) Optional MIMO config patch driven by environment variables
      `CPPMEGA_MAMBA3_MIMO=1` and `CPPMEGA_MAMBA_NUM_GROUPS`. Patches
      `TransformerConfig.__post_init__` to set
      `cppmega_mamba3_is_mimo=True`, `cppmega_mamba3_mimo_rank=4`, and
      `cppmega_mamba3_chunk_size=16` so the AuthorMamba3Mixer picks up the full
      7/7 MIMO feature configuration. Also optionally overrides
      `mamba_num_groups`.

  (3) `TransformerConfig.__getattr__` fallback that raises `AttributeError` on
      unknown `cppmega_mamba3_*` attribute names. This allows `getattr(config,
      "cppmega_mamba3_rope_fraction", 0.5)` to correctly fall back to the
      default 0.5 (returning None would trip downstream asserts like
      `rope_fraction in (0.5, 1.0)`).

  (4) `Float16Module.__init__` **one-shot** post-init patch that restores
      Mamba3 fp32 parameters (`dt_bias`, `D`, `B_bias`, `C_bias`, `mimo_x`,
      `mimo_z`, `mimo_o`) AFTER Megatron's blanket bf16 cast. The
      upstream `mamba_ssm.modules.mamba3.Mamba3` module initializes these
      parameters in fp32 because the TileLang `mamba_mimo_fwd_kernel` requires
      fp32 for them; Megatron's `Float16Module` wrapper walks all parameters
      and casts them to bf16 indiscriminately, breaking the kernel contract.

      This replaces an earlier PER-FORWARD pre-hook workaround that called
      `.data.float()` ~400 times per iteration (7 params * ~16 Mamba layers *
      fwd+bwd+optimizer accesses). Per nsys profile 2026-04-11 that was the
      #1 iter-time bottleneck (305 ms elementwise = 25.7% of the 1186 ms/iter
      baseline). The one-shot patch does the cast exactly once at model init
      and keeps the params fp32 permanently.

Upstream fix lives in `docs/upstream_bugs.md`. The per-forward hook pattern is
deprecated; do not re-introduce it.
"""
from __future__ import annotations

import os


# -----------------------------------------------------------------------------
# (0) GB10 TE MXFP8 recipe enablement
# -----------------------------------------------------------------------------
# TE 2.14's low-level MXFP8Quantizer + general_gemm works on GB10 / sm_121 in
# our probe, but the high-level fp8_autocast recipe guard still rejects all
# sm_12.x devices with "MXFP8 ... is not supported on 12.0+ architectures yet".
# Keep this as an explicit local escape hatch until upstream TE removes or
# narrows that guard.

if os.environ.get("CPPMEGA_ALLOW_TE_MXFP8_SM12", "0") == "1":
    try:
        import torch as _torch
        import transformer_engine.pytorch.quantization as _te_quantization

        _orig_check_mxfp8_support = _te_quantization.check_mxfp8_support

        def _cppmega_check_mxfp8_support():
            if _torch.cuda.is_available():
                major, minor = _torch.cuda.get_device_capability()
                if major == 12:
                    return True, ""
            return _orig_check_mxfp8_support()

        _te_quantization.check_mxfp8_support = _cppmega_check_mxfp8_support
        print("[cppmega_fp8_shim] TE MXFP8 sm_12.x recipe guard bypass enabled")
    except Exception as _exc:  # pragma: no cover
        import sys
        print(f"[cppmega_fp8_shim] TE MXFP8 sm_12.x bypass failed: {_exc}", file=sys.stderr)


# -----------------------------------------------------------------------------
# (0b) TE block-scaled backward override / GB10 MXFP8 TN adapter
# -----------------------------------------------------------------------------
# Upstream TransformerEngine main added `backward_override` to MXFP8/NVFP4
# recipes. TE 2.14 deployed on GB10 does not expose that knob, and its MXFP8
# Linear backward paths can fail in cuBLASLt on sm_12.x. Keep this opt-in and
# narrow.
#
# Preferred experimental path:
#   CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1
# retargets compact MXFP8 columnwise payloads as rowwise transposed operands
# and rewrites Linear backward GEMMs to the GB10-supported TN layout:
#   dgrad: general_gemm(weight.T_mxfp8, dy_mxfp8, layout="TN")
#   wgrad: general_gemm(x.T_mxfp8, dy.T_mxfp8, layout="TN")
#
# If the adapter cannot be used for a specific call (missing columnwise payload,
# swizzled scales, non-MXFP8 tensor, unsupported overlap), the BF16/dequantized
# path is used only when explicitly allowed. NVFP4 keeps using the older BF16
# override path when enabled.

_te_backward_override = os.environ.get("NVTE_BACKWARD_OVERRIDE", None)
_te_mxfp8_dgrad_bf16 = os.environ.get("CPPMEGA_TE_MXFP8_DGRAD_BF16", "0") == "1"
_te_mxfp8_wgrad_bf16 = os.environ.get("CPPMEGA_TE_MXFP8_WGRAD_BF16", "0") == "1"
_te_mxfp8_bwd_tn_adapter = os.environ.get("CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER", "0") == "1"
_te_mxfp8_bwd_allow_bf16_fallback = os.environ.get(
    "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK",
    "1"
    if (
        _te_mxfp8_dgrad_bf16
        or _te_mxfp8_wgrad_bf16
        or _te_backward_override in ("dequantized", "high_precision")
    )
    else "0",
) == "1"
_te_mxfp8_bwd_debug = os.environ.get("CPPMEGA_TE_MXFP8_BWD_DEBUG", "0") == "1"
_te_mxfp8_bf16_bridge_requested = (
    _te_mxfp8_dgrad_bf16
    or _te_mxfp8_wgrad_bf16
    or _te_backward_override in ("dequantized", "high_precision")
    or _te_mxfp8_bwd_allow_bf16_fallback
)
_te_mxfp8_bf16_bridge_ack_env = (
    "CPPMEGA_I_UNDERSTAND_MXFP8_BF16_BACKWARD_BRIDGE_IS_DEPRECATED_AND_SLOW"
)
if _te_mxfp8_bf16_bridge_requested:
    if os.environ.get(_te_mxfp8_bf16_bridge_ack_env, "0") != "1":
        raise RuntimeError(
            "Deprecated MXFP8 BF16 backward bridge requested. This path "
            "materializes backward operands in BF16, is slower, hides adapter "
            "coverage bugs, and must not be used for MXFP8 acceptance runs. "
            "Use CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1, "
            "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0, "
            "CPPMEGA_TE_MXFP8_DGRAD_BF16=0, "
            "CPPMEGA_TE_MXFP8_WGRAD_BF16=0, NVTE_BACKWARD_OVERRIDE=none. "
            f"If you are intentionally running the old bridge for archaeology, set "
            f"{_te_mxfp8_bf16_bridge_ack_env}=1."
        )
    import sys as _sys

    print(
        "[cppmega_fp8_shim] DEPRECATED: MXFP8 BF16 backward bridge enabled "
        "with explicit ACK. This path is obsolete; use "
        "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1 and "
        "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0 instead.",
        file=_sys.stderr,
    )
if (
    _te_mxfp8_bwd_tn_adapter
    or _te_mxfp8_dgrad_bf16
    or _te_mxfp8_wgrad_bf16
    or _te_backward_override in ("dequantized", "high_precision")
):
    try:
        import functools as _functools
        import atexit as _atexit
        import torch as _torch
        from transformer_engine.common.recipe import MXFP8BlockScaling as _TE_MXFP8Recipe
        from transformer_engine.pytorch.tensor import MXFP8Quantizer as _TE_MXFP8Quantizer
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor as _TE_MXFP8Tensor

        try:
            from transformer_engine.common.recipe import NVFP4BlockScaling as _TE_NVFP4Recipe
        except Exception:  # pragma: no cover
            _TE_NVFP4Recipe = None

        _cppmega_backward_override = _te_backward_override
        if _cppmega_backward_override is None and (
            _te_mxfp8_dgrad_bf16 or _te_mxfp8_wgrad_bf16
        ):
            _cppmega_backward_override = "dequantized"

        _cppmega_te_bwd_stats = {
            "mxfp8_tn_adapter_dgrad": 0,
            "mxfp8_tn_adapter_wgrad": 0,
            "bf16_fallback_dgrad": 0,
            "bf16_fallback_wgrad": 0,
            "native_passthrough_dgrad": 0,
            "native_passthrough_wgrad": 0,
            "fallback_reasons": {},
        }

        def _cppmega_te_bwd_stats_snapshot():
            _snapshot = dict(_cppmega_te_bwd_stats)
            _snapshot["fallback_reasons"] = dict(_cppmega_te_bwd_stats["fallback_reasons"])
            return _snapshot

        def _cppmega_record_bwd_stat(_key, _reason=None):
            _cppmega_te_bwd_stats[_key] += 1
            if _reason:
                _reasons = _cppmega_te_bwd_stats["fallback_reasons"]
                _reasons[_reason] = _reasons.get(_reason, 0) + 1

        def _cppmega_print_bwd_stats():
            _stats = _cppmega_te_bwd_stats_snapshot()
            print(f"[cppmega_fp8_shim] TE block-scaled backward stats: {_stats}")

        globals()["cppmega_te_mxfp8_bwd_stats"] = _cppmega_te_bwd_stats
        globals()["cppmega_te_mxfp8_bwd_stats_snapshot"] = _cppmega_te_bwd_stats_snapshot
        _atexit.register(_cppmega_print_bwd_stats)

        def _cppmega_mark_recipe(_recipe_cls):
            if _recipe_cls is None:
                return
            _orig_post_init = getattr(_recipe_cls, "__post_init__", None)

            def _post_init(self):
                if _orig_post_init is not None:
                    _orig_post_init(self)
                self.backward_override = _cppmega_backward_override
                # Match DelayedScaling's public tuple convention:
                # (fprop, dgrad, wgrad).  BF16 flags still describe the
                # high-precision fallback, not the TN adapter fast path.
                self.override_linear_precision = (
                    False,
                    _te_mxfp8_dgrad_bf16,
                    _te_mxfp8_wgrad_bf16,
                )
                self.cppmega_mxfp8_bwd_tn_adapter = _te_mxfp8_bwd_tn_adapter

            _recipe_cls.__post_init__ = _post_init
            _recipe_cls.backward_override = _cppmega_backward_override
            _recipe_cls.override_linear_precision = (
                False,
                _te_mxfp8_dgrad_bf16,
                _te_mxfp8_wgrad_bf16,
            )
            _recipe_cls.cppmega_mxfp8_bwd_tn_adapter = _te_mxfp8_bwd_tn_adapter

        _cppmega_mark_recipe(_TE_MXFP8Recipe)
        _cppmega_mark_recipe(_TE_NVFP4Recipe)

        def _cppmega_is_mxfp8_tensor(_x):
            return "MXFP8" in type(_x).__name__

        def _cppmega_is_block_scaled_tensor(_x):
            return _cppmega_is_mxfp8_tensor(_x) or "NVFP4" in type(_x).__name__

        def _cppmega_is_block_scaled_recipe(_recipe):
            return _recipe is not None and (
                "MXFP8" in type(_recipe).__name__ or "NVFP4" in type(_recipe).__name__
            )

        def _cppmega_is_mxfp8_recipe(_recipe):
            return _recipe is not None and "MXFP8" in type(_recipe).__name__

        def _cppmega_dequantize_if_needed(_x, _dtype):
            if hasattr(_x, "dequantize"):
                try:
                    return _x.dequantize(dtype=_dtype)
                except TypeError:
                    return _x.dequantize().to(_dtype)
            if isinstance(_x, _torch.Tensor) and _x.dtype != _dtype:
                return _x.to(_dtype)
            return _x

        def _cppmega_has_comm_overlap(_kwargs):
            return (
                _kwargs.get("ub", None) is not None
                or _kwargs.get("ub_type", None) is not None
                or _kwargs.get("extra_output", None) is not None
                or bool(_kwargs.get("bulk_overlap", False))
            )

        def _cppmega_mxfp8_colwise_as_rowwise_transpose(_x):
            if not _cppmega_is_mxfp8_tensor(_x):
                raise TypeError(f"expected MXFP8 tensor, got {type(_x).__name__}")
            if getattr(_x, "_with_gemm_swizzled_scales", False):
                raise ValueError("MXFP8 TN adapter requires compact, non-swizzled scales")
            _data = getattr(_x, "_columnwise_data", None)
            _scale = getattr(_x, "_columnwise_scale_inv", None)
            if _data is None or _scale is None:
                raise ValueError("MXFP8 TN adapter requires columnwise data and scales")
            if _data.dim() < 2:
                raise ValueError("MXFP8 TN adapter requires matrix-like columnwise data")
            if _scale.dim() != 2:
                raise ValueError("MXFP8 TN adapter requires 2D compact columnwise scales")
            _data_2d = _data.reshape(-1, _data.shape[-1])
            _rowwise_data = _data_2d.t().contiguous()
            _rowwise_scale = _scale.t().contiguous()
            _fp8_dtype = getattr(_x, "_fp8_dtype", None)
            if _fp8_dtype is None:
                raise ValueError("MXFP8 tensor is missing _fp8_dtype")
            _fake_dtype = getattr(_x, "_dtype", getattr(_x, "dtype", _torch.bfloat16))
            _quantizer = _TE_MXFP8Quantizer(_fp8_dtype, rowwise=True, columnwise=False)
            _quantizer.internal = True
            _quantizer.optimize_for_gemm = False
            return _TE_MXFP8Tensor(
                shape=_rowwise_data.shape,
                dtype=_fake_dtype,
                fp8_dtype=_fp8_dtype,
                rowwise_data=_rowwise_data,
                rowwise_scale_inv=_rowwise_scale,
                columnwise_data=None,
                columnwise_scale_inv=None,
                quantizer=_quantizer,
                requires_grad=False,
                with_gemm_swizzled_scales=False,
            )

        def _cppmega_try_mxfp8_tn_adapter(_orig_general_gemm, _args, _kwargs):
            _layout = _kwargs.get("layout")
            if not _te_mxfp8_bwd_tn_adapter:
                return False, "adapter_disabled"
            if not _kwargs.get("grad", False):
                return False, "not_backward_gemm"
            if _layout not in ("NN", "NT"):
                return False, f"unsupported_layout:{_layout}"
            if len(_args) < 2:
                return False, "missing_operands"
            if _cppmega_has_comm_overlap(_kwargs):
                return False, "comm_overlap_not_covered"

            _new_args = list(_args)
            _new_kwargs = dict(_kwargs)
            _new_kwargs["layout"] = "TN"
            # The GB10 probe validated the adapter with compact scales and
            # non-split accumulation. Native MXFP8 NN/NT fails regardless of
            # split-accumulator setting, so force the known-good TN mode.
            _new_kwargs["use_split_accumulator"] = False

            if _layout == "NN":
                if not (
                    _cppmega_is_mxfp8_tensor(_new_args[0])
                    and _cppmega_is_mxfp8_tensor(_new_args[1])
                ):
                    return False, "non_mxfp8_operands"
                _new_args[0] = _cppmega_mxfp8_colwise_as_rowwise_transpose(_new_args[0])
            else:
                if not (
                    _cppmega_is_mxfp8_tensor(_new_args[0])
                    and _cppmega_is_mxfp8_tensor(_new_args[1])
                ):
                    return False, "non_mxfp8_operands"
                _new_args[0] = _cppmega_mxfp8_colwise_as_rowwise_transpose(_new_args[0])
                _new_args[1] = _cppmega_mxfp8_colwise_as_rowwise_transpose(_new_args[1])

            return True, _orig_general_gemm(*_new_args, **_new_kwargs)

        def _cppmega_dequantized_backward_gemm(_orig_general_gemm, _args, _kwargs):
            _out_dtype = _kwargs.get("out_dtype", None)
            if _out_dtype is None:
                _out_dtype = getattr(_args[1], "dtype", _torch.bfloat16)
            _new_args = list(_args)
            _new_args[0] = _cppmega_dequantize_if_needed(_new_args[0], _out_dtype)
            _new_args[1] = _cppmega_dequantize_if_needed(_new_args[1], _out_dtype)
            _new_kwargs = dict(_kwargs)
            _new_kwargs["quantization_params"] = None
            _new_kwargs["use_split_accumulator"] = False
            return _orig_general_gemm(*_new_args, **_new_kwargs)

        def _cppmega_wrap_general_gemm(_module):
            _orig_general_gemm = getattr(_module, "general_gemm", None)
            if _orig_general_gemm is None or getattr(
                _orig_general_gemm, "_cppmega_blockscaled_bwd", False
            ):
                return False

            @_functools.wraps(_orig_general_gemm)
            def _general_gemm(*_args, **_kwargs):
                _layout = _kwargs.get("layout")
                _is_target_bwd = (
                    len(_args) >= 2
                    and _kwargs.get("grad", False)
                    and _layout in ("NN", "NT")
                    and (
                        _cppmega_is_block_scaled_tensor(_args[0])
                        or _cppmega_is_block_scaled_tensor(_args[1])
                    )
                )
                if not _is_target_bwd:
                    return _orig_general_gemm(*_args, **_kwargs)

                _op_kind = "dgrad" if _layout == "NN" else "wgrad"
                if _te_mxfp8_bwd_tn_adapter and (
                    _cppmega_is_mxfp8_tensor(_args[0])
                    or _cppmega_is_mxfp8_tensor(_args[1])
                ):
                    _fallback_reason = None
                    try:
                        _adapted_ok, _adapted_result = _cppmega_try_mxfp8_tn_adapter(
                            _orig_general_gemm, _args, _kwargs
                        )
                        if _adapted_ok:
                            _cppmega_record_bwd_stat(f"mxfp8_tn_adapter_{_op_kind}")
                            if _te_mxfp8_bwd_debug:
                                print(
                                    "[cppmega_fp8_shim] MXFP8 TN adapter "
                                    f"{_op_kind} layout={_layout}->TN"
                                )
                            return _adapted_result
                        _fallback_reason = str(_adapted_result)
                    except Exception as _adapter_exc:  # pragma: no cover
                        _fallback_reason = (
                            f"{type(_adapter_exc).__name__}: {_adapter_exc}"
                        )
                    if _te_mxfp8_bwd_allow_bf16_fallback:
                        _cppmega_record_bwd_stat(
                            f"bf16_fallback_{_op_kind}", _fallback_reason
                        )
                        print(
                            "[cppmega_fp8_shim] MXFP8 TN adapter BF16 fallback "
                            f"for {_op_kind}: {_fallback_reason}"
                        )
                        return _cppmega_dequantized_backward_gemm(
                            _orig_general_gemm, _args, _kwargs
                        )
                    _cppmega_record_bwd_stat(
                        f"native_passthrough_{_op_kind}", _fallback_reason
                    )
                    print(
                        "[cppmega_fp8_shim] MXFP8 TN adapter unavailable and "
                        "BF16 fallback disabled; using native TE GEMM for "
                        f"{_op_kind}: {_fallback_reason}"
                    )
                    return _orig_general_gemm(*_args, **_kwargs)

                if (
                    (_layout == "NN" and _te_mxfp8_dgrad_bf16)
                    or (_layout == "NT" and _te_mxfp8_wgrad_bf16)
                    or _te_backward_override in ("dequantized", "high_precision")
                ):
                    _cppmega_record_bwd_stat(
                        f"bf16_fallback_{_op_kind}", "legacy_bf16_override"
                    )
                    print(
                        "[cppmega_fp8_shim] TE block-scaled BF16 override "
                        f"for {_op_kind}: legacy_bf16_override"
                    )
                    return _cppmega_dequantized_backward_gemm(
                        _orig_general_gemm, _args, _kwargs
                    )

                return _orig_general_gemm(*_args, **_kwargs)

            _general_gemm._cppmega_blockscaled_bwd = True
            _module.general_gemm = _general_gemm
            return True

        from transformer_engine.pytorch.module import base as _te_module_base
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager as _TE_FP8State

        _orig_grad_output_preprocess = (
            _te_module_base.TransformerEngineBaseModule.grad_output_preprocess
        )
        if not getattr(_orig_grad_output_preprocess, "_cppmega_blockscaled_bwd", False):

            def _grad_output_preprocess(ctx, grad_output, row_parallel_mode, quantizer):
                _recipe = getattr(ctx, "fp8_recipe", None)
                if (
                    getattr(ctx, "fp8", False)
                    and _cppmega_is_mxfp8_recipe(_recipe)
                    and _te_mxfp8_bwd_tn_adapter
                ):
                    # Let TE quantize grad_output normally. The TN adapter
                    # needs both rowwise and compact columnwise MXFP8 payloads.
                    return _orig_grad_output_preprocess(
                        ctx, grad_output, row_parallel_mode, quantizer
                    )
                if getattr(ctx, "fp8", False) and _cppmega_is_block_scaled_recipe(_recipe):
                    grad_output = grad_output.reshape((-1, grad_output.shape[-1]))
                    grad_output = grad_output.contiguous()
                    if row_parallel_mode and getattr(ctx, "sequence_parallel", False):
                        if not getattr(ctx, "ub_overlap_ag", False):
                            grad_output, _ = _te_module_base.gather_along_first_dim(
                                grad_output, ctx.tp_group
                            )
                        else:
                            grad_output, _ = _te_module_base.fill_userbuffers_buffer_for_all_gather(
                                ctx.ub_obj_gradout,
                                grad_output,
                                None,
                                ctx.tp_group,
                            )
                    return grad_output, None
                return _orig_grad_output_preprocess(ctx, grad_output, row_parallel_mode, quantizer)

            _grad_output_preprocess._cppmega_blockscaled_bwd = True
            _te_module_base.TransformerEngineBaseModule.grad_output_preprocess = staticmethod(
                _grad_output_preprocess
            )

        def _cppmega_force_compact_if_needed(_quantizer, _recipe):
            if (
                _quantizer is not None
                and _te_mxfp8_bwd_tn_adapter
                and _cppmega_is_mxfp8_recipe(_recipe)
                and hasattr(_quantizer, "optimize_for_gemm")
            ):
                _quantizer.optimize_for_gemm = False

        def _cppmega_wrap_get_quantizers(_module_cls):
            _orig_get_quantizers = getattr(_module_cls, "_get_quantizers", None)
            if _orig_get_quantizers is None or getattr(
                _orig_get_quantizers, "_cppmega_backward_override", False
            ):
                return False

            @_functools.wraps(_orig_get_quantizers)
            def _get_quantizers(self, fp8_output, fp8_grad, is_grad_enabled):
                _quantizers = _orig_get_quantizers(self, fp8_output, fp8_grad, is_grad_enabled)
                try:
                    _recipe = _TE_FP8State.get_fp8_recipe()
                    if _cppmega_is_block_scaled_recipe(_recipe):
                        for _q in (_quantizers[0], _quantizers[5]):
                            if _q is not None and hasattr(_q, "optimize_for_gemm"):
                                _q.optimize_for_gemm = False
                except Exception:
                    pass
                return _quantizers

            _get_quantizers._cppmega_backward_override = True
            _module_cls._get_quantizers = _get_quantizers
            return True

        def _cppmega_wrap_get_weight_quantizers(_module_cls):
            _orig_get_weight_quantizers = getattr(_module_cls, "_get_weight_quantizers", None)
            if _orig_get_weight_quantizers is None or getattr(
                _orig_get_weight_quantizers, "_cppmega_mxfp8_compact", False
            ):
                return False

            @_functools.wraps(_orig_get_weight_quantizers)
            def _get_weight_quantizers(self):
                _quantizers = _orig_get_weight_quantizers(self)
                try:
                    _recipe = _TE_FP8State.get_fp8_recipe()
                    for _q in _quantizers:
                        _cppmega_force_compact_if_needed(_q, _recipe)
                except Exception:
                    pass
                return _quantizers

            _get_weight_quantizers._cppmega_mxfp8_compact = True
            _module_cls._get_weight_quantizers = _get_weight_quantizers
            return True

        _patched_gemm_modules = []
        _patched_quantizer_modules = []
        _patched_weight_quantizer_modules = []
        for _module_name in (
            "transformer_engine.pytorch.module.linear",
            "transformer_engine.pytorch.module.layernorm_linear",
            "transformer_engine.pytorch.module.grouped_linear",
            "transformer_engine.pytorch.ops.basic.basic_linear",
        ):
            try:
                _mod = __import__(_module_name, fromlist=["general_gemm"])
                if _cppmega_wrap_general_gemm(_mod):
                    _patched_gemm_modules.append(_module_name.rsplit(".", 1)[-1])
                for _class_name in ("Linear", "LayerNormLinear"):
                    _class = getattr(_mod, _class_name, None)
                    if _class is not None and _cppmega_wrap_get_quantizers(_class):
                        _patched_quantizer_modules.append(_class_name)
                    if _class is not None and _cppmega_wrap_get_weight_quantizers(_class):
                        _patched_weight_quantizer_modules.append(_class_name)
            except Exception:
                pass

        print(
            "[cppmega_fp8_shim] TE block-scaled backward override installed "
            f"(backward_override={_cppmega_backward_override}, "
            f"override_linear_precision={(False, _te_mxfp8_dgrad_bf16, _te_mxfp8_wgrad_bf16)}, "
            f"mxfp8_bwd_tn_adapter={_te_mxfp8_bwd_tn_adapter}, "
            f"mxfp8_bwd_allow_bf16_fallback={_te_mxfp8_bwd_allow_bf16_fallback}, "
            f"gemm_modules={_patched_gemm_modules}, "
            f"quantizer_modules={_patched_quantizer_modules}, "
            f"weight_quantizer_modules={_patched_weight_quantizer_modules})"
        )
    except Exception as _exc:  # pragma: no cover
        import sys
        print(f"[cppmega_fp8_shim] TE block-scaled backward override failed: {_exc}", file=sys.stderr)


# -----------------------------------------------------------------------------
# (1) deprecate_inference_params compatibility shim
# -----------------------------------------------------------------------------
try:
    from megatron.core.inference.contexts import static_context as _sc
    if not hasattr(_sc, "deprecate_inference_params"):
        try:
            from megatron.core.utils import deprecate_inference_params as _dip
        except ImportError:
            def _dip(inference_context, inference_params):
                if inference_context is None and inference_params is not None:
                    return inference_params
                return inference_context
        _sc.deprecate_inference_params = _dip
except Exception as _exc:  # pragma: no cover
    import sys
    print(f"[cppmega_fp8_shim] static_context alias skipped: {_exc}", file=sys.stderr)


# -----------------------------------------------------------------------------
# (2) Optional MIMO config patch (env-driven)
# -----------------------------------------------------------------------------
_mimo_on = os.environ.get("CPPMEGA_MAMBA3_MIMO", "0") == "1"
_num_groups_override = os.environ.get("CPPMEGA_MAMBA_NUM_GROUPS", "")
if _mimo_on or _num_groups_override:
    try:
        from megatron.core.transformer.transformer_config import TransformerConfig
        _orig_post_init = TransformerConfig.__post_init__

        def _cppmega_mimo_post_init(self):
            if _num_groups_override:
                try:
                    _ng = int(_num_groups_override)
                    object.__setattr__(self, "mamba_num_groups", _ng)
                    print(f"[cppmega_fp8_shim] mamba_num_groups override -> {_ng}")
                except Exception as _e:
                    import sys
                    print(f"[cppmega_fp8_shim] mamba_num_groups override failed: {_e}", file=sys.stderr)
            _orig_post_init(self)
            if _mimo_on:
                if not getattr(self, "cppmega_mamba3_is_mimo", False):
                    object.__setattr__(self, "cppmega_mamba3_is_mimo", True)
                if not getattr(self, "cppmega_mamba3_mimo_rank", None):
                    object.__setattr__(self, "cppmega_mamba3_mimo_rank", 4)
                if not getattr(self, "cppmega_mamba3_chunk_size", None):
                    object.__setattr__(self, "cppmega_mamba3_chunk_size", 16)

        TransformerConfig.__post_init__ = _cppmega_mimo_post_init
        if _mimo_on:
            print("[cppmega_fp8_shim] MIMO patch installed (rank=4, chunk=16)")
        if _num_groups_override:
            print(f"[cppmega_fp8_shim] mamba_num_groups override arm set -> {_num_groups_override}")
    except Exception as _exc:  # pragma: no cover
        import sys
        print(f"[cppmega_fp8_shim] MIMO patch failed: {_exc}", file=sys.stderr)


# -----------------------------------------------------------------------------
# (3) TransformerConfig dynamic attribute fallback
# -----------------------------------------------------------------------------
try:
    from megatron.core.transformer.transformer_config import TransformerConfig
    _TC_BASE_GETATTR = getattr(TransformerConfig, "__getattr__", None)

    def _cppmega_getattr(self, name):
        if name.startswith("cppmega_mamba3_"):
            raise AttributeError(name)
        if _TC_BASE_GETATTR is not None:
            return _TC_BASE_GETATTR(self, name)
        raise AttributeError(name)

    if not hasattr(TransformerConfig, "_cppmega_mamba3_attr_patched"):
        TransformerConfig.__getattr__ = _cppmega_getattr
        TransformerConfig._cppmega_mamba3_attr_patched = True
except Exception:
    pass


# -----------------------------------------------------------------------------
# (4) Float16Module one-shot Mamba3 fp32 param patch
# -----------------------------------------------------------------------------
# Replaces the deprecated per-forward `register_forward_pre_hook` workaround.
# Per nsys profile 2026-04-11 the per-forward hook was responsible for 305 ms/iter
# (25.7% of the 1186 ms MIMO 7/7 baseline iter) via ~400 D2D copies per iter.
# The one-shot approach restores Mamba3 fp32 params exactly once after
# Megatron's bf16 cast at model-init time and keeps them fp32 permanently.

_MIMO_FP32_PARAMS = (
    "dt_bias",
    "D",
    "B_bias",
    "C_bias",
    "mimo_x",
    "mimo_z",
    "mimo_o",
)

try:
    import torch as _torch
    from megatron.core.transformer.module import Float16Module

    if not getattr(Float16Module, "_cppmega_mamba3_fp32_patched", False):
        _orig_f16_init = Float16Module.__init__

        def _cppmega_f16_init(self, config, module, *args, **kwargs):
            _orig_f16_init(self, config, module, *args, **kwargs)
            # After Megatron's blanket bf16 cast, walk the wrapped module and
            # restore fp32 dtype on Mamba3 bias/D/dt parameters that the
            # TileLang `mamba_mimo_fwd_kernel` contract requires in fp32.
            patched = 0
            for submod in self.module.modules():
                if type(submod).__name__ == "Mamba3":
                    for _name in _MIMO_FP32_PARAMS:
                        _p = getattr(submod, _name, None)
                        if _p is not None and _p.dtype != _torch.float32:
                            _p.data = _p.data.to(_torch.float32)
                            patched += 1
            if patched:
                print(f"[cppmega_fp8_shim] Float16Module one-shot Mamba3 fp32 patch: restored {patched} params")

        Float16Module.__init__ = _cppmega_f16_init
        Float16Module._cppmega_mamba3_fp32_patched = True
        print("[cppmega_fp8_shim] Float16Module one-shot Mamba3 fp32 patch installed")
except Exception as _exc:  # pragma: no cover
    import sys
    print(f"[cppmega_fp8_shim] Float16Module patch failed: {_exc}", file=sys.stderr)


# -----------------------------------------------------------------------------
# (5) CUDA graph compatibility: bypass _broadcast_cu_seqlens at TP=1
# -----------------------------------------------------------------------------
# When CUDA graphs capture forward_step, get_batch is called inside the
# captured region.  _broadcast_cu_seqlens does torch.tensor(n, dtype=int64,
# device=cuda) which creates a tensor from a Python int — requires pinned
# source memory under graph capture, which isn't set up.  At TP=1 the
# broadcast is a no-op anyway (only needed for TP>1 data sync), so we
# bypass it entirely.  This unblocks --cuda-graph-impl local on MIMO 7/7.

try:
    from megatron.training import utils as _mt_utils

    if hasattr(_mt_utils, "_broadcast_cu_seqlens"):
        _orig_bcs = _mt_utils._broadcast_cu_seqlens

        def _cppmega_broadcast_cu_seqlens(cu_seqlens, *args, **kwargs):
            # At TP=1 the broadcast is a no-op — skip the torch.tensor()
            # call that breaks CUDA graph capture.
            try:
                from megatron.core import parallel_state as _ps
                if _ps.get_tensor_model_parallel_world_size() == 1:
                    return cu_seqlens
            except Exception:
                pass
            return _orig_bcs(cu_seqlens, *args, **kwargs)

        _mt_utils._broadcast_cu_seqlens = _cppmega_broadcast_cu_seqlens
        print("[cppmega_fp8_shim] _broadcast_cu_seqlens TP=1 bypass installed (CUDA graph compat)")
except Exception as _exc:  # pragma: no cover
    import sys
    print(f"[cppmega_fp8_shim] _broadcast_cu_seqlens patch failed: {_exc}", file=sys.stderr)


# -----------------------------------------------------------------------------
# (6) MTP fused linear cross-entropy route
# -----------------------------------------------------------------------------
# Default to Megatron's LinearCrossEntropyModule route.  On GB10 this is routed
# to Apple CCE by cppmega.megatron.apply_linear_ce_patch; on supported Hopper /
# Blackwell stacks it can use the native Megatron backend.  The old Liger/off
# routes are fail-closed and require explicit archaeology ACK env vars.

try:
    from cppmega.megatron.deprecated_paths import require_deprecated_ack

    _mtp_ce_kernel = os.environ.get("CPPMEGA_MTP_CE_KERNEL", "native").lower()
    if _mtp_ce_kernel in ("native", "linear", "output_layer", "cce", "auto"):
        os.environ.setdefault("CPPMEGA_MTP_NATIVE_HOPPER_CE", "1")
        from cppmega.megatron.mtp_native_hopper_ce import patch_mtp_native_hopper_ce
        patch_mtp_native_hopper_ce()
    elif _mtp_ce_kernel == "liger":
        require_deprecated_ack(
            feature="CPPMEGA_MTP_CE_KERNEL=liger",
            ack_env="CPPMEGA_I_UNDERSTAND_MTP_LIGER_CE_IS_DEPRECATED",
            replacement="CPPMEGA_MTP_CE_KERNEL=native",
            reason="MTP now routes through Megatron LinearCE/native or CCE.",
        )
        from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
        patch_mtp_loss_with_liger()
    elif _mtp_ce_kernel in ("0", "none", "off", "false"):
        require_deprecated_ack(
            feature=f"CPPMEGA_MTP_CE_KERNEL={_mtp_ce_kernel}",
            ack_env="CPPMEGA_I_UNDERSTAND_MTP_CE_DISABLED_IS_DEPRECATED",
            replacement="CPPMEGA_MTP_CE_KERNEL=native",
            reason="Disabling the patch can fall back to the old materializing path.",
        )
        print("[cppmega_fp8_shim] MTP CE patch disabled by CPPMEGA_MTP_CE_KERNEL")
    else:
        raise ValueError(f"unsupported CPPMEGA_MTP_CE_KERNEL={_mtp_ce_kernel!r}")
except Exception as _exc:  # pragma: no cover
    import sys
    print(f"[cppmega_fp8_shim] MTP CE patch failed: {_exc}", file=sys.stderr)
    raise


# -----------------------------------------------------------------------------
# (7) TileLang SparseMLA monkey-patch for DSA sparse attention (env-driven)
# -----------------------------------------------------------------------------
# Replaces Megatron's ``unfused_dsa_fn`` (which materializes the FULL
# [b*np, sq, sk] FP32 attention scores = 7 GiB at production shape) with
# TileLang's fused online-softmax sparse MLA kernel from Megatron-LM PR #3674.
#
# Controlled by CPPMEGA_DSA_SPARSE_MODE env var:
#   "tilelang" (default) — fused TileLang kernel, ~40% throughput improvement
#   "gather_scatter"     — PyTorch gather-scatter fallback (no TileLang dep)
#
# The TileLang kernel is parameterized for arbitrary d_v (works with d_v=512,
# d_v=96, etc.) despite earlier comments claiming d_v=96 only.

from cppmega.megatron.deprecated_paths import require_deprecated_ack as _cppmega_require_deprecated_ack


def _cppmega_require_gather_scatter_ack(feature, reason):
    _cppmega_require_deprecated_ack(
        feature=feature,
        ack_env="CPPMEGA_I_UNDERSTAND_DSA_GATHER_SCATTER_IS_DEPRECATED_AND_SLOW",
        replacement="CPPMEGA_DSA_SPARSE_MODE=tilelang",
        reason=reason,
    )


_sparse_mode = os.environ.get("CPPMEGA_DSA_SPARSE_MODE", "tilelang").strip().lower()
if _sparse_mode not in ("gather_scatter", "gather-scatter", "pytorch"):
    try:
        from megatron.core.transformer.experimental_attention_variant import dsa as _dsa_mod

        _existing_unfused = getattr(_dsa_mod, "unfused_dsa_fn", None)
        if _existing_unfused is not None and not getattr(
            _existing_unfused, "__cppmega_sparse_dsa_patched__", False
        ):
            from cppmega.megatron.sparse_mla_ops.sparse_mla import (
                sparse_mla_as_unfused_dsa as _sparse_mla_fn,
            )

            setattr(_sparse_mla_fn, "__cppmega_sparse_dsa_patched__", True)
            _dsa_mod.unfused_dsa_fn = _sparse_mla_fn
            print(
                "[cppmega_fp8_shim] TileLang SparseMLA applied "
                "(replaces unfused_dsa_fn: fused online-softmax sparse attention)"
            )
        else:
            print("[cppmega_fp8_shim] TileLang SparseMLA: already patched or unfused_dsa_fn not found")
    except Exception as _exc:
        import sys
        print(f"[cppmega_fp8_shim] TileLang SparseMLA patch failed: {_exc}", file=sys.stderr)
        _cppmega_require_gather_scatter_ack(
            "TileLang SparseMLA patch failure fallback to gather_scatter",
            "Silent fallback hides kernel/import failures and can materialize more memory.",
        )
        from megatron.core.transformer.experimental_attention_variant import dsa as _dsa_mod
        from cppmega.megatron.dsa_sparse_attention import sparse_dsa_fn as _sparse_dsa_fn

        setattr(_sparse_dsa_fn, "__cppmega_sparse_dsa_patched__", True)
        _dsa_mod.unfused_dsa_fn = _sparse_dsa_fn
        print("[cppmega_fp8_shim] DEPRECATED fallback: gather_scatter sparse_dsa_fn applied")
else:
    # Explicit gather_scatter mode
    _cppmega_require_gather_scatter_ack(
        f"CPPMEGA_DSA_SPARSE_MODE={_sparse_mode}",
        "The fused TileLang SparseMLA path is the current default.",
    )
    try:
        from megatron.core.transformer.experimental_attention_variant import dsa as _dsa_mod
        from cppmega.megatron.dsa_sparse_attention import sparse_dsa_fn as _sparse_dsa_fn

        _existing_unfused = getattr(_dsa_mod, "unfused_dsa_fn", None)
        if _existing_unfused is not None and not getattr(
            _existing_unfused, "__cppmega_sparse_dsa_patched__", False
        ):
            setattr(_sparse_dsa_fn, "__cppmega_sparse_dsa_patched__", True)
            _dsa_mod.unfused_dsa_fn = _sparse_dsa_fn
            print(
                "[cppmega_fp8_shim] gather_scatter sparse_dsa_fn applied "
                "(CPPMEGA_DSA_SPARSE_MODE=gather_scatter)"
            )
    except Exception as _exc:
        import sys
        print(f"[cppmega_fp8_shim] gather_scatter patch failed: {_exc}", file=sys.stderr)
        raise
