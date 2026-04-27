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

_raw_nvte_backward_override = os.environ.get("NVTE_BACKWARD_OVERRIDE", None)
if _raw_nvte_backward_override in ("", "none", "None", "NONE"):
    # Older cppmega launchers used "none" as a sentinel to override scripts
    # that defaulted this variable to "dequantized". Fresh TE reads this env
    # while defining pydantic dataclasses, so normalize before any TE import.
    os.environ.pop("NVTE_BACKWARD_OVERRIDE", None)


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
# Preferred experimental path, configured by cppmega.recipes.run_profiles and
# rendered as CPPMEGA_* only because this shim is imported before Megatron args
# are available:
#   CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1
# retargets compact MXFP8 columnwise payloads as rowwise transposed operands
# and rewrites Linear backward GEMMs to the GB10-supported TN layout:
#   dgrad: general_gemm(weight.T_mxfp8, dy_mxfp8, layout="TN")
#   wgrad: general_gemm(x.T_mxfp8, dy.T_mxfp8, layout="TN")
#
# Backend selection:
#   CPPMEGA_TE_MXFP8_BWD_BACKEND=te_tn_adapter      # default, TE TN GEMM
#   CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native     # GB10 CUTLASS SM120/121
#   CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND=compact     # direct compact-scale mainloop
#   CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND=prepack     # old native-scale prepack A/B
#
# The CUTLASS backend uses a cppmega SM120 mainloop fork by default.  Regular
# TN keeps stock A/B TMA with compact scale copies in the mainloop; backward
# NN/NT uses a manual payload+scale loader so original compact TE columnwise
# operands do not need rowwise-transpose sidecars.
#
# If the adapter cannot be used for a specific call (missing columnwise payload,
# swizzled scales, non-MXFP8 tensor, unsupported overlap), the BF16/dequantized
# path is used only when explicitly allowed. NVFP4 keeps using the older BF16
# override path when enabled.

_te_backward_override = os.environ.get("NVTE_BACKWARD_OVERRIDE", None)
_te_mxfp8_dgrad_bf16 = os.environ.get("CPPMEGA_TE_MXFP8_DGRAD_BF16", "0") == "1"
_te_mxfp8_wgrad_bf16 = os.environ.get("CPPMEGA_TE_MXFP8_WGRAD_BF16", "0") == "1"
_te_mxfp8_bwd_tn_adapter = os.environ.get("CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER", "0") == "1"
_te_mxfp8_bwd_backend = os.environ.get(
    "CPPMEGA_TE_MXFP8_BWD_BACKEND", "te_tn_adapter"
).lower()
if _te_mxfp8_bwd_backend not in ("te_tn_adapter", "cutlass_native"):
    raise RuntimeError(
        "Unsupported CPPMEGA_TE_MXFP8_BWD_BACKEND="
        f"{_te_mxfp8_bwd_backend!r}; expected te_tn_adapter or cutlass_native"
    )
_te_mxfp8_transpose_emit_default = (
    "off" if _te_mxfp8_bwd_backend == "cutlass_native" else "auto"
)
_te_mxfp8_transpose_emit_backend = os.environ.get(
    "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND",
    _te_mxfp8_transpose_emit_default,
).lower()
if _te_mxfp8_transpose_emit_backend not in ("auto", "te", "off"):
    raise RuntimeError(
        "Unsupported CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND="
        f"{_te_mxfp8_transpose_emit_backend!r}; expected auto, te, or off"
    )
_te_mxfp8_transpose_emit_strict = os.environ.get(
    "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_STRICT", "0"
) == "1"
_te_mxfp8_transpose_emit_swizzled_default = (
    "0" if _te_mxfp8_bwd_backend == "cutlass_native" else "1"
)
_te_mxfp8_transpose_emit_swizzled = os.environ.get(
    "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED",
    _te_mxfp8_transpose_emit_swizzled_default,
) == "1"
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
_te_blockscaled_dequantized_backward = (
    _te_mxfp8_dgrad_bf16
    or _te_mxfp8_wgrad_bf16
    or _te_backward_override in ("dequantized", "high_precision")
)
_te_mxfp8_bf16_bridge_requested = (
    _te_blockscaled_dequantized_backward
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
        import transformer_engine_torch as _tex
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
            "mxfp8_tn_adapter_te_emit": 0,
            "mxfp8_tn_adapter_saved_transpose_operand": 0,
            "mxfp8_tn_adapter_te_emit_swizzled": 0,
            "mxfp8_tn_adapter_te_emit_swizzled_unavailable": 0,
            "mxfp8_tn_adapter_copy_transpose": 0,
            "mxfp8_tn_adapter_missing_sidecar_copy": 0,
            "mxfp8_tn_adapter_te_emit_failed": 0,
            "mxfp8_norm_quantize_sidecar_bridge": 0,
            "mxfp8_cutlass_native_dgrad": 0,
            "mxfp8_cutlass_native_wgrad": 0,
            "bf16_fallback_dgrad": 0,
            "bf16_fallback_wgrad": 0,
            "native_passthrough_dgrad": 0,
            "native_passthrough_wgrad": 0,
            "fallback_reasons": {},
        }

        def _cppmega_te_bwd_stats_snapshot():
            _snapshot = dict(_cppmega_te_bwd_stats)
            _snapshot["fallback_reasons"] = dict(_cppmega_te_bwd_stats["fallback_reasons"])
            try:
                _registry = _cppmega_mxfp8_tn_sidecar_registry
                _snapshot["mxfp8_tn_sidecar_registry_size"] = len(_registry)
                _snapshot["mxfp8_tn_sidecar_registry_persistent"] = sum(
                    1 for _entry in _registry.values() if _entry[1]
                )
                _snapshot["mxfp8_tn_sidecar_registry_peak"] = (
                    _cppmega_mxfp8_tn_sidecar_registry_peak[0]
                )
            except NameError:
                pass
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

        _cppmega_mxfp8_tn_sidecar_attr = "_cppmega_mxfp8_rowwise_transpose"
        _cppmega_mxfp8_tn_sidecar_persistent_attr = (
            "_cppmega_mxfp8_rowwise_transpose_persistent"
        )

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
        _cppmega_quantize_weight_debug_count = [0]
        _cppmega_attach_debug_count = [0]
        _cppmega_mxfp8_tn_sidecar_registry = {}
        # Forward can create hundreds of MXFP8 transpose sidecars before the
        # matching backward GEMM consumes the earliest ones. Keep enough entries
        # for the full local NAM56R-quarter graph; entries are still one-shot
        # and are removed on lookup in backward.
        _cppmega_mxfp8_tn_sidecar_registry_limit = 8192
        _cppmega_mxfp8_tn_sidecar_registry_peak = [0]

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

        def _cppmega_mxfp8_debug_desc(_x):
            def _shape(_maybe_tensor):
                return (
                    tuple(_maybe_tensor.shape)
                    if isinstance(_maybe_tensor, _torch.Tensor)
                    else None
                )

            _attrs = []
            if hasattr(_x, "__dict__"):
                _attrs = sorted(str(_k) for _k in _x.__dict__.keys())
            return (
                f"type={type(_x).__name__}, shape={getattr(_x, 'shape', None)}, "
                f"dtype={getattr(_x, 'dtype', None)}, "
                f"fp8_dtype={getattr(_x, '_fp8_dtype', None)}, "
                f"rowwise_data={_shape(getattr(_x, '_rowwise_data', None))}, "
                f"columnwise_data={_shape(getattr(_x, '_columnwise_data', None))}, "
                f"rowwise_scale={_shape(getattr(_x, '_rowwise_scale_inv', None))}, "
                f"columnwise_scale={_shape(getattr(_x, '_columnwise_scale_inv', None))}, "
                f"swizzled={getattr(_x, '_with_gemm_swizzled_scales', None)}, "
                f"quantizer={type(getattr(_x, '_quantizer', None)).__name__}, "
                f"has_sidecar={getattr(_x, _cppmega_mxfp8_tn_sidecar_attr, None) is not None}, "
                f"attrs={_attrs}"
            )

        def _cppmega_mxfp8_sidecar_key(_x):
            _data = getattr(_x, "_columnwise_data", None)
            _scale = getattr(_x, "_columnwise_scale_inv", None)
            if not isinstance(_data, _torch.Tensor) or not isinstance(_scale, _torch.Tensor):
                return None
            return (
                int(_data.data_ptr()),
                int(_scale.data_ptr()),
                tuple(_data.shape),
                tuple(_scale.shape),
                str(getattr(_x, "_fp8_dtype", None)),
            )

        def _cppmega_register_mxfp8_sidecar(_x, _sidecar, *, persistent=False):
            _key = _cppmega_mxfp8_sidecar_key(_x)
            if _key is None:
                return
            _cppmega_mxfp8_tn_sidecar_registry[_key] = (_sidecar, bool(persistent))
            _cppmega_mxfp8_tn_sidecar_registry_peak[0] = max(
                _cppmega_mxfp8_tn_sidecar_registry_peak[0],
                len(_cppmega_mxfp8_tn_sidecar_registry),
            )
            while len(_cppmega_mxfp8_tn_sidecar_registry) > (
                _cppmega_mxfp8_tn_sidecar_registry_limit
            ):
                _cppmega_mxfp8_tn_sidecar_registry.pop(
                    next(iter(_cppmega_mxfp8_tn_sidecar_registry))
                )

        def _cppmega_get_mxfp8_sidecar_entry(_x):
            _sidecar = getattr(_x, _cppmega_mxfp8_tn_sidecar_attr, None)
            _key = _cppmega_mxfp8_sidecar_key(_x)
            if _sidecar is not None:
                _persistent = bool(
                    getattr(_x, _cppmega_mxfp8_tn_sidecar_persistent_attr, False)
                )
                if _key is not None and not _persistent:
                    _cppmega_mxfp8_tn_sidecar_registry.pop(_key, None)
                return _sidecar, _persistent
            _entry = _cppmega_mxfp8_tn_sidecar_registry.get(_key)
            if _entry is not None and not _entry[1]:
                _cppmega_mxfp8_tn_sidecar_registry.pop(_key, None)
            return _entry

        def _cppmega_get_mxfp8_sidecar(_x):
            _entry = _cppmega_get_mxfp8_sidecar_entry(_x)
            return None if _entry is None else _entry[0]

        def _cppmega_unregister_mxfp8_sidecar(_x):
            _key = _cppmega_mxfp8_sidecar_key(_x)
            if _key is not None:
                _cppmega_mxfp8_tn_sidecar_registry.pop(_key, None)

        def _cppmega_mark_rowwise_transpose_operand(_x):
            setattr(_x, "_te_rowwise_transpose_for_backward_operand", True)
            setattr(_x, "_cppmega_mxfp8_rowwise_transpose_operand", True)
            return _x

        def _cppmega_propagate_mxfp8_sidecar(_src, _dst):
            if not (_cppmega_is_mxfp8_tensor(_src) and _cppmega_is_mxfp8_tensor(_dst)):
                return _dst
            if getattr(_src, "_fp8_dtype", None) != getattr(_dst, "_fp8_dtype", None):
                return _dst
            _src_data = getattr(_src, "_columnwise_data", None)
            _dst_data = getattr(_dst, "_columnwise_data", None)
            if not isinstance(_src_data, _torch.Tensor) or not isinstance(
                _dst_data, _torch.Tensor
            ):
                return _dst
            if tuple(_src_data.shape) != tuple(_dst_data.shape):
                return _dst
            _entry = _cppmega_get_mxfp8_sidecar_entry(_src)
            if _entry is None:
                return _dst
            _sidecar, _persistent = _entry
            setattr(_dst, _cppmega_mxfp8_tn_sidecar_attr, _sidecar)
            setattr(_dst, _cppmega_mxfp8_tn_sidecar_persistent_attr, _persistent)
            _cppmega_register_mxfp8_sidecar(_dst, _sidecar, persistent=_persistent)
            return _dst

        def _cppmega_attach_mxfp8_rowwise_transpose(_out, _quantizer, _source):
            if not (
                _te_mxfp8_bwd_tn_adapter
                and _te_mxfp8_transpose_emit_backend in ("auto", "te")
                and _cppmega_is_mxfp8_tensor(_out)
                and isinstance(_source, _torch.Tensor)
                and not _cppmega_is_block_scaled_tensor(_source)
            ):
                return _out
            if getattr(_out, "_with_gemm_swizzled_scales", False):
                return _out
            _columnwise_scale = getattr(_out, "_columnwise_scale_inv", None)
            if _columnwise_scale is None:
                return _out
            if getattr(_out, _cppmega_mxfp8_tn_sidecar_attr, None) is not None:
                return _out
            if not hasattr(_quantizer, "quantize_rowwise_transpose"):
                _cppmega_record_bwd_stat(
                    "mxfp8_tn_adapter_te_emit_failed", "missing_quantize_rowwise_transpose"
                )
                return _out
            try:
                with _torch.no_grad():
                    _fake_dtype = getattr(
                        _out,
                        "_dtype",
                        getattr(_out, "dtype", _source.dtype),
                    )
                    _transpose_kwargs = {"fake_dtype": _fake_dtype}
                    if _te_mxfp8_transpose_emit_swizzled:
                        _transpose_kwargs["with_gemm_swizzled_scales"] = True
                    try:
                        _sidecar = _quantizer.quantize_rowwise_transpose(
                            _source,
                            _columnwise_scale,
                            **_transpose_kwargs,
                        )
                    except TypeError as _type_exc:
                        if (
                            not _te_mxfp8_transpose_emit_swizzled
                            or "with_gemm_swizzled_scales" not in str(_type_exc)
                        ):
                            raise
                        if (
                            _te_mxfp8_transpose_emit_backend == "te"
                            and _te_mxfp8_transpose_emit_strict
                        ):
                            raise
                        _cppmega_record_bwd_stat(
                            "mxfp8_tn_adapter_te_emit_swizzled_unavailable",
                            "missing_quantize_rowwise_transpose_swizzle_arg",
                        )
                        _transpose_kwargs.pop("with_gemm_swizzled_scales", None)
                        _sidecar = _quantizer.quantize_rowwise_transpose(
                            _source,
                            _columnwise_scale,
                            **_transpose_kwargs,
                        )
                    if _te_mxfp8_transpose_emit_swizzled:
                        if getattr(_sidecar, "_with_gemm_swizzled_scales", False):
                            _cppmega_record_bwd_stat(
                                "mxfp8_tn_adapter_te_emit_swizzled"
                            )
                        else:
                            _cppmega_record_bwd_stat(
                                "mxfp8_tn_adapter_te_emit_swizzled_unavailable",
                                "compact_transpose_emit_returned",
                            )
                            if (
                                _te_mxfp8_transpose_emit_backend == "te"
                                and _te_mxfp8_transpose_emit_strict
                            ):
                                raise RuntimeError(
                                    "TE MXFP8 transpose emit returned compact scales "
                                    "while swizzled scales were required"
                                )
                _cppmega_mark_rowwise_transpose_operand(_sidecar)
                setattr(_out, "_te_rowwise_transpose_for_backward", _sidecar)
                setattr(
                    _out,
                    "_te_rowwise_transpose_for_backward_unregister",
                    _cppmega_unregister_mxfp8_sidecar,
                )
                setattr(_out, _cppmega_mxfp8_tn_sidecar_attr, _sidecar)
                setattr(
                    _out,
                    "_cppmega_mxfp8_rowwise_transpose_unregister",
                    _cppmega_unregister_mxfp8_sidecar,
                )
                # The transpose sidecar is scoped to the autograd use of this
                # quantized tensor. Keeping even weight sidecars past first
                # backward lookup can accumulate one large sidecar per layer
                # per iteration when TE allocates fresh quantized workspaces.
                setattr(_out, _cppmega_mxfp8_tn_sidecar_persistent_attr, False)
                _cppmega_register_mxfp8_sidecar(_out, _sidecar, persistent=False)
                if _te_mxfp8_bwd_debug and _cppmega_attach_debug_count[0] < 16:
                    _cppmega_attach_debug_count[0] += 1
                    print(
                        "[cppmega_fp8_shim] MXFP8 transpose sidecar attached "
                        f"source_shape={tuple(_source.shape)} "
                        f"source_type={type(_source).__name__} "
                        f"out={_cppmega_mxfp8_debug_desc(_out)}"
                    )
            except Exception as _emit_exc:  # pragma: no cover
                _cppmega_record_bwd_stat(
                    "mxfp8_tn_adapter_te_emit_failed",
                    f"{type(_emit_exc).__name__}: {_emit_exc}",
                )
                if _te_mxfp8_transpose_emit_backend == "te":
                    raise
            return _out

        _orig_mxfp8_quantize = getattr(_TE_MXFP8Quantizer, "quantize", None)
        if _orig_mxfp8_quantize is not None and not getattr(
            _orig_mxfp8_quantize, "_cppmega_transpose_emit", False
        ):

            @_functools.wraps(_orig_mxfp8_quantize)
            def _mxfp8_quantize_with_rowwise_transpose(self, tensor, *args, **kwargs):
                _out = _orig_mxfp8_quantize(self, tensor, *args, **kwargs)
                return _cppmega_attach_mxfp8_rowwise_transpose(_out, self, tensor)

            _mxfp8_quantize_with_rowwise_transpose._cppmega_transpose_emit = True
            _TE_MXFP8Quantizer.quantize = _mxfp8_quantize_with_rowwise_transpose

        _orig_mxfp8_update_quantized = getattr(_TE_MXFP8Quantizer, "update_quantized", None)
        if _orig_mxfp8_update_quantized is not None and not getattr(
            _orig_mxfp8_update_quantized, "_cppmega_transpose_emit", False
        ):

            @_functools.wraps(_orig_mxfp8_update_quantized)
            def _mxfp8_update_quantized_with_rowwise_transpose(
                self, src, dst, *args, **kwargs
            ):
                _out = _orig_mxfp8_update_quantized(self, src, dst, *args, **kwargs)
                if _out is None:
                    _out = dst
                return _cppmega_attach_mxfp8_rowwise_transpose(_out, self, src)

            _mxfp8_update_quantized_with_rowwise_transpose._cppmega_transpose_emit = True
            _TE_MXFP8Quantizer.update_quantized = (
                _mxfp8_update_quantized_with_rowwise_transpose
            )

        _orig_tex_split_quantize = getattr(_tex, "split_quantize", None)
        if _orig_tex_split_quantize is not None and not getattr(
            _orig_tex_split_quantize, "_cppmega_transpose_emit", False
        ):

            @_functools.wraps(_orig_tex_split_quantize)
            def _split_quantize_with_rowwise_transpose(
                tensor,
                split_sections,
                quantizers,
                *args,
                **kwargs,
            ):
                _outputs = _orig_tex_split_quantize(
                    tensor,
                    split_sections,
                    quantizers,
                    *args,
                    **kwargs,
                )
                if not (
                    _te_mxfp8_bwd_tn_adapter
                    and _te_mxfp8_transpose_emit_backend in ("auto", "te")
                    and isinstance(tensor, _torch.Tensor)
                    and isinstance(split_sections, (list, tuple))
                    and isinstance(quantizers, (list, tuple))
                    and isinstance(_outputs, (list, tuple))
                ):
                    return _outputs
                _start = 0
                for _out, _size, _quantizer in zip(_outputs, split_sections, quantizers):
                    _size = int(_size)
                    if isinstance(_quantizer, _TE_MXFP8Quantizer) and _cppmega_is_mxfp8_tensor(
                        _out
                    ):
                        _source = tensor.narrow(0, _start, _size)
                        _cppmega_attach_mxfp8_rowwise_transpose(
                            _out,
                            _quantizer,
                            _source,
                        )
                    _start += _size
                return _outputs

            _split_quantize_with_rowwise_transpose._cppmega_transpose_emit = True
            _tex.split_quantize = _split_quantize_with_rowwise_transpose

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
            if getattr(_x, "_te_rowwise_transpose_for_backward_operand", False) or getattr(
                _x, "_cppmega_mxfp8_rowwise_transpose_operand", False
            ):
                _cppmega_record_bwd_stat("mxfp8_tn_adapter_saved_transpose_operand")
                return _x
            _sidecar = _cppmega_get_mxfp8_sidecar(_x)
            if _sidecar is not None:
                _cppmega_mark_rowwise_transpose_operand(_sidecar)
                setattr(_x, _cppmega_mxfp8_tn_sidecar_attr, _sidecar)
                _persistent = bool(
                    getattr(_x, _cppmega_mxfp8_tn_sidecar_persistent_attr, False)
                )
                setattr(_x, _cppmega_mxfp8_tn_sidecar_persistent_attr, _persistent)
            if _sidecar is not None:
                _cppmega_record_bwd_stat("mxfp8_tn_adapter_te_emit")
                return _sidecar
            if (
                _te_mxfp8_transpose_emit_backend == "te"
                and _te_mxfp8_transpose_emit_strict
            ):
                raise ValueError(
                    "MXFP8 TN adapter is configured for TE transpose-emit, but operand "
                    "is missing the rowwise-transpose sidecar; "
                    f"{_cppmega_mxfp8_debug_desc(_x)}"
                )
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
            if _te_mxfp8_transpose_emit_backend == "te":
                _cppmega_record_bwd_stat("mxfp8_tn_adapter_missing_sidecar_copy")
            _cppmega_record_bwd_stat("mxfp8_tn_adapter_copy_transpose")
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

        _cppmega_cutlass_mxfp8_module = [None]

        def _cppmega_load_cutlass_mxfp8_module():
            if _cppmega_cutlass_mxfp8_module[0] is not None:
                return _cppmega_cutlass_mxfp8_module[0]
            from cppmega.megatron import cutlass_mxfp8_gemm as _cutlass_mxfp8_gemm

            _cppmega_cutlass_mxfp8_module[0] = _cutlass_mxfp8_gemm
            return _cutlass_mxfp8_gemm

        def _cppmega_mxfp8_rowwise_payload_and_scale(_x):
            _data = getattr(_x, "_rowwise_data", None)
            _scale = getattr(_x, "_rowwise_scale_inv", None)
            if not isinstance(_data, _torch.Tensor) or not isinstance(_scale, _torch.Tensor):
                raise ValueError("MXFP8 CUTLASS backend requires rowwise data and scales")
            if _data.dim() < 2 or _scale.dim() != 2:
                raise ValueError(
                    "MXFP8 CUTLASS backend requires matrix-like rowwise payloads "
                    f"and 2D scales; {_cppmega_mxfp8_debug_desc(_x)}"
                )
            if _data.dtype != _torch.uint8 or _scale.dtype != _torch.uint8:
                raise ValueError("MXFP8 CUTLASS backend requires uint8 payloads/scales")
            if _data.dim() > 2:
                _data = _data.reshape(-1, _data.shape[-1])
            return _data, _scale

        def _cppmega_mxfp8_colwise_payload_and_scale(_x):
            if getattr(_x, "_with_gemm_swizzled_scales", False):
                raise ValueError("MXFP8 CUTLASS direct backend requires compact, non-swizzled scales")
            _data = getattr(_x, "_columnwise_data", None)
            _scale = getattr(_x, "_columnwise_scale_inv", None)
            if not isinstance(_data, _torch.Tensor) or not isinstance(_scale, _torch.Tensor):
                raise ValueError("MXFP8 CUTLASS direct backend requires columnwise data and scales")
            if _data.dim() < 2 or _scale.dim() != 2:
                raise ValueError(
                    "MXFP8 CUTLASS direct backend requires matrix-like columnwise payloads "
                    f"and 2D scales; {_cppmega_mxfp8_debug_desc(_x)}"
                )
            if _data.dtype != _torch.uint8 or _scale.dtype != _torch.uint8:
                raise ValueError("MXFP8 CUTLASS direct backend requires uint8 payloads/scales")
            if _data.dim() > 2:
                _data = _data.reshape(-1, _data.shape[-1])
            return _data, _scale

        def _cppmega_cutlass_kwargs(_kwargs):
            _out_dtype = _kwargs.get("out_dtype", _torch.bfloat16)
            _out = _kwargs.get("out", None)
            if _out_dtype is not None and _out_dtype != _torch.bfloat16:
                raise ValueError(f"CUTLASS MXFP8 backend requires BF16 out_dtype, got {_out_dtype}")
            if _kwargs.get("bias", None) is not None:
                raise ValueError("CUTLASS MXFP8 backend does not fuse bias/bgrad")
            if _kwargs.get("gelu", False) or _kwargs.get("gelu_in", None) is not None:
                raise ValueError("CUTLASS MXFP8 backend does not fuse GELU")
            if _kwargs.get("quantization_params", None) is not None:
                raise ValueError("CUTLASS MXFP8 backend does not quantize GEMM outputs")
            return {
                "out": _out,
                "accumulate": bool(_kwargs.get("accumulate", False)),
                "alpha": float(_kwargs.get("alpha", 1.0)),
                "beta": _kwargs.get("beta", None),
            }

        def _cppmega_cutlass_tn_gemm(_a, _b, _kwargs):
            _a_data, _a_scale = _cppmega_mxfp8_rowwise_payload_and_scale(_a)
            _b_data, _b_scale = _cppmega_mxfp8_rowwise_payload_and_scale(_b)
            _cutlass = _cppmega_load_cutlass_mxfp8_module()
            _result = _cutlass.tn_gemm(
                _a_data,
                _a_scale,
                _b_data,
                _b_scale,
                **_cppmega_cutlass_kwargs(_kwargs),
            )
            return _result, None, None, None

        def _cppmega_cutlass_dgrad_nn_gemm(_weight, _dy, _kwargs):
            _dy_data, _dy_scale = _cppmega_mxfp8_rowwise_payload_and_scale(_dy)
            _weight_data, _weight_scale = _cppmega_mxfp8_colwise_payload_and_scale(_weight)
            _cutlass = _cppmega_load_cutlass_mxfp8_module()
            _result = _cutlass.dgrad_nn_gemm(
                _dy_data,
                _dy_scale,
                _weight_data,
                _weight_scale,
                **_cppmega_cutlass_kwargs(_kwargs),
            )
            return _result, None, None, None

        def _cppmega_cutlass_wgrad_nt_gemm(_x, _dy, _kwargs):
            _dy_data, _dy_scale = _cppmega_mxfp8_colwise_payload_and_scale(_dy)
            _x_data, _x_scale = _cppmega_mxfp8_colwise_payload_and_scale(_x)
            _cutlass = _cppmega_load_cutlass_mxfp8_module()
            _result = _cutlass.wgrad_nt_gemm(
                _dy_data,
                _dy_scale,
                _x_data,
                _x_scale,
                **_cppmega_cutlass_kwargs(_kwargs),
            )
            return _result, None, None, None

        def _cppmega_try_mxfp8_cutlass_native(_args, _kwargs):
            _layout = _kwargs.get("layout")
            if _layout not in ("NN", "NT"):
                return False, f"unsupported_layout:{_layout}"
            if len(_args) < 2:
                return False, "missing_operands"
            if _cppmega_has_comm_overlap(_kwargs):
                return False, "comm_overlap_not_covered"
            if not (
                _cppmega_is_mxfp8_tensor(_args[0])
                and _cppmega_is_mxfp8_tensor(_args[1])
            ):
                return False, "non_mxfp8_operands"

            if _layout == "NN":
                # dgrad: dy[M, N] @ weight[N, K].
                # CUTLASS direct consumes dy rowwise and aliases original
                # compact TE weight columnwise storage as logical weight.T.
                return True, _cppmega_cutlass_dgrad_nn_gemm(_args[0], _args[1], _kwargs)

            # wgrad: dy.T[N, M] @ x[M, K].
            return True, _cppmega_cutlass_wgrad_nt_gemm(_args[0], _args[1], _kwargs)

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

            if _te_mxfp8_bwd_backend == "cutlass_native":
                try:
                    return _cppmega_try_mxfp8_cutlass_native(_args, _kwargs)
                except Exception as _cutlass_exc:  # pragma: no cover
                    return (
                        False,
                        f"cutlass_native_unavailable:{type(_cutlass_exc).__name__}: "
                        f"{_cutlass_exc}",
                    )

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

        def _cppmega_wrap_general_grouped_gemm(_module):
            _orig_general_grouped_gemm = getattr(_module, "general_grouped_gemm", None)
            if _orig_general_grouped_gemm is None or getattr(
                _orig_general_grouped_gemm, "_cppmega_blockscaled_bwd", False
            ):
                return False

            @_functools.wraps(_orig_general_grouped_gemm)
            def _general_grouped_gemm(A, B, out, *args, **kwargs):
                _layout = kwargs.get("layout", "TN")
                if (
                    not _te_mxfp8_bwd_tn_adapter
                    or not kwargs.get("grad", False)
                    or _layout not in ("NN", "NT")
                    or not isinstance(A, (list, tuple))
                    or not isinstance(B, (list, tuple))
                ):
                    return _orig_general_grouped_gemm(A, B, out, *args, **kwargs)

                def _convert_mxfp8_list(_items):
                    _converted = []
                    _count = 0
                    for _item in _items:
                        if _cppmega_is_mxfp8_tensor(_item):
                            _converted.append(
                                _cppmega_mxfp8_colwise_as_rowwise_transpose(_item)
                            )
                            _count += 1
                        else:
                            _converted.append(_item)
                    return _converted, _count

                _new_kwargs = dict(kwargs)
                _new_kwargs["layout"] = "TN"
                _new_kwargs["use_split_accumulator"] = False
                if _layout == "NN":
                    _new_A, _converted_A = _convert_mxfp8_list(A)
                    if _converted_A == 0:
                        return _orig_general_grouped_gemm(A, B, out, *args, **kwargs)
                    _cppmega_record_bwd_stat("mxfp8_tn_adapter_dgrad")
                    if _te_mxfp8_bwd_debug:
                        print(
                            "[cppmega_fp8_shim] MXFP8 TN adapter grouped dgrad "
                            f"layout=NN->TN converted_A={_converted_A}/{len(A)}"
                        )
                    return _orig_general_grouped_gemm(
                        _new_A, B, out, *args, **_new_kwargs
                    )

                _new_A, _converted_A = _convert_mxfp8_list(A)
                _new_B, _converted_B = _convert_mxfp8_list(B)
                if _converted_A == 0 and _converted_B == 0:
                    return _orig_general_grouped_gemm(A, B, out, *args, **kwargs)
                _cppmega_record_bwd_stat("mxfp8_tn_adapter_wgrad")
                if _te_mxfp8_bwd_debug:
                    print(
                        "[cppmega_fp8_shim] MXFP8 TN adapter grouped wgrad "
                        "layout=NT->TN "
                        f"converted_A={_converted_A}/{len(A)} "
                        f"converted_B={_converted_B}/{len(B)}"
                    )
                return _orig_general_grouped_gemm(
                    _new_A, _new_B, out, *args, **_new_kwargs
                )

            _general_grouped_gemm._cppmega_blockscaled_bwd = True
            _module.general_grouped_gemm = _general_grouped_gemm
            return True

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
                            if _te_mxfp8_bwd_backend == "cutlass_native":
                                _cppmega_record_bwd_stat(f"mxfp8_cutlass_native_{_op_kind}")
                            else:
                                _cppmega_record_bwd_stat(f"mxfp8_tn_adapter_{_op_kind}")
                            if _te_mxfp8_bwd_debug:
                                print(
                                    "[cppmega_fp8_shim] MXFP8 backward backend "
                                    f"{_te_mxfp8_bwd_backend} {_op_kind} "
                                    f"layout={_layout}->TN"
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
                    raise RuntimeError(
                        "MXFP8 TN adapter unavailable and BF16 fallback disabled "
                        f"for {_op_kind}; refusing native MXFP8 {_layout} GEMM on GB10: "
                        f"{_fallback_reason}"
                    )

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
                if (
                    getattr(ctx, "fp8", False)
                    and _cppmega_is_block_scaled_recipe(_recipe)
                    and _te_blockscaled_dequantized_backward
                ):
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

        def _cppmega_force_compact_many_if_needed(_quantizers, _recipe):
            if isinstance(_quantizers, (list, tuple)):
                for _quantizer in _quantizers:
                    _cppmega_force_compact_many_if_needed(_quantizer, _recipe)
                return
            _cppmega_force_compact_if_needed(_quantizers, _recipe)

        def _cppmega_wrap_get_quantizers(_module_cls):
            _orig_get_quantizers = getattr(_module_cls, "_get_quantizers", None)
            if _orig_get_quantizers is None or getattr(
                _orig_get_quantizers, "_cppmega_backward_override", False
            ):
                return False

            @_functools.wraps(_orig_get_quantizers)
            def _get_quantizers(self, *args, **kwargs):
                _quantizers = _orig_get_quantizers(self, *args, **kwargs)
                try:
                    _recipe = _TE_FP8State.get_fp8_recipe()
                    for _q in (_quantizers[0], _quantizers[5]):
                        _cppmega_force_compact_many_if_needed(_q, _recipe)
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
                        _cppmega_force_compact_many_if_needed(_q, _recipe)
                except Exception:
                    pass
                return _quantizers

            _get_weight_quantizers._cppmega_mxfp8_compact = True
            _module_cls._get_weight_quantizers = _get_weight_quantizers
            return True

        def _cppmega_wrap_quantize_weight(_module):
            _orig_quantize_weight = getattr(_module, "quantize_weight", None)
            if _orig_quantize_weight is None or getattr(
                _orig_quantize_weight, "_cppmega_transpose_emit", False
            ):
                return False

            @_functools.wraps(_orig_quantize_weight)
            def _quantize_weight_with_rowwise_transpose(*args, **kwargs):
                _weightmat, _new_workspace = _orig_quantize_weight(*args, **kwargs)
                _source = kwargs.get("tensor", None)
                _quantizer = kwargs.get("quantizer", None)
                if _source is not None and _quantizer is not None:
                    _cppmega_attach_mxfp8_rowwise_transpose(_weightmat, _quantizer, _source)
                    if _new_workspace is not None and _new_workspace is not _weightmat:
                        _cppmega_attach_mxfp8_rowwise_transpose(
                            _new_workspace, _quantizer, _source
                        )
                if (
                    _te_mxfp8_bwd_debug
                    and _cppmega_is_mxfp8_tensor(_weightmat)
                    and _cppmega_quantize_weight_debug_count[0] < 8
                ):
                    _cppmega_quantize_weight_debug_count[0] += 1
                    print(
                        "[cppmega_fp8_shim] quantize_weight MXFP8 "
                        f"source={type(_source).__name__ if _source is not None else None} "
                        f"quantizer={type(_quantizer).__name__ if _quantizer is not None else None} "
                        f"has_sidecar={getattr(_weightmat, _cppmega_mxfp8_tn_sidecar_attr, None) is not None} "
                        f"{_cppmega_mxfp8_debug_desc(_weightmat)}"
                    )
                return _weightmat, _new_workspace

            _quantize_weight_with_rowwise_transpose._cppmega_transpose_emit = True
            _module.quantize_weight = _quantize_weight_with_rowwise_transpose
            return True

        def _cppmega_wrap_gather_along_first_dim(_module):
            _orig_gather = getattr(_module, "gather_along_first_dim", None)
            if _orig_gather is None or getattr(_orig_gather, "_cppmega_sidecar", False):
                return False

            @_functools.wraps(_orig_gather)
            def _gather_along_first_dim_with_sidecar(input_, *args, **kwargs):
                _out, _work = _orig_gather(input_, *args, **kwargs)
                _cppmega_propagate_mxfp8_sidecar(input_, _out)
                return _out, _work

            _gather_along_first_dim_with_sidecar._cppmega_sidecar = True
            _module.gather_along_first_dim = _gather_along_first_dim_with_sidecar
            return True

        def _cppmega_wrap_apply_normalization(_module):
            _orig_apply_normalization = getattr(_module, "apply_normalization", None)
            if _orig_apply_normalization is None or getattr(
                _orig_apply_normalization, "_cppmega_transpose_emit", False
            ):
                return False

            @_functools.wraps(_orig_apply_normalization)
            def _apply_normalization_with_sidecar(
                inputmat,
                ln_out,
                ln_weight,
                ln_bias,
                eps,
                output_quantizer,
                output_dtype,
                normalization,
                fwd_ln_sm_margin,
                zero_centered_gamma,
            ):
                if (
                    _te_mxfp8_bwd_tn_adapter
                    and isinstance(output_quantizer, _TE_MXFP8Quantizer)
                    and _te_mxfp8_transpose_emit_backend in ("auto", "te")
                ):
                    _ln_out, _mu, _rsigma = _orig_apply_normalization(
                        inputmat,
                        ln_out,
                        ln_weight,
                        ln_bias,
                        eps,
                        None,
                        output_dtype,
                        normalization,
                        fwd_ln_sm_margin,
                        zero_centered_gamma,
                    )
                    _cppmega_record_bwd_stat("mxfp8_norm_quantize_sidecar_bridge")
                    return output_quantizer(_ln_out), _mu, _rsigma
                return _orig_apply_normalization(
                    inputmat,
                    ln_out,
                    ln_weight,
                    ln_bias,
                    eps,
                    output_quantizer,
                    output_dtype,
                    normalization,
                    fwd_ln_sm_margin,
                    zero_centered_gamma,
                )

            _apply_normalization_with_sidecar._cppmega_transpose_emit = True
            _module.apply_normalization = _apply_normalization_with_sidecar
            return True

        _patched_gemm_modules = []
        _patched_grouped_gemm_modules = []
        _patched_quantizer_modules = []
        _patched_weight_quantizer_modules = []
        _patched_quantize_weight_modules = []
        _patched_gather_modules = []
        _patched_norm_modules = []
        for _module_name in (
            "transformer_engine.pytorch.module.linear",
            "transformer_engine.pytorch.module.layernorm_linear",
            "transformer_engine.pytorch.module.layernorm_mlp",
            "transformer_engine.pytorch.module.grouped_linear",
            "transformer_engine.pytorch.ops.basic.basic_linear",
            "transformer_engine.pytorch.ops.basic.grouped_linear",
        ):
            try:
                _mod = __import__(_module_name, fromlist=["general_gemm"])
                if _cppmega_wrap_general_gemm(_mod):
                    _patched_gemm_modules.append(_module_name.rsplit(".", 1)[-1])
                if _cppmega_wrap_general_grouped_gemm(_mod):
                    _patched_grouped_gemm_modules.append(
                        _module_name.rsplit(".", 1)[-1]
                    )
                if _cppmega_wrap_quantize_weight(_mod):
                    _patched_quantize_weight_modules.append(_module_name.rsplit(".", 1)[-1])
                if _cppmega_wrap_gather_along_first_dim(_mod):
                    _patched_gather_modules.append(_module_name.rsplit(".", 1)[-1])
                if _cppmega_wrap_apply_normalization(_mod):
                    _patched_norm_modules.append(_module_name.rsplit(".", 1)[-1])
                for _class_name in ("Linear", "LayerNormLinear", "GroupedLinear"):
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
            f"mxfp8_transpose_emit_backend={_te_mxfp8_transpose_emit_backend}, "
            f"mxfp8_transpose_emit_swizzled={_te_mxfp8_transpose_emit_swizzled}, "
            f"mxfp8_bwd_allow_bf16_fallback={_te_mxfp8_bwd_allow_bf16_fallback}, "
            f"gemm_modules={_patched_gemm_modules}, "
            f"grouped_gemm_modules={_patched_grouped_gemm_modules}, "
            f"quantizer_modules={_patched_quantizer_modules}, "
            f"weight_quantizer_modules={_patched_weight_quantizer_modules}, "
            f"quantize_weight_modules={_patched_quantize_weight_modules}, "
            f"gather_modules={_patched_gather_modules}, "
            f"norm_modules={_patched_norm_modules})"
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
