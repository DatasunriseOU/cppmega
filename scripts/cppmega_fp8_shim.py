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
# (6) MTP Liger fused linear cross-entropy (env-driven: CPPMEGA_MTP_LIGER_CE=1)
# -----------------------------------------------------------------------------
# Replaces the per-depth output_layer(hidden) + CE pair in process_mtp_loss
# with Liger-Kernel's fused_linear_cross_entropy Triton kernel.  Eliminates
# the [B*S, V] logits materialization (~4.3 GB at B=4,S=4096,V=65536), saving
# ~82% activation memory per MTP depth.  Only active when TP=1.

try:
    from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
    patch_mtp_loss_with_liger()
except Exception as _exc:  # pragma: no cover
    import sys
    print(f"[cppmega_fp8_shim] MTP Liger CE patch failed: {_exc}", file=sys.stderr)


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
        # Fall back to gather_scatter
        try:
            from megatron.core.transformer.experimental_attention_variant import dsa as _dsa_mod
            from cppmega.megatron.dsa_sparse_attention import sparse_dsa_fn as _sparse_dsa_fn

            setattr(_sparse_dsa_fn, "__cppmega_sparse_dsa_patched__", True)
            _dsa_mod.unfused_dsa_fn = _sparse_dsa_fn
            print("[cppmega_fp8_shim] Fallback: gather_scatter sparse_dsa_fn applied")
        except Exception as _exc2:
            print(f"[cppmega_fp8_shim] gather_scatter fallback also failed: {_exc2}", file=sys.stderr)
else:
    # Explicit gather_scatter mode
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
