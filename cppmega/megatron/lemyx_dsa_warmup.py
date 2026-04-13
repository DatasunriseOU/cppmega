"""Megatron integration for lemyx/tilelang-dsa fused FA+KL warmup kernel.

The lemyx kernel fuses FlashAttention + lightning-indexer score computation
+ KL divergence into a single TileLang pass. Used for DSA indexer warmup
(first ~1000 steps). After warmup, IndexCache skips indexer on Shared layers
and Full layers use the standard indexer path.

REQUIREMENT: heads == index_heads (set --dsa-indexer-n-heads == --num-attention-heads).
With heads=32 hidden=4096 this is satisfied and FP8-compatible (32 % 8 == 0).

Usage::

    from cppmega.megatron.lemyx_dsa_warmup import apply_lemyx_dsa_patch
    apply_lemyx_dsa_patch()  # call after Megatron imports, before training
"""

from __future__ import annotations

import os
import sys

import torch

_PATCH_MARKER = "__cppmega_lemyx_dsa_patched__"
_DsaWarmupFunc = None


def _get_lemyx_repo():
    """Resolve tilelang-dsa repo path: sibling of cppmega-root."""
    # Same REMOTE_ROOT used everywhere: /mnt/data/cppmega-root
    root = os.environ.get("REMOTE_ROOT", "/mnt/data/cppmega-root")
    return os.path.join(root, "tilelang-dsa")


def _ensure_lemyx_imported():
    global _DsaWarmupFunc
    if _DsaWarmupFunc is not None:
        return

    repo = _get_lemyx_repo()
    assert os.path.isdir(repo), f"tilelang-dsa repo not found at {repo}"

    if repo not in sys.path:
        sys.path.insert(0, repo)

    from kernel_bf16_training_dsa_warmup_lightning_indexer import (
        _DsaWarmupFunc as _Func,
    )

    _DsaWarmupFunc = _Func
    print(f"[cppmega] lemyx _DsaWarmupFunc imported from {repo}")


# ---------------------------------------------------------------------------
# Layout conversion: Megatron SBHD -> lemyx varlen unpadded
# ---------------------------------------------------------------------------

def _to_varlen_unpad(t_sbhd, b, s):
    """Reshape [s, b, ...] -> [b*s, ...] preserving autograd."""
    # permute to [b, s, ...], then flatten batch into sequence.
    dims = list(range(t_sbhd.ndim))
    dims[0], dims[1] = 1, 0  # swap s <-> b
    return t_sbhd.permute(dims).reshape(b * s, *t_sbhd.shape[2:])


def _build_cu_seqlens(b, seqlen, device):
    """Build uniform cu_seqlens [0, seqlen, 2*seqlen, ..., b*seqlen]."""
    return torch.arange(
        0, (b + 1) * seqlen, seqlen, dtype=torch.int32, device=device
    )


# ---------------------------------------------------------------------------
# Drop-in replacement for FusedDSAIndexerLoss
# ---------------------------------------------------------------------------

class _LemyxFusedDSAIndexerLoss:
    """Drop-in for ``FusedDSAIndexerLoss`` using the lemyx fused FA+KL kernel.

    NOT a torch.autograd.Function -- autograd is handled by the inner
    ``_DsaWarmupFunc`` (which IS a torch.autograd.Function).  This class
    only provides a compatible ``apply()`` interface.
    """

    @staticmethod
    def apply(
        q,              # indexer q: [sq, b, index_heads, index_dim]
        weights,        # indexer weights: [sq, b, index_heads]
        k,              # indexer k: [sk, b, index_dim]
        query,          # attention query: [sq, b, np, hn]  (detached)
        key,            # attention key: [sk, b, np, hn]    (detached)
        softmax_scale,
        topk,
        loss_coeff,
        mask,
        sparse_loss,
        pg_collection,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        use_relu: bool = True,
    ):
        _ensure_lemyx_imported()

        sq, b, np_, hn = query.shape
        sk = key.shape[0]
        index_heads = q.shape[2]

        assert np_ == index_heads, (
            f"lemyx fused kernel requires heads == index_heads, "
            f"got {np_} != {index_heads}. Set --dsa-indexer-n-heads {np_}"
        )

        # -- Layout conversion (all ops are differentiable) --
        # All outputs must be contiguous: TileLang static-shape kernels
        # check strides and reject non-C-contiguous tensors.

        # Attention Q: [sq, b, np, hn] -> [b*sq, np, hn]
        q_unpad = _to_varlen_unpad(query, b, sq).contiguous()

        # Attention K: [sk, b, np, hn] -> MQA squeeze head 0 -> [b*sk, hn]
        # Lemyx kernel uses MQA-style K: [total_kv, dim].  In DSA warmup
        # all heads share the same K (nkv=1), so taking head 0 is correct.
        k_unpad = _to_varlen_unpad(key[:, :, 0:1, :], b, sk).squeeze(1).contiguous()

        # Indexer Q: [sq, b, index_heads, index_dim] -> [b*sq, index_heads, index_dim]
        idx_q_unpad = _to_varlen_unpad(q, b, sq).contiguous()

        # Indexer K: [sk, b, index_dim] -> [b*sk, index_dim]
        idx_k_unpad = _to_varlen_unpad(k, b, sk).contiguous()

        # Indexer weights: [sq, b, index_heads] -> [b*sq, index_heads]
        # Lemyx kernel expects float32 weights; Megatron passes bf16.
        idx_w_unpad = _to_varlen_unpad(weights, b, sq).float().contiguous()

        # V is needed by the fused kernel but we discard attention output.
        v_unpad = torch.zeros_like(k_unpad)

        cu_seqlens_q = _build_cu_seqlens(b, sq, query.device)
        cu_seqlens_k = _build_cu_seqlens(b, sk, key.device)

        # -- Run fused kernel --
        # _DsaWarmupFunc.apply handles its own autograd (fwd + bwd).
        # Gradients flow: loss -> dkl -> _DsaWarmupFunc.backward
        #   -> idx_q_unpad -> q,  idx_k_unpad -> k,  idx_w_unpad -> weights
        _o_unpad, dkl = _DsaWarmupFunc.apply(
            q_unpad,            # [total_q, heads, dim]
            k_unpad,            # [total_kv, dim]
            v_unpad,            # [total_kv, dim]
            idx_q_unpad,        # [total_q, index_heads, index_dim]
            idx_k_unpad,        # [total_kv, index_dim]
            cu_seqlens_q,       # [batch+1]
            cu_seqlens_k,       # [batch+1]
            sq,                 # max_seqlen_q
            idx_w_unpad,        # [total_q, index_heads]
        )

        # dkl: [total_q, heads] -- per-position per-head KL divergence.
        # Upstream loss: kl_per_row.mean() * loss_coeff
        loss = dkl.sum(dim=-1).mean() * loss_coeff

        # Compute topk indices via the standard path.  The lemyx kernel
        # does not produce topk; it only fuses the KL loss computation.
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            fused_qk_topk_naive,
        )
        with torch.no_grad():
            _, topk_indices = fused_qk_topk_naive(
                q, k, weights, topk,
                mask=mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
                use_relu=use_relu,
            )

        return topk_indices, loss


setattr(_LemyxFusedDSAIndexerLoss, _PATCH_MARKER, True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_lemyx_dsa_patch():
    """Monkey-patch Megatron DSA to use the lemyx fused FA+KL kernel.

    Replaces ``FusedDSAIndexerLoss`` in the DSA module so that the fused
    TileLang kernel handles the combined attention + KL divergence
    computation in a single pass.

    Crashes on failure — no fallback.
    """
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_mod

    existing = getattr(dsa_mod, "FusedDSAIndexerLoss", None)
    assert existing is not None, "megatron dsa.FusedDSAIndexerLoss not found"

    if getattr(existing, _PATCH_MARKER, False):
        return  # already applied

    _ensure_lemyx_imported()

    dsa_mod.FusedDSAIndexerLoss = _LemyxFusedDSAIndexerLoss
    print(
        f"[cppmega] lemyx DSA fused FA+KL patch applied. "
        f"FusedDSAIndexerLoss -> TileLang fused kernel."
    )
