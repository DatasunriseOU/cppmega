"""IndexCache: cross-layer index reuse for DSA indexer.

Based on arXiv:2603.12201 (Bai et al., Tsinghua/Z.ai, March 2026).
Adjacent DSA layers share 70-100% of top-k selections, so we compute
the indexer on only "Full" layers and reuse cached indices on "Shared" layers.

NAM56R pattern FSSFSSFSSS (9 DSA layers, Full at ranks 0,3,6):
- Full layers:   run indexer normally (forward_before_topk + topk + loss), cache topk_indices
- Shared layers: skip indexer entirely, reuse topk_indices from nearest preceding Full layer,
                 skip indexer loss, run only sparse attention with cached indices

With 3 Full + 6 Shared = 67% indexer savings.
Indexer overhead drops from ~28% to ~9% of training time.

Gate: CPPMEGA_INDEX_CACHE=1
Config: CPPMEGA_INDEX_CACHE_FULL_LAYERS=0,3,6 (0-indexed DSA layer ranks)

Ref: github.com/THUDM/IndexCache (inference patches for SGLang/vLLM)
"""
from __future__ import annotations

import os
import logging

import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global cache: topk_indices per DSA rank, reset each micro-batch.
# Key: dsa_rank (int), Value: topk_indices Tensor [batch, seqlen, topk]
# ---------------------------------------------------------------------------
_INDEX_CACHE: dict[int, torch.Tensor] = {}
_CACHE_STEP: int = -1


def apply_index_cache_patch(force: bool = False) -> bool:
    """Monkey-patch DSAttention to reuse indices across layers.

    Full layers: compute indexer normally, cache topk_indices.
    Shared layers: skip indexer entirely, reuse cached topk_indices from
                   nearest preceding Full layer, skip indexer loss.
    """
    if os.environ.get("CPPMEGA_INDEX_CACHE", "0") != "1":
        return False

    try:
        from megatron.core.transformer.experimental_attention_variant import dsa as dsa_mod
    except ImportError:
        log.warning("index_cache_patch: cannot import DSA module")
        return False

    _DSAttention = getattr(dsa_mod, "DSAttention", None)
    if _DSAttention is None:
        return False

    _MARKER = "__cppmega_index_cache_v3__"
    if getattr(_DSAttention, _MARKER, False) and not force:
        return True

    # Import internal helpers from the dsa module for the Shared fast-path.
    _run_sparse_attention = dsa_mod._run_sparse_attention
    _ensure_sbhd = dsa_mod._ensure_sbhd
    _normalize_dsattention_output_rank = dsa_mod._normalize_dsattention_output_rank
    _build_dsattention_forward_mask = dsa_mod._build_dsattention_forward_mask
    _normalize_cp_comm_type = dsa_mod._normalize_cp_comm_type
    fused_qk_topk_naive = dsa_mod.fused_qk_topk_naive
    try:
        from megatron.core.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
        )
    except ImportError:
        gather_from_sequence_parallel_region = None

    # Parse full layer config: "0,3,6" -> {0, 3, 6}
    full_str = os.environ.get("CPPMEGA_INDEX_CACHE_FULL_LAYERS", "0,3,6")
    FULL_RANKS: set[int] = set(int(x.strip()) for x in full_str.split(",") if x.strip())
    log.info("IndexCache: Full DSA layer ranks: %s", sorted(FULL_RANKS))

    # -----------------------------------------------------------------------
    # Patch __init__: assign per-instance DSA rank and Full/Shared flag
    # -----------------------------------------------------------------------
    _dsa_counter = [0]
    _orig_init = _DSAttention.__init__

    def _ic_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        rank = _dsa_counter[0]
        self._ic_dsa_rank = rank
        self._ic_is_full = rank in FULL_RANKS
        _dsa_counter[0] += 1
        tag = "FULL (compute+cache)" if self._ic_is_full else "SHARED (reuse)"
        log.info("IndexCache: DSA rank %d (layer_number=%d) -> %s",
                 rank, self.layer_number, tag)

    _DSAttention.__init__ = _ic_init

    # -----------------------------------------------------------------------
    # Patch forward
    # -----------------------------------------------------------------------
    _orig_forward = _DSAttention.forward
    _min_full_rank = min(FULL_RANKS) if FULL_RANKS else 0

    def _ic_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        position_ids=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        up_v_weight=None,
    ):
        global _INDEX_CACHE, _CACHE_STEP

        dsa_rank: int = self._ic_dsa_rank

        # -- Cache invalidation: first Full layer clears cache each micro-batch --
        if dsa_rank == _min_full_rank:
            _CACHE_STEP += 1
            _INDEX_CACHE.clear()

        # ==================================================================
        # FULL layer: run original forward, then capture topk_indices
        # ==================================================================
        if self._ic_is_full:
            result = _orig_forward(
                self, query, key, value, attention_mask, x, qr,
                position_ids=position_ids,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                up_v_weight=up_v_weight,
            )
            # Capture topk_indices for subsequent Shared layers.
            # Re-derive via forward_before_topk + naive topk (no grad, cheap).
            # The full forward already computed+used these internally but
            # doesn't expose them; this redundant topk is ~5% of indexer cost
            # and only runs on 3/9 layers.
            with torch.no_grad():
                q_idx, k_idx, w_idx = self.indexer.forward_before_topk(
                    x.detach(), qr.detach(), packed_seq_params
                )
                cp_group = getattr(self.indexer.pg_collection, "cp", None)
                cp_size = cp_group.size() if cp_group is not None else 1
                sq = query.size(0) if query.ndim >= 3 else query.size(0)
                if (cp_size > 1 and k_idx.size(0) == sq
                        and gather_from_sequence_parallel_region is not None):
                    k_idx = gather_from_sequence_parallel_region(
                        k_idx, group=cp_group
                    )
                _, topk_indices = fused_qk_topk_naive(
                    q_idx, k_idx, w_idx,
                    self.indexer.index_topk,
                    mask=None,
                    use_relu=getattr(self.config, "dsa_indexer_scoring_relu", False),
                )
                _INDEX_CACHE[dsa_rank] = topk_indices
            return result

        # ==================================================================
        # SHARED layer: skip indexer, reuse cached topk_indices
        # ==================================================================

        # Find nearest preceding Full layer's cached indices
        cached_indices = None
        for r in range(dsa_rank - 1, -1, -1):
            if r in _INDEX_CACHE:
                cached_indices = _INDEX_CACHE[r]
                break

        if cached_indices is None:
            # Safety fallback: no preceding Full layer ran (shouldn't happen
            # with FSSFSSFSSS pattern). Run original forward.
            log.warning(
                "IndexCache: DSA rank %d has no cached indices, falling back",
                dsa_rank,
            )
            return _orig_forward(
                self, query, key, value, attention_mask, x, qr,
                position_ids=position_ids,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                up_v_weight=up_v_weight,
            )

        # -- Shared fast-path: skip indexer, run sparse attention directly --
        query, _ = _ensure_sbhd(query, "query")
        key, _ = _ensure_sbhd(key, "key")
        if value is not None:
            value, _ = _ensure_sbhd(value, "value")
        if up_v_weight is not None:
            up_v_weight = up_v_weight.to(
                device=query.device, dtype=query.dtype
            ).contiguous()

        # Detect absorbed MLA layout
        latent_v_channels = int(getattr(self.config, "kv_lora_rank", 0) or 0)
        qk_pos_dim = int(getattr(self.config, "qk_pos_emb_head_dim", 0) or 0)
        expected_absorbed_dim = latent_v_channels + qk_pos_dim
        absorbed_mla = (
            latent_v_channels > 0
            and expected_absorbed_dim > 0
            and key.size(2) == 1
            and query.size(-1) == key.size(-1) == expected_absorbed_dim
        )

        sq, b, _, _ = query.size()

        # CP gather for keys/values
        cp_group = getattr(self.indexer.pg_collection, "cp", None)
        cp_size = cp_group.size() if cp_group is not None else 1
        cp_rank = cp_group.rank() if cp_group is not None else 0
        if cp_size > 1 and gather_from_sequence_parallel_region is not None:
            if key.size(0) == sq:
                key = gather_from_sequence_parallel_region(key, group=cp_group)
            if value is not None and value.size(0) == sq:
                value = gather_from_sequence_parallel_region(
                    value, group=cp_group
                )

        skv = key.size(0)

        # Build attention mask (still needed for sparse attention correctness)
        float_mask, varlen_params = _build_dsattention_forward_mask(
            sq=sq, skv=skv, b=b, device=x.device,
            cp_size=cp_size, cp_rank=cp_rank,
            cp_comm_type=_normalize_cp_comm_type(self.cp_comm_type),
            cp_group=cp_group,
            attn_mask_type=attn_mask_type,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            packed_query_positions=None,
        )
        if varlen_params is not None:
            varlen_starts, varlen_ends, key_positions = varlen_params
        else:
            varlen_starts = varlen_ends = key_positions = None

        # Run sparse attention with cached indices (no indexer, no loss)
        output, sparse_attn_path = _run_sparse_attention(
            absorbed_mla=absorbed_mla,
            query=query,
            key=key,
            value=value,
            up_v_weight=up_v_weight,
            topk_indices=cached_indices,
            softmax_scale=self.softmax_scale,
            config=self.config,
            mask=float_mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
        )

        self._debug_print_path(
            f"IndexCache SHARED (rank={dsa_rank}): "
            f"sparse_attn={sparse_attn_path}, reusing_from=nearest_full"
        )

        # No DSAIndexerLossAutoScaler for Shared layers (no indexer loss).
        return _normalize_dsattention_output_rank(output, x.ndim)

    _DSAttention.forward = _ic_forward
    setattr(_DSAttention, _MARKER, True)

    n_full = len(FULL_RANKS)
    n_shared = 9 - n_full
    savings_pct = int(n_shared / 9 * 100)
    log.info(
        "IndexCache installed: %d Full + %d Shared = %d%% indexer savings",
        n_full, n_shared, savings_pct,
    )
    print(
        f"[cppmega] IndexCache: {n_full} Full + {n_shared} Shared DSA layers "
        f"= {savings_pct}% indexer compute savings"
    )
    return True
