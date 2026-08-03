"""FA4 vs TE dense-bias forward/backward parity test for H200.

Compares:
- Reference (PRIMARY): TE 2.16 cuDNN FusedAttention with post_scale_bias
  (the real production path on Nebius H200)
- Reference (SECONDARY): manual PyTorch attention with dense [B,1,Sq,Sk]
  post_scale_bias (mathematically equivalent, for environments without TE)
- Test: FA4 flash_attn_func with chunk-native score_mod via production
  factory (_make_graph_score_mod)

On the SAME input (Q, K, V, graph edges), verifies:
- Forward outputs match within bf16 tolerance (~2e-3 atol)
- Backward gradients match (dQ, dK, dV) within bf16 tolerance (~5e-3 atol)
- Loss values match

Runs on GPU (H200 via Modal) or is skipped on CPU-only machines.

Usage:
    # Via pytest on GPU:
    pytest tests/test_fa4_h200_parity.py -v

    # Via Modal:
    modal run tests/test_fa4_h200_parity.py

Design doc: docs/fa4_parity_test_design.md
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import timedelta
from queue import Empty
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip on non-GPU machines
# ---------------------------------------------------------------------------

_HAS_CUDA = torch.cuda.is_available()
_SKIP_REASON = "FA4 parity test requires CUDA (H200)"

# ---------------------------------------------------------------------------
# Detect Transformer Engine availability
# ---------------------------------------------------------------------------

try:
    import transformer_engine.pytorch as te_pytorch  # noqa: F401

    _HAS_TE = True
except ImportError:
    _HAS_TE = False

_TE_SKIP_REASON = (
    "TE reference requires transformer_engine with cuDNN FusedAttention "
    "(TE 2.16+); not installed"
)

from cppmega.megatron.fa4_score_mod_adapter import (
    ChunkNativeGraphBias,
    CppMegaFA4ScoreModAttention,
    build_chunk_native_graph_bias,
    _make_document_causal_mask_mod,
    _make_graph_score_mod,
    _make_graph_score_mod_bwd,
)
from cppmega.megatron.graph_route_attention_bias_patch import (
    _graph_attention_bias_for_layer,
    build_dense_graph_attention_bias_from_structure_batch,
    invalidate_bias_cache,
)
from cppmega.megatron.structure_dataset_patch import (
    _get_current_structure_batch,
    _set_current_structure_batch,
)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

B = 2  # batch size
S = 128  # sequence length (multiple FA4 tiles at tile=64)
H = 8  # attention heads
D = 64  # head dimension
BETA = 2.0  # graph bias beta

# Relation weights
CALL_WEIGHT = 1.0
TYPE_WEIGHT = 1.0
DOMAIN_WEIGHT = 3.0
BUILD_WEIGHT = 4.0

# Tolerances for bf16
FWD_ATOL = 2e-3
FWD_RTOL = 1e-2
BWD_ATOL = 5e-3
BWD_RTOL = 2e-2
# The beta23 custom-mask SM90 reduction had raw H200 outliers up to 0.015625;
# ``assert_close`` still passes its per-element ``atol + rtol * abs(ref)``
# bound. Keep the wider absolute floor local to packed-document tests.
DOCUMENT_BWD_ATOL = 1e-2


@contextmanager
def _structure_batch(batch: dict[str, torch.Tensor]) -> Iterator[None]:
    """Bind sidecars for the exact production forward being exercised."""
    previous = _get_current_structure_batch()
    _set_current_structure_batch(batch)
    try:
        yield
    finally:
        _set_current_structure_batch(previous)


# ---------------------------------------------------------------------------
# Mock structure batch with known graph edges
# ---------------------------------------------------------------------------


def _parity_structure_batch() -> dict[str, torch.Tensor]:
    """Structure batch exercising both chunk-pair and rare token edges.

    Chunks: [0,32), [32,64), [64,96), [96,128)  (4 chunks, 128 tokens)
    Call edges (chunk pairs): (0,2), (1,3), (0,1)
    Type edges (chunk pairs): (2,0), (3,1)
    Domain edges (token triples): (10, 80, 5), (50, 20, 5)
    Build edges (token triples): (100, 5, 7)
    """
    return {
        "graph_call_edges": torch.tensor(
            [[[0, 2], [1, 3], [0, 1], [-1, -1]]], dtype=torch.long
        ),
        "graph_call_edge_counts": torch.tensor([3], dtype=torch.long),
        "graph_type_edges": torch.tensor(
            [[[2, 0], [3, 1], [-1, -1], [-1, -1]]], dtype=torch.long
        ),
        "graph_type_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_domain_edges": torch.tensor(
            [[[10, 80, 5], [50, 20, 5], [-1, -1, -1]]], dtype=torch.long
        ),
        "graph_domain_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_build_edges": torch.tensor(
            [[[100, 5, 7], [-1, -1, -1]]], dtype=torch.long
        ),
        "graph_build_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 32, 64, 96]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[32, 64, 96, 128]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([4], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Reference (PRIMARY): TE cuDNN FusedAttention with post_scale_bias
# ---------------------------------------------------------------------------


def _te_reference_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dense_bias: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """TE DotProductAttention with post_scale_bias (cuDNN FusedAttention).

    This is the PRIMARY reference: the real production path on Nebius H200.

    Args:
        q: [B, S, H, D] queries (bf16, requires_grad)
        k: [B, S, H, D] keys (bf16, requires_grad)
        v: [B, S, H, D] values (bf16, requires_grad)
        dense_bias: [B, 1, Sq, Sk] additive bias (broadcast over heads)
        causal: whether to apply causal masking

    Returns:
        out: [B, S, H, D] attention output
    """
    import transformer_engine.pytorch as te

    # TE DotProductAttention expects [B, S, H, D] layout
    attn = te.DotProductAttention(
        num_attention_heads=H,
        kv_channels=D,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type="causal" if causal else "no_mask",
    )
    attn = attn.to(device=q.device, dtype=q.dtype)

    # TE 2.16 forward: pass qkv_format and max_seqlen explicitly so cuDNN
    # FusedAttention selects the correct backend for post_scale_bias.
    out = attn(
        q,
        k,
        v,
        attention_mask=None,
        qkv_format="bshd",
        max_seqlen_q=q.shape[1],
        max_seqlen_kv=k.shape[1],
        core_attention_bias_type="post_scale_bias",
        core_attention_bias=dense_bias,
    )
    return out.reshape_as(q)


# ---------------------------------------------------------------------------
# Reference (SECONDARY): manual PyTorch attention with dense bias
# ---------------------------------------------------------------------------


def _reference_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dense_bias: torch.Tensor,
    *,
    causal: bool = True,
    document_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Manual scaled dot-product attention with dense post-scale bias.

    Mathematically equivalent to TE cuDNN FusedAttention with post_scale_bias.
    Used as a secondary check when TE is not available.

    Args:
        q: [B, S, H, D] queries (bf16, requires_grad)
        k: [B, S, H, D] keys (bf16, requires_grad)
        v: [B, S, H, D] values (bf16, requires_grad)
        dense_bias: [B, 1, Sq, Sk] additive bias (broadcast over heads)
        causal: whether to apply causal masking

    Returns:
        out: [B, S, H, D] attention output
    """
    # [B, S, H, D] -> [B, H, S, D]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    scale = 1.0 / math.sqrt(D)

    # scores: [B, H, Sq, Sk]
    scores = torch.matmul(q_t.float(), k_t.float().transpose(-2, -1)) * scale

    # Add dense bias [B, 1, Sq, Sk] -> broadcasts over H
    scores = scores + dense_bias.float()

    if document_ids is not None:
        expected_shape = (q.shape[0], k.shape[1])
        if tuple(document_ids.shape) != expected_shape:
            raise ValueError(
                f"document_ids shape {tuple(document_ids.shape)} != "
                f"{expected_shape}"
            )
        query_start = k.shape[1] - q.shape[1]
        query_document_ids = document_ids[:, query_start : query_start + q.shape[1]]
        same_document = (
            query_document_ids[:, None, :, None] == document_ids[:, None, None, :]
        )
        scores = scores.masked_fill(~same_document, float("-inf"))

    # Causal mask: -inf above diagonal
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        query_start = sk - sq
        causal_mask = (
            torch.arange(sk, device=scores.device)[None, :]
            > torch.arange(sq, device=scores.device)[:, None] + query_start
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_t.float())  # [B, H, Sq, D]

    # [B, H, S, D] -> [B, S, H, D]
    return out.transpose(1, 2).to(q.dtype)


# ---------------------------------------------------------------------------
# FA4 attention forward (uses production factory _make_graph_score_mod)
# ---------------------------------------------------------------------------


def _fa4_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias_state: ChunkNativeGraphBias,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """FA4 flash_attn_func with production _make_graph_score_mod factory.

    Builds the 6 flat aux_tensors and creates the score_mod via the
    production factory with c_plus_1 captured in closure.

    Args:
        q: [B, S, H, D] queries (bf16, requires_grad)
        k: [B, S, H, D] keys (bf16, requires_grad)
        v: [B, S, H, D] values (bf16, requires_grad)
        bias_state: ChunkNativeGraphBias from build_chunk_native_graph_bias
        causal: whether to apply causal masking

    Returns:
        out: [B, S, H, D] attention output
    """
    from flash_attn.cute.interface import flash_attn_func

    scale = 1.0 / math.sqrt(D)

    # c_plus_1 = chunk_bias square dimension (C+1)
    c_plus_1 = bias_state.chunk_bias.shape[1]

    # Flatten chunk_bias [B, C+1, C+1] -> [B, (C+1)*(C+1)]
    chunk_bias_flat = bias_state.chunk_bias.reshape(
        bias_state.chunk_bias.shape[0], -1
    ).contiguous()

    # 6 flat aux_tensors (c_plus_1 is in the factory closure, not in the list)
    # Layout: [token_to_chunk_q, token_to_chunk_k, chunk_bias_flat,
    #          rare_row_offsets, rare_k, rare_w]
    aux_tensors = [
        bias_state.token_to_chunk_q,
        bias_state.token_to_chunk_k,
        chunk_bias_flat,
        bias_state.rare_row_offsets,
        bias_state.rare_k,
        bias_state.rare_w,
    ]

    # Create score_mod via production factory
    max_rare = int(bias_state.rare_k.shape[1])
    score_mod = _make_graph_score_mod(c_plus_1, max_rare)
    score_mod_bwd = _make_graph_score_mod_bwd(c_plus_1)

    out = flash_attn_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=scale,
        causal=causal,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        aux_tensors=aux_tensors,
        block_sparse_tensors=None,
        mask_mod=None,
        return_lse=False,
    )

    if isinstance(out, tuple):
        out = out[0]

    return out


def _production_fa4_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    structure_batch: dict[str, torch.Tensor],
    bias_state: ChunkNativeGraphBias | None,
) -> torch.Tensor:
    """Exercise the real Megatron-layout production module and sidecar binding."""
    module = CppMegaFA4ScoreModAttention(
        num_attention_heads=q.shape[2],
        head_dim=q.shape[3],
        softmax_scale=1.0 / math.sqrt(q.shape[3]),
    )
    with _structure_batch(structure_batch):
        output = module(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            attention_bias=bias_state,
        )
    sequence_length, batch_size, _hidden_size = output.shape
    return output.reshape(
        sequence_length, batch_size, q.shape[2], q.shape[3]
    ).transpose(0, 1)


def _fa4_context_parallel_worker(
    rank: int,
    init_method: str,
    results,
) -> None:
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(
            "nccl",
            init_method=init_method,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=180),
        )
        cp_group = torch.distributed.group.WORLD

        from megatron.core.ssm.mamba_context_parallel import (
            _redo_attention_load_balancing,
        )

        batch_size = 1
        torch.manual_seed(420)
        global_q, global_k, global_v = (
            torch.randn(
                batch_size,
                S,
                H,
                D,
                device=device,
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        )
        document_ids = torch.tensor(
            [[1] * (S // 2) + [2] * (S // 2)],
            device=device,
            dtype=torch.int32,
        )
        structure_batch = _parity_structure_batch()
        structure_batch["document_ids"] = document_ids
        bias_state = build_chunk_native_graph_bias(
            structure_batch,
            batch_size=batch_size,
            seqlen_q=S,
            seqlen_k=S,
            device=device,
            dtype=torch.bfloat16,
            beta=BETA,
            call_weight=CALL_WEIGHT,
            type_weight=TYPE_WEIGHT,
            domain_weight=DOMAIN_WEIGHT,
            build_weight=BUILD_WEIGHT,
        )

        reference_q, reference_k, reference_v = (
            tensor.detach().clone().requires_grad_(True)
            for tensor in (global_q, global_k, global_v)
        )
        reference_module = CppMegaFA4ScoreModAttention(
            config=SimpleNamespace(
                attention_dropout=0.0,
                context_parallel_size=1,
                sequence_parallel=False,
            ),
            layer_number=1,
            num_attention_heads=H,
            head_dim=D,
            pg_collection=SimpleNamespace(cp=None),
        )
        with _structure_batch(structure_batch):
            reference_output = reference_module(
                reference_q.transpose(0, 1),
                reference_k.transpose(0, 1),
                reference_v.transpose(0, 1),
                attention_bias=bias_state,
            )
        torch.manual_seed(421)
        output_probe = torch.randn_like(reference_output)
        (reference_output * output_probe).sum().backward()

        def local_zigzag(tensor: torch.Tensor) -> torch.Tensor:
            balanced = _redo_attention_load_balancing(
                tensor.transpose(0, 1),
                2,
                packed_seq_params=None,
            )
            return balanced.chunk(2, dim=0)[rank].contiguous()

        local_q, local_k, local_v = (
            local_zigzag(tensor).detach().clone().requires_grad_(True)
            for tensor in (global_q, global_k, global_v)
        )
        cp_module = CppMegaFA4ScoreModAttention(
            config=SimpleNamespace(
                attention_dropout=0.0,
                context_parallel_size=2,
                sequence_parallel=False,
            ),
            layer_number=1,
            num_attention_heads=H,
            head_dim=D,
            pg_collection=SimpleNamespace(cp=cp_group),
        )
        with _structure_batch(structure_batch):
            local_output = cp_module(
                local_q,
                local_k,
                local_v,
                attention_bias=bias_state,
            )

        expected_local_output = local_zigzag(
            reference_output.transpose(0, 1)
        ).reshape_as(local_output)
        local_probe = local_zigzag(
            output_probe.transpose(0, 1)
        ).reshape_as(local_output)
        torch.testing.assert_close(
            local_output,
            expected_local_output,
            atol=3e-3,
            rtol=FWD_RTOL,
        )
        (local_output * local_probe).sum().backward()

        for name, local_tensor, reference_tensor in (
            ("dQ", local_q, reference_q),
            ("dK", local_k, reference_k),
            ("dV", local_v, reference_v),
        ):
            if local_tensor.grad is None or reference_tensor.grad is None:
                raise RuntimeError(f"FA4 CP {name} gradient is missing")
            expected_local_grad = local_zigzag(reference_tensor.grad)
            torch.testing.assert_close(
                local_tensor.grad,
                expected_local_grad,
                atol=DOCUMENT_BWD_ATOL,
                rtol=BWD_RTOL,
            )

        torch.distributed.barrier()
        results.put(("ok", rank))
    except Exception:
        results.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        _set_current_structure_batch(None)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _local_tp_cp_sequence(
    tensor: torch.Tensor,
    *,
    tp_rank: int,
    tp_size: int,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Apply Megatron's CP zigzag shard, then its contiguous TP/SP shard."""
    local = tensor
    if cp_size > 1:
        from megatron.core.ssm.mamba_context_parallel import (
            _redo_attention_load_balancing,
        )

        local = _redo_attention_load_balancing(
            local,
            cp_size,
            packed_seq_params=None,
        ).chunk(cp_size, dim=0)[cp_rank]
    if tp_size > 1:
        local = local.chunk(tp_size, dim=0)[tp_rank]
    return local.contiguous()


def _fa4_tp_sp_worker(
    rank: int,
    world_size: int,
    cp_size: int,
    init_method: str,
    results,
) -> None:
    """Exercise the production TE QKV/FA4/TE projection SP+CP composition."""
    parallel_state = None
    try:
        import inspect
        import json
        from importlib import metadata

        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(
            "nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=300),
        )

        from flash_attn.cute.interface import flash_attn_func
        from megatron.core import parallel_state
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TERowParallelLinear,
        )
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.random import (
            model_parallel_cuda_manual_seed,
        )
        from megatron.core.transformer.attention import (
            SelfAttention,
            SelfAttentionSubmodules,
        )
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.transformer_config import TransformerConfig

        tp_size = 2
        if world_size != tp_size * cp_size:
            raise RuntimeError(
                f"world_size={world_size} != tp_size*cp_size={tp_size * cp_size}"
            )
        device_name = torch.cuda.get_device_name(rank)
        if "H200" not in device_name:
            raise RuntimeError(f"expected H200 on rank {rank}, got {device_name!r}")
        if metadata.version("flash-attn-4") != "4.0.0b23":
            raise RuntimeError(
                "FA4 SP/CP gate requires flash-attn-4==4.0.0b23, got "
                f"{metadata.version('flash-attn-4')}"
            )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(0xFA423)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "cp", "pp"],
        )
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
        actual_tp_size = torch.distributed.get_world_size(tp_group)
        tp_rank = torch.distributed.get_rank(tp_group)
        actual_cp_size = (
            1
            if cp_group is None
            else torch.distributed.get_world_size(cp_group)
        )
        cp_rank = (
            0
            if cp_group is None
            else torch.distributed.get_rank(cp_group)
        )
        if (actual_tp_size, actual_cp_size) != (tp_size, cp_size):
            raise RuntimeError(
                "unexpected process mesh: "
                f"actual TPxCP={actual_tp_size}x{actual_cp_size}, "
                f"expected {tp_size}x{cp_size}"
            )

        for name, value in (
            ("CPPMEGA_STRUCTURE_ENABLED", "1"),
            ("CPPMEGA_GRAPH_ROUTES_ENABLED", "1"),
            ("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", "1"),
            ("CPPMEGA_FA4_SCORE_MOD", "1"),
            ("CPPMEGA_GRAPH_BIAS_BETA", str(BETA)),
        ):
            os.environ[name] = value

        hidden_size = H * D

        def config(*, sequence_parallel: bool, context_parallel_size: int):
            return TransformerConfig(
                num_layers=1,
                hidden_size=hidden_size,
                num_attention_heads=H,
                num_query_groups=H,
                ffn_hidden_size=4 * hidden_size,
                tensor_model_parallel_size=tp_size,
                pipeline_model_parallel_size=1,
                context_parallel_size=context_parallel_size,
                sequence_parallel=sequence_parallel,
                attention_dropout=0.0,
                params_dtype=torch.bfloat16,
                bf16=True,
                add_bias_linear=False,
                use_cpu_initialization=False,
            )

        submodules = SelfAttentionSubmodules(
            linear_qkv=TELayerNormColumnParallelLinear,
            core_attention=CppMegaFA4ScoreModAttention,
            linear_proj=TERowParallelLinear,
        )
        reference_config = config(
            sequence_parallel=False,
            context_parallel_size=1,
        )
        reference_pgs = ProcessGroupCollection(
            tp=tp_group,
            cp=None,
            pp=pg_collection.pp,
        )
        torch.manual_seed(0xFA424)
        torch.cuda.manual_seed_all(0xFA424)
        reference_attention = SelfAttention(
            config=reference_config,
            submodules=submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=reference_pgs,
        ).to(device)
        reference_attention.train()

        # CP ranks are replicas of the same TP partition. Keep their local
        # parameters bit-identical before comparing CP-sharded gradients.
        if cp_size > 1:
            cp_ranks = torch.distributed.get_process_group_ranks(cp_group)
            for parameter in reference_attention.parameters():
                torch.distributed.broadcast(
                    parameter.data,
                    src=cp_ranks[0],
                    group=cp_group,
                )

        test_config = config(
            sequence_parallel=True,
            context_parallel_size=cp_size,
        )
        test_attention = SelfAttention(
            config=test_config,
            submodules=submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_collection,
        ).to(device)
        test_attention.load_state_dict(reference_attention.state_dict())
        test_attention.train()

        torch.manual_seed(0xFA425)
        global_hidden = torch.randn(
            S,
            1,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        torch.distributed.broadcast(global_hidden, src=0)
        document_ids = torch.tensor(
            [[1] * (S // 2) + [2] * (S // 2)],
            device=device,
            dtype=torch.int32,
        )
        structure_batch = _parity_structure_batch()
        structure_batch["document_ids"] = document_ids
        position_ids = torch.arange(S, device=device).view(S, 1, 1)
        local_positions = _local_tp_cp_sequence(
            position_ids,
            tp_rank=tp_rank,
            tp_size=tp_size,
            cp_rank=cp_rank,
            cp_size=cp_size,
        ).flatten()
        local_document_ids = document_ids[0].index_select(0, local_positions)

        reference_input = global_hidden.detach().clone().requires_grad_(True)
        local_input = (
            _local_tp_cp_sequence(
                global_hidden,
                tp_rank=tp_rank,
                tp_size=tp_size,
                cp_rank=cp_rank,
                cp_size=cp_size,
            )
            .detach()
            .clone()
            .requires_grad_(True)
        )

        # Exercise the real TE fused QKV projection independently so the
        # receipt records its actual sequence shape before CP gather.
        with torch.no_grad():
            projected_q, projected_k, projected_v = (
                test_attention.get_query_key_value_tensors(local_input.detach())
            )
        expected_projected_length = S // cp_size
        for name, projected in (
            ("Q", projected_q),
            ("K", projected_k),
            ("V", projected_v),
        ):
            if projected.shape[0] != expected_projected_length:
                raise RuntimeError(
                    f"TE sequence-parallel {name} length={projected.shape[0]}, "
                    f"expected CP-local length={expected_projected_length}"
                )

        invalidate_bias_cache()
        layer = SimpleNamespace(
            config=test_config,
            self_attention=test_attention,
        )
        with _structure_batch(structure_batch):
            bias_state = _graph_attention_bias_for_layer(layer, local_input)
            if not isinstance(bias_state, ChunkNativeGraphBias):
                raise RuntimeError(
                    f"expected ChunkNativeGraphBias, got {type(bias_state).__name__}"
                )
            reference_output, _ = reference_attention(
                reference_input,
                None,
                attention_bias=bias_state,
            )
            local_output, _ = test_attention(
                local_input,
                None,
                attention_bias=bias_state,
            )

        expected_local_output = _local_tp_cp_sequence(
            reference_output.detach(),
            tp_rank=tp_rank,
            tp_size=tp_size,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )
        torch.testing.assert_close(
            local_output,
            expected_local_output,
            atol=3e-2,
            rtol=5e-2,
        )

        torch.manual_seed(0xFA426)
        output_probe = torch.randn_like(reference_output)
        torch.distributed.broadcast(output_probe, src=0)
        local_probe = _local_tp_cp_sequence(
            output_probe,
            tp_rank=tp_rank,
            tp_size=tp_size,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )
        (reference_output * output_probe).sum().backward()
        (local_output * local_probe).sum().backward()
        if reference_input.grad is None or local_input.grad is None:
            raise RuntimeError("FA4 SP/CP input gradient is missing")
        expected_local_input_grad = _local_tp_cp_sequence(
            reference_input.grad,
            tp_rank=tp_rank,
            tp_size=tp_size,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )
        torch.testing.assert_close(
            local_input.grad,
            expected_local_input_grad,
            atol=8e-2,
            rtol=1e-1,
        )

        reference_parameters = dict(reference_attention.named_parameters())
        max_parameter_grad_diff = torch.zeros((), device=device)
        finalized_sequence_parallel_parameters = []
        for name, parameter in test_attention.named_parameters():
            reference_parameter = reference_parameters[name]
            if parameter.grad is None or reference_parameter.grad is None:
                raise RuntimeError(f"missing parameter gradient for {name}")
            actual_grad = parameter.grad.detach().clone()
            if getattr(parameter, "sequence_parallel", False):
                torch.distributed.all_reduce(actual_grad, group=tp_group)
                finalized_sequence_parallel_parameters.append(name)
            if cp_size > 1:
                torch.distributed.all_reduce(actual_grad, group=cp_group)
            torch.testing.assert_close(
                actual_grad,
                reference_parameter.grad,
                atol=8e-2,
                rtol=1e-1,
                msg=lambda message, name=name: f"{name}: {message}",
            )
            max_parameter_grad_diff = torch.maximum(
                max_parameter_grad_diff,
                (actual_grad.float() - reference_parameter.grad.float()).abs().max()
            )

        # A perturbation confined to document 1 must not change any document
        # 2 output, through the complete TP/SP(+CP) production module.
        perturbed_global = global_hidden.detach().clone()
        feature_perturbation = torch.linspace(
            -16,
            16,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        ).view(1, 1, hidden_size)
        perturbed_global[: S // 2].add_(feature_perturbation)
        perturbed_local = _local_tp_cp_sequence(
            perturbed_global,
            tp_rank=tp_rank,
            tp_size=tp_size,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )
        with torch.no_grad(), _structure_batch(structure_batch):
            perturbed_output, _ = test_attention(
                perturbed_local,
                None,
                attention_bias=bias_state,
            )
        document_two = local_document_ids == 2
        local_leakage = torch.zeros((), device=device)
        if document_two.any():
            local_leakage = (
                perturbed_output[document_two].float()
                - local_output.detach()[document_two].float()
            ).abs().max()
        torch.distributed.all_reduce(
            local_leakage,
            op=torch.distributed.ReduceOp.MAX,
        )
        if local_leakage.item() != 0:
            raise AssertionError(
                f"packed-document forward leakage={local_leakage.item()}"
            )
        local_own_document_delta = torch.zeros((), device=device)
        document_one = local_document_ids == 1
        if document_one.any():
            local_own_document_delta = (
                perturbed_output[document_one].float()
                - local_output.detach()[document_one].float()
            ).abs().max()
        torch.distributed.all_reduce(
            local_own_document_delta,
            op=torch.distributed.ReduceOp.MAX,
        )
        if local_own_document_delta.item() <= 0:
            raise AssertionError(
                "feature-varying document-1 perturbation did not change its "
                "own output"
            )

        # Backpropagate only from document 2 and demand exactly zero gradient
        # in document 1 while proving the ordinary gradient path stays live.
        test_attention.zero_grad(set_to_none=True)
        cross_input = (
            _local_tp_cp_sequence(
                global_hidden,
                tp_rank=tp_rank,
                tp_size=tp_size,
                cp_rank=cp_rank,
                cp_size=cp_size,
            )
            .detach()
            .clone()
            .requires_grad_(True)
        )
        with _structure_batch(structure_batch):
            cross_output, _ = test_attention(
                cross_input,
                None,
                attention_bias=bias_state,
            )
        cross_probe = torch.randn_like(cross_output)
        cross_probe[local_document_ids != 2] = 0
        (cross_output * cross_probe).sum().backward()
        if cross_input.grad is None:
            raise RuntimeError("FA4 SP/CP isolation gradient is missing")
        local_cross_grad = torch.zeros((), device=device)
        if document_one.any():
            local_cross_grad = cross_input.grad[document_one].abs().max()
        torch.distributed.all_reduce(
            local_cross_grad,
            op=torch.distributed.ReduceOp.MAX,
        )
        if local_cross_grad.item() != 0:
            raise AssertionError(
                f"packed-document cross gradient={local_cross_grad.item()}"
            )

        ordinary_grad_l1 = cross_input.grad[document_two].float().abs().sum()
        torch.distributed.all_reduce(ordinary_grad_l1)
        if ordinary_grad_l1.item() <= 0:
            raise AssertionError("ordinary document-2 input gradient is zero")
        live_parameter_grads = {}
        for suffix in ("linear_qkv.weight", "linear_proj.weight"):
            matches = [
                parameter.grad
                for name, parameter in test_attention.named_parameters()
                if name.endswith(suffix)
            ]
            if len(matches) != 1 or matches[0] is None:
                raise RuntimeError(f"missing unique live gradient for {suffix}")
            grad_l1 = matches[0].float().abs().sum()
            torch.distributed.all_reduce(grad_l1)
            if grad_l1.item() <= 0:
                raise AssertionError(f"ordinary {suffix} gradient is zero")
            live_parameter_grads[suffix] = grad_l1.item()

        forward_diff = (
            local_output.float() - expected_local_output.float()
        ).abs().max()
        input_grad_diff = (
            local_input.grad.float() - expected_local_input_grad.float()
        ).abs().max()
        for metric in (forward_diff, input_grad_diff, max_parameter_grad_diff):
            torch.distributed.all_reduce(
                metric,
                op=torch.distributed.ReduceOp.MAX,
            )
        diagnostic = {
            "backend_distribution": "flash-attn-4",
            "backend_version": metadata.version("flash-attn-4"),
            "backend_callable_file": inspect.getfile(flash_attn_func),
            "device": device_name,
            "global_rank": rank,
            "world_size": world_size,
            "tp_rank": tp_rank,
            "tp_size": actual_tp_size,
            "cp_rank": cp_rank,
            "cp_size": actual_cp_size,
            "global_hidden_shape": list(global_hidden.shape),
            "local_hidden_shape": list(local_input.shape),
            "projected_q_shape_before_cp_gather": list(projected_q.shape),
            "graph_bias_q_shape": list(bias_state.token_to_chunk_q.shape),
            "local_output_shape": list(local_output.shape),
            "forward_max_abs": forward_diff.item(),
            "input_grad_max_abs": input_grad_diff.item(),
            "parameter_grad_max_abs": max_parameter_grad_diff.item(),
            "finalized_sequence_parallel_parameters": (
                finalized_sequence_parallel_parameters
            ),
            "forward_cross_document_leakage_max": local_leakage.item(),
            "own_document_output_delta_max": local_own_document_delta.item(),
            "cross_document_input_grad_max": local_cross_grad.item(),
            "ordinary_input_grad_l1": ordinary_grad_l1.item(),
            "ordinary_parameter_grad_l1": live_parameter_grads,
        }
        torch.distributed.barrier()
        results.put(("ok", rank, diagnostic))
    except Exception:
        results.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        _set_current_structure_batch(None)
        if parallel_state is not None:
            try:
                parallel_state.destroy_model_parallel()
            except Exception:
                pass
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _join_distributed_workers(
    processes,
    results,
    *,
    timeout_seconds: float,
) -> list:
    """Join one worker world by a common deadline and reap every timeout."""

    deadline = time.monotonic() + timeout_seconds
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))

    timed_out = [process for process in processes if process.is_alive()]
    for process in timed_out:
        process.terminate()
    for process in timed_out:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)

    messages = []
    result_deadline = time.monotonic() + (1 if timed_out else 10)
    for _index in range(len(processes)):
        try:
            messages.append(
                results.get(timeout=max(0.0, result_deadline - time.monotonic()))
            )
        except Empty:
            break
    results.close()
    results.join_thread()

    exitcodes = [process.exitcode for process in processes]
    if timed_out:
        raise AssertionError(
            "FA4 distributed workers timed out after "
            f"{timeout_seconds:g}s; timed_out_pids="
            f"{[process.pid for process in timed_out]}; "
            f"exitcodes={exitcodes}; messages={messages!r}"
        )
    if len(messages) != len(processes):
        raise AssertionError(
            f"expected {len(processes)} FA4 worker reports, got "
            f"{len(messages)}; exitcodes={exitcodes}; messages={messages!r}"
        )
    return messages


def _run_fa4_tp_sp_world(
    *,
    cp_size: int,
    init_method: str,
) -> None:
    tp_size = 2
    world_size = tp_size * cp_size
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_fa4_tp_sp_worker,
            args=(rank, world_size, cp_size, init_method, results),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    messages = _join_distributed_workers(
        processes,
        results,
        timeout_seconds=600,
    )
    assert all(process.exitcode == 0 for process in processes), messages
    assert sorted((message[0], message[1]) for message in messages) == [
        ("ok", rank) for rank in range(world_size)
    ]
    for _, _, diagnostic in sorted(messages, key=lambda message: message[1]):
        print(
            "CPPMEGA_FA4_SP_CP_DIAGNOSTIC "
            + json.dumps(diagnostic, sort_keys=True),
            flush=True,
        )


# ---------------------------------------------------------------------------
# Parity test class
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason=_SKIP_REASON)
class TestFA4H200Parity:
    """Forward/backward parity: FA4 score_mod vs TE dense bias on H200."""

    def _setup_inputs(self, device: torch.device):
        """Create Q, K, V and both bias representations on device."""
        torch.manual_seed(42)

        q = torch.randn(
            B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True
        )

        sb = _parity_structure_batch()

        # Build dense bias (TE reference path) — must match QKV dtype (bf16)
        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=B,
            seqlen_q=S,
            seqlen_k=S,
            device=device,
            dtype=torch.bfloat16,
            call_weight=CALL_WEIGHT,
            type_weight=TYPE_WEIGHT,
            domain_weight=DOMAIN_WEIGHT,
            build_weight=BUILD_WEIGHT,
            beta=BETA,
        )

        # Build chunk-native bias (FA4 path)
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=B,
            seqlen_q=S,
            seqlen_k=S,
            device=device,
            dtype=torch.float32,
            beta=BETA,
            call_weight=CALL_WEIGHT,
            type_weight=TYPE_WEIGHT,
            domain_weight=DOMAIN_WEIGHT,
            build_weight=BUILD_WEIGHT,
        )

        return q, k, v, dense_bias, chunk_native

    def _get_reference_output(self, q, k, v, dense_bias):
        """Get reference output using TE (primary) or manual PyTorch (secondary)."""
        if _HAS_TE:
            return _te_reference_attention_forward(q, k, v, dense_bias, causal=True)
        return _reference_attention_forward(q, k, v, dense_bias, causal=True)

    @pytest.mark.skipif(not _HAS_TE, reason=_TE_SKIP_REASON)
    def test_forward_parity_te(self):
        """FA4 forward output matches TE cuDNN FusedAttention (PRIMARY ref)."""
        device = torch.device("cuda")
        q, k, v, dense_bias, chunk_native = self._setup_inputs(device)

        # PRIMARY Reference: TE cuDNN FusedAttention with post_scale_bias
        out_ref = _te_reference_attention_forward(q, k, v, dense_bias, causal=True)

        # Test: FA4 with production score_mod factory
        out_fa4 = _fa4_attention_forward(q, k, v, chunk_native, causal=True)

        assert (
            out_ref.shape == out_fa4.shape
        ), f"Shape mismatch: ref={out_ref.shape}, fa4={out_fa4.shape}"

        torch.testing.assert_close(
            out_fa4,
            out_ref,
            atol=FWD_ATOL,
            rtol=FWD_RTOL,
            msg="FA4 forward output diverges from TE cuDNN FusedAttention",
        )

    def test_forward_parity(self):
        """FA4 forward output matches reference within bf16 tol."""
        device = torch.device("cuda")
        q, k, v, dense_bias, chunk_native = self._setup_inputs(device)

        # Reference: TE (primary) or manual PyTorch (secondary)
        out_ref = self._get_reference_output(q, k, v, dense_bias)

        # Test: FA4 with production score_mod factory
        out_fa4 = _fa4_attention_forward(q, k, v, chunk_native, causal=True)

        assert (
            out_ref.shape == out_fa4.shape
        ), f"Shape mismatch: ref={out_ref.shape}, fa4={out_fa4.shape}"

        torch.testing.assert_close(
            out_fa4,
            out_ref,
            atol=FWD_ATOL,
            rtol=FWD_RTOL,
            msg="FA4 forward output diverges from dense-bias reference",
        )

    def test_backward_parity_dq(self):
        """FA4 dQ gradient matches reference within bf16 tol."""
        device = torch.device("cuda")

        # Reference path
        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = self._get_reference_output(q_ref, k_ref, v_ref, dense_bias)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        dq_ref = q_ref.grad.clone()

        # FA4 path (fresh tensors, same seed)
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _fa4_attention_forward(q_fa4, k_fa4, v_fa4, chunk_native, causal=True)
        loss_fa4 = out_fa4.sum()
        loss_fa4.backward()
        dq_fa4 = q_fa4.grad.clone()

        torch.testing.assert_close(
            dq_fa4,
            dq_ref,
            atol=BWD_ATOL,
            rtol=BWD_RTOL,
            msg="FA4 dQ gradient diverges from dense-bias reference",
        )

    def test_backward_parity_dk(self):
        """FA4 dK gradient matches reference within bf16 tol."""
        device = torch.device("cuda")

        # Reference path
        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = self._get_reference_output(q_ref, k_ref, v_ref, dense_bias)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        dk_ref = k_ref.grad.clone()

        # FA4 path
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _fa4_attention_forward(q_fa4, k_fa4, v_fa4, chunk_native, causal=True)
        loss_fa4 = out_fa4.sum()
        loss_fa4.backward()
        dk_fa4 = k_fa4.grad.clone()

        torch.testing.assert_close(
            dk_fa4,
            dk_ref,
            atol=BWD_ATOL,
            rtol=BWD_RTOL,
            msg="FA4 dK gradient diverges from dense-bias reference",
        )

    def test_backward_parity_dv(self):
        """FA4 dV gradient matches reference within bf16 tol."""
        device = torch.device("cuda")

        # Reference path
        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = self._get_reference_output(q_ref, k_ref, v_ref, dense_bias)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        dv_ref = v_ref.grad.clone()

        # FA4 path
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _fa4_attention_forward(q_fa4, k_fa4, v_fa4, chunk_native, causal=True)
        loss_fa4 = out_fa4.sum()
        loss_fa4.backward()
        dv_fa4 = v_fa4.grad.clone()

        torch.testing.assert_close(
            dv_fa4,
            dv_ref,
            atol=BWD_ATOL,
            rtol=BWD_RTOL,
            msg="FA4 dV gradient diverges from dense-bias reference",
        )

    def test_loss_parity(self):
        """FA4 loss matches reference loss."""
        device = torch.device("cuda")

        # Reference path
        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = self._get_reference_output(q_ref, k_ref, v_ref, dense_bias)
        loss_ref = out_ref.sum()

        # FA4 path
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _fa4_attention_forward(q_fa4, k_fa4, v_fa4, chunk_native, causal=True)
        loss_fa4 = out_fa4.sum()

        # Loss is a scalar; compare with tight tolerance relative to magnitude
        loss_ref_val = loss_ref.item()
        loss_fa4_val = loss_fa4.item()
        rel_err = abs(loss_fa4_val - loss_ref_val) / (abs(loss_ref_val) + 1e-8)

        assert rel_err < 1e-3, (
            f"Loss divergence: ref={loss_ref_val:.6f}, fa4={loss_fa4_val:.6f}, "
            f"rel_err={rel_err:.6e}"
        )

    def test_all_gradients_combined(self):
        """Combined dQ+dK+dV parity in a single forward/backward pass."""
        device = torch.device("cuda")

        # Reference path
        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = self._get_reference_output(q_ref, k_ref, v_ref, dense_bias)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # FA4 path
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _fa4_attention_forward(q_fa4, k_fa4, v_fa4, chunk_native, causal=True)
        loss_fa4 = out_fa4.sum()
        loss_fa4.backward()

        # Check all gradients
        for name, grad_fa4, grad_ref in [
            ("dQ", q_fa4.grad, q_ref.grad),
            ("dK", k_fa4.grad, k_ref.grad),
            ("dV", v_fa4.grad, v_ref.grad),
        ]:
            assert grad_fa4 is not None, f"{name} is None for FA4 path"
            assert grad_ref is not None, f"{name} is None for reference path"
            torch.testing.assert_close(
                grad_fa4,
                grad_ref,
                atol=BWD_ATOL,
                rtol=BWD_RTOL,
                msg=f"FA4 {name} gradient diverges from dense-bias reference",
            )

    def test_context_parallel_global_document_forward_backward_parity(
        self,
        tmp_path,
    ):
        """Two H200 ranks match the global packed-document FA4 computation."""
        if torch.cuda.device_count() < 2:
            pytest.skip("FA4 context-parallel parity requires two CUDA devices")
        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        init_method = f"file://{tmp_path / 'fa4-cp-init'}"
        processes = [
            context.Process(
                target=_fa4_context_parallel_worker,
                args=(rank, init_method, results),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        messages = _join_distributed_workers(
            processes,
            results,
            timeout_seconds=300,
        )
        assert all(process.exitcode == 0 for process in processes), messages
        assert sorted(messages) == [("ok", 0), ("ok", 1)]

    def test_sequence_parallel_tp2_production_forward_backward_isolation(
        self,
        tmp_path,
    ):
        """Real TE TP2/SP QKV and projection preserve packed-document isolation."""
        if torch.cuda.device_count() < 2:
            pytest.skip("FA4 TP2/SP parity requires two H200 devices")
        _run_fa4_tp_sp_world(
            cp_size=1,
            init_method=f"file://{tmp_path / 'fa4-tp2-sp-init'}",
        )

    def test_cartesian_tp2_cp2_production_forward_backward_isolation(
        self,
        tmp_path,
    ):
        """Real TP2/SP + CP2 zigzag composition is isolated and reference-exact."""
        if torch.cuda.device_count() < 4:
            pytest.skip("FA4 Cartesian TP2/SP+CP2 parity requires four H200 devices")
        _run_fa4_tp_sp_world(
            cp_size=2,
            init_method=f"file://{tmp_path / 'fa4-tp2-sp-cp2-init'}",
        )

    def test_document_mask_only_partial_tile_forward_backward_parity(self):
        """Exact mask handles short unaligned documents and a partial tile."""
        device = torch.device("cuda")
        sequence_length = 129
        document_ids = torch.tensor(
            [[1] * 3 + [2] * 17 + [3] * 109] * B,
            device=device,
            dtype=torch.int32,
        )
        zero_bias = torch.zeros(
            B,
            1,
            sequence_length,
            sequence_length,
            device=device,
            dtype=torch.bfloat16,
        )

        def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            torch.manual_seed(42)
            return tuple(
                torch.randn(
                    B,
                    sequence_length,
                    H,
                    D,
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                for _ in range(3)
            )

        q_ref, k_ref, v_ref = inputs()
        out_ref = _reference_attention_forward(
            q_ref,
            k_ref,
            v_ref,
            zero_bias,
            causal=True,
            document_ids=document_ids,
        )
        torch.manual_seed(73)
        output_probe = torch.randn_like(out_ref)
        (out_ref * output_probe).sum().backward()

        q_fa4, k_fa4, v_fa4 = inputs()
        out_fa4 = _production_fa4_attention_forward(
            q_fa4,
            k_fa4,
            v_fa4,
            structure_batch={"document_ids": document_ids},
            bias_state=None,
        )
        (out_fa4 * output_probe).sum().backward()

        with torch.no_grad():
            k_perturbed = k_fa4.detach().clone()
            v_perturbed = v_fa4.detach().clone()
            k_perturbed[:, :3].add_(32)
            v_perturbed[:, :3].sub_(32)
            out_perturbed = _production_fa4_attention_forward(
                q_fa4.detach(),
                k_perturbed,
                v_perturbed,
                structure_batch={"document_ids": document_ids},
                bias_state=None,
            )

        output_diff = (out_fa4.float() - out_ref.float()).abs()
        leakage_diff = (
            out_perturbed[:, 3:].float() - out_fa4[:, 3:].detach().float()
        ).abs()
        print(
            "FA4 document-mask-only output diff: "
            f"max={output_diff.max().item():.6g}, "
            f"mean={output_diff.mean().item():.6g}, "
            f"leakage_max={leakage_diff.max().item():.6g}"
        )
        torch.testing.assert_close(
            out_perturbed[:, 3:],
            out_fa4[:, 3:].detach(),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            out_fa4,
            out_ref,
            atol=3e-3,
            rtol=FWD_RTOL,
        )
        for name, grad_fa4, grad_ref in (
            ("dQ", q_fa4.grad, q_ref.grad),
            ("dK", k_fa4.grad, k_ref.grad),
            ("dV", v_fa4.grad, v_ref.grad),
        ):
            grad_diff = (grad_fa4.float() - grad_ref.float()).abs()
            print(
                f"FA4 document-mask-only {name} diff: "
                f"max={grad_diff.max().item():.6g}, "
                f"mean={grad_diff.mean().item():.6g}"
            )
            torch.testing.assert_close(
                grad_fa4,
                grad_ref,
                atol=DOCUMENT_BWD_ATOL,
                rtol=BWD_RTOL,
            )

    def test_document_mask_rectangular_unaligned_decode_forward_backward_parity(
        self,
    ):
        """Rectangular decode uses absolute query positions without leakage."""
        device = torch.device("cuda")
        seqlen_q, seqlen_k = 7, 129
        document_ids = torch.tensor(
            [[1] * 3 + [2] * 17 + [3] * 109] * B,
            device=device,
            dtype=torch.int32,
        )
        zero_bias = torch.zeros(
            B,
            1,
            seqlen_q,
            seqlen_k,
            device=device,
            dtype=torch.bfloat16,
        )

        def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            torch.manual_seed(111)
            q = torch.randn(
                B,
                seqlen_q,
                H,
                D,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            k = torch.randn(
                B,
                seqlen_k,
                H,
                D,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            v = torch.randn(
                B,
                seqlen_k,
                H,
                D,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            return q, k, v

        q_ref, k_ref, v_ref = inputs()
        out_ref = _reference_attention_forward(
            q_ref,
            k_ref,
            v_ref,
            zero_bias,
            causal=True,
            document_ids=document_ids,
        )
        torch.manual_seed(112)
        output_probe = torch.randn_like(out_ref)
        (out_ref * output_probe).sum().backward()

        q_fa4, k_fa4, v_fa4 = inputs()
        out_fa4 = _production_fa4_attention_forward(
            q_fa4,
            k_fa4,
            v_fa4,
            structure_batch={"document_ids": document_ids},
            bias_state=None,
        )
        (out_fa4 * output_probe).sum().backward()

        with torch.no_grad():
            k_perturbed = k_fa4.detach().clone()
            v_perturbed = v_fa4.detach().clone()
            k_perturbed[:, :20].add_(32)
            v_perturbed[:, :20].sub_(32)
            out_perturbed = _production_fa4_attention_forward(
                q_fa4.detach(),
                k_perturbed,
                v_perturbed,
                structure_batch={"document_ids": document_ids},
                bias_state=None,
            )

        output_diff = (out_fa4.float() - out_ref.float()).abs()
        leakage_diff = (out_perturbed.float() - out_fa4.detach().float()).abs()
        print(
            "FA4 rectangular document decode output diff: "
            f"max={output_diff.max().item():.6g}, "
            f"mean={output_diff.mean().item():.6g}, "
            f"leakage_max={leakage_diff.max().item():.6g}"
        )
        torch.testing.assert_close(
            out_perturbed,
            out_fa4.detach(),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            out_fa4,
            out_ref,
            atol=3e-3,
            rtol=FWD_RTOL,
        )
        for name, grad_fa4, grad_ref in (
            ("dQ", q_fa4.grad, q_ref.grad),
            ("dK", k_fa4.grad, k_ref.grad),
            ("dV", v_fa4.grad, v_ref.grad),
        ):
            grad_diff = (grad_fa4.float() - grad_ref.float()).abs()
            print(
                f"FA4 rectangular document decode {name} diff: "
                f"max={grad_diff.max().item():.6g}, "
                f"mean={grad_diff.mean().item():.6g}"
            )
            torch.testing.assert_close(
                grad_fa4,
                grad_ref,
                atol=DOCUMENT_BWD_ATOL,
                rtol=BWD_RTOL,
            )

    def test_graph_route_aux_multi_document_forward_backward_parity(self):
        """Native mask_mod isolates documents while score_mod keeps graph bias."""
        device = torch.device("cuda")
        document_ids = torch.tensor(
            [[1] * (S // 2) + [2] * (S // 2)] * B,
            device=device,
            dtype=torch.int32,
        )

        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = _reference_attention_forward(
            q_ref,
            k_ref,
            v_ref,
            dense_bias,
            causal=True,
            document_ids=document_ids,
        )
        torch.manual_seed(73)
        output_probe = torch.randn_like(out_ref)
        (out_ref * output_probe).sum().backward()

        structure_batch = _parity_structure_batch()
        structure_batch["document_ids"] = document_ids
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _production_fa4_attention_forward(
            q_fa4,
            k_fa4,
            v_fa4,
            structure_batch=structure_batch,
            bias_state=chunk_native,
        )
        (out_fa4 * output_probe).sum().backward()

        with torch.no_grad():
            out_causal_only = _reference_attention_forward(
                q_ref.detach(),
                k_ref.detach(),
                v_ref.detach(),
                dense_bias,
                causal=True,
            )
            out_document_only = _reference_attention_forward(
                q_ref.detach(),
                k_ref.detach(),
                v_ref.detach(),
                dense_bias,
                causal=False,
                document_ids=document_ids,
            )

        output_diff = (out_fa4.float() - out_ref.float()).abs()
        causal_only_diff = (out_fa4.float() - out_causal_only.float()).abs()
        document_only_diff = (out_fa4.float() - out_document_only.float()).abs()
        print(
            "FA4 multi-document output diff: "
            f"max={output_diff.max().item():.6g}, "
            f"mean={output_diff.mean().item():.6g}, "
            f"doc1_max={output_diff[:, : S // 2].max().item():.6g}, "
            f"doc2_max={output_diff[:, S // 2 :].max().item():.6g}, "
            f"causal_only_max={causal_only_diff.max().item():.6g}, "
            f"document_only_max={document_only_diff.max().item():.6g}"
        )

        with torch.no_grad():
            k_perturbed = k_fa4.detach().clone()
            v_perturbed = v_fa4.detach().clone()
            k_perturbed[:, : S // 2].add_(32)
            v_perturbed[:, : S // 2].sub_(32)
            out_perturbed = _production_fa4_attention_forward(
                q_fa4.detach(),
                k_perturbed,
                v_perturbed,
                structure_batch=structure_batch,
                bias_state=chunk_native,
            )
        leakage_diff = (
            out_perturbed[:, S // 2 :].float() - out_fa4[:, S // 2 :].detach().float()
        ).abs()
        print(
            "FA4 multi-document cross-document leakage probe: "
            f"max={leakage_diff.max().item():.6g}, "
            f"mean={leakage_diff.mean().item():.6g}"
        )
        torch.testing.assert_close(
            out_perturbed[:, S // 2 :],
            out_fa4[:, S // 2 :].detach(),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            out_fa4,
            out_ref,
            # The custom-mask SM90 path uses a different reduction schedule
            # than dense PyTorch attention. One near-zero BF16 element can
            # differ by one quantization step beyond the shared 2e-3 bound.
            atol=3e-3,
            rtol=FWD_RTOL,
            msg=lambda details: (
                "FA4 multi-document output diverges from dense masked reference\n"
                f"{details}"
            ),
        )
        gradient_pairs = (
            ("dQ", q_fa4.grad, q_ref.grad),
            ("dK", k_fa4.grad, k_ref.grad),
            ("dV", v_fa4.grad, v_ref.grad),
        )
        for name, grad_fa4, grad_ref in gradient_pairs:
            grad_diff = (grad_fa4.float() - grad_ref.float()).abs()
            print(
                f"FA4 multi-document {name} diff: "
                f"max={grad_diff.max().item():.6g}, "
                f"mean={grad_diff.mean().item():.6g}"
            )
        for name, grad_fa4, grad_ref in gradient_pairs:
            torch.testing.assert_close(
                grad_fa4,
                grad_ref,
                atol=DOCUMENT_BWD_ATOL,
                rtol=BWD_RTOL,
                msg=lambda details, name=name: (
                    f"FA4 multi-document {name} diverges from dense masked reference\n"
                    f"{details}"
                ),
            )


# ---------------------------------------------------------------------------
# Bias equivalence sanity check (runs on CPU, no GPU needed)
# ---------------------------------------------------------------------------


class TestBiasEquivalenceCPU:
    """Verify dense and chunk-native biases are equivalent (CPU, no FA4)."""

    def test_dense_and_chunk_native_bias_equivalent(self):
        """For every (b,q,k), chunk_native bias == dense bias exactly."""
        device = torch.device("cpu")
        sb = _parity_structure_batch()

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=B,
            seqlen_q=S,
            seqlen_k=S,
            device=device,
            dtype=torch.float32,
            call_weight=CALL_WEIGHT,
            type_weight=TYPE_WEIGHT,
            domain_weight=DOMAIN_WEIGHT,
            build_weight=BUILD_WEIGHT,
            beta=BETA,
        )

        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=B,
            seqlen_q=S,
            seqlen_k=S,
            device=device,
            dtype=torch.float32,
            beta=BETA,
            call_weight=CALL_WEIGHT,
            type_weight=TYPE_WEIGHT,
            domain_weight=DOMAIN_WEIGHT,
            build_weight=BUILD_WEIGHT,
        )

        # Reconstruct dense bias from chunk-native representation
        reconstructed = torch.zeros(B, S, S, dtype=torch.float32)
        for b in range(B):
            for q_idx in range(S):
                qc = int(chunk_native.token_to_chunk_q[b, q_idx].item())
                for k_idx in range(S):
                    kc = int(chunk_native.token_to_chunk_k[b, k_idx].item())
                    val = float(chunk_native.chunk_bias[b, qc, kc].item())
                    # Add rare edge contribution
                    lo = int(chunk_native.rare_row_offsets[b, q_idx].item())
                    hi = int(chunk_native.rare_row_offsets[b, q_idx + 1].item())
                    for i in range(lo, hi):
                        if int(chunk_native.rare_k[b, i].item()) == k_idx:
                            val += float(chunk_native.rare_w[b, i].item())
                            break
                    reconstructed[b, q_idx, k_idx] = val

        # Compare: dense_bias is [B, 1, S, S], reconstructed is [B, S, S]
        dense_squeezed = dense_bias.squeeze(1)
        torch.testing.assert_close(
            reconstructed,
            dense_squeezed,
            atol=1e-5,
            rtol=1e-5,
            msg="Chunk-native bias does not reproduce dense bias",
        )


# ---------------------------------------------------------------------------
# Modal entrypoint (run as: modal run tests/test_fa4_h200_parity.py)
# ---------------------------------------------------------------------------

# The immutable base and the overlaid checkout have separate provenance. A
# development overlay must never masquerade as an image built from its HEAD.
_GHCR_IMAGE_DIGEST = os.environ.get(
    "CPPMEGA_CANDIDATE_IMAGE_DIGEST",
    "sha256:ff03c4faff1513878bcae31437b018323a2297668eb728f9c45bb838d1275a0c",
)
_BASE_IMAGE_CPPMEGA_SHA = os.environ.get(
    "CPPMEGA_BASE_IMAGE_CPPMEGA_SHA",
    "ce8d41e99b24f805aaa5aa6bbfbe0d565ba693fd",
)
_CANDIDATE_CPPMEGA_SHA = os.environ.get("CPPMEGA_CANDIDATE_CPPMEGA_SHA", "")
_EXPECTED_H200_TEST_COUNT = 14
_EXPECTED_SP_CP_DIAGNOSTIC_COUNT = 6


def _modal_image():
    """Build the Modal image for H200 parity testing."""
    import pathlib
    from typing import Any

    import modal

    _REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
    if re.fullmatch(r"sha256:[0-9a-f]{64}", _GHCR_IMAGE_DIGEST) is None:
        raise RuntimeError("CPPMEGA_CANDIDATE_IMAGE_DIGEST must be an OCI digest")
    if re.fullmatch(r"[0-9a-f]{40}", _BASE_IMAGE_CPPMEGA_SHA) is None:
        raise RuntimeError(
            "CPPMEGA_BASE_IMAGE_CPPMEGA_SHA must be a full commit SHA"
        )
    if re.fullmatch(r"[0-9a-f]{40}", _CANDIDATE_CPPMEGA_SHA) is None:
        raise RuntimeError(
            "CPPMEGA_CANDIDATE_CPPMEGA_SHA is required and must be a full "
            "overlay checkout commit SHA"
        )
    local_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()
    if local_sha != _CANDIDATE_CPPMEGA_SHA:
        raise RuntimeError(
            "FA4 parity overlay checkout mismatch: "
            f"local={local_sha} candidate={_CANDIDATE_CPPMEGA_SHA}"
        )
    local_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()
    if local_status:
        overlay_mode = "development-dirty"
    elif _CANDIDATE_CPPMEGA_SHA == _BASE_IMAGE_CPPMEGA_SHA:
        overlay_mode = "release-clean"
    else:
        overlay_mode = "development-clean"
    GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
    GHCR_REF = f"{GHCR_REPO}@{_GHCR_IMAGE_DIGEST}"

    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env(
        {
            "CPPMEGA_MEGATRON_COMMIT": ("ba7b5ebce12af60627a80985792a1449ce45f46c"),
            "CPPMEGA_BASE_IMAGE_CPPMEGA_SHA": _BASE_IMAGE_CPPMEGA_SHA,
            "CPPMEGA_CANDIDATE_CPPMEGA_SHA": _CANDIDATE_CPPMEGA_SHA,
            "CPPMEGA_SOURCE_OVERLAY_MODE": overlay_mode,
            "CPPMEGA_SOURCE_IMAGE_REF": GHCR_REF,
            "MEGATRON_LM_REPO": "/opt/megatron-lm",
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "WANDB_MODE": "disabled",
        }
    )
    img = (
        img.add_local_dir(
            str(_REPO_ROOT / "cppmega"),
            remote_path="/opt/cppmega/cppmega",
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
        .add_local_dir(
            str(_REPO_ROOT / "data"),
            remote_path="/opt/cppmega/data",
            copy=True,
        )
        .add_local_file(
            str(_REPO_ROOT / "tests" / "test_fa4_h200_parity.py"),
            remote_path="/opt/cppmega/tests/test_fa4_h200_parity.py",
        )
        .add_local_file(
            str(_REPO_ROOT / "tests" / "test_fa4_document_isolation.py"),
            remote_path="/opt/cppmega/tests/test_fa4_document_isolation.py",
        )
        .add_local_file(
            str(_REPO_ROOT / "tests" / "test_graph_route_attention_bias_patch.py"),
            remote_path=(
                "/opt/cppmega/tests/test_graph_route_attention_bias_patch.py"
            ),
        )
        .add_local_file(
            str(_REPO_ROOT / "pyproject.toml"),
            remote_path="/opt/cppmega/pyproject.toml",
        )
    )
    return img


try:
    import modal as _modal
except ImportError:
    _modal = None

if _modal is not None:
    import json as _json
    import pathlib as _pathlib
    import subprocess as _subprocess

    _MODAL_GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:4")
    _modal_app = _modal.App("cppmega-fa4-parity")
    _results_vol = _modal.Volume.from_name(
        "cppmega-test-results", create_if_missing=True
    )

    @_modal_app.function(
        image=(
            _modal_image()
            if _modal.is_local() and os.environ.get("MODAL_IS_REMOTE") != "1"
            else None
        ),
        gpu=_MODAL_GPU_SPEC,
        timeout=1200,
        volumes={"/results": _results_vol},
    )
    def run_parity() -> dict:
        import inspect
        import xml.etree.ElementTree as ET

        def finish(result: dict) -> dict:
            _pathlib.Path("/results").mkdir(parents=True, exist_ok=True)
            _pathlib.Path("/results/fa4_parity.json").write_text(
                _json.dumps(result, indent=2)
            )
            _results_vol.commit()
            return result

        probe_code = """
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
import subprocess
from importlib import metadata
import tilelang
import torch
import transformer_engine
import tvm.ffi
from flash_attn.cute.interface import flash_attn_func
versions = {
    "flash-attn": metadata.version("flash-attn"),
    "flash-attn-4": metadata.version("flash-attn-4"),
    "nvidia-cutlass-dsl": metadata.version("nvidia-cutlass-dsl"),
    "quack-kernels": metadata.version("quack-kernels"),
    "apache-tvm-ffi": metadata.version("apache-tvm-ffi"),
    "tilelang": metadata.version("tilelang"),
    "z3-solver": metadata.version("z3-solver"),
    "transformer-engine": transformer_engine.__version__,
}
assert versions["flash-attn"] == "2.8.3", versions
assert versions["flash-attn-4"] == "4.0.0b23", versions
assert versions["nvidia-cutlass-dsl"] == "4.6.0.dev0", versions
assert versions["quack-kernels"] == "0.5.3", versions
assert versions["apache-tvm-ffi"] == "0.1.13.post5", versions
assert versions["tilelang"] == "0.1.9", versions
assert versions["z3-solver"] == "4.15.4.0", versions
assert versions["transformer-engine"].startswith("2.16"), versions
pip_check = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    check=False,
    capture_output=True,
    text=True,
)
assert pip_check.returncode == 0, pip_check.stdout + pip_check.stderr
fa2_files = {
    str(path) for path in metadata.distribution("flash-attn").files or ()
}
fa4_files = {
    str(path) for path in metadata.distribution("flash-attn-4").files or ()
}
assert not any(path.startswith("flash_attn/cute/") for path in fa2_files)
assert "flash_attn/cute/utils.py" in fa4_files
source_paths = (
    Path("/opt/cppmega/cppmega/megatron/document_isolation.py"),
    Path("/opt/cppmega/cppmega/megatron/fa4_graph_attention.py"),
    Path("/opt/cppmega/cppmega/megatron/fa4_score_mod_adapter.py"),
    Path("/opt/cppmega/cppmega/megatron/graph_route_attention_bias_patch.py"),
    Path("/opt/cppmega/cppmega/megatron/structure_dataset_patch.py"),
    Path("/opt/cppmega/tests/test_fa4_h200_parity.py"),
    Path("/opt/cppmega/tests/test_fa4_document_isolation.py"),
    Path("/opt/cppmega/tests/test_graph_route_attention_bias_patch.py"),
)
image_source = json.loads(Path("/opt/cppmega-image-source.json").read_text())
base_image_sha = os.environ["CPPMEGA_BASE_IMAGE_CPPMEGA_SHA"]
candidate_sha = os.environ["CPPMEGA_CANDIDATE_CPPMEGA_SHA"]
overlay_mode = os.environ["CPPMEGA_SOURCE_OVERLAY_MODE"]
assert re.fullmatch(r"[0-9a-f]{40}", base_image_sha), base_image_sha
assert re.fullmatch(r"[0-9a-f]{40}", candidate_sha), candidate_sha
assert overlay_mode in {
    "development-dirty",
    "development-clean",
    "release-clean",
}, overlay_mode
assert image_source["cppmega_sha"] == base_image_sha, image_source
megatron_commit = subprocess.check_output(
    ["git", "-C", "/opt/megatron-lm", "rev-parse", "HEAD"],
    text=True,
).strip()
assert megatron_commit == os.environ["CPPMEGA_MEGATRON_COMMIT"], megatron_commit
print(json.dumps({
    "versions": versions,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "source_image_ref": os.environ["CPPMEGA_SOURCE_IMAGE_REF"],
    "image_source": image_source,
    "base_image_cppmega_sha": base_image_sha,
    "source_overlay_checkout_sha": candidate_sha,
    "source_overlay_mode": overlay_mode,
    "release_bound": (
        overlay_mode == "release-clean" and candidate_sha == base_image_sha
    ),
    "pip_check": pip_check.stdout.strip(),
    "megatron_commit": megatron_commit,
    "source_files_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    },
    "tvm_ffi_runtime_version": tvm.ffi.__version__,
    "flash_attn_func_signature": str(inspect.signature(flash_attn_func)),
}))
"""
        probe = _subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if probe.returncode != 0:
            return finish(
                {
                    "returncode": 2,
                    "exact_pass": False,
                    "expected_test_count": _EXPECTED_H200_TEST_COUNT,
                    "gpu": _MODAL_GPU_SPEC,
                    "base_image_cppmega_sha": os.environ[
                        "CPPMEGA_BASE_IMAGE_CPPMEGA_SHA"
                    ],
                    "source_overlay_checkout_sha": os.environ[
                        "CPPMEGA_CANDIDATE_CPPMEGA_SHA"
                    ],
                    "source_overlay_mode": os.environ[
                        "CPPMEGA_SOURCE_OVERLAY_MODE"
                    ],
                    "probe_stdout": probe.stdout,
                    "probe_stderr_tail": "\n".join(probe.stderr.splitlines()[-80:]),
                }
            )
        stack = _json.loads(probe.stdout.splitlines()[-1])

        env = os.environ.copy()
        env["CPPMEGA_FA4_MAX_RARE_PER_ROW"] = "8"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_fa4_h200_parity.py",
            "-v",
            "-s",
            "--tb=short",
            "--disable-warnings",
            "--junitxml=/tmp/fa4-parity-junit.xml",
        ]
        try:
            proc = _subprocess.run(
                command,
                cwd="/opt/cppmega",
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=1100,
            )
            returncode = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except _subprocess.TimeoutExpired as exc:
            returncode = 124
            stdout = (exc.stdout or b"").decode(errors="replace")
            stderr = (exc.stderr or b"").decode(errors="replace")

        junit = {
            "present": False,
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        }
        junit_path = _pathlib.Path("/tmp/fa4-parity-junit.xml")
        if junit_path.is_file():
            root = ET.parse(junit_path).getroot()
            suites = [root] if root.tag.rsplit("}", 1)[-1] == "testsuite" else [
                child
                for child in root
                if child.tag.rsplit("}", 1)[-1] == "testsuite"
            ]
            junit = {
                "present": bool(suites),
                **{
                    name: sum(
                        int(suite.attrib.get(name, "0")) for suite in suites
                    )
                    for name in ("tests", "failures", "errors", "skipped")
                },
            }
        diagnostic_marker = "CPPMEGA_FA4_SP_CP_DIAGNOSTIC "
        sp_cp_diagnostics = []
        diagnostic_errors = []
        for line in stdout.splitlines():
            if diagnostic_marker not in line:
                continue
            payload = line.split(diagnostic_marker, 1)[1].strip()
            try:
                sp_cp_diagnostics.append(_json.loads(payload))
            except _json.JSONDecodeError as exc:
                diagnostic_errors.append(f"{exc}: {payload}")

        expected_topology = {
            (1, 2, 0, 0, 0),
            (1, 2, 1, 1, 0),
            (2, 4, 0, 0, 0),
            (2, 4, 1, 1, 0),
            (2, 4, 2, 0, 1),
            (2, 4, 3, 1, 1),
        }
        actual_topology = {
            (
                int(record.get("cp_size", -1)),
                int(record.get("world_size", -1)),
                int(record.get("global_rank", -1)),
                int(record.get("tp_rank", -1)),
                int(record.get("cp_rank", -1)),
            )
            for record in sp_cp_diagnostics
        }
        if actual_topology != expected_topology:
            diagnostic_errors.append(
                "unexpected TP/CP diagnostics topology: "
                f"{sorted(actual_topology)}"
            )
        for record in sp_cp_diagnostics:
            if record.get("backend_version") != "4.0.0b23":
                diagnostic_errors.append(
                    f"wrong FA4 backend record: {record.get('backend_version')}"
                )
            cp_record_size = int(record.get("cp_size", -1))
            if cp_record_size not in (1, 2):
                diagnostic_errors.append(
                    f"unexpected diagnostic CP size: {cp_record_size}"
                )
                continue
            expected_local_length = S // (2 * cp_record_size)
            expected_projected_length = S // cp_record_size
            expected_shapes = {
                "global_hidden_shape": [S, 1, H * D],
                "local_hidden_shape": [expected_local_length, 1, H * D],
                "projected_q_shape_before_cp_gather": [
                    expected_projected_length,
                    1,
                    H // 2,
                    D,
                ],
                "graph_bias_q_shape": [1, S],
                "local_output_shape": [expected_local_length, 1, H * D],
            }
            for name, expected_shape in expected_shapes.items():
                if record.get(name) != expected_shape:
                    diagnostic_errors.append(
                        f"{name}={record.get(name)} expected {expected_shape}"
                    )
            for metric in (
                "forward_cross_document_leakage_max",
                "cross_document_input_grad_max",
            ):
                if float(record.get(metric, float("nan"))) != 0:
                    diagnostic_errors.append(
                        f"{metric} is not exact zero: {record.get(metric)}"
                    )
            for metric in (
                "own_document_output_delta_max",
                "ordinary_input_grad_l1",
            ):
                if float(record.get(metric, 0)) <= 0:
                    diagnostic_errors.append(
                        f"{metric} is not positive: {record.get(metric)}"
                    )
            live_gradients = record.get("ordinary_parameter_grad_l1", {})
            for name in ("linear_qkv.weight", "linear_proj.weight"):
                if float(live_gradients.get(name, 0)) <= 0:
                    diagnostic_errors.append(
                        f"{name} gradient is not positive: "
                        f"{live_gradients.get(name)}"
                    )
        exact_pass = (
            returncode == 0
            and junit["present"]
            and junit["tests"] == _EXPECTED_H200_TEST_COUNT
            and junit["failures"] == 0
            and junit["errors"] == 0
            and junit["skipped"] == 0
            and stack["cuda_device_count"] == 4
            and len(sp_cp_diagnostics) == _EXPECTED_SP_CP_DIAGNOSTIC_COUNT
            and not diagnostic_errors
        )
        probe_mask = _make_document_causal_mask_mod(0, 1, causal=True)
        result = {
            "returncode": returncode,
            "exact_pass": exact_pass,
            "expected_test_count": _EXPECTED_H200_TEST_COUNT,
            "expected_sp_cp_diagnostic_count": _EXPECTED_SP_CP_DIAGNOSTIC_COUNT,
            "junit": junit,
            "gpu": _MODAL_GPU_SPEC,
            **stack,
            "document_mask_mod_signature": str(inspect.signature(probe_mask)),
            "pytest_command": command,
            "max_rare_per_row": int(env["CPPMEGA_FA4_MAX_RARE_PER_ROW"]),
            "sp_cp_diagnostics": sp_cp_diagnostics,
            "sp_cp_diagnostic_errors": diagnostic_errors,
            "stdout_tail": "\n".join(stdout.splitlines()[-80:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-30:]),
        }
        return finish(result)

    @_modal_app.local_entrypoint()
    def main() -> None:
        if not _CANDIDATE_CPPMEGA_SHA or (
            "CPPMEGA_CANDIDATE_IMAGE_DIGEST" not in os.environ
        ):
            raise RuntimeError(
                "release evidence requires CPPMEGA_CANDIDATE_CPPMEGA_SHA "
                "and CPPMEGA_CANDIDATE_IMAGE_DIGEST"
            )
        if _CANDIDATE_CPPMEGA_SHA != _BASE_IMAGE_CPPMEGA_SHA:
            raise RuntimeError(
                "release evidence requires the immutable image source and "
                "clean overlay checkout to name the same cppmega commit"
            )
        repository_root = _pathlib.Path(__file__).resolve().parents[1]
        local_status = _subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
            text=True,
        ).strip()
        if local_status:
            raise RuntimeError(
                "release evidence requires a clean overlay checkout; "
                f"dirty paths:\n{local_status}"
            )
        result = run_parity.remote()
        print(_json.dumps(result, indent=2))
        if not result["exact_pass"]:
            raise SystemExit(1)
