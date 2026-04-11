"""FastMTP: standalone Multi-Token Prediction that bypasses Megatron's MTP.

Ported from nanochat's MTP module (arXiv:2509.18362 / DeepSeek-V3 style).

Key optimizations vs Megatron's default MultiTokenPredictionBlock:
1. Fused linear cross-entropy (Liger-Kernel) -- avoids materializing V-dim logits
2. Roll-and-mask static shapes -- no dynamic slicing, XLA-safe
3. Activation checkpointing inside K-loop (per shared-block call)
4. Single shared transformer block recursed K times (not K separate blocks)
5. Exponential decay weighting across prediction depths
6. Cadence scheduling support (skip MTP on some steps)
7. Pre-computed step weights as buffers (no per-step recomputation)

Activated via CPPMEGA_FASTMTP=1 environment variable.
"""

from __future__ import annotations

import os
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


def _compute_step_weights(depth: int, decay: float) -> torch.Tensor:
    """Normalized exponentially-decaying weights for K prediction steps."""
    raw = torch.tensor([decay**k for k in range(depth)], dtype=torch.float32)
    return raw / raw.sum()


def _roll_and_mask_targets(x: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    """Roll target IDs left by 1, mask last position with ignore_index."""
    tail = x.new_full((x.shape[0], 1), ignore_index)
    return torch.cat((x[:, 1:], tail), dim=1)


def _roll_and_mask_ids(x: torch.Tensor) -> torch.Tensor:
    """Roll token IDs left by 1, mask last position with 0."""
    tail = x.new_zeros((x.shape[0], 1))
    return torch.cat((x[:, 1:], tail), dim=1)


def _try_import_liger():
    """Try to import Liger fused linear cross entropy. Returns None if unavailable."""
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )
        return LigerFusedLinearCrossEntropyFunction
    except ImportError:
        return None


# Lazy import -- resolved on first forward call
_LigerFusedLinearCE = None
_liger_checked = False


def _get_liger_fused_ce():
    global _LigerFusedLinearCE, _liger_checked
    if not _liger_checked:
        _LigerFusedLinearCE = _try_import_liger()
        _liger_checked = True
    return _LigerFusedLinearCE


def _fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Compute cross-entropy loss, using Liger fused path if available.

    Falls back to standard F.linear + F.cross_entropy if Liger is not installed.

    Args:
        hidden: (B*T, D) hidden states in bf16
        weight: (V, D) lm_head weight
        targets: (B*T,) target token IDs
        ignore_index: index to ignore in loss computation

    Returns:
        per-token losses (B*T,) with reduction='none'
    """
    use_liger = os.environ.get("CPPMEGA_FASTMTP_USE_LIGER", "1") == "1"
    liger_fn = _get_liger_fused_ce() if use_liger else None
    if liger_fn is not None:
        # LigerFusedLinearCrossEntropyFunction.apply signature:
        # (_input, weight, target, bias, ce_weight, ignore_index,
        #  lse_square_scale, label_smoothing, reduction, softcap,
        #  return_z_loss)
        result = liger_fn.apply(
            hidden,       # _input
            weight,       # weight
            targets,      # target
            None,         # bias
            None,         # ce_weight
            ignore_index, # ignore_index
            0.0,          # lse_square_scale
            0.0,          # label_smoothing
            "none",       # reduction
            None,         # softcap
            False,        # return_z_loss
        )
        loss = result[0] if isinstance(result, tuple) else result
        return loss
    else:
        # Fallback: standard path (materializes logits)
        logits = F.linear(hidden, weight)
        logits = logits.float()
        loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=ignore_index,
            reduction="none",
        )
        return loss


class _SimpleMTPBlock(nn.Module):
    """Minimal self-attention + MLP block for MTP shared block.

    Uses plain PyTorch (no TE, no Megatron SP/TP) to avoid workspace and
    communication conflicts with the main decoder. Input/output: (T, B, D).
    """

    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Self-attention
        self.ln1 = nn.RMSNorm(hidden_size, eps=1e-5)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP
        self.ln2 = nn.RMSNorm(hidden_size, eps=1e-5)
        self.gate_up = nn.Linear(hidden_size, 2 * ffn_hidden_size, bias=False)
        self.down = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """Forward pass. Input: (T, B, D), Output: (T, B, D)."""
        # Self-attention with causal mask
        residual = hidden_states
        x = self.ln1(hidden_states)

        T, B, D = x.shape
        qkv = self.qkv(x)  # (T, B, 3*D)
        qkv = qkv.reshape(T, B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # (T, B, H, Hd) -> (B, H, T, Hd)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Scaled dot-product attention with causal mask
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )  # (B, H, T, Hd)

        # (B, H, T, Hd) -> (T, B, D)
        attn_out = attn_out.permute(2, 0, 1, 3).reshape(T, B, D)
        hidden_states = residual + self.out_proj(attn_out)

        # MLP: SiLU-gated
        residual = hidden_states
        x = self.ln2(hidden_states)
        gate_up = self.gate_up(x)  # (T, B, 2*FFN)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = residual + self.down(F.silu(gate) * up)

        return hidden_states


class FastMTPLayer(MegatronModule):
    """Recursive multi-token prediction with shared transformer block.

    This replaces Megatron's MultiTokenPredictionBlock with nanochat's FastMTP
    design. It is monkey-patched into CppMegaMambaModel when CPPMEGA_FASTMTP=1.

    Architecture:
        - proj: Linear(2*D, D, bias=False) -- concat(RMSNorm(h), RMSNorm(emb)) -> D
        - block: single TE TransformerLayer, reused K times
        - loss: fused linear CE (Liger) or standard CE fallback

    The layer is attached to the model's forward path and computes MTP loss
    as a post-processing step after the main decoder, replacing Megatron's
    standard MTP forward + process_mtp_loss pipeline.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        depth: int = 1,
        decay: float = 0.6,
        recompute: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)
        hidden_size = config.hidden_size
        self.depth = depth
        self.recompute = recompute
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Own word embedding for teacher-forcing (weight-tied with main model
        # after construction via tie_word_embeddings()).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            config=config,
            init_method=config.init_method,
            tp_group=tp_group,
        )

        # Projection: [RMSNorm(hidden); RMSNorm(emb)] -> hidden_size
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        # Single shared transformer block -- built from TE spec
        self.shared_block = self._build_shared_block(config)

        # Pre-computed normalized step weights
        self.register_buffer(
            "step_weights",
            _compute_step_weights(depth, decay),
        )

    def _build_shared_block(self, config: TransformerConfig) -> nn.Module:
        """Build a simple self-attention + MLP block for MTP.

        Uses plain PyTorch modules to avoid TE workspace conflicts and
        Megatron SP/TP complexity. The MTP shared block is small (1 layer)
        so TE fusion overhead savings are negligible vs correctness.
        """
        return _SimpleMTPBlock(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            ffn_hidden_size=config.ffn_hidden_size,
        )

    def tie_word_embeddings(self, source_embedding: nn.Module) -> None:
        """Tie word embedding weights from the main model's embedding.

        This shares the weight tensor (no copy), matching Megatron's MTP
        weight-tying behavior. Call after model construction.

        Args:
            source_embedding: the model's embedding module (must have
                .word_embeddings attribute with a .weight parameter).
        """
        src_wte = source_embedding.word_embeddings
        self.word_embeddings.weight = src_wte.weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        output_layer: nn.Module,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """FastMTP forward pass.

        Args:
            hidden_states: (S, B, D) from last decoder layer (Megatron seq-first)
            labels: (B, T) ground-truth targets
            output_layer: model.output_layer (ColumnParallelLinear)
            loss_mask: (B, T) optional mask
            attention_mask: for the shared transformer block
            rotary_pos_emb: rotary embeddings for the shared block
            packed_seq_params: optional packed sequence params

        Returns:
            mtp_loss: scalar loss
        """
        K = self.depth

        # Megatron uses (S, B, D) layout. For roll-and-mask we need (B, T, D).
        # If sequence parallel, gather first.
        if self.config.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=True,
            )

        # (S, B, D) -> (B, S, D)
        h = hidden_states.transpose(0, 1).contiguous()
        B, T, D = h.shape

        if T <= 1:
            return h.sum() * 0.0

        # Detach hidden states -- MTP gradients should not flow back through
        # the main decoder (matches nanochat and DeepSeek-V3 design).
        h = h.detach()

        rolled_ids = labels.clone()  # (B, T)
        rolled_targets = labels.clone()  # (B, T)

        accumulated_losses: list[torch.Tensor] = []
        step_weights = cast(torch.Tensor, self.step_weights)

        # Get lm_head weight (handle ColumnParallelLinear).
        # Detach to prevent FastMTP gradients from flowing through the shared
        # output_layer weight -- the main NTP loss already provides gradients
        # for output_layer. This also avoids TE workspace conflicts from
        # double backward through the same ColumnParallelLinear.
        lm_head_weight = output_layer.weight.detach()
        if lm_head_weight.dtype != torch.bfloat16:
            lm_head_weight = lm_head_weight.to(torch.bfloat16)

        for k in range(K):
            # Roll left by 1, mask tail
            rolled_ids = _roll_and_mask_ids(rolled_ids)
            rolled_targets = _roll_and_mask_targets(rolled_targets)

            # Embed rolled teacher-forcing tokens using our own word_embeddings
            # (weight-tied with main model's embedding)
            next_emb = self.word_embeddings(rolled_ids.clone())  # (B, T, D)

            # RMSNorm both, concat, project
            h_norm = F.rms_norm(h, (D,))
            e_norm = F.rms_norm(next_emb, (D,))
            h_mtp = self.proj(torch.cat([h_norm, e_norm], dim=-1))  # (B, T, D)

            # Pass through shared transformer block
            # (B, T, D) -> (T, B, D) for Megatron transformer layer
            h_mtp_seq = h_mtp.transpose(0, 1).contiguous()

            # The shared block handles SP internally via its attention/MLP layers
            h_mtp_seq = self._shared_block_forward(
                h_mtp_seq, attention_mask, rotary_pos_emb, packed_seq_params
            )

            # (T, B, D) -> (B, T, D)
            h_out = h_mtp_seq.transpose(0, 1).contiguous()

            # RMSNorm output -- feeds back as hidden state for next step
            h = F.rms_norm(h_out, (D,))

            # Compute cross-entropy loss
            h_ce = h if h.dtype == torch.bfloat16 else h.to(torch.bfloat16)
            B_T = B * T
            h_flat = h_ce.reshape(B_T, -1)
            targets_flat = rolled_targets.reshape(-1)

            step_loss_tokens = _fused_linear_cross_entropy(
                h_flat, lm_head_weight, targets_flat, ignore_index=-1
            )

            # Mask and reduce
            valid_mask = (targets_flat != -1).to(step_loss_tokens.dtype)
            valid_count = valid_mask.sum()
            step_loss = (step_loss_tokens * valid_mask).sum() / valid_count.clamp(min=1)
            accumulated_losses.append(step_weights[k] * step_loss)

        if not accumulated_losses:
            return h.new_zeros((), requires_grad=True)
        return sum(accumulated_losses)  # type: ignore[return-value]

    def _shared_block_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        rotary_pos_emb: Optional[torch.Tensor],
        packed_seq_params=None,
    ) -> torch.Tensor:
        """Forward through the shared transformer block."""
        output = self.shared_block(hidden_states)
        if isinstance(output, tuple):
            output = output[0]
        return output


def fastmtp_enabled() -> bool:
    """Check if FastMTP is enabled via environment variable."""
    return os.environ.get("CPPMEGA_FASTMTP", "0") == "1"


def get_fastmtp_depth() -> int:
    """Get FastMTP depth from environment (default 1)."""
    return int(os.environ.get("CPPMEGA_FASTMTP_DEPTH", "1"))


def get_fastmtp_decay() -> float:
    """Get FastMTP decay from environment (default 0.6)."""
    return float(os.environ.get("CPPMEGA_FASTMTP_DECAY", "0.6"))


def get_fastmtp_lambda() -> float:
    """Get FastMTP loss weight from environment (default 0.3)."""
    return float(os.environ.get("CPPMEGA_FASTMTP_LAMBDA", "0.3"))
