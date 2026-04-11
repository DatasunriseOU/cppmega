"""Replace Megatron MTP's output_layer + CE with Liger fused_linear_cross_entropy.

Env: CPPMEGA_MTP_LIGER_CE=1 enables this optimization.

When enabled, monkey-patches ``process_mtp_loss`` so that each MTP depth
replaces the two-step:

    logits = output_layer(hidden_states)          # materializes [S*B, V] fp32
    loss   = compute_language_model_loss(labels, logits)

with a single fused Triton kernel that chunks the matmul and computes
cross-entropy without ever materializing the full logits tensor.

Tradeoff (bench3 H200, B=4, S=4096, H=3584, V=65536, 4 MTP depths):

    Metric          Original    Liger       Delta
    ------------------------------------------------
    Time (4 depths) 178.8 ms    483.2 ms    +304 ms (SLOWER)
    Peak memory     27.36 GB    5.49 GB     -21.88 GB (-80%)

Liger chunks the output GEMM into ~16 small pieces to avoid materializing
the full logits tensor.  This saves ~80% activation memory but the small
GEMMs are much less efficient on H200 tensor cores.

Use cases:
  - RECOMMENDED on memory-constrained GPUs (A100 40/80GB) where the logits
    tensor causes OOM or forces smaller batch sizes.
  - NOT recommended on H200 (141 GB HBM) where memory is abundant and GEMM
    throughput is the binding constraint.

Only valid when TP = 1.  Falls back to original path when
``scale_logits_fn`` is not None (MuP).
"""
from __future__ import annotations

import os
import sys


def patch_mtp_loss_with_liger():
    """Monkey-patch process_mtp_loss to use Liger fused linear CE.

    Safe to call unconditionally — does nothing unless
    ``CPPMEGA_MTP_LIGER_CE=1`` is set.
    """
    if os.environ.get("CPPMEGA_MTP_LIGER_CE", "0") != "1":
        return

    try:
        import torch
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )
        from megatron.core.transformer import multi_token_prediction as mtp_mod

        _orig_process_mtp_loss = mtp_mod.process_mtp_loss

        # Keep references to helpers we need inside the replacement.
        _roll_tensor = mtp_mod.roll_tensor
        _MTPLossLoggingHelper = mtp_mod.MTPLossLoggingHelper
        _MTPLossAutoScaler = mtp_mod.MTPLossAutoScaler

        def _liger_process_mtp_loss(
            hidden_states,
            labels,
            loss_mask,
            output_layer,
            output_weight,
            runtime_gather_output,
            is_training,
            compute_language_model_loss,
            config,
            cp_group=None,
            packed_seq_params=None,
            scale_logits_fn=None,
        ):
            """Drop-in replacement for process_mtp_loss using Liger fused CE.

            Falls back to the original implementation when scale_logits_fn
            is set (MuP) since Liger cannot fuse the logit scaling.
            """
            # ---- Fall back if scale_logits_fn is used (MuP) ----
            if scale_logits_fn is not None:
                return _orig_process_mtp_loss(
                    hidden_states,
                    labels,
                    loss_mask,
                    output_layer,
                    output_weight,
                    runtime_gather_output,
                    is_training,
                    compute_language_model_loss,
                    config,
                    cp_group=cp_group,
                    packed_seq_params=packed_seq_params,
                    scale_logits_fn=scale_logits_fn,
                )

            # ---- Chunk hidden states by MTP depth ----
            hidden_states_list = torch.chunk(
                hidden_states, 1 + config.mtp_num_layers, dim=0
            )
            hidden_states = hidden_states_list[0]

            if labels is None:
                return hidden_states

            mtp_labels = labels.clone()
            if loss_mask is None:
                loss_mask = torch.ones_like(mtp_labels)

            original_num_tokens = loss_mask.sum()

            # ---- Resolve the weight tensor once ----
            # output_layer is a ColumnParallelLinear.  When output_weight is
            # supplied (tied embeddings) we use that; otherwise grab the
            # module's own weight.  At TP=1 this is the full [V, H] matrix.
            if output_weight is not None:
                w = output_weight
            elif hasattr(output_layer, "weight") and output_layer.weight is not None:
                w = output_layer.weight
            else:
                raise RuntimeError(
                    "[mtp_liger_ce] Cannot resolve output_layer weight for Liger CE"
                )

            from megatron.core import parallel_state

            for mtp_layer_number in range(config.mtp_num_layers):
                # hidden for this depth: [s, b, h]
                h = hidden_states_list[mtp_layer_number + 1]
                s, b, hdim = h.shape

                # Reshape to [s*b, h] for Liger (contiguous row-major)
                h_2d = h.reshape(s * b, hdim)

                # Roll labels and loss_mask (same as original)
                mtp_labels, _ = _roll_tensor(
                    mtp_labels,
                    shifts=-1,
                    dims=-1,
                    cp_group=cp_group,
                    packed_seq_params=packed_seq_params,
                )
                loss_mask, num_tokens = _roll_tensor(
                    loss_mask,
                    shifts=-1,
                    dims=-1,
                    cp_group=cp_group,
                    packed_seq_params=packed_seq_params,
                )

                # Labels are [b, s] — transpose to [s, b] then flatten to
                # match the h_2d row ordering.
                target_1d = mtp_labels.transpose(0, 1).reshape(-1)

                # ---- Liger fused linear cross-entropy ----
                # Returns (loss_1d, z_loss, token_accuracy)
                # reduction='none' gives per-token loss [s*b]
                liger_loss_1d, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
                    h_2d,       # [s*b, h] bf16
                    w,          # [V, h] bf16
                    target_1d,  # [s*b] int64
                    None,       # bias
                    None,       # ce_weight
                    -100,       # ignore_index
                    0.0,        # lse_square_scale
                    0.0,        # label_smoothing
                    "none",     # reduction — per-token
                    None,       # softcap
                    False,      # return_z_loss
                )

                # Reshape back to [b, s] (transpose from [s, b])
                mtp_loss = liger_loss_1d.reshape(s, b).transpose(0, 1).contiguous()

                # Apply loss mask (same as original)
                mtp_loss = loss_mask * mtp_loss

                if is_training:
                    mtp_loss_for_log = (
                        torch.sum(mtp_loss) / num_tokens
                        if num_tokens > 0
                        else mtp_loss.new_tensor(0.0)
                    )
                    _MTPLossLoggingHelper.save_loss_to_tracker(
                        mtp_loss_for_log,
                        mtp_layer_number,
                        config.mtp_num_layers,
                        avg_group=parallel_state.get_data_parallel_group(
                            with_context_parallel=True
                        ),
                    )

                mtp_loss_scale = (
                    config.mtp_loss_scaling_factor / config.mtp_num_layers
                )
                if config.calculate_per_token_loss:
                    num_tokens_safe = torch.clamp(num_tokens, min=1)
                    mtp_loss_normalized = (
                        mtp_loss_scale
                        * mtp_loss
                        * (original_num_tokens / num_tokens_safe)
                    )
                    hidden_states = _MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_normalized
                    )
                else:
                    safe_num_tokens = num_tokens.clamp(min=1)
                    hidden_states = _MTPLossAutoScaler.apply(
                        hidden_states,
                        mtp_loss_scale * mtp_loss / safe_num_tokens,
                    )

            return hidden_states

        # ---- Install the patch ----
        mtp_mod.process_mtp_loss = _liger_process_mtp_loss
        print("[cppmega] MTP Liger fused CE patch installed")

    except Exception as exc:
        print(f"[cppmega] MTP Liger CE patch FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
