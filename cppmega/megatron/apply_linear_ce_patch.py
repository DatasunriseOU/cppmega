"""Swap MambaModel.output_layer to LinearCrossEntropyModule to fuse GEMM + CE.

Env gate: ``CPPMEGA_MAIN_HEAD_LINEAR_CE=1``

Upstream Megatron GPT model uses ``LinearCrossEntropyModule`` (subclass of
``ColumnParallelLinear``) which — when ``config.cross_entropy_loss_fusion=True``
and ``config.cross_entropy_fusion_impl="linear"`` — fuses the output matmul
with cross-entropy, never materializing the full ``[seq, batch, vocab]``
logits tensor. On NAM56R MBS=12 FP8 tensorwise this saves ~12 GiB/rank and
unblocks the CE-head-memory OOM documented in
``reference_main_head_liger_ce_gap.md``.

**Upstream bug**: GPT model uses ``LinearCrossEntropyModule`` unconditionally
(``gpt_model.py:251``), but Mamba model still uses plain ``ColumnParallelLinear``
(``mamba_model.py:264``). The ``self.fuse_linear_cross_entropy`` branch at
``mamba_model.py:482-486`` calls ``self.output_layer(output_cross_entropy_loss=True, ...)``
which unconditionally fails with TypeError on the plain class.

**Fix**: after model init, if the fusion flag is set, reassign
``self.output_layer.__class__`` to ``LinearCrossEntropyModule``. Safe because
the two classes share all attributes and init args — only the ``forward``
method differs.
"""
from __future__ import annotations

import os
import sys


def _patch_linear_ce_route_to_liger() -> None:
    """Reroute LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss
    to Liger kernel on Hopper (cc 9.x) where Megatron's native kernel asserts.

    Megatron's ``fused_linear_cross_entropy.py:40`` hard-asserts ``cc[0] == 10``
    (Blackwell only). On H200 we need Liger's H100/H200-compatible fused linear
    cross-entropy kernel instead. This patch overrides the method to call Liger
    directly when we're on Hopper.
    """
    import torch
    from megatron.core.transformer.linear_cross_entropy import (
        LinearCrossEntropyModule,
    )
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )
    except ImportError:
        print(
            "[cppmega] liger_kernel not installed — cannot route linear CE to Liger",
            file=sys.stderr,
        )
        return

    if not torch.cuda.is_available():
        return
    dev = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(dev)
    if cc[0] >= 10:
        return  # Blackwell+ — use native Megatron path.

    _orig = LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss

    def _liger_compute_linear_and_cross_entropy_loss(
        self,
        hidden,
        weight,
        labels=None,
        reduction="none",
        ignore_index=-100,
    ):
        # Shapes entering: hidden [s, b, h], weight [V, h], labels [b, s].
        s, b, hdim = hidden.shape
        # Transpose labels to [s, b] → flatten to [s*b] matching hidden row order.
        labels_sb = labels.transpose(0, 1).contiguous().reshape(-1)
        # Liger requires contiguous inputs — reshape() of a non-contiguous tensor
        # can return a view that the kernel reads garbage from.
        hidden_2d = hidden.contiguous().reshape(s * b, hdim)
        weight = weight.contiguous()

        # Liger fused linear CE (per-token).
        liger_loss_1d, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
            hidden_2d,
            weight,
            labels_sb,
            None,         # bias
            None,         # ce_weight
            ignore_index,
            0.0,          # lse_square_scale
            0.0,          # label_smoothing
            "none",       # reduction
            None,         # softcap
            False,        # return_z_loss
        )

        # Liger returned [s*b]; reshape to [s, b] then transpose to [b, s].
        if reduction == "none":
            return liger_loss_1d.reshape(s, b).transpose(0, 1).contiguous()
        elif reduction == "sum":
            return liger_loss_1d.sum()
        elif reduction == "mean":
            return liger_loss_1d.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

    LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss = (
        _liger_compute_linear_and_cross_entropy_loss
    )
    print(
        f"[cppmega] LinearCrossEntropyModule routed to Liger kernel (cc={cc[0]}.{cc[1]}, pre-Blackwell)"
    )


def patch_mamba_output_layer_with_linear_ce() -> None:
    """Monkey-patch MambaModel.__init__ to swap output_layer class.

    Safe to call unconditionally — does nothing unless
    ``CPPMEGA_MAIN_HEAD_LINEAR_CE=1`` is set.
    """
    if os.environ.get("CPPMEGA_MAIN_HEAD_LINEAR_CE", "0") != "1":
        return

    try:
        from megatron.core import tensor_parallel
        from megatron.core.models.mamba.mamba_model import MambaModel
        from megatron.core.transformer.linear_cross_entropy import (
            LinearCrossEntropyModule,
        )

        # Re-route fused kernel to Liger on Hopper so the native path doesn't
        # hit ``ValueError: Unsupported architecture: 9``.
        _patch_linear_ce_route_to_liger()

        _orig_init = MambaModel.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)

            # Only swap on post_process rank where output_layer exists.
            if not hasattr(self, "output_layer"):
                return
            if not isinstance(self.output_layer, tensor_parallel.ColumnParallelLinear):
                return
            if isinstance(self.output_layer, LinearCrossEntropyModule):
                return  # already correct

            cfg = self.config
            want_fusion = (
                getattr(cfg, "cross_entropy_loss_fusion", False)
                and getattr(cfg, "cross_entropy_fusion_impl", None) == "linear"
            )
            if not want_fusion:
                return

            # LinearCrossEntropyModule is a pure subclass of ColumnParallelLinear
            # that adds an `output_cross_entropy_loss` kwarg to forward(). All
            # attributes (weight, bias, config, tp_group, etc.) are compatible,
            # so __class__ reassignment is safe.
            self.output_layer.__class__ = LinearCrossEntropyModule

            # Ensure fuse_linear_cross_entropy flag reflects the swap so the
            # mamba_model.forward() branch at line 482-486 uses the fused path.
            self.fuse_linear_cross_entropy = True

            print(
                "[cppmega] main-head linear_cross_entropy fusion enabled "
                "(MambaModel.output_layer -> LinearCrossEntropyModule)"
            )

        MambaModel.__init__ = _patched_init

    except Exception as exc:  # pragma: no cover
        print(
            f"[cppmega] main-head linear_cross_entropy patch FAILED: {exc}",
            file=sys.stderr,
        )
        import traceback

        traceback.print_exc(file=sys.stderr)
