"""Route MTP head cross-entropy to native Megatron LinearCrossEntropyModule.

Env gate: ``CPPMEGA_MTP_NATIVE_HOPPER_CE=1``

Post-PR #3345 the upstream ``process_mtp_loss`` in
``megatron/core/transformer/multi_token_prediction.py`` **already has** a
``fuse_linear_cross_entropy`` branch (lines 665-679) that routes through
``output_layer(..., output_cross_entropy_loss=True, labels=mtp_labels, ...)``
— the exact same API path that PR #3345's native Hopper kernel supports for
the main head. That branch activates when both of these are set on the
:class:`TransformerConfig`:

    * ``config.cross_entropy_loss_fusion = True``
    * ``config.cross_entropy_fusion_impl = "linear"``

Both are flipped by ``--cross-entropy-loss-fusion --cross-entropy-fusion-impl
linear`` on the Megatron CLI.

The ``output_layer.__class__`` swap is performed by
``cppmega.megatron.apply_linear_ce_patch`` on :class:`MambaModel` when
``CPPMEGA_MAIN_HEAD_LINEAR_CE=1``. ``process_mtp_loss`` sees the same
``self.output_layer`` reference so the class swap propagates transparently.

HOWEVER: the upstream fuse_linear_cross_entropy branch in
``process_mtp_loss`` calls ``output_layer(..., reduction="none")`` (the
default), then multiplies elementwise by ``loss_mask``, then sums. That
routes ``LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss``
through the ``reduction="none"`` path at lines 61-74, which does a
``labels.transpose(0, 1).contiguous()`` before the kernel and the inverse
``loss.view_as(labels).transpose(0, 1).contiguous()`` after. Empirically this
transpose round-trip on a shared ``output_weight`` called multiple times
(main head + N MTP depths) corrupts autograd graph and yields NaN
``grad_norm`` after the first iter.

The fix: **monkey-patch ``process_mtp_loss`` to call ``output_layer`` with
``reduction="sum"`` and ``ignore_index=-100``**, folding the loss_mask into
the labels via ``torch.where(loss_mask.bool(), mtp_labels, -100)``. This
sidesteps the transpose round-trip because the ``reduction != "none"`` path
returns a scalar and never reshapes.

So this module is BOTH a verification shim AND a monkey-patch:

  * Preflight: refuse to install if ``CPPMEGA_MTP_LIGER_CE=1`` (Liger
    replacement would override ``process_mtp_loss`` and undo our patch).
  * Preflight: refuse to install if ``CPPMEGA_MAIN_HEAD_LINEAR_CE=0``
    (no class swap → ColumnParallelLinear raises on kwargs).
  * Preflight: assert ``--cross-entropy-loss-fusion --cross-entropy-fusion-
    impl linear`` on the CLI via a post-init hook on :class:`MambaModel`.
  * Monkey-patch: replace ``process_mtp_loss`` with a shim that routes
    through ``output_layer(..., reduction="sum", ignore_index=-100,
    labels=masked_labels)`` then divides by ``safe_num_tokens``.

Memory cost at NAM56R MBS=10:
    * Vanilla ``vocab_parallel_cross_entropy`` on MTP materializes
      ~6 GiB of [s*b, V] logits per depth × 2 depths = 12 GiB saved when
      routed through the native kernel.
    * ``LigerFusedLinearCrossEntropyFunction`` ``reduction="none"`` bwd
      grad corruption (issue #968) is sidestepped entirely.
    * PR #3345 transpose round-trip NaN bug on shared weight multi-call
      is sidestepped by using ``reduction="sum"``.
"""
from __future__ import annotations

import os
import sys


def patch_mtp_native_hopper_ce() -> None:
    """Verify the native MTP linear-CE fusion path is wired and log the route.

    Safe to call unconditionally — does nothing unless
    ``CPPMEGA_MTP_NATIVE_HOPPER_CE=1`` is set.
    """
    if os.environ.get("CPPMEGA_MTP_NATIVE_HOPPER_CE", "0") != "1":
        return

    # Hard error if user tries to stack with MTP Liger: the Liger patch
    # replaces process_mtp_loss entirely, which would bypass the native
    # fusion and reintroduce the bwd bug we're trying to avoid.
    if os.environ.get("CPPMEGA_MTP_LIGER_CE", "0") == "1":
        raise RuntimeError(
            "[cppmega] CPPMEGA_MTP_NATIVE_HOPPER_CE=1 is mutually exclusive "
            "with CPPMEGA_MTP_LIGER_CE=1. The Liger MTP patch monkey-patches "
            "process_mtp_loss and would defeat the native fusion. Set "
            "CPPMEGA_MTP_LIGER_CE=0 to use the native path."
        )

    # Class swap on output_layer is what makes the native fusion branch in
    # process_mtp_loss work. Without it, output_layer is plain
    # ColumnParallelLinear which raises TypeError on
    # output_cross_entropy_loss=True / labels= kwargs.
    if os.environ.get("CPPMEGA_MAIN_HEAD_LINEAR_CE", "0") != "1":
        raise RuntimeError(
            "[cppmega] CPPMEGA_MTP_NATIVE_HOPPER_CE=1 requires "
            "CPPMEGA_MAIN_HEAD_LINEAR_CE=1 (the class swap on "
            "MambaModel.output_layer -> LinearCrossEntropyModule is shared "
            "between main head and MTP head)."
        )

    try:
        from megatron.core.models.mamba.mamba_model import MambaModel
        from megatron.core.transformer.linear_cross_entropy import (
            LinearCrossEntropyModule,
        )
    except Exception as exc:  # pragma: no cover
        print(
            f"[cppmega] MTP native Hopper CE patch aborted — cannot import "
            f"Megatron ({exc}); check that PR #3345 is in the tree.",
            file=sys.stderr,
        )
        return

    # Wrap MambaModel.__init__ AFTER apply_linear_ce_patch ran so we see
    # the already-swapped output_layer and assert the config is coherent.
    _prev_init = MambaModel.__init__

    def _patched_init(self, *args, **kwargs):
        _prev_init(self, *args, **kwargs)

        if not hasattr(self, "output_layer"):
            return  # non-post-process rank, no output layer

        cfg = self.config
        fuse_ok = (
            getattr(cfg, "cross_entropy_loss_fusion", False)
            and getattr(cfg, "cross_entropy_fusion_impl", None) == "linear"
        )
        if not fuse_ok:
            raise RuntimeError(
                "[cppmega] CPPMEGA_MTP_NATIVE_HOPPER_CE=1 requires "
                "--cross-entropy-loss-fusion --cross-entropy-fusion-impl "
                "linear on the Megatron CLI (config.cross_entropy_loss_fusion"
                f"={getattr(cfg, 'cross_entropy_loss_fusion', False)}, "
                "cross_entropy_fusion_impl="
                f"{getattr(cfg, 'cross_entropy_fusion_impl', None)!r})."
            )

        if not isinstance(self.output_layer, LinearCrossEntropyModule):
            raise RuntimeError(
                "[cppmega] MTP native Hopper CE expected output_layer to be "
                "LinearCrossEntropyModule (installed by apply_linear_ce_patch)"
                f" but found {type(self.output_layer).__name__}. This most "
                "likely means apply_linear_ce_patch ran BEFORE MambaModel was "
                "constructed — check shim ordering in pretrain_mamba.py."
            )

        mtp_layers = getattr(cfg, "mtp_num_layers", None)
        if mtp_layers and getattr(self, "mtp_process", False):
            print(
                f"[cppmega] MTP native Hopper CE enabled — "
                f"process_mtp_loss fuse_linear_cross_entropy branch will "
                f"route all {mtp_layers} MTP depths through the same "
                f"LinearCrossEntropyModule as the main head "
                f"(no logits materialization, PR #3345 Hopper kernel on cc 9.x)."
            )
        else:
            print(
                "[cppmega] CPPMEGA_MTP_NATIVE_HOPPER_CE=1 set but "
                f"mtp_num_layers={mtp_layers} / mtp_process="
                f"{getattr(self, 'mtp_process', False)}; no MTP head on "
                "this rank, gate is a no-op."
            )

    MambaModel.__init__ = _patched_init

    # ------------------------------------------------------------------
    # Monkey-patch process_mtp_loss to avoid the PR #3345 reduction="none"
    # + transpose round-trip NaN bug on shared-weight multi-call.
    # ------------------------------------------------------------------
    _install_process_mtp_loss_patch()


def _install_process_mtp_loss_patch() -> None:
    """Replace upstream ``process_mtp_loss`` with a reduction="sum" variant.

    The wrapper mirrors the upstream logic byte-for-byte in the non-fusion
    branch but in the ``fuse_linear_cross_entropy`` branch it:

      1. Builds ``masked_labels = where(loss_mask.bool(), mtp_labels, -100)``
         so ignored positions become ``ignore_index`` instead of being
         zero-weighted after the kernel.
      2. Calls ``output_layer(..., reduction="sum", ignore_index=-100,
         labels=masked_labels)`` — this hits the
         ``reduction != "none"`` branch in ``LinearCrossEntropyModule`` that
         returns a scalar and skips the transpose round-trip at
         ``linear_cross_entropy.py:61, 74``.
      3. Divides by ``safe_num_tokens`` and forwards through
         :class:`MTPLossAutoScaler` exactly like upstream.
    """
    import torch

    from megatron.core import parallel_state
    from megatron.core.transformer import multi_token_prediction as _mtp_mod
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossAutoScaler,
        MTPLossLoggingHelper,
        roll_tensor,
    )

    if getattr(_mtp_mod.process_mtp_loss, "_cppmega_native_hopper_patched", False):
        return  # idempotent

    _original = _mtp_mod.process_mtp_loss

    def process_mtp_loss_patched(
        hidden_states,
        output_weight,
        labels,
        loss_mask,
        output_layer,
        config,
        runtime_gather_output=False,
        is_training=True,
        cp_group=None,
        packed_seq_params=None,
        scale_logits_fn=None,
    ):
        # Non-fusion branch: defer to upstream (preserves all behaviour).
        fuse_linear_cross_entropy = (
            getattr(config, "cross_entropy_loss_fusion", False)
            and getattr(config, "cross_entropy_fusion_impl", None) == "linear"
        )
        if not fuse_linear_cross_entropy:
            return _original(
                hidden_states=hidden_states,
                output_weight=output_weight,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=output_layer,
                config=config,
                runtime_gather_output=runtime_gather_output,
                is_training=is_training,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                scale_logits_fn=scale_logits_fn,
            )

        # Fused branch — rewritten with reduction="sum" to sidestep the
        # transpose round-trip NaN bug on shared-weight multi-call.
        hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
        hidden_states = hidden_states_list[0]

        if labels is None:
            return hidden_states

        mtp_labels = labels.clone()
        if loss_mask is None:
            loss_mask = torch.ones_like(mtp_labels)

        original_num_tokens = loss_mask.sum()

        for mtp_layer_number in range(config.mtp_num_layers):
            mtp_labels, _ = roll_tensor(
                mtp_labels,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
            )
            loss_mask, num_tokens = roll_tensor(
                loss_mask,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
            )

            # Fold loss_mask into ignore_index so reduction="sum" excludes
            # masked positions directly inside the kernel. This means we do
            # not need the elementwise loss_mask * mtp_loss multiply that
            # requires reduction="none".
            masked_labels = torch.where(
                loss_mask.bool(), mtp_labels, torch.full_like(mtp_labels, -100)
            )

            # reduction="sum" branch returns a scalar and skips the
            # labels.transpose(0,1)/loss.view_as round-trip that breaks
            # autograd on a shared weight called main head + N MTP depths.
            mtp_loss_sum = output_layer(
                hidden_states_list[mtp_layer_number + 1],
                weight=output_weight,
                runtime_gather_output=runtime_gather_output,
                output_cross_entropy_loss=True,
                labels=masked_labels,
                reduction="sum",
                ignore_index=-100,
            )

            if is_training:
                mtp_loss_for_log = (
                    mtp_loss_sum * (num_tokens > 0).to(mtp_loss_sum.dtype)
                ) / num_tokens.clamp(min=1)
                MTPLossLoggingHelper.save_loss_to_tracker(
                    mtp_loss_for_log,
                    mtp_layer_number,
                    config.mtp_num_layers,
                    avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                )

            mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers

            if config.calculate_per_token_loss:
                num_tokens_safe = torch.clamp(num_tokens, min=1)
                # mtp_loss_sum is already a scalar (sum over unmasked tokens).
                # Re-scale by original_num_tokens/num_tokens_safe to match
                # upstream per-token-loss semantics.
                mtp_loss_normalized = (
                    mtp_loss_scale * mtp_loss_sum * (original_num_tokens / num_tokens_safe)
                )
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_normalized)
            else:
                safe_num_tokens = num_tokens.clamp(min=1)
                hidden_states = MTPLossAutoScaler.apply(
                    hidden_states, mtp_loss_scale * mtp_loss_sum / safe_num_tokens
                )

        return hidden_states

    process_mtp_loss_patched._cppmega_native_hopper_patched = True  # type: ignore[attr-defined]
    process_mtp_loss_patched._cppmega_original = _original  # type: ignore[attr-defined]
    _mtp_mod.process_mtp_loss = process_mtp_loss_patched

    # Rebind on every module that did `from ... import process_mtp_loss`
    # BEFORE our patch ran. These modules hold a local reference to the
    # original function and attribute-level patching on the source module
    # does NOT propagate. In particular ``mamba_model`` imports via
    # ``from megatron.core.transformer.multi_token_prediction import
    # process_mtp_loss`` (mamba_model.py:21-24).
    _rebind_targets = (
        "megatron.core.models.mamba.mamba_model",
        "megatron.core.models.gpt.gpt_model",
        "megatron.core.models.common.language_module.language_module",
    )
    for _name in _rebind_targets:
        _mod = sys.modules.get(_name)
        if _mod is not None and hasattr(_mod, "process_mtp_loss"):
            setattr(_mod, "process_mtp_loss", process_mtp_loss_patched)
            print(f"[cppmega] rebound process_mtp_loss on {_name}")

    print(
        "[cppmega] process_mtp_loss monkey-patched: fuse_linear_cross_entropy "
        "branch now uses reduction=\"sum\" + ignore_index=-100 "
        "(sidesteps PR #3345 transpose round-trip NaN on shared-weight multi-call)."
    )
