"""Swap MambaModel.output_layer to LinearCrossEntropyModule to fuse GEMM + CE.

Always-on patch: installs unconditionally when imported.

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

**Kernel selection** (probe-based, post-PR #3345):
  0. Probe Megatron's native ``Platform()`` dispatcher. If it accepts the
     current compute capability (cc[0]==9 Hopper via PR #3345, cc[0]==10
     Blackwell native), keep the native path — it has correct ``reduction``
     handling and is the reference implementation going forward.
  1. Otherwise (cc[0]==12 GB10 sm_121, or any other unsupported cc), prefer
     Apple Cut Cross Entropy (CCE) if ``cut_cross_entropy`` is importable.
     CCE's backward handles ``reduction="none"`` correctly (unlike Liger,
     which has a known bug where the bwd assumes ``reduction="sum"``).
  2. Fall back to Liger ``LigerFusedLinearCrossEntropyFunction`` when CCE
     is not installed.

Overrides:
  * ``CPPMEGA_LINEAR_CE_KERNEL=liger|cce`` — force fallback kernel even
    when native is available (for A/B testing).
  * ``CPPMEGA_PREFER_NATIVE_HOPPER_CE=1`` — legacy override, skip reroute
    entirely. Equivalent to the probe-succeeded path on cc==9 boxes.
"""
from __future__ import annotations

import os
import sys

from cppmega.megatron.deprecated_paths import require_deprecated_ack


# ---------------------------------------------------------------------------
# Optional kernel imports — order matters: CCE preferred, Liger fallback.
# ---------------------------------------------------------------------------
_CCE_AVAILABLE = False
try:
    from cut_cross_entropy import linear_cross_entropy as _cce_linear_cross_entropy
    _CCE_AVAILABLE = True
except ImportError:
    _cce_linear_cross_entropy = None  # type: ignore[assignment]


def _native_dispatcher_supports(cc) -> bool:
    """Probe Megatron's native LinearCrossEntropy ``Platform()`` dispatcher.

    Returns True iff the native dispatcher has a kernel for the current
    compute capability (i.e. ``Platform()`` initializes without raising).

    Post-PR #3345: ``cc[0]==9`` (Hopper / H100 / H200) imports
    ``.linear_cross_entropy.hopper.entry``. ``cc[0]==10`` (Blackwell SM100)
    imports ``.linear_cross_entropy.blackwell.entry``. Anything else —
    including GB10 (cc[0]==12, sm_121) and Ada/Ampere — raises
    ``ValueError('Unsupported architecture: ...')``. We use that as the
    signal to install Liger/CCE instead of hardcoding cc-based checks, so
    this file stays correct as new cc backends land upstream.
    """
    try:
        from megatron.core.fusions.fused_linear_cross_entropy import (
            _get_platform,
        )
    except Exception as exc:  # pragma: no cover
        print(
            f"[cppmega] could not import native LCE dispatcher ({exc}); "
            f"will install Liger/CCE fallback",
            file=sys.stderr,
        )
        return False

    try:
        _get_platform()
    except ValueError as exc:
        # Expected signal from ``raise ValueError(f"Unsupported architecture: {cc[0]}")``.
        print(
            f"[cppmega] native LCE dispatcher rejects cc={cc[0]}.{cc[1]} "
            f"({exc}); installing Liger/CCE fallback"
        )
        return False
    except Exception as exc:  # pragma: no cover
        # Any other import/compile error -> assume native not usable.
        print(
            f"[cppmega] native LCE dispatcher failed to init at cc={cc[0]}.{cc[1]} "
            f"({type(exc).__name__}: {exc}); installing Liger/CCE fallback",
            file=sys.stderr,
        )
        return False
    return True


def _patch_linear_ce_route_to_liger() -> None:
    """Route LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss
    to the best available fused kernel.

    Decision table:

      ``CPPMEGA_LINEAR_CE_KERNEL`` (default "auto"):
        * "auto": probe ``Platform()``; use native if it accepts this cc,
          otherwise install CCE (preferred) / Liger fallback.
        * "cce": force CCE install regardless of native availability.
        * "liger": force Liger install regardless of native availability.

      ``CPPMEGA_PREFER_NATIVE_HOPPER_CE=1`` (legacy override, deprecated):
        Skip reroute unconditionally. Equivalent to "auto" on cc==9 boxes
        where PR #3345 is in the tree. Kept for backward compat with
        existing scripts.

    Post-PR #3345 behaviour per machine:
      cc[0] == 9  (Hopper H100/H200, bench3): native hopper kernel -> no reroute
      cc[0] == 10 (Blackwell SM100):           native blackwell kernel -> no reroute
      cc[0] == 12 (GB10, sm_121):              native raises -> CCE / Liger installed
    """
    import torch

    # Legacy env override: skip everything and trust native.
    if os.environ.get("CPPMEGA_PREFER_NATIVE_HOPPER_CE", "0") == "1":
        require_deprecated_ack(
            feature="CPPMEGA_PREFER_NATIVE_HOPPER_CE=1",
            ack_env="CPPMEGA_I_UNDERSTAND_PREFER_NATIVE_HOPPER_CE_IS_DEPRECATED",
            replacement="CPPMEGA_LINEAR_CE_KERNEL=auto",
            reason="It bypasses the probe-based CCE/native LinearCE route.",
        )
        return

    if not torch.cuda.is_available():
        return
    dev = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(dev)

    kernel_pref = os.environ.get("CPPMEGA_LINEAR_CE_KERNEL", "auto").lower()

    # "auto": use native path when the dispatcher accepts this cc.
    # Explicit "liger" / "cce" force the fallback regardless of cc.
    if kernel_pref == "auto" and _native_dispatcher_supports(cc):
        path = (
            "hopper (PR #3345)" if cc[0] == 9
            else "blackwell" if cc[0] == 10
            else f"sm_{cc[0]}{cc[1]}"
        )
        print(
            f"[cppmega] LinearCrossEntropyModule using native Megatron "
            f"dispatcher for cc={cc[0]}.{cc[1]} ({path})"
        )
        return

    from megatron.core.transformer.linear_cross_entropy import (
        LinearCrossEntropyModule,
    )

    # Workaround #2: non-fused Liger CE (PR #680 fixed reduction="none" bwd
    # in ops/cross_entropy.py). We materialize logits ourselves via a plain
    # matmul, then pass into LigerCrossEntropyFunction with reduction="none".
    # Correctness: exact per-token grad. Cost: [s*b, V] logits materialized
    # (~6 GiB at MBS=10 NAM56R). See reference_main_head_liger_ce_gap.md.
    if os.environ.get("CPPMEGA_MAIN_HEAD_LIGER_NONFUSED", "0") == "1":
        require_deprecated_ack(
            feature="CPPMEGA_MAIN_HEAD_LIGER_NONFUSED=1",
            ack_env=(
                "CPPMEGA_I_UNDERSTAND_MAIN_HEAD_LIGER_NONFUSED_"
                "IS_DEPRECATED_AND_MATERIALIZES_LOGITS"
            ),
            replacement="CPPMEGA_LINEAR_CE_KERNEL=auto or cce",
            reason="It materializes full logits and is only a diagnostic bridge.",
        )
        try:
            from liger_kernel.ops.cross_entropy import (
                LigerCrossEntropyFunction as _LigerCrossEntropyFunction,
            )
        except ImportError as exc:
            print(
                f"[cppmega] CPPMEGA_MAIN_HEAD_LIGER_NONFUSED=1 but "
                f"liger_kernel.ops.cross_entropy import failed ({exc}); "
                f"aborting patch",
                file=sys.stderr,
            )
            return
        _install_liger_nonfused_compute(
            LinearCrossEntropyModule, cc, _LigerCrossEntropyFunction
        )
        return

    use_cce = _CCE_AVAILABLE and kernel_pref in ("auto", "cce")

    # Always try importing Liger so we can fall back if CCE is unavailable or
    # the user requested it explicitly.
    _LigerFusedLinearCrossEntropyFunction = None
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction as _LigerFusedLinearCrossEntropyFunction,
        )
    except ImportError:
        if not use_cce:
            print(
                "[cppmega] liger_kernel not installed and CCE unavailable — "
                "cannot route linear CE to a compatible kernel",
                file=sys.stderr,
            )
            return

    if use_cce:
        _install_cce_compute(LinearCrossEntropyModule, cc)
        return

    if _LigerFusedLinearCrossEntropyFunction is None:
        return

    _install_liger_compute(
        LinearCrossEntropyModule, cc, _LigerFusedLinearCrossEntropyFunction
    )


def _install_cce_compute(LinearCrossEntropyModule, cc) -> None:
    """Install CCE-based ``_compute_linear_and_cross_entropy_loss``.

    CCE's ``linear_cross_entropy`` supports ``reduction="none" | "sum" | "mean"``
    end-to-end (fwd + bwd) and never materializes logits. Its backward gives
    correct per-token gradients for the LM head weight.
    """
    import torch

    assert _cce_linear_cross_entropy is not None

    def _cce_compute_linear_and_cross_entropy_loss(
        self,
        hidden,
        weight,
        labels=None,
        reduction="none",
        ignore_index=-100,
    ):
        # Shapes entering: hidden [s, b, h], weight [V, h], labels [b, s].
        s, b, hdim = hidden.shape
        labels_sb = labels.transpose(0, 1).contiguous().reshape(-1)
        hidden_2d = hidden.contiguous().reshape(s * b, hdim)
        weight = weight.contiguous()

        # CCE's backward only accepts bf16/fp16 hidden states.
        if hidden_2d.dtype not in (torch.float16, torch.bfloat16):
            target_dtype = (
                weight.dtype
                if weight.dtype in (torch.float16, torch.bfloat16)
                else torch.bfloat16
            )
            hidden_2d = hidden_2d.to(target_dtype)

        # Clone targets defensively to avoid autograd in-place version bumps
        # under compile / graph capture (matches nanochat CCE wrapper).
        labels_sb = labels_sb.detach().clone()

        loss = _cce_linear_cross_entropy(
            e=hidden_2d,
            c=weight,
            targets=labels_sb,
            ignore_index=ignore_index,
            softcap=None,
            reduction=reduction,
            # Disable filter_eps: with a small/uncentered lm_head init the
            # default "high" threshold can skip all backward blocks → zero
            # gradients. Per nanochat CCE wrapper.
            filter_eps=None,
        )

        if reduction == "none":
            # CCE returns [s*b]; reshape to [s, b] then transpose to [b, s].
            return loss.reshape(s, b).transpose(0, 1).contiguous()
        if reduction in ("sum", "mean"):
            return loss
        raise ValueError(f"Unsupported reduction: {reduction}")

    LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss = (
        _cce_compute_linear_and_cross_entropy_loss
    )
    print(
        f"[cppmega] LinearCrossEntropyModule routed to Apple CCE kernel "
        f"(cc={cc[0]}.{cc[1]}, pre-Blackwell; supports reduction='none' in bwd)"
    )


def _install_liger_nonfused_compute(
    LinearCrossEntropyModule, cc, LigerCrossEntropyFunction
) -> None:
    """Install non-fused Liger CE compute (workaround #2 for Liger fused bwd bug).

    ### Scope of Liger PR #680 — read carefully

    Liger PR #680 (merged, Apr 2025) fixed ``reduction="none"`` backward for
    ``LigerCrossEntropyLoss`` / ``LigerCrossEntropyFunction`` — the plain
    CE-on-logits path. The fix hunk is::

        elif grad_output.ndim > 0:
            _input = _input * grad_output.unsqueeze(dim=1)

    **PR #680 does NOT fix ``LigerFusedLinearCrossEntropyFunction`` (FLCE).**
    In fact #680 contains a commit *"Remove reduction='none' from flce"* —
    upstream explicitly acknowledged the FLCE backward was broken and
    removed the option rather than fix it. Liger issue #968 tracks the
    FLCE silent-corruption bug; draft PR #1126 only adds an assertion guard,
    not a fix. FLCE ``reduction="none"`` backward remains broken upstream
    as of 2026-04-14.

    ### How the three workaround paths in this file map to #680

    * **Path #2 (this function) — non-fused Liger.** Manually does the
      ``hidden @ weight.T`` matmul, then calls ``LigerCrossEntropyFunction``
      (the plain-CE path that #680 DID fix). ``reduction="none"`` backward
      is correct here. Memory regression (~6 GiB at NAM56R MBS=10, V≈150k,
      bf16) is the cost of materialising logits. Peak-alloc watch: MBS=12
      adds ~1.2× and is expected to crash the 141 GiB budget.
    * **Path #3 — fused Liger with reduction="mean" broadcast** (see
      ``_install_liger_compute`` below). Uses ``LigerFusedLinearCrossEntropy
      Function`` but only calls it with ``reduction="mean"`` because the
      FLCE ``reduction="none"`` backward (the bug PR #680 did NOT fix)
      still silently corrupts gradients. The mean-scalar is then broadcast
      to ``[b, s]`` so Megatron's tensor-shaped loss contract holds, while
      the kernel only ever executes the mean code path.

    **Boundary:** #680 fixed plain CE (path #2). #680 did NOT fix FLCE
    (path #3 exists precisely because FLCE backward is still broken).

    API:
        LigerCrossEntropyFunction.apply(
            _input: [BT, V] logits,
            target: [BT] int64,
            weight: Optional[V],     # class weight (unused here)
            ignore_index: int,
            lse_square_scale: float,
            label_smoothing: float,
            reduction: str,
            softcap: Optional[float],
            return_z_loss: bool,
            return_token_accuracy: bool,
        ) -> (loss, z_loss, token_accuracy)

    ``loss`` shape is ``[BT]`` when reduction=="none", scalar otherwise.
    """
    import torch

    def _liger_nonfused_compute_linear_and_cross_entropy_loss(
        self,
        hidden,
        weight,
        labels=None,
        reduction="none",
        ignore_index=-100,
    ):
        # Shapes entering: hidden [s, b, h], weight [V, h], labels [b, s].
        s, b, hdim = hidden.shape
        labels_sb = labels.transpose(0, 1).contiguous().reshape(-1)  # [s*b]
        hidden_2d = hidden.contiguous().reshape(s * b, hdim)         # [s*b, h]

        # Materialize logits via standard ColumnParallelLinear-style matmul.
        # NOTE: weight.shape == [V, h], so logits = hidden_2d @ weight.T.
        # We explicitly use bf16 to match weight dtype and avoid fp32 blow-up.
        logits = torch.matmul(hidden_2d, weight.t().contiguous())    # [s*b, V]

        # Liger expects int64 target.
        if labels_sb.dtype != torch.int64:
            labels_sb = labels_sb.to(torch.int64)

        per_token_loss, _z, _acc = LigerCrossEntropyFunction.apply(
            logits,
            labels_sb,
            None,          # weight (class weight; we don't use it)
            ignore_index,
            0.0,           # lse_square_scale
            0.0,           # label_smoothing
            "none",        # reduction — PR #680 fix makes bwd correct here
            None,          # softcap
            False,         # return_z_loss
            False,         # return_token_accuracy
        )

        if reduction == "none":
            # per_token_loss is [s*b]; reshape [s, b] then transpose to [b, s].
            return per_token_loss.reshape(s, b).transpose(0, 1).contiguous()
        if reduction == "mean":
            # Mean over non-ignored tokens (matches torch CE default).
            valid = (labels_sb != ignore_index)
            n = valid.sum().clamp_min(1).to(per_token_loss.dtype)
            return per_token_loss.sum() / n
        if reduction == "sum":
            return per_token_loss.sum()
        raise ValueError(f"Unsupported reduction: {reduction}")

    LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss = (
        _liger_nonfused_compute_linear_and_cross_entropy_loss
    )
    print(
        f"[cppmega] LinearCrossEntropyModule routed to Liger NON-FUSED kernel "
        f"(cc={cc[0]}.{cc[1]}, fix path #2; PR #680 reduction='none' bwd fix; "
        f"materializes [s*b, V] logits)"
    )


def _install_liger_compute(
    LinearCrossEntropyModule, cc, LigerFusedLinearCrossEntropyFunction
) -> None:
    """Install Liger-based ``_compute_linear_and_cross_entropy_loss`` (fallback)."""

    def _liger_compute_linear_and_cross_entropy_loss(
        self,
        hidden,
        weight,
        labels=None,
        reduction="none",
        ignore_index=-100,
    ):
        # Shapes entering: hidden [s, b, h], weight [V, h], labels [b, s].
        #
        # **Fix path A for Liger backward bug** (linkedin/Liger-Kernel#968,
        # draft fix #1126). Liger's ``LigerFusedLinearCrossEntropyFunction
        # .backward`` uses a scalar-pointer path for ``grad_output`` when
        # ``reduction != "mean"``; passing ``reduction="none"`` produces a
        # per-token grad_output tensor that the kernel then reads OOB →
        # silent corruption / nan. Symptom on NAM56R MBS=10 was erratic
        # grad_norm (93 -> 58 -> 417 -> 168) and MBS=12 backward-nan.
        #
        # Mitigation: always call Liger with ``reduction="mean"`` (the only
        # code path whose backward is correct today). When the caller wants
        # ``reduction="none"`` (Megatron's default for the ``[b, s]`` tensor
        # path), we broadcast the scalar mean to a ``[b, s]`` tensor via
        # ``.expand().contiguous()``. Forward sum = scalar_mean * N (same
        # value as a true per-token sum, since mean * N = sum). Backward:
        # ``d(sum([b,s]))/d(scalar_mean) = N``, which cancels exactly against
        # the ``1/N`` factor baked into Liger's mean-reduction backward —
        # giving the same gradient as a correct per-token reduction path.
        # Tradeoff: Megatron's ``loss_mask`` applied element-wise becomes
        # effectively a multiplier by ``mask.sum() / N_valid``. For
        # pretraining where mask is 1 on all non-pad tokens and pad tokens
        # are ``ignore_index`` anyway, mask.sum() == N_valid and this is
        # exact. Per-token loss logging becomes uniform across ``[b, s]``
        # (all entries == mean), but the reported scalar loss is unchanged.
        s, b, hdim = hidden.shape
        labels_sb = labels.transpose(0, 1).contiguous().reshape(-1)
        hidden_2d = hidden.contiguous().reshape(s * b, hdim)
        weight = weight.contiguous()

        liger_loss_scalar, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
            hidden_2d,
            weight,
            labels_sb,
            None,         # bias
            None,         # ce_weight
            ignore_index,
            0.0,          # lse_square_scale
            0.0,          # label_smoothing
            "mean",       # reduction — NEVER "none" (upstream bug #968)
            None,         # softcap
            False,        # return_z_loss
        )

        if reduction == "mean":
            return liger_loss_scalar
        if reduction == "sum":
            # sum == mean * N_valid. Backward multiplies grad by N_valid,
            # which combined with Liger's 1/N_valid mean bwd gives the
            # correct per-token sum gradient.
            n_valid = (labels_sb != ignore_index).sum().clamp_min(1)
            return liger_loss_scalar * n_valid.to(liger_loss_scalar.dtype)
        if reduction == "none":
            # Broadcast scalar to [b, s]. `.expand` is a view whose backward
            # sums into the source scalar — total factor = b*s (or N_valid
            # after loss_mask). Combined with Liger's mean 1/N_valid bwd
            # this yields the same grad as a true per-token reduction in
            # the common case where loss_mask == (labels != ignore_index).
            return liger_loss_scalar.expand(b, s).contiguous()
        raise ValueError(f"Unsupported reduction: {reduction}")

    LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss = (
        _liger_compute_linear_and_cross_entropy_loss
    )
    print(
        f"[cppmega] LinearCrossEntropyModule routed to Liger kernel "
        f"(cc={cc[0]}.{cc[1]}, pre-Blackwell; using reduction='mean' + "
        f"broadcast to work around Liger bwd bug #968)"
    )


def patch_mamba_output_layer_with_linear_ce() -> None:
    """Monkey-patch MambaModel.__init__ to swap output_layer class.

    Always installs — no env gate.  Upstream Megatron ``MambaModel`` uses
    plain ``ColumnParallelLinear`` for output_layer while ``GPTModel`` uses
    ``LinearCrossEntropyModule`` (PR #3226 wired both, PR #3207 silently
    reverted Mamba — see upstream_prs/11_megatron_mamba_linear_ce_module.md).
    Without this class swap the fused linear CE path raises TypeError on
    ``output_cross_entropy_loss`` / ``labels`` kwargs, forcing the vanilla
    CE path which materialises ~12 GiB FP32 logits — MBS=10 OOM.

    Raises on any import/install failure — config is invalid, don't
    silently degrade.
    """

    from megatron.core import tensor_parallel
    from megatron.core.models.mamba.mamba_model import MambaModel
    from megatron.core.transformer.linear_cross_entropy import (
        LinearCrossEntropyModule,
    )

    # Re-route fused kernel to CCE (preferred) or Liger (fallback) on
    # Hopper so the native path doesn't hit
    # ``ValueError: Unsupported architecture: 9``.
    _patch_linear_ce_route_to_liger()

    if getattr(MambaModel, "_cppmega_linear_ce_patched", False):
        return

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
    MambaModel._cppmega_linear_ce_patched = True
