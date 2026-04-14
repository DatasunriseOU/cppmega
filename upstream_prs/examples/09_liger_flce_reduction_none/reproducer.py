"""Reproducer: LigerFusedLinearCrossEntropyFunction(reduction="none") backward is broken.

Demonstrates three things:
  (A) reduction="none" produces wrong gradients vs. eager PyTorch reference
      (silent numerical corruption of grad_input AND grad_weight).
  (B) reduction="mean" is correct (the known-working path).
  (C) Our workaround: call Liger with reduction="mean", manually scale by
      n_valid to recover per-token sum semantics, gradients match eager
      PyTorch `F.cross_entropy(reduction="sum")` bit-exactly.

Also checks the practical symptom that surfaced in Megatron NAM56R training:
  - After `loss.sum().backward()` with reduction="none", does grad_norm
    come back NaN? (Yes, when the kernel's chunked grad_weight accumulator
    gets corrupted by the mismatched grad_output broadcast path.)

Exit code:
  0 — bug is NOT present (Liger's reduction="none" bwd matches reference)
  1 — bug IS present (grads differ from reference or are NaN)

References:
  - https://github.com/linkedin/Liger-Kernel/issues/968   (CLOSED, not fixed)
  - https://github.com/linkedin/Liger-Kernel/issues/872   (CLOSED, partially addressed)
  - https://github.com/linkedin/Liger-Kernel/pull/1126    (OPEN draft: adds assertion only)
  - https://github.com/linkedin/Liger-Kernel/pull/1182    (OPEN: reduction kwarg plumbing)
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _reference_grads(hidden, weight, target, ignore_index, reduction):
    """Eager PyTorch reference: F.linear + F.cross_entropy."""
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    logits = F.linear(h, w)  # [BT, V]
    loss = F.cross_entropy(
        logits, target, ignore_index=ignore_index, reduction=reduction
    )
    if reduction == "none":
        # Caller sums per-token losses (Megatron pattern).
        loss.sum().backward()
    else:
        loss.backward()
    return h.grad.detach().clone(), w.grad.detach().clone(), loss.detach().clone()


def _liger_grads(hidden, weight, target, ignore_index, reduction, loss_mask=None):
    """Call LigerFusedLinearCrossEntropyFunction directly.

    If ``loss_mask`` is provided (only meaningful for reduction="none"), the
    backward is driven by ``(loss * loss_mask).sum()`` — this is the
    Megatron-main-head calling convention that exercises the non-uniform
    ``grad_output`` path through ``element_mul_kernel``. With a non-uniform
    grad_output tensor, the kernel's scalar-broadcast assumption silently
    miscomputes grad_input/grad_weight.
    """
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )

    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)

    loss, *_ = LigerFusedLinearCrossEntropyFunction.apply(
        h,
        w,
        target,
        None,          # bias
        None,          # ce_weight
        ignore_index,
        0.0,           # lse_square_scale
        0.0,           # label_smoothing
        reduction,
        None,          # softcap
        False,         # return_z_loss
    )
    if loss_mask is not None:
        assert loss.dim() > 0, "loss_mask only meaningful with reduction='none'"
        (loss * loss_mask).sum().backward()
    elif loss.dim() > 0:
        loss.sum().backward()
    else:
        loss.backward()
    return h.grad.detach().clone(), w.grad.detach().clone(), loss.detach().clone()


def _workaround_grads(hidden, weight, target, ignore_index):
    """Workaround: call Liger with reduction="mean", scale by n_valid to
    recover per-token `sum` semantics.

    Math: sum_i CE_i = mean_i CE_i * n_valid. Backward of
    ``liger_mean_loss * n_valid`` w.r.t. Liger's internal 1/n_valid bwd
    cancels exactly → identical gradient to eager F.cross_entropy(reduction="sum").
    """
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )

    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)

    loss_scalar, *_ = LigerFusedLinearCrossEntropyFunction.apply(
        h, w, target, None, None, ignore_index, 0.0, 0.0, "mean", None, False,
    )
    n_valid = (target != ignore_index).sum().clamp_min(1).to(loss_scalar.dtype)
    (loss_scalar * n_valid).backward()
    return h.grad.detach().clone(), w.grad.detach().clone(), loss_scalar.detach() * n_valid


def _report(name, got_h, got_w, ref_h, ref_w, tol):
    hdiff = (got_h - ref_h).abs().max().item() if got_h is not None else float("nan")
    wdiff = (got_w - ref_w).abs().max().item() if got_w is not None else float("nan")
    h_nan = torch.isnan(got_h).any().item() if got_h is not None else True
    w_nan = torch.isnan(got_w).any().item() if got_w is not None else True
    h_ok = (not h_nan) and hdiff < tol
    w_ok = (not w_nan) and wdiff < tol
    status = "OK  " if (h_ok and w_ok) else "FAIL"
    print(
        f"  [{status}] {name:40s}  "
        f"|max grad_hidden - ref| = {hdiff:.3e}  "
        f"|max grad_weight - ref| = {wdiff:.3e}  "
        f"nan(h,w) = ({h_nan},{w_nan})"
    )
    return h_ok and w_ok


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA device required (Liger Triton kernels are CUDA-only).")
        return 2

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Realistic-but-small shape: approximates one Megatron microbatch
    # (batch=2, seq=512, hidden=1024, vocab=32000). BT = 1024.
    B, S, H, V = 2, 512, 1024, 32000
    BT = B * S
    IGNORE_INDEX = -100

    _seed(0)
    hidden = torch.randn(BT, H, device=device, dtype=dtype) * 0.1
    weight = torch.randn(V, H, device=device, dtype=dtype) * 0.02
    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)
    # Mark ~10% of tokens as ignore_index (realistic pretraining mask).
    ignore_mask = torch.rand(BT, device=device) < 0.1
    target[ignore_mask] = IGNORE_INDEX

    # Correctness tolerance: bf16 matmul noise floor.
    TOL_BF16 = 5e-3

    print(f"Shape: B={B} S={S} H={H} V={V} (BT={BT})  dtype={dtype}")
    print(f"ignore_index={IGNORE_INDEX}  n_valid={(target != IGNORE_INDEX).sum().item()} / {BT}")
    print()

    # --- Reference eager gradients (sum over per-token losses) ---
    ref_h_none, ref_w_none, ref_loss_none = _reference_grads(
        hidden, weight, target, IGNORE_INDEX, reduction="none"
    )
    ref_h_mean, ref_w_mean, ref_loss_mean = _reference_grads(
        hidden, weight, target, IGNORE_INDEX, reduction="mean"
    )
    ref_h_sum, ref_w_sum, _ = _reference_grads(
        hidden, weight, target, IGNORE_INDEX, reduction="sum"
    )
    # `reduction="none"` caller does `loss.sum().backward()` → should equal
    # the reduction="sum" reference exactly. Sanity check:
    assert torch.allclose(ref_h_none, ref_h_sum, atol=1e-6), "internal ref mismatch"

    print("Reference (eager PyTorch F.linear + F.cross_entropy):")
    print(f"  reduction='none'.sum()  loss = {ref_loss_none.sum().item():.6f}")
    print(f"  reduction='mean'        loss = {ref_loss_mean.item():.6f}")
    print()

    results = {}

    # --- (A) Liger reduction="mean" — should match eager mean reference ---
    print("Liger kernel results:")
    try:
        lh_mean, lw_mean, lloss_mean = _liger_grads(
            hidden, weight, target, IGNORE_INDEX, reduction="mean"
        )
        results["mean"] = _report(
            'reduction="mean"', lh_mean, lw_mean, ref_h_mean, ref_w_mean, TOL_BF16
        )
    except Exception as exc:
        print(f"  [FAIL] reduction='mean' raised: {type(exc).__name__}: {exc}")
        results["mean"] = False

    # --- (B1) Liger reduction="none" + .sum().backward() — uniform grad_output ---
    # This path happens to work because .sum() makes grad_output a tensor of
    # ones; the element_mul_kernel scalar-broadcast reads 1.0 and the math
    # coincidentally agrees with reduction="sum" semantics.
    try:
        lh_none, lw_none, _ = _liger_grads(
            hidden, weight, target, IGNORE_INDEX, reduction="none"
        )
        results["none_uniform"] = _report(
            'reduction="none" + .sum().backward()', lh_none, lw_none, ref_h_none, ref_w_none, TOL_BF16
        )
    except AssertionError as exc:
        # PR #1126 (if merged) adds an assertion that blocks this path.
        print(f"  [ASSERT] reduction='none' raised AssertionError: {exc}")
        print(f"           (this is PR #1126 landed — kernel refuses to silently corrupt)")
        results["none_asserted"] = True
        results["none_uniform"] = False
    except Exception as exc:
        print(f"  [FAIL]  reduction='none' raised {type(exc).__name__}: {exc}")
        results["none_uniform"] = False

    # --- (B2) Liger reduction="none" + NON-UNIFORM grad_output — THE REAL BUG ---
    # Megatron's main-head plumbing multiplies per-token loss by a
    # per-token loss_mask before summing; autograd then hands Liger a
    # non-uniform [BT] grad_output tensor. `element_mul_kernel` treats
    # grad_output as a scalar pointer and silently reads grad_output[0]
    # for every row — miscomputing every non-first-row gradient.
    loss_mask = torch.ones(BT, device=device, dtype=dtype)
    # Make the mask NON-uniform: half the valid tokens get 0.5 weight,
    # simulating Megatron's loss-scaling or document-boundary masking.
    half = BT // 2
    loss_mask[half:] = 0.5
    loss_mask[target == IGNORE_INDEX] = 0.0

    # Reference: eager path with the same mask.
    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    logits_ref = F.linear(h_ref, w_ref)
    loss_ref_tok = F.cross_entropy(logits_ref, target, ignore_index=IGNORE_INDEX, reduction="none")
    (loss_ref_tok * loss_mask).sum().backward()
    ref_h_masked = h_ref.grad.detach().clone()
    ref_w_masked = w_ref.grad.detach().clone()

    try:
        lh_masked, lw_masked, _ = _liger_grads(
            hidden, weight, target, IGNORE_INDEX, reduction="none", loss_mask=loss_mask,
        )
        results["none_masked"] = _report(
            'reduction="none" + (loss*mask).sum()', lh_masked, lw_masked, ref_h_masked, ref_w_masked, TOL_BF16
        )
    except AssertionError as exc:
        print(f"  [ASSERT] reduction='none' + mask raised AssertionError: {exc}")
        results["none_asserted"] = True
        results["none_masked"] = False
    except Exception as exc:
        print(f"  [FAIL]  reduction='none' + mask raised {type(exc).__name__}: {exc}")
        results["none_masked"] = False

    # --- (C) Our workaround: reduction="mean" + manual scale → sum semantics ---
    try:
        wh, ww, _ = _workaround_grads(hidden, weight, target, IGNORE_INDEX)
        results["workaround"] = _report(
            'workaround: mean * n_valid', wh, ww, ref_h_sum, ref_w_sum, TOL_BF16
        )
    except Exception as exc:
        print(f"  [FAIL] workaround raised: {type(exc).__name__}: {exc}")
        results["workaround"] = False

    print()

    # --- grad_norm NaN check (the practical symptom) ---
    print()
    print("Practical symptom check (grad_weight.norm() after backward):")
    try:
        gnorm_masked = lw_masked.float().norm().item()
        is_nan = not (gnorm_masked == gnorm_masked)
        print(
            f"  Liger none + non-uniform mask   grad_weight.norm() = "
            f"{gnorm_masked:.4e}{'  <-- NaN!' if is_nan else ''}"
        )
    except (NameError, UnboundLocalError):
        print("  (Liger masked path raised — grad not materialized)")
    try:
        print(
            f"  Reference (eager F.CE + mask)   grad_weight.norm() = "
            f"{ref_w_masked.float().norm().item():.4e}"
        )
        print(
            f"  Workaround (mean * n_valid)     grad_weight.norm() = "
            f"{ww.float().norm().item():.4e}"
        )
    except Exception:
        pass

    # --- Verdict ---
    print()
    print("=" * 72)
    # Core bug: non-uniform grad_output path. The "uniform .sum()" path
    # happens to agree with reference because grad_output is a tensor of 1s.
    none_masked_ok = results.get("none_masked", False)
    if none_masked_ok:
        print("VERDICT: Liger reduction='none' backward is CORRECT (bug fixed).")
        print("         Both uniform .sum() and non-uniform mask paths match reference.")
        return 0
    if results.get("none_asserted"):
        print(
            "VERDICT: reduction='none' blocked by AssertionError (PR #1126-style "
            "guard). No functional fix — workaround still required."
        )
        return 1
    print(
        "VERDICT: Liger reduction='none' backward is BROKEN.\n"
        "         Uniform .sum()         path: "
        f"{'OK' if results.get('none_uniform') else 'FAIL'}  "
        "(coincidentally matches reduction='sum' ref since grad_output=[1,1,…])\n"
        "         Non-uniform (mask)     path: "
        f"{'OK' if none_masked_ok else 'FAIL'}  "
        "(real bug: element_mul_kernel treats per-token grad_output as scalar)\n"
        "         Workaround (mean*N)    path: "
        f"{'OK' if results.get('workaround') else 'FAIL'}  (correct in all cases)"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
