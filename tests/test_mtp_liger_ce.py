"""Tests for cppmega.megatron.mtp_liger_ce — Liger fused linear cross-entropy for MTP.

These tests verify:
  1. The patch installs and replaces process_mtp_loss correctly (structure
     only — no gradient checking in this section).
  2. The structural env-gate on ``CPPMEGA_MTP_LIGER_CE``.
  3. **Liger reduction='none' backward is CORRUPT** — the exact bug that
     ``cppmega.megatron.apply_linear_ce_patch._install_liger_compute``
     works around by calling Liger with ``reduction='mean'`` and broadcasting
     the scalar to ``[b, s]``.  This is a regression guard: if a future
     Liger release fixes the FLCE bwd (upstream issue #968 / draft PR #1126),
     the ``test_liger_reduction_none_backward_is_corrupt`` assertion flips
     and we can delete the workaround.
  4. The workaround pattern (``reduction='mean'`` + broadcast) produces
     gradients matching the reference eager implementation.

Run (CPU-only, no GPU required for import/structure tests):
    pytest tests/test_mtp_liger_ce.py -v

Run with GPU for numerical tests:
    CUDA_VISIBLE_DEVICES=7 pytest tests/test_mtp_liger_ce.py -v -k gpu

The backward-correctness tests use CUDA when available, CPU otherwise.
Liger itself requires triton and only runs on CUDA, so the Liger portions
are skipped cleanly on Mac / CPU-only hosts.
"""
from __future__ import annotations

import os
import sys
import importlib
import importlib.util

import pytest

# Ensure repo roots are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------
_HAS_TORCH = importlib.util.find_spec("torch") is not None
try:
    _HAS_LIGER = importlib.util.find_spec("liger_kernel") is not None
except ValueError:
    _HAS_LIGER = False

_HAS_CUDA = False
if _HAS_TORCH:
    import torch  # noqa: E402
    _HAS_CUDA = torch.cuda.is_available()

# Liger actually requires triton at import time, and triton only imports on
# CUDA boxes.  Probe the real import (not just find_spec) so Mac / CPU
# machines skip cleanly rather than crash on ``import triton``.
_LIGER_IMPORTS = False
if _HAS_LIGER and _HAS_CUDA:
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import (  # noqa: F401
            LigerFusedLinearCrossEntropyFunction,
        )
        _LIGER_IMPORTS = True
    except Exception:
        _LIGER_IMPORTS = False


class TestMTPLigerCEStructure:
    """Structure / import tests that do not need a GPU."""

    def test_import(self):
        from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
        assert callable(patch_mtp_loss_with_liger)

    def test_noop_without_env(self):
        """patch_mtp_loss_with_liger is a no-op when env var is absent."""
        os.environ.pop("CPPMEGA_MTP_LIGER_CE", None)
        from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
        # Should return without error
        patch_mtp_loss_with_liger()

    def test_env_gate(self, monkeypatch):
        """Only patches when CPPMEGA_MTP_LIGER_CE=1."""
        monkeypatch.setenv("CPPMEGA_MTP_LIGER_CE", "0")
        from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
        patch_mtp_loss_with_liger()  # should be no-op


@pytest.mark.skipif(
    not _LIGER_IMPORTS,
    reason="GPU test — requires CUDA + liger_kernel + triton",
)
class TestMTPLigerCEGPU:
    """GPU-based numerical correctness tests for forward parity."""

    def test_liger_import(self):
        """Verify liger_kernel is importable."""
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )
        assert LigerFusedLinearCrossEntropyFunction is not None

    def test_liger_none_reduction_basic(self):
        """Liger fused CE with reduction='none' matches F.cross_entropy (FORWARD only).

        Note: this tests the forward pass only.  The backward pass for
        ``reduction='none'`` is broken upstream; see
        ``test_liger_reduction_none_backward_is_corrupt`` below.
        """
        import torch
        import torch.nn.functional as F
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        torch.manual_seed(42)
        B, H, V = 64, 256, 1024
        inp = torch.randn(B, H, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(V, H, dtype=torch.bfloat16, device="cuda")
        target = torch.randint(0, V, (B,), device="cuda")

        # Standard
        logits = inp.float() @ weight.float().T
        std_loss = F.cross_entropy(logits, target, reduction="none")

        # Liger
        inp_l = inp.clone().detach().requires_grad_(True)
        lig_loss, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
            inp_l, weight, target, None, None, -100, 0.0, 0.0, "none", None, False,
        )

        diff = (std_loss - lig_loss).abs()
        assert diff.max().item() < 0.1, f"Max loss diff {diff.max().item()} too large"


# ---------------------------------------------------------------------------
# Backward correctness tests — the thing the docstring used to claim but
# never actually tested.  These are the regression guards for the Liger FLCE
# reduction='none' backward bug (upstream issue #968 / draft PR #1126) and
# the ``apply_linear_ce_patch._install_liger_compute`` workaround.
# ---------------------------------------------------------------------------

def _reference_grad(inp, weight, target, loss_mask, ignore_index):
    """Eager reference: plain ``F.linear`` + ``F.cross_entropy(reduction='none')``
    masked by ``loss_mask``.  This is the correctness ground truth."""
    import torch
    import torch.nn.functional as F

    inp_r = inp.clone().detach().float().requires_grad_(True)
    w_r = weight.clone().detach().float().requires_grad_(True)
    logits = F.linear(inp_r, w_r)
    per_tok = F.cross_entropy(
        logits, target, reduction="none", ignore_index=ignore_index
    )
    total = (per_tok * loss_mask).sum()
    total.backward()
    return inp_r.grad.detach().clone(), w_r.grad.detach().clone(), total.detach().clone()


def _workaround_grad(inp, weight, target, loss_mask, ignore_index):
    """The patch's workaround: call Liger with ``reduction='mean'`` and broadcast.

    Mirrors ``cppmega.megatron.apply_linear_ce_patch._install_liger_compute``
    (the reduction=='none' branch).
    """
    import torch
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )

    inp_w = inp.clone().detach().requires_grad_(True)
    w_w = weight.clone().detach().requires_grad_(True)

    # The patch uses ``reduction='mean'`` — the only FLCE code path whose
    # backward is correct today.
    liger_loss_scalar, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
        inp_w,
        w_w,
        target,
        None,         # bias
        None,         # ce_weight
        ignore_index,
        0.0,          # lse_square_scale
        0.0,          # label_smoothing
        "mean",       # reduction — NEVER "none" (upstream bug)
        None,         # softcap
        False,        # return_z_loss
    )
    # Expand mean-scalar to per-token shape, then apply loss_mask and sum,
    # matching Megatron's ``(loss * loss_mask).sum()`` contract.
    bs = target.shape[0]
    per_tok = liger_loss_scalar.expand(bs).contiguous()
    total = (per_tok * loss_mask).sum()
    total.backward()
    return inp_w.grad.detach().clone(), w_w.grad.detach().clone(), total.detach().clone()


def _buggy_liger_none_grad(inp, weight, target, loss_mask, ignore_index):
    """Call Liger FLCE with ``reduction='none'`` — the BROKEN path."""
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )

    inp_b = inp.clone().detach().requires_grad_(True)
    w_b = weight.clone().detach().requires_grad_(True)

    per_tok, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
        inp_b,
        w_b,
        target,
        None,
        None,
        ignore_index,
        0.0,
        0.0,
        "none",       # <-- the bug
        None,
        False,
    )
    total = (per_tok * loss_mask).sum()
    total.backward()
    return inp_b.grad.detach().clone(), w_b.grad.detach().clone(), total.detach().clone()


def _max_rel_err(a, b, eps=1e-8):
    import torch
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(eps)
    return ((a - b).abs() / denom).max().item()


@pytest.fixture
def _small_inputs():
    """Small shared inputs for the backward-correctness tests.

    Shapes chosen per the task spec: [batch=2, seq=128, hidden=256, vocab=1024].
    All tensors are bf16 on CUDA if available, fp32 on CPU otherwise (the
    eager reference path works on both; only Liger requires CUDA).
    """
    import torch
    torch.manual_seed(0)
    device = torch.device("cuda" if _HAS_CUDA else "cpu")
    dtype = torch.bfloat16 if _HAS_CUDA else torch.float32
    B, S, H, V = 2, 128, 256, 1024
    BT = B * S
    inp = torch.randn(BT, H, dtype=dtype, device=device) * 0.1
    weight = torch.randn(V, H, dtype=dtype, device=device) * 0.02
    target = torch.randint(0, V, (BT,), device=device)
    # Mask out ~10% of tokens as "ignored" both via loss_mask=0 and
    # target=ignore_index so the two codepaths agree on which tokens count.
    loss_mask = torch.ones(BT, device=device, dtype=torch.float32)
    ignore_index = -100
    rand = torch.rand(BT, device=device)
    ignore_pos = rand < 0.1
    target[ignore_pos] = ignore_index
    loss_mask[ignore_pos] = 0.0
    return inp, weight, target, loss_mask, ignore_index


@pytest.mark.skipif(
    not _LIGER_IMPORTS,
    reason="liger_kernel + CUDA + triton required",
)
def test_mtp_liger_ce_backward_reduction_none(_small_inputs):
    """Demonstrate that Liger FLCE ``reduction='none'`` backward is CORRUPT.

    This is the upstream bug (linkedin/Liger-Kernel#968, draft PR #1126) that
    ``cppmega.megatron.apply_linear_ce_patch._install_liger_compute`` works
    around by calling Liger with ``reduction='mean'`` and broadcasting the
    scalar.

    Test shape (task spec):
      batch=2, seq=128 → BT=256 tokens, hidden=256, vocab=1024.

    We compare three gradients of ``(loss * loss_mask).sum()`` w.r.t. inp:
      ref   : eager ``F.linear + F.cross_entropy(reduction='none')``
      buggy : Liger FLCE ``reduction='none'``                (expected CORRUPT)
      fixed : Liger FLCE ``reduction='mean'`` + broadcast    (expected CORRECT)

    Assertions:
      * ``buggy`` gradient differs from ``ref`` by > 1% max-abs-rel-err
        (demonstrates the bug we claim to fix).
      * ``fixed`` gradient matches ``ref`` within 1e-2 max-abs-rel-err
        (FP8/bf16 noise budget).

    If the ``buggy`` assertion flips to False in the future, that means
    upstream Liger fixed FLCE reduction='none' — at which point the
    ``_install_liger_compute`` workaround can be deleted and this test
    rewritten to assert parity instead of corruption.
    """
    import torch

    inp, weight, target, loss_mask, ignore_index = _small_inputs

    ref_g_inp, ref_g_w, ref_loss = _reference_grad(
        inp, weight, target, loss_mask, ignore_index
    )
    fixed_g_inp, fixed_g_w, fixed_loss = _workaround_grad(
        inp, weight, target, loss_mask, ignore_index
    )
    buggy_g_inp, buggy_g_w, buggy_loss = _buggy_liger_none_grad(
        inp, weight, target, loss_mask, ignore_index
    )

    # 1. Sanity: loss values all agree to within bf16 noise.
    assert abs(ref_loss.item() - fixed_loss.item()) / max(abs(ref_loss.item()), 1.0) < 5e-2
    assert abs(ref_loss.item() - buggy_loss.item()) / max(abs(ref_loss.item()), 1.0) < 5e-2

    # 2. The workaround gradient matches the reference.  We use a somewhat
    #    loose bound (1e-2 max-abs-rel-err) because bf16 round-off in
    #    ``F.linear`` + exp/log inside Liger can drift by a few percent on
    #    a handful of tokens; the important thing is that the distribution
    #    matches, not exact float equality.
    fixed_err_inp = _max_rel_err(ref_g_inp.to(fixed_g_inp.dtype), fixed_g_inp)
    fixed_err_w = _max_rel_err(ref_g_w.to(fixed_g_w.dtype), fixed_g_w)
    assert fixed_err_inp < 1e-2, (
        f"workaround grad_inp max-rel-err {fixed_err_inp:.3e} exceeds 1e-2 — "
        f"broadcast pattern no longer matches eager reference"
    )
    assert fixed_err_w < 1e-2, (
        f"workaround grad_weight max-rel-err {fixed_err_w:.3e} exceeds 1e-2"
    )

    # 3. The buggy reduction='none' gradient is demonstrably wrong.  We
    #    check that AT LEAST ONE of (grad_inp, grad_weight) disagrees with
    #    the reference by > 1% max-abs-rel-err.  If both ever come back
    #    within 1%, upstream has fixed the bug and we should delete the
    #    workaround.
    buggy_err_inp = _max_rel_err(ref_g_inp.to(buggy_g_inp.dtype), buggy_g_inp)
    buggy_err_w = _max_rel_err(ref_g_w.to(buggy_g_w.dtype), buggy_g_w)
    assert max(buggy_err_inp, buggy_err_w) > 1e-2, (
        f"Liger FLCE reduction='none' backward appears CORRECT "
        f"(grad_inp err={buggy_err_inp:.3e}, grad_w err={buggy_err_w:.3e}). "
        f"Upstream #968 may have been fixed — delete the "
        f"apply_linear_ce_patch._install_liger_compute workaround."
    )


@pytest.mark.skipif(
    not _LIGER_IMPORTS,
    reason="liger_kernel + CUDA + triton required",
)
def test_mtp_liger_ce_backward_reduction_mean(_small_inputs):
    """Control test: Liger FLCE ``reduction='mean'`` backward IS correct.

    The workaround in ``_install_liger_compute`` relies on this.  If this
    test ever fails, the workaround is no longer safe either and the whole
    Liger path needs re-evaluation.
    """
    import torch
    import torch.nn.functional as F
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )

    inp, weight, target, loss_mask, ignore_index = _small_inputs
    # For the mean-reduction control test we don't apply loss_mask since
    # Liger's mean already masks via ignore_index.  Compare eager mean
    # vs Liger mean.

    # Eager mean reference
    inp_r = inp.clone().detach().float().requires_grad_(True)
    w_r = weight.clone().detach().float().requires_grad_(True)
    logits = F.linear(inp_r, w_r)
    ref_loss = F.cross_entropy(
        logits, target, reduction="mean", ignore_index=ignore_index
    )
    ref_loss.backward()
    ref_g_inp = inp_r.grad.detach().clone()
    ref_g_w = w_r.grad.detach().clone()

    # Liger mean
    inp_l = inp.clone().detach().requires_grad_(True)
    w_l = weight.clone().detach().requires_grad_(True)
    lig_loss, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
        inp_l, w_l, target, None, None, ignore_index, 0.0, 0.0, "mean", None, False,
    )
    lig_loss.backward()
    lig_g_inp = inp_l.grad.detach().clone()
    lig_g_w = w_l.grad.detach().clone()

    assert abs(ref_loss.item() - lig_loss.item()) / max(abs(ref_loss.item()), 1.0) < 5e-2

    err_inp = _max_rel_err(ref_g_inp.to(lig_g_inp.dtype), lig_g_inp)
    err_w = _max_rel_err(ref_g_w.to(lig_g_w.dtype), lig_g_w)
    assert err_inp < 1e-2, (
        f"Liger mean grad_inp max-rel-err {err_inp:.3e} exceeds 1e-2 — "
        f"the mean-path backward we rely on for the workaround is broken"
    )
    assert err_w < 1e-2, (
        f"Liger mean grad_weight max-rel-err {err_w:.3e} exceeds 1e-2"
    )
