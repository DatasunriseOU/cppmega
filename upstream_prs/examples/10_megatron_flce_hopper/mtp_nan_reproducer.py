"""Reproducer: PR #3345 linear_cross_entropy + MTP process_mtp_loss path produces NaN.

Symptoms observed in our cppmega bench3 run (2026-04-14, NAM56R, H200):
    - Main head alone + native Hopper CE: grad_norm 84 -> 59 -> 364 (sane)
    - Main head + MTP both native Hopper CE: grad_norm = NaN every iter,
      OOM iter 4 (63.5 GiB CG private pool)
    - Vanilla MTP (vocab_parallel_cross_entropy) + native main head: sane
      -> isolates the MTP fuse_linear_cross_entropy branch as the culprit

This script reproduces the core defect **without Megatron** — it calls
``LinearCrossEntropy.apply`` (the autograd Function behind PR #3345) TWICE
per step against the **same** weight matrix (simulating main head +
MTP depth 0), then runs a single ``.backward()`` and checks ``weight.grad``
for NaN / Inf and checks that accumulation matches a reference
``F.linear + F.cross_entropy`` gradient.

What we expect to expose:
  1. If ``weight.grad`` is finite and ≈ sum of two per-call d_weights -> OK,
     the bug is elsewhere (MTP-specific wrapping).
  2. If ``weight.grad`` has NaN/Inf after step 2 -> PR #3345 kernel has a
     state-reset / stride-assumption defect when invoked multiple times
     per-step on a shared weight.
  3. If the kernel is called with ``reduction="none"`` and the returned
     tensor is transposed+contiguous (the LinearCrossEntropyModule.forward
     pattern), and the backward gradient arrives in that transposed layout,
     the kernel's ``dlogprobs.view(-1)`` may reinterpret in b-major order
     while the saved ``global_hidden`` is s-major — producing garbage
     d_weight. This script toggles `simulate_module_transpose=True` to
     exercise exactly that path.

Usage:
    CUDA_VISIBLE_DEVICES=0 python mtp_nan_reproducer.py

Exit codes:
    0  — both calls produce finite, ref-matching d_weight (no bug)
    1  — NaN/Inf in accumulated grad or gross ref mismatch (bug reproduced)
    77 — SKIP: cannot import megatron.core.transformer.linear_cross_entropy
         (pre-PR#3345 tree, or no CUDA, or cc != 9/10).

References:
    - PR #3345: https://github.com/NVIDIA/Megatron-LM/pull/3345
    - dev branch fuse branch: megatron/core/transformer/multi_token_prediction.py
      lines 664-681 (function process_mtp_loss, for-loop over mtp_num_layers)
    - Our gate shim: cppmega/megatron/mtp_native_hopper_ce.py
"""
from __future__ import annotations

import os
import sys
import traceback

SKIP = 77


def _skip(msg: str) -> int:
    print(f"SKIP: {msg}", file=sys.stderr)
    return SKIP


def main() -> int:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        return _skip(f"torch import failed: {exc}")

    if not torch.cuda.is_available():
        return _skip("CUDA not available")

    cc = torch.cuda.get_device_capability(0)
    if cc[0] not in (9, 10):
        return _skip(f"compute capability {cc} not supported by PR #3345 (needs 9 or 10)")

    try:
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropy
    except Exception as exc:
        return _skip(
            f"megatron.core.transformer.linear_cross_entropy not importable ({exc}); "
            "likely pre-PR#3345 tree."
        )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # NAM56R-ish shapes, scaled down for single-GPU repro.
    # seq_len * batch ~ 2048, hidden = 1024, vocab = 32000
    s, b, h, v = 512, 4, 1024, 32000
    num_tokens = s * b

    torch.manual_seed(0)

    # Shared output weight (the MambaModel.output_layer.weight in practice).
    weight = torch.randn(v, h, device=device, dtype=dtype, requires_grad=True) * 0.02

    # Two distinct hidden states: "main head" and "MTP depth 0".
    # In Megatron both are [s, b, h].
    hidden_main = torch.randn(s, b, h, device=device, dtype=dtype, requires_grad=True) * 0.1
    hidden_mtp0 = torch.randn(s, b, h, device=device, dtype=dtype, requires_grad=True) * 0.1

    # Labels: main in [b, s] per Megatron label convention.
    labels_main_bs = torch.randint(0, v, (b, s), device=device, dtype=torch.long)
    labels_mtp_bs = torch.randint(0, v, (b, s), device=device, dtype=torch.long)

    # ------- simulate LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss -------
    # (the exact wrapping that process_mtp_loss invokes with output_cross_entropy_loss=True)
    #
    #   labels_kernel = labels.transpose(0,1).contiguous()   -> [s, b]
    #   hidden_kernel = hidden                                -> [s, b, h]
    #   loss = LinearCrossEntropy.apply(hidden, weight, labels_kernel, None, "none", -100, False)
    #   loss = loss.view_as(labels_kernel).transpose(0,1).contiguous()   -> [b, s]
    def module_wrapped_forward(hidden_sbh, labels_bs):
        labels_sb = labels_bs.transpose(0, 1).contiguous()
        hidden_flat = hidden_sbh.reshape(-1, hidden_sbh.shape[-1])  # [s*b, h]
        labels_flat = labels_sb.reshape(-1)  # [s*b] -- *** ORDER: s-major! ***
        loss = LinearCrossEntropy.apply(
            hidden_flat, weight, labels_flat, None, "none", -100, False
        )
        # Reshape back -- matches LinearCrossEntropyModule wrapper.
        loss = loss.view_as(labels_sb).transpose(0, 1).contiguous()  # [b, s]
        return loss

    # Call 1: main head.
    loss_main = module_wrapped_forward(hidden_main, labels_main_bs)
    # Call 2: MTP depth 0 -- SAME shared weight.
    loss_mtp = module_wrapped_forward(hidden_mtp0, labels_mtp_bs)

    # Apply per-token loss mask (ones for simplicity) then a sum -- this is what
    # MTPLossAutoScaler + main reduce_loss does; simulate with mean.
    total = loss_main.mean() + 0.3 * loss_mtp.mean()  # 0.3 = mtp_loss_scaling_factor
    total.backward()

    g = weight.grad
    has_nan = bool(torch.isnan(g).any().item())
    has_inf = bool(torch.isinf(g).any().item())
    gmax = float(g.abs().max().item())
    print(f"after 2 calls: weight.grad max abs = {gmax:.4e}, has_nan={has_nan}, has_inf={has_inf}")

    # Reference: F.linear + F.cross_entropy, same per-call setup.
    weight_ref = weight.detach().clone().requires_grad_(True)

    def ref_forward(hidden_sbh, labels_bs):
        hidden_flat = hidden_sbh.reshape(-1, hidden_sbh.shape[-1])
        labels_sb = labels_bs.transpose(0, 1).contiguous()
        labels_flat = labels_sb.reshape(-1)
        logits = F.linear(hidden_flat.float(), weight_ref.float())
        logp = F.cross_entropy(logits, labels_flat, reduction="none", ignore_index=-100)
        return logp.view_as(labels_sb).transpose(0, 1).contiguous().to(hidden_sbh.dtype)

    hidden_main_ref = hidden_main.detach().clone().requires_grad_(True)
    hidden_mtp0_ref = hidden_mtp0.detach().clone().requires_grad_(True)
    loss_main_ref = ref_forward(hidden_main_ref, labels_main_bs)
    loss_mtp_ref = ref_forward(hidden_mtp0_ref, labels_mtp_bs)
    total_ref = loss_main_ref.mean() + 0.3 * loss_mtp_ref.mean()
    total_ref.backward()
    g_ref = weight_ref.grad
    rel_err = ((g.float() - g_ref.float()).norm() / g_ref.float().norm().clamp(min=1e-8)).item()
    print(f"ref relative error = {rel_err:.4e}")

    bug = has_nan or has_inf or (rel_err > 1e-2)
    if bug:
        print(
            "BUG REPRODUCED: shared-weight double-call through PR #3345 kernel produces "
            "NaN/Inf or >1% mismatch vs F.linear + F.cross_entropy reference."
        )
        return 1
    print("OK: two-call shared-weight path is clean in isolation; bug must be in MTP-specific "
          "wrapping (loss_mask * mtp_loss, MTPLossAutoScaler, or CUDA-graph capture).")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
