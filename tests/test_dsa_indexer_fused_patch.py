"""Correctness test for the DSA indexer fused per-head accumulation patch.

Verifies :func:`compute_index_scores_fused_bf16` matches the upstream
Megatron BF16 ``_compute_index_scores`` einsum implementation to within
FP32 associative-reorder tolerance.

Run on GB10 (or any CUDA GPU) with a small shape; no Megatron checkout
required — this test inlines the upstream reference.
"""

from __future__ import annotations

import torch

from cppmega.megatron.dsa_indexer_fused_patch import compute_index_scores_fused_bf16


def _upstream_reference(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    use_relu: bool = True,
) -> torch.Tensor:
    """Byte-identical clone of upstream Megatron ``_compute_index_scores``."""
    index_scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    if use_relu:
        index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    return index_scores.transpose(0, 1)  # [b, sq, sk]


def test_fused_matches_reference_bf16_relu():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sq, sk, b, h, d = 128, 128, 2, 8, 64

    q = torch.randn(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(sq, b, h, dtype=torch.bfloat16, device=device)

    ref = _upstream_reference(q, w, k, use_relu=True)
    fused = compute_index_scores_fused_bf16(q, w, k, use_relu=True)

    assert ref.shape == fused.shape == (b, sq, sk)
    assert ref.dtype == fused.dtype == torch.float32

    # FP32 associative reorder tolerance: per-head accum vs fused-sum over h.
    abs_err = (ref - fused).abs().max().item()
    ref_abs = ref.abs().max().item()
    rel_err = abs_err / max(ref_abs, 1e-6)
    print(f"relu=True abs_err={abs_err:.3e} rel_err={rel_err:.3e}")
    assert rel_err < 1e-3, f"rel_err {rel_err} too high"


def test_fused_matches_reference_bf16_no_relu():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sq, sk, b, h, d = 256, 256, 1, 16, 64

    q = torch.randn(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(sq, b, h, dtype=torch.bfloat16, device=device)

    ref = _upstream_reference(q, w, k, use_relu=False)
    fused = compute_index_scores_fused_bf16(q, w, k, use_relu=False)

    abs_err = (ref - fused).abs().max().item()
    rel_err = abs_err / max(ref.abs().max().item(), 1e-6)
    print(f"relu=False abs_err={abs_err:.3e} rel_err={rel_err:.3e}")
    assert rel_err < 1e-3, f"rel_err {rel_err} too high"


def test_fused_nam56r_shape():
    """Test a realistic NAM56R indexer shape: h=32 d=64 sq=sk=4096 b=1."""
    if not torch.cuda.is_available():
        return
    torch.manual_seed(2)
    device = torch.device("cuda")
    sq, sk, b, h, d = 4096, 4096, 1, 32, 64

    q = torch.randn(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(sq, b, h, dtype=torch.bfloat16, device=device)

    ref = _upstream_reference(q, w, k, use_relu=True)
    fused = compute_index_scores_fused_bf16(q, w, k, use_relu=True)

    abs_err = (ref - fused).abs().max().item()
    rel_err = abs_err / max(ref.abs().max().item(), 1e-6)
    print(f"nam56r-shape abs_err={abs_err:.3e} rel_err={rel_err:.3e}")
    assert rel_err < 5e-3, f"rel_err {rel_err} too high"


if __name__ == "__main__":
    test_fused_matches_reference_bf16_relu()
    test_fused_matches_reference_bf16_no_relu()
    test_fused_nam56r_shape()
    print("All parity tests passed.")
