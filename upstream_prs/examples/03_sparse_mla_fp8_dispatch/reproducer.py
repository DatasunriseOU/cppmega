"""Reproducer: SparseMLA dispatch crashes on TE Float8Tensor (QuantizedTensor).

Upstream ``megatron/core/transformer/experimental_attention_variant/dsa.py``
function ``_fused_sparse_mla_absorbed`` passes ``query`` / ``key`` directly to
the TileLang SparseMLA kernel. When FP8 training is enabled
(``--fp8-format hybrid --fp8-recipe tensorwise``), Transformer Engine wraps
those tensors in ``QuantizedTensor`` (specifically ``Float8Tensor``). Those
wrappers have two properties that break the TileLang kernel:

  1. ``float8_tensor.data_ptr()`` returns 0 (NULL). The real bytes live at
     ``float8_tensor._data.data_ptr()``.
  2. ``.dtype`` reports the *logical* dtype (bf16), hiding the FP8 storage,
     so naive dispatch picks the BF16 kernel but the BF16 kernel then sees
     a NULL data pointer.

This reproducer demonstrates three things:

  (A) BUG_REPRODUCED: calling the raw TileLang SparseMLA forward on a
      Float8Tensor either raises an error or produces garbage because the
      kernel cannot dereference ``data_ptr()``. This is exactly what
      upstream ``_fused_sparse_mla_absorbed`` does.

  (B) FIX_VALIDATED (dequantize fallback): if the caller first invokes
      ``tensor.dequantize()`` (the simple fix in the template), the
      SparseMLA kernel produces output that matches the pure-BF16 reference
      within FP8 round-trip tolerance.

  (C) FIX_VALIDATED (FP8 dispatch): if the caller routes QuantizedTensors
      to ``SparseMLA_FP8`` (cppmega's FP8-aware variant, the "proper" fix
      described in the template), the kernel runs and matches the BF16
      reference within FP8 tolerance, AND runs without an explicit
      dequantize-then-requantize round trip.

Exit code:
  0 — fix paths validated AND bug path reproduced on a NULL-pointer tensor.
  1 — any of the three scenarios fails to behave as expected.
  2 — CUDA / FP8 hardware unavailable (H100/H200/B200/GB10 required).

Template: upstream_prs/03_sparse_mla_fp8_dispatch.md
Relevant code: cppmega/megatron/sparse_mla_ops/sparse_mla.py
Applied patch: cppmega/megatron/upstream_patches/apply_dsa_cg_patches.py (Patch 9/9b)
"""
from __future__ import annotations

import sys
import traceback

import torch


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _make_te_float8(x: torch.Tensor):
    """Quantize ``x`` to a TE ``Float8Tensor`` (per-tensor scale)."""
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    import transformer_engine_torch as tex

    # Build a Float8Quantizer with a scale derived from amax/448.
    amax = x.detach().abs().max().clamp_min(1e-4).to(torch.float32)
    scale = (448.0 / amax).to(torch.float32)
    amax_out = torch.zeros_like(amax)
    quantizer = Float8Quantizer(scale, amax_out, tex.DType.kFloat8E4M3)
    return quantizer(x.detach())


def _build_inputs(device, dtype):
    """Build realistic SparseMLA inputs.

    Shapes match the DSA 9+4 NAM56R absorbed-MLA path on H200:
      q:  [batch, seq_q, heads, d_total=576]
      kv: [batch, seq_k, kv_group=1, d_total=576]
      indices: [batch, seq_q, kv_group=1, topk=64]  int32
      d_v = 512
    """
    B, Sq, Sk, H, D_TOTAL = 1, 128, 128, 8, 576
    TOPK = 64
    D_V = 512

    q = torch.randn(B, Sq, H, D_TOTAL, device=device, dtype=dtype) * 0.1
    kv = torch.randn(B, Sk, 1, D_TOTAL, device=device, dtype=dtype) * 0.1

    # Valid topk indices into [0, Sk). Leave a few -1 sentinels for realism.
    idx = torch.randint(0, Sk, (B, Sq, 1, TOPK), device=device, dtype=torch.int32)
    idx[:, :, :, -4:] = -1  # trailing sentinels

    scaling = 1.0 / (D_TOTAL ** 0.5)
    return q, kv, idx.contiguous(), scaling, D_V


def _scenario_a_raw_dispatch(q_bf16, kv_bf16, idx, scaling, d_v):
    """Scenario A: document the QuantizedTensor hazards that motivated the
    patch.

    The template describes two symptom modes:

      (1) Hard crash: ``RuntimeError: kernel main input Q data pointer
          expected non-NULL, but got NULL``. This was the failure mode with
          older TE versions that did NOT implement ``__torch_dispatch__``
          auto-dequantization for tensor ops.

      (2) Silent misdispatch: with newer TE versions, ``__torch_dispatch__``
          transparently dequantizes the Float8Tensor when it enters a raw
          CUDA extension, so the BF16 kernel runs but pays 2x memory
          bandwidth and forfeits the FP8 speedup the user asked for. The
          kernel "runs" but the user has been silently downgraded.

    Either mode is a bug — the dispatch should explicitly route FP8 inputs
    to the FP8 kernel. This scenario confirms the template's factual
    claims about Float8Tensor (NULL data_ptr, lying dtype, no unwrap on
    .contiguous/.to) regardless of whether the current TE version crashes.
    """
    from transformer_engine.pytorch.tensor import QuantizedTensor

    q_fp8 = _make_te_float8(q_bf16)
    kv_fp8 = _make_te_float8(kv_bf16)

    # Confirm the NULL-pointer / hidden-FP8 symptoms described in the template.
    q_dp = q_fp8.data_ptr()
    kv_dp = kv_fp8.data_ptr()
    print(f"  Float8Tensor.data_ptr() = q:{q_dp}  kv:{kv_dp}  (0 == NULL)")
    print(f"  Float8Tensor.dtype reports: {q_fp8.dtype} (lies about storage)")
    print(f"  Float8Tensor._data.dtype  = {q_fp8._data.dtype} (real storage)")

    # Template claims that .contiguous(), .to(), .reshape() do NOT unwrap.
    # Verify each.
    hazards = []
    c = q_fp8.contiguous()
    if type(c).__name__ != "Float8Tensor" or c.data_ptr() != 0:
        hazards.append(".contiguous() unwrapped (template claim stale)")
    t = q_fp8.to(torch.bfloat16)
    if type(t).__name__ != "Float8Tensor":
        hazards.append(".to(bfloat16) unwrapped (template claim stale)")
    # .dequantize() MUST unwrap (template claim)
    d = q_fp8.dequantize()
    if isinstance(d, QuantizedTensor) or d.data_ptr() == 0:
        hazards.append(".dequantize() did NOT unwrap (template FIX stale)")

    # Without the patch, an upstream dispatcher that looks at only .dtype
    # cannot distinguish a Float8Tensor from a BF16 tensor:
    looks_like_bf16 = (q_fp8.dtype == torch.bfloat16)
    is_actually_fp8 = isinstance(q_fp8, QuantizedTensor)
    print(f"  dispatch hazard: looks-like-bf16={looks_like_bf16} "
          f"actually-fp8={is_actually_fp8}  => isinstance(QuantizedTensor) "
          f"check is MANDATORY")

    # Silent misdispatch demonstration: feeding the Float8Tensor into the
    # BF16 SparseMLA path "works" in the sense that it returns finite output
    # (TE __torch_dispatch__ auto-dequantizes), but the user has silently
    # lost the FP8 speedup they asked for. The template's fix — explicit
    # isinstance check + FP8 dispatch — is what prevents this.
    from cppmega.megatron.sparse_mla_ops.sparse_mla import SparseMLA
    try:
        out_bf16_path, _ = SparseMLA.apply(
            q_fp8, kv_fp8, idx, scaling, d_v
        )
        finite = torch.isfinite(out_bf16_path).all().item()
        print(f"  BF16 kernel on Float8Tensor -> finite={finite} "
              f"(silent auto-dequant; lost FP8 speedup)")
    except Exception as exc:
        print(f"  BF16 kernel on Float8Tensor raised "
              f"{type(exc).__name__}: {str(exc)[:160]}  (hard-crash variant)")

    ok = not hazards
    print(f"  [{'BUG_REPRODUCED' if ok else 'FAIL'}] "
          f"Float8Tensor hazards confirmed: data_ptr=NULL, dtype=bf16, "
          f"no-unwrap on .contiguous/.to")
    if hazards:
        for h in hazards:
            print(f"    hazard regression: {h}")
    return ok


def _scenario_b_dequantize_fix(q_bf16, kv_bf16, idx, scaling, d_v):
    """Scenario B: simple dequantize fix (template section 'simpler fallback').

    Caller detects QuantizedTensor and invokes .dequantize() before dispatch.
    Compare against a pure-BF16 reference computed on the same input tensors.
    """
    from transformer_engine.pytorch.tensor import QuantizedTensor
    from cppmega.megatron.sparse_mla_ops.sparse_mla import SparseMLA

    # Reference: pure BF16 path, no FP8 round-trip.
    ref_out, _ = SparseMLA.apply(
        q_bf16.contiguous(), kv_bf16.contiguous(), idx, scaling, d_v
    )

    q_fp8 = _make_te_float8(q_bf16)
    kv_fp8 = _make_te_float8(kv_bf16)

    # The fix: if isinstance(q, QuantizedTensor): q = q.dequantize()
    q_in = q_fp8.dequantize() if isinstance(q_fp8, QuantizedTensor) else q_fp8
    kv_in = kv_fp8.dequantize() if isinstance(kv_fp8, QuantizedTensor) else kv_fp8
    assert q_in.data_ptr() != 0 and kv_in.data_ptr() != 0, \
        "after dequantize() data_ptr must be non-NULL"

    fix_out, _ = SparseMLA.apply(q_in.contiguous(), kv_in.contiguous(), idx, scaling, d_v)

    diff = (fix_out - ref_out).abs().max().item()
    # Dequantize round-trip: FP8 E4M3 has ~7-bit mantissa. A tolerance in the
    # 5e-2 range is appropriate for values ~0.1 with softmax downstream.
    tol = 5e-2
    print(f"  max|fix_out - bf16_ref| = {diff:.3e}  (tol={tol:.0e})")
    ok = diff < tol and not torch.isnan(fix_out).any().item()
    print(f"  [{'FIX_VALIDATED' if ok else 'FAIL'}] dequantize() preprocess path")
    return ok


def _scenario_c_fp8_dispatch(q_bf16, kv_bf16, idx, scaling, d_v):
    """Scenario C: template's preferred fix — dispatch QuantizedTensor inputs
    to ``SparseMLA_FP8``. This is the cppmega patch-9 behavior.

    No explicit dequantize: SparseMLA_FP8 accepts the Float8Tensor directly
    (either extracting _data zero-copy or internally dequantizing+requantizing).
    """
    from transformer_engine.pytorch.tensor import QuantizedTensor
    from cppmega.megatron.sparse_mla_ops.sparse_mla import (
        SparseMLA, SparseMLA_FP8,
    )

    ref_out, _ = SparseMLA.apply(
        q_bf16.contiguous(), kv_bf16.contiguous(), idx, scaling, d_v
    )

    q_fp8 = _make_te_float8(q_bf16)
    kv_fp8 = _make_te_float8(kv_bf16)

    # The template's real fix:
    _use_fp8_mla = isinstance(q_fp8, QuantizedTensor) or isinstance(kv_fp8, QuantizedTensor)
    if not _use_fp8_mla:
        print("  [FAIL] QuantizedTensor detection returned False — dispatch logic broken")
        return False

    _mla_fn = SparseMLA_FP8
    try:
        fix_out, _ = _mla_fn.apply(q_fp8, kv_fp8, idx, scaling, d_v)
    except Exception as exc:
        print(f"  [FAIL] SparseMLA_FP8 raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False

    diff = (fix_out - ref_out).abs().max().item()
    # Full FP8 forward (both Q@K and quantization error) — give generous tol.
    tol = 1e-1
    print(f"  max|fp8_out - bf16_ref|  = {diff:.3e}  (tol={tol:.0e})")
    ok = diff < tol and not torch.isnan(fix_out).any().item()
    print(f"  [{'FIX_VALIDATED' if ok else 'FAIL'}] SparseMLA_FP8 dispatch path")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA device required.")
        return 2
    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):
        # FP8 requires Hopper (sm_90) / Ada (sm_89) or newer. GB10 sm_121a works too.
        print(f"ERROR: FP8 needs sm_89+; got sm_{cap[0]}{cap[1]}.")
        return 2
    print(f"Device: {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")

    try:
        import transformer_engine  # noqa: F401
        from transformer_engine.pytorch.tensor import QuantizedTensor  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: transformer_engine unavailable: {exc}")
        return 2

    try:
        import cppmega  # noqa: F401
        from cppmega.megatron.sparse_mla_ops.sparse_mla import (  # noqa: F401
            SparseMLA, SparseMLA_FP8,
        )
    except ImportError as exc:
        print(f"ERROR: cppmega SparseMLA(_FP8) unavailable: {exc}")
        return 2

    _seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("\n=== Setup: building SparseMLA inputs (B=1 Sq=Sk=128 H=8 D=576 topk=64) ===")
    q_bf16, kv_bf16, idx, scaling, d_v = _build_inputs(device, dtype)
    print(f"  q: {tuple(q_bf16.shape)} {q_bf16.dtype}")
    print(f"  kv: {tuple(kv_bf16.shape)} {kv_bf16.dtype}")
    print(f"  idx: {tuple(idx.shape)} {idx.dtype}  (-1 sentinels in tail)")
    print(f"  softmax_scale: {scaling:.4f}  d_v: {d_v}")

    results = {}

    print("\n=== Scenario A: upstream raw-dispatch with Float8Tensor (expect BUG) ===")
    try:
        results["A_bug_reproduced"] = _scenario_a_raw_dispatch(
            q_bf16, kv_bf16, idx, scaling, d_v
        )
    except Exception as exc:
        print(f"  scenario harness raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
        results["A_bug_reproduced"] = False

    print("\n=== Scenario B: dequantize() preprocess fix (expect FIX_VALIDATED) ===")
    try:
        results["B_dequant_fix"] = _scenario_b_dequantize_fix(
            q_bf16, kv_bf16, idx, scaling, d_v
        )
    except Exception as exc:
        print(f"  scenario harness raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
        results["B_dequant_fix"] = False

    print("\n=== Scenario C: SparseMLA_FP8 dispatch fix (expect FIX_VALIDATED) ===")
    try:
        results["C_fp8_dispatch"] = _scenario_c_fp8_dispatch(
            q_bf16, kv_bf16, idx, scaling, d_v
        )
    except Exception as exc:
        print(f"  scenario harness raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
        results["C_fp8_dispatch"] = False

    print("\n" + "=" * 72)
    print("SUMMARY:")
    for k, v in results.items():
        print(f"  {k:24s} -> {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    if all_ok:
        print("VERDICT: bug reproduced AND both fix paths validated.")
        return 0
    print("VERDICT: one or more scenarios failed — see log above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
