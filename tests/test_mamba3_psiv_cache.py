"""Correctness scaffolding for Mamba3 MIMO P2 PsiV cache.

Status: scaffold only (2026-04-14). Every test is `skip`-gated until the
Phase A/B/C implementation lands per
`docs/mamba3_mimo_p2_psiv_cache_design.md`.

What these tests will verify (in order):

  1. `test_estimate_cache_bytes_matches_design_doc`:
       arithmetic sanity — PsiV cache size per layer matches §6 of design.
       This one runs today (no GPU, no impl needed).

  2. `test_gate_off_is_silent`:
       with `CPPMEGA_MAMBA3_P2_PSIV_CACHE` unset, module does nothing +
       raises no side-effects.

  3. `test_phase_a_precompute_matches_reference`:
       Phase-A materializer computes `V * MIMO_V` with the same dtype cast the
       TileLang kernel uses.

  4. `test_gate_on_without_kernel_impl_raises`:
       with gate ON but no TileLang integration landed, kernel wrappers MUST
       raise NotImplementedError (per `feedback_no_silent_fallbacks.md`).

  5. `test_phase_a_matches_baseline` (SKIP until kernel path exists):
       call upstream mamba3_mimo with materialised PsiV vs baseline;
       outputs bit-identical (or ≤1e-3 rel_err from float reorder).

  5. `test_fwd_bwd_with_cache_matches_baseline` (SKIP until Phase B/C):
       same but through the full cppmega_tilelang_mimo_combined path,
       checking all 14 gradient tensors with rel_err < 0.02 (the P1 /
       TMA-layout-fix tolerance).

  6. `test_cuda_graph_capture_survives` (SKIP until Phase C):
       capture a 2-step graph with the cache enabled, replay, check
       bit-identical results to eager.

Run:
    python -m pytest tests/test_mamba3_psiv_cache.py -v
"""

from __future__ import annotations

import os

import pytest
import torch


# ---------------------------------------------------------------------------
# Test 1 — pure arithmetic, runs everywhere
# ---------------------------------------------------------------------------

def test_estimate_cache_bytes_matches_design_doc():
    """Design §6: 64 MiB / sample / layer at NAM56R shape (bf16).

    If this changes, the design doc's memory-overhead column is stale.
    """
    from cppmega.megatron.mamba3_psiv_cache import estimate_cache_bytes

    # NAM56R shape per plan.md / reference_nam56r_*:
    #   B=1 (per-sample), S=8192, H=16, R=4, P=64, bf16=2B
    bytes_per_sample = estimate_cache_bytes(
        batch=1, seqlen=8192, nheads=16, rank=4, headdim_v=64,
        dtype=torch.bfloat16, num_layers=1,
    )
    # 1 * 8192 * 16 * 4 * 64 * 2 = 67_108_864 bytes = 64 MiB exactly
    expected = 8192 * 16 * 4 * 64 * 2
    assert bytes_per_sample == expected, (
        f"Cache size drift: got {bytes_per_sample}, expected {expected}. "
        "Update docs/mamba3_mimo_p2_psiv_cache_design.md §6."
    )


def test_estimate_state_checkpoint_bytes_matches_p3_doc():
    from cppmega.megatron.mamba3_psiv_cache import estimate_state_checkpoint_bytes

    bytes_per_sample = estimate_state_checkpoint_bytes(
        batch=1,
        seqlen=8192,
        nheads=16,
        dstate=64,
        headdim_v=64,
        dtype=torch.float32,
        chunk_size=32,
        num_layers=1,
    )
    # B * H * (S/chunk) * N * P * fp32
    assert bytes_per_sample == 1 * 16 * 256 * 64 * 64 * 4


# ---------------------------------------------------------------------------
# Test 2 — gate-off is silent (import OK, is_enabled=False)
# ---------------------------------------------------------------------------

def test_gate_off_is_silent(monkeypatch):
    """With the env gate unset, module import + is_enabled() must be safe."""
    monkeypatch.delenv("CPPMEGA_MAMBA3_P2_PSIV_CACHE", raising=False)
    # Re-import to pick up the env var change — importlib.reload guarantees
    # `is_enabled()` re-reads the environment.
    import importlib

    import cppmega.megatron.mamba3_psiv_cache as mod
    importlib.reload(mod)
    assert mod.is_enabled() is False


# ---------------------------------------------------------------------------
# Test 3 — Phase-A materialization helper
# ---------------------------------------------------------------------------

def test_phase_a_precompute_matches_reference():
    from cppmega.megatron.mamba3_psiv_cache import precompute_psi_v

    V = torch.arange(2 * 3 * 4 * 5, dtype=torch.bfloat16).reshape(2, 3, 4, 5)
    mimo_v = torch.arange(4 * 2 * 5, dtype=torch.float32).reshape(4, 2, 5) / 10.0
    got = precompute_psi_v(V, mimo_v)
    expected = V.unsqueeze(3) * mimo_v.to(dtype=V.dtype).unsqueeze(0).unsqueeze(0)

    assert got.shape == (2, 3, 4, 2, 5)
    assert got.dtype == V.dtype
    assert got.is_contiguous()
    torch.testing.assert_close(got, expected)


def test_psiv_cache_pool_reuses_released_tensor():
    from cppmega.megatron.mamba3_psiv_cache import PsiVCachePool

    pool = PsiVCachePool()
    first = pool.acquire(1, 8, 2, 4, 4, torch.bfloat16, torch.device("cpu"))
    first_id = id(first)
    pool.release(first)
    second = pool.acquire(1, 8, 2, 4, 4, torch.bfloat16, torch.device("cpu"))
    assert id(second) == first_id


def test_patch_site_probe_reads_static_source(tmp_path):
    from cppmega.megatron.upstream_patches.apply_mamba3_p2_psiv_patches import probe_patch_sites

    files = {
        "mamba3_mimo_fwd.py": "PsiV_frag = T.alloc_fragment\n",
        "mamba3_mimo_fwd_varlen.py": "PsiV_frag = T.alloc_fragment\n",
        "mamba3_mimo_bwd.py": (
            "PsiV_frag = T.alloc_fragment\n"
            "T.gemm(q_shared, dPhiO_scaled_frag, dstates_frag\n"
        ),
        "mamba3_mimo_bwd_varlen.py": (
            "PsiV_frag = T.alloc_fragment\n"
            "T.gemm(q_shared, dPhiO_scaled_frag, dstates_frag\n"
        ),
        "mamba3_mimo.py": "class _Mamba3Function:\n    pass\nctx.save_for_backward()\n",
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)

    result = probe_patch_sites(tmp_path)
    assert result.ready_for_patch_skeleton is True
    assert result.fwd_psiv_materializations == 2
    assert result.bwd_psiv_materializations == 2
    assert result.bwd_dstates_updates == 2


# ---------------------------------------------------------------------------
# Test 4 — gate-on-without-kernel-impl raises (primary safety contract)
# ---------------------------------------------------------------------------

def test_gate_on_without_kernel_impl_raises(monkeypatch):
    """If someone flips the gate today, kernel entrypoints must crash loudly."""
    monkeypatch.setenv("CPPMEGA_MAMBA3_P2_PSIV_CACHE", "1")
    import importlib

    import cppmega.megatron.mamba3_psiv_cache as mod
    importlib.reload(mod)
    assert mod.is_enabled() is True

    # Phase B/C wrappers
    with pytest.raises(NotImplementedError):
        mod.forward_with_cache(lambda *a, **kw: None)
    with pytest.raises(NotImplementedError):
        mod.backward_fwd_with_cache(lambda *a, **kw: None)
    with pytest.raises(NotImplementedError):
        mod.backward_bwd_with_cache(lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# Test 5 — Phase A reproducer (skipped until kernel path exists)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for Phase A correctness test.",
)
@pytest.mark.skip(reason="Phase A not yet implemented — see design doc §9.")
def test_phase_a_matches_baseline():
    """Reproducer: `y_materialised == y_baseline` element-wise.

    Plan (once Phase A lands):
      1. Draw random V, mimo_v, Q, K, ... at NAM56R shape.
      2. Call baseline `mamba_mimo_fwd(V, mimo_v, ...)`.
      3. Call `psi_v = precompute_psi_v(V, mimo_v)`, then a modified kernel
         path that takes psi_v (materialised V surrogate).
      4. Assert output tensors match within bf16-noise bound
         (rel_err < 1e-3, bad_frac < 1e-4 at rtol=1e-2 atol=1e-2).

    This is the first go/no-go gate for P2 — if Phase A wins, commit to
    kernel work. If not, archive per §14 of the design doc.
    """


# ---------------------------------------------------------------------------
# Test 5 — Full fwd+bwd correctness with cache (skipped until Phase B/C)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for end-to-end correctness test.",
)
@pytest.mark.skip(reason="Phase B/C not yet implemented — see design doc §9.")
def test_fwd_bwd_with_cache_matches_baseline():
    """End-to-end: cppmega_tilelang_mimo_combined with cache vs without.

    Plan (once Phase C lands):
      1. Same random NAM56R inputs, requires_grad=True.
      2. Two runs:
         - baseline: CPPMEGA_MAMBA3_P2_PSIV_CACHE unset, run fwd+bwd, grab
           all 14 gradient tensors.
         - cached: CPPMEGA_MAMBA3_P2_PSIV_CACHE=1, apply patch, run fwd+bwd.
      3. Compare all 14 gradients element-wise. Pass iff
         stable_max_rel_err < 0.02 AND bad_frac(rtol=0.1, atol=0.1) < 0.001
         for every tensor. Criterion matches P1 notes §Backward combined.
    """


# ---------------------------------------------------------------------------
# Test 6 — CUDA graph compatibility (skipped until Phase C)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for CUDA graph test.",
)
@pytest.mark.skip(reason="Phase C not yet implemented — see design doc §11.")
def test_cuda_graph_capture_survives():
    """Phase-C risk item: PsiV cache alloc must live inside the captured pool.

    Plan:
      1. Warm pool outside capture.
      2. Capture 2-step graph with cache acquire/release in both steps.
      3. Replay 4×. Assert outputs identical to eager run (all 4 replays).
      4. Assert no hidden `torch.empty` allocs during replay (instrument
         with `torch.cuda.memory_allocated()` snapshots).
    """


# ---------------------------------------------------------------------------
# Test 7 — memory budget check (skipped until Phase B)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Phase B not yet implemented — memory accounting pending.")
def test_memory_budget_within_headroom():
    """NAM56R MBS=8 peak ~132 GiB, +5.6 GiB cache = ~138 GiB vs 141 GiB cap.

    Once Phase B lands, run one iteration at NAM56R shape with cache ON
    and assert `torch.cuda.max_memory_reserved() < 140 * 2**30`. If this
    fails, P2 is blocked on MBS=10 (bench3 record config) unless we
    quantize the cache per design doc §6.
    """
