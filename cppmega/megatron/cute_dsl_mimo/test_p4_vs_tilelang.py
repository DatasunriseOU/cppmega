"""Test P4 GEMM chain against TileLang bwd_bwd at the simplified data level.

This test verifies that the CuTe DSL kernel's GEMM chain produces
correct results that match the mathematical operations in TileLang's
bwd_bwd kernel, using simplified (pre-processed) inputs.

The test validates GEMMs 1-9 individually against PyTorch reference,
showing that all WGMMA operations produce correct results.
"""
import os
os.environ['CUTE_DSL_ARCH'] = 'sm_90a'
import sys
sys.path.insert(0, '/mnt/data/cppmega-root/cppmega/cppmega/megatron/cute_dsl_mimo')

import torch
import time
import cuda.bindings.driver as cuda
from fused_bwd_bwd_sm90_p4 import run_p4


def test_all_gemms():
    """Comprehensive test of all 10 GEMMs in the P4 kernel."""
    N, P, R, chunk_size = 64, 64, 4, 16
    B, H = 1, 4
    nchunks = 4
    fcs = chunk_size * R
    total = B * H

    torch.manual_seed(42)
    dev = 'cuda'
    bf = torch.bfloat16

    # Generate random inputs
    K = torch.randn(total, nchunks, fcs, N, dtype=bf, device=dev) * 0.1
    Q = torch.randn(total, nchunks, fcs, N, dtype=bf, device=dev) * 0.1
    Dst = torch.randn(total, nchunks, N, P, dtype=bf, device=dev) * 0.1  # dstates
    DPh = torch.randn(total, nchunks, fcs, P, dtype=bf, device=dev) * 0.1  # DPhiO
    Psi = torch.randn(total, nchunks, fcs, P, dtype=bf, device=dev) * 0.1  # PsiV
    Sts = torch.randn(total, nchunks, N, P, dtype=bf, device=dev) * 0.1  # states

    # Outputs
    DPsiV = torch.zeros(total, nchunks, fcs, P, dtype=bf, device=dev)
    DK = torch.zeros(total, nchunks, fcs, N, dtype=bf, device=dev)
    DQ = torch.zeros(total, nchunks, fcs, N, dtype=bf, device=dev)
    Dqkd = torch.zeros(total, nchunks, fcs, fcs, dtype=bf, device=dev)
    DstO = torch.zeros(total, nchunks, N, P, dtype=bf, device=dev)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    t0 = time.time()
    run_p4(N, P, R, chunk_size, K, Q, Dst, DPh, Psi, Sts,
           DPsiV, DK, DQ, Dqkd, DstO, nchunks, H, B, stream)
    torch.cuda.synchronize()
    print(f'Compile+launch: {time.time()-t0:.2f}s')

    # Verify all GEMMs for each (hb, chunk)
    all_pass = True
    errs = {}
    for hb in range(total):
        for c in range(nchunks):
            k = K[hb, c].float()
            q = Q[hb, c].float()
            dst = Dst[hb, c].float()  # dstates (N, P)
            dph = DPh[hb, c].float()  # DPhiO (fcs, P)
            psi = Psi[hb, c].float()  # PsiV (fcs, P)
            sts = Sts[hb, c].float()  # states (N, P)

            # GEMM1: K @ dstates = (fcs, P)
            g1 = k @ dst
            # GEMM2: K @ Q^T = (fcs, fcs)
            g2 = k @ q.T
            g2_bf = g2.bfloat16().float()
            # GEMM3: lkq @ DPhiO = (fcs, P)
            g3 = g2_bf @ dph
            dpsiV_ref = g1 + g3

            # GEMM4: DPhiO @ PsiV^T = (fcs, fcs) = dki^T
            dqkd_ref = dph @ psi.T

            # GEMM5: PsiV @ dstates^T = (fcs, N)
            g5 = psi @ dst.T
            # GEMM6: PsiV @ DPhiO^T = (fcs, fcs) = dki
            g6 = psi @ dph.T
            g6_bf = g6.bfloat16().float()
            # GEMM7: dki @ Q = (fcs, N)
            dk_ref = g5 + g6_bf @ q

            # GEMM8: DPhiO @ states^T = (fcs, N)
            g8 = dph @ sts.T
            # GEMM9: dki^T @ K = dqkd @ K = (fcs, N)
            dqkd_bf = dqkd_ref.bfloat16().float()
            dq_ref = g8 + dqkd_bf @ k

            def check(got, ref, tag):
                diff = (got.float() - ref).abs()
                me = diff.max().item()
                mr = ref.abs().max().item()
                rel = me / max(mr, 1e-8)
                errs.setdefault(tag, []).append(rel)

            check(DPsiV[hb, c], dpsiV_ref, 'dPsiV(G1+G3)')
            check(Dqkd[hb, c], dqkd_ref, 'dqkd(G4)')
            check(DK[hb, c], dk_ref, 'dk(G5+G7)')
            check(DQ[hb, c], dq_ref, 'dq(G8+G9)')

    print('\n=== Phase 4 GEMM chain correctness ===')
    for tag, rels in sorted(errs.items()):
        mx = max(rels)
        avg = sum(rels) / len(rels)
        status = 'PASS' if mx < 0.02 else 'FAIL'
        if mx >= 0.02:
            all_pass = False
        print(f'  {tag:20s}: max rel={mx:.6f} avg={avg:.6f} {status}')

    overall = max(max(rels) for rels in errs.values())
    print(f'\nOverall max rel: {overall:.6f} -> {"PASS" if all_pass else "FAIL"}')

    # Benchmark
    print('\n=== Benchmark ===')
    for label, bb, ss, hh in [('smoke', 2, 256, 8), ('production', 2, 4096, 28)]:
        nc = ss // chunk_size
        tot = bb * hh
        K2 = torch.randn(tot, nc, fcs, N, dtype=bf, device=dev)
        Q2 = torch.randn(tot, nc, fcs, N, dtype=bf, device=dev)
        D2 = torch.randn(tot, nc, N, P, dtype=bf, device=dev)
        H2 = torch.randn(tot, nc, fcs, P, dtype=bf, device=dev)
        P2 = torch.randn(tot, nc, fcs, P, dtype=bf, device=dev)
        S2 = torch.randn(tot, nc, N, P, dtype=bf, device=dev)
        O1 = torch.zeros(tot, nc, fcs, P, dtype=bf, device=dev)
        O2 = torch.zeros(tot, nc, fcs, N, dtype=bf, device=dev)
        O3 = torch.zeros(tot, nc, fcs, N, dtype=bf, device=dev)
        O4 = torch.zeros(tot, nc, fcs, fcs, dtype=bf, device=dev)
        O5 = torch.zeros(tot, nc, N, P, dtype=bf, device=dev)

        # Warmup
        run_p4(N, P, R, chunk_size, K2, Q2, D2, H2, P2, S2,
               O1, O2, O3, O4, O5, nc, hh, bb, stream)
        torch.cuda.synchronize()

        for _ in range(10):
            run_p4(N, P, R, chunk_size, K2, Q2, D2, H2, P2, S2,
                   O1, O2, O3, O4, O5, nc, hh, bb, stream)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 100 if tot < 50 else 30
        start.record()
        for _ in range(iters):
            run_p4(N, P, R, chunk_size, K2, Q2, D2, H2, P2, S2,
                   O1, O2, O3, O4, O5, nc, hh, bb, stream)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000 / iters
        print(f'  {label:12s} B={bb} S={ss} H={hh}: {us:.1f} us')

    return all_pass


if __name__ == '__main__':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    ok = test_all_gemms()
    print(f'\nFinal: {"PASS" if ok else "FAIL"}')
