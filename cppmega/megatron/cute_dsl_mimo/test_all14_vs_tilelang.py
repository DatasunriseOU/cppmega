"""Test all bwd_bwd outputs: PyTorch epilogue vs TileLang reference.
Uses torch.allclose with rtol=1e-2, atol=1e-2 as the criterion.
"""
import os
os.environ['CUTE_DSL_ARCH'] = 'sm_90a'

import math
import torch

from cppmega.megatron.cute_dsl_mimo.full_bwd_bwd_epilogue import full_bwd_bwd_pytorch
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import (
    mamba_mimo_bwd_combined, mamba_mimo_bwd_fwd, mamba_mimo_bwd_bwd
)
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton


def test_all14(B=1, S=64, H=2, G=1, R=4, N=64, P=64, chunk_size=16):
    dev = 'cuda'
    dtype = torch.bfloat16
    rdim = N // 4
    nchunks = S // chunk_size

    torch.manual_seed(42)
    q = torch.randn(B, S, R, G, N, dtype=dtype, device=dev) * 0.1
    k = torch.randn(B, S, R, G, N, dtype=dtype, device=dev) * 0.1
    v = torch.randn(B, S, H, P, dtype=dtype, device=dev) * 0.1
    dout = torch.randn(B, S, H, P, dtype=dtype, device=dev) * 0.1
    q_bias = torch.randn(H, R, N, dtype=torch.float32, device=dev) * 0.01
    k_bias = torch.randn(H, R, N, dtype=torch.float32, device=dev) * 0.01
    mimo_v = torch.randn(H, R, P, dtype=torch.float32, device=dev) * 0.1
    mimo_o = torch.randn(H, R, P, dtype=torch.float32, device=dev) * 0.1
    angles = torch.randn(B, S, H, rdim, dtype=torch.float32, device=dev) * 0.5
    dt = torch.randn(B, H, S, dtype=torch.float32, device=dev).abs() * 0.1
    trap = torch.randn(B, H, S, dtype=dtype, device=dev) * 0.5
    dA = torch.randn(B, H, S, dtype=torch.float32, device=dev) * 0.05
    dA_cs, dA_cs_rev, segsum = compute_dacs_segsum_triton(dA, chunk_size)
    D = torch.randn(H, dtype=torch.float32, device=dev) * 0.1

    # Run TileLang bwd_fwd to get STATES and QK_DOT
    nchunks_v = math.ceil(S / chunk_size)
    states_cache = torch.empty(B, H, nchunks_v, N, P, dtype=dtype, device=dev)
    qk_dot_cache = torch.zeros(B, H, S, R, R, dtype=dtype, device=dev)
    dmimo_o_cache = torch.empty(B, H, R, P, dtype=torch.float32, device=dev)

    bwd_fwd_kernel = mamba_mimo_bwd_fwd(B, S, H, G, N, P, R,
                                         False, True, True,
                                         chunk_size, 4, 'bfloat16', 128, 0)
    bwd_fwd_kernel(dout, q, k, v, q_bias, k_bias, mimo_v, mimo_o,
                    dmimo_o_cache, states_cache,
                    None, None, None, None,
                    angles, dA_cs, dA_cs_rev, dt, trap, D,
                    qk_dot_cache, segsum)
    torch.cuda.synchronize()

    # Run TileLang bwd_bwd (raw)
    dk_tl = torch.empty(B, S*R, H, N, dtype=dtype, device=dev)
    dq_tl = torch.empty(B, S*R, H, N, dtype=dtype, device=dev)
    dv_tl = torch.empty_like(v)
    dmv_tl = torch.empty(B, H, R, P, dtype=torch.float32, device=dev)
    dD_tl = torch.empty(B, H, dtype=torch.float32, device=dev)
    dang_tl = torch.zeros(B, S, H, rdim, dtype=torch.float32, device=dev)
    dfac_tl = torch.zeros(B, H, S, dtype=torch.float32, device=dev)
    dgd_tl = torch.zeros(B, H, S, dtype=torch.float32, device=dev)
    ddA_tl = torch.zeros(B, H, S, dtype=torch.float32, device=dev)
    dSSdA_tl = torch.zeros(B, H, nchunks_v, chunk_size, chunk_size, dtype=torch.float32, device=dev)
    ddA_cs_rev_tl = torch.zeros(B, H, S, dtype=torch.float32, device=dev)
    ddA_cs_tl = torch.zeros(B, H, S, dtype=torch.float32, device=dev)

    bwd_bwd_kernel = mamba_mimo_bwd_bwd(B, S, H, G, N, P, R,
                                         False, True, True,
                                         chunk_size, 4, 'bfloat16', 256, 0)
    bwd_bwd_kernel(dout, q, k, v, q_bias, k_bias, mimo_v, mimo_o,
                    dk_tl, dv_tl, dmv_tl, states_cache, dq_tl,
                    None, None, angles, dA_cs, dA_cs_rev, dt, trap,
                    dfac_tl, dgd_tl, dang_tl, D, dD_tl,
                    qk_dot_cache, ddA_tl, dSSdA_tl, ddA_cs_rev_tl, ddA_cs_tl,
                    segsum)
    torch.cuda.synchronize()

    # Run our PyTorch epilogue
    our = full_bwd_bwd_pytorch(
        dout, q, k, v,
        q_bias, k_bias, mimo_v, mimo_o,
        angles, dA_cs, dA_cs_rev, dt, trap, D, segsum,
        states_cache, qk_dot_cache,
        chunk_size=chunk_size, R=R, rotary_dim_divisor=4,
    )
    torch.cuda.synchronize()

    print(f'\n=== B={B} S={S} H={H} G={G} R={R} N={N} P={P} cs={chunk_size} ===')
    
    def check(name, got, ref, rtol=1e-2, atol=1e-2):
        g, r = got.float(), ref.float()
        if g.shape != r.shape:
            print(f'  {name:16s}: SHAPE MISMATCH got={list(g.shape)} ref={list(r.shape)}')
            return False
        diff = (g - r).abs()
        me = diff.max().item()
        mr = r.abs().max().item()
        rel = me / max(mr, 1e-8)
        ok = torch.allclose(g, r, rtol=rtol, atol=atol)
        print(f'  {name:16s}: max_err={me:.6f} max_ref={mr:.4f} rel={rel:.6f} {"PASS" if ok else "FAIL"}')
        return ok

    results = {}
    results['DK'] = check('DK', our['DK'], dk_tl)
    results['DQ'] = check('DQ', our['DQ'], dq_tl)
    results['DV'] = check('DV', our['DV'], dv_tl)
    results['DMIMO_V'] = check('DMIMO_V', our['DMIMO_V'], dmv_tl)
    results['DDA_CS'] = check('DDA_CS', our['DDA_CS'], ddA_cs_tl)
    results['DDA_CS_REV'] = check('DDA_CS_REV', our['DDA_CS_REV'], ddA_cs_rev_tl)
    results['DFACTOR'] = check('DFACTOR', our['DFACTOR'], dfac_tl)
    results['DGAMMA_DIAG'] = check('DGAMMA_DIAG', our['DGAMMA_DIAG'], dgd_tl)
    results['DSSDA'] = check('DSSDA', our['DSSDA'], dSSdA_tl)
    results['DDA'] = check('DDA', our['DDA'], ddA_tl)
    results['DANGLES'] = check('DANGLES', our['DANGLES'], dang_tl)

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f'\nOverall: {n_pass}/{n_total} PASS')
    return all(results.values())


if __name__ == '__main__':
    ok1 = test_all14(B=1, S=64, H=2)
    ok2 = test_all14(B=1, S=256, H=4)
    ok3 = test_all14(B=2, S=128, H=8)
    print(f'\nFinal: S64={"OK" if ok1 else "FAIL"} S256={"OK" if ok2 else "FAIL"} S128={"OK" if ok3 else "FAIL"}')
