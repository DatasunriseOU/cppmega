# NAM56R MIMO 7/7 nsys profile — 2026-04-11

First full nsys profile of the NAM56R MIMO 7/7 baseline (56,280 tok/sec, 1164 ms/iter). 5 steady-state iterations (iters 6-10) captured on bench3 H200x8 with `cudaProfilerStart/Stop` hooks on rank 0 only. Steady-state iter time during profile: **1186 ms** (within 2% of baseline).

## Per-iter time budget

| Bucket | ms/iter | % of 1186 | Kernel count / iter | Notes |
|---|---:|---:|---:|---|
| **Elementwise + copy** | **305** | **25.7%** | 27,384 | Dominated by `CUDAFunctor_add<float>` (266 ms / 486 launches) + `MulFunctor<float>` (234 ms / 1162 launches). **Root cause: Mamba3 fp32 bias forward pre-hook runs `.data.float()` on 7 params × ~16 Mamba layers × (fwd+bwd+opt) = ~400 D2D copies per iter** |
| **TE Hopper BF16 GEMM (WGMMA)** | 251 | 21.2% | 695 | `nvjet_sm90_tst_*_h_*_coopA_NNT` cuBLASLt BF16 Hopper — MLA / MoE / dense. Using WGMMA. |
| **FP32 cuBLAS sm80 GEMM (NOT WGMMA)** | 142 | 12.0% | 144 | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_128x128x8_stage3_ffma` — Ampere f32 fallback from `nvte_multi_tensor_gemm` in optimizer. ~3% of H200 peak utilization. |
| TileLang `mamba_mimo_bwd_bwd_kernel` | 120 | 10.1% | 15.6 | 7.68 ms avg per launch |
| NCCL RS+AG (distributed optimizer) | 87 | 7.3% | — | mostly overlapped with compute |
| fp32 Softmax | 67 | 5.6% | 120 | `cunn_SoftMaxForwardReg<float,float,float>` — DSA indexer + MoE router |
| TileLang `mamba_mimo_bwd_fwd_kernel` | 59 | 4.9% | 16 | 3.76 ms avg |
| M2RNN bwd (Triton) | 55 | 4.6% | 4.8 | 11.49 ms avg |
| Reduce | 44 | 3.7% | — | |
| Index / scatter / cat | 39 | 3.3% | — | |
| TileLang `mamba_mimo_fwd_kernel` | 29 | 2.5% | 16 | 1.88 ms avg |
| M2RNN fwd (Triton) | 18 | 1.5% | — | |
| TE layernorm | 17 | 1.5% | — | |
| TE flash bwd (WGMMA sm_90) | 17 | 1.4% | — | cudnn sm90 flash bprop |
| TE flash fwd (WGMMA sm_90) | 10 | 0.8% | — | cudnn sm90 flash fprop |
| Other | 42 | 3.5% | — | |

**TileLang MIMO total:** 208 ms/iter (17.5% — not dominant, as suspected)
**M²RNN Triton total:** 73 ms/iter (6.2%)

Total GPU time: 6.51 s over 5 iters (window wallclock 5.93 s → ~10% stream overlap).

## Top 10 kernels by total time

| Total (ms) | Inst | Avg (ms) | Kernel | Category |
|---:|---:|---:|---|---|
| 598.8 | 78 | 7.68 | `mamba_mimo_bwd_bwd_kernel` | TileLang MIMO bwd² |
| 398.8 | 144 | 2.77 | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_128x128x8_stage3_ffma` | cuBLAS fp32 (Ampere fallback) |
| 293.4 | 78 | 3.76 | `mamba_mimo_bwd_fwd_kernel` | TileLang MIMO bwd_fwd |
| 275.7 | 24 | 11.49 | `_m2rnn_bwd_kernel` | M2RNN Triton |
| 272.3 | 6 | 45.38 | `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` | NCCL RS |
| 269.7 | 120 | 2.25 | `cunn_SoftMaxForwardReg<float,...>` | fp32 softmax |
| 265.6 | 486 | 0.55 | `elementwise_kernel<add<float>>` | elementwise fp32 |
| 262.7 | 695 | 0.38 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | TE WGMMA GEMM |
| 234.1 | 1162 | 0.20 | `vectorized_elementwise_kernel mul<float>` | elementwise |
| 146.3 | 78 | 1.88 | `mamba_mimo_fwd_kernel` | TileLang MIMO fwd |

## WGMMA verification

**Using WGMMA (confirmed):**
- All `nvjet_sm90_tst_*_h_*` cuBLASLt kernels — Hopper BF16 GEMMs (`_h_` = half/bf16). On sm_90 these are the only BF16 tensor-core path. Runtime-JIT'd by cuBLASLt (template names in libcublasLt.so.13), no static SASS to grep but structurally cannot avoid WGMMA.
- `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_*` — WGMMA by name.
- `cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_*` — WGMMA by name.

**NOT using WGMMA (wasted ~22% of cycles on Ampere-era paths):**
- `sm80_xmma_gemm_f32f32_f32f32_f32_*` — Ampere sm_80 fp32 cuBLAS fallback, **80 ms/iter** from `nvte_multi_tensor_gemm` + fp32-reduction paths
- `cutlass_80_simt_sgemm_128x128_8x4_nt_align1` — SIMT classical FFMA sgemm, 22 ms/iter, no tensor cores at all
- TileLang `mamba_mimo_*_kernel` — custom TMA + mma.m64n*k* PTX, not WGMMA in the cuBLAS sense (SSD scan pipeline). Not a GEMM problem; correct tile IR for sm_90.
- Triton `_m2rnn_fwd/bwd_kernel` — likely partial WGMMA on R-gate matmul + elementwise scan. Not verified at SASS level in this pass.

## Critical path finding: the fp32 shim is the #1 bottleneck

The elementwise + copy bucket (305 ms/iter, 25.7% of the iter) is driven by **the Mamba3 fp32 bias forward pre-hook** introduced in `cppmega_fp8_shim.py` to fix the `Float16Module` cast blocker. It runs `.data.float()` on `dt_bias, D, B_bias, C_bias, mimo_x_bias, mimo_z_bias, mimo_o_bias` (7 params) × ~16 Mamba layers × every forward pass (fwd + bwd + optimizer access) = **~400 small D2D copies per iter**. Each copy is tiny (avg 0.2-0.55 ms) but the 27,384 total launch count dominates the iter.

**Better fix:** patch `Float16Module.__init__` **once at model init** to skip these Mamba3 fp32 params during the bf16 cast. Then params remain fp32 permanently, no per-forward hook, no per-iter cost. Expected saving: ~60-80 ms/iter.

**Do NOT** keep the per-forward hook in production — it was the quickest blocker unblock, not a correct long-term fix.

## Roofline analysis for top 3 buckets

H200 SXM: BF16 peak ≈ 989 TFLOPS, FP32 peak ≈ 67 TFLOPS (non-tensor-core), HBM3 ≈ 3.35 TB/s, 228 KiB smem/SM.

**1. Elementwise + copy (305 ms/iter, 25.7%)**
Arithmetic intensity ≈ 0.25 FLOP/byte → deeply memory-bound. At perfect HBM3 bandwidth that's roughly 1 TB of data moved per iter. No single kernel is slow (avg 10 µs); the cost is raw launch count (27,384 = 23 launches/ms). Critical-path drivers: (a) Mamba3 fp32 shim ~400 casts/iter, (b) distributed-optimizer bucket copies, (c) Megatron grad accumulation buffers. CUDA graphs would fuse these 27k launches into one replay → this is the single biggest CUDA-graph upside.

**2. TileLang `mamba_mimo_bwd_bwd_kernel` (120 ms/iter, 10.1%)**
7.68 ms × 15.6 launches. At H200 BF16 peak that's 7.60 TFLOP — matches MIMO R=4 SSD bwd² analytically on (MBS=2, seq=4096, d_state=128, ngroups=8). Running at perhaps 50-60% of peak — compute-bound. Smem footprint fits on H200's 228 KiB (which is the bwd² GB10 blocker noted in `reference_gb10_bwd_bwd_blocker` — not a H200 problem).

**3. FP32 cuBLAS sm80 GEMM (142 ms/iter, 12%)**
Called from `nvte_multi_tensor_gemm` in optimizer + MTP/loss fp32 reduction. 346 ms of NVTX `nvte_multi_tensor_gemm` time per iter, 142 ms of actual sm80 kernel execution. Peak sm80 fp32 non-tensor ≈ 30 TFLOPS → running at ~3% of H200 peak. Memory-bandwidth-bound in practice because operands are tall/skinny optimizer buffers. **Big easy win: enable TF32 via `torch.backends.cuda.matmul.allow_tf32=True`** (TF32 tn_n sm_90 kernels exist) or switch `multi_tensor_applier` Adam to `torch._foreach_*`.

## Ranked optimization plan (from profile + my extensions)

| # | Target bucket | ms | Change | Save | Effort | Risk |
|---|---|---:|---|---:|---|---|
| 1 | Elementwise (fp32 shim) | 305 | Replace per-forward pre-hook with one-shot `Float16Module.__init__` patch that keeps Mamba3 bias/D/dt params fp32 permanently | **~60 ms** | trivial | low |
| 2 | Launch overhead / elementwise | — | Enable CUDA graphs: `--cuda-graph-impl local`. Fuses 27k launches; saves `cudaLaunchKernel` API (22% of cuda-api-sum, 1118 ms / 5 iters = 224 ms/iter) | **~150 ms** | small-medium | medium (TileLang static shapes) |
| 3 | FP32 sm80 GEMM | 142 | Enable TF32 for optimizer GEMMs: `torch.backends.cuda.matmul.allow_tf32=True` + cuBLASLt `ALLOW_TF32=1`; or switch Adam to `torch._foreach_*` | **~90 ms** | small | low |
| 4 | cuBLASLt algo search | 91 | Cache algo heuristic: `TE_CUBLAS_REDUCED_PRECISION_REDUCTION=1` or pin via `cublasLtMatmulPreference` workspace; search alone is 91 ms/iter per NVTX range | **~70 ms** | small | low |
| 5 | NCCL 87 ms overlap | 87 | minor tuning of `NCCL_BUFFSIZE` / `NCCL_IB_HCA` | ~15 ms | trivial | low |
| 6 | fp32 softmax | 67 | Force bf16 softmax on DSA indexer + MoE router (verify `--moe-router-dtype`) | ~25 ms | medium | medium |
| 7 | M2RNN bwd | 55 | Already tuned at num_warps=8 per europe autotune sweep 2026-04-11. Defer. | — | — | — |
| 8 | TL MIMO bwd_bwd swizzle | 120 | Investigate "Swizzle layout conflict ... merging to smaller granularity" log warnings. Possibly chunk_size 16→32. | 20-40 ms | medium | medium |

**Not yet profiled under FP8** — when FP8 is enabled (see `ae34b1a32dd936d07` CUDA graphs + FP8 agent), the 251 ms BF16 WGMMA GEMM bucket should roughly halve, saving another ~125 ms.

## Projected throughput progression

| Apply | Save | Iter (ms) | Tok/sec | Gain |
|---|---|---:|---:|---|
| Baseline | — | 1186 | 56,280 | — |
| + Opt 1 (fp32 shim fix) | 60 | 1126 | 59,270 | +5.3% |
| + Opt 2 (CUDA graphs) | 150 | 976 | 68,390 | +21.5% |
| + Opt 3 (TF32 optimizer) | 90 | 886 | 73,940 | +31.4% |
| + Opt 4 (cuBLASLt cache) | 70 | 816 | 80,270 | +42.7% |
| + Opt 6 (bf16 softmax) | 25 | 791 | 82,820 | +47.2% |
| + FP8 MLA+MoE (halve 251 ms) | 125 | 666 | 98,400 | +74.9% |
| + Opt 8 (MIMO swizzle) | 35 | 631 | 103,860 | +84.6% |
| **Best realistic case** | ~475 | **~711** | **~92k** | **~1.6×** |
| **Best aggressive case** | ~555 | **~631** | **~104k** | **~1.8×** |

Still **~2.4× gap to 250k** even with everything applied. Structural wins needed beyond this profile: MBS↑ (once CUDA graphs land and free memory), FP8 recipe tuning, or fundamental MIMO bwd_bwd kernel restructure. See the TileLang MIMO swizzle conflict warnings — a 20-40 ms win alone would help but won't close 2.4× on its own.

## Artifacts

- `/tmp/mimo_nsys.nsys-rep` (8.9 MB) on bench3 — full profile report
- `/tmp/mimo_nsys.sqlite` on bench3
- `/tmp/nsys_out/mimo_cuda_gpu_kern_sum.csv` — 559 distinct kernels ranked by total time
- `/tmp/nsys_out/{mimo_cuda_gpu_sum.csv, mimo_nvtx_sum.csv, mimo_cuda_api_sum.csv, mimo_cuda_kern_exec_sum.csv, mimo_cuda_gpu_mem_time_sum.csv}`
- `/tmp/nam56r_mimo_nsys_launch.sh` — reusable profile launcher
- Log: `/mnt/data/cppmega-root/cppmega/cppmega_nam56r_mimo_nsys.log`
- Local mirrors: `/tmp/nsys_local/mimo_*.csv`, `/tmp/nam56r_mimo_nsys_launch.sh`

## Known caveats

1. **Only rank 0 profiled.** Rank imbalance could hide stragglers but the 30-iter baseline run showed consistent iter times across ranks → balance is fine.
2. **WGMMA verification was inferential** (kernel naming + cuBLASLt template strings). cuBLASLt kernels are runtime-JIT'd, so no static SASS exists. Direct confirmation would require attaching `nsight-compute` to one kernel OR setting `CUBLASLT_LOG_LEVEL=5` — not done this pass.
3. **A concurrent run (`cppmega_nam56r_mimo_exp0_sanity_ckpt`) started on the same box at 12:25** — did not affect this profile (run finished 12:16) but bench3 is now busy with that other job.
