# NAM56R MIMO 7/7 full-stack baseline — 2026-04-11

First successful end-to-end run of the complete NAM56R stack with **full Mamba3 MIMO R=4** (the 7/7 feature configuration) on `h200_1` (LOCATION_1, 8x H200).

## Result

- **56,280 tok/sec** steady-state on 8x H200
- **1164 ms/iter** (median iters 10-28)
- **Target: 250,000 tok/sec** → **4.44x gap**
- All 30 iterations completed cleanly, no NaN grad norms, loss 11.75 → 4.82
- Peak GPU memory per rank: 103 GB / 140 GB

## Configuration

| Dimension | Value |
|---|---|
| Architecture | NAM56R hybrid, 52 layers, pattern `AEMEAEMEAEMR` |
| Spec | `cppmega.megatron.nam56r_full_spec build_cppmega_nam56r_full_stack_spec` |
| Hidden size | 3584 |
| FFN hidden | 18944 |
| Attention | **MLA** on most A-layers (q_lora 64, kv_lora 64, qk_head 64+32, v_head 64, GQA 28/8) + **DSA** on A-ranks 0/4/8 |
| Mamba | **MIMO R=4** full 7/7 features via `AuthorMamba3Mixer` with `cppmega_mamba3_is_mimo=True`, ngroups=8, headdim=64, dstate=128, chunk_size=16 |
| M²RNN R-layers | `CppMegaM2RNNMixer` at layer indices 12/24/36/48 (fused Triton kernel with inline PTX `tanh.approx.f32`) |
| MoE | 16 experts, top-4 dropless, + shared expert 1024 |
| MTP | 1 layer, hybrid mode |
| Data | `clang_semantic_4k_v10_train` (real, NOT mock) |
| Parallelism | TP=1, PP=1, CP=1, distributed optimizer |
| Batch | MBS=2, GBS=16 |
| Sequence | 4096 |
| Precision | BF16 (no FP8, no CUDA graphs) |
| Iters | 30 |

## Iter-by-iter timings

| Iter | Time (ms) | Note |
|---|---|---|
| 1 | 96,171 | TileLang JIT first compile (mimo_fwd, mimo_bwd_fwd, mimo_bwd_bwd) |
| 2 | 48,854 | TileLang bwd kernel JIT |
| 3-4 | 1207 / 1197 | Warm-up |
| 5 | 1176.6 | First steady-state |
| 10 | 1162.5 | |
| 20 | 1155.1 | |
| 30 | 1164.5 | |

Steady-state median over iters 10-28 ≈ 1160 ms.

Throughput = `16 * 4096 / 1.1645s` = **~56,280 tok/sec**.

## Loss trajectory (sanity)

| Iter | lm loss | grad norm |
|---|---|---|
| 1 | 11.7463 | 79.67 |
| 5 | 10.5335 | 34.57 |
| 10 | 8.2498 | 19.41 |
| 20 | 5.6597 | 7.23 |
| 30 | 4.8173 | 5.57 |
| Eval | 4.916 (PPL 136) | — |

No NaN, gradient norm converges cleanly — architecture is training correctly.

## Three blockers fixed to make this run work

See `docs/upstream_bugs.md` for full details. Summary:

1. **`fast_hadamard_transform` PyPI sdist broken** — missing `csrc/fast_hadamard_transform.cpp`. DSA dependency. Fix: install from GitHub (`pip install --no-build-isolation 'git+https://github.com/Dao-AILab/fast-hadamard-transform.git'`).

2. **TileLang `nvrtc` backend broken on cu13.2** — bundled cutlass `cute/container/array.hpp` conflicts with system CCCL 13.2.27 tuple-interface macros. Previous project notes said MIMO needs `TILELANG_EXECUTION_BACKEND=nvrtc`; that's wrong on bench3 cu13.2. Fix: use `TILELANG_EXECUTION_BACKEND=cython` (NVCC subprocess) instead.

3. **Megatron `Float16Module` silently casts Mamba3 fp32 bias/D/dt tensors to bf16** — TileLang `mamba_mimo_fwd_kernel` requires `Q_BIAS`, `K_BIAS (C_bias, B_bias)`, `D`, `dt_bias`, `mimo_x/z/o_bias` in fp32; upstream `mamba_ssm.modules.mamba3.Mamba3` initializes them in fp32 deliberately; Megatron's bf16 wrapper doesn't know and breaks the contract. Fix: forward pre-hook on every `Mamba3` module that re-upcasts `.data` of these parameters to fp32 each forward. Installed in `cppmega_fp8_shim.py` alongside the existing MIMO `__post_init__` patch.

## Gap analysis: why 56k and not 250k

The 4.44× gap is primarily the sum of:

- **4× MIMO state compute** vs SISO — mamba3 MIMO R=4 has 4× the state-tensor work per token vs SISO. This is architectural, not optimizable.
- **No CUDA graphs** — `--cuda-graph-impl none`. Typical saving 10-15% once enabled.
- **No FP8** — BF16-only everywhere. MLA attention + MoE GroupedMLP + dense FFN are all candidates for FP8; mamba3/M²RNN scan kernels must stay BF16 (known slower under FP8, see `docs/fp8_path_status.md`). Typical saving 8-15% end-to-end.
- **No WGMMA verification** — cuobjdump SASS scan not done; we don't know for sure whether the MLA/MoE kernels are actually emitting WGMMA vs falling back to older HMMA.
- **TileLang swizzle conflicts** — the run log has many `Swizzle layout conflict for buffer dPhiO_shared/dstates_shared, merging to smaller granularity` warnings from TileLang. These are info-level but indicate the MIMO kernel is falling back to a less-optimal smem layout. Kernel-level tuning opportunity.
- **nsys profile not yet run** — until we profile, the exact bottleneck distribution is guesswork.

## Next steps (planned as of 2026-04-11)

1. **nsys profile** the baseline to find top-5 kernels by total time, compute % breakdown (GEMM / TileLang scan / M²RNN / TE layernorm / optimizer / comm / other). **Highest priority** — everything else is guessing without this data.
2. **CUDA graphs** — add `--cuda-graph-impl local`. First try full-iteration capture, fall back to `attn-only` or `mamba-only` scope if full capture hits unsupported ops. Expected 10-15%.
3. **FP8 selective** — enable TE FP8 recipe (`--fp8-format hybrid`) for MLA + MoE + dense FFN; mamba3 scan and M²RNN stay BF16 automatically. Expected 8-15% end-to-end.
4. **WGMMA SASS verification** — run `cuobjdump --dump-sass` on the Triton/TE cubins and confirm `wgmma.mma_async` opcodes are present on hot GEMMs.
5. **TileLang MIMO swizzle fix** — investigate the "merging to smaller granularity" warnings and either patch upstream or work around by adjusting ngroups/chunk_size.
6. **Triton M²RNN autotune sweep** — full 225-config space (BLOCK_N×BLOCK_H×num_warps×num_stages) with num_warps ∈ {1,2,4,8,16} and num_stages ∈ {1..5}. Currently only 2/4/8 warps manually tested.
7. **MBS increase** — after CUDA graphs land, try MBS=3 or 4 at GBS=32/64. Peak memory is 103 GB / 140 GB so there's room.

## Artifacts

- Training log (rank 0): `/mnt/data/cppmega-root/cppmega/cppmega_nam56r_mimo_full_realdata.log` (on bench3, ~1.1 MB)
- Launch wrapper: `/tmp/nam56r_mimo_full_launch.sh` (bench3)
- Spec: `cppmega/megatron/nam56r_full_spec.py` `build_cppmega_nam56r_full_stack_spec`
- Shim (MIMO `__post_init__` + Mamba3 fp32 pre-hook): pattern from `scripts/remote_smoke_h200_fp8_mamba3_matrix.sh` lines 110-168, extended with the fp32-param hook
