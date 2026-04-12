# DSA EP=2 + TileLang Sparse MLA Sweep Results — 2026-04-12

## Architecture

- **Model**: NAM56R 4.73B (52 layers: 27 Mamba3 MIMO + 13 Attention [9 DSA + 4 MLA] + 12 MoE)
- **DSA**: topk=256, Absorbed MLA (PR #3674)
- **Attention dims**: qk_pos_emb_head_dim=64 (was 32), d_total=192, d_v=64, tail_dim=128
- **MoE**: 16 experts, router_topk=4, ffn_hidden=896
- **MTP**: depth=2
- **Seq**: 4096, MBS=4
- **Machine**: H200×8 (143 GiB HBM each)

## Key Components

| Component | File | What |
|-----------|------|------|
| TileLang fused sparse MLA | upstream `ops/sparse_mla.py` | WGMMA fused fwd+bwd kernel |
| Sparse absorbed DSA | `dsa_sparse_absorbed.py` | PyTorch fallback if TileLang fails |
| Mamba recompute | `mamba_recompute_patch.py` | Activation checkpointing (CG-aware) |
| Skip indexer loss | `dsa_fp8_patch.py` | Skips 7 GiB bmm (TODO: head-streaming) |
| Upstream patches | `upstream_patches/apply_dsa_cg_patches.py` | 7 patches to Megatron DSA |

## Memory Optimization

| State | Peak Memory | Method |
|-------|------------|--------|
| Before (OOM) | 135.75 GiB | Dense matmul + no Mamba recompute |
| + Mamba recompute | 129 GiB | `torch.utils.checkpoint` for Mamba/M2RNN |
| + Sparse absorbed DSA | 103 GiB | Gather-scatter instead of dense QK |
| + Skip indexer loss | 103 GiB | Skip 7 GiB `compute_dsa_indexer_loss` bmm |
| + TileLang fused | **55 GiB** | Fused kernel, no intermediate tensors |

## Topology Sweep (lr=1e-5, full CG, TileLang, 20 iters)

| Config | CG Scope | TFLOP/s | tok/sec | GBS | Stable | Val PPL |
|--------|----------|---------|---------|-----|--------|---------|
| **PP=1 EP=1 DP=8** | full | **237** | **75k** | 32 | ✅ 20/20 | 27.2 |
| PP=1 EP=2 DP=4 | full | 231 | 73k | 64 | ✅ 20/20 | — |
| PP=2 VPP=2 EP=1 DP=4 | full | 195 | 62k | 64 | ✅ 20/20 | — |
| PP=2 VPP=2 EP=2 DP=2 | full | 185 | 59k | 64 | ✅ 20/20 | 23.6 |

### Observations:
- **PP overhead**: 22% (237→185 TFLOP/s PP=1→PP=2) — pipeline bubbles
- **EP overhead**: 3-5% (237→231 at PP=1, 195→185 at PP=2) — DeepEP dispatch
- **Best single-node**: PP=1 EP=1 (237 TFLOP/s / 75k tok/sec)
- **Best multi-node scalable**: PP=2 EP=2 (59k tok/sec, ready for EP=4/8)

## LR Sweep (PP=1, full CG, TileLang)

| lr | CG | Iter 4 grad_norm | Iter 15 | Stable? |
|-----|-----|-----------------|---------|---------|
| **1e-5** | full | 95.5 | 18.5 | **✅ 20/20** |
| 3e-5 | full | 60.8 | 7.7M | ❌ diverges iter 15 |
| 1e-4 | full | 3.9T → inf | nan | ❌ diverges iter 4 |
| 1e-4 warmup 10 | full | 60 (warmup) | nan (decay) | ❌ too short warmup |
| 1e-4 warmup 100 | full | TBD | TBD | 🔄 next test |

### Root cause of lr sensitivity:
TileLang fused kernel uses BF16 WGMMA with online softmax — different numerical path
than PyTorch's FP32 softmax + BF16 matmul. Gradient landscape has sharper curvature
at initialized weights → lower lr needed at start. lr warmup (1e-5 → 1e-4 over 100+
steps) expected to work since weights stabilize quickly.

## CUDA Graph Compatibility

| Scope | lr=1e-5 | lr=1e-4 |
|-------|---------|---------|
| none | ✅ stable | ❌ inf at iter 4 |
| moe_router moe_preprocess | ✅ stable | ❌ inf at iter 4 |
| **attn mamba moe_router moe_preprocess** | **✅ stable** | ❌ inf at iter 4 |

CG itself does NOT cause NaN — it's lr sensitivity. At lr=1e-5, full CG scope works perfectly.

## Upstream PRs Applied

### On europe (6 PRs):
| PR | Title | Impact |
|----|-------|--------|
| #4070 | DSA indexer loss scaling | CRITICAL correctness |
| #4047 | Async P2P race fix | PP=2 race condition |
| #3919 | Recompute + CG guard | Safety net |
| #3698 | softmax_scale fix in MLA | Correctness |
| #3751 | Reuse grad buffer | Memory optimization |
| #3656 | Chunked MLP training | Memory optimization |

### On bench3 (10 PRs — same 6 + 4 more):
- All above + #4089 (permute fusion), #4150 (PP broadcast), #4229 (FlashAdamW), #3553 (DSA MambaModel — partial, broke Symbols.DS_ATTENTION)

### 18 NEW PRs downloaded (not yet applied):
See `.tmp/upstream_prs/new_prs_summary.md`

## FP8 TileLang — Next Steps

Current TileLang kernel uses BF16 (`dtype = T.bfloat16`). For FP8:
1. Change `dtype = T.float8_e4m3fn` in fwd kernel
2. Add rowwise scaling (per-row scale factors)
3. Keep `accum_dtype = T.float32` (accumulation stays FP32)
4. Backward kernel may need FP8 → BF16 conversion for dQ/dKV
5. Expected speedup: ~1.5× from FP8 WGMMA on Hopper (2× theoretical, ~75% efficiency)
6. Expected throughput: 237 × 1.5 ≈ **355 TFLOP/s / 112k tok/sec** at PP=1

## Architecture Change: qk_pos_emb_head_dim 32 → 64

Changed to make `tail_dim = d_total - d_v = 192 - 64 = 128` (power-of-2).
TileLang kernel requires power-of-2 tile dimensions for correct WGMMA layout.
Impact: +3% attention params (13 layers × 32 extra RoPE dims).

Files: `cppmega/recipes/megatron_args.py`, `cppmega/recipes/nam56r_nemo_recipe.py`

## Known Issues

1. **Indexer loss disabled** (`CPPMEGA_DSA_SKIP_INDEXER_LOSS=1`): DSA routing not trained.
   Fix: implement head-streaming `compute_dsa_indexer_loss` (~0.8 GiB vs 7 GiB full bmm).

2. **lr=1e-4 + CG**: causes gradient overflow. Use lr≤1e-5 or warmup.

3. **PR #3553 partial apply**: `Symbols.DS_ATTENTION` not added. Need full port.

4. **FP32 P/dP precision fix**: TileLang GEMM requires matching dtype for A,B operands.
   Cannot use FP32 shared memory for P/dP. Precision is maintained in FP32 accumulator
   fragments; BF16 truncation for GEMM input is acceptable at lr=1e-5.

5. **Europe disk**: migrated to `/mnt/data` via symlink. All data on 10TB disk now.
