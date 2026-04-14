# cppmega

Megatron-first training framework for **NAM56R** — a 4.73B hybrid Mamba3 + MLA + DSA + MoE model.

## Production Configuration

**289 TFLOP/s** on 8xH200 (verified 2026-04-14). BF16 outperforms FP8 tensorwise by 3.5%.

| Parameter    | Value                                                     |
| ------------ | --------------------------------------------------------- |
| Model        | NAM56R 4.73B (52 layers, hybrid `*EME*EME*EMM*`)          |
| Architecture | heads=32, hidden=4096, headdim=128, d_state=128           |
| Mamba        | Mamba-3 MIMO rank=4, chunk=16, SISO+MIMO TileLang kernels |
| Attention    | 4 MLA layers (q_lora=64, kv_lora=64, qk_pos=64)           |
| DSA          | 9 DSA layers (sparse attention, indexer topk=256)         |
| MoE          | 16 experts, topk=4, shared expert, grouped GEMM           |
| MTP          | 2 prediction layers                                       |
| Precision    | BF16 (FP8 tensorwise available but -3.5% slower)          |
| Parallelism  | PP=1, TP=1, EP=4, DP=2                                    |
| Micro-batch  | MBS=4, GBS=64, seq_len=4096                               |
| Hardware     | 8x NVIDIA H200 141GB (NVLink)                             |

## Quick Start

### Prerequisites

```bash
# Megatron-LM (our dev_latest branch with PR #3674 + PR #4268)
cd /mnt/data/cppmega-root/megatron-lm

# After any Megatron update, MUST apply patches:
cd /mnt/data/cppmega-root/cppmega
python -m cppmega.megatron.upstream_patches.apply_dsa_cg_patches

# Required packages
pip install dualpipe  # from github: pip install git+https://github.com/deepseek-ai/DualPipe.git
pip install liger-kernel
pip install apex  # NVIDIA apex from source with --cpp_ext --cuda_ext
```

### Training

```bash
cd /mnt/data/cppmega-root/cppmega

# Production run: PP=1 EP=4 h32 + all optimizations
CPPMEGA_INDEX_CACHE=1 \
CPPMEGA_LEMYX_DSA=1 \
CPPMEGA_LIGER_CE=1 \
bash scripts/remote_smoke_h200_dsa_9_4_m.sh
```

The script auto-applies:
- Regional torch.compile (4 Mamba3 elementwise regions)
- IndexCache (67% DSA indexer savings)
- lemyx fused FA+KL kernel (DSA warmup)
- Liger chunked cross-entropy (MTP memory savings)
- FP8 tensorwise recipe
- expandable_segments (mandatory)

## Optimization Stack

### Always On (no env gates)

| Component              | File                      | Effect                                          |
| ---------------------- | ------------------------- | ----------------------------------------------- |
| Regional torch.compile | `mamba3_compile_patch.py` | Fuses Mamba3 elementwise ops (5.93x data-dep-A) |
| expandable_segments    | script env var            | Prevents CUDA allocator fragmentation OOM       |
| Unfused DSA banned     | `apply_dsa_cg_patches.py` | Crash if fused SparseMLA unavailable            |

### Env-Gated

| Component         | Gate                          | File                         | Effect                                               |
| ----------------- | ----------------------------- | ---------------------------- | ---------------------------------------------------- |
| IndexCache        | `CPPMEGA_INDEX_CACHE=1`       | `index_cache_patch.py`       | 3 Full + 6 Shared DSA layers = 67% indexer savings   |
| lemyx DSA         | `CPPMEGA_LEMYX_DSA=1`         | `lemyx_dsa_warmup.py`        | Fused FA+KL TileLang kernel for indexer warmup       |
| Liger CE          | `CPPMEGA_LIGER_CE=1`          | `mtp_liger_ce.py`            | Chunked fused cross-entropy, saves 21 GiB MTP logits |
| Selective FP8 MoE | `CPPMEGA_SELECTIVE_FP8_MOE=1` | `selective_fp8_moe_patch.py` | FP8 only on MoE expert GEMMs                         |
| Mamba recompute   | `CPPMEGA_MAMBA_RECOMPUTE=1`   | `mamba_recompute_patch.py`   | Activation checkpointing for Mamba layers            |
| FP8 param-gather  | `CPPMEGA_FP8_PARAM_GATHER=1`  | Megatron `--fp8-param-gather`| -5 GiB (FP8 all-gather bucket, master stays FP32)    |
| DualPipeV         | `CPPMEGA_DUALPIPEV=1`         | `apply_dualpipev_patch.py`   | V-shape PP; forces Megatron PP=1, carves own 2-rank group (experimental) |

### Pipeline Schedules

| Schedule      | File                      | When to use                                |
| ------------- | ------------------------- | ------------------------------------------ |
| Standard 1F1B | Megatron built-in         | PP=2 VPP=2 (194 TFLOP/s)                   |
| DualPipeV     | `dualpipev_schedule.py`   | PP=2 near-zero bubble (205 TFLOP/s, MBS=2) |
| combined_1f1b | `hybrid_schedule_plan.py` | PP=1 EP A2A overlap (hides MoE all-to-all) |

### Upstream Patches (apply_dsa_cg_patches.py)

**MUST run after every Megatron update.** 9 patches:

1. CUDA graph safety (ban torch.equal/.any CPU syncs)
2. Remove 576/512 dim hardcodes (our dims: 128/64)
3. SparseMLA d_v propagation
4. sparse_mla.py d_v parameter
5. tilelang_sparse_mla_fwd.py dim assertions relaxed
6. tilelang_sparse_mla_bwd.py D=512 hardcode removed
7. tilelang_sparse_mla_bwd.py FP32 P/dP precision
8. CG-safe _scatter_topk_into_index_mask (branchless)
9. FP8 SparseMLA dispatch (QuantizedTensor → SparseMLA_FP8)

## Throughput Results

| Config                        | TFLOP/s | tok/sec/GPU | Notes             |
| ----------------------------- | ------- | ----------- | ----------------- |
| **PP=1 MBS=8 BF16 no-CG + compile** | **289** | ~9,250 | Production config |
| PP=1 MBS=8 FP8 no-CG + compile | 279 | ~8,950 | FP8 overhead > gain |
| PP=1 EP=4 MBS=4 + compile | 265 | ~8,500 | With CUDA graphs |
| PP=2 VPP=2 EP=4 MBS=4         | 194     | ~6,200      | Save verified     |
| DualPipeV PP=2 MBS=2          | 205     | ~6,600      | No DSA/FP8 yet    |
| Selective FP8 MoE (no DSA)    | 205     | ~6,600      | MoE-only FP8      |
| PP=1 EP=1 MBS=10 (h=3584)     | 272     | ~8,700      | Previous golden   |

## Machines

| Machine | Zone           | IP            | Path                     | GPU              |
| ------- | -------------- | ------------- | ------------------------ | ---------------- |
| europe  | LOCATION_2 | H200_2_IP  | `/mnt/data/cppmega-root` | 8x H200          |
| bench3  | LOCATION_1  | H200_1_IP | `/mnt/data/cppmega-root` | 8x H200          |
| GB10    | local network  | gb10          | `/home/dave`             | 1x GB10 (sm_121) |

### Megatron Version

Branch `dev_latest` on top of NVIDIA/Megatron-LM `dev`:
- `core_v0.15.0rc7` + PR #3674 (DSA absorbed MLA) + PR #4268 (delayed wgrad overlap)

### Software Stack

- PyTorch 2.12 nightly + cu132
- Transformer Engine 2.13
- TileLang 0.1.8
- mamba-ssm 2.3.1
- Megatron Core 0.18
- NVIDIA Apex (from source)
- dualpipe 1.0.0+030ce43 (from github)
- liger-kernel

## Key Design Decisions

- **heads=32, hidden=4096**: FP8 compatible (32%8=0), WGMMA tiling, lemyx (heads==index_heads)
- **52 layers** (NAM56R_DEPTH): divides by 4 (VPP=4) and 2 (VPP=2)
- **Single DSA path**: lemyx + IndexCache only, no fallbacks, crash on failure
- **No _apply guard**: D/dt_bias stay bf16 after Float16Module, use .float() in forward
- **No silent fallbacks**: critical patches crash on failure, never try/except + continue
- **expandable_segments mandatory**: hardcoded, not overridable
- **Unfused DSA banned forever**: if fused kernel unavailable, crash immediately

## Project Structure

```
cppmega/
  megatron/
    index_cache_patch.py       # DSA index reuse (3 Full + 6 Shared)
    lemyx_dsa_warmup.py        # Fused FA+KL TileLang kernel
    mamba3_compile_patch.py    # Regional torch.compile (4 regions)
    dualpipev_schedule.py      # DualPipeV pipeline schedule (167 LOC)
    hybrid_schedule_plan.py    # build_schedule_plan for MambaModel
    mtp_liger_ce.py            # Liger fused cross-entropy for MTP
    selective_fp8_moe_patch.py # FP8 only for MoE experts
    mamba_recompute_patch.py   # Mamba activation checkpointing
    noconv_mamba_mixer.py      # Mamba3 mixer (no conv1d)
    custom_mamba_model.py      # CppMegaMambaModel wrapper
    sparse_mla_ops/            # TileLang SparseMLA fwd/bwd + FP8
    upstream_patches/
      apply_dsa_cg_patches.py  # 9 patches for DSA + CG + FP8 + dims
  recipes/
    nam56r_nemo_recipe.py      # NAM56R configuration
scripts/
  remote_smoke_h200_dsa_9_4_m.sh  # Production training script
docs/
  optimization_session_2026_04_13.md     # Session notes (English)
  optimization_session_2026_04_13_ru.md  # Session notes (Russian)
```
