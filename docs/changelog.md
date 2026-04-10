# NAM56R NeMo Migration Changelog

## 2026-04-10: Mamba3-Native Mixer (CppMegaMamba3Mixer v2)

### New Approach: Inject Mamba3 into Native SSD Kernel

Rewrote `CppMegaMamba3Mixer` to keep the **native `mamba_chunk_scan_combined` kernel**
while adding Mamba3 features between conv1d and scan.  Previous `mamba3_te` (127k tok/sec)
used Author Mamba3 kernels (`mamba3_siso_combined`) which broke CUDA graph capture.

**Key architectural changes:**
1. Override `_ssm_training` to use **separate conv1d + scan** (not fused `mamba_split_conv1d_scan_combined`)
2. Inject QK-Norm and B/C bias on B, C between conv1d output and scan input
3. Data-dependent A via "A=-1/dt trick" preserving kernel compatibility
4. Fixed z-gating bug: was `self.norm(y)`, corrected to `self.norm(y, z)`

### The A=-1/dt Trick for Data-Dependent A

The SSD kernel takes scalar A per head. For data-dependent A_dd(x):
```
A_kernel = -1.0                    # constant per head
dt_kernel = -(A_dd * dt_eff)       # positive, per position
x_scaled = x / |A_dd|             # compensate input scaling

Decay:  exp(-1 * -(A_dd*dt)) = exp(A_dd*dt)  ✓ (data-dependent)
Input:  (-A_dd*dt) * B * (x/|A_dd|) = dt * B * x  ✓ (unchanged)
D skip: handled separately with original x (not scaled)
```

Requires `dt_softplus=False` (we pre-apply softplus) and `D=None` (added manually).
Works correctly because `rmsnorm=True` mode does z-gating externally.

### Files Changed
- `cppmega/megatron/mamba3_mixer.py` — rewritten with `_split_conv_scan` shared logic,
  data-dep A support, fixed z-gating
- `cppmega/recipes/nam56r_nemo_recipe.py` — added `nam56r_mamba3_native_pretrain()` (nheads=56)
  and `nam56r_mamba3_native_max_throughput()` (FP8, nheads=64, MBS=5, GBS=320)
- `scripts/remote_smoke_h200_mamba3_native.sh` — 3-way benchmark script
- `tests/test_mamba3_mixer.py` — 16 tests (math proofs, recipe tests, env control)

### Expected Performance

| Config | Kernel | Expected tok/sec | Notes |
|--------|--------|-----------------|-------|
| nemo_native (baseline) | fused split+scan | 165k | vanilla Mamba-2 |
| **mamba3_native QK-Norm+bias** | split conv + scan | ~155-165k | +2 RMSNorm + bias adds |
| mamba3_native + data-dep A | split conv + scan | ~145-160k | +softplus + norm + div |
| mamba3_te (old, Author kernels) | mamba3_siso_combined | 127k | CUDA graph breakage |

Key: if mamba3_native matches ~165k, then FP8 + CUDA graphs should reach 200k+ with Mamba3.

### Benchmark Results (8×H200, BF16 + CUDA graphs, MBS=4, GBS=32)

| Config | Steady-state (ms) | tok/sec | vs Baseline |
|--------|-------------------|---------|-------------|
| **Baseline** (Mamba-2 SSD, fused kernel) | **743** | **176k** | — |
| **Mamba3 native** (QK-Norm + B/C bias) | **788** | **166k** | **+6.1%** |
| mamba3_te (Author kernels, old) | 1,035 | 127k | +39% |

The 6.1% overhead comes from replacing `mamba_split_conv1d_scan_combined` (fused conv+scan)
with separate `causal_conv1d_fn` + `mamba_chunk_scan_combined` + 2×RMSNorm + bias.

With FP8 + MoE CUDA graph + MBS=5 + GBS=320, extrapolated: ~198k tok/sec (vs 211k baseline).

### Optimization: Fused vs Split (benchmarked on R595/CUDA 13.2)

| Config | Avg 5-15 (ms) | tok/sec | Overhead |
|--------|---------------|---------|----------|
| Baseline (Mamba-2 SSD) | 748 | 175k | — |
| Mamba3 Fused (pre-conv QK-Norm) | 784 | 167k | +4.9% |
| Mamba3 Split (post-conv QK-Norm) | 784 | 167k | +4.9% |

Fused and split give identical overhead — the 4.9% is from QK-Norm + bias ops
themselves, not from kernel split vs fused.

---

## 2026-04-10: Mamba3 Feature Gap — What We Have vs Real Mamba3

### Current CppMegaMamba3Mixer: Mamba-2 + 2/7 Mamba3 Features

The "Mamba3 native" mixer is NOT Mamba3. It's Mamba-2 SSD with QK-Norm and B/C bias:

| Mamba-3 Feature | Status | Why Missing |
|-----------------|--------|-------------|
| QK-Norm on B/C | **DONE** | RMSNorm before/after conv1d |
| Learnable B/C bias | **DONE** | Element-wise addition |
| Trapezoidal discretization | **NOT DONE** | Requires modified scan: h_t = α*h_{t-1} + β*v_{t-1} + γ*v_t |
| Data-dependent A | **CODE EXISTS, OFF** | A=-1/dt trick ready, not benchmarked |
| Complex RoPE on B/C | **NOT DONE** | Pre-scan rotation, implementable in PyTorch |
| No conv1d | **NOT DONE** | We keep conv1d for kernel compatibility |
| MIMO | **NOT DONE** | Shared state, native SSD can't express directly |

### Core Mamba-3 Innovations Missing

The defining features of Mamba-3 (ICLR 2026) are:
1. **Trapezoidal discretization** — replaces exponential-Euler with a 2-band
   bidiagonal matrix that implicitly includes a size-2 convolution
2. **Data-dependent A** — per-position, per-head decay factor
3. **MIMO** — R-rank shared-state scan for better capacity

Without these, our mixer is essentially Mamba-2 + normalization tricks.

### Author Kernels: Fast Compute, Broken Integration

| Kernel | Compute Speed | CUDA Graph? | Why |
|--------|--------------|-------------|-----|
| `mamba_chunk_scan_combined` (Mamba-2) | baseline | **YES** | Clean Triton |
| `mamba3_siso_combined` (SISO) | ~same | **NO** | 27 saved tensors, non-tensor autograd inputs |
| `mamba3_mimo_combined` (MIMO) | ~same | **NO** | All SISO issues + TileLang JIT + TVM PackedFunc |

The 30% throughput gap (1035ms vs 793ms) is NOT from kernel compute —
it's from loss of CUDA graph capture. Without graphs, every iteration
pays full kernel launch overhead for 17 Mamba layers × 5 bwd kernels.

### Nanochat Reference Implementation (Pure PyTorch)

`/Users/dave/sources/nanochat/nanochat/mamba2.py` implements ALL Mamba3
features in pure PyTorch using a chunked SSD reference scan:

| Feature | Implementation | Lines | CUDA Graph OK? |
|---------|---------------|-------|----------------|
| Trapezoidal | B pre-scaling + diagonal correction | 2079-2189 | YES |
| MIMO | Shared-state einsum scan | 3252-3451 | YES |
| Data-dependent A | (B,T,H) shaped A in chunked scan | 1960-1971 | YES |
| Complex RoPE | cos/sin rotation on B/C pairs | 919-1168 | YES |

**Key insight:** nanochat's trapezoidal works by pre-scaling B:
```
gamma = sigmoid(trap) * dt
shifted_gamma = (1 - sigmoid(trap_next)) * dt_next
scale = gamma + shifted_gamma
B_scaled = B * (scale / dt)  # pre-scale BEFORE scan
```
Then the standard scan with `dt * B_scaled * x = scale * B * x` gives trapezoidal
weights. A diagonal correction subtracts the excess `shifted_gamma` contribution.

**Catch:** nanochat uses `_ssd_scan_ref` (pure PyTorch, O(T*chunk_size) per chunk) —
much slower than the fused Triton `mamba_chunk_scan_combined`. This is WHY nanochat
is slow: it falls back to Python reference scan for trapezoidal/dd_A/MIMO.

### Path Forward: Real Mamba3 at Production Speed

Three approaches, ordered by effort:

1. **PyTorch reference scan** — port nanochat's `_ssd_scan_ref` to cppmega.
   Complete Mamba3 but slow (~2-3x slower than native kernel).
   Good for R&D and correctness verification.

2. **Hybrid: pre-process + native kernel** — for trapezoidal and RoPE,
   apply pre-scaling/rotation in PyTorch then use `mamba_chunk_scan_combined`.
   Works for trapezoidal (B-scaling trick) and RoPE (pre-rotation).
   Does NOT work for data-dependent A (kernel expects scalar A per head)
   or MIMO (kernel gives independent states per batch, not shared).

3. **Custom Triton kernel** — write a CUDA-graph-compatible chunked SSD scan
   that natively supports trapezoidal + dd_A + MIMO. Essentially rewrite
   `mamba_chunk_scan_combined` with Mamba3 math. Highest effort but production speed.

### BREAKTHROUGH: Author Kernels + CUDA Graphs = WORK

Previous analysis claimed Author Mamba3 kernels (mamba3_siso_combined) were
incompatible with CUDA graphs. **THIS WAS WRONG.**

Tested on H200x8 with `--cuda-graph-impl local`:
- Small model (4 layers): 8 graphs created in 0.29s, EXIT 0
- Full NAM56R (52 layers, 8×H200): **796 ms/iter = 165k tok/sec**

| Config | ms/iter | tok/sec | Mamba3 features | Overhead |
|--------|---------|---------|-----------------|----------|
| Baseline (Mamba-2 SSD + TE graphs) | 748 | 175k | 0/7 | — |
| CppMegaMamba3Mixer (native SSD) | 784 | 167k | 2/7 | +4.9% |
| **Author SISO + local graphs** | **796** | **165k** | **6/7** | **+6.4%** |
| Author SISO no graphs (old) | 1,035 | 127k | 6/7 | +38% |

The 6.4% overhead is from actual kernel compute (QK-Norm, RoPE, trapezoidal,
data-dependent A), NOT from CUDA graph loss. CUDA graphs via `--cuda-graph-impl local`
capture Author Triton kernels correctly because `torch.cuda.CUDAGraph` captures
at CUDA driver level — Python/autograd dispatch runs on CPU during capture only.

Note: `--cuda-graph-impl local` + `--moe-shared-expert-overlap` = assertion error
in MoE shared_experts.py. Remove `--moe-shared-expert-overlap` for local graphs.
TE graphs (`--cuda-graph-impl transformer_engine`) also work on small models.

### TileLang MIMO: NVRTC Backend Bypasses TVM

TileLang has `execution_backend="nvrtc"` which bypasses TVM runtime entirely:
- Compiles to cubin via NVRTC
- Launches with `cuLaunchKernelEx` (CUDA driver API, graph-compatible)
- Set via: `TILELANG_EXECUTION_BACKEND=nvrtc`

### DSL Landscape (Mamba3 uses 3 DSLs)

| Component | DSL | CUDA Graph? |
|-----------|-----|-------------|
| SISO prefill | Triton | YES (confirmed on H200) |
| MIMO prefill | TileLang | YES (via nvrtc backend) |
| Decode | CuTe DSL | YES (CUTLASS 4.3.4 fixed refcnt bug) |
| FA4 | CuTe DSL | YES |

### Driver Update

Updated bench machine from 580.126.09 to **595.58.03** (CUDA 13.2).

### Upstream Status

**No Mamba3 integration exists in NVIDIA Megatron-LM, NeMo, or TransformerEngine.**
Zero PRs as of 2026-04-10. The upstream Mamba3 code lives only in
`state-spaces/mamba` PR #858 (merged) with Author kernels.

### PR #909 Fix Applied

Patched `mamba3_siso_combined.py` on bench machine: cache `ctx.saved_tensors`
for FSDP activation checkpointing compatibility. Uploaded to
`sftp://BUCKET_TRAINING_DATA/artifacts/cu132/mamba3_siso_combined_pr909_patched.py`.

---

## 2026-04-10: Mamba3 Feature Status & Gap Analysis

### What the Production Config Actually Is

The production recipe `nam56r_nemo_native_max_throughput()` achieving **211k tok/sec / 50.1% MFU**
uses **vanilla Megatron Mamba-2 SSD** — NOT Mamba3. The spec is:

```python
spec_module="megatron.core.models.mamba.mamba_layer_specs"  # standard Megatron
```

No Mamba3 features are active in the production config. The throughput comes entirely from
NeMo 3 Nano optimizations (FP8 tensorwise, TE CUDA graphs, MoE drop-and-pad, gradient accumulation).

### Mamba3 Features: Built but Not Production-Ready

Six Mamba3 features were ported into TE-compatible modules:

| Feature | Module | Status | Impact on Speed |
|---------|--------|--------|-----------------|
| QK-Norm on B/C | `mamba3_te_mixer.py` | Tests pass | -23% throughput |
| Learnable B/C bias | `mamba3_te_mixer.py` | Tests pass | -23% throughput |
| Trapezoidal discretization | `mamba3_te_mixer.py` | Tests pass | -23% throughput |
| Complex RoPE on SSM | `mamba3_te_mixer.py` | Tests pass | -23% throughput |
| Data-dependent A | `mamba3_te_mixer.py` | Tests pass | -23% throughput |
| No-conv (conv1d removed) | `noconv_mamba_mixer.py` | Tests pass | Not benchmarked |

The -23% comes from using Author Mamba3 scan kernels (`mamba3_siso_combined`) which
cannot participate in TE CUDA graph capture, breaking the fusion pipeline.

### Mamba3 vs Mamba2 Throughput Comparison

| Mode | Scan Kernel | Iter (ms) | tok/sec | MFU | CUDA Graphs |
|------|-------------|-----------|---------|-----|-------------|
| **nemo_native** (production) | `mamba_chunk_scan_combined` | 810 | 165k | 37.2% | yes |
| **nemo_native + FP8 + MoE graph** | `mamba_chunk_scan_combined` | 6,207 (GBS=320) | **211k** | **50.1%** | yes |
| mamba3_te | `mamba3_siso_combined` | 1,035 | 127k | ~29% | partial |
| author_dp (legacy wrap) | Author Mamba3 native | 39,800 | 3.3k | <1% | no |

### Features NOT Implemented

| Feature | Source | Status |
|---------|--------|--------|
| **M²RNN** (Mamba3 R-layers) | Author Mamba3 / accelerated-model-architectures | **Not implemented** |
| **MIMO** (multi-input multi-output SSM) | `mamba3_mimo_combined` kernel | Kernel reference only, not wired |
| **Output projection norm** (RMSNormGated before out_proj) | `mamba3_te_out_proj.py` | Built, not benchmarked |

### Path Forward for Mamba3 at Production Speed

The `noconv_mamba_mixer.py` module takes a different approach: pre-processes Mamba3 features
(data-dependent A, trapezoidal scale) into **Megatron's native `mamba_chunk_scan_combined`** kernel
using the A=-1/dt=-ADT trick. This preserves TE CUDA graph compatibility but is an approximation:

- Data-dependent A: exact (A_kernel=-1, dt_kernel=-ADT, so cumsum(-1 * -ADT) = cumsum(ADT))
- Trapezoidal: approximate (pre-multiplied into B, not fused into scan)
- QK-Norm: exact (applied to B/C before kernel)
- Complex RoPE: not supported (would need kernel modification)

This approach has **not been benchmarked on H200** yet. If it matches nemo_native speed,
it would give us Mamba3 features at 200k+ tok/sec.

### Test Status (211 pass, 3 fail, 6 skip)

Failing tests are in `test_nam56r_full_spec.py` and `test_nam56r_launch.py` related to
MLA + PP layer offset — not related to Mamba3 or production throughput.

---

## 2026-04-10 (update): CUDA Graphs + FP8 Throughput Optimization

### Megatron Upgrade
- Upgraded from `fd762549` to `e40feed4a` (CUDA graph scope support)
- TE-scoped CUDA graphs (attn, mamba, moe_router, moe_preprocess) reduce kernel launch overhead
- Optimizer CUDA graph not compatible (stream capture error with distributed optimizer)

### Performance Optimizations Applied
1. **CUDA graphs (TE-scoped)**: `--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe_router moe_preprocess`
2. **overlap-param-gather**: Enabled for TP=1 (works across DP dimension)
3. **moe-router-fusion**: TE 2.7+ fused TopK routing kernel
4. **FP8 tensorwise**: Per-tensor current scaling (NeMo Nano v2 style), requires nheads=64 (multiple of 16)
5. **No selective recompute with CUDA graphs**: core_attn recompute conflicts with graphed attention

### mamba_num_heads Correction
- Fixed from 112 → 56 for nemo_native mode (Megatron MambaMixer: nheads=hidden/headdim=3584/64=56)
- 112 was for Author Mamba3 expand=2 (hidden*2/headdim), not Megatron's built-in mixer
- FP8 mode uses nheads=64 (FP8 alignment: `(2*d_inner + 2*ngroups*d_state + nheads) % 16 == 0`)

### Throughput Results (8×H200, Megatron e40feed4a)

| Config                       | MBS | GBS | Iter (ms) | tok/sec  | Memory/GPU |
| ---------------------------- | --- | --- | --------- | -------- | ---------- |
| BF16, no CUDA graphs         | 4   | 32  | 1,014     | 129k     | 90 GiB     |
| BF16 + CUDA graphs           | 4   | 32  | 793       | 165k     | 90 GiB     |
| BF16 + CUDA graphs           | 5   | 40  | 940       | 174k     | 105 GiB    |
| FP8 tensorwise + CUDA graphs | 4   | 32  | 760       | 172k     | 91 GiB     |
| FP8 tensorwise + CUDA graphs | 5   | 40  | 897       | **183k** | 104 GiB    |

### Real Data Training
- Fixed .idx format: `num_documents` must be `num_sequences + 1` (sentinel value)
- Fixed `--data-path` parsing: split space-separated blend into individual CLI args
- Single dataset training working: 766ms/iter = 171k tok/sec on clang_commits_4k_v1
- Loss: 19.0 → 3.4 in 30 iterations (lr=3e-4 cosine, BF16 + CUDA graphs)

### MFU Analysis
```
Best config: FP8 tensorwise + CUDA graphs + MBS=5
Iter time: 897 ms
Tokens/iter: 40 * 4096 = 163,840
Active params: ~3.03B
FLOPs/iter: 163,840 * 6 * 3.03B = 2.979 PFLOP
TFLOP/s/GPU: 2979 / 0.897 / 8 = 415.0
H200 BF16 peak: 989 TFLOP/s
MFU (vs BF16 peak) = 415.0 / 989 = 42.0%
```

### Full MoE CUDA Graph + Gradient Accumulation (breakthrough)

The `moe` CUDA graph scope captures the **entire MoE layer** (router + dispatch + expert compute + combine)
in a single graph, but requires drop-and-pad mode (`--moe-expert-capacity-factor 1.5 --moe-pad-expert-input-to-capacity`).

Combined with gradient accumulation (GBS > MBS*DP), the optimizer step overhead is amortized:

| Config               | MBS | GBS | Grad Accum | Iter (ms) | tok/sec  | MFU       |
| -------------------- | --- | --- | ---------- | --------- | -------- | --------- |
| FP8 + full MoE graph | 4   | 32  | 1x         | 705       | 186k     | 42.5%     |
| FP8 + full MoE graph | 5   | 40  | 1x         | 853       | 192k     | 44.0%     |
| FP8 + full MoE graph | 4   | 64  | 2x         | 1,333     | 197k     | 45.0%     |
| FP8 + full MoE graph | 4   | 128 | **4x**     | **2,584** | **203k** | **48.1%** |
| FP8 + full MoE graph | 4   | 256 | 8x         | 5,069     | 207k     | 49.1%     |
| FP8 + full MoE graph | 4   | 512 | 16x        | 10,048    | 209k     | 49.5%     |

**BOTH TARGETS ACHIEVED: 211k tok/sec at 50.1% MFU** (MBS=5, GBS=320)

| Config                   | MBS   | GBS     | Accum  | Iter (ms) | tok/sec  | MFU       |
| ------------------------ | ----- | ------- | ------ | --------- | -------- | --------- |
| FP8 + full MoE graph     | 4     | 128     | 4x     | 2,584     | 203k     | 48.1%     |
| FP8 + full MoE graph     | 4     | 256     | 8x     | 5,039     | 208k     | 49.4%     |
| **FP8 + full MoE graph** | **5** | **320** | **8x** | **6,207** | **211k** | **50.1%** |
| FP8 + full MoE graph     | 4     | 384     | 12x    | 7,518     | 209k     | 49.7%     |

Validated on real clang code data (clang_commits_4k_v1, 9.86B tokens):
- GBS=128: 205k tok/sec, loss 11.95→3.95 in 20 iters
- GBS=320 (production): **211k tok/sec**, 50.1% MFU

### MFU Calculation (production config: MBS=5, GBS=320)
```
Iter time: 6,207 ms
Active params: ~3.13B (nheads=64 for FP8 alignment)
Tokens/iter: 320 * 4096 = 1,310,720
FLOPs/iter: 1,310,720 * 6 * 3.13B = 24.62 PFLOP
TFLOP/s/GPU: 24620 / 6.207 / 8 = 495.7
H200 BF16 peak: 989 TFLOP/s
MFU = 495.7 / 989 = 50.1%
```

### Key: MBS=5 > MBS=4 for MFU
MBS=5 achieves higher MFU than MBS=4 despite larger per-step time (776ms vs 630ms per micro-step)
because the 25% more work per kernel launch improves GPU utilization. The MBS=4 configs top out
at 49.4% MFU regardless of gradient accumulation.

### Production Training Run (500 iterations)
Trained on clang_commits_4k_v1 (9.86B tokens), single dataset:

| Iter | Loss      | Grad norm | tok/sec |
| ---- | --------- | --------- | ------- |
| 50   | 2.81      | 4.42      | 211.6k  |
| 100  | 1.91      | 4.05      | 209.6k  |
| 200  | 0.85      | 0.46      | 207.5k  |
| 300  | 0.66      | 0.20      | 207.8k  |
| 400  | 0.59      | 0.13      | 208.2k  |
| 500  | **0.569** | 0.108     | 207.4k  |

160k samples (655M tokens) processed. 5 checkpoints saved (100-500).
Zero NaN iterations, zero skipped iterations. Completely stable in FP8.

### Blended Dataset
- Fixed: `--split 100,0,0` (all data to train, no valid split avoids empty dataloader assert)
- Semantic (30%) + Commits (70%) blend working at same throughput (212k tok/sec)
- Recipe auto-adds `--split 100,0,0` when data_path is set

---

## 2026-04-10: NAM56R NeMo-Native Baseline on H200x8

### Environment Setup

**Machine:** `h200_1` (GCP `LOCATION_2`, `a3-ultragpu-8g`)
- 8x NVIDIA H200 (141 GiB VRAM each)
- CUDA Toolkit 13.2
- cuDNN 9.20.0.48

**Software stack (all cu132):**
- PyTorch 2.12.0.dev20260409+cu132
- Transformer Engine 2.13.0
- mamba-ssm 2.3.1 (pip, no Author Mamba3 module)
- flash-attn 2.8.3
- Megatron-LM (commit fd762549)
- cppmega 0.1.0

### Model Architecture (NAM56R 4.73B)

| Parameter         | Value                         |
| ----------------- | ----------------------------- |
| Pattern           | AEMEAEMEAEMR                  |
| Total layers      | 52 (13 A + 22 E + 13 M + 4 R) |
| Hidden size       | 3,584                         |
| FFN hidden size   | 18,944                        |
| Attention heads   | 56 (GQA 7:1 vs 8 KV heads)    |
| Head dim          | 64                            |
| Seq length        | 4,096                         |
| Vocab size        | 65,536                        |
| MoE experts       | 16 routed, top-k=4            |
| MoE FFN hidden    | 896 per expert                |
| MoE shared expert | 1,024                         |
| Mamba state dim   | 64                            |
| Mamba head dim    | 64                            |
| Mamba num heads   | 56                            |
| Mamba num groups  | 8                             |
| Total params      | ~4.73B                        |
| Active params     | ~3.03B (MoE sparse)           |
| Precision         | BF16                          |

### Parallelism Configurations Tested

#### Test 1: TP=2, SP=True, DP=4 (NeMo Nano v2 style)

| Run | micro_batch | GBS           | Iter time (ms) | tok/sec  | Memory/GPU |
| --- | ----------- | ------------- | -------------- | -------- | ---------- |
| A   | 4           | 32            | 1,450          | ~90,400  | 52 GiB     |
| B   | 8           | 64            | 2,597          | ~101,000 | 86 GiB     |
| C   | 4           | 128 (4 accum) | 5,600          | ~93,600  | 52 GiB     |
| D   | 16          | 128           | OOM            | -        | >141 GiB   |

**Conclusion:** TP=2 communication overhead limits throughput. ~90-101k tok/sec maximum.

#### Test 2: TP=1, PP=1, DP=8 (optimal for model that fits single GPU)

| Run          | micro_batch | GBS | Iter time (ms) | tok/sec  | Memory/GPU | MFU    |
| ------------ | ----------- | --- | -------------- | -------- | ---------- | ------ |
| E (28 heads) | 4           | 32  | 780            | ~168,000 | 88 GiB     | ~38.6% |
| F (56 heads) | 4           | 32  | 810            | ~161,800 | 88 GiB     | ~37.2% |
| G (56 heads) | 8           | 64  | OOM            | -        | >141 GiB   | -      |

**Conclusion:** TP=1 is 1.85x faster than TP=2 for this model size. ~162k tok/sec achieved.

### MFU Calculation (Test F - true NAM56R)

```
Iter time: 810 ms
Active params: 3.03B (MoE, top-4 of 16 experts)
FLOPs/token: 6 * 3.03B = 18.18 GFLOP
Tokens/iter: 32 * 4096 = 131,072
FLOPs/iter: 131,072 * 18.18G = 2.383 PFLOP
TFLOP/s/GPU: 2383 / 0.81 / 8 = 367.7
H200 BF16 peak: 989 TFLOP/s
MFU = 367.7 / 989 = 37.2%
```

### Key Findings

1. **TP=1 >> TP=2** for models that fit on a single GPU. The NAM56R 4.73B MoE model at BF16 uses ~88 GiB/GPU, well within H200's 141 GiB. No tensor-parallel communication needed.

2. **Throughput scales with batch, not TP.** The per-sample compute time is ~0.1 ms, dominated by kernel launch and memory bandwidth. Larger batches amortize this.

3. **micro_batch=4 is the max at TP=1.** Each GPU holds the full model weights (~31 GiB) + optimizer states (~31 GiB distributed) + activations (~26 GiB at micro_batch=4). micro_batch=8 OOMs.

4. **Loss converges quickly** on mock data: 11.8 -> 7.5 in 10 iterations with lr=3e-4 cosine schedule.

### Path to 200k+ tok/sec and 50%+ MFU

1. **FP8 precision** (NeMo standard): Halves memory for activations, allows micro_batch=8+. Expected ~1.5x throughput boost.
2. **CUDA graphs** (NeMo uses `cuda_graph_scope="full"`): Eliminates kernel launch overhead. Expected ~10-20% boost.
3. **Selective recomputation** (`recompute_granularity="selective"`): Trades compute for memory, enabling larger batches.
4. **Communication overlap** (`overlap-grad-reduce` already enabled): AllReduce during backward is already overlapped.

### Training Data

Downloaded from GCS to `/home/dave/cppmega-root/data/parquet/`:
- `clang_semantic_4k_v10`: 66 shards, 6.0 GiB (code with structure metadata)
- `clang_commits_4k_v1`: 104 shards, 18 GiB (commit diffs with metadata)

Conversion to Megatron binary format (.bin + .idx) in progress.

### Mamba-3 Features (CppMegaMamba3Mixer)

Created `cppmega/megatron/mamba3_mixer.py` — subclasses Megatron's `MambaMixer`,
adds qknorm + B/C bias on the SSD scan inputs while keeping ALL TE optimizations.

| Mode             | Iter time  | tok/sec  | Features                     |
| ---------------- | ---------- | -------- | ---------------------------- |
| nemo_native BF16 | **810 ms** | **165k** | Mamba-2 SSD + TE             |
| mamba3_te BF16   | 1,035 ms   | 127k     | + qknorm + B/C bias          |
| FP8 delayed      | 838 ms     | 156k     | FP8 overhead cancels benefit |
| author_dp (old)  | 39,800 ms  | 3.3k     | Broken TE integration        |

### Layer-Level Profiling (810ms breakdown)

| Component         | Layers | Time/layer | Total  | %   |
| ----------------- | ------ | ---------- | ------ | --- |
| Mamba (M+R)       | 17     | 5.2 ms     | 88 ms  | 11% |
| Attention+MLP (A) | 13     | 22.9 ms    | 298 ms | 37% |
| MoE (E)           | 22     | 19.3 ms    | 424 ms | 52% |

### MFU Analysis

- 37.2% MFU at 165k tok/sec (3.03B active params)
- 200k tok/sec requires 46% MFU (+24% improvement)
- Achievable with TE-scoped CUDA graphs or torch.compile (needs newer Megatron)
- NeMo Nano 9B at 60% MFU only does 88k tok/sec (3x more FLOPs/token)

### Files Created/Modified

**New files:**
- `cppmega/recipes/nam56r_nemo_recipe.py` - NeMo 3 Nano-style recipe configuration
- `scripts/remote_setup_bench.sh` - H200 bench machine setup
- `scripts/remote_sync_bench.sh` - Code sync to bench machine
- `scripts/remote_train_h200_nam56r_nemo.sh` - NeMo-style training launcher
- `scripts/data_prep_parquet_to_megatron.py` - Parquet-to-Megatron converter
- `scripts/remote_data_prep_bench.sh` - Remote data preparation
- `tests/test_nam56r_nemo_recipe.py` - Recipe unit tests (25 tests, all passing)
- `docs/changelog.md` - This file
