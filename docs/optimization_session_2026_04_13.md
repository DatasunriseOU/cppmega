# NAM56R Optimization Session -- 2026-04-13

**Model**: NAM56R 4.73B hybrid (27 Mamba3 + 9 DSA + 4 full MLA + 12 MoE), heads=32, hidden=4096, 52 layers
**Hardware**: H200x8 single-node (bench3 LOCATION_1, europe LOCATION_2), GB10 (for compile benchmarks)
**Branch**: `main` (commit `b28a9f1`)
**Previous baseline**: 240 TFLOP/s (PP=2 MBS=4, h=3584), 267 TFLOP/s golden config (FP8 tensorwise MBS=10)

---

## 1. Completed and Verified

### 1.1. Save Checkpoint Fix

| Item | Detail |
|------|--------|
| **Root cause** | `Mamba3._apply` guard kept D/dt_bias as fp32 while Float16Module cast the rest to bf16. Distributed optimizer cross-mapped optimizer states between bf16/fp32 buffers during save |
| **Fix** | Removed `_apply` guard entirely. D/dt_bias now bf16 after Float16Module. Forward uses `.float()` cast inline |
| **Verification** | save@step=10 and save@step=20 both report `successfully saved` on PP=1 and PP=2 |
| **Impact** | Unblocks all checkpoint-dependent workflows (resume, evaluation, export) |

### 1.2. Code Cleanup

Removed fallback paths and obsolete files to enforce the single DSA path (lemyx + IndexCache):

| Action | File |
|--------|------|
| Removed fallbacks | `cppmega/megatron/index_cache_patch.py` |
| Removed fallbacks | `cppmega/megatron/lemyx_dsa_warmup.py` |
| Deleted | `dsa_fp8_patch.py` |
| Deleted | `dsa_fp8_indexer.py` |
| Deleted | `dsa_tilelang_fused_kl.py` |
| Removed env var gates | `CPPMEGA_INDEX_CACHE`, `CPPMEGA_LEMYX_DSA` -- now always-on |

### 1.3. Architecture Change: heads=32, hidden=4096

Changed from heads=28 to heads=32 to satisfy three constraints simultaneously:

| Constraint | Requirement | heads=28 | heads=32 |
|------------|-------------|----------|----------|
| FP8 tensorwise | `heads % 8 == 0` | 28%8=4 FAIL | 32%8=0 OK |
| WGMMA tiling | `head_dim` aligned to tile | Misaligned | Aligned |
| lemyx kernel | `heads == index_heads` | Mismatch | Match |

52 layers confirmed (`NAM56R_DEPTH=52`). Divides evenly by 4 (VPP=4) and 2 (VPP=2).

### 1.4. NVIDIA Apex Installed

Apex installed from source with CUDA extensions on both H200 machines:
- bench3 (LOCATION_1, `/mnt/data/venv`)
- europe (LOCATION_2, `/home/dave/cppmega-root/cppmega-venv`)

### 1.5. Megatron Rebase

Rebased onto latest upstream dev. Cherry-picked PRs:

| PR | Description | Status |
|----|-------------|--------|
| #3674 | DSA absorbed MLA | Applied |
| #4268 | Delayed wgrad overlap with P2P backward | Applied |

Synced to: bench3, GB10, GCS.

### 1.6. DualPipeV Standalone Test

`deepseek-ai/DualPipe` package installed on all 3 machines (europe, bench3, GB10).

| Test | Result |
|------|--------|
| PP=2 stage mapping | 4 stages: rank 0 = stages (0,3), rank 1 = (1,2) |
| Loss correctness | Matches reference |
| Gradient correctness | Matches reference |

Integration code: `cppmega/megatron/dualpipev_schedule.py` (897 LOC). Implements:
- Stage splitting: 52 layers into 4x13 virtual stages
- Loss function wrapping
- Training step replacement via monkey-patch

### 1.7. Regional torch.compile Results (GB10)

Benchmarked individual submodules with `torch.compile` on GB10:

| Submodule | Speedup | Verdict |
|-----------|---------|---------|
| Data-dependent A computation | **5.93x** | COMPILE |
| Mamba3 pre-processing | **2.66x** | COMPILE |
| Mamba3 post-processing | **1.84x** | COMPILE |
| SiLU + gate multiply | **1.35x** | COMPILE |
| RMSNorm | 0.41x | DO NOT compile |
| RMSNormGated | 0.47x | DO NOT compile |
| MoE Router | 0.97x | DO NOT compile |
| MLA projections | 1.01x | DO NOT compile |

Patch file: `cppmega/megatron/mamba3_compile_patch.py` (423 LOC).

### 1.8. Throughput Results (h=4096 heads=32 config)

| Configuration | TFLOP/s | Notes |
|---------------|---------|-------|
| PP=2 VPP=2 EP=4, no compile | 193-194 | Save works, production-ready |
| PP=1 EP=4 MBS=8, no compile | 193 | No pipeline bubble, larger model per GPU |
| PP=2 MBS=4 (old h=3584) | 240 | Baseline before architecture change |

The h=4096 config is ~20% slower than h=3584 due to larger model size. Recovering this gap is the goal of DualPipeV, regional compile, and EP overlap work.

### 1.9. combined_1f1b Research

Flag: `--overlap-moe-expert-parallel-comm`. Requires `build_schedule_plan()` on the model object.

| Finding | Detail |
|---------|--------|
| GPTModel | Has `build_schedule_plan()` |
| MambaModel | Does NOT have it |
| Upstream PR | None exists for MambaModel support |
| Confirmed by | 6 independent search agents |

Blocker: must write `hybrid_schedule_plan.py` to add `build_schedule_plan()` for hybrid MambaModel (MoE layers reuse GPT `TransformerLayerNode`, Mamba layers use single opaque node). Estimated ~150-200 LOC.

---

## 2. In Progress

### 2.1. Regional Compile Integration

File: `cppmega/megatron/mamba3_compile_patch.py` (423 LOC, rewritten).

**Current blocker**: `CppMegaMamba3TE.forward` patch has `padding_mask` kwarg conflict with `te_checkpoint` + `torch.compile`. The `padding_mask=None` default in the patched signature is not propagated through TE's checkpoint wrapper.

**Fix in progress**: Add explicit `padding_mask=None` to the compiled function signature, bypassing TE checkpoint's kwargs forwarding.

### 2.2. build_schedule_plan for Hybrid Model

Writing `hybrid_schedule_plan.py`:
- MoE layers: reuse GPT `TransformerLayerNode` (has A2A comm annotations)
- Mamba layers: single opaque `ScheduleNode` (no overlappable comms)
- Remove `isinstance(GPTModel)` assert in Megatron's schedule builder
- Target: ~150-200 LOC

### 2.3. VPP=4 Test

Configuration: PP=2 VPP=4, 52/4=13 layers per virtual stage. Testing on bench3. Expected to reduce pipeline bubble vs VPP=2 (from ~24% to ~12% bubble).

### 2.4. lemyx Kernel heads=32

Kernel updated for new head count. Standalone test produced `cudaErrorLaunchFailure`. Root cause: TileLang JIT cache has stale compiled kernel for heads=28. Needs full recompilation with new head count.

---

## 3. Plan (Next Steps)

### 3.1. combined_1f1b EP Overlap for Hybrid Model

**Prerequisite**: `hybrid_schedule_plan.py` (section 2.2).
**Flag**: `--overlap-moe-expert-parallel-comm --delay-wgrad-compute`
**Expected**: Hide EP AlltoAll behind Mamba compute. AlltoAll currently ~8% of step time at EP=4.

### 3.2. DualPipeV Full Training Test

Wire `cppmega/megatron/dualpipev_schedule.py` into the training entrypoint. Test PP=2 with DualPipeV schedule.
**Expected**: Near-zero pipeline bubble (DualPipeV overlaps forward/backward of different micro-batches across pipeline stages).

### 3.3. Regional Compile on H200

Fix `padding_mask` issue, measure real throughput gain on H200.
GB10 showed **5.93x** on data-dependent A. Expect ~8% total model throughput improvement from compiling the 4 winning submodules.

### 3.4. PR #3116 (Seq1F1B)

30 files changed, conflicts with our codebase. Needs manual merge. Enables sequence-level pipeline parallelism for long context training.

### 3.5. Selective FP8 MoE

File: `cppmega/megatron/selective_fp8_moe_patch.py`.
Needs debugging (env var propagation through Megatron's FP8 context manager). MoE GEMMs represent ~15% of compute.

### 3.6. Mamba SSM Kernel Optimization

SSM = 34.5% of GPU time, 255 registers, 6.25% occupancy.

| Priority | Optimization | Expected Impact |
|----------|-------------|-----------------|
| P1 | TMA loads (replace manual gmem->smem) | +15-20% bandwidth |
| P2 | State checkpoint (reduce recompute in bwd) | -20% SSM bwd time |
| P3 | Register pressure reduction (target <128 regs) | 2x occupancy |
| P4 | Fused elementwise in SSM kernel | -5% total |
| P5 | Multi-chunk pipelining | +10% overlap |

---

## 4. Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **heads=32, hidden=4096** | Production config. FP8/WGMMA/lemyx all compatible |
| **52 layers** (NAM56R_DEPTH) | Divides by 4 (VPP=4) and 2 (VPP=2) |
| **Single DSA path** | lemyx (warmup) + IndexCache (production). No fallbacks |
| **No _apply guard** | D/dt_bias stay bf16 after Float16Module. `.float()` in forward |
| **DualPipeV preferred** | Over standard 1F1B PP for pipeline parallelism |
| **combined_1f1b for EP overlap** | At PP=1 once build_schedule_plan ready |

---

## 5. Key References

| Reference | Description |
|-----------|-------------|
| `cppmega/megatron/dualpipev_schedule.py` | DualPipeV integration (897 LOC) |
| `cppmega/megatron/mamba3_compile_patch.py` | Regional compile patch (423 LOC) |
| `cppmega/megatron/index_cache_patch.py` | IndexCache integration (cleaned) |
| `cppmega/megatron/lemyx_dsa_warmup.py` | lemyx DSA warmup (cleaned) |
| `cppmega/megatron/selective_fp8_moe_patch.py` | Selective FP8 for MoE layers |
| `scripts/remote_smoke_h200_dsa_9_4_m.sh` | H200 smoke test script (564 LOC) |
| Megatron PR #4268 | Delayed wgrad overlap (cherry-picked) |
| Megatron PR #4099 | MambaModel to HybridModel rename (open, not applied) |
| Megatron PR #3116 | Seq1F1B (30 files, not applied) |
| Megatron Issue #1810 | Deadlock with A2A overlap on uneven pipeline stages |
| `deepseek-ai/DualPipe` | DualPipeV reference implementation |
| DeepSeek-V3 GB200 guide | Production EP tuning reference |

---

## 6. Machine Status

| Machine | Location | Env Path | Status |
|---------|----------|----------|--------|
| bench3 (H200x8) | LOCATION_1 | `/mnt/data/venv` | Primary test machine. Apex installed. Megatron rebased |
| europe (H200x8) | LOCATION_2 | `/home/dave/cppmega-root/cppmega-venv` | Secondary. Apex installed. Megatron rebased |
| GB10 | local | N/A | Compile benchmarks only. DualPipe installed |

All machines: torch 2.12+cu132, mamba_ssm 2.3.1, TE 2.13, megatron-core 0.18.
