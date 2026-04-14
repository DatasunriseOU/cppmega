# FP8 Research Session — 2026-04-14

Comprehensive parallel research into FP8 paths across NAM56R for Mamba3 SSM, MLA/DSA attention backward, weight storage, and mixed-dtype GEMM. 15 research/exec agents, 3 machines (europe, bench3, GB10), ~3h wall-clock.

## Executive summary

| Path | Status | Verdict |
|---|---|---|
| FP8 MoE Linear GEMMs | ✅ live | Works; selective_fp8_moe + DSA coexist (verified GB10) |
| FP8 MLA/DSA forward (zero-copy) | ✅ live | SparseMLA_FP8 via Float8Tensor._data, 0% net on our compute mix |
| FP8 MLA/DSA backward (E5M2 dO) | 🧪 R&D | TileLang patch feasible (~110 LOC, not 30 as earlier estimated), gain ~0.7%, risk grad_norm 100× |
| FP8 SSM scan | ❌ **dead path** (post-full-port) | Full MIMO port: 0.73-0.91× GB10, ~1.07× H200 projection. Kernel latency-bound, not GEMM-bound |
| FP8 param-gather | 🧪 testing | -5 GiB memory (enables MBS=10+); throughput neutral per prior docs |
| FP8 model replica (TE fp8_model_init) | ❌ not taken | PR #2245 is FSDP2-specific, we use Megatron dist-opt |

## Research synthesis (9 parallel web research agents)

### FP8 Mamba SSM — literature = empty niche
Four independent agents (perplexity deep / exa deep / HF papers / brave+tavily) all confirmed:
- **No OSS project trains SSM recurrence in FP8.** Nemotron-H, Bamba, Jamba, Zamba — all keep scan in BF16/FP32, only quantize surrounding Linear.
- Jamba literally excludes Mamba entirely: `llm_int8_skip_modules=["mamba"]`.
- Nemotron-H FP8 recipe: "FP8 quantizes MoE GEMMs and Mamba **GEMMs**, state cache FP16, **scan is not FP8**."
- Only PTQ work exists: Quamba/Quamba2 (ICML 2025, W8A8/W4A8 inference), MambaQuant.

**Technical obstacles** (per perplexity reasoning):
- Recurrence `h_{t+1} = A_t*h_t + B_t*x_t` compounds multiplicatively across T steps (unlike attention's softmax-normalized additive aggregation).
- Failure order in E4M3: `dt` discretization (softplus steep near 0) → input `x_t` (causally contaminates all later state) → state `h` storage cast between scan steps.
- Per-chunk amax tractable (SSD chunks 256) but scan is serial in `h` → single scale per chunk, can't adapt mid-chunk.
- SSM is **register/latency bound** (255 regs, 6.25% occupancy per `reference_mamba_ssm_optimization_plan.md`), FP8 payoff is bandwidth/capacity, not flops.

### FP8 attention backward — no OSS reference exists
Three agents (exa deep / exa github / perplexity reasoning) converged:

- **FlashAttention 3/4**: FP8 fwd only, bwd DOES NOT EXIST. `DTYPE_MAP_BWD = {fp16, bf16}`. Tri Dao (#1848, #1420, #1169): "no plans, accuracy open problem".
- **CUTLASS hopper_fmha**: bwd fp16 only.
- **TileLang deepseek_v32 sparse_mla_bwd**: `assert dtype == T.bfloat16`.
- **flashinfer**: inference-only, no bwd at all.
- **Only production FP8 attn bwd path**: NVIDIA TE cuDNN SDPA (`fused_attn_bwd` via `META_DO/DQKV/S/DP` meta slots). But **does not support sparse top-k** → useless for our DSA.

**Hardware constraint** (brave+tavily): Hopper WGMMA does NOT support FP8×BF16. PTX `wgmma.mma_async` allows `{e4m3,e5m2}×{e4m3,e5m2}` and `bf16×bf16` but no mixing. All "mixed FP8/BF16" GEMM = software dequantize before MMA.

**Canonical dO→E5M2 pattern** (perplexity reasoning + exa github):
- Host-side Python cast via `Float8Quantizer(scale, amax, fp8_dtype=E5M2)`
- amax from **DelayedScaling** history (length 16, non-blocking) — do NOT roll own amax
- `dS = P*(dP-rowsum)` requantization needs its **own** E5M2 scale entry (missed in earlier plan)
- Fused prologue alternative via `T.reduce_absmax` + `T.cast(x*scale, "float8_e5m2")` — only compatible with CurrentScaling recipe

**Verdict**: writing OSS-first FP8 attn bwd = ~110 LOC across 3 files + DelayedScaling history integration (high risk). Gain: attention is 4% of compute × 15-20% GEMM speedup = ~0.7% total. Not worth it for now.

### FP8 weight storage (fp8_model_init / --fp8-param-gather)
One agent mapped the ecosystem:

- **NVIDIA internal**: production via Megatron `--fp8-param-gather` (PR #880, PR #1121, PR #1544). Used in Nemotron, Mixtral recipes.
- **External adoption**: limited. Most keep BF16 replica + FP8 GEMMs.
- **Memory savings**: 28-39% for GPT-175B, 11-14% for 1.5B. Our 4.73B ≈ 15-20%. Comes from eliminating BF16 all-gather buffer in dist-opt, not from halving per-layer weight bytes.
- **Throughput**: Mixtral 8×22B = 1.26-1.30× on 128×H100.
- **Dist-opt compat**: ✅ via `--fp8-param-gather`. Master stays FP32.
- **FSDP2 compat**: PR #2245 (Oct 2025, too new). **Not useful for us** — we use Megatron dist-opt.
- **MoE + EP**: supported; expert Linear weights convert to Float8Tensor through `fp8_model_init()`.
- **Custom modules** (TileLang SparseMLA, cppmega Mamba3): automatically stay BF16 (created outside `fp8_model_init` context).

## Empirical results

### GB10: FP8 Mamba SSM on TileLang **works**
Branch: `fp8-mamba-ssm-exploration` @ 9270506 (pushed to origin).

On sm_121 (no WGMMA, no tcgen05, no TMEM):

| Kernel | BF16 | FP8 e4m3fn | Speedup |
|---|---|---|---|
| Single GEMM Q@K^T | 18.4µs | 13.6µs | **1.36×** |
| 2-GEMM scan-loop (4 chunks) | 25.3µs | 15.2µs | **1.66×** |

Per-token scaled single GEMM rel err: 3.6%.

**Constraint**: `T.gemm(A, B, C)` requires `A.dtype == B.dtype`. Hybrid FP8×BF16 fails at compile time. Full Mamba3 MIMO FP8 requires all 3 GEMMs + `qk_intrachunk_shared` intermediate in FP8.

**H200 implications**: Same source path lowers to WGMMA FP8 on sm_90 — higher ceiling without code change.

**Initial hypothesis**: SSM = 34.5% GPU time. At 1.5× speedup on 34.5% compute → +17% total → **339 TFLOP/s = 34% MFU** from current 289.

### FP8 Mamba SSM full port — hypothesis REFUTED (commit c0c6bd1 on branch)

Full `mamba3_mimo_fwd.py` ported to FP8 (478 LOC, 5 GEMMs, all operands e4m3fn, accumulator fp32):
- GB10 sm_121: **0.73-0.91×** vs BF16 — **net slowdown 10-27%**
- H200 WGMMA projection: **~1.07×** — marginal
- Numerics OK (rel_err 3-10%, stable on amp 0.5-2.0)
- Cross-compiles sm_90a, emits `tl::wgmma_ss<Float8_e4m3, Float8_e4m3, Float32, 64, 64, 32>`

**Root cause**: MIMO kernel is NOT GEMM-bound. Rotary / trap-scaling / SEGSUM mask / state-update / diagonal-reduction / Z gate / D term overhead dominate the kernel time. FP8 cast-before-GEMM overhead exceeds the modest GEMM speedup. Consistent with `reference_mamba_ssm_optimization_plan.md` (255 regs, 6.25% occupancy → register/latency-bound).

**Lesson**: single-GEMM microbenchmark (1.66×) did NOT predict full-kernel performance. Always measure on full kernel before projecting gains.

**Decision**: FP8 Mamba SSM = dead path for NAM56R production. Branch kept as R&D artifact. Full numerical tables in `docs/fp8_mamba_ssm_exploration_notes.md` on branch `fp8-mamba-ssm-exploration`.

### GB10: FP8 attn bwd piggyback exploration
Branch: `fp8-bwd-piggyback-exploration` @ d91f5a9 (GB10 local, not pushed).

- `Float8Quantizer` works standalone with `fp8_dtype=E5M2`, `amax` updates on call.
- Direct E5M2 dO into existing `sparse_mla_bwd_fp8` fails: `preprocess_kernel input dO dtype expected bfloat16, got float8_e5m2`.
- Dequant E5M2→BF16 then bwd: OK, rel_err dq=5.9e-2, dkv=4.3e-2 vs pure BF16 baseline.
- TileLang does NOT have cuDNN codegen (`SUPPORTED_TARGETS = {auto, cuda, hip, metal, llvm, webgpu, c, cutedsl}`) — TVM relay `partition_for_cudnn` is graph-level, doesn't apply to TileLang kernels.
- TE `fused_attn_bwd` cannot handle sparse top-k (no gather-indices `attn_mask_type`) → cannot piggyback cuDNN backend either.

### Fix: Selective FP8 MoE + DSA coexistence
Commit f208e15 (pushed to main):
- **Patch 9b** in `apply_dsa_cg_patches.py`: removes a stray `query.dequantize() + key.dequantize()` pair that had drifted into installed dsa.py, killing zero-copy FP8.
- Selective FP8 MoE layer summary now uses `print` (visible in WARNING logs).
- Defensive check warns if DSA Patch 9 is missing.

### Debunked: DualPipeV 205 TFLOP/s
The "205 TFLOP/s PP=2 DualPipeV" baseline cited in earlier summaries **does not exist**. Grep over bench3 logs found no such measurement. `dualpipev_schedule.py` (167 LOC) exists but:
- No script imports it
- No env var activates it
- Architectural contradiction: asserts `PP==2` while asserting `len(decoder.layers) == 52` per rank (PP=2 gives 26 layers/rank)
- Requires ~200 LOC integration work to actually wire up

Memory updated: `project_dualpipev_unwired.md`.

### bench3 vs europe unexplained 100 TFLOP/s gap
Same arch:
- europe: PP=1 MBS=8 BF16 no-CG = 289 TFLOP/s
- bench3: PP=2 VPP=2 EP=4 MBS=4 FP8 = 190-192 TFLOP/s

Configs aren't apples-to-apples but gap too large for config alone. Candidates: NVLink bandwidth, thermals, driver/CUDA versions, NCCL env, NUMA pinning, network topology, CPU freq scaling. Closing half this gap (50 TFLOP/s) dwarfs any algorithmic micro-optimization on the table. Memory: `project_bench3_vs_europe_delta.md`.

## Integration changes (pushed to origin/main)

- `ba5ab30` — `CPPMEGA_FP8_PARAM_GATHER=1` env-gate in `scripts/remote_smoke_h200_dsa_9_4_m.sh` + README table entry
- `f208e15` — Patch 9b + selective_fp8_moe log visibility
- `fp8-mamba-ssm-exploration` branch pushed

## Pending tests (in flight)

- **bench3**: `--fp8-param-gather` at MBS=8/10/12 (agent a30418797fd66a63b)
- **europe**: `combined_1f1b` EP overlap (agent a34b50e71bf1c3a2d, hybrid_schedule_plan.py modified locally)

## Next steps (post-current tests)

1. If `--fp8-param-gather` + MBS=10/12 gives clean training: commit as default, update golden config.
2. If FP8 Mamba SSM GB10 exploration looks convergent: port to full mamba3_mimo_fwd on bench3 H200 (WGMMA FP8 path), target 35% MFU.
3. If combined_1f1b gains >2%: commit hybrid_schedule_plan.py modifications.
4. Investigate bench3 vs europe 100 TFLOP/s system gap (NVLink/thermal/driver).

## Sources (key references)

- TE cuDNN FP8 bwd: `transformer_engine/pytorch/cpp_extensions/fused_attn.py`, `transformer_engine/common/util/cast_kernels.cuh`
- Megatron `--fp8-param-gather`: `arguments.py:1739-1741`, `distrib_optimizer.py:376-409`, `fp8_utils.py:612`
- Nemotron-H FP8 recipe: NVIDIA tech report (Mamba GEMMs FP8, state cache FP16, scan BF16)
- FlashAttention-3 paper §3.3: blockwise FP8 attention (fwd only)
- TileLang `act_quant_kernel` reference: DeepSeek-Math-V2 `inference/tilelang_kernel.py`
- TileLang issue #1199: "gradient between fp8 and bf16 requires additional consideration" (our kernel family)
- TVM FP8 codegen: PR #16548 (E4M3/E5M2 cvt intrinsics, no fused amax+scale+cast)
