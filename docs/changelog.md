# NAM56R NeMo Migration Changelog

## 2026-04-14: FP8 paths exploration + Mamba3 MIMO P1 + pipeline topology closures

Multi-stream session across europe H200, bench3 H200, and GB10 sm_121.
Established production config 289 TFLOP/s on europe (PP=1 EP=4 MBS=8 BF16
no-CG) and an earlier 253 TFLOP/s bench3 waypoint on the old PP=1 EP=4 MBS=8
FP8 topology, closed multiple exploration directions, shipped infrastructure
for future work. Canonical current production numbers live in
`docs/production_status.md`; bench3's active production record is 268 TFLOP/s
on PP=1 EP=8 MBS=10 v3 with the Liger `reduction="mean"` broadcast workaround.

### Production records at the time (historical; see `docs/production_status.md` for current canon)

- **europe PP=1 EP=4 MBS=8 BF16 no-CG** = **289 TFLOP/s / ~9,250 tok/sec/GPU** — production gold
- **bench3 PP=1 EP=4 MBS=8 FP8 tensorwise** = **253 TFLOP/s / ~8,100 tok/sec/GPU** — historical pre-v3 bench3 waypoint; superseded by the current 268 TFLOP/s EP=8 MBS=10 v3 production config in `docs/production_status.md`
- europe PP=2 VPP=2 EP=4 MBS=4 MTP=2 = 193 TFLOP/s — standard PP=2 baseline (confirms)

### FP8 direction audit

**Dead paths (empirically refuted)**:
- FP8 Mamba SSM MIMO: full 478-LOC port gives **0.73-0.91× on GB10 sm_121** and **0.45-0.51× on H200 sm_90** (2× slower!). Kernel is NOT GEMM-bound — rotary/trap-scaling/SEGSUM/state-update dominate. Cast-before-GEMM overhead > modest FP8 GEMM speedup. Branch `fp8-mamba-ssm-exploration` @ c0c6bd1 kept as reference port (numerically correct). Memory: `reference_fp8_mamba_ssm_dead_path.md`.
- `--fp8-param-gather`: net-neutral (-0.5% noise). Saves only 2.6 GiB (not 5 as hypothesized) because custom BF16 modules (TileLang SparseMLA, Mamba3) sit outside `fp8_model_init` context. Env gate kept as safe option, default OFF. Memory: `reference_fp8_param_gather_net_neutral.md`.
- FSDP2 FP8 params (PR #2245): not our path (we use Megatron dist-opt), dismissed.

**Live paths**:
- FP8 MoE grouped-GEMM: `CPPMEGA_SELECTIVE_FP8_MOE=1` coexists with DSA (both stay on when set, verified by GB10 agent). Commit `f208e15` adds Patch 9b that removes a stray `query.dequantize()` pair that had drifted into installed dsa.py (killing zero-copy FP8 by re-quantizing BF16 per-token).
- FP8 attention **backward R&D** on branch `fp8-bwd-piggyback-exploration` @ 4d79332: working E5M2 dO path via TE `Float8Quantizer`, rel_err dq=1.56e-02 / dkv=1.40e-02 on GB10, 5-iter smoke stable. Env-gated `CPPMEGA_SPARSEMLA_FP8_BWD=1`, default OFF. Convergence validation on H200 pending.

### Mamba3 MIMO P1 — TMA + warp specialization

- `apply_mamba3_mimo_p1_patches.py` commits `4f115ea` → `d80bf9c`: flip `TL_DISABLE_TMA_LOWER/WARP_SPECIALIZED` in upstream mamba_ssm kernels + propagate `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True`.
- **Discovered bug**: inserting lines into patched file breaks `inspect.getsource` via `co_firstlineno` desync when `import mamba_ssm` auto-imports Mamba3 BEFORE `apply_all()` runs. Fixed with line-count-preserving merge onto FAST_MATH line. Memory: `reference_py_patch_line_shift_bug.md`.
- **Race fix**: 8-rank concurrent writes caused `IndentationError` from torn reads. Added `LOCAL_RANK=0 + fcntl.LOCK_EX + sentinel + atomic writes` pattern. Non-rank-0 ranks `LOCK_SH + poll` (120 s deadline).
- **bench3 selective-fwd P1 result**: **wash** (183.016 → 183.005 TFLOP/s, -0.006%). Fwd kernel not the bottleneck at MBS=8; 20-30% fwd speedup ≈ +1% total, below measurement noise. Env gate kept default OFF.
- **Full P1 (fwd + bwd + bwd_bwd)** blocked by TileLang TMA layout bug: bwd kernels use rank-3 smem descriptors, `LowerBulkCopy` asserts `InputDim == 2`. Memory: `reference_p1_blocked_tilelang_tma_layout.md`.
- **TMA layout fix** on branch `tma-layout-fix-3d-to-2d` @ `31dc695`: flattens `qk_dot_shared [chunk, R, R] → [chunk, R*R]` and `Q/K [B, S, R, G, N] → [B, S*R, G, N]`. Correctness verified GB10: 14 gradient tensors rel_err 0.0038-0.0116. H200 perf measurement in progress.

### Pipeline topology closures

- **DualPipeV integration**: commit `06269f0` adds Megatron PP=1 + carved 2-rank DualPipe group via `dist.new_group`, `apply_dualpipev_patch.py` hooks `setup_model_and_optimizer`. Env-gated `CPPMEGA_DUALPIPEV=1`. Experimental — **not measurable with EP>1** because V-shape puts ranks at different layers simultaneously while DeepEP A2A requires synchronized EP peers (deadlock at `deep_ep/buffer.py:97`).
- **combined_1f1b closed** (`reference_combined_1f1b_dead_for_nam56r.md`): OOM at every PP=2 MBS down to MBS=1 GBS=32 (overlap holds ~40 GiB extra pipe buffers + 95 GiB baseline > 141 GiB). PP=4 impossible (52 layers don't divide PP*VPP for VPP>1). Infrastructure kept (commit `13e2d7d` fixes MTP bypass + IdentityOp.backward_dw + MambaStack.final_layernorm). Env gate `CPPMEGA_EP_OVERLAP=1`, default OFF.
- **CP (Context Parallel) closed** (`reference_cp_blocked_by_custom_mixers.md`): all three custom mixers (CppMegaMamba3TE, dsa_sparse_*, mla_shared) lack `cp_group` plumbing. Multi-week port AND would be net-negative because Mamba CP = head-parallel = TP-equivalent (3.2× Mamba slowdown per TP=2 memory). Only reopens at 128k extension phase.

### Integration quality improvements

- **libnvrtc RTLD_GLOBAL workaround** (`39cb474`): force-load libnvrtc.so.13 at cppmega sparse_mla_ops import to prevent TileLang from aborting with "libnvrtc symbols not found globally" when compiling a second kernel variant in the same process. Candidate for upstream TileLang issue.
- `docs/long_context_roadmap.md`: documents 4k → 16k → 128k thresholds. SWA switch at seq > 16k. MLA `window_size` plumbing is a ~5 LOC add-on when needed (not preempted).
- `plan.md`: 10-hour optimization block with hard rules (no external PRs without explicit user approval, no silent fallbacks, CG must be OFF at PP=1).
- `upstream_prs/07_mamba3_mimo_3d_to_2d_smem_refactor.md` + `08_tilelang_tma_bulk_copy_3d_smem_issue.md` — drafted locally, **not posted** (hard rule: external PRs require explicit user approval).

### Hard-won knowledge (memory entries added)

- `reference_py_patch_line_shift_bug.md` — inserting lines breaks inspect.getsource
- `reference_mamba_ssm_reinstall.md` — `MAMBA_FORCE_BUILD=TRUE pip install ... .` (not raw `--force-reinstall`)
- `reference_gb10_bwd_bwd_blocker.md` — FIXED via `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE`
- `project_bench3_vs_europe_delta.md` — gap was PP topology, not hardware; real = 13%
- `project_dualpipev_unwired.md` → integrated but EP-incompatible
- `reference_fp8_param_gather_net_neutral.md`
- `reference_fp8_mamba_ssm_dead_path.md`
- `reference_combined_1f1b_dead_for_nam56r.md`
- `reference_p1_blocked_tilelang_tma_layout.md`
- `reference_cp_blocked_by_custom_mixers.md`

### Honest reality check on stretch targets

User set aspirational "250k tok/sec + MFU > 50%" goal via test-loop:
- 250k tok/sec on 8×H200 = ~982 TFLOP/s = **99% MFU** = essentially at hardware peak; not achievable on current architecture in 10 hours or realistically months without CUTLASS-level kernel rewrites
- Realistic ceiling this session: **31-35% MFU** (~105-110k tok/sec) if P1 full lands
- Documented honestly in `plan.md`

### Europe FP8 sweep — all paths regress (2026-04-14 night)

Tested 3 FP8-related configs on europe vs BF16 MBS=8 baseline 289 TFLOP/s:

| Config | TFLOP/s | Δ |
|---|---|---|
| BF16 MBS=8 EP=4 baseline | 289 | — |
| EP=8 v3 (take-3) | 262 | -9.3% |
| FP8 tensorwise MBS=10 | 190 | -34% |
| BF16 + CPPMEGA_SELECTIVE_FP8_MOE=1 | 247 | -14% |

**Pattern**: every non-baseline config on europe regresses. Europe's
NVLink/NVSwitch fabric baseline is already fast enough that the amax
recalibration cost + EP=8 fanout overhead exceed any FP8 GEMM speedup.

**Decision**: europe ships pure BF16 MBS=8 EP=4 (no FP8, no EP=8).
Memory: `reference_europe_fp8_all_paths_regress.md`.

### Bench3 FP8 + CPPMEGA_SELECTIVE_FP8_MOE=1 (2026-04-14 night)

Tested gate additive to FP8 tensorwise baseline: **267.7 TFLOP/s =
identical to 268.0 baseline** (σ<0.5 within noise). Gate is a no-op
when global FP8 tensorwise is active (MoE GEMMs already FP8 via TE).
Memory: `reference_fp8_moe_gate_net_neutral.md`.

### FP8 tensorwise MBS=10 cross-machine (late night — 2026-04-14)

Ran FP8 tensorwise MBS=10 GBS=80 tests on both machines simultaneously to
resolve the 2026-04-13 golden config 273 TFLOP/s claim:

| Machine | Config | Iters | TFLOP/s | Peak alloc | Val PPL |
|---|---|---|---|---|---|
| **bench3** | v3 EP=8 MBS=10 FP8 | 30/30 | **268.0** (σ<0.5) | 115.3 GiB | 207 |
| europe | v1 EP=4 DP=2 MBS=10 FP8 | 30/30 | 190.5 (σ=13) | — | — |

**Bench3 new record = 268 TFLOP/s** (+4.3% vs BF16 EP=8 v3 baseline of 257).
Ship as bench3 production config. Matches 2026-04-13 "golden config" claim
exactly — turns out that claim was specifically for bench3, not europe.

**Europe FP8 = REGRESSION (-34% vs BF16 MBS=8 = 289)**. High per-iter
variance (170-213 TFLOP/s range, σ=13) suggests amax recalibration churn
overhead exceeds FP8 GEMM speedup on europe's fabric. Keep europe on BF16.

Memory entries: `reference_fp8_mbs10_bench3_wins.md`,
`reference_fp8_mbs10_europe_regression.md`.

### Europe EP=8 v3 take-3 (late evening confirmation)

Third europe attempt with GQA patch restored + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:

- **25/25 iters clean**, steady state **262.0 TFLOP/s** (σ ≈ 0.1), val PPL 16.7
- Peak alloc **108.5 GiB** / reserved 117 GiB per rank (well within 140 GiB budget)
- **-9.3% regression vs europe PP=1 EP=4 baseline (289 TFLOP/s)**

**Verdict**: EP=8 v3 is machine-specific. Ship on bench3 (+1.6% win), keep EP=4 on europe. Prior OOM on take-2 was env drift (GQA patch missing), not a memory pressure issue. With patch restored, memory is healthy but throughput regresses — A2A cost scales badly with EP=8 on europe's NVLink fabric, and grouped-GEMM per-rank tile count drops from 4 local experts to 2. Memory updated in `reference_ep8_v3_machine_specific.md`.

### Open (at session end)

- Europe full P1 + TMA fix measurement (agent `ab0cdd07a098d108a`) — if wins ≥3%, merge `tma-layout-fix-3d-to-2d` and ship. Cross-machine validation on bench3 after.
- P2 post-rotary Q/K + PsiV cache — design ready, ~10-15% bwd_bwd saving, ~1% total. Deferred pending P1 result.
- P3 register split 255→130 — design ready (`docs/mamba3_mimo_p3_register_split_design.md`), ~1% total, week+ work. Deferred.
- FP8 attention backward on H200 — branch `fp8-bwd-piggyback-exploration` @ 4d79332 pending convergence validation after GB10 5-iter smoke stable.

---

## 2026-04-12: DSA/TP/recompute optimization session

Major optimization session covering TP=2 investigation, DSA 9+4 memory
optimization, selective recompute root cause discovery, and Blackwell
feature assessment.

### TP=2 investigation (Streams B, B v2)

Wrote `CppmegaMamba3TPMixer` (589 LOC), TE-native TP-aware Mamba-3 mixer
following Megatron `MambaMixer` pattern. TP=1 vs TP=2 numeric parity
passes (max_abs=1.5625e-2 bf16). Found and fixed B/C layout bug
(upstream `(r,g,n)` -> `(g,r,n)` for TP>1) and `angle_proj` SP backward
bug (`tensor_parallel_output_grad=False` -> `True`).

**TP=2 throughput: 34,672 tok/sec = 3.2x slower than TP=1 (112k
baseline).** Confirmed by both v1 and v2 runs. Root causes: collective
overhead + compute bandwidth-bound + PP=2 VPP=2 is more efficient
topology. Verdict: TP>1 is net loss for NAM56R Mamba-3 MIMO on
single-node H200x8.

### DSA 9+4 permanent attention layout

User decision: 13 A-layers = 9 DSA + 4 full MLA. A-ranks DSA:
`[1,2,3,5,6,7,9,10,11]`, MLA: `[0,4,8,12]`. Env var:
`CPPMEGA_DSA_A_LAYER_RANKS="1,2,3,5,6,7,9,10,11"`. Mechanism wired via
`CppMegaSelectiveAttentionLayer` in `nam56r_full_spec.py`.

### DSA memory optimization saga

- Stream D v1: DSA 9+4 BF16 OOM at PP=2 (136 GB, stage 1 MoE activation)
- Stream E: FP8 indexer port from DeepSeek V3.2 via `torch._scaled_mm`,
  9.3-13.4x peak delta reduction, saves ~26 GB stage 0 forward
- Stream G: Backward FP8 cleanup (indexer-only 69.5% savings, full-path 0.7%)
- Stream D v2: FP8 indexer applied, stage 0 OK, stage 1 OOM (MoE + MTP=2)
- Stream J: 4-variant sweep (FP8, +MoE recompute, +MTP redistrib, PP=4) ALL OOM
- Stream L/M: EP=2/EP=4 + loss_coeff==0 gate, ALL OOM
- Real bottleneck: `unfused_dsa_fn` at `dsa.py:920` materializes 7.0 GiB
  per DSA layer; `sparse_dsa_fn` written (gather-scatter, ~250x reduction)
- Ready-made kernels found: TileLang `sparse_mla_fwd.py`, NVIDIA PR #3674,
  `fla-org/native-sparse-attention`, `lemyx/tilelang-dsa`, PR #4039

### ROOT CAUSE: No selective recompute (commit f4f192c)

Memory diagnostic: 99.7 GB of 119.8 GB per rank = ACTIVATIONS with NO
recompute. nanochat uses `recompute_granularity="selective"` by default;
cppmega never had it. Fix: `--recompute-granularity selective
--recompute-modules moe_act` added to all launchers.

### Blackwell features (Stream C)

GB10 NAM56R-half real-data baseline: 4303.8 tok/sec (first honest
measurement). 5 Blackwell features tested, all blocked. Modal B200 DSA
indexer bench: FP8 11.4% slower than BF16.

### Environment fixes

- bench3 SSH IP updated (H200_1_IP -> H200_1_IP)
- europe: git SSH key + fresh clone + github auth
- europe: zombie cuTe DSL bench killed, kernel patched for GQA G<H

### New files

- `cppmega/megatron/cppmega_mamba3_tp_mixer.py` (589 LOC)
- `cppmega/megatron/dsa_fp8_indexer.py` (FP8 + head-streaming)
- `cppmega/megatron/dsa_fp8_patch.py` (3-tier monkey-patch)
- `cppmega/megatron/dsa_sparse_attention.py` (gather-scatter sparse)
- `cppmega/megatron/dsa_splitk_indexer_loss.py` (PR #4039 port)
- `cppmega/megatron/dsa_tilelang_fused_kl.py` (lemyx port)
- `cppmega/megatron/memory_debug.py`, `fp8_activations.py` (from nanochat)
- `cppmega/megatron/tilelang_sparse_mla/` (from tilelang examples)
- `tests/test_cppmega_mamba3_tp_mixer.py` (13 tests)
- `tests/test_dsa_fp8_indexer.py` (11 tests)
- `tests/test_dsa_splitk_indexer_loss.py` (6 tests)
- `tests/test_dsa_tilelang_fused_kl.py` (17 tests)
- `scripts/modal_dsa_indexer_bench.py` (804 LOC)
- Multiple launcher scripts for DSA/EP/TP/grid sweep

## 2026-04-11: M²RNN Triton kernel libdevice.tanh + smoke14

Replaced the manual stable-tanh in the fused Triton M²RNN kernel
(`cppmega/megatron/m2rnn_triton.py`) with the hardware `tanh.approx.f32`
PTX instruction via `tl.inline_asm_elementwise`, and eliminated the
per-step `h_new` recompute in the backward kernel by saving the
pre-gate `h_new` candidate from the forward pass.

**Why inline PTX and not `libdevice.tanh`:** Triton ships
`triton.language.extra.libdevice.tanh` which was expected to lower to
the SFU `tanh.approx.f32` op. On bench3 H200 with Triton 3.7.0 it maps
instead to `__nv_tanh` from the cuda libdevice bitcode (a software
polynomial), measured ~22% *slower* than the prior manual stable-tanh
formula.  Switching to `tl.inline_asm_elementwise` with the raw
`tanh.approx.f32 $0, $1;` instruction gives a true 1-SFU-op tanh.

**H200 microbench at NAM56R dims (B=2 S=4096 H=8 K=64 V=16, bf16):**

| kernel variant                                            | fwd (ms) | fwd+bwd (ms) |
| --------------------------------------------------------- | -------: | -----------: |
| manual stable-tanh (smoke13)                              |     4.82 |        16.18 |
| `libdevice.tanh` (polynomial)                             |     5.88 |        19.64 |
| inline PTX `tanh.approx.f32`                              |     3.97 |        15.08 |
| + bwd loads saved `h_new`                                 |     3.92 |        12.95 |
| num_warps=2 (rejected)                                    |     6.76 |        17.44 |
| num_warps=8 (rejected)                                    |     3.95 |        16.07 |
| input_precision=tf32 (rejected — breaks fp32 parity test) |     3.74 |        11.58 |

Final kernel (inline PTX tanh + saved h_new + num_warps=4): fwd=3.92 ms,
fwd+bwd=12.95 ms. fwd −18.7%, fwd+bwd −20.0% vs the manual stable-tanh
baseline.  All 9 tests in `tests/test_m2rnn_triton.py` pass on both
bench3 H200 and GB10 (sm_121).

**End-to-end smoke14 on bench3 H200×8 (15 iter, PP=4 MBS=2 GBS=16
seq=4096 BF16 te_attn graphs, identical config to smoke13):**

| run                               | iter 5-15 mean (ms) | first loss | last loss |  NaN  |
| --------------------------------- | ------------------: | ---------: | --------: | :---: |
| smoke13 (manual stable)           |             1396.98 |     12.499 |     7.753 |  no   |
| smoke14 (inline PTX + h_new save) |             1365.15 |     12.499 |     7.822 |  no   |

Training-time delta: −31.8 ms/iter (−2.28%). Modest gain vs the
microbench's 3.2 ms kernel-level win per call because the 4 R-layers
are distributed across PP=4 stages, where each stage is still bound
by its M-layer cost (Mamba-2 SSD + MLA). Remaining gap to the Path A
`mamba3_te` target (~1080 ms/iter) is ~285 ms, well beyond the
maximum ~64 ms the M²RNN kernel can contribute across all 4 R-layers
at the new speed.

**Other optimizations tried and rejected:**

- `input_precision="tf32"` on all three `tl.dot` calls: measured fwd=3.74
  fwd+bwd=11.58 ms (−28% vs inline-PTX tanh alone) BUT broke
  `test_fp32_smoke` with `out max_abs` exceeding the 1e-2 bound on the
  fp32-input path. TF32's 10-bit mantissa accumulates >1e-2 drift over
  the 128-step scan. Not "by a hair" — reverted per parity rule.
- `num_warps=2`: −occupancy, +latency. fwd 3.97 → 6.76, fwd+bwd
  12.95 → 17.44. Reverted.
- `num_warps=8`: bwd register pressure blows up; fwd unchanged, fwd+bwd
  12.95 → 16.07. Reverted.
- `libdevice.fast_tanhf`: present on the python-level libdevice wrapper
  but the cuda backend libdevice module does not surface it, compile
  error at `ast_to_ttir`. Used raw `inline_asm_elementwise` instead.

**What's left on the table to close the ~285 ms gap to Path A:**

- The M²RNN backward still does 2 dots per step (`dh_from_mm` and
  `dW += h_prev.T @ d_pre`). Both are (16,16)×(16,16) tiny dots that
  likely saturate the tensor-core launch overhead, not the math — a
  chunked version processing N>1 time steps at once could amortize
  that.  Would need a rewrite.
- Residual 12.95 ms fwd+bwd × 4 R-layers ≈ 52 ms total M²RNN cost is
  already small enough that the remaining ~285 ms must be outside the
  R-layer — likely Mamba-2 SSD, MLA, or MoE on the non-R stages, or
  pipeline bubbles. Profiling with nsys on a PP=4 run should be the
  next step.

## 2026-04-10 (latest): FP8 validated on Mamba3 paths (guards were never tested)

Removed two precautionary `NotImplementedError` guards that were both
added without any empirical FP8 test:

  - `cppmega/megatron/author_mamba3_spec.py` (AuthorMamba3Mixer, 2 lines)
  - `cppmega/megatron/m2rnn_spec.py`         (CppMegaM2RNNMixer, 2 lines)

The previous internal framing ("blocked on upstream MoE grouped GEMM
FP8 bug") was inaccurate — the real blockers were our own untested
guards, not any upstream bug. Smoke-tested on 8× H200
(`h200_1`) with `--fp8-format hybrid --fp8-amax-history-len 16
--fp8-amax-compute-algo max --cuda-graph-impl none`, 5 iters, PP=4,
MBS=2, GBS=16, NAM56R feature plan with MLA+MTP+MoE:

| Path | Spec                                    | FP8 result                   | iter 3-5 ms                                                                                                                     |
| ---- | --------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| A    | `mamba3_te_stack_spec`                  | PASS                         | 1116/1082/1064                                                                                                                  |
| B    | `mamba3_author_spec` (SISO)             | PASS                         | 1416/1082/1047                                                                                                                  |
| C    | `mamba3_author_spec` + MIMO rank=4      | FAIL (historical on one env) | backward originally failed in one local `mamba3_mimo_bwd.py` install; this was later fixed and is not the current TP bottleneck |
| D    | `nam56r_noconv_spec` (NoConvBC + M2RNN) | PASS                         | 38279/37625/37909                                                                                                               |

Path C's failure above was a historical local-environment issue in one
installed `mamba3_mimo_bwd.py` wrapper, not the durable explanation for
the current NAM56R TP result. The later single-node H200 TP=2 verdict is
still "net loss", but because of collective overhead plus a compute
profile that does not shrink enough under TP, not because of an
inherent `ngroups` kernel constraint.

See `docs/fp8_path_status.md` for the full table.

## 2026-04-10 (later): Branch B — NoConvMamba3BCMixer wired for SSD path

Following the GB10 autotune exploration hitting a hard hardware wall (see below
for shared-memory blocker on `mamba_mimo_bwd_bwd`), work pivoted back to
Branch B from the Test-loop task: "port Mamba3 features onto the vanilla
Mamba-2 SSD kernel" to get the QK-Norm/B-bias feature surface at
`mamba_chunk_scan_combined` speed.

### New mixer: `NoConvMamba3BCMixer`

Added to `cppmega/megatron/noconv_mamba_mixer.py`.  Subclasses
`NoConvMambaMixer` and overrides `_ssm_noconv` to apply:

  - `F.rms_norm(B) * B_norm_weight + B_bias`
  - `F.rms_norm(C) * C_norm_weight + C_bias`

before calling `mamba_chunk_scan_combined`.  That is the entire Mamba3
feature surface for this class — no trapezoidal / dd_A / RoPE — which
intentionally matches the `mamba3_te` recipe's "QK-Norm, B/C bias" column.

**Initialisation correction.**  The existing `Mamba3NoConvMixer` class
initialises `B_bias` / `C_bias` to **ones** (line 742-747 in the pre-change
file), which shifts every SSM input by +1 before training has moved a single
step.  `NoConvMamba3BCMixer` initialises them to **zeros** so an untuned
model starts behaviourally identical to vanilla Mamba-2 SSD.  This is
important for loading vanilla checkpoints into the new class and vice versa.

**in_proj layout.**  Left unchanged from the base `NoConvMambaMixer`
(`[z, x, B, C, dt]`, same widths as Megatron's `MambaMixer`), so partition
sizes match vanilla SSD exactly — unlike `Mamba3NoConvMixer` which adds
`dd_A`, `trap`, `angles` components and is not checkpoint-compatible.

### `Mamba3NoConvMixer` audit: math bugs found

While designing `NoConvMamba3BCMixer` I audited the full `Mamba3NoConvMixer`
and found the following issues that would break any H200 benchmark.  These
are **not fixed** in this change (the class is not used anywhere) but are
documented here so a future pass can correct them:

1. **Data-dependent A in `_mamba3_scan` doesn't scale `x`.**  The trick is
   `A_kernel = -1`, `dt_kernel = -A_dd*DT` (positive), which gives the
   correct `decay = exp(A_dd*DT)` but leaves the input term as
   `dt_kernel * B * x = |A_dd|*DT * B * x` — off by a factor of `|A_dd|`.
   The working version in `cppmega/megatron/mamba3_mixer.py::_apply_data_dep_a`
   compensates by passing `x_scan = x / |A_dd|` and `D=None` (then adds
   `D*x` manually with original `x`); `Mamba3NoConvMixer._mamba3_scan`
   passes the un-scaled `x` and `D=D`, so the input term is wrong.

2. **Trapezoidal scaling in `_preprocess_bc_mamba3` is off by a factor of `dt`.**
   The code computes `scale = dt_scale` (per head) and applies
   `B = B * scale_g`, but `mamba_chunk_scan_combined` then computes
   `input = dt_kernel * B_new * x = dt_kernel * (B * dt_scale) * x`.  To get
   the intended trapezoidal `input = dt_scale * B * x` the scale must be
   `dt_scale / dt_kernel` (per position), not `dt_scale`.

3. **B_bias / C_bias init = ones** (note above) — same file, lines ~742-747.

The docstring "Complex RoPE: not supported (would need kernel modification)"
in `Path Forward for Mamba3 at Production Speed` below is **stale**: the
code in `_apply_rope_on_state_dim` and `_preprocess_bc_mamba3` does apply
RoPE to B and C along the state dimension.  Whether the math is faithful to
the in-kernel Author-Mamba3 complex RoPE is a separate question that
requires end-to-end validation.

### Wiring: `nam56r_noconv_spec.py`

Mirrors `nam56r_te_spec.py` (which wraps Author Mamba3) but substitutes
`NoConvMamba3BCMixer` for M-layers.  R-layers still go to
`CppMegaM2RNNMixer`.  All non-Mamba layers (GDN, attention, MLP, MoE, MTP)
use the upstream TE-fused submodules from `mamba_stack_spec.submodules`
unchanged, so TE CUDA graph fusion, TE norms and TE MoE all stay in the
hot path.

The selector class `CppMegaNoConvSelectiveMambaMixer` has the same call
signature as `CppMegaSelectiveMambaMixerTE` and accepts the `r_layer_indices`
param at build time, so the NAM56R R-layer positions (12, 24, 36, 48 by
default) get the M²RNN mixer exactly as in `nam56r_full_spec`.

### Launch script: `remote_train_h200_nam56r_noconv.sh`

Near-identical copy of `remote_train_h200_nam56r_full.sh` with a single
diff: `--spec cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec`.
All other args (hidden_size, PP, BF16, MoE layout, MTP mode, MLA enable,
feature plan) are unchanged so the benchmark is directly comparable to the
`nam56r_full` baseline when run against the same cluster.

**Status:** blocked on H200 availability.  `h200_legacy` is
TERMINATED (preemptible, LOCATION_3 shows STOCKOUT for `a3-ultragpu-8g`),
so the noconv recipe has not been benchmarked yet.  The two running H200
instances (`h200_1` in LOCATION_1, `h200_1`
in LOCATION_2) do not have the cppmega stack installed.

### Tests

`tests/test_nam56r_noconv_spec.py`: 16 tests (13 source-level + 3 importability
gated on megatron/mamba_ssm).  Verifies:

  - `NoConvMamba3BCMixer` applies QK-Norm + B/C bias inside `_ssm_noconv`
  - Uses `mamba_chunk_scan_combined` (not `mamba3_siso_combined`)
  - No conv1d, no trap, no dd_A, no angles in the class body
  - B_bias / C_bias initialised to zeros; norm_weight to ones
  - The spec wires `CppMegaNoConvSelectiveMambaMixer` and the upstream TE
    substrate, does not import Author Mamba3
  - Spec builder has the module-level alias required for `--spec MODULE NAME`

Tests use an AST helper (`_extract_class_body_code`) that strips class and
method docstrings so the keyword scans aren't fooled by the prose in
docstrings.

Full local suite passes: 230 tests (9 skipped / megatron-gated).

### Isolated layer benchmark on H200 (h200_1 in LOCATION_1)

Ran NoConvMambaMixer (baseline) vs NoConvMamba3BCMixer (Mamba3 B/C features)
head-to-head at NAM56R shapes: hidden=3584, nheads=112, ngroups=8, d_state=64,
head_dim=64, seq=4096, batch=2, bf16, single layer, fwd+bwd per iter, warmup=3,
timed=10.  Single H200 GPU, no TP/PP/CP.

| Config                                     | ms/iter fwd+bwd | Params         | Overhead       |
| ------------------------------------------ | --------------- | -------------- | -------------- |
| `NoConvMambaMixer` (baseline)              | 9.4-9.6         | 429,080        | —              |
| `NoConvMamba3BCMixer` (QK-Norm + B/C bias) | 9.95-9.98       | 429,336 (+256) | +3.8% to +6.1% |

So the Python preprocessing (2 rms_norms + 2 elementwise multiplies + 2
adds + dtype downcast) costs ~3.8-6.1% on top of the scan kernel.  At full
NAM56R scale this should extrapolate to roughly the same +5% that
`CppMegaMamba3Mixer` (the conv1d-keeping variant) measured earlier:
784 ms / 167k tok/sec vs 748 ms / 175k tok/sec baseline.

### Bug fix: dtype promotion downcast

First isolated-layer run on H200 hit a Triton backward error:
``Both operands must be same dtype. Got bf16 and fp32`` in
``mamba_chunk_scan_combined``'s `tl.dot(dout, c)`.  Root cause: the
norm_weight / B_bias / C_bias parameters are fp32 (PyTorch default), so
``F.rms_norm(B) * self.B_norm_weight + self.B_bias`` type-promotes the
bf16 input B to fp32 via standard PyTorch promotion rules.  The forward
kernel tolerated fp32 C but saved it for backward in the same dtype,
which then mismatched the bf16 dout.

Fix: explicit `.to(bc_dtype)` cast after the preprocessing chain so B and
C go into the scan kernel in the same dtype as x.  The cast is in
`NoConvMamba3BCMixer._ssm_noconv` right after the norm+bias expression.
This was the crucial diff between "crashes on backward" and "runs
cleanly at +3.8% overhead."  Note: `CppMegaMamba3Mixer` uses the same
fp32 bias pattern but routes through the fused kernel which takes fp32
B/C internally; the noconv direct-call path doesn't have that escape
hatch.

### Why this matters

`NoConvMamba3BCMixer` gives the same ~5% overhead as
`CppMegaMamba3Mixer` but via a cleaner code path (direct kernel call,
no causal_conv1d dependency) and uses the upstream vanilla
``mamba_chunk_scan_combined`` without any Megatron-internal subclassing.
That makes it a better starting point for further Mamba3 feature ports:
any new feature (RoPE, data-dep A, trapezoidal) can be added as another
preprocessing step with the same type-safety pattern, without touching
the conv1d integration path.  The ~5% overhead is the same cost we
already accept for `CppMegaMamba3Mixer`, so adopting this path costs
nothing in perf — the two are equivalent — but gives us a cleaner
foundation.

The real gap from 167k to 200k+ tok/sec is NOT about mixer choice; it
comes from FP8 GEMMs in attention/MLP/MoE + larger MBS + CUDA graph
scope tuning.  Branch B ended here as a correct-and-fast wire-up; the
next optimisation lever is the FP8/CUDA-graph axis from Branch A.

### GB10 autotune detour

Spent a session on GB10 (Grace Blackwell consumer, sm_121) trying to enable
the upstream `@autotune` block in `mamba3_mimo_fwd.py` / `mamba3_mimo_bwd.py`
(`state-spaces/mamba`) to measure the config-sweep speedup.  Standalone
autotune verified that tuned configs give ~19% speedup over the hardcoded
`{threads: 128, num_stages: 0}` default on the forward kernel.

**Hardware blocker on `mamba_mimo_bwd_bwd`.**  The gradient kernel requires
140-144 KB of dynamic shared memory per block at the smoke-test dimensions
(B=2 S=256 H=8 G=1 N=64 P=64 R=4 chunk_size=16).  GB10 reports
`shared_memory_per_block_optin = 101376` bytes (≈99 KB) and
`shared_memory_per_multiprocessor = 102400` bytes (100 KB), so both
`{threads: 128}` and `{threads: 256}` configs fail at bench time with
`tvm.error.InternalError: Failed to set the allowed dynamic shared memory
size to 140960`.  The forward and the `bwd_fwd` (activation recompute)
kernels fit in GB10's budget and autotune successfully; only the true
`bwd_bwd` exceeds the limit.

See `memory/reference_gb10_bwd_bwd_blocker.md` for the full reproduction
and device properties dump.  This is not a blocker for H200 (sm_90 has
228 KB per SM), only for any eventual GB10 / consumer Blackwell deployment.

---

## 2026-04-10: MIMO R=4 Working on Full NAM56R + CUDA Graphs

Full NAM56R with Author Mamba3 MIMO R=4 kernel (TileLang), CUDA graphs captured via
`--cuda-graph-impl local`. All 7/7 Mamba3 features active: trapezoidal discretization,
data-dependent A, complex RoPE, QK-Norm, B/C bias, no-conv, MIMO rank-4.

### Final benchmark table (8xH200, BF16, MBS=4, GBS=32)

| Config                             | Mamba3 features | ms/iter  | tok/sec  | vs Baseline |
| ---------------------------------- | --------------- | -------- | -------- | ----------- |
| Baseline (Mamba-2 SSD)             | 0/7             | 748      | 175k     | —           |
| CppMegaMamba3Mixer (native SSD)    | 2/7             | 784      | 167k     | +4.9%       |
| Author SISO + local graphs         | 6/7 (no MIMO)   | 796      | 165k     | +6.4%       |
| **Author MIMO R=4 + local graphs** | **7/7**         | **1267** | **103k** | **+69%**    |

### MIMO overhead breakdown (+59% over SISO) — nsys measured

Profiled with `nsys profile --trace=cuda,nvtx` on 8-layer isolated iter-4 window.
The +12.3 ms kernel-time delta (MIMO vs SISO) breaks down as:

| Category                                     | SISO ms | MIMO ms   | Delta      |
| -------------------------------------------- | ------- | --------- | ---------- |
| **mamba_custom** (Triton/TileLang scan+proj) | 2.77    | **12.99** | **+10.22** |
| elementwise (pointwise/copy/activation)      | 11.62   | 12.31     | +0.69      |
| gemm_cublaslt (linear proj + FFN)            | 6.55    | 6.59      | +0.04      |
| attn_flash (cuDNN SDPA)                      | 2.30    | 2.30      | 0          |
| reduce                                       | 1.78    | 2.48      | +0.70      |
| optimizer (Adam)                             | 0.91    | 0.92      | 0          |
| layernorm                                    | 0.38    | 0.73      | +0.35      |
| Other                                        | 1.01    | 1.04      | 0          |

**83% of the overhead is in the Author Mamba kernels themselves** — everything else flat.
GEMM, attention, elementwise ops are unchanged.

Author Mamba kernels breakdown per iter (4 M-layers):

| Kernel                                        | SISO ms  | MIMO ms   |
| --------------------------------------------- | -------- | --------- |
| mamba3_siso_fwd_kernel                        | 0.907    | —         |
| mamba3_siso_bwd_kernel_dqkv                   | 0.847    | —         |
| mamba3_siso_bwd_kernel_rotary_bias_angles     | 0.446    | —         |
| mamba3_siso_bwd_kernel_dzdo                   | 0.178    | —         |
| angle_dt_bwd_kernel                           | 0.205    | —         |
| **SISO subtotal**                             | **2.77** | —         |
| mamba_mimo_fwd_kernel                         | —        | 2.330     |
| mamba_mimo_bwd_fwd_kernel (recompute in bwd!) | —        | 4.741     |
| mamba_mimo_bwd_bwd_kernel                     | —        | 5.752     |
| helpers (bwd_dadt/segsum/dtrap/dacs)          | —        | 0.314     |
| **MIMO subtotal**                             | —        | **13.14** |

**Key insight: MIMO bwd triggers RECOMPUTE of fwd** (`mamba_mimo_bwd_fwd_kernel_kernel = 4.74ms`)
This is activation recomputation inside the Author TileLang backward path — doubles the
scan cost on backward. Combined with 2.5x slower fwd (2.33 vs 0.91 ms) the total Mamba
kernel time grows from 2.77 → 13.14 ms = **+10.37 ms** ≈ the full +12.3 ms kernel delta.

Earlier speculation about "O(R^2), chunk_size=16, R copies of B/C" was partially correct
(R^2 explains the fwd 2.5x slowdown) but missed the dominant factor: **bwd recompute
of the whole fwd scan** is the biggest single cost.

### MIMO kernel requirements
- TileLang 0.1.8 + apache-tvm-ffi 0.1.9 (pin <0.1.10)
- `ngroups` must be 1 or nheads (not arbitrary)
- All bias/projection tensors must be fp32 (B_bias, C_bias, mimo_x/z/o, D, DT, ADT)
- chunk_size for MIMO: `min(original, 64 // rank)`

### Training stability
Loss converges cleanly: 11.81 → 8.17 over 15 iterations, no NaN/skipped iterations,
CUDA graph replay active from iter 3.

---

## 2026-04-10: cuTile Python H200 Compatibility Matrix (tested)

NVIDIA cuTile Python (`cuda-tile` 1.2.0, CUDA 13.2) cannot target H200 (sm_90a).
Exhaustive test on h200_1:

| tileiras target                                    | Compile                                          | H200 runtime    |
| -------------------------------------------------- | ------------------------------------------------ | --------------- |
| sm_80, sm_86, sm_87, sm_88, sm_89 (Ampere/Ada)     | OK                                               | no_kernel_image |
| **sm_90 (Hopper)**                                 | **REJECTED: "Cannot find option named 'sm_90'"** | n/a             |
| sm_100, sm_103, sm_110, sm_120, sm_121 (Blackwell) | OK                                               | no_kernel_image |

- All Ampere/Ada/Blackwell cubins load-fail on H200 (ISA binary incompatibility)
- sm_90 is explicitly missing from tileiras compiler 13.2
- NVIDIA README: "tileiras compiler (version 13.2) only supports Blackwell GPU
  and Ampere/Ada GPU. Hopper GPU will be supported in the coming versions."
- Sources:
  - https://github.com/NVIDIA/cutile-python/blob/main/README.md
  - https://docs.nvidia.com/cuda/cutile-python/quickstart.html
  - https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/

**For H200 we stay on TileLang** (`pip install tilelang apache-tvm-ffi<0.1.10`).
cuTile Python port is deferred until NVIDIA adds sm_90 to tileiras.

---

## 2026-04-10: TileLang MIMO Unblocked (apache-tvm-ffi<0.1.10)

### Root Cause: tvm-ffi 0.1.10 Regression

TileLang 0.1.8 crashes on import when `apache-tvm-ffi==0.1.10` is installed:
```
AttributeError: '_NestedLoopCheckVisitor' object has no attribute '_inst'
```

**Chain:**
1. `apache-tvm-ffi 0.1.10` (2026-04-07) introduced [PR #480](https://github.com/apache/tvm-ffi/pull/480)
   which enforces `__slots__=()` on all `Object` subclasses via `_ObjectSlotsMeta`.
2. `apache/tvm` fixed this in [PR #18938](https://github.com/apache/tvm/pull/18938) (2026-03-28)
   by adding `__slots__ = ("__dict__", "__weakref__")` to `TVMDerivedObject`.
3. TileLang's vendored TVM fork is pinned to commit `882a774` which does NOT have the fix.
4. TileLang main branch pinned [apache-tvm-ffi<0.1.10](https://github.com/tile-ai/tilelang/pull/2020)
   on 2026-04-08 as a workaround.

### Fix

Pin `apache-tvm-ffi<0.1.10` in setup. Updated `scripts/remote_setup_bench.sh` to install
TileLang with `apache-tvm-ffi<0.1.10` for MIMO kernel support.

### Stack (verified working on bench machine)
- PyTorch 2.12.0.dev20260409+cu132
- CUDA 13.2 (driver 595.58.03 — updated from 580.126.09)
- Transformer Engine 2.13.0
- mamba-ssm 2.3.1 + PR #909 patch applied
- flash-attn 2.8.3
- TileLang 0.1.8 + apache-tvm-ffi 0.1.9
- Megatron-LM e40feed4a

---

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

| Config                          | Kernel               | Expected tok/sec | Notes                  |
| ------------------------------- | -------------------- | ---------------- | ---------------------- |
| nemo_native (baseline)          | fused split+scan     | 165k             | vanilla Mamba-2        |
| **mamba3_native QK-Norm+bias**  | split conv + scan    | ~155-165k        | +2 RMSNorm + bias adds |
| mamba3_native + data-dep A      | split conv + scan    | ~145-160k        | +softplus + norm + div |
| mamba3_te (old, Author kernels) | mamba3_siso_combined | 127k             | CUDA graph breakage    |

Key: if mamba3_native matches ~165k, then FP8 + CUDA graphs should reach 200k+ with Mamba3.

### Benchmark Results (8×H200, BF16 + CUDA graphs, MBS=4, GBS=32)

| Config                                   | Steady-state (ms) | tok/sec  | vs Baseline |
| ---------------------------------------- | ----------------- | -------- | ----------- |
| **Baseline** (Mamba-2 SSD, fused kernel) | **743**           | **176k** | —           |
| **Mamba3 native** (QK-Norm + B/C bias)   | **788**           | **166k** | **+6.1%**   |
| mamba3_te (Author kernels, old)          | 1,035             | 127k     | +39%        |

The 6.1% overhead comes from replacing `mamba_split_conv1d_scan_combined` (fused conv+scan)
with separate `causal_conv1d_fn` + `mamba_chunk_scan_combined` + 2×RMSNorm + bias.

With FP8 + MoE CUDA graph + MBS=5 + GBS=320, extrapolated: ~198k tok/sec (vs 211k baseline).

### Optimization: Fused vs Split (benchmarked on R595/CUDA 13.2)

| Config                           | Avg 5-15 (ms) | tok/sec | Overhead |
| -------------------------------- | ------------- | ------- | -------- |
| Baseline (Mamba-2 SSD)           | 748           | 175k    | —        |
| Mamba3 Fused (pre-conv QK-Norm)  | 784           | 167k    | +4.9%    |
| Mamba3 Split (post-conv QK-Norm) | 784           | 167k    | +4.9%    |

Fused and split give identical overhead — the 4.9% is from QK-Norm + bias ops
themselves, not from kernel split vs fused.

---

## 2026-04-10: Mamba3 Feature Gap — What We Have vs Real Mamba3

### Current CppMegaMamba3Mixer: Mamba-2 + 2/7 Mamba3 Features

The "Mamba3 native" mixer is NOT Mamba3. It's Mamba-2 SSD with QK-Norm and B/C bias:

| Mamba-3 Feature            | Status               | Why Missing                                                 |
| -------------------------- | -------------------- | ----------------------------------------------------------- |
| QK-Norm on B/C             | **DONE**             | RMSNorm before/after conv1d                                 |
| Learnable B/C bias         | **DONE**             | Element-wise addition                                       |
| Trapezoidal discretization | **NOT DONE**         | Requires modified scan: h_t = α*h_{t-1} + β*v_{t-1} + γ*v_t |
| Data-dependent A           | **CODE EXISTS, OFF** | A=-1/dt trick ready, not benchmarked                        |
| Complex RoPE on B/C        | **NOT DONE**         | Pre-scan rotation, implementable in PyTorch                 |
| No conv1d                  | **NOT DONE**         | We keep conv1d for kernel compatibility                     |
| MIMO                       | **NOT DONE**         | Shared state, native SSD can't express directly             |

### Core Mamba-3 Innovations Missing

The defining features of Mamba-3 (ICLR 2026) are:
1. **Trapezoidal discretization** — replaces exponential-Euler with a 2-band
   bidiagonal matrix that implicitly includes a size-2 convolution
2. **Data-dependent A** — per-position, per-head decay factor
3. **MIMO** — R-rank shared-state scan for better capacity

Without these, our mixer is essentially Mamba-2 + normalization tricks.

### Author Kernels: Fast Compute, Broken Integration

| Kernel                                | Compute Speed | CUDA Graph? | Why                                             |
| ------------------------------------- | ------------- | ----------- | ----------------------------------------------- |
| `mamba_chunk_scan_combined` (Mamba-2) | baseline      | **YES**     | Clean Triton                                    |
| `mamba3_siso_combined` (SISO)         | ~same         | **NO**      | 27 saved tensors, non-tensor autograd inputs    |
| `mamba3_mimo_combined` (MIMO)         | ~same         | **NO**      | All SISO issues + TileLang JIT + TVM PackedFunc |

The 30% throughput gap (1035ms vs 793ms) is NOT from kernel compute —
it's from loss of CUDA graph capture. Without graphs, every iteration
pays full kernel launch overhead for 17 Mamba layers × 5 bwd kernels.

### Nanochat Reference Implementation (Pure PyTorch)

`/Users/dave/sources/nanochat/nanochat/mamba2.py` implements ALL Mamba3
features in pure PyTorch using a chunked SSD reference scan:

| Feature          | Implementation                      | Lines     | CUDA Graph OK? |
| ---------------- | ----------------------------------- | --------- | -------------- |
| Trapezoidal      | B pre-scaling + diagonal correction | 2079-2189 | YES            |
| MIMO             | Shared-state einsum scan            | 3252-3451 | YES            |
| Data-dependent A | (B,T,H) shaped A in chunked scan    | 1960-1971 | YES            |
| Complex RoPE     | cos/sin rotation on B/C pairs       | 919-1168  | YES            |

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

| Config                             | ms/iter | tok/sec  | Mamba3 features | Overhead  |
| ---------------------------------- | ------- | -------- | --------------- | --------- |
| Baseline (Mamba-2 SSD + TE graphs) | 748     | 175k     | 0/7             | —         |
| CppMegaMamba3Mixer (native SSD)    | 784     | 167k     | 2/7             | +4.9%     |
| **Author SISO + local graphs**     | **796** | **165k** | **6/7**         | **+6.4%** |
| Author SISO no graphs (old)        | 1,035   | 127k     | 6/7             | +38%      |

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

| Component    | DSL      | CUDA Graph?                          |
| ------------ | -------- | ------------------------------------ |
| SISO prefill | Triton   | YES (confirmed on H200)              |
| MIMO prefill | TileLang | YES (via nvrtc backend)              |
| Decode       | CuTe DSL | YES (CUTLASS 4.3.4 fixed refcnt bug) |
| FA4          | CuTe DSL | YES                                  |

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

| Feature                    | Module                  | Status     | Impact on Speed |
| -------------------------- | ----------------------- | ---------- | --------------- |
| QK-Norm on B/C             | `mamba3_te_mixer.py`    | Tests pass | -23% throughput |
| Learnable B/C bias         | `mamba3_te_mixer.py`    | Tests pass | -23% throughput |
| Trapezoidal discretization | `mamba3_te_mixer.py`    | Tests pass | -23% throughput |
| Complex RoPE on SSM        | `mamba3_te_mixer.py`    | Tests pass | -23% throughput |
| Data-dependent A           | `mamba3_te_mixer.py`    | Tests pass | -23% throughput |
| No-conv (conv1d removed)   | `noconv_mamba_mixer.py` | Tests pass | Not benchmarked |

The -23% comes from using Author Mamba3 scan kernels (`mamba3_siso_combined`) which
cannot participate in TE CUDA graph capture, breaking the fusion pipeline.

### Mamba3 vs Mamba2 Throughput Comparison

| Mode                              | Scan Kernel                 | Iter (ms)       | tok/sec  | MFU       | CUDA Graphs |
| --------------------------------- | --------------------------- | --------------- | -------- | --------- | ----------- |
| **nemo_native** (production)      | `mamba_chunk_scan_combined` | 810             | 165k     | 37.2%     | yes         |
| **nemo_native + FP8 + MoE graph** | `mamba_chunk_scan_combined` | 6,207 (GBS=320) | **211k** | **50.1%** | yes         |
| mamba3_te                         | `mamba3_siso_combined`      | 1,035           | 127k     | ~29%      | partial     |
| author_dp (legacy wrap)           | Author Mamba3 native        | 39,800          | 3.3k     | <1%       | no          |

### Features NOT Implemented

| Feature                                                   | Source                                          | Status                           |
| --------------------------------------------------------- | ----------------------------------------------- | -------------------------------- |
| **M²RNN** (Mamba3 R-layers)                               | Author Mamba3 / accelerated-model-architectures | **Not implemented**              |
| **MIMO** (multi-input multi-output SSM)                   | `mamba3_mimo_combined` kernel                   | Kernel reference only, not wired |
| **Output projection norm** (RMSNormGated before out_proj) | `mamba3_te_out_proj.py`                         | Built, not benchmarked           |

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
