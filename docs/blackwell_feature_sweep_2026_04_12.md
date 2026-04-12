# Blackwell feature sweep — NAM56R-half throughput (2026-04-12)

Task #77 Stream C: Blackwell-feature validation on NAM56R-half (26 layers,
hidden=1792, ffn=9472, heads=14, seq=2048) on two targets — GB10 sm_121a
(single GPU via `ssh gb10`) and Modal B200:2 sm_100.

All runs append-only below. Columns follow:

`target | feature | baseline_tok/s | with_feature_tok/s | delta_% | status | notes`

Status semantics:
- PASS    = feature works and throughput delta > +5%
- NEUTRAL = feature works, throughput delta within ±5% (flat)
- REGRESS = feature works but throughput delta < -5% (rolled back per constraint)
- BLOCKED = feature not runnable in current code or on current hardware

## Model config for every row

| target | layers | hidden | ffn | heads | seq | mbs | gbs | MLA | MTP | MoE |
|---|---|---|---|---|---|---|---|---|---|---|
| GB10 single | 26 | 1792 | 9472 | 14 | 2048 | 2 | 2 | yes | hybrid (d=1) | yes (dropless topk=4, 16 experts) |

Spec: `cppmega.megatron.nam56r_noconv_spec.build_cppmega_nam56r_noconv_stack_spec`
NEM pattern: `AEMEAEMEAEMR` (depth=26 for half)
Tokenizer: `HuggingFaceTokenizer` at `/home/dave/cppmega-root/cpp_tokenizer_hf` (real cpp_tokenizer_hf)
Data: `/home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10_train` (real clang_semantic data)
CUDA graph: `transformer_engine` impl, `attn` scope (dropless-MoE compatible)

## Stack versions

| | GB10 | bench3 H200 (ref) |
|---|---|---|
| torch  | 2.12 cu132 (nightly) | same |
| TE     | 2.13.0 | 2.13.0 |
| mamba_ssm | 2.3.1 | 2.3.1 |
| CUTLASS Python DSL | 4.4.2 | n/a |

## Baseline verification

**Real data + real tokenizer**: both prior-session GB10 half runs and my
initial invocation silently fell back to `NullTokenizer` + `MockGPTDataset`
because `scripts/remote_train_gb10_nam56r_single.sh` hardcoded
`/home/dave/cppmega-root/data/tokenizer` which does not exist on GB10; the
actual HF tokenizer lives at `/home/dave/cppmega-root/cpp_tokenizer_hf`.
**Fixed in this task**: the launcher now accepts `CPPMEGA_DATA_PATH`,
`CPPMEGA_TOKENIZER_MODEL`, `CPPMEGA_TOKENIZER_TYPE`, `CPPMEGA_VOCAB_SIZE`
env vars. All runs below use real data and real HF tokenizer; baseline
is reported under that corrected configuration.

Baseline log: `gb10:/home/dave/cppmega/cppmega_sweep77_half_baseline_10.log`
10 iters, iter 5-10 steady-state `elapsed_ms/iter`: `955.4, 950.3, 949.3,
949.1, 951.9, 954.3` — mean **951.7 ms/iter** → **4303.8 tok/sec**.
(Loss also converges cleanly 11.4 → 5.29 — confirms real-data training.)

## Results

| target | feature | baseline_tok/s | with_feature_tok/s | delta_% | status | notes |
|---|---|---|---|---|---|---|
| GB10 | (0) real data + real HF tokenizer (vs mock) | N/A | 4303.8 | N/A | fixed-input-bug | Prior runs in `cppmega_nam56r_gb10_half_*.log` used NullTokenizer+MockGPTDataset. Wrong tokenizer path in launcher. Fixed in this task. All rows below are real-data numbers. |
| GB10 | (1) CuTe DSL BF16 warp MMA + TMA wired into Mamba3 fwd | 4303.8 | — | — | BLOCKED (not-wired) | `cppmega/megatron/cute_dsl_mimo/*.py` exist as ISOLATED kernel tests/benchmarks only; nothing imports them from `nam56r_noconv_spec.py` or the mamba builder. End-to-end plumbing is a multi-hour wiring task beyond this sweep. Kernel-level proof still collected (see row 5). |
| GB10 | (2) FP8 hybrid (`--fp8-format hybrid`) | 4303.8 | 0 (crash) | -100% | BLOCKED (TE assertion) | Crashes in backward at `transformer_engine/common/gemm/cublaslt_gemm.cu:157: CanonicalizeGemmInput: Assertion failed: ret.lda % 16 == 0. Leading dimension requirement on A for FP8 GEMM. Caller must pad.` Reproduced with mbs=2/gbs=2 AND mbs=4/gbs=4 so not a batch-dim issue. NAM56R-half dims are all %16-aligned (hidden 1792, ffn 9472, MLA q_lora/kv_lora 64, qk_head 64, v_head 64, pe 32, moe_ffn 896, num_experts 16) so the offending lda is coming from a weight slice TE materializes internally in the backward — same code path PASSES on bench3 H200 (TE 2.13, cuBLAS 13.2) per `docs/fp8_path_status.md` so it is **GB10 sm_121a + cuBLASLt FP8 specific**. Logged in `docs/upstream_bugs.md`. |
| GB10 | (3) FP4 DSA (DeepSeek Sparse Attention) indexer | 4303.8 | — | — | BLOCKED (no code) | No DSA indexer is implemented in `cppmega.megatron` at all. `grep -r dsa_indexer cppmega/` returns nothing. `scripts/remote_smoke_h200_dsa*.sh` exists but wires H200 DSA training at the Megatron arg level only — no FP4 indexer module. Feature is plan-only. |
| GB10 | (4) ThunderKittens FA4-style fused mamba3 kernel | 4303.8 | — | — | BLOCKED (no code) | No ThunderKittens source in `cppmega/` (only doc references in `docs/nam56r_mtp_optimization_plan_*.md`). ThunderKittens requires tcgen05 on Blackwell which GB10 lacks per `reference_sm121_gb10_hw_caps`. The fused-mamba3 adapter in `cppmega/megatron/cute_dsl_mimo/fa4_bwd_adapter*.py` is a CuTe DSL file that uses FA4 patterns but is not wired into training. |
| GB10 | (5) CuTe DSL kernel-level smoke (health check for feature (1)) | N/A (µs) | 38.46 µs/iter | — | PASS (kernel only) | Ran `/home/dave/PyTorch_cuda_13_1_main_4fd1b9b7/third_party/cutlass/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py --mnkl 1024,1024,1024,1 --a_dtype BFloat16 --b_dtype BFloat16 --c_dtype BFloat16 --iterations 5 --warmup_iterations 2` on GB10: `PASS` + `Execution time: 38.464 µs/iter`. Matches `docs/gb10_software_stack.md` empirical claim exactly. Confirms CuTe DSL BF16 warp MMA + TMA + persistent scheduler WORK on sm_121a without any compat shim. Only the wiring-to-NAM56R step remains (row 1). |
| GB10 | (6) TMA single-CTA in mamba3 kernels | 4303.8 | — | — | BLOCKED (wrong kernel path) | `cppmega/megatron/noconv_mamba_mixer.py` uses `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined` (Triton SSD, not the TileLang mamba3 MIMO kernels). All 4 `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_{fwd,bwd,fwd_varlen,bwd_varlen}.py` files set `TL_DISABLE_TMA_LOWER: True` but they are not on the NAM56R-half training hot path at all. Flipping the TMA flag is therefore a no-op for this spec. Triton's lowering handles TMA at its own level; Triton MoE/MXFP4 has gaps on sm_121 per `docs/gb10_software_stack.md` but BF16/FP16 works fine. Bottom line: TMA is not a throughput knob on the noconv path. |
| Modal B200:2 | all features (1-5) | — | — | — | BLOCKED (no training infra) | cppmega has **no** Modal B200 NAM56R training script. Only `scripts/modal_cutile_b200.py`, `scripts/modal_cutile_b200_variant_sweep.py`, and `scripts/modal_cutile_mamba_mimo.py` exist — all three run isolated kernel-level cuTile/TileLang benchmarks on B200, not Megatron training. Writing a Modal B200 Megatron NAM56R launcher (image build + megatron clone + TE source-build + data mount + tokenizer mount + pretrain_mamba wrapper) would take several hours and multiple Modal billable runs, which exceeds this sweep task's budget. Deferring to a dedicated task. The existing `docs/modal_b200_cutile_variant_sweep_2026_04_11.md` already covers the kernel-level B200 results. |

## Upstream bug log

See `docs/upstream_bugs.md` for the FP8 `lda % 16` entry added by this task.

## DSA indexer FP4/FP8/BF16 micro-bench on Modal B200 (2026-04-12, Stream F)

Standalone kernel-level micro-bench of the DeepSeek-V3.2-Exp DSA Lightning
Indexer compute on Modal **1× B200 sm_100**. Not a training run — the 4
replicated linear sub-modules + `_compute_index_scores` + topk were
inlined from `megatron/core/transformer/experimental_attention_variant/dsa.py`
(bench3 ref, 1119 LOC, fetched 2026-04-12) into a standalone script and
timed across BF16 / FP8 / FP4 variants with 10 warmup + 50-100 iters each.

Script: `scripts/modal_dsa_indexer_bench.py`
Modal run: https://modal.com/apps/jewelmusic/main/ap-FnXZ31G6Q0D6nIqJglAtdU

**Stack observed at runtime (NGC pytorch:25.03-py3 base):**
- torch `2.7.0a0+7c8ec84dab.nv25.03`, CUDA 12.8
- device `NVIDIA B200`, cap `(10, 0)`
- transformer_engine `2.1.0+450146a` (NOT the 2.13 we targeted — NGC
  25.03 pre-dates TE 2.13)
- Available TE recipes on this TE version:
  `['Callable', 'DelayedScaling', 'Enum', 'Format', 'Literal',
    'MXFP8BlockScaling', 'NamedTuple', 'Optional', 'Recipe', 'Union',
    'annotations', 'dataclass', 'warnings']`
  → `DelayedScaling` (classic FP8 HYBRID) and `MXFP8BlockScaling` (FP8
    block-scaled) only. **No `NVFP4` / `MXFP4` / `Float4CurrentScaling`**.

**Production shape (NAM56R, verified against `cppmega/recipes/megatron_args.py`
lines 38/54-57):**

| param | value |
|---|---|
| hidden_size | 3584 |
| q_lora_rank | 64 |
| index_n_heads | 8 |
| index_head_dim | 64 |
| index_topk | 16 |
| batch | 4 |
| seqlen | 4096 |

### Latency table (mean µs, lower is better)

| dtype | linear_fwd_µs | linear_bwd_µs | index_compute_fwd_µs | index_compute_bwd_µs | fused_qk_topk_µs | peak_memory_MB | topk_overlap_vs_bf16% |
|---|---|---|---|---|---|---|---|
| BF16 | 5462.3 | 13482.1 | 3011.7 | 8191.5 | 4163.0 | 6726.6 | 100.0 |
| FP8  | 6082.4 (+11.4%) | 14081.7 (+4.4%) | 3028.0 (+0.5%) | 8269.0 (+0.9%) | 4181.6 (+0.4%) | 6875.1 (+2.2%) | 0.41 |
| FP4  | SKIPPED — TE 2.1.0 has no FP4 recipe | — | — | — | — | — | — |

(p50/p99 tracked but omitted for compactness; always within ±5% of mean.)

### Key findings

1. **FP8 is SLOWER than BF16 on the DSA indexer inner loop** (+11.4% on
   linear_fwd, +4.4% on linear_bwd). This is not a bug in TE — it is the
   expected behavior when the GEMMs are too small for FP8 to amortise the
   quantize / dequantize / amax-update overhead. The DSA linears are:
   - `linear_wq_b`   : 64 → 512 (Q-bottleneck expansion — TINY)
   - `linear_wk`     : 3584 → 64 (key projection — narrow output)
   - `linear_weights_proj` : 3584 → 8 (head mix — out=8 is NOT %16, so FP8
     is impossible for this linear at the cuBLASLt level)
   - `k_norm`        : LayerNorm(64) (not a GEMM)

   Only `linear_wk` has one dim big enough (3584) to plausibly pay back
   the FP8 fixed overhead, and even it has only 64 on the other dim. The
   cuBLAS FP8 crossover on Blackwell sits well above 512×512×512 (see
   `docs/fp8_path_status.md`); every DSA linear is below that.

2. **`_compute_index_scores` latency is precision-invariant** (3012 µs BF16
   vs 3028 µs FP8 — within noise). That's because the reference FP32
   accumulator dominates: the kernel is bandwidth-limited on the
   `[S, B, H, S_k]` intermediate (4096×4×8×4096×4B = 2.15 GB scratch)
   regardless of q/k/w input dtype. Moving q/k/w to FP4/FP8 does nothing
   to the accumulator bandwidth. (Real DeepSeek-V3.2-Exp has a fused
   Triton `fp8_index` kernel that skips the FP32 materialization — that
   kernel is what needs porting, not the dtype knob.)

3. **Peak memory barely moves** (+2.2% for FP8). Again, dominated by the
   `[B, Sq, Sk]=[4, 4096, 4096]` index_scores tensor (256 MB FP32) and
   the FP32 einsum intermediate (2.15 GB), not the linear weights
   (whose FP8 saving is < 1 MB on these dims).

4. **Topk overlap FP8 vs BF16 = 0.41%** — this is ~1/64, essentially
   random overlap for picking 16 out of 4096 indices
   (random expected ≈ 16/4096 = 0.39%). This is NOT representative of
   real training: our inputs are random BF16 tensors, so after the
   random-weight linears + Hadamard + ReLU the post-sum scores are
   numerically near-uniform and the topk is noise-dominated even at
   BF16 precision. Under FP8 the tie-breaking resolves to different
   indices. In production, weights are trained so the scores have real
   signal and ties are rare, but this bench cannot confirm FP8 topk
   fidelity on trained weights without Stream E's training run.

5. **FP4 not testable today on this stack.** NGC pytorch:25.03-py3
   ships TE 2.1.0 which only has `DelayedScaling` (FP8 DS) and
   `MXFP8BlockScaling` (FP8 MX). There is NO FP4 recipe class under any
   of `NVFP4BlockScaling`, `MXFP4BlockScaling`, `Float4CurrentScaling`,
   `NVFP4`, `MXFP4`. TE 2.13 (which claims preliminary FP4 support per
   NVIDIA's 2026 Q1 release notes) is not in a pre-built wheel or NGC
   container that matches torch 2.12 cu132 as of 2026-04-12. To unblock
   FP4 testing we need EITHER:
     - (a) NGC pytorch:25.04+ container (expected to ship TE 2.13), OR
     - (b) source-build TE 2.13 on cu13.2 with nvcc + cuDNN 9.x headers
           from `nvidia-cudnn-cu13` wheel (tried in an earlier iteration
           of this bench rig — blocked by `cudnn.h` missing from
           `CPLUS_INCLUDE_PATH` in debian_slim + nvidia wheel install).

### Recommendation

**Do NOT port an FP4 DSA indexer into Megatron-LM production after Stream
E finishes the FP8 port.** Reasoning:

- The indexer is not GEMM-bound. The 4 linears are 64→512, 3584→64, and
  3584→8 — three of four are too small for FP8, let alone FP4, to pay
  back the quantize overhead on B200. FP8 already costs +11.4% on linear
  fwd in this bench.
- The indexer IS bandwidth-bound on `_compute_index_scores` (3 ms for the
  einsum + weighted sum on a 2.15 GB intermediate). Changing q/k/w to
  FP4 does not reduce this intermediate — the FP32 accumulator does.
  The real optimization target is to **fuse einsum + ReLU + weighted sum
  + topk** into a single Triton/CuTile kernel that never materializes
  `[S, B, H, S_k]` in DRAM. DeepSeek-V3.2-Exp's
  `inference/kernel.py::fp8_index` is exactly such a fused kernel —
  porting THAT (not the FP4 dtype) is the real Stream E work.
- FP4 topk-fidelity is unknowable from this bench. FP8 already gives
  random-looking topk on random inputs; FP4 would be worse. Even if
  trained weights restore signal, the extra precision loss from FP4
  exacerbates the ReLU-near-zero tie-breaking that dominates DSA
  scores.
- Hardware risk: GB10 sm_121 (our deployment target alongside B200) has
  no FP4 tensor cores per `reference_sm121_gb10_hw_caps` — any FP4 path
  would be a B200-only feature, doubling the Megatron code path.

**Stop at Stream E's FP8 port.** When FP8 DSA lands, re-run this bench
against the TRAINED indexer checkpoint (not random weights) to re-check
topk overlap. If overlap > 95% on trained weights and Stream E shows
end-to-end throughput win, FP8 stays. FP4 is not worth the porting +
QA cost for <5% theoretical speedup that our GEMM sizes can't realise.

**Unblocker for a future FP4 retry:** update Modal image to NGC
`pytorch:25.04-py3` or later (when TE 2.13+ ships there), re-run this
same script. No other change needed.

## Files touched

- `scripts/remote_train_gb10_nam56r_single.sh` — added `CPPMEGA_DATA_PATH`,
  `CPPMEGA_TOKENIZER_MODEL`, `CPPMEGA_TOKENIZER_TYPE`, `CPPMEGA_VOCAB_SIZE`,
  `CPPMEGA_GB10_FP8` env vars. The tokenizer-path hardcode was pointing at
  a non-existent directory and silently falling back to NullTokenizer +
  MockGPTDataset; fix also unblocks all future real-data GB10 runs.
- `docs/blackwell_feature_sweep_2026_04_12.md` (this file)
- `docs/upstream_bugs.md` — new row for the GB10 FP8 lda assertion
- `scripts/modal_dsa_indexer_bench.py` — new standalone Modal B200
  kernel-bench for DSA indexer FP4/FP8/BF16 comparison (Stream F,
  task #82, 2026-04-12)
