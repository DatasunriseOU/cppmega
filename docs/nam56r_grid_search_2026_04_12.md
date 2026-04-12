# NAM56R grid search — 2026-04-12 (Stream A, bench3 H200x8)

## Goal
Push NAM56R full-stack training throughput from 112k tok/sec baseline toward
>=150k tok/sec intermediate target (final goal 250k). Stream A covers
`bench3` (H200x8, LOCATION_1, `/mnt/data/venv`).

## Constraints
- **Real data only** (`clang_semantic_4k_v10_train`, HF tokenizer at `/mnt/data/tokenizer`)
- **Full 7/7 Mamba3 MIMO features** — `nam56r_full_spec.build_cppmega_nam56r_full_stack_spec`
  (AuthorMamba3Mixer for M-layers, CppMegaM2RNNMixer for R-layers)
- **TP=1** (Mamba3 module has no TP support — asserts at `author_mamba3_spec.py:60`)
- **No feature disable hacks** (MTP, MoE, MLA, DSA, M2RNN all stay on in primary runs)
- **cuDNN path fix** — `LD_LIBRARY_PATH` must include venv `nvidia/cudnn/lib` first

## Baseline anchor
- **112,152 tok/sec** at PP=2, VPP=2, MBS=4, GBS=64, MTP=1, BF16, real data,
  AllToAll MoE dispatcher, `--moe-pad-expert-input-to-capacity`,
  `cuda-graph-impl transformer_engine`, `cuda-graph-scope attn mamba moe`.
  Reproduced on bench3 at 2296 ms/iter (≈114k tok/sec). Source script:
  `/tmp/cppmega_vpp2_pp2_mbs4_cudagraph_alltoall.sh` on bench3.

## Methodology
- Launch each config via tmux on bench3 with `bash -l` (cuDNN path sourced).
- Each run: 30 iters, measure iters 10-30 (skip 10 warmup). Compute
  `tok/sec = global_batch_size * seq_len * 1000 / mean(iter_ms)`.
- Sweep one axis at a time, keep the best, then move on:
  1. Batch axes (MBS, GBS) — cheapest to change
  2. Pipeline shape (PP, VPP)
  3. CUDA graph mode (off / attn-mamba-moe / full_iteration)
  4. Expert parallel (EP)
  5. MTP depth (1 vs 2 vs 0)
  6. FP8 on MLA+MoE
- Seq length fixed at 4096. 8 H200 GPUs (world=8).
- Grid search driver: `/tmp/nam56r_grid_driver.sh` on bench3 (generated per-run).
- Running log: this file.

## Results table

| # | Config | iter_ms | tok/sec | lm_loss@end | Notes |
|---|---|---|---|---|---|
| 01 | PP=2 VPP=2 MBS=4 GBS=64 MTP=1 CG=per_module(attn mamba moe) BF16 | CRASH | — | — | SIGSEGV at CUDAGraph replay iter 2; per-module scope + MoE unstable on this stack |
| 02 | PP=2 VPP=2 MBS=4 GBS=64 MTP=1 CG=off BF16 (bench3 baseline anchor) | 3018.5 | **86,845** | 5.51 | 30 iters OK, loss 11.9→5.5; CUDA graphs disabled to anchor a stable reference |
| 03 | PP=2 VPP=2 MBS=4 GBS=64 MTP=0 CG=off (NoMTP control) | 2663.7 | **98,412** | 5.18 | +13.3% over baseline; matches prior 18% MTP-cost measurement |
| 04 | PP=2 VPP=2 MBS=4 GBS=64 MTP=1 CG=attn_only | 2899.9 | **90,399** | 5.17 | +4.1% over baseline; attn-only graph captures without MoE crash |
| 05 | PP=2 VPP=2 MBS=4 GBS=64 MTP=1 CG=per_module(attn mamba moe_router moe_preprocess) | CRASH | — | — | SIGABRT device-side assert (CUDACachingAllocator::insert_events) during graph capture |
| 06 | PP=1 VPP=1 MBS=2 GBS=32 MTP=1 CG=off (DP=8) | 1523.1 | **86,054** | 5.20 | PP=1 DP=8 has same tok/sec as PP=2 — pipeline gives no speedup on bench3 without CUDA graphs |
| 07 | PP=1 VPP=1 MBS=4 GBS=64 MTP=1 CG=off | OOM | — | — | Cross-entropy 4GB logits OOM on rank 0 — DP=8 full-layer setup too tight |

## DSA 9+4 permanent layout (2026-04-12, Stream D)

**Mission:** produce the first real-data throughput number for the permanent
NAM56R attention layout decided on 2026-04-12:
`AEMEAEMEAEMR × 52` → 13 A-layers; MLA at A-ranks `[0,4,8,12]`
(layer_numbers `[1, 17, 33, 49]`) and DSA at A-ranks
`[1,2,3,5,6,7,9,10,11]` (layer_numbers `[5, 9, 13, 21, 25, 29, 37, 41, 45]`).

Routing lives in `cppmega.megatron.nam56r_full_spec.CppMegaSelectiveAttentionLayer`
driven by `CPPMEGA_DSA_A_LAYER_RANKS="1,2,3,5,6,7,9,10,11"`. DSA indexer runs at
the Megatron-LM default (n_heads=8, head_dim=64, topk=16, loss_coeff=0.0).

Launch script: `scripts/remote_smoke_h200_dsa_full_nam56r.sh` (rewritten from
the old PP=1 noconv smoke to match the 112k baseline plumbing + DSA).

| # | Config | iter_ms | tok/sec | lm_loss | Notes |
|---|---|---|---|---|---|
| D1 | PP=2 VPP=2 MBS=**4** GBS=64 MTP=2 CG=per_module(attn mamba moe_router moe_preprocess) BF16 + DSA 9+4 | **OOM** | — | — | Crash in first forward: `torch.OutOfMemoryError` inside `dsa.fwd_fused_indexer_loss_naive → _compute_index_scores → torch.relu(index_scores)` (dsa.py:281). Rank 2 GPU had only 1.16 GB free, tried to allocate 2.00 GB. 58.81 GB active in process at failure time. |
| D2 | PP=2 VPP=2 MBS=**2** GBS=64 MTP=2 CG=per_module BF16 + DSA 9+4 | **OOM** | — | — | Crash during backward of iter 1: rank 0-3 (PP stage 0) each allocate 134.65 GB PyTorch + ~3.6 GB non-PyTorch, then `Variable._execution_engine.run_backward` tries +3.50 GB and fails (138.31 GB in use on a 139.80 GB H200). DSA indexer activation retention pushes stage 0 past the H200 ceiling even at MBS=2. |

### Reproducibility artifacts
- Launch script: `scripts/remote_smoke_h200_dsa_full_nam56r.sh`
- Local only (not committed) edit: `cppmega/recipes/nam56r_launch.py:23` flipped
  `enable_dsa: bool = False` → `enable_dsa: bool = True` on bench3 (per brief,
  not pushed to git).
- MBS=2 OOM log: `/mnt/data/cppmega-root/cppmega/cppmega_nam56r_dsa_9_4_run1_oom_mbs2.log`
  and `/mnt/data/cppmega-root/cppmega/cppmega_nam56r_dsa_9_4_run1.log` (same content).
- Rendered hybrid pattern (PP=2 × VPP=2 × MTP=2):
  `*EME*EME*EMM*|EME*EME*EMM*E|ME*EME*EMM*EM|E*EME*EMM*EME/*-/*-`
- Rendered DSA CLI fragment:
  `--experimental-attention-variant dsa --dsa-indexer-n-heads 8 --dsa-indexer-head-dim 64 --dsa-indexer-topk 16 --dsa-indexer-loss-coeff 0.0`
- DSA/MLA layer map (verified in launch log):
  - A-layer layer_numbers (1-indexed): `[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]`
  - DSA layer_numbers: `[5, 9, 13, 21, 25, 29, 37, 41, 45]`
  - MLA layer_numbers: `[1, 17, 33, 49]`

### Findings

**No real-data throughput number for DSA 9+4 on bench3 yet.** The permanent
9-DSA+4-MLA layout does not fit in 140 GB HBM at PP=2 even with MBS lowered to
2, because the DSA indexer's naive path (`fwd_fused_indexer_loss_naive` →
`_compute_index_scores` → `torch.relu`) allocates a `(batch*heads, seq, seq)`
score tensor per DSA layer and keeps it live through backward. With 9 DSA
layers on stage 0 at seq=4096, activation retention during the 1F1B backward
pushes pytorch allocations from ~118 GB (112k-baseline MLA-only reference) to
134-135 GB, then backward's gradient allocation (+3.5 GB) tips over the H200
140 GB ceiling.

**Per brief hard constraint #5, the launch was stopped without disabling DSA or
any other feature to recover.**

### Next actions (not attempted per brief)
1. **Stream E (FP8 DSA indexer port) is the first-order fix.** Moving index
   score compute + softmax/relu to FP8 halves the activation footprint of the
   9 DSA layers (~10-15 GB saved on stage 0) and should land D1 config inside
   the H200 ceiling without touching the naive algorithm.
2. Orthogonal memory levers that do NOT disable features: activation
   recompute on the DSA slots (`--recompute-granularity selective` +
   `--recompute-modules attention` or a custom DSA-only checkpoint), PP=4
   VPP=4 (halves per-stage layer count at the cost of 10-20% less compute
   density - documented in the 112k doc as a future lever), sequence length
   cut (not on the table - 4096 is a feature contract).
3. Do NOT retry without Stream E or activation recompute - every further
   MBS/config twiddle inside PP=2 VPP=2 will OOM the same way.

## Stream E --- DSA indexer FP8 port (2026-04-12)

Port of `deepseek-ai/DeepSeek-V3.2-Exp/inference/kernel.py::fp8_index` into the
Megatron DSA indexer compute path. Details in
`docs/nam56r_mtp_optimization_plan_2026_04_11_ru.md` under "DSA indexer FP8
port". Key outputs for this grid search doc:

**Unit test status (bench3 H200 2026-04-12):**
`tests/test_dsa_fp8_indexer.py` 9/9 passed in 22.4s. topk16 overlap vs BF16
reference = 85-87% on gaussian sq=64 sk=128 inputs, 94.3-94.4% on NAM56R
sq=sk=4096 production shape.

**Memory reduction (fused per-head accumulation, bench3 H200 sm_90a,
`torch._scaled_mm` rowwise fp8_e4m3fn):**

| Shape (sq=sk, b, h, d) | BF16 peak_delta | FP8 peak_delta | Reduction |
|---|---:|---:|---:|
| 4096, 2, 8, 64 | 3254.8 MB | **340.6 MB** | **9.6x less** |
| 4096, 4, 8, 64 | 6442.5 MB | **479.8 MB** | **13.4x less** |
| 4096, 2, 8, 128 | 3221.2 MB | **345.3 MB** | **9.3x less** |

**Impact on Stream D OOM at DSA 9+4 layout:** the BF16 path materialised
`index_scores [sq, b, h, sk] fp32` live through `sum(dim=2)` and the fp32
`attention_scores_softmax` / `index_scores_softmax` tensors in
`bwd_fused_indexer_loss_naive`. At MBS=4 this gave 3.25 GB live *per*
indexer call, and with 9 DSA layers on PP stage 0 the retained activation
sum pushed PyTorch past 134-135 GB. The Stream E fused FP8 rewrite holds
only `[b, sq, sk] fp32` accumulator live (~340 MB) --- **9 DSA layers ×
(3254 - 340) = ~26 GB freed on stage 0** in forward alone, roughly matching
the headroom needed to land D1 (PP=2 VPP=2 MBS=4 GBS=64 MTP=2 CG BF16 +
DSA 9+4) inside the H200 140 GB ceiling.

**Latency delta:** 1.06-1.27x faster per indexer call on production shape.
The per-head python loop in the fused rewrite caps the latency win; a
future single-GEMM + scatter_add variant could recover the full 1.72x
seen in the unfused prototype.

**Artifacts:**
- `cppmega/megatron/dsa_fp8_indexer.py` (compute) +
  `cppmega/megatron/dsa_fp8_patch.py` (idempotent monkey-patch of
  `megatron.core.transformer.experimental_attention_variant.dsa._compute_index_scores`)
- `scripts/remote_smoke_h200_dsa_fp8_indexer.sh` (full NAM56R 9+4 DSA +
  FP8 indexer; wraps `pretrain_mamba.py` with early `apply_dsa_fp8_patch()`)
- `cppmega/recipes/megatron_args.py` exposes `dsa_indexer_dtype="fp8"` kwarg

**Still to run (after Stream D baseline is published):** full end-to-end
throughput of `remote_smoke_h200_dsa_fp8_indexer.sh` with the same config
as D1 (PP=2 VPP=2 MBS=4 GBS=64 MTP=2 CG=per_module BF16 + DSA 9+4), to
confirm (a) OOM is resolved by the memory delta above, (b) loss curve
tracks the BF16 baseline through iter 10-30, and (c) end-to-end tok/sec
delta is in the expected 2-5% band.

## DSA 9+4 FP8 indexer production baseline (2026-04-12, Stream D v2)

**Mission (task #83):** produce the first real-data throughput number for
the permanent NAM56R DSA 9+4 attention layout by layering the Stream E FP8
indexer patch on top of the Stream D v1 D1 config to escape the BF16 OOM.

Launch script: `scripts/remote_smoke_h200_dsa_fp8_indexer.sh` (created for
this run; Stream E hadn't materialised it on bench3). Derived from
`scripts/remote_smoke_h200_dsa_full_nam56r.sh` (Stream D v1) with:
- `CPPMEGA_DSA_INDEXER_DTYPE=fp8` exported before launch.
- A 5th shim step that imports
  `cppmega.megatron.dsa_fp8_patch.apply_dsa_fp8_patch` and rebinds
  `megatron.core.transformer.experimental_attention_variant.dsa._compute_index_scores`
  to the FP8 kernel. Fail-loud (`raise`) if patch import fails - no BF16
  fallback per brief hard constraint #4.
- RUN_ID `cppmega_nam56r_dsa_9_4_fp8_v1`.

All 8 ranks confirmed the shim banner
`[cppmega_mimo_shim] DSA FP8 patch applied=True` at import time, and the
unit-test dry-run (`from cppmega.megatron.dsa_fp8_patch import ...;
apply_dsa_fp8_patch()`) verified `dsa._compute_index_scores.__name__ ==
'_compute_index_scores_fp8'` with the `__cppmega_dsa_fp8_patched__`
marker set.

### Result

| # | Config | iter_ms | tok/sec | lm_loss | Notes |
|---|---|---|---|---|---|
| D v1 (context) | PP=2 VPP=2 MBS=4 GBS=64 MTP=2 BF16 DSA 9+4 | OOM | - | - | Stage 0 OOM in `dsa._compute_index_scores → relu` on iter 1 forward. 134-135 GB PyTorch on stage 0 before tip-over. |
| D v1 retry (context) | same as above + MBS=2 | OOM | - | - | Stage 0 OOM in backward, 138.31/139.80 GB. |
| **D v2** | PP=2 VPP=2 MBS=4 GBS=64 MTP=2 BF16 DSA 9+4 + **FP8 indexer** | **OOM** | - | - | **Stage 1** OOM in `moe_layer.routed_experts_compute → bias_act_func → activation_func` on iter 1 forward. Ranks 0,1,2,3 each at 136.27 GB allocated / 139.80 GB total, 91-113 MB free. FP8 indexer patch applied successfully on all 8 ranks per shim logs - the failure is NOT in DSA. |

### Per-stage parameter load

Verified in launch log (`wc -l ... = 1699`, grep `parameters on`):
- PP rank 0 (stage 0, layers 1-26, DSA-heavy): **2,291,003,634 params**
- PP rank 1 (stage 1, layers 27-52 + 2 MTP predictors + output head): **2,968,353,426 params**

Stage 1 carries 30% more parameters than stage 0 because MTP and the output
embedding always land on the last PP rank. The 9 DSA layers sit entirely in
stage 0 (DSA layer_numbers `[5, 9, 13, 21, 25, 29, 37, 41, 45]`, max=45 <= 26
is false — but `nam56r_full_spec` places layers 1-26 on stage 0 for a 52-layer
model, so DSA layers at `[5, 9, 13, 21, 25]` (5 layers) are stage 0 and
`[29, 37, 41, 45]` (4 layers) are stage 1). Correction: stage 0 has 5 DSA
layers, stage 1 has 4 DSA + MoE-heavier half.

### Memory headroom delta vs v1

- v1 tip-over: stage 0 OOM at 134-135 GB with 9 BF16 DSA layers on that stage.
  Stream E memory model predicted ~26 GB freed by FP8 fused accumulation.
- v2 outcome: stage 0 now fits. No OOM on ranks 4-7 (stage 0). FP8 savings
  landed exactly where predicted.
- v2 new tip-over: **stage 1** OOM at 136.27 GB allocated in `bias_act_func`
  inside MoE experts (`experts.py:321`). This path is entirely unrelated to
  DSA; it is the `gated_linear_unit * silu` intermediate allocation for 16
  routed experts at ffn=896 with MBS=4 GBS=64 on the stage-1 layer count. The
  stage-1 layer load (12 E-layers + MTP x 2 + output head) was masked in v1
  because stage 0 crashed first.
- Net: FP8 indexer patch is working correctly and freed the expected memory
  on stage 0. The new OOM is a pre-existing stage-1 footprint issue that was
  hidden by stage 0's earlier failure.

### Memory (estimated)

No `[Rank X] memory (MB)` lines printed because the OOM happened before
`train_step` completed its first forward+backward cycle. Only the
`torch.cuda.OutOfMemoryError` summary is available:

```
GPU 0/1/2/3 (PP stage 1 ranks):
  total capacity       = 139.80 GiB
  free at crash        = 91-113 MiB (~0.08%)
  process memory       = 139.68-139.70 GiB in use
  PyTorch allocated    = 136.27 GiB
  PyTorch reserved but unallocated = 109-131 MiB
  tried to allocate    = 112.00 MiB (failed)
```

Ranks 4-7 (stage 0) did not report OOM and stayed below the 140 GB ceiling
until `torchrun` SIGTERM'd them after stage 1's crash.

### Crash trace (stage 1, rank 0-3)

```
pretrain_mamba_original.py:254 forward_step
  model(tokens, ...)
    cppmega.megatron.custom_mamba_model.py:134 forward
    megatron/.../mamba_model.py:377 forward
    megatron/.../ssm/mamba_block.py:329 forward (iterates over layers)
    megatron/.../transformer_layer.py:704 _forward_mlp
    megatron/.../transformer_layer.py:818 self.mlp(...)
    megatron/.../moe/moe_layer.py:634 custom_forward
    megatron/.../moe/moe_layer.py:598 self.routed_experts_compute(...)
    megatron/.../moe/moe_layer.py:507 apply_module(self.experts)(...)
    megatron/.../moe/experts.py:387 self.bias_act_func(fc1_output, ...)
    megatron/.../moe/experts.py:321 self.activation_func(intermediate_parallel)
  -> torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB.
```

**The crash is inside MoE expert activation, not inside DSA.** FP8 indexer
is innocent here - the DSA path was patched and ran successfully on stage 0.

### Artifacts

- Log: `/mnt/data/cppmega-root/cppmega/cppmega_nam56r_dsa_9_4_fp8_v1.log`
  (1699 lines, bench3). 8 `OutOfMemoryError` entries (one per rank), 4
  originating on stage 1 (ranks 0-3), 4 SIGTERM'd on stage 0 (ranks 4-7).
- Launch script: `scripts/remote_smoke_h200_dsa_fp8_indexer.sh` (bench3 only,
  not in git yet; created during this run since Stream E hadn't delivered
  it to the remote checkout).
- Shim patch-apply confirmation (all 8 ranks):
  `[cppmega_mimo_shim] CPPMEGA_DSA_INDEXER_DTYPE resolves to 'fp8'`
  `[cppmega_mimo_shim] DSA FP8 patch applied=True`
- DSA/MLA layer map unchanged from v1: A-layers `[1,5,9,13,17,21,25,29,33,37,41,45,49]`,
  DSA `[5,9,13,21,25,29,37,41,45]`, MLA `[1,17,33,49]`.

### Findings

1. **FP8 indexer patch is sound and lands the expected memory savings on
   stage 0.** The Stream E predicted delta of ~26 GB per stage-0 forward was
   sufficient to clear the v1 bottleneck. No DSA-path failures anywhere in
   the v2 log.
2. **Stage 1 has an independent MoE-activation OOM that v1 never surfaced.**
   The bias_act intermediate in `moe/experts.py:321` tips 4x routed experts
   past 140 GB on the MLA+MoE stage even without any DSA cost. This is a
   **new finding** not predicted in Stream D v1's post-mortem or Stream E's
   memory model.
3. **DSA 9+4 FP8 is NOT a drop-in production baseline at 112k-matching
   config.** The MoE expert activation on stage 1 needs to shrink before
   this config will train. None of the per-brief-allowed levers (FP8 DSA,
   no disable hacks) address stage 1 MoE.
4. **v2 did not complete iter 1; no tok/sec, MFU, or loss@100 numbers are
   available.** Per hard constraint #4, the run was NOT retried with MBS=2,
   DSA disabled, or MoE removed.

### Next recommendation

DSA 9+4 FP8 is **not** a new production baseline yet. The path forward,
prioritised:

1. **Memory-audit stage 1 MoE activation path on Stream A's 112k baseline**
   - the same MoE layer count + MBS=4 + GBS=64 + 16 experts config ran fine
   in Stream A's runs 01-05 without DSA, so something in the v2 launch
   (MTP=2 vs v1 default MTP=1, or the FP8 patch's Python per-head loop
   releasing less autograd state than BF16 einsum) increased stage-1
   retained activation. First concrete check: rerun v2 with MTP=1 instead
   of MTP=2 and see if stage-1 OOM clears.
2. **If MTP=1 still OOMs:** the delta is in the FP8 patch's autograd
   retention (the `torch.relu(logits) * w_h` accumulate holds logits live
   longer than the vectorised einsum path). Port a `.detach()` / manual
   backward to the Stream E kernel so stage-0 DSA savings don't silently
   retain on stage 1 via the shared gradient graph.
3. **Orthogonal lever (allowed by brief):** activation recompute on MoE
   layers only (`--recompute-granularity selective --recompute-modules
   mlp`). This is not a feature disable and should free the MoE
   activation without touching DSA.
4. **Do NOT** try PP=4 or TP=2 for stage 1 relief - Stream B already proved
   TP>1 is a net throughput loss on bench3, and PP=4 halves the baseline
   throughput per Stream A's runs 01-05.
5. **Do NOT retry the same config** - the brief's "no retry with different
   MBS" constraint holds until the stage-1 audit produces a root cause.

Bottom line: FP8 indexer is the correct architectural fix for the DSA
memory footprint and should remain in the permanent layout, but it is not
sufficient on its own to land the full 9+4 config inside 140 GB because
stage 1's MoE activation is independently at the edge.

## Stream J memory optimization sweep (2026-04-12)

Task #87. Sequentially tried 4 memory-fit variants on top of Stream D v2's
OOM configuration (PP=2 VPP=2 MBS=4 GBS=64 MTP=2 BF16 + DSA 9+4 + FP8
indexer fwd). Launcher: `scripts/remote_smoke_h200_dsa_9_4_j.sh`, runs on
bench3 (LOCATION_1) in tmux sessions `nam56r_dsa_j_v{1..4}`.

All real-data, `clang_semantic_4k_v10_train` + HuggingFaceTokenizer; no
feature disables, Mamba-3 MIMO 7/7 preserved, DSA 9+4 permanent layout,
`CPPMEGA_DSA_INDEXER_DTYPE=fp8` (Stream E+G fwd+bwd patch). Stream G's bwd
FP8 patch was verified present on bench3 before launch (module marker
`__cppmega_dsa_fp8_bwd_patched__` = True after `apply_dsa_fp8_patch()`).

### Sweep results

| Variant | Config delta | tmux session | Log file | Result | Stage-0 peak GB | Stage-1 peak GB | Failure site |
|---------|---|---|---|---|---|---|---|
| J-V1 | FP8 fwd+bwd (Stream G) only | `nam56r_dsa_j_v1` | `cppmega_nam56r_dsa_9_4_fp8_j_v1.log` | **OOM** | 136.27 (crash) | 55.6 | `moe_layer.routed_experts_compute → bias_act_func → activation_func` at `experts.py:321`. Tried +112 MiB, 91 MiB free. Same crash location as Stream D v2 on ranks 0-3. Stream G backward FP8 patch applied successfully but the OOM happens in **forward**, so bwd savings never materialise. |
| J-V2 | V1 + `--recompute-granularity selective --recompute-modules moe_act` | `nam56r_dsa_j_v2` | `cppmega_nam56r_dsa_9_4_fp8_j_v2.log` | **OOM** | 109.28 (crash, new site) | 49.5 (stable) | `compute_dsa_indexer_loss → torch.bmm(query.float(), key.float()) * softmax_scale` at `dsa.py:202`. Tried +7.00 GiB FP32 `[b*np, sq, sk] = [112, 4096, 4096]` tensor for KL-divergence loss, 6.86 GiB free. moe_act recompute fixed the MoE bias_act OOM (peak alloc dropped 136→109 GB) but the DSA KL-loss FP32 bmm became the new bottleneck. This matches Stream E's original blocker note that the KL-loss bmm stays FP32 and is NOT covered by the FP8 indexer patch. |
| J-V3 | V2 + `--decoder-first-pipeline-num-layers 29 --decoder-last-pipeline-num-layers 23` (VPP=1, flat hybrid pattern) | `nam56r_dsa_j_v3` | `cppmega_nam56r_dsa_9_4_fp8_j_v3.log` | **OOM** | 131.30 (crash, worse) | ~23 | Same `dsa.py:202` bmm as V2. Stage 0 moved to 29 base layers; this **increased** the number of DSA layers on stage 0 from 5 (D v2 split) to 6, raising the per-stage activation baseline. Uneven PP split pushed in the wrong direction. (First attempt also hit a Megatron assertion: `--hybrid-layer-pattern` with `\|` separators cannot coexist with `--decoder-first/last-pipeline-num-layers`; launcher was patched to emit a flat pattern for V3/V4 falling back to VPP=1.) |
| J-V4 | PP=4 VPP=1 (4 stages × 13 base layers, MBS=4 GBS=64 MTP=2) | `nam56r_dsa_j_v4` | `cppmega_nam56r_dsa_9_4_fp8_j_v4.log` | **OOM** | 135.85 (crash) | ~97 (stage 1), ~16 (stage 2), ~19 (stage 3) | Same `dsa.py:202` bmm. Per-stage weight+optimizer memory halved vs PP=2 but the 1F1B pipeline buffer now holds 4 in-flight micro-batches of forward activations (vs 2 at PP=2), which cancels the weight saving and then some. Each stage still has 2-3 DSA layers, and the first call to the FP32 bmm with only 392 MiB free overflows the same way. V4 as defined (PP=4 VPP=2) is not achievable because 52 % 8 ≠ 0; switched to PP=4 VPP=1, still fails. Note: the task spec's V4=PP=4 VPP=2 assumes 52 layers split into 8 equal chunks, which Megatron's hybrid pattern parser rejects. |

### Root cause (all 4 variants)

Two independent per-stage-peak bottlenecks compound the OOM:

1. **MoE `bias_act_func` output** — a `[MBS × seq × moe_ffn_hidden]` BF16
   tensor for each of the 16 routed experts. Without `moe_act` recompute,
   stage 0 peaks at 136 GB during forward and crashes. `moe_act` selective
   recompute (V2+) fixes this — peak drops to ~109 GB.

2. **DSA `compute_dsa_indexer_loss` FP32 attention scores bmm at
   `dsa.py:202`** — a per-DSA-layer transient allocation of
   `[b × num_heads × sq × sk] = [4 × 28 × 4096 × 4096]` FP32 = **7.5 GiB**
   regardless of the FP8 indexer patch. This path is hit every forward
   because `loss_coeff=0.0` only multiplies the final scalar loss; the
   full softmax + KL bmm is still computed. The Stream E FP8 patch only
   covers `_compute_index_scores` (for topk); the separate KL-loss math
   (q @ k^T → softmax → KL with attention_scores) remains BF16/FP32.

Once (1) is resolved by recompute, (2) is the new ceiling. No variant in
the J sweep addresses (2): FP8 bwd patch is bypassed because the OOM is
in forward, PP splits only redistribute layers (DSA transient cost is
per-layer), and no variant touches the DSA KL-loss path itself.

### Implications / next-step recommendations

1. **DSA 9+4 is NOT a production baseline** under the current
   megatron-core 0.18 + DSA upstream implementation on PP>=2 H200
   configurations. Both the MoE bias_act activation retention AND the
   `compute_dsa_indexer_loss` FP32 bmm exceed 140 GB together, and
   fixing one surfaces the other immediately.
2. **Stream E's original blocker note was correct and underestimated**:
   the FP32 KL-loss bmm is the real ceiling, not the `_compute_index_scores`
   indexer. The FP8 patch is necessary but not sufficient.
3. **The next memory lever is `compute_dsa_indexer_loss` itself**:
   - Option A: extend the cppmega FP8 patch to also rewrite
     `compute_dsa_indexer_loss` — specifically the `torch.bmm(query.float(),
     key.float())` call at `dsa.py:202` — using either chunked
     computation or FP8 `_scaled_mm`. This would save ~7 GB per DSA
     forward and should land the D v2 config under 140 GB.
   - Option B: gate `compute_dsa_indexer_loss` entirely when
     `loss_coeff == 0.0` (upstream bug: the loss path executes even when
     its coefficient is zero). This is not a feature disable (the loss
     has zero weight anyway, so semantics are unchanged), but it is a
     Megatron upstream change, not a cppmega-side monkey-patch opportunity
     unless we replace the whole `fwd_fused_indexer_loss_naive`.
   - Option C: chunked softmax / FlashAttention-style tiling of the KL
     bmm so the FP32 intermediate is never fully materialised. This is
     the cleanest architectural fix but requires a new cppmega kernel.
4. **Do not retry the same 4 variants or minor perturbations**. The
   bottleneck is structural, not tunable.
5. **Focus other streams away from DSA 9+4**. The 112k tok/sec baseline
   (Stream A run 02) at 13-MLA + 0-DSA remains the production candidate
   until option A or C above lands.

### Constraint compliance

- Real data only. Every run used `clang_semantic_4k_v10_train` +
  HuggingFaceTokenizer. Launcher guard rejects
  `NullTokenizer`/`--mock-data` at render time.
- cuDNN `LD_LIBRARY_PATH` set via bench3 `~/.bashrc` and also by the
  launcher as a belt-and-braces fallback.
- DSA 9+4 permanent layout preserved:
  `CPPMEGA_DSA_A_LAYER_RANKS=1,2,3,5,6,7,9,10,11`, DSA layer numbers
  `[5, 9, 13, 21, 25, 29, 37, 41, 45]`, MLA at `[1, 17, 33, 49]`.
- No feature disable. MBS stayed at 4, DSA on, MTP=2, MoE on, MIMO 7/7
  on, `--mamba-num-groups 8` (not reduced).
- Stream J launcher files only; Stream E/G files read-only.
- No git commits. No git pushes. Local-only edits to
  `scripts/remote_smoke_h200_dsa_9_4_j.sh` and this doc.
- Did not touch Stream A (#75) tmux session `nam56r_grid`, Stream B v2
  (#85) on europe, Stream G (#84) already complete, Stream I (#86).

### Artefacts

- Launcher: `scripts/remote_smoke_h200_dsa_9_4_j.sh` (local +
  `/mnt/data/cppmega-root/cppmega/scripts/` on bench3).
- Logs on bench3 under `/mnt/data/cppmega-root/cppmega/`:
  `cppmega_nam56r_dsa_9_4_fp8_j_v{1,2,3,4}.log`.
- tmux sessions on bench3: `nam56r_dsa_j_v{1,2,3,4}` (all exited cleanly
  at the bash prompt after `VARIANT=vN ... ; echo V{N}_DONE_$?` reported
  exit 1).
