# Cppmega Architecture And Precision Status

Status: canonical
Last updated: 2026-04-28
Scope: durable rollup of the current cppmega architecture, precision routes,
GB10/H200 caveats, and rationale behind the active paths.

This document intentionally points to dated probe/session notes as evidence
instead of replacing them. If a fact changes, update this file first and leave
the dated note as historical evidence.

## Current Defaults

- H200 production remains the real training target. Bench3 uses FP8
  tensorwise when it fits and europe uses BF16 where FP8 regresses; see
  `docs/production_status.md`.
- Local GB10 is a correctness/probe lane, not the production throughput lane.
  The current local default is the typed `local_gb10_quarter` run profile in
  `cppmega/recipes/run_profiles.py`: MTP depth 2, CCE MTP CE,
  `--bf16 --fp8-format hybrid --fp8-recipe tensorwise`, Muon q8 momentum,
  no-master BF16 optimizer storage, DSA TileLang sparse attention, and the
  attention backend set to `flash` through the patched FA4 SM120 source tree.
- For `--fp8-recipe mxfp8`, the typed profile now resolves
  `optimizer.param_storage=auto` to primary MXFP8 parameter storage and emits
  Megatron `--mxfp8-param-storage`.  This sets `TransformerConfig.fp8_param`
  without distributed `--fp8-param-gather`.
- This GB10-local MTP default is deliberately different from older docs:
  `scripts/cppmega_fp8_shim.py` now routes `CPPMEGA_MTP_CE_KERNEL=cce` through
  the main LinearCE path. Deprecated Liger MTP CE requires an explicit ACK and is
  no longer emitted by the typed local profile.
- `cppmega/recipes/nam56r_launch.py` now passes `mtp_num_predictors` through to
  `build_megatron_args_bundle()`. The run profile sets one `model.mtp_depths`
  field, and that field drives both the hybrid layer-pattern suffix and
  `--mtp-num-layers`, so depth 2 can no longer silently collapse to one
  predictor.

## Precision Paths By Block

| Block | BF16 / base path | FP8 tensorwise path | MXFP8 / block-scaled path |
| --- | --- | --- | --- |
| Dense TE Linear / Mamba projections | Params and ordinary grads stay BF16 under `Float16Module`; no FP32 master params in the local no-master Muon path. | H200 bench3 uses TE FP8 tensorwise where beneficial. Local GB10 quarter also defaults to tensorwise FP8 while keeping BF16 params/grads. | Accepted for short local GB10 training through the `mxfp8` run-profile lane. `Float16Module` preserves TE `QuantizedTensor` params instead of dequantizing them during the BF16 wrapper cast, so TE GEMM weights are authoritative primary MXFP8 storage. Native GB10 Linear backward `NN`/`NT` still fails, so the accepted default route rewrites to TN through `scripts/cppmega_fp8_shim.py` with FlashInfer/CUTLASS. The opt-in `--mxfp8-compact-columnwise-backward` route now covers real dense dgrad/wgrad operands with `cutlass_native` and clears the 34 one-step dense copy-transpose calls, but it is slower than the default and stays probe-only for performance. |
| SparseMLA / DSA main attention | `CPPMEGA_DSA_SPARSE_MODE=tilelang` replaces Megatron `unfused_dsa_fn` with TileLang SparseMLA and avoids full `[b*np,sq,sk]` score materialization. Gather-scatter is deprecated and ACK-gated. | `SparseMLA_FP8` consumes TE current/tensorwise `Float8Tensor` storage zero-copy where possible. The old local per-token requant path is removed/fail-fast because it materialized large BF16/FP32 temporaries. | QK-only and fused-forward MXFP8 prototypes exist behind `CPPMEGA_SPARSE_MLA_BLOCKSCALED_QK=1` and `CPPMEGA_SPARSE_MLA_BLOCKSCALED_FUSED=1`. They are not defaults: full training backward still lacks finite-gradient validation. |
| DSA indexer / top-k | Current supported path is BF16 per-head fused accumulation in `dsa_indexer_fused_patch.py`; it removes the upstream `[sq,b,h,sk]` FP32 intermediate. | The old FP8 indexer path was removed. Tests reject `dsa_indexer_dtype="fp8"` in `tests/test_megatron_args.py`; the launcher should not emit `--dsa-indexer-dtype`. | No MXFP8 indexer path is accepted. The target is streaming top-k from the per-head accumulator, not block-scaled dense score materialization. |
| MoE router / DeepEP | Router probabilities remain FP32 when flex/DeepEP is used. This is a DeepEP API fact, not an accidental precision leak. | TE FP8 can cover MoE GEMMs, but router dtype is still FP32 for flex. | No current MXFP8 MoE training default. DeepGEMM is not a GB10 drop-in; use TE/CUTLASS SM120 work only after the layout contract is explicit. |
| Mamba / M2RNN scan state | Mamba no-conv parameters are stored BF16 after wrapping, with transient FP32 `A_log.float()` / `dt_bias.float()`. M2RNN still has FP32 recurrent checkpoints/backward accumulators. | FP8 correctness passed for the main Mamba3 paths on H200, but scan kernels themselves are not magically MXFP8; TE covers surrounding linears. | MXFP8 is limited to selected TE projection/GEMM boundaries. It is not a scan-state format. |
| MTP head / CE | Global preferred direction is native/LinearCE/CCE; H200 native MTP CE remains blocked by NaN validation. | Local GB10 currently chooses CCE MTP CE with depth 2 through the typed profile. Deprecated Liger MTP CE is ACK-gated and not a default. | No MXFP8-specific MTP path is accepted. MTP validation is about CE routing and shared-weight backward correctness, not block-scaled GEMM alone. |

## TE MXFP8 Linear And CUTLASS GB10 Layout Boundary

The GB10 problem is not "TE forgot a transpose flag". It is a compact-layout
semantic mismatch:

- TE MXFP8 compact rowwise scales are shaped like `[rows, cols/32]`.
  Compact columnwise scales are shaped like `[rows/32, cols]`.
- GB10 native MXFP8 `TN` works, while native TE Linear backward layouts
  `NN` dgrad and `NT` wgrad fail with cuBLASLt no-algorithm on the local
  TE 2.14 + CUDA 13.2 stack.
- PyTorch `.t()` metadata is not enough. TE passes MXFP8 tensors to C++ with
  data pointer and shape; the current GEMM operand path does not carry
  arbitrary PyTorch strides for compact payload/scale storage. No-copy probes
  produced bad math.
- Stock CUTLASS SM120 block-scaled builders expect TN/K-major operand
  contracts and native SM1xx scale layouts. Directly pointing CUTLASS scale
  TMA at TE compact columnwise scale bytes reads the wrong bytes or aborts
  descriptor creation.

Local `/home/dave/TransformerEngine` (`cppmega-mxfp8-transpose-emit`,
8e19460b) does have newer GEMM-swizzled MXFP8 scale machinery:
`MXFP8Quantizer.optimize_for_gemm`, `with_gemm_swizzled_scales`,
`CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`, and common swizzle kernels. That is
not the same as a no-copy GB10 backward adapter. TE's cuBLASLt path expects
both operands to already carry GEMM-swizzled scale tensors, while the current
GB10 problem is that Linear backward wants `NN`/`NT` math and the runnable
SM120 stock block-scaled path accepts `TN`/K-major operands. Swizzling scales
solves only the scale-layout half; it does not make original TE columnwise
payload bytes equal to the physical rowwise-transposed payload that stock `TN`
expects.

The patched TE branch also contains
`MXFP8Quantizer.quantize_rowwise_transpose()`. That emits real
rowwise-transposed MXFP8 storage from the BF16 source plus compact columnwise
scales. It removes adapter-side copying of an existing MXFP8 payload, but it is
still materialized transposed storage. The current local TE `Linear` and
`LayerNormLinear` patches use that storage, or a compact columnwise one-shot
transpose when the BF16 source is unavailable, as the saved backward operand
when no all-gather/offload/FSDP path is active. The covered backward edge no
longer holds the BF16 activation as its Linear saved tensor, and the global shim
no longer keeps parameter transpose sidecars alive across the forward pass. This
is a bridge toward authoritative MXFP8 activation storage, not proof that a
descriptor-only no-transpose path exists.

## Optimizer / Param-Storage Contract

Local MXFP8 training now uses two separate switches with different meanings:

- `--mxfp8-param-storage`: local no-master Muon contract. TE module parameters
  are initialized as primary MXFP8 tensors and updated in place; no persistent
  BF16/FP32 master copy is created.
- `--fp8-param-gather`: Megatron distributed/FSDP all-gather contract. This is
  not used by the single-GB10 no-master lane.

The important implementation detail is `Float16Module`: Megatron used to call
`module.bfloat16()`, and TE `QuantizedTensor.bfloat16()` means dequantize to a
plain BF16 tensor.  The local Megatron patch casts regular tensors while
preserving TE quantized parameters with `QuantizedTensor.to_dtype()`.

Validation on real clang semantic 4k data:

```text
log: /home/dave/logs/param_storage_mxfp8_storagebreakdown_1step_20260427_003434.log
storage: MXFP8Tensor 1,436,352,512 params / 2.759 GiB
         BF16       607,499,050 params / 1.132 GiB
step 1: lm loss 11.65989, grad norm 71.314
peak:   max allocated 26.041 GiB
fallback: bf16_fallback_dgrad=0, bf16_fallback_wgrad=0

log: /home/dave/logs/param_storage_mxfp8_4step_20260427_003618.log
loss: 11.65989 -> 11.11296 -> 9.180536 -> 7.473222
hot steps 3-4: about 5.95-5.97 s/step, about 2.74k tok/s at 16,384 tok/step
val/test at step 4: 6.834810 / 6.771402
fallback: bf16_fallback_dgrad=0, bf16_fallback_wgrad=0

log: /home/dave/logs/gb10_mxfp8_zero_sidecars_20260428_171130.log
loss: 11.66119 -> 11.22491 -> 9.627202 -> 8.113233 -> 7.561619 -> 6.631005
hot steps 3-6: 5.78-5.97 s/step, about 2.74-2.83k tok/s at 16,384 tok/step
val/test at step 6: 5.908483 / 5.862091
max allocated after step 2: 26468.95 MB
fallback: bf16_fallback_dgrad=0, bf16_fallback_wgrad=0
sidecars: mxfp8_tn_sidecar_attr_attached=0,
          mxfp8_tn_sidecar_registry_peak=0,
          mxfp8_tn_sidecar_registry_peak_bytes=0
remaining bridge: mxfp8_tn_adapter_copy_transpose=3084 and
                  mxfp8_tn_adapter_missing_sidecar_copy=3084 over 6 train steps

log: /home/dave/logs/gb10_mxfp8_grouped_direct_smoke9_20260428_183814.log
change: grouped MoE MXFP8 backward now uses direct compact operands
step 1: lm loss 11.66119, mtp_1 loss 11.97334, mtp_2 loss 11.96537
val/test after step 1: 10.66364 / 10.71262
peak: max allocated 24050.16 MB, max reserved 24866 MB
grouped direct: mxfp8_grouped_direct_dgrad=10,
                mxfp8_grouped_direct_wgrad=10,
                mxfp8_grouped_direct_miss_dgrad=0,
                mxfp8_grouped_direct_miss_wgrad=0
grouped fallback: mxfp8_grouped_transpose_copy_fallback_dgrad=0,
                  mxfp8_grouped_transpose_copy_fallback_wgrad=0
dense remaining bridge: mxfp8_tn_adapter_copy_transpose=34,
                        mxfp8_tn_adapter_missing_sidecar_copy=34
fallback: bf16_fallback_dgrad=0, bf16_fallback_wgrad=0,
          native_passthrough_dgrad=0, native_passthrough_wgrad=0,
          fallback_reasons={}
```

MXFP8 primary weights do not by themselves cut model-param bytes in half
because TE stores rowwise and columnwise payloads for GEMM use. The memory win
comes from removing BF16 masters, saved BF16 Linear activations, and duplicated
sidecars. The 2026-04-28 GB10 receipts removed persistent forward-side sidecars
and then removed the grouped MoE transpose bridge. The remaining target is
dense Linear: 34 one-step dense copies in the latest grouped-direct smoke, or
3084 copies over the older 6-step receipt. The experimental dense
compact-columnwise path is available through the typed profile flag
`--mxfp8-compact-columnwise-backward`, but it is not the default because the
current SM120 direct dense loader is slower than the default FlashInfer/copy
route on the full-model smoke.

Latest dense compact-columnwise receipt:

```text
log: /home/dave/logs/gb10_mxfp8_dense_compact_native5_20260428_224251.log
change: dense Linear MXFP8 backward uses direct compact-columnwise operands
step 1: lm loss 11.66117, mtp_1 loss 11.97331, mtp_2 loss 11.96518
val/test after step 1: 10.68362 / 10.72901
iteration time: 241558.2 ms
peak: max allocated 24017.47 MB, max reserved 25202 MB
dense direct: mxfp8_cutlass_native_dgrad=34,
              mxfp8_cutlass_native_wgrad=34
dense old route: mxfp8_flashinfer_dgrad=0,
                 mxfp8_flashinfer_wgrad=0
copy bridge: mxfp8_tn_adapter_copy_transpose=0,
             mxfp8_tn_adapter_missing_sidecar_copy=0,
             mxfp8_tn_adapter_saved_transpose_operand=0
grouped direct: mxfp8_grouped_direct_dgrad=10,
                mxfp8_grouped_direct_wgrad=10,
                mxfp8_grouped_direct_miss_dgrad=0,
                mxfp8_grouped_direct_miss_wgrad=0
remaining bridge: mxfp8_norm_quantize_sidecar_bridge=35
fallback: bf16_fallback_dgrad=0, bf16_fallback_wgrad=0,
          native_passthrough_dgrad=0, native_passthrough_wgrad=0,
          fallback_reasons={}
```

This is a copy-contract win, not a speed win: the previous one-step route with
the dense FlashInfer/copy bridge took `177950.9 ms` and peaked at about
`24049 MB`, while this dense direct run took `241558.2 ms` and peaked at
`24017 MB`. The next dense work is loader/mainloop performance, not another
wrapper.

Current full-model MXFP8 token/storage path:

- Tokens remain integer IDs. Embedding output, residual hidden tensors between
  blocks, CE inputs, GEMM outputs, ordinary non-TE parameters, Mamba scan state,
  DSA/indexer tensors, and local optimizer fallback state remain BF16 unless a
  narrower block-level note says otherwise.
- TE dense Linear / LayerNormLinear / GroupedLinear weights in the MXFP8 profile
  are authoritative MXFP8 storage via `--mxfp8-param-storage`. They carry TE
  rowwise and compact columnwise payload/scale storage; there is no FP32 master
  copy in the local no-master Muon lane. The latest storage receipt still has
  BF16 model storage for non-covered tensors.
- Linear backward for covered non-FSDP edges saves GEMM-ready MXFP8 operands in
  autograd instead of retaining the BF16 Linear input. This is materialized
  MXFP8 transpose storage, counted by
  `mxfp8_tn_adapter_saved_transpose_operand=408` in the latest full-model run,
  not a persistent producer-side sidecar.
- FP32 exists in kernel accumulators and reductions: GEMM tensor-core
  accumulators, CE/loss reductions, attention reductions, router probabilities
  when DeepEP/flex is active, DSA score/top-k work, and local Mamba scalar math.
  Those are not persistent FP32 model parameters.
- The latest full-model path is zero persistent sidecars, not zero-copy
  end-to-end. `_cppmega_mxfp8_colwise_as_rowwise_transpose` still materializes
  rowwise transposed MXFP8 payload/scale copies when only compact columnwise
  storage is available, as shown by
  `mxfp8_tn_adapter_copy_transpose=3084`,
  `mxfp8_tn_adapter_missing_sidecar_copy=3084`, and
  `mxfp8_norm_quantize_sidecar_bridge=100`.

Current code reflects those facts:

- `scripts/cppmega_fp8_shim.py` retargets MXFP8 backward to GB10-supported TN.
- `cppmega/megatron/cutlass_mxfp8_gemm.py` exposes the narrow SM120/SM121
  CUTLASS TN entry points and direct original-columnwise helpers.
- `cppmega/megatron/grouped_mxfp8_gemm.py` exposes the one-launch grouped
  MXFP8 direct backend for MoE dgrad/wgrad. It consumes per-expert pointer
  lists to avoid Python-side stacking and avoids
  `_cppmega_mxfp8_colwise_as_rowwise_transpose`.
- `cppmega/megatron/cuda_ext/cppmega_sm120_blockscaled_mma_tma_compact_scale.hpp`
  and `cppmega/megatron/cuda_ext/cutlass_mxfp8_gemm.cu` contain the local
  mainloop/loader work.
- `tools/probes/gb10_accepted_path_validation.py` is the acceptance gate for
  zero BF16 fallback and zero native passthrough in GB10 MXFP8 runs.

## Next MXFP8 Zero-Copy Acceptance Contract

The next accepted grouped direct backend and dense compact-columnwise path must
be accepted on counters, not on wrapper names:

- Direct dense compact-columnwise use is evidenced by
  `mxfp8_cutlass_native_dgrad>0` and `mxfp8_cutlass_native_wgrad>0`.
- Grouped direct use is evidenced by `mxfp8_grouped_direct_dgrad>0`,
  `mxfp8_grouped_direct_wgrad>0`,
  `mxfp8_grouped_direct_miss_dgrad=0`,
  `mxfp8_grouped_direct_miss_wgrad=0`,
  `mxfp8_grouped_transpose_copy_fallback_dgrad=0`, and
  `mxfp8_grouped_transpose_copy_fallback_wgrad=0`. A grouped TE call should
  remain a grouped backend launch; do not accept per-expert Python unrolling as
  the grouped path.
- Accepted direct paths must make zero calls to
  `_cppmega_mxfp8_colwise_as_rowwise_transpose` for dgrad/wgrad/grouped
  operands. Counter evidence is all transpose/materialization counters at zero:
  `mxfp8_tn_adapter_te_emit=0`,
  `mxfp8_tn_adapter_te_emit_deferred=0`,
  `mxfp8_tn_adapter_saved_transpose_operand=0`,
  `mxfp8_tn_adapter_te_emit_swizzled=0`,
  `mxfp8_tn_adapter_te_emit_swizzled_unavailable=0`,
  `mxfp8_tn_adapter_copy_transpose=0`,
  `mxfp8_tn_adapter_missing_sidecar_copy=0`, and
  `mxfp8_norm_quantize_sidecar_bridge=0`.
- No logical `x.T`, `dy.T`, or `weight.T` MXFP8 operand may be materialized as a
  tensor for accepted direct paths. The backend must consume the original TE
  compact columnwise payload/scales directly.
- Persistent sidecar counters must stay zero:
  `mxfp8_tn_sidecar_attr_attached=0`,
  `mxfp8_tn_sidecar_attr_attached_bytes=0`,
  `mxfp8_tn_sidecar_registry_size=0`,
  `mxfp8_tn_sidecar_registry_persistent=0`,
  `mxfp8_tn_sidecar_registry_peak=0`,
  `mxfp8_tn_sidecar_registry_current_bytes=0`,
  `mxfp8_tn_sidecar_registry_peak_bytes=0`,
  `mxfp8_tn_sidecar_tracked_attr_current_bytes=0`, and
  `mxfp8_tn_sidecar_tracked_attr_peak_bytes=0`.
- Fallback/passthrough counters must stay zero:
  `bf16_fallback_dgrad=0`, `bf16_fallback_wgrad=0`,
  `native_passthrough_dgrad=0`, `native_passthrough_wgrad=0`, and
  `fallback_reasons={}`.

Evidence:

- `/home/dave/logs/perf_mxfp8_te_emit_savedtranspose2_train_20260427_001248.log`
- `docs/gb10_dense_mxfp8_status_2026_04_25.md`
- `docs/te_mxfp8_backward_gb10_plan_2026_04_25.md`
- `docs/status/cutlass_mxfp8_native_transpose_probe_2026_04_26.md`
- `docs/status/cutlass_fp8_blockwise_transpose_probe_2026_04_26.md`
- `docs/status/cutlass_mxfp8_gb10_integrated_backend_2026_04_26.md`

## Why Direct No-Sidecar Is Slow

The current direct no-sidecar CUTLASS path is a correctness/coverage path, not
the final performance shape. It avoids TE rowwise-transpose sidecars by reading
original TE compact columnwise payload/scales and manually writing the CUTLASS
shared-memory payload and scale layouts. That manual producer path currently
copies full 128x128 payload tiles serially from producer warp lanes.

Observed GB10 microprofile evidence in
`docs/status/cutlass_mxfp8_gb10_integrated_backend_2026_04_26.md`:

- Compact rowwise-scale TN path, after extension load, shape `128x128x128`:
  `dgrad_ms=0.01436`, `wgrad_ms=0.01792`.
- Direct no-sidecar path, same shape with preallocated BF16 outputs:
  `dgrad_ms=0.16552`, `wgrad_ms=0.11676`.
- The direct nsys capture is
  `/home/dave/logs/cutlass_mxfp8_gb10_direct_micro_20260426.sqlite`; the
  kernel averaged `142.77 us` over the profiled calls.

Short-term path:

- Keep H200 on the known BF16 / FP8 tensorwise production lanes.
- Keep local GB10 default on tensorwise FP8 plus BF16 storage unless an
  MXFP8-specific probe is being run.
- Keep GB10 attention on patched FA4.  The `flashinfer_cutlass` name below is
  the dense MXFP8 Linear GEMM backward backend, not an attention backend.
- For GB10 MXFP8 experiments, use typed profile arguments, for example
  `bash scripts/local_gb10_quarter_train.sh --fp8-recipe mxfp8
  --mxfp8-bwd-backend te_tn_adapter --mxfp8-transpose-emit-backend te
  --mxfp8-transpose-emit-swizzled --mxfp8-transpose-emit-strict`, then run
  `tools/probes/gb10_accepted_path_validation.py`.
- Treat direct no-sidecar CUTLASS as accepted only for probe
  coverage/correctness, not for real train, until it handles the real TE wgrad
  columnwise operand shape/layouts.

Long-term path:

- Own a narrow SM120 mainloop/loader that consumes TE compact layouts directly
  without global sidecars and without serial byte loops.
- For SparseMLA, fuse block-scaled QK, online softmax, PV, and backward inside
  the sparse kernel. Dense TE `general_gemm` is not the right boundary for
  sparse online-softmax attention.
- Upstreamable TE direction: add an MXFP8 operand descriptor/emitter that can
  express the transposed compact-layout semantics, or emit the needed backward
  TN sidecar at the original quantize/cast point while the high-precision
  source is available.

## DSA Indexer Materialization And Streaming Top-K Target

Current state:

- Upstream `_compute_index_scores` materializes `[sq,b,h,sk]` FP32.
  `cppmega/megatron/dsa_indexer_fused_patch.py` replaces it with per-head
  BF16 bmm accumulation into `[b,sq,sk]`, saving about 40 GiB at H200
  MBS=10 NAM56R.
- That patch still materializes `[b,sq,sk]` because `fused_qk_topk_naive`
  expects a dense score matrix before top-k.
- `IndexCache` (`cppmega/megatron/index_cache_patch.py`) reduces repeated
  indexer work across DSA layers, but Full layers still need the dense top-k
  input.
- Local GB10 defaults skip the indexer loss with
  `CPPMEGA_DSA_SKIP_INDEXER_LOSS=1` and `CPPMEGA_DSA_INDEXER_LOSS_COEFF=0`.

Target:

- Fold top-k selection into the per-head/tiled accumulation so the runtime
  emits only `[b,sq,topk]` indices and optional top-k scores.
- Keep the BF16 indexer as the accepted path until this streaming top-k path
  has parity and end-to-end loss validation. Do not revive the removed FP8
  indexer as a shortcut.

## DeepEP And Router Dtype Facts

`cppmega/recipes/megatron_args.py` emits:

- `--moe-token-dispatcher-type flex`
- `--moe-router-dtype fp32`

The inline rationale is that DeepEP's flex dispatcher requires FP32 router
probabilities. Do not downcast router probabilities to BF16/FP8 just to make
the dtype table look cleaner.

Boundary conditions:

- Flex/DeepEP needs `TP * EP > 1`.
- EP=1 lanes, including local GB10 single-GPU, must use alltoall. The local
  GB10 launcher appends `--moe-token-dispatcher-type alltoall`; H200 scripts
  also strip the flex router dtype when they force alltoall.
- DeepEP remains an H200/multi-GPU communication path. It is not a GB10
  single-GPU requirement.

## MTP Status And Validation Gap

Current facts:

- General shim direction: `scripts/cppmega_fp8_shim.py` defaults
  `CPPMEGA_MTP_CE_KERNEL=native`, installing `mtp_native_hopper_ce.py`.
- Current local GB10 quarter default: the `local_gb10_quarter` typed run profile
  sets `model.mtp_depths=2` and `runtime.mtp_ce_kernel="cce"`. The shim forces
  `CPPMEGA_LINEAR_CE_KERNEL=cce` for that route, so MTP uses the same
  `LinearCrossEntropyModule` API as the main head but lands on Apple CCE on
  GB10 instead of the deprecated Liger MTP path.
- H200 production status still says not to enable
  `CPPMEGA_MTP_NATIVE_HOPPER_CE=1`; it produced `grad_norm=NaN` and needs
  retained validation before becoming a default.
- PR #3345 / native Hopper LinearCE coverage is patched-tree validation, not
  a retained unfixed-tree reproducer. `upstream_prs/SUBMISSION_CHECKLIST.md`
  keeps that pack not ready until an H200 receipt is attached.
- `FastMTP` exists (`cppmega/megatron/fastmtp_layer.py` and
  `scripts/remote_train_h200_fastmtp.sh`) but is not the accepted production
  MTP path. It has structure/unit coverage, not a retained H200/GB10 training
  validation receipt.

Needed validation before changing defaults:

- H200: native MTP CE with the reduction/ignore-index patch must run with
  finite grad norms through main-head + MTP shared-weight multi-call, and it
  must beat or match the Liger/local path on memory and speed.
- GB10: local depth-2 MTP default should keep logging `lm` and `mtp_1` losses
  plus CE route/counter state. CCE standalone probes validate the shared-weight
  three-call pattern, but a retained local training receipt should still be
  recorded before treating the route as a production H200 replacement.

## Local Logs And Probe Receipts

- `tools/probes/linear_ce_probe.py` on GB10, 2026-04-27:
  CCE shared-weight 3-call `reduction="sum"` finite with loss rel error
  `9.78e-08`; synthetic `[4096,3584] x [65536,3584]` BF16 timing was
  CCE `none+mask` 131 ms, CCE `sum+masked` 118 ms, CCE `sum+filter=high`
  75 ms, Liger mean-broadcast 380 ms.
- `/home/dave/logs/gb10_100_tensorwise_mbs4_20260425_2025.log`
- `/home/dave/logs/gb10_100_mamba_mxfp8_tn_mbs4_20260425_2034.log`
- `/home/dave/logs/cutlass_mxfp8_gb10_integrated_micro_20260426.sqlite`
- `/home/dave/logs/cutlass_mxfp8_gb10_compact_scale_micro_20260426.sqlite`
- `/home/dave/logs/cutlass_mxfp8_gb10_direct_micro_20260426.sqlite`
- `tools/probes/te_blockscaled_backward_probe.py`
- `tools/probes/gb10_accepted_path_validation.py`
- `tools/probes/sparse_mla_blockscaled_qk_probe.py`
- `tools/probes/sparse_mla_blockscaled_fused_probe.py`

## External Sources Already Linked In Evidence Docs

- NVIDIA CUTLASS: <https://github.com/NVIDIA/cutlass>
- NVIDIA CUDA compute capabilities: <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>
- NVIDIA NVCC docs: <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/>
- DeepEP: <https://github.com/deepseek-ai/DeepEP>
- Megatron-LM MTP implementation: <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/multi_token_prediction.py>
- Megatron-LM MTP docs: <https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/multi_token_prediction.md>
- Megatron-LM PR #3345: <https://github.com/NVIDIA/Megatron-LM/pull/3345>
- Megatron-LM PR #3674: <https://github.com/NVIDIA/Megatron-LM/pull/3674>
- DeepSeek-V3 Technical Report: <https://arxiv.org/abs/2412.19437>
- Meta Better & Faster Multi-Token Prediction: <https://arxiv.org/abs/2404.19737>
- Liger fused linear CE source: <https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py>
- Liger issue #968: <https://github.com/linkedin/Liger-Kernel/issues/968>
