# Cppmega Architecture And Precision Status

Status: canonical
Last updated: 2026-04-27
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
  `cppmega/recipes/run_profiles.py`: MTP depth 2, Liger MTP CE,
  `--bf16 --fp8-format hybrid --fp8-recipe tensorwise`, Muon q8 momentum,
  no-master BF16 optimizer storage, DSA TileLang sparse attention, and the
  explicit deprecated-path ACK carried by that profile.
- For `--fp8-recipe mxfp8`, the typed profile now resolves
  `optimizer.param_storage=auto` to primary MXFP8 parameter storage and emits
  Megatron `--mxfp8-param-storage`.  This sets `TransformerConfig.fp8_param`
  without distributed `--fp8-param-gather`.
- This GB10-local MTP default is deliberately different from older docs and
  from the generic shim direction. `scripts/cppmega_fp8_shim.py` defaults
  `CPPMEGA_MTP_CE_KERNEL=native`, and `docs/deprecated_path_gates_2026_04_25.md`
  lists native as the replacement. The local GB10 launcher overrides that
  because native MTP CE still has a validation gap and Liger is the known local
  memory-constrained path.
- `cppmega/recipes/nam56r_launch.py` now passes `mtp_num_predictors` through to
  `build_megatron_args_bundle()`. The run profile sets one `model.mtp_depths`
  field, and that field drives both the hybrid layer-pattern suffix and
  `--mtp-num-layers`, so depth 2 can no longer silently collapse to one
  predictor.

## Precision Paths By Block

| Block | BF16 / base path | FP8 tensorwise path | MXFP8 / block-scaled path |
| --- | --- | --- | --- |
| Dense TE Linear / Mamba projections | Params and ordinary grads stay BF16 under `Float16Module`; no FP32 master params in the local no-master Muon path. | H200 bench3 uses TE FP8 tensorwise where beneficial. Local GB10 quarter also defaults to tensorwise FP8 while keeping BF16 params/grads. | Accepted for short local GB10 training through the `mxfp8` run-profile lane. `Float16Module` preserves TE `QuantizedTensor` params instead of dequantizing them during the BF16 wrapper cast, so TE GEMM weights are authoritative primary MXFP8 storage. Native GB10 Linear backward `NN`/`NT` still fails, so the accepted route rewrites to TN through `scripts/cppmega_fp8_shim.py` with TE rowwise-transpose emit and swizzled scales. For local non-FSDP Linear backward, TE saves that GEMM-ready MXFP8 operand instead of keeping the BF16 activation edge plus a separate transpose sidecar. `cutlass_native` is correctness/probe-only until it covers real TE wgrad operands at acceptable speed. |
| SparseMLA / DSA main attention | `CPPMEGA_DSA_SPARSE_MODE=tilelang` replaces Megatron `unfused_dsa_fn` with TileLang SparseMLA and avoids full `[b*np,sq,sk]` score materialization. Gather-scatter is deprecated and ACK-gated. | `SparseMLA_FP8` consumes TE current/tensorwise `Float8Tensor` storage zero-copy where possible. The old local per-token requant path is removed/fail-fast because it materialized large BF16/FP32 temporaries. | QK-only and fused-forward MXFP8 prototypes exist behind `CPPMEGA_SPARSE_MLA_BLOCKSCALED_QK=1` and `CPPMEGA_SPARSE_MLA_BLOCKSCALED_FUSED=1`. They are not defaults: full training backward still lacks finite-gradient validation. |
| DSA indexer / top-k | Current supported path is BF16 per-head fused accumulation in `dsa_indexer_fused_patch.py`; it removes the upstream `[sq,b,h,sk]` FP32 intermediate. | The old FP8 indexer path was removed. Tests reject `dsa_indexer_dtype="fp8"` in `tests/test_megatron_args.py`; the launcher should not emit `--dsa-indexer-dtype`. | No MXFP8 indexer path is accepted. The target is streaming top-k from the per-head accumulator, not block-scaled dense score materialization. |
| MoE router / DeepEP | Router probabilities remain FP32 when flex/DeepEP is used. This is a DeepEP API fact, not an accidental precision leak. | TE FP8 can cover MoE GEMMs, but router dtype is still FP32 for flex. | No current MXFP8 MoE training default. DeepGEMM is not a GB10 drop-in; use TE/CUTLASS SM120 work only after the layout contract is explicit. |
| Mamba / M2RNN scan state | Mamba no-conv parameters are stored BF16 after wrapping, with transient FP32 `A_log.float()` / `dt_bias.float()`. M2RNN still has FP32 recurrent checkpoints/backward accumulators. | FP8 correctness passed for the main Mamba3 paths on H200, but scan kernels themselves are not magically MXFP8; TE covers surrounding linears. | MXFP8 is limited to selected TE projection/GEMM boundaries. It is not a scan-state format. |
| MTP head / CE | Global preferred direction is native/LinearCE, but H200 native MTP CE remains blocked by NaN validation. | Local GB10 currently chooses Liger MTP CE with depth 2 for memory. This is a launcher-local exception with an ACK, not a global default. | No MXFP8-specific MTP path is accepted. MTP validation is about CE routing and shared-weight backward correctness, not block-scaled GEMM alone. |

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
still materialized transposed storage. The current local TE `Linear` patch uses
that storage as the saved backward input when no all-gather/offload/FSDP path is
active, so the covered backward edge no longer holds the BF16 activation as its
Linear saved tensor. This is a bridge toward authoritative MXFP8 activation
storage, not proof that a descriptor-only no-transpose path exists.

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
```

MXFP8 primary weights do not by themselves cut model-param bytes in half
because TE stores rowwise and columnwise payloads for GEMM use. The memory win
comes from removing BF16 masters, saved BF16 Linear activations, and duplicated
sidecars; the remaining target is to reduce the dual-layout/transpose storage
without breaking TE backward.

Current code reflects those facts:

- `scripts/cppmega_fp8_shim.py` retargets MXFP8 backward to GB10-supported TN.
- `cppmega/megatron/cutlass_mxfp8_gemm.py` exposes the narrow SM120/SM121
  CUTLASS TN entry points and direct original-columnwise helpers.
- `cppmega/megatron/cuda_ext/cppmega_sm120_blockscaled_mma_tma_compact_scale.hpp`
  and `cppmega/megatron/cuda_ext/cutlass_mxfp8_gemm.cu` contain the local
  mainloop/loader work.
- `tools/probes/gb10_accepted_path_validation.py` is the acceptance gate for
  zero BF16 fallback and zero native passthrough in GB10 MXFP8 runs.

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
  sets `model.mtp_depths=2`, `runtime.mtp_ce_kernel="liger"`, and the explicit
  Liger ACK. The shell launcher renders those profile fields instead of owning
  separate MTP defaults.
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
  plus CE route/counter state. If native is tested locally, keep it opt-in and
  record a separate log instead of overwriting the Liger default.

## Local Logs And Probe Receipts

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
