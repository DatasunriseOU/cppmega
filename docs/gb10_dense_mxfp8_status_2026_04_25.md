# GB10 Dense MXFP8 / NVFP4 Status (2026-04-25)

## Summary

The fast dense result from the standalone TE probe is a forward-GEMM result:
prequantized MXFP8/NVFP4 weight plus quantized activation passed to
`Transformer Engine general_gemm`. It does not mean whole-model TE MXFP8
training is ready on GB10.

For training, weights should not be quantized on every forward. The intended
contract is:

1. Quantize activations every forward, because they change every call.
2. Quantize/cache weights once per optimizer step or via TE's weight workspace.
3. Reuse the cached quantized weight for forward/recompute until the optimizer
   mutates the BF16/FP16 parameter.

## Measured Probe

Agent worktree `/home/dave/source/cppmega-te-blockscaled-dense-agent` measured
`M=256, N=4096, K=4096` on GB10:

| Format | Prequantized GEMM | Quantize + GEMM | BF16 matmul |
| --- | ---: | ---: | ---: |
| MXFP8 | 0.0604 ms | 0.3020 ms | 0.2380 ms |
| NVFP4 | 0.0478 ms | 0.3905 ms | 0.2410 ms |

Conclusion: prequantized GEMM is fast; quantizing the weight each call loses.

## Whole-Model MXFP8 Training Test

Command:

```bash
RUN_ID=smoke_mxfp8_padded2_bypass_mbs4_1it \
CPPMEGA_ALLOW_TE_MXFP8_SM12=1 \
CPPMEGA_FP8_RECIPE=mxfp8 \
CPPMEGA_TRAIN_ITERS=1 \
CPPMEGA_MICRO_BATCH_SIZE=4 \
CPPMEGA_GLOBAL_BATCH_SIZE=4 \
TORCH_EXTENSIONS_DIR=/home/dave/.cache/torch_extensions/cppmega_gb10_ab \
scripts/local_gb10_quarter_train.sh
```

What happened:

1. TE 2.14 high-level `fp8_autocast` rejects MXFP8 on sm_12.x by default with:
   `MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet`.
2. `CPPMEGA_ALLOW_TE_MXFP8_SM12=1` bypasses only that Python recipe guard.
3. The NAM56R no-conv Mamba packed in-projection needed MXFP8 padding:
   logical local output dim `16496` is not divisible by 32, so the code pads
   this projection to `16512` only when `fp8_recipe=mxfp8`, then slices the 16
   unused channels before splitting `[z, x, B, C, dt]`.
4. Forward then reached backward, but TE Linear dgrad failed in cuBLASLt:
   `Unable to find suitable cuBLAS GEMM algorithm`.

This validates the upstream TE guard: on this GB10 stack, TE MXFP8 is not a
drop-in whole-model training recipe yet, even though selected forward
`general_gemm` calls work.

## Backward Override

TransformerEngine upstream `origin/main` has added a broader
`backward_override={None,"high_precision","dequantized"}` recipe knob for
MXFP8/NVFP4 instead of the older `override_linear_precision` tuple. Local TE
2.14 does not expose that field, so `scripts/cppmega_fp8_shim.py` backports the
minimal behavior we need:

```bash
NVTE_BACKWARD_OVERRIDE=dequantized
CPPMEGA_TE_MXFP8_DGRAD_BF16=1
CPPMEGA_TE_MXFP8_WGRAD_BF16=1
CPPMEGA_PAD_MAMBA_IN_PROJ_FOR_MXFP8=1
```

The shim annotates MXFP8/NVFP4 recipes with
`override_linear_precision=(False, True, True)` for visibility and intercepts
only TE backward GEMMs whose operands are MXFP8/NVFP4: dgrad is
`grad=True, layout="NN"` and wgrad is `grad=True, layout="NT"`. Those operands
are dequantized to the active BF16/FP16 dtype and the backward outputs are
produced in high precision. Forward stays on the normal TE quantized path.

Smoke result after enabling this path:

```bash
RUN_ID=smoke_mamba_mxfp8_bwd_override_pad_mbs4_1it \
CPPMEGA_TE_PRECISION_CONFIG_FILE=/home/dave/source/cppmega/configs/te_precision/mamba_mxfp8_eval_bf16.yaml \
CPPMEGA_TRAIN_ITERS=1 \
CPPMEGA_MICRO_BATCH_SIZE=4 \
CPPMEGA_GLOBAL_BATCH_SIZE=4 \
scripts/local_gb10_quarter_train.sh
```

This completed one real-data train iteration on GB10. Metrics:
`lm loss=1.165876E+01`, `mtp_1 loss=1.164849E+01`,
`grad norm=78.271`, `max allocated=23519.87 MB`, validation
`lm loss=1.014407E+01`. No MXFP8 dgrad/wgrad cuBLASLt failure occurred.

## Current Decision

- Keep `CPPMEGA_FP8_RECIPE=tensorwise` as the GB10 training default.
- Keep `CPPMEGA_ALLOW_TE_MXFP8_SM12=1` as an explicit experiment/probe switch,
  not as a default.
- Use TE precision config for Mamba-only MXFP8 experiments; whole-model MXFP8
  remains blocked by unvalidated non-Mamba TE paths.
- Keep the Mamba in-proj MXFP8 padding code because it is inert unless the
  MXFP8 experiment flags are enabled.
- Do not replace Megatron/TE training Linears with the standalone
  `general_gemm` probe: it is forward-only and would not preserve the existing
  backward/weight-gradient contract.

## Current 100-Step Real-Data A/B

Both runs used the local GB10 quarter NAM56R script on real clang parquet data:
`CPPMEGA_TRAIN_ITERS=100`, `CPPMEGA_MICRO_BATCH_SIZE=4`,
`CPPMEGA_GLOBAL_BATCH_SIZE=4`, no nsys/torch profiler, real
`clang_semantic_4k_v10_train`, sequence length 4096, FlashAttention, q8 Muon
momentum, no-master BF16 fallback, Adam8bit scalar fallback, and contiguous DDP
grad buffer disabled.

Current logs:

```text
/home/dave/logs/gb10_100_tensorwise_mbs4_20260425_2025.log
/home/dave/logs/gb10_100_mamba_mxfp8_tn_mbs4_20260425_2034.log
```

| Run | Train lm loss @100 | Val lm loss @100 | Test lm loss @100 | Avg ms/iter, steps 10-100 | Peak max allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| TE tensorwise default | 1.608007 | 1.840445 | 1.917589 | 4437.158 | 25703.02 MB |
| Mamba in/out MXFP8, TN backward adapter | 1.609526 | 1.855715 | 1.927164 | 4515.766 | 25737.33 MB |

Throughput:

```text
tensorwise steps 10-100: 3692.5 tok/s
MXFP8/TN steps 10-100:  3628.2 tok/s
```

The MXFP8/TN path is numerically viable for this short run but is not a speed
or memory win yet: it is about 1.8% slower and uses the same memory. The covered
Mamba projections use MXFP8 forward and TE block-scaled backward through the
TN-adapter route; the rest of the model remains on the regular tensorwise/BF16
training path.

The TN-adapter run hit the intended acceptance target for covered Mamba
projections:

```text
mxfp8_tn_adapter_dgrad=600
mxfp8_tn_adapter_wgrad=600
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
fallback_reasons={}
```

This is now the only accepted Mamba-MXFP8 backward route on GB10. It is not yet
a throughput win versus tensorwise default, but it removes the old BF16
materialization bridge for the covered dgrad/wgrad GEMMs and survives 100
real-data steps.

The old BF16 bridge is now fail-closed. It requires
`CPPMEGA_I_UNDERSTAND_MXFP8_BF16_BACKWARD_BRIDGE_IS_DEPRECATED_AND_SLOW=1`;
otherwise any attempt to enable `CPPMEGA_TE_MXFP8_DGRAD_BF16`,
`CPPMEGA_TE_MXFP8_WGRAD_BF16`,
`CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=1`, or
`NVTE_BACKWARD_OVERRIDE=dequantized/high_precision` raises before training.

Both accepted runs finished with zero skipped and zero NaN iterations. Both had
occasional transient grad-norm spikes on real batches:

```text
tensorwise max grad norm: 28002.506
MXFP8/TN max grad norm:  16424.762
```

The spikes did not coincide with loss NaNs or skipped iterations, but longer
runs should keep loss/grad-norm monitoring enabled before changing defaults.

## GB10/H100/B200 Support Boundary

Official support summary from the 2026-04-25 research pass:

- CUDA lists H100/H200/GH200 as SM90, B200/GB200 as SM100, B300/GB300 as
  SM103, and GB10/DGX Spark as SM121.
- H100 does not support MXFP8 or NVFP4 training. It supports ordinary FP8 and
  Hopper FP8 block-scaling modes with FP32 scales.
- TE documents MXFP8 and NVFP4 training support for Blackwell datacenter
  targets SM100/SM103. NVFP4 inference is listed as SM100+.
- CUDA 13.2 release notes mention improved DGX Spark/GB10 GEMM performance for
  MXFP8 and NVFP4, so GB10 has partial support.
- cuBLAS layout rules point to TN-only coverage for Ada, Hopper, and Blackwell
  GeForce/CC12.x-style paths. That matches local behavior: native MXFP8 TN
  works, while native MXFP8 NN dgrad and NT wgrad return
  `CUBLAS_STATUS_NOT_SUPPORTED`.

Local probe result:

```text
mxfp8_fprop_native_TN: pass
mxfp8_dgrad_native_NN: CUBLAS_STATUS_NOT_SUPPORTED
mxfp8_wgrad_native_NT: CUBLAS_STATUS_NOT_SUPPORTED
mxfp8_dgrad_shim_NN_to_TN: pass
mxfp8_wgrad_shim_NT_to_TN: pass
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
```

Interpretation: GB10 can do standard BF16/FP16 and some MXFP8/NVFP4 matmuls,
but it does not expose the same full block-scaled GEMM layout surface as B200
through the current TE 2.14 + CUDA 13.2 stack. Adding this to TE is plausible
as a lower-level adapter or emitter, but simply removing the transpose and
calling the existing native NN/NT TE path is not currently supported on this
machine.

## Torch Profiler

Short profiler captures:

```text
/home/dave/logs/gb10_prof_tensorwise_mbs4_20260425_2044_torch_profile/
/home/dave/logs/gb10_prof_mamba_mxfp8_tn_mbs4_20260425_2046_torch_profile/
```

Step-6 profile highlights:

| Area | Tensorwise step 6 | MXFP8/TN step 6 | Comment |
| --- | ---: | ---: | --- |
| `TensorParallelMuon.step` CUDA total | 2.597 s | 2.627 s | still top bottleneck |
| `Optimizer.step` self CUDA | 1.497 s | 1.465 s | q8 Muon path dominates |
| `LinearCrossEntropyFunctionBackward` | 796 ms | 797 ms | CCE backward is next large item |
| `aten::addmm` CUDA total | 1.749 s | 1.809 s | GEMM aggregate slightly worse in MXFP8/TN |
| `aten::copy_` self CUDA | 149 ms | 207 ms | extra copies/contiguous activity in MXFP8/TN |
| `aten::clone` CUDA total | 100 ms | 142 ms | MXFP8/TN adds clone/contiguous pressure |
| `FlashAttnFuncBackward` | 100 ms | 102 ms | FlashAttention is active and stable |
| `_moe_chunk_sort` | 42 ms | 42 ms | MoE token movement remains visible |
| `qmuon_update_multi_kernel` | 41 ms | 40 ms | q8 momentum kernel itself is not the whole optimizer cost |

Next profiler targets are still: fuse/reduce Python and launch overhead around
Muon/Newton-Schulz, reduce CCE backward cost, remove remaining copy/clone paths
around MXFP8/TN, and then revisit MoE token movement.

## Next Viable Paths

1. For dense inference or forward-only probes, use cached/prequantized TE
   MXFP8/NVFP4 weights and `general_gemm`.
2. For training, optimize the TN-adapter transpose/copy overhead or patch TE
   lower down so compact columnwise payloads can be consumed as transposed
   rowwise operands without materializing a contiguous transpose.
3. For SparseMLA, use a custom CuTe/TileLang kernel with block scales inside
   the K-block loop; TE dense `generic_gemm` cannot express the sparse online
   softmax fusion directly.
