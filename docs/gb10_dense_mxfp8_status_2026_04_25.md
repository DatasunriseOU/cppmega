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

## 100-Step Real-Data A/B

Both runs used the local GB10 quarter NAM56R script on real clang parquet data:
`CPPMEGA_TRAIN_ITERS=100`, `CPPMEGA_MICRO_BATCH_SIZE=4`,
`CPPMEGA_GLOBAL_BATCH_SIZE=4`, no nsys/torch profiler.

| Run | Train lm loss @100 | Val lm loss @100 | Test lm loss @100 | Avg ms/iter, steps 10-100 | Peak max allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| TE tensorwise default | 1.568469 | 1.820359 | 1.891402 | 4460.641 | 25697.59 MB |
| Mamba in/out MXFP8, old BF16 backward bridge | 1.577934 | 1.835458 | 1.890061 | 4624.763 | 25735.36 MB |
| Mamba in/out MXFP8, TN backward adapter | 1.566162 | 1.826358 | 1.902454 | 4501.460 | 25737.67 MB |

This run is a baseline for the old bridge path, not the target MXFP8 training
path. It proves the model can survive 100 real-data steps when Mamba forward
uses MXFP8 and backward dequantizes unsupported dgrad/wgrad operands to BF16,
but it is not a win: it is about 3.7% slower and uses the same memory.

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

The old bridge run showed larger transient grad-norm spikes in earlier smoke
logs. The 100-step TN-adapter run also saw spikes (`grad_norm=787.971` at step
100), but finished with no skipped or NaN iterations. Longer runs still need
loss/grad-norm monitoring before making this a default.

## Next Viable Paths

1. For dense inference or forward-only probes, use cached/prequantized TE
   MXFP8/NVFP4 weights and `general_gemm`.
2. For training, optimize the TN-adapter transpose/copy overhead or patch TE
   lower down so compact columnwise payloads can be consumed as transposed
   rowwise operands without materializing a contiguous transpose.
3. For SparseMLA, use a custom CuTe/TileLang kernel with block scales inside
   the K-block loop; TE dense `generic_gemm` cannot express the sparse online
   softmax fusion directly.
