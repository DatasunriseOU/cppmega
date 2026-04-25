# TE MXFP8/NVFP4 Linear Backward on GB10

Date: 2026-04-25

Worktree: `/home/dave/source/cppmega-te-backward-agent`

## Summary

There is no installed drop-in replacement for TE MXFP8/NVFP4 Linear backward
on GB10. Installed TransformerEngine 2.14 native MXFP8 backward still fails in
cuBLASLt for the native `NN` dgrad and `NT` wgrad layouts.

This worktree now prototypes the viable MXFP8 path in
`scripts/cppmega_fp8_shim.py`:

1. Keep TE forward and saved operands in MXFP8.
2. Force compact MXFP8 scales (`optimize_for_gemm=False`) for input, weight,
   and grad-output quantizers while the adapter is enabled.
3. Retarget compact columnwise MXFP8 payloads as rowwise transposed operands.
4. Rewrite TE Linear backward GEMMs from native unsupported layouts to the
   GB10-supported `TN` GEMM:
   - dgrad: `general_gemm(weight.T_mxfp8, dy_mxfp8, layout="TN")`
   - wgrad: `general_gemm(x.T_mxfp8, dy.T_mxfp8, layout="TN")`

The adapter copies/transposes FP8 payload bytes and scale bytes. It does not
materialize BF16 operands for covered dgrad/wgrad GEMMs. Gradients returned to
the optimizer remain BF16, matching the existing optimizer contract.

This transpose is currently a real layout conversion, not just a Python
metadata trick. TE MXFP8 compact rowwise scales are shaped like `[M, K/32]`,
while compact columnwise scales are shaped like `[M/32, K]`. Reinterpreting
columnwise payloads for `W.T` requires rowwise payload/scale storage for shape
`[K, N]`, i.e. transposed contiguous data and transposed contiguous scales.

A probe with non-contiguous `.t()` views passed through TE without an exception
but produced wrong math (`rel_l2 ~= 1.33` versus `~0.037` with contiguous
retargeted buffers). This means TE `general_gemm` is effectively consuming the
MXFP8 rowwise payload as dense packed storage, not respecting arbitrary PyTorch
strides for this representation. Removing the copy needs a lower-level TE/CUDA
patch that teaches the GEMM path about the opposite compact payload layout, not
only a layout flag swap.

## No-Copy Follow-Up: 2026-04-25

Follow-up worktree:
`/home/dave/source/cppmega-te-nocopy-agent`

Local `/home/dave/TransformerEngine` was inspected before changing the probe.
The relevant source points are:

- `transformer_engine/pytorch/csrc/type_converters.cpp`
  `NVTETensorFromMXFP8Tensor` passes MXFP8 PyTorch tensors into TE with
  `data_ptr()` and shape only. PyTorch strides are not represented in the
  `TensorWrapper`.
- `transformer_engine/common/gemm/cublaslt_gemm.cu`
  `CanonicalizeGemmInput` chooses MXFP8 rowwise storage for transposed operands
  and columnwise storage for non-transposed operands, while preserving the
  requested cuBLAS transpose flags. There is no Python-visible descriptor knob
  for "consume this compact columnwise buffer as the logical rowwise storage of
  the transposed operand".
- `transformer_engine/pytorch/csrc/util.h` has a lower-level
  `convert_block_scaling_to_mxfp8_tensor` helper that can reinterpret
  block-scaled columnwise data as rowwise to avoid a transpose, but it is scoped
  to FP8 block scaling conversion and is not exposed as a pure MXFP8 GEMM
  operand path.

`tools/probes/te_blockscaled_backward_probe.py` now has
`--probe-nocopy`, which tests the current copied adapter against no-copy
metadata variants:

```bash
PYTHONPATH=/worktree python tools/probes/te_blockscaled_backward_probe.py \
  --format mxfp8 --probe-nocopy --m 64 --n 96 --k 128
```

Result on NVIDIA GB10, sm_12.1:

| Variant | dgrad rel L2 | dgrad status | wgrad rel L2 | wgrad status |
| --- | ---: | --- | ---: | --- |
| copied payload + copied scale | 0.037086 | pass | 0.037205 | pass |
| `.t()` view payload + copied scale | 1.448428 | bad_math | 1.517260 | bad_math |
| copied payload + `.t()` view scale | 0.580868 | bad_math | 0.907368 | bad_math |
| `.t()` view payload + `.t()` view scale | 1.337617 | bad_math | 1.161306 | bad_math |
| reshape-only payload + reshape-only scale | 1.337617 | bad_math | 1.161306 | bad_math |

This proves both compact MXFP8 payload bytes and compact E8M0 scale bytes need
real reordered storage for TE 2.14's current Python/C++ GEMM path. A no-copy
solution requires a TE/CUDA patch that carries an alternate MXFP8 operand
descriptor into `CanonicalizeGemmInput`/`nvte_cublas_gemm_v2`, or a fused
quantize/cast path that directly emits the rowwise-transposed backward
operand storage.

The probe also reports the current adapter copy volume:

```json
{
  "dgrad_payload_and_scale_bytes": 12800,
  "wgrad_payload_and_scale_bytes": 15360
}
```

for the `m=64, n=96, k=128` probe shape. For projection-like
`m=256, n=4096, k=4096`, the required shim probe reports:

```json
{
  "dgrad_payload_and_scale_bytes": 17301504,
  "wgrad_payload_and_scale_bytes": 2162688
}
```

## TE/CUDA Transpose-Avoidance Follow-Up: 2026-04-25

Follow-up worktree:
`/home/dave/source/cppmega-te-cuda-mxfp8-nocopy-agent`

This pass checked whether the GB10 MXFP8 backward path can avoid transposed
emission by changing GEMM operand order, tile order, or a lower TE/CUDA
descriptor without changing math. The short answer is no for TE 2.14's current
pure MXFP8 compact tensor path.

Relevant low-level findings:

- The only exact no-transposed-emission backward math is TE's native layout:
  dgrad `dy @ W` as `NN`, and wgrad `dy.T @ x` as `NT`. TE already reaches
  those `NN`/`NT` cuBLASLt calls, and on GB10 they still fail with
  `CUBLAS_STATUS_NOT_SUPPORTED`.
- `NVTETensorFromMXFP8Tensor` in
  `transformer_engine/pytorch/csrc/type_converters.cpp` passes MXFP8 data
  pointers plus shapes into TE. PyTorch strides are not part of the current
  `TensorWrapper` representation for the GEMM operand.
- `CanonicalizeGemmInput` in
  `transformer_engine/common/gemm/cublaslt_gemm.cu` selects MXFP8 rowwise
  storage for transposed operands and columnwise storage for non-transposed
  operands while preserving the requested GEMM transpose flags. It does not
  expose a descriptor bit that means "treat compact columnwise storage for
  `[M, K]` as compact rowwise storage for `[K, M]`".
- The MXFP8 cast path writes rowwise and columnwise FP8 payload tensors in the
  same logical row-major `[rows, cols]` element order; the difference is the
  block-scale grouping. Compact columnwise scales are shaped like
  `[rows / 32, cols]`. Rowwise scales for the logical transpose are shaped like
  `[cols, rows / 32]`. A tile-order-only reinterpretation would attach the
  wrong payload elements and/or wrong E8M0 scale groups to the GEMM operands.
- TE's `convert_block_scaling_to_mxfp8_tensor` helper is useful for the older
  FP8 block-scaling representation, where columnwise/GEMM-ready data has
  different physical semantics. It is not a pure MXFP8 no-copy GEMM operand
  path.

That makes this a compact-layout semantic issue, not just a missing Python
stride or cuBLASLt tile-order knob. Rewriting unsupported GB10 backward GEMMs
to supported `TN` still needs logical transposed operand storage unless TE
adds a new MXFP8 GEMM path that can consume the original compact layout with
different scale-index semantics and prove equivalent math.

The lower-copy prototype added here is therefore a transposed-emission probe,
not a no-transpose descriptor tweak:

- `tools/probes/te_mxfp8_transpose_emit_ext.py` builds a local runtime CUDA
  extension. It does not mutate installed TransformerEngine.
- `tools/probes/te_blockscaled_backward_probe.py --prototype-transpose-emit`
  uses that extension to emit rowwise MXFP8 storage for `source.T` directly
  from BF16 source values and TE's compact columnwise E8M0 scales.
- This removes copying existing MXFP8 payload bytes from the backward adapter
  path in the prototype. It still transposes compact scale bytes, and because
  it is a standalone probe it reads BF16 source values instead of being fused
  into TE's original quantize/cast emission point.

Small GB10 check:

```bash
PYTHONPATH=/home/dave/source/cppmega-te-cuda-mxfp8-nocopy-agent:/home/dave/source/cppmega-te-cuda-mxfp8-nocopy-agent/scripts:$PYTHONPATH \
CPPMEGA_ALLOW_TE_MXFP8_SM12=1 \
CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1 \
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0 \
CPPMEGA_TE_MXFP8_DGRAD_BF16=0 \
CPPMEGA_TE_MXFP8_WGRAD_BF16=0 \
NVTE_BACKWARD_OVERRIDE=none \
python tools/probes/te_blockscaled_backward_probe.py \
  --format mxfp8 --use-shim --probe-nocopy --prototype-transpose-emit \
  --m 64 --n 96 --k 128
```

Result on NVIDIA GB10 sm_12.1:

| Case | rel L2 vs BF16 | rel L2 vs copy adapter | Max abs vs copy adapter | Status |
| --- | ---: | ---: | ---: | --- |
| MXFP8 transpose-emit dgrad `TN` | 0.037086 | 0.0 | 0.0 | pass |
| MXFP8 transpose-emit wgrad `TN` | 0.037205 | 0.0 | 0.0 | pass |
| MXFP8 native dgrad `NN` | n/a | n/a | n/a | cuBLASLt no algorithm |
| MXFP8 native wgrad `NT` | n/a | n/a | n/a | cuBLASLt no algorithm |

Small-shape byte/timing accounting:

```json
{
  "adapter_copy_bytes": {
    "dgrad_payload_and_scale_bytes": 12800,
    "wgrad_payload_and_scale_bytes": 15360
  },
  "prototype": {
    "dgrad": {
      "bf16_source_read_bytes": 24576,
      "emitted_payload_bytes": 12288,
      "scale_transpose_bytes": 512,
      "existing_mxfp8_payload_copy_bytes": 0,
      "copy_adapter_transpose_elapsed_ms": 0.023776,
      "emit_elapsed_ms": 0.05264
    },
    "wgrad_x": {
      "bf16_source_read_bytes": 16384,
      "emitted_payload_bytes": 8192,
      "scale_transpose_bytes": 512,
      "existing_mxfp8_payload_copy_bytes": 0,
      "copy_adapter_transpose_elapsed_ms": 0.051136,
      "emit_elapsed_ms": 0.017312
    },
    "wgrad_dy": {
      "bf16_source_read_bytes": 12288,
      "emitted_payload_bytes": 6144,
      "scale_transpose_bytes": 512,
      "existing_mxfp8_payload_copy_bytes": 0,
      "copy_adapter_transpose_elapsed_ms": 0.022752,
      "emit_elapsed_ms": 0.01792
    }
  }
}
```

Projection-like GB10 check:

```bash
PYTHONPATH=/home/dave/source/cppmega-te-cuda-mxfp8-nocopy-agent:/home/dave/source/cppmega-te-cuda-mxfp8-nocopy-agent/scripts:$PYTHONPATH \
CPPMEGA_ALLOW_TE_MXFP8_SM12=1 \
CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1 \
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0 \
CPPMEGA_TE_MXFP8_DGRAD_BF16=0 \
CPPMEGA_TE_MXFP8_WGRAD_BF16=0 \
NVTE_BACKWARD_OVERRIDE=none \
python tools/probes/te_blockscaled_backward_probe.py \
  --format mxfp8 --use-shim --prototype-transpose-emit \
  --m 256 --n 4096 --k 4096
```

Result on NVIDIA GB10 sm_12.1:

| Case | rel L2 vs BF16 | rel L2 vs copy adapter | Max abs vs copy adapter | GEMM elapsed ms | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| MXFP8 transpose-emit dgrad `TN` | 0.037767 | 0.0 | 0.0 | 0.184384 | pass |
| MXFP8 transpose-emit wgrad `TN` | 0.037756 | 0.0 | 0.0 | 0.20784 | pass |
| MXFP8 native dgrad `NN` | n/a | n/a | n/a | n/a | cuBLASLt no algorithm |
| MXFP8 native wgrad `NT` | n/a | n/a | n/a | n/a | cuBLASLt no algorithm |

Projection-like byte/timing accounting:

```json
{
  "adapter_copy_bytes": {
    "dgrad_payload_and_scale_bytes": 17301504,
    "wgrad_payload_and_scale_bytes": 2162688
  },
  "prototype": {
    "dgrad": {
      "bf16_source_read_bytes": 33554432,
      "emitted_payload_bytes": 16777216,
      "scale_transpose_bytes": 524288,
      "existing_mxfp8_payload_copy_bytes": 0,
      "copy_adapter_transpose_elapsed_ms": 0.572992,
      "emit_elapsed_ms": 0.291392
    },
    "wgrad_x": {
      "bf16_source_read_bytes": 2097152,
      "emitted_payload_bytes": 1048576,
      "scale_transpose_bytes": 32768,
      "existing_mxfp8_payload_copy_bytes": 0,
      "copy_adapter_transpose_elapsed_ms": 0.064832,
      "emit_elapsed_ms": 0.018784
    },
    "wgrad_dy": {
      "bf16_source_read_bytes": 2097152,
      "emitted_payload_bytes": 1048576,
      "scale_transpose_bytes": 32768,
      "existing_mxfp8_payload_copy_bytes": 0,
      "copy_adapter_transpose_elapsed_ms": 0.196704,
      "emit_elapsed_ms": 0.020992
    }
  }
}
```

Shim stats remained fail-closed in both runs:

```text
mxfp8_tn_adapter_dgrad=1
mxfp8_tn_adapter_wgrad=1
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
fallback_reasons={}
```

This is not production-ready as written. It is a TE patch candidate: to become
production-useful, TE's quantizer/cast path would need an option to emit and
stash the backward `TN` rowwise-transposed compact MXFP8 operand at the time
the high-precision source is already available. Running the standalone probe
kernel in backward from BF16 source would only be valid for paths that still
save that source and would trade the MXFP8 payload copy for a BF16 read plus
requantization.

Additional recipe/context checks did not reveal a higher-level workaround:
local `../nanochat` and Megatron Core MXFP8 recipe plumbing select TE recipes
and quantizer scope, but do not add an MXFP8 backward descriptor/stride path.
No local Megatron Bridge checkout was present under `/home/dave` during this
pass.

## Runtime Env Flags

Preserved existing flags:

```bash
CPPMEGA_ALLOW_TE_MXFP8_SM12=1
CPPMEGA_TE_MXFP8_DGRAD_BF16=1
CPPMEGA_TE_MXFP8_WGRAD_BF16=1
NVTE_BACKWARD_OVERRIDE=dequantized
```

New experimental path:

```bash
CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1
```

New control/debug flags:

```bash
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0  # default for adapter-only runs
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=1  # emergency fallback
CPPMEGA_TE_MXFP8_BWD_DEBUG=1                # per-GEMM routing log
```

For an end-to-end MXFP8 backward acceptance run, use:

```bash
CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0
CPPMEGA_TE_MXFP8_DGRAD_BF16=0
CPPMEGA_TE_MXFP8_WGRAD_BF16=0
NVTE_BACKWARD_OVERRIDE=none
```

`NVTE_BACKWARD_OVERRIDE=none` is used only to prevent launchers that default
the env var with `${VAR:-dequantized}` from re-enabling the old BF16 bridge.

The old BF16 bridge is now fail-closed. Any of these settings require an
explicit archaeology ACK or the shim raises before training starts:

```bash
CPPMEGA_TE_MXFP8_DGRAD_BF16=1
CPPMEGA_TE_MXFP8_WGRAD_BF16=1
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=1
NVTE_BACKWARD_OVERRIDE=dequantized
NVTE_BACKWARD_OVERRIDE=high_precision
```

The ACK is intentionally long:

```bash
CPPMEGA_I_UNDERSTAND_MXFP8_BF16_BACKWARD_BRIDGE_IS_DEPRECATED_AND_SLOW=1
```

With the ACK, the shim prints a loud deprecated warning and still counts every
BF16 bridge call. Without it, the path cannot run accidentally.

## Counters And Fallback Logging

The shim exposes:

```python
cppmega_te_mxfp8_bwd_stats
cppmega_te_mxfp8_bwd_stats_snapshot()
```

The atexit log prints:

```text
[cppmega_fp8_shim] TE block-scaled backward stats: {...}
```

Tracked keys:

- `mxfp8_tn_adapter_dgrad`
- `mxfp8_tn_adapter_wgrad`
- `bf16_fallback_dgrad`
- `bf16_fallback_wgrad`
- `native_passthrough_dgrad`
- `native_passthrough_wgrad`
- `fallback_reasons`

With `CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0`, an unsupported adapter case
is counted as `native_passthrough_*` and then handed to native TE, which fails
loudly on current GB10 MXFP8 `NN`/`NT`. With fallback enabled, the same case is
counted as `bf16_fallback_*`, logged with a reason, and then dequantized through
the old bridge.

The old explicit BF16 bridge flags still work, but those calls are now counted
with reason `legacy_bf16_override`.

## Adapter Preconditions

The adapter is intentionally narrow:

- operands must be TE MXFP8 tensors;
- GEMM must be backward `grad=True`;
- layout must be native TE Linear backward `NN` or `NT`;
- comm-overlap/userbuffer paths are not covered;
- MXFP8 scales must be compact, non-swizzled scales;
- columnwise payload and columnwise scale buffers must exist.

For 3D/ND TE Linear inputs, the adapter treats the operand as a matrix
`[prod(leading_dims), hidden]` before building the transposed rowwise MXFP8
view. This is required by real Mamba projection wgrad.

## Probe Results

Installed stack:

- `transformer_engine==2.14.0`
- `torch==2.13.0.dev20260417+cu132`
- device: NVIDIA GB10, compute capability `12.1`

Baseline probe:

```bash
python tools/probes/te_blockscaled_backward_probe.py --format both
```

Small shape `M=64, N=96, K=128`:

| Case | Result | rel L2 vs BF16 matmul |
| --- | --- | ---: |
| MXFP8 native fprop `TN` | pass | 0.037809 |
| MXFP8 native dgrad `NN` | fail | cuBLASLt no algorithm |
| MXFP8 native wgrad `NT` | fail | cuBLASLt no algorithm |
| MXFP8 adapter dgrad via `TN` | pass | 0.037086 |
| MXFP8 adapter wgrad via `TN` | pass | 0.037205 |
| NVFP4 native fprop `TN`, RHT off | pass | 0.144425 |
| NVFP4 native dgrad `NN`, RHT off | pass | 0.145079 |
| NVFP4 native wgrad `NT`, RHT off | pass | 0.133225 |

Shim route probe:

```bash
python tools/probes/te_blockscaled_backward_probe.py --format mxfp8 --use-shim
```

Small shape `M=64, N=96, K=128`:

| Case | Result | rel L2 vs BF16 matmul |
| --- | --- | ---: |
| MXFP8 native dgrad `NN` | fail | cuBLASLt no algorithm |
| MXFP8 native wgrad `NT` | fail | cuBLASLt no algorithm |
| Shim dgrad `NN -> TN` | pass | 0.037086 |
| Shim wgrad `NT -> TN` | pass | 0.037205 |

Shim stats:

```text
mxfp8_tn_adapter_dgrad=1
mxfp8_tn_adapter_wgrad=1
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
fallback_reasons={}
```

Projection-like shim route probe:

```bash
python tools/probes/te_blockscaled_backward_probe.py \
  --format mxfp8 --use-shim --m 256 --n 4096 --k 4096
```

| Case | Result | rel L2 vs BF16 matmul |
| --- | --- | ---: |
| MXFP8 native dgrad `NN` | fail | cuBLASLt no algorithm |
| MXFP8 native wgrad `NT` | fail | cuBLASLt no algorithm |
| Shim dgrad `NN -> TN` | pass | 0.037767 |
| Shim wgrad `NT -> TN` | pass | 0.037756 |

Shim stats:

```text
mxfp8_tn_adapter_dgrad=1
mxfp8_tn_adapter_wgrad=1
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
fallback_reasons={}
```

3D wgrad coverage smoke:

```bash
CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1 \
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0 \
CPPMEGA_TE_MXFP8_BWD_DEBUG=1 \
python - <<'PY'
# quantize x[4,16,128], dy[4,16,96], weight[96,128],
# then call wrapped general_gemm with native NN/NT layouts.
PY
```

Result:

```text
dgrad_shape=(4, 16, 128), rel_l2=0.037086
wgrad_shape=(96, 128), rel_l2=0.037205
mxfp8_tn_adapter_dgrad=1
mxfp8_tn_adapter_wgrad=1
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
```

## High-Level TE Linear Smoke

Command:

```bash
PYTHONPATH=scripts:$PYTHONPATH \
CPPMEGA_ALLOW_TE_MXFP8_SM12=1 \
CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1 \
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0 \
CPPMEGA_TE_MXFP8_BWD_DEBUG=1 \
python - <<'PY'
import torch
import cppmega_fp8_shim
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

torch.manual_seed(123)
layer = te.Linear(128, 96, bias=False, params_dtype=torch.bfloat16).cuda()
x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
with te.fp8_autocast(enabled=True, fp8_recipe=recipe.MXFP8BlockScaling()):
    y = layer(x)
    loss = y.float().pow(2).mean()
loss.backward()
torch.cuda.synchronize()
print(cppmega_fp8_shim.cppmega_te_mxfp8_bwd_stats_snapshot())
PY
```

Result:

```text
loss=0.0678034276
x_grad=torch.bfloat16, shape=(64, 128)
w_grad=torch.bfloat16, shape=(96, 128)
mxfp8_tn_adapter_dgrad=1
mxfp8_tn_adapter_wgrad=1
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
```

## Real-Data 1-Step Smoke

This worktree intentionally did not copy unrelated dirty main-worktree changes
for Mamba MXFP8 padding, TE precision config, and launcher wiring. To run a
real-data smoke without editing either checkout, I used a temporary overlay
root:

- `cppmega/` symlinked to `/home/dave/source/cppmega/cppmega`
- `configs/` symlinked to `/home/dave/source/cppmega/configs`
- `scripts/cppmega_fp8_shim.py` symlinked to this agent worktree

Command:

```bash
TMPROOT=/tmp/cppmega-te-bwd-real-smoke-root
rm -rf "$TMPROOT"
mkdir -p "$TMPROOT/scripts"
ln -s /home/dave/source/cppmega/cppmega "$TMPROOT/cppmega"
ln -s /home/dave/source/cppmega/configs "$TMPROOT/configs"
ln -s /home/dave/source/cppmega-te-backward-agent/scripts/cppmega_fp8_shim.py \
  "$TMPROOT/scripts/cppmega_fp8_shim.py"

RUN_ID=smoke_mxfp8_tn_adapter_1it_20260425_retry2
ROOT="$TMPROOT" \
RUN_ID="$RUN_ID" \
LOG="/home/dave/logs/${RUN_ID}.log" \
NVSMI_LOG="/home/dave/logs/${RUN_ID}.nvsmi.log" \
CPPMEGA_TE_PRECISION_CONFIG_FILE="$TMPROOT/configs/te_precision/mamba_mxfp8_eval_bf16.yaml" \
CPPMEGA_TRAIN_ITERS=1 \
CPPMEGA_MICRO_BATCH_SIZE=4 \
CPPMEGA_GLOBAL_BATCH_SIZE=4 \
CPPMEGA_ALLOW_TE_MXFP8_SM12=1 \
CPPMEGA_PAD_MAMBA_IN_PROJ_FOR_MXFP8=1 \
CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1 \
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0 \
CPPMEGA_TE_MXFP8_BWD_DEBUG=1 \
CPPMEGA_TE_MXFP8_DGRAD_BF16=0 \
CPPMEGA_TE_MXFP8_WGRAD_BF16=0 \
NVTE_BACKWARD_OVERRIDE=none \
TORCH_EXTENSIONS_DIR=/home/dave/.cache/torch_extensions/cppmega_gb10_ab \
timeout 900 /home/dave/source/cppmega/scripts/local_gb10_quarter_train.sh
```

Result: pass, one real-data training iteration completed.

Metrics:

```text
iteration=1/1
elapsed_time_per_iteration_ms=31292.2
lm_loss=1.165876E+01
mtp_1_loss=1.164851E+01
grad_norm=77.980
skipped_iterations=0
nan_iterations=0
max_allocated=23519.87 MB
validation_lm_loss=1.015219E+01
test_lm_loss=1.012940E+01
```

Adapter acceptance counters:

```text
mxfp8_tn_adapter_dgrad=6
mxfp8_tn_adapter_wgrad=6
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
fallback_reasons={}
```

The first real-data attempt before the ND flatten fix failed intentionally with
fallback disabled:

```text
mxfp8_tn_adapter_dgrad=1
mxfp8_tn_adapter_wgrad=0
native_passthrough_wgrad=1
fallback_reasons={'RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D': 1}
```

That failure identified the real Mamba wgrad input shape issue and is fixed by
flattening leading dimensions before retargeting columnwise MXFP8 payloads.

## NVFP4 State

## Real-Data 100-Step Main-Worktree Run

After merging the adapter into the main checkout, this command completed 100
real-data GB10 steps:

```bash
RUN_ID=ab100_mamba_mxfp8_tn_adapter_mbs4_20260425 \
CPPMEGA_TE_PRECISION_CONFIG_FILE=/home/dave/source/cppmega/configs/te_precision/mamba_mxfp8_eval_bf16.yaml \
CPPMEGA_TRAIN_ITERS=100 \
CPPMEGA_MICRO_BATCH_SIZE=4 \
CPPMEGA_GLOBAL_BATCH_SIZE=4 \
CPPMEGA_TORCH_PROFILE=0 \
CPPMEGA_NSYS_PROFILE=0 \
scripts/local_gb10_quarter_train.sh
```

Result:

```text
avg_ms_10_100=4501.460
lm_loss_100=1.566162
mtp_1_loss_100=1.690768
validation_lm_loss_100=1.826358
test_lm_loss_100=1.902454
max_allocated=25737.67 MB
max_reserved=27320 MB
skipped_iterations=0
nan_iterations=0
mxfp8_tn_adapter_dgrad=600
mxfp8_tn_adapter_wgrad=600
bf16_fallback_dgrad=0
bf16_fallback_wgrad=0
native_passthrough_dgrad=0
native_passthrough_wgrad=0
fallback_reasons={}
```

Comparison on the same 100-step real-data setup:

| Run | Avg ms/iter 10-100 | Train lm @100 | Val lm @100 | Test lm @100 | Max allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| TE tensorwise default | 4460.641 | 1.568469 | 1.820359 | 1.891402 | 25697.59 MB |
| Old MXFP8 BF16 backward bridge | 4624.763 | 1.577934 | 1.835458 | 1.890061 | 25735.36 MB |
| MXFP8 TN backward adapter | 4501.460 | 1.566162 | 1.826358 | 1.902454 | 25737.67 MB |

This makes the TN adapter the accepted GB10 Mamba-MXFP8 route. It is close to
the tensorwise default and faster than the old BF16 bridge, but not yet a net
memory or throughput win.

NVFP4 is not covered by the MXFP8 TN adapter. Low-level TE 2.14 native NVFP4
backward `NN`/`NT` GEMMs work on GB10 only when random Hadamard transform is
disabled:

```python
NVFP4BlockScaling(
    disable_rht=True,
    disable_stochastic_rounding=True,
)
```

Equivalent env defaults in installed TE 2.14:

```bash
NVTE_NVFP4_DISABLE_RHT=1
NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1
```

Default high-level NVFP4 with RHT still fails on this stack.

## Upstream And Kernel Candidates

Local `/home/dave/TransformerEngine` and installed TE 2.14 were inspected.
Upstream TE `origin/main` has `backward_override={None,"high_precision",
"dequantized"}` support, but that is the old bridge class of solution:

- `high_precision` saves original high-precision operands for backward.
- `dequantized` consumes saved quantized payloads, then dequantizes them before
  backward GEMMs.

Neither is a keep-MXFP8 backward path.

DeepGEMM and Hugging Face `finegrained-fp8` are not drop-in replacements for
TE Linear backward because they do not consume TE MXFP8/NVFP4 tensor storage or
autograd contracts directly.

## Decision

No drop-in replacement exists.

The best current path is the gated shim prototype:

```bash
CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1
CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=0
```

It is validated for covered Mamba MXFP8 projections in a 100-step real-data
GB10 run with `adapter_count > 0` and `fallback_count == 0`.

Remaining BF16 fallback is only:

1. explicit old bridge mode via `CPPMEGA_TE_MXFP8_DGRAD_BF16=1`,
   `CPPMEGA_TE_MXFP8_WGRAD_BF16=1`, or
   `NVTE_BACKWARD_OVERRIDE=dequantized/high_precision`; or
2. emergency adapter fallback when
   `CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=1`.

Both paths are counted/logged and should not be reported as end-to-end MXFP8
backward runs.
