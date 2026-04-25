# Dense Block-Scaled Weight Cache Probe (2026-04-25)

## Purpose

This probe checks whether a practical dense MXFP8/NVFP4 path exists on the
local GB10 stack without quantizing weights before every GEMM.

The answer is yes for forward GEMM/probe scope: Transformer Engine already
exposes cached weight storage through
`TransformerEngineBaseModule.get_weight_workspace(cache_name="weight")`, and
`general_gemm` can consume the cached `MXFP8Tensor` / `NVFP4Tensor` directly.
No custom dense kernel is needed for this path.

For training integration, this means the viable contract is:

1. Quantize activations every forward call.
2. Quantize/update weights only when the optimizer has changed the BF16/FP16
   parameter, matching TE's `is_first_microbatch` / weight workspace contract.
3. Reuse cached weight workspace for repeated forward/recompute calls within
   the same optimizer step.

## Inspected TE Exposure

Environment:

```text
device=NVIDIA GB10, capability=(12, 1)
torch=2.13.0.dev20260417+cu132
transformer_engine=2.14.0
```

Relevant public APIs:

```text
Float8CurrentScalingQuantizer(fp8_dtype, device, *, rowwise=True, columnwise=True, ...)
MXFP8Quantizer(fp8_dtype, *, rowwise=True, columnwise=True)
NVFP4Quantizer(fp4_dtype=kFloat4E2M1, rowwise=True, columnwise=True, ...)
general_gemm(A, B, out_dtype=None, quantization_params=None, ..., layout="TN", ...)
get_weight_workspace(tensor=None, quantizer=None, cache_name=None,
                     update_workspace=True, skip_update_flag=None,
                     fsdp_group=None, workspace_dtype=None)
```

TE Linear uses the same mechanism internally:

```text
cache_name = None if is_first_microbatch is None else "weight"
update_workspace = is_first_microbatch is None or is_first_microbatch
weightmat = module.get_weight_workspace(..., cache_name=cache_name,
                                        update_workspace=update_workspace)
```

Megatron's TE wrappers keep `disable_parameter_transpose_cache=False` by
default and pass `is_first_microbatch` to TE, so TE's weight workspace is the
right integration point for dense cached block-scaled weights.

## Timing

Command:

```bash
source /home/dave/cppmega-venv/bin/activate
cd /home/dave/source/cppmega-dense-fp8-cache-agent
scripts/probe_te_dense_blockscaled_cache.py --warmup 20 --iters 100 --verbose
```

Shape: `x=(256, 4096)`, `weight=(4096, 4096)`, output dtype BF16.

| Format | Mode | ms/iter | rel_l2 | Note |
| --- | --- | ---: | ---: | --- |
| BF16 | `torch_matmul` | 0.2416 | | Baseline `x @ weight.T` |
| MXFP8 | `quantize_weight_each_call` | 0.4134 | 0.037740 | Weight + activation quantized in timed loop |
| MXFP8 | `prequantized_weight_gemm` | 0.0837 | 0.037740 | Weight quantized once, activation each call |
| MXFP8 | `te_workspace_update_each_call` | 0.4178 | 0.037740 | `get_weight_workspace(..., update_workspace=True)` every call |
| MXFP8 | `te_workspace_cached_gemm` | 0.0721 | 0.037740 | `update_workspace=False`, same cached object reused |
| NVFP4 | `quantize_weight_each_call` | 0.4492 | 0.134493 | Weight + activation quantized in timed loop |
| NVFP4 | `prequantized_weight_gemm` | 0.0541 | 0.134493 | Weight quantized once, activation each call |
| NVFP4 | `te_workspace_update_each_call` | 0.4499 | 0.134493 | `get_weight_workspace(..., update_workspace=True)` every call |
| NVFP4 | `te_workspace_cached_gemm` | 0.0541 | 0.134493 | `update_workspace=False`, same cached object reused |

## Conclusion

Dense block-scaled GEMM is only practical when the weight workspace is cached.
On this GB10 run, quantizing the weight every call made MXFP8/NVFP4 slower than
BF16. Reusing a prequantized TE weight workspace made MXFP8 about 3.4x faster
than BF16 for this forward GEMM shape, and NVFP4 about 4.5x faster.

This does not by itself make full training merge-ready: the existing MXFP8
training path still depends on TE backward behavior and the local GB10 TN
adapter coverage. The result is a narrow, mergeable probe plus documentation of
the exact TE cache primitive to use.
