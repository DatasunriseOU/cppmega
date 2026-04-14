# PR: FP8 SparseMLA dispatch for absorbed MLA path

## Problem

When FP8 training is enabled (`--fp8-format hybrid --fp8-recipe tensorwise`), Transformer Engine wraps tensors in `QuantizedTensor` (Float8Tensor). These wrappers have several properties that break downstream TileLang kernels:

- `.dtype` returns the logical dtype (bf16), hiding the actual FP8 storage
- `.data_ptr()` returns NULL (the real data is at `._data.data_ptr()`)
- `.to()`, `.contiguous()`, `.reshape()` do NOT unwrap — the tensor stays as Float8Tensor
- Only `.dequantize()`, `.float()`, `.permute()`, `.unsqueeze()` actually unwrap

When `_fused_sparse_mla_absorbed()` passes these Float8Tensors to the TileLang SparseMLA kernel, the kernel gets NULL data pointers and crashes.

**TE version caveat**: With Transformer Engine ≥ 2.13, the `__torch_dispatch__`
hook silently auto-dequantizes `Float8Tensor` when passed to a raw CUDA
kernel. So the headline `RuntimeError: data pointer expected non-NULL`
does NOT fire on current stacks — instead users pay silent 2× memory
bandwidth (asked for FP8, got BF16 on an auto-dequantized tensor). The
underlying hazards (NULL `data_ptr()`, lying `.dtype=bf16`, `.contiguous()`
does not unwrap) are still present; only `.dequantize()` unwraps. The
dispatch fix is therefore about correctness of intent, not crash prevention.

## Solution

Detect `QuantizedTensor` inputs in `_fused_sparse_mla_absorbed()` and dispatch to an FP8-aware SparseMLA variant:

```python
_use_fp8_mla = False
try:
    from transformer_engine.pytorch.tensor import QuantizedTensor
    if isinstance(query, QuantizedTensor) or isinstance(key, QuantizedTensor):
        _use_fp8_mla = True
except ImportError:
    pass

if _use_fp8_mla:
    _mla_fn = SparseMLA_FP8  # FP8 TileLang kernel (2x throughput on H200 WGMMA)
else:
    _mla_fn = SparseMLA      # BF16 TileLang kernel
```

The FP8 variant (`SparseMLA_FP8`) uses `T.float8_e4m3fn` dtype in TileLang GEMMs, getting 2x tensor-core throughput on Hopper WGMMA and 50% shared memory reduction (1B vs 2B per element).

For models without a custom `SparseMLA_FP8`, a simpler fallback is to dequantize before the kernel call:

```python
if isinstance(query, QuantizedTensor):
    query = query.dequantize()
if isinstance(key, QuantizedTensor):
    key = key.dequantize()
```

## Files Changed

- `megatron/core/transformer/experimental_attention_variant/dsa.py`

## Testing

- Verified with `--fp8-format hybrid --fp8-recipe tensorwise` on 8xH200
- Without fix: `RuntimeError: kernel main input Q data pointer expected non-NULL, but got NULL`
- With dequantize: training runs, SparseMLA uses bf16 path
- With SparseMLA_FP8: training runs, 2x SparseMLA GEMM throughput
