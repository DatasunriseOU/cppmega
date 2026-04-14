# PR: SparseMLA backward — use accum_dtype for P/dP shared buffers

## Problem

In the SparseMLA backward TileLang kernel (`tilelang_sparse_mla_bwd.py`), the shared-memory buffers `P_shared_cast` and `dP_shared_cast` are allocated with `dtype` (bf16) before being consumed by the dKV gradient GEMM. The intermediate P and dP values have a wide dynamic range (exp of scaled scores, then multiplied by dO accumulations); storing them in bf16 loses precision in the dKV path and causes gradient drift vs an fp32-reference backward.

## Solution

Allocate `P_shared_cast` and `dP_shared_cast` with `accum_dtype` (fp32) so the dKV GEMM reads fp32 P/dP instead of bf16-rounded values.

```python
# Before:
P_shared_cast = T.alloc_shared([...], dtype)
dP_shared_cast = T.alloc_shared([...], dtype)

# After:
P_shared_cast = T.alloc_shared([...], accum_dtype)
dP_shared_cast = T.alloc_shared([...], accum_dtype)
```

## Files Changed

- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_bwd.py`

## Testing

- dKV gradient matches `torch.autograd.gradcheck` at fp64 reference within tightened tolerance (bf16 P/dP buffers exceed tolerance; fp32 buffers pass).
- No measurable throughput change on 8×H200 (shared-memory budget still fits within the 99 KiB Hopper cap for the kernel tile).
