# PR: SparseMLA backward — use accum_dtype for P/dP shared buffers

## Problem

In the SparseMLA backward TileLang kernel (`tilelang_sparse_mla_bwd.py`), the shared-memory buffers `P_shared_cast` and `dP_shared_cast` are allocated with `dtype` (bf16) before being consumed by the dKV gradient GEMM. The intermediate P and dP values have a wide dynamic range (exp of scaled scores, then multiplied by dO accumulations); storing them in bf16 loses precision in the dKV path and causes gradient drift vs an fp32-reference backward.

## Solution

Allocate `P_shared_cast` and `dP_shared_cast` with `accum_dtype` (fp32) so the dKV GEMM reads fp32 P/dP instead of bf16-rounded values.

This pack is currently a local patch note, not a fully receipted upstream-ready bundle: there is no checked-in example directory or `validation_manifest.yaml` entry yet, so the wording below is intentionally limited to the code-level change and the expected validation target.

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

- Validation receipt is still missing in this tree: no example bundle or manifest entry currently demonstrates fresh gradcheck or H200 throughput evidence for this change.
- Intended validation target: show improved dKV accuracy against an fp32/fp64 reference and confirm no material Hopper regression once a reproducible example bundle is added.
