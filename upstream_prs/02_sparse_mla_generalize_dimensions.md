# PR: Generalize SparseMLA TileLang kernel dimensions

## Problem

The fused SparseMLA TileLang kernels are hardcoded for DeepSeek V3.2 dimensions:
- `dsa.py:_fused_sparse_mla_absorbed()` returns `None` if `query.size(-1) != 576 or v_channels != 512`
- `tilelang_sparse_mla_fwd.py` asserts `dim_plus_tail_dim == 576`
- `tilelang_sparse_mla_bwd.py` hardcodes `D = 512`
- Both fwd/bwd assert `dim == next_power_of_2(dim)` (unnecessarily restrictive)

Any model with different MLA dimensions (e.g., kv_lora_rank=64, qk_pos_emb_head_dim=64 → d_total=128, v_channels=64) falls through to the **unfused attention path** which materializes the full attention matrix — O(seq² × heads) memory, causing OOM on large sequences.

## Solution

### 1. Remove dimension guard in dsa.py

The TileLang kernel is already parameterized over dimensions — the guard is unnecessary:

```python
# Before:
if query.size(-1) != 576 or v_channels != 512:
    return None

# After: remove this check entirely
```

### 2. Propagate d_v through SparseMLA autograd Function

The `SparseMLA.apply()` call does not pass `v_channels` to the kernel. Add `d_v` parameter:

```python
# dsa.py:
out, _ = SparseMLA.apply(q_t, kv_t, idx_t, softmax_scale, v_channels)

# sparse_mla.py SparseMLA.forward:
def forward(ctx, q, kv, indices, scaling, d_v=512):
    ctx.d_v = d_v
    tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling, d_v=d_v)

# sparse_mla.py SparseMLA.backward:
    d_v = ctx.d_v
    tl_dq, tl_dkv = sparse_mla_bwd(q, kv, ..., d_v=d_v)
```

### 3. Relax dimension assertions in TileLang kernels

```python
# tilelang_sparse_mla_fwd.py — before:
assert dim == tilelang.math.next_power_of_2(dim)
assert dim_plus_tail_dim == 576

# After:
assert dim % 16 == 0, f"dim must be multiple of 16 for warp ops, got {dim}"
# Remove 576 assertion — kernel handles any dim_plus_tail_dim
```

### 4. Remove D=512 hardcode in backward

```python
# tilelang_sparse_mla_bwd.py — before:
D = 512

# After:
def sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=None, d_v=None, ...):
    D = d_v if d_v is not None else o.shape[-1]
```

## Files Changed

- `megatron/core/transformer/experimental_attention_variant/dsa.py`
- `megatron/core/transformer/experimental_attention_variant/ops/sparse_mla.py`
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_fwd.py`
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_bwd.py`

## Out of scope

The `P_shared_cast` / `dP_shared_cast` fp32 precision fix in the backward kernel is tracked separately in `14_sparse_mla_precision.md` — it is independent of dimension generalization.

## Testing

- Verified with d_total=128, v_channels=64 (kv_lora_rank=64, qk_pos_emb_head_dim=64) on 8xH200
- Fused SparseMLA kernel compiles and runs correctly for non-576/512 dimensions
- Loss convergence matches unfused baseline (before this PR, unfused was the only path for non-DeepSeek dims)
- Backward gradients match torch.autograd.gradcheck at float64
