# Upstream Megatron-LM Patches

Patches applied to the Megatron-LM installation for NAM56R DSA 9+4 training.

## Target Version

- **Branch**: `dev` (commit `7960a311f` + PR #3674 merged)
- **Files patched**: `megatron/core/transformer/experimental_attention_variant/dsa.py`
  and `megatron/core/transformer/experimental_attention_variant/ops/`

## How to Apply

```bash
# From cppmega root, with Megatron venv activated:
python -m cppmega.megatron.upstream_patches.apply_dsa_cg_patches
```

The script is idempotent — safe to run multiple times.

## Patches

### Patch 1: CUDA Graph Compatibility (dsa.py)

**Problem**: `torch.equal()` and `.item()` are CPU-sync ops banned during
CUDA graph stream capture. DSA's `_build_fused_indexer_varlen_bounds()` calls
these in the forward path, crashing at warmup step 4 with
`"operation not permitted when stream is capturing"`.

**Fix**: Replace `torch.equal(a, b)` checks with `if False:` since we always
use causal masks without variable-length sequences or sequence packing.

**Lines**: 614, 633, 645 in `dsa.py`

### Patch 2: Dimension Hardcodes (dsa.py)

**Problem**: `_fused_sparse_mla_absorbed()` returns `None` (falling back to
slow PyTorch gather-scatter) when `query.size(-1) != 576 or v_channels != 512`.
These are DeepSeek V3.2 dimensions. NAM56R has `d_total=160, v_channels=64`.

**Fix**: Remove the dimension guard — TileLang kernel is fully parameterized.

**Lines**: 1172, 1217 in `dsa.py`

### Patch 3: SparseMLA d_v Propagation (dsa.py + sparse_mla.py)

**Problem**: `SparseMLA.apply(q, kv, idx, scale)` doesn't pass `v_channels`
to the TileLang kernel. Forward uses default `d_v=512`, backward uses `D=512`.
NAM56R needs `d_v=64`.

**Fix**: Add `d_v` parameter to `SparseMLA.forward()`, propagate through
`sparse_mla_fwd_interface()` and `sparse_mla_bwd()`.

**Files**: `dsa.py`, `ops/sparse_mla.py`

### Patch 4: TileLang Forward Assertions (tilelang_sparse_mla_fwd.py)

**Problem**: Forward kernel asserts `dim == next_power_of_2(dim)` and
`tail_dim == next_power_of_2(tail_dim)`. NAM56R has `tail_dim=96` (not power-of-2).

**Fix**: Relax to `dim % 16 == 0` — sufficient for warp-level ops.
Also removes `dim_plus_tail_dim == 576` assertion.

**Lines**: 123-135, 297 in `tilelang_sparse_mla_fwd.py`

### Patch 5: TileLang Backward D=512 Hardcode (tilelang_sparse_mla_bwd.py)

**Problem**: Backward kernel hardcodes `D = 512`. NAM56R has `d_v=64`,
causing `assert dim_plus_tail_dim >= D` to fail (96 < 512).

**Fix**: Replace `D = 512` with `D = d_v if d_v is not None else o.shape[-1]`.

**Line**: ~420 in `tilelang_sparse_mla_bwd.py`

## Performance Impact

| Config | Without patches | With patches |
|--------|----------------|-------------|
| Sparse DSA attention | PyTorch gather-scatter (5.9s/iter, 140 TFLOP/s) | TileLang fused kernel (4.5s/iter, 184 TFLOP/s) |
| CUDA graphs | Crash at warmup step 4 | Works with full scope (attn mamba moe_router moe_preprocess) |
| Peak memory | 135.75 GiB (OOM) | ~55 GiB with TileLang |

## Upstream Status

These issues should be reported upstream:
- CUDA graph torch.equal: trivial fix, could PR
- Dimension hardcodes: upstream only tested with DeepSeek V3.2
- D=512 hardcode: same, upstream limitation
