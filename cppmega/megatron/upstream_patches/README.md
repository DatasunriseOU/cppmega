# Upstream Megatron-LM Patches

Runtime monkey-patches applied to the Megatron-LM + mamba_ssm + TileLang
stack to support NAM56R DSA 9+4 training. All patches are idempotent and
gated where needed by env vars.

## Target versions

- **Megatron**: `dev_latest` (on top of NVIDIA/Megatron-LM `dev`) @
  `core_v0.15.0rc7` + PR #3674 (DSA absorbed MLA) + PR #4268 (delayed
  wgrad overlap)
- **TransformerEngine**: 2.13
- **TileLang**: 0.1.8
- **mamba_ssm**: 2.3.1 (from source, built for sm_121+PTX on GB10)

Files touched:
- `megatron/core/transformer/experimental_attention_variant/dsa.py`
- `megatron/core/transformer/experimental_attention_variant/ops/sparse_mla.py`
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_fwd.py`
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_bwd.py`
- `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_{fwd,bwd}{,_varlen}.py` (via P1 and layout fix patches)

## How to Apply

```bash
# From cppmega root, with Megatron venv activated:
python -m cppmega.megatron.upstream_patches.apply_dsa_cg_patches
# Optional (opt-in):
python -m cppmega.megatron.upstream_patches.apply_mamba3_mimo_p1_patches  # CPPMEGA_MAMBA3_P1=1 to activate
```

All scripts are idempotent — safe to run multiple times. Must re-run
after every Megatron or mamba_ssm re-install.

## Active patches (apply_dsa_cg_patches.py — 10 total)

### Patch 1: CUDA Graph Compatibility (dsa.py)

`torch.equal()` / `.any()` / `.item()` are CPU-sync ops banned during
CUDA graph capture. `_build_fused_indexer_varlen_bounds` calls these
→ crash at warmup step 4 with `"operation not permitted when stream is
capturing"`. Fix replaces validation `torch.equal()` checks with
`if False:` (invariants hold by construction during training).

### Patch 2: Dimension Hardcodes (dsa.py)

`_fused_sparse_mla_absorbed()` returned `None` (falling through to slow
PyTorch gather-scatter) unless `query.size(-1) == 576 and v_channels
== 512`. These are DeepSeek V3.2 dims. NAM56R has `d_total=128,
v_channels=64` (q_lora=64, kv_lora=64, qk_pos=64 → qk=128; v_head_dim=64).
Fix removes the dim guard — TileLang kernel is fully parameterized.

### Patch 3: SparseMLA d_v Propagation (dsa.py + sparse_mla.py)

`SparseMLA.apply(q, kv, idx, scale)` didn't pass `v_channels` to the
kernel. Forward used default `d_v=512`, backward hardcoded `D=512`.
Fix adds `d_v` parameter to `SparseMLA.forward/backward`, propagates
through `sparse_mla_fwd_interface` and `sparse_mla_bwd`.

### Patch 4: sparse_mla.py d_v plumbing

Completes Patch 3 by wiring `d_v` through the autograd Function and
into the low-level interface.

### Patch 5: TileLang Forward Assertions (tilelang_sparse_mla_fwd.py)

Forward asserted `dim == next_power_of_2(dim)` and
`tail_dim == next_power_of_2(tail_dim)`. NAM56R has `tail_dim=96`
(not power-of-2). Fix relaxes to `dim % 16 == 0` (sufficient for warp
ops). Also removes `dim_plus_tail_dim == 576` assertion.

### Patch 6: TileLang Backward D=512 hardcode (tilelang_sparse_mla_bwd.py)

Backward hardcoded `D = 512`, failing `assert dim_plus_tail_dim >= D`
(96 < 512) for NAM56R dims. Fix: `D = d_v if d_v is not None else
o.shape[-1]`.

### Patch 7: tilelang_sparse_mla_bwd.py P/dP precision check

Prior experiment set `P_shared_cast` + `dP_shared_cast` to `accum_dtype`
(fp32) for numerical stability. This broke TileLang's GEMM
same-dtype constraint (A.dtype == B.dtype). Patch 7 now REVERTS to
`dtype` (bf16) — fp32 would be desirable but requires deeper kernel
surgery (branch `fp8-bwd-piggyback-exploration` has the groundwork).

### Patch 8: CG-safe `_scatter_topk_into_index_mask` (dsa.py)

Replaces `if torch.any(idx_chunk < 0): ... valid_topk.any()` branching
with branchless `clamp+scatter+fixup` using `.any(dim=-1)` which
operates on last dim only (no scalar reduction, no CPU sync).

### Patch 9: FP8 SparseMLA dispatch (dsa.py)

`_fused_sparse_mla_absorbed()` detects `QuantizedTensor` input and
dispatches to `SparseMLA_FP8` with zero-copy via `Float8Tensor._data`
+ `._scale_inv`.

### Patch 9b: Remove stray dequantize+FP8 dispatch inconsistency

A `query.dequantize() + key.dequantize()` pair had drifted into installed
dsa.py (from some prior session). This killed zero-copy FP8 by
re-quantizing BF16 tensors per-token before handing them to
`SparseMLA_FP8`. Patch 9b detects and removes the stray dequantize.

## Performance Impact

| Config | Without patches | With patches |
|--------|----------------|-------------|
| Sparse DSA fused path (NAM56R dims) | N/A — returns None, OOM on fallback matmul | Works, ~4% of compute |
| CUDA graphs | crash at warmup step 4 | works at PP>1 (PP=1 has separate CG-memory blocker) |
| Production config (PP=1 EP=4 MBS=8 BF16) | N/A | **289 TFLOP/s europe, 253 bench3** |

## Mamba3 MIMO P1 Patches (opt-in, separate script)

`apply_mamba3_mimo_p1_patches.py` flips `TL_DISABLE_TMA_LOWER` /
`TL_DISABLE_WARP_SPECIALIZED` from `True` to `False` on all 8
`@tilelang.jit` pass_configs in upstream `mamba_ssm.ops.tilelang.mamba3.*`
(fwd, bwd, bwd_varlen, fwd_varlen). Also ensures
`TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True` is present on every
site so GB10 sm_121 keeps working on shapes that remain in-budget.

**Known limitation (2026-04-14)**: bwd kernels fail compile with
`tvm.error.InternalError: Check failed: (shared_layout->InputDim() == 2)`
when TMA is enabled, because Mamba3 bwd uses rank-3 smem
descriptors (`qk_dot_shared [chunk_size, R, R]`). Fixed on separate
branch `tma-layout-fix-3d-to-2d` via `apply_mamba3_mimo_tma_layout_fix.py`
(not yet merged to main — pending H200 perf measurement).

Env-gated: `CPPMEGA_MAMBA3_P1=1` to opt in. Default OFF. See
`docs/mamba3_mimo_p1_notes.md` for correctness tables and the H200
perf TODO.

To apply manually:
```bash
python -m cppmega.megatron.upstream_patches.apply_mamba3_mimo_p1_patches
```

## DualPipeV integration (opt-in, separate script)

`apply_dualpipev_patch.py` hooks Megatron's `setup_model_and_optimizer`
to wrap the model in a 2-rank DualPipeV process group carved out of
the world. Requires `--pipeline-model-parallel-size 1` on the Megatron
side.

Env-gated: `CPPMEGA_DUALPIPEV=1`, default OFF.
`CPPMEGA_DUALPIPEV_CHUNKS` (default 4) controls num_chunks.

**Known limitation (2026-04-14)**: incompatible with `--expert-model-parallel-size
>1`. DualPipeV's V-shape puts ranks at different layers simultaneously,
while DeepEP's `fused_dispatch` A2A requires all EP peers to hit the
same MoE layer synchronously → deadlock at `deep_ep/buffer.py:97`.
Either scope EP within pipe_rank (narrow topology) or disable EP
entirely when using DualPipeV.

## libnvrtc RTLD_GLOBAL workaround (not a patch, but related)

`cppmega/megatron/sparse_mla_ops/__init__.py` force-loads
`libnvrtc.so.13` with `RTLD_GLOBAL` at import time to prevent TileLang
from aborting with *"libnvrtc symbols not found globally"* when
compiling a second kernel variant in the same process (cu13.2 bug).
Candidate for upstream TileLang issue.
