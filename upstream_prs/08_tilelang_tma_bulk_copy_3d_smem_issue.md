# Issue: `LowerBulkCopy` asserts `InputDim()==2` for 3D smem layouts

**Target repo:** `tile-ai/tilelang`

## Summary

TileLang's TMA bulk-copy lowering path requires `shared_layout->InputDim()
== 2` and asserts otherwise, preventing TMA from being used with
rank-3+ shared-memory descriptors even when the underlying PTX
(`cp.async.bulk.tensor.3d` etc.) supports them.

## Reproducer

Minimal repro compiled cleanly on cu13.2 + TileLang 0.1.8, sm_121a /
sm_90a, with `TL_DISABLE_TMA_LOWER=True` but fails with the flag set to
`False`:

```python
import tilelang as tl
import tilelang.language as T

@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    },
)
def k(X: T.Tensor([128, 4, 64], "bfloat16"),
      Y: T.Tensor([128, 4, 64], "bfloat16")):
    with T.Kernel(1) as (_,):
        xs = T.alloc_shared([16, 4, 64], "bfloat16")  # 3D smem
        T.copy(X[0:16, :, :], xs)
        # ... compute ...
        T.copy(xs, Y[0:16, :, :])

k(x, y)
```

Full repro at
https://github.com/DatasunriseOU/cppmega/blob/tma-layout-fix-3d-to-2d/scripts/exploration/tma_layout_repro.py
(requires running on NVIDIA GB10 / H100-class hardware).

## Error

```
tvm.error.InternalError: Check failed: (shared_layout->InputDim() == 2)
is false: Cannot detect TMA layout.
```

Origin: `tvm::tl::CopyNode::LowerBulkCopy` in
`src/transform/lower_tile_op.cc` (search for the `InputDim()==2` check).

## Why this matters

Mamba3 MIMO backward kernels use three rank-3 smem descriptors:
- `qk_dot_shared` is structurally 3D (`[chunk_size, R, R]`)
- Q/K loads land in `[chunk_size, R, N]` smem views

These kernels cannot enable TMA+warp-spec (the standard Hopper
throughput win — 20-30% reported by NVIDIA on similar kernels) without
first refactoring to 2D layouts. We landed such a refactor as a
workaround in state-spaces/mamba (see linked PR) but TileLang should
probably support 3D TMA natively, since PTX exposes it:

- `cp.async.bulk.tensor.2d.shared::cta.global` ✓ supported
- `cp.async.bulk.tensor.3d.shared::cta.global` ✗ blocked by assertion
- `cp.async.bulk.tensor.4d` ✗ same
- `cp.async.bulk.tensor.5d` ✗ same

## Possible fixes

1. **Short-term**: gate the assertion on rank and fall back to
   `cp.async` (non-bulk) for >2D smem, rather than hard-asserting. This
   preserves compile-time viability without regressing the 2D-TMA
   fast path.

2. **Proper fix**: extend the TMA descriptor construction to handle
   rank-3+ layouts. `cp.async.bulk.tensor.3d.*` PTX exists; generalising
   `DetectTMALayout` to emit 3D descriptors should be straightforward.

3. **Document current limitation**: even if 1 or 2 aren't implemented,
   update TileLang docs to list the 2D-only constraint so users don't
   hit a compile crash.

## Workaround we're using

We flattened Mamba3 MIMO bwd's 3D smem to 2D in
https://github.com/state-spaces/mamba/pull/<TBD> (linked once the PR is
opened). This sidesteps the assertion entirely for our kernel family
but doesn't help other Mamba-class kernels with intrinsically 3D state.

## Environment

- PyTorch 2.12 nightly + cu132
- TileLang 0.1.8
- TVM 0.22 (bundled)
- NVIDIA GB10 (sm_121a, 100 KiB smem cap) and H200 SXM (sm_90a)
