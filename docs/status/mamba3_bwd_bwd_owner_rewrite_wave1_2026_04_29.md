# Mamba3 bwd_bwd Owner/Layout Rewrite Wave 1 - 2026-04-29

Branch: `worker/mamba3-bwd-bwd-owner-rewrite`

Base: `worker/mamba3-bwd-bwd-crossp-accum` / `f77c8f0`

Goal: probe a more radical ownership/layout decomposition for Mamba3 MIMO
`bwd_bwd` that reduces live-set and avoids the discarded 1 GiB fp32
`dstates_before_chunks` handoff path.

## Prototype

File:

- `scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave1.py`

This is a Triton microbench, not a full TileLang replacement. It isolates the
cross-P reduction pattern that dominates the proposed ownership rewrite:

```text
DK-like: [F, P] @ [P, N] -> [F, N]
DQ-like: [F, P] @ [P, N] -> [F, N]
F = chunk_size * R = 64, N = 64, P = 128
```

Compared variants:

- `fullp`: one owner per `(B,H,chunk)` computes the full-P DQ/DK-like GEMMs.
- `ptile_atomic`: one owner per `(B,H,chunk,p_tile)` computes partial GEMMs and
  reduces directly into the final fp32 DQ/DK-like tensors with `tl.atomic_add`.

The candidate deliberately does not create fp32 partial-output scratch. For the
productionish shape, storing DQ/DK partials for `P_TILE=64` would be 2 GiB, and
the discarded fp32 `dstates_before_chunks` path would be 1 GiB.

## Validation

Local:

```text
python -m py_compile scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave1.py
```

Modal H200 image:

- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- GPU: `NVIDIA H200`, capability `(9, 0)`, device count `2`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`
- Triton: `3.7.0`

No web/MCP docs were used for this wave.

## Stage2 Baseline Context

From `docs/status/mamba3_stage2_force_nontma_profile_matrix_2026_04_29.md`,
productionish full-kernel H200 numbers:

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | read |
| --- | ---: | ---: | ---: | --- |
| baseline | 1.8718 | 3.7084 | 5.5628 | upstream non-WS |
| stage2 default `(bf=1,bb=0)` | 1.7886 | 3.6940 | 5.4567 | current best stage2 |

The wave-1 microbench is not directly comparable to the full `bwd_bwd` time,
but its productionish DQ/DK-like full-P reference costs `0.812 ms`, so the
isolated reduction subproblem is large enough to reject bad decompositions.

## H200 Results

Run 1:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave1.py \
  --shape-csv tiny,smoke_p128 --p-tile-csv 64 --warmup 1 --iters 3
```

App: `ap-DcKqt0xICq6GDfk3hw8Fuw`, stopped.

Run 2:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave1.py \
  --shape-csv representative,productionish --p-tile-csv 64 \
  --warmup 2 --iters 8
```

App: `ap-wBYaaIhzrkdjjBXOJwUMsM`, stopped.

Run 3:

```text
CPPMEGA_MODAL_GPU=H200:2 timeout 1200 modal run \
  scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave1.py \
  --shape-csv smoke_p128,representative --p-tile-csv 32,64 \
  --warmup 2 --iters 6
```

App: `ap-kGySwjSWYyHUomORxa1l3F`, stopped.

All correctness checks passed against the full-P Triton reference at
`rtol=1e-2, atol=1e-2`.

| shape | p_tile | full-P ms | atomic compute ms | atomic+zero ms | atomic+zero / full-P | live input reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_p128 | 64 | 0.0243-0.0325 | 0.0217-0.0244 | 0.0329-0.0355 | 1.36x slower | 2.0x |
| smoke_p128 | 32 | 0.0268 | 0.0231 | 0.0468 | 1.74x slower | 4.0x |
| representative | 64 | 0.0444-0.0477 | 0.0686-0.0705 | 0.0787-0.0828 | 1.74x slower | 2.0x |
| representative | 32 | 0.0471 | 0.1138 | 0.1315 | 2.79x slower | 4.0x |
| productionish | 64 | 0.8119 | 1.7801 | 1.9991 | 2.46x slower | 2.0x |

Productionish memory model for `P_TILE=64`:

- final fp32 DQ/DK-like outputs: `1,073,741,824` bytes
- fp32 DQ/DK partial handoff if not atomic: `2,147,483,648` bytes
- discarded fp32 `dstates_before_chunks` path: `1,073,741,824` bytes
- extra atomic handoff: `0` bytes
- per-owner live input: `65,536` bytes full-P vs `32,768` bytes p-tile

## Read

Direct atomic reductions are correctness-feasible and eliminate the giant fp32
partial handoff, but they are too slow at production fanout. On H200
productionish, the isolated DQ/DK-like subproblem regresses from `0.812 ms` to
`1.999 ms` when zeroing is included. `P_TILE=32` reduces per-owner live input
to 16 KiB, but the extra atomic programs make representative worse than
`P_TILE=64`.

This rejects the direct-atomic `per-(B,H,chunk,p_tile)` rewrite as the primary
wave-2 direction.

## Integration Plan

Do not port `ptile_atomic` directly into TileLang `mamba_mimo_bwd_bwd`.

Recommended next wave:

1. Keep chunk ownership, but split only the diagonal/DQ/DK live-set inside the
   existing non-WS `bwd_bwd` owner rather than splitting P ownership across
   programs.
2. Specialize the H200 `P=128` path with two serial `P_TILE=64` passes inside
   one `(B,H,chunk)` owner, using on-chip accumulators for DQ/DK and avoiding
   atomics.
3. Hoist or recompute only the tensors that shrink the live-set enough to allow
   better register allocation; do not write fp32 DQ/DK partials or
   `dstates_before_chunks` to global memory.
4. Keep the current best stage2 setting `(bf=1, bb=0)` as the integration
   baseline: `bwd_bwd` WS/TMA remains a regression until live-set pressure is
   reduced.

Modal cleanup:

- Wave-1 apps `ap-DcKqt0xICq6GDfk3hw8Fuw`,
  `ap-wBYaaIhzrkdjjBXOJwUMsM`, and `ap-kGySwjSWYyHUomORxa1l3F` completed and
  stopped.
- Leftover ephemeral Modal apps `ap-ec0qSMP2jCagbyO66xu7cs`,
  `ap-ANV25CfKNjO9P3079VxSgE`, `ap-j0JEH6jXpopRUr8lCrP4GF`, and
  `ap-39bPeh65V5iD0EopLwDavx` were stopped with `modal app stop -y`.
