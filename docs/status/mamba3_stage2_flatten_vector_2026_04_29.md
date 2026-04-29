# Mamba3 Stage-2 Float32 Vector Flatten - 2026-04-29

Branch: `worker/mamba3-stage2-flatten-vector`

Base: `worker/mamba3-hopper-true-ws` (`884c797`)

Goal: test an alternative `num_stages=2` fix that keeps TMA enabled for the
large Q/K/QK producer copies and makes the failing small `float32` vector
descriptors legal by flattening them from `[B,H,S]` to `[B*H,S]` at the
TileLang kernel ABI.

## Artifacts

- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_flatten_vector.patch`
- `scripts/modal_mamba3_stage2_flatten_vector_probe.py`

The patch is non-production. It includes the earlier Q/K/QK rank-2 TMA layout
changes, enables TMA + warp specialization, sets wrapper defaults to
`bf_num_stages=2` / `bb_num_stages=2`, adds the float32 vector flattening
described below, and includes the final attempted `STATES` flatten that exposed
the remaining blocker.

## Flattened Tensors

The failing descriptor from the true-WS probe was:

```text
format CU_TENSOR_MAP_DATA_TYPE_FLOAT32
dim 3
globalDim [64, 4, 1]
globalStridesRaw [4, 256, 1024]
boxDim [16, 1, 1]
```

That corresponds to a contiguous chunk slice from a public `[B,H,S]` float32
tensor with `B=1`, `H=4`, `S=64`, `chunk_size=16`. The stage-2 patch changes
the TileLang ABI for these tensors to `[B*H,S]`, preserving the public wrapper
layout with zero-copy `.view(B * H, S)`.

Inputs flattened:

- `DA_CS`
- `DA_CS_REV`
- `DT`

Outputs flattened:

- `DFACTOR`
- `DGAMMA_DIAG`
- `DDA`
- `DDA_CS_REV`
- `DDA_CS`

The row index is `i_bh = i_b * H + i_h`. A slice such as:

```python
DA_CS[i_b, i_h, chunk_start:chunk_start + chunk_size]
```

becomes:

```python
DA_CS[i_bh, chunk_start:chunk_start + chunk_size]
```

For the smoke shape, the descriptor should become rank-2:

```text
globalDim [64, 4]
globalStridesRaw [4, 256]
boxDim [16, 1]
```

The base row offset is `(i_b * H + i_h) * S * sizeof(float32)`. With `S=64`,
that is a 256-byte stride between rows, and each chunk starts at
`chunk_start * sizeof(float32)`, a 64-byte multiple for `chunk_size=16`.

## Public Wrapper Semantics

The public wrapper still accepts and returns the original `[B,H,S]` tensors.
Only the kernel call boundary changes:

```python
bh = B * H
dA_cs_flat = dA_cs.view(bh, S)
dA_cs_rev_flat = dA_cs_rev.view(bh, S)
dt_flat = dt.view(bh, S)
dfactor_flat = dfactor.view(bh, S)
dgamma_diag_flat = dgamma_diag.view(bh, S)
ddA_flat = ddA.view(bh, S)
ddA_cs_rev_flat = ddA_cs_rev.view(bh, S)
ddA_cs_flat = ddA_cs.view(bh, S)
```

Q/K remain flattened from `[B,S,R,G,N]` to `[B,S*R,G,N]`, and QK_DOT remains
flattened from `[B,H,S,R,R]` to `[B,H,S,R*R]`, so the large producer copies stay
on the TMA path.

The final patch also flattens the intermediate BF16 `STATES` ABI after the
float32 vector fix exposed it as the next failing descriptor:

```python
states_flat = states.view(B * H * ceildiv(S, chunk_size) * N, P)
```

The kernel indexes each chunk/head row with `i_state = (i_b * H + i_h) *
nchunks + chunk_idx`, then copies `STATES[i_state * N:i_state * N + N, :]`.
This preserves the public `[B,H,nchunks,N,P]` storage and presents TileLang with
a 2D `[B*H*nchunks*N,P]` tensor.

## Local Sanity

```text
python -m py_compile scripts/modal_mamba3_stage2_flatten_vector_probe.py
patch --dry-run -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_stage2_flatten_vector.patch
```

Both passed locally.

## Modal Results

All runs used `GHCR_REF=ghcr.io/jewelmusicee/cppmega:latest`, TileLang
`0.1.8+cu132.gitf309d814`, Torch `2.13.0.dev20260426+cu132`, and CUDA `13.2`.

App IDs:

- H100 vector-only iteration: `ap-AJLjZZmPzT43vCA7PIkHpp`
- H100 vector + rank-3 `STATES` iteration: `ap-WaiI3xH29mIY0TJEbhhhlF`
- H100 final vector + 2D `STATES` iteration: `ap-Yf42drahLvPsggHQMCZ5d8`
- H200 final vector + 2D `STATES` iteration: `ap-lfPf5kh0U8mrIXcuGlNPSi`

The original float32 vector descriptor was eliminated. The probe metadata for
the final patch reported:

```text
flat_vector_signature_refs: 11
rank3_float_vector_signature_refs: 0
flat_states_signature_refs: 2
disable_tma_lower_false_refs: 2
disable_tma_lower_true_refs: 0
```

Stage-2 compile succeeded and true WS fired on both H100 and H200:

| GPU | `bf_num_stages` | `bb_num_stages` | bwd_fwd WS | bwd_bwd WS | tma_loads |
| --- | --- | --- | --- | --- | --- |
| H100 | 2 | 2 | yes | yes | 5 / 8 |
| H200 | 2 | 2 | yes | yes | 5 / 8 |

Stage-2 smoke did not pass. The failure moved forward:

1. Vector-only flatten removed the original float32 descriptor
   `[64,4,1]` / `boxDim [16,1,1]`, then failed on BF16 `STATES`
   rank-5 descriptor:

   ```text
   format CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
   dim 5
   globalDim [64, 64, 4, 4, 1]
   boxDim [64, 64, 1, 1, 1]
   ```

2. Flattening `STATES` to `[B*H*nchunks,N,P]` changed that to rank 3:

   ```text
   format CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
   dim 3
   globalDim [64, 64, 16]
   boxDim [64, 64, 1]
   ```

3. Final flattening `STATES` to `[B*H*nchunks*N,P]` changed it to rank 2, but
   CUDA still rejected the swizzled BF16 map on both H100 and H200:

   ```text
   format CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
   dim 2
   globalDim [64, 1024]
   globalStridesRaw [2, 128]
   boxDim [64, 64]
   swizzle CU_TENSOR_MAP_SWIZZLE_128B
   CUDA_ERROR_MISALIGNED_ADDRESS
   ```

Correctness diff did not run because the final stage-2 smoke failed before
completion. The included harness still contains the correctness comparator
against upstream TMA-disabled `num_stages=0`, guarded behind smoke success.

## Conclusion

Flattening fixed the specific small float32 vector descriptor blocker. It did
not fix stage-2 end to end. With the original descriptor gone, `bwd_bwd` hits a
separate BF16 `STATES` TMA descriptor legality/runtime issue that persists even
when represented as a rank-2 contiguous `[64,1024]` / `boxDim [64,64]` map.

## Non-TMA Contrast

The per-copy `T.copy(..., disable_tma=True)` route can avoid the bad descriptor
by taking the small vector copies off TMA. This branch does not use that route:
it keeps those copies representable as legal rank-2 TMA maps and leaves the
large Q/K/QK TMA producers enabled. The Modal harness uses the upstream
TMA-disabled stage-0 kernel only as a correctness oracle, not as the proposed
fix.
