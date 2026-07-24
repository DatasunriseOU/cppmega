# FA4 `score_mod` Backend for cppmega Graph-Route Attention Bias — Design

Status: design only (no implementation in this commit)
Owners: cppmega attention/runtime
Targets: H200 (SM90) and Blackwell (SM100/SM120); FA4 = `flash-attn-4[cu13]==4.0.0b19`,
namespace `flash_attn.cute` (per `STACK.lock` and `docker/Dockerfile`).

---

## 1. Problem Statement

`cppmega/megatron/graph_route_attention_bias_patch.py` currently injects the
compiler-derived graph prior into dense TE/GQA attention by materializing a
**dense** `[B, 1, Sq, Sk]` bias tensor and forwarding it through
`TransformerLayer.forward(..., attention_bias=...)` to
`TEDotProductAttention` → TE `FusedAttention` (cuDNN backend).

Two problems:

1. **Memory blow-up.** `[B,1,Sq,Sk]` is `O(B·Sq·Sk)` even though the graph
   prior is sparse (a handful of edges per row). The module already carries a
   fail-loud cap `CPPMEGA_GRAPH_DENSE_MAX_SEQ=16384` precisely because bf16
   dense bias is ~4 GiB at `B=8, S=16384`. Long-context cppmega cannot use
   this path.
2. **Kernel regression.** Passing a non-`None` `attention_bias` forces TE off
   the FA3/FA4 fast path and onto cuDNN `FusedAttention`. We lose FA4's
   pipelined Hopper/Blackwell kernels and the CuTe-DSL extension surface.

The graph routes are inherently sparse:

- `graph_call_edges` / `graph_type_edges` — chunk-index pairs, expanded
  through `graph_chunk_starts/ends` into token-span rectangles.
- `graph_domain_edges`, `graph_build_edges`, `graph_shell_edges`,
  `graph_diagnostic_edges`, `graph_cross_domain_edges` — token-position
  triples `(src, dst, kind)`.
- `graph_generated_query_edges` — token-position pairs added at decode time.

We want a backend that:

- carries the prior as **CSR-style edge tensors** plus a coarse
  **block-sparse mask** at 128×128 granularity,
- runs inside FA4 via `score_mod` + `aux_tensors` + `block_sparse_tensors`,
- is a drop-in `core_attention` for Megatron `ModuleSpec`,
- preserves the existing fail-closed contract (no silent token-only fallback),
- supports both prefill (square) and incremental decode (rectangular
  `Sq << Sk`).

---

## 2. Current Edge Format (recap from source)

Helpers in `cppmega/megatron/dsa_indexer_fused_patch.py`:

- `_as_batched_edges(structure_batch, edge_key, count_key, batch_size, device)`
  → `(edges [B, max_edges, 2] long, counts [B] long)`; validates dims,
  broadcasts `B=1` to `B`, casts to `device/long`. Used for chunk-index
  relations (`graph_call_edges`, `graph_type_edges`) and for
  `graph_generated_query_edges` (token pairs).
- `_as_batched_edge_triples(...)` → `(edges [B, max_edges, 3] long, counts [B])`
  for `(src, dst, kind)` token-position relations (domain/build/shell/
  diagnostic/cross-domain). `kind >= 0` marks an active triple in
  `_scatter_edges_(..., require_kind=True)`.
- `_as_batched_chunks(...)` → `(starts [B,C], ends [B,C], counts [B])` long;
  chunk spans are non-overlapping, ordered, and inside `[0, length]`.
- `_scatter_edges_(bias, edges, counts, *, weight, sq, sk, require_kind)` —
  vectorized scatter into a dense `[B,Sq,Sk]` bias via a single
  `index_add_` keyed by `(b*sq + src)*sk + dst`. Raises on any active edge
  outside `[0,sq)×[0,sk)` or any count outside `[0,max_edges]`. **One device
  sync per relation per layer** for the validity check.
- `_scatter_chunk_relation_edges_(...)` — builds a chunk-level adjacency
  `[B,C,C]`, then gathers through `_token_chunk_map(...)` to expand each
  chunk-pair into a token-span rectangle.
- `build_graph_route_bias_from_structure_batch(...)` — orchestrates the above
  into `S_graph [B,Sq,Sk]` (no head dim, no beta) for DSA indexer scoring.

`graph_route_attention_bias_patch.py` then:

- `build_dense_graph_attention_bias_from_structure_batch(...)` —
  `S_graph.unsqueeze(1).contiguous()` → `[B,1,Sq,Sk]`, scaled by
  `effective_beta`, with optional `graph_generated_query_edges` overlay;
  refuses to materialize beyond `CPPMEGA_GRAPH_DENSE_MAX_SEQ`.
- `build_rectangular_graph_attention_bias_from_structure_batch(...)` —
  same idea for incremental decode: only the new query rows
  `[query_start, query_start+Sq)` against the full cached key length `Sk`.
  Chunk relations are clipped to the query window in Python; token triples
  go through `_scatter_rectangular_token_edges_` which raises on any
  `(src,dst)` outside the decode bounds.
- `apply_graph_route_attention_bias_patch()` — wraps
  `TransformerLayer.forward` and stamps `bound.arguments["attention_bias"]`
  when the layer is `dense` (TE/GQA), skips `dsa`, raises on `mla`, raises on
  `sequence_parallel`/`context_parallel_size>1`.

`PromptGraphInferenceState(structure_batch, query_start, key_length)` is
attached to Megatron's `inference_context` so decode steps can rebuild the
rectangular bias from the original prompt structure batch.

The dense path also writes optional `CPPMEGA_H200_GRAPH_PRIOR_RECEIPT`
receipts via `cppmega.megatron.h200_preflight.observe_graph_prior`.

---

## 3. FA4 Surface We Will Use

From `flash_attn/cute/interface.py` (FA4 = `flash-attn-4==4.0.0b19`):

```python
flash_attn.cute.interface.flash_attn_func(
    q, k, v,
    qv=None,
    gather_kv_indices=None,
    softmax_scale=None,
    causal=False,
    window_size=(None, None),
    learnable_sink=None,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    score_mod=None,            # cute.jit callable, see signature below
    score_mod_bwd=None,        # required if score_mod is provided
    mask_mod=None,
    aux_tensors=None,          # list[torch.Tensor] threaded into the kernel
    aux_scalars=None,          # tuple of runtime scalars captured by score_mod
    block_sparse_tensors=None,         # BlockSparseTensorsTorch (fwd)
    block_sparse_tensors_bwd=None,     # BlockSparseTensorsTorch (bwd, Q-direction)
    return_lse=False,
)
```

`score_mod` signature (CuTe-DSL, see `utils.create_softcap_scoremod`):

```python
@cute.jit
def score_mod(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx,
              seqlen_info, aux_tensors):
    # acc_S_SSA is the tile of pre-mask scores (already scaled by softmax_scale
    # when score_mod is present — see utils.compute_softmax_scale_log2).
    return modified_score
```

`score_mod_bwd` signature:

```python
@cute.jit
def score_mod_bwd(grad_out_SSA, score_SSA, batch_idx, head_idx,
                  q_idx, kv_idx, seqlen_info, aux_tensors):
    return grad_in_SSA   # d(score')/d(score) chain rule
```

Constraints noted in the FA4 source:

- Custom `score_mod` is **SM90+ only**; SM8x raises `NotImplementedError`.
- `softcap` and `score_mod` are mutually exclusive.
- `score_mod_bwd` is **required** whenever `score_mod` is provided.
- `aux_tensors` are passed as a `list[torch.Tensor]`; FA4 hashes the
  callable and the aux tensor metadata into the kernel compile key, so
  re-using stable tensor *shapes/dtypes* across steps avoids recompiles
  (the underlying buffers can be rewritten in place).

`BlockSparseTensorsTorch` (from `flash_attn/cute/block_sparsity.py`):

```python
class BlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor          # [B, H, M]   int32, CUDA
    mask_block_idx: torch.Tensor          # [B, H, M, N] int32, CUDA
    full_block_cnt: torch.Tensor | None   # same shape rules
    full_block_idx: torch.Tensor | None
    cu_total_m_blocks: torch.Tensor | None      # varlen only
    cu_block_idx_offsets: torch.Tensor | None   # varlen only
    block_size: tuple[int, int] | None          # (sparse_q, sparse_kv)
    dq_write_order: torch.Tensor | None         # int32, parallel to mask_block_idx
    dq_write_order_full: torch.Tensor | None
    spt: bool | None
```

Validation rules we must respect:

- `mask_block_cnt`/`mask_block_idx` are int32 CUDA, shapes
  `(B, H, M)` / `(B, H, M, N)`. `B` and `H` may be `1` for broadcast.
- `block_size = (sparse_block_size_q, sparse_block_size_kv)`:
  - `sparse_block_size_kv` must be a multiple of the kernel `tile_n`;
  - `sparse_block_size_q` must be a multiple of `q_stage * tile_m`;
  - with `block_size=(128, 128)` and FA4's default `tile_m=tile_n=128`
    (or 64 with `q_stage=2`), both checks pass on H200/Blackwell.
- `mask_block_idx.shape[3]` may be **smaller** than `ceildiv(Sk, 128)` —
  FA4 only reads indices `0..cnt-1` per query tile (the "compact block
  sparse indices" note). This is what lets us stay sub-`O(N²)` at long
  context.
- Backward expects **Q-direction** metadata (`q_mask_cnt`/`q_mask_idx`),
  i.e. for each KV block, the list of Q blocks that touch it. We must
  build both directions on the host.
- `dq_write_order` (+ optional `dq_write_order_full`) is required for
  deterministic block-sparse backward; computed by
  `compute_dq_write_order(...)` from the fwd+bwd mask pairs. We will
  generate it whenever `deterministic=True`, and pass `None` otherwise.

`mask_mod` is also available but is intended for *static* position-based
masks (causal, sliding window). Our prior is data-dependent and additive,
so it belongs in `score_mod`, with `block_sparse_tensors` providing the
coarse skip.

---

## 4. CSR/Tiled Representation

### 4.1 Logical model

For each batch element `b` we have a set of weighted directed edges
`(q, k, w)` in *global* token coordinates. Multiple relations can produce
the same `(q, k)`; we **sum** their weights, matching today's
`index_add_`/`+=` semantics. The score modification is:

```
score'[b, h, q, k] = score[b, h, q, k] + beta * W[b, q, k]
```

where `W` is the sparse edge-weight matrix (head-broadcast). `beta` is
`resolve_graph_bias_beta()` (validated by `validate_graph_bias_beta`).

### 4.2 Per-batch CSR layout (chosen)

We pack edges **sorted by query row** into a flat tensor with row offsets,
so `score_mod` can do a bounded binary search per `(b, q)` row.

Auxiliary tensors threaded through `aux_tensors` (all on CUDA; integer
tensors are `int32` to match FA4 metadata; weights match Q dtype, bf16):

| Name                  | Shape                          | Dtype          | Notes |
|-----------------------|--------------------------------|----------------|-------|
| `csr_row_offsets`     | `[B, Sq + 1]`                  | int32          | `row_offsets[b, q:q+2]` brackets row `q`'s edges in the flat arrays. `row_offsets[b, 0] = 0`, `row_offsets[b, Sq] = nnz[b]`. |
| `csr_col_idx`         | `[B, max_nnz_per_batch]`       | int32          | Key index `k` for each edge, **sorted ascending within each row**. Padding slots after `nnz[b]` are zero. |
| `csr_weight`          | `[B, max_nnz_per_batch]`       | bf16 (Q dtype) | Pre-multiplied by `beta * relation_weight` on host. Padding zeros. |
| `csr_meta`            | `[4]`                          | int32          | `[Sq, Sk, max_nnz_per_batch, flags]` for bounds checks in kernel; `flags` bit0 = "rectangular decode" (use `q + query_start` as global row). |
| `query_start_scalar`  | passed via `aux_scalars`       | python int     | Decode-mode global offset for the query axis (0 in prefill). |

Why per-batch padded `[B, max_nnz]` instead of a single flat `[total_nnz]`
with `cu_seqlens`-style offsets:

- FA4's `to_cute_aux_tensor` wants ordinary strided tensors; a fixed `[B, N]`
  shape keeps the **compile key stable** across microbatches as long as
  `max_nnz_per_batch` is held at a configured high-water mark.
- Per-batch padding wastes at most `B * (max_nnz - nnz[b])` int32+bf16
  slots — at the high-water marks below this is single-digit MiB, far below
  the GiB-scale dense bias we are replacing.
- We still expose `nnz[b] = row_offsets[b, Sq]` so the kernel never reads
  padding.

A fully packed `[total_nnz]` + `[B+1]` batch-offset variant is a future
optimization if we ever see `max_nnz_per_batch` blow up; the score_mod
contract below is identical, only the address arithmetic changes.

### 4.3 Block-sparse mask at 128×128

Coarse skip is expressed as `BlockSparseTensorsTorch` with
`block_size=(128, 128)`:

- `M = ceildiv(Sq, 128)`, `N = ceildiv(Sk, 128)`.
- For each `(b, m)` we compute the **union** of `k`-blocks touched by any
  edge whose `q ∈ [128m, 128m+128)`, plus the diagonal block (so the
  baseline causal/local attention is never skipped), plus any blocks
  implied by `causal`/`window_size`.
- `mask_block_cnt[b, 0, m]` = number of touched blocks (head-broadcast:
  `H` dim is `1` because the prior is head-independent).
- `mask_block_idx[b, 0, m, :cnt]` = touched block ids, **ascending**.
- We do **not** populate `full_block_*`: even a "dense" 128×128 block almost
  always contains zero graph edges in cppmega, so the score_mod must run
  inside every kept block. Marking blocks "full" would let FA4 skip
  `score_mod`, which would silently drop the prior. (Future: a block can be
  marked full only if **every** `(q,k)` in it has a graph edge — never true
  for our edge densities.)
- Backward direction: for each KV block `n`, list the Q blocks `m` that
  touch it. We compute it on host from the same edge list (transpose), and
  feed `compute_dq_write_order(...)` to produce `dq_write_order` when
  `deterministic=True`.

Memory at `B=8, Sq=Sk=16384, H=1` (head-broadcast):

- `mask_block_cnt`: `8·1·128·4 B = 4 KiB`
- `mask_block_idx` (compact, assume ≤16 kept blocks/row): `8·1·128·16·4 B = 64 KiB`
  vs. the dense `8·1·128·128·4 B = 512 KiB` worst case — both trivial.

The savings are in *FLOPs and HBM traffic inside FA4*: skipped KV blocks
never load `K`/`V` and never run the MMA.

### 4.4 Host-side builder

`build_fa4_graph_route_aux(structure_batch, *, batch_size, query_start,
seqlen_q, seqlen_k, device, q_dtype, beta, weights...) -> FA4GraphRouteAux`
reuses the existing helpers (no new edge parsing):

1. Resolve `effective_beta` via `resolve_graph_bias_beta` /
   `validate_graph_bias_beta` (same as dense path).
2. For each relation, call `_as_batched_edges` / `_as_batched_edge_triples`
   / `_as_batched_chunks` exactly like
   `build_graph_route_bias_from_structure_batch`. Reuse the same
   fail-closed validation (count ranges, active-edge bounds, chunk spans).
3. Expand chunk-index relations into **token rectangles**
   `(q0, q1, k0, k1, weight)` using `graph_chunk_starts/ends`, clipped to
   the active query window `[query_start, query_start+Sq)` and key window
   `[0, Sk)` — same clipping logic as
   `build_rectangular_graph_attention_bias_from_structure_batch`.
4. Token-triple relations contribute `(q, k, weight)` directly; subtract
   `query_start` from `q` to land in local row space, drop rows outside
   `[query_start, query_start+Sq)` after the bounds check (raise, do not
   silently drop, mirroring `_scatter_rectangular_token_edges_`).
5. `graph_generated_query_edges` overlay with weight `1.0` (matches the
   dense path) before the beta multiply.
6. **Materialize rectangles into edges.** A chunk-pair rectangle of size
   `Δq × Δk` becomes either:
   - `Δq · Δk` individual `(q,k)` edges (exact, matches dense semantics), or
   - a *block-edge* `(q, k_block)` entry plus a per-row "span" flag, if we
     later want to compress. **Phase 1 uses individual edges** for
     correctness parity; the kernel contract already treats every entry as
     a single `(q,k,w)` so compression is a builder-only change.
7. Sum duplicate `(b,q,k)` weights (the dense path's `index_add_`
   semantics). Implement with a per-batch
   `torch.unique(..., return_inverse=True)` + `index_add_` on a
   `[B, max_nnz]` scratch, or with a sort-and-segmented-sum; both stay on
   device with one sync for the `nnz` upper bound.
8. Sort each row's surviving edges by `k` ascending (required for the
   in-kernel binary search). Use a segmented sort
   (`torch.sort` on `[B, max_nnz]` with padding pushed to `+inf`).
9. Build `csr_row_offsets` via per-row counts → cumulative sum.
10. Build the 128×128 `mask_block_cnt/idx` from the same edge list:
    `block_id = (q // 128, k // 128)`; segmented-unique per `q_block`;
    union with diagonal/causal blocks; sort ascending; pack into
    `[B,1,M,max_kept]`. `max_kept` is a configured high-water mark
    (default 32); the builder raises if exceeded so we never silently
    drop a block.
11. Build the backward (Q-direction) block list and `dq_write_order` via
    `flash_attn.cute.block_sparsity.compute_dq_write_order` when
    `deterministic=True`.
12. Return a frozen dataclass `FA4GraphRouteAux(csr_row_offsets, csr_col_idx,
    csr_weight, csr_meta, block_sparse_fwd, block_sparse_bwd,
    query_start, max_nnz_per_batch, nnz_per_batch)` plus a stable
    `compile_key = (B, Sq, Sk, max_nnz, max_kept, q_dtype, flags)`.

The builder is the **only** place that knows about edge semantics. The
attention module sees opaque tensors.

### 4.5 In-kernel `score_mod` (pseudocode)

```python
@cute.jit
def graph_route_score_mod(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx,
                          seqlen_info, aux_tensors):
    row_offsets, col_idx, weight, meta = aux_tensors  # cute tensors
    Sq, Sk, max_nnz, flags = meta[0], meta[1], meta[2], meta[3]
    q_global = q_idx + aux_scalars[0]      # query_start (0 in prefill)

    # Per-thread tile: acc_S_SSA has shape [tile_m, tile_n] in registers.
    # Each lane handles one (q, k) pair.
    if q_global >= Sq:                     # padding q rows: identity
        return acc_S_SSA

    lo = row_offsets[batch_idx, q_global]
    hi = row_offsets[batch_idx, q_global + 1]

    # Binary search the sorted column list for kv_idx.
    # Edge counts per row are tiny (<<32), so this is a bounded loop the
    # CuTe DSL compiler unrolls into a register scan.
    found_w = 0.0
    left, right = lo, hi
    while left < right:
        mid = (left + right) >> 1
        c = col_idx[batch_idx, mid]
        if c < kv_idx:
            left = mid + 1
        else:
            right = mid
    if left < hi and col_idx[batch_idx, left] == kv_idx:
        found_w = weight[batch_idx, left].to(acc_S_SSA.dtype)

    return acc_S_SSA + found_w   # beta already folded into weight on host
```

Backward (required by FA4):

```python
@cute.jit
def graph_route_score_mod_bwd(grad_out_SSA, score_SSA, batch_idx, head_idx,
                              q_idx, kv_idx, seqlen_info, aux_tensors):
    # score' = score + bias  =>  d(score')/d(score) = 1, bias is constant.
    return grad_out_SSA
```

Notes:

- `acc_S_SSA` arrives **already scaled** by `softmax_scale` (FA4 keeps the
  scale separate when `score_mod` is present, see
  `compute_softmax_scale_log2`). Our bias must therefore be in
  *post-scale* units; the host builder multiplies weights by
  `softmax_scale` along with `beta`. The dense TE path applies bias
  *post-scale* too, so this preserves numerics.
- `head_idx` is unused (head-broadcast prior) but present in the signature.
- `mask_mod` is left `None`; coarse skipping is fully handled by
  `block_sparse_tensors`.
- The binary search reads only `csr_col_idx[batch_idx, lo:hi]`, which is
  tiny; we expect the compiler to keep `lo`/`hi`/the comparison in
  registers. If profiling shows this is hot, we can switch to a per-row
  small hash or a per-`kv_block` mini-CSR; the aux contract is unchanged.

---

## 5. `CppMegaFA4DotProductAttention` Module

### 5.1 Class shape

A Megatron-`ModuleSpec`-compatible `MegatronModule` that replaces
`TEDotProductAttention` as the `core_attention` sub-module of
`SelfAttention`. QKV projections, FP8 GEMMs, RoPE, and the output projection
**stay in TE**; only the dot-product/softmax kernel changes.

```python
class CppMegaFA4DotProductAttention(MegatronModule):
    def __init__(self, config, layer_number,
                 attention_type, num_attention_heads,
                 attention_dropout=0.0,
                 softmax_scale=None,           # default 1/sqrt(head_dim)
                 causal=False,
                 window_size=(None, None),
                 deterministic=False,
                 aux_high_water_marks=None,    # see §5.4
                 **_ignored_te_kwargs):
        ...
        # Stash config; no TE backend instantiation.

    def forward(self, query, key, value,
                attention_mask=None,
                attn_mask_type=None,
                attention_bias=None,           # must be None or graph-route aux
                packed_seq_params=None,
                inference_context=None,
                **kwargs):
        ...
```

The constructor signature mirrors `TEDotProductAttention` closely enough
that existing `ModuleSpec(... submodules=TransformerLayerSubmodules(
self_attention=..., core_attention=ModuleSpec(
module=CppMegaFA4DotProductAttention, params={...})))` wiring works.
Unsupported TE kwargs (e.g. `qk_layernorm` inside core attention) raise
fail-closed.

### 5.2 Forward flow

1. **Inputs.** Megatron hands us `query/key/value` already projected by TE
   (FP8 GEMMs upstream are untouched). Layout is `[B, S, H, D]` (or
   `[S, B, H, D]` if `sequence_parallel` — we raise on that, mirroring the
   dense patch). `attention_bias` is one of:
   - `None` → run FA4 with no `score_mod` (plain attention).
   - `FA4GraphRouteAux` (our dataclass) → run FA4 with `score_mod` +
     `block_sparse_tensors`.
   - A raw `torch.Tensor` → raise: the FA4 backend refuses dense bias by
     contract (the whole point is to not materialize it). The
     `apply_graph_route_attention_bias_patch` wrapper is updated to build
     `FA4GraphRouteAux` instead of `[B,1,Sq,Sk]` when the layer's
     `core_attention` is `CppMegaFA4DotProductAttention`.
2. **Geometry checks.** `B = query.shape[0]`, `Sq = query.shape[1]`,
   `Sk = key.shape[1]`, `H = query.shape[2]`, `D = query.shape[3]`. KV
   heads `Hk = key.shape[2]` must divide `H` (GQA). `head_dim` must be in
   FA4's supported set.
3. **Decode path.** If `inference_context` is not None, fetch
   `PromptGraphInferenceState` (existing dataclass) and rebuild the
   rectangular aux via the same builder, with
   `query_start = state.query_start`, `Sk = state.key_length`. The
   `sequence_len_offset` cross-check from the dense patch carries over
   verbatim.
4. **Call FA4.**

   ```python
   out = flash_attn.cute.interface.flash_attn_func(
       q=query, k=key, v=value,
       softmax_scale=self.softmax_scale,
       causal=self.causal,
       window_size=self.window_size,
       deterministic=self.deterministic,
       score_mod=graph_route_score_mod if aux else None,
       score_mod_bwd=graph_route_score_mod_bwd if aux else None,
       aux_tensors=[aux.csr_row_offsets, aux.csr_col_idx,
                    aux.csr_weight, aux.csr_meta] if aux else None,
       aux_scalars=(aux.query_start,) if aux else None,
       block_sparse_tensors=aux.block_sparse_fwd if aux else None,
       block_sparse_tensors_bwd=aux.block_sparse_bwd if aux else None,
       return_lse=False,
   )
   ```

5. **Output.** Return `out` in the layout Megatron expects (`[B, S, H, D]`,
   then re-shaped by the caller). No dropout in core attention; FA4 does
   not implement attention dropout in the score_mod path, and cppmega's
   production runs use `attention_dropout=0`. Constructor raises if a
   non-zero dropout is requested.

### 5.3 Backward

`flash_attn_func` is a `torch.autograd.Function` (`FlashAttnFunc.apply`).
With `score_mod` set, FA4 also requires `score_mod_bwd`; ours is the
identity (`d(score + bias)/d(score) = 1`) because the graph bias is
**non-learnable**: no gradient flows to `csr_weight` (it is built fresh
each step from compiler edges, not a `nn.Parameter`). `beta` is a python
float resolved from env; no autograd edge.

`d(out)/d(q)`, `d(out)/d(k)`, `d(out)/d(v)` are computed by FA4's fused
backward, with `block_sparse_tensors_bwd` providing the Q-direction skip
list. `dq_write_order` is supplied for `deterministic=True` runs.

### 5.4 Compile-key stability & caching

FA4 recompiles when the `score_mod` hash, `aux_tensors` metadata, or
`block_sparse_tensors` shapes change. To avoid per-step recompiles:

- The builder takes `aux_high_water_marks = {"max_nnz_per_batch": int,
  "max_kept_blocks": int}` and **always pads to the high-water mark**.
  Defaults are sized from the dataset inventory
  (`corpus_inventory.json`) plus a 2× safety margin; the builder raises
  if a real batch overflows, prompting a config bump (no silent
  recompiles, no silent drops).
- `csr_meta` is a fixed `[4]` int32 tensor; per-step `Sq/Sk/max_nnz` go in
  its slots, but the **shape** never changes.
- `query_start` is passed via `aux_scalars` (a runtime scalar capture),
  not via tensor shape, so decode steps with different offsets reuse the
  same kernel.
- The `score_mod` callable is a module-level `@cute.jit` function; its
  hash is stable across processes.

### 5.5 Fail-closed contract (parity with dense patch)

- `graph_routes_active()` and `require_graph_routes_for_production()` gate
  the path identically.
- `attention_layer_route_kind` returns `"dense"` for layers using
  `CppMegaFA4DotProductAttention` (it has `core_attention`-equivalent
  semantics); the patch wrapper detects FA4 vs TE by `isinstance` and
  builds the right aux.
- MLA still raises. DSA still owns its own path (no double-apply).
- `sequence_parallel=True` and `context_parallel_size>1` raise.
- A declared edge outside `[query_start, query_start+Sq) × [0, Sk)` raises
  (mirrors `_scatter_rectangular_token_edges_`).
- `max_nnz` / `max_kept_blocks` overflow raises with a config-bump hint.

---

## 6. Integration Points

### 6.1 `apply_graph_route_attention_bias_patch`

Extend `_graph_attention_bias_for_layer` to dispatch on the layer's
`core_attention` type:

- `TEDotProductAttention` → existing dense `[B,1,Sq,Sk]` path (kept for
  cuDNN fallback / ablation).
- `CppMegaFA4DotProductAttention` → call
  `build_fa4_graph_route_aux(...)` and return the `FA4GraphRouteAux`
  dataclass; the wrapper still assigns it to
  `bound.arguments["attention_bias"]`, and the FA4 module's `forward`
  recognizes the type.
- Anything else → existing behavior.

The pinned `TransformerLayer.forward` signature is unchanged; no Megatron
core edits.

### 6.2 `ModuleSpec` wiring

Add `cppmega/megatron/fa4_attention_spec.py` exposing
`get_fa4_dot_product_attention_spec(config)` returning a `ModuleSpec` for
`CppMegaFA4DotProductAttention` with the params block (softmax_scale,
causal, window_size, deterministic, high-water marks). Existing builders
(`gpt_builder.py`, `nam56r_te_spec.py`) opt in by swapping the
`core_attention` spec; no other layer changes.

### 6.3 Run-profile flag

Add `CPPMEGA_ATTENTION_BACKEND` ∈ `{te, fa4}` (default `te` until
validated). When `fa4`:

- `gpt_builder` selects the FA4 spec;
- `apply_graph_route_attention_bias_patch` builds aux instead of dense;
- `CPPMEGA_GRAPH_DENSE_MAX_SEQ` no longer applies (the FA4 path has its
  own `max_nnz` / `max_kept_blocks` guards).

A second env `CPPMEGA_FA4_FORCE_BLOCK_SPARSE=1` (default on) lets us
ablate the block-sparse skip vs. `score_mod`-only.

### 6.4 Receipts & observability

Reuse `observe_graph_prior` with `consumer="fa4_score_mod"`:

- The builder writes a receipt with `nnz_per_batch`, `max_nnz`,
  `kept_blocks_per_row` histogram, `effective_beta`, and the compile key.
- The attention module emits a one-line `[cppmega] FA4 score_mod ...`
  banner on first use (mirrors the existing patch banner).

### 6.5 Tests (planned, not in this commit)

- **Equivalence:** for `B ∈ {1,2,8}`, `Sq=Sk ∈ {128, 512, 2048}`,
  random sparse edges, compare FA4 output to the dense TE path with the
  same `beta * S_graph` bias. Tolerance: bf16 `atol=2e-2, rtol=2e-2`
  (FA4 vs cuDNN baseline), plus a stricter fp32-reference check.
- **Backward:** gradient equivalence on `q,k,v` with the same tolerances;
  assert no gradient is requested for aux tensors.
- **Decode:** rectangular `Sq=1, Sk ∈ {512, 4096}` against
  `build_rectangular_graph_attention_bias_from_structure_batch`.
- **Fail-closed:** corrupt counts, out-of-range edges, `max_nnz`
  overflow, MLA layer, `sequence_parallel=True`, dense-tensor
  `attention_bias` into the FA4 module — all raise.
- **Compile-key stability:** run two consecutive steps with different
  edge counts but identical high-water marks; assert FA4's compile cache
  hit count does not change (via `flash_attn.cute` compile counters).

---

## 7. Memory & Performance Analysis

### 7.1 Memory (per microbatch, bf16 weights, int32 indices)

Dense path today, `B=8, Sq=Sk=16384`:

- bias `[B,1,Sq,Sk]` bf16 = `8 · 16384² · 2 B` = **4 GiB** (the cap's
  exact motivating example).

FA4 path, same shape, with `max_nnz_per_batch=4096`,
`max_kept_blocks=32`:

- `csr_row_offsets`: `8 · 16385 · 4 B ≈ 0.5 MiB`
- `csr_col_idx`: `8 · 4096 · 4 B = 128 KiB`
- `csr_weight` bf16: `8 · 4096 · 2 B = 64 KiB`
- `mask_block_cnt`: `8 · 1 · 128 · 4 B = 4 KiB`
- `mask_block_idx`: `8 · 1 · 128 · 32 · 4 B = 128 KiB`
- backward block lists + `dq_write_order`: same order, ~256 KiB.

**Total ≈ 1.1 MiB**, vs. 4 GiB dense — a ~3700× reduction at the cap.
Even with `max_nnz_per_batch=65536` (16× headroom) we stay under 16 MiB.

### 7.2 Kernel-side cost

- **Skipped KV blocks:** with edge density `ρ` (fraction of 128×128
  blocks touched), FA4 loads `K/V` and runs MMAs only on `~ρ·N` blocks
  per query tile. cppmega edge densities are typically `ρ < 5%`, so the
  forward MMA work approaches the block-sparse floor.
- **`score_mod` overhead:** one bounded binary search per `(q,k)` pair in
  kept blocks. With per-row `nnz` in the single digits this is a small
  constant; the CuTe DSL compiler keeps the search in registers.
- **Backward:** identical skip ratio in the Q-direction; `dq_write_order`
  preserves determinism without an extra kernel.

### 7.3 Host-side cost

The builder runs once per microbatch on the existing structure batch. It
reuses `_as_batched_*` (already vectorized, single sync per relation) and
adds:

- one segmented sort + segmented unique per batch (`O(nnz log nnz)` on
  device, `nnz` ≪ `Sq·Sk`);
- one block-id sort per batch (`O(M · max_kept log max_kept)`).

We expect <100 µs at `B=8, Sq=16384` — well under the per-step kernel
time, and amortizable across the layer count by caching the aux per
`(structure_batch_id, query_start, Sq, Sk)` (the structure batch is
identical for every layer in a step). The module-level cache lives on the
inference context / structure-batch token, with a `weakref`-keyed dict so
nothing leaks across steps.

### 7.4 Numerical parity

- Bias is added **post-scale** in both paths (TE applies `attention_bias`
  after `softmax_scale`; FA4 with `score_mod` keeps `softmax_scale`
  separate and applies it before `score_mod`, so we pre-multiply weights
  by `softmax_scale` on host).
- Duplicate-edge accumulation matches `index_add_` (sum).
- `beta` resolution and validation are unchanged.
- bf16 accumulation in `csr_weight` matches the dense bf16 bias path;
  fp32 builder option is available via the existing `dtype` argument if a
  recipe needs it.

---

## 8. Risks & Open Questions

1. **FA4 SM target.** Custom `score_mod` raises on SM8x. cppmega's H200
   (SM90) and Blackwell (SM100/SM120) targets are fine; A100 runs must
   stay on the dense TE path. The backend selector enforces this.
2. **`tile_m`/`q_stage` variability.** `block_size=(128,128)` is valid for
   FA4's default Hopper/Blackwell tile choices, but a future FA4 bump
   could change `q_stage`. The builder reads the kernel's expected shapes
   through `infer_block_sparse_expected_shapes` at construction time and
   raises if `(128,128)` stops being a multiple of `q_stage*tile_m`;
   mitigation is to bump `block_size` to `(256,128)` or `(128,256)`.
3. **Rectangular chunk relations.** Expanding a chunk-pair into individual
   edges can blow past `max_nnz` for very large chunks (e.g. a
   whole-file → whole-file call edge). Phase 1 raises with a config-bump
   hint; Phase 2 may compress to per-row spans (builder-only change).
4. **`score_mod` register pressure.** A binary search inside the inner
   loop adds registers; if FA4 occupancy drops measurably, switch to a
   per-`kv_block` mini-CSR (one `weight` per `(q, kv_block)`, scanned in
   `O(kept_blocks)`). The aux contract is unchanged.
5. **TE FP8 descale interaction.** FA4 supports `q_descale/k_descale/
   v_descale` for FP8 inputs. cppmega currently does FP8 GEMMs in TE and
   hands bf16 QKV to core attention, so this is out of scope; if we later
   pipe FP8 QKV directly into FA4, the descales must be threaded through
   `CppMegaFA4DotProductAttention.forward`.
6. **Dropout.** FA4 `score_mod` path does not implement attention dropout.
   We raise on non-zero `attention_dropout`; cppmega production already
   runs with `0`.

---

## 9. Deliverables (for the implementation phase)

1. `cppmega/megatron/fa4_graph_route_aux.py` — `FA4GraphRouteAux`
   dataclass + `build_fa4_graph_route_aux(...)` + score_mod callables.
2. `cppmega/megatron/fa4_dot_product_attention.py` —
   `CppMegaFA4DotProductAttention` Megatron module.
3. `cppmega/megatron/fa4_attention_spec.py` — `ModuleSpec` helper.
4. Edits to `graph_route_attention_bias_patch.py` to dispatch on
   `core_attention` type and build aux for FA4 layers; dense path
   preserved.
5. `tests/megatron/test_fa4_score_mod_graph_routes.py` — equivalence,
   backward, decode, fail-closed, compile-key stability.
6. Run-profile flag `CPPMEGA_ATTENTION_BACKEND` plumbed through
   `gpt_builder.py` and the relevant run scripts.

---

## 10. References (local)

- `cppmega/megatron/graph_route_attention_bias_patch.py` — dense bias
  builder, `TransformerLayer.forward` patch, `PromptGraphInferenceState`.
- `cppmega/megatron/dsa_indexer_fused_patch.py` — `_as_batched_edges`,
  `_as_batched_edge_triples`, `_as_batched_chunks`, `_scatter_edges_`,
  `_scatter_chunk_relation_edges_`,
  `build_graph_route_bias_from_structure_batch`.
- `cppmega/megatron/graph_objective_loss.py` — `graph_routes_active`,
  `resolve_graph_bias_beta`, `validate_graph_bias_beta`.
- `cppmega/megatron/h200_preflight.py` — `observe_graph_prior` receipts.
- `STACK.lock`, `docker/Dockerfile` — `flash-attn-4[cu13]==4.0.0b19`
  pin and `flash_attn.cute` import smoke test.
- FA4 source (read-only reference, not vendored):
  `flash_attn/cute/interface.py::flash_attn_func`,
  `flash_attn/cute/block_sparsity.py::BlockSparseTensorsTorch`,
  `flash_attn/cute/utils.py::create_softcap_scoremod`.
