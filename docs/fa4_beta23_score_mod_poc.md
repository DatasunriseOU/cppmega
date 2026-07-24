# FA4 beta23 `score_mod` POC — Chunk-Native Graph-Route Attention Bias

Status: research/design only (no implementation in this commit)
Owners: cppmega attention/runtime
Targets: H200 (SM90), Blackwell (SM100/SM120)
FA4 pin in image: `flash-attn-4[cu13]==4.0.0b19` (per `STACK.lock`,
`docker/Dockerfile`); this POC targets the **beta23** `score_mod` API surface
so the design must be re-validated against the actual `4.0.0b23` wheel before
implementation. Companion doc: `docs/fa4_score_mod_design.md` (CSR-based
design for the b19 surface). This POC deliberately chooses a different,
simpler representation that mirrors how cppmega's compiler routes are already
structured.

---

## 1. Problem statement (recap)

`cppmega/megatron/graph_route_attention_bias_patch.py` injects the
compiler-derived graph prior into dense TE/GQA attention by materializing a
**dense** `[B, 1, Sq, Sk]` bias and forwarding it through
`TransformerLayer.forward(..., attention_bias=...)` →
`TEDotProductAttention` → TE `FusedAttention` (cuDNN).

Two failure modes motivate this POC:

1. **Memory blow-up.** `[B,1,Sq,Sk]` is `O(B·Sq·Sk)` even though the prior is
   *chunk-structured*: a handful of call/type chunk-pairs expanded into
   token×token rectangles. The dense path already carries a fail-loud cap
   `CPPMEGA_GRAPH_DENSE_MAX_SEQ=16384` precisely because bf16 dense bias is
   ~4 GiB at `B=8, S=16384`. Long-context cppmega cannot use this path.
2. **Kernel regression.** A non-`None` `attention_bias` forces TE off the
   FA3/FA4 fast path and onto cuDNN. We lose FA4's pipelined Hopper/Blackwell
   kernels and the CuTe-DSL extension surface.

The graph prior is naturally **chunk-level**, not token-level:

- `graph_call_edges` / `graph_type_edges` are pairs of **chunk indices**
  (see `_as_batched_edges` + `_scatter_chunk_relation_edges_` in
  `cppmega/megatron/dsa_indexer_fused_patch.py`).
- `graph_chunk_starts/ends/counts` give a non-overlapping ordered chunk
  layout per sample.
- The remaining relations (domain/build/shell/diagnostic/cross-domain) and
  `graph_generated_query_edges` are token-position pairs/triples — sparse
  point overlays on top of the chunk rectangles.

The dense builder *expands* the chunk pairs into token rectangles and then
hands a `[B,1,Sq,Sk]` tensor to TE. This POC inverts that: keep the prior in
chunk space, hand FA4 a tiny `[B, C+1, C+1]` bias plus a `[B, S]` token→chunk
map, and let `score_mod` do the gather inside the kernel.

---

## 2. Current edge format (recap from source)

From `cppmega/megatron/dsa_indexer_fused_patch.py`:

- `_as_batched_edges(structure_batch, edge_key, count_key, batch_size, device)`
  → `(edges [B, max_edges, 2] long, counts [B] long)`. Used for chunk-index
  relations (`graph_call_edges`, `graph_type_edges`) and for token-pair
  relations (`graph_generated_query_edges`).
- `_as_batched_edge_triples(...)` → `(edges [B, max_edges, 3] long, counts [B])`
  for `(src, dst, kind)` token-position relations. `kind >= 0` marks an
  active triple in `_scatter_edges_(..., require_kind=True)`.
- `_as_batched_chunks(...)` → `(starts [B,C], ends [B,C], counts [B])` long;
  chunk spans are non-overlapping, ordered, inside `[0, length]`. Validated
  by `_token_chunk_map`.
- `_token_chunk_map(starts, ends, counts, length=S)` → `(chunk_ids [B,S],
  valid [B,S])` via `searchsorted` over chunk starts. `chunk_ids` is clamped
  to `[0, C-1]`; `valid` is false for tokens that fall in inter-chunk gaps.
- `_scatter_chunk_relation_edges_(...)` (line 539) is the key precedent. It:
  1. builds a chunk-level adjacency `[B, C, C]` and `index_add_`s `weight`
     at each declared `(src_chunk, dst_chunk)` pair — duplicates accumulate;
  2. gathers rows by `q_chunks` then columns by `k_chunks` to expand each
     chunk pair into a token-span rectangle;
  3. zeros rectangles where either endpoint token is `~valid` (inter-chunk
     gap);
  4. adds the result into the dense `[B, Sq, Sk]` bias.
- `build_graph_route_bias_from_structure_batch(...)` orchestrates the above
  into `S_graph [B, Sq, Sk]` (no head dim, no beta) for DSA indexer scoring.

From `cppmega/megatron/graph_route_attention_bias_patch.py:172-267`
(`build_dense_graph_attention_bias_from_structure_batch`):

- Calls `build_graph_route_bias_from_structure_batch` with per-relation
  weights `call/type/domain/build/shell/diagnostic/cross_domain` (each
  defaulting to `1.0`, env-overridable via `CPPMEGA_GRAPH_ATTENTION_*_WEIGHT`).
- Overlays `graph_generated_query_edges` with weight `1.0` via
  `_scatter_edges_(..., require_kind=False)` — these are **token-position
  pairs**, not chunk pairs, and are the only relation that does not align to
  chunk boundaries in the dense path.
- Multiplies by `effective_beta = resolve_graph_bias_beta()` (validated by
  `validate_graph_bias_beta`; same scalar DSA uses).
- Returns `S_graph.unsqueeze(1).contiguous()` → `[B, 1, Sq, Sk]`, with the
  `CPPMEGA_GRAPH_DENSE_MAX_SEQ` cap and an optional
  `observe_graph_prior(consumer="dense_attention", ...)` receipt.

Edge kinds and weights (current dense path):

| Relation         | Sidecar key                  | Encoding              | Default weight |
|------------------|------------------------------|-----------------------|----------------|
| call             | `graph_call_edges`           | chunk-index pair      | 1.0            |
| type             | `graph_type_edges`           | chunk-index pair      | 1.0            |
| domain           | `graph_domain_edges`         | token triple          | 1.0            |
| build            | `graph_build_edges`          | token triple          | 1.0            |
| shell            | `graph_shell_edges`          | token triple          | 1.0            |
| diagnostic       | `graph_diagnostic_edges`     | token triple          | 1.0            |
| cross_domain     | `graph_cross_domain_edges`   | token triple          | 1.0            |
| generated_query  | `graph_generated_query_edges`| token pair (decode)   | 1.0            |

All weights are then multiplied by `beta`. Duplicates accumulate (sum),
matching `index_add_` semantics.

---

## 3. FA4 beta23 surface used by this POC

Target call (beta23 API; verify against the actual wheel before coding):

```python
flash_attn.cute.interface.flash_attn_func(
    q, k, v,
    softmax_scale=...,
    causal=True,                    # cppmega is causal-only here
    score_mod=graph_chunk_score_mod,
    score_mod_bwd=graph_chunk_score_mod_bwd,
    aux_tensors=[token_to_chunk_q, token_to_chunk_k, chunk_bias,
                 rare_q, rare_k, rare_w, rare_meta],
    aux_scalars=(beta_scaled_marker,),   # optional, see §4.4
    block_sparse_tensors=None,      # POC: no block-sparse skip
    mask_mod=None,                  # POC: no static mask
    return_lse=False,
)
```

`score_mod` signature (per the beta23 contract used by this POC):

```python
@cute.jit
def score_mod(score, batch, head, query_idx, key_idx,
              seqlen_info, aux_tensors):
    return modified_score
```

Constraints carried over from the b19 surface and assumed unchanged in
beta23 (must be re-verified):

- Custom `score_mod` is **SM90+ only**; SM8x raises `NotImplementedError`.
- `softcap` and `score_mod` are mutually exclusive.
- `score_mod_bwd` is **required** whenever `score_mod` is provided.
- `aux_tensors` is a `list[torch.Tensor]`; FA4 hashes the callable plus aux
  tensor *metadata* (shape/dtype/device) into the kernel compile key. Stable
  shapes across steps avoid recompiles even when the underlying buffers are
  rewritten in place.
- `score` arrives **already scaled** by `softmax_scale` (FA4 keeps the scale
  separate when `score_mod` is present, see
  `flash_attn/cute/utils.py::compute_softmax_scale_log2`). Bias must be in
  post-scale units; the host builder pre-multiplies weights by
  `softmax_scale` along with `beta`. The dense TE path applies
  `attention_bias` post-scale too, so this preserves numerics.

Why `block_sparse_tensors=None` and `mask_mod=None` for the POC:

- The chunk prior is *additive*, not masking. `mask_mod` is intended for
  static position-based masks (causal, sliding window); ours is
  data-dependent and additive, so it belongs in `score_mod`.
- A coarse 128×128 block-sparse skip is a Phase-2 optimization (already
  designed in `docs/fa4_score_mod_design.md` §4.3). The POC keeps
  `block_sparse_tensors=None` so the only moving part is the chunk-native
  `score_mod`; FA4 then runs the full causal tile schedule and applies
  `score_mod` on every kept tile.
- `causal=True` gives us the lower-triangular skip for free; cppmega is
  causal-only on this path.

---

## 4. Chunk-native representation

### 4.1 Logical model

For each batch element `b`:

- Tokens are partitioned into `C_b` non-overlapping ordered chunks via
  `graph_chunk_starts/ends`. Tokens that fall in inter-chunk gaps map to a
  sentinel "no chunk" id `C` (the `+1` slot).
- A chunk-pair bias matrix `A_b ∈ R^{(C+1)×(C+1)}` carries the summed weight
  for each `(src_chunk, dst_chunk)` pair across all chunk-index relations
  (call + type). The sentinel row/column is identically zero so gap tokens
  contribute no bias.
- A small overlay list of *rare token edges* `(q, k, w)` carries everything
  that does **not** align to chunk boundaries: token-triple relations
  (domain/build/shell/diagnostic/cross-domain) and `generated_query_edges`.

The score modification is:

```
score'[b, h, q, k] = score[b, h, q, k]
                   + chunk_bias[b, token_to_chunk[b, q], token_to_chunk[b, k]]
                   + rare_token_edge(b, q, k)
```

`head` is unused (head-broadcast prior) but present in the signature.
`beta` and `softmax_scale` are folded into the host-side weights so the
kernel is a pure additive lookup.

### 4.2 Tensors threaded through `aux_tensors`

All on CUDA. Integer tensors are `int32` (FA4 metadata convention); weights
match Q dtype (bf16) so the add is type-compatible with `score`.

| Name                | Shape                              | Dtype          | Notes |
|---------------------|------------------------------------|----------------|-------|
| `token_to_chunk_q`  | `[B, Sq]`                          | int32          | `chunk_id ∈ [0, C]`; `C` = sentinel "no chunk" (inter-chunk gap). Built from `graph_chunk_starts/ends/counts` via the same `searchsorted` logic as `_token_chunk_map`. |
| `token_to_chunk_k`  | `[B, Sk]`                          | int32          | Same map for keys. Identical to `_q` in prefill (`Sq == Sk`); rectangular in decode. |
| `chunk_bias`        | `[B, C+1, C+1]`                    | bf16 (Q dtype) | Pre-multiplied by `beta * softmax_scale`. Slot `[C, *]` and `[*, C]` are zero. Duplicates already summed on host. |
| `rare_q`            | `[B, max_rare]`                    | int32          | Query index (local, i.e. `global_q - query_start`) for each point edge. Padding pushed to a sentinel row `Sq` (out-of-range so the kernel match fails). Sorted ascending per batch for binary search. |
| `rare_k`            | `[B, max_rare]`                    | int32          | Key index. Sorted by `(rare_q, rare_k)` ascending. |
| `rare_w`            | `[B, max_rare]`                    | bf16           | Pre-multiplied by `beta * relation_weight * softmax_scale`. Padding zeros. |
| `rare_row_offsets`  | `[B, Sq + 1]`                      | int32          | CSR row offsets over `rare_q` so the kernel does a bounded scan per query row. `rare_row_offsets[b, q:q+2]` brackets row `q`'s point edges. |
| `rare_meta`         | `[4]`                              | int32          | `[Sq, Sk, max_rare, flags]`; `flags` bit0 = "rectangular decode" (use `q + query_start` semantics for any host-side checks; kernel reads `q_idx` directly because aux tensors are already in local-row space). |

`query_start` (decode-mode global offset) is **not** needed inside the
kernel for this representation: the host builder writes `token_to_chunk_q`
and `rare_q` already in *local* row space `[0, Sq)`. This keeps the kernel
arithmetic trivial and the compile key independent of the decode offset.

### 4.3 Why chunk-native (vs the CSR design in `fa4_score_mod_design.md`)

The CSR design flattens every chunk-pair rectangle into `Δq · Δk`
individual `(q, k, w)` edges. That is correct but throws away structure:

- A whole-file → whole-file call edge at `Δq = Δk = 256` becomes 65 536 CSR
  entries; the chunk-native form is **one** `(src_chunk, dst_chunk)` slot.
- CSR `max_nnz_per_batch` is a fragile high-water mark; chunk-native
  `C ≤ 256` is bounded by the chunker contract and the per-batch
  `(C+1)² ≤ 66 049` slots are a fixed compile-time shape.
- The kernel inner loop is two gathers + one add, no binary search over
  per-row CSR. The rare-edge overlay still uses a tiny per-row CSR because
  point edges are genuinely sparse and unstructured.

The CSR design remains the right answer if/when we add 128×128
`block_sparse_tensors` skipping (it carries the per-edge geometry needed to
compute kept blocks). The POC deliberately scopes that out.

### 4.4 Host-side builder

`build_fa4_chunk_native_aux(structure_batch, *, batch_size, query_start,
seqlen_q, seqlen_k, device, q_dtype, beta, softmax_scale, weights...,
max_chunks, max_rare_per_batch) -> FA4ChunkNativeAux` reuses the existing
helpers (no new edge parsing):

1. Resolve `effective_beta` via `resolve_graph_bias_beta` /
   `validate_graph_bias_beta` (same as dense path). Compute
   `weight_multiplier = effective_beta * softmax_scale`.
2. `_as_batched_chunks(structure_batch, batch_size, device)` →
   `(starts, ends, chunk_counts)`. Validate as today (count ranges, span
   ordering, span bounds). Let `C = max_chunks` be the configured high-water
   mark; raise if any `chunk_counts[b] > C`.
3. Build `token_to_chunk_q [B, Sq]` and `token_to_chunk_k [B, Sk]` with the
   same `searchsorted` logic as `_token_chunk_map`, but write the sentinel
   `C` (not a clamped id) where `valid` is false. For decode, the query map
   is built over the global window
   `[query_start, query_start + Sq)` and then re-based to local rows
   `[0, Sq)`.
4. Allocate `chunk_bias = zeros([B, C+1, C+1], dtype=q_dtype)`. For each
   chunk-index relation (`graph_call_edges`, `graph_type_edges`):
   - `_as_batched_edges(...)` → `(edges, counts)`;
   - validate active endpoints `< chunk_counts[b]` (mirrors
     `_scatter_chunk_relation_edges_`);
   - `index_add_` `relation_weight * weight_multiplier` at
     `(b, src_chunk, dst_chunk)` — duplicates sum, matching dense semantics.
   - Sentinel row/column `[C, *]`/`[*, C]` is never written.
5. Build the rare-edge overlay:
   - For each token-triple relation
     (domain/build/shell/diagnostic/cross-domain), `_as_batched_edge_triples`
     → `(edges, counts)`; for each active triple (`kind >= 0`), validate
     `(src, dst)` against the **global** sequence bounds, then require
     `query_start <= src < query_start + Sq` (raise on violation, mirroring
     `_scatter_rectangular_token_edges_`); emit
     `(src - query_start, dst, relation_weight * weight_multiplier)`.
   - For `graph_generated_query_edges` (token pairs, weight `1.0`): same
     bounds/raise behavior; emit `(src - query_start, dst, 1.0 *
     weight_multiplier)`.
   - Sum duplicate `(b, q_local, k)` weights via `torch.unique(...,
     return_inverse=True)` + `index_add_` (matches the dense path's
     `index_add_` semantics).
   - Sort by `(q_local, k)` ascending per batch; build `rare_row_offsets`
     via per-row counts → cumulative sum; pad to `max_rare_per_batch` with
     sentinel `q = Sq` and zero weights. Raise on overflow with a
     config-bump hint (no silent drops, no silent recompiles).
6. Return a frozen dataclass:

   ```python
   @dataclass(frozen=True)
   class FA4ChunkNativeAux:
       token_to_chunk_q: torch.Tensor   # [B, Sq] int32
       token_to_chunk_k: torch.Tensor   # [B, Sk] int32
       chunk_bias: torch.Tensor         # [B, C+1, C+1] q_dtype
       rare_q: torch.Tensor             # [B, max_rare] int32
       rare_k: torch.Tensor             # [B, max_rare] int32
       rare_w: torch.Tensor             # [B, max_rare] q_dtype
       rare_row_offsets: torch.Tensor   # [B, Sq+1] int32
       rare_meta: torch.Tensor          # [4] int32
       batch_size: int
       seqlen_q: int
       seqlen_k: int
       max_chunks: int                  # C
       max_rare_per_batch: int
       query_start: int
       compile_key: tuple               # (B, Sq, Sk, C, max_rare, q_dtype, flags)
   ```

The builder is the **only** place that knows about edge semantics. The
attention module sees opaque tensors.

### 4.5 In-kernel `score_mod` (pseudocode)

```python
@cute.jit
def graph_chunk_score_mod(score, batch, head, query_idx, key_idx,
                          seqlen_info, aux_tensors):
    (token_to_chunk_q, token_to_chunk_k, chunk_bias,
     rare_q, rare_k, rare_w, rare_row_offsets, rare_meta) = aux_tensors

    # --- Chunk-pair gather: two int loads + one bf16 load + one add ---
    qc = token_to_chunk_q[batch, query_idx]   # int32, in [0, C]
    kc = token_to_chunk_k[batch, key_idx]     # int32, in [0, C]
    bias = chunk_bias[batch, qc, kc]          # bf16; sentinel slot is 0
    out = score + bias.to(score.dtype)

    # --- Rare token-edge overlay: bounded scan over the query row ---
    # Per-row nnz is tiny (<<16); the CuTe DSL compiler unrolls the scan
    # into a register loop. rare_q is sorted ascending within each row, so
    # we can early-exit on rare_k >= key_idx.
    lo = rare_row_offsets[batch, query_idx]
    hi = rare_row_offsets[batch, query_idx + 1]
    i = lo
    while i < hi:
        k_i = rare_k[batch, i]
        if k_i == key_idx:
            out = out + rare_w[batch, i].to(score.dtype)
            break                # at most one entry per (q, k) after dedup
        if k_i > key_idx:
            break                # sorted; no further match possible
        i = i + 1

    return out
```

Backward (required by FA4):

```python
@cute.jit
def graph_chunk_score_mod_bwd(grad_out, score, batch, head,
                              query_idx, key_idx, seqlen_info, aux_tensors):
    # score' = score + chunk_bias + rare_w  =>  d(score')/d(score) = 1.
    # chunk_bias / rare_w are non-learnable (built fresh each step from
    # compiler edges), so no gradient flows to the aux tensors.
    return grad_out
```

Notes:

- `head` is unused (head-broadcast prior) but present in the signature.
- `mask_mod=None`, `block_sparse_tensors=None`: causal skip is handled by
  `causal=True`; the prior is purely additive.
- The sentinel slot `chunk_bias[b, C, *] = chunk_bias[b, *, C] = 0` makes
  inter-chunk-gap tokens contribute no chunk bias, matching the dense
  path's `masked_fill_(~(q_valid & k_valid), 0)` in
  `_scatter_chunk_relation_edges_`.
- The rare-edge scan reads only `rare_*[batch, lo:hi]`, a tiny per-row
  range. If profiling shows this is hot, the overlay can be promoted to a
  per-`(q, kv_block)` mini-CSR later; the aux contract is unchanged.

---

## 5. Memory comparison

Reference shape from the task: `B = 192`, `S = 1024`, `C = 64`.

Dense path today (`[B, 1, Sq, Sk]` bf16):

```
192 · 1024 · 1024 · 2 B = 384 MiB
```

(At the production cap `B=8, S=16384` this is the 4 GiB blowup that
`CPPMEGA_GRAPH_DENSE_MAX_SEQ` guards against.)

Chunk-native POC, same `B=192, S=1024, C=64`, with `max_rare_per_batch = 256`
(generous; cppmega point-edge densities are well under this):

| Tensor                | Shape                  | Bytes (bf16/int32)        |
|-----------------------|------------------------|---------------------------|
| `chunk_bias`          | `192 · 65 · 65`        | `· 2 B = 1.62 MiB` (bf16) |
| `token_to_chunk_q`    | `192 · 1024`           | `· 4 B = 0.75 MiB`        |
| `token_to_chunk_k`    | `192 · 1024`           | `· 4 B = 0.75 MiB`        |
| `rare_q`              | `192 · 256`            | `· 4 B = 0.19 MiB`        |
| `rare_k`              | `192 · 256`            | `· 4 B = 0.19 MiB`        |
| `rare_w`              | `192 · 256`            | `· 2 B = 0.09 MiB`        |
| `rare_row_offsets`    | `192 · 1025`           | `· 4 B = 0.75 MiB`        |
| `rare_meta`           | `4`                    | negligible                |

**Total ≈ 4.3 MiB**, vs **384 MiB** dense — an **~89× reduction** at the
reference shape, and the gap widens quadratically with `S`. The task's
back-of-envelope `chunk_bias = 192 · 65 · 65 · 4 B = 3.2 MB` assumes fp32;
the bf16 figure above is half that. We keep bf16 to match Q dtype and the
dense path's bf16 numerics; an fp32 builder option is available via the
existing `dtype` argument if a recipe needs it.

At the production cap `B=8, S=16384, C=256`:

- dense: `8 · 16384² · 2 B = 4 GiB` (the cap's exact motivating example);
- chunk-native: `chunk_bias 8 · 257² · 2 B ≈ 1.0 MiB`,
  `token_to_chunk 2 · 8 · 16384 · 4 B = 1 MiB`,
  rare overlay `≈ 0.5 MiB` ⇒ **≈ 2.5 MiB total**, a ~1600× reduction.

The savings are HBM residency only; the POC does **not** reduce FLOPs
(no `block_sparse_tensors` skip). FA4 still runs the full causal tile
schedule. Phase 2 (CSR design §4.3) adds the 128×128 skip and recovers the
FLOP savings; the chunk-native aux is compatible with that addition.

---

## 6. Direct FA4 adapter architecture

### 6.1 Why a direct adapter (TE 2.16 cannot do this for us)

TE 2.16 (`transformer_engine == 2.16.0.dev0+8e19460b` per
`artifacts/mamba3_wave29_modal_h200_preflight_20260430/preflight.json`,
`artifacts/mamba3_wave32_h200_20step_gate/.../backend_probe.json`) routes
attention through `TEDotProductAttention` → TE `FusedAttention`. TE's
attention API accepts an `attention_bias` tensor (the dense path cppmega
uses today) but **does not expose FA4's `score_mod` / `aux_tensors` /
`block_sparse_tensors` arguments**. There is no TE-side knob we can flip to
thread a custom `score_mod` into the FA4 kernel TE selects. See §7 for the
full blocker analysis.

Therefore the POC installs a **narrow adapter at the `core_attention`
seam**: replace `TEDotProductAttention` with `CppMegaFA4DotProductAttention`
in the layer's `ModuleSpec`, and call `flash_attn.cute.interface.flash_attn_func`
directly from its `forward`. Everything else in the layer stays in TE.

### 6.2 What stays in TE, what moves to the adapter

Stays in TE (untouched):

- QKV projection (`TEColumnParallelLinear`, including FP8 GEMMs);
- KV projection / GQA layout;
- RoPE / position embedding application;
- output projection (`TERowParallelLinear`, including FP8 GEMMs);
- layer-norm / residual fusion in `TransformerLayer`;
- checkpointing / recompute hooks at the layer boundary.

Moves to the adapter:

- the dot-product / softmax / PV kernel — replaced by a direct
  `flash_attn_func(...)` call carrying our `score_mod` + `aux_tensors`;
- attention dropout — **removed** (see blocker §7.1).

This is exactly the seam Megatron exposes via
`TransformerLayerSubmodules(self_attention=..., core_attention=ModuleSpec(
module=CppMegaFA4DotProductAttention, params={...}))`. No Megatron core
edits, no TE edits.

### 6.3 Forward flow

```
TransformerLayer.forward(hidden_states, attention_bias=None, ...)
  └─ self_attention(hidden_states, attention_mask, ...)
       ├─ TE QKV projection (FP8 GEMMs) → query/key/value [B, S, H, D]
       ├─ RoPE
       └─ core_attention(query, key, value, attention_bias=aux_or_none, ...)
            └─ CppMegaFA4DotProductAttention.forward
                 ├─ resolve FA4ChunkNativeAux (from attention_bias arg, or
                 │  rebuild from PromptGraphInferenceState in decode)
                 ├─ flash_attn_func(q, k, v,
                 │      softmax_scale=self.softmax_scale,
                 │      causal=True,
                 │      score_mod=graph_chunk_score_mod,
                 │      score_mod_bwd=graph_chunk_score_mod_bwd,
                 │      aux_tensors=[token_to_chunk_q, token_to_chunk_k,
                 │                   chunk_bias, rare_q, rare_k, rare_w,
                 │                   rare_row_offsets, rare_meta],
                 │      block_sparse_tensors=None,
                 │      mask_mod=None,
                 │      return_lse=False)
                 └─ return out [B, Sq, H, D]
       └─ TE output projection (FP8 GEMMs)
```

The existing `apply_graph_route_attention_bias_patch` wrapper is extended to
dispatch on `core_attention` type:

- `TEDotProductAttention` → existing dense `[B,1,Sq,Sk]` path (kept for
  cuDNN fallback / ablation);
- `CppMegaFA4DotProductAttention` → call
  `build_fa4_chunk_native_aux(...)` and assign the resulting dataclass to
  `bound.arguments["attention_bias"]`; the adapter's `forward` recognizes
  the type. A raw `torch.Tensor` arriving at the adapter raises (the FA4
  backend refuses dense bias by contract).

The pinned `TransformerLayer.forward` signature is unchanged
(`_PINNED_TRANSFORMER_PARAMETERS` in
`graph_route_attention_bias_patch.py`).

### 6.4 Adapter class shape

```python
class CppMegaFA4DotProductAttention(MegatronModule):
    def __init__(self, config, layer_number,
                 attention_type, num_attention_heads,
                 attention_dropout=0.0,        # must be 0; see §7.1
                 softmax_scale=None,           # default 1/sqrt(head_dim)
                 causal=True,                  # POC is causal-only
                 deterministic=False,
                 max_chunks=256,               # C high-water mark
                 max_rare_per_batch=256,       # rare-edge high-water mark
                 **_ignored_te_kwargs):
        ...
        if attention_dropout != 0.0:
            raise ValueError(...)              # blocker §7.1
        if not causal:
            raise ValueError("POC is causal-only")

    def forward(self, query, key, value,
                attention_mask=None,
                attn_mask_type=None,
                attention_bias=None,           # None | FA4ChunkNativeAux
                packed_seq_params=None,
                inference_context=None,
                **kwargs):
        ...
```

Fail-closed contract (parity with the dense patch):

- `graph_routes_active()` and `require_graph_routes_for_production()` gate
  the path identically.
- A raw `torch.Tensor` `attention_bias` raises (dense bias refused).
- MLA still raises. DSA still owns its own path (no double-apply).
- `sequence_parallel=True` and `context_parallel_size>1` raise.
- A declared edge outside the active query/key window raises (mirrors
  `_scatter_rectangular_token_edges_`).
- `max_chunks` / `max_rare_per_batch` overflow raises with a config-bump
  hint.

### 6.5 Backward

`flash_attn_func` is a `torch.autograd.Function`. With `score_mod` set, FA4
also requires `score_mod_bwd`; ours is the identity
(`d(score + bias)/d(score) = 1`) because the graph bias is **non-learnable**:
no gradient flows to `chunk_bias` / `rare_w` (built fresh each step from
compiler edges, not `nn.Parameter`s). `beta` is a python float resolved from
env; no autograd edge. `d(out)/d(q,k,v)` are computed by FA4's fused
backward.

### 6.6 Compile-key stability

FA4 recompiles when the `score_mod` hash, `aux_tensors` metadata, or
`block_sparse_tensors` shapes change. To avoid per-step recompiles:

- The builder always pads `chunk_bias` to `[B, C+1, C+1]` with `C =
  max_chunks` (high-water mark) and `rare_*` to `[B, max_rare_per_batch]`.
  Defaults are sized from the dataset inventory (`corpus_inventory.json`)
  plus a 2× safety margin; the builder raises on overflow.
- `rare_meta` is a fixed `[4]` int32 tensor; per-step `Sq/Sk/max_rare` go
  in its slots, but the **shape** never changes.
- `query_start` is folded into the host-side tensors (local row space), not
  passed as a runtime scalar, so decode steps with different offsets reuse
  the same kernel as long as `Sq/Sk` match. If decode `Sq` varies (e.g.
  spec-decode), the `Sq` slot in the compile key changes and FA4 recompiles
  — acceptable for the POC, fixable later via `aux_scalars` if needed.
- The `score_mod` callable is a module-level `@cute.jit` function; its hash
  is stable across processes.

---

## 7. Blocker analysis

Two hard blockers gate production adoption. Both are external (FA4 / TE
upstream), not cppmega-internal.

### 7.1 Blocker A — FA4 beta23 has no attention dropout

**Status (assumed for beta23, must be re-verified on the wheel):** the
`flash_attn.cute` `score_mod` path does not implement attention dropout.
The b19 surface in our image exposes no `dropout_p` argument on
`flash_attn_func` when `score_mod` is set, and we assume beta23 inherits
this.

**Why it matters:** Nebius's published H200 reference runs (the
`case6_nebius_h200_runbook.md` lineage) use `attention_dropout = 0.1`.
Any recipe that needs dropout for regularization or for parity with a
published baseline **cannot** use this POC path.

**Mitigations considered:**

1. **Drop dropout (chosen for cppmega production).** cppmega's production
   GPT runs already use `attention_dropout = 0` (the dense graph-route path
   is gated to those recipes). The adapter constructor raises on non-zero
   dropout, fail-closed. This unblocks the POC for cppmega-internal use.
2. **Emulate dropout in `score_mod`.** Possible in principle (sample a
   per-tile Bernoulli mask, scale by `1/(1-p)`, multiply into `score`), but
   requires a deterministic RNG contract across fwd/bwd that FA4 does not
   currently expose, and would change the kernel's numerical contract
   (dropout is multiplicative on attention probabilities, not additive on
   scores — emulating it via `score_mod` requires `-inf` masking, which
   interacts badly with the additive chunk bias). Rejected for the POC.
3. **Wait for upstream.** Track the FA4 beta changelog for native dropout
   support; revisit if/when it lands.

**Decision:** ship the POC with `attention_dropout = 0` only; document the
Nebius `0.1` recipe as out of scope until upstream FA4 adds dropout.

### 7.2 Blocker B — TE 2.16 does not expose `score_mod` to FA4

**Status (verified locally):** TE 2.16 (`2.16.0.dev0+8e19460b` in our
artifacts) routes attention through `TEDotProductAttention` →
`FusedAttention`. TE's attention forward accepts `attention_bias` (a dense
tensor) but has **no parameter** for FA4's `score_mod` /
`score_mod_bwd` / `aux_tensors` / `aux_scalars` / `block_sparse_tensors`.
Even when TE selects an FA4 backend internally, there is no TE-side surface
through which cppmega can hand a custom `score_mod` to that kernel.

**Why it matters:** the obvious integration ("just pass `score_mod` through
TE") is impossible without forking TE. The dense `[B,1,Sq,Sk]` bias path is
the only thing TE 2.16 can express, and that path forces cuDNN
`FusedAttention` (losing the FA4 fast path) and incurs the `O(B·Sq·Sk)`
memory blowup.

**Mitigations considered:**

1. **Narrow adapter at `core_attention` (chosen).** Replace
   `TEDotProductAttention` with `CppMegaFA4DotProductAttention` in the
   layer's `ModuleSpec`; call `flash_attn_func` directly. QKV/output
   projections, FP8 GEMMs, RoPE, layer-norm fusion all stay in TE — only
   the dot-product/softmax kernel moves out. This is the design in §6.
   Cost: cppmega owns one more attention module; we must re-validate
   numerics against the TE/cuDNN baseline (planned equivalence test in
   §8). Benefit: zero TE forks, zero Megatron core edits, the seam is the
   already-pinned `core_attention` slot.
2. **Patch TE to thread `score_mod` through.** Possible but high-cost:
   TE's `FusedAttention` backend selection, FP8 descale plumbing, and
   cuDNN fallback all assume a dense-bias contract. A TE fork would have
   to be maintained across TE releases. Rejected for the POC; revisit if
   upstream TE adds a `score_mod` pass-through.
3. **Wait for upstream TE.** TE is actively integrating FA4; a future
   release may expose `score_mod`. Track and revisit.

**Decision:** ship the POC via the narrow `core_attention` adapter; keep
the dense TE path live for ablation and for recipes that need TE-specific
features (e.g. attention dropout once blocker A is resolved upstream).

### 7.3 Secondary risks (not blockers, but tracked)

- **FA4 SM target.** Custom `score_mod` raises on SM8x. cppmega's H200
  (SM90) and Blackwell (SM100/SM120) targets are fine; A100 runs stay on
  the dense TE path. The backend selector enforces this.
- **beta23 API drift.** The image pins `4.0.0b19`; the POC targets the
  beta23 surface. Before implementation, run a one-shot probe against the
  actual `4.0.0b23` wheel: confirm `score_mod` signature, `aux_tensors`
  hashing, `score_mod_bwd` requirement, and the absence of dropout. Any
  drift updates this doc before code lands.
- **Numerical parity.** Bias is added post-scale in both paths (TE applies
  `attention_bias` after `softmax_scale`; FA4 with `score_mod` keeps
  `softmax_scale` separate and applies it before `score_mod`, so we
  pre-multiply weights by `softmax_scale` on host). bf16 accumulation in
  `chunk_bias` / `rare_w` matches the dense bf16 bias path. Duplicate-edge
  accumulation matches `index_add_` (sum).
- **No FLOP savings in POC.** Without `block_sparse_tensors`, FA4 runs the
  full causal tile schedule; the win is HBM only. Phase 2 adds the 128×128
  skip per `docs/fa4_score_mod_design.md` §4.3.
- **FP8 QKV descale.** cppmega currently does FP8 GEMMs in TE and hands
  bf16 QKV to core attention, so FA4 FP8 descales are out of scope. If we
  later pipe FP8 QKV directly into FA4, the descales must be threaded
  through the adapter.

---

## 8. Validation plan (for the implementation phase)

- **Equivalence (the gate).** For `B ∈ {1, 2, 8}`, `Sq = Sk ∈ {128, 512,
  1024}`, `C ∈ {8, 32, 64}`, random sparse chunk edges + random rare point
  edges: compare FA4 output to the dense TE path with the same
  `beta * S_graph` bias. Tolerance: bf16 `atol = 2e-2, rtol = 2e-2` (FA4 vs
  cuDNN baseline), plus a stricter fp32-reference check (`atol = 1e-4`).
- **Backward.** Gradient equivalence on `q, k, v` with the same tolerances;
  assert no gradient is requested for aux tensors.
- **Decode (rectangular).** `Sq = 1, Sk ∈ {512, 4096}` against
  `build_rectangular_graph_attention_bias_from_structure_batch`.
- **Sentinel correctness.** Tokens in inter-chunk gaps must contribute zero
  chunk bias; verify against the dense path's
  `masked_fill_(~(q_valid & k_valid), 0)` behavior.
- **Fail-closed.** Corrupt counts, out-of-range edges, `max_chunks` /
  `max_rare_per_batch` overflow, MLA layer, `sequence_parallel=True`,
  dense-tensor `attention_bias` into the adapter, non-zero
  `attention_dropout` — all raise.
- **Compile-key stability.** Run two consecutive steps with different edge
  counts but identical high-water marks; assert FA4's compile cache hit
  count does not change (via `flash_attn.cute` compile counters).
- **Memory.** Assert the aux tensors at `B=192, S=1024, C=64` are under
  8 MiB total (well under the 384 MiB dense baseline).

---

## 9. Deliverables (for the implementation phase)

1. `cppmega/megatron/fa4_chunk_native_aux.py` — `FA4ChunkNativeAux`
   dataclass + `build_fa4_chunk_native_aux(...)` + `score_mod` callables.
2. `cppmega/megatron/fa4_dot_product_attention.py` —
   `CppMegaFA4DotProductAttention` Megatron module (the §6 adapter).
3. `cppmega/megatron/fa4_attention_spec.py` — `ModuleSpec` helper.
4. Edits to `graph_route_attention_bias_patch.py` to dispatch on
   `core_attention` type and build chunk-native aux for FA4 layers; dense
   path preserved.
5. `tests/megatron/test_fa4_chunk_native_score_mod.py` — equivalence,
   backward, decode, sentinel, fail-closed, compile-key stability, memory.
6. Run-profile flag `CPPMEGA_ATTENTION_BACKEND ∈ {te, fa4_chunk_native}`
   (default `te` until validated) plumbed through `gpt_builder.py` and the
   relevant run scripts.

---

## 10. References (local)

- `cppmega/megatron/graph_route_attention_bias_patch.py` — dense bias
  builder (`build_dense_graph_attention_bias_from_structure_batch`,
  lines 172-267), `TransformerLayer.forward` patch,
  `PromptGraphInferenceState`.
- `cppmega/megatron/dsa_indexer_fused_patch.py` — `_as_batched_edges`,
  `_as_batched_edge_triples`, `_as_batched_chunks`, `_token_chunk_map`,
  `_scatter_chunk_relation_edges_` (line 539), `_scatter_edges_`,
  `build_graph_route_bias_from_structure_batch`.
- `cppmega/megatron/graph_objective_loss.py` — `graph_routes_active`,
  `resolve_graph_bias_beta`, `validate_graph_bias_beta`.
- `cppmega/megatron/fa4_graph_attention.py` — existing CSR-based b19
  implementation (companion approach; not used by this POC).
- `docs/fa4_score_mod_design.md` — CSR + 128×128 block-sparse design for
  the b19 surface (Phase-2 skip strategy reused later).
- `docs/case6_nebius_h200_runbook.md` — Nebius H200 reference (dropout
  `0.1` baseline; blocker A context).
- `STACK.lock`, `docker/Dockerfile` — `flash-attn-4[cu13]==4.0.0b19` pin
  and `flash_attn.cute` import smoke test.
- `artifacts/mamba3_wave29_modal_h200_preflight_20260430/preflight.json`,
  `artifacts/mamba3_wave32_h200_20step_gate/wave32_h200_backend_preflight_20260430/backend_probe.json`
  — TE `2.16.0.dev0+8e19460b` evidence (blocker B context).
- FA4 source (read-only reference, not vendored):
  `flash_attn/cute/interface.py::flash_attn_func`,
  `flash_attn/cute/utils.py::compute_softmax_scale_log2`,
  `flash_attn/cute/utils.py::create_softcap_scoremod`.
