# Document isolation: varlen/cu_seqlens path design

**Status**: design + deferred prototype (2026-08-01)  
**Scope**: `cppmega/megatron/document_isolation.py` attention paths and the
`map_sequence_by_document` ponytail.  
**Depends on**: P073 (packed-document isolation gates landed in `cppmega@27f77825`).

---

## 1. What exists today

`document_isolation.py` has four attention seams:

| Seam | Multi-document handling today | Notes |
|---|---|---|
| `TEDotProductAttention` (`_patch_te_attention`) | Uses `PackedSeqParams(qkv_format="thd", cu_seqlens_q=..., cu_seqlens_kv=...)` | Already varlen. Only active when `attention_bias is None`. |
| `DotProductAttention` (`_patch_torch_attention`) | Builds a dense bool mask `[B,1,S,S]` from `document_layout` | Generic, no TE/FA4 dependency. |
| `DSAttention` (`_patch_dsa_attention`) | Same dense bool mask + `-1` sentinel in sparse indices | Mask is a safety net; the real isolation is the `-1` sentinel in `mask_sparse_topk_by_document`. |
| FA4 graph-route (`fa4_score_mod_adapter.py`) | Uses `mask_mod`/`document_ids_q/k` aux | Handles packed docs via FA4's custom mask, not cu_seqlens. |

For stateful non-attention modules (`map_sequence_by_document`) the code currently
falls back to the "ponytail" path: segments are padded to a common length,
stacked into one big batch, the user function is called once, and results are
sliced back. This is correct but allocates the padded tensor and forces one
large kernel launch.

The varlen question applies to two different layers:

1. **Attention isolation** — can we replace dense masks with `cu_seqlens` for
   backends that support it (flash-attn, FA4 varlen, TE)?
2. **Stateful module isolation** — can we replace the ponytail with a true
   varlen kernel once cppmega has one?

This doc focuses on (1), because (2) requires a custom CUDA/TileLang kernel that
does not exist yet.

---

## 2. Where cu_seqlens is already cheaper

### 2.1 TE path

`TEDotProductAttention.forward` already takes `PackedSeqParams`. When
`attention_bias is None` and the row packs multiple documents, the code computes
lengths, builds `cu_seqlens`, reshapes Q/K/V to `(total_tokens, ..., h, d)` and
passes them to TE. This is the desired end state for the TE backend.

Remaining gap: when `attention_bias` is a dense torch.Tensor, the code falls back
to per-document slicing and `N` separate attention calls. A varlen path for
chunked bias is not implemented and is left for future work.

### 2.2 Torch attention path

`DotProductAttention` is Megatron's reference attention. On CUDA it usually
routes to `torch.nn.functional.scaled_dot_product_attention` or
`flash_attn.flash_attn_func`. For packed documents the current code injects a
dense bool mask. This works on CPU and CUDA but:

- wastes compute on masked-out positions,
- increases memory traffic for the `[B,1,S,S]` mask,
- disables some flash-attn fast paths that expect `cu_seqlens`.

**Candidate change**: when `flash_attn` is importable and the row packs multiple
documents, convert `document_layout` lengths into `cu_seqlens` and call
`flash_attn_varlen_func` (or `flash_attn_varlen_qkvpacked_func`). When a dense
`attention_mask` is also supplied, combine it with the document mask only for
positions that are not already masked by the varlen boundaries.

**Constraints**:
- Must keep CPU fallback.
- Must preserve numerical parity within the existing bf16-noise tolerance.
- Must handle SP/CP (sequence/context parallelism) the same way the mask path does.

### 2.3 FA4 graph-route path

`fa4_score_mod_adapter.py` builds `document_ids_q/k` aux tensors and passes them
to the FA4 `mask_mod`. FA4 also has a varlen API (`flash_attn_func` with
`cu_seqlens` in some versions). The current path was chosen because:

- It reuses the same `mask_mod` machinery for both graph-route and packed
  documents.
- It does not require a separate code branch for varlen shapes.

A dedicated varlen branch would need:
- conversion of `document_layout` to `cu_seqlens_q/k`,
- rectangular decode support (`seqlen_q != seqlen_k`),
- parity checks against the existing `mask_mod` path.

The expected win is small for the graph-route use case because the `mask_mod`
aux is already cheap (`document_ids` are int32 and the mask is computed lazily).
The main cost is the attention computation itself, which varlen cannot reduce
for graph-route sparse patterns.

### 2.4 DSA path

DSA sparse attention uses block-sparse indices and a `-1` sentinel for
out-of-document selections. Varlen formatting does not map cleanly to this
sparse index space, so the mask path is kept as a safety net. The real isolation
remains `mask_sparse_topk_by_document`.

---

## 3. Proposed design

### 3.1 New helper: `_document_layout_to_cu_seqlens`

Add a small helper next to `document_layout` that returns lengths and
`cu_seqlens` for the varlen backends:

```python
def _document_layout_to_cu_seqlens(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return cu_seqlens for packed documents, or None if no row is multi-doc."""
    ids, spans, multiple = document_layout(
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
    )
    if ids is None or not multiple:
        return None
    lengths = [
        end - start
        for spans_row in spans
        for start, end in spans_row
        if ids[0, start] > 0  # skip padding spans
    ]
    return torch.tensor(
        [0, *accumulate(lengths)],
        dtype=torch.int32,
        device=device,
    )
```

### 3.2 Torch attention varlen branch

In `_patch_torch_attention`, after detecting multiple documents and before
building the dense mask, try the varlen branch:

1. Require `flash_attn` and `query.dim() == 4`.
2. Compute `cu_seqlens`.
3. Reshape Q/K/V from `[S,B,H,D]` to `[total_tokens,H,D]` using the spans.
4. Call `flash_attn_varlen_func` with `cu_seqlens_q == cu_seqlens_k`.
5. Reshape back.

If any requirement fails, fall back to the mask path.

### 3.3 Stateful module varlen branch (future)

`map_sequence_by_document` keeps the ponytail until cppmega has a custom autograd
Function or TileLang kernel that accepts `cu_seqlens` and a user-provided
per-segment callable. The comment on line 195 should reference this doc.

---

## 4. Parity / measurement plan

When an H200/H100 slot is available:

1. **Correctness**: generate packed `document_ids` with 1–4 docs per row, run
   both mask path and varlen path, assert `max_rel_err < 1e-2` and
   `bad_frac(rtol=0.1, atol=0.1) < 1e-3` on the output and all gradients.
2. **Performance**: nsys profile on a realistic microbatch (NAM56R shape,
   MBS=8, S=8192/16384/32768/131072) comparing:
   - kernel time,
   - peak memory,
   - number of kernel launches.
3. **Win threshold**: adopt the varlen branch only if it is ≥5 % faster or
   saves ≥5 % memory at 128 k without regressing parity.

---

## 5. Decision and next steps

- **TE path**: already optimal for the no-bias case; bias case stays as sliced
  calls.
- **Torch attention**: implement the varlen branch behind a runtime flag
  `CPPMEGA_DOC_ISOLATION_VARLEN_ATTN=1`, keep mask path as default until parity
  and performance are proven on H200.
- **FA4 graph-route**: keep `mask_mod` path; varlen does not obviously help and
  adds rectangular-decode complexity.
- **DSA**: keep `-1` sentinel path.
- **Stateful modules**: keep ponytail; revisit after a cppmega varlen kernel
  exists.

**This design is deferred to a future GPU profiling slot.** The issue is closed
with this doc so that the next session does not re-discover the trade-offs.
