# Document isolation + Sliding Window Attention: `window_size` plumbing design

**Status**: design + partial implementation (2026-08-01)  
**Scope**: `cppmega/megatron/document_isolation.py`, `cppmega/megatron/fa4_score_mod_adapter.py`,
`cppmega/megatron/mla_shared.py`, and the Megatron `TransformerConfig` → `ModuleSpec` wiring.  
**Depends on**: P075 (beta23 gate evidence) for training seq_len > 8 k; the plumbing itself is
local and can land earlier.

---

## 1. What exists today

### 1.1 FA4 chunk-native attention (`fa4_score_mod_adapter.py`)

`CppMegaFA4ScoreModAttention` already builds a packed-document `mask_mod` when
`document_layout` reports multiple documents per row.  The helper that builds the
mask callback accepts `window_size_left` / `window_size_right`:

```python
# cppmega/megatron/fa4_score_mod_adapter.py:1254
_make_document_causal_mask_mod(
    document_ids_q_index,
    document_ids_k_index,
    *,
    causal: bool,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
)
```

But `CppMegaFA4ScoreModAttention.__init__` does **not** accept a `window_size`
and `forward` always calls `_make_document_causal_mask_mod(..., causal=causal)`,
leaving both window values at `None`.  The native FA4 `flash_attn_func` call also
omits `window_size`, so even single-document rows run full causal attention.

### 1.2 MLA (`mla_shared.py`)

The adapters (`CppMegaMLASelfAttentionAdapter`, `CppMegaFusedMLASelfAttentionAdapter`,
`CppMegaAbsorbedMLASelfAttentionAdapter`) pass `*args, **kwargs` straight to the
upstream Megatron MLA class.  Megatron's MLA classes already forward `window_size`
to `te.DotProductAttention` **if** the value is present in their constructor
kwargs.  The gap is not in the adapter code — it is in the `ModuleSpec` params
that the spec builders supply.  No cppmega spec currently passes `window_size`.

### 1.3 Document isolation masks

- `_patch_te_attention`: when `attention_bias is None`, builds `PackedSeqParams`
  and lets TE handle causal + SWA.  TE respects its own `window_size` argument,
  but the patch never passes one, so multi-document rows fall back to full causal.
- `_patch_torch_attention`: builds a dense bool mask `cross_document | future`.
  It ignores any configured window, so even single-document rows attend to the
  full history.
- `_patch_dsa_attention`: same dense mask as the torch path, used only as a
  safety net; real isolation is `mask_sparse_topk_by_document`.

---

## 2. Proposed plumbing

### 2.1 Source of truth

The single source of truth for SWA is the run profile:

```bash
# argparse in the training launch script
--attention-window-size 8192   # default None (full attention)
```

This value is placed on `TransformerConfig` as `config.window_size` (tuple).  We
recommend the Megatron convention:

```python
window_size = (args.attention_window_size, 0)  # (left, right); 0 = attend only to self
```

`0` on the right is the standard SWA convention: a token may attend to itself
and up to `left` tokens before it.  If the recipe ever needs bidirectional SWA,
use `(left, right)` with both values positive.

### 2.2 FA4 chunk-native attention

Change `CppMegaFA4ScoreModAttention`:

1. Add `window_size: tuple[int | None, int | None] = (None, None)` to
   `__init__` and store `self.window_size`.
2. In `forward`, when `document_mask_aux is not None`, pass
   `window_size_left` / `window_size_right` to `_make_document_causal_mask_mod`.
3. When `document_mask_aux is None` (single-document row), pass
   `window_size=self.window_size` to `flash_attn_func` so the native FA4 SWA
   fast path is used.
4. Keep `native_causal = causal if mask_mod_fn is None else False` — the
   `mask_mod` callback now owns the complete predicate for packed rows, including
   SWA.

The existing `_make_document_causal_mask_mod` implementation is already correct:

```python
if window_size_left is not None and window_size_left >= 0:
    keep = keep & (kv_idx >= q_absolute - window_left)
if window_size_right is not None and window_size_right >= 0:
    keep = keep & (kv_idx <= q_absolute + window_right)
```

The only change is wiring the values through.

### 2.3 MLA spec builders

Update every cppmega spec builder that emits an MLA self-attention `ModuleSpec`
to include `window_size` in the params block when it is configured.  The adapter
classes pass kwargs through, so no code change is needed inside
`mla_shared.py`; only the callers need to pass it.

Example in `nam56r_te_spec.py` (or equivalent):

```python
self_attention=ModuleSpec(
    module=CppMegaFusedMLASelfAttentionAdapter,
    params={
        ...,
        "window_size": getattr(config, "window_size", (None, None)),
    },
),
```

If the upstream Megatron `FusedMLASelfAttention` constructor does not accept
`window_size`, wrap it with the adapter and absorb the argument there before
calling `super().__init__`.  The adapters should therefore be defensive:

```python
class CppMegaFusedMLASelfAttentionAdapter(FusedMLASelfAttention):
    def __init__(self, *args, window_size=(None, None), pp_layer_offset=None, **kwargs):
        # window_size is consumed here and forwarded only if upstream accepts it.
        if pp_layer_offset is not None and _supports_pp_layer_offset(FusedMLASelfAttention):
            super().__init__(*args, pp_layer_offset=pp_layer_offset, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self._cppmega_window_size = window_size
```

### 2.4 Document isolation masks (torch / DSA)

For `_patch_torch_attention` and `_patch_dsa_attention`, extend the dense mask:

```python
mask = cross_document | future
if window_size_left is not None and window_size_left >= 0:
    mask |= kv_idx < q_idx - window_size_left
```

`window_size_left` is read from `self.config.window_size` on the attention module
instance.  The patch already has access to `self.config`.  This branch is a
fallback; the FA4 and MLA paths are the production ones.

For `_patch_te_attention`, the varlen path already delegates causal masking to
TE.  TE's `TEDotProductAttention` will apply `window_size` if the spec builder
passes it.  No extra code is required in the patch.

---

## 3. Per-backend behavior summary

| Backend | Packed rows | Single-document rows | Where `window_size` comes from |
|---|---|---|---|
| TE / `TEDotProductAttention` | `PackedSeqParams` → TE applies window_size | TE applies window_size | `ModuleSpec` params |
| FA4 chunk-native (`CppMegaFA4ScoreModAttention`) | `mask_mod` with left/right window | `flash_attn_func(window_size=...)` | `__init__` param / `ModuleSpec` params |
| Torch (`DotProductAttention`) | dense bool mask + window mask | dense bool mask + window mask | `self.config.window_size` read in patch |
| DSA (`DSAttention`) | `-1` sentinel + dense bool mask + window mask | dense bool mask + window mask | `self.config.window_size` read in patch |

---

## 4. Testing plan

### 4.1 Unit tests (CPU, no GPU)

1. `test_fa4_score_mod_attention_accepts_window_size`
   - Construct `CppMegaFA4ScoreModAttention(window_size=(4, 0))`.
   - Assert `attn.window_size == (4, 0)`.
2. `test_fa4_score_mod_attention_passes_window_to_mask_mod`
   - Monkey-patch `_make_document_causal_mask_mod` to record arguments.
   - Call `forward` with multi-document `document_ids` and `window_size=(4, 0)`.
   - Assert recorded `window_size_left == 4` and `window_size_right == 0`.
3. `test_torch_attention_mask_includes_window`
   - Construct a `DotProductAttention` mock with `config.window_size = (2, 0)`.
   - Run `_patch_torch_attention` isolated forward.
   - Assert that positions more than 2 tokens behind are masked.

### 4.2 Numerical parity (GPU required)

Run on H200/H100 when available:

1. Single-document row, `window_size=(8192, 0)`, compare
   `CppMegaFA4ScoreModAttention` output against `flash_attn_func` reference with
   the same window.  Tolerance: bf16 `atol=2e-2, rtol=2e-2`.
2. Multi-document row, `window_size=(8192, 0)`, compare `mask_mod` path against
   a CPU reference that builds a dense `[B,1,Sq,Sk]` bool mask with cross-doc +
   causal + window predicates.  Tolerance: `max_abs_diff < 1e-3` on a small fp32
   shape.
3. MLA with `--attention-window-size 8192`: run a short training step at 32 k
   seq and verify loss matches the full-attention run within statistical noise.

### 4.3 No-op boundary

At seq_len <= `window_size_left`, SWA must be a mathematical no-op.  Add a unit
test that compares the mask produced by `_make_document_causal_mask_mod` with
`window_size=(8, 0)` against the same helper with `window_size=(None, None)` for
a sequence of length 8.  The masks must be identical.

---

## 5. Risks and open questions

1. **FA4 `mask_mod` + native causal interaction.**  When `mask_mod` is provided,
   FA4 beta23 disables the native causal fast path.  For packed rows this is
   unavoidable — the callback must own the complete predicate.  For
   single-document rows we keep `mask_mod=None` and use `window_size` directly,
   preserving the native fast path.
2. **Right-side window with causal.**  `window_size_right=0` means "attend only
   to self and earlier tokens", which is the canonical SWA convention.  If a
   recipe needs a true local band (right > 0), the mask callback supports it,
   but the native FA4 `window_size=(left, right)` path also supports it only
   when `mask_mod` is None.
3. **CP / SP interaction.**  `gather_context_parallel_sequence` restores the
   global order before `map_sharded_sequence_by_document` isolates documents.
   The window predicate in `_make_document_causal_mask_mod` uses absolute token
   indices (`q_absolute = q_idx + query_start`), so it is compatible with CP as
   long as the global sequence is reconstructed first.  This is already the
   contract in `map_sharded_sequence_by_document`.
4. **MLA spec builders.**  The exact list of spec builders that need the param
   must be enumerated.  Candidates: `nam56r_te_spec.py`, `nam56r_full_spec.py`,
   `nam56r_noconv_spec.py`, `nam56r_lite_spec.py`, and any GPT spec that uses
   `mla_shared.py`.

---

## 6. Deliverables

1. `docs/document_isolation_swa_design.md` — this document.
2. `CppMegaFA4ScoreModAttention` accepts and forwards `window_size`.
3. Adapter classes in `mla_shared.py` accept `window_size` and forward it only
   when upstream supports it.
4. Spec builders pass `config.window_size` into MLA `ModuleSpec` params.
5. `_patch_torch_attention` / `_patch_dsa_attention` read window from
   `self.config` and extend the dense mask.
6. CPU unit tests covering construction, mask-mod argument forwarding, and
   torch/DSA mask behavior.

**GPU-required parity tests are deferred to the next H200/Modal slot and tracked
as a sub-checklist inside this issue.**
