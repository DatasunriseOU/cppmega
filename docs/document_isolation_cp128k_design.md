# Document isolation + Context Parallelism for 128 k training — design

**Status**: design only (2026-08-01)  
**Scope**: `cppmega/megatron/document_isolation.py`,
`cppmega/megatron/fa4_score_mod_adapter.py`, CP-aware spec builders, and the
128 k context-extension phase described in `docs/long_context_roadmap.md`.  
**Depends on**: P083 (SWA `window_size` plumbing, closed locally) and P075
(beta23 H200 gate).

---

## 1. Why CP reopens at 128 k

`docs/long_context_roadmap.md` keeps CP closed until the context-extension phase
because at 4 k the overhead of CP (zigzag reorder, extra communication,
restrictions on custom sequence mixers) is not worth the activation savings.
At 128 k the O(seq²) MLA cost dominates memory, so sharding the sequence
dimension becomes profitable.  This doc describes what has to change in
document isolation and the attention backends before CP can be enabled for
packed-document training.

---

## 2. What already works

`document_isolation.py` already has the sequence-level CP primitives:

- `gather_context_parallel_sequence(tensor, cp_group)` — restores the global
  order from Megatron's CP zigzag layout using
  `_undo_attention_load_balancing(..., packed_seq_params=None)`.
- `scatter_context_parallel_sequence(tensor, cp_group)` — scatters a global
  tensor back into the local CP layout using
  `_redo_attention_load_balancing(..., packed_seq_params=None)`.
- `_GatherSequence` / `_ScatterSequence` — autograd-aware sequence gather/scatter
  used for SP.
- `_validate_model_parallel_topology(config, tp_group, cp_group, ...)` — fail-fast
  if configured TP/CP sizes do not match the process mesh.
- `_assert_mamba_cp_signatures()` — pins the private Megatron helper signatures
  so a silent upstream change cannot corrupt the reorder path.

`map_sharded_sequence_by_document` already implements the correct high-level
flow for stateful modules:

```
if SP:  gather sequence
if CP:  gather_context_parallel_sequence
apply user function per document
if CP:  scatter_context_parallel_sequence
if SP:  scatter sequence
```

This means CP for **stateful non-attention modules** (MTP roll, custom SSM
post-processing, loss masking) is mostly a question of validation, not new code.

---

## 3. Gaps for the attention backends

### 3.1 TE path (`_patch_te_attention`)

Today the patch raises as soon as `context_parallel_size > 1`:

```python
if int(getattr(self.config, "context_parallel_size", 1)) != 1:
    raise NotImplementedError(
        "packed-document TE attention does not support context parallelism"
    )
```

For 128 k this path must work because the 4 MLA layers are the memory
blockers.  TE's `TEDotProductAttention` supports CP when it is given the right
`PackedSeqParams` / `cp_group`.  The current multi-document branch already builds
`cu_seqlens` from `document_layout` and reshapes Q/K/V to `thd` format.  The
missing pieces are:

1. Pass `cp_group` (from the layer's process-group collection) into the TE
   attention call when `context_parallel_size > 1`.
2. Verify that `cu_seqlens` built from document boundaries is compatible with
   CP: the global sequence length must be divisible by `cp_size`, and every
   document boundary must fall inside the gathered global tensor.  Both are true
   because `map_sharded_sequence_by_document` gathers the full global sequence
   before isolation.
3. Confirm numerically that TE's CP varlen path gives the same result as the
   non-CP varlen path when `cp_size = 1` (regression guard).

### 3.2 FA4 chunk-native path (`CppMegaFA4ScoreModAttention`)

This module also raises on `context_parallel_size > 1`.  The right fix depends
on what FA4 beta23 supports:

- **Option A — CP inside `flash_attn_func`.**  If FA4 beta23 exposes CP-group
  arguments for `score_mod` + `mask_mod`, pass the local CP group and let FA4
  handle the sequence sharding.  This is the smallest change but requires
  verifying that `mask_mod` with packed-document aux is CP-safe.
- **Option B — gather/scatter around the module.**  Mirror
  `map_sharded_sequence_by_document`: gather the full CP sequence, run FA4
  attention with the existing packed-document mask, then scatter the output and
  gradients.  This is heavier communication but reuses the already-tested
  `mask_mod` path.

**Recommended**: start with Option B because it keeps the kernel contract
identical to the non-CP case; switch to Option A only if profiling shows Option
B is communication-bound.

For Option B the module needs:

1. Accept `cp_group` (or discover it from `parallel_state` at forward time).
2. Detect `context_parallel_size > 1` and call
   `gather_context_parallel_sequence` on Q/K/V before the SBHD → BSHD transpose.
3. Adjust `seqlen_q` / `seqlen_k` to the gathered global length.
4. Run the existing forward logic.
5. Scatter the output with `scatter_context_parallel_sequence` before returning
   to Megatron's local layout.

Important: the graph-route `ChunkNativeGraphBias` is built from the structure
batch which describes the **global** document layout, so it must be built
**after** gathering, using the gathered sequence lengths.  The same is true for
the `document_ids` aux used by `mask_mod`.

### 3.3 Torch / DSA fallback paths

`_patch_torch_attention` and `_patch_dsa_attention` do not explicitly reject CP
today, but they build the mask from the local `document_layout` which only sees
the local sequence slice.  With CP enabled the patches must either:

- gather the CP sequence first (like the FA4 Option B), or
- rely on `map_sharded_sequence_by_document` wrapping these modules.

Because these are fallback/reference paths, the recommended fix is to make the
patches fail-closed when CP is active until a dedicated CP branch is added:

```python
if int(getattr(self.config, "context_parallel_size", 1)) != 1:
    raise NotImplementedError(
        "packed-document torch/DSA attention does not yet support context parallelism; "
        "use the TE or FA4 backend for CP runs"
    )
```

This avoids silent wrong results from local-only masks.

### 3.4 MTP roll (`roll_tensor_by_document`)

Today:

```python
if cp_group is not None and cp_group.size() > 1:
    raise NotImplementedError("packed-document MTP does not support context parallelism")
```

At 128 k, MTP roll still needs cross-document zeroing.  With CP the tensor is
local; the global document IDs are not available on the local rank.  Options:

1. **Gather before roll.**  Use `gather_context_parallel_sequence` to get the
   global tensor and global `document_ids`, roll, then scatter.  This is simple
   and correct but adds an extra gather/scatter around MTP.
2. **CP-aware roll.**  If the CP zigzag local layout preserves enough document
   boundary information, zero only the boundaries that are local.  This is more
   efficient but subtle and easy to get wrong.

**Recommended**: Option 1 for the first 128 k phase; revisit Option 2 only if
MTP becomes a bottleneck.

---

## 4. Process-group and topology constraints

CP reintroduces the same topology constraints that `map_sharded_sequence_by_document`
already checks:

- SP and CP must be distinct Cartesian axes (no rank belongs to both groups
  except the trivial overlap of the current rank).
- TP and CP may overlap only at the current rank.
- The global sequence length must be divisible by `2 * cp_size` because Megatron's
  zigzag reorder works on pairs of chunks.
- Document IDs must be consistent across CP ranks: every rank sees the same
  global `document_ids` after `gather_context_parallel_sequence`.

These are validated by the existing helpers; no new validation code is needed
for the gather/scatter paths.

---

## 5. Proposed implementation plan

### Phase 1 — fail-closed + design (this doc)

1. Add explicit CP rejection to `_patch_torch_attention` and
   `_patch_dsa_attention` with a message pointing to this doc.
2. Keep the existing CP rejection in `_patch_te_attention` and
   `CppMegaFA4ScoreModAttention` but update the error messages to reference
   `docs/document_isolation_cp128k_design.md`.
3. Add a CP smoke test that asserts the rejections fire with a fake CP group.

### Phase 2 — TE CP varlen path (H200 required)

1. Plumb `cp_group` from the layer's `pg_collection` to the
   `TEDotProductAttention` call inside `_patch_te_attention`.
2. Run a parity check: CP=2 vs CP=1 vs no-CP on a small packed-document batch
   at 16 k or 32 k.
3. If parity passes, remove the CP rejection from `_patch_te_attention`.

### Phase 3 — FA4 CP path (H200 required)

1. Implement Option B (gather → FA4 → scatter) in
   `CppMegaFA4ScoreModAttention`.
2. Verify that `ChunkNativeGraphBias` and `document_ids` aux are built from the
   gathered global sequence.
3. Parity-check against the non-CP FA4 path and against the TE path.

### Phase 4 — MTP + stateful modules (H200 required)

1. Implement gather → `roll_tensor_by_document` → scatter for CP.
2. Run MTP-specific parity tests.

### Phase 5 — production enablement

1. Update the 128 k run profile to set `context_parallel_size=2` (or higher)
   and re-evaluate MFU.
2. Document the final CP configuration in `docs/long_context_roadmap.md`.

---

## 6. Memory and performance estimate

At 128 k, 4 MLA layers, batch size 1, full attention:

- MLA activation memory: ~260 GiB / sample (from the roadmap table).
- CP=2 halves the per-rank activation footprint to ~130 GiB / sample.
- Communication: one all-gather / reduce-scatter per layer for CP, plus the
  existing SP/TP collectives.
- Expected net win: enables MBS > 1, which recovers GEMM efficiency.  The exact
  win depends on the interconnect and is measured in Phase 2.

With SWA (`window_size = 8192`) the activation cost is much lower, but CP still
helps because the MLA layers remain the dominant memory consumer.

---

## 7. Testing plan

### 7.1 CPU / local tests

1. `test_te_attention_rejects_cp_until_phase2` — assert the updated error
   message references `docs/document_isolation_cp128k_design.md`.
2. `test_fa4_score_mod_attention_rejects_cp` — same for
   `CppMegaFA4ScoreModAttention`.
3. `test_torch_and_dsa_attention_reject_cp` — new fail-closed checks.
4. `test_cp_gather_scatter_roundtrip` — using `gloo`, verify
   `gather_context_parallel_sequence(scatter_context_parallel_sequence(x)) == x`
   for a sequence divisible by `2 * cp_size`.

### 7.2 GPU parity tests (H200 / Modal / Nebius)

1. TE varlen path, CP=2 vs CP=1, packed documents, 16 k / 32 k seq.
2. FA4 chunk-native path, CP=2 vs CP=1, packed documents, 16 k / 32 k seq.
3. End-to-end 128 k context-extension step with CP=2 + SWA.

---

## 8. Decision and next steps

- **Keep CP closed at 4 k / 16 k** — the overhead is not justified.
- **Reopen CP investigation at 128 k** as planned in the roadmap.
- **Phase 1 (this doc + fail-closed messages + CPU tests)** can land now.
- **Phase 2–4 require H200/Modal time** and are blocked by P075 (beta23 gate).

This issue is closed with this design doc so that the next 128 k scoping
session does not re-discover the same gaps.
