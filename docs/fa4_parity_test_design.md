# FA4 Forward/Backward Parity Test Design

Status: implementation complete
Targets: H200 (SM90) via Modal
Companion: `tests/test_fa4_h200_parity.py`

---

## 1. Objective

Verify that the FA4 chunk-native `score_mod` path produces numerically
equivalent results to the TE cuDNN FusedAttention dense `[B,1,Sq,Sk]`
post_scale_bias path (current Nebius production path) on the SAME input.

Checks:
- Forward outputs match within bf16 tolerance (~1e-3 atol, ~1e-2 rtol)
- Backward gradients (dQ, dK, dV) match within bf16 tolerance
- Loss values match

---

## 2. Paths Under Test

### Reference: TE cuDNN FusedAttention (dense bias)

The current Nebius path materializes a dense `[B, 1, Sq, Sk]` bias via
`build_dense_graph_attention_bias_from_structure_batch` and passes it as
`attention_bias` to `TEDotProductAttention`. TE forwards it to cuDNN
FusedAttention as a post-scale additive bias:

```
scores = (Q @ K^T) * softmax_scale + dense_bias   # [B, H, Sq, Sk]
attn   = causal_softmax(scores)
out    = attn @ V                                  # [B, H, Sq, D]
```

For the parity test, we use a manual PyTorch implementation of this exact
computation as the reference (mathematically identical to what cuDNN computes).
This avoids instantiating the full Megatron TransformerLayer while preserving
the mathematical contract.

### Test: FA4 chunk-native score_mod

The new path builds a `ChunkNativeGraphBias` via `build_chunk_native_graph_bias`
and passes it to `flash_attn_func` (FA4 beta23+) as `score_mod` + `aux_tensors`:

```
score_mod(score, b, h, seqlen_info, q, k, aux_tensors):
    score' = score + chunk_bias[b, token_to_chunk_q[b,q], token_to_chunk_k[b,k]]
                   + rare_edge_weight(b, q, k)
```

FA4 applies `softmax_scale` internally BEFORE calling `score_mod`, so the bias
is added to already-scaled scores (TE post_scale_bias semantics).

---

## 3. Test Input Construction

### Q, K, V

Random bf16 tensors on H200:
- Q: `[B, S, H, D]` with `requires_grad=True`
- K: `[B, S, H, D]` with `requires_grad=True`
- V: `[B, S, H, D]` with `requires_grad=True`

Default dimensions: `B=2, S=128, H=8, D=64` (small enough for fast iteration,
large enough to exercise multiple FA4 tiles).

### Mock structure_batch

A known graph with:
- 4 chunks covering the sequence: `[0,32), [32,64), [64,96), [96,128)`
- Call edges (chunk pairs): `(0,2), (1,3), (0,1)`
- Type edges (chunk pairs): `(2,0), (3,1)`
- Domain edges (token triples): `(10, 80, 5), (50, 20, 5)`
- Build edges (token triples): `(100, 5, 7)`

This exercises both the chunk-pair bias path AND the rare-edge overlay.

### Bias construction

Both paths use the SAME `structure_batch` and the SAME weights/beta:
- `beta = 2.0`
- `call_weight = 1.0, type_weight = 1.0`
- `domain_weight = 3.0, build_weight = 4.0`

---

## 4. Forward Comparison

### Reference (manual PyTorch, TE-equivalent)

```python
scale = 1.0 / sqrt(D)
scores = (Q @ K.transpose(-2,-1)) * scale  # [B, H, Sq, Sk]
scores = scores + dense_bias               # [B, 1, Sq, Sk] broadcast over H
scores = scores + causal_mask              # -inf above diagonal
attn   = softmax(scores, dim=-1)
out    = attn @ V                          # [B, H, Sq, D]
```

### Test (FA4 flash_attn_func)

```python
out = flash_attn_func(
    q=Q, k=K, v=V,
    softmax_scale=scale,
    causal=True,
    score_mod=graph_score_mod,
    score_mod_bwd=graph_score_mod_bwd,
    aux_tensors=[token_to_chunk_q, token_to_chunk_k, chunk_bias,
                 rare_q, rare_k, rare_w],
)
```

### Tolerance

bf16 forward: `atol=2e-3, rtol=1e-2`

The FA4 kernel uses online softmax with tile-level accumulation, which
introduces O(1e-3) rounding differences vs the naive PyTorch reference.
This is expected and acceptable for bf16.

---

## 5. Backward Comparison

Both paths compute `loss = out.sum()` and backpropagate to get dQ, dK, dV.

### Reference backward

Standard PyTorch autograd through the manual attention computation.

### Test backward

FA4's `score_mod_bwd` is the identity (additive bias has gradient 1.0),
so dQ/dK/dV gradients flow through the standard flash attention backward
with the modified scores.

### Tolerance

bf16 backward: `atol=5e-3, rtol=2e-2`

Backward gradients accumulate more rounding error due to the chain rule
through softmax and the tile-level reduction in FA4's backward kernel.

---

## 6. Modal Integration

The test is designed as a Modal function targeting H200:

```bash
modal run tests/test_fa4_h200_parity.py
# or via pytest on GPU:
CPPMEGA_MODAL_GPU=H200:1 modal run scripts/modal_cppmega_run_tests.py
```

The test file uses `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`
so it runs locally on GPU machines and is skipped on CPU-only CI.

When run via Modal, the GHCR image provides:
- torch 2.13+ cu132
- flash-attn-4 beta23+ (flash_attn.cute)
- transformer_engine 2.13

---

## 7. Failure Modes This Test Catches

1. **Bias scaling mismatch**: If FA4 score_mod accidentally divides by
   softmax_scale (double-scaling), forward outputs diverge by O(scale).

2. **Causal mask disagreement**: If FA4's causal implementation differs
   from the reference mask, outputs diverge at boundary tokens.

3. **Rare-edge overlay bugs**: If token-level edges are dropped or
   double-counted, specific (q,k) positions show bias errors.

4. **Chunk-boundary off-by-one**: If token_to_chunk maps are wrong at
   chunk boundaries, entire token spans get wrong bias.

5. **Backward identity violation**: If score_mod_bwd is not identity,
   gradients diverge from the reference.

6. **GQA head broadcast**: If the bias is not properly broadcast across
   heads (it should be head-independent), per-head outputs differ.

---

## 8. Test Parameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| B | 2 | Exercises per-batch chunk maps |
| S | 128 | Multiple FA4 tiles (tile=64) |
| H | 8 | GQA broadcast check |
| D | 64 | Standard head dim |
| beta | 2.0 | Non-trivial scaling |
| dtype | bf16 | Production dtype |
| causal | True | Production constraint |
| fwd_atol | 2e-3 | bf16 forward tolerance |
| fwd_rtol | 1e-2 | bf16 forward tolerance |
| bwd_atol | 5e-3 | bf16 backward tolerance |
| bwd_rtol | 2e-2 | bf16 backward tolerance |
