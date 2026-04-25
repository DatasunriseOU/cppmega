# SparseMLA Block-Scaled Fused Backend Prototype (2026-04-25)

Worktree:
`/home/dave/source/cppmega-sparse-mla-fused-agent`

Branch:
`agent/sparse-mla-fused-backend`

Base commit:
`43ec47c16dedc2a05d6e019c588bbabcd09cb566`

## What Landed

This branch adds an experimental MXFP8 fused SparseMLA path:

- Public forward wrapper:
  `cppmega.megatron.sparse_mla_ops.sparse_mla_blockscaled_mxfp8_forward`
- Public backward wrapper:
  `cppmega.megatron.sparse_mla_ops.sparse_mla_blockscaled_mxfp8_backward`
- TileLang fused forward and partial backward prototype:
  `cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_fused.py`
- Probe:
  `tools/probes/sparse_mla_blockscaled_fused_probe.py`

The default `SparseMLA_FP8` TE tensorwise path is unchanged. The fused
block-scaled path raises unless:

```bash
CPPMEGA_SPARSE_MLA_BLOCKSCALED_FUSED=1
```

The public backward wrapper is fail-closed. The TileLang backward kernel is
kept in-tree for debugging, but it is not exposed by default because the
current kernel prototype produced non-finite gradients during validation. The
correctness backward path is an explicit BF16-dequant PyTorch reference and
requires:

```bash
CPPMEGA_SPARSE_MLA_BLOCKSCALED_BWD_REFERENCE_ACK=1
```

The unsafe TileLang backward can only be reached for kernel debugging with:

```bash
CPPMEGA_SPARSE_MLA_BLOCKSCALED_TILELANG_BWD_UNSAFE=1
```

## Runtime ABI

MXFP8 forward consumes pre-quantized block-scaled tensors:

```text
q_data:   [B, S,  H, D_total]      torch.float8_e4m3fn
kv_data:  [B, SK, G, D_total]      torch.float8_e4m3fn
q_scale:  [B, S,  H, D_total/32]   torch.float32
kv_scale: [B, SK, G, D_total/32]   torch.float32
indices:  [B, S,  G, topk]         torch.int32, -1 sentinel
output:   [B, S,  H, d_v]          torch.bfloat16
lse:      [B, S,  H]               torch.float32, log2 units like existing TileLang SparseMLA
```

`d_v` is the value prefix length. `tail_dim = D_total - d_v` must currently be
positive and divisible by 16. `D_total` must be divisible by 32.

Forward does not materialize full sparse scores or full BF16 Q/K/V tensors. It:

1. runs one FP8 QK MMA per 32-channel block,
2. applies `q_scale[..., block] * kv_scale[..., block]` to each partial block
   accumulator,
3. updates online softmax state across sparse top-k tiles,
4. dequantizes only the current V tile into shared memory,
5. accumulates `P @ V` and writes BF16 output plus LSE.

NVFP4 and TE compact MXFP8/NVFP4 tensor extraction are not wired into this fused
path. TE `MXFP8Tensor` rowwise data is visible, but its scale buffer is swizzled
and not the same ABI as the probe's `[B,S,H,D/32]` FP32 scale tensor. The code
therefore fails closed instead of guessing a scale layout.

## Backward Status

Implemented:

- Correctness backward wrapper with explicit BF16 reference ACK.
- Internal TileLang backward prototype that recomputes block-scaled QK and
  dequantizes per-tile Q/K/V into shared memory.

Not production-ready:

- The TileLang backward prototype currently compiles but is disabled by default
  because it produced non-finite gradients in validation.
- The ACK-gated reference backward materializes BF16 Q/KV and sparse scores, so
  it is only a correctness/probe tool.
- Gradients are returned with respect to the logical dequantized Q/KV tensors,
  not with respect to FP8 bytes or scale tensors.

Additional 2026-04-25 unsafe-kernel check:

```text
CPPMEGA_SPARSE_MLA_BLOCKSCALED_FUSED=1
CPPMEGA_SPARSE_MLA_BLOCKSCALED_TILELANG_BWD_UNSAFE=1
MCORE_DSA_TILELANG_SEQ_BUCKET=1
MCORE_DSA_TILELANG_TOPK_BUCKET=64
```

Direct small-shape calls still produced non-finite gradients even with no
padding and with `invalid_fraction=0.0`:

```text
H=16: dq NaNs=640,  dkv NaNs=5504
H=28: dq NaNs=896,  dkv NaNs=5504
H=32: dq NaNs=1408, dkv NaNs=5504
H=64: dq NaNs=2303, dkv NaNs=5504
```

Forward output and LSE were finite, and the preprocess kernel was finite in
isolation. The remaining bug is inside the unsafe TileLang backward mainloop
or block-scaled score recompute, not the public wrapper. Do not route training
through this kernel until it has a finite-gradient parity test.

## Validation

Syntax check:

```bash
python -m py_compile \
  cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_fused.py \
  cppmega/megatron/sparse_mla_ops/sparse_mla.py \
  cppmega/megatron/sparse_mla_ops/__init__.py \
  tools/probes/sparse_mla_blockscaled_fused_probe.py
```

Small GB10 probe:

```bash
python tools/probes/sparse_mla_blockscaled_fused_probe.py \
  --batch 1 --seq 2 --seq-kv 64 --heads 28 --kv-group 1 \
  --topk 64 --dim-total 128 --d-v 64 --iters 1 --invalid-fraction 0.05
```

Result:

```text
runtime_fwd_ms=0.5708 ref_fwd_ms=0.5652
runtime_bwd_reference_ms=1.9864 ref_bwd_ms=34.2854
fwd_max_abs=0.000244141
lse_nat_max_abs=4.76837e-07
dq_max_abs=0 dkv_max_abs=0
peak_mb runtime_fwd=2.96 ref_fwd=33.35 runtime_bwd=65.50 ref_bwd=65.39
materialization runtime_forward_full_scores=False runtime_forward_full_bf16_qkv=False
materialization runtime_backward_backend=explicit_bf16_reference_ack
```

DeepSeek-style value tile smoke:

```bash
python tools/probes/sparse_mla_blockscaled_fused_probe.py \
  --batch 1 --seq 1 --seq-kv 64 --heads 28 --kv-group 1 \
  --topk 64 --dim-total 576 --d-v 512 --iters 1 \
  --invalid-fraction 0.05 --fwd-atol 8e-4 --fwd-rtol 1e-2
```

Result:

```text
runtime_fwd_ms=1.3650 ref_fwd_ms=0.6592
runtime_bwd_reference_ms=1.5911 ref_bwd_ms=31.9947
fwd_max_abs=0.00012207
lse_nat_max_abs=4.76837e-07
dq_max_abs=0 dkv_max_abs=0
peak_mb runtime_fwd=18.96 ref_fwd=40.18 runtime_bwd=72.58 ref_bwd=72.24
materialization runtime_forward_full_scores=False runtime_forward_full_bf16_qkv=False
```

The large relative forward errors in the probe are from near-zero reference
elements; absolute error is the useful signal for these small random tensors.

## Hugging Face Kernel Search

No Hugging Face kernel found on 2026-04-25 is a drop-in replacement for this
SparseMLA training backend:

- `kernels-community/flash-mla` / DeepSeek FlashMLA is the closest forward
  reference, but it is sparse MLA forward/decode-oriented, uses a different FP8
  KV cache ABI, and does not provide MXFP8 block-scaled sparse training
  backward.
- FlashAttention-4/FlexAttention is useful as a backward and block-sparse
  scheduling reference, but it is not token-topk SparseMLA and not this
  `q_data/kv_data/q_scale/kv_scale` ABI.
- `finegrained-fp8`, `fp8-fbgemm`, and DeepGEMM-style kernels are useful GEMM
  references, not fused sparse online-softmax/PV backward replacements.
- vLLM/FlashInfer MLA sparse paths are inference/decode paths, not training
  backward.

Conclusion: clone/probe FlashMLA and FA4/FlexAttention for ideas, but keep the
cppmega ABI and implement the MXFP8 SparseMLA backward here.

## Sharding Implications

Sequence sharding:

- Q and `q_scale` shard together over local query sequence `S`.
- `indices` is local to the query shard and must reference positions in the
  local visible KV shard or use a global-to-local remap before launch.
- Online softmax state (`m`, `sumexp`, output accumulator, LSE) is per local
  `[B,S,H]` row. If one logical query row is split across KV shards, each shard
  must return partial `(m, sumexp, acc_o)` and a second reduction must merge
  them with the FlashAttention-style online softmax combine rule.

KV/block sharding:

- `kv_data` and `kv_scale` shard together over `SK`.
- Sparse `indices` must be owned by the same KV shard as the selected block or
  encoded with `-1` for non-local selections.
- Duplicate selected KV positions are accumulated exactly like the existing
  sparse top-k semantics; backward dKV needs atomic accumulation or a reduce
  over repeated indices and over query shards.

Tensor parallelism:

- Query heads are TP-sharded. `q_data` and `q_scale` ownership follows the
  local head shard.
- KV groups are either replicated or group-sharded. If a KV group is sharded,
  `kv_data`, `kv_scale`, and the corresponding `indices[..., group, :]` must
  be colocated.
- Per-block scale indexing is always along the local last dimension:
  `scale_idx = channel // 32`. If TP ever shards the last dimension, producers
  must preserve 32-channel alignment or adjust scale indexing at shard
  boundaries.

Data parallelism:

- DP has no special scale sharing. Each rank owns its local microbatch,
  quantized payloads, scales, indices, output, and LSE.
- Gradients from the correctness backward are BF16 logical gradients and follow
  the same DP reduction path as the existing SparseMLA gradients.

Backward online-softmax stats:

- The fused backward must use the same LSE units as forward. Existing TileLang
  SparseMLA stores LSE in log2 units, so backward reconstructs
  `P = exp2(score * softmax_scale * log2(e) - lse)`.
- For KV-sharded attention, backward cannot reconstruct global probabilities
  from local scores alone unless forward saved or recomputes the globally merged
  LSE. The local partial LSE is insufficient after cross-shard online-softmax
  combine.

## Replacement Status

This cannot replace the current backend yet.

Forward is a real fused MXFP8 block-scaled prototype and avoids the QK score
materialization that the earlier QK-only helper produced. Backward remains
experimental: the public correctness path requires explicit BF16 materialization
ACK, while the partial TileLang backward kernel is disabled until finite-gradient
validation passes.
