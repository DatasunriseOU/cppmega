# SparseMLA Block-Scaled QK Runtime Probe (2026-04-25)

Worktree:
`/home/dave/source/cppmega-sparse-blockscaled-agent-runtime`

Base commit:
`0eca74c5f57d30e08c8ae4a5112ec5b04d491401` ("Gate deprecated training
paths")

## What Landed

This branch adds an env-gated, QK-only experimental path:

- Public wrapper:
  `cppmega.megatron.sparse_mla_ops.sparse_mla_blockscaled_qk_scores`
- TileLang implementation:
  `cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_qk.py`
- Probe:
  `tools/probes/sparse_mla_blockscaled_qk_probe.py`

The default SparseMLA runtime is unchanged.  `SparseMLA_FP8` still uses the
existing TE current/tensorwise FP8 path.  The new helper raises unless:

```bash
CPPMEGA_SPARSE_MLA_BLOCKSCALED_QK=1
```

## Runtime ABI

The helper consumes already-quantized Q/K payloads and block scales:

MXFP8:

```text
q_data:  [B, S,  H, D]      torch.float8_e4m3fn
kv_data: [B, SK, G, D]      torch.float8_e4m3fn
q_scale: [B, S,  H, D/32]   torch.float32
kv_scale:[B, SK, G, D/32]   torch.float32
indices: [B, S,  G, topk]   torch.int32, -1 sentinel
output:  [B, S,  H, topk]   torch.float32 QK logits
```

NVFP4:

```text
q_data:  [B, S,  H, D/2]    packed torch.uint8
kv_data: [B, SK, G, D/2]    packed torch.uint8
q_scale: [B, S,  H, D/16]   torch.float32
kv_scale:[B, SK, G, D/16]   torch.float32
indices: [B, S,  G, topk]   torch.int32, -1 sentinel
output:  [B, S,  H, topk]   torch.float32 QK logits
```

The MXFP8 TileLang kernel applies block scales inside the K-block loop:

```text
for each K block:
    partial = dot(q_block, k_block)
    score += partial * q_block_scale * k_block_scale
```

That is the operation missing from the current tensorwise SparseMLA FP8
kernel, which can only apply one Q scale and one KV scale after the full dot.
The NVFP4 path is correctness-oriented: it decodes packed E2M1 nibbles in the
kernel and applies the same block-scale accumulation, but it is not using a
native NVFP4 tensor-core MMA path.

## Validation

Syntax/import checks:

```bash
python -m py_compile \
  cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_qk.py \
  tools/probes/sparse_mla_blockscaled_qk_probe.py \
  cppmega/megatron/sparse_mla_ops/sparse_mla.py \
  cppmega/megatron/sparse_mla_ops/__init__.py
```

Existing TE tensorwise extraction tests:

```bash
PYTHONPATH=/home/dave/source/cppmega-sparse-blockscaled-agent-runtime \
pytest -q tests/test_sparse_mla_te_fp8_extract.py
```

Result:

```text
4 passed, 21 warnings in 0.56s
```

Block-scaled QK probe results:

| Command shape | Format | Runtime QK | BF16-dequant reference | Max abs | Max rel |
| --- | --- | ---: | ---: | ---: | ---: |
| `B=1,S=2,SK=64,H=28,G=1,topk=64,D=576` | MXFP8 | 0.4433 ms | 0.6979 ms | 0.000423506 | 0.003814 |
| `B=1,S=2,SK=64,H=28,G=1,topk=64,D=576` | NVFP4 | 1.3533 ms | 3.8248 ms | 0.00048542 | 0.00390625 |
| `B=1,S=2,SK=32,H=32,G=2,topk=64,D=128` | MXFP8 | 0.0053 ms | 0.3296 ms | 0.000919342 | 0.0038438 |
| `B=1,S=2,SK=32,H=32,G=2,topk=64,D=128` | NVFP4 | 0.6943 ms | 1.0726 ms | 0.00094223 | 0.00384615 |

Commands:

```bash
python tools/probes/sparse_mla_blockscaled_qk_probe.py \
  --format mxfp8 --batch 1 --seq 2 --seq-kv 64 \
  --heads 28 --kv-group 1 --topk 64 --dim 576 --iters 2

python tools/probes/sparse_mla_blockscaled_qk_probe.py \
  --format nvfp4 --batch 1 --seq 2 --seq-kv 64 \
  --heads 28 --kv-group 1 --topk 64 --dim 576 --iters 2

python tools/probes/sparse_mla_blockscaled_qk_probe.py \
  --format mxfp8 --batch 1 --seq 2 --seq-kv 32 \
  --heads 32 --kv-group 2 --topk 64 --dim 128 --iters 2

python tools/probes/sparse_mla_blockscaled_qk_probe.py \
  --format nvfp4 --batch 1 --seq 2 --seq-kv 32 \
  --heads 32 --kv-group 2 --topk 64 --dim 128 --iters 2
```

The probe materializes BF16 Q/K only in the validation reference.  The runtime
helper takes quantized payloads plus scale tensors and never dequantizes a full
Q/K tensor.

## Runtime Blockers

This is mergeable as an experimental QK probe, but it is not a complete
SparseMLA block-scaled backend yet.

1. Full SparseMLA forward still needs online softmax and `P @ V` wired after
   the block-scaled QK accumulator.  The current helper returns QK logits only.
2. Backward is not implemented for block-scaled QK.  The existing
   `SparseMLA_FP8.backward` remains tensorwise FP8.
3. The wrapper does not yet extract TE `MXFP8Tensor` or `NVFP4Tensor` payloads
   directly.  The probe uses quack quantization to provide the data/scale ABI.
4. NVFP4 is not a performant runtime path in TileLang here.  It scalar-decodes
   packed nibbles because the installed TileLang path does not expose native
   Blackwell NVFP4 MMA for this sparse attention shape.
5. The value path needs a producer/ABI decision.  QK block scales are per
   K-block, while `P @ V` either needs a compatible block-scaled V path or a
   deliberate BF16 V tile dequant strategy.
