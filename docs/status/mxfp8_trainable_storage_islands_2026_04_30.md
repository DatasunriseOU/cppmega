# Trainable MXFP8 Storage Islands

Status: Wave13A tested increment
Last updated: 2026-04-30
Scope: remove persistent BF16 model-parameter storage from selected non-TE
islands while keeping them trainable.

## Sources Checked

- NVIDIA Transformer Engine MXFP8 user guide:
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html
- NVIDIA Transformer Engine PyTorch API guide:
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html
- Local TE source:
  `/home/dave/TransformerEngine/transformer_engine/pytorch/tensor/mxfp8_tensor.py`
  and
  `/home/dave/TransformerEngine/transformer_engine/pytorch/quantized_tensor.py`
- Local Megatron no-master optimizer path:
  `/home/dave/megatron-lm/megatron/core/optimizer/optimizer.py`

TE MXFP8 tensors require the last dimension and the product of earlier
dimensions to be divisible by 32. Rowwise and columnwise MXFP8 payloads are
independent; this path stores rowwise payloads only by default because the
embedding and LinearCE routes consume dequantized rowwise values, not GEMM
columnwise operands.

## Implementation

The new typed mode is `trainable_mxfp8`; `frozen_mxfp8` is rejected at both the
profile layer and env parser. The selected parameters are quantized with
`MXFP8Quantizer(rowwise=True, columnwise=False)` and re-wrapped as
`torch.nn.Parameter(q_tensor, requires_grad=current.requires_grad)`, preserving
Megatron embedding/output parameter tags.

Selected paths:

- `embedding.word_embeddings.weight`
- `output_layer.weight`
- `embedding.cppmega_ngram_hash.unified_table.weight`
- `embedding.cppmega_ngram_hash.out_proj.weight`
- `embedding.cppmega_structure.stacked_emb.weight`
- `embedding.cppmega_structure.up_proj.weight`

Shape padding is deliberately conservative:

- Row padding is allowed for `nn.Embedding` tables only.
- Output-layer row padding is refused because it would change logits/classes.
- Column padding is allowed only for the ngram hash table; the ngram module
  slices the padded feature dimension back to `embed_dim`.

LinearCE/CrossEntropy paths dequantize an MXFP8 output weight transiently before
calling CCE/Liger fallback kernels. That preserves autograd to the trainable
MXFP8 parameter, but it is still a materialized forward bridge rather than a
native MXFP8 CE kernel.

## Default Local NAM-Shape Accounting

Assumptions: local NAM56R-quarter shape with `V=65536`, `H=3584`, ngram defaults
`orders=(2,3)`, `num_heads=8`, `embed_dim=16`, default ngram prime rows
`7,998,862`, and structure core `25 x 64`.

| Parameter | BF16 bytes | MXFP8 bytes | Delta |
| --- | ---: | ---: | ---: |
| `embedding.word_embeddings.weight` | 469,762,048 | 242,221,056 | -227,540,992 |
| `output_layer.weight` | 469,762,048 | 242,221,056 | -227,540,992 |
| `embedding.cppmega_ngram_hash.unified_table.weight` | 255,963,584 | 287,960,064 | +31,996,480 |
| `embedding.cppmega_ngram_hash.out_proj.weight` | 1,835,008 | 946,176 | -888,832 |
| `embedding.cppmega_structure.stacked_emb.weight` | 3,200 | 2,560 | -640 |
| `embedding.cppmega_structure.up_proj.weight` | 458,752 | 243,712 | -215,040 |
| Total selected | 1,197,784,640 | 773,594,624 | -424,190,016 |

Total selected storage moves from 1.115524 GiB BF16 to 0.720466 GiB MXFP8,
saving 0.395058 GiB. The ngram table grows by 30.514 MiB because its logical
16-wide feature dimension is padded to MXFP8's 32-wide block requirement.

## BF16 Storage That Can Remain

For selected parameters that fail shape/device/import checks, the runtime logs
`MXFP8 trainable storage islands BF16 remains` with per-parameter MiB. Expected
default NAM shapes above convert all selected 2D islands.

Known unconverted storage:

- `embedding.cppmega_structure.component_scales`, if present, is a tiny 1D BF16
  scalar vector and is not part of this 2D MXFP8 path.
- Transient CE dequantization materializes a dense tensor for the forward call;
  this is not persistent model-parameter storage.

This path does not enable BF16 MXFP8 dense-backward fallbacks. Existing
`bf16_fallback_dgrad` and `bf16_fallback_wgrad` counters remain owned by the
TE Linear MXFP8 backend and must stay zero in full training receipts.

## Validation

CPU/unit:

```text
CUDA_VISIBLE_DEVICES= PYTHONPATH=. /home/dave/cppmega-venv/bin/python -m pytest -q \
  tests/test_mxfp8_storage_islands.py -k 'not cuda_conversion' \
  tests/test_run_profiles.py tests/test_ngram_hash.py

37 passed, 1 deselected, 19 warnings
```

CUDA microprobe under `flock /tmp/cppmega_gpu_profile.lock`:

```text
torch 2.13.0.dev20260417+cu132 cuda True (12, 1)
embedding grad <class 'torch.Tensor'> torch.bfloat16 torch.Size([64, 32]) True True
changed 0.416015625 p type <class 'transformer_engine.pytorch.tensor.mxfp8_tensor.MXFP8Tensor'> data type <class 'transformer_engine.pytorch.tensor.mxfp8_tensor.MXFP8Tensor'> storage bytes 5120
```

CUDA unit under `flock /tmp/cppmega_gpu_profile.lock`:

```text
PYTHONPATH=. /home/dave/cppmega-venv/bin/python -m pytest -q \
  tests/test_mxfp8_storage_islands.py::test_storage_island_cuda_conversion_keeps_parameters_trainable

1 passed, 19 warnings
```

## Acceptance

This is not acceptable for main as a final training solution yet. It is a small
tested increment that proves trainable TE MXFP8 parameters can replace selected
non-TE BF16 model storage and survive local autograd plus an optimizer step.
Main acceptance still needs a 1-2 step full-model training receipt with these
islands enabled, zero BF16 fallback counters, and no hidden BF16 master/storage
regression from the scalar optimizer path.
