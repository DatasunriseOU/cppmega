# 03 — SparseMLA FP8 dispatch reproducer

Validates the bug described in
`upstream_prs/03_sparse_mla_fp8_dispatch.md`:
upstream Megatron-LM `_fused_sparse_mla_absorbed` in
`megatron/core/transformer/experimental_attention_variant/dsa.py` does not
detect TE `QuantizedTensor` inputs, so when FP8 training is enabled
(`--fp8-format hybrid --fp8-recipe tensorwise`) the SparseMLA kernel
receives a `Float8Tensor` whose `.data_ptr()` is NULL and whose `.dtype`
falsely reports `bfloat16`.

The template proposes two fixes, both validated here:

1. Simple fallback: call `tensor.dequantize()` before SparseMLA.
2. Preferred fix: dispatch `QuantizedTensor` inputs to `SparseMLA_FP8`
   (cppmega's FP8-aware variant, applied today via Patch 9 in
   `cppmega/megatron/upstream_patches/apply_dsa_cg_patches.py`).

## Hardware / software

- Requires a GPU with FP8 support: H100 / H200 / B200 (sm_90+), Ada
  sm_89, or GB10 sm_121a.
- Verified on h200_1 (H200, sm_90, torch 2.12 cu132,
  transformer-engine 2.13.0, cppmega editable install).
- See `requirements.txt` for the exact pinned stack.

## Running

On bench3:

```bash
scp reproducer.py h200_1:/tmp/reproducer_03.py
ssh h200_1 \
  'cd /mnt/data/cppmega-root/cppmega && \
   CUDA_VISIBLE_DEVICES=0 /mnt/data/venv/bin/python /tmp/reproducer_03.py'
```

Locally (if you have H100+ and the cppmega editable install):

```bash
cd /path/to/cppmega
CUDA_VISIBLE_DEVICES=0 python upstream_prs/examples/03_sparse_mla_fp8_dispatch/reproducer.py
```

## Expected output

```
Device: NVIDIA H200 sm_90

=== Setup: building SparseMLA inputs (B=1 Sq=Sk=128 H=8 D=576 topk=64) ===
  q: (1, 128, 8, 576) torch.bfloat16
  kv: (1, 128, 1, 576) torch.bfloat16
  idx: (1, 128, 1, 64) torch.int32  (-1 sentinels in tail)
  softmax_scale: 0.0417  d_v: 512

=== Scenario A: upstream raw-dispatch with Float8Tensor (expect BUG) ===
  Float8Tensor.data_ptr() = q:0  kv:0  (0 == NULL)
  Float8Tensor.dtype reports: torch.bfloat16 (lies about storage)
  Float8Tensor._data.dtype  = torch.uint8 (real storage)
  dispatch hazard: looks-like-bf16=True actually-fp8=True
                   => isinstance(QuantizedTensor) check is MANDATORY
  BF16 kernel on Float8Tensor -> finite=True
                   (silent auto-dequant; lost FP8 speedup)
  [BUG_REPRODUCED] Float8Tensor hazards confirmed: data_ptr=NULL,
                   dtype=bf16, no-unwrap on .contiguous/.to

=== Scenario B: dequantize() preprocess fix (expect FIX_VALIDATED) ===
  max|fix_out - bf16_ref| = 2.441e-03  (tol=5e-02)
  [FIX_VALIDATED] dequantize() preprocess path

=== Scenario C: SparseMLA_FP8 dispatch fix (expect FIX_VALIDATED) ===
  max|fp8_out - bf16_ref|  = 3.174e-03  (tol=1e-01)
  [FIX_VALIDATED] SparseMLA_FP8 dispatch path

========================================================================
SUMMARY:
  A_bug_reproduced         -> PASS
  B_dequant_fix            -> PASS
  C_fp8_dispatch           -> PASS
VERDICT: bug reproduced AND both fix paths validated.
```

Exit codes: `0` if all three scenarios behave as expected, `1` if any
scenario fails, `2` if CUDA or FP8 hardware is unavailable.

## Notes on behavior differences across TE versions

The template describes the failure mode as a hard
`RuntimeError: kernel main input Q data pointer expected non-NULL`. With
transformer-engine 2.13 this specific crash does not fire for Python-level
kernel calls because TE installs a `__torch_dispatch__` hook that silently
dequantizes `Float8Tensor` inputs at the torch.Tensor boundary.

The dispatch bug the template addresses is still real, just in a different
mode: without the explicit `isinstance(query, QuantizedTensor)` check, a
user who asked for FP8 silently gets BF16-kernel execution on an
auto-dequantized tensor — paying 2x memory bandwidth and losing the FP8
speedup the config promised. Scenario A reports that silent-auto-dequant
outcome alongside the underlying hazards (NULL `data_ptr()`, lying
`.dtype`, `.contiguous()` / `.to()` not unwrapping) so the motivation for
the patch is visible even on newer TE.

## Relation to applied cppmega patches

- `cppmega/megatron/sparse_mla_ops/sparse_mla.py` already contains
  `_unwrap_quantized` and the `SparseMLA_FP8` autograd function.
- `cppmega/megatron/upstream_patches/apply_dsa_cg_patches.py` Patch 9
  injects the `isinstance(query, QuantizedTensor)` dispatch into upstream
  `dsa.py` at runtime. Patch 9b removes a stray `.dequantize()` round
  trip that a previous drift introduced.
- This reproducer exercises both fix paths against a pure-BF16
  SparseMLA reference computed on the same numeric inputs.
