# DSA `_compute_index_scores` memory reproducer

Self-contained, no-Megatron-required reproducer for the memory waste in
`megatron/core/transformer/experimental_attention_variant/dsa.py`
`_compute_index_scores` (upstream `main` @ 2026-04-14, line 255–295).

See: `upstream_prs/12_megatron_dsa_compute_index_scores_memory.md` for the
full PR template and problem writeup.

The cppmega production patch that ships the fused variant in-tree lives at:
`cppmega/megatron/dsa_indexer_fused_patch.py` — this reproducer extracts
just the math, with no Megatron import.

## Running

```bash
pip install -r requirements.txt
python reproducer.py               # autodetect: CUDA + small + prod (+ full if >=40 GiB)
python reproducer.py --cpu         # CPU-only, small shape only
python reproducer.py --shapes prod # specific shape
```

Exits non-zero on correctness failure.

## What it does

Implements two mathematically equivalent variants:

- **upstream**: verbatim copy of Megatron's `_compute_index_scores` body
  (`torch.einsum('sbhd,tbd->sbht', ...)` → relu → weight → sum → T).
- **fused**: per-head `bmm` into a `[b, sq, sk]` FP32 buffer, never
  materialising the `[sq, b, h, sk]` intermediate.

For each shape it reports:

- Expected `[sq, b, h, sk]` intermediate size (what upstream pays for).
- Peak `torch.cuda.max_memory_allocated` during each variant.
- Max relative error between the two outputs (`< 1e-4` to pass).

It also runs a `torch.autograd.gradcheck` (float64, small shape) on both
variants and verifies bit-identical forward/backward parity in double
precision.

## Expected output

On H200/H100 at shape `prod` (b=4, sq=sk=4096, h=8, d=128):

```
  expected upstream [sq,b,h,sk] fp32 intermediate: 2048.0 MiB
  expected output   [b,sq,sk]   fp32:              256.0 MiB
  upstream         peak_alloc=~2400 MiB (delta vs inputs +~2100 MiB)
  fused            peak_alloc= ~330 MiB (delta vs inputs +~260 MiB)
  correctness: max rel_err = ~2e-7
  memory:      upstream ~2400 MiB -> fused ~330 MiB   (saved ~2070 MiB, ~7x)
  PASS: correctness within 1e-4
```

At shape `full` (NAM56R MBS=8 exactly, b=8 sq=sk=4096 h=32 d=128) the
upstream intermediate is 16 GiB; fused stays under 600 MiB. Requires at
least 40 GiB HBM to run (H100/H200/B200/GB200).

`gradcheck` in float64 at small shape should show `rel_err < 1e-10` for
both forward and all three backward paths (dq, dw, dk), confirming full
autograd parity — the fused variant is not just a forward replacement,
it is a drop-in `_compute_index_scores` that backprops correctly if
upstream ever adds autograd wrapping.

## Mapping to production patch

`cppmega/megatron/dsa_indexer_fused_patch.py::compute_index_scores_fused_bf16`
is the function this reproducer's `fused_compute_index_scores` extracts.
The production patch adds:

- Idempotent monkey-patch application via `apply_dsa_indexer_fused_patch`.
- Env-var kill-switch: `CPPMEGA_DSA_INDEXER_FUSED=0` to restore upstream.
- `**kwargs` forward-compat for upstream adding new args (e.g. PR #3674's
  `mask=`).

The math in both is identical.
