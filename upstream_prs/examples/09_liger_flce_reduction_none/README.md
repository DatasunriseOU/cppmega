# Reproducer: `LigerFusedLinearCrossEntropyFunction(reduction="none")` backward is silently wrong

Demonstrates that Liger's fused linear cross-entropy kernel produces
incorrect / NaN gradients during backward when invoked with
`reduction="none"`. Forward completes with a finite tensor, but
gradients diverge from eager PyTorch reference and in Megatron training
surface as NaN `grad_norm` → CUDA illegal-memory-access on the next step.

## Upstream status (as of 2026-04-14)

- **Issue [#968](https://github.com/linkedin/Liger-Kernel/issues/968)** — root bug report, *CLOSED without fix*. Collaborator comment: "This feature is not implemented since it's not a common case."
- **Issue [#872](https://github.com/linkedin/Liger-Kernel/issues/872)** — related bug for the public `LigerFusedLinearCrossEntropyLoss` wrapper, closed.
- **PR [#496](https://github.com/linkedin/Liger-Kernel/pull/496)** — merged 2024-12; added `reduction="none"` to forward, did **not** fix backward.
- **PR [#1126](https://github.com/linkedin/Liger-Kernel/pull/1126)** — OPEN draft (Nick Knight, 2026-03-03): adds an `assert` that blocks the backward path. Prevents silent corruption but no functional fix.
- **PR [#1182](https://github.com/linkedin/Liger-Kernel/pull/1182)** — OPEN: adds `reduction` kwarg plumbing; does not fix the kernel bwd.

No PR currently implements a correct `reduction="none"` backward.

## Run

```bash
pip install -r requirements.txt
python reproducer.py
```

Requires a CUDA device (the Liger Triton kernels are CUDA-only).

## Expected output — bug present (current state, liger-kernel 0.7.0, verified on H200 bench3 2026-04-14)

```
Liger kernel results:
  [OK  ] reduction="mean"                          |max grad_hidden - ref| = 4.768e-07  ...
  [OK  ] reduction="none" + .sum().backward()      |max grad_hidden - ref| = 4.883e-04  ...
  [FAIL] reduction="none" + (loss*mask).sum()      |max grad_hidden - ref| = 4.712e-02  |max grad_weight - ref| = 2.451e-01  ...
  [OK  ] workaround: mean * n_valid                |max grad_hidden - ref| = 4.883e-04  ...

VERDICT: Liger reduction='none' backward is BROKEN.
         Uniform .sum()         path: OK  (coincidentally matches reduction='sum' ref since grad_output=[1,1,…])
         Non-uniform (mask)     path: FAIL  (real bug: element_mul_kernel treats per-token grad_output as scalar)
         Workaround (mean*N)    path: OK  (correct in all cases)
```

Exit code: **1**.

The uniform `.sum().backward()` path passes _by coincidence_: autograd hands back
a `[BT]` tensor of ones, which `element_mul_kernel`'s scalar-broadcast happens
to read correctly as `1.0`. Any non-uniform `grad_output` (the Megatron
loss-mask pattern, document-boundary masking, per-token loss weighting, etc.)
silently reads only `grad_output[0]` and mis-scales every other row.

## Expected output — PR #1126 merged (assertion guard only)

```
  [ASSERT] reduction='none' raised AssertionError: ...

VERDICT: reduction='none' blocked by AssertionError (PR #1126-style guard).
         No functional fix — workaround still required.
```

Exit code: **1**.

## Expected output — functional fix merged

```
  [OK  ] reduction="none"  (BUG)                  |max grad_hidden - ref| = 1.9e-03  ...

VERDICT: Liger reduction='none' backward is CORRECT (bug fixed).
```

Exit code: **0**.

## Workaround

When callers need per-token loss semantics from Liger (Megatron's main
LM head plumbing returns `[b, s]` even when downstream consumers only
need the scalar), wrap the call:

```python
loss_scalar, *_ = LigerFusedLinearCrossEntropyFunction.apply(
    hidden, weight, target, None, None, ignore_index,
    0.0, 0.0, "mean",  # <-- NEVER "none"
    None, False,
)
# Recover sum semantics: sum_i CE_i == mean_i CE_i * n_valid. Backward
# through `mean_loss * n_valid` cancels Liger's internal 1/n_valid
# mean-reduction factor exactly, giving the same gradient as
# F.cross_entropy(reduction="sum").
n_valid = (target != ignore_index).sum().clamp_min(1).to(loss_scalar.dtype)
loss_sum = loss_scalar * n_valid

# If caller wants a per-token [b, s] tensor whose .sum() equals loss_sum
# (e.g. because Megatron writes loss * loss_mask and then sums), broadcast
# a uniform scalar: expand().contiguous(). Per-token values will all be
# equal (mean), but the reduction is exact when loss_mask == (target != ignore_index).
per_token = loss_scalar.expand(b, s).contiguous()
```

Used in production in
`cppmega/megatron/apply_linear_ce_patch.py::_install_liger_compute`.

## Root cause (for reviewers)

`fused_linear_cross_entropy_forward` always normalizes grad_input inside
the CE kernel by `n_non_ignore` (the denominator for reduction="mean"),
independent of the user-requested `reduction`. When the user selects
`reduction="none"`, forward correctly returns a per-token loss vector,
but the grad_input / grad_weight accumulated during forward were computed
as if the caller would later divide by `n_non_ignore`. The backward
(`element_mul_kernel`) then multiplies by `grad_output`, which for
`reduction="none"` is a per-token tensor rather than a scalar — the
kernel reads the first element only, producing a silent miscalculation
that in practice becomes NaN once the chunked grad_weight accumulation
overflows.

A correct `reduction="none"` path would need to either:
1. Pass `reduction="none"` semantics (no 1/N normalization) into
   `liger_cross_entropy_kernel` at grad-logits-store time, **and**
2. Scatter the per-token `grad_output` into the chunked grad_weight /
   grad_bias accumulators instead of scaling by a scalar after the fact.

Both break the current forward-time chunked grad_weight precomputation,
which is why the kernel was left with a silent-TODO comment ("Not
supporting reduction='none' now").
