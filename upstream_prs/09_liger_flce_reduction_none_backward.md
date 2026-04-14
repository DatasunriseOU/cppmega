# Issue: `LigerFusedLinearCrossEntropyFunction(reduction="none")` backward silently corrupts gradients

**Target repo:** `linkedin/Liger-Kernel`

**Relates to:** issue [#968](https://github.com/linkedin/Liger-Kernel/issues/968) (CLOSED, not fixed), issue [#872](https://github.com/linkedin/Liger-Kernel/issues/872), PR [#496](https://github.com/linkedin/Liger-Kernel/pull/496) (merged, forward-only), PR [#1126](https://github.com/linkedin/Liger-Kernel/pull/1126) (OPEN draft: adds assertion), PR [#1182](https://github.com/linkedin/Liger-Kernel/pull/1182) (OPEN: reduction plumbing).

## Summary

`LigerFusedLinearCrossEntropyFunction.apply(..., reduction="none")` returns a
forward loss tensor of shape `[BT]` that looks reasonable, but the saved
`grad_input` / `grad_weight` are multiplied in backward by `element_mul_kernel`,
which assumes `grad_output` is a scalar and reads only `grad_output[0]` — so
any non-uniform per-token `grad_output` (loss-mask weighting, document-boundary
masking, per-token loss scaling, etc.) silently produces the wrong gradient
for every row except the first.

Two distinct call patterns:

1. `loss.sum().backward()` — `grad_output = [1, 1, …, 1]`. Scalar-read of the
   first element reads `1.0`, math coincidentally agrees with
   `reduction="sum"`. **Passes silently.**
2. `(loss * loss_mask).sum().backward()` — `grad_output = loss_mask`, which is
   non-uniform. Scalar-read returns `loss_mask[0]` and scales every row by
   that one value. **Silently wrong** — reproducer shows `max|Δgrad_hidden|
   = 4.7e-2`, `max|Δgrad_weight| = 2.5e-1` vs eager PyTorch (bf16 noise floor
   is `5e-3`).

Downstream effect in a Megatron LM-head integration (NAM56R, MBS=10–12 bf16),
where loss-mask is applied before reduction: first iteration reports
`grad_norm = NaN`, the optimizer step poisons the weights, iteration 2
crashes with CUDA illegal memory access.

## Problem

`src/liger_kernel/ops/fused_linear_cross_entropy.py`:

1. `fused_linear_cross_entropy_forward` precomputes `grad_input` and
   `grad_weight` during the forward chunked loop, before `grad_output` is
   known. For `reduction="none"` it correctly skips the final `1/N` scaling
   (kernel receives `reduction="none"`, so per-token grad_logits are stored
   unscaled), and the forward loss tensor `[BT]` is correct.

2. `fused_linear_cross_entropy_backward` then calls `element_mul_kernel`
   to scale those saved accumulators by the backward `grad_output`:

   ```python
   def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
       if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
           element_mul_kernel[(n_rows,)](
               grad_input, grad_input.stride(-2),
               grad_output,   # <-- assumed scalar; kernel does `grad_input[i,j] *= *grad_output`
               H, ...,
           )
   ```

   `element_mul_kernel` is a scalar-broadcast kernel: it loads a single value
   from `grad_output` and multiplies every row of `grad_input` / `grad_weight`
   by it. Works fine when the caller passes a scalar
   (`reduction="mean"|"sum"`) or a uniform tensor of ones
   (`reduction="none"` + plain `.sum().backward()`). Silently wrong for any
   non-uniform per-token `grad_output` (mask-weighted reductions,
   document-boundary masking, per-token loss scaling, MoE load-balance
   weighted losses, …).

3. The `torch.equal(grad_output, tensor(1.0))` fast-path check additionally
   compares a `[BT]` tensor against a scalar — `torch.equal` short-circuits
   to `False` in that case, so the bad scalar-multiply runs anyway.

## Reproducer

Single-file, self-contained (torch + liger-kernel, no Megatron):

[`examples/09_liger_flce_reduction_none/reproducer.py`](examples/09_liger_flce_reduction_none/reproducer.py)

Shape mirrors one NAM56R microbatch: `B=2 S=512 H=1024 V=32000` bf16, ~10% of
tokens set to `ignore_index`. Compares three paths against eager
`F.linear + F.cross_entropy`:

| path                                       | expected                       | observed (liger-kernel 0.7.0 on H200 sm_90a, bench3)             |
| ------------------------------------------ | ------------------------------ | ---------------------------------------------------------------- |
| `reduction="mean"`                         | `max\|Δ\| < 5e-3` (bf16 noise) | **OK** (`4.8e-7 / 3.8e-6`)                                       |
| `reduction="none"` + `.sum().backward()`   | match `reduction="sum"` ref    | **OK** (`4.9e-4 / 2.0e-3`) — *coincidental*: grad_output=[1,1,…] |
| `reduction="none"` + `(loss*mask).sum()`   | match masked eager ref         | **FAIL** (`4.7e-2 / 2.5e-1`) — the real bug                      |
| workaround: `reduction="mean"` × `n_valid` | match `reduction="sum"` ref    | **OK** (`4.9e-4 / 3.9e-3`)                                       |

Run:
```bash
cd examples/09_liger_flce_reduction_none
pip install -r requirements.txt
python reproducer.py
```

Exit code `0` when fixed, `1` when bug present (current state).

## Expected vs. actual behavior

**Expected** (mirrors `F.cross_entropy(reduction="none")`): per-token forward
tensor `[BT]`, and on `loss.sum().backward()` the weight/input gradients match
`F.cross_entropy(reduction="sum")` bit-exactly (modulo bf16 matmul rounding).

**Actual** (current main, 0.7.0): forward tensor is finite and numerically
correct; backward produces `grad_input` off by an O(1) factor and `grad_weight`
containing NaN once the chunked forward accumulator sees the garbage `grad_output`
broadcast. PR #1126 (draft) blocks this via assertion but adds no functional path.

## Suggested fix

Two options, roughly ranked by effort:

### (a) Ship PR #1126 (assertion) as a safety guard — loudly fail, don't corrupt

Merge [#1126](https://github.com/linkedin/Liger-Kernel/pull/1126) so users get a
clear `AssertionError: reduction='none' backward not supported` instead of silent
NaN. This is the minimal viable short-term fix and already implemented; the
draft just needs to land. Our reproducer detects and reports the assertion path
separately from the functional path.

### (b) Implement a functional per-token-`grad_output` backward

Replace the scalar `element_mul_kernel` with a row-broadcast kernel when
`grad_output` is a tensor:

```python
def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    if grad_output.dim() > 0:
        # Per-token grad_output (reduction="none" path).
        BT = grad_input.shape[0]
        assert grad_output.shape == (BT,), (
            f"grad_output must be [BT] for reduction='none', got {grad_output.shape}"
        )
        # grad_input[i, :] *= grad_output[i]
        grad_input.mul_(grad_output.unsqueeze(-1).to(grad_input.dtype))
        # grad_weight was precomputed as sum_i grad_logits_i[:, None] * input_i[:, None, :],
        # i.e. `grad_logits.t() @ input`. To scale each i-term by grad_output[i],
        # we need to redo the chunked MM in bwd — OR scale grad_logits stored
        # per-chunk during forward then recombine. Easiest correct fix:
        # recompute grad_weight in bwd using saved grad_logits chunks and
        # scaled input chunks.
        ...
    elif not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # Scalar grad_output path (mean/sum) — unchanged.
        element_mul_kernel[(n_rows,)](grad_input, ..., grad_output, ...)
```

The grad_weight side needs forward to also save `grad_logits_chunk` tensors
(or recompute them from `_input @ weight.T` + softmax in bwd) so backward can
do `grad_weight = sum_chunks (grad_logits_chunk * grad_output_chunk[:, None]).t() @ _input_chunk`.
This costs ~1 extra chunked matmul in bwd vs current code, but preserves the
forward memory footprint.

### (c) Document the limitation

Add a prominent note in the docstring of `LigerFusedLinearCrossEntropyFunction`
and `LigerFusedLinearCrossEntropyLoss` that `reduction="none"` is
forward-only; any backward pass silently corrupts gradients. Today this is
buried in a commented-out block of the forward source.

## Our workaround

Used in production in
[`cppmega/megatron/apply_linear_ce_patch.py`](https://github.com/DatasunriseOU/cppmega/blob/main/cppmega/megatron/apply_linear_ce_patch.py)
(`_install_liger_compute`): always call Liger with `reduction="mean"`, scale
the scalar output by `n_valid` (or broadcast to `[b, s]` for caller APIs that
expect per-token tensors). The reproducer verifies this yields bit-exact
agreement with `F.cross_entropy(reduction="sum")` on the same shape.

The broadcast path is only exact when the caller's `loss_mask` equals
`(labels != ignore_index)` — true for Megatron pretraining. For general
masked-loss callers (mask non-constant within non-ignored tokens), the
scalar-broadcast is a uniform-loss approximation, not a bit-exact reduction.

## Environment

- PyTorch 2.12 nightly + cu132 (bench3 H200 SXM, europe H200 SXM)
- liger-kernel 0.7.0 (latest PyPI as of 2026-04-14)
- Triton 3.2+
- NVIDIA H200 SXM (sm_90a); expected to reproduce on any CUDA device
- Also reproduces on GB10 (sm_121f), bf16

## Impact on us

NAM56R main-head LM Liger fusion blocks. Without this workaround:

- MBS=10 training: `grad_norm` NaN on iter 1, CUDA IMA on iter 2.
- MBS=12 training: backward crashes with "nan in gradient" before step 1 completes.

With the workaround in place, bench3 MBS=10 + Liger main-head hits 269 TFLOP/s
(new record vs. the previous 267 TFLOP/s golden config) — see
`reference_main_head_liger_ce_gap.md` in our tree.
