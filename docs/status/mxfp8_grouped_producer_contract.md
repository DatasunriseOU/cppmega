# MXFP8 Grouped Producer Contract

This note tracks the grouped/MoE MXFP8 producer route used by the GB10
performance lanes. It is intentionally a stable status document rather than a
date-stamped session log.

## Runtime Knob

The route is selected through `PrecisionProfile.mxfp8_grouped_quantize_producer`
and rendered as `CPPMEGA_TE_MXFP8_GROUPED_QUANTIZE_PRODUCER`.

- `single_output`: compatibility route. It wraps
  `transformer_engine_torch.split_quantize` and may emit per-output
  rowwise-transposed operands.
- `multi_output`: strict acceptance route. It requires
  `transformer_engine_torch.mxfp8_split_quantize_with_rowwise_transpose` and
  fails fast if the symbol is missing.

The `multi_output` route calls the TE extension as:

```python
mxfp8_split_quantize_with_rowwise_transpose(
    tensor,
    split_sections,
    quantizers,
    disable_bulk_allocation,
    transpose_scales_with_gemm_swizzled,
)
```

Each returned MXFP8 output must either already be a GEMM-ready rowwise-transpose
operand or carry `_te_gemm_ready_rowwise_transpose_for_backward`. A profile that
selects `multi_output` but returns no GEMM-ready operands is invalid.

## Counters

Acceptance counters for this producer are:

- `mxfp8_grouped_quantize_producer_multi_output > 0`
- `mxfp8_grouped_quantize_producer_multi_output_consumed > 0`
- `mxfp8_grouped_quantize_producer_multi_output_missing_api == 0`
- `mxfp8_grouped_quantize_producer_single_output == 0`

This producer route is still not the final TE Linear/autograd memory contract.
The final target is to save only GEMM-ready MXFP8 operands for backward, with no
BF16 saved activation and no persistent transpose sidecar registry.
