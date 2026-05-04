# MXFP8 Grouped Quantize Producer Contract

This note records the acceptance contract for the grouped/MoE MXFP8 split
producer. The runtime knob is owned by the typed `RunProfile` dataclass as
`PrecisionProfile.mxfp8_grouped_quantize_producer`, and launchers render it as
`CPPMEGA_TE_MXFP8_GROUPED_QUANTIZE_PRODUCER`.

Valid values:

- `single_output`: wrap `transformer_engine_torch.split_quantize` and attach one
  GEMM-ready rowwise-transpose operand per split.
- `multi_output`: require
  `transformer_engine_torch.mxfp8_split_quantize_with_rowwise_transpose` and use
  that fused producer for all split outputs. This mode must fail fast if the TE
  symbol is missing, because falling back to `single_output` hides producer
  launch and materialization overhead.

Acceptance counters for `multi_output`:

- `mxfp8_grouped_quantize_producer_multi_output > 0`
- `mxfp8_grouped_quantize_producer_multi_output_consumed > 0`
- `mxfp8_grouped_quantize_producer_multi_output_missing_api = 0`
- `mxfp8_grouped_quantize_producer_single_output = 0`

The required TE symbol must return a list/tuple of MXFP8 outputs. Each MXFP8
output must either already be marked as a rowwise-transpose backward operand or
carry `_te_gemm_ready_rowwise_transpose_for_backward` pointing to the GEMM-ready
MXFP8 operand.
