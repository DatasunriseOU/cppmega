# MXFP8 Wave 1 Agent B GB10 Variant Search - 2026-04-29

## Result

No MXFP8 variant in this wave beat the BF16/tensorwise GB10 baseline.  The best
measured no-BF16-bridge variant is the TE TN adapter backward path, so the local
GB10 typed profile now defaults MXFP8 backward to `te_tn_adapter` instead of
`flashinfer_cutlass`.

## Measurements

Baseline BF16/tensorwise:

- Log: `/home/dave/logs/gb10_quarter_post_agents_smoke_20260429.log`
- Hot steps 3-5: ~4886.8 ms/step
- Max allocated: 27.667 GiB

Previous safe MXFP8 default (`flashinfer_cutlass`, grouped direct off):

- Log: `/home/dave/logs/mxfp8_safe_torch_why_20260429.log`
- Torch profile: `/home/dave/logs/mxfp8_safe_torch_why_20260429_torch_profile`
- Hot steps 3-8: ~5761.8 ms/step
- Max allocated: 25.849 GiB
- Counters: `mxfp8_flashinfer_dgrad=272`,
  `mxfp8_flashinfer_wgrad=272`, `bf16_fallback_dgrad=0`,
  `bf16_fallback_wgrad=0`, `mxfp8_tn_adapter_copy_transpose=4112`
- Torch profile step 2: FlashInfer/CUTLASS MXFP8 kernel bucket was 878.3 ms
  across 68 calls; `aten::copy_` 470.7 ms, `aten::clone` 403.4 ms,
  `aten::contiguous` 392.2 ms

Best measured MXFP8 variant (`te_tn_adapter`, grouped direct off):

- Log: `/home/dave/logs/wave1b_mxfp8_te_tn_5step_20260429.log`
- Hot steps 3-5: ~5295.7 ms/step
- Max allocated: 25.832 GiB
- Counters: `mxfp8_tn_adapter_dgrad=220`,
  `mxfp8_tn_adapter_wgrad=220`, `mxfp8_flashinfer_dgrad=0`,
  `mxfp8_flashinfer_wgrad=0`, `bf16_fallback_dgrad=0`,
  `bf16_fallback_wgrad=0`, `mxfp8_tn_adapter_copy_transpose=1370`
- Delta vs previous safe MXFP8: ~466 ms/step faster, ~8.1%
- Delta vs BF16/tensorwise: ~409 ms/step slower, ~8.4%

Default-profile verification after changing `local_gb10_quarter`:

- Log: `/home/dave/logs/wave1b_mxfp8_te_default_8step_20260429.log`
- Hot steps 4-8: ~5299.4 ms/step; steps 3-8: ~5367.1 ms/step
- Max allocated: 25.830 GiB
- Counters: `mxfp8_tn_adapter_dgrad=352`,
  `mxfp8_tn_adapter_wgrad=352`, `mxfp8_flashinfer_dgrad=0`,
  `mxfp8_flashinfer_wgrad=0`, `bf16_fallback_dgrad=0`,
  `bf16_fallback_wgrad=0`, `mxfp8_tn_adapter_copy_transpose=2192`
- Delta vs BF16/tensorwise on steps 4-8: ~407 ms/step slower, ~8.3%
- Delta vs previous safe MXFP8 on steady steps: ~462 ms/step faster, ~8.0%

Grouped direct compact kernels from Darwin's worktree remain unsuitable for the
full GB10 profile in this wave: the scalar grouped direct route measured
~39.3 s/step in `/home/dave/logs/bisect_mxfp8_current_fixed_5step_20260429.log`.

## FlashInfer Tactic Probes

The two representative FlashInfer/CUTLASS SM120 MXFP8 probes found no faster
direct tactic:

- `/home/dave/logs/wave1b_probe_flashinfer_mxfp8_m16384_n16512_k3584_20260429.json`
  measured `mm_mxfp8=38.55 ms`, `direct_tactic=38.47 ms`; tactics 1 and 2
  failed CUTLASS initialization.
- `/home/dave/logs/wave1b_probe_flashinfer_mxfp8_m16512_n3584_k16384_20260429.json`
  measured `mm_mxfp8=31.87 ms`, `direct_tactic=31.86 ms`; tactics 1 and 2
  failed CUTLASS initialization.

## External References Checked

- NVIDIA Transformer Engine MXFP8 user guide:
  <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html>
- CUTLASS changelog and SM120 MXFP8 examples:
  <https://docs.nvidia.com/cutlass/4.4.0/CHANGELOG.html>
  and
  <https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu>
- FlashInfer GEMM docs and local SM120 MXFP8 CUTLASS binding:
  <https://docs.flashinfer.ai/generated/flashinfer.gemm.gemm_fp8_nt_groupwise.html>
  and `/home/dave/flashinfer/csrc/mxfp8_gemm_cutlass_sm120.cu`

## Next Target

The remaining gap is not FlashInfer tactic selection.  The largest actionable
target is still eliminating rowwise-transpose copy paths, especially grouped
fallback copies, or replacing the scalar compact/grouped direct kernels with a
real SM120 tiled mainloop.
