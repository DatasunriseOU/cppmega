Status: active
Canonical: none
Date: 2026-04-27
Scope: Reproducible BF16 vs MXFP8 GB10 A/B log summarization and profiler handling.

# BF16 vs MXFP8 A/B Harness

Use the parser when comparing local GB10 BF16 and MXFP8 runs:

```bash
python tools/profiling/compare_bf16_mxfp8.py \
  --bf16-log /home/dave/logs/ab_bf16_100_mem_20260427_014727.log \
  --mxfp8-log /home/dave/logs/ab_mxfp8_100_mem_20260427_015833.log \
  --bf16-extra-log /home/dave/logs/ab_bf16_100_torch_20260427_023522.log \
  --mxfp8-extra-log /home/dave/logs/ab_mxfp8_100_torch_20260427_024650.log \
  --bf16-extra-log /home/dave/logs/ab_bf16_100_nsys_clean_20260427_033900.log \
  --mxfp8-extra-log /home/dave/logs/ab_mxfp8_100_nsys_20260427_022234.log \
  --bf16-extra-log /home/dave/logs/ab_bf16_100_ncu_range_20260427_031020_ncu.log \
  --mxfp8-extra-log /home/dave/logs/ab_mxfp8_100_ncu_range_20260427_032120_ncu.log
```

The primary logs should be the memory/perf runs. Extra logs are scanned for
torch profiler, nsys, and ncu artifacts without mixing profiler timing into the
hot-step average. Use `--format json` for machine-readable output.

Profiler constraints:

- Do not combine torch profiler and nsys in one run. CUPTI permits only one
  active subscriber; the combined BF16 run showed
  `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`.
- NCU should capture the Megatron `cudaProfilerStart/Stop` range. Enable it via
  typed run-profile CLI fields such as `--cuda-profile`,
  `--cuda-profile-step-start`, and `--cuda-profile-step-end`.

Current parse result from the logs above:

```text
metric                                   bf16            mxfp8           mxfp8-bf16
hot_step_avg_ms                      5922.101         6020.436  +98.335 ms (+1.66%)
tok_per_sec                            2766.7           2721.5 -45.146 tok/s (-1.63%)
setup_alloc_gib                         3.821            5.279 +1.458 GiB (+38.16%)
max_alloc_gib                          27.249           28.413  +1.164 GiB (+4.27%)
param_bytes_gib                         3.807            3.891  +0.084 GiB (+2.21%)
final_train_loss                     1.570702         1.667334      +0.097 (+6.15%)
final_val_loss                       1.813475         1.889000      +0.076 (+4.16%)
final_test_loss                      1.893944         1.981903      +0.088 (+4.64%)
skipped_iterations                          0                0
nan_iterations                              0                0
```

Parameter storage in the same parse:

```text
BF16:  torch.bfloat16 = 3.807 GiB
MXFP8: MXFP8Tensor    = 2.759 GiB
       torch.bfloat16 = 1.132 GiB
```

Conclusion from these existing logs: MXFP8 is slower by 1.6%, allocates more at
setup and peak, stores 2.759 GiB in MXFP8 plus 1.132 GiB still in BF16, and ends
with worse train/validation/test losses over the 100-step run.

## 2026-04-27 TE Emit Sidecar Cleanup Smoke

Post-patch smoke:

```bash
RUN_ID=postpatch_mxfp8_te_10_mem_20260427_111254 \
scripts/local_gb10_quarter_train.sh \
  --fp8-recipe mxfp8 \
  --mxfp8-bwd-backend te_tn_adapter \
  --mxfp8-transpose-emit-backend te \
  --train-iters 10 \
  --mem-profile \
  --mem-profile-steps 2
```

Result on real clang 4k data:

```text
hot_step_avg_ms(iter 3-10)       5991.363
tok_per_sec                      2734.6
setup_alloc_gib                     3.900
step_2_max_alloc_gib               28.430
final_train_lm_loss                 4.435116
final_val_lm_loss                   3.958559
final_test_lm_loss                  4.043954
skipped_iterations                  0
nan_iterations                      0
```

The saved-activation sidecar leak is fixed in this run:

```text
bf16_fallback_dgrad/wgrad                 0 / 0
native_passthrough_dgrad/wgrad            0 / 0
mxfp8_tn_adapter_dgrad/wgrad            440 / 440
mxfp8_tn_adapter_te_emit               5506
mxfp8_tn_adapter_saved_transpose_operand 120
mxfp8_tn_adapter_copy_transpose          194
mxfp8_tn_sidecar_registry_size             0
mxfp8_tn_sidecar_registry_current_bytes    0
mxfp8_tn_sidecar_tracked_attr_current_bytes 0
```

The remaining `copy_transpose=194` is not activation storage. It is the current
bridge for primary MXFP8 parameters: those parameters do not pass through a BF16
quantizer, so TE cannot emit a rowwise-transposed sidecar from a BF16 source.
Until the native SM120/CUTLASS loader reads compact columnwise MXFP8 parameter
payloads directly, dgrad uses a one-shot compact payload transpose for those
operands.
