| case | variant | production throughput | status | steps seen | tok/sec | avg step ms | peak alloc GiB | peak reserved GiB | reached Mamba bwd | log |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| te_flash_full | baseline | yes | crash | 0 |  |  | 16.983 | 17.107 | no | /vol/benchmarks/mamba3_wave30_modal_h200_attn_debug/wave30_h200_attn_debug_20260430/te_flash_full_baseline.log |
| fallback_auto_full | baseline | no | crash | 0 |  |  | 37.946 | 38.400 | yes | /vol/benchmarks/mamba3_wave30_modal_h200_attn_debug/wave30_h200_attn_debug_20260430/fallback_auto_full_baseline.log |
| fallback_unfused_no_flash_full | baseline | no | crash | 0 |  |  | 48.881 | 49.590 | yes | /vol/benchmarks/mamba3_wave30_modal_h200_attn_debug/wave30_h200_attn_debug_20260430/fallback_unfused_no_flash_full_baseline.log |
| fallback_auto_no_flash_seq2048 | baseline | no | crash | 0 |  |  | 29.642 | 29.883 | yes | /vol/benchmarks/mamba3_wave30_modal_h200_attn_debug/wave30_h200_attn_debug_20260430/fallback_auto_no_flash_seq2048_baseline.log |
