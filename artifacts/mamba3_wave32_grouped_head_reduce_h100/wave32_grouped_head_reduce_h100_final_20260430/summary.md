# Wave32 Grouped-Head Reduce H100

- run_id: `wave32_grouped_head_reduce_h100_final_20260430`
- gpu: `NVIDIA H100 80GB HBM3`
- warmup/iters: `20/100`

| shape | torch ms | triton ms | speedup | peak torch MiB | peak triton MiB | max_abs dq/dk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_hpg2 | 0.024692 | 4.062719 | 0.006 | 8.00 | 8.00 | 0/0 |
| half_seq_hpg16 | 0.207740 | 4.106624 | 0.051 | 64.00 | 64.00 | 0/0 |
| fullish_seq4096_hpg16 | 0.399202 | 4.192784 | 0.095 | 128.00 | 128.00 | 3.57628e-07/1.19209e-07 |
