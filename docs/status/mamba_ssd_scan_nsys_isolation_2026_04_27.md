# Mamba SSD scan isolation - 2026-04-27

Scope: Worker 8/8, Mamba scan kernels visible in `cuda_gpu_kern_sum`:
`_chunk_scan_*`, `_state_passing_*`, `_chunk_state_*`, and related Mamba2/Mamba3
scan paths.

## Profile Source

Valid local GB10 nsys profile:

```text
/home/dave/logs/gb10_nsys_full_fixed2_20260427_173123_nsys.sqlite
```

The older fork profile at
`/home/dave/logs/gb10_quarter_mbs4_nsys_fork_20260425_111137_nsys.sqlite`
does not expose the scan kernels clearly enough for this isolation pass.

## Kernel Attribution

The current local GB10 quarter noconv stack uses Mamba2 SSD Triton kernels, not
the Author Mamba3 TileLang MIMO kernels:

- `cppmega/megatron/nam56r_noconv_spec.py`: routes M layers to
  `NoConvMamba3BCMixer`; R layers still route to `CppMegaM2RNNMixer`.
- `cppmega/megatron/noconv_mamba_mixer.py`: calls
  `mamba_chunk_scan_combined` in the no-conv and Mamba3-BC paths.
- `mamba_ssm/ops/triton/ssd_combined.py`: dispatches `_chunk_cumsum_fwd`,
  `_chunk_state_fwd`, `_state_passing_fwd`, `_bmm_chunk_fwd`, `_chunk_scan_fwd`
  and their backward kernels.

The Author Mamba3 paths remain separate:

- `cppmega/megatron/mamba3_te_mixer.py`: `mamba3_siso_combined` or
  `mamba3_mimo_combined`; MIMO caps `chunk_size` by `chunk_size * rank <= 64`.
- `cppmega/megatron/cppmega_mamba3_tp_mixer.py`: TileLang MIMO autograd path.
- `cppmega/megatron/tilelang_mimo_autograd.py`: recomputes DA prefix sums in
  backward instead of saving three additional tensors.

## Raw vs Steady State

Raw scan-family kernel summary from the full profile is polluted by Triton
autotune launches in the first 30 seconds. That is visible in the per-second
scan totals: 8-29s are dense, while later steady windows are sparse.

Post-autotune approximation used:

```sql
with base as (select min(start) t0 from CUPTI_ACTIVITY_KIND_KERNEL)
select s.value as name, count(*) calls,
       round(sum(k.end-k.start)/1e6,3) total_ms,
       round(avg(k.end-k.start)/1e6,3) avg_ms
from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id
where (s.value like '_chunk_%'
   or s.value like '_state_passing_%'
   or s.value like '_bmm_chunk_%')
  and (k.start-(select t0 from base)) > 40000000000
group by s.value order by total_ms desc;
```

Top post-40s steady scan costs:

| Kernel | Calls | Total ms | Avg ms |
| --- | ---: | ---: | ---: |
| `_state_passing_fwd_kernel` | 26 | 101.721 | 3.912 |
| `_chunk_state_fwd_kernel` | 26 | 99.714 | 3.835 |
| `_chunk_scan_fwd_kernel` | 15 | 72.170 | 4.811 |
| `_chunk_scan_chunk_state_bwd_dx_kernel` | 11 | 71.910 | 6.537 |
| `_state_passing_bwd_kernel` | 11 | 71.146 | 6.468 |
| `_chunk_scan_bwd_dstates_kernel` | 11 | 41.706 | 3.791 |
| `_chunk_scan_bwd_dc_kernel` | 11 | 31.558 | 2.869 |
| `_chunk_state_bwd_db_kernel` | 11 | 31.097 | 2.827 |

Conclusion: after removing autotune noise, the best target is not cumsum or BMM.
It is the main SSD scan/state handoff: state passing, chunk state forward, scan
forward, and backward dx/state passing.

## Chunk Size Microbench

Focused single-layer SSD scan fwd+bwd microbench, exact local-quarter M-shape:

```text
device=NVIDIA GB10
shape=B4 L4096 H112 P64 G8 N128 dtype=torch.bfloat16
```

Results:

| Chunk size | Cold fwd+bwd ms | Warm median fwd+bwd ms | Peak alloc GiB |
| ---: | ---: | ---: | ---: |
| 64 | 44197.355 | 87.611 | 3.75 |
| 128 | 15530.184 | 63.571 | 3.75 |
| 256 | 17489.984 | 56.995 | 3.75 |

Small-shape parity check, `chunk_size=128` vs `256`, passed finite output and
grad checks. Output max absolute delta was `1.52588e-05`; all checked gradient
max absolute deltas were <= `7.62939e-06`.

## Patch

The local GB10 typed profile now sets the no-conv SSD scan chunk size to 256.
This is not a global Mamba default change; H200/default profiles remain unset
and keep the existing mixer default unless they opt in.

Changed surfaces:

- `cppmega/recipes/run_profiles.py`: adds typed
  `RuntimePatchProfile.noconv_mamba_chunk_size`, validates it as a positive
  power of two, and sets local GB10 quarter to 256.
- `scripts/cppmega_fp8_shim.py`: bridges `CPPMEGA_NOCONV_MAMBA_CHUNK_SIZE` into
  `TransformerConfig.cppmega_noconv_mamba_chunk_size`.
- `cppmega/megatron/nam56r_noconv_spec.py`: passes the optional config value to
  `NoConvMamba3BCMixer(chunk_size=...)` for M layers only. R layers stay on
  `CppMegaM2RNNMixer`.

## Remaining

The cheapest next profile experiment is a short delayed nsys capture after the
Triton autotune window, with the patched profile:

```bash
RUN_PROFILE=local_gb10_quarter \
  scripts/local_gb10_quarter_train.sh \
  --train-iters 5 --nsys-profile --nsys-capture-mode delay \
  --nsys-delay 40 --nsys-duration 20
```

Do not use the raw full-process `cuda_gpu_kern_sum` alone for scan ranking; keep
the post-warmup SQLite filter or delayed capture, otherwise Triton autotune
variants dominate the scan totals.
