# Mamba3 Wave32 Lane B: bwd_bwd Vectorized Redesign

Date: 2026-04-30
Branch: `worker/mamba3-wave32-bwd-bwd-vectorized-redesign`
Scope: H100-only component compile/perf for a larger bwd_bwd redesign after the Wave31 scalar shared-diag no-go.

## Candidate

The final candidate is a guarded source experiment layered on current stage2:

- keep stage2 force-nonTMA layout (`bf=1, bb=0`);
- remove full `dqk_from_diag_shared [fused_chunk_size, fused_chunk_size]`;
- compute same-step `R x R` diagonal blocks with a per-step vectorized `R*R x P` product plus `T.reduce_sum`;
- stage only `[chunk, R * R]` `dqk_diag_shared` for downstream `DGAMMA_DIAG`, `DK`, and `DQ`.

This avoids repeating Wave31's scalar `for p in T.serial(P)` shared-diag candidate.

## Iterations

| run | app | result |
| --- | --- | --- |
| `r1` | `ap-N1WsGS2gi8RYcv5YZ0LCLh` | full-GEMM diag extraction did not compile: TileLang layout could not prove divisibility for `dqk_from_diag_frag[cs*R+r_out, cs*R+r_in]`. |
| `r2` | `ap-NytZO57Dgu3pVNgmzsp9qy` | Modal CLI stopped before result. |
| `r3` | `ap-cT63ehxpCVOhWlqoZFvDOr` | tiny `R x R` per-step `T.gemm` did not compile: `M must be divisible by 16, but got 4`. |
| `r4` | `ap-4Y0I0k63mSKNZ8wvrGzplB` | padded `16 x 16` per-step `T.gemm` did not compile: `m_warp * n_warp must equal num_warps`, got one warp for an 8-warp kernel. |
| `r5` | `ap-axeRIlYGQV0azAHcRErraM` | vectorized reduce candidate compiled, correct within small fp noise, but lost perf. |
| `r6` | `ap-wSoJlF0t6kUuInFVNuwAWS` | profiler/NVTX run stopped before `report.json`; progress reached `bench_start`. |
| verify | `ap-2CefZpInZMWRd98JJiOTag` | guarded mutation + rollback passed; restored original source bytes. |

All apps above are stopped with 0 tasks.

## H100 r5 Metrics

GPU: `NVIDIA H100 80GB HBM3`, CUDA `13.2`, torch `2.13.0.dev20260426+cu132`.

Smoke shape: `B=1 S=128 H=4 G=1 N=64 P=64 R=4 chunk=16`

| variant | bwd_fwd | bwd_bwd | chain | peak alloc delta | max_abs vs stage2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `0.03756 ms` | `0.08149 ms` | `0.47040 ms` | `1.2109 MiB` | - |
| stage2_current | `0.03790 ms` | `0.08181 ms` | `0.46379 ms` | `1.2109 MiB` | `0.0` |
| wave32_vectorized_diag | `0.03818 ms` | `0.09281 ms` | `0.43089 ms` | `1.2109 MiB` | `1.82e-6` |

Representative shape: `B=2 S=512 H=8 G=1 N=64 P=64 R=4 chunk=16`

| variant | bwd_fwd | bwd_bwd | chain | peak alloc delta | max_abs vs stage2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `0.13335 ms` | `0.31692 ms` | `1.11903 ms` | `33.0405 MiB` | - |
| stage2_current | `0.13643 ms` | `0.31268 ms` | `2.26296 ms` | `33.0405 MiB` | `0.0` |
| wave32_vectorized_diag | `0.13642 ms` | `0.36228 ms` | `0.74126 ms` | `33.0405 MiB` | `2.57e-6` |

The split-kernel metric is the stable signal here: candidate `bwd_bwd` is `+15.9%` slower than stage2 current on the representative shape. Chain timings vary across repeated compile/cache paths and are not used for the decision.

## Commands

```bash
python -m py_compile cppmega/megatron/upstream_patches/apply_mamba3_bwd_bwd_vectorized_patches.py scripts/modal_mamba3_wave32_lane_b_h100.py
pytest -q tests/test_mamba3_bwd_bwd_vectorized_applier.py tests/test_mamba3_stage2_force_nontma_applier.py
git diff --check
modal run scripts/modal_mamba3_wave32_lane_b_h100.py::main --run-id wave32_lane_b_h100_vectorized_20260430_r5
modal run scripts/modal_mamba3_wave32_lane_b_h100.py::main --run-id wave32_lane_b_h100_vectorized_profile_20260430_r6 --profile-nvtx --cuda-profile --profile-target wave32_vectorized_diag
modal run scripts/modal_mamba3_wave32_lane_b_h100.py::main --run-id wave32_lane_b_h100_vectorized_verify_applier_20260430 --verify-applier
modal volume get cppmega-mamba3-benchmarks /benchmarks/mamba3_wave32_lane_b_h100/wave32_lane_b_h100_vectorized_20260430_r5 artifacts/mamba3_wave32_lane_b_h100/
```

## Judgment

Do not move this kernel change to production main. It is correct, guarded, and useful as a rejected candidate record, but it does not reduce H100 `bwd_bwd`; it trades the full shared tile for a reduce microkernel and loses about 16% on the representative split timing. The only safe-for-main value is the default-off applier, harness, tests, and no-go artifact/doc.
