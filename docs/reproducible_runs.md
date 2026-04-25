# Reproducible Runs

Single-command launchers for each verified production config. Each wraps
`scripts/remote_smoke_h200_dsa_9_4_m.sh` with the correct env so that a
fresh checkout of the repo produces the measurement reported in
[production_status.md](production_status.md).

## Dispatcher

```bash
cd /mnt/data/cppmega-root/cppmega     # bench3 path; europe = /home/dave/cppmega-root/cppmega
bash scripts/launch.sh bench3-fp8     # or: europe-bf16, bench3-smoke, gb10
```

The dispatcher `scripts/launch.sh` is a thin `case` over the four presets
below.  Each preset is also executable on its own
(`bash scripts/run_<preset>.sh`) and accepts env-var overrides (e.g.
`TRAIN_ITERS=30 bash scripts/run_bench3_golden_fp8.sh`).

All scripts call `"$(dirname "$0")/..."` so they work from any cwd, as long
as `REMOTE_ROOT` is set correctly (autodetected for bench3/europe, explicit
for GB10).

---

## 1. `bench3-fp8` — bench3 golden FP8 (production)

**Script**: `scripts/run_bench3_golden_fp8.sh`

| Item              | Value                                                                |
| ----------------- | -------------------------------------------------------------------- |
| Model             | NAM56R (52 hybrid Mamba3-MIMO + 9 DSA + 4 full-attn, MoE 16 experts) |
| Hardware          | 8x H200 SXM, LOCATION_1 (`h200_1`)                                   |
| Precision         | FP8 tensorwise (`--fp8-format hybrid`)                               |
| Topology          | TP=1 PP=1 EP=8 DP=1, MBS=10 GBS=80, seq=4096, MTP=2                  |
| Target throughput | **268 TFLOP/s ± 0.5** (27.1% MFU)                                    |
| Peak memory       | ~115 GiB / 141 GiB per rank (`CG_FLAGS=NONE` mandatory)              |
| Iters             | 15 (override via `TRAIN_ITERS`)                                      |

**Active patches** — see README.md "Always On" + "Env-Gated":
- Always on: MTP Liger CE, Mamba LinearCE class-swap, DSA indexer fused per-head bmm, Mamba3 regional torch.compile
- Enabled here: IndexCache, lemyx fused DSA, main-head Liger CE via `--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear`

**Login + run**:
```bash
gcloud compute ssh dave@h200_1 --zone=LOCATION_1
cd /mnt/data/cppmega-root/cppmega
bash scripts/launch.sh bench3-fp8
```

**Logs**: `${REMOTE_ROOT}/cppmega/cppmega_nam56r_dsa_9_4_fp8_m_v3.log`

**Example output** (abbreviated, realistic numbers):
```
 iteration        4/      15 | ...
 | lm loss: 9.972E+00 | grad norm: 4.821 | throughput per GPU (TFLOP/s/GPU): 266.3
 iteration        5/      15 | ...
 | lm loss: 9.947E+00 | grad norm: 4.714 | throughput per GPU (TFLOP/s/GPU): 268.1
 iteration        6/      15 | ...
 | lm loss: 9.921E+00 | grad norm: 4.603 | throughput per GPU (TFLOP/s/GPU): 268.4
...
[stream_m_peak_mem] rank=0 peak_alloc_gib=114.2 peak_reserved_gib=129.8
```

**Reading the output**:
- Iters 1-2 are cold (TileLang JIT + CUDA graph capture), ignore
- Steady state iters 4-15 should converge to 265-269 TFLOP/s
- `grad norm` finite (< 10) and `lm loss ~ ln(vocab_size) ≈ 10` at step 1
- `peak_alloc_gib < 135` is healthy; > 140 approaches the OOM cliff

---

## 2. `europe-bf16` — europe baseline BF16

**Script**: `scripts/run_europe_baseline_bf16.sh`

| Item              | Value                                              |
| ----------------- | -------------------------------------------------- |
| Model             | NAM56R (identical to bench3 variant)               |
| Hardware          | 8x H200 SXM, LOCATION_2 (`h200_1`)                 |
| Precision         | BF16 (FP8 regresses -34% on this fabric)           |
| Topology          | TP=1 PP=1 EP=4 DP=2, MBS=8 GBS=64, seq=4096, MTP=2 |
| Target throughput | **289 TFLOP/s** (29.2% MFU) — europe gold record   |
| Peak memory       | ~127 GiB / 141 GiB per rank (MBS=10 OOMs)          |
| Iters             | 15 (override via `TRAIN_ITERS`)                    |

**Active patches**: same unconditional set as bench3; same opt-in gates
(IndexCache, lemyx, Liger main-head CE). No FP8 flags.

**Login + run**:
```bash
gcloud compute ssh dave@h200_1 --zone=LOCATION_2
cd /home/dave/cppmega-root/cppmega
bash scripts/launch.sh europe-bf16
```

**Logs**: `${REMOTE_ROOT}/cppmega/cppmega_nam56r_dsa_9_4_fp8_m_v1.log`

**Example output**:
```
 iteration        4/      15 | ...
 | lm loss: 9.968E+00 | grad norm: 4.795 | throughput per GPU (TFLOP/s/GPU): 287.6
 iteration        5/      15 | ...
 | lm loss: 9.943E+00 | grad norm: 4.688 | throughput per GPU (TFLOP/s/GPU): 289.1
 iteration        6/      15 | ...
 | lm loss: 9.917E+00 | grad norm: 4.577 | throughput per GPU (TFLOP/s/GPU): 289.4
...
[stream_m_peak_mem] rank=0 peak_alloc_gib=126.8 peak_reserved_gib=135.2
```

---

## 3. `bench3-smoke` — 7-iter smoke test

**Script**: `scripts/run_bench3_smoke_quick.sh`

Use this to verify a fresh checkout produces sane TFLOP/s before committing
to a full training run. Identical config to `bench3-fp8` except
`TRAIN_ITERS=7`, so it finishes in ~90 seconds including model load and
TileLang JIT compile.

| Item        | Value                                     |
| ----------- | ----------------------------------------- |
| Hardware    | same as `bench3-fp8` (8x H200, bench3)    |
| Iters       | 7                                         |
| Target      | TFLOP/s converging to 260-268 by iter 4-7 |
| Peak memory | ~115 GiB (same as golden)                 |

**Login + run**:
```bash
gcloud compute ssh dave@h200_1 --zone=LOCATION_1
cd /mnt/data/cppmega-root/cppmega
bash scripts/launch.sh bench3-smoke
```

**Example output**:
```
 iteration        3/       7 | throughput per GPU (TFLOP/s/GPU): 254.1
 iteration        4/       7 | throughput per GPU (TFLOP/s/GPU): 263.7
 iteration        5/       7 | throughput per GPU (TFLOP/s/GPU): 266.9
 iteration        6/       7 | throughput per GPU (TFLOP/s/GPU): 267.8
 iteration        7/       7 | throughput per GPU (TFLOP/s/GPU): 268.2
[stream_m_peak_mem] rank=0 peak_alloc_gib=114.6 peak_reserved_gib=130.1
```

**Fail fast signals**:
- `grad_norm = nan` or `inf` → Liger #968 regression. Check that
  `reduction="mean"` broadcast path is active in
  `cppmega/megatron/apply_linear_ce_patch.py`, and confirm
  `CPPMEGA_MTP_NATIVE_HOPPER_CE` is NOT set.
- OOM at iter 1 → verify `CG_FLAGS=NONE` is actually exported (the CUDA
  Graph private pool is 63.5 GiB — `CG_FLAGS=NONE` is mandatory at MBS=10).
- TFLOP/s < 200 steady state → stack drift; diff venv vs
  `docs/reference_bench3_h200_stack.md`.

---

## 4. `gb10` — GB10 single-GPU correctness test

**Script**: `scripts/run_gb10_correctness.sh`

A correctness check, **not** a throughput run: verifies TileLang kernels
compile under the sm_121 99 KiB smem cap (no tcgen05, no WGMMA, no TMEM)
and produce finite grads end-to-end.

| Item      | Value                                                                                      |
| --------- | ------------------------------------------------------------------------------------------ |
| Model     | NAM56R, EP=1 (no MoE routing; 1 expert/rank)                                               |
| Hardware  | 1x NVIDIA GB10 (Grace+Blackwell, sm_121, 128 GB unified)                                   |
| Precision | BF16 (FP8 is a dead path on GB10 — 0.73-0.91x, see `reference_fp8_mamba_ssm_dead_path.md`) |
| Topology  | TP=1 PP=1 EP=1 DP=1, MBS=1 GBS=1, seq=2048, MTP=0                                          |
| Target    | all iters complete, `grad_norm` finite, preflight smem check passes                        |
| Iters     | 5 (override via `TRAIN_ITERS`)                                                             |

**Preflight smem check**: enable with `CPPMEGA_SMEM_CHECK=1` (default in
the script). Promotes to hard fail with `CPPMEGA_SMEM_CHECK_STRICT=1`. See
`cppmega/megatron/preflight_smem_check.py` and
`reference_gb10_bwd_bwd_blocker.md`.

**Login + run**:
```bash
ssh gb10
cd /home/dave/cppmega
bash scripts/launch.sh gb10
```

**Logs**: `${REMOTE_ROOT}/cppmega/cppmega_nam56r_dsa_9_4_fp8_m_v0.log`

**Example output**:
```
[preflight_smem_check] audited 34 TileLang kernels, max smem = 94720 B (under 99 KiB cap)
 iteration        1/       5 | lm loss: 1.013E+01 | grad norm: 5.842
 iteration        2/       5 | lm loss: 1.012E+01 | grad norm: 5.797
 iteration        3/       5 | lm loss: 1.011E+01 | grad norm: 5.703
 iteration        4/       5 | lm loss: 1.010E+01 | grad norm: 5.612
 iteration        5/       5 | lm loss: 1.009E+01 | grad norm: 5.524
[stream_m_peak_mem] rank=0 peak_alloc_gib=18.7 peak_reserved_gib=22.4
```

**Pass criteria**: 5 iters complete, no TileLang compile errors, smem
preflight passes, all `grad_norm` finite.

**Known-not-to-run paths on GB10**:
- FP8 Mamba SSM (dead path)
- FA4 (no Blackwell tcgen05 on GB10)
- TileLang bwd_bwd kernels with 3D+ TMA smem descriptors (use -arch=sm_120f
  over sm_121a for the 9x-better fallback)

---

## Common troubleshooting

- **Venv missing NVIDIA libs**: `scripts/remote_smoke_h200_dsa_9_4_m.sh`
  auto-prepends `${REMOTE_VENV}/lib/python3.13/site-packages/nvidia/*/lib`
  to `LD_LIBRARY_PATH`. If you see `libcudnn.so.9 not found`, confirm the
  venv has `nvidia-cudnn-cu13` installed.
- **Dataset missing**: the underlying launcher asserts the rendered
  command contains neither `NullTokenizer` nor `--mock-data`. Prep data
  via `scripts/data_prep_parquet_to_megatron.py` first.
- **Megatron rebase**: after any Megatron update run
  `python -m cppmega.megatron.upstream_patches.apply_dsa_cg_patches` before
  any of these scripts; see `feedback_mandatory_patches.md`.

## References

- [production_status.md](production_status.md) — canonical throughput numbers
- [../README.md](../README.md) — "Always On" and "Env-Gated" patch tables
- `reference_bench3_h200_stack.md` — bench3 stack fingerprint
- `reference_env_drift_bench3_europe.md` — cross-machine divergence audit
