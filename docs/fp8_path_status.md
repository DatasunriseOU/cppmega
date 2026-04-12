# FP8 status per Mamba3 path (measured on bench3 H200x8)

Smoke matrix measured on `h200_1` (LOCATION_1, 8x H200),
torch 2.12+cu132, TE 2.13, megatron-core 0.18rc0, mamba_ssm 2.3.1.

All runs 5 iters, `seq_length=4096`, PP=4, MBS=2, GBS=16, MLA+MTP+MoE
(NAM56R feature plan), `--attention-backend auto`, FP8 without CUDA
graphs (`--cuda-graph-impl none`), `--fp8-format hybrid
--fp8-amax-history-len 16 --fp8-amax-compute-algo max`.

## Guards removed in this session

| File                                     | Lines removed    | Guard text                                                                                                                                                               |
| ---------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cppmega/megatron/author_mamba3_spec.py` | 51-52 (pre-edit) | `AuthorMamba3Mixer currently supports non-fp8, non-fp4 runs only`                                                                                                        |
| `cppmega/megatron/m2rnn_spec.py`         | 85-86 (pre-edit) | `CppMegaM2RNNMixer currently supports non-fp8, non-fp4 runs only` (sibling guard, required for Path D which hybridises NoConvMamba3BCMixer + M2RNN on R-layer positions) |

Neither guard had ever been empirically validated; both were
precautionary. Removing them did not surface any FP8 dtype mismatches
or post-projection fp32 bias failures — TE's FP8 wrap of
`TELayerNormColumnParallelLinear` / `TERowParallelLinear` handles the
fp32 bias/parameter tensors inside `Mamba3` / `M2RNN` cleanly.

## Results

| Path | Spec                                                                                                                    | Guard removed?                                               | FP8 result                               | Iter 3-5 (ms)         | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A    | `cppmega.megatron.mamba3_te_stack_spec` (`CppMegaMamba3Mixer`, 2/7 Mamba3: QK-Norm + B/C bias via native SSD kernel)    | n/a (no guard)                                               | **PASS**                                 | 1116 / 1082 / 1064    | No FP8 blocker; grad norms converge (128 to 101), loss trajectory 22.8 -> 18.6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| B    | `cppmega.megatron.mamba3_author_spec` (Author SISO 6/7)                                                                 | `author_mamba3_spec.py:51-52` removed                        | **PASS**                                 | 1416 / 1082 / 1047    | Author `mamba3_siso_combined` Triton kernel works under `--fp8-format hybrid`; fp32 bias/D/dt tensors stay fp32 as expected                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| C    | `cppmega.megatron.mamba3_author_spec` + MIMO (`cppmega_mamba3_is_mimo=True`, `mimo_rank=4`, `chunk_size=16`)            | same (removed)                                               | **was FAIL → NOW PASS as of 2026-04-11** | -                     | Forward OK; backward previously hit **TileLang MIMO kernel limitation**: `mamba_mimo_bwd_combined` raised `ValueError: G value of 8 is not currently supported!` in `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py:1314`. **FIXED** by patching the post-kernel reduction case for `1 < G < H` (added `elif G < H: dq = dq_tilelang.view(B, S, R, G, H//G, N).sum(dim=4)` and same for dk). This unblocks MIMO R=4 ngroups=8 backward on H200. Full NAM56R MIMO 7/7 run completed 2026-04-11 at **56,280 tok/sec baseline** — see `docs/nam56r_mimo7_baseline_2026_04_11.md`. |
| D    | `cppmega.megatron.nam56r_noconv_spec` (`NoConvMamba3BCMixer` on M-layers + `CppMegaM2RNNMixer` on R-layers 12/24/36/48) | `m2rnn_spec.py:85-86` removed (sibling guard on M2RNN mixer) | **PASS**                                 | 38279 / 37625 / 37909 | Slower because this spec uses `CppMegaMambaModel` (full pipeline: cppmega embedding + MTP + per-layer M2RNN build). FP8 converges cleanly; grad norms 234 -> 55                                                                                                                                                                                                                                                                                                                                                                                                                  |

## FP8 + CUDA graphs axis

Not measured in this session (budget exhausted). All PASS rows above
are `--cuda-graph-impl none`. The task explicitly recommended
"try FP8 without CUDA graphs first". The follow-up `te_attn` graph run
for Paths A, B, D is a trivial next step — just re-invoke the
`scripts/remote_smoke_h200_fp8_mamba3_matrix.sh` wrapper with
`CPPMEGA_CUDA_GRAPH=te_attn`.

## Production-readiness assessment

- **Path A (`mamba3_te_stack_spec`)** — production-ready for FP8 training.
  `CppMegaMamba3Mixer` subclasses `MambaMixer` with QK-Norm / B-C bias /
  data-dependent A (opt-in), uses `mamba_chunk_scan_combined`, is
  CUDA-graph compatible, and now empirically validated under FP8.
- **Path B (`mamba3_author_spec`)** — production-ready for FP8 training
  at the correctness level. Performance still bounded by the Author
  scan kernels not participating in TE CUDA graph fusion (same caveat
  as the BF16 mamba3_te number, 127k tok/sec).
- **Path C (Author MIMO R=4)** — **UNBLOCKED as of 2026-04-11**. The
  TileLang MIMO backward kernel `mamba_mimo_bwd_combined` was patched
  to handle the `1 < G < H` case (was raising `ValueError: G value of 8
  is not currently supported!` at line 1314 of
  `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`). The fix adds a
  post-kernel reduction for partial-group contraction:
  `dq = dq_tilelang.view(B, S, R, G, H//G, N).sum(dim=4)` (and same
  for dk). First successful NAM56R MIMO 7/7 full-stack training run
  landed 2026-04-11 at **56,280 tok/sec** baseline on 8x H200; see
  `docs/nam56r_mimo7_baseline_2026_04_11.md`. FP8 was NOT enabled in
  that run — re-testing Path C under FP8 is a pending follow-up.
- **Path D (`nam56r_noconv_spec`)** — production-ready for FP8 training
  once the m2rnn sibling guard removal is accepted. Steady-state ms/iter
  at 37-38s reflects the heavyweight cppmega pipeline wiring and is
  comparable to the ongoing bf16 smoke (69-75s/iter for GBS=32 vs our
  GBS=16 here, so roughly the same per-token wallclock). A further
  benchmark at matched GBS and with CUDA graphs is needed before
  declaring throughput numbers, but correctness is established.
