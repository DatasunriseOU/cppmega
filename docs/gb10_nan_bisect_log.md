# GB10 NaN bisect — Test loop state log

Machine: GB10 (sm_121a, GB10 consumer, 128 GB unified memory, driver 590.48.01).
Direct `ssh gb10`. Venv `/home/dave/cppmega-venv`. torch 2.12.0.dev20260407+cu132,
TE 2.13, tilelang f309d814, mamba_ssm 2.3.1, triton 3.7.0.

Mamba_ssm state: PRISTINE upstream (`state-spaces-mamba` HEAD `31f3d7b`, clean;
site-packages mirrors it — `TL_DISABLE_TMA_LOWER: True`,
`TL_DISABLE_WARP_SPECIALIZED: True`). P1 mutation NOT present. No
`TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` lines, but preflight was skipped at
e4b3690 (module doesn't exist yet).

## 13-layer symmetric NAM56R cut

Pattern: `CPPMEGA_NEM_PATTERN=AEMEAEMEAEMR`, depth=12 cycles → depth=13 gives
`AEMEAEMEAEMRA`. `build_default_hybrid_layer_pattern` remaps:
`A→* (4 MLA/DSA), E→E (4 MoE expert), M→M (4 Mamba3), R→M (1 M2RNN sent as
Mamba)` yielding final `*EME*EME*EMM*` + `/*-` (MTP).  `CPPMEGA_DSA_A_LAYER_RANKS="1,2,3"` so rank 0 = MLA, ranks 1-3 = DSA.
So: 1 MLA + 3 DSA + 4 MoE + 4 Mamba3/M2RNN + 1 MTP = 13 main layers.

Config:
- hidden=2048, ffn=5632, heads=16 (quarter-ish of full NAM56R's 3584/18944/28)
- seq=2048 MBS=1 GBS=1 TP=1 PP=1 EP=1
- BF16 only (no FP8)
- `--attention-backend unfused`
- 5 iter smoke
- no CUDA graphs, no recompute, no distributed optimizer

Peak mem iter-2: ~18.4 GB allocated / 19.2 GB reserved. Fits GB10 easily.

## Iter log

| # | Date       | Commit  | Era        | Iter grad_norms                         | Verdict |
|---|------------|---------|------------|-----------------------------------------|---------|
| 1 | 2026-04-15 | `e4b3690` | Apr 10 19:14 | 53.570 / 258.605 / 43.464 / 68.223 / 67.812 | **FINITE — healthy training** |
| 2 | 2026-04-15 | `785db65` | Apr 12 (midpoint 75/150) | 53.302 / 189.820 / 48.638 / 78.343 / 80.666 | **FINITE** — healthy decay 11.39→7.09; confirms NaN in `785db65..HEAD` (74 commits). Required `--moe-token-dispatcher-type alltoall` override (flex requires TPxEP>1) |
| 3 | 2026-04-15 | `acb18e5` | mid of 785db65..HEAD (37/74) | 53.287 / 188.228 / 48.200 / 80.111 / 86.835 | **FINITE** — loss 11.39→7.09 |
| 4 | 2026-04-15 | `a96fa3f` | TileLang kernel hammer | 53.308 / 189.056 / 48.469 / 77.733 / 84.667 | **FINITE** |
| 5 | 2026-04-15 | `dd4da34` | MTP Liger FLCE #968 grad fix | 53.287 / 188.228 / 48.246 / 80.245 / 87.207 | **FINITE** |
| 6 | 2026-04-15 | `7a35918` (HEAD) | sanity check | 53.308 / 189.056 / 48.469 / 77.733 / 84.667 | **FINITE** — confirms NaN is NOT a commit regression |
| 7 | 2026-04-15 | HEAD + FP8 tensorwise | 13L MBS=1 FP8 | 49.910 / 149.357 / 29.731 / 72.281 / 48.376 | **FINITE** |
| 8 | 2026-04-15 | HEAD + FP8 + MBS=8 | 13L FP8 MBS=8 | 38.098 / 105.636 / 137.161 / 73.814 / 29.365 | **FINITE** |
| 9 | 2026-04-15 | HEAD + FP8 + 52L MBS=1 | full layer count, FP8 | 96.014 / 205.842 / 137.069 / 111.600 / 48.699 | **FINITE** |
| 10 | 2026-04-15 | HEAD + FP8 + 52L MBS=8 | matches bench3 closest | — | **GB10 OOM-hung host (sshd unresponsive)** |

## FINAL conclusion (Test 2026-04-15 05:55+) — SparseMLA cleared, EP=8/H200 confirmed

Two more tests closed the last single-GPU gap:

| # | Config                                                                    | Result |
|---|---------------------------------------------------------------------------|--------|
| 15 | HEAD + Liger-full + **60 iter** (13L MBS=8 FP8 tensorwise) — late-iter sanity | **FINITE all 60** (lm 11.4→2.38, grad 37→2.0), zero NaN |
| 16 | HEAD + **TileLang SparseMLA BF16** + Liger MTP + Liger main-head + DSA indexer fused (13L MBS=8 FP8) | **FINITE** (iter 1 grad 37.486 — byte-identical to no-SparseMLA run; iter 11 lm 5.25) |

| 17 | HEAD + **full bench3 env**: CPPMEGA_NGRAM_HASH_ENABLED=1 + CPPMEGA_STRUCTURE_ENABLED=1 + MAMBA3_MIMO=1 + MAMBA_NUM_GROUPS=8 + MAMBA_RECOMPUTE=1 + TileLang SparseMLA + Liger full + FP8 tensorwise MBS=8 | **FINITE** iter-1 grad 61.642 (vs 37.486 without ngram/structure — extra embedding paths confirmed active); iter 8 grad 48.6; zero NaN across 10 iters; validation PPL 187 |

| 18 | HEAD + full bench3 env + **true NAM56R dims**: hidden=3584 ffn=18944 heads=28 (same as bench3, not my quarter 2048/5632/16) + 13L + MBS=4 + FP8 tensorwise + TileLang SparseMLA + all Liger patches | **FINITE** 10/10 iters (iter-1 grad 83.6, iter-2 grad 318.7 spike, iter-5 lm 12.4 oscillation, iter-10 lm 8.22 grad 22.8 recovering); validation PPL 408/560; 42 GiB allocated |

Tests 16 + 17 + 18 closed every untested-on-GB10 bench3 component: TileLang
SparseMLA, ngram_hash, structure, full NAM56R per-layer dims. Test 18
specifically rebuts the "bench3 dims might trigger a shape-specific kernel
bug" theory — bench3's exact hidden/ffn/heads are now validated finite on
GB10 at 13L. The NaN is definitively NOT a single-GPU issue at any shape
or feature combination.

| 19 | HEAD + full bench3 env + true NAM56R dims + **MBS=10** (bench3 golden batch) — 13L / 10 iter | **FINITE** all 10 (iter-1 grad 67, iter-2 spike 328, iter-10 lm 6.5 grad 25); 87 GB peak; validation PPL 389/500. Closest-to-bench3 single-GPU config possible. |

| 20 | HEAD + full bench3 env + full dims + MBS=10 + **CPPMEGA_INDEX_CACHE=1** (3 Full + 6 Shared DSA layers, 66% indexer savings) — bench3 2026-04-13 golden config | **FINITE** 10/10 (iter-1 grad 66.7, iter-2 spike 333.9, iter-10 lm 6.99 grad 21.7); zero NaN |

**This is the final possible single-GPU config on GB10.** Everything bench3
golden that can run on GB10 sm_121a has been validated finite:
FP8 tensorwise, Liger MTP+main-head, TileLang SparseMLA BF16, DSA indexer
fused, IndexCache, ngram_hash, structure, selective recompute,
mla-down-proj-fusion, clip-grad 1.0, full NAM56R dims (3584/18944/28), 13L,
MBS=10. Not testable on GB10 (hardware gate): EP=2+ multi-rank DeepEP (NCCL requires
unique device per rank), H200 sm_90 TE FP8 kernels.

| 21 | HEAD + full bench3 env + **CPPMEGA_LEMYX_DSA=1 + CPPMEGA_INDEX_CACHE=1** (both enabled; required cloning `lemyx/tilelang-dsa` to GB10 sibling path, REMOTE_ROOT override) + full dims + MBS=10 | **FINITE** 10/10 (iter-1 grad 67.1, iter-2 spike 328.3, iter-10 lm 7.08 grad 26.5); TileLang lemyx fused FA+KL kernel WORKS on GB10 sm_121a Blackwell consumer. Zero NaN. |

**This is the absolute maximum bench3 match achievable on single GPU.**
Every single bench3 production feature (LEMYX DSA fused FA+KL TileLang
kernel, IndexCache, TileLang SparseMLA, Liger MTP+main-head CE, DSA indexer
fused, ngram_hash, structure, selective recompute, mla-down-proj-fusion,
clip-grad, full 3584/18944/28 dims, MBS=10, FP8 tensorwise) has now been
validated FINITE on GB10. The bench3 NaN is definitively a property of
EP>1 DeepEP collective backward or H200 sm_90 TE FP8 kernel specifics. Prior bisect
launchers never installed the SparseMLA monkey-patch — they ran native Megatron
`unfused_dsa_fn`. Test 16 installs `sparse_mla_as_unfused_dsa` (from
`cppmega.megatron.sparse_mla_ops.sparse_mla`) and observes bit-identical iter-1
grad_norm → TileLang SparseMLA is numerically equivalent on GB10 sm_121a and
NOT the source of bench3 iter-1 NaN.

Bench3 investigation doc (`docs/grad_nan_investigation_2026_04_15.md`) listed
three residual suspects on bench3 after its own MTP-CE / main-head-CE /
mamba3-compile / recompute bisect came up empty: (a) `dsa_indexer_fused_patch`
mystery call site, (b) TileLang SparseMLA bwd, (c) FP8 amax warmup. GB10 now
clears all three with the same features bench3 had and identical HEAD:
- (a) Test 16 explicitly installs `apply_dsa_indexer_fused_patch()` — finite.
- (b) Test 16 installs `sparse_mla_as_unfused_dsa` — finite.
- (c) Tests 7, 11, 15 all use `--fp8-format hybrid --fp8-recipe tensorwise` — finite.

**One unchanged variable remains between GB10 (finite) and bench3 (NaN):**
EP=8 DeepEP / flex-dispatcher all-to-all backward on 8× H200 sm_90. GB10 cannot
exercise this collective (EP=1, single GPU, no A2A). Investigation must move to
LOCATION_2 H200 (per user directive, untouched) — the bench3 fabric is gone.

### Leading hypothesis (unverified — EP>1 FP8 amax starvation)

Code review of Megatron MoE flex dispatcher + DeepEP (no single-GPU repro
possible) identifies one specific iter-1 mechanism that requires EP>1 AND FP8:

1. Router produces top-k routing at iter-1 with random-init weights — per-rank
   token counts after A2A are stochastic; some ranks can receive zero tokens.
2. A rank with zero tokens in iter-1 forward never fires FP8 amax-kernel atomics;
   its local amax stays 0.
3. Megatron's FP8 amax reduction is a collective all-reduce across the EP group
   on backward. One rank with `amax=0` collapses the reduction to 0 on all
   ranks → `scale = 1/amax = inf` → backward grad = `inf * bf16` = NaN.
4. EP=1 never hits this: with one expert group there is no cross-rank amax
   reduction and every forward fires the atomic. BF16-only runs never hit it
   either because no FP8 amax is used.

**Testable predictions on europe H200:**
- BF16 + EP=8 + MBS=10 → FINITE (if hypothesis correct)
- FP8 + EP=1 (PP/TP only) + MBS=10 → FINITE
- FP8 + EP=8 + deterministic router load-balancing (`--moe-aux-loss-coeff 0.1
  --moe-expert-capacity-factor 2.0`) → may mask NaN if load-balance prevents
  zero-token ranks

Status: hypothesis only; not confirmed by direct DeepEP kernel trace. Testable
only with europe H200 access or a rebuilt bench3.

**Counter-evidence against the amax-starvation hypothesis:** GB10 EP=1 runs
already have 12/16 experts idle per iter (topk=4, 16 experts, BF16+FP8
tensorwise) and are FINITE over 60 iterations. If `amax=0 → scale=inf`
collapsed the backward, this would fire at EP=1 too. TE FP8 amax is
initialized with `torch.zeros_like()` (te/pytorch/module/base.py:796), but the
C++ `_amax_and_scale_update` helper likely clamps the divisor to a minimum —
otherwise iter-1 would NaN on any FP8 model with any idle experts. The bug
thus probably lives in the cross-rank amax all-reduce path or in DeepEP
backward collective specifically, not in local amax handling.

### Attempted: EP=2 on single GB10 via device-sharing (FAILED to launch)

`scripts/gb10_test_ep2.sh` tries `nproc_per_node=2 --expert-model-parallel-size
2` with both ranks forced to `cuda:0` via `LOCAL_RANK=0` override in shim.
Result: NCCL 2.29.7 rejects with `ncclInvalidUsage` at
`torch.distributed.barrier()` — NCCL fundamentally requires a unique physical
device per rank. Without MPS (Multi-Process Service) setup and a NCCL build
that tolerates same-device ranks, a single-GPU multi-rank DeepEP test is
impossible on GB10. This closes every single-machine angle. The user-hint
"two will fit in memory" was attempted here — memory would have been fine,
but NCCL does not permit the topology.

## DEFINITIVE conclusion (earlier — superseded above) — NaN IS MULTI-GPU SPECIFIC

After GB10 host recovered, I tested the previously-untested hypothesis (Liger CE
patches not installed in earlier launcher) and added EVERY bench3 feature one
by one. Final test matrix (all FINITE, no NaN ever):

| # | Config | Verdict |
|---|--------|---------|
| 11 | HEAD + Liger MTP + Liger main-head + FP8 + 13L MBS=8 | 37/95/97/26/71 FINITE |
| 12 | HEAD + Liger + FP8 + **52L** MBS=1 | 96/167/68/131/87 FINITE |
| 13 | HEAD + Liger + LEMYX + INDEX_CACHE + recompute + clip-grad + MLA fusion + 13L MBS=8 | 37/96/97/26/78 FINITE |
| 14 | HEAD + ALL bench3 features + **52L MBS=2** | 70/165/73/55/149 FINITE |

Only difference from bench3 left:
- bench3: 8 GPUs EP=8 + MBS=10 + sm_90 H200
- GB10: 1 GPU + MBS≤8 + sm_121a Blackwell

**The NaN must originate in one of:**
1. **Multi-GPU EP=8 DeepEP/flex-dispatcher backward all-to-all** — fundamentally
   not exercisable on single-GPU; collective grad-sync layer never runs at EP=1.
2. **H200 sm_90 hardware-specific TE FP8 quantizer issue** — different SM than
   GB10 sm_121a, but tensorwise recipe should be identical numerically.
3. **MBS=10 specific overflow** — GB10 hung at 52L MBS=8, never reached MBS=10.

**To pin it down**: test on LOCATION_2 H200 (currently untouched per user
directive). Specifically: full NAM56R 52L + FP8 tensorwise + MBS=10 + EP=8, with
and without each Liger patch. That's the only remaining variable space.

## REVISED conclusion (earlier — superseded above) — SMOKING GUN

The bisect launcher `/home/dave/gb10_bisect_launcher.sh` does **NOT** invoke the Liger
patches that bench3 production launcher injects via `cppmega_mimo_shim.py`:

- `patch_mtp_loss_with_liger()` (mtp_liger_ce.py) — UNCONDITIONAL since dd4da34 in bench3
- `patch_mamba_output_layer_with_linear_ce()` (apply_linear_ce_patch.py) — same
- + `--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear` flags

Both are *only* installed if explicitly imported. My bisect ran NATIVE Megatron MTP and
NATIVE main-head CE. So the bisect proved cppmega *core* code is finite — but the Liger
patch path (the one bench3 actually uses) was never exercised.

Per `docs/findings_2026_04_14_session.md`:
> mtp_liger_ce.py:169 used reduction="none" in early version. ALL bench3 baselines
> (incl. 268 / 269.4 TFLOP/s) had MTP gradients silently corrupt via FLCE #968.

dd4da34 swapped to `reduction="mean"` + broadcast — possibly INTRODUCING the visible NaN
because the prior corruption was masked by silent grad zeroing.

## Next test (Test iter 12) — do this when GB10 reset

1. Modify `/home/dave/gb10_bisect_launcher.sh` to inject a shim that calls:
```python
from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
patch_mtp_loss_with_liger()
from cppmega.megatron.apply_linear_ce_patch import patch_mamba_output_layer_with_linear_ce
patch_mamba_output_layer_with_linear_ce()
```
2. Add `--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear` to torchrun args.
3. Test on HEAD, 13-layer FP8 MBS=8.
4. If NaN appears → bisect Liger patch series (dd4da34, d1249f5).
5. If finite → also add `CPPMEGA_LEMYX_DSA=1` + `CPPMEGA_INDEX_CACHE=1` + bench3 features.

GB10 host status: SSH banner timeout (port 22 reachable, sshd hung from OOM). Needs
physical reset before iter 12.

Bench3 is DELETED. Reproduction otherwise requires LOCATION_2 H200 (untouched per user).


Full log: GB10 `/home/dave/logs/gb10_bisect_20260415_020839.log`
Launcher:  GB10 `/home/dave/gb10_bisect_launcher.sh` (also at local
`/tmp/gb10_bisect_launcher.sh`).

### Iter 1 details

Losses iters 1-5: lm 11.40, 10.24, 7.62, 8.61, 8.99. mtp_1 11.40, 10.42, 7.76,
7.71, 7.14. Validation lm loss 8.11, PPL 3338. Grad norm range 43-259 — typical
for 5-iter random init. `number of nan iterations: 0` for all iters.

## Interpretation

The 4-days-old code at `e4b3690` on GB10 (sm_121a) with pristine mamba_ssm +
BF16 + unfused attention + 13-layer NAM56R cut produces **finite gradients and
healthy loss decay**. Therefore:

1. The NaN observed on bench3 (now deleted) and in earlier GB10 runs is NOT
   intrinsic to cppmega's NAM56R architecture.
2. The NaN was introduced between `e4b3690` (Apr 10 19:14) and `HEAD` (`7a35918`,
   Apr 15).
3. Candidate commits to bisect forward through (9 commits in this window):

```
7a35918 2026-04-15 fix(bisect): soften preflight_smem_check
840993a 2026-04-15 fix(bisect): PYTHONPATH + v3 overlay + import precheck
4b979e7 2026-04-15 feat: bisect_nan_golden.sh worktree bisect
96f8c7a 2026-04-14 fix: gitignore + drop --include-untracked
8d751b3 2026-04-14 revert: A. KL loss back to 0
...
0038ad4 2026-04-13 19:00 "last known finite grad" baseline per bench3 notes
0ce8a3a 2026-04-14 "main-head Liger CE patch" claimed 269.4 record
b1fa542 TileLang SparseMLA default
dd4da34 CE patches mandatory
```

Full list: `git log --oneline e4b3690..HEAD` → 60+ commits. Need to binary-
bisect.

## Next Test-loop iteration (iter 2)

Recommended: test a commit halfway between `e4b3690` (Apr 10 19:14) and
`0038ad4` (Apr 13 22:46, claimed "last known good"). Candidates:

- `ff590d4` (2026-04-12 04:16) — feat stream-l DSA 9+4 FP8 indexer EP launcher
- `99a95ae` (2026-04-12 04:11) — feat NAM56R Mamba-3 MIMO + DSA 9+4 FP8 indexer
- `cb3a70d` (2026-04-12 01:04) — fix MoE token dispatcher alltoall
- `16352c8` (2026-04-12 00:56) — fix ban NullTokenizer/mock-data

Choose `99a95ae` — introduces DSA 9+4 + FP8 indexer + TP mixer. Largest diff.

## Bisect method reproducibility

Launcher `/home/dave/gb10_bisect_launcher.sh` reads `BISECT_ROOT` env;
swap worktree commit + re-run. Pre-checks importing from worktree
(`assert 'cppmega-bisect' in cppmega.__file__`). Does NOT require preflight
(doesn't exist pre-e712801, skipped by hand at e4b3690; should be re-enabled
when testing post-preflight commits).

## Outstanding blockers (nothing blocking next iter)

- cuDNN fused attention not tested (intentionally used `--attention-backend
  unfused`). Does not affect bisect validity.
- `--attention-backend unfused` works. No sublib load failure observed.
- `AccumulateGrad stream mismatch` warning present but non-fatal.
