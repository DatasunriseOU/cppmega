# grad_norm=NaN investigation — 2026-04-15 (bench3 H200 NAM56R)

## Summary

User suspected MTP Liger FLCE path (`apply_linear_ce_patch.py` + `mtp_liger_ce.py`,
both rewritten in `dd4da34` to use `reduction="mean"` + broadcast trick to sidestep
Liger #968).

**Empirically refuted.** Bisect across MTP CE, main-head CE, mamba3 regional
compile, and selective recompute — every variant still produces `grad norm: nan`
on iter 1. The bug is upstream of all four candidates.

The "golden 268 TFLOP/s" run that user referenced
(`retained_bench3_268_2026-04-14.log`) NEVER ran a real iteration — it crashed
during init. There is no recorded finite-grad run on bench3 against
HEAD `8d751b3`. Last known finite-grad run is `cppmega_tilelang_test.log`
(2026-04-13 22:46, commit ~`0038ad4`, before `b1fa542`). Many code paths shifted
between those two points.

## Test matrix

All runs on `h200_1` zone `LOCATION_1`, HEAD `8d751b3`.
Logs in `/mnt/data/cppmega-root/cppmega/test*.log`.

| # | Mutation                                                                 | Iter-1 grad_norm | Verdict |
|---|---|---|---|
| 1 | `MTP_DEPTHS=0` (skip MTP path entirely)                                  | **nan**         | MTP innocent |
| 2 | + `CPPMEGA_PREFER_NATIVE_HOPPER_CE=1` (skip main-head Liger reroute, native PR #3345 used) | **nan**         | Main-head Liger innocent |
| 3 | + `apply_linear_ce_patch` short-circuited + drop `--cross-entropy-loss-fusion` (vanilla CE on logits) | **nan**         | LinearCE class-swap innocent |
| 4 | Revert `mamba3_te_mixer.py` to inline `_A/DT/ADT` math (no `_compiled_data_dep_A`) | **nan**         | Mamba3 regional compile innocent |
| 5 | `--recompute-granularity` removed (no selective recompute)               | **nan**         | Recompute innocent |

All five variants converge on the same iter-1 result: `lm loss ~12 (finite),
grad norm: nan`. Forward is fine, backward emits NaN gradients on the very
first iteration regardless of which CE / compile / recompute combination is
used.

## What's still on the table (NOT bisected)

Items that are common to ALL five tests and would require additional bisect
to clear:

1. **`dsa_indexer_fused_patch`** — always-applied. Tried to disable via
   `DISABLED_FOR_BISECT:` comments in `remote_smoke_h200_dsa_9_4_m.sh` lines
   324–325 (pre-existing local edit on bench3 by another agent), but the
   "[cppmega] DSA indexer fused patch applied" message still prints in every
   test log. Some other call site is invoking it that we did not locate.
   `grep -r apply_dsa_indexer_fused` only finds the `dsa_indexer_fused_patch.py`
   itself + commented call in `remote_smoke_h200_dsa_9_4_m.sh` + an unrelated
   `remote_sweep_h200_dsa_production.sh`. The mystery call site needs to be
   found before this candidate can be cleared.
2. **TileLang SparseMLA** monkey-patch (replaces `unfused_dsa_fn`) — applied
   in all five tests via `cppmega_fp8_shim.py:262` and inline shim line 357.
3. **TE FP8 amax warmup** — `--fp8-format hybrid` with empty `fp8_amax_history`
   on iter 1 may produce inf scaling factors. Working `tilelang_test` log
   (Apr 13 22:46) ALSO had `fp8 hybrid` and got finite grads, so this is not
   sufficient on its own — but it could be a contributor in combination with
   another change.
4. **Mamba3 `_apply` guard DISABLED** — bf16 D/dt_bias + `.float()` in fwd
   (per `feedback_no_apply_guard`). Working test ALSO had this disabled.
   Same comment as FP8 amax — necessary but not sufficient.
5. **mamba3 P1 patches** (`apply_mamba3_mimo_p1_patches.py`) — opt-in via
   `CPPMEGA_MAMBA3_P1=1`, default 0. Was OFF in our tests, so not the cause.
6. **`hybrid_schedule_plan`** rewrite — heavy diff between 0038ad4 and HEAD.
   Touches MTP and 1F1B, but Test 1 with MTP=0 still NaNs, so
   process_mtp_loss isn't the culprit.

## Forensic pointers

- Working ref: `cppmega_tilelang_test.log` (2026-04-13 22:46) iter-1 `lm loss
  1.197, grad norm: 89.206`. Same machine, same FP8 hybrid, same TileLang
  SparseMLA, same `_apply` guard disabled, same Mamba3 regional compile (via
  the OLD `_patch_cppmega_mamba3_te()` path — wrapped in
  `@torch._dynamo.disable`).
- Broken HEAD any-test: iter-1 `lm loss ~12, grad norm: nan`.
- Diff in code between the two: `b1fa542` (TileLang SparseMLA as default), 
  `2377c73` (inline `_compiled_data_dep_A`), `dd4da34` (CE patches mandatory),
  `b6fb886`/`67c51d7` (single-path DSA + DualPipeV + recompute infrastructure).
- The three CE / mamba3 / recompute candidates have all been ruled out
  empirically. That leaves the SparseMLA replacement, indexer fused patch,
  hybrid_schedule_plan, or some interaction between FP8 amax warmup and any
  of these.

## Recommended next steps (NOT executed in this session)

1. **Find the mystery call site** for `apply_dsa_indexer_fused_patch`. The
   message prints despite the script-level call being commented. Likely
   imported as side-effect via another module. Until found, this candidate
   cannot be cleanly bisected.
2. **Switch SparseMLA back to `gather_scatter`**:
   `CPPMEGA_DSA_SPARSE_MODE=gather_scatter` and re-test. If grad finite,
   TileLang SparseMLA bwd is the bug (likely the d_v parameter change in
   commit `cc6b6d3` or the BF16 path changes in `b1fa542`/`174810b`).
3. **Disable FP8 entirely** (drop `--fp8-format hybrid`) and re-test. If grad
   finite, FP8 amax warmup interaction is implicated.
4. **Disable both** SparseMLA replacement AND FP8 simultaneously to
   establish whether either alone is sufficient or only the combination
   triggers NaN.
5. **Print iter-1 gradient norms per layer**: instrument `clip_grad_norm` to
   dump per-parameter `.main_grad.abs().max()` so we can identify which
   parameter group first goes NaN. This narrows the search to a specific
   sub-module.

## Bench3 cleanup performed

- Reverted edits to `cppmega/megatron/mamba3_te_mixer.py` and
  `cppmega/megatron/apply_linear_ce_patch.py` (`git checkout`).
- All local test logs left in place under
  `/mnt/data/cppmega-root/cppmega/test*.log` for follow-up investigation.
- The pre-existing local edit in `scripts/remote_smoke_h200_dsa_9_4_m.sh`
  (DISABLED_FOR_BISECT comments at lines 324–325) was NOT made by this
  session and was left as-is.
- All training processes killed, GPU memory verified free.

## What was NOT done

- No fix committed — root cause still unidentified.
- Did not write per-parameter gradient probe (would have required a Megatron
  monkey-patch and another full training run cycle).
