# grad_norm=NaN bisect on bench3 — 2026-04-15

Environment: `h200_1` (LOCATION_1), 8× H200, venv `/mnt/data/venv`, torch 2.12+cu132, TE 2.13, mamba_ssm 2.3.1, megatron-core 0.18 pre-installed.

## Problem

- **Known-good (claimed)**: commit `0ce8a3a` "feat: main-head Liger CE patch — bench3 269.4 TFLOP/s new record", measured 2026-04-14 11:12, per memory `reference_main_head_liger_ce_gap.md`: FP8 tensorwise + MBS=10 + Liger CE, reported 269.4 TFLOP/s with FINITE `grad_norm`.
- **Known-bad**: current HEAD (`4b979e7`), `grad norm: nan` every iteration.
- **Primary suspect at outset**: `dd4da34` "fix: remove env-gate fallbacks + fix MTP Liger FLCE #968 silent grad corruption" — switched Liger reduction="none"→"mean"+broadcast on main head AND MTP.

## Method

Clean worktree at `/mnt/data/cppmega-root/cppmega-bisect` checked out to target commit. `PYTHONPATH` prepended so python imports the worktree copy. Launcher script was pinned from main tree `scripts/remote_smoke_h200_dsa_9_4_m.sh` (to avoid older committed launchers that hardcode europe paths).

Golden env used: `VARIANT=v3 PP=1 EP=8 MBS=10 GBS=80 TRAIN_ITERS=5`, FP8 hybrid tensorwise, `CPPMEGA_MAIN_HEAD_LINEAR_CE=1 CPPMEGA_LINEAR_CE_KERNEL=liger CPPMEGA_MTP_LIGER_CE=1 CPPMEGA_INDEX_CACHE=1 CPPMEGA_LEMYX_DSA=1 CPPMEGA_DSA_INDEXER_LOSS_COEFF=0 CPPMEGA_DSA_SKIP_INDEXER_LOSS=1`, `--recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj --mla-down-proj-fusion --clip-grad 1.0`.

## Result table

| Commit | Label | TFLOP/s | grad_norm (5 iters) | Verdict |
|---|---|---|---|---|
| `0ce8a3a` | Claimed golden, 2026-04-14 | 30.5 / 55.1 / 295.6 / 295.7 / 295.5 | nan / nan / nan / nan / nan | **BAD** (ALL NaN) |
| `4b979e7` HEAD (via parallel agent `test4_mamba3_inline_v2`, PP=2 EP=4 MBS=4) | current main | ~185 | nan ×76 iters | BAD |

Full iter trace (0ce8a3a, `/mnt/data/cppmega-root/cppmega/bisect_0ce8a3a.log`):
- iter 1: lm loss 11.88, mtp_1 11.79, mtp_2 11.96, grad norm **nan**
- iters 2-5: grad norm **nan**, steady 295.5 TFLOP/s

## Conclusion

**`dd4da34` is NOT the root cause.** The NaN reproduces on the claimed-golden commit `0ce8a3a` under the same env — so it cannot be a regression introduced by any later commit.

Per bisect rubric in the task spec (Case e): "If (a) also NaN → bisect BACKWARD from golden; dd4da34 not cause; mamba_ssm fork changes or environment drift. Abort if so — different investigation." **Bisect aborted as instructed.**

## Observed environment state (bench3, 2026-04-15)

Both the 2026-04-14 golden run AND today's bisect ran against the same in-situ mamba_ssm fork with uncommitted patches; file mtimes show they predate the golden measurement, so the patched fork state itself is not new. Inventory:

- `/mnt/data/cppmega-root/state-spaces-mamba` — uncommitted diff in `mamba_ssm/modules/mamba3.py` (fp32 upcast of `dd_dt + self.dt_bias` → `.to(torch.float32)` before softplus, in both `forward` and `_preprocess`), plus the GQA branch in `mamba3_mimo_bwd.py` / `mamba3_mimo_bwd_varlen.py` / `mamba3_siso_combined.py`. File mtimes 2026-04-13 23:10 — predate golden, so the patched fork cannot explain the regression BY ITSELF. But the memory note `reference_env_drift_bench3_europe.md` flags these as the kind of local drift that has masked prior bench3 claims.
- `/mnt/data/cppmega-root/megatron-lm` — NO `.git` directory. Version unverifiable from VCS on this machine. Could have been freshly rsynced / force-copied since 2026-04-14.
- `transformer_engine` 2.13, `mamba_ssm` 2.3.1, torch 2.12 nightly.

## Proposed next steps (NOT executed, per task scope)

1. **Verify the golden claim was ever reproducible.** Current evidence strongly suggests the 269.4 TFLOP/s + finite grad measurement at commit `0ce8a3a` was a point-in-time observation that has never been re-confirmed on this machine. Memory `reference_fp8_mbs10_europe_regression.md` already notes FP8 MBS=10 regressed on europe; bench3 may be in an analogous state now.
2. **Hypothesis: megatron-lm version drifted.** The absence of `.git` in `/mnt/data/cppmega-root/megatron-lm` makes it impossible to prove which Megatron commit ran against the golden baseline. First action should be to re-establish megatron-lm as a real git checkout at a pinned SHA (per `reference_megatron_version.md`: `core_v0.15.0rc7 dev_latest`, PR #3674 applied), then re-run the 5-iter smoke.
3. **Hypothesis: TE 2.13 FP8 tensorwise scaling.** FP8 hybrid tensorwise is exactly the path `reference_fp8_mbs10_europe_regression.md` flags as fragile. Disable FP8 (`FP8_FLAGS=""` in the launcher) and re-run at commit `0ce8a3a`. If BF16 golden is finite, regression is in the FP8 stack (TE / Liger-under-FP8 interaction).
4. **Hypothesis: Liger main-head CE is itself silently producing NaN in current TE 2.13 FP8 path.** Disable with `CPPMEGA_MAIN_HEAD_LINEAR_CE=0 CPPMEGA_MTP_LIGER_CE=0` at `0ce8a3a` and re-check. That would confirm whether the apply_linear_ce_patch is the NaN source under drifted stack — and flip the interpretation of `dd4da34` (from "fix that broke things" to "fix that merely surfaced pre-existing FLCE #968 silent corruption").
5. **Do NOT revert `dd4da34`.** It is not the root cause per this evidence.

## Artifacts

- Step log: bench3 `/home/dave/logs/bisect_nan_0ce8a3a.log`
- Train log: bench3 `/mnt/data/cppmega-root/cppmega/bisect_0ce8a3a.log` (iter grad_norm trace)
- Worktree cleaned up at end of run. Main tree at `/mnt/data/cppmega-root/cppmega` left on `origin/main` (`4b979e7`).
- Parallel agent corroboration (PP=2 EP=4 MBS=4 run `test4_mamba3_inline_v2`): `grad norm: nan` for 76 consecutive iterations.

## 2026-04-15 second attempt — bisect methodology broken

Attempted to re-test commit `0038ad4` (Apr 13 22:46 era) via `scripts/bisect_nan_golden.sh`. Two blockers discovered that invalidate the entire bisect approach as currently coded:

### Blocker 1: launcher venv path — FIXED by env override

First run of `bash scripts/bisect_nan_golden.sh 0038ad4` failed in 4 sec with:
```
scripts/remote_smoke_h200_dsa_9_4_m.sh: line 35: /home/dave/cppmega-root/cppmega-venv/bin/activate: No such file or directory
```
The launcher at commit `0038ad4` hardcodes the europe venv path. The main-branch launcher has bench3/europe autodetection, but the bisect script `cd`s into the worktree and runs the worktree's launcher. Overriding `REMOTE_ROOT=/mnt/data/cppmega-root REMOTE_VENV=/mnt/data/venv` in the outer env passed through `env REPO=... bash ...` and fixed this specific failure.

### Blocker 2: VARIANT=v3 didn't exist at 0038ad4 — no clean fix

Second run failed with:
```
ERROR: unknown VARIANT=v3 (expected v0|v1|v2)
```
The bisect script hardcodes `VARIANT=v3` (EP=8, introduced later, added to launcher at `f6d6bb1` per task context). At commit `0038ad4`, the launcher only knows:
- v0: EP=1
- v1: EP=4 DP=1
- v2: EP=2 DP=2

The "golden" config by definition uses EP=8, which the launcher at `0038ad4` cannot express. The `case` block overwrites any externally-passed `EP_SIZE`. Bisecting backward from the v3 introduction point is structurally impossible without editing the worktree launcher (which would invalidate the test).

### Blocker 3 (FATAL): editable install does not shadow PYTHONPATH

Verified on bench3 with the worktree pip-installed:
```
export PYTHONPATH=/mnt/data/cppmega-root/cppmega:/mnt/data/cppmega-root/megatron-lm
/mnt/data/venv/bin/python -c "import cppmega; print(cppmega.__file__)"
# -> /mnt/data/cppmega-root/cppmega/cppmega/__init__.py   <-- MAIN repo, NOT worktree
```
The editable install `.pth` installs a meta-path finder that points at `/mnt/data/cppmega-root/cppmega-bisect/cppmega`, but the launcher prepends `${REMOTE_ROOT}/cppmega` to `PYTHONPATH`, and filesystem path-based imports take precedence over meta-path finders. **Every bisect run has actually been exercising main-repo code, not the requested commit.**

Evidence: file `apply_linear_ce_patch.py` exists in main but NOT in the 0038ad4 worktree (added in a later commit). Yet the golden env sets `CPPMEGA_MAIN_HEAD_LINEAR_CE=1` which loads this patch — and the earlier 0ce8a3a bisect run did NOT crash at import, confirming it was importing from main.

### Implication for the 2026-04-15 "0ce8a3a is ALSO bad" finding

The earlier bisect result `RESULT_BAD_GRAD_NAN` at `0ce8a3a` was produced by main-repo code at that session's HEAD, not `0ce8a3a` code. The claim "`0ce8a3a` reproduces NaN" is unverified — it actually only says "current main (whatever HEAD was at the time) reproduces NaN with the golden env on bench3." That is the same finding as HEAD being bad, with no additional information.

### Required fixes to make bisect work

1. **Launcher selection**: copy the main-branch launcher into the worktree (or force `PATH`-style override) so it supports v3 variant AND has bench3 venv autodetect.
2. **Import isolation**: unset `PYTHONPATH`, OR set `PYTHONPATH=${WORKTREE}:${WORKTREE%/*}/megatron-lm` (worktree first), OR remove the main-repo `cppmega` prefix from `PYTHONPATH`. Verify post-launch by logging `python -c "import cppmega; print(cppmega.__file__)"` to confirm it resolves to the worktree.
3. **Variant compatibility**: add `v3` to commits that predate `f6d6bb1` by overlaying main's launcher onto the worktree, OR adapt the test to use `v1` (EP=4) at pre-v3 commits (different config, cannot be apples-to-apples golden reproduction).
4. **Verify PYTHONPATH is working BEFORE reporting a verdict**: add a `python -c "import cppmega; assert 'bisect' in cppmega.__file__"` pre-check in the bisect script.

### State after this session

- Worktree at `/mnt/data/cppmega-root/cppmega-bisect` left checked out at `0038ad4` (pip-installed editable but shadowed by PYTHONPATH).
- Editable install `.pth` currently targets `cppmega-bisect`. If anyone runs the main repo launcher WITHOUT setting PYTHONPATH, they will pick up 0038ad4 code via the editable finder. **Recommend running `pip install --no-deps -e /mnt/data/cppmega-root/cppmega` to restore editable install to main repo** before any further work on bench3.
- Logs: `/home/dave/logs/bisect_nan_0038ad4.log`, `/home/dave/logs/bisect_train_0038ad4.log`.

## 2026-04-15 third attempt — bisect script fixed, results verified

Bisect script `scripts/bisect_nan_golden.sh` rewritten (commits `840993a` + `7a35918`) to fix the three blockers above:

1. **PYTHONPATH shadow fixed**: script now runs `pip install --no-deps --force-reinstall -e "$BISECT_ROOT"` to re-point the editable install, sets `PYTHONPATH="${BISECT_ROOT}:${MEGATRON_ROOT}"` with the worktree FIRST, and passes `BISECT_PYTHONPATH` through to the launcher via an inner `bash -c 'export PYTHONPATH=…'` wrapper.
2. **v3 variant available on all commits**: script now `cp`s main's `scripts/remote_smoke_h200_dsa_9_4_m.sh` into the worktree at test time. Older commits' launchers are replaced, so v3 + bench3 venv autodetect always work.
3. **Import pre-check mandatory**: `python -c "assert 'cppmega-bisect' in cppmega.__file__"` runs before training; `RESULT_ABORT_IMPORT_WRONG` if it ever misresolves.
4. **Preflight softened**: `python -m cppmega.megatron.preflight_smem_check` invocation in the overlaid launcher is `sed`-patched to `|| echo "missing — skipped"` because that module was added after 0038ad4.

### Verified results (all with import pre-check IMPORT_OK)

| Commit | Runner log | cppmega import path | Iters (grad_norm) | TFLOP/s steady | Verdict |
|---|---|---|---|---|---|
| `840993a` (HEAD) | `bisect_nan_840993a85f37cfaa03be457253b90b8bd7a94d03.log` | `/mnt/data/cppmega-root/cppmega-bisect/cppmega/__init__.py` | nan/nan/nan/nan/nan | 295.5 | **BAD** (5/5 nan) |
| `0038ad4` (Apr 13 19:00 claimed golden era) | `bisect_nan_0038ad4.log` (third run) | `/mnt/data/cppmega-root/cppmega-bisect/cppmega/__init__.py` | nan/nan/nan/nan/nan | 301.9 | **BAD** (5/5 nan) |
| `0ce8a3a` (Apr 14 claimed 269.4 TFLOP/s record) | `bisect_nan_0ce8a3a.log` (second run, fixed tooling) | `/mnt/data/cppmega-root/cppmega-bisect/cppmega/__init__.py` | nan/nan/nan/nan/nan | 295.5 | **BAD** (5/5 nan) |

All three commits — HEAD, the Apr 13 19:00 "real golden" candidate, and the Apr 14 claimed record — produce `grad norm: nan` for every iteration under the exact same env. With the bisect tool verified-correct (worktree code actually imported, launcher invoking v3), the NaN clearly does **NOT** live in any of these cppmega commits.

### Implication

The regression is NOT in cppmega source. Candidate causes, in decreasing likelihood:

1. **Bench3 environment drift** between Apr 14 (when 269.4 TFLOP/s golden was measured) and today (2026-04-15). Per `reference_env_drift_bench3_europe.md`, bench3 carries uncommitted patches in the local mamba_ssm fork. File mtimes on mamba3.py + mamba3_mimo_bwd.py show 2026-04-13 23:10 — BEFORE the golden measurement — but something in the stack has since changed.
2. **megatron-lm version**: `/mnt/data/cppmega-root/megatron-lm` has no `.git` directory. Version is unverifiable and may have been rsynced/force-copied since 2026-04-14. Cannot prove which SHA ran against the golden baseline.
3. **FP8 tensorwise path fragility**: `reference_fp8_mbs10_europe_regression.md` already documents FP8 tensorwise regressing on europe. bench3 may be in an analogous drifted state.

### Next actions (not executed — outside task scope)

- Re-run golden smoke with `FP8_FLAGS=""` (pure BF16) at HEAD to test hypothesis 3.
- Re-run with `CPPMEGA_MAIN_HEAD_LINEAR_CE=0 CPPMEGA_MTP_LIGER_CE=0` at HEAD to test hypothesis: Liger LM-head CE under FP8 producing silent NaN.
- Re-establish `/mnt/data/cppmega-root/megatron-lm` as a real git checkout at pinned SHA per `reference_megatron_version.md` (`core_v0.15.0rc7 dev_latest`, PR #3674 applied).
- Diff bench3 mamba_ssm fork against what was at Apr 14 11:12 (need snapshot / pre-image hash) to rule out stealth edits.

### Artifacts (bench3)

- Script: `/mnt/data/cppmega-root/cppmega/scripts/bisect_nan_golden.sh` (HEAD `7a35918`)
- Runner logs: `/home/dave/logs/bisect_nan_840993a*.log`, `/home/dave/logs/bisect_nan_0038ad4.log`, `/home/dave/logs/bisect_nan_0ce8a3a.log`
- Train logs: `/home/dave/logs/bisect_train_*.log`
