# Mamba3 Wave28 Lane A Production Gate - 2026-04-30

Branch: `worker/mamba3-wave28-prod-20step-gate`

Base: `d7513ec1d77024d27a904434a1dfc5b113d14062`

Scope: production benchmark harness/docs only. No Mamba3 kernel, model, recipe,
or training behavior was changed.

## Runner Choice

The least risky full-size runner in `main` is:

```text
scripts/remote_production_h200_nam56r_v1.sh
```

Reasons:

- It already targets the full NAM56R production boundary, not a mini/component
  shape.
- It uses real production flags including `--log-throughput`.
- It already reports per-rank peak GPU memory through `[production_peak_mem]`.
- It does not require adding a new training stack.

Added wrapper:

```text
scripts/remote_mamba3_stage2_prod_ab_gate.sh
```

This wrapper runs the production runner twice:

1. `baseline`: roll back/verify the installed `mamba_ssm` bwd file is clean.
2. `stage2_force_nontma_bf1_bb0`: apply and verify the guarded candidate with
   `bf_num_stages=1, bb_num_stages=0`.

The wrapper:

- Defaults to `TRAIN_ITERS=20` and refuses fewer than 20.
- Refuses non-H200 GPUs unless explicitly overridden.
- Runs the guarded applier default-off no-op check before training.
- Verifies baseline clean state before launch.
- Verifies candidate patched state before launch.
- Rolls back and verifies clean state after candidate, including an exit trap if
  the candidate training command fails.
- Writes `summary.csv`, `summary.json`, and `summary.md` from tok/sec and peak
  memory lines in the production logs.

## Local Validation

Run from the Lane A worktree:

```text
bash -n scripts/remote_mamba3_stage2_prod_ab_gate.sh

PYTHONPATH=. python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
# SKIP CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA is not set

PYTHONPATH=. CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1 \
  python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
# FAIL: Refusing to mutate installed mamba_ssm without
# MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1

scripts/remote_mamba3_stage2_prod_ab_gate.sh
# ERROR: this full-boundary gate requires H200 by default.

CPPMEGA_ALLOW_NON_H200_FULL_GATE=1 TRAIN_ITERS=19 \
  scripts/remote_mamba3_stage2_prod_ab_gate.sh
# ERROR: TRAIN_ITERS=19; production A/B gate requires at least 20.
```

These checks do not launch GPU training on the local GB10.

## Full H200 Run Status

No full 20-step NAM56R run was started in this Lane A pass.

Blocker captured at `2026-04-30T13:01:32Z`: GCloud credentials need
reauthentication and the CLI cannot prompt in this non-interactive session.

Command attempted:

```text
gcloud compute instances list --format='table(name,zone.basename(),machineType.basename(),status)'
```

Blocker:

```text
ERROR: (gcloud.compute.instances.list) There was a problem refreshing your current auth tokens: Reauthentication failed. cannot prompt during non-interactive execution.
Please run:

  $ gcloud auth login

to obtain new credentials.
```

Modal was checked but not used for this full NAM56R gate. `main` does not
contain the previous stage2 full benchmark Modal script, and creating a new
full training Modal stack would be higher risk than wrapping the established
production H200 runner. No Lane A Modal app was started.

## Reproducible Full-Run Command

On an H200 host with the Lane A branch checked out:

```text
cd /path/to/cppmega
git checkout worker/mamba3-wave28-prod-20step-gate
CPPMEGA_GATE_ID=wave28_lane_a_h200_stage2_prod_ab_20260430 \
TRAIN_ITERS=20 \
bash scripts/remote_mamba3_stage2_prod_ab_gate.sh
```

Through GCloud after reauthentication, use the H200 host/zone for the target:

```text
gcloud compute ssh <h200-host> --zone <zone> --command \
  'cd /path/to/cppmega && git checkout worker/mamba3-wave28-prod-20step-gate && CPPMEGA_GATE_ID=wave28_lane_a_h200_stage2_prod_ab_20260430 TRAIN_ITERS=20 bash scripts/remote_mamba3_stage2_prod_ab_gate.sh'
```

Artifacts will be written under:

```text
artifacts/mamba3_stage2_prod_gate/wave28_lane_a_h200_stage2_prod_ab_20260430/
```

Expected files:

- `baseline.log`
- `candidate_stage2_force_nontma.log`
- `default_off_noop.txt`
- `pre_baseline_rollback.txt`
- `pre_baseline_clean_check.txt`
- `candidate_apply.txt`
- `candidate_patched_check.txt`
- `post_candidate_rollback.txt`
- `post_candidate_clean_check.txt`
- `summary.csv`
- `summary.json`
- `summary.md`

## Results

The production A/B table is pending the full H200 run:

| variant | GPU | train iters | tok/sec | peak alloc GiB | peak reserved GiB | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H200 | 20 | pending | pending | pending | blocked by GCloud reauth |
| stage2_force_nontma_bf1_bb0 | H200 | 20 | pending | pending | pending | blocked by GCloud reauth |

## Main Safety

Nothing here is a performance result. The harness is safe to keep off the main
runtime path: it is a standalone production gate wrapper and docs only. The
stage2 candidate itself is not validated safe for main default-on behavior by
this pass because the full 20-step comparison did not run.
