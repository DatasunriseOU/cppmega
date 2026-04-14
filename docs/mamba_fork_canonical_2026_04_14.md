# Mamba_ssm fork canonical-state reconciliation — 2026-04-14

Investigation triggered by spot-VM evacuation risk on H200 bench3 / europe. Goal:
pick a canonical HEAD + modified-file set before either machine disappears.

## HEADs

Both machines are on the **same** upstream commit — the user's note of `4f4857f`
on europe was out of date.

| Machine | Path                                              | git HEAD      | Commit subject |
|---------|---------------------------------------------------|---------------|----------------|
| bench3  | `/mnt/data/cppmega-root/state-spaces-mamba`       | `31f3d7baba`  | Fix Mamba3 Step Fn deprecation warnings: make_fragment -> make_rmem_tensor, arch.exp -> math.exp (#898) |
| europe  | `/mnt/data/cppmega-root/state-spaces-mamba`       | `31f3d7baba`  | (same)         |

No branch divergence. All deltas live in the working tree (unstaged mods).

## File-level md5 comparison (unstaged working-tree contents)

| File (relative to repo root)                                      | bench3 md5                         | europe md5                         | Status    |
|-------------------------------------------------------------------|------------------------------------|------------------------------------|-----------|
| `mamba_ssm/modules/mamba3.py`                                     | `9b8cb10360fc84bfa574e28365e2ff82` | `9b8cb10360fc84bfa574e28365e2ff82` | identical |
| `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`                | `eacf8f23051dccd2916e8c698f5c515d` | `eacf8f23051dccd2916e8c698f5c515d` | identical |
| `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd_varlen.py`         | `8b3b9cc70d3bcceb05d190f296672de3` | `8b3b9cc70d3bcceb05d190f296672de3` | identical |
| `mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py`             | `258b99b3e9bbcbc4bc770f1aa990024c` | `32c816775e75f695106a6928a90a6610` | **differs** |

### Only real divergence: `mamba3_siso_combined.py`

bench3 has a small local patch tagged "PR #909: cache for checkpoint compat" on
top of europe's version:

```
<<< bench3 (wins)
        _saved = ctx.saved_tensors  # PR #909: cache for checkpoint compat
        if len(_saved) == 0:
        ...
        Final_SSM_State_save, cu_seqlens_save) = _saved
=== europe (older)
        if len(ctx.saved_tensors) == 0:
        ...
        Final_SSM_State_save, cu_seqlens_save) = ctx.saved_tensors
```

Semantically identical — bench3 just caches `ctx.saved_tensors` once to satisfy
the gradient-checkpointing contract (accessing `ctx.saved_tensors` twice on a
recomputed node raises). This is a pure correctness/compat patch, not a perf
patch.

### Other observations

- europe has an **untracked** `mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py.orig`
  (1777 lines). `diff` against the tracked `mamba3_siso_bwd.py` is empty — it is
  a stray backup from a sed/patch run, **not** a meaningful delta. Safe to ignore
  or `git clean -f`.
- bench3 working tree lists `mamba3_siso_combined.py` as modified; europe lists
  the other three files as modified. The three "identical" files above are byte-
  identical between the two working trees even though git reports them as
  modified relative to HEAD — so the unstaged patches are the same patches.

## Canonical decision

**Canonical HEAD = `31f3d7baba69d0ccad1635ace1e477367899e408` (both machines).**

**Canonical working-tree patches = bench3's set.** It is a strict superset of
europe's: identical patches on the three tilelang/modules files, plus the
`_saved = ctx.saved_tensors` checkpoint-compat tweak on `mamba3_siso_combined.py`.

### Path forward

1. On europe, copy bench3's `mamba3_siso_combined.py` in place:
   ```
   gcloud compute scp --zone=LOCATION_1 \
     dave@h200_1:/mnt/data/cppmega-root/state-spaces-mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py \
     /tmp/mamba3_siso_combined.py
   gcloud compute scp --zone=LOCATION_2 /tmp/mamba3_siso_combined.py \
     dave@h200_1:/mnt/data/cppmega-root/state-spaces-mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py
   ```
2. On europe, `rm mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py.orig` (stray backup).
3. Snapshot all four canonical files into a patch series on `main` so the state
   survives loss of either machine — see `.tmp/artifacts/mamba_fork_variants/`
   for the raw bench3 versions already pulled off the VMs.

### Artefact backup

Copies of both variants now live in

```
/Volumes/external/sources/cppmega/.tmp/artifacts/mamba_fork_variants/
├── bench3_mamba3.py                         # canonical
├── bench3_mamba3_mimo_bwd.py                # canonical
├── bench3_mamba3_mimo_bwd_varlen.py         # canonical
├── bench3_mamba3_siso_combined.py           # canonical (wins)
├── europe_mamba3_siso_combined.py           # older; superseded
└── europe_mamba3_siso_bwd.py.orig           # stray backup, ignore
```

## Gotchas

- The "bench3=31f3d7b vs europe=4f4857f" split in the brief is stale; both are
  `31f3d7b` today. Always confirm with `git rev-parse HEAD` before trusting
  machine-tag claims (see `reference_env_drift_bench3_europe.md`).
- The three tilelang/modules files are **identical across machines** even
  though both working trees report them modified vs HEAD. The local patches
  converged at some point via manual copy, so the drift risk is low for those
  files — only `mamba3_siso_combined.py` needs reconciling.
