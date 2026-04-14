# Megatron-LM restoration recipe — bench3 tarball

Date: 2026-04-14
Source of truth: `sftp://BUCKET_ARTIFACTS/backups/backup_bench3_2026_04_14/megatron_lm_tree.tar.gz` (11.7 MB compressed, ~56 MB uncompressed).

## Why this doc exists

`/mnt/data/cppmega-root/megatron-lm/` on bench3 was captured at 2026-04-14 11:15 UTC
as a **flat tree without `.git`**. If bench3 is rebuilt from scratch, the only way
to restore an equivalent environment is:

1. Clone fresh upstream NVIDIA/Megatron-LM
2. Check out the correct base commit/branch
3. Re-apply our locally-maintained patches on top
4. `pip install -e .` from the result

The tarball (`megatron_lm_tree.tar.gz` in the GS bucket) is the authoritative
reference for diffing if any patch re-application fails.

---

## Base version identification (best effort, no .git on bench3)

### What the tarball says about itself

| Source | Value |
|---|---|
| `megatron/core/package_info.py` | `MAJOR=0, MINOR=16, PATCH=0, PRE_RELEASE='rc0'` → `0.16.0rc0` |
| `megatron_core.egg-info/PKG-INFO` | `megatron-core 0.16.0rc0` (installed version) |
| `.coderabbit.yaml` mtime | 2026-04-10 00:06 UTC |
| Present files | `gpt_builders.py`, `mamba_builders.py`, `pretrain_mamba.py`, `uv.lock`, `greptile.json`, `megatron/core/transformer/experimental_attention_variant/` subtree |
| Missing files | no `CHANGELOG` on disk, no `.git` directory |

### Cross-check from europe

Europe's megatron-lm (at `/mnt/data/cppmega-root/megatron-lm/`) **does** have a live
`.git`. Its HEAD at backup time (2026-04-14) was:

```
ec6a9e900 cherry-pick: PR #4268 — delayed wgrad overlap with P2P backward (PP>1 A2A overlap)
2eeabc668 merge: PR #3674 — DSA absorbed MLA + TileLang fused sparse ops
(parent from origin/dev)
1fc92009b [Dev] Fix TE version check for retain_pinned_cpu_buffers in cpu offload (#4266)
c0c4fdc45 [Dev] Paged Stashing (#2690)
0f6fcb0c5 [dev] fix(ssm): handle alignment padding in GDN packed seq + CP (#4230)
```

Branch on europe: `dev_latest` (our local branch; 2 commits ahead of
`origin/dev`). Europe is tracking NVIDIA's upstream **`dev`** branch, not a
release tag.

### Upstream commit hash (bench3)

**UNKNOWN — best effort reconstruction from tarball.**

The bench3 tree was installed separately from europe (no git, different directory
structure, egg-info shows 0.16.0rc0). Empirical diffing of the bench3 tree
against europe's ec6a9e900 tree (in the backup) is the only way to know for sure
whether bench3 corresponds to:

- the same `ec6a9e900` as europe (europe's `dev_latest` HEAD with both our cherry-picks applied), OR
- the `2eeabc668` parent (only PR #3674 applied, not PR #4268), OR
- an earlier `origin/dev` commit.

**Working hypothesis**: bench3 was installed from `2eeabc668` (PR #3674 only),
because bench3 logs reference the DSA sparse MLA path (shipped in PR #3674) but
PR #4268 (delayed wgrad overlap, PP>1) has never been exercised on bench3
(bench3 always runs PP=1). This is a hypothesis, not a confirmed fact — diff
required.

### How to confirm (procedure)

On a machine with network:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git /tmp/megatron-ref
cd /tmp/megatron-ref

# Try the two candidate commits in turn
for ref in ec6a9e900 2eeabc668 origin/dev; do
    git checkout -- .
    git checkout "$ref" 2>/dev/null || continue
    # Extract bench3 tarball parallel
    mkdir -p /tmp/bench3_mlm && tar -xzf megatron_lm_tree.tar.gz -C /tmp/bench3_mlm/
    # Diff
    diff -r megatron/core /tmp/bench3_mlm/megatron-lm/megatron/core \
        --exclude=__pycache__ --exclude=*.pyc --brief | head -40
done
```

Whichever `ref` produces the shortest diff is the bench3 base.

---

## Our locally-maintained patches

Both `0001-merge-PR-3674-DSA-absorbed-MLA-TileLang-fused-sparse.patch` and
`0002-cherry-pick-PR-4268-delayed-wgrad-overlap-with-P2P-b.patch` are available
in the **europe backup** at:

```
sftp://BUCKET_ARTIFACTS/backups/backup_europe_2026_04_14/europe_backup_2026_04_14/megatron_unpushed_patches/
```

Files touched (per `0001-*.patch` stat):
- `megatron/core/transformer/experimental_attention_variant/dsa.py` (+2400)
- `megatron/core/transformer/experimental_attention_variant/ops/indexer.py` (+95)
- `megatron/core/transformer/experimental_attention_variant/ops/sparse_mla.py` (+48)
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_indexer_bwd.py` (+239)
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_indexer_fwd.py` (+217)
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_bwd.py` (+494)
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_fwd.py` (+353)
- `megatron/core/transformer/multi_latent_attention.py` (+178)
- `megatron/core/transformer/transformer_config.py` (+17)

### Additional FP8 sparse_mla files unique to europe

Europe has two uncommitted new files (tracked as `Untracked files` in git status):

- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_bwd_fp8.py`
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_fwd_fp8.py`
- `megatron/core/transformer/experimental_attention_variant/ops/__init__.py` (new)

These exist **only on europe** and are captured in
`europe_backup_2026_04_14/europe_megatron_modified.tar.gz`. They are NOT present
in the bench3 tarball. Port to bench3 by extracting that tarball if FP8 sparse
MLA path is desired on bench3.

### Europe-specific runtime edits (uncommitted)

Europe `git status` (2026-04-14) shows these tracked files locally modified on top
of `ec6a9e900`:

- `megatron/core/transformer/experimental_attention_variant/dsa.py`
- `megatron/core/transformer/experimental_attention_variant/ops/sparse_mla.py`
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_bwd.py`
- `megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_fwd.py`
- `megatron/core/transformer/transformer_layer.py`

Captured diff: `europe_backup_2026_04_14/megatron-lm_diff.patch`. These are iterative
debug edits; decide on a per-file basis which should be promoted upstream.

---

## Restoration recipe (bench3-equivalent)

```bash
# 1. Clone upstream
git clone https://github.com/NVIDIA/Megatron-LM.git ~/megatron-lm
cd ~/megatron-lm

# 2. Check out base — UPDATE HASH after running diff procedure above
git checkout 2eeabc668       # working hypothesis (PR #3674 only)
# OR
# git checkout ec6a9e900     # if bench3 also has PR #4268

# If hashes not reachable from upstream (our branches), add our fork:
git remote add cppmega-fork git@github.com:DatasunriseOU/megatron-lm-fork.git  # if it exists
git fetch cppmega-fork

# 3. Reinstall pip package
pip install -e .

# 4. If diff shows bench3 had additional changes beyond the chosen ref,
#    overlay them from the tarball:
curl -L "https://storage.googleapis.com/BUCKET_ARTIFACTS/backups/backup_bench3_2026_04_14/megatron_lm_tree.tar.gz" \
    -o /tmp/megatron_lm_tree.tar.gz
mkdir /tmp/bench3_mlm
tar -xzf /tmp/megatron_lm_tree.tar.gz -C /tmp/bench3_mlm/
# Diff and selectively copy over
diff -r megatron /tmp/bench3_mlm/megatron-lm/megatron --brief | grep -v __pycache__
cp -r /tmp/bench3_mlm/megatron-lm/megatron/. megatron/   # LAST RESORT — overlay everything

# 5. Verify version
python -c "from megatron.core.package_info import __version__; print(__version__)"
# Expected: 0.16.0rc0
```

## Restoration recipe (europe-equivalent)

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git ~/megatron-lm
cd ~/megatron-lm
git checkout dev
git am /path/to/megatron_unpushed_patches/0001-merge-PR-3674-DSA-absorbed-MLA-TileLang-fused-sparse.patch
git am /path/to/megatron_unpushed_patches/0002-cherry-pick-PR-4268-delayed-wgrad-overlap-with-P2P-b.patch

# Untracked FP8 sparse_mla + runtime edits:
tar -xzf europe_megatron_modified.tar.gz -C /tmp/emm
cp /tmp/emm/megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_{fwd,bwd}_fp8.py \
   megatron/core/transformer/experimental_attention_variant/ops/
cp /tmp/emm/megatron/core/transformer/experimental_attention_variant/ops/__init__.py \
   megatron/core/transformer/experimental_attention_variant/ops/__init__.py
# For the 5 runtime-modified tracked files, apply europe's diff.patch:
git apply megatron-lm_diff.patch   # from europe_backup_2026_04_14/

pip install -e .
```

---

## Authoritative source-of-truth locations

- **bench3 full tree** (no git): `sftp://BUCKET_ARTIFACTS/backups/backup_bench3_2026_04_14/megatron_lm_tree.tar.gz`
- **europe full backup** (with patches + git hashes): `sftp://BUCKET_ARTIFACTS/backups/backup_europe_2026_04_14/`
- **europe unpushed patches** (format-patch format-patch): `europe_backup_2026_04_14/megatron_unpushed_patches/`
- **europe megatron modified**: `europe_backup_2026_04_14/europe_megatron_modified.tar.gz`

Until a proper fork repo exists on GitHub, these tarballs ARE the source of truth.
Confirm that bench3 has git properly initialized (`git init && git remote add …`) as
part of next rebuild — we should never be in a "no .git" state again.

## Open actions

- [ ] Run the 3-way diff procedure on a machine with network to pin the exact upstream base commit for bench3
- [ ] Initialize a fork repo on GitHub (e.g. `DatasunriseOU/megatron-lm-fork`) and push both cherry-picks + europe runtime edits so the recipe above doesn't rely on `git am` from a backup tarball
- [ ] Once fork exists, document exact commit hash of bench3 install in this file
