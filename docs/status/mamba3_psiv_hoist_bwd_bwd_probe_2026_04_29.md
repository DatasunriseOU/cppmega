# Mamba3 bwd_bwd PsiV-Hoist Probe - 2026-04-29

Branch: `worker/mamba3-psiv-hoist-bwd-bwd`

Worktree: `/home/dave/source/cppmega/.claude/worktrees/mamba3-psiv-hoist-bwd-bwd`

## Base Status

Requested base was `worker/mamba3-stage2-force-nontma` or commit `972608d`.
In this repo's active git-dir neither ref/object is present, and the existing
`mamba3-stage2-force-nontma` directory has a broken `.git` pointer to a missing
`.git/worktrees/mamba3-stage2-force-nontma` entry. I therefore created this
branch from the available repo HEAD (`d50f970`) and kept the work as a standalone
upstream patch/probe artifact.

The actual upstream source inspected for kernel context was:

`/home/dave/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`

## Source Areas Inspected

Docs:

- `docs/mamba3_mimo_p2_psiv_cache_design.md`
- `docs/mamba3_mimo_p3_register_split_design.md`

Kernel source:

- `mamba_mimo_bwd_fwd_kernel` around the existing `PsiV_frag = v * Psi` producer
  at lines ~225-232.
- `mamba_mimo_bwd_bwd_kernel` around the recompute/live-set region at lines
  ~873-930.
- `mamba_mimo_bwd_combined` around the local `states`/`qk_dot` intermediates and
  the two kernel launches at lines ~1189-1297.

## Exact PsiV Live-Set Finding

In `bwd_bwd`, the relevant block is after `dV` and `dPsi` are computed:

- `v_frag` is loaded from `v_shared` for `dPsi`.
- `Psi_frag` stays live from the head-level MIMO_V load.
- Baseline then allocates `PsiV_frag: [chunk_size, R, P]`, clears it, fills it
  via `v_frag[cs, p] * Psi_frag[r, p]`, allocates `PsiV_shared:
  [fused_chunk_size, P]`, and copies the fragment into shared memory.
- `PsiV_shared` then feeds three downstream consumers:
  `dPhiO @ PsiV^T` for `dqk_from_diag`, `PsiV @ dstates^T` for `dk_frag`, and
  `PsiV @ dPhiO^T` for `dk_intrachunk_frag`.

This means the removable recompute is specifically the `PsiV_frag` fragment and
its multiply loop. `v_frag` and `Psi_frag` cannot be removed entirely because
they are still needed for `dPsi`/`dV`.

## Prototype Patch

Patch:

- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_psiv_hoist_probe.patch`
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_psiv_hoist_after_stage2_probe.patch`

Probe script:

- `scripts/modal_mamba3_psiv_hoist_probe.py`

Patch behavior:

- Adds `PSIV_CACHE: [B, H, S * R, P]` to `mamba_mimo_bwd_fwd_kernel` as an output
  side tensor.
- After `bwd_fwd` builds `PsiV_shared`, stores that row-flattened tile into
  `PSIV_CACHE`.
- Adds `PSIV_CACHE` to `mamba_mimo_bwd_bwd_kernel`.
- Replaces the `bwd_bwd` `PsiV_frag = v_frag * Psi_frag` block with a 2D
  `T.copy(PSIV_CACHE[..., fused_chunk_start:fused_chunk_start+fused_chunk_size, :],
  PsiV_shared)`.
- Allocates `psiv_cache = torch.empty([B, H, S * R, P], dtype=v.dtype)` inside
  `mamba_mimo_bwd_combined` and passes it from `bwd_fwd` to `bwd_bwd`.

This is deliberately a local backward-pass intermediate, not a forward
saved-tensor API change. It avoids changing the public autograd wrapper for the
first probe and does not enable bwd_bwd WS/TMA.

The `after_stage2` patch is the same PsiV change as an incremental patch on top
of `mamba3_bwd_stage2_force_nontma.patch`. The Modal script defaults to this
mode and mounts the stage2 patch from the sibling worktree because the requested
stage2 branch/ref is not present in this repo's active git-dir.

## Local Validation

Commands run:

```text
python -m py_compile scripts/modal_mamba3_psiv_hoist_probe.py
python scripts/modal_mamba3_psiv_hoist_probe.py --local-dry-run --patch-mode standalone
python scripts/modal_mamba3_psiv_hoist_probe.py --local-dry-run --patch-mode after_stage2
```

Result:

```text
standalone patch_rc: 0
after_stage2 base patch_rc: 0
after_stage2 psiv patch_rc: 0
```

The current local Python environment does not import `mamba_ssm.ops.tilelang`,
so local kernel compile/smoke was not possible outside Modal.

## Modal Status

Command run:

```text
CPPMEGA_MODAL_GPU=H200:1 timeout 10m modal run scripts/modal_mamba3_psiv_hoist_probe.py
CPPMEGA_MODAL_GPU=H200:1 CPPMEGA_PSIV_PATCH_MODE=after_stage2 timeout 15m modal run scripts/modal_mamba3_psiv_hoist_probe.py
```

App:

- Standalone run URL: `https://modal.com/apps/jewelmusic/main/ap-s9lPDBc2gJBTJNjxhLvdJA`
- Stage2+PsiV run URL: `https://modal.com/apps/jewelmusic/main/ap-n9x4x85dCLMfOBU17cyBXp`
- Status: completed; app stopped after local entrypoint completed.

Environment:

- GPU: `NVIDIA H200`, capability `(9, 0)`, device count `1`
- Image: `ghcr.io/jewelmusicee/cppmega:latest`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`

Shape:

```text
B=1, S=64, H=4, G=1, N=64, P=64, R=4, chunk=16
```

Stage2+PsiV compile/smoke:

- Stage2 baseline `mamba_mimo_bwd_fwd_kernel` compiled.
- Stage2 baseline `mamba_mimo_bwd_bwd_kernel` compiled.
- Stage2+PsiV `mamba_mimo_bwd_fwd_kernel` compiled.
- Stage2+PsiV `mamba_mimo_bwd_bwd_kernel` compiled.
- Stage2 patch and incremental PsiV patch both applied in the container with
  `patch_rc: 0`.
- Status: `smoke_ok`.

Correctness vs baseline:

All returned non-None tensors were bit-exact:

- `dQ`, `dK`, `dV`
- `dADT`, `dDT`, `dTrap`
- `dQ_bias`, `dK_bias`
- `dMIMO_V`, `dMIMO_Out`
- `dAngles`

`dMIMO_Z`, `dD`, and `dZ` were `None` for this reduced no-Z/no-D smoke shape.

Reduced-shape timing:

| Variant | Warmup | Iters | Mean ms |
| --- | ---: | ---: | ---: |
| stage2 baseline | 2 | 6 | 0.3939 |
| stage2+psiv_hoist | 2 | 6 | 0.3869 |

This timing is only a smoke-level signal. The shape is too small to infer
productionish throughput, but it confirms the extra cache tensor does not create
an obvious reduced-shape regression.

Notes:

- A first `after_stage2` Modal attempt (`ap-IQhqVhIEyJx035UItsMcF8`) was
  interrupted during NVCC compilation and stopped by the CLI before JSON output.
  The repeat run above completed.
- The tiny shape emitted `[WS] skipped: no TMA copies in pipeline loop` warnings
  during the stage2 patched compiles. This is acceptable for the smoke because
  the target here was not to enable bwd_bwd WS/TMA.

## Blockers / Risks

- The branch is not actually based on `972608d` because that object is absent
  from the available git-dir.
- The standalone patch is against the currently available
  `/home/dave/state-spaces-mamba` source. The `after_stage2` patch was
  dry-run-tested and Modal-tested after `mamba3_bwd_stage2_force_nontma.patch`.
- The prototype trades `bwd_bwd` fragment/multiply live range for a global
  memory round trip written by `bwd_fwd` and read by `bwd_bwd`. This only wins if
  bwd_bwd register pressure/spill relief is larger than the extra bandwidth and
  bwd_fwd store cost.
- It does not remove `v_frag` or `Psi_frag`; those remain necessary for dV/dPsi.
- A production patch still needs a decision on whether to keep this as a local
  backward intermediate or promote a fuller fwd/bwd saved PsiV cache API.

## Worth Production Work?

Tentative: worth one productionish H200 A/B, not yet worth production
integration. The change is small, compiles on H200, is bit-exact on the reduced
shape, and attacks the exact P3-identified live range. The likely gain is still
bounded because only `PsiV_frag` and the multiply loop are removed while
`v_frag`, `Psi_frag`, and `PsiV_shared` remain, and the patch adds a global
memory write/read of `[B, H, S * R, P]`. Production work should require a
productionish shape run layered on the stage2 force-nonTMA patch and a
neutral-to-positive end-to-end timing result.
