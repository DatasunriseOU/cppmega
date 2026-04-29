# Mamba3 bwd_bwd P_TILE Probe - 2026-04-29

Branch: `worker/mamba3-bwd-bwd-ptile-probe`

Base: `worker/mamba3-stage2-force-nontma` / `972608d`

Goal: test whether the pressure-heavy Mamba3 MIMO `bwd_bwd` TileLang path can
structurally process `P=128` in `P_TILE=64` slices to reduce live fragment and
shared-memory pressure. This explicitly avoids another `num_stages` sweep and
does not change WS/TMA policy.

## Files

- Patch: `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_bwd_ptile_compile_only.patch`
- Harness: `scripts/modal_mamba3_bwd_bwd_ptile_probe.py`
- Status: `docs/status/mamba3_bwd_bwd_ptile_probe_2026_04_29.md`

## Allocation / GEMM Findings

The current `bwd_bwd` holds several `P`-scaled objects live for productionish
`P=128`:

- `dPhiO_shared [fused_chunk_size, P]`
- `dPsiV_combined_shared [fused_chunk_size, P]`
- local `PsiV_shared [fused_chunk_size, P]`
- `dstates_shared/dstates_frag [N, P]`
- `states_shared/states_frag [N, P]`
- `dPsiV_frag` and `dPsiV_D_fused_frag [fused_chunk_size, P]`
- `dv_frag [chunk_size, P]`, `dPsi_acc/dPsi_frag [R, P]`
- `dPhiO_scaled_frag [fused_chunk_size, P]`

The tempting independent-P paths are real:

- `dPsiV = k @ dstates + lkq @ dPhiO`
- diagonal `D` and `qk_dot` additions to `dPsiV`
- `DV` and `DMIMO_V`
- per-P `dstates` recurrence

But several consumers reduce across the full `P` dimension, so a correct
P-tiled production kernel needs cross-P accumulators before output writes:

- `dqk_from_diag = dPhiO @ PsiV.T`, consumed by `DGAMMA_DIAG`, `DK`, and `DQ`
- `dk_frag = PsiV @ dstates.T`, consumed by `DDA_CS_REV`, `DK`, and `DFACTOR`
- `dk_intrachunk = PsiV @ dPhiO.T`, consumed by `DSSDA` and the intrachunk `DK/DQ`
- `dq_frag = dPhiO @ states.T`, consumed by `DDA_CS` and `DQ`
- scalar reductions over `P`: `DD`, state-passing `DDA`, and related `DDA*`

So `P_TILE=64` is not a local shape substitution. It requires either on-chip
full-shape accumulators for the `P` reductions, temporary global partials, or a
larger structural split of the bwd_bwd algorithm.

## Prototype

The patch adds a separate compile-only function:

- `mamba_mimo_bwd_bwd_ptile_compile_only(..., p_tile=64)`

It does not replace `mamba_mimo_bwd_bwd`. The prototype uses `P_TILE=64` for:

- `dPhiO_shared_tile [fused_chunk_size, P_TILE]`
- `PsiV_shared_tile [fused_chunk_size, P_TILE]`
- `dPsiV_frag_tile/dPsiV_D_fused_frag_tile [fused_chunk_size, P_TILE]`
- `dPsiV_combined_shared_tile [fused_chunk_size, P_TILE]`
- `dstates_shared_tile/dstates_frag_tile [N, P_TILE]`
- `states_shared_tile [N, P_TILE]`
- `v_shared_tile [chunk_size, P_TILE]`

It intentionally leaves full-correctness writes incomplete for `DK`, `DQ`,
`DSSDA`, `DGAMMA_DIAG`, `DDA`, `DDA_CS_REV`, `DDA_CS`, `DFACTOR`, and
`DANGLES`, because those need the cross-P accumulators listed above.

## Local Validation

```text
python -m py_compile scripts/modal_mamba3_bwd_bwd_ptile_probe.py

patch --dry-run -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_stage2_force_nontma.patch
patch -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_stage2_force_nontma.patch
patch --dry-run -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_bwd_ptile_compile_only.patch

python -m py_compile /tmp/.../patched/mamba3_mimo_bwd.py
```

All local checks passed.

## Modal H200 Result

Run:

```text
CPPMEGA_MODAL_GPU=H200:2 \
CPPMEGA_MAMBA3_PTILE_MATRIX='64,0' \
timeout 600 modal run scripts/modal_mamba3_bwd_bwd_ptile_probe.py
```

App:

- `ap-JxpzfPAaBFelWcKkQWzOlW`
- completed and stopped normally

Environment:

- device: `NVIDIA H200`, capability `(9, 0)`, device count `2`
- image: `ghcr.io/jewelmusicee/cppmega:latest`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`
- TileLang: `0.1.8+cu132.gitf309d814`

Compile:

| `P_TILE` | `bb_num_stages` | status | elapsed | notes |
| --- | --- | --- | --- | --- |
| 64 | 0 | compiled | 25.186s | compile-only P_TILE kernel |

Source markers:

- `source_chars`: `39514`
- `launch_bounds`: `[(256, 1)]`
- `tma_load_count`: `0`
- `tma_store_count`: `0`
- `producer_guard`: `false`
- TileLang warning: `[WS] skipped: no TMA copies in pipeline loop`
- layout warnings: swizzle conflicts for `dPhiO_shared_tile` and `dstates_shared_tile`

The no-TMA marker is expected for this prototype because it uses manual
dynamic-`p_start` indexing for P-tile slices rather than large TMA-capable
copies. This run is compile feasibility only, not perf.

## Verdict

P-tiling is technically expressible in TileLang, but this particular direction
is not worth pursuing as a small patch. A correct version needs several new
cross-P accumulation surfaces, and those would either keep large full-shape
on-chip accumulators or introduce extra global-memory partial writes.

The likely better next step is not a direct `P_TILE=64` rewrite of the current
single kernel. If we continue, the work should be framed as an algorithm split:
compute the full-P reduction products (`dqk_from_diag`, `dk_intrachunk`,
`dk_frag`, `dq_frag`, scalar `DDA*`) in a dedicated phase or use a lower-level
kernel where partial reductions can be controlled explicitly. For the current
TileLang bwd_bwd kernel, P-tiling looks too invasive for the expected gain.
