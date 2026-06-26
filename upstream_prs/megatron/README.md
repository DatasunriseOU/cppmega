# Megatron Patch Artifacts

These patches were recovered from the Hetzner backup host under:

`/data/gs-nam56r-artifacts/backups/europe_2026_04_14/megatron_unpushed_patches/`

They preserve the previously used cppmega Megatron deltas after the old
external `jewelmusicee/Megatron-LM` source path became unavailable.

## Files

- `0001-merge-PR-3674-DSA-absorbed-MLA-TileLang-fused-sparse.patch`
  - Adds the DSA absorbed-MLA TileLang sparse stack.
  - Patch header base includes `dsa.py` blob `5c5f77363`.
  - The matching local Megatron history point is around commit `b0eb9143`
    (`Fix several bugs for DSA rope and spec. (#3026)`), not current
    `core_v0.16.0`.
- `0002-cherry-pick-PR-4268-delayed-wgrad-overlap-with-P2P-b.patch`
  - Adds delayed wgrad overlap controls for PP/P2P backward overlap.

## Current Docker Status

Do not apply these patches blindly in `docker/Dockerfile`.

`git apply --check` against `NVIDIA/Megatron-LM@core_v0.16.0` fails for the
PR3674 patch because that release does not have the same DSA/MLA base files.
The Docker image must either:

1. pin a Megatron commit compatible with these patch files, or
2. move to an upstream Megatron ref where the needed DSA sparse stack already
   exists, or
3. carry a refreshed patch generated against the selected Docker Megatron ref.

