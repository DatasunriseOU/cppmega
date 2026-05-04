# Mamba3 Monolithic CUDA Chunk Wave 10 Final Summary - 2026-04-30

Status: evidence
Canonical: none
Date: 2026-04-30
Scope: Final branch decision for `worker/mamba3-mono-cuda-chunk`.

Branch: `worker/mamba3-mono-cuda-chunk`

## Final Decision

Keep this CUDA branch as reference/profiling material only.  Do not merge any
monolithic CUDA chunk implementation from this branch as a production path.

Wave 9 already closed the branch on measured evidence: the best final-output
CUDA variant is Wave 8 tile-stream WMMA at `11.1550079981486 ms` on H200, while
the existing TileLang full `bwd_bwd` baseline is `3.70674 ms`.  That leaves the
CUDA path `3.009385065623324x` slower even before considering integration risk.

The CUDA artifacts are still useful for:

- correctness scaffolding against the bf16-staged torch reference;
- profiling ownership shapes, register pressure, smem pressure, and launch
  decomposition;
- explaining why the monolithic WMMA chunk-owner class should not be pursued
  as the NAM56R production Mamba3 backward.

They should not be wired into production configs, default launch paths, or
NAM56R full-boundary runs.

## Evidence To Keep

- Wave 9 closure: `docs/status/mamba3_mono_cuda_chunk_closure_wave9_2026_04_30.md`
- Best final-output CUDA implementation:
  `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8.py`
- Wave 8 Modal harness:
  `scripts/modal_mamba3_mono_cuda_chunk_wave8.py`
- Earlier Wave 1-8 notes under `docs/status/` remain historical evidence only.

## Optional H100 Sanity

Wave 10 added no algorithm and needs no new GPU run.  If a future reviewer wants
a small local sanity check, use H100 only:

```text
CUDA_VISIBLE_DEVICES=0 \
python -c 'import torch; name=torch.cuda.get_device_name(0); assert "H100" in name, f"expected H100, got {name}"; print(name)'

CUDA_VISIBLE_DEVICES=0 \
TORCH_CUDA_ARCH_LIST=9.0 \
RR_DIAG_CUDA_EXT_SUFFIX=local_h100_wave10_final_smoke \
python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8.py \
  --shape smoke \
  --device cuda \
  --iters 1 \
  --warmup 0
```

Do not spend H200 on this branch.  New CUDA scheduling variants are out of
scope.

## Wave 10 Scope

This finalization is docs/reference only:

- no new CUDA kernels;
- no new algorithmic variants;
- no production merge recommendation;
- no H200 run.
