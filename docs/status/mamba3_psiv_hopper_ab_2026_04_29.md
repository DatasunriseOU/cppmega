# Mamba3 Hopper Hoist-PsiV A/B - 2026-04-29

Branch: `worker/mamba3-hopper-psiv-ab`

Base: `worker/mamba3-psiv-modal-dryrun` at `ed31598`

Goal: measure Hoist-PsiV on the current production non-TMA/non-warpspec
TileLang Mamba3 MIMO path. Production defaults stay unchanged.

## Implementation

Added an explicit temp-source patch helper:

- `patch_source_tree_for_hopper_psiv_ab(root)`
- patches only a caller-provided copied `.../mamba_ssm/ops/tilelang/mamba3`
  tree;
- `mamba3_mimo_fwd.py`: adds `PSI_V_OUT` to the regular fwd kernel and writes
  the already-computed `PsiV_frag`;
- `mamba3_mimo_bwd.py`: adds `PSI_V_IN` to regular `bwd_fwd` and `bwd_bwd`;
- `bwd_fwd`: loads `PsiV_in` into `PsiV_shared` instead of recomputing `V *
  Psi`;
- `bwd_bwd`: keeps `V` and `MIMO_V` for `dV`/`dMIMO_V`, but loads `PsiV_in`
  for the later `dqk`/`dK` paths.

`apply_all()` remains a production refusal; no env gate or default path now
applies this patch.

## Modal Harness

Script: `scripts/modal_mamba3_psiv_hopper_ab.py`

The harness pulls GHCR, overlays local `cppmega/` and source
`state-spaces-mamba/mamba_ssm`, copies the source tree inside the container,
patches the copy, then imports baseline and patched modules from different
source roots. It measures:

- raw fwd baseline vs fwd with `PsiV_out`;
- precomputed `torch.mul(..., out=psi_v)` write cost;
- raw `bwd_fwd` baseline vs `PsiV_in`;
- raw `bwd_bwd` baseline vs `PsiV_in`;
- correctness for fwd output, `PsiV_out`, bwd_fwd intermediates, and selected
  bwd_bwd grads.

Shape: `prod_mbs4`: `B=4 S=4096 H=32 G=1 N=64 P=128 R=4 chunk=16`, bf16.
Cache size: `512 MiB/layer`.

Commands run:

```bash
CPPMEGA_MODAL_GPU=H100:2 CPPMEGA_PSIV_AB_SHAPE=prod_mbs4 \
  CPPMEGA_PSIV_AB_ITERS=8 CPPMEGA_PSIV_AB_WARMUP=3 \
  modal run scripts/modal_mamba3_psiv_hopper_ab.py

CPPMEGA_MODAL_GPU=H200:2 CPPMEGA_PSIV_AB_SHAPE=prod_mbs4 \
  CPPMEGA_PSIV_AB_ITERS=8 CPPMEGA_PSIV_AB_WARMUP=3 \
  modal run scripts/modal_mamba3_psiv_hopper_ab.py
```

The first H100/H200 runs used the intended default shape, but Modal did not
propagate the local `CPPMEGA_PSIV_AB_ITERS=8` env var into the remote function,
so the recorded outputs used the script default of `10` iterations. The script
now passes shape/warmup/iters explicitly to `run_probe.remote(...)`.

Note: the first two runs used `ghcr.io/jewelmusicee/cppmega:latest` while the
image repo code was stale (`f6c15a2`), but this harness overlays local
`cppmega/` and local `state-spaces-mamba/mamba_ssm`; the relevant patch code
and source under test were not taken from the stale image. The base dependency
stack still reported TileLang `0.1.8+cu132.gitf309d814` behavior.

Attempted rerun with `GHCR_TAG=785c3fd` after main was pushed, but the tag was
not published yet:

```text
reading manifest 785c3fd in ghcr.io/jewelmusicee/cppmega: manifest unknown
```

## Results

All correctness checks were bit-identical (`max_abs=0`, `max_rel=0`) on both
H100 and H200 for:

- fwd output;
- fwd-written `PsiV_out` vs precomputed `PsiV`;
- `bwd_fwd` `dmimo_o`, `states`, `qk_dot`;
- `bwd_bwd` `dq`, `dk`, `dv`, `dmimo_v`.

Per-layer CUDA-event means:

| GPU | PsiV write | fwd write delta | bwd_fwd saved | bwd_bwd saved | bwd saved total | Net vs write |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | 0.778 ms | 0.215 ms | 0.015 ms | 0.095 ms | 0.109 ms | -0.669 ms |
| H200 | 0.784 ms | 0.189 ms | 0.007 ms | 0.052 ms | 0.059 ms | -0.724 ms |

For 9 Mamba3 layers/rank:

| GPU | PsiV write/rank | fwd write delta/rank | bwd saved/rank | Net vs write/rank |
| --- | ---: | ---: | ---: | ---: |
| H100 | 7.00 ms | 1.93 ms | 0.98 ms | -6.02 ms |
| H200 | 7.05 ms | 1.70 ms | 0.53 ms | -6.52 ms |

Full H200 timing detail:

- fwd baseline: `1.331 ms`; fwd+write: `1.520 ms`
- `bwd_fwd` baseline: `1.878 ms`; PsiV-in: `1.871 ms`
- `bwd_bwd` baseline: `3.709 ms`; PsiV-in: `3.657 ms`
- precomputed write proxy: `0.784 ms`

## Decision

No-go for production Hoist-PsiV on the current Hopper non-TMA TileLang Mamba3
MIMO path.

The patched kernels are correct, but the saved `bwd_fwd`/`bwd_bwd` time is far
smaller than the PsiV write cost and memory pressure. On the target H200 shape,
the best measured bwd saving is `0.059 ms/layer`, while writing the 512 MiB
cache costs `0.784 ms/layer` as a standalone write or `0.189 ms/layer` as an
actual fwd-kernel delta. This is negative even before accounting for the 4.5
GiB/rank activation footprint across 9 layers at MBS=4.

Do not wire this into the production env gate. The useful artifact is the A/B
harness and the proof that simple PsiV hoisting is not enough; any future work
needs a larger bwd restructuring than replacing the local `V * Psi` multiply
with a gmem load.

## References

- HF kernels card `kernels-community/mamba-ssm`: useful reference package, but
  broad dependency scope and no direct Mamba3 MIMO TileLang drop-in:
  https://huggingface.co/kernels-community/mamba-ssm
- Together Mamba-3 blog: confirms Mamba-3 kernels use Triton, TileLang, and
  CuTe DSL; MIMO prefill uses TileLang for explicit memory hierarchy control:
  https://www.together.ai/blog/mamba-3
