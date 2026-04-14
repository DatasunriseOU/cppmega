# Reproducer: Mamba3 DT fp32 cast + MIMO GQA backward branch

> **Scope note.** This reproducer exercises **two distinct upstream bugs**
> that co-trigger in the same Megatron + Mamba3 training config, so it is
> kept as one pack for convenience. The bugs target **different
> repositories** and are filed as two separate PRs:
>
> - **PR 05** (`state-spaces/mamba`) — the Mamba3-side MIMO GQA backward
>   branch (`G value of 2 not supported`). The `gqa_unpatched` /
>   `gqa_patched` stages below exercise this.
> - **PR 16** (`NVIDIA/Megatron-LM`) — the `Float16Module` silent cast of
>   Mamba3's fp32-contract parameters (DT/D/bias). The `bf16` / `fp32`
>   stages below exercise this. See `docs/upstream_bugs.md:187` for the
>   Megatron-side writeup.
>
> When filing upstream, split the evidence by stage: bf16/fp32 → PR 16,
> gqa_unpatched/gqa_patched → PR 05.

The local convenience patch `upstream_prs/05_mamba3_dt_fp32_gqa_bwd.patch`
currently bundles both fixes because they co-trigger in the same training
lane, but upstream filing should split them by repository as noted above.

The shared reproducer validates both bug/fix pairs:

1. **Problem 1 (DT dtype)** — `mamba_ssm/modules/mamba3.py` calls
   `F.softplus(dd_dt + self.dt_bias)`. When Megatron's `Float16Module`
   casts the `dt_bias` parameter to bf16, the returned `DT` tensor is
   bf16. The TileLang MIMO forward kernel declares
   `DT: T.Tensor([B, H, S], T.float32)` (fp32 only), so the call raises
   a low-level TVM-FFI dtype error or (depending on which argument is
   validated first) silently corrupts the result.

   Fix: `DT = F.softplus((dd_dt + self.dt_bias).to(torch.float32))`
   (applied at both softplus call sites in `mamba3.py`).

2. **Problem 2 (GQA backward branch missing)** —
   `mamba_mimo_bwd_combined` in
   `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py` only handles
   `G == 1` (MHA) and `G == H` (per-head). Any intermediate GQA shape
   (`1 < G < H` with `H % G == 0`, e.g. NAM56R's `ngroups=8,
   nheads=128`) hits `else: raise ValueError("G value of {G} is not
   currently supported!")`.

   Fix: add an `elif H % G == 0:` branch that reduces `dq`/`dk` via
   `view(B, S, R, G, hpg, N).sum(dim=4)`.

## Parameters that trigger the bugs

- **DT dtype bug:** any call with `DT.dtype != torch.float32`. The
  reproducer exercises this by constructing `DT` as bf16 directly.
- **GQA backward bug:** `nheads_qk` (= kernel's `G`) in the half-open
  range `(1, nheads)` with `nheads % nheads_qk == 0`. The reproducer
  uses `G=2, H=16`. Setting `G=1` or `G=H` hides the bug.

The default config (edit top of `reproducer.py` to tweak):

```
B=1, S=256, R=4, G=2, H=16, N(headdim_qk)=32, P(headdim_v)=64, chunk_size=16
```

## Run

```bash
pip install -r requirements.txt            # + mamba-ssm fork (see below)
python reproducer.py
```

Requires:
- CUDA-capable device (validated on **NVIDIA H200** `sm_90a`).
- mamba-ssm installed from the **state-spaces/mamba** fork (Dao AI Lab /
  Goombalab) with the TileLang MIMO kernels — the PyPI wheel
  `mamba-ssm==2.3.1` does **not** ship `mamba_ssm.ops.tilelang` or
  `mamba_ssm.modules.mamba3` and will fail at import.
- The reproducer assumes the installed bwd source is **already in the
  patched form**. It flips that file between patched and unpatched on
  disk to exercise both paths, then restores the patched state in a
  `finally` block. If your install has the upstream (unpatched) source,
  either apply `05_mamba3_dt_fp32_gqa_bwd.patch` first or invert the
  `PATCHED_GQA_BLOCK` / `UNPATCHED_RAISE_LINE` literal swap in the
  reproducer.

## Expected output — bug reproduced, fixes validated

```
Device: NVIDIA H200
Config: B=1 S=256 R=4 G(nheads_qk)=2 H(nheads)=16 N=32 P=64
    -> GQA: 1 < G=2 < H=16 and H % G == 0   (trips the branch upstream does not implement)

========================================================================
Subprocess stage: bf16
========================================================================
[stage bf16] Building inputs with dt_dtype=bfloat16...
[stage bf16] RESULT: kernel refused bf16 DT with: RuntimeError: kernel mamba_mimo_fwd_kernel input DA_CS dtype expected float32, but got bfloat16
STAGE_RESULT=bf16_refused

========================================================================
Subprocess stage: fp32
========================================================================
[stage fp32] Building inputs with dt_dtype=float32...
[stage fp32] RESULT: finite grads. max|grad|: Q=3.359e+00, K=5.438e+00, V=2.088e+01,
                                                DT=1.962e+02, ADT=4.068e+01,
                                                Q_bias=1.862e+01, K_bias=1.688e+01,
                                                MIMO_V=2.064e+01, D=2.002e+01
STAGE_RESULT=fp32_ok

========================================================================
Subprocess stage: gqa_unpatched
========================================================================
[stage gqa_unpatched] RESULT: ValueError raised as expected: G value of 2 is not currently supported!
STAGE_RESULT=unpatched_raised

========================================================================
Subprocess stage: gqa_patched
========================================================================
[stage gqa_patched] RESULT: finite grads. max|grad|: Q=3.359e+00, K=5.438e+00, V=2.088e+01,
                                                      DT=1.962e+02, ADT=4.068e+01,
                                                      Q_bias=1.862e+01, K_bias=1.688e+01,
                                                      MIMO_V=2.064e+01, D=2.002e+01
STAGE_RESULT=patched_ok

========================================================================
Summary:
  stage bf16               -> bf16_refused
  stage fp32               -> fp32_ok
  stage gqa_unpatched      -> unpatched_raised
  stage gqa_patched        -> patched_ok

Problem 1: BUG_REPRODUCED (bf16 DT rejected by TileLang kernel) ; FIX_VALIDATED (fp32 DT fwd+bwd produces finite grads)
Problem 2: BUG_REPRODUCED (unpatched GQA bwd raises ValueError) ; FIX_VALIDATED (patched GQA bwd produces finite grads)
```

Exit code: **0** (both bugs reproduced AND both fixes validate).

## Why the stages run in subprocesses

A TileLang kernel that fails mid-launch (the bf16 stage) leaves the
CUDA context in a state where *any* subsequent kernel launch on the
same process can return `cudaErrorMisalignedAddress` (seen on GB10 and
H200). To keep each stage independent, the reproducer spawns a fresh
Python subprocess per stage via `MAMBA3_REPRO_STAGE=<name>`.

## Stack used to validate (2026-04-14)

- Machine: `h200_1` (H200, sm_90a, 2×GPU — uses GPU0).
- torch: `2.12.0.dev20260410+cu132`
- mamba-ssm: Dao AI Lab fork at
  `/mnt/data/venv/lib/python3.13/site-packages/mamba_ssm` (commit as
  shipped on bench3 with the `05` patch already applied).
- tilelang: dev build from `/mnt/data/tilelang-build`.

GB10 currently cannot run this reproducer because of an unrelated
TileLang `LayoutInference` FloorMod constant-folding crash when the
`mamba_mimo_bwd_bwd` kernel is compiled (see pack 13 for the minimal
repro; see `docs/mamba3_mimo_p1_notes.md` for the broader P1 blocker
writeup). Problem 1 (bf16 DT, `STAGE_RESULT=bf16_refused`) *does*
reproduce on GB10 since it fails inside the forward kernel and never
reaches bwd compile.

## Related, but NOT the same bug

- Mamba3 B/C layout `(r,g,n)` vs `(g,r,n)` latent bug, only triggered
  at TP>1 with `ngroups>1`. This reproducer is about `dt` dtype and
  the `dq`/`dk` *reduction* shape in the backward — different tensors,
  different code path, different fix. See `upstream_prs/05_mamba3_dt_fp32_gqa_bwd.md`
  § "Not in scope" for the layout-bug discussion.

## Files

- `reproducer.py` — self-contained CUDA test; subprocess-isolated stages.
- `requirements.txt` — versions.
- `README.md` — this file.
