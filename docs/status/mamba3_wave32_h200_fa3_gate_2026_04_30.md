# Mamba3 Wave32 H200 FA3 Gate - 2026-04-30

## Scope

Worktree: `worker/mamba3-wave32-h200-20step-gate`

Base: `main` at `abb04a2` (`fix(mamba3): add guarded grouped-head backward support`)

The H200 production policy is explicit:

- H100/H200 MLA production requires FA3.
- B200/GB10 MLA production requires the patched FA4 path.
- TE `FusedAttention` fallback is diagnostic-only and must not be reported as the production gate.

## Harness Changes

Added `scripts/modal_mamba3_wave32_h200_20step_gate.py`.

Key behavior:

- `launch_gate` defaults to production `te_flash_full`, not `fallback_auto_full`.
- H100/H200 production gate runs a backend probe first and returns `IMAGE_BACKEND_BLOCKED` if FA3 is missing or unusable.
- B200/GB10 policy is represented as patched-FA4-required for future reuse.
- `fallback_auto_full` remains diagnostic-only.
- Variant workdirs are isolated so baseline and stage2 cannot share checkpoints.
- Stage2 composition order is stage2 patch first, grouped-head patch second.
- Local artifacts are under `artifacts/mamba3_wave32_h200_20step_gate/`.

## Commands

Backend probe:

```bash
modal run scripts/modal_mamba3_wave32_h200_20step_gate.py::launch_backend_preflight \
  --run-id wave32_h200_backend_preflight_20260430
```

Production FA3 gate:

```bash
modal run scripts/modal_mamba3_wave32_h200_20step_gate.py::launch_gate \
  --run-id wave32_h200_fa3_prod_gate_v2_20260430 \
  --train-iters 20 \
  --case-label te_flash_full \
  --timeout-per-variant-s 14400
```

Fallback diagnostic, not production:

```bash
modal run scripts/modal_mamba3_wave32_h200_20step_gate.py::launch_debug_sweep \
  --run-id wave32_h200_fallback_diagnostic_20260430 \
  --train-iters 1 \
  --cases fallback_auto_full \
  --timeout-per-case-s 3600 \
  --include-stage2
```

## Production Gate Result

Data source: not resolved. The production gate was blocked by image backend policy before dataset resolution.

| gate | gpu | required backend | FA3 | FA4 | installed flash-attn | status |
| --- | --- | --- | --- | --- | --- | --- |
| `te_flash_full` | `H200:2` / sm90 | FA3 | missing | missing | `flash-attn 2.8.3` | `IMAGE_BACKEND_BLOCKED` |

No production baseline/stage2 training rows exist. This is intentional: without FA3, an H200 MLA run would fall back to TE `FusedAttention`, which is not the production backend.

## Fallback Diagnostic

Data source: `synthetic_full_shape_mock_data`.

The mounted volumes did not contain a ready Megatron `.bin/.idx` + HF tokenizer pair. The harness found:

- `/vol/mock_data`
- `/data_vol/parquet`
- `/data_vol/parquet/clang_semantic_4k_v10_concat`

Diagnostic TE backend selection:

- `flash_attn_version`: `2.8.3`
- `flash_attn_3_version`: not installed
- `flash_attn_4_version`: not installed
- FlashAttention 2 rejected for MLA
- TE selected `FusedAttention (sub-backend 1)`

| class | variant | backend | status | steps | peak alloc GiB | peak reserved GiB | final lm loss | grad norm | note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| diagnostic | grouped-head baseline | TE FusedAttention | ok | 1 | 54.864 | 55.668 | 11.406640 | NaN | Reached `mamba_mimo_bwd_fwd_kernel` and `mamba_mimo_bwd_bwd_kernel`; not production. |

The diagnostic stage2 continuation was stopped after identifying harness ordering/checkpoint issues. The committed harness fixes those issues, but I did not spend another H200 cycle on a non-production fallback path.

## Modal Apps

H200 production-gate/diagnostic apps from this lane are stopped with `0` tasks:

- `ap-4z52ctNEvWChDGYhW6wSUp`
- `ap-dLWaGo7LVUf3kGsP9toZoI`
- `ap-sryoCfRE0X3aIOl8pjSYL3`
- `ap-EoIVidElECuWqP4mJh3sKx`
- `ap-NtjffeR09mU8C6VoRGlvtN`
- `ap-I2C1Q2CTprb4wv20gk0xJt`

Final `modal app list --json` also showed the visible Wave32 H100/helper apps stopped with `0` tasks. Those are not H200 production-gate rows and are not used for the tables above.

## Judgment

Safe for main as harness/docs/artifacts only.

Not safe to claim a production H200 Mamba3/MLA throughput gate from this image. The image must be rebuilt with usable FA3 for H100/H200 production gates. TE `FusedAttention` fallback can remain a reachability diagnostic only.
