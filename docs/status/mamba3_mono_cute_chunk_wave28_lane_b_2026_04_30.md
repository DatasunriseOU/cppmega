# Mamba3 Mono CuTe Chunk Wave28 Lane B - 2026-04-30

Status: R&D only
Date: 2026-04-30
Scope: Gate Wave10 `uint4`/128-bit G2S copy evidence into the CuTe fused
multi-chunk scan-owner path without changing the default scalar-copy behavior.

## Change

- Added `CPPMEGA_MAMBA3_CUTE_MULTI_UINT4_G2S`.
- Default remains scalar BF16 universal G2S for the fused multi-chunk path.
- Setting the env var to `1` compiles the multi-chunk scan-owner with 128-bit
  G2S operand copies using the Wave10 contract from
  `mamba3-mono-triton-model` commit `65ef653`.
- The one-chunk fused consumer path stays scalar; an attempted 128-bit G2S
  one-chunk run failed correctness and was not kept.
- Modal H100 harness propagation was added so the opt-in can be tested remotely.

## H100 Validation

Default command:

```bash
CPPMEGA_MODAL_APP_NAME=cppmega-mamba3-mono-chunk-wave28-lane-b-default-h100 \
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode cute-lkq-chain --iters 20
```

Opt-in command:

```bash
CPPMEGA_MODAL_APP_NAME=cppmega-mamba3-mono-chunk-wave28-lane-b-optin-h100 \
CPPMEGA_MAMBA3_CUTE_MULTI_UINT4_G2S=1 \
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py \
  --mode cute-lkq-chain --iters 20
```

Results:

| Mode | Correctness | 2 chunks | 4 chunks | 8 chunks |
| --- | --- | ---: | ---: | ---: |
| Baseline at start of lane | pass | 104.872 us | 105.526 us | 105.757 us |
| Default after gate | pass | 98.822 us | 105.864 us | 108.507 us |
| Opt-in uint4 multi-chunk | pass | 108.888 us | 108.880 us | 110.690 us |

Correctness tolerance was `1e-5`.  Peak memory was not measured by this harness.

## Modal Cleanup

Apps started in this lane:

- `ap-Nn2XHwo7ObexOznqepHwjy` baseline
- `ap-gWurK6zoWimgFr7um36xEv` all-vector attempt, failed one-chunk correctness
- `ap-8huo3XW8GOVQWhAytgNNXr` multi-only vector attempt
- `ap-ZB1KcgUwT7BPW1CG54ku4M` default gate validation
- `ap-5KWdQfP8CCGeQgjdPT1OUX` opt-in gate validation

All completed through the local entrypoint.  Run `modal app list` after the
lane to verify that no tasks remain running.

## Production Status

Not safe for production main.  The opt-in vector path preserved correctness on
H100 mini shapes, but did not beat the prior CuTe mini baseline.  Remaining
production blockers are unchanged: optional `D * dPhi`, combined
`dPsiV_D.to(bf16)`, full `DGAMMA_DIAG`/`DK`/`DQ`, vectorized or warp-reduced
consumers, production `DMIMO_V` ownership, internal `Q.T`/`DPh.T`, and NAM56R
integration.
