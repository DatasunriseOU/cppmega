# Mamba3 Hopper TMA perf/correctness - 2026-04-29

Branch: `worker/mamba3-hopper-tma-perf`

Base: `worker/mamba3-hopper-tma-ws-fix` (`ccd9679`)

Harness: `scripts/modal_mamba3_hopper_tma_perf.py`

Scope: temp-only comparison of upstream/source non-TMA baseline vs copied
`mamba3_mimo_bwd.py` with the Hopper TMA layout patch plus the
`qk_shared_direct` workaround. No production defaults were changed.

## Runs

Commands used for the successful full runs:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100:2 modal run scripts/modal_mamba3_hopper_tma_perf.py
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 modal run scripts/modal_mamba3_hopper_tma_perf.py
```

| GPU request | Successful Modal app | Actual device | TileLang |
| --- | --- | --- | --- |
| `H100:2` | `ap-l5j0o5sQMi4Ti1DoQaaGgj` | `NVIDIA H100 80GB HBM3` | `0.1.8+cu132.gitf309d814` |
| `H200:2` | `ap-JIRu4Ab2uD21Z97Kk9dDzV` | `NVIDIA H200` | `0.1.8+cu132.gitf309d814` |

Note: the first successful JSON reported `image_ref: ...:latest` because Modal
re-imported the script in the container without the local `GHCR_TAG` env. The
run commands above did set `GHCR_TAG=785c3fd`; the harness now also injects
`CPPMEGA_IMAGE_REF=ghcr.io/jewelmusicee/cppmega:785c3fd` into the image so
future JSON is unambiguous. Follow-up metadata reruns were stopped and are not
used for the results below.

All candidate compiles still logged:

```text
[WS] skipped: no TMA copies in pipeline loop
```

So this remains a TMA-lowered source-layout workaround, not an effective
warp-specialized producer/consumer path.

## Shapes

All runs used `hasZ=False`, `hasD=False`, `reduceO=True`, BF16 inputs, and
`chunk_size=16`.

| Name | Shape |
| --- | --- |
| `smoke` | `B=1 S=64 H=4 G=1 N=64 P=64 R=4` |
| `representative` | `B=2 S=1024 H=16 G=1 N=64 P=64 R=4` |
| `productionish` | `B=4 S=4096 H=32 G=1 N=64 P=128 R=4` |

## Correctness

Compared baseline vs `qk_shared_direct` for:

`dmimo_o`, `states`, `qk_dot`, `dk`, `dv`, `dmimo_v`, `dq`, `dfactor`,
`dgamma_diag`, `dangles`, `dd`, `dda`, `dssda`, `dda_cs_rev`, `dda_cs`.

Result: every compared tensor had `max_abs=0.0` on H100 and H200 for all three
shapes. Main-grad summary:

| GPU | Shape | max main-grad abs diff |
| --- | --- | ---: |
| H100 | smoke | 0.0 |
| H100 | representative | 0.0 |
| H100 | productionish | 0.0 |
| H200 | smoke | 0.0 |
| H200 | representative | 0.0 |
| H200 | productionish | 0.0 |

## Timings

CUDA event timings, milliseconds per run. `speedup` is
`baseline combined_ms / qk_shared_direct combined_ms`.

### H100

| Shape | Baseline bwd_fwd | Baseline bwd_bwd | Baseline combined | qk bwd_fwd | qk bwd_bwd | qk combined | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 0.0190 | 0.0393 | 0.0571 | 0.0191 | 0.0389 | 0.0571 | 1.0000x |
| representative | 0.2972 | 0.6437 | 0.9255 | 0.3158 | 0.6409 | 0.9241 | 1.0015x |
| productionish | 1.8931 | 3.6332 | 5.4309 | 1.9733 | 3.5619 | 5.5282 | 0.9824x |

### H200

| Shape | Baseline bwd_fwd | Baseline bwd_bwd | Baseline combined | qk bwd_fwd | qk bwd_bwd | qk combined | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 0.0192 | 0.0397 | 0.0578 | 0.0191 | 0.0393 | 0.0575 | 1.0052x |
| representative | 0.2584 | 0.6488 | 0.9288 | 0.2742 | 0.6456 | 0.9457 | 0.9821x |
| productionish | 1.8692 | 3.6993 | 5.5554 | 1.9573 | 3.6783 | 5.6236 | 0.9879x |

## Kernel source size

Source chars from successful runs:

| GPU | Shape | Baseline fwd | Baseline bwd | qk fwd | qk bwd |
| --- | --- | ---: | ---: | ---: | ---: |
| H100 | smoke | 39453 | 89276 | 39628 | 89206 |
| H100 | representative | 40038 | 90430 | 40198 | 90360 |
| H100 | productionish | 40918 | 83429 | 41077 | 83375 |
| H200 | smoke | 39474 | 89276 | 39649 | 89206 |
| H200 | representative | 40038 | 90430 | 40198 | 90360 |
| H200 | productionish | 40918 | 83429 | 41077 | 83354 |

## Conclusion

Correctness: GO. `qk_shared_direct` is bitwise identical to the non-TMA baseline
for the tested representative Hopper shapes.

Production performance: NO-GO. The candidate is neutral on small/representative
H100, slightly slower on H200 representative, and slower on production-ish
combined time on both H100 and H200. Since WS is still skipped, there is no
producer/consumer overlap benefit to justify wiring this into production
defaults.

Recommended next step: keep the harness as a regression/probe tool only. A real
production candidate needs a source shape that places TMA copies inside a
pipeline loop that TileLang WS can transform, or a different kernel structure.
