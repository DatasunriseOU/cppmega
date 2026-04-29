# Mamba3 B200+ Modal Retry Status - 2026-04-29

Branch: `worker/mamba3-b200-plus-retry`

Base: `worker/mamba3-b200-plus-logging` at `2111e13`

Image: `GHCR_TAG=785c3fd`

Harness: `scripts/modal_mamba3_b200_paths.py`

Volume: `cppmega-mamba3-b200-plus-logging`

## Summary

Retried the B200+ Modal harness with the corrected flexible Blackwell specs:
primary `B200+:2`, then fallback `B200+`.

Both apps were accepted by Modal, but neither allocated a function container
within the bounded capacity wait. Both were stopped manually before the
15-minute total wait limit. No B200/B300 CUDA device report, TileLang timing,
PsiV timing, or result artifacts were produced.

## Attempts

| Requested GPU | Run ID | App ID | Created | Stopped | Final state | Tasks | Container start | Volume artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `B200+:2` | `b200_plus2_785c3fd_20260429_retry1` | `ap-gPKJOzQYHe0Jm79fIFMIRq` | `2026-04-29 11:27:12 UTC` | `2026-04-29 11:32:14 UTC` | `stopped` | `0` | no `[run:start]` | none |
| `B200+` | `b200_plus1_785c3fd_20260429_retry1` | `ap-qzBDagBuwU0NOKBTSyuWDW` | `2026-04-29 11:32:40 UTC` | `2026-04-29 11:38:33 UTC` | `stopped` | `0` | no `[run:start]` | none |

Log tails for both apps contain only the manual stop marker:

```text
Stopping app - user stopped from CLI.
```

## Device And Artifact Status

No function container started, so the harness never reached `_run_probe`.

| Field | Value |
| --- | --- |
| Device name | not available |
| CUDA capability | not available |
| Device report path | not created |
| TileLang timing path | not created |
| PsiV timing path | not created |
| Combined result path | not created |

Expected paths, if a future container starts:

```text
/mamba3_b200_plus/b200_plus2_785c3fd_20260429_retry1/
/mamba3_b200_plus/b200_plus1_785c3fd_20260429_retry1/
```

Both paths were checked with `modal volume ls` and were absent.

## Modal Hygiene

Final check showed both `cppmega-mamba3-b200-plus-paths` retry apps stopped:

```text
ap-gPKJOzQYHe0Jm79fIFMIRq stopped Tasks=0
ap-qzBDagBuwU0NOKBTSyuWDW stopped Tasks=0
```

No live B200+ retry app was left behind.
