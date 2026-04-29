# Mamba3 B200+ Modal Logging Status - 2026-04-29

Branch: `worker/mamba3-b200-plus-logging`

Base: `worker/mamba3-b200-paths` at `f77084c`

Scope: updated only the B200 Modal harness path and added this status note.
No production defaults were changed.

## Harness Update

Updated `scripts/modal_mamba3_b200_paths.py` to request Modal's exact flexible
Blackwell GPU string:

```python
@app.function(..., gpu="B200+", ...)
def run_b200_plus(...)
```

The old default specs `B200+:2`, `B200:2`, and `B200:1` are no longer the
default path for this harness. `CPPMEGA_MAMBA3_B200_SPECS` now accepts only
`B200+`.

Results are written stage-by-stage to Modal Volume
`cppmega-mamba3-b200-plus-logging`:

```text
/mamba3_b200_plus/<run_id>/
  metadata.json
  events.jsonl
  result.partial.json
  result.json
  stages/device.json
  stages/patch_sites.json
  stages/psiv_costs.json
  stages/tilelang_bwd_split.json
  stages/cutile_cute_possibility.json
  stages/hf_kernel_candidates.json
```

The remote function also prints `[run:*]` and `[stage:*]` markers to stdout, so
`modal app logs <app-id>` is enough to identify whether a container actually
started.

## B200+ Attempt

Command:

```bash
GHCR_TAG=785c3fd modal run --detach scripts/modal_mamba3_b200_paths.py::run_b200_plus --run-id b200_plus_785c3fd_20260429_1
```

Result:

| Field | Value |
| --- | --- |
| App ID | `ap-3ALFMULx0bAB5iGKEoouFy` |
| App description | `cppmega-mamba3-b200-plus-paths` |
| Image tag | `GHCR_TAG=785c3fd` |
| Requested Modal GPU | exact `gpu="B200+"` |
| Created | `2026-04-29 10:57:04 UTC` |
| Stopped | `2026-04-29 11:00:40 UTC` |
| State at stop | `stopped`, `Tasks=0` |
| Container start | no function container reached `_run_probe` |
| Volume artifacts | none; Volume exists but `/mamba3_b200_plus` is empty |

App logs show only image build output and the manual stop marker. There is no
`[run:start]` line, no CUDA device report, and no stage artifact. This means
Modal accepted and built the exact `B200+` app, but no B200/B300 function
container was allocated during the wait window.

Log command used:

```bash
modal app logs ap-3ALFMULx0bAB5iGKEoouFy --timestamps --show-function-call-id --show-container-id --tail 300
```

Volume check used:

```bash
modal volume ls cppmega-mamba3-b200-plus-logging /
```

## How To Re-Poll Or Re-Run

Launch a new detached attempt:

```bash
GHCR_TAG=785c3fd modal run --detach scripts/modal_mamba3_b200_paths.py::run_b200_plus --run-id <run_id>
```

Find the app id:

```bash
modal app list --json
```

Poll logs:

```bash
modal app logs <app-id> --timestamps --show-function-call-id --show-container-id --tail 300
modal app logs <app-id> -f
```

Check whether the Volume has run artifacts:

```bash
modal volume ls cppmega-mamba3-b200-plus-logging /mamba3_b200_plus
modal volume ls cppmega-mamba3-b200-plus-logging /mamba3_b200_plus/<run_id>
```

Download run artifacts after a container starts:

```bash
modal volume get cppmega-mamba3-b200-plus-logging /mamba3_b200_plus/<run_id> ./modal_b200_plus_<run_id>
```

Stop an accepted-but-not-provisioned app:

```bash
modal app stop --yes <app-id>
```

## Current Status

Current status: **B200+ accepted, no container allocated**.

No B200/B300 CUDA device report, TileLang baseline, PsiV write timing, or
Mamba3 TileLang split timing was obtained in this attempt. The next useful
signal is any run whose logs show `[run:start]`; after that, the Volume should
contain per-stage JSON even if a later benchmark crashes.
