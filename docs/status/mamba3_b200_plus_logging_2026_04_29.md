# Mamba3 B200+ Modal Logging Status - 2026-04-29

Branch: `worker/mamba3-b200-plus-logging`

Base: `worker/mamba3-b200-paths` at `f77084c`

Scope: updated only the B200 Modal harness path and added this status note.
No production defaults were changed.

## Harness Update

Updated `scripts/modal_mamba3_b200_paths.py` to request Modal's flexible
Blackwell GPU type with explicit quantity syntax:

```python
@app.function(..., gpu="B200+:2", ...)
def run_b200_plus_2(...)

@app.function(..., gpu="B200+", ...)
def run_b200_plus(...)
```

Default specs are now `B200+:2,B200+`: two-GPU flexible Blackwell is primary,
and single-GPU `B200+` is the fallback. The harness keeps separate remote
functions for both specs so the Modal SDK sees literal `gpu=` strings.
`B200:2` and `B200:1` remain intentionally out of the default path.

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

## Earlier B200+ Single-GPU Attempt

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

Launch the primary two-GPU attempt:

```bash
GHCR_TAG=785c3fd modal run --detach scripts/modal_mamba3_b200_paths.py::run_b200_plus_2 --run-id <run_id>
```

Launch the single-GPU fallback:

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

## Status After Earlier Attempt

Earlier status: **B200+ accepted, no container allocated**.

No B200/B300 CUDA device report, TileLang baseline, PsiV write timing, or
Mamba3 TileLang split timing was obtained in this attempt. The next useful
signal is any run whose logs show `[run:start]`; after that, the Volume should
contain per-stage JSON even if a later benchmark crashes.

## Corrected B200+ Attempts

Correction: Modal's flexible Blackwell type is `B200+`; the `:n` suffix is the
GPU count. For this task the primary request is therefore `B200+:2`, with
`B200+` as the single-GPU fallback.

Primary command:

```bash
GHCR_TAG=785c3fd timeout 240s modal run --detach scripts/modal_mamba3_b200_paths.py::run_b200_plus_2 --run-id b200_plus2_785c3fd_20260429_2
```

Fallback command:

```bash
GHCR_TAG=785c3fd timeout 240s modal run --detach scripts/modal_mamba3_b200_paths.py::run_b200_plus --run-id b200_plus1_785c3fd_20260429_2
```

Results:

| Requested GPU | Run ID | App ID | Created | Stopped | Final state | Tasks | Log tail | Volume artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `B200+:2` | `b200_plus2_785c3fd_20260429_2` | `ap-pJDL36lOBa1HlLg6vQfaae` | `2026-04-29 11:03:20 UTC` | `2026-04-29 11:05:40 UTC` | `stopped` | `0` | only `Stopping app - user stopped from CLI.` | none |
| `B200+` | `b200_plus1_785c3fd_20260429_2` | `ap-zZWAqMpHkqfYshH785weqJ` | `2026-04-29 11:03:43 UTC` | `2026-04-29 11:05:40 UTC` | `stopped` | `0` | only `Stopping app - user stopped from CLI.` | none |

Both apps were accepted by Modal and created the expected functions
`run_b200_plus_2` and `run_b200_plus`, but neither allocated a function
container during the wait window. There were no `[run:start]` markers and no
remote writes to `cppmega-mamba3-b200-plus-logging`.

Commands used for verification:

```bash
modal app logs ap-pJDL36lOBa1HlLg6vQfaae --timestamps --show-function-call-id --show-container-id --tail 100
modal app logs ap-zZWAqMpHkqfYshH785weqJ --timestamps --show-function-call-id --show-container-id --tail 100
modal app list --json
modal volume ls cppmega-mamba3-b200-plus-logging /
modal app stop --yes ap-pJDL36lOBa1HlLg6vQfaae
modal app stop --yes ap-zZWAqMpHkqfYshH785weqJ
```

Updated current status: **`B200+:2` primary and `B200+` fallback both accepted,
but neither got a container**.
