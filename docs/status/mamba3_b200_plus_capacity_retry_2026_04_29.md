# Mamba3 B200+ Capacity Retry Notes - 2026-04-29

Branch: `worker/mamba3-b200-plus-retry`

Commit under test: `96f41d1`

Goal: understand why Modal accepted `B200+:2` and `B200+` GPU specs but no
container started for the retry probe.

## Observed Runs

No new Modal jobs were launched for this investigation. The local checks were
read-only.

Recent B200 attempts:

| spec | app id | created UTC | stopped UTC | container list | task count | volume output |
| --- | --- | --- | --- | --- | --- | --- |
| `B200+:2` | `ap-gPKJOzQYHe0Jm79fIFMIRq` | `2026-04-29 11:27:12` | `2026-04-29 11:32:14` | `[]` | `0` | none |
| `B200+` | `ap-qzBDagBuwU0NOKBTSyuWDW` | `2026-04-29 11:32:40` | `2026-04-29 11:38:33` | `[]` | `0` | none |

The logs contained only the stop event, with no probe-side `[run:start]`.

Interpretation:

- Modal accepted the GPU spec syntax.
- The function body did not execute.
- No CUDA, image import, TileLang, or Python runtime error was observed.
- The likely state was scheduler pending / capacity wait, interrupted before a
  GPU container was assigned.

## Modal Details That Matter

Modal documents `B200`, `B200:2`, `B200+`, and `B200+:2` as valid GPU specs.
`B200+` allows Modal to place the call on B200 or B300 capacity, and B300
requires CUDA 13.0 or newer. Our image is CUDA 13.2, so B300 compatibility should
not be rejected at the image level.

Do not constrain `region=` or `cloud=` for these retries unless a specific
debugging question requires it. Broad/no region gives the scheduler the largest
pool.

Useful checks:

```text
modal app list --json
modal app logs <app-id> --timestamps --show-function-id --show-function-call-id --show-container-id --tail 1000
modal container list --app-id <app-id> --json
modal volume ls cppmega-mamba3-b200-plus-logging /mamba3_b200_plus
```

Signals:

- `container list=[]`, no `[run:start]`, empty Volume: no GPU container was
  assigned.
- `ta-*` container exists but no `[run:start]`: startup/import/global-scope
  failure.
- `[run:start]` exists: code entered the GPU container; failures after that are
  harness/runtime failures.

## Retry Plan

1. Do not run `B200+` and `B200+:2` simultaneously; they compete for the same
   scarce pool.
2. First run single-GPU `B200+` with no region/cloud pin and a 60-120 minute
   scheduling budget.
3. Count success only when logs show `[run:start]` and the Volume contains the
   run metadata.
4. After one B200/B300 container starts successfully, try `B200+:2` with a
   90-180 minute budget.
5. Prefer a `.spawn()` wrapper for long retries so we can print the
   `FunctionCall` id and poll `get_current_stats()` / `get_call_graph()` every
   30-60 seconds, then cancel explicitly at the deadline.

If broad `B200+` remains pending for hours with no `ta-*` container and no
Volume output, the next step is Modal support with the app ids, UTC windows,
workspace, GPU specs, and proof that no container was assigned.
