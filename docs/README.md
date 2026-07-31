# Documentation Index

Use this file as the front door for repo documentation.

## Current Status

- [status/README.md](status/README.md) defines the doc status labels,
  retention rules, and canonical status entry points.
- [production_status.md](production_status.md) is the source of truth for
  current production configs and launch numbers.
- [reproducible_runs.md](reproducible_runs.md) contains the current launch
  commands for validated configs.
- [porting_policy.md](porting_policy.md) captures what belongs in `cppmega`
  versus upstream Megatron, Transformer Engine, TileLang, or other projects.
- [objective_mixture_contract.md](objective_mixture_contract.md) defines the
  typed pre-materialized objective handoff and fail-closed Megatron/H200 gates.
- [status/training_data_inventory.md](status/training_data_inventory.md) is the
  canonical entry point for live corpus counts, sidecars, physical paths,
  training eligibility, and the append-only data-status changelog.

## Session And Probe Notes

- [sessions/README.md](sessions/README.md) indexes dated session/probe notes
  such as `*_2026_04_25.md`.

Dated notes are evidence, not automatically canonical status. When a dated note
changes current behavior, summarize the decision in the relevant status doc and
link the dated note as supporting evidence.
