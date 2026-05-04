# Mamba3 Mono AB Prod Wave 2 - 2026-04-30

Worktree: `/home/dave/source/cppmega/.claude/worktrees/mamba3-mono-ab-prod`

Branch: `worker/mamba3-mono-ab-prod`

Scope: Lane D follow-up for production A/B harness integration. Wave 2 did not
add a monolithic kernel. It made Lane A/B/C component receipts pluggable and
tightened Modal app hygiene so production A/B runs can fail fast on leaks.

## Implemented

Updated:

- `cppmega/megatron/mamba3_mono_ab_schema.py`
- `scripts/modal_mamba3_cuda_full_bwd_ab.py`
- `tests/test_mamba3_mono_ab_schema.py`

Added:

- `docs/status/mamba3_mono_ab_prod_wave2_2026_04_30.md`

## Component Record Ingress

The schema now accepts component candidate records from either:

1. Inline JSON via `--candidate-record-json`.
2. JSON files or markdown docs via `--candidate-record-path-csv`.

Markdown docs must carry a fenced JSON payload. The accepted top-level keys are
`mamba3_mono_ab_component_records`, `candidate_component_records`, or
`component_records`.

Minimal example:

```json
{
  "candidate_component_records": [
    {
      "candidate_id": "mono_lane_c",
      "lane": "C",
      "shape": "productionish",
      "components": [
        {
          "component_id": "diag_qkdv",
          "mean_ms": 1.92990,
          "launches": 1,
          "covered_slots": ["dk", "dq", "dv", "dgamma_diag"]
        },
        {
          "component_id": "qk_dmimov",
          "mean_ms": 0.53634,
          "launches": 1,
          "covered_slots": ["dmimo_v"]
        }
      ],
      "reference": {
        "stage2_bwd_fwd_ms": 1.81156,
        "stage2_bwd_bwd_ms": 3.72232,
        "stage2_chain_ms": 5.50098
      },
      "max_memory_allocated_gib": 6.92774
    }
  ]
}
```

The harness normalizes each record, filters it by shape, and computes:

- `projected_bwd_bwd_ms`
- ratio and speedup floor versus stage2 `bwd_bwd`
- remaining budget before matching stage2
- stage2-chain floor with unchanged `bwd_fwd`
- launch count
- covered and missing output slots
- memory delta versus stage2 when both sides report peak memory

If a Modal A/B run has live stage2 timings, those live timings override any
reference numbers supplied in the record. Dry-run mode uses record references.

## Modal Hygiene

The local entrypoint still auto-stops exact-name app instances with `Tasks=0`.
Wave 2 adds same-campaign detection across this default prefix:

```text
cppmega-mamba3-
```

The new `--modal-hygiene-enforcement` option accepts:

| mode | behavior |
| --- | --- |
| `warn` | default; records a warning verdict if same-campaign apps remain active |
| `fail` | exits non-zero after printing the summary if active same-campaign apps remain |
| `off` | records app list data but does not gate |

This gives production A/B lanes a concrete leak gate without stopping unrelated
apps by prefix. Exact harness apps with zero tasks are still safe-stopped.

## Validation

Local checks:

```text
python -m py_compile \
  cppmega/megatron/mamba3_mono_ab_schema.py \
  scripts/modal_mamba3_cuda_full_bwd_ab.py

pytest -q tests/test_mamba3_mono_ab_schema.py
# 8 passed
```

Light dry-run:

```text
timeout 300s modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
  --dry-run-schema \
  --shape-csv smoke \
  --monolithic-candidate-csv mono_lane_c \
  --candidate-record-json '{"candidate_component_records":[...]}' \
  --modal-hygiene-enforcement warn
```

Receipt:

- app: `ap-OtxzL0gNO95csCs97vCXhv`
- remote GPU work: none; schema dry-run only
- candidate record projection: `0.05125 ms` smoke `bwd_bwd` floor
- remaining smoke budget versus stage2: `0.11131 ms`
- covered slots: `dk,dv,dmimo_v,dq,dgamma_diag`
- missing slots: `dfactor,dangles,dd,dda,dssda,dda_cs_rev,dda_cs`
- hygiene verdict: pass after auto-stopping the zero-task ephemeral app

Final Modal cleanup:

- stopped active same-campaign app `ap-i10KzqPzk9EX5w6c5gpjav`
  (`cppmega-mamba3-mono-chunk-wave2-h200`, `Tasks=1`)
- stopped zero-task same-campaign app `ap-yLByZhcewODeSt1PQqsMFv`
  (`cppmega-mamba3-mono-cuda-chunk-wave2-h200`, `Tasks=0`)
- stopped active same-campaign app `ap-dIhme5Hbldl7PyMatHy29a`
  (`cppmega-mamba3-mono-cuda-chunk-wave2-h200`, `Tasks=1`)
- stopped active same-campaign app `ap-HZQNMW6M9D2rUTsyVLQxkC`
  (`cppmega-mamba3-mono-chunk-wave2-h200`, `Tasks=1`)
- stopped zero-task same-campaign app `ap-YfQFhrtHjcoEFHwvn7g4zC`
  (`cppmega-mamba3-mono-triton-pruned-wave2`, `Tasks=0`)
- stopped active same-campaign app `ap-bk7XQGFUUe4yipXWH7cOp6`
  (`cppmega-mamba3-mono-chunk-wave2-h200`, `Tasks=1`)
- stopped active same-campaign app `ap-ggF6eSLTO0cgYamBY523kl`
  (`cppmega-mamba3-mono-cuda-chunk-wave2-h200`, `Tasks=1`)
- stopped zero-task same-campaign app `ap-lddL3zHajpVqjUfwGPToHb`
  (`cppmega-mamba3-mono-triton-pruned-wave2`, `Tasks=0`)
- final delayed `modal app list --json` showed no active same-campaign apps

H200 smoke was not rerun. Wave 2 changed schema ingestion and local hygiene
automation only; Wave 1 already has an H200 smoke receipt for the harness path.

## Production A/B Gate Impact

Production A/B can now accept Lane A/B/C component receipts without hand-editing
the harness:

1. Lane A/B/C attach a JSON record to their status doc or provide a JSON file.
2. Lane D passes the doc/file through `--candidate-record-path-csv`.
3. The harness computes per-shape budget, slot coverage, and projected chain
   impact against the live stage2 reference.
4. Component records remain marked partial until every `mamba_mimo_bwd_bwd`
   output slot is covered and rechecked at the real call boundary.
5. Production runs can set `--modal-hygiene-enforcement fail` so leaked
   same-campaign Modal apps fail the gate after artifact capture.

This keeps guarded stage2 as the production reference while making future
monolithic candidates easy to score and hard to accidentally leave running.
