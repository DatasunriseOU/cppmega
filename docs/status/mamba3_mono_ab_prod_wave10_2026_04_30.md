# Mamba3 Mono AB Prod Wave 10 Final Gate - 2026-04-30

Status: complete
Canonical production decision: `docs/status/mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md`
Scope: Lane D final production gate/status summary for the 10-wave Mamba3
`mamba_mimo_bwd_bwd` campaign.

## Gate Result

Wave 10 closes the campaign with no production replacement accepted for
`mamba_mimo_bwd_bwd`.

The only movement that should merge to main is the guarded TileLang stage2
configuration:

```text
bf_num_stages = 1
bb_num_stages = 0
```

That path is exact against the baseline outputs and repeatedly improves the
H200 productionish chain by about `1.6-1.9%`. It is a small production win, not
a full CUDA/CuTe replacement.

All CUDA, CuTe, and WGMMA-copy replacement receipts remain research evidence.
After Wave 9 commit `155247c`, Lane D rejects every research receipt unless it
has all of the following:

- full-boundary integrated `mamba_mimo_bwd_bwd` timing;
- all 12 required `bwd_bwd` output slots;
- full-boundary correctness;
- resource, CTA, memory, H200, Modal hygiene, and budget evidence.

Missing full-boundary integrated timing is rejected as
`non_integrated_timing_receipt`. Missing any required output slot is rejected as
`missing_required_output_slots`. Component timing, local tile timing,
micro-GEMM timing, copy-lane metadata, and subset correctness earn zero
production credit.

## GPU Budget Rule

Future gate work should use this allocation rule:

| workload | GPU class |
| --- | --- |
| Mini, smoke, component, portability, and schema/harness checks | H100 |
| Full-size NAM56R production runs and full-boundary timing where memory or target hardware matters | H200 |

This Wave 10 Lane D doc update did not use GPU. It is a gate/status closeout
only.

## 10-Wave Outcome

The campaign produced one mergeable production change and several useful
research directions:

| wave area | outcome |
| --- | --- |
| Guarded TileLang stage2 `(bf=1, bb=0)` | Production candidate. Exact and repeatedly faster on H200 chain by about `1.6-1.9%`. |
| CUDA warp-owner covered subset | Best replacement economics so far: current covered subset is `2.48042 ms` versus `3.70674 ms` TileLang `bwd_bwd`, but it is not a full-boundary candidate. |
| CUDA qk-`DMIMO_V` output-owner all-R sidecar | Correct direction for avoiding partial tensors and atomics; still a sidecar and not a full production boundary. |
| CUDA tile-stream WMMA subset | Correct subset and reasonable resource shape, but `11.180607795715332 ms` H200 productionish is slower than full TileLang `bwd_bwd`. |
| CUDA row-stream low-live-set | Rejected as scalar recompute dead end; `179.76535034179688 ms` H200 productionish. |
| CUDA scan-owner DV/`DMIMO_V`/`DSSDA` subset | Correct subset evidence, but underfilled H200 grid and `14.08131217956543 ms` productionish timing. |
| CuTe fused masked-LKQ/apply and multi-chunk state/apply consumers | Best materialization-reduction direction; no production credit without full-boundary integrated timing and all output slots. |
| WGMMA copy path `uint4` 12-tile evidence | Implementation enabler only; ptxas/layout evidence does not satisfy output or timing gates. |
| Older CuTe micro-GEMM, WMMA, split/post, scan-owner receipts | Retained as historical evidence only. |
| ParaRNN / Apple-style M2RNN direction | Closed for exact dense `V=16` M2RNN replacement economics. |

## Final Branch Ranking

Ranking is by production readiness, not isolated component speed.

| rank | branch/path | final stance |
| ---: | --- | --- |
| 1 | guarded TileLang stage2 `(bf=1, bb=0)` | Merge to main behind the existing guard/default path. This is the only accepted production movement. |
| 2 | CUDA covered subset / warp-owner path | Continue as R&D. Reopen production discussion only after full-boundary output parity, integrated memory, and full H200 timing. |
| 3 | CuTe multi-chunk fused state/apply consumers | Continue as R&D for materialization removal. No main merge as production code without full-boundary gate receipts. |
| 4 | WGMMA copy path `uint4` 12-tile evidence | Keep as an implementation enabler or benchmark aid only. No production merge by itself. |
| 5 | CUDA tile-stream WMMA subset | Keep as research evidence; do not merge to main as a production replacement. |
| 6 | CUDA row-stream low-live-set and scan-owner subsets | Reject for production. Do not spend more gate time unless a new design removes scalar recompute and full-boundary blockers. |
| 7 | Older micro-GEMM, split/post, and local-only receipts | Archive as evidence. They have zero production credit under the final gate. |

## Mainline Merge Decision

Should merge to main:

- guarded TileLang stage2 `(bf=1, bb=0)`;
- the fail-closed Lane D receipt gate and its tests;
- status documentation that records the final gate rules and decision.

Should not merge to main as production behavior:

- CUDA warp-owner covered subset;
- CUDA qk-`DMIMO_V` sidecar as a standalone production replacement;
- CUDA tile-stream WMMA subset;
- CUDA row-stream low-live-set or scan-owner subset kernels;
- CuTe local tile, one-chunk, or multi-chunk receipts without full-boundary
  integrated timing;
- WGMMA copy-lane evidence as anything more than an enabler.

Research code can remain on worker branches or behind explicit benchmark-only
paths, but it should not become the production `mamba_mimo_bwd_bwd` path until
it passes the final gate.

## Next Concrete Production Work

1. Merge or confirm main already contains the guarded TileLang stage2
   `(bf=1, bb=0)` default.
2. Keep the Lane D gate in CI or pre-merge review for any future
   `mamba_mimo_bwd_bwd` replacement receipt.
3. Add one full-boundary integrated harness target for the next CUDA/CuTe
   candidate before collecting more component timings.
4. Require the candidate to produce all 12 output slots in the real call
   boundary: `dk`, `dv`, `dmimo_v`, `dq`, `dfactor`, `dgamma_diag`,
   `dangles`, `dd`, `dda`, `dssda`, `dda_cs_rev`, and `dda_cs`.
5. Measure integrated peak memory in the real autograd lifetime and compare it
   to TileLang stage2, not to standalone component harness memory.
6. Use H100 for smoke/component validation and reserve H200 for full NAM56R or
   full-boundary production timing.
7. Run training A/B against guarded stage2 only after full-boundary correctness,
   memory, and H200 timing pass.

## Production Stance

Final stance: guarded TileLang stage2 `(bf=1, bb=0)` is the production answer
from this campaign. CUDA/CuTe/WGMMA replacement work remains research.

The CUDA covered subset is still worth pursuing because its current economics
leave a plausible budget for a future full replacement. It is not mergeable as
production because it lacks off-time/state work, scalar outputs, complete
`DK/DQ/DV/DMIMO_V` accumulation, full output parity, integrated memory proof,
and training A/B.

## Validation

Local checks for this doc-only closeout:

```text
python -m json.tool docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json
PYTHONPATH=. pytest -q tests/test_mamba3_mono_ab_schema.py tests/test_mamba3_mono_ab_modal_hygiene.py
PYTHONPATH=. python -m py_compile cppmega/megatron/mamba3_mono_ab_schema.py scripts/modal_mamba3_cuda_full_bwd_ab.py
git diff --check
```
