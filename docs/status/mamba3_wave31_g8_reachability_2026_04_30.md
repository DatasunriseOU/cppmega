# Mamba3 Wave31 G=8 Reachability

Date: 2026-04-30
Branch: `worker/mamba3-wave31-g8-support`

## Change

Added a guarded source applier for the Mamba3 MIMO backward grouped-head reduction
branch:

- `CPPMEGA_MAMBA3_GROUPED_HEAD_BWD=1`
- `MAMBA3_GROUPED_HEAD_BWD_ALLOW_FILE_MUTATION=1`

The applier patches the installed Modal image source at:

`/usr/local/lib/python3.13/dist-packages/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`

and the varlen sibling:

`/usr/local/lib/python3.13/dist-packages/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd_varlen.py`

The inserted branch handles `1 < G < H` with `H % G == 0` by reducing
`dq_tilelang` and `dk_tilelang` over `heads_per_group`:

`[B, S, R, H, N] -> [B, S, R, G, H//G, N] -> sum(dim=4)`.

## Validation Commands

```bash
python -m py_compile \
  cppmega/megatron/upstream_patches/apply_mamba3_grouped_head_bwd_patches.py \
  scripts/modal_mamba3_wave31_g8_reachability.py

pytest -q \
  tests/test_mamba3_grouped_head_bwd_applier.py \
  tests/test_mamba3_stage2_force_nontma_applier.py

PYTHONPATH=. python -m cppmega.megatron.upstream_patches.apply_mamba3_grouped_head_bwd_patches

PYTHONPATH=. CPPMEGA_MAMBA3_GROUPED_HEAD_BWD=1 \
  python -m cppmega.megatron.upstream_patches.apply_mamba3_grouped_head_bwd_patches

modal run scripts/modal_mamba3_wave31_g8_reachability.py::launch_g8_reachability \
  --run-id wave31_g8_h200_reachability_20260430 \
  --train-iters 1 \
  --timeout-per-case-s 2400
```

## Local Results

- `py_compile`: pass.
- `pytest`: `8 passed`.
- No-gate applier: skipped as expected.
- Gate-without-allow applier: refused mutation as expected.
- Local GPU is GB10, not H100; no local H100 smoke was possible in this worktree.

## Modal H200:2 Result

App:

- `ap-FwYHGBOJr9X9ID5r2fVQ53`
- Description: `cppmega-wave31-g8-reachability`
- Final state: `stopped`, `0` tasks.

Run:

- Run id: `wave31_g8_h200_reachability_20260430`
- Case: `fallback_auto_full`
- Variant: `grouped_head_bwd_baseline`
- Dataset: `synthetic_full_shape_mock_data`; real Modal volume NAM56R data was not present.
- Result: completed iteration 1 and got past `mamba_mimo_bwd_combined` G=8.

Numbers from `summary.md` / log:

- status: `ok`
- steps seen: `1`
- elapsed, cold compile included: `326948.9 ms`
- tok/sec from last step: `100.224`
- peak alloc: `54.864 GiB`
- peak reserved: `55.688 GiB`
- reached Mamba backward: yes
- TE attention fallback: FusedAttention selected; FlashAttention rejected for MLA.

The baseline log shows successful compilation/execution of:

- `mamba_mimo_fwd_kernel`
- `mamba_mimo_bwd_fwd_kernel`
- `mamba_mimo_bwd_bwd_kernel`

and then an iteration-1 training line. The previous Wave30 blocker
`ValueError: G value of 8 is not currently supported!` did not occur.

## Notes

The first run used an earlier harness version whose `_kernel_status` helper had
bad indentation inside the inline Python string, so the saved `kernel` status
field is an error even though the applier stdout confirms both grouped-head patches were
written. The helper is fixed in this branch after the run.

The harness originally started the stage2 variant after the successful baseline.
I stopped the app to avoid burning more H200 after the reachability target was
already met. The committed `launch_g8_reachability` entrypoint now runs baseline
only by default; `launch_debug_sweep` still accepts `include_stage2=True`.

Safe-for-main judgment: safe as a guarded source-applier plus reachability
harness. Do not make it unconditional until a real-data 20-step gate has clean
grad/loss behavior; this synthetic run had `grad norm: nan`, so it only proves
the `G=8` source blocker is removed.
