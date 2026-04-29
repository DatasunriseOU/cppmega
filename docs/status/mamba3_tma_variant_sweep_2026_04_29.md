# Mamba3 TMA Variant Sweep - 2026-04-29

Branch: `worker/mamba3-tma-variant-sweep`

Base: `worker/mamba3-hopper-tma-ws-fix` (`ccd9679`)

Scope: temp-only Modal harness plus patch artifacts for Mamba3 MIMO
`bwd_fwd`/`bwd_bwd` TMA experiments. No upstream `mamba_ssm` files are modified
in-place.

## Modal Stop

Per operator request, all Modal work was stopped immediately. No further Modal
apps or containers were launched after the stop request.

Known app IDs from this turn were checked with `modal app stop`:

- `ap-okRR3MJofixt1oEC9D2csk` - already stopped at `2026-04-29 11:13:55+00:00`.
- `ap-hXHUJyY5fPoUcydrT0FRJ1` - already finished at `2026-04-29 11:11:44+00:00`.
- `ap-LJbsX6J8W2aP8TTiI3A77E` - already finished at `2026-04-29 11:03:22+00:00`.

## Artifacts Added

- `scripts/modal_mamba3_tma_variant_sweep.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_tma_variant_qk_direct_smem_read.patch`
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_tma_variant_smem_bias_no_fragment.patch`
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_tma_variant_qk_recompute.patch`
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_tma_variant_qk_dot_rs_layout.patch`

The sweep harness supports:

- baseline non-TMA compile/smoke;
- TMA + WS layout patch failure classification;
- `qk_serial_p`;
- `qk_direct` direct shared-memory read;
- direct smem bias update without local Q/K bias fragments;
- `qk_recompute` instead of loading cached `QK_DOT`;
- altered `QK_DOT` layout `[B, H, R*R, S]`;
- `bb_threads`/`bb_num_stages` variants around the direct-smem path.

## Completed Local Validation

Local-only checks completed:

- `python -m py_compile scripts/modal_mamba3_tma_variant_sweep.py`
- temp source generation for all default variants from
  `/home/dave/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`
- `py_compile` of every generated temp source

No additional Modal results are claimed beyond already printed logs and existing
status artifacts.

## Existing GPU Evidence Reused

From `docs/status/mamba3_hopper_tma_ws_probe_2026_04_29.md` on the base branch:

| Variant | H100 | H200 | Notes |
| --- | --- | --- | --- |
| `layout_patch` | compile fails | compile fails | Hits TileLang `FloorMod` divide-by-zero. |
| `no_floormod` | compile fails | compile fails | Then hits `Loop layout is not injective` on `qk_dot` copy. |
| `qk_serial_p` | compile fails | compile fails | Avoids exact copy error but hits `contains inner var p`. |
| `qk_shared_direct` / `qk_direct` | compile + smoke OK | compile + smoke OK | Directly reads `qk_dot_shared[cs, r_out * R + r_in]`. |

Both successful previous runs logged:

```text
[WS] skipped: no TMA copies in pipeline loop
```

So `TL_DISABLE_WARP_SPECIALIZED=False` is accepted, but the producer-consumer
WS pass still does not transform the Mamba3 loops.

## This Turn's Partial Modal Evidence

All new runs used `GHCR_TAG=785c3fd`.

Observed before stop:

- `qk_direct` on `H100:2` compiled `bwd_fwd` and repeatedly compiled
  `bwd_bwd` for smoke attempts.
- The repeated compile was traced to an early harness version that timed the
  wrapper `mamba_mimo_bwd`, causing TileLang to rebuild the kernel inside the
  measurement loop. The harness was fixed to precompile `bwd_fwd`/`bwd_bwd`
  handles and call those handles directly.
- A later all-variant H100 run began compiling `baseline_notma`; it was
  interrupted before a JSON result was emitted.

No trustworthy new perf number was produced before Modal was stopped.

## Ranking

Current ranking by confidence:

1. `qk_direct`: best known working option. Already compile/smoke OK on H100 and
   H200 from the base probe; still no evidence of true WS transform.
2. `qk_recompute`: local temp source generation OK; expected slower than
   qk-dot cache, but useful to test whether removing the global `QK_DOT` read
   changes TMA behavior.
3. `qk_direct_smem_bias`: local temp source generation OK; removes some
   Q/K fragment traffic, but untested on GPU.
4. `qk_dot_rs_layout`: local temp source generation OK; tests alternate
   `QK_DOT` memory layout, untested on GPU.
5. `qk_serial_p`: known compile failure from previous probe.
6. `layout_patch_tma_ws`: known compile failure from previous probe.

No variant has yet demonstrated performance faster than the non-TMA baseline or
a true WS trigger.
