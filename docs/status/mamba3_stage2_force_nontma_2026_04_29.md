# Mamba3 Stage2 Force Non-TMA Slice Probe - 2026-04-29

Branch: `worker/mamba3-stage2-force-nontma`

Goal: make Mamba3 MIMO backward true WS usable with `num_stages > 1` on
Hopper by keeping the large Q/K/QK_DOT copies TMA-capable while forcing the
small rank-3 vector-slice copies off TMA.

## Prior Result

Stage1 proved that true producer/consumer WS fires when:

- Q/K are flattened from `[B, S, R, G, N]` to `[B, S*R, G, N]`;
- QK_DOT is flattened from `[B, H, S, R, R]` to `[B, H, S, R*R]`;
- `num_stages >= 1` is used.

Stage1 `num_stages=2` compiled and WS fired, but runtime smoke failed with:

```text
Failed to initialize the TMA descriptor 716
CUDA_ERROR_MISALIGNED_ADDRESS
format CU_TENSOR_MAP_DATA_TYPE_FLOAT32
dim 3
globalDim [64, 4, 1]
globalStridesRaw [4, 256, 1024]
boxDim [16, 1, 1]
```

That descriptor matches small float32 `[B, H, S]` vector slices, not the large
Q/K/QK_DOT paths we need for WS producer copies.

## Stage2 Patch

Patch:

- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_force_nontma.patch`

Harness:

- `scripts/modal_mamba3_stage2_force_nontma_probe.py`

The patch is still non-production. It removes the deprecated global
`TL_DISABLE_TMA_LOWER` pass config and uses per-copy `disable_tma=True` on the
small vector-slice copies:

- `TRAP` and `DT` chunk vector loads;
- `DA_CS` and `DA_CS_REV` chunk vector loads;
- `DGAMMA_DIAG`, `DDA_CS_REV`, `DFACTOR`, `DDA`, and `DDA_CS` chunk vector stores.

The patch deliberately does not add `disable_tma=True` to Q/K/QK_DOT. Those
large copies remain TMA-capable so `producer_consumer_ws.cc::ClassifyCopy` can
still classify them as WS producers.

## Local Validation

```text
patch --dry-run -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_stage2_force_nontma.patch
checking file /tmp/.../mamba3_mimo_bwd.py

python -m py_compile scripts/modal_mamba3_stage2_force_nontma_probe.py
```

Additional static check from the patch text:

- `disable_tma=True` count: 13;
- no added `TL_DISABLE_TMA_LOWER` false override;
- Q/K/QK_DOT large-copy paths do not include `disable_tma=True`.

## Modal Hygiene

The stage2 Modal harness uses:

- app name: `cppmega-mamba3-stage2-force-nontma-probe`;
- default GPU: `H200:2`, override with `CPPMEGA_MODAL_GPU`;
- function timeout: 600 seconds;
- default stage matrix: `(2,0)`, `(0,2)`, `(2,2)`, `(1,1)`.

Run at most one app per GPU class at a time and stop each app after collecting
the JSON result.

## Modal Results

H200 bounded run:

```text
CPPMEGA_MODAL_GPU=H200:2 \
CPPMEGA_MAMBA3_STAGE_MATRIX='2,0;0,2;2,2' \
timeout 600 modal run scripts/modal_mamba3_stage2_force_nontma_probe.py
```

App:

| GPU | App ID | Status | Notes |
| --- | --- | --- | --- |
| H200 | `ap-oWeMR0sUwyf1KNCpZ5D7Kd` | completed, already stopped at `2026-04-29 11:39:11+00:00` | One app, bounded under 10 minutes. |

Environment:

- device: `NVIDIA H200`, capability `(9, 0)`, device count `2`;
- image: `ghcr.io/jewelmusicee/cppmega:latest`;
- Torch: `2.13.0.dev20260426+cu132`;
- CUDA: `13.2`;
- TileLang: `0.1.8+cu132.gitf309d814`.

Compile summary:

| `bf_num_stages` | `bb_num_stages` | bwd_fwd WS | bwd_bwd WS | Status |
| --- | --- | --- | --- | --- |
| 2 | 0 | yes | no | compiled |
| 0 | 2 | no | yes | compiled |
| 2 | 2 | yes | yes | compiled |
| 1 | 1 | yes | yes | compiled |

Patched-source checks reported by the harness for each compile/smoke:

- `deprecated_global_tma_disable_count`: 0;
- `disable_tma_count`: 13;
- `large_q_copy_disable_tma`: false;
- `large_k_copy_disable_tma`: false;
- `qk_copy_disable_tma`: false;
- `qk_tma_capable_copy_count`: 1.

Smoke summary:

| `bf_num_stages` | `bb_num_stages` | Status | `qk_dot_absmax` | `dq_absmax` | `dk_absmax` | `dv_absmax` |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0 | smoke_ok | 0.005767822265625 | 6.148e-10 | 1.295e-09 | 3.449e-09 |
| 0 | 2 | smoke_ok | 0.005767822265625 | 6.148e-10 | 1.295e-09 | 3.449e-09 |
| 2 | 2 | smoke_ok | 0.005767822265625 | 6.148e-10 | 1.295e-09 | 3.449e-09 |
| 1 | 1 | smoke_ok | 0.005767822265625 | 6.148e-10 | 1.295e-09 | 3.449e-09 |

The previous stage2 failure signature did not recur: no TMA descriptor 716 and
no `CUDA_ERROR_MISALIGNED_ADDRESS` appeared in the H200 smoke output.

## Follow-Up: Smem-Safe Default

The `(2,2)` path later crashed on the productionish shape at launch time with:

```text
Failed to set the allowed dynamic shared memory size to 231712
```

The patch default was therefore moved to `bf_num_stages=1` and
`bb_num_stages=1`. The earlier smoke matrix already showed `(1,1)` keeps WS in
both kernels, and the follow-up H200 benchmark confirms the productionish shape
now launches successfully.

Detailed result:

- `docs/status/mamba3_stage2_force_nontma_smemsafe_2026_04_29.md`
