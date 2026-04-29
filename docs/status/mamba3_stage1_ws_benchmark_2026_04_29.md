# Mamba3 stage-1 WS benchmark status, 2026-04-29

Branch/worktree:

- Worktree: `/home/dave/source/cppmega/.claude/worktrees/mamba3-stage1-ws-benchmark`
- Branch: `worker/mamba3-stage1-ws-benchmark`
- Base image tag: `GHCR_TAG=785c3fd`

Harness:

- `scripts/modal_mamba3_stage1_ws_benchmark.py`
- Modal app name: `cppmega-mamba3-stage1-ws-benchmark`
- Modal Volume: `cppmega-mamba3-benchmarks`
- Remote function timeout: 600 seconds

Variants:

- `baseline`: upstream TileLang MIMO backward, non-TMA/non-WS, `bf_num_stages=0`, `bb_num_stages=0`
- `qk_direct_non_ws`: `mamba3_bwd_layout_fix.patch`, flattened Q/K and QK_DOT, non-WS, `bf_num_stages=0`, `bb_num_stages=0`
- `qk_shared_direct_stage1_ws`: layout fix plus qk_shared_direct rewrite, TMA/WS pass configs enabled, `bf_num_stages=1`, `bb_num_stages=1`

Representative shapes:

- `smoke`: `B=1 S=256 H=4 G=1 N=64 P=64 R=4`
- `representative`: `B=2 S=1024 H=16 G=1 N=64 P=64 R=4`
- `productionish`: `B=4 S=4096 H=32 G=1 N=64 P=128 R=4`

Run commands:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 10m modal run scripts/modal_mamba3_stage1_ws_benchmark.py --run-id h200_representative_20260429T1138Z --shape-csv representative --iters 8 --warmup 2
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 10m modal run scripts/modal_mamba3_stage1_ws_benchmark.py --run-id h200_productionish_20260429T1142Z --shape-csv productionish --iters 4 --warmup 1
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100:2 timeout 10m modal run scripts/modal_mamba3_stage1_ws_benchmark.py --run-id h100_representative_20260429T1147Z --shape-csv representative --iters 6 --warmup 2
```

Modal hygiene:

- Checked `modal app list --json` before each run.
- Waited for unrelated live apps to stop before launching more H200 work.
- All harness runs used local `timeout 10m` and remote function `timeout=600`.
- Explicit `modal app stop -y <app-id>` was attempted after each run; Modal reported each app was already stopped after local entrypoint completion.

Run records:

| GPU | Shape | Modal app id | Local log path | Modal Volume path |
| --- | --- | --- | --- | --- |
| H200:2 | representative | `ap-6PN6By2VtfJ1MRrWiP7GdX` | `.modal_logs/h200_representative_20260429T1138Z.log` | `/mamba3_stage1_ws_benchmark/h200_representative_20260429T1138Z` |
| H200:2 | productionish | `ap-qcrBX48l9ty7Nr4H2OrZ0h` | `.modal_logs/h200_productionish_20260429T1142Z.log` | `/mamba3_stage1_ws_benchmark/h200_productionish_20260429T1142Z` |
| H100:2 | representative | `ap-4BQa1ayYUmVNPCGbm1ss2S` | `.modal_logs/h100_representative_20260429T1147Z.log` | `/mamba3_stage1_ws_benchmark/h100_representative_20260429T1147Z` |

Results:

| GPU | Shape | Variant | bwd_fwd ms | bwd_bwd ms | chain ms | chain speedup vs baseline | max main grad abs diff | WS confirmed | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| H200 | representative | baseline | 0.2731 | 0.6607 | 0.9364 | 1.0000 | n/a | no | non-TMA/non-WS |
| H200 | representative | qk_direct_non_ws | n/a | n/a | n/a | n/a | n/a | no | compile failed: TileLang `InternalError`, divide by zero |
| H200 | representative | qk_shared_direct_stage1_ws | 0.2841 | 0.7281 | 1.0076 | 0.9293 | 0.0 | yes | bwd_fwd/bwd_bwd TMA loads 5/8 |
| H200 | productionish | baseline | 1.8600 | 3.7127 | 5.5514 | 1.0000 | n/a | no | non-TMA/non-WS |
| H200 | productionish | qk_direct_non_ws | n/a | n/a | n/a | n/a | n/a | no | compile failed: TileLang `InternalError`, divide by zero |
| H200 | productionish | qk_shared_direct_stage1_ws | 1.7533 | 4.0551 | 5.7876 | 0.9592 | 1.00e-9 | yes | bwd_fwd faster, bwd_bwd/chain slower |
| H100 | representative | baseline | 0.3196 | 0.6628 | 0.9430 | 1.0000 | n/a | no | non-TMA/non-WS |
| H100 | representative | qk_direct_non_ws | n/a | n/a | n/a | n/a | n/a | no | compile failed: TileLang `InternalError`, divide by zero |
| H100 | representative | qk_shared_direct_stage1_ws | 0.3177 | 0.7326 | 1.0173 | 0.9270 | 0.0 | yes | bwd_fwd parity, bwd_bwd/chain slower |

Diff summary:

- H200 representative: `qk_shared_direct_stage1_ws` matched baseline exactly across checked outputs (`max_main_grad_abs_diff=0.0`).
- H200 productionish: max main gradient absolute diff was `1.00e-9`; qk_dot/states matched exactly.
- H100 representative: `qk_shared_direct_stage1_ws` matched baseline exactly across checked outputs (`max_main_grad_abs_diff=0.0`).
- `qk_direct_non_ws` did not produce correctness or timing data because the old qk_direct path still hits TileLang compile-time divide-by-zero.

Production decision:

- No-go for production default as benchmarked.
- Rationale: stage-1 WS is correct and true WS is confirmed, but it regresses end-to-end chain timing by 4.1-7.3% on the measured H200/H100 shapes, driven by a consistent bwd_bwd regression of roughly 9-10%. The only bwd_fwd win was H200 productionish (+6.1%), which did not offset bwd_bwd.
- Old `qk_direct_non_ws` remains non-viable because it does not compile.
