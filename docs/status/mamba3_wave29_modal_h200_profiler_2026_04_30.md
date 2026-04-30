# Mamba3 Wave29 Lane A Modal H200 Profiler - 2026-04-30

Branch: `worker/mamba3-wave29-modal-h200-profiler`

Base: `fdd7546`

Scope: Modal H200:2 gate/profiler harness plus captured results. No runtime
model or kernel default was changed.

## Added Harness

`scripts/modal_mamba3_wave29_h200_profiler.py`

- Uses GHCR prebuilt image `ghcr.io/jewelmusicee/cppmega:785c3fd`.
- Uses Modal GPU spec `H200:2`.
- Mounts Modal volumes:
  - `/vol`: `cppmega-mamba3-benchmarks`
  - `/cache`: `cppmega-modal-cache`
- Overlays local `cppmega/`, `upstream_prs/`, and `pyproject.toml` onto
  `/opt/cppmega`.
- Refuses full gate invocations with fewer than 20 train steps.
- Compares baseline against `stage2_force_nontma_bf1_bb0` by mutating the
  installed `mamba_ssm` TileLang file only inside the Modal container.
- Rolls back stage2 mutation in `finally` and records final kernel markers.
- Uses real Modal-volume data only if present; otherwise creates indexed
  synthetic data and labels the result `synthetic_full_shape_mock_data`.
- Records per-step timing when available, tok/sec when available, peak GPU
  memory from the production atexit hook, kernel marker counts, command JSON,
  and log tail.
- Profile mode enables `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` for future reruns and
  supports `CPPMEGA_WAVE29_ATTN_BACKEND=<flash|auto|unfused|...>` fallback
  experiments.

## Commands Run

```bash
modal run scripts/modal_mamba3_wave29_h200_profiler.py::launch_preflight \
  --run-id wave29_h200_preflight_20260430

modal run scripts/modal_mamba3_wave29_h200_profiler.py::launch_gate \
  --run-id wave29_h200_gate_profile_20260430 \
  --train-iters 20 \
  --profile \
  --timeout-per-variant-s 2400

modal app list --json
```

Local validation:

```bash
python -m py_compile scripts/modal_mamba3_wave29_h200_profiler.py
```

## Modal Apps

| app id | description | status |
| --- | --- | --- |
| `ap-W8EYZX725vU4k1qpqu66XJ` | `cppmega-wave29-modal-h200-profiler` preflight | stopped |
| `ap-GIMlEL0zRhqkQYugT8s6lg` | `cppmega-wave29-modal-h200-profiler` 20-step gate/profile attempt | stopped |

`modal app list --json` also showed a live `cppmega-wave29-lane-c-h100` app
owned by another concurrent lane. It was not started by this lane and was not
stopped.

## Preflight Result

Artifact:
`artifacts/mamba3_wave29_modal_h200_profiler/wave29_h200_preflight_20260430/preflight.json`

| field | value |
| --- | --- |
| GPU | 2 x NVIDIA H200 |
| image | `ghcr.io/jewelmusicee/cppmega:785c3fd` |
| torch | `2.13.0.dev20260426+cu132` |
| CUDA | `13.2` |
| Transformer Engine | `2.16.0.dev0+8e19460b` |
| TileLang | `0.1.8+cu132.gitf309d814` |
| stage2 apply | passed; markers `flat_q=true`, `flat_qk=true`, `bf_num_stages_1=true`, `bb_num_stages_0=true`, `disable_tma_count=13` |
| rollback | passed; final kernel clean with `disable_tma_count=0`, `flat_q=false`, `flat_qk=false` |

The pre-rollback check returned nonzero only because no backup existed yet and
the baseline kernel was already clean. The final rollback after apply succeeded.

## H200:2 Gate/Profile Result

Artifact:
`artifacts/mamba3_wave29_modal_h200_profiler/wave29_h200_gate_profile_20260430/result.json`

Dataset status: `synthetic_full_shape_mock_data`. No real
`clang_semantic_4k_v10_train_text_document` + HF tokenizer pair was present on
the Modal volume, so the harness generated indexed mock data under
`/vol/mock_data` and did not claim a production real-data result.

Command shape:

| knob | value |
| --- | --- |
| GPUs | `H200:2` |
| nproc | 2 |
| TP/PP/VPP/EP | `1/2/2/1` |
| hidden/ffn/heads | `3584/18944/28` |
| seq len | 4096 |
| MBS/GBS | `1/8` |
| MTP depths | 2 |
| train iters requested | 20 |
| tokens/iter | 32768 |
| profile mode | true |

Result table:

| variant | steps seen | tok/sec | peak alloc GiB | peak reserved GiB | status |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0 | n/a | 16.983 | 17.107 | blocked before step 1 |
| `stage2_force_nontma_bf1_bb0` | n/a | n/a | n/a | n/a | not run because baseline gate failed |

Blocker:

```text
ValueError: No dot product attention backend is available for the provided inputs.
Please run with NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 to find out the reasons for disabling all backends.
```

The exception occurred in the baseline forward path before reaching training
iteration 1:

```text
cppmega/megatron/mla_shared.py -> Megatron multi_latent_attention.py ->
transformer_engine/pytorch/attention/dot_product_attention.py
```

The app exited normally after recording the blocker and final rollback state.
No profiler trace for Mamba3 backward was feasible because the full path failed
in the attention backend before Mamba3 backward launched.

## Research Notes And Optimization Ideas

Sources checked with MCP search:

- NVIDIA Hopper Tuning Guide:
  https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- Transformer Engine attention/debug docs:
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html
- TileLang builtin docs:
  https://tilelang.com/autoapi/tilelang/language/builtin/index.html
- PyTorch Hopper TMA deep dive:
  https://pytorch.org/blog/hopper-tma-unit/
- Colfax CUTLASS TMA tutorial:
  https://research.colfax-intl.com/tutorial-hopper-tma/
- upstream Mamba3 module:
  https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba3.py
- upstream Mamba3 TileLang backward:
  https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py
- Modal GPU/profiling docs:
  https://modal.com/docs/guide/gpu and https://modal.com/docs/examples/torch_profiling
- Mamba backward optimization issue:
  https://github.com/state-spaces/mamba/issues/530

Concrete next ideas:

1. Rerun the exact full gate with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` already
   wired by profile mode to identify whether TE rejected the flash/cuDNN
   backend due to MLA QKV layout, head dimension, mask type, dropout, or dtype.
2. Add a controlled fallback run with `CPPMEGA_WAVE29_ATTN_BACKEND=auto` or
   `unfused` to unblock the full Mamba3 path. Keep it labeled as a fallback
   profiler run, not the production flash-attention gate.
3. If the fallback reaches Mamba3 backward, compare baseline vs stage2 using
   the recorded kernel markers: baseline has no targeted `disable_tma=True`
   copies, while stage2 should show `disable_tma_count=13`, flat Q/QK layout,
   `bf_num_stages=1`, and `bb_num_stages=0`.
4. Use Hopper TMA only on copy patterns that amortize descriptor/barrier cost.
   NVIDIA and PyTorch both emphasize TMA's benefit for large async tensor
   transfers and producer/consumer overlap; the stage2 hypothesis remains that
   `bwd_fwd` benefits while `bwd_bwd` regresses on smaller/vector-like copies.
5. Reduce shared-memory live range in `mamba_mimo_bwd_bwd`. The upstream
   TileLang bwd file allocates many shared buffers; stage2 already removes
   some TMA lowering and flattens QK, but the next lane should inspect whether
   `dA_cs`/`dA_cs_rev` staging and `qk_dot_full_shared` can be narrowed or
   split.
6. Tune producer/consumer register allocation if TMA/warp specialization is
   re-enabled. TileLang exposes `set_max_nreg`, producer/consumer register
   annotations, and warp-group register controls; Hopper's register limit and
   TMA's register-saving design make this a plausible lever.
7. Revisit scan-backward auxiliary operations. The Mamba issue about replacing
   expensive reverse-copy/cumsum-like work in backward is directionally relevant
   to Mamba3's backward scans and `dA`/`dt` suffix-style reductions.
8. Add Modal torch profiler output once the attention blocker is bypassed.
   Modal documents PyTorch profiler/TensorBoard workflows; saving traces to the
   existing volume would give kernel names, CUDA time, and CPU launch overhead
   without requiring an interactive Nsight UI.
9. Keep real-data gating separate from synthetic full-shape gating. Modal H200
   can run the full shape with synthetic data, but production claims require the
   real indexed dataset/tokenizer pair to be present on the volume.
