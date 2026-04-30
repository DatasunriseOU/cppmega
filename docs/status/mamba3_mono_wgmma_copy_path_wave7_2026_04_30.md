# Mamba3 Mono WGMMA Copy Evidence Wave7 - 2026-04-30

Status: active, red until full alignment evidence is attached.
Canonical receipt:
`docs/status/mamba3_mono_wgmma_copy_path_wave7_receipt_2026_04_30.json`

## Scope

Wave7 adds an evidence ingestion layer on top of the Wave6 copy-path model:

- `tools/probes/mamba3_wgmma_wave7_copy_evidence.py`
- `tools/probes/mamba3_wgmma_wave7_copy_probe.cu`
- `tests/test_mamba3_wgmma_wave7_copy_evidence.py`

The receipt is intentionally failing by default. It can be promoted only when
both evidence classes are attached:

```text
python tools/probes/mamba3_wgmma_wave7_copy_evidence.py \
  --ptxas-log ptxas.log \
  --dynamic-smem-bytes 118784 \
  --alignment-evidence alignment.json
```

Stable placeholder check:

```text
python tools/probes/mamba3_wgmma_wave7_copy_evidence.py \
  --check docs/status/mamba3_mono_wgmma_copy_path_wave7_receipt_2026_04_30.json
```

## Local ptxas Evidence

This host has CUDA 13.2 and GB10 (`sm_121`). The minimal CUDA/CuTe one-tile
copy probe compiled successfully:

```text
python tools/probes/mamba3_wgmma_wave7_copy_evidence.py \
  --compile-probe --cuda-arch sm_121
```

Captured ptxas/resource fields:

| field | value |
| --- | ---: |
| registers/thread | `40` |
| static SMEM bytes | `0` |
| dynamic SMEM bytes | `8192` |
| spill stores bytes | `0` |
| spill loads bytes | `0` |

Retained evidence:
`docs/status/mamba3_mono_wgmma_copy_path_wave7_local_ptxas_2026_04_30.json`

This is compiler/toolchain evidence for a minimal one-tile `uint4` copy probe.
It is not full monolithic WGMMA kernel evidence, and it does not by itself
prove all 12 Wave6 tile movements or the `K_T/Q_T` transpose layout.

## Alignment Evidence Gate

The placeholder receipt remains red because the following full-kernel evidence
is still missing:

- `vector_bytes=16`, `tile_rows=64`, `tile_cols=64`, `dtype_bytes=2`
- `row_bytes=128`, `tile_bytes=8192`, `row_tail_bytes=0`,
  `tile_tail_bytes=0`
- global and SMEM base alignment at least `16 B`
- runtime global and SMEM alignment guards
- row stride alignment as a multiple of `16 B`
- generated copy path uses a 16-byte contiguous vector type
- generated code has no masked tail path
- `tiles_covered` contains all 12 Wave6 large-copy tiles
- `K_T/Q_T` prove a vector-compatible SMEM layout/view, not scalar scatter

## TMA Descriptor Smoke Checklist

Before the TMA/cp.async path can claim yellow/green:

1. Descriptor/tensor-map construction succeeds for exactly the 10 global
   Wave6 tiles: `K`, `Q`, `K_T`, `Q_T`, `state_panel_p0/1`,
   `dPhiO_panel_p0/1`, and `PsiV_panel_p0/1`.
2. No descriptor is attempted for `dstates_panel_p0/1` or tiny scalar/vector
   slices.
3. Every descriptor reports rank `2`, `box_dim_elements=[64, 64]`,
   `element_type=bf16`, and `8192 B` per tile.
4. Aggregate expected bytes are `81920 B/chunk`.
5. Global pointer, shared pointer, and transfer size are all 16-byte aligned.
6. mbarrier expected-byte accounting is set before waits.
7. Wait/fence ordering is before any WGMMA consumer reads the destination SMEM.
8. TMA ptxas resources are `<=192` regs/thread, `<=131072 B` total SMEM, and
   zero spills.

## Wave8 Recommendation

Implement the narrow-vector path first, not TMA. Attach the generated ptxas log
and the full 12-tile alignment JSON to the Wave7 receipt, then use the receipt
to decide whether the path can enter timing. Keep TMA behind the smoke checklist
until descriptor scope, mbarrier byte accounting, and wait/fence ordering are
receipted.
