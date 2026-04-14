# Reproducer: DSA CUDA graph capture breaks on CPU-sync ops

Upstream Megatron
`megatron/core/transformer/experimental_attention_variant/dsa.py` contains
several calls that implicitly synchronize the CUDA stream:

- `torch.equal(finite, expected)` (validation of indexer outputs)
- `torch.equal(key_positions, expected_key_pos)`
- `torch.equal(mask[bi], ref_mask)`
- `torch.any(idx_chunk < 0)` + `valid_topk.any()` branches inside
  `_scatter_topk_into_index_mask`

All of these lower to `.item()`, which is forbidden inside
`torch.cuda.graph(...)` capture and raises
`cudaErrorStreamCaptureUnsupported`. They block `--cuda-graph-impl
transformer_engine` on any NAM56R-style config with DSA enabled.

See `upstream_prs/01_dsa_cuda_graph_safety.md` for the full problem
statement and fix (branchless clamp+scatter+fixup + gating the
`torch.equal` checks behind `torch.cuda.is_current_stream_capturing()`).

## Run

```bash
pip install -r requirements.txt
python reproducer.py
```

Requires a CUDA device.

## Expected output — bug present (upstream today)

```
(A) UNPATCHED — expecting CUDA graph capture to FAIL
  capture raised (as expected): RuntimeError: ... stream is capturing ...
  BUG_REPRODUCED

(B) PATCHED — expecting CUDA graph capture to SUCCEED
  capture OK, replay matches eager reference exactly
  FIX_VALIDATED

VERDICT: bug reproduced on unpatched + fix validated on patched.
```

Exit code: **0**.

## Expected output — bug fixed upstream

If upstream merges the fix, the unpatched module will capture cleanly:

```
(A) UNPATCHED — expecting CUDA graph capture to FAIL
  capture UNEXPECTEDLY succeeded — bug is not present on this build.
...
VERDICT: BUG NOT REPRODUCED on unpatched module.
```

Exit code: **1**.

## What the patched path does

See `cppmega/megatron/upstream_patches/apply_dsa_cg_patches.py` (Patch 1
and Patch 8) for the production version applied to Megatron.

- Validation checks (`torch.equal`) are gated off (`if False:`); a safer
  alternative is
  `if not torch.cuda.is_current_stream_capturing(): assert torch.equal(...)`.
- `_scatter_topk_into_index_mask` is rewritten branchless:

  ```python
  sentinel = idx_chunk < 0
  safe_chunk = idx_chunk.clamp(min=0)
  index_mask[:, s0:s1].scatter_(-1, safe_chunk, 0.0)
  has_sent = sentinel.any(dim=-1)                       # [b, chunk_len]
  has_real0 = ((idx_chunk == 0) & ~sentinel).any(dim=-1)
  fixup = has_sent & ~has_real0
  index_mask[:, s0:s1, 0].masked_fill_(fixup, float("-inf"))
  ```

  The `.any(dim=-1)` calls reduce along the last dim only (returning a
  tensor, not a scalar) — no `.item()` is invoked, so capture proceeds.
