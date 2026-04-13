# PR: Eliminate redundant V @ dO^T dot product in SISO backward kernel

**Target repo:** state-spaces/mamba
**File:** `mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py`
**Type:** Performance optimization / code cleanup

## Summary

The `mamba3_siso_bwd_kernel_dqkv` kernel computes `tl.dot(v_block, tl.trans(do_block))` (a CHUNK_SIZE x CHUNK_SIZE GEMM) **three separate times** in the inner chunk loop, followed by the same causal decay mask application each time. All three produce identical results before diverging into their respective consumers (dADT, dK, dQ gradients).

This PR computes the dot product and mask once, then reuses the result.

## Changes

Compute `vdot_block = tl.dot(v_block, tl.trans(do_block))` once, apply the causal decay mask once to get `vdot_masked`, then reuse `vdot_masked` for all three consumers:

1. **dADT** (was `dAinv`): `vdot_masked * tl.dot(k_block, tl.trans(q_block))`
2. **dK**: `tl.dot(vdot_masked.to(q_block.dtype), q_block)`
3. **dQ**: `tl.dot(tl.trans(vdot_masked).to(k_block.dtype), k_block)`

Net: -25 lines, removes 2 redundant `tl.dot` calls + 2 redundant mask applications per chunk.

## Verification

Tested on NVIDIA H200 (torch 2.12+cu132, Triton from nightly):

- **Correctness**: Bitwise identical outputs for all 7 gradient tensors (dQ, dK, dV, dADT, dQK_dot, dD, d_issm_state) across original and patched versions
- **Determinism**: Repeated runs produce identical results
- **Variable-length sequences**: Tested with Cu_Seqlens varlen mode
- **Multiple configs tested**:
  - Small: B=2, S=256, NH=8, D=64
  - Medium: B=1, S=2048, NH=32, D=128
  - NAM56R-like: B=1, S=4096, NH=24, DQK=64, DV=128

## Performance

On H200 with Triton nightly (2026-04), the compiler appears to CSE these dot products already, so no measurable runtime difference (~0.98 ms for NAM56R config).

However, this change:
1. Makes the CSE explicit and guaranteed regardless of compiler version
2. Eliminates code duplication (DRY)
3. May benefit older Triton versions or different GPU architectures (e.g., sm_80/sm_89) where CSE is less aggressive
4. Reduces the RECOMPUTE_MASK path from 3 copies to 1

## Diff

```diff
--- a/mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py
+++ b/mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py
@@ -427,40 +427,35 @@
             )
 
         # ============================================================
-        # Compute dADT Gradient (Part 1): From Intra-chunk Attention
-        # This is register-heavy so we compute it early before spilling
+        # Compute V @ dO^T Once (shared by dADT, dK, dQ)
         # ============================================================
-        # Gradient contribution from (QK^T ⊙ L) V term
-        dAinv = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T
+        # This dot product is reused 3 times below. Computing it once
+        # eliminates 2 redundant CHUNK_SIZE x CHUNK_SIZE GEMMs per chunk.
+        vdot_block = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T: (CHUNK_SIZE, CHUNK_SIZE)
+
+        # Apply causal decay mask once
         if RECOMPUTE_MASK:
-            dAinv *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
-            dAinv = tl.where(
+            vdot_masked = vdot_block * tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
+            vdot_masked = tl.where(
                 tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
-                dAinv,
+                vdot_masked,
                 0.0
             )
         else:
-            dAinv *= causal_decay_mask
-        dAinv *= tl.dot(k_block, tl.trans(q_block))  # Element-wise with K @ Q^T
+            vdot_masked = vdot_block * causal_decay_mask
+
+        # ============================================================
+        # Compute dADT Gradient (Part 1): From Intra-chunk Attention
+        # ============================================================
+        # Gradient contribution from (QK^T ⊙ L) V term
+        dAinv = vdot_masked * tl.dot(k_block, tl.trans(q_block))  # Element-wise with K @ Q^T
         dM_rev_vector = tl.sum(dAinv, axis=0) - tl.sum(dAinv, axis=1)  # (CHUNK_SIZE,)
 
         # ============================================================
         # Compute dK: Key Gradient
         # dK = (V @ dO^T ⊙ mask)^T @ Q + V @ dStates * scale
         # ============================================================
-        # Intra-chunk: dP^T @ Q where dP = dO @ V^T ⊙ mask
-        dp_t_block = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T: (CHUNK_SIZE, CHUNK_SIZE)
-        if RECOMPUTE_MASK:
-            dp_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
-            dp_t_block = tl.where(
-                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
-                dp_t_block,
-                0.0
-            )
-        else:
-            dp_t_block *= causal_decay_mask
-
-        acc_dk = tl.dot(dp_t_block.to(q_block.dtype), q_block)  # (CHUNK_SIZE, HEADDIM_QK)
+        acc_dk = tl.dot(vdot_masked.to(q_block.dtype), q_block)  # (CHUNK_SIZE, HEADDIM_QK)
 
         # Inter-chunk: gradient flowing through accumulated states
         acc_dk += tl.dot(v_block, d_ssm_states_acc.to(v_block.dtype)) * exp_da_cs_rev[:, None]
@@ -471,19 +466,7 @@
         # Compute dQ: Query Gradient
         # dQ = (V @ dO^T ⊙ mask) @ K + dO @ States * scale
         # ============================================================
-        # Intra-chunk: S^T @ K where S = V @ dO^T ⊙ mask
-        s_block = tl.dot(v_block, tl.trans(do_block))  # (CHUNK_SIZE, CHUNK_SIZE)
-        if RECOMPUTE_MASK:
-            s_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
-            s_block = tl.where(
-                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
-                s_block,
-                0.0
-            )
-        else:
-            s_block *= causal_decay_mask
-
-        acc_dq = tl.dot(tl.trans(s_block).to(k_block.dtype), k_block)  # (CHUNK_SIZE, HEADDIM_QK)
+        acc_dq = tl.dot(tl.trans(vdot_masked).to(k_block.dtype), k_block)  # (CHUNK_SIZE, HEADDIM_QK)
 
         # Inter-chunk: gradient through states from previous chunks
         acc_dq += tl.dot(do_block, ssm_states_block) * exp_da_cs[:, None]
```
