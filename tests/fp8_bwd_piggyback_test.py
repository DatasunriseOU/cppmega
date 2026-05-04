#!/usr/bin/env python3
# FP8 bwd piggyback micro-test on GB10.
# Goal: see whether TE Float8Quantizer (E5M2) output can be fed into the
# existing TileLang sparse_mla_bwd_fp8 kernel (which currently hard-codes
# dO in BF16).

import os, sys, traceback, torch

sys.path.insert(0, "/home/dave/cppmega/cppmega")  # package root

from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
import transformer_engine_torch as tex

from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_fwd_fp8 import (
    per_token_cast_to_fp8,
    sparse_mla_fwd_fp8_interface,
)
from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_bwd_fp8 import (
    sparse_mla_bwd_fp8,
)

torch.manual_seed(0)
device = "cuda"
B, S, H, kv_group, topk = 1, 256, 32, 1, 64
D, D_tail = 128, 64  # DeepSeek MLA: d_v=128, tail=64 (rope)

q_bf16  = torch.randn(B, S, H, D + D_tail, device=device, dtype=torch.bfloat16) * 0.1
kv_bf16 = torch.randn(B, S, kv_group, D + D_tail, device=device, dtype=torch.bfloat16) * 0.1
indices = torch.randint(0, S, (B, S, kv_group, topk), device=device, dtype=torch.int32)
do_bf16 = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) * 0.1

# --- 1. Forward (FP8) to produce O and LSE ----------------------------------
q_fp8_flat,  q_scale_flat  = per_token_cast_to_fp8(q_bf16.reshape(-1, H * (D + D_tail)))
kv_fp8_flat, kv_scale_flat = per_token_cast_to_fp8(kv_bf16.reshape(-1, kv_group * (D + D_tail)))
q_fp8  = q_fp8_flat.reshape(q_bf16.shape).contiguous()
kv_fp8 = kv_fp8_flat.reshape(kv_bf16.shape).contiguous()
q_scale  = q_scale_flat.reshape(B, S).contiguous()
kv_scale = kv_scale_flat.reshape(B, S).contiguous()

tl_out, tl_lse = sparse_mla_fwd_fp8_interface(
    q_fp8, kv_fp8, indices,
    q_scale=q_scale, kv_scale=kv_scale,
    sm_scale=1.0 / (D ** 0.5), d_v=D,
)
print("[fwd-fp8] out", tl_out.shape, tl_out.dtype, "lse", tl_lse.shape, tl_lse.dtype)

# --- 2. Baseline: BF16 dO into FP8 bwd ---------------------------------------
dq_bf, dkv_bf = sparse_mla_bwd_fp8(
    q_fp8, kv_fp8, q_scale, kv_scale,
    tl_out, do_bf16.contiguous(), indices, tl_lse,
    sm_scale=1.0 / (D ** 0.5),
)
print("[bwd-fp8 BF16-dO] dq", dq_bf.shape, dq_bf.dtype, "dkv", dkv_bf.shape, dkv_bf.dtype)

# --- 3. Quantize dO to E5M2 via TE Float8Quantizer --------------------------
scale = torch.ones(1, device=device)
amax  = torch.zeros(1, device=device)
qz    = Float8Quantizer(scale=scale, amax=amax, fp8_dtype=tex.DType.kFloat8E5M2)
do_f8 = qz(do_bf16)
print("[quant dO] Float8Tensor dtype=", do_f8.dtype, "raw _data.dtype=", do_f8._data.dtype,
      "scale_inv=", do_f8._scale_inv.item(), "amax=", qz.amax.item())

# Reinterpret as torch.float8_e5m2 tensor that the kernel can consume.
do_e5m2_raw = do_f8._data.view(torch.float8_e5m2).contiguous()
print("[dO as e5m2] shape=", do_e5m2_raw.shape, "dtype=", do_e5m2_raw.dtype)

# --- 4. Attempt A: pass raw E5M2 dO directly -----------------------
print("\n=== ATTEMPT A: pass E5M2 dO directly into existing FP8 bwd ===")
try:
    dq_a, dkv_a = sparse_mla_bwd_fp8(
        q_fp8, kv_fp8, q_scale, kv_scale,
        tl_out, do_e5m2_raw, indices, tl_lse,
        sm_scale=1.0 / (D ** 0.5),
    )
    print("[A] SUCCESS dq", dq_a.dtype, "dkv", dkv_a.dtype)
except Exception as e:
    print("[A] FAIL:", type(e).__name__, str(e)[:400])

# --- 5. Attempt B: dequantize dO back to BF16 through TE ---------------------
print("\n=== ATTEMPT B: TE dequant(E5M2) -> BF16, pass into existing kernel ===")
try:
    do_bf16_roundtrip = do_f8.dequantize().to(torch.bfloat16).contiguous()
    dq_b, dkv_b = sparse_mla_bwd_fp8(
        q_fp8, kv_fp8, q_scale, kv_scale,
        tl_out, do_bf16_roundtrip, indices, tl_lse,
        sm_scale=1.0 / (D ** 0.5),
    )
    rel_dq  = (dq_b.float() - dq_bf.float()).abs().max() / dq_bf.float().abs().max().clamp(min=1e-6)
    rel_dkv = (dkv_b.float() - dkv_bf.float()).abs().max() / dkv_bf.float().abs().max().clamp(min=1e-6)
    print(f"[B] OK rel_err dq={rel_dq.item():.3e} dkv={rel_dkv.item():.3e}")
except Exception as e:
    print("[B] FAIL:", type(e).__name__, str(e)[:400])

# --- 6. TileLang cuDNN backend probe ----------------------------------------
print("\n=== TileLang cuDNN probe ===")
from tilelang.utils.target import SUPPORTED_TARGETS
print("SUPPORTED_TARGETS =", list(SUPPORTED_TARGETS.keys()))
# Is there any cuDNN-related symbol in tilelang?
import tilelang, pkgutil
matched = []
for m in pkgutil.walk_packages(tilelang.__path__, prefix="tilelang."):
    if "cudnn" in m.name.lower(): matched.append(m.name)
print("tilelang modules with 'cudnn':", matched or "NONE")

print("\nDONE")
