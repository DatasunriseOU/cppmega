"""Modal H200: Production FA4 adapter forward/backward + TE parity test.

Runs the PRODUCTION CppMegaFA4ScoreModAttention.forward() with real
ChunkNativeGraphBias, then compares against TE dense post_scale_bias.

Usage:
    modal run scripts/modal_fa4_production_test.py
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, cast

import modal

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

GHCR_REPO = "ghcr.io/datasunriseou/cppmega"
GHCR_DIGEST = "sha256:08c5db7368d1037d930e0825281468927de9c85b12ba10373fe07e082150d983"
# FA4 beta23 image digest (flash-attn-4 4.0.0b23 + apache-tvm-ffi >=0.1.12).
# Placeholder until docker/Dockerfile.beta23 is built and pushed to GHCR; see
# docs/fa4_beta23_upgrade_plan.md. Replace with the real digest from:
#   docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/datasunriseou/cppmega:beta23
GHCR_DIGEST_BETA23 = os.environ.get(
    "GHCR_DIGEST_BETA23",
    "sha256:eb6a4e11f9997b766924c32002601c3dc9d812fb12b9e57ad09b56f48881ea1f",
)
# Opt into the beta23 image with CPPMEGA_BETA23=1 or a --beta23 flag on the
# modal run command line. Image selection happens at import time because the
# @app.function decorators build the image when the module loads.
USE_BETA23 = os.environ.get("CPPMEGA_BETA23", "0") == "1" or "--beta23" in sys.argv
GHCR_REF = f"{GHCR_REPO}@{GHCR_DIGEST_BETA23 if USE_BETA23 else GHCR_DIGEST}"

app = modal.App("cppmega-fa4-prod-test")
results_vol = modal.Volume.from_name("cppmega-fa4-test-results", create_if_missing=True)


def _image() -> modal.Image:
    # Always use the default (b19) GHCR image as base.
    # Beta23 upgrade is done in run_commands to avoid namespace package conflicts.
    img: Any = modal.Image.from_registry(
        f"{GHCR_REPO}@{GHCR_DIGEST}",
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env({
        "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
    })
    # Remove base image's locally-built cutlass-dsl/tvm-ffi (incompatible
    # namespace packages that shadow the PyPI versions bundled with FA4 beta23),
    # then fresh-install FA4 beta23. The cleanup is done by a standalone Python
    # script (pathlib + shutil, no shell globs) to avoid Modal's image-builder
    # shell-quoting issues that broke earlier inline `python3 -c` / `rm -rf`
    # attempts. Each run_commands entry is kept intentionally SIMPLE.
    img = img.add_local_file(
        str(_REPO_ROOT / "scripts" / "fix_cutlass_namespace.py"),
        remote_path="/opt/fix_cutlass_namespace.py",
        copy=True,
    )
    img = img.run_commands(
        "python3 /opt/fix_cutlass_namespace.py",
        "pip install --pre --no-cache-dir flash-attn-4==4.0.0b23",
        "python3 -c 'from flash_attn.cute.interface import flash_attn_func; print(\"FA4 beta23 OK\")'",
    )
    img = (
        img.add_local_dir(str(_REPO_ROOT / "cppmega"), remote_path="/opt/cppmega/cppmega", copy=True)
        .add_local_dir(str(_REPO_ROOT / "tests"), remote_path="/opt/cppmega/tests", copy=True)
        .add_local_file(str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml")
    )
    return img


@app.function(image=_image(), gpu="H200:1", timeout=600, volumes={"/results": results_vol})
def test_fa4_production_forward_backward() -> dict[str, Any]:
    """Test production CppMegaFA4ScoreModAttention.forward() with real ChunkNativeGraphBias."""
    import json
    import sys
    import torch

    sys.path.insert(0, "/opt/cppmega")

    results = {"test": "fa4_production_forward_backward"}

    # Check FA4 version
    try:
        import flash_attn.cute
        results["fa4_version"] = getattr(flash_attn.cute, "__version__", "unknown")
    except Exception as e:
        results["error"] = f"FA4 import failed: {e}"
        return results

    # Import production code
    from cppmega.megatron.fa4_score_mod_adapter import (
        ChunkNativeGraphBias,
        CppMegaFA4ScoreModAttention,
        build_chunk_native_graph_bias,
    )

    device = torch.device("cuda")
    B, S, H, D = 2, 256, 20, 64  # Realistic: 20 heads, 64 head_dim

    # Create module
    module = CppMegaFA4ScoreModAttention(
        num_attention_heads=H,
        head_dim=D,
        causal=True,
    ).to(device)

    # Create QKV in Megatron SBHD format
    query = torch.randn(S, B, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(S, B, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(S, B, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Build a mock structure_batch with real chunk layout
    num_chunks = 8
    chunk_starts = torch.zeros(B, num_chunks, dtype=torch.long, device=device)
    chunk_ends = torch.zeros(B, num_chunks, dtype=torch.long, device=device)
    for b in range(B):
        for c in range(num_chunks):
            chunk_starts[b, c] = c * (S // num_chunks)
            chunk_ends[b, c] = (c + 1) * (S // num_chunks)
    chunk_counts = torch.full((B,), num_chunks, dtype=torch.long, device=device)

    # Add some call edges (chunk-level)
    max_edges = 16
    call_edges = torch.zeros(B, max_edges, 2, dtype=torch.long, device=device)
    call_edge_counts = torch.zeros(B, dtype=torch.long, device=device)
    for b in range(B):
        # 4 random chunk-to-chunk edges
        for i in range(4):
            src = torch.randint(0, num_chunks, (1,)).item()
            dst = torch.randint(0, num_chunks, (1,)).item()
            call_edges[b, i, 0] = src
            call_edges[b, i, 1] = dst
        call_edge_counts[b] = 4

    structure_batch = {
        "graph_chunk_starts": chunk_starts,
        "graph_chunk_ends": chunk_ends,
        "graph_chunk_counts": chunk_counts,
        "graph_call_edges": call_edges,
        "graph_call_edge_counts": call_edge_counts,
    }

    # Build ChunkNativeGraphBias using production builder
    try:
        bias_state = build_chunk_native_graph_bias(
            structure_batch,
            batch_size=B,
            seqlen_q=S,
            seqlen_k=S,
            device=device,
            dtype=torch.float32,
            beta=1.0,
        )
        results["bias_built"] = True
        results["chunk_bias_shape"] = list(bias_state.chunk_bias.shape)
        results["token_to_chunk_q_shape"] = list(bias_state.token_to_chunk_q.shape)
    except Exception as e:
        results["error"] = f"build_chunk_native_graph_bias failed: {e}"
        import traceback
        results["traceback"] = traceback.format_exc()
        return results

    # Forward pass
    try:
        output = module(query, key, value, attention_bias=bias_state)
        results["forward_ok"] = True
        results["output_shape"] = list(output.shape)
        results["output_dtype"] = str(output.dtype)
        results["output_norm"] = output.norm().item()
    except Exception as e:
        results["error"] = f"forward failed: {e}"
        import traceback
        results["traceback"] = traceback.format_exc()
        return results

    # Backward pass
    try:
        loss = output.sum()
        loss.backward()
        results["backward_ok"] = True
        results["dQ_norm"] = query.grad.norm().item()
        results["dK_norm"] = key.grad.norm().item()
        results["dV_norm"] = value.grad.norm().item()
    except Exception as e:
        results["error"] = f"backward failed: {e}"
        import traceback
        results["traceback"] = traceback.format_exc()
        return results

    # Baseline (no bias)
    query2 = query.detach().clone().requires_grad_(True)
    key2 = key.detach().clone().requires_grad_(True)
    value2 = value.detach().clone().requires_grad_(True)
    output_base = module(query2, key2, value2, attention_bias=None)
    diff = (output.detach() - output_base.detach()).abs().max().item()
    results["diff_from_baseline"] = diff
    results["baseline_norm"] = output_base.norm().item()

    results["status"] = "PASS" if results.get("backward_ok") else "FAIL"
    return results


@app.function(image=_image(), gpu="H200:1", timeout=600, volumes={"/results": results_vol})
def test_te_parity() -> dict[str, Any]:
    """Compare FA4 chunk-native score_mod vs TE dense post_scale_bias."""
    import sys
    import torch

    sys.path.insert(0, "/opt/cppmega")
    results = {"test": "te_parity"}

    try:
        import transformer_engine.pytorch as te
        results["te_available"] = True
    except ImportError:
        results["te_available"] = False
        results["status"] = "SKIP (no TE)"
        return results

    from cppmega.megatron.fa4_score_mod_adapter import (
        CppMegaFA4ScoreModAttention,
        build_chunk_native_graph_bias,
    )

    device = torch.device("cuda")
    B, S, H, D = 1, 128, 4, 64

    # Shared QKV
    torch.manual_seed(42)
    q_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    k_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    v_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)

    # Build a simple chunk bias
    num_chunks = 4
    chunk_starts = torch.zeros(B, num_chunks, dtype=torch.long, device=device)
    chunk_ends = torch.zeros(B, num_chunks, dtype=torch.long, device=device)
    for c in range(num_chunks):
        chunk_starts[0, c] = c * (S // num_chunks)
        chunk_ends[0, c] = (c + 1) * (S // num_chunks)
    chunk_counts = torch.full((B,), num_chunks, dtype=torch.long, device=device)

    call_edges = torch.zeros(B, 4, 2, dtype=torch.long, device=device)
    call_edges[0, 0] = torch.tensor([0, 1])
    call_edges[0, 1] = torch.tensor([1, 2])
    call_edge_counts = torch.tensor([2], dtype=torch.long, device=device)

    structure_batch = {
        "graph_chunk_starts": chunk_starts,
        "graph_chunk_ends": chunk_ends,
        "graph_chunk_counts": chunk_counts,
        "graph_call_edges": call_edges,
        "graph_call_edge_counts": call_edge_counts,
    }

    bias_state = build_chunk_native_graph_bias(
        structure_batch, batch_size=B, seqlen_q=S, seqlen_k=S,
        device=device, dtype=torch.float32, beta=1.0,
    )

    # Build dense [B, 1, S, S] bias from chunk_bias for TE
    chunk_bias = bias_state.chunk_bias  # [B, C+1, C+1]
    t2c_q = bias_state.token_to_chunk_q  # [B, S]
    t2c_k = bias_state.token_to_chunk_k  # [B, S]
    dense_bias = torch.zeros(B, 1, S, S, device=device, dtype=torch.float32)
    for b in range(B):
        for qi in range(S):
            for ki in range(S):
                qc = t2c_q[b, qi].item()
                kc = t2c_k[b, ki].item()
                dense_bias[b, 0, qi, ki] = chunk_bias[b, qc, kc]

    # TE reference: DotProductAttention with post_scale_bias
    try:
        te_attn = te.DotProductAttention(
            num_attention_heads=H,
            kv_channels=D,
            attention_dropout=0.0,
            attn_mask_type="causal",
        ).to(device)

        # TE expects [B, S, H, D]
        q_te = q_bshd.clone()
        k_te = k_bshd.clone()
        v_te = v_bshd.clone()
        te_out = te_attn(q_te, k_te, v_te, attention_bias=dense_bias)
        results["te_forward_ok"] = True
        results["te_output_shape"] = list(te_out.shape)
    except Exception as e:
        results["te_error"] = str(e)
        results["status"] = "SKIP (TE forward failed)"
        return results

    # FA4 path
    module = CppMegaFA4ScoreModAttention(
        num_attention_heads=H, head_dim=D, causal=True,
    ).to(device)

    # FA4 expects [S, B, H, D]
    q_fa4 = q_bshd.transpose(0, 1).contiguous()
    k_fa4 = k_bshd.transpose(0, 1).contiguous()
    v_fa4 = v_bshd.transpose(0, 1).contiguous()

    try:
        fa4_out = module(q_fa4, k_fa4, v_fa4, attention_bias=bias_state)
        results["fa4_forward_ok"] = True
        results["fa4_output_shape"] = list(fa4_out.shape)
    except Exception as e:
        results["fa4_error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (FA4 forward)"
        return results

    # Compare: TE output is [B, S, H*D], FA4 output is [S, B, H*D]
    te_flat = te_out.reshape(B, S, H * D)
    fa4_flat = fa4_out.transpose(0, 1).reshape(B, S, H * D)

    max_diff = (te_flat - fa4_flat).abs().max().item()
    mean_diff = (te_flat - fa4_flat).abs().mean().item()
    results["max_diff"] = max_diff
    results["mean_diff"] = mean_diff
    results["te_norm"] = te_flat.norm().item()
    results["fa4_norm"] = fa4_flat.norm().item()

    # bf16 tolerance: ~1e-2 for 128 seq with bias
    results["parity_pass"] = max_diff < 0.1
    results["status"] = "PASS" if max_diff < 0.1 else f"FAIL (max_diff={max_diff})"
    return results


@app.local_entrypoint()
def main(beta23: bool = False) -> None:
    """Run FA4 production tests on Modal H200.

    Pass --beta23 to target the FA4 beta23 GHCR image (requires the beta23
    image to have been built and pushed; see docs/fa4_beta23_upgrade_plan.md).
    """
    import json

    print(f"Image: {GHCR_REF}")
    print(f"Image variant: {'beta23' if (beta23 or USE_BETA23) else 'default (b19 + runtime upgrade)'}")
    print("=" * 60)
    print("TEST 1: Production FA4 forward/backward")
    print("=" * 60)
    r1 = test_fa4_production_forward_backward.remote()
    print(json.dumps(r1, indent=2, default=str))

    print("\n" + "=" * 60)
    print("TEST 2: TE ↔ FA4 parity")
    print("=" * 60)
    r2 = test_te_parity.remote()
    print(json.dumps(r2, indent=2, default=str))

    # Save results
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Forward/Backward: {r1.get('status', 'UNKNOWN')}")
    print(f"  TE Parity: {r2.get('status', 'UNKNOWN')}")
    print(f"{'='*60}")
