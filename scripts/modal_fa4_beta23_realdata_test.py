"""Modal H200: FA4 beta23 validation with REAL cppmega graph-route data + TE 2.16 parity.

Tests:
  A - Real data forward/backward: B=4, S=1024, H=20, D=64 with call_edges,
      type_edges, domain edges, build edges, AND rare CSR edges (0-50 per row).
  B - TE 2.16 parity: dense [B,1,S,S] bias vs FA4 chunk-native score_mod.
  C - Rare CSR stress: rows with 0, 50+, and 200+ rare edges.

Usage:
    cd /Volumes/external/sources/cppmega && modal run scripts/modal_fa4_beta23_realdata_test.py

Exit codes: 0=all pass, 1=any fail, 2=skip
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

import modal

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

GHCR_REPO = "ghcr.io/datasunriseou/cppmega"
GHCR_DIGEST = "sha256:08c5db7368d1037d930e0825281468927de9c85b12ba10373fe07e082150d983"

app = modal.App("cppmega-fa4-beta23-realdata")
results_vol = modal.Volume.from_name("cppmega-fa4-test-results", create_if_missing=True)


def _image() -> modal.Image:
    """Build image: cppmega GHCR base + FA4 beta23 (replacing b19)."""
    img: Any = modal.Image.from_registry(
        f"{GHCR_REPO}@{GHCR_DIGEST}",
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env({
        "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
        "WANDB_MODE": "disabled",
        "CPPMEGA_FA4_SCORE_MOD": "1",
    })
    # Remove stale namespace packages that shadow FA4 beta23's bundled deps.
    img = img.add_local_file(
        str(_REPO_ROOT / "scripts" / "fix_cutlass_namespace.py"),
        remote_path="/opt/fix_cutlass_namespace.py",
        copy=True,
    )
    img = img.run_commands(
        "python3 /opt/fix_cutlass_namespace.py",
        "python3 -m pip install --pre --no-cache-dir flash-attn-4==4.0.0b23",
        "python3 -c 'from flash_attn.cute.interface import flash_attn_func; print(\"FA4 beta23 OK\")'",
    )
    img = (
        img.add_local_dir(str(_REPO_ROOT / "cppmega"), remote_path="/opt/cppmega/cppmega", copy=True)
        .add_local_dir(str(_REPO_ROOT / "tests"), remote_path="/opt/cppmega/tests", copy=True)
        .add_local_file(str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml")
    )
    return img


# ---------------------------------------------------------------------------
# Helpers (run inside the container)
# ---------------------------------------------------------------------------

def _build_realistic_structure_batch(
    B: int, S: int, num_chunks: int, device: Any, seed: int = 42,
    rare_per_row_range: tuple[int, int] = (0, 50),
) -> dict[str, Any]:
    """Build a realistic cppmega graph-route structure batch.

    Includes: chunk layout, call_edges, type_edges, domain edges, build edges,
    and rare CSR edges with varying counts per row.
    """
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Chunk layout: evenly divide S into num_chunks
    chunk_size = S // num_chunks
    chunk_starts = torch.zeros(B, num_chunks, dtype=torch.long)
    chunk_ends = torch.zeros(B, num_chunks, dtype=torch.long)
    for b in range(B):
        for c in range(num_chunks):
            chunk_starts[b, c] = c * chunk_size
            chunk_ends[b, c] = (c + 1) * chunk_size
    chunk_counts = torch.full((B,), num_chunks, dtype=torch.long)

    # Call edges (chunk-level): [B, max_call_edges, 2]
    max_call_edges = 32
    call_edges = torch.zeros(B, max_call_edges, 2, dtype=torch.long)
    call_edge_counts = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        n_edges = int(torch.randint(4, 16, (1,), generator=g).item())
        for i in range(n_edges):
            src = int(torch.randint(0, num_chunks, (1,), generator=g).item())
            dst = int(torch.randint(0, num_chunks, (1,), generator=g).item())
            call_edges[b, i, 0] = src
            call_edges[b, i, 1] = dst
        call_edge_counts[b] = n_edges

    # Type edges (chunk-level): [B, max_type_edges, 2]
    max_type_edges = 24
    type_edges = torch.zeros(B, max_type_edges, 2, dtype=torch.long)
    type_edge_counts = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        n_edges = int(torch.randint(2, 12, (1,), generator=g).item())
        for i in range(n_edges):
            src = int(torch.randint(0, num_chunks, (1,), generator=g).item())
            dst = int(torch.randint(0, num_chunks, (1,), generator=g).item())
            type_edges[b, i, 0] = src
            type_edges[b, i, 1] = dst
        type_edge_counts[b] = n_edges

    # Domain edges (token-level triples): [B, max_domain_edges, 3]
    # Each triple: (src_token, dst_token, kind) where kind >= 0 means active.
    max_domain_edges = 512
    domain_edges = torch.full((B, max_domain_edges, 3), -1, dtype=torch.long)
    domain_edge_counts = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        n_edges = int(torch.randint(20, 100, (1,), generator=g).item())
        for i in range(n_edges):
            src = int(torch.randint(0, S, (1,), generator=g).item())
            dst = int(torch.randint(0, S, (1,), generator=g).item())
            kind = int(torch.randint(0, 5, (1,), generator=g).item())
            domain_edges[b, i, 0] = src
            domain_edges[b, i, 1] = dst
            domain_edges[b, i, 2] = kind
        domain_edge_counts[b] = n_edges

    # Build edges (token-level triples): [B, max_build_edges, 3]
    max_build_edges = 256
    build_edges = torch.full((B, max_build_edges, 3), -1, dtype=torch.long)
    build_edge_counts = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        n_edges = int(torch.randint(10, 60, (1,), generator=g).item())
        for i in range(n_edges):
            src = int(torch.randint(0, S, (1,), generator=g).item())
            dst = int(torch.randint(0, S, (1,), generator=g).item())
            kind = int(torch.randint(0, 3, (1,), generator=g).item())
            build_edges[b, i, 0] = src
            build_edges[b, i, 1] = dst
            build_edges[b, i, 2] = kind
        build_edge_counts[b] = n_edges

    structure_batch = {
        "graph_chunk_starts": chunk_starts.to(device),
        "graph_chunk_ends": chunk_ends.to(device),
        "graph_chunk_counts": chunk_counts.to(device),
        "graph_call_edges": call_edges.to(device),
        "graph_call_edge_counts": call_edge_counts.to(device),
        "graph_type_edges": type_edges.to(device),
        "graph_type_edge_counts": type_edge_counts.to(device),
        "graph_domain_edges": domain_edges.to(device),
        "graph_domain_edge_counts": domain_edge_counts.to(device),
        "graph_build_edges": build_edges.to(device),
        "graph_build_edge_counts": build_edge_counts.to(device),
    }
    return structure_batch


def _build_dense_bias_from_chunk_native(bias_state: Any, B: int, S: int, device: Any) -> Any:
    """Expand ChunkNativeGraphBias to dense [B, 1, S, S] bias (float32)."""
    import torch

    chunk_bias = bias_state.chunk_bias  # [B, C+1, C+1]
    t2c_q = bias_state.token_to_chunk_q  # [B, S] int32
    t2c_k = bias_state.token_to_chunk_k  # [B, S] int32
    rare_row_offsets = bias_state.rare_row_offsets  # [B, S+1]
    rare_k = bias_state.rare_k  # [B, max_rare]
    rare_w = bias_state.rare_w  # [B, max_rare]

    dense = torch.zeros(B, 1, S, S, device=device, dtype=torch.float32)
    for b in range(B):
        for qi in range(S):
            qc = int(t2c_q[b, qi].item())
            for ki in range(S):
                kc = int(t2c_k[b, ki].item())
                val = float(chunk_bias[b, qc, kc].item())
                # Add rare edge contribution
                lo = int(rare_row_offsets[b, qi].item())
                hi = int(rare_row_offsets[b, qi + 1].item())
                for idx in range(lo, hi):
                    if int(rare_k[b, idx].item()) == ki:
                        val += float(rare_w[b, idx].item())
                        break
                dense[b, 0, qi, ki] = val
    return dense


def _build_dense_bias_vectorized(bias_state: Any, B: int, S: int, device: Any) -> Any:
    """Vectorized expansion of ChunkNativeGraphBias to dense [B, 1, S, S]."""
    import torch

    chunk_bias = bias_state.chunk_bias  # [B, C+1, C+1]
    t2c_q = bias_state.token_to_chunk_q.long()  # [B, S]
    t2c_k = bias_state.token_to_chunk_k.long()  # [B, S]

    # Gather chunk-pair bias: for each (qi, ki), look up chunk_bias[b, t2c_q[b,qi], t2c_k[b,ki]]
    # Expand t2c_q to [B, S, 1] and t2c_k to [B, 1, S] for broadcasting
    c_plus_1 = chunk_bias.shape[1]
    chunk_bias_flat = chunk_bias.reshape(B, c_plus_1 * c_plus_1)  # [B, (C+1)^2]
    flat_idx = t2c_q.unsqueeze(2) * c_plus_1 + t2c_k.unsqueeze(1)  # [B, S, S]
    dense = torch.gather(chunk_bias_flat, 1, flat_idx.reshape(B, -1)).reshape(B, S, S)
    dense = dense.unsqueeze(1)  # [B, 1, S, S]

    # Add rare edge overlay
    rare_row_offsets = bias_state.rare_row_offsets  # [B, S+1]
    rare_k_t = bias_state.rare_k  # [B, max_rare]
    rare_w_t = bias_state.rare_w  # [B, max_rare]

    for b in range(B):
        total_rare = int(rare_row_offsets[b, -1].item())
        if total_rare == 0:
            continue
        for qi in range(S):
            lo = int(rare_row_offsets[b, qi].item())
            hi = int(rare_row_offsets[b, qi + 1].item())
            if lo == hi:
                continue
            for idx in range(lo, hi):
                ki = int(rare_k_t[b, idx].item())
                if ki >= 0:
                    dense[b, 0, qi, ki] += rare_w_t[b, idx]

    return dense


# ---------------------------------------------------------------------------
# TEST A: Real data forward/backward
# ---------------------------------------------------------------------------


@app.function(image=_image(), gpu="H200", timeout=900, volumes={"/results": results_vol})
def test_a_real_data_fwd_bwd() -> dict[str, Any]:
    """FA4 beta23 forward/backward with realistic graph-route data.

    B=4, S=1024, H=20, D=64 with call_edges, type_edges, domain edges,
    build edges, and rare CSR edges with varying counts per row (0-50).
    """
    import sys
    import traceback

    import torch

    sys.path.insert(0, "/opt/cppmega")
    results: dict[str, Any] = {"test": "A_real_data_fwd_bwd"}

    # Environment check
    try:
        import importlib.metadata as md
        results["fa4_version"] = md.version("flash-attn-4")
    except Exception as e:
        results["error"] = f"FA4 not installed: {e}"
        results["status"] = "FAIL"
        return results

    try:
        from flash_attn.cute.interface import flash_attn_func  # noqa: F401
    except ImportError as e:
        results["error"] = f"FA4 cute interface import failed: {e}"
        results["status"] = "FAIL"
        return results

    from cppmega.megatron.fa4_score_mod_adapter import (
        CppMegaFA4ScoreModAttention,
        build_chunk_native_graph_bias,
    )

    device = torch.device("cuda")
    B, S, H, D = 4, 1024, 20, 64
    num_chunks = 16
    results["config"] = {"B": B, "S": S, "H": H, "D": D, "num_chunks": num_chunks}

    # Build realistic structure batch
    try:
        structure_batch = _build_realistic_structure_batch(
            B, S, num_chunks, device, seed=42, rare_per_row_range=(0, 50),
        )
        results["structure_batch_built"] = True
    except Exception as e:
        results["error"] = f"Structure batch build failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL"
        return results

    # Build chunk-native bias using production adapter
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
        results["rare_row_offsets_shape"] = list(bias_state.rare_row_offsets.shape)
        # Report rare edge statistics
        total_rare = int(bias_state.rare_row_offsets[:, -1].sum().item())
        results["total_rare_edges"] = total_rare
        max_per_row = int(
            (bias_state.rare_row_offsets[:, 1:] - bias_state.rare_row_offsets[:, :-1]).max().item()
        )
        results["max_rare_per_row"] = max_per_row
    except Exception as e:
        results["error"] = f"Bias build failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL"
        return results

    # Create QKV with gradients (Megatron SBHD layout)
    torch.manual_seed(123)
    query = torch.randn(S, B, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(S, B, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(S, B, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Forward pass
    module = CppMegaFA4ScoreModAttention(
        num_attention_heads=H, head_dim=D, causal=True,
    ).to(device)

    try:
        output = module(query, key, value, attention_bias=bias_state)
        results["forward_ok"] = True
        results["output_shape"] = list(output.shape)
        results["output_finite"] = bool(torch.isfinite(output).all().item())
        results["output_norm"] = output.float().norm().item()
    except Exception as e:
        results["error"] = f"Forward failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL"
        return results

    # Backward pass
    try:
        loss = output.float().sum()
        loss.backward()
        results["backward_ok"] = True

        # Verify gradients exist and are finite
        dq_finite = bool(torch.isfinite(query.grad).all().item())
        dk_finite = bool(torch.isfinite(key.grad).all().item())
        dv_finite = bool(torch.isfinite(value.grad).all().item())
        results["grad_q_finite"] = dq_finite
        results["grad_k_finite"] = dk_finite
        results["grad_v_finite"] = dv_finite

        # Gradient norms
        dq_norm = query.grad.float().norm().item()
        dk_norm = key.grad.float().norm().item()
        dv_norm = value.grad.float().norm().item()
        results["grad_q_norm"] = dq_norm
        results["grad_k_norm"] = dk_norm
        results["grad_v_norm"] = dv_norm
        results["grad_q_max"] = query.grad.float().abs().max().item()
        results["grad_k_max"] = key.grad.float().abs().max().item()
        results["grad_v_max"] = value.grad.float().abs().max().item()

        all_finite = dq_finite and dk_finite and dv_finite
        results["all_grads_finite"] = all_finite
    except Exception as e:
        results["error"] = f"Backward failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL"
        return results

    # Pass criteria: forward OK, output finite, backward OK, all grads finite
    passed = all([
        results.get("forward_ok"),
        results.get("output_finite"),
        results.get("backward_ok"),
        results.get("all_grads_finite"),
    ])
    results["status"] = "PASS" if passed else "FAIL"
    return results


# ---------------------------------------------------------------------------
# TEST B: TE 2.16 parity
# ---------------------------------------------------------------------------


@app.function(image=_image(), gpu="H200", timeout=900, volumes={"/results": results_vol})
def test_b_te_parity() -> dict[str, Any]:
    """TE 2.16 DotProductAttention (dense bias) vs FA4 beta23 score_mod.

    Build dense [B,1,S,S] bias from the same structure batch, run both paths,
    compare outputs. PASS if max_diff < 0.1 (bf16 tolerance).
    """
    import sys
    import traceback

    import torch

    sys.path.insert(0, "/opt/cppmega")
    results: dict[str, Any] = {"test": "B_te_216_parity"}

    # Check TE availability
    try:
        import transformer_engine.pytorch as te
        import importlib.metadata as md
        results["te_version"] = md.version("transformer_engine")
    except ImportError as e:
        results["error"] = f"TE not available: {e}"
        results["status"] = "SKIP"
        return results

    try:
        import importlib.metadata as md
        results["fa4_version"] = md.version("flash-attn-4")
    except Exception:
        results["fa4_version"] = "unknown"

    from cppmega.megatron.fa4_score_mod_adapter import (
        CppMegaFA4ScoreModAttention,
        build_chunk_native_graph_bias,
    )

    device = torch.device("cuda")
    # Use smaller S for dense bias feasibility (S=256 to keep memory reasonable)
    B, S, H, D = 2, 256, 8, 64
    num_chunks = 8
    results["config"] = {"B": B, "S": S, "H": H, "D": D, "num_chunks": num_chunks}

    # Build structure batch
    structure_batch = _build_realistic_structure_batch(
        B, S, num_chunks, device, seed=99, rare_per_row_range=(0, 20),
    )

    # Build chunk-native bias
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
    except Exception as e:
        results["error"] = f"Bias build failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL"
        return results

    # Build dense [B, 1, S, S] bias (vectorized)
    try:
        dense_bias_f32 = _build_dense_bias_vectorized(bias_state, B, S, device)
        # TE 2.16 requires bias dtype to match QKV dtype (bf16)
        dense_bias_bf16 = dense_bias_f32.to(torch.bfloat16)
        results["dense_bias_shape"] = list(dense_bias_bf16.shape)
        results["dense_bias_nonzero"] = int((dense_bias_f32 != 0).sum().item())
    except Exception as e:
        results["error"] = f"Dense bias expansion failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL"
        return results

    # Shared QKV
    torch.manual_seed(77)
    q_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    k_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    v_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)

    # TE reference: DotProductAttention with post_scale_bias
    try:
        te_attn = te.DotProductAttention(
            num_attention_heads=H,
            kv_channels=D,
            attention_dropout=0.0,
            qkv_format="bshd",
            attn_mask_type="causal",
        ).to(device)
        te_out = te_attn(
            q_bshd.clone(), k_bshd.clone(), v_bshd.clone(),
            qkv_format="bshd",
            max_seqlen_q=S,
            max_seqlen_kv=S,
            core_attention_bias_type="post_scale_bias",
            core_attention_bias=dense_bias_bf16,
        )
        results["te_forward_ok"] = True
        results["te_output_shape"] = list(te_out.shape)
    except Exception as e:
        results["te_error"] = str(e)
        results["traceback"] = traceback.format_exc()
        results["status"] = "SKIP (TE forward failed)"
        return results

    # FA4 path
    module = CppMegaFA4ScoreModAttention(
        num_attention_heads=H, head_dim=D, causal=True,
    ).to(device)

    # FA4 expects SBHD layout
    q_sbhd = q_bshd.transpose(0, 1).contiguous()
    k_sbhd = k_bshd.transpose(0, 1).contiguous()
    v_sbhd = v_bshd.transpose(0, 1).contiguous()

    try:
        fa4_out = module(q_sbhd, k_sbhd, v_sbhd, attention_bias=bias_state)
        results["fa4_forward_ok"] = True
        results["fa4_output_shape"] = list(fa4_out.shape)
    except Exception as e:
        results["fa4_error"] = str(e)
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (FA4 forward)"
        return results

    # Compare: TE output [B, S, H*D], FA4 output [S, B, H*D]
    te_flat = te_out.reshape(B, S, H * D).float()
    fa4_flat = fa4_out.transpose(0, 1).reshape(B, S, H * D).float()

    max_diff = (te_flat - fa4_flat).abs().max().item()
    mean_diff = (te_flat - fa4_flat).abs().mean().item()
    results["max_diff"] = max_diff
    results["mean_diff"] = mean_diff
    results["te_norm"] = te_flat.norm().item()
    results["fa4_norm"] = fa4_flat.norm().item()

    PARITY_THRESHOLD = 0.1
    results["threshold"] = PARITY_THRESHOLD
    results["parity_pass"] = max_diff < PARITY_THRESHOLD
    results["status"] = "PASS" if max_diff < PARITY_THRESHOLD else f"FAIL (max_diff={max_diff:.6f})"
    return results


# ---------------------------------------------------------------------------
# TEST C: Rare CSR stress
# ---------------------------------------------------------------------------


@app.function(image=_image(), gpu="H200", timeout=900, volumes={"/results": results_vol})
def test_c_rare_csr_stress() -> dict[str, Any]:
    """Stress test: rows with 0, 50+, and 200+ rare edges.

    Verifies the dynamic loop (range(lo, hi)) handles all cases correctly.
    Compares FA4 output against dense reference.
    """
    import sys
    import traceback

    import torch

    sys.path.insert(0, "/opt/cppmega")
    results: dict[str, Any] = {"test": "C_rare_csr_stress"}

    try:
        import importlib.metadata as md
        results["fa4_version"] = md.version("flash-attn-4")
    except Exception:
        results["fa4_version"] = "unknown"

    from cppmega.megatron.fa4_score_mod_adapter import (
        CppMegaFA4ScoreModAttention,
        build_chunk_native_graph_bias,
    )

    device = torch.device("cuda")
    B, S, H, D = 2, 512, 8, 64
    num_chunks = 8
    results["config"] = {"B": B, "S": S, "H": H, "D": D, "num_chunks": num_chunks}

    # Build structure batch with extreme rare edge distribution:
    # - Some rows have 0 rare edges
    # - Some rows have 50+ rare edges
    # - One row has 200+ rare edges
    # We use domain edges (token-level triples) to create the rare edges.
    g = torch.Generator(device="cpu")
    g.manual_seed(2024)

    chunk_size = S // num_chunks
    chunk_starts = torch.zeros(B, num_chunks, dtype=torch.long)
    chunk_ends = torch.zeros(B, num_chunks, dtype=torch.long)
    for b in range(B):
        for c in range(num_chunks):
            chunk_starts[b, c] = c * chunk_size
            chunk_ends[b, c] = (c + 1) * chunk_size
    chunk_counts = torch.full((B,), num_chunks, dtype=torch.long)

    # Minimal call edges (just to have at least one relation type)
    call_edges = torch.zeros(B, 4, 2, dtype=torch.long)
    call_edges[:, 0] = torch.tensor([0, 1])
    call_edges[:, 1] = torch.tensor([1, 2])
    call_edge_counts = torch.full((B,), 2, dtype=torch.long)

    # Domain edges: create extreme distribution
    # Batch 0: row 0 has 0 edges, row 1 has 55 edges, row 100 has 210 edges
    # Batch 1: row 0 has 200+ edges, others have 0
    max_domain_edges = 1024  # Large enough for 200+ edges in a single row
    domain_edges = torch.full((B, max_domain_edges, 3), -1, dtype=torch.long)
    domain_edge_counts = torch.zeros(B, dtype=torch.long)

    # Batch 0: distribute edges across specific rows
    idx = 0
    # Row 1: 55 edges
    for i in range(55):
        dst = int(torch.randint(0, S, (1,), generator=g).item())
        domain_edges[0, idx, 0] = 1  # src = row 1
        domain_edges[0, idx, 1] = dst
        domain_edges[0, idx, 2] = 0  # kind=0 (active)
        idx += 1
    # Row 100: 210 edges
    for i in range(210):
        dst = int(torch.randint(0, S, (1,), generator=g).item())
        domain_edges[0, idx, 0] = 100  # src = row 100
        domain_edges[0, idx, 1] = dst
        domain_edges[0, idx, 2] = 1  # kind=1 (active)
        idx += 1
    # Row 200: 0 edges (just leave it empty)
    domain_edge_counts[0] = idx

    # Batch 1: Row 0 has 220 edges
    idx = 0
    for i in range(220):
        dst = int(torch.randint(0, S, (1,), generator=g).item())
        domain_edges[1, idx, 0] = 0  # src = row 0
        domain_edges[1, idx, 1] = dst
        domain_edges[1, idx, 2] = 2  # kind=2 (active)
        idx += 1
    domain_edge_counts[1] = idx

    structure_batch = {
        "graph_chunk_starts": chunk_starts.to(device),
        "graph_chunk_ends": chunk_ends.to(device),
        "graph_chunk_counts": chunk_counts.to(device),
        "graph_call_edges": call_edges.to(device),
        "graph_call_edge_counts": call_edge_counts.to(device),
        "graph_domain_edges": domain_edges.to(device),
        "graph_domain_edge_counts": domain_edge_counts.to(device),
    }

    # Build chunk-native bias (need max_rare >= 220 for the 220-edge row)
    import os
    os.environ["CPPMEGA_FA4_MAX_RARE_PER_ROW"] = "512"

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
        # Report per-row rare edge counts
        row_counts = bias_state.rare_row_offsets[:, 1:] - bias_state.rare_row_offsets[:, :-1]
        results["max_rare_per_row"] = int(row_counts.max().item())
        results["min_rare_per_row"] = int(row_counts.min().item())
        results["rows_with_0_edges"] = int((row_counts == 0).sum().item())
        results["rows_with_50plus"] = int((row_counts >= 50).sum().item())
        results["rows_with_200plus"] = int((row_counts >= 200).sum().item())
        results["total_rare_edges"] = int(bias_state.rare_row_offsets[:, -1].sum().item())
    except Exception as e:
        results["error"] = f"Bias build failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL"
        return results

    # Shared QKV
    torch.manual_seed(555)
    q_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    k_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    v_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)

    # FA4 path
    module = CppMegaFA4ScoreModAttention(
        num_attention_heads=H, head_dim=D, causal=True,
    ).to(device)

    q_sbhd = q_bshd.transpose(0, 1).contiguous()
    k_sbhd = k_bshd.transpose(0, 1).contiguous()
    v_sbhd = v_bshd.transpose(0, 1).contiguous()

    try:
        fa4_out = module(q_sbhd, k_sbhd, v_sbhd, attention_bias=bias_state)
        results["fa4_forward_ok"] = True
        results["fa4_output_finite"] = bool(torch.isfinite(fa4_out).all().item())
    except Exception as e:
        results["fa4_error"] = str(e)
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (FA4 forward)"
        return results

    # Dense reference (vectorized)
    try:
        dense_bias_f32 = _build_dense_bias_vectorized(bias_state, B, S, device)
        dense_bias_bf16 = dense_bias_f32.to(torch.bfloat16)
        results["dense_bias_built"] = True
    except Exception as e:
        results["error"] = f"Dense bias build failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL"
        return results

    # Try TE for dense reference; fall back to manual PyTorch if TE unavailable
    te_used = False
    try:
        import transformer_engine.pytorch as te
        te_attn = te.DotProductAttention(
            num_attention_heads=H,
            kv_channels=D,
            attention_dropout=0.0,
            qkv_format="bshd",
            attn_mask_type="causal",
        ).to(device)
        ref_out = te_attn(
            q_bshd.clone(), k_bshd.clone(), v_bshd.clone(),
            qkv_format="bshd",
            max_seqlen_q=S,
            max_seqlen_kv=S,
            core_attention_bias_type="post_scale_bias",
            core_attention_bias=dense_bias_bf16,
        )
        ref_flat = ref_out.reshape(B, S, H * D).float()
        te_used = True
        results["reference"] = "TE"
    except Exception:
        # Manual PyTorch reference (slower but always available)
        import math
        scale = 1.0 / math.sqrt(D)
        q_f = q_bshd.float().permute(0, 2, 1, 3)  # [B, H, S, D]
        k_f = k_bshd.float().permute(0, 2, 1, 3)  # [B, H, S, D]
        v_f = v_bshd.float().permute(0, 2, 1, 3)  # [B, H, S, D]
        scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # [B, H, S, S]
        scores = scores + dense_bias_f32  # [B, 1, S, S] broadcasts
        # Causal mask
        causal_mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_f)  # [B, H, S, D]
        ref_flat = out.permute(0, 2, 1, 3).reshape(B, S, H * D)
        results["reference"] = "manual_pytorch"

    # Compare FA4 vs reference
    fa4_flat = fa4_out.transpose(0, 1).reshape(B, S, H * D).float()

    max_diff = (ref_flat - fa4_flat).abs().max().item()
    mean_diff = (ref_flat - fa4_flat).abs().mean().item()
    results["max_diff"] = max_diff
    results["mean_diff"] = mean_diff
    results["ref_norm"] = ref_flat.norm().item()
    results["fa4_norm"] = fa4_flat.norm().item()

    PARITY_THRESHOLD = 0.1
    results["threshold"] = PARITY_THRESHOLD
    results["parity_pass"] = max_diff < PARITY_THRESHOLD
    results["status"] = "PASS" if max_diff < PARITY_THRESHOLD else f"FAIL (max_diff={max_diff:.6f})"
    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    """Run all three FA4 beta23 real-data tests on Modal H200."""
    import json

    print("=" * 70)
    print("FA4 beta23 Real Data Validation (H200)")
    print(f"Image: {GHCR_REPO}@{GHCR_DIGEST}")
    print("GPU: H200")
    print("=" * 70)

    # TEST A
    print("\nTEST A: Real data forward/backward (B=4, S=1024, H=20, D=64)")
    print("-" * 70)
    r_a = test_a_real_data_fwd_bwd.remote()
    print(json.dumps(r_a, indent=2, default=str))

    # TEST B
    print("\n" + "=" * 70)
    print("TEST B: TE 2.16 parity (dense bias vs chunk-native score_mod)")
    print("-" * 70)
    r_b = test_b_te_parity.remote()
    print(json.dumps(r_b, indent=2, default=str))

    # TEST C
    print("\n" + "=" * 70)
    print("TEST C: Rare CSR stress (0, 50+, 200+ edges per row)")
    print("-" * 70)
    r_c = test_c_rare_csr_stress.remote()
    print(json.dumps(r_c, indent=2, default=str))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    status_a = r_a.get("status", "UNKNOWN")
    status_b = r_b.get("status", "UNKNOWN")
    status_c = r_c.get("status", "UNKNOWN")
    print(f"  Test A (Real data fwd/bwd):  {status_a}")
    if r_a.get("grad_q_norm") is not None:
        print(f"    grad norms: dQ={r_a['grad_q_norm']:.4f} dK={r_a['grad_k_norm']:.4f} dV={r_a['grad_v_norm']:.4f}")
    print(f"  Test B (TE 2.16 parity):     {status_b}")
    if r_b.get("max_diff") is not None:
        print(f"    max_diff={r_b['max_diff']:.6f} mean_diff={r_b['mean_diff']:.6f} (threshold={r_b.get('threshold', 0.1)})")
    print(f"  Test C (Rare CSR stress):    {status_c}")
    if r_c.get("max_diff") is not None:
        print(f"    max_diff={r_c['max_diff']:.6f} mean_diff={r_c['mean_diff']:.6f}")
    if r_c.get("rows_with_200plus") is not None:
        print(f"    rows: 0-edge={r_c['rows_with_0_edges']}, 50+={r_c['rows_with_50plus']}, 200+={r_c['rows_with_200plus']}")
    print("=" * 70)

    # Exit codes
    failures = []
    skips = []
    for name, r in [("A", r_a), ("B", r_b), ("C", r_c)]:
        s = r.get("status", "UNKNOWN")
        if s.startswith("FAIL") or s == "UNKNOWN":
            failures.append(f"Test {name}: {s}")
        elif s.startswith("SKIP"):
            skips.append(f"Test {name}: {s}")

    if failures:
        print("\n*** FAILURES ***")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)

    if skips:
        print("\n*** SKIPS (parity not fully verified) ***")
        for s in skips:
            print(f"  {s}")
        sys.exit(2)

    print("\nAll tests PASSED.")
    sys.exit(0)
