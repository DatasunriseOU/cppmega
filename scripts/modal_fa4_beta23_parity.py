"""Modal H200: TE<->FA4 parity + Megatron training step on the beta23 image.

Validates the FA4 beta23 (flash-attn-4 4.0.0b23) chunk-native score_mod path
against the TE dense post_scale_bias reference, then runs a single Megatron
training step (forward + backward + optimizer.step) and checkpoint save/reload.

PREREQUISITES:
  - The beta23 image must be built and pushed to GHCR:
      docker build -f docker/Dockerfile.beta23 -t ghcr.io/datasunriseou/cppmega:beta23 .
      docker push ghcr.io/datasunriseou/cppmega:beta23
  - Set GHCR_DIGEST_BETA23 to the real digest:
      docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/datasunriseou/cppmega:beta23
  - TileLang ref must be bumped to a tvm-ffi>=0.1.12-compatible commit
    (see docs/fa4_beta23_upgrade_plan.md section 4 and 12.4).

Usage:
    # With the beta23 GHCR image (preferred):
    CPPMEGA_BETA23=1 modal run scripts/modal_fa4_beta23_parity.py

    # With runtime pip upgrade on the default b19 image (slower, for testing):
    modal run scripts/modal_fa4_beta23_parity.py
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

import modal

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

GHCR_REPO = "ghcr.io/datasunriseou/cppmega"
GHCR_DIGEST = "sha256:08c5db7368d1037d930e0825281468927de9c85b12ba10373fe07e082150d983"
# FA4 beta23 image digest. Replace PLACEHOLDER with the real digest after push:
#   docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/datasunriseou/cppmega:beta23
GHCR_DIGEST_BETA23 = os.environ.get(
    "GHCR_DIGEST_BETA23",
    "sha256:PLACEHOLDER_BETA23_DIGEST_NOT_YET_PUSHED",
)
USE_BETA23 = os.environ.get("CPPMEGA_BETA23", "0") == "1" or "--beta23" in sys.argv
GHCR_REF = f"{GHCR_REPO}@{GHCR_DIGEST_BETA23 if USE_BETA23 else GHCR_DIGEST}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:1")

app = modal.App("cppmega-fa4-beta23-parity")
results_vol = modal.Volume.from_name("cppmega-fa4-test-results", create_if_missing=True)


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env({
        "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
        "WANDB_MODE": "disabled",
        "CPPMEGA_FA4_SCORE_MOD": "1",
    })
    # The default image needs both matching local wheels; upgrading only
    # tvm-ffi recreates the TileLang C++ ABI mismatch.
    if not USE_BETA23:
        wheel_dir = _REPO_ROOT / "wheels"
        ffi_wheel = wheel_dir / (
            "apache_tvm_ffi-0.1.13.post1-cp313-cp313-linux_x86_64.whl"
        )
        tilelang_wheel = wheel_dir / "tilelang-0.1.9-cp38-abi3-linux_x86_64.whl"
        missing = [
            wheel for wheel in (ffi_wheel, tilelang_wheel) if not wheel.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing beta23 compatibility wheels: {missing}")
        img = (
            img.add_local_file(str(ffi_wheel), "/tmp/tvm_ffi.whl", copy=True)
            .add_local_file(str(tilelang_wheel), "/tmp/tilelang.whl", copy=True)
            .run_commands(
                "pip install --pre 'apache-tvm-ffi>=0.1.12,<0.2' "
                "'flash-attn-4[cu13]==4.0.0b23' "
                "--extra-index-url https://pypi.nvidia.com && "
                "pip install --force-reinstall --no-deps "
                "/tmp/tvm_ffi.whl /tmp/tilelang.whl"
            )
        )
    img = (
        img.add_local_dir(str(_REPO_ROOT / "cppmega"), remote_path="/opt/cppmega/cppmega", copy=True)
        .add_local_dir(str(_REPO_ROOT / "tests"), remote_path="/opt/cppmega/tests", copy=True)
        .add_local_file(str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml")
    )
    return img


# ---------------------------------------------------------------------------
# TEST 1: TE <-> FA4 parity (dense post_scale_bias vs chunk-native score_mod)
# ---------------------------------------------------------------------------


@app.function(image=_image(), gpu=GPU_SPEC, timeout=600, volumes={"/results": results_vol})
def test_te_fa4_parity() -> dict[str, Any]:
    """Compare TE DotProductAttention (dense bias) vs FA4 chunk-native score_mod.

    Reference: TE DotProductAttention with dense [B,1,S,S] post_scale_bias.
    Test: FA4 CppMegaFA4ScoreModAttention with ChunkNativeGraphBias.
    Pass criterion: max_diff < 0.1 (bf16 tolerance for S=128 with bias).
    """
    import json
    import sys
    import traceback

    import torch

    sys.path.insert(0, "/opt/cppmega")
    results: dict[str, Any] = {"test": "te_fa4_parity_beta23"}

    # --- Environment check ---
    try:
        import flash_attn.cute  # noqa: F401
        import importlib.metadata as md
        results["fa4_version"] = md.version("flash-attn-4")
    except Exception as e:
        results["error"] = f"FA4 import failed: {e}"
        results["status"] = "FAIL (FA4 import)"
        return results

    try:
        import importlib.metadata as md
        results["tvm_ffi_version"] = md.version("apache-tvm-ffi")
    except Exception:
        results["tvm_ffi_version"] = "unknown"

    try:
        import transformer_engine.pytorch as te
        results["te_available"] = True
    except ImportError as e:
        results["error"] = f"TE import failed: {e}"
        results["status"] = "SKIP (no TE)"
        return results

    from cppmega.megatron.fa4_score_mod_adapter import (
        CppMegaFA4ScoreModAttention,
        build_chunk_native_graph_bias,
    )

    device = torch.device("cuda")
    B, S, H, D = 1, 128, 4, 64
    results["config"] = {"B": B, "S": S, "H": H, "D": D}

    # Shared QKV (BSHD layout for TE)
    torch.manual_seed(42)
    q_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    k_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    v_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)

    # --- Build chunk structure ---
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
        "token_chunk_starts": chunk_starts,
        "token_chunk_ends": chunk_ends,
        "token_chunk_counts": chunk_counts,
        "graph_call_edges": call_edges,
        "graph_call_edge_counts": call_edge_counts,
    }

    bias_state = build_chunk_native_graph_bias(
        structure_batch, batch_size=B, seqlen_q=S, seqlen_k=S,
        device=device, dtype=torch.float32, beta=1.0,
    )

    # --- Expand to dense [B,1,S,S] bias for TE reference ---
    chunk_bias = bias_state.chunk_bias  # [B, C+1, C+1]
    t2c_q = bias_state.token_to_chunk_q  # [B, S]
    t2c_k = bias_state.token_to_chunk_k  # [B, S]
    # Keep float32 reference for mathematical comparison (FA4 vs manual PyTorch)
    dense_bias_f32 = torch.zeros(B, 1, S, S, device=device, dtype=torch.float32)
    for b in range(B):
        for qi in range(S):
            for ki in range(S):
                qc = t2c_q[b, qi].item()
                kc = t2c_k[b, ki].item()
                dense_bias_f32[b, 0, qi, ki] = chunk_bias[b, qc, kc]
    # TE 2.16 requires bias dtype to match QKV dtype (bf16) for post_scale_bias
    dense_bias = dense_bias_f32.to(torch.bfloat16)

    # --- TE reference: DotProductAttention with post_scale_bias ---
    try:
        te_attn = te.DotProductAttention(
            num_attention_heads=H,
            kv_channels=D,
            attention_dropout=0.0,
            qkv_format="bshd",
            attn_mask_type="causal",
        ).to(device)
        te_out = te_attn(q_bshd.clone(), k_bshd.clone(), v_bshd.clone(),
                         qkv_format="bshd",
                         max_seqlen_q=S,
                         max_seqlen_kv=S,
                         core_attention_bias_type="post_scale_bias",
                         core_attention_bias=dense_bias)
        results["te_forward_ok"] = True
        results["te_output_shape"] = list(te_out.shape)
    except Exception as e:
        results["te_error"] = str(e)
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (TE forward)"
        return results

    # --- FA4 test: CppMegaFA4ScoreModAttention with ChunkNativeGraphBias ---
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

    # --- Compare outputs ---
    # TE output: [B, S, H*D], FA4 output: [S, B, H*D]
    te_flat = te_out.reshape(B, S, H * D)
    fa4_flat = fa4_out.transpose(0, 1).reshape(B, S, H * D)

    max_diff = (te_flat - fa4_flat).abs().max().item()
    mean_diff = (te_flat - fa4_flat).abs().mean().item()
    results["max_diff"] = max_diff
    results["mean_diff"] = mean_diff
    results["te_norm"] = te_flat.norm().item()
    results["fa4_norm"] = fa4_flat.norm().item()

    # bf16 tolerance: < 0.1 for S=128 with additive bias
    PARITY_THRESHOLD = 0.1
    results["parity_pass"] = max_diff < PARITY_THRESHOLD
    results["threshold"] = PARITY_THRESHOLD
    results["status"] = "PASS" if max_diff < PARITY_THRESHOLD else f"FAIL (max_diff={max_diff:.6f})"
    return results


# ---------------------------------------------------------------------------
# TEST 2: Single Megatron training step (forward + backward + optimizer.step)
# ---------------------------------------------------------------------------


@app.function(image=_image(), gpu=GPU_SPEC, timeout=900, volumes={"/results": results_vol})
def test_megatron_training_step() -> dict[str, Any]:
    """Run a single Megatron training step with FA4 score_mod attention.

    Exercises: model forward, loss computation, backward, optimizer.step,
    checkpoint save, and checkpoint reload.
    """
    import json
    import sys
    import tempfile
    import traceback

    import torch

    sys.path.insert(0, "/opt/cppmega")
    results: dict[str, Any] = {"test": "megatron_training_step_beta23"}

    # --- Environment ---
    try:
        import importlib.metadata as md
        results["fa4_version"] = md.version("flash-attn-4")
        results["tvm_ffi_version"] = md.version("apache-tvm-ffi")
    except Exception:
        pass

    try:
        import transformer_engine.pytorch as te
        from cppmega.megatron.fa4_score_mod_adapter import (
            CppMegaFA4ScoreModAttention,
            build_chunk_native_graph_bias,
        )
    except ImportError as e:
        results["error"] = f"Import failed: {e}"
        results["status"] = "FAIL (import)"
        return results

    device = torch.device("cuda")
    torch.manual_seed(123)

    # --- Build a minimal transformer-like model using TE + FA4 attention ---
    B, S, H, D = 2, 256, 8, 64
    hidden_size = H * D  # 512
    results["config"] = {"B": B, "S": S, "H": H, "D": D, "hidden": hidden_size}

    class MiniTransformerBlock(torch.nn.Module):
        """Minimal block: LayerNorm -> FA4 Attention -> residual -> LayerNorm -> MLP -> residual."""

        def __init__(self):
            super().__init__()
            self.ln1 = torch.nn.LayerNorm(hidden_size).to(device=device, dtype=torch.bfloat16)
            self.attn = CppMegaFA4ScoreModAttention(
                num_attention_heads=H, head_dim=D, causal=True,
            ).to(device)
            self.ln2 = torch.nn.LayerNorm(hidden_size).to(device=device, dtype=torch.bfloat16)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size * 4, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_size * 4, hidden_size, bias=False),
            ).to(device=device, dtype=torch.bfloat16)

        def forward(self, x_sbhd: torch.Tensor, bias_state) -> torch.Tensor:
            # x_sbhd: [S, B, H*D]
            residual = x_sbhd
            h = self.ln1(x_sbhd)
            # Reshape for attention: [S, B, H, D]
            h = h.view(S, B, H, D)
            h = self.attn(h, h, h, attention_bias=bias_state)
            h = h.view(S, B, hidden_size)
            x_sbhd = residual + h
            residual = x_sbhd
            h = self.ln2(x_sbhd)
            h = self.mlp(h)
            return residual + h

    try:
        model = MiniTransformerBlock()
        results["model_built"] = True
        n_params = sum(p.numel() for p in model.parameters())
        results["n_params"] = n_params
    except Exception as e:
        results["error"] = f"Model build failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (model build)"
        return results

    # --- Build attention bias ---
    num_chunks = 8
    chunk_starts = torch.zeros(B, num_chunks, dtype=torch.long, device=device)
    chunk_ends = torch.zeros(B, num_chunks, dtype=torch.long, device=device)
    for b in range(B):
        for c in range(num_chunks):
            chunk_starts[b, c] = c * (S // num_chunks)
            chunk_ends[b, c] = (c + 1) * (S // num_chunks)
    chunk_counts = torch.full((B,), num_chunks, dtype=torch.long, device=device)

    max_edges = 16
    call_edges = torch.zeros(B, max_edges, 2, dtype=torch.long, device=device)
    call_edge_counts = torch.zeros(B, dtype=torch.long, device=device)
    for b in range(B):
        for i in range(4):
            src = torch.randint(0, num_chunks, (1,)).item()
            dst = torch.randint(0, num_chunks, (1,)).item()
            call_edges[b, i, 0] = src
            call_edges[b, i, 1] = dst
        call_edge_counts[b] = 4

    structure_batch = {
        "token_chunk_starts": chunk_starts,
        "token_chunk_ends": chunk_ends,
        "token_chunk_counts": chunk_counts,
        "graph_call_edges": call_edges,
        "graph_call_edge_counts": call_edge_counts,
    }

    try:
        bias_state = build_chunk_native_graph_bias(
            structure_batch, batch_size=B, seqlen_q=S, seqlen_k=S,
            device=device, dtype=torch.float32, beta=1.0,
        )
        results["bias_built"] = True
    except Exception as e:
        results["error"] = f"Bias build failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (bias build)"
        return results

    # --- Forward pass ---
    x = torch.randn(S, B, hidden_size, device=device, dtype=torch.bfloat16)
    target = torch.randn(S, B, hidden_size, device=device, dtype=torch.bfloat16)

    try:
        output = model(x, bias_state)
        loss = torch.nn.functional.mse_loss(output, target)
        results["forward_ok"] = True
        results["loss"] = loss.item()
        results["output_shape"] = list(output.shape)
    except Exception as e:
        results["error"] = f"Forward failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (forward)"
        return results

    # --- Backward pass ---
    try:
        loss.backward()
        grad_norms = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norms[name] = p.grad.norm().item()
        results["backward_ok"] = True
        results["grad_norms"] = grad_norms
        results["total_grad_norm"] = sum(v**2 for v in grad_norms.values()) ** 0.5
    except Exception as e:
        results["error"] = f"Backward failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (backward)"
        return results

    # --- Optimizer step ---
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        # Capture pre-step params for verification
        pre_step_params = {n: p.clone() for n, p in model.named_parameters()}
        optimizer.step()
        optimizer.zero_grad()

        # Verify params changed
        params_changed = 0
        for n, p in model.named_parameters():
            if not torch.equal(p, pre_step_params[n]):
                params_changed += 1
        results["optimizer_step_ok"] = True
        results["params_changed"] = params_changed
        results["total_params_tensors"] = len(pre_step_params)
    except Exception as e:
        results["error"] = f"Optimizer step failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (optimizer.step)"
        return results

    # --- Checkpoint save ---
    ckpt_dir = tempfile.mkdtemp(prefix="cppmega_beta23_ckpt_")
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }, ckpt_path)
        results["checkpoint_save_ok"] = True
        results["checkpoint_size_bytes"] = os.path.getsize(ckpt_path)
    except Exception as e:
        results["error"] = f"Checkpoint save failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (checkpoint save)"
        return results

    # --- Checkpoint reload ---
    try:
        model2 = MiniTransformerBlock()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model2.load_state_dict(ckpt["model_state_dict"])

        # Verify reloaded model produces same output
        with torch.no_grad():
            out1 = model(x, bias_state)
            out2 = model2(x, bias_state)
        reload_diff = (out1 - out2).abs().max().item()
        results["checkpoint_reload_ok"] = True
        results["reload_max_diff"] = reload_diff
        results["reload_exact"] = reload_diff == 0.0
    except Exception as e:
        results["error"] = f"Checkpoint reload failed: {e}"
        results["traceback"] = traceback.format_exc()
        results["status"] = "FAIL (checkpoint reload)"
        return results

    # --- Second forward to verify no recompilation issues ---
    try:
        # Different edge count but same high-water marks (compile-key stability)
        call_edge_counts_2 = torch.tensor([3, 2], dtype=torch.long, device=device)
        structure_batch_2 = dict(structure_batch)
        structure_batch_2["graph_call_edge_counts"] = call_edge_counts_2
        bias_state_2 = build_chunk_native_graph_bias(
            structure_batch_2, batch_size=B, seqlen_q=S, seqlen_k=S,
            device=device, dtype=torch.float32, beta=1.0,
        )
        with torch.no_grad():
            out3 = model(x, bias_state_2)
        results["second_forward_ok"] = True
        results["second_output_finite"] = bool(torch.isfinite(out3).all().item())
    except Exception as e:
        results["second_forward_error"] = str(e)
        results["second_forward_ok"] = False

    # --- Summary ---
    all_pass = all([
        results.get("forward_ok"),
        results.get("backward_ok"),
        results.get("optimizer_step_ok"),
        results.get("checkpoint_save_ok"),
        results.get("checkpoint_reload_ok"),
        results.get("reload_exact", False),
    ])
    results["status"] = "PASS" if all_pass else "FAIL"
    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    """Run TE<->FA4 parity and Megatron training step on beta23."""
    import json

    print(f"Image: {GHCR_REF}")
    print(f"Image variant: {'beta23 GHCR' if USE_BETA23 else 'default (b19 + runtime upgrade)'}")
    print(f"GPU: {GPU_SPEC}")
    print("=" * 70)

    print("\nTEST 1: TE <-> FA4 parity (dense bias vs chunk-native score_mod)")
    print("-" * 70)
    r1 = test_te_fa4_parity.remote()
    print(json.dumps(r1, indent=2, default=str))

    print("\n" + "=" * 70)
    print("TEST 2: Megatron training step (fwd + bwd + optim + ckpt)")
    print("-" * 70)
    r2 = test_megatron_training_step.remote()
    print(json.dumps(r2, indent=2, default=str))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  TE<->FA4 Parity:       {r1.get('status', 'UNKNOWN')}")
    if r1.get("max_diff") is not None:
        print(f"    max_diff={r1['max_diff']:.6f} (threshold={r1.get('threshold', 0.1)})")
    print(f"  Megatron Training Step: {r2.get('status', 'UNKNOWN')}")
    if r2.get("loss") is not None:
        print(f"    loss={r2['loss']:.6f}")
    if r2.get("reload_max_diff") is not None:
        print(f"    ckpt reload diff={r2['reload_max_diff']}")
    print("=" * 70)

    # Exit non-zero if any test failed
    if r1.get("status") != "PASS" or r2.get("status") != "PASS":
        sys.exit(1)
