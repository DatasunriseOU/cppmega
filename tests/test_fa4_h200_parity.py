"""FA4 vs TE dense-bias forward/backward parity test for H200.

Compares:
- Reference (PRIMARY): TE 2.16 cuDNN FusedAttention with post_scale_bias
  (the real production path on Nebius H200)
- Reference (SECONDARY): manual PyTorch attention with dense [B,1,Sq,Sk]
  post_scale_bias (mathematically equivalent, for environments without TE)
- Test: FA4 flash_attn_func with chunk-native score_mod via production
  factory (_make_graph_score_mod)

On the SAME input (Q, K, V, graph edges), verifies:
- Forward outputs match within bf16 tolerance (~2e-3 atol)
- Backward gradients match (dQ, dK, dV) within bf16 tolerance (~5e-3 atol)
- Loss values match

Runs on GPU (H200 via Modal) or is skipped on CPU-only machines.

Usage:
    # Via pytest on GPU:
    pytest tests/test_fa4_h200_parity.py -v

    # Via Modal:
    modal run tests/test_fa4_h200_parity.py

Design doc: docs/fa4_parity_test_design.md
"""

from __future__ import annotations

import math
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip on non-GPU machines
# ---------------------------------------------------------------------------

_HAS_CUDA = torch.cuda.is_available()
_SKIP_REASON = "FA4 parity test requires CUDA (H200)"

# ---------------------------------------------------------------------------
# Detect Transformer Engine availability
# ---------------------------------------------------------------------------

try:
    import transformer_engine.pytorch as te_pytorch  # noqa: F401

    _HAS_TE = True
except ImportError:
    _HAS_TE = False

_TE_SKIP_REASON = (
    "TE reference requires transformer_engine with cuDNN FusedAttention "
    "(TE 2.16+); not installed"
)

# ---------------------------------------------------------------------------
# Mock flash_attn.cute for import-time on CPU (tests are skipped anyway)
# ---------------------------------------------------------------------------

_FA4_MOCK_MODULES = (
    "flash_attn",
    "flash_attn.cute",
    "flash_attn.cute.interface",
    "flash_attn.cute.block_sparsity",
    "flash_attn.cute.utils",
)


def _ensure_flash_attn_importable() -> None:
    """Install mock flash_attn.cute if not present (for import on CPU)."""
    if _HAS_CUDA:
        return  # Real flash_attn available on GPU
    for name in _FA4_MOCK_MODULES:
        if name not in sys.modules:
            mod = MagicMock(spec=ModuleType)
            mod.__name__ = name
            mod.__package__ = name.rpartition(".")[0] or name
            mod.__spec__ = MagicMock()
            mod.__spec__.name = name
            sys.modules[name] = mod
    if "flash_attn.cute.interface" in sys.modules:
        sys.modules["flash_attn.cute.interface"].flash_attn_func = MagicMock()


_ensure_flash_attn_importable()

from cppmega.megatron.fa4_score_mod_adapter import (  # noqa: E402
    ChunkNativeGraphBias,
    build_chunk_native_graph_bias,
    _make_graph_score_mod,
    _make_graph_score_mod_bwd,
)
from cppmega.megatron.graph_route_attention_bias_patch import (  # noqa: E402
    build_dense_graph_attention_bias_from_structure_batch,
)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

B = 2       # batch size
S = 128     # sequence length (multiple FA4 tiles at tile=64)
H = 8       # attention heads
D = 64      # head dimension
BETA = 2.0  # graph bias beta

# Relation weights
CALL_WEIGHT = 1.0
TYPE_WEIGHT = 1.0
DOMAIN_WEIGHT = 3.0
BUILD_WEIGHT = 4.0

# Tolerances for bf16
FWD_ATOL = 2e-3
FWD_RTOL = 1e-2
BWD_ATOL = 5e-3
BWD_RTOL = 2e-2


# ---------------------------------------------------------------------------
# Mock structure batch with known graph edges
# ---------------------------------------------------------------------------


def _parity_structure_batch() -> dict[str, torch.Tensor]:
    """Structure batch exercising both chunk-pair and rare token edges.

    Chunks: [0,32), [32,64), [64,96), [96,128)  (4 chunks, 128 tokens)
    Call edges (chunk pairs): (0,2), (1,3), (0,1)
    Type edges (chunk pairs): (2,0), (3,1)
    Domain edges (token triples): (10, 80, 5), (50, 20, 5)
    Build edges (token triples): (100, 5, 7)
    """
    return {
        "graph_call_edges": torch.tensor(
            [[[0, 2], [1, 3], [0, 1], [-1, -1]]], dtype=torch.long
        ),
        "graph_call_edge_counts": torch.tensor([3], dtype=torch.long),
        "graph_type_edges": torch.tensor(
            [[[2, 0], [3, 1], [-1, -1], [-1, -1]]], dtype=torch.long
        ),
        "graph_type_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_domain_edges": torch.tensor(
            [[[10, 80, 5], [50, 20, 5], [-1, -1, -1]]], dtype=torch.long
        ),
        "graph_domain_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_build_edges": torch.tensor(
            [[[100, 5, 7], [-1, -1, -1]]], dtype=torch.long
        ),
        "graph_build_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 32, 64, 96]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[32, 64, 96, 128]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([4], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Reference (PRIMARY): TE cuDNN FusedAttention with post_scale_bias
# ---------------------------------------------------------------------------


def _te_reference_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dense_bias: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """TE DotProductAttention with post_scale_bias (cuDNN FusedAttention).

    This is the PRIMARY reference: the real production path on Nebius H200.

    Args:
        q: [B, S, H, D] queries (bf16, requires_grad)
        k: [B, S, H, D] keys (bf16, requires_grad)
        v: [B, S, H, D] values (bf16, requires_grad)
        dense_bias: [B, 1, Sq, Sk] additive bias (broadcast over heads)
        causal: whether to apply causal masking

    Returns:
        out: [B, S, H, D] attention output
    """
    import transformer_engine.pytorch as te

    # TE DotProductAttention expects [B, S, H, D] layout
    attn = te.DotProductAttention(
        num_attention_heads=H,
        kv_channels=D,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type="causal" if causal else "no_mask",
    )
    attn = attn.to(device=q.device, dtype=q.dtype)

    # TE forward: (query, key, value, attention_mask=None, core_attention_bias_type="post_scale_bias", core_attention_bias=dense_bias)
    out = attn(
        q,
        k,
        v,
        attention_mask=None,
        core_attention_bias_type="post_scale_bias",
        core_attention_bias=dense_bias,
    )
    return out


# ---------------------------------------------------------------------------
# Reference (SECONDARY): manual PyTorch attention with dense bias
# ---------------------------------------------------------------------------


def _reference_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dense_bias: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Manual scaled dot-product attention with dense post-scale bias.

    Mathematically equivalent to TE cuDNN FusedAttention with post_scale_bias.
    Used as a secondary check when TE is not available.

    Args:
        q: [B, S, H, D] queries (bf16, requires_grad)
        k: [B, S, H, D] keys (bf16, requires_grad)
        v: [B, S, H, D] values (bf16, requires_grad)
        dense_bias: [B, 1, Sq, Sk] additive bias (broadcast over heads)
        causal: whether to apply causal masking

    Returns:
        out: [B, S, H, D] attention output
    """
    # [B, S, H, D] -> [B, H, S, D]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    scale = 1.0 / math.sqrt(D)

    # scores: [B, H, Sq, Sk]
    scores = torch.matmul(q_t.float(), k_t.float().transpose(-2, -1)) * scale

    # Add dense bias [B, 1, Sq, Sk] -> broadcasts over H
    scores = scores + dense_bias.float()

    # Causal mask: -inf above diagonal
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(
            torch.full((sq, sk), float("-inf"), device=scores.device),
            diagonal=1,
        )
        scores = scores + causal_mask

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_t.float())  # [B, H, Sq, D]

    # [B, H, S, D] -> [B, S, H, D]
    return out.transpose(1, 2).to(q.dtype)


# ---------------------------------------------------------------------------
# FA4 attention forward (uses production factory _make_graph_score_mod)
# ---------------------------------------------------------------------------


def _fa4_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias_state: ChunkNativeGraphBias,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """FA4 flash_attn_func with production _make_graph_score_mod factory.

    Builds the 6 flat aux_tensors and creates the score_mod via the
    production factory with c_plus_1 captured in closure.

    Args:
        q: [B, S, H, D] queries (bf16, requires_grad)
        k: [B, S, H, D] keys (bf16, requires_grad)
        v: [B, S, H, D] values (bf16, requires_grad)
        bias_state: ChunkNativeGraphBias from build_chunk_native_graph_bias
        causal: whether to apply causal masking

    Returns:
        out: [B, S, H, D] attention output
    """
    from flash_attn.cute.interface import flash_attn_func

    scale = 1.0 / math.sqrt(D)

    # c_plus_1 = chunk_bias square dimension (C+1)
    c_plus_1 = bias_state.chunk_bias.shape[1]

    # Flatten chunk_bias [B, C+1, C+1] -> [B, (C+1)*(C+1)]
    chunk_bias_flat = bias_state.chunk_bias.reshape(
        bias_state.chunk_bias.shape[0], -1
    ).contiguous()

    # 6 flat aux_tensors (c_plus_1 is in the factory closure, not in the list)
    aux_tensors = [
        bias_state.token_to_chunk_q,
        bias_state.token_to_chunk_k,
        chunk_bias_flat,
        bias_state.rare_q,
        bias_state.rare_k,
        bias_state.rare_w,
    ]

    # Create score_mod via production factory
    score_mod = _make_graph_score_mod(c_plus_1)
    score_mod_bwd = _make_graph_score_mod_bwd(c_plus_1)

    out = flash_attn_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=scale,
        causal=causal,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        aux_tensors=aux_tensors,
        block_sparse_tensors=None,
        mask_mod=None,
        return_lse=False,
    )

    if isinstance(out, tuple):
        out = out[0]

    return out


# ---------------------------------------------------------------------------
# Parity test class
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason=_SKIP_REASON)
class TestFA4H200Parity:
    """Forward/backward parity: FA4 score_mod vs TE dense bias on H200."""

    def _setup_inputs(self, device: torch.device):
        """Create Q, K, V and both bias representations on device."""
        torch.manual_seed(42)

        q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)

        sb = _parity_structure_batch()

        # Build dense bias (TE reference path)
        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=B,
            seqlen_q=S,
            seqlen_k=S,
            device=device,
            dtype=torch.float32,
            call_weight=CALL_WEIGHT,
            type_weight=TYPE_WEIGHT,
            domain_weight=DOMAIN_WEIGHT,
            build_weight=BUILD_WEIGHT,
            beta=BETA,
        )

        # Build chunk-native bias (FA4 path)
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=B,
            seqlen_q=S,
            seqlen_k=S,
            device=device,
            dtype=torch.float32,
            beta=BETA,
            call_weight=CALL_WEIGHT,
            type_weight=TYPE_WEIGHT,
            domain_weight=DOMAIN_WEIGHT,
            build_weight=BUILD_WEIGHT,
        )

        return q, k, v, dense_bias, chunk_native

    def _get_reference_output(self, q, k, v, dense_bias):
        """Get reference output using TE (primary) or manual PyTorch (secondary)."""
        if _HAS_TE:
            return _te_reference_attention_forward(q, k, v, dense_bias, causal=True)
        return _reference_attention_forward(q, k, v, dense_bias, causal=True)

    @pytest.mark.skipif(not _HAS_TE, reason=_TE_SKIP_REASON)
    def test_forward_parity_te(self):
        """FA4 forward output matches TE cuDNN FusedAttention (PRIMARY ref)."""
        device = torch.device("cuda")
        q, k, v, dense_bias, chunk_native = self._setup_inputs(device)

        # PRIMARY Reference: TE cuDNN FusedAttention with post_scale_bias
        out_ref = _te_reference_attention_forward(q, k, v, dense_bias, causal=True)

        # Test: FA4 with production score_mod factory
        out_fa4 = _fa4_attention_forward(q, k, v, chunk_native, causal=True)

        assert out_ref.shape == out_fa4.shape, (
            f"Shape mismatch: ref={out_ref.shape}, fa4={out_fa4.shape}"
        )

        torch.testing.assert_close(
            out_fa4,
            out_ref,
            atol=FWD_ATOL,
            rtol=FWD_RTOL,
            msg="FA4 forward output diverges from TE cuDNN FusedAttention",
        )

    def test_forward_parity(self):
        """FA4 forward output matches reference within bf16 tol."""
        device = torch.device("cuda")
        q, k, v, dense_bias, chunk_native = self._setup_inputs(device)

        # Reference: TE (primary) or manual PyTorch (secondary)
        out_ref = self._get_reference_output(q, k, v, dense_bias)

        # Test: FA4 with production score_mod factory
        out_fa4 = _fa4_attention_forward(q, k, v, chunk_native, causal=True)

        assert out_ref.shape == out_fa4.shape, (
            f"Shape mismatch: ref={out_ref.shape}, fa4={out_fa4.shape}"
        )

        torch.testing.assert_close(
            out_fa4,
            out_ref,
            atol=FWD_ATOL,
            rtol=FWD_RTOL,
            msg="FA4 forward output diverges from dense-bias reference",
        )

    def test_backward_parity_dq(self):
        """FA4 dQ gradient matches reference within bf16 tol."""
        device = torch.device("cuda")

        # Reference path
        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = self._get_reference_output(q_ref, k_ref, v_ref, dense_bias)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        dq_ref = q_ref.grad.clone()

        # FA4 path (fresh tensors, same seed)
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _fa4_attention_forward(q_fa4, k_fa4, v_fa4, chunk_native, causal=True)
        loss_fa4 = out_fa4.sum()
        loss_fa4.backward()
        dq_fa4 = q_fa4.grad.clone()

        torch.testing.assert_close(
            dq_fa4,
            dq_ref,
            atol=BWD_ATOL,
            rtol=BWD_RTOL,
            msg="FA4 dQ gradient diverges from dense-bias reference",
        )

    def test_backward_parity_dk(self):
        """FA4 dK gradient matches reference within bf16 tol."""
        device = torch.device("cuda")

        # Reference path
        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = self._get_reference_output(q_ref, k_ref, v_ref, dense_bias)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        dk_ref = k_ref.grad.clone()

        # FA4 path
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _fa4_attention_forward(q_fa4, k_fa4, v_fa4, chunk_native, causal=True)
        loss_fa4 = out_fa4.sum()
        loss_fa4.backward()
        dk_fa4 = k_fa4.grad.clone()

        torch.testing.assert_close(
            dk_fa4,
            dk_ref,
            atol=BWD_ATOL,
            rtol=BWD_RTOL,
            msg="FA4 dK gradient diverges from dense-bias reference",
        )

    def test_backward_parity_dv(self):
        """FA4 dV gradient matches reference within bf16 tol."""
        device = torch.device("cuda")

        # Reference path
        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = self._get_reference_output(q_ref, k_ref, v_ref, dense_bias)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        dv_ref = v_ref.grad.clone()

        # FA4 path
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _fa4_attention_forward(q_fa4, k_fa4, v_fa4, chunk_native, causal=True)
        loss_fa4 = out_fa4.sum()
        loss_fa4.backward()
        dv_fa4 = v_fa4.grad.clone()

        torch.testing.assert_close(
            dv_fa4,
            dv_ref,
            atol=BWD_ATOL,
            rtol=BWD_RTOL,
            msg="FA4 dV gradient diverges from dense-bias reference",
        )

    def test_loss_parity(self):
        """FA4 loss matches reference loss."""
        device = torch.device("cuda")

        # Reference path
        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = self._get_reference_output(q_ref, k_ref, v_ref, dense_bias)
        loss_ref = out_ref.sum()

        # FA4 path
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _fa4_attention_forward(q_fa4, k_fa4, v_fa4, chunk_native, causal=True)
        loss_fa4 = out_fa4.sum()

        # Loss is a scalar; compare with tight tolerance relative to magnitude
        loss_ref_val = loss_ref.item()
        loss_fa4_val = loss_fa4.item()
        rel_err = abs(loss_fa4_val - loss_ref_val) / (abs(loss_ref_val) + 1e-8)

        assert rel_err < 1e-3, (
            f"Loss divergence: ref={loss_ref_val:.6f}, fa4={loss_fa4_val:.6f}, "
            f"rel_err={rel_err:.6e}"
        )

    def test_all_gradients_combined(self):
        """Combined dQ+dK+dV parity in a single forward/backward pass."""
        device = torch.device("cuda")

        # Reference path
        q_ref, k_ref, v_ref, dense_bias, _ = self._setup_inputs(device)
        out_ref = self._get_reference_output(q_ref, k_ref, v_ref, dense_bias)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # FA4 path
        q_fa4, k_fa4, v_fa4, _, chunk_native = self._setup_inputs(device)
        out_fa4 = _fa4_attention_forward(q_fa4, k_fa4, v_fa4, chunk_native, causal=True)
        loss_fa4 = out_fa4.sum()
        loss_fa4.backward()

        # Check all gradients
        for name, grad_fa4, grad_ref in [
            ("dQ", q_fa4.grad, q_ref.grad),
            ("dK", k_fa4.grad, k_ref.grad),
            ("dV", v_fa4.grad, v_ref.grad),
        ]:
            assert grad_fa4 is not None, f"{name} is None for FA4 path"
            assert grad_ref is not None, f"{name} is None for reference path"
            torch.testing.assert_close(
                grad_fa4,
                grad_ref,
                atol=BWD_ATOL,
                rtol=BWD_RTOL,
                msg=f"FA4 {name} gradient diverges from dense-bias reference",
            )


# ---------------------------------------------------------------------------
# Bias equivalence sanity check (runs on CPU, no GPU needed)
# ---------------------------------------------------------------------------


class TestBiasEquivalenceCPU:
    """Verify dense and chunk-native biases are equivalent (CPU, no FA4)."""

    def test_dense_and_chunk_native_bias_equivalent(self):
        """For every (b,q,k), chunk_native bias == dense bias exactly."""
        device = torch.device("cpu")
        sb = _parity_structure_batch()

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=B,
            seqlen_q=S,
            seqlen_k=S,
            device=device,
            dtype=torch.float32,
            call_weight=CALL_WEIGHT,
            type_weight=TYPE_WEIGHT,
            domain_weight=DOMAIN_WEIGHT,
            build_weight=BUILD_WEIGHT,
            beta=BETA,
        )

        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=B,
            seqlen_q=S,
            seqlen_k=S,
            device=device,
            dtype=torch.float32,
            beta=BETA,
            call_weight=CALL_WEIGHT,
            type_weight=TYPE_WEIGHT,
            domain_weight=DOMAIN_WEIGHT,
            build_weight=BUILD_WEIGHT,
        )

        # Reconstruct dense bias from chunk-native representation
        reconstructed = torch.zeros(B, S, S, dtype=torch.float32)
        for b in range(B):
            for q_idx in range(S):
                qc = int(chunk_native.token_to_chunk_q[b, q_idx].item())
                for k_idx in range(S):
                    kc = int(chunk_native.token_to_chunk_k[b, k_idx].item())
                    val = float(chunk_native.chunk_bias[b, qc, kc].item())
                    # Add rare edge contribution
                    lo = int(chunk_native.rare_row_offsets[b, q_idx].item())
                    hi = int(chunk_native.rare_row_offsets[b, q_idx + 1].item())
                    for i in range(lo, hi):
                        if int(chunk_native.rare_k[b, i].item()) == k_idx:
                            val += float(chunk_native.rare_w[b, i].item())
                            break
                    reconstructed[b, q_idx, k_idx] = val

        # Compare: dense_bias is [B, 1, S, S], reconstructed is [B, S, S]
        dense_squeezed = dense_bias.squeeze(1)
        torch.testing.assert_close(
            reconstructed,
            dense_squeezed,
            atol=1e-5,
            rtol=1e-5,
            msg="Chunk-native bias does not reproduce dense bias",
        )


# ---------------------------------------------------------------------------
# Modal entrypoint (run as: modal run tests/test_fa4_h200_parity.py)
# ---------------------------------------------------------------------------

# Pinned GHCR image digest (immutable, reproducible)
_GHCR_IMAGE_DIGEST = (
    "sha256:08c5db7368d1037d930e0825281468927de9c85b12ba10373fe07e082150d983"
)


def _modal_image():
    """Build the Modal image for H200 parity testing."""
    import pathlib
    from typing import Any

    import modal

    _REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
    GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
    GHCR_REF = f"{GHCR_REPO}@{_GHCR_IMAGE_DIGEST}"

    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env(
        {
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "WANDB_MODE": "disabled",
        }
    )
    img = img.pip_install("pytest")
    img = (
        img.add_local_dir(
            str(_REPO_ROOT / "cppmega"), remote_path="/opt/cppmega/cppmega", copy=True
        )
        .add_local_dir(
            str(_REPO_ROOT / "tests"), remote_path="/opt/cppmega/tests", copy=True
        )
        .add_local_file(
            str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml"
        )
    )
    return img


def _run_parity_on_modal() -> None:
    """Modal entrypoint: run the parity test on H200."""
    import json
    import pathlib
    import subprocess

    import modal

    GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:1")
    app = modal.App("cppmega-fa4-parity")
    results_vol = modal.Volume.from_name("cppmega-test-results", create_if_missing=True)

    @app.function(
        image=_modal_image(),
        gpu=GPU_SPEC,
        timeout=600,
        volumes={"/results": results_vol},
    )
    def run_parity() -> dict:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/test_fa4_h200_parity.py",
                "-v", "--tb=short",
                "-k", "TestFA4H200Parity",
            ],
            cwd="/opt/cppmega",
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=500,
        )
        result = {
            "returncode": proc.returncode,
            "gpu": GPU_SPEC,
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-80:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-30:]),
        }
        pathlib.Path("/results").mkdir(parents=True, exist_ok=True)
        pathlib.Path("/results/fa4_parity.json").write_text(json.dumps(result, indent=2))
        results_vol.commit()
        return result

    @app.local_entrypoint()
    def main() -> None:
        result = run_parity.remote()
        print(json.dumps(result, indent=2))
        if result["returncode"] != 0:
            raise SystemExit(1)

    main()


if __name__ == "__main__":
    _run_parity_on_modal()
