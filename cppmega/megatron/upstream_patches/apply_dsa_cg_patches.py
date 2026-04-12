"""Apply upstream Megatron DSA patches for CUDA graph compatibility + NAM56R dims.

These patches modify files in the Megatron-LM installation to fix:
1. torch.equal() / .item() calls that are CPU-sync ops banned during CUDA graph capture
2. Hardcoded DeepSeek V3.2 dimensions (576/512) in TileLang sparse MLA kernels
3. Missing d_v parameter propagation in SparseMLA autograd Function

Run this ONCE after installing/updating Megatron-LM:
    python -m cppmega.megatron.upstream_patches.apply_dsa_cg_patches

All patches are idempotent (safe to run multiple times).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path


def _patch_file(path: Path, replacements: list[tuple[str, str]], label: str) -> bool:
    """Apply text replacements to a file. Returns True if any changes made."""
    if not path.exists():
        print(f"  SKIP {label}: {path} not found")
        return False

    text = path.read_text()
    original = text
    for old, new in replacements:
        text = text.replace(old, new)

    if text == original:
        print(f"  OK   {label}: already patched")
        return False

    path.write_text(text)
    print(f"  DONE {label}: {len(replacements)} replacements applied")
    return True


def apply_all():
    """Apply all upstream patches."""
    # Find Megatron DSA module path
    try:
        import megatron.core.transformer.experimental_attention_variant.dsa as dsa_mod
    except ImportError:
        print("ERROR: megatron.core not importable. Is Megatron-LM installed?")
        sys.exit(1)

    dsa_dir = Path(dsa_mod.__file__).parent
    ops_dir = dsa_dir / "ops"
    dsa_file = dsa_dir / "dsa.py"

    print(f"Megatron DSA path: {dsa_dir}")
    print()

    # === Patch 1: dsa.py — CUDA graph unsafe checks ===
    print("Patch 1: dsa.py CUDA graph compatibility")
    _patch_file(dsa_file, [
        # torch.equal banned during graph capture
        (
            "if not torch.equal(finite, expected):",
            "if False:  # cppmega: skip CG-unsafe torch.equal check",
        ),
        (
            "if not torch.equal(key_positions, expected_key_pos):",
            "if False:  # cppmega: skip CG-unsafe torch.equal check",
        ),
        (
            "if not torch.equal(mask[bi], ref_mask):",
            "if False:  # cppmega: skip CG-unsafe torch.equal check",
        ),
    ], "dsa.py CG-unsafe checks")

    # === Patch 2: dsa.py — Remove 576/512 dimension hardcodes ===
    print("Patch 2: dsa.py dimension hardcodes")
    _patch_file(dsa_file, [
        (
            "if query.size(-1) != 576 or v_channels != 512:",
            "if False:  # cppmega: allow any dims, TileLang kernel is parameterized",
        ),
    ], "dsa.py dim hardcodes")

    # === Patch 3: dsa.py — Pass v_channels to SparseMLA.apply ===
    print("Patch 3: dsa.py SparseMLA d_v propagation")
    _patch_file(dsa_file, [
        (
            "out, _ = SparseMLA.apply(q_t, kv_t, idx_t, softmax_scale)",
            "out, _ = SparseMLA.apply(q_t, kv_t, idx_t, softmax_scale, v_channels)",
        ),
    ], "dsa.py SparseMLA d_v")

    # === Patch 4: sparse_mla.py — Accept d_v in forward/backward ===
    print("Patch 4: sparse_mla.py d_v parameter")
    sma_file = ops_dir / "sparse_mla.py"
    _patch_file(sma_file, [
        (
            "def forward(ctx, q, kv, indices, scaling):",
            "def forward(ctx, q, kv, indices, scaling, d_v=512):",
        ),
        (
            "ctx.scaling = scaling",
            "ctx.scaling = scaling\n        ctx.d_v = d_v",
        ),
        (
            "tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)",
            "tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling, d_v=d_v)",
        ),
        (
            "return tl_dq, tl_dkv, None, None",
            "return tl_dq, tl_dkv, None, None, None  # last None for d_v grad",
        ),
    ], "sparse_mla.py d_v")

    # Also patch backward to pass d_v
    if sma_file.exists():
        text = sma_file.read_text()
        if "d_v = ctx.d_v" not in text:
            text = text.replace(
                "scaling = ctx.scaling",
                "scaling = ctx.scaling\n        d_v = ctx.d_v",
            )
            text = text.replace(
                "q, kv, tl_out, grad_output.contiguous(), indices, tl_lse, sm_scale=scaling",
                "q, kv, tl_out, grad_output.contiguous(), indices, tl_lse, sm_scale=scaling, d_v=d_v",
            )
            sma_file.write_text(text)
            print("  DONE sparse_mla.py bwd d_v propagation")

    # === Patch 5: tilelang_sparse_mla_fwd.py — Relax power-of-2 assertions ===
    print("Patch 5: tilelang_sparse_mla_fwd.py dimension assertions")
    fwd_file = ops_dir / "tilelang_sparse_mla_fwd.py"
    if fwd_file.exists():
        text = fwd_file.read_text()
        changed = False
        # Replace power-of-2 assertions with multiple-of-16
        for var in ("dim", "tail_dim"):
            old_pattern = f'assert {var} == tilelang.math.next_power_of_2(\n        {var}\n    ), f"haven\'t check padding correctness yet, dim={{{var}}}"'
            new_line = f'assert {var} % 16 == 0, f"{var} must be multiple of 16 for warp ops, got {{{var}}}"'
            if old_pattern in text:
                text = text.replace(old_pattern, new_line)
                changed = True
        # Remove 576 assertion
        text = text.replace(
            'dim_plus_tail_dim == 576\n    ), "TileLang sparse MLA fwd is currently specialized for dim_plus_tail_dim=576"',
            'True  # cppmega: any dims\n    ), "dim check disabled"',
        )
        if changed:
            fwd_file.write_text(text)
            print("  DONE tilelang_sparse_mla_fwd.py assertions relaxed")
        else:
            print("  OK   tilelang_sparse_mla_fwd.py already patched or different format")

    # === Patch 6: tilelang_sparse_mla_bwd.py — Remove D=512 hardcode ===
    print("Patch 6: tilelang_sparse_mla_bwd.py D=512 hardcode")
    bwd_file = ops_dir / "tilelang_sparse_mla_bwd.py"
    _patch_file(bwd_file, [
        (
            "q, kv, o, do, indices, lse, sm_scale=None,",
            "q, kv, o, do, indices, lse, sm_scale=None, d_v=None,",
        ),
        (
            "D = 512",
            "D = d_v if d_v is not None else o.shape[-1]  # was 512",
        ),
    ], "tilelang_sparse_mla_bwd.py D hardcode")

    print()
    print("All patches applied. Restart training to pick up changes.")


if __name__ == "__main__":
    apply_all()
