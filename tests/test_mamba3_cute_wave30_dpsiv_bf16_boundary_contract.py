from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multi_chunk_consumers_round_combined_dpsiv_d_before_consumers() -> None:
    src = ROOT / "cppmega/megatron/cute_dsl_mimo/state_apply_consumers.py"
    text = src.read_text()

    assert "dPsiV_D.to(bf16)" in text
    assert "dpsi += direct_d * Float32(sDPhT[p, f])" in text
    assert "Float32(gQKDot_c[t, r_out, r])" in text
    assert text.count("dpsi = Float32(self.dtype(dpsi))") == 2


def test_lkq_probe_reference_uses_combined_dpsiv_d_bf16_boundary() -> None:
    src = ROOT / "cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py"
    text = src.read_text()

    assert "dpsi_pre_boundary = state.float() + apply.float() + d_contrib + qk_contrib" in text
    assert "dpsi = dpsi_pre_boundary.to(torch.bfloat16).float()" in text
    assert "wave30_multi_chunk_semantics: combined dPsiV_D BF16 boundary" in text
    assert "dpsiv_d_bf16_boundary_for_multi_chunk_path" in text
    assert "dpsiv_d_global_materialized_for_multi_chunk_path" in text
