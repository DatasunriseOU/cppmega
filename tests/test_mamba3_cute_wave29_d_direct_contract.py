from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multi_chunk_fused_consumers_include_direct_d_dphi_contract() -> None:
    src = ROOT / "cppmega/megatron/cute_dsl_mimo/state_apply_consumers.py"
    text = src.read_text()

    assert "gD: cute.Tensor" in text
    assert "direct_d = Float32(gD[0])" in text
    assert "dpsi += direct_d * Float32(sDPhT[p, f])" in text
    assert "fake_f32_1d((1,))" in text


def test_lkq_probe_exercises_wave29_direct_d_reference() -> None:
    src = ROOT / "cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py"
    text = src.read_text()

    assert "wave29_multi_chunk_semantics: direct D*dPhi contribution included" in text
    assert "d_contrib = inputs.d.float()[0] * inputs.dphi[c].float()" in text
    assert "d_direct_contribution_for_multi_chunk_path" in text
    assert "peak_cuda_memory" in text
