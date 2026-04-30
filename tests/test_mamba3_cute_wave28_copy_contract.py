from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fused_state_apply_consumers_use_wave10_uint4_g2s_contract() -> None:
    src = ROOT / "cppmega/megatron/cute_dsl_mimo/state_apply_consumers.py"
    text = src.read_text()

    assert "WAVE10_UINT4_COPY_BITS = 128" in text
    assert "WAVE10_UINT4_COPY_BYTES = WAVE10_UINT4_COPY_BITS // 8" in text
    assert "CPPMEGA_MAMBA3_CUTE_MULTI_UINT4_G2S" in text
    assert "copy_bits=self.g2s_copy_bits" in text
    assert "key = (dim, rank, chunk_size, nchunks, copy_bits)" in text


def test_lkq_probe_reports_wave10_uint4_copy_contract() -> None:
    src = ROOT / "cppmega/megatron/cute_dsl_mimo/lkq_tile_chain_test.py"
    text = src.read_text()

    assert "wave10_uint4_128bit_g2s_multi_chunk_opt_in" in text
    assert "active_multi_chunk_copy_bits" in text
    assert "bf16_elements_per_copy" in text
    assert "mamba3-mono-triton-model commit 65ef653" in text


def test_modal_harness_propagates_wave28_uint4_opt_in() -> None:
    src = ROOT / "scripts/modal_mamba3_mono_chunk_wave2.py"
    text = src.read_text()

    assert "CPPMEGA_MAMBA3_CUTE_MULTI_UINT4_G2S" in text
