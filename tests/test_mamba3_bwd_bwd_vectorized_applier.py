from cppmega.megatron.upstream_patches import (
    apply_mamba3_bwd_bwd_vectorized_patches as applier,
)
from cppmega.megatron.upstream_patches import (
    apply_mamba3_stage2_force_nontma_patches as stage2,
)


def _stage2_markers() -> str:
    return "\n".join([*stage2._PATCHED_MARKERS.values(), *["disable_tma=True"] * 10])


def _stage2_source_text() -> str:
    return "\n".join(
        [
            _stage2_markers(),
            applier._ALLOC_ORIGINAL,
            applier._LAYOUT_ORIGINAL,
            applier._GEMM_ORIGINAL,
            applier._GAMMA_SCALE_ORIGINAL,
            applier._DK_ORIGINAL,
            applier._DQ_ORIGINAL,
        ]
    )


def test_vectorized_diag_patch_preserves_full_gemm_and_removes_full_shared():
    patched, changed = applier._patch_stage2_text(_stage2_source_text())

    assert changed
    assert "Wave32: vectorized per-step R*R x P reduction microkernel" in patched
    assert "dqk_step_prereduce_frag = T.alloc_fragment([R * R, P], accum_dtype)" in patched
    assert "T.reduce_sum(dqk_step_prereduce_frag, dqk_step_frag, dim=-1, clear=True)" in patched
    assert "T.gemm(dPhiO_step_frag, PsiV_step_frag, dqk_step_frag" not in patched
    assert "T.gemm(dPhiO_shared, PsiV_shared, dqk_from_diag_frag" not in patched
    assert "dqk_diag_shared = T.alloc_shared([chunk_size, R * R], accum_dtype)" in patched
    assert "dqk_from_diag_shared = T.alloc_shared" not in patched
    assert "T.copy(dqk_from_diag_frag, dqk_from_diag_shared)" not in patched
    assert "for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size)" not in patched
    assert "dqk_diag_shared[cs, r_out * R + (csr_in % R)]" in patched
    assert "dqk_diag_shared[cs, (csr_out % R) * R + r_in]" in patched
    applier._validate_patched_text(patched)


def test_vectorized_diag_patch_is_idempotent():
    patched, changed = applier._patch_stage2_text(_stage2_source_text())

    patched_again, changed_again = applier._patch_stage2_text(patched)

    assert changed
    assert not changed_again
    assert patched_again == patched


def test_vectorized_diag_patch_rejects_unpatched_stage2_base():
    try:
        applier._patch_stage2_text(applier._ALLOC_ORIGINAL)
    except RuntimeError as exc:
        assert "stage2" in str(exc)
    else:
        raise AssertionError("unpatched stage2 base should be rejected")


def test_vectorized_diag_partial_marker_rejected():
    text = _stage2_source_text().replace(
        applier._ALLOC_ORIGINAL,
        applier._ALLOC_PATCHED,
    )

    try:
        applier._patch_stage2_text(text)
    except RuntimeError as exc:
        assert "partial" in str(exc)
    else:
        raise AssertionError("partial Wave32 vectorized diag markers should be rejected")
