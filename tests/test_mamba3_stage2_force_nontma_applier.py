from cppmega.megatron.upstream_patches import apply_mamba3_stage2_force_nontma_patches as applier


def test_stage2_partial_marker_ignores_baseline_bb_default():
    text = "def mamba_mimo_bwd_combined(bb_num_stages=0):\n    pass\n"

    assert not applier._is_patched(text)
    assert not applier._has_partial_stage2_markers(text)


def test_stage2_partial_marker_flags_structural_subset():
    text = "\n".join(
        [
            "Q: T.Tensor([B, S * R, G, N], dtype)",
            "bb_num_stages=0",
        ]
    )

    assert not applier._is_patched(text)
    assert applier._has_partial_stage2_markers(text)
