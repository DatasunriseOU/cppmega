from cppmega.megatron.upstream_patches import apply_mamba3_stage2_force_nontma_patches as applier


def _patched_text() -> str:
    return "\n".join([*applier._PATCHED_MARKERS.values(), *["disable_tma=True"] * 10])


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


def test_stage2_done_predicates_accept_expected_states(tmp_path, monkeypatch):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    monkeypatch.setattr(applier, "_find_mamba3_bwd_file", lambda: kernel)

    kernel.write_text(_patched_text())
    assert applier._is_stage2_patch_applied()
    assert not applier._is_stage2_patch_absent()

    kernel.write_text("def mamba_mimo_bwd_combined(bb_num_stages=0):\n    pass\n")
    assert not applier._is_stage2_patch_applied()
    assert applier._is_stage2_patch_absent()


def test_stage2_done_predicates_reject_partial_states(tmp_path, monkeypatch):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    kernel.write_text("Q: T.Tensor([B, S * R, G, N], dtype)\nbb_num_stages=0\n")
    monkeypatch.setattr(applier, "_find_mamba3_bwd_file", lambda: kernel)

    assert not applier._is_stage2_patch_applied()
    assert not applier._is_stage2_patch_absent()
