import pytest

from cppmega.megatron.upstream_patches import (
    apply_mamba3_stage2_force_nontma_patches as applier,
)


def _patched_text() -> str:
    return "\n".join(
        [
            *applier._PATCHED_MARKERS.values(),
            applier._PATCHED_MARKERS["bwd_tma_disabled"],
            applier._PATCHED_MARKERS["bwd_ws_disabled"],
            *["disable_tma=True"] * 10,
        ]
    )


def test_stage2_patch_preserves_fail_closed_bwd_pass_configs():
    patch = applier._patch_path().read_text()

    unsafe_patch_lines = (
        "-        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,",
        "-        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,",
        "+        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,",
        "+        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,",
    )
    assert not set(unsafe_patch_lines).intersection(patch.splitlines())


def test_stage2_partial_marker_ignores_baseline_bb_default():
    text = "def mamba_mimo_bwd_combined(bb_num_stages=0):\n    pass\n"

    assert not applier._is_patched(text)
    assert not applier._has_partial_stage2_markers(text)


def test_stage2_partial_marker_flags_structural_subset():
    text = "Q: T.Tensor([B, S * R, G, N], dtype)\nbb_num_stages=0"

    assert not applier._is_patched(text)
    assert applier._has_partial_stage2_markers(text)


def test_stage2_validator_accepts_expected_state(tmp_path):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    kernel.write_text(_patched_text())
    applier._validate_patched(kernel)
    assert applier._is_patched(kernel.read_text())


def test_stage2_validator_rejects_unsafe_bwd_pass_configs(tmp_path):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    text = _patched_text().replace(
        applier._PATCHED_MARKERS["bwd_tma_disabled"],
        "",
        1,
    )
    kernel.write_text(text)

    with pytest.raises(RuntimeError, match="both backward kernels"):
        applier._validate_patched(kernel)
