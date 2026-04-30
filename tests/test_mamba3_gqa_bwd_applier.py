from pathlib import Path

from cppmega.megatron.upstream_patches import apply_mamba3_gqa_bwd_patches as applier


def _module_text(block: str) -> str:
    return (
        "def mamba_mimo_bwd_combined(G, H):\n"
        "    if G == 1:\n"
        "        pass\n"
        f"{block}"
        "    return G, H\n"
    )


def _write_target(path: Path, block: str) -> None:
    path.write_text(_module_text(block))


def test_gqa_patch_text_adds_intermediate_group_branch():
    path = Path("mamba3_mimo_bwd.py")
    text = _module_text(applier._REGULAR_ORIGINAL_BLOCK)

    patched, changed = applier._patch_text(text, path)

    assert changed
    assert "elif H % G == 0:" in patched
    assert "hpg = H // G" in patched
    assert "dq_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)" in patched
    assert "G must divide H" in patched


def test_gqa_patch_text_is_idempotent():
    path = Path("mamba3_mimo_bwd.py")
    text = _module_text(applier._REGULAR_PATCHED_BLOCK)

    patched, changed = applier._patch_text(text, path)

    assert not changed
    assert patched == text


def test_gqa_partial_marker_rejected():
    path = Path("mamba3_mimo_bwd.py")
    text = _module_text(applier._REGULAR_ORIGINAL_BLOCK).replace(
        'raise ValueError(f"G value of {G} is not currently supported!")',
        'raise ValueError(f"G value of {G} is not currently supported (H={H}, G must divide H)!")',
    )

    try:
        applier._patch_text(text, path)
    except RuntimeError as exc:
        assert "partial" in str(exc)
    else:
        raise AssertionError("partial GQA patch markers should be rejected")


def test_gqa_do_patch_and_rollback_from_backups(tmp_path, monkeypatch):
    regular = tmp_path / "mamba3_mimo_bwd.py"
    varlen = tmp_path / "mamba3_mimo_bwd_varlen.py"
    _write_target(regular, applier._REGULAR_ORIGINAL_BLOCK)
    _write_target(varlen, applier._VARLEN_ORIGINAL_BLOCK)
    monkeypatch.setattr(applier, "_target_paths", lambda: [regular, varlen])

    applier._do_patch()

    assert applier._is_patched_text(regular.read_text())
    assert applier._is_patched_text(varlen.read_text())
    assert applier._backup_path(regular).exists()
    assert applier._backup_path(varlen).exists()
    assert "dmimo_v.sum(dim=(0, 2))" in varlen.read_text()

    applier.rollback()

    assert not applier._is_patched_text(regular.read_text())
    assert not applier._is_patched_text(varlen.read_text())
    assert applier._REGULAR_ORIGINAL_BLOCK in regular.read_text()
    assert applier._VARLEN_ORIGINAL_BLOCK in varlen.read_text()
