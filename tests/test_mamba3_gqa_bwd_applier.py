from cppmega.megatron.upstream_patches import apply_mamba3_gqa_bwd_patches as applier


def _unpatched_files(tmp_path):
    nonvar = tmp_path / "mamba3_mimo_bwd.py"
    varlen = tmp_path / "mamba3_mimo_bwd_varlen.py"
    prefix = "def f(G, H):\n    if G == 1:\n        pass\n"
    nonvar.write_text(prefix + applier._NONVARLEN_UNPATCHED + "\n")
    varlen.write_text(prefix + applier._VARLEN_UNPATCHED + "\n")
    return {
        "mamba3_mimo_bwd.py": nonvar,
        "mamba3_mimo_bwd_varlen.py": varlen,
    }


def test_gqa_bwd_text_markers_accept_patched_block():
    text = applier._NONVARLEN_PATCHED

    assert applier._is_patched_text(text)
    assert not applier._has_partial_gqa_markers(text)


def test_gqa_bwd_partial_markers_rejected():
    text = "elif H % G == 0:\n        hpg = H // G\n"

    assert not applier._is_patched_text(text)
    assert applier._has_partial_gqa_markers(text)


def test_gqa_bwd_patch_and_rollback(tmp_path, monkeypatch):
    files = _unpatched_files(tmp_path)
    monkeypatch.setattr(applier, "_find_mamba3_bwd_files", lambda: files)

    assert applier._is_gqa_bwd_patch_absent()
    assert not applier._is_gqa_bwd_patch_applied()

    applier._do_patch()
    for path in files.values():
        assert applier._is_patched_text(path.read_text())
        assert path.with_name(path.name + applier._BACKUP_SUFFIX).exists()
    assert applier._is_gqa_bwd_patch_applied()
    assert not applier._is_gqa_bwd_patch_absent()

    applier.rollback()
    for path in files.values():
        assert not applier._is_patched_text(path.read_text())
    assert applier._is_gqa_bwd_patch_absent()


def test_gqa_bwd_done_predicates_reject_partial_state(tmp_path, monkeypatch):
    files = _unpatched_files(tmp_path)
    files["mamba3_mimo_bwd.py"].write_text("elif H % G == 0:\n")
    monkeypatch.setattr(applier, "_find_mamba3_bwd_files", lambda: files)

    assert not applier._is_gqa_bwd_patch_applied()
    assert not applier._is_gqa_bwd_patch_absent()
