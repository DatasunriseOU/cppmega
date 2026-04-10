from pathlib import Path


def test_remote_mixed_a_smoke_enables_native_dsa_args():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_nam56r_mixed_a.sh"
    ).read_text()

    assert "build_nam56r_megatron_native_args" in script
    assert "enable_mla=True" in script
    assert "enable_mtp=True" in script
    assert "enable_moe=True" in script
    assert "enable_dsa=True" in script
