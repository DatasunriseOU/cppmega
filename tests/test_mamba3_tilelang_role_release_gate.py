from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = _ROOT / "scripts/modal_mamba3_tilelang_role_release_gate.py"


def test_release_image_contains_and_applies_exact_stage2_patch():
    dockerignore = (_ROOT / ".dockerignore").read_text()
    dockerfile = (_ROOT / "docker/Dockerfile").read_text()

    assert "upstream_prs/examples/*" in dockerignore
    assert (
        "!upstream_prs/examples/13_tilelang_floormod_dbz/"
        "mamba3_bwd_stage2_force_nontma.patch"
    ) in dockerignore
    assert "CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1" in dockerfile
    assert "MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1" in dockerfile
    assert "assert stage2._is_stage2_patch_applied()" in dockerfile
    assert "assert not stage2._is_stage2_patch_absent()" in dockerfile


def test_release_gate_is_exact_read_only_and_reopens_full_ordered_chain():
    harness = _HARNESS.read_text()

    for required_env in (
        "CPPMEGA_CANDIDATE_CPPMEGA_SHA",
        "CPPMEGA_CANDIDATE_IMAGE_DIGEST",
        "CPPMEGA_RELEASE_MANIFEST_SHA256",
        "CPPMEGA_CANDIDATE_WHEELS_JSON",
        "CPPMEGA_H200_GATE_PHASE",
    ):
        assert required_env in harness
    assert '"one": {' in harness
    assert '"r2": {' in harness
    assert '"r4": {' in harness
    assert '"prerequisite_phase": "one"' in harness
    assert '"prerequisite_phase": "r2"' in harness
    assert "status.stdout" in harness
    assert '"--untracked-files=all"' in harness

    assert "installed payload differs from exact release wheel" in harness
    assert "verified_payload_identity_sha256" in harness
    assert "verify_phase_artifact(str(prerequisite_phase))" in harness
    assert "prior.get(\"prerequisite\") != actual_prerequisite" in harness
    assert '"modal_derived_image_stage2_mutated": False' in harness
    assert "refusing to overwrite an existing exact gate attempt" in harness
    assert 'receipt["gpu_health_before_test"]' in harness
    assert "isinstance(exc, subprocess.TimeoutExpired)" in harness
    assert ".add_local_file(\n            str(_LOCAL_STAGE2_PATCH)" not in harness
