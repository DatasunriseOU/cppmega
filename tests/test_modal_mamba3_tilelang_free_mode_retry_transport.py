from pathlib import Path

_HARNESS = (
    Path(__file__).resolve().parents[1]
    / "scripts/modal_mamba3_tilelang_free_mode_retry_gate.py"
)


def test_retry_gate_uses_sandbox_and_persists_before_failing() -> None:
    source = _HARNESS.read_text(encoding="utf-8")
    local_entrypoint = source[source.index("@app.local_entrypoint()") :]

    assert "modal.Volume" not in source
    assert ".commit()" not in source
    assert "@app.function" not in source
    assert "str(SCRIPT_PATH)" in source
    assert "remote_path=REMOTE_RUNNER" in source
    assert (
        'modal.Sandbox.create(\n            "sleep",\n            "infinity"' in source
    )
    assert "sandbox.exec(" in local_entrypoint
    assert '"CPPMEGA_GATE_RUN_ID": run_id' in local_entrypoint
    assert '"CPPMEGA_MODAL_SANDBOX_ID": sandbox_id' in local_entrypoint
    assert '"CPPMEGA_SANDBOX_WORKER": "1"' in local_entrypoint
    assert (
        'modal.is_local() and os.environ.get("CPPMEGA_SANDBOX_WORKER") != "1"' in source
    )
    assert 'paths = _artifact_paths(pathlib.Path("/tmp"), run_id)' in source
    assert "sandbox.filesystem.read_bytes(str(remote_path))" in local_entrypoint
    assert "sandbox.terminate(wait=True)" in local_entrypoint
    assert local_entrypoint.index(
        "sandbox.filesystem.read_bytes"
    ) < local_entrypoint.index("sandbox.terminate(wait=True)")
    assert local_entrypoint.index("result = json.loads") < local_entrypoint.index(
        'result.get("status") != "green"'
    )
