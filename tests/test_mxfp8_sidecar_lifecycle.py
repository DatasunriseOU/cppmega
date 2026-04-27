from __future__ import annotations

class _Dummy:
    pass


def test_clear_mxfp8_sidecar_refs_removes_all_producer_references(monkeypatch):
    for key in (
        "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER",
        "CPPMEGA_TE_MXFP8_DGRAD_BF16",
        "CPPMEGA_TE_MXFP8_WGRAD_BF16",
        "NVTE_BACKWARD_OVERRIDE",
    ):
        monkeypatch.delenv(key, raising=False)
    from scripts.cppmega_fp8_shim import _cppmega_clear_mxfp8_sidecar_refs

    tensor = _Dummy()
    sidecar = _Dummy()

    tensor._te_rowwise_transpose_for_backward = sidecar
    tensor._te_rowwise_transpose_for_backward_unregister = lambda _x: None
    tensor._cppmega_mxfp8_rowwise_transpose = sidecar
    tensor._cppmega_mxfp8_rowwise_transpose_unregister = lambda _x: None
    tensor._cppmega_mxfp8_rowwise_transpose_persistent = False

    assert _cppmega_clear_mxfp8_sidecar_refs(tensor)
    assert not hasattr(tensor, "_te_rowwise_transpose_for_backward")
    assert not hasattr(tensor, "_te_rowwise_transpose_for_backward_unregister")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose_unregister")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose_persistent")

    assert not _cppmega_clear_mxfp8_sidecar_refs(tensor)
