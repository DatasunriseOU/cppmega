from __future__ import annotations

import json

import pytest

from scripts.h200_megatron_preflight import _load_backend_dispatch_receipt


def _binding() -> dict[str, object]:
    return {
        "schema": "cppmega_case6_receipt_binding_v1",
        "bundle_id": "bundle-1",
        "artifact_set_sha256": "a" * 64,
        "prefix_manifest_sha256s": {"data/train.json": "b" * 64},
        "checkpoint_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "command_sha256": "e" * 64,
        "run_id": "run-1",
    }


def _receipt() -> dict[str, object]:
    return {
        "schema": "cppmega_backend_dispatch_v1",
        "selected_backend": "tilelang",
        "forward": {"status": "passed", "finite": True},
        "backward": {"status": "passed", "finite": True},
        "numerical": {"status": "passed", "max_abs_error": 0.001},
        "binding": _binding(),
    }


def test_backend_dispatch_receipt_is_required_and_run_bound(tmp_path) -> None:
    path = tmp_path / "dispatch.json"
    path.write_text(json.dumps(_receipt()), encoding="utf-8")

    receipt = _load_backend_dispatch_receipt(
        path, claims=("tilelang",), receipt_binding=_binding()
    )

    assert receipt["selected_backend"] == "tilelang"
    stale = _binding()
    stale["run_id"] = "run-2"
    with pytest.raises(RuntimeError, match="run_id"):
        _load_backend_dispatch_receipt(
            path, claims=("tilelang",), receipt_binding=stale
        )


def test_backend_dispatch_receipt_cannot_be_omitted(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="was not written"):
        _load_backend_dispatch_receipt(
            tmp_path / "missing.json",
            claims=("tilelang",),
            receipt_binding=_binding(),
        )
