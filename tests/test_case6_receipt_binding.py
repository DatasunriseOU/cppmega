from __future__ import annotations

import copy

import pytest

from cppmega.receipt_binding import build_receipt_binding, validate_receipt_binding


def _binding() -> dict[str, object]:
    return build_receipt_binding(
        bundle_id="bundle-abc",
        artifact_set_sha256="a" * 64,
        prefix_manifest_sha256s={"data/seq_1024/train.json": "b" * 64},
        checkpoint_sha256="c" * 64,
        config={"profile": "h200_cpp_world_mini", "seq": 1024},
        command=["python", "pretrain.py", "--train-iters", "1"],
        run_id="case6-run-001",
    )


def test_receipt_binding_covers_every_case6_identity_dimension() -> None:
    binding = _binding()

    assert binding["schema"] == "cppmega_case6_receipt_binding_v1"
    assert binding["bundle_id"] == "bundle-abc"
    assert binding["artifact_set_sha256"] == "a" * 64
    assert binding["prefix_manifest_sha256s"] == {
        "data/seq_1024/train.json": "b" * 64
    }
    assert binding["checkpoint_sha256"] == "c" * 64
    assert binding["run_id"] == "case6-run-001"
    assert len(binding["config_sha256"]) == 64
    assert len(binding["command_sha256"]) == 64


@pytest.mark.parametrize(
    "field",
    [
        "bundle_id",
        "artifact_set_sha256",
        "prefix_manifest_sha256s",
        "checkpoint_sha256",
        "config_sha256",
        "command_sha256",
        "run_id",
    ],
)
def test_receipt_binding_rejects_stale_mismatch(field: str) -> None:
    expected = _binding()
    stale = copy.deepcopy(expected)
    if field == "prefix_manifest_sha256s":
        stale[field] = {"data/train.json": "f" * 64}
    elif field.endswith("_sha256"):
        stale[field] = "f" * 64
    else:
        stale[field] = "stale"

    with pytest.raises(RuntimeError, match=field):
        validate_receipt_binding(stale, expected=expected, where="batch receipt")
