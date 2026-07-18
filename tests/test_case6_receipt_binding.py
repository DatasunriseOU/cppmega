from __future__ import annotations

import copy

import pytest

from cppmega.receipt_binding import (
    IMPLEMENTATION_BINDING_SCHEMA,
    build_data_producer_binding,
    build_implementation_binding,
    build_receipt_binding,
    complete_training_implementation_binding,
    validate_implementation_binding,
    validate_receipt_binding,
)


def _binding() -> dict[str, object]:
    return build_receipt_binding(
        bundle_id="bundle-abc",
        artifact_set_sha256="a" * 64,
        prefix_manifest_sha256s={"data/seq_1024/train.json": "b" * 64},
        checkpoint_sha256="c" * 64,
        config={"profile": "h200_cpp_world_mini", "seq": 1024},
        command=["python", "pretrain.py", "--train-iters", "1"],
        run_id="case6-run-001",
        implementation=_implementation_binding(),
    )


def test_receipt_binding_covers_every_case6_identity_dimension() -> None:
    binding = _binding()

    assert binding["schema"] == "cppmega_case6_receipt_binding_v2"
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


def _implementation_binding() -> dict[str, object]:
    return build_implementation_binding(
        cppmega_commit="a" * 40,
        cppmega_tree_sha256="b" * 64,
        megatron_commit="c" * 40,
        cppmega_mlx_commit="d" * 40,
        cppmega_mlx_tree_sha256="e" * 64,
        clang_indexer_sha256="f" * 64,
        clang_indexer_dependency_closure_sha256="0" * 64,
    )


def test_implementation_binding_requires_both_repositories_and_indexer_closure() -> None:
    binding = _implementation_binding()

    assert binding["schema"] == IMPLEMENTATION_BINDING_SCHEMA
    assert set(binding["components"]) == {
        "cppmega",
        "megatron",
        "cppmega_mlx",
        "clang_indexer",
    }
    assert binding["components"]["clang_indexer"]["dependency_closure_sha256"] == (
        "0" * 64
    )


def test_data_producer_binding_is_a_strict_subset_for_bundle_manifests() -> None:
    binding = build_data_producer_binding(
        cppmega_commit="e" * 40,
        cppmega_tree_sha256="f" * 64,
        cppmega_mlx_commit="a" * 40,
        cppmega_mlx_tree_sha256="b" * 64,
        clang_indexer_sha256="c" * 64,
        clang_indexer_dependency_closure_sha256="d" * 64,
    )

    assert set(binding["components"]) == {
        "cppmega",
        "cppmega_mlx",
        "clang_indexer",
    }
    with pytest.raises(ValueError, match="missing components"):
        validate_implementation_binding(binding, where="full receipt")

    completed = complete_training_implementation_binding(
        binding,
        megatron_commit="9" * 40,
    )
    assert completed["components"]["megatron"] == {"commit": "9" * 40}
    validate_implementation_binding(completed, where="full receipt")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["components"].pop("cppmega_mlx"),
        lambda value: value["components"]["clang_indexer"].pop(
            "dependency_closure_sha256"
        ),
        lambda value: value["components"]["megatron"].update(commit="stale"),
    ],
)
def test_implementation_binding_rejects_missing_or_invalid_identity(mutation) -> None:
    stale = copy.deepcopy(_implementation_binding())
    mutation(stale)

    with pytest.raises(ValueError, match="implementation"):
        validate_implementation_binding(stale, where="receipt")
