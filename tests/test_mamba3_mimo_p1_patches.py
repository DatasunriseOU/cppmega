from __future__ import annotations

import pytest

from cppmega.megatron.upstream_patches import apply_mamba3_mimo_p1_patches as p1


def test_apply_if_requested_noops_when_primary_gate_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_apply_all() -> None:
        nonlocal called
        called = True

    monkeypatch.delenv("CPPMEGA_MAMBA3_P1", raising=False)
    monkeypatch.setattr(p1, "apply_all", fake_apply_all)

    assert p1.apply_if_requested() is False
    assert called is False


def test_apply_all_refuses_without_file_mutation_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CPPMEGA_MAMBA3_P1", "1")
    monkeypatch.delenv("MAMBA3_P1_ALLOW_FILE_MUTATION", raising=False)
    monkeypatch.setattr(p1, "_do_patch", lambda: None)

    with pytest.raises(RuntimeError, match="Refusing to mutate"):
        p1.apply_all()


def test_apply_all_runs_with_both_mutation_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_do_patch() -> None:
        nonlocal called
        called = True

    monkeypatch.setenv("CPPMEGA_MAMBA3_P1", "1")
    monkeypatch.setenv("MAMBA3_P1_ALLOW_FILE_MUTATION", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(p1, "_do_patch", fake_do_patch)

    p1.apply_all()

    assert called is True
