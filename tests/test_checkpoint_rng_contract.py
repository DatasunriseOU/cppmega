from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from cppmega.megatron.checkpoint_restore_preflight import (
    _megatron_cuda_rng_tracker_state,
)


def _install_fake_tracker(monkeypatch: pytest.MonkeyPatch, tracker: object) -> None:
    random_module = ModuleType("megatron.core.tensor_parallel.random")
    random_module.get_cuda_rng_tracker = lambda: tracker  # type: ignore[attr-defined]
    for name in ("megatron", "megatron.core", "megatron.core.tensor_parallel"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "megatron.core.tensor_parallel.random", random_module
    )


def test_cuda_rng_tracker_state_requires_named_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tracker(monkeypatch, SimpleNamespace(get_states=lambda: {}))

    with pytest.raises(RuntimeError, match="no named RNG states"):
        _megatron_cuda_rng_tracker_state()


def test_cuda_rng_tracker_state_returns_megatron_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"model-parallel-rng": b"state"}
    _install_fake_tracker(
        monkeypatch, SimpleNamespace(get_states=lambda: expected)
    )

    assert _megatron_cuda_rng_tracker_state() == expected
