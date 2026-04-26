from __future__ import annotations

import pytest


def test_runtime_config_honors_env_changes_after_import(monkeypatch):
    import cppmega.megatron.m2rnn_triton as _mod

    monkeypatch.delenv("CPPMEGA_M2RNN_SAVE_HNEW", raising=False)
    monkeypatch.delenv("CPPMEGA_M2RNN_BWD_CHUNK_SIZE", raising=False)
    _mod.reset_m2rnn_runtime_config_cache()

    default_config = _mod.get_m2rnn_runtime_config()
    assert default_config.save_hnew is False
    assert default_config.bwd_chunk_size == 64

    monkeypatch.setenv("CPPMEGA_M2RNN_SAVE_HNEW", "1")
    monkeypatch.setenv("CPPMEGA_M2RNN_BWD_CHUNK_SIZE", "8")

    updated_config = _mod.get_m2rnn_runtime_config()
    assert updated_config.save_hnew is True
    assert updated_config.bwd_chunk_size == 8


def test_runtime_config_cache_can_be_reset(monkeypatch):
    import cppmega.megatron.m2rnn_triton as _mod

    monkeypatch.setenv("CPPMEGA_M2RNN_SAVE_HNEW", "1")
    monkeypatch.setenv("CPPMEGA_M2RNN_BWD_CHUNK_SIZE", "16")
    _mod.reset_m2rnn_runtime_config_cache()

    cached_config = _mod.get_m2rnn_runtime_config()
    assert _mod.get_m2rnn_runtime_config() is cached_config

    _mod.reset_m2rnn_runtime_config_cache()
    reset_config = _mod.get_m2rnn_runtime_config()
    assert reset_config == cached_config
    assert reset_config is not cached_config


@pytest.mark.parametrize(
    ("save_raw", "expected_save"),
    [
        (None, False),
        ("0", False),
        ("1", True),
        ("true", False),
    ],
)
@pytest.mark.parametrize(
    ("chunk_raw", "expected_chunk_size"),
    [
        (None, 64),
        ("bad", 64),
        ("0", 1),
        ("-7", 1),
        ("32", 32),
    ],
)
def test_runtime_config_parses_legacy_env_values(
    monkeypatch,
    save_raw,
    expected_save,
    chunk_raw,
    expected_chunk_size,
):
    import cppmega.megatron.m2rnn_triton as _mod

    if save_raw is None:
        monkeypatch.delenv("CPPMEGA_M2RNN_SAVE_HNEW", raising=False)
    else:
        monkeypatch.setenv("CPPMEGA_M2RNN_SAVE_HNEW", save_raw)
    if chunk_raw is None:
        monkeypatch.delenv("CPPMEGA_M2RNN_BWD_CHUNK_SIZE", raising=False)
    else:
        monkeypatch.setenv("CPPMEGA_M2RNN_BWD_CHUNK_SIZE", chunk_raw)

    _mod.reset_m2rnn_runtime_config_cache()
    config = _mod.get_m2rnn_runtime_config()
    assert config.save_hnew is expected_save
    assert config.bwd_chunk_size == expected_chunk_size
