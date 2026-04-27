from __future__ import annotations

import pytest


def test_runtime_config_honors_env_changes_after_import(monkeypatch):
    import cppmega.megatron.m2rnn_triton as _mod

    monkeypatch.delenv("CPPMEGA_M2RNN_SAVE_HNEW", raising=False)
    monkeypatch.delenv("CPPMEGA_M2RNN_BWD_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CPPMEGA_M2RNN_FWD_AUTOTUNE", raising=False)
    monkeypatch.delenv("CPPMEGA_M2RNN_FWD_NUM_WARPS", raising=False)
    monkeypatch.delenv("CPPMEGA_M2RNN_FWD_NUM_STAGES", raising=False)
    _mod.reset_m2rnn_runtime_config_cache()

    default_config = _mod.get_m2rnn_runtime_config()
    assert default_config.save_hnew is False
    assert default_config.bwd_chunk_size == 64
    assert default_config.fwd_autotune is False
    assert default_config.fwd_num_warps == 4
    assert default_config.fwd_num_stages == 3

    monkeypatch.setenv("CPPMEGA_M2RNN_SAVE_HNEW", "1")
    monkeypatch.setenv("CPPMEGA_M2RNN_BWD_CHUNK_SIZE", "8")
    monkeypatch.setenv("CPPMEGA_M2RNN_FWD_AUTOTUNE", "1")
    monkeypatch.setenv("CPPMEGA_M2RNN_FWD_NUM_WARPS", "8")
    monkeypatch.setenv("CPPMEGA_M2RNN_FWD_NUM_STAGES", "2")

    updated_config = _mod.get_m2rnn_runtime_config()
    assert updated_config.save_hnew is True
    assert updated_config.bwd_chunk_size == 8
    assert updated_config.fwd_autotune is True
    assert updated_config.fwd_num_warps == 8
    assert updated_config.fwd_num_stages == 2


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


@pytest.mark.parametrize(
    ("autotune_raw", "expected_autotune"),
    [
        (None, False),
        ("0", False),
        ("1", True),
        ("true", False),
    ],
)
@pytest.mark.parametrize(
    ("warps_raw", "expected_warps"),
    [
        (None, 4),
        ("bad", 4),
        ("3", 4),
        ("1", 1),
        ("16", 16),
    ],
)
@pytest.mark.parametrize(
    ("stages_raw", "expected_stages"),
    [
        (None, 3),
        ("bad", 3),
        ("0", 3),
        ("1", 1),
        ("4", 4),
    ],
)
def test_runtime_config_parses_forward_launch_env_values(
    monkeypatch,
    autotune_raw,
    expected_autotune,
    warps_raw,
    expected_warps,
    stages_raw,
    expected_stages,
):
    import cppmega.megatron.m2rnn_triton as _mod

    if autotune_raw is None:
        monkeypatch.delenv("CPPMEGA_M2RNN_FWD_AUTOTUNE", raising=False)
    else:
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_AUTOTUNE", autotune_raw)
    if warps_raw is None:
        monkeypatch.delenv("CPPMEGA_M2RNN_FWD_NUM_WARPS", raising=False)
    else:
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_NUM_WARPS", warps_raw)
    if stages_raw is None:
        monkeypatch.delenv("CPPMEGA_M2RNN_FWD_NUM_STAGES", raising=False)
    else:
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_NUM_STAGES", stages_raw)

    _mod.reset_m2rnn_runtime_config_cache()
    config = _mod.get_m2rnn_runtime_config()
    assert config.fwd_autotune is expected_autotune
    assert config.fwd_num_warps == expected_warps
    assert config.fwd_num_stages == expected_stages
