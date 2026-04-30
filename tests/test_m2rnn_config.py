from types import SimpleNamespace

from cppmega.features.m2rnn import CppMegaM2RNNConfig, build_cppmega_m2rnn_config


_RUNTIME_ENV = [
    "CPPMEGA_M2RNN_KERNEL",
    "CPPMEGA_M2RNN_SAVE_HNEW",
    "CPPMEGA_M2RNN_BWD_CHUNK_SIZE",
    "CPPMEGA_M2RNN_FWD_AUTOTUNE",
    "CPPMEGA_M2RNN_FWD_NUM_WARPS",
    "CPPMEGA_M2RNN_FWD_NUM_STAGES",
    "CPPMEGA_M2RNN_BROADCAST_VIEWS",
    "CPPMEGA_M2RNN_BWD_REDUCE_BROADCAST_QK",
]


def _clear_runtime_env(monkeypatch):
    for name in _RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)


def test_build_cppmega_m2rnn_config_maps_expected_fields(monkeypatch):
    _clear_runtime_env(monkeypatch)
    megatron_config = SimpleNamespace(
        hidden_size=512,
        m2rnn_k_head_dim=32,
        m2rnn_v_head_dim=8,
        m2rnn_conv_kernel=5,
        m2rnn_gradient_clipping=0.5,
        m2rnn_use_xma=False,
    )

    assert build_cppmega_m2rnn_config(megatron_config) == CppMegaM2RNNConfig(
        d_model=512,
        k_head_dim=32,
        v_head_dim=8,
        conv_kernel=5,
        gradient_clipping=0.5,
        use_xma=False,
    )


def test_build_cppmega_m2rnn_config_maps_runtime_fields(monkeypatch):
    _clear_runtime_env(monkeypatch)
    megatron_config = SimpleNamespace(
        hidden_size=512,
        m2rnn_kernel="torch",
        m2rnn_save_hnew=True,
        m2rnn_bwd_chunk_size=16,
        m2rnn_fwd_autotune=True,
        m2rnn_fwd_num_warps=8,
        m2rnn_fwd_num_stages=2,
        m2rnn_broadcast_views=False,
        m2rnn_bwd_reduce_broadcast_qk=False,
    )

    config = build_cppmega_m2rnn_config(megatron_config)

    assert config.runtime_kernel == "torch"
    assert config.runtime_save_hnew is True
    assert config.runtime_bwd_chunk_size == 16
    assert config.runtime_fwd_autotune is True
    assert config.runtime_fwd_num_warps == 8
    assert config.runtime_fwd_num_stages == 2
    assert config.runtime_broadcast_views is False
    assert config.runtime_bwd_reduce_broadcast_qk is False


def test_build_cppmega_m2rnn_config_maps_env_runtime_fallback(monkeypatch):
    monkeypatch.setenv("CPPMEGA_M2RNN_KERNEL", "torch")
    monkeypatch.setenv("CPPMEGA_M2RNN_SAVE_HNEW", "1")
    monkeypatch.setenv("CPPMEGA_M2RNN_BWD_CHUNK_SIZE", "32")
    monkeypatch.setenv("CPPMEGA_M2RNN_FWD_AUTOTUNE", "1")
    monkeypatch.setenv("CPPMEGA_M2RNN_FWD_NUM_WARPS", "16")
    monkeypatch.setenv("CPPMEGA_M2RNN_FWD_NUM_STAGES", "4")
    monkeypatch.setenv("CPPMEGA_M2RNN_BROADCAST_VIEWS", "0")
    monkeypatch.setenv("CPPMEGA_M2RNN_BWD_REDUCE_BROADCAST_QK", "0")

    config = build_cppmega_m2rnn_config(SimpleNamespace(hidden_size=512))

    assert config.runtime_kernel == "torch"
    assert config.runtime_save_hnew is True
    assert config.runtime_bwd_chunk_size == 32
    assert config.runtime_fwd_autotune is True
    assert config.runtime_fwd_num_warps == 16
    assert config.runtime_fwd_num_stages == 4
    assert config.runtime_broadcast_views is False
    assert config.runtime_bwd_reduce_broadcast_qk is False
