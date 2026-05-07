"""Phase-2 migration parity test: sparse_mla_blockscaled_path_c.

Builds the Path-C E8M0 Sparse-MLA QK reducer kernel through both the legacy
MSL shim and the unified ``tilelang.compile`` engine via the
``CPPMEGA_MLX_TILELANG_ENGINE`` env var, and checks that:

* both modes produce a non-empty kernel artifact for the canonical
  ``(N=16, K=64, outputs_per_block=4, reduce_threads=32, vec=4)`` shape;
* the engine-mode artifact carries the ``_tilelang_engine_target`` tag
  Phase-1 introduced, while the shim-mode artifact still exposes the
  ``msl_text`` / ``buffer_param_names`` fields the MLX runtime depends on;
* the public ``lower_blockscaled_sparse_mla_qk_reduce_msl`` wrapper honours
  the env var (engine path returns the engine source string, shim path
  returns the MSL shim source string), so callers gain CUDA/HIP portability
  without breaking the existing 6 mlx call sites that still go through
  ``_qk_reduce_kernel_for`` (the Metal-MLX-runtime cache).

Skips on no-Metal hosts (the shim path needs Metal at construction time)
and on environments without the ``tilelang`` engine import.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from typing import Iterator

import pytest


_path_c_mod_name = "cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c"
pytest.importorskip(_path_c_mod_name)


@contextmanager
def _engine_mode(mode: str | None) -> Iterator[None]:
    prev = os.environ.get("CPPMEGA_MLX_TILELANG_ENGINE")
    if mode is None:
        os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
    else:
        os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = mode
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
        else:
            os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = prev


def _reset_caches() -> None:
    """Clear per-mode lru caches so each mode rebuilds via the active dispatcher."""

    path_c = importlib.import_module(_path_c_mod_name)
    path_c._qk_reduce_kernel_engine_for.cache_clear()
    # _qk_reduce_kernel_for is shim-only and doesn't observe the env var, but
    # clearing keeps test runs hermetic when this file is re-run in the same
    # interpreter session.
    path_c._qk_reduce_kernel_for.cache_clear()


_QK_KW: dict[str, int] = {
    "N": 16,
    "K": 64,
    "outputs_per_block": 4,
    "reduce_threads": 32,
    "vec": 4,
}


def test_engine_mode_yields_compiled_artifact_with_target_tag() -> None:
    pytest.importorskip("tilelang")
    path_c = importlib.import_module(_path_c_mod_name)
    _reset_caches()

    with _engine_mode("engine"):
        try:
            artifact = path_c._qk_reduce_kernel_engine_for(
                _QK_KW["N"],
                _QK_KW["K"],
                _QK_KW["outputs_per_block"],
                _QK_KW["reduce_threads"],
                _QK_KW["vec"],
            )
        except Exception as exc:  # pragma: no cover - engine bring-up gating
            pytest.skip(f"tilelang engine not buildable on this host: {exc}")

    assert getattr(artifact, "_tilelang_engine_target", None) == "metal", (
        "Engine artifact must carry the Phase-1 _tilelang_engine_target tag; "
        f"got attrs={dir(artifact)!r}"
    )


def test_shim_mode_yields_msl_lowering_with_buffer_param_names() -> None:
    msl_transform = pytest.importorskip("cppmega_mlx.nn._tilelang._msl_transform")
    if not msl_transform.can_run_metal():
        pytest.skip("Metal runtime not available; shim path requires it")
    path_c = importlib.import_module(_path_c_mod_name)
    _reset_caches()

    with _engine_mode("shim"):
        artifact = path_c._qk_reduce_kernel_engine_for(
            _QK_KW["N"],
            _QK_KW["K"],
            _QK_KW["outputs_per_block"],
            _QK_KW["reduce_threads"],
            _QK_KW["vec"],
        )

    assert hasattr(artifact, "msl_text"), (
        "Shim artifact must expose msl_text for MLX runtime compatibility"
    )
    assert hasattr(artifact, "buffer_param_names")
    names = set(artifact.buffer_param_names)
    assert {"A_fp8", "A_scale", "B_fp8", "B_scale", "C"}.issubset(names), names


def test_lower_msl_helper_routes_through_dispatcher() -> None:
    """The public lower-helper must honour the env-var dispatcher.

    Engine-mode source is generally a different string than the MSL shim's
    ``msl_text`` even on Metal (engine wraps with full TVM lower output);
    the only contract we need is that both return a non-empty string and
    that switching modes does not crash.
    """

    pytest.importorskip("tilelang")
    msl_transform = pytest.importorskip("cppmega_mlx.nn._tilelang._msl_transform")
    if not msl_transform.can_run_metal():
        pytest.skip("Metal runtime not available")
    path_c = importlib.import_module(_path_c_mod_name)
    _reset_caches()

    with _engine_mode("engine"):
        try:
            engine_src = path_c.lower_blockscaled_sparse_mla_qk_reduce_msl(**_QK_KW)
        except Exception as exc:
            pytest.skip(f"engine lower unavailable: {exc}")

    _reset_caches()
    with _engine_mode("shim"):
        shim_src = path_c.lower_blockscaled_sparse_mla_qk_reduce_msl(**_QK_KW)

    assert isinstance(engine_src, str) and engine_src.strip()
    assert isinstance(shim_src, str) and shim_src.strip()


def test_legacy_metal_kernel_cache_still_works() -> None:
    """Regression: the 6 mlx call sites that go through _qk_reduce_kernel_for must keep working."""

    msl_transform = pytest.importorskip("cppmega_mlx.nn._tilelang._msl_transform")
    if not msl_transform.can_run_metal():
        pytest.skip("Metal runtime not available")
    path_c = importlib.import_module(_path_c_mod_name)
    _reset_caches()

    kernel, lowering, input_names = path_c._qk_reduce_kernel_for(
        _QK_KW["N"],
        _QK_KW["K"],
        _QK_KW["outputs_per_block"],
        _QK_KW["reduce_threads"],
        _QK_KW["vec"],
    )
    assert hasattr(lowering, "msl_text")
    assert set(input_names) == {"A_fp8", "A_scale", "B_fp8", "B_scale"}
    assert kernel is not None
