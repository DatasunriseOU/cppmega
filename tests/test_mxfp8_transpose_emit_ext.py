from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def ext() -> Any:
    ext_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "probes"
        / "te_mxfp8_transpose_emit_ext.py"
    )
    spec = importlib.util.spec_from_file_location("test_mxfp8_transpose_emit_ext", ext_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _u8_arange(shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    return (torch.arange(numel, device=device, dtype=torch.int64) % 256).to(torch.uint8).view(shape)


def _gemm_swizzled_scale_idx(row: int, k_block: int, num_tiles_x: int) -> int:
    tile_idx_x = k_block // 4
    tile_idx_y = row // 128
    idx_in_tile_x = k_block % 4
    idx_in_tile_y = row % 128
    return (tile_idx_y * num_tiles_x + tile_idx_x) * 512 + (
        idx_in_tile_y % 32
    ) * 16 + (idx_in_tile_y // 32) * 4 + idx_in_tile_x


def _expected_swizzled_scale(
    compact: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    k_blocks = (cols + 31) // 32
    padded_rows = ((rows + 127) // 128) * 128
    padded_k_blocks = ((k_blocks + 3) // 4) * 4
    num_tiles_x = (cols + 127) // 128
    expected = torch.zeros((padded_rows * padded_k_blocks,), dtype=torch.uint8)
    compact_cpu = compact.cpu()
    for row in range(rows):
        for k_block in range(k_blocks):
            expected[_gemm_swizzled_scale_idx(row, k_block, num_tiles_x)] = compact_cpu[
                row, k_block
            ]
    return expected


def test_transpose_payload_swizzle_scale_transposes_payload_and_scales(ext: Any) -> None:
    device = torch.device("cuda")
    rows, cols = 64, 32
    data = _u8_arange((rows, cols), device=device)
    columnwise_scale = _u8_arange((rows // 32, cols), device=device)

    rowwise_data, rowwise_scale = ext.transpose_payload_swizzle_scale(data, columnwise_scale)
    torch.cuda.synchronize()

    assert rowwise_data.shape == (cols, rows)
    torch.testing.assert_close(rowwise_data.T, data)

    assert rowwise_scale.shape == (128, 4)
    expected = torch.zeros((128 * 4,), dtype=torch.uint8)
    scale_cpu = columnwise_scale.cpu()
    num_tiles_x = (rows + 127) // 128
    for row in range(cols):
        for k_block in range(rows // 32):
            expected[_gemm_swizzled_scale_idx(row, k_block, num_tiles_x)] = scale_cpu[
                k_block, row
            ]
    torch.testing.assert_close(rowwise_scale.cpu().view(-1), expected)


def test_swizzle_rowwise_scale_matches_reference(ext: Any) -> None:
    device = torch.device("cuda")
    rows, cols = 64, 64
    compact = _u8_arange((rows, cols // 32), device=device)

    out = ext.swizzle_rowwise_scale(compact, rows, cols)
    torch.cuda.synchronize()

    assert out.shape == (128 * 4,)
    torch.testing.assert_close(out.cpu(), _expected_swizzled_scale(compact, rows, cols))


def test_swizzle_rowwise_scale_reuses_valid_out(ext: Any) -> None:
    device = torch.device("cuda")
    rows, cols = 64, 32
    compact = torch.zeros((rows, cols // 32), device=device, dtype=torch.uint8)
    out = torch.empty((128 * 4,), device=device, dtype=torch.uint8)

    result = ext.swizzle_rowwise_scale(compact, rows, cols, out=out)
    torch.cuda.synchronize()

    assert result is out


def test_swizzle_rowwise_scale_rejects_bad_out_shape(ext: Any) -> None:
    device = torch.device("cuda")
    compact = torch.zeros((64, 1), device=device, dtype=torch.uint8)
    bad_out = torch.empty((100,), device=device, dtype=torch.uint8)

    with pytest.raises(ValueError, match="out must have shape"):
        ext.swizzle_rowwise_scale(compact, 64, 32, out=bad_out)
