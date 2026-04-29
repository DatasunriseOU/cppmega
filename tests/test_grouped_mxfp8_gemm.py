"""Tests for the grouped direct MXFP8 backend prototype."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cppmega.megatron import grouped_mxfp8_gemm as grouped


class _FakeMXFP8Tensor:
    pass


def test_supported_shape_matches_16_expert_contract() -> None:
    assert grouped.is_supported_shape(64, 32, 64, 16)
    assert not grouped.is_supported_shape(64, 32, 64, 8)
    assert not grouped.is_supported_shape(64, 31, 64, 16)
    assert not grouped.is_supported_shape(0, 32, 64, 16)


def test_resolve_beta_matches_cutlass_wrapper_contract() -> None:
    assert grouped._resolve_beta(None, accumulate=True) == 1.0
    assert grouped._resolve_beta(0.25, accumulate=True) == 0.25
    assert grouped._resolve_beta(None, accumulate=False) == 0.0
    with pytest.raises(ValueError, match="accumulate=True"):
        grouped._resolve_beta(1.0, accumulate=False)


def test_backend_report_names_direct_grouped_surface() -> None:
    report = grouped.backend_capability_report()

    assert report["num_experts"] == 16
    assert report["launches_per_call"] == 1
    assert "dgrad_nn_gemm" in report["apis"]
    assert "wgrad_nt_gemm" in report["apis"]
    assert "Python loop over experts" in report["avoids"]


def test_backend_source_does_not_use_transpose_bridge() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "cppmega" / "megatron" / "grouped_mxfp8_gemm.py",
        root / "cppmega" / "megatron" / "cuda_ext" / "grouped_mxfp8_gemm.cpp",
        root / "cppmega" / "megatron" / "cuda_ext" / "grouped_mxfp8_gemm.cu",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    assert "_cppmega_mxfp8_colwise_as_rowwise_transpose" not in text
    assert ".t(" not in text
    assert ".transpose(" not in text
    assert "expert_offsets[kNumExperts].item" not in text
    assert "m_splits.detach().cpu().tolist()" not in text
    assert "Python loop over experts" in text


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="grouped MXFP8 CUDA correctness tests require CUDA",
)


def _offsets(device: torch.device) -> torch.Tensor:
    # Includes non-32-aligned expert boundaries to exercise direct columnwise
    # scale indexing for wgrad.
    counts = [3, 0, 5, 1, 7, 2, 4, 6, 0, 8, 3, 1, 2, 5, 4, 6]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def _fp8_payload(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    values = torch.tensor([0, 48, 50, 54, 56, 58, 176, 178, 182, 184, 186], dtype=torch.uint8, device=device)
    idx = torch.randint(0, int(values.numel()), shape, device=device)
    return values[idx].contiguous()


def _as_float(payload: torch.Tensor) -> torch.Tensor:
    return payload.view(torch.float8_e4m3fn).float()


def _split_rows(data: torch.Tensor, counts: list[int]) -> list[torch.Tensor]:
    parts = []
    start = 0
    for count in counts:
        parts.append(data[start : start + count])
        start += count
    return parts


def _offset_values(counts: list[int]) -> list[int]:
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + int(count))
    return offsets


def _wrap(
    *,
    rowwise_data: torch.Tensor | None = None,
    rowwise_scale: torch.Tensor | None = None,
    columnwise_data: torch.Tensor | None = None,
    columnwise_scale: torch.Tensor | None = None,
) -> _FakeMXFP8Tensor:
    tensor = _FakeMXFP8Tensor()
    if rowwise_data is not None:
        tensor._rowwise_data = rowwise_data
    if rowwise_scale is not None:
        tensor._rowwise_scale_inv = rowwise_scale
    if columnwise_data is not None:
        tensor._columnwise_data = columnwise_data
    if columnwise_scale is not None:
        tensor._columnwise_scale_inv = columnwise_scale
    return tensor


def test_wgrad_list_binding_passes_total_rows_without_cuda_offset_item(monkeypatch) -> None:
    counts = [1] * 16
    n = 32
    k = 64
    dy = [torch.empty((count, n), dtype=torch.uint8) for count in counts]
    x = [torch.empty((count, k), dtype=torch.uint8) for count in counts]
    dy_scale = [torch.empty(((count + 31) // 32, n), dtype=torch.uint8) for count in counts]
    x_scale = [torch.empty(((count + 31) // 32, k), dtype=torch.uint8) for count in counts]
    out = [torch.empty((n, k), dtype=torch.bfloat16) for _ in counts]
    offsets = torch.tensor(_offset_values(counts), dtype=torch.int64)
    calls = []

    class FakeExt:
        def wgrad_nt_ptrs(self, *args):
            calls.append(args)

    monkeypatch.setattr(grouped, "_check_uint8_cuda_contiguous", lambda _tensor, _name: None)
    monkeypatch.setattr(grouped, "_load_cuda_ext", lambda: FakeExt())

    result = grouped.wgrad_nt_gemm_list(
        dy,
        dy_scale,
        x,
        x_scale,
        offsets,
        out=out,
    )

    assert all(actual is expected for actual, expected in zip(result, out))
    assert len(calls) == 1
    args = calls[0]
    assert args[9] == sum(counts)
    assert args[10] == n
    assert args[11] == k


@requires_cuda
def test_try_grouped_direct_cuda_m_splits_uses_shape_offsets_without_cpu_readback(monkeypatch) -> None:
    counts = [1] * 16
    total = sum(counts)
    n = 32
    k = 64
    m_splits = torch.tensor(counts, dtype=torch.int64, device="cuda")
    captured = {}
    sentinel = object()

    A = [
        _wrap(
            columnwise_data=torch.empty((n, k), dtype=torch.uint8),
            columnwise_scale=torch.empty((n // 32, k), dtype=torch.uint8),
        )
        for _ in counts
    ]
    B = [
        _wrap(
            rowwise_data=torch.empty((count, n), dtype=torch.uint8),
            rowwise_scale=torch.empty((count, n // 32), dtype=torch.uint8),
        )
        for count in counts
    ]
    out = [torch.empty((total, k), dtype=torch.bfloat16)]

    def fail_cpu(self):
        raise AssertionError("direct grouped path read CUDA m_splits on CPU")

    def fake_dgrad_nn_gemm_list(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(torch.Tensor, "cpu", fail_cpu)
    monkeypatch.setattr(grouped, "dgrad_nn_gemm_list", fake_dgrad_nn_gemm_list)

    ok, result = grouped.try_grouped_direct(
        A,
        B,
        out,
        layout="NN",
        grad=True,
        m_splits=m_splits,
        out_dtype=torch.bfloat16,
    )

    assert ok, result
    assert result is sentinel
    assert captured["args"][4] == _offset_values(counts)
    assert captured["kwargs"]["out"] is out[0]


@requires_cuda
def test_grouped_dgrad_nn_matches_materialized_reference() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    offsets = _offsets(device)
    total = int(offsets[-1].item())
    n = 32
    k = 64

    dy = _fp8_payload((total, n), device)
    weight = _fp8_payload((16, n, k), device)
    dy_scale = torch.full((total, n // 32), 127, dtype=torch.uint8, device=device)
    weight_scale = torch.full((16, n // 32, k), 127, dtype=torch.uint8, device=device)

    out = grouped.dgrad_nn_gemm(dy, dy_scale, weight, weight_scale, offsets)
    torch.cuda.synchronize()

    dy_f = _as_float(dy)
    weight_f = _as_float(weight)
    expected = torch.empty((total, k), dtype=torch.float32, device=device)
    offsets_cpu = offsets.cpu().tolist()
    for expert in range(16):
        start, end = offsets_cpu[expert], offsets_cpu[expert + 1]
        if start != end:
            expected[start:end] = dy_f[start:end] @ weight_f[expert]

    torch.testing.assert_close(out.float(), expected.to(torch.bfloat16).float(), rtol=0.02, atol=0.1)


@requires_cuda
def test_grouped_wgrad_nt_matches_materialized_reference() -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    offsets = _offsets(device)
    total = int(offsets[-1].item())
    n = 32
    k = 64
    row_blocks = (total + 31) // 32

    dy = _fp8_payload((total, n), device)
    x = _fp8_payload((total, k), device)
    dy_scale = torch.full((row_blocks, n), 127, dtype=torch.uint8, device=device)
    x_scale = torch.full((row_blocks, k), 127, dtype=torch.uint8, device=device)

    out = grouped.wgrad_nt_gemm(dy, dy_scale, x, x_scale, offsets)
    torch.cuda.synchronize()

    dy_f = _as_float(dy)
    x_f = _as_float(x)
    expected = torch.empty((16, n, k), dtype=torch.float32, device=device)
    offsets_cpu = offsets.cpu().tolist()
    for expert in range(16):
        start, end = offsets_cpu[expert], offsets_cpu[expert + 1]
        if start == end:
            expected[expert].zero_()
        else:
            expected[expert] = dy_f[start:end].mT @ x_f[start:end]

    torch.testing.assert_close(out.float(), expected.to(torch.bfloat16).float(), rtol=0.02, atol=0.1)


@requires_cuda
def test_try_grouped_direct_accepts_split_local_wgrad_scales() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    counts = [3, 0, 5, 1, 7, 2, 4, 6, 0, 8, 3, 1, 2, 5, 4, 6]
    total = sum(counts)
    n = 32
    k = 64

    dy = _fp8_payload((total, n), device)
    x = _fp8_payload((total, k), device)
    dy_scales = torch.full(
        (sum((count + 31) // 32 for count in counts), n),
        127,
        dtype=torch.uint8,
        device=device,
    )
    x_scales = torch.full(
        (sum((count + 31) // 32 for count in counts), k),
        127,
        dtype=torch.uint8,
        device=device,
    )
    out_storage = torch.empty((16, n, k), dtype=torch.bfloat16, device=device)

    dy_parts = _split_rows(dy, counts)
    x_parts = _split_rows(x, counts)
    scale_counts = [(count + 31) // 32 for count in counts]
    dy_scale_parts = _split_rows(dy_scales, scale_counts)
    x_scale_parts = _split_rows(x_scales, scale_counts)
    A = [
        _wrap(columnwise_data=x_part, columnwise_scale=x_scale)
        for x_part, x_scale in zip(x_parts, x_scale_parts)
    ]
    B = [
        _wrap(columnwise_data=dy_part, columnwise_scale=dy_scale)
        for dy_part, dy_scale in zip(dy_parts, dy_scale_parts)
    ]
    out_parts = [out_storage[idx] for idx in range(16)]

    ok, result = grouped.try_grouped_direct(
        A,
        B,
        out_parts,
        layout="NT",
        grad=True,
        m_splits=counts,
        out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()

    assert ok, result
    assert result.data_ptr() == out_storage.data_ptr()
    dy_f = _as_float(dy)
    x_f = _as_float(x)
    expected = torch.empty((16, n, k), dtype=torch.float32, device=device)
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        if count == 0:
            expected[expert].zero_()
        else:
            expected[expert] = dy_f[start:end].mT @ x_f[start:end]
        start = end

    torch.testing.assert_close(out_storage.float(), expected.to(torch.bfloat16).float(), rtol=0.02, atol=0.1)
