from __future__ import annotations

from tools.profiling.trace_gemm_sources import collect_gemm_groups, render_table


def test_collect_gemm_groups_links_kernel_to_cpu_op_by_external_id() -> None:
    trace = {
        "traceEvents": [
            {
                "cat": "cpu_op",
                "name": "aten::mm",
                "dur": 40,
                "args": {
                    "External id": 1,
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                    "Input Dims": [[1024, 3584], [3584, 65536]],
                },
            },
            {
                "cat": "kernel",
                "name": "nvjet_sm121_tst_mma_192x144",
                "dur": 100,
                "args": {"External id": 1},
            },
            {
                "cat": "cpu_op",
                "name": "_LinearBackward",
                "dur": 80,
                "args": {
                    "External id": 2,
                    "Input type": ["c10::BFloat16"],
                    "Input Dims": [[4096, 4, 3584]],
                },
            },
            {
                "cat": "kernel",
                "name": "void cutlass::Kernel2<cutlass::gemm::kernel::DeviceGemmMxfp8GemmSm120>",
                "dur": 200,
                "args": {"External id": 2},
            },
        ]
    }

    groups = collect_gemm_groups(trace)

    assert [(group.kind, group.cpu_op, group.calls, group.total_us) for group in groups] == [
        ("flashinfer_mxfp8_cutlass", "_LinearBackward", 1, 200.0),
        ("nvjet_sm121_tc", "aten::mm", 1, 100.0),
    ]
    assert groups[1].input_dims == ((1024, 3584), (3584, 65536))


def test_render_table_contains_stable_columns() -> None:
    groups = collect_gemm_groups(
        {
            "traceEvents": [
                {"cat": "cpu_op", "name": "aten::addmm", "args": {"External id": "7"}},
                {
                    "cat": "kernel",
                    "name": "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm>",
                    "dur": 1234,
                    "args": {"External id": "7"},
                },
            ]
        }
    )

    table = render_table(groups, limit=10)

    assert "total_ms" in table
    assert "cutlass_bf16_tc" in table
    assert "aten::addmm" in table
