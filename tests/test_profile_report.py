from pathlib import Path

from tools.profiling.profile_report import build_profile_report, render_table


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_profile_report_summarizes_training_and_profiler_artifacts(tmp_path):
    log = _write(
        tmp_path / "run_baseline_20260429.log",
        """
  seq_length ...................................... 1024
[mem_profile] step_1_post: alloc=5.000 GiB reserved=7.000 GiB max_alloc=8.000 GiB max_reserved=9.000 GiB
 [2026-04-29 00:00:01] iteration        1/       3 | consumed samples:            2 | elapsed time per iteration (ms): 4000.0 | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 3.000000E+00 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2026-04-29 00:00:02] iteration        2/       3 | consumed samples:            4 | elapsed time per iteration (ms): 1000.0 | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 2.000000E+00 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2026-04-29 00:00:03] iteration        3/       3 | consumed samples:            6 | elapsed time per iteration (ms): 500.0 | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 1.500000E+00 | number of skipped iterations:   0 | number of nan iterations:   0 |
 validation loss at iteration 3 on validation set | lm loss value: 1.250000E+00 |
 validation loss at iteration 3 on test set | lm loss value: 1.750000E+00 |
""",
    )

    torch_dir = tmp_path / "run_torchprof_20260429_torch_profile"
    torch_dir.mkdir()
    _write(
        torch_dir / "train_step_2_cuda_table.txt",
        """
Name  Self CPU %  Self CPU  CPU total %  CPU total  CPU time avg  Self CUDA  Self CUDA %  CUDA total  CUDA time avg  CPU Mem  Self CPU Mem  CUDA Mem  Self CUDA Mem  # of Calls
aten::addmm  0.00%  1.000ms  0.00%  1.000ms  1.000ms  2.000ms  40.00%  3.000ms  1.500ms  0 B  0 B  0 B  0 B  2
_cce_backward_kernel  0.00%  0.000us  0.00%  0.000us  0.000us  1.000ms  20.00%  1.000ms  1.000ms  0 B  0 B  0 B  0 B  1
""",
    )
    _write(
        tmp_path / "run_nsys_20260429_cuda_gpu_kern_sum.txt",
        """
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)  Name
     23.5      12000000000        300   40000000.0   39000000.0  1  2  3  _cce_backward_kernel
      5.0       2500000000          3  833333333.3  830000000.0  1  2  3  _cce_lse_forward_kernel
""",
    )
    _write(
        tmp_path / "run_nsys_20260429_cuda_gpu_mem_sum.txt",
        """
 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)   Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
    1296.500      42    30.869     6.000     0.000   100.000        1.000  [CUDA memcpy Device-to-Device]
      12.000       3     4.000     4.000     4.000     4.000        0.000  [CUDA memcpy Host-to-Device]
""",
    )
    _write(
        tmp_path / "run_ncu_20260429_details.txt",
        """
[123] python
  _cce_lse_forward_kernel (1, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.1
    Duration                         ms        78.47
    Memory Throughput                 %        50.57
    Compute (SM) Throughput           %        81.99
  cast_transpose_optimized_kernel(float const*) (2, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.1
    Duration                         us       748.77
    Memory Throughput                 %        10.59
    Compute (SM) Throughput           %         3.76
""",
    )

    report = build_profile_report(log, hot_step_start=2)

    assert report.training.hot_step_avg_ms == 750.0
    assert report.training.tok_per_sec == 3072.0
    assert report.training.final_train_loss == 1.5
    assert report.training.final_val_loss == 1.25
    assert report.training.final_test_loss == 1.75
    assert report.training.max_alloc_gib == 8.0
    assert report.torch_ops[0].name == "aten::addmm"
    assert report.nsys_kernels[0].name == "_cce_backward_kernel"
    assert report.memops[0].operation == "[CUDA memcpy Device-to-Device]"
    assert report.memops[0].total_bytes == 1_296_500_000
    assert report.ncu_kernels[0].name == "_cce_lse_forward_kernel"

    table = render_table(report, top_n=2)
    assert "steady avg ms" in table
    assert "D2D copy bytes" in table
    assert "_cce_backward_kernel" in table
    assert "ncu sampled kernels" in table
