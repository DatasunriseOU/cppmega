from pathlib import Path

from tools.profiling.compare_bf16_mxfp8 import (
    LogInput,
    build_comparison_report,
    render_table,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_compare_bf16_mxfp8_parses_core_metrics_and_artifacts(tmp_path):
    bf16_trace = tmp_path / "bf16_torch_profile" / "train_step_1.json"
    bf16_table = tmp_path / "bf16_torch_profile" / "train_step_1_cuda_table.txt"
    bf16_trace.parent.mkdir()
    bf16_trace.write_text("{}", encoding="utf-8")
    bf16_table.write_text("table", encoding="utf-8")
    mxfp8_ncu = tmp_path / "mxfp8.ncu-rep"
    mxfp8_ncu.write_text("ncu", encoding="utf-8")

    bf16_log = _write(
        tmp_path / "bf16.log",
        f"""
  seq_length ...................................... 1024
[mem_profile] after_setup_model_and_optimizer: alloc=3.000 GiB reserved=4.000 GiB max_alloc=5.000 GiB max_reserved=6.000 GiB
[mem_profile] total_params=2,000 param_bytes=0.004 GiB
[mem_profile] by_storage:
[mem_profile]   torch.bfloat16              2,000 elems    0.004 GiB
[mem_profile] top_parameters:
[mem_profile] step_1_post: alloc=5.000 GiB reserved=7.000 GiB max_alloc=8.000 GiB max_reserved=9.000 GiB
 [2026-04-27 00:00:01] iteration        1/       3 | consumed samples:            2 | elapsed time per iteration (ms): 4000.0 | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 3.000000E+00 | mtp_1 loss: 4.000000E+00 | loss scale: 1.0 | grad norm: 1.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2026-04-27 00:00:02] iteration        2/       3 | consumed samples:            4 | elapsed time per iteration (ms): 1000.0 | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 2.000000E+00 | mtp_1 loss: 3.000000E+00 | loss scale: 1.0 | grad norm: 1.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2026-04-27 00:00:03] iteration        3/       3 | consumed samples:            6 | elapsed time per iteration (ms): 500.0 | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 1.500000E+00 | mtp_1 loss: 2.500000E+00 | loss scale: 1.0 | grad norm: 1.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 validation loss at iteration 3 on validation set | lm loss value: 1.250000E+00 |
 validation loss at iteration 3 on test set | lm loss value: 1.750000E+00 |
[torch_profile] step=1 trace={bf16_trace} table={bf16_table}
WARNING: CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
""",
    )
    mxfp8_log = _write(
        tmp_path / "mxfp8.log",
        f"""
  seq_length ...................................... 1024
[mem_profile] after_setup_model_and_optimizer: alloc=4.000 GiB reserved=5.000 GiB max_alloc=6.000 GiB max_reserved=7.000 GiB
[mem_profile] total_params=2,000 param_bytes=0.003 GiB
[mem_profile] by_storage:
[mem_profile]   MXFP8Tensor                 1,000 elems    0.001 GiB
[mem_profile]   torch.bfloat16              1,000 elems    0.002 GiB
[mem_profile] top_parameters:
[mem_profile] step_1_post: alloc=7.000 GiB reserved=8.000 GiB max_alloc=9.000 GiB max_reserved=10.000 GiB
 [2026-04-27 00:00:01] iteration        1/       3 | consumed samples:            2 | elapsed time per iteration (ms): 5000.0 | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 3.100000E+00 | mtp_1 loss: 4.100000E+00 | loss scale: 1.0 | grad norm: 1.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2026-04-27 00:00:02] iteration        2/       3 | consumed samples:            4 | elapsed time per iteration (ms): 2000.0 | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 2.100000E+00 | mtp_1 loss: 3.100000E+00 | loss scale: 1.0 | grad norm: 1.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
 [2026-04-27 00:00:03] iteration        3/       3 | consumed samples:            6 | elapsed time per iteration (ms): 1000.0 | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 1.600000E+00 | mtp_1 loss: 2.600000E+00 | loss scale: 1.0 | grad norm: 1.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
 validation loss at iteration 3 on validation set | lm loss value: 1.350000E+00 |
 validation loss at iteration 3 on test set | lm loss value: 1.850000E+00 |
==PROF== Report: {mxfp8_ncu}
""",
    )

    report = build_comparison_report(
        bf16=LogInput(label="bf16", log=bf16_log),
        mxfp8=LogInput(label="mxfp8", log=mxfp8_log),
        hot_step_start=2,
        hot_step_end=None,
    )

    assert report.bf16.hot_step_avg_ms == 750.0
    assert report.bf16.tok_per_sec == 3072.0
    assert report.mxfp8.hot_step_avg_ms == 1500.0
    assert report.mxfp8.tok_per_sec == 1536.0
    assert report.bf16.setup_alloc_gib == 3.0
    assert report.mxfp8.max_alloc_gib == 9.0
    assert report.mxfp8.param_gib_by_storage == {
        "MXFP8Tensor": 0.001,
        "torch.bfloat16": 0.002,
    }
    assert report.mxfp8.final_train_loss == 1.6
    assert report.mxfp8.final_val_loss == 1.35
    assert report.mxfp8.final_test_loss == 1.85
    assert report.mxfp8.skipped_iterations == 1
    assert "torch_trace" in report.bf16.artifact_paths
    assert "ncu_report" in report.mxfp8.artifact_paths
    assert any("CUPTI multiple subscribers" in warning for warning in report.bf16.warnings)

    table = render_table(report)
    assert "hot_step_avg_ms" in table
    assert "param bytes by storage" in table
    assert "CUPTI allows only one active subscriber" in table
