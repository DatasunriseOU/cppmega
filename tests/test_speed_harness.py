from pathlib import Path

import pytest

from cppmega.recipes.run_profiles import get_run_profile
from tools.probes.speed_harness import (
    SpeedRunInput,
    build_run_inputs,
    build_speed_comparison_report,
    parse_nsys_kernel_csv,
    render_speed_table,
)
from tools.scripts import speed_compare


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _log(step2_ms: float, step3_ms: float, final_loss: float) -> str:
    return f"""
 [2026-04-28 00:00:01] iteration        1/       3 | consumed samples:            2 | elapsed time per iteration (ms): 4000.0 | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 3.000000E+00 | loss scale: 1.0 | grad norm: 1.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
Theoretical memory footprints not yet supported for hybrid Mamba-Transformer models.
[Rank 0] (after 1 iterations) memory (MB) | allocated: 1024.00 | max allocated: 2048.00 | reserved: 3072.00 | max reserved: 4096.00
 [2026-04-28 00:00:02] iteration        2/       3 | consumed samples:            4 | elapsed time per iteration (ms): {step2_ms:.1f} | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: 2.000000E+00 | loss scale: 1.0 | grad norm: 1.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2026-04-28 00:00:03] iteration        3/       3 | consumed samples:            6 | elapsed time per iteration (ms): {step3_ms:.1f} | learning rate: 1.000000E-04 | global batch size:     2 | lm loss: {final_loss:.6E} | loss scale: 1.0 | grad norm: 1.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 validation loss at iteration 3 on validation set | lm loss value: 1.250000E+00 |
 validation loss at iteration 3 on test set | lm loss value: 1.750000E+00 |
"""


def test_parse_nsys_kernel_csv_sorts_top_kernels(tmp_path):
    csv_path = _write(
        tmp_path / "kernels.csv",
        """Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
20.0,2000000000,2,1000000000,900000000,800000000,1200000000,1000,slow_kernel
30.0,3000000000,3,1000000000,1000000000,900000000,1100000000,1000,"fast, quoted kernel"
""",
    )

    summary = parse_nsys_kernel_csv(csv_path, top_n=1)

    assert summary.status == "ok"
    assert summary.row_count == 2
    assert summary.top_kernels[0].name == "fast, quoted kernel"
    assert summary.top_kernels[0].total_time_ms == 3000.0
    assert summary.top_kernels[0].instances == 3


def test_speed_comparison_reuses_profile_tokens_when_log_lacks_seq_length(tmp_path):
    baseline = _write(tmp_path / "baseline.log", _log(1000.0, 500.0, 1.5))
    candidate = _write(tmp_path / "candidate.log", _log(500.0, 500.0, 1.25))
    csv_path = _write(
        tmp_path / "candidate_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv",
        """Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
55.0,1100000000,11,100000000,100000000,90000000,120000000,1000,_cce_backward_kernel
""",
    )
    profile = get_run_profile("local_gb10_quarter")
    profile.training.seq_length = 1024
    profile.training.global_batch_size = 2

    report = build_speed_comparison_report(
        [
            SpeedRunInput(label="baseline", log=baseline),
            SpeedRunInput(label="candidate", log=candidate, nsys_kernel_csv=csv_path),
        ],
        profile=profile,
        hot_step_start=2,
    )

    assert report.profile_tokens_per_step == 2048
    assert report.runs[0].tokens_per_step == 2048
    assert report.runs[0].hot_step_avg_ms == 750.0
    assert report.runs[0].tok_per_sec == pytest.approx(2730.667, abs=0.001)
    assert report.deltas_vs_baseline["candidate"].tok_per_sec_pct == pytest.approx(50.0)
    assert report.runs[1].nsys.status == "ok"
    assert report.runs[1].nsys.top_kernels[0].name == "_cce_backward_kernel"

    table = render_speed_table(report)
    assert "candidate" in table
    assert "nsys top kernels: candidate" in table


def test_build_run_inputs_rejects_orphan_nsys_csv(tmp_path):
    log = _write(tmp_path / "run.log", _log(1000.0, 1000.0, 1.0))

    with pytest.raises(ValueError, match="unknown run label"):
        build_run_inputs([f"a={log}"], nsys_kernel_csv_values=[f"b={log}.csv"])


def test_speed_compare_cli_uses_typed_profile_validation(tmp_path):
    log = _write(tmp_path / "run.log", _log(1000.0, 1000.0, 1.0))

    with pytest.raises(SystemExit, match="nsys_capture_mode=delay requires"):
        speed_compare.main(
            [
                "--run",
                f"a={log}",
                "--nsys-capture-mode",
                "delay",
                "--nsys-duration",
                "-1",
            ]
        )
