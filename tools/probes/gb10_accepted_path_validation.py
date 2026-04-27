#!/usr/bin/env python3
"""Validate the accepted GB10 Mamba MXFP8 training path.

The accepted path is:

* MXFP8 backward uses the TN adapter.
* BF16/dequantized MXFP8 backward fallbacks stay disabled and unused.
* Deprecated BF16 bridge, DSA gather-scatter, and FP8 activation legacy packer
  fail closed unless their explicit ACK env vars are set.

This probe is intentionally orchestration-only.  It runs the existing
``te_blockscaled_backward_probe.py`` for the CUDA/TE MXFP8 path and contains
small parsers for the probe output and one-step training logs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

if __package__:
    from .gb10_accepted_path_validation_helpers import (
        ADAPTER_STAT_KEYS,
        CUTLASS_STAT_KEYS,
        FLASHINFER_STAT_KEYS,
        FALLBACK_STAT_KEYS,
        MATERIALIZATION_STAT_KEYS,
        SIDECAR_REGISTRY_ZERO_KEYS,
        extract_first_json_object,
        parse_training_log,
        validate_probe_report,
    )
else:
    _HELPER_PATH = Path(__file__).with_name("gb10_accepted_path_validation_helpers.py")
    _HELPER_SPEC = importlib.util.spec_from_file_location(
        "_gb10_accepted_path_validation_helpers", _HELPER_PATH
    )
    if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
        raise ImportError(f"could not load helper module at {_HELPER_PATH}")
    _helpers = importlib.util.module_from_spec(_HELPER_SPEC)
    _HELPER_SPEC.loader.exec_module(_helpers)
    ADAPTER_STAT_KEYS = _helpers.ADAPTER_STAT_KEYS
    CUTLASS_STAT_KEYS = _helpers.CUTLASS_STAT_KEYS
    FLASHINFER_STAT_KEYS = _helpers.FLASHINFER_STAT_KEYS
    FALLBACK_STAT_KEYS = _helpers.FALLBACK_STAT_KEYS
    MATERIALIZATION_STAT_KEYS = _helpers.MATERIALIZATION_STAT_KEYS
    SIDECAR_REGISTRY_ZERO_KEYS = _helpers.SIDECAR_REGISTRY_ZERO_KEYS
    extract_first_json_object = _helpers.extract_first_json_object
    parse_training_log = _helpers.parse_training_log
    validate_probe_report = _helpers.validate_probe_report


MXFP8_BF16_ACK = "CPPMEGA_I_UNDERSTAND_MXFP8_BF16_BACKWARD_BRIDGE_IS_DEPRECATED_AND_SLOW"
DSA_GATHER_ACK = "CPPMEGA_I_UNDERSTAND_DSA_GATHER_SCATTER_IS_DEPRECATED_AND_SLOW"
FP8_ACT_LEGACY_ACK = (
    "CPPMEGA_I_UNDERSTAND_FP8_ACTIVATION_LEGACY_BACKEND_IS_DEPRECATED_AND_SYNCY"
)


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    data: dict[str, Any] | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_env() -> dict[str, str]:
    root = repo_root()
    env = os.environ.copy()
    for key in (MXFP8_BF16_ACK, DSA_GATHER_ACK, FP8_ACT_LEGACY_ACK):
        env.pop(key, None)
    pythonpath = [
        str(root / "scripts"),
        str(root),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath if part)
    return env


def _run_python(
    code: str,
    *,
    extra_env: dict[str, str] | None = None,
    timeout_s: int = 30,
) -> subprocess.CompletedProcess[str]:
    env = _base_env()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root(),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )


def _combined_output(proc: subprocess.CompletedProcess[str]) -> str:
    return f"{proc.stdout}\n{proc.stderr}".strip()


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _expect_fail_closed(
    name: str,
    proc: subprocess.CompletedProcess[str],
    expected: tuple[str, ...],
) -> CheckResult:
    output = _combined_output(proc)
    if proc.returncode == 0:
        return CheckResult(
            name=name,
            status="fail",
            detail="deprecated path did not fail closed",
            data={"returncode": proc.returncode, "output_tail": _tail(output)},
        )
    missing = [needle for needle in expected if needle not in output]
    if missing:
        return CheckResult(
            name=name,
            status="fail",
            detail=f"deprecated path failed, but missing expected text: {missing}",
            data={"returncode": proc.returncode, "output_tail": _tail(output)},
        )
    return CheckResult(
        name=name,
        status="pass",
        detail="failed closed without ACK",
        data={"returncode": proc.returncode},
    )


def check_mxfp8_bf16_bridge_gate() -> CheckResult:
    proc = _run_python(
        "import cppmega_fp8_shim",
        extra_env={
            "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK": "1",
            "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER": "0",
            "CPPMEGA_TE_MXFP8_DGRAD_BF16": "0",
            "CPPMEGA_TE_MXFP8_WGRAD_BF16": "0",
            "NVTE_BACKWARD_OVERRIDE": "none",
        },
    )
    return _expect_fail_closed(
        "old_mxfp8_bf16_bridge_requires_ack",
        proc,
        ("Deprecated MXFP8 BF16 backward bridge requested", MXFP8_BF16_ACK),
    )


def check_dsa_gather_scatter_gate() -> CheckResult:
    # Stub the unrelated MTP native patch so this import reaches the DSA gate
    # even on minimal test environments without Megatron installed.
    proc = _run_python(
        "\n".join(
            [
                "import sys, types",
                "mod = types.ModuleType('cppmega.megatron.mtp_native_hopper_ce')",
                "mod.patch_mtp_native_hopper_ce = lambda: None",
                "sys.modules['cppmega.megatron.mtp_native_hopper_ce'] = mod",
                "import cppmega_fp8_shim",
            ]
        ),
        extra_env={
            "CPPMEGA_DSA_SPARSE_MODE": "gather_scatter",
            "CPPMEGA_MTP_CE_KERNEL": "native",
            "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER": "0",
            "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK": "0",
        },
    )
    return _expect_fail_closed(
        "dsa_gather_scatter_requires_ack",
        proc,
        ("CPPMEGA_DSA_SPARSE_MODE=gather_scatter is DEPRECATED", DSA_GATHER_ACK),
    )


def check_fp8_activation_legacy_gate() -> CheckResult:
    proc = _run_python(
        "\n".join(
            [
                "from cppmega.megatron import fp8_activations as fp8",
                "fp8._use_te_packer()",
            ]
        ),
        extra_env={"CPPMEGA_FP8_ACTIVATION_BACKEND": "legacy"},
    )
    return _expect_fail_closed(
        "fp8_activation_legacy_requires_ack",
        proc,
        ("CPPMEGA_FP8_ACTIVATION_BACKEND=legacy is DEPRECATED", FP8_ACT_LEGACY_ACK),
    )


def _probe_prereqs_available(timeout_s: int) -> tuple[bool, str]:
    proc = _run_python(
        "\n".join(
            [
                "import torch",
                "import transformer_engine",  # noqa: F401
                "import transformer_engine_torch",  # noqa: F401
                "print('cuda=' + str(torch.cuda.is_available()))",
            ]
        ),
        timeout_s=timeout_s,
    )
    if proc.returncode != 0:
        return False, _tail(_combined_output(proc), 1000)
    if "cuda=True" not in proc.stdout:
        return False, "torch.cuda.is_available() is false"
    return True, "CUDA and Transformer Engine are available"


def run_mxfp8_shim_probe(args: argparse.Namespace) -> CheckResult:
    if args.skip_mxfp8_probe:
        return CheckResult(
            name="mxfp8_tn_adapter_probe",
            status="skip",
            detail="skipped by --skip-mxfp8-probe",
        )

    available, reason = _probe_prereqs_available(args.probe_timeout_s)
    if not available and not args.require_mxfp8_probe:
        return CheckResult(
            name="mxfp8_tn_adapter_probe",
            status="skip",
            detail=f"CUDA/TE probe prerequisites unavailable: {reason}",
        )

    cmd = [
        sys.executable,
        str(repo_root() / "tools" / "probes" / "te_blockscaled_backward_probe.py"),
        "--format",
        "mxfp8",
        "--use-shim",
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--mxfp8-bwd-backend",
        "flashinfer_cutlass",
    ]
    env = _base_env()
    env.update(
        {
            "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER": "1",
            "CPPMEGA_TE_MXFP8_BWD_BACKEND": "flashinfer_cutlass",
            "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND": "te",
            "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED": "1",
            "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_STRICT": "1",
            "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK": "0",
            "CPPMEGA_TE_MXFP8_DGRAD_BF16": "0",
            "CPPMEGA_TE_MXFP8_WGRAD_BF16": "0",
            "NVTE_BACKWARD_OVERRIDE": "none",
            "CPPMEGA_TE_MXFP8_BWD_DEBUG": "1",
        }
    )
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        env=env,
        text=True,
        capture_output=True,
        timeout=args.probe_timeout_s,
        check=False,
    )
    output = _combined_output(proc)
    if proc.returncode != 0:
        return CheckResult(
            name="mxfp8_tn_adapter_probe",
            status="fail",
            detail="te_blockscaled_backward_probe.py failed",
            data={"returncode": proc.returncode, "output_tail": _tail(output)},
        )
    try:
        report = extract_first_json_object(output)
        errors = validate_probe_report(report)
    except Exception as exc:
        return CheckResult(
            name="mxfp8_tn_adapter_probe",
            status="fail",
            detail=f"could not parse/validate probe output: {type(exc).__name__}: {exc}",
            data={"output_tail": _tail(output)},
        )
    if errors:
        return CheckResult(
            name="mxfp8_tn_adapter_probe",
            status="fail",
            detail="probe did not satisfy accepted-path counters",
            data={"errors": errors, "shim_stats": report.get("shim_stats")},
        )
    return CheckResult(
        name="mxfp8_tn_adapter_probe",
        status="pass",
        detail="shim routed MXFP8 backward through the accepted MXFP8 backend with BF16 fallback and live sidecar counters at zero",
        data={"shim_stats": report.get("shim_stats")},
    )


def run_mxfp8_te_emit_probe(args: argparse.Namespace) -> CheckResult:
    if args.skip_mxfp8_probe or args.skip_te_emit_probe:
        return CheckResult(
            name="mxfp8_te_transpose_emit_probe",
            status="skip",
            detail="skipped by probe options",
        )

    available, reason = _probe_prereqs_available(args.probe_timeout_s)
    if not available and not args.require_mxfp8_probe:
        return CheckResult(
            name="mxfp8_te_transpose_emit_probe",
            status="skip",
            detail=f"CUDA/TE probe prerequisites unavailable: {reason}",
        )

    cmd = [
        sys.executable,
        str(repo_root() / "tools" / "probes" / "te_blockscaled_backward_probe.py"),
        "--format",
        "mxfp8",
        "--use-shim",
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--mxfp8-bwd-backend",
        "flashinfer_cutlass",
    ]
    env = _base_env()
    env.update(
        {
            "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER": "1",
            "CPPMEGA_TE_MXFP8_BWD_BACKEND": "flashinfer_cutlass",
            "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND": "te",
            "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED": "1",
            "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_STRICT": "1",
            "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK": "0",
            "CPPMEGA_TE_MXFP8_DGRAD_BF16": "0",
            "CPPMEGA_TE_MXFP8_WGRAD_BF16": "0",
            "NVTE_BACKWARD_OVERRIDE": "none",
            "CPPMEGA_TE_MXFP8_BWD_DEBUG": "1",
        }
    )
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        env=env,
        text=True,
        capture_output=True,
        timeout=args.probe_timeout_s,
        check=False,
    )
    output = _combined_output(proc)
    if proc.returncode != 0:
        return CheckResult(
            name="mxfp8_te_transpose_emit_probe",
            status="fail",
            detail="TE transpose-emission backward probe failed",
            data={"returncode": proc.returncode, "output_tail": _tail(output)},
        )
    try:
        report = extract_first_json_object(output)
        errors = validate_probe_report(report)
    except Exception as exc:
        return CheckResult(
            name="mxfp8_te_transpose_emit_probe",
            status="fail",
            detail=f"could not parse/validate probe output: {type(exc).__name__}: {exc}",
            data={"output_tail": _tail(output)},
        )
    stats = report.get("shim_stats", {})
    for key in (
        "mxfp8_tn_adapter_copy_transpose",
        "mxfp8_tn_adapter_missing_sidecar_copy",
        "mxfp8_tn_adapter_te_emit_swizzled_unavailable",
    ):
        if int(stats.get(key, 0)) != 0:
            errors.append(f"{key}={stats.get(key)}; expected 0 for TE transpose emit")
    for key in ("mxfp8_tn_adapter_te_emit", "mxfp8_tn_adapter_te_emit_swizzled"):
        if int(stats.get(key, 0)) <= 0:
            errors.append(f"{key}={stats.get(key)}; expected >0 for TE transpose emit")
    if errors:
        return CheckResult(
            name="mxfp8_te_transpose_emit_probe",
            status="fail",
            detail="TE transpose-emission counters did not satisfy acceptance checks",
            data={"errors": errors, "shim_stats": stats},
        )
    return CheckResult(
        name="mxfp8_te_transpose_emit_probe",
        status="pass",
        detail="TE emitted swizzled rowwise-transpose sidecars; BF16 fallback and adapter copy counters stayed zero",
        data={"shim_stats": stats},
    )


def validate_training_log(path: Path) -> CheckResult:
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_training_log(text)
    errors: list[str] = []
    if not parsed["losses"]:
        errors.append("no lm or mtp_1 loss found")
    counters = parsed["counters"]
    if not counters:
        errors.append("no TE block-scaled backward counters found")
    else:
        for key in FALLBACK_STAT_KEYS:
            if int(counters.get(key, -1)) != 0:
                errors.append(f"{key}={counters.get(key)}; expected 0")
        adapter_used = False
        direct_cutlass_used = False
        flashinfer_used = False
        for adapter_key, cutlass_key, flashinfer_key in zip(
            ADAPTER_STAT_KEYS,
            CUTLASS_STAT_KEYS,
            FLASHINFER_STAT_KEYS,
        ):
            adapter_count = int(counters.get(adapter_key, 0))
            cutlass_count = int(counters.get(cutlass_key, 0))
            flashinfer_count = int(counters.get(flashinfer_key, 0))
            adapter_used = adapter_used or adapter_count > 0
            direct_cutlass_used = direct_cutlass_used or cutlass_count > 0
            flashinfer_used = flashinfer_used or flashinfer_count > 0
            if adapter_count <= 0 and cutlass_count <= 0 and flashinfer_count <= 0:
                errors.append(
                    f"{adapter_key}={counters.get(adapter_key)} and "
                    f"{cutlass_key}={counters.get(cutlass_key)} and "
                    f"{flashinfer_key}={counters.get(flashinfer_key)}; expected one >0"
                )
        for key in SIDECAR_REGISTRY_ZERO_KEYS:
            if key not in counters:
                errors.append(f"{key} missing; expected 0")
            elif int(counters.get(key, -1)) != 0:
                errors.append(f"{key}={counters.get(key)}; expected 0")
        if direct_cutlass_used and not adapter_used and not flashinfer_used:
            for key in MATERIALIZATION_STAT_KEYS:
                if int(counters.get(key, 0)) != 0:
                    errors.append(f"{key}={counters.get(key)}; expected 0 for cutlass_native")
            if int(counters.get("mxfp8_tn_sidecar_registry_peak", -1)) != 0:
                errors.append(
                    "mxfp8_tn_sidecar_registry_peak="
                    f"{counters.get('mxfp8_tn_sidecar_registry_peak')}; expected 0 for cutlass_native"
                )
    reasons = parsed.get("fallback_reasons")
    if reasons not in ({}, None):
        errors.append(f"fallback_reasons={reasons!r}; expected empty")
    if errors:
        return CheckResult(
            name="one_step_log_parse",
            status="fail",
            detail="training log does not satisfy accepted-path evidence",
            data={"path": str(path), "errors": errors, **parsed},
        )
    return CheckResult(
        name="one_step_log_parse",
        status="pass",
        detail="parsed loss and zero-BF16-fallback MXFP8 counters from training log",
        data={"path": str(path), **parsed},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-mxfp8-probe", action="store_true")
    parser.add_argument("--skip-te-emit-probe", action="store_true")
    parser.add_argument(
        "--require-mxfp8-probe",
        action="store_true",
        help="Fail instead of skipping when CUDA/Transformer Engine prerequisites are unavailable.",
    )
    parser.add_argument("--probe-timeout-s", type=int, default=180)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument(
        "--train-log",
        type=Path,
        help="Optional one-step training log to parse for loss and MXFP8 counters.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    checks = [
        check_mxfp8_bf16_bridge_gate(),
        check_dsa_gather_scatter_gate(),
        check_fp8_activation_legacy_gate(),
        run_mxfp8_shim_probe(args),
        run_mxfp8_te_emit_probe(args),
    ]
    if args.train_log is not None:
        checks.append(validate_training_log(args.train_log))

    failed = any(check.status == "fail" for check in checks)
    if args.require_mxfp8_probe:
        failed = failed or any(
            check.name in ("mxfp8_tn_adapter_probe", "mxfp8_te_transpose_emit_probe")
            and check.status == "skip"
            for check in checks
        )

    print(
        json.dumps(
            {
                "status": "fail" if failed else "pass",
                "checks": [asdict(check) for check in checks],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
