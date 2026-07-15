import importlib.util
import inspect
from pathlib import Path

import pytest


pytest.importorskip("modal")


def _load_wave31_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "modal_mamba3_wave31_g8_reachability.py"
    spec = importlib.util.spec_from_file_location("modal_mamba3_wave31_g8_reachability", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_gate_defaults_to_current_production_flash_case():
    mod = _load_wave31_module()

    assert inspect.signature(mod.gate.get_raw_f()).parameters["case_label"].default == "te_flash_full"
    assert mod._CASES["te_flash_full"]["attention_backend"] == "flash"
    assert mod._CASES["te_flash_full"]["production_throughput"] is True
    assert mod._CASES["fallback_auto_full"]["production_throughput"] is False


def test_parse_log_extracts_te_backend_selection(tmp_path):
    mod = _load_wave31_module()
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "DEBUG:DotProductAttention:Disabling FlashAttention 2 as it does not support MLA.",
                "DEBUG:DotProductAttention:Available backends = {FlashAttention=False, FusedAttention=True (sub-backend 1), UnfusedDotProductAttention=True}",
                "DEBUG:DotProductAttention:Selected backend = FusedAttention (sub-backend 1).",
                "[2026-04-30] iteration        1/      20 | elapsed time per iteration (ms): 100.0 |",
                "[production_peak_mem] rank=0 device=0 peak_alloc_gib=12.500 peak_reserved_gib=13.000",
            ]
        )
    )

    metrics = mod._parse_log(log, tokens_per_iter=4096)

    assert metrics["te_flash_mla_rejected"] is True
    assert metrics["te_selected_backend_last"] == "FusedAttention (sub-backend 1)"
    assert metrics["te_available_backends_last"].startswith("{FlashAttention=False")
    assert metrics["iterations_seen"] == 1
    assert metrics["tok_sec_from_last_step"] == 40960.0
