from dataclasses import dataclass

import torch

from cppmega.megatron import muon_dtype_audit


@dataclass
class DummyQuantizedState:
    data: torch.Tensor
    absmax: torch.Tensor


def test_qmuon_audit_records_bf16_grad_and_int8_state():
    muon_dtype_audit.reset_muon_dtype_audit()
    state = DummyQuantizedState(
        data=torch.zeros(8, dtype=torch.int8),
        absmax=torch.ones(1, dtype=torch.float32),
    )
    grad = torch.zeros(2, 4, dtype=torch.bfloat16)

    def update(states, grads):
        return states, grads

    wrapped = muon_dtype_audit._wrap_qmuon_group_update(update)

    assert wrapped([state], [grad]) == ([state], [grad])
    snapshot = muon_dtype_audit.get_muon_dtype_audit_snapshot()

    assert snapshot["qmuon_group_update_calls"] == 1
    assert snapshot["qmuon_grad_dtype_bfloat16_tensors"] == 1
    assert snapshot["qmuon_grad_dtype_bfloat16_elems"] == 8
    assert snapshot["qmuon_state_dtype_int8_tensors"] == 1
    assert snapshot["qmuon_absmax_dtype_float32_tensors"] == 1
    assert snapshot["bf16_owned_path_observed"] == 1


def test_newton_schulz_audit_records_step_and_lowmem_dtypes():
    muon_dtype_audit.reset_muon_dtype_audit()
    x = torch.zeros(2, 2, dtype=torch.bfloat16)

    def ns_step(tensor):
        return tensor.float()

    def ns_lowmem(tensor, steps, coefficient_type, *, already_normalized=False):
        return tensor

    wrapped_step = muon_dtype_audit._wrap_ns_step(ns_step)
    wrapped_lowmem = muon_dtype_audit._wrap_ns_lowmem(ns_lowmem)

    assert wrapped_step(x).dtype == torch.float32
    assert (
        wrapped_lowmem(x, 3, "cubic_quintic", already_normalized=True).dtype
        == torch.bfloat16
    )
    snapshot = muon_dtype_audit.get_muon_dtype_audit_snapshot()

    assert snapshot["ns_step_calls"] == 1
    assert snapshot["ns_step_dtype_bfloat16_tensors"] == 1
    assert snapshot["ns_step_dtype_bfloat16_elems"] == 4
    assert snapshot["ns_lowmem_calls"] == 1
    assert snapshot["ns_lowmem_input_dtype_bfloat16_tensors"] == 1
    assert snapshot["ns_lowmem_already_normalized_input_dtype_bfloat16_tensors"] == 1
    assert snapshot["ns_lowmem_output_dtype_bfloat16_tensors"] == 1
    assert snapshot["bf16_owned_path_observed"] == 1


def test_audit_format_is_one_greppable_key_value_line():
    muon_dtype_audit.reset_muon_dtype_audit()

    line = muon_dtype_audit.format_muon_dtype_audit_snapshot()

    assert line.startswith("[cppmega_muon_dtype_audit] ")
    assert "bf16_owned_path_observed=0" in line
