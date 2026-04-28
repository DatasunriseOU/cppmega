"""Tests for the CCE main+MTP launch-fusion helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cppmega.megatron.mtp_native_hopper_ce import (
    _install_linear_ce_forward_cache_patch,
    attach_fused_main_mtp_cce_loss,
    fused_main_mtp_cce_loss,
)


class _FakeCceOutputLayer(torch.nn.Module):
    _cppmega_linear_ce_backend = "cce"

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(
        self,
        input_,
        weight=None,
        runtime_gather_output=None,
        output_cross_entropy_loss=False,
        labels=None,
        reduction="none",
        ignore_index=-100,
    ):
        assert output_cross_entropy_loss
        del runtime_gather_output
        weight = self.weight if weight is None else weight
        seq, batch, hidden = input_.shape
        hidden_2d = input_.reshape(seq * batch, hidden)
        target_1d = labels.transpose(0, 1).contiguous().reshape(-1)
        per_token = F.cross_entropy(
            F.linear(hidden_2d, weight),
            target_1d,
            ignore_index=ignore_index,
            reduction="none",
        )
        if reduction == "none":
            return per_token.reshape(seq, batch).transpose(0, 1).contiguous()
        if reduction == "sum":
            return per_token.sum()
        if reduction == "mean":
            return per_token[target_1d != ignore_index].mean()
        raise ValueError(reduction)


def _config(mtp_num_layers: int = 2, calculate_per_token_loss: bool = False):
    return SimpleNamespace(
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="linear",
        mtp_num_layers=mtp_num_layers,
        mtp_loss_scaling_factor=0.1,
        calculate_per_token_loss=calculate_per_token_loss,
        use_mup=False,
    )


def _roll_masked(labels: torch.Tensor, loss_mask: torch.Tensor):
    rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    rolled_labels[:, -1] = 0
    rolled_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
    rolled_mask[:, -1] = 0
    masked_labels = torch.where(
        rolled_mask.bool(),
        rolled_labels,
        torch.full_like(rolled_labels, -100),
    )
    return rolled_labels, rolled_mask, masked_labels


def _main_loss_total(loss_tokens, loss_mask, calculate_per_token_loss):
    total = (loss_tokens * loss_mask).sum()
    if calculate_per_token_loss:
        return total
    return total / loss_mask.sum().clamp(min=1)


def _separate_upstream_per_token_autoscaler_total(
    *,
    hidden_states,
    labels,
    loss_mask,
    output_layer,
    cfg,
):
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    seq = labels.shape[-1]
    hidden_main = hidden_states[:seq]
    mtp_labels = labels.clone()
    mtp_mask = loss_mask.clone()
    original_num_tokens = mtp_mask.sum()
    mtp_scale = cfg.mtp_loss_scaling_factor / cfg.mtp_num_layers

    for depth_idx in range(cfg.mtp_num_layers):
        mtp_labels, mtp_mask, masked_labels = _roll_masked(mtp_labels, mtp_mask)
        mtp_loss = output_layer(
            hidden_states[(depth_idx + 1) * seq : (depth_idx + 2) * seq],
            output_cross_entropy_loss=True,
            labels=masked_labels,
            reduction="none",
            ignore_index=-100,
        )
        mtp_loss = mtp_loss * mtp_mask
        if cfg.calculate_per_token_loss:
            num_tokens_safe = mtp_mask.sum().clamp(min=1)
            mtp_loss_normalized = mtp_scale * mtp_loss * (
                original_num_tokens / num_tokens_safe
            )
        else:
            mtp_loss_normalized = mtp_scale * mtp_loss / mtp_mask.sum().clamp(min=1)
        hidden_main = MTPLossAutoScaler.apply(hidden_main, mtp_loss_normalized)

    main_loss = output_layer(
        hidden_main,
        output_cross_entropy_loss=True,
        labels=labels,
        reduction="none",
        ignore_index=-100,
    )
    return _main_loss_total(main_loss, loss_mask, cfg.calculate_per_token_loss)


def test_fused_main_mtp_cce_loss_matches_separate_calls(monkeypatch):
    monkeypatch.setenv("CPPMEGA_CCE_FUSE_MAIN_MTP_CE", "1")
    torch.manual_seed(1234)

    batch, seq, hidden, vocab, mtp_depth = 2, 5, 4, 17, 2
    hidden_states = torch.randn((1 + mtp_depth) * seq, batch, hidden, requires_grad=True)
    weight = torch.randn(vocab, hidden, requires_grad=True)
    labels = torch.randint(0, vocab, (batch, seq))
    loss_mask = torch.ones(batch, seq)
    loss_mask[:, -1] = 0

    output_layer = _FakeCceOutputLayer(weight)
    cfg = _config(mtp_depth)

    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    fused_loss_tokens = fused_main_mtp_cce_loss(
        hidden_states=hidden_states,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=output_layer,
        output_weight=output_layer.weight,
        runtime_gather_output=False,
        is_training=False,
        config=cfg,
    )
    assert fused_loss_tokens is not None
    fused_total = (fused_loss_tokens * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    fused_total.backward()
    fused_hidden_grad = hidden_states.grad.detach().clone()
    fused_weight_grad = output_layer.weight.grad.detach().clone()

    hidden_ref = hidden_states.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    main = F.cross_entropy(
        F.linear(hidden_ref[:seq].transpose(0, 1), weight_ref).reshape(-1, vocab),
        labels.reshape(-1),
        reduction="none",
    ).reshape(batch, seq)
    ref_main_total = (main * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    ref_total = ref_main_total

    mtp_labels = labels.clone()
    mtp_mask = loss_mask.clone()
    mtp_scale = cfg.mtp_loss_scaling_factor / cfg.mtp_num_layers
    for depth_idx in range(mtp_depth):
        mtp_labels, mtp_mask, masked_labels = _roll_masked(mtp_labels, mtp_mask)
        mtp_logits = F.linear(
            hidden_ref[(depth_idx + 1) * seq : (depth_idx + 2) * seq].transpose(0, 1),
            weight_ref,
        )
        mtp_sum = F.cross_entropy(
            mtp_logits.reshape(-1, vocab),
            masked_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        ref_total = ref_total + mtp_scale * mtp_sum / mtp_mask.sum().clamp(min=1)
    ref_total.backward()

    assert torch.allclose(fused_total, ref_main_total, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused_hidden_grad, hidden_ref.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused_weight_grad, weight_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_fused_main_mtp_cce_loss_matches_upstream_per_token_autoscaler(
    monkeypatch, calculate_per_token_loss
):
    monkeypatch.setenv("CPPMEGA_CCE_FUSE_MAIN_MTP_CE", "1")
    torch.manual_seed(4321)

    batch, seq, hidden, vocab, mtp_depth = 2, 6, 5, 19, 2
    hidden_states = torch.randn((1 + mtp_depth) * seq, batch, hidden, requires_grad=True)
    weight = torch.randn(vocab, hidden)
    labels = torch.randint(0, vocab, (batch, seq))
    loss_mask = torch.ones(batch, seq)
    loss_mask[:, -2:] = 0

    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    cfg = _config(mtp_depth, calculate_per_token_loss=calculate_per_token_loss)
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    fused_layer = _FakeCceOutputLayer(weight.detach().clone())
    fused_loss_tokens = fused_main_mtp_cce_loss(
        hidden_states=hidden_states,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=fused_layer,
        output_weight=fused_layer.weight,
        runtime_gather_output=False,
        is_training=False,
        config=cfg,
    )
    assert fused_loss_tokens is not None
    fused_total = _main_loss_total(
        fused_loss_tokens, loss_mask, calculate_per_token_loss
    )
    fused_total.backward()
    fused_hidden_grad = hidden_states.grad.detach().clone()
    fused_weight_grad = fused_layer.weight.grad.detach().clone()

    hidden_ref = hidden_states.detach().clone().requires_grad_(True)
    ref_layer = _FakeCceOutputLayer(weight.detach().clone())
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    ref_total = _separate_upstream_per_token_autoscaler_total(
        hidden_states=hidden_ref,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=ref_layer,
        cfg=cfg,
    )
    ref_total.backward()

    assert torch.allclose(fused_total, ref_total, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused_hidden_grad, hidden_ref.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused_weight_grad, ref_layer.weight.grad, atol=1e-6, rtol=1e-6)


def test_attached_fused_loss_is_consumed_by_linear_ce_forward(monkeypatch):
    monkeypatch.setenv("CPPMEGA_CCE_FUSE_MAIN_MTP_CE", "1")
    torch.manual_seed(5678)

    class DummyLinearCe(torch.nn.Module):
        def forward(
            self,
            input_,
            weight=None,
            runtime_gather_output=None,
            output_cross_entropy_loss=False,
            labels=None,
            reduction="none",
            ignore_index=-100,
        ):
            del input_, weight, runtime_gather_output, output_cross_entropy_loss
            del labels, reduction, ignore_index
            raise AssertionError("cached fused loss was not consumed")

    _install_linear_ce_forward_cache_patch(DummyLinearCe)

    batch, seq, hidden, vocab, mtp_depth = 2, 4, 3, 11, 2
    hidden_states = torch.randn((1 + mtp_depth) * seq, batch, hidden, requires_grad=True)
    labels = torch.randint(0, vocab, (batch, seq))
    output_layer = _FakeCceOutputLayer(torch.randn(vocab, hidden))
    hidden_main = attach_fused_main_mtp_cce_loss(
        hidden_states=hidden_states,
        labels=labels,
        loss_mask=torch.ones(batch, seq),
        output_layer=output_layer,
        output_weight=output_layer.weight,
        runtime_gather_output=False,
        is_training=False,
        config=_config(mtp_depth),
    )
    assert hidden_main is not None

    cached = DummyLinearCe().forward(
        hidden_main,
        output_cross_entropy_loss=True,
        labels=labels,
        reduction="none",
    )
    assert cached.shape == labels.shape


def test_fused_main_mtp_cce_loss_declines_non_cce_backend():
    output_layer = torch.nn.Module()
    output_layer._cppmega_linear_ce_backend = "liger"
    loss = fused_main_mtp_cce_loss(
        hidden_states=torch.empty(6, 1, 2),
        labels=torch.empty(1, 2, dtype=torch.long),
        loss_mask=torch.ones(1, 2),
        output_layer=output_layer,
        output_weight=None,
        runtime_gather_output=False,
        is_training=False,
        config=_config(mtp_num_layers=2),
    )
    assert loss is None
