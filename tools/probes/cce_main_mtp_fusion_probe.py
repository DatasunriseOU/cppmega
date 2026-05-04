#!/usr/bin/env python3
"""Measure the main+MTP LinearCE launch-fusion path.

The probe intentionally uses the same typed run-profile renderer as training.
The process environment is only the transport that existing cppmega/Megatron
patch gates consume; the selected values come from ``RuntimePatchProfile``.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from cppmega.megatron import mtp_native_hopper_ce as mtp_ce_mod
from cppmega.megatron.mtp_native_hopper_ce import fused_main_mtp_cce_loss
from cppmega.recipes.run_profiles import get_run_profile, profile_shell_assignments


@dataclass(slots=True)
class CceMtpProbeConfig:
    profile: str = "local_gb10_quarter"
    batch: int = 2
    seq: int = 64
    hidden: int = 256
    vocab: int = 4096
    mtp_depth: int = 2
    warmup: int = 5
    iters: int = 20
    device: str = "cuda"
    dtype: str = "bf16"
    backend: str = "fake"


DEFAULT_PROBE_CONFIG = CceMtpProbeConfig()


@dataclass(slots=True)
class CceMtpProbeInputs:
    hidden_states: torch.Tensor
    weight: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor


class CountingFakeCceOutputLayer(torch.nn.Module):
    _cppmega_linear_ce_backend = "cce"

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.calls = 0

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
        self.calls += 1
        weight = self.weight if weight is None else weight
        seq, batch, hidden = input_.shape
        logits = F.linear(input_.reshape(seq * batch, hidden), weight)
        target = labels.transpose(0, 1).contiguous().reshape(-1)
        loss = F.cross_entropy(
            logits,
            target,
            ignore_index=ignore_index,
            reduction="none",
        )
        if reduction == "none":
            return loss.reshape(seq, batch).transpose(0, 1).contiguous()
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss[target != ignore_index].mean()
        raise ValueError(reduction)


class CountingCutCrossEntropyOutputLayer(torch.nn.Module):
    _cppmega_linear_ce_backend = "cce"

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.calls = 0

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
        from cut_cross_entropy import linear_cross_entropy

        self.calls += 1
        weight = self.weight if weight is None else weight
        seq, batch, hidden = input_.shape
        hidden_2d = input_.reshape(seq * batch, hidden)
        target = labels.transpose(0, 1).contiguous().reshape(-1)
        loss = linear_cross_entropy(
            hidden_2d,
            weight,
            target,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        if reduction == "none":
            return loss.reshape(seq, batch).transpose(0, 1).contiguous()
        return loss


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _apply_typed_profile(profile_name: str) -> dict[str, str]:
    profile = get_run_profile(profile_name)
    env = profile_shell_assignments(profile)
    for key in ("CPPMEGA_MTP_CE_KERNEL", "CPPMEGA_CCE_FUSE_MAIN_MTP_CE"):
        os.environ[key] = env[key]
    return env


def _model_config(cfg: CceMtpProbeConfig):
    return SimpleNamespace(
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="linear",
        mtp_num_layers=cfg.mtp_depth,
        mtp_loss_scaling_factor=0.1,
        calculate_per_token_loss=False,
        use_mup=False,
    )


def _roll_masked(labels: torch.Tensor, mask: torch.Tensor):
    rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    rolled_labels[:, -1] = 0
    rolled_mask = torch.roll(mask, shifts=-1, dims=-1)
    rolled_mask[:, -1] = 0
    masked_labels = torch.where(
        rolled_mask.bool(), rolled_labels, torch.full_like(rolled_labels, -100)
    )
    return rolled_labels, rolled_mask, masked_labels


def _make_inputs(cfg: CceMtpProbeConfig) -> CceMtpProbeInputs:
    device = torch.device(cfg.device)
    dtype = _torch_dtype(cfg.dtype)
    if device.type == "cpu" and dtype in (torch.bfloat16, torch.float16):
        dtype = torch.float32
    hidden_states = torch.randn(
        (1 + cfg.mtp_depth) * cfg.seq,
        cfg.batch,
        cfg.hidden,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(cfg.vocab, cfg.hidden, device=device, dtype=dtype)
    labels = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq), device=device)
    loss_mask = torch.ones(cfg.batch, cfg.seq, device=device, dtype=torch.float32)
    loss_mask[:, -cfg.mtp_depth :] = 0
    return CceMtpProbeInputs(hidden_states, weight, labels, loss_mask)


def _output_layer(cfg: CceMtpProbeConfig, weight: torch.Tensor):
    if cfg.backend == "fake":
        return CountingFakeCceOutputLayer(weight)
    if cfg.backend == "cce":
        return CountingCutCrossEntropyOutputLayer(weight)
    raise ValueError(f"unsupported backend: {cfg.backend}")


def _step_tensors(inputs: CceMtpProbeInputs):
    hidden_states = inputs.hidden_states.detach().clone().requires_grad_(True)
    weight = inputs.weight.detach().clone()
    return hidden_states, weight, inputs.labels, inputs.loss_mask


def _separate_step(cfg: CceMtpProbeConfig, inputs: CceMtpProbeInputs):
    hidden_states, weight, labels, loss_mask = _step_tensors(inputs)
    layer = _output_layer(cfg, weight)
    model_cfg = _model_config(cfg)
    auto_scaler = mtp_ce_mod._get_mtp_runtime_helpers().MTPLossAutoScaler
    auto_scaler.set_loss_scale(torch.tensor(1.0, device=hidden_states.device))

    hidden_main = hidden_states[: cfg.seq]
    mtp_labels = labels.clone()
    mtp_mask = loss_mask.clone()
    scale = model_cfg.mtp_loss_scaling_factor / model_cfg.mtp_num_layers
    for depth in range(cfg.mtp_depth):
        mtp_labels, mtp_mask, masked_labels = _roll_masked(mtp_labels, mtp_mask)
        mtp_loss = layer(
            hidden_states[(depth + 1) * cfg.seq : (depth + 2) * cfg.seq],
            output_cross_entropy_loss=True,
            labels=masked_labels,
            reduction="none",
            ignore_index=-100,
        )
        mtp_loss = scale * (mtp_loss * mtp_mask) / mtp_mask.sum().clamp(min=1)
        hidden_main = auto_scaler.apply(hidden_main, mtp_loss)

    main_loss = layer(
        hidden_main,
        output_cross_entropy_loss=True,
        labels=labels,
        reduction="none",
        ignore_index=-100,
    )
    total = (main_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    total.backward()
    return layer.calls, float(total.detach().cpu())


def _fused_step(cfg: CceMtpProbeConfig, inputs: CceMtpProbeInputs):
    hidden_states, weight, labels, loss_mask = _step_tensors(inputs)
    layer = _output_layer(cfg, weight)
    model_cfg = _model_config(cfg)
    auto_scaler = mtp_ce_mod._get_mtp_runtime_helpers().MTPLossAutoScaler
    auto_scaler.set_loss_scale(torch.tensor(1.0, device=hidden_states.device))

    main_loss = fused_main_mtp_cce_loss(
        hidden_states=hidden_states,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=layer,
        output_weight=layer.weight,
        runtime_gather_output=False,
        is_training=False,
        config=model_cfg,
    )
    if main_loss is None:
        raise RuntimeError("fused_main_mtp_cce_loss declined the fake CCE backend")
    total = (main_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    total.backward()
    return layer.calls, float(total.detach().cpu())


def _time(fn, cfg: CceMtpProbeConfig, inputs: CceMtpProbeInputs):
    for _ in range(cfg.warmup):
        fn(cfg, inputs)
    if cfg.device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last = None
        for _ in range(cfg.iters):
            last = fn(cfg, inputs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / cfg.iters, last

    t0 = time.perf_counter()
    last = None
    for _ in range(cfg.iters):
        last = fn(cfg, inputs)
    return (time.perf_counter() - t0) * 1000.0 / cfg.iters, last


def parse_args() -> CceMtpProbeConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=DEFAULT_PROBE_CONFIG.profile)
    parser.add_argument("--batch", type=int, default=DEFAULT_PROBE_CONFIG.batch)
    parser.add_argument("--seq", type=int, default=DEFAULT_PROBE_CONFIG.seq)
    parser.add_argument("--hidden", type=int, default=DEFAULT_PROBE_CONFIG.hidden)
    parser.add_argument("--vocab", type=int, default=DEFAULT_PROBE_CONFIG.vocab)
    parser.add_argument("--mtp-depth", type=int, default=DEFAULT_PROBE_CONFIG.mtp_depth)
    parser.add_argument("--warmup", type=int, default=DEFAULT_PROBE_CONFIG.warmup)
    parser.add_argument("--iters", type=int, default=DEFAULT_PROBE_CONFIG.iters)
    parser.add_argument("--device", default=DEFAULT_PROBE_CONFIG.device, choices=("cpu", "cuda"))
    parser.add_argument("--dtype", default=DEFAULT_PROBE_CONFIG.dtype, choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--backend", default=DEFAULT_PROBE_CONFIG.backend, choices=("fake", "cce"))
    ns = parser.parse_args()
    if ns.device == "cuda" and not torch.cuda.is_available():
        ns.device = "cpu"
        ns.dtype = "fp32"
    return CceMtpProbeConfig(**vars(ns))


def main() -> None:
    cfg = parse_args()
    env = _apply_typed_profile(cfg.profile)
    torch.manual_seed(1234)
    inputs = _make_inputs(cfg)
    sep_ms, (sep_calls, sep_loss) = _time(_separate_step, cfg, inputs)
    fused_ms, (fused_calls, fused_loss) = _time(_fused_step, cfg, inputs)
    speedup = sep_ms / fused_ms if fused_ms else float("inf")
    print(
        "cce_main_mtp_fusion_probe "
        f"profile={cfg.profile} backend={cfg.backend} device={cfg.device} dtype={cfg.dtype} "
        f"shape=batch:{cfg.batch},seq:{cfg.seq},hidden:{cfg.hidden},vocab:{cfg.vocab},"
        f"mtp_depth:{cfg.mtp_depth}"
    )
    print(
        "typed_profile "
        f"CPPMEGA_MTP_CE_KERNEL={env['CPPMEGA_MTP_CE_KERNEL']} "
        f"CPPMEGA_CCE_FUSE_MAIN_MTP_CE={env['CPPMEGA_CCE_FUSE_MAIN_MTP_CE']}"
    )
    print(f"separate calls={sep_calls} avg_ms={sep_ms:.4f} loss={sep_loss:.6f}")
    print(f"fused    calls={fused_calls} avg_ms={fused_ms:.4f} loss={fused_loss:.6f}")
    print(f"call_reduction={sep_calls}->{fused_calls} speedup={speedup:.3f}x")


if __name__ == "__main__":
    main()
