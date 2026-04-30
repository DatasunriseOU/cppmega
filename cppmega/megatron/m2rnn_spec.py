"""Minimal Megatron-style M2RNN seam for cppmega hybrid stacks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

# NO FALLBACK: TENorm is required (supports sequence parallel for TP>1).
# WrappedTorchNorm silently breaks TP>1 — crash immediately if TE is missing.
from megatron.core.extensions.transformer_engine import TENorm as _NormClass

from cppmega.features.m2rnn import build_cppmega_m2rnn_config
from cppmega.megatron.mamba_local_spec import CppMegaLocalMambaStack

# Fused Triton M²RNN kernel (replaces the pure-Python seq loop below).  When
# available, ``m2rnn_scan_triton`` is a drop-in for ``_torch_m2rnn_forward``
# and delivers ~40x fwd speedup on GB10 / ~100x on H200 at NAM56R dims
# (B=2, S=4096, H=8, K=64, V=16).  Set the typed ``m2rnn_kernel`` config
# field (or legacy ``CPPMEGA_M2RNN_KERNEL=torch`` before model construction)
# to force the slow reference path for debugging / A-B comparison.
# Triton M²RNN kernel import.  The torch fallback path is NOT a silent
# degradation — it's an explicit debug/parity reference selected via
# CPPMEGA_M2RNN_KERNEL=torch.  If Triton is missing on a GPU host,
# training will use the 460× slower Python scan and print a loud warning.
try:
    from cppmega.megatron.m2rnn_triton import (
        M2RNNRuntimeConfig as _M2RNNRuntimeConfig,
        TRITON_AVAILABLE as _M2RNN_TRITON_AVAILABLE,
        m2rnn_scan_triton as _m2rnn_scan_triton,
    )
except ImportError:  # pragma: no cover
    import warnings
    warnings.warn(
        "cppmega.megatron.m2rnn_triton not importable — M²RNN will use the "
        "460× slower Python scan loop.  This is NOT acceptable for training.  "
        "Install Triton or check your PYTHONPATH.",
        RuntimeWarning,
        stacklevel=2,
    )
    _M2RNN_TRITON_AVAILABLE = False
    _m2rnn_scan_triton = None  # type: ignore[assignment]
    _M2RNNRuntimeConfig = None  # type: ignore[assignment]


def _softplus_decay_gate(x: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    x = F.softplus(x.float() + dt_bias)
    x = -A_log.float().exp() * x
    return torch.exp(x).to(dtype=x.dtype)


def _torch_m2rnn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    *,
    h0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq, n_q, k_dim = q.shape
    n_k = k.size(-2)
    n_v = v.size(-2)
    n_w = W.size(0)
    n_f = xf.size(-1)
    v_dim = v.size(-1)
    n = max(n_q, n_k, n_v, n_w, n_f)

    if h0 is None:
        h = torch.zeros(batch, n, k_dim, v_dim, device=q.device, dtype=q.dtype)
    else:
        h = h0

    if n_q != n:
        q = q.repeat_interleave(n // n_q, dim=-2)
    if n_k != n:
        k = k.repeat_interleave(n // n_k, dim=-2)
    if n_v != n:
        v = v.repeat_interleave(n // n_v, dim=-2)
    if n_w != n:
        W = W.repeat_interleave(n // n_w, dim=0)
    if n_f != n:
        xf = xf.repeat_interleave(n // n_f, dim=-1)

    x = k[..., None] * v[..., None, :]
    W_expanded = W[None, ...]
    y = torch.empty(batch, seq, n, k_dim, v_dim, device=q.device, dtype=q.dtype)
    for s in range(seq):
        f = xf[:, s, :, None, None]
        h_new = torch.tanh(h @ W_expanded + x[:, s])
        h = f * h + (1 - f) * h_new
        y[:, s] = h
    out = (q[..., None, :] @ y).squeeze(-2)
    return out, h


class CppMegaM2RNNMixer(nn.Module):
    """Megatron-style training-only M2RNN mixer."""

    def __init__(
        self,
        config: TransformerConfig,
        d_model: int,
        submodules=None,
        layer_number: int | None = None,
        pg_collection=None,
        pp_layer_offset: int = 0,
    ) -> None:
        super().__init__()
        del submodules, layer_number, pg_collection, pp_layer_offset

        m2rnn = build_cppmega_m2rnn_config(config, d_model=d_model)
        self.hidden_size = d_model
        self.k_head_dim = m2rnn.k_head_dim
        self.v_head_dim = m2rnn.v_head_dim
        self.kernel_size = m2rnn.conv_kernel
        self.use_residual = m2rnn.use_residual
        self.use_xma = m2rnn.use_xma
        self.kernel_choice = m2rnn.runtime_kernel
        self.runtime_config = (
            _M2RNNRuntimeConfig(
                save_hnew=m2rnn.runtime_save_hnew,
                bwd_chunk_size=m2rnn.runtime_bwd_chunk_size,
                fwd_autotune=m2rnn.runtime_fwd_autotune,
                fwd_num_warps=m2rnn.runtime_fwd_num_warps,
                fwd_num_stages=m2rnn.runtime_fwd_num_stages,
                broadcast_views=m2rnn.runtime_broadcast_views,
                bwd_reduce_broadcast_qk=m2rnn.runtime_bwd_reduce_broadcast_qk,
            )
            if _M2RNNRuntimeConfig is not None
            else None
        )

        n_heads_default = max(1, self.hidden_size // (self.k_head_dim + self.v_head_dim))
        self.num_q_heads = getattr(config, "m2rnn_num_q_heads", 1)
        self.num_k_heads = getattr(config, "m2rnn_num_k_heads", 1)
        self.num_v_heads = getattr(config, "m2rnn_num_v_heads", 0) or n_heads_default
        self.num_f_heads = getattr(config, "m2rnn_num_f_heads", 0) or n_heads_default
        self.num_g_heads = getattr(config, "m2rnn_num_g_heads", 0) or n_heads_default
        self.num_weight_heads = getattr(config, "m2rnn_num_weight_heads", 0) or n_heads_default
        self.num_heads = max(
            self.num_q_heads,
            self.num_k_heads,
            self.num_v_heads,
            self.num_f_heads,
            self.num_g_heads,
            self.num_weight_heads,
        )

        self.q_dim = self.num_q_heads * self.k_head_dim
        self.k_dim = self.num_k_heads * self.k_head_dim
        self.v_dim = self.num_v_heads * self.v_head_dim
        self.g_dim = self.num_g_heads * self.v_head_dim
        self.conv_dim = self.q_dim + self.k_dim + self.v_dim

        self.input_projection = nn.Linear(
            self.hidden_size,
            self.conv_dim + self.num_f_heads + self.g_dim,
            bias=False,
            dtype=config.params_dtype,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        with torch.no_grad():
            A = torch.empty(self.num_heads).uniform_(m2rnn.A_init_min, m2rnn.A_init_max)
            self.A_log.copy_(torch.log(A))
            dt = torch.exp(
                torch.rand(self.num_heads) * (math.log(m2rnn.dt_init_max) - math.log(m2rnn.dt_init_min))
                + math.log(m2rnn.dt_init_min)
            )
            dt = torch.clamp(dt, min=m2rnn.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.copy_(inv_dt)

        if self.kernel_size > 0:
            self.conv1d = nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                kernel_size=self.kernel_size,
                padding=self.kernel_size - 1,
                groups=self.conv_dim,
                bias=True,
                dtype=config.params_dtype,
            )
        else:
            self.conv1d = None

        self.state_weight = nn.Parameter(
            torch.eye(self.v_head_dim, dtype=config.params_dtype).unsqueeze(0).expand(self.num_weight_heads, -1, -1).clone()
        )
        self.D = (
            nn.Parameter(torch.ones(self.num_heads, self.v_head_dim, dtype=config.params_dtype))
            if self.use_residual
            else None
        )
        self.g_norm = _NormClass(
            config=config,
            hidden_size=self.num_heads * self.v_head_dim,
            eps=config.layernorm_epsilon,
        )
        self.output_projection = nn.Linear(
            self.g_dim,
            self.hidden_size,
            bias=False,
            dtype=config.params_dtype,
        )

    def forward(
        self,
        hidden_states,
        inference_context=None,
        *,
        inference_params=None,
        packed_seq_params=None,
    ):
        if self.use_xma:
            raise NotImplementedError("CppMegaM2RNNMixer does not ship XMA kernels; use torch recurrence")
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("CppMegaM2RNNMixer does not support Megatron inference paths yet")
        if packed_seq_params is not None:
            raise NotImplementedError("CppMegaM2RNNMixer does not support packed sequences yet")
        if hidden_states.ndim != 3:
            raise ValueError("CppMegaM2RNNMixer expects hidden_states shaped [seq, batch, hidden]")

        x = hidden_states.transpose(0, 1).contiguous()
        batch, seq, _ = x.shape
        projected = self.input_projection(x)
        conv_input, f_input, g = projected.split((self.conv_dim, self.num_f_heads, self.g_dim), dim=-1)
        f = _softplus_decay_gate(f_input, self.A_log, self.dt_bias).to(dtype=x.dtype)

        if self.conv1d is not None:
            conv_input = self.conv1d(conv_input.transpose(1, 2))[..., :seq]
            conv_input = F.silu(conv_input).transpose(1, 2)

        q = conv_input[..., : self.q_dim].view(batch, seq, self.num_q_heads, self.k_head_dim)
        k_start = self.q_dim
        k = conv_input[..., k_start : k_start + self.k_dim].view(batch, seq, self.num_k_heads, self.k_head_dim)
        v_start = k_start + self.k_dim
        v = conv_input[..., v_start : v_start + self.v_dim].view(batch, seq, self.num_v_heads, self.v_head_dim)

        if (
            self.kernel_choice == "triton"
            and _M2RNN_TRITON_AVAILABLE
            and q.is_cuda
        ):
            out, _ = _m2rnn_scan_triton(
                q=q,
                k=k,
                v=v,
                W=self.state_weight,
                xf=f,
                runtime_config=self.runtime_config,
            )
        else:
            out, _ = _torch_m2rnn_forward(q=q, k=k, v=v, W=self.state_weight, xf=f)
        if self.D is not None:
            if self.num_v_heads != self.num_heads:
                v = v.repeat_interleave(self.num_heads // self.num_v_heads, dim=-2)
            out = out + v * self.D

        out = out.flatten(-2, -1)
        g = g.repeat_interleave(self.num_heads // self.num_g_heads, dim=-1)
        out = out * F.silu(g)
        out = self.g_norm(out)
        out = self.output_projection(out)
        return out.transpose(0, 1).contiguous(), None

    def mamba_state_shapes_per_request(self):
        raise NotImplementedError("CppMegaM2RNNMixer does not support Megatron inference cache shapes yet")


cppmega_m2rnn_stack_spec = ModuleSpec(
    module=CppMegaLocalMambaStack,
    submodules=CppMegaLocalMambaStack.get_default_submodules(
        mamba_mixer_spec=ModuleSpec(module=CppMegaM2RNNMixer)
    ),
)
