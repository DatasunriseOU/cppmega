"""Probe cppmega GB10 attention backend routing.

This is a launch-contract and TE selector probe, not an end-to-end training run.
It reports the FlashAttention packages visible to Transformer Engine, the
profile-rendered Megatron attention backend, and TE backend choices for NAM56R
MLA-shaped attention.  Use --smoke to run a small fwd+bwd attention op for one
or more forced backends.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import pathlib
import shlex
import subprocess
import sys
from collections.abc import Iterable


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cppmega.recipes.run_profiles import get_run_profile, profile_shell_assignments


BACKEND_ENVS = {
    "auto": {"NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "1"},
    "flash": {"NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "0"},
    "fused": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "0"},
    "unfused": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "1"},
}


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<not-installed>"


def _module_file(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - probe diagnostics
        return f"<import-failed: {exc}>"
    return str(getattr(module, "__file__", "<no-file>"))


def _flag_value(args: list[str], flag: str) -> str | None:
    try:
        idx = args.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _set_backend_env(backend: str) -> None:
    for key, value in BACKEND_ENVS[backend].items():
        os.environ[key] = value


def _format_te_selection(selection: tuple[object, ...]) -> str:
    use_flash, flash_backend, use_fused, fused_backend, use_unfused, available = selection
    if use_flash:
        selected = f"flash:{flash_backend}"
    elif use_fused:
        selected = f"fused:{fused_backend}"
    elif use_unfused:
        selected = "unfused"
    else:
        selected = "none"
    return (
        f"selected={selected} "
        f"available=flash:{int(bool(available[0]))},"
        f"fused:{int(bool(available[1]))},unfused:{int(bool(available[2]))}"
    )


def _select_te_backend(
    *,
    backend: str,
    label: str,
    batch_size: int,
    seqlen: int,
    num_heads: int,
    num_gqa_groups: int,
    head_dim_qk: int,
    head_dim_v: int,
) -> str:
    _set_backend_env(backend)

    import torch
    from transformer_engine.pytorch.attention.dot_product_attention.backends import (  # noqa: F401
        fa_utils,
    )
    from transformer_engine.pytorch.attention.dot_product_attention.utils import (
        AttentionParams,
        get_attention_backend,
    )

    params = AttentionParams(
        qkv_dtype=torch.bfloat16,
        qkv_layout="sbhd_sbhd_sbhd",
        batch_size=batch_size,
        num_heads=num_heads,
        num_gqa_groups=num_gqa_groups,
        max_seqlen_q=seqlen,
        max_seqlen_kv=seqlen,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        attn_mask_type="causal",
        core_attention_bias_type="no_bias",
        core_attention_bias_shape=None,
        attention_dropout=0.0,
        is_training=True,
    )
    return f"{label} backend={backend} {_format_te_selection(get_attention_backend(params))}"


def _smoke_backend(backend: str, *, seqlen: int, heads: int, qk_dim: int, v_dim: int) -> str:
    env = os.environ.copy()
    env.update(BACKEND_ENVS[backend])
    env.setdefault("PYTHONPATH", str(ROOT))
    env["CPPMEGA_PROBE_S"] = str(seqlen)
    env["CPPMEGA_PROBE_H"] = str(heads)
    env["CPPMEGA_PROBE_DQK"] = str(qk_dim)
    env["CPPMEGA_PROBE_DV"] = str(v_dim)
    code = r'''
import json
import os
import time

import torch
from transformer_engine.pytorch.attention import DotProductAttention

s = int(os.environ["CPPMEGA_PROBE_S"])
h = int(os.environ["CPPMEGA_PROBE_H"])
dqk = int(os.environ["CPPMEGA_PROBE_DQK"])
dv = int(os.environ["CPPMEGA_PROBE_DV"])
b = 1
g = h

torch.manual_seed(1234)
torch.cuda.set_device(0)
q = torch.randn(s, b, h, dqk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(s, b, g, dqk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(s, b, g, dv, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dpa = DotProductAttention(
    num_attention_heads=h,
    kv_channels=(dqk, dv),
    num_gqa_groups=g,
    qkv_format="sbhd",
    attention_dropout=0.0,
    attn_mask_type="causal",
)
torch.cuda.synchronize()
start = time.perf_counter()
out = dpa(
    q,
    k,
    v,
    qkv_format="sbhd",
    max_seqlen_q=s,
    max_seqlen_kv=s,
    attn_mask_type="causal",
    core_attention_bias_type="no_bias",
)
out.float().sum().backward()
torch.cuda.synchronize()
print(json.dumps({"ok": True, "ms_fwd_bwd": (time.perf_counter() - start) * 1000.0}))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    if proc.returncode == 0:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        return f"smoke backend={backend} ok ms_fwd_bwd={payload['ms_fwd_bwd']:.3f}"
    tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
    return f"smoke backend={backend} failed rc={proc.returncode} tail={tail}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="local_gb10_quarter")
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-backend",
        action="append",
        choices=tuple(BACKEND_ENVS),
        help="Backend to fwd+bwd smoke; can be repeated. Defaults to profile backend.",
    )
    parser.add_argument("--smoke-seqlen", type=int, default=512)
    parser.add_argument("--smoke-heads", type=int, default=8)
    parser.add_argument("--smoke-qk-dim", type=int, default=64)
    parser.add_argument("--smoke-v-dim", type=int, default=64)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import torch
    import transformer_engine as te

    env = profile_shell_assignments(get_run_profile(args.profile))
    native_args = shlex.split(env["NATIVE_ARGS"])

    print(f"profile={args.profile}")
    print(f"profile_attention_backend={env['CPPMEGA_ATTN_BACKEND']}")
    print(f"profile_use_flash_attn={env['CPPMEGA_USE_FLASH_ATTN']}")
    print(f"profile_dsa_sparse_mode={env['CPPMEGA_DSA_SPARSE_MODE']}")
    print(f"profile_dsa_fp8_attention={env['CPPMEGA_DSA_FP8_ATTENTION']}")
    print(f"native_experimental_attention={_flag_value(native_args, '--experimental-attention-variant')}")
    print(f"native_q_lora_rank={_flag_value(native_args, '--q-lora-rank')}")
    print(f"native_kv_lora_rank={_flag_value(native_args, '--kv-lora-rank')}")
    print(f"native_qk_head_dim={_flag_value(native_args, '--qk-head-dim')}")
    print(f"native_qk_pos_emb_head_dim={_flag_value(native_args, '--qk-pos-emb-head-dim')}")
    print(f"native_v_head_dim={_flag_value(native_args, '--v-head-dim')}")

    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
        print(f"cuda_capability={torch.cuda.get_device_capability(0)}")
    else:
        print("cuda_device=<unavailable>")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"transformer_engine={te.__version__} path={pathlib.Path(te.__file__).parent}")
    print(f"flash_attn={_version('flash-attn')} path={_module_file('flash_attn')}")
    print(f"flash_attn_2_cuda_path={_module_file('flash_attn_2_cuda')}")
    print(f"flash_attn_4={_version('flash-attn-4')}")
    print(f"flash_attn_cute_path={_module_file('flash_attn.cute.interface')}")
    print(f"tilelang={_version('tilelang')} path={_module_file('tilelang')}")

    shapes = [
        ("nam56r_standard_mla", args.batch_size, args.seqlen, 28, 28, 128, 64),
        ("nam56r_dsa_absorbed_mqa", args.batch_size, args.seqlen, 28, 1, 128, 64),
        ("equal_dim_reference", args.batch_size, args.seqlen, 28, 28, 64, 64),
    ]
    for label, batch_size, seqlen, heads, groups, qk_dim, v_dim in shapes:
        for backend in ("auto", "flash", "fused"):
            print(
                _select_te_backend(
                    backend=backend,
                    label=label,
                    batch_size=batch_size,
                    seqlen=seqlen,
                    num_heads=heads,
                    num_gqa_groups=groups,
                    head_dim_qk=qk_dim,
                    head_dim_v=v_dim,
                )
            )

    if args.smoke:
        smoke_backends = args.smoke_backend or [env["CPPMEGA_ATTN_BACKEND"]]
        for backend in smoke_backends:
            print(
                _smoke_backend(
                    backend,
                    seqlen=args.smoke_seqlen,
                    heads=args.smoke_heads,
                    qk_dim=args.smoke_qk_dim,
                    v_dim=args.smoke_v_dim,
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
