#!/usr/bin/env python3
"""Run the cppmega Megatron-backed C++ mini lane on one Nebius H200.

This runner is intentionally not a standalone PyTorch smoke.  It launches the
``h200_cpp_world_mini`` typed profile through Megatron's ``pretrain_mamba.py``
with ``CppMegaMambaModel`` and the cppmega structure sidecar patch installed.
The default batch sweep is short and meant to find the one-H200 memory ceiling.
"""

from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_ID = "project-e00w4wx5pr008qm6txmcgr"
DEFAULT_SUBNET_ID = "vpcsubnet-e00wm5th5ywn69bb0e"
DEFAULT_SECURITY_GROUP_ID = "vpcsecuritygroup-e00qtzsrtgdv7wh8e7"
DEFAULT_IMAGE_ID = "computeimage-e00hbfk8kmf3w3prch"
DEFAULT_DOCKER_IMAGE = "ghcr.io/datasunriseou/cppmega:latest"
DEFAULT_SIDECAR_PREFIX = (
    ROOT.parent
    / "cppmega.mlx"
    / "outputs"
    / "nebius_smoke"
    / "megatron"
    / "cppmega_1024_smoke_mix_train"
)
DEFAULT_TOKENIZER_DIR = ROOT / "data" / "tokenizer_v2"
OVERLAY_PATHS = (
    "cppmega/recipes/run_profiles.py",
    "cppmega/megatron/structure_dataset_patch.py",
)


def default_ssh_key() -> Path:
    for name in ("id_ed25519", "id_rsa", "google_compute_engine"):
        path = Path.home() / ".ssh" / name
        if path.exists() and Path(str(path) + ".pub").exists():
            return path
    return Path.home() / ".ssh" / "id_ed25519"


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    text: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[nebius-sweep] $ {printable}", flush=True)
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=text,
        timeout=timeout,
    )


def run_json(cmd: list[str], *, timeout: int | None = None) -> object:
    proc = run(cmd, capture=True, timeout=timeout)
    if not proc.stdout.strip():
        return {}
    return json.loads(proc.stdout)


def make_overlay_tar(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for rel in OVERLAY_PATHS:
            tf.add(ROOT / rel, arcname=rel)


def make_sidecar_tar(prefix: Path, tokenizer_dir: Path, path: Path) -> None:
    required = [prefix.with_suffix(".bin"), prefix.with_suffix(".idx"), prefix.with_suffix(".json")]
    manifest = json.loads(prefix.with_suffix(".json").read_text())
    for entry in manifest.get("side_channel_paths", {}).values():
        required.append(prefix.parent / entry["path"])
    required.extend(sorted(tokenizer_dir.iterdir()))

    for item in required:
        if not item.exists():
            raise FileNotFoundError(item)

    with tarfile.open(path, "w:gz") as tf:
        for item in required:
            if item.is_file() and item.parent == prefix.parent:
                tf.add(item, arcname=f"cppmega_sidecar/{item.name}")
            elif item.is_file() and item.parent == tokenizer_dir:
                tf.add(item, arcname=f"cpp_tokenizer_hf/{item.name}")


def _docker_auth_from_config(host: str = "ghcr.io") -> tuple[str, str] | None:
    config_path = Path.home() / ".docker" / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return None

    auths = config.get("auths") or {}
    for key in (host, f"https://{host}"):
        entry = auths.get(key)
        if isinstance(entry, dict) and entry.get("auth"):
            try:
                decoded = base64.b64decode(entry["auth"]).decode()
                username, secret = decoded.split(":", 1)
            except Exception:
                continue
            if username and secret:
                return username, secret

    helper_name = None
    cred_helpers = config.get("credHelpers") or {}
    if isinstance(cred_helpers, dict):
        helper_name = cred_helpers.get(host) or cred_helpers.get(f"https://{host}")
    helper_name = helper_name or config.get("credsStore")
    if not helper_name:
        return None

    helper = f"docker-credential-{helper_name}"
    for server in (host, f"https://{host}"):
        try:
            proc = subprocess.run(
                [helper, "get"],
                input=server,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0 or not proc.stdout.strip():
            continue
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            continue
        username = payload.get("Username") or payload.get("username")
        secret = payload.get("Secret") or payload.get("secret")
        if username and secret:
            return str(username), str(secret)
    return None


def resolve_ghcr_auth(args: argparse.Namespace) -> tuple[str, str] | None:
    if args.no_ghcr_auth:
        return None
    username = args.ghcr_user or os.environ.get("GHCR_USER") or os.environ.get("GITHUB_ACTOR")
    token = os.environ.get("GHCR_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if args.ghcr_token_file:
        token = args.ghcr_token_file.read_text().strip()
    if username and token:
        return username, token
    return _docker_auth_from_config("ghcr.io")


def make_ghcr_auth_tar(args: argparse.Namespace, path: Path) -> bool:
    auth = resolve_ghcr_auth(args)
    if auth is None:
        return False
    username, token = auth
    with tempfile.TemporaryDirectory(prefix="cppmega-ghcr-auth-") as tmp:
        root = Path(tmp) / "cppmega_auth"
        root.mkdir()
        (root / "ghcr_user").write_text(username)
        token_path = root / "ghcr_token"
        token_path.write_text(token)
        token_path.chmod(0o600)
        with tarfile.open(path, "w:gz") as tf:
            tf.add(root / "ghcr_user", arcname="cppmega_auth/ghcr_user")
            tf.add(token_path, arcname="cppmega_auth/ghcr_token")
    return True


def first_public_ip(obj: object) -> str | None:
    addresses: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            try:
                ip = ipaddress.ip_address(value)
            except ValueError:
                try:
                    ip = ipaddress.ip_interface(value).ip
                except ValueError:
                    return
            if ip.version == 4 and not ip.is_private and not ip.is_loopback:
                addresses.append(str(ip))

    walk(obj)
    return addresses[0] if addresses else None


def cloud_init(user: str, ssh_pubkey: str) -> str:
    return textwrap.dedent(
        f"""\
        #cloud-config
        users:
          - name: {user}
            sudo: ALL=(ALL) NOPASSWD:ALL
            shell: /bin/bash
            ssh_authorized_keys:
              - {ssh_pubkey}
        package_update: false
        runcmd:
          - mkdir -p /data
          - chown {user}:{user} /data
        """
    )


def create_instance(args: argparse.Namespace, ssh_pubkey: str) -> str:
    network = [
        {
            "name": "eth0",
            "subnet_id": args.subnet_id,
            "ip_address": {},
            "public_ip_address": {"static": False},
            "security_groups": [{"id": args.security_group_id}],
        }
    ]
    payload = cloud_init(args.ssh_user, ssh_pubkey)
    created = run_json(
        [
            "nebius",
            "compute",
            "instance",
            "create",
            "--parent-id",
            args.parent_id,
            "--name",
            args.instance_name,
            "--hostname",
            args.instance_name,
            "--resources-platform",
            args.platform,
            "--resources-preset",
            args.preset,
            "--boot-disk-attach-mode",
            "read_write",
            "--boot-disk-managed-disk-name",
            f"{args.instance_name}-boot",
            "--boot-disk-managed-disk-type",
            args.disk_type,
            "--boot-disk-managed-disk-size-gibibytes",
            str(args.disk_size_gib),
            "--boot-disk-managed-disk-source-image-id",
            args.image_id,
            "--network-interfaces",
            json.dumps(network),
            "--cloud-init-user-data",
            payload,
            "--preemptible-on-preemption",
            "stop",
            "--recovery-policy",
            "fail",
            "--format",
            "json",
            "--no-progress",
            "--timeout",
            "20m",
        ],
        timeout=1500,
    )
    instance_id = find_id(created)
    if not instance_id:
        raise RuntimeError(f"cannot find created instance id in: {created!r}")
    print(f"[nebius-sweep] created instance_id={instance_id}", flush=True)
    return instance_id


def find_id(obj: object) -> str | None:
    if isinstance(obj, dict):
        for key in ("id", "resource_id"):
            value = obj.get(key)
            if isinstance(value, str) and value.startswith("computeinstance-"):
                return value
        for value in obj.values():
            found = find_id(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_id(value)
            if found:
                return found
    return None


def get_instance(instance_id: str) -> object:
    return run_json(
        [
            "nebius",
            "compute",
            "instance",
            "get",
            instance_id,
            "--format",
            "json",
            "--no-progress",
        ]
    )


def wait_for_ip(instance_id: str, timeout_s: int = 900) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = get_instance(instance_id)
        ip = first_public_ip(info)
        if ip:
            print(f"[nebius-sweep] public_ip={ip}", flush=True)
            return ip
        time.sleep(10)
    raise TimeoutError(f"timed out waiting for public IP for {instance_id}")


def ssh_base(args: argparse.Namespace, ip: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(args.ssh_key),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=15",
        f"{args.ssh_user}@{ip}",
    ]


def wait_for_ssh(args: argparse.Namespace, ip: str, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        proc = subprocess.run(
            ssh_base(args, ip) + ["true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode == 0:
            print("[nebius-sweep] ssh ready", flush=True)
            return
        time.sleep(10)
    raise TimeoutError(f"timed out waiting for ssh to {ip}")


def ssh(args: argparse.Namespace, ip: str, command: str, *, timeout: int | None = None) -> None:
    run(ssh_base(args, ip) + [command], timeout=timeout)


def stream_tar_to_remote(args: argparse.Namespace, ip: str, tar_path: Path, target: str) -> None:
    cmd = (
        f"mkdir -p {shlex.quote(target)} && "
        f"tar -xzf - -C {shlex.quote(target)}"
    )
    ssh_cmd = ssh_base(args, ip) + [cmd]
    printable = " ".join(shlex.quote(part) for part in ssh_cmd)
    print(f"[nebius-sweep] streaming {tar_path.name} -> {target}: {printable}", flush=True)
    with tar_path.open("rb") as f:
        subprocess.run(ssh_cmd, stdin=f, check=True)


def remote_run_script(batch_sizes: list[int], train_iters: int, docker_image: str) -> str:
    batches = " ".join(str(v) for v in batch_sizes)
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        sudo mkdir -p /data/cppmega_h200_results /data/cppmega_overlay
        sudo chown -R "$USER":"$USER" /data

        if ! command -v docker >/dev/null 2>&1; then
          sudo apt-get update
          sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
            docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
          sudo systemctl enable --now docker
        fi
        sudo usermod -aG docker "$USER" || true

        if ! sudo docker info 2>/dev/null | grep -qi nvidia; then
          if command -v nvidia-ctk >/dev/null 2>&1; then
            sudo nvidia-ctk runtime configure --runtime=docker
            sudo systemctl restart docker
          fi
        fi

        nvidia-smi
        if [[ -s /data/cppmega_auth/ghcr_token ]]; then
          sudo docker login ghcr.io \
            -u "$(cat /data/cppmega_auth/ghcr_user)" \
            --password-stdin < /data/cppmega_auth/ghcr_token
          rm -f /data/cppmega_auth/ghcr_token
        fi
        sudo docker pull {shlex.quote(docker_image)}

        cat >/data/cppmega_h200_results/container_run.sh <<'INNER'
        set -euo pipefail
        cp -a /overlay/. /opt/cppmega/
        export PYTHONPATH="/opt/cppmega:/opt/megatron-lm:${{PYTHONPATH:-}}"
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export NCCL_GRAPH_REGISTER=0
        export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
        export TRITON_CACHE_DIR="/data/.triton-cache"
        mkdir -p "$TRITON_CACHE_DIR" /data/cppmega_h200_results

        python - <<'PY'
        import importlib
        import json
        import torch

        modules = [
            "torch",
            "transformer_engine",
            "transformer_engine.pytorch",
            "flash_attn",
            "flash_attn_3",
            "flash_attn.cute",
            "cutlass",
            "quack",
            "mamba_ssm",
            "megatron.core",
            "cppmega",
        ]
        report = {{}}
        for name in modules:
            mod = importlib.import_module(name)
            report[name] = {{
                "file": getattr(mod, "__file__", None),
                "version": getattr(mod, "__version__", None),
            }}
        import megatron.core.utils as core_utils
        report["megatron.core.utils.get_batch_on_this_tp_rank"] = hasattr(
            core_utils, "get_batch_on_this_tp_rank"
        )
        import cppmega.megatron.structure_dataset_patch
        report["cuda"] = {{
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }}
        print("CPPMEGA_STACK_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
        assert report["megatron.core.utils.get_batch_on_this_tp_rank"], report
        PY

        for BS in {batches}; do
          LOG="/data/cppmega_h200_results/bs_${{BS}}.log"
          NVSMI="/data/cppmega_h200_results/bs_${{BS}}.nvsmi.csv"
          echo "[container] starting batch=${{BS}}" | tee "$LOG"
          (
            while true; do
              ts="$(date '+%Y-%m-%dT%H:%M:%S')"
              nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
                --format=csv,noheader,nounits |
                while IFS=, read -r mu mt ug tg; do
                  echo "${{ts}},${{mu}},${{mt}},${{ug}},${{tg}}"
                done
              sleep 1
            done
          ) > "$NVSMI" 2>&1 &
          NVSMI_PID=$!

          set +e
          bash -lc "
            set -euo pipefail
            WORKDIR=\\$(mktemp -d /tmp/cppmega-h200-world.XXXXXX)
            trap 'rm -rf \\\"\\$WORKDIR\\\"' EXIT
            cat >\\\"\\$WORKDIR/pretrain_mamba.py\\\" <<'PYWRAP'
        from __future__ import annotations
        import atexit
        import os
        import runpy
        import sys

        if os.environ.get('CPPMEGA_STRUCTURE_ENABLED', '0') == '1':
            import cppmega.megatron.structure_dataset_patch  # noqa: F401

        @atexit.register
        def _cppmega_peak_memory_report():
            try:
                import torch
                if torch.cuda.is_available():
                    print(
                        'CPPMEGA_CUDA_PEAK allocated_gib='
                        f'{{torch.cuda.max_memory_allocated() / 1024**3:.3f}} '
                        'reserved_gib='
                        f'{{torch.cuda.max_memory_reserved() / 1024**3:.3f}}',
                        flush=True,
                    )
            except Exception as exc:
                print(f'CPPMEGA_CUDA_PEAK_ERROR {{exc}}', flush=True)

        _workdir = os.path.dirname(os.path.abspath(__file__))
        _inner = '/opt/megatron-lm/pretrain_mamba.py'
        sys.path.insert(0, _workdir)
        sys.path.insert(1, os.path.dirname(_inner))
        sys.argv[0] = _inner
        runpy.run_path(_inner, run_name='__main__')
        PYWRAP
            cat >\\\"\\$WORKDIR/mamba_builders.py\\\" <<'PY'
        from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder
        PY
            eval \\\"\\$(python -m cppmega.recipes.run_profiles shell h200_cpp_world_mini \\
              --seq-length 1024 \\
              --micro-batch-size ${{BS}} \\
              --global-batch-size ${{BS}} \\
              --train-iters {train_iters} \\
              --fp8-recipe off)\\\"

            DATA_ARGS=(--data-path 1.0 /data/cppmega_sidecar/cppmega_1024_smoke_mix_train)
            OPTIMIZER_ARGS=(--optimizer \\\"\\$CPPMEGA_OPTIMIZER\\\")
            if [[ \\\"\\$CPPMEGA_OPTIMIZER\\\" == muon || \\\"\\$CPPMEGA_OPTIMIZER\\\" == dist_muon || \\\"\\$CPPMEGA_OPTIMIZER\\\" == adaptive_muon ]]; then
              OPTIMIZER_ARGS+=(--muon-momentum \\\"\\$CPPMEGA_MUON_MOMENTUM\\\" --muon-scale-mode \\\"\\$CPPMEGA_MUON_SCALE_MODE\\\" --muon-num-ns-steps \\\"\\$CPPMEGA_MUON_NUM_NS_STEPS\\\" --muon-tp-mode \\\"\\$CPPMEGA_MUON_TP_MODE\\\" --muon-scalar-optimizer \\\"\\$CPPMEGA_MUON_SCALAR_OPTIMIZER\\\")
              if [[ \\\"\\$CPPMEGA_MUON_QUANTIZED_MOMENTUM\\\" == 1 ]]; then
                OPTIMIZER_ARGS+=(--muon-quantized-momentum --muon-quantized-momentum-dtype \\\"\\$CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE\\\" --muon-quantized-momentum-block-size \\\"\\$CPPMEGA_MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE\\\")
              fi
            fi
            if [[ \\\"\\$CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER\\\" == 1 ]]; then OPTIMIZER_ARGS+=(--use-bf16-no-master-emerging-optimizer); fi
            if [[ \\\"\\$CPPMEGA_USE_BF16_NO_MASTER_EMERGING_FALLBACK_OPTIMIZER\\\" == 1 ]]; then OPTIMIZER_ARGS+=(--use-bf16-no-master-emerging-fallback-optimizer); fi
            if [[ \\\"\\$CPPMEGA_GRAD_REDUCE_IN_BF16\\\" == 1 || \\\"\\$CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER\\\" == 1 ]]; then OPTIMIZER_ARGS+=(--grad-reduce-in-bf16); fi
            if [[ \\\"\\$CPPMEGA_LOCAL_DDP_DISABLE_CONTIGUOUS_GRAD_BUFFER\\\" == 1 ]]; then OPTIMIZER_ARGS+=(--local-ddp-disable-contiguous-grad-buffer); fi

            GQA_ARGS=(--group-query-attention --num-query-groups \\\"\\$CPPMEGA_NUM_QUERY_GROUPS\\\" --kv-channels \\\"\\$CPPMEGA_KV_CHANNELS\\\" --swiglu --rotary-base 10000)

            python -m torch.distributed.run --nproc_per_node=1 \\\"\\$WORKDIR/pretrain_mamba.py\\\" \\
              \\\"\\${{DATA_ARGS[@]}}\\\" \\
              --tokenizer-type HuggingFaceTokenizer \\
              --tokenizer-model /data/cpp_tokenizer_hf \\
              --vocab-size 65536 \\
              --make-vocab-size-divisible-by 128 \\
              --tensor-model-parallel-size 1 \\
              --pipeline-model-parallel-size 1 \\
              --context-parallel-size 1 \\
              --no-gradient-accumulation-fusion \\
              --no-persist-layer-norm \\
              --no-masked-softmax-fusion \\
              --hybrid-layer-pattern \\\"\\$HYBRID_LAYER_PATTERN\\\" \\
              --hidden-size \\\"\\$CPPMEGA_HIDDEN_SIZE\\\" \\
              --ffn-hidden-size \\\"\\$CPPMEGA_FFN_HIDDEN_SIZE\\\" \\
              --num-attention-heads \\\"\\$CPPMEGA_NUM_ATTN_HEADS\\\" \\
              \\\"\\${{GQA_ARGS[@]}}\\\" \\
              --seq-length 1024 \\
              --max-position-embeddings 1024 \\
              --micro-batch-size ${{BS}} \\
              --global-batch-size ${{BS}} \\
              --train-iters {train_iters} \\
              --eval-interval 50000000 \\
              --eval-iters 0 \\
              --lr \\\"\\$CPPMEGA_LR\\\" \\
              --min-lr \\\"\\$CPPMEGA_MIN_LR\\\" \\
              --lr-decay-style constant \\
              --position-embedding-type rope \\
              --no-rope-fusion \\
              --normalization RMSNorm \\
              --disable-bias-linear \\
              --bf16 \\
              --use-mcore-models \\
              --transformer-impl transformer_engine \\
              --use-flash-attn \\
              --attention-backend flash \\
              --spec cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec \\
              --cross-entropy-loss-fusion \\
              --cross-entropy-fusion-impl linear \\
              --recompute-granularity selective \\
              --recompute-modules mlp \\
              --clip-grad 1.0 \\
              \\\"\\${{OPTIMIZER_ARGS[@]}}\\\" \\
              --no-check-for-nan-in-loss-and-grad \\
              --rerun-mode disabled \\
              --save-interval 50000000 \\
              --log-interval 1
          " >>"$LOG" 2>&1
          status=$?
          kill "$NVSMI_PID" 2>/dev/null || true
          wait "$NVSMI_PID" 2>/dev/null || true
          set -e
          peak="$(awk -F, '{{ if ($2+0 > peak) peak=$2+0 }} END {{ print peak+0 }}' "$NVSMI")"
          echo "CPPMEGA_NVIDIA_SMI_PEAK batch=${{BS}} peak_used_mib=${{peak}}" | tee -a "$LOG"
          if [[ "$status" != 0 ]]; then
            echo "CPPMEGA_BATCH_RESULT batch=${{BS}} status=FAIL exit=${{status}}" | tee -a "$LOG"
            if grep -qiE 'out of memory|cuda error: out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$LOG"; then
              echo "CPPMEGA_BATCH_OOM batch=${{BS}}" | tee -a "$LOG"
              exit 0
            fi
            exit "$status"
          fi
          echo "CPPMEGA_BATCH_RESULT batch=${{BS}} status=OK" | tee -a "$LOG"
        done
        INNER

        sudo docker run --gpus all --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
          -v /data:/data \
          -v /data/cppmega_overlay:/overlay:ro \
          {shlex.quote(docker_image)} \
          bash /data/cppmega_h200_results/container_run.sh
        """
    )


def parse_batches(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("empty batch list")
    if any(value <= 0 for value in values):
        raise ValueError(f"batch sizes must be positive: {values}")
    return values


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-id", default=os.environ.get("NEBIUS_PARENT_ID", DEFAULT_PARENT_ID))
    parser.add_argument("--subnet-id", default=os.environ.get("NEBIUS_SUBNET_ID", DEFAULT_SUBNET_ID))
    parser.add_argument(
        "--security-group-id",
        default=os.environ.get("NEBIUS_SECURITY_GROUP_ID", DEFAULT_SECURITY_GROUP_ID),
    )
    parser.add_argument("--image-id", default=os.environ.get("NEBIUS_IMAGE_ID", DEFAULT_IMAGE_ID))
    parser.add_argument("--platform", default="gpu-h200-sxm")
    parser.add_argument("--preset", default="1gpu-16vcpu-200gb")
    parser.add_argument("--disk-type", default="network_ssd")
    parser.add_argument("--disk-size-gib", type=int, default=512)
    parser.add_argument("--instance-name", default=f"cppmega-h200-megatron-{int(time.time())}")
    parser.add_argument("--ssh-user", default="dave")
    parser.add_argument("--ssh-key", type=Path, default=default_ssh_key())
    parser.add_argument("--ssh-pubkey", type=Path, default=None)
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--ghcr-user", default=None)
    parser.add_argument("--ghcr-token-file", type=Path, default=None)
    parser.add_argument("--no-ghcr-auth", action="store_true")
    parser.add_argument("--sidecar-prefix", type=Path, default=DEFAULT_SIDECAR_PREFIX)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--batch-sizes", default="256,512,1024")
    parser.add_argument("--train-iters", type=int, default=3)
    parser.add_argument("--keep-instance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    batches = parse_batches(args.batch_sizes)
    pubkey_path = args.ssh_pubkey or Path(str(args.ssh_key) + ".pub")
    if not pubkey_path.exists():
        raise FileNotFoundError(f"ssh public key not found: {pubkey_path}")
    ssh_pubkey = pubkey_path.read_text().strip()

    if args.dry_run:
        print(f"parent_id={args.parent_id}")
        print(f"sidecar_prefix={args.sidecar_prefix}")
        print(f"tokenizer_dir={args.tokenizer_dir}")
        print(f"batches={batches}")
        print(remote_run_script(batches, args.train_iters, args.docker_image)[:4000])
        return 0

    instance_id: str | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="cppmega-h200-sweep-") as tmp:
            overlay_tar = Path(tmp) / "cppmega_overlay.tgz"
            sidecar_tar = Path(tmp) / "cppmega_sidecar.tgz"
            ghcr_auth_tar = Path(tmp) / "cppmega_ghcr_auth.tgz"
            make_overlay_tar(overlay_tar)
            make_sidecar_tar(args.sidecar_prefix, args.tokenizer_dir, sidecar_tar)
            has_ghcr_auth = make_ghcr_auth_tar(args, ghcr_auth_tar)
            if args.docker_image.startswith("ghcr.io/") and not has_ghcr_auth and not args.no_ghcr_auth:
                raise RuntimeError(
                    "GHCR image selected but no auth was found. Set GHCR_TOKEN/GITHUB_TOKEN, "
                    "pass --ghcr-token-file, or docker login ghcr.io locally."
                )

            instance_id = create_instance(args, ssh_pubkey)
            ip = wait_for_ip(instance_id)
            wait_for_ssh(args, ip)
            stream_tar_to_remote(args, ip, overlay_tar, "/data/cppmega_overlay")
            stream_tar_to_remote(args, ip, sidecar_tar, "/data")
            if has_ghcr_auth:
                stream_tar_to_remote(args, ip, ghcr_auth_tar, "/data")
            script = remote_run_script(batches, args.train_iters, args.docker_image)
            ssh(args, ip, f"cat > /data/run_cppmega_h200_sweep.sh <<'EOF'\n{script}\nEOF\nchmod +x /data/run_cppmega_h200_sweep.sh")
            try:
                ssh(args, ip, "bash /data/run_cppmega_h200_sweep.sh", timeout=7200)
            finally:
                out_dir = ROOT / "outputs" / "nebius" / args.instance_name
                out_dir.mkdir(parents=True, exist_ok=True)
                scp_cmd = [
                    "scp",
                    "-i",
                    str(args.ssh_key),
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    "-o",
                    "ConnectTimeout=15",
                    "-r",
                    f"{args.ssh_user}@{ip}:/data/cppmega_h200_results/.",
                    str(out_dir),
                ]
                try:
                    run(scp_cmd, check=False)
                except FileNotFoundError:
                    print("[nebius-sweep] scp unavailable; skipping log fetch", file=sys.stderr)
    finally:
        if instance_id and not args.keep_instance:
            run(
                [
                    "nebius",
                    "compute",
                    "instance",
                    "delete",
                    instance_id,
                    "--format",
                    "json",
                    "--no-progress",
                    "--timeout",
                    "20m",
                ],
                check=False,
                timeout=1500,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
