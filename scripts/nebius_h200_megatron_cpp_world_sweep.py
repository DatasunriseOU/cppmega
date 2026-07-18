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
import re
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cppmega.recipes.run_profiles import get_run_profile  # noqa: E402
from scripts.data.publish_megatron_bundle_to_nebius_s3 import (  # noqa: E402
    _validate_bundle,
    _validate_prefix_manifest_contract,
    _validate_tokenizer_directory,
)
from scripts.h200_megatron_preflight import (  # noqa: E402
    derive_graph_capacity_receipt,
)


DEFAULT_PARENT_ID = "project-e00w4wx5pr008qm6txmcgr"
DEFAULT_SUBNET_ID = "vpcsubnet-e00wm5th5ywn69bb0e"
DEFAULT_SECURITY_GROUP_ID = "vpcsecuritygroup-e00qtzsrtgdv7wh8e7"
DEFAULT_IMAGE_ID = "computeimage-e00hbfk8kmf3w3prch"
DEFAULT_DOCKER_IMAGE = (
    "ghcr.io/datasunriseou/cppmega@"
    "sha256:08c5db7368d1037d930e0825281468927de9c85b12ba10373fe07e082150d983"
)
DEFAULT_SIDECAR_PREFIX = (
    ROOT.parent
    / "cppmega.mlx"
    / "outputs"
    / "megatron_ready"
    / "cppmega_1024_current_mix_graph_train"
)
DEFAULT_TOKENIZER_DIR = ROOT / "data" / "tokenizer_v2"
OVERLAY_PATHS = (
    "cppmega",
    "scripts/h200_megatron_preflight.py",
    "scripts/data/publish_megatron_bundle_to_nebius_s3.py",
    "data/domain_schema_v1.json",
    "data/tokenizer_v2/tokenizer_contract_v1.json",
)
_IMMUTABLE_IMAGE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


def validate_docker_image_digest(image: str) -> str:
    if not _IMMUTABLE_IMAGE_RE.fullmatch(image):
        raise ValueError(
            "--docker-image must be an immutable image digest "
            "(registry/repository@sha256:<64 lowercase hex chars>); mutable tags are rejected"
        )
    return image


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
    def keep_source(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
            return None
        return info

    with tarfile.open(path, "w:gz") as tf:
        for rel in OVERLAY_PATHS:
            tf.add(ROOT / rel, arcname=rel, filter=keep_source)


def _assert_prefix_contract(prefix: Path) -> dict:
    manifest, _referenced = _validate_prefix_manifest_contract(prefix)
    return manifest


def _sidecar_required_files(prefix: Path, tokenizer_dir: Path) -> list[Path]:
    required = [prefix.with_suffix(".bin"), prefix.with_suffix(".idx"), prefix.with_suffix(".json")]
    manifest = _assert_prefix_contract(prefix)
    for entry in manifest.get("side_channel_paths", {}).values():
        required.append(prefix.parent / entry["path"])
    for entry in manifest.get("graph_sidecar_paths", {}).values():
        required.append(prefix.parent / entry["offsets_path"])
        required.append(prefix.parent / entry["data_path"])
    source_platform = manifest.get("source_platform_sidecar")
    if source_platform is not None:
        if not isinstance(source_platform, dict):
            raise ValueError("source_platform_sidecar must be a JSON object")
        for key in (
            "sequence_doc_offsets_path",
            "doc_platform_offsets_path",
            "platform_ids_path",
        ):
            if key not in source_platform:
                raise KeyError(f"source_platform_sidecar missing {key}")
            required.append(prefix.parent / source_platform[key])
    _validate_tokenizer_directory(tokenizer_dir)
    required.extend(sorted(tokenizer_dir.iterdir()))

    for item in required:
        if not item.exists():
            raise FileNotFoundError(item)
    return required


def make_multi_sidecar_tar(prefixes: list[Path], tokenizer_dir: Path, path: Path) -> None:
    required: list[Path] = []
    seen: set[Path] = set()
    for prefix in prefixes:
        for item in _sidecar_required_files(prefix, tokenizer_dir):
            key = item.resolve()
            if key in seen:
                continue
            required.append(item)
            seen.add(key)

    with tempfile.TemporaryDirectory(prefix="cppmega-sidecar-stage-") as stage_raw:
        stage = Path(stage_raw)
        sidecar_stage = stage / "cppmega_sidecar"
        tokenizer_stage = stage / "cpp_tokenizer_hf"
        sidecar_stage.mkdir()
        tokenizer_stage.mkdir()
        for item in required:
            if item.is_file() and any(item.parent == prefix.parent for prefix in prefixes):
                target = sidecar_stage / item.name
            elif item.is_file() and item.parent == tokenizer_dir:
                target = tokenizer_stage / item.name
            else:
                continue
            if target.exists():
                raise FileExistsError(f"duplicate archive member: {target.name}")
            os.symlink(item.resolve(), target)

        cmd = [
            "tar",
            "-czhf",
            str(path),
            "-C",
            str(stage),
            "cppmega_sidecar",
            "cpp_tokenizer_hf",
        ]
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(f"[nebius-sweep] $ GZIP=-1 COPYFILE_DISABLE=1 {printable}", flush=True)
        env = {**os.environ, "GZIP": "-1", "COPYFILE_DISABLE": "1"}
        subprocess.run(cmd, check=True, env=env)


def make_sidecar_tar(prefix: Path, tokenizer_dir: Path, path: Path) -> None:
    make_multi_sidecar_tar([prefix], tokenizer_dir, path)


def make_bundle_tar(
    bundle_root: Path,
    prefixes: list[Path],
    path: Path,
    *,
    hash_jobs: int = 4,
) -> tuple[list[str], str]:
    bundle_root = bundle_root.resolve()
    manifest, artifacts = _validate_bundle(bundle_root, hash_jobs)
    declared_prefixes = {
        (bundle_root / str(result["prefix"])).resolve(): str(result["prefix"])
        for result in manifest["bucket_results"]
    }
    remote_prefixes: list[str] = []
    for prefix in prefixes:
        relative = declared_prefixes.get(prefix.resolve())
        if relative is None:
            raise ValueError(f"selected prefix is not declared by bundle: {prefix}")
        remote_prefixes.append(relative)
    tokenizer_relative = str(manifest["tokenizer"]["path"])

    with tempfile.TemporaryDirectory(prefix="cppmega-bundle-stage-") as stage_raw:
        stage_root = Path(stage_raw) / "cppmega_bundle"
        stage_root.mkdir()
        sources = [bundle_root / "manifest.json"] + [
            bundle_root / str(record["path"]) for record in artifacts
        ]
        for source in sources:
            target = stage_root / source.relative_to(bundle_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(source, target)
        cmd = [
            "tar",
            "-czhf",
            str(path),
            "-C",
            stage_raw,
            "cppmega_bundle",
        ]
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(f"[nebius-sweep] $ GZIP=-1 COPYFILE_DISABLE=1 {printable}", flush=True)
        env = {**os.environ, "GZIP": "-1", "COPYFILE_DISABLE": "1"}
        subprocess.run(cmd, check=True, env=env)
    return remote_prefixes, tokenizer_relative


def make_checkpoint_tar(checkpoint_dir: Path, path: Path) -> None:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(checkpoint_dir)
    if not (checkpoint_dir / "latest_checkpointed_iteration.txt").exists():
        raise FileNotFoundError(
            f"{checkpoint_dir} does not look like a Megatron checkpoint root: "
            "missing latest_checkpointed_iteration.txt"
        )

    cmd = ["tar", "-czf", str(path), "-C", str(checkpoint_dir), "."]
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[nebius-sweep] $ GZIP=-1 COPYFILE_DISABLE=1 {printable}", flush=True)
    env = {**os.environ, "GZIP": "-1", "COPYFILE_DISABLE": "1"}
    subprocess.run(cmd, check=True, env=env)


def _docker_auth_from_config(host: str = "ghcr.io") -> tuple[str, str] | None:
    config_path = Path.home() / ".docker" / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"Docker config is configured but broken: {config_path} exists but is "
            f"unparseable ({exc})"
        ) from exc

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
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Docker credential helper {helper!r} is configured but broken: it "
                f"returned malformed (non-JSON) output for {server!r} ({exc})"
            ) from exc
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


def remote_run_script(
    batch_sizes: list[int],
    train_iters: int,
    docker_image: str,
    *,
    data_prefix_name: str = "cppmega_1024_smoke_mix_train",
    seq_data_prefixes: list[tuple[int, str, int, int]] | None = None,
    graph_capacity: tuple[int, int] | None = None,
    fp8_recipe: str = "off",
    disable_nvrtc: bool = False,
    enable_dsa_patch: bool = True,
    save_checkpoint: bool = False,
    save_interval: int | None = None,
    save_model_only: bool = True,
    load_checkpoint_remote: str | None = None,
    load_model_only: bool = True,
    bundle_root: str = "/data/cppmega_bundle",
    tokenizer_model: str = "/data/cppmega_bundle/tokenizer",
    run_id: str = "nebius-h200-sweep",
) -> str:
    validate_docker_image_digest(docker_image)
    if train_iters <= 0:
        raise ValueError("train_iters must be positive")
    if not enable_dsa_patch:
        raise ValueError(
            "production graph auxiliary contract requires the fused DSA patch"
        )
    dsa_profile = get_run_profile("h200_cpp_world_mini")
    dsa_profile.model.dense = False
    dsa_profile.spec_module = "cppmega.megatron.nam56r_full_spec"
    dsa_profile.spec_function = "build_cppmega_nam56r_full_stack_spec"
    dsa_native_args = shlex.split(dsa_profile.native_args_fragment())
    if "--experimental-attention-variant" not in dsa_native_args:
        raise ValueError("H200 graph auxiliary launcher requires native DSA args")
    dsa_args = " ".join(shlex.quote(value) for value in dsa_native_args)
    dsa_spec_args = " ".join(
        shlex.quote(value)
        for value in (dsa_profile.spec_module, dsa_profile.spec_function)
    )
    batches = " ".join(str(v) for v in batch_sizes)
    if seq_data_prefixes is None:
        if graph_capacity is None:
            raise ValueError("graph_capacity is required for a single-prefix sweep")
        tests = [(1024, data_prefix_name, *graph_capacity)]
    else:
        tests = seq_data_prefixes
    for seq, name, max_edges, max_chunks in tests:
        if seq <= 0 or not name or max_edges <= 0 or max_chunks <= 0:
            raise ValueError(f"invalid sequence graph capacity spec: {(seq, name, max_edges, max_chunks)}")
    test_lines = "\n".join(
        f"          {shlex.quote(':'.join(map(str, (seq, name, max_edges, max_chunks))))}"
        for seq, name, max_edges, max_chunks in tests
    )
    eval_interval = min(train_iters, 100)
    effective_save_interval = save_interval or train_iters
    if save_interval is not None and not save_checkpoint:
        raise ValueError("save_interval requires save_checkpoint")
    if save_checkpoint and not 0 < effective_save_interval <= train_iters:
        raise ValueError("save_interval must be between 1 and train_iters")
    checkpoint_lines = ["            CHECKPOINT_ARGS=()"]
    if load_checkpoint_remote:
        checkpoint_lines.append(
            f"            CHECKPOINT_ARGS+=(--load {shlex.quote(load_checkpoint_remote)})"
        )
        if load_model_only:
            checkpoint_lines.append("            CHECKPOINT_ARGS+=(--no-load-optim --no-load-rng)")
    if save_checkpoint:
        checkpoint_lines.extend(
            [
                "            CHECKPOINT_ROOT=/data/cppmega_h200_checkpoints/seq_${SEQ}_bs_${BS}",
                "            mkdir -p \\$CHECKPOINT_ROOT",
                f"            CHECKPOINT_ARGS+=(--save \\$CHECKPOINT_ROOT --save-interval {effective_save_interval})",
            ]
        )
        if save_model_only:
            checkpoint_lines.append("            CHECKPOINT_ARGS+=(--no-save-optim --no-save-rng)")
    else:
        checkpoint_lines.append("            # Checkpointing intentionally disabled for this sweep.")
    checkpoint_block = "\n".join(checkpoint_lines) + "\n"
    preflight_dsa_arg = (
        "          --enable-dsa-patch \\\n" if enable_dsa_patch else ""
    )
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
        export NVTE_DISABLE_NVRTC="{1 if disable_nvrtc else 0}"
        export CPPMEGA_STRUCTURE_ENABLED="${{CPPMEGA_STRUCTURE_ENABLED:-1}}"
        export CPPMEGA_GRAPH_ROUTES_ENABLED="${{CPPMEGA_GRAPH_ROUTES_ENABLED:-1}}"
        export CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS=0
        export CPPMEGA_DSA_PATCH_ENABLED="{1 if enable_dsa_patch else 0}"
        export CPPMEGA_DSA_GRAPH_AUX_ENABLED=1
        export CPPMEGA_DSA_GRAPH_AUX_WEIGHT=1
        export CPPMEGA_DSA_INDEXER_LOSS_COEFF=0.001
        export CPPMEGA_DSA_SKIP_INDEXER_LOSS=0
        export CPPMEGA_H200_DSA_GRAPH_RECEIPTS=1
        export CPPMEGA_BUNDLE_ROOT={shlex.quote(bundle_root)}
        export CPPMEGA_TOKENIZER_MODEL={shlex.quote(tokenizer_model)}
        mkdir -p "$TRITON_CACHE_DIR" /data/cppmega_h200_results

        python - <<'PY'
        import importlib
        import json
        import os
        import torch
        from cppmega.megatron.graph_route_attention_bias_patch import apply_graph_route_attention_bias_patch
        from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch

        apply_te_checkpoint_kwarg_patch()
        if os.environ.get("CPPMEGA_DSA_PATCH_ENABLED", "0") == "1":
            from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
            apply_dsa_indexer_fused_patch()
        apply_graph_route_attention_bias_patch()

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
        if not report["megatron.core.utils.get_batch_on_this_tp_rank"]:
            raise RuntimeError(
                "cppmega stack: get_batch_on_this_tp_rank patch missing: " + json.dumps(report)
            )
        PY

        TEST_SPECS=(
{test_lines}
        )
        IFS=: read -r PREFLIGHT_SEQ PREFLIGHT_PREFIX_NAME PREFLIGHT_MAX_EDGES PREFLIGHT_MAX_CHUNKS <<< "${{TEST_SPECS[0]}}"
        DATA_PREFIX="$CPPMEGA_BUNDLE_ROOT/${{PREFLIGHT_PREFIX_NAME}}"
        export CPPMEGA_GRAPH_MAX_EDGES="$PREFLIGHT_MAX_EDGES"
        export CPPMEGA_GRAPH_MAX_CHUNKS="$PREFLIGHT_MAX_CHUNKS"
        if [[ ! -s "${{DATA_PREFIX}}.bin" || ! -s "${{DATA_PREFIX}}.idx" || ! -s "${{DATA_PREFIX}}.json" ]]; then
          echo "CPPMEGA_H200_PREFLIGHT_STATUS=FAIL reason=missing_data_prefix prefix=${{DATA_PREFIX}}" | tee -a /data/cppmega_h200_results/summary.log
          exit 2
        fi
        python /opt/cppmega/scripts/h200_megatron_preflight.py \
          --bundle-root "$CPPMEGA_BUNDLE_ROOT" \
          --data-prefix "$DATA_PREFIX" \
          --tokenizer-model "$CPPMEGA_TOKENIZER_MODEL" \
          --run-id {shlex.quote(run_id)} \
          --sequence-length "$PREFLIGHT_SEQ" \
          --micro-batch-size 1 \
          --fp8-recipe {shlex.quote(fp8_recipe)} \
{preflight_dsa_arg}\
          --output /data/cppmega_h200_results/h200_preflight.json
        echo "CPPMEGA_H200_PREFLIGHT_STATUS=PASS" | tee -a /data/cppmega_h200_results/summary.log

        for SPEC in "${{TEST_SPECS[@]}}"; do
          IFS=: read -r SEQ DATA_PREFIX_NAME GRAPH_MAX_EDGES GRAPH_MAX_CHUNKS <<< "$SPEC"
          DATA_PREFIX="$CPPMEGA_BUNDLE_ROOT/${{DATA_PREFIX_NAME}}"
          export DATA_PREFIX
          export CPPMEGA_GRAPH_MAX_EDGES="$GRAPH_MAX_EDGES"
          export CPPMEGA_GRAPH_MAX_CHUNKS="$GRAPH_MAX_CHUNKS"
          if [[ ! -s "${{DATA_PREFIX}}.bin" || ! -s "${{DATA_PREFIX}}.idx" || ! -s "${{DATA_PREFIX}}.json" ]]; then
            echo "CPPMEGA_TEST_RESULT seq=${{SEQ}} status=FAIL reason=missing_data_prefix prefix=${{DATA_PREFIX}}" | tee -a /data/cppmega_h200_results/summary.log
            exit 2
          fi
          CAPACITY_RECEIPT="/data/cppmega_h200_results/graph_capacity_seq_${{SEQ}}.json"
          python - "$DATA_PREFIX" "$SEQ" "$GRAPH_MAX_EDGES" "$GRAPH_MAX_CHUNKS" "$CAPACITY_RECEIPT" <<'PYCAP'
        import sys
        from pathlib import Path
        from scripts.h200_megatron_preflight import write_graph_capacity_receipt

        receipt = write_graph_capacity_receipt(
            Path(sys.argv[1]),
            sequence_length=int(sys.argv[2]),
            output=Path(sys.argv[5]),
        )
        expected = (int(sys.argv[3]), int(sys.argv[4]))
        actual = (int(receipt["graph_max_edges"]), int(receipt["graph_max_chunks"]))
        if actual != expected:
            raise RuntimeError(f"launcher/remote graph capacity mismatch: {{expected}} != {{actual}}")
        PYCAP
          echo "CPPMEGA_GRAPH_CAPACITY seq=${{SEQ}} max_edges=${{GRAPH_MAX_EDGES}} max_chunks=${{GRAPH_MAX_CHUNKS}} receipt=${{CAPACITY_RECEIPT}}" | tee -a /data/cppmega_h200_results/summary.log
          SEQ_OOM=0
          for BS in {batches}; do
          LOG="/data/cppmega_h200_results/seq_${{SEQ}}_bs_${{BS}}.log"
          NVSMI="/data/cppmega_h200_results/seq_${{SEQ}}_bs_${{BS}}.nvsmi.csv"
          if [[ "$SEQ_OOM" == 1 ]]; then
            echo "CPPMEGA_BATCH_RESULT seq=${{SEQ}} batch=${{BS}} status=SKIP reason=previous_oom" | tee "$LOG" | tee -a /data/cppmega_h200_results/summary.log
            continue
          fi
          echo "[container] starting seq=${{SEQ}} batch=${{BS}} prefix=${{DATA_PREFIX}}" | tee "$LOG"
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

        from cppmega.megatron.graph_route_attention_bias_patch import apply_graph_route_attention_bias_patch
        from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch

        apply_te_checkpoint_kwarg_patch()
        if os.environ.get('CPPMEGA_DSA_PATCH_ENABLED', '0') == '1':
            from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
            apply_dsa_indexer_fused_patch()
        apply_graph_route_attention_bias_patch()

        if os.environ.get('CPPMEGA_STRUCTURE_ENABLED', '0') == '1':
            import cppmega.megatron.structure_dataset_patch  # noqa: F401

        @atexit.register
        def _cppmega_distributed_shutdown():
            try:
                import torch
                import torch.distributed as dist

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f'CPPMEGA_DISTRIBUTED_SHUTDOWN_ERROR {{exc}}', flush=True)

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
            cat >\\\"\\$WORKDIR/hybrid_builders.py\\\" <<'PY'
        from cppmega.megatron.mamba_builder import cppmega_mamba_builder as hybrid_builder
        PY
            eval \\\"\\$(python -m cppmega.recipes.run_profiles shell h200_cpp_world_mini \\
              --seq-length ${{SEQ}} \\
              --micro-batch-size ${{BS}} \\
              --global-batch-size ${{BS}} \\
              --train-iters {train_iters} \\
              --fp8-recipe {fp8_recipe})\\\"
            export CPPMEGA_DSA_GRAPH_AUX_ENABLED=1
            export CPPMEGA_DSA_GRAPH_AUX_WEIGHT=1
            export CPPMEGA_DSA_INDEXER_LOSS_COEFF=0.001
            export CPPMEGA_DSA_SKIP_INDEXER_LOSS=0
            unset CPPMEGA_DENSE_GQA

            DATA_ARGS=(--data-path 1.0 \\\"\\$DATA_PREFIX\\")
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

            DSA_ARGS=({dsa_args})
            ATTN_ARGS=(--attention-backend \\\"\\$CPPMEGA_ATTN_BACKEND\\\")
            if [[ \\\"\\$CPPMEGA_USE_FLASH_ATTN\\\" == 1 ]]; then
              ATTN_ARGS=(--use-flash-attn \\\"\\${{ATTN_ARGS[@]}}\\\")
            fi
            export NVTE_DEBUG=\\\"\\${{NVTE_DEBUG:-1}}\\\"
            export NVTE_DEBUG_LEVEL=\\\"\\${{NVTE_DEBUG_LEVEL:-2}}\\\"
            FP8_ARGS=()
            if [[ \\\"\\$CPPMEGA_FP8_RECIPE\\\" == tensorwise ]]; then
              FP8_ARGS+=(--fp8-format \\\"\\$CPPMEGA_FP8_FORMAT\\\" --fp8-recipe tensorwise --fp8-amax-history-len 16 --fp8-amax-compute-algo max)
            elif [[ \\\"\\$CPPMEGA_FP8_RECIPE\\\" != off ]]; then
              echo \\\"Unsupported H200 sweep FP8 recipe: \\$CPPMEGA_FP8_RECIPE\\\" >&2
              exit 2
            fi
            RECOMPUTE_ARGS=(--recompute-granularity selective --recompute-modules mlp)
{checkpoint_block}

            python -m torch.distributed.run --nproc_per_node=1 \\\"\\$WORKDIR/pretrain_mamba.py\\\" \\
              \\\"\\${{DATA_ARGS[@]}}\\\" \\
              --tokenizer-type HuggingFaceTokenizer \\
              --tokenizer-model \\\"\\$CPPMEGA_TOKENIZER_MODEL\\\" \\
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
              \\\"\\${{DSA_ARGS[@]}}\\\" \\
              --seq-length ${{SEQ}} \\
              --max-position-embeddings ${{SEQ}} \\
              --micro-batch-size ${{BS}} \\
              --global-batch-size ${{BS}} \\
              --train-iters {train_iters} \\
              --eval-interval {eval_interval} \\
              --eval-iters 1 \\
              --lr \\\"\\$CPPMEGA_LR\\\" \\
              --min-lr \\\"\\$CPPMEGA_MIN_LR\\\" \\
              --lr-decay-style constant \\
              --position-embedding-type rope \\
              --no-rope-fusion \\
              --normalization RMSNorm \\
              --disable-bias-linear \\
              --bf16 \\
              \\\"\\${{FP8_ARGS[@]}}\\\" \\
              --use-mcore-models \\
              --transformer-impl transformer_engine \\
              \\\"\\${{ATTN_ARGS[@]}}\\\" \\
              --spec {dsa_spec_args} \\
              --cross-entropy-loss-fusion \\
              --cross-entropy-fusion-impl te \\
              \\\"\\${{RECOMPUTE_ARGS[@]}}\\\" \\
              --clip-grad 1.0 \\
              \\\"\\${{OPTIMIZER_ARGS[@]}}\\\" \\
              --rerun-mode disabled \\
              \\\"\\${{CHECKPOINT_ARGS[@]}}\\\" \\
              --log-interval 1
          " >>"$LOG" 2>&1
          status=$?
          kill "$NVSMI_PID" 2>/dev/null || true
          wait "$NVSMI_PID" 2>/dev/null || true
          set -e
          peak="$(awk -F, '{{ if ($2+0 > peak) peak=$2+0 }} END {{ print peak+0 }}' "$NVSMI")"
          echo "CPPMEGA_NVIDIA_SMI_PEAK seq=${{SEQ}} batch=${{BS}} peak_used_mib=${{peak}}" | tee -a "$LOG"
          if [[ "$status" != 0 ]]; then
            if grep -qiE 'out of memory|OutOfMemoryError|cuda error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|failed to CUDA calloc|cuda calloc async|CUDA calloc async' "$LOG"; then
              echo "CPPMEGA_BATCH_RESULT seq=${{SEQ}} batch=${{BS}} status=OOM exit=${{status}}" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
              echo "CPPMEGA_BATCH_OOM seq=${{SEQ}} batch=${{BS}}" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
              SEQ_OOM=1
              continue
            fi
            if grep -qE 'iteration[[:space:]]+{train_iters}/[[:space:]]+{train_iters}' "$LOG" && \\
               grep -qE 'validation loss at iteration {train_iters}' "$LOG" && \\
               grep -q 'transformer_engine::rtc::Kernel::~Kernel' "$LOG" && \\
               grep -q 'SIGSEGV' "$LOG"; then
              echo "CPPMEGA_BATCH_RESULT seq=${{SEQ}} batch=${{BS}} status=FAIL_TE_CLEANUP_SIGSEGV exit=${{status}}" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
              exit "$status"
            fi
            echo "CPPMEGA_BATCH_RESULT seq=${{SEQ}} batch=${{BS}} status=FAIL exit=${{status}}" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
            exit "$status"
          fi
          LOSS_RECEIPT="/data/cppmega_h200_results/seq_${{SEQ}}_bs_${{BS}}_loss_gate.json"
          if ! python - "$LOG" "{train_iters}" "$LOSS_RECEIPT" <<'PYLOSS'
        import sys
        from pathlib import Path
        from scripts.h200_megatron_preflight import write_training_loss_receipt

        write_training_loss_receipt(
            Path(sys.argv[1]),
            expected_iteration=int(sys.argv[2]),
            output=Path(sys.argv[3]),
            expected_dsa_coefficient=0.001,
        )
        PYLOSS
          then
            echo "CPPMEGA_BATCH_RESULT seq=${{SEQ}} batch=${{BS}} status=FAIL reason=finite_loss_gate" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
            exit 4
          fi
          echo "CPPMEGA_BATCH_RESULT seq=${{SEQ}} batch=${{BS}} status=OK loss_receipt=${{LOSS_RECEIPT}}" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
          done
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


def instance_delete_allowed(*, keep_instance: bool, retrieval_succeeded: bool) -> bool:
    return not keep_instance and retrieval_succeeded


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
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--hash-jobs", type=int, default=4)
    parser.add_argument("--ghcr-user", default=None)
    parser.add_argument("--ghcr-token-file", type=Path, default=None)
    parser.add_argument("--no-ghcr-auth", action="store_true")
    parser.add_argument("--sidecar-prefix", type=Path, default=DEFAULT_SIDECAR_PREFIX)
    parser.add_argument(
        "--sidecar-prefixes",
        default=None,
        help=(
            "Comma-separated seq=prefix entries for one multi-seq sweep, e.g. "
            "1024=/path/train1024,2048=/path/train2048. Overrides --sidecar-prefix."
        ),
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--batch-sizes", default="256,512,1024")
    parser.add_argument("--train-iters", type=int, default=3)
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save Megatron checkpoints under /data/cppmega_h200_checkpoints and copy them back.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Checkpoint save interval when --save-checkpoint is set. Defaults to --train-iters.",
    )
    parser.add_argument(
        "--save-full-state",
        action="store_true",
        help="Save optimizer/RNG state too. By default only model weights are saved for quick testing.",
    )
    parser.add_argument(
        "--load-checkpoint-local",
        type=Path,
        default=None,
        help=(
            "Upload a local Megatron checkpoint root to /data/cppmega_load_checkpoint "
            "and load it on the remote run."
        ),
    )
    parser.add_argument(
        "--load-checkpoint-remote",
        default=None,
        help="Load an already-present checkpoint root on the remote instance.",
    )
    parser.add_argument(
        "--load-full-state",
        action="store_true",
        help="Load optimizer/RNG state too. By default load model weights only for smoke tests.",
    )
    parser.add_argument(
        "--remote-timeout-s",
        type=int,
        default=7200,
        help="Timeout for the remote training command.",
    )
    parser.add_argument("--fp8-recipe", choices=("off", "tensorwise"), default="off")
    parser.add_argument(
        "--disable-nvrtc",
        action="store_true",
        help=(
            "Set NVTE_DISABLE_NVRTC=1 inside the container to avoid the TE RTC teardown path. "
            "This is automatic for --fp8-recipe tensorwise unless --enable-nvrtc is set."
        ),
    )
    parser.add_argument(
        "--enable-nvrtc",
        action="store_true",
        help=(
            "Keep TE NVRTC enabled for FP8 tensorwise sweeps. This is faster on the current "
            "H200 image but reproduces the TE KernelManager cleanup segfault with TE 2.16.0."
        ),
    )
    parser.add_argument(
        "--enable-dsa-patch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install the required fused DSA graph-objective patch.",
    )
    parser.add_argument("--keep-instance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    validate_docker_image_digest(args.docker_image)
    if args.train_iters <= 0 or args.hash_jobs <= 0:
        raise ValueError("train iters and hash jobs must be positive")
    batches = parse_batches(args.batch_sizes)
    effective_disable_nvrtc = args.disable_nvrtc or (
        args.fp8_recipe == "tensorwise" and not args.enable_nvrtc
    )
    if args.sidecar_prefixes:
        seq_prefixes: list[tuple[int, Path]] = []
        for raw in args.sidecar_prefixes.split(","):
            if not raw.strip():
                continue
            if "=" not in raw:
                raise ValueError(f"--sidecar-prefixes entries must be seq=path, got {raw!r}")
            seq_raw, path_raw = raw.split("=", 1)
            seq_prefixes.append((int(seq_raw), Path(path_raw)))
        if not seq_prefixes:
            raise ValueError("--sidecar-prefixes did not contain any entries")
    else:
        seq_prefixes = [(1024, args.sidecar_prefix)]
    bundle_root = args.bundle_root.resolve()
    manifest, _bundle_artifacts = _validate_bundle(bundle_root, args.hash_jobs)
    declared_prefixes = {
        (bundle_root / str(result["prefix"])).resolve(): str(result["prefix"])
        for result in manifest["bucket_results"]
    }
    remote_prefixes: list[tuple[int, str, int, int]] = []
    for sequence_length, prefix in seq_prefixes:
        relative = declared_prefixes.get(prefix.resolve())
        if relative is None:
            raise ValueError(f"selected prefix is not declared by bundle: {prefix}")
        capacity = derive_graph_capacity_receipt(
            prefix,
            sequence_length=sequence_length,
        )
        remote_prefixes.append(
            (
                sequence_length,
                relative,
                int(capacity["graph_max_edges"]),
                int(capacity["graph_max_chunks"]),
            )
        )
    tokenizer_relative = str(manifest["tokenizer"]["path"])
    bundle_tokenizer = (bundle_root / tokenizer_relative).resolve()
    if args.tokenizer_dir is not None and args.tokenizer_dir.resolve() != bundle_tokenizer:
        raise ValueError(
            "--tokenizer-dir must be the descriptor-bound tokenizer inside --bundle-root"
        )
    if args.load_checkpoint_local and args.load_checkpoint_remote:
        raise ValueError("--load-checkpoint-local and --load-checkpoint-remote are mutually exclusive")
    load_checkpoint_remote = args.load_checkpoint_remote
    if args.load_checkpoint_local:
        load_checkpoint_remote = "/data/cppmega_load_checkpoint"
    pubkey_path = args.ssh_pubkey or Path(str(args.ssh_key) + ".pub")
    if not pubkey_path.exists():
        raise FileNotFoundError(f"ssh public key not found: {pubkey_path}")
    ssh_pubkey = pubkey_path.read_text().strip()

    if args.dry_run:
        print(f"parent_id={args.parent_id}")
        print(f"sidecar_prefixes={seq_prefixes}")
        print(f"bundle_root={bundle_root}")
        print(f"tokenizer_dir={bundle_tokenizer}")
        print(f"batches={batches}")
        print(
            remote_run_script(
                batches,
                args.train_iters,
                args.docker_image,
                seq_data_prefixes=remote_prefixes,
                fp8_recipe=args.fp8_recipe,
                disable_nvrtc=effective_disable_nvrtc,
                enable_dsa_patch=args.enable_dsa_patch,
                save_checkpoint=args.save_checkpoint,
                save_interval=args.save_interval,
                save_model_only=not args.save_full_state,
                load_checkpoint_remote=load_checkpoint_remote,
                load_model_only=not args.load_full_state,
                tokenizer_model=f"/data/cppmega_bundle/{tokenizer_relative}",
                run_id=args.instance_name,
            )[:4000]
        )
        return 0

    instance_id: str | None = None
    retrieval_succeeded = False
    try:
        with tempfile.TemporaryDirectory(prefix="cppmega-h200-sweep-") as tmp:
            overlay_tar = Path(tmp) / "cppmega_overlay.tgz"
            sidecar_tar = Path(tmp) / "cppmega_sidecar.tgz"
            ghcr_auth_tar = Path(tmp) / "cppmega_ghcr_auth.tgz"
            load_checkpoint_tar = Path(tmp) / "cppmega_load_checkpoint.tgz"
            make_overlay_tar(overlay_tar)
            archived_prefixes, archived_tokenizer = make_bundle_tar(
                bundle_root,
                [prefix for _, prefix in seq_prefixes],
                sidecar_tar,
                hash_jobs=args.hash_jobs,
            )
            if archived_prefixes != [prefix for _, prefix, _, _ in remote_prefixes]:
                raise RuntimeError("bundle archive prefix layout drifted after validation")
            if archived_tokenizer != tokenizer_relative:
                raise RuntimeError("bundle archive tokenizer layout drifted after validation")
            if args.load_checkpoint_local:
                make_checkpoint_tar(args.load_checkpoint_local, load_checkpoint_tar)
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
            if args.load_checkpoint_local:
                stream_tar_to_remote(args, ip, load_checkpoint_tar, "/data/cppmega_load_checkpoint")
            if has_ghcr_auth:
                stream_tar_to_remote(args, ip, ghcr_auth_tar, "/data")
            script = remote_run_script(
                batches,
                args.train_iters,
                args.docker_image,
                seq_data_prefixes=remote_prefixes,
                fp8_recipe=args.fp8_recipe,
                disable_nvrtc=effective_disable_nvrtc,
                enable_dsa_patch=args.enable_dsa_patch,
                save_checkpoint=args.save_checkpoint,
                save_interval=args.save_interval,
                save_model_only=not args.save_full_state,
                load_checkpoint_remote=load_checkpoint_remote,
                load_model_only=not args.load_full_state,
                tokenizer_model=f"/data/cppmega_bundle/{tokenizer_relative}",
                run_id=args.instance_name,
            )
            ssh(args, ip, f"cat > /data/run_cppmega_h200_sweep.sh <<'EOF'\n{script}\nEOF\nchmod +x /data/run_cppmega_h200_sweep.sh")
            try:
                ssh(args, ip, "bash /data/run_cppmega_h200_sweep.sh", timeout=args.remote_timeout_s)
            finally:
                # A training exception may be propagating through this finally; detect it
                # so the transfers below stay loud without masking the primary error.
                training_failed = sys.exc_info()[0] is not None
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
                results_failure: str | None = None
                try:
                    log_proc = run(scp_cmd, check=False)
                except FileNotFoundError:
                    results_failure = "scp is unavailable; results were NOT copied back"
                else:
                    if log_proc.returncode != 0:
                        results_failure = (
                            f"results scp exited {log_proc.returncode}; results were NOT "
                            f"copied to {out_dir}"
                        )
                ckpt_failure: str | None = None
                if args.save_checkpoint:
                    ckpt_dir = ROOT / "outputs" / "checkpoints" / args.instance_name
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_scp_cmd = [
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
                        f"{args.ssh_user}@{ip}:/data/cppmega_h200_checkpoints/.",
                        str(ckpt_dir),
                    ]
                    # The user explicitly asked to save+copy checkpoints, so a failed or
                    # absent transfer is silent data loss -> raise. But if a training
                    # exception is already propagating, do not mask it: be very loud instead.
                    try:
                        ckpt_proc = run(ckpt_scp_cmd, check=False)
                    except FileNotFoundError:
                        ckpt_failure = "scp is unavailable; checkpoints were NOT copied back"
                    else:
                        if ckpt_proc.returncode != 0:
                            ckpt_failure = (
                                f"scp exited {ckpt_proc.returncode}; checkpoints were NOT "
                                f"copied to {ckpt_dir}"
                            )
                retrieval_succeeded = results_failure is None and ckpt_failure is None
                if not retrieval_succeeded:
                    failures = "; ".join(
                        failure
                        for failure in (results_failure, ckpt_failure)
                        if failure is not None
                    )
                    message = f"artifact retrieval failed: {failures}"
                    print(f"[nebius-sweep] ERROR: {message}", file=sys.stderr)
                    if not training_failed:
                        raise RuntimeError(message)
    finally:
        if instance_id:
            if instance_delete_allowed(
                keep_instance=args.keep_instance,
                retrieval_succeeded=retrieval_succeeded,
            ):
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
            elif not args.keep_instance:
                print(
                    f"[nebius-sweep] preserving instance {instance_id}: "
                    "results/checkpoints were not fully retrieved",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
