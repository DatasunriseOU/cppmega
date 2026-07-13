#!/usr/bin/env python3
"""Run the cppmega C++ world model curriculum on one Nebius H200.

Unlike the batch-sweep smoke runner, this script keeps one remote machine alive
and runs staged training sequentially:

    1024 -> 2048 -> 4096 -> 8192 -> 16384

Each stage saves a full Megatron checkpoint under its own stage directory and
the next stage loads the previous checkpoint as pretrained weights.  The data
iterator is reset at each context bucket, so stage train-iters and save-intervals
are local to that bucket rather than cumulative across different datasets.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "scripts" / "nebius_h200_megatron_cpp_world_sweep.py"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("cppmega_nebius_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sweep helper from {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_sweep_module()


@dataclass(frozen=True)
class Stage:
    index: int
    seq: int
    batch: int
    micro_batch: int
    iters: int
    prefix: Path

    @property
    def prefix_name(self) -> str:
        return self.prefix.name

    @property
    def remote_checkpoint_root(self) -> str:
        return (
            f"/data/cppmega_curriculum_checkpoints/"
            f"stage_{self.index:02d}_seq_{self.seq}_gbs_{self.batch}_mbs_{self.micro_batch}"
        )


def _parse_stage(raw: str, index: int) -> Stage:
    parts = raw.split("=")
    if len(parts) not in (4, 5):
        raise ValueError(
            "stage entries must be seq=batch=iters=prefix or "
            "seq=global_batch=micro_batch=iters=prefix, "
            f"got {raw!r}"
        )
    if len(parts) == 4:
        seq_s, batch_s, iters_s, prefix_s = parts
        micro_batch_s = batch_s
    else:
        seq_s, batch_s, micro_batch_s, iters_s, prefix_s = parts
    prefix = Path(prefix_s).expanduser().resolve()
    for suffix in (".bin", ".idx", ".json"):
        path = prefix.with_suffix(suffix)
        if not path.exists():
            raise FileNotFoundError(path)
    return Stage(
        index=index,
        seq=int(seq_s),
        batch=int(batch_s),
        micro_batch=int(micro_batch_s),
        iters=int(iters_s),
        prefix=prefix,
    )


def _default_stages() -> list[Stage]:
    base = ROOT.parent / "cppmega.mlx" / "outputs" / "megatron_ready"
    specs = [
        (
            1024,
            192,
            192,
            1421,
            base / "cppmega_reindexed_seq1024_lossmask_graph_train",
        ),
        (
            2048,
            96,
            96,
            1686,
            base / "cppmega_reindexed_seq2048_lossmask_graph_train",
        ),
        (
            4096,
            40,
            40,
            2311,
            base / "cppmega_reindexed_seq4096_lossmask_graph_train",
        ),
        # H200-observed defaults. Long-context stages keep a larger global batch
        # but split it into smaller microbatches; the previous micro==global
        # runs OOMed on TE cross-entropy/logits materialization before graph
        # routes became the limiting factor.
        (
            8192,
            16,
            4,
            2756,
            base / "cppmega_reindexed_seq8192_lossmask_graph_train",
        ),
        (
            16384,
            8,
            2,
            2391,
            base / "cppmega_reindexed_seq16384_lossmask_graph_train",
        ),
    ]
    return [
        Stage(index=i, seq=seq, batch=batch, micro_batch=micro_batch, iters=iters, prefix=prefix)
        for i, (seq, batch, micro_batch, iters, prefix) in enumerate(specs, 1)
    ]


def _assert_prefix_contract(stages: list[Stage]) -> None:
    required_sidecars = {
        "loss_mask",
        "doc_ids",
        "token_domain_ids",
        "token_role_ids",
        "token_entity_ids",
        "token_scope_ids",
        "token_confidence_ids",
        "token_structure_ids",
        "token_dep_levels",
        "token_ast_depth",
        "token_sibling_index",
        "token_ast_node_type",
        "token_symbol_ids",
        "token_call_targets",
        "token_type_refs",
        "token_def_use",
        "token_change_mask_pre",
        "token_change_mask_post",
    }
    required_graph_schema = "cppmega_graph_routes_v2"
    for stage in stages:
        manifest = json.loads(stage.prefix.with_suffix(".json").read_text())
        side = set(manifest.get("side_channel_paths", {}))
        graph_schema = manifest.get("graph_sidecar_schema")
        missing = sorted(required_sidecars - side)
        if missing:
            raise KeyError(f"{stage.prefix}: missing required sidecars {missing}")
        if graph_schema != required_graph_schema:
            raise ValueError(
                f"{stage.prefix}: graph_sidecar_schema={graph_schema!r}, "
                f"expected {required_graph_schema!r}"
            )
        source_platform = manifest.get("source_platform_sidecar")
        if not isinstance(source_platform, dict) or source_platform.get(
            "schema"
        ) != "cppmega_source_platform_v1":
            raise ValueError(
                f"{stage.prefix}: compact source platform sidecar missing or invalid"
            )


def _make_curriculum_manifest(stages: list[Stage], path: Path) -> None:
    payload = {
        "schema": "cppmega_h200_curriculum_v1",
        "stages": [
            {
                "index": stage.index,
                "seq": stage.seq,
                "batch": stage.batch,
                "global_batch": stage.batch,
                "micro_batch": stage.micro_batch,
                "stage_iters": stage.iters,
                "prefix": str(stage.prefix),
                "prefix_name": stage.prefix_name,
                "remote_checkpoint_root": stage.remote_checkpoint_root,
            }
            for stage in stages
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_curriculum_tar(stages: list[Stage], path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="cppmega-curriculum-manifest-") as raw:
        tmp = Path(raw)
        manifest = tmp / "curriculum_manifest.json"
        _make_curriculum_manifest(stages, manifest)
        with tarfile.open(path, "w:gz") as tf:
            tf.add(manifest, arcname="cppmega_curriculum/curriculum_manifest.json")


def _make_checkpoint_tar(checkpoint_root: Path, path: Path) -> None:
    checkpoint_root = checkpoint_root.expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise NotADirectoryError(checkpoint_root)
    latest = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not latest.is_file():
        raise FileNotFoundError(latest)
    with tarfile.open(path, "w:gz") as tf:
        tf.add(
            checkpoint_root,
            arcname=f"cppmega_curriculum_checkpoints/{checkpoint_root.name}",
        )


def _remote_script(
    stages: list[Stage],
    *,
    docker_image: str,
    fp8_recipe: str,
    initial_checkpoint_root: str = "",
    initial_cum_iters: int = 0,
) -> str:
    stage_lines = "\n".join(
        "          "
        + shlex.quote(
            ":".join(
                (
                    str(stage.index),
                    str(stage.seq),
                    str(stage.batch),
                    str(stage.micro_batch),
                    str(stage.iters),
                    stage.prefix_name,
                    stage.remote_checkpoint_root,
                )
            )
        )
        for stage in stages
    )
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        sudo mkdir -p /data/cppmega_h200_results /data/cppmega_overlay
        sudo chown -R "$USER":"$USER" /data

        if ! command -v docker >/dev/null 2>&1; then
          sudo apt-get update
          sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \\
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
          sudo docker login ghcr.io \\
            -u "$(cat /data/cppmega_auth/ghcr_user)" \\
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
        export NVTE_DISABLE_NVRTC=1
        export CPPMEGA_STRUCTURE_ENABLED=1
        export CPPMEGA_DOMAIN_EMBEDDING_ENABLED=1
        export CPPMEGA_GRAPH_ROUTES_ENABLED=1
        export CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS="${{CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS:-1}}"
        export CPPMEGA_GRAPH_MAX_EDGES="${{CPPMEGA_GRAPH_MAX_EDGES:-256}}"
        export CPPMEGA_GRAPH_MAX_CHUNKS="${{CPPMEGA_GRAPH_MAX_CHUNKS:-256}}"
        mkdir -p "$TRITON_CACHE_DIR" /data/cppmega_h200_results /data/cppmega_curriculum_checkpoints

        python - <<'PY'
        import importlib
        import json
        import torch
        from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
        from cppmega.megatron.graph_route_attention_bias_patch import apply_graph_route_attention_bias_patch
        from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch

        apply_te_checkpoint_kwarg_patch()
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
        report["megatron.core.utils.get_batch_on_this_tp_rank"] = hasattr(core_utils, "get_batch_on_this_tp_rank")
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

        STAGES=(
        {stage_lines}
        )

        PREV_CHECKPOINT_ROOT={shlex.quote(initial_checkpoint_root)}
        PREV_CUM_ITERS={int(initial_cum_iters)}
        for SPEC in "${{STAGES[@]}}"; do
          IFS=: read -r STAGE_IDX SEQ BS MBS STAGE_ITERS DATA_PREFIX_NAME CHECKPOINT_ROOT <<< "$SPEC"
          CUM_ITERS=$((PREV_CUM_ITERS + STAGE_ITERS))
          TARGET_ITERS=$STAGE_ITERS
          DATA_PREFIX="/data/cppmega_sidecar/${{DATA_PREFIX_NAME}}"
          LOG="/data/cppmega_h200_results/stage_${{STAGE_IDX}}_seq_${{SEQ}}_gbs_${{BS}}_mbs_${{MBS}}.log"
          NVSMI="/data/cppmega_h200_results/stage_${{STAGE_IDX}}_seq_${{SEQ}}_gbs_${{BS}}_mbs_${{MBS}}.nvsmi.csv"
          export DATA_PREFIX
          echo "CPPMEGA_CURRICULUM_STAGE_START stage=${{STAGE_IDX}} seq=${{SEQ}} global_batch=${{BS}} micro_batch=${{MBS}} stage_iters=${{STAGE_ITERS}} target_iter=${{TARGET_ITERS}} cumulative_after=${{CUM_ITERS}} prefix=${{DATA_PREFIX}}" | tee "$LOG" | tee -a /data/cppmega_h200_results/summary.log
          if [[ ! -s "${{DATA_PREFIX}}.bin" || ! -s "${{DATA_PREFIX}}.idx" || ! -s "${{DATA_PREFIX}}.json" ]]; then
            echo "CPPMEGA_CURRICULUM_STAGE_RESULT stage=${{STAGE_IDX}} status=FAIL reason=missing_data_prefix" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
            exit 2
          fi

          (
            while true; do
              ts="$(date '+%Y-%m-%dT%H:%M:%S')"
              nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits |
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
            WORKDIR=\\$(mktemp -d /tmp/cppmega-curriculum.XXXXXX)
            trap 'rm -rf \\\"\\$WORKDIR\\\"' EXIT
            cat >\\\"\\$WORKDIR/pretrain_mamba.py\\\" <<'PYWRAP'
        from __future__ import annotations
        import atexit
        import os
        import runpy
        import sys

        from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
        from cppmega.megatron.graph_route_attention_bias_patch import apply_graph_route_attention_bias_patch
        from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch

        apply_te_checkpoint_kwarg_patch()
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
              --micro-batch-size ${{MBS}} \\
              --global-batch-size ${{BS}} \\
              --train-iters ${{TARGET_ITERS}} \\
              --fp8-recipe {fp8_recipe})\\\"

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

            GQA_ARGS=(--group-query-attention --num-query-groups \\\"\\$CPPMEGA_NUM_QUERY_GROUPS\\\" --kv-channels \\\"\\$CPPMEGA_KV_CHANNELS\\\" --swiglu --rotary-base 10000)
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
              echo \\\"Unsupported H200 curriculum FP8 recipe: \\$CPPMEGA_FP8_RECIPE\\\" >&2
              exit 2
            fi
            RECOMPUTE_ARGS=(--recompute-granularity selective --recompute-modules mlp)
            CHECKPOINT_ARGS=(--save \\\"$CHECKPOINT_ROOT\\\" --save-interval ${{TARGET_ITERS}})
            if [[ -n \\\"$PREV_CHECKPOINT_ROOT\\\" ]]; then
              CHECKPOINT_ARGS+=(--load \\\"$PREV_CHECKPOINT_ROOT\\\" --finetune --no-load-optim --no-load-rng --override-opt-param-scheduler)
            fi

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
              --seq-length ${{SEQ}} \\
              --max-position-embeddings ${{SEQ}} \\
              --micro-batch-size ${{MBS}} \\
              --global-batch-size ${{BS}} \\
              --train-iters ${{TARGET_ITERS}} \\
              --eval-interval 50000000 \\
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
              --spec cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec \\
              --cross-entropy-loss-fusion \\
              --cross-entropy-fusion-impl te \\
              \\\"\\${{RECOMPUTE_ARGS[@]}}\\\" \\
              --clip-grad 1.0 \\
              \\\"\\${{OPTIMIZER_ARGS[@]}}\\\" \\
              --no-check-for-nan-in-loss-and-grad \\
              --rerun-mode disabled \\
              \\\"\\${{CHECKPOINT_ARGS[@]}}\\\" \\
              --log-interval 1
          " >>"$LOG" 2>&1
          status=$?
          kill "$NVSMI_PID" 2>/dev/null || true
          wait "$NVSMI_PID" 2>/dev/null || true
          set -e
          peak="$(awk -F, '{{ if ($2+0 > peak) peak=$2+0 }} END {{ print peak+0 }}' "$NVSMI")"
          echo "CPPMEGA_NVIDIA_SMI_PEAK stage=${{STAGE_IDX}} seq=${{SEQ}} global_batch=${{BS}} micro_batch=${{MBS}} peak_used_mib=${{peak}}" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
          if [[ "$status" != 0 ]]; then
            echo "CPPMEGA_CURRICULUM_STAGE_RESULT stage=${{STAGE_IDX}} seq=${{SEQ}} global_batch=${{BS}} micro_batch=${{MBS}} status=FAIL exit=${{status}}" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
            exit "$status"
          fi
          if [[ ! -s "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" ]]; then
            echo "CPPMEGA_CURRICULUM_STAGE_RESULT stage=${{STAGE_IDX}} seq=${{SEQ}} batch=${{BS}} status=FAIL reason=missing_checkpoint" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
            exit 3
          fi
          echo "CPPMEGA_CURRICULUM_STAGE_RESULT stage=${{STAGE_IDX}} seq=${{SEQ}} global_batch=${{BS}} micro_batch=${{MBS}} status=OK checkpoint_root=${{CHECKPOINT_ROOT}}" | tee -a "$LOG" | tee -a /data/cppmega_h200_results/summary.log
          PREV_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
          PREV_CUM_ITERS="$CUM_ITERS"
        done
        INNER

        sudo docker run --gpus all --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
          -v /data:/data \\
          -v /data/cppmega_overlay:/overlay:ro \\
          {shlex.quote(docker_image)} \\
          bash /data/cppmega_h200_results/container_run.sh
        """
    )


def _scp_from_remote(args: argparse.Namespace, ip: str, remote_path: str, local_path: Path) -> int:
    local_path.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        f"{args.ssh_user}@{ip}:{remote_path.rstrip('/')}/.",
        str(local_path),
    ]
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[nebius-curriculum] $ {printable}", flush=True)
    return subprocess.run(cmd).returncode


def _remote_has_stage_summary(args: argparse.Namespace, ip: str) -> bool:
    cmd = (
        "test -s /data/cppmega_h200_results/summary.log "
        "&& grep -q 'CPPMEGA_CURRICULUM_STAGE_START' "
        "/data/cppmega_h200_results/summary.log"
    )
    return _ssh_run_no_check(args, ip, cmd, timeout=30) == 0


def _ssh_run_no_check(args: argparse.Namespace, ip: str, command: str, *, timeout: int | None = None) -> int:
    cmd = sweep.ssh_base(args, ip) + [command]
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[nebius-curriculum] $ {printable}", flush=True)
    return subprocess.run(cmd, timeout=timeout).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-id", default=sweep.DEFAULT_PARENT_ID)
    parser.add_argument("--subnet-id", default=sweep.DEFAULT_SUBNET_ID)
    parser.add_argument("--security-group-id", default=sweep.DEFAULT_SECURITY_GROUP_ID)
    parser.add_argument("--image-id", default=sweep.DEFAULT_IMAGE_ID)
    parser.add_argument("--platform", default="gpu-h200-sxm")
    parser.add_argument("--preset", default="1gpu-16vcpu-200gb")
    parser.add_argument("--disk-type", default="network_ssd")
    parser.add_argument("--disk-size-gib", type=int, default=512)
    parser.add_argument("--instance-name", default=f"cppmega-h200-curriculum-{int(time.time())}")
    parser.add_argument("--ssh-user", default="dave")
    parser.add_argument("--ssh-key", type=Path, default=sweep.default_ssh_key())
    parser.add_argument("--ssh-pubkey", type=Path, default=None)
    parser.add_argument("--docker-image", default=sweep.DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--ghcr-user", default=None)
    parser.add_argument("--ghcr-token-file", type=Path, default=None)
    parser.add_argument("--no-ghcr-auth", action="store_true")
    parser.add_argument("--tokenizer-dir", type=Path, default=sweep.DEFAULT_TOKENIZER_DIR)
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help=(
            "Stage entry seq=batch=iters=prefix or seq=global_batch=micro_batch=iters=prefix. May be repeated. "
            "Defaults to the current reindexed 1024/2048/4096/8192/16384 prefixes."
        ),
    )
    parser.add_argument("--fp8-recipe", choices=["tensorwise"], default="tensorwise")
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        help="First stage index to run. Use 2+ with --initial-checkpoint-root to resume a failed curriculum.",
    )
    parser.add_argument(
        "--initial-checkpoint-root",
        type=Path,
        default=None,
        help="Local checkpoint root to upload and use as PREV_CHECKPOINT_ROOT before --start-stage.",
    )
    parser.add_argument("--remote-timeout-s", type=int, default=86400)
    parser.add_argument("--keep-instance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_stages = (
        [_parse_stage(raw, i) for i, raw in enumerate(args.stage, 1)]
        if args.stage
        else _default_stages()
    )
    if args.start_stage < 1:
        raise ValueError("--start-stage must be >= 1")
    stages = [stage for stage in all_stages if stage.index >= args.start_stage]
    if not stages:
        raise ValueError(f"--start-stage={args.start_stage} selected no stages")
    previous_cum_iters = sum(stage.iters for stage in all_stages if stage.index < args.start_stage)
    if args.start_stage > 1 and args.initial_checkpoint_root is None:
        raise ValueError("--initial-checkpoint-root is required when --start-stage > 1")
    _assert_prefix_contract(stages)

    args.ssh_key = args.ssh_key.expanduser().resolve()
    ssh_pubkey_path = args.ssh_pubkey or Path(str(args.ssh_key) + ".pub")
    ssh_pubkey = ssh_pubkey_path.read_text().strip()

    print("[nebius-curriculum] stages:", flush=True)
    cumulative = previous_cum_iters
    if args.initial_checkpoint_root is not None:
        print(
            f"  initial_checkpoint={args.initial_checkpoint_root.expanduser().resolve()} "
            f"initial_cumulative={previous_cum_iters}",
            flush=True,
        )
    for stage in stages:
        cumulative += stage.iters
        print(
            f"  stage={stage.index} seq={stage.seq} bs={stage.batch} "
            f"stage_iters={stage.iters} target_iters={stage.iters} "
            f"cumulative_after={cumulative} prefix={stage.prefix}",
            flush=True,
        )

    if args.dry_run:
        return 0

    instance_id: str | None = None
    ip: str | None = None
    remote_status = 1
    out_results = ROOT / "outputs" / "nebius" / args.instance_name
    out_checkpoints = ROOT / "outputs" / "checkpoints" / args.instance_name

    with tempfile.TemporaryDirectory(prefix="cppmega-h200-curriculum-") as raw_tmp:
        tmp = Path(raw_tmp)
        overlay_tar = tmp / "cppmega_overlay.tgz"
        sidecar_tar = tmp / "cppmega_sidecar.tgz"
        auth_tar = tmp / "cppmega_ghcr_auth.tgz"
        curriculum_tar = tmp / "cppmega_curriculum.tgz"
        checkpoint_tar = tmp / "cppmega_initial_checkpoint.tgz"

        sweep.make_overlay_tar(overlay_tar)
        sweep.make_multi_sidecar_tar([stage.prefix for stage in stages], args.tokenizer_dir, sidecar_tar)
        _make_curriculum_tar(stages, curriculum_tar)
        initial_remote_checkpoint_root = ""
        if args.initial_checkpoint_root is not None:
            initial_checkpoint_root = args.initial_checkpoint_root.expanduser().resolve()
            _make_checkpoint_tar(initial_checkpoint_root, checkpoint_tar)
            initial_remote_checkpoint_root = (
                f"/data/cppmega_curriculum_checkpoints/{initial_checkpoint_root.name}"
            )
        has_auth = sweep.make_ghcr_auth_tar(args, auth_tar)

        try:
            instance_id = sweep.create_instance(args, ssh_pubkey)
            ip = sweep.wait_for_ip(instance_id)
            sweep.wait_for_ssh(args, ip)
            sweep.stream_tar_to_remote(args, ip, overlay_tar, "/data/cppmega_overlay")
            sweep.stream_tar_to_remote(args, ip, sidecar_tar, "/data")
            sweep.stream_tar_to_remote(args, ip, curriculum_tar, "/data")
            if args.initial_checkpoint_root is not None:
                sweep.stream_tar_to_remote(args, ip, checkpoint_tar, "/data")
            if has_auth:
                sweep.stream_tar_to_remote(args, ip, auth_tar, "/data")

            script = _remote_script(
                stages,
                docker_image=args.docker_image,
                fp8_recipe=args.fp8_recipe,
                initial_checkpoint_root=initial_remote_checkpoint_root,
                initial_cum_iters=previous_cum_iters,
            )
            sweep.ssh(
                args,
                ip,
                "cat > /data/run_cppmega_h200_curriculum.sh <<'EOF'\n"
                + script
                + "\nEOF\nchmod +x /data/run_cppmega_h200_curriculum.sh",
            )
            remote_status = _ssh_run_no_check(
                args,
                ip,
                "bash /data/run_cppmega_h200_curriculum.sh",
                timeout=args.remote_timeout_s,
            )
            if remote_status == 0 and not _remote_has_stage_summary(args, ip):
                print(
                    "[nebius-curriculum] remote script exited 0 but no stage "
                    "summary was written",
                    file=sys.stderr,
                    flush=True,
                )
                remote_status = 70
        finally:
            if ip is not None:
                _scp_from_remote(args, ip, "/data/cppmega_h200_results", out_results)
                _scp_from_remote(args, ip, "/data/cppmega_curriculum_checkpoints", out_checkpoints)
            if instance_id is not None and not args.keep_instance:
                sweep.run(
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

    return remote_status


if __name__ == "__main__":
    raise SystemExit(main())
