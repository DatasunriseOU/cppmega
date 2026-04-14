"""DualPipeV schedule integration for NAM56R Megatron training.

Architecture (fixed 2026-04-14):

- Megatron runs with ``--pipeline-model-parallel-size 1`` — each Megatron
  rank holds the **full** 52-layer model (no Megatron pipeline split).
- We carve a dedicated 2-rank DualPipe process group out of the world.
  With 8 GPUs: pipe groups ``(0,1), (2,3), (4,5), (6,7)`` → DP=4 replicas.
  Each rank's local ``pipe_rank`` is 0 or 1 inside its pipe group.
- DualPipeV splits the 52 layers into 4 virtual stages of 13 layers each,
  assigning (stage0, stage3) to pipe_rank 0 and (stage1, stage2) to
  pipe_rank 1 — the canonical DualPipeV V-shape for near-zero bubble.
- Grads are synchronized across DP replicas via a standard
  ``torch.distributed.new_group`` per pipe_rank (ranks 0/2/4/6 for
  pipe_rank=0, ranks 1/3/5/7 for pipe_rank=1).

The ``assert len(dec.layers) == 52`` is kept — PP=1 guarantees the full
model on each rank. If the model shape changes, update NUM_LAYERS /
LAYERS_PER_STAGE.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

NUM_LAYERS = 52
NUM_STAGES = 4
LAYERS_PER_STAGE = NUM_LAYERS // NUM_STAGES  # 13
DUALPIPE_SIZE = 2  # DualPipeV always pairs 2 ranks


# ---------------------------------------------------------------------------
# Process groups
# ---------------------------------------------------------------------------


@dataclass
class DualPipeGroups:
    """Custom process groups for DualPipeV, carved out of the world.

    ``pipe_group`` is the 2-rank group that DualPipeV operates on.
    ``dp_group`` is the peer group across replicas at the same pipe_rank
    (used by Megatron's data-parallel allreduce for gradient sync).
    """

    pipe_group: "dist.ProcessGroup"
    pipe_rank: int  # 0 or 1 inside pipe_group
    dp_group: Optional["dist.ProcessGroup"]
    dp_size: int


def build_dualpipe_groups() -> DualPipeGroups:
    """Create the DualPipe pipe_group + peer DP group from the world.

    Assumes ``torch.distributed`` is already initialised (Megatron does
    this before calling us). World size must be a multiple of 2.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "DualPipeV: torch.distributed is not initialised. "
            "Call after Megatron world setup."
        )
    world_size = dist.get_world_size()
    if world_size % DUALPIPE_SIZE != 0:
        raise RuntimeError(
            f"DualPipeV: world_size={world_size} is not a multiple of "
            f"DUALPIPE_SIZE={DUALPIPE_SIZE}."
        )
    my_rank = dist.get_rank()
    num_dp = world_size // DUALPIPE_SIZE

    # Carve pipe_groups: [(0,1), (2,3), ...] — contiguous pairs.  Every
    # rank must call new_group with the same rank lists; we enumerate.
    my_pipe_group = None
    my_pipe_rank = -1
    for dp_idx in range(num_dp):
        ranks = [dp_idx * DUALPIPE_SIZE + i for i in range(DUALPIPE_SIZE)]
        grp = dist.new_group(ranks=ranks)
        if my_rank in ranks:
            my_pipe_group = grp
            my_pipe_rank = ranks.index(my_rank)
    assert my_pipe_group is not None and my_pipe_rank >= 0, (
        f"DualPipeV: could not find my pipe group (rank={my_rank})"
    )

    # Carve dp_groups: peers across replicas at the same pipe_rank.
    # pipe_rank 0 -> ranks [0, 2, 4, ...];  pipe_rank 1 -> ranks [1, 3, 5, ...].
    my_dp_group = None
    for pr in range(DUALPIPE_SIZE):
        peers = [dp_idx * DUALPIPE_SIZE + pr for dp_idx in range(num_dp)]
        grp = dist.new_group(ranks=peers) if len(peers) > 1 else None
        if my_rank in peers:
            my_dp_group = grp

    return DualPipeGroups(
        pipe_group=my_pipe_group,
        pipe_rank=my_pipe_rank,
        dp_group=my_dp_group,
        dp_size=num_dp,
    )


# ---------------------------------------------------------------------------
# Stage splitting
# ---------------------------------------------------------------------------


class StageModule(nn.Module):
    """A contiguous slice of decoder layers, optionally owning I/O heads.

    Stage 0 owns embedding; stage (NUM_STAGES-1) owns final_norm + output.
    Non-terminal stages just run transformer layers and hand off to the
    next stage via DualPipeV P2P.
    """

    def __init__(
        self,
        stage_id: int,
        layers: nn.ModuleList,
        *,
        embedding=None,
        final_norm=None,
        output_layer=None,
        rotary_pos_emb=None,
    ):
        super().__init__()
        self.stage_id = stage_id
        self.layers = layers
        self.embedding = embedding
        self.final_norm = final_norm
        self.output_layer = output_layer
        self.rotary_pos_emb = rotary_pos_emb
        self._is_first = stage_id == 0
        self._is_last = stage_id == NUM_STAGES - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from megatron.core.transformer.transformer_layer import TransformerLayer

        # Stage 0: x is [B,T] token ids -> embedding returns [T,B,D].
        # Other stages: x is [B,T,D] from P2P; transpose to [T,B,D] for layers.
        if self._is_first:
            x = self.embedding(input_ids=x.long(), position_ids=None)
        else:
            x = x.transpose(0, 1).contiguous()
        rope = (
            self.rotary_pos_emb(x.size(0))
            if self.rotary_pos_emb is not None
            else None
        )
        for layer in self.layers:
            kw = dict(hidden_states=x, attention_mask=None)
            if isinstance(layer, TransformerLayer):
                kw["rotary_pos_emb"] = rope
            x = layer(**kw)
            if isinstance(x, tuple):
                x = x[0]
        if self._is_last:
            if self.final_norm is not None:
                x = self.final_norm(x)
            if self.output_layer is not None:
                x, _ = self.output_layer(x)
        x = x.transpose(0, 1).contiguous()  # [T,B,D] -> [B,T,D] for P2P
        if not x.requires_grad and torch.is_grad_enabled():
            x.requires_grad_()
        return x


def _unwrap(model):
    """Strip DDP / Float16Module / FullyShardedDP wrappers to find the core
    Megatron model with ``.decoder`` attribute.
    """
    m = model
    for _ in range(6):
        if hasattr(m, "decoder"):
            return m
        if hasattr(m, "module"):
            m = m.module
            continue
        break
    raise RuntimeError(
        f"DualPipeV: cannot find '.decoder' on {type(model).__name__}"
    )


def build_stages(model: nn.Module, pipe_rank: int) -> Tuple[StageModule, StageModule]:
    """Split the full 52-layer model into (fwd_stage, rev_stage) for a
    given pipe_rank (0 or 1).

    DualPipeV V-shape layout:
        pipe_rank 0 owns stages (0, 3)  -> layers [0:13] and [39:52]
        pipe_rank 1 owns stages (1, 2)  -> layers [13:26] and [26:39]
    """
    u = _unwrap(model)
    dec = u.decoder
    if len(dec.layers) != NUM_LAYERS:
        raise RuntimeError(
            f"DualPipeV: expected {NUM_LAYERS} decoder layers (Megatron "
            f"PP must be 1 so the full model lives on each rank), "
            f"got {len(dec.layers)}.  Did you forget "
            f"--pipeline-model-parallel-size 1 ?"
        )
    emb = getattr(u, "embedding", None)
    fnorm = getattr(dec, "final_norm", None)
    olayer = getattr(u, "output_layer", None)
    rope = getattr(u, "rotary_pos_emb", None)

    def _mk(sid: int) -> StageModule:
        lo = sid * LAYERS_PER_STAGE
        hi = lo + LAYERS_PER_STAGE
        return StageModule(
            sid,
            nn.ModuleList(list(dec.layers[lo:hi])),
            embedding=emb if sid == 0 else None,
            final_norm=fnorm if sid == NUM_STAGES - 1 else None,
            output_layer=olayer if sid == NUM_STAGES - 1 else None,
            rotary_pos_emb=rope,
        )

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    fwd_stage = _mk(pipe_rank)  # stage 0 or 1
    rev_stage = _mk(NUM_STAGES - 1 - pipe_rank)  # stage 3 or 2
    return fwd_stage.to(dev), rev_stage.to(dev)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


class DualPipeVLossFn:
    """Cross-entropy loss normalised by number of chunks.

    DualPipeV calls this on the output of stage 3 with labels.  We divide
    by ``num_chunks`` so the accumulated loss matches the non-chunked
    baseline.
    """

    def __init__(self, num_chunks: int):
        self.num_chunks = num_chunks

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            labels.reshape(-1),
            ignore_index=-1,
        )
        return loss / self.num_chunks if self.num_chunks > 1 else loss


# ---------------------------------------------------------------------------
# Driver state + forward_backward replacement
# ---------------------------------------------------------------------------


@dataclass
class DualPipeVState:
    dpv: Any  # dualpipe.DualPipeV
    loss_fn: DualPipeVLossFn
    groups: DualPipeGroups
    num_chunks: int
    mbs: int
    seq_len: int


def _no_sync_ctx(cfg):
    nsf = cfg.no_sync_func
    if isinstance(nsf, list):

        @contextlib.contextmanager
        def _c():
            with contextlib.ExitStack() as s:
                for fn in nsf:
                    s.enter_context(fn())
                yield

        return _c
    return nsf or contextlib.nullcontext


def dualpipev_forward_backward(state: DualPipeVState, **kw) -> List[dict]:
    """Drop-in replacement for Megatron's ``forward_backward_no_pipelining``.

    Consumes num_microbatches batches of (tokens, labels), folds them into
    num_chunks-sized groups, and runs DualPipeV on each group.
    """
    from megatron.core.utils import get_model_config

    model_ref = kw["model"]
    model_ref = model_ref[0] if isinstance(model_ref, list) else model_ref
    cfg = get_model_config(model_ref)
    num_microbatches = kw["num_microbatches"]
    num_chunks = state.num_chunks
    mbs = kw["micro_batch_size"]
    seq_len = kw["seq_length"]
    if num_microbatches < num_chunks or num_microbatches % num_chunks != 0:
        raise RuntimeError(
            f"DualPipeV: num_microbatches={num_microbatches} must be a "
            f"positive multiple of num_chunks={num_chunks}."
        )
    no_sync = _no_sync_ctx(cfg)
    if cfg.timers is not None:
        cfg.timers("forward-backward", log_level=1).start(
            barrier=cfg.barrier_with_L1_time
        )
    results: List[dict] = []

    def _to_cuda(t):
        return (
            t.cuda(non_blocking=True) if t is not None and not t.is_cuda else t
        )

    for _ in range(num_microbatches // num_chunks):
        inp = lab = None
        if state.groups.pipe_rank == 0:
            it = kw["data_iterator"]
            it = it[0] if isinstance(it, list) else it
            toks, labs = [], []
            for _ in range(num_chunks):
                b = next(it)
                if isinstance(b, (tuple, list)):
                    b = {"tokens": b[0], "labels": b[1]}
                toks.append(_to_cuda(b["tokens"]))
                labs.append(_to_cuda(b["labels"]))
            inp = torch.cat(toks, 0)
            lab = torch.cat(labs, 0)
        with no_sync():
            loss_t, _ = state.dpv.step(
                inp,
                num_chunks=num_chunks,
                criterion=state.loss_fn if state.groups.pipe_rank == 0 else None,
                labels=[lab] if state.groups.pipe_rank == 0 else [],
                return_outputs=False,
            )
        if state.groups.pipe_rank == 0:
            tot = loss_t.sum()
            num_tokens = torch.tensor(
                [mbs * seq_len * num_chunks],
                dtype=torch.int,
                device=tot.device,
            )
            results.append(
                {
                    "lm loss": torch.cat(
                        [tot.detach().view(1), num_tokens.float().view(1)]
                    )
                }
            )
        else:
            results.append({"lm loss": torch.zeros(2, device="cuda")})

    if (
        cfg.finalize_model_grads_func is not None
        and not kw.get("forward_only", False)
    ):
        num_tokens = torch.tensor(
            [mbs * seq_len * num_microbatches], dtype=torch.int, device="cuda"
        )
        cfg.finalize_model_grads_func(
            [model_ref],
            num_tokens if cfg.calculate_per_token_loss else None,
            pg_collection=kw.get("pg_collection"),
            force_all_reduce=kw.get("force_all_reduce", False),
        )
    if cfg.timers is not None:
        cfg.timers("forward-backward").stop()
    return results


def patch_forward_backward_func(state: DualPipeVState) -> None:
    """Monkey-patch Megatron's ``get_forward_backward_func`` to return our
    DualPipeV schedule closure.  This overrides the default
    ``forward_backward_no_pipelining`` (used when PP=1).
    """
    import megatron.core.pipeline_parallel as pp
    import megatron.core.pipeline_parallel.schedules as sched

    def _fb(**kw):
        return dualpipev_forward_backward(state, **kw)

    def _get(*a, **k):
        return _fb

    sched.get_forward_backward_func = _get
    if hasattr(pp, "get_forward_backward_func"):
        pp.get_forward_backward_func = _get
    try:
        import megatron.training.training as _t

        _t.get_forward_backward_func = _get
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# One-call setup
# ---------------------------------------------------------------------------


def setup_dualpipev_from_args(
    model: nn.Module, args: Any
) -> DualPipeVState:
    """Top-level setup: verify topology, carve groups, split stages, wrap
    in DualPipeV, patch Megatron's forward_backward dispatch.

    Call this once after Megatron builds ``model`` but before the first
    training step.  Requires Megatron started with
    ``--pipeline-model-parallel-size 1``.
    """
    from dualpipe import DualPipeV, set_p2p_tensor_dtype, set_p2p_tensor_shapes
    from megatron.core import parallel_state as ps

    if ps.get_pipeline_model_parallel_world_size() != 1:
        raise RuntimeError(
            "DualPipeV requires Megatron PP=1 (full model per rank).  "
            "Got --pipeline-model-parallel-size "
            f"{ps.get_pipeline_model_parallel_world_size()}."
        )

    groups = build_dualpipe_groups()
    mbs = args.micro_batch_size
    seq_len = args.seq_length
    hidden = args.hidden_size

    # num_microbatches per step must be a multiple of num_chunks (we pick
    # 4 for the V-shape — 2 fwd + 2 bwd "waves" in the pipeline).
    num_chunks = int(os.environ.get("CPPMEGA_DUALPIPEV_CHUNKS", "4"))
    dp_size = groups.dp_size
    gbs = args.global_batch_size
    if gbs % (mbs * dp_size) != 0:
        raise RuntimeError(
            f"DualPipeV: GBS={gbs} not divisible by mbs*dp={mbs*dp_size}"
        )
    num_microbatches = gbs // (mbs * dp_size)
    if num_microbatches < num_chunks or num_microbatches % num_chunks != 0:
        raise RuntimeError(
            f"DualPipeV: num_microbatches={num_microbatches} must be a "
            f"positive multiple of num_chunks={num_chunks}. "
            f"Adjust GBS or MBS."
        )

    fwd_stage, rev_stage = build_stages(model, groups.pipe_rank)

    set_p2p_tensor_shapes([(mbs, seq_len, hidden)])
    set_p2p_tensor_dtype(torch.bfloat16)

    dpv = DualPipeV(
        modules=(fwd_stage, rev_stage),
        batch_dim=0,
        process_group=groups.pipe_group,
    )

    state = DualPipeVState(
        dpv=dpv,
        loss_fn=DualPipeVLossFn(num_chunks=num_chunks),
        groups=groups,
        num_chunks=num_chunks,
        mbs=mbs,
        seq_len=seq_len,
    )
    patch_forward_backward_func(state)
    return state
