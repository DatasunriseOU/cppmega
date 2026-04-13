"""DualPipeV schedule for NAM56R Megatron training (PP=2, 4 stages of 13 layers)."""
from __future__ import annotations
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import torch, torch.nn as nn
from dualpipe import DualPipeV, set_p2p_tensor_shapes, set_p2p_tensor_dtype

NUM_LAYERS, NUM_STAGES, LAYERS_PER_STAGE = 52, 4, 13

class StageModule(nn.Module):
    def __init__(self, stage_id, layers, *, embedding=None, final_norm=None,
                 output_layer=None, rotary_pos_emb=None):
        super().__init__()
        self.stage_id, self.layers = stage_id, layers
        self.embedding, self.final_norm = embedding, final_norm
        self.output_layer, self.rotary_pos_emb = output_layer, rotary_pos_emb
        self._is_first, self._is_last = stage_id == 0, stage_id == NUM_STAGES - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from megatron.core.transformer.transformer_layer import TransformerLayer
        # Stage 0: x is [B,T] token ids; embedding outputs [T,B,D].
        # Other stages: x is [B,T,D] from DualPipeV P2P; transpose to [T,B,D].
        if self._is_first:
            x = self.embedding(input_ids=x.long(), position_ids=None)
        else:
            x = x.transpose(0, 1).contiguous()
        rope = self.rotary_pos_emb(x.size(0)) if self.rotary_pos_emb is not None else None
        for layer in self.layers:
            kw = dict(hidden_states=x, attention_mask=None)
            if isinstance(layer, TransformerLayer):
                kw['rotary_pos_emb'] = rope
            x = layer(**kw)
            if isinstance(x, tuple):
                x = x[0]
        if self._is_last:
            if self.final_norm is not None:
                x = self.final_norm(x)
            if self.output_layer is not None:
                x, _ = self.output_layer(x)
        x = x.transpose(0, 1).contiguous()  # back to [B,T,D]
        if not x.requires_grad and torch.is_grad_enabled():
            x.requires_grad_()
        return x

class DualPipeVLossFn:
    def __init__(self, n: int): self.n = n
    def __call__(self, logits, labels):
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), labels.view(-1), ignore_index=-1)
        return loss / self.n if self.n > 1 else loss

def _unwrap(model):
    m = model
    for _ in range(5):
        if hasattr(m, 'decoder'): return m
        if hasattr(m, 'module'): m = m.module
    raise RuntimeError(f"Cannot find 'decoder' on {type(m).__name__}")

def build_stages(model, pp_rank):
    """Rank 0 -> stages (0,3), Rank 1 -> stages (1,2). 13 layers each."""
    u = _unwrap(model)
    dec = u.decoder
    assert len(dec.layers) == NUM_LAYERS
    emb, fnorm = getattr(u, 'embedding', None), getattr(dec, 'final_norm', None)
    olayer, rope = getattr(u, 'output_layer', None), getattr(u, 'rotary_pos_emb', None)
    def _mk(sid):
        s = sid * LAYERS_PER_STAGE
        return StageModule(sid, nn.ModuleList(list(dec.layers[s:s+LAYERS_PER_STAGE])),
            embedding=emb if sid == 0 else None,
            final_norm=fnorm if sid == NUM_STAGES-1 else None,
            output_layer=olayer if sid == NUM_STAGES-1 else None, rotary_pos_emb=rope)
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    return _mk(pp_rank).to(dev), _mk(NUM_STAGES-1-pp_rank).to(dev)

@dataclass
class DualPipeVState:
    dpv: DualPipeV; loss_fn: DualPipeVLossFn; pp_rank: int
    num_chunks: int; mbs: int; seq_len: int

def _no_sync_ctx(cfg):
    nsf = cfg.no_sync_func
    if isinstance(nsf, list):
        @contextlib.contextmanager
        def _c():
            with contextlib.ExitStack() as s:
                for fn in nsf: s.enter_context(fn())
                yield
        return _c
    return nsf or contextlib.nullcontext

def dualpipev_forward_backward(state, **kw):
    """Drop-in replacement for Megatron's forward_backward_no_pipelining."""
    from megatron.core.utils import get_model_config
    model_ref = kw['model']
    model_ref = model_ref[0] if isinstance(model_ref, list) else model_ref
    cfg = get_model_config(model_ref)
    nm, nc, mbs, sl = kw['num_microbatches'], state.num_chunks, kw['micro_batch_size'], kw['seq_length']
    assert nm >= nc and nm % nc == 0
    nsf = _no_sync_ctx(cfg)
    if cfg.timers is not None:
        cfg.timers('forward-backward', log_level=1).start(barrier=cfg.barrier_with_L1_time)
    results = []
    _cuda = lambda t: t.cuda(non_blocking=True) if t is not None and not t.is_cuda else t
    for _ in range(nm // nc):
        inp = lab = None
        if state.pp_rank == 0:
            it = kw['data_iterator']
            it = it[0] if isinstance(it, list) else it
            toks, labs = [], []
            for _ in range(nc):
                b = next(it)
                if isinstance(b, (tuple, list)): b = {'tokens': b[0], 'labels': b[1]}
                toks.append(_cuda(b['tokens'])); labs.append(_cuda(b['labels']))
            inp, lab = torch.cat(toks, 0), torch.cat(labs, 0)
        with nsf():
            loss_t, _ = state.dpv.step(inp, num_chunks=nc,
                criterion=state.loss_fn if state.pp_rank == 0 else None,
                labels=[lab] if state.pp_rank == 0 else [], return_outputs=False)
        if state.pp_rank == 0:
            tot = loss_t.sum()
            nt = torch.tensor([mbs*sl*nc], dtype=torch.int, device=tot.device)
            results.append({'lm loss': torch.cat([tot.detach().view(1), nt.float().view(1)])})
        else:
            results.append({'lm loss': torch.zeros(2, device='cuda')})
    if cfg.finalize_model_grads_func is not None and not kw.get('forward_only', False):
        nt = torch.tensor([mbs*sl*nm], dtype=torch.int, device='cuda')
        cfg.finalize_model_grads_func([model_ref],
            nt if cfg.calculate_per_token_loss else None,
            pg_collection=kw.get('pg_collection'), force_all_reduce=kw.get('force_all_reduce', False))
    if cfg.timers is not None:
        cfg.timers('forward-backward').stop()
    return results

def patch_forward_backward_func(state: DualPipeVState) -> None:
    """Monkey-patch Megatron's get_forward_backward_func to return DualPipeV schedule."""
    import megatron.core.pipeline_parallel.schedules as sched
    import megatron.core.pipeline_parallel as pp
    _fb = lambda **kw: dualpipev_forward_backward(state, **kw)
    _get = lambda *a, **k: _fb
    sched.get_forward_backward_func = _get
    if hasattr(pp, 'get_forward_backward_func'):
        pp.get_forward_backward_func = _get
    try:
        import megatron.training.training as _t
        _t.get_forward_backward_func = _get
    except (ImportError, AttributeError):
        pass

def setup_dualpipev_from_args(model: nn.Module, args: Any) -> DualPipeVState:
    """One-call setup: split model, create DualPipeV, patch Megatron."""
    from megatron.core import parallel_state as ps
    pp_rank = ps.get_pipeline_model_parallel_rank()
    assert ps.get_pipeline_model_parallel_world_size() == 2
    mbs, sl, hs = args.micro_batch_size, args.seq_length, args.hidden_size
    nm = args.global_batch_size // (mbs * ps.get_data_parallel_world_size())
    nc = 4
    assert nm >= nc and nm % nc == 0
    fwd, rev = build_stages(model, pp_rank)
    set_p2p_tensor_shapes([(mbs, sl, hs)])
    set_p2p_tensor_dtype(torch.bfloat16)
    dpv = DualPipeV(modules=(fwd, rev), batch_dim=0,
                    process_group=ps.get_pipeline_model_parallel_group())
    st = DualPipeVState(dpv=dpv, loss_fn=DualPipeVLossFn(nm),
                        pp_rank=pp_rank, num_chunks=nc, mbs=mbs, seq_len=sl)
    patch_forward_backward_func(st)
    return st
