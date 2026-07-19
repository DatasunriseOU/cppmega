"""Integration tests for the DSA indexer fused graph-gradient patch.

Verifies :func:`compute_index_scores_fused_bf16` matches the upstream
Megatron BF16 ``_compute_index_scores`` einsum implementation to within
FP32 associative-reorder tolerance.

The gradient integration cases import the real sibling Megatron DSA contract;
the larger BF16 parity cases remain runnable on CPU and CUDA.
"""

from __future__ import annotations

import subprocess
import sys
import json
import os
import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from cppmega.megatron.dsa_indexer_fused_patch import (
    build_graph_route_bias_from_structure_batch,
    compute_index_scores_fused_bf16,
)


def _real_megatron_subprocess_environment(
    **overrides: str,
) -> dict[str, str]:
    from megatron.core.transformer.experimental_attention_variant import dsa

    source_file = Path(inspect.getsourcefile(dsa) or "").resolve()
    source_root = source_file.parents[4]
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    pythonpath = [str(repo_root), str(source_root)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    # core_v0.18.0 package_info otherwise shells out to git during every clean
    # subprocess import; the source checkout is pinned by the pytest receipt.
    environment["NO_VCS_VERSION"] = "1"
    environment.update(overrides)
    return environment


def _upstream_reference(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    use_relu: bool = True,
) -> torch.Tensor:
    """Byte-identical clone of upstream Megatron ``_compute_index_scores``."""
    index_scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    if use_relu:
        index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    return index_scores.transpose(0, 1)  # [b, sq, sk]


def test_fused_matches_reference_bf16_relu():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sq, sk, b, h, d = 128, 128, 2, 8, 64

    q = torch.randn(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(sq, b, h, dtype=torch.bfloat16, device=device)

    ref = _upstream_reference(q, w, k, use_relu=True)
    fused = compute_index_scores_fused_bf16(q, w, k, use_relu=True)

    assert ref.shape == fused.shape == (b, sq, sk)
    assert ref.dtype == fused.dtype == torch.float32

    # FP32 associative reorder tolerance: per-head accum vs fused-sum over h.
    abs_err = (ref - fused).abs().max().item()
    ref_abs = ref.abs().max().item()
    rel_err = abs_err / max(ref_abs, 1e-6)
    print(f"relu=True abs_err={abs_err:.3e} rel_err={rel_err:.3e}")
    assert rel_err < 1e-3, f"rel_err {rel_err} too high"


def test_fused_matches_reference_bf16_no_relu():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sq, sk, b, h, d = 256, 256, 1, 16, 64

    q = torch.randn(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(sq, b, h, dtype=torch.bfloat16, device=device)

    ref = _upstream_reference(q, w, k, use_relu=False)
    fused = compute_index_scores_fused_bf16(q, w, k, use_relu=False)

    abs_err = (ref - fused).abs().max().item()
    rel_err = abs_err / max(ref.abs().max().item(), 1e-6)
    print(f"relu=False abs_err={abs_err:.3e} rel_err={rel_err:.3e}")
    assert rel_err < 1e-3, f"rel_err {rel_err} too high"


def test_graph_route_bias_from_structure_batch_scatter_edges():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structure_batch = {
        "graph_call_edges": torch.tensor(
            [[[0, 2], [1, 3], [-1, -1]]], dtype=torch.long
        ),
        "graph_call_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_type_edges": torch.tensor([[[2, 1], [-1, -1]]], dtype=torch.long),
        "graph_type_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([4], dtype=torch.long),
    }

    bias = build_graph_route_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device=device,
        call_weight=2.0,
        type_weight=3.0,
    )

    assert tuple(bias.shape) == (1, 4, 4)
    assert bias[0, 0, 2].item() == 2.0
    assert bias[0, 1, 3].item() == 2.0
    assert bias[0, 2, 1].item() == 3.0
    assert bias.sum().item() == 7.0


def test_graph_route_chunk_edge_expands_to_token_span_block():
    structure_batch = {
        "graph_call_edges": torch.tensor([[[0, 1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 2]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[2, 4]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([2], dtype=torch.long),
    }

    bias = build_graph_route_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device=torch.device("cpu"),
        call_weight=2.0,
    )

    expected = torch.zeros((1, 4, 4))
    expected[0, 0:2, 2:4] = 2.0
    assert torch.equal(bias, expected)


def test_graph_objective_golden_span_expansion_uses_doc_and_upstream_masks():
    from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

    structure_batch = {
        "graph_call_edges": torch.tensor(
            [[[2, 1], [0, 2]]], dtype=torch.long
        ),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 2, 4]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[2, 4, 6]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([3], dtype=torch.long),
        "graph_document_ids": torch.tensor(
            [[101, 101, 202, 202, 202, 202]], dtype=torch.long
        ),
    }
    upstream_mask = torch.zeros((6, 6), dtype=torch.float32)
    upstream_mask[5, 3] = float("-inf")

    targets, pair_mask = fused_patch.build_graph_objective_tensors(
        structure_batch,
        relations=("call",),
        batch_size=1,
        seqlen_q=6,
        seqlen_k=6,
        device=torch.device("cpu"),
        upstream_mask=upstream_mask,
    )

    expected_targets = torch.zeros((1, 6, 6), dtype=torch.bool)
    expected_targets[0, 4, 2] = True
    expected_targets[0, 4, 3] = True
    expected_targets[0, 5, 2] = True
    assert torch.equal(targets, expected_targets)
    assert pair_mask[0, 1, 0]
    assert not pair_mask[0, 2, 1]
    assert not pair_mask[0, 5, 3]
    assert torch.all(~targets | pair_mask)


def test_graph_route_bias_from_structure_batch_scatter_domain_edge_triples():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structure_batch = {
        "graph_domain_edges": torch.tensor(
            [[[0, 2, 20], [-1, -1, -1]]], dtype=torch.long
        ),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_diagnostic_edges": torch.tensor(
            [[[1, 3, 60], [-1, -1, -1]]], dtype=torch.long
        ),
        "graph_diagnostic_edge_counts": torch.tensor([1], dtype=torch.long),
    }

    bias = build_graph_route_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device=device,
        domain_weight=2.0,
        diagnostic_weight=5.0,
    )

    assert bias[0, 0, 2].item() == 2.0
    assert bias[0, 1, 3].item() == 5.0
    assert bias.sum().item() == 7.0


def test_graph_route_bias_rejects_fractional_production_sidecars():
    structure_batch = {
        "graph_domain_edges": torch.tensor(
            [[[1.5, 0.0, 5.0]]], dtype=torch.float32
        ),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
    }

    with pytest.raises(TypeError, match="integer"):
        build_graph_route_bias_from_structure_batch(
            structure_batch,
            batch_size=1,
            seqlen_q=2,
            seqlen_k=2,
            device=torch.device("cpu"),
        )


def test_explicit_empty_graph_batch_has_zero_prior():
    structure_batch = {
        "graph_domain_edges": torch.empty((1, 0, 3), dtype=torch.long),
        "graph_domain_edge_counts": torch.zeros((1,), dtype=torch.long),
    }

    bias = build_graph_route_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=2,
        seqlen_k=2,
        device=torch.device("cpu"),
    )

    assert torch.equal(bias, torch.zeros_like(bias))


def test_fused_scores_add_graph_route_bias_before_topk():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sq = sk = 4
    b, h, d = 1, 2, 8
    q = torch.zeros(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.zeros(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.ones(sq, b, h, dtype=torch.bfloat16, device=device)
    graph_bias = torch.zeros((b, sq, sk), dtype=torch.float32, device=device)
    graph_bias[0, 1, 3] = 1.0
    graph_bias[0, 2, 0] = 0.5

    scores = compute_index_scores_fused_bf16(
        q,
        w,
        k,
        use_relu=True,
        graph_bias=graph_bias,
        graph_beta=10.0,
    )

    assert scores[0, 1, 3].item() == 10.0
    assert scores[0, 2, 0].item() == 5.0
    assert scores[0, 0, 0].item() == 0.0


def test_fused_scores_reject_nonpositive_graph_beta():
    q = torch.zeros((2, 1, 1, 4), dtype=torch.bfloat16)
    k = torch.zeros((2, 1, 4), dtype=torch.bfloat16)
    weights = torch.ones((2, 1, 1), dtype=torch.bfloat16)
    graph_bias = torch.zeros((1, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="graph_beta"):
        compute_index_scores_fused_bf16(
            q,
            weights,
            k,
            graph_bias=graph_bias,
            graph_beta=0.0,
        )


def test_fused_scores_use_canonical_graph_beta_by_default():
    q = torch.zeros((2, 1, 1, 4), dtype=torch.bfloat16)
    k = torch.zeros((2, 1, 4), dtype=torch.bfloat16)
    weights = torch.ones((2, 1, 1), dtype=torch.bfloat16)
    graph_bias = torch.zeros((1, 2, 2), dtype=torch.float32)
    graph_bias[0, 1, 0] = 1.0

    with patch.dict(
        os.environ,
        {"CPPMEGA_GRAPH_BIAS_BETA": "3"},
        clear=True,
    ):
        scores = compute_index_scores_fused_bf16(
            q,
            weights,
            k,
            graph_bias=graph_bias,
        )

    assert scores[0, 1, 0].item() == 3.0


def test_real_pinned_dsa_topk_emits_selector_receipt(tmp_path):
    """Exercise the installed pinned Megatron selector in a clean subprocess."""

    receipt_path = tmp_path / "dsa-selector.json"
    script = r'''
import json
import os
from pathlib import Path

import torch

from megatron.core.transformer.experimental_attention_variant import dsa
from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

structure = {
    "graph_domain_edges": torch.tensor([[[0, 3, 7]]], dtype=torch.long),
    "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
}
token = fused_patch._set_graph_batch_override(structure)
try:
    fused_patch.apply_dsa_indexer_fused_patch(force=True)
    q = torch.zeros((2, 1, 1, 4), dtype=torch.bfloat16)
    k = torch.zeros((4, 1, 4), dtype=torch.bfloat16)
    weights = torch.ones((2, 1, 1), dtype=torch.bfloat16)
    mask = torch.zeros((1, 2, 4), dtype=torch.float32)
    scores, indices = dsa.fused_qk_topk_naive(q, k, weights, 2, mask)
    dsa.fused_qk_topk_naive(q, k, weights, 2, mask)
    print(json.dumps({"indices": indices.tolist(), "scores": scores.tolist()}))
finally:
    fused_patch._reset_graph_batch_override(token)
'''
    environment = os.environ.copy()
    environment.update(
        {
            "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
            "CPPMEGA_GRAPH_BIAS_BETA": "1",
            "CPPMEGA_H200_GRAPH_PRIOR_RECEIPT": str(receipt_path),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output["indices"][0][0][0] == 3
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selector = receipt["selector"]
    assert selector["formula"] == "I_neural + beta*S_graph -> mask -> topk"
    assert selector["observation_count"] == 1
    observation = selector["observations"][0]
    assert observation["indices_match"] is True
    assert observation["equation_max_abs_error"] <= 1e-5
    assert len(observation["topk_indices"]["sha256"]) == 64


def test_real_pinned_dsa_topk_applies_graph_before_mask_and_selection(tmp_path):
    """The actual pinned Megatron selector must consume graph-biased scores."""

    script = r'''
import json
import torch

from megatron.core.transformer.experimental_attention_variant import dsa
from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

structure = {
    "graph_domain_edges": torch.tensor([[[0, 3, 7]]], dtype=torch.long),
    "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
}
token = fused_patch._set_graph_batch_override(structure)
try:
    fused_patch.apply_dsa_indexer_fused_patch(force=True)
    q = torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16)
    k = torch.zeros((4, 1, 2), dtype=torch.bfloat16)
    weights = torch.ones((1, 1, 1), dtype=torch.bfloat16)
    plain_scores, plain_indices = dsa.fused_qk_topk_naive(
        q, k, weights, 1, None
    )
    mask = torch.zeros((1, 1, 4), dtype=torch.float32)
    mask[..., 3] = float("-inf")
    masked_scores, masked_indices = dsa.fused_qk_topk_naive(
        q, k, weights, 1, mask
    )
    print(json.dumps({
        "plain_scores": plain_scores.tolist(),
        "plain_indices": plain_indices.tolist(),
        "masked_scores": masked_scores.tolist(),
        "masked_indices": masked_indices.tolist(),
    }))
finally:
    fused_patch._reset_graph_batch_override(token)
'''
    environment = os.environ.copy()
    environment.update(
        {
            "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
            "CPPMEGA_GRAPH_BIAS_BETA": "4",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output["plain_indices"][0][0][0] == 3
    assert output["plain_scores"][0][0][3] == pytest.approx(4.0)
    assert output["masked_indices"][0][0][0] != 3
    assert output["masked_scores"][0][0][3] == float("-inf")


def test_real_pinned_dsa_token_only_fallback_matches_upstream():
    script = r'''
import inspect
import json
import torch

from megatron.core.transformer.experimental_attention_variant import dsa
from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

assert tuple(inspect.signature(dsa.FusedDSAIndexerLoss.forward).parameters) == (
    "ctx", "q", "weights", "k", "query", "key", "softmax_scale", "topk",
    "loss_coeff", "mask", "sparse_loss", "pg_collection",
)
fused_patch.apply_dsa_indexer_fused_patch(force=True)
q = torch.tensor(
    [[[[1.0, 0.5]]], [[[0.5, 1.0]]]], dtype=torch.float32
)
k = torch.tensor(
    [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32
)
weights = torch.ones((2, 1, 1), dtype=torch.float32)
reference = torch.einsum("sbhd,tbd->sbht", q, k)
reference = torch.relu(reference) * weights.unsqueeze(-1)
reference = reference.sum(dim=2).transpose(0, 1)
scores, indices = dsa.fused_qk_topk_naive(q, k, weights, 1, None)
print(json.dumps({
    "scores": scores.tolist(),
    "reference": reference.tolist(),
    "indices": indices.tolist(),
}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=_real_megatron_subprocess_environment(
            CPPMEGA_GRAPH_ROUTES_ENABLED="0",
            CPPMEGA_DSA_GRAPH_AUX_ENABLED="0",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output["scores"] == output["reference"]
    assert output["indices"] == [[[0], [1]]]


def test_real_pinned_fused_dsa_edge_changes_selection_total_loss_and_gradients():
    script = r'''
import json
from types import SimpleNamespace

import torch

from megatron.core.transformer.experimental_attention_variant import dsa
from cppmega.megatron import dsa_indexer_fused_patch as fused_patch
from cppmega.megatron.structure_dataset_patch import _set_current_structure_batch

class TPGroup:
    @staticmethod
    def size():
        return 1

pg_collection = SimpleNamespace(tp=TPGroup())
fused_patch.apply_dsa_indexer_fused_patch(force=True)

def run(edge_destination):
    structure = {
        "graph_domain_edges": torch.tensor(
            [[[3, edge_destination, 5]]], dtype=torch.long
        ),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_document_ids": torch.full((1, 4), 11, dtype=torch.long),
    }
    _set_current_structure_batch(structure)
    q = torch.ones((4, 1, 1, 2), dtype=torch.float32, requires_grad=True)
    k = torch.tensor(
        [[[2.0, 0.0]], [[0.5, 0.0]], [[0.25, 0.0]], [[0.1, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = torch.ones((4, 1, 1), dtype=torch.float32, requires_grad=True)
    query = torch.zeros((4, 1, 1, 2), dtype=torch.float32)
    key = torch.zeros((4, 1, 1, 2), dtype=torch.float32)
    mask = torch.triu(
        torch.full((4, 4), float("-inf"), dtype=torch.float32), diagonal=1
    )
    indices, total_loss = dsa.FusedDSAIndexerLoss.apply(
        q,
        weights,
        k,
        query,
        key,
        1.0,
        1,
        0.0,
        mask,
        False,
        pg_collection,
    )
    total_loss.backward()
    return {
        "selected": int(indices[0, 3, 0]),
        "loss": float(total_loss.detach()),
        "q_grad": q.grad.tolist(),
        "weights_grad": weights.grad.tolist(),
        "k_grad": k.grad.tolist(),
    }

try:
    first = run(0)
    second = run(1)
finally:
    _set_current_structure_batch(None)
print(json.dumps({"first": first, "second": second}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=_real_megatron_subprocess_environment(
            CPPMEGA_STRUCTURE_ENABLED="1",
            CPPMEGA_GRAPH_ROUTES_ENABLED="1",
            CPPMEGA_DSA_GRAPH_AUX_ENABLED="1",
            CPPMEGA_DSA_GRAPH_AUX_RELATIONS="domain",
            CPPMEGA_DSA_GRAPH_AUX_WEIGHT="1",
            CPPMEGA_DSA_INDEXER_LOSS_COEFF="1",
            CPPMEGA_DSA_GRAPH_LAYER_WEIGHT="1",
            CPPMEGA_DSA_GRAPH_BCE_WEIGHT="1",
            CPPMEGA_DSA_GRAPH_COVERAGE_WEIGHT="1",
            CPPMEGA_DSA_GRAPH_AUX_TOPK="1",
            CPPMEGA_DSA_GRAPH_POS_WEIGHT="1",
            CPPMEGA_DSA_GRAPH_MARGIN="1",
            CPPMEGA_GRAPH_BIAS_BETA="8",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    output = json.loads(result.stdout.strip().splitlines()[-1])
    first = output["first"]
    second = output["second"]
    assert first["selected"] == 0
    assert second["selected"] == 1
    assert first["loss"] != pytest.approx(second["loss"])
    assert first["q_grad"] != second["q_grad"]
    assert first["weights_grad"] != second["weights_grad"]
    assert first["k_grad"] != second["k_grad"]


def test_real_pinned_fused_dsa_explicit_ablation_is_token_only():
    script = r'''
import json
from types import SimpleNamespace

import torch

from megatron.core.transformer.experimental_attention_variant import dsa
from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

class TPGroup:
    @staticmethod
    def size():
        return 1

fused_patch.apply_dsa_indexer_fused_patch(force=True)
q = torch.ones((2, 1, 1, 2), dtype=torch.float32, requires_grad=True)
k = torch.tensor(
    [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32, requires_grad=True
)
weights = torch.ones((2, 1, 1), dtype=torch.float32, requires_grad=True)
query = torch.zeros((2, 1, 1, 2), dtype=torch.float32)
key = torch.zeros((2, 1, 1, 2), dtype=torch.float32)
mask = torch.triu(
    torch.full((2, 2), float("-inf"), dtype=torch.float32), diagonal=1
)
indices, total_loss = dsa.FusedDSAIndexerLoss.apply(
    q,
    weights,
    k,
    query,
    key,
    1.0,
    1,
    0.0,
    mask,
    False,
    SimpleNamespace(tp=TPGroup()),
)
total_loss.backward()
print(json.dumps({
    "indices": indices.tolist(),
    "loss": float(total_loss.detach()),
    "q_grad_nonzero": int(torch.count_nonzero(q.grad)),
    "weights_grad_nonzero": int(torch.count_nonzero(weights.grad)),
    "k_grad_nonzero": int(torch.count_nonzero(k.grad)),
}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=_real_megatron_subprocess_environment(
            CPPMEGA_STRUCTURE_ENABLED="1",
            CPPMEGA_GRAPH_ROUTES_ENABLED="0",
            CPPMEGA_GRAPH_ROUTES_ABLATION="1",
            CPPMEGA_DSA_GRAPH_AUX_ENABLED="1",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output["loss"] == 0.0
    assert output["q_grad_nonzero"] == 0
    assert output["weights_grad_nonzero"] == 0
    assert output["k_grad_nonzero"] == 0


def test_graph_edges_change_production_indexer_loss_and_gradient():
    from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

    base_scores = torch.zeros((1, 4, 4), dtype=torch.float32)
    base_scores[0, 3, 0] = 5.0
    base_scores[0, 3, 1] = -5.0
    environment = {
        "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
        "CPPMEGA_DSA_GRAPH_AUX_ENABLED": "1",
        "CPPMEGA_DSA_GRAPH_AUX_RELATIONS": "domain",
        "CPPMEGA_DSA_GRAPH_AUX_WEIGHT": "1",
        "CPPMEGA_DSA_INDEXER_LOSS_COEFF": "1",
        "CPPMEGA_DSA_GRAPH_LAYER_WEIGHT": "1",
        "CPPMEGA_DSA_GRAPH_BCE_WEIGHT": "1",
        "CPPMEGA_DSA_GRAPH_COVERAGE_WEIGHT": "1",
        "CPPMEGA_DSA_GRAPH_AUX_TOPK": "4",
        "CPPMEGA_GRAPH_BIAS_BETA": "1",
    }

    def run(edge_destination: int) -> tuple[torch.Tensor, torch.Tensor]:
        scores = base_scores.clone().requires_grad_(True)
        structure = {
            "graph_domain_edges": torch.tensor(
                [[[3, edge_destination, 5]]], dtype=torch.long
            ),
            "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
            "graph_document_ids": torch.full((1, 4), 7, dtype=torch.long),
        }
        token = fused_patch._set_graph_batch_override(structure)
        try:
            with patch.dict(os.environ, environment, clear=True):
                loss = fused_patch._graph_objective_from_index_scores(scores)
            loss.backward()
            return loss.detach(), scores.grad.detach().clone()
        finally:
            fused_patch._reset_graph_batch_override(token)

    low_loss, low_grad = run(0)
    high_loss, high_grad = run(1)

    assert torch.isfinite(low_loss)
    assert torch.isfinite(high_loss)
    assert high_loss.item() > low_loss.item()
    assert low_grad[0, 3, 0].abs().item() > 0
    assert high_grad[0, 3, 1].abs().item() > 0
    assert not torch.equal(low_grad, high_grad)


def test_production_tensor_only_dsa_fails_closed_unless_explicit_ablation():
    from cppmega.megatron.dsa_indexer_fused_patch import (
        apply_dsa_indexer_fused_patch,
        require_graph_routes_for_production,
    )

    production_environment = {
        "CPPMEGA_GRAPH_ROUTES_ENABLED": "0",
        "CPPMEGA_H200_GRAPH_PRIOR_RECEIPT": "/tmp/graph-prior.json",
    }
    with patch.dict(os.environ, production_environment, clear=True):
        with pytest.raises(RuntimeError, match="tensor-only"):
            apply_dsa_indexer_fused_patch(force=True)

    with patch.dict(
        os.environ,
        {
            **production_environment,
            "CPPMEGA_GRAPH_ROUTES_ABLATION": "1",
        },
        clear=True,
    ):
        require_graph_routes_for_production()


def test_dsa_graph_prior_receipt_binds_canonical_beta(tmp_path):
    from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

    structure_batch = {
        "graph_domain_edges": torch.tensor([[[1, 3, 5]]], dtype=torch.long),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
    }
    receipt_path = tmp_path / "dsa-prior.json"
    token = fused_patch._set_graph_batch_override(structure_batch)
    try:
        with patch.dict(
            os.environ,
            {
                "CPPMEGA_GRAPH_BIAS_BETA": "2",
                "CPPMEGA_H200_GRAPH_PRIOR_RECEIPT": str(receipt_path),
            },
            clear=True,
        ):
            bias = fused_patch._current_graph_route_bias(
                batch_size=1,
                seqlen_q=4,
                seqlen_k=4,
                device=torch.device("cpu"),
            )
    finally:
        fused_patch._reset_graph_batch_override(token)

    assert bias[0, 1, 3].item() == 1.0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["consumer"] == "dsa_indexer"
    assert receipt["bias_beta"]["value"] == "2"


def test_scatter_edges_vectorized_matches_reference_loop():
    from cppmega.megatron.dsa_indexer_fused_patch import _scatter_edges_

    torch.manual_seed(0)
    B, sq, sk, maxE = 8, 32, 24, 6
    edges = torch.full((B, maxE, 2), -1, dtype=torch.int32)
    counts = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        n = int(torch.randint(0, maxE + 1, (1,)))
        counts[b] = n
        if n:
            flat = torch.randperm(sq * sk)[:n]  # unique (src,dst) per sample
            edges[b, :n, 0] = (flat // sk).to(torch.int32)
            edges[b, :n, 1] = (flat % sk).to(torch.int32)

    bias_vec = torch.zeros(B, sq, sk)
    _scatter_edges_(bias_vec, edges, counts, weight=2.0, sq=sq, sk=sk, require_kind=False)

    bias_ref = torch.zeros(B, sq, sk)  # naive per-sample reference (old semantics)
    for b in range(B):
        for e in range(int(counts[b])):
            bias_ref[b, int(edges[b, e, 0]), int(edges[b, e, 1])] += 2.0

    assert torch.equal(bias_vec, bias_ref)

    # shared single-row edges must broadcast across the batch identically
    shared = torch.tensor([[[0, 1], [2, 3], [-1, -1]]], dtype=torch.int32)
    bias_b = torch.zeros(4, sq, sk)
    _scatter_edges_(bias_b, shared, torch.tensor([2]), weight=1.0, sq=sq, sk=sk, require_kind=False)
    assert bias_b[:, 0, 1].tolist() == [1.0] * 4 and bias_b[:, 2, 3].tolist() == [1.0] * 4


def test_scatter_edges_raises_on_count_out_of_range():
    from cppmega.megatron.dsa_indexer_fused_patch import _scatter_edges_

    edges = torch.tensor([[[0, 1]]], dtype=torch.long)  # max_edges = 1
    with pytest.raises(ValueError, match="out of range"):
        _scatter_edges_(torch.zeros(1, 4, 4), edges, torch.tensor([5]), weight=1.0, sq=4, sk=4, require_kind=False)
    with pytest.raises(ValueError, match="out of range"):
        _scatter_edges_(torch.zeros(1, 4, 4), edges, torch.tensor([-1]), weight=1.0, sq=4, sk=4, require_kind=False)


def test_scatter_edges_raises_on_declared_edges_with_zero_slots():
    from cppmega.megatron.dsa_indexer_fused_patch import _scatter_edges_

    edges = torch.zeros(1, 0, 2, dtype=torch.long)  # zero edge slots
    with pytest.raises(ValueError, match="out of range"):
        _scatter_edges_(torch.zeros(1, 4, 4), edges, torch.tensor([1]), weight=1.0, sq=4, sk=4, require_kind=False)
    # all-zero counts with zero slots is a legit no-op (must not raise)
    bias = torch.zeros(1, 4, 4)
    _scatter_edges_(bias, edges, torch.tensor([0]), weight=1.0, sq=4, sk=4, require_kind=False)
    assert float(bias.sum()) == 0.0


def test_graph_route_bias_raises_on_out_of_range_edge():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structure_batch = {
        # count says 1 real edge, but dst=9 is out of range for seqlen 4.
        "graph_call_edges": torch.tensor([[[0, 9], [-1, -1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([4], dtype=torch.long),
    }
    with pytest.raises(ValueError, match="unavailable chunk"):
        build_graph_route_bias_from_structure_batch(
            structure_batch,
            batch_size=1,
            seqlen_q=4,
            seqlen_k=4,
            device=device,
            call_weight=2.0,
        )


def test_graph_route_bias_requires_structure_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(RuntimeError, match="no current cppmega structure batch"):
        build_graph_route_bias_from_structure_batch(
            None,
            batch_size=1,
            seqlen_q=4,
            seqlen_k=4,
            device=device,
        )


def test_structure_dataset_patch_imports_against_real_sibling_megatron():
    result = subprocess.run(
        [sys.executable, "-c", "import cppmega.megatron.structure_dataset_patch"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_dsa_backward_reuses_its_own_microbatch_graph(monkeypatch):
    from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

    class FakeFusedLoss:
        @staticmethod
        def forward(ctx):
            active = fused_patch._GRAPH_BATCH_OVERRIDE.get()
            ctx.forward_seen = active["graph_call_edges"].clone()
            return "forward"

        @staticmethod
        def backward(ctx):
            active = fused_patch._GRAPH_BATCH_OVERRIDE.get()
            ctx.backward_seen = active["graph_call_edges"].clone()
            return "backward"

    first = {
        "graph_call_edges": torch.tensor([[[0, 1]]]),
        "graph_call_edge_counts": torch.tensor([1]),
    }
    second = {
        "graph_call_edges": torch.tensor([[[2, 3]]]),
        "graph_call_edge_counts": torch.tensor([1]),
    }
    fake_module = SimpleNamespace(FusedDSAIndexerLoss=FakeFusedLoss)
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    current = {"batch": first}
    monkeypatch.setattr(
        fused_patch,
        "_capture_current_graph_batch",
        lambda: {
            key: value.detach().clone()
            for key, value in current["batch"].items()
        },
    )
    fused_patch._patch_fused_dsa_autograd(fake_module)
    ctx = SimpleNamespace()

    assert FakeFusedLoss.forward(ctx) == "forward"
    current["batch"] = second
    assert FakeFusedLoss.backward(ctx) == "backward"

    assert torch.equal(ctx.forward_seen, first["graph_call_edges"])
    assert torch.equal(ctx.backward_seen, first["graph_call_edges"])
    assert not torch.equal(ctx.backward_seen, second["graph_call_edges"])


def test_dsa_indexer_loss_wrapper_adds_weighted_graph_objective(monkeypatch):
    from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

    def base_indexer_loss(index_scores, *_args, **_kwargs):
        return index_scores.new_tensor(2.0)

    fake_module = SimpleNamespace(compute_dsa_indexer_loss=base_indexer_loss)
    structure_batch = {
        "graph_call_edges": torch.tensor([[[1, 0]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 1]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[1, 2]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([2], dtype=torch.long),
        "graph_document_ids": torch.tensor([[7, 7]], dtype=torch.long),
    }
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_RELATIONS", "call")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_WEIGHT", "0.5")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_BCE_WEIGHT", "1.0")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_COVERAGE_WEIGHT", "0.25")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_TOPK", "1")
    token = fused_patch._set_graph_batch_override(structure_batch)
    fused_patch._patch_dsa_graph_objective(fake_module)
    scores = torch.zeros((1, 2, 2), requires_grad=True)

    try:
        total = fake_module.compute_dsa_indexer_loss(scores)
        total.backward()
    finally:
        fused_patch._reset_graph_batch_override(token)

    assert total.item() > 2.0
    assert scores.grad is not None
    assert torch.count_nonzero(scores.grad).item() > 0


def test_patched_fused_dsa_backward_propagates_graph_only_gradients(monkeypatch):
    from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

    class TensorParallelGroup:
        @staticmethod
        def size():
            return 1

    pg_collection = SimpleNamespace(tp=TensorParallelGroup())
    dsa_module = SimpleNamespace()

    def patched_index_scores(q, weights, k, use_relu=True):
        sq, batch, _heads, _dim = q.shape
        graph_bias = fused_patch._current_graph_route_bias(
            batch_size=batch,
            seqlen_q=sq,
            seqlen_k=k.shape[0],
            device=q.device,
        )
        return fused_patch.compute_index_scores_fused_bf16(
            q,
            weights,
            k,
            use_relu=use_relu,
            graph_bias=graph_bias,
        )

    def base_indexer_loss(
        index_scores,
        _topk_indices,
        _query,
        _key,
        _softmax_scale,
        loss_coeff,
        _sparse_loss,
        _pg_collection,
        *,
        mask=None,
    ):
        del mask
        return index_scores.square().mean() * loss_coeff

    def forward_loss(
        q,
        weights,
        k,
        query,
        key,
        topk,
        softmax_scale,
        loss_coeff,
        mask,
        sparse_loss,
        pg_collection,
    ):
        index_scores = dsa_module._compute_index_scores(q, weights, k)
        masked_scores = index_scores if mask is None else index_scores + mask
        topk_indices = masked_scores.topk(topk, dim=-1).indices
        loss = dsa_module.compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            softmax_scale,
            loss_coeff,
            sparse_loss,
            pg_collection,
            mask=mask,
        )
        return topk_indices, loss

    def backward_loss(
        q,
        weights,
        k,
        _query,
        _key,
        _topk_indices,
        _softmax_scale,
        loss_coeff,
        _sparse_loss,
        _mask,
        grad_loss,
        _pg_collection,
    ):
        with torch.enable_grad():
            q_recompute = q.detach().requires_grad_(True)
            weights_recompute = weights.detach().requires_grad_(True)
            k_recompute = k.detach().requires_grad_(True)
            index_scores = dsa_module._compute_index_scores(
                q_recompute,
                weights_recompute,
                k_recompute,
            )
            kl_only_loss = index_scores.square().mean() * loss_coeff
            return torch.autograd.grad(
                kl_only_loss,
                (q_recompute, weights_recompute, k_recompute),
                grad_outputs=grad_loss,
            )

    class PinnedFusedDSAIndexerLoss(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q,
            weights,
            k,
            query,
            key,
            softmax_scale,
            topk,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection,
        ):
            topk_indices, loss = dsa_module.fwd_fused_indexer_loss_naive(
                q,
                weights,
                k,
                query,
                key,
                topk,
                softmax_scale,
                loss_coeff,
                mask,
                sparse_loss,
                pg_collection,
            )
            ctx.save_for_backward(q, weights, k, query, key, topk_indices)
            ctx.softmax_scale = softmax_scale
            ctx.loss_coeff = loss_coeff
            ctx.sparse_loss = sparse_loss
            ctx.mask = mask
            ctx.pg_collection = pg_collection
            ctx.use_relu = True
            return topk_indices, loss

        @staticmethod
        def backward(ctx, _grad_topk_indices, grad_loss):
            q, weights, k, query, key, topk_indices = ctx.saved_tensors
            grad_q, grad_weights, grad_k = dsa_module.bwd_fused_indexer_loss_naive(
                q,
                weights,
                k,
                query,
                key,
                topk_indices,
                ctx.softmax_scale,
                ctx.loss_coeff,
                ctx.sparse_loss,
                ctx.mask,
                grad_loss,
                ctx.pg_collection,
            )
            return (
                grad_q,
                grad_weights,
                grad_k,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    dsa_module._compute_index_scores = patched_index_scores
    dsa_module.compute_dsa_indexer_loss = base_indexer_loss
    dsa_module.fwd_fused_indexer_loss_naive = forward_loss
    dsa_module.bwd_fused_indexer_loss_naive = backward_loss
    dsa_module.FusedDSAIndexerLoss = PinnedFusedDSAIndexerLoss

    structure_batch = {
        "graph_call_edges": torch.tensor([[[1, 0]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 1]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[1, 2]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([2], dtype=torch.long),
        "graph_document_ids": torch.tensor([[7, 7]], dtype=torch.long),
    }
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_RELATIONS", "call")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_WEIGHT", "1")
    monkeypatch.setenv("CPPMEGA_DSA_INDEXER_LOSS_COEFF", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_LAYER_WEIGHT", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_BCE_WEIGHT", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_COVERAGE_WEIGHT", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_TOPK", "1")
    monkeypatch.setattr(
        fused_patch,
        "_capture_current_graph_batch",
        lambda: {
            key: value.detach().clone()
            for key, value in structure_batch.items()
        },
    )
    fused_patch._patch_dsa_graph_objective(dsa_module)
    fused_patch._patch_fused_dsa_autograd(dsa_module)

    q = torch.full((2, 1, 2, 3), 0.5, requires_grad=True)
    weights = torch.full((2, 1, 2), 0.25, requires_grad=True)
    k = torch.full((2, 1, 3), 0.75, requires_grad=True)
    query = torch.zeros((2, 1, 1, 3))
    key = torch.zeros((2, 1, 1, 3))

    _topk_indices, graph_only_loss = PinnedFusedDSAIndexerLoss.apply(
        q,
        weights,
        k,
        query,
        key,
        1.0,
        1,
        0.0,
        None,
        False,
        pg_collection,
    )
    graph_only_loss.backward()

    assert graph_only_loss.item() > 0.0
    for tensor in (q, weights, k):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad).item() > 0


def test_real_megatron_fused_dsa_preserves_mask_and_scales_parameter_gradients(
    monkeypatch,
    capsys,
):
    from cppmega.megatron import dsa_indexer_fused_patch as fused_patch
    from megatron.core.transformer import MLATransformerConfig
    from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module

    class TensorParallelGroup:
        @staticmethod
        def size():
            return 1

    class ContractFusedDSAIndexerLoss(dsa_module.FusedDSAIndexerLoss):
        pass

    class TinyLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, **_kwargs):
            super().__init__()
            values = torch.arange(output_size * input_size, dtype=torch.float32)
            self.weight = torch.nn.Parameter(
                values.reshape(output_size, input_size) / 20.0 + 0.05
            )

        def forward(self, inputs):
            return torch.nn.functional.linear(inputs, self.weight), None

    class TinyNorm(torch.nn.Module):
        def __init__(self, config, hidden_size, eps, **_kwargs):
            super().__init__()
            del config
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
            self.eps = eps

        def forward(self, inputs):
            return torch.nn.functional.layer_norm(
                inputs,
                (inputs.shape[-1],),
                self.weight,
                self.bias,
                self.eps,
            )

    class FakeRotaryEmbedding:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_rotary_seq_len(self, _inference, _decoder, inputs, *_args):
            return inputs.shape[0]

        def __call__(self, *_args, **_kwargs):
            return torch.empty(0)

    pg_collection = SimpleNamespace(tp=TensorParallelGroup(), cp=None)
    monkeypatch.setattr(
        dsa_module,
        "FusedDSAIndexerLoss",
        ContractFusedDSAIndexerLoss,
    )
    monkeypatch.setattr(dsa_module, "RotaryEmbedding", FakeRotaryEmbedding)
    monkeypatch.setattr(dsa_module, "rotate_activation", lambda tensor: tensor)
    monkeypatch.setattr(
        dsa_module.DSAIndexer,
        "__init__",
        dsa_module.DSAIndexer.__init__,
    )
    monkeypatch.setattr(
        dsa_module.DSAttention,
        "__init__",
        dsa_module.DSAttention.__init__,
    )
    monkeypatch.setattr(
        dsa_module.DSAttention,
        "forward",
        dsa_module.DSAttention.forward,
    )
    monkeypatch.setattr(
        dsa_module.DSAttention,
        fused_patch._RUNTIME_RECEIPT_PATCH_MARKER,
        False,
        raising=False,
    )

    def patched_index_scores(q, weights, k, use_relu=True):
        sq, batch, _heads, _dim = q.shape
        graph_bias = fused_patch._current_graph_route_bias(
            batch_size=batch,
            seqlen_q=sq,
            seqlen_k=k.shape[0],
            device=q.device,
        )
        return fused_patch.compute_index_scores_fused_bf16(
            q,
            weights,
            k,
            use_relu=use_relu,
            graph_bias=graph_bias,
        )

    monkeypatch.setattr(dsa_module, "_compute_index_scores", patched_index_scores)
    monkeypatch.setattr(
        dsa_module,
        "compute_dsa_indexer_loss",
        dsa_module.compute_dsa_indexer_loss,
    )

    seen_masks: list[torch.Tensor | None] = []
    original_graph_objective = fused_patch._graph_objective_from_index_scores

    def recording_graph_objective(index_scores, *, upstream_mask=None):
        seen_masks.append(upstream_mask)
        return original_graph_objective(
            index_scores,
            upstream_mask=upstream_mask,
        )

    monkeypatch.setattr(
        fused_patch,
        "_graph_objective_from_index_scores",
        recording_graph_objective,
    )
    structure_batch = {
        "graph_call_edges": torch.tensor([[[2, 0]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([3], dtype=torch.long),
        "graph_document_ids": torch.tensor([[7, 7, 7]], dtype=torch.long),
    }
    monkeypatch.setattr(
        fused_patch,
        "_capture_current_graph_batch",
        lambda: {
            key: value.detach().clone()
            for key, value in structure_batch.items()
        },
    )
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_RELATIONS", "call")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_WEIGHT", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_LAYER_WEIGHT", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_BCE_WEIGHT", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_COVERAGE_WEIGHT", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_TOPK", "1")
    fused_patch._patch_dsa_graph_objective(dsa_module)
    fused_patch._patch_fused_dsa_autograd(dsa_module)
    fused_patch._patch_dsa_runtime_receipts(dsa_module)

    upstream_mask = torch.tensor(
        [
            [0.0, float("-inf"), float("-inf")],
            [0.0, 0.0, float("-inf")],
            [0.0, float("-inf"), 0.0],
        ],
        dtype=torch.float32,
    )
    indexer_config = MLATransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        ffn_hidden_size=8,
        kv_channels=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_head_dim=2,
        qk_pos_emb_head_dim=2,
        v_head_dim=4,
        rope_type="rope",
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=2,
        dsa_indexer_topk=2,
        dsa_indexer_loss_coeff=0.001,
        use_cpu_initialization=True,
        transformer_impl="local",
    )
    indexer_submodules = dsa_module.DSAIndexerSubmodules(
        linear_wq_b=TinyLinear,
        linear_wk=TinyLinear,
        k_norm=TinyNorm,
        linear_weights_proj=TinyLinear,
    )
    x = torch.tensor(
        [
            [[0.2, 0.3, 0.4, 0.5]],
            [[0.5, 0.1, 0.2, 0.7]],
            [[0.9, 0.2, 0.8, 0.1]],
        ]
    )
    qr = torch.tensor(
        [
            [[0.1, 0.8, 0.2, 0.6]],
            [[0.4, 0.2, 0.7, 0.3]],
            [[0.9, 0.1, 0.5, 0.4]],
        ]
    )

    def run(indexer_weight: str, *, emit_receipts: bool):
        monkeypatch.setenv("CPPMEGA_DSA_INDEXER_LOSS_COEFF", indexer_weight)
        if emit_receipts:
            monkeypatch.setenv("CPPMEGA_H200_DSA_GRAPH_RECEIPTS", "1")
        else:
            monkeypatch.delenv("CPPMEGA_H200_DSA_GRAPH_RECEIPTS", raising=False)
        indexer = dsa_module.DSAIndexer(
            indexer_config,
            indexer_submodules,
            pg_collection,
        )
        indexer.cppmega_dsa_layer_number = 3
        indexer._apply_rope = MethodType(
            lambda _self, tensor, *_args: tensor,
            indexer,
        )
        q, k, weights = indexer.forward_before_topk(x, qr)
        query = torch.zeros((3, 1, 1, 2))
        key = torch.zeros((3, 1, 1, 2))

        layer_token = fused_patch._DSA_LAYER_CONTEXT.set(3)
        try:
            _topk_indices, graph_only_loss = ContractFusedDSAIndexerLoss.apply(
                q,
                weights,
                k,
                query,
                key,
                1.0,
                2,
                0.0,
                upstream_mask,
                False,
                pg_collection,
            )
        finally:
            fused_patch._DSA_LAYER_CONTEXT.reset(layer_token)
        graph_only_loss.backward()
        parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in indexer.named_parameters()
            if parameter.grad is not None
        }
        return graph_only_loss, parameter_grads

    unit_loss, unit_grads = run("1", emit_receipts=False)
    milli_loss, milli_grads = run("0.001", emit_receipts=True)

    assert unit_loss.item() > 0.0
    assert torch.isfinite(unit_loss)
    assert torch.isfinite(milli_loss)
    assert len(seen_masks) == 4
    assert seen_masks[0] is seen_masks[1]
    assert seen_masks[2] is seen_masks[3]
    for seen in seen_masks:
        assert seen is not None
        assert torch.equal(seen, upstream_mask)
        assert seen.untyped_storage().data_ptr() == (
            upstream_mask.untyped_storage().data_ptr()
        )
    assert unit_grads.keys() == milli_grads.keys()
    assert unit_grads.keys() == {
        "linear_wq_b.weight",
        "linear_wk.weight",
        "k_norm.weight",
        "k_norm.bias",
        "linear_weights_proj.weight",
    }
    for name in unit_grads:
        unit_grad = unit_grads[name]
        milli_grad = milli_grads[name]
        assert torch.isfinite(unit_grad).all()
        assert torch.isfinite(milli_grad).all()
        assert torch.count_nonzero(unit_grad).item() > 0
        assert torch.count_nonzero(milli_grad).item() > 0
        assert torch.linalg.vector_norm(milli_grad).item() == pytest.approx(
            torch.linalg.vector_norm(unit_grad).item() * 0.001,
            rel=1e-4,
            abs=1e-8,
        )

    from scripts.h200_megatron_preflight import _dsa_graph_gradient_evidence

    receipt_evidence = _dsa_graph_gradient_evidence(
        capsys.readouterr().out,
        expected_coefficient=0.001,
    )
    assert len(receipt_evidence["graph_losses"]) == 1
    assert len(receipt_evidence["per_indexer"]) == 1
    assert receipt_evidence["effective_coefficient"] == pytest.approx(0.001)
    assert receipt_evidence["graph_losses"][0] == pytest.approx(milli_loss.item())
    assert receipt_evidence["per_indexer"][0]["grad_norm"] > 0.0
    assert set(receipt_evidence["per_indexer"][0]["parameter_grad_norms"]) == set(
        milli_grads
    )


def test_dsa_graph_objective_receives_upstream_additive_mask(monkeypatch):
    from cppmega.megatron import dsa_indexer_fused_patch as fused_patch

    seen: dict[str, torch.Tensor] = {}

    def base_indexer_loss(index_scores, *, mask=None):
        del mask
        return index_scores.new_zeros(())

    def graph_objective(index_scores, *, upstream_mask=None):
        seen["mask"] = upstream_mask
        return index_scores.new_zeros(())

    fake_module = SimpleNamespace(compute_dsa_indexer_loss=base_indexer_loss)
    upstream_mask = torch.tensor(
        [[0.0, float("-inf")], [0.0, 0.0]], dtype=torch.float32
    )
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_DSA_GRAPH_AUX_ENABLED", "1")
    monkeypatch.setattr(
        fused_patch, "_graph_objective_from_index_scores", graph_objective
    )
    fused_patch._patch_dsa_graph_objective(fake_module)

    fake_module.compute_dsa_indexer_loss(
        torch.zeros((1, 2, 2)), mask=upstream_mask
    )

    assert seen["mask"] is upstream_mask


def test_fused_nam56r_shape():
    """Test a realistic NAM56R indexer shape: h=32 d=64 sq=sk=4096 b=1."""
    if not torch.cuda.is_available():
        return
    torch.manual_seed(2)
    device = torch.device("cuda")
    sq, sk, b, h, d = 4096, 4096, 1, 32, 64

    q = torch.randn(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(sq, b, h, dtype=torch.bfloat16, device=device)

    ref = _upstream_reference(q, w, k, use_relu=True)
    fused = compute_index_scores_fused_bf16(q, w, k, use_relu=True)

    abs_err = (ref - fused).abs().max().item()
    rel_err = abs_err / max(ref.abs().max().item(), 1e-6)
    print(f"nam56r-shape abs_err={abs_err:.3e} rel_err={rel_err:.3e}")
    assert rel_err < 5e-3, f"rel_err {rel_err} too high"


if __name__ == "__main__":
    test_fused_matches_reference_bf16_relu()
    test_fused_matches_reference_bf16_no_relu()
    test_fused_nam56r_shape()
    print("All parity tests passed.")
