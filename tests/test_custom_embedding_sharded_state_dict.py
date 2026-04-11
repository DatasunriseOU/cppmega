"""Unit tests for ``CppMegaLanguageModelEmbedding.sharded_state_dict``.

These tests reproduce the distributed-checkpointing bug that surfaces on
bench3 for any NAM56R production run with PP>1 and MTP enabled:

    megatron.core.dist_checkpointing.core.CheckpointingException:
    Invalid sharding pattern validation. Errors: Invalid access pattern for
    ShardedTensor(key='embedding.cppmega_ngram_hash.table_offsets', ...)

The root cause is that the default ``MegatronModule.sharded_state_dict``
walker assigns ``replica_id=(0, tp_rank, dp_rank)`` on every PP stage that
holds the embedding, producing two "main replicas" when MTP replicates the
embedding on a non-pipeline-first stage. The fix (see
``cppmega/megatron/custom_embedding.py``) rewrites the leading replica-id
axis to 1 on the MTP copy, matching
``megatron.core.transformer.multi_token_prediction.tie_word_embeddings_state_dict``.

The tests use a lightweight fake ``MegatronModule`` + ``LanguageModelEmbedding``
so they can run in the local macOS dev environment without a real Megatron
install.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import pytest
import torch


# ---------------------------------------------------------------------------
# Fake Megatron wiring. This must be in place BEFORE importing the module
# under test so that ``custom_embedding`` picks up the fake LanguageModelEmbedding
# and parallel_state.
# ---------------------------------------------------------------------------


@dataclass
class _FakeShardedTensor:
    key: str
    data: torch.Tensor
    replica_id: Tuple[int, ...] = (0, 0, 0)


@dataclass
class _FakeConfig:
    hidden_size: int = 16
    hidden_dropout: float = 0.0
    sequence_parallel: bool = False
    clone_scatter_output_in_embedding: bool = False
    fp32_residual_connection: bool = False
    use_mup: bool = False
    mup_embedding_mult: float = 1.0
    perform_initialization: bool = True
    use_cpu_initialization: bool = False

    @staticmethod
    def embedding_init_method(weight: torch.Tensor) -> None:
        torch.nn.init.zeros_(weight)


class _FakeLanguageModelEmbedding(torch.nn.Module):
    """Minimal stand-in for megatron.core ``LanguageModelEmbedding``.

    Only what ``CppMegaLanguageModelEmbedding.__init__`` and the sharded-state-dict
    test path need is implemented: a ``config``, a trivial word-embedding, a
    placeholder ``tp_group``, and a ``sharded_state_dict`` method that walks
    ``named_parameters`` / ``named_buffers`` recursively and wraps each entry in
    a :class:`_FakeShardedTensor` with ``replica_id=(0, 0, 0)`` - mirroring the
    buggy default that Megatron's ``MegatronModule`` produces before the fix.
    """

    def __init__(
        self,
        *,
        config: _FakeConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: str = "rope",
        scatter_to_sequence_parallel: bool = False,
        tp_group: Any = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.position_embedding_type = position_embedding_type
        self.add_position_embedding = position_embedding_type == "learned_absolute"
        self.scatter_to_sequence_parallel = scatter_to_sequence_parallel
        self.reduce_scatter_embeddings = False
        self.tp_group = tp_group
        self.num_tokentypes = 0
        self.tokentype_embeddings = None
        self.word_embeddings = torch.nn.Embedding(vocab_size, config.hidden_size)
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                max_sequence_length, config.hidden_size
            )
        self.embedding_dropout = torch.nn.Dropout(config.hidden_dropout)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Dict | None = None,
    ) -> Dict[str, _FakeShardedTensor]:
        """Emit a flat dict mirroring MegatronModule's default walker.

        Every parameter and buffer under ``self`` is wrapped in a
        _FakeShardedTensor with ``replica_id=(0, 0, 0)``. This is exactly what
        the upstream default path does for the cppmega custom submodules
        before the replica-id rewrite.
        """

        result: Dict[str, _FakeShardedTensor] = {}
        for name, tensor in self.state_dict().items():
            key = f"{prefix}{name}"
            result[key] = _FakeShardedTensor(key=key, data=tensor, replica_id=(0, 0, 0))
        return result


def _install_fake_megatron(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Install a fake ``megatron`` package hierarchy into ``sys.modules``.

    Returns a namespace holding the mutable flag controlling what
    ``is_pipeline_first_stage`` returns for the duration of the test.
    """

    state = types.SimpleNamespace(is_first_stage=True)

    megatron_pkg = types.ModuleType("megatron")
    core_pkg = types.ModuleType("megatron.core")
    tp_pkg = types.ModuleType("megatron.core.tensor_parallel")
    parallel_state_pkg = types.ModuleType("megatron.core.parallel_state")
    models_pkg = types.ModuleType("megatron.core.models")
    common_pkg = types.ModuleType("megatron.core.models.common")
    embeddings_pkg = types.ModuleType("megatron.core.models.common.embeddings")
    lme_pkg = types.ModuleType(
        "megatron.core.models.common.embeddings.language_model_embedding"
    )

    lme_pkg.LanguageModelEmbedding = _FakeLanguageModelEmbedding

    def _is_pipeline_first_stage(ignore_virtual: bool = True, vp_stage: Any = None) -> bool:
        return state.is_first_stage

    parallel_state_pkg.is_pipeline_first_stage = _is_pipeline_first_stage

    # Minimal tensor_parallel surface used by the forward path (not exercised
    # in these tests but imported at module level).
    tp_pkg.scatter_to_sequence_parallel_region = lambda x, group=None: x

    class _FakeRngTracker:
        def fork(self):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield

            return _cm()

    tp_pkg.get_cuda_rng_tracker = lambda: _FakeRngTracker()

    core_pkg.tensor_parallel = tp_pkg
    core_pkg.parallel_state = parallel_state_pkg

    monkeypatch.setitem(sys.modules, "megatron", megatron_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core", core_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tp_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core.models", models_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core.models.common", common_pkg)
    monkeypatch.setitem(
        sys.modules, "megatron.core.models.common.embeddings", embeddings_pkg
    )
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.models.common.embeddings.language_model_embedding",
        lme_pkg,
    )

    # Force a re-import of ``cppmega.megatron.custom_embedding`` so it picks up
    # the fakes (it caches ``LanguageModelEmbedding`` and ``parallel_state`` at
    # import time).
    monkeypatch.delitem(sys.modules, "cppmega.megatron.custom_embedding", raising=False)

    return state


@pytest.fixture()
def fake_megatron(monkeypatch):
    state = _install_fake_megatron(monkeypatch)
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_ORDERS", "2,3")
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_HEADS", "2")
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_TABLE_SIZE", "97")
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_EMBED_DIM", "4")
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_STRUCTURE_COMPONENTS", "core")
    return state


def _build_embedding():
    from cppmega.megatron.custom_embedding import CppMegaLanguageModelEmbedding

    return CppMegaLanguageModelEmbedding(
        config=_FakeConfig(),
        vocab_size=64,
        max_sequence_length=8,
        position_embedding_type="rope",
        scatter_to_sequence_parallel=False,
        tp_group=None,
    )


def _assert_all_main(state_dict: Dict[str, _FakeShardedTensor], *, prefix: str) -> None:
    matching = [key for key in state_dict if key.startswith(prefix)]
    assert matching, f"expected at least one sharded tensor under prefix {prefix!r}"
    for key in matching:
        replica = state_dict[key].replica_id
        assert replica[0] == 0, f"{key}: expected main replica, got {replica}"


def _assert_all_copy(state_dict: Dict[str, _FakeShardedTensor], *, prefix: str) -> None:
    matching = [key for key in state_dict if key.startswith(prefix)]
    assert matching, f"expected at least one sharded tensor under prefix {prefix!r}"
    for key in matching:
        replica = state_dict[key].replica_id
        assert replica[0] == 1, f"{key}: expected copy replica, got {replica}"


def test_sharded_state_dict_pp_first_stage_marks_cppmega_tensors_main(fake_megatron):
    fake_megatron.is_first_stage = True
    embedding = _build_embedding()
    result = embedding.sharded_state_dict(prefix="embedding.")

    # Expected cppmega submodule keys must be present and all marked as the
    # main replica (replica_id[0] == 0).
    _assert_all_main(result, prefix="embedding.cppmega_ngram_hash.")
    _assert_all_main(result, prefix="embedding.cppmega_structure.")

    # Sanity: at least the buffers that showed up in the bench3 error log are
    # present so the test actually covers the regression surface.
    expected_keys = {
        "embedding.cppmega_ngram_hash.table_offsets",
        "embedding.cppmega_ngram_hash.table_sizes_t",
        "embedding.cppmega_ngram_hash.hash_mults",
        "embedding.cppmega_ngram_hash.hash_bias",
        "embedding.cppmega_ngram_hash.order_for_table",
        "embedding.cppmega_ngram_hash.order_mask",
        "embedding.cppmega_ngram_hash.unified_table.weight",
        "embedding.cppmega_ngram_hash.out_proj.weight",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_sharded_state_dict_mtp_stage_marks_cppmega_tensors_as_copies(fake_megatron):
    fake_megatron.is_first_stage = False
    embedding = _build_embedding()
    result = embedding.sharded_state_dict(prefix="embedding.")

    # With MTP enabled the embedding is built on both pre_process and
    # mtp_process stages. The MTP copy must have replica_id[0] == 1 so that
    # validate_sharding_integrity sees exactly one main replica per DP rank.
    _assert_all_copy(result, prefix="embedding.cppmega_ngram_hash.")
    _assert_all_copy(result, prefix="embedding.cppmega_structure.")


def test_sharded_state_dict_preserves_non_cppmega_replica_ids(fake_megatron):
    """Keys outside cppmega_* prefixes must not be mutated by our override."""

    fake_megatron.is_first_stage = False  # worst case: MTP stage
    embedding = _build_embedding()
    result = embedding.sharded_state_dict(prefix="embedding.")

    for key, value in result.items():
        if key.startswith("embedding.cppmega_"):
            continue
        assert value.replica_id == (0, 0, 0), (
            f"non-cppmega key {key!r} should keep upstream replica_id (0,0,0), "
            f"got {value.replica_id}"
        )


def test_sharded_state_dict_combined_ranks_have_exactly_one_main_replica(fake_megatron):
    """End-to-end: across pre_process and mtp_process stages, every cppmega
    sharded key must have exactly one main replica, matching the invariant
    that ``validate_sharding_integrity._compute_shards_access`` checks.
    """

    fake_megatron.is_first_stage = True
    main_state = _build_embedding().sharded_state_dict(prefix="embedding.")

    fake_megatron.is_first_stage = False
    copy_state = _build_embedding().sharded_state_dict(prefix="embedding.")

    cppmega_keys = [k for k in main_state if k.startswith("embedding.cppmega_")]
    assert cppmega_keys, "no cppmega sharded tensors emitted"

    for key in cppmega_keys:
        main_count = int(all(r == 0 for r in main_state[key].replica_id))
        copy_count = int(all(r == 0 for r in copy_state[key].replica_id))
        total_main = main_count + copy_count
        assert total_main == 1, (
            f"{key}: expected exactly one main replica across PP stages, "
            f"got main_stage={main_state[key].replica_id} "
            f"mtp_stage={copy_state[key].replica_id}"
        )


def test_sharded_state_dict_missing_parallel_state_is_safe(monkeypatch):
    """When Megatron is not importable (local dev), the override must be a no-op
    rather than raising - it should still return the base dict.
    """

    state = _install_fake_megatron(monkeypatch)
    # Remove parallel_state from sys.modules and the already-imported reference
    # inside custom_embedding, simulating the "parallel state not initialized"
    # code path.
    state.is_first_stage = True
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_ORDERS", "2,3")
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_HEADS", "2")
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_TABLE_SIZE", "97")
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_EMBED_DIM", "4")

    from cppmega.megatron import custom_embedding

    def _raise(*args, **kwargs):
        raise AssertionError("parallel_state not initialized")

    monkeypatch.setattr(
        custom_embedding._mcore_parallel_state,
        "is_pipeline_first_stage",
        _raise,
    )

    embedding = custom_embedding.CppMegaLanguageModelEmbedding(
        config=_FakeConfig(),
        vocab_size=64,
        max_sequence_length=8,
        position_embedding_type="rope",
        scatter_to_sequence_parallel=False,
        tp_group=None,
    )
    result = embedding.sharded_state_dict(prefix="embedding.")
    # Falls back to is_first_stage=True, so cppmega tensors are main.
    _assert_all_main(result, prefix="embedding.cppmega_ngram_hash.")
