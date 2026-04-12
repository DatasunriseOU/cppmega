"""Structural tests for ``scripts/remote_smoke_h200_dsa_fp8_indexer.sh``.

These tests intentionally do NOT spawn the remote script; they only assert
on the string contents so the cppmega launcher stays in sync with the
bench3 Megatron patching protocol the BF16 lane uses.
"""

from pathlib import Path


def _script() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "remote_smoke_h200_dsa_fp8_indexer.sh"
    ).read_text()


def test_fp8_smoke_targets_bench3_by_default():
    script = _script()
    assert 'REMOTE_HOST="${REMOTE_HOST:-h200_1}"' in script
    assert 'REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_1}"' in script


def test_fp8_smoke_sets_indexer_dtype_env_var():
    script = _script()
    assert "export CPPMEGA_DSA_INDEXER_DTYPE=fp8" in script


def test_fp8_smoke_applies_fp8_monkey_patch_before_pretrain_mamba():
    script = _script()
    # The wrapper module must import and call apply_dsa_fp8_patch BEFORE
    # the upstream pretrain_mamba.py body is exec'd.
    assert "from cppmega.megatron.dsa_fp8_patch import apply_dsa_fp8_patch" in script
    assert "_patched = apply_dsa_fp8_patch()" in script
    # And the upstream pretrain_mamba.py is exec'd afterwards.
    assert '"${REMOTE_ROOT}/cppmega-root/megatron-lm/pretrain_mamba.py"' in script
    apply_idx = script.index("_patched = apply_dsa_fp8_patch()")
    exec_idx = script.index('exec(compile(_src, "pretrain_mamba.py", "exec"), globals())')
    assert apply_idx < exec_idx, (
        "FP8 monkey patch must be applied before pretrain_mamba.py body runs"
    )


def test_fp8_smoke_plumbs_dsa_indexer_dtype_through_native_args():
    script = _script()
    assert "dsa_indexer_dtype='fp8'" in script
    assert "enable_dsa=True" in script


def test_fp8_smoke_reuses_bf16_megatron_patches():
    script = _script()
    # Same three idempotent patches as scripts/remote_smoke_h200_dsa_full_nam56r.sh
    assert "kw_args['apply_rope_fusion'] = False" in script
    assert 'if config.experimental_attention_variant == "dsa":\\n' in script
    assert 'if attention.metainfo.get("fuse_input_layernorm", False)\\n' in script


def test_fp8_smoke_uses_full_nam56r_layout_and_real_data():
    script = _script()
    # Real data + HF tokenizer, not a toy model.
    assert '${REMOTE_ROOT}/data/megatron/clang_semantic_4k_v10_train' in script
    assert "--tokenizer-type HuggingFaceTokenizer" in script
    assert "--hidden-size 3584" in script
    assert "--ffn-hidden-size 18944" in script
    assert "--num-attention-heads 28" in script
    assert "--seq-length 4096" in script
    assert "cppmega.megatron.nam56r_noconv_spec" in script


def test_fp8_smoke_keeps_cp_size_one_for_dsa():
    script = _script()
    # Megatron forbids DSA with context parallel, so the launcher must pin CP=1.
    assert "--context-parallel-size 1 \\" in script
