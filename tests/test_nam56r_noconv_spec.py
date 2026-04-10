"""Source-level and importability tests for ``nam56r_noconv_spec``.

All heavy-weight tests (actual forward pass, megatron config construction)
are gated on ``megatron`` being importable, since the module only runs on the
remote H200 bench.  The source-level tests verify that the file:

  1. Imports from ``noconv_mamba_mixer.NoConvMamba3BCMixer`` (not Author kernels)
  2. Builds a selective mixer with the correct class name
  3. Uses the vanilla ``mamba_stack_spec`` as the upstream substrate so TE
     norms / attention / MoE stay fused
  4. Exposes a ``build_cppmega_nam56r_noconv_stack_spec`` callable and a
     module-level alias suitable for ``--spec``

Importability tests (skipped without megatron installed) ensure the spec
builder doesn't raise at import and the resulting spec has the expected shape.
"""

import ast
import importlib
import pathlib

import pytest

_has_megatron = importlib.util.find_spec("megatron") is not None
_has_mamba_ssm = importlib.util.find_spec("mamba_ssm") is not None

_repo_root = pathlib.Path(__file__).parent.parent
_spec_path = _repo_root / "cppmega" / "megatron" / "nam56r_noconv_spec.py"
_mixer_path = _repo_root / "cppmega" / "megatron" / "noconv_mamba_mixer.py"


def _read_spec_source() -> str:
    return _spec_path.read_text()


def _read_mixer_source() -> str:
    return _mixer_path.read_text()


def _extract_class_body_code(source: str, class_name: str) -> str:
    """Return only the executable statements (no docstrings / no comments)
    from a class body, identified by name, as a single joined string.

    Used to check invariants like 'the implementation does not reference
    conv1d' without being tripped by class/method docstrings that discuss it.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            # Drop docstring (first Expr(Constant(str)) in the body)
            body = list(node.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]

            # For each method, also drop its docstring so method docs don't
            # leak keywords into the scanned region.
            def _strip_method_docs(stmt):
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if (
                        stmt.body
                        and isinstance(stmt.body[0], ast.Expr)
                        and isinstance(stmt.body[0].value, ast.Constant)
                        and isinstance(stmt.body[0].value.value, str)
                    ):
                        stmt.body = stmt.body[1:]
                return stmt

            body = [_strip_method_docs(s) for s in body]
            return "\n".join(ast.unparse(s) for s in body)
    raise AssertionError(f"class {class_name!r} not found in source")


# ---------------------------------------------------------------------------
# Source-level invariants (no imports required)
# ---------------------------------------------------------------------------


class TestNoConvMamba3BCMixerSource:
    """Verify ``NoConvMamba3BCMixer`` class exists and matches the Branch-B contract."""

    def test_class_defined(self):
        assert "class NoConvMamba3BCMixer(NoConvMambaMixer):" in _read_mixer_source()

    def test_has_bc_norm_and_bias_parameters(self):
        text = _read_mixer_source()
        # Parameters must be declared inside NoConvMamba3BCMixer.__init__
        # (not just in Mamba3NoConvMixer which is the buggy variant)
        marker = "class NoConvMamba3BCMixer(NoConvMambaMixer):"
        assert marker in text
        post = text.split(marker, 1)[1]
        for name in ("B_norm_weight", "C_norm_weight", "B_bias", "C_bias"):
            assert f"self.{name} = nn.Parameter" in post, f"{name} not declared"

    def test_bias_initialised_to_zeros(self):
        """B_bias / C_bias must be zero-init so the fresh model is identity vs vanilla."""
        text = _read_mixer_source()
        marker = "class NoConvMamba3BCMixer(NoConvMambaMixer):"
        post = text.split(marker, 1)[1]
        # Both B_bias and C_bias initialisers must use torch.zeros
        b_bias_block = post.split("self.B_bias = nn.Parameter(", 1)[1]
        b_bias_block = b_bias_block.split("setattr(p,", 1)[0]
        assert "torch.zeros(" in b_bias_block, "B_bias must be zero-initialised"

        c_bias_block = post.split("self.C_bias = nn.Parameter(", 1)[1]
        c_bias_block = c_bias_block.split("setattr(p,", 1)[0]
        assert "torch.zeros(" in c_bias_block, "C_bias must be zero-initialised"

    def test_norm_weight_initialised_to_ones(self):
        text = _read_mixer_source()
        marker = "class NoConvMamba3BCMixer(NoConvMambaMixer):"
        post = text.split(marker, 1)[1]
        b_norm_block = post.split("self.B_norm_weight = nn.Parameter(", 1)[1]
        b_norm_block = b_norm_block.split("self.C_norm_weight", 1)[0]
        assert "torch.ones(" in b_norm_block

    def test_applies_qknorm_and_bias_in_ssm_path(self):
        """The rms_norm + bias add must land inside ``_ssm_noconv`` before the kernel call."""
        text = _read_mixer_source()
        marker = "class NoConvMamba3BCMixer(NoConvMambaMixer):"
        post = text.split(marker, 1)[1]
        assert "def _ssm_noconv" in post
        assert "F.rms_norm(B" in post
        assert "F.rms_norm(C" in post
        assert "self.B_norm_weight" in post
        assert "self.B_bias" in post
        assert "self.C_norm_weight" in post
        assert "self.C_bias" in post

    def test_bc_downcast_to_input_dtype(self):
        """The preprocessed B and C must be explicitly cast back to the input dtype.

        Rationale: ``norm_weight`` and bias parameters default to fp32, so
        ``F.rms_norm(B) * self.B_norm_weight + self.B_bias`` type-promotes the
        bf16 input ``B`` to fp32.  ``mamba_chunk_scan_combined``'s Triton
        backward kernel then hits a dtype mismatch in ``tl.dot(dout, c)``
        because ``dout`` is bf16 but ``c`` was saved as fp32.  A regression
        that removes the ``.to(bc_dtype)`` cast silently breaks all backward
        passes on H200 -- the forward runs fine, but training crashes on the
        first loss.backward().

        This test pins the cast so a future refactor can't drop it accidentally.
        """
        code = _extract_class_body_code(_read_mixer_source(), "NoConvMamba3BCMixer")
        # Must capture the B dtype before the preprocessing chain
        assert "bc_dtype = B.dtype" in code, (
            "NoConvMamba3BCMixer must snapshot B.dtype before the QK-norm / bias "
            "ops so we can cast the result back"
        )
        # Both B and C must be cast back to bc_dtype after the preprocessing
        # (either via ``.to(bc_dtype)`` or an equivalent pattern)
        assert code.count(".to(bc_dtype)") >= 2, (
            "NoConvMamba3BCMixer must cast BOTH B and C back to the input dtype "
            "after QK-norm + bias; without this, mamba_chunk_scan_combined's "
            "Triton backward kernel crashes with bf16/fp32 dtype mismatch"
        )

    def test_calls_mamba_chunk_scan_combined(self):
        """Must use the vanilla Mamba-2 SSD kernel, NOT mamba3_siso_combined."""
        code = _extract_class_body_code(_read_mixer_source(), "NoConvMamba3BCMixer")
        assert "mamba_chunk_scan_combined(" in code
        assert "mamba3_siso_combined" not in code
        assert "mamba3_mimo_combined" not in code

    def test_no_conv1d_in_ssm_path(self):
        code = _extract_class_body_code(_read_mixer_source(), "NoConvMamba3BCMixer")
        assert "conv1d" not in code.lower()
        assert "causal_conv1d" not in code

    def test_no_trap_or_rope_or_data_dep_a(self):
        """Feature surface must match the 127k recipe description exactly:
        QK-Norm + B/C bias ONLY.  Any trap/angles/dd_A in the class body would
        pull in the buggy Mamba3NoConvMixer preprocessing path."""
        code = _extract_class_body_code(_read_mixer_source(), "NoConvMamba3BCMixer")
        assert "trap" not in code.lower()
        assert "dd_a" not in code.lower()
        assert "angles" not in code.lower()
        assert "_mamba3_scan" not in code
        assert "_preprocess_bc_mamba3" not in code


class TestNam56rNoconvSpecSource:
    """Source-level checks on nam56r_noconv_spec.py."""

    def test_imports_noconv_mamba3_bc_mixer(self):
        text = _read_spec_source()
        assert "NoConvMamba3BCMixer" in text
        assert "from cppmega.megatron.noconv_mamba_mixer import" in text

    def test_does_not_import_author_mamba3(self):
        """The noconv spec must NOT wrap Author Mamba3 kernels."""
        text = _read_spec_source()
        assert "AuthorMamba3Mixer" not in text
        assert "mamba_ssm.modules.mamba3" not in text

    def test_defines_selective_mixer_class(self):
        text = _read_spec_source()
        assert "class CppMegaNoConvSelectiveMambaMixer" in text

    def test_selector_routes_r_layers_to_m2rnn(self):
        text = _read_spec_source()
        assert "CppMegaM2RNNMixer" in text
        assert "r_layer_indices" in text

    def test_defines_build_function(self):
        text = _read_spec_source()
        assert "def build_cppmega_nam56r_noconv_stack_spec" in text

    def test_has_module_level_alias(self):
        """The spec builder alias is required for ``--spec MODULE NAME`` form."""
        text = _read_spec_source()
        # Either the callable itself or an alias to it must exist at module top-level
        assert (
            "cppmega_nam56r_noconv_stack_spec = build_cppmega_nam56r_noconv_stack_spec"
            in text
        )

    def test_uses_mamba_stack_spec_substrate(self):
        """Upstream TE fused submodules (attention, MoE, norms) must be preserved."""
        text = _read_spec_source()
        assert "mamba_stack_spec" in text
        assert "upstream.gdn_layer" in text
        assert "upstream.attention_layer" in text
        assert "upstream.mlp_layer" in text
        assert "upstream.moe_layer" in text
        assert "upstream.mtp_block_spec" in text

    def test_uses_mambamixer_submodules_for_linears(self):
        """in_proj/out_proj come from upstream TE specs, not ColumnParallelLinear strings."""
        text = _read_spec_source()
        # We route submodules through upstream_mamba_sub.mixer.submodules which is
        # the TE-fused MambaMixerSubmodules (TELayerNormColumnParallelLinear /
        # TERowParallelLinear) — no hardcoded linear class names.
        assert "upstream_mamba_sub.mixer.submodules" in text


# ---------------------------------------------------------------------------
# Importability (megatron + mamba_ssm required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_megatron, reason="megatron not installed locally")
@pytest.mark.skipif(not _has_mamba_ssm, reason="mamba_ssm not installed locally")
def test_spec_module_imports():
    from cppmega.megatron.nam56r_noconv_spec import (
        CppMegaNoConvSelectiveMambaMixer,
        build_cppmega_nam56r_noconv_stack_spec,
        cppmega_nam56r_noconv_stack_spec,
    )
    assert CppMegaNoConvSelectiveMambaMixer is not None
    assert callable(build_cppmega_nam56r_noconv_stack_spec)
    # The module-level alias should be the same callable
    assert cppmega_nam56r_noconv_stack_spec is build_cppmega_nam56r_noconv_stack_spec


@pytest.mark.skipif(not _has_megatron, reason="megatron not installed locally")
@pytest.mark.skipif(not _has_mamba_ssm, reason="mamba_ssm not installed locally")
def test_noconv_mamba3_bc_mixer_importable():
    from cppmega.megatron.noconv_mamba_mixer import NoConvMamba3BCMixer
    # Class must be a subclass of the base NoConvMambaMixer (inherits in_proj layout)
    from cppmega.megatron.noconv_mamba_mixer import NoConvMambaMixer
    assert issubclass(NoConvMamba3BCMixer, NoConvMambaMixer)


@pytest.mark.skipif(not _has_megatron, reason="megatron not installed locally")
@pytest.mark.skipif(not _has_mamba_ssm, reason="mamba_ssm not installed locally")
def test_selector_signature_matches_te_spec():
    """``CppMegaNoConvSelectiveMambaMixer`` must accept the same kwargs Megatron
    passes to Mamba mixers (``config``, ``d_model``, ``submodules``, ``layer_number``,
    ``pg_collection``, ``pp_layer_offset``, ``r_layer_indices``)."""
    import inspect
    from cppmega.megatron.nam56r_noconv_spec import CppMegaNoConvSelectiveMambaMixer
    sig = inspect.signature(CppMegaNoConvSelectiveMambaMixer.__init__)
    params = set(sig.parameters.keys())
    required = {
        "self", "config", "d_model", "submodules", "layer_number",
        "pg_collection", "pp_layer_offset", "r_layer_indices",
    }
    assert required.issubset(params), f"Missing params: {required - params}"
