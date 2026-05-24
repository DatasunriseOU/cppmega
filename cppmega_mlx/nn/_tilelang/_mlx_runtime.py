# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""TileLang Metal runtime adapters.

The Triton -> TileLang -> Metal -> MLX numeric harness emits a complete
Metal Shading Language (MSL) function with positional buffer parameters
named ``A``, ``B``, ``C``, ... -- one per ``T.Tensor`` argument of the
TileLang ``@T.prim_func`` -- followed by Metal builtin attributes
(``thread_position_in_grid`` etc.). ``mx.fast.metal_kernel`` builds the
kernel signature itself from caller-supplied ``input_names`` /
``output_names`` and only takes the *body* of the kernel; by convention
those names must be ``inp0``, ``inp1``, ..., ``out0``, ``out1``, ...

The legacy helper in this module bridges those two worlds.
``wrap_tilelang_metal_kernel`` takes
a TileLang compile artifact (with ``.kernel_source`` / ``.rt_mod``),
parses the device-qualified parameter names out of the emitted ``kernel
void`` signature, renames the first ``input_count`` to ``inp0..inpN-1``
and the last ``output_count`` to ``out0..outM-1`` (token-level rewrite
on the kernel body), and hands the renamed body to
``mx.fast.metal_kernel``. The resulting callable accepts ``mx.array``
inputs and returns ``mx.array`` outputs.

This is the path-A (numeric harness) sibling of
``cppmega_mlx.nn._tilelang.fp8_vecmat_path_c._fp8_vecmat_kernel_for``,
which performs the same buffer-name dance for the production fp8 vecmat
kernel via ``_msl_transform`` (its body is hand-authored / IR-rewritten
with the right ``inp*`` / ``out*`` names already, so the rename is a
no-op there). For TileLang's stock Metal emitter the names are positional,
so we rename them here.

New production Path C code should use the native TVM-FFI boundary instead:
``NativeTileLangKernel`` wraps a TileLang
``tilelang.compile(..., execution_backend="tvm_ffi", out_idx=...)`` artifact
and requires caller-owned outputs by default. It does not rewrite MSL and does
not build ``mx.fast.metal_kernel``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence


__all__ = [
    "MLXRuntimeError",
    "NativeTileLangKernel",
    "NativeTileLangRuntimeError",
    "TileLangMetalAdapter",
    "normalize_out_idx",
    "_rewrite_tilelang_metal_to_mlx",
    "wrap_tilelang_metal_kernel",
]


class MLXRuntimeError(RuntimeError):
    """Raised when the TileLang Metal source cannot be adapted to MLX."""


class NativeTileLangRuntimeError(RuntimeError):
    """Raised when the native TileLang TVM-FFI boundary is misused."""


def normalize_out_idx(
    out_idx: int | Sequence[int] | None,
    *,
    num_params: int | None = None,
) -> tuple[int, ...]:
    """Normalize TileLang ``out_idx`` values to positive parameter indices."""

    if out_idx is None:
        return ()
    if isinstance(out_idx, int):
        raw_indices = (out_idx,)
    else:
        raw_indices = tuple(int(idx) for idx in out_idx)

    normalized: list[int] = []
    for idx in raw_indices:
        if idx < 0:
            if num_params is None:
                normalized.append(idx)
                continue
            idx = num_params + idx
        if num_params is not None and (idx < 0 or idx >= num_params):
            raise NativeTileLangRuntimeError(
                f"out_idx {idx} is outside the PrimFunc parameter range "
                f"[0, {num_params})"
            )
        normalized.append(idx)
    if len(set(normalized)) != len(normalized):
        raise NativeTileLangRuntimeError(
            f"out_idx contains duplicate result positions: {tuple(normalized)}"
        )
    return tuple(normalized)


def _owner_output_sequence(out: Any, expected_count: int) -> tuple[Any, ...]:
    if expected_count == 0:
        return ()
    if expected_count == 1:
        if isinstance(out, (list, tuple)):
            if len(out) != 1:
                raise NativeTileLangRuntimeError(
                    f"kernel expects 1 owner output, got {len(out)}"
                )
            return (out[0],)
        return (out,)
    if not isinstance(out, (list, tuple)):
        raise NativeTileLangRuntimeError(
            f"kernel expects {expected_count} owner outputs, but out= is not a sequence"
        )
    if len(out) != expected_count:
        raise NativeTileLangRuntimeError(
            f"kernel expects {expected_count} owner outputs, got {len(out)}"
        )
    return tuple(out)


def _validate_owner_result(result: Any, expected_outputs: tuple[Any, ...]) -> Any:
    """Fail if a native call did not return the caller-owned output objects."""

    if not expected_outputs:
        return result
    if len(expected_outputs) == 1:
        expected = expected_outputs[0]
        if result is expected:
            return result
        if isinstance(result, (list, tuple)) and len(result) == 1 and result[0] is expected:
            return result
        raise NativeTileLangRuntimeError(
            "native TileLang TVM-FFI call did not return the caller-owned output"
        )
    if not isinstance(result, (list, tuple)) or len(result) != len(expected_outputs):
        raise NativeTileLangRuntimeError(
            "native TileLang TVM-FFI call returned an unexpected output shape"
        )
    for pos, (got, expected) in enumerate(zip(result, expected_outputs, strict=True)):
        if got is not expected:
            raise NativeTileLangRuntimeError(
                "native TileLang TVM-FFI call did not return caller-owned "
                f"output at result position {pos}"
            )
    return result


@dataclass(frozen=True)
class NativeTileLangKernel:
    """Strict native wrapper for TileLang ``execution_backend="tvm_ffi"``.

    The wrapper deliberately requires caller-owned ``out=`` buffers whenever
    the PrimFunc has result indices. That keeps the boundary honest: no MSL
    text rewrite, no ``mx.fast.metal_kernel`` allocation semantics, and no
    hidden Python-side staging to satisfy a wrapper ABI.
    """

    artifact: Any
    result_indices: tuple[int, ...]
    num_params: int
    target: Any
    allow_graph_outputs: bool = False

    def __call__(
        self,
        *inputs: Any,
        out: Any | None = None,
        _tilelang_metal_command_buffer_domain: Any | None = None,
    ) -> Any:
        expected_inputs = self.num_params - len(self.result_indices)
        using_full_abi_outputs = (
            out is None and bool(self.result_indices) and len(inputs) == self.num_params
        )
        if (
            self.result_indices
            and out is None
            and not using_full_abi_outputs
            and not self.allow_graph_outputs
        ):
            raise NativeTileLangRuntimeError(
                "native TileLang TVM-FFI kernels require caller-owned out= "
                "buffers by default; pass allow_graph_outputs=True only for "
                "an explicit native MLX graph-output route"
            )

        if out is not None and len(inputs) != expected_inputs:
            raise NativeTileLangRuntimeError(
                f"native TileLang TVM-FFI kernel expected {expected_inputs} "
                f"inputs with out=, got {len(inputs)}"
            )
        if out is None and len(inputs) not in {expected_inputs, self.num_params}:
            raise NativeTileLangRuntimeError(
                f"native TileLang TVM-FFI kernel expected {expected_inputs} "
                f"inputs, or {self.num_params} full-ABI arguments including "
                f"outputs, got {len(inputs)}"
            )

        kwargs: dict[str, Any] = {}
        expected_outputs: tuple[Any, ...] = ()
        if out is not None:
            expected_outputs = _owner_output_sequence(out, len(self.result_indices))
            kwargs["out"] = out
        elif self.result_indices and len(inputs) == self.num_params:
            expected_outputs = tuple(inputs[idx] for idx in self.result_indices)
        if _tilelang_metal_command_buffer_domain is not None:
            kwargs["_tilelang_metal_command_buffer_domain"] = (
                _tilelang_metal_command_buffer_domain
            )

        try:
            result = self.artifact(*inputs, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- preserve the original cause
            raise NativeTileLangRuntimeError(
                f"native TileLang TVM-FFI dispatch failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _validate_owner_result(result, expected_outputs)


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------


# ``kernel void <name>(<sig>) { <body> }`` --- the canonical TileLang Metal
# emitter shape. We do not require ``[[ kernel ]]`` annotations because
# TileLang emits the bare ``kernel void`` form (matches ``_msl_transform``).
_KERNEL_DEF_RE = re.compile(r"kernel\s+void\s+(?P<name>\w+)\s*\(", re.MULTILINE)
_COMMENT_OR_STRING_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)


def _mask_comments_and_strings(src: str) -> str:
    """Replace comment/string spans with same-length whitespace.

    Preserves source offsets so brace/paren matching can be done against
    the masked copy without disturbing positions in the real source.
    """

    return _COMMENT_OR_STRING_RE.sub(lambda m: " " * len(m.group(0)), src)


def _split_kernel(src: str) -> tuple[str, str, str, str]:
    """Split ``src`` into ``(prelude, kernel_name, signature, body)``.

    ``signature`` excludes the surrounding ``(`` / ``)`` and ``body`` excludes
    the outer ``{`` / ``}``. Raises :class:`MLXRuntimeError` when the source
    does not match the expected ``kernel void name(...) { ... }`` shape.
    """

    masked = _mask_comments_and_strings(src)
    match = _KERNEL_DEF_RE.search(masked)
    if match is None:
        raise MLXRuntimeError(
            "unsupported TileLang Metal pattern: no 'kernel void' declaration found"
        )
    # Reject multi-kernel sources: emit one kernel per artifact, please.
    second = _KERNEL_DEF_RE.search(masked, match.end())
    if second is not None:
        raise MLXRuntimeError(
            "unsupported TileLang Metal pattern: multiple 'kernel void' "
            "declarations in one source"
        )

    kernel_name = match.group("name")
    prelude = src[: match.start()].rstrip()

    sig_start = match.end()
    depth = 1
    i = sig_start
    while i < len(masked) and depth > 0:
        ch = masked[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    if depth != 0:
        raise MLXRuntimeError(
            "unsupported TileLang Metal pattern: unbalanced parens in signature"
        )
    signature = src[sig_start : i - 1]

    j = i
    while j < len(src) and src[j].isspace():
        j += 1
    if j >= len(src) or src[j] != "{":
        raise MLXRuntimeError(
            "unsupported TileLang Metal pattern: missing body '{' after signature"
        )
    body_start = j + 1
    depth = 1
    j += 1
    while j < len(masked) and depth > 0:
        ch = masked[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        j += 1
    if depth != 0:
        raise MLXRuntimeError(
            "unsupported TileLang Metal pattern: unbalanced braces in body"
        )
    body = src[body_start : j - 1]
    return prelude, kernel_name, signature, body


# Per-decl strip of Metal attributes (``[[ buffer(0) ]]`` etc.) and Metal
# qualifiers (``device``, ``constant``, ``threadgroup``, ``__restrict``,
# ``const``). Matches the strategy used in
# ``_msl_transform._parse_buffer_param_names`` but inlined here so this
# module is independent of the legacy Path-C lowering helper.
_ATTR_RE = re.compile(r"\[\[[^\]]*\]\]")


def _strip_attribute_markers(decl: str) -> str:
    return _ATTR_RE.sub(" ", decl)


def _split_signature_decls(sig_text: str) -> list[str]:
    """Split a kernel signature into top-level comma-separated decls."""

    decls: list[str] = []
    depth = 0
    last = 0
    masked = _mask_comments_and_strings(sig_text)
    for i, ch in enumerate(masked):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth == 0:
            decls.append(sig_text[last:i])
            last = i + 1
    if last < len(sig_text):
        decls.append(sig_text[last:])
    return [d for d in decls if d.strip()]


_PARAM_NAME_RE = re.compile(r"\b([A-Za-z_]\w*)\s*$")


def _extract_param_identifier(decl: str) -> str | None:
    """Return the parameter identifier from a stripped decl, or None."""

    cleaned = _strip_attribute_markers(decl).strip()
    # Drop trailing array extents.
    cleaned = re.sub(r"\[[^\]]*\]\s*$", "", cleaned).strip()
    cleaned = cleaned.replace("*", " ").replace("&", " ").strip()
    m = _PARAM_NAME_RE.search(cleaned)
    return m.group(1) if m else None


# Metal builtins are pass-through grid/threadgroup descriptors; they are
# NOT user buffers and must not be renamed.
_METAL_BUILTIN_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "thread_position_in_grid",
        "thread_position_in_threadgroup",
        "threadgroup_position_in_grid",
        "thread_index_in_threadgroup",
        "thread_index_in_simdgroup",
        "simdgroup_index_in_threadgroup",
        "threads_per_threadgroup",
        "threadgroups_per_grid",
        "thread_execution_width",
        "grid_size",
        "gridDim",
        "blockDim",
        "blockIdx",
        "threadIdx",
    }
)


def _parse_buffer_param_names(sig_text: str) -> list[str]:
    """Return ``device``/``constant``-qualified buffer names, in order.

    Skips TileLang's auto-emitted scalar-args struct (``constant
    foo_kernel_args_t& arg [[ buffer(N) ]]``). That parameter is a
    *reference to a struct of scalars* (e.g. ``n_elements``,
    ``gridDim_0``) -- it is NOT a user data tensor and the caller
    cannot pass an ``mx.array`` for it. Detection is via two stable
    markers: the type ends in ``_args_t`` AND the parameter is passed
    by reference (``&``) rather than by pointer (``*``). User data
    buffers always come through as ``device <T>* <name>``.
    """

    names: list[str] = []
    for decl in _split_signature_decls(sig_text):
        clean = _strip_attribute_markers(decl).strip()
        if not clean:
            continue
        if re.search(r"\bthreadgroup\b", clean):
            continue
        is_device = re.search(r"\bdevice\b", clean) is not None
        is_constant = re.search(r"\bconstant\b", clean) is not None
        if not (is_device or is_constant):
            continue
        # Strip array extents to keep the by-ref detection clean.
        type_part = re.sub(r"\[[^\]]*\]\s*$", "", clean).strip()
        # TileLang's args struct: ``constant foo_kernel_args_t& arg``. We
        # detect by the pair (passed-by-reference, type-name ends in
        # ``_args_t``). Either heuristic alone is too aggressive.
        if "&" in type_part and re.search(r"_args_t\s*&", type_part):
            continue
        ident = _extract_param_identifier(clean)
        if ident is None:
            continue
        if ident in _METAL_BUILTIN_PARAM_NAMES:
            continue
        names.append(ident)
    return names


# ---------------------------------------------------------------------------
# Body renaming
# ---------------------------------------------------------------------------


def _rename_identifiers_in_code(
    code: str,
    rename: dict[str, str],
) -> str:
    """Rewrite whole-word identifiers in ``code`` per ``rename``.

    Comments and string literals are skipped so we don't accidentally
    touch a parameter name that happens to appear in a doc-comment.
    """

    if not rename:
        return code

    # Match the longest old-name first so a name like ``A`` doesn't shadow
    # ``A_scale`` in the regex alternation.
    keys = sorted(rename.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")

    def _rewrite_segment(seg: str) -> str:
        return pattern.sub(lambda m: rename[m.group(1)], seg)

    chunks: list[str] = []
    last = 0
    for match in _COMMENT_OR_STRING_RE.finditer(code):
        chunks.append(_rewrite_segment(code[last : match.start()]))
        chunks.append(match.group(0))
        last = match.end()
    chunks.append(_rewrite_segment(code[last:]))
    return "".join(chunks)


# ---------------------------------------------------------------------------
# CUDA-style identifier rewrite for MLX
# ---------------------------------------------------------------------------


# Map from TileLang's CUDA-style scalar builtin identifiers (declared in the
# emitted kernel signature as ``uint blockIdx [[threadgroup_position_in_grid]]``
# and ``uint threadIdx [[thread_position_in_threadgroup]]``) to the
# expressions ``mx.fast.metal_kernel`` injects into the body scope. MLX
# always provides the *vector* form (``uint3``) of the position builtins,
# so we substitute with the ``.x`` component to preserve the original
# scalar semantics. Callers that need the y/z components can extend this
# table; for the conformance harness only the .x slice is used.
_CUDA_BUILTIN_REWRITE: dict[str, str] = {
    "blockIdx": "threadgroup_position_in_grid.x",
    "threadIdx": "thread_position_in_threadgroup.x",
}


# Regex-based substitutions for dotted CUDA builtins (``blockIdx.y``,
# ``blockDim.x`` etc.) and zero-arg device intrinsics (``__syncthreads()``).
# These cannot go through ``_rename_identifiers_in_code`` because that
# helper only matches bare identifiers (``\b<name>\b``); the dotted forms
# require a multi-token match. We apply this pass BEFORE the bare-identifier
# rename so the bare ``blockIdx`` rewrite (``-> threadgroup_position_in_grid.x``)
# does not eagerly rewrite the prefix of ``blockIdx.y``.
#
# Coverage rationale (J1.5):
#   * ``threadIdx.{y,z}`` / ``blockIdx.{y,z}`` -- multi-dim grid kernels.
#   * ``blockDim.{x,y,z}`` -- threadgroup-size queries (TileLang sometimes
#     emits these for static shapes; preserve the semantics).
#   * ``gridDim.{x,y,z}`` -- grid-size queries.
#   * ``__syncthreads()`` -- CUDA's threadgroup barrier maps to Metal's
#     ``threadgroup_barrier(metal::mem_flags::mem_threadgroup)``.
#
# Warp-shuffle intrinsics (``__shfl_sync``, ``__shfl_xor_sync``,
# ``__shfl_up_sync``, ``__shfl_down_sync``, ``__ballot_sync``,
# ``__any_sync``, ``__all_sync``) have a Metal counterpart in the
# ``simd_shuffle*`` family but the argument shape is incompatible
# (CUDA's leading mask argument has no Metal equivalent and Metal's
# ``simd_*`` intrinsics work at SIMD-group granularity, which is 32 on
# Apple GPUs). The matmul kernels we currently lower do not emit them,
# so we leave them unrewritten and let the Metal compiler produce a
# clear "use of undeclared identifier" diagnostic if a future kernel
# needs them. The diagnostic is louder than a silent miscompile.
_CUDA_DOTTED_REWRITE: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bblockIdx\.x\b"), "threadgroup_position_in_grid.x"),
    (re.compile(r"\bblockIdx\.y\b"), "threadgroup_position_in_grid.y"),
    (re.compile(r"\bblockIdx\.z\b"), "threadgroup_position_in_grid.z"),
    (re.compile(r"\bthreadIdx\.x\b"), "thread_position_in_threadgroup.x"),
    (re.compile(r"\bthreadIdx\.y\b"), "thread_position_in_threadgroup.y"),
    (re.compile(r"\bthreadIdx\.z\b"), "thread_position_in_threadgroup.z"),
    (re.compile(r"\bblockDim\.x\b"), "threads_per_threadgroup.x"),
    (re.compile(r"\bblockDim\.y\b"), "threads_per_threadgroup.y"),
    (re.compile(r"\bblockDim\.z\b"), "threads_per_threadgroup.z"),
    (re.compile(r"\bgridDim\.x\b"), "threadgroups_per_grid.x"),
    (re.compile(r"\bgridDim\.y\b"), "threadgroups_per_grid.y"),
    (re.compile(r"\bgridDim\.z\b"), "threadgroups_per_grid.z"),
    (
        re.compile(r"\b__syncthreads\s*\(\s*\)"),
        "threadgroup_barrier(metal::mem_flags::mem_threadgroup)",
    ),
)


# Warp-shuffle intrinsics that we do NOT currently rewrite. If we see one
# of these we leave the source alone (so the Metal compiler errors out
# loudly) but the caller can introspect via ``re.search`` if they want
# to pre-flight the kernel.
_CUDA_WARP_SHUFFLE_RE = re.compile(
    r"\b__(?:shfl(?:_xor|_up|_down)?_sync|ballot_sync|any_sync|all_sync)\b"
)


_ARGS_STRUCT_DECL_RE = re.compile(
    r"\bconstant\s+\w+_args_t\s*&\s*arg\s*\[\[\s*buffer\s*\(\s*\d+\s*\)\s*\]\]"
    r"\s*,?\s*",
    re.MULTILINE,
)


def _rewrite_tilelang_metal_to_mlx(
    source: str,
    *,
    args_struct_inline: dict[str, Any] | None = None,
) -> str:
    """Rewrite TileLang's CUDA-style Metal source for ``mx.fast.metal_kernel``.

    TileLang's Metal emitter produces a kernel that

    1. declares ``uint blockIdx [[threadgroup_position_in_grid]]`` and
       ``uint threadIdx [[thread_position_in_threadgroup]]`` as scalar
       params, then references them as bare ``blockIdx`` / ``threadIdx``
       identifiers in the body, and
    2. accepts a ``constant <kernel>_args_t& arg [[buffer(N)]]`` struct
       holding scalar runtime args (e.g. ``arg.arg3[0]`` for
       ``n_elements``).

    ``mx.fast.metal_kernel`` rebuilds the kernel signature itself from
    ``input_names`` / ``output_names`` and only injects MLX's own builtin
    bindings (``thread_position_in_grid``, ``thread_position_in_threadgroup``
    -- both ``uint3``) into the body scope. The kernel-author-declared
    ``blockIdx`` / ``threadIdx`` / ``arg`` parameters do NOT survive that
    rebuild, so any reference to them is an undeclared-identifier error
    when MLX hands the source to Metal's compiler.

    This rewrite bridges the gap textually:

    * ``blockIdx``/``threadIdx`` identifiers (bare or inside casts like
      ``((int)blockIdx)``) become ``threadgroup_position_in_grid.x`` /
      ``thread_position_in_threadgroup.x``.
    * ``arg.<field>[0]`` accesses are inlined to the integer values from
      ``args_struct_inline`` (mapping field name -> int). When a field has
      no inline value, we leave the access alone and let the caller see
      the resulting compile error -- silently dropping the access would
      hide a real configuration bug.

    Whole-token substitution is used (``\\b`` boundaries) so identifiers
    like ``arg0``/``arg1`` (the renamed user buffers) are never confused
    with the scalar-args struct ``arg``.
    """

    args_struct_inline = args_struct_inline or {}

    # 1) Inline the args-struct field accesses BEFORE we drop the struct
    # parameter declaration, so we don't lose the field-name information.
    # Pattern: ``arg.<field>[<index>]`` -- TileLang emits each int field as
    # a ``int <field>[2]`` array, with the value at index 0.
    def _inline_field(match: "re.Match[str]") -> str:
        field = match.group("field")
        idx = match.group("idx")
        if field in args_struct_inline and idx == "0":
            return str(int(args_struct_inline[field]))
        return match.group(0)

    source = re.sub(
        r"\barg\.(?P<field>[A-Za-z_]\w*)\s*\[\s*(?P<idx>\d+)\s*\]",
        _inline_field,
        source,
    )

    # 2a) Rewrite dotted CUDA builtins (``blockIdx.y``, ``blockDim.x``,
    # ``gridDim.z``, ...) and zero-arg device intrinsics (``__syncthreads()``)
    # BEFORE the bare-identifier pass below so the bare ``blockIdx`` rule
    # (which appends ``.x``) does not eagerly rewrite the prefix of
    # ``blockIdx.y`` into ``threadgroup_position_in_grid.x.y``.
    #
    # We do NOT mask comments/strings here because the dotted patterns are
    # specific enough that a stray match inside a doc-comment is a non-issue;
    # if that ever changes, route through ``_rename_identifiers_in_code``'s
    # masking helper.
    for pattern, replacement in _CUDA_DOTTED_REWRITE:
        source = pattern.sub(replacement, source)

    # 2b) Rewrite bare CUDA-style builtin identifiers via whole-token
    # substitution. After step 2a, only un-suffixed references like
    # ``blockIdx`` (no ``.x``/``.y``) remain to be rewritten.
    source = _rename_identifiers_in_code(source, _CUDA_BUILTIN_REWRITE)

    # 3) Drop the ``constant <kernel>_args_t& arg [[buffer(N)]]`` parameter
    # from the kernel signature. ``mx.fast.metal_kernel`` synthesizes its
    # own signature, so leaving this declaration in produces a duplicate-
    # parameter error when we splice the body. We do this last so the
    # field-inlining step above sees the original identifiers.
    source = _ARGS_STRUCT_DECL_RE.sub("", source)

    return source


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileLangMetalAdapter:
    """Adapter container returned by :func:`wrap_tilelang_metal_kernel`.

    ``__call__`` dispatches a single-output or multi-output Metal kernel
    on Mac GPU through ``mx.fast.metal_kernel``.
    """

    kernel_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    header: str
    body: str
    buffer_names: tuple[str, ...]
    # The ``mx.fast.metal_kernel(...)`` callable; built lazily so this
    # adapter can be inspected on hosts where MLX is not importable.
    _kernel_factory: Callable[[], Any]

    def build(self) -> Any:
        """Return the underlying ``mx.fast.metal_kernel`` callable."""

        return self._kernel_factory()

    def __call__(
        self,
        inputs: Sequence[Any],
        *,
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[Any],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int] = (1, 1, 1),
    ) -> list[Any]:
        kernel = self.build()
        if len(inputs) != len(self.input_names):
            raise MLXRuntimeError(
                f"input count mismatch: got {len(inputs)}, expected "
                f"{len(self.input_names)} ({list(self.input_names)})"
            )
        if len(output_shapes) != len(self.output_names):
            raise MLXRuntimeError(
                f"output count mismatch: got {len(output_shapes)}, expected "
                f"{len(self.output_names)} ({list(self.output_names)})"
            )
        return kernel(
            inputs=list(inputs),
            output_shapes=[tuple(s) for s in output_shapes],
            output_dtypes=list(output_dtypes),
            grid=tuple(int(g) for g in grid),
            threadgroup=tuple(int(t) for t in threadgroup),
        )


def _extract_kernel_source(artifact: Any) -> str:
    """Pull MSL source out of a TileLang compile artifact, or raise."""

    src = getattr(artifact, "kernel_source", None)
    if isinstance(src, str) and src.strip():
        return src
    rt_mod = getattr(artifact, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "get_source"):
        try:
            text = rt_mod.get_source()
        except Exception as exc:  # noqa: BLE001 -- broad, surfaced
            raise MLXRuntimeError(
                f"artifact.rt_mod.get_source() raised: {type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(text, str) and text.strip():
            return text
    # Fallback: ``artifact`` may itself already be the source (callers
    # that already ran the codegen and held onto the string).
    if isinstance(artifact, str) and artifact.strip():
        return artifact
    raise MLXRuntimeError(
        "TileLang artifact has neither .kernel_source nor .rt_mod.get_source()"
    )


def wrap_tilelang_metal_kernel(
    artifact: Any,
    *,
    input_count: int,
    output_count: int,
    name: str | None = None,
    args_struct_inline: dict[str, Any] | None = None,
    input_buffer_names: Sequence[str] | None = None,
    output_buffer_names: Sequence[str] | None = None,
    allow_mx_fast_metal_kernel: bool = False,
) -> TileLangMetalAdapter:
    """Adapt a TileLang Metal artifact for ``mx.fast.metal_kernel``.

    ``artifact`` may be a TileLang ``CompiledArtifact``, a raw MSL
    string, or anything exposing ``.kernel_source`` / ``.rt_mod.get_source()``.

    By default, the first ``input_count`` device buffers in the kernel
    signature are renamed to ``inp0..inp{input_count-1}``; the next
    ``output_count`` to ``out0..out{output_count-1}``.

    Callers that already know the PrimFunc ABI may pass
    ``input_buffer_names`` / ``output_buffer_names``. That mode is required
    for TileLang kernels whose Metal signature interleaves outputs with inputs
    or omits unused input tensors after lowering. ``input_count`` remains the
    runtime input ABI count; the adapter exposes only the subset of those
    named inputs that are present in the emitted Metal signature.

    This raw MLX fast-kernel bridge is fail-closed by default because the
    production Path C boundary is tvm-ffi/owner-output. Tests and explicit
    proof harnesses must pass ``allow_mx_fast_metal_kernel=True`` so they
    cannot accidentally become a silent production fallback.

    Returns a :class:`TileLangMetalAdapter` whose ``__call__`` dispatches
    on Mac GPU. The ``mx.fast.metal_kernel`` instance is built lazily on
    first ``__call__`` so this function can be invoked on hosts without
    MLX (it will only fail when actually launched).
    """

    if not allow_mx_fast_metal_kernel:
        raise MLXRuntimeError(
            "wrap_tilelang_metal_kernel is fail-closed for production: "
            "the supported production Path C boundary is tvm-ffi/owner-output, "
            "not a raw mx.fast.metal_kernel wrapper. Pass "
            "allow_mx_fast_metal_kernel=True only from tests, POC harnesses, "
            "or explicit migration tooling."
        )

    if input_count < 0 or output_count < 0:
        raise MLXRuntimeError(
            f"input_count/output_count must be non-negative, got "
            f"input_count={input_count}, output_count={output_count}"
        )
    if input_buffer_names is not None:
        input_buffer_names = tuple(str(name) for name in input_buffer_names)
        if len(input_buffer_names) != input_count:
            raise MLXRuntimeError(
                f"input_buffer_names has {len(input_buffer_names)} entries, "
                f"but input_count={input_count}"
            )
    if output_buffer_names is not None:
        output_buffer_names = tuple(str(name) for name in output_buffer_names)
        if len(output_buffer_names) != output_count:
            raise MLXRuntimeError(
                f"output_buffer_names has {len(output_buffer_names)} entries, "
                f"but output_count={output_count}"
            )
    if (input_buffer_names is None) != (output_buffer_names is None):
        raise MLXRuntimeError(
            "input_buffer_names and output_buffer_names must be provided together"
        )

    declared_total = input_count + output_count
    if declared_total == 0:
        raise MLXRuntimeError("kernel must have at least one buffer parameter")

    src = _extract_kernel_source(artifact)
    # Rewrite CUDA-style identifiers and inline scalar-args struct accesses
    # BEFORE we split the kernel: the rewrite drops the ``_args_t& arg``
    # parameter from the signature and substitutes ``arg.<field>[0]`` in
    # the body, which both must happen before ``_parse_buffer_param_names``
    # runs (so the args struct is gone from the signature) and before the
    # buffer-rename step (so we don't accidentally rename ``blockIdx``
    # away while it still looks like a CUDA identifier).
    src = _rewrite_tilelang_metal_to_mlx(
        src, args_struct_inline=args_struct_inline
    )
    prelude, kernel_name, signature, body = _split_kernel(src)

    buffer_names = _parse_buffer_param_names(signature)
    if input_buffer_names is None or output_buffer_names is None:
        if len(buffer_names) != declared_total:
            raise MLXRuntimeError(
                f"buffer count mismatch: parsed {len(buffer_names)} device/constant "
                f"buffers from kernel signature ({buffer_names!r}), but caller "
                f"declared input_count={input_count} + output_count={output_count} "
                f"= {declared_total}"
            )
        input_sources = tuple(buffer_names[:input_count])
        output_sources = tuple(buffer_names[input_count:])
    else:
        parsed = set(buffer_names)
        input_sources = tuple(name for name in input_buffer_names if name in parsed)
        output_sources = tuple(output_buffer_names)
        missing_outputs = tuple(name for name in output_sources if name not in parsed)
        if missing_outputs:
            raise MLXRuntimeError(
                "output_buffer_names contains buffers missing from the Metal "
                f"signature: {missing_outputs!r}; parsed={buffer_names!r}"
            )
        aliased_names = set(input_sources) & set(output_sources)
        if aliased_names:
            raise MLXRuntimeError(
                f"Input and output buffer names must be mutually disjoint (aliasing is not supported): "
                f"aliased={sorted(list(aliased_names))}"
            )
        explicit_sources = input_sources + output_sources
        if len(set(explicit_sources)) != len(explicit_sources):
            raise MLXRuntimeError(
                f"explicit buffer mapping contains duplicate source names: "
                f"{explicit_sources!r}"
            )
        unmapped = tuple(name for name in buffer_names if name not in explicit_sources)
        if unmapped:
            raise MLXRuntimeError(
                f"explicit buffer mapping does not cover emitted buffers "
                f"{unmapped!r}; parsed={buffer_names!r}"
            )
        if not explicit_sources:
            raise MLXRuntimeError(
                "explicit buffer mapping did not match any emitted Metal buffers"
            )

    if len(set(buffer_names)) != len(buffer_names):
        raise MLXRuntimeError(
            f"unsupported TileLang Metal pattern: duplicate buffer names "
            f"{buffer_names!r}"
        )

    # Ensure no input/output aliasing overlap
    aliased_names = set(input_sources) & set(output_sources)
    if aliased_names:
        raise MLXRuntimeError(
            f"Input and output buffer names must be mutually disjoint (aliasing is not supported): "
            f"aliased={sorted(list(aliased_names))}"
        )

    input_names = tuple(f"inp{i}" for i in range(len(input_sources)))
    output_names = tuple(f"out{i}" for i in range(len(output_sources)))
    rename: dict[str, str] = {}
    for src_name, mlx_name in zip(input_sources, input_names):
        rename[src_name] = mlx_name
    for src_name, mlx_name in zip(output_sources, output_names):
        rename[src_name] = mlx_name

    renamed_body = _rename_identifiers_in_code(body, rename)

    # The header for ``mx.fast.metal_kernel`` is the prelude (typedefs,
    # helper macros, constants) emitted before the kernel definition --
    # NOT the kernel signature, which MLX builds itself from input_names
    # and output_names.
    header = prelude

    final_name = name or kernel_name

    def _build_kernel() -> Any:
        try:
            import mlx.core as mx  # type: ignore
        except Exception as exc:  # noqa: BLE001 -- surfaced verbatim
            raise MLXRuntimeError(
                f"mlx.core import failed: {type(exc).__name__}: {exc}"
            ) from exc
        fast = getattr(mx, "fast", None)
        ctor = getattr(fast, "metal_kernel", None) if fast is not None else None
        if ctor is None:
            raise MLXRuntimeError(
                "mx.fast.metal_kernel constructor unavailable on this MLX build"
            )
        return ctor(
            name=final_name,
            input_names=list(input_names),
            output_names=list(output_names),
            source=renamed_body,
            header=header,
            ensure_row_contiguous=True,
        )

    return TileLangMetalAdapter(
        kernel_name=final_name,
        input_names=input_names,
        output_names=output_names,
        header=header,
        body=renamed_body,
        buffer_names=tuple(buffer_names),
        _kernel_factory=_build_kernel,
    )
