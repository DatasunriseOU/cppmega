"""FP8 activation compression for gradient checkpointing.

Quantizes saved activations to FP8 (e4m3fn) during gradient checkpoint save,
decompresses to bf16 during recompute. Saves ~2x activation memory with
minimal quality impact.

Two modes:
1. **Transformer Engine** (default when installed): TE current-scaling FP8
   quantize/dequantize hooks for torch.utils.checkpoint.
2. **COAT** (optional): NVlabs COAT library (ICLR 2025) for mixed-granularity
   quantization + FP8 optimizer states. Requires `pip install fp8-coat`.

Pack/unpack uses Transformer Engine by default. The older Triton/PyTorch path
is fail-closed and requires an explicit deprecated-path acknowledgement.

Usage:
    from nanochat.fp8_activations import enable_fp8_activation_checkpointing

    # Native mode (no extra deps)
    enable_fp8_activation_checkpointing(model)

    # COAT mode (if installed)
    enable_fp8_activation_checkpointing(model, backend="coat")

Hardware: SM90+ (H100/H200/B200). Falls back to bf16 on older GPUs.

References:
    - COAT: https://github.com/NVlabs/COAT (ICLR 2025)
    - PyTorch checkpoint: torch.utils.checkpoint.checkpoint
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, TypeAlias, cast

import torch

from cppmega.megatron.deprecated_paths import require_deprecated_ack

logger = logging.getLogger(__name__)

try:
    import transformer_engine  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor import Float8CurrentScalingQuantizer

    _TE_AVAILABLE = True
except ImportError:
    tex = None  # type: ignore[assignment]
    Float8CurrentScalingQuantizer = None  # type: ignore[assignment]
    _TE_AVAILABLE = False

# FP8 dtype for activations (e4m3 = 4 exponent, 3 mantissa, no NaN encoding)
_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

# Minimum tensor size (in elements) to bother with FP8 quantization.
# Tensors smaller than this get more overhead from quantize/dequantize than
# they save in memory. Default 16384 elements = 32KB in BF16.
# Configurable via NANOCHAT_FP8_MIN_ELEMENTS env var.
_FP8_MIN_ELEMENTS = int(os.environ.get("NANOCHAT_FP8_MIN_ELEMENTS", "16384"))

# Check hardware support
_FP8_AVAILABLE = False
_COAT_AVAILABLE = False
_TRITON_AVAILABLE = False
triton: Any | None = None
tl: Any | None = None

try:
    if torch.cuda.is_available():
        _major, _ = torch.cuda.get_device_capability()
        _FP8_AVAILABLE = _major >= 9 and _FP8_DTYPE is not None
except Exception:
    pass

try:
    import coat  # noqa: F401  # pyright: ignore[reportMissingImports]
    _COAT_AVAILABLE = True
except ImportError:
    pass

try:
    import triton
    import triton.language as tl  # pyright: ignore[reportMissingModuleSource]
    _TRITON_AVAILABLE = True
except ImportError:
    pass


PackedActivation: TypeAlias = (
    torch.Tensor
    | tuple[Any, Any, torch.dtype]
    | int
    | None
)


def _require_triton() -> Any:
    if triton is None:
        raise RuntimeError("Triton is not available")
    return triton


def _require_tl() -> Any:
    if tl is None:
        raise RuntimeError("Triton language is not available")
    return tl


if TYPE_CHECKING:
    _TritonConstexpr: TypeAlias = int
else:
    _TritonConstexpr = Any

# FP8 e4m3fn max value, computed once at import time
_FP8_MAX: float = torch.finfo(_FP8_DTYPE).max if _FP8_DTYPE is not None else 448.0
_TE_PACK_SENTINEL = "te_fp8"
_TE_QUANTIZER_CACHE: dict[tuple[int, int], Any] = {}


def _torch_dtype_to_te(dtype: torch.dtype):
    if tex is None:
        raise RuntimeError("Transformer Engine is not available")
    if dtype == torch.bfloat16:
        return tex.DType.kBFloat16
    if dtype == torch.float16:
        return tex.DType.kFloat16
    if dtype == torch.float32:
        return tex.DType.kFloat32
    raise TypeError(f"unsupported FP8 activation unpack dtype: {dtype}")


def _te_fp8_dtype():
    if tex is None:
        raise RuntimeError("Transformer Engine is not available")
    return tex.DType.kFloat8E4M3


# ═══════════════════════════════════════════════════════════════════════════
# Thread-local flag for selectively disabling FP8 quantization
# ═══════════════════════════════════════════════════════════════════════════

_thread_local = threading.local()


def _is_fp8_disabled() -> bool:
    """Check if FP8 quantization is disabled for the current thread."""
    return getattr(_thread_local, "fp8_hooks_disabled", False)


class fp8_quantize_disabled:
    """Context manager to disable FP8 activation quantization.

    Tensors saved for backward within this context will NOT be quantized
    to FP8, preserving their original dtype.  Use this to protect
    numerically-sensitive modules (Mamba state, MoE router logits,
    RMSNorm outputs, M2RNN state, embeddings) from FP8 quantization loss.

    Thread-safe: uses a per-thread flag.  Nested calls are safe -- the flag
    is restored to its previous value on exit.

    Uses a class-based __enter__/__exit__ instead of @contextmanager so that
    torch.compile can trace through it without a graph break (generator-based
    context managers create GenericContextWrappingVariable graph breaks).

    Example::

        from nanochat.fp8_activations import fp8_quantize_disabled

        class MyModule(nn.Module):
            def forward(self, x):
                with fp8_quantize_disabled():
                    # Tensors saved here keep original dtype
                    return self.sensitive_op(x)
    """

    __slots__ = ("_prev",)

    def __enter__(self):
        self._prev = getattr(_thread_local, "fp8_hooks_disabled", False)
        _thread_local.fp8_hooks_disabled = True
        return self

    def __exit__(self, *exc):
        _thread_local.fp8_hooks_disabled = self._prev
        return False


def no_fp8_quantize(fn):
    """Decorator that wraps a function/method with fp8_quantize_disabled().

    Use on forward() methods of modules whose saved tensors must not be
    quantized to FP8 (Mamba state, M2RNN state, MoE router, RMSNorm, etc.).

    Uses direct thread-local flag manipulation instead of a context manager
    so that torch.compile can trace through it without graph breaks.

    Example::

        class Mamba2Layer(nn.Module):
            @no_fp8_quantize
            def forward(self, x, ...):
                ...
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        prev = getattr(_thread_local, "fp8_hooks_disabled", False)
        _thread_local.fp8_hooks_disabled = True
        try:
            return fn(*args, **kwargs)
        finally:
            _thread_local.fp8_hooks_disabled = prev

    return wrapper


# ═══════════════════════════════════════════════════════════════════════════
# Layer-aware FP8 skip: first/last layers stay in BF16 for stability
# ═══════════════════════════════════════════════════════════════════════════

# Thread-local state: tracks which layer is currently executing its forward
# pass so the FP8 pack hook can decide whether to quantize or pass through.
_layer_state = threading.local()


def _get_current_layer_idx() -> int:
    """Return the layer index currently executing, or -1 if unknown."""
    return getattr(_layer_state, "current_layer_idx", -1)


def set_current_layer_idx(idx: int) -> None:
    """Set the layer index for the currently executing forward pass.

    Called by GPT.forward() before each block to let the FP8 activation
    packer know which layer is active.  Set to -1 after the loop to reset.
    """
    _layer_state.current_layer_idx = idx


def is_bf16_layer(layer_idx: int, n_layers: int,
                  bf16_first: int, bf16_last: int) -> bool:
    """Check if *layer_idx* should stay in BF16 (skip FP8 quantization).

    Follows Megatron-Core's ``is_first_last_bf16_layer`` logic:
    the first ``bf16_first`` layers and the last ``bf16_last`` layers
    are kept in BF16 even when FP8 activations are enabled.

    Args:
        layer_idx: Global layer index (0-based).
        n_layers: Total number of transformer layers.
        bf16_first: Number of initial layers to keep in BF16.
        bf16_last: Number of final layers to keep in BF16.

    Returns:
        True if the layer should stay in BF16, False if FP8 is OK.
    """
    if layer_idx < 0:
        return False  # unknown layer — allow FP8
    if bf16_first > 0 and layer_idx < bf16_first:
        return True
    if bf16_last > 0 and layer_idx >= n_layers - bf16_last:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Fused Triton kernels for FP8 pack/unpack
# ═══════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:
    _triton = _require_triton()
    _tl = _require_tl()

    # IMPORTANT: Triton kernel bodies must use the module-level `tl` binding,
    # NOT the local `_tl` closure.  When inductor's codecache re-emits the
    # kernel source to its own file, only module-level symbols are imported
    # into the new scope — a local assignment like `_tl = _require_tl()` is
    # lost, raising `NameError: _tl is not defined`.  See commit history
    # (fp8 tl.constexpr fix) + NAM56R bare bench failure mode.

    @cast(Callable[..., Any], _triton.jit)
    def _amax_kernel(
        input_ptr,
        amax_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Block-parallel abs-max reduction.

        Each block computes a local amax over BLOCK_SIZE elements and
        atomically updates a single global amax scalar. This fuses
        abs() + amax() into one pass over the data (saves one full
        elementwise kernel launch compared to separate abs() then amax()).
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        local_amax = tl.max(tl.abs(vals), axis=0)
        tl.atomic_max(amax_ptr, local_amax)

    @cast(Callable[..., Any], _triton.jit)
    def _quantize_kernel(
        input_ptr,
        output_ptr,
        inv_scale,
        n_elements,
        fp8_max,
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Scale + clamp + FP8 cast in one pass.

        Reads input, multiplies by inv_scale (= fp8_max / amax), clamps to
        [-fp8_max, fp8_max], casts to float8_e4m3fn, and stores. Replaces
        the unfused sequence: tensor / scale -> clamp -> to(fp8).
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        scaled = vals * inv_scale
        clamped = tl.clamp(scaled, -fp8_max, fp8_max)
        fp8_vals = clamped.to(tl.float8e4nv)
        tl.store(output_ptr + offsets, fp8_vals, mask=mask)

    @cast(Callable[..., Any], _triton.jit)
    def _fp8_unpack_kernel(
        input_ptr,
        output_ptr,
        scale_val,
        n_elements,
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        OUT_DTYPE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Fused FP8-to-dtype cast + scale multiply kernel for unpacking.

        Single pass: load FP8 -> cast to output dtype -> multiply by scale -> store.
        Replaces two separate kernel launches (to(dtype) + mul(scale)).
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        fp8_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        # Cast from FP8 to output dtype (fp32 intermediate for precision)
        float_vals = fp8_vals.to(OUT_DTYPE)
        result = float_vals * scale_val

        tl.store(output_ptr + offsets, result, mask=mask)


def _next_power_of_2(n: int) -> int:
    """Round up to the next power of 2, minimum 1."""
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _pick_block_size(n_elements: int) -> int:
    """Choose BLOCK_SIZE for FP8 pack/unpack kernels.

    Targets ~256 blocks for large tensors (good SM occupancy), clamped to
    [1024, 65536] and rounded up to a power of 2.
    """
    return min(_next_power_of_2(max(n_elements // 256, 1024)), 65536)


def _triton_fp8_pack(tensor: torch.Tensor, *, clamp: bool = False):
    """Fused FP8 pack using Triton (2 kernel launches instead of 3+).

    Launch 1: _amax_kernel -- fused abs + block-reduce + atomic-max
    Launch 2: _quantize_kernel -- fused scale + clamp + FP8 cast

    The unfused path does: abs() | amax() | where() | div() | to(fp8)
    which is 3-5 separate kernel launches depending on PyTorch fusion.

    Returns (fp8_tensor, scale, orig_dtype) matching the unfused API.
    """
    if clamp:
        tensor = tensor.clamp(-_FP8_MAX, _FP8_MAX)
    n_elements = tensor.numel()
    flat = tensor.reshape(-1)

    BLOCK_SIZE = _pick_block_size(n_elements)
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch 1: fused abs + amax reduction
    amax_buf = torch.zeros(1, dtype=torch.float32, device=tensor.device)
    _amax_kernel[(n_blocks,)](
        flat, amax_buf, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Compute scale on host (requires sync to read amax, but unavoidable --
    # inv_scale is a kernel argument for launch 2)
    amax_val = amax_buf.item()
    if amax_val > 0:
        scale_val = amax_val / _FP8_MAX
        inv_scale_val = _FP8_MAX / amax_val
    else:
        scale_val = 1.0
        inv_scale_val = 1.0

    # Launch 2: fused scale + clamp + FP8 cast
    fp8_out = torch.empty(n_elements, dtype=_FP8_DTYPE, device=tensor.device)
    _quantize_kernel[(n_blocks,)](
        flat, fp8_out, inv_scale_val, n_elements,
        fp8_max=_FP8_MAX,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    fp8_out = fp8_out.reshape(tensor.shape)
    scale = torch.tensor(scale_val, dtype=torch.float32, device=tensor.device)
    return (fp8_out, scale, tensor.dtype)


# Map torch dtypes to triton constexpr dtype tokens for the unpack kernel
_DTYPE_TO_TL = {}
if _TRITON_AVAILABLE:
    _DTYPE_TO_TL = {
        torch.bfloat16: _require_tl().bfloat16,
        torch.float16: _require_tl().float16,
        torch.float32: _require_tl().float32,
    }


def _triton_fp8_unpack(fp8_tensor: torch.Tensor, scale: torch.Tensor, orig_dtype: torch.dtype):
    """Fused FP8 unpack using Triton (1 kernel launch instead of 2).

    Single kernel: load FP8 -> cast to orig_dtype -> multiply by scale -> store.
    Replaces to(dtype) + mul(scale) = 2 separate launches.
    """
    n_elements = fp8_tensor.numel()
    flat = fp8_tensor.reshape(-1)

    tl_dtype = _DTYPE_TO_TL.get(orig_dtype)
    if tl_dtype is None:
        # Fallback for unsupported dtypes
        return fp8_tensor.to(orig_dtype) * scale

    out = torch.empty(n_elements, dtype=orig_dtype, device=fp8_tensor.device)

    BLOCK_SIZE = _pick_block_size(n_elements)
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    scale_val = scale.item()

    _fp8_unpack_kernel[(n_blocks,)](
        flat, out, scale_val, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        OUT_DTYPE=tl_dtype,
    )

    return out.reshape(fp8_tensor.shape)


# ═══════════════════════════════════════════════════════════════════════════
# Unfused (reference) pack/unpack — used as fallback on non-CUDA devices
# ═══════════════════════════════════════════════════════════════════════════

def _unfused_fp8_pack(tensor: torch.Tensor, *, clamp: bool = False):
    """Reference (unfused) FP8 pack: 3 separate kernel launches.

    When *clamp* is True, outliers are hard-clipped to ``[-fp8_max, fp8_max]``
    before computing the per-tensor scale.  This prevents a single outlier
    from inflating the scale and destroying precision for the majority of
    values.  Replaces the old per-forward ``--fp8_activation_clamp`` path.
    """
    if clamp:
        tensor = tensor.clamp(-_FP8_MAX, _FP8_MAX)
    amax = tensor.abs().amax()
    scale = amax / _FP8_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    fp8_tensor = (tensor / scale).to(_FP8_DTYPE)
    return (fp8_tensor, scale, tensor.dtype)


def _unfused_fp8_unpack(fp8_tensor: torch.Tensor, scale: torch.Tensor, orig_dtype: torch.dtype):
    """Reference (unfused) FP8 unpack: 2 separate kernel launches."""
    return fp8_tensor.to(orig_dtype) * scale


def _te_quantizer_for(device: torch.device):
    if not _TE_AVAILABLE or tex is None or Float8CurrentScalingQuantizer is None:
        raise RuntimeError("Transformer Engine FP8 quantizer is not available")
    if device.type != "cuda":
        raise RuntimeError("Transformer Engine FP8 activation pack requires CUDA tensors")
    index = torch.cuda.current_device() if device.index is None else device.index
    key = (index, int(_te_fp8_dtype()))
    quantizer = _TE_QUANTIZER_CACHE.get(key)
    if quantizer is None:
        quantizer = Float8CurrentScalingQuantizer(
            _te_fp8_dtype(),
            device=torch.device("cuda", index),
            rowwise=True,
            columnwise=False,
        )
        _TE_QUANTIZER_CACHE[key] = quantizer
    return quantizer


def _te_fp8_pack(tensor: torch.Tensor, *, clamp: bool = False):
    """Pack with Transformer Engine current-scaling FP8, without CPU sync."""
    if clamp:
        tensor = tensor.clamp(-_FP8_MAX, _FP8_MAX)
    fp8_tensor = tex.quantize(tensor, _te_quantizer_for(tensor.device))  # type: ignore[union-attr]
    return (_TE_PACK_SENTINEL, fp8_tensor, tensor.dtype)


def _te_fp8_unpack(fp8_tensor: Any, orig_dtype: torch.dtype) -> torch.Tensor:
    """Unpack a TE Float8Tensor to the original tensor dtype."""
    return tex.dequantize(fp8_tensor, _torch_dtype_to_te(orig_dtype))  # type: ignore[union-attr]


def is_fp8_activation_available() -> bool:
    """Check if FP8 activation compression is available."""
    return _FP8_AVAILABLE


def is_coat_available() -> bool:
    """Check if COAT library is installed."""
    return _COAT_AVAILABLE


# ═══════════════════════════════════════════════════════════════════════════
# FP8 activation checkpoint hooks
# ═══════════════════════════════════════════════════════════════════════════

def _allow_legacy_activation_backend(feature: str, reason: str) -> None:
    require_deprecated_ack(
        feature=feature,
        ack_env=(
            "CPPMEGA_I_UNDERSTAND_FP8_ACTIVATION_LEGACY_BACKEND_"
            "IS_DEPRECATED_AND_SYNCY"
        ),
        replacement="CPPMEGA_FP8_ACTIVATION_BACKEND=te",
        reason=reason,
    )


def _use_te_packer() -> bool:
    """Return True if Transformer Engine should pack saved activations."""
    backend = os.environ.get("CPPMEGA_FP8_ACTIVATION_BACKEND", "te").lower()
    if backend in {"legacy", "triton", "torch"}:
        _allow_legacy_activation_backend(
            f"CPPMEGA_FP8_ACTIVATION_BACKEND={backend}",
            "The legacy activation packer can synchronize the CUDA stream and "
            "does not use TE Float8Tensor quantize/dequantize.",
        )
        return False
    if backend not in {"te", "transformer_engine", "auto"}:
        logger.warning(
            "Unknown CPPMEGA_FP8_ACTIVATION_BACKEND=%s; using Transformer Engine if available.",
            backend,
        )
    if not _TE_AVAILABLE and torch.cuda.is_available():
        _allow_legacy_activation_backend(
            "FP8 activation checkpointing without Transformer Engine",
            "Silent fallback would route to the old Triton/PyTorch packer.",
        )
        return False
    return _TE_AVAILABLE and torch.cuda.is_available()


def _use_triton_fused() -> bool:
    """Return True if legacy fused Triton FP8 kernels should be used.

    Can be disabled via NANOCHAT_FP8_NO_TRITON=1. Needed on torch 2.12
    nightly + triton 3.7 + sm90 where inductor async precompile of the
    quantize kernel hits a Triton LLIR PassManager::run failure during
    `.to(tl.float8e4nv)` lowering.  The unfused PyTorch path (abs().max()
    then div().clamp().to(fp8)) gets fused by inductor transparently.
    """
    if os.environ.get("NANOCHAT_FP8_NO_TRITON", "0") == "1":
        return False
    if os.environ.get("CPPMEGA_FP8_NO_TRITON", "0") == "1":
        return False
    return _TRITON_AVAILABLE and torch.cuda.is_available()


class FP8ActivationPacker:
    """Pack/unpack hooks for torch.utils.checkpoint that quantize activations to FP8.

    During forward (save): bf16 activation -> fp8_e4m3fn + scale (per-tensor)
    During recompute (load): fp8 x scale -> bf16

    Memory savings: ~2x on activation tensors (bf16=2B -> fp8=1B per element).

    The default CUDA backend uses Transformer Engine current-scaling quantize
    and dequantize, avoiding the host-side scale readback in the legacy path.
    The older Triton/PyTorch implementation is deprecated and requires an
    explicit ``CPPMEGA_I_UNDERSTAND_*`` acknowledgement.
    """

    @staticmethod
    def pack(tensor: torch.Tensor | int | None) -> PackedActivation:
        """Quantize activation tensor to FP8 for checkpoint storage."""
        if not isinstance(tensor, torch.Tensor):
            return tensor  # non-tensor (e.g., int, None) -- pass through
        if _is_fp8_disabled():
            return tensor  # module opted out via fp8_quantize_disabled()
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return tensor  # already quantized or integer -- pass through
        if tensor.numel() < _FP8_MIN_ELEMENTS:
            return tensor  # small tensor -- quantize/dequantize overhead exceeds savings

        # Make contiguous before quantizing. Non-contiguous tensors (from
        # transpose/view in attention) would otherwise have strides that differ
        # from the contiguous result of unpack's to()+mul, causing
        # assert_size_stride failures in torch.compile's backward graph.
        tensor = tensor.contiguous()

        if _use_te_packer() and tensor.is_cuda:
            return _te_fp8_pack(tensor)
        if _use_triton_fused() and tensor.is_cuda:
            return _triton_fp8_pack(tensor)
        return _unfused_fp8_pack(tensor)

    @staticmethod
    def unpack(packed) -> torch.Tensor:
        """Dequantize FP8 activation back to original dtype."""
        if not isinstance(packed, tuple) or len(packed) != 3:
            return cast(torch.Tensor, packed)
        fp8_tensor, scale, orig_dtype = packed
        if isinstance(fp8_tensor, str) and fp8_tensor == _TE_PACK_SENTINEL:
            return _te_fp8_unpack(scale, orig_dtype)
        if fp8_tensor.numel() < _FP8_MIN_ELEMENTS:
            logger.debug(
                "FP8 unpack: tensor with %d elements (< %d threshold) was quantized",
                fp8_tensor.numel(), _FP8_MIN_ELEMENTS,
            )

        if _use_triton_fused() and fp8_tensor.is_cuda:
            return _triton_fp8_unpack(fp8_tensor, scale, orig_dtype)
        return _unfused_fp8_unpack(fp8_tensor, scale, orig_dtype)


class ClampingFP8Packer:
    """FP8 packer that hard-clips outliers to ``[-448, 448]`` before quantizing.

    This replaces the old per-forward ``--fp8_activation_clamp`` path that added
    two extra ``.clamp()`` kernel launches per MLP layer on every forward pass.
    The default TE backend applies the clamp immediately before TE quantization;
    the legacy Triton backend can still fold it into its quantization kernel.

    The clamp prevents a single outlier from inflating the per-tensor scale
    factor.  Without it, one large value can cause the scale to be very large,
    destroying precision for the majority of in-range values after FP8 cast.
    """

    def pack(self, tensor: torch.Tensor | int | None) -> PackedActivation:
        """Quantize with outlier clamping."""
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if _is_fp8_disabled():
            return tensor
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return tensor
        if tensor.numel() < _FP8_MIN_ELEMENTS:
            return tensor

        tensor = tensor.contiguous()

        if _use_te_packer() and tensor.is_cuda:
            return _te_fp8_pack(tensor, clamp=True)
        if _use_triton_fused() and tensor.is_cuda:
            return _triton_fp8_pack(tensor, clamp=True)
        return _unfused_fp8_pack(tensor, clamp=True)

    @staticmethod
    def unpack(packed) -> torch.Tensor:
        """Dequantize -- identical to FP8ActivationPacker.unpack."""
        return FP8ActivationPacker.unpack(packed)


def enable_fp8_activation_checkpointing(
    model: torch.nn.Module,
    backend: str = "native",
) -> dict:
    """Enable FP8 activation compression for gradient checkpointing.

    Args:
        model: The model (must have gradient_checkpointing enabled on blocks)
        backend: "native" (PyTorch hooks) or "coat" (NVlabs COAT)

    Returns:
        dict with status info
    """
    if not _FP8_AVAILABLE:
        logger.warning(
            "FP8 activations not available (need SM90+ GPU with torch.float8_e4m3fn). "
            "Falling back to bf16 activations."
        )
        return {"enabled": False, "reason": "no_fp8_hardware"}

    if backend == "coat":
        if not _COAT_AVAILABLE:
            logger.warning("COAT not installed (pip install fp8-coat). Using native FP8.")
            backend = "native"
        else:
            return _enable_coat_fp8(model)

    # Native: register pack/unpack hooks on checkpoint context
    # This works with torch.utils.checkpoint.checkpoint() calls
    # The hooks are set globally via checkpoint context
    import torch.utils.checkpoint as ckpt

    # Set the default checkpoint context to use FP8 packing
    if hasattr(ckpt, "set_checkpoint_early_stop"):
        # torch 2.12+ has configurable checkpoint context
        pass

    # For now, we provide the packer as a utility that blocks can use
    # The actual integration is in gpt.py's checkpoint call
    setattr(model, "_fp8_activation_packer", FP8ActivationPacker)

    n_blocks_with_ckpt = 0
    for block in getattr(getattr(model, "transformer", None), "h", []):
        if getattr(getattr(block, "config", None), "gradient_checkpointing", False):
            n_blocks_with_ckpt += 1

    logger.info(
        f"FP8 activation compression enabled (native mode). "
        f"~2x activation memory savings on {n_blocks_with_ckpt} checkpointed blocks."
    )

    return {
        "enabled": True,
        "backend": "native",
        "blocks_with_checkpoint": n_blocks_with_ckpt,
        "packer": "FP8ActivationPacker",
    }


def _enable_coat_fp8(model: torch.nn.Module) -> dict:
    """Enable COAT FP8 training (activations + optimizer states).

    COAT provides:
    - Mixed-granularity activation quantization (per-tensor + per-group)
    - FP8 optimizer states (Adam m/v in FP8)
    - Dynamic range expansion for optimizer stability
    - 1.54x total memory reduction vs bf16
    """
    try:
        import coat  # pyright: ignore[reportMissingImports]
        # COAT API: coat.fp8_training.enable(model, ...)
        # The exact API depends on COAT version — check at runtime
        if hasattr(coat, "enable_fp8_training"):
            coat.enable_fp8_training(model)
        elif hasattr(coat, "fp8_model_init"):
            coat.fp8_model_init(model)
        else:
            logger.warning("COAT installed but API not recognized. Using native FP8.")
            return enable_fp8_activation_checkpointing(model, backend="native")

        logger.info("COAT FP8 training enabled (activations + optimizer states)")
        return {
            "enabled": True,
            "backend": "coat",
            "features": ["fp8_activations", "fp8_optimizer_states"],
        }
    except Exception as e:
        logger.warning(f"COAT initialization failed: {e}. Using native FP8.")
        return enable_fp8_activation_checkpointing(model, backend="native")


class LayerAwareFP8Packer:
    """FP8 pack/unpack hooks that skip quantization for designated BF16 layers.

    Wraps :class:`FP8ActivationPacker` but checks the thread-local
    ``_layer_state.current_layer_idx`` against the first/last BF16 layer
    configuration.  When the current layer is in the BF16 set, pack()
    returns the tensor unchanged (no quantization overhead, full precision
    preserved for the embedding-adjacent layers that benefit most from it).

    This follows Megatron-Core's ``first_last_layers_bf16`` pattern: the
    first N and last N transformer blocks run in BF16 even when FP8 is
    enabled, improving training stability for layers that directly consume
    raw embeddings or feed into lm_head.
    """

    def __init__(self, n_layers: int, bf16_first: int = 1, bf16_last: int = 1,
                 clamp: bool = False):
        self.n_layers = n_layers
        self.bf16_first = bf16_first
        self.bf16_last = bf16_last
        self._clamping_packer = ClampingFP8Packer() if clamp else None

    def pack(self, tensor: torch.Tensor | int | None) -> PackedActivation:
        if _is_fp8_disabled():
            return tensor  # module opted out via fp8_quantize_disabled()
        layer_idx = _get_current_layer_idx()
        if is_bf16_layer(layer_idx, self.n_layers, self.bf16_first, self.bf16_last):
            return tensor  # BF16 layer — skip FP8 quantization
        if self._clamping_packer is not None:
            return self._clamping_packer.pack(tensor)
        return FP8ActivationPacker.pack(tensor)

    def unpack(self, packed) -> torch.Tensor:
        return FP8ActivationPacker.unpack(packed)


def get_fp8_activation_checkpoint_context(
    *,
    n_layers: int = 0,
    bf16_first: int = 0,
    bf16_last: int = 0,
    clamp: bool = False,
):
    """Get a context manager for FP8 activation checkpointing.

    Use this in place of torch.utils.checkpoint.checkpoint():

        from nanochat.fp8_activations import get_fp8_activation_checkpoint_context

        ctx = get_fp8_activation_checkpoint_context()
        with ctx:
            output = checkpoint(block, input, use_reentrant=False)

    When *bf16_first* or *bf16_last* > 0, returns a layer-aware context
    that skips FP8 quantization for the first/last N layers.  The caller
    must call :func:`set_current_layer_idx` before each block's forward.

    When *clamp* is True, outliers are hard-clipped to the FP8 representable
    range ([-448, 448] for e4m3fn) before computing the per-tensor scale.
    This replaces the old per-forward ``--fp8_activation_clamp`` path
    (two ``.clamp()`` calls per MLP layer) with zero per-forward overhead --
    the clamp is folded into the pack hook's quantisation pass.

    Args:
        n_layers: Total number of transformer layers in the model.
        bf16_first: Number of initial layers kept in BF16 (default 0).
        bf16_last: Number of final layers kept in BF16 (default 0).
        clamp: Clamp outliers to FP8 range before quantising (default False).
    """
    if not _FP8_AVAILABLE:
        from contextlib import nullcontext
        return nullcontext()

    if bf16_first > 0 or bf16_last > 0:
        packer = LayerAwareFP8Packer(n_layers, bf16_first, bf16_last, clamp=clamp)
        return torch.autograd.graph.saved_tensors_hooks(
            packer.pack,
            packer.unpack,
        )

    if clamp:
        clamp_packer = ClampingFP8Packer()
        return torch.autograd.graph.saved_tensors_hooks(
            clamp_packer.pack,
            clamp_packer.unpack,
        )

    return torch.autograd.graph.saved_tensors_hooks(
        FP8ActivationPacker.pack,
        FP8ActivationPacker.unpack,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Inductor partitioner FP8 awareness
# ═══════════════════════════════════════════════════════════════════════════

def fp8_budget_adjustment(budget: float) -> float:
    """Adjust activation_memory_budget for FP8 activation compression.

    The inductor partitioner overestimates activation memory by ~2x because
    it uses dtype.itemsize (bf16=2 bytes) but saved_tensors_hooks compress
    activations to FP8 (1 byte). Compensate by doubling the budget.

    Args:
        budget: Original activation_memory_budget (0.0-1.0)
    Returns:
        Adjusted budget, capped at 1.0
    """
    if budget >= 1.0:
        return budget
    return min(1.0, budget * 2.0)
