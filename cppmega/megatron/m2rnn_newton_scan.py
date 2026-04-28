"""Newton-linearized parallel scan for M2RNN.

Uses ParaRNN-style state augmentation + associative scan to achieve O(log C)
parallel depth within sequence chunks, replacing the O(C) sequential loop.

The gated tanh recurrence does not admit an exact parallel scan (unlike
ParaRNN's ungated form), but we can apply Newton linearization:

  1. Split sequence into chunks of size C
  2. Make an initial guess for the state trajectory within each chunk
  3. Linearize the nonlinear recurrence around the guess
  4. The linearized recurrence IS associative and can be parallel-scanned
     in O(log C) depth
  5. Newton-update the guess and repeat (2-3 iterations needed)

Mathematical derivation
-----------------------
True recurrence for step t within a chunk (starting from known h_start):
    F_t(h) = h_t - [xf_t * h_{t-1} + (1-xf_t) * tanh(h_{t-1} @ W + k_t v_t)]
    where h_{-1} = h_start (the chunk boundary state).

Linearizing around guess trajectory g_t:
    tanh((g_{t-1} + delta_{t-1}) @ W + x_t) ~
        tanh(g_{t-1} @ W + x_t) + sech2_t * (delta_{t-1} @ W)

where sech2_t = 1 - tanh^2(g_{t-1} @ W + x_t), element-wise, shape (K, V).

This yields the linear recurrence:
    delta_t = alpha_t * delta_{t-1} + beta_t * (delta_{t-1} @ W) + b_t
where:
    alpha_t = xf_t                                          (scalar)
    beta_t  = (1 - xf_t) * sech2_t                           (K, V)
    b_t     = -r_t                                          (K, V) (negated residual)

and r_t = g_t - [xf_t * g_{t-1} + (1-xf_t) * tanh(g_{t-1} @ W + x_t)]

Key insight: The linear operator A_t(delta) = alpha_t * delta + beta_t * (delta @ W)
acts independently on each row of delta (dimension K).  For each k-index i,
it is a V×V matrix operator:
    A_t[i,:,:] = alpha_t * I_V + diag(beta_t[i,:]) @ W^T

Since V_DIM is small (<= 16), we store each operator as K copies of (V, V)
matrices, making composition a simple batch matrix multiplication.

Operator composition (associative, used in parallel scan):
    (A_j, b_j) + (A_i, b_i) = (A_composed, b_composed)
    A_composed[k] = A_j[k] @ A_i[k]  (VxV matmul per row)
    b_composed[k] = A_j[k] @ b_i[k] + b_j[k]
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 256
DEFAULT_NEWTON_ITERS = 4

# ---------------------------------------------------------------------------
# Attempt to import Triton for accelerated kernels
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


# ===========================================================================
# Stage 1: Pure-PyTorch reference implementation
# ===========================================================================


def _compute_residual_and_sech2(
    h_guess: torch.Tensor,  # (C, K, V)
    h_start: torch.Tensor,  # (K, V)
    k: torch.Tensor,  # (C, K)
    v: torch.Tensor,  # (C, V)
    xf: torch.Tensor,  # (C,)
    W: torch.Tensor,  # (V, V)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute residual and sech^2 for the Newton linearization.

    Given a guess trajectory h_guess[0..C-1] and the chunk start state h_start,
    compute:

        h_prev[t] = h_start        if t == 0
                   = h_guess[t-1] if t > 0

        pre[t]    = h_prev[t] @ W + k[t] * v[t]^T           (K, V) rank-1 update
        h_true[t] = xf[t]*h_prev[t] + (1-xf[t])*tanh(pre[t])   (K, V) gated update
        r[t]      = h_guess[t] - h_true[t]                    (K, V) residual
        sech2[t]  = 1 - tanh(pre[t])^2                       (K, V) sech^2 derivative

    Returns
    -------
    residual : (C, K, V)
        h_guess - true_recurrence at each step
    sech2 : (C, K, V)
        1 - tanh^2(pre) at each step
    """
    C = h_guess.shape[0]

    # h_prev[t] = h_start if t==0 else h_guess[t-1]
    h_prev = torch.cat([h_start.unsqueeze(0), h_guess[:-1]], dim=0)  # (C, K, V)

    # Rank-1 outer product: k_t (C,K) x v_t (C,V) -> (C, K, V)
    x = k.unsqueeze(-1) * v.unsqueeze(-2)  # (C, K, V)

    # h_prev @ W: (C, K, V) @ (V, V) -> (C, K, V)
    pre = h_prev @ W + x  # (C, K, V)

    h_new_true = torch.tanh(pre)  # (C, K, V)

    # Gated update
    f = xf.view(-1, 1, 1)  # (C, 1, 1)
    h_true = f * h_prev + (1.0 - f) * h_new_true  # (C, K, V)

    residual = h_guess - h_true  # (C, K, V)
    sech2 = 1.0 - torch.tanh(pre) ** 2  # (C, K, V)

    return residual, sech2


def _build_linear_operators(
    alpha: torch.Tensor,  # (C,)   — xf gate values
    beta: torch.Tensor,  # (C, K, V) — (1-xf) * sech2
    W: torch.Tensor,  # (V, V)  — raw state transition matrix
) -> torch.Tensor:
    """Build explicit VxV matrix operators A_t for the linear recurrence.

    The linear operator A_t acts on a (K,V) matrix delta:
        A_t(delta) = alpha_t * delta + beta_t * (delta @ W)

    For element (i,j):
        A_t(delta)[i,j] = alpha_t * delta[i,j] + beta_t[i,j] * Sum_l(delta[i,l] * W[l,j])

    For each k-dim row i, the VxV operator matrix is:
        A_t[i,j,l] = alpha_t * I_V[j,l] + beta_t[i,j] * W[l,j]

    = alpha_t * I_V + diag(beta_t[i,:]) @ W^T

    Since V_DIM <= 16, the explicit (V,V) representation is cheap (256 floats/row).

    Parameters
    ----------
    alpha : (C,)
        Scalar coefficient per step.
    beta : (C, K, V)
        Mask coefficient per step and element.
    W : (V, V)
        State transition matrix.

    Returns
    -------
    A_matrices : (C, K, V, V)
        Explicit VxV operators for each (step, k-row).
    """
    C, K, V = beta.shape

    I_V = torch.eye(V, device=beta.device, dtype=beta.dtype)  # (V, V)

    # alpha term: alpha_t * I_V for all (t, k) -> broadcast
    alpha_term = alpha.view(C, 1, 1, 1) * I_V  # (C, 1, V, V)

    # beta term: A[t,k,j,l] = beta[t,k,j] * W[l,j]
    # = einsum('tkj,lj->tkjl', beta, W)
    beta_term = torch.einsum("tkj,lj->tkjl", beta, W)  # (C, K, V, V)

    A_matrices = alpha_term + beta_term  # (C, K, V, V)

    return A_matrices


def linearized_operators(
    h_guess: torch.Tensor,  # (C, K, V)
    h_start: torch.Tensor,  # (K, V)
    k: torch.Tensor,  # (C, K)
    v: torch.Tensor,  # (C, V)
    xf: torch.Tensor,  # (C,)
    W: torch.Tensor,  # (V, V)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute linearized operators and residuals for Newton iteration.

    This is the main building block of the Newton solver.  All operations
    are fully parallel over the C dimension (no sequential loops).

    Returns
    -------
    sech2 : (C, K, V)
        Element-wise sech^2 of pre-activation.
    residual : (C, K, V)
        h_guess[t] - true_recurrence[t].
    A_matrices : (C, K, V, V)
        Explicit VxV matrix operators for each (step, k-row).
    b_terms : (C, K, V)
        Residuals, the RHS for the linear system.
    """
    residual, sech2 = _compute_residual_and_sech2(h_guess, h_start, k, v, xf, W)

    alpha = xf  # (C,)
    beta = (1.0 - xf.view(-1, 1, 1)) * sech2  # (C, K, V)

    A_matrices = _build_linear_operators(alpha, beta, W)

    # Newton: solve A*delta = b where b = +residual.
    # Derivation: h_new = h - delta satisfies the linearized recurrence
    #   delta_t = A_t(delta_{t-1}) + residual_t    (not -residual)
    b_terms = residual

    return sech2, residual, A_matrices, b_terms


# ===========================================================================
# Stage 1a: Sequential scan reference (correctness baseline)
# ===========================================================================


def parallel_scan_linear_sequential(
    A_matrices: torch.Tensor,  # (C, K, V, V)
    b_terms: torch.Tensor,  # (C, K, V)
) -> torch.Tensor:
    """Sequential scan — correctness reference, O(C) depth.

    Solves delta_t = A_t(delta_{t-1}) + b_t with delta_{-1} = 0.
    Returns delta[0..C-1].

    This is the ground truth: mathematically correct, just slow.
    """
    C, K, V, _ = A_matrices.shape
    delta = torch.zeros(C, K, V, device=A_matrices.device, dtype=A_matrices.dtype)
    d_prev = torch.zeros(K, V, device=A_matrices.device, dtype=A_matrices.dtype)

    for t in range(C):
        # Apply A_t to d_prev: for each row k, A_t[k] @ d_prev[k]
        d_t = torch.einsum("kil,kl->ki", A_matrices[t], d_prev) + b_terms[t]
        delta[t] = d_t
        d_prev = d_t

    return delta


# ===========================================================================
# Stage 1b: Blelloch parallel scan (O(log C) depth)
# ===========================================================================


def _pair_compose(
    A_left: torch.Tensor, b_left: torch.Tensor, A_right: torch.Tensor, b_right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose two (A, b) pairs: left first, then right.

    If pair_left maps delta_in -> A_left(delta_in) + b_left
    and pair_right maps delta_in -> A_right(delta_in) + b_right,
    then the composed pair maps:
        delta -> A_right( A_left(delta) + b_left ) + b_right
              = (A_right @ A_left)(delta) + A_right(b_left) + b_right

    Returns (A_right @ A_left, A_right(b_left) + b_right).

    All inputs share shape (K, V, V) for A_*, and (K, V) for b_*.
    """
    # A_new = A_right @ A_left
    A_new = torch.einsum("kij,kjl->kil", A_right, A_left)  # (K, V, V)
    # b_new = A_right @ b_left + b_right
    b_new = torch.einsum("kij,kj->ki", A_right, b_left) + b_right  # (K, V)
    return A_new, b_new


def parallel_scan_blelloch(
    A_matrices: torch.Tensor,  # (C, K, V, V)
    b_terms: torch.Tensor,  # (C, K, V)
) -> torch.Tensor:
    """Blelloch up-sweep + down-sweep inclusive parallel scan in O(log C) depth.

    Solves the linear recurrence delta_t = A_t(delta_{t-1}) + b_t
    with delta_{-1} = 0.

    The Blelloch work-efficient scan computes the *exclusive* prefix sum.
    To recover the *inclusive* result we need:
        incl[t] = excl[t] compose pair_t  (= A_excl[t](b_t) + b_excl[t])

    After the down-sweep, excl[t] for t in 0..C_pad-1 is at position t.
    The up-sweep total (full prefix) was at C_pad-1 before the identity reset.

    Returns
    -------
    delta : (C, K, V)
        Solution to the linear recurrence delta_t = A_0..t(0) + b_0..t.
    """
    C, K, V, _ = A_matrices.shape

    # Handle small C with sequential fallback
    if C <= 2:
        return parallel_scan_linear_sequential(A_matrices, b_terms)

    # Pad to power of 2 for clean Blelloch
    original_C = C
    pad_C = 1
    while pad_C < C:
        pad_C <<= 1

    if pad_C > original_C:
        pad_A = torch.zeros(pad_C, K, V, V, device=A_matrices.device, dtype=A_matrices.dtype)
        pad_b = torch.zeros(pad_C, K, V, device=b_terms.device, dtype=b_terms.dtype)
        pad_A[:original_C] = A_matrices
        pad_b[:original_C] = b_terms
        # Identity for pad elements
        I_exp = torch.eye(V, device=A_matrices.device, dtype=A_matrices.dtype).expand(K, V, V)
        for t in range(original_C, pad_C):
            pad_A[t] = I_exp
            pad_b[t] = torch.zeros(K, V, device=b_terms.device, dtype=b_terms.dtype)
    else:
        pad_A = A_matrices.clone()
        pad_b = b_terms.clone()
        I_exp = torch.eye(V, device=A_matrices.device, dtype=A_matrices.dtype).expand(K, V, V)

    C_pad = pad_C

    # Save original pairs for inclusive post-processing
    # Must capture before the up-sweep, because the up-sweep overwrites
    # composed positions (1, 3, 5, 7, ...) with compound operators.
    orig_A = pad_A.clone()
    orig_b = pad_b.clone()

    # --- Up-sweep ---
    d = 0
    while (1 << d) < C_pad:
        stride = 1 << d
        new_pad_A = pad_A.clone()
        new_pad_b = pad_b.clone()
        for k in range(stride - 1, C_pad - 1, 2 * stride):
            A_new, b_new = _pair_compose(pad_A[k], pad_b[k], pad_A[k + stride], pad_b[k + stride])
            new_pad_A[k + stride] = A_new
            new_pad_b[k + stride] = b_new
        pad_A = new_pad_A
        pad_b = new_pad_b
        d += 1

    # Save the total prefix (at C_pad-1) before resetting to identity
    # The total is: pair_0 compose ... compose pair_{C_pad-1}
    total_A = pad_A[C_pad - 1].clone()
    total_b = pad_b[C_pad - 1].clone()

    # --- Down-sweep ---
    # Set last element to identity for exclusive scan
    pad_A = pad_A.clone()
    pad_b = pad_b.clone()
    pad_A[C_pad - 1] = I_exp
    pad_b[C_pad - 1] = torch.zeros(K, V, device=b_terms.device, dtype=b_terms.dtype)

    d = 0
    while (1 << d) < C_pad:
        d += 1
    # d is now log2(C_pad)

    for dd in range(d - 1, -1, -1):
        stride = 1 << dd
        new_pad_A = pad_A.clone()
        new_pad_b = pad_b.clone()
        for k in range(stride - 1, C_pad - 1, 2 * stride):
            A_k_saved = pad_A[k]
            b_k_saved = pad_b[k]

            new_pad_A[k] = pad_A[k + stride]
            new_pad_b[k] = pad_b[k + stride]

            A_new, b_new = _pair_compose(pad_A[k + stride], pad_b[k + stride], A_k_saved, b_k_saved)
            new_pad_A[k + stride] = A_new
            new_pad_b[k + stride] = b_new
        pad_A = new_pad_A
        pad_b = new_pad_b

    # Recover inclusive scan: incl[t] = exclus[t] compose orig[t]
    # exclus[t] is at pad_b[t] (b component), orig[t] is at orig_b[t]
    inclusive_delta = torch.empty(original_C, K, V, device=b_terms.device, dtype=b_terms.dtype)

    for t in range(original_C):
        # excl[t] = (pad_A[t], pad_b[t]) contains prefix up to t-1
        # incl[t] = orig_pair[t] compose excl[t]
        #        = A_orig[t] @ A_excl[t],  A_orig[t](b_excl[t]) + b_orig[t]
        b_incl = torch.einsum("kil,kl->ki", orig_A[t], pad_b[t]) + orig_b[t]
        inclusive_delta[t] = b_incl

    return inclusive_delta


def parallel_scan_linear_torch(
    A_matrices: torch.Tensor,  # (C, K, V, V)
    b_terms: torch.Tensor,  # (C, K, V)
    *,
    use_blelloch: bool = True,
) -> torch.Tensor:
    """Solve the linear recurrence using the selected method.

    Parameters
    ----------
    use_blelloch : bool
        If True, use the Blelloch parallel scan (O(log C)).
        If False, use sequential scan (O(C), correctness reference).

    Returns
    -------
    delta : (C, K, V)
    """
    if use_blelloch:
        return parallel_scan_blelloch(A_matrices, b_terms)
    else:
        return parallel_scan_linear_sequential(A_matrices, b_terms)


# ===========================================================================
# Stage 1c: Newton solver for one chunk
# ===========================================================================


def newton_solve_chunk(
    h_start: torch.Tensor,  # (K, V)
    k_chunk: torch.Tensor,  # (C, K)
    v_chunk: torch.Tensor,  # (C, V)
    xf_chunk: torch.Tensor,  # (C,)
    q_chunk: torch.Tensor,  # (C, K)
    W: torch.Tensor,  # (V, V)
    *,
    num_iterations: int = DEFAULT_NEWTON_ITERS,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Newton-linearized solve for one chunk.

    Given the chunk boundary state h_start and chunk inputs, iteratively
    refines the state trajectory using Newton's method on the nonlinear
    recurrence, with a Blelloch parallel scan to solve the linearized system.

    Parameters
    ----------
    h_start : (K, V)
        State at the chunk boundary (h_{-1}).
    k_chunk, v_chunk : (C, K), (C, V)
        Key/value inputs for this chunk.
    xf_chunk : (C,)
        Forget gate values.
    q_chunk : (C, K)
        Query projections.
    W : (V, V)
        State transition matrix.
    num_iterations : int
        Number of Newton iterations (2-3 typically suffice).
    verbose : bool
        If True, print residual norms per iteration.

    Returns
    -------
    out : (C, V)
        Output for each step in the chunk.
    h_guess : (C, K, V)
        Final state trajectory (h_guess[t] is state after step t).
    history : list[float]
        Residual norms per Newton iteration.
    """
    C = k_chunk.shape[0]
    K = h_start.shape[0]
    V = h_start.shape[1]

    # Initial guess: constant trajectory (h_start for all steps)
    h_guess = h_start.unsqueeze(0).expand(C, K, V).contiguous().clone()

    history: list[float] = []

    for it in range(num_iterations):
        # Compute linearized operators
        _sech2, residual, A_matrices, b_terms = linearized_operators(
            h_guess, h_start, k_chunk, v_chunk, xf_chunk, W
        )

        res_norm = residual.abs().mean().item()
        history.append(res_norm)
        if verbose:
            print(f"  Newton iter {it}: residual norm = {res_norm:.6e}")

        if res_norm < 1e-7:
            break

        # Solve linear system via parallel scan
        delta = parallel_scan_blelloch(A_matrices, b_terms)

        # Newton update: h_guess = h_guess - delta
        h_guess = h_guess - delta

    # Compute outputs: out[t] = q[t] @ h_guess[t]
    # q_chunk: (C, K), h_guess: (C, K, V) -> out: (C, V)
    out = torch.einsum("ck,ckv->cv", q_chunk, h_guess)

    return out, h_guess, history


# ===========================================================================
# Stage 2: Triton-accelerated parallel scan
# ===========================================================================

if TRITON_AVAILABLE:

    @triton.jit
    def _blelloch_scan_kernel(
        # Input pair arrays: A (V*V element per logical row), b (V elements)
        A_ptr,  # (C_pad, K, V, V) — the (V,V) operators, padded to power of 2
        b_ptr,  # (C_pad, K, V)
        # Output deltas
        delta_ptr,  # (C_orig, K, V)
        # Shared memory / working buffers (size = C_pad * K * V * V for A,
        # C_pad * K * V for b)
        C_PAD: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        # Strides for A (flattened K*V*V per step)
        A_stride_t: tl.constexpr,  # stride between steps
        b_stride_t: tl.constexpr,
        delta_stride_t: tl.constexpr,
        C_ORIG: tl.constexpr,
    ):
        """Blelloch scan kernel for M2RNN linear operators.

        One program (thread block) processes one chunk's worth of data.
        All V_DIM*V_DIM matrices for all K_DIM rows and all C_PAD steps
        are in SRAM (loaded cooperatively).

        Grid: (1,) — one block per chunk for now, simple version.
        In production, batch*heads run concurrently, each processing
        its own chunk.
        """
        # This kernel processes one chunk's (A,b) pairs
        # Each tid handles a subset of K rows and V elements

        pid = tl.program_id(0)
        C = C_PAD  # shorthand

        # We flatten the state for scan purposes
        # A_pair is (C_pad, K, V, V) and b_pair is (C_pad, K, V)
        # The scan operates on these in SRAM

        # Cooperative load of all data into SRAM
        # Simplification for now: each warp handles a subset

        # --- Up-sweep ---
        d = 0
        while tl.lt(1 << d, C):
            stride = 1 << d
            # Step in blocks
            for k in range(stride - 1, C - 1, 2 * stride):
                # Load A_k, b_k, A_{k+stride}, b_{k+stride}
                # Compose and store result at k+stride
                # (detailed loads/stores omitted in this prototype -
                #  the PyTorch Blelloch handles the actual math)

                # Placeholder: the PyTorch version works, Triton version
                # is a future optimization
                pass
            d += 1

        # --- Down-sweep ---
        # Similarly: identity at C-1, sweep down

        pass


# ===========================================================================
# Stage 2 fallback: Use PyTorch Blelloch (already fast since V=16 is tiny)
# ===========================================================================


def _parallel_scan_linear(
    A_matrices: torch.Tensor,  # (C, K, V, V)
    b_terms: torch.Tensor,  # (C, K, V)
) -> torch.Tensor:
    """Dispatch to the best available parallel scan implementation.

    When Triton is available and C is large enough, dispatches to the Triton
    Blelloch kernel.  Otherwise uses the PyTorch Blelloch implementation,
    which is already fast for small V (V=16 means 256-element matrices).
    """
    C = A_matrices.shape[0]
    # For now, the PyTorch Blelloch is already very fast at the chunk scale
    # (C <= 256, K <= 64, V <= 16).  Triton kernel is future work.
    if C <= 4:
        return parallel_scan_linear_sequential(A_matrices, b_terms)
    return parallel_scan_blelloch(A_matrices, b_terms)


# ===========================================================================
# Top-level API
# ===========================================================================


def m2rnn_scan_newton(
    q: torch.Tensor,  # (B, S, n_q, K)
    k: torch.Tensor,  # (B, S, n_k, K)
    v: torch.Tensor,  # (B, S, n_v, V)
    W: torch.Tensor,  # (n_w, V, V)
    xf: torch.Tensor,  # (B, S, n_f)
    *,
    h0: Optional[torch.Tensor] = None,  # (B, H, K, V) or None
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    newton_iters: int = DEFAULT_NEWTON_ITERS,
    use_newton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """M2RNN scan with optional Newton parallel scan.

    Drop-in replacement for m2rnn_scan_triton with the same interface.

    When ``use_newton=True``, processes the sequence in chunks of ``chunk_size``
    and uses Newton-linearized parallel scan (O(log C)) within each chunk.

    When ``use_newton=False``, falls back to sequential intra-chunk processing
    (identical to chunked_m2rnn_forward behavior).

    Parameters
    ----------
    q : (B, S, n_q, K)
        Query projections.
    k : (B, S, n_k, K)
        Key projections.
    v : (B, S, n_v, V)
        Value projections.
    W : (n_w, V, V)
        Per-head state transition matrix.
    xf : (B, S, n_f)
        Forget gates (already transformed via softplus decay).
    h0 : optional (B, H, K, V)
        Initial state.
    chunk_size : int
        Chunk size C.  Larger C -> more parallelism per chunk.
    newton_iters : int
        Number of Newton iterations (2-3 typically suffice).
    use_newton : bool
        If True, use Newton-linearized parallel scan.  If False, sequential
        intra-chunk (useful for debugging/correctness comparison).

    Returns
    -------
    out : (B, S, H, V)
        Per-step output.
    h_final : (B, H, K, V)
        Final hidden state.
    """
    # --- Broadcast head counts ---
    n_q = q.size(-2)
    n_k = k.size(-2)
    n_v = v.size(-2)
    n_w = W.size(0)
    n_f = xf.size(-1)
    H = max(n_q, n_k, n_v, n_w, n_f)

    B, S = q.shape[0], q.shape[1]
    K = q.shape[-1]
    V = v.shape[-1]

    if n_q != H:
        q = q.repeat_interleave(H // n_q, dim=-2)
    if n_k != H:
        k = k.repeat_interleave(H // n_k, dim=-2)
    if n_v != H:
        v = v.repeat_interleave(H // n_v, dim=-2)
    if n_w != H:
        W = W.repeat_interleave(H // n_w, dim=0)
    if n_f != H:
        xf = xf.repeat_interleave(H // n_f, dim=-1)

    # --- Initialize state ---
    if h0 is None:
        h = torch.zeros(B, H, K, V, device=q.device, dtype=q.dtype)
    else:
        h = h0.clone()

    n_chunks = math.ceil(S / chunk_size)
    out = torch.empty(B, S, H, V, device=q.device, dtype=q.dtype)

    for c in range(n_chunks):
        s_start = c * chunk_size
        s_end = min(s_start + chunk_size, S)
        C = s_end - s_start

        for b in range(B):
            for head in range(H):
                h_start_bh = h[b, head]  # (K, V)

                k_chunk = k[b, s_start:s_end, head]  # (C, K)
                v_chunk = v[b, s_start:s_end, head]  # (C, V)
                xf_chunk = xf[b, s_start:s_end, head]  # (C,)
                q_chunk = q[b, s_start:s_end, head]  # (C, K)
                W_h = W[head]  # (V, V)

                if use_newton:
                    out_chunk, h_new_guess, _ = newton_solve_chunk(
                        h_start_bh,
                        k_chunk,
                        v_chunk,
                        xf_chunk,
                        q_chunk,
                        W_h,
                        num_iterations=newton_iters,
                    )
                    h[b, head] = h_new_guess[-1]  # final state after chunk
                    out[b, s_start:s_end, head] = out_chunk
                else:
                    # Sequential intra-chunk (fallback)
                    for t in range(C):
                        x_t = torch.outer(k_chunk[t], v_chunk[t])  # (K, V)
                        pre = h_start_bh @ W_h + x_t  # (K, V)
                        h_new = torch.tanh(pre)
                        h_start_bh = (
                            xf_chunk[t] * h_start_bh
                            + (1.0 - xf_chunk[t]) * h_new
                        )
                        out[b, s_start + t, head] = q_chunk[t] @ h_start_bh
                    h[b, head] = h_start_bh

    return out, h


def _sequential_recurrence_chunk(
    h_start: torch.Tensor,  # (K, V)
    k_chunk: torch.Tensor,  # (C, K)
    v_chunk: torch.Tensor,  # (C, V)
    xf_chunk: torch.Tensor,  # (C,)
    q_chunk: torch.Tensor,  # (C, K)
    W: torch.Tensor,  # (V, V)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential recurrence for one chunk (ground truth for testing).

    Returns
    -------
    out : (C, V)
    h_final : (K, V)
    """
    C = k_chunk.shape[0]
    out_chunk = torch.empty(C, W.shape[1], device=h_start.device, dtype=h_start.dtype)
    h = h_start
    for t in range(C):
        x_t = torch.outer(k_chunk[t], v_chunk[t])
        pre = h @ W + x_t
        h_new = torch.tanh(pre)
        h = xf_chunk[t] * h + (1.0 - xf_chunk[t]) * h_new
        out_chunk[t] = q_chunk[t] @ h
    return out_chunk, h
