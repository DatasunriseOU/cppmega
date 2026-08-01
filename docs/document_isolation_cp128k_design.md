# Packed-document attention with context parallelism

**Status**: open; production paths remain fail-closed (2026-08-01)

This is the implementation contract for packed logical documents at long
context. It is not evidence that context parallelism (CP) works with TE, FA4,
torch attention, DSA, or MTP.

## Current safe boundary

`document_ids` isolates attention and loss for multiple logical documents in
one physical row when `context_parallel_size == 1`.

The repository already has sequence gather/scatter and topology validation
helpers used by stateful sequence transforms. Those helpers do not make the
attention modules CP-aware: the attention patches currently see a local CP
slice while document boundaries describe the global logical sequence.

Until parity is proven, packed TE, FA4, torch, and DSA attention reject
`context_parallel_size > 1` through the shared
`_require_packed_attention_cp1` guard. MTP roll has its own equivalent
fail-closed check. This prevents silent cross-document mixing.

## Required implementation

### TE

Plumb the layer CP process group into the packed varlen path and establish
whether TE accepts global document `cu_seqlens` with its CP layout. Do not
remove the guard until the real TE runtime proves:

- CP=2 output and gradients match CP=1 for the same packed batch;
- the first and last token of every document cannot attend across a boundary;
- local and global sequence lengths survive Megatron's CP zigzag reorder.

### FA4

First implement the simple correctness path:

1. gather Q/K/V and `document_ids` into global sequence order;
2. build the existing `mask_mod` auxiliary data from that global layout;
3. run the unchanged FA4 score-mod kernel;
4. scatter output and gradients back to the local CP layout.

Only replace this with native FA4 CP arguments if beta23 exposes a compatible
`score_mod`/`mask_mod` contract and profiling shows gather/scatter is the
bottleneck.

### Torch and DSA

Keep these reference/fallback backends closed until they use the same global
document layout. Building a mask from only the local CP slice is incorrect
even when tensor shapes look valid.

### Stateful paths

MTP roll and every other transform whose state crosses sequence positions must
use the existing gather -> document-local operation -> scatter flow. A helper
existing in the repository is not proof that every caller routes through it.

## Acceptance gates

Local CPU tests must keep the fail-closed boundary and distributed
gather/scatter contracts green.

H200 validation must then pass, in order:

1. TE packed-document CP=2 versus CP=1 output and gradient parity;
2. FA4 packed-document CP=2 versus CP=1 and versus TE parity;
3. MTP/stateful CP=2 parity;
4. an end-to-end long-context training step with real sidecars;
5. restart from a checkpoint without loss or document-boundary drift.

Each gate needs the exact repository commits, dependency versions, launch
command, topology, and output receipt. The fail-closed guard is removed only
for a backend whose own gate passed.

## Decision

CP stays disabled for packed-document production training today. The immediate
landed change is one shared guard and an explicit contract, not a claim of CP
support. Implementation and H200 parity remain open work.
