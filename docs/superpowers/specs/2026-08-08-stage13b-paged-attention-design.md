# Stage 13B — Block-Aware Paged Attention Decode

## Scope

Stage 13B removes compact dense historical K/V materialization from the normal paged decode path.
It builds on Stage 13A ownership, reservation, rollback, cancellation, and evidence contracts. Dense
execution remains the numerical reference, and the existing Stage 11 continuous executor remains
available.

Initial prompt prefill and learned-position overflow re-prefill still use the existing dense model
interfaces. Only single-token decode against an existing paged cache becomes block-aware. Prefix
sharing, copy-on-write, variable block sizes, GPU kernels, BPE, distributed serving, and checkpoint
storage remain outside scope.

This stage is a Python/PyTorch reference implementation. It does not provide a high-performance
fused PagedAttention kernel, and its descriptive benchmark does not support a speedup claim.

## Read-only block views

`PagedKVCachePool` exposes a read-only request view for each model layer. A view contains the ordered
physical block slices, logical cache length, and configured block size. Slices alias pool storage;
constructing a view does not concatenate, clone, or pad historical K/V.

```text
request table [7, 2, 19]
    -> layer 0 views of blocks 7, 2, 19
    -> layer 1 views of blocks 7, 2, 19
    -> ...
```

The final slice is shortened to the request's actual tail occupancy, so unused tail slots never
participate in attention. The view is valid only during the owner-thread decode call; allocator
mutation remains owned by `EngineRunner` through `ServingEngine`.

## Block-aware attention

For each Transformer layer, the model projects the current token into Q/K/V once. For each request
row, it computes score chunks directly against its ordered historical key-block views, followed by
the current key. Score chunks are concatenated only along the scalar token-score dimension for one
global softmax. The weighted context is then accumulated block by block from value views.

```text
Q(current) @ K(block 0..n) -> score chunks -> one softmax
softmax slices @ V(block 0..n) -> accumulated context
```

Historical K/V tensors are never assembled into a compact or padded dense tensor. Concatenating
attention scores is intentional: softmax must normalize over the complete logical sequence. The
model returns only the current token's per-layer K/V delta. The pool appends that delta transactionally
and advances logical cache length by one.

## Executor and lifecycle

`PagedAttentionExecutor` reuses the Stage 11B batched prefill policy and Stage 9 overflow fallback.
For normal decode it groups all eligible requests into one model call while allowing every row to
have a different block table and cache length. Invalid requests fail in isolation before batching;
sampling remains per-request with the existing generator, temperature, and top-k semantics.

`ServingEngine` selects the direct path only for this executor and verifies that the executor and
engine reference the same pool. Other executors on the paged backend retain Stage 13A dense
materialization. The direct executor is invalid with the dense backend.

## Correctness and performance policy

Tests compare dense continuous decode, Stage 13A paged-plus-materialization, and direct paged
attention under identical weights, workloads, scheduler settings, and request seeds. Required
equivalence includes generated tokens, terminal state/cancellation, FIFO admission, logical events,
request metrics, overflow behavior, and zero resource leaks. Model tests additionally compare logits
and the appended K/V delta within explicit floating-point tolerances.

A guard test replaces `PagedKVCachePool.materialize` with a failure during direct decode; successful
generation proves that the normal Stage 13B path does not call it. Overflow re-prefill remains an
explicit dense exception and is measured separately.

Benchmarks are CPU-descriptive and separate block-view assembly, block-aware model time, delta
scatter, end-to-end time, and Stage 13A dense-materialization time. A speedup claim is allowed only
for stable, comparable measurements; otherwise the verdict is `not_comparable` or descriptive only.
