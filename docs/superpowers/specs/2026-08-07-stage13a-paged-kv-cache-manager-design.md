# Stage 13A — Paged KV Cache Memory Manager

## Scope

Stage 13A replaces long-lived per-request dense KV tensors with an optional fixed-block storage
backend. The existing dense backend remains the reference. The paged backend changes storage and
admission accounting only: model weights, scheduler ordering, request RNG, generated tokens,
terminal states, logical events, and request latency definitions remain unchanged.

This is not PagedAttention. Decode temporarily materializes a compact dense cache from the request
block table, calls the Stage 11 executor, and commits only the newly appended token. Learned-position
overflow rebuilds the most recent full model window transactionally. Dense materialization can make
this backend slower; Stage 13A makes no throughput claim.

Prefix sharing, copy-on-write, variable blocks, an HTTP API change, BPE, GPU storage, distributed
execution, and checkpoint persistence are outside this stage.

## Physical layout and ownership

One physical block ID addresses the same token interval in every Transformer layer:

```text
key/value: [layers, num_blocks, heads, block_tokens, head_size]
```

The byte capacity is `2 * layers * num_blocks * heads * block_tokens * head_size * element_size`.
The pool tensors are ordinary runtime tensors owned by `PagedKVCachePool`; they are not model
parameters, buffers, or checkpoint state.

Each admitted request owns one table:

```text
request_id -> (block_ids, cache_length, reserved_blocks)
position   -> (position // block_tokens, position % block_tokens)
```

The logical block index selects an entry in `block_ids`, which selects the physical block. Physical
IDs need not be contiguous. The lowest available ID is allocated first so reuse is deterministic.
`EngineRunner` remains the single owner of serving mutations; HTTP threads only enqueue commands.

## Reservation, allocation, and admission

Logical Stage 10 token reservations remain visible and unchanged. Paged storage adds an independent
physical reservation:

```text
max_cache_tokens = min(model_block_size, prompt_tokens + max_new_tokens - 1)
required_blocks  = ceil(max_cache_tokens / block_tokens)
```

The engine rejects a request deterministically if `required_blocks > num_blocks`. Temporary pool
pressure leaves the FIFO head in `WAITING`; later requests cannot bypass it. Admission creates the
request table and reserves its worst-case block count, while physical allocation happens lazily as
prefill or decode writes cross block boundaries. Thus `reserved_blocks >= allocated_blocks`.

Finish, cancellation, isolated failure, runner shutdown, and catastrophic worker failure all release
the request table, owned blocks, and reservation. Releasing an unknown request or a block owned by a
different request is an error rather than a silent double free.

## Transaction and rollback rules

Prefill, append, and overflow rebuild first validate tensor shape, layer count, dtype, device, and
target capacity. New physical blocks are allocated through one rollback-capable transaction. The
request table is published only after all writes succeed. If validation, allocation, or tensor copy
fails, new blocks return to the free heap, overwritten existing slices are restored, and the original
table remains authoritative.

Normal decode writes only the final K/V position from the executor result. It does not rewrite prior
blocks. Overflow rebuild takes a backup of the request's occupied slices, resets the logical cache,
writes the latest compact window, and restores both tensors and metadata on failure. Request-level
commit failure becomes `FAILED`; peers continue to run.

The internal verifier checks after tests and evidence runs:

- free and owned block sets are disjoint;
- every physical block is exactly free or owned;
- a block has at most one owner;
- a request table contains no duplicate physical IDs;
- cache length is within allocated capacity;
- reservation covers allocation; and
- free plus owned block counts equal total capacity.

## Metrics and equivalence

Engine metrics retain logical reserved and occupied token counts. Paged storage additionally reports
total/free/allocated/reserved/peak blocks, used/allocated token slots, tail-block internal
fragmentation, and allocation/free/reuse counters. Fixed interchangeable blocks do not have the
traditional contiguous external-fragmentation metric.

The simulator can execute the same config with dense and paged storage and rejects divergence in
generated tokens, terminal states/cancellation, FIFO admission, logical events, request metrics, or
logical cache accounting. Storage-specific metrics are intentionally allowed to differ. Evidence is
hash-bound and its verifier reruns allocator invariants before accepting the package.

## Stage 13B boundary

Stage 13B should remove dense K/V materialization from decode without weakening these ownership and
rollback contracts. A block-aware attention path should consume the request block table directly,
compute scores over logical token order, append the new per-layer K/V position through this pool, and
retain a dense reference path for token equivalence. CPU benchmarks must separate block traversal,
attention/model time, and end-to-end latency before making any performance claim.
