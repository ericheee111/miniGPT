# Stage 15 — Cache-Aware Batched Paged Prefill

## Scope

Stage 15 restores tensor-level prefill batching for Automatic Prefix Caching requests while
continuing to skip cached-prefix Transformer work. It adds one reusable batched model primitive for
`paged history + variable-length new token segments`; the APC suffix is the first consumer and a
future Stage 16 prompt chunk can reuse the same contract.

This stage does not add chunked prefill, token-budget scheduling, partial-block sharing or
copy-on-write, KV-pressure preemption, speculative decoding, BPE, GPU/CUDA, custom fused kernels,
`torch.compile`, quantization, distributed serving, or a new HTTP API. `ServingEngine` and
`PagedKVCachePool` remain the only owners of scheduling, reservations, block ownership, promotion,
reference counts, eviction, cancellation, and shutdown cleanup.

## Execution strategies

The paged executor exposes an internal prefix-prefill strategy:

```text
sequential  Stage 14 reference: one suffix model call per non-exact request
batched     Stage 15 path: FIFO suffix groups and one model call per group
```

The strategy is selectable by the simulator and benchmark and may be wired through the existing
server construction path, but it is not a new HTTP request parameter. Stage 15 uses `batched` for
its production comparison; `sequential` remains available as a correctness and performance
reference.

Exact full-boundary hits are not members of either model-call group. They read immutable cached
boundary logits, sample with the request generator, and report zero model prefill calls.

## Batched paged-history model primitive

The model-level operation is conceptually:

```text
prefill_paged_batch(
    right_padded_new_tokens: LongTensor[B, T],
    new_token_lengths: LongTensor[B],
    paged_history_views: Sequence[PagedKVCacheView | None],
    past_lengths: LongTensor[B],
) -> (
    all_suffix_logits: Tensor[B, T, vocab],
    padded_suffix_kv_delta: KVCache[B, heads, T, head_size],
)
```

Every row has `1 <= new_token_lengths[row] <= T`, `past_lengths[row] >= 0`, and
`past_lengths[row] + new_token_lengths[row] <= model.block_size`. A zero past length requires a
`None`/empty history view; a positive past length requires a paged view whose every layer has the
same exact cache length. This permits zero-hit misses and prefix hits to share one call without
inventing dense historical K/V.

The returned logits and K/V delta are padded only for transport. Callers must slice every row to its
valid new-token length. Padding logits and padding K/V never participate in sampling, promotion, or
pool writes.

## Absolute learned positions

Positions are row-local:

```text
position[row, offset] = past_lengths[row] + offset
```

Only offsets below `new_token_lengths[row]` are semantically valid. Padding positions are replaced
with a safe in-range sentinel before the embedding lookup, so a short row with a near-window-limit
past cannot index beyond the learned position table merely because another row has a longer new
segment.

Examples in one call:

```text
row A: past=64, new=12 -> positions 64..75
row B: past=96, new=7  -> positions 96..102
row C: past=32, new=18 -> positions 32..49
```

Batch padding and other rows' prefix lengths cannot change these positions. Stage 9 overflow keeps
its existing dense rebuild path and is not routed through this primitive after the model window is
full.

## Block-aware batched suffix attention

Token/position embedding, QKV projection, layer normalization, output projection, MLP, and final
logit projection remain batched over `[B, T, ...]`. Inside attention, the Python/PyTorch reference
implementation iterates by batch row and then by ordered physical blocks.

For one row with paged history `H` and valid new keys/values `S`, it computes score chunks for each
physical history block plus a causal suffix score matrix. The global softmax is taken across:

```text
all paged historical keys + valid current-suffix keys
```

The value result is accumulated block by block plus the current suffix values. Historical K/V is
never concatenated or compact-materialized. Concatenating attention score chunks is allowed; calling
`PagedKVCachePool.materialize()` or constructing an equivalent dense historical K/V tensor on the
normal Stage 15 path is forbidden.

For suffix tokens `E F G` after cached `A B C D`, the causal relation is:

```text
E -> A B C D E
F -> A B C D E F
G -> A B C D E F G
```

Right-padding positions are neither valid queries nor keys. Their intermediate tensors may be
computed by ordinary batched projections, but are ignored at every row-scoped attention and are
discarded during scatter.

## FIFO suffix grouping

The executor validates requests independently, removes exact hits from model work, and considers the
remaining requests in their original order. Grouping uses computed suffix lengths:

```text
computed_length = len(prompt_tokens) - prefix_hit_tokens
useful_tokens   = sum(computed_length)
padded_tokens   = batch_size * max(computed_length)
padding_waste   = (padded_tokens - useful_tokens) / padded_tokens
```

The existing `max_batch_size`, `max_batch_tokens`, and `max_padding_ratio` bounds apply to these
values. Cached prefix tokens consume neither useful nor padded compute budget. A request that does
not fit closes the current FIFO group and starts the next; requests are never reordered and the
executor does not wait for future arrivals.

Zero-hit rows use `past_length=0` and can share a group with prefix-hit rows. This is intentionally a
single model call through the common primitive, not a claim that a separate dense call is unified.

## Scatter, sampling, and promotion

After one successful model call, the executor scatters results in the original row order:

1. slice all valid suffix logits for the row;
2. slice each layer's suffix K/V delta to the valid new-token length;
3. sample the first generated token from the row's final valid suffix logit using only that request's
   generator; and
4. return `ExecutionResult` with the complete valid suffix-logit matrix.

Per-row sampling failures remain isolated. Batching composition cannot share or reorder RNG calls,
so dense reference, paged direct, sequential APC, and batched APC preserve identical generated
tokens and per-request generator state for identical request inputs.

The model does not mutate the pool. `ServingEngine` commits each successful row through the existing
transactional sequence:

```text
write suffix KV delta
promote complete prompt blocks
canonicalize duplicate promotions
release duplicate private blocks
update reservations and active refs
```

Complete block-boundary logits are selected from the retained valid suffix logits. Padding logits
cannot become promotion metadata. Partial prompt tails remain PRIVATE.

## Failure and cancellation semantics

A model/attention failure fails every request in that model-call group. No pool K/V or ownership
metadata has been mutated at that point, so normal terminal cleanup releases the already attached
shared refs and reservations.

Sampling occurs after the shared model call and remains per-row isolated. A promotion failure after a
successful batch affects that request's commit only: its private blocks and reservation are released,
canonical shared blocks remain valid, and other successfully committed rows are not rolled back.

Waiting, prefill, decode, HTTP disconnect, stream backpressure, explicit cancellation, worker
failure, and shutdown continue to use the Stage 14 lifecycle. Every mutation is followed by paged
pool invariant verification. Final active refs, private ownership, and reservations must be zero;
shutdown follows the Stage 14 policy for clearing resident zero-ref cache blocks.

## Observations and metrics

Prefill observations carry an explicit execution mode rather than inferring behavior from batch
size or prefix-hit counters:

```text
full_dense
sequential_apc_suffix
batched_apc_suffix
exact_cache_hit
overflow_dense_rebuild
```

They also record whether a model call occurred. Stage 15 aggregates:

- cache-aware prefill batches and model calls;
- suffix batch sizes, average/max suffix batch size;
- suffix useful/padded tokens and padding-waste ratio;
- exact-hit request count and batched-suffix request count;
- existing prefix-hit, computed-prefill, and avoided-prefill token counters.

An ideal four-row equal-length APC suffix group must produce one `batched_apc_suffix` observation and
one model call. The permanent fake-batching regression test spies on the model primitive and requires
`call_count == 1`, not merely a batch-size metric greater than one.

## Simulator and benchmark

The Stage 15 simulator runs sequential APC and batched APC from identical model/request seeds. It
compares generated tokens, generator hashes, terminal states, cancellation, FIFO admission, logical
events, prefix identity, cache ownership, and final cleanup. Batch-only events are filtered for the
logical equivalence comparison but retained as structural evidence.

Fresh-process benchmark workloads cover repeated prefix/short suffix, variable suffix, mixed prefix
lengths, exact hits mixed with suffix hits, concurrent same prefix, low reuse/random prompts, and
padding pressure. Sequential APC and batched APC retain raw samples and report median, MAD, and CV
for TTFT, E2E, request/token throughput, and peak RSS.

Structural claims are independent:

- cached-prefix Transformer work avoided;
- prefill model-call count reduced; and
- realized suffix batch size/padding cost.

Wall-clock improvement is claimed only when the aggregate strict comparison verdict is `pass`. A
`fail` or `not_comparable` verdict is reported as `no wall-clock performance improvement claim`.

## Evidence package

Stage 15 publishes a new immutable package without modifying Stage 7–14 evidence:

```text
docs/results/cache-aware-batched-prefill/
├── README.md
├── summary.json
├── artifact_manifest.json
└── evidence/
    ├── correctness.json
    ├── batching.json
    ├── benchmark.json
    └── lifecycle_tests.json
```

The manifest binds exact membership, byte counts, SHA-256 digests, and the reviewed source commit.
The README states the Python/PyTorch reference boundary and explicitly excludes chunked prefill,
partial-block COW, preemption, and fused kernels.

## Verification and forward compatibility

Correctness covers equal/variable suffix lengths, different past lengths, mixed hit/miss/exact rows,
duplicate promotion, overflow, model and sampling failure, cancellation, disconnect, backpressure,
and shutdown. A deterministic randomized matrix mixes these operations and verifies pool invariants
after every mutation.

Stage 16 may call the same model primitive with `past paged history + next prompt chunk`. Stage 15
does not add a prefill cursor, intermediate chunk state, or token-budget scheduler. The primitive
therefore returns all new-segment logits and K/V delta without deciding whether the segment is a final
prompt suffix or an intermediate future chunk.
