# Stage 14 — Automatic Prefix Caching

## Scope

Stage 14 extends the Stage 13 fixed physical block pool with automatic reuse of immutable,
full-token-block prompt prefixes. It does not add partial-block sharing, copy-on-write, chunked
prefill, speculative decoding, GPU/custom kernels, a new HTTP API, or model architecture changes.

The existing `ServingEngine` owner thread remains the only mutator of scheduler, request, block,
reference-count, reservation, and LRU state. Dense and uncached direct-paged execution remain the
correctness references.

## Namespace and hash identity

Every cache instance has one canonical namespace document containing:

- checkpoint/model identity;
- model configuration identity;
- dtype and device;
- physical `block_tokens`;
- prefix-cache schema version; and
- learned absolute-position embedding semantics.

The namespace is serialized deterministically and hashed with SHA-256. Prefix block hashes form a
chain:

```text
h0 = SHA256(namespace_digest || token_block_0)
hn = SHA256(namespace_digest || h(n-1) || token_block_n)
```

The cache also stores the exact token tuple and its independent fingerprint for collision defense.
A hash match without exact metadata equality is an invariant failure, never a cache hit.

## Physical block states

Each physical block is exactly one of:

```text
FREE
PRIVATE(owner=request_id)
SHARED(prefix_hash, active_refcount, last_used)
```

SHARED blocks are immutable. Active request tables hold explicit references; finish, cancel,
failure, overflow rebuild, and shutdown decrement them exactly once. A zero-ref SHARED block may
remain resident and becomes LRU-evictable. A positive-ref block cannot be evicted. Each prefix hash
maps to at most one canonical physical block.

Only complete prompt blocks are promoted. The final incomplete prompt block and all ordinary decode
growth remain PRIVATE.

## Lookup, suffix prefill, and boundary logits

Admission computes the prompt's full-block hash chain and stops at the first miss. It attaches only
the longest contiguous prefix and records hit blocks/tokens plus miss tokens.

For a partial hit, the model evaluates only `prompt[prefix_hit_tokens:]`. Suffix queries attend the
ordered shared prefix K/V block views and causally attend earlier suffix K/V. Absolute learned
positions start at `prefix_hit_tokens`. Historical prefix K/V is not compact-materialized and prefix
Transformer layers are not rerun.

Promotion stores detached next-token logits at every complete prompt-block boundary. Therefore an
exact prompt ending on a fully cached boundary can sample directly from the cached boundary logits
without recomputing even its final prompt token. Boundary logits are namespace-bound immutable
metadata, not model state or checkpoint state.

Learned-position overflow keeps Stage 9 semantics: the latest full model window is rebuilt densely,
all old shared references are detached transactionally, and no old absolute-position K/V is reused.

## Promotion and duplicate handling

After successful prompt prefill, each complete PRIVATE prompt block is promoted in logical order.
If its hash is absent, the block becomes the canonical SHARED block. If a canonical block already
exists, exact token metadata is verified, the request table is switched to the canonical block, and
the duplicate PRIVATE block is returned. Promotion, duplicate replacement, refcounts, reservations,
and rollback are one exception-safe mutation.

Completion blocks are not promoted in this stage.

## Capacity, reservation, and eviction

Capacity protection distinguishes:

- resident active SHARED blocks;
- request PRIVATE allocations and future PRIVATE growth reservation;
- resident zero-ref cached SHARED blocks; and
- truly FREE blocks.

The invariant is:

```text
unique active shared occupancy + protected private capacity <= pool capacity
```

Zero-ref cached blocks do not consume protected capacity because they can be evicted before a
PRIVATE allocation or admission attachment. Admission performs lookup, deterministic LRU eviction,
shared attachment, and private reservation atomically. Allocation uses FREE blocks first, then
evicts zero-ref blocks ordered by `(last_used, block_id)`. Active SHARED blocks are never eviction
candidates. If capacity is still insufficient, existing FIFO/backpressure semantics apply.

## Events, metrics, and verification

Stage 14 adds `PREFIX_LOOKUP`, `PREFIX_HIT`, `PREFIX_PROMOTE`, and `PREFIX_EVICT` events. Existing
logical request events remain unchanged when these cache-only events are filtered.

Metrics include lookup/hit requests and tokens, miss/computed prefill tokens, resident and evictable
cache blocks, evictions, and reused blocks. The primary workload-independent performance evidence is
avoided prefill tokens. Wall-clock results remain fresh-process descriptive evidence unless the
strict median/MAD/CV comparison policy passes.

Every mutation ends with invariant verification. Deterministic randomized stress covers lookup,
attach, allocate, promote, release, eviction, cancel/fail cleanup, and shutdown. Shutdown releases
all active refs, PRIVATE ownership, and reservations; this implementation also clears zero-ref
prefix-cache residency for a completely FREE final pool.
