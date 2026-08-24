# Stage 18 — Lazy KV Growth Reservation + Controlled Overcommit

## Goal

Stage 18 replaces admission-time worst-case KV protection with an opt-in, bounded growth model for the direct paged serving path.

A request still has a full lifetime cache demand, but admission protects only the KV capacity required for work that can execute immediately:

- a new request protects its prompt cache;
- a resumed request protects its recompute history;
- future decode cache capacity grows before the corresponding model work is scheduled.

This lets multiple requests coexist in a pool that could not protect every request's worst-case lifetime simultaneously. Stage 17 whole-request preemption remains the correctness fallback when a resident request cannot grow its protected KV capacity.

Stage 18 is a Python/PyTorch reference and control-plane feature. It makes no wall-clock speedup claim.

## Configuration

`SchedulerConfig` gains two opt-in fields:

```text
lazy_kv_reservation: bool = false
kv_overcommit_ratio: float = 1.0
```

The legacy path remains unchanged when `lazy_kv_reservation=false`.

Lazy reservation requires:

- `kv_preemption=true`;
- Stage 16 token-budget scheduling;
- the direct `PagedAttentionExecutor`;
- a paged KV-cache pool.

`kv_overcommit_ratio` must be finite and at least `1.0`. A ratio other than `1.0` is invalid when lazy reservation is disabled.

## Two reservation quantities

Every resident request has two distinct quantities.

### Full lifetime demand

The existing scheduler reservation formula remains the maximum cache tokens the request can require before completion:

```text
full_lifetime_tokens = min(
    block_size,
    min(prompt_length, block_size) + max(max_new_tokens - 1, 0),
)
```

A request whose own full lifetime demand exceeds the logical cache budget or the physical pool remains intrinsically impossible and fails immediately. It must never trigger preemption.

### Current protected capacity

Current capacity is what the engine can safely allocate without overcommit at this moment.

For lazy admission:

```text
new request       = min(prompt_length, block_size)
preempted resume  = len(all_tokens[:-1][-block_size:])
```

For the legacy path, current capacity equals full lifetime demand.

`RequestState.reserved_cache_tokens` and `reserved_cache_blocks` represent current protected capacity. Full lifetime demand is derived from the immutable request.

## Controlled overcommit

Admission is allowed only when all of these constraints hold:

```text
active current protected tokens + candidate current tokens
    <= max_cached_tokens

active current protected physical blocks + candidate current blocks
    <= physical pool capacity

active full lifetime demand + candidate full lifetime demand
    <= floor(max_cached_tokens * kv_overcommit_ratio)

active request count
    < max_active_requests
```

The ratio bounds risk explicitly. It does not permit one request to exceed the real logical or physical capacity by itself.

A waiting head blocked only by the aggregate lifetime-demand cap is a recoverable KV-pressure case: preempting another resident request removes that request's resident demand and may admit the waiting head.

## Paged allocator contract

A request table separates:

```text
current private protected blocks
current shared prefix blocks
full lifetime max_blocks
```

The invariant becomes:

```text
private allocated blocks
    <= current private protected blocks

current private protected blocks + current shared blocks
    <= max_blocks

sum(current private protected blocks)
  + active shared physical blocks
    <= pool capacity
```

Legacy reservations continue to satisfy equality between current total and `max_blocks`.

### Growth

The pool exposes deterministic reservation growth:

```text
can_grow_reservation(request_id, target_total_blocks)
grow_reservation(request_id, target_total_blocks)
```

Growth:

- never allocates or runs model work;
- never exceeds `max_blocks`;
- fails without mutation when protected capacity is unavailable;
- records growth operations and blocks;
- preserves all allocator invariants.

### APC

Prefix attachment counts shared blocks toward current total capacity. Prompt promotion converts a current private reservation into an active shared block without changing current protected total or `max_blocks`.

A preempted request releases all active shared references. Resume does not reattach original-position APC blocks; it protects and rebuilds private recompute history as established by Stage 17.

### Overflow rebuild

Overflow rebuild may use only current protected total capacity. It must not silently expand a lazy reservation to `max_blocks`.

When shared blocks are released for a private rebuild, the same current total capacity is converted to private protection before allocation. Failure restores block metadata, tensors, request-table fields, free lists, prefix index, and counters.

## Work-time reservation growth

Before every `DECODING` or `RECOMPUTING` operation, the scheduler computes the cache capacity the result will require:

```text
normal decode            = cached_tokens + 1
overflow dense rebuild   = resulting sliding-window cache length
preemption recompute     = actual history length
```

The target must not exceed the request's full lifetime demand.

The scheduler grows logical and physical protection before selecting the operation. If growth fails:

- the model operation is not executed;
- request output, RNG, cache, and lifecycle do not advance;
- the blocked request and target are recorded;
- later FIFO decode rows are deferred for that tick.

Successful growth emits `RESERVATION_GROWN`. A failed attempt emits `RESERVATION_GROWTH_BLOCKED` and increments request/engine counters.

Model-token budget accounting remains unchanged. Reservation mutation costs no model tokens.

## Growth-pressure preemption

A growth-blocked resident request cannot be its own victim.

Victim selection is deterministic:

1. scan active requests in FIFO order;
2. prefer another `DECODING` request that successfully decoded in the current tick;
3. if none exists, select the first other `DECODING` request that did not just finish recompute in the current tick;
4. preempt at most one request per tick.

The fallback in step 3 is required to break a reservation deadlock where the FIFO decoder cannot grow and therefore no request receives decode work in that tick.

After the victim releases its reservation, the engine immediately retries reservation growth for the blocked request. This retry performs no model work. It prevents the preempted victim from being re-admitted on the next tick and reclaiming the capacity before the blocked request can grow.

The ordinary Stage 17 waiting-head pressure policy retains its progress gate. Preemption events identify whether pressure came from a waiting head or reservation growth.

## Lifecycle and cleanup

Cancellation, request failure, recompute failure, HTTP disconnect, stream backpressure, graceful shutdown, and catastrophic owner-thread cleanup must clear:

- current logical reservation;
- current physical reservation;
- growth-blocked cursor and target when they refer to the request;
- retained intermediate prefill logits;
- private blocks and active shared references.

No terminal request may retain growth pressure state or KV resources.

## Metrics and events

New request metrics:

```text
reservation_growth_count
reservation_growth_tokens
reservation_growth_blocked_count
```

New engine metrics:

```text
lazy_kv_reservation_enabled
kv_overcommit_ratio
lifetime_reserved_cache_tokens
overcommitted_cache_tokens
reservation_growths
reservation_growth_tokens
reservation_growth_blocked
growth_pressure_preemptions
peak_lifetime_reserved_cache_tokens
peak_overcommitted_cache_tokens
```

New events:

```text
RESERVATION_GROWN
RESERVATION_GROWTH_BLOCKED
```

`PREEMPTED` detail identifies `waiting_head` or `reservation_growth` pressure.

## Correctness contracts

Stage 18 must prove:

1. Lazy admission protects less than full lifetime demand.
2. Controlled overcommit never exceeds the configured ratio.
3. A tiny pool can hold more concurrent requests than the full-reservation baseline.
4. Every model operation has enough protected cache capacity before execution.
5. Reservation growth is deterministic and allocator-safe.
6. Growth pressure preempts another decoder, retries growth immediately, and cannot deadlock.
7. Lazy-pressure generated tokens and per-request RNG match a roomy full-reservation reference.
8. Learned-position overflow remains equivalent.
9. Preempt/recompute resume remains equivalent.
10. APC attachment, promotion, release, and later growth preserve refcounts and positions.
11. Intrinsically impossible logical and physical waiting heads fail without preemption.
12. Cancellation, failure, and shutdown release all resources.
13. Finite mixed workloads complete without starvation.
14. Per-tick model work remains within the Stage 16 token budget.

## Evidence and claim policy

The evidence package is stored under:

```text
docs/results/lazy-kv-reservation/
```

It records:

- current versus lifetime reservation;
- configured and observed overcommit ratio;
- avoided upfront protected blocks;
- concurrent residency versus full reservation;
- growth attempts, successes, blocks, and pressure preemptions;
- recompute and overflow work;
- correctness and RNG hashes;
- deterministic stress and lifecycle results;
- exact artifact membership and SHA-256 hashes.

The benchmark verdict is `descriptive_only`. `wall_clock_performance_improvement` remains `false`.

## Scope boundaries

Stage 18 does not implement:

- CPU swap or offload;
- partial-block sharing or copy-on-write;
- speculative decoding;
- GPU/CUDA or fused kernels;
- scheduler priorities or configurable victim policies;
- a new HTTP request API;
- wall-clock performance claims.
