# Stage 17 — KV-Pressure Preemption + Recompute Resume

## Goal

Stage 17 closes the serving resource-management loop left after Stage 16. The current paged allocator protects each admitted request's worst-case lifetime KV capacity up front. That prevents allocator OOM, but under a tiny pool the FIFO head can remain blocked until an earlier long-running request finishes completely.

Stage 17 adds deterministic whole-request preemption under real KV reservation pressure. A decoding request may relinquish its resident KV cache and reservation, keep its logical generation state, re-enter the waiting queue, later rebuild the cache without sampling, and continue decoding with identical request-local RNG semantics.

This is a control-plane/reference feature. It does not claim a wall-clock speedup.

## Why this stage

Stage 13 established paged ownership and rollback. Stage 14 added shared immutable prefix blocks. Stage 15 added cache-aware suffix prefill. Stage 16 added actual model-token budgeting and decode/prefill fairness. The remaining serving gap is that resident requests cannot yield KV capacity to blocked requests.

A CPU swap path would add no useful storage tier in this CPU-only project. Dynamic/lazy reservation overcommit would simultaneously rewrite allocator capacity invariants and scheduler policy. Stage 17 deliberately keeps the existing full-reservation safety model while adding the smallest useful preemption/recompute loop. Lazy reservation can be evaluated later with Stage 17 correctness as a baseline.

## Configuration

`SchedulerConfig` gains one opt-in field:

```text
kv_preemption: bool = false
```

Stage 17 requires:

- paged KV backend;
- direct `PagedAttentionExecutor`;
- Stage 16 token-budget scheduling (`max_scheduled_tokens` + `prefill_chunk_tokens`).

The legacy path remains unchanged when `kv_preemption=false`.

## Lifecycle

Two non-terminal request states are added:

```text
PREEMPTED
RECOMPUTING
```

A preempted request keeps:

- original prompt;
- already generated tokens;
- request-local `torch.Generator` state;
- first-token and output timestamps;
- prefix-hit accounting already observed;
- failure/cancellation semantics.

It releases:

- request block table;
- private KV blocks;
- active shared-prefix references;
- protected reservation;
- dense `kv_cache` pointer;
- retained intermediate prefill logits.

The request is appended to the FIFO waiting tail.

## Pressure trigger

Preemption is evaluated after the current tick's scheduled model work. This guarantees that a resident request makes progress before it can yield capacity.

A pressure event exists only when the current waiting head cannot fit because of logical KV reservation or paged physical protected capacity. Hitting only `max_active_requests` is not a KV-pressure event.

Stage 17 never preempts `PREFILLING` or `RECOMPUTING` requests. Only `DECODING` requests are victims.

Victim policy is deterministic round-robin friendly FIFO:

1. scan active requests in active/FIFO order;
2. only consider a `DECODING` request that successfully completed decode work in the current tick;
3. choose the first such eligible request;
4. exclude a request that completed recomputation in the current tick;
5. preempt at most one victim per tick.

This progress gate matters when Stage 16 fairness gives prefill the current tick: a decoder that was deliberately deferred by the token-budget scheduler cannot be immediately preempted before receiving its next decode opportunity.

The victim is removed from active state and appended to the waiting tail. With a finite workload, every resumed request gets at least one decode opportunity before it can be preempted again.

## Resume admission

A `PREEMPTED` request uses the same full-lifetime reservation size as its original admission. It is re-admitted only when that reservation fits.

Resume does not attach APC prompt blocks. The logical history may already be inside a learned-position sliding window whose positions differ from the original prompt positions, so blindly reusing original APC blocks would be incorrect.

After reservation, the request enters `RECOMPUTING`; it does not run hidden model work inside admission.

## Cache recomputation

For a decoding request, the newest generated token is not yet represented in KV. Therefore the cache to restore is exactly:

```text
history = state.all_tokens[:-1][-block_size:]
```

Recompute:

- performs dense model prefill over `history`;
- produces KV only;
- does not sample;
- does not advance the request generator;
- does not emit a token;
- writes the rebuilt KV into the fresh private paged table;
- returns the request to `DECODING`.

The recompute work cost is `len(history)` model tokens. Stage 16 already requires the scheduled budget to fit one full learned-position window, so one recomputation can always be scheduled when it reaches the front of the model-work order.

## Token-budget integration

Stage 17 extends Stage 16's actual model-work accounting:

```text
normal paged decode = 1
overflow dense rebuild = actual rebuild tokens
preemption recompute = actual history tokens
prefill chunk = actual chunk tokens
exact APC hit = 0
```

`DECODING` and `RECOMPUTING` requests share one deterministic active-order selector. Consecutive one-token normal decode rows may still batch. Overflow and recompute operations execute individually in the selected order. No model work may exceed `max_scheduled_tokens`.

The existing decode/prefill fairness cursor remains authoritative when model work cannot fit together in one tick.

## Events and metrics

New engine events:

```text
PREEMPTED
RECOMPUTE_STARTED
RESUMED
```

Request metrics add:

```text
preemption_count
recompute_tokens
```

Engine metrics add totals for the same values.

Recompute is also recorded as a `PrefillBatchObservation` with a dedicated execution mode so evidence can account actual model-token work.

## Correctness contracts

Stage 17 must prove:

1. Tiny-pool pressure preempts a decoding victim and lets the blocked FIFO head become resident.
2. A preempted request releases private blocks, shared refs, reservation, and retained logits immediately.
3. Recompute restores `all_tokens[:-1][-block_size:]` exactly.
4. Recompute does not advance request RNG.
5. Generated tokens match an equivalent roomy-pool no-preemption reference for deterministic finite workloads.
6. Learned-position overflow remains correct after preempt/resume.
7. Cancelling a `PREEMPTED` request is leak-free and does not recompute it.
8. Recompute failure is isolated and releases all request resources.
9. APC-active victims release shared refs; resume uses private recompute rather than unsafe original-position prefix reuse.
10. Finite mixed workloads make progress without starvation.

## Stress evidence

Run deterministic mixed pressure operations with invariant verification after every mutation:

```text
submit
tick
preempt
resume/recompute
cancel
finish
failure
```

Final state must have:

- no active private ownership;
- no active shared references;
- no reservations;
- no terminal request retaining intermediate prefill logits;
- all finite non-cancelled requests terminal.

## Benchmark / claim policy

Stage 17 benchmark evidence is `descriptive_only`.

Report structural metrics such as:

- preemptions;
- resumes;
- recompute tokens;
- waiting-time distribution;
- request completion order;
- peak resident/protected blocks.

Do not claim a throughput or latency improvement unless a later dedicated fresh-process benchmark establishes one under a strict comparison policy.

## Scope boundaries

Stage 17 does not implement:

- dynamic/lazy KV reservation overcommit;
- CPU swap/offload;
- partial-block sharing or COW;
- speculative decoding;
- GPU/CUDA;
- fused attention kernels;
- new HTTP request API;
- scheduler priorities or user-configurable victim policies.

The single deterministic victim policy is intentional for this stage.
