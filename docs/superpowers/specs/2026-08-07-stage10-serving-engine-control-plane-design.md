# Stage 10 — MiniServe Request Scheduling and Engine Control Plane

## Scope

Stage 10 adds a deterministic multi-request serving control plane around the Stage 9
`GPT.prefill()` and `GPT.decode()` APIs. It does not add tensor-level continuous batching. The
reference executor deliberately invokes the model once per request so scheduler semantics can be
tested without claiming a throughput optimization.

Out of scope: HTTP, BPE, GPU, distributed execution, `torch.compile`, new attention kernels, model
changes, and training/checkpoint changes. Stage 11 is the first stage that may replace the reference
executor with true tensor-level continuous batching.

## Request state machine

```text
WAITING --admit--> PREFILLING --first token--> DECODING --limit reached--> FINISHED
   |                    |                         |
   +--cancel----------> CANCELLED <--------------+
                        |
           execution/validation failure
                        v
                      FAILED
```

`max_new_tokens=0` follows `WAITING -> FINISHED` during admission and never allocates a KV cache.
A cancellation request is idempotent. Terminal requests never transition again.

Each request owns its prompt, sampling configuration, seed-derived `torch.Generator`, generated
tokens, KV cache, lifecycle timestamps, decode latencies, and failure reason. Sampling never uses
the global PyTorch RNG.

## Scheduler and cache capacity

The scheduler is strict FIFO. It admits from the queue head while both limits allow:

1. `max_active_requests`;
2. `max_cached_tokens`.

Admission reserves the maximum cache length that the request can reach:

```text
min(block_size, min(prompt_length, block_size) + max(max_new_tokens - 1, 0))
```

The final sampled token is not inserted into a cache unless another decode step needs it. Reserving
the maximum reachable cache length prevents an admitted request from exceeding capacity later.
Actual occupancy is reported separately from reservation. A request whose reservation exceeds the
entire budget fails deterministically so it cannot starve every later request. A temporarily blocked
head remains queued and is never bypassed. Finishing, cancellation, and failure immediately release
the reservation and cache.

## Engine tick

Every tick has a stable order:

1. apply pending cancellations;
2. admit FIFO requests;
3. prefill requests that were already admitted before this tick;
4. decode each request that was already decoding at tick start, at most once;
5. publish occupancy and count metrics.

New admissions remain `PREFILLING` until the next tick. This makes cancellation between admission
and model execution observable and testable. A prefill samples at most the first token. Each decode
samples at most one later token. A failure result is isolated to its request and does not abort the
rest of the tick.

## Reference executor

The executor protocol exposes `prefill(requests)` and `decode(active_requests)`. The Stage 10
reference executor processes those sequences in order but calls the model separately for every
request. Iteration-level co-scheduling therefore means only that multiple request state machines
advance in one engine tick; it is not tensor-level batching.

When a request reaches the learned-position limit, the reference executor preserves the Stage 9
fallback: it rebuilds the cache by prefilling the latest `block_size` tokens, then continues
sampling. Per-request exceptions become failure results.

## Time, events, and metrics

Both engine and executor accept an injectable monotonic clock. Tests and the simulator use logical
or stepped clocks; production callers may use `time.perf_counter`.

The append-only event stream records submission, cancellation request/application, admission,
prefill, token production, finish, and failure with stable sequence numbers and scheduler snapshots.
Request metrics define:

- queue time: admission minus arrival;
- prefill latency: executor duration of the initial prompt pass;
- TTFT: first-token timestamp minus arrival;
- per-token decode latency: executor duration for every token after the first;
- TPOT: mean per-token decode latency, or zero when no decode token exists;
- end-to-end latency: terminal timestamp minus arrival.

Engine metrics include terminal counts, current/peak active and waiting counts, current/peak actual
cache occupancy, current/peak reserved cache tokens, completed request throughput, and generated
token throughput. Throughput is descriptive workload accounting only; Stage 10 makes no performance
improvement claim.

## Deterministic simulator and evidence

The offline simulator accepts strict JSON or YAML. Requests specify arrival time, prompt tokens or
prompt length, maximum new tokens, sampling options, and optional cancellation time. A fixed model
seed, independent request seeds, logical tick interval, and stepped executor clock make tokens and
events reproducible.

Every run writes `events.jsonl`, `requests.csv`, `summary.json`, and `timeline.md`. The committed
evidence package contains single-request, burst-arrival, and cache-pressure-with-cancellation
scenarios plus a SHA-256 manifest.

## Stage 11 seam

Stage 11 should retain the request state machine, scheduler, reservations, event schema, metrics,
and simulator. It may replace only the executor implementation with ragged/paged tensor batching,
explicit batch assembly/scatter, and model-side batched cache operations. Equivalence tests must run
the reference and continuous executors against identical requests and seeds.
