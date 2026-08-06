# Stage 10 — MiniServe Request Scheduling and Engine Control Plane

## Outcome

This package demonstrates deterministic request scheduling, lifecycle transitions, cache
reservation/release, cancellation, backpressure, failure isolation, and metric
accounting.
It does not claim a throughput improvement.

The executor remains explicitly per-request: each engine iteration may advance multiple
request state machines, but every model call is still a separate `GPT.prefill()` or
`GPT.decode()` invocation. Iteration-level scheduling is not tensor-level continuous
batching; that executor change is reserved for Stage 11.

## Why online serving needs scheduling

Independent arrivals compete for active slots and KV-cache capacity. Strict FIFO
admission provides deterministic fairness, while backpressure keeps requests waiting
until both an active slot and their worst-case cache reservation are available.
Completion, cancellation, or failure releases the reservation immediately.

Prefill and decode are interleaved by engine tick. Newly admitted requests remain
`PREFILLING` until the next tick, so cancellation can be applied before model execution.
Each decoding request emits at most one token per tick. Learned-position overflow
preserves the Stage 9 sliding-window re-prefill fallback.

## Metric definitions

Queue time is admission minus arrival. TTFT is first token minus arrival. Prefill latency
measures the initial prompt call. Per-token decode latency records each later token call,
and TPOT is their mean. E2E latency is terminal time minus arrival. Request and token
throughput are descriptive logical-workload accounting, not benchmark performance.

## Fixed scenarios

| Scenario | Requests | Completed | Cancelled | Failed | Peak active | Peak cache |
|---|---:|---:|---:|---:|---:|---:|
| single-request | 1 | 1 | 0 | 0 | 1 | 7 |
| burst-arrivals | 4 | 4 | 0 | 0 | 2 | 11 |
| cache-pressure-cancellation | 3 | 2 | 1 | 0 | 1 | 6 |

Every scenario directory contains the exact workload plus `events.jsonl`,
`requests.csv`, `summary.json`, and `timeline.md`. `artifact_manifest.json` binds all
committed artifacts by byte count and SHA-256.

## Known limitations

There is no HTTP layer, tensor batch assembly, paged cache, BPE, GPU path,
distributed execution, or production admission policy. Logical clocks prove metric
formulas and event determinism; they are not wall-clock performance measurements.
