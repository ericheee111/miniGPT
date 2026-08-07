# Stage 11A — Decode Continuous Batching

## Scope

Stage 11A retains the Stage 10 request state machine, strict FIFO scheduler, admission control,
cache reservations, event ordering, and request/engine metric definitions. It adds a second
executor that batches only the single-token decode work already selected by one engine tick.

Prefill remains per request. HTTP serving, batched prefill, paged attention, BPE, GPU and
distributed execution, new model structures, and training/checkpoint changes remain out of scope.

## Executor contract

Both executors implement the existing `prefill(requests)` contract. The reference executor also
keeps per-request decode. The continuous executor exposes the explicit model-work seam:

```text
decode_batch(requests: Sequence[RequestState]) -> Sequence[ExecutionResult]
```

The engine continues to call `decode()` through the shared executor protocol. For the continuous
executor, `decode()` validates every request independently, sends valid non-overflow requests to
one `decode_batch()` call, handles learned-position overflow through the Stage 9 re-prefill
fallback, and returns one result for every input request in original engine order.

## Variable-length dense KV representation

Each caller-owned request cache remains compact and has layer tensors shaped:

```text
[1, n_head, cache_length, head_size]
```

For a decode batch of size `B`, assembly computes `P = max(cache_length)` and creates one dense
cache per layer:

```text
key/value:       [B, n_head, P, head_size]
cache_lengths:   [B]
validity mask:   [B, 1, 1, P + 1]
position offset: cache_lengths[row]
token ids:       [B, 1]
```

Historical K/V is copied into the left-aligned prefix `[0:cache_length]`; the remainder is zero
padding. The mask allows exactly those prefix positions plus the new-token column at index `P`.
Learned absolute position embeddings use each row's real `cache_length`, never `P`.

The model projects all `B` new tokens together and appends their K/V at dense column `P`. This
temporary output has shape `[B, n_head, P + 1, head_size]`. Scatter builds each next request cache
from its valid historical prefix and the new-token column, producing compact tensors of length
`cache_length + 1`. Neither the caller's old tensor nor padding storage is returned.

## Validation and failure isolation

Before assembly, each request is checked for a generated token, layer count, batch size one,
head/head-size dimensions, key/value shape equality, common layer length, real cached-length
agreement, model dtype/device, detached tensors, and remaining learned-position capacity.

An invalid request becomes an isolated failed `ExecutionResult` and does not enter the tensor
batch. Valid peers continue. An exception from the actual assembled model call is a batch-level
failure for only the requests that entered that call; every such request receives complete failure
evidence. Overflow is not corruption: a full cache takes the existing sliding-window re-prefill
path independently.

## Sampling and deterministic order

The batched model call returns logits only. Sampling occurs after scatter, one request at a time,
with the `torch.Generator` already owned by that request. There is no global RNG use. Results are
indexed by request identity and returned in input order, so permuting batch rows, adding or
cancelling peers, and isolating a bad peer cannot change a surviving request's RNG stream.

## Timing and metrics

Each decode tick records an executor observation with batch size, useful cache tokens, padded cache
tokens, padding waste, assembly time, model execution time, scatter time, and total executor time.
Engine aggregates expose mean/max batch size, total useful/padded cache tokens, padding waste ratio,
and the existing request/token throughput, TTFT, TPOT, and E2E definitions.

Control-plane logical timing stays separate from canonical wall-clock benchmark timing. Profiler
output remains descriptive and cannot be substituted for benchmark throughput.

## Simulator equivalence

The simulator accepts `executor: reference` or `executor: continuous_decode`. A comparison entry
point runs both with identical deterministic model weights, requests, seeds, logical ticks, and
per-operation clock policy. It verifies tokens, terminal states, cancellation, admission order,
logical event semantics, and request metrics. Executor-specific batch telemetry and wall-clock
timings are compared separately.

## Benchmark and evidence policy

The fresh-process serving benchmark covers burst sizes 2/4/8, staggered arrivals, mixed
prompt/cache lengths, and cancellation. Deterministic weights and greedy/forced-token workloads
remove sampling noise. Raw replicates bind environment, resolved config, execution order, and
SHA-256 identities. Reports include median, MAD, CV, throughput, TTFT, TPOT, E2E, batch utilization,
and padding waste.

A performance claim is allowed only when the strict comparison policy passes. Otherwise the
verdict is `not_comparable`, with the failed stability or identity precondition recorded. Dense
padding may improve throughput while increasing memory traffic or single-request latency; this
stage makes no unconditional speedup claim and is not paged attention.
