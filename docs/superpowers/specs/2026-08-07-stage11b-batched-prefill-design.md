# Stage 11B — Length-Bucketed Batched Prefill

## Scope

Stage 11B keeps the Stage 10 FIFO scheduler, admission and cache-reservation rules, request state
machine, cancellation ordering, logical request events, and request metrics. It also keeps both
Stage 11A executors unchanged in meaning:

- `reference`: per-request prefill and per-request decode;
- `continuous_decode`: per-request prefill and tensor-batched decode;
- `continuous`: length-bucketed tensor-batched prefill and tensor-batched decode.

Eligible requests are never delayed to wait for a larger prefill batch. HTTP, paged attention,
BPE, GPU/distributed execution, and model-structure changes remain out of scope.

## Model prefill representation

For `B` prompts with real lengths `L[0:B]`, assembly creates a right-padded token tensor with
`T = max(L)`:

```text
token_ids:       [B, T]                 torch.int64
prompt_lengths:  [B]                    torch.int64
allowed mask:    [B, 1, T, T]           boolean
layer key/value: [B, n_head, T, head_size]
final logits:    [B, 1, vocab_size]
```

Valid tokens remain left-aligned and keep learned absolute positions `0..L[row]-1`. The attention
mask is the normal causal mask intersected with valid key columns. Therefore a valid query can
never read padding K/V. The final row logits are gathered at `L[row]-1`.

The model returns a dense temporary cache. Executor scatter clones only `:L[row]` from each layer,
producing the existing per-request shape `[1, n_head, L[row], head_size]`. Padding storage and the
caller-owned token tuples never become part of a request cache and are never modified in place.

## FIFO length-aware batching

`PrefillBatchConfig` defines `max_batch_size`, `max_batch_tokens`, and `max_padding_ratio`. The
executor scans the already-eligible request sequence once in FIFO order. The current batch begins
with the queue head. Only the immediately following request may be appended. A candidate is
accepted when:

```text
candidate size <= max_batch_size
candidate size * candidate max length <= max_batch_tokens
(padded tokens - useful tokens) / padded tokens <= max_padding_ratio
```

When any constraint fails, the current batch closes and the rejected request becomes the head of
the next batch. No later request may be selected around it. A singleton is always executed even if
its prompt alone exceeds `max_batch_tokens`, preventing starvation while making the oversize cost
visible in telemetry.

## Validation and failure isolation

Every request is validated before grouping: prompt is non-empty, length is within `block_size`,
all token values are integers in the model vocabulary, and the request is in the expected prefill
state. Invalid requests receive isolated failed results and release reservations through the
existing engine path. Other valid requests continue into batches.

Model input validation additionally enforces rank-two token IDs, rank-one prompt lengths, common
batch/device, `torch.int64` dtype, valid vocabulary IDs, and lengths in `[1, T]`. An exception from
one actual model batch fails all and only the requests in that batch and records `batch_failed`.

## Events, metrics, and equivalence

Prefill batch telemetry is separate from the immutable request logical event stream. Every actual
prefill model call records `PREFILL_BATCH_STARTED` and `PREFILL_BATCH_FINISHED` with request IDs,
batch size, useful/padded prompt tokens, padding ratio, phase timings, and failure status. Reference
and continuous-decode record size-one calls; continuous records its real grouped calls.

Engine aggregates expose prefill batch sizes, useful/padded prompt tokens, prompt padding waste,
prefill executor/model/assembly-scatter time, and the existing decode and request latency metrics.
The three-executor simulator compares tokens, terminal/cancellation states, FIFO admission, cache
accounting, request logical events, and request metrics. Executor-specific prefill batch telemetry
and wall-clock timing are reported but not required to be structurally identical.

## Benchmark and evidence policy

The fresh-process benchmark primarily compares `continuous_decode` with `continuous`, isolating
the incremental value of prefill batching. It covers equal, mixed, short-heavy, long-heavy,
staggered, and high-padding-pressure prompts, while retaining `reference` as a whole-serving
baseline. Reports include raw replicates, environment, execution order, worker peak RSS,
median/MAD/CV, TTFT, queue/prefill/E2E timings, throughput, batch size, padding waste, and SHA-256.

Profiler timing remains descriptive and separate. Improvement may be claimed only when the strict
comparison passes. Dense prompt padding repeats full Transformer work at every padded position, so
its cost can be materially higher than decode cache padding. A throughput gain may coexist with
worse TTFT, and mixed-length workloads may correctly produce a `not_comparable` or no-improvement
result. This stage is dense batching, not paged attention.
