# Stage 16 — Chunked Prefill and Token-Budget Scheduling

## Scope

Stage 16 bounds prompt work per engine tick so long prompts no longer monopolize the serving loop. It
reuses Stage 15's `paged history + variable-length new token segment` model primitive and adds an
opt-in token-budget scheduler that can interleave decode work with prompt chunks.

This stage does not add KV-pressure preemption, partial-block sharing/copy-on-write, speculative
decoding, BPE, GPU/CUDA, custom fused kernels, `torch.compile`, quantization, distributed serving,
or a new HTTP request API. Learned-position overflow keeps the existing dense rebuild semantics.

## Configuration

`SchedulerConfig` gains two optional fields:

```text
max_scheduled_tokens
prefill_chunk_tokens
```

They are either both unset (legacy Stage 10–15 scheduling) or both positive (Stage 16 mode).
Stage 16 mode requires the paged backend. `prefill_chunk_tokens` must be a multiple of the physical
`block_tokens`. `max_scheduled_tokens` must be large enough for at least one full learned-position
window rebuild, because an overflow decode may execute dense fallback over that entire context.

The budget counts actual model-token work, not padding. A normal paged decode costs one token;
learned-position overflow costs the actual dense rebuild context length; a prompt chunk costs its
computed chunk length. Stage 15 `PrefillBatchConfig` still independently bounds tensor batch size,
padded tokens, and padding waste.

## Scheduling policy

Each Stage 16 tick applies cancellation and admission first, then schedules only requests that were
already live at the tick boundary. New admissions never wait for a future arrival to form a batch and
begin model work on a later tick, preserving the existing no-wait control-plane contract.

For each eligible request, model work is charged before execution:

1. normal paged decode costs `1`;
2. learned-position overflow costs the actual dense rebuild token count;
3. each PREFILLING request receives at most `prefill_chunk_tokens` prompt tokens;
4. exact full-prefix hits cost zero model tokens;
5. work that does not fit the remaining budget is deferred and is never executed over budget.

Scheduling follows deterministic FIFO order. When a high-cost decode blocks the remaining budget while
prefill is also live, a persistent fairness cursor gives the prefill side the next tick before resuming
the deferred decode. This bounded alternation preserves FIFO progress without starving either decode or
prefill. Consecutive normal decode rows may still batch together, and multiple selected chunks may use
one Stage 15 paged-history batch when padding limits allow.

## Chunk boundaries

An intermediate chunk must end on a physical `block_tokens` boundary. The final chunk may be shorter.
Admission prefix hits are already complete-block aligned, so this rule guarantees that every
non-final next chunk starts from a complete-block boundary.

This deliberately avoids partial-block COW. Already-computed full blocks remain request-private until
the final prompt chunk reuses the existing Automatic Prefix Cache promotion transaction.

## Chunk execution

For one selected row:

```text
start = current paged cache length
end   = start + selected chunk tokens
segment = prompt[start:end]
```

The executor calls Stage 15's `prefill_paged_batch` with the current request view as paged history.
It returns suffix-only logits and K/V delta. The owner thread transactionally appends that delta to the
request's paged table.

Intermediate chunks:

- do not sample;
- do not advance request RNG;
- do not emit a generated token;
- keep the request in `PREFILLING`;
- keep newly completed full prompt blocks private until the final prompt chunk promotion transaction.

The final chunk samples the first generated token from its final valid prompt logit and transitions to
normal decode exactly as Stage 15 does.

## Paged append and APC promotion

Stage 16 deliberately reuses the existing transactional `write_prefill_suffix()` path rather than
adding a second paged allocator API. Because every intermediate chunk ends on a physical block
boundary, the next chunk always starts at an append-safe complete-block boundary; the final chunk may
leave the existing PRIVATE partial tail.

When APC is enabled, intermediate chunk logits are retained only as request-local promotion evidence.
When APC is disabled, those logits are not retained at all. Full prompt blocks remain request-private
until the final prompt chunk. The final commit concatenates the computed suffix logits across chunks
and reuses the existing `promote_prompt_blocks` transaction once, with the semantic prefix inferred
from the original APC hit rather than from the current chunk cursor. This preserves Stage 14
canonicalization, duplicate release, rollback, and refcount semantics without introducing partial-block
sharing or a second promotion protocol.

Partial tails remain PRIVATE.

## Request lifecycle

A request may remain `PREFILLING` across multiple ticks. Its paged table length is the authoritative
prefill cursor. Cancellation, disconnect, backpressure failure, model failure, promotion failure, and
shutdown reuse the existing owner-thread cleanup path and must release active refs, private ownership,
reservations, and any retained intermediate `prefill_logits_chunks`. Terminal requests must not keep
promotion-only logits tensors reachable after cleanup.

The request's first-token timestamp is set only after the final prompt chunk. `prefill_tokens_computed`
is cumulative across chunks and still excludes APC-hit tokens.

## Observability

Stage 16 adds explicit chunk events and counters without changing generated-token events:

```text
PREFILL_CHUNK_STARTED
PREFILL_CHUNK_FINISHED
```

Each chunk records request ID, logical start/end, useful tokens, and whether it is final. Engine metrics
report total chunk batches, chunk count, and useful chunk tokens; the strict simulator config records
the configured scheduling budget/chunk cap. Existing Stage 15 prefill batch observations continue to
describe actual tensor batches and padding.

## Correctness and evidence

Tests/evidence must cover:

- one long prompt split into multiple chunks;
- mixed short/long prompts under one token budget;
- decode + prefill interleaving in the same workload;
- APC hit followed by chunked suffix prefill;
- exact APC hit with zero model work;
- block-aligned intermediate chunks and private final tail;
- identical generated tokens and per-request RNG state versus unchunked Stage 15;
- cancellation between chunks;
- model failure during an intermediate chunk;
- duplicate prefix promotion and eviction pressure;
- final zero active refs/private ownership/reservations;
- deterministic repeated simulation.

Performance evidence must separate structural scheduling claims from wall-clock claims. Stage 16 may
claim bounded useful prefill work per tick and observed decode/prefill interleaving regardless of timing.
Wall-clock TTFT/E2E/throughput improvement is claimed only when a fresh-process strict comparison
verdict is `pass`.
