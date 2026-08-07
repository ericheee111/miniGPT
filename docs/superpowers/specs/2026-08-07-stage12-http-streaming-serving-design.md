# Stage 12 — OpenAI-Compatible HTTP Serving and Streaming

## Scope

Stage 12 exposes the existing Stage 10 FIFO scheduler and Stage 11 executors through an optional
HTTP boundary. It adds `GET /healthz`, `GET /v1/models`, and an OpenAI-compatible subset of
`POST /v1/completions`. It does not add chat completions, BPE, PagedAttention, GPU/distributed
execution, a new model, or a second admission scheduler.

The accepted completion fields are exactly `model`, `prompt`, `max_tokens`, `temperature`,
`stream`, and `seed`. Unknown fields are rejected. The only advertised model is
`minigpt-char`. A prompt is encoded by the persisted character tokenizer and must be non-empty,
representable by that vocabulary, and no longer than the model `block_size`.

## Ownership and command flow

The checkpoint, tokenizer, resolved experiment config, and CPU model are loaded once at process
startup. The HTTP layer may encode and decode text, but it never calls the model or
`ServingEngine`. One dedicated worker thread owns every call to the engine and its executor:

```text
async HTTP handler
    -> bounded thread-safe command queue
    -> dedicated EngineRunner worker
    -> ServingEngine.submit/cancel/tick/metrics
    -> per-request result or stream channel
    -> async HTTP response
```

Commands are explicit: submit, cancel, metrics snapshot, and shutdown. While work is live, the
worker drains pending commands between ticks, executes one tick, and publishes the newly appended
engine events. When the engine is idle it blocks on the command queue; it does not poll or
busy-spin. Each request has a `concurrent.futures.Future` for its terminal result. Streaming
requests additionally own an independent bounded queue.

The runner command queue protects the service boundary only. Accepted generation requests still
enter the existing `ServingEngine.submit()` FIFO waiting queue, and all active-request and
worst-case KV-cache reservations remain Stage 10 policy.

## Completion and error contracts

Non-streaming success uses `object: "text_completion"`, a stable completion ID and creation
timestamp, the requested model, one choice, `finish_reason: "length"`, and prompt/completion/total
token usage. The character tokenizer has no EOS token, so a successful request finishes only when
`max_tokens` is reached, including the zero-token case.

Errors use one stable envelope:

```json
{"error":{"message":"...","type":"...","param":null,"code":"..."}}
```

Malformed JSON or schema violations map to 400/422, invalid or overlong prompts to 400, an unknown
model to 404, a saturated/stopping service to 429/503, and isolated executor failures to 500.
Unsupported OpenAI parameters are schema errors rather than ignored input.

## SSE streaming

`stream=true` returns `text/event-stream`. Every generated character is published as one JSON SSE
chunk as soon as the corresponding engine token event is observed. Token chunks have
`finish_reason: null`; a final empty-text chunk carries `finish_reason: "length"` and exact usage,
then normal completion sends `data: [DONE]`. Concatenating token-chunk text is therefore identical
to the non-streaming `choices[0].text` for the same prompt and seed.

The async generator is the consumer. Its `finally` path requests cancellation whenever the client
disconnects or closes the stream before normal completion. The worker applies that cancellation
at the next tick's first phase, so waiting or active state becomes `CANCELLED` and any KV-cache
reservation is released by the existing engine path. Cancellation, failure, shutdown, and
backpressure never emit `[DONE]`.

## Backpressure and failure isolation

Every streaming request uses a configurable bounded token queue. Producer writes are non-blocking,
so a slow client cannot stop another request's decode. If the queue is full, the policy is
deterministic: record a backpressure event, request engine cancellation, replace one buffered token
with a terminal stream failure signal, and never emit `[DONE]`. Memory is bounded by
`stream_buffer_size` per stream plus the bounded command queue.

Runner lifecycle and backpressure events are append-only and separate from Stage 10 request events.
An executor failure completes only the matching channel with an internal-generation error; other
requests remain live. Graceful shutdown stops admission, cancels all waiting and active requests,
ticks until their cancellation is applied, completes every channel, releases reservations, and
joins the worker.

## Testing and benchmark policy

Most API tests use ASGI in process with a tiny deterministic model/checkpoint/tokenizer fixture.
Runner tests directly cover ownership, independent RNG, FIFO/cache invariants, waiting and active
cancellation, backpressure, failure isolation, and shutdown. A small localhost subprocess smoke
starts Uvicorn, probes health/completion, and verifies termination without using a public port.

`benchmark_server.py` is a separate end-to-end HTTP workload, not an alias for the Stage 11
executor benchmark. It measures API validation, JSON/SSE serialization, cross-thread queues,
scheduler wait, and engine execution at concurrency 1/2/4/8 for short and mixed prompts in stream
and non-stream modes. It reports requests/s, generated tokens/s, TTFT, TPOT, E2E P50/P95/P99,
HTTP errors, cancellations, peak active requests, and observed prefill/decode batch sizes. Shared CI
results are correctness smoke evidence, not performance truth, and no speedup is assumed.
