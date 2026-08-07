# Stage 12 — OpenAI-Compatible HTTP Serving and Streaming

## Outcome

Stage 12 exposes the existing Stage 10 FIFO scheduler and Stage 11 executors through
`GET /healthz`, `GET /v1/models`, and an OpenAI-compatible subset of
`POST /v1/completions`. Checkpoint, tokenizer, config, and model load once at startup.
The async HTTP layer never calls the model: it submits commands to one dedicated
`EngineRunner` thread, which is the only owner that calls `ServingEngine` and its model.

The subset accepts only `model`, `prompt`, `max_tokens`, `temperature`, `stream`, and
`seed`; unsupported OpenAI fields are rejected. Chat Completions, BPE, PagedAttention,
GPU, distributed serving, and new model structures remain out of scope.

## Ordinary completion

```console
curl -X POST http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"minigpt-char","prompt":"ROMEO:","max_tokens":8,"temperature":1.0,"stream":false,"seed":42}'
```

```json
{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "text": "\nAstloul"
    }
  ],
  "created": 1786081905,
  "id": "cmpl-d3e59a01362f4ca29ee9c4741a1fd90a",
  "model": "minigpt-char",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 8,
    "prompt_tokens": 6,
    "total_tokens": 14
  }
}
```

## Streaming SSE

```console
curl -N -X POST http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"minigpt-char","prompt":"ROMEO:","max_tokens":8,"temperature":1.0,"stream":true,"seed":42}'
```

```text
data: {"id":"cmpl-c5a836f7291f41029b74c3d366a6e5bb","object":"text_completion","created":1786081905,"model":"minigpt-char","choices":[{"text":"\n","index":0,"finish_reason":null}]}
data: {"id":"cmpl-c5a836f7291f41029b74c3d366a6e5bb","object":"text_completion","created":1786081905,"model":"minigpt-char","choices":[{"text":"A","index":0,"finish_reason":null}]}
data: {"id":"cmpl-c5a836f7291f41029b74c3d366a6e5bb","object":"text_completion","created":1786081905,"model":"minigpt-char","choices":[{"text":"s","index":0,"finish_reason":null}]}
data: {"id":"cmpl-c5a836f7291f41029b74c3d366a6e5bb","object":"text_completion","created":1786081905,"model":"minigpt-char","choices":[{"text":"t","index":0,"finish_reason":null}]}
data: {"id":"cmpl-c5a836f7291f41029b74c3d366a6e5bb","object":"text_completion","created":1786081905,"model":"minigpt-char","choices":[{"text":"l","index":0,"finish_reason":null}]}
data: {"id":"cmpl-c5a836f7291f41029b74c3d366a6e5bb","object":"text_completion","created":1786081905,"model":"minigpt-char","choices":[{"text":"o","index":0,"finish_reason":null}]}
data: {"id":"cmpl-c5a836f7291f41029b74c3d366a6e5bb","object":"text_completion","created":1786081905,"model":"minigpt-char","choices":[{"text":"u","index":0,"finish_reason":null}]}
data: {"id":"cmpl-c5a836f7291f41029b74c3d366a6e5bb","object":"text_completion","created":1786081905,"model":"minigpt-char","choices":[{"text":"l","index":0,"finish_reason":null}]}
data: {"id":"cmpl-c5a836f7291f41029b74c3d366a6e5bb","object":"text_completion","created":1786081905,"model":"minigpt-char","choices":[{"text":"","index":0,"finish_reason":"length"}],"usage":{"prompt_tokens":6,"completion_tokens":8,"total_tokens":14}}
data: [DONE]
```

Each token chunk is produced independently. The final JSON chunk carries `length` and
usage, followed by `[DONE]` only on normal completion. Concatenated token chunks equal
the non-stream result for the same seed. Client disconnect requests engine cancellation;
Stage 10 cancellation releases the KV reservation. A bounded per-stream queue prevents a
slow client from blocking peers; overflow records backpressure, cancels that request, and
does not emit `[DONE]`.

## Concurrent workload

Concurrent example status codes: `[200, 200, 200, 200]`;
independent fixed-seed outputs: `["\nGo's ma", "\nGo's ma", "\nThat th", "\nThatt a"]`.

HTTP concurrency is the number of requests in flight at the API boundary. Continuous
batching is the executor's tensor-level grouping of eligible prefill/decode work after
FIFO admission. The former creates opportunity; it does not guarantee the latter's batch
size.

## End-to-end HTTP benchmark

This benchmark includes HTTP validation, JSON/SSE serialization, async-to-thread queues,
the scheduler, and engine execution. It is separate from the Stage 11 executor benchmark,
which isolates executor strategies without HTTP system overhead. These localhost numbers
are descriptive for this machine and are not a speedup claim or shared-CI performance truth.

| Concurrency | Prompts | Stream | req/s | tokens/s | TTFT P50/P95/P99 | TPOT P50/P95/P99 | E2E P50/P95/P99 | Errors |
|---:|---|---|---:|---:|---|---|---|---:|
| 1 | short | False | 39.588399 | 316.707195 | 0.023428/0.032686/0.034636 | n/a/n/a/n/a | 0.023428/0.032686/0.034636 | 0 |
| 1 | short | True | 31.674249 | 253.393995 | 0.005948/0.006384/0.006390 | 0.003492/0.003730/0.003757 | 0.031552/0.033330/0.033672 | 0 |
| 1 | mixed | False | 43.944768 | 351.558144 | 0.021789/0.028373/0.028474 | n/a/n/a/n/a | 0.021789/0.028373/0.028474 | 0 |
| 1 | mixed | True | 33.688508 | 269.508068 | 0.006338/0.006506/0.006517 | 0.003118/0.003642/0.003745 | 0.029594/0.032761/0.033789 | 0 |
| 2 | short | False | 55.886723 | 447.093786 | 0.034987/0.038781/0.039516 | n/a/n/a/n/a | 0.034987/0.038781/0.039516 | 0 |
| 2 | short | True | 40.328963 | 322.631707 | 0.014491/0.017862/0.018831 | 0.004593/0.004954/0.005057 | 0.048688/0.052375/0.053267 | 0 |
| 2 | mixed | False | 52.968802 | 423.750416 | 0.036013/0.045950/0.047960 | n/a/n/a/n/a | 0.036013/0.045950/0.047960 | 0 |
| 2 | mixed | True | 41.417643 | 331.341145 | 0.014040/0.016832/0.016835 | 0.004530/0.005208/0.005247 | 0.047607/0.053019/0.054699 | 0 |
| 4 | short | False | 67.634745 | 541.077963 | 0.057133/0.063574/0.063653 | n/a/n/a/n/a | 0.057133/0.063574/0.063653 | 0 |
| 4 | short | True | 53.368913 | 426.951301 | 0.025758/0.030646/0.030984 | 0.006178/0.006776/0.006793 | 0.067000/0.084822/0.085144 | 0 |
| 4 | mixed | False | 65.854245 | 526.833959 | 0.054568/0.064291/0.064329 | n/a/n/a/n/a | 0.054568/0.064291/0.064329 | 0 |
| 4 | mixed | True | 52.692210 | 421.537677 | 0.023956/0.027317/0.027420 | 0.006753/0.007125/0.007176 | 0.071592/0.080733/0.080768 | 0 |
| 8 | short | False | 77.732315 | 621.858521 | 0.093388/0.100921/0.101157 | n/a/n/a/n/a | 0.093388/0.100921/0.101157 | 0 |
| 8 | short | True | 53.768572 | 430.148576 | 0.054863/0.068100/0.068958 | 0.010904/0.011218/0.011222 | 0.137092/0.145067/0.145180 | 0 |
| 8 | mixed | False | 75.976873 | 607.814982 | 0.091761/0.100791/0.101010 | n/a/n/a/n/a | 0.091761/0.100791/0.101010 | 0 |
| 8 | mixed | True | 53.769945 | 430.159562 | 0.049922/0.064274/0.065297 | 0.011314/0.011764/0.011794 | 0.137596/0.145234/0.145765 | 0 |

TTFT is time to the first observable token chunk for streaming. For non-streaming,
the first observable completion payload arrives at E2E, so TTFT equals E2E and TPOT
is not observable. TPOT is the mean interval between streamed output tokens. P95/P99
show tail latency and must be interpreted with request count and machine context.

Canonical matrix: 16 cases, 128 requests, 0 HTTP errors, 0 cancellations, peak active 8, average prefill batch 1.2307692307692308, and average decode batch 1.8436213991769548.

The lifecycle evidence separately covers active/waiting cancellation, real localhost
disconnect, bounded-buffer backpressure, failure isolation, graceful shutdown, and KV
reservation release. Historical Stage evidence was not modified.
