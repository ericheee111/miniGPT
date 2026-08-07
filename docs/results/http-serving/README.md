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
| 1 | short | False | 43.284141 | 346.273127 | 0.022216/0.029315/0.030359 | n/a/n/a/n/a | 0.022216/0.029315/0.030359 | 0 |
| 1 | short | True | 31.827368 | 254.618947 | 0.005791/0.006035/0.006045 | 0.003501/0.003734/0.003737 | 0.031475/0.032984/0.033179 | 0 |
| 1 | mixed | False | 37.912536 | 303.300286 | 0.027616/0.028129/0.028152 | n/a/n/a/n/a | 0.027616/0.028129/0.028152 | 0 |
| 1 | mixed | True | 32.099212 | 256.793698 | 0.006084/0.007613/0.008038 | 0.003339/0.003852/0.003901 | 0.030390/0.035000/0.035597 | 0 |
| 2 | short | False | 55.605601 | 444.844805 | 0.035519/0.039730/0.040469 | n/a/n/a/n/a | 0.035519/0.039730/0.040469 | 0 |
| 2 | short | True | 37.551158 | 300.409261 | 0.016173/0.018313/0.018677 | 0.004997/0.005270/0.005293 | 0.052384/0.055273/0.055296 | 0 |
| 2 | mixed | False | 48.652390 | 389.219117 | 0.041562/0.046541/0.047135 | n/a/n/a/n/a | 0.041562/0.046541/0.047135 | 0 |
| 2 | mixed | True | 39.057522 | 312.460180 | 0.014672/0.018987/0.019889 | 0.004722/0.005226/0.005241 | 0.049071/0.054550/0.054964 | 0 |
| 4 | short | False | 78.166831 | 625.334652 | 0.045250/0.055539/0.055630 | n/a/n/a/n/a | 0.045250/0.055539/0.055630 | 0 |
| 4 | short | True | 56.411681 | 451.293449 | 0.024102/0.024751/0.024792 | 0.006188/0.006515/0.006580 | 0.069933/0.073552/0.073887 | 0 |
| 4 | mixed | False | 80.628824 | 645.030593 | 0.048796/0.052722/0.052923 | n/a/n/a/n/a | 0.048796/0.052722/0.052923 | 0 |
| 4 | mixed | True | 57.695581 | 461.564647 | 0.022536/0.025898/0.026062 | 0.005824/0.006789/0.006798 | 0.063589/0.078276/0.078337 | 0 |
| 8 | short | False | 89.046303 | 712.370423 | 0.084367/0.087166/0.087333 | n/a/n/a/n/a | 0.084367/0.087166/0.087333 | 0 |
| 8 | short | True | 64.360315 | 514.882518 | 0.047277/0.049157/0.049637 | 0.009668/0.009876/0.009879 | 0.119608/0.122077/0.122528 | 0 |
| 8 | mixed | False | 69.272646 | 554.181167 | 0.107031/0.112569/0.112819 | n/a/n/a/n/a | 0.107031/0.112569/0.112819 | 0 |
| 8 | mixed | True | 58.556237 | 468.449899 | 0.049590/0.066267/0.066532 | 0.010035/0.010723/0.010906 | 0.125433/0.134413/0.134508 | 0 |

TTFT is time to the first observable token chunk for streaming. For non-streaming,
the first observable completion payload arrives at E2E, so TTFT equals E2E and TPOT
is not observable. TPOT is the mean interval between streamed output tokens. P95/P99
show tail latency and must be interpreted with request count and machine context.

Canonical matrix: 16 cases, 128 requests, 0 HTTP errors, 0 cancellations, peak active 8, average prefill batch 1.2549019607843137, and average decode batch 1.8589211618257262.

The lifecycle evidence separately covers active/waiting cancellation, real localhost
disconnect, bounded-buffer backpressure, failure isolation, graceful shutdown, and KV
reservation release. Historical Stage evidence was not modified.
