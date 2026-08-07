# Stage 11A — Decode Continuous Batching

## Outcome

Stage 10 advanced multiple request state machines per engine iteration but invoked the
model separately for every request. Stage 11A retains that control plane and replaces
eligible decode execution with one tensor-level single-token batch per tick.
Prefill remains per request.

Caller caches stay compact `[1, H, L, D]`. Assembly creates right-padded dense
`[B, H, max(L), D]` layers plus a cache-valid mask and per-row learned-position offset.
Scatter takes each valid historical prefix and the new-token column, so padding never
enters returned caches and old caller tensors are not modified.

This is dense padded attention, not paged attention. Padding increases memory traffic;
mixed-cache benchmark observed the largest waste. Decode batching can improve aggregate
throughput while increasing a single request's TPOT or E2E latency, so both are reported.

Overall strict benchmark verdict: `pass`.

| Scenario | Verdict | Speedup | Continuous avg batch | Padding waste | Token/s |
|---|---|---:|---:|---:|---:|
| burst-2 | pass | 1.238892891700994 | 2.0 | 0.0 | 1886.0584946064685 |
| burst-4 | pass | 1.6960814870115735 | 4.0 | 0.0 | 2745.0375283880567 |
| burst-8 | pass | 2.2196051319955683 | 8.0 | 0.0 | 3489.4308266166336 |
| staggered-arrival | pass | 1.4353639872559376 | 2.8 | 0.09411764705882353 | 2273.663856587097 |
| mixed-cache-lengths | pass | 1.2550082791399895 | 2.3636363636363638 | 0.354251012145749 | 2028.3289883052444 |
| cancellation | pass | 1.2682428143476336 | 2.3333333333333335 | 0.0 | 1958.4341935382906 |

Canonical timings come from alternating fresh processes using `time.perf_counter`.
Raw replicates are unfiltered. Environment, resolved configuration, execution order,
median, MAD, CV, TTFT, TPOT, E2E, throughput, utilization, and hashes are included.
Profiler output is explicitly separate and is not used for throughput claims.

The simulator evidence runs reference and continuous executors from identical model
weights, workloads, and request seeds. It checks tokens, terminal/cancellation
states, FIFO admission order, complete logical events, and request metrics.

## Limits and Stage 11B

There is no HTTP layer, batched prefill, paged cache, BPE, GPU path, distributed
execution, or new model structure. Stage 11B should consider length-bucketed batched
prefill with explicit admission/TTFT guardrails; it should not reuse decode padding
without measuring prompt-side memory amplification.
