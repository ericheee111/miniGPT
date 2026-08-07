# Stage 11B — Length-Bucketed Batched Prefill

## Outcome

Stage 11A batches one decode token per eligible request after each prompt has already
been processed separately. Stage 11B keeps that decode path and additionally groups the
currently eligible FIFO prompt prefix into tensor-level prefill batches. It never waits
for future requests and never skips the FIFO head to improve utilization.

Prompt rows are right-padded to `[B, Tmax]`; valid learned absolute positions remain
`0..L-1`. A per-row causal/key-valid mask prevents valid queries from reading padding.
Final logits are gathered at `L-1`. Dense layer caches are scattered back to compact
`[1, H, L, D]` tensors, so padding does not enter caller-owned request state.

Prompt padding is more expensive than decode cache padding: prefill computes projection,
attention, MLP, and residual work for every padded query position in every layer, whereas
Stage 11A adds only one new decode query per row. Length bucketing bounds that repeated
work with batch-size, padded-token-budget, and padding-ratio limits.

This remains dense batching, not paged attention. Throughput can improve while TTFT for
some requests worsens, so the canonical report includes both rather than treating
throughput alone as success.

Overall strict benchmark verdict: `not_comparable`.

| Scenario | Verdict | Prefill batch | Prompt waste | Decode TTFT | Full TTFT | Decode req/s | Full req/s |
|---|---|---:|---:|---:|---:|---:|---:|
| burst-equal-length | not_comparable | 8.0 | 0.0 | 0.0009190500713504543 | 0.0020381999201012162 | 546.8814052221524 | 843.4103150259699 |
| burst-mixed-lengths | not_comparable | 4.0 | 0.22807017543859648 | 0.0007223999126063842 | 0.0013549497816848545 | 684.7087405589533 | 919.9737834391998 |
| short-prompt-heavy | not_comparable | 4.0 | 0.125 | 0.0007766500347481089 | 0.0011707501249072127 | 615.3325522227634 | 956.000104281597 |
| long-prompt-heavy | not_comparable | 2.4 | 0.125 | 0.0007655998928040952 | 0.002093150047950426 | 625.7773374395761 | 943.4184943777349 |
| staggered-arrivals | not_comparable | 1.6 | 0.0967741935483871 | 0.00311489996982332 | 0.0026552999624154037 | 516.0757601980748 | 592.4960420152557 |
| high-padding-pressure | not_comparable | 1.0 | 0.0 | 0.0007292499069687819 | 0.0008084500442048954 | 741.2967062659336 | 693.355060668828 |

Every canonical scenario is `not_comparable` because at least one primary executor
exceeded the 10% CV threshold. Therefore the medians above are descriptive only and
do not establish a performance improvement. Equal, mixed, short-heavy, and long-heavy
bursts show higher full-continuous request/s medians, but burst TTFT is generally
higher. Staggered arrivals form smaller batches. High padding pressure is split into
size-one prefills by the FIFO policy and shows no descriptive throughput gain.

Canonical timings use alternating fresh processes and `time.perf_counter`; raw
replicates are unfiltered. Median, MAD, CV, queue/prefill/TTFT/E2E timing,
request and token throughput, worker peak RSS, utilization, environment, order,
and hashes are included. Profiler output is descriptive only and is not canonical
benchmark data.

Simulator evidence runs `reference`, `continuous_decode`, and `continuous` from the
same weights, workloads, seeds, scheduler, and cache budget. It verifies tokens,
terminal/cancellation states, FIFO admission, cache accounting, logical request
events, and request metrics. Executor-specific batch events are reported separately.

There is no HTTP layer, paged cache, BPE, GPU/distributed path, or model upgrade.
