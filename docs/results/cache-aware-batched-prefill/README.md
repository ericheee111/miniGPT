# Stage 15 — Cache-Aware Batched Paged Prefill

Stage 15 removes Stage 14's sequential APC suffix-prefill regression. Cached-prefix
Transformer work is still skipped, while multiple variable-length suffix segments execute
in one batched paged-history model call. Historical K/V remains in physical paged blocks
and is never dense-materialized on the normal path.

Batch admission uses computed suffix tokens for max batch size, token budget, and padding
ratio. Exact full-boundary hits use cached boundary logits and perform zero model prefill.
All valid suffix logits and suffix-only K/V deltas are scattered through the existing
owner-thread write, promotion, duplicate canonicalization, and refcount transactions.

This remains a Python/PyTorch reference implementation. There is no chunked prefill,
partial-block sharing or copy-on-write, KV-pressure preemption, speculative decoding,
GPU/CUDA path, custom fused kernel, or new HTTP API.

## Fresh-process benchmark

| Workload | Sequential calls | Batched calls | Avg batch | Max batch | Verdict |
|---|---:|---:|---:|---:|---|
| concurrent_same_prefix | 4 | 1 | 4.0 | 4 | pass |
| exact_hits_mixed_with_suffix_hits | 4 | 2 | 2.0 | 3 | pass |
| low_reuse_random_prompts | 6 | 1 | 6.0 | 6 | pass |
| mixed_prefix_lengths | 6 | 2 | 3.0 | 3 | pass |
| padding_pressure | 5 | 3 | 1.6666666666666667 | 2 | fail |
| repeated_prefix_short_suffix | 5 | 2 | 2.5 | 4 | pass |
| repeated_prefix_variable_suffix | 4 | 2 | 2.0 | 3 | fail |

Every raw timing sample runs in a fresh Python process and retains median, MAD, and CV
for TTFT, E2E, req/s, tokens/s, and peak RSS. Avoided prefix work and model-call
reduction are reported independently from wall-clock timing.

Aggregate strict verdict: `fail`.
Wall-clock performance improvement claim: `False`.
When the aggregate verdict is not `pass`, this package makes no wall-clock performance improvement claim.
