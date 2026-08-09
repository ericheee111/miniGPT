# Stage 14 — Automatic Prefix Caching

## Outcome

Stage 14 adds namespace-bound Automatic Prefix Caching on the Stage 13 paged KV pool.
Only complete prompt blocks become immutable SHARED blocks; incomplete tails remain
request-private. The chained SHA-256 identity binds namespace, every historical block,
the current full token block, and therefore its absolute logical context. Exact token
metadata is retained as a collision-defense invariant.

Longest-contiguous-prefix lookup attaches canonical physical IDs with active refcounts.
Suffix prefill begins at the true absolute prefix position and attends shared K/V plus
earlier suffix K/V without rerunning the cached Transformer prefix. Zero-ref shared blocks
remain resident and are evicted by deterministic LRU; active shared blocks are protected.

This stage has no partial-block sharing or copy-on-write, chunked prefill, speculative
decoding, GPU/CUDA path, custom fused PagedAttention kernel, or new HTTP API.
It remains a Python/PyTorch reference implementation.

## Correctness and lifecycle

Dense, materialized paged, direct paged, and direct paged + APC runs have identical
generated tokens, RNG state, terminal/cancellation states, FIFO admission, and logical
request events. Exact hits, common-prefix suffixes, partial tails, concurrent references,
learned-position overflow rebuild, cancellation, HTTP failure/disconnect, stream
backpressure, shutdown, duplicate-promotion rollback, and LRU pressure are covered.
The deterministic allocator/refcount stress executes 5,000 verified mutations.

## Fresh-process CPU benchmark

| Workload | Hit request ratio | Hit token ratio | Avoided prefill tokens | Evictions | Verdict |
|---|---:|---:|---:|---:|---|
| common_prefix_different_suffix | 0.6666666666666666 | 0.5714285714285714 | 12 | 0 | fail |
| concurrent_same_prefix | 0.75 | 0.75 | 12 | 0 | pass |
| eviction_pressure | 0.0 | 0.0 | 0 | 7 | fail |
| exact_repeated_prompt | 0.5 | 0.5 | 4 | 0 | not_comparable |
| low_reuse_random_prompts | 0.0 | 0.0 | 0 | 0 | fail |
| partial_block_prefix | 0.5 | 0.4444444444444444 | 4 | 0 | not_comparable |

Each raw timing sample uses a fresh Python process and reports median, MAD, and CV
for TTFT, E2E, req/s, tokens/s, and peak RSS. The aggregate strict verdict is
`fail`. Wall-clock improvement is reported as
`False` and is true only for a strict pass.
Avoided prefill tokens demonstrate skipped Transformer work independently of noisy
CPU wall-clock measurements.
