# Stage 9 — KV Cache Autoregressive Generation Implementation Plan

Date: 2026-08-06
Design: `docs/superpowers/specs/2026-08-06-stage9-kv-cache-generation-design.md`

## Batch 1 — Cache and attention contracts

1. Add failing model tests for cache shape/dtype/device/length/bytes and invalid caches.
2. Add failing prefill and single-/multi-token decode equivalence tests, including batch > 1.
3. Introduce immutable cache types and cached attention/block methods.
4. Implement `GPT.prefill()` and `GPT.decode()` without changing `GPT.forward()`.
5. Run model tests, Ruff, and basedpyright; commit the logical batch.

Teaching checkpoint: explain why K/V are reusable, why historical Q is not, cached tensor shapes,
and why each new query still reads all historical keys.

## Batch 2 — Cached generation and overflow

1. Add failing tests for zero-token generation, fixed-generator sampling equivalence, near-full
   prompts, prompts longer than the context, and generation crossing `block_size`.
2. Factor shared sampling and implement `GPT.generate_cached()`.
3. Re-prefill the latest full window whenever learned-position semantics require a slide.
4. Keep `generate()` as the uncached baseline and add an opt-in cached CLI switch.
5. Run focused generation, checkpoint, exact-resume, and Stage 8 tests; commit the logical batch.

Teaching checkpoint: explain prefill/decode compute, space-for-time cost, and learned absolute
position overflow.

## Batch 3 — Isolated inference benchmark

1. Define config, worker protocol, raw record, statistics, comparison, and artifact tests first.
2. Implement fresh-process deterministic cached/uncached workers with untimed warmup.
3. Report prefill/TTFT, median decode latency, tokens/s, end-to-end time, peak RSS, cache bytes,
   median, MAD, CV, and replicate count.
4. Bind raw evidence, environment, config, execution order, and summaries with SHA-256.
5. Add canonical and smoke configs plus root CLI; run focused benchmark tests and commit.

Teaching checkpoint: distinguish TTFT, TPOT, throughput, and end-to-end latency; explain why small
workloads need not improve.

## Batch 4 — Evidence and final verification

1. Run the canonical isolated benchmark in paired cached/uncached order and evaluate strict policy.
2. Run a separate descriptive profiler experiment.
3. Generate `docs/results/kv-cache-generation/` with README, summary, evidence, and manifest.
4. Verify all committed hashes from the repository checkout.
5. Run pip check, Ruff format/check, basedpyright, full pytest, and `git diff --check main...HEAD`.
6. Confirm Stage 8/reference evidence are unchanged and record local commits; do not push or open a
   PR.
