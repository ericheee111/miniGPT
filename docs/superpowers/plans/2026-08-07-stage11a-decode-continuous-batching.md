# Stage 11A — Decode Continuous Batching implementation plan

1. Add model tests for variable-length batched single-token decode, cache masks, compact scatter,
   dtype/device/shape invariants, batch size one, and no caller-cache mutation.
2. Add executor/engine tests for reference equivalence at batch sizes 1/2/4, mixed lengths,
   permutation invariance, independent RNG, cancellation, invalid-request isolation, overflow, and
   event/metric equivalence.
3. Implement dense cache assembly/scatter and the model-side masked batched decode path.
4. Implement `ContinuousDecodeExecutor` while preserving per-request prefill and overflow fallback.
5. Extend simulator configuration and add automatic reference/continuous logical equivalence.
6. Add fresh-process serving benchmark, strict statistics/comparison, and separate profiler entry.
7. Generate the hash-bound `docs/results/decode-continuous-batching/` evidence package.
8. Run package, Ruff, basedpyright, pytest, diff, manifest, and fresh-checkout verification before
   publication and merge.
