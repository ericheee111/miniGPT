# Stage 11B implementation plan

1. Add model-level padded prefill validation, attention masking, final-logit gather, and compact
   cache equivalence tests without changing training or Stage 9 interfaces.
2. Add `PrefillBatchConfig`, prefill observations/events, and a `ContinuousExecutor` that performs
   deterministic FIFO contiguous batching before reusing Stage 11A batched decode.
3. Extend engine metrics and tests for batch limits, padding, ordering, RNG, cancellation, invalid
   prompts, boundary lengths, and batch failures.
4. Add the third simulator executor, three-way logical equivalence, prefill event artifacts, and
   mixed/overflow workloads.
5. Extend fresh-process benchmark configuration, worker metrics, strict comparison, and separate
   profiler support for equal, mixed, short-heavy, long-heavy, staggered, and padding-pressure
   scenarios.
6. Run canonical benchmark replicates, publish hash-bound evidence under
   `docs/results/batched-prefill/`, then run all gates and verify hashes from a fresh checkout.
