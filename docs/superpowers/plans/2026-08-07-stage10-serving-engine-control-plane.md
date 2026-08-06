# Stage 10 — MiniServe Control Plane Implementation Plan

1. Freeze the serving boundary and state machine in a design document.
2. Add contract tests for FIFO admission, cache reservation/release, lifecycle transitions,
   cancellation, failure isolation, deterministic events, independent RNG, starvation, metrics,
   and Stage 9 overflow fallback.
3. Implement typed request, scheduler, executor-result, event, and metrics structures.
4. Implement strict FIFO admission and the cancellation/admission/prefill/decode tick order.
5. Implement the per-request `GPT.prefill()`/`GPT.decode()` reference executor.
6. Add a strict JSON/YAML workload loader, deterministic simulation loop, four output artifacts,
   and three fixed scenarios.
7. Generate a hash-bound Stage 10 evidence package and document results without throughput claims.
8. Run focused tests, all repository quality gates, historical regression tests, diff checks, and
   verify that Stage 7A, Stage 8, and Stage 9 evidence trees are byte-identical to `main`.
