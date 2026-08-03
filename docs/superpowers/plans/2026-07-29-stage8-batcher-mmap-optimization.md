# Stage 8 — TokenBatcher / mmap Optimization Implementation Plan

Date: 2026-07-29
Design: `docs/superpowers/specs/2026-07-29-stage8-batcher-mmap-optimization-design.md`

## Batch 1 — Design and accepted baseline

1. Record the current mmap/copy path and behavioral invariants.
2. Run all unchanged quality gates.
3. Calibrate same-code Benchmark v2 until all cases satisfy the existing policy without dropping
   outliers or relaxing thresholds;
4. retain two accepted baseline manifests and their `pass` comparison locally;
5. run an independent baseline profile and record scope-level attribution;
6. commit only the design and implementation plan.

Expected commit: `design: 设计 mmap batch 数据路径优化`

## Batch 2 — Equivalence and ownership tests

1. Add an independent reference implementation of the old batching algorithm in tests.
2. Assert multiple-batch fixed-seed equivalence.
3. Assert RNG capture/restore.
4. Cover mmap, ndarray, `Sequence`, read-only input, dtype/device/shift, source retention, view
   storage, and prior-output ownership;
5. Define and test a fresh-process batch-only microbenchmark with strict input/result schemas;
6. run focused tests and retain the expected failures against the old implementation;
7. run and retain the batch-only baseline before changing `TokenBatcher`.

Expected commit: `test: 定义 TokenBatcher 等价性与所有权契约`

## Batch 3 — Vectorized batcher implementation

1. retain source dtype/storage with `np.asarray(tokens)`;
2. precompute `[0..T]` offsets;
3. gather into one local `int64` matrix with `np.take(..., out=...)`;
4. create one tensor owner and return shifted views;
5. run focused tests, model tests, checkpoint tests, exact-resume tests, and static gates;
6. commit only the batcher implementation.

Expected commit: `perf: 优化 TokenBatcher mmap 批处理路径`

## Batch 4 — Training-behavior regression verification

1. Run the same batch-only benchmark on the candidate commit;
2. validate raw evidence, stability, environment, and descriptive changes;
3. run model, checkpoint, and exact-resume regression tests again;
4. commit tool, config, and tests, not machine-local raw runs.

Expected commit: `test: 验证 checkpoint 与训练行为不变`

## Batch 5 — Candidate evidence

1. run all quality gates;
2. use the accepted Stage 8A preconditioning, affinity, power, config, and policy unchanged;
3. run a complete candidate reference matrix;
4. compare candidate to both accepted baseline manifests;
5. run the independent candidate profile;
6. calculate batch-only, end-to-end, memory, natural-noise, and per-case results;
7. do not claim improvement for unstable or incompatible evidence.

## Batch 6 — Compact report and final verification

1. generate `docs/results/batcher-optimization/`;
2. hash every committed result artifact;
3. update README and AGENTS status without embedding hand-copied unbound claims;
4. rerun pip check, Ruff format/check, basedpyright, and full pytest;
5. verify the branch commit list and clean worktree;
6. restore the original Windows balanced power scheme.

Expected commit: `docs: 发布 batcher 优化性能证据`
