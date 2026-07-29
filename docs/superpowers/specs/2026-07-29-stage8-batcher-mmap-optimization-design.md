# Stage 8 — TokenBatcher / mmap Data-Path Optimization Design

Date: 2026-07-29
Status: approved for implementation after Stage 8A calibration
Scope: CPU `TokenBatcher` construction and `next_batch()` data movement

## 1. Goal and non-goals

Stage 8 reduces avoidable memory and Python overhead between persisted token arrays and CPU training
batches while preserving training behavior exactly. The target is the existing character-token
pipeline, not a new data format or a larger model.

The stage must:

- retain a `uint16` mmap instead of eagerly materializing the complete corpus as `int64`;
- replace the per-row Python loop with vectorized indexed gathering;
- convert only the current `[B,T+1]` batch to `int64`;
- cross the NumPy-to-PyTorch boundary once per batch;
- preserve the public `TokenBatcher` constructor, `next_batch()`, and RNG state APIs;
- preserve the exact `numpy.default_rng().integers(...)` call and RNG sequence;
- ensure a later `next_batch()` never changes a previously returned tensor;
- keep checkpoint v2 exact resume and model/training math unchanged.

This stage does not implement KV cache, BPE, GPU, LoRA, distributed training, `torch.compile`,
mixed precision, new model architecture, or unrelated performance work.

## 2. Current implementation and cost

`build_training_components()` loads `train.npy` and `val.npy` with `mmap_mode="r"`, so the source
objects are read-only `uint16` memmaps. `TokenBatcher.__init__()` immediately calls:

```python
np.asarray(tokens, dtype=np.int64)
```

The dtype change forces a complete `int64` allocation and copy. The batcher therefore keeps the
copy, not the intended mmap-backed corpus. Every token expands from two bytes to eight before the
first batch is requested.

`next_batch()` performs one correct, deterministic RNG call for `B` starts, then:

1. allocates an `int64` NumPy `[B,T+1]` array;
2. loops over `B` in Python and copies one slice per row;
3. calls `torch.tensor(...)` for `x`;
4. calls `torch.tensor(...)` again for `y`.

The final two calls create two independent tensors and copy overlapping data twice. The overlap is
all but one column of each row.

## 3. Stage 8A baseline and measurement decision

The merged Stage 7B commit is `18436725aee764660bfdb99938af98c04fb5a87c`. All unchanged quality
gates passed: Ruff format/check, basedpyright, and 376 passed / 1 skipped pytest tests.

The committed 20-step reference config was not sufficiently stable on this Windows hybrid-core
machine. Same-code runs produced one to three cases above the comparison policy's 5% CV limit.
Calibration found two independent effects:

- a cold-machine turbo/thermal transient made the first tasks systematically faster;
- 100 measured steps still left one thread configuration marginally above 5% CV.

The accepted local Stage 8 comparison procedure is:

- Windows high-performance power scheme;
- logical CPU affinity `0..15`, verified with `GetSystemCpuSetInformation` as the eight P-cores and
  their SMT siblings on this i7-14700;
- 120 seconds of unmeasured steady-state preconditioning immediately before a run;
- 15 worker warmup steps;
- 200 measured steps per replicate;
- seven fresh-process replicates per case;
- the unchanged schema-v1 comparison policy: at least five successful replicates, CV at or below
  5%, equal replicate counts, and a strictly-greater-than 7.5% regression boundary.

Two complete 70-worker runs were stable in all ten cases and compared with verdict `pass`.

Baseline manifests:

- `reports/benchmark-v2/20260729T151337Z-18436725aee7-73f82ff3bbb3/run_manifest.json`
- `reports/benchmark-v2/20260729T153745Z-18436725aee7-73f82ff3bbb3/run_manifest.json`

The comparison identity is
`f31b26e82ff94a5c0a31cef25732c8c7815c7189d2345787ebc50b6fa04810bb`. Same-code descriptive
step-time changes ranged from -3.29% to +6.58%; this is the observed local noise floor, not an
optimization result.

The independent profile
`reports/benchmark-v2/profiles/profile-20260729T160108Z-c7dbbef80027/profile_manifest.json`
recorded 3.1032 ms total `data_preparation` CPU time over 20 active steps, about 0.155 ms per step
and 0.264% of the three high-level scope totals. Profiler timings contain instrumentation overhead
and are used only for attribution.

## 4. Behavioral invariants

For the same source values, batch size, block size, seed, and call count:

- sampled start indices must be identical to the old implementation;
- every returned `x` and `y` token must be bit-identical;
- `x.shape == y.shape == (batch_size, block_size)`;
- both outputs must use `torch.int64` on the requested device;
- `y[:, :-1]` must equal `x[:, 1:]`;
- capture/restore must reproduce the next batch;
- no input ndarray, memmap, or sequence may be mutated;
- outputs from call N remain unchanged after call N+1;
- checkpoint resume continues with the same train and validation batches.

The implementation may change output strides. Contiguity is not part of the current public
contract, and the model already uses `reshape()` for targets.

## 5. Selected batch construction

### 5.1 Source retention

The constructor uses `np.asarray(tokens)` without a dtype conversion. Existing ndarray and memmap
storage remains referenced. A Python `Sequence[int]` is materialized once by NumPy, as before, but
no extra corpus-wide dtype conversion is requested.

Validation remains one-dimensional, positive-size, and large-enough validation. A precomputed
`int64` offset vector `[0, 1, ..., T]` is stored because it depends only on `block_size`.

### 5.2 Vectorized indexed gather

`next_batch()` keeps the existing RNG call byte-for-byte in meaning:

```python
starts = rng.integers(0, start_limit, size=batch_size, dtype=np.int64)
```

It creates a `[B,T+1]` index matrix with broadcasting, allocates one local `int64` window matrix,
and uses `np.take(source, indices, out=windows)` to gather and cast directly into that output.
This removes the Python row loop and avoids an intermediate `uint16` gathered batch.

### 5.3 One Tensor owner plus two views

`torch.from_numpy(windows)` creates one CPU tensor sharing the call-local NumPy allocation. Moving
to a non-CPU device, if requested, occurs once for the complete `[B,T+1]` tensor. The returned
values are:

```python
x = batch[:, :-1]
y = batch[:, 1:]
```

They are non-contiguous views sharing one call-owned storage allocation. This is selected over two
contiguous tensors because it avoids the second NumPy-to-PyTorch copy and preserves the exact shift
relationship without duplicating overlapping tokens.

The tradeoff is that consumers requiring contiguous storage may trigger a later copy. The current
GPT embedding accepts strided token IDs, and loss construction already calls `reshape()` on
targets. Training, checkpoint, and benchmark tests will exercise the real consumers. If measured
end-to-end behavior regresses or a public consumer proves to require contiguous tensors, the
fallback is two explicit contiguous slices; that fallback must be measured and documented rather
than assumed.

Every call allocates a new local `windows` owner. No reusable shared output buffer is permitted, so
subsequent calls cannot overwrite earlier results.

## 6. Alternatives rejected

### Keep the whole `int64` corpus

This preserves the old gather dtype but defeats the mmap/memory goal.

### Advanced-index to `uint16`, then call `astype(int64)`

This removes the loop but creates both a gathered `uint16` matrix and an `int64` matrix.
`np.take(..., out=int64_windows)` performs the local gather and cast in one destination.

### Return two contiguous tensors

This preserves current strides but duplicates overlapping tokens and retains two tensor copies.
It remains a measured fallback, not the initial implementation.

### Reuse a persistent output buffer

This would improve allocation behavior but silently changes ownership: call N would be overwritten
by call N+1. It is explicitly prohibited.

### Use `as_strided` or a sliding-window view of the corpus

Randomly selecting rows from the view still requires an indexed gather. It adds stride complexity
without removing the batch allocation.

## 7. Batch-only microbenchmark

The end-to-end training matrix is dominated by model and optimizer work, while the baseline
Profiler attributes only about 0.264% to data preparation. A separate batch-only benchmark is
therefore necessary to say whether the data path itself improved.

The microbenchmark will:

- run in fresh worker processes;
- create a deterministic `.npy` `uint16` corpus and reopen it with `mmap_mode="r"`;
- construct `TokenBatcher` outside the timer;
- warm up outside the timer;
- time one outer loop containing only `next_batch()`;
- retain raw replicate durations and report median, MAD, population CV, calls/second, tokens/second,
  final RSS, and worker-lifetime peak RSS;
- record Git SHA, dirty state, config, environment, source corpus hash, and artifact hashes;
- never use Profiler timings as benchmark inputs.

The batch-only result describes the isolated sampler. The unchanged Benchmark v2 matrix remains
the authority for end-to-end training changes.

## 8. TDD contract

Tests are added before changing `batching.py` and must initially fail on the old implementation
where appropriate.

The tests cover:

- multiple fixed-seed batches against an independent old-path reference implementation;
- RNG capture/restore;
- shape, dtype, device, and one-token shift;
- `uint16` mmap, ordinary `uint16`/`int64` ndarray, and `Sequence[int]`;
- no mutation of writable or read-only sources;
- retention of the mmap-backed source rather than a detached corpus-wide copy;
- shared x/y call-local storage and supported non-contiguous views;
- prior outputs unchanged after another call;
- real GPT forward/loss with the returned views;
- checkpoint v2 exact-resume equivalence;
- batch-only config, worker protocol, statistics, and artifact generation.

Tests compare parsed numbers and semantic fields, not timing thresholds or binary artifact bytes.

## 9. Performance acceptance

After correctness and quality gates pass:

1. run the batch-only microbenchmark on the pre-optimization commit;
2. run it unchanged on the optimization commit;
3. run the complete calibrated Benchmark v2 matrix on the candidate;
4. compare candidate to both accepted same-code baseline runs under the same policy;
5. run the same independent profile case;
6. report every case, including unchanged or slower cases.

A performance conclusion is allowed only when the relevant baseline and candidate are complete,
stable, and environment-compatible. Batch-only improvement does not imply end-to-end improvement.
Profiler scope changes are explanatory only.

## 10. Evidence package

`docs/results/batcher-optimization/` will contain a compact generated or mechanically derived
package with:

- `README.md` explaining method, results, limitations, and exact commands;
- baseline/candidate manifest identities and SHA-256 values;
- per-case medians, CV, MAD, throughput, worker-lifetime peak RSS, and descriptive deltas;
- comparison policy hash and verdict;
- batch-only summary;
- baseline/candidate Profiler scope summaries;
- an artifact manifest hashing every committed evidence file.

Raw run directories, traces, local configs, and worker logs stay under gitignored `reports/`.

## 11. Expected tracked changes

- `src/minigpt/batching.py`
- new batch-only benchmark/report modules and root CLI
- `tests/test_data.py`
- new batch-only benchmark tests
- focused checkpoint/training tests only if an uncovered contract requires them
- this design and its implementation plan
- `docs/results/batcher-optimization/`
- `README.md` and `AGENTS.md` for the final Stage 8 status and reproduction commands
