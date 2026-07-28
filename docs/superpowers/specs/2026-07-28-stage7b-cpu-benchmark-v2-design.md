# Stage 7B — Isolated and Reproducible CPU Benchmark v2 Design

Date: 2026-07-28  
Status: approved for implementation  
Scope: trustworthy CPU measurement infrastructure, not performance optimization

## 1. Goal and non-goals

Benchmark v2 must produce low-overhead, reviewable CPU training measurements whose process state,
measurement method, environment, and comparison eligibility are explicit. It replaces benchmark
v1 as the recommended performance workflow without deleting the legacy CLI or changing GPT
mathematics.

This stage does not implement KV cache, BPE, GPU, LoRA, distributed training, `torch.compile`, mmap
or batcher optimization, or any other performance optimization. It does not regenerate Stage 7A
reference-training evidence and does not treat a shared CI runner as performance truth.

## 2. Benchmark v1 findings

Benchmark v1 creates one `TrainingStepWorkload` per case in the parent process, warms it once, then
reuses the same batcher, model, optimizer, allocator, and thread-pool state for every repeat. All
cases run sequentially in that same process.

Each repeat uses one outer `perf_counter` around `measurement_steps`. The timed region includes
batch acquisition, `zero_grad`, forward and loss, backward, gradient clipping, and AdamW step. It
excludes construction, warmup, garbage collection, logging, and artifact writes.

The v1 `cpu_memory_mb` value is RSS sampled after a repeat. It is neither a peak nor isolated to one
replicate. Warmup and repeat order are deterministic rather than randomized. Output lacks Git and
config identity, complete environment evidence, run completeness, stability status, and compatible
baseline/candidate comparison. Therefore its numbers are descriptive only.

Stage 7A reference training times each optimization step and additionally records phase-level
timers. It has training/provenance goals, scheduled events, real data, and a long-lived process.
Benchmark v2 uses synthetic data, isolated short-lived workers, a single outer timer, independent
replicates, and no training events. The measurements must not be compared directly.

## 3. Alternatives considered

### 3.1 Selected: subprocess worker with a JSON protocol

The parent expands and randomizes tasks. Every task is sent to a new
`python -m minigpt.benchmark_v2_worker` process through JSON on stdin. The worker emits exactly one
JSON response on stdout and diagnostics on stderr.

This creates a visible OS process boundary, permits explicit timeout and cleanup, is portable to
Windows spawn behavior, and leaves a protocol that can be reproduced outside the Python API.

### 3.2 Rejected: `multiprocessing` spawn with queues

Spawn would isolate Python state, but request/response serialization, timeout behavior, child
cleanup, and failure diagnostics would be more coupled to the parent implementation. A standalone
worker protocol is easier to test and audit.

### 3.3 Rejected: one process per case with repeated measurements

This reduces startup overhead but reuses model, optimizer, allocator, batcher, and thread-pool state
across replicates. It does not satisfy the core isolation requirement.

## 4. Benchmark unit and process lifecycle

One `BenchmarkCase` × one replicate equals one new worker process.

The worker lifecycle is:

1. parse and validate one request;
2. record PID and start time;
3. set optional CPU affinity;
4. set PyTorch intra-op and inter-op thread counts before parallel work;
5. seed Python, NumPy, and PyTorch;
6. create the synthetic batcher, GPT, and AdamW optimizer;
7. execute configured warmup steps outside the timer;
8. execute all measurement steps inside one outer high-resolution timer;
9. compute aggregate step time and throughput for this replicate;
10. capture final RSS, process-lifetime peak RSS, actual affinity, thread counts, and environment;
11. emit one response and exit.

The same deterministic workload seed is used for every replicate of a case. Replicates measure
system variation rather than model initialization variation. Execution-order randomization uses a
separate orchestrator RNG so it cannot alter worker model state.

The parent never imports or constructs the measured workload. It owns orchestration, durable raw
results, summary generation, run status, and comparison.

## 5. Timing contract

The canonical throughput timer is one `time.perf_counter()` interval around the complete
measurement loop. Every step in that loop includes:

- batch acquisition;
- `optimizer.zero_grad(set_to_none=True)`;
- model forward;
- cross-entropy loss;
- backward;
- gradient clipping;
- optimizer step.

It excludes:

- worker startup and imports;
- environment and thread setup;
- model, optimizer, and batcher construction;
- warmup;
- garbage collection performed before the timer;
- memory/environment reads performed after the timer;
- parent/worker JSON transport;
- logging and report writes;
- profiler instrumentation;
- checkpointing, validation, and text generation.

No phase-level clocks or profiler scopes are used inside the canonical timed loop. The result for a
replicate is the total elapsed interval divided by `measurement_steps`; individual steps are not
treated as statistically independent samples.

## 6. Case identity and configuration

Benchmark v2 uses explicit `cases`, not a model/thread/block/batch Cartesian product. This prevents
accidental oversized matrices and makes one-factor comparisons reviewable.

The schema contains:

```yaml
schema_version: 2
experiment_name: benchmark_v2_smoke
benchmark_seed: 1337
output_root: reports/benchmark-v2
worker_timeout_seconds: 60
warmup_steps: 1
measurement_steps: 2
replicates: 2
torch_num_interop_threads: 1
cpu_affinity: null
max_cv_percent: 10.0
minimum_replicates: 2
regression_threshold_percent: 5.0
relevant_environment_variables:
  - OMP_NUM_THREADS
  - MKL_NUM_THREADS
models:
  tiny:
    n_layer: 1
    n_head: 1
    n_embd: 32
cases:
  - name: tiny_t1_s32_b2
    model: tiny
    torch_num_threads: 1
    block_size: 32
    batch_size: 2
profile:
  case: tiny_t1_s32_b2
  warmup_steps: 1
  active_steps: 2
```

`cpu_affinity: null` means no affinity change. A non-empty list contains explicit logical CPU IDs.
No code assumes that low IDs are performance cores.

The full resolved YAML has a SHA-256 config hash. A case identity is a canonical-JSON SHA-256 over
all workload-affecting fields: case name, model dimensions, vocabulary, block/batch size, intra-op
and inter-op threads, affinity, dtype/device, optimizer type and hyperparameters, gradient clipping,
warmup steps, measurement steps, and workload seed. Output paths, run IDs, timestamps, report
thresholds, and replicate count are not part of case identity.

## 7. Task expansion and execution order

The parent expands every `case × replicate_index` pair, then shuffles the complete list with
`random.Random(benchmark_seed)`. The resulting task IDs and actual order are saved before workers
start.

The execution is sequential in v2. Parallel workers would compete for the same CPU and invalidate
the intended measurement. Fixed-seed expansion is deterministic for an identical resolved config.

`execution_order.json` preserves:

- zero-based execution index;
- task ID;
- case name and case identity;
- replicate index;
- worker seed;
- eventual PID and status when known.

## 8. Worker protocol, failures, and partial runs

The worker request and response use versioned JSON mappings. Unknown schema versions and malformed
fields are rejected.

A successful response includes:

- task/case/replicate identity;
- worker PID;
- start/end UTC;
- warmup and measurement step counts;
- total elapsed seconds and average step time;
- tokens per step and tokens per second;
- parameter count;
- final and peak RSS evidence;
- actual intra-op/inter-op threads and CPU affinity;
- worker environment identity.

A failed response or parent-side failure includes task identity, status, error type, message,
worker return code when available, stderr, and timeout status. Python tracebacks remain diagnostic
stderr and are not represented as successful measurement data.

The parent appends one raw JSONL record after every task result. A timeout uses
`subprocess.run(..., timeout=...)`, which terminates the child and waits for cleanup before
continuing. Parent interruption stops launching new workers and writes a partial manifest.

Run status is:

- `complete`: every expected task has one successful result and all required artifacts are written;
- `partial`: at least one result exists, but a worker failed, timed out, or the parent was
  interrupted;
- `failed`: no successful measurement exists or run initialization failed.

Only `complete` runs are eligible for performance verdicts. A partial run may still produce raw
records and descriptive summaries but is never silently treated as success.

## 9. CPU controls

The worker calls `torch.set_num_threads(case.torch_num_threads)` and
`torch.set_num_interop_threads(config.torch_num_interop_threads)` before constructing the workload
or initiating parallel tensor work.

Affinity behavior is explicit:

- null: retain and record the inherited affinity;
- list: validate IDs, call the OS/psutil affinity API, then read back the actual affinity;
- unsupported or failed: return a worker failure naming the requested IDs and reason.

Process priority is recorded but not changed by default. The Windows active power scheme is read
with `powercfg /getactivescheme` when available; failure produces `null` plus a reason.

## 10. Memory semantics

`final_rss_mib` is the worker RSS read immediately after the canonical measurement loop.

`peak_rss_mib` is a process-lifetime high-water mark:

- Windows: native peak working set (`peak_wset`) when exposed by psutil;
- Linux: `resource.getrusage(RUSAGE_SELF).ru_maxrss`, converted from KiB;
- otherwise: unavailable (`null`) with a reason in the method field.

The accompanying fields are:

```text
final_rss_mib
peak_rss_mib
peak_rss_method
peak_rss_sampling_interval_ms
```

Native high-water marks add no sampling thread to the timed region, so
`peak_rss_sampling_interval_ms` is null. Process-lifetime peak includes imports, construction,
warmup, and measurement; the report must not call it “model memory” or “measurement-only peak.”

No fallback sampler is implemented in this stage because Windows and Linux, the supported
platforms, provide native high-water marks. An unsupported platform reports missing peak evidence
rather than guessing.

## 11. Statistics and stability

One successful replicate contributes one average step time. All raw replicates remain present;
outliers are never deleted automatically.

Per case, v2 calculates:

- median, min, and max step time;
- population standard deviation;
- median absolute deviation (MAD);
- coefficient of variation (population standard deviation / arithmetic mean × 100);
- median tokens per second;
- replicate, success, and failure counts;
- median final RSS and maximum observed peak RSS.

Stability status is:

- `insufficient_samples` when successful replicates are below `minimum_replicates`;
- `unstable` when sample count is sufficient but CV is greater than `max_cv_percent`;
- `stable` otherwise.

CV is a warning about relative spread, not permission to remove a slow result. Median is robust to
an extreme value, MAD expresses typical absolute deviation around the median, population standard
deviation expresses overall spread, and CV normalizes that spread by the mean.

## 12. Run identity and artifacts

Each invocation creates a new run directory:

```text
reports/benchmark-v2/<run-id>/
├── run_manifest.json
├── environment.json
├── resolved_config.yaml
├── raw_replicates.jsonl
├── summary.csv
├── summary.md
└── execution_order.json
```

`run-id` contains a UTC timestamp, Git short SHA, and config-hash prefix. Reports are gitignored by
default. Tests use temporary output roots.

The environment includes:

- Git commit SHA, branch, and dirty state;
- benchmark config SHA-256;
- OS/platform/machine;
- Python, PyTorch, and NumPy versions;
- CPU name and physical/logical core counts;
- CUDA availability;
- configured and actual intra-op/inter-op threads;
- configured and actual affinity;
- selected environment variable values;
- process priority;
- Windows active power scheme or null plus reason;
- run start/end UTC and execution order.

The manifest includes schema version, run ID/status, expected/successful/failed counts, timestamps,
config/environment identity, case identities, artifact repository/run-relative paths, sizes, and
SHA-256 values. The manifest excludes its own hash.

All JSON/YAML/CSV/Markdown artifacts are written to temporary files and atomically replaced.
`raw_replicates.jsonl` is append-and-flush durable after each task so partial evidence survives.

## 13. Comparison and regression decisions

The comparison CLI is:

```powershell
python compare_benchmarks.py `
  --baseline reports/benchmark-v2/<baseline>/run_manifest.json `
  --candidate reports/benchmark-v2/<candidate>/run_manifest.json
```

Cases align only by identical case identity, never by a display label alone. The comparison always
reports descriptive median step-time and throughput deltas for aligned cases.

It refuses a pass/fail regression verdict when:

- either run is not complete;
- case sets or identities do not align;
- either case has fewer than the configured minimum successful replicates;
- either case is not stable;
- required environment compatibility fields differ.

Compatibility fields are OS/machine, CPU identity and core topology, Python/PyTorch/NumPy versions,
CUDA availability, relevant environment variables, process priority, active power scheme, and
measurement methodology. Git SHA and timestamps are recorded but intentionally may differ.

For comparable cases:

```text
step_time_change_percent = (candidate_median / baseline_median - 1) × 100
```

A regression is reported only when the relative increase is strictly greater than
`regression_threshold_percent`; equality is not a failure. The run-level verdict fails if any
comparable case regresses.

Comparison writes JSON and Markdown beside the candidate manifest. It lists every incompatibility
instead of hiding descriptive differences.

## 14. Profiler separation

`profile_benchmark_v2.py` is a separate CLI. It selects one named case from a v2 config, runs in a
fresh process, and writes under a separate profile directory:

```text
profile/
├── profile_manifest.json
├── top_operators.csv
├── profile_report.md
└── trace.json
```

The profile manifest binds the selected case identity, config hash, Git SHA, environment, and
artifact hashes. Profiler scopes may break down data, forward/backward, and optimizer work, but
profile timings never populate raw replicates, summaries, or comparison inputs.

The benchmark runner never automatically launches the profiler.

## 15. Configurations

`configs/benchmark_v2_smoke.yaml` contains a tiny model, two explicit cases, two replicates, and
short warmup/measurement counts. It is suitable for CLI and CI correctness tests, not performance
truth.

`configs/benchmark_v2_reference.yaml` uses the Stage 7A teaching-sized model and 5–7 independent
replicates. Its explicit cases vary one main factor at a time:

- intra-op thread count at fixed sequence length and batch size;
- sequence length at fixed thread count and batch size;
- batch size at fixed thread count and sequence length.

It is a local execution recommendation only and is not run automatically during Stage 7B.

## 16. Test strategy

Unit and integration tests must not assert that one real configuration is faster than another.
They use synthetic result records or tiny workloads with bounded timeouts.

Tests cover:

- distinct PID for every replicate and no worker reuse across cases;
- warmup exclusion and exact measurement count;
- fixed-seed execution-order reproducibility;
- preservation of every raw success/failure;
- median/min/max/population standard deviation/MAD/CV and stability boundaries;
- affinity unset, success, and explicit failure;
- distinct final/peak RSS fields and method metadata;
- worker exception/timeout preservation and child cleanup;
- partial runs never reported as complete;
- config hash, manifest identity, artifact hashes, and atomic output;
- case alignment and environment compatibility;
- regression threshold equality and greater-than boundary;
- benchmark smoke CLI;
- profiler artifact separation and identity.

Subprocess tests use short but realistic timeouts and check that worker PIDs no longer exist after
completion or failure.

CI executes format/lint/type/tests and the tiny smoke correctness path. It must not enforce a
tokens/s threshold or compare shared-runner performance.

## 17. Commit and evidence policy

Logical commits separate the agent guide, design, process isolation, environment/memory evidence,
statistics/reporting, comparison, end-to-end tests, and docs.

The repository commits source, configs, tests, schema/design docs, and one compact deterministic
comparison fixture. It does not commit local benchmark runs, large traces, or machine-specific
performance claims.

Stage 7B ends with a clean feature branch and local verification results. It does not push, merge,
run the full reference matrix, or begin data-path optimization without separate authorization.
