# Stage 7B CPU Benchmark v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh-process, reproducible CPU training benchmark with durable raw evidence,
stability reporting, compatible baseline/candidate comparison, and explicitly separate profiling.

**Architecture:** A parent orchestrator expands explicit benchmark cases into randomized
case×replicate tasks and launches one JSON-protocol subprocess per task. Workers own all measured
PyTorch state, apply CPU controls, run warmup outside one canonical timer, and return one aggregate
measurement plus environment and RSS evidence. Pure statistics, artifact generation, comparison,
and profiling modules consume the versioned records without participating in the timed region.

**Tech Stack:** Python 3.11–3.14, PyTorch CPU, NumPy, psutil, PyYAML, dataclasses, `subprocess`,
JSON/JSONL/CSV/Markdown, pytest, Ruff ALL, basedpyright all.

## Global Constraints

- Preserve the existing GPT, training, checkpoint v2, exact-resume, Stage 7A evidence, benchmark v1,
  and profiler v1 behavior.
- Do not add KV cache, BPE, GPU, LoRA, distributed training, `torch.compile`, mmap/batcher
  optimization, or performance tuning.
- One `BenchmarkCase × replicate` must execute in one new worker process; workers run sequentially.
- Canonical timing uses one outer `perf_counter` and contains no phase timers or profiler scopes.
- Preserve every raw replicate and never remove outliers automatically.
- Shared CI may verify correctness only; it must not enforce performance thresholds.
- Full reference configuration is not executed during Stage 7B without separate confirmation.
- Reports, raw local runs, traces, model weights, and machine-specific benchmark numbers remain
  gitignored.
- Every implementation task follows red → green → refactor and ends in a logical Chinese commit.

---

## File responsibility map

- `src/minigpt/benchmark_v2_types.py`: versioned immutable cases, tasks, worker records, summaries,
  run/artifact records, JSON-safe aliases, and dedicated errors.
- `src/minigpt/benchmark_v2_config.py`: strict YAML parsing, explicit case resolution, canonical
  config/case serialization, and SHA-256 identities.
- `src/minigpt/benchmark_v2_environment.py`: Git/platform/CPU/toolchain identity, affinity control,
  process priority, power scheme, final RSS, and native process peak RSS.
- `src/minigpt/benchmark_v2_worker.py`: stdin/stdout worker protocol and canonical workload timer.
- `src/minigpt/benchmark_v2_statistics.py`: pure replicate aggregation and stability classification.
- `src/minigpt/benchmark_v2_report.py`: atomic environment/config/order/raw/summary/manifest writers
  and strict manifest/summary loaders.
- `src/minigpt/benchmark_v2.py`: deterministic task expansion, subprocess lifecycle, partial-run
  orchestration, and completion validation.
- `src/minigpt/benchmark_v2_compare.py`: environment/case alignment, descriptive deltas, guarded
  regression verdicts, JSON and Markdown outputs.
- `src/minigpt/benchmark_v2_profile.py`: separate profiled worker path and profile manifest binding.
- `benchmark_v2.py`, `compare_benchmarks.py`, `profile_benchmark_v2.py`: typed public CLIs.
- `configs/benchmark_v2_smoke.yaml`, `configs/benchmark_v2_reference.yaml`: correctness-smoke and
  recommended local reference methodologies.
- `tests/test_benchmark_v2_*.py`: focused config, worker, orchestration, report, comparison, CLI, and
  profiler contracts.
- `tests/fixtures/benchmark_v2/`: compact fixed manifests/summaries used without real timing claims.
- `README.md`: v1/v2 distinction, commands, timing/memory/statistics interpretation, comparison,
  profiler separation, and local reference-run advice.

### Task 1: Define strict v2 configuration and identities

**Files:**
- Create: `src/minigpt/benchmark_v2_types.py`
- Create: `src/minigpt/benchmark_v2_config.py`
- Create: `tests/test_benchmark_v2_config.py`

**Interfaces:**
- Produces:
  - `BenchmarkV2Config`, `BenchmarkV2Case`, `ProfileV2Settings`
  - `load_benchmark_v2_config(path: Path) -> BenchmarkV2Config`
  - `resolved_config_document(config: BenchmarkV2Config) -> dict[str, JsonValue]`
  - `resolved_config_sha256(config: BenchmarkV2Config) -> str`
  - `case_identity(config: BenchmarkV2Config, case: BenchmarkV2Case) -> str`
- Consumes: existing `benchmark_config.ConfigValue` style and `settings.GPTConfig` dimensions only.

- [ ] **Step 1: Write failing strict-schema and identity tests**

```python
def test_v2_config_resolves_explicit_cases_and_stable_hash(tmp_path: Path) -> None:
    # Given: a schema-v2 YAML with one model and two explicit cases.
    path = write_v2_config(tmp_path)

    # When: it is parsed twice.
    first = load_benchmark_v2_config(path)
    second = load_benchmark_v2_config(path)

    # Then: order and identities are stable without creating a Cartesian product.
    assert [case.name for case in first.cases] == ["tiny_t1_s32_b2", "tiny_t2_s32_b2"]
    assert resolved_config_sha256(first) == resolved_config_sha256(second)
    assert case_identity(first, first.cases[0]) != case_identity(first, first.cases[1])
```

Add parameterized failures for unknown keys, schema version, duplicate names, unknown model,
non-positive step/replicate/timeout values, invalid head divisibility, empty affinity, negative
logical CPU IDs, duplicate affinity IDs, and `minimum_replicates > replicates`.

- [ ] **Step 2: Run the new tests and verify red**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_benchmark_v2_config.py -q
```

Expected: collection fails because `minigpt.benchmark_v2_config` does not exist.

- [ ] **Step 3: Implement immutable records and exact-key parsing**

Define focused dataclasses:

```python
@dataclass(frozen=True, slots=True)
class BenchmarkV2Case:
    name: str
    model_name: str
    n_layer: int
    n_head: int
    n_embd: int
    torch_num_threads: int
    block_size: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class BenchmarkV2Config:
    schema_version: int
    experiment_name: str
    benchmark_seed: int
    vocab_size: int
    output_root: Path
    worker_timeout_seconds: float
    warmup_steps: int
    measurement_steps: int
    replicates: int
    torch_num_interop_threads: int
    cpu_affinity: tuple[int, ...] | None
    max_cv_percent: float
    minimum_replicates: int
    regression_threshold_percent: float
    relevant_environment_variables: tuple[str, ...]
    cases: tuple[BenchmarkV2Case, ...]
    profile: ProfileV2Settings
```

Reject unexpected keys at every mapping level. Serialize resolved paths with forward slashes and
hash canonical JSON using `sort_keys=True`, `separators=(",", ":")`, and UTF-8.

- [ ] **Step 4: Run config tests, Ruff, and basedpyright**

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/benchmark_v2_types.py src/minigpt/benchmark_v2_config.py tests/test_benchmark_v2_config.py
.\.venv\Scripts\ruff.exe check src/minigpt/benchmark_v2_types.py src/minigpt/benchmark_v2_config.py tests/test_benchmark_v2_config.py
.\.venv\Scripts\basedpyright.exe src/minigpt/benchmark_v2_types.py src/minigpt/benchmark_v2_config.py tests/test_benchmark_v2_config.py
.\.venv\Scripts\pytest.exe tests/test_benchmark_v2_config.py -q
```

Expected: all pass with 0 type errors.

- [ ] **Step 5: Commit**

```powershell
git add src/minigpt/benchmark_v2_types.py src/minigpt/benchmark_v2_config.py tests/test_benchmark_v2_config.py
git commit -m "定义 Benchmark v2 配置与实验身份"
```

### Task 2: Implement the isolated worker, timing boundary, CPU controls, and RSS evidence

**Files:**
- Create: `src/minigpt/benchmark_v2_environment.py`
- Create: `src/minigpt/benchmark_v2_worker.py`
- Create: `tests/test_benchmark_v2_worker.py`
- Modify: `src/minigpt/benchmark_workload.py`

**Interfaces:**
- Consumes: `BenchmarkV2Config`, `BenchmarkV2Case`, existing `TrainingStepWorkload`.
- Produces:
  - `WorkerRequest`, `WorkerResult`, `WorkerFailure`
  - `run_worker_request(request: WorkerRequest) -> WorkerResult`
  - `worker_main() -> int`
  - `capture_worker_environment(...) -> WorkerEnvironment`
  - `apply_cpu_affinity(requested: tuple[int, ...] | None) -> tuple[int, ...] | None`
  - `read_process_memory() -> ProcessMemoryEvidence`

- [ ] **Step 1: Write failing worker boundary and environment tests**

Use a counting test workload factory so tests can assert:

```python
assert workload.constructed_before_timer is True
assert workload.warmup_calls == request.warmup_steps
assert workload.measured_calls == request.measurement_steps
assert result.measurement_steps == request.measurement_steps
assert result.final_rss_mib > 0
assert result.peak_rss_method in {
    "windows_peak_working_set",
    "linux_getrusage_ru_maxrss",
}
assert result.peak_rss_sampling_interval_ms is None
```

Patch `perf_counter` with a deterministic sequence to prove warmup is outside the two timer calls.
Test affinity null, successful set/readback, and an injected `AccessDenied` failure. Test malformed
stdin returns nonzero with a JSON failure response.

- [ ] **Step 2: Run worker tests and verify red**

```powershell
.\.venv\Scripts\pytest.exe tests/test_benchmark_v2_worker.py -q
```

Expected: import failure for the worker/environment modules.

- [ ] **Step 3: Expose uninstrumented workload creation without changing its math**

Add a small constructor helper or protocol-compatible factory around `TrainingStepWorkload`; do not
change `step()`. The canonical timer remains:

```python
for _ in range(request.warmup_steps):
    workload.step()

_ = gc.collect()
gc_was_enabled = gc.isenabled()
gc.disable()
try:
    started = perf_counter()
    for _ in range(request.measurement_steps):
        workload.step()
    elapsed_seconds = perf_counter() - started
finally:
    if gc_was_enabled:
        gc.enable()
```

Set `torch.set_num_threads`, `torch.set_num_interop_threads`, and affinity before workload
construction. Capture memory only after the timed interval.

- [ ] **Step 4: Implement versioned stdin/stdout protocol**

`worker_main()` reads one JSON object from stdin, validates every required field, catches ordinary
exceptions into a failure response, writes exactly one compact JSON object to stdout, and returns
nonzero for failure. It lets `KeyboardInterrupt` exit with the conventional interrupted status.

- [ ] **Step 5: Run focused gates**

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/benchmark_v2_environment.py src/minigpt/benchmark_v2_worker.py src/minigpt/benchmark_workload.py tests/test_benchmark_v2_worker.py
.\.venv\Scripts\ruff.exe check src/minigpt/benchmark_v2_environment.py src/minigpt/benchmark_v2_worker.py src/minigpt/benchmark_workload.py tests/test_benchmark_v2_worker.py
.\.venv\Scripts\basedpyright.exe src/minigpt/benchmark_v2_environment.py src/minigpt/benchmark_v2_worker.py tests/test_benchmark_v2_worker.py
.\.venv\Scripts\pytest.exe tests/test_benchmark_v2_worker.py tests/test_benchmark.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add src/minigpt/benchmark_v2_environment.py src/minigpt/benchmark_v2_worker.py src/minigpt/benchmark_workload.py tests/test_benchmark_v2_worker.py
git commit -m "实现独立进程 Benchmark worker"
```

### Task 3: Orchestrate randomized fresh-process replicates and preserve partial runs

**Files:**
- Create: `src/minigpt/benchmark_v2.py`
- Create: `tests/test_benchmark_v2_orchestration.py`

**Interfaces:**
- Consumes: config/case identities and worker JSON protocol.
- Produces:
  - `expand_benchmark_tasks(config: BenchmarkV2Config) -> tuple[BenchmarkTask, ...]`
  - `execute_worker(task: BenchmarkTask, timeout_seconds: float) -> RawReplicate`
  - `run_benchmark_v2(config: BenchmarkV2Config) -> BenchmarkV2Artifacts`
- Injects a `WorkerLauncher` callable in unit tests; production uses `subprocess.run`.

- [ ] **Step 1: Write failing deterministic-order, PID-isolation, timeout, and partial-run tests**

```python
def test_task_order_is_seeded_and_contains_every_replicate(config: BenchmarkV2Config) -> None:
    first = expand_benchmark_tasks(config)
    second = expand_benchmark_tasks(config)
    assert first == second
    assert {(task.case.name, task.replicate_index) for task in first} == {
        ("tiny_t1_s32_b2", 0),
        ("tiny_t1_s32_b2", 1),
        ("tiny_t2_s32_b2", 0),
        ("tiny_t2_s32_b2", 1),
    }
```

Use the real tiny worker for an integration test and assert every PID is unique and has exited.
Use injected launchers for return-code failure, malformed response, and timeout; assert all raw
records remain ordered and run status is `partial`, never `complete`.

- [ ] **Step 2: Run orchestration tests and verify red**

```powershell
.\.venv\Scripts\pytest.exe tests/test_benchmark_v2_orchestration.py -q
```

- [ ] **Step 3: Implement deterministic expansion and subprocess cleanup**

Shuffle tasks with `random.Random(config.benchmark_seed).shuffle(tasks)`. Execute:

```python
completed = subprocess.run(
    [sys.executable, "-m", "minigpt.benchmark_v2_worker"],
    input=worker_request_json,
    capture_output=True,
    text=True,
    timeout=config.worker_timeout_seconds,
    check=False,
)
```

Convert `TimeoutExpired`, nonzero exit, invalid stdout, and worker-declared failure into typed raw
failure records. Never reuse `Popen`, pools, models, or optimizer objects.

- [ ] **Step 4: Implement interruption-safe run state**

Write initial run identity/order before worker launch. Append and flush one raw record after every
task. In `except KeyboardInterrupt`, finalize status as partial and re-raise after durable state is
updated.

- [ ] **Step 5: Run worker/orchestration tests and full legacy benchmark tests**

```powershell
.\.venv\Scripts\pytest.exe tests/test_benchmark_v2_worker.py tests/test_benchmark_v2_orchestration.py tests/test_benchmark.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add src/minigpt/benchmark_v2.py tests/test_benchmark_v2_orchestration.py
git commit -m "隔离并随机化 Benchmark replicate"
```

### Task 4: Aggregate stability and generate durable run evidence

**Files:**
- Create: `src/minigpt/benchmark_v2_statistics.py`
- Create: `src/minigpt/benchmark_v2_report.py`
- Create: `tests/test_benchmark_v2_report.py`
- Modify: `src/minigpt/benchmark_v2.py`

**Interfaces:**
- Produces:
  - `summarize_replicates(...) -> BenchmarkV2Summary`
  - `write_run_artifacts(...) -> BenchmarkV2Artifacts`
  - `load_run_manifest(path: Path) -> RunManifest`
  - atomic JSON/YAML/CSV/Markdown helpers.

- [ ] **Step 1: Write failing pure-statistics tests**

Use exact values `[10.0, 11.0, 12.0]` and assert median 11, min/max 10/12, population stddev
`0.81649658`, MAD 1, and CV `7.422696`. Add:

```python
assert summarize(..., minimum_replicates=5).stability == "insufficient_samples"
assert summarize(..., minimum_replicates=3, max_cv_percent=7.0).stability == "unstable"
assert summarize(..., minimum_replicates=3, max_cv_percent=7.5).stability == "stable"
```

Include failed records in raw counts but exclude them from numeric statistics.

- [ ] **Step 2: Write failing artifact/manifest tests**

Generate a tiny synthetic complete run and assert these files exist:

```text
run_manifest.json
environment.json
resolved_config.yaml
raw_replicates.jsonl
summary.csv
summary.md
execution_order.json
```

Verify every manifest path is run-relative and each declared SHA-256/size matches bytes. Verify a
failed worker yields `partial`, nonzero failure count, and no success claim in Markdown.

- [ ] **Step 3: Implement pure statistics**

Compute from one aggregate average per successful replicate. Never filter or sort away raw records.
Use `statistics.median`, `statistics.pstdev`, and the arithmetic mean for CV.

- [ ] **Step 4: Implement atomic artifacts and manifest**

Write complete content to a sibling temporary file, flush, then `Path.replace`. Keep the manifest
self-excluded. Render Markdown methodology with the exact timer boundary, memory method, stability
threshold, and explicit “not a shared-runner performance gate” statement.

- [ ] **Step 5: Integrate finalization into the orchestrator**

Create a unique run ID from UTC timestamp, short Git SHA, and config-hash prefix. Resolve collision
by rejecting the existing directory rather than overwriting evidence.

- [ ] **Step 6: Run report/orchestration tests and static gates**

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/benchmark_v2_statistics.py src/minigpt/benchmark_v2_report.py src/minigpt/benchmark_v2.py tests/test_benchmark_v2_report.py
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\basedpyright.exe
.\.venv\Scripts\pytest.exe tests/test_benchmark_v2_orchestration.py tests/test_benchmark_v2_report.py -q
```

- [ ] **Step 7: Commit**

```powershell
git add src/minigpt/benchmark_v2_statistics.py src/minigpt/benchmark_v2_report.py src/minigpt/benchmark_v2.py tests/test_benchmark_v2_report.py
git commit -m "记录 Benchmark 环境内存与稳定性证据"
```

### Task 5: Add guarded baseline/candidate comparison

**Files:**
- Create: `src/minigpt/benchmark_v2_compare.py`
- Create: `compare_benchmarks.py`
- Create: `tests/test_benchmark_v2_compare.py`
- Create: `tests/fixtures/benchmark_v2/baseline/run_manifest.json`
- Create: `tests/fixtures/benchmark_v2/baseline/environment.json`
- Create: `tests/fixtures/benchmark_v2/baseline/summary.csv`
- Create: `tests/fixtures/benchmark_v2/candidate/run_manifest.json`
- Create: `tests/fixtures/benchmark_v2/candidate/environment.json`
- Create: `tests/fixtures/benchmark_v2/candidate/summary.csv`

**Interfaces:**
- Consumes strict manifest/environment/summary loaders.
- Produces:
  - `compare_runs(baseline: Path, candidate: Path) -> BenchmarkComparison`
  - `write_comparison(...) -> ComparisonArtifacts`
  - root CLI with required baseline/candidate paths.

- [ ] **Step 1: Write failing case-alignment and compatibility tests**

Cover identical case identities, missing/extra identities, incomplete run status, CPU/Python/Torch/
NumPy/power-scheme/env-var mismatch, unstable summary, and insufficient replicates. Assert
descriptive deltas remain present when verdict is `not_comparable`.

- [ ] **Step 2: Write failing regression threshold boundary tests**

```python
assert compare_step_times(100.0, 105.0, threshold_percent=5.0).regressed is False
assert compare_step_times(100.0, 105.01, threshold_percent=5.0).regressed is True
```

- [ ] **Step 3: Implement strict comparison**

Align by `case_identity`. Compatibility excludes Git SHA/timestamps but includes the documented
machine, toolchain, control, and methodology fields. Return explicit incompatibility entries:

```python
EnvironmentMismatch(field="pytorch_version", baseline="2.12.1", candidate="2.13.0")
```

Only comparable stable cases receive pass/fail; one regression makes the run-level verdict fail.

- [ ] **Step 4: Implement JSON/Markdown comparison CLI**

Default outputs beside the candidate manifest with a baseline run-ID suffix. Never mutate either
source run.

- [ ] **Step 5: Run comparison tests and CLI fixture**

```powershell
.\.venv\Scripts\pytest.exe tests/test_benchmark_v2_compare.py -q
.\.venv\Scripts\python.exe compare_benchmarks.py --baseline tests/fixtures/benchmark_v2/baseline/run_manifest.json --candidate tests/fixtures/benchmark_v2/candidate/run_manifest.json
```

- [ ] **Step 6: Commit**

```powershell
git add src/minigpt/benchmark_v2_compare.py compare_benchmarks.py tests/test_benchmark_v2_compare.py tests/fixtures/benchmark_v2
git commit -m "比较兼容的 Benchmark 基线与候选结果"
```

### Task 6: Add public benchmark v2 CLI and canonical configs

**Files:**
- Create: `benchmark_v2.py`
- Create: `configs/benchmark_v2_smoke.yaml`
- Create: `configs/benchmark_v2_reference.yaml`
- Create: `tests/test_benchmark_v2_cli.py`

**Interfaces:**
- CLI consumes `--config PATH` and prints run directory, manifest, raw JSONL, summary CSV, and
  Markdown paths.

- [ ] **Step 1: Write failing CLI smoke contract**

Copy the repository smoke config to `tmp_path`, replace only `output_root`, run with
`timeout=120`, and assert:

- return code zero;
- 2 cases × 2 replicates = 4 raw records;
- four distinct worker PIDs;
- manifest status complete;
- summary has two rows;
- no profile directory is created.

- [ ] **Step 2: Add smoke config**

Use one tiny model and two explicit intra-op cases, `warmup_steps: 1`,
`measurement_steps: 2`, `replicates: 2`, `minimum_replicates: 2`, inter-op 1, affinity null.

- [ ] **Step 3: Add reference config without executing it**

Use the 4-layer/4-head/128-embedding teaching model, 7 replicates, 10 warmup steps, 20 measurement
steps, and explicit one-factor cases for:

- threads 1/4/8/12/20 at block 128, batch 16;
- blocks 64/256 at threads 12, batch 16;
- batches 4/8/32 at threads 12, block 128.

The shared baseline case appears once, producing 10 cases rather than a Cartesian product.

- [ ] **Step 4: Implement typed root CLI and run smoke**

```powershell
.\.venv\Scripts\python.exe benchmark_v2.py --config configs/benchmark_v2_smoke.yaml
```

Record the real run path and inspect manifest/raw/summary files, but do not commit the report.

- [ ] **Step 5: Run CLI tests and quality checks**

```powershell
.\.venv\Scripts\ruff.exe format benchmark_v2.py tests/test_benchmark_v2_cli.py
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\basedpyright.exe
.\.venv\Scripts\pytest.exe tests/test_benchmark_v2_cli.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add benchmark_v2.py configs/benchmark_v2_smoke.yaml configs/benchmark_v2_reference.yaml tests/test_benchmark_v2_cli.py
git commit -m "提供 Benchmark v2 CLI 与固定配置"
```

### Task 7: Bind a separate profiler run to v2 case identity

**Files:**
- Create: `src/minigpt/benchmark_v2_profile.py`
- Create: `profile_benchmark_v2.py`
- Create: `tests/test_benchmark_v2_profile.py`
- Reuse without changing math: `src/minigpt/profiling.py`

**Interfaces:**
- Consumes one named v2 case and profile settings.
- Produces a distinct profile directory and `profile_manifest.json` with config hash, case identity,
  Git SHA, environment, and artifact hashes.

- [ ] **Step 1: Write failing separation and identity tests**

Run a tiny profile in `tmp_path`; assert top operators, Markdown, and trace exist, the manifest
matches selected case/config/Git identity, and none of its paths or timings can be loaded as a
benchmark raw replicate or summary.

- [ ] **Step 2: Implement separate profiled subprocess**

The root CLI launches one worker mode dedicated to `profiled_step()`. It does not call
`run_benchmark_v2` and does not write `raw_replicates.jsonl` or `summary.csv`.

- [ ] **Step 3: Run focused profile tests**

```powershell
.\.venv\Scripts\pytest.exe tests/test_benchmark_v2_profile.py tests/test_benchmark.py -q
```

- [ ] **Step 4: Commit**

```powershell
git add src/minigpt/benchmark_v2_profile.py profile_benchmark_v2.py tests/test_benchmark_v2_profile.py
git commit -m "分离并绑定 Benchmark v2 Profiler 证据"
```

### Task 8: Harden Stage 7A provenance failure finalization

**Files:**
- Modify: `train.py`
- Modify: `tests/test_run_provenance.py`

**Interfaces:**
- Preserve the existing CLI and `run_provenance.json` schema.
- Ensure ordinary exceptions and `KeyboardInterrupt` finalize the active segment as failed or
  interrupted, then re-raise the original exception.

- [ ] **Step 1: Write failing tests for ordinary exception and KeyboardInterrupt**

Patch `run_training` to raise `TypeError("boom")` and `KeyboardInterrupt`. Assert the segment no
longer remains `running`, the status/error fields are durable, and the exact exception propagates.

- [ ] **Step 2: Run tests and verify the existing narrow catch fails**

```powershell
.\.venv\Scripts\pytest.exe tests/test_run_provenance.py -q
```

- [ ] **Step 3: Implement explicit finalization without swallowing**

Catch `KeyboardInterrupt` separately for an interrupted status. Catch `Exception` for ordinary
failures. In both paths, attempt provenance finalization, then use bare `raise` to preserve the
original traceback. Do not catch `BaseException` broadly.

- [ ] **Step 4: Run Stage 7A/6 regression tests**

```powershell
.\.venv\Scripts\pytest.exe tests/test_run_provenance.py tests/test_trainer.py tests/test_checkpoint.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add train.py tests/test_run_provenance.py
git commit -m "完善训练 provenance 异常终态"
```

### Task 9: Publish methodology documentation and end-to-end contracts

**Files:**
- Modify: `README.md`
- Create: `docs/superpowers/plans/2026-07-28-stage7b-cpu-benchmark-v2.md` (this file)
- Create or modify: `tests/test_readme.py`
- Modify: `.github/workflows/quality.yml` only if the smoke CLI remains bounded enough for CI;
  never add a performance threshold.

**Interfaces:**
- Documents v1 as legacy/descriptive and v2 as canonical.
- Links the design, plan, configs, CLIs, artifact schema, comparison, and profiler separation.

- [ ] **Step 1: Write failing README contract tests**

Assert the README includes:

```text
benchmark_v2.py
compare_benchmarks.py
profile_benchmark_v2.py
fresh process
final_rss_mib
peak_rss_mib
stable
unstable
insufficient_samples
shared CI runner
```

Also assert every new local Markdown link exists.

- [ ] **Step 2: Update README without changing Stage 7A evidence**

Explain:

- warmup and process isolation;
- intra-op versus inter-op threads;
- median/MAD/population standard deviation/CV;
- no automatic outlier deletion;
- thermal/power drift and cautious affinity on hybrid CPUs;
- comparison eligibility and threshold semantics;
- profiler overhead and CI limitations;
- reference config command as a recommendation, not an automatically executed result.

- [ ] **Step 3: Decide CI smoke inclusion from measured runtime**

If the smoke CLI consistently finishes within 120 seconds locally, add a correctness-only workflow
step that writes to an ignored temporary report and asserts exit success. Otherwise keep coverage
in pytest subprocess tests and document why no extra workflow step is added.

- [ ] **Step 4: Run README and benchmark suites**

```powershell
.\.venv\Scripts\pytest.exe tests/test_readme.py tests/test_benchmark.py tests/test_benchmark_v2_config.py tests/test_benchmark_v2_worker.py tests/test_benchmark_v2_orchestration.py tests/test_benchmark_v2_report.py tests/test_benchmark_v2_compare.py tests/test_benchmark_v2_cli.py tests/test_benchmark_v2_profile.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add README.md tests/test_readme.py .github/workflows/quality.yml
git commit -m "更新 CPU Benchmark v2 使用说明"
```

### Task 10: Verify, review, and hand off the clean feature branch

**Files:**
- Review all Stage 7B source, configs, tests, fixtures, and docs.
- Do not create or commit machine-specific formal reference results.

**Interfaces:**
- Produces final local evidence and commit list; does not push or merge.

- [ ] **Step 1: Run the required full gates**

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe format --check src tests
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\basedpyright.exe
.\.venv\Scripts\pytest.exe
```

- [ ] **Step 2: Run the canonical smoke CLIs**

```powershell
.\.venv\Scripts\python.exe benchmark_v2.py --config configs/benchmark_v2_smoke.yaml
.\.venv\Scripts\python.exe profile_benchmark_v2.py --config configs/benchmark_v2_smoke.yaml
.\.venv\Scripts\python.exe compare_benchmarks.py --baseline tests/fixtures/benchmark_v2/baseline/run_manifest.json --candidate tests/fixtures/benchmark_v2/candidate/run_manifest.json
```

Inspect all manifests, hashes, worker PIDs, execution order, RSS semantics, stability, and profiler
separation. Remove only disposable smoke/profiler reports after preserving command output.

- [ ] **Step 3: Request independent code review**

Ask the reviewer to inspect process isolation, subprocess cleanup, timing instrumentation,
environment compatibility, partial-run logic, artifact hash integrity, and absence of performance
claims from smoke data. Fix every Critical/Important finding with a focused test and commit.

- [ ] **Step 4: Re-run full verification after review fixes**

Repeat Step 1 and the benchmark v2 smoke CLI. Do not rely on earlier green output.

- [ ] **Step 5: Confirm clean branch and report**

```powershell
git status --short
git log --oneline --decorate -20
git diff main...HEAD --stat
```

Report the modified files, teaching points, exact commands/results, design evidence, limitations,
commit SHAs, clean status, the unexecuted local reference command, and the proposed next stage:
optimize batcher/mmap data flow using Benchmark v2. Do not begin that optimization.
