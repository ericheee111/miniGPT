# Stage 7A Reference Training Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one complete, provenance-linked CPU reference-training result and a compact,
automatically generated evidence package suitable for repository documentation and honest resume
claims.

**Architecture:** Keep checkpoint v2 and the core trainer unchanged. Add a typed read-only
checkpoint metadata boundary, an opt-in run-provenance sidecar around the existing training CLI,
and a pure report pipeline that validates source artifacts before generating CSV, figures,
Markdown, environment evidence, and a hash manifest.

**Tech Stack:** Python 3.11–3.14, PyTorch CPU, NumPy, PyYAML, psutil, matplotlib with the Agg
backend, pytest, Ruff, basedpyright, Git, PowerShell.

## Global Constraints

- Work only on Stage 7A; do not implement Benchmark v2, KV cache, BPE, GPU, LoRA, or distributed
  training.
- Preserve checkpoint format v2 and Stage 6 exact-resume semantics.
- Keep the base training dependency set free of matplotlib; expose it through `report` and `dev`
  extras.
- Keep `typeCheckingMode = "all"`, Ruff `select = ["ALL"]`, and Python 3.11 syntax.
- Do not add `type: ignore`, `pyright: ignore`, broad `Any`, or CUDA assumptions.
- Tests use local data/checkpoints and never download Tiny Shakespeare.
- Calibration and formal source artifacts remain ignored; only compact generated evidence enters
  Git.
- The formal run starts from a clean Git tree and uses one real process exit plus v2 resume with
  the unchanged full config.
- The final checkpoint reaches `max_steps - 1` and is hashed but never committed.
- Metrics, samples, and report conclusions are generated mechanically without hand editing.
- Each logical task passes focused Ruff, basedpyright, and pytest checks before a Chinese commit.

---

## File Map

**Create**

- `src/minigpt/run_provenance.py`: environment, Git identity, and multi-process run journal.
- `src/minigpt/training_report.py`: strict source parsing, validation, summaries, plots, Markdown,
  and manifest generation.
- `report_training.py`: typed root CLI for report generation.
- `tests/test_run_provenance.py`: provenance and resume-journal contracts.
- `tests/test_training_report.py`: source validation and generated-artifact contracts.
- `configs/char_gpt_reference.yaml`: canonical CPU reference experiment selected after calibration.
- `docs/results/reference-training/*`: generated compact evidence.

**Modify**

- `src/minigpt/checkpoint.py`: public read-only checkpoint metadata.
- `train.py`: optional `--provenance` journal integration.
- `pyproject.toml`: optional `report` dependency and test-time matplotlib.
- `README.md`: link only to verified generated reference evidence.
- `tests/test_checkpoint.py`: metadata matrix.
- `tests/test_trainer.py`: provenance CLI smoke coverage.

---

### Task 1: Expose Typed Read-Only Checkpoint Metadata

**Files:**

- Modify: `src/minigpt/checkpoint.py`
- Modify: `tests/test_checkpoint.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    format_version: int
    completed_step: int
    config: ExperimentConfig
    dataset_fingerprints: DatasetFingerprints


def load_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    """Load validated v2 identity and completion metadata without restoring mutable state."""
```

- v1 metadata requests raise `LegacyCheckpointResumeError`; report evidence is intentionally v2
  only.

- [ ] **Step 1: Write failing metadata tests**

Add BDD tests proving a saved v2 checkpoint returns version `2`, its exact completed step, resolved
config, and all three fingerprints. Add a v1 test:

```python
with pytest.raises(LegacyCheckpointResumeError):
    _ = checkpoint.load_checkpoint_metadata(legacy_path)
```

- [ ] **Step 2: Run the red tests**

```powershell
.\.venv\Scripts\pytest.exe tests/test_checkpoint.py -q
```

Expected: import/attribute failure for `load_checkpoint_metadata`.

- [ ] **Step 3: Implement through the existing versioned parser**

Use `_load_versioned_payload()` so metadata validation cannot drift from config/model loading:

```python
def load_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    payload = _load_versioned_payload(path)
    if payload["format_version"] == _LEGACY_CHECKPOINT_FORMAT_VERSION:
        raise LegacyCheckpointResumeError(path)
    fingerprints = payload["dataset_fingerprints"]
    return CheckpointMetadata(
        format_version=payload["format_version"],
        completed_step=payload["completed_step"],
        config=parse_experiment_config(payload["config_yaml"], path),
        dataset_fingerprints=DatasetFingerprints(
            tokenizer_sha256=fingerprints["tokenizer_sha256"],
            train_sha256=fingerprints["train_sha256"],
            val_sha256=fingerprints["val_sha256"],
        ),
    )
```

- [ ] **Step 4: Run focused gates**

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/checkpoint.py tests/test_checkpoint.py
.\.venv\Scripts\ruff.exe check src/minigpt/checkpoint.py tests/test_checkpoint.py
.\.venv\Scripts\basedpyright.exe src/minigpt/checkpoint.py tests/test_checkpoint.py
.\.venv\Scripts\pytest.exe tests/test_checkpoint.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add src/minigpt/checkpoint.py tests/test_checkpoint.py
git commit -m "公开只读 checkpoint v2 元数据"
```

---

### Task 2: Record Clean, Multi-Segment Run Provenance

**Files:**

- Create: `src/minigpt/run_provenance.py`
- Create: `tests/test_run_provenance.py`
- Modify: `train.py`
- Modify: `tests/test_trainer.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True, slots=True)
class GitIdentity:
    commit_sha: str
    branch: str
    dirty: bool


@dataclass(frozen=True, slots=True)
class RunSegment:
    started_at_utc: str
    ended_at_utc: str | None
    status: Literal["running", "completed", "failed"]
    argv: tuple[str, ...]
    run_until_step: int | None
    resume_checkpoint_sha256: str | None
    final_completed_step: int | None


def begin_run_segment(
    path: Path,
    *,
    repository_root: Path,
    config_path: Path,
    config: ExperimentConfig,
    argv: tuple[str, ...],
    run_until_step: int | None,
    resume_path: Path | None,
) -> RunSegment:
    """Validate clean provenance and atomically record a running segment."""


def complete_run_segment(
    path: Path,
    *,
    segment: RunSegment,
    checkpoint_path: Path,
    final_step: int,
) -> None:
    """Replace the running segment with validated completion metadata."""


def fail_run_segment(path: Path, *, segment: RunSegment) -> None:
    """Record an ended failed segment without claiming a completed step."""
```

- `train.py` adds:

```text
--provenance PATH
```

Normal invocations without this option remain unchanged.

- [ ] **Step 1: Add failing pure provenance tests**

Use a temporary local Git repository initialized with `git init`, one committed config, and
byte-sized tokenizer/train/val fixtures. Test:

- environment values serialize with schema version `1`;
- Windows/Linux-independent core counts and versions have concrete types;
- the first segment records `dirty: false`, start time, argv, data hashes, and source config hash;
- a dirty repository raises `DirtyReferenceRunError`;
- resume rejects a changed commit/config/data identity;
- completion records checkpoint SHA, resolved config SHA, and final step;
- a second completed segment with `resume_checkpoint_sha256` proves resume;
- a failed segment has status `failed`, an end time, and no final completed step.

- [ ] **Step 2: Add failing training CLI test**

Run the existing tiny experiment with:

```python
[
    sys.executable,
    "train.py",
    "--config",
    str(config_path),
    "--run-until-step",
    "1",
    "--provenance",
    str(provenance_path),
]
```

Assert exit zero and one completed provenance segment.

- [ ] **Step 3: Confirm red**

```powershell
.\.venv\Scripts\pytest.exe tests/test_run_provenance.py tests/test_trainer.py -q
```

Expected: missing module and unknown CLI option.

- [ ] **Step 4: Implement stable identity helpers**

Implement streaming SHA-256, repository-relative POSIX paths, UTC timestamps ending in `Z`, and
atomic JSON replacement. CPU name resolution order is:

1. Windows registry `ProcessorNameString`;
2. Linux `/proc/cpuinfo` `model name`;
3. `platform.processor()`;
4. `"unknown"`.

Invoke Git without a shell:

```python
subprocess.run(
    ["git", "-C", str(repository_root), "status", "--porcelain"],
    check=True,
    capture_output=True,
    text=True,
)
```

Parse every JSON field through explicit runtime validators rather than casting untrusted JSON.

- [ ] **Step 5: Integrate the optional CLI journal**

In `train.main()`:

```python
segment = None
if provenance_path is not None:
    segment = begin_run_segment(...)
try:
    result = run_training(...)
except Exception:
    if segment is not None:
        fail_run_segment(provenance_path, segment=segment)
    raise
if segment is not None:
    complete_run_segment(
        provenance_path,
        segment=segment,
        checkpoint_path=result.checkpoint_path,
        final_step=result.final_step,
    )
```

Record `argv` as a JSON array, not a reconstructed shell string.

- [ ] **Step 6: Run focused gates**

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/run_provenance.py train.py tests/test_run_provenance.py tests/test_trainer.py
.\.venv\Scripts\ruff.exe check src/minigpt/run_provenance.py train.py tests/test_run_provenance.py tests/test_trainer.py
.\.venv\Scripts\basedpyright.exe src/minigpt/run_provenance.py train.py tests/test_run_provenance.py tests/test_trainer.py
.\.venv\Scripts\pytest.exe tests/test_run_provenance.py tests/test_trainer.py tests/test_checkpoint.py -q
```

- [ ] **Step 7: Commit**

```powershell
git add src/minigpt/run_provenance.py train.py tests/test_run_provenance.py tests/test_trainer.py
git commit -m "记录参考训练环境与分段恢复来源"
```

---

### Task 3: Parse and Validate Training Evidence

**Files:**

- Create: `src/minigpt/training_report.py`
- Create: `tests/test_training_report.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True, slots=True)
class MetricRecord:
    step: int
    train_loss: float
    val_loss: float | None
    learning_rate: float
    step_time_ms: float
    tokens_per_sec: float
    data_time_ms: float
    forward_backward_time_ms: float
    optimizer_time_ms: float
    cpu_memory_mb: float


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    step: int
    text: str


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    total_steps: int
    initial_train_loss: float
    final_train_loss: float
    best_train_loss: float
    initial_val_loss: float
    final_val_loss: float
    best_val_loss: float
    median_tokens_per_sec: float
    peak_cpu_memory_mb: float


def load_metric_records(path: Path) -> tuple[MetricRecord, ...]: ...
def load_generated_samples(path: Path) -> tuple[GeneratedSample, ...]: ...
def validate_reference_sources(inputs: ReportInputs) -> ValidatedReportData: ...
```

- [ ] **Step 1: Add failing parser tests**

Construct JSONL records with sparse validation points. Assert:

```python
assert [item.step for item in records] == [0, 1, 2]
assert [(item.step, item.val_loss) for item in records if item.val_loss is not None] == [
    (1, 3.5)
]
```

Add failures for malformed JSON, missing/extra fields, booleans passed as integers, NaN/Inf, duplicate
steps, gaps, and nonzero starting step.

Parse samples in the exact trainer format:

```text
step=1
first text

step=3
middle text

step=5
final text

```

Assert representative indices are `0`, `len(samples) // 2`, and `len(samples) - 1`.

- [ ] **Step 2: Add failing cross-artifact validation tests**

Build a tiny real v2 checkpoint plus matching provenance. Test rejection for:

- v1 checkpoint;
- final completed step below `max_steps - 1`;
- metrics count/step mismatch;
- data fingerprint mismatch;
- resolved config hash mismatch;
- learning rate differing from `learning_rate_at_step()` by more than `1e-12` relative tolerance;
- no completed resume segment;
- sample interval not producing a final scheduled sample.

- [ ] **Step 3: Confirm red**

```powershell
.\.venv\Scripts\pytest.exe tests/test_training_report.py -q
```

Expected: missing report API.

- [ ] **Step 4: Implement strict parsing and summaries**

Use `json.loads()` only at one boundary and validate every field. Calculate:

```python
TrainingSummary(
    total_steps=len(records),
    initial_train_loss=records[0].train_loss,
    final_train_loss=records[-1].train_loss,
    best_train_loss=min(item.train_loss for item in records),
    initial_val_loss=validation_records[0].val_loss,
    final_val_loss=validation_records[-1].val_loss,
    best_val_loss=min(item.val_loss for item in validation_records),
    median_tokens_per_sec=statistics.median(item.tokens_per_sec for item in records),
    peak_cpu_memory_mb=max(item.cpu_memory_mb for item in records),
)
```

Validation loss expressions must be narrowed to `float` before aggregation.
Character-level perplexity is `math.exp(loss)` and is displayed only for finite loss.

- [ ] **Step 5: Implement cross-artifact checks**

Resolve the source config with `CharTokenizer`, compare its canonical YAML SHA-256 with checkpoint
metadata and provenance, call `compute_dataset_fingerprints()`, and recompute every learning rate
from:

```python
learning_rate_at_step(
    record.step,
    max_learning_rate=config.optimizer.learning_rate,
    min_learning_rate=config.optimizer.min_learning_rate,
    warmup_steps=config.training.warmup_steps,
    lr_decay_steps=config.training.lr_decay_steps,
)
```

Read the current repository HEAD/status before generating output and require the same commit SHA
captured by provenance plus a clean tree. Count train/validation tokens directly from the `.npy`
arrays for the generated dataset section; do not hand-enter corpus sizes.

Run all validation before creating the destination directory.

- [ ] **Step 6: Run focused gates**

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/training_report.py tests/test_training_report.py
.\.venv\Scripts\ruff.exe check src/minigpt/training_report.py tests/test_training_report.py
.\.venv\Scripts\basedpyright.exe src/minigpt/training_report.py tests/test_training_report.py
.\.venv\Scripts\pytest.exe tests/test_training_report.py tests/test_checkpoint.py -q
```

- [ ] **Step 7: Commit**

```powershell
git add src/minigpt/training_report.py tests/test_training_report.py
git commit -m "校验参考训练指标与证据一致性"
```

---

### Task 4: Generate the Compact Evidence Package

**Files:**

- Modify: `src/minigpt/training_report.py`
- Create: `report_training.py`
- Modify: `tests/test_training_report.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    readme: Path
    environment: Path
    resolved_config: Path
    metrics_csv: Path
    loss_curve: Path
    learning_rate_curve: Path
    throughput_curve: Path
    generated_samples: Path
    manifest: Path


def generate_training_report(inputs: ReportInputs, output_dir: Path) -> ReportArtifacts:
    """Validate sources and atomically generate the complete evidence package."""
```

- Root CLI:

```text
python report_training.py
  --config PATH
  --metrics PATH
  --samples PATH
  --checkpoint PATH
  --provenance PATH
  --output-dir PATH
```

- [ ] **Step 1: Add optional dependencies**

In `pyproject.toml`:

```toml
[project.optional-dependencies]
report = [
    "matplotlib>=3.10",
]
dev = [
    "basedpyright>=1.39.9",
    "matplotlib>=3.10",
    "pytest>=8.0",
    "ruff>=0.15.20",
]
```

Reinstall and verify:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,report]"
.\.venv\Scripts\python.exe -m pip check
```

- [ ] **Step 2: Add failing artifact tests**

For a tiny valid source set, assert all nine required artifacts exist. Parse CSV back and assert
every source float is preserved. Parse environment/manifest JSON and assert required keys, relative
paths, byte sizes, and SHA-256 values.

Assert a pre-existing output directory raises `ReportOutputExistsError` instead of mixing results.

- [ ] **Step 3: Add failing series/render tests**

Expose pure series helpers and assert:

- train-loss points include all steps;
- validation points contain only observed steps;
- LR and throughput points preserve raw values;
- titles include experiment name and short Git SHA.

Do not compare PNG bytes.

- [ ] **Step 4: Confirm red**

```powershell
.\.venv\Scripts\pytest.exe tests/test_training_report.py -q
```

- [ ] **Step 5: Implement CSV and fixed figures**

Use `matplotlib.use("Agg")` before importing pyplot. Use fixed `figsize=(12, 6.75)`, `dpi=120`, no
smoothing, validation scatter only, and fixed metadata:

```python
figure.savefig(
    path,
    dpi=120,
    metadata={"Software": f"miniGPT {git_short_sha}"},
)
```

CSV uses `lineterminator="\n"`.

- [ ] **Step 6: Implement environment, samples, Markdown, and manifest**

Reconstruct an untrained baseline by calling `seed_everything()`, constructing `GPT` from the
resolved config, switching to `eval()`, and passing `create_sample_generator(seed)` to
`GPT.generate()`.

Copy the complete source samples bytes to `generated_samples.txt`. Markdown representative samples
use fixed first/middle/final positions. Render every numeric conclusion from `TrainingSummary`.

Write generated files in a temporary sibling directory. Generate the manifest last, excluding
itself explicitly, then rename the complete temporary directory to the absent destination.

- [ ] **Step 7: Implement the typed root CLI**

Follow existing argparse/cast boundaries. On success print the report and manifest paths. On invalid
evidence, allow the dedicated exception message to identify the exact artifact/field.

- [ ] **Step 8: Verify regeneration**

Generate into two temporary output directories from the same fixtures. Assert byte equality for
README, environment JSON, resolved YAML, CSV, and samples. Assert the parsed numerical series for
each pair of figures are identical through the pure series helpers.

- [ ] **Step 9: Run focused gates**

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/training_report.py report_training.py tests/test_training_report.py
.\.venv\Scripts\ruff.exe check src/minigpt/training_report.py report_training.py tests/test_training_report.py
.\.venv\Scripts\basedpyright.exe src/minigpt/training_report.py report_training.py tests/test_training_report.py
.\.venv\Scripts\pytest.exe tests/test_training_report.py tests/test_run_provenance.py -q
```

- [ ] **Step 10: Commit**

```powershell
git add pyproject.toml src/minigpt/training_report.py report_training.py tests/test_training_report.py
git commit -m "生成可复现的参考训练证据包"
```

---

### Task 5: Calibrate and Fix the Canonical Reference Configuration

**Files:**

- Create locally, do not commit: `outputs/reference-calibration.yaml`
- Create: `configs/char_gpt_reference.yaml`

**Interfaces:**

- Consumes the unchanged architecture and batch dimensions from `configs/char_gpt.yaml`.
- Produces a fixed full experiment config for Task 6.

- [ ] **Step 1: Prepare Tiny Shakespeare**

```powershell
.\.venv\Scripts\python.exe prepare_data.py --data-dir data
```

Inspect `data/processed/metadata.json`, file sizes, tokenizer vocabulary size, and the three Stage 6
fingerprints.

- [ ] **Step 2: Create the ignored calibration config**

Use the canonical architecture with:

```yaml
training:
  max_steps: 120
  warmup_steps: 10
  lr_decay_steps: 120
  eval_interval: 20
  eval_batches: 10
  log_interval: 10
  checkpoint_interval: 40
  sample_interval: 40
  sample_tokens: 200
  output_dir: outputs/reference-calibration
  checkpoint_dir: checkpoints/reference-calibration
  tensorboard_dir: outputs/reference-calibration/tensorboard
```

- [ ] **Step 3: Run calibration from fresh ignored outputs**

Verify the exact targets are under the repository's ignored output/checkpoint roots, then remove
only those two calibration artifact directories. Keep the separate
`outputs/reference-calibration.yaml` config and run:

```powershell
.\.venv\Scripts\python.exe train.py --config outputs/reference-calibration.yaml
```

- [ ] **Step 4: Calculate calibration evidence**

From `metrics.jsonl`, report:

- step 0, final, and best train loss;
- all observed validation losses;
- median tokens/s, step time, and RSS over steps `10..119`;
- total elapsed process time;
- sample count and final calibration sample step;
- v2 checkpoint version, `completed_step == 119`, and successful model/config loading.

Reject calibration if any value is non-finite, train loss does not decrease, fewer than two
validation observations exist, samples are missing, or checkpoint loading fails.

- [ ] **Step 5: Select the formal horizon mechanically**

Let `stable_median_step_ms` be the median over steps `10..119`. Select:

```text
estimated_steps_for_10_minutes = floor(600_000 / stable_median_step_ms)
max_steps = clamp(round_down_to_multiple(estimated_steps_for_10_minutes, 100), 2_000, 5_000)
warmup_steps = max_steps // 20
lr_decay_steps = max_steps
eval_interval = max_steps // 20
checkpoint_interval = max_steps // 10
sample_interval = max_steps // 10
```

Because `max_steps` is a multiple of 100, each interval is integral and `sample_interval` divides
the full horizon. Keep at least 20 validation observations and 10 scheduled samples.

- [ ] **Step 6: Create and validate `configs/char_gpt_reference.yaml`**

Use output roots:

```yaml
output_dir: outputs/reference
checkpoint_dir: checkpoints/reference
tensorboard_dir: outputs/reference/tensorboard
```

Load the config through `load_experiment_config()`, resolve the tokenizer vocabulary, and assert the
schedule constraint:

```text
0 <= warmup_steps < lr_decay_steps <= max_steps
```

- [ ] **Step 7: Add a config contract test**

In `tests/test_training_components.py`, load the canonical config and assert the fixed architecture,
batch dimensions, output roots, `lr_decay_steps == max_steps`, and divisibility of eval/checkpoint/
sample intervals.

- [ ] **Step 8: Run gates and commit only the canonical config/test**

```powershell
.\.venv\Scripts\ruff.exe format tests/test_training_components.py
.\.venv\Scripts\ruff.exe check tests/test_training_components.py
.\.venv\Scripts\basedpyright.exe tests/test_training_components.py
.\.venv\Scripts\pytest.exe tests/test_training_components.py -q
git status --short
git add configs/char_gpt_reference.yaml tests/test_training_components.py
git commit -m "固化 CPU 参考训练实验配置"
```

Confirm calibration outputs remain ignored and unstaged.

---

### Task 6: Execute the Clean Formal Training With Real Resume

**Files:**

- Create ignored: `outputs/reference/*`
- Create ignored: `checkpoints/reference/latest.pt`

**Interfaces:**

- Consumes the committed canonical config and provenance-aware training CLI.
- Produces complete raw evidence for Task 7.

- [ ] **Step 1: Run full source gates and confirm cleanliness**

```powershell
.\.venv\Scripts\ruff.exe format --check src tests
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\basedpyright.exe
.\.venv\Scripts\pytest.exe
git status --short
```

Expected: all gates green and no output from `git status --short`.

- [ ] **Step 2: Isolate previous formal artifacts**

Resolve and verify these exact absolute targets remain under repository `outputs/reference` and
`checkpoints/reference`, then remove only those directories. Do not remove `data/`.

- [ ] **Step 3: Choose the interruption point**

With canonical `N = max_steps` and `E = eval_interval`, set:

```text
K = N // 2 + E // 2
```

Assert `0 < K < N` and `K % E != 0`, so normal exit proves exit-only checkpoint behavior at a
non-evaluation/sample boundary.

- [ ] **Step 4: Run the first formal segment**

```powershell
$referenceStop = .\.venv\Scripts\python.exe -c "from pathlib import Path; from minigpt.config import load_experiment_config; c = load_experiment_config(Path('configs/char_gpt_reference.yaml')); print(c.training.max_steps // 2 + c.training.eval_interval // 2)"
.\.venv\Scripts\python.exe train.py `
  --config configs/char_gpt_reference.yaml `
  --run-until-step $referenceStop `
  --provenance outputs/reference/run_provenance.json
```

Record wall-clock elapsed time from the provenance segment, not from a hand-entered value. Verify
checkpoint metadata reports `next_step == K`.

- [ ] **Step 5: Resume in a second process**

```powershell
.\.venv\Scripts\python.exe train.py `
  --config configs/char_gpt_reference.yaml `
  --resume checkpoints/reference/latest.pt `
  --provenance outputs/reference/run_provenance.json
```

- [ ] **Step 6: Validate raw completion**

Require:

- final checkpoint format `2` and `completed_step == N - 1`;
- checkpoint SHA-256 and size recorded;
- metrics steps equal `list(range(N))`;
- every numeric value finite;
- LR recomputation matches;
- train loss materially below initial loss;
- validation results reported without requiring improvement;
- provenance has exactly two completed segments and the second has a resume checkpoint hash;
- samples contain exactly `N / sample_interval` records and the last is at step `N - 1`;
- Git SHA/config/data fingerprints match both segments and final checkpoint.

Do not edit raw metrics or samples.

---

### Task 7: Generate and Publish the Reference Evidence

**Files:**

- Create: `docs/results/reference-training/README.md`
- Create: `docs/results/reference-training/environment.json`
- Create: `docs/results/reference-training/resolved_config.yaml`
- Create: `docs/results/reference-training/metrics.csv`
- Create: `docs/results/reference-training/loss_curve.png`
- Create: `docs/results/reference-training/learning_rate_curve.png`
- Create: `docs/results/reference-training/throughput_curve.png`
- Create: `docs/results/reference-training/generated_samples.txt`
- Create: `docs/results/reference-training/artifact_manifest.json`
- Modify: `README.md`

**Interfaces:**

- Consumes only validated Task 6 source artifacts.
- Produces the final compact, reviewable evidence package.

- [ ] **Step 1: Generate the report once**

```powershell
.\.venv\Scripts\python.exe report_training.py `
  --config configs/char_gpt_reference.yaml `
  --metrics outputs/reference/metrics.jsonl `
  --samples outputs/reference/samples.txt `
  --checkpoint checkpoints/reference/latest.pt `
  --provenance outputs/reference/run_provenance.json `
  --output-dir docs/results/reference-training
```

- [ ] **Step 2: Regenerate independently**

Generate from the same sources into `outputs/reference-report-regeneration`. Compare bytes for
README, environment JSON, resolved config, metrics CSV, and generated samples. Parse both manifests
and assert all numerical/source identities match. Confirm every required PNG exists and is nonzero.

- [ ] **Step 3: Inspect figures visually**

Open all three PNGs and check labels, legend, title, sparse validation points, non-clipped text, and
readability. Do not change source data to improve appearance.

- [ ] **Step 4: Verify manifest hashes**

Recompute SHA-256 and size for every listed source and generated artifact. Confirm paths are
repository-relative and contain no drive letters or user profile. Compute and record the manifest's
own SHA-256 separately for the final handoff.

- [ ] **Step 5: Add a concise repository README link**

Link `docs/results/reference-training/README.md` and summarize only mechanically verified facts:
model dimensions, parameter count, steps, train/validation outcome, resume status, and limitations.
Do not manually duplicate a table of step-level metrics.

- [ ] **Step 6: Run report and README tests**

```powershell
.\.venv\Scripts\pytest.exe tests/test_training_report.py tests/test_readme.py -q
```

- [ ] **Step 7: Commit compact results**

Before staging, prove no checkpoint, outputs, TensorBoard events, data, or provenance sidecar are
untracked. Then:

```powershell
git add README.md docs/results/reference-training
git diff --staged --check
git commit -m "发布 CPU 参考训练证据"
```

---

### Task 8: Final Verification and Handoff

**Files:** No new implementation files.

- [ ] **Step 1: Run required gates in order**

```powershell
.\.venv\Scripts\ruff.exe format src tests
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\basedpyright.exe
.\.venv\Scripts\pytest.exe
```

- [ ] **Step 2: Verify report from source one final time**

Run report generation into a fresh ignored directory and compare numerical conclusions and source
hashes to the committed report.

- [ ] **Step 3: Inspect repository state**

```powershell
git diff --check
git status --short --branch
git log --oneline --decorate -15
```

Expected: clean Stage 7A branch with local commits only and ignored raw training artifacts.

- [ ] **Step 4: Report evidence**

Provide:

- canonical config and exact two training commands;
- calibration observations and the mechanical horizon calculation;
- formal initial/final/best losses and perplexities;
- validation outcome;
- median tokens/s, peak RSS, and elapsed time explicitly labeled non-benchmark telemetry;
- model parameter count;
- Git SHA, data fingerprints, resolved config hash, checkpoint SHA/size, and manifest SHA;
- generated sample selection rule and honest qualitative assessment;
- full gate outputs and commit list;
- remaining limitations and a Stage 7B Benchmark v2 proposal without implementation.
