# Stage 6 Correctness and Portability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make miniGPT numerically trustworthy across planned interruption/resume, isolate sampling from training randomness, restore idiomatic PyTorch module calls, and support Python 3.11 through 3.14 with cross-platform CI.

**Architecture:** Use explicit checkpoint v1/v2 dispatch, with v2 carrying full training state, independent sample-generator state, and SHA-256 dataset identity. Keep the complete experiment in `ExperimentConfig`, pass the exclusive process boundary separately as `run_until_step`, and drive validation, sampling, and periodic checkpointing only from absolute completed-step counts. Validate all immutable resume inputs before mutating model, optimizer, or RNG state.

**Tech Stack:** Python 3.11–3.14, PyTorch CPU, NumPy, PyYAML, TensorBoard, pytest, Ruff, basedpyright, GitHub Actions, PowerShell.

## Global Constraints

- Support exactly `requires-python = ">=3.11,<3.15"`.
- Keep Ruff `select = ["ALL"]` and basedpyright `typeCheckingMode = "all"`.
- Do not add `type: ignore`, `pyright: ignore`, broad `Any`, CUDA assumptions, or framework dependencies.
- Preserve BDD Given/When/Then comments and full test annotations.
- Network-sensitive tests must use local `file://` fixtures.
- v1 checkpoints support config/model/inference only and must never resume training.
- `max_steps` is the complete experiment; `run_until_step` is an exclusive process boundary outside the config.
- Validation and sampling are scheduled only by absolute step; process exit may force only a checkpoint.
- Each logical task must pass its focused Ruff, basedpyright, and pytest checks before a Chinese local commit.
- Do not push.

---

## File Map

**Create**

- `.github/workflows/quality.yml`: Windows 3.14 and Linux 3.11 quality gates.
- `docs/superpowers/plans/2026-07-27-stage6-correctness-portability.md`: this plan.

**Modify — model and runtime**

- `src/minigpt/layers.py`: idiomatic submodule calls.
- `src/minigpt/model.py`: idiomatic block calls, explicit unexpected-block error, generator-aware sampling.
- `src/minigpt/optimization.py`: Python-compatible imports, schedule horizon, sample generator construction.
- `src/minigpt/training_runtime.py`: model calls, fingerprints, sample generator ownership, scheduled validation.
- `src/minigpt/trainer.py`: process boundary, event ordering, sample generator use, exit checkpoint.
- `src/minigpt/benchmark_workload.py`: idiomatic model calls.

**Modify — configuration and persistence**

- `src/minigpt/settings.py`: optimizer type and `lr_decay_steps`.
- `src/minigpt/config.py`: Python 3.11 aliases, new fields, legacy v1 normalization.
- `src/minigpt/checkpoint.py`: v1/v2 payload dispatch, fingerprints, strict resume validation.
- `src/minigpt/data.py`, `src/minigpt/metrics.py`, `src/minigpt/benchmark_config.py`,
  `src/minigpt/benchmark_types.py`, `src/minigpt/batching.py`: Python 3.11 aliases/imports.
- `configs/char_gpt.yaml`, `configs/char_gpt_smoke.yaml`: explicit AdamW type and LR horizon.
- `train.py`: replace `--max-steps` with `--run-until-step`.
- `generate.py`: local generation RNG.
- `pyproject.toml`: Python range, typing-extensions, static-tool targets.

**Modify — tests and documentation**

- `tests/test_model.py`: hooks, unexpected block, generator isolation.
- `tests/test_checkpoint.py`: v1 inference matrix, v2 state/fingerprint validation.
- `tests/test_trainer.py`: run boundary, sampling independence, exact resume.
- `tests/test_training_components.py`: config and schedule semantics.
- `tests/test_data.py`, `tests/test_benchmark.py`: Python 3.11 aliases and module calls.
- `tests/test_readme.py`: current compatibility/resume contracts.
- `README.md`: supported versions, checkpoint matrix, run boundary, exact-resume evidence.

---

### Task 1: Restore Idiomatic PyTorch Module Calls

**Files:**

- Modify: `tests/test_model.py`
- Modify: `src/minigpt/layers.py`
- Modify: `src/minigpt/model.py`
- Modify: `src/minigpt/training_runtime.py`
- Modify: `src/minigpt/benchmark_workload.py`

**Interfaces:**

- Produces: `UnexpectedTransformerBlockError(index: int, actual_type: str)`.
- Preserves: `GPT.forward(Tensor, Tensor | None) -> tuple[Tensor, Tensor | None]`.
- Requires all model execution to pass through `nn.Module.__call__`.

- [ ] **Step 1: Add failing hook and unexpected-block tests**

Add to `tests/test_model.py`:

```python
from torch import Tensor, nn


def test_gpt_calls_transformer_block_through_module_protocol() -> None:
    # Given: a GPT block with a registered forward hook.
    gpt = model.GPT(tiny_config())
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    hook_outputs: list[Tensor] = []

    def record_output(
        module: nn.Module,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        del module, inputs
        hook_outputs.append(output)

    handle = gpt.blocks[0].register_forward_hook(record_output)
    try:
        # When: execution enters through GPT.__call__.
        _ = gpt(token_ids)
    finally:
        handle.remove()

    # Then: the nested block hook observes its output.
    assert len(hook_outputs) == 1
    assert hook_outputs[0].shape == (1, 4, 8)


def test_gpt_rejects_unexpected_module_list_entry() -> None:
    # Given: a GPT whose first block was replaced by an unrelated module.
    gpt = model.GPT(tiny_config())
    gpt.blocks[0] = nn.Identity()
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    # When/Then: the invalid architecture fails instead of silently skipping the entry.
    with pytest.raises(model.UnexpectedTransformerBlockError, match=r"block 0.*Identity"):
        _ = gpt(token_ids)
```

- [ ] **Step 2: Run the focused tests and confirm red state**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_model.py::test_gpt_calls_transformer_block_through_module_protocol tests/test_model.py::test_gpt_rejects_unexpected_module_list_entry -q
```

Expected:

- hook test fails because `module.forward(...)` bypasses the hook;
- unexpected-block test fails because the module is silently skipped or the exception is missing.

- [ ] **Step 3: Implement explicit block validation and module calls**

In `src/minigpt/model.py`, add:

```python
@dataclass(frozen=True, slots=True)
class UnexpectedTransformerBlockError(RuntimeError):
    """Report an unexpected module in the Transformer stack."""

    index: int
    actual_type: str

    @override
    def __str__(self) -> str:
        """Render the invalid block position and type."""
        return f"unexpected GPT block {self.index}: expected TransformerBlock, got {self.actual_type}"
```

Replace the block loop with:

```python
for index, block in enumerate(self.blocks):
    if not isinstance(block, TransformerBlock):
        raise UnexpectedTransformerBlockError(index, type(block).__name__)
    hidden_states = block(hidden_states)
```

Replace every direct `.forward(...)` call in production and tests with `module(...)`, including
linear, dropout, normalization, GPT, training runtime, benchmark workload, and `GPT.generate`.

- [ ] **Step 4: Verify no direct forward calls remain**

Run:

```powershell
rg "\.forward\(" src tests
```

Expected: no matches.

- [ ] **Step 5: Run focused quality gates**

Run:

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/layers.py src/minigpt/model.py src/minigpt/training_runtime.py src/minigpt/benchmark_workload.py tests/test_model.py
.\.venv\Scripts\ruff.exe check src/minigpt/layers.py src/minigpt/model.py src/minigpt/training_runtime.py src/minigpt/benchmark_workload.py tests/test_model.py
.\.venv\Scripts\basedpyright.exe src/minigpt/layers.py src/minigpt/model.py src/minigpt/training_runtime.py src/minigpt/benchmark_workload.py tests/test_model.py
.\.venv\Scripts\pytest.exe tests/test_model.py tests/test_benchmark.py tests/test_trainer.py -q
```

Expected: zero static errors and all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
git add -- src/minigpt/layers.py src/minigpt/model.py src/minigpt/training_runtime.py src/minigpt/benchmark_workload.py tests/test_model.py
git diff --staged --check
git commit -m "恢复 PyTorch 模块惯用调用语义"
```

---

### Task 2: Add Python 3.11 Compatibility and Explicit Schedule Configuration

**Files:**

- Modify: `pyproject.toml`
- Modify: `src/minigpt/settings.py`
- Modify: `src/minigpt/config.py`
- Modify: `src/minigpt/optimization.py`
- Modify: all Python files containing PEP 695 aliases or `typing.override`
- Modify: `configs/char_gpt.yaml`
- Modify: `configs/char_gpt_smoke.yaml`
- Modify: tests constructing `OptimizerSettings` or `TrainingSettings`
- Test: `tests/test_training_components.py`

**Interfaces:**

- Produces: `OptimizerSettings.optimizer_type: Literal["adamw"]`.
- Produces: `TrainingSettings.lr_decay_steps: int`.
- Changes:

```python
learning_rate_at_step(
    step: int,
    *,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    lr_decay_steps: int,
) -> float
```

- Produces: `parse_legacy_experiment_config(yaml_text: str, source: Path) -> ExperimentConfig`.

- [ ] **Step 1: Add failing config and schedule tests**

Update the YAML fixture in `tests/test_training_components.py`:

```yaml
optimizer:
  type: adamw
  learning_rate: 0.001
training:
  max_steps: 10
  warmup_steps: 2
  lr_decay_steps: 8
```

Assert:

```python
assert experiment.optimizer.optimizer_type == "adamw"
assert experiment.training.lr_decay_steps == 8
```

Change the schedule test to calculate steps `0..9` with `lr_decay_steps=6` and assert:

```python
assert isclose(learning_rates[5], 0.1)
assert learning_rates[6:] == [0.1, 0.1, 0.1, 0.1]
```

Add invalid-config cases proving:

```text
warmup_steps >= lr_decay_steps -> InvalidExperimentConfigError
lr_decay_steps > max_steps -> InvalidExperimentConfigError
optimizer.type != adamw -> InvalidExperimentConfigError
```

- [ ] **Step 2: Run focused tests and confirm red state**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_training_components.py -q
```

Expected: parsing fails because the new fields and schedule signature do not exist.

- [ ] **Step 3: Implement settings and parser changes**

Add fields:

```python
@dataclass(frozen=True, slots=True)
class OptimizerSettings:
    optimizer_type: Literal["adamw"]
    learning_rate: float
    # existing fields...


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    max_steps: int
    warmup_steps: int
    lr_decay_steps: int
    # existing fields...
```

Parse YAML key `optimizer.type`, serialize it back as `type`, parse `training.lr_decay_steps`, and
validate:

```python
if optimizer_type != "adamw":
    raise InvalidExperimentConfigError(source, "optimizer.type must be adamw")
if not 0 <= warmup_steps < lr_decay_steps <= max_steps:
    raise InvalidExperimentConfigError(
        source,
        "training steps must satisfy 0 <= warmup_steps < lr_decay_steps <= max_steps",
    )
```

Add a legacy parser used only for v1 inference. It injects missing YAML keys in memory before using
the strict parser:

```python
def parse_legacy_experiment_config(yaml_text: str, source: Path) -> ExperimentConfig:
    """Parse a v1 inference config with deterministic compatibility defaults."""
```

Its only defaults are:

```text
optimizer.type = adamw
training.lr_decay_steps = training.max_steps
```

- [ ] **Step 4: Make the full source tree Python 3.11-parseable**

In `pyproject.toml`:

```toml
requires-python = ">=3.11,<3.15"
dependencies = [
    # existing dependencies...
    "typing-extensions>=4.12",
]

[tool.basedpyright]
pythonVersion = "3.11"

[tool.ruff]
target-version = "py311"
```

Replace each PEP 695 alias:

```python
from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
```

Import `override` from `typing_extensions`, not `typing`. Move `None` to the end of unions flagged by
RUF036. Update every settings constructor in tests and benchmark code with
`optimizer_type="adamw"` and `lr_decay_steps=max_steps` or the intended horizon.

- [ ] **Step 5: Reinstall and run compatibility-oriented gates**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe format src tests
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\basedpyright.exe
.\.venv\Scripts\pytest.exe tests/test_training_components.py tests/test_checkpoint.py tests/test_trainer.py tests/test_benchmark.py -q
```

Expected: pip is consistent, Ruff and basedpyright report zero errors, and selected tests pass on
local Python 3.14. Python 3.11 execution is deferred to Task 7 CI.

- [ ] **Step 6: Commit**

```powershell
git add -- pyproject.toml configs src tests
git diff --staged --check
git commit -m "支持 Python 3.11 至 3.14 并分离调度周期"
```

---

### Task 3: Make Text Generation Use an Explicit Generator

**Files:**

- Modify: `tests/test_model.py`
- Modify: `src/minigpt/model.py`
- Modify: `src/minigpt/optimization.py`
- Modify: `generate.py`

**Interfaces:**

- Changes:

```python
GPT.generate(
    token_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor
```

- Produces: `create_sample_generator(seed: int) -> torch.Generator`.

- [ ] **Step 1: Add failing generator-isolation tests**

Add to `tests/test_model.py`:

```python
def test_generate_uses_explicit_generator_without_consuming_global_rng() -> None:
    # Given: a model, a local sample generator, and a captured global RNG state.
    _ = torch.default_generator.manual_seed(23)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    sample_generator = torch.Generator(device="cpu")
    _ = sample_generator.manual_seed(29)
    global_state = torch.get_rng_state().clone()

    # When: generation samples through the explicit generator.
    _ = gpt.generate(prompt, max_new_tokens=4, generator=sample_generator)

    # Then: global Torch randomness is unchanged.
    assert torch.equal(torch.get_rng_state(), global_state)
```

Add a second test creating two local generators with the same seed and asserting identical generated
tokens.

- [ ] **Step 2: Run tests and confirm red state**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_model.py -q
```

Expected: `GPT.generate()` rejects the unknown `generator` argument.

- [ ] **Step 3: Implement local sampling**

Pass the parameter directly:

```python
next_token = torch.multinomial(
    probabilities,
    num_samples=1,
    generator=generator,
)
```

In `src/minigpt/optimization.py`:

```python
_SAMPLE_SEED_OFFSET: Final = 2


def create_sample_generator(seed: int) -> torch.Generator:
    """Create the independent CPU generator used for observable text samples."""
    generator = torch.Generator(device="cpu")
    _ = generator.manual_seed(seed + _SAMPLE_SEED_OFFSET)
    return generator
```

In `generate.py`, replace global seeding with:

```python
generator = torch.Generator(device="cpu")
_ = generator.manual_seed(seed)
generated = model.generate(..., generator=generator)
```

- [ ] **Step 4: Run focused quality gates**

Run:

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/model.py src/minigpt/optimization.py tests/test_model.py generate.py
.\.venv\Scripts\ruff.exe check src/minigpt/model.py src/minigpt/optimization.py tests/test_model.py generate.py
.\.venv\Scripts\basedpyright.exe src/minigpt/model.py src/minigpt/optimization.py tests/test_model.py
.\.venv\Scripts\pytest.exe tests/test_model.py tests/test_trainer.py::test_generate_cli_loads_checkpoint_and_appends_text -q
```

Expected: zero static errors and all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- generate.py src/minigpt/model.py src/minigpt/optimization.py tests/test_model.py
git diff --staged --check
git commit -m "隔离文本生成随机数状态"
```

---

### Task 4: Introduce Dataset Fingerprints and Checkpoint Format v2

**Files:**

- Modify: `tests/test_checkpoint.py`
- Modify: `src/minigpt/checkpoint.py`
- Modify: `src/minigpt/training_runtime.py`
- Modify: `src/minigpt/trainer.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True, slots=True)
class DatasetFingerprints:
    tokenizer_sha256: str
    train_sha256: str
    val_sha256: str


def compute_dataset_fingerprints(config: DataSettings) -> DatasetFingerprints
```

- Extends `CheckpointResources` with:

```python
sample_generator: torch.Generator
dataset_fingerprints: DatasetFingerprints
```

- v2 `save_checkpoint()` writes `completed_step` and all required v2 fields.
- `load_checkpoint_config()` and `load_model_state()` support v1 and v2.
- `load_checkpoint()` rejects v1 with `LegacyCheckpointResumeError`.

- [ ] **Step 1: Add fingerprint and v2 round-trip tests**

Add tests that create temporary tokenizer/train/val files, call
`compute_dataset_fingerprints()`, modify one byte-bearing file at a time, and assert only its SHA
changes.

Extend the checkpoint round-trip fixture with:

```python
sample_generator = torch.Generator(device="cpu")
_ = sample_generator.manual_seed(15)
fingerprints = checkpoint.compute_dataset_fingerprints(config.data)
resources = CheckpointResources(
    gpt,
    optimizer,
    train_batcher,
    val_batcher,
    sample_generator,
    fingerprints,
)
```

After save, consume `torch.multinomial(..., generator=sample_generator)`, reload, and assert the next
sample matches.

- [ ] **Step 2: Add v1 inference-only tests**

Write a test helper that saves the current v1 shape with:

```python
legacy_payload = {
    "format_version": 1,
    "step": 0,
    "config_yaml": legacy_config_yaml,
    "model_state": gpt.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "python_random_state": random.getstate(),
    "numpy_random_state_json": legacy_numpy_state_json,
    "torch_random_state": torch.get_rng_state(),
    "train_batcher_random_state": train_batcher.capture_random_state(),
    "val_batcher_random_state": val_batcher.capture_random_state(),
}
torch.save(legacy_payload, path)
```

Verify:

- `load_checkpoint_config(path)` returns normalized AdamW and `lr_decay_steps=max_steps`;
- `load_model_state(path, fresh_model)` restores weights;
- `load_checkpoint(path, resources=...)` raises `LegacyCheckpointResumeError`;
- the exception occurs before any model parameter changes.

- [ ] **Step 3: Run checkpoint tests and confirm red state**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_checkpoint.py -q
```

Expected: missing fingerprint/sample-generator interfaces and no v1/v2 dispatch.

- [ ] **Step 4: Implement fingerprints and typed v2 payload**

Use streaming SHA-256:

```python
def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
```

Define separate `CheckpointV1Payload` and `CheckpointV2Payload` `TypedDict` types and explicit loaders:

```python
def _load_v1_payload(path: Path, loaded: StateValue) -> CheckpointV1Payload
def _load_v2_payload(path: Path, loaded: StateValue) -> CheckpointV2Payload
def _load_versioned_payload(path: Path) -> CheckpointV1Payload | CheckpointV2Payload
```

`save_checkpoint()` always writes format 2. It saves `resources.sample_generator.get_state()` and the
three fingerprint strings. `load_checkpoint()` checks version before mutating resources and restores
the local generator with `set_state()`.

- [ ] **Step 5: Wire resources through training component construction**

In `TrainingComponents`, add:

```python
sample_generator: torch.Generator
dataset_fingerprints: DatasetFingerprints
```

Construct both once in `build_training_components()` and include them in `CheckpointResources`.
Update `_append_sample()` to receive and pass the generator.

- [ ] **Step 6: Run focused quality gates**

Run:

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/checkpoint.py src/minigpt/training_runtime.py src/minigpt/trainer.py tests/test_checkpoint.py
.\.venv\Scripts\ruff.exe check src/minigpt/checkpoint.py src/minigpt/training_runtime.py src/minigpt/trainer.py tests/test_checkpoint.py
.\.venv\Scripts\basedpyright.exe src/minigpt/checkpoint.py src/minigpt/training_runtime.py src/minigpt/trainer.py tests/test_checkpoint.py
.\.venv\Scripts\pytest.exe tests/test_checkpoint.py tests/test_trainer.py -q
```

Expected: v2 state round-trip and v1 inference matrix pass with zero static errors.

- [ ] **Step 7: Commit**

```powershell
git add -- src/minigpt/checkpoint.py src/minigpt/training_runtime.py src/minigpt/trainer.py tests/test_checkpoint.py
git diff --staged --check
git commit -m "实现严格的 checkpoint v2 状态格式"
```

---

### Task 5: Validate Resume Compatibility Before State Mutation

**Files:**

- Modify: `tests/test_checkpoint.py`
- Modify: `src/minigpt/checkpoint.py`
- Modify: `src/minigpt/trainer.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True, slots=True)
class ResumeConfigMismatch:
    field: str
    checkpoint_value: str
    current_value: str


@dataclass(frozen=True, slots=True)
class IncompatibleResumeConfigError(ValueError):
    mismatches: tuple[ResumeConfigMismatch, ...]
```

- Changes:

```python
load_checkpoint(
    path: Path,
    *,
    resources: CheckpointResources,
    config: ExperimentConfig,
) -> ResumeState
```

- [ ] **Step 1: Add failing immutable-field tests**

Parameterize changes to:

```text
runtime.seed
runtime.num_threads
data.block_size
data.batch_size
model.n_layer
model.dropout
optimizer.beta1
optimizer.weight_decay
optimizer.learning_rate
training.max_steps
training.warmup_steps
training.lr_decay_steps
training.eval_interval
training.eval_batches
```

For each case:

1. save a v2 checkpoint;
2. clone a model parameter before load;
3. call `load_checkpoint(..., config=changed_config)`;
4. assert `IncompatibleResumeConfigError`;
5. assert the parameter and sample generator state remain unchanged;
6. assert the error mentions the exact dotted field.

- [ ] **Step 2: Add fingerprint mismatch tests**

Modify tokenizer/train/val content after saving, recompute current fingerprints, and assert errors for:

```text
dataset.tokenizer_sha256
dataset.train_sha256
dataset.val_sha256
```

Again prove no state mutation occurs.

- [ ] **Step 3: Add allowed-field tests**

Change only:

```text
data.directory while copying byte-identical files
training.output_dir
training.checkpoint_dir
training.tensorboard_dir
training.log_interval
training.checkpoint_interval
training.sample_interval
training.sample_tokens
training.sample_prompt
```

Assert resume succeeds and returns the expected `next_step`.

- [ ] **Step 4: Run tests and confirm red state**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_checkpoint.py -q
```

Expected: current loader accepts incompatible configs or has no config argument.

- [ ] **Step 5: Implement explicit compatibility projections**

Do not compare serialized YAML strings. Build ordered dotted-field/value pairs for immutable fields:

```python
def _immutable_config_values(config: ExperimentConfig) -> tuple[tuple[str, object], ...]:
    return (
        ("runtime.seed", config.runtime.seed),
        ("runtime.num_threads", config.runtime.num_threads),
        # every immutable field from the design
    )
```

Convert values to stable reader-facing strings only when constructing error records. Compare
fingerprints separately. Run `_validate_resume_compatibility()` immediately after payload parsing and
before `load_state_dict`, `random.setstate`, `torch.set_rng_state`, or batcher/generator restoration.

- [ ] **Step 6: Run focused quality gates**

Run:

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/checkpoint.py src/minigpt/trainer.py tests/test_checkpoint.py
.\.venv\Scripts\ruff.exe check src/minigpt/checkpoint.py src/minigpt/trainer.py tests/test_checkpoint.py
.\.venv\Scripts\basedpyright.exe src/minigpt/checkpoint.py src/minigpt/trainer.py tests/test_checkpoint.py
.\.venv\Scripts\pytest.exe tests/test_checkpoint.py tests/test_trainer.py -q
```

Expected: all mismatch/allowlist tests pass and static checks report zero errors.

- [ ] **Step 7: Commit**

```powershell
git add -- src/minigpt/checkpoint.py src/minigpt/trainer.py tests/test_checkpoint.py
git diff --staged --check
git commit -m "校验断点恢复实验与数据身份"
```

---

### Task 6: Separate Process Boundaries and Prove Exact Resume

**Files:**

- Modify: `tests/test_trainer.py`
- Modify: `tests/test_training_components.py`
- Modify: `src/minigpt/optimization.py`
- Modify: `src/minigpt/training_runtime.py`
- Modify: `src/minigpt/trainer.py`
- Modify: `train.py`

**Interfaces:**

- Changes:

```python
run_training(
    config: ExperimentConfig,
    *,
    resume_path: Path | None = None,
    run_until_step: int | None = None,
) -> TrainingResult
```

- Produces: `InvalidRunBoundaryError`.
- CLI removes `--max-steps` and adds `--run-until-step`.

- [ ] **Step 1: Add failing CLI and run-boundary tests**

Update the train CLI test to invoke:

```text
--run-until-step 1
```

Add assertions:

- `train.py --help` contains `--run-until-step`;
- it does not contain `--max-steps`;
- boundary `0` and boundary greater than `max_steps` fail clearly;
- resume boundary less than or equal to checkpoint `next_step` raises `InvalidRunBoundaryError`.

- [ ] **Step 2: Add event-semantics tests**

Create a config where:

```text
max_steps = 6
run_until_step = 3
eval_interval = 2
sample_interval = 2
checkpoint_interval = 2
```

Assert:

- metrics contain steps `[0, 1, 2]`;
- validation/sample occur only at completed-count 2, not at process exit 3;
- exit still writes a checkpoint with `next_step == 3`.

Create a second case with `run_until_step=2` and assert the scheduled checkpoint is written only
once by wrapping the public save boundary or checking one atomic replacement result and one event
record, without accessing private functions.

- [ ] **Step 3: Add failing sample-independence test**

Use two fresh runs with:

```text
dropout = 0.2
same seed and complete experiment
first sample_interval = 1
second sample_interval > max_steps
```

Load both final v2 checkpoints and recursively assert model and optimizer state equality. This test
must fail before the trainer passes its independent sample generator because frequent
`torch.multinomial` calls change the global dropout sequence.

- [ ] **Step 4: Add the exact-resume test**

Implement typed helpers in `tests/test_trainer.py`:

```python
def assert_state_values_equal(expected: object, actual: object) -> None:
    """Recursively compare checkpoint state primitives, containers, and tensors."""


def metric_without_system_fields(record: dict[str, MetricValue]) -> dict[str, MetricValue]:
    """Return step, losses, and learning rate only."""
```

Use one complete config with nonzero dropout, `max_steps=N`, `lr_decay_steps=N`, and `K` that is not a
validation/sample boundary:

```text
A: run_training(config)
B1: run_training(config, run_until_step=K)
B2: run_training(config, resume_path=checkpoint)
```

After A, copy its final checkpoint and metrics to reference paths, remove only the test's temporary
output/checkpoint/TensorBoard directories, then run B using the same config object.

Assert:

- every model tensor is equal;
- recursive optimizer state is equal;
- restored next train batch is equal;
- restored next validation batch is equal;
- subsequent `model.generate(..., generator=restored_sample_generator)` output is equal;
- non-system metric fields are equal for every step;
- learning-rate sequences are equal;
- B metrics steps equal `list(range(N))`.

- [ ] **Step 5: Run new tests and confirm red state**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_trainer.py tests/test_training_components.py -q
```

Expected: unknown run boundary, old final-event behavior, global sample RNG interference, or schedule
horizon mismatch causes failures.

- [ ] **Step 6: Implement process boundary and event predicates**

Resolve the boundary once:

```python
boundary = config.training.max_steps if run_until_step is None else run_until_step
```

Validate it before creating output writers. After resume, reject `boundary <= start_step`.

Replace max-step-aware event logic with:

```python
def _scheduled_event_due(completed_step: int, interval: int) -> bool:
    return (completed_step + 1) % interval == 0
```

Remove `step == max_steps - 1` from validation, sample, log, and periodic checkpoint predicates.
Loop with:

```python
for step in range(start_step, boundary):
```

Track whether a checkpoint was saved for the final completed step. On normal process exit, save one
checkpoint if the scheduled event did not already do so. Always pass
`components.sample_generator` to generation.

- [ ] **Step 7: Update CLI**

In `train.py`:

```python
parser.add_argument(
    "--run-until-step",
    type=int,
    help="exclusive absolute step boundary for this process",
)
```

Delete the `dataclasses.replace` max-step override and call:

```python
run_training(
    config,
    resume_path=arguments.resume,
    run_until_step=arguments.run_until_step,
)
```

- [ ] **Step 8: Run focused quality gates**

Run:

```powershell
.\.venv\Scripts\ruff.exe format src/minigpt/optimization.py src/minigpt/training_runtime.py src/minigpt/trainer.py train.py tests/test_trainer.py tests/test_training_components.py
.\.venv\Scripts\ruff.exe check src/minigpt/optimization.py src/minigpt/training_runtime.py src/minigpt/trainer.py train.py tests/test_trainer.py tests/test_training_components.py
.\.venv\Scripts\basedpyright.exe src/minigpt/optimization.py src/minigpt/training_runtime.py src/minigpt/trainer.py tests/test_trainer.py tests/test_training_components.py
.\.venv\Scripts\pytest.exe tests/test_trainer.py tests/test_checkpoint.py tests/test_training_components.py -q
```

Expected: all exact-resume invariants pass and static checks report zero errors.

- [ ] **Step 9: Commit**

```powershell
git add -- train.py src/minigpt/optimization.py src/minigpt/training_runtime.py src/minigpt/trainer.py tests/test_trainer.py tests/test_training_components.py
git diff --staged --check
git commit -m "分离训练计划与单次运行边界"
```

---

### Task 7: Add Cross-Platform CI and Align Documentation

**Files:**

- Create: `.github/workflows/quality.yml`
- Modify: `README.md`
- Modify: `tests/test_readme.py`
- Modify: `configs/char_gpt.yaml`
- Modify: `configs/char_gpt_smoke.yaml`

**Interfaces:**

- CI covers `windows-latest`/Python 3.14 and `ubuntu-latest`/Python 3.11.
- README documents only behavior proven by Tasks 1–6.

- [ ] **Step 1: Add failing README contract assertions**

Require README to contain:

```text
Python 3.11
Python 3.14
lr_decay_steps
--run-until-step
checkpoint format v2
v1
LegacyCheckpointResumeError
Windows
Linux
```

Assert the old resume example containing `--max-steps` is absent.

- [ ] **Step 2: Run README tests and confirm red state**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_readme.py -q
```

Expected: README lacks the new compatibility and resume contracts.

- [ ] **Step 3: Add the CI workflow**

Create `.github/workflows/quality.yml`:

```yaml
name: quality

on:
  push:
  pull_request:

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: windows-latest
            python-version: "3.14"
          - os: ubuntu-latest
            python-version: "3.11"
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install
        run: python -m pip install -e ".[dev]"
      - name: Check dependencies
        run: python -m pip check
      - name: Check formatting
        run: ruff format --check src tests
      - name: Lint
        run: ruff check src tests
      - name: Type check
        run: basedpyright
      - name: Test
        run: pytest
```

No CI step may call `prepare_data.py` without a local source fixture.

- [ ] **Step 4: Rewrite README resume and environment sections**

Document:

- Python 3.11–3.14 and CPU PyTorch;
- Windows-first with Linux CI portability coverage;
- `optimizer.type: adamw` and `training.lr_decay_steps`;
- `max_steps` as complete plan and `--run-until-step` as exclusive process boundary;
- scheduled validation/sample versus exit-only checkpoint;
- v2 exact resume contents and SHA-256 fingerprints;
- v1 config/model/generation support and `LegacyCheckpointResumeError` for training;
- sample-configuration changes preserve parameter trajectory but change sample stream;
- exact resume is backed by model, optimizer, batcher, sample RNG, LR, loss, and step tests.

Do not claim that wall-clock metrics are identical.

- [ ] **Step 5: Run documentation and full automated gates**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_readme.py -q
.\.venv\Scripts\ruff.exe format src tests
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\basedpyright.exe
.\.venv\Scripts\pytest.exe
```

Expected: README contracts pass, Ruff/basedpyright report zero errors, and the full pytest count is
reported.

- [ ] **Step 6: Verify CLI help**

Run:

```powershell
.\.venv\Scripts\python.exe prepare_data.py --help
.\.venv\Scripts\python.exe train.py --help
.\.venv\Scripts\python.exe generate.py --help
.\.venv\Scripts\python.exe benchmark.py --help
.\.venv\Scripts\python.exe profile_model.py --help
```

Expected: all exit 0; train help shows `--run-until-step` and not `--max-steps`.

- [ ] **Step 7: Inspect repository state**

Run:

```powershell
git diff --check
git status --short
git diff --stat
git diff
```

Expected: only Stage 6 source, tests, configs, workflow, plan, and README changes; no data,
checkpoint, report, TensorBoard, or trace artifacts.

- [ ] **Step 8: Commit**

```powershell
git add -- .github/workflows/quality.yml README.md configs tests/test_readme.py
git diff --staged --check
git commit -m "增加跨平台质量门禁并更新恢复文档"
```

---

## Final Verification

- [ ] Run the required gates in order:

```powershell
.\.venv\Scripts\ruff.exe format src tests
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\basedpyright.exe
.\.venv\Scripts\pytest.exe
```

- [ ] Confirm no direct forward calls:

```powershell
rg "\.forward\(" src tests
```

Expected: no matches.

- [ ] Confirm dependency health and CLI boundaries:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe train.py --help
```

- [ ] Inspect commits and cleanliness:

```powershell
git log --oneline --decorate -12
git diff --check
git status --short --branch
```

Expected: local Stage 6 commits are present, no uncommitted tracked changes remain, and the branch is
ahead of `origin/main` without any push.
