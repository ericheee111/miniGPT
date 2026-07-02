# Stage 5 Bilingual README and Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a reproducible bilingual project README, a direct data-preparation CLI, and a repository that passes Ruff, basedpyright, and pytest without weakening the configured rules.

**Architecture:** Keep data preparation in `minigpt.data` and expose it through a thin root CLI. Repair type information from the lowest-level tensor modules upward so `Any` does not cascade into models, checkpoints, trainers, or tests. Treat README commands and local links as stable user-facing contracts, then validate every documented workflow through its real CLI.

**Tech Stack:** Python 3.14, PyTorch 2.12 CPU, NumPy, PyYAML, pytest, Ruff, basedpyright, PowerShell, Markdown, Mermaid.

---

## File map

**Create**

- `prepare_data.py`: command-line adapter for `prepare_tiny_shakespeare`.
- `tests/test_prepare_data_cli.py`: end-to-end data CLI coverage with a local `file://` corpus.
- `tests/test_readme.py`: stable README section, command, and local-link contracts.
- `docs/superpowers/plans/2026-07-02-stage5-readme-quality.md`: this execution plan.

**Modify**

- `README.md`: bilingual project documentation and measured results.
- `src/minigpt/__init__.py`: package metadata typing and package documentation.
- `src/minigpt/batching.py`: typed mutable sampler state and NumPy/Torch boundary.
- `src/minigpt/layers.py`: explicit tensor module contracts and registered-buffer typing.
- `src/minigpt/model.py`: typed module calls, errors, generation, and exports.
- `src/minigpt/settings.py`: documented immutable settings and typed errors.
- `src/minigpt/config.py`: typed YAML parsing boundary.
- `src/minigpt/data.py`: typed JSON parsing, URL handling, NumPy arrays, and file-write results.
- `src/minigpt/optimization.py`: deterministic RNG and optimizer typing.
- `src/minigpt/metrics.py`: typed JSON serialization and write results.
- `src/minigpt/checkpoint.py`: parsed checkpoint payload and random-state types.
- `src/minigpt/trainer.py`: smaller training orchestration units and typed runtime state.
- `tests/test_data.py`, `tests/test_model.py`, `tests/test_checkpoint.py`,
  `tests/test_trainer.py`, `tests/test_training_components.py`, `tests/test_package.py`:
  remove `Any` propagation while preserving behavior.

**Do not modify**

- Checkpoint format version or serialized field names.
- Training numerical semantics, CLI flags, model dimensions, or benchmark methodology.
- Ruff `select = ["ALL"]` or basedpyright `typeCheckingMode = "all"`.

### Task 1: Add a direct data-preparation CLI

**Files:**

- Create: `prepare_data.py`
- Create: `tests/test_prepare_data_cli.py`
- Reuse: `src/minigpt/data.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).parents[1]


def test_prepare_data_cli_writes_all_artifacts(tmp_path: Path) -> None:
    # Given: a local UTF-8 corpus and an empty output directory.
    source = tmp_path / "source.txt"
    _ = source.write_text("abcdeabcde", encoding="utf-8")
    data_dir = tmp_path / "data"

    # When: data preparation runs through the public CLI.
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "prepare_data.py",
            "--data-dir",
            str(data_dir),
            "--source-url",
            source.as_uri(),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: every durable artifact exists and is reported.
    assert completed.returncode == 0, completed.stderr
    assert "train.npy" in completed.stdout
    assert "val.npy" in completed.stdout
    assert (data_dir / "raw" / "input.txt").is_file()
    assert (data_dir / "processed" / "train.npy").is_file()
    assert (data_dir / "processed" / "val.npy").is_file()
    assert (data_dir / "processed" / "tokenizer.json").is_file()
    assert (data_dir / "processed" / "metadata.json").is_file()


def test_prepare_data_cli_help_lists_boundary_options() -> None:
    # Given: the public data preparation entrypoint.
    # When: its help page is requested.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "prepare_data.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: both supported boundary options are documented.
    assert completed.returncode == 0, completed.stderr
    assert "--data-dir" in completed.stdout
    assert "--source-url" in completed.stdout
```

- [ ] **Step 2: Run the tests and verify the red state**

Run:

```powershell
pytest tests/test_prepare_data_cli.py -q
```

Expected: both tests fail because `prepare_data.py` does not exist.

- [ ] **Step 3: Implement the thin CLI**

```python
"""Prepare Tiny Shakespeare artifacts for MiniTrainGPT."""

import argparse
import sys
from pathlib import Path

from minigpt.data import TINY_SHAKESPEARE_URL, prepare_tiny_shakespeare


class PrepareArguments(argparse.Namespace):
    """Store parsed data-preparation options with concrete types."""

    data_dir: Path
    source_url: str


def build_parser() -> argparse.ArgumentParser:
    """Create the data-preparation command-line parser."""
    parser = argparse.ArgumentParser(description="Prepare Tiny Shakespeare training data.")
    _ = parser.add_argument("--data-dir", type=Path, default=Path("data"))
    _ = parser.add_argument("--source-url", default=TINY_SHAKESPEARE_URL)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare data and print the durable artifact paths."""
    arguments = build_parser().parse_args(argv, namespace=PrepareArguments())
    prepared = prepare_tiny_shakespeare(arguments.data_dir, arguments.source_url)
    lines = (
        f"raw={prepared.raw_path}",
        f"train={prepared.train_path}",
        f"validation={prepared.val_path}",
        f"tokenizer={prepared.tokenizer_path}",
        f"metadata={prepared.metadata_path}",
        "",
    )
    _ = sys.stdout.write("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify the green state**

Run:

```powershell
ruff format prepare_data.py tests/test_prepare_data_cli.py
ruff check prepare_data.py tests/test_prepare_data_cli.py
basedpyright tests/test_prepare_data_cli.py
pytest tests/test_prepare_data_cli.py -q
```

Expected: 2 tests pass and both static checks report zero errors.

### Task 2: Repair tensor-layer and model type foundations

**Files:**

- Modify: `src/minigpt/batching.py`
- Modify: `src/minigpt/layers.py`
- Modify: `src/minigpt/model.py`
- Modify: `src/minigpt/settings.py`
- Modify: `src/minigpt/optimization.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Capture the current static red state**

Run:

```powershell
ruff check src/minigpt/batching.py src/minigpt/layers.py src/minigpt/model.py src/minigpt/settings.py src/minigpt/optimization.py tests/test_model.py
basedpyright src/minigpt/batching.py src/minigpt/layers.py src/minigpt/model.py src/minigpt/settings.py src/minigpt/optimization.py tests/test_model.py
```

Expected: Ruff reports the existing module/docstring/import/exception issues and basedpyright reports
unknown tensor/module attributes. Save the exact counts in the implementation evidence.

- [ ] **Step 2: Make module state explicit**

Apply these concrete patterns:

```python
from typing import Final, final, override


@final
class LayerNorm(nn.Module):
    """Normalize each token across its embedding dimension."""

    weight: nn.Parameter
    bias: nn.Parameter | None

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize the final tensor dimension using population variance."""
        centered = hidden_states - hidden_states.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.epsilon)
        shifted = normalized * self.weight
        return shifted if self.bias is None else shifted + self.bias
```

Declare `token_embedding`, `position_embedding`, `blocks`, `final_norm`, `lm_head`, attention
projections, dropout modules, and `causal_mask: Tensor` on their owning classes. Mark concrete
modules `@final`, annotate every `forward`, and call typed `.forward` methods internally when
PyTorch's generic `nn.Module.__call__` would otherwise return `Any`.

For `TokenBatcher`, annotate `_tokens`, `_batch_size`, `_block_size`, `_rng`, `_device`, and
`__slots__`. Convert NumPy scalar starts with `int(start)` only after narrowing the array dtype.

- [ ] **Step 3: Centralize typed validation errors**

Use dataclass exceptions with `@override` on `__str__` and module-level `Final` reason constants:

```python
INVALID_TEMPERATURE_REASON: Final = "temperature must be positive"


@dataclass(frozen=True, slots=True)
class InvalidGenerationConfigError(ValueError):
    """Report invalid sampling controls."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the invalid generation setting."""
        return f"invalid generation configuration: {self.reason}"
```

Do not change error text asserted by existing tests. Add module and public API docstrings required by
Ruff. Import `GPTConfig`, `OptimizerSettings`, and other settings from `minigpt.settings`, not through
private re-exports in `minigpt.config`.

- [ ] **Step 4: Preserve deterministic optimization behavior**

Keep Python, NumPy, and torch global seeding because checkpoint resume serializes those exact global
states. Add narrow local Ruff annotations for non-cryptographic RNG and the boolean deterministic
flag instead of replacing the RNG model. Name the weight-decay dimensionality threshold:

```python
MATRIX_PARAMETER_DIMENSIONS: Final = 2
destination = (
    decayed_parameters
    if parameter.ndim >= MATRIX_PARAMETER_DIMENSIONS
    else non_decayed_parameters
)
```

- [ ] **Step 5: Remove test-side `Any` propagation**

Replace module alias patterns that Ruff rejects with direct static imports, use `_ = torch.manual_seed`
for intentional return values, and call typed `model.forward` in assertions where needed.
Replace `pytest.approx` calls that basedpyright sees as partially unknown with `math.isclose` and an
explicit tolerance.

- [ ] **Step 6: Verify this dependency layer**

Run:

```powershell
ruff check src/minigpt/batching.py src/minigpt/layers.py src/minigpt/model.py src/minigpt/settings.py src/minigpt/optimization.py tests/test_model.py
basedpyright src/minigpt/batching.py src/minigpt/layers.py src/minigpt/model.py src/minigpt/settings.py src/minigpt/optimization.py tests/test_model.py
pytest tests/test_model.py tests/test_data.py::test_token_batcher_returns_shifted_cpu_batches tests/test_data.py::test_token_batcher_is_reproducible_for_fixed_seed -q
```

Expected: zero static errors and all selected tests pass.

### Task 3: Parse YAML, JSON, and metric boundaries into typed values

**Files:**

- Modify: `src/minigpt/config.py`
- Modify: `src/minigpt/data.py`
- Modify: `src/minigpt/metrics.py`
- Modify: `src/minigpt/__init__.py`
- Test: `tests/test_data.py`
- Test: `tests/test_training_components.py`
- Test: `tests/test_package.py`

- [ ] **Step 1: Record boundary failures**

Run:

```powershell
ruff check src/minigpt/config.py src/minigpt/data.py src/minigpt/metrics.py src/minigpt/__init__.py tests/test_data.py tests/test_training_components.py tests/test_package.py
basedpyright src/minigpt/config.py src/minigpt/data.py src/minigpt/metrics.py src/minigpt/__init__.py tests/test_data.py tests/test_training_components.py tests/test_package.py
```

Expected: current JSON/YAML `Any`, unused call results, type-only import, and documentation failures.

- [ ] **Step 2: Add recursive boundary parsers**

Retain the existing recursive `ConfigValue` union. Assign untyped library results to that union only
through functions that inspect every supported variant:

```python
type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def parse_json_value(value: JsonValue) -> JsonValue:
    """Return a recursively typed JSON value."""
    match value:
        case str() | int() | float() | bool() | None:
            return value
        case list() as items:
            return [parse_json_value(item) for item in items]
        case dict() as mapping:
            return {str(key): parse_json_value(item) for key, item in mapping.items()}
```

Use a project-local typed wrapper around `yaml.safe_load` and `json.loads`. Do not expose raw
`dict`, `Any`, or `object` in function signatures. Keep validation at file boundaries and pass only
`ConfigMapping`, tokenizer documents, and metadata records inward.

- [ ] **Step 3: Make I/O effects explicit**

Assign intentionally ignored returns:

```python
_ = path.write_text(payload, encoding="utf-8")
_ = destination.write_bytes(content)
```

Use `collections.abc.Mapping` for runtime mapping behavior and `typing` only for static-only names.
Add `@override` to exception string methods and concise public module/API documentation.

- [ ] **Step 4: Type tests at the same boundary**

In tests, parse JSON into a `TypedDict` helper after asserting each field type. Keep the existing
`file://` download pattern and Given/When/Then comments. Replace dynamic module imports with static
imports and assign write-call results to `_`.

- [ ] **Step 5: Verify data and configuration**

Run:

```powershell
ruff check src/minigpt/config.py src/minigpt/data.py src/minigpt/metrics.py src/minigpt/__init__.py tests/test_data.py tests/test_training_components.py tests/test_package.py
basedpyright src/minigpt/config.py src/minigpt/data.py src/minigpt/metrics.py src/minigpt/__init__.py tests/test_data.py tests/test_training_components.py tests/test_package.py
pytest tests/test_data.py tests/test_training_components.py tests/test_package.py -q
```

Expected: zero static errors and all selected tests pass.

### Task 4: Parse checkpoint payloads without changing the format

**Files:**

- Modify: `src/minigpt/checkpoint.py`
- Modify: `tests/test_checkpoint.py`

- [ ] **Step 1: Run checkpoint tests and static checks before refactoring**

Run:

```powershell
pytest tests/test_checkpoint.py -q
ruff check src/minigpt/checkpoint.py tests/test_checkpoint.py
basedpyright src/minigpt/checkpoint.py tests/test_checkpoint.py
```

Expected: checkpoint runtime tests pass while static checks fail on raw torch payloads and random
state types.

- [ ] **Step 2: Define explicit serialized state types**

Keep all existing checkpoint keys. Define `TypedDict` records for NumPy state JSON and checkpoint
metadata, and Protocols for model/optimizer state loading:

```python
class NumpyRandomStateDocument(TypedDict):
    algorithm: str
    keys: list[int]
    position: int
    has_gauss: int
    cached_gaussian: float


class StateLoadResult(Protocol):
    missing_keys: list[str]
    unexpected_keys: list[str]


class StateDictLoader(Protocol):
    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        *,
        strict: bool = True,
    ) -> StateLoadResult:
        raise NotImplementedError
```

At `_load_payload`, validate the `torch.load` result key by key before constructing
`CheckpointPayload`. Use explicit unions for primitive, tensor, mapping, and tuple values; do not
return an unchecked generic dictionary.

- [ ] **Step 3: Preserve global RNG restoration**

Keep `np.random.get_state()` and `np.random.set_state()` because they are part of the versioned
resume contract. Add the narrowest Ruff annotations on those exact calls. Convert NumPy keys to and
from `list[int]` through a typed `np.ndarray` and validate algorithm/position/gaussian fields.

- [ ] **Step 4: Keep save/load signatures reviewable**

If `save_checkpoint` still violates `PLR0913`, introduce one frozen `CheckpointState` dataclass that
groups model, optimizer, step, config, and batcher state. Update only its two production callers and
tests. Do not change on-disk keys or `FORMAT_VERSION`.

- [ ] **Step 5: Verify checkpoint compatibility**

Run:

```powershell
ruff check src/minigpt/checkpoint.py tests/test_checkpoint.py
basedpyright src/minigpt/checkpoint.py tests/test_checkpoint.py
pytest tests/test_checkpoint.py tests/test_trainer.py::test_run_training_resume_continues_without_duplicate_steps -q
```

Expected: zero static errors; model, optimizer, Python RNG, NumPy RNG, torch RNG, and batcher RNG
restore tests pass.

### Task 5: Decompose and type the trainer orchestration

**Files:**

- Modify: `src/minigpt/trainer.py`
- Modify: `tests/test_trainer.py`

- [ ] **Step 1: Capture the trainer red state**

Run:

```powershell
pytest tests/test_trainer.py -q
ruff check src/minigpt/trainer.py tests/test_trainer.py
basedpyright src/minigpt/trainer.py tests/test_trainer.py
```

Expected: runtime tests pass; static checks report the current oversized argument list, 63-statement
training function, typed-loss, JSON record, and CLI subprocess issues.

- [ ] **Step 2: Introduce focused mutable runtime state**

Use frozen value objects for paths/results and one documented mutable state object for the evolving
step:

```python
@dataclass(frozen=True, slots=True)
class TrainingPaths:
    metrics: Path
    samples: Path
    latest_checkpoint: Path


@dataclass(slots=True)  # noqa: MUTABLE_OK
class TrainingLoopState:
    """Track values that intentionally evolve across training steps."""

    next_step: int
    last_validation_loss: float | None
```

Extract `_load_training_inputs`, `_build_training_components`, `_run_training_step`,
`_record_due_events`, and `_save_due_checkpoint`. Each helper receives at most five arguments or a
single typed context dataclass. Keep `run_training` as the orchestration boundary.

- [ ] **Step 3: Preserve timing semantics**

`_run_training_step` must continue measuring:

```text
data_time_ms
forward_backward_time_ms
optimizer_time_ms
step_time_ms
tokens_per_sec = batch_size * block_size / step_seconds
```

Do not move logging, evaluation, sampling, checkpointing, or file I/O into the timed region. Call
`model.forward` directly where the generic PyTorch call loses type information. Use a typed callable
Protocol for optimizer `step` rather than a broad cast or ignore.

- [ ] **Step 4: Parse metrics JSON in tests**

Define a test-local `TrainingMetricRecord` `TypedDict`, validate JSON field types once, and assert
steps and required keys from typed records. Mark fixed `sys.executable` subprocess calls with the
existing narrow `# noqa: S603`.

- [ ] **Step 5: Verify trainer behavior**

Run:

```powershell
ruff check src/minigpt/trainer.py tests/test_trainer.py
basedpyright src/minigpt/trainer.py tests/test_trainer.py
pytest tests/test_trainer.py -q
```

Expected: zero static errors and all artifact, resume, train CLI, and generate CLI tests pass.

### Task 6: Close all repository-wide static gaps

**Files:**

- Modify only files still reported by Ruff or basedpyright.
- Test: all files under `tests/`.

- [ ] **Step 1: Run repository-wide checks**

```powershell
ruff format src tests
ruff check src tests
basedpyright
```

Expected before cleanup: a smaller residual list. Record every remaining file and rule.

- [ ] **Step 2: Resolve residuals without weakening configuration**

Allowed resolution patterns:

- Add a missing annotation/docstring or correct an import.
- Replace an unchecked library boundary with an explicit parser or Protocol.
- Assign an intentional return value to `_`.
- Refactor an oversized function or argument list.
- Add a narrow line-level Ruff annotation only where the behavior is semantically required
  (`S310` for validated `file://`/HTTPS URL opening, `S603` for fixed `sys.executable`,
  `NPY002` for versioned global RNG state).

Forbidden:

- New global Ruff ignores.
- Changing `typeCheckingMode`.
- `type: ignore`, `pyright: ignore`, `Any`, or raw `object` annotations.
- Deleting or weakening tests.

- [ ] **Step 3: Run the no-excuse audit**

```powershell
python C:\Users\Administrator\.codex\plugins\cache\sisyphuslabs\omo\4.13.0\skills\programming\scripts\python\check-no-excuse-rules.py src tests prepare_data.py
```

Expected: `no violations`.

- [ ] **Step 4: Confirm all automated gates**

```powershell
ruff format src tests
ruff check src tests
basedpyright
pytest
```

Expected: Ruff reports no changes/errors, basedpyright reports `0 errors`, and all tests pass.

### Task 7: Lock stable README contracts

**Files:**

- Create: `tests/test_readme.py`
- Modify later: `README.md`

- [ ] **Step 1: Write contract tests before replacing README**

```python
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
README_PATH = PROJECT_ROOT / "README.md"
REQUIRED_COMMANDS = (
    "python prepare_data.py",
    "python train.py",
    "python generate.py",
    "python benchmark.py",
    "python profile_model.py",
)
REQUIRED_HEADINGS = (
    "## 项目概览",
    "## Quick Start",
    "## 核心架构",
    "## 性能分析",
    "## Roadmap",
    "## Resume",
)


def test_readme_documents_stable_user_contracts() -> None:
    # Given: the repository README.
    readme = README_PATH.read_text(encoding="utf-8")

    # When: required sections and commands are inspected.
    # Then: every stable public workflow is documented.
    assert all(heading in readme for heading in REQUIRED_HEADINGS)
    assert all(command in readme for command in REQUIRED_COMMANDS)
    assert "ruff check src tests" in readme
    assert "basedpyright" in readme
    assert "pytest" in readme


def test_readme_local_markdown_links_exist() -> None:
    # Given: local Markdown links in the README.
    readme = README_PATH.read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)", readme)

    # When: every local target is resolved from the project root.
    missing = [target for target in targets if not (PROJECT_ROOT / target).exists()]

    # Then: the README contains no broken local links.
    assert missing == []
```

- [ ] **Step 2: Verify the contract tests fail for the two-line README**

Run:

```powershell
pytest tests/test_readme.py -q
```

Expected: section/command test fails because the README is incomplete.

### Task 8: Write the bilingual README from verified evidence

**Files:**

- Modify: `README.md`
- Test: `tests/test_readme.py`
- Source evidence: `configs/*.yaml`, `outputs/smoke/metrics.jsonl`,
  `reports/benchmark_smoke/benchmark_report.md`

- [ ] **Step 1: Replace the corrupted README**

Write the sections in the exact order approved by the design:

1. Title, English subtitle, CPU-first badges/text.
2. Chinese project overview.
3. Concise English summary.
4. Features and explicit non-goals.
5. Python 3.14/Windows CPU requirements and editable install.
6. Five-minute quick start.
7. Data, training, resume, generation, TensorBoard commands.
8. Configuration explanation.
9. Mermaid architecture flowchart.
10. Benchmark methodology and commands.
11. Profiler scopes and Chrome trace instructions.
12. Smoke training and benchmark tables with provenance.
13. Interpretation and limitations.
14. Project tree.
15. Four quality commands.
16. Roadmap.
17. Chinese and English resume bullets.

Use these measured benchmark rows, labeled as a smoke run:

```markdown
| Case | Params | Median ms | Tokens/s | CV % | RSS MiB |
|---|---:|---:|---:|---:|---:|
| small-t1-b2-s32 | 108,992 | 4.577 | 14,010.0 | 4.28 | 290.9 |
| small-t1-b2-s64 | 111,040 | 5.596 | 22,880.4 | 1.49 | 292.8 |
| small-t4-b2-s32 | 108,992 | 4.311 | 14,937.0 | 7.86 | 292.9 |
| small-t4-b2-s64 | 111,040 | 4.932 | 25,956.4 | 0.70 | 293.7 |
```

State that results came from Python 3.14.5, PyTorch 2.12.1+cpu, Windows 11, and
Intel i7-14700, and that temperature, power policy, and background load affect results.

- [ ] **Step 2: Verify README contracts**

```powershell
pytest tests/test_readme.py -q
```

Expected: both README tests pass.

- [ ] **Step 3: Manually inspect rendering-sensitive content**

Confirm:

- UTF-8 Chinese text displays correctly.
- Mermaid syntax has one flowchart and no unsupported directives.
- Every code block has a language.
- Tables have aligned header/column counts.
- Local links point only to tracked files, not ignored runtime artifacts.

### Task 9: Drive every documented workflow and create the final implementation commit

**Files:**

- Verify all changed source, tests, and README files.
- Commit all stage 5 implementation changes in one atomic commit, as required by the approved design.

- [ ] **Step 1: Reinstall declared development dependencies**

```powershell
python -m pip install -e ".[dev]"
python -m pip check
```

Expected: editable installation succeeds and pip reports no broken requirements.

- [ ] **Step 2: Run the four required gates in order**

```powershell
ruff format src tests
ruff check src tests
basedpyright
pytest
```

Expected: every command exits 0; report the final pytest count.

- [ ] **Step 3: Run real data and training smoke flows**

Use temporary output/config paths where needed so existing ignored artifacts do not hide failures:

```powershell
python prepare_data.py --data-dir data
python train.py --config configs/char_gpt_smoke.yaml
python train.py --config configs/char_gpt_smoke.yaml --resume checkpoints/smoke/latest.pt --max-steps 3
python generate.py --checkpoint checkpoints/smoke/latest.pt --prompt "ROMEO:" --max-new-tokens 16 --seed 1337
```

Expected:

- Data paths are printed and all five artifacts exist.
- Training prints step metrics and writes a checkpoint.
- Resume continues after the saved step without duplicate metrics.
- Generation prints the prompt plus 16 new characters.

- [ ] **Step 4: Run performance smoke flows**

```powershell
python benchmark.py --config configs/benchmark_smoke.yaml
python profile_model.py --config configs/benchmark_smoke.yaml
```

Expected:

- Benchmark emits 8 raw rows and 4 summary rows.
- Profile trace parses as JSON and contains `traceEvents`.
- `data_preparation`, `forward_backward`, and `optimizer_step` each have the configured count.

- [ ] **Step 5: Inspect the complete diff**

```powershell
git status --short
git diff --check
git diff --stat
git diff
```

Expected: only stage 5 implementation, tests, and README changes; no generated data, reports,
checkpoints, debug files, or unrelated edits.

- [ ] **Step 6: Commit the approved stage**

```powershell
git add -- README.md prepare_data.py src/minigpt tests
git diff --staged --check
git diff --staged --stat
git commit -m "完善双语文档与项目质量门禁"
git log -1 --oneline
git status --short --branch
```

Expected: one Chinese implementation commit and a clean worktree. Do not push unless explicitly
requested.
