import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from minigpt import trainer
from minigpt.checkpoint import IncompatibleResumeConfigError, StateValue, load_checkpoint
from minigpt.data import CharTokenizer
from minigpt.settings import (
    DataSettings,
    ExperimentConfig,
    ModelSettings,
    OptimizerSettings,
    RuntimeSettings,
    TrainingSettings,
)
from minigpt.training_runtime import TrainingComponents, build_training_components

if TYPE_CHECKING:
    from collections.abc import Callable

MetricValue: TypeAlias = int | float | None

PROJECT_ROOT = Path(__file__).parents[1]


def create_processed_data(directory: Path) -> None:
    text = ("abcde\n" * 20) + "ROMEO:"
    tokenizer = CharTokenizer.from_text(text)
    tokens = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    directory.mkdir(parents=True)
    save_tokens = cast(
        "Callable[[Path, npt.NDArray[np.uint16]], None]",
        np.save,
    )
    save_tokens(directory / "train.npy", tokens[:100])
    save_tokens(directory / "val.npy", tokens[100:])
    tokenizer.save(directory / "tokenizer.json")


def run_git(repository: Path, *arguments: str) -> None:
    _ = subprocess.run(  # noqa: S603
        ["git", "-C", str(repository), *arguments],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )


def tiny_experiment(tmp_path: Path, *, max_steps: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        runtime=RuntimeSettings(seed=17, num_threads=1, device="cpu"),
        data=DataSettings(
            directory=tmp_path / "processed",
            block_size=4,
            batch_size=2,
        ),
        model=ModelSettings(
            vocab_size=None,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        ),
        optimizer=OptimizerSettings(
            optimizer_type="adamw",
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            grad_clip=1.0,
        ),
        training=TrainingSettings(
            max_steps=max_steps,
            warmup_steps=0,
            lr_decay_steps=max_steps,
            eval_interval=1,
            eval_batches=1,
            log_interval=1,
            checkpoint_interval=1,
            sample_interval=1,
            sample_tokens=2,
            sample_prompt="ab",
            output_dir=tmp_path / "outputs",
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
        ),
    )


def assert_state_values_equal(expected: StateValue, actual: StateValue) -> None:
    """Recursively compare checkpoint state primitives, containers, and tensors."""
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        assert torch.equal(expected, actual)
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        assert expected.keys() == actual.keys()
        for key, expected_value in expected.items():
            assert_state_values_equal(expected_value, actual[key])
        return
    if isinstance(expected, list) and isinstance(actual, list):
        assert len(expected) == len(actual)
        for expected_value, actual_value in zip(expected, actual, strict=True):
            assert_state_values_equal(expected_value, actual_value)
        return
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        assert len(expected) == len(actual)
        for expected_value, actual_value in zip(expected, actual, strict=True):
            assert_state_values_equal(expected_value, actual_value)
        return
    assert expected == actual


def metric_without_system_fields(record: dict[str, MetricValue]) -> dict[str, MetricValue]:
    """Keep deterministic training metrics and omit wall-clock/system observations."""
    return {
        "step": record["step"],
        "train_loss": record["train_loss"],
        "val_loss": record["val_loss"],
        "learning_rate": record["learning_rate"],
    }


def read_metric_records(path: Path) -> list[dict[str, MetricValue]]:
    return [
        cast("dict[str, MetricValue]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def restore_components(config: ExperimentConfig, checkpoint_path: Path) -> TrainingComponents:
    components = build_training_components(config)
    _ = load_checkpoint(
        checkpoint_path,
        resources=components.checkpoint_resources,
        config=components.config,
    )
    return components


def test_run_training_writes_all_observable_artifacts(tmp_path: Path) -> None:
    # Given: a tiny processed corpus and two-step CPU experiment.
    create_processed_data(tmp_path / "processed")
    experiment = tiny_experiment(tmp_path)

    # When: the full training runner executes.
    result = trainer.run_training(experiment)

    # Then: metrics, samples, TensorBoard events, and latest checkpoint exist.
    metric_lines = result.metrics_path.read_text(encoding="utf-8").splitlines()
    records = [cast("dict[str, MetricValue]", json.loads(line)) for line in metric_lines]
    assert [record["step"] for record in records] == [0, 1]
    assert all(record["val_loss"] is not None for record in records)
    assert all(
        isinstance(record["tokens_per_sec"], int | float) and record["tokens_per_sec"] > 0
        for record in records
    )
    assert result.samples_path.read_text(encoding="utf-8").count("step=") == 2
    assert result.checkpoint_path.is_file()
    assert any(result.tensorboard_dir.glob("events.out.tfevents.*"))


def test_run_training_resume_rejects_changed_experiment_horizon(tmp_path: Path) -> None:
    # Given: a completed two-step experiment and a changed four-step definition.
    create_processed_data(tmp_path / "processed")
    initial = tiny_experiment(tmp_path)
    initial_result = trainer.run_training(initial)
    resumed = replace(
        initial,
        training=replace(initial.training, max_steps=4),
    )

    # When/Then: resume rejects redefining max_steps.
    with pytest.raises(IncompatibleResumeConfigError, match=r"training\.max_steps"):
        _ = trainer.run_training(resumed, resume_path=initial_result.checkpoint_path)


def test_train_cli_runs_one_step_from_yaml(tmp_path: Path) -> None:
    # Given: a runnable YAML experiment and processed corpus.
    create_processed_data(tmp_path / "processed")
    experiment = tiny_experiment(tmp_path, max_steps=2)
    config_path = tmp_path / "experiment.yaml"
    _ = config_path.write_text(experiment.to_yaml(), encoding="utf-8")

    # When: the documented train.py surface stops this process at step one.
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "train.py",
            "--config",
            str(config_path),
            "--run-until-step",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: the command succeeds and writes latest.pt.
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "checkpoints" / "latest.pt").is_file()


def test_train_cli_exposes_only_the_process_boundary() -> None:
    # Given/When: the training CLI help is requested.
    completed = subprocess.run(
        [sys.executable, "train.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: it exposes the process boundary and no experiment-horizon override.
    assert completed.returncode == 0, completed.stderr
    assert "--run-until-step" in completed.stdout
    assert "--max-steps" not in completed.stdout


def test_train_cli_records_completed_reference_provenance(tmp_path: Path) -> None:
    # Given: a clean temporary repository containing a tiny canonical experiment.
    create_processed_data(tmp_path / "processed")
    experiment = tiny_experiment(tmp_path)
    config_path = tmp_path / "reference.yaml"
    _ = config_path.write_text(experiment.to_yaml(), encoding="utf-8")
    _ = (tmp_path / ".gitignore").write_text(
        "processed/\noutputs/\ncheckpoints/\ntensorboard/\n",
        encoding="utf-8",
    )
    run_git(tmp_path, "init")
    run_git(tmp_path, "config", "user.email", "tests@example.com")
    run_git(tmp_path, "config", "user.name", "miniGPT tests")
    run_git(tmp_path, "add", ".gitignore", "reference.yaml")
    run_git(tmp_path, "commit", "-m", "reference fixture")
    provenance_path = tmp_path / "outputs" / "run_provenance.json"

    # When: the real training CLI runs one bounded segment with provenance enabled.
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--config",
            str(config_path),
            "--run-until-step",
            "1",
            "--provenance",
            str(provenance_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: the command succeeds and records one completed segment at step zero.
    assert completed.returncode == 0, completed.stderr
    document = cast("dict[str, object]", json.loads(provenance_path.read_text(encoding="utf-8")))
    segments = cast("list[dict[str, object]]", document["segments"])
    assert len(segments) == 1
    assert segments[0]["status"] == "completed"
    assert segments[0]["final_completed_step"] == 0


@pytest.mark.parametrize("boundary", [0, 3])
def test_run_training_rejects_boundary_outside_experiment(
    tmp_path: Path,
    boundary: int,
) -> None:
    # Given: a valid two-step experiment and an invalid process boundary.
    create_processed_data(tmp_path / "processed")
    experiment = tiny_experiment(tmp_path)

    # When/Then: the boundary is rejected before training starts.
    with pytest.raises(trainer.InvalidRunBoundaryError, match="expected"):
        _ = trainer.run_training(experiment, run_until_step=boundary)


def test_process_exit_forces_only_checkpoint_not_validation_or_sample(tmp_path: Path) -> None:
    # Given: scheduled events every two completions and an off-schedule exit at three.
    create_processed_data(tmp_path / "processed")
    base = tiny_experiment(tmp_path, max_steps=6)
    experiment = replace(
        base,
        training=replace(
            base.training,
            eval_interval=2,
            checkpoint_interval=2,
            sample_interval=2,
        ),
    )

    # When: this process stops after completing absolute steps zero through two.
    result = trainer.run_training(experiment, run_until_step=3)

    # Then: metrics are complete, but validation and sampling occur only at completion two.
    records = [
        cast("dict[str, MetricValue]", json.loads(line))
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in records] == [0, 1, 2]
    assert [record["step"] for record in records if record["val_loss"] is not None] == [1]
    assert result.samples_path.read_text(encoding="utf-8").count("step=") == 1
    components = build_training_components(experiment)
    resume_state = load_checkpoint(
        result.checkpoint_path,
        resources=components.checkpoint_resources,
        config=components.config,
    )
    assert resume_state.next_step == 3


@pytest.mark.parametrize("boundary", [1, 2])
def test_resume_rejects_boundary_at_or_before_next_step(
    tmp_path: Path,
    boundary: int,
) -> None:
    # Given: an experiment checkpointed with absolute next step two.
    create_processed_data(tmp_path / "processed")
    experiment = tiny_experiment(tmp_path, max_steps=4)
    partial = trainer.run_training(experiment, run_until_step=2)

    # When/Then: a process cannot stop before it performs another step.
    with pytest.raises(trainer.InvalidRunBoundaryError, match="2 < boundary"):
        _ = trainer.run_training(
            experiment,
            resume_path=partial.checkpoint_path,
            run_until_step=boundary,
        )


def test_sampling_frequency_does_not_change_training_trajectory(tmp_path: Path) -> None:
    # Given: two dropout-enabled runs differing only in isolated sampling frequency.
    create_processed_data(tmp_path / "processed")
    base = tiny_experiment(tmp_path, max_steps=4)
    frequent = replace(
        base,
        model=replace(base.model, dropout=0.2),
        training=replace(
            base.training,
            output_dir=tmp_path / "frequent-output",
            checkpoint_dir=tmp_path / "frequent-checkpoints",
            tensorboard_dir=tmp_path / "frequent-runs",
            sample_interval=1,
        ),
    )
    disabled = replace(
        frequent,
        training=replace(
            frequent.training,
            output_dir=tmp_path / "disabled-output",
            checkpoint_dir=tmp_path / "disabled-checkpoints",
            tensorboard_dir=tmp_path / "disabled-runs",
            sample_interval=5,
        ),
    )

    # When: both complete from the same seed.
    frequent_result = trainer.run_training(frequent)
    disabled_result = trainer.run_training(disabled)
    frequent_components = restore_components(frequent, frequent_result.checkpoint_path)
    disabled_components = restore_components(disabled, disabled_result.checkpoint_path)

    # Then: sampling volume does not perturb model or optimizer state.
    frequent_model_state = cast(
        "dict[str, torch.Tensor]",
        frequent_components.model.state_dict(),
    )
    disabled_model_state = cast(
        "dict[str, torch.Tensor]",
        disabled_components.model.state_dict(),
    )
    for name, frequent_tensor in frequent_model_state.items():
        assert torch.equal(frequent_tensor, disabled_model_state[name])
    assert_state_values_equal(
        cast("StateValue", frequent_components.optimizer.state_dict()),
        cast("StateValue", disabled_components.optimizer.state_dict()),
    )


def test_interrupted_resume_matches_uninterrupted_training_exactly(tmp_path: Path) -> None:
    # Given: one complete dropout-enabled experiment and an off-schedule split point.
    create_processed_data(tmp_path / "processed")
    base = tiny_experiment(tmp_path, max_steps=5)
    experiment = replace(
        base,
        model=replace(base.model, dropout=0.2),
        training=replace(
            base.training,
            eval_interval=2,
            checkpoint_interval=4,
            sample_interval=2,
        ),
    )
    uninterrupted = trainer.run_training(experiment)
    reference_checkpoint = tmp_path / "uninterrupted.pt"
    reference_metrics = tmp_path / "uninterrupted-metrics.jsonl"
    _ = shutil.copy2(uninterrupted.checkpoint_path, reference_checkpoint)
    _ = shutil.copy2(uninterrupted.metrics_path, reference_metrics)
    for directory in (
        experiment.training.output_dir,
        experiment.training.checkpoint_dir,
        experiment.training.tensorboard_dir,
    ):
        shutil.rmtree(directory)

    # When: the identical experiment exits at K=3, restores v2, and finishes at N=5.
    partial = trainer.run_training(experiment, run_until_step=3)
    resumed = trainer.run_training(experiment, resume_path=partial.checkpoint_path)
    expected_components = restore_components(experiment, reference_checkpoint)
    actual_components = restore_components(experiment, resumed.checkpoint_path)

    # Then: model, optimizer, next batches, and next isolated sample are bit-identical.
    expected_model_state = cast(
        "dict[str, torch.Tensor]",
        expected_components.model.state_dict(),
    )
    actual_model_state = cast(
        "dict[str, torch.Tensor]",
        actual_components.model.state_dict(),
    )
    for name, expected_tensor in expected_model_state.items():
        assert torch.equal(expected_tensor, actual_model_state[name])
    assert_state_values_equal(
        cast("StateValue", expected_components.optimizer.state_dict()),
        cast("StateValue", actual_components.optimizer.state_dict()),
    )
    expected_train_batch = expected_components.train_batcher.next_batch()
    actual_train_batch = actual_components.train_batcher.next_batch()
    expected_val_batch = expected_components.val_batcher.next_batch()
    actual_val_batch = actual_components.val_batcher.next_batch()
    assert torch.equal(expected_train_batch[0], actual_train_batch[0])
    assert torch.equal(expected_train_batch[1], actual_train_batch[1])
    assert torch.equal(expected_val_batch[0], actual_val_batch[0])
    assert torch.equal(expected_val_batch[1], actual_val_batch[1])
    prompt = torch.tensor(
        [expected_components.tokenizer.encode(experiment.training.sample_prompt)],
        dtype=torch.long,
    )
    _ = expected_components.model.eval()
    _ = actual_components.model.eval()
    expected_sample = expected_components.model.generate(
        prompt,
        max_new_tokens=4,
        temperature=0.8,
        top_k=3,
        generator=expected_components.sample_generator,
    )
    actual_sample = actual_components.model.generate(
        prompt,
        max_new_tokens=4,
        temperature=0.8,
        top_k=3,
        generator=actual_components.sample_generator,
    )
    assert torch.equal(expected_sample, actual_sample)

    # And: every deterministic metric is present exactly once with the same value.
    expected_metrics = read_metric_records(reference_metrics)
    actual_metrics = read_metric_records(resumed.metrics_path)
    assert [record["step"] for record in actual_metrics] == list(range(5))
    assert [metric_without_system_fields(record) for record in actual_metrics] == [
        metric_without_system_fields(record) for record in expected_metrics
    ]
    assert [record["learning_rate"] for record in actual_metrics] == [
        record["learning_rate"] for record in expected_metrics
    ]


def test_generate_cli_loads_checkpoint_and_appends_text(tmp_path: Path) -> None:
    # Given: a one-step checkpoint and a two-character prompt.
    create_processed_data(tmp_path / "processed")
    experiment = tiny_experiment(tmp_path, max_steps=1)
    result = trainer.run_training(experiment)

    # When: the documented generate.py surface samples two new characters.
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "generate.py",
            "--checkpoint",
            str(result.checkpoint_path),
            "--prompt",
            "ab",
            "--max-new-tokens",
            "2",
            "--temperature",
            "0.8",
            "--top-k",
            "3",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: the command succeeds and emits prompt plus two characters.
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().startswith("ab")
    assert len(completed.stdout.strip()) == 4
