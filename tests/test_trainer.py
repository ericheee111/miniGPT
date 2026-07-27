import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pytest

from minigpt import trainer
from minigpt.checkpoint import IncompatibleResumeConfigError
from minigpt.data import CharTokenizer
from minigpt.settings import (
    DataSettings,
    ExperimentConfig,
    ModelSettings,
    OptimizerSettings,
    RuntimeSettings,
    TrainingSettings,
)

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

    # When: the documented train.py surface overrides max_steps.
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "train.py",
            "--config",
            str(config_path),
            "--max-steps",
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
