import json
from math import isclose
from pathlib import Path
from typing import cast

from minigpt import config, metrics, optimization

type MetricValue = int | float | None


def test_load_experiment_config_parses_nested_yaml(tmp_path: Path) -> None:
    # Given: a complete CPU training configuration.
    config_path = tmp_path / "experiment.yaml"
    _ = config_path.write_text(
        """
runtime:
  seed: 123
  num_threads: 2
  device: cpu
data:
  directory: data/processed
  block_size: 16
  batch_size: 4
model:
  vocab_size: null
  n_layer: 2
  n_head: 2
  n_embd: 32
  dropout: 0.1
  bias: false
optimizer:
  learning_rate: 0.001
  min_learning_rate: 0.0001
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
training:
  max_steps: 10
  warmup_steps: 2
  eval_interval: 5
  eval_batches: 3
  log_interval: 1
  checkpoint_interval: 5
  sample_interval: 5
  sample_tokens: 20
  sample_prompt: "ROMEO:"
  output_dir: outputs
  checkpoint_dir: checkpoints
  tensorboard_dir: outputs/tensorboard
""".strip(),
        encoding="utf-8",
    )

    # When: YAML crosses the typed configuration boundary.
    experiment = config.load_experiment_config(config_path)

    # Then: values and paths are represented by immutable settings.
    assert experiment.runtime.seed == 123
    assert experiment.data.block_size == 16
    assert experiment.model.vocab_size is None
    assert experiment.model.n_embd == 32
    assert experiment.optimizer.beta2 == 0.95
    assert experiment.training.checkpoint_dir == Path("checkpoints")


def test_learning_rate_uses_linear_warmup_then_cosine_decay() -> None:
    # Given: a six-step schedule with two warmup steps.
    # When: every step's learning rate is calculated.
    learning_rates = [
        optimization.learning_rate_at_step(
            step,
            max_learning_rate=1.0,
            min_learning_rate=0.1,
            warmup_steps=2,
            max_steps=6,
        )
        for step in range(6)
    ]

    # Then: warmup reaches the peak and cosine decay reaches the minimum.
    assert isclose(learning_rates[0], 0.5)
    assert isclose(learning_rates[1], 1.0)
    assert isclose(learning_rates[2], 1.0)
    assert isclose(learning_rates[-1], 0.1)
    assert learning_rates[2:] == sorted(learning_rates[2:], reverse=True)


def test_append_jsonl_writes_all_required_training_metrics(tmp_path: Path) -> None:
    # Given: one complete metrics record.
    record = metrics.TrainingMetrics(
        step=3,
        train_loss=2.5,
        val_loss=2.7,
        learning_rate=3e-4,
        step_time_ms=20.0,
        tokens_per_sec=1_600.0,
        data_time_ms=1.0,
        forward_backward_time_ms=15.0,
        optimizer_time_ms=4.0,
        cpu_memory_mb=256.0,
    )
    metrics_path = tmp_path / "metrics.jsonl"

    # When: the record is appended.
    metrics.append_jsonl(metrics_path, record)

    # Then: the JSON line contains every required field.
    payload = cast(
        "dict[str, MetricValue]",
        json.loads(metrics_path.read_text(encoding="utf-8")),
    )
    assert payload == {
        "step": 3,
        "train_loss": 2.5,
        "val_loss": 2.7,
        "learning_rate": 0.0003,
        "step_time_ms": 20.0,
        "tokens_per_sec": 1600.0,
        "data_time_ms": 1.0,
        "forward_backward_time_ms": 15.0,
        "optimizer_time_ms": 4.0,
        "cpu_memory_mb": 256.0,
    }
