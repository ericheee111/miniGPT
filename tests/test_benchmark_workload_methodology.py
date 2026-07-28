"""Test that Benchmark v2 methodology defines both identity and executed work."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import minigpt.benchmark_workload_methodology as methodology_module
from minigpt.benchmark_v2_config import case_identity, load_benchmark_v2_config
from minigpt.benchmark_workload import create_training_step_workload

if TYPE_CHECKING:
    from pathlib import Path


def test_optimizer_methodology_change_affects_identity_and_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A benchmark optimizer change updates the identity and constructed AdamW settings."""
    # Given: a compact real benchmark case and its original workload identity.
    config_path = tmp_path / "benchmark.yaml"
    _ = config_path.write_text(
        """
schema_version: 2
experiment_name: methodology_test
benchmark_seed: 1
vocab_size: 17
output_root: reports
worker_timeout_seconds: 1.0
warmup_steps: 0
measurement_steps: 1
replicates: 1
torch_num_interop_threads: 1
cpu_affinity: null
max_cv_percent: 10.0
minimum_replicates: 1
regression_threshold_percent: 5.0
relevant_environment_variables: []
models:
  tiny:
    n_layer: 1
    n_head: 1
    n_embd: 8
cases:
  - name: tiny
    model_name: tiny
    torch_num_threads: 1
    block_size: 4
    batch_size: 1
profile:
  enabled: false
  case_name: tiny
  warmup_steps: 1
  active_steps: 1
""".lstrip(),
        encoding="utf-8",
    )
    config = load_benchmark_v2_config(config_path)
    before = case_identity(config, config.cases[0])

    # When: the shared methodology changes the actual AdamW learning rate.
    monkeypatch.setattr(methodology_module, "OPTIMIZER_LEARNING_RATE", 0.0125)
    workload = create_training_step_workload(config.cases[0], seed=1, vocab_size=config.vocab_size)

    # Then: both the identity and optimizer behavior bind that one methodology value.
    assert case_identity(config, config.cases[0]) != before
    assert {group["lr"] for group in workload.optimizer.param_groups} == {0.0125}


def test_unsupported_benchmark_optimizer_type_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported methodology optimizer cannot silently construct AdamW."""
    # Given: a requested optimizer with no benchmark implementation.
    monkeypatch.setattr(methodology_module, "OPTIMIZER_TYPE", "sgd")

    # When/Then: the dispatch rejects it before any benchmark measurement occurs.
    with pytest.raises(ValueError, match="unsupported benchmark optimizer type"):
        _ = methodology_module.create_benchmark_optimizer(torch.nn.Linear(1, 1))
