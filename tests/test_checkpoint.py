from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from minigpt import checkpoint
from minigpt.checkpoint import CheckpointResources
from minigpt.data import TokenBatcher
from minigpt.model import GPT, GPTConfig
from minigpt.settings import (
    DataSettings,
    ExperimentConfig,
    ModelSettings,
    OptimizerSettings,
    RuntimeSettings,
    TrainingSettings,
)

if TYPE_CHECKING:
    from pathlib import Path


def experiment_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        runtime=RuntimeSettings(seed=7, num_threads=1, device="cpu"),
        data=DataSettings(directory=tmp_path, block_size=4, batch_size=2),
        model=ModelSettings(
            vocab_size=11,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        ),
        optimizer=OptimizerSettings(
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            grad_clip=1.0,
        ),
        training=TrainingSettings(
            max_steps=4,
            warmup_steps=1,
            eval_interval=2,
            eval_batches=1,
            log_interval=1,
            checkpoint_interval=2,
            sample_interval=2,
            sample_tokens=4,
            sample_prompt="a",
            output_dir=tmp_path / "outputs",
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
        ),
    )


def test_checkpoint_restores_model_optimizer_and_step(tmp_path: Path) -> None:
    # Given: a model and optimizer saved after one update.
    gpt = GPT(GPTConfig(vocab_size=11, block_size=4, n_layer=1, n_head=1, n_embd=8))
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    train_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=1)
    val_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=2)
    resources = CheckpointResources(gpt, optimizer, train_batcher, val_batcher)
    checkpoint_path = tmp_path / "latest.pt"
    saved_parameter = next(gpt.parameters()).detach().clone()
    checkpoint.save_checkpoint(
        checkpoint_path,
        resources=resources,
        step=3,
        config=experiment_config(tmp_path),
    )
    with torch.no_grad():
        _ = next(gpt.parameters()).add_(10.0)

    # When: the checkpoint is loaded into the existing objects.
    resume_state = checkpoint.load_checkpoint(
        checkpoint_path,
        resources=resources,
    )

    # Then: parameters and the next training step are restored.
    assert torch.equal(next(gpt.parameters()), saved_parameter)
    assert resume_state.next_step == 4


def test_checkpoint_restores_random_and_batcher_states(tmp_path: Path) -> None:
    # Given: global RNGs and batchers captured in a checkpoint.
    random.seed(10)
    np.random.seed(11)  # noqa: NPY002
    _ = torch.default_generator.manual_seed(12)
    gpt = GPT(GPTConfig(vocab_size=11, block_size=4, n_layer=1, n_head=1, n_embd=8))
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    train_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=13)
    val_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=14)
    resources = CheckpointResources(gpt, optimizer, train_batcher, val_batcher)
    checkpoint_path = tmp_path / "latest.pt"
    checkpoint.save_checkpoint(
        checkpoint_path,
        resources=resources,
        step=0,
        config=experiment_config(tmp_path),
    )
    expected_python = random.random()  # noqa: S311
    expected_numpy = float(np.random.random())  # noqa: NPY002
    expected_torch = torch.rand(3)
    expected_batch = train_batcher.next_batch()

    # When: all random states are restored.
    _ = checkpoint.load_checkpoint(
        checkpoint_path,
        resources=resources,
    )

    # Then: subsequent random values and sampled batches are identical.
    assert random.random() == expected_python  # noqa: S311
    assert float(np.random.random()) == expected_numpy  # noqa: NPY002
    assert torch.equal(torch.rand(3), expected_torch)
    actual_batch = train_batcher.next_batch()
    assert torch.equal(actual_batch[0], expected_batch[0])
    assert torch.equal(actual_batch[1], expected_batch[1])
